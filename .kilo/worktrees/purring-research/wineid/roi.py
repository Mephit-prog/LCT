"""Orient source before indexing bbox; do not treat a full-frame photo as a found label."""
import hashlib
import io
import time
from dataclasses import dataclass
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 24_000_000
MAX_CROP_PIXELS = 6_000_000
MAX_CROP_BYTES = 12 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
ALLOWED = {'JPEG': 'image/jpeg', 'PNG': 'image/png', 'WEBP': 'image/webp'}


class InvalidImage(ValueError):
    pass


class AmbiguousScene(ValueError):
    pass


@dataclass(frozen=True)
class ROI:
    roi_id: str
    source_sha256: str
    source_size: tuple[int, int]
    bbox: tuple[int, int, int, int]
    status: str
    mime: str
    crop: bytes
    crop_version: str = 'oriented-rgb-bbox-v1'


def decode(data: bytes, deadline: float):
    if not data or len(data) > MAX_BYTES:
        raise InvalidImage('invalid_size')
    if time.monotonic() >= deadline:
        raise TimeoutError('deadline')
    try:
        with Image.open(io.BytesIO(data)) as src:
            if src.format not in ALLOWED:
                raise InvalidImage('unsupported_format')
            mime = ALLOWED[src.format]
            if src.width * src.height > MAX_PIXELS:
                raise InvalidImage('too_many_pixels')
            img = ImageOps.exif_transpose(src)
            img.load()
            if time.monotonic() >= deadline:
                raise TimeoutError('deadline')
            # Composite transparency onto white instead of losing glyphs on dark backgrounds.
            if img.mode in ('RGBA', 'LA') or 'transparency' in img.info:
                img = img.convert('RGBA')
                background = Image.new('RGBA', img.size, 'white')
                img = Image.alpha_composite(background, img).convert('RGB')
            else:
                img = img.convert('RGB')
            return img, mime
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImage('decode_error') from exc


def full_frame_roi(data: bytes, deadline: float) -> ROI:
    """Explicit experimental full-frame ablation, not a detected label."""
    img, _ = decode(data, deadline)
    if img.width * img.height > MAX_CROP_PIXELS:
        raise InvalidImage('crop_too_large')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    if buf.tell() > MAX_CROP_BYTES:
        raise InvalidImage('crop_too_large')
    if time.monotonic() >= deadline:
        raise TimeoutError('deadline')
    return ROI('full-frame-0', hashlib.sha256(data).hexdigest(), img.size,
               (0, 0, *img.size), 'fallback_bbox', 'image/png', buf.getvalue(),
               'oriented-rgb-full-frame-v1')


def _bbox_crop(img, data, bbox, deadline, padding, roi_id, status, version):
    if not 0 <= padding <= 0.2:
        raise ValueError('invalid padding')
    if len(bbox) != 4 or any(type(x) is not int for x in bbox):
        raise InvalidImage('invalid_bbox')
    x0, y0, x1, y1 = bbox
    w, h = img.size
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise InvalidImage('invalid_bbox')
    px, py = round((x1-x0)*padding), round((y1-y0)*padding)
    padded = max(0, x0-px), max(0, y0-py), min(w, x1+px), min(h, y1+py)
    if (padded[2]-padded[0]) * (padded[3]-padded[1]) > MAX_CROP_PIXELS:
        raise InvalidImage('crop_too_large')
    buf = io.BytesIO()
    img.crop(padded).save(buf, format='PNG')  # strips metadata; format from content
    if buf.tell() > MAX_CROP_BYTES:
        raise InvalidImage('crop_too_large')
    if time.monotonic() >= deadline:
        raise TimeoutError('deadline')
    return ROI(roi_id, hashlib.sha256(data).hexdigest(), img.size, padded,
               status, 'image/png', buf.getvalue(), version)


def manual_roi(data: bytes, bbox: tuple[int, int, int, int], deadline: float,
               padding: float = 0.03, roi_id: str = 'manual-0') -> ROI:
    img, _ = decode(data, deadline)
    return _bbox_crop(img, data, bbox, deadline, padding, roi_id,
                      'manual', 'oriented-rgb-bbox-v1')


def center_crop_80_roi(data: bytes, deadline: float) -> ROI:
    """ТЗ §4.2: central 80% crop fallback when no trained label detector is available."""
    img, _ = decode(data, deadline)
    w, h = img.size
    x0, y0 = round(w * 0.10), round(h * 0.10)
    x1, y1 = max(x0 + 1, round(w * 0.90)), max(y0 + 1, round(h * 0.90))
    return _bbox_crop(img, data, (x0, y0, x1, y1), deadline, 0.0,
                      'center-80-0', 'fallback_bbox', 'oriented-rgb-center-80-v1')


def auto_bbox_roi(data: bytes, deadline: float) -> ROI:
    """Conservative high-contrast rectangle proposal, NOT learned segmentation.

    Experimental baseline for validation against manual bbox. Multiple separate
    rectangles are never merged into a single OCR input. No proposal => refusal.
    """
    import cv2
    import numpy as np
    img, _ = decode(data, deadline)
    w, h = img.size
    scale = min(1., 640/max(w, h))
    small = img.resize((max(1, round(w*scale)), max(1, round(h*scale))))
    if time.monotonic() >= deadline:
        raise TimeoutError('deadline')
    gray = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 135)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sw, sh = small.size
    proposals = []
    for contour in contours:
        if time.monotonic() >= deadline:
            raise TimeoutError('deadline')
        x, y, bw, bh = cv2.boundingRect(contour)
        fraction = bw * bh / (sw * sh)
        if (not .012 <= fraction <= .5 or not .65 <= bw/bh <= 3.5
                or x <= 1 or y <= 1 or x+bw >= sw-1 or y+bh >= sh-1):
            continue
        fill = cv2.contourArea(contour) / (bw*bh)
        if fill < .72:
            continue
        # Strongly prefer a closed rectangular boundary; ignore text contours.
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, .035*perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        box = (x, y, x+bw, y+bh)
        if any(_iou(box, existing) > .65 for existing in proposals):
            continue
        proposals.append(box)
    if not proposals:
        raise InvalidImage('no_label_bbox')
    # Two spatially separate plausible labels may mean two bottles or front/back.
    # Neither one has a justified priority without an external bottle selector.
    if len(proposals) > 1:
        raise AmbiguousScene('multiple_label_bboxes')
    x0, y0, x1, y1 = proposals[0]
    box = (max(0, int(x0/scale)), max(0, int(y0/scale)),
           min(w, int(np.ceil(x1/scale))), min(h, int(np.ceil(y1/scale))))
    return _bbox_crop(img, data, box, deadline, .05, 'auto-bbox-0',
                      'fallback_bbox', 'oriented-rgb-auto-bbox-v1')


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1-x0)*max(0, y1-y0)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return intersection/(area_a+area_b-intersection)

