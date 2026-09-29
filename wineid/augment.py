"""Synthetic "field" views of studio packshots (no external data).

Used only by the explicit ``--synthetic-views`` training mode: the catalog has
one studio frame per wine, so real field frames required by ТЗ §4.2 are
absent. These views imitate shelf photos (perspective, tilt, partial crop,
warm/dim light, glare, blur, sensor noise, JPEG) but are NOT field photos:
metrics measured on them say nothing about real user photos.
"""
import io

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

MAX_SIDE = 768


def _perspective_coeffs(src, dst):
    """Coefficients for PIL PERSPECTIVE mapping output points dst -> input src."""
    matrix, rhs = [], []
    for (x, y), (u, v) in zip(dst, src):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.extend([u, v])
    return np.linalg.solve(np.asarray(matrix, dtype='float64'),
                           np.asarray(rhs, dtype='float64')).tolist()


def _background(rng, size):
    """Smooth random colour gradient with noise: shelf/interior stand-in."""
    w, h = size
    top = rng.uniform(20, 235, 3)
    bottom = rng.uniform(20, 235, 3)
    t = np.linspace(0, 1, h)[:, None, None]
    grad = top * (1 - t) + bottom * t
    grad = np.broadcast_to(grad, (h, w, 3)) + rng.normal(0, 8, (h, w, 3))
    return Image.fromarray(np.clip(grad, 0, 255).astype('uint8'))


def _backdrop_mask(arr):
    """Near-white pixels connected to the image border (the studio backdrop).

    White paper labels inside the bottle outline are not border-connected, so
    they are kept intact.
    """
    white = Image.fromarray(np.where(arr.min(axis=2) > 235, 255, 0).astype('uint8'))
    w, h = white.size
    for x in (0, w // 2, w - 1):
        for y in (0, h - 1):
            if white.getpixel((x, y)) == 255:
                ImageDraw.floodfill(white, (x, y), 128)
    for y in (h // 2,):
        for x in (0, w - 1):
            if white.getpixel((x, y)) == 255:
                ImageDraw.floodfill(white, (x, y), 128)
    return np.asarray(white) == 128


def field_view(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Return a randomly degraded RGB copy of ``image`` (input is not modified)."""
    img = image.convert('RGB')
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    w, h = img.size

    # Studio packshots sit on near-white; swap it for a random background so the
    # model cannot rely on the clean backdrop.
    arr = np.asarray(img).astype('int16')
    mask = _backdrop_mask(arr)
    if rng.random() < 0.8 and mask.mean() > 0.2:
        bg = np.asarray(_background(rng, (w, h)))
        arr = np.where(mask[..., None], bg, arr)
        img = Image.fromarray(arr.astype('uint8'))
    fill = tuple(int(c) for c in rng.uniform(20, 200, 3))

    # Camera tilt and perspective (shooting at an angle to the shelf).
    img = img.rotate(float(rng.uniform(-12, 12)), resample=Image.BICUBIC,
                     expand=True, fillcolor=fill)
    w, h = img.size
    d = 0.12
    dst = [(0, 0), (w, 0), (w, h), (0, h)]
    src = [(x + rng.uniform(-d, d) * w, y + rng.uniform(-d, d) * h) for x, y in dst]
    img = img.transform((w, h), Image.PERSPECTIVE, _perspective_coeffs(src, dst),
                        resample=Image.BICUBIC, fillcolor=fill)

    # Partial framing, biased towards the lower-middle body where the front
    # label sits, so the label stays (mostly) in frame.
    scale = rng.uniform(0.7, 1.0)
    cw, ch = int(w * scale), int(h * scale)
    x0 = int(rng.uniform(0.25, 0.75) * (w - cw))
    y0 = int(rng.uniform(0.5, 1.0) * (h - ch))
    img = img.crop((x0, y0, x0 + cw, y0 + ch))

    # Store lighting: brightness/contrast/saturation and warm/cold cast.
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.55, 1.25))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.6, 1.3))
    arr = np.asarray(img).astype('float32')
    arr *= rng.uniform(0.85, 1.15, 3)

    # Specular glare: soft bright ellipse on the glass.
    if rng.random() < 0.6:
        ch, cw = arr.shape[:2]
        yy, xx = np.mgrid[0:ch, 0:cw]
        cx, cy = rng.uniform(0.2, 0.8) * cw, rng.uniform(0.1, 0.9) * ch
        rx, ry = rng.uniform(0.04, 0.15) * cw, rng.uniform(0.1, 0.35) * ch
        blob = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
        arr += blob[..., None] * rng.uniform(80, 200)

    arr += rng.normal(0, rng.uniform(0, 8), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype('uint8'))

    # Optics: defocus or motion blur, low-resolution capture.
    r = rng.random()
    if r < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.0)))
    elif r < 0.5:
        img = img.filter(ImageFilter.BoxBlur(rng.uniform(1.0, 2.5)))
    if rng.random() < 0.4:
        factor = rng.uniform(0.35, 0.7)
        small = (max(32, int(img.width * factor)), max(32, int(img.height * factor)))
        img = img.resize(small, Image.BILINEAR).resize(img.size, Image.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=int(rng.integers(35, 90)))
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert('RGB')
