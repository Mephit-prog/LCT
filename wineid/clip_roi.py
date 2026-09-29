"""Offline experimental CLIP Grad-CAM label proposal; NOT a validated detector.

The CAM is a weakly supervised image-text explanation, not a label mask. No server
integration or automatic acceptance is provided. Real runs require explicit consent
for checkpoint download and independently annotated query bbox.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .data import jsonl
from .roi import (AmbiguousScene, InvalidImage, _bbox_crop, decode,
                  MAX_CROP_PIXELS)

MODEL = 'ViT-B-16'
PRETRAINED = 'laion2b_s34b_b88k'
VERSION = 'clip-cam-letterbox-v1'
PROMPT = 'a photograph of the front label on a wine bottle'
NEGATIVE = 'a photograph of a bottle without a visible label'


class CLIPCAM:
    """Compute one positive-minus-negative image/text CAM on penultimate ViT block.

    The square letterbox prevents the default OpenCLIP center crop from silently
    removing parts of wide/tall photos. Only compatible OpenCLIP ViT blocks work.
    """
    def __init__(self, model=MODEL, pretrained=PRETRAINED):
        import open_clip
        import torch
        self.torch = torch
        self.model_name, self.pretrained = model, pretrained
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model)
        visual = self.model.visual
        blocks = getattr(getattr(visual, 'transformer', None), 'resblocks', ())
        if len(blocks) < 2 or not hasattr(visual, 'grid_size'):
            raise ValueError('CLIP CAM requires OpenCLIP ViT with spatial patch tokens')
        self.layer = blocks[-2]  # final ViT block can ignore patch tokens in CLS output
        self.grid = visual.grid_size
        if isinstance(self.grid, int):
            self.grid = (self.grid, self.grid)
        with torch.inference_mode():
            tokens = self.tokenizer([PROMPT, NEGATIVE])
            text = self.model.encode_text(tokens).float()
            self.text = torch.nn.functional.normalize(text, dim=-1).detach()

    def __call__(self, square):
        import torch
        activation = []
        gradient = []
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise ValueError('unsupported ViT block output')
            activation.append(output)
            output.register_hook(lambda grad: gradient.append(grad))
        handle = self.layer.register_forward_hook(hook)
        try:
            # DO NOT use inference_mode or no_grad: gradients are essential here.
            with torch.enable_grad():
                image = self.preprocess(square).unsqueeze(0)
                image.requires_grad_(True)
                features = self.model.encode_image(image).float()
                features = torch.nn.functional.normalize(features, dim=-1)
                score = (features * (self.text[0] - self.text[1])).sum()
                self.model.zero_grad(set_to_none=True)
                score.backward()
                if len(activation) != 1 or len(gradient) != 1:
                    raise ValueError('CAM hook did not fire exactly once')
                acts, grads = activation[0].detach(), gradient[0].detach()
                n = self.grid[0] * self.grid[1]
                # OpenCLIP ViT versions use either [tokens,batch,channels] or
                # [batch,tokens,channels]. Never guess when sizes are unexpected.
                if acts.shape[0:2] == (n+1, 1):
                    patches, changes = acts[1:, 0], grads[1:, 0]
                elif acts.shape[0:2] == (1, n+1):
                    patches, changes = acts[0, 1:], grads[0, 1:]
                else:
                    raise ValueError('unexpected spatial token shape')
                heat = torch.relu((patches * changes.mean(dim=0)).sum(dim=-1))
                return heat.reshape(self.grid).cpu().numpy().copy()
        finally:
            handle.remove()
            self.model.zero_grad(set_to_none=True)


def cam_bbox(heat, image_size, *, threshold=0.4, canvas_size=None):
    """Map CAM from square-letterbox coordinates to oriented original half-open bbox.

    Refuse disconnected substantial activations or missing evidence; CAM magnitude
    is not a calibrated confidence. Geometry is approximate at patch resolution.
    """
    import cv2
    if not 0 < threshold < 1:
        raise ValueError('invalid threshold')
    heat = np.asarray(heat, dtype='float32')
    if heat.ndim != 2 or min(heat.shape) < 2 or not np.isfinite(heat).all():
        raise InvalidImage('invalid_cam')
    top = float(heat.max())
    if top <= 1e-10 or float(np.ptp(heat)) <= 1e-10:
        raise InvalidImage('empty_cam')
    mask = (heat >= top * threshold).astype('uint8')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    significant = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 2]
    if not significant:
        raise InvalidImage('no_label_bbox')
    if len(significant) > 1:
        raise AmbiguousScene('multiple_cam_regions')
    x, y, cw, ch, _ = stats[significant[0]]
    # Include adjacent patch to avoid cutting off text on a coarse ViT grid.
    margin = 1
    x0, y0 = max(0, x-margin), max(0, y-margin)
    x1, y1 = min(heat.shape[1], x+cw+margin), min(heat.shape[0], y+ch+margin)
    w, h = image_size
    side = max(w, h)
    canvas = side if canvas_size is None else canvas_size
    if not 1 <= canvas <= side:
        raise ValueError('invalid canvas_size')
    rw, rh = max(1, round(w*canvas/side)), max(1, round(h*canvas/side))
    ox, oy = (canvas-rw)//2, (canvas-rh)//2
    left = max(0, int(np.floor((x0/heat.shape[1]*canvas - ox)*w/rw)))
    upper = max(0, int(np.floor((y0/heat.shape[0]*canvas - oy)*h/rh)))
    right = min(w, int(np.ceil((x1/heat.shape[1]*canvas - ox)*w/rw)))
    lower = min(h, int(np.ceil((y1/heat.shape[0]*canvas - oy)*h/rh)))
    if right <= left or lower <= upper:
        raise InvalidImage('cam_outside_image')
    if (right-left)*(lower-upper) > MAX_CROP_PIXELS:
        raise InvalidImage('crop_too_large')
    return left, upper, right, lower


def clip_cam_roi(data, deadline, cam, *, threshold=0.4):
    """Injectable CAM callable enables tests without weights or model downloads."""
    img, _ = decode(data, deadline)
    side = max(img.size)
    canvas = min(side, 1024)
    scale = canvas / side
    resized = img.resize((max(1, round(img.width*scale)), max(1, round(img.height*scale))))
    square = Image.new('RGB', (canvas, canvas), 'white')
    square.paste(resized, ((canvas-resized.width)//2, (canvas-resized.height)//2))
    if time.monotonic() >= deadline:
        raise TimeoutError('deadline')
    heat = cam(square)
    if time.monotonic() >= deadline:
        raise TimeoutError('deadline')
    bbox = cam_bbox(heat, img.size, threshold=threshold, canvas_size=canvas)
    return _bbox_crop(img, data, bbox, deadline, 0.05, 'clip-cam-0',
                      'fallback_bbox', VERSION)


def overlap(predicted, gt):
    x0, y0 = max(predicted[0], gt[0]), max(predicted[1], gt[1])
    x1, y1 = min(predicted[2], gt[2]), min(predicted[3], gt[3])
    inter = max(0, x1-x0)*max(0, y1-y0)
    area = lambda box: (box[2]-box[0])*(box[3]-box[1])
    return inter / (area(predicted)+area(gt)-inter), inter / area(gt)


def experiment(manifest, cam, *, split='validation', threshold=0.4, seconds=8):
    """Paired localization-only results, including refusals in denominators."""
    path = Path(manifest).resolve()
    if 'eval' in path.parts:
        raise ValueError('eval/ must not be used for tuning')
    if split not in ('validation', 'test') or not 0 < threshold < 1 or not 0 < seconds <= 300:
        raise ValueError('invalid experiment parameters')
    records = jsonl(path)
    groups = {}
    ids_all, hashes = set(), {}
    for row in records:
        group = row.get('source_group')
        qid = row.get('query_id')
        if not group or row.get('split') not in ('train', 'validation', 'test') or not qid or qid in ids_all:
            raise ValueError('missing source_group/split or duplicate query ID')
        ids_all.add(qid)
        photo = (path.parent / row['image_path']).resolve()
        if not photo.is_relative_to(path.parent):
            raise ValueError(f'unsafe path: {qid}')
        digest = hashlib.sha256(photo.read_bytes()).hexdigest()
        if digest != row.get('image_sha256'):
            raise ValueError(f'image hash mismatch: {qid}')
        if digest in hashes and hashes[digest] != row['split']:
            raise ValueError(f'image hash across splits: {qid}')
        hashes[digest] = row['split']
        if group in groups and groups[group] != row['split']:
            raise ValueError(f'source_group across splits: {group}')
        groups[group] = row['split']
    selected = [r for r in records if r['split'] == split]
    if not selected:
        raise ValueError('empty split')
    ids = set()
    results = []
    for row in selected:
        start = time.monotonic()
        qid = row['query_id']
        if (qid in ids or row.get('gt_status') not in ('known', 'unknown', 'ambiguous')
                or (row['gt_status'] == 'known' and (not row.get('gt_slug') or not row.get('gt_source')))
                or (row['gt_status'] != 'known' and row.get('gt_slug') is not None)):
            raise ValueError(f'duplicate id or incomplete provenance: {qid}')
        ids.add(qid)
        photo = (path.parent / row['image_path']).resolve()
        if not photo.is_relative_to(path.parent):
            raise ValueError(f'unsafe path: {qid}')
        data = photo.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != row.get('image_sha256'):
            raise ValueError(f'image hash mismatch: {qid}')
        try:
            img, _ = decode(data, start+seconds)
        except (InvalidImage, TimeoutError) as exc:
            raise ValueError(f'invalid image: {qid}: {exc}') from exc
        gt = row.get('bbox')
        present = row.get('label_present', True)
        if type(present) is not bool or (present and (
                not isinstance(gt, list) or len(gt) != 4 or any(type(x) is not int for x in gt) or not (
                0 <= gt[0] < gt[2] <= img.width and 0 <= gt[1] < gt[3] <= img.height))) or (
                not present and gt is not None):
            raise ValueError(f'invalid manual bbox/label_present: {qid}')
        record = {'query_id': qid, 'split': split, 'source_group': row['source_group'],
                  'image_sha256': digest, 'label_present': present,
                  'gt_bbox': gt, 'bbox': None, 'status': 'unusable',
                  'reason': None, 'iou': None, 'gt_coverage': None, 'crop_version': VERSION}
        try:
            roi = clip_cam_roi(data, start+seconds, cam, threshold=threshold)
            record['bbox'] = list(roi.bbox)
            record['status'] = roi.status
            if present:
                record['iou'], record['gt_coverage'] = overlap(roi.bbox, gt)
        except (InvalidImage, AmbiguousScene, TimeoutError) as exc:
            record['reason'] = str(exc)
        record['latency_ms'] = round((time.monotonic()-start)*1000)
        results.append(record)
    return results


def summary(rows):
    n = len(rows)
    good = [r for r in rows if r['bbox'] is not None]
    negatives = [r for r in rows if not r.get('label_present', True)]
    return {'total': n, 'located': len(good), 'located_fraction': len(good)/n,
            'negative_total': len(negatives),
            'false_localizations': sum(r['bbox'] is not None for r in negatives),
            'iou_ge_0_5': sum(r['iou'] is not None and r['iou'] >= 0.5 for r in good),
            'gt_coverage_ge_0_9': sum(r['gt_coverage'] is not None and r['gt_coverage'] >= 0.9 for r in good),
            'latency_ms_p50': float(np.percentile([r['latency_ms'] for r in rows], 50)),
            'latency_ms_p95': float(np.percentile([r['latency_ms'] for r in rows], 95)),
            'note': 'Localization geometry only; no OCR/readability/production claims.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', help='independent annotated query JSONL; never eval/')
    parser.add_argument('output', help='new JSONL; not inside eval/')
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--seconds', type=float, default=30, help='per-image offline deadline')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--pretrained', default=PRETRAINED)
    parser.add_argument('--allow-checkpoint-download', action='store_true',
                        help='explicitly approve external model download and license review')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if out.exists() or out == Path(args.manifest).resolve() or 'eval' in out.parts:
        parser.error('output must be new, distinct from input and outside eval/')
    if not args.allow_checkpoint_download:
        parser.error('review checkpoint/license first; pass --allow-checkpoint-download to run')
    cam = CLIPCAM(args.model, args.pretrained)
    if not 0 < args.seconds <= 300:
        parser.error('--seconds must be in (0, 300]')
    rows = experiment(args.manifest, cam, split=args.split, threshold=args.threshold,
                      seconds=args.seconds)
    with out.open('x', encoding='utf-8') as dest:
        for row in rows:
            dest.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    run = {'model': args.model, 'pretrained': args.pretrained, 'prompt': PROMPT,
           'negative': NEGATIVE, 'threshold': args.threshold, 'seconds': args.seconds,
           'split': args.split, 'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
           'results': summary(rows)}
    metadata = out.with_name(out.name + '.run.json')
    with metadata.open('x', encoding='utf-8') as dest:
        json.dump(run, dest, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(run, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
