"""ROI retention statistics (plans/jina_clip_roi_catalog_enrichment.md §2).

Pure geometry: map a model field of view (FOV) into original **oriented**
coordinates and measure how much of the ground-truth label / text fields it
retains. The key diagnostic is ``label_retention`` (fraction of the GT label area
inside the FOV), because the native Jina processor resizes the shortest side to
512 and then center-crops, which can drop most of a tall product shot.

These numbers require GT annotations; production has none. They are an
evaluation/ablation instrument, never a runtime confidence and never a claim of
OCR/recognition quality.
"""
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

NATIVE_MODEL_SIZE = 512


def _valid_box(box):
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(type(x) is int for x in box)
            and 0 <= box[0] < box[2] and 0 <= box[1] < box[3])


def area(box) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection(a, b) -> int:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def iou(a, b) -> float:
    union = area(a) + area(b) - intersection(a, b)
    return intersection(a, b) / union if union else 0.0


def retention(gt, box) -> float:
    """Fraction of the GT box area retained inside ``box`` (0..1)."""
    return intersection(gt, box) / area(gt) if area(gt) else 0.0


def native_center_crop_fov(width: int, height: int,
                           model_size: int = NATIVE_MODEL_SIZE) -> tuple[int, int, int, int]:
    """Jina ``shortest`` resize + center crop mapped back to original coordinates.

    Returns half-open ``(x0, y0, x1, y1)`` in the original oriented image. The
    visible original area is a square of side ``model_size / scale`` centered in
    the image, where ``scale = model_size / min(width, height)``.
    """
    if width < 1 or height < 1 or model_size < 1:
        raise ValueError('positive width, height and model_size required')
    scale = model_size / min(width, height)
    side = model_size / scale  # == min(width, height); the visible square side
    x0 = math.floor((width - side) / 2)
    y0 = math.floor((height - side) / 2)
    return (x0, y0, min(width, math.ceil(x0 + side)), min(height, math.ceil(y0 + side)))


def frame_fov(width: int, height: int) -> tuple[int, int, int, int]:
    """Aspect-preserving pad / manual full-frame FOV keeps the whole image."""
    return (0, 0, width, height)


def box_stats(gt, frame_size) -> dict:
    """Area fraction, aspect ratio and normalized centroid of a label box."""
    width, height = frame_size
    x0, y0, x1, y1 = gt
    return {'label_area_fraction': area(gt) / (width * height) if width and height else None,
            'label_width_px': x1 - x0, 'label_height_px': y1 - y0,
            'label_aspect': (x1 - x0) / (y1 - y0) if y1 > y0 else None,
            'centroid': [round(((x0 + x1) / 2) / width, 4), round(((y0 + y1) / 2) / height, 4)]}


def field_retentions(field_boxes: dict, fov) -> dict:
    return {name: round(retention(box, fov), 4) for name, box in field_boxes.items()}


@dataclass(frozen=True)
class RoiRow:
    query_id: str
    image_size: tuple[int, int]
    label_bbox: tuple[int, int, int, int]
    field_bboxes: dict
    source_group: str | None = None


def _parse_row(row) -> RoiRow:
    size = row.get('image_size')
    label = row.get('label_bbox')
    if (not isinstance(size, list) or len(size) != 2
            or not all(type(x) is int and x > 0 for x in size)):
        raise ValueError(f'invalid image_size: {row.get("query_id")}')
    if not _valid_box(label):
        raise ValueError(f'invalid label_bbox: {row.get("query_id")}')
    fields = row.get('field_bboxes') or {}
    if not isinstance(fields, dict) or any(not _valid_box(v) for v in fields.values()):
        raise ValueError(f'invalid field_bboxes: {row.get("query_id")}')
    return RoiRow(str(row.get('query_id')), tuple(size), tuple(label),
                  {k: tuple(v) for k, v in fields.items()}, row.get('source_group'))


def retention_report(rows, *, mode: str = 'native',
                     model_size: int = NATIVE_MODEL_SIZE) -> dict:
    """Per-query retention + aggregates with explicit denominators.

    ``mode='native'`` uses the Jina shortest+center-crop FOV, ``mode='pad'`` the
    full frame, ``mode='bbox'`` the row-provided ``model_bbox``. Never reports a
    recognition metric.
    """
    if mode not in ('native', 'pad', 'bbox'):
        raise ValueError("mode must be 'native', 'pad' or 'bbox'")
    parsed = [_parse_row(row) for row in rows]
    if mode == 'bbox':
        for row, raw in zip(parsed, rows):
            if not _valid_box(raw.get('model_bbox')):
                raise ValueError(f'model_bbox required for mode=bbox: {row.query_id}')
    results = []
    for row, raw in zip(parsed, rows):
        width, height = row.image_size
        if mode == 'native':
            fov = native_center_crop_fov(width, height, model_size)
        elif mode == 'pad':
            fov = frame_fov(width, height)
        else:
            fov = tuple(raw['model_bbox'])
        results.append({'query_id': row.query_id, 'source_group': row.source_group,
                        'fov': list(fov),
                        'label_retention': round(retention(row.label_bbox, fov), 4),
                        'field_retention': field_retentions(row.field_bboxes, fov),
                        **box_stats(row.label_bbox, (width, height))})
    values = [r['label_retention'] for r in results]
    fields = sorted({name for r in results for name in r['field_retention']})
    field_means = {name: _mean([r['field_retention'][name] for r in results
                                if name in r['field_retention']]) for name in fields}
    return {'mode': mode, 'queries': len(results),
            'source_groups': len({r['source_group'] for r in results if r['source_group']}),
            'label_retention': {'min': min(values, default=None),
                                'mean': _mean(values), 'n': len(values)},
            'field_retention_mean': field_means,
            'label_area_fraction_mean': _mean([r['label_area_fraction'] for r in results]),
            'retained_below_0_9': {'numerator': sum(v < 0.9 for v in values),
                                   'denominator': len(values)},
            'per_query': results,
            'note': 'Requires GT annotations; not a recognition or OCR quality metric.'}


def _mean(values):
    return round(sum(values) / len(values), 4) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotations', help='JSONL: query_id, image_size, label_bbox, field_bboxes, model_bbox (mode=bbox)')
    parser.add_argument('--mode', choices=('native', 'pad', 'bbox'), default='native')
    parser.add_argument('--model-size', type=int, default=NATIVE_MODEL_SIZE)
    args = parser.parse_args()
    path = Path(args.annotations)
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    print(json.dumps(retention_report(rows, mode=args.mode, model_size=args.model_size),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
