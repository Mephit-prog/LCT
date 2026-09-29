"""Decisions are abstentions unless a validation-derived policy is explicitly configured."""
import time
from threading import BoundedSemaphore
from dataclasses import asdict, dataclass
from .roi import (manual_roi, full_frame_roi, auto_bbox_roi,
                  center_crop_80_roi, decode, InvalidImage, AmbiguousScene)
from .search import BLEND_VERSION
from .text_policy import TextPolicy


@dataclass(frozen=True)
class Policy:
    min_score: float
    min_margin: float
    max_distance: float
    version: str

    def __post_init__(self):
        if not (0 <= self.min_score <= 100 and 0 <= self.min_margin <= 100
                and 0 <= self.max_distance <= 1 and self.version):
            raise ValueError('invalid policy')


def decide(candidates, policy: 'Policy | TextPolicy | None', text: str):
    if not candidates:
        return 'unknown', None, ['no_candidates']
    if policy is None:
        return 'ambiguous', None, ['policy_not_calibrated']
    top = candidates[0]
    if top.score < policy.min_score or top.normalized_distance > policy.max_distance:
        return 'unknown', None, ['insufficient_text_match']
    if len(candidates) > 1 and top.score - candidates[1].score < policy.min_margin:
        return 'ambiguous', None, ['close_candidates']
    # Score measures edit similarity only; a short shared brand alone is not
    # evidence for a unique SKU when the actual label might omit its vintage.
    return 'accepted', top.slug, ['text_only_policy']


class Pipeline:
    def __init__(self, index, ocr=None, policy=None, csv_sha256=None, roi_mode='refuse',
                 max_concurrent_ocr=2):
        if roi_mode not in ('refuse', 'full_frame_experimental', 'auto_bbox_experimental', 'center_80_crop'):
            raise ValueError('invalid roi_mode')
        if type(max_concurrent_ocr) is not int or max_concurrent_ocr < 1:
            raise ValueError('max_concurrent_ocr must be a positive integer')
        self.index, self.ocr, self.policy, self.csv_sha256 = index, ocr, policy, csv_sha256
        self.roi_mode = roi_mode
        self.ocr_slots = BoundedSemaphore(max_concurrent_ocr)
        # Text-branch configuration a calibrated policy is bound to (B05).
        self.text_config = {'csv_sha256': csv_sha256, 'search_backend': index.backend,
                            'blend_version': BLEND_VERSION}
        if isinstance(policy, TextPolicy):
            policy.check_config(self.text_config)

    def predict(self, image: bytes, *, bbox=None, manual_text=None, seconds=8.0,
                roi_mode: str | None = None):
        start = time.monotonic()
        deadline = start + seconds
        active_roi_mode = roi_mode or self.roi_mode
        if active_roi_mode not in ('refuse', 'full_frame_experimental',
                                   'auto_bbox_experimental', 'center_80_crop'):
            raise ValueError('invalid roi_mode')
        result = {'status': 'error', 'slug': None, 'reason_codes': [],
                  'versions': {'csv_sha256': self.csv_sha256,
                               'search_backend': self.index.backend,
                               'policy': self.policy.version if self.policy else None,
                               'crop': 'oriented-rgb-bbox-v1' if bbox is not None else active_roi_mode},
                  'timings_ms': {}, 'candidates': [], 'ocr': None, 'roi': None}
        def finish(status, reason, slug=None):
            result.update(status=status, slug=slug, reason_codes=reason)
            result['timings_ms']['total'] = round((time.monotonic()-start)*1000)
            return result
        try:
            if bbox is None:
                if active_roi_mode == 'full_frame_experimental':
                    roi = full_frame_roi(image, deadline)
                elif active_roi_mode == 'auto_bbox_experimental':
                    roi = auto_bbox_roi(image, deadline)
                elif active_roi_mode == 'center_80_crop':
                    roi = center_crop_80_roi(image, deadline)
                else:
                    # Never imply that a central crop is a detected label.
                    decode(image, deadline)
                    return finish('unreadable', ['automatic_roi_unavailable'])
            else:
                roi = manual_roi(image, bbox, deadline)
            result['roi'] = {'roi_id': roi.roi_id, 'source_sha256': roi.source_sha256,
                             'source_size': roi.source_size, 'bbox': roi.bbox,
                             'status': roi.status, 'crop_version': roi.crop_version}
            if manual_text is not None:
                from .ocr import MockOCR
                ocr = MockOCR(manual_text)
            else:
                ocr = self.ocr
            if ocr is None:
                return finish('error', ['ocr_not_configured'])
            # Do not queue external OCR calls past the shared request deadline.
            if not self.ocr_slots.acquire(blocking=False):
                return finish('error', ['ocr_capacity_exceeded'])
            try:
                if time.monotonic() >= deadline:
                    return finish('error', ['deadline'])
                observed = ocr.recognize(roi, deadline)
            finally:
                self.ocr_slots.release()
            result['ocr'] = asdict(observed)
            if observed.status == 'no_text':
                return finish('unreadable', ['no_text'])
            if observed.status != 'ok':
                return finish('error', ['ocr_' + observed.status, observed.error_code or 'provider_error'])
            if time.monotonic() >= deadline:
                return finish('error', ['deadline'])
            candidates = self.index.search(observed.raw_text)
            result['candidates'] = [asdict(c) for c in candidates]
            if time.monotonic() >= deadline:
                return finish('error', ['deadline'])
            status, slug, reasons = decide(candidates, self.policy, observed.raw_text)
            if bbox is None:
                reasons.append(active_roi_mode)
            return finish(status, reasons, slug)
        except AmbiguousScene as exc:
            return finish('ambiguous_scene', [str(exc)])
        except InvalidImage as exc:
            return finish('unreadable', [str(exc)])
        except TimeoutError:
            return finish('error', ['deadline'])
