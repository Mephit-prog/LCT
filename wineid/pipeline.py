"""Decisions are abstentions unless a validation-derived policy is explicitly configured."""
import time
from threading import BoundedSemaphore
from dataclasses import asdict, dataclass
from .roi import (manual_roi, full_frame_roi, auto_bbox_roi,
                  center_crop_80_roi, decode, InvalidImage, AmbiguousScene)
from .search import BLEND_VERSION
from .text_policy import TextPolicy
from .blend import Blender, CandidateEvidence, VERSION as FUSION_VERSION, clean_ocr
from .fusion_policy import FusionPolicy
from .blend_decision import rank_and_decide
from .profile import decision_code_sha256, dependency_profile, ocr_profile
from .text_policy import fingerprint

CONDITIONAL_PHOTO_SCORE = .35
CONDITIONAL_PHOTO_GAP = .05


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
                 max_concurrent_ocr=2, vision=None, blender=None, conditional_ocr=False,
                 request_timeout=8.5):
        if roi_mode not in ('refuse', 'full_frame_experimental', 'auto_bbox_experimental', 'center_80_crop'):
            raise ValueError('invalid roi_mode')
        if type(max_concurrent_ocr) is not int or max_concurrent_ocr < 1:
            raise ValueError('max_concurrent_ocr must be a positive integer')
        if (not isinstance(request_timeout, (int, float)) or isinstance(request_timeout, bool)
                or not 0 < request_timeout <= 600):
            raise ValueError('invalid request_timeout')
        self.request_timeout = float(request_timeout)
        self.max_concurrent_ocr = max_concurrent_ocr
        self.index, self.ocr, self.policy, self.csv_sha256 = index, ocr, policy, csv_sha256
        self.roi_mode = roi_mode
        self.vision = vision
        self.blender = blender if blender is not None else (Blender(index.wines, aliases=index.aliases)
                                                       if vision is not None else None)
        self.conditional_ocr = conditional_ocr
        self.ocr_slots = BoundedSemaphore(max_concurrent_ocr)
        # Heavy integrity/profile checks happen at bootstrap, not in health.
        # Recognition still recomputes the live profile before policy acceptance.
        self.text_config = self.current_text_config
        self._health_fusion_config = self.fusion_config if isinstance(policy, FusionPolicy) else None
        if isinstance(policy, TextPolicy):
            if self.blender is not None:
                raise ValueError('text policy cannot accept blended rankings')
            policy.check_config(self.text_config)
        if self.blender is not None and policy is not None and not isinstance(policy, FusionPolicy):
            raise ValueError('blend requires a versioned FusionPolicy')
        if isinstance(policy, FusionPolicy):
            if self.blender is None:
                raise ValueError('fusion policy requires blender')
            policy.check_config(self._health_fusion_config)

    @property
    def ocr_ready(self):
        return self.ocr is not None and bool(getattr(self.ocr, 'ready', False))

    @property
    def recognition_ready(self):
        catalog_ready = bool(self.index.wines) and self.csv_sha256 == self.index.wines[0].csv_sha256
        return catalog_ready and ((self.vision is not None and
                                   bool(getattr(self.vision, 'ready', True))) or
                                  (self.ocr_ready and self.roi_mode != 'refuse'))

    @property
    def acceptance_ready(self):
        if not self.recognition_ready:
            return False
        try:
            if isinstance(self.policy, FusionPolicy):
                return (self.blender is not None and
                        (self.vision is None or bool(getattr(self.vision, 'ready', True))) and
                        (self.ocr is None or self.ocr_ready) and
                        self.policy.artifact_sha256 == fingerprint(self.policy.payload())
                        and self.policy.config == self._health_fusion_config)
            if isinstance(self.policy, TextPolicy):
                return (self.blender is None and self.ocr_ready and
                        self.policy.artifact_sha256 == fingerprint(self.policy.payload())
                        and self.policy.config == self.text_config)
        except (AttributeError, TypeError, ValueError):
            return False
        return False

    @property
    def fusion_config(self):
        import json
        vision = self.vision
        return json.loads(json.dumps({'csv_sha256': self.csv_sha256, 'search_backend': self.index.backend,
                'search_version': BLEND_VERSION, 'fusion_version': FUSION_VERSION,
                'blend': asdict(self.blender.config),
                'aliases_sha256': fingerprint(self.index.aliases.variants if self.index.aliases else {}),
                'class_sha256': (getattr(vision.classes, 'archive_sha256',
                                   fingerprint(vision.classes.meta))
                                   if vision and vision.classes else None),
                'gallery_sha256': vision.gallery.archive_sha256 if vision and vision.gallery else None,
                'encoder': vision.encoder.config if vision else None,
                'vision_device': getattr(vision.encoder, 'device', None) if vision else None,
                'vision_executor': ('isolated-v1' if hasattr(vision.encoder, 'encode_images_deadline')
                                    else 'inprocess-v1') if vision else None,
                'query_views': getattr(vision, 'query_views', 1) if vision else None,
                'roi_mode': self.roi_mode,
                'ocr': ocr_profile(self.ocr),
                'dependencies': dependency_profile(roi_mode=self.roi_mode, vision=vision),
                'request_timeout': self.request_timeout,
                'max_concurrent_ocr': self.max_concurrent_ocr,
                'conditional_ocr': {'enabled': self.conditional_ocr,
                                    'min_photo_score': CONDITIONAL_PHOTO_SCORE,
                                    'min_photo_gap': CONDITIONAL_PHOTO_GAP},
                'decision_code_sha256': decision_code_sha256()}, ensure_ascii=False))

    def predict(self, image: bytes, *, bbox=None, manual_text=None, seconds=8.0,
                roi_mode: str | None = None, force_top1: bool = False,
                _include_evidence: bool = False):
        start = time.monotonic()
        deadline = start + seconds
        active_roi_mode = roi_mode or self.roi_mode
        if force_top1 and bbox is None and active_roi_mode in ('refuse', 'auto_bbox_experimental'):
            active_roi_mode = 'center_80_crop'  # explicit evaluation fallback, not label detection
        if active_roi_mode not in ('refuse', 'full_frame_experimental',
                                   'auto_bbox_experimental', 'center_80_crop'):
            raise ValueError('invalid roi_mode')
        request_profile_mismatch = (bbox is not None or active_roi_mode != self.roi_mode
                                    or manual_text is not None)
        result = {'status': 'error', 'slug': None, 'reason_codes': [],
                  'versions': {'csv_sha256': self.csv_sha256,
                               'search_backend': self.index.backend,
                               'policy': self.policy.version if self.policy else None,
                               'fusion': FUSION_VERSION if self.blender else None,
                               'crop': 'oriented-rgb-bbox-v1' if bbox is not None else active_roi_mode},
                  'timings_ms': {}, 'candidates': [], 'ocr': None, 'roi': None}
        def finish(status, reason, slug=None):
            if time.monotonic() >= deadline or 'deadline' in reason:
                status, slug = 'error', None
                if 'deadline' not in reason:
                    reason = [*reason, 'deadline']
            # A request-level ROI override or oracle text was not calibrated by
            # the startup policy; fail closed without altering forced eval.
            if slug is not None and not force_top1 and (request_profile_mismatch or
                    (self.policy is not None and seconds > self.request_timeout)):
                status, reason, slug = 'ambiguous', [*reason, 'uncalibrated_request_profile'], None
            # Vision may validate and rank a large image even when the bounded
            # OCR crop fails. Conversely, never force a slug for invalid bytes
            # or a request that exhausted its shared deadline.
            validated = result.pop('_valid_image', False) or result['roi'] is not None
            if (force_top1 and slug is None and validated
                    and not any(code in reason for code in ('deadline', 'capacity_exceeded', 'busy',
                                                              'vision_worker_failed', 'vision_worker_unavailable',
                                                              'vision_unavailable', 'vision_input_too_large',
                                                              'worker_unavailable', 'worker_failed',
                                                              'inference_failed'))
                    and time.monotonic() < deadline):
                if result['candidates']:
                    slug = result['candidates'][0]['slug']
                    reason = [*reason, 'eval_forced_top1']
                elif self.index.wines:
                    slug = min(w.slug for w in self.index.wines)
                    reason = [*reason, 'eval_catalog_fallback_no_evidence']
            result.update(status=status, slug=slug, reason_codes=reason)
            if not _include_evidence:
                result.pop('_evidence', None)
            result['timings_ms']['total'] = round((time.monotonic()-start)*1000)
            return result
        try:
            if self.blender is not None:
                return self._predict_blended(image, bbox, manual_text, active_roi_mode,
                                             deadline, result, finish, _include_evidence)
            if bbox is None:
                if active_roi_mode == 'full_frame_experimental':
                    roi = full_frame_roi(image, deadline)
                elif active_roi_mode == 'auto_bbox_experimental':
                    roi = auto_bbox_roi(image, deadline)
                elif active_roi_mode == 'center_80_crop':
                    roi = center_crop_80_roi(image, deadline)
                else:
                    # Never imply that a central crop is a detected label.
                    decode(image, deadline)[0].close()
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
            if time.monotonic() >= deadline:
                return finish('error', ['deadline'])
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
            if status == 'accepted' and (not isinstance(self.policy, TextPolicy)
                    or self.policy.artifact_sha256 != fingerprint(self.policy.payload())
                    or self.policy.config != self.current_text_config):
                status, slug, reasons = 'ambiguous', None, ['policy_configuration_changed']
            if bbox is None:
                reasons.append(active_roi_mode)
            return finish(status, reasons, slug)
        except AmbiguousScene as exc:
            return finish('ambiguous_scene', [str(exc)])
        except InvalidImage as exc:
            if force_top1 and str(exc) == 'crop_too_large' and self.index.wines:
                # A valid decoded frame can exceed the permitted OCR crop size.
                # Do not turn a crop failure into an HTTP refusal in eval mode.
                decode(image, deadline)[0].close()
                return finish('ambiguous', ['eval_catalog_fallback_crop_too_large'],
                              min(w.slug for w in self.index.wines))
            return finish('unreadable', [str(exc)])
        except TimeoutError:
            return finish('error', ['deadline'])

    @property
    def current_text_config(self):
        return {'csv_sha256': self.csv_sha256, 'search_backend': self.index.backend,
                'blend_version': BLEND_VERSION,
                'aliases_sha256': fingerprint(self.index.aliases.variants if self.index.aliases else {}),
                'roi_mode': self.roi_mode, 'ocr': ocr_profile(self.ocr),
                'dependencies': dependency_profile(roi_mode=self.roi_mode, vision=self.vision),
                'request_timeout': self.request_timeout,
                'max_concurrent_ocr': self.max_concurrent_ocr,
                'decision_code_sha256': decision_code_sha256()}

    def _predict_blended(self, image, bbox, manual_text, mode, deadline, result, finish,
                         include_evidence=False):
        """Independent visual and OCR branches under the same deadline.

        A failed/missing branch does not zero out other sources. Full image is
        vision input; manual bbox selects the vision label crop as well.
        """
        from .ocr import MockOCR
        import io
        from PIL import Image

        roi = None
        roi_error = None
        if bbox is not None:
            roi = manual_roi(image, bbox, deadline)
        elif mode in ('full_frame_experimental', 'center_80_crop'):
            try:
                roi = (full_frame_roi(image, deadline) if mode == 'full_frame_experimental'
                       else center_crop_80_roi(image, deadline))
            except InvalidImage as exc:
                if str(exc) != 'crop_too_large' or self.vision is None:
                    raise
                roi_error = 'crop_too_large'  # vision can use the decoded full frame
        elif mode == 'auto_bbox_experimental':
            try:
                roi = auto_bbox_roi(image, deadline)
            except InvalidImage as exc:
                if self.vision is None:
                    raise
                # No OCR ROI, but whole-frame vision is still a valid weak signal.
                roi_error = str(exc)
                roi = None
        if roi is not None:
            result['roi'] = {'roi_id': roi.roi_id, 'source_sha256': roi.source_sha256,
                             'source_size': roi.source_size, 'bbox': roi.bbox,
                             'status': roi.status, 'crop_version': roi.crop_version}
        elif mode == 'refuse' and self.vision is None:
            decode(image, deadline)[0].close()
            return finish('unreadable', ['automatic_roi_unavailable'])

        evidence = []
        reasons = [roi_error] if roi_error else []
        if self.vision is not None:
            visual_image = (Image.open(io.BytesIO(roi.crop)).convert('RGB') if bbox and roi
                            else decode(image, deadline)[0])
            result['_valid_image'] = True  # successful decode, independent of OCR crop
            try:
                evidence = self.vision.observe(visual_image, deadline)
            except RuntimeError as exc:
                code = str(exc)
                reasons.append(code if code in ('vision_capacity_exceeded', 'vision_worker_failed',
                                                'vision_worker_unavailable', 'vision_input_too_large')
                               else 'vision_unavailable')
            finally:
                visual_image.close()
        ocr = MockOCR(manual_text) if manual_text is not None else self.ocr
        skip_ocr = False
        if self.conditional_ocr and manual_text is None and self.vision and evidence:
            photo = sorted((e for e in evidence if e.source == 'photo'),
                           key=lambda e: (e.rank, e.slug))
            if (len(photo) > 1 and photo[0].score >= CONDITIONAL_PHOTO_SCORE
                    and photo[0].score - photo[1].score >= CONDITIONAL_PHOTO_GAP
                    and len(self.blender.families[self.blender.key[photo[0].slug]]) == 1):
                skip_ocr = True
                reasons.append('conditional_ocr_skipped_experimental')
        if roi is not None and ocr is not None and not skip_ocr:
            if not self.ocr_slots.acquire(blocking=False):
                reasons.append('ocr_capacity_exceeded')
            else:
                try:
                    if time.monotonic() < deadline:
                        observed = ocr.recognize(roi, deadline)
                        result['ocr'] = asdict(observed)
                        if observed.status == 'ok' and observed.raw_text:
                            cleaned = clean_ocr(observed.raw_text)
                            result['ocr']['cleaned_text'] = cleaned
                            text_candidates = self.index.search(cleaned, k=self.blender.config.top_k)
                            evidence.extend(CandidateEvidence(c.slug, 'text', c.score,
                                               c.rank, 'text_score_0_100', BLEND_VERSION)
                                            for c in text_candidates)
                        elif observed.status != 'no_text':
                            reasons.append('ocr_' + observed.status)
                            if observed.error_code:
                                reasons.append(observed.error_code)
                    else:
                        reasons.append('deadline')
                finally:
                    self.ocr_slots.release()
        elif not skip_ocr:
            reasons.append('ocr_not_configured' if ocr is None else 'automatic_roi_unavailable')
        if include_evidence:
            result['_evidence'] = [asdict(e) for e in evidence]
        if time.monotonic() >= deadline:
            return finish('error', [*reasons, 'deadline'])
        text = (result['ocr'] or {}).get('cleaned_text')
        classes = getattr(self.vision, 'classes', None)
        compatible = (isinstance(self.policy, FusionPolicy) and
                      self.policy.artifact_sha256 == fingerprint(self.policy.payload()) and
                      self.policy.config == self.fusion_config)
        ranked, status, slug, policy_reasons, confidence = rank_and_decide(
            self.blender, evidence, text, policy=self.policy if compatible else None,
            classes=classes, config=self.fusion_config if compatible else None)
        if self.policy is not None and not compatible:
            policy_reasons = ['policy_configuration_changed']
        if slug is not None and any(r.startswith(('vision_', 'ocr_')) and r not in
                                     ('ocr_not_configured',) for r in reasons):
            # A policy calibrated with all sources cannot accept a partial
            # observation when an inference worker is unavailable.
            status, slug = 'ambiguous', None
            policy_reasons = [*policy_reasons, 'partial_evidence']
        if time.monotonic() >= deadline:
            return finish('error', [*reasons, 'deadline'])
        result['candidates'] = ranked[:20]
        result['versions']['sources'] = sorted({e.source for e in evidence})
        if any(code in reasons for code in ('ocr_capacity_exceeded',
                                            'vision_capacity_exceeded', 'busy')):
            # Saturation must not be silently promoted to acceptance using the
            # other branch's evidence under an uncalibrated partial profile.
            return finish('error', [*reasons, 'capacity_exceeded'])
        if not ranked:
            if any(r in reasons for r in ('vision_worker_failed', 'vision_worker_unavailable',
                                          'vision_input_too_large', 'worker_unavailable',
                                          'worker_failed', 'inference_failed')):
                return finish('error', [*reasons, 'no_candidates'])
            return finish('unreadable' if not evidence else 'unknown',
                          [*reasons, 'no_candidates'])
        result['margin'] = (ranked[0]['blend_score'] - ranked[1]['blend_score']
                            if len(ranked) > 1 else None)
        if confidence is not None:
            result['confidence'] = confidence
        if mode != 'refuse' and bbox is None:
            reasons.append(mode)
        return finish(status, [*reasons, *policy_reasons], slug)
