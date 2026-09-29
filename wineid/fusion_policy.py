"""Acceptance gate separate from forced closed-set evaluation ranking.

Requires an audited real validation manifest. Thresholds are never supplied by
this module; synthetic examples are not calibration for unknown wines.
"""
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from .data import query_manifest_errors
from .text_policy import fingerprint

VERSION = 'fusion-policy-v4'
CONFIG_KEYS = {'csv_sha256', 'search_backend', 'search_version', 'fusion_version',
               'blend', 'aliases_sha256', 'class_sha256', 'gallery_sha256',
               'encoder', 'vision_device', 'vision_executor', 'query_views', 'roi_mode', 'ocr', 'conditional_ocr',
               'dependencies', 'request_timeout', 'max_concurrent_ocr',
               'decision_code_sha256'}


@dataclass(frozen=True)
class FusionPolicy:
    version: str
    config: dict
    calibration_manifest_sha256: str
    calibration_source_groups: list
    min_cosine: float
    min_text_score: float
    min_margin: float
    artifact_sha256: str
    temperature: float = 1.0

    def __post_init__(self):
        if (self.version != VERSION or not isinstance(self.config, dict)
                or set(self.config) != CONFIG_KEYS
                or any(not isinstance(self.config.get(k), str) or len(self.config[k]) != 64
                       or any(c not in '0123456789abcdef' for c in self.config[k])
                       for k in ('aliases_sha256', 'decision_code_sha256'))
                or self.config.get('vision_executor') not in (None, 'isolated-v1', 'inprocess-v1')
                or not isinstance(self.config.get('blend'), dict)
                or not isinstance(self.config.get('dependencies'), dict)
                or not isinstance(self.config.get('request_timeout'), (int, float))
                or not math.isfinite(self.config['request_timeout'])
                or not 0 < self.config['request_timeout'] <= 600
                or type(self.config.get('max_concurrent_ocr')) is not int
                or self.config['max_concurrent_ocr'] < 1
                or not isinstance(self.config.get('conditional_ocr'), dict)
                or set(self.config['conditional_ocr']) != {'enabled', 'min_photo_score', 'min_photo_gap'}
                or type(self.config['conditional_ocr']['enabled']) is not bool
                or not isinstance(self.calibration_manifest_sha256, str)
                or len(self.calibration_manifest_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.calibration_manifest_sha256)
                or not isinstance(self.calibration_source_groups, list)
                or not self.calibration_source_groups
                or len(set(self.calibration_source_groups)) != len(self.calibration_source_groups)
                or any(not isinstance(g, str) or not g for g in self.calibration_source_groups)
                or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in
                           (self.min_cosine, self.min_text_score, self.min_margin))
                or not -1 <= self.min_cosine <= 1 or not 0 <= self.min_text_score <= 100
                or self.min_margin < 0 or not math.isfinite(self.temperature)
                or self.temperature <= 0 or self.artifact_sha256 != fingerprint(self.payload())):
            raise ValueError('invalid fusion policy artifact')

    def payload(self):
        return {k: getattr(self, k) for k in ('version', 'config',
                'calibration_manifest_sha256', 'calibration_source_groups',
                'min_cosine', 'min_text_score', 'min_margin', 'temperature')}

    def check_config(self, config):
        if self.config != config:
            raise ValueError('fusion policy configuration mismatch')

    def confidence(self, ranked):
        """Conditional-on-retrieved-pool softmax, NOT probability of correct SKU."""
        if not ranked:
            return None
        peak = ranked[0]['blend_score']
        exp = [math.exp((r['blend_score'] - peak) / self.temperature) for r in ranked]
        total = sum(exp)
        probs = [x / total for x in exp]
        return {'p_top1_given_pool': probs[0], 'p_top5_given_pool': sum(probs[:5]),
                'margin_given_pool': probs[0] - (probs[1] if len(probs) > 1 else 0.)}

    def decide(self, ranked):
        if not ranked:
            return 'unknown', None, ['no_candidates']
        top = ranked[0]
        signals = top['signals']
        if not any((sig['score'] >= self.min_cosine if source in ('photo', 'class')
                    else sig['score'] >= self.min_text_score)
                   for source, sig in signals.items()):
            return 'unknown', None, ['weak_evidence']
        if len(ranked) < 2 or top['blend_score'] - ranked[1]['blend_score'] < self.min_margin:
            return 'ambiguous', None, ['close_candidates']
        return 'accepted', top['slug'], ['fusion_policy']


def build_fusion_policy(manifest, *, wines, config, min_cosine, min_text_score, min_margin,
                        temperature=1., gallery_manifest=None):
    path = Path(manifest).resolve()
    if config.get('gallery_sha256') and not gallery_manifest:
        raise ValueError('gallery manifest required for leakage audit')
    from .fusion_eval import checked_queries
    rows = checked_queries(path, wines, gallery_manifest)
    if not rows:
        raise ValueError('empty calibration manifest')
    errors, groups = query_manifest_errors(rows, catalog_ids={w.slug for w in wines})
    if errors:
        raise ValueError(errors[0])
    if any(split != 'validation' for split in groups.values()):
        raise ValueError('calibration manifest must be validation-only')
    payload = dict(version=VERSION, config=config,
                   calibration_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   calibration_source_groups=sorted(groups), min_cosine=min_cosine,
                   min_text_score=min_text_score, min_margin=min_margin,
                   temperature=temperature)
    return FusionPolicy(**payload, artifact_sha256=fingerprint(payload))
