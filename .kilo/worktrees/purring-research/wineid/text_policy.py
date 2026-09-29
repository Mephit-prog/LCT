"""Text-branch accept policy bound to catalog/backend/calibration provenance (B05).

A bare threshold string (``Policy.version``) is not enough: the text policy must
be tied to the exact catalog checksum, search backend and blend version it was
calibrated against, carry the calibration manifest hash and its source groups,
and its own artifact checksum. Incompatible artifacts and calibration/test
leakage are rejected instead of silently producing accepted answers.

The runtime scoring itself stays in :mod:`wineid.search` / :mod:`wineid.pipeline`;
this module only owns the versioned policy artifact and its validation.
"""
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .data import jsonl, query_manifest_errors, calibration_test_leakage_errors

TEXT_POLICY_VERSION = 'text-policy-v1'
TEXT_POLICY_CONFIG_KEYS = ('csv_sha256', 'search_backend', 'blend_version')


def fingerprint(payload: dict) -> str:
    """Deterministic SHA-256 over a JSON payload (same rule as VisualPolicy)."""
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


@dataclass(frozen=True)
class TextPolicy:
    """Versioned text accept policy with configuration and calibration binding."""

    version: str
    min_score: float
    min_margin: float
    max_distance: float
    config: dict
    calibration_manifest_sha256: str
    calibration_source_groups: tuple
    artifact_sha256: str

    def __post_init__(self):
        if (self.version != TEXT_POLICY_VERSION
                or not all(math.isfinite(x) for x in
                           (self.min_score, self.min_margin, self.max_distance))
                or not 0 <= self.min_score <= 100
                or not 0 <= self.min_margin <= 100
                or not 0 <= self.max_distance <= 1
                or not isinstance(self.config, dict)
                or any(key not in self.config for key in TEXT_POLICY_CONFIG_KEYS)
                or not _is_sha256(self.config.get('csv_sha256'))
                or not isinstance(self.config.get('search_backend'), str)
                or not self.config.get('search_backend')
                or not isinstance(self.config.get('blend_version'), str)
                or not self.config.get('blend_version')
                or not _is_sha256(self.calibration_manifest_sha256)
                or not isinstance(self.calibration_source_groups, (list, tuple))
                or not self.calibration_source_groups
                or any(not isinstance(g, str) or not g for g in self.calibration_source_groups)
                or len(set(self.calibration_source_groups)) != len(self.calibration_source_groups)
                or self.artifact_sha256 != fingerprint(self.payload())):
            raise ValueError('invalid text policy artifact or checksum')

    def payload(self) -> dict:
        return {'version': self.version, 'min_score': self.min_score,
                'min_margin': self.min_margin, 'max_distance': self.max_distance,
                'config': self.config,
                'calibration_manifest_sha256': self.calibration_manifest_sha256,
                'calibration_source_groups': list(self.calibration_source_groups)}

    def check_config(self, config: dict) -> None:
        """Reject a policy calibrated for a different catalog/backend/blend."""
        if self.config != config:
            raise ValueError('text policy configuration mismatch')


def build_text_policy(calibration_manifest, *, min_score: float, min_margin: float,
                      max_distance: float, config: dict) -> TextPolicy:
    """Build a policy from a validation-only calibration manifest.

    ``config`` must contain the current ``csv_sha256``, ``search_backend`` and
    ``blend_version`` (see :mod:`wineid.search`). All calibration rows must be
    ``split='validation'`` and pass the shared whole-manifest audit.
    """
    path = Path(calibration_manifest).resolve()
    if not path.is_file():
        raise ValueError('calibration manifest not found')
    rows = jsonl(path)
    if not rows:
        raise ValueError('empty calibration manifest')
    errors, groups = query_manifest_errors(rows)
    if errors:
        raise ValueError(errors[0])
    if any(split != 'validation' for split in groups.values()):
        raise ValueError('calibration manifest must be validation-only')
    payload = {'version': TEXT_POLICY_VERSION, 'min_score': min_score,
               'min_margin': min_margin, 'max_distance': max_distance,
               'config': config,
               'calibration_manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'calibration_source_groups': sorted(groups)}
    return TextPolicy(**payload, artifact_sha256=fingerprint(payload))


def check_calibration_test_leakage(policy: TextPolicy, calibration_manifest,
                                  test_manifest) -> None:
    """Reject a test manifest sharing groups, hashes or query ids with calibration.

    Also verifies that the calibration manifest actually matches the policy hash.
    """
    calibration_path = Path(calibration_manifest).resolve()
    test_path = Path(test_manifest).resolve()
    if hashlib.sha256(calibration_path.read_bytes()).hexdigest() != policy.calibration_manifest_sha256:
        raise ValueError('calibration manifest hash mismatch')
    errors = calibration_test_leakage_errors(jsonl(calibration_path), jsonl(test_path),
                                             calibration_path=calibration_path,
                                             test_path=test_path)
    if errors:
        raise ValueError(errors[0])


def load_text_policy(data: dict) -> TextPolicy:
    return TextPolicy(**data)


def load_policy(path) -> object:
    """Load either a ``TextPolicy`` (versioned) or a legacy ``Policy`` artifact."""
    from .pipeline import Policy
    data = json.loads(Path(path).read_text())
    if data.get('version') == TEXT_POLICY_VERSION:
        return load_text_policy(data)
    return Policy(**data)
