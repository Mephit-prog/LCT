"""One blended ranking + acceptance gate for online and independent evaluation."""
from .fusion_policy import FusionPolicy
from .text_policy import fingerprint


def rank_and_decide(blender, evidence, text, *, policy=None, classes=None,
                    config=None):
    """Return (ranked, status, slug, reasons, confidence).

    ``config`` is the *current* Pipeline.fusion_config when accepting. Never
    apply an artifact to offline evidence without checking this configuration.
    """
    ranked = blender.rank(evidence, text)
    if not ranked:
        return ranked, 'unknown', None, ['no_candidates'], None
    if not isinstance(policy, FusionPolicy):
        return ranked, 'ambiguous', None, ['policy_not_calibrated'], None
    if (config is None or policy.config != config
            or policy.artifact_sha256 != fingerprint(policy.payload())):
        raise ValueError('fusion policy configuration mismatch')
    confidence = policy.confidence(ranked)
    status, slug, reasons = policy.decide(ranked)
    collisions = classes.meta.get('prompt_collisions', {}) if classes else {}
    if (status == 'accepted' and slug in collisions and
            not any(s != 'class' for s in ranked[0]['signals'])):
        status, slug, reasons = 'ambiguous', None, ['indistinguishable_class_prompt']
    return ranked, status, slug, reasons, confidence
