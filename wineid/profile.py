"""Decision-path provenance for versioned acceptance policies.

Any edit to the listed scoring/normalization/ROI/OCR sources invalidates an old
policy, even if a developer forgets to bump a hand-maintained version string.
This is intentionally conservative (comments also change the digest).
"""
import hashlib
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

DECISION_FILES = ('augment.py', 'blend.py', 'blend_decision.py', 'canonical.py', 'catalog.py',
                  'clip_zero_shot.py', 'clip_roi.py', 'fusion_eval.py', 'fusion_policy.py',
                  'fusion_run.py', 'fusion_vision.py',
                  'jina_clip.py', 'local_ocr.py', 'local_ocr_worker.py', 'ocr.py', 'pipeline.py',
                  'profile.py', 'roi.py', 'runtime.py', 'search.py', 'text_policy.py',
                  'visual.py', 'vision_ipc.py', 'vision_worker.py',
                  'warm_ocr.py', 'warm_ocr_worker.py')


def decision_code_sha256():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in DECISION_FILES:
        digest.update(name.encode('ascii') + b'\x00')
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def dependency_profile(*, roi_mode, vision):
    """Pin libraries that change image decoding, normalization or scoring.

    Release bundle pins are still needed: this only refuses a changed installed
    version for an existing calibrated policy.
    """
    packages = ['Pillow', 'fuzzywuzzy', 'numpy']
    if roi_mode == 'auto_bbox_experimental':
        # Both opencv-python and the headless wheel expose the same cv2 module.
        # Pin the imported implementation rather than one distribution name.
        import cv2
        opencv_version = cv2.__version__
    else:
        opencv_version = None
    if vision is not None:
        packages.extend(('torch', 'transformers'))
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {'python': '.'.join(map(str, sys.version_info[:3])),
            'packages': versions, 'opencv': opencv_version}


def ocr_profile(ocr):
    if ocr is None:
        return None
    from .text_policy import fingerprint
    data = {'provider': type(ocr).__module__ + '.' + type(ocr).__name__,
            'model': getattr(ocr, 'model', None)}
    from .warm_ocr import WarmPaddleOCR
    from .local_ocr import LocalOCR
    from .ocr import MistralOCR, MockOCR
    if isinstance(ocr, WarmPaddleOCR):
        # Set only after canary + full model-file verification at startup.
        data['manifest_sha256'] = ocr._manifest_digest
        data['python'] = ocr.command[0]
    elif isinstance(ocr, LocalOCR):
        data['backend'] = ocr.backend
        data['python'] = ocr.python if ocr.backend != 'tesseract' else None
        if ocr.backend == 'tesseract':
            import shutil
            binary = shutil.which(ocr.tesseract)
            data['binary_sha256'] = (hashlib.sha256(Path(binary).read_bytes()).hexdigest()
                                     if binary else None)
            # For acceptance, pin the explicit directory, not a host's
            # implicit TESSDATA_PREFIX/default package installation.
            directory = Path(ocr.tessdata_dir) if ocr.tessdata_dir else None
            data['traineddata_sha256'] = {
                lang: hashlib.sha256((directory / (lang + '.traineddata')).read_bytes()).hexdigest()
                if directory and (directory / (lang + '.traineddata')).is_file() else None
                for lang in ('rus', 'eng')}
    elif isinstance(ocr, MistralOCR):
        data['endpoint'] = ocr.endpoint
    elif isinstance(ocr, MockOCR):
        data['fixture_sha256'] = fingerprint({'text': ocr.text})
    return data
