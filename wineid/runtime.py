"""Single HTTP bootstrap with preflight validation and deterministic OCR cleanup.

The stdlib and FastAPI adapters share this process-local runtime. Neither
adapter supplies cross-process resource limits; run with one API process.
"""
import atexit
import importlib.util
import json
import math
import os
import re
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from .blend import Blender, BlendConfig
from .canonical import ProducerAliases
from .catalog import load_catalog
from .fusion_vision import load_vision
from .ocr import configured_ocr
from .pipeline import Pipeline
from .search import Index
from .text_policy import load_policy


@dataclass(frozen=True)
class Settings:
    csv: str
    aliases: str
    policy: str | None
    classes: str | None
    gallery: str | None
    adapter: str | None
    blend_config: str | None
    roi_mode: str
    query_views: int
    conditional_ocr: bool
    remote_code: bool
    device: str | None
    timeout: float
    vision_startup_timeout: float
    max_concurrent_ocr: int
    max_workers: int
    host: str
    port: int

    @classmethod
    def from_env(cls, env):
        def positive_float(key, default, maximum):
            try:
                value = float(env.get(key, default))
            except (ValueError, TypeError) as exc:
                raise ValueError(f'invalid {key}') from exc
            if not math.isfinite(value) or not 0 < value <= maximum:
                raise ValueError(f'invalid {key}')
            return value

        def integer(key, default, low, high):
            try:
                value = int(env.get(key, default))
            except (ValueError, TypeError) as exc:
                raise ValueError(f'invalid {key}') from exc
            if not low <= value <= high or str(value) != str(env.get(key, default)):
                raise ValueError(f'invalid {key}')
            return value

        def flag(key):
            value = env.get(key, '0')
            if value not in ('0', '1'):
                raise ValueError(f'invalid {key}')
            return value == '1'

        roi = env.get('WINE_ROI_MODE', 'refuse')
        if roi not in ('refuse', 'center_80_crop', 'full_frame_experimental',
                       'auto_bbox_experimental'):
            raise ValueError('invalid WINE_ROI_MODE')
        if roi == 'auto_bbox_experimental' and importlib.util.find_spec('cv2') is None:
            raise ValueError('auto bbox requires OpenCV')
        provider = env.get('WINE_OCR_PROVIDER', 'none')
        mode = env.get('WINE_OCR_MODE', 'cold')
        if (provider not in ('none', 'tesseract', 'paddle', 'mineru', 'mineru_local', 'mistral', 'mock')
                or mode not in ('cold', 'warm')
                or (mode == 'warm' and provider != 'paddle')):
            raise ValueError('invalid OCR provider/mode')
        if provider == 'mock' and not flag('WINE_ENABLE_MOCK_OCR'):
            raise ValueError('mock OCR must be explicitly enabled')
        if provider == 'mineru' and not env.get('MINERU_TOKEN'):
            raise ValueError('MINERU_TOKEN required')
        if provider == 'mistral' and not env.get('MISTRAL_API_KEY'):
            raise ValueError('MISTRAL_API_KEY required')
        if provider == 'paddle' and mode == 'warm':
            for key in ('WINE_OCR_MANIFEST', 'WINE_PADDLE_DET_DIR', 'WINE_PADDLE_REC_DIR'):
                if not env.get(key):
                    raise ValueError(f'{key} required for warm Paddle')
            for key in ('WINE_PADDLE_DET_DIR', 'WINE_PADDLE_REC_DIR'):
                if not Path(env[key]).is_dir():
                    raise ValueError(f'missing directory: {key}')
            positive_float('WINE_OCR_STARTUP_TIMEOUT', '120', 600)
        if provider == 'tesseract' and env.get('WINE_TESSDATA_DIR') and not Path(env['WINE_TESSDATA_DIR']).is_dir():
            raise ValueError('missing WINE_TESSDATA_DIR')
        if env.get('WINE_OCR_STARTUP_TIMEOUT'):
            positive_float('WINE_OCR_STARTUP_TIMEOUT', '120', 600)
        if env.get('WINE_POLICY') and provider in ('mock', 'mistral', 'mineru', 'mineru_local'):
            raise ValueError('acceptance requires a pinned local OCR profile')
        if env.get('WINE_POLICY') and provider == 'paddle' and mode != 'warm':
            raise ValueError('acceptance requires verified warm Paddle weights')
        if env.get('WINE_POLICY') and provider == 'tesseract' and not env.get('WINE_TESSDATA_DIR'):
            raise ValueError('acceptance requires pinned Tesseract traineddata')
        cfg = cls(
            csv=env.get('WINE_CSV', 'strapi_output0709.csv'),
            aliases=env.get('WINE_ALIASES', 'producer_aliases.txt'),
            policy=env.get('WINE_POLICY') or None,
            classes=env.get('WINE_CLASS_INDEX') or None,
            gallery=env.get('WINE_GALLERY') or None,
            adapter=env.get('WINE_ADAPTER') or None,
            blend_config=env.get('WINE_BLEND_CONFIG') or None,
            roi_mode=roi, query_views=integer('WINE_QUERY_VIEWS', '1', 1, 3),
            conditional_ocr=flag('WINE_CONDITIONAL_OCR'),
            remote_code=flag('WINE_ALLOW_REMOTE_CODE'),
            device=env.get('WINE_DEVICE') or None,
            timeout=positive_float('WINE_REQUEST_TIMEOUT', '8.5', 600),
            vision_startup_timeout=positive_float('WINE_VISION_STARTUP_TIMEOUT', '120', 600),
            max_concurrent_ocr=integer('WINE_MAX_CONCURRENT_OCR', '2', 1, 32),
            max_workers=integer('WINE_MAX_WORKERS', '8', 1, 128),
            host=env.get('WINE_HOST', '127.0.0.1'),
            port=integer('WINE_PORT', '8080', 1, 65535))
        if cfg.query_views not in (1, 3):
            raise ValueError('invalid WINE_QUERY_VIEWS')
        if (cfg.classes or cfg.gallery) and not cfg.remote_code:
            raise ValueError('approve remote code before loading vision')
        if cfg.adapter and not (cfg.classes or cfg.gallery):
            raise ValueError('adapter requires a vision index')
        if cfg.query_views != 1 and not (cfg.classes or cfg.gallery):
            raise ValueError('query views require a vision index')
        if cfg.conditional_ocr and (not (cfg.classes or cfg.gallery) or provider == 'none'):
            raise ValueError('conditional OCR requires vision and an OCR provider')
        if cfg.device and not (cfg.classes or cfg.gallery):
            raise ValueError('device requires a vision index')
        if cfg.device and not re.fullmatch(r'(cpu|mps|cuda(?::\d+)?)', cfg.device):
            raise ValueError('invalid WINE_DEVICE')
        for key, path in (('WINE_CSV', cfg.csv), ('WINE_POLICY', cfg.policy),
                          ('WINE_CLASS_INDEX', cfg.classes), ('WINE_GALLERY', cfg.gallery),
                          ('WINE_ADAPTER', cfg.adapter), ('WINE_BLEND_CONFIG', cfg.blend_config),
                          ('WINE_OCR_MANIFEST', env.get('WINE_OCR_MANIFEST'))):
            if path and not Path(path).is_file():
                raise ValueError(f'missing file: {key}')
        if not Path(cfg.aliases).is_file():
            raise ValueError('missing file: WINE_ALIASES')
        if provider == 'tesseract':
            import shutil
            if not shutil.which(env.get('WINE_TESSERACT', 'tesseract')):
                raise ValueError('missing Tesseract binary')
        if provider in ('paddle', 'mineru_local') and env.get('WINE_OCR_PYTHON'):
            if not Path(env['WINE_OCR_PYTHON']).is_file():
                raise ValueError('missing WINE_OCR_PYTHON')
        return cfg



@dataclass(frozen=True)
class Runtime:
    settings: Settings
    pipeline: Pipeline
    wines: list
    catalog: dict


@contextmanager
def open_runtime(env=None):
    """Fail before serving; close OCR on both normal and partial startup failure."""
    env = dict(os.environ if env is None else env)
    settings = Settings.from_env(env)  # no model construction before preflight
    if env.get('WINE_RELEASE_MANIFEST'):
        from .release_bundle import verify
        manifest_path = Path(env['WINE_RELEASE_MANIFEST']).resolve()
        manifest = verify(manifest_path.parent,
                          json.loads(manifest_path.read_text(encoding='utf-8')))
        roles = {'csv': settings.csv, 'aliases': settings.aliases,
                 'policy': settings.policy, 'classes': settings.classes,
                 'gallery': settings.gallery, 'adapter': settings.adapter,
                 'ocr_manifest': env.get('WINE_OCR_MANIFEST')}
        if env.get('WINE_OCR_PROVIDER') == 'tesseract':
            import shutil
            binary = shutil.which(env.get('WINE_TESSERACT', 'tesseract'))
            directory = env.get('WINE_TESSDATA_DIR')
            roles.update({'tesseract': binary,
                          'rus-traineddata': str(Path(directory) / 'rus.traineddata') if directory else None,
                          'eng-traineddata': str(Path(directory) / 'eng.traineddata') if directory else None})
        for role, configured in roles.items():
            recorded = manifest['files'].get(role)
            if bool(configured) != bool(recorded) or (configured and
                    Path(configured).resolve() !=
                    (manifest_path.parent / recorded['path']).resolve()):
                raise ValueError('release bundle/runtime path mismatch: ' + role)
    policy = load_policy(settings.policy, allow_legacy=False) if settings.policy else None
    config = (BlendConfig(**json.loads(Path(settings.blend_config).read_text(encoding='utf-8')))
              if settings.blend_config else BlendConfig())
    from .fusion_policy import FusionPolicy
    from .text_policy import TextPolicy
    has_blend = bool(settings.classes or settings.gallery or settings.blend_config)
    if (isinstance(policy, TextPolicy) and has_blend) or (isinstance(policy, FusionPolicy) and not has_blend):
        raise ValueError('policy/vision/blend profile mismatch')
    wines, catalog = load_catalog(settings.csv)
    aliases = ProducerAliases.load(settings.aliases) if Path(settings.aliases).is_file() else None
    index = Index(wines, aliases=aliases)
    with ExitStack() as stack:
        ocr = configured_ocr(env)
        if hasattr(ocr, 'close'):
            stack.callback(ocr.close)
            # Last-resort cleanup for interpreter exit; ExitStack is primary.
            atexit.register(ocr.close)
            stack.callback(atexit.unregister, ocr.close)
        # A failed OCR canary/manifest must not trigger expensive vision loading.
        if hasattr(ocr, 'start'):
            ocr.start()
        vision = load_vision(wines, classes_path=settings.classes,
                             gallery_path=settings.gallery, adapter_path=settings.adapter,
                             allow_remote_code=settings.remote_code, device=settings.device,
                             query_views=settings.query_views, isolated=True,
                             startup_timeout=settings.vision_startup_timeout)
        if hasattr(vision, 'close'):
            stack.callback(vision.close)
        blender = Blender(wines, aliases=aliases, config=config) if vision or settings.blend_config else None
        if policy is not None:
            from .profile import ocr_profile
            profile = ocr_profile(ocr)
            if profile and 'traineddata_sha256' in profile and (
                    not profile['binary_sha256'] or
                    not all(profile['traineddata_sha256'].values())):
                raise ValueError('Tesseract weights/binary unavailable for acceptance')
            if profile and 'manifest_sha256' in profile and not profile['manifest_sha256']:
                raise ValueError('unverified OCR manifest')
        pipeline = Pipeline(index, ocr, policy, catalog['csv_sha256'], settings.roi_mode,
                            vision=vision, blender=blender,
                            conditional_ocr=settings.conditional_ocr,
                            max_concurrent_ocr=settings.max_concurrent_ocr,
                            request_timeout=settings.timeout)
        yield Runtime(settings, pipeline, wines, catalog)
