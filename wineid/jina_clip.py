"""Frozen local Jina CLIP v2 encoders. Importing this module downloads nothing."""
import re

import numpy as np
from PIL import Image

MODEL = 'jinaai/jina-clip-v2'
REVISION = 'e10d47f5691d0454a0fb5d13f46f2199b74cb436'
CODE_REVISION = '39e6a55ae971b59bea6e44675d237c99762e7ee2'
PREPROCESS = f'jina-native-512-rgb-fp32-v1:{CODE_REVISION}'


class IncompatibleEmbeddings(ValueError):
    pass


def unit_vectors(vectors, count, dim=None):
    vectors = np.asarray(vectors, dtype='float32')
    if (vectors.ndim != 2 or vectors.shape[0] != count or vectors.shape[1] < 1
            or (dim is not None and vectors.shape[1] != dim)
            or not np.isfinite(vectors).all()):
        raise IncompatibleEmbeddings('invalid embeddings/dimensions')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms < 1e-12).any():
        raise IncompatibleEmbeddings('zero or invalid embedding')
    return vectors / norms


class JinaCLIPEmbedder:
    """One shared model for image/text encodings, without English task prefixes.

    Remote code executes locally: explicit approval is required even with cached
    weights. Upstream also loads the text tower's config/code transitively.
    FP32 and standard attention keep CPU use possible without CUDA extensions.
    """
    def __init__(self, model_name=MODEL, revision=REVISION, *, batch_size=8,
                 device=None, allow_remote_code=False):
        if model_name != MODEL or not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('expected jinaai/jina-clip-v2 and a full HF commit SHA')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be positive')
        if not allow_remote_code:
            raise ValueError('review CC BY-NC 4.0 and HF remote code; explicit approval required')
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

        self.model_name, self.revision = model_name, revision
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch = torch
        options = dict(revision=revision, code_revision=CODE_REVISION, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name, **options, torch_dtype=torch.float32,
            use_safetensors=True, use_text_flash_attn=False, use_vision_xformers=False,
        ).to(self.device).eval()
        # Upstream lazy getters otherwise reload these from unpinned main.
        self.model.tokenizer = AutoTokenizer.from_pretrained(model_name, **options)
        self.model.preprocess = AutoImageProcessor.from_pretrained(model_name, **options, use_fast=False)
        self.model.requires_grad_(False)

    @property
    def config(self):
        return {'model': self.model_name, 'revision': self.revision,
                'code_revision': CODE_REVISION, 'preprocess': PREPROCESS,
                'text_task': None, 'normalize': 'l2', 'dimensions': 1024}

    def encode_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError('nonempty texts required')
        with self.torch.inference_mode():
            vectors = self.model.encode_text(
                texts, task=None, batch_size=self.batch_size,
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False, truncate_dim=None,
            )
        return unit_vectors(vectors, len(texts), 1024)

    def encode_images(self, images):
        images = list(images)
        if not images or any(not isinstance(image, Image.Image) for image in images):
            raise ValueError('nonempty local PIL images required (no URLs)')
        with self.torch.inference_mode():
            vectors = self.model.encode_image(
                images, batch_size=self.batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False, truncate_dim=None,
            )
        return unit_vectors(vectors, len(images), 1024)
