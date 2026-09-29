"""Vision-LoRA fine-tuning per tz-clip-text-ranking.md §6 (text tower frozen).

Rules implemented:
- text tower fully frozen; LoRA on Q,K,V + output projection of the last
  ``lora_blocks`` vision blocks (rank 8, alpha 16, dropout 0.05);
- vision projection trained separately (not a LoRA layer), AdamW;
- symmetric InfoNCE, learnable temperature (start 0.07);
- positive = one frame and the canonical text of its wine_id; multiple frames
  of one wine are multi-positives to the same t_wine;
- batch negative weights: same-producer 2, random 1 (torch path mirrors the
  numpy reference ``info_nce_loss``);
- gates: train requires >=300 labels with field frames; after 3 epochs abort if
  val Recall@1 does not beat the frozen zero-shot index (ТЗ §6.1/§6.2);
- early stopping by val Recall@1, never by loss.

The trained adapter is saved as {"meta": ..., "state": ...} and loaded back
with ``load_adapter`` for inference. Before training, run ``--verify-only`` to
smoke-check the LoRA projection mapping on the target architecture (review
P1-5): ``attach_lora`` supports HF CLIP (``encoder.layers[i].self_attn.
{q,k,v,out}_proj``) and the EVA02 vision tower of jina-clip-v2
(``blocks[i].attn.{q,k,v}_proj`` + ``attn.proj``); ``--verify-only`` also runs
one forward/backward and checks that every LoRA module receives a gradient.
The trainable vision projection is ``visual_projection`` (jina-clip-v2).

``images_root`` resolves the images.csv ``path`` column (relative to the
catalog_export ``--uploads`` root); defaults to the images.csv directory.

The module imports torch lazily so dataset/loss logic stays testable offline.
"""
import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .canonical import canonical_text
from .jina_clip import REVISION

MIN_TRAIN_LABELS = 300
MIN_FRAMES_PER_WINE = 2


@dataclass(frozen=True)
class TrainItem:
    image_id: str
    wine_id: str
    path: str
    view: str
    producer: str
    split: str
    name_ru: str
    color: str
    vintage: str
    usable_train: int


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def load_datasets(wines_csv, images_csv) -> dict[str, list[TrainItem]]:
    """Join wines.csv/images.csv; rows are grouped per split by producer."""
    wines_rows = {r['wine_id']: r for r in _read_csv(Path(wines_csv))}
    images_rows = _read_csv(Path(images_csv))
    items = []
    for row in images_rows:
        wine = wines_rows.get(row['wine_id'])
        if wine is None:
            raise ValueError(f"images.csv references unknown wine_id {row['wine_id']}")
        items.append(TrainItem(
            image_id=row['image_id'], wine_id=row['wine_id'], path=row['path'],
            view=row['view'], producer=wine['producer'], split=wine['producer_split'],
            name_ru=wine['name_ru'], color=wine['color'], vintage=wine['vintage'],
            usable_train=int(row.get('usable_train', 0))))
    by_split: dict[str, list[TrainItem]] = {'train': [], 'validation': [], 'test': []}
    for item in items:
        if item.split not in by_split:
            raise ValueError(f"invalid producer_split {item.split}")
        by_split[item.split].append(item)
    return by_split


def contrastive_train_items(items: list[TrainItem]) -> list[TrainItem]:
    """ТЗ §6.4: single-frame wines are excluded from contrastive training."""
    frames: dict[str, list[TrainItem]] = defaultdict(list)
    for item in items:
        if item.usable_train:
            frames[item.wine_id].append(item)
    out = []
    for wine_items in frames.values():
        if len({i.path for i in wine_items}) < MIN_FRAMES_PER_WINE:
            continue
        if not any(i.view != 'studio_front' for i in wine_items):
            continue
        out.extend(wine_items)
    return out


def info_nce_loss(image_logits, text_logits, temperature: float,
                  producers) -> tuple[float, dict]:
    """Symmetric InfoNCE with same-producer hard-negative weighting (ТЗ §6.3).

    Pure numpy reference used by tests and by the torch training step.
    image_logits/text_logits are pre-tempered (unnormalized logits matrix).
    """
    logits = image_logits / temperature
    n = logits.shape[0]
    weights = np.ones_like(logits)
    for i in range(n):
        for j in range(n):
            if i != j and producers[i] and producers[i] == producers[j]:
                weights[i, j] = 2.0

    def ce_row(row, target):
        scaled = row * weights[target]
        scaled[target] = row[target]
        max_ = scaled.max()
        exp = np.exp(scaled - max_)
        probs = exp / exp.sum()
        return -math.log(max(probs[target], 1e-12))

    loss_i = sum(ce_row(logits[i], i) for i in range(n)) / n
    loss_t = sum(ce_row(logits[:, i], i) for i in range(n)) / n
    return 0.5 * (loss_i + loss_t), {'rows': loss_i, 'cols': loss_t}


def build_text_index(items: list[TrainItem]) -> dict[str, str]:
    """One canonical text per wine_id (ТЗ §2.3); identical for all frames."""
    return {i.wine_id: canonical_text(name_ru=i.name_ru, producer=i.producer,
                                      color=i.color or None, vintage=i.vintage or None)
            for i in items}


def recall_at_1(embeddings, text_matrix, targets) -> dict:
    """Top-1 accuracy of L2-normalized image embeddings vs text matrix."""
    scores = embeddings @ text_matrix.T
    top = np.argmax(scores, axis=1)
    correct = sum(1 for i, t in enumerate(targets) if top[i] == t)
    return {'numerator': correct, 'denominator': len(targets),
            'value': correct / len(targets) if targets else None}


# ---------------------------------------------------------------------------
# torch training path (imported lazily; weights downloaded only on run)
# ---------------------------------------------------------------------------

# Supported vision towers: HF CLIP keeps blocks in ``encoder.layers`` with
# ``self_attn.{q,k,v,out}_proj``; jina-clip-v2 is an EVA02 ViT with blocks in
# ``blocks`` and ``attn.{q,k,v}_proj`` + output ``attn.proj``.
VISION_BLOCK_PATHS = (('encoder', 'layers'), ('blocks',))
ATTENTION_ATTRS = ('self_attn', 'attn')
DEFAULT_PROJECTIONS = ('q_proj', 'k_proj', 'v_proj', 'out_proj', 'proj')


def _vision_blocks(model):
    """Return (dotted path, block list) of the vision transformer blocks."""
    vision = getattr(model, 'vision_model', None)
    for path in VISION_BLOCK_PATHS:
        node = vision
        for attr in path:
            node = getattr(node, attr, None)
        if node is not None:
            return 'vision_model.' + '.'.join(path), node
    raise RuntimeError('unexpected vision architecture: no vision_model.encoder.layers '
                       'or vision_model.blocks; provide explicit modules')


def _attention(layer):
    """Return (attribute name, module) holding the attention projections."""
    for attr in ATTENTION_ATTRS:
        module = getattr(layer, attr, None)
        if module is not None:
            return attr, module
    return None, layer


def _vision_projection(model):
    """Trainable vision projection: jina ``visual_projection`` or HF-style head."""
    from torch import nn
    for module in (getattr(model, 'visual_projection', None),
                   getattr(getattr(model, 'vision_model', None), 'head', None)):
        if isinstance(module, nn.Linear):
            return module
    return None


def inspect_vision_modules(model, projection_map=DEFAULT_PROJECTIONS) -> dict:
    """JSON-ready report of the vision tower structure for the LoRA mapping.

    Smoke check (review P1-5) before training: per block, the attention
    container and its Linear children with (in, out) shapes, plus which
    projections ``attach_lora`` would wrap.
    """
    from torch import nn
    path, layers = _vision_blocks(model)
    blocks = {}
    for index, layer in enumerate(layers):
        name, container = _attention(layer)
        linear = {child: [module.in_features, module.out_features]
                  for child, module in container.named_children()
                  if isinstance(module, nn.Linear)}
        blocks[index] = {'attention': name, 'linear': linear,
                         'lora_targets': [a for a in projection_map if a in linear]}
    projection = _vision_projection(model)
    return {'blocks_path': path, 'num_blocks': len(layers), 'blocks': blocks,
            'vision_projection': (list(projection.weight.shape)
                                  if projection is not None else None)}


def attach_lora(model, *, lora_blocks=4, rank=8, alpha=16, dropout=0.05,
                projection_map=None):
    """Wrap Q,K,V and output projection of the last vision blocks with LoRA.

    Returns the list of patched parameter names. Blocks are found in
    ``vision_model.encoder.layers`` (HF CLIP) or ``vision_model.blocks`` (EVA /
    jina-clip-v2); projections (default ``q_proj/k_proj/v_proj/out_proj/proj``)
    are looked up on ``self_attn``/``attn`` or on the block itself.

    The wrapper also exposes the merged ``weight``: EVA attention calls
    ``F.linear(x, self.q_proj.weight, self.q_bias)`` instead of the module, so
    the LoRA delta must live in ``weight`` too. On that path LoRA dropout is not
    applied (it acts on the input in ``forward`` only).
    """
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
            super().__init__()
            self.base = base
            self.base.requires_grad_(False)
            self.r = r
            self.scale = alpha / r
            options = {'device': base.weight.device, 'dtype': base.weight.dtype}
            self.lora_a = nn.Parameter(torch.zeros(base.in_features, r, **options))
            self.lora_b = nn.Parameter(torch.zeros(r, base.out_features, **options))
            self.dropout = nn.Dropout(dropout)
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        @property
        def weight(self):
            return self.base.weight + (self.lora_a @ self.lora_b).T * self.scale

        @property
        def bias(self):
            return self.base.bias

        def forward(self, x):
            return self.base(x) + self.dropout(x) @ self.lora_a @ self.lora_b * self.scale

    _, layers = _vision_blocks(model)
    if projection_map is None:
        projection_map = DEFAULT_PROJECTIONS
    patched = []
    for layer in layers[-lora_blocks:]:
        name, container = _attention(layer)
        for attr in projection_map:
            module = getattr(container, attr, None)
            if isinstance(module, nn.Linear):
                setattr(container, attr, LoRALinear(module, rank, alpha, dropout))
                patched.append(f'{name or "layer"}.{attr}')
    if not patched:
        structure = inspect_vision_modules(model, projection_map)
        raise RuntimeError(
            'no Q/K/V/out projections found in vision blocks; '
            'run --verify-only / inspect_vision_modules() to inspect the '
            f'actual structure ({structure})')
    return patched


def smoke_backward(model, *, lora_blocks=4, rank=8, alpha=16, dropout=0.05,
                   image_size=None) -> dict:
    """Attach LoRA, run one forward/backward on a blank image and check grads.

    ``lora_b`` starts at zero, so its gradient is the one that must be nonzero
    in every wrapped projection; the frozen weights must receive none. Mutates
    ``model`` (LoRA stays attached): use a throwaway model instance.
    """
    from PIL import Image
    patched = attach_lora(model, lora_blocks=lora_blocks, rank=rank, alpha=alpha,
                          dropout=dropout)
    projection = _vision_projection(model)
    if projection is not None:
        projection.requires_grad_(True)
    size = image_size or getattr(model.vision_model, 'image_size', 512)
    size = size[0] if isinstance(size, (tuple, list)) else size
    pixels = model.preprocess([Image.new('RGB', (size, size), 'white')]).to(model.device)
    model.train()
    emb = model.get_image_features(pixels)
    emb.float().pow(2).sum().backward()
    lora_b = {name: float(p.grad.abs().sum()) if p.grad is not None else 0.0
              for name, p in model.named_parameters() if name.endswith('lora_b')}
    frozen_with_grad = [name for name, p in model.named_parameters()
                        if p.grad is not None and not _trainable(name)]
    model.zero_grad(set_to_none=True)
    model.eval()
    return {'patched_per_block': patched[:len(patched) // lora_blocks],
            'lora_modules': len(lora_b),
            'lora_b_without_grad': [n for n, g in lora_b.items() if g == 0.0],
            'frozen_params_with_grad': frozen_with_grad,
            'projection_trainable': projection is not None,
            'ok': bool(lora_b) and all(lora_b.values()) and not frozen_with_grad}


def save_checkpoint(path, state, *, lora_blocks, rank, alpha, dropout, revision):
    """Persist a trained adapter as {"meta": ..., "state": ...}.

    ``state`` holds the trainable LoRA params, the vision projection and the
    learned ``temperature``; ``meta`` records the hyperparameters needed by
    ``load_adapter`` to re-attach the LoRA layers.
    """
    import torch
    payload = {'meta': {'lora_blocks': lora_blocks, 'rank': rank, 'alpha': alpha,
                        'dropout': dropout, 'revision': revision},
               'state': state}
    torch.save(payload, path)


def load_adapter(model, checkpoint_path, *, device=None) -> dict:
    """Load a trained vision-LoRA adapter into ``model`` for inference.

    Re-attaches the LoRA layers with the hyperparameters recorded in the
    checkpoint, applies the LoRA/vision-projection weights and returns the
    checkpoint metadata (including the learned ``temperature``). Inference
    after training::

        encoder = JinaCLIPEmbedder(revision=..., device=..., allow_remote_code=True)
        meta = load_adapter(encoder.model, 'adapter.pt', device=encoder.device)
        # encoder.model now runs with the adapted vision tower.

    Raises ``ValueError`` for malformed checkpoints or unknown keys.
    """
    import torch
    payload = torch.load(checkpoint_path, map_location=device or 'cpu', weights_only=True)
    if not isinstance(payload, dict) or 'meta' not in payload or 'state' not in payload:
        raise ValueError('invalid adapter checkpoint: expected {"meta": ..., "state": ...}')
    meta = dict(payload['meta'])
    attach_lora(model, lora_blocks=meta['lora_blocks'], rank=meta['rank'],
                alpha=meta['alpha'], dropout=meta['dropout'])
    _, unexpected = model.load_state_dict(payload['state'], strict=False)
    unexpected = [k for k in unexpected if k != 'temperature']
    if unexpected:
        raise ValueError(f'adapter checkpoint contains unknown keys: {unexpected[:5]}')
    meta['temperature'] = payload['state'].get('temperature')
    return meta


def load_image_bytes(root: Path, path: str):
    from .roi import decode, InvalidImage, MAX_BYTES
    import time
    source = root / path
    if source.stat().st_size > MAX_BYTES:
        # Enforce the byte limit before reading the file into memory (P2-15).
        raise InvalidImage('invalid_size')
    with source.open('rb') as f:
        data = f.read()
    image, _ = decode(data, time.monotonic() + 60)
    return image


def train(*, wines_csv, images_csv, revision=REVISION, device=None, epochs=8,
          batch_size=64, lora_blocks=4, rank=8, alpha=16, dropout=0.05,
          lr_lora=1e-4, lr_proj=5e-5, weight_decay=0.01, warmup_fraction=0.05,
          allow_remote_code=False, out=None, max_epochs_without_gain=2,
          images_root=None):
    """Run the gated vision-LoRA fine-tune; returns the report dict.

    Text embeddings of the canonical strings are computed once with the frozen
    text tower; only the vision path trains. ``out`` receives an adapter
    checkpoint ({"meta": ..., "state": ...}) loadable via ``load_adapter``.
    ``images_root`` is the uploads root that images.csv ``path`` entries are
    relative to (catalog_export contract); defaults to the images.csv directory.
    """
    import torch

    if not allow_remote_code:
        raise ValueError('review CC BY-NC 4.0 and HF remote code first')
    root = Path(images_root).resolve() if images_root else Path(images_csv).resolve().parent
    splits = load_datasets(wines_csv, images_csv)
    train_items = contrastive_train_items(splits['train'])
    val_items = splits['validation']
    train_wines = {i.wine_id for i in train_items}
    if len(train_wines) < MIN_TRAIN_LABELS:
        return {'gate': 'train_labels_below_minimum', 'train_wines': len(train_wines),
                'minimum': MIN_TRAIN_LABELS,
                'note': 'keep the frozen zero-shot index; no fine-tune'}
    text_index = build_text_index(train_items + val_items)

    from .jina_clip import JinaCLIPEmbedder
    encoder = JinaCLIPEmbedder(revision=revision, device=device,
                               batch_size=batch_size, allow_remote_code=True)
    model = encoder.model
    model.requires_grad_(False)
    model.eval()

    texts = sorted({t for t in text_index.values()})
    with torch.inference_mode():
        text_vectors = encoder.encode_text(texts)
    text_matrix = torch.tensor(text_vectors, device=model.device)
    text_row = {text: idx for idx, text in enumerate(texts)}

    # Zero-shot baseline on validation (frozen model, no LoRA).
    baseline = evaluate_recall(model, encoder, root, val_items, text_index,
                               text_matrix, text_row, batch_size)

    patched = attach_lora(model, lora_blocks=lora_blocks, rank=rank,
                          alpha=alpha, dropout=dropout)
    temperature = torch.nn.Parameter(torch.tensor(0.07, device=model.device))
    lora_params = [p for p in model.parameters() if p.requires_grad]
    # model.requires_grad_(False) froze the projection too; unfreeze it here so
    # it trains in its own optimizer group (not a LoRA layer).
    projection = _vision_projection(model)
    proj_params = ([p for p in projection.parameters()] if projection is not None else [])
    for p in proj_params:
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [{'params': lora_params, 'lr': lr_lora},
         {'params': [temperature] + proj_params, 'lr': lr_proj}],
        weight_decay=weight_decay)
    total_steps = math.ceil(len(train_items) / batch_size) * epochs
    warmup_steps = max(1, int(total_steps * warmup_fraction))

    def lr_at(step):
        return min(1.0, (step + 1) / warmup_steps) if step < warmup_steps else 1.0

    best_val, best_epoch, best_state = None, 0, None
    history = []
    steps = 0
    for epoch in range(1, epochs + 1):
        rng = np.random.default_rng(epoch)
        perm = rng.permutation(len(train_items))
        model.train()
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(perm), batch_size):
            lora_group, proj_group = optimizer.param_groups
            lora_group['lr'] = lr_lora * lr_at(steps)
            proj_group['lr'] = lr_proj * lr_at(steps)
            idx = perm[start:start + batch_size]
            batch = [train_items[i] for i in idx]
            images = [load_image_bytes(root, i.path) for i in batch]
            try:
                logits = encode_images_train(model, encoder, images)
            finally:
                for image in images:
                    image.close()
            producers = [i.producer for i in batch]
            rows = [text_row[text_index[i.wine_id]] for i in batch]
            logits_t = logits @ text_matrix[rows].T
            loss = clip_loss(logits_t, temperature, producers)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
            steps += 1
        model.eval()
        val = evaluate_recall(model, encoder, root, val_items, text_index,
                              text_matrix, text_row, batch_size)
        history.append({'epoch': epoch, 'val_recall_at_1': val['value'],
                        'loss': epoch_loss / max(1, batches)})
        if best_val is None or val['value'] > best_val:
            best_val, best_epoch = val['value'], epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items() if _trainable(k)}
            best_state['temperature'] = temperature.detach().cpu().clone()
        # ТЗ §6.1: after three epochs, drop the adapter if val is not better
        # than the frozen index.
        if epoch >= 3 and best_val is not None and baseline['value'] is not None \
                and best_val <= baseline['value']:
            return {'gate': 'no_improvement_over_frozen', 'baseline': baseline,
                    'history': history, 'note': 'adapter discarded; frozen index wins'}
        if epoch - best_epoch >= max_epochs_without_gain and best_epoch > 0:
            break
    if out and best_state:
        save_checkpoint(out, best_state, lora_blocks=lora_blocks, rank=rank,
                        alpha=alpha, dropout=dropout, revision=revision)
    return {'gate': 'trained', 'baseline': baseline, 'best_val': best_val,
            'history': history, 'epochs_run': epoch,
            'patched': patched, 'state_path': out}


def _trainable(key: str) -> bool:
    return ('lora' in key or 'vision_model.head' in key or 'visual_projection' in key
            or key == 'temperature')


def encode_images_train(model, encoder, images):
    """L2-normalized image embeddings with autograd (same preprocessing as inference).

    Upstream ``encode_image`` forces ``eval()`` and returns detached outputs, so
    the pinned processor and ``get_image_features`` are called directly; grad
    mode is left to the caller (training loop vs ``no_grad`` evaluation).
    """
    import torch
    pixels = encoder.model.preprocess([image.convert('RGB') for image in images])
    emb = model.get_image_features(pixels.to(model.device))
    return torch.nn.functional.normalize(emb, dim=-1)


def clip_loss(logits_t, temperature, producers):
    """Symmetric InfoNCE with same-producer hard-negative weighting (ТЗ §6.3).

    Mirrors the numpy reference ``info_nce_loss``: negative pairs from the same
    producer are weighted 2x before the softmax in both directions (rows for the
    image side, columns for the text side); positive pairs are never reweighted.
    """
    import torch
    logits = (logits_t + logits_t.T) / 2.0
    n = logits.shape[0]
    weights = torch.ones_like(logits)
    for i in range(n):
        for j in range(n):
            if i != j and producers[i] and producers[i] == producers[j]:
                weights[i, j] = 2.0
    targets = torch.arange(n, device=logits.device)
    tempered = logits / temperature
    scaled = tempered * weights
    scaled[targets, targets] = tempered[targets, targets]
    loss_i = torch.nn.functional.cross_entropy(scaled, targets)
    loss_t = torch.nn.functional.cross_entropy(scaled.T, targets)
    return 0.5 * (loss_i + loss_t)


def evaluate_recall(model, encoder, root, items, text_index, text_matrix,
                    text_row, batch_size):
    import torch
    if not items:
        return {'numerator': 0, 'denominator': 0, 'value': None}
    vectors, targets = [], []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        images = [load_image_bytes(root, item.path) for item in batch]
        try:
            with torch.no_grad():
                emb = encode_images_train(model, encoder, images)
        finally:
            for image in images:
                image.close()
        vectors.extend(emb.cpu().numpy())
        targets.extend(text_row[text_index[item.wine_id]] for item in batch)
    embeddings = np.asarray(vectors, dtype='float32')
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return recall_at_1(embeddings, text_matrix.cpu().numpy(), targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wines', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--images-root', default=None,
                        help='uploads root that images.csv "path" entries are relative '
                             'to (catalog_export --uploads); defaults to the images.csv dir')
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--device', default=None)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lora-blocks', type=int, default=4)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--alpha', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--lr-lora', type=float, default=1e-4)
    parser.add_argument('--lr-proj', type=float, default=5e-5)
    parser.add_argument('--allow-remote-code', action='store_true')
    parser.add_argument('--out', default=None)
    parser.add_argument('--verify-only', action='store_true',
                        help='smoke-check the vision LoRA projection mapping and exit '
                             '(requires --allow-remote-code; review P1-5)')
    args = parser.parse_args()
    import json
    if args.verify_only:
        if not args.allow_remote_code:
            parser.error('--verify-only requires --allow-remote-code '
                         '(reviews the HF remote code first)')
        from .jina_clip import JinaCLIPEmbedder
        encoder = JinaCLIPEmbedder(revision=args.revision, device=args.device,
                                   batch_size=args.batch_size, allow_remote_code=True)
        structure = inspect_vision_modules(encoder.model)
        smoke = smoke_backward(encoder.model, lora_blocks=args.lora_blocks, rank=args.rank,
                               alpha=args.alpha, dropout=args.dropout)
        blocks = structure.pop('blocks')
        print(json.dumps({'revision': args.revision, 'device': encoder.device,
                          'vision_structure': structure,
                          'last_block': blocks[len(blocks) - 1],
                          'smoke_backward': smoke},
                         ensure_ascii=False, indent=2))
        if not smoke['ok']:
            raise SystemExit(1)
        return
    report = train(wines_csv=args.wines, images_csv=args.images,
                   images_root=args.images_root, revision=args.revision,
                   device=args.device, epochs=args.epochs,
                   batch_size=args.batch_size, lora_blocks=args.lora_blocks,
                   rank=args.rank, alpha=args.alpha, dropout=args.dropout,
                   lr_lora=args.lr_lora, lr_proj=args.lr_proj,
                   allow_remote_code=args.allow_remote_code, out=args.out)
    print(json.dumps({k: v for k, v in report.items() if k != 'patched'},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()