"""Catalog feature table. Unknown categories are fixed mean embeddings, not observations."""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


def onehot(codes, n, l2=False):
    codes = np.asarray(codes)
    if codes.ndim != 1 or not np.issubdtype(codes.dtype, np.integer) or n < 1 or np.any((codes < -1) | (codes >= n)):
        raise ValueError("invalid categorical codes (expected -1 or 0..n-1)")
    known = codes >= 0
    P = np.zeros((len(codes), n))
    P[np.where(known)[0], codes[known]] = 1.0
    if (~known).any():
        marg = np.bincount(codes[known], minlength=n) / max(known.sum(), 1)
        # If no values are known, use a uniform prior instead of a zero vector.
        P[~known] = marg if known.any() else np.full(n, 1.0 / n)
    if l2:
        P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    return P


def multihot(lists, vocab_size, l2=True):
    if vocab_size < 1:
        raise ValueError("empty vocabulary")
    P = np.zeros((len(lists), vocab_size))
    for i, values in enumerate(lists):
        for g in values:
            if not 0 <= g < vocab_size:
                raise ValueError("unknown category")
            P[i, g] += 1.0
    empty = P.sum(1) == 0
    if empty.any():
        marg = P[~empty].sum(0)
        P[empty] = marg / marg.sum() if marg.sum() else np.full(vocab_size, 1.0 / vocab_size)
    if l2:
        P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    return P


@dataclass
class WineTable:
    cat: List[np.ndarray] = field(default_factory=list)
    cat_names: List[str] = field(default_factory=list)
    ordv: List[np.ndarray] = field(default_factory=list)
    ord_names: List[str] = field(default_factory=list)
    geo: List[np.ndarray] = field(default_factory=list)
    geo_names: List[str] = field(default_factory=list)
    num: Optional[np.ndarray] = None
    num_names: List[str] = field(default_factory=list)
    # Retained for compatibility: missingness is encoded in num, never as pairwise variance.
    num_sx: Optional[np.ndarray] = None
    text: Optional[np.ndarray] = None
    ids: Optional[np.ndarray] = None

    def __len__(self):
        for block in (*self.cat, *self.ordv, *self.geo, self.num, self.text, self.ids):
            if block is not None:
                return len(block)
        return 0

    @property
    def d(self):
        return 0 if self.num is None else self.num.shape[1]

    def validate(self):
        n = len(self)
        if n == 0:
            raise ValueError("catalog is empty")
        if len(self.cat_names) not in (0, len(self.cat)) or len(self.ord_names) not in (0, len(self.ordv)):
            raise ValueError("feature names do not match blocks")
        if self.geo_names and len(self.geo_names) != len(self.geo):
            raise ValueError("geo names do not match blocks")
        for block in (*self.cat, *self.ordv, *self.geo, self.num, self.text):
            if block is not None and (block.ndim != 2 or block.shape[0] != n or
                                      block.shape[1] == 0 or not np.isfinite(block).all()):
                raise ValueError("feature blocks must be nonempty, aligned and finite")
        if self.ids is not None and len(self.ids) != n:
            raise ValueError("ids are not aligned")
        return self
