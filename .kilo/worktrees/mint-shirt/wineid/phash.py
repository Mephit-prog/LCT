"""Perceptual hash (pHash) for near-duplicate detection (tz-wine-label-retrieval.md §4.1).

A pHash is a 64-bit DCT-based fingerprint of a grayscale image. Hamming distance
below :data:`NEAR_DUPLICATE_DISTANCE` means visually near-identical frames. It is
used only to **drop both** cross-wine candidate links (ТЗ §4.1: never pick a
"best" frame), and is never evidence of a verified media relation. Hamming bands
(8–16) are also exposed for the hard-pair recall slice of ТЗ §8.

No extra dependency is used: the 2-D DCT is built from a cosine matrix with NumPy.
"""
import numpy as np
from PIL import Image

HASH_SIZE = 8
HIGHFREQ_FACTOR = 4
NEAR_DUPLICATE_DISTANCE = 6


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n).reshape(-1, 1)
    m = np.arange(n).reshape(1, -1)
    return np.cos(np.pi * (2 * m + 1) * k / (2 * n))


def phash(image: Image.Image, hash_size: int = HASH_SIZE,
          highfreq_factor: int = HIGHFREQ_FACTOR) -> int:
    """Return a deterministic 64-bit DCT perceptual hash of ``image``.

    The DC coefficient is excluded from the median so flat/low-contrast images do
    not degenerate into an all-ones hash.
    """
    size = hash_size * highfreq_factor
    gray = image.convert('L').resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype='float64')
    matrix = _dct_matrix(size)
    dct = matrix @ pixels @ matrix.T
    low = dct[:hash_size, :hash_size]
    # Round tiny numerical noise to zero so flat/low-contrast frames hash
    # deterministically instead of depending on floating-point residue.
    low = np.round(low, 6)
    median = np.median(low.flatten()[1:])
    value = 0
    for bit in (low > median).flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    """Bit distance between two hashes."""
    return int(a ^ b).bit_count()


def near_duplicate_indices(hashes, owners, max_distance: int = NEAR_DUPLICATE_DISTANCE):
    """Indices to drop: both sides of every cross-owner pair closer than ``max_distance``.

    ``hashes`` and ``owners`` are parallel sequences; ``None`` hashes never match.
    Frames of the same owner are allowed to be near-duplicates (multi-frame SKU).
    """
    drop = set()
    for i, left in enumerate(hashes):
        if left is None:
            continue
        for j in range(i + 1, len(hashes)):
            right = hashes[j]
            if right is None or owners[i] == owners[j]:
                continue
            if hamming(left, right) < max_distance:
                drop.add(i)
                drop.add(j)
    return drop


def distance_band_pairs(hashes, owners, low: int = 8, high: int = 16):
    """Cross-owner index pairs whose Hamming distance lies in ``[low, high]``.

    Supports the ТЗ §8 hard-pair recall slice (visually similar but distinct
    labels); it is a diagnostic list, not a split assignment.
    """
    pairs = []
    for i, left in enumerate(hashes):
        if left is None:
            continue
        for j in range(i + 1, len(hashes)):
            right = hashes[j]
            if right is None or owners[i] == owners[j]:
                continue
            distance = hamming(left, right)
            if low <= distance <= high:
                pairs.append((i, j, distance))
    return pairs
