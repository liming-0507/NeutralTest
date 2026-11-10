"""The fixed-shape kernels a feature is free to call between quantisation passes.

The operators of the matrix are built from pure ``torch`` so that they have a
float64 baseline; the kernels below are the parts an implemented backend is
expected to ship. They are not reference *implementations* -- the float64
reference of every column lives in :mod:`neutraltest.gbz73.inference.ops`.

What matters for the tests is the *shape* of the contract:

* positional-only tensors first, keyword options after -- bit-exact alternative
  spellings of the same request are therefore possible, and a backend that
  spells an option differently is a contract mismatch, not a rounding error;
* two optional tensors, ``index`` and ``mask``, which are how the slot write and
  the probability-masking are addressed without inventing a second signature;
* float32 in, float32 out, except where a column documents otherwise (the cache
  write and the cache read take and return the cache dtype).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

Args = tuple[Any, ...]
Kwargs = dict[str, Any]

#: Options each kernel accepts, for the argument-contract feature. A name that
#: is not here is a deliberate rejection, not an omission.
OPTIONS: dict[str, tuple[str, ...]] = {
    "add": ("alpha",),
    "residual_add": ("alpha",),
    "flash_attention": ("is_causal", "softmax_scale"),
    "kv_cache_write": ("quantise",),
    "kv_cache_read": ("quantise",),
    "softmax": ("dim",),
    "rope": ("interleaved",),
    "rms_norm": ("eps",),
}


def _alpha(alpha: float | None) -> float:
    return 1.0 if alpha is None else float(alpha)


def add(a: torch.Tensor, b: torch.Tensor, *, alpha: float | None = None) -> torch.Tensor:
    """Elementwise residual add: ``a + alpha * b``, broadcast like ``torch.add``."""
    return a + b * _alpha(alpha)


def residual_add(a: torch.Tensor, b: torch.Tensor, *, alpha: float | None = None) -> torch.Tensor:
    """Alias of :func:`add`, kept because the two names are what deployments use."""
    return add(a, b, alpha=alpha)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention over ``(batch, heads, seq, head_dim)``."""
    scale = softmax_scale
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)


def _groups(q: torch.Tensor, value: torch.Tensor) -> tuple[int, int, int, int, int]:
    if q.dim() != 4 or value.dim() != 4:
        raise ValueError(f"expected two 4-D tensors, got {q.shape} and {value.shape}")
    q_heads, v_heads = q.shape[1], value.shape[1]
    if q_heads % v_heads:
        raise ValueError(f"{q_heads} query heads cannot be split over {v_heads} value heads")
    return q.shape[0], v_heads, q_heads // v_heads, value.shape[2], value.shape[3]


def merged_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Grouped-query attention: ``k``/``v`` carry fewer heads than ``q``."""
    batch, v_heads, repeats, seq, head_dim = _groups(q, k)
    left = v.shape[1] if v is not k else k.shape[1]
    if left != v_heads:
        raise ValueError(f"k and v disagree on the head count: {k.shape[1]} vs {v.shape[1]}")
    kk = k.repeat_interleave(repeats, dim=1)
    vv = v.repeat_interleave(repeats, dim=1)
    scale = 1.0 / math.sqrt(head_dim)
    scores = (q @ kk.transpose(-1, -2)) * scale
    return scores.softmax(-1) @ vv


def masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-head attention under an additive mask ``mask`` broadcastable to the scores."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-1, -2)) * scale
    if mask is not None:
        scores = scores + mask
    return scores.softmax(-1) @ v


def kv_cache_write(
    k: torch.Tensor,
    v: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    index: torch.Tensor,
    *,
    quantise: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write the ``t`` new key/value tokens into the cache rows named by ``index``.

    Returns the whole cache: a slot write is observable only by its effect on the
    buffer it wrote into, so the buffer is the result.
    """
    slot = _slot_index(index, cache_k.shape[0])
    if k.ndim != 2 or cache_k.ndim != 2 or cache_k.shape[1] != k.shape[1]:
        raise ValueError(f"cannot write {k.shape} into a cache of {cache_k.shape}")
    out_k = cache_k.clone()
    out_v = cache_v.clone()
    out_k[slot] = k
    out_v[slot] = v
    return out_k, out_v


def kv_cache_read(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    index: torch.Tensor,
    *,
    quantise: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the cache rows named by ``index``, in the order they were named."""
    slot = _slot_index(index, cache_k.shape[0])
    return cache_k[slot], cache_v[slot]


def _slot_index(index: torch.Tensor, rows: int) -> torch.Tensor:
    if index.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"index must be int32 or int64, not {index.dtype}")
    if index.dim() != 1:
        raise ValueError(f"index must be 1-D, got shape {tuple(index.shape)}")
    if index.numel() and (int(index.min()) < 0 or int(index.max()) >= rows):
        raise ValueError(f"index out of range for a cache of {rows} rows: {index.tolist()}")
    return index.to(torch.long)


def softmax(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax along ``dim``."""
    return torch.softmax(x, dim=dim)


def rope(x: torch.Tensor, position: torch.Tensor, *, interleaved: bool = True) -> torch.Tensor:
    """Rotary position embedding on the last axis.

    The kernel is written the way an accelerator would write it -- pair the two
    lanes of an angle, then apply the four coefficients of that one angle --
    rather than by laying two full-width tables over the whole tensor. That
    keeps it honest against the reference: both are now the same rotation, and
    the rotation's norm preservation is visible in this code alone.

    ``interleaved`` selects the pairing: adjacent lanes ``(2i, 2i+1)`` when true,
    the two halves of the axis when false.
    """
    if x.shape[-1] % 2:
        raise ValueError(f"the rotated axis must be even, got {x.shape[-1]}")
    # A 2-D position carries one position per sequence slot: (batch, seq).
    if position.dim() != 1 and not (
        position.dim() == 2 and position.shape[0] == x.shape[0] and position.shape[1] == x.shape[1]
    ):
        raise ValueError(f"position {tuple(position.shape)} does not match input {tuple(x.shape)}")

    width = x.shape[-1]
    half = width // 2
    inv = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float64) / half)
    angles = position.to(torch.float64)[..., None] * inv
    cos = angles.cos().to(x.dtype)
    sin = angles.sin().to(x.dtype)
    # (batch, seq, 1, half) or (1, seq, 1, half): heads broadcast, sequence stays.
    if cos.dim() == 2:
        cos, sin = cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2)
    else:
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)

    if interleaved:
        a, b = x[..., ::2], x[..., 1::2]
    else:
        a, b = x[..., :half], x[..., half:]
    out_a = a * cos - b * sin
    out_b = a * sin + b * cos
    if interleaved:
        return torch.stack((out_a, out_b), dim=-1).flatten(-2)
    return torch.cat((out_a, out_b), dim=-1)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Root-mean-square normalisation over the last axis, then a per-channel scale."""
    if weight.shape[-1] != x.shape[-1]:
        raise ValueError(f"weight width {weight.shape[-1]} does not match {x.shape[-1]}")
    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(mean_square + eps) * weight


FAMILIES: tuple[str, ...] = (
    "add",
    "flash_attention",
    "kv_cache_write",
    "kv_cache_read",
    "softmax",
    "rope",
    "rms_norm",
)


def for_family(family: str):
    """The kernel implementing ``family`` (``kv_cache`` families share one)."""
    if family not in FAMILIES:
        raise KeyError(f"unknown family {family!r}; known: {', '.join(FAMILIES)}")
    return globals()[family]
