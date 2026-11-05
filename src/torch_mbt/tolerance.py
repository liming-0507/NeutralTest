"""Tolerance tables for cross-backend numeric comparison.

Two ideas drive this module:

1. **Tolerance is a function of dtype and arithmetic depth.**  A ``relu`` on
   ``float16`` is allowed to differ from a ``float64`` reference by roughly a
   ``float16`` ulp; a ``conv2d`` that accumulates thousands of products, or a
   ``softmax`` that chains ``exp``/``sum``/``div``, needs more room.  Encoding
   that as one number per dtype produces either false failures or blind spots.

2. **The reference runs at higher precision than the device.**  Almost every
   vendor backend implements ``float64`` poorly or not at all, so the harness
   computes the golden value on CPU in ``float64`` and compares it against the
   device result in its native dtype.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["Tolerance", "KINDS", "EXACT", "tolerance_for", "describe"]


@dataclass(frozen=True)
class Tolerance:
    """Relative/absolute tolerance pair, in the ``torch.testing`` sense."""

    rtol: float
    atol: float

    @property
    def is_exact(self) -> bool:
        """True when the comparison must be bit-exact."""
        return self.rtol == 0.0 and self.atol == 0.0

    def scaled(self, factor: float) -> Tolerance:
        return Tolerance(self.rtol * factor, self.atol * factor)

    def as_kwargs(self) -> dict[str, float]:
        return {"rtol": self.rtol, "atol": self.atol}

    def __str__(self) -> str:
        if self.is_exact:
            return "exact"
        return f"rtol={self.rtol:.3g} atol={self.atol:.3g}"


EXACT = Tolerance(0.0, 0.0)

#: Per-dtype baseline.  Values follow what PyTorch's own operator test suites
#: use for a single elementwise operation on CPU.
_BASE: dict[torch.dtype, tuple[float, float]] = {
    torch.float64: (1e-12, 1e-12),
    torch.float32: (1.3e-6, 1e-5),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1.6e-2, 1e-2),
    torch.complex128: (1e-12, 1e-12),
    torch.complex64: (1.3e-6, 1e-5),
}

#: Multipliers on top of the dtype baseline, by arithmetic class.
#: ``0`` means "must be exact" (pure data movement / indexing).
KIND_SCALE: dict[str, float] = {
    "exact": 0.0,
    "indexing": 0.0,
    "elementwise": 1.0,
    "comparison": 0.0,
    "transcendental": 2.0,
    "reduction": 2.0,
    "normalization": 4.0,
    "loss": 4.0,
    "matmul": 4.0,
    "composite": 8.0,
    "convolution": 8.0,
    "attention": 16.0,
}

KINDS: frozenset[str] = frozenset(KIND_SCALE)


def tolerance_for(
    dtype: torch.dtype,
    kind: str = "elementwise",
    multiplier: float = 1.0,
) -> Tolerance:
    """Return the tolerance for ``dtype`` performing ``kind`` arithmetic.

    Args:
        dtype: dtype of the *device* result being judged.
        kind: arithmetic class, one of :data:`KINDS`.
        multiplier: extra relaxation, normally
            :attr:`torch_mbt.policy.RunPolicy.tolerance_multiplier`.

    Non-floating dtypes (integer, bool) are always compared exactly.
    """
    if not (dtype.is_floating_point or dtype.is_complex):
        return EXACT

    try:
        scale = KIND_SCALE[kind]
    except KeyError:
        raise KeyError(
            f"unknown tolerance kind {kind!r}; expected one of {sorted(KINDS)}"
        ) from None

    if scale == 0.0:
        return EXACT

    base = _BASE.get(dtype)
    if base is None:  # float8 and future dtypes: fail safe on the widest fp32 budget
        base = _BASE[torch.float32]
        scale = max(scale, KIND_SCALE["elementwise"]) * 2.0

    if multiplier <= 0:
        raise ValueError(f"multiplier must be > 0, got {multiplier}")

    return Tolerance(base[0] * scale * multiplier, base[1] * scale * multiplier)


def describe(dtype: torch.dtype) -> str:
    """Human readable one-liner used by ``torch-mbt info``."""
    return ", ".join(f"{kind}: {tolerance_for(dtype, kind)}" for kind in sorted(KINDS))
