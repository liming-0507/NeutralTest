"""FP8 quantisation, dequantisation and the error floor they imply.

Everything here is expressed with public ``torch`` dtypes, so the same code
runs on CPU and on any backend that implements the float8 casts.

Two conventions matter and are applied everywhere:

* a **scale** is a positive float32 tensor; ``quantise`` divides by it and
  ``dequantise`` multiplies by it again, so the scale never leaks into the
  comparison;
* the **error floor** of a format is derived from its own definition (mantissa
  bits + round-to-nearest-even), not from a tuned constant. A correct
  implementation cannot beat that floor, and the tests use it as the baseline
  that separates "FP8 is doing its job" from "the operator is broken".
"""

from __future__ import annotations

import torch

#: Formats this package can construct with plain torch.
FP8_DTYPES: tuple[torch.dtype, ...] = tuple(
    d
    for d in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None))
    if d is not None
)

#: Alias used by the standard's operator clause.
DEFAULT_FP8: torch.dtype = FP8_DTYPES[0]

#: Mantissa bits per format, straight from the FP8 definitions.
_MANTISSA_BITS: dict[torch.dtype, int] = {
    torch.float8_e4m3fn: 3,
    torch.float8_e5m2: 2,
}


def mantissa_bits(dtype: torch.dtype) -> int:
    """Mantissa bits of an FP8 format."""
    try:
        return _MANTISSA_BITS[dtype]
    except KeyError:  # pragma: no cover - guarded by callers
        raise ValueError(f"{dtype} is not an FP8 format this package knows") from None


def half_ulp(dtype: torch.dtype) -> float:
    """Worst-case relative error of one round-to-nearest-even conversion.

    With ``p`` mantissa bits the spacing in ``[1, 2)`` is ``2**-p``, so a value
    rounds to within half of that: ``2**-(p+1)`` relative.
    """
    return 2.0 ** -(mantissa_bits(dtype) + 1)


def max_finite(dtype: torch.dtype) -> float:
    """Largest finite value of a format (448 for E4M3, 57344 for E5M2)."""
    return float(torch.finfo(dtype).max)


def min_subnormal(dtype: torch.dtype) -> float:
    """Smallest positive subnormal, i.e. the underflow boundary."""
    info = torch.finfo(dtype)
    return float(info.tiny) * 2.0 ** -(mantissa_bits(dtype))


def quantise(
    tensor: torch.Tensor, scale: torch.Tensor | float, dtype: torch.dtype = DEFAULT_FP8
) -> torch.Tensor:
    """Float tensor -> FP8 tensor, divided by ``scale`` before the cast."""
    return (tensor / scale).to(dtype)


def dequantise(tensor: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    """FP8 tensor -> float32 tensor, multiplied by ``scale``."""
    return tensor.to(torch.float32) * scale


def roundtrip(
    tensor: torch.Tensor, scale: torch.Tensor | float = 1.0, dtype: torch.dtype = DEFAULT_FP8
) -> torch.Tensor:
    """quantise followed by dequantise: the purest form of the error floor."""
    return dequantise(quantise(tensor, scale, dtype), scale)


def dynamic_scale(
    tensor: torch.Tensor,
    dtype: torch.dtype = DEFAULT_FP8,
    *,
    dim: int | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-tensor (``dim=None``) or per-channel scale that uses the full range."""
    if dim is None:
        amax = tensor.abs().max().clamp_min(eps)
    else:
        amax = tensor.abs().amax(dim=dim, keepdim=True).clamp_min(eps)
    return (amax / max_finite(dtype)).to(torch.float32)


def byte_view(tensor: torch.Tensor) -> torch.Tensor:
    """Raw FP8 bytes as ``uint8``, for the comparisons that must be exact."""
    if tensor.dtype in FP8_DTYPES:
        return tensor.view(torch.uint8)
    return tensor


def elements_are_byte_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """True when two FP8 tensors hold exactly the same bytes."""
    if left.dtype != right.dtype:
        return False
    return bool(torch.equal(byte_view(left), byte_view(right)))
