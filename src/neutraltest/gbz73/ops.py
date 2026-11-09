"""The operator column of the matrix, expressed with plain ``torch``.

Five families, one per column: a grouped GEMM, a normalisation, two
activations and a reduction. Every one of them

* takes float tensors and returns a float tensor, so the FP8 rounding can be
  applied on either side of the call;
* runs in float32 (the ordinary baseline) and float64 (the golden baseline)
  without special-casing, because it is written with pure ``torch`` ops;
* is differentiable, which is what the gradient feature needs.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

Args = tuple[Any, ...]
Kwargs = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Op:
    """One operator of the matrix."""

    name: str
    family: str
    description: str
    variants: tuple[str, ...]
    sample: Callable[[str, torch.dtype, torch.Generator], tuple[Args, Kwargs]]
    forward: Callable[..., torch.Tensor]
    discrete: bool = False  # output is a selection: byte equality is meaningful
    #: True when the operator does communication-shaped work (a grouped GEMM over
    #: variable segment lengths), so its cost is not a pure function of the shape.
    communicated: bool = False
    differentiable: bool = True
    tags: frozenset[str] = field(default_factory=frozenset)

    def make(self, variant: str, dtype: torch.dtype, seed: int = 0) -> tuple[Args, Kwargs]:
        """Deterministic inputs for ``variant`` in ``dtype``."""
        if variant not in self.variants:
            raise ValueError(f"{self.name}: unknown variant {variant!r}")
        generator = torch.Generator().manual_seed(seed)
        args, kwargs = self.sample(variant, dtype, generator)
        return args, kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.forward(*args, **kwargs)


def _grouped_gemm_sample(
    variant: str, dtype: torch.dtype, generator: torch.Generator
) -> tuple[Args, Kwargs]:
    sizes = [16, 16, 16, 16] if variant == "g4" else [16, 4, 16, 1]
    groups, k, n = len(sizes), 24, 16
    total = sum(sizes)
    a = torch.randn(total, k, generator=generator).to(dtype) * 0.5
    b = torch.randn(groups, k, n, generator=generator).to(dtype) * 0.5
    offsets = torch.tensor([0, *itertools.accumulate(sizes)], dtype=torch.int32)
    return (a, b, offsets), {}


def _validate_groups(a: torch.Tensor, b: torch.Tensor, group_list: torch.Tensor) -> list[int]:
    """Reject a group list that cannot describe this call.

    The reference is also the specification, so it validates the same things a
    kernel must: that the offsets start at zero, never decrease, end at the rows
    available, and that the contraction dimensions agree.
    """
    if group_list.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"group_list must be int32 or int64, not {group_list.dtype}")
    if group_list.dim() != 1 or group_list.numel() != b.shape[0] + 1:
        raise ValueError(
            f"group_list needs {b.shape[0] + 1} entries for {b.shape[0]} groups, got {group_list.numel()}"
        )
    if a.dim() != 2 or b.dim() != 3:
        raise ValueError(
            f"expected a 2-D left side and a 3-D right side, got {a.shape} and {b.shape}"
        )
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"contraction dimension mismatch: {a.shape[1]} vs {b.shape[1]}")
    bounds = [int(v) for v in group_list.tolist()]
    if bounds[0] != 0:
        raise ValueError(f"group_list must start at 0, not {bounds[0]}")
    if any(high < low for low, high in zip(bounds, bounds[1:], strict=False)):
        raise ValueError(f"group_list must be non-decreasing, got {bounds}")
    if bounds[-1] != a.shape[0]:
        raise ValueError(f"group_list ends at {bounds[-1]} but the left side has {a.shape[0]} rows")
    return bounds


def _grouped_gemm_forward(
    a: torch.Tensor, b: torch.Tensor, group_list: torch.Tensor
) -> torch.Tensor:
    bounds = _validate_groups(a, b, group_list)
    pieces = []
    for group in range(b.shape[0]):
        low, high = bounds[group], bounds[group + 1]
        if high > low:
            pieces.append(a[low:high] @ b[group])
    if not pieces:
        return a.new_zeros((0, b.shape[-1]))
    return torch.cat(pieces, dim=0)


def _layer_norm_sample(
    variant: str, dtype: torch.dtype, generator: torch.Generator
) -> tuple[Args, Kwargs]:
    rows, width = (8, 64) if variant == "d64" else (4, 128)
    x = (torch.randn(rows, width, generator=generator) * 2.0).to(dtype)
    weight = (torch.randn(width, generator=generator) * 0.1 + 1.0).to(dtype)
    bias = (torch.randn(width, generator=generator) * 0.1).to(dtype)
    return (x, weight, bias), {"eps": 1e-5}


def _layer_norm_forward(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def _gelu_sample(
    variant: str, dtype: torch.dtype, generator: torch.Generator
) -> tuple[Args, Kwargs]:
    x = (torch.randn(16, 32, generator=generator) * 1.5).to(dtype)
    return (x,), {"approximate": "tanh" if variant == "tanh" else "none"}


def _gelu_forward(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    return F.gelu(x, approximate=approximate)


def _relu_sample(
    variant: str, dtype: torch.dtype, generator: torch.Generator
) -> tuple[Args, Kwargs]:
    x = (torch.randn(16, 32, generator=generator) * 3.0).to(dtype)
    return (x,), {}


def _relu_forward(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


def _amax_sample(
    variant: str, dtype: torch.dtype, generator: torch.Generator
) -> tuple[Args, Kwargs]:
    x = (torch.randn(16, 32, generator=generator) * 4.0).to(dtype)
    return (x,), {"dim": -1}


def _amax_forward(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.amax(x.abs(), dim=dim)


OPS: tuple[Op, ...] = (
    Op(
        name="grouped_gemm",
        family="gemm",
        description="grouped matrix multiplication with a group offset list",
        variants=("g4", "g4-tail"),
        sample=_grouped_gemm_sample,
        forward=_grouped_gemm_forward,
        communicated=True,
        tags=frozenset({"reduction"}),
    ),
    Op(
        name="layer_norm",
        family="normalisation",
        description="layer normalisation over the last dimension",
        variants=("d64", "d128"),
        sample=_layer_norm_sample,
        forward=_layer_norm_forward,
        tags=frozenset({"reduction", "transcendental"}),
    ),
    Op(
        name="gelu",
        family="activation",
        description="Gaussian error linear unit, exact and tanh-approximated",
        variants=("exact", "tanh"),
        sample=_gelu_sample,
        forward=_gelu_forward,
        tags=frozenset({"transcendental"}),
    ),
    Op(
        name="relu",
        family="activation",
        description="rectified linear unit",
        variants=("plain",),
        sample=_relu_sample,
        forward=_relu_forward,
        discrete=True,
        communicated=False,
        tags=frozenset(),
    ),
    Op(
        name="amax",
        family="reduction",
        description="maximum absolute value along the last dimension",
        variants=("row",),
        sample=_amax_sample,
        forward=_amax_forward,
        communicated=False,
        tags=frozenset({"reduction"}),
    ),
)

BY_NAME: dict[str, Op] = {op.name: op for op in OPS}


def get(name: str) -> Op:
    """Look an operator up by the name used in the matrix columns."""
    try:
        return BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown operator {name!r}; known: {', '.join(BY_NAME)}") from None
