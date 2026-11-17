"""Shared helpers for the part 7.2 feature scripts.

A part 7.2 cell is a (feature, column) pair.  The column decides what is run --
which family, whether the forward kernel or its vector-Jacobian product is under
test -- and the feature decides what is observed.  Every script in
:mod:`neutraltest.gbz72.items` therefore takes one :class:`InferenceContext` and
calls one of the helpers here, so that "the reference agrees", "the gradient
agrees" and "the scale applies" are each computed one way across the twelve
columns.

Tolerances are not invented per script: :func:`budget` reads the dtype base
tolerance from :mod:`neutraltest.tolerance` and multiplies it by the family and
direction budget the 7.2 context already defines.  The one exception is the
gradient check, which always compares two analytic derivatives and is therefore
held at ``JACOBIAN_RTOL``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ..gbz73 import metrics
from ..gbz73.context import NotApplicable
from ..gbz73.items._support import (
    NOISE_SLACK,
    NotVerifiable,
    measure,
    require_resolvable,
)
from ..tolerance import tolerance_for

#: Analytic derivatives all come from the same float64 path, so the comparison of
#: two of them is a rounding check, not an accuracy check.
JACOBIAN_RTOL = 1e-9

Args = tuple[torch.Tensor, ...]
Kwargs = Mapping[str, Any]

__all__ = [
    "JACOBIAN_RTOL",
    "NOISE_SLACK",
    "NotApplicable",
    "NotVerifiable",
    "budget",
    "check",
    "clone_args",
    "close",
    "flat",
    "forward",
    "jacobian",
    "make_inputs",
    "on_device",
    "measure",
    "observe",
    "reference",
    "require_resolvable",
    "numeric_option",
    "value_slots",
]


def make_inputs(
    *,
    family: str,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    rows: int = 8,
    width: int = 16,
    tokens: int = 4,
    batch: int = 2,
    heads: int = 2,
    seq: int = 8,
    variant: str = "",
    index: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
) -> tuple[Args, dict[str, Any]]:
    """Arguments for one call of ``family``, shaped like the 7.2 reference sample.

    The defaults reproduce the shapes the reference sampler uses; the keyword
    arguments let a script reshape one dimension without rebuilding the rest.
    The generator is seeded from ``seed`` so two calls with the same arguments
    produce the same tensors.
    """
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).to(dtype)

    if family == "add":
        return (randn(rows, width), randn(width)), {"alpha": 2.0}
    if family == "flash_attention":
        return (
            randn(batch, heads, seq, width),
            randn(batch, heads, seq, width),
            randn(batch, heads, seq, width),
        ), {"is_causal": variant == "causal"}
    if family == "kv_cache":
        slots = index
        if slots is None:
            slots = torch.arange(tokens, dtype=torch.int64) % rows
        return (
            randn(tokens, width),
            randn(tokens, width),
            randn(rows, width),
            randn(rows, width),
            slots,
        ), {}
    if family == "softmax":
        return (randn(rows, width),), {"dim": -1}
    if family == "rope":
        x = randn(batch, seq, heads, width)
        position = positions
        if position is None:
            position = (
                torch.arange(seq, dtype=torch.float32).unsqueeze(0).expand(batch, -1).contiguous()
            )
        return (x, position), {"interleaved": variant != "split"}
    if family == "rms_norm":
        scale = weight
        if scale is None:
            scale = randn(width) * 0.1 + 1.0
        return (randn(rows, width), scale), {"eps": 1e-6}
    raise ValueError(f"unknown family {family!r}")


def on_device(context: Any, args: Args, kwargs: Kwargs) -> tuple[Args, Kwargs]:
    """Arguments a script built itself, moved to the device under test.

    ``make_inputs`` always builds on the CPU, which is right for the reference
    and wrong for the kernel once the backend is a device.  Every script that
    reshapes an input, perturbs one element or repeats a call therefore hands
    its arguments through here; leaving that to each script is how the suite
    grows rows that fail on every device for a reason that has nothing to do
    with the thing they measure.
    """
    move = getattr(context, "on_device", None)
    if move is None:
        return args, kwargs
    moved = move(tuple(args))
    return moved, dict(kwargs)


def clone_args(args: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """A detached copy of every tensor in ``args``.

    Every argument this harness builds is a tensor, so there are no ``None``
    slots to preserve; a caller that works on the forward arguments can keep the
    tuple type and mutate the copy.
    """
    return tuple(value.clone() for value in args)


def flat(value: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
    """One float64 vector holding every element of ``value``."""
    items = [value] if isinstance(value, torch.Tensor) else list(value)
    parts = [item.reshape(-1).to(torch.float64) for item in items]
    if not parts:
        return torch.zeros(0, dtype=torch.float64)
    return torch.cat(parts)


def close(
    actual: torch.Tensor | Sequence[torch.Tensor],
    expected: torch.Tensor | Sequence[torch.Tensor],
    *,
    rel: float,
    msg: str,
) -> float:
    """Assert the normalised error is within ``rel``; return the measured error."""
    difference = metrics.compare(flat(actual), flat(expected))
    metrics.assert_within(difference, rel=rel, msg=msg)
    return difference.rel


def budget(context: Any, dtype: torch.dtype = torch.float32, *, kind: str = "elementwise") -> float:
    """The dtype and family budget a result computed in ``dtype`` may spend.

    ``dtype`` is the width the *device* computes in, not the width the reference
    computes in.  The reference is always float64, so a row that asks for a
    float64 result is only entitled to the float64 budget when the backend really
    computes in float64; a backend that widens through float32 has already spent
    the float32 budget before the comparison starts.
    """
    return tolerance_for(
        effective_compute_dtype(context, dtype), kind, context.tolerance_multiplier()
    ).rtol


def effective_compute_dtype(context: Any, dtype: torch.dtype) -> torch.dtype:
    """The narrower of ``dtype`` and what the backend can actually compute in.

    A backend that advertises no float64 arithmetic (``torch.float64 not in
    torch._C._get_privateuse1_backend_name`` backed kernels) still answers a
    float64 request -- it computes in float32 and widens afterwards.  Judging
    that answer against the float64 budget would fail every float64 row on a
    correct backend, so the budget tracks the width the arithmetic really used.

    The downgrade is recorded rather than silent: :func:`note_downgrade` pushes
    the fact onto the column so the report can show it.
    """
    if dtype is not torch.float64:
        return dtype
    if _computes_in_float64(context):
        return dtype
    return torch.float32


def _computes_in_float64(context: Any) -> bool:
    """Whether the backend under test performs float64 arithmetic natively.

    CPU and CUDA do.  A private-use backend does only if it says so through the
    ``float64`` capability probe; absent a probe the answer is "no", because
    telling a backend that widens through float32 that it owes float64 accuracy
    is the failure this function exists to prevent.
    """
    probe = getattr(context, "computes_in_float64", None)
    if probe is not None:
        return bool(probe())
    return True


def note_downgrade(context: Any, asked: torch.dtype, used: torch.dtype) -> None:
    """Record that a row asked for one width and the backend computed in another."""
    if asked is used:
        return
    note = getattr(context, "note", None)
    if note is not None:
        note(
            f"asked for {asked} and the backend computes in {used}; the budget "
            f"follows the arithmetic, not the request"
        )


def as_tuple(value: Any) -> tuple[torch.Tensor, ...]:
    """Normalise a kernel result to a tuple.

    A kernel that returns one tensor is the common case and the private API lets
    it; every script here wants to index and count the outputs, so the boundary
    is made uniform once, in :func:`observe` and :func:`reference`.
    """
    if isinstance(value, tuple):
        return value
    return (value,)


def observe(context: Any, args: Args, kwargs: Kwargs) -> tuple[torch.Tensor, ...]:
    """Run the kernel under test on ``args`` and return its result as a tuple.

    The movement to the device happens here, so a script that built its own
    arguments does not have to know which backend it is running on.
    """
    moved, moved_kwargs = on_device(context, args, kwargs)
    return as_tuple(context.call_with(moved, moved_kwargs))


def reference(
    context: Any, args: Args, kwargs: Kwargs, *, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, ...]:
    """Run the float64 reference for the same call."""
    return as_tuple(context.reference_with(args, kwargs, dtype=dtype))


def forward(
    context: Any,
    args: Args,
    kwargs: Kwargs,
    *,
    rel: float | None = None,
    dtype: torch.dtype = torch.float32,
    msg: str = "",
) -> float:
    """The check almost every row performs: the kernel agrees with the reference.

    ``rel`` defaults to the budget for ``dtype``; a row that forms an output by a
    different but equally exact route (a reordering of a sum, for instance) may
    pass a tighter or looser number and say why in its docstring.
    """
    got = observe(context, args, kwargs)
    want = reference(context, args, kwargs, dtype=dtype)
    used = effective_compute_dtype(context, dtype)
    note_downgrade(context, dtype, used)
    allowed = budget(context, dtype) if rel is None else rel
    return close(got, want, rel=allowed, msg=msg or f"{context.column.name}: forward")


def jacobian(
    context: Any,
    args: Args,
    kwargs: Kwargs,
    *,
    rel: float = JACOBIAN_RTOL,
    msg: str = "",
) -> float:
    """Both derivatives of the same forward, compared at ``rel``.

    The backward columns are the operator's vector-Jacobian product, so the
    oracle is autograd on the same float64 forward.  Nothing is compared against
    a float32 run here: the two sides are exact derivatives of the same function
    and differ only by rounding.
    """
    analytic, shapes = context.gradients_with(args, kwargs, dtype=torch.float64)
    automatic = context.autograd_with(args, kwargs, dtype=torch.float64)
    if len(analytic) != len(automatic):
        raise AssertionError(
            f"{context.column.name}: the reference returned {len(analytic)} derivatives "
            f"but autograd returned {len(automatic)}"
        )
    for index, (got, want) in enumerate(zip(analytic, automatic, strict=True)):
        if got.shape != want.shape:
            raise AssertionError(
                f"{context.column.name}: derivative {index} has shape {tuple(got.shape)}, "
                f"autograd has {tuple(want.shape)}"
            )
        if index < len(shapes) and tuple(shapes[index]) != tuple(got.shape):
            raise AssertionError(
                f"{context.column.name}: derivative {index} has shape {tuple(got.shape)}, "
                f"the inputs expected {tuple(shapes[index])}"
            )
    return close(
        analytic,
        automatic,
        rel=rel,
        msg=msg or f"{context.column.name}: VJP vs autograd",
    )


def check(
    context: Any,
    args: Args | None = None,
    kwargs: Kwargs | None = None,
    *,
    rel: float | None = None,
    dtype: torch.dtype = torch.float32,
) -> float:
    """The row's default verdict: reference agreement, or the VJP on a backward column."""
    if args is None or kwargs is None:
        args, kwargs = context.inputs(dtype, context.variant)
    if context.backward:
        return jacobian(context, args, kwargs, rel=JACOBIAN_RTOL if rel is None else rel)
    return forward(context, args, kwargs, rel=rel, dtype=dtype)


#: Which positional arguments carry a value the operator reads, per family.
#: Everything else in the signature is a route: a reduction axis, a slot list,
#: an angle table. Separating the two is what lets a test sweep dtypes or
#: options across the operands without also sweeping the metadata -- mixing a
#: route is a different claim (C4), and mixing it here would test it twice.
#: The cache's two destination rows are left out on purpose: they are written,
#: not read, so widening them is not the mixed-input question this table answers.
VALUE_SLOTS: dict[str, tuple[int, ...]] = {
    "add": (0, 1),
    "flash_attention": (0, 1, 2),
    "kv_cache": (0, 1),
    "softmax": (0,),
    "rope": (0,),
    "rms_norm": (0, 1),
}

#: The one numeric knob each family exposes, and the values worth sweeping.
#: A family with no knob is absent: its remaining options are choices between
#: behaviours (a causal mask, a split rotation), which D1 and C4 already cover.
NUMERIC_OPTION: dict[str, tuple[str, tuple[float, ...]]] = {
    "add": ("alpha", (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.5)),
    "flash_attention": ("softmax_scale", (0.25, 0.5, 1.0, 1.0 / 4**0.5, 2.0)),
    "rms_norm": ("eps", (1e-12, 1e-9, 1e-6, 1e-3, 1e-1)),
}


def value_slots(family: str) -> tuple[int, ...]:
    """The argument positions of ``family`` that carry a value."""
    return VALUE_SLOTS.get(family, ())


def numeric_option(family: str) -> tuple[str, tuple[float, ...]]:
    """``(name, values)`` for the family's numeric knob; ``("", ())`` if it has none."""
    return NUMERIC_OPTION.get(family, ("", ()))
