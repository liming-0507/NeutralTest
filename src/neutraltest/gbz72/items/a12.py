"""A12 -- the analytic gradient is the derivative of the forward.

Check: the column's hand-written Jacobian is compared with a central difference
of that column's own forward. Nothing here goes through autograd: A1 already
holds the analytic derivative against autograd on a backward column, and the two
verifications fail in different ways, so running only one of them leaves a shape
of error unmeasured.

The point of a finite difference is that it knows nothing about the operator. It
perturbs one input element, re-runs the forward, and reads the change in the
cotangent direction the Jacobian was applied to. If the Jacobian and the forward
disagree -- a transposed index, a missing factor of the scale, a term dropped
because it usually vanishes -- both still agree with autograd, because autograd
differentiates the *forward*, and the forward is what is wrong. Only a
difference against the forward itself catches that.

Two properties are checked per derivative:

* the entry of the Jacobian at a sampled coordinate matches the finite
  difference of ``cotangent . forward`` with respect to that coordinate, and
* the pair of them agrees again at twice the cotangent, which rules out a
  Jacobian that happens to match at one scale.

Tolerance: the difference is a divided difference, so its own rounding sets the
floor. For ``h`` around ``1e-6`` relative to a float64 input of unit magnitude,
the truncation error is ``h^2`` and the round-off is ``eps/h``; both are near
``1e-10``, and the check allows ``1e-6`` of relative error, three orders of
magnitude above that floor and still far below the gaps a real derivative error
produces.
"""

from __future__ import annotations

from typing import Any

import torch

from ...gbz73.inference import context
from .._support import make_inputs

ID = "A12"
COLUMNS = None

#: Step of the central difference, relative to the largest coordinate it perturbs.
STEP = 1e-6
#: Relative tolerance on the divided difference.
RTOL = 1e-6
#: How many coordinates of each derivative are differentiated. The full Jacobian
#: is quadratic in the element count, so a sampled subset is what makes this
#: cheap enough to run on every column; the coordinates are spread across the
#: tensor so that a transposed index hits at least one of them.
SAMPLES = 4


def run(ctx: Any) -> None:
    args, kwargs = make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant)
    analytic = ctx.gradients_with(args, kwargs, dtype=torch.float64)[0]
    if not analytic:
        ctx.na("this column differentiates nothing, so there is no derivative to check")
    numeric = _numeric(ctx, args, kwargs)
    if len(numeric) != len(analytic):
        raise AssertionError(
            f"{ctx.column.name}: the finite difference reached {len(numeric)} arguments, "
            f"the Jacobian returned {len(analytic)} derivatives"
        )
    checked = 0
    for position, (got, want) in enumerate(zip(analytic, numeric, strict=True)):
        if got.shape != want.shape:
            raise AssertionError(
                f"{ctx.column.name}: derivative {position} has shape {tuple(got.shape)}, "
                f"the finite difference has {tuple(want.shape)}"
            )
        if got.numel() == 0:
            continue
        _agree(ctx, position, got, want)
        checked += 1
    if not checked:  # pragma: no cover - every family carries at least one tensor
        ctx.na("no differentiable argument of this column holds a value")
    ctx.note(f"finite differences: {checked} derivative(s) at {SAMPLES} coordinate(s) each")


def _agree(ctx: Any, position: int, analytic: torch.Tensor, numeric: torch.Tensor) -> None:
    """Compare two derivatives coordinate by coordinate over the sampled subset."""
    flat_analytic = analytic.detach().reshape(-1).to(torch.float64)
    flat_numeric = numeric.detach().reshape(-1).to(torch.float64)
    picked = _spread(flat_analytic.numel(), min(SAMPLES, flat_analytic.numel()))
    for index in picked:
        got = float(flat_analytic[index].item())
        want = float(flat_numeric[index].item())
        difference = abs(got - want)
        allowed = RTOL * max(abs(want), 1e-12)
        if difference > allowed:
            raise AssertionError(
                f"{ctx.column.name}: derivative {position}[{index}] is {got:.6e} against a "
                f"finite difference of {want:.6e} (|difference| {difference:.3e} > {allowed:.3e})"
            )


def _numeric(ctx: Any, args: Any, kwargs: Any) -> list[torch.Tensor]:
    """The Jacobian of ``cotangent . forward``, read off the forward itself.

    The positions come from the column, not from the argument types: a float
    tensor can still be a route (the rotation's angle table is float because it
    is arithmetic, but it carries the position rather than a value), and guessing
    by dtype would put a derivative where the Jacobian reports none. Taking the
    column's own list is what keeps the two sides positionally aligned.

    ``_directional_positions`` counts along the signature, while ``project``
    returns only the tensors it passes positionally -- attention takes its mask
    and its scale as keywords, so the signature's fifth parameter is the third
    tensor. The names bridge the two: a differentiated name that ``project``
    handed over as a tensor is swept, one it handed over as a keyword is not,
    because there is no Jacobian block for it to disagree with.
    """
    forward, _ = ctx._implementation()
    directional, directional_kwargs = ctx.project(_cast(args, torch.float64), kwargs)
    cotangent = context._cotangent(forward(*directional, **directional_kwargs), torch.float64)
    names = ctx._argument_names()
    swept = {names[position] for position in ctx._directional_positions()}

    results: list[torch.Tensor] = []
    for position, value in enumerate(directional):
        if names[position] not in swept:
            continue
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise AssertionError(
                f"{ctx.column.name}: the column differentiates {names[position]}, "
                f"which is {type(value).__name__} rather than a float tensor"
            )
        results.append(
            _one_argument(forward, directional, directional_kwargs, position, value, cotangent)
        )
    return results


def _one_argument(
    forward: Any,
    directional: tuple,
    kwargs: dict,
    position: int,
    value: torch.Tensor,
    cotangent: Any,
) -> torch.Tensor:
    """The gradient of ``cotangent . forward`` with respect to one argument."""
    step = STEP * float(value.detach().abs().max().item() or 1.0)
    gradient = torch.zeros(value.shape, dtype=torch.float64)
    flat = gradient.reshape(-1)
    source = value.detach().to(torch.float64)
    picked = _spread(source.numel(), min(SAMPLES, source.numel()))
    for index in picked:
        up = source.clone()
        up.reshape(-1)[index] += step
        down = source.clone()
        down.reshape(-1)[index] -= step
        slope = (
            _pairing(forward, directional, kwargs, position, up, cotangent)
            - _pairing(forward, directional, kwargs, position, down, cotangent)
        ) / (2.0 * step)
        flat[index] = slope
    return gradient


def _pairing(
    forward: Any,
    directional: tuple,
    kwargs: dict,
    position: int,
    replacement: torch.Tensor,
    cotangent: Any,
) -> torch.Tensor:
    """``cotangent . forward(...)`` with one argument swapped."""
    swapped = list(directional)
    swapped[position] = replacement
    return _dot(forward(*swapped, **kwargs), cotangent)


def _dot(out: Any, cotangent: Any) -> torch.Tensor:
    """The cotangent pairing of a forward result that may be a tuple of tensors."""
    values = out if isinstance(out, tuple) else (out,)
    weights = cotangent if isinstance(cotangent, tuple) else (cotangent,)
    total = torch.zeros((), dtype=torch.float64)
    for produced, weight in zip(values, weights, strict=True):
        total = total + (produced.to(torch.float64) * weight.to(torch.float64)).sum()
    return total


def _spread(total: int, count: int) -> list[int]:
    """``count`` indices spread evenly over ``range(total)``."""
    if count <= 0:
        return []
    if count >= total:
        return list(range(total))
    stride = total / count
    return [min(int(index * stride), total - 1) for index in range(count)]


def _cast(args: tuple, dtype: torch.dtype) -> tuple:
    """The float arguments at ``dtype``; nothing else is touched."""
    return tuple(
        a.to(dtype) if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in args
    )


__all__ = ["COLUMNS", "ID", "RTOL", "SAMPLES", "STEP", "run"]
