"""F3 -- against a baseline that is not the operator itself.

The clause asks what the operator costs next to a baseline. On this matrix the
operator and its reference are the same arithmetic written once -- ``Op.forward``
*is* the reference -- so timing ``forward`` against ``forward`` measures
scheduling jitter and can only ever come out near 1.0. That is not a baseline;
it is the operator compared with itself, and a row that does it reports a number
that no reader can act on.

So the baseline here is each operator written out as the plain eager expression
of its contract (``BASELINES``), which is what a caller without the operator
would write. The two are timed on the same inputs and the ratio is recorded. On
a CPU build the two are the same arithmetic in the same interpreter and the
ratio is 1.0 by construction, so the row is declared not applicable there --
that is the same reasoning the part 7.2 F3 uses for its CPU cells.

Tolerance: 5 x the noise slack (7.5x). The question is order of magnitude, not
percent.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..context import RunContext
from ._support import NOISE_SLACK, adapt, measure, require_resolvable

ID = "F3"
OPS = None


def _baseline_grouped_gemm(a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    bounds = [int(v) for v in offsets.tolist()]
    pieces = [
        a[bounds[i] : bounds[i + 1]] @ b[i] for i in range(b.shape[0]) if bounds[i + 1] > bounds[i]
    ]
    return torch.cat(pieces, dim=0) if pieces else a.new_zeros((0, b.shape[-1]))


def _baseline_layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    centred = x - x.mean(dim=-1, keepdim=True)
    variance = (centred * centred).mean(dim=-1, keepdim=True)
    return centred / torch.sqrt(variance + eps) * weight + bias


def _baseline_gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """GELU written out, honouring the same ``approximate`` switch as the operator."""
    if approximate == "tanh":
        inner = torch.sqrt(torch.tensor(2.0 / torch.pi, dtype=x.dtype))
        return 0.5 * x * (1.0 + torch.tanh(inner * (x + 0.044715 * x**3)))
    return 0.5 * x * (1.0 + torch.erf(x / float(2.0**0.5)))


def _baseline_relu(x: torch.Tensor) -> torch.Tensor:
    return x.clamp_min(0)


def _baseline_amax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x.abs().max(dim=dim).values


#: One plain-expressive baseline per operator, written without calling the operator.
BASELINES: dict[str, Callable[..., torch.Tensor]] = {
    "grouped_gemm": _baseline_grouped_gemm,
    "layer_norm": _baseline_layer_norm,
    "gelu": _baseline_gelu,
    "relu": _baseline_relu,
    "amax": _baseline_amax,
}


def run(ctx: RunContext) -> None:
    if ctx.device.type == "cpu":
        ctx.na(
            f"on a CPU build {ctx.op.name} and its baseline are the same eager expression, "
            "so the ratio would be 1.0 by construction"
        )
        return
    baseline = BASELINES.get(ctx.op.name)
    if baseline is None:
        ctx.na(f"no plain-expression baseline is written for {ctx.op.name}")
        return

    generator = torch.Generator().manual_seed(ctx.seed)
    x = torch.randn(128, 32, generator=generator)
    args, kwargs = adapt(ctx.op.name, x)
    moved = tuple(ctx.move(a).float() for a in args)
    measured = require_resolvable(measure(lambda: ctx.op.forward(*moved, **kwargs)), ctx.op.name)

    plain_args = tuple(a.float() for a in args)
    reference = require_resolvable(
        measure(lambda: baseline(*plain_args, **kwargs)),
        f"{ctx.op.name} (baseline)",
    )

    if measured <= 0 or reference <= 0:
        raise AssertionError(f"{ctx.op.name}: a timing came back as zero")
    ratio = measured / reference
    if not ratio <= 5.0 * NOISE_SLACK:
        raise AssertionError(
            f"{ctx.op.name}: {ratio:.2f}x the plain-expression baseline, which is not the same "
            "arithmetic at a constant factor"
        )
    ctx.note(f"ratio to the plain-expression baseline {ratio:.2f}x")
