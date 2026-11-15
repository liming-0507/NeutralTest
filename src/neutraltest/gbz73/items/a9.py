"""A9 -- the mixed-precision path: FP8 storage with a wider intermediate.

The clause asks for two different things depending on the operator, and the
row has to serve both:

* A compute with two operands (Grouped GEMM, LayerNorm) can put *one* of them
  in FP8 and leave the other in float.  That is the deployment shape most of
  the time -- weights quantised, activations not -- and it must be at least as
  accurate as quantising both, because it pays one rounding instead of two.
* A single-operand activation (GELU) has no second operand to leave in float.
  What the clause asks of it is the storage-and-compute split written down for
  that operator: *FP8 input, FP16/BF16/FP32 intermediate, FP8 output.*  A
  single-operand operator therefore still has a mixed-precision path; it is
  the width of the intermediate, not the width of a second operand.

Both shapes are measured against the same golden, and both are reported as a
ratio to the error the FP8 representation alone would cause -- a run that
merely equals its own quantisation floor has not shown an implementation
problem, and a run far above it has.
"""

from __future__ import annotations

import torch

from .. import metrics, quant
from ..context import NotApplicable, RunContext

ID = "A9"
OPS = None


def _ratio_to_floor(diff: metrics.Diff, floor: float) -> float:
    """How many times the representation floor this run's error is."""
    return float(diff.rel) / floor if floor > 0 else float("inf")


def _scale_for(args: tuple[torch.Tensor, ...], target: torch.dtype) -> torch.Tensor:
    floats = [a for a in args if isinstance(a, torch.Tensor) and a.is_floating_point()]
    if not floats:
        raise NotApplicable("no floating-point input to place on the FP8 grid")
    amax = torch.stack([t.abs().max() for t in floats]).max().clamp_min(1e-12)
    return torch.as_tensor(amax / quant.max_finite(target), dtype=torch.float32)


def run(ctx: RunContext) -> None:
    name = ctx.op.name
    args, kwargs = ctx.inputs(torch.float32)
    target = ctx.fp8_dtype
    floor = ctx.floor()

    floats = [
        i for i, a in enumerate(args) if isinstance(a, torch.Tensor) and a.is_floating_point()
    ]
    if not floats:
        ctx.na(f"{name} has no floating-point input, so nothing can be placed on the FP8 grid")

    golden = ctx.golden()
    quantised, quantised_kwargs, scale = ctx.quantise_inputs(args, kwargs)

    if len(floats) >= 2:
        # Two operands: one in FP8, the other left in float.  Accuracy must beat
        # quantising both, because one rounding is paid instead of two.
        # Only the first floating-point operand crosses the grid; every other
        # operand keeps full float width.  An earlier draft left the first one
        # quantised and widened *all the others*, which measured a different
        # arrangement entirely and could not detect a kernel that rounds the
        # wide operand internally.
        mixed = list(args)
        mixed[floats[0]] = quantised[floats[0]]
        out = ctx.op.forward(*mixed, **kwargs).to(torch.float32)
        both = ctx.op.forward(*quantised, **quantised_kwargs).to(torch.float32)

        mixed_diff = ctx.diff(out, golden)
        both_diff = ctx.diff(both, golden)
        metrics.assert_within(
            mixed_diff, rel=floor, msg=f"{name}: one operand on the FP8 grid vs golden"
        )
        if mixed_diff.rel > both_diff.rel:
            raise AssertionError(
                f"{name}: quantising one operand is worse than quantising both "
                f"({mixed_diff.rel:.4e} > {both_diff.rel:.4e}); the wider operand is not "
                "reaching the kernel as a wider operand"
            )
        ctx.note(
            f"one operand FP8 {mixed_diff.rel:.4e} vs both FP8 {both_diff.rel:.4e} "
            f"(floor {floor:.4e}), scale {float(scale):.6f}"
        )
        return

    # One operand: the clause's storage-and-compute split.  The input crosses the
    # FP8 grid and the operator computes in float32, so what is left is the
    # representation error plus whatever the implementation adds on top of it.
    on_grid = quant.roundtrip(args[floats[0]], scale, target)
    stored = list(args)
    stored[floats[0]] = on_grid
    out = ctx.op.forward(*stored, **kwargs).to(torch.float32)
    diff = ctx.diff(out, golden)

    # The same arithmetic on the same inputs one width up: FP16 is far enough
    # from FP8 that its share of the error is a rounding, not the subject.
    wider = args[floats[0]].to(torch.float16).to(torch.float32)
    narrow = list(args)
    narrow[floats[0]] = wider
    narrow_diff = ctx.diff(ctx.op.forward(*narrow, **kwargs).to(torch.float32), golden)

    if not narrow_diff.rel < diff.rel:
        raise AssertionError(
            f"{name}: storing the input wider did not reduce the error "
            f"({narrow_diff.rel:.4e} vs {diff.rel:.4e}); the input is not reaching the "
            "compute at the width the call asked for"
        )
    metrics.assert_within(diff, rel=floor, msg=f"{name}: FP8 input, float intermediate, vs golden")
    ctx.note(
        f"FP8 in / float mid {diff.rel:.4e} ({_ratio_to_floor(diff, floor):.2f}x floor), "
        f"FP16 in {narrow_diff.rel:.4e}, scale {float(scale):.6f}"
    )
