"""A6 -- NaN and infinity must survive, or be refused, but never disappear.

A quantiser that maps NaN to zero turns a poisoned gradient into a plausible
number, which is the worst failure mode a training operator can have.

The assertion is deliberately implementation-neutral.  What the poison does to
an element differs between operators -- ``relu(-inf)`` is 0, ``amax`` takes the
magnitude, ``gelu(+inf)`` is +inf but an exact float32 kernel may also produce
NaN, and normalising infinities is undefined -- so this row does not hard-code
one answer.  It requires the one thing common to all of them: an element the
poison reaches may only become **zero or non-finite**, never a plausible
non-zero finite number, and the poison must change something.
"""

from __future__ import annotations

import torch

from ..context import RunContext

ID = "A6"
OPS = None


def _changed(clean: torch.Tensor, poisoned: torch.Tensor) -> torch.Tensor:
    """Element-wise "really changed", with NaN treated as equal to NaN."""
    if clean.shape != poisoned.shape:
        raise AssertionError(
            "the output shape changed when the input was poisoned: "
            f"{tuple(clean.shape)} -> {tuple(poisoned.shape)}"
        )
    return ~torch.isclose(clean, poisoned, equal_nan=True)


def run(ctx: RunContext) -> None:
    args, kwargs = ctx.inputs(torch.float32)
    floats = [
        index
        for index, value in enumerate(args)
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    ]
    if not floats:
        ctx.na("no floating-point input to poison")

    position = floats[0]
    clean = ctx.op.forward(*args, **kwargs)
    # Poison the most significant element: flat index 0 is often already inside
    # relu's dead zone, where any poison is legitimately invisible.
    flat = args[position].detach().reshape(-1)
    hot = int(flat.abs().argmax())

    for poison, name in ((float("nan"), "NaN"), (float("inf"), "Inf"), (-float("inf"), "-Inf")):
        poisoned_args = list(args)
        sample = poisoned_args[position].clone()
        sample.view(-1)[hot] = poison
        poisoned_args[position] = sample

        try:
            out = ctx.op.forward(*poisoned_args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - a refusal is a legitimate answer
            raise AssertionError(
                f"{ctx.op.name}: {name} in the input was refused with {type(exc).__name__}: {exc}"
            ) from exc

        touched = _changed(clean, out)
        if not bool(touched.any()):
            if ctx.op.name == "relu" and name == "-Inf":
                # relu maps every non-positive value -- including -Inf -- to the
                # same zero, so "nothing changed" is the contract, not a miss.
                if not torch.equal(out, clean):
                    raise AssertionError("relu: -Inf changed the output in a non-zero way")
                ctx.note("relu: -Inf is indistinguishable from any other negative input")
                continue
            raise AssertionError(
                f"{ctx.op.name}: {name} in the input vanished without a trace "
                "(no output element changed)"
            )

        reachable = out[touched]
        if name == "NaN":
            if not bool(torch.isnan(reachable).any()):
                raise AssertionError(
                    f"{ctx.op.name}: NaN in the input did not reach the output "
                    f"(changed elements: {reachable.tolist()[:8]})"
                )
            continue

        finite = reachable[torch.isfinite(reachable)]
        non_finite = int((~torch.isfinite(reachable)).sum())
        zeros = int((finite == 0).sum())
        plausible = finite[finite != 0]
        if plausible.numel():
            raise AssertionError(
                f"{ctx.op.name}: {name} in the input produced plausible finite values "
                f"{plausible.tolist()[:8]} instead of zero or a non-finite value "
                f"({non_finite} non-finite, {zeros} zero)"
            )
        if name == "Inf" and non_finite == 0:
            raise AssertionError(
                f"{ctx.op.name}: +Inf in the input produced only zeros "
                f"({zeros} elements); the value should propagate"
            )
