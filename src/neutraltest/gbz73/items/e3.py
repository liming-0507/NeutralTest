"""E3 -- many callers at once, one operator.

Two callers share one device in the ordinary serving case, so the row asks the
same question of every backend and only the mechanism changes: when several
calls overlap, does each answer belong to its own arguments?

*Without a stream API* the overlap is made with threads. Four threads call the
operator at the same time on one set of arguments -- the answers must be the one
a serial call gives -- and then on four different sets of arguments at once,
each of which must equal what it gives alone. The second half is the one that
matters: an operator that parks a partial sum, a shape or a workspace in a
module-level global answers the single-input case correctly by accident and
mixes the four up.

*With a stream API* the overlap is made with two streams on the device under
test. The same arguments are queued on both and must agree with the serial
answer; then two different sets of arguments are queued on the two streams and
each must be its own. The mechanism actually used for the overlap is written to
the row's note, because "synchronised with the stream context manager" and
"synchronised by hand afterwards" are not the same amount of evidence.

Tolerance: exact for the serial comparison (the device under test against
itself, so there is nothing a tolerance could excuse), and :meth:`RunContext.check`
-- the format floor -- when a stream forces a choice of dtype the serial path
did not make.

Neither half is not applicable. A backend with no streams still serves
concurrent callers, and a backend with streams still has to survive two of them;
the row reports which half it could make on this backend and leaves nothing
blank.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
from typing import Any

import torch

from ..context import RunContext
from ..ops import Args, Kwargs

ID = "E3"
OPS = None

#: Threads the no-stream half uses. Four is the smallest number that can show a
#: two-way mix-up *and* a rotation, and it is what a serving process does.
THREADS = 4


def run(ctx: RunContext) -> None:
    args, kwargs = ctx.inputs(torch.float32)
    serial = ctx.run(args, kwargs)
    if ctx.backend.has_streams():
        _streams(ctx, serial)
    else:
        _threads(ctx, serial)


# -- no stream API: threads ----------------------------------------------


def _threads(ctx: RunContext, serial: torch.Tensor) -> None:
    """Four callers on the same operator, first on one input then on four.

    The distinct-inputs half is what a shared cache cannot survive: the four
    answers have to be distinguishable from each other *and* each equal to its
    own serial answer, so an operator that reuses whatever the last caller left
    behind fails even when every call would individually succeed.
    """
    args, kwargs = ctx.inputs(torch.float32)
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as pool:
        same = list(pool.map(lambda _: ctx.run(args, kwargs), range(THREADS)))
    for position, answer in enumerate(same):
        if answer.shape != serial.shape:
            raise AssertionError(
                f"{ctx.op.name}: concurrent call {position + 1} returned "
                f"{tuple(answer.shape)}, the serial call returned {tuple(serial.shape)}"
            )
        if not torch.equal(answer, serial):
            raise AssertionError(
                f"{ctx.op.name}: concurrent call {position + 1} disagreed with the serial one "
                f"by up to {_worst(answer, serial):.3e}"
            )

    cases = _cases(ctx)
    if not cases:
        raise AssertionError(
            f"{ctx.op.name}: the sampler produced no distinct inputs to interleave"
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases)) as pool:
        mixed = list(pool.map(lambda case: ctx.run(case[0], case[1]), cases))
    for position, (answer, case) in enumerate(zip(mixed, cases, strict=True)):
        alone = ctx.run(case[0], case[1])
        if not torch.equal(answer, alone):
            raise AssertionError(
                f"{ctx.op.name}: with {len(cases)} inputs in flight at once, input {position} "
                f"answered differently than it does alone (by up to {_worst(answer, alone):.3e}), "
                f"so the calls share state"
            )
    if len({tuple(answer.reshape(-1).tolist()) for answer in mixed}) != len(mixed):
        raise AssertionError(
            f"{ctx.op.name}: two of the {len(cases)} inputs in flight got the same answer, "
            f"so the calls share state"
        )
    ctx.note(
        f"concurrency (no stream API): {THREADS} threads, identical and distinct inputs, "
        f"{len(cases)} inputs interleaved, each equal to its serial answer"
    )


def _cases(ctx: RunContext) -> list[tuple[Args, Kwargs]]:
    """Distinct sets of arguments, each valid for the operator, that disagree.

    Windows of a long vector rather than four independent draws: the last three
    overlap the first, so an operator that computes on a stale view of its
    argument is caught at the seam and not only in the interior.
    """
    args, kwargs = ctx.inputs(torch.float32)
    template = next(
        (arg for arg in args if isinstance(arg, torch.Tensor) and arg.is_floating_point()), None
    )
    if template is None:
        return [(args, kwargs), ctx.inputs(torch.float32)]

    values = template.reshape(-1)
    if values.numel() < 2 * THREADS:
        return [(args, kwargs), ctx.inputs(torch.float32)]
    cases: list[tuple[Args, Kwargs]] = []
    for start in range(THREADS):
        window = torch.roll(values, shifts=-start * (values.numel() // THREADS)).reshape(
            template.shape
        )
        cases.append(
            (
                tuple(window if arg is template else arg for arg in args),
                dict(kwargs),
            )
        )
    return cases


# -- with a stream API: two streams --------------------------------------


def _streams(ctx: RunContext, serial: torch.Tensor) -> None:
    """Two streams on the device under test, on one input and then on two.

    Both streams are entered around one block so the two launches are genuinely
    queued at the same time rather than one after the other, and the operands are
    mirrored per stream: a call that was handed buffers already queued on the
    other stream would be serialised by the device, and the row would report an
    overlap it never made.
    """
    args, kwargs = ctx.inputs(torch.float32)
    first = ctx.backend.new_stream(0)
    second = ctx.backend.new_stream(1)
    method = _overlap_method(ctx)

    with ctx.backend.stream_context(first), ctx.backend.stream_context(second):
        same = [
            _onto(ctx, first, args, kwargs),
            _onto(ctx, second, args, kwargs),
        ]
    _settle(ctx)
    for position, answer in enumerate(same):
        if answer.shape != serial.shape:
            raise AssertionError(
                f"{ctx.op.name}: the call on stream {position} returned {tuple(answer.shape)}, "
                f"the serial call returned {tuple(serial.shape)}"
            )
        if not torch.equal(answer, serial):
            raise AssertionError(
                f"{ctx.op.name}: the call on stream {position} disagreed with the serial one "
                f"by up to {_worst(answer, serial):.3e}"
            )

    cases = _cases(ctx)
    with ctx.backend.stream_context(first), ctx.backend.stream_context(second):
        mixed = [_onto(ctx, first, *cases[0]), _onto(ctx, second, *cases[-1])]
    _settle(ctx)
    for position, (answer, case) in enumerate(zip(mixed, (cases[0], cases[-1]), strict=True)):
        # Run alone on a stream of its own, so "what it does alone" is measured
        # under the same mechanism and not on the default stream.
        with ctx.backend.stream_context(ctx.backend.new_stream(0)):
            alone = _onto(ctx, ctx.backend.new_stream(0), case[0], case[1])
        _settle(ctx)
        if not torch.equal(answer, alone):
            raise AssertionError(
                f"{ctx.op.name}: on two streams at once, the call on stream {position} "
                f"answered differently than it does alone (by up to {_worst(answer, alone):.3e}), "
                f"so the streams share state"
            )
    if torch.equal(mixed[0], mixed[1]):
        raise AssertionError(
            f"{ctx.op.name}: two different inputs on two streams came back with the same "
            f"answer, so the streams share state"
        )
    ctx.note(
        f"concurrency (stream API): 2 streams, identical and distinct inputs, {method}, "
        f"each equal to its serial answer"
    )


def _onto(ctx: RunContext, stream: Any, args: Args, kwargs: Kwargs) -> torch.Tensor:
    """Run on ``stream`` (already current) and bring the result back.

    The operands are mirrored onto the stream's device first: an operator handed
    buffers that the default stream still owns is serialised against that stream
    by the device itself, which is the one way to overlap two streams and observe
    nothing.  The wait is a stream event rather than ``synchronize()`` for the
    same reason -- a device-wide synchronisation between the two launches would
    leave no overlap to observe.
    """
    moved = _mirror(ctx, stream, args)
    out = ctx.op.forward(*moved, **kwargs)
    wait = getattr(stream, "wait_stream", None)
    current = getattr(ctx.backend, "current_stream", None)
    if callable(wait) and callable(current):
        with contextlib.suppress(Exception):
            wait(current())
    return ctx.collect(out)


def _mirror(ctx: RunContext, stream: Any, args: Args) -> Args:
    """``args`` on the device ``stream`` belongs to, through fresh buffers."""
    described = getattr(stream, "device", None)
    if described is None:
        return tuple(ctx.move(argument) for argument in args)
    return tuple(
        ctx.move(argument).to(described) if isinstance(argument, torch.Tensor) else argument
        for argument in args
    )


def _overlap_method(ctx: RunContext) -> str:
    """How the overlap was actually arranged, for the row's note."""
    stream = ctx.backend.new_stream(0)
    described = getattr(stream, "device", None)
    where = f" on {described}" if described is not None else ""
    return (
        f"queued on two streams{where} with {type(stream).__name__} context managers and "
        f"per-stream operand buffers"
    )


def _settle(ctx: RunContext) -> None:
    """Wait for everything queued before anything is compared."""
    ctx.backend.synchronize()


def _worst(got: torch.Tensor, want: torch.Tensor) -> float:
    if got.shape != want.shape:
        return float("nan")
    if got.numel() == 0:
        return 0.0
    return float((got.double() - want.double()).abs().max())
