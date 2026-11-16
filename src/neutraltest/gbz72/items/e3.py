"""E3 -- many callers at once, one operator.

Check: the same column is called from several callers at the same time, first on
one set of arguments and then on several, and every answer has to equal the
answer the same call gives when it runs alone. A kernel that keeps state between
calls -- a scratch buffer reused without being cleared, a counter, a cached
handle keyed on nothing -- answers the second caller with the first caller's
leftovers, and only a call that overlaps another can show it.

Two mechanisms are used, in the order the backend offers them. A backend with a
stream API runs two real streams; everything else runs four threads. The threads
are the weaker of the two at the hardware level, but they are not nothing: a
kernel's scratch workspace, its cached descriptors and its Python-side state are
shared by threads exactly as they are shared by streams -- and a device queue
reached from several host threads overlaps on the device on its own.

Tolerance: exact equality. The concurrent call and the serial call are the same
computation on the same operands, so any difference is shared state rather than
rounding -- which is the only thing this row is looking for.

The distinct-inputs half is what a shared cache cannot survive: the answers have
to be distinguishable from each other *and* each equal to its own serial answer,
so an operator that hands back whatever the last caller produced fails even when
every call would individually succeed.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Sequence
from typing import Any

import torch

from .._support import make_inputs, on_device

ID = "E3"
COLUMNS = None

#: Callers the thread half uses. Four is the smallest number that can show a
#: two-way mix-up *and* a rotation, and it is what a serving process does.
THREADS = 4

#: Distinct argument sets the interleaved half puts in flight at once.
DISTINCT = 3


def run(ctx: Any) -> None:
    # The samples are built on the host, so they travel to the device here: a
    # gradient is taken by re-running the call, and that path does not move the
    # arguments the way a plain call does.
    args, kwargs = on_device(
        ctx, *make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant)
    )
    serial = _answer(ctx, args, kwargs)
    if _has_streams(ctx):
        _streams(ctx, args, kwargs, serial)
    else:
        _threads(ctx, args, kwargs, serial)


# -- the two mechanisms ---------------------------------------------------


def _streams(ctx: Any, args: Any, kwargs: Any, serial: Any) -> None:
    """Two real streams: the backend's own scheduling, not the host's.

    The two calls are queued inside the same block rather than one after the
    other, because a call that has already been waited on is a serial call. The
    first stream's answer is compared with the serial one; the second is
    compared with what the same arguments answer when they are the only call in
    flight -- a stream that ends up reading the other stream's workspace shows up
    in the second comparison, and an operator that is simply not reentrant shows
    up in both.
    """
    backend = ctx.backend
    first = args
    second = _perturbed(ctx, args)
    with backend.stream_context(None), backend.stream_context(None):
        # Both calls are made inside the block; the block itself is the argument.
        left_answer = _answer(ctx, first, kwargs)
        right_answer = _answer(ctx, second, kwargs)
    if not _equal(left_answer, serial):
        raise AssertionError(
            f"{ctx.column.name}: two concurrent streams disagreed with the serial call by up to "
            f"{_worst(left_answer, serial):.3e}, so the calls share state"
        )
    alone = _answer(ctx, second, kwargs)
    if not _equal(right_answer, alone):
        raise AssertionError(
            f"{ctx.column.name}: the second stream answered differently than the same call does "
            f"alone (by up to {_worst(right_answer, alone):.3e}), so the calls share state"
        )
    ctx.note(
        "concurrency (streams): two streams queued together, the first equal to its serial "
        "answer and the second equal to its own"
    )


def _threads(ctx: Any, args: Any, kwargs: Any, serial: Any) -> None:
    """Four callers on the same operator, first on one input then on four."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as pool:
        same = list(pool.map(lambda _: _answer(ctx, args, kwargs), range(THREADS)))
    for position, answer in enumerate(same):
        if not _equal(answer, serial):
            raise AssertionError(
                f"{ctx.column.name}: concurrent call {position + 1} disagreed with the serial one "
                f"by up to {_worst(answer, serial):.3e}, so the calls share state"
            )

    # make_inputs builds on the host, so every sample has to travel to the
    # device here: a device kernel handed one host tensor and one device tensor
    # refuses the call before any concurrency is involved, which would turn a
    # missing move into a report about shared state.
    cases = [
        on_device(ctx, *make_inputs(family=ctx.family, seed=ctx.seed + k, variant=ctx.variant))
        for k in range(DISTINCT)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases)) as pool:
        mixed = list(pool.map(lambda case: _answer(ctx, case[0], case[1]), cases))
    for position, (answer, case) in enumerate(zip(mixed, cases, strict=True)):
        alone = _answer(ctx, case[0], case[1])
        if not _equal(answer, alone):
            raise AssertionError(
                f"{ctx.column.name}: with {len(cases)} inputs in flight at once, input "
                f"{position} answered differently than it does alone (by up to "
                f"{_worst(answer, alone):.3e}), so the calls share state"
            )
    # The distinct-answer check is a guard on the sample, not on the operator: if
    # two of the interleaved samples would answer alike even when run one at a
    # time, then a tie says nothing about shared state.  A backward column is the
    # common case -- a gradient with respect to a linear operator's cotangent does
    # not depend on the operand's values -- so the guard is applied only where the
    # samples really do differ when run alone.
    alone_answers = [_answer(ctx, case[0], case[1]) for case in cases]
    if len({_fingerprint(answer) for answer in alone_answers}) == len(alone_answers) and len(
        {_fingerprint(answer) for answer in mixed}
    ) != len(mixed):
        raise AssertionError(
            f"{ctx.column.name}: two of the {len(cases)} inputs in flight got the same answer, "
            f"so the calls share state"
        )
    ctx.note(
        f"concurrency (no stream API): {THREADS} threads, identical and distinct inputs, "
        f"{len(cases)} inputs interleaved, each equal to its serial answer"
    )


# -- helpers --------------------------------------------------------------


def _has_streams(ctx: Any) -> bool:
    """Whether this backend offers a stream API this row can drive."""
    backend = getattr(ctx, "backend", None)
    checker = getattr(backend, "has_streams", None)
    return bool(checker()) if callable(checker) else False


def _answer(ctx: Any, args: Any, kwargs: Any) -> Any:
    """What the column answers for ``args``: a tuple of tensors, forward or backward."""
    if ctx.backward:
        gradients, _ = ctx.gradients_with(args, kwargs)
        return tuple(gradients)
    return _as_tuple(ctx.call_with(args, kwargs))


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, Sequence) and not isinstance(value, torch.Tensor):
        return tuple(value)
    return (value,)


def _perturbed(ctx: Any, args: Any) -> Any:
    """A copy of ``args`` with the first floating operand scaled, on the device."""
    copies = list(args)
    for position, value in enumerate(copies):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            scaled = value * 1.5
            copies[position] = ctx.move(scaled)
            break
    return tuple(copies)


def _equal(left: Any, right: Any) -> bool:
    """Bit-equality of two answers, as tuples of tensors."""
    lefts, rights = _as_tuple(left), _as_tuple(right)
    if len(lefts) != len(rights):
        return False
    return all(
        a.shape == b.shape and torch.equal(a.detach().cpu(), b.detach().cpu())
        for a, b in zip(lefts, rights, strict=True)
    )


def _worst(left: Any, right: Any) -> float:
    worst = 0.0
    for a, b in zip(_as_tuple(left), _as_tuple(right), strict=False):
        if a.shape != b.shape:
            return float("inf")
        if a.is_floating_point() and a.numel():
            gap = (a.detach().cpu().float() - b.detach().cpu().float()).abs().max()
            worst = max(worst, float(gap))
    return worst


def _fingerprint(answer: Any) -> tuple[bytes, ...]:
    return tuple(value.detach().cpu().contiguous().numpy().tobytes() for value in _as_tuple(answer))
