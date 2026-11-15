"""E1 -- the same input gives the same result inside one process.

Check: the column is called three times on one set of arguments and the three
answers must be identical bit for bit. The same call is then made with the
things a kernel does not see removed from it one at a time: the process-wide
determinism switch, the size of the thread pool, and the memory the operands
arrive in. A kernel whose answer only holds because a framework flag happens to
be on, because the work was cut into the number of pieces it was cut into, or
because its input landed at the address it landed at, is reported here -- each
of those is an outside-the-call decision leaking into the result. The
process-wide random generator is read before and after every call: a
deterministic operator is a function of its arguments, and a function of its
arguments does not draw from the global stream. Finally the arguments are
checked for modification, because a call that edits its input has made the next
call's answer depend on how many calls came before it.

Why this is not E7. E7 (part 7.3) runs the operator on the deployment it was
measured on and asks whether two calls in that deployment agree, recording the
environment -- python, torch, backend, format floor -- they agreed in: "can this
build be run again?". This row answers a narrower question inside one process:
is the operator a *function* of its arguments? A kernel can be reproducible and
still not deterministic -- it always comes out one way for this process's launch
order, thread cut and memory layout, and another way for a different one -- and
E7 passes on it while this row reports it, because E7 varies nothing and this
row varies exactly the outside-the-call conditions. E7 is also a 7.3 row over
one operator rather than over these twelve columns. A green here says the
operator is a pure function of its inputs; a green in E7 says the deployment can
be repeated.

Tolerance: none -- every comparison is ``torch.equal``, bit for bit. A
tolerance would let a kernel whose low mantissa bits depend on the launch order
pass, which is the dependence this row exists to find. The single exception is
cross-thread repetition on a backend whose arithmetic happens on an accelerator:
there the thread pool decides host-side work rather than the operands'
summation, so the two answers are held to the column's ordinary elementwise
budget and the note says the comparison was relaxed. On a CPU backend, where
this row runs by default, the comparison is exact.

Scope: E3 (concurrency and streams) and E6 (multi-card) own races with other
threads and with a second device; this row keeps one process on one device and
varies only what the operator itself does not see.
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import budget, close, observe

ID = "E1"
COLUMNS = None

#: How many identical calls in a row are compared. Two is the smallest number
#: that can differ; three is the smallest that can tell "the first call is
#: special" -- a cold cache, a buffer initialised on first use -- from "the calls
#: drift", which are different defects.
REPEATS = 3

#: Thread-pool sizes the same call is repeated under, besides the process's own.
#: One worker is the control: a split that only exists for more than one cannot
#: decide an answer that is a function of the operands.
THREAD_COUNTS = (1, 2)

Answers = tuple[torch.Tensor, ...]


def run(ctx: Any) -> None:
    args, kwargs = ctx.inputs(torch.float32)
    # Photographed as they are handed over, and the very same buffers re-checked
    # after every call -- not just once at the end. A kernel that edits an
    # operand also moves every later answer, so a row that noticed only at the
    # end would report drift and send the reader after the wrong defect; checking
    # after each call keeps the first edit, which is the one that explains the rest.
    before = _sample(ctx)
    _unchanged(ctx, before, args)
    baseline, drift = _repeats(ctx, args, kwargs, before)
    _determinism_switch(ctx, args, kwargs, baseline, before)
    relaxed = _threads(ctx, args, kwargs, baseline, before)
    _fresh_copies(ctx, args, kwargs, baseline, before)
    ctx.note(
        f"{_shape(baseline)}, bit-identical over {REPEATS} calls, under the determinism "
        f"switch, and on fresh copies; thread counts "
        f"{', '.join(map(str, THREAD_COUNTS))} "
        f"{'compared at the budget' if relaxed else 'bit-identical'} against the "
        f"process's {_thread_count()}; the global generator was not advanced; "
        f"drift={drift:.3e}"
    )


def _shape(answers: Answers) -> str:
    """The outputs the repeats were compared over, for the row's note."""
    shapes = ", ".join(
        f"{tuple(a.shape)}/{a.dtype}" if isinstance(a, torch.Tensor) else type(a).__name__
        for a in answers
    )
    return f"{len(answers)} output(s) [{shapes}]"


def _repeats(ctx: Any, args: tuple, kwargs: dict, before: tuple) -> tuple[Answers, float]:
    """The column ``REPEATS`` times; every answer identical to the first.

    The global generator's state is compared around each call rather than
    reseeded before it: a row that reseeded would hide the leak it is hunting
    -- a kernel drawing from the global stream would be reset to the same state
    before every call and come out identical.
    """
    state = torch.random.get_rng_state()
    answers: list[Answers] = []
    for call in range(REPEATS):
        answers.append(observe(ctx, args, kwargs))
        _unchanged(ctx, before, args)
        moved = torch.random.get_rng_state()
        if bool((moved != state).any()):
            raise AssertionError(
                f"{ctx.column.name}: call {call + 1} advanced the global random generator, "
                f"so its result came from state outside its arguments"
            )
    for call, produced in enumerate(answers[1:], start=2):
        _same(ctx, answers[0], produced, f"call {call} of {REPEATS}")
    return answers[0], _largest_gap(answers)


def _unchanged(ctx: Any, before: Answers, after: tuple | None = None) -> None:
    """The calls must not have modified what they were handed.

    ``before`` is a snapshot of the operands taken at the start of the row. With
    ``after`` given the operands themselves are checked against it -- the same
    buffers, so an edit that was made and undone between calls is caught too;
    without it the sample is rebuilt from its seed and compared, which catches an
    edit that is still there when the row looks. The row does both: the first
    before any call is made, the second after the last.
    """
    current = _sample(ctx) if after is None else after
    for position, (now, was) in enumerate(zip(current, before, strict=True)):
        if isinstance(now, torch.Tensor) and not _identical(was, now):
            raise AssertionError(
                f"{ctx.column.name}: the call modified its input {position}, so the second "
                f"call did not receive the first call's arguments"
            )


def _sample(ctx: Any) -> tuple:
    """A host copy of the operands, to compare the row's own tensors against.

    ``clone`` rather than ``_host``: for a CPU tensor ``to("cpu")`` hands back the
    same tensor, so a copy taken that way would be the operand itself and would
    move with the very edit it is there to reveal.
    """
    return tuple(
        value.clone() if isinstance(value, torch.Tensor) else value
        for value in ctx.inputs(torch.float32)[0]
    )


def _determinism_switch(
    ctx: Any, args: tuple, kwargs: dict, baseline: Answers, before: tuple
) -> None:
    """The answer must not depend on the framework's determinism switch.

    ``torch.use_deterministic_algorithms`` asks the framework to make its own
    primitives reproducible. An operator that answers one way with the switch on
    and another way with it off relies on the framework rather than on its own
    order of operations, and a deployment that reaches it through a call path
    without the switch gets the other answer. The switch is restored whatever
    happens, so a row that fails cannot leave the setting behind for the rows
    that run after it.
    """
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        produced = observe(ctx, args, kwargs)
    finally:
        torch.use_deterministic_algorithms(previous)
    if torch.are_deterministic_algorithms_enabled() != previous:
        raise AssertionError(f"{ctx.column.name}: the determinism switch was not restored")
    _unchanged(ctx, before, args)
    _same(ctx, baseline, produced, "the call with deterministic algorithms enabled")


def _threads(ctx: Any, args: tuple, kwargs: dict, baseline: Answers, before: tuple) -> bool:
    """The answer must not depend on how many workers split the call.

    The thread count decides how a parallel reduction is cut up: a reduction
    ordered by the kernel's own code is stable across the cut, one that lets the
    cut decide the summation order is not. On a CPU that is the visible form of
    the defect a device shows as an unordered reduction. Each count is set and
    the original is restored in the ``finally``, so a failure leaves the process
    with the pool size it started with.
    """
    original = torch.get_num_threads()
    relaxed = False
    try:
        for count in THREAD_COUNTS:
            if count == original:
                continue
            torch.set_num_threads(count)
            produced = observe(ctx, args, kwargs)
            _unchanged(ctx, before, args)
            if ctx.backend.device_type == "cpu":
                _same(ctx, baseline, produced, f"the call with {count} thread(s)")
            else:
                _close(ctx, baseline, produced, count)
                relaxed = True
    finally:
        torch.set_num_threads(original)
    return relaxed


def _fresh_copies(ctx: Any, args: tuple, kwargs: dict, baseline: Answers, before: tuple) -> None:
    """The answer must follow the values, not the buffers they arrived in.

    ``clone`` hands back a new allocation, so a kernel that reads its input's
    address, or infers a stride from where the buffer starts, answers
    differently here. The copies are built eagerly in one expression per operand
    so that the allocator has to place each of them separately rather than
    reusing the one it just freed.
    """
    rebuilt = []
    for position, value in enumerate(args):
        if isinstance(value, torch.Tensor):
            rebuilt.append(_fresh(value, position))
        else:
            rebuilt.append(value)
    produced = observe(ctx, tuple(rebuilt), kwargs)
    _unchanged(ctx, before, args)
    _same(ctx, baseline, produced, "the call on freshly allocated copies of the inputs")


def _fresh(value: torch.Tensor, position: int) -> torch.Tensor:
    """A copy of ``value`` in a buffer nothing else has held.

    ``torch.empty_like`` followed by ``copy_`` is deliberately not
    ``clone()``: clone is allowed to serve the copy from the allocator's
    small-block cache, so two clones in a row can share an address. The filler
    allocation between them pushes the cache past this size for every operand
    this clause uses.
    """
    pad = 17 + position
    filler = torch.zeros((pad, 1021), dtype=torch.float64)  # 139 KiB: above the small-block pool
    copy = value.new_empty(value.shape)
    copy.copy_(value)
    if float(filler.sum()) != 0.0:  # pragma: no cover - keeps the allocation alive
        raise AssertionError("the filler allocation was elided")
    return copy


def _same(ctx: Any, reference: Answers, produced: Answers, what: str) -> None:
    """Every output of ``produced`` identical, bit for bit, to ``reference``."""
    if len(produced) != len(reference):
        raise AssertionError(
            f"{ctx.column.name}: {what} returned {len(produced)} outputs, the first call "
            f"returned {len(reference)}"
        )
    for index, (got, want) in enumerate(zip(produced, reference, strict=True)):
        if isinstance(got, torch.Tensor) != isinstance(want, torch.Tensor):
            raise AssertionError(
                f"{ctx.column.name}: {what} returned {type(got).__name__} as output {index}, "
                f"the first call returned {type(want).__name__}"
            )
        if not isinstance(got, torch.Tensor):
            if got != want:
                raise AssertionError(f"{ctx.column.name}: {what} changed output {index}")
            continue
        got, want = _host(got), _host(want)
        if got.shape != want.shape or got.dtype != want.dtype:
            raise AssertionError(
                f"{ctx.column.name}: {what} returned output {index} as "
                f"{tuple(got.shape)}/{got.dtype}, the first call returned "
                f"{tuple(want.shape)}/{want.dtype}"
            )
        if not _identical(want, got):
            raise AssertionError(
                f"{ctx.column.name}: {what} differs from the first call by up to "
                f"{_worst(want, got):.3e}, so the answer is not a function of the inputs"
            )


def _close(ctx: Any, reference: Answers, produced: Answers, count: int) -> None:
    """The accelerator fallback: the same answer to the ordinary elementwise budget."""
    dtype = next((t.dtype for t in reference if isinstance(t, torch.Tensor)), torch.float32)
    close(
        produced,
        reference,
        rel=budget(ctx, dtype),
        msg=f"{ctx.column.name}: the call with {count} thread(s)",
    )


def _identical(want: torch.Tensor, got: torch.Tensor) -> bool:
    """``torch.equal``, except that a NaN in one place must be a NaN in both."""
    if torch.equal(want, got):
        return True
    if want.numel() == 0:
        return True
    nan = want.isnan() | got.isnan()
    return bool(
        nan.any() and torch.equal(want[nan], got[nan]) and torch.equal(want[~nan], got[~nan])
    )


def _host(value: torch.Tensor) -> torch.Tensor:
    """A copy the comparison can index, so a device tensor need not implement the mask."""
    return value.detach().to("cpu")


def _worst(want: torch.Tensor, got: torch.Tensor) -> float:
    if want.shape != got.shape or want.numel() == 0:
        return float("nan") if want.shape != got.shape else 0.0
    return float((want.to(torch.float64) - got.to(torch.float64)).abs().max())


def _largest_gap(answers: list[Answers]) -> float:
    """The largest deviation between the repetitions; zero on a passing row."""
    worst = 0.0
    for produced in answers[1:]:
        for got, want in zip(produced, answers[0], strict=True):
            if isinstance(got, torch.Tensor) and isinstance(want, torch.Tensor):
                worst = max(worst, float("0") if _identical(want, got) else _worst(want, got))
    return worst


def _thread_count() -> int:
    try:
        return int(torch.get_num_threads())
    except Exception:  # noqa: BLE001 - a build that cannot report it is still a build
        return -1
