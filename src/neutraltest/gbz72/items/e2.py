"""E2 -- the random number behaviour: where a column's numbers come from.

Check: a column either draws random numbers or it does not, and the row
establishes which before it judges anything, rather than assuming either.

*Does it draw?* The column is run on the same arguments with the process-wide
generator in three states -- as the run left it, seeded with ``0``, seeded with
``12345`` -- and with the backend's own generator reseeded. A column with no
random source answers the same every time; one that moves when a global stream
moves has taken a number from that stream, and the named state is the evidence.

*A column that draws none must be said to draw none.* The verdict is recorded
positively, with the same evidence, instead of being left as an unexplained
pass: "declares no random source", plus the counts of the output. This is also
where the seed question is answered honestly rather than by inspection: taking
``torch.manual_seed`` away from the harness cannot change a column that never
asks for a number, and the row measures that -- the answers under three
different seeds are compared, and the difference is reported as zero. A column
that did become seed-dependent would be sent down the second branch instead of
being silently accepted.

*A draw must be reproducible and must not be reused.* When a random source is
found, the row shapes it three ways: the same seed must replay the same values;
two different seeds must not produce the same values (a source that ignores its
seed is worse than none, because the run looks reproducible while none is); and
repeated draws in one run must not repeat a value. The discrimination tests
inject a source by wrapping the kernel, which is how this branch is proven to
work at all -- no column of the twelve has one today.

Why not a single draw. Two correct sources disagree on any individual draw, so
"the two draws differ" is evidence of nothing; the same draw matching is
evidence of nothing either. The claim is therefore about replay and reuse
across several draws, not about one value.

Tolerance: exact (``torch.equal``). The verdict is "did the answer move at
all", and any tolerance would hide exactly the movement being looked for. The
seeded-source half compares distinct values and repetition counts, never values
against an expectation, because there is no reference distribution for a draw.

Relation to E1: E1 holds a column to being a function of its inputs, which for a
column without a random source already forbids drawing, and fails on a column
that consumes the global generator. This row is the statement the clause asks
for -- per column, *that* there is no source -- and it is what would flag a
column acquiring one; a column with a source and an internal fixed seed could
still satisfy E1, and only this row could say that its draws are reused.
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import NotVerifiable, observe

ID = "E2"
COLUMNS = None

#: Generator states the column is rerun under. ``None`` is the state the run
#: arrived in, kept first because that is the state a deployment arrives in.
GENERATOR_STATES: tuple[int | None, ...] = (None, 0, 12345)

#: Draws a reported random source is judged on: enough to show that the same
#: seed replays them and that a run does not repeat one value.
DRAWS = 4


def run(ctx: Any) -> None:
    args, kwargs = ctx.inputs(torch.float32)
    reference = observe(ctx, args, kwargs)
    moved = _seed_dependence(ctx, args, kwargs, reference)
    if moved is None:
        _no_source(ctx, args, kwargs, reference)
    else:
        _random_source(ctx, moved)


def _seed_dependence(
    ctx: Any, args: tuple, kwargs: dict, reference: tuple[torch.Tensor, ...]
) -> str | None:
    """Rerun under other generator states; name the first one that changed the answer.

    Three probes cover the three ways a kernel reaches a number nobody asked it
    to draw: the global CPU stream, the seed the harness hands out (which reaches
    a kernel only through the global stream on a CPU backend), and the backend's
    own stream on an accelerator. Each is set away from the sample's own seed,
    the call is made, and the answer compared exactly.
    """
    saved = torch.random.get_rng_state()
    try:
        for seed in GENERATOR_STATES:
            if seed is not None:
                torch.manual_seed(seed)
            if not _same(reference, observe(ctx, args, kwargs)):
                return f"the CPU generator {_describe(seed)} changed the answer"
        if _seeds_itself(ctx):
            ctx.backend.manual_seed(int(ctx.seed) + 101)
            if not _same(reference, observe(ctx, args, kwargs)):
                return f"the backend generator reseeded to {int(ctx.seed) + 101} changed the answer"
    finally:
        torch.random.set_rng_state(saved)
    return None


def _no_source(ctx: Any, args: tuple, kwargs: dict, reference: tuple[torch.Tensor, ...]) -> None:
    """The column has no random source; state that, with the evidence.

    Restated against the sample's own seed -- not just the process's -- so the
    note can name both sides of the comparison. A result that is not finite is
    not this row's finding (A1 reports it), and stopping here would hide it
    behind this row's verdict, so it is recorded and the row moves on.
    """
    torch.manual_seed(int(ctx.seed) + 1)
    if not _same(reference, observe(ctx, args, kwargs)):
        raise AssertionError(
            f"{ctx.column.name}: the answer moved when the generator was reseeded, but the "
            f"column declares no random source"
        )
    head = (
        "declares no random source: the answer is unchanged with the CPU generator "
        f"{_listed(GENERATOR_STATES)}"
        f"{' and with the backend generator reseeded' if _seeds_itself(ctx) else ''}, "
        "so taking the seed away cannot change it"
    )
    if not reference:
        ctx.note(f"{head}; the call returned no outputs")
        return
    output = reference[0]
    if output.is_floating_point() and not bool(torch.isfinite(output).all()):
        ctx.note(f"{head}; the output holds non-finite values (reported by A1)")
        return
    if output.numel() == 0:
        ctx.note(f"{head}; the sample has no elements")
        return
    ctx.note(f"{head}; {_distribution(output)}")


def _random_source(ctx: Any, moved: str) -> None:
    """A column that draws: its draws must replay under a seed and must not repeat.

    Reached only when a probe moved, so the column does draw. The source must
    then be a *seeded* one: the row's own seed is applied twice and required to
    replay the draws, a different seed is required to change them, and repeated
    draws in one run are required to differ. A column that moved for the
    generator but has no seed that replays it is refused rather than passed --
    "it moved when the generator moved and nothing replays it" is the least
    reproducible state there is -- while a column whose seed does control its
    draws but which is judged on too little data to prove anything is disclosed
    as not verifiable.
    """
    # No seed is published, so nothing replays the draw and nothing advances it:
    # the answers moved, they moved for the generator, and the run cannot show
    # that two of them would agree. That is a failure, not a disclosure.
    seeds = _seed_control(ctx)
    if seeds is None:
        raise AssertionError(
            f"{ctx.column.name}: {moved}, but the column exposes no seed this row can set, "
            f"so its draws cannot be replayed"
        )
    first, second = seeds
    if not first:
        raise NotVerifiable(
            f"{ctx.column.name}: {moved}, but the draws hold too few values to check for reuse"
        )
    if len(first) != len(second) or first == second:
        raise AssertionError(
            f"{ctx.column.name}: {moved}; two seeds gave "
            f"{'the same draws' if first == second else 'a different number of draws'}, so the "
            f"seed does not control the values"
        )
    if len(set(first)) < len(first):
        raise AssertionError(
            f"{ctx.column.name}: {moved}; {len(first)} draws under one seed repeated a value, "
            f"so the source is reused rather than advanced"
        )
    ctx.note(
        f"has a random source ({moved}); {DRAWS} draws replay under one seed, move under "
        f"another, and repeat no value"
    )


def _seed_control(ctx: Any) -> tuple[list[float], list[float]] | None:
    """The source's draws under two seeds, or ``None`` if no seed reaches it.

    The clues the context publishes are tried in turn: a generator object the
    kernel takes, then a seed argument the kernel takes. Each draw is the first
    floating element of the first output. The list is empty when the kernel
    returns nothing to judge.
    """
    generator = _generator_clue(ctx)
    if generator is not None:
        return _draw_with_generator(ctx, generator)
    return _draw_with_seed(ctx)


def _generator_clue(ctx: Any) -> str | None:
    names = _kernel_parameters(ctx)
    if "generator" in names:
        return "generator"
    return None


def _draw_with_generator(ctx: Any, name: str) -> tuple[list[float], list[float]] | None:
    def draw(seed: int) -> list[float]:
        args, kwargs = ctx.inputs(torch.float32)
        assert isinstance(kwargs, dict)
        kwargs[name] = torch.Generator().manual_seed(seed)
        return _values(observe(ctx, args, kwargs))

    return draw(int(ctx.seed)), draw(int(ctx.seed) + 1)


def _draw_with_seed(ctx: Any) -> tuple[list[float], list[float]] | None:
    if "seed" not in _kernel_parameters(ctx):
        return None

    def draw(seed: int) -> list[float]:
        args, kwargs = ctx.inputs(torch.float32)
        assert isinstance(kwargs, dict)
        kwargs["seed"] = seed
        return _values(observe(ctx, args, kwargs))

    return draw(int(ctx.seed)), draw(int(ctx.seed) + 1)


def _kernel_parameters(ctx: Any) -> tuple[str, ...]:
    import inspect

    try:
        return tuple(inspect.signature(ctx.kernel()).parameters)
    except (TypeError, ValueError):  # pragma: no cover - a builtin kernel
        return ()


def _values(produced: tuple[torch.Tensor, ...]) -> list[float]:
    """The draws behind one call: the first floating element of each output."""
    draws: list[float] = []
    for output in produced:
        if isinstance(output, torch.Tensor) and output.is_floating_point() and output.numel():
            draws.append(float(output.reshape(-1).to(torch.float64)[0]))
    return draws


def _seeds_itself(ctx: Any) -> bool:
    """Whether the backend publishes a device-side seeder this row can probe."""
    seeder = getattr(ctx.backend, "manual_seed", None)
    if not callable(seeder) or ctx.backend.device_type == "cpu":
        return False
    try:
        seeder(int(ctx.seed))
    except (NotImplementedError, AttributeError):
        return False
    return True


def _describe(seed: int | None) -> str:
    return "as the run left it" if seed is None else f"seed={seed}"


def _listed(states: tuple[int | None, ...]) -> str:
    """The generator states as one clause: "left as it was, seed=0 or seed=12345"."""
    named = [_describe(state) for state in states]
    if len(named) < 2:
        return named[0] if named else "unchanged"
    return f"{', '.join(named[:-1])} or {named[-1]}"


def _same(reference: tuple[torch.Tensor, ...], produced: tuple[torch.Tensor, ...]) -> bool:
    """Every output identical to ``reference``, a NaN matching a NaN."""
    if len(reference) != len(produced):
        return False
    for want, got in zip(reference, produced, strict=True):
        if isinstance(got, torch.Tensor) != isinstance(want, torch.Tensor):
            return False
        if not isinstance(got, torch.Tensor):
            if got != want:
                return False
            continue
        if got.shape != want.shape or got.dtype != want.dtype:
            return False
        if not torch.equal(want, got):
            return False
    return True


def _distribution(value: torch.Tensor) -> str:
    """A one-line description of what the column's output looks like."""
    flat = value.reshape(-1)
    if not flat.is_floating_point():
        return f"{flat.numel()} elements of {flat.dtype}, not floating point"
    exact = flat.to(torch.float64)
    return (
        f"{flat.numel()} elements, {int(torch.unique(exact).numel())} distinct values, "
        f"mean={float(exact.mean()):.4g}, std={float(exact.std()):.4g}"
    )
