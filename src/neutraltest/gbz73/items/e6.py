"""E6 -- the same work split across devices.

Reported per backend, and both halves really execute.

*One device* is the ordinary case and is not skipped. The clause is about work
that arrives in more than one piece -- a batch produced in two shards, a stack of
groups written by two calls -- so the row computes the sample in one go and then
computes it again as two pieces, each in its own device buffer. The two answers
must be bit-identical. That comparison is not about a second card: it catches an
operator that reads past its own shard, that assumes a length or an alignment
only the whole sample gives, or that carries a state from one call to the next,
and it does it with the hardware in the room.

What the split cuts depends on the operator, because two things count as "one
more piece" and only one of them is the batch:

* a *batched* argument (``a`` of a grouped contraction, the rows of a stack)
  cuts along dim 0 and the two answers are joined along dim 0;
* a *grouped* argument (a call that takes its groups as a table) cuts the group
  table: the first call gets the leading groups, the second the rest with its
  rows re-based, and the two answers are joined along dim 0. This is the split a
  caller actually makes when a batch of groups is too large for one launch, and
  it is the one where an operator that counts its groups from zero fails.

*Two devices or more*, when the backend has them, is the clause read literally:
the same inputs go to device 0 and device 1 and the two results have to agree.
Merely asking whether a second device exists would pass on a build where device 1
is advertised and then computes the wrong thing, which is the deployment fault
this row is here to find. The number of devices is recorded in the note, because
"agree across two cards" and "agree across eight" are not the same evidence.

Tolerance: none. Both halves compare the device against itself on work that is
the same on purpose, so a disagreement is never a rounding question.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from ..context import RunContext
from ..ops import Args, Kwargs

ID = "E6"
OPS = None


def run(ctx: RunContext) -> None:
    args, kwargs = ctx.inputs(torch.float32)
    whole = ctx.run(args, kwargs)
    count = ctx.backend.device_count()
    if count >= 2:
        _across(ctx, count, args, kwargs, whole)
        return
    # A sample that carries a group table is asking to be cut by groups: its
    # leading operand is indexed by those bounds, not by a batch dimension, so
    # halving the rows would leave the two halves describing different groups.
    if _group_table(args) is not None or not _split(ctx, args, kwargs, whole, count):
        _split_groups(ctx, args, kwargs, whole, count)


# -- one device: the work arrives in pieces -------------------------------


def _split(ctx: RunContext, args: Args, kwargs: Kwargs, whole: torch.Tensor, count: int) -> bool:
    """Cut the batched operand along dim 0 and join the answers."""
    found = next(
        (
            (position, argument)
            for position, argument in enumerate(args)
            if isinstance(argument, torch.Tensor)
            and argument.is_floating_point()
            and argument.dim() >= 2
            and argument.shape[0] >= 2
        ),
        None,
    )
    if found is None:
        return False
    position, tensor = found
    rows = -(-tensor.shape[0] // 2)
    pieces = (tensor[:rows], tensor[rows:])
    if pieces[1].shape[0] == 0:
        return False
    halves = [ctx.run(_replaced(args, position, piece), kwargs) for piece in pieces]
    joined = torch.cat([_host(half) for half in halves], dim=0)
    if joined.shape != whole.shape:
        raise AssertionError(
            f"{ctx.op.name}: the two pieces joined into {tuple(joined.shape)}, "
            f"the whole sample is {tuple(whole.shape)}"
        )
    if not torch.equal(joined, _host(whole)):
        raise AssertionError(
            f"{ctx.op.name}: split over 2 buffers the result differs from the whole sample "
            f"by up to {_worst(joined, _host(whole)):.3e}, so the answer depends on how the "
            f"work was cut up"
        )
    ctx.note(
        f"single device, split across 2 device buffers: {tuple(tensor.shape)} cut into "
        f"{tuple(pieces[0].shape)} + {tuple(pieces[1].shape)}, joined result identical to "
        f"the whole sample; {ctx.backend.name} exposes {count} device"
    )
    return True


def _split_groups(
    ctx: RunContext, args: Args, kwargs: Kwargs, whole: torch.Tensor, count: int
) -> None:
    """Cut the group table: the leading groups, then the rest re-based at zero."""
    # The table is ``args[n]`` for a grouped call ``args[n - 2], args[n - 1],
    # args[n]``; ``_operands`` walks the trailing offsets so the guess cannot
    # land on an unrelated integer argument.
    found = next(
        (
            (args[index - 2], args[index - 1], args[index])
            for index, argument in enumerate(args)
            if isinstance(argument, torch.Tensor)
            and argument.dim() == 1
            and not argument.is_floating_point()
            and index >= 2
            and isinstance(args[index - 1], torch.Tensor)
            and args[index - 1].dim() == 3
            and args[index - 1].shape[0] >= 2
        ),
        None,
    )
    if found is None:
        _refuse(ctx)
        return
    left, right, table = found
    groups = right.shape[0]
    bounds = [int(value) for value in table.tolist()]
    if len(bounds) != groups + 1 or bounds[0] != 0:
        _refuse(ctx)
        return
    middle = groups // 2
    if bounds[middle] <= bounds[0] or bounds[-1] <= bounds[middle]:
        _refuse(ctx)
        return
    seams = (
        (bounds[: middle + 1], 0, bounds[middle], right[:middle]),
        (
            [value - bounds[middle] for value in bounds[middle:]],
            bounds[middle],
            bounds[-1],
            right[middle:],
        ),
    )
    halves = []
    for rows, low, high, chosen in seams:
        shard = _replaced(args, _position(args, table), torch.tensor(rows, dtype=table.dtype))
        shard = _replaced(shard, _position(args, right), chosen)
        shard = _replaced(shard, _position(args, left), left[low:high])
        halves.append(ctx.run(shard, kwargs))
    joined = torch.cat([_host(half) for half in halves], dim=0)
    if joined.shape != whole.shape or not torch.equal(joined, _host(whole)):
        raise AssertionError(
            f"{ctx.op.name}: {groups} groups in two launches differ from one launch by up to "
            f"{_worst(joined, _host(whole)):.3e}, so the answer depends on how the groups "
            f"were cut up"
        )
    head, tail = len(seams[0][0]) - 1, len(seams[1][0]) - 1
    ctx.note(
        f"single device, groups split across 2 device buffers: {groups} groups as "
        f"{head} + {tail}, second launch re-based at row {bounds[middle]}, joined result "
        f"identical to one launch; {ctx.backend.name} exposes {count} device"
    )


def _group_table(args: Args) -> torch.Tensor | None:
    """The ``n + 1``-entry offset table a grouped call passes, if it passes one."""
    for argument in args:
        if (
            isinstance(argument, torch.Tensor)
            and argument.dim() == 1
            and not argument.is_floating_point()
        ):
            return argument
    return None


def _refuse(ctx: RunContext) -> None:
    ctx.na(
        f"{ctx.op.name}: the sample has no batched operand and no group table to cut, "
        f"so there is no second piece to compare against"
    )


def _position(args: Args, needle: object) -> int:
    return next(index for index, argument in enumerate(args) if argument is needle)


def _replaced(args: Args, position: int, piece: object) -> Args:
    """``args`` with ``position`` swapped for ``piece``, in its own device buffer."""
    return tuple(piece if index == position else argument for index, argument in enumerate(args))


def _host(value: Any) -> Any:
    """A tensor back on the host for comparison; anything else untouched."""
    return value.detach().to("cpu") if isinstance(value, torch.Tensor) else value


# -- two devices or more: the clause read literally -----------------------


def _across(ctx: RunContext, count: int, args: Args, kwargs: Kwargs, whole: torch.Tensor) -> None:
    """The same inputs on device 0 and on device 1."""
    second = _on(ctx, 1, args, kwargs)
    if second.shape != whole.shape:
        raise AssertionError(
            f"{ctx.op.name}: device 0 returned {tuple(whole.shape)}, "
            f"device 1 returned {tuple(second.shape)}"
        )
    if not torch.equal(second, whole):
        raise AssertionError(
            f"{ctx.op.name}: device 1 differs from device 0 by up to "
            f"{_worst(second, whole):.3e} on the same inputs"
        )
    ctx.note(f"{count} devices: device 1 identical to device 0 on the same inputs")


def _on(ctx: RunContext, index: int, args: Args, kwargs: Kwargs) -> torch.Tensor:
    """Run ``ctx.op`` on device ``index`` and bring the result back."""
    if index != ctx.device_index:
        return replace(ctx, device_index=index).run(args, kwargs)
    return ctx.run(args, kwargs)


def _worst(got: torch.Tensor, want: torch.Tensor) -> float:
    if got.shape != want.shape:
        return float("nan")
    if got.numel() == 0:
        return 0.0
    return float((got.double() - want.double()).abs().max())
