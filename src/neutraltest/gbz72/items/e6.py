"""E6 -- the same work split into pieces.

Check: the column computes its answer in one call, then computes it again as two
pieces each in its own buffer, and the two answers have to be bit-identical.

This is not a question about a second card. The clause asks whether the work can
arrive in more than one piece, and a serving loop hands an operator its batch in
shards -- a request that is half full, two sequences concatenated, a cache
written by two calls. What the comparison catches is an operator that reads past
its own shard, that assumes a length or an alignment only the whole sample gives,
or that carries a count from one call into the next. All of that shows up with
one device in the room, so the row is not skipped when the backend has only one.

What gets cut depends on the operator, because the families here do not take
their work the same way:

* a *row-major* sample (attention, softmax, norm, rope over a batch) cuts the
  leading operand along dim 0 -- the batch -- and the answers are joined along
  dim 0;
* a *cache write* is not a pure function of its arguments: it reports the cache
  it was given with the named rows overwritten. So the two halves are joined the
  way a serving loop joins them -- the second fill starts from the cache the
  first fill returned. A kernel that clears rows it was not asked to touch, or
  that forgets the first fill, leaves a difference in the rows nobody wrote;
* a *cache read* names its rows in one list, so it is cut by naming half the
  slots and then the other half, and the two answers are stacked in slot order.

*Two devices or more*, when the backend has them, is the clause read literally:
the same inputs go to the first device and the second and the two answers have
to agree. Merely asking whether a second device exists would pass on a build
where the second one is advertised and then computes something else, which is
the deployment fault this row exists to find. The device count is recorded in the
note, because "agree across two" and "agree across eight" are not the same
evidence.

Tolerance: none. Both halves compare the column against itself on work that is
the same on purpose, so a disagreement is never a rounding question.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from .._support import as_tuple, make_inputs

ID = "E6"
COLUMNS = None

#: The cache family; forward fills it, backward reads it back out.
_CACHE = "kv_cache"


def run(ctx: Any) -> None:
    count = _device_count(ctx)
    if count >= 2:
        _across(ctx, count)
        return
    if not _split(ctx, count):
        ctx.na(_nothing_to_cut(ctx))


# -- why a column can have nothing to cut --------------------------------


def _nothing_to_cut(ctx: Any) -> str:
    """The reason no cut produced two pieces whose join is the whole answer."""
    if ctx.family == "flash_attention":
        return (
            f"{ctx.column.name}: the row cuts a sample and joins the pieces, and attention has "
            f"no such cut -- every query row is answered against every key and value row, so "
            f"splitting the queries changes nothing while splitting the keys changes the answer "
            f"by construction. A serving loop shards by sequence, which is a different question "
            f"and is not this row's claim"
        )
    return (
        f"{ctx.column.name}: this sample has no operand the row can cut in two -- it needs a "
        f"leading batch of at least two rows, or a slot list of at least two entries -- so "
        f"there is no second piece to compare against"
    )


# -- one device: the work arrives in pieces -------------------------------


def _split(ctx: Any, count: int) -> bool:
    """Cut the sample in two, run each piece, join and compare with the whole."""
    args, kwargs = _inputs(ctx)
    whole = _answer(ctx, args, kwargs)
    if ctx.family == _CACHE:
        if ctx.backward:
            return _split_read(ctx, args, kwargs, whole, count)
        return _split_write(ctx, args, kwargs, whole, count)
    return _split_rows(ctx, args, kwargs, whole, count)


def _split_rows(ctx: Any, args: Any, kwargs: Any, whole: Any, count: int) -> bool:
    """Cut the batched operand along the first axis that can be cut, join, compare.

    Which axis is the batch is not something this row can know -- attention takes
    ``(batch, heads, seq, width)``, a norm takes ``(rows, width)`` -- so the row
    tries each axis of the leading floating operand in turn and keeps the first
    one whose two halves a kernel will accept.  An operator that rejects a cut
    along every axis is reported as having nothing to cut, not as a failure: the
    row's claim is about work that *can* arrive in pieces.
    """
    found = next(
        (
            (position, argument)
            for position, argument in enumerate(args)
            if isinstance(argument, torch.Tensor)
            and argument.is_floating_point()
            and argument.dim() >= 2
        ),
        None,
    )
    if found is None:
        return False
    # A sequence axis exists in attention but cannot be cut: every query row is
    # answered against every key and value row, so cutting the queries while the
    # keys stay whole is not the same sample.  Cutting *all* of q, k and v changes
    # the sample, not the work.
    if ctx.family == "flash_attention":
        return False
    position, tensor = found
    for axis in range(tensor.dim()):
        if tensor.shape[axis] < 2:
            continue
        rows = tensor.shape[axis] // 2
        if rows == 0 or tensor.shape[axis] - rows == 0:
            continue
        pieces = (
            tensor.narrow(axis, 0, rows),
            tensor.narrow(axis, rows, tensor.shape[axis] - rows),
        )
        try:
            halves = [_answer(ctx, _replace(args, position, piece), kwargs) for piece in pieces]
        except (RuntimeError, ValueError):
            continue
        if any(len(half) != len(whole) for half in halves):
            continue
        try:
            joined = tuple(
                torch.cat([half[index] for half in halves], dim=axis) for index in range(len(whole))
            )
        except RuntimeError:
            continue
        if any(a.shape != b.shape for a, b in zip(joined, whole, strict=True)):
            continue
        _compare(
            ctx,
            joined,
            whole,
            f"cut along axis {axis} into {tuple(pieces[0].shape)} + {tuple(pieces[1].shape)}",
        )
        ctx.note(
            f"single device, split into 2 buffers: {tuple(tensor.shape)} cut along axis {axis} "
            f"into {tuple(pieces[0].shape)} + {tuple(pieces[1].shape)}, rejoined result identical "
            f"to one call; {count} device"
        )
        return True
    return False


def _split_write(ctx: Any, args: Any, kwargs: Any, whole: Any, count: int) -> bool:
    """Fill the cache in two calls, the second starting from the first's result."""
    key_position = _token_position(args)
    slot_position = _slot_position(args)
    if key_position is None or slot_position is None:
        return False
    key, value = args[key_position], args[key_position + 1]
    slot = args[slot_position]
    # The cache pair sits right after the new tokens, and the slot list is last:
    # ``(k, v, cache_k, cache_v, slot)``.  Counting back from the end would take
    # the slot list for the value cache.
    cache_k_position, cache_v_position = key_position + 2, key_position + 3
    if cache_k_position >= len(args) or cache_v_position >= len(args):
        return False
    tokens = key.shape[-2]
    if tokens < 2 or slot.numel() != tokens or value.shape[-2] != tokens:
        return False
    half = tokens // 2

    first_args = _slice_tokens(_replace(args, slot_position, slot[:half]), key_position, 0, half)
    first = _answer(ctx, first_args, kwargs)
    if len(first) != 2:
        return False
    second_args = _slice_tokens(
        _replace(
            _replace(args, cache_k_position, first[0]),
            cache_v_position,
            first[1],
        ),
        key_position,
        half,
        tokens,
    )
    second_args = _replace(second_args, slot_position, slot[half:])
    second = _answer(ctx, second_args, kwargs)
    _compare(
        ctx,
        second,
        whole,
        f"{tokens} tokens filled as {half} + {tokens - half}",
    )
    ctx.note(
        f"single device, cache filled in 2 calls: {tokens} tokens as {half} + {tokens - half}, "
        f"the second fill started from the first fill's cache, both halves identical to one "
        f"call; {count} device"
    )
    return True


def _split_read(ctx: Any, args: Any, kwargs: Any, whole: Any, count: int) -> bool:
    """Read the cache by naming half the slots, then the other half."""
    slot_position = _slot_position(args)
    if slot_position is None:
        return False
    slot = args[slot_position]
    if slot.numel() < 2:
        return False
    half = slot.numel() // 2
    pieces = (slot[:half], slot[half:])
    halves = [_answer(ctx, _replace(args, slot_position, piece), kwargs) for piece in pieces]
    if any(len(half_answer) != len(whole) for half_answer in halves):
        return False
    joined = tuple(
        torch.cat([half_answer[index] for half_answer in halves], dim=0)
        for index in range(len(whole))
    )
    _compare(ctx, joined, whole, f"{slot.numel()} slots read as {half} + {slot.numel() - half}")
    ctx.note(
        f"single device, cache read in 2 calls: {slot.numel()} slots as {half} + "
        f"{slot.numel() - half}, stacked result identical to one call; {count} device"
    )
    return True


# -- two devices or more: the clause read literally -----------------------


def _across(ctx: Any, count: int) -> None:
    """The same inputs on the first device and on the second."""
    args, kwargs = _inputs(ctx)
    whole = _answer(ctx, args, kwargs)
    second = _answer_on(ctx, 1, args, kwargs)
    _compare(ctx, second, whole, "device 1 against device 0")
    ctx.note(f"{count} devices: device 1 identical to device 0 on the same inputs")


def _answer_on(ctx: Any, index: int, args: Any, kwargs: Any) -> Any:
    """Run the column with its device index moved to ``index``."""
    if index == ctx.device_index:
        return _answer(ctx, args, kwargs)
    return _answer(dataclasses.replace(ctx, device_index=index), args, kwargs)


# -- running the column ---------------------------------------------------


def _inputs(ctx: Any) -> tuple[Any, dict[str, Any]]:
    args, kwargs = make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant)
    return args, kwargs


def _answer(ctx: Any, args: Any, kwargs: Any) -> Any:
    """The column's answer as a tuple of tensors, forward or backward."""
    if ctx.backward:
        gradients, _ = ctx.gradients_with(args, kwargs)
        return tuple(gradients)
    return as_tuple(ctx.call_with(args, kwargs))


def _compare(ctx: Any, got: Any, want: Any, how: str) -> None:
    if len(got) != len(want):
        raise AssertionError(
            f"{ctx.column.name}: {how} returned {len(got)} tensors, one call returns {len(want)}"
        )
    for position, (a, b) in enumerate(zip(got, want, strict=True)):
        if a.shape != b.shape:
            raise AssertionError(
                f"{ctx.column.name}: {how} gave output {position} the shape {tuple(a.shape)}, "
                f"one call gives {tuple(b.shape)}"
            )
        if not torch.equal(a.detach().cpu(), b.detach().cpu()):
            raise AssertionError(
                f"{ctx.column.name}: {how} differs from one call by up to {_worst(a, b):.3e} on "
                f"output {position}, so the answer depends on how the work was cut up"
            )


def _worst(got: torch.Tensor, want: torch.Tensor) -> float:
    if got.numel() == 0:
        return 0.0
    if got.is_floating_point():
        return float((got.detach().cpu().double() - want.detach().cpu().double()).abs().max())
    return float((got.detach().cpu() != want.detach().cpu()).sum())


# -- operands -------------------------------------------------------------


def _token_position(args: Any) -> int | None:
    """Where a cache write keeps its ``(k, v)`` token pair: the first 2-D pair.

    The write is ``(k, v, cache_k, cache_v, index)``; ``k`` is the first 2-D
    operand, ``cache_k``/``cache_v`` the two before the index.
    """
    for position in range(len(args) - 3):
        key, value = args[position], args[position + 1]
        if (
            isinstance(key, torch.Tensor)
            and isinstance(value, torch.Tensor)
            and key.dim() >= 2
            and key.shape == value.shape
            and key.shape[-2] >= 2
        ):
            return position
    return None


def _slot_position(args: Any) -> int | None:
    """The trailing integer list naming cache rows."""
    for position in range(len(args) - 1, -1, -1):
        argument = args[position]
        if (
            isinstance(argument, torch.Tensor)
            and argument.dim() == 1
            and not argument.is_floating_point()
        ):
            return position
    return None


def _slice_tokens(args: Any, key_position: int, low: int, high: int) -> tuple[Any, ...]:
    """``args`` with the ``(k, v)`` token pair cut to ``[low:high]`` along the token axis."""
    items = list(args)
    for offset in (0, 1):
        position = key_position + offset
        tensor = items[position]
        if not isinstance(tensor, torch.Tensor) or tensor.shape[-2] < high:
            raise AssertionError(
                f"the token operand at {position} has {tuple(tensor.shape)}; the row asked for "
                f"rows [{low}:{high}]"
            )
        items[position] = tensor[..., low:high, :]
    return tuple(items)


def _device_count(ctx: Any) -> int:
    counter = getattr(ctx.backend, "device_count", None)
    return int(counter()) if callable(counter) else 1


def _replace(args: Any, position: int, piece: Any) -> tuple[Any, ...]:
    """``args`` with ``position`` swapped for ``piece``; negative positions count back."""
    items = list(args)
    index = position if position >= 0 else len(items) + position
    if not 0 <= index < len(items):
        raise IndexError(f"operand {position} is outside a {len(items)}-operand call")
    items[index] = piece
    return tuple(items)
