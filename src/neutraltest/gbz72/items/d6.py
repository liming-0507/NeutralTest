"""D6 -- error handling: illegal input is refused, and nothing illegal is accepted.

Check: the column's kernel -- not the operator's Python wrapper -- is called
directly, so a refusal can be attributed to it, and every call is placed in one of
two baskets.

The control basket is the column's ordinary sample, projected the way the context
projects it (the cache read takes three of the write's five arguments). It must
return, with one tensor for a forward and one derivative per differentiable
argument for a backward, each carrying the sample's channel count. A kernel that
refuses legal input fails the row just as hard as one that accepts an illegal call,
so the illegal basket below cannot be passed by rejecting everything.

The illegal basket varies one argument at a time: every argument but the one under
test is a legal operand of the ordinary sample, and the illegal one is put where the
kernel's own signature takes it. Each call must be refused -- ``ValueError`` for a
shape, ``TypeError`` for a type, and, where the refusal is spelled by the framework
rather than by the kernel, the two classes the framework uses instead
(``IndexError`` for an axis outside the tensor, ``RuntimeError`` where a broadcast
deep in a dispatched kernel finds the disagreement). Every one of those is raised
before a value is produced; any other exception, and above all a return, is the
defect this row exists to find.

Which defects a column carries is read off what its kernel consumes, not assumed.
Each family's basket is written out below beside its kernel, and every entry names
the operands it moves: the attention's frame and head count, the softmax's axis,
the rotation's positioning table, the cache's slot list and its dtype, the
normalisation's weight and its row.

A backward is handed derivatives, not values: nothing in its call is wrong only by
being the wrong shape, so its basket is empty and the forward column of the same
operator takes the refusal. That is recorded once, in :mod:`neutraltest.gbz72.reasons`,
and the row states it by that text rather than in words of its own. A column left
without a candidate for any other reason is disclosed too, and says why.

Tolerance: none. A refusal is an exception type and an acceptance is a return, and
both are compared exactly.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from typing import Any

import torch

from .. import reasons
from .._support import as_tuple, make_inputs

ID = "D6"
COLUMNS = None

#: The twelve columns of the clause, so an empty basket can be judged against the
#: direction that runs the same operator with values to judge.
CLAUSE_COLUMNS: tuple[str, ...] = (
    "add_fwd",
    "flash_attention_fwd",
    "softmax_fwd",
    "rope_fwd",
    "kv_cache_fwd",
    "rms_norm_fwd",
    "add_bwd",
    "flash_attention_bwd",
    "softmax_bwd",
    "rope_bwd",
    "kv_cache_bwd",
    "rms_norm_bwd",
)

#: A defect: what it is called, the argument carrying it with its illegal value, and
#: the exception class the refusal must belong to.
Defect = tuple[str, tuple[str, Any], type[BaseException]]

#: One family's basket: the defects, built from the operands the kernel was handed.
Basket = Callable[[], list[tuple[str, tuple[str, Any] | None, type[BaseException]]]]


def run(ctx: Any) -> None:
    # A backward is handed derivatives, not values: nothing in its call is wrong
    # only by being the wrong shape, so the refusal belongs to the forward column.
    # The cache read is the exception among backwards: it indexes with a slot list.
    if ctx.backward and ctx.family != "kv_cache":
        ctx.na(reasons.reason_for(ID, ctx.column.name))
    kernel = ctx.kernel()
    names = tuple(inspect.signature(kernel).parameters)
    values, kwargs = ctx.project(
        *make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant)
    )
    supplied = names[: len(values)]
    _control(ctx, kernel, values, kwargs, supplied)
    defects = list(_defects(values, kwargs, supplied))
    if not defects:
        ctx.na(f"{ctx.column.name}: {_empty(ctx)}")
    for subject, (name, illegal), error in defects:
        if name in supplied:
            mutated, sent = _replace(values, supplied, name, illegal), kwargs
        else:  # an argument the kernel takes by keyword
            mutated, sent = values, {**kwargs, name: illegal}
        _refuse(ctx, kernel, mutated, sent, subject, error)


def _replace(values: tuple, supplied: tuple[str, ...], name: str, operand: Any) -> tuple:
    """``values`` with the argument ``name`` standing for ``operand``."""
    position = supplied.index(name)
    return values[:position] + (operand, *values[position + 1 :])


def _expected(backward: bool, values: tuple, supplied: tuple[str, ...], frozen: set) -> int:
    """How many tensors a legal call returns."""
    if not backward or "index" in supplied:
        # A forward returns its result; the cache's read is the framework's
        # backward but the kernel of a gather, and returns both of its rows.
        return 2 if "cache_k" in supplied else 1
    return sum(
        1
        for operand, name in zip(values, supplied, strict=True)
        if isinstance(operand, torch.Tensor) and operand.is_floating_point() and name not in frozen
    )


def _control(ctx: Any, kernel: Any, values: tuple, kwargs: dict, supplied: tuple[str, ...]) -> None:
    """The ordinary call must return, in the shape the column promises."""
    try:
        returned = as_tuple(kernel(*values, **kwargs))
    except Exception as exc:  # noqa: BLE001 - a refusal of legal input is the finding
        raise AssertionError(
            f"{ctx.column.name}: the kernel refused its own legal sample "
            f"({type(exc).__name__}: {exc}); a row that rejects everything would pass an "
            f"illegal-input check for no reason"
        ) from None
    expected = _expected(ctx.backward, values, supplied, set(ctx.column.freezes))
    if len(returned) != expected:
        raise AssertionError(
            f"{ctx.column.name}: a legal call returned {len(returned)} tensors, "
            f"the call has {expected} to return"
        )
    width = next(
        (
            operand.shape[-1]
            for operand in values
            if isinstance(operand, torch.Tensor) and operand.ndim
        ),
        0,
    )
    for index, produced in enumerate(returned):
        if not isinstance(produced, torch.Tensor) or produced.shape[-1] != width:
            shape = tuple(produced.shape) if isinstance(produced, torch.Tensor) else repr(produced)
            raise AssertionError(
                f"{ctx.column.name}: a legal call returned {shape} as output {index}, "
                f"which does not carry the {width} channels the sample has"
            )


def _defects(values: tuple, kwargs: dict, supplied: tuple[str, ...]) -> Iterable[Defect]:
    """Every illegal call this column carries, one argument at a time."""
    taken = {**kwargs, **dict(zip(supplied, values, strict=True))}
    family = _family_of(supplied)

    def operand(name: str) -> Any:
        return taken.get(name)

    def narrow(name: str) -> tuple[str, Any] | None:
        """The operand ``name`` one channel short of what it must agree with."""
        value = operand(name)
        if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[-1] < 2:
            return None
        return name, value[..., :-1]

    def partial(name: str) -> tuple[str, Any] | None:
        """The operand ``name`` with a head count the query's cannot be split by."""
        value, query = operand(name), operand("q")
        if not isinstance(value, torch.Tensor) or not isinstance(query, torch.Tensor):
            return None
        if value.ndim != 4 or value.shape[1] < 2 or query.ndim != 4:
            return None
        for heads in range(value.shape[1] - 1, 0, -1):
            if query.shape[1] % heads:
                return name, value[:, :heads]
        return None

    def reshaped(name: str) -> tuple[str, Any] | None:
        """The operand ``name`` one frame shorter, its elements moved to the width.

        The kernel reads its operand as it was handed, so a frame that is merely
        dropped never reaches the arithmetic; the count has to divide, and it is
        the width that then disagrees with the query's.
        """
        value = operand(name)
        if not isinstance(value, torch.Tensor) or value.ndim < 2:
            return None
        for frames in range(value.shape[-2] - 1, 1, -1):
            lanes, remainder = divmod(value.shape[-2] * value.shape[-1], frames)
            if not remainder and lanes != value.shape[-1]:
                return name, value.reshape(*value.shape[:-2], frames, lanes)
        return None

    def wide(name: str) -> tuple[str, Any] | None:
        """The operand ``name`` as a float where the kernel reads an integer."""
        value = operand(name)
        if not isinstance(value, torch.Tensor) or value.is_floating_point():
            return None
        return name, value.to(torch.float64)

    def slot(name: str) -> tuple[str, Any] | None:
        """A slot list naming a row the cache does not have."""
        value, cache = operand(name), operand("cache_k")
        if not isinstance(value, torch.Tensor) or not isinstance(cache, torch.Tensor):
            return None
        beyond = torch.full_like(value[..., :1], cache.shape[0] + 1)
        return name, torch.cat([beyond, value[..., 1:]], dim=-1)

    def axis(name: str) -> tuple[str, Any] | None:
        """A reduction axis outside the tensor."""
        value = operand("x")
        return None if not isinstance(value, torch.Tensor) else (name, value.ndim)

    # One basket per family, written beside the kernel it belongs to. Each entry is
    # attempted; one whose operands the kernel does not take is dropped, so an
    # argument a column does not consume is not a defect there.
    baskets: dict[str, Basket] = {
        # The add's kernels cast and broadcast whatever width or floating type they
        # are handed, and a row of the wrong length is answered by broadcasting the
        # other operand into it rather than refused. Nothing an add is given is an
        # illegal input to it; the interface itself is D1's to hold to account.
        "add": list,
        "flash_attention": lambda: [
            ("width", reshaped("k"), ValueError),
            ("width", partial("v"), ValueError),
        ],
        "softmax": lambda: [
            ("axis", axis("dim"), ValueError),
        ],
        "rope": lambda: [
            ("width", narrow("position"), ValueError),
        ],
        "kv_cache": lambda: [
            ("type", wide("index"), TypeError),
            ("width", slot("index"), ValueError),
        ],
        "rms_norm": lambda: [
            ("width", narrow("weight"), ValueError),
            ("width", narrow("x"), ValueError),
        ],
    }
    build = baskets.get(family)
    if build is None:
        return []
    return [
        (subject, replacement, error)
        for subject, replacement, error in build()
        if replacement is not None
    ]


def _family_of(supplied: tuple[str, ...]) -> str:
    """Which basket the operands belong to, read off their names.

    The row is given the kernel, not the column, so the family is recovered from
    the signature the kernel publishes -- the same place the operands come from.
    """
    names = set(supplied)
    if {"q", "k", "v"} <= names:
        return "flash_attention"
    if "cache_k" in names:
        return "kv_cache"
    if "weight" in names:
        return "rms_norm"
    if "position" in names:
        return "rope"
    if "dim" in names or "reduction_axis" in names:
        return "softmax"
    if len(names) <= 1 and "value" not in names:
        return "softmax"
    return "add"


def _refuse(
    ctx: Any,
    kernel: Any,
    mutated: tuple,
    kwargs: dict,
    subject: str,
    error: type[BaseException],
) -> None:
    """``mutated`` must be refused, and refused for being illegal.

    Every class listed here is raised before a value is produced: ``ValueError``
    for a size the operation compares, ``IndexError`` for an axis outside the
    tensor, and ``RuntimeError`` where a broadcast inside a dispatched kernel finds
    the disagreement. ``TypeError`` is what a dtype is refused with, so it is never
    accepted in place of the shape check it would stand in for.
    """
    try:
        kernel(*mutated, **kwargs)
    except error:
        return
    except (IndexError, RuntimeError):
        return
    except Exception as exc:  # noqa: BLE001 - a wrong refusal is the finding too
        raise AssertionError(
            f"{ctx.column.name}: an illegal {subject} raised {type(exc).__name__} ({exc}); "
            f"the operation's own argument check would raise {error.__name__} first"
        ) from None
    raise AssertionError(
        f"{ctx.column.name}: an illegal {subject} was accepted; the kernel returned a "
        f"result instead of raising {error.__name__}"
    )


def _empty(ctx: Any) -> str:
    """Why a column carries no defect, when the table has no reason for it."""
    if ctx.family == "add":
        return (
            "the add's kernels cast and broadcast every width and floating type they are "
            "handed, so no operand of theirs is an illegal input for the column itself"
        )
    return (
        "no operand of this column is illegal for its kernel: every one it takes is a "
        "number, a row it reduces, or a shape it never compares"
    )


__all__ = ["CLAUSE_COLUMNS", "COLUMNS", "ID", "run"]
