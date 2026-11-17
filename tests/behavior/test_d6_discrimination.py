"""D6's row has discriminating power: it calls the kernel and reads the refusal.

The row is worth a clause cell only if it tells a kernel that refuses illegal input
from one that does not. These tests run it against the real kernels and against
kernels made to misbehave on purpose, and check both directions of the error: a
kernel that accepts everything and a kernel that refuses everything must both turn
the row red.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from neutraltest.backends import CpuBackend
from neutraltest.gbz72 import items as rows
from neutraltest.gbz72 import marks, reasons, runner
from neutraltest.gbz73.context import NotApplicable
from neutraltest.gbz73.features import FEATURES
from neutraltest.gbz73.inference import kernels
from neutraltest.gbz73.inference.columns import BY_NAME
from neutraltest.gbz73.inference.context import InferenceContext

#: The row under test, loaded by name so that importing it cannot be mistaken for
#: loading some other row's script into the process.
d6 = rows.module_for("D6")

D6 = next(feature for feature in FEATURES if feature.id == "D6")

#: The columns where an illegal call has something to refuse. The rest of the
#: twelve are measured too -- D6 is required of the implementation, so its row is
#: run everywhere -- but on them the row can only report that the column gives the
#: kernel no operand to reject, which ``COLUMN_LACKS_OPERAND`` records.
MARKED = tuple(marks.columns_of("D6"))
REFUSAL = (
    "flash_attention_fwd",
    "softmax_fwd",
    "rope_fwd",
    "kv_cache_fwd",
    "kv_cache_bwd",
    "rms_norm_fwd",
)
NO_OPERAND = tuple(name for name in MARKED if name not in REFUSAL)


def context(name: str) -> InferenceContext:
    return InferenceContext(column=BY_NAME[name], backend=CpuBackend(), feature_id="D6", seed=0)


def supplied(ctx: InferenceContext, values: tuple) -> tuple[str, ...]:
    """The argument names the kernel gives the positional operands it was handed."""
    return tuple(inspect.signature(ctx.kernel()).parameters)[: len(values)]


def test_the_row_is_marked_where_a_kernel_reads_what_it_is_given() -> None:
    assert tuple(marks.APPLICABILITY["D6"]) == MARKED
    assert marks.is_marked("D6", "add_fwd")
    assert marks.is_marked("D6", "kv_cache_bwd")


def test_every_marked_column_is_run_on_the_cpu() -> None:
    """No measured column is left without an answer, and none turns red.

    A column that can refuse something passes by refusing it; a column of the
    row's that hands the kernel no illegal operand answers ``na`` with its
    recorded reason. What the row may not do is leave a column unrun.
    """
    results = [
        (name, runner.run_column(D6, BY_NAME[name], CpuBackend(), seed=0)) for name in MARKED
    ]
    expected = [(name, "na" if name in NO_OPERAND else "passed") for name in MARKED]
    assert [(name, result.outcome) for name, result in results] == expected, [
        result.detail for _, result in results if result.outcome not in {"passed", "na"}
    ]


@pytest.mark.parametrize("name", MARKED)
def test_a_marked_column_says_what_it_had_to_refuse(name: str) -> None:
    """Every measured column either has a refusal to make or says it has none.

    A column that reports neither would be a cell the row cannot be judged on,
    which is exactly the gap this row exists to close.  The row answers ``na``
    itself for a backward (it is handed derivatives, not values, so nothing in
    the call can be wrong only by being the wrong shape) and for a column whose
    operands are all broadcast away; either way the answer must be a sentence
    the reason table already owns, optionally prefixed with the column's name.
    """
    result = runner.run_column(D6, BY_NAME[name], CpuBackend(), seed=0)
    if name in NO_OPERAND:
        assert result.outcome == "na"
        recorded = reasons.reason_for("D6", name)
        assert result.detail == recorded or result.detail.endswith(recorded), result.detail
        assert remember_cause(name) == reasons.COLUMN_LACKS_OPERAND
    else:
        assert result.outcome == "passed", result.detail


def remember_cause(name: str) -> str:
    """The class the reason table records, read from the row's own entry.

    ``cause_of`` answers :data:`~neutraltest.gbz72.reasons.NO_SCRIPT` for a
    marked pair -- by design, since a marked pair is one a script claims -- so
    the recorded class is read directly here.
    """
    return reasons._recorded("D6", name).cause


@pytest.mark.parametrize("name", REFUSAL)
def test_the_row_refuses_nothing_the_operator_accepts(name: str) -> None:
    """The control basket: the ordinary sample of the column returns."""
    ctx = context(name)
    values, kwargs = ctx.project(*ctx.inputs())
    d6._control(ctx, ctx.kernel(), values, kwargs, supplied(ctx, values))


@pytest.mark.parametrize("name", REFUSAL)
def test_each_marked_column_has_an_illegal_call_to_refuse(name: str) -> None:
    """A marked column whose basket is empty would be a pass bought with nothing."""
    ctx = context(name)
    values, kwargs = ctx.project(*ctx.inputs())
    defects = list(d6._defects(values, kwargs, supplied(ctx, values)))
    assert defects, f"{name} carries no defect, so the row cannot judge it"
    assert {error for _, _, error in defects} <= {ValueError, TypeError}


def test_a_kernel_that_answers_every_call_turns_the_row_red(monkeypatch) -> None:
    """An illegal axis was accepted: the row's own finding."""
    monkeypatch.setattr(kernels, "softmax", lambda x, *, dim=-1: torch.softmax(x, dim=-1))
    with pytest.raises(AssertionError, match="was accepted"):
        d6.run(context("softmax_fwd"))


def test_a_kernel_that_refuses_every_call_turns_the_row_red(monkeypatch) -> None:
    """Rejecting everything must not earn the row."""

    def refuse(*args, **kwargs):
        raise ValueError("no")

    monkeypatch.setattr(kernels, "rms_norm", refuse)
    with pytest.raises(AssertionError, match="refused its own legal sample"):
        d6.run(context("rms_norm_fwd"))


def test_a_refusal_of_the_wrong_class_turns_the_row_red(monkeypatch) -> None:
    """Raising something is not raising the check: the class is part of the claim."""

    def refuse_differently(x, *, dim=-1):
        if dim not in range(-x.ndim, x.ndim):
            raise KeyError("neither of the classes a check raises")
        return torch.softmax(x, dim=dim)

    monkeypatch.setattr(kernels, "softmax", refuse_differently)
    with pytest.raises(AssertionError, match="raised KeyError"):
        d6.run(context("softmax_fwd"))


def test_a_kernel_broken_at_its_sample_is_reported_as_such(monkeypatch) -> None:
    """A defect refused before the check is not credited to the row either way."""

    def refuse_late(index, rows):
        raise TypeError("refused, but not before the sample was accepted")

    monkeypatch.setattr(kernels, "_slot_index", refuse_late)
    with pytest.raises(AssertionError, match="refused its own legal sample"):
        d6.run(context("kv_cache_fwd"))


def test_the_operands_are_the_signature_s_of_the_kernel_under_test(monkeypatch) -> None:
    """The row reads the kernel it was handed, not a table of its own."""
    seen: list[tuple[str, ...]] = []
    original = d6._defects

    def spy(values, kwargs, names):
        seen.append(tuple(names))
        return original(values, kwargs, names)

    monkeypatch.setattr(d6, "_defects", spy)
    d6.run(context("rms_norm_fwd"))
    assert seen == [("x", "weight")]


def test_an_empty_basket_is_not_reported_as_a_pass(monkeypatch) -> None:
    """With nothing illegal to try, the column is disclosed rather than passed."""
    monkeypatch.setattr(d6, "_defects", lambda *args, **kwargs: [])
    with pytest.raises(NotApplicable) as raised:
        d6.run(context("rms_norm_fwd"))
    assert "rms_norm_fwd" in raised.value.reason


@pytest.mark.parametrize("name", ("kv_cache_fwd", "kv_cache_bwd"))
def test_both_directions_of_the_cache_carry_a_type_and_a_reach_defect(name: str) -> None:
    ctx = context(name)
    values, kwargs = ctx.project(*ctx.inputs())
    defects = list(d6._defects(values, kwargs, supplied(ctx, values)))
    assert [subject for subject, _, _ in defects] == ["type", "width"]
    assert [error for _, _, error in defects] == [TypeError, ValueError]
