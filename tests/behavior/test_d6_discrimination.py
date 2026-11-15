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

#: The columns the row leaves marked, and the ones it discloses instead.
MARKED = (
    "flash_attention_fwd",
    "softmax_fwd",
    "rope_fwd",
    "kv_cache_fwd",
    "kv_cache_bwd",
    "rms_norm_fwd",
)
UNMARKED = ("add_fwd", "add_bwd", "flash_attention_bwd", "softmax_bwd", "rope_bwd", "rms_norm_bwd")


def context(name: str) -> InferenceContext:
    return InferenceContext(column=BY_NAME[name], backend=CpuBackend(), feature_id="D6", seed=0)


def supplied(ctx: InferenceContext, values: tuple) -> tuple[str, ...]:
    """The argument names the kernel gives the positional operands it was handed."""
    return tuple(inspect.signature(ctx.kernel()).parameters)[: len(values)]


def test_the_row_is_marked_where_a_kernel_reads_what_it_is_given() -> None:
    assert tuple(marks.APPLICABILITY["D6"]) == MARKED
    assert not marks.is_marked("D6", "add_fwd")
    assert marks.is_marked("D6", "kv_cache_bwd")


def test_every_marked_column_passes_on_the_cpu() -> None:
    results = [
        (name, runner.run_column(D6, BY_NAME[name], CpuBackend(), seed=0)) for name in MARKED
    ]
    assert [(name, result.outcome) for name, result in results] == [
        (name, "passed") for name in MARKED
    ], [result.detail for _, result in results if result.outcome != "passed"]


@pytest.mark.parametrize("name", UNMARKED)
def test_an_unmarked_column_carries_a_recorded_reason(name: str) -> None:
    result = runner.run_column(D6, BY_NAME[name], CpuBackend(), seed=0)
    assert result.outcome == "na"
    assert result.detail == reasons.reason_for("D6", name)
    assert reasons.cause_of("D6", name) in set(reasons.CAUSES) - {reasons.NO_SCRIPT}


@pytest.mark.parametrize("name", MARKED)
def test_the_row_refuses_nothing_the_operator_accepts(name: str) -> None:
    """The control basket: the ordinary sample of the column returns."""
    ctx = context(name)
    values, kwargs = ctx.project(*ctx.inputs())
    d6._control(ctx, ctx.kernel(), values, kwargs, supplied(ctx, values))


@pytest.mark.parametrize("name", MARKED)
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
