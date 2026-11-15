"""A7's row reads sign bits, and these tests prove it does.

The row exists because ``torch.equal(-0.0, 0.0)`` is ``True``: a kernel that
answers ``+0`` where the reference has ``-0`` passes every value comparison there
is.  Two directions are tested here.  A kernel that changes only the sign must
turn the row red -- that is the defect the row is for.  And the comparison itself
must go red when it stops reading the sign, because a check that cannot fail is
worse than no check: it reports "the kernel keeps signed zeros" on the strength of
having looked at values.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from neutraltest.backends import CpuBackend
from neutraltest.gbz72 import marks, reasons, runner
from neutraltest.gbz72.items import a7
from neutraltest.gbz73.context import NotApplicable
from neutraltest.gbz73.features import FEATURES
from neutraltest.gbz73.inference import kernels
from neutraltest.gbz73.inference.columns import BY_NAME
from neutraltest.gbz73.inference.context import InferenceContext

A7 = next(feature for feature in FEATURES if feature.id == "A7")

#: The columns the row claims.
MARKED = (
    "add_fwd",
    "softmax_fwd",
    "rope_fwd",
    "kv_cache_fwd",
    "rms_norm_fwd",
    "kv_cache_bwd",
    "rms_norm_bwd",
)

#: The columns it leaves alone, and the class of reason it gives for each.
UNMARKED = (
    "flash_attention_fwd",
    "flash_attention_bwd",
    "add_bwd",
    "softmax_bwd",
    "rope_bwd",
)


def context(name: str) -> InferenceContext:
    return InferenceContext(
        column=BY_NAME[name], backend=CpuBackend(), feature_id="A7", seed=7, variant=""
    )


def outcome(name: str) -> runner.Result:
    return runner.run_column(A7, BY_NAME[name], CpuBackend(), seed=7)


def positive_zero(value: Any) -> Any:
    """A copy of ``value`` whose zeros are all ``+0``; nonzero values are untouched."""
    if isinstance(value, tuple):
        return tuple(positive_zero(item) for item in value)
    out = value.clone()
    out[torch.signbit(out)] = 0.0
    return out


def negative_zero(value: Any) -> Any:
    """A copy of ``value`` whose zeros are all ``-0``; nonzero values are untouched."""
    if isinstance(value, tuple):
        return tuple(negative_zero(item) for item in value)
    out = value.clone()
    out[out == 0] = -0.0
    return out


# --------------------------------------------------------------------------
# what the row claims, and what it does not
# --------------------------------------------------------------------------
def test_the_row_claims_the_columns_whose_zero_can_carry_a_sign() -> None:
    assert tuple(marks.APPLICABILITY["A7"]) == MARKED
    assert a7.COLUMNS == MARKED
    assert marks.is_marked("A7", "add_fwd") is True
    assert marks.is_marked("A7", "flash_attention_fwd") is False


def test_every_claimed_column_passes_on_the_cpu() -> None:
    results = [outcome(name) for name in MARKED]
    assert [(result.column, result.outcome) for result in results] == [
        (name, "passed") for name in MARKED
    ], [result.detail for result in results if result.outcome != "passed"]


@pytest.mark.parametrize("name", UNMARKED)
def test_an_unclaimed_column_says_why_and_does_not_run(name: str) -> None:
    result = outcome(name)
    assert result.outcome == "na"
    assert result.detail == reasons.reason_for("A7", name)
    assert reasons.cause_of("A7", name) in reasons.CAUSES
    assert reasons.cause_of("A7", name) != reasons.NO_SCRIPT


def test_the_row_is_a_partial_row_not_an_empty_one() -> None:
    """A7 used to claim nothing at all; the claim now lives as a partial cause."""
    assert "A7" not in reasons.ROW_REASONS
    assert "A7" in reasons.PARTIAL_REASONS
    assert reasons.cause_of("A7", "flash_attention_fwd") == reasons.COLUMN_LACKS_CAPABILITY


def test_the_partial_reason_does_not_stand_in_for_a_script_that_is_missing() -> None:
    """The unclaimed columns are a judgement about the operators, not work not done."""
    for name in UNMARKED:
        assert reasons.cause_of("A7", name) != reasons.NO_SCRIPT
    for name in MARKED:
        # A claimed pair has no recorded cause; asking anyway answers "no script",
        # which is the table saying the work would be owed if nobody claimed it.
        assert name not in marks.APPLICABILITY["A7"] or reasons.cause_of("A7", name) == (
            reasons.NO_SCRIPT
        )


# --------------------------------------------------------------------------
# the defect the row exists for: same values, different bits
# --------------------------------------------------------------------------
def test_a_sign_only_change_compares_equal_on_values() -> None:
    """Ground the whole row: the change it catches is invisible to ``==`` and ``equal``."""
    want = torch.tensor([[-0.0, 0.0, 1.5]])
    got = positive_zero(want)
    assert torch.equal(got, want)
    assert bool((got == want).all())
    assert not torch.equal(torch.signbit(got), torch.signbit(want))


#: The two claimed columns with no ``-0`` of their own to flush: softmax divides a
#: zero by a positive sum, and the scale gradient at a zero row is ``+0``.  They are
#: claimed for the other half of the contract (no sign may be invented), and the
#: flush test below would be measuring nothing on them.
WITHOUT_A_NEGATIVE_ZERO = ("softmax_fwd", "rms_norm_bwd")


@pytest.mark.parametrize("name", [name for name in MARKED if name not in WITHOUT_A_NEGATIVE_ZERO])
def test_a_kernel_that_drops_the_sign_of_a_zero_turns_the_row_red(name: str) -> None:
    """Flush ``-0`` to ``+0`` inside the kernel: the values are untouched."""
    original = context(name).kernel()
    family = _kernel_attribute(name)

    def flushed(*args: Any, **kwargs: Any) -> Any:
        return positive_zero(original(*args, **kwargs))

    kernels.__dict__[family] = flushed
    try:
        with pytest.raises((AssertionError, NotApplicable)):
            a7.run(context(name))
    finally:
        kernels.__dict__[family] = original


def test_a_kernel_that_invents_a_negative_zero_turns_the_row_red() -> None:
    """The other direction: a sign where the reference has none is a defect too."""

    original = kernels.add

    def signed(*args: Any, **kwargs: Any) -> Any:
        return negative_zero(original(*args, **kwargs))

    kernels.add = signed
    try:
        with pytest.raises(AssertionError, match="-0|negative"):
            a7.run(context("add_fwd"))
    finally:
        kernels.add = original


def test_a_kernel_that_only_touches_nonzeros_is_not_flagged() -> None:
    """The control: a no-op wrapper on the nonzero lanes leaves every sign alone."""
    original = kernels.rope

    def rewrapped(*args: Any, **kwargs: Any) -> torch.Tensor:
        out = original(*args, **kwargs)
        return torch.where(out == 0, out, out)

    kernels.rope = rewrapped
    try:
        a7.run(context("rope_fwd"))
    finally:
        kernels.rope = original


def test_a_kernel_that_ignores_eps_turns_the_row_red() -> None:
    """``eps = 0`` on a zero row gives ``nan`` on both sides; only finiteness sees it."""
    original = kernels.rms_norm

    def no_eps(x: torch.Tensor, weight: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
        return original(x, weight, eps=0.0)

    kernels.rms_norm = no_eps
    try:
        with pytest.raises(AssertionError, match="non-finite"):
            a7.run(context("rms_norm_fwd"))
    finally:
        kernels.rms_norm = original


def _kernel_attribute(name: str) -> str:
    """The ``kernels`` attribute the column actually calls."""
    ctx = context(name)
    for attribute in dir(kernels):
        if getattr(kernels, attribute) is ctx.kernel():
            return attribute
    raise AssertionError(f"{name} calls a kernel that is not in the kernels module")


# --------------------------------------------------------------------------
# a comparison that stops reading the sign must go red
# --------------------------------------------------------------------------
def test_a_blinded_signbit_does_not_reach_the_rows_verdict() -> None:
    """The comparison reads the sign bit, not a name that can be pointed elsewhere.

    ``signbit`` is the function the check *means*; a check that called it would be
    one monkeypatch away from agreeing with anything.  Here the flushed output is
    still refused with ``signbit`` blinded, because the bit is what is read.
    """
    ctx = context("add_fwd")
    want = torch.full((8, 8), -0.0, dtype=torch.float64)
    flushed = positive_zero(want)
    original = torch.signbit
    torch.signbit = lambda tensor: torch.zeros(tensor.shape, dtype=torch.bool)
    try:
        with pytest.raises(AssertionError, match="-0|negative"):
            a7._compare_signs(ctx, flushed, want, "a flushed zero")
    finally:
        torch.signbit = original


def test_every_claimed_column_reads_its_sign_through_the_judge() -> None:
    """A column that never reaches the judge cannot fail a sign check.

    Blinding :func:`a7._sign_mask` removes the sign from every verdict the row
    makes.  A column that still passes is one that decided its answer without
    asking -- an early ``return``, or a comparison of its own.  The claim is
    proven by the *stub's call count*: every marked column must call it at least
    once, and none of them is allowed to pass without doing so.
    """
    calls: dict[str, int] = {}
    real = a7._sign_mask

    def counted(tensor: torch.Tensor) -> torch.Tensor:
        calls[_CURRENT[0]] = calls.get(_CURRENT[0], 0) + 1
        return real(tensor)

    try:
        a7._sign_mask = counted
        for name in MARKED:
            _CURRENT[0] = name
            _passes(name)
    finally:
        a7._sign_mask = real
    silent = [name for name in MARKED if not calls.get(name)]
    assert not silent, f"{silent} reached a verdict without reading a sign"


_CURRENT: list[str] = [""]


def _passes(name: str) -> bool:
    """True when the row answers "passed" for this column."""
    try:
        a7.run(context(name))
    except (AssertionError, NotApplicable):
        return False
    return True
