"""Tolerance table behaviour."""

from __future__ import annotations

import pytest
import torch

from torch_mbt.tolerance import EXACT, KIND_SCALE, KINDS, Tolerance, describe, tolerance_for


def test_elementwise_budget_matches_dtype_table() -> None:
    tol = tolerance_for(torch.float32)
    assert tol.rtol == pytest.approx(1.3e-6)
    assert tol.atol == pytest.approx(1e-5)
    assert not tol.is_exact


def test_budget_grows_with_arithmetic_depth() -> None:
    elementwise = tolerance_for(torch.float16, "elementwise")
    transcendental = tolerance_for(torch.float16, "transcendental")
    convolution = tolerance_for(torch.float16, "convolution")
    assert elementwise.rtol < transcendental.rtol < convolution.rtol
    assert tolerance_for(torch.float16, "matmul").rtol == pytest.approx(elementwise.rtol * 4)


def test_ordering_of_matmul_and_transcendental_depends_on_kind_scale_only() -> None:
    assert KIND_SCALE["matmul"] == 4.0
    assert KIND_SCALE["attention"] == 16.0
    assert tolerance_for(torch.float32, "attention").rtol == pytest.approx(1.3e-6 * 16)


def test_exact_kinds_are_bit_exact() -> None:
    for kind in ("exact", "indexing", "comparison"):
        tol = tolerance_for(torch.float32, kind)
        assert tol == EXACT
        assert tol.is_exact


def test_integer_and_bool_are_always_exact() -> None:
    for dtype in (torch.int8, torch.int32, torch.int64, torch.bool):
        assert tolerance_for(dtype, "matmul") == EXACT


def test_multiplier_scales_both_components() -> None:
    base = tolerance_for(torch.bfloat16, "reduction")
    scaled = tolerance_for(torch.bfloat16, "reduction", multiplier=2.5)
    assert scaled.rtol == pytest.approx(base.rtol * 2.5)
    assert scaled.atol == pytest.approx(base.atol * 2.5)


def test_multiplier_must_be_positive() -> None:
    with pytest.raises(ValueError, match="multiplier"):
        tolerance_for(torch.float32, "elementwise", multiplier=0)


def test_unknown_kind_is_rejected_with_the_valid_set() -> None:
    with pytest.raises(KeyError, match="unknown tolerance kind"):
        tolerance_for(torch.float32, "not-a-kind")


def test_unknown_dtype_falls_back_to_a_wider_float32_budget() -> None:
    # float8 is the realistic case for "a dtype this table has never heard of".
    float8 = getattr(torch, "float8_e4m3fn", None)
    if float8 is None:
        pytest.skip("this torch build has no float8 dtype")
    fallback = tolerance_for(float8, "elementwise")
    assert fallback.rtol >= tolerance_for(torch.float32, "elementwise").rtol * 2


def test_tolerance_value_object() -> None:
    tol = Tolerance(1e-3, 1e-4)
    assert tol.as_kwargs() == {"rtol": 1e-3, "atol": 1e-4}
    assert tol.scaled(2).rtol == pytest.approx(2e-3)
    assert "rtol" in str(tol)
    assert str(EXACT) == "exact"
    assert Tolerance(1e-3, 1e-4) == Tolerance(1e-3, 1e-4)
    with pytest.raises(AttributeError):
        tol.rtol = 1.0  # type: ignore[misc]


def test_kind_catalogue_is_consistent() -> None:
    assert frozenset(KIND_SCALE) == KINDS
    assert all(name == name.lower() for name in KINDS)


def test_describe_mentions_every_kind() -> None:
    text = describe(torch.float32)
    assert "elementwise:" in text
    assert all(kind in text for kind in KINDS)
