"""Comparison helpers: tolerance maths, flattening, exactness, gradients."""

from __future__ import annotations

import pytest
import torch

from torch_mbt.compare import assert_all_finite, assert_close, assert_grad_close, flatten, to_cpu
from torch_mbt.tolerance import Tolerance


def test_identical_tensors_pass() -> None:
    tensor = torch.randn(4, 4)
    assert_close(tensor, tensor)


def test_large_deviation_fails_with_the_case_id() -> None:
    actual = torch.zeros(3)
    expected = torch.ones(3)
    with pytest.raises(AssertionError, match="add/broken"):
        assert_close(actual, expected, dtype=torch.float32, msg="add/broken")


def test_error_within_float16_budget_passes() -> None:
    expected = torch.linspace(-1, 1, 64, dtype=torch.float64)
    actual = (expected + 1e-4).to(torch.float16)
    assert_close(actual, expected, dtype=torch.float16)
    # The same error must fail once the budget is tightened by the multiplier.
    with pytest.raises(AssertionError):
        assert_close(actual, expected, dtype=torch.float16, multiplier=0.05)


def test_reference_at_higher_precision_is_the_default_assumption() -> None:
    expected = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    actual = expected.to(torch.float32)
    # Different dtypes, same values: allowed, because check_dtype defaults off.
    assert_close(actual, expected, dtype=torch.float32)
    with pytest.raises(AssertionError, match="dtype mismatch"):
        assert_close(actual, expected, dtype=torch.float32, check_dtype=True)


def test_shape_mismatch_is_reported() -> None:
    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_close(torch.zeros(2, 3), torch.zeros(3, 2), dtype=torch.float32)


def test_shape_check_can_be_disabled() -> None:
    assert_close(torch.zeros(6), torch.zeros(2, 3), dtype=torch.float32, check_shape=False)


def test_integers_are_compared_exactly() -> None:
    assert_close(torch.tensor([1, 2]), torch.tensor([1, 2]))
    with pytest.raises(AssertionError):
        assert_close(torch.tensor([1, 2]), torch.tensor([1, 3]))
    with pytest.raises(AssertionError):
        assert_close(torch.tensor([True, False]), torch.tensor([True, True]))


def test_nested_outputs_are_flattened_in_order() -> None:
    actual = {"values": torch.ones(2), "indices": torch.tensor([0, 1])}
    expected = (torch.ones(2, dtype=torch.float64), torch.tensor([0, 1]))
    assert_close(actual, expected, dtype=torch.float32)


def test_arity_mismatch_is_an_error_not_a_pass() -> None:
    with pytest.raises(AssertionError, match="arity"):
        assert_close((torch.ones(2), torch.ones(2)), torch.ones(2))


def test_nan_handling_is_configurable() -> None:
    actual = torch.tensor([1.0, float("nan")])
    expected = torch.tensor([1.0, float("nan")])
    assert_close(actual, expected, dtype=torch.float32, equal_nan=True)
    with pytest.raises(AssertionError):
        assert_close(actual, expected, dtype=torch.float32, equal_nan=False)


def test_exact_flag_forces_bit_exactness() -> None:
    actual = torch.tensor([1.0 + 1e-6])
    expected = torch.tensor([1.0])
    assert_close(actual, expected, dtype=torch.float32)
    with pytest.raises(AssertionError):
        assert_close(actual, expected, dtype=torch.float32, exact=True)


def test_explicit_tolerance_overrides_the_table() -> None:
    actual = torch.tensor([1.5])
    expected = torch.tensor([1.0])
    assert_close(actual, expected, tolerance=Tolerance(1.0, 1.0))
    with pytest.raises(AssertionError):
        assert_close(actual, expected, tolerance=Tolerance(1e-6, 1e-6))


def test_python_scalars_are_accepted() -> None:
    assert_close(1.0, torch.tensor(1.0, dtype=torch.float64), dtype=torch.float32)
    assert_close(torch.tensor(2.0), 2.0, dtype=torch.float32)


def test_all_finite_detects_overflow_but_ignores_integers() -> None:
    with pytest.raises(AssertionError, match="non-finite"):
        assert_all_finite(torch.tensor([1.0, float("inf")]), msg="conv/overflow")
    with pytest.raises(AssertionError, match="non-finite"):
        assert_all_finite(torch.tensor([float("nan")]))
    assert_all_finite(torch.tensor([1, 2, 3]))
    assert_all_finite(torch.tensor([1.0, 2.0]))
    assert_all_finite(torch.empty(0))


def test_flatten_and_to_cpu_handles_containers() -> None:
    tensor = torch.ones(2)
    assert len(flatten(tensor)) == 1
    assert len(flatten((tensor, [tensor, tensor]))) == 3
    assert len(flatten({"a": tensor, "b": (tensor,)})) == 2
    assert to_cpu(tensor).device.type == "cpu"
    assert to_cpu(3) == 3


def test_grad_close_accepts_none_on_both_sides() -> None:
    grad = torch.ones(2)
    assert_grad_close([grad, None], [grad.clone(), None], dtype=torch.float32)
    with pytest.raises(AssertionError, match="no gradient"):
        assert_grad_close([grad], [None], dtype=torch.float32)
    with pytest.raises(AssertionError, match="count mismatch"):
        assert_grad_close([grad], [grad, grad], dtype=torch.float32)
    with pytest.raises(AssertionError):
        assert_grad_close([torch.zeros(2)], [torch.ones(2)], dtype=torch.float32)
