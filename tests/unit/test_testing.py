"""The declarative op-case engine."""

from __future__ import annotations

import pytest
import torch

from neutraltest.backends import CpuBackend
from neutraltest.errors import UnsupportedByBackend
from neutraltest.policy import patched_policy, reset_policy
from neutraltest.seeding import randn
from neutraltest.testing import (
    OpCase,
    build_inputs,
    cast_float,
    check_backward,
    check_case,
    evaluate,
    move_to,
    run_operator,
    unsupported_policy,
)


@pytest.fixture(autouse=True)
def _restore_policy():
    yield
    reset_policy()


@pytest.fixture
def cpu() -> CpuBackend:
    return CpuBackend()


def _pair(dtype: torch.dtype, generator: torch.Generator) -> tuple[tuple, dict]:
    return (randn((8, 8), dtype, generator), randn((8, 8), dtype, generator)), {}


def _case(**overrides) -> OpCase:
    defaults = {
        "id": "add/elementwise",
        "family": "elementwise",
        "op": "add",
        "fn": torch.add,
        "build": _pair,
    }
    defaults.update(overrides)
    return OpCase(**defaults)


def test_op_case_defaults_describe_the_happy_path() -> None:
    case = _case()
    assert case.kind == "elementwise"
    assert case.reference_dtype is torch.float64
    assert case.grad is False
    assert case.check_dtype is True
    assert case.tolerance_multiplier == 1.0
    assert case.parametrize_ids == [
        "add/elementwise-float32",
        "add/elementwise-bfloat16",
        "add/elementwise-float16",
    ]


def test_build_inputs_is_deterministic_per_case_seed() -> None:
    left, _ = build_inputs(_case(), torch.float32)
    right, _ = build_inputs(_case(), torch.float32)
    assert torch.equal(left[0], right[0])
    other, _ = build_inputs(_case(seed=99), torch.float32)
    assert not torch.equal(left[0], other[0])


def test_move_to_and_cast_float_walk_nested_containers() -> None:
    payload = {"a": torch.ones(2), "b": [torch.ones(2), 3]}
    moved = move_to(payload, torch.device("cpu"))
    assert isinstance(moved["b"], list)
    casted = cast_float({"f": torch.ones(2), "i": torch.ones(2, dtype=torch.int64)}, torch.float64)
    assert casted["f"].dtype == torch.float64
    assert casted["i"].dtype == torch.int64


def test_evaluate_returns_device_result_and_float64_reference(cpu: CpuBackend) -> None:
    actual, expected = evaluate(_case(), torch.device("cpu"), torch.float32, backend=cpu)
    assert actual.dtype == torch.float32
    assert expected.dtype == torch.float64
    assert torch.allclose(actual.double(), expected)


def test_check_case_passes_for_a_correct_operator(cpu: CpuBackend) -> None:
    check_case(_case(), cpu, torch.float32)
    check_case(_case(), cpu, torch.float16)


def test_check_case_catches_a_wrong_operator(cpu: CpuBackend) -> None:
    # Correct for the float64 reference, wrong on the device: exactly the bug
    # class a conformance suite exists to find.
    def add_or_sub(a, b):
        return a + b if a.dtype is torch.float64 else a - b

    with pytest.raises(AssertionError, match="add/elementwise"):
        check_case(_case(fn=add_or_sub), cpu, torch.float32)


def test_check_case_catches_a_shape_regression(cpu: CpuBackend) -> None:
    # float32 (the device dtype) keeps the shape; the float64 reference is
    # flattened, so the metadata check must fire before any value comparison.
    def single(dtype: torch.dtype, generator: torch.Generator):
        return (randn((8, 8), dtype, generator),), {}

    def flatten_reference(x):
        return x if x.dtype is torch.float32 else x.reshape(-1)

    case = _case(fn=flatten_reference, build=single)
    with pytest.raises(AssertionError, match="shape"):
        check_case(case, cpu, torch.float32)


def test_check_case_catches_a_dtype_regression(cpu: CpuBackend) -> None:
    case = _case(out_dtypes=(torch.int64,))
    with pytest.raises(AssertionError, match="dtype"):
        check_case(case, cpu, torch.float32)


def test_out_dtypes_must_line_up_with_the_number_of_outputs(cpu: CpuBackend) -> None:
    case = _case(out_dtypes=(torch.float32, torch.float32))
    with pytest.raises(AssertionError, match="out_dtypes"):
        check_case(case, cpu, torch.float32)


def test_index_outputs_are_declared_and_compared_exactly(cpu: CpuBackend) -> None:
    def _topk(dtype: torch.dtype, generator: torch.Generator):
        return (randn((16,), dtype, generator),), {"k": 3}

    case = _case(
        id="topk/elementwise",
        family="reduction",
        op="topk",
        fn=torch.topk,
        build=_topk,
        out_dtypes=(None, torch.int64),
        kind="indexing",
    )
    check_case(case, cpu, torch.float32)


def test_unsupported_errors_are_classified_not_leaked(cpu: CpuBackend) -> None:
    def boom(*_args, **_kwargs):
        raise NotImplementedError("aten::fancy_op not implemented for 'cpu'")

    with pytest.raises(UnsupportedByBackend, match="not implemented"):
        run_operator(boom, (), {})
    case = _case(fn=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no kernel for float16")))
    with pytest.raises(UnsupportedByBackend):
        check_case(case, cpu, torch.float32)


def test_unsupported_policy_fails_by_default(cpu: CpuBackend) -> None:
    def boom(*_args, **_kwargs):
        raise NotImplementedError("aten::fancy_op is not implemented")

    with pytest.raises(UnsupportedByBackend), unsupported_policy("fancy_op"):
        run_operator(boom, (), {})


def test_unsupported_policy_can_downgrade_to_skip() -> None:
    def boom(*_args, **_kwargs):
        raise NotImplementedError("aten::fancy_op is not implemented")

    with (
        patched_policy(on_unsupported="skip"),
        pytest.raises(pytest.skip.Exception, match="fancy_op"),
        unsupported_policy("fancy_op"),
    ):
        run_operator(boom, (), {})


def test_unsupported_policy_can_downgrade_to_xfail() -> None:
    def boom(*_args, **_kwargs):
        raise NotImplementedError("aten::fancy_op is not implemented")

    with (
        patched_policy(on_unsupported="xfail"),
        pytest.raises(pytest.xfail.Exception),
        unsupported_policy(),
    ):
        run_operator(boom, (), {})


def test_other_errors_are_not_swallowed_by_the_policy() -> None:
    with pytest.raises(ZeroDivisionError), unsupported_policy("broken"):
        raise ZeroDivisionError("arithmetic")


def _mul_case(**overrides) -> OpCase:
    defaults = {
        "id": "mul/backward",
        "family": "elementwise",
        "op": "mul",
        "fn": torch.mul,
        "build": _pair,
        "grad": True,
        "grad_inputs": (0,),
    }
    defaults.update(overrides)
    return OpCase(**defaults)


def test_check_backward_matches_the_float64_reference(cpu: CpuBackend) -> None:
    check_backward(_mul_case(), cpu, torch.float32)


def test_check_backward_requires_opt_in(cpu: CpuBackend) -> None:
    with pytest.raises(ValueError, match="opt into gradient"):
        check_backward(_case(), cpu, torch.float32)


def test_check_backward_requires_declared_inputs(cpu: CpuBackend) -> None:
    with pytest.raises(ValueError, match="grad_inputs"):
        check_backward(_mul_case(grad_inputs=()), cpu, torch.float32)


def test_check_backward_catches_a_missing_gradient(cpu: CpuBackend) -> None:
    # Same forward values in every dtype, but the device dtype detaches the
    # graph, so the float32 run produces no gradient where float64 produces one.
    def forward(a, b):
        if a.dtype is torch.float32:
            return (a * b).detach()  # forward values are right, the graph is gone
        return a * b

    case = _mul_case(fn=forward)
    with pytest.raises(AssertionError, match="backward"):
        check_backward(case, cpu, torch.float32)
