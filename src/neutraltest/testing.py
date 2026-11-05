"""Declarative operator cases and the engine that checks them.

A conformance test should describe *what* to feed an operator, not *how* to
compare across devices.  That is what :class:`OpCase` captures: one operator,
one or more input builders, and a tolerance class.  :func:`check_case` then

1. builds the inputs on CPU (so the device never has to implement ``randn``
   identically),
2. runs the operator on the device,
3. runs the same operator on CPU at a higher precision,
4. compares shapes/dtypes, then values with the dtype+kind tolerance,
5. optionally repeats for gradients.

Example
-------
>>> case = OpCase(
...     id="add/broadcast",
...     family="elementwise",
...     op="add",
...     fn=torch.add,
...     build=lambda dtype, gen: ((randn((8, 16), dtype, gen), randn((16,), dtype, gen)), {}),
... )
>>> check_case(case, backend, torch.float32)  # doctest: +SKIP
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NoReturn

import torch

from .compare import assert_all_finite, assert_close, assert_grad_close, flatten
from .errors import UnsupportedByBackend, is_unsupported_error
from .policy import get_policy
from .seeding import make_generator

__all__ = [
    "OpCase",
    "check_case",
    "check_backward",
    "evaluate",
    "move_to",
    "cast_float",
    "unsupported_policy",
    "run_operator",
]


@dataclass(frozen=True)
class OpCase:
    """One operator invocation plus the tolerances it needs.

    Attributes:
        id: stable identifier, e.g. ``"conv2d/groups"``; appears in failure output.
        family: grouping used by the report (``elementwise``, ``matmul``, ...).
        op: operator name for the report.
        fn: callable to test; receives the built positional/keyword arguments.
        build: ``build(dtype, generator) -> (args, kwargs)``, executed on CPU.
        dtypes: input dtypes to exercise.
        kind: tolerance class, see :data:`neutraltest.tolerance.KINDS`.
        reference_dtype: precision used for the CPU golden value.  ``None`` means
            "same dtype as the device run".
        grad: also check the backward pass.
        grad_inputs: positional argument indices that require gradients.
        out_dtypes: expected output dtypes, positionally.  ``None`` entries mean
            "same as the input dtype".  Required for operators returning indices.
        check_dtype: assert the output dtypes (default on).
        seed: per-case seed for input generation.
        tags: free-form labels (``"slow"``, ``"training"``, ...).
        tolerance_multiplier: extra relaxation for this case only.
    """

    id: str
    family: str
    op: str
    fn: Callable[..., Any]
    build: Callable[[torch.dtype, torch.Generator], tuple[Sequence[Any], Mapping[str, Any]]]
    dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.bfloat16, torch.float16)
    kind: str = "elementwise"
    reference_dtype: torch.dtype | None = torch.float64
    grad: bool = False
    grad_inputs: tuple[int, ...] = ()
    out_dtypes: tuple[torch.dtype | None, ...] | None = None
    check_dtype: bool = True
    seed: int = 0
    tags: tuple[str, ...] = field(default_factory=tuple)
    tolerance_multiplier: float = 1.0

    @property
    def parametrize_ids(self) -> list[str]:
        return [f"{self.id}-{str(dtype).removeprefix('torch.')}" for dtype in self.dtypes]


# --------------------------------------------------------------------------- helpers


def _map(value: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    """Apply *fn* to every tensor inside nested containers."""
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, Mapping):
        return {key: _map(item, fn) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        mapped = [_map(item, fn) for item in value]
        return (
            type(value)(mapped)
            if isinstance(value, tuple) and hasattr(value, "_fields")
            else mapped
        )
    return value


def move_to(value: Any, device: torch.device) -> Any:
    """Copy every tensor in *value* to *device*."""
    return _map(value, lambda tensor: tensor.to(device))


def cast_float(value: Any, dtype: torch.dtype | None) -> Any:
    """Cast floating-point tensors to *dtype* (integers are left alone)."""
    if dtype is None:
        return value
    return _map(
        value,
        lambda tensor: tensor.to(dtype) if tensor.is_floating_point() else tensor,
    )


def run_operator(fn: Callable[..., Any], args: Sequence[Any], kwargs: Mapping[str, Any]) -> Any:
    """Call *fn*, translating "no kernel" errors into :class:`UnsupportedByBackend`."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - classified below
        if is_unsupported_error(exc):
            raise UnsupportedByBackend(f"{type(exc).__name__}: {exc}") from exc
        raise


def build_inputs(
    case: OpCase, dtype: torch.dtype, *, seed: int | None = None
) -> tuple[Sequence[Any], Mapping[str, Any]]:
    generator = make_generator(case.seed if seed is None else seed)
    args, kwargs = case.build(dtype, generator)
    return tuple(args), dict(kwargs)


# --------------------------------------------------------------------------- engine


def evaluate(
    case: OpCase,
    device: torch.device,
    dtype: torch.dtype,
    *,
    backend: Any | None = None,
    reference_dtype: torch.dtype | None | str = "case",
) -> tuple[Any, Any]:
    """Run the case on *device* and on CPU; return ``(actual, expected)``.

    Errors that look like "this backend has no kernel" are re-raised as
    :class:`UnsupportedByBackend` so callers can decide the policy.
    """
    args, kwargs = build_inputs(case, dtype)
    actual = run_operator(case.fn, move_to(args, device), move_to(kwargs, device))
    if backend is not None:
        with contextlib.suppress(NotImplementedError):
            backend.synchronize()

    golden: torch.dtype | None
    if reference_dtype == "case":
        golden = case.reference_dtype
    elif isinstance(reference_dtype, str):
        raise TypeError(f"unknown reference_dtype {reference_dtype!r}")
    else:
        golden = reference_dtype
    if golden is None:
        golden = dtype
    expected = case.fn(*cast_float(args, golden), **cast_float(kwargs, golden))
    return actual, expected


def check_case(case: OpCase, backend: Any, dtype: torch.dtype, *, index: int = 0) -> None:
    """Assert that *case* on *backend* matches the CPU reference.

    Raises:
        AssertionError: shapes, dtypes, finiteness or values disagree.
        UnsupportedByBackend: the backend has no kernel for this op/dtype.
    """
    device = backend.make_device(index)
    actual, expected = evaluate(case, device, dtype, backend=backend)
    _check_metadata(case, actual, expected, dtype)

    multiplier = get_policy().tolerance_multiplier * case.tolerance_multiplier
    assert_close(
        actual,
        expected,
        dtype=dtype,
        kind=case.kind,
        multiplier=multiplier,
        msg=case.id,
    )
    _check_finiteness(case, actual, expected)


def check_backward(case: OpCase, backend: Any, dtype: torch.dtype, *, index: int = 0) -> None:
    """Assert that gradients on *backend* match the CPU reference."""
    if not case.grad:
        raise ValueError(f"{case.id}: case does not opt into gradient checking")

    device = backend.make_device(index)
    args, kwargs = build_inputs(case, dtype)
    if not case.grad_inputs:
        raise ValueError(f"{case.id}: grad=True requires grad_inputs")

    got = _backward(case, args, kwargs, device=device, dtype=dtype)
    reference_dtype = dtype if case.reference_dtype is None else case.reference_dtype
    expected = _backward(
        case,
        cast_float(args, reference_dtype),
        cast_float(kwargs, reference_dtype),
        device=torch.device("cpu"),
        dtype=reference_dtype,
    )

    multiplier = get_policy().tolerance_multiplier * case.tolerance_multiplier
    assert_grad_close(
        got,
        expected,
        dtype=dtype,
        kind="composite",
        multiplier=multiplier,
        msg=f"{case.id} backward",
    )


def _backward(
    case: OpCase,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor | None]:
    args = list(move_to(tuple(args), device))
    for position in case.grad_inputs:
        tensor = args[position]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{case.id}: grad_inputs[{position}] is not a tensor")
        args[position] = tensor.detach().to(dtype).clone().requires_grad_(True)

    out = run_operator(case.fn, args, move_to(kwargs, device))
    loss = _sum_floats(out)
    # A missing graph is reported by the comparison ("one side has no
    # gradient"), which is a far better message than autograd's RuntimeError.
    if loss.requires_grad:
        loss.backward()
    return [args[position].grad for position in case.grad_inputs]


def _sum_floats(value: Any) -> torch.Tensor:
    """Scalar loss over every floating-point tensor in an operator's output."""
    total: torch.Tensor | None = None
    for item in flatten(value):
        if not isinstance(item, torch.Tensor) or not item.is_floating_point():
            continue
        contribution = item.float().sum()
        total = contribution if total is None else total + contribution
    if total is None:
        raise TypeError("operator output has no floating point tensor to call backward on")
    return total


def _check_metadata(case: OpCase, actual: Any, expected: Any, dtype: torch.dtype) -> None:
    got = flatten(actual)
    want = flatten(expected)
    if len(got) != len(want):
        raise AssertionError(f"{case.id}: expected {len(want)} outputs, got {len(got)}")
    if case.out_dtypes is not None and len(case.out_dtypes) != len(got):
        raise AssertionError(
            f"{case.id}: out_dtypes has {len(case.out_dtypes)} entries for {len(got)} outputs"
        )
    for position, (actual_item, expected_item) in enumerate(zip(got, want, strict=True)):
        actual_tensor = torch.as_tensor(actual_item)
        expected_tensor = torch.as_tensor(expected_item)
        if actual_tensor.shape != expected_tensor.shape:
            raise AssertionError(
                f"{case.id}: output[{position}] shape {tuple(actual_tensor.shape)} "
                f"!= reference {tuple(expected_tensor.shape)}"
            )
        if not case.check_dtype:
            continue
        declared = case.out_dtypes[position] if case.out_dtypes else dtype
        wanted = dtype if declared is None else declared
        if actual_tensor.dtype != wanted:
            raise AssertionError(
                f"{case.id}: output[{position}] dtype {actual_tensor.dtype} != expected {wanted}"
            )


def _check_finiteness(case: OpCase, actual: Any, expected: Any) -> None:
    """Overflow in low precision shows up as inf/nan where the reference is finite."""
    for position, (actual_item, expected_item) in enumerate(
        zip(flatten(actual), flatten(expected), strict=True)
    ):
        expected_tensor = torch.as_tensor(expected_item)
        if not expected_tensor.is_floating_point():
            continue
        reference = expected_tensor.to(torch.float64)
        if not bool(torch.isfinite(reference).all()):
            continue
        assert_all_finite(torch.as_tensor(actual_item), msg=f"{case.id}: output[{position}]")


@contextlib.contextmanager
def unsupported_policy(what: str = "") -> Iterator[None]:
    """Apply the run policy to :class:`UnsupportedByBackend`.

    Default policy is ``fail`` (an unsupported operator is a conformance
    finding, not a pass).  ``--on-unsupported=skip`` turns it into a skip, which
    is what you want when collecting coverage numbers on a partial backend.
    """
    try:
        yield
    except UnsupportedByBackend as exc:
        message = f"{what}: {exc}" if what else str(exc)
        policy = get_policy()
        if policy.on_unsupported == "skip" or policy.on_unsupported == "xfail":
            import pytest

            if policy.on_unsupported == "skip":
                pytest.skip(message)
            pytest.xfail(message)
        raise


def _never() -> NoReturn:  # pragma: no cover - typing helper
    raise AssertionError("unreachable")
