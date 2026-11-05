"""Helpers shared by the conformance modules.

The conformance suite is deliberately organised the same way for every operator:
a module declares a list of :class:`~torch_mbt.testing.OpCase` objects and hands
them to :func:`test_operator`, which parametrises over dtype and runs the case
through the backend contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch

from torch_mbt.seeding import randn
from torch_mbt.testing import OpCase, check_backward, check_case, unsupported_policy

__all__ = ["params", "run", "run_case", "unary", "binary", "positive", "short_dtype"]


def short_dtype(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def params(cases: Sequence[OpCase]) -> list[Any]:
    """One pytest parameter per (case, dtype), carrying the report markers."""
    out: list[Any] = []
    for case in cases:
        marks = [pytest.mark.mbt_family(case.family), pytest.mark.mbt_op(case.op)]
        if case.grad:
            marks.append(pytest.mark.mbt_grad)
        if "slow" in case.tags:
            marks.append(pytest.mark.mbt_slow)
        for dtype in case.dtypes:
            out.append(
                pytest.param(
                    case,
                    dtype,
                    id=f"{case.id}-{short_dtype(dtype)}",
                    marks=list(marks),
                )
            )
    return out


def run(case: OpCase, backend: Any, dtype: torch.dtype) -> None:
    """Check one case in one dtype, honouring ``--on-unsupported``."""
    with unsupported_policy(f"{case.id} [{short_dtype(dtype)}]"):
        check_case(case, backend, dtype)
        if case.grad:
            check_backward(case, backend, dtype)


def run_case(case: OpCase, dtype: torch.dtype, backend: Any) -> None:
    """Body of every conformance test; named so pytest never collects it."""
    run(case, backend, dtype)


# --------------------------------------------------------------------------- builders
# A builder is ``build(dtype, generator) -> (args, kwargs)`` and always runs on
# CPU, so the device never has to implement random number generation.


def unary(shape: tuple[int, ...] = (8, 16), scale: float = 1.0):
    def build(dtype: torch.dtype, generator: torch.Generator):
        return (randn(shape, dtype, generator, scale=scale),), {}

    return build


def binary(shape: tuple[int, ...] = (8, 16), other: tuple[int, ...] = (16,), scale: float = 1.0):
    def build(dtype: torch.dtype, generator: torch.Generator):
        return (
            randn(shape, dtype, generator, scale=scale),
            randn(other, dtype, generator, scale=scale),
        ), {}

    return build


def positive(shape: tuple[int, ...] = (8, 16), floor: float = 0.5):
    """Strictly positive samples, for ``log``/``sqrt``/``pow``."""

    def build(dtype: torch.dtype, generator: torch.Generator):
        return (randn(shape, dtype, generator).abs() + floor,), {}

    return build
