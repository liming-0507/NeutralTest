"""The float64 probe tells arithmetic apart from storage."""

from __future__ import annotations

import torch

from neutraltest.backends import CpuBackend
from neutraltest.gbz72._support import _computes_in_float64, effective_compute_dtype
from neutraltest.gbz73.inference.columns import BY_NAME
from neutraltest.gbz73.inference.context import InferenceContext


class WideThroughFloat32(CpuBackend):
    """A backend that stores float64 and computes in float32, as a device does."""

    name = "wide-through-float32"

    def make_device(self, index: int = 0) -> torch.device:  # noqa: ARG002 - CPU only
        return torch.device("cpu")


class _Stub:
    """A context whose only relevant method is the probe."""

    def __init__(self, native: bool) -> None:
        self._native = native

    def computes_in_float64(self) -> bool:
        return self._native

    def note(self, message: str) -> None:
        self.notes.append(message)

    notes: list[str] = []


def _context() -> InferenceContext:
    return InferenceContext(column=BY_NAME["softmax_fwd"], backend=CpuBackend(), feature_id="A1")


def test_cpu_arithmetic_is_float64() -> None:
    assert _context().computes_in_float64() is True


def test_effective_dtype_keeps_float64_when_the_backend_computes_in_it() -> None:
    assert effective_compute_dtype(_context(), torch.float64) is torch.float64


def test_effective_dtype_follows_the_arithmetic_when_the_backend_widens() -> None:
    stub = _Stub(native=False)
    assert effective_compute_dtype(stub, torch.float64) is torch.float32


def test_effective_dtype_is_noop_for_narrower_requests() -> None:
    stub = _Stub(native=False)
    assert effective_compute_dtype(stub, torch.float32) is torch.float32


def test_absence_of_a_probe_is_not_evidence_of_narrow_arithmetic() -> None:
    class NoProbe:
        pass

    assert _computes_in_float64(NoProbe()) is True


def test_the_probe_asks_the_backend_rather_than_the_dtype_system() -> None:
    # Allocating a float64 tensor is not proof of float64 arithmetic: the CPU
    # backend here can hold one, yet the probe must decide by computing.
    device = torch.device("cpu")
    held = torch.empty(1, dtype=torch.float64, device=device)
    assert held.dtype is torch.float64
    assert _context().computes_in_float64() is True
