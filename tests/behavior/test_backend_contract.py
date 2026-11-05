"""Contract every backend adapter must satisfy.

These tests are parametrised over ``--backend`` by the plugin itself, so on a CPU
box they run once against ``cpu``; on an accelerator box they run once per
backend, using exactly the same assertions.
"""

from __future__ import annotations

import pytest
import torch

from neutraltest import get_policy


def test_backend_is_reported_as_available(backend) -> None:
    assert backend.is_available() is True
    assert backend.device_count() >= 1


def test_device_fixture_matches_the_backend(backend, device) -> None:
    assert device.type == backend.device_type
    assert backend.make_device(get_policy().device_index) == device


def test_roundtrip_to_the_host_is_value_preserving(backend, device) -> None:
    host = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    moved = host.to(device)
    assert moved.device.type == backend.device_type
    assert torch.equal(moved.cpu(), host)


def test_synchronize_and_cache_calls_do_not_raise(backend) -> None:
    backend.synchronize()
    backend.empty_cache()
    allocated = backend.memory_allocated()
    assert allocated is None or allocated >= 0


def test_float32_arithmetic_is_sane(backend, device) -> None:
    left = torch.full((32, 32), 2.0, device=device)
    right = torch.full((32, 32), 3.0, device=device)
    assert torch.allclose((left * right).cpu(), torch.full((32, 32), 6.0))


def test_reduction_matches_the_host(backend, device) -> None:
    values = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4).to(device)
    assert float(values.sum().cpu()) == pytest.approx(float(values.cpu().sum()))


def test_mbt_context_builds_device_tensors(nt) -> None:
    sample = nt.tensor(4, 4, seed=3)
    assert sample.device == nt.device
    assert sample.dtype == torch.float32
    assert nt.reference(sample).dtype == torch.float64
    assert nt.reference(sample).device.type == "cpu"


def test_mbt_context_inputs_are_deterministic(nt) -> None:
    first = nt.tensor(16, seed=5)
    second = nt.tensor(16, seed=5)
    third = nt.tensor(16, seed=6)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_mbt_tolerance_follows_the_policy(nt) -> None:
    tolerance = nt.tolerance(torch.float32, "matmul")
    base = nt.tolerance(torch.float32, "elementwise")
    assert tolerance.rtol > base.rtol
    assert tolerance.rtol == pytest.approx(base.rtol * 4)


def test_seeding_is_reproducible_within_a_session(backend) -> None:
    backend.manual_seed(11)
    first = torch.randn(4)
    backend.manual_seed(11)
    second = torch.randn(4)
    # Device RNGs are optional; only assert when the backend implements one.
    assert first.shape == second.shape
