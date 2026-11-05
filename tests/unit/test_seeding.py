"""Reproducible input generation."""

from __future__ import annotations

import random

import torch

from torch_mbt.seeding import make_generator, rand, randn, seed_all, seeded


def test_generator_is_deterministic() -> None:
    left = randn((4, 4), torch.float32, make_generator(7))
    right = randn((4, 4), torch.float32, make_generator(7))
    assert torch.equal(left, right)
    other = randn((4, 4), torch.float32, make_generator(8))
    assert not torch.equal(left, other)


def test_generator_is_cpu_bound_so_devices_never_generate_data() -> None:
    generator = make_generator(0)
    assert generator.device.type == "cpu"


def test_low_precision_shares_the_float32_underlying_values() -> None:
    base = randn((16,), torch.float32, make_generator(3))
    for dtype in (torch.float16, torch.bfloat16, torch.float64):
        sample = randn((16,), dtype, make_generator(3))
        assert sample.dtype == dtype
        if dtype in (torch.float16, torch.bfloat16):
            assert torch.allclose(sample.float(), base, atol=0.05)
        else:
            assert torch.equal(sample, base.double())


def test_scale_is_applied_before_casting() -> None:
    sample = randn((64,), torch.float32, make_generator(11), scale=100.0)
    assert sample.abs().max() > 10


def test_uniform_samples_stay_in_range() -> None:
    sample = rand((128,), torch.float32, make_generator(5))
    assert sample.min() >= 0.0
    assert sample.max() < 1.0


def test_seed_all_makes_the_global_rngs_reproducible() -> None:
    seed_all(123)
    first = (torch.randn(3), random.random())
    seed_all(123)
    second = (torch.randn(3), random.random())
    assert torch.equal(first[0], second[0])
    assert first[1] == second[1]


def test_seed_all_tolerates_backends_without_seeding() -> None:
    seed_all(1, device_type="definitely-not-a-device-type")


def test_seeded_context_restores_previous_state() -> None:
    torch.manual_seed(999)
    expected = torch.randn(4)
    torch.manual_seed(999)
    with seeded(1):
        torch.randn(4)
    assert torch.equal(torch.randn(4), expected)


def test_seeded_context_seeds_python_random_too() -> None:
    with seeded(42):
        first = random.random()
    with seeded(42):
        second = random.random()
    assert first == second


def test_seeded_context_does_not_leak_the_global_rng() -> None:
    random.seed(5)
    expected = random.random()
    random.seed(5)
    with seeded(99):
        pass
    assert random.random() == expected
