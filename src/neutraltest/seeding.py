"""Reproducibility helpers.

Cross-backend tests are only meaningful if both sides see *identical* inputs,
and a failure is only debuggable if it reproduces.  The convention used
throughout the harness is:

* draw random data on CPU from an explicit :class:`torch.Generator`,
* cast/normalise it there,
* then copy it to the device.

That way the device never has to implement ``randn`` identically, and the CPU
golden sees bit-identical inputs.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Iterator
from typing import Any

import torch

__all__ = ["make_generator", "seeded", "seed_all", "randn", "rand"]


def make_generator(seed: int) -> torch.Generator:
    """A CPU generator with a fixed seed (device-side RNG is never trusted)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return gen


def randn(
    shape: Any, dtype: torch.dtype, generator: torch.Generator, *, scale: float = 1.0
) -> torch.Tensor:
    """Standard normal sample drawn in float32, then cast to ``dtype``.

    Drawing in float32 first keeps every dtype on the same underlying values,
    which makes a float16 run and a float64 reference directly comparable.
    """
    base = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
    if scale != 1.0:
        base = base * scale
    return base.to(dtype)


def rand(shape: Any, dtype: torch.dtype, generator: torch.Generator) -> torch.Tensor:
    """Uniform [0, 1) sample, cast to ``dtype``."""
    return torch.rand(tuple(shape), generator=generator, dtype=torch.float32).to(dtype)


def seed_all(seed: int, device_type: str | None = None) -> None:
    """Seed python, numpy-free and torch RNGs, plus a device module if present."""
    random.seed(seed)
    torch.manual_seed(seed)
    if device_type:
        module = getattr(torch, device_type, None)
        manual_seed = getattr(module, "manual_seed", None)
        if callable(manual_seed):
            with contextlib.suppress(Exception):  # backend may not support seeding
                manual_seed(seed)


@contextlib.contextmanager
def seeded(seed: int, device_type: str | None = None) -> Iterator[None]:
    """Run a block with deterministic RNG state, restoring it afterwards."""
    cpu_state = torch.random.get_rng_state()
    py_state = random.getstate()
    try:
        seed_all(seed, device_type)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        random.setstate(py_state)
