"""E7 -- the environment the numbers came from.

A result that cannot be reproduced because nobody wrote down the versions is
not a result. This row records the environment and checks that the pieces that
matter are actually present and behaving (FP8 constructible, deterministic
seeding available).
"""

from __future__ import annotations

import platform

import torch

from .. import quant
from ..context import RunContext

ID = "E7"
OPS = None


def run(ctx: RunContext) -> None:
    half_ulp = quant.half_ulp(ctx.fp8_dtype)
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "backend": ctx.backend.name,
        "device_type": ctx.backend.device_type,
        "fp8": str(ctx.fp8_dtype),
        "half_ulp": half_ulp,
        "max_finite": quant.max_finite(ctx.fp8_dtype),
    }
    if not hasattr(torch, "use_deterministic_algorithms"):
        raise AssertionError("this torch cannot be asked for deterministic algorithms")
    if not isinstance(info["torch"], str) or not info["torch"]:
        raise AssertionError("the torch version is not reported")
    if not half_ulp > 0:
        raise AssertionError(f"{ctx.fp8_dtype}: half an ULP is {half_ulp}")
    ctx.note(" ".join(f"{k}={v}" for k, v in info.items()))
