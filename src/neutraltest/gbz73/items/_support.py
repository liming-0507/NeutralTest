"""Small builders shared by the shape-oriented features.

Not a feature script (the leading underscore keeps it out of the catalogue).
These adapters let an item hand a tensor of *any* shape to any operator, which
is what the degenerate-shape, layout and scale-sweep features need.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch

from ..ops import Args, Kwargs

#: Width of the last dimension above which a LayerNorm weight is generated.
_MAX_WEIGHT = 1 << 14


def adapt(op_name: str, x: torch.Tensor, *, groups: list[int] | None = None) -> tuple[Args, Kwargs]:
    """Wrap ``x`` into the argument list of ``op_name``."""
    if op_name in {"relu", "gelu"}:
        return (x,), {}
    if op_name == "amax":
        return (x,), {"dim": -1}
    if op_name == "layer_norm":
        width = x.shape[-1]
        if width > _MAX_WEIGHT:
            raise ValueError(f"last dimension {width} is too wide for a LayerNorm weight")
        weight = torch.ones(width, dtype=x.dtype)
        bias = torch.zeros(width, dtype=x.dtype)
        return (x, weight, bias), {"eps": 1e-5}
    if op_name == "grouped_gemm":
        k = x.shape[-1]
        b = torch.eye(k, dtype=x.dtype).unsqueeze(0)
        sizes = groups if groups is not None else [x.shape[0]]
        offsets = torch.tensor([0, *sizes], dtype=torch.int32)
        b = b.expand(len(sizes), k, k).contiguous()
        return (x, b, offsets), {}
    raise ValueError(f"no adapter for {op_name}")


def run(op_name: str, x: torch.Tensor, *, forward) -> torch.Tensor:
    """Run ``forward`` on ``x`` with the right argument list for ``op_name``."""
    args, kwargs = adapt(op_name, x)
    return forward(*args, **kwargs)


def elementwise_input(
    size: int = 64, *, dtype: torch.dtype = torch.float32, seed: int = 0
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(size, generator=generator).to(dtype)


def within_half_ulp(
    back: torch.Tensor, original: torch.Tensor, dtype: torch.dtype, *, slack: float = 1.0005
) -> bool:
    """Is every element within half an ULP of where it started?

    A relative bound is the wrong test near zero -- the distance between two
    neighbours shrinks with the exponent, so a value that moves by a denormal
    step looks like a 100 % error. The bound is therefore taken against the
    exponent bucket, with the subnormal range floored at the smallest subnormal.
    """
    from .. import quant

    half = quant.half_ulp(dtype)
    # Below the smallest normal number every step is one and the same, so the
    # floor must be that number's own ULP -- not the value's exponent.
    smallest_normal = float(torch.finfo(dtype).tiny)
    allowed = half * original.detach().abs().clamp_min(smallest_normal)
    return bool((back.detach() - original.detach()).abs().le(allowed * slack).all())


class NotVerifiable(RuntimeError):
    """A check that needs a number this machine cannot resolve reliably."""


#: Below this, the clock is comparing noise: a CPU thread migrated to a busy core
#: can take twice as long on the same input, and no timing check can tell that from
#: a slow kernel.
LOW_RESOLUTION_NS = 200.0


def measure(fn, *, warmup: int = 10, repeat: int = 25) -> float:
    """Median wall time of `fn`, after enough warm-up to leave the cold path.

    Per-call numbers in this suite run well under a microsecond on a CPU, which is
    the same order as scheduling noise. One measurement is therefore both the
    median and an interleaved one, and the caller decides whether the median is
    above the resolution floor before believing anything derived from it.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


#: A ratio of two timings has this much noise before the operator does anything:
#: on a shared CPU the same call repeats within about 30 %, so 1.5x is the
#: narrowest ratio this protocol can defend.
NOISE_SLACK = 1.5


def require_resolvable(seconds: float, what: str) -> float:
    """Reject a timing the clock cannot resolve; `na` beats a guess."""
    if seconds * 1e9 < LOW_RESOLUTION_NS:
        raise NotVerifiable(
            f"{what} takes {seconds * 1e9:.0f} ns, below the {LOW_RESOLUTION_NS:.0f} ns this "
            "machine can time reliably; the clock, not the operator, would decide the verdict"
        )
    return seconds


def device_call(ctx) -> Callable[[], object]:
    """A zero-argument callable that runs the operator on the device under test."""
    args, kwargs = ctx.inputs(torch.float32)
    moved = tuple(ctx.move(a) for a in args)

    def call():
        return ctx.op.forward(*moved, **kwargs)

    return call
