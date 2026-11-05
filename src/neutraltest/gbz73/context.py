"""What one feature needs in order to inspect one operator on one backend.

A feature never talks to a device driver and never invents its own tolerance.
It asks the context for a baseline and for a quantised run, compares them with
:mod:`neutraltest.gbz73.metrics`, and compares the result against
:meth:`RunContext.floor` -- the error floor implied by the FP8 format itself.

The two baselines are the two standards the tests are held to:

``normal()``
    the operator on the CPU in float32 -- what any framework's ordinary path
    would produce.
``golden()``
    the same operator on the CPU in float64 -- the reference the numbers are
    judged against, so that the float32 baseline's own rounding is not mistaken
    for the operator's error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NoReturn

import torch

from ..backends.base import Backend
from . import metrics, quant
from .features import Feature
from .ops import Args, Kwargs, Op

#: Documented amplification per operator family, on top of the format floor.
#:
#: Every quantised run pays twice: once for rounding the input and once for
#: rounding the output, hence the factor 2 that every family starts from. The
#: rest covers what the operator itself does to that error -- a normalisation
#: divides by a spread that may be smaller than one, a GEMM sums ``K`` rounded
#: products. These are *derived* numbers, documented here once, not tuned per
#: test: a run that exceeds them is reported, not accommodated.
FAMILY_AMPLIFICATION: dict[str, float] = {
    "activation": 1.0,
    "reduction": 1.0,
    "normalisation": 2.0,
    "gemm": 2.0,
}

#: Two roundings per run: the input and the output.
ROUNDING_EVENTS = 2.0


class NotApplicable(RuntimeError):
    """Raised when a feature cannot be exercised here, with the reason attached."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class RunContext:
    """Everything a feature is allowed to touch."""

    op: Op
    backend: Backend
    feature: Feature
    fp8_dtype: torch.dtype = quant.DEFAULT_FP8
    seed: int = 0
    variant: str = ""
    device_index: int = 0
    notes: list[str] = field(default_factory=list)
    _device: torch.device | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.variant:
            self.variant = self.op.variants[0]

    # -- inputs ---------------------------------------------------------
    def inputs(
        self, dtype: torch.dtype = torch.float32, variant: str | None = None
    ) -> tuple[Args, Kwargs]:
        """Deterministic inputs for this operator, on the CPU."""
        return self.op.make(variant or self.variant, dtype, self.seed)

    @property
    def device(self) -> torch.device:
        """The device under test, resolved once from the backend and the policy."""
        if self._device is None:
            self._device = self.backend.make_device(self.device_index)
        return self._device

    def move(self, value: Any) -> Any:
        """Send tensors to the device under test; leave everything else alone."""
        if isinstance(value, torch.Tensor) and self.device.type != "cpu":
            return value.to(self.device)
        return value

    def collect(self, value: Any) -> torch.Tensor:
        """Bring a result back to the CPU so the comparisons stay portable."""
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{self.op.name} returned {type(value).__name__}, not a tensor")
        return value if value.device.type == "cpu" else value.detach().cpu()

    def run(self, args: Args, kwargs: Kwargs) -> torch.Tensor:
        """Call the operator on the device under test and collect the result."""
        moved = tuple(self.move(a) for a in args)
        self.backend.manual_seed(self.seed) if self.device.type != "cpu" else None
        out = self.op.forward(*moved, **kwargs)
        self.backend.synchronize()
        return self.collect(out)

    # -- the two baselines ----------------------------------------------
    def normal(self, variant: str | None = None) -> torch.Tensor:
        """CPU float32 result: the ordinary standard."""
        args, kwargs = self.inputs(torch.float32, variant)
        return self.op.forward(*args, **kwargs)

    def golden(self, variant: str | None = None) -> torch.Tensor:
        """CPU float64 result: the golden standard."""
        args, kwargs = self.inputs(torch.float64, variant)
        return self.op.forward(*args, **kwargs)

    # -- the quantised path ---------------------------------------------
    def quantise_inputs(
        self,
        args: Args,
        kwargs: Kwargs,
        *,
        dtype: torch.dtype | None = None,
        scale: torch.Tensor | float | None = None,
    ) -> tuple[Args, Kwargs, torch.Tensor]:
        """Round every float input to FP8 and back, returning the scale used."""
        target = dtype or self.fp8_dtype
        floats = [a for a in args if isinstance(a, torch.Tensor) and a.is_floating_point()]
        if not floats:
            raise NotApplicable(f"{self.op.name}: no floating-point input to quantise")
        if scale is None:
            amax = torch.stack([t.abs().max() for t in floats]).max().clamp_min(1e-12)
            scale = amax / quant.max_finite(target)
        scale = torch.as_tensor(scale, dtype=torch.float32)
        out_args = tuple(
            quant.roundtrip(a, scale, target)
            if isinstance(a, torch.Tensor) and a.is_floating_point()
            else a
            for a in args
        )
        return out_args, kwargs, scale

    def quantised(
        self,
        *,
        dtype: torch.dtype | None = None,
        scale: torch.Tensor | float | None = None,
        variant: str | None = None,
        quantise_output: bool = True,
    ) -> torch.Tensor:
        """quantise inputs -> compute in float32 -> quantise output -> dequantise."""
        target = dtype or self.fp8_dtype
        args, kwargs = self.inputs(torch.float32, variant)
        q_args, kwargs, used_scale = self.quantise_inputs(args, kwargs, dtype=target, scale=scale)
        out = self.op.forward(*q_args, **kwargs)
        if quantise_output:
            # The output has its own range: an accumulator is not the input.
            # Reusing the input scale here would saturate every result.
            out_scale = quant.dynamic_scale(out.detach().reshape(-1).to(torch.float32), target)
            out = quant.roundtrip(out, out_scale, target)
        return out.to(torch.float32)

    # -- judgement -------------------------------------------------------
    def floor(self, dtype: torch.dtype | None = None) -> float:
        """Largest normalised error an honest FP8 run of this operator can show."""
        target = dtype or self.fp8_dtype
        amplification = FAMILY_AMPLIFICATION[self.op.family]
        return ROUNDING_EVENTS * quant.half_ulp(target) * amplification

    def diff(self, actual: torch.Tensor, expected: torch.Tensor) -> metrics.Diff:
        return metrics.compare(actual, expected)

    def check(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        rel: float | None = None,
        msg: str = "",
    ) -> metrics.Diff:
        """Compare and fail if the normalised error exceeds the format floor."""
        diff = self.diff(actual, expected)
        metrics.assert_within(
            diff,
            rel=self.floor() if rel is None else rel,
            msg=msg or f"{self.feature.id}/{self.op.name}",
        )
        return diff

    # -- gradients -------------------------------------------------------
    def gradients(
        self, dtype: torch.dtype = torch.float64, *, variant: str | None = None
    ) -> list[torch.Tensor | None]:
        """Gradients of the summed output w.r.t. every floating-point input."""
        args, kwargs = self.inputs(dtype, variant)
        leaves: list[torch.Tensor] = []
        rebuilt: list[Any] = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.is_floating_point():
                leaf = arg.detach().clone().requires_grad_(True)
                leaves.append(leaf)
                rebuilt.append(leaf)
            else:
                rebuilt.append(arg)
        out = self.op.forward(*rebuilt, **kwargs)
        out.sum().backward()
        return [leaf.grad for leaf in leaves]

    # -- flow control ----------------------------------------------------
    def na(self, reason: str) -> NoReturn:
        """Declare the feature not applicable here, with the reason."""
        raise NotApplicable(reason)

    def require(self, condition: bool, reason: str) -> None:
        """Declare the feature not applicable unless ``condition`` holds."""
        if not condition:
            self.na(reason)

    def note(self, message: str) -> None:
        self.notes.append(message)
