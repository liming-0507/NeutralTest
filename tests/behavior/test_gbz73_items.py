"""Each part 7.3 row fails when the thing it checks is broken.

A row that passes everywhere is indistinguishable from a row that checks
nothing -- this file is what tells the two apart.  Every test here keeps the
honest run green and then breaks exactly one thing, so that a later refactor
cannot quietly turn the row into a formality.
"""

from __future__ import annotations

import pytest

from neutraltest.backends import CpuBackend
from neutraltest.gbz73 import quant
from neutraltest.gbz73.context import RunContext
from neutraltest.gbz73.features import FEATURES
from neutraltest.gbz73.ops import OPS


def _feature(feature_id: str):
    return next(feature for feature in FEATURES if feature.id == feature_id)


def _context(feature_id: str, op_name: str) -> RunContext:
    op = next(candidate for candidate in OPS if candidate.name == op_name)
    return RunContext(op=op, backend=CpuBackend(), feature=_feature(feature_id), seed=7)


# -- A3: saturation above the largest finite value ----------------------


def test_a3_passes_on_every_operator_it_runs() -> None:
    from neutraltest.gbz73.items import a3

    for op in OPS:
        a3.run(_context("A3", op.name))


@pytest.mark.parametrize("op_name", [op.name for op in OPS])
def test_a3_reports_a_nan_on_saturation(monkeypatch: pytest.MonkeyPatch, op_name: str) -> None:
    """A quantiser that returns NaN above the range must not pass unnoticed.

    This is not a hypothetical: ``Tensor.to(float8_e4m3fn)`` really does hand
    back NaN on overflow in some PyTorch builds, which is why
    :func:`quant.quantise` clamps first.  If that clamp is ever removed, A3 has
    to be the row that says so.
    """
    from neutraltest.gbz73.items import a3

    real = quant.roundtrip

    def saturating_to_nan(tensor, scale=1.0, dtype=None):
        target = dtype or quant.DEFAULT_FP8
        out = real(tensor, scale, target)
        over = (tensor / scale).abs() > quant.max_finite(target)
        return out.masked_fill(over, float("nan"))

    monkeypatch.setattr(quant, "roundtrip", saturating_to_nan)
    with pytest.raises(AssertionError, match="NaN on saturation"):
        a3.run(_context("A3", op_name))


def test_a3_declares_a_result_that_cannot_be_scaled_out_of_range() -> None:
    """An all-zero result has nothing to saturate, and says so instead of passing."""
    import dataclasses

    import torch

    from neutraltest.gbz73.context import NotApplicable
    from neutraltest.gbz73.items import a3

    context = _context("A3", "relu")
    (first, *_), _ = context.inputs(torch.float32)
    shape = tuple(first.shape)
    context.op = dataclasses.replace(context.op, forward=lambda *args, **kwargs: torch.zeros(shape))
    with pytest.raises(NotApplicable, match="nothing can exceed the range"):
        a3.run(context)
