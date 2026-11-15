"""Each part 7.3 row fails when the thing it checks is broken.

A row that passes everywhere is indistinguishable from a row that checks
nothing -- this file is what tells the two apart.  Every test here keeps the
honest run green and then breaks exactly one thing, so that a later refactor
cannot quietly turn the row into a formality.
"""

from __future__ import annotations

import pytest
import torch

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


# -- E7: the operator actually runs on the deployment -------------------


@pytest.mark.parametrize("op_name", [op.name for op in OPS])
def test_e7_calls_the_operator_at_least_twice(op_name: str) -> None:
    """E7 records the environment *and* proves the operator runs there.

    The row used to print versions and call nothing, which passed on a machine
    where the operator does not load at all -- the one condition deployment is
    meant to catch.  Count the calls: recording the environment is not the test.
    """
    from dataclasses import replace

    from neutraltest.gbz73.items import e7

    ctx = _context("E7", op_name)
    real = ctx.op.forward
    calls: list[int] = []

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    ctx.op = replace(ctx.op, forward=counting)
    e7.run(ctx)
    assert len(calls) >= 2, f"E7 called {op_name} {len(calls)} time(s); it must run it"


@pytest.mark.parametrize("op_name", [op.name for op in OPS])
def test_e7_reports_an_operator_that_is_not_repeatable(op_name: str) -> None:
    """A deployment that drifts between two identical calls must fail E7."""
    from dataclasses import replace

    from neutraltest.gbz73.items import e7

    ctx = _context("E7", op_name)
    real = ctx.op.forward
    state = {"calls": 0}

    def drifting(*args, **kwargs):
        out = real(*args, **kwargs) + state["calls"] * 1e-3
        state["calls"] += 1
        return out

    ctx.op = replace(ctx.op, forward=drifting)
    with pytest.raises(AssertionError, match="differ"):
        e7.run(ctx)


# -- F3: the baseline is not the operator -------------------------------


def test_f3_has_a_plain_baseline_for_every_operator() -> None:
    """Every operator of the matrix needs a baseline written without it."""
    from neutraltest.gbz73.items.f3 import BASELINES

    missing = [op.name for op in OPS if op.name not in BASELINES]
    assert not missing, f"no plain-expression baseline for {missing}"


def test_f3_declares_a_cpu_run_not_applicable() -> None:
    """On a CPU the operator and its baseline are the same expression."""
    from neutraltest.gbz73.items import f3

    with pytest.raises(Exception, match="same eager expression"):
        f3.run(_context("F3", "relu"))


@pytest.mark.parametrize("op_name", [op.name for op in OPS])
def test_f3_baseline_agrees_with_the_reference(op_name: str) -> None:
    """The baseline is a second expression of the same contract, not a rewrite."""
    import torch

    from neutraltest.gbz73.items._support import adapt
    from neutraltest.gbz73.items.f3 import BASELINES

    ctx = _context("F3", op_name)
    generator = torch.Generator().manual_seed(ctx.seed)
    x = torch.randn(128, 32, generator=generator)
    args, kwargs = adapt(op_name, x)
    want = ctx.op.forward(*args, **kwargs)
    got = BASELINES[op_name](*args, **kwargs)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5), f"{op_name}: baseline disagrees"


# -- E6: the second device actually computes ----------------------------


class _TwoDevice(CpuBackend):
    """A backend that advertises two devices, both of them the CPU."""

    name = "two-device"

    def device_count(self) -> int:  # noqa: D102 - stub
        return 2

    def make_device(self, index: int = 0):  # noqa: D102 - stub
        return torch.device("cpu")


@pytest.mark.parametrize("op_name", [op.name for op in OPS])
def test_e6_runs_on_the_second_device(op_name: str) -> None:
    """E6 must compute on device 1, not merely notice that it exists.

    With both "devices" being the CPU the two results agree, so the only way to
    tell a real run from a version that just asks ``device_count`` is to count
    which devices the operator was actually invoked with.
    """
    from dataclasses import replace

    from neutraltest.gbz73.items import e6

    op = next(candidate for candidate in OPS if candidate.name == op_name)
    ctx = RunContext(op=op, backend=_TwoDevice(), feature=_feature("E6"), seed=7)
    seen: list[int] = []
    real = op.forward

    def recording(*args, **kwargs):
        seen.append(1)
        return real(*args, **kwargs)

    ctx.op = replace(op, forward=recording)
    e6.run(ctx)
    # one call for device 0 and one for device 1
    assert len(seen) == 2, f"E6 ran {op_name} {len(seen)} time(s); expected one per device"


def test_e6_reports_a_second_device_that_disagrees(monkeypatch: pytest.MonkeyPatch) -> None:
    """A device 1 that computes something else must fail E6."""
    from dataclasses import replace

    from neutraltest.gbz73.items import e6

    op = next(candidate for candidate in OPS if candidate.name == "relu")
    ctx = RunContext(op=op, backend=_TwoDevice(), feature=_feature("E6"), seed=7)
    real = op.forward
    state = {"calls": 0}

    def drifting(*args, **kwargs):
        out = real(*args, **kwargs) + state["calls"] * 1e-3
        state["calls"] += 1
        return out

    ctx.op = replace(op, forward=drifting)
    with pytest.raises(AssertionError, match="differs"):
        e6.run(ctx)
