"""F7's verdict: a device call inside the launch floor is not a verdict about the operator."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from neutraltest.gbz73.items import f7
from neutraltest.gbz73.items._support import device_call


class _Ctx:
    """The parts of RunContext F7 touches, with a device that is not the CPU."""

    def __init__(self) -> None:
        self.device = torch.device("privateuseone")
        self.backend = SimpleNamespace(synchronize=lambda: None)
        self.op = SimpleNamespace(name="relu", communicated=False, forward=torch.relu)
        self.variant = ""
        self.notes: list[str] = []

    def note(self, text: str) -> None:
        self.notes.append(text)

    def na(self, reason: str) -> None:
        raise AssertionError(f"not applicable: {reason}")

    def move(self, value: object) -> object:
        return value


def test_the_device_side_is_waited_for_before_the_clock_stops() -> None:
    seen: list[str] = []

    class _Backend:
        device_type = "privateuseone"

        def make_device(self, index: int = 0) -> torch.device:
            return torch.device("privateuseone")

        def synchronize(self) -> None:
            seen.append("sync")

    ctx = _Ctx()
    ctx.inputs = lambda dtype: ((torch.ones(4),), {})  # type: ignore[attr-defined]
    device_call(ctx, backend=_Backend())()
    assert seen == ["sync"]


def test_a_ratio_inside_the_launch_floor_is_a_note_not_a_failure(monkeypatch) -> None:
    ctx = _Ctx()
    ctx.inputs = lambda dtype: ((torch.ones(4),), {})  # type: ignore[attr-defined]
    times = iter([2e-5, 2e-6])  # 20 us on the device (10x), both under the floor
    monkeypatch.setattr(f7, "measure", lambda fn: next(times))
    f7.run(ctx)
    assert any("launch floor" in note for note in ctx.notes)


def test_a_slow_device_outside_the_floor_still_fails(monkeypatch) -> None:
    ctx = _Ctx()
    ctx.inputs = lambda dtype: ((torch.ones(4),), {})  # type: ignore[attr-defined]
    times = iter([5e-3, 2e-6])  # 5 ms on the device: a real result, not a floor
    monkeypatch.setattr(f7, "measure", lambda fn: next(times))
    with pytest.raises(AssertionError, match="slower than the CPU"):
        f7.run(ctx)
