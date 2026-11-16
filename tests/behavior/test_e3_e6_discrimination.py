"""E3 and E6 have to be able to go red.

Each test breaks exactly one thing a concurrency or multi-device row exists to
catch and checks that the row notices. A row that only ever goes green is
indistinguishable from a row that does nothing, and these are the cheapest
possible witnesses that this one does something.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any

import pytest
import torch

from neutraltest.backends import CpuBackend
from neutraltest.gbz73.context import RunContext
from neutraltest.gbz73.features import FEATURES
from neutraltest.gbz73.items import e3, e6
from neutraltest.gbz73.ops import OPS, Op

FEATURE = {feature.id: feature for feature in FEATURES}
GELU = next(op for op in OPS if op.name == "gelu")
LAYER_NORM = next(op for op in OPS if op.name == "layer_norm")
GEMM = next(op for op in OPS if op.name == "grouped_gemm")


class Streaming(CpuBackend):
    """A stub backend that claims a stream API and nothing else.

    The streams behave like the real ones in the two respects the row depends
    on: entering one makes it current for the duration of the block, and the
    object carries a device. What runs underneath is still the CPU reference, so
    a single stubbed forward can decide what each stream answers.
    """

    def __init__(self, streams: int = 2) -> None:
        super().__init__()
        self._streams = streams
        self.current = _Stream(0)

    def has_streams(self) -> bool:
        return True

    def device_count(self) -> int:
        return self._streams

    def new_stream(self, index: int = 0) -> Any:
        return _Stream(index)

    def stream_context(self, stream: Any) -> Any:
        return _Switch(self, stream)


class _Switch:
    """Make one stub stream current for the block, restoring the last one."""

    def __init__(self, backend: Streaming, stream: _Stream) -> None:
        self.backend = backend
        self.stream = stream
        self.previous: _Stream | None = None

    def __enter__(self) -> _Stream:
        self.previous = self.backend.current
        self.backend.current = self.stream
        return self.stream

    def __exit__(self, *_: object) -> None:
        if self.previous is not None:
            self.backend.current = self.previous


class _Stream:
    """Enough of a stream for the row: an index, a device, and a wait."""

    def __init__(self, index: int) -> None:
        self.index = index
        self.device = torch.device("cpu")

    def wait_stream(self, other: Any) -> None:
        del other


def _context(op: Op, backend: CpuBackend, feature: str, forward: Any = None) -> RunContext:
    chosen = op if forward is None else replace(op, forward=forward)
    return RunContext(op=chosen, backend=backend, feature=FEATURE[feature], seed=7)


# -- E3: threads ---------------------------------------------------------


def test_a_shared_scratch_buffer_turns_the_thread_half_red() -> None:
    """The second caller must not be able to overwrite what the first returned."""
    state: dict[str, Any] = {"lock": threading.Lock(), "last": None}

    def reusing(*args: Any, **kwargs: Any) -> torch.Tensor:
        with state["lock"]:
            first = GELU.forward(*args, **kwargs)
            state["last"] = first if state["last"] is None else state["last"]
            return state["last"]

    with pytest.raises(AssertionError):
        e3.run(_context(GELU, CpuBackend(), "E3", reusing))


def test_concurrent_callers_with_different_inputs_must_not_be_mixed_up() -> None:
    """A cache keyed by nothing answers the first caller's question to everyone."""
    state: dict[str, Any] = {"lock": threading.Lock(), "calls": 0}

    def answers_the_first(*args: Any, **kwargs: Any) -> torch.Tensor:
        with state["lock"]:
            state["calls"] += 1
            first = state["calls"] == 1
        if first:
            state["answer"] = GELU.forward(*args, **kwargs)
        return state["answer"]

    with pytest.raises(AssertionError):
        e3.run(_context(GELU, CpuBackend(), "E3", answers_the_first))


def test_the_thread_half_passes_on_a_clean_operator() -> None:
    """The witness needs a control: without one, a red test proves nothing."""
    context = _context(GELU, CpuBackend(), "E3")
    e3.run(context)
    assert "no stream API" in context.notes[0]
    assert "distinct" in context.notes[0]


# -- E3: streams ---------------------------------------------------------


def test_streams_that_disagree_turn_the_stream_half_red() -> None:
    """Two streams answering differently is the failure the stream half exists for."""
    backend = Streaming()

    def per_stream(*args: Any, **kwargs: Any) -> torch.Tensor:
        base = GELU.forward(*args, **kwargs)
        return base + (1e-3 if backend.current.index == 1 else 0.0)

    with pytest.raises(AssertionError, match="disagreed with the serial one"):
        e3.run(_context(GELU, backend, "E3", per_stream))


def test_two_streams_sharing_one_scratch_buffer_turn_the_stream_half_red() -> None:
    """The answer of the first launch is handed to the second one."""
    state: dict[str, Any] = {"calls": 0, "last": None}

    def reuses_its_last(*args: Any, **kwargs: Any) -> torch.Tensor:
        state["calls"] += 1
        if state["last"] is None or state["calls"] == 1:
            state["last"] = GELU.forward(*args, **kwargs)
        return state["last"]

    with pytest.raises(AssertionError):
        e3.run(_context(GELU, Streaming(), "E3", reuses_its_last))


def test_the_stream_half_passes_on_a_clean_operator() -> None:
    context = _context(GELU, Streaming(), "E3")
    e3.run(context)
    assert "stream API" in context.notes[0]
    assert "2 streams" in context.notes[0]


# -- E6: one device ------------------------------------------------------


def test_a_batch_split_that_changes_the_answer_turns_the_row_red() -> None:
    """A kernel with a per-launch bias is fine whole and wrong in pieces."""

    def biased_by_length(x: torch.Tensor, *rest: Any, **kwargs: Any) -> torch.Tensor:
        base = LAYER_NORM.forward(x, *rest, **kwargs)
        return base + (1e-3 if x.shape[0] < 8 else 0.0)

    with pytest.raises(AssertionError, match="cut up"):
        e6.run(_context(LAYER_NORM, CpuBackend(), "E6", biased_by_length))


def test_a_group_table_counted_from_zero_turns_the_row_red() -> None:
    """A grouped call that ignores its own row offset fails only when split."""

    def counts_from_zero(
        a: torch.Tensor, b: torch.Tensor, group_list: torch.Tensor
    ) -> torch.Tensor:
        return GEMM.forward(a * (0.0 + 1.0), b, group_list) + (a.shape[0] - 64) * 1e-3

    with pytest.raises(AssertionError, match="differ"):
        e6.run(_context(GEMM, CpuBackend(), "E6", counts_from_zero))


def test_the_single_device_half_passes_on_a_clean_operator() -> None:
    """Both shapes of split -- batched and grouped -- must go green on the CPU."""
    batch = _context(LAYER_NORM, CpuBackend(), "E6")
    e6.run(batch)
    assert "single device" in batch.notes[0]

    grouped = _context(GEMM, CpuBackend(), "E6")
    e6.run(grouped)
    assert "groups split" in grouped.notes[0]


# -- E6: two devices -----------------------------------------------------


class Devices(CpuBackend):
    """A stub backend advertising more devices than the machine has."""

    def __init__(self, count: int) -> None:
        super().__init__()
        self._count = count

    def device_count(self) -> int:
        return self._count


def test_a_second_device_that_disagrees_turns_the_row_red() -> None:
    """The failure the multi-device half exists for."""

    calls = {"n": 0}

    def second_device_drifts(*args: Any, **kwargs: Any) -> torch.Tensor:
        """Device 0 is right, device 1 comes back with a different last bit."""
        calls["n"] += 1
        answer = LAYER_NORM.forward(*args, **kwargs)
        return answer + (1e-3 if calls["n"] > 1 else 0.0)

    with pytest.raises(AssertionError, match="device 1 differs"):
        e6.run(_context(LAYER_NORM, Devices(2), "E6", second_device_drifts))


def test_the_multi_device_half_records_how_many_devices_it_saw() -> None:
    context = _context(LAYER_NORM, Devices(4), "E6")
    e6.run(context)
    assert context.notes[0].startswith("4 devices")
