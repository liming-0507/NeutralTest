"""The sampler's own matrix: twelve rows, one direction.

Part 7.2 names seven operators. Six are columns of the feature matrix; the
seventh is not, and this module is where that difference is made explicit
rather than papered over.

Why a separate matrix instead of two more columns:

* **There is no backward.** Sampling draws from a distribution. The draw is not
  differentiable in the logits -- no framework ships a gradient for it -- so a
  backward column would not be an untested cell, it would be a cell that cannot
  exist. Listing it beside ``add_bwd`` would imply it could pass.
* **The oracle is different in kind.** Every other column is compared against a
  reference element by element. A sampler's output is an index, and two correct
  samplers disagree on any draw, so the comparison moves from the draw to the
  distribution it was drawn from (see :mod:`.oracle`).

What does *not* differ is the bookkeeping: the same outcome vocabulary, the same
``P/F/-`` symbols, the same disclosure of what was not exercised, so the two
matrices read side by side.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from .items import ROWS, SamplingRow
from .ops import reference_mass
from .oracle import DRAWS, assert_matches

SYMBOLS = {"passed": "P", "failed": "F", "na": "-", "skipped": "·"}
VERDICTS = ("passed", "failed", "na", "skipped")

#: The filters every sampler is asked to support. ``None`` is plain multinomial
#: sampling, which is the operator without either knob turned on.
SETTINGS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("plain", {}),
    ("top_k", {"top_k": 4}),
    ("top_p", {"top_p": 0.75}),
    ("top_k+top_p", {"top_k": 4, "top_p": 0.75}),
    ("temperature", {"temperature": 0.7}),
)


@dataclass(slots=True)
class Result:
    """One (row, setting) cell."""

    row: str
    setting: str
    outcome: str
    detail: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.outcome in ("passed", "na", "skipped")


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)

    def add(self, result: Result) -> None:
        self.results.append(result)

    def by_outcome(self, outcome: str) -> list[Result]:
        return [r for r in self.results if r.outcome == outcome]

    @property
    def failed(self) -> list[Result]:
        return self.by_outcome("failed")

    def cell(self, row: str, setting: str) -> Result | None:
        for result in self.results:
            if result.row == row and result.setting == setting:
                return result
        return None

    def as_dict(self) -> dict[str, Any]:
        counts = {outcome: len(self.by_outcome(outcome)) for outcome in VERDICTS}
        return {
            "cells": counts,
            "results": [
                {
                    "row": r.row,
                    "setting": r.setting,
                    "outcome": r.outcome,
                    "detail": r.detail,
                    "seconds": round(r.seconds, 6),
                }
                for r in self.results
            ],
        }


#: The reference shape: a few rows of a modest vocabulary. Small enough that a
#: 4096-draw distribution check finishes quickly, large enough that Top-K and
#: Top-P are not trivially the same set.
SHAPE = {"rows": 4, "vocabulary": 16}


def _logits(seed: int = 11) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(SHAPE["rows"], SHAPE["vocabulary"], generator=generator)


def run_cell(
    row: SamplingRow,
    setting: str,
    kwargs: dict[str, Any],
    device_draw: Any,
    *,
    draws: int = DRAWS,
) -> Result:
    """Check one row under one setting, on the device under test.

    A row whose claim is about the filter is checked once (on the CPU) rather
    than once per generator state: repeating it changes nothing and would
    triple the runtime to say the same thing three times.

    The distribution claim is the one every backend is held to. On the CPU the
    kernel *is* the reference, so the claim reduces to "the reference is
    consistent with itself" -- still worth running, because a broken oracle
    would otherwise only be discovered on hardware.
    """
    started = time.perf_counter()
    try:
        logits = _logits()
        if device_draw is None:
            row.check()
            return Result(row.id, setting, "passed", "", time.perf_counter() - started)
        assert_matches(
            device_draw,
            logits,
            draws=draws,
            msg=f"{row.id}/{setting}",
            **kwargs,
        )
        return Result(row.id, setting, "passed", "", time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 - the outcome is data, not control flow
        return Result(
            row.id,
            setting,
            "failed",
            f"{type(exc).__name__}: {exc}",
            time.perf_counter() - started,
        )


def run_all(
    device_draw: Any = None,
    *,
    rows: tuple[SamplingRow, ...] = ROWS,
    settings: tuple[tuple[str, dict[str, Any]], ...] = SETTINGS,
    draws: int = DRAWS,
) -> Report:
    """Every row against every setting."""
    report = Report()
    for row in rows:
        for name, kwargs in settings:
            report.add(run_cell(row, name, kwargs, device_draw, draws=draws))
    return report


def _symbol(result: Result | None) -> str:
    return SYMBOLS.get(result.outcome, "?") if result else "?"


def render_matrix(report: Report, *, width: int = 22) -> str:
    """The row × setting grid."""
    settings = [name for name, _ in SETTINGS]
    header = f"{'id':<4} {'subject':<{width}} " + " ".join(f"{s[:12]:>12}" for s in settings)
    lines = [header]
    for row in ROWS:
        cells = " ".join(f"{_symbol(report.cell(row.id, s)):>12}" for s in settings)
        lines.append(f"{row.id:<4} {row.subject:<{width}} {cells}")
    return "\n".join(lines)


def render_summary(report: Report) -> str:
    counts = {outcome: len(report.by_outcome(outcome)) for outcome in VERDICTS}
    lines = ["cells: " + ", ".join(f"{k}: {counts[k]}" for k in VERDICTS)]
    for result in report.failed:
        lines.append(f"FAILED {result.row}/{result.setting}: {result.detail}")
    return "\n".join(lines)


def describe_reference() -> str:
    """One line stating what the oracle is, for the report header."""
    logits = _logits()
    mass = reference_mass(logits)
    return (
        f"sampler reference: {SHAPE['rows']}×{SHAPE['vocabulary']} logits, "
        f"{DRAWS} draws per cell, compared as a distribution "
        f"(max reference mass {float(mass.max()):.3f})"
    )
