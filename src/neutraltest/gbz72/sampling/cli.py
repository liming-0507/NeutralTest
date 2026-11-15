"""Render the part 7.2 sampler matrix, with or without a device.

Kept out of :mod:`neutraltest.gbz72.sampling.runner` so the runner stays free of
console and argument concerns, and out of :mod:`neutraltest.cli` so that module
does not grow a fourth thing to import lazily.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from .oracle import DRAWS
from .runner import Report, describe_reference, render_matrix, render_summary, run_all


def _device_draw(backend: Any, draws: int):
    """The backend's sampling, drawn under a fixed seed per call."""
    if backend.device_type == "cpu":
        return None

    def device_draw(logits, *, seed=0, num_samples=1, **kwargs):
        backend.manual_seed(seed)
        from .ops import draw as reference_draw

        out = reference_draw(logits.to(backend.make_device(0)), num_samples=num_samples, **kwargs)
        backend.synchronize()
        return out.to("cpu")

    return device_draw


def run(backend: Any = None, *, draws: int = DRAWS) -> Report:
    return run_all(None if backend is None else _device_draw(backend, draws), draws=draws)


def render(report: Report, *, backend: Any = None) -> str:
    where = "cpu (reference)" if backend is None else f"{backend.name}"
    return "\n".join(
        [
            f"backend: {where}",
            describe_reference(),
            "",
            render_matrix(report),
            "",
            render_summary(report),
        ]
    )


def main(backend: Any = None, *, as_json: bool = False, draws: int = DRAWS) -> int:
    report = run(backend, draws=draws)
    if as_json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(render(report, backend=backend))
    if report.failed:
        print(f"error: {len(report.failed)} sampler cell(s) failed", file=sys.stderr)
        return 1
    return 0
