"""Result collection for the pytest terminal summary and the JSON report."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Result", "Recorder", "SUMMARY_OUTCOMES"]

SUMMARY_OUTCOMES = ("passed", "failed", "skipped", "xfailed", "xpassed", "error")


@dataclass
class Result:
    """One test outcome, attributed to a backend."""

    nodeid: str
    backend: str
    outcome: str
    duration: float = 0.0
    family: str = ""
    op: str = ""
    dtype: str = ""
    unsupported: bool = False
    message: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return self.backend, self.family, self.op


@dataclass
class Recorder:
    """Accumulates results; also renders the summary table."""

    results: list[Result] = field(default_factory=list)

    def add(self, result: Result) -> None:
        self.results.append(result)

    # ------------------------------------------------------------------ summary

    def by_backend(self) -> dict[str, Counter[str]]:
        table: dict[str, Counter[str]] = defaultdict(Counter)
        for result in self.results:
            table[result.backend][result.outcome] += 1
            if result.unsupported:
                table[result.backend]["unsupported"] += 1
        return dict(table)

    def by_family(self) -> dict[str, Counter[str]]:
        table: dict[str, Counter[str]] = defaultdict(Counter)
        for result in self.results:
            table[result.family or "<ungrouped>"][result.outcome] += 1
        return dict(table)

    def duration_by_backend(self) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for result in self.results:
            totals[result.backend] += result.duration
        return dict(totals)

    def totals(self) -> Counter[str]:
        counter: Counter[str] = Counter()
        for result in self.results:
            counter[result.outcome] += 1
        return counter

    # ------------------------------------------------------------------ output

    def as_dict(self) -> dict[str, Any]:
        backends = self.by_backend()
        return {
            "totals": dict(self.totals()),
            "backends": {name: dict(counts) for name, counts in backends.items()},
            "families": {name: dict(counts) for name, counts in self.by_family().items()},
            "duration_seconds": round(sum(self.duration_by_backend().values()), 3),
            "results": [asdict(result) for result in self.results],
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        return target


def render_table(rows: Iterable[tuple[str, ...]], headers: tuple[str, ...]) -> str:
    """Fixed-width table, dependency free (the CLI must not need ``rich``)."""
    body = [tuple(str(cell) for cell in row) for row in rows]
    widths = [len(header) for header in headers]
    for row in body:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)).rstrip()
    ]
    lines.append("  ".join("-" * width for width in widths))
    for row in body:
        lines.append(
            "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip()
        )
    return "\n".join(lines)
