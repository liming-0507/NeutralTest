"""Run the 45 features over the operator columns and render the matrix.

The matrix is the deliverable: one row per feature, one column per operator,
and, next to the results, the coverage the standard's own text declares. The
two are printed together on purpose -- the difference is the finding.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..backends.base import Backend
from .features import COLUMNS, FEATURES, Feature
from .ops import OPS, Op

SYMBOLS = {"passed": "P", "failed": "F", "na": "-", "known": "K", "not_run": "\u00b7"}
VERDICTS = ("passed", "failed", "na", "known")


@dataclass(slots=True)
class Result:
    """Outcome of one (feature, operator) cell."""

    feature: str
    op: str
    outcome: str
    detail: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.outcome == "passed"

    def as_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "op": self.op,
            "outcome": self.outcome,
            "detail": self.detail,
            "seconds": round(self.seconds, 6),
        }


@dataclass(slots=True)
class Report:
    """All cells of one run, plus the bookkeeping the summary needs."""

    results: list[Result] = field(default_factory=list)
    fp32_precision: str = ""

    def add(self, result: Result) -> None:
        self.results.append(result)

    def cell(self, feature: str, op: str) -> Result | None:
        for result in self.results:
            if result.feature == feature and result.op == op:
                return result
        return None

    def by_outcome(self, outcome: str) -> list[Result]:
        return [r for r in self.results if r.outcome == outcome]

    @property
    def failed(self) -> list[Result]:
        return self.by_outcome("failed")

    def as_dict(self) -> dict[str, object]:
        counts = {outcome: len(self.by_outcome(outcome)) for outcome in VERDICTS}
        return {
            "totals": counts,
            "fp32_precision": self.fp32_precision,
            "ops": list(COLUMNS),
            "results": [r.as_dict() for r in self.results],
        }


def run_feature(feature: Feature, op: Op, backend: Backend, *, seed: int = 0) -> Result:
    """Execute one feature against one operator, capturing the outcome."""
    # Imported here so that the item package can import this module without a cycle.
    from . import items

    module = items.module_for(feature.id)
    started = time.perf_counter()
    try:
        context = (
            module.make_context(op, backend, feature, seed=seed)
            if hasattr(module, "make_context")
            else None
        )
        if context is None:
            from .context import RunContext

            context = RunContext(op=op, backend=backend, feature=feature, seed=seed)
        if module.OPS is not None and op.name not in module.OPS:
            return Result(
                feature.id, op.name, "na", f"not applicable to {op.family} operators", 0.0
            )
        module.run(context)
        return Result(feature.id, op.name, "passed", "", time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 - the outcome is data, not control flow
        from .context import NotApplicable
        from .items._support import NotVerifiable

        if isinstance(exc, (NotApplicable, NotVerifiable)):
            reason = exc.reason if isinstance(exc, NotApplicable) else str(exc)
            return Result(feature.id, op.name, "na", reason, time.perf_counter() - started)
        entry = _known_entry(feature, op, backend)
        if entry is not None:
            return Result(
                feature.id,
                op.name,
                "known",
                f"KNOWN[{entry.id}] {entry.summary}; measured {entry.deviation()}",
                time.perf_counter() - started,
            )
        return Result(
            feature.id,
            op.name,
            "failed",
            f"{type(exc).__name__}: {exc}",
            time.perf_counter() - started,
        )


def _known_entry(feature: Feature, op: Op, backend: Backend):
    """The registered deviation covering this cell, if the policy wants one."""
    from .. import known
    from ..policy import get_policy

    if get_policy().known == "strict":
        return None
    return known.registry().find(f"gbz73/{feature.id}/{op.name}", "float32", backend)


def run_all(
    backend: Backend,
    *,
    ops: tuple[Op, ...] | None = None,
    features: tuple[Feature, ...] | None = None,
    seed: int = 0,
    fp32_precision: str | None = None,
) -> Report:
    """Run every feature against every operator column.

    ``fp32_precision`` pins the float32 matmul contract first (``highest`` keeps
    a 24-bit mantissa); ``None`` leaves the session's own setting untouched and
    records whatever it happens to be.
    """
    from .. import precision

    report = Report()
    report.fp32_precision = (
        precision.pin(fp32_precision) if fp32_precision else precision.describe()
    )
    for feature in features or FEATURES:
        for op in ops or OPS:
            report.add(run_feature(feature, op, backend, seed=seed))
    return report


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def render_matrix(report: Report, *, ops: tuple[str, ...] = COLUMNS, width: int = 34) -> str:
    """The coverage matrix: our results beside the coverage the text declares."""
    header = f"{'id':<4} {'tier':<12} {'feature':<{width}} {'text':<6} " + " ".join(
        f"{o[:6]:>6}" for o in ops
    )
    lines = [header, "-" * len(header)]
    for feature in FEATURES:
        cells = []
        for op in ops:
            result = report.cell(feature.id, op)
            cells.append(
                f"{SYMBOLS.get(result.outcome, '?'):>6}" if result else f"{SYMBOLS['not_run']:>6}"
            )
        bits = feature.bits[: len(ops)] if len(ops) <= len(COLUMNS) else feature.bits
        lines.append(
            f"{feature.id:<4} {feature.tier:<12} {_clip(feature.title, width):<{width}} {bits:<6} "
            + " ".join(cells)
        )
    return "\n".join(lines)


def render_summary(report: Report) -> str:
    """Counts per outcome and per operator, plus the first failures."""
    counts = {outcome: len(report.by_outcome(outcome)) for outcome in VERDICTS}
    parts = [f"{outcome}: {counts[outcome]}" for outcome in VERDICTS]
    lines = ["cells: " + ", ".join(parts)]
    for op in COLUMNS:
        cells = [r for r in report.results if r.op == op]
        if cells:
            passed = sum(1 for r in cells if r.outcome == "passed")
            na = sum(1 for r in cells if r.outcome == "na")
            known_cells = sum(1 for r in cells if r.outcome == "known")
            line = f"  {op:<14} passed {passed:>2}/{len(cells):<3} na {na:>2}"
            if known_cells:
                line += f" known {known_cells:>2}"
            lines.append(line)
    for result in report.failed[:10]:
        lines.append(f"  FAIL {result.feature}/{result.op}: {_clip(result.detail, 120)}")
    return "\n".join(lines)


def render_disclosure(report: Report) -> str:
    """The N/A list, one line per cell, with the reason attached."""
    lines = []
    for result in report.by_outcome("na"):
        lines.append(f"{result.feature}/{result.op}: {result.detail}")
    return "\n".join(lines)


def coverage_gap(report: Report) -> tuple[int, int]:
    """(cells the text declares, cells this run actually exercises)."""
    declared = sum(1 for f in FEATURES for op in COLUMNS if f.covered_by_standard(op))
    exercised = sum(1 for r in report.results if r.outcome in {"passed", "failed"})
    return declared, exercised
