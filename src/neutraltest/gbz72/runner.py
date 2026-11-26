"""Runner for the part 7.2 feature matrix.

One cell is a (feature, column) pair: a numbered feature of the catalogue
exercised against one of the twelve columns of the clause.  The machinery is the
one part 7.3 uses -- the same outcome vocabulary, the same ``P/F/-/K`` symbols,
the same disclosure of everything that was not exercised -- so the two matrices
read the same way and can be printed by the same command.

The difference is the gate.  A 7.3 cell is skipped when the item module declines
the operator; here the applicability table itself decides, and a pair the table
does not mark is disclosed with the reason recorded for its feature.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..gbz73.features import FEATURES, Feature
from ..gbz73.inference.columns import BY_NAME, Column
from . import marks, reasons

SYMBOLS = {
    "passed": "P",
    "failed": "F",
    "na": "-",
    "known": "K",
    "unsupported": "U",
    "not_run": "·",
}
VERDICTS = ("passed", "failed", "na", "known", "unsupported")

#: The glyph for a pass on a pair the applicability table does not mark: the row
#: runs, the result is real, but the clause does not ask this column for it. It
#: is a pass and counts as one; it is spelled differently so that a table quoted
#: as coverage does not grow cells nobody claimed.
OFF_TABLE = "p"

#: What :data:`OFF_TABLE` means, printed under every matrix that can show it.
OFF_TABLE_LEGEND = f"{OFF_TABLE} = passed, though the clause does not ask this column to"


@dataclass(slots=True)
class Result:
    """One (feature, column) cell."""

    feature: str
    column: str
    outcome: str
    detail: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        """Whether the cell is not a failure.

        ``unsupported`` counts as not-a-failure for the same reason the
        conformance engine treats it that way: the clause asks for a column the
        backend has no kernel for, which is a property of that backend build.
        The cell still appears in the matrix under its own symbol, so the
        disclosure is not lost -- only the gate stops blaming this harness.
        """
        return self.outcome in ("passed", "na", "known", "unsupported")

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "column": self.column,
            "outcome": self.outcome,
            "detail": self.detail,
            "cause": self.cause,
            "seconds": round(self.seconds, 6),
        }

    @property
    def cause(self) -> str | None:
        """The reason class of a cell that was not exercised, else ``None``.

        Derived rather than stored: a marked pair has no class by construction,
        and a runtime not-applicable is a class of its own (the scenario exists
        and this sample did not reach it), so the two cannot drift apart.
        """
        if self.outcome == "unsupported":
            return "unsupported_by_backend"
        if self.outcome != "na":
            return None
        return reasons.cause_of(self.feature, self.column)


@dataclass
class Report:
    """The cells of one run, in the order they were produced."""

    results: list[Result] = field(default_factory=list)
    columns: tuple[str, ...] = marks.TABLE_ORDER
    fp32_precision: str = ""

    def add(self, result: Result) -> None:
        self.results.append(result)

    def cell(self, feature: str, column: str) -> Result | None:
        for result in self.results:
            if result.feature == feature and result.column == column:
                return result
        return None

    def by_outcome(self, outcome: str) -> list[Result]:
        return [r for r in self.results if r.outcome == outcome]

    @property
    def failed(self) -> list[Result]:
        return self.by_outcome("failed")

    def as_dict(self) -> dict[str, Any]:
        counts = {outcome: len(self.by_outcome(outcome)) for outcome in VERDICTS}
        return {
            "totals": counts,
            "fp32_precision": self.fp32_precision,
            "columns": list(self.columns),
            "results": [r.as_dict() for r in self.results],
        }


def run_column(feature: Feature, column: Column, backend: Any, *, seed: int = 0) -> Result:
    """Execute one feature against one column, capturing the outcome."""
    from . import items

    started = time.perf_counter()
    if not marks.is_marked(feature.id, column.name):
        return Result(
            feature.id,
            column.name,
            "na",
            reasons.reason_for(feature.id, column.name),
            0.0,
        )
    try:
        from ..gbz73.inference.context import InferenceContext

        module = items.module_for(feature.id)
        if module.COLUMNS is not None and column.name not in module.COLUMNS:
            return Result(
                feature.id,
                column.name,
                "na",
                f"the script for {feature.id} does not apply to {column.family} columns",
                0.0,
            )
        context = InferenceContext(column=column, backend=backend, feature_id=feature.id, seed=seed)
        module.run(context)
        return Result(feature.id, column.name, "passed", "", time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 - the outcome is data, not control flow
        from ..gbz73.context import NotApplicable
        from ..gbz73.items._support import NotVerifiable

        if isinstance(exc, (NotApplicable, NotVerifiable)):
            reason = exc.reason if isinstance(exc, NotApplicable) else str(exc)
            return Result(feature.id, column.name, "na", reason, time.perf_counter() - started)
        if _unsupported(feature, column, backend, exc):
            return Result(
                feature.id,
                column.name,
                "unsupported",
                f"{type(exc).__name__}: {exc}",
                time.perf_counter() - started,
            )
        entry = _known_entry(feature, column, backend)
        if entry is not None:
            return Result(
                feature.id,
                column.name,
                "known",
                f"KNOWN[{entry.id}] {entry.summary}; measured {entry.deviation()}",
                time.perf_counter() - started,
            )
        return Result(
            feature.id,
            column.name,
            "failed",
            f"{type(exc).__name__}: {exc}",
            time.perf_counter() - started,
        )


def _unsupported(feature: Feature, column: Column, backend: Any, exc: BaseException) -> bool:
    """Whether *exc* is "this backend has no kernel", under the active policy.

    The conformance engine and the pytest plugin both honour ``on_unsupported``;
    a test that calls :func:`run_column` gets a :class:`Result` instead of an
    exception, so nothing downstream can apply the policy for it.  Without this
    the option would silently cover only part of the suite -- and the CI leg for
    the oldest supported torch, which sets it precisely because that build lacks
    half-precision kernels, would fail on a property of the wheel.
    """
    from ..errors import is_unsupported_error
    from ..policy import get_policy

    if not is_unsupported_error(exc):
        return False
    return get_policy().on_unsupported != "fail"


def _known_entry(feature: Feature, column: Column, backend: Any):
    """The registered deviation covering this cell, if the policy wants one."""
    from .. import known
    from ..policy import get_policy

    if get_policy().known == "strict":
        return None
    return known.registry().find(f"gbz72/{feature.id}/{column.name}", "float32", backend)


def run_all(
    backend: Any,
    *,
    columns: tuple[Column, ...] | None = None,
    features: tuple[Feature, ...] | None = None,
    seed: int = 0,
    fp32_precision: str | None = None,
) -> Report:
    """Run every selected feature against every selected column."""
    from .. import precision

    chosen_columns = _table_columns() if columns is None else columns
    chosen_features = tuple(FEATURES if features is None else features)
    report = Report(results=[], columns=tuple(c.name for c in chosen_columns))
    if fp32_precision is not None:
        report.fp32_precision = precision.pin(fp32_precision)
    else:
        report.fp32_precision = precision.describe()
    for feature in chosen_features:
        for column in chosen_columns:
            report.add(run_column(feature, column, backend, seed=seed))
    return report


def _symbol(result: Result | None, feature: Feature | None = None) -> str:
    if result is None:
        return SYMBOLS["not_run"]
    if (
        result.outcome == "passed"
        and feature is not None
        and not marks.is_marked(feature.id, result.column)
    ):
        return OFF_TABLE
    return SYMBOLS.get(result.outcome, "?")


def off_table_passes(report: Report) -> int:
    """Cells that passed although the applicability table leaves the pair unmarked.

    They are passes, and every count below still counts them as passes; this is a
    second reading of the same run, for the one thing the counts cannot say: how
    much of what passed was asked for.
    """
    return sum(
        1
        for result in report.results
        if result.outcome == "passed" and not marks.is_marked(result.feature, result.column)
    )


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def applied(feature: Feature, columns: tuple[Column, ...] | None = None) -> str:
    """The applicability bits the table records for ``feature``, in column order."""
    chosen = _table_columns() if columns is None else columns
    return "".join("1" if marks.is_marked(feature.id, c.name) else "0" for c in chosen)


def _table_columns() -> tuple[Column, ...]:
    """The twelve columns as :class:`Column` objects, in the standard's order."""
    return tuple(BY_NAME[name] for name in marks.TABLE_ORDER)


def asks(feature: Feature, column: Column) -> bool:
    """Whether the standard's text states a requirement for this (row, column).

    Read from the column, not from the row: ``Column.standard_covers`` is the
    transcription of the clause, and the applicability table above is the
    judgement of what to run. They agree today -- ``tests/unit/test_gbz72_columns_bits.py``
    holds them to it -- but they answer different questions, and the matrix has
    to keep the answers apart to be able to print both.
    """
    return column.standard_covers(feature.id)


def render_matrix(
    report: Report, *, columns: tuple[Column, ...] | None = None, width: int = 34
) -> str:
    """The feature × column grid, one row per catalogue entry."""
    columns = _table_columns() if columns is None else columns
    header = f"{'id':<4} {'tier':<12} {'feature':<{width}} {'text':<{len(columns)}} " + " ".join(
        f"{c.name[:6]:>6}" for c in columns
    )
    lines = [header]
    for feature in FEATURES:
        # The bits are printed for the columns actually shown, so the row of 1/0
        # lines up with the row of cells under it.
        bits = "".join("1" if asks(feature, c) else "0" for c in columns)
        cells = " ".join(f"{_symbol(report.cell(feature.id, c.name), feature):>6}" for c in columns)
        lines.append(
            f"{feature.id:<4} {feature.tier:<12} {_clip(feature.title, width):<{width}} "
            f"{bits:<{len(columns)}} {cells}"
        )
    lines.append("-" * len(header))
    lines.append(OFF_TABLE_LEGEND)
    return "\n".join(lines)


def render_summary(report: Report) -> str:
    """Totals, per-column coverage, and the first failures."""
    counts = {outcome: len(report.by_outcome(outcome)) for outcome in VERDICTS}
    lines = ["cells: " + ", ".join(f"{outcome}: {counts[outcome]}" for outcome in VERDICTS)]
    for name in report.columns:
        cells = [r for r in report.results if r.column == name]
        passed = sum(1 for r in cells if r.outcome == "passed")
        na = sum(1 for r in cells if r.outcome == "na")
        known = sum(1 for r in cells if r.outcome == "known")
        suffix = f" known {known:>2}" if known else ""
        lines.append(f"  {name:<20} passed {passed:>2}/{len(cells):<3} na {na:>2}{suffix}")
    for result in report.failed[:10]:
        lines.append(f"  FAIL {result.feature}/{result.column}: {_clip(result.detail, 120)}")
    lines.append(f"off-standard passes: {off_table_passes(report)} cell(s)")
    return "\n".join(lines)


def render_disclosure(report: Report) -> str:
    """One line per cell that was not exercised, with its class and its reason.

    The class comes first because it is what a reader triages on: the cells that
    cannot exist and the cells nobody has written yet otherwise read the same
    here, which is how a table of excuses hides its own gaps.
    """
    lines = [f"{r.feature}/{r.column} [{r.cause}]: {r.detail}" for r in report.by_outcome("na")]
    return "\n".join(lines)


def coverage_gap(report: Report) -> tuple[int, int]:
    """``(cells the table declares, cells this run exercised)``."""
    declared = len(marks.marked_pairs())
    exercised = sum(1 for r in report.results if r.outcome in ("passed", "failed"))
    return declared, exercised
