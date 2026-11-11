"""Clause-aware facade over the two matrices.

The command line and the JSON report need one entry point that knows which
matrices exist.  Part 7.3 is a feature x operator matrix; part 7.2 is a feature x
column matrix over the same 45-feature catalogue.  A :class:`Run` pairs one
clause's report with the columns it was produced for, so the printer does not
have to branch on the clause at every line.

Nothing here changes a single-clause run: ``--clause 73`` produces exactly the
report, the summary, the disclosure and the coverage line the command produced
before this module existed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .clause import selection
from .gbz72 import runner as matrix72
from .gbz72.marks import COLUMN_NAMES, STANDARD_COLUMNS, TABLE_ORDER
from .gbz73 import runner as matrix73
from .gbz73.inference.columns import BY_NAME, COLUMNS
from .gbz73.ops import OPS


@dataclass(frozen=True)
class Run:
    """One clause's report plus the columns/operators it covers."""

    clause: str
    report: Any
    columns: tuple[Any, ...]

    def render(self) -> str:
        if self.clause == "72":
            return matrix72.render_matrix(self.report, columns=self.columns)
        names = tuple(item.name for item in self.columns)
        return matrix73.render_matrix(self.report, ops=names)

    def summary(self) -> str:
        if self.clause == "72":
            return matrix72.render_summary(self.report)
        return matrix73.render_summary(self.report)

    def disclosure(self) -> str:
        if self.clause == "72":
            return matrix72.render_disclosure(self.report)
        return matrix73.render_disclosure(self.report)

    def coverage(self) -> tuple[int, int]:
        if self.clause == "72":
            return matrix72.coverage_gap(self.report)
        return matrix73.coverage_gap(self.report)

    def as_dict(self) -> dict[str, Any]:
        payload = self.report.as_dict()
        payload["clause"] = self.clause
        return payload


def _table_columns() -> tuple[Any, ...]:
    """The twelve columns in the order the standard's table lists them."""
    return tuple(BY_NAME[name] for name in TABLE_ORDER)


def column_names() -> tuple[str, ...]:
    """Every way a part 7.2 column may be named on the command line."""
    return tuple(COLUMNS) + STANDARD_COLUMNS


def resolve_columns(names: tuple[str, ...] | None) -> tuple[Any, ...]:
    """Map ``--ops`` values onto part 7.2 columns.

    Both the column identifiers (``add_fwd``) and the standard's own labels
    (``Add-前``) are accepted.  An empty selection means every column.
    """
    if not names:
        return _table_columns()
    selected = []
    for name in names:
        key = COLUMN_NAMES.get(name, name)
        if key not in BY_NAME:
            raise KeyError(name)
        selected.append(BY_NAME[key])
    return tuple(selected)


def resolve_operators(names: tuple[str, ...] | None) -> tuple[Any, ...]:
    """Map ``--ops`` values onto part 7.3 operators; empty means every operator."""
    if not names:
        return OPS
    known = {operator.name: operator for operator in OPS}
    selected = []
    for name in names:
        if name not in known:
            raise KeyError(name)
        selected.append(known[name])
    return tuple(selected)


def _narrow(resolve: Any, names: tuple[str, ...] | None, everything: tuple[Any, ...]) -> Any:
    """Apply a ``--ops`` filter, or keep every column when it names none of ours.

    Under ``--clause all`` a filter that only names operators (``relu``) or only
    names columns (``add_fwd``) still has to leave the other matrix whole rather
    than fail or disappear.
    """
    if not names:
        return everything
    try:
        return resolve(names)
    except KeyError:
        return everything


def run(
    backend: Any,
    *,
    clause: str = "73",
    ops: tuple[str, ...] | None = None,
    features: tuple[Any, ...] | None = None,
    fp32_precision: str | None = None,
    seed: int = 0,
) -> list[Run]:
    """Produce one report per clause the selection covers."""
    runs: list[Run] = []
    for item in selection(clause):
        if item == "72":
            columns = _narrow(resolve_columns, ops, _table_columns())
            report: Any = matrix72.run_all(
                backend,
                columns=columns,
                features=features,
                seed=seed,
                fp32_precision=fp32_precision,
            )
        else:
            operators = _narrow(resolve_operators, ops, OPS)
            report = matrix73.run_all(
                backend,
                ops=operators,
                features=features,
                seed=seed,
                fp32_precision=fp32_precision,
            )
            columns = operators
        runs.append(Run(clause=item, report=report, columns=tuple(columns)))
    return runs
