"""What the part 7.2 matrix owes, cell by cell -- for readers outside this repo.

The reasons in :mod:`neutraltest.gbz72.reasons` are written once per feature,
which is right for a human reading the table but not enough for a report: an
outside tool that wants a legend per cell needs the pair resolved -- is it marked
and exercised, and if not, which class of not.  This module is that resolution
and nothing else, so a table builder imports one function and never reaches into
the runner or the applicability tables itself.

``--catalogue`` on the command line prints :func:`catalogue`, so a consumer can
also stay out of the library API entirely.
"""

from __future__ import annotations

import json
from typing import Any

from . import marks, reasons


def cause(feature_id: str, column: str) -> str | None:
    """The class recorded for ``feature_id``/``column``, or ``None`` if it runs.

    ``None`` means the table claims the cell and a script answers it: the cell's
    outcome is then whatever the run reports (``passed``/``failed``/``known``),
    and no reason class applies.
    """
    if marks.is_marked(feature_id, column):
        return None
    return reasons.cause_of(feature_id, column)


def explain(feature_id: str, column: str) -> tuple[str | None, str]:
    """``(cause, text)`` for one cell; ``(None, "")`` when the cell is exercised."""
    resolved = cause(feature_id, column)
    if resolved is None:
        return None, ""
    return resolved, reasons.reason_for(feature_id, column)


def symbol(feature_id: str, column: str) -> str:
    """One glyph for a cell that is not exercised; ``""`` when it is."""
    resolved = cause(feature_id, column)
    return "" if resolved is None else reasons.CAUSE_SYMBOLS[resolved]


def legend() -> dict[str, str]:
    """Class -> one-line meaning, for a legend under a rendered table."""
    return dict(reasons.CAUSE_TITLES)


def catalogue() -> dict[str, Any]:
    """The whole 7.2 table as data: classes, legend, and every cell.

    Each cell carries ``feature``, ``column``, ``marked`` (the table claims it),
    ``cause`` (``None`` when it is exercised) and ``reason``.  ``marked`` and
    ``cause is None`` are the same statement today; both are emitted so a
    consumer does not have to know that.
    """
    cells = [
        {
            "feature": feature_id,
            "column": column,
            "marked": marks.is_marked(feature_id, column),
            "cause": cause(feature_id, column),
            "reason": explain(feature_id, column)[1],
        }
        for feature_id in marks.APPLICABILITY
        for column in marks.TABLE_ORDER
    ]
    return {
        "version": 1,
        "clause": "7.2",
        "marked": marks.MARKED,
        "causes": list(reasons.CAUSES),
        "legend": legend(),
        "symbols": dict(reasons.CAUSE_SYMBOLS),
        "columns": list(marks.TABLE_ORDER),
        "cells": cells,
    }


def dumps(indent: int | None = 2) -> str:
    """The catalogue as JSON text."""
    return json.dumps(catalogue(), ensure_ascii=False, indent=indent)
