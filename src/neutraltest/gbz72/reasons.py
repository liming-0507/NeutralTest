"""Why a (feature, column) pair of part 7.2 is not exercised, and of what kind.

The applicability table in :mod:`neutraltest.gbz72.marks` says *which* pairs the
clause covers.  This module says why the others are left out, one recorded
reason per feature: the row is disclosed as not applicable on a column rather
than silently dropped, exactly as the part 7.3 side discloses its own
not-applicable cells.

A reason is written once per feature, not per pair, because a row fails to apply
for one structural reason across all the columns it misses.  Twelve features
are not applied to any of the twelve columns; for those the same text is the
reason for the whole row and is what :func:`reason_for` returns for every column.

A row that is exercised on every column has no entry at all -- E1 and E2, which
establish determinism and the absence of a random source for all twelve. Their
verdict lives in the item script rather than in this table, so keeping a
"not applicable" sentence here would be a second, contradicting account of the
same cell.


Every entry is a pair -- a :data:`Cause` and the sentence explaining it -- because
a reader of the matrix needs the class first and the argument second.  "Not
applicable" is one class of several, and the one that matters most is the last:
a cell nobody has written yet must not hide among the cells that cannot exist.
:func:`explain` is the per-pair entry point, :func:`cause_of` the class alone.
"""

from __future__ import annotations

from typing import NamedTuple

from . import marks

#: Why a pair is not exercised, as a closed set of classes.
#:
#: The classes are ordered from "cannot exist" to "must be written": a reader who
#: only looks at the first three sees a sound table, and a reader who wants to
#: know what is still owed looks at :data:`NO_SCRIPT`.
NOT_IN_STANDARD = "not_in_standard"
"""The standard's own table has no such cell, so nothing is owed on it."""

COLUMN_LACKS_CAPABILITY = "column_lacks_capability"
"""The cell needs something the column's structure cannot hold: no stored copy to
round-trip, no capability to select, no second implementation to time against."""

COLUMN_LACKS_OPERAND = "column_lacks_operand"
"""The algorithm has no parameter the cell would need: no broadcast shadow, no
scale factor, no continuous knob, no weight to differentiate."""

NEEDS_DEVICE = "needs_device"
"""The cell needs hardware or a runtime resource a CPU run does not have: streams,
a second card, or a launch whose cost is visible at all."""

COVERED_ELSEWHERE = "covered_elsewhere"
"""The behaviour *is* exercised, by the row and column named in the reason."""

INPUT_DID_NOT_REACH = "input_did_not_reach"
"""The scenario exists and the row was written for it, but this run's sample did
not put the column in it.  Not a gap in the table -- a gap in this seed."""

NO_SCRIPT = "no_script"
"""The cell is due -- the table claims it or could claim it -- and nothing
implements it.  This is the class a table of excuses must not hide: it is the
one that says work is owed, so it is kept apart from every "cannot" above and the
applicability table raises rather than reports it (see :func:`cause_of`)."""

#: Every class, in the order a report should print them.
CAUSES: tuple[str, ...] = (
    NOT_IN_STANDARD,
    COLUMN_LACKS_CAPABILITY,
    COLUMN_LACKS_OPERAND,
    NEEDS_DEVICE,
    COVERED_ELSEWHERE,
    INPUT_DID_NOT_REACH,
    NO_SCRIPT,
)

#: One-line meaning of each class, for a legend under the matrix.
CAUSE_TITLES: dict[str, str] = {
    NOT_IN_STANDARD: "not a cell of the standard's table",
    COLUMN_LACKS_CAPABILITY: "the column cannot carry it",
    COLUMN_LACKS_OPERAND: "the algorithm has no such parameter",
    NEEDS_DEVICE: "needs a device or a runtime resource",
    COVERED_ELSEWHERE: "exercised by another row",
    INPUT_DID_NOT_REACH: "this run's sample did not reach it",
    NO_SCRIPT: "owed: nothing written",
}

#: Class -> symbol for the cell itself.  The applicability symbols in
#: :mod:`neutraltest.gbz72.runner` use ``-`` for every not-exercised cell; these
#: replace it with one glyph per class, and ``N`` is deliberately loud.
CAUSE_SYMBOLS: dict[str, str] = {
    NOT_IN_STANDARD: "·",
    COLUMN_LACKS_CAPABILITY: "-",
    COLUMN_LACKS_OPERAND: "b",
    NEEDS_DEVICE: "d",
    COVERED_ELSEWHERE: "c",
    INPUT_DID_NOT_REACH: "n",
    NO_SCRIPT: "N",
}


class Cause(NamedTuple):
    """A reason class with the sentence that argues it."""

    cause: str
    text: str


def _c(cause: str, text: str) -> Cause:
    """Build a :class:`Cause`, refusing a class the matrix cannot print."""
    if cause not in CAUSES:  # pragma: no cover - a typo in this module
        raise ValueError(f"unknown cause {cause!r}")
    return Cause(cause, text)


#: Reasons for features that apply to some columns but not others.
PARTIAL_REASONS: dict[str, Cause] = {
    "D6": _c(
        COLUMN_LACKS_OPERAND,
        "the kernels of this column broadcast every row, axis and floating type they are "
        "handed, so no operand of theirs is an illegal input to the column itself; what an "
        "add does with a row of the wrong length is a floating-point answer, not a refusal, "
        "and the interface it carries is D1's to hold to account for arity and keyword names",
    ),
    "A2": _c(
        COLUMN_LACKS_CAPABILITY,
        "the row stores a quantised copy and reloads it, which needs a column with "
        "storage a float8 round-trip can bend: only the add, the cache write and the "
        "normalisation have one, while attention, softmax and the rotation compute "
        "their result without keeping an operand",
    ),
    "A4": _c(
        COLUMN_LACKS_CAPABILITY,
        "the row needs an arithmetic path that keeps a subnormal pair's norm, and only "
        "the rotation manipulates a pair element-wise; the reductions flush the pair "
        "through a sum before any check could see it",
    ),
    "A5": _c(
        COLUMN_LACKS_CAPABILITY,
        "the row reads back the exact bit pattern a store produced, and the other "
        "columns only compute a value; the cache write is the one place the bits are "
        "kept",
    ),
    "A6": _c(
        COLUMN_LACKS_CAPABILITY,
        "the row poisons one element and watches where the non-finite value spreads; "
        "the cache stores what it is given rather than propagating it, and the "
        "normalisation's single reduction is already covered by A1",
    ),
    "A7": _c(
        COLUMN_LACKS_CAPABILITY,
        "the row reads the sign of a zero the operator produced on purpose; attention "
        "under zero queries and keys returns the cancellation of signed zeros, which "
        "IEEE-754 fixes at +0, and the residual's gradients are exact copies, so no "
        "zero in either column can carry a sign only the kernel decides. The forward "
        "add and the six columns whose zero survives a multiply, a store or a positive "
        "scale are marked and measured",
    ),
    "A8": _c(
        COLUMN_LACKS_OPERAND,
        "the row measures the order of a long accumulation; only the rotation sums two "
        "products per element, while the other columns either copy or reduce a whole "
        "row, which A1 already compares",
    ),
    "A10": _c(
        COLUMN_LACKS_OPERAND,
        "the row asks what a scale argument multiplies; softmax takes an axis and the "
        "rotation takes an angle, so neither has a scale factor whose application could "
        "be isolated",
    ),
    "A11": _c(
        COLUMN_LACKS_OPERAND,
        "the row chains five training steps and watches the drift; only the "
        "normalisation has a weight those steps update, and the other columns are "
        "stateless between calls",
    ),
    "B6": _c(
        COLUMN_LACKS_OPERAND,
        "the row broadcasts a second operand whose shape differs from the result's; "
        "only the add and the cache have one, since attention's three operands share a "
        "shape and the reductions have a single operand",
    ),
    "B7": _c(
        COLUMN_LACKS_OPERAND,
        "the row hands the column a non-contiguous view; only the normalisation reads a "
        "weight whose layout can differ from the activation's, and the other columns "
        "receive freshly built contiguous operands",
    ),
    "B8": _c(
        COLUMN_LACKS_OPERAND,
        "the row perturbs one sequence position and asks that the others not move; the "
        "cache writes whole rows under an explicit index, which B6 covers, and the "
        "normalisation mixes a whole row by construction",
    ),
    "C3": _c(
        COLUMN_LACKS_OPERAND,
        "the row varies the granularity of a quantisation scale; softmax and the "
        "rotation hold no scale, so there is no granularity to choose",
    ),
    "C4": _c(
        COLUMN_LACKS_OPERAND,
        "the row mixes operand widths; attention and softmax have one value dtype and "
        "the rotation's angle is not a value operand, so no pair of widths can be built",
    ),
    "C5": _c(
        COLUMN_LACKS_OPERAND,
        "the row reads back the output dtype a promotion produced; only the add and the "
        "normalisation have an operand pair whose promotion decides it, and the other "
        "columns return the dtype they were given",
    ),
    "D2": _c(
        COLUMN_LACKS_OPERAND,
        "the row sweeps a continuous numeric parameter; only the add's scale and the "
        "normalisation's epsilon are continuous, while the other columns expose a "
        "discrete choice or none",
    ),
    "D3": _c(
        COVERED_ELSEWHERE,
        "the row checks in-place and alias behaviour on a value operand; the rotation "
        "is the column marked for a view, and B7 covers the normalisation's weight",
    ),
    "D4": _c(
        COLUMN_LACKS_CAPABILITY,
        "the row exercises a structural variant; the cache's structure is its index "
        "list, which B6 covers, and the normalisation has a single shape contract",
    ),
    "E4": _c(
        COLUMN_LACKS_CAPABILITY,
        "the row watches for memory written outside the result; the other columns have "
        "no internal buffer, while attention and the rotation each materialise one "
        "(the mask and the angle table)",
    ),
    "E5": _c(
        COLUMN_LACKS_CAPABILITY,
        "the row bounds an intermediate table; only the rotation materialises angles, "
        "and the other columns work in place",
    ),
    "F3": _c(
        NEEDS_DEVICE,
        "the row times against a second implementation of the same operator, and on a "
        "CPU this framework has only one: the rotation and the normalisation have no "
        "hand-written baseline to compare with, and the remaining families compare an "
        "expression with itself, so the ratio would be 1.00 by construction. The four "
        "forward columns the row does claim are timed where two implementations exist.",
    ),
}

#: Reasons for features the clause does not apply to any of the twelve columns.
ROW_REASONS: dict[str, Cause] = {
    "A12": _c(
        NOT_IN_STANDARD,
        "the differentiable column of the text is the grouped matrix, and part 7.2 has "
        "no such column; the backward columns here are judged by the Jacobian check in "
        "A1 instead",
    ),
    "B1": _c(
        NOT_IN_STANDARD,
        "no column of this clause accepts a tensor above 2**31 elements: the operator "
        "interfaces fix the shapes, so the row has nothing to allocate",
    ),
    "B2": _c(
        COLUMN_LACKS_OPERAND,
        "the only integer operand is the cache's slot list, a row index rather than a "
        "byte offset; B6 exercises its shape and its range",
    ),
    "B3": _c(
        COVERED_ELSEWHERE,
        "a zero-element or single-element call is refused by the reference's shape "
        "checks for every column, and asserting that refusal is D1's contract check "
        "rather than a row of its own",
    ),
    "C2": _c(
        COLUMN_LACKS_CAPABILITY,
        "an MX block needs an E8M0 scale dtype, and no column of this clause stores a "
        "block-scaled format",
    ),
    "D5": _c(
        COLUMN_LACKS_CAPABILITY,
        "each column has a single interface in this framework, so there is no previous "
        "version to compare against",
    ),
    "F4": _c(
        COLUMN_LACKS_CAPABILITY,
        "host-side launch cost belongs to the backend rather than to a column, and F1 "
        "times the whole call including it",
    ),
    "F6": _c(
        NOT_IN_STANDARD,
        "this is the disclosure row itself: the not-applicable bookkeeping is what the "
        "disclosure section prints, so it cannot be a cell",
    ),
    "F7": _c(
        NOT_IN_STANDARD,
        "the part 7.2 applicability table has no column for this row: it was drawn up "
        "before the row existed, so no cell is claimed rather than guessed",
    ),
    "E3": _c(
        NOT_IN_STANDARD,
        "the row runs on the table's columns, and where the backend has no stream API it "
        "runs on threads instead, so it is never waiting for a device: the pair below is "
        "a column this row does not claim rather than one waiting on hardware",
    ),
    "E6": _c(
        NOT_IN_STANDARD,
        "the row asks whether work can arrive in pieces, which a cut of the sample answers "
        "on one device; a backward column is left out because a shard's gradient for a "
        "shared operand is a sum over shards, not a join, and that is A12's question",
    ),
    "E7": _c(
        NOT_IN_STANDARD,
        "the row repeats one deployment's answer on the table's columns, so the pair below "
        "is a column this row does not claim rather than one waiting on hardware",
    ),
}


def _recorded(feature_id: str, column: str) -> Cause:
    """The table's entry for ``feature_id``, or the loud KeyError."""
    try:
        return PARTIAL_REASONS[feature_id]
    except KeyError:
        pass
    try:
        return ROW_REASONS[feature_id]
    except KeyError as exc:  # pragma: no cover - the two tables cover the catalogue
        raise KeyError(
            f"no recorded reason for {feature_id} on {column}; the applicability table "
            "and the reason table disagree"
        ) from exc


def reason_for(feature_id: str, column: str) -> str:
    """The recorded reason ``feature_id`` is not exercised on ``column``."""
    return _recorded(feature_id, column).text


def cause_of(feature_id: str, column: str) -> str:
    """The :data:`Cause` recorded for the pair.

    A pair the table marks *is* exercised, so it has no recorded cause: if a
    marked pair lands here it means no script claims it, which is exactly
    :data:`NO_SCRIPT`, and the conclusion is drawn rather than copied from the
    table.  An unmarked pair answers with its own class, which is never
    :data:`NO_SCRIPT` -- an unmarked pair is a decision, not an omission.
    """
    if marks.is_marked(feature_id, column):
        return NO_SCRIPT
    return _recorded(feature_id, column).cause


def symbol_of(feature_id: str, column: str) -> str:
    """One glyph for the cell: the class's symbol, see :data:`CAUSE_SYMBOLS`."""
    return CAUSE_SYMBOLS[cause_of(feature_id, column)]


def explain(feature_id: str, column: str) -> tuple[str, str]:
    """``(cause, reason)`` for one pair -- the whole answer about one cell.

    The pair is marked and exercised: the callers that render a cell handle that
    before asking; here a marked pair would answer :data:`NO_SCRIPT`, because
    from this module's point of view a claimed cell with nothing recorded is
    work owed.
    """
    return cause_of(feature_id, column), reason_for(feature_id, column)


def tally(
    pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...],
) -> dict[str, list[tuple[str, str]]]:
    """Group ``(feature, column)`` pairs by class, in :data:`CAUSES` order.

    The groups keep the input order inside a class, so a table can print them in
    catalogue order without sorting again.
    """
    grouped: dict[str, list[tuple[str, str]]] = {cause: [] for cause in CAUSES}
    for feature_id, column in pairs:
        grouped[cause_of(feature_id, column)].append((feature_id, column))
    return grouped
