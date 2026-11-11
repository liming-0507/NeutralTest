"""Why a (feature, column) pair of part 7.2 is not exercised.

The applicability table in :mod:`neutraltest.gbz72.marks` says *which* pairs the
clause covers.  This module says why the others are left out, one recorded
reason per feature: the row is disclosed as not applicable on a column rather
than silently dropped, exactly as the part 7.3 side discloses its own
not-applicable cells.

A reason is written once per feature, not per pair, because a row fails to apply
for one structural reason across all the columns it misses.  Fifteen features
are not applied to any of the twelve columns; for those the same text is the
reason for the whole row and is what :func:`reason_for` returns for every column.
"""

from __future__ import annotations

#: Reasons for features that apply to some columns but not others.
PARTIAL_REASONS: dict[str, str] = {
    "A2": (
        "the row stores a quantised copy and reloads it, which needs a column with "
        "storage a float8 round-trip can bend: only the add, the cache write and the "
        "normalisation have one, while attention, softmax and the rotation compute "
        "their result without keeping an operand"
    ),
    "A3": (
        "the row drives an operand to the format's overflow boundary; attention and "
        "the normalisation rescale or reduce it away before the result is formed, and "
        "the add has no stored value to saturate"
    ),
    "A4": (
        "the row needs an arithmetic path that keeps a subnormal pair's norm, and only "
        "the rotation manipulates a pair element-wise; the reductions flush the pair "
        "through a sum before any check could see it"
    ),
    "A5": (
        "the row reads back the exact bit pattern a store produced, and the other "
        "columns only compute a value; the cache write is the one place the bits are "
        "kept"
    ),
    "A6": (
        "the row poisons one element and watches where the non-finite value spreads; "
        "the cache stores what it is given rather than propagating it, and the "
        "normalisation's single reduction is already covered by A1"
    ),
    "A8": (
        "the row measures the order of a long accumulation; only the rotation sums two "
        "products per element, while the other columns either copy or reduce a whole "
        "row, which A1 already compares"
    ),
    "A10": (
        "the row asks what a scale argument multiplies; softmax takes an axis and the "
        "rotation takes an angle, so neither has a scale factor whose application could "
        "be isolated"
    ),
    "A11": (
        "the row chains five training steps and watches the drift; only the "
        "normalisation has a weight those steps update, and the other columns are "
        "stateless between calls"
    ),
    "B6": (
        "the row broadcasts a second operand whose shape differs from the result's; "
        "only the add and the cache have one, since attention's three operands share a "
        "shape and the reductions have a single operand"
    ),
    "B7": (
        "the row hands the column a non-contiguous view; only the normalisation reads a "
        "weight whose layout can differ from the activation's, and the other columns "
        "receive freshly built contiguous operands"
    ),
    "B8": (
        "the row perturbs one sequence position and asks that the others not move; the "
        "cache writes whole rows under an explicit index, which B6 covers, and the "
        "normalisation mixes a whole row by construction"
    ),
    "C3": (
        "the row varies the granularity of a quantisation scale; softmax and the "
        "rotation hold no scale, so there is no granularity to choose"
    ),
    "C4": (
        "the row mixes operand widths; attention and softmax have one value dtype and "
        "the rotation's angle is not a value operand, so no pair of widths can be built"
    ),
    "C5": (
        "the row reads back the output dtype a promotion produced; only the add and the "
        "normalisation have an operand pair whose promotion decides it, and the other "
        "columns return the dtype they were given"
    ),
    "D2": (
        "the row sweeps a continuous numeric parameter; only the add's scale and the "
        "normalisation's epsilon are continuous, while the other columns expose a "
        "discrete choice or none"
    ),
    "D3": (
        "the row checks in-place and alias behaviour on a value operand; the rotation "
        "is the column marked for a view, and B7 covers the normalisation's weight"
    ),
    "D4": (
        "the row exercises a structural variant; the cache's structure is its index "
        "list, which B6 covers, and the normalisation has a single shape contract"
    ),
    "E4": (
        "the row watches for memory written outside the result; the other columns have "
        "no internal buffer, while attention and the rotation each materialise one "
        "(the mask and the angle table)"
    ),
    "E5": (
        "the row bounds an intermediate table; only the rotation materialises angles, "
        "and the other columns work in place"
    ),
    "F3": (
        "the row times against a second implementation of the same operator; the "
        "rotation and the normalisation have no hand-written baseline in this framework "
        "to compare with"
    ),
}

#: Reasons for features the clause does not apply to any of the twelve columns.
ROW_REASONS: dict[str, str] = {
    "A7": (
        "the twelve columns are float32 kernels over IEEE values and none of them "
        "exposes a signed-zero contract beyond the bit-for-bit comparison A1 already "
        "performs on its outputs"
    ),
    "A12": (
        "the differentiable column of the text is the grouped matrix, and part 7.2 has "
        "no such column; the backward columns here are judged by the Jacobian check in "
        "A1 instead"
    ),
    "B1": (
        "no column of this clause accepts a tensor above 2**31 elements: the operator "
        "interfaces fix the shapes, so the row has nothing to allocate"
    ),
    "B2": (
        "the only integer operand is the cache's slot list, a row index rather than a "
        "byte offset; B6 exercises its shape and its range"
    ),
    "B3": (
        "a zero-element or single-element call is refused by the reference's shape "
        "checks for every column, and asserting that refusal is D1's contract check "
        "rather than a row of its own"
    ),
    "C2": (
        "an MX block needs an E8M0 scale dtype, and no column of this clause stores a "
        "block-scaled format"
    ),
    "D5": (
        "each column has a single interface in this framework, so there is no previous "
        "version to compare against"
    ),
    "D6": (
        "the illegal-input contract is exercised by D1 (arity and keyword names) and B6 "
        "(a shape the operator refuses); a separate row would repeat them"
    ),
    "E1": (
        "every column is deterministic by construction -- no atomics, no unordered "
        "reduction -- so the row would re-run A1 on the same inputs"
    ),
    "E2": ("no column draws a random number; the harness seeds the inputs and the cotangents"),
    "E3": (
        "the columns own no streams or concurrency; the backend's scheduling is covered "
        "by the backend tests rather than by a column-level row"
    ),
    "E6": (
        "a single device is the only configuration this framework runs, so there is no "
        "second card on which to place a column"
    ),
    "E7": (
        "the environment row belongs to the deployment, not to a column: the backend "
        "tests assert the versions and the driver"
    ),
    "F4": (
        "host-side launch cost belongs to the backend rather than to a column, and F1 "
        "times the whole call including it"
    ),
    "F6": (
        "this is the disclosure row itself: the not-applicable bookkeeping is what the "
        "disclosure section prints, so it cannot be a cell"
    ),
    "F7": (
        "the part 7.2 applicability table has no column for this row: it was drawn up "
        "before the row existed, so no cell is claimed rather than guessed"
    ),
}


def reason_for(feature_id: str, column: str) -> str:
    """The recorded reason ``feature_id`` is not exercised on ``column``."""
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
