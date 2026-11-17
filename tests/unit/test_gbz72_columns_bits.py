"""The coverage bits on a column say the same thing as the table they came from.

``Column.bits`` is the standard's coverage table read down a column; the
applicability table is the same coverage read across a row. A table that
disagrees with its own rows is worse than no table, so both directions are
asserted here: column bit *i* is row *i*'s bit for that column, and row bit *j*
is that column's bit for row *j*.
"""

from __future__ import annotations

import pytest

from neutraltest.gbz72 import marks
from neutraltest.gbz73.features import FEATURES
from neutraltest.gbz73.inference.columns import BY_NAME, COLUMN_SPECS, FEATURE_ORDER

FEATURE_IDS = tuple(feature.id for feature in FEATURES)


def test_the_catalogue_the_bits_are_written_in_is_the_catalogue_that_is_tested() -> None:
    assert FEATURE_ORDER == FEATURE_IDS
    assert tuple(feature.id for feature in FEATURES) == tuple(marks.APPLICABILITY)


def test_every_column_carries_one_bit_per_feature() -> None:
    for spec in COLUMN_SPECS:
        assert len(spec.bits) == len(FEATURE_IDS), spec.name
        assert set(spec.bits) <= {"0", "1"}, spec.name


def test_the_columns_cover_the_twelve_of_the_standard() -> None:
    """Same twelve columns, as a set.

    ``COLUMN_SPECS`` lists each operator forward then backward; ``TABLE_ORDER``
    prints all the forwards and then all the backwards. Both are the standard's
    twelve columns, and the bits are indexed by feature, not by that order, so
    the set is what has to hold.
    """
    assert {spec.name for spec in COLUMN_SPECS} == set(marks.TABLE_ORDER)
    assert len(COLUMN_SPECS) == len(marks.TABLE_ORDER) == 12


@pytest.mark.parametrize("column", marks.TABLE_ORDER)
def test_a_column_runs_every_cell_the_clause_asks_for(column: str) -> None:
    """A cell the clause asks for is never left unrun.

    This is deliberately one-way. The bit says *the clause asks for this cell*;
    the applicability table says *the harness measures this cell*. A clause that
    asks for something is not optional, so the bit implies the mark -- but the
    converse does not hold, and must not: a required feature is measured on
    every column whether or not the clause names it, and forcing those extra
    runs back into the bits would report coverage the standard never granted.
    """
    spec = BY_NAME[column]
    missing = [
        feature_id
        for feature_id in FEATURE_IDS
        if spec.standard_covers(feature_id) and not marks.is_marked(feature_id, column)
    ]
    assert not missing, (
        f"{column}: the clause asks for {missing}, but the harness does not run them"
    )


@pytest.mark.parametrize("feature", FEATURES, ids=lambda feature: feature.id)
def test_a_row_runs_every_cell_the_clause_asks_for(feature) -> None:
    """The same guard read across a row."""
    unrun = [
        column
        for column in marks.TABLE_ORDER
        if BY_NAME[column].standard_covers(feature.id) and not marks.is_marked(feature.id, column)
    ]
    assert not unrun, (
        f"{feature.id}: the clause asks for {unrun}, but the harness does not run them"
    )


def test_the_two_transcriptions_of_the_table_agree() -> None:
    """``_BITS`` (read down) and ``_STANDARD_TABLE`` (read across) are one text.

    Both are hand copies of the standard's table, and this is what keeps them
    from drifting apart. It is unrelated to what the harness runs.
    """
    from neutraltest.gbz73.inference.columns import _STANDARD_TABLE

    rows = list(_STANDARD_TABLE.values())
    assert len(rows) == len(FEATURE_IDS)
    for position, feature_id in enumerate(FEATURE_IDS):
        assert rows[position] == _STANDARD_TABLE[feature_id]
    for column in marks.TABLE_ORDER:
        index = marks.TABLE_ORDER.index(column)
        read_down = "".join(row[index] if len(row) > index else "0" for row in rows)
        assert BY_NAME[column].bits == read_down, column


def test_the_column_answers_for_itself() -> None:
    spec = BY_NAME["add_fwd"]
    assert spec.standard_covers("A1")
    assert not spec.standard_covers("A4")
    assert spec.bits[FEATURE_ORDER.index("A1")] == "1"
    assert spec.bits[FEATURE_ORDER.index("A4")] == "0"
    assert spec.covered_by_standard == spec.bits
    with pytest.raises(ValueError):
        spec.standard_covers("Z9")


def test_a_bit_string_of_the_wrong_length_or_alphabet_is_refused() -> None:
    with pytest.raises(ValueError, match="coverage bits"):
        type(BY_NAME["add_fwd"])(
            name="x",
            family="add",
            description="d",
            quadratic=False,
            freezes=(),
            notes="",
            bits="101",
        )
    with pytest.raises(ValueError, match="0 or 1"):
        type(BY_NAME["add_fwd"])(
            name="x",
            family="add",
            description="d",
            quadratic=False,
            freezes=(),
            notes="",
            bits="2" * len(FEATURE_IDS),
        )


def test_the_number_of_stated_cells_is_how_many_the_table_runs() -> None:
    """The standard's cells are a subset of the ones the harness measures.

    ``MARKED`` may exceed the stated count: a required feature is measured on
    every column, including columns the clause never mentions. The guard is that
    it can never be *fewer* -- that would mean the clause asked for a cell the
    harness skipped.
    """
    stated = sum(spec.bits.count("1") for spec in COLUMN_SPECS)
    marked = len(marks.marked_pairs())
    assert stated <= marked, (
        f"the clause asks for {stated} cells but the harness only runs {marked}"
    )
    assert marked == marks.MARKED
