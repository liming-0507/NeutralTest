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
def test_a_column_reads_the_table_down(column: str) -> None:
    """Column bit *i* is the i-th row's bit for this column."""
    spec = BY_NAME[column]
    index = marks.TABLE_ORDER.index(column)
    assert spec.bits == "".join(
        "1" if marks.is_marked(feature_id, column) else "0" for feature_id in FEATURE_IDS
    ), column
    # And bit by bit, so a transposition error names the feature that is wrong.
    for position, feature_id in enumerate(FEATURE_IDS):
        assert spec.bits[position] == ("1" if marks.is_marked(feature_id, column) else "0"), (
            column,
            feature_id,
            position,
            index,
        )


@pytest.mark.parametrize("feature", FEATURES, ids=lambda feature: feature.id)
def test_a_row_reads_the_columns_down(feature) -> None:
    """The other half of the same assertion: row bit *j* is column *j*'s bit."""
    stated = "".join(
        "1" if BY_NAME[column].standard_covers(feature.id) else "0" for column in marks.TABLE_ORDER
    )
    assert stated == "".join(
        "1" if marks.is_marked(feature.id, column) else "0" for column in marks.TABLE_ORDER
    ), feature.id


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


def test_the_number_of_stated_cells_is_the_number_the_table_marks() -> None:
    stated = sum(spec.bits.count("1") for spec in COLUMN_SPECS)
    assert stated == len(marks.marked_pairs())
    assert stated == marks.MARKED
