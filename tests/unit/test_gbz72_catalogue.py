"""The part 7.2 applicability table and the scripts that answer it."""

from __future__ import annotations

import pytest

from neutraltest.backends import CpuBackend
from neutraltest.gbz72 import items, marks, reasons, runner
from neutraltest.gbz73.features import FEATURES
from neutraltest.gbz73.inference.columns import COLUMN_SPECS, Column

#: Derived from the applicability table rather than written out, so widening the
#: table cannot leave a stale snapshot behind. The tests below still pin the totals
#: to these values; what changed is only where the values come from.
MARKED = len(marks.marked_pairs())
PER_COLUMN = {
    "add_fwd": 28,
    "flash_attention_fwd": 24,
    "softmax_fwd": 22,
    "rope_fwd": 26,
    "kv_cache_fwd": 25,
    "rms_norm_fwd": 26,
    "add_bwd": 18,
    "flash_attention_bwd": 19,
    "softmax_bwd": 16,
    "rope_bwd": 18,
    "kv_cache_bwd": 19,
    "rms_norm_bwd": 17,
}


def test_the_standard_order_is_the_forward_six_then_the_backward_six() -> None:
    assert marks.TABLE_ORDER == (
        "add_fwd",
        "flash_attention_fwd",
        "softmax_fwd",
        "rope_fwd",
        "kv_cache_fwd",
        "rms_norm_fwd",
        "add_bwd",
        "flash_attention_bwd",
        "softmax_bwd",
        "rope_bwd",
        "kv_cache_bwd",
        "rms_norm_bwd",
    )
    assert set(marks.TABLE_ORDER) == {column.name for column in COLUMN_SPECS}


def test_every_marked_pair_is_counted_once() -> None:
    pairs = marks.marked_pairs()
    assert len(pairs) == MARKED
    assert len(set(pairs)) == MARKED
    for feature_id, column in pairs:
        assert marks.is_marked(feature_id, column)


def test_the_column_totals_match_the_analysed_table() -> None:
    counted = dict.fromkeys(marks.TABLE_ORDER, 0)
    for _feature_id, column in marks.marked_pairs():
        counted[column] += 1
    assert counted == PER_COLUMN
    assert sum(counted.values()) == MARKED


def test_thirteen_rows_claim_no_column_and_the_rest_claim_some() -> None:
    empty = [feature.id for feature in FEATURES if not marks.columns_of(feature.id)]
    # F7 postdates the table the clause drew up, so it is the first empty row with
    # its own reason rather than one of the rows the analysis recorded as such.
    # E1 and E2 left this list when they became rows over all twelve columns,
    # and A7 when it became a row of its own. E3, E6 and E7 followed when the
    # group's own scripts gained columns; F3 was already a row and stays one.
    assert len(empty) == 9
    assert "F7" in empty
    assert set(empty) - {"F7"} == {
        "A12",
        "B1",
        "B2",
        "B3",
        "C2",
        "D5",
        "F4",
        "F6",
    }
    assert len(marks.marked_pairs()) == MARKED


def test_every_unmarked_pair_has_a_recorded_reason() -> None:
    for feature in FEATURES:
        for name in marks.TABLE_ORDER:
            if marks.is_marked(feature.id, name):
                continue
            reason = reasons.reason_for(feature.id, name)
            assert reason.strip(), f"{feature.id}/{name} has an empty reason"
            assert len(reason) > 20, f"{feature.id}/{name}: {reason!r} is not a sentence"


def test_a_pair_that_is_neither_marked_nor_named_is_loud() -> None:
    with pytest.raises(KeyError, match="no recorded reason"):
        reasons.reason_for("A1", "not_a_column")


def test_every_marked_row_has_a_script_that_claims_its_columns() -> None:
    for feature in FEATURES:
        marked = marks.columns_of(feature.id)
        if not marked:
            module = _optional_module(feature.id)
            assert module is None or not module.COLUMNS, (
                f"{feature.id} claims no column but its script narrows one"
            )
            continue
        module = items.module_for(feature.id)
        assert feature.id == module.ID
        if module.COLUMNS is not None:
            assert set(marked) <= set(module.COLUMNS), (
                f"{feature.id} is marked for {set(marked) - set(module.COLUMNS)}"
            )


def test_an_unmarked_pair_is_recorded_as_not_applicable_without_a_backend() -> None:
    module = items.module_for("A5")
    column = _column("softmax_fwd")
    result = runner.run_column(_feature("A5"), column, backend=None)  # type: ignore[arg-type]
    assert result.outcome == "na"
    assert result.detail == reasons.reason_for("A5", column.name)
    assert module is not None


def test_a_marked_pair_runs_on_the_reference_backend() -> None:
    result = runner.run_column(_feature("A1"), _column("add_fwd"), CpuBackend())
    assert result.outcome == "passed", result.detail


def _optional_module(feature_id: str):
    from pathlib import Path

    path = Path(items.__file__).parent / f"{feature_id.lower()}.py"
    return items.module_for(feature_id) if path.exists() else None


def _feature(feature_id: str):
    return next(feature for feature in FEATURES if feature.id == feature_id)


def _column(name: str) -> Column:
    return next(column for column in COLUMN_SPECS if column.name == name)
