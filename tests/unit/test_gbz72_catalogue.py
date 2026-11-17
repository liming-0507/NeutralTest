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
    "add_fwd": 35,
    "flash_attention_fwd": 32,
    "softmax_fwd": 33,
    "rope_fwd": 35,
    "kv_cache_fwd": 35,
    "rms_norm_fwd": 34,
    "add_bwd": 30,
    "flash_attention_bwd": 30,
    "softmax_bwd": 30,
    "rope_bwd": 30,
    "kv_cache_bwd": 31,
    "rms_norm_bwd": 31,
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


#: Required rows that are not measured per operator-column, and why.
#: F6 asks for the not-applicable bookkeeping the disclosure section prints --
#: it is a statement about the report, not a behaviour of any one operator.
#: F7 postdates the twelve-column table and is therefore reported in the
#: "measured outside the table" section instead of being attributed to columns
#: the clause never drew up for it.
NOT_PER_COLUMN: dict[str, str] = {
    "F6": "the disclosure row itself: the report prints it, no operator owns it",
    "F7": "postdates the twelve-column table; reported outside it",
}


def test_every_required_row_runs_every_column() -> None:
    """A row the analysis calls required is measured on all twelve columns.

    The clause does not ask for every one of those cells -- the required rows
    are required of the *implementation*, and a vendor cannot claim to support
    one of these operators while leaving, say, its gradient unmeasured just
    because the clause named that feature under the training operators. The two
    exceptions are structural and named above; anything else is a hole.
    """
    hollow = {
        feature.id: sorted(set(marks.TABLE_ORDER) - set(marks.columns_of(feature.id)))
        for feature in FEATURES
        if feature.tier == "required"
        and feature.id not in NOT_PER_COLUMN
        and len(marks.columns_of(feature.id)) != len(marks.TABLE_ORDER)
    }
    assert not hollow, f"required rows without a full set of columns: {hollow}"


def test_the_rows_that_claim_no_column_are_the_ones_with_a_reason_to() -> None:
    """A row with no column at all has to justify why, and none may be required.

    F6 is the only required row that stays empty, and it is empty for a reason
    of a different kind: it is the disclosure row itself -- what it asks for is
    the not-applicable bookkeeping printed by the disclosure section, which
    cannot be attributed to one operator-column. Every other empty row is
    conditional or optional.
    """
    empty = {feature.id for feature in FEATURES if not marks.columns_of(feature.id)}
    assert empty == {"B1", "B2", "C2", "D5", "F4", "F6", "F7"}, sorted(empty)
    tiers = {feature.id: feature.tier for feature in FEATURES}
    # The required rows that run no column are exactly the ones named above, so
    # the exception list and the catalogue cannot drift apart.
    assert {fid for fid in empty if tiers[fid] == "required"} == set(NOT_PER_COLUMN)
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
