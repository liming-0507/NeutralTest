"""The reason classes behind the part 7.2 matrix: every non-exercised cell says why."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from neutraltest.gbz72 import inventory, marks, reasons, runner
from neutraltest.gbz73.features import FEATURES
from neutraltest.gbz73.inference.columns import BY_NAME

FEATURE_IDS = tuple(feature.id for feature in FEATURES)


def run_cli(*arguments: str, timeout: int = 300) -> subprocess.CompletedProcess:
    # The test tree is not an importable package, so the helper is local rather
    # than imported from the sibling module that already has one.
    src = Path(__import__("neutraltest").__file__).resolve().parent.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("NEUTRALTEST_BACKEND", None)
    return subprocess.run(
        [sys.executable, "-m", "neutraltest", *arguments],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def test_every_class_prints_distinctly_and_in_the_legend() -> None:
    assert set(reasons.CAUSE_TITLES) == set(reasons.CAUSES)
    assert set(reasons.CAUSE_SYMBOLS) == set(reasons.CAUSES)
    assert len(set(reasons.CAUSE_SYMBOLS.values())) == len(reasons.CAUSES)
    # The classes the report prints instead of "-" must not collide with P/F/K.
    assert not set(reasons.CAUSE_SYMBOLS.values()) & set("PFK")


def test_every_unmarked_cell_has_a_class_and_a_sentence() -> None:
    for feature_id in FEATURE_IDS:
        for column in marks.TABLE_ORDER:
            if marks.is_marked(feature_id, column):
                continue
            cause, text = inventory.explain(feature_id, column)
            assert cause in reasons.CAUSES, f"{feature_id}/{column}: {cause!r}"
            assert len(text) > 20, f"{feature_id}/{column}: {text!r} is not a sentence"
            # An unmarked cell is a decision, never an omission.
            assert cause != reasons.NO_SCRIPT


def test_a_marked_cell_has_no_class_until_it_is_missing_a_script() -> None:
    assert inventory.explain("A3", "add_fwd") == (None, "")
    assert inventory.symbol("A3", "add_fwd") == ""
    assert inventory.explain("A3", "add_bwd") == (None, "")
    assert inventory.cause("A2", "add_fwd") is None


def test_the_script_finder_claims_every_class_it_can_still_stand_for() -> None:
    # NO_SCRIPT is a conclusion, not a table entry: it is what answers a pair the
    # table claims and no script covers. The table itself must not use it to
    # excuse an unmarked cell -- that is the laundering this class exists to stop.
    recorded = [entry.cause for entry in reasons.PARTIAL_REASONS.values()]
    recorded += [entry.cause for entry in reasons.ROW_REASONS.values()]
    assert reasons.NO_SCRIPT not in recorded
    assert reasons.COLUMN_LACKS_CAPABILITY in recorded
    # NEEDS_DEVICE stays a legal class but no longer answers any cell: the rows
    # that used to wait for hardware now run on one device, and the one timing
    # row that cannot is timing against a second implementation, not a device.
    assert reasons.COVERED_ELSEWHERE in recorded
    assert reasons.NOT_IN_STANDARD in recorded


def test_the_catalogue_marks_exactly_the_pairs_the_table_marks() -> None:
    data = inventory.catalogue()
    assert data["marked"] == marks.MARKED
    assert set(data["legend"]) == set(reasons.CAUSES)
    assert data["columns"] == list(marks.TABLE_ORDER)
    assert len(data["cells"]) == len(FEATURE_IDS) * len(marks.TABLE_ORDER)
    exercised = [cell for cell in data["cells"] if cell["cause"] is None]
    assert len(exercised) == marks.MARKED
    assert all(cell["marked"] for cell in exercised)
    assert all(cell["reason"] for cell in data["cells"] if cell["cause"] is not None)


def test_a_not_applicable_cell_reported_by_the_runner_names_the_same_class() -> None:
    column = BY_NAME["softmax_fwd"]
    feature = next(feature for feature in FEATURES if feature.id == "A5")
    result = runner.run_column(feature, column, None)  # type: ignore[arg-type]
    assert result.outcome == "na"
    cause, text = inventory.explain("A5", column.name)
    assert cause == reasons.COLUMN_LACKS_CAPABILITY
    assert result.detail == text


def test_the_catalogue_command_emits_the_same_data() -> None:
    result = run_cli("catalogue", "--compact")
    assert result.returncode == 0, result.stderr
    printed = json.loads(result.stdout)
    expected = inventory.catalogue()
    assert printed["marked"] == expected["marked"]
    assert printed["legend"] == expected["legend"]
    assert printed["cells"] == expected["cells"]


@pytest.mark.parametrize(
    ("feature_id", "expected"),
    [
        # A2 and F3 are now run on every column of the table, so no cell of
        # theirs is left to a reason class at all; the classes they used to
        # stand for are checked through the rows below, which still have one.
        ("A2", None),
        ("F3", None),
        ("B6", reasons.COLUMN_LACKS_OPERAND),
        # D6 is run everywhere too: what used to be disclosed as a column its
        # kernels cannot refuse anything on is now reported by the row itself.
        ("D6", None),
        ("F7", reasons.NOT_IN_STANDARD),
    ],
)
def test_the_sample_classes_are_the_ones_the_analysis_recorded(
    feature_id: str, expected: str | None
) -> None:
    """Each row's class, read from a cell the row does not run.

    ``None`` means the row now runs every column, so there is no such cell and
    nothing to classify -- which is the stronger statement, not a waiver.
    """
    exercised = set(marks.columns_of(feature_id))
    unrun = [name for name in marks.TABLE_ORDER if name not in exercised]
    if expected is None:
        assert not unrun, f"{feature_id} still leaves {unrun} unrun"
        return
    assert unrun, f"{feature_id} runs every column, so it has no class listed"
    assert {inventory.cause(feature_id, name) for name in unrun} == {expected}
