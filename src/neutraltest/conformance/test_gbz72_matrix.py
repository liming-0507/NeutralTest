"""The part 7.2 feature matrix: one script per row, one cell per column.

The rows come from the same 45-feature catalogue the 7.3 matrix uses; the twelve
columns come from the part 7.2 table. Which cells exist is decided by the
applicability marks, not by this module: a pair the table does not mark is
skipped with the reason recorded for its row, and a pair it does mark is run.
"""

from __future__ import annotations

import pytest

from neutraltest import clause
from neutraltest.gbz72 import runner
from neutraltest.gbz72.marks import TABLE_ORDER
from neutraltest.gbz73.features import FEATURES
from neutraltest.gbz73.inference.columns import BY_NAME

if not clause.covers(clause.get_clause(), "72"):
    pytest.skip(f"clause {clause.get_clause()} does not cover part 7.2", allow_module_level=True)

COMBINATIONS = [(feature, BY_NAME[name]) for feature in FEATURES for name in TABLE_ORDER]
IDS = [f"{feature.id}-{column.name}" for feature, column in COMBINATIONS]


@pytest.mark.parametrize(("feature", "column"), COMBINATIONS, ids=IDS)
def test_gbz72(feature, column, nt) -> None:
    result = runner.run_column(feature, column, nt.backend, seed=nt.policy.seed)
    if result.outcome == "na":
        pytest.skip(f"{feature.id}/{column.name}: {result.detail}")
    if result.outcome == "known":
        # The cell ran and failed in a way the registry explains: still visible,
        # still counted, and an unexpected pass shows up as an xpass.
        pytest.xfail(result.detail)
    if result.outcome != "passed":
        pytest.fail(f"{feature.id}/{column.name}: {result.detail}", pytrace=False)
