"""The FP8 clause matrix: one script per row, one cell per operator.

The rows and the operators come from the catalogue in ``neutraltest.gbz73``, so
this module holds no case list of its own -- add a row there, and it is
collected here.
"""

from __future__ import annotations

import pytest

from neutraltest.gbz73 import runner
from neutraltest.gbz73.features import FEATURES
from neutraltest.gbz73.ops import OPS

COMBINATIONS = [(feature, op) for op in OPS for feature in FEATURES]
IDS = [f"{feature.id}-{op.name}" for feature, op in COMBINATIONS]


@pytest.mark.parametrize(("feature", "op"), COMBINATIONS, ids=IDS)
def test_gbz73(feature, op, nt) -> None:
    result = runner.run_feature(feature, op, nt.backend, seed=nt.policy.seed)
    if result.outcome == "na":
        pytest.skip(f"{feature.id}/{op.name}: {result.detail}")
    if result.outcome != "passed":
        pytest.fail(f"{feature.id}/{op.name}: {result.detail}", pytrace=False)
