"""F6 -- the disclosure row.

The point of the matrix is that every cell is either checked or explained. This
row enforces the explaining half: an item cannot declare itself not applicable
without saying why, and every row of the standard has a script behind it.
"""

from __future__ import annotations

from .. import runner
from ..context import RunContext
from ..features import FEATURES
from ..items import module_for

ID = "F6"
OPS = None


def run(ctx: RunContext) -> None:
    # Every row must have an importable script that agrees about its own id.
    for feature in FEATURES:
        module = module_for(feature.id)
        if feature.id != module.ID:
            raise AssertionError(f"items/{feature.id.lower()}.py declares ID {module.ID}")

    # A not-applicable cell without a reason is a hole, not a disclosure.
    for feature_id in ("D5", "E5"):
        result = runner.run_feature(FEATURES_BY_ID(feature_id), ctx.op, ctx.backend, seed=ctx.seed)
        if result.outcome != "na":
            raise AssertionError(
                f"{feature_id} unexpectedly ran instead of declaring itself not applicable"
            )
        if not result.detail.strip():
            raise AssertionError(f"{feature_id} declared itself not applicable without a reason")

    ctx.note(f"{len(FEATURES)} rows have scripts; not-applicable cells carry a reason")


def FEATURES_BY_ID(feature_id: str):
    from ..features import BY_ID

    return BY_ID[feature_id]
