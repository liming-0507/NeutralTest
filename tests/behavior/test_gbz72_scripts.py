"""The part 7.2 scripts run, and they fail when the arithmetic is wrong."""

from __future__ import annotations

import pytest

from neutraltest.backends import CpuBackend
from neutraltest.gbz72 import items, marks, runner
from neutraltest.gbz73.features import FEATURES
from neutraltest.gbz73.inference.columns import COLUMN_SPECS

SLICE = ("A1", "A4", "B5", "C3", "D3", "E4", "F3")


def _feature(feature_id: str):
    return next(feature for feature in FEATURES if feature.id == feature_id)


def _columns(feature_id: str):
    marked = set(marks.columns_of(feature_id))
    return [column for column in COLUMN_SPECS if column.name in marked]


@pytest.mark.parametrize("feature_id", SLICE)
def test_a_representative_script_passes_every_column_it_claims(feature_id: str) -> None:
    backend = CpuBackend()
    results = [
        runner.run_column(_feature(feature_id), column, backend, seed=7)
        for column in _columns(feature_id)
    ]
    assert results, f"{feature_id} claims no column"
    failures = [f"{result.column}: {result.detail}" for result in results if not result.ok]
    assert not failures, failures


def test_the_forward_script_rejects_a_perturbed_kernel() -> None:
    module = items.module_for("A1")
    column = next(column for column in COLUMN_SPECS if column.name == "add_fwd")
    context = _context(column)
    module.run(context)
    real = context.kernel()

    def wrong(*args, **kwargs):
        out = real(*args, **kwargs)
        return out + 1e-3

    context.kernel = lambda: wrong  # type: ignore[method-assign]
    with pytest.raises(AssertionError):
        module.run(context)


def _context(column):
    from neutraltest.gbz73.inference.context import InferenceContext

    return InferenceContext(
        column=column, backend=CpuBackend(), feature_id="A1", seed=7, variant=""
    )


def test_every_row_that_claims_a_column_has_a_script() -> None:
    """The rows the applicability table exercises are exactly the rows with scripts.

    The two sides are easy to drift apart: a row can claim columns with no
    script behind it, and the run only reports that when someone happens to
    parametrise it; a script can be added for a row no column ever reaches, and
    nothing runs it at all.  ``available_features`` once globbed for
    ``[a-f][0-9].py`` -- a single digit -- so A10 and A11 had scripts that no
    listing could see, and nothing failed because nothing asked.
    """
    listed = set(items.available_features())
    claimed = {fid for fid, columns in marks.APPLICABILITY.items() if columns}
    assert not claimed - listed, (
        f"rows claiming columns but with no reachable script: {claimed - listed}"
    )
    assert not listed - claimed, f"scripts for rows no column reaches: {listed - claimed}"


def test_the_listing_is_in_catalogue_order_not_string_order() -> None:
    """``A10`` follows ``A9``; a plain sort would put it after ``A1``."""
    listed = items.available_features()
    numbers = [(fid[0], int(fid[1:])) for fid in listed]
    assert numbers == sorted(numbers), listed
