"""The SDK band is reported, never assumed, and never silently passes a run.

These cells are about the *declaration* rather than a numeric property of an
operator, so they are satisfied by the process state the plugin set up: this
keeps "did the environment match the release we certified" in the same report
as the cells it qualifies.
"""

from __future__ import annotations

import pytest

from neutraltest import sdk

pytestmark = [pytest.mark.nt_family("environment"), pytest.mark.nt_op("sdk")]


@pytest.mark.nt_backend("cpu")
def test_the_running_torch_release_is_in_the_support_table() -> None:
    """The bundled table must at least contain the release it was built for.

    The window returned by :func:`sdk.torch_window` lists *python* versions; the
    check is that the running interpreter is inside it, which is the same
    question the default profile asks.
    """
    release = sdk.pytorch_release()
    window, exact = sdk.torch_window(release)
    if not exact:
        pytest.skip(f"torch {release} is newer than the bundled support table")
    assert sdk.interval(window, sdk.detect().python) == "ok"


@pytest.mark.nt_backend("cpu")
def test_the_interpreter_is_judged_against_a_published_window() -> None:
    status = sdk.measure()
    if status.band == "unknown":
        # An unknown band is a disclosure, not a defect: nothing published a
        # window for this combination, so there is nothing to be inside of.
        pytest.skip(sdk.describe(status))
    assert status.snapshot.bands["python"] in {"ok", "below", "above"}


@pytest.mark.nt_backend("cpu")
def test_deviations_name_the_window_they_violate() -> None:
    snapshot = sdk.detect()
    for deviation in snapshot.deviations:
        assert deviation
