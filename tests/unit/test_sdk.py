"""Unit tests for :mod:`neutraltest.sdk` -- window maths and detection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from neutraltest import sdk

# --------------------------------------------------------------------------- intervals


@pytest.mark.parametrize(
    ("spec", "version", "expected"),
    [
        (">=3.11,<3.16", "3.13.13", "ok"),
        (">=3.11,<3.16", "3.10.4", "below"),
        (">=3.11,<3.16", "3.16.0", "above"),
        (">=3.10,<=3.15", "3.15.0", "ok"),
        (">=3.10,<=3.15", "3.16.0", "above"),
        (">3.10", "3.10.0", "below"),
        (">3.10", "3.10.1", "ok"),
        # A 7.4 unit pin is a floor; only the major.minor is compared.
        (">3.10", "3.10", "ok"),
        ("<3.13", "3.12.9", "ok"),
        ("<3.13", "3.13.0", "above"),
        ("3.12.*", "3.12.4", "ok"),
        ("3.12.*", "3.13.0", "above"),
        ("3.12", "3.12.4", "ok"),
        ("3.12", "3.11.9", "below"),
        ("~=2.4", "2.4.9", "ok"),
        ("~=2.4", "2.5", "above"),
        ("~=2.4", "2.3.1", "below"),
        ("==2.9.1", "2.9.1+cpu", "ok"),
        (">=3.11,<3.16", "2.15.0+cpu", "below"),
        (">=2.13,<2.16", "2.15.0+cpu", "ok"),
        ("==2.9.1", "2.9.2", "above"),
        ("", "anything", "ok"),
        ("   ", "anything", "ok"),
    ],
)
def test_interval(spec: str, version: str, expected: str) -> None:
    assert sdk.interval(spec, version) == expected


def test_parse_interval_normalises_aliases() -> None:
    assert sdk.parse_interval(">=3.10,<=(3.15)") == ((">=", "3.10"), ("<=", "3.15"))
    assert sdk.parse_interval("3.12.*") == ((">=", "3.12"), ("<", "3.13"))
    assert sdk.parse_interval("3.12") == ((">=", "3.12"), ("<", "3.13"))
    assert sdk.parse_interval("~=2.4") == ((">=", "2.4"), ("<", "2.5"))
    assert sdk.parse_interval("") == ()


def test_pytorch_release_strips_local_suffix_and_patch() -> None:
    assert sdk.pytorch_release("2.15.0+cpu") == "2.15"
    assert sdk.pytorch_release("2.9.1") == "2.9"


def test_torch_window_exact_and_fallback() -> None:
    window, exact = sdk.torch_window("2.15.0+cpu")
    assert exact is True
    assert window == sdk.PYTORCH_SUPPORT["2.15"]

    # An unreleased release is judged by the newest promise at or below it, and
    # the caller is told the table did not have it.
    window, exact = sdk.torch_window("2.99.0")
    assert exact is False
    assert window == sdk.PYTORCH_SUPPORT["2.15"]


# --------------------------------------------------------------------------- detection


def _profile(**overrides):
    base = {
        "name": "test-sdk",
        "torch_window": ">=2.13,<2.14",
        "python_window": ">=3.10,<3.13",
        "components": (),
    }
    base.update(overrides)
    return sdk.SdkProfile(**base)


def test_default_profile_is_in_band_for_a_supported_pair() -> None:
    snapshot = sdk.detect(sdk.default_profile(), python="3.13.13", torch_version="2.15.0+cpu")
    assert snapshot.in_band
    assert snapshot.bands["torch"] == "ok"
    assert snapshot.bands["python"] == "ok"


def test_default_profile_flags_python_outside_the_torch_window() -> None:
    # torch 2.15 requires python >= 3.11; 3.10 is inside the package's own claim
    # but outside the interpreter window of that torch.
    snapshot = sdk.detect(sdk.default_profile(), python="3.10.4", torch_version="2.15.0+cpu")
    assert not snapshot.in_band
    assert snapshot.bands["python"] == "below"
    assert "3.11" in snapshot.deviations[0]


def test_profile_python_window_wins_over_pytorch_window() -> None:
    profile = _profile(python_window=">=3.10,<3.16")
    snapshot = sdk.detect(profile, python="3.13.0", torch_version="2.13.0+cpu")
    assert snapshot.bands["python"] == "ok"


def test_component_versions_that_are_declared_are_reported_as_deviation() -> None:
    profile = sdk.SdkProfile(
        name="p",
        torch_window=">=2.13,<2.14",
        python_window=">=3.10,<3.16",
        components=(
            sdk.Component(name="runtime", version="7.8.0", window=">=7.9", measured="declared"),
        ),
    )
    snapshot = sdk.detect(profile, python="3.13.0", torch_version="2.13.0+cpu")
    assert any("runtime 7.8.0" in deviation for deviation in snapshot.deviations)
    rows = sdk.status_rows(snapshot)
    assert ("runtime", "7.8.0", ">=7.9", "below") in rows


def test_a_probed_component_reports_the_measured_version_not_the_declared_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The profile says 7.9.0; the process says 7.8.0.  The measurement wins, and
    # it is the measurement that decides the band.
    fake = type("fake", (), {"__version__": "7.8.0"})()
    monkeypatch.setitem(sys.modules, "example_runtime", fake)
    profile = sdk.SdkProfile(
        name="p",
        python_window=">=3.10,<3.16",
        components=(
            sdk.Component(
                name="runtime",
                version="7.9.0",
                window=">=7.9",
                measured="runtime",
                probe="example_runtime",
            ),
        ),
    )
    snapshot = sdk.detect(profile, python="3.13.0", torch_version="2.13.0+cpu")
    row = next(r for r in sdk.status_rows(snapshot) if r[0] == "runtime")
    assert row[1] == "7.8.0"
    assert row[3] == "below"
    assert any("runtime 7.8.0 is below" in d for d in snapshot.deviations)


def test_a_probe_only_reads_modules_that_are_already_imported() -> None:
    # Nothing named like an adapter is imported by measuring; a module that is
    # absent is simply unreportable.
    before = set(sys.modules)
    snapshot = sdk.detect(
        sdk.SdkProfile(
            name="p",
            python_window=">=3.10,<3.16",
            components=(sdk.Component(name="x", window=">=1", probe="definitely_not_imported"),),
        ),
        python="3.13.0",
        torch_version="2.13.0+cpu",
    )
    assert "definitely_not_imported" not in sys.modules
    assert not set(sys.modules) - before
    assert snapshot.unmeasured == ("x",)


def test_unmeasured_components_are_not_passes() -> None:
    profile = sdk.SdkProfile(
        name="p",
        python_window=">=3.10,<3.16",
        components=(sdk.Component(name="not-imported", measured="runtime"),),
    )
    snapshot = sdk.detect(profile, python="3.13.0", torch_version="2.13.0+cpu")
    assert snapshot.unmeasured == ("not-imported",)
    row = next(row for row in sdk.status_rows(snapshot) if row[0] == "not-imported")
    assert row[1] == "-"
    assert row[3] == "unknown"


def test_measure_bands() -> None:
    ok = sdk.measure(sdk.default_profile())
    assert ok.band in {"ok", "unknown"}

    deviating = _profile()
    status = sdk.measure(deviating)
    assert status.band == "deviation"
    assert sdk.describe(status).startswith("sdk-out-of-band")


# --------------------------------------------------------------------------- profiles


def test_load_catalogue_reads_profile_and_refreshes_the_table(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "pytorch_support": {"9.9": ">=3.11,<3.12", "2.15": ">=3.11,<3.16"},
                "profile": {
                    "name": "from-file",
                    "torch_window": ">=9.9,<9.10",
                    "python_window": ">=3.11,<3.12",
                    "notes": "n",
                    "components": [{"name": "runtime", "version": "1.0", "window": ">=1.0"}],
                },
            }
        ),
        encoding="utf-8",
    )
    profile = sdk.load_catalogue(path)
    try:
        assert profile.name == "from-file"
        assert profile.notes == "n"
        assert profile.components[0].name == "runtime"
        assert profile.components[0].measured == "declared"
        assert sdk.PYTORCH_SUPPORT["9.9"] == ">=3.11,<3.12"
    finally:
        # The table is module state on purpose (a site points at its own copy);
        # put it back so other tests see the bundled snapshot.
        sdk.PYTORCH_SUPPORT.clear()
        sdk.PYTORCH_SUPPORT.update(
            json.loads(
                json.dumps(
                    {
                        "2.15": ">=3.11,<3.16",
                        "2.14": ">=3.10,<3.16",
                        "2.13": ">=3.10,<3.16",
                        "2.12": ">=3.10,<3.15",
                        "2.11": ">=3.10,<3.15",
                        "2.10": ">=3.10,<3.15",
                        "2.9": ">=3.10,<3.15",
                        "2.8": ">=3.9,<3.14",
                        "2.7": ">=3.9,<3.14",
                        "2.6": ">=3.9,<3.14",
                        "2.5": ">=3.9,<3.13",
                        "2.4": ">=3.8,<3.13",
                        "2.3": ">=3.8,<3.12",
                        "2.2": ">=3.8,<3.12",
                        "2.1": ">=3.8,<3.12",
                        "2.0": ">=3.8,<3.12",
                    }
                )
            )
        )
        sdk._release_window.cache_clear()


def test_merge_keeps_first_seen_order_and_intersects_windows() -> None:
    left = sdk.SdkProfile(
        name="a",
        torch_window=">=2.13",
        components=(sdk.Component("runtime", "7.8.0", ">=7.8"),),
    )
    right = sdk.SdkProfile(
        name="b",
        torch_window="<2.14",
        components=(
            sdk.Component("runtime", "", "<7.9"),
            sdk.Component("driver", "3.8.1", ">=3.8"),
        ),
    )
    merged = sdk.merge([left, right])
    assert merged.name == "a + b"
    assert [component.name for component in merged.components] == ["runtime", "driver"]
    assert merged.components[0].version == "7.8.0"
    assert merged.components[0].window == ">=7.8,<7.9"
    assert merged.torch_window == ">=2.13,<2.14"


def test_set_profile_round_trips() -> None:
    profile = _profile()
    previous = sdk.set_profile(profile)
    try:
        assert sdk.active_profile() is profile
    finally:
        sdk.set_profile(previous)
    assert sdk.active_profile().name != "test-sdk" or sdk._selected is previous


def test_render_mentions_source_and_deviations() -> None:
    snapshot = sdk.detect(_profile(), python="3.13.0", torch_version="2.15.0+cpu")
    text = sdk.render(snapshot)
    assert "deviations: 2" in text
    assert sdk.SUPPORT_SOURCE in text
    assert "component" in text
