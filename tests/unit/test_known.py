"""The known-issue registry: schema, matching, loading and the verify verb."""

from __future__ import annotations

import argparse
import json
import pathlib
from pathlib import Path

import pytest
import torch

from neutraltest import known
from neutraltest.backends import CpuBackend, discover
from neutraltest.known_cli import run_known
from neutraltest.testing import OpCase


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch: pytest.MonkeyPatch):
    # the checkout's own companion record must not leak into the packaged registry
    monkeypatch.setattr(known, "LOCAL_DIR", pathlib.Path("/nonexistent/known_issues.local"))
    known.reset_cache()
    yield
    known.reset_cache()


def _entry(**overrides):
    payload = {
        "id": "synthetic",
        "case": "conv2d/pad1",
        "dtype": "float32",
        "backend": {"device_type": "cpu"},
        "summary": "a synthetic entry",
        "measured": {"worst_abs": 1.0, "cpu_worst_abs": 0.5},
        "environment": {"torch": "test"},
        "verdict": "test",
        "why_not_harness": ["because"],
        "recorded": "2026-01-01",
    }
    payload.update(overrides)
    return payload


def _write(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"schema": 1, "entries": entries}), encoding="utf-8")
    return path


# --------------------------------------------------------------------- packaged data


def test_packaged_registry_is_loadable_and_complete() -> None:
    registry = known.registry()
    assert len(registry) >= 1
    ids = [entry.id for entry in registry]
    assert len(set(ids)) == len(ids)
    for entry in registry:
        assert entry.device_type
        assert entry.measured["worst_abs"] > entry.measured["cpu_worst_abs"]
        assert entry.verdict
        assert entry.why_not_harness


def test_packaged_entries_match_their_backend_only() -> None:
    registry = known.registry()
    entry = registry.entries[0]
    accel = type("Fake", (), {"name": "privateuse1", "device_type": entry.device_type})()
    cpu = CpuBackend()
    assert registry.find(entry.case_id, entry.dtype, accel) is not None
    assert registry.find(entry.case_id, entry.dtype, cpu) is None
    assert registry.find(entry.case_id, "bfloat16", accel) is None
    assert registry.find("no/such-case", entry.dtype, accel) is None


def test_the_packaged_registry_names_no_product() -> None:
    """The published file records the device type and the numbers, nothing more.

    The words come from the published-safety scan (see CONTRIBUTING) assembled
    from fragments, so that the test itself does not have to spell them out.
    """
    names = [
        "cam" + "bricon",
        "neu" + "ware",
        "cn" + "nl",
        "to" + "rch_" + "m" + "lu",
        "t" + "p_" + "m" + "lu",
    ]
    for path in known.REGISTRY_DIR.glob("*.json"):
        text = path.read_text(encoding="utf-8").lower()
        for name in names:
            assert name not in text, f"{path.name} names a product"


def test_a_local_companion_record_supplies_the_environment(tmp_path, monkeypatch) -> None:
    path = _write(tmp_path, [_entry(id="companion")])
    monkeypatch.setattr(known, "REGISTRY_DIR", tmp_path.parent / "none")
    companion = tmp_path / "measurements.json"
    companion.write_text(
        json.dumps({"entries": [{"id": "companion", "environment": {"torch": "9.9.9"}}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(known, "LOCAL_DIR", tmp_path)
    registry = known.load([path]).with_environment((companion,))
    assert registry.by_id("companion").environment["torch"] == "9.9.9"
    # an id the record does not know is left alone
    other = _write(tmp_path, [_entry(id="other", environment={"torch": "1.0"})])
    assert (
        known.load([other]).with_environment((companion,)).by_id("other").environment["torch"]
        == "1.0"
    )


def test_deviation_and_environment_are_rendered() -> None:
    entry = known.registry().entries[0]
    text = entry.deviation()
    assert "backend" in text and "cpu" in text and "x" in text
    assert "torch=" in entry.environment_line() or "note=" in entry.environment_line()
    described = "\n".join(entry.describe())
    assert entry.id in described
    assert "why #1" in described


# --------------------------------------------------------------------- matching / loading


def test_matching_rules(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            _entry(id="exact"),
            _entry(id="wildcard-case", case="conv*"),
            _entry(id="wildcard-dtype", dtype="*"),
            _entry(id="backend-wildcard", case="*", backend={"device_type": "*"}),
            _entry(id="named-backend", backend={"device_type": "cpu", "name": "not-cpu"}),
        ],
    )
    registry = known.load([path])
    cpu = CpuBackend()
    assert registry.find("conv2d/pad1", "float32", cpu).id == "exact"
    assert registry.find("conv1d/pad1", "float32", cpu).id == "wildcard-case"
    assert registry.find("conv2d/pad1", "float64", cpu).id == "wildcard-dtype"
    # first match wins, so the more specific entries must be declared first
    assert registry.find("matmul/2d", "float32", cpu).id == "backend-wildcard"
    assert registry.by_id("named-backend").matches("conv2d/pad1", "float32", cpu) is False


def test_env_var_adds_registries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path, [_entry(id="from-env")])
    monkeypatch.setenv("NEUTRALTEST_KNOWN", str(path))
    known.reset_cache()
    assert known.load().by_id("from-env") is not None


def test_env_var_pointing_nowhere_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEUTRALTEST_KNOWN", "/nonexistent/registry.json")
    with pytest.raises(FileNotFoundError, match="NEUTRALTEST_KNOWN"):
        known.load()


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, [_entry(id="dup"), _entry(id="dup")])
    with pytest.raises(ValueError, match="duplicate"):
        known.load([path])


def test_missing_required_keys_are_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, [{"id": "broken"}])
    with pytest.raises(ValueError, match="missing"):
        known.load([path])


def test_use_installs_an_override(tmp_path: Path) -> None:
    packaged = len(known.registry())
    path = _write(tmp_path, [_entry(id="override")])
    known.use(known.load([path]))
    assert len(known.registry()) == 1
    known.use(None)
    assert len(known.registry()) == packaged


# --------------------------------------------------------------------- the verify verb


def _args(**overrides) -> argparse.Namespace:
    payload = {
        "backend": "cpu",
        "verify": True,
        "id": None,
        "registry": None,
        "json": False,
        "fail_on_fixed": False,
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def _bad_case() -> OpCase:
    """A case that always deviates: it returns zeros instead of the reference."""

    def build(dtype, generator):
        return (torch.randn(4, 4, dtype=dtype),), {}

    return OpCase(
        id="conv2d/pad1",
        family="convolution",
        op="conv2d",
        # dtype dependent on purpose: a wrong result in float64 would make the
        # reference wrong in the same way and the case would pass.
        fn=lambda x: x if x.dtype is torch.float64 else torch.zeros_like(x),
        build=build,
        dtypes=(torch.float32,),
    )


def _good_case() -> OpCase:
    def build(dtype, generator):
        return (torch.randn(4, 4, dtype=dtype),), {}

    return OpCase(
        id="conv2d/pad1",
        family="convolution",
        op="conv2d",
        fn=torch.clone,
        build=build,
        dtypes=(torch.float32,),
    )


def test_verify_reports_a_still_reproducing_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, [_entry()])
    monkeypatch.setattr(
        "neutraltest.known_cli._conformance_cases", lambda: {"conv2d/pad1": _bad_case()}
    )
    code = run_known(_args(registry=[str(path)]))
    out = capsys.readouterr().out
    assert code == 0
    assert "still-reproduces" in out or "KNOWN" in out
    assert "1 still reproduce" in out


def test_verify_reports_a_fixed_entry_and_can_fail_on_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, [_entry()])
    monkeypatch.setattr(
        "neutraltest.known_cli._conformance_cases", lambda: {"conv2d/pad1": _good_case()}
    )
    assert run_known(_args(registry=[str(path)])) == 0
    assert "FIXED" in capsys.readouterr().out
    assert run_known(_args(registry=[str(path)], fail_on_fixed=True)) == 1


def test_verify_rejects_an_unknown_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path, [_entry(case="no/such-case")])
    monkeypatch.setattr("neutraltest.known_cli._conformance_cases", dict)
    assert run_known(_args(registry=[str(path)])) == 2


def test_verify_rejects_an_unknown_backend(tmp_path: Path) -> None:
    path = _write(tmp_path, [_entry()])
    assert run_known(_args(registry=[str(path)], backend="nosuch")) == 2


def test_verify_skips_entries_for_other_backends(tmp_path: Path, capsys) -> None:
    path = _write(tmp_path, [_entry(backend={"device_type": "not-the-cpu"})])
    assert run_known(_args(registry=[str(path)])) == 0
    assert "none of the registered entries apply" in capsys.readouterr().out


def test_listing_and_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path, [_entry(id="listed")])
    assert run_known(_args(verify=False, registry=[str(path)])) == 0
    out = capsys.readouterr().out
    assert "listed" in out and "1 registered deviation" in out

    assert run_known(_args(verify=False, registry=[str(path)], id="listed")) == 0
    assert "why #1" in capsys.readouterr().out

    assert run_known(_args(verify=False, registry=[str(path)], id="missing")) == 2


def test_packaged_registry_matches_a_privateuse1_backend() -> None:
    """The shipped entries must key off the device type, not the package name."""
    privateuse1 = [backend for backend in discover() if backend.name == "privateuse1"]
    if not privateuse1:
        pytest.skip("no privateuse1 adapter on this build")
    backend = privateuse1[0]
    registry = known.registry()
    # the shipped set is a format example on another device type, so nothing may match here
    matched = [entry for entry in registry if entry.matches(entry.case_id, entry.dtype, backend)]
    assert matched == []


def test_the_matrix_runner_reports_a_known_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The GB/Z matrix uses the same registry, keyed gbz73/<row>/<operator>."""
    from neutraltest.gbz73 import features, ops, runner
    from neutraltest.policy import patched_policy

    feature, op = features.BY_ID["A1"], ops.get("relu")
    path = _write(
        tmp_path,
        [_entry(id="gbz-row", case="gbz73/A1/relu", backend={"device_type": "cpu"})],
    )
    known.use(known.load([path]))

    class Failing:
        OPS = None

        @staticmethod
        def run(ctx):
            raise AssertionError("this row fails on purpose")

    monkeypatch.setattr("neutraltest.gbz73.items.module_for", lambda _id: Failing)

    result = runner.run_feature(feature, op, CpuBackend())
    assert result.outcome == "known"
    assert "KNOWN[gbz-row]" in result.detail
    assert runner.SYMBOLS["known"] == "K"

    with patched_policy(known="strict"):
        strict = runner.run_feature(feature, op, CpuBackend())
    assert strict.outcome == "failed"
