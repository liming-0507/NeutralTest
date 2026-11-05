"""Option handling, end to end: real pytest subprocesses over tiny suites.

These tests load the plugin explicitly (with entry-point autoload disabled) so
they check the plugin's own behaviour rather than whatever happens to be
installed in the environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import neutraltest

SRC = Path(neutraltest.__file__).resolve().parent.parent

BASIC_SUITE = """
import pytest
import torch


def test_device(backend, device):
    assert device.type == backend.device_type


def test_ones(backend, device):
    assert torch.equal(torch.ones(3, device=device).cpu(), torch.ones(3))


@pytest.mark.nt_backend("cuda")
def test_cuda_only(backend):
    assert backend.name == "cuda"
"""

UNSUPPORTED_SUITE = """
from neutraltest.errors import UnsupportedByBackend


def test_missing_kernel(backend):
    raise UnsupportedByBackend("aten::fancy is not implemented for this backend")
"""


def _write(directory: Path, name: str, body: str) -> Path:
    (directory / name).write_text(body, encoding="utf-8")
    return directory


def run_pytest(suite: Path, *arguments: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("NEUTRALTEST_BACKEND", None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "neutraltest.pytest_plugin",
            str(suite),
            "-q",
            *arguments,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


def test_backend_selection_and_markers(tmp_path: Path) -> None:
    suite = _write(tmp_path, "test_basic.py", BASIC_SUITE)
    # -rs surfaces skip reasons, which -q otherwise hides.
    result = run_pytest(suite, "--backend=cpu", "-rs")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert "1 skipped" in result.stdout
    assert "not in ['cuda']" in result.stdout


def test_unavailable_backends_are_skipped_not_dropped(tmp_path: Path) -> None:
    suite = _write(tmp_path, "test_basic.py", BASIC_SUITE)
    result = run_pytest(suite, "--backend=all")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert "skipped" in result.stdout


def test_strict_mode_turns_a_missing_backend_into_a_usage_error(tmp_path: Path) -> None:
    suite = _write(tmp_path, "test_basic.py", BASIC_SUITE)
    result = run_pytest(suite, "--backend=cuda", "--backend-strict")
    assert result.returncode == pytest_usage_error()
    assert "unavailable" in (result.stdout + result.stderr).lower()


def pytest_usage_error() -> int:
    # pytest returns 4 for a usage error.
    return 4


def test_list_backends_short_circuits_the_session(tmp_path: Path) -> None:
    suite = _write(tmp_path, "test_basic.py", BASIC_SUITE)
    result = run_pytest(suite, "--list-backends")
    assert result.returncode == 0
    assert "cpu" in result.stdout
    assert "* cpu" in result.stdout
    assert "passed" not in result.stdout


def test_json_report_is_written_with_backend_attribution(tmp_path: Path) -> None:
    suite = _write(tmp_path, "test_basic.py", BASIC_SUITE)
    report = tmp_path / "report.json"
    result = run_pytest(suite, "--backend=cpu", f"--nt-report={report}")
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["backends"]["cpu"]["passed"] == 2
    assert payload["backends"]["cpu"]["skipped"] == 1
    assert payload["totals"]["passed"] == 2


def test_terminal_summary_is_printed(tmp_path: Path) -> None:
    suite = _write(tmp_path, "test_basic.py", BASIC_SUITE)
    result = run_pytest(suite, "--backend=cpu")
    assert "neutralTest summary" in result.stdout
    assert "backend" in result.stdout
    assert "unsupported" in result.stdout


def test_summary_can_be_silenced(tmp_path: Path) -> None:
    suite = _write(tmp_path, "test_basic.py", BASIC_SUITE)
    result = run_pytest(suite, "--backend=cpu", "--nt-no-summary")
    assert "neutralTest summary" not in result.stdout


def test_unsupported_policy_fails_by_default(tmp_path: Path) -> None:
    suite = _write(tmp_path, "test_unsupported.py", UNSUPPORTED_SUITE)
    result = run_pytest(suite, "--backend=cpu")
    assert result.returncode == 1
    assert "1 failed" in result.stdout
    assert "UnsupportedByBackend" in result.stdout


def test_unsupported_policy_can_skip(tmp_path: Path) -> None:
    suite = _write(tmp_path, "test_unsupported.py", UNSUPPORTED_SUITE)
    result = run_pytest(suite, "--backend=cpu", "--on-unsupported=skip")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 skipped" in result.stdout


def test_unsupported_policy_can_xfail(tmp_path: Path) -> None:
    suite = _write(tmp_path, "test_unsupported.py", UNSUPPORTED_SUITE)
    result = run_pytest(suite, "--backend=cpu", "--on-unsupported=xfail")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 xfailed" in result.stdout


def test_unknown_backend_is_a_usage_error(tmp_path: Path) -> None:
    suite = _write(tmp_path, "test_basic.py", BASIC_SUITE)
    result = run_pytest(suite, "--backend=nope")
    assert result.returncode == 4
    assert "unknown backend" in (result.stdout + result.stderr)


def test_report_header_shows_the_selection(tmp_path: Path) -> None:
    suite = _write(tmp_path, "test_basic.py", BASIC_SUITE)
    # -q suppresses the header block, so this one run asks for verbosity.
    result = run_pytest(suite, "--backend=cpu", "-v")
    assert "neutralTest: backends = cpu" in result.stdout
    assert "tolerance_multiplier:1" in result.stdout
