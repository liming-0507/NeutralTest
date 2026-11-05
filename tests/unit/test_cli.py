"""End-to-end checks of the CLI: it must work from any directory."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch_mbt

SRC = Path(torch_mbt.__file__).resolve().parent.parent


def run_cli(
    *arguments: str, cwd: Path | None = None, timeout: int = 240
) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("TORCH_MBT_BACKEND", None)
    return subprocess.run(
        [sys.executable, "-m", "torch_mbt", *arguments],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
    )


def test_version_flag() -> None:
    result = run_cli("--version")
    assert result.returncode == 0
    assert torch_mbt.__version__ in result.stdout


def test_info_lists_the_discovered_backends() -> None:
    result = run_cli("info")
    assert result.returncode == 0, result.stderr
    assert "torch" in result.stdout
    assert "cpu" in result.stdout
    assert "mechanism" in result.stdout
    assert "tolerance budgets" in result.stdout


def test_info_rejects_an_unknown_backend_with_exit_code_2() -> None:
    result = run_cli("info", "--backend=not-a-backend")
    assert result.returncode == 2
    assert "unknown backend" in result.stderr


def test_doctor_smoke_tests_the_selected_backend() -> None:
    result = run_cli("doctor", "--backend=cpu")
    assert result.returncode == 0, result.stderr
    assert "matmul" in result.stdout
    assert "FAIL" not in result.stdout
    assert result.stdout.count("PASS") >= 6


def test_doctor_skips_unavailable_backends() -> None:
    result = run_cli("doctor", "--backend=cuda")
    if "unavailable" in result.stdout:
        assert result.returncode == 0
    else:  # pragma: no cover - only on a machine with CUDA
        assert result.returncode == 0


def test_test_subcommand_runs_pytest_and_forwards_arguments(tmp_path: Path) -> None:
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    result = run_cli("test", "--backend=cpu", str(tmp_path), "-q")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


def test_test_subcommand_forwards_failures(tmp_path: Path) -> None:
    (tmp_path / "test_bad.py").write_text("def test_bad():\n    assert False\n", encoding="utf-8")
    result = run_cli("test", "--backend=cpu", str(tmp_path), "-q")
    assert result.returncode == 1
    assert "1 failed" in result.stdout
