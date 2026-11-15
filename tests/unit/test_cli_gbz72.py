"""The ``--clause`` option on the ``gbz73`` command."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import neutraltest

SRC = Path(neutraltest.__file__).resolve().parent.parent


def run_cli(*arguments: str, timeout: int = 300) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("NEUTRALTEST_BACKEND", None)
    return subprocess.run(
        [sys.executable, "-m", "neutraltest", *arguments],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def test_the_default_is_part_7_3_only_and_unchanged() -> None:
    default = run_cli("gbz73", "--features", "A1", "--ops", "relu")
    explicit = run_cli("gbz73", "--clause", "73", "--features", "A1", "--ops", "relu")
    assert default.returncode == 0, default.stderr
    assert explicit.returncode == 0, explicit.stderr
    assert default.stdout == explicit.stdout
    assert "clause:" not in default.stdout
    assert "declared by the text: 49 cells; exercised by this run: 1" in default.stdout


def test_part_7_2_renders_the_twelve_columns_of_the_standard() -> None:
    result = run_cli("gbz73", "--clause", "72", "--features", "A1", "--ops", "add_fwd,rms_norm_bwd")
    assert result.returncode == 0, result.stderr
    # The clause line is printed only when more than one clause runs; a single
    # clause is visible in the columns themselves.
    assert "clause:" not in result.stdout
    assert "declared by the text: 191 cells; exercised by this run: 2" in result.stdout
    header = next(line for line in result.stdout.splitlines() if line.startswith("id "))
    assert header.count("add_fw") == 1
    assert "rms_no" in header


def test_part_7_2_accepts_the_standard_column_labels() -> None:
    by_name = run_cli("gbz73", "--clause", "72", "--features", "A1", "--ops", "add_fwd")
    by_label = run_cli("gbz73", "--clause", "72", "--features", "A1", "--ops", "Add-前")
    assert by_label.returncode == 0, by_label.stderr
    assert by_label.stdout == by_name.stdout


def test_part_7_2_refuses_a_name_that_is_not_a_column() -> None:
    result = run_cli("gbz73", "--clause", "72", "--ops", "relu")
    assert result.returncode == 2
    assert "unknown column" in result.stderr


def test_both_clauses_run_when_all_is_asked_for() -> None:
    result = run_cli("gbz73", "--clause", "all", "--features", "A1", "--ops", "add_fwd")
    assert result.returncode == 0, result.stderr
    assert "clause: 73" in result.stdout
    assert "clause: 72" in result.stdout
    assert "declared by the text: 49 cells" in result.stdout
    assert "declared by the text: 191 cells" in result.stdout
