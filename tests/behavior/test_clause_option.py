"""The plugin's ``--clause`` option reaches the tests it collects."""

from __future__ import annotations

from pathlib import Path

from _pytest_runner import run_pytest


def _suite(tmp_path: Path) -> Path:
    suite = tmp_path / "test_where.py"
    suite.write_text(
        "from neutraltest import clause\n"
        "\n"
        "def test_the_clause_is_visible():\n"
        "    assert clause.get_clause() in {'73', '72', 'all'}\n"
        "    assert clause.covers(clause.get_clause(), '72') or "
        "clause.get_clause() == '73'\n",
        encoding="utf-8",
    )
    return suite


def test_the_default_selection_runs_both_clauses(tmp_path: Path) -> None:
    result = run_pytest(_suite(tmp_path), "--backend=cpu")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "neutralTest: clause" not in result.stdout


def test_an_explicit_selection_is_reported_and_visible_to_the_test(tmp_path: Path) -> None:
    suite = tmp_path / "test_only72.py"
    suite.write_text(
        "from neutraltest import clause\n"
        "\n"
        "def test_the_selection_is_72():\n"
        "    assert clause.get_clause() == '72'\n",
        encoding="utf-8",
    )
    result = run_pytest(suite, "--backend=cpu", "--clause=72")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


def test_an_unknown_clause_is_refused_at_startup(tmp_path: Path) -> None:
    result = run_pytest(_suite(tmp_path), "--backend=cpu", "--clause=71")
    assert result.returncode != 0
    assert "choose from 73, 72, all" in result.stdout + result.stderr


def _conformance() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "neutraltest" / "conformance"


def test_only_the_selected_clause_is_collected() -> None:
    only73 = run_pytest(_conformance(), "--collect-only", "--clause=73")
    assert only73.returncode == 0, only73.stdout + only73.stderr
    assert "test_gbz72[" not in only73.stdout
    assert "does not cover part 7.2" in only73.stdout

    only72 = run_pytest(_conformance(), "--collect-only", "--clause=72")
    assert only72.returncode == 0, only72.stdout + only72.stderr
    assert "test_gbz73[" not in only72.stdout
    assert "does not cover part 7.3" in only72.stdout
