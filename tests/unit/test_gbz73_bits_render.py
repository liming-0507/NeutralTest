"""A pass on a cell the clause does not ask for is still a pass, and is still said.

The two matrices print their results beside the coverage the standard's text
declares. Those are two different claims, and the point of printing them in the
same row is that a reader can see when only one of them is true. These tests hold
the rendering to both halves of that: the glyph changes, and nothing else does --
the counts, and with them the exit code of a run, are the ones they were before
there was a second glyph at all.
"""

from __future__ import annotations

import pytest

from neutraltest.backends import CpuBackend
from neutraltest.gbz72 import runner as matrix72
from neutraltest.gbz73 import runner as matrix73
from neutraltest.gbz73.features import COLUMNS as FEATURE_COLUMNS
from neutraltest.gbz73.features import FEATURES, Feature
from neutraltest.gbz73.inference.columns import BY_NAME, COLUMN_SPECS
from neutraltest.gbz73.ops import get as operator


def feature(fid: str) -> Feature:
    return next(item for item in FEATURES if item.id == fid)


def cell(fid: str, op: str, outcome: str) -> matrix73.Result:
    return matrix73.Result(feature=fid, op=op, outcome=outcome, detail="", seconds=0.0)


def report(*cells: matrix73.Result) -> matrix73.Report:
    result = matrix73.Report()
    result.results.extend(cells)
    return result


# --------------------------------------------------------------------------- 7.3


def test_a_pass_the_clause_asks_for_prints_as_before() -> None:
    """A1 states a requirement for all five operators; those cells keep their P."""
    rendered = matrix73.render_matrix(report(*(cell("A1", op, "passed") for op in FEATURE_COLUMNS)))
    row = next(line for line in rendered.splitlines() if line.startswith("A1 "))
    assert row.split()[-5:] == ["P"] * 5


def test_a_pass_the_clause_does_not_ask_for_prints_lower_case() -> None:
    """A2 is stated for none of the five operators."""
    assert not any(feature("A2").covered_by_standard(op) for op in FEATURE_COLUMNS)
    rendered = matrix73.render_matrix(report(*(cell("A2", op, "passed") for op in FEATURE_COLUMNS)))
    row = next(line for line in rendered.splitlines() if line.startswith("A2 "))
    assert row.split()[-5:] == ["p"] * 5


def test_the_two_glyphs_are_not_only_different_but_where_the_bits_differ() -> None:
    """One row, two operators: one the text names, one it does not."""
    stated = next(op for op in FEATURE_COLUMNS if feature("A3").covered_by_standard(op))
    unstated = next(op for op in FEATURE_COLUMNS if not feature("A3").covered_by_standard(op))
    rendered = matrix73.render_matrix(
        report(cell("A3", stated, "passed"), cell("A3", unstated, "passed"))
    )
    row = next(line for line in rendered.splitlines() if line.startswith("A3 "))
    cells = dict(zip(FEATURE_COLUMNS, row.split()[-5:], strict=True))
    assert cells[stated] == "P"
    assert cells[unstated] == "p"


def test_the_legend_explains_the_lower_case_glyph() -> None:
    rendered = matrix73.render_matrix(report())
    assert matrix73.OFF_STANDARD_LEGEND in rendered
    assert "passed, though the clause does not ask this column to" in rendered


def test_the_counts_do_not_move() -> None:
    """p and P are both passes: every count that existed still says what it said."""
    cells = tuple(cell("A2", op, "passed") for op in FEATURE_COLUMNS) + (
        cell("A1", FEATURE_COLUMNS[0], "failed"),
    )
    result = report(*cells)
    summary = matrix73.render_summary(result)

    passed = sum(1 for r in result.results if r.outcome == "passed")
    failed = sum(1 for r in result.results if r.outcome == "failed")
    assert f"passed: {passed}" in summary
    assert f"failed: {failed}" in summary
    # The pass count is the same as the one the plain symbol table would give.
    naive = sum(1 for r in result.results if matrix73.SYMBOLS[r.outcome] in {"P", "p"})
    assert naive == passed


def test_the_summary_says_how_many_passes_were_off_standard() -> None:
    result = report(*(cell("A2", op, "passed") for op in FEATURE_COLUMNS))
    summary = matrix73.render_summary(result)
    assert "off-standard passes: 5 cell(s)" in summary
    assert matrix73.off_standard_passes(result) == len(FEATURE_COLUMNS)


def test_failures_are_never_softened_by_the_new_glyph() -> None:
    result = report(*[cell("A2", op, "failed") for op in FEATURE_COLUMNS])
    assert matrix73.off_standard_passes(result) == 0
    row = next(
        line for line in matrix73.render_matrix(result).splitlines() if line.startswith("A2 ")
    )
    assert row.split()[-5:] == ["F"] * 5


@pytest.mark.parametrize("fid", ["A1", "A2"])
def test_a_real_run_still_counts_its_passes_the_same_way(fid: str) -> None:
    """End to end on the CPU backend, so the claim is about a run and not a stub."""
    ran = matrix73.run_all(
        CpuBackend(), features=(feature(fid),), ops=(operator(FEATURE_COLUMNS[0]),)
    )
    summary = matrix73.render_summary(ran)
    passed = sum(1 for r in ran.results if r.outcome == "passed")
    assert f"passed: {passed}" in summary
    assert matrix73.off_standard_passes(ran) == (passed if fid == "A2" else 0)


# --------------------------------------------------------------------------- 7.2


def test_the_part_7_2_matrix_marks_the_same_way() -> None:
    """A pair the table marks keeps P; a pair it leaves unmarked prints p."""
    result = matrix72.run_all(CpuBackend(), features=FEATURES)
    row = next(
        line for line in matrix72.render_matrix(result).splitlines() if line.startswith("A1 ")
    )
    assert "P" in row
    # Every one of the twelve columns passes A1, but only the marked ones print P.
    marked = sum(1 for spec in COLUMN_SPECS if matrix72.marks.is_marked("A1", spec.name))
    assert len(COLUMN_SPECS) == 12 and marked <= 12


def test_an_unmarked_pass_in_7_2_is_off_table_and_counted_apart() -> None:
    columns = tuple(BY_NAME[name] for name in ("add_fwd", "kv_cache_bwd"))
    # A4 is marked on kv_cache_fwd only -- not on either column shown here.
    result = matrix72.run_all(CpuBackend(), features=(feature("A4"),), columns=columns)
    summary = matrix72.render_summary(result)
    assert "off-standard passes: 0 cell(s)" in summary
    rendered = matrix72.render_matrix(result, columns=columns)
    assert matrix72.OFF_TABLE_LEGEND in rendered


def test_the_part_7_2_summary_still_counts_passes() -> None:
    columns = tuple(BY_NAME[name] for name in ("add_fwd", "kv_cache_bwd"))
    result = matrix72.run_all(CpuBackend(), features=(feature("A1"),), columns=columns)
    passed = sum(1 for r in result.results if r.outcome == "passed")
    summary = matrix72.render_summary(result)
    assert f"passed: {passed}" in summary
    assert f"off-standard passes: {matrix72.off_table_passes(result)} cell(s)" in summary
