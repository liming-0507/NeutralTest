"""Selection of the clause a run belongs to."""

from __future__ import annotations

import pytest

from neutraltest import clause


@pytest.fixture(autouse=True)
def _restore_clause():
    yield
    clause.reset_clause()


def test_both_clauses_are_offered_and_the_default_is_all() -> None:
    assert clause.CLAUSES == ("73", "72")
    assert clause.CHOICES == ("73", "72", "all")
    assert clause.DEFAULT == "all"
    assert clause.get_clause() == "all"


def test_a_single_clause_selects_only_itself() -> None:
    assert clause.selection("73") == ("73",)
    assert clause.selection("72") == ("72",)


def test_the_word_all_selects_every_clause_in_order() -> None:
    assert clause.selection("all") == clause.CLAUSES


def test_covers_answers_the_question_the_tests_ask() -> None:
    assert clause.covers("all", "72")
    assert clause.covers("all", "73")
    assert clause.covers("72", "72")
    assert not clause.covers("72", "73")
    assert not clause.covers("73", "72")


def test_an_unknown_word_is_refused_and_names_the_choices() -> None:
    with pytest.raises(ValueError, match="unknown clause"):
        clause.selection("71")
    with pytest.raises(ValueError, match="all"):
        clause.set_clause("both")


def test_set_returns_the_previous_selection_and_reset_restores_the_default() -> None:
    assert clause.set_clause("72") == "all"
    assert clause.get_clause() == "72"
    assert clause.set_clause("all") == "72"
    assert clause.reset_clause() == "all"
    assert clause.get_clause() == "all"


def test_patched_clause_restores_the_previous_value_even_on_failure() -> None:
    clause.set_clause("73")
    with clause.patched_clause("72"):
        assert clause.get_clause() == "72"
    assert clause.get_clause() == "73"
    with pytest.raises(RuntimeError), clause.patched_clause("all"):
        raise RuntimeError
    assert clause.get_clause() == "73"
