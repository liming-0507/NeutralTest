"""Which clause of the standard a run covers.

Part 7.3 of the text describes 45 test features over five operator columns; part
7.2 describes twelve columns (six operators, each forward and backward).  The
same 45-feature catalogue is applied to both, so a run has to be told which
matrix it is producing.

The default is ``all``: a bare ``pytest`` run collects every conformance test,
which is what it did before this module existed.  The ``neutraltest gbz73``
command defaults to ``73`` so its output is unchanged.

The selection is process state rather than a field of :class:`RunPolicy`
because it is not a per-test policy -- it decides *which tests exist*, and
``RunPolicy.as_dict`` is part of the documented policy contract.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

#: The two clauses this framework implements.
CLAUSES: tuple[str, ...] = ("73", "72")

#: The value that selects both.
ALL = "all"

#: Accepted values, in help order.
CHOICES: tuple[str, ...] = ("73", "72", ALL)

#: Default for a pytest run: collect everything.
DEFAULT = ALL

_selected: str = DEFAULT


def selection(choice: str) -> tuple[str, ...]:
    """Expand a ``--clause`` value into the clauses it selects."""
    if choice == ALL:
        return CLAUSES
    if choice not in CLAUSES:
        raise ValueError(f"unknown clause {choice!r}; expected one of {', '.join(CHOICES)}")
    return (choice,)


def covers(choice: str, clause: str) -> bool:
    """Whether ``choice`` selects ``clause``."""
    return clause in selection(choice)


def get_clause() -> str:
    return _selected


def set_clause(choice: str) -> str:
    """Set the process-wide selection; returns the previous value."""
    global _selected
    selection(choice)  # validate before mutating
    previous, _selected = _selected, choice
    return previous


def reset_clause() -> str:
    return set_clause(DEFAULT)


@contextmanager
def patched_clause(choice: str) -> Iterator[str]:
    """Temporarily select ``choice`` and restore the previous value."""
    previous = set_clause(choice)
    try:
        yield choice
    finally:
        set_clause(previous)
