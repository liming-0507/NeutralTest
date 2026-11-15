"""One script per numbered feature of part 7.2.

Each module in this package is a feature -- ``a1.py`` for the first row of the
catalogue, ``f5.py`` for the last -- and declares what to check against a column:

``ID``
    The row's identifier, so a file name and a row number cannot drift apart.
``COLUMNS``
    ``None`` when the script handles every column the applicability table marks,
    or a tuple of column names when it deliberately narrows them.
``run(ctx)``
    The check itself. ``ctx`` is an
    :class:`~neutraltest.gbz73.inference.context.InferenceContext`; the script
    calls :func:`~neutraltest.gbz72._support.check` for the ordinary
    reference-agreement or VJP check, or assembles its own arguments and calls
    the individual verbs. A check that cannot be performed raises
    :class:`NotApplicable` with the reason, which the runner records as ``-``.

A row without a script is a bug in the coverage table rather than a silent gap:
the runner asks for the module by name and fails loudly when the file is absent.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import ModuleType

_MODULE_CACHE: dict[str, ModuleType] = {}


def module_for(feature_id: str) -> ModuleType:
    """Import and validate the script for ``feature_id``, caching it."""
    cached = _MODULE_CACHE.get(feature_id)
    if cached is not None:
        return cached
    name = f"{__name__}.{feature_id.lower()}"
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:  # pragma: no cover - a missing row is a bug
        if exc.name != name:
            raise
        raise ModuleNotFoundError(f"feature {feature_id} has no script: {name}") from exc
    declared = getattr(module, "ID", None)
    if declared != feature_id:
        raise RuntimeError(f"{name} declares ID {declared!r}, expected {feature_id!r}")
    if not callable(getattr(module, "run", None)):
        raise RuntimeError(f"{name} has no run(ctx) function")
    if getattr(module, "COLUMNS", None) is not None and not isinstance(module.COLUMNS, tuple):
        raise RuntimeError(f"{name}.COLUMNS must be None or a tuple of column names")
    _MODULE_CACHE[feature_id] = module
    return module


_SCRIPT_NAME = re.compile(r"^[a-f]\d+$")


def available_features() -> tuple[str, ...]:
    """The rows that have a script, in catalogue order.

    The name pattern allows more than one digit on purpose: ``a10.py`` and
    ``a11.py`` are real rows, and a glob of ``[a-f][0-9].py`` silently drops
    every row past the ninth in a group -- silently, because a missing script is
    exactly what this function is supposed to report.
    """
    found = (
        path.stem.upper()
        for path in Path(__file__).parent.glob("*.py")
        if _SCRIPT_NAME.match(path.stem)
    )
    return tuple(sorted(found, key=_catalogue_key))


def _catalogue_key(feature_id: str) -> tuple[str, int]:
    """Sort by group letter, then numerically so A10 follows A9 rather than A1."""
    return feature_id[0], int(feature_id[1:])


def columns_of(feature_id: str) -> tuple[str, ...] | None:
    """The columns a script narrows itself to, or ``None`` for all marked ones."""
    return getattr(module_for(feature_id), "COLUMNS", None)
