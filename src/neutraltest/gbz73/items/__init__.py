"""One module per numbered feature: ``a1.py`` .. ``f6.py``.

Each module is the script for exactly one row of the matrix, which is the
traceability rule this package follows: a feature exists in the report only if
a file exists that implements it. A module declares

``ID``
    the feature id (``"A2"``), which is matched against the catalogue;
``OPS``
    ``None`` for every operator column, or a tuple of the columns it applies to;
``run(ctx)``
    the checks, raising :class:`~neutraltest.gbz73.context.NotApplicable` when
    the feature cannot be exercised here.

Nothing else is required, and nothing else is consulted.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from ..features import FEATURES, Feature

_MODULE_CACHE: dict[str, ModuleType] = {}


def module_for(feature_id: str) -> ModuleType:
    """Import (once) the script that implements ``feature_id``."""
    if feature_id in _MODULE_CACHE:
        return _MODULE_CACHE[feature_id]
    name = f"{__name__}.{feature_id.lower()}"
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:  # pragma: no cover - a missing script is a bug
        raise ModuleNotFoundError(f"feature {feature_id} has no script: {name}") from exc
    if getattr(module, "ID", None) != feature_id:
        raise RuntimeError(
            f"{name} declares ID={getattr(module, 'ID', None)!r}, expected {feature_id!r}"
        )
    if not callable(getattr(module, "run", None)):
        raise RuntimeError(f"{name} has no run(ctx)")
    _MODULE_CACHE[feature_id] = module
    return module


def available_features() -> tuple[Feature, ...]:
    """Every catalogue feature that has a script on disk."""
    found = []
    for feature in FEATURES:
        try:
            module_for(feature.id)
        except ModuleNotFoundError:
            continue
        found.append(feature)
    return tuple(found)


def ops_of(feature_id: str) -> tuple[str, ...] | None:
    """Operator columns a feature applies to (``None`` means all of them)."""
    return getattr(module_for(feature_id), "OPS", None)
