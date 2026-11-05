"""Backend discovery and ``--backend`` resolution.

Discovery order is: built-in mechanism adapters, then third-party adapters
published under the ``neutraltest.backends`` entry point group.  Results are
de-duplicated by device type, because two adapters that resolve to the same
``torch.device`` type would run every test twice for no benefit.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from importlib import metadata

from ..errors import BackendNotFound
from .base import Backend, as_backend
from .builtin import builtin_backends

__all__ = [
    "ENTRY_POINT_GROUP",
    "discover",
    "resolve_spec",
    "find_backend",
    "plugin_backends",
]

#: Entry point group a third-party adapter registers itself under::
#:
#:     [project.entry-points."neutraltest.backends"]
#:     acme = "acme_torch.backend:AcmeBackend"
ENTRY_POINT_GROUP = "neutraltest.backends"


def plugin_backends() -> list[Backend]:
    """Load adapters published as entry points, skipping broken ones."""
    found: list[Backend] = []
    try:
        selected = list(metadata.entry_points(group=ENTRY_POINT_GROUP))
    except Exception:  # noqa: BLE001 - metadata can fail on exotic installs
        return found
    for entry_point in selected:
        try:
            found.append(as_backend(entry_point.load()))
        except Exception as exc:  # noqa: BLE001 - a bad plugin must not kill discovery
            import warnings

            warnings.warn(
                f"neutraltest: entry point {entry_point.name!r} failed to load: {exc!r}",
                stacklevel=2,
            )
    return found


def _dedupe(backends: Iterable[Backend]) -> list[Backend]:
    """Keep one adapter per device type, preferring the natively named one."""
    by_device_type: dict[str, Backend] = {}
    for backend in backends:
        key = backend.device_type
        current = by_device_type.get(key)
        if current is None:
            by_device_type[key] = backend
            continue
        # Prefer the adapter whose handle is the device type itself; otherwise
        # prefer the higher priority one so explicit adapters beat fallbacks.
        current_is_native = current.name == key
        candidate_is_native = backend.name == key
        if (
            candidate_is_native
            and not current_is_native
            or candidate_is_native == current_is_native
            and backend.priority > current.priority
        ):
            by_device_type[key] = backend
    return list(by_device_type.values())


def discover(*, with_entry_points: bool = True, extra: Sequence[Backend] = ()) -> list[Backend]:
    """Return every known backend, de-duplicated, CPU last, stable order."""
    candidates: list[Backend] = list(extra) + builtin_backends()
    if with_entry_points:
        candidates = plugin_backends() + candidates
    backends = _dedupe(candidates)
    return sorted(backends, key=lambda b: (b.name == "cpu", -b.priority, b.name))


def find_backend(backends: Sequence[Backend], name: str) -> Backend:
    """Look up one backend by handle, device type or alias."""
    for backend in backends:
        if backend.matches(name):
            return backend
    known = ", ".join(sorted(b.name for b in backends)) or "<none>"
    raise BackendNotFound(f"unknown backend {name!r}; known backends: {known}")


def resolve_spec(spec: str | None, backends: Sequence[Backend] | None = None) -> list[Backend]:
    """Turn a ``--backend`` value into an ordered backend list.

    Supported forms::

        auto             available backends only (default)
        all              every discovered backend, including unavailable ones
        cpu,xpu          explicit list, same order as written
        *pu              glob over names / device types

    Unavailable backends are *returned* rather than filtered out; the pytest
    plugin parametrises them so the skip is visible in the report instead of
    silently shrinking coverage.
    """
    discovered = list(backends) if backends is not None else discover()
    if spec is None:
        spec = "auto"
    tokens = [token.strip().lower() for token in str(spec).split(",") if token.strip()]
    if not tokens:
        raise BackendNotFound("--backend was given an empty value")

    if "auto" in tokens:
        if len(tokens) > 1:
            raise BackendNotFound("'auto' cannot be combined with explicit backend names")
        return [backend for backend in discovered if backend.is_available()]

    if "all" in tokens:
        if len(tokens) > 1:
            raise BackendNotFound("'all' cannot be combined with explicit backend names")
        return list(discovered)

    selected: list[Backend] = []
    for token in tokens:
        matches = [backend for backend in discovered if backend.matches(token)]
        if not matches:
            known = ", ".join(sorted({backend.name for backend in discovered}))
            raise BackendNotFound(f"unknown backend {token!r}; known backends: {known}")
        for backend in matches:
            if backend not in selected:
                selected.append(backend)
    return selected
