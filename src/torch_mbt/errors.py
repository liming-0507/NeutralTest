"""Exception types used across :mod:`torch_mbt`."""

from __future__ import annotations

__all__ = [
    "TorchMbtError",
    "BackendNotFound",
    "BackendUnavailable",
    "UnsupportedByBackend",
    "is_unsupported_error",
]


class TorchMbtError(Exception):
    """Base class for every error raised by this package."""


class BackendNotFound(TorchMbtError, KeyError):
    """A backend name was requested that no discovered backend provides."""

    def __str__(self) -> str:  # KeyError quotes its argument, which looks noisy here
        args = self.args
        return str(args[0]) if args else ""


class BackendUnavailable(TorchMbtError):
    """A backend exists in this installation but cannot be used right now."""


class UnsupportedByBackend(RuntimeError):
    """The backend has no kernel for the requested operation or dtype.

    This is deliberately *not* a test skip by itself: a conformance suite should
    report a missing operator as a finding.  Use
    :func:`torch_mbt.testing.unsupported_policy` (or ``--on-unsupported=skip``)
    to downgrade it to a skip.
    """


#: Substrings that vendor runtimes use when a kernel/dtype is not implemented.
#: Matching is case-insensitive and purely heuristic; the default policy still
#: fails the test, so a false positive is visible rather than silent.
_UNSUPPORTED_PATTERNS = (
    "not implemented for",
    "is not implemented",
    "not supported",
    "unsupported",
    "does not support",
    "doesn't support",
    "no kernel",
    "could not run",
    "not available for",
    "not yet implemented",
    "nyi",
    "unsupported operation",
    "missing kernel",
)


def is_unsupported_error(exc: BaseException) -> bool:
    """Return ``True`` if *exc* looks like "this backend has no such kernel"."""
    if isinstance(exc, UnsupportedByBackend):
        return True
    if isinstance(exc, (NotImplementedError,)):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(pattern in text for pattern in _UNSUPPORTED_PATTERNS)
