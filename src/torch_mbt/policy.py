"""Process-wide execution policy.

The policy is configured once by the pytest plugin (or by an embedding
application) and read by helpers that need to know how strict to be.  Keeping it
in one place means a test author never has to thread options through fixtures.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Literal

__all__ = [
    "RunPolicy",
    "get_policy",
    "set_policy",
    "reset_policy",
    "patched_policy",
    "OnUnsupported",
]

OnUnsupported = Literal["fail", "skip", "xfail"]

_VALID_ON_UNSUPPORTED = ("fail", "skip", "xfail")


@dataclass
class RunPolicy:
    """How the harness should behave when a backend cannot run something."""

    #: What to do when the backend raises "no kernel / dtype not supported".
    on_unsupported: OnUnsupported = "fail"
    #: Multiplier applied to every tolerance (useful for noisy accelerators).
    tolerance_multiplier: float = 1.0
    #: Fail at collection time when a requested backend is unavailable.
    strict_backends: bool = False
    #: Device ordinal used by the ``device`` fixture.
    device_index: int = 0
    #: Base seed for reproducible input generation.
    seed: int = 20240101

    def __post_init__(self) -> None:
        if self.on_unsupported not in _VALID_ON_UNSUPPORTED:
            raise ValueError(
                f"on_unsupported must be one of {_VALID_ON_UNSUPPORTED}, "
                f"got {self.on_unsupported!r}"
            )
        if self.tolerance_multiplier <= 0:
            raise ValueError(f"tolerance_multiplier must be > 0, got {self.tolerance_multiplier}")
        if self.device_index < 0:
            raise ValueError(f"device_index must be >= 0, got {self.device_index}")

    def as_dict(self) -> dict[str, object]:
        return {
            "on_unsupported": self.on_unsupported,
            "tolerance_multiplier": self.tolerance_multiplier,
            "strict_backends": self.strict_backends,
            "device_index": self.device_index,
            "seed": self.seed,
        }


_POLICY = RunPolicy()


def get_policy() -> RunPolicy:
    """Return the live policy object (mutate through :func:`set_policy`)."""
    return _POLICY


def set_policy(**changes: object) -> RunPolicy:
    """Update fields of the live policy in place and return it."""
    global _POLICY
    candidate = replace(_POLICY, **changes)  # type: ignore[arg-type]
    _POLICY.__dict__.update(candidate.__dict__)
    return _POLICY


def reset_policy() -> RunPolicy:
    """Restore the default policy."""
    return set_policy(**RunPolicy().as_dict())


@contextmanager
def patched_policy(**changes: object) -> Iterator[RunPolicy]:
    """Temporarily change policy fields, e.g. in unit tests."""
    global _POLICY
    saved = _POLICY
    _POLICY = replace(saved, **changes)  # type: ignore[arg-type]
    try:
        yield _POLICY
    finally:
        _POLICY = saved
