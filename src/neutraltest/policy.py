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
    "OnSdkOutOfBand",
    "KnownMode",
]

OnUnsupported = Literal["fail", "skip", "xfail"]
#: What to do when the environment is outside the SDK / PyTorch support window.
#: ``warn`` is the default: a deviation is information, and the conformance
#: results keep their meaning -- only a run that claims to be on a certified
#: baseline should turn it into an error.
OnSdkOutOfBand = Literal["warn", "skip", "fail"]
KnownMode = Literal["report", "strict"]

_VALID_ON_UNSUPPORTED = ("fail", "skip", "xfail")
_VALID_ON_SDK = ("warn", "skip", "fail")
_VALID_KNOWN = ("report", "strict")


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
    #: How to treat a cell registered in ``known_issues``: ``report`` marks it
    #: KNOWN (the deviation is expected), ``strict`` fails it anyway.
    known: KnownMode = "report"
    #: What to do when the running torch/python (or a recorded SDK component)
    #: sits outside the support window of the selected SDK profile.
    on_sdk_out_of_band: OnSdkOutOfBand = "warn"

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
        if self.known not in _VALID_KNOWN:
            raise ValueError(f"known must be one of {_VALID_KNOWN}, got {self.known!r}")
        if self.on_sdk_out_of_band not in _VALID_ON_SDK:
            raise ValueError(
                f"on_sdk_out_of_band must be one of {_VALID_ON_SDK}, "
                f"got {self.on_sdk_out_of_band!r}"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "on_unsupported": self.on_unsupported,
            "tolerance_multiplier": self.tolerance_multiplier,
            "strict_backends": self.strict_backends,
            "device_index": self.device_index,
            "seed": self.seed,
            "known": self.known,
            "on_sdk_out_of_band": self.on_sdk_out_of_band,
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
