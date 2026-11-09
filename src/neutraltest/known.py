"""Known, explained backend deviations.

A *known issue* is a deviation that has been measured and attributed to the
backend rather than to this harness.  It is deliberately **not** a skip: the
cell still runs, and the report prints it as ``KNOWN`` together with the entry
id, so the deviation stays visible and traceable instead of disappearing into a
green run.

Every entry records

* what was measured --- the deviation on the backend *and* on the CPU baseline,
* where it was measured --- torch, the backend package, the card, the container,
* why it is a backend behaviour --- the reasoning that rules this harness out.

Registries are JSON files under ``neutraltest/known_issues/``.  Extra files can
be loaded with ``NEUTRALTEST_KNOWN`` (``os.pathsep`` separated) or an explicit
``--known-registry`` on the command line.

The packaged registry is deliberately anonymous: it names the device *type*
(``privateuse1``) and the numbers, but not the product or the container the
numbers were taken in.  That environment is what makes the record credible, so
it lives in a companion file under ``known_issues.local/`` of the checkout the
measurement was made in; entry ids are the join key between the two.
"""

from __future__ import annotations

import fnmatch
import json
import os
import pathlib
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

__all__ = [
    "KnownIssue",
    "Registry",
    "load",
    "registry",
    "use",
    "reset_cache",
    "MODES",
    "REGISTRY_DIR",
    "LOCAL_DIR",
]

#: How a run should treat a cell that matches a registered entry.
#: ``report`` marks it KNOWN (the deviation is expected); ``strict`` fails it
#: anyway so the raw assertion can be inspected while triaging.
MODES = ("report", "strict")

REGISTRY_DIR = pathlib.Path(__file__).resolve().parent / "known_issues"
#: Optional, git-ignored companion tree with the measured environment and evidence.
LOCAL_DIR = pathlib.Path(__file__).resolve().parent / "known_issues.local"

_ENV_VAR = "NEUTRALTEST_KNOWN"


@dataclass(frozen=True)
class KnownIssue:
    """One registered deviation, with the evidence that justifies it."""

    id: str
    summary: str
    case_id: str
    dtype: str
    operation: str = ""
    device_type: str = "*"
    backend: str = "*"
    shape: dict[str, Any] = None  # type: ignore[assignment]
    call: dict[str, Any] = None  # type: ignore[assignment]
    measured: dict[str, Any] = None  # type: ignore[assignment]
    environment: dict[str, Any] = None  # type: ignore[assignment]
    verdict: str = ""
    why_not_harness: tuple[str, ...] = ()
    reference: str = ""
    recorded: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        for name in ("shape", "call", "measured", "environment"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, {})

    # -- construction ----------------------------------------------------
    @classmethod
    def from_dict(cls, payload: dict[str, Any], source: str = "") -> KnownIssue:
        missing = [key for key in ("id", "case", "dtype") if key not in payload]
        if missing:
            raise ValueError(f"{source or 'registry'}: entry is missing {missing}")
        backend = payload.get("backend") or {}
        if isinstance(backend, str):  # the short form: "cuda"
            backend = {"device_type": backend}
        why = payload.get("why_not_harness") or ()
        return cls(
            id=str(payload["id"]),
            summary=str(payload.get("summary", "")),
            case_id=str(payload["case"]),
            dtype=str(payload["dtype"]),
            operation=str(payload.get("operation", "")),
            device_type=str(backend.get("device_type", "*")),
            backend=str(backend.get("name", "*")),
            shape=dict(payload.get("shape") or {}),
            call=dict(payload.get("call") or {}),
            measured=dict(payload.get("measured") or {}),
            environment=dict(payload.get("environment") or {}),
            verdict=str(payload.get("verdict", "")),
            why_not_harness=tuple(str(item) for item in why),
            reference=str(payload.get("reference", "")),
            recorded=str(payload.get("recorded", "")),
            source=source,
        )

    # -- matching --------------------------------------------------------
    def matches(self, case_id: str, dtype: str, backend: Any) -> bool:
        """True when this entry covers ``case_id`` / ``dtype`` on ``backend``."""
        if not fnmatch.fnmatchcase(case_id, self.case_id):
            return False
        if self.dtype not in ("*", dtype):
            return False
        device_type = str(getattr(backend, "device_type", backend))
        if self.device_type not in ("*", "") and self.device_type.lower() != device_type.lower():
            return False
        name = str(getattr(backend, "name", ""))
        return self.backend in ("*", "") or fnmatch.fnmatchcase(name, self.backend)

    # -- rendering -------------------------------------------------------
    @property
    def worst(self) -> float | None:
        value = self.measured.get("worst_abs")
        return float(value) if isinstance(value, int | float) else None

    @property
    def cpu_worst(self) -> float | None:
        value = self.measured.get("cpu_worst_abs")
        return float(value) if isinstance(value, int | float) else None

    def deviation(self) -> str:
        """A one-line "what was measured" summary for the report."""
        parts: list[str] = []
        if self.worst is not None:
            parts.append(f"backend {self.worst:.3e}")
        if self.cpu_worst is not None:
            parts.append(f"cpu {self.cpu_worst:.3e}")
        if self.worst and self.cpu_worst:
            parts.append(f"{self.worst / self.cpu_worst:.0f}x")
        return ", ".join(parts)

    def environment_line(self) -> str:
        order = ("torch", "backend_package", "device", "driver", "container")
        items = [f"{key}={self.environment[key]}" for key in order if key in self.environment]
        items += [f"{k}={v}" for k, v in self.environment.items() if k not in order]
        return ", ".join(items)

    def describe(self) -> list[str]:
        lines = [f"{self.id}: {self.summary}"]
        lines.append(f"  case      {self.case_id} [{self.dtype}]")
        if self.shape:
            shapes = ", ".join(f"{k}{tuple(v)}" for k, v in self.shape.items())
            lines.append(f"  shapes    {shapes}")
        if self.call:
            lines.append(f"  call      {json.dumps(self.call, sort_keys=True)}")
        if self.measured:
            lines.append(f"  measured  {json.dumps(self.measured, sort_keys=True)}")
        if self.environment:
            lines.append(f"  env       {self.environment_line()}")
        if self.verdict:
            lines.append(f"  verdict   {self.verdict}")
        for index, reason in enumerate(self.why_not_harness, start=1):
            lines.append(f"  why #{index}   {reason}")
        if self.recorded:
            lines.append(f"  recorded  {self.recorded}")
        if self.reference:
            lines.append(f"  evidence  {self.reference}")
        return lines


@dataclass(frozen=True)
class Registry:
    """An immutable set of :class:`KnownIssue` entries."""

    entries: tuple[KnownIssue, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def find(self, case_id: str, dtype: str, backend: Any) -> KnownIssue | None:
        for entry in self.entries:
            if entry.matches(case_id, dtype, backend):
                return entry
        return None

    def by_id(self, entry_id: str) -> KnownIssue | None:
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None

    def for_backend(self, backend: Any) -> list[KnownIssue]:
        return [
            entry for entry in self.entries if entry.matches(entry.case_id, entry.dtype, backend)
        ]

    def with_environment(self, paths: tuple[pathlib.Path, ...]) -> Registry:
        """Fold measured-environment records into the entries they name.

        A record is ``{"id": ..., "environment": {...}}``; anything the record
        names replaces the entry's placeholder.  Unknown ids are ignored: the
        file belongs to one checkout and may outlive an entry.
        """
        environment: dict[str, dict[str, str]] = {}
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for record in payload.get("entries", []) or []:
                if "id" in record and isinstance(record.get("environment"), dict):
                    environment[record["id"]] = dict(record["environment"])
        if not environment:
            return self
        entries = []
        for entry in self.entries:
            found = environment.get(entry.id)
            if found is None:
                entries.append(entry)
            else:
                merged = {k: v for k, v in entry.environment.items() if k != "note"}
                merged.update(found)
                entries.append(
                    replace(entry, environment=merged, source=f"{entry.source} + {LOCAL_DIR.name}")
                )
        return Registry(tuple(entries))


def _local_file() -> pathlib.Path | None:
    """The companion record of how the packaged entries were measured, if present."""
    path = LOCAL_DIR / "measurements.json"
    return path if path.is_file() else None


def _registry_files() -> list[pathlib.Path]:
    # The companion record carries only environments and is folded in later, so it
    # is not picked up here.
    files = sorted(REGISTRY_DIR.glob("*.json")) if REGISTRY_DIR.is_dir() else []
    extra = os.environ.get(_ENV_VAR, "")
    for chunk in extra.split(os.pathsep):
        chunk = chunk.strip()
        if not chunk:
            continue
        path = pathlib.Path(chunk).expanduser()
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"{_ENV_VAR} points at {path}, which does not exist")
    return files


def load(paths: list[str | pathlib.Path] | None = None) -> Registry:
    """Load the registries.

    ``paths`` may name JSON files or directories of JSON files; ``None`` means
    "the packaged registry plus whatever ``NEUTRALTEST_KNOWN`` lists".
    """
    files: list[pathlib.Path]
    if paths is None:
        files = _registry_files()
    else:
        files = []
        for item in paths:
            path = pathlib.Path(item).expanduser()
            if path.is_dir():
                files.extend(sorted(path.glob("*.json")))
            else:
                files.append(path)
    entries: list[KnownIssue] = []
    seen: set[str] = set()
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("entries", []) if isinstance(payload, dict) else payload
        for item in raw:
            entry = KnownIssue.from_dict(item, source=path.name)
            if entry.id in seen:
                raise ValueError(f"duplicate known-issue id {entry.id!r} (in {path})")
            seen.add(entry.id)
            entries.append(entry)
    return Registry(tuple(entries))


_OVERRIDE: Registry | None = None


@lru_cache(maxsize=1)
def _packaged() -> Registry:
    registry = load()
    local = _local_file()
    if local is not None:
        registry = registry.with_environment((local,))
    return registry


def registry() -> Registry:
    """The registry in force: an override if one was installed, else the packaged one."""
    return _OVERRIDE if _OVERRIDE is not None else _packaged()


def use(entries: Registry | None) -> None:
    """Install ``entries`` as the registry for this process, or clear with ``None``."""
    global _OVERRIDE
    _OVERRIDE = entries
    _packaged.cache_clear()


def reset_cache() -> None:
    """Forget the cached registry and any override (tests, embedding applications)."""
    global _OVERRIDE
    _OVERRIDE = None
    _packaged.cache_clear()
