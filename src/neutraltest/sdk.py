"""Which SDK release the harness is running against, and what that costs.

The harness sits between two versioned things that move at different speeds:

* **PyTorch**, which publishes a fixed support window per release and a release
  calendar a year ahead, and
* **the accelerator SDK**, which is typically shipped as an all-in-one release
  (driver, runtime, operator libraries, framework adapters, containers) that is
  rebuilt as a whole against one PyTorch version.

A cell that fails is only meaningful once you know whether it failed inside the
window each of them promises.  This module records that promise, measures where
the current interpreter sits, and renders the difference.

Two rules make the answer honest:

1. **Only what can be measured is measured.**  The interpreter version and the
   PyTorch version are read from the running process; nothing about the SDK is
   guessed.  A component that the SDK does not publish a runtime version for is
   reported as *not measured*, never assumed to match.
2. **Deviation is not failure.**  A component outside its window raises the
   ``sdk-out-of-band`` finding, which callers can turn into a skip or a warning
   -- or, for a run that claims to be on a certified baseline, an error.  The
   conformance results themselves do not change.

The window data is data, not code: a snapshot of the upstream release table
lives in :data:`PYTORCH_SUPPORT`, and a site can replace it with
``neutraltest sdk --catalogue FILE`` from JSON without touching source.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import torch

__all__ = [
    "BAND_NAMES",
    "PYTORCH_SUPPORT",
    "SdkProfile",
    "SdkSnapshot",
    "SdkStatus",
    "SUPPORT_SOURCE",
    "SdkVerdict",
    "component_names",
    "default_profile",
    "detect",
    "interval",
    "load_catalogue",
    "merge",
    "parse_interval",
    "pytorch_release",
    "render",
    "set_profile",
    "status_rows",
]

# ---------------------------------------------------------------------------
# Support windows
# ---------------------------------------------------------------------------

#: Where :data:`PYTORCH_SUPPORT` was copied from, and the day it was copied.
#: A window table is only as good as its age; ``neutraltest sdk`` prints both so
#: that nobody mistakes a stale snapshot for a current promise.
SUPPORT_SOURCE = "https://github.com/pytorch/pytorch/blob/main/RELEASE.md (retrieved 2026-11-14)"

#: PyTorch release -> the Python range it supports, from the upstream release
#: compatibility matrix.  Upper bounds are exclusive unless written ``<=``.
PYTORCH_SUPPORT: dict[str, str] = {
    "2.15": ">=3.11,<3.16",
    "2.14": ">=3.10,<3.16",
    "2.13": ">=3.10,<3.16",
    "2.12": ">=3.10,<3.15",
    "2.11": ">=3.10,<3.15",
    "2.10": ">=3.10,<3.15",
    "2.9": ">=3.10,<3.15",
    "2.8": ">=3.9,<3.14",
    "2.7": ">=3.9,<3.14",
    "2.6": ">=3.9,<3.14",
    "2.5": ">=3.9,<3.13",
    "2.4": ">=3.8,<3.13",
    "2.3": ">=3.8,<3.12",
    "2.2": ">=3.8,<3.12",
    "2.1": ">=3.8,<3.12",
    "2.0": ">=3.8,<3.12",
}

#: The oldest Python the package declares support for.  A release that allows
#: more than this does not widen what is tested -- CI only ever installs the
#: versions in its matrix -- so the effective floor is the intersection.
#: It is only applied to the *interpreter* window, never to PyTorch's own: a
#: torch window is a table of python versions, not of torch versions.
PACKAGE_PYTHON_FLOOR = "3.10"

#: How a component sits inside its window.
BAND_NAMES: tuple[str, ...] = ("ok", "below", "above", "unknown")

#: Where a measured version sits relative to its window.
Band = Literal["ok", "below", "above", "unknown"]

#: The verdict a caller acts on -- the band, plus the two cases where there is
#: nothing to be inside of ("unknown") or where it matters ("deviation").
SdkVerdict = Literal["ok", "unknown", "deviation"]

#: Longest operators first: ``>=`` must be matched before ``>``, otherwise a
#: window written ``>=3.11`` would parse as ``>`` with the bound ``=3.11``.
_OPERATORS = ("~=", "==", ">=", "<=", "=", ">", "<")


def _version_key(text: str) -> tuple[int, ...]:
    """Comparable key with a fixed width, so ``3.10`` and ``3.10.0`` are equal.

    Trailing zero components are dropped rather than padded: comparing tuples
    of different length in Python treats ``(3, 10)`` as *smaller* than
    ``(3, 10, 0)``, which would make ``">=3.10"`` reject ``3.10.0``.
    """
    parts: list[int] = []
    for chunk in str(text).strip().split("."):
        digits = "".join(char for char in chunk if char.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts or (0,))


def _next_minor(version: str) -> str:
    """``"3.12"`` -> ``"3.13"``; ``"2"`` -> ``"3"``."""
    parts = version.split(".")
    if len(parts) > 1:
        return f"{parts[0]}.{int(parts[1]) + 1}"
    return str(int(parts[0]) + 1)


def _compare(left: str, right: str) -> int:
    return (_version_key(left) > _version_key(right)) - (_version_key(left) < _version_key(right))


def parse_interval(spec: str) -> tuple[tuple[str, str], ...]:
    """Split an interval spec into ``(operator, version)`` pairs.

    Accepts the usual spellings, including the ones the upstream tables use and
    the caret/tilde shorthands that are common in package metadata::

        ">=3.11,<3.16"     -> ((">=", "3.11"), ("<", "3.16"))
        ">=3.10,<=(3.15)"  -> ((">=", "3.10"), ("<=", "3.15"))
        "3.12.*"           -> ((">=", "3.12"), ("<", "3.13"))
        "~=2.4"            -> ((">=", "2.4"), ("<", "2.5"))
        ""                 -> ()            # anything
    """
    bounds: list[tuple[str, str]] = []
    for raw in spec.replace("(", " ").replace(")", " ").replace(" ", " ").split(","):
        token = raw.strip()
        if not token:
            continue
        for operator in _OPERATORS:
            if token.startswith(operator):
                value = token[len(operator) :].strip()
                break
        else:
            operator, value = "=", token
        wildcard = value.endswith(".*")
        value = value.rstrip(".*")
        if not value:
            continue
        if wildcard or (operator == "=" and len(value.split(".")) < 3):
            # Upstream tables spell "this minor line" as ``3.12.*``; package
            # metadata spells it ``3.12``.  Both mean the whole minor line, so
            # neither is treated as an exact version.
            bounds.append((">=", value))
            bounds.append(("<", _next_minor(value)))
            continue
        if operator == "~=":
            bounds.append((">=", value))
            bounds.append(("<", _next_minor(value)))
            continue
        if operator == "=":
            operator = "=="
        bounds.append((operator, value))
    return tuple(bounds)


def interval(spec: str, version: str) -> Band:
    """Where *version* sits relative to *spec*: ``ok``, ``below``, ``above``."""
    # Stripping the local/ABI suffix here (rather than at every call site) keeps
    # "2.15.0+cpu" comparable with a window written for "2.15".
    version = str(version).split("+")[0].strip()
    if not spec.strip():
        return "ok"
    bounds = parse_interval(spec)
    if not bounds:
        return "ok"
    for operator, bound in bounds:
        order = _compare(version, bound)
        if operator == ">=" and order < 0:
            return "below"
        if operator == ">" and order <= 0:
            # A bare floor such as ">3.10" means "this minor line or newer".
            # Released versions always carry a patch component, so an exact
            # major.minor match is inside the line, not below it.
            if _compare(version, bound) == 0 and version.strip().count(".") < 2:
                continue
            return "below"
        if operator == "<=" and order > 0:
            return "above"
        if operator == "<" and order >= 0:
            return "above"
        if operator == "==" and order != 0:
            return "below" if order < 0 else "above"
    return "ok"


def pytorch_release(version: str | None = None) -> str:
    """The ``major.minor`` of *version* (default: the running torch)."""
    text = torch.__version__ if version is None else str(version)
    parts = text.split("+")[0].split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else text


# ---------------------------------------------------------------------------
# Profile and snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Component:
    """One published version inside an SDK release."""

    name: str
    version: str = ""
    #: Interval the component must satisfy, empty for "no constraint published".
    window: str = ""
    #: Where the version came from: ``runtime`` (read from the process),
    #: ``declared`` (from a profile file) or ``none``.
    measured: str = "none"
    #: Import name of the module that reports this version, if it reports one.
    #: Set by the profile, never guessed: a component that is not probed is
    #: reported as unmeasured rather than assumed to match.
    probe: str = ""

    #: Other import names that may carry the same version (an adapter can be
    #: published under more than one top-level module).
    probe_aliases: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "window": self.window,
            "measured": self.measured,
            "probe": self.probe,
            "band": interval(self.window, self.version) if self.version else "unknown",
        }


@dataclass(frozen=True)
class SdkProfile:
    """What an SDK release promises: its PyTorch window and its components."""

    name: str = ""
    #: Interval of PyTorch versions this SDK release is built against.
    torch_window: str = ""
    #: Interval of Python versions the SDK ships wheels for.
    python_window: str = ""
    components: tuple[Component, ...] = ()
    #: Free-form, copied into the report (e.g. the container tag).
    notes: str = ""
    #: The *torch* window the profile's own release table gives for the running
    #: release.  Kept apart from ``python_window`` even though both are lists of
    #: python versions, because the report judges them against different things.
    torch_support: str = ""

    def component(self, name: str) -> Component | None:
        for component in self.components:
            if component.name == name:
                return component
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "torch_window": self.torch_window,
            "python_window": self.python_window,
            "torch_support": self.torch_support,
            "notes": self.notes,
            "components": [component.as_dict() for component in self.components],
        }


@dataclass(frozen=True)
class SdkSnapshot:
    """The measured state of this interpreter against a profile."""

    python: str
    torch: str
    torch_release: str
    profile: SdkProfile = field(default_factory=SdkProfile)
    #: ``torch`` / ``python`` / ``runtime``, in the order they are reported.
    bands: Mapping[str, Band] = field(default_factory=dict)
    #: Components whose version could not be read from the running process.
    unmeasured: tuple[str, ...] = ()
    #: Every deviation, already worded for a human reader.
    deviations: tuple[str, ...] = ()

    @property
    def in_band(self) -> bool:
        """True when every measurable pair sits inside the profile's promise."""
        return not self.deviations

    def as_dict(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "torch": self.torch,
            "torch_release": self.torch_release,
            "profile": self.profile.as_dict(),
            "bands": dict(self.bands),
            "unmeasured": list(self.unmeasured),
            "deviations": list(self.deviations),
            "in_band": self.in_band,
            "support_source": SUPPORT_SOURCE,
            "pytorch_support": dict(PYTORCH_SUPPORT),
        }

    def summary(self) -> str:
        """One line for a report header."""
        profile = self.profile.name or "unprofiled"
        verdict = "in band" if self.in_band else f"{len(self.deviations)} deviation(s)"
        return f"{profile}: python {self.python}, torch {self.torch} ({verdict})"


def default_profile() -> SdkProfile:
    """The profile with no SDK knowledge: PyTorch's own published window.

    With this profile the snapshot still answers the useful question -- "is the
    torch in this environment one whose Python support is guaranteed?" -- and
    says nothing it cannot measure about the SDK.
    """
    release = pytorch_release()
    window = PYTORCH_SUPPORT.get(release, "")
    if window:
        # Intersect with what this package claims: a wider upstream window does
        # not widen the range CI installs.  The intersection belongs to the
        # interpreter window; torch_window stays empty because the release
        # table describes pythons, not torches.
        window = f"{window},>={PACKAGE_PYTHON_FLOOR}"
    return SdkProfile(
        name=f"PyTorch {release}",
        python_window=window,
        torch_window="",
        torch_support=window,
    )


def load_catalogue(path: str | os.PathLike[str]) -> SdkProfile:
    """Read a profile from JSON.

    The file may carry the PyTorch table itself, so a site can refresh the
    support window without upgrading the package::

        {"pytorch_support": {"2.16": ">=3.11,<3.16"},
         "profile": {"name": "sdk-x-2.13",
                     "torch_window": ">=2.13,<2.14",
                     "python_window": ">=3.10,<3.13",
                     "components": [{"name": "operator-library", "version": "2.3.3",
                                     "window": ">=2.3,<2.4"}]}}
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    table = payload.get("pytorch_support")
    if isinstance(table, Mapping):
        PYTORCH_SUPPORT.clear()
        PYTORCH_SUPPORT.update({str(k): str(v) for k, v in table.items()})
        _release_window.cache_clear()
    profile = payload.get("profile", payload)
    components: list[Component] = []
    for item in profile.get("components", ()):
        aliases = item.get("probe_aliases") or ()
        if isinstance(aliases, str):
            aliases = (aliases,)
        components.append(
            Component(
                name=str(item["name"]),
                version=str(item.get("version", "")),
                window=str(item.get("window", "")),
                measured=str(item.get("measured", "declared")),
                probe=str(item.get("probe", "")),
                probe_aliases=tuple(str(alias) for alias in aliases),
            )
        )
    return SdkProfile(
        name=str(profile.get("name", Path(path).stem)),
        torch_window=str(profile.get("torch_window", "")),
        python_window=str(profile.get("python_window", "")),
        components=tuple(components),
        notes=str(profile.get("notes", "")),
    )


@lru_cache(maxsize=1)
def _release_window() -> tuple[tuple[str, str], ...]:
    return tuple(sorted(PYTORCH_SUPPORT.items(), key=lambda item: _version_key(item[0])))


def _matches_key(pattern: str, version: str) -> bool:
    if not pattern:
        return True
    pattern = pattern.replace("*", "")
    return version.startswith(pattern) or _compare(pattern, version) == 0


def torch_window(version: str | None = None) -> tuple[str, bool]:
    """The PyTorch support window for *version*'s release, and whether it matched.

    *version* is a **PyTorch** version (default: the running one) -- the window
    it returns lists *python* versions, so the two must not be swapped.  An exact
    major.minor wins; otherwise the newest release at or below the queried one is
    used and the second element is False (a build of an unreleased release is
    judged by the newest promise available, and the caller is told so).
    """
    release = pytorch_release(version)
    window = PYTORCH_SUPPORT.get(release)
    if window:
        return window, True
    candidates = [item for item in _release_window() if _compare(item[0], release) <= 0]
    if not candidates:
        return "", False
    return candidates[-1][1], False


def component_names(profile: SdkProfile | None = None) -> tuple[str, ...]:
    return tuple(component.name for component in (profile or default_profile()).components)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

#: Attribute a module publishes its version on.  No vendor is named anywhere in
#: this module: a profile says which import name to look for (``probe``), and
#: every adapter that follows the usual convention answers this attribute.
#:
#: Nothing is imported.  A module is only consulted if it is *already* in
#: ``sys.modules`` -- importing an adapter can initialise hardware, and a version
#: check must never do that.
_PROBE_ATTRIBUTE = "__version__"


def _probe(module_name: str, attribute: str = _PROBE_ATTRIBUTE) -> str:
    module = sys.modules.get(module_name)
    if module is None:
        return ""
    return str(getattr(module, attribute, "") or "")


def _measure_component(component: Component) -> str:
    """The version of *component* that the running process can actually report.

    Reported values take priority over the profile's declared ones: the point of
    a measurement is to confirm the declaration, and a declared version that is
    merely echoed back confirms nothing.
    """
    for module_name in (component.probe, *component.probe_aliases):
        if not module_name:
            continue
        version = _probe(module_name)
        if version:
            return version
    return component.version if component.version else ""


def detect(
    profile: SdkProfile | None = None, python: str | None = None, torch_version: str | None = None
) -> SdkSnapshot:
    """Measure the running interpreter against *profile* (default: PyTorch's own)."""
    python_version = python or ".".join(str(part) for part in sys.version_info[:3])
    torch_version = torch_version or torch.__version__
    active = profile or default_profile()

    window, exact = torch_window(torch_version)
    # A profile's own python window is authoritative for the components it
    # ships; PyTorch's published window only fills in when it does not say.
    published = active.torch_support or (window + f",>={PACKAGE_PYTHON_FLOOR}" if window else "")
    python_window = active.python_window or published
    python_band = interval(python_window, python_version)
    torch_band: Band = interval(active.torch_window, torch_version) if active.torch_window else "ok"

    deviations: list[str] = []
    if python_band != "ok":
        if active.python_window:
            deviations.append(
                f"python {python_version} is {python_band} {active.name}'s window ({python_window})"
            )
        else:
            deviations.append(
                f"python {python_version} is {python_band} torch "
                f"{pytorch_release(torch_version)}'s supported range "
                f"({published or 'unpublished'})"
            )
    if torch_band != "ok":
        deviations.append(
            f"torch {torch_version} is {torch_band} {active.name}'s window ({active.torch_window})"
        )

    unmeasured: list[str] = []
    for component in active.components:
        version = _measure_component(component)
        if not version:
            unmeasured.append(component.name)
            continue
        band = interval(component.window, version)
        if band != "ok":
            deviations.append(
                f"{component.name} {version} is {band} its window ({component.window})"
            )

    bands: dict[str, Band] = {"torch": torch_band, "python": python_band}
    if not exact:
        bands["torch-catalogue"] = "unknown"

    return SdkSnapshot(
        python=python_version,
        torch=torch_version,
        torch_release=pytorch_release(torch_version),
        profile=active,
        bands=bands,
        unmeasured=tuple(unmeasured),
        deviations=tuple(deviations),
    )


def status_rows(snapshot: SdkSnapshot, profile: SdkProfile | None = None) -> list[tuple[str, ...]]:
    """``(component, version, window, band)`` rows for the CLI table.

    ``version`` is ``-`` when the process does not publish it: an unmeasured
    component is reported as such rather than as a pass.
    """
    active = profile or snapshot.profile
    rows: list[tuple[str, ...]] = [
        (
            "python",
            snapshot.python,
            active.python_window or active.torch_support or "-",
            snapshot.bands.get("python", "unknown"),
        ),
        (
            "torch",
            snapshot.torch,
            active.torch_window or "-",
            snapshot.bands.get("torch", "unknown"),
        ),
    ]
    for component in active.components:
        version = _measure_component(component)
        band = interval(component.window, version) if version else "unknown"
        rows.append((component.name, version or "-", component.window or "-", band))
    return rows


def merge(profiles: Iterable[SdkProfile]) -> SdkProfile:
    """Combine several profiles into one, with the components in first-seen order.

    Useful when an environment is assembled from more than one published
    release: the merged profile answers "does the whole stack hold?" even when
    no single release describes it.
    """
    merged: list[Component] = []
    seen: dict[str, int] = {}
    names: list[str] = []
    torch_windows: list[str] = []
    python_windows: list[str] = []
    for profile in profiles:
        if profile.name:
            names.append(profile.name)
        if profile.torch_window:
            torch_windows.append(profile.torch_window)
        if profile.python_window:
            python_windows.append(profile.python_window)
        for component in profile.components:
            if component.name in seen:
                previous = merged[seen[component.name]]
                merged[seen[component.name]] = Component(
                    name=component.name,
                    version=component.version or previous.version,
                    window=_intersect(previous.window, component.window),
                    measured=component.measured
                    if component.measured != "none"
                    else previous.measured,
                )
                continue
            seen[component.name] = len(merged)
            merged.append(component)
    return SdkProfile(
        name=" + ".join(names) if names else "",
        torch_window=",".join(torch_windows),
        python_window=",".join(python_windows),
        components=tuple(merged),
    )


def _intersect(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    return f"{left},{right}"


def render(snapshot: SdkSnapshot) -> str:
    """The whole picture as text: profile, table, deviations, support source."""
    from .report import render_table

    lines: list[str] = []
    profile = snapshot.profile
    lines.append(f"profile    {profile.name or '(none)'}")
    if profile.notes:
        lines.append(f"notes      {profile.notes}")
    lines.append(f"python     {snapshot.python} (this interpreter)")
    lines.append(f"torch      {snapshot.torch} (release {snapshot.torch_release})")
    lines.append("")
    lines.append(render_table(status_rows(snapshot), ("component", "version", "window", "band")))
    if snapshot.bands.get("torch-catalogue") == "unknown":
        lines.append("")
        lines.append(
            f"note: torch {snapshot.torch_release} is not in the support table; the newest "
            "release at or below it was used. Refresh with --catalogue."
        )
    if snapshot.unmeasured:
        lines.append("")
        lines.append(
            "not measured: "
            + ", ".join(snapshot.unmeasured)
            + " (the profile names a component this process does not publish a version for)"
        )
    lines.append("")
    if snapshot.deviations:
        lines.append(f"deviations: {len(snapshot.deviations)}")
        for deviation in snapshot.deviations:
            lines.append(f"  - {deviation}")
    else:
        lines.append("deviations: none")
    lines.append("")
    lines.append(f"support source: {SUPPORT_SOURCE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Process-wide selection, mirroring :mod:`neutraltest.clause`
# ---------------------------------------------------------------------------

_selected: SdkProfile | None = None


def set_profile(profile: SdkProfile | None) -> SdkProfile | None:
    """Set the process-wide profile; returns the previous one.

    ``None`` restores the default (PyTorch's own published window).
    """
    global _selected
    previous, _selected = _selected, profile
    return previous


def active_profile() -> SdkProfile:
    return _selected or default_profile()


@dataclass(frozen=True)
class SdkStatus:
    """The result a caller acts on: the snapshot plus a verdict.

    ``band`` is what the callers branch on:

    * ``ok`` -- measured, inside the window.
    * ``deviation`` -- measured, outside it.
    * ``unknown`` -- the profile could not be judged (no window published, or
      the release is not in the table).
    """

    snapshot: SdkSnapshot
    band: SdkVerdict = "ok"
    reason: str = ""

    @property
    def measured(self) -> bool:
        return self.band != "unknown"


def measure(profile: SdkProfile | None = None) -> SdkStatus:
    """Detect against *profile* and decide the verdict the callers act on."""
    active = profile or active_profile()
    snapshot = detect(active)
    if snapshot.deviations:
        return SdkStatus(snapshot=snapshot, band="deviation", reason="; ".join(snapshot.deviations))
    if snapshot.bands.get("python") == "unknown" or not active.python_window:
        return SdkStatus(
            snapshot=snapshot,
            band="unknown",
            reason=f"no published support window for torch {snapshot.torch_release}",
        )
    return SdkStatus(snapshot=snapshot)


def describe(status: SdkStatus) -> str:
    """A one-line reason suitable for a skip message."""
    if status.band == "deviation":
        return f"sdk-out-of-band: {status.reason}"
    if status.band == "unknown":
        return f"sdk-unknown: {status.reason}"
    return f"sdk in band: {status.snapshot.summary()}"


def _sequence(value: Sequence[str] | str) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else tuple(value)
