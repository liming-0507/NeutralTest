"""``neutraltest known`` -- list the registered deviations and re-check them.

Listing is the traceability surface: every registered cell, the numbers behind
it, and the environment it was measured in.  ``--verify`` is the upgrade path:
it re-runs exactly those cells and says, per entry, whether the deviation still
reproduces.  That is what turns a registration from "we gave up" into "we know
what to watch".

Exit codes: ``0`` the listing succeeded (and, for ``--verify``, the run did too),
``1`` a registered cell could not be reproduced *and* ``--fail-on-fixed`` was
given, ``2`` the request itself was wrong (unknown id, unknown backend, missing
case).
"""

from __future__ import annotations

import importlib
import json
import pathlib
import sys
from typing import Any

import torch

from . import known
from .backends import discover
from .backends.registry import resolve_spec
from .errors import BackendNotFound, UnsupportedByBackend
from .report import render_table
from .testing import OpCase, check_case, evaluate

__all__ = ["run_known"]

_DTYPE_NAMES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}

_STATUS_SYMBOL = {
    "still-reproduces": "KNOWN",
    "no-longer-reproduces": "FIXED",
    "backend-unavailable": "-",
    "unsupported": "-",
    "case-not-found": "?",
    "error": "!",
}


def _conformance_cases() -> dict[str, OpCase]:
    """Every ``OpCase`` in the bundled conformance suite, keyed by case id."""
    package = pathlib.Path(__file__).resolve().parent / "conformance"
    index: dict[str, OpCase] = {}
    for path in sorted(package.glob("test_*.py")):
        module = importlib.import_module(f"neutraltest.conformance.{path.stem}")
        for name, value in vars(module).items():
            if name.endswith("CASES") and isinstance(value, list):
                for case in value:
                    index[case.id] = case
    return index


def _registry(args: Any) -> known.Registry:
    paths = getattr(args, "registry", None)
    return known.load(paths) if paths else known.registry()


def _list(args: Any) -> int:
    registry = _registry(args)
    if not registry:
        print("no known issues registered")
        return 0

    wanted = getattr(args, "id", None)
    if wanted:
        entry = registry.by_id(wanted)
        if entry is None:
            print(f"error: no known issue called {wanted!r}", file=sys.stderr)
            return 2
        print("\n".join(entry.describe()))
        return 0

    if getattr(args, "json", False):
        print(json.dumps([entry.__dict__ for entry in registry], indent=2, default=str))
        return 0

    rows = [
        (
            entry.id,
            entry.case_id,
            entry.dtype,
            entry.device_type,
            entry.deviation(),
            entry.recorded,
        )
        for entry in registry
    ]
    print(render_table(rows, ("id", "case", "dtype", "backend", "measured (abs)", "recorded")))
    print()
    print(f"{len(registry)} registered deviation(s).")
    print("  `neutraltest known --id <id>` prints the full evidence for one entry")
    print("  `neutraltest known --verify` re-runs every entry and reports what still reproduces")
    return 0


def _observe(entry: known.KnownIssue, backend: Any, cases: dict[str, OpCase]) -> dict[str, Any]:
    case = cases.get(entry.case_id)
    if case is None:
        return {"status": "case-not-found", "observed": "", "detail": "case id not in the suite"}
    dtype = _DTYPE_NAMES.get(entry.dtype)
    if dtype is None:
        return {"status": "error", "observed": "", "detail": f"unknown dtype {entry.dtype!r}"}
    index = int(getattr(backend, "default_index", 0) or 0)
    try:
        check_case(case, backend, dtype, index=index)
    except UnsupportedByBackend as exc:
        return {"status": "unsupported", "observed": "", "detail": str(exc)[:160]}
    except Exception:
        # Still deviating: measure how far, so the number can be compared with
        # the one recorded at registration time.
        try:
            actual, expected = evaluate(case, backend.make_device(index), dtype, backend=backend)
            observed = f"{_worst_abs(actual.cpu(), expected.cpu()):.3e}"
        except Exception as exc:
            # Say why instead of a bare "n/a": the operator may have raised, or the
            # backend may still be compiling, and the reader cannot tell them apart.
            observed = f"unreadable ({type(exc).__name__}: {str(exc)[:60]})"
        return {"status": "still-reproduces", "observed": observed, "detail": ""}
    return {"status": "no-longer-reproduces", "observed": "within tolerance", "detail": ""}


def _worst_abs(actual: Any, expected: Any) -> float:
    """Largest absolute difference, over every output the case returns."""
    from .compare import flatten

    worst = 0.0
    for got, want in zip(flatten(actual), flatten(expected), strict=True):
        difference = float((got.double() - want.double()).abs().max()) if got.numel() else 0.0
        worst = max(worst, difference)
    return worst


def _verify(args: Any) -> int:
    registry = _registry(args)
    if not registry:
        print("no known issues registered")
        return 0
    cases = _conformance_cases()
    try:
        backends = resolve_spec(args.backend, discover())
    except BackendNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rows: list[tuple[str, ...]] = []
    records: list[dict[str, Any]] = []
    fixed = 0
    for entry in registry:
        for backend in backends:
            if not entry.matches(entry.case_id, entry.dtype, backend):
                continue
            if not backend.is_available():
                result = {"status": "backend-unavailable", "observed": "", "detail": ""}
            else:
                result = _observe(entry, backend, cases)
            if result["status"] == "no-longer-reproduces":
                fixed += 1
            rows.append(
                (
                    entry.id,
                    backend.name,
                    entry.deviation(),
                    result["observed"],
                    _STATUS_SYMBOL.get(result["status"], result["status"]),
                )
            )
            records.append({"id": entry.id, "backend": backend.name, **result})
            if result["status"] == "case-not-found":
                print(f"error: {entry.id}: {result['detail']}", file=sys.stderr)
                return 2

    if getattr(args, "json", False):
        print(json.dumps(records, indent=2))
    else:
        if rows:
            print(render_table(rows, ("id", "backend", "registered", "observed now", "status")))
            print()
            still = sum(1 for record in records if record["status"] == "still-reproduces")
            print(
                f"{still} still reproduce, {fixed} no longer reproduce, "
                f"{len(records) - still - fixed} not checkable here"
            )
            if fixed:
                print(
                    "a registered deviation no longer reproduces -- the registration can be "
                    "retired (delete the entry) or the tolerance re-examined"
                )
        else:
            print("none of the registered entries apply to the selected backend(s)")

    if fixed and getattr(args, "fail_on_fixed", False):
        return 1
    return 0


def run_known(args: Any) -> int:
    """Entry point for the ``known`` subcommand."""
    return _verify(args) if getattr(args, "verify", False) else _list(args)
