"""A PrivateUse1 backend registered by a third party must show up as ``privateuse1``.

Out-of-tree backends that do not rebuild torch register themselves as
``privateuse1`` under a new name (rename_privateuse1_backend + a device module).
We cannot install one here, so the stub below plays that role: it proves the
adapter contract without any hardware.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

# rename_privateuse1_backend() is process-wide and one-shot, so each scenario
# needs a fresh interpreter.
STUB = r"""
import json
import sys
import types

import torch

name = sys.argv[1]
available = sys.argv[2] == "1"

torch.utils.rename_privateuse1_backend(name)
module = types.ModuleType(f"torch.{name}")
module.is_available = lambda: available
module.device_count = lambda: 2 if available else 0
module.current_device = lambda: 0
module.set_device = lambda index: None
torch._register_device_module(name, module)

from neutraltest.backends import discover

backend = next(b for b in discover() if b.name == "privateuse1")
report = {
    "registered": torch._C._get_privateuse1_backend_name(),
    "device_type": backend.device_type,
    "available": backend.is_available(),
    "count": backend.device_count(),
    "device": str(backend.make_device(1)) if backend.is_available() else None,
    "aliases": list(backend.aliases()),
    "matches_own_name": backend.matches(name),
    "matches_glob": backend.matches("private*"),
    "matches_cpu": backend.matches("cpu"),
}
print(json.dumps(report))
"""


def run_stub(name: str, available: bool) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", STUB, name, "1" if available else "0"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_available_privateuse1_backend_is_detected() -> None:
    report = run_stub("acme", available=True)
    assert report["registered"] == "acme"
    assert report["device_type"] == "acme"
    assert report["available"] is True
    assert report["count"] == 2
    assert report["device"] == "acme:1"


def test_privateuse1_answers_to_its_name_and_both_placeholders() -> None:
    report = run_stub("acme", available=True)
    assert report["matches_own_name"] is True
    assert report["matches_glob"] is True
    assert report["matches_cpu"] is False
    assert {"acme", "privateuseone", "privateuse1"} <= set(report["aliases"])


def test_unavailable_privateuse1_reports_no_devices() -> None:
    report = run_stub("acme", available=False)
    assert report["available"] is False
    assert report["count"] == 0
    assert report["device"] is None


def test_unavailable_privateuse1_refuses_a_device() -> None:
    from neutraltest.backends import discover

    # In this interpreter nothing is registered, so the built-in placeholder is
    # unavailable and must not invent device 0.
    backend = next(b for b in discover() if b.name == "privateuse1")
    assert backend.is_available() is False
    assert backend.device_count() == 0
    with pytest.raises(IndexError):
        backend.make_device(0)
