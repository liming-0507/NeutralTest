"""Run pytest in a subprocess against a throwaway suite.

Kept in its own module so two behaviour modules can share it without importing
each other's tests (which breaks under rootdir-relative collection).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"


def run_pytest(suite: Path, *arguments: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("NEUTRALTEST_BACKEND", None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "neutraltest.pytest_plugin",
            str(suite),
            "-q",
            *arguments,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
