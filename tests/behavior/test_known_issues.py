"""Known issues, end to end: a registered deviation is KNOWN, not green and not red."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _pytest_runner import run_pytest

#: A case that is wrong by construction: the float64 reference is right, the
#: float32 result is off by 1.0, which no sane tolerance accepts.
SUITE = """
import torch

from neutraltest.conformance.support import run_case
from neutraltest.testing import OpCase

CASE = OpCase(
    id="toy/wrong",
    family="toy",
    op="wrong",
    fn=lambda x: x if x.dtype is torch.float64 else x + 1.0,
    build=lambda dtype, generator: ((torch.zeros(4, 4, dtype=dtype),), {}),
    dtypes=(torch.float32,),
)


def test_toy(backend):
    run_case(CASE, torch.float32, backend)
"""


def _registry(tmp_path: Path, *, device_type: str = "cpu") -> Path:
    path = tmp_path / "known.json"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "entries": [
                    {
                        "id": "toy-wrong-narrow-mantissa",
                        "summary": "the toy operator is off by one on purpose",
                        "backend": {"device_type": device_type},
                        "case": "toy/wrong",
                        "dtype": "float32",
                        "measured": {"worst_abs": 1.0, "cpu_worst_abs": 0.25},
                        "environment": {"torch": "test"},
                        "verdict": "test entry",
                        "why_not_harness": ["because the test says so"],
                        "recorded": "2026-01-01",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_a_registered_deviation_is_reported_as_known(tmp_path: Path) -> None:
    suite = tmp_path / "test_toy.py"
    suite.write_text(SUITE, encoding="utf-8")
    registry = _registry(tmp_path)

    result = run_pytest(suite, "--backend=cpu", "--known-registry", str(registry))
    output = result.stdout + result.stderr

    # pytest 9 announces the summary in a header printed before the tests run, so
    # -q cannot show it; ask for exactly what is being checked instead
    result = run_pytest(suite, "--backend=cpu", "--known-registry", str(registry), "-rx")
    output = result.stdout + result.stderr
    assert "KNOWN[toy-wrong-narrow-mantissa]" in output
    assert result.returncode == 0, output


def test_without_a_registry_the_same_cell_fails(tmp_path: Path) -> None:
    suite = tmp_path / "test_toy.py"
    suite.write_text(SUITE, encoding="utf-8")
    result = run_pytest(suite, "--backend=cpu")
    assert result.returncode != 0
    assert "1 failed" in result.stdout


def test_a_registry_for_another_backend_does_not_hide_the_failure(tmp_path: Path) -> None:
    suite = tmp_path / "test_toy.py"
    suite.write_text(SUITE, encoding="utf-8")
    registry = _registry(tmp_path, device_type="not-the-backend-under-test")
    result = run_pytest(suite, "--backend=cpu", "--known-registry", str(registry))
    assert result.returncode != 0


def test_known_strict_fails_anyway(tmp_path: Path) -> None:
    suite = tmp_path / "test_toy.py"
    suite.write_text(SUITE, encoding="utf-8")
    registry = _registry(tmp_path)
    result = run_pytest(suite, "--backend=cpu", "--known-registry", str(registry), "--known=strict")
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "1 failed" in output


@pytest.mark.parametrize("mode", ["report", "strict"])
def test_known_mode_reaches_the_policy(tmp_path: Path, mode: str) -> None:
    suite = tmp_path / "test_toy.py"
    body = (
        "\ndef test_policy(backend):\n"
        "    from neutraltest.policy import get_policy\n"
        f'    assert get_policy().known == "{mode}"\n'
    )
    suite.write_text(body, encoding="utf-8")
    result = run_pytest(suite, "--backend=cpu", f"--known={mode}")
    assert result.returncode == 0, result.stdout + result.stderr
