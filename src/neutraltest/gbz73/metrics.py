"""The three numbers the clause asks for, plus the judgement built on them.

The clause names ``max_abs_err``, ``RMSE`` and ``cosine``. Those describe a
result but cannot accept or reject it, which is why the same module also
computes the *normalised* difference -- the quantity that actually has a known
bound for a given FP8 format:

``rel = sum|actual - golden| / sum|golden|``

Dividing by the sum (rather than dividing element-wise) keeps the metric honest
where the output crosses zero, which GELU and LayerNorm both do.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class Diff:
    """How far a result sits from its reference."""

    rel: float
    max_abs: float
    rmse: float
    cosine: float
    max_rel: float
    exact: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "rel": self.rel,
            "max_abs": self.max_abs,
            "rmse": self.rmse,
            "cosine": self.cosine,
            "max_rel": self.max_rel,
            "exact": self.exact,
        }

    def __str__(self) -> str:
        return (
            f"rel={self.rel:.3e} max_abs={self.max_abs:.3e} rmse={self.rmse:.3e} "
            f"cos={self.cosine:.6f} max_rel={self.max_rel:.3e}"
        )


def _as_double(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(torch.float64)


def compare(actual: torch.Tensor, expected: torch.Tensor) -> Diff:
    """Diff of ``actual`` against ``expected``, computed in float64."""
    left = _as_double(actual).reshape(-1)
    right = _as_double(expected).reshape(-1)
    if left.numel() != right.numel():
        raise ValueError(f"cannot compare {left.numel()} values with {right.numel()}")
    if left.numel() == 0:
        return Diff(0.0, 0.0, 0.0, 1.0, 0.0, True)

    delta = (left - right).abs()
    denom = right.abs().sum().item() + _EPS
    rel = float(delta.sum().item() / denom)
    max_abs = float(delta.max().item())
    rmse = float(torch.sqrt((delta**2).mean()).item())
    denom_norm = float(left.norm().item() * right.norm().item())
    cosine = 1.0 if denom_norm == 0.0 else float(torch.dot(left, right).item() / denom_norm)
    element = delta / (right.abs() + _EPS)
    exact = bool(torch.equal(left, right))
    return Diff(rel, max_abs, rmse, cosine, float(element.max().item()), exact)


def assert_within(diff: Diff, *, rel: float, msg: str) -> None:
    """Fail with the measured numbers when ``diff.rel`` exceeds ``rel``."""
    if not diff.rel <= rel:
        raise AssertionError(
            f"{msg}: normalised error {diff.rel:.4e} > {rel:.4e} (allowed) [{diff}]"
        )


def assert_exact(diff: Diff, msg: str) -> None:
    """Fail unless the two tensors are bit-identical."""
    if not diff.exact:
        raise AssertionError(f"{msg}: expected bit-identical results [{diff}]")


def summarise(diffs: dict[str, Diff]) -> str:
    """One line per name, sorted, for assertion messages."""
    return "; ".join(f"{name}: {diff}" for name, diff in sorted(diffs.items()))
