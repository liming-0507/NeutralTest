"""Comparison helpers that understand cross-precision, cross-device results.

Everything funnels through :func:`assert_close`, which:

* flattens nested outputs (``topk`` returns a namedtuple, losses a scalar, ...),
* moves both sides to CPU (device results must be comparable with a CPU golden),
* promotes differing float dtypes instead of failing on dtype alone,
* compares integers/bools exactly, floats with a :class:`~torch_mbt.tolerance.Tolerance`,
* reports the offending case id in the assertion message.

``check_dtype`` defaults to ``False`` because the golden value is intentionally
computed at higher precision.  Shape is always checked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .policy import get_policy
from .tolerance import Tolerance, tolerance_for

__all__ = [
    "assert_close",
    "assert_grad_close",
    "assert_all_finite",
    "flatten",
    "to_cpu",
]


def to_cpu(value: Any) -> Any:
    """Detach and move a tensor to CPU; pass anything else through."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu")
    return value


def flatten(value: Any) -> list[Any]:
    """Flatten nested outputs preserving order.

    ``torch.return_types.*`` and ``torch.Size`` are tuples, so namedtuple outputs
    (``max(dim=...)``, ``topk``, ``sort``, ...) are handled without special cases.
    """
    if isinstance(value, Mapping):
        out: list[Any] = []
        for key in value:
            out.extend(flatten(value[key]))
        return out
    if isinstance(value, (list, tuple)) and not isinstance(value, torch.Tensor):
        out = []
        for item in value:
            out.extend(flatten(item))
        return out
    return [value]


def _as_tensor(value: Any, *, side: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (int, float, bool, complex)):
        return torch.as_tensor(value)
    raise TypeError(
        f"{side} must be a tensor or a python scalar, got {type(value).__name__}. "
        "Nested sequences are flattened before this check."
    )


def _is_real_number(t: torch.Tensor) -> bool:
    return t.dtype.is_floating_point or t.dtype.is_complex


def assert_all_finite(
    tensor: torch.Tensor, *, msg: str = "tensor", allow_empty: bool = True
) -> None:
    """Fail if *tensor* holds ``inf``/``nan`` values.

    Only used when the reference result is known to be finite, which makes it a
    cheap and very effective detector of overflow in low precision dtypes.
    """
    if not _is_real_number(tensor) or tensor.numel() == 0:
        return
    finite = torch.isfinite(tensor.detach())
    if not bool(finite.all()):
        bad = int((~finite).sum().item())
        sample = tensor.detach().flatten()[~finite.flatten()][:4]
        raise AssertionError(
            f"{msg}: {bad}/{tensor.numel()} non-finite values in result, first ones: {sample.tolist()}"
        )


def assert_close(
    actual: Any,
    expected: Any,
    *,
    tolerance: Tolerance | None = None,
    dtype: torch.dtype | None = None,
    kind: str = "elementwise",
    multiplier: float | None = None,
    check_dtype: bool = False,
    check_shape: bool = True,
    equal_nan: bool = True,
    exact: bool = False,
    msg: str = "",
) -> None:
    """Assert that ``actual`` matches ``expected`` within the dtype's tolerance.

    Args:
        actual: Device result (tensor or nested structure of tensors).
        expected: Golden result, possibly at higher precision.
        tolerance: Explicit tolerance; overrides ``dtype``/``kind`` lookup.
        dtype: dtype used to look up the tolerance. Defaults to ``actual.dtype``.
        kind: arithmetic class, see :data:`torch_mbt.tolerance.KINDS`.
        multiplier: relaxes the tolerance; defaults to the run policy value.
        check_dtype: Require identical dtypes. Off by default on purpose.
        check_shape: Require identical shapes (default on).
        equal_nan: Treat matching NaN positions as equal (default on).
        exact: Force bit-exact comparison regardless of dtype.
        msg: Prefix for assertion messages, normally the case id.
    """
    actual_flat = flatten(actual)
    expected_flat = flatten(expected)
    prefix = f"{msg}: " if msg else ""

    if len(actual_flat) != len(expected_flat):
        raise AssertionError(
            f"{prefix}output arity mismatch: actual has {len(actual_flat)} tensor(s), "
            f"expected has {len(expected_flat)}"
        )

    mult = get_policy().tolerance_multiplier if multiplier is None else multiplier

    for index, (got, want) in enumerate(zip(actual_flat, expected_flat, strict=True)):
        item_msg = f"{prefix}output[{index}]" if len(actual_flat) > 1 else prefix.rstrip(": ")
        _assert_one(
            _as_tensor(got, side="actual"),
            _as_tensor(want, side="expected"),
            tolerance=tolerance,
            dtype=dtype,
            kind=kind,
            multiplier=mult,
            check_dtype=check_dtype,
            check_shape=check_shape,
            equal_nan=equal_nan,
            exact=exact,
            msg=item_msg,
        )


def _assert_one(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    tolerance: Tolerance | None,
    dtype: torch.dtype | None,
    kind: str,
    multiplier: float,
    check_dtype: bool,
    check_shape: bool,
    equal_nan: bool,
    exact: bool,
    msg: str,
) -> None:
    a = to_cpu(actual)
    e = to_cpu(expected)
    label = msg or "tensor"

    if check_shape and a.shape != e.shape:
        raise AssertionError(
            f"{label}: shape mismatch, actual {tuple(a.shape)} vs expected {tuple(e.shape)}"
        )
    if check_dtype and a.dtype != e.dtype:
        raise AssertionError(f"{label}: dtype mismatch, actual {a.dtype} vs expected {e.dtype}")
    if not check_shape:
        # "shape irrelevant" means "compare the values in order", not "compare
        # a 1-D and a 2-D tensor", which torch.testing would reject on shape.
        a = a.reshape(-1)
        e = e.reshape(-1)

    # Integer / bool results: pad, index and mask operators must be bit-exact.
    if not _is_real_number(a) or not _is_real_number(e):
        if a.dtype != e.dtype:
            a = a.to(e.dtype)
        torch.testing.assert_close(
            a,
            e,
            rtol=0,
            atol=0,
            equal_nan=False,
            check_device=False,
            check_dtype=False,
            check_stride=False,
            msg=lambda m: f"{label}: {m}",
        )
        return

    tol = tolerance
    if exact:
        tol = Tolerance(0.0, 0.0)
    elif tol is None:
        tol = tolerance_for(dtype or a.dtype, kind, multiplier)

    if a.dtype != e.dtype:
        common = torch.promote_types(a.dtype, e.dtype)
        a = a.to(common)
        e = e.to(common)

    if tol.is_exact:
        torch.testing.assert_close(
            a,
            e,
            rtol=0,
            atol=0,
            equal_nan=False,
            check_device=False,
            check_dtype=False,
            check_stride=False,
            msg=lambda m: f"{label}: {m}",
        )
        return

    torch.testing.assert_close(
        a,
        e,
        rtol=tol.rtol,
        atol=tol.atol,
        equal_nan=equal_nan,
        check_device=False,
        check_dtype=False,
        check_stride=False,
        msg=lambda m: f"{label} ({tol}): {m}",
    )


def assert_grad_close(
    actual: Sequence[torch.Tensor | None],
    expected: Sequence[torch.Tensor | None],
    *,
    dtype: torch.dtype,
    kind: str = "composite",
    multiplier: float | None = None,
    msg: str = "gradient",
) -> None:
    """Compare gradient lists, allowing ``None`` on both sides."""
    if len(actual) != len(expected):
        raise AssertionError(f"{msg}: gradient count mismatch {len(actual)} vs {len(expected)}")
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        item = f"{msg}[{index}]"
        if got is None and want is None:
            continue
        if got is None or want is None:
            raise AssertionError(
                f"{item}: one side has no gradient (actual={got!r}, expected={want!r})"
            )
        assert_close(
            got,
            want,
            dtype=dtype,
            kind=kind,
            multiplier=multiplier,
            check_dtype=False,
            msg=item,
        )
