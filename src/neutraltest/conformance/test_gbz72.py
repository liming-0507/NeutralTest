"""The inference clause: twelve columns, each collected once.

Six operators, each with a forward column (what an inference deployment runs)
and a backward column (what a training job needs from the same kernel). Every
column is checked three ways, and the checks are deliberately independent:

``forward``
    the device result against the float32 CPU reference, under the column's own
    budget. A backward column is checked on its Jacobian instead -- comparing a
    backward kernel's output against a forward expression would pass for any
    implementation that returns the right *shape*.

``jacobian``
    the analytic Jacobian against ``autograd``. This is the internal consistency
    check: if the reference and the autograd graph disagree, the reference is
    wrong and no backend result should be believed until that is fixed. It runs
    on the CPU, so a CPU-only machine still exercises it.

``rotation``
    for the rotary embedding only, the invariant that identifies the operator at
    all: rotating a vector pair must preserve its norm. It is independent of the
    reference, of the kernel and of the layout, which is exactly why it caught a
    reference that agreed with an equally wrong kernel.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from neutraltest.gbz73.inference import columns, kernels, ops
from neutraltest.gbz73.inference.columns import Column
from neutraltest.gbz73.inference.context import InferenceContext

COMBINATIONS: list[tuple[Column, str]] = [
    (column, variant)
    for column in (columns.column(name) for name in columns.COLUMNS)
    for variant in ops.variants(column.family)
]

IDS = [f"{column.name}-{variant or 'default'}" for column, variant in COMBINATIONS]

#: The invariant has to hold to float32 rounding, measured against the row norm.
NORM_DRIFT = 1e-5


def _context(column, variant, backend):
    return InferenceContext(
        column=column, backend=backend, feature_id="A1", seed=7, variant=variant
    )


@pytest.mark.parametrize(("column", "variant"), COMBINATIONS, ids=IDS)
def test_forward_matches_the_reference(column, variant, nt) -> None:
    ctx = _context(column, variant, nt.backend)
    actual = ctx.call()
    expected = ctx.normal()
    # A cache column answers with more than one tensor (keys and values); the
    # pair is compared item by item rather than handed to a single-tensor diff.
    assert type(actual) is type(expected), (
        f"{column.name}: kernel returned {type(actual).__name__}, "
        f"reference returned {type(expected).__name__}"
    )
    actuals = actual if isinstance(actual, tuple) else (actual,)
    expecteds = expected if isinstance(expected, tuple) else (expected,)
    assert len(actuals) == len(expecteds), f"{column.name}: different number of outputs"
    for index, (got, want) in enumerate(zip(actuals, expecteds, strict=True)):
        ctx.check(got, want, msg=f"{column.name}[{index}]: device vs float32 reference")


def _fail_message(column: Column, position: int) -> Callable[[str], str]:
    return lambda message: f"{column.name} gradient {position}: {message}"


@pytest.mark.parametrize(("column", "variant"), COMBINATIONS, ids=IDS)
def test_jacobian_matches_autograd(column, variant, nt) -> None:
    ctx = _context(column, variant, nt.backend)
    # The analytic side carries its gradients next to the shapes they must have;
    # only the gradients themselves are compared here.
    actual, _shapes = ctx.gradients(torch.float64)
    expected = ctx.autograd_gradients(torch.float64)
    if len(actual) != len(expected):
        pytest.fail(
            f"{column.name}: {len(actual)} analytic gradient(s) against {len(expected)} "
            "from autograd",
            pytrace=False,
        )
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        if want is None:
            pytest.fail(f"{column.name}: autograd produced no gradient", pytrace=False)
        torch.testing.assert_close(
            got.to(torch.float64),
            want.to(torch.float64),
            rtol=1e-9,
            atol=1e-12,
            msg=_fail_message(column, index),
        )


@pytest.mark.parametrize(
    ("variant",),
    [(variant,) for variant in ops.variants("rope")],
    ids=lambda variant: variant or "default",
)
def test_rotation_preserves_the_norm(variant) -> None:
    """Angle pairs keep their length -- true for every layout, every backend."""
    ctx = InferenceContext(
        column=columns.column("rope_fwd"),
        backend=__import__("neutraltest.backends.builtin", fromlist=["x"]).CpuBackend(),
        feature_id="F1",
        seed=7,
        variant=variant,
    )
    args, kwargs = ctx.inputs(torch.float32)
    x, position = args
    for name, produced in (
        ("reference", ops.rope_forward(x, position, **kwargs)),
        ("kernel", kernels.rope(x, position, **kwargs)),
    ):
        before = x.norm(dim=-1)
        after = produced.norm(dim=-1)
        drift = float((before - after).abs().max())
        assert drift < NORM_DRIFT, (
            f"{name} with variant {variant or 'default'!r} changes the row norm by {drift:.3e}; "
            "a rotation cannot change the length of the pair it rotates"
        )


def test_every_column_has_a_kernel_and_a_reference() -> None:
    """The catalogue is complete: no column is listed twice or left unimplemented."""
    assert len(columns.COLUMNS) == 12
    assert set(columns.COLUMNS) == set(columns.BY_NAME)
    for name in columns.FAMILIES:
        for suffix in ("_fwd", "_bwd"):
            assert f"{name}{suffix}" in columns.BY_NAME
    families = {column.family for column in columns.COLUMN_SPECS}
    assert families == set(columns.FAMILIES)
    option_keys = set(kernels.OPTIONS)
    for family in families:
        # The kernel layer names the two cache directions separately (a device
        # implements them as two entry points); every other family is one name.
        expected_keys = {family} if family != "kv_cache" else {"kv_cache_write", "kv_cache_read"}
        assert expected_keys <= option_keys, f"{family}: no kernel options registered"
        assert ops.IMPLEMENTATIONS[family][0] is not None
