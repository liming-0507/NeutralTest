"""A12 has to be able to fail: a wrong Jacobian must turn the row red.

A12 compares the column's analytic Jacobian with a finite difference of the
column's own forward. That only means something if the comparison notices a
Jacobian that is wrong while the forward is untouched -- which is exactly the
situation autograd cannot catch, because autograd differentiates the forward and
inherits whatever is wrong with it.

Each stub below replaces the reference Jacobian of one family with a wrong one,
runs the row on the real sample, and requires the row to refuse it. The final
test stubs a difference *inside* the tolerance and requires the row to accept it:
without that one, a row that refused everything would pass for discrimination.
"""

# ===== A12 (clause 7.2) =====

from typing import Any

import pytest
import torch

from neutraltest.backends import CpuBackend
from neutraltest.gbz72 import items
from neutraltest.gbz73.inference import context as inference
from neutraltest.gbz73.inference import ops
from neutraltest.gbz73.inference.columns import BY_NAME

#: ``softmax`` is the family the stubs are built on: one tensor in, one tensor
#: out, a Jacobian that is dense, so a scaled or transposed answer stays shaped
#: like a plausible one and cannot be caught by an accident of shape alone.
COLUMN = "softmax_fwd"
FAMILY = "softmax"


def _context() -> Any:
    return inference.InferenceContext(
        column=BY_NAME[COLUMN], backend=CpuBackend(), feature_id="A12", seed=7
    )


def _run() -> None:
    """Run A12 on the real sample; raises whatever the row refused with."""
    items.module_for("A12").run(_context())


def _stub_jacobian(monkeypatch: pytest.MonkeyPatch, wrap: Any) -> None:
    """Replace the family's reference Jacobian with ``wrap`` of the real one.

    ``ops.IMPLEMENTATIONS`` maps a family to a ``(forward, jacobian)`` pair, and
    the jacobian is itself a factory: called with the operands it returns the map
    from a cotangent to the gradient list. ``wrap`` therefore takes and returns
    that inner map, so the stub changes the derivative and nothing else.
    """
    forward, jacobian = ops.IMPLEMENTATIONS[FAMILY]

    def stubbed(*args: Any, **kwargs: Any) -> Any:
        real = jacobian(*args, **kwargs)

        def wrapped(cotangent: Any) -> Any:
            return wrap(real(cotangent))

        return wrapped

    monkeypatch.setitem(ops.IMPLEMENTATIONS, FAMILY, (forward, stubbed))


def test_a12_passes_on_the_real_jacobian() -> None:
    """The control: the row is green on the operator as it ships."""
    _run()


def test_a12_refuses_a_jacobian_scaled_by_a_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dropped factor -- the shape is right, only the magnitude is not."""
    _stub_jacobian(monkeypatch, lambda gradients: [gradients[0] * 1.5])
    with pytest.raises(AssertionError, match="finite difference"):
        _run()


def test_a12_refuses_a_transposed_jacobian(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transposed index -- the classic Jacobian defect, invisible by value."""
    _stub_jacobian(monkeypatch, lambda gradients: [gradients[0].transpose(-1, -2)])
    with pytest.raises(AssertionError, match="shape"):
        _run()


def test_a12_refuses_a_small_relative_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tenth of a percent -- inside what a numeric code path could drift by."""
    _stub_jacobian(monkeypatch, lambda gradients: [gradients[0] * (1 - 1e-3)])
    with pytest.raises(AssertionError, match="finite difference"):
        _run()


def test_a12_accepts_an_error_inside_the_tolerance(monkeypatch: pytest.MonkeyPatch) -> None:
    """A difference the tolerance is meant to absorb.

    Without this one every test above would still pass on a row that refused
    every Jacobian it was given, and the row would be measuring nothing.
    """
    _stub_jacobian(monkeypatch, lambda gradients: [gradients[0] * (1 - 1e-7)])
    _run()


def test_a12_differentiates_the_columns_own_arguments() -> None:
    """The sweep covers what the column differentiates, and nothing else.

    ``rope`` carries its angle table as a float tensor, so a sweep that went by
    dtype would differentiate the position as well; ``kv_cache`` carries a slot
    list that is not differentiable at all. Both are routes, and both have to be
    left out -- ``gradients_with`` reports no block for either.

    The last number is what ``project`` hands over positionally among the swept
    names: attention passes its mask and its scale as keywords, so its three
    tensors are all that can carry a Jacobian block.
    """
    for column, expected in (
        ("rope_fwd", 1),
        ("kv_cache_fwd", 4),
        ("flash_attention_fwd", 3),
        ("rms_norm_fwd", 2),
    ):
        context = inference.InferenceContext(
            column=BY_NAME[column], backend=CpuBackend(), feature_id="A12", seed=7
        )
        args, kwargs = context.inputs(torch.float64, None)
        directional, directional_kwargs = context.project(args, kwargs)
        names = context._argument_names()
        swept = [names[index] for index in context._directional_positions()]
        routed = [name for name in swept if name in directional_kwargs]
        assert not routed, f"{column}: {routed} are keywords, not operands"
        # Every swept operand that survives is a float tensor, so the sweep has
        # a derivative to take rather than a route to trip over.
        for position, value in enumerate(directional):
            if names[position] in swept:
                assert isinstance(value, torch.Tensor) and value.is_floating_point(), (
                    f"{column}: {names[position]} is swept but holds {type(value).__name__}"
                )
        assert len(swept) == expected, f"{column}: swept {swept}"


__all__ = ["COLUMN", "FAMILY"]
