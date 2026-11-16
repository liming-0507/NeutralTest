"""What one feature of the inference clause needs in order to inspect one column.

This is the clause-7.2 counterpart of :class:`neutraltest.gbz73.context.RunContext`
and deliberately a *separate* object: the FP8 clause runs in float64 and judges
every result against a format floor, while this clause has no quantisation of
its own and runs in float32. Sharing one context would put a 1e-12 tolerance on
a float32 attention kernel -- the second clause would inherit a standard it is
not held to.

Three verbs are the whole surface a feature script needs:

``call()``
    run the column on the device under test (or on any backend), returning a
    float32 tensor;
``gradients()``
    the analytic Jacobian of the **same operator**, on the CPU, in float64 --
    never ``autograd`` on the same tensor, which would make the check a
    tautology;
``check_gradients()``
    both of the above compared through the ordinary tolerance budget.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, NoReturn

import torch

from ...backends.base import Backend
from ...tolerance import tolerance_for
from .. import metrics

# ``NotApplicable`` lives in the sibling clause's context module.  A plain
# ``from .context import ...`` cannot say that from here: this *file* is named
# ``context`` too, so the relative import resolves back onto the module that is
# still half-built.  Importing the sibling under the package path unambiguously
# picks the other file.
from ..context import NotApplicable
from . import kernels, ops
from .columns import Column

#: A float32 column pays for its own arithmetic once, and for the backend's
#: preference once more (a backend is free to use a fused or a wider form).
#: These are the two documented multipliers -- not per-test tuning.
FAMILY_SCALE: dict[str, float] = {
    "add": 2.0,
    "flash_attention": 16.0,
    "kv_cache": 1.0,
    "softmax": 2.0,
    "rope": 2.0,
    "rms_norm": 4.0,
}

#: A backward pays for the forward it differentiates plus the product rule.
BACKWARD_SCALE = 2.0


@dataclass
class InferenceContext:
    """Everything a clause-7.2 feature is allowed to touch."""

    column: Column
    backend: Backend
    feature_id: str
    seed: int = 0
    variant: str = ""
    device_index: int = 0
    notes: list[str] = field(default_factory=list)
    _device: torch.device | None = field(default=None, repr=False)

    # -- identity --------------------------------------------------------
    @property
    def op(self) -> Column:
        """Alias so a shared helper can ask ``ctx.op.name`` of either clause."""
        return self.column

    @property
    def family(self) -> str:
        return self.column.family

    @property
    def backward(self) -> bool:
        return self.column.backward

    def kernel(self):
        return kernels.for_family(self._kernel_family())

    def _kernel_family(self) -> str:
        """The kernel that implements this column.

        The reference addresses the cache as one operator reached two ways
        (write rows, read rows); the kernel layer names those two directions
        separately because a device implements them as two entry points.
        """
        if self.family == "kv_cache":
            return "kv_cache_write" if not self.backward else "kv_cache_read"
        return self.family

    # -- device ----------------------------------------------------------
    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = self.backend.make_device(self.device_index)
        return self._device

    def move(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor) and self.device.type != "cpu":
            return value.to(self.device)
        return value

    def collect(self, value: Any) -> Any:
        if isinstance(value, tuple):
            return tuple(self.collect(v) for v in value)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{self.column.name} returned {type(value).__name__}, not a tensor")
        return value if value.device.type == "cpu" else value.detach().cpu()

    def on_device(self, value: Any) -> Any:
        """``value`` on the device under test, rebuilt by the sampler's own route.

        A feature script that reshapes one dimension builds its arguments with
        the sampler, on the CPU, and then hands them to the kernel.  On a device
        backend that would put two devices in one call -- the error reading
        "<device>:0 and cpu" -- and
        the right fix is not "every script remembers to move its arguments" but
        one place that knows how arguments travel.  Scalars, ``None`` slots and
        index tensors pass through: the sampler keeps them on the CPU on
        purpose, and the kernel's own wrapper moves what it needs.

        Arguments the device cannot hold (no such dtype) fall back to the CPU
        copy, so this can never turn a clean input into a new failure.
        """
        if isinstance(value, tuple):
            return tuple(self.on_device(v) for v in value)
        if not isinstance(value, torch.Tensor):
            return value
        return self.move(value)

    # -- inputs ----------------------------------------------------------
    def sample(self, dtype: torch.dtype = torch.float32, variant: str | None = None) -> ops.Sample:
        return ops.sample(self.family, dtype, seed=self.seed, variant=variant or self.variant)

    def inputs(
        self, dtype: torch.dtype = torch.float32, variant: str | None = None
    ) -> tuple[tuple, dict]:
        """Deterministic inputs, on the CPU, in the shape the reference declares."""
        taken = self.sample(dtype, variant)
        return taken.args, taken.kwargs

    # -- the three verbs --------------------------------------------------
    def call(self, backend: Backend | None = None, *, variant: str | None = None) -> Any:
        """Run the column's kernel on ``backend`` (default: the device under test)."""
        args, kwargs = self.inputs(torch.float32, variant)
        return self.call_with(args, kwargs, backend=backend)

    def call_with(self, args: tuple, kwargs: dict, *, backend: Backend | None = None) -> Any:
        """Run the kernel on arguments a feature script built itself.

        The same movement, synchronisation and collection as :meth:`call`, so a
        script that needs a different shape or a poisoned operand does not have
        to re-implement the device dance.
        """
        target = backend or self.backend
        directional, directional_kwargs = self.project(args, kwargs)
        if target.device_type == "cpu":
            moved = directional
        else:
            device = target.make_device(self.device_index)
            moved = tuple(a.to(device) if isinstance(a, torch.Tensor) else a for a in directional)
        out = self.kernel()(*moved, **directional_kwargs)
        target.synchronize()
        return self.collect(out)

    def project(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """The arguments of the operator *this column* differentiates.

        The cache family is the only one where the forward and the backward are
        different entry points. ``inputs`` always describes the write; the read
        column takes the cache and the slot list from that same call.

        Every consumer goes through here -- kernel call, reference, Jacobian,
        autograd -- so the two directions cannot end up on different data, and
        adding a second directional family is a one-line change instead of four.
        """
        if self.family != "kv_cache":
            return args, kwargs
        keys, values, cache_k, cache_v, index = args
        if self.backward:
            # The read is defined by the rows someone wrote: same cache, same
            # slots, no tokens.
            return (cache_k, cache_v, index), kwargs
        # The write differentiates the rows it stores, into the cache it stores
        # them in: the destination is the tensor the kernel returns, so all five
        # arguments stay.
        return (keys, values, cache_k, cache_v, index), kwargs

    def _directional_inputs(self, dtype: torch.dtype, variant: str | None) -> tuple[tuple, dict]:
        return self.project(*self.inputs(dtype, variant))

    def _directional_positions(self) -> tuple[int, ...]:
        """Argument positions this column differentiates.

        ``freezes`` names what the column holds fixed -- the reduction axis, the
        slot list, the angle source. The reference Jacobian leaves those out of
        its answer, so the autograd cross-check must leave them out too.
        """
        pinned = set(self.column.freezes)
        return tuple(
            index for index, name in enumerate(self._argument_names()) if name not in pinned
        )

    def _argument_names(self) -> tuple[str, ...]:
        forward, _ = self._implementation()
        return tuple(inspect.signature(forward).parameters)

    def _implementation(self) -> tuple[Any, Any]:
        """The ``(forward, jacobian)`` pair this column is measured against.

        The cache is the one family where the column selects the entry point:
        the forward column writes slots, the backward column reads them. Every
        other family has a single implementation, and ``IMPLEMENTATIONS`` is the
        answer for it. Resolving here rather than at each call site is what keeps
        the reference, the Jacobian and the autograd cross-check on one operator.
        """
        if self.family == "kv_cache" and not self.backward:
            return ops.kv_write_forward, ops.kv_write_jacobian
        return ops.IMPLEMENTATIONS[self.family]

    def _forward_reference(self, dtype: torch.dtype, variant: str | None, backward: bool) -> Any:
        # One route for every column: the write stores its rows, the read returns
        # them, and nothing here re-implements either.
        forward, _ = self._implementation()
        directional, directional_kwargs = self._directional_inputs(dtype, variant)
        return forward(*directional, **directional_kwargs)

    def normal(self, variant: str | None = None) -> Any:
        """CPU float32 result: the ordinary baseline."""
        return self._forward_reference(torch.float32, variant, self.backward)

    def golden(self, variant: str | None = None) -> Any:
        """CPU float64 result: the golden baseline."""
        return self._forward_reference(torch.float64, variant, self.backward)

    def reference_with(
        self, args: tuple, kwargs: dict, *, dtype: torch.dtype = torch.float32
    ) -> Any:
        """The reference on arguments a feature script built itself.

        ``normal`` and ``golden`` always run the sampler's own call; a feature
        that varies a shape, an operand or a parameter needs the reference on
        exactly the same call. Floating arguments are cast to ``dtype`` so the
        baseline can be taken in float64 without the caller rebuilding them.
        """
        forward, _ = self._implementation()
        directional, directional_kwargs = self.project(_cast(args, dtype), kwargs)
        return forward(*directional, **directional_kwargs)

    # -- judgement -------------------------------------------------------
    def tolerance_multiplier(self) -> float:
        """How far this column is allowed to drift from its own baseline."""
        return FAMILY_SCALE[self.family] * (BACKWARD_SCALE if self.backward else 1.0)

    def budget(self, kind: str = "elementwise") -> float:
        """The normalised error budget for this column, in float32."""
        return tolerance_for(torch.float32, kind, self.tolerance_multiplier()).rtol

    def diff(self, actual: Any, expected: Any) -> metrics.Diff:
        return metrics.compare(actual, expected)

    def check(
        self, actual: Any, expected: Any, *, rel: float | None = None, msg: str = ""
    ) -> float:
        """Compare against a baseline; return the relative error, raise if over budget."""
        diff = self.diff(actual, expected)
        metrics.assert_within(
            diff, rel=self.budget() if rel is None else rel, msg=msg or self.column.name
        )
        return diff.rel

    # -- gradients -------------------------------------------------------
    def gradients(
        self, dtype: torch.dtype = torch.float64, *, variant: str | None = None
    ) -> tuple[list[torch.Tensor], list[tuple[int, ...]]]:
        """The analytic Jacobian of this column, on the CPU, in ``dtype``.

        Returns the gradients next to the shapes they are supposed to have. The
        two travel together: a gradient vector with the wrong shape is a bug that
        a value comparison alone will happily miss (it broadcasts).
        """
        args, kwargs = self.inputs(dtype, variant)
        return self.gradients_with(args, kwargs, dtype=dtype)

    def gradients_with(
        self, args: tuple, kwargs: dict, *, dtype: torch.dtype = torch.float64
    ) -> tuple[list[torch.Tensor], list[tuple[int, ...]]]:
        """The analytic Jacobian of this column on a caller's own arguments."""
        forward, jacobian = self._implementation()
        directional, directional_kwargs = self.project(_cast(args, dtype), kwargs)
        cotangent = _cotangent(forward(*directional, **directional_kwargs), dtype)
        jac = jacobian(*directional, **directional_kwargs)
        return [g for g in jac(cotangent) if g is not None], _prepared_shapes(directional)

    def autograd_gradients(
        self, dtype: torch.dtype = torch.float64, *, variant: str | None = None
    ) -> list[torch.Tensor]:
        """Reference gradients straight from autograd, used only to cross-check."""
        args, kwargs = self.inputs(dtype, variant)
        return self.autograd_with(args, kwargs, dtype=dtype)

    def autograd_with(
        self, args: tuple, kwargs: dict, *, dtype: torch.dtype = torch.float64
    ) -> list[torch.Tensor]:
        """Autograd's answer for a caller's own arguments."""
        directional, directional_kwargs = self.project(_cast(args, dtype), kwargs)
        leaves: list[torch.Tensor] = []
        rebuilt: list[Any] = []
        for position, arg in enumerate(directional):
            # A free argument is route metadata, not a value the operator
            # differentiates; both sides must ignore it or the comparison is
            # between different questions.
            if (
                position in self._directional_positions()
                and isinstance(arg, torch.Tensor)
                and arg.is_floating_point()
                and arg.is_leaf
            ):
                leaf = arg.detach().clone().requires_grad_(True)
                leaves.append(leaf)
                rebuilt.append(leaf)
            else:
                rebuilt.append(arg)
        forward, _ = self._implementation()
        out = forward(*rebuilt, **directional_kwargs)
        cotangent = _cotangent(out, dtype)
        grads = torch.autograd.grad(out, leaves, cotangent, allow_unused=True)
        return [g for g in grads if g is not None]

    def check_gradients(self, *, rel: float | None = None, msg: str = "") -> tuple[float, float]:
        """Analytic Jacobian vs the backend's own backward, and vs autograd.

        The first number compares the column's gradient (on the device when the
        column is a backward, on the CPU reference otherwise) with the analytic
        Jacobian; the second is the internal consistency check of the reference
        itself, so a broken Jacobian cannot masquerade as a backend defect.
        """
        actual, shapes = self.gradients(torch.float32)
        expected, _ = self.gradients(torch.float64)
        if len(actual) != len(expected):
            raise AssertionError(
                f"{self.column.name}: gradient arity {len(actual)} != {len(expected)}"
            )
        for got, want in zip(actual, expected, strict=True):
            if got.shape != want.shape:
                raise AssertionError(
                    f"{self.column.name}: gradient shape {tuple(got.shape)} != {tuple(want.shape)}"
                )
        budget = self.budget() if rel is None else rel

        # ``metrics.compare`` takes one tensor pair. A column can hand back more
        # than one gradient (attention differentiates q, k and v), so the diff is
        # taken over the concatenation: one budget for the whole column, which is
        # stricter than judging each gradient on its own.
        def _flat(items: list[torch.Tensor]) -> torch.Tensor:
            return torch.cat([t.reshape(-1) for t in items]) if items else torch.zeros(0)

        diff = metrics.compare(_flat(actual), _flat(expected))
        metrics.assert_within(diff, rel=budget, msg=msg or f"{self.column.name}: gradients")
        return diff.rel, budget

    # -- flow control ----------------------------------------------------
    def na(self, reason: str) -> NoReturn:
        raise NotApplicable(reason)

    def require(self, condition: bool, reason: str) -> None:
        if not condition:
            self.na(reason)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def computes_in_float64(self) -> bool:
        """Whether this backend does float64 arithmetic, not just float64 storage.

        Allocating a float64 tensor is not the question: a backend without a
        native float64 kernel hands one out happily and quietly computes in
        float32. The honest probe runs an operation whose float32 answer is
        wrong and asks whether float64 gets it right.
        """
        base = float(2**24)
        try:
            large = torch.tensor([base], dtype=torch.float64, device=self.device)
            one = torch.tensor([1.0], dtype=torch.float64, device=self.device)
            total = (large + one - large).to("cpu")
        except Exception:  # noqa: BLE001 - a backend that cannot run it does not have it
            return False
        # 2**24 + 1 is the smallest integer float32 cannot hold; a backend that
        # really computes in float64 keeps the +1.
        return bool(total.item() == 1.0)


def _cast(args: tuple, dtype: torch.dtype) -> tuple:
    """Every floating tensor in ``args``, viewed in ``dtype``."""
    return tuple(
        a.to(dtype) if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in args
    )


def _cotangent(out: Any, dtype: torch.dtype) -> Any:
    """A deterministic cotangent for every output the operator returns.

    One cotangent per output, never just the first: an operator that answers with
    two tensors has two routes back to the inputs, and a single cotangent silently
    tests one of them. The values are seeded, so a failure reproduces.
    """
    outputs = out if isinstance(out, tuple) else (out,)
    cotangents = tuple(_one_cotangent(t, dtype) for t in outputs)
    return cotangents if isinstance(out, tuple) else cotangents[0]


def _one_cotangent(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # The values are drawn on the host -- a seeded generator is the only way to
    # make them reproducible -- but they have to travel with the output they
    # weight: a device operator handed a host cotangent refuses the call, and a
    # refusal here reads as a broken Jacobian rather than as a stray move.
    generator = torch.Generator().manual_seed(7)
    drawn = torch.randn(tensor.shape, generator=generator).to(tensor.dtype).to(dtype)
    return drawn.to(tensor.device) * 0.5


def _prepared_shapes(args: tuple) -> list[tuple[int, ...]]:
    return [tuple(a.shape) for a in args if isinstance(a, torch.Tensor) and a.is_floating_point()]
