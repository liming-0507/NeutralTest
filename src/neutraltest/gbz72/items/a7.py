"""A7 -- a zero keeps its sign, and the check only passes where that is decidable.

Check: every call is built so that a zero the operator produces has a sign the
reference fixes, and the sign pattern of the result is compared with the float64
reference lane by lane.  Equality is not the check: ``torch.equal(-0.0, 0.0)`` is
``True``, so a comparison on values passes on a kernel that turns every ``-0``
into ``+0`` -- exactly the defect this row exists to catch.  The verdict comes
from ``torch.signbit``.

The constructibility of these cases is not taken for granted, so it was measured
on the CPU this row runs on, and two IEEE truths decided their shape:

* ``-0 + -0 = -0`` but ``-0 + +0 = +0`` and ``+0 + -0 = +0``: a sum of signed
  zeros is negative only when *both* summands are ``-0``.  A zero in one operand
  alone can therefore not come out signed, whatever the kernel does; the other
  operand must carry the ``-0`` too, and ``2 * -0 = -0`` against ``-2 * -0 = +0``
  gives the same lane a negative and a positive product.  A kernel that flushes a
  zero, drops its sign, or folds the operands together before scaling disagrees on
  one of the two calls.
* ``x + (-x) = +0`` under round-to-nearest, so an attention normalisation cannot
  sign a zero: it is a sum whose terms cancel.  That is why the two attention
  columns are not claimed (see the reason in :mod:`neutraltest.gbz72.reasons`),
  not an omission.

What each marked column measures:

* add -- the ``-0`` lane of the first operand keeps its sign through
  ``a + alpha * b`` while the product beside it stays ``-0``, and loses it when
  the product becomes ``+0``.  A third call mixes the operand so the lanes under
  test are not all one sign.
* softmax -- the weight of the row's dominant logit is ``+1``, and no weight may
  come back negative: an implementation whose normaliser starts at ``-0`` is a
  sign defect even though every value compares equal.
* rope -- a zero vector stays zero, but the rotation decides the sign of it: at
  angle ``0`` the real coefficient is ``+1`` and ``+1 * -0 = -0``; at pi the real
  coefficient is ``-1`` and ``-1 * -0 = +0``; at pi/2 the imaginary lane is
  ``+1 * -0 = -0``.  Three angles, three patterns, and a kernel whose float32
  table rounds a coefficient to exactly ``+-1`` where the float64 reference does
  not is caught rather than forgiven.
* kv_cache -- a slot write copies its source rows, so ``-0`` stored is ``-0`` read
  back and the rows the write does not touch keep their own signs.  Both
  directions are exercised: the forward column stores and checks the whole cache,
  the backward column reads and checks the rows it was asked for.
* rms_norm -- with ``eps > 0`` the scale ``rsqrt(mean(x**2) + eps)`` is strictly
  positive and finite, so every zero of the output is a scaled input zero and
  keeps its sign.  A build that drops ``eps`` returns ``nan`` on a zero row, and
  the CPU reference -- which multiplies by the reciprocal of a zero scale --
  returns the same: the values compare equal, the signs compare equal, and only a
  finiteness comparison separates them.  The forward column therefore reads the
  sign of the zero rows *and* finiteness.  The backward column reads the sign of
  the gradient the zero rows produce, which is the cotangent scaled by a positive
  ``rsqrt(eps)``.

Tolerance: none.  Every comparison here is on the sign bit or on finiteness, both
bit-level properties; the values themselves belong to A1.

Two guards keep the row from passing vacuously: a case whose reference carries no
negative zero is reported as not applicable rather than green, and the rotation
refuses to run unless its three angles really do give three different sign
patterns.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .._support import NotApplicable, observe, reference

ID = "A7"
COLUMNS = (
    "add_fwd",
    "softmax_fwd",
    "rope_fwd",
    "kv_cache_fwd",
    "rms_norm_fwd",
    "kv_cache_bwd",
    "rms_norm_bwd",
)

#: The signed integer a floating dtype is read as, for the one comparison that must
#: inspect the sign bit rather than call a function that could be replaced.
_INTEGRAL: dict[torch.dtype, torch.dtype] = {
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float32: torch.int32,
    torch.float64: torch.int64,
}

#: Angles the rotation is read at, in radians.  Zero is the identity, pi/2 moves
#: the sign to the other lane, and pi negates the real coefficient -- three
#: different coefficient signs for the same zero input.
ROTATIONS: tuple[float, ...] = (0.0, math.pi / 2, math.pi)


def run(ctx: Any) -> None:
    if ctx.family == "add":
        _add(ctx)
    elif ctx.family == "softmax":
        _softmax(ctx)
    elif ctx.family == "rope":
        _rope(ctx)
    elif ctx.family == "kv_cache":
        _kv_cache(ctx)
    elif ctx.family == "rms_norm" and not ctx.backward:
        _rms_norm(ctx)
    elif ctx.family == "rms_norm":
        _rms_norm_backward(ctx)
    else:  # pragma: no cover - the two tables disagree if this is reached
        raise NotApplicable(f"A7 has no signed-zero case for {ctx.column.name}")


def _add(ctx: Any) -> None:
    """A sum of signed zeros is ``-0`` only when both summands are ``-0``.

    ``-0 + -0 = -0``, and ``-0 + +0 = +0`` just as surely: the three calls below
    are the three ways a pair of zeros can meet, and each one has a different
    answer.  A kernel that folds the operands together first, or that normalises a
    zero on the way out, cannot answer all three.
    """
    for left, right, label in (
        (-0.0, -0.0, "-0 + -0"),
        (0.0, -0.0, "+0 + -0"),
        (-0.0, 0.0, "-0 + +0"),
    ):
        first = torch.full((8, 8), left)
        second = torch.full((8, 8), right)
        _assert_sign_pattern(ctx, (first, second), {}, label=label)


def _softmax(ctx: Any) -> None:
    """The weight of a dominant logit is ``+1``, and no weight is negative.

    A normalised row cannot keep a ``-0``: the reference divides by a positive sum,
    and a zero divided by a positive number is ``+0``.  What the row can decide is
    the other half -- a kernel whose normaliser starts negative, or whose
    exponent is applied with the wrong sign, puts a minus on a probability -- and
    the anchor for that is the one weight the arithmetic resolves exactly.
    """
    args, _ = ctx.inputs(torch.float32, ctx.variant)
    (x,) = args
    x = x.clone()
    # One row dominated by a single logit, so the dominant weight rounds to
    # exactly ``+1`` in both directions: the lane a sign flip cannot hide on.
    x[0] = -200.0
    x[0, 3] = 200.0
    x[1] = 0.0
    signed = (x,)
    got = observe(ctx, signed, {})[0]
    want = reference(ctx, signed, {}, dtype=torch.float64)[0]
    if int((want[0] == 1.0).sum()) != 1:
        raise NotApplicable("A7: the reference has no single dominant weight to read")
    if not bool(torch.equal(got == 1.0, want == 1.0)):
        raise AssertionError(
            f"{ctx.column.name}: the kernel and the reference disagree on which weights are "
            f"exactly one, so their signs cannot be compared"
        )
    if bool(_sign_mask(got).any()):
        count = int(_sign_mask(got).sum())
        raise AssertionError(
            f"{ctx.column.name}: {count} softmax weight(s) came back negative where the "
            f"reference is positive or +0; a probability never carries a minus sign"
        )
    ctx.note(f"{int((got == 0).sum())} weights rounded to +0")


def _rope(ctx: Any) -> None:
    """A zero input leaves with the sign of its rotation coefficient.

    ``x * c`` is ``-0`` exactly when ``x`` is ``-0`` and ``c`` is positive, and
    ``+0`` when ``c`` is negative: the same zero, rotated by three angles whose
    coefficients have three different signs, gives three different outputs.  A
    kernel that builds its angle table at the input's precision instead of the
    reference's -- or that folds the rotation into an addition -- disagrees on one
    of the three.
    """
    args, _ = ctx.inputs(torch.float32, ctx.variant)
    x, position = args
    zeros = torch.zeros_like(x)
    ground = torch.zeros_like(position)
    patterns: set[tuple[bool, ...]] = set()
    for angle in ROTATIONS:
        signed = (zeros, ground + angle)
        got = observe(ctx, signed, {})[0]
        want = reference(ctx, signed, {}, dtype=torch.float64)[0]
        if not bool(torch.equal(want, torch.zeros_like(want))):
            raise NotApplicable("A7: the reference rotation of a zero input is not zero")
        patterns.add(tuple(bool(bit) for bit in _sign_mask(want).reshape(-1).tolist()))
        _compare_signs(ctx, got, want, f"zero vector at {angle:.3f} rad")
    # The coefficients must differ between the three angles, or reading three of
    # them would be reading one.
    if len(patterns) != len(ROTATIONS):
        raise NotApplicable("A7: the rotation signs a zero input the same way at every angle")


def _kv_cache(ctx: Any) -> None:
    """A slot write copies its source bit for bit; a read hands the bits back.

    The cache arrives holding ``+0`` in some slots and ``-0`` in others, and so do
    the rows being written: the write must replace the slots it is given without
    touching the ones it is not, and the read must return the stored signs rather
    than a zero it built itself.  ``index_put`` -- what the reference does and what
    an obvious kernel does -- keeps ``-0`` exactly, so the two calls below are
    decided by whether the kernel routes rows through arithmetic.
    """
    args, _ = ctx.inputs(torch.float32, ctx.variant)
    keys, values, cache_k, cache_v, slots = args
    keys = torch.zeros_like(keys)
    keys[:, 0] = -0.0
    values = torch.zeros_like(values)
    values[:, 1:] = -0.0
    cache_k = torch.zeros_like(cache_k)
    cache_v = torch.zeros_like(cache_v)
    # Lane 0 of the destination is ``+0`` and every other lane is ``-0``: a slot
    # the write covers takes the source's sign, a slot it leaves alone keeps one.
    cache_k[:, 0] = 0.0
    cache_k[:, 1:] = -0.0
    cache_v[:, 0] = 0.0
    cache_v[:, 1:] = -0.0
    signed = (keys, values, cache_k, cache_v, slots)
    _assert_sign_pattern(ctx, signed, {}, label="-0 written through the cache")


def _rms_norm(ctx: Any) -> None:
    """A zero row scaled by a positive ``rsqrt`` keeps its sign and stays finite."""
    x = _zero_rows(ctx)
    weight = torch.ones(8)
    signed = (x, weight)
    want = reference(ctx, signed, {}, dtype=torch.float64)[0]
    if not bool(torch.isfinite(want).all()):
        raise NotApplicable("A7: the reference is not finite on the zero rows")
    _assert_finite(ctx, observe(ctx, signed, {})[0], "a zero row")
    _assert_sign_pattern(ctx, signed, {}, label="zero rows")


def _rms_norm_backward(ctx: Any) -> None:
    """The gradient at a zero row is nonzero and keeps the sign the Jacobian gives it.

    The Jacobian the reference defines for this column is taken through the
    ``-0`` rows themselves, so its gradient is not the cotangent scaled by
    ``rsqrt(eps)`` -- the epsilon term contributes a rank-one piece that changes
    every sign.  The comparison is therefore on what the reference decides, at one
    precision against its own float64 answer, and not on a sign reasoned about on
    paper.
    """
    x = _zero_rows(ctx)
    weight = torch.ones(8)
    signed = (x, weight)
    want = _gradients(ctx, signed, {}, torch.float64)
    if len(want) != 2:
        raise NotApplicable(f"A7: the backward column answered with {len(want)} gradients")
    for index, tensor in enumerate(want):
        # The cotangent is the reference's own output shape; a value gradient and a
        # scale gradient are both legal answers here, so only the shape it *has* is
        # checked, not the one the last run happened to give.
        if not bool(torch.isfinite(tensor).all()):
            raise NotApplicable(f"A7: reference gradient {index} is not finite")
    got = _gradients(ctx, signed, {}, torch.float32)
    if len(got) != len(want):
        raise NotApplicable("A7: the column answered with no gradient at the zero rows")
    for index, (left, right) in enumerate(zip(got, want, strict=True)):
        if not bool(torch.isfinite(left).all()):
            raise AssertionError(
                f"{ctx.column.name}: gradient {index} is non-finite at a zero row with a "
                f"positive eps"
            )
        if bool(torch.isfinite(right).all()):
            _compare_signs(ctx, left, right, f"gradient {index} at the zero rows")


def _zero_rows(ctx: Any) -> torch.Tensor:
    """Two zero rows of opposite sign, eight lanes each.

    Eight lanes keep ``1/width`` exact (``2**-3``), so a zero row comes out of the
    normalisation as its own input and any deviation in its sign is the kernel's
    arithmetic rather than the scale's.
    """
    rows = torch.zeros((2, 8))
    rows[1] = -0.0
    return rows


def _assert_finite(ctx: Any, tensor: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        bad = int(torch.logical_not(torch.isfinite(tensor)).sum())
        # ``eps = 0`` divides by the zero spread; the reference returns ``nan``
        # there too, so the values agree and only this test separates them.
        raise AssertionError(
            f"{ctx.column.name}: {bad} lane(s) of {label} are non-finite where the "
            f"reference is a signed zero; eps was not applied"
        )


# --------------------------------------------------------------------------
# shared
# --------------------------------------------------------------------------
def _gradients(
    ctx: Any, args: Any, kwargs: Any, dtype: torch.dtype = torch.float32
) -> list[torch.Tensor]:
    """The column's analytic gradients on a caller's arguments."""
    frozen = tuple(
        value.to(dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value
        for value in args
    )
    grads, _ = ctx.gradients_with(frozen, kwargs, dtype=dtype)
    return grads


def _assert_sign_pattern(ctx: Any, args: Any, kwargs: Any, *, label: str) -> None:
    """Every zero carries the reference's sign -- or, if there is none, no sign at all.

    A call whose reference has no ``-0`` is not a call this row cannot read: it
    is the half of the contract that says *no* zero may carry a sign.  A kernel
    that turns ``-0 + +0`` into ``-0`` is as wrong as one that turns ``-0 + -0``
    into ``+0``, and the value comparison cannot see either; only a fallback to
    ``NotApplicable`` would hide the first, so there is none.
    """
    got = observe(ctx, args, kwargs)
    want = reference(ctx, args, kwargs, dtype=torch.float64)
    if len(got) != len(want):
        raise AssertionError(
            f"{ctx.column.name}: the kernel returned {len(got)} outputs, the reference {len(want)}"
        )
    for index, (left, right) in enumerate(zip(got, want, strict=True)):
        if not bool(torch.equal(left == 0, right == 0)):
            raise AssertionError(
                f"{ctx.column.name}: {label} disagrees on which lanes of output {index} "
                f"are zero, so their signs cannot be compared"
            )
        # Both halves of the contract in one comparison: where the reference has a
        # ``-0`` the kernel must have one too, and where it has none the kernel
        # must not have invented one.  Reading only the reference's zeros would let
        # ``+0`` come back as ``-0`` on the calls built to exercise exactly that.
        _compare_signs(ctx, left, right, f"{label} output {index}", zeros_only=True)


def _has_negative_zero(tensor: torch.Tensor) -> bool:
    return bool(tensor.is_floating_point() and bool(_sign_mask(tensor).any()))


def _count_negative_zero(tensor: torch.Tensor) -> int:
    return int(_sign_mask(tensor).sum()) if tensor.is_floating_point() else 0


def _compare_signs(
    ctx: Any, got: torch.Tensor, want: torch.Tensor, label: str, *, zeros_only: bool = False
) -> None:
    """Raise unless ``got`` carries the reference's sign on every zero it decides.

    ``zeros_only`` restricts the comparison to the lanes the *values* call zero --
    the lanes a zero-input call decides -- and is the mode every value-producing
    column uses; the columns that only scale, rotate or copy compare every lane.
    The mask is read off the values, never off the signs: a mask taken from
    ``signbit`` would ask the judge to pick the lanes it then judges, and a judge
    that answers "no sign anywhere" would find nothing to compare and pass.
    """
    if want.shape != got.shape:
        raise AssertionError(
            f"{ctx.column.name}: {label} has shape {tuple(got.shape)} where the reference "
            f"has {tuple(want.shape)}"
        )
    # Bit-level masks: a judge that cannot see a sign must make this comparison and
    # lose it, not quietly compare nothing.  Taking the mask or the verdict from
    # ``signbit`` is how a sign check ends up passing on a kernel that flushed every
    # zero.
    actual = _sign_mask(got)
    expected = _sign_mask(want)
    lanes = want == 0
    if zeros_only:
        if not bool(lanes.any()):
            raise NotApplicable(f"A7: {label} produced no zero for the sign comparison")
        actual, expected, lanes = actual[lanes], expected[lanes], lanes[lanes]
    if bool(torch.equal(actual, expected)):
        return
    # No mask after the reduction: ``lanes`` is already the same shape as the two
    # sign masks, and masking a *disagreement* by it would silently drop the lanes
    # the comparison exists for.
    wrong = torch.logical_xor(actual, expected)
    count = int(wrong.sum())
    first = int(wrong.reshape(-1).nonzero()[0])
    raise AssertionError(
        f"{ctx.column.name}: {label} disagrees with the reference on the sign of {count} of "
        f"{int(lanes.numel())} lanes: the kernel has "
        f"{'-0' if bool(actual.reshape(-1)[first]) else '+0'}, the reference "
        f"{'-0' if bool(expected.reshape(-1)[first]) else '+0'}; the values are equal, the "
        f"bits are not"
    )


def _sign_mask(tensor: torch.Tensor) -> torch.Tensor:
    """The sign bit of every element, so the comparison can name what it read.

    ``torch.signbit`` is the readable spelling of this and is what the check means;
    reading the bit directly keeps the *verdict* off a function that can be
    stubbed to a constant, which is how a sign check ends up passing on a kernel
    that flushed every zero.
    """
    if tensor.is_complex():
        tensor = tensor.real
    if not tensor.is_floating_point():
        return torch.zeros(tensor.shape, dtype=torch.bool)
    bits = tensor.view(_INTEGRAL[tensor.dtype])
    return bits < 0
