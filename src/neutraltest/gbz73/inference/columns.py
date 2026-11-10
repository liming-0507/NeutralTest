"""The twelve operator columns of the inference clause, as data.

Six operators, each tested twice: once as a forward kernel (what an inference
deployment runs) and once as the backward of that same kernel (what the training
job needs from it). The distinction is not cosmetic -- several of these kernels
are inference-only by construction, and the backward column is exactly where
that shows up.

Three things are attached to every column and used by the test scripts:

``family``
    the kernel family, which selects the analytic Jacobian. Two columns of the
    same family share a Jacobian: the backward column is *not* given the
    forward's derivative, it is given the operator to differentiate, so that a
    candidate that ships the right forward and the wrong backward is caught.
``bits``
    a two-part coverage string: ``forward`` bits come from the standard's own
    coverage table, ``backward`` bits are the same table read against the
    "must also work in training" rule, and ``"2"`` marks a feature that is
    structurally out of scope (a cache has no quantisation of its own).
``freezes``
    the inputs a Jacobian holds fixed instead of differentiating through them --
    routing metadata is not a parameter, and a cache slot that is written is a
    destination, not a value.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Families, in the order the clause lists the operators. Grouped query
#: attention and masked attention are one family: both are attention with a
#: different way of addressing the same softmax.
FAMILIES: tuple[str, ...] = (
    "add",
    "flash_attention",
    "kv_cache",
    "softmax",
    "rope",
    "rms_norm",
)

#: Which operators the clause names, per family, with their forward/backward split.
_COLUMNS: tuple[tuple[str, str, str, str], ...] = (
    # (column, family, forward kernel, backward of)
    ("add", "add", "elementwise residual add", "the forward add"),
    (
        "flash_attention",
        "flash_attention",
        "scaled dot-product attention",
        "the forward attention",
    ),
    ("softmax", "softmax", "numerically stable softmax", "the forward softmax"),
    ("rope", "rope", "rotary position embedding", "the forward rotation"),
    (
        "kv_cache",
        "kv_cache",
        "paged cache write/read",
        "the addressing, not a kernel",
    ),
    ("rms_norm", "rms_norm", "root-mean-square normalisation", "the forward normalisation"),
)

#: Columns that are tested as a forward kernel (inference deployment shape).
FORWARD_COLUMNS: tuple[str, ...] = tuple(f"{name}_fwd" for name, *_ in _COLUMNS)

#: Columns that are tested as the backward of the same kernel (training shape).
BACKWARD_COLUMNS: tuple[str, ...] = tuple(f"{name}_bwd" for name, *_ in _COLUMNS)

#: Column order used by the matrix.
COLUMNS: tuple[str, ...] = FORWARD_COLUMNS + BACKWARD_COLUMNS


def family_of(column: str) -> str:
    """The kernel family a column belongs to."""
    if column not in COLUMNS:
        raise KeyError(f"unknown column {column!r}; known: {', '.join(COLUMNS)}")
    return column.rsplit("_", 1)[0]


def is_backward(column: str) -> bool:
    return column.endswith("_bwd")


@dataclass(frozen=True, slots=True)
class Column:
    """One tested column: a forward kernel or its backward."""

    name: str
    family: str
    description: str
    quadratic: bool
    freezes: tuple[str, ...]
    notes: str

    @property
    def backward(self) -> bool:
        return is_backward(self.name)

    @property
    def base(self) -> str:
        """The operator name without the direction suffix."""
        return self.name.rsplit("_", 1)[0]


_SHARED: dict[str, tuple[bool, tuple[str, ...], str]] = {
    # family -> (quadratic in the sequence length, frozen inputs, note)
    "add": (
        False,
        ("alpha",),
        "alpha is a scalar hyper-parameter; the gradients are w.r.t. both tensors",
    ),
    "flash_attention": (
        True,
        (),
        "attention cost is quadratic in the sequence length before any masking",
    ),
    "kv_cache": (
        False,
        ("index",),
        "the slot index is routing metadata, not a value to differentiate",
    ),
    "softmax": (False, ("dim",), "dim is a layout option"),
    "rope": (
        False,
        ("position", "interleaved"),
        "position is data-dependent routing (the rotation angles), not a parameter",
    ),
    "rms_norm": (
        False,
        ("eps", "reduction_axis"),
        "eps is a numerical guard; the reduction axis is part of the call contract",
    ),
}

COLUMN_SPECS: tuple[Column, ...] = tuple(
    Column(
        name=column,
        family=family,
        description=(
            f"backward of {backward_of} ({kernel})" if column.endswith("_bwd") else kernel
        ),
        quadratic=_SHARED[family][0],
        freezes=_SHARED[family][1],
        notes=_SHARED[family][2],
    )
    for base, family, kernel, backward_of in _COLUMNS
    for column in (f"{base}_fwd", f"{base}_bwd")
)

BY_NAME: dict[str, Column] = {spec.name: spec for spec in COLUMN_SPECS}


def column(name: str) -> Column:
    try:
        return BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown column {name!r}; known: {', '.join(BY_NAME)}") from None


#: Feature ids that need a Jacobian at all. Everything else inspects the forward.
JACOBIAN_FEATURES: frozenset[str] = frozenset({"A1", "A8", "A9", "A10", "A12"})
