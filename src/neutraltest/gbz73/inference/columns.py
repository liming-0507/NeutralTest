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
    the standard's coverage string for the column: one character per feature of
    the part 7.2 catalogue, in catalogue order, ``1`` where the text states a
    requirement for this column and ``0`` where it does not. It is the
    applicability column of the matrix (:func:`neutraltest.gbz72.marks.is_marked`)
    read down instead of across, and it is written out rather than derived so
    that a run can print it beside the result it produced -- "the clause asks
    for this" and "we measured this" are two different claims, and the
    interesting cells are the ones where only one of them is true.
``freezes``
    the inputs a Jacobian holds fixed instead of differentiating through them --
    routing metadata is not a parameter, and a cache slot that is written is a
    destination, not a value.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..features import FEATURES

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
    #: One bit per part 7.2 feature, in catalogue order; see the module docstring.
    bits: str = ""

    def __post_init__(self) -> None:
        if self.bits and len(self.bits) != len(FEATURE_ORDER):
            raise ValueError(
                f"{self.name}: expected {len(FEATURE_ORDER)} coverage bits, got {len(self.bits)!r}"
            )
        if set(self.bits) - {"0", "1"}:
            raise ValueError(f"{self.name}: coverage bits must be 0 or 1, got {self.bits!r}")

    @property
    def covered_by_standard(self) -> str:
        """The bits, for a reader that wants the column's coverage as data."""
        return self.bits

    def standard_covers(self, feature_id: str) -> bool:
        """Whether the standard's text states a requirement for ``feature_id`` here."""
        return self.bits[FEATURE_ORDER.index(feature_id)] == "1"

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

#: The catalogue order the coverage bits are written in: the 46 features of the
#: standard, as :mod:`neutraltest.gbz73.features` lists them. A column's ``bits``
#: has one character per entry here.
FEATURE_ORDER: tuple[str, ...] = tuple(feature.id for feature in FEATURES)

#: The catalogue as the standard prints it: feature -> the twelve columns that
#: row names, in :data:`neutraltest.gbz72.marks.TABLE_ORDER`. This is the source
#: form -- the standard is a table printed row by row -- and a ``Column.bits``
#: string is one character of every row, read down. The two are checked against
#: each other in both directions by ``tests/unit/test_gbz72_columns_bits.py``,
#: so neither can drift without the other being wrong too.
_STANDARD_TABLE: dict[str, str] = {
    "A1": "111111111111",
    "A2": "100011000000",
    "A3": "111111111111",
    "A4": "000100000000",
    "A5": "000010000000",
    "A6": "111100000000",
    "A7": "101111000011",
    "A8": "000100000000",
    "A9": "111111111111",
    "A10": "110011110011",
    "A11": "000001000000",
    "A12": "000000000000",
    "B1": "000000000000",
    "B2": "000000000000",
    "B3": "000000000000",
    "B4": "111111111111",
    "B5": "111111111111",
    "B6": "100010000000",
    "B7": "000001000000",
    "B8": "111100111100",
    "B9": "111111111111",
    "C1": "111111111111",
    "C2": "000000000000",
    "C3": "110011110010",
    "C4": "100011000000",
    "C5": "100001000000",
    "D1": "111111111111",
    "D2": "100001000000",
    "D3": "000100000100",
    "D4": "111100000000",
    "D5": "000000000000",
    "D6": "011111000010",
    "E1": "111111111111",
    "E2": "111111111111",
    "E3": "000000000000",
    "E4": "010100010100",
    "E5": "000100000000",
    "E6": "000000000000",
    "E7": "000000000000",
    "F1": "111111111111",
    "F2": "111111111111",
    "F3": "111010000000",
    "F4": "000000000000",
    "F5": "111111111111",
    "F6": "000000000000",
    "F7": "000000000000",
}

#: Column -> coverage bits: ``_STANDARD_TABLE`` read down, one column at a time.
#: Written out rather than computed because it is a transcription of a text, not
#: a derivation from code: a reader can compare these strings with the standard
#: without running anything, and the matrix prints them beside the results.
_BITS: dict[str, str] = {
    "add_fwd": (
        "111001101100"  # A1-A12
        "000111011101"  # B1-B9, C1-C5
        "111101001100"  # D1-D6, E1-E7, F1-F2
        "0001110100"  # F3-F7
    ),
    "flash_attention_fwd": (
        "101001001100"  # A1-A12
        "000110011101"  # B1-B9, C1-C5
        "001001011101"  # D1-D6, E1-E7, F1-F2
        "0001110100"  # F3-F7
    ),
    "softmax_fwd": (
        "101001101000"  # A1-A12
        "000110011100"  # B1-B9, C1-C5
        "001001011100"  # D1-D6, E1-E7, F1-F2
        "0001110100"  # F3-F7
    ),
    "rope_fwd": (
        "101101111000"  # A1-A12
        "000110011100"  # B1-B9, C1-C5
        "001011011101"  # D1-D6, E1-E7, F1-F2
        "1001100100"  # F3-F7
    ),
    "kv_cache_fwd": (
        "111010101100"  # A1-A12
        "000111001101"  # B1-B9, C1-C5
        "101000011100"  # D1-D6, E1-E7, F1-F2
        "0001110100"  # F3-F7
    ),
    "rms_norm_fwd": (
        "111000101110"  # A1-A12
        "000110101101"  # B1-B9, C1-C5
        "111100011100"  # D1-D6, E1-E7, F1-F2
        "0001100100"  # F3-F7
    ),
    "add_bwd": (
        "101000001100"  # A1-A12
        "000110011101"  # B1-B9, C1-C5
        "001000001100"  # D1-D6, E1-E7, F1-F2
        "0001100100"  # F3-F7
    ),
    "flash_attention_bwd": (
        "101000001100"  # A1-A12
        "000110011101"  # B1-B9, C1-C5
        "001000001101"  # D1-D6, E1-E7, F1-F2
        "0001100100"  # F3-F7
    ),
    "softmax_bwd": (
        "101000001000"  # A1-A12
        "000110011100"  # B1-B9, C1-C5
        "001000001100"  # D1-D6, E1-E7, F1-F2
        "0001100100"  # F3-F7
    ),
    "rope_bwd": (
        "101000001000"  # A1-A12
        "000110011100"  # B1-B9, C1-C5
        "001010001101"  # D1-D6, E1-E7, F1-F2
        "0001100100"  # F3-F7
    ),
    "kv_cache_bwd": (
        "101000101100"  # A1-A12
        "000110001101"  # B1-B9, C1-C5
        "001000011100"  # D1-D6, E1-E7, F1-F2
        "0001100100"  # F3-F7
    ),
    "rms_norm_bwd": (
        "101000101100"  # A1-A12
        "000110001100"  # B1-B9, C1-C5
        "001000001100"  # D1-D6, E1-E7, F1-F2
        "0001100100"  # F3-F7
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
        bits=_BITS[column],
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
