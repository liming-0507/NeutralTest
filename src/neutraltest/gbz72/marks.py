"""Which of the 45 numbered features part 7.2 applies to which column.

The text of part 7.2 crosses the same 45-feature catalogue with twelve columns:
six operators, each listed forward and backward.  Not every feature applies to
every column -- a quantised round-trip needs a column that quantises, a
broadcast row needs a second operand -- and the clause does not say which is
which.  This table is that judgement, transcribed from the applicability
analysis kept outside the repository; it is data, not a derivation, and the
conformance run treats it as the declaration of record.

A column named here is exercised; a column not named is disclosed as not
applicable with the reason in :mod:`neutraltest.gbz72.reasons`.
"""

from __future__ import annotations

#: The twelve column labels, in the order the part 7.2 table prints them.
STANDARD_COLUMNS: tuple[str, ...] = (
    "Add-前",
    "FA-前",
    "SM-前",
    "RoPE-前",
    "KVC-前",
    "RMS-前",
    "Add-反",
    "FA-反",
    "SM-反",
    "RoPE-反",
    "KVC-反",
    "RMS-反",
)

#: The framework's own name for each column of the part 7.2 table.
COLUMN_NAMES: dict[str, str] = {
    "Add-前": "add_fwd",
    "FA-前": "flash_attention_fwd",
    "SM-前": "softmax_fwd",
    "RoPE-前": "rope_fwd",
    "KVC-前": "kv_cache_fwd",
    "RMS-前": "rms_norm_fwd",
    "Add-反": "add_bwd",
    "FA-反": "flash_attention_bwd",
    "SM-反": "softmax_bwd",
    "RoPE-反": "rope_bwd",
    "KVC-反": "kv_cache_bwd",
    "RMS-反": "rms_norm_bwd",
}

#: The twelve columns in the order the standard's table lists them: the six
#: forward columns, then the six backward ones. The framework's own column order
#: is family-major; the matrix is printed in the table's order so a reader can
#: compare it with the clause side by side.
TABLE_ORDER: tuple[str, ...] = tuple(COLUMN_NAMES[label] for label in STANDARD_COLUMNS)

#: Feature -> the columns it is exercised on, in table order.
APPLICABILITY: dict[str, tuple[str, ...]] = {
    "A1": (
        "add_fwd",
        "flash_attention_fwd",
        "softmax_fwd",
        "rope_fwd",
        "kv_cache_fwd",
        "rms_norm_fwd",
        "add_bwd",
        "flash_attention_bwd",
        "softmax_bwd",
        "rope_bwd",
        "kv_cache_bwd",
        "rms_norm_bwd",
    ),
    "A2": ("add_fwd", "kv_cache_fwd", "rms_norm_fwd"),
    "A3": (
        "add_fwd",
        "flash_attention_fwd",
        "softmax_fwd",
        "rope_fwd",
        "kv_cache_fwd",
        "rms_norm_fwd",
        "add_bwd",
        "flash_attention_bwd",
        "softmax_bwd",
        "rope_bwd",
        "kv_cache_bwd",
        "rms_norm_bwd",
    ),
    "A4": ("rope_fwd",),
    "A5": ("kv_cache_fwd",),
    "A6": ("add_fwd", "flash_attention_fwd", "softmax_fwd", "rope_fwd"),
    "A7": (
        "add_fwd",
        "softmax_fwd",
        "rope_fwd",
        "kv_cache_fwd",
        "rms_norm_fwd",
        "kv_cache_bwd",
        "rms_norm_bwd",
    ),
    "A8": ("rope_fwd",),
    "A9": (
        "add_fwd",
        "flash_attention_fwd",
        "softmax_fwd",
        "rope_fwd",
        "kv_cache_fwd",
        "rms_norm_fwd",
        "add_bwd",
        "flash_attention_bwd",
        "softmax_bwd",
        "rope_bwd",
        "kv_cache_bwd",
        "rms_norm_bwd",
    ),
    "A10": (
        "add_fwd",
        "flash_attention_fwd",
        "kv_cache_fwd",
        "rms_norm_fwd",
        "add_bwd",
        "flash_attention_bwd",
        "kv_cache_bwd",
        "rms_norm_bwd",
    ),
    "A11": ("rms_norm_fwd",),
    "A12": (),
    "B1": (),
    "B2": (),
    "B3": (),
    "B4": (
        "add_fwd",
        "flash_attention_fwd",
        "softmax_fwd",
        "rope_fwd",
        "kv_cache_fwd",
        "rms_norm_fwd",
        "add_bwd",
        "flash_attention_bwd",
        "softmax_bwd",
        "rope_bwd",
        "kv_cache_bwd",
        "rms_norm_bwd",
    ),
    "B5": (
        "add_fwd",
        "flash_attention_fwd",
        "softmax_fwd",
        "rope_fwd",
        "kv_cache_fwd",
        "rms_norm_fwd",
        "add_bwd",
        "flash_attention_bwd",
        "softmax_bwd",
        "rope_bwd",
        "kv_cache_bwd",
        "rms_norm_bwd",
    ),
    "B6": ("add_fwd", "kv_cache_fwd"),
    "B7": ("rms_norm_fwd",),
    "B8": (
        "add_fwd",
        "flash_attention_fwd",
        "softmax_fwd",
        "rope_fwd",
        "add_bwd",
        "flash_attention_bwd",
        "softmax_bwd",
        "rope_bwd",
    ),
    "B9": (
        "add_fwd",
        "flash_attention_fwd",
        "softmax_fwd",
        "rope_fwd",
        "kv_cache_fwd",
        "rms_norm_fwd",
        "add_bwd",
        "flash_attention_bwd",
        "softmax_bwd",
        "rope_bwd",
        "kv_cache_bwd",
        "rms_norm_bwd",
    ),
    "C1": (
        "add_fwd",
        "flash_attention_fwd",
        "softmax_fwd",
        "rope_fwd",
        "kv_cache_fwd",
        "rms_norm_fwd",
        "add_bwd",
        "flash_attention_bwd",
        "softmax_bwd",
        "rope_bwd",
        "kv_cache_bwd",
        "rms_norm_bwd",
    ),
    "C2": (),
    "C3": (
        "add_fwd",
        "flash_attention_fwd",
        "kv_cache_fwd",
        "rms_norm_fwd",
        "add_bwd",
        "flash_attention_bwd",
        "kv_cache_bwd",
    ),
    "C4": ("add_fwd", "kv_cache_fwd", "rms_norm_fwd"),
    "C5": ("add_fwd", "rms_norm_fwd"),
    "D1": (
        "add_fwd",
        "flash_attention_fwd",
        "softmax_fwd",
        "rope_fwd",
        "kv_cache_fwd",
        "rms_norm_fwd",
        "add_bwd",
        "flash_attention_bwd",
        "softmax_bwd",
        "rope_bwd",
        "kv_cache_bwd",
        "rms_norm_bwd",
    ),
    "D2": ("add_fwd", "rms_norm_fwd"),
    "D3": ("rope_fwd", "rope_bwd"),
    "D4": ("add_fwd", "flash_attention_fwd", "softmax_fwd", "rope_fwd"),
    "D5": (),
    #: Error handling is exercised where the kernel is handed values to refuse.
    #: A backward is handed derivatives: nothing in its call is wrong only by
    #: being the wrong shape, so the forward column of the same operator takes
    #: the refusal and the backward column records why it carries none.  The add
    #: is left to the framework's own contract (D1's arity check) because its
    #: kernels cast and broadcast whatever width or type they are given.  The
    #: cache read keeps the row: though it is the backward column, its kernel
    #: reads a slot list and indexes with it, and it refuses both.
    "D6": (
        "flash_attention_fwd",
        "softmax_fwd",
        "rope_fwd",
        "kv_cache_fwd",
        "kv_cache_bwd",
        "rms_norm_fwd",
    ),
    #: Determinism is a property of every column: each is a function of its
    #: arguments, and E1 is what establishes that rather than assuming it.
    "E1": TABLE_ORDER,
    #: Likewise the random-number behaviour: a column either draws or is shown
    #: not to, and E2 states which for every column.
    "E2": TABLE_ORDER,
    "E3": (),
    "E4": ("flash_attention_fwd", "rope_fwd", "flash_attention_bwd", "rope_bwd"),
    "E5": ("rope_fwd",),
    "E6": (),
    "E7": (),
    "F1": (
        "add_fwd",
        "flash_attention_fwd",
        "softmax_fwd",
        "rope_fwd",
        "kv_cache_fwd",
        "rms_norm_fwd",
        "add_bwd",
        "flash_attention_bwd",
        "softmax_bwd",
        "rope_bwd",
        "kv_cache_bwd",
        "rms_norm_bwd",
    ),
    "F2": (
        "add_fwd",
        "flash_attention_fwd",
        "softmax_fwd",
        "rope_fwd",
        "kv_cache_fwd",
        "rms_norm_fwd",
        "add_bwd",
        "flash_attention_bwd",
        "softmax_bwd",
        "rope_bwd",
        "kv_cache_bwd",
        "rms_norm_bwd",
    ),
    "F3": ("add_fwd", "flash_attention_fwd", "softmax_fwd", "kv_cache_fwd"),
    "F4": (),
    "F5": (
        "add_fwd",
        "flash_attention_fwd",
        "softmax_fwd",
        "rope_fwd",
        "kv_cache_fwd",
        "rms_norm_fwd",
        "add_bwd",
        "flash_attention_bwd",
        "softmax_bwd",
        "rope_bwd",
        "kv_cache_bwd",
        "rms_norm_bwd",
    ),
    "F6": (),
    #: The catalogue gained this row after the part 7.2 table was drawn up, so the
    #: table does not list it at all and no column is claimed for it.
    "F7": (),
}

#: Number of marked (feature, column) pairs; the coverage denominator.
MARKED = 228


def columns_of(feature_id: str) -> tuple[str, ...]:
    """The columns ``feature_id`` is exercised on (possibly empty)."""
    try:
        return APPLICABILITY[feature_id]
    except KeyError as exc:  # pragma: no cover - guarded by the catalogue
        raise KeyError(f"feature {feature_id} is not in the part 7.2 table") from exc


def is_marked(feature_id: str, column: str) -> bool:
    return column in columns_of(feature_id)


def marked_pairs() -> tuple[tuple[str, str], ...]:
    """Every marked pair, in catalogue order then table order."""
    return tuple(
        (feature_id, column) for feature_id, columns in APPLICABILITY.items() for column in columns
    )


def unmarked_features() -> tuple[str, ...]:
    """Features the clause does not apply to any of the twelve columns."""
    return tuple(fid for fid, columns in APPLICABILITY.items() if not columns)
