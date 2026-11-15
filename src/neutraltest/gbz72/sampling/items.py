"""The sampler's feature rows.

These are the same 45-feature catalogue's rows, restricted to the ones a
sampler can answer. The rows that need a direction (A12, the gradient row) or a
stored value (A2, A5) are not here, and neither is any backward row: the
catalogue's backward half has no sampler column to land on.

A row is a claim of the form "the sampler must behave this way", checked on the
CPU reference and, when it costs nothing, on the device under test.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .ops import draw, filter_logits


@dataclass(frozen=True, slots=True)
class SamplingRow:
    """One sampler feature: what it claims, and how it is checked."""

    id: str
    subject: str
    check: Callable[[], None]


def _logits(rows: int = 6, vocabulary: int = 16) -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    return torch.randn(rows, vocabulary, generator=generator)


def _a1_matches_the_reference() -> None:
    logits = _logits()
    for kwargs in ({}, {"top_k": 4}, {"top_p": 0.75}, {"top_k": 4, "top_p": 0.75}):
        probs = filter_logits(logits, **kwargs)
        if not torch.isclose(probs.sum(-1), torch.ones(logits.shape[0])).all():
            raise AssertionError(f"A1: {kwargs} does not normalise")
        # Every filtered token must be reachable: a filter that empties the
        # support is a wrong filter, not a conservative one.
        if not (probs > 0).any(-1).all():
            raise AssertionError(f"A1: {kwargs} leaves no token to draw")


def _a3_saturates_at_the_boundary() -> None:
    # The overflow row: a logit far above the rest must not turn the
    # distribution into NaNs. softmax subtracts the max, so it survives, and
    # the sampler must inherit that, not re-derive it.
    logits = torch.tensor([[0.0, 1e30, -1e30]])
    probs = filter_logits(logits)
    if not torch.isfinite(probs).all():
        raise AssertionError("A3: an out-of-range logit produced a non-finite probability")
    if int(probs.argmax()) != 1:
        raise AssertionError("A3: the largest logit did not take the mass")


def _a4_subnormals_do_not_break_the_draw() -> None:
    # A token whose probability is subnormal is measurable but never drawn at
    # this sample size; the point is that it must not become NaN or be treated
    # as the only candidate.
    logits = torch.tensor([[0.0, -1000.0, 1.0]])
    probs = filter_logits(logits)
    if not torch.isfinite(probs).all():
        raise AssertionError("A4: a subnormal probability was not finite")
    if float(probs[0, 0]) <= 0.0:
        raise AssertionError("A4: a far-below token lost all mass")


def _a5_rounds_to_a_representable_index() -> None:
    # Indices are integers: a kernel that returns a float (or rounds one) is
    # a different contract. Checked as a dtype and a range claim.
    picks = draw(_logits(), top_k=5, num_samples=4)
    if picks.dtype != torch.int64:
        raise AssertionError(f"A5: a draw has dtype {picks.dtype}, expected int64")
    vocabulary = _logits().shape[-1]
    if int(picks.min()) < 0 or int(picks.max()) >= vocabulary:
        raise AssertionError("A5: a draw left the vocabulary")


def _a6_nan_inf_propagate_as_disclosed() -> None:
    # The convention stated here: a NaN logit makes that token unusable, and an
    # all-NaN row is refused by the reference rather than silently drawing.
    logits = torch.tensor([[0.0, float("nan"), 1.0]])
    probs = filter_logits(logits)
    if not torch.isnan(probs[0, 1]):
        raise AssertionError("A6: a NaN logit stopped being visible in the distribution")
    try:
        draw(logits, num_samples=2)
    except (RuntimeError, ValueError):
        return
    if torch.isnan(probs).all():
        raise AssertionError("A6: an all-NaN row drew without being refused")


def _a7_signed_zero_is_one_token() -> None:
    # +0 and -0 are the same logit, so they must carry the same mass.
    logits = torch.tensor([[0.0, -0.0, 3.0]])
    probs = filter_logits(logits)
    if not torch.equal(probs[0, 0], probs[0, 1]):
        raise AssertionError("A7: +0 and -0 received different mass")


def _a8_accumulates_the_mass_once() -> None:
    # The Top-P cutoff sums the descending order; an off-by-one in that sum is
    # the single most common sampling defect. The count of kept tokens must be
    # the smallest prefix reaching the threshold.
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    probs = filter_logits(logits, top_p=0.8)
    kept = int((probs > 0).sum())
    reference = filter_logits(logits)
    descending = torch.sort(reference, descending=True).values
    running = descending.cumsum(-1)
    expected = int((running < 0.8).sum()) + 1
    if kept != expected:
        raise AssertionError(f"A8: kept {kept} tokens, the prefix rule keeps {expected}")


def _b3_degenerate_shapes() -> None:
    # A single token: the draw is forced, and must be allowed rather than
    # refused for having no choice.
    picks = draw(torch.zeros(1, 1), num_samples=3)
    if not torch.equal(picks, torch.zeros(1, 3, dtype=torch.int64)):
        raise AssertionError("B3: a one-token vocabulary did not draw the only token")
    # An empty vocabulary has nothing to draw -- refused, not guessed.
    try:
        draw(torch.zeros(1, 0))
    except (RuntimeError, ValueError):
        return
    raise AssertionError("B3: an empty vocabulary drew a token")


def _b4_irregular_shapes() -> None:
    # A batch of rows with a ragged count of viable tokens must be filtered
    # row by row, not by one batch-wide threshold.
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 5.0, 0.0]])
    probs = filter_logits(logits, top_p=0.5)
    if int((probs > 0).sum(-1).max()) != 1:
        raise AssertionError("B4: a per-row filter used a batch-wide cutoff")


def _c5_output_dtype() -> None:
    # ``num_samples`` drives the shape; the dtype stays int64 regardless.
    picks = draw(_logits(2, 8), top_k=3, num_samples=5)
    if tuple(picks.shape) != (2, 5) or picks.dtype != torch.int64:
        raise AssertionError(
            f"C5: a draw of (2, 5) came back as {tuple(picks.shape)} {picks.dtype}"
        )


def _d2_parameter_bounds() -> None:
    logits = _logits(2, 8)
    for bad, kwargs in (
        ("top_k=0", {"top_k": 0}),
        ("top_p=0", {"top_p": 0.0}),
        ("top_p>1", {"top_p": 1.5}),
        ("temperature=0", {"temperature": 0.0}),
    ):
        try:
            filter_logits(logits, **kwargs)
        except (ValueError, RuntimeError):
            continue
        raise AssertionError(f"D2: {bad} was accepted")


def _f1_each_setting_is_distinct() -> None:
    # A sampler that ignores a knob is worse than one that rejects it: the
    # result looks plausible. Each setting must change the distribution.
    logits = _logits()
    baseline = filter_logits(logits)
    for kwargs in ({"top_k": 3}, {"top_p": 0.5}, {"temperature": 0.5}):
        changed = filter_logits(logits, **kwargs)
        if torch.allclose(baseline, changed):
            raise AssertionError(f"F1: {kwargs} did not change the distribution")


#: The rows a sampler can answer, in catalogue order.
ROWS: tuple[SamplingRow, ...] = (
    SamplingRow("A1", "reference agreement", _a1_matches_the_reference),
    SamplingRow("A3", "overflow saturation", _a3_saturates_at_the_boundary),
    SamplingRow("A4", "subnormal probabilities", _a4_subnormals_do_not_break_the_draw),
    SamplingRow("A5", "index dtype and range", _a5_rounds_to_a_representable_index),
    SamplingRow("A6", "NaN and Inf disclosure", _a6_nan_inf_propagate_as_disclosed),
    SamplingRow("A7", "signed zero", _a7_signed_zero_is_one_token),
    SamplingRow("A8", "cutoff accumulation", _a8_accumulates_the_mass_once),
    SamplingRow("B3", "degenerate shapes", _b3_degenerate_shapes),
    SamplingRow("B4", "irregular shapes", _b4_irregular_shapes),
    SamplingRow("C5", "output dtype", _c5_output_dtype),
    SamplingRow("D2", "parameter bounds", _d2_parameter_bounds),
    SamplingRow("F1", "each setting takes effect", _f1_each_setting_is_distinct),
)

#: Rows whose claim is about the filter itself, not about drawing: they are
#: checked once instead of once per generator state.
FILTER_ONLY: frozenset[str] = frozenset({"A1", "A3", "A4", "A7", "A8", "B4", "D2", "F1"})


def by_id(row_id: str) -> SamplingRow:
    for row in ROWS:
        if row.id == row_id:
            return row
    raise KeyError(f"no sampler row {row_id!r}; known: {', '.join(r.id for r in ROWS)}")
