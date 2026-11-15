"""The sampler's reference, and the two different oracles a draw needs.

Two properties are tested, and they answer two different questions.

``exact``
    *Which* token was drawn. A draw is only comparable to another draw when the
    device's generator is the reference's generator -- it is not. So this is
    asserted only on the CPU backend, where the kernel *is* the reference, and
    it exists to keep the kernel honest about ``torch.multinomial``'s contract
    (returns indices, respects the filtered support, replaces when asked).

``mass``
    *What probability each token had.* The filtered distribution is deterministic
    even though the draw is not, so every backend can be held to it: run the
    kernel many times with different generator states and compare the empirical
    frequency of each token against the reference's probability. This is the
    oracle that survives on a device, and it is the one that fails a kernel
    which quietly returns ``argmax`` (mass 1.0 on one token, 0 elsewhere) or
    forgets to apply the Top-P cutoff (mass leaks to the tail).

The two are kept apart on purpose. A fixed seed makes a device draw *repeatable*
but not *equal to the reference* -- two samplers consuming one generator state
differently both stay correct -- so agreeing draw-for-draw would be testing the
generator's implementation, not the operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

Args = tuple[Any, ...]
Kwargs = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Sampling:
    """One deterministic call of the sampler: logits in, a draw out."""

    args: Args
    kwargs: Kwargs


@dataclass(frozen=True, slots=True)
class KERNELS:
    """The sampler's kernel surface, named once and used by every test.

    ``top_k`` and ``top_p`` are the two filters; ``temperature`` is applied to
    the logits before either. Every one of them is optional, and passing neither
    is ordinary multinomial sampling -- which is what makes ``Top-K`` and
    ``Top-P`` testable as two settings of one operator rather than two operators.
    """

    # The keyword names are the reference's own: a kernel that spells them
    # differently is a different call, and D1 catches that on the other six
    # operators without this file needing to.
    TOP_K = "top_k"
    TOP_P = "top_p"
    TEMPERATURE = "temperature"


def filter_logits(
    logits: torch.Tensor,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float | None = None,
) -> torch.Tensor:
    """The sampling distribution itself: filter, then renormalise.

    Written as the plain definition rather than in log space, because the
    interesting failures here are boundary ones (``top_p`` exactly reaching 1.0,
    ``top_k`` larger than the vocabulary) and log-space arithmetic hides them.

    Top-K keeps the ``k`` largest logits. Top-P keeps the smallest prefix of the
    descending order whose mass reaches ``p`` -- the token that crosses the
    threshold is kept, which is the convention ``torch``'s own generators use
    when given a truncated distribution. Both filters are applied when both are
    given, Top-K first, which is the order a decoding loop applies them in.
    """
    if temperature is not None:
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        logits = logits / temperature

    if top_k is not None:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        k = min(int(top_k), logits.shape[-1])
        threshold = torch.topk(logits, k, dim=-1).values[..., -1:]
        # ``>=`` keeps every tie at the boundary: dropping one of two equal
        # logits would make the result depend on the sort's tie-breaking.
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p is not None:
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        sorted_logits, order = torch.sort(logits, dim=-1, descending=True)
        mass = torch.softmax(sorted_logits, dim=-1)
        # The mass that precedes each token in descending order. A token is
        # kept when that prefix had not yet reached the threshold, so the
        # crossing token itself stays in the support.
        before = mass.cumsum(dim=-1) - mass
        keep_sorted = before < top_p
        # Scatter back into the original vocabulary order: ``order`` maps a
        # sorted position to the token it came from, so ``keep[order]`` would
        # read the wrong direction -- the assignment must be scattered, not
        # gathered.
        keep = torch.zeros_like(keep_sorted)
        keep.scatter_(-1, order, keep_sorted)
        logits = logits.masked_fill(~keep, float("-inf"))

    return torch.softmax(logits, dim=-1)


def reference_mass(
    logits: torch.Tensor,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float | None = None,
) -> torch.Tensor:
    """The probability of every token after filtering -- the device oracle."""
    return filter_logits(logits, top_k=top_k, top_p=top_p, temperature=temperature)


def sample_indices(
    logits: torch.Tensor,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float | None = None,
    generator: torch.Generator | None = None,
    num_samples: int = 1,
) -> torch.Tensor:
    """One draw, as indices -- the reference implementation of the kernel.

    One distribution per leading index: row ``i`` of a 2-D logits tensor draws
    from row ``i`` of the filtered probabilities and may repeat tokens, since a
    decoder draws with replacement. ``torch.multinomial`` over the whole
    flattened vector would draw across rows instead.
    """
    probs = reference_mass(logits, top_k=top_k, top_p=top_p, temperature=temperature)
    rows = probs.reshape(-1, probs.shape[-1])
    picks = torch.stack(
        [torch.multinomial(row, num_samples, replacement=True, generator=generator) for row in rows]
    )
    return picks.reshape(*probs.shape[:-1], num_samples)


def draw(
    logits: torch.Tensor,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float | None = None,
    seed: int = 0,
    num_samples: int = 1,
) -> torch.Tensor:
    """``sample_indices`` with the generator built from ``seed``."""
    generator = torch.Generator().manual_seed(seed)
    return sample_indices(
        logits,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        generator=generator,
        num_samples=num_samples,
    )


def supports(name: str, logits: torch.Tensor, **kwargs: Any) -> bool:
    """Whether a filter leaves at least one token with mass -- a sanity gate."""
    try:
        probs = filter_logits(logits, **{name: kwargs.get(name)})
    except (ValueError, RuntimeError):
        return False
    return bool(torch.isfinite(probs).all()) and float(probs.sum()) > 0.0
