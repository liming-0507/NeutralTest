"""The oracle a drawn index needs: the distribution, not the draw.

A sampler is the one operator here whose output is not a function of its input
in any usable sense. Two correct samplers given the same logits produce
different indices, so ``assert_close(device, reference)`` has nothing to compare
and a fixed seed does not rescue it: a seed makes a device repeatable, not equal
to the reference.

So the claim tested is the one a sampler actually makes -- *the probability it
draws each token with*. :func:`mass` recovers that claim empirically by drawing
many times and counting, and compares the counts against
:func:`..ops.reference_mass`. It is a two-sided test: a kernel returning
``argmax`` (all mass on one token) fails it, and so does a kernel that drops the
Top-P cutoff (mass leaking to the tail).

The tolerance is a binomial interval rather than a fixed epsilon. With ``N``
draws at probability ``p`` the standard error of the frequency is
``sqrt(p(1-p)/N)``; the bound below allows six of those plus a floor so that a
token with small mass cannot fail on rounding alone. That is a statement about
sampling noise, which is the whole point: an epsilon small enough to catch a
real defect would fail correct samplers at random.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .ops import reference_mass

#: Draws per batch. Large enough that a 0.5-mass token's frequency settles to a
#: few percent, small enough that the device run is still a test and not a job.
DRAWS = 4096

#: Standard errors allowed on a frequency. Six sigma is far beyond what noise
#: produces in a passing kernel, and still tight enough that a kernel which
#: moves a tenth of the mass to the wrong token fails.
SIGMA = 6.0

#: Floor on the allowance, so a token the reference gives 0.002 mass is not
#: judged at that scale by 4096 draws.
FLOOR = 0.01


def draw_many(
    draw: Any,
    logits: torch.Tensor,
    *,
    draws: int = DRAWS,
    seed: int = 0,
    **kwargs: Any,
) -> torch.Tensor:
    """``draws`` sample sets, stacked: shape ``(draws, *leading, num_samples)``.

    Each call gets its own generator state, so the count is over independent
    draws rather than one repeated draw -- a kernel that ignores the generator
    and returns a constant would pass a repeated-draw count and fails this one.
    """
    picks = [draw(logits, seed=seed + index, num_samples=1, **kwargs) for index in range(draws)]
    return torch.stack([p.reshape(-1) for p in picks])


def frequency(
    draw: Any,
    logits: torch.Tensor,
    *,
    draws: int = DRAWS,
    seed: int = 0,
    **kwargs: Any,
) -> torch.Tensor:
    """Empirical probability of each token: shape ``(rows, vocabulary)``."""
    picks = draw_many(draw, logits, draws=draws, seed=seed, **kwargs)
    rows = logits.shape[0]
    vocabulary = logits.shape[-1]
    # ``draw_many`` stacks one draw of every row per call: (draws, rows).
    picks = picks.reshape(draws, rows)
    counts = torch.zeros(rows, vocabulary, dtype=torch.float64)
    for index in range(rows):
        counts[index] = torch.bincount(picks[:, index].to(torch.int64), minlength=vocabulary).to(
            torch.float64
        )
    return counts / draws


def allowance(mass: torch.Tensor, draws: int = DRAWS) -> torch.Tensor:
    """The binomial bound on a frequency of ``mass``, per token."""
    deviation = torch.sqrt(torch.clamp(mass * (1.0 - mass), min=0.0) / draws)
    return torch.clamp(SIGMA * deviation, min=FLOOR)


def compare(
    draw: Any,
    logits: torch.Tensor,
    *,
    draws: int = DRAWS,
    seed: int = 0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(measured, expected, worst excess)`` for the distribution claim.

    ``worst excess`` is the largest amount by which a measured frequency
    overshot the reference's probability beyond its allowance. It is the number
    the report quotes, and it is zero for a kernel that draws correctly.
    """
    expected = reference_mass(logits, **_filter_of(kwargs))
    measured = frequency(draw, logits, draws=draws, seed=seed, **kwargs)
    excess = measured - expected - allowance(expected, draws)
    return measured, expected, excess


def assert_matches(
    draw: Any,
    logits: torch.Tensor,
    *,
    draws: int = DRAWS,
    seed: int = 0,
    msg: str = "",
    **kwargs: Any,
) -> float:
    """Raise unless the measured distribution matches; return the worst excess."""
    measured, expected, excess = compare(draw, logits, draws=draws, seed=seed, **kwargs)
    worst = float(excess.max())
    if worst > 0.0:
        index = int(excess.reshape(-1).argmax())
        token = index % logits.shape[-1]
        row = index // logits.shape[-1]
        raise AssertionError(
            f"{msg or 'sampler'}: token {token} of row {row} was drawn "
            f"{float(measured[row, token]):.3f} of the time against a reference "
            f"of {float(expected[row, token]):.3f} "
            f"(allowed drift {float(allowance(expected)[row, token]):.3f}, "
            f"{draws} draws)"
        )
    return worst


def _filter_of(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The keyword arguments that shape the distribution (not the draw)."""
    return {key: value for key, value in kwargs.items() if key in {"top_k", "top_p", "temperature"}}


def max_total_variation(measured: torch.Tensor, expected: torch.Tensor) -> float:
    """Half the L1 distance between the two distributions, as a summary number."""
    return float(((measured - expected).abs().sum(-1) / 2).max())


def standard_error(draws: int = DRAWS) -> float:
    """The worst-case standard error of a frequency at ``draws`` draws."""
    return math.sqrt(0.25 / draws)
