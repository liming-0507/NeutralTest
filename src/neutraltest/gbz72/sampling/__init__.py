"""The seventh operator of part 7.2: Top-K / Top-P sampling.

The clause names seven inference operators. Six of them are *columns* of the
feature matrix -- an operator crossed with the direction it is run in. The
seventh is not a column, and the reason is structural rather than editorial:

**A sampler has no backward.** Sampling draws from a distribution; the draw is
not differentiable with respect to the logits, and no framework ships a gradient
for it. A backward column would not be "untested", it would be meaningless, so
this operator gets its own forward-only matrix instead of a twelfth-and-a-
thirteenth column that could never pass.

It is still the same kind of object as the other six, and is tested the same
way: against a reference written in plain ``torch`` on the CPU, under the same
dtype budget, with everything not exercised disclosed by name.

What is *not* the same is the verdict. A sampler returns *indices*, and two
correct samplers disagree on any given draw. The oracle therefore has to be a
property of the distribution rather than of one sample; :mod:`neutraltest.gbz72
.sampling.ops` says which property, and why a fixed seed cannot rescue a
per-element comparison.
"""

from __future__ import annotations

from .ops import (
    KERNELS,
    Sampling,
    draw,
    filter_logits,
    reference_mass,
    sample_indices,
    supports,
)

__all__ = [
    "KERNELS",
    "Sampling",
    "draw",
    "filter_logits",
    "reference_mass",
    "sample_indices",
    "supports",
]
