# The inference clause 7.2 matrix

Clause 7.2 of the same standard lists twelve columns: six operators — the
residual add, the attention, the softmax, the rotation, the cache and the
normalisation — each in a forward and a backward column. Clause 7.3's 45 rows
are the test items; this package applies those rows to the twelve columns.

The applicability data is not guesswork. `marks.py` holds the per-(row, column)
table the analysis drew up: 191 marked pairs, spread over 30 rows, with the
fifteen rows that mark nothing recorded with a reason in `reasons.py`. A row
with no mark claims no column, and a pair without a mark is reported as `-` with
that reason in the disclosure, exactly as the 7.3 side reports an operator a row
does not apply to.

## One script per row

`items/` mirrors `gbz73/items/` one for one: `a1.py` is the reference-agreement
row, `b4.py` the irregular shapes, `f2.py` the size sweep. Each module declares
`ID`, `COLUMNS` (`None` when the applicability table governs the row) and
`run(ctx)`, where `ctx` is an
`InferenceContext` over one column. A row that is marked for a column has a
script; a row that is not marked anywhere has no script and a recorded reason.

The first thing every script does is state its check and its tolerance in the
module docstring. The tolerance comes from the same table as 7.3
(`tolerance_for(dtype, kind, multiplier)`), with two multipliers on top: one for
the operator family (attention is the widest) and a further factor of two for
the backward column, because a derivative spends the forward's rounding twice.

## The twelve columns

The forward columns are the six operators; the backward columns are the same six
judged by their Jacobian. `marks.TABLE_ORDER` is the standard's order — the six
forward columns, then the six backward ones — and it is what the matrix header
and the JSON report use.

The cache is the one column whose direction changes the call: the forward write
takes `(k, v, cache_k, cache_v, index)`, the backward read takes
`(cache_k, cache_v, index)`. `InferenceContext.project` owns that split, so a
script always builds the five forward arguments and the context narrows them.

## Running it

```
neutraltest gbz73 --clause 72
neutraltest gbz73 --clause 72 --features A1 --ops add_fwd
python -m pytest src/neutraltest/conformance --clause 72
```

`--clause 73` is the default and behaves exactly as before the option existed;
`--clause all` runs both matrices. The pytest plugin defaults to `all`, so a
plain `pytest` runs both; pass `--clause 73` to collect only the 7.3 rows.

## What a cell means

`P` the check passed, `F` it failed with the exception in the disclosure and the
summary, `-` the applicability table does not mark this pair and the reason is
printed, `K` a registered known issue reproduced, `·` the cell was not part of
this run (a narrowed `--ops` or `--features`).

## The seventh operator: sampling

Clause 7.2 names seven inference operators. Six are columns of the table above.
The seventh — Top-K / Top-P sampling — is not, and the reason is structural
rather than editorial:

> **A sampler has no backward.** Sampling draws from a distribution; the draw is
> not differentiable with respect to the logits, and no framework ships a
> gradient for it. A backward column would not be "untested", it would be
> meaningless.

So the sampler gets its own forward-only matrix (`gbz72/sampling/`) instead of
two extra columns that could never pass. It keeps the same bookkeeping — the
same row ids from the feature catalogue, the same `P/F/-/·` vocabulary, the same
disclosure of what was not exercised — so the two matrices read side by side.

### The oracle is a distribution, not a draw

Every other column compares tensors element by element. Two correct samplers
disagree on any individual draw, so a fixed seed does not rescue a per-element
comparison: it makes the output *repeatable*, not *equal to the reference*.

The claim is therefore about the distribution the device draws from:

| Piece | What it is |
| --- | --- |
| draws per cell | 4096 (`oracle.DRAWS`) |
| measured | empirical frequency of each token, per row |
| expected | `filter_logits(...)` → `softmax`, in `float64` |
| allowed drift | `clamp(6·sqrt(p(1−p)/N), min 0.01)` per token |

The floor of 0.01 matters for the tokens a filter erases: with `p = 0` the
textbook interval collapses to zero, and a single stray draw would fail the
cell. The floor gives those tokens the same room a rare token gets.

This catches both directions of the obvious mistakes: a generator that returns
the argmax puts all the mass on one token, and a generator that forgets the
Top-P cutoff leaves mass in the tail that should be exactly zero.

### The twelve rows and five settings

`items.py` holds twelve rows — reference agreement, overflow saturation,
subnormal probabilities, index dtype and range, NaN/Inf disclosure, signed zero,
cutoff accumulation, degenerate shapes, irregular shapes, output dtype,
parameter bounds, and whether each setting takes effect. Each row is checked
under five settings: `plain`, `top_k=4`, `top_p=0.75`, both together, and
`temperature=0.7`.

Rows whose claim is about the filter itself are checked once, on the CPU; the
distribution claim is the one every backend is held to. On the CPU the kernel
*is* the reference, so that cell reduces to a consistency check — it still runs,
because an oracle that rejects truth would otherwise only be discovered on
hardware.

```
neutraltest sampler
neutraltest sampler --json
neutraltest sampler --backend cpu --draws 8192
python -m pytest src/neutraltest/conformance/test_gbz72_sampling.py -q
```

A sampler cell reports `P` when no token was drawn more often than its reference
mass allows for the sample size, and `F` with the worst offending token, its
measured frequency, its reference mass and the allowed drift.
