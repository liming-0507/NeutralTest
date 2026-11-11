# The inference clause 7.2 matrix

Clause 7.2 of the same standard lists twelve columns: six operators — the
residual add, the attention, the softmax, the rotation, the cache and the
normalisation — each in a forward and a backward column. Clause 7.3's 45 rows
are the test items; this package applies those rows to the twelve columns.

The applicability data is not guesswork. `marks.py` holds the per-(row, column)
table the analysis drew up: 182 marked pairs, spread over 30 rows, with the
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
