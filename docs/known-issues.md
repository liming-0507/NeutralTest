# Known issues

Some deviations are real operator behaviour on a given backend rather than defects in this
harness. They must not be silently skipped -- a skip hides the cell from the report -- and they
must not be quietly absorbed by widening a tolerance, which would let a future regression land in
the same cell unnoticed. They are registered instead: the cell keeps running, keeps its own
outcome, and states why it is expected.

A registered cell has its own outcome, `known`, with its own symbol (`K` in the matrix), and a
`KNOWN[...]` reason that names the entry. It is never counted as passed and never as failed.

## What may be registered

All four must hold. If any of them does not, the cell is a defect and stays red.

1. **It is a deviation, not a refusal.** The operator runs and produces numbers; it does not
   raise, abort or hang.
2. **The reference is not the suspect.** The same case on CPU, against the same float64
   reference, is orders of magnitude closer -- so the measurement is not simply our tolerance.
3. **It is systematic, not a single element.** Most elements agree, and the disagreement is a
   uniform loss of mantissa. A layout or indexing error disagrees on a *shape*, not on 7-8 digits
   of the same value.
4. **It is reproducible with the recorded environment.** The entry records torch, the backend
   package, the device and the container; `neutraltest known --verify` re-runs it.

A cell that is *unimplemented* is a different thing: raise `UnsupportedByBackend` and let
`--on-unsupported` decide (`skip`/`xfail`/`fail`). Registered deviations are for implemented
operators that are simply less precise than the contract.

## What is registered today

Five float32 convolutions on an accelerator that exposes itself as `privateuse1`. Every row
below is traceable: the packaged entry carries the numbers, and the exact environment (torch,
the backend package, the card, the container) and the raw log are kept alongside the suite in
`src/neutraltest/known_issues.local/` (git-ignored), joined to the entries by id.

| id | case | out shape | backend worst abs | elements off | rel. to scale | CPU worst abs | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `accelerator-fp32-conv2d-pad1` | conv2d/pad1 | (2,4,8,8) | 6.203e-03 | 92.0% | 3.1e-04 | 3.344e-06 | 1855x |
| `accelerator-fp32-conv2d-groups-2` | conv2d/groups-2 | (2,4,6,6) | 4.107e-03 | 91.3% | 2.3e-04 | 1.251e-06 | 3283x |
| `accelerator-fp32-conv2d-stride2` | conv2d/stride-2 | (1,2,5,5) | 3.645e-03 | 98.0% | 4.9e-04 | 6.453e-07 | 5649x |
| `accelerator-fp32-conv-transpose2d` | conv-transpose2d | (2,2,6,6) | 3.154e-03 | 88.9% | 3.0e-04 | 1.298e-06 | 2430x |
| `accelerator-fp32-conv1d-pad1` | conv1d/pad1 | (2,4,10) | 2.622e-03 | 86.2% | 5.1e-04 | 5.457e-07 | 4805x |

The relative figures put all five at a ten-bit mantissa, which is what a float32 convolution
computed through a narrow internal format looks like.

### Why this is the backend, not this harness

Each entry carries these four arguments in `why_not_harness`:

- **The CPU is 2000-6000x closer on the same case** (1.7e-07 relative). Measurement noise does
  not have that shape.
- **86-98% of the elements agree to the last digits** and the worst element is a rounding-scale
  difference, not a shift of the image. An indexing or padding bug moves values by whole strides.
- **It does not follow `--fp32-precision`** -- the same five cells fail at `highest`, `high` and
  `medium` -- while `torch.matmul` on the same device reaches 1.5e-06 with `highest`. The device
  can do full float32; this path does not.
- **bfloat16 convolutions pass on the same backend** under the bfloat16 budget, so the tolerance
  is not simply too tight: the operator is consistent with its own dtype and only with that one.

## What the report looks like

The matrix marks the cell `K`, the summary counts it under `known`, and the reason line names the
entry:

```
$ neutraltest gbz73 --backend=privateuse1
fp32 matmul precision: highest (float32 matmul keeps a 24-bit mantissa)
...
known issues: 5 entry/entries in the registry -- see known_issues/; KNOWN <id> marks a reproduced one
  KNOWN  accelerator-fp32-conv1d-pad1 x1
  KNOWN  accelerator-fp32-conv2d-groups-2 x1
  KNOWN  accelerator-fp32-conv2d-pad1 x1
  KNOWN  accelerator-fp32-conv2d-stride2 x1
  KNOWN  accelerator-fp32-conv-transpose2d x1
  ...
```

Under pytest the same cells are `xfail` reports whose reason carries the entry id and the measured
deviation, so `-rx` lists them, and the summary table has a `known` column:

```
neutralTest summary
backend      passed  failed  skipped  known  ...
privateuse1     197       0       22      5  ...
```

`--known=strict` turns every registered cell back into a failure. That is what a release job
should run, so a registration can never silently outlive its reason.

## Re-checking a registration

```
neutraltest known                     # list what is registered, with the recorded deviation
neutraltest known --id <id>           # one entry in full: evidence, why_not_harness, environment
neutraltest known --verify            # re-run each entry against its backend
neutraltest known --verify --fail-on-fixed    # ... and exit 1 if one no longer reproduces
```

`--verify` prints one line per entry and backend:

```
id                                         backend     registered               observed now            status
accelerator-fp32-conv2d-pad1       privateuse1 6.203e-03 @ (0,0,1,6)     6.203e-03 @ (0,0,1,6)   KNOWN
```

Statuses: `KNOWN` (still reproduces), `FIXED` (within tolerance now -- retire the entry),
`-` (backend unavailable here), `?` (the case or operator is gone), `!` (error). Exit codes:
0 all good, 1 `--fail-on-fixed` found a fixed entry, 2 an id, backend or case could not be found.

When a backend is fixed, delete the entry (or leave it with `--fail-on-fixed` in CI so the build
tells you). Nothing else in the suite has to change: the cell simply starts passing.

## Adding an entry

Put a JSON file in `src/neutraltest/known_issues/` (or point `--known-registry` at one):

```json
{
  "schema": 1,
  "entries": [{
    "id": "accelerator-fp32-conv2d-pad1",
    "summary": "float32 conv2d on this backend is computed with about a 10-bit mantissa",
    "backend": {"device_type": "<the device type this backend registers>", "name": "*"},
    "case": "conv2d/pad1",
    "dtype": "float32",
    "measured": {"worst_abs": 0.0062, "cpu_worst_abs": 3.3e-06,
                 "ratio_backend_over_cpu": 1855, "effective_mantissa_bits": 10},
    "environment": {"note": "measured environment recorded in known_issues.local/"},
    "verdict": "known fp32 convolution narrowing on this backend",
    "why_not_harness": ["cpu is 1855x closer on the same case", "86% of the elements agree"],
    "recorded": "2026-11-06",
    "reference": "report/neutraltest-caip-20261106/caip-run.log"
  }]
}
```

Matching: `case` and `backend.name` are glob patterns, `dtype` is exact, `device_type` is
case-insensitive. The first matching entry wins, so order entries from specific to general.
`NEUTRALTEST_KNOWN` adds registries the same way (path list, `os.pathsep`-separated).

The packaged registry is anonymous on purpose: it names the device *type* and the numbers, never
the product. If you measured an entry yourself, put the environment next to the suite in
`src/neutraltest/known_issues.local/measurements.json`:

```json
{"schema": 1, "entries": [{"id": "accelerator-fp32-conv2d-pad1", "environment": {"torch": "...", "...": "..."}}]}
```

The registry loads that file when it exists and folds each record into the entry with the same
id, so `neutraltest known --id <id>` prints the full environment in the checkout where the
measurement was taken, and the published file stays free of it.

Every entry needs an `id`, `case`, `dtype`, `backend`, `measured`, `environment`, `verdict` and at
least one `why_not_harness` line; a registry that is missing them, or that repeats an id, fails to
load.
