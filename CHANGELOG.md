# Changelog

## Unreleased

### Added — the part 7.2 sampler

- Clause 7.2 names seven inference operators; six are columns of the matrix. The
  seventh, Top-K / Top-P sampling, now has a forward-only matrix of its own
  (`gbz72/sampling/`) with twelve rows — reference agreement, overflow,
  subnormals, index dtype and range, NaN/Inf disclosure, signed zero, cutoff
  accumulation, degenerate and irregular shapes, output dtype, parameter bounds,
  and whether each knob takes effect — each checked under five settings.
- The sampler is not a thirteenth column because a sampler has no backward: the
  draw is not differentiable in the logits, so a backward cell could never pass.
- The judgement is a distribution, not a draw. Two correct samplers disagree on
  any individual draw, so a fixed seed makes a run repeatable, not equal. Cells
  compare the measured token frequencies over 4096 draws against the reference
  masses, with a per-token allowance of `6·sqrt(p(1−p)/N)` floored at `0.01` so
  that a token a filter erases is not held to a zero-width interval.
- `neutraltest sampler [--json] [--draws N]`, and 60 conformance cells in
  `test_gbz72_sampling.py`. The matrix is shown to go red — 48 of 60 cells — on a
  generator that ignores the filters, so a green run is evidence rather than a
  tautology.

### Added — SDK version matching

- `neutraltest sdk [--sdk FILE] [--json]` reports where this interpreter sits
  against the SDK release it claims to run (torch window, python window, per
  component) and against PyTorch's own published support window. The bundled
  table is a snapshot of upstream `RELEASE.md`, and the source line is printed
  with every run.
- Profiles are JSON, so no vendor is compiled in: `--sdk FILE` or the
  `NEUTRALTEST_SDK_PROFILE` environment variable, see `examples/sdk-profile.json`
  and `docs/sdk.md`.
- `--on-sdk-out-of-band {warn,skip,fail}` on every test-running verb, default
  `warn`: a deviation from the declared window is disclosed, and a run that
  claims a certified baseline can refuse to start. No conformance cell changes
  its outcome because of a version.
- `RunPolicy.on_sdk_out_of_band`, and `neutraltest.sdk` as a library
  (`detect`, `measure`, `merge`, `load_catalogue`).
- Three environment cells in the conformance suite check that the table covers
  the running release and that every deviation names the window it violates.

### Added — the 7.2 matrix

- Clause 7.2's twelve columns are now rows of the same catalogue: `gbz72/` holds
  the applicability table (`marks.py`, 191 marked pairs over 30 rows), the
  recorded reason for every unmarked pair (`reasons.py`), one script per marked
  row (`items/`), and a runner that renders the same `P/F/-/K` matrix with the
  standard's column order.
- `--clause {73,72,all}` on `neutraltest gbz73` and on the pytest plugin. The
  command defaults to `73` and prints exactly what it printed before the option
  existed; the plugin defaults to `all`, so a plain `pytest` exercises both.
- `neutraltest.matrix` is the clause-aware facade the command uses: it resolves
  `--ops` against operators for 7.3 and column names (or the standard's own
  labels, e.g. `Add-前`) for 7.2, and narrows only the clauses a name belongs to.
- `docs/gbz72.md`: how the applicability table, the scripts and the tolerance
  multipliers fit together.

### Added — known issues

- A registered deviation is its own outcome: `known`, `K` in the GB/Z matrix, a
  `known` column in the pytest summary, and a `KNOWN[<id>]` reason line carrying the
  measured deviation. It is never counted as passed and never as failed.
- Any registry is loaded from a path, so a registration set is data rather than
  source: `--known-registry PATH` (repeatable), `NEUTRALTEST_KNOWN`, or a JSON file
  dropped into `known_issues.local/` (git-ignored). The shipped
  `known_issues/example.json` is a format example with illustrative numbers; a
  registry that names a product does not belong in this repository, and the
  measured environment of a real entry belongs in `known_issues.local/`, where it
  is folded in by id at load time.
- The summary lists every entry in the loaded registry and marks the ones that
  reproduced (`KNOWN  <id> x2`), so a registration that stops matching is visible as a
  bare id rather than as silence.
- `neutraltest known` lists the registry, `--id` shows one entry, `--verify`
  re-runs every entry against its backend and reports `KNOWN` / `FIXED`
  (`--fail-on-fixed` exits 1 so CI notices a retired registration).
- `--known {report,strict}` and `--known-registry PATH` (pytest), plus the
  `NEUTRALTEST_KNOWN` and `NEUTRALTEST_KNOWN_MODE` environment variables.
- `docs/known-issues.md`: the four criteria for registering instead of fixing,
  the current table, what the report looks like, and the entry schema.

### Fixed (found by running on a second machine)

- `quantise()` clamps to the format's finite range before casting: `tensor.to(float8_e4m3fn)`
  saturates on torch 2.15 and returns NaN on torch 2.9, so the oracle cannot inherit it.
- A6 no longer assumes one answer for infinities. It compares against a clean run and
  requires that an element the poison reaches may only become zero or non-finite --
  `relu(-inf)` is a fixed zero point, `gelu(+inf)` is inf on one torch and NaN on another.
- B7 holds a strided transcendental result to the tolerance table instead of a
  hard-coded 1e-6, and reports the measured deviation.
- The privateuse1 stub scenarios skip, with a reason, on a torch build that already
  registers a privateuse1 backend (the rename is process-wide and one-shot).

### Fixed — timing rows on a machine that cannot resolve them

- F2, F3 and F5 are ratios of two measured timings, and on a shared machine two
  identical calls differ by 30-50 %: F2 failed on a 128-row layer_norm it had passed
  minutes earlier, F3 and F5 on operators that take 25 us a call. Every ratio bound
  now carries `NOISE_SLACK` (1.5x), and a call under `LOW_RESOLUTION_NS` (200 ns) is
  reported as `na` with the reason instead of being judged; `measure()` warms up 10
  and takes 25 samples instead of 3 and 7. A verdict about the clock is not a verdict
  about the operator.

### Added — F7, the accelerator against the CPU

- F7 runs the same operator, on the same inputs, through the same measurement
  protocol on the backend under test and on the CPU, and reports the ratio. Every
  other performance row measures one implementation against itself, so a backend that
  never beats the CPU passed all of them; a ratio above `SAME_ORDER` x `NOISE_SLACK`
  (2.25x) now fails. It is `na` on the CPU itself, which is the reference.

### Added

- `--fp32-precision {highest,high,medium}` on `neutraltest gbz73` and on pytest, plus a
  `fp32 matmul precision:` line in the report: a container that leaves the session at
  `high` turns an accurate float32 operator into an apparent 1e-2 error.
- A float32 mismatch far beyond its budget now carries a hint naming the mantissa width
  the deviation is consistent with (TF32-class, bfloat16-class).

## 0.1.0

First release.

### Framework

- `Backend` adapter ABC over PyTorch dispatch mechanisms (`cpu`, `cuda`, `xpu`,
  `mps`, `privateuse1`, custom), with availability probing, device selection,
  dtype support probing and optional synchronize / cache / memory / stream hooks.
- Backend discovery: built-ins, `neutraltest.backends` entry points, `--backend`
  specs (`auto`, `all`, comma lists, globs), deduplication per device type.
- `OpCase` test description plus `check_case` / `check_backward`: CPU-generated
  inputs, device execution, CPU `float64` reference, metadata-then-value checks,
  finiteness checks and gradient comparison.
- Tolerance model: dtype budget × numerical class (`exact`, `elementwise`,
  `transcendental`, `reduction`, `normalization`, `loss`, `matmul`, `composite`,
  `convolution`, `attention`), globally scalable.
- Run policy: unsupported-operator handling (`fail` / `skip` / `xfail`),
  tolerance multiplier, strict backend selection, device ordinal, seed.
- Deterministic seeding shared between the device run and the reference.

### pytest plugin

- Options: `--backend`, `--device-index`, `--list-backends`, `--backend-strict`,
  `--on-unsupported`, `--tolerance-multiplier`, `--nt-seed`, `--nt-report`,
  `--nt-no-summary`.
- Fixtures: `backend`, `device`, `device_index`, `nt`, `cpu_device`.
- Markers: `nt_backend`, `nt_exclude_backend`, `nt_family`, `nt_op`,
  `nt_grad`, `nt_slow`.
- Per-backend summary table and JSON result report.

### Suite

- 81 operator cases in 11 families, 151 dtype-parametrised tests: elementwise,
  activations, matmul, reduction, normalization, convolution, pooling, shape
  manipulation, indexing/scatter, losses, attention.
- Framework self-tests: 126 unit and behaviour tests, including real pytest
  subprocess runs of every option.

### CLI

- `neutraltest info`, `neutraltest doctor`, `neutraltest test`.

### FP8 clause 7.3 matrix

- `neutraltest gbz73` — the 45-item test catalogue of the accelerator interface
  standard's FP8 clause, applied to five operator families (grouped GEMM,
  LayerNorm, GELU, ReLU, Amax): 225 cells, **187 passed / 38 not applicable / 0
  failed** on CPU.
- Every row is one script under `src/neutraltest/gbz73/items/`; the catalogue,
  its tiers and the standard's own coverage bits live in `features.py`.
- Both baselines are CPU: float32 as the ordinary standard, float64 as the
  golden one. Tolerances are derived from the format
  (`2 x half_ulp x amplification`), not hard-coded.
- A `-` (not applicable) cell always carries a reason, printed by
  `render_disclosure`, and row F6 fails the build when it does not.
- Two defects found and fixed while calibrating: `RunContext.quantised()` was
  reusing the *input* scale for the output (every accumulator saturated), and
  the grouped-GEMM reference accepted a decreasing group list.

### Working next to a large dependency

- [docs/space.md](docs/space.md): how to keep the footprint small — CPU wheel
  index, shared interpreter, `--no-deps`, sparse version matrix, cache hygiene —
  with measured sizes.
- `tools/torch_matrix.sh`: run the suite against several torch versions through
  `uv`, reusing one content-addressed cache instead of one full copy per row.
- CI gained a sparse torch-compatibility job (oldest supported torch on the
  oldest Python, newest torch on the newest Python) instead of a full matrix.
- Verified on both ends of the range: torch 2.15 / Python 3.13 → 287 passed;
  torch 2.1.2 / Python 3.10 → 285 passed, 2 skipped (`F.rms_norm` needs
  torch >= 2.4; that torch build has no `bfloat16` `mse_loss` kernel on CPU,
  which the harness reports as *unsupported*, not as a numerical failure).
- Pinned the CPU wheel index per package: passing `--extra-index-url pypi.org`
  while installing torch lets the resolver pick the manylinux CUDA build of the
  same version (+2 GB of `libtorch_cuda.so` and `nvidia-*` wheels).
- The terminal summary now reports the policy that was actually in force and
  stops suggesting `--on-unsupported=skip` when it is already active.

### Out-of-tree backends

- [docs/privateuse1.md](docs/privateuse1.md): the two ways a vendor joins (register
  as `privateuse1`, or ship a torch whose device type is already `cuda`/`xpu`),
  the one-shot rename rule, the `privateuseone` placeholder, and the adapter
  checklist to run before blaming an operator.
- `tests/behavior/test_privateuse1_stub.py`: registers a stub `privateuse1`
  device in a subprocess and asserts the adapter reports name, count, device and
  aliases — and that an unavailable one never invents device 0.

### Known limitations

- Only the `cpu` backend is exercised on the development machine; `cuda`,
  `xpu`, `mps` and `privateuse1` adapters are detection-only there.
- Stream, memory-accounting and custom-RNG hooks are part of the adapter
  contract but not yet exercised by the bundled suite.
