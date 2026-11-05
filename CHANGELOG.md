# Changelog

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
