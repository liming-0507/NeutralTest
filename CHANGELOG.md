# Changelog

## 0.1.0

First release.

### Framework

- `Backend` adapter ABC over PyTorch dispatch mechanisms (`cpu`, `cuda`, `xpu`,
  `mps`, `privateuse1`, custom), with availability probing, device selection,
  dtype support probing and optional synchronize / cache / memory / stream hooks.
- Backend discovery: built-ins, `torch_mbt.backends` entry points, `--backend`
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
  `--on-unsupported`, `--tolerance-multiplier`, `--mbt-seed`, `--mbt-report`,
  `--mbt-no-summary`.
- Fixtures: `backend`, `device`, `device_index`, `mbt`, `cpu_device`.
- Markers: `mbt_backend`, `mbt_exclude_backend`, `mbt_family`, `mbt_op`,
  `mbt_grad`, `mbt_slow`.
- Per-backend summary table and JSON result report.

### Suite

- 81 operator cases in 11 families, 151 dtype-parametrised tests: elementwise,
  activations, matmul, reduction, normalization, convolution, pooling, shape
  manipulation, indexing/scatter, losses, attention.
- Framework self-tests: 126 unit and behaviour tests, including real pytest
  subprocess runs of every option.

### CLI

- `torch-mbt info`, `torch-mbt doctor`, `torch-mbt test`.

### Known limitations

- Only the `cpu` backend is exercised on the development machine; `cuda`,
  `xpu`, `mps` and `privateuse1` adapters are detection-only there.
- Stream, memory-accounting and custom-RNG hooks are part of the adapter
  contract but not yet exercised by the bundled suite.
