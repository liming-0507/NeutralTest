# Contributing

## Ground rules

1. **No vendor-specific code, names or paths.** The core describes PyTorch
   mechanisms, not products. A vendor belongs in an out-of-tree adapter package
   registered through the `neutraltest.backends` entry point (see
   [docs/backends.md](docs/backends.md)). Pull requests that mention a specific
   accelerator's API in `src/` will be asked to move it out.
2. **The CPU suite must stay green without an accelerator.** `tests/unit` and
   `tests/behavior` must pass on a machine with nothing but CPU PyTorch;
   `src/neutraltest/conformance` runs there too, against the `cpu` backend.
3. **Every new operator case needs a tolerance class it deserves.** Do not widen
   a `kind` to make a case pass — a too-loose budget hides real regressions.
4. **New framework behaviour needs a test that fails without it.** The unit
   suite is expected to catch the framework's bugs, not just cover its lines.

## Setup

```bash
python -m venv .venv --system-site-packages   # reuse an installed CPU torch
.venv/bin/pip install -e . --no-deps
.venv/bin/pip install pytest ruff mypy
```

## Disk

Torch is large; read [docs/space.md](docs/space.md) before adding a dependency,
a fixture file or a CI job that installs another torch. The short version:
`--no-deps`, the CPU wheel index, one installation per interpreter, sparse
version matrix.

## The checks CI runs

```bash
ruff format --check .
ruff check .
mypy
pytest tests -q
```

`ruff format` is authoritative — run `ruff format .` before committing. mypy
checks `src/` under `python_version = "3.12"` (the numpy stubs used in CI need
PEP 695 syntax); the package itself still supports Python ≥ 3.10.

## Adding operator coverage

Append an `OpCase` to the matching module in `src/neutraltest/conformance/`, or add a new
module if the family is new (follow the shape of `test_matmul.py`: a `CASES`
tuple, one parametrised test, `run_case`). Rules for the `build` callable are in
[docs/writing-tests.md](docs/writing-tests.md#the-fast-path-opcase). Regenerate
the matrix table:

```bash
python tools/gen_operator_matrix.py
```

If the operator is not implemented by a given backend, raise
`UnsupportedByBackend` (or let PyTorch's "no kernel" error propagate) and leave
the decision to `--on-unsupported`.

## Adding a backend

- Adapter contract: [docs/backends.md](docs/backends.md).
- If it is a mechanism that exists in upstream PyTorch, add it to
  `src/neutraltest/backends/builtin.py` and to `builtin_backends()`.
- Otherwise ship it as a separate distribution with an entry point, and consider
  contributing the adapter's *contract tests* here so the next backend author can
  reuse them.

## Reporting a conformance failure

Include:

- `neutraltest info` output (backend table, dtype matrix, tolerance budgets),
- the smallest failing case: `pytest --backend=<name> -k "<case id>" -q`,
- the full failure message — it names the case, the dtype, the offending index
  and the observed vs expected values,
- the `--nt-report` JSON if the failure is intermittent or shape-dependent.

A conformance failure is either the accelerator's numerics or the adapter's
plumbing. `neutraltest doctor` separates the two in seconds; please paste its
output too.

## Commit messages

Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `build:`, `ci:`,
`refactor:`), one logical change each.
