# Architecture

`torch-mbt` is a thin, layered harness. Each layer has one job, and only the top
two know that pytest exists.

```
        src/torch_mbt/conformance/*.py      data: OpCase lists
                              │                 (one file per operator family)
                              ▼
  ┌───────────────────────────────────────────────────────────────────────┐
  │ pytest_plugin.py   options → policy → fixture `backend` → parametrise │
  │                    markers, summary table, JSON report                │
  └───────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌───────────────────────────────────────────────────────────────────────┐
  │ testing.py         OpCase → build_inputs → run on device → reference  │
  │                    metadata checks, value checks, gradient checks     │
  └───────────────────────────────────────────────────────────────────────┘
        │                    │                   │                │
        ▼                    ▼                   ▼                ▼
   seeding.py          compare.py         tolerance.py       policy.py
   deterministic       exact/fuzzy        dtype × kind        what to do when
   CPU inputs          comparison         budgets             a backend says no
                              │
                              ▼
  ┌───────────────────────────────────────────────────────────────────────┐
  │ backends/          Backend ABC (a mechanism + a device type)          │
  │                    discovery, entry points, --backend spec parsing    │
  └───────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        torch (cpu, cuda, xpu, mps, privateuse1 …)
```

## Layers

### Backends — *how do I reach a device?*

`Backend` is an abstract adapter. It answers: which mechanism does this device
use, what string does `torch.device` expect, is it available, how many devices
are there, and a handful of optional services (synchronize, empty_cache,
streams, memory accounting). Everything else in the framework talks only to this
interface — there is no `if vendor == ...` anywhere.

Built-in adapters cover the mechanisms that exist in upstream PyTorch: `cpu`,
`cuda`, `xpu`, `mps`, and `privateuse1` (whose device type is whatever
`rename_privateuse1_backend()` set, if anything). Further adapters can be
installed out of tree through the `torch_mbt.backends` entry point group; see
[backends.md](backends.md).

Discovery returns *unavailable* backends too. That is deliberate: pytest
parametrises them, the `backend` fixture skips them with a clear reason, and the
scope of a run is visible in the report instead of silently shrinking.

### Policy — *what should happen when the backend cannot do something?*

One immutable `RunPolicy` holds the knobs that should not be scattered across
tests: how to treat an unimplemented operator (`fail` / `skip` / `xfail`), a
global tolerance multiplier, whether an unavailable backend is an error, the
device ordinal, and the seed. `get_policy()` returns the live policy,
`patched_policy(**changes)` overrides it for a block.

### Comparison — *did the backend compute the same numbers?*

`compare.py` flattens arbitrary outputs (tensors, tuples, lists, dicts,
namedtuples, `torch.Size`), compares arity, then compares values. Non-float
tensors are compared exactly — an off-by-one index is never a tolerance
question. Float tensors are promoted to a common dtype and compared with the
budget from `tolerance.py`.

### Tolerance — *how close is close enough?*

Two axes: the storage precision (float32 vs bfloat16 differ by four orders of
magnitude) and the operator's numerical class (a `sin` accumulating round-off
through argument reduction is not held to the same standard as an `add`). The
table is small, explicit and auditable; `--tolerance-multiplier` scales it.

### Seeding — *are both sides seeing the same inputs?*

Inputs are always generated on CPU, in float32, and then cast. That has three
benefits: devices never need to implement `randn`; every dtype in a case sees
the same underlying values, so a float16 failure is comparable with the float32
one; and the float64 reference is computed from *identical* bytes rather than a
similar-looking draw.

### The test engine — *one function per question*

`check_case(case, backend, dtype)` is the whole contract:

1. build inputs on CPU with a seeded generator;
2. cast to the tested dtype and move to the device;
3. run the operator on the device;
4. run the same operator on CPU in `float64`;
5. check metadata (arity, shapes, output dtypes) — **before** values, so a
   wrong dtype is reported as a dtype bug rather than a numeric one;
6. check values against the tolerance budget;
7. check that everything the reference says is finite actually is finite;
8. optionally replay with `requires_grad` and compare every input gradient.

### The pytest plugin — *the only pytest-aware code*

It adds the options, builds the policy, parametrises a session-scoped `backend`
fixture over the selected backends, skips unavailable ones, records every
outcome, and prints a per-backend summary. It also translates
`UnsupportedByBackend` raised *anywhere* in a test into skip/xfail according to
the policy, so a suite written by hand gets the same behaviour as one using the
bundle helpers.

## Why the reference is CPU float64

The alternative — compare against CUDA, or against a stored golden file — makes
the harness depend on a specific vendor implementation and on a specific version
of it. CPU `float64` is the only implementation that every PyTorch user already
has, is accurate to enough digits that it can be treated as ground truth for
float32 work, and makes the failure message meaningful: the numbers differ by X,
and the difference is the accelerator's.

Two consequences are handled explicitly:

- **Indices and boolean masks** cannot be compared fuzzily, so those operators
  run through the exact path.
- **Non-deterministic reductions** (e.g. atomics-based sums) legitimately differ
  in the last bits; they get a wider budget, not a blind one.

## Extension points

| Point | How |
| --- | --- |
| New backend | subclass `Backend`, install, register an entry point |
| New operator coverage | append an `OpCase` to a module in `src/torch_mbt/conformance/` |
| New numerical class | add a scale to `KIND_SCALE` in `tolerance.py` |
| Different CI semantics | `--on-unsupported`, `--tolerance-multiplier`, `--backend-strict` |
| Machine-readable results | `--mbt-report results.json` |
