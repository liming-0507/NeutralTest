# torch-mbt

**Vendor-neutral multi-backend conformance testing for PyTorch operators.**

<sub>English · [简体中文](README.zh-CN.md)</sub>

`torch-mbt` is a pytest plugin plus a library. It answers one question for any
PyTorch build:

> Does this accelerator compute the same numbers as reference PyTorch — for the
> same inputs, the same dtypes, forward *and* backward?

It does that without knowing which accelerator it is looking at.

```console
$ torch-mbt info
torch-mbt 0.1.0
python      3.13.13 (x86_64)
torch       2.15.0+cpu
registered  privateuseone

               mechanism    device_type    available  devices
-------------  -----------  -------------  ---------  -------
  privateuse1  privateuse1  privateuseone  no         0
  xpu          xpu          xpu            no         0
  cuda         cuda         cuda           no         0
  mps          mps          mps            no         0
* cpu          cpu          cpu            yes        1

* selected by --backend=auto
```

## Why

PyTorch's dispatch layer has a **fixed, small set of device types**: `cpu`,
`cuda`, `xpu`, `mps`, `meta`, and one slot reserved for out-of-tree backends,
`privateuse1` (renameable to a single custom name via
`torch.utils.rename_privateuse1_backend`). Accelerator vendors do not get one
device type each; they either claim that single renameable slot or maintain a
fork that adds a type of their own.

Two consequences follow, and both hurt testability:

1. A conformance suite written against `device="cuda"` silently skips every
   accelerator that is not CUDA — which is most of them.
2. Every vendor ends up writing its own suite, with its own tolerances, its own
   definition of "passing", and its own conventions for comparing against the
   CPU reference. Cross-vendor numbers stop being comparable.

`torch-mbt` takes the opposite approach: **test mechanisms, not brands.** The
framework parametrises every test over the *dispatch mechanisms* a machine
actually exposes (`cpu`, `cuda`, `xpu`, `mps`, `privateuse1`). Whatever device
an adapter ends up pointing at, the same operator matrix, the same tolerances
and the same reference computation apply, so results across machines and across
vendors are directly comparable.

Nothing in this repository is vendor-specific: no vendor names, no vendor
headers, no vendor runtime. A backend is described by an adapter that maps a
mechanism onto a device type, and adapters can also be shipped out-of-tree as
entry points (see [docs/backends.md](docs/backends.md)).

## Install

```bash
pip install torch-mbt
```

From a checkout:

```bash
python -m venv .venv --system-site-packages   # reuse an installed CPU torch
.venv/bin/pip install -e . --no-deps
```

Requires Python ≥ 3.10, `torch` ≥ 2.1 and `pytest` ≥ 7.4.

## Quick start

```bash
torch-mbt info                    # what this machine exposes
torch-mbt doctor                  # 7 smoke checks per available backend
torch-mbt test                    # run the bundled conformance suite
torch-mbt test --backend=cpu -k matmul
```

Or drive pytest directly — the plugin is registered by the `pytest11` entry
point, so no `conftest.py` is needed:

```bash
pytest --backend=cpu src/torch_mbt/conformance
pytest --backend=all --on-unsupported=skip      # unavailable backends are skipped, not failed
pytest --backend=cpu,xpu --mbt-report=report.json
```

### `--backend` specs

| Spec | Meaning |
| --- | --- |
| `auto` (default) | every discovered backend that is available on this machine |
| `all` | every discovered backend, available or not (unavailable ones skip) |
| `cpu` | one backend by name |
| `cpu,xpu` | a comma-separated list, order preserved |
| `*pu` | a shell glob over backend names, device types and aliases |

The `TORCH_MBT_BACKEND` environment variable is the fallback when the option is
absent, so CI can select a backend without editing the command line.

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--backend` | `auto` | backend spec (see above) |
| `--device-index` | `0` | which device ordinal of each backend to test |
| `--list-backends` | — | print the backend table and exit |
| `--backend-strict` | off | unavailable requested backend is a usage error, not a skip |
| `--on-unsupported` | `fail` | what to do when an operator is not implemented: `fail`, `skip`, `xfail` |
| `--tolerance-multiplier` | `1.0` | global scale on the tolerance budget |
| `--mbt-seed` | `20240101` | seed for generated inputs |
| `--mbt-report PATH` | — | write a JSON result report |
| `--mbt-no-summary` | off | suppress the terminal summary table |

Markers, so a run can be narrowed without touching test code: `mbt_backend`,
`mbt_exclude_backend`, `mbt_family`, `mbt_op`, `mbt_grad`, `mbt_slow`.

### What a run looks like

```console
$ pytest -q --backend=cpu src/torch_mbt/conformance
=============================== torch-mbt summary ===============================
backend  passed  failed  skipped  xfailed  xpassed  error  unsupported  time
-------  ------  ------  -------  -------  -------  -----  -----------  -----
cpu      161     0       0        0        0        0      0            0.34s
```

`unsupported` counts the tests the backend reported as "no kernel for this op".
By default those are failures; `--on-unsupported=skip` turns them into skips so
a partial backend can still show what *does* work.

## How a test is built

A conformance test is data, not code: an `OpCase` names the operator, the input
builder, the dtypes to try, and the numerical class the operator belongs to.

```python
import torch
from torch_mbt import OpCase, check_case, randn

def _build(dtype, generator):
    x = randn((8, 16), dtype, generator)
    y = randn((16,), dtype, generator)
    return (x, y), {}

ADD = OpCase(
    id="add/broadcast",
    family="elementwise",
    op="add",
    fn=torch.add,
    build=_build,
    kind="elementwise",
    grad=True,
    grad_inputs=(0, 1),
)

def test_add(mbt):
    check_case(ADD, mbt.backend, torch.float32)
```

`check_case` does the work:

1. builds inputs **on CPU** with a seeded generator — the device never has to
   implement `randn`, and both sides see identical bytes;
2. moves them to the backend device with the tested dtype;
3. recomputes the same call on CPU in `float64` as the reference;
4. compares metadata first (arity, shapes, output dtypes) and then values, with
   a tolerance budget derived from the dtype and the operator's numerical class;
5. with `grad=True`, replays the same forward with leaves requiring grad, sums
   the outputs and compares every input gradient.

Because the reference is plain CPU PyTorch, a mismatch is *always* the
accelerator's — or the adapter's.

## Operator matrix

The bundled suite covers **11 families / 81 cases / 151 dtype-parametrised
tests**, chosen to exercise every shape of numeric behaviour an accelerator
tends to get wrong: elementwise, activations, matmul, reduction, normalization,
convolution, pooling, shape manipulation, indexing/scatter, losses and
attention.

See [docs/operator-matrix.md](docs/operator-matrix.md) for the full table.

## Growing next to PyTorch

Torch is 800 MB unpacked for a CPU build and several GB with the CUDA stack, so
the harness is built to reuse one installation instead of carrying its own:

- the package depends on `torch` and never vendors it — the repository is under
  1 MB with no golden files, because inputs are generated from a seed and the
  reference is computed at run time;
- environments are cheap (13 MB with `--system-site-packages`), a second torch is
  not — `pip install -e . --no-deps` and let the existing torch be used;
- CI installs from `https://download.pytorch.org/whl/cpu` (~800 MB instead of
  3–5 GB, no `nvidia-*`/`triton`/`cudnn`);
- compatibility is sampled sparsely — old torch on one Python, the rest on the
  latest — with `tools/torch_matrix.sh`.

Details and the numbers: [docs/space.md](docs/space.md).

## Tolerances

Comparing floating point across implementations needs a budget, not a constant.
`torch-mbt` derives one from the dtype and the operator's numerical class:

| kind | example ops | scale |
| --- | --- | --- |
| `exact` | `cat`, `transpose`, index ops | 0 (bitwise) |
| `elementwise` | `add`, `mul`, `relu`, `max` | 1× |
| `transcendental` | `exp`, `log`, `sin`, `gelu`, `softmax` | 2× |
| `reduction` | `sum`, `mean`, `topk`, `var` | 2× |
| `loss` / `matmul` / `normalization` | `mse_loss`, `matmul`, `layer_norm` | 4× |
| `convolution` / `composite` | `conv2d`, backward passes | 8× |
| `attention` | `scaled_dot_product_attention` | 16× |

`float32` starts from `(rtol=1.3e-6, atol=1e-5)`, `float16` from `1e-3`,
`bfloat16` from `1.6e-2`, `float64` from `1e-12`; integers and booleans are
compared exactly. `--tolerance-multiplier` scales the whole table, which is the
honest way to say "this backend is known to be looser" without hiding an
operator that is wrong by a factor of a thousand.

## Adding a backend

An adapter is a small class describing *how to reach* the device:

```python
from torch_mbt import Backend, Mechanism

class AcceleratorBackend(Backend):
    name = "accelerator"            # what --backend matches
    mechanism = Mechanism.PRIVATEUSE1

    @property
    def device_type(self) -> str:
        return "vendorx"            # whatever rename_privateuse1_backend() used

    @property
    def is_available(self) -> bool:
        return torch.vendorx.is_available()
```

Ship it as an entry point and the plugin discovers it automatically:

```toml
[project.entry-points."torch_mbt.backends"]
accelerator = "vendorx.torch_test:AcceleratorBackend"
```

Full contract, plus the `privateuse1` specifics and how to test an adapter
itself: [docs/backends.md](docs/backends.md).

## CLI

| Command | Purpose |
| --- | --- |
| `torch-mbt info` | version, torch build, registered `privateuse1` name, backend table, dtype support matrix, tolerance budgets |
| `torch-mbt doctor` | 7 smoke checks per available backend (allocate, elementwise, matmul, reduction, softmax, backward, host↔device transfer); exits non-zero if any check fails |
| `torch-mbt test [pytest args]` | runs the bundled suite (shipped inside the package); unknown arguments are forwarded to pytest |

`doctor` is the fastest way to tell "the device is not there" from "the device
is there and computes the wrong thing".

## Repository layout

```
src/torch_mbt/
  backends/      Backend ABC, built-in adapters, discovery and --backend parsing
  testing.py     OpCase, input building, check_case / check_backward
  compare.py     tensor comparison, exact/fuzzy, gradient comparison
  tolerance.py   dtype × numerical-class tolerance table
  seeding.py     deterministic input generation
  policy.py      run policy (unsupported handling, tolerance scale, seed, device)
  pytest_plugin.py  CLI options, fixtures, markers, summary, JSON report
  report.py      result recording and the summary table
  cli.py         info / doctor / test
tests/
  unit/          the framework's own tests (no accelerator needed)
  behavior/      the plugin driven through real pytest subprocesses
  conformance/   the operator matrix — this is the part you point at a device
tools/         matrix generator, torch-version matrix runner
docs/
```

## Development

```bash
python -m venv .venv --system-site-packages
.venv/bin/pip install -e . --no-deps
.venv/bin/python -m pytest tests -q
ruff format --check . && ruff check .
mypy
```

The suite is designed to run on a machine with no accelerator at all: `cpu` is
always present, unavailable backends show up as skips, and `tests/unit` never
touches a device.

## Status

Alpha. The framework and the operator matrix are complete and green on CPU
PyTorch; the `cuda`, `xpu`, `mps` and `privateuse1` adapters are detection-only
here because no such device exists on the development machine. Results against
accelerators are exactly what this project wants contributions of.

## License

Apache-2.0 — see [LICENSE](LICENSE).
