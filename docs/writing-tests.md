# Writing tests

Two ways to use the harness: describe the operator as an `OpCase` and let
`check_case` do everything (what `src/torch_mbt/conformance/` does), or call the
primitives yourself when the shape of the check does not fit.

## The fast path: `OpCase`

```python
import torch
from torch_mbt import OpCase, check_backward, check_case, randn

def _build(dtype, generator):
    x = randn((4, 8), dtype, generator)
    w = randn((8, 16), dtype, generator)
    return (x, w), {}

MATMUL = OpCase(
    id="mm/2d",                 # stable, appears in failure output and the report
    family="matmul",            # grouping in the summary
    op="mm",                    # operator name in the summary
    fn=torch.mm,                # what to call
    build=_build,               # how to make its inputs
    dtypes=(torch.float32,),
    kind="matmul",              # tolerance class
    grad=True,                  # also compare input gradients
    grad_inputs=(0, 1),         # positional args that require grad
)

def test_mm(mbt):
    check_case(MATMUL, mbt.backend, torch.float32)
```

| Field | Meaning |
| --- | --- |
| `id` | stable identifier; also the pytest id |
| `family` | reporting group |
| `op` | operator name for the report |
| `fn` | the callable under test |
| `build(dtype, generator)` | returns `(args, kwargs)`; **always runs on CPU** |
| `dtypes` | dtypes to exercise (default `float32`, `bfloat16`, `float16`) |
| `kind` | tolerance class — see [tolerances](#tolerances) |
| `reference_dtype` | precision of the CPU golden; `None` = same as the device run |
| `grad` / `grad_inputs` | replay forward with leaves requiring grad and compare gradients |
| `out_dtypes` | expected output dtypes, positionally; `None` = input dtype. **Required for ops that return indices** (`max`, `topk`, `argmax`) |
| `check_dtype` | assert output dtypes (default on) |
| `seed` | per-case input seed |
| `tags` | free-form labels (`"slow"`, `"training"`) |
| `tolerance_multiplier` | per-case relaxation, e.g. `2.0` for an accumulation-heavy variant |

### Rules for `build`

1. **Use the provided generator.** `randn(shape, dtype, generator)` and
   `torch.randint(..., generator=generator)` keep the run reproducible and keep
   the device run and the float64 reference on identical bytes.
2. **Do not move to a device.** `check_case` does the transfer; building on CPU
   is what makes the reference trustworthy.
3. **Keep indices integral.** `torch.randint` for index tensors, so both sides
   gather the same elements.
4. **Stay away from domain edges** unless the case is *about* them — `log` wants
   `positive()`, `sqrt` wants `positive()`, `div` wants a non-zero denominator.
   A NaN in the reference turns a value check into a finiteness check.

`src/torch_mbt/conformance/support.py` provides `unary`, `binary` and `positive`
builders, and the `params(cases)` helper that turns a list of cases into pytest
parameters with the right marks:

```python
import pytest
from support import params, run_case

def test_elementwise(case, dtype, backend):
    run_case(case, dtype, backend)

@pytest.mark.parametrize("case,dtype", params(CASES))
```

## The manual path

```python
import torch
from torch_mbt import assert_close, assert_grad_close, move_to, tolerance_for

def test_custom(backend, device):
    a = torch.linspace(-1, 1, 16, dtype=torch.float64)
    expected = torch.special.erf(a)
    actual = torch.special.erf(move_to(a.to(torch.float32), device))
    assert_close(actual, expected, dtype=torch.float32, kind="transcendental")
```

Available pieces:

| Function | Purpose |
| --- | --- |
| `check_case(case, backend, dtype)` | the full pipeline for an `OpCase` |
| `check_backward(case, backend, dtype)` | gradient-only check |
| `evaluate(case, device, dtype)` | returns `(actual, expected)` without asserting |
| `assert_close(actual, expected, ...)` | compare two results, any nesting |
| `assert_grad_close(actual, expected, ...)` | compare gradient lists, tolerating `None` |
| `assert_all_finite(tensor, msg)` | NaN/Inf guard |
| `tolerance_for(dtype, kind)` | the budget itself |
| `move_to(value, device)` | recursive transfer for tensors/nested containers |
| `seed_all(seed)` / `seeded(seed)` | deterministic RNG for a block |
| `unsupported_policy(label)` | apply the run policy around a block |

## The `mbt` fixture

```python
def test_something(mbt):
    x = mbt.tensor(8, 16)                 # CPU randn -> device, float32
    y = mbt.reference(x)                  # same values, CPU, float64
    mbt.sync()                            # backend.synchronize()
    tol = mbt.tolerance(torch.float32, "matmul")
    print(mbt.backend.name, mbt.device, mbt.policy)
```

Other fixtures: `backend` (session-scoped, skips when unavailable), `device`
(`backend.make_device(--device-index)`), `device_index`, `cpu_device`.

## Tolerances

The budget is `base[dtype] × KIND_SCALE[kind] × multiplier`.

| kind | typical operators | scale |
| --- | --- | --- |
| `exact` | `cat`, `transpose`, `gather`, `masked_fill` | 0 (bitwise) |
| `elementwise` | `add`, `mul`, `relu`, `maximum`, `argmax` | 1 |
| `transcendental` | `exp`, `log`, `sin`, `cos`, `gelu`, `softmax` | 2 |
| `reduction` | `sum`, `mean`, `var`, `logsumexp`, pooling | 2 |
| `normalization` | `layer_norm`, `batch_norm`, `group_norm` | 4 |
| `loss` | `mse_loss`, `cross_entropy`, `kl_div` | 4 |
| `matmul` | `matmul`, `bmm`, `linear` | 4 |
| `composite` | backward passes, fused epilogues | 8 |
| `convolution` | `conv2d`, `conv_transpose1d/2d` | 8 |
| `attention` | `scaled_dot_product_attention` | 16 |

Base budgets: `float64` 1e-12, `float32` rtol 1.3e-6 / atol 1e-5, `float16` 1e-3,
`bfloat16` rtol 1.6e-2 / atol 1e-2. Integers and booleans are always exact.

Pick the **narrowest kind that is honest**. Widening the kind to make a test go
green hides real regressions at the scale of the widening; if the operator is
genuinely looser, `tolerance_multiplier=2.0` on that one case is a better
record than a new class.

## When a backend does not implement the operator

Raise it, do not skip it by hand:

```python
from torch_mbt import UnsupportedByBackend

def test_fft(mbt):
    if not hasattr(torch, "fft_plan"):
        raise UnsupportedByBackend("fft is not implemented for this backend")
```

`--on-unsupported` then decides: `fail` (default — an unimplemented operator is
a result, not an accident), `skip` (report the gap and keep going), or `xfail`.
The same policy applies to `NotImplementedError` and to the error strings
PyTorch raises for missing kernels, so most vendor gaps need no explicit raise
at all. The summary table counts them in the `unsupported` column either way.

## Narrowing a run

```bash
pytest src/torch_mbt/conformance -m "mbt_family('matmul')"
pytest src/torch_mbt/conformance -m "mbt_grad"
pytest src/torch_mbt/conformance -m "not mbt_slow"
pytest src/torch_mbt/conformance -m "mbt_backend('cpu')"
pytest src/torch_mbt/conformance --backend=all -m "mbt_exclude_backend('cpu')"
```

## JSON report

`--mbt-report=results.json` writes:

```json
{
  "totals": {"passed": 161, "failed": 0, "skipped": 0},
  "backends": {"cpu": {"passed": 161}},
  "families": {"matmul": {"passed": 12}},
  "duration_seconds": 0.34,
  "results": [
    {"nodeid": "...", "backend": "cpu", "outcome": "passed", "family": "matmul",
     "op": "mm", "dtype": "float32", "unsupported": false, "duration": 0.001}
  ]
}
```

That file is the intended input for a dashboard or a per-backend CI gate: the
`unsupported` flag and the `families` histogram are what you usually want to
diff between two machines or two driver versions.
