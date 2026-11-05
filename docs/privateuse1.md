# Plugging in an out-of-tree backend

PyTorch only dispatches to device types it was compiled with (`cpu`, `cuda`,
`xpu`, `mps`, `meta`, …). A vendor accelerator that is not in that list has two
options, and `torch-mbt` has to work with both.

## Model A — register as `privateuse1` (no torch rebuild)

This is the normal route: torch keeps its own binaries, the vendor package adds
kernels for the `PrivateUse1` dispatch key and gives the placeholder a name at
import time.

```python
import torch, types

torch.utils.rename_privateuse1_backend("acme")          # one-shot, process-wide
torch.utils.generate_methods_for_privateuse1_backend()  # torch.Tensor.to / factories

module = types.ModuleType("torch.acme")
module.is_available = lambda: True
module.device_count = lambda: 4
module.current_device = lambda: 0
module.set_device = lambda index: None
torch._register_device_module("acme", module)           # makes torch.acme importable
```

After that `torch.device("acme:1")` is a real device and `torch.acme` behaves
like `torch.cuda`. What matters for a test harness:

| Fact | Consequence |
| --- | --- |
| The rename is **one-shot and process-global** | it must happen before any tensor work; a second call raises `RuntimeError: ... has already been set` |
| Before the rename the name is `privateuseone` | a harness cannot guess the vendor name — it must ask |
| After the rename `privateuse1` is **no longer a valid device string** | match on both names, never construct `privateuse1:0` |
| Everything hangs off `torch._C._get_privateuse1_backend_name()` | that is the only reliable way to learn the name |

`torch-mbt` models this as the `privateuse1` adapter, which is always present but
only *available* when a device module answers:

```bash
torch-mbt info                      # lists the registered name, or "privateuseone" when free
torch-mbt test --list-backends      # what this interpreter can actually reach
```

`tests/behavior/test_privateuse1_stub.py` exercises the whole path with a stub
module, so the contract is tested on machines without any accelerator.

## Model B — ship a torch that already knows the device

Some stacks keep the CUDA API surface (driver, compiler and tensor libraries are
CUDA-compatible), so they distribute their **own PyTorch build** where the device
type is simply `cuda`. Nothing is registered at import time; `torch.cuda` is the
accelerator.

For the harness this is not a different mechanism, only a different binary: the
`cuda` adapter is picked up, `--backend=cuda` selects it, and the whole suite
runs unchanged. The same holds for a vendor that reuses the `xpu` device type.

Practical consequences:

- `--backend=auto` is what you want: it selects whatever this interpreter can
  actually reach, whichever model the vendor chose.
- `torch-mbt doctor --backend=...` is the boundary test — it proves the build can
  allocate, compute, backpropagate and move data before any operator result is
  blamed on the operator.
- Do not name backends after products in the suite: `--backend=cuda` in the
  vendor's own environment means that vendor, and the report says so.

## Model C — vendor patches torch itself

Also seen in practice: a full PyTorch source tree with vendor patches, distributed
as a source archive or a container image. Nothing changes for `torch-mbt`: install
their build, then `pip install -e . --no-deps` and run the suite. Do not rebuild
their torch to run tests (see [space.md](space.md)).

## What a new adapter has to provide

An adapter is ~40 lines; see [backends.md](backends.md) for the full skeleton.
The load-bearing parts:

| Member | Why it matters |
| --- | --- |
| `name`, `mechanism`, `device_type` | selection, reporting, and the glob `--backend=priv*` |
| `is_available()` | the only honest source of "can this interpreter reach the device" |
| `device_count()` | must return 0 when unavailable — never invent device 0 |
| `make_device(index)` | the default implementation already validates the index |
| `synchronize()` | without it, timing and error attribution are both wrong |
| `memory_allocated()` / `memory_reserved()` | the OOM reports in failures |
| `manual_seed()` | reproducible inputs per device |
| `supports_dtype(dtype)` | turns "this backend has no bfloat16" into a skip, not a failure |

## Checklist before pointing the suite at a new device

1. `torch-mbt info` — the device is listed, available, with the expected count.
2. `torch-mbt doctor --backend=<name>` — allocate / elementwise / matmul /
   reduction / softmax / backward / transfer all pass.
3. `torch-mbt test --backend=<name> --on-unsupported=skip -rs` — the operator
   matrix runs; every skip is a named missing kernel, not a silent no-op.
4. `torch-mbt test --backend=<name> --mbt-report report.json` — keep the JSON:
   it records backend, family, outcome and dtype per test, which is what a
   conformance claim should cite.
