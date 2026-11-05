# Backends

A **backend** is an adapter that maps a PyTorch *dispatch mechanism* onto a
device type the framework can actually allocate on. The framework knows nothing
else about it: no vendor names, no vendor APIs, no capability tables baked into
the core.

## What upstream PyTorch gives you

| Mechanism | Device type | Who can use it |
| --- | --- | --- |
| `CPU` | `cpu` | everyone |
| `CUDA` | `cuda` | CUDA builds (NVIDIA or compatible) |
| `XPU` | `xpu` | XPU builds |
| `MPS` | `mps` | Apple silicon |
| `PRIVATEUSE1` | `privateuse1`, or one custom name | any out-of-tree accelerator |
| `CUSTOM` | anything else | forks that add a device type |

`privateuse1` is the only slot upstream reserves for out-of-tree devices, and it
can be renamed **once** per process:

```python
import torch
torch.utils.rename_privateuse1_backend("vendorx")   # once, at import time
torch._C._get_privateuse1_backend_name()            # -> "vendorx"
```

A vendor that renamed the slot gets `device_type == "vendorx"`; a vendor that
did not rename it still reports `"privateuseone"`, and the adapter is simply
unavailable — which is the correct behaviour, because nothing can allocate.

## The adapter contract

Two members are abstract; everything else has a sensible default.

```python
import torch
from torch_mbt import Backend, Mechanism


class VendorXBackend(Backend):
    name = "vendorx"                    # matches --backend and markers
    mechanism = Mechanism.PRIVATEUSE1   # the dispatch mechanism
    priority = 100                      # discovery order hint (higher = earlier)
    requires_index = True               # False for single-device mechanisms (cpu, mps)

    @property
    def device_type(self) -> str:
        """The string torch.device accepts, e.g. "cuda", "xpu", "vendorx"."""
        return "vendorx"

    @property
    def is_available(self) -> bool:
        """True when this process can allocate on the device right now."""
        return hasattr(torch, "vendorx") and torch.vendorx.is_available()
```

### Optional members, and when to override them

| Member | Default | Override when |
| --- | --- | --- |
| `aliases()` | `()` | the device answers to several names (`privateuse1` does: the custom name plus the literal `privateuseone`) |
| `device_count()` | `module.device_count()`, or 1 | the mechanism reports devices differently — **must return 0 when unavailable** |
| `make_device(index)` | `torch.device(device_type, index)` | unusual device construction |
| `current_device_index()` / `set_device_index(i)` | module attribute / setter | the backend has no current-device concept |
| `synchronize(index)` | module `synchronize`, else `torch.accelerator.synchronize`, else no-op for CPU | the backend needs an explicit barrier before comparison |
| `empty_cache()` | module `empty_cache` / `torch.accelerator.empty_cache` | memory pressure handling is custom |
| `memory_allocated()` / `memory_reserved()` | module equivalents, else `None` | memory reporting exists |
| `manual_seed(seed)` | `torch.manual_seed` | the device has its own RNG |
| `new_stream()` / `stream()` / `has_streams()` | `torch.Stream` via the module, or unsupported | streams exist |
| `supports_dtype(dtype)` | probes `torch.empty(1, dtype=..., device=...)` | a cheaper/safer probe exists |
| `properties(index)` | dict of `name`/`device_type`/`mechanism`/`index`/`capability`/`memory` | extra diagnostics are useful in `torch-mbt info` |

Two defaults are load-bearing and worth repeating:

- **`device_count()` returns 0 for an unavailable backend.** Some vendor modules
  happily report a device count on a machine with no driver; trusting that turns
  into a segfault or a device-0 allocation that cannot possibly succeed.
- **`make_device(index)` never invents ordinal 0.** An out-of-range request
  raises `IndexError`; an unknown device type (registered mechanism, but the
  torch build does not know the string) raises `BackendUnavailable` with the
  hint to call `rename_privateuse1_backend()`.

## Registering an adapter

### In this repository

Add the class to `src/torch_mbt/backends/builtin.py` and include it in
`builtin_backends()`. Only do this for mechanisms that exist in upstream
PyTorch — the built-ins deliberately describe the platform, not a vendor.

### Out of tree (recommended for vendors)

Ship an entry point; the plugin discovers it at startup, no `conftest.py`
needed:

```toml
[project.entry-points."torch_mbt.backends"]
vendorx = "vendorx.torch_test:VendorXBackend"
```

Discovery is defensive: an entry point that fails to import produces a warning
and is skipped, never a crash — a broken plugin must not take down the harness.

If two adapters resolve to the same `device_type`, one is kept: the adapter whose
`name` equals its `device_type`, otherwise the one with the higher `priority`.
This keeps a vendor adapter from being shadowed by the built-in generic
`privateuse1` adapter, and vice versa.

### Ad hoc, inside a test session

```python
from torch_mbt.backends import discover

backends = discover(extra=(VendorXBackend(),))
```

## Selecting backends

```
--backend=auto          # available backends only (default)
--backend=all           # everything discovered; unavailable ones skip
--backend=cpu,vendorx   # explicit list
--backend='*pu'         # glob over names, device types and aliases
```

`auto` and `all` cannot be combined with other tokens; an unknown name is a
usage error listing the known ones (`--backend-strict` extends that to
*unavailable* names, which are otherwise reported as skips).

## What the framework will do with your adapter

`torch-mbt doctor` runs seven checks per available backend — allocate, elementwise,
matmul, reduction, softmax, backward, host↔device round-trip — each followed by
`synchronize()`, so asynchronous failures surface at the right check instead of
in the next one. If those pass and `pytest --backend=vendorx src/torch_mbt/conformance`
still fails, the adapter is fine and the operator is not.

## Testing an adapter without hardware

Adapters are ordinary objects; `tests/unit/test_backends.py` exercises the
contract against a fake `Backend` whose `device_type` is `"meta"` — a real
PyTorch device type that needs no accelerator. Use the same trick: a fake module
namespace plus `torch.device("meta")` covers the device-type plumbing, and a
`Meta` tensor can be allocated without any vendor runtime present.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `BackendUnavailable: ... is not a device type this torch build knows` | the mechanism is registered but `rename_privateuse1_backend()` was never called, or the fork that adds the type is not installed |
| backend reported unavailable although the device exists | `is_available` requires the vendor package to be imported first; import it in the adapter module |
| `IndexError: device index 0 out of range (count=0)` | `device_count()` returned 0 — the adapter says it cannot see the device |
| tests skip with "backend is unavailable" during CI | expected: unavailable backends parametrise and skip; use `--backend-strict` to make that a failure |
