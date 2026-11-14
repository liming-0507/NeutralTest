# SDK versions: what the environment is, and what it was certified against

A conformance result only means something next to the versions it was produced
on. Two versioned things move independently:

- **PyTorch** publishes a support window per release (which Pythons a wheel is
  built and tested for) and a release calendar a year ahead.
- **An accelerator SDK** ships as one release — driver, runtime, operator
  libraries, framework adapters and containers — rebuilt as a whole against one
  PyTorch version.

So "the suite failed" and "the suite failed on a stack nobody certified" are
different facts, and the second one is usually the more useful. `neutraltest sdk`
makes that difference measurable.

```bash
neutraltest sdk                                  # against PyTorch's own window
neutraltest sdk --sdk examples/sdk-profile.json  # against a release profile
neutraltest sdk --sdk profile.json --json        # machine readable
```

```
profile    example-sdk-2.13
python     3.13.13 (this interpreter)
torch      2.15.0+cpu (release 2.15)

component          version     window        band
-----------------  ----------  ------------  -------
python             3.13.13     >=3.10,<3.13  above
torch              2.15.0+cpu  >=2.13,<2.14  above
driver             3.8.1       >=3.8,<3.9    ok
runtime            7.8.0       >=7.8,<7.9    ok
operator-library   2.3.3       >=2.3,<2.4    ok
framework-adapter  -           >=1.31,<1.32  unknown

not measured: framework-adapter (the profile names a component this process does
not publish a version for)

deviations: 2
  - python 3.13.13 is above example-sdk-2.13's window (>=3.10,<3.13)
  - torch 2.15.0+cpu is above example-sdk-2.13's window (>=2.13,<2.14)
```

`band` is one of `ok`, `below`, `above`, `unknown`. `unknown` is what a
component gets when nothing publishes a version for it — it is **not** a pass.

## Two rules

**Only what can be measured is measured.** The interpreter and PyTorch versions
are read from the running process. A component the profile declares (because the
SDK publishes it) but this process cannot confirm is listed under *not
measured*, never assumed to match. Framework adapters are only probed for a
`__version__` **after** something imported them, because importing an adapter
can initialise hardware — a version check must not do that.

**A deviation is a disclosure, not a failure.** No conformance cell changes its
outcome because the environment is outside a window. What a deviation *means*
is the run's decision:

| `--on-sdk-out-of-band` | effect on a pytest run |
| --- | --- |
| `warn` (default) | one `PytestWarning` in the header, the run proceeds |
| `skip` | every test is deselectable via `-m 'not nt_sdk_out_of_band'`; the tests are marked, not silently dropped |
| `fail` | `pytest.UsageError` before collection — the run refuses to claim a certified baseline |

`neutraltest sdk --on-sdk-out-of-band fail` exits 1 on a deviation, so a
pipeline can gate without parsing output.

## Profiles are data

A profile is JSON; nothing about a vendor is compiled in. See
[`examples/sdk-profile.json`](../examples/sdk-profile.json):

```json
{
  "profile": {
    "name": "example-sdk-2.13",
    "torch_window": ">=2.13,<2.14",
    "python_window": ">=3.10,<3.13",
    "components": [
      {"name": "driver", "version": "3.8.1", "window": ">=3.8,<3.9"}
    ]
  }
}
```

Fields:

- `torch_window` — which PyTorch versions this release is built against.
- `python_window` — which interpreters it ships for. When a profile gives one,
  it is authoritative; PyTorch's own window only fills in when it does not.
- `components[].version` — the version the release ships.
- `components[].probe` — the import name of the module that reports the version
  at runtime (e.g. a framework adapter). The module is only consulted if it is
  **already imported**; a probed component that is not importable in this
  process is listed as *not measured*. `probe_aliases` covers a second name.
  A component with no probe is judged by its declared version.
- `pytorch_support` (top level, optional) — replaces the bundled snapshot of
  PyTorch's release table, so a site can refresh the table without upgrading
  the package.

Windows use the usual interval syntax — `>=3.10,<3.13`, `3.12`, `3.12.*`,
`~=2.4`, `==2.9.1` — and only the components present in the version are
compared, so `3.12` and `3.12.0` are the same version.

## The bundled PyTorch table

`SUPPORT_SOURCE` records exactly where the snapshot came from and when it was
copied. If the running torch is newer than the table, the newest release at or
below it is used and the report says so (`torch-catalogue: unknown`) rather than
pretending a promise exists. `neutraltest sdk` prints the source line on every
run.

## What it does not decide

The advisory is about *fit*, not about *defects*. A stack inside its window can
still fail every cell, and a stack outside it can pass everything — that is
exactly why the two facts are reported separately. Registered, explained
deviations stay in [known-issues.md](known-issues.md); version fit is not one
of them, because it is a property of the environment rather than of an operator.
