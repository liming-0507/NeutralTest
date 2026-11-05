# Space: living next to a very large dependency

PyTorch is ~800 MB unpacked for a CPU build, several GB with the CUDA stack, and
it is a fast-moving upstream project. `torch-mbt` therefore treats it strictly as
a **runtime dependency you install once**, never as something to copy, patch in
place or check in.

Measured on a CPU-only development machine (Linux, torch 2.15.0+cpu):

| What | Size | Notes |
| --- | --- | --- |
| `site-packages/torch` | 809 MB | unpacked CPU build |
| Same, from the default PyPI wheel + `nvidia-*` deps | 3–5 GB | avoid on Linux; see below |
| Our package installed editable | 13 MB | the whole venv, `--system-site-packages` |
| Our repository (`.git` included) | under 1 MB, 53 files | no `torch`, no fixtures, no golden files |
| Our wheel | tens of KB | code only |

## Rules

### 1. Depend on torch, never vendor it

`pyproject.toml` asks for `torch>=2.1`; nothing imports a copy from the tree. That
keeps the repository small, the license situation clean, and the installed torch
patches applicable — a vendor that needs a modified dispatch layer installs *its
own* torch build, and `torch-mbt` tests whatever it finds.

The same applies to test data: inputs are generated from a seed at run time, and
reference values are computed on the fly in CPU `float64`. There are no golden
tensor files, so no test-data growth ever.

### 2. Always install the CPU wheel on CI

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

This is the single largest space lever: ~800 MB instead of 3–5 GB, because the
CPU index has no `nvidia-*`, `cudnn` or `triton` wheels.

Do **not** write `--index-url .../whl/cpu --extra-index-url https://pypi.org/simple`
and let the resolver choose `torch`: on Linux plain PyPI carries the manylinux
CUDA build of the same version number, and a resolver that sees both indexes may
pick it — 2 GB of `libtorch_cuda.so` plus `nvidia-*` wheels, for a test suite that
never touches a GPU. Pin the index per package instead:

```bash
pip install "torch==2.1.2" --index-url https://download.pytorch.org/whl/cpu
pip install "pytest>=7.4" --index-url https://pypi.org/simple
```

(`uv` behaves the same way; `tools/torch_matrix.sh` installs torch and pytest in
two separate steps for exactly this reason.) It is also the honest
target — this harness compares numerics against CPU PyTorch, so the CUDA runtime
buys nothing. Our CI does this in every job.

### 3. Many virtual environments, one copy of torch

A venv is cheap (13 MB here) — a *second torch* is not.

```bash
python -m venv .venv --system-site-packages   # reuse the interpreter's torch
.venv/bin/pip install -e . --no-deps          # never let pip resolve torch again
```

`--no-deps` is not an optimisation, it is the point: without it pip will happily
fetch a different (larger) torch wheel for the same interpreter.

With `uv` the sharing is even better — files are hardlinked out of one content
addressed cache, so ten environments using the same torch build occupy the disk
once:

```bash
uv venv -p 3.10 .venv-t21
uv pip install -e . --no-deps
uv pip install "torch==2.1.2" pytest --index-url https://download.pytorch.org/whl/cpu
```

For a one-off check without keeping an environment at all:

```bash
uv run --with torch==2.5.1 --with pytest pytest -q
```

### 4. Test a sparse matrix, not the full square

Torch wheels are ABI-specific (`cp310`, `cp311`, …), so environments for
different Python versions cannot share bytes. Four Python versions × five torch
versions is 20 installs, ~16 GB, and hours of CI for very little extra signal.

Use a sparse design instead — each row is a gate, not a sample:

| Gate | Shape | Answers |
| --- | --- | --- |
| our code on supported Pythons | 4 × latest torch | packaging, syntax, pytest-plugin behaviour |
| our code on old torch | 1 × oldest supported (`2.1`) | do we use an API that did not exist yet |
| boundaries | oldest py × oldest torch, newest py × newest torch | the corners where both drift at once |

`tools/torch_matrix.sh` runs exactly that shape and reuses the `uv` cache:

```bash
tools/torch_matrix.sh --python 3.10 --torch 2.1.2 2.5.1 2.9.0
```

### 5. Do not build PyTorch from source to run this suite

A PyTorch build tree is 20–40 GB and hours of CPU. If you need a modified
dispatch layer:

1. build **once**, on a machine big enough, into a wheel;
2. publish that wheel (local index, artifact store, container image);
3. install it like any other torch (`pip install ./torch-....whl`), then run
   `torch-mbt doctor` and `torch-mbt test`.

The same logic applies to container images: start from the vendor's image (which
already carries its torch and runtime), add `torch-mbt` on top — do not rebuild
torch inside the test image.

### 6. Housekeeping that actually reclaims space

Caches are what grow without anyone noticing. On the machine this was written on:
`~/.cache/pip` 630 MB, `~/.cache/uv` 1.2 GB, and a stray `.mypy_cache` of 95 MB
inside the repository.

```bash
uv cache prune            # drop unused versions from the shared cache
pip cache purge
rm -rf .mypy_cache .pytest_cache .ruff_cache
find . -name __pycache__ -type d -prune -exec rm -rf {} +
```

All of these are in `.gitignore`, so they never reach the published repository —
but they do reach a CI disk, and GitHub-hosted runners only have a few gigabytes
free. If a job installs several torch versions, purge between matrix legs or run
each leg on a fresh runner (which our CI does by construction).

### 7. If you truly need many copies

Use a copy-on-write filesystem: reflinks (`cp --reflink=auto`), btrfs/zfs, APFS,
or overlay mounts share unchanged extents. A container layer already does this,
which is why "one image per torch version, several containers" is cheaper than
"several venvs on the host".

## What this buys

- The published repository stays under 1 MB and clones in a second.
- A contributor reuses the torch they already have: `--system-site-packages` plus
  `--no-deps`, 13 MB.
- CI installs ~800 MB per job instead of 3–5 GB, and the matrix stays small
  enough that a pull request finishes in minutes.
- Testing a third-party accelerator never requires rebuilding PyTorch: their
  image, their torch, our wheel.
