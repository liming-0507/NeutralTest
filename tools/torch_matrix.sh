#!/usr/bin/env bash
# Run the suite against several torch versions without multiplying disk usage.
#
# The torch wheels are ABI-specific, so each row needs its own environment; the
# uv cache is content addressed and hardlinks into every environment, so N rows
# cost one download of each torch version, not N copies.
#
#   tools/torch_matrix.sh                                  # default: latest only
#   tools/torch_matrix.sh --profile                        # oldest / middle / latest
#   tools/torch_matrix.sh --python 3.10 --torch 2.1.2 2.5.1
#   tools/torch_matrix.sh --python 3.10,3.12 --torch 2.9.0 --clean
#
# --profile runs the sampled compatibility set the package claims to support:
# the oldest declared torch on the oldest supported python, a middle release on
# the running python, and whatever torch is newest on it.  That is the same
# question CI's torch-compat job asks, one release in between included.
set -uo pipefail

PYTHONS=""
TORCHES=""
CLEAN=0
PROFILE=0
ROOT_DIR=""
PYTEST_ARGS=()

usage() {
  sed -n '2,12p' "$0" | grep '^#' | sed 's|^# \{0,1\}||'
  cat <<'EOF'

Options:
  --python LIST    comma separated python versions (default: the running one)
  --torch LIST     space or comma separated torch versions (default: latest)
  --profile        sampled set: oldest declared torch, a middle one, and latest
  --pytest ARGS    extra arguments for pytest (quote them)
  --clean          remove the environments afterwards
  -h, --help       this text
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --python) PYTHONS="$2"; shift 2 ;;
    --torch)  TORCHES="$2"; shift 2 ;;
    --pytest) PYTEST_ARGS+=("$2"); shift 2 ;;
    --clean)  CLEAN=1; shift ;;
    --profile) PROFILE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v uv >/dev/null || { echo "this helper needs uv: https://docs.astral.sh/uv/" >&2; exit 2; }

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
DEFAULT_PY=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')

# Keep the two in step with pyproject.toml (dependencies / requires-python) and
# docs/space.md; they are read here rather than hard-coded twice.
OLDEST_TORCH=$(sed -n 's/.*"torch>= *\([0-9][0-9.]*\).*/\1/p' "$ROOT_DIR/pyproject.toml" | head -1)
FLOOR_PY=$(sed -n 's/.*requires-python *= *">=\([0-9][0-9.]*\).*/\1/p' "$ROOT_DIR/pyproject.toml" | head -1)
if [ -z "$OLDEST_TORCH" ] || [ -z "$FLOOR_PY" ]; then
  echo "could not read the supported versions from pyproject.toml" >&2; exit 2
fi
# torch 2.1 ships wheels up to python 3.11; a profile row pins a pair rather
# than asking for a wheel that does not exist.
case "$OLDEST_TORCH" in
  2.1|2.2|2.3|2.4) OLDEST_PY="${FLOOR_PY%.*}.11" ;;
  *) OLDEST_PY="$FLOOR_PY" ;;
esac
[ -n "${MIDDLE_TORCH:-}" ] || MIDDLE_TORCH="2.9.0"
[ -n "$PYTHONS" ] || PYTHONS="$DEFAULT_PY"
[ -n "$TORCHES" ] || TORCHES=""
if [ "$PROFILE" -eq 1 ]; then
  # One row per (python, torch).  The explicit --python/--torch forms above stay
  # a square; only the profile is a diagonal, because the oldest torch simply
  # has no wheel for the newest interpreter.
  ROWS=("$OLDEST_PY:$OLDEST_TORCH" "$DEFAULT_PY:$MIDDLE_TORCH" "$DEFAULT_PY:")
fi
WORK="$ROOT_DIR/.venv-matrix"
mkdir -p "$WORK"

CPU_INDEX=https://download.pytorch.org/whl/cpu
RESULTS=()
FAILED=0

# ROWS is "python:torch"; the explicit paths build the same shape from their
# two lists, so there is one loop below rather than two.
if [ "$PROFILE" -ne 1 ]; then
  ROWS=()
  for py in ${PYTHONS//,/ }; do
    for tv in ${TORCHES//,/ }; do ROWS+=("$py:$tv"); done
done
fi

for row in "${ROWS[@]}"; do
  py="${row%%:*}"
  tv="${row#*:}"
  env_dir="$WORK/py$py-torch${tv:-latest}"
  label="py$py torch ${tv:-latest}"
  printf '\n=== %s ===\n' "$label"

  rm -rf "$env_dir"
  if ! uv venv -q -p "$py" "$env_dir" >/dev/null 2>&1; then
    echo "SKIP  no Python $py available"; RESULTS+=("SKIP  $label  (no python $py)"); continue
  fi
  py_bin="$env_dir/bin/python"

  if ! uv pip install -q --python "$py_bin" -e "$ROOT_DIR" --no-deps >/dev/null 2>&1; then
    echo "FAIL  could not install neutraltest"; RESULTS+=("FAIL  $label  (install neutraltest)"); FAILED=1; continue
  fi

  # torch from the CPU index ONLY: adding pypi.org as an extra index makes the
  # resolver prefer the manylinux CUDA build on Linux (+2 GB, plus nvidia-*).
  spec=("torch")
  [ -n "$tv" ] && spec=("torch==$tv")
  if ! uv pip install -q --python "$py_bin" "${spec[@]}" --index-url "$CPU_INDEX" >/dev/null 2>&1; then
    echo "SKIP  no torch ${tv:-latest} wheel for python $py"
    RESULTS+=("SKIP  $label  (no wheel)")
    continue
  fi

  uv pip install -q --python "$py_bin" "pytest>=7.4" --index-url https://pypi.org/simple >/dev/null 2>&1

  got=$("$py_bin" -c 'import torch; print(torch.__version__)')
  echo "torch $got"
  if ( cd "$ROOT_DIR" && "$py_bin" -m pytest -q "${PYTEST_ARGS[@]}" 2>&1 | tail -3 ); then
    RESULTS+=("PASS  $label  (torch $got)")
  else
    RESULTS+=("FAIL  $label  (torch $got)"); FAILED=1
  fi

  if [ "$CLEAN" -eq 1 ]; then rm -rf "$env_dir"; fi
done

printf '\n=========================== torch matrix ===========================\n'
for line in "${RESULTS[@]}"; do printf '%s\n' "$line"; done
printf 'environments: %s  (du: %s)\n' "$WORK" "$(du -sh "$WORK" 2>/dev/null | cut -f1)"
printf 'uv cache: %s\n' "$(du -sh "${UV_CACHE_DIR:-$HOME/.cache/uv}" 2>/dev/null | cut -f1)"

exit "$FAILED"
