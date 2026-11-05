#!/usr/bin/env bash
# Run the suite against several torch versions without multiplying disk usage.
#
# The torch wheels are ABI-specific, so each row needs its own environment; the
# uv cache is content addressed and hardlinks into every environment, so N rows
# cost one download of each torch version, not N copies.
#
#   tools/torch_matrix.sh                                  # default: latest only
#   tools/torch_matrix.sh --python 3.10 --torch 2.1.2 2.5.1
#   tools/torch_matrix.sh --python 3.10,3.12 --torch 2.9.0 --clean
set -uo pipefail

PYTHONS=""
TORCHES=""
CLEAN=0
ROOT_DIR=""
PYTEST_ARGS=()

usage() {
  sed -n '2,12p' "$0" | grep '^#' | sed 's|^# \{0,1\}||'
  cat <<'EOF'

Options:
  --python LIST    comma separated python versions (default: the running one)
  --torch LIST     space or comma separated torch versions (default: latest)
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
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v uv >/dev/null || { echo "this helper needs uv: https://docs.astral.sh/uv/" >&2; exit 2; }

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
DEFAULT_PY=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
[ -n "$PYTHONS" ] || PYTHONS="$DEFAULT_PY"
[ -n "$TORCHES" ] || TORCHES=""
WORK="$ROOT_DIR/.venv-matrix"
mkdir -p "$WORK"

CPU_INDEX=https://download.pytorch.org/whl/cpu
RESULTS=()
FAILED=0

for py in ${PYTHONS//,/ }; do
  for tv in ${TORCHES//,/ }; do
    env_dir="$WORK/py$py-torch${tv:-latest}"
    label="py$py torch ${tv:-latest}"
    printf '\n=== %s ===\n' "$label"

    rm -rf "$env_dir"
    if ! uv venv -q -p "$py" "$env_dir" >/dev/null 2>&1; then
      echo "SKIP  no Python $py available"; RESULTS+=("SKIP  $label  (no python $py)"); continue
    fi
    py_bin="$env_dir/bin/python"

    if ! uv pip install -q --python "$py_bin" -e "$ROOT_DIR" --no-deps >/dev/null 2>&1; then
      echo "FAIL  could not install torch-mbt"; RESULTS+=("FAIL  $label  (install torch-mbt)"); FAILED=1; continue
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
done

printf '\n=========================== torch matrix ===========================\n'
for line in "${RESULTS[@]}"; do printf '%s\n' "$line"; done
printf 'environments: %s  (du: %s)\n' "$WORK" "$(du -sh "$WORK" 2>/dev/null | cut -f1)"
printf 'uv cache: %s\n' "$(du -sh "${UV_CACHE_DIR:-$HOME/.cache/uv}" 2>/dev/null | cut -f1)"

exit "$FAILED"
