#!/usr/bin/env bash
# Run the conformance suite on a device and leave both result files behind.
#
# The table builder reads `<feature> <column> PASSED|KNOWN|FAILED` lines.  They
# are not what pytest prints, so the run is not finished when pytest returns:
# the cells have to be pulled out of `-rA` output first, and the sed rules that
# do it are easy to get subtly wrong (an XFAIL line has no trailing `]`, and a
# run started from the repository root reports different ids than one started
# from a parent directory).
#
# Usage, on the machine that has the device:
#   tools/device_cells.sh [pytest args...]
#
# Writes $CELLS_OUT (default /tmp/cells.txt) with one line per measured cell:
#   <backend> <feature> <column> PASSED|KNOWN|FAILED
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${CELLS_OUT:-/tmp/cells.txt}"
log="${CELLS_LOG:-/tmp/conformance.log}"

# The interpreter matters: a stock python3 often has no pytest, and on a
# development container the one that does is the image's own.  PYTHON overrides.
python="${PYTHON:-python3}"

# Run from the parent of the repository and name the directory absolutely: the
# short ids pytest prints in that form are the ones the rules below match.
cd "$(dirname "$here")"
"$python" -m pytest "$here/src/neutraltest/conformance" -q -p no:cacheprovider -rA "$@" \
    >"$log" 2>&1 || true

grep -E '^(PASSED|XFAIL|FAILED) ' "$log" \
    | sed -E \
        -e 's/^PASSED .*::test_[a-z0-9_]+\[(privateuse1|cpu)-([A-F][0-9]+)-([a-z_0-9+]+)\]$/\1 \2 \3 PASSED/' \
        -e 's/^XFAIL .*::test_[a-z0-9_]+\[(privateuse1|cpu)-([A-F][0-9]+)-([a-z_0-9+]+)\] - reason: .*$/\1 \2 \3 KNOWN/' \
        -e 's/^FAILED .*::test_[a-z0-9_]+\[(privateuse1|cpu)-([A-F][0-9]+)-([a-z_0-9+]+)\]$/\1 \2 \3 FAILED/' \
    | grep -E '^(privateuse1|cpu) ' \
    | sort -u >"$out"

echo "wrote $out: $(wc -l <"$out") cell(s)"
awk '{ print $1 }' "$out" | sort | uniq -c
