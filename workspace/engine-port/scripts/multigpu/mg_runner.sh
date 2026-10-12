#!/bin/bash
# Multi-GPU boot runner (PREREG_MULTIGPU rev6).  One boot + its step list; see mg_boot.py.
#   mg_runner.sh <spec.json> [--dry-run]
# * client RLIMIT_NOFILE soft is raised here (H6) and `ulimit -n` is written to <out_root>/ULIMIT_N.txt;
#   mg_loadgen.py raises it again per cell and records the achieved value (fd_limit.json).
# * PDMUX_VENV (optional) is activated like engine_bench_runner.sh does.
# * N2/N3/E2 runner paths are untouched.
set -uo pipefail
spec="${1:?spec json required}"; shift || true
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PDMUX_VENV:-}" && -f "${PDMUX_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PDMUX_VENV}/bin/activate"
fi
ulimit -S -n "$(ulimit -H -n | sed 's/unlimited/65536/')" 2>/dev/null || ulimit -S -n 65536 2>/dev/null || true
out_root="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["out_root"])' "${spec}")" || exit 2
mkdir -p "${out_root}"
echo "ulimit_n_soft=$(ulimit -S -n) ulimit_n_hard=$(ulimit -H -n) at $(date -u +%FT%TZ)" >> "${out_root}/ULIMIT_N.txt"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
exec python3 "${here}/mg_boot.py" --spec "${spec}" "$@"
