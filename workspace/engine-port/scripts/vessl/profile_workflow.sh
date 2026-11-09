#!/usr/bin/bash
# Timeline + power profile with what works in a VESSL container without added privileges:
# nsys CUDA/NVTX/OS-runtime trace plus NVML power and clocks polled every 100 ms, saved as CSV.
# Probe and debug use only: tracing perturbs the workload, so numbers from here are not serving results.
#
# Usage: profile_workflow.sh <out_dir> -- <command...>
#   NSYS=<path to nsys>  (default: PATH, then /data/tools/nsight)
#   TRACE=cuda,nvtx,osrt  POLL_MS=100
set -u
OUT=${1:?usage: profile_workflow.sh <out_dir> -- <command...>}; shift
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "no command given" >&2; exit 2; }

NSYS=${NSYS:-$(command -v nsys || find /data/tools/nsight/root -type f -path '*target-linux-x64/nsys' 2>/dev/null | head -1)}
[ -x "$NSYS" ] || { echo "nsys not found; set NSYS=" >&2; exit 2; }
TRACE=${TRACE:-cuda,nvtx,osrt}
POLL_MS=${POLL_MS:-100}
mkdir -p "$OUT"

{
  echo "start_utc=$(date -u +%FT%T.%3NZ)"
  echo "command=$*"
  echo "trace=$TRACE poll_ms=$POLL_MS"
  "$NSYS" --version
  nvidia-smi --query-gpu=name,uuid,driver_version,power.limit --format=csv,noheader
} >"$OUT/meta.txt" 2>&1

nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem --format=csv -lms "$POLL_MS" \
  >"$OUT/nvml_power.csv" 2>"$OUT/nvml_power.err" &
POLLER=$!
trap 'kill "$POLLER" 2>/dev/null' EXIT

"$NSYS" profile -t "$TRACE" --force-overwrite=true -o "$OUT/trace" "$@"
RC=$?
kill "$POLLER" 2>/dev/null; wait "$POLLER" 2>/dev/null
echo "end_utc=$(date -u +%FT%T.%3NZ) rc=$RC" >>"$OUT/meta.txt"

if [ -f "$OUT/trace.nsys-rep" ]; then
  "$NSYS" stats --force-export=true --format csv --report cuda_gpu_kern_sum --report cuda_api_sum --report nvtx_sum \
    -o "$OUT/stats" "$OUT/trace.nsys-rep" >"$OUT/stats.log" 2>&1
fi
ls -l "$OUT" >>"$OUT/meta.txt"
exit "$RC"
