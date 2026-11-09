#!/usr/bin/bash
# nsys kernel lane under green contexts + cudagraph, plus wall-time overhead of tracing. Diagnostics and
# user-space install only. Usage: bash gpu_obs_green_probe.sh <out_dir> [runs]
# Needs green_readout.py (src/multiplex/) and graph_census.py (opchar branch) beside it; workspace/engine-port/results/vessl_gpu_obs_probe_2026-11-02/green_cudagraph/scripts_used/ (opchar branch) holds the set that ran.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?out_dir}; RUNS=${2:-3}
PY=${PY:-/opt/conda/bin/python}
TOOLS=/data/tools
NSYS=$(find "$TOOLS/nsight/root" -type f -path '*target-linux-x64/nsys' | head -1)
export PATH=/opt/conda/bin:$PATH PYTHONWARNINGS=ignore
mkdir -p "$OUT"
LOAD="$HERE/gpu_obs_green_load.py"

# sgl_kernel links libnuma.so.1, which this Workspace image may lack: extract it, do not install it.
if ! "$PY" -c 'import torch; import sgl_kernel' >"$OUT/import_check.txt" 2>&1; then
  if grep -q libnuma "$OUT/import_check.txt"; then
    mkdir -p "$TOOLS/libnuma/dl"
    # apt lists are empty here and refreshing them would change system state, so read the archive index directly.
    UB=http://archive.ubuntu.com/ubuntu
    read -r F SHA < <(curl -fsSL -m 120 "$UB/dists/noble/main/binary-amd64/Packages.gz" | gunzip |
      awk '$1=="Package:"{m=($2=="libnuma1")} m&&$1=="Filename:"{f=$2} m&&$1=="SHA256:"{print f,$2;exit}')
    D="$TOOLS/libnuma/dl/$(basename "$F")"
    { curl -fsSL -m 120 -o "$D" "$UB/$F" && echo "$SHA  $D" | sha256sum -c - && dpkg-deb -x "$D" "$TOOLS/libnuma/root"; } >>"$OUT/import_check.txt" 2>&1
    export LD_LIBRARY_PATH="$(dirname "$(find "$TOOLS/libnuma/root" -name 'libnuma.so.1*' | head -1)"):${LD_LIBRARY_PATH:-}"
  fi
  "$PY" -c 'import torch; import sgl_kernel; print("sgl_kernel ok")' >>"$OUT/import_check.txt" 2>&1 || { echo "sgl_kernel import failed"; exit 3; }
fi
echo "ld=${LD_LIBRARY_PATH:-}" >>"$OUT/import_check.txt"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader >"$OUT/gpu.txt"

# Conditions rotate within each round so drift cannot line up with one condition.
for r in $(seq 1 "$RUNS"); do
  timeout 300 "$PY" "$LOAD" --out "$OUT/plain_r$r.json" >"$OUT/plain_r$r.log" 2>&1
  for mode in graph node; do
    timeout 600 "$NSYS" profile -t cuda,nvtx --cuda-graph-trace="$mode" --force-overwrite=true \
      -o "$OUT/${mode}_r$r" "$PY" "$LOAD" --out "$OUT/${mode}_r$r.json" >"$OUT/${mode}_r$r.log" 2>&1
    timeout 300 "$NSYS" export --type sqlite --force-overwrite=true -o "$OUT/${mode}_r$r.sqlite" "$OUT/${mode}_r$r.nsys-rep" >>"$OUT/${mode}_r$r.log" 2>&1
    "$PY" "$HERE/gpu_obs_green_score.py" "$OUT/${mode}_r$r.sqlite" "$OUT/${mode}_r$r.json" "$mode" >"$OUT/${mode}_r$r.score.json" 2>"$OUT/${mode}_r$r.score.err"
  done
done
echo done >"$OUT/DONE"
