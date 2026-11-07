#!/usr/bin/bash
# GPU observability/control probe for a VESSL Workspace. Diagnostics and user-space install only:
# no sudo, no module reload, no retries or workarounds of denied control calls.
# Usage: bash gpu_obs_probe.sh <out_dir> [tools_dir]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?out_dir}
TOOLS=${2:-/data/tools/nsight}
PY=${PY:-/opt/conda/bin/python}
export PATH=/opt/conda/bin:$PATH
mkdir -p "$OUT" "$TOOLS"
INDEX=$OUT/INDEX.tsv
: >"$INDEX"

# Captures stdout+stderr verbatim to <label>.txt; the rc goes to INDEX.tsv. A failure never stops the probe.
run() {
  local label=$1 limit=$2; shift 2
  { echo "\$ $*"; timeout "$limit" "$@"; } >"$OUT/$label.txt" 2>&1
  local rc=$?
  printf '%s\t%s\t%s\n' "$label" "$rc" "$*" >>"$INDEX"
}

# ---- 1. permissions ----
grep -E 'Cap(Eff|Prm|Bnd)|NoNewPrivs|Seccomp:' /proc/self/status >"$OUT/1_status.txt"
CAPEFF=$(awk '/CapEff/{print $2}' /proc/self/status)
if command -v capsh >/dev/null; then
  run 1_capsh 20 capsh --decode="$CAPEFF"
else
  echo "capsh absent; decoded from the CapEff bits (bit 19 SYS_PTRACE, 21 SYS_ADMIN)" >"$OUT/1_capsh.txt"
fi
"$PY" -c "v=int('$CAPEFF',16);print('CapEff',hex(v),'SYS_PTRACE',bool(v>>19&1),'SYS_ADMIN',bool(v>>21&1),'PERFMON',bool(v>>38&1))" >>"$OUT/1_capsh.txt" 2>&1
run 1_rmprofiling 20 grep -i RmProfilingAdminOnly /proc/driver/nvidia/params
run 1_perf_paranoid 20 cat /proc/sys/kernel/perf_event_paranoid
run 1_smi_power_clock 60 nvidia-smi -q -d POWER,CLOCK
run 1_smi_list 30 nvidia-smi --query-gpu=name,uuid,driver_version,power.limit,power.default_limit,power.min_limit,power.max_limit --format=csv
run 1_smi_header 30 nvidia-smi
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)

# ---- user-space install, only for what the image lacks. Driver 580 = CUDA 13.0 toolkit vintage ----
IMG_NSYS=$(command -v nsys || true); IMG_NCU=$(command -v ncu || true)
echo "driver=$DRIVER image_nsys=${IMG_NSYS:-none} image_ncu=${IMG_NCU:-none}" >"$OUT/1_install.txt"
REPO=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64
install_deb() {
  local deb=$1 sha=$2 d="$TOOLS/dl"
  mkdir -p "$d"
  [ -f "$d/$deb" ] || curl -fsSL -m 900 -o "$d/$deb" "$REPO/$deb" || { echo "download failed: $deb"; return 1; }
  echo "$sha  $d/$deb" | sha256sum -c - || return 1
  command -v dpkg-deb >/dev/null || { echo "dpkg-deb absent"; return 1; }
  dpkg-deb -x "$d/$deb" "$TOOLS/root"
}
pkg_sha() { curl -fsSL -m 60 "$REPO/Packages.gz" | gunzip | awk -v f="./$1" '$1=="Filename:"{m=($2==f)} m&&$1=="SHA256:"{print $2;exit}'; }
NSYS_DEB=nsight-systems-2025.5.2_2025.5.2.266-1_amd64.deb
NCU_DEB=nsight-compute-2025.3.1_2025.3.1.4-1_amd64.deb
if [ "${DRIVER%%.*}" != 580 ]; then
  echo "driver is not 580: pinned Nsight versions unverified for it; no install" >>"$OUT/1_install.txt"
else
  [ -n "$IMG_NSYS" ] || install_deb "$NSYS_DEB" "$(pkg_sha "$NSYS_DEB")" >>"$OUT/1_install.txt" 2>&1
  [ -n "$IMG_NCU" ] || install_deb "$NCU_DEB" "$(pkg_sha "$NCU_DEB")" >>"$OUT/1_install.txt" 2>&1
fi
NSYS=${IMG_NSYS:-$(find "$TOOLS/root" -type f -path '*target-linux-x64/nsys' 2>/dev/null | head -1)}
NCU=${IMG_NCU:-$(find "$TOOLS/root" -type f -name ncu -path '*target*x86_64*' 2>/dev/null | head -1)}
[ -n "$NCU" ] || NCU=$(find "$TOOLS/root" \( -type f -o -type l \) -name ncu 2>/dev/null | head -1)
echo "nsys=${NSYS:-none} ncu=${NCU:-none}" | tee -a "$OUT/1_install.txt" >"$OUT/tool_paths.txt"
[ -n "$NSYS" ] && run 1_nsys_version 30 "$NSYS" --version
[ -n "$NCU" ] && run 1_ncu_version 30 "$NCU" --version
[ -n "$NSYS" ] && run 1_nsys_status 60 "$NSYS" status -e
[ -n "$NSYS" ] && run 1_nsys_gpu_metrics_help 60 "$NSYS" profile --gpu-metrics-devices=help

# ---- 2. per-feature tests, same matmul load ----
MM="$HERE/gpu_obs_matmul.py"
if [ -n "$NSYS" ]; then
  run 2a_nsys_trace 300 "$NSYS" profile -t cuda,nvtx,osrt --force-overwrite=true -o "$OUT/2a_trace" "$PY" "$MM"
  run 2a_nsys_stats 180 "$NSYS" stats --force-export=true --report cuda_gpu_kern_sum --report cuda_api_sum --report nvtx_sum "$OUT/2a_trace.nsys-rep"
  [ -f "$OUT/2a_trace.sqlite" ] && run 2a_sqlite 60 "$PY" "$HERE/gpu_obs_sqlite.py" "$OUT/2a_trace.sqlite"
  run 2b_nsys_gpu_metrics 300 "$NSYS" profile --gpu-metrics-devices=0 --force-overwrite=true -o "$OUT/2b_metrics" "$PY" "$MM" --iters 100
  if [ -f "$OUT/2b_metrics.nsys-rep" ]; then
    run 2b_nsys_export 180 "$NSYS" export --type sqlite --force-overwrite=true -o "$OUT/2b_metrics.sqlite" "$OUT/2b_metrics.nsys-rep"
    [ -f "$OUT/2b_metrics.sqlite" ] && run 2b_sqlite 60 "$PY" "$HERE/gpu_obs_sqlite.py" "$OUT/2b_metrics.sqlite"
  fi
fi
[ -n "$NCU" ] && run 2c_ncu_speedoflight 300 "$NCU" --section SpeedOfLight -f -o "$OUT/2c_ncu" "$PY" "$MM" --iters 4 --size 1024
run 2d_torch_profiler 300 "$PY" "$MM" --iters 50 --torch-profiler "$OUT/2d_torch_trace.json"

nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem --format=csv -lms 100 >"$OUT/2e_smi_100ms.csv" 2>"$OUT/2e_smi_100ms.err" &
SMI=$!
run 2e_load_for_smi 300 "$PY" "$MM" --iters 100
kill "$SMI" 2>/dev/null; wait "$SMI" 2>/dev/null
printf '2e_smi_poll\t%s\t%s\n' "$(($(wc -l <"$OUT/2e_smi_100ms.csv") - 1)) rows" "nvidia-smi --query-gpu=... -lms 100" >>"$INDEX"

# Control calls, one attempt each. Values are no-ops where possible, so a surprise success changes nothing:
# -pl uses the current limit; only if -lgc is accepted is it undone once with -rgc.
LIMIT=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
run 2f_smi_pl 30 nvidia-smi -i 0 -pl "$LIMIT"
run 2f_smi_lgc 30 nvidia-smi -i 0 -lgc 1410,1410
if [ "$(awk -F'\t' '$1=="2f_smi_lgc"{print $2}' "$INDEX")" = 0 ]; then
  run 2f_smi_rgc_restore 30 nvidia-smi -i 0 -rgc
fi
echo done >"$OUT/DONE"
