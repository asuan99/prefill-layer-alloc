#!/bin/bash
# multigpu_probe.sh -- M1 multi-GPU substrate probe (DRAFT 2026-10-12, never run on a GPU).
#
# WHAT.  reports/multigpu_scaling_design_2026-10-12.md sec 1 + sec 6 (M1): what does an A100 xN
# VESSL node look like?  Facts only -- NO serving performance number, NO verdict, NO substrate-
# equivalence statement.  Interconnect numbers (all-reduce, P2P copy) are substrate descriptors.
#   F1 topo      nvidia-smi -L / -q / topo -m / topo -p2p / nvlink -s, per-GPU uuid, clocks, power,
#                SM count (torch + NVML), cgroup cpu.max/cpu.stat, nproc, NUMA, NCCL env NAMES set
#   F2 nccl      NCCL version (torch); nccl-tests all_reduce_perf for 2/4/8 GPUs if on PATH, else
#                torch.distributed NCCL all-reduce sweep (helper ar-bench)
#   F3 p2p       can_device_access_peer matrix + ordered-pair copy GB/s (helper p2p) + small-message
#                copy latency per ordered pair (helper p2p-lat; AMD-track V0 descriptor, NOT the V1 crossing
#                micro-benchmark -- eager, no cudagraph, no B grid)
#   F4 greenctx  per GPU id (one process each): sgl_kernel.spatial.create_greenctx_stream_by_value(p,d,gpu_id)
#                for (16,92),(44,64),(92,16) + driver read-out.  created != realized: a per-GPU %smid
#                census is NOT done here.
#   F5 rdma      AMD-track V0: RDMA NIC presence (/sys/class/infiniband, ibv_devinfo), net link names,
#                mooncake importability (PD-disagg transfer path prerequisite)
#   TODO block   per-GPU deviation of the SAME 1-GPU serving cell on each GPU -- needs a pre-registration
#                (cell, rotation order, n, metric); deliberately not implemented in this draft.
#
# RUN MODES
#   Job   : launch.sh --campaign multigpu_probe --target workspace/engine-port/scripts/vessl/multigpu_probe.sh
#           (job_entry.sh runs it with cwd=project; output goes under results/multigpu_probe/<job>/ which
#            job_entry exports, hashes into SHA256SUMS and marks DONE; fetch.sh merges it back.)
#           ★ launch.sh uploads a git bundle of HEAD: this script must be COMMITTED before --submit.
#   Local : bash multigpu_probe.sh --dry-run   (no GPU; CPU facts + parser self-checks; any machine)
#   Manual (Workspace): bash multigpu_probe.sh [--out-root DIR]
# ENV   PDMUX_JOB_NAME (set by job_entry) names the run dir; PDMUX_MP_NS="2 4 8" GPU counts for all-reduce
#       (counts above the visible GPU count are skipped and recorded); PDMUX_MP_TIMEOUT per-step seconds.
# NEVER dumps the environment (tokens); only NCCL_* variable NAMES+values are recorded.
set -uo pipefail

DRY=0; OUT_ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --out-root) OUT_ROOT="${2:?}"; shift ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="$(cd "${SELF_DIR}/../.." && pwd)"
HELPER="${SELF_DIR}/multigpu_probe_helper.py"
export PATH="/opt/conda/bin:${PATH}"
export PYTHONUNBUFFERED=1 LANG=C.UTF-8
case ":${LD_LIBRARY_PATH:-}:" in
  *:/usr/local/cuda/lib64:*) ;;
  *) export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac
STEP_TO="${PDMUX_MP_TIMEOUT:-600}"
NS="${PDMUX_MP_NS:-2 4 8}"
UTC="$(date -u +%Y%m%dT%H%M%SZ)"
NAME="${PDMUX_JOB_NAME:-manual_${UTC}}"
[[ ${DRY} == 1 ]] && NAME="${NAME}_dryrun"
if [[ -z "${OUT_ROOT}" ]]; then
  if [[ ${DRY} == 1 ]]; then OUT_ROOT="${TMPDIR:-/tmp}/multigpu_probe"; else OUT_ROOT="${ENGINE}/results/multigpu_probe"; fi
fi
RUN="${OUT_ROOT}/${NAME}"
mkdir -p "${RUN}"/{meta,logs,status,f1,f2,f3,f4,f5} || { echo "cannot create ${RUN}" >&2; exit 2; }
echo "run dir: ${RUN}"
export TRITON_CACHE_DIR="${RUN}/cache/triton"; mkdir -p "${TRITON_CACHE_DIR}"

set_status() {   # stage status reason
  python3 - "${RUN}/status/$1.json" "$1" "$2" "$3" <<'PY'
import json, sys
p, stage, st, why = sys.argv[1:5]
json.dump({"stage": stage, "status": st, "reason": why}, open(p, "w"), indent=2)
print(f"{stage}: {st} -- {why}")
PY
}
run_t() { local secs="$1" log="$2"; shift 2
  echo "+ $* (timeout ${secs}s)" >> "${RUN}/logs/commands.txt"
  timeout -k 20 "${secs}" "$@" > "${log}" 2>&1; }

clk_pid=""
finish() {
  local rc=$?
  [[ -n "${clk_pid}" ]] && kill "${clk_pid}" 2>/dev/null
  python3 "${HELPER}" summary --rundir "${RUN}" > "${RUN}/logs/summary.log" 2>&1 || echo "summary failed"
  rm -rf "${RUN}/cache"
  (cd "${RUN}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS)
  [[ -f "${RUN}/SUMMARY.md" ]] && cat "${RUN}/SUMMARY.md"
  echo "run dir: ${RUN}"
  exit "${rc}"
}
trap finish EXIT
trap 'exit 130' INT TERM

# ---- meta (no env dump)
python3 - "${RUN}/meta/run.json" "${DRY}" "${NAME}" <<'PY'
import json, os, platform, sys, time
out, dry, name = sys.argv[1:4]
json.dump({"mode": "dry-run" if dry == "1" else "gpu", "run": name, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": platform.node(), "spec": os.environ.get("PDMUX_SPEC", ""), "commit": os.environ.get("PDMUX_COMMIT", ""),
           "ns_requested": os.environ.get("PDMUX_MP_NS", "2 4 8")}, open(out, "w"), indent=2)
PY

# ================================================================== F1 topology / machine facts
have_smi=0; command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > "${RUN}/f1/nvidia_smi_L.txt" 2>&1 && have_smi=1
NGPU=0
{ echo "nproc=$(nproc)"; echo "cpu.max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"; cat /sys/fs/cgroup/cpu.stat 2>/dev/null
  echo "# memory.max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)"; } > "${RUN}/f1/cgroup_cpu_start.txt"
lscpu > "${RUN}/f1/lscpu.txt" 2>&1
command -v numactl > /dev/null 2>&1 && numactl -H > "${RUN}/f1/numactl_H.txt" 2>&1
env | grep -E '^NCCL_' > "${RUN}/f1/nccl_env.txt" 2>/dev/null || echo "(no NCCL_* set)" > "${RUN}/f1/nccl_env.txt"
if [[ ${have_smi} == 1 ]]; then
  NGPU=$(grep -c '^GPU ' "${RUN}/f1/nvidia_smi_L.txt")
  nvidia-smi > "${RUN}/f1/nvidia_smi.txt" 2>&1
  nvidia-smi -q > "${RUN}/f1/nvidia_smi_q.txt" 2>&1
  nvidia-smi topo -m > "${RUN}/f1/topo_m.txt" 2>&1
  nvidia-smi topo -p2p r > "${RUN}/f1/topo_p2p_read.txt" 2>&1
  nvidia-smi topo -p2p n > "${RUN}/f1/topo_p2p_nvlink.txt" 2>&1
  nvidia-smi nvlink -s > "${RUN}/f1/nvlink_s.txt" 2>&1
  nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,compute_cap,memory.total,clocks.max.sm,power.limit,mig.mode.current,compute_mode \
    --format=csv > "${RUN}/f1/gpu.csv" 2>&1
  # all-GPU clock/throttle trace (job_entry's meta/clk.csv is the same query; this one is kept inside the run dir)
  nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv -lms 200 > "${RUN}/f1/clk_all.csv" 2> "${RUN}/f1/clk_all.err" &
  clk_pid=$!
  python3 "${HELPER}" parse-topo --inp "${RUN}/f1/topo_m.txt" --out "${RUN}/f1/topo_m.json" > "${RUN}/logs/parse_topo.log" 2>&1
  run_t 300 "${RUN}/logs/gpu_facts.log" python3 "${HELPER}" gpu-facts --out "${RUN}/f1/gpu_facts.json"
fi
if [[ ${DRY} == 1 ]]; then
  # parser self-check on a canned 2-GPU matrix (no GPU needed)
  printf '\tGPU0\tGPU1\tCPU Affinity\nGPU0\t X \tNV12\t0-21\nGPU1\tNV12\t X \t0-21\n' > "${RUN}/f1/topo_selfcheck.txt"
  python3 "${HELPER}" parse-topo --inp "${RUN}/f1/topo_selfcheck.txt" --out "${RUN}/f1/topo_selfcheck.json" \
    && grep -q '"all_pairs_nvlink": true' "${RUN}/f1/topo_selfcheck.json" \
    && set_status F1 SKIPPED "dry-run: CPU facts + topo parser self-check only (GPUs visible: ${NGPU})" \
    || set_status F1 FAIL "dry-run: topo parser self-check failed"
elif [[ ${have_smi} == 0 ]]; then set_status F1 FAIL "nvidia-smi unavailable"
elif [[ ${NGPU} -lt 1 ]]; then set_status F1 FAIL "no GPU listed"
elif [[ ! -s "${RUN}/f1/gpu_facts.json" ]]; then set_status F1 UNRESOLVED "torch/NVML gpu-facts failed (logs/gpu_facts.log); nvidia-smi facts recorded"
else set_status F1 PASS "GPUs visible: ${NGPU}; classes: $(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("classes"))' "${RUN}/f1/topo_m.json" 2>/dev/null)"
fi

if [[ ${DRY} == 1 ]]; then
  for s in F2 F3 F4 F5; do set_status "$s" SKIPPED "dry-run: no GPU work"; done
  echo "GPU commands that would run: all-reduce for N in [${NS}] (nccl-tests if all_reduce_perf on PATH else helper ar-bench); helper p2p; helper greenctx-one --gpu 0..N-1"
  grep -q FAIL "${RUN}"/status/*.json && exit 1
  exit 0
fi
[[ ${NGPU} -ge 1 ]] || exit 1

# ================================================================== F2 NCCL / all-reduce
python3 -c 'import torch;print("nccl", ".".join(map(str, torch.cuda.nccl.version())))' > "${RUN}/f2/nccl_version.txt" 2>&1
f2_bad=""; f2_done=""
for n in ${NS}; do
  if (( n > NGPU )); then echo "skipped: N=${n} > visible ${NGPU}" >> "${RUN}/f2/skipped.txt"; continue; fi
  if command -v all_reduce_perf > /dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n - 1))) run_t "${STEP_TO}" "${RUN}/f2/nccltests_ar_${n}.txt" \
      all_reduce_perf -b 8 -e 1G -f 2 -g "${n}" -c 1 \
      && python3 "${HELPER}" parse-nccl --inp "${RUN}/f2/nccltests_ar_${n}.txt" --out "${RUN}/f2/nccltests_ar_${n}.json" \
      && f2_done+=" ${n}(nccl-tests)" || f2_bad+=" ${n}"
  else
    run_t "${STEP_TO}" "${RUN}/f2/arbench_${n}.log" python3 "${HELPER}" ar-bench --n "${n}" --out "${RUN}/f2/arbench_${n}.json" \
      && f2_done+=" ${n}(torch.distributed)" || f2_bad+=" ${n}"
  fi
done
if [[ -n "${f2_bad}" ]]; then set_status F2 FAIL "all-reduce failed for N=${f2_bad}; done:${f2_done}"
elif [[ -z "${f2_done}" ]]; then set_status F2 UNRESOLVED "no N in [${NS}] <= ${NGPU} visible GPUs"
else set_status F2 PASS "all-reduce measured:${f2_done} (descriptor, see f2/)"
fi

# ================================================================== F3 P2P
if (( NGPU >= 2 )); then
  f3_bad=""
  run_t "${STEP_TO}" "${RUN}/logs/p2p.log" python3 "${HELPER}" p2p --out "${RUN}/f3/p2p.json" || f3_bad+=" p2p"
  run_t "${STEP_TO}" "${RUN}/logs/p2p_lat.log" python3 "${HELPER}" p2p-lat --out "${RUN}/f3/p2p_lat.json" || f3_bad+=" p2p-lat"
  if [[ -z "${f3_bad}" ]]; then
    set_status F3 PASS "peer-access matrix + copy GB/s + small-message latency recorded (f3/)"
  else set_status F3 FAIL "failed:${f3_bad} (logs/p2p*.log)"; fi
else set_status F3 SKIPPED "1 GPU visible"; fi

# ================================================================== F4 green context per GPU
g_ok=0; g_bad=""
for ((g = 0; g < NGPU; g++)); do
  if run_t 180 "${RUN}/f4/gctx_${g}.log" python3 "${HELPER}" greenctx-one --gpu "${g}" --out "${RUN}/f4/gctx_${g}.json" \
     && python3 -c 'import json,sys;r=json.load(open(sys.argv[1]));sys.exit(0 if all(p["created"] for p in r["pairs"]) else 1)' "${RUN}/f4/gctx_${g}.json"; then
    g_ok=$((g_ok + 1)); else g_bad+=" ${g}"; fi
done
if [[ -z "${g_bad}" ]]; then set_status F4 PASS "all pairs created on all ${g_ok}/${NGPU} GPUs (created != realized; census TODO)"
else set_status F4 FAIL "creation failed on GPU(s):${g_bad} (f4/gctx_<g>.log)"; fi

# ================================================================== F5 RDMA / mooncake (AMD-track V0)
{ echo "# /sys/class/infiniband"; ls -l /sys/class/infiniband 2>&1
  echo "# ibv_devinfo"; if command -v ibv_devinfo > /dev/null 2>&1; then ibv_devinfo 2>&1; else echo "(ibv_devinfo not on PATH)"; fi
  echo "# net links"; ls /sys/class/net 2>&1
  echo "# rdma link"; if command -v rdma > /dev/null 2>&1; then rdma link 2>&1; else echo "(rdma tool not on PATH)"; fi
} > "${RUN}/f5/rdma.txt"
python3 -c 'import importlib.util as u;print("mooncake", "present" if u.find_spec("mooncake") else "absent")' > "${RUN}/f5/mooncake.txt" 2>&1
n_ib=$(ls /sys/class/infiniband 2>/dev/null | wc -l)
set_status F5 PASS "infiniband devices: ${n_ib}; $(cat "${RUN}/f5/mooncake.txt" | tail -1) (descriptor; RDMA usability not tested)"

# ================================================================== TODO (pre-registration required)
# TODO(M-step prereg): run the SAME 1-GPU serving cell on each GPU 0..N-1 in a pre-registered rotation
#   (cell, order, n per GPU, metric) to measure between-GPU deviation.  Do NOT implement before the
#   pre-registration is audited; gate 3 (n>=4) and "same Job for all arms" apply.  Intentionally empty.

{ echo "# at end $(date -u +%H:%M:%S)"; cat /sys/fs/cgroup/cpu.stat 2>/dev/null; } > "${RUN}/f1/cgroup_cpu_end.txt"
grep -lq '"status": "FAIL"' "${RUN}"/status/*.json 2>/dev/null && exit 1
exit 0
