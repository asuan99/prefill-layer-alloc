#!/bin/bash
# substrate_probe.sh -- VESSL A100 Workspace substrate check, B0 + B3 + B4 in ONE run.
#
# WHAT.  handoff-report/gpu_rental_checklist_2026-09-18.md sec 3 (B0, B3, B4) as one script for
# the A100 Workspace `pdmux-probe` (handoff-report/vessl_operating_model_2026-10-02.md sec 2, sec 7
# step 5).  Every stage writes status/<stage>.json = PASS | FAIL | UNRESOLVED | SKIPPED with the
# mechanical facts behind it; SUMMARY.md tabulates them.
#   ★ NO PERFORMANCE NUMBER is produced or judged (no latency/throughput/goodput anywhere).
#   ★ NOT a substrate-equivalence verdict: KISTI artefacts are listed beside this run's values,
#     whether the substrates count as "the same" is a user decision (CLAUDE.md gate 1 extension).
#   ★ Workspace = bring-up/probe only (operating model sec 1-1).  Nothing here is a campaign.
#
# STAGES (strict order; a stage that is not PASS skips every later stage that depends on it --
#         override with --keep-going, which is recorded in meta/run.json):
#   PRE    engine install manifest present + `sha256sum -c` + manual edits applied + install commit
#          == checked-out HEAD + thread-local role patch importable.  FAIL -> stop.
#   B0     nvidia-smi machine facts, nvcc, torch/triton/sgl_kernel/flashinfer versions,
#          sgl_kernel.spatial.get_sm_available(0).  Gate: one A100, cc 8.0, MIG off, 108 SM.
#   B3-R0  results/smid_census/smid_l0_census.py verbatim (selftest-cpu, selftest-analyzer, run,
#          analyze) -- %smid label consistency; premise of every set statement below and of P0-A
#          (PREREG_P0A sec9-2).  KISTI reference: smid_l0_verdict_889631.json.
#   B3-plain / B3a / B3b  (substrate_probe_helper.py b3-grid): plain census D (= the engine's
#          (108,0)/(0,108) groups, which are plain streams) before/after green creation; every grid
#          pair of benchmarks/configs/pdmux_r2.yml + pdmux_homog5.yml created in ONE process;
#          B3a = |%smid set| == requested on both halves, disjoint, inside D;
#          B3b = both halves launched concurrently: time overlap > 0, no shared label, no escape.
#   B3-gran  odd/small requests, one process each -> table requested -> realized (descriptive).
#   B3c    results/bcg_probe/p0a_graph_sm_confinement.py verbatim (selftest, run, analyze):
#          cudagraph capture+replay on the green decode stream keeps the label set.
#          KISTI reference: p0a_verdict_890893.json.
#   B4     Nemotron-Nano-9B-v2-Base PD-mux boot, cudagraph ON (no --disable-cuda-graph), argv =
#          e2_sticky.sbatch:221-228 OFF arm -> /health -> 6 greedy prompts sequentially then
#          concurrently (e2_sticky.sbatch:288-299) -> teardown.  B4-green = in-server driver
#          read-out (PDMUX_GREEN_READOUT, observation only) vs the config targets.
#
# ESTIMATED GPU WALL (A100 Workspace billed while running; $1.48/h, vessl.env):
#   PRE 0.5 min | B0 1 | B3-R0 2-3 | grid 1-2 | gran 2-4 (15 processes) | P0-A 4-6 (1.5 of it is
#   the CPU selftest) | B4 8-20 (18 GB load from /data, flashinfer JIT on a cold per-run cache,
#   cudagraph capture for 5 stream groups) => typical 20-35 min, target <= 45 min.
#   Hard stops: per-step `timeout`, plus --budget-min (default 60) after which remaining stages are
#   SKIPPED.  Pause the Workspace right after (operating model sec 2).
#
# PREREQUISITE (once per Workspace start; the container's /opt/pdmux is not persistent):
#   export PATH=/opt/conda/bin:$PATH PDMUX_ROOT=/opt/pdmux SGLANG_ENGINE_DEV=/opt/pdmux/sglang_engine_dev/python
#   PDMUX_PROJECT_ROOT=/opt/pdmux/prefill-layer-alloc bash \
#     /opt/pdmux/prefill-layer-alloc/workspace/engine-port/scripts/vessl/install_runtime.sh \
#     /opt/pdmux/runtime_source_manifest.sha256
#
# USAGE
#   bash substrate_probe.sh [--dry-run] [--manifest PATH] [--out-root DIR] [--skip-b4]
#                           [--keep-going] [--budget-min N]
#   --dry-run   no GPU work: PRE, imports, config/argv parsing, CPU selftests of R0/P0-A, model
#               presence; prints the GPU commands.  Usable in the image without a GPU.
# OUTPUT  <out-root>/substrate_probe_<UTC>/{SUMMARY.md, summary.json, status/, meta/, b0/, b3/,
#         b4/, logs/, SHA256SUMS}.  meta/clk.csv = 100 ms clock/throttle trace (VESSL_A100_CONTEXT
#         sec 5-1).  The environment is NEVER dumped (VESSLCTL/HF tokens may be in it).
set -uo pipefail

# ------------------------------------------------------------------ args
DRY=0; SKIP_B4=0; KEEP_GOING=0; BUDGET_MIN=60
MANIFEST="${PDMUX_RUNTIME_MANIFEST:-/opt/pdmux/runtime_source_manifest.sha256}"
OUT_ROOT="/data/runs"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --skip-b4) SKIP_B4=1 ;;
    --keep-going) KEEP_GOING=1 ;;
    --manifest) MANIFEST="${2:?}"; shift ;;
    --out-root) OUT_ROOT="${2:?}"; shift ;;
    --budget-min) BUDGET_MIN="${2:?}"; shift ;;
    -h|--help) sed -n '2,60p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

# ------------------------------------------------------------------ env (ssh does not inherit image ENV)
export PATH="/opt/conda/bin:${PATH}"
export PDMUX_ROOT="${PDMUX_ROOT:-/opt/pdmux}"
export SGLANG_ENGINE_DEV="${SGLANG_ENGINE_DEV:-/opt/pdmux/sglang_engine_dev/python}"
export HF_HOME="${HF_HOME:-/data/hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 LANG=C.UTF-8
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
case ":${LD_LIBRARY_PATH:-}:" in
  *:/usr/local/cuda/lib64:*) ;;
  *) export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="$(cd "${SELF_DIR}/../.." && pwd)"
PROJECT="$(cd "${ENGINE}/../.." && pwd)"
export PDMUX_PROJECT_ROOT="${PROJECT}"
HELPER="${SELF_DIR}/substrate_probe_helper.py"
R0_TOOL="${ENGINE}/results/smid_census/smid_l0_census.py"
P0A_TOOL="${ENGINE}/results/bcg_probe/p0a_graph_sm_confinement.py"
GRID_CFGS="${ENGINE}/benchmarks/configs/pdmux_r2.yml,${ENGINE}/results/longctx_conflict/probes/pdmux_homog5.yml"

# B4 registered control factors -- results/r2_eval/e2_sticky_prereg/e2_sticky.sbatch:72-79 (OFF arm)
MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
BACKEND="flashinfer"
CFG="${ENGINE}/results/longctx_conflict/probes/pdmux_homog5.yml"
CTX=16384; MEM_FRACTION=0.82; MAX_RUNNING=48; SERVER_SEED=1; FIXED_DSM=44
B4_BOOT_TIMEOUT="${PDMUX_PROBE_BOOT_TIMEOUT:-1500}"

UTC="$(date -u +%Y%m%dT%H%M%SZ)"
TAG="vessl${UTC}"
if [[ ${DRY} == 1 && ! -d "${OUT_ROOT}" ]]; then
  OUT_ROOT="${TMPDIR:-/tmp}/substrate_probe_dryrun"
fi
RUN="${OUT_ROOT}/substrate_probe_${UTC}$([[ ${DRY} == 1 ]] && echo _dryrun)"
mkdir -p "${RUN}"/{meta,logs,status,b0,b3,b4,cache/triton} || { echo "cannot create ${RUN}" >&2; exit 2; }
export TRITON_CACHE_DIR="${RUN}/cache/triton"        # per-run JIT cache (job_entry.sh:150 convention)
echo "run dir: ${RUN}"

# ------------------------------------------------------------------ machinery
declare -A T0
clk_pid=""; srv_pid=""; srv_port=""
now() { date +%s.%N; }
stage_begin() { T0[$1]="$(now)"; echo "=== [$(date -u +%H:%M:%S)] $1 ==="; }
stage_end() { printf '%s\t%s\t%s\n' "$1" "${T0[$1]}" "$(now)" >> "${RUN}/meta/stage_times.tsv"; }
set_status() {   # stage status reason
  python3 - "${RUN}/status/$1.json" "$1" "$2" "$3" <<'PY'
import json, sys
p, stage, st, why = sys.argv[1:5]
json.dump({"stage": stage, "status": st, "reason": why, "facts": {}}, open(p, "w"), indent=2)
print(f"{stage}: {st} -- {why}")
PY
}
status_of() {
  python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1]))["status"])
except Exception: print("MISSING")' "${RUN}/status/$1.json"
}
all_pass() {     # all named stages PASS (or --keep-going)
  local s
  [[ ${KEEP_GOING} == 1 ]] && return 0
  for s in "$@"; do [[ "$(status_of "$s")" == PASS ]] || return 1; done
  return 0
}
over_budget() { (( SECONDS > BUDGET_MIN * 60 )); }
skip() { set_status "$1" SKIPPED "$2"; }
run_t() {        # secs logfile cmd...  -> rc (124 = timeout)
  local secs="$1" log="$2"; shift 2
  echo "+ $* (timeout ${secs}s) > ${log#${RUN}/}" >> "${RUN}/logs/commands.txt"
  timeout -k 30 "${secs}" "$@" > "${log}" 2>&1
}

stop_server() {
  [[ -z "${srv_pid}" ]] && return 0
  kill -TERM -- "-${srv_pid}" 2>/dev/null || kill -TERM "${srv_pid}" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "${srv_pid}" 2>/dev/null || break; sleep 2; done
  kill -KILL -- "-${srv_pid}" 2>/dev/null
  [[ -n "${srv_port}" ]] && pkill -9 -f "launch_server.*--port ${srv_port}" 2>/dev/null
  srv_pid=""
}

finish() {
  local rc=$?
  stop_server
  [[ -n "${clk_pid}" ]] && kill "${clk_pid}" 2>/dev/null
  printf 'ALL\t%s\t%s\n' "${RUN_T0}" "$(now)" >> "${RUN}/meta/stage_times.tsv"
  python3 "${HELPER}" summary --rundir "${RUN}" > "${RUN}/logs/summary.log" 2>&1 \
    || echo "summary failed (see logs/summary.log)"
  (cd "${RUN}" && find . -type f ! -path './cache/*' ! -name SHA256SUMS -print0 | sort -z \
     | xargs -0 -r sha256sum > SHA256SUMS)
  echo ""
  [[ -f "${RUN}/SUMMARY.md" ]] && cat "${RUN}/SUMMARY.md"
  echo "run dir: ${RUN}"
  exit "${rc}"
}
RUN_T0="$(now)"
trap finish EXIT
trap 'exit 130' INT TERM

# ------------------------------------------------------------------ meta (no env dump)
python3 - "${RUN}/meta/run.json" "${DRY}" "${MANIFEST}" "${PROJECT}" "${KEEP_GOING}" "${SKIP_B4}" \
  "${BUDGET_MIN}" "${PDMUX_ROOT}" <<'PY'
import hashlib, json, os, platform, subprocess, sys, time
out, dry, man, proj, keep, skipb4, budget, opt = sys.argv[1:9]
def sh(*a):
    try: return subprocess.run(a, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return f"ERROR {e!r}"
def sha(p):
    try: return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except OSError: return None
rec = {"mode": "dry-run" if dry == "1" else "gpu", "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "host": platform.node(), "commit": sh("git", "-C", proj, "rev-parse", "HEAD"),
       "git_dirty": bool(sh("git", "-C", proj, "status", "--porcelain", "--untracked-files=no")),
       "manifest": man, "manifest_sha256": sha(man), "keep_going": keep == "1",
       "skip_b4": skipb4 == "1", "budget_min": int(budget),
       "image_pip_freeze_sha256": sha(os.path.join(opt, "pip_freeze.txt")),
       "image_python_torch_sha256": sha(os.path.join(opt, "python_torch.txt")),
       "script_sha256": {n: sha(os.path.join(proj, "workspace/engine-port/scripts/vessl", n))
                         for n in ("substrate_probe.sh", "substrate_probe_helper.py")}}
json.dump(rec, open(out, "w"), indent=2)
PY

# ================================================================== PRE
stage_begin PRE
pre_fail=""
inst="${MANIFEST}.install.txt"
if [[ ! -d "${SGLANG_ENGINE_DEV}/sglang" ]]; then
  pre_fail="SGLANG_ENGINE_DEV tree missing: ${SGLANG_ENGINE_DEV}"
elif [[ ! -s "${MANIFEST}" ]]; then
  pre_fail="install manifest missing: ${MANIFEST} (run install_runtime.sh <manifest> first; see header)"
elif ! sha256sum -c --quiet "${MANIFEST}" > "${RUN}/logs/pre_manifest_check.log" 2>&1; then
  pre_fail="installed tree does not match manifest (logs/pre_manifest_check.log)"
elif [[ ! -f "${inst}" ]]; then
  pre_fail="install record missing: ${inst}"
else
  cp "${MANIFEST}" "${inst}" "${RUN}/meta/"
  n_ent=$(grep -c . "${MANIFEST}")
  inst_commit=$(sed -n 's/^commit=//p' "${inst}")
  head_commit=$(git -C "${PROJECT}" rev-parse HEAD 2>/dev/null || echo unknown)
  if ! grep -Eq '^manual_edits_patch=(applied|already-applied) ' "${inst}"; then
    pre_fail="manual edits not recorded as applied in ${inst}"
  elif (( n_ent < 25 )) || ! grep -q "models/nemotron_h.py" "${MANIFEST}"; then
    pre_fail="manifest has ${n_ent} entries or lacks models/nemotron_h.py (expected >=25)"
  elif ! [[ "${head_commit}" =~ ^[0-9a-f]{40}$ ]]; then
    pre_fail="cannot read checkout HEAD (${head_commit}); git safe.directory?"
  elif [[ "${inst_commit}" != "${head_commit}" ]]; then
    pre_fail="installed from ${inst_commit}, checkout is ${head_commit} -- re-run install_runtime.sh"
  elif ! grep -q "def pdmux_role_is_thread_local" "${SGLANG_ENGINE_DEV}/sglang/srt/distributed/parallel_state.py"; then
    pre_fail="thread-local role patch not in installed parallel_state.py"
  elif ! run_t 300 "${RUN}/logs/pre_imports.log" python3 -c '
from sglang.srt.distributed.parallel_state import pdmux_role_is_thread_local
assert pdmux_role_is_thread_local() is True
import sglang.srt.multiplex.green_readout, sglang.srt.multiplex.pdmux_context
print("thread-local role patch importable; green_readout/pdmux_context import OK")'; then
    pre_fail="engine imports failed (logs/pre_imports.log)"
  else
    for f in "${HELPER}" "${R0_TOOL}" "${P0A_TOOL}" "${CFG}"; do
      [[ -f "$f" ]] || pre_fail="missing tool/config: $f"
    done
    python3 "${HELPER}" print-grid --configs "${GRID_CFGS}" > "${RUN}/meta/grid.json" 2> "${RUN}/logs/print_grid.err" \
      || pre_fail="grid configs did not parse (logs/print_grid.err)"
  fi
fi
if [[ -n "${pre_fail}" ]]; then set_status PRE FAIL "${pre_fail}"
else set_status PRE PASS "manifest verified (${n_ent} entries), manual edits applied, install commit == HEAD, thread-local patch importable"
fi
stage_end PRE
if [[ "$(status_of PRE)" != PASS ]]; then
  for s in B0 B3-R0 B3-plain B3a B3b B3-gran B3c B4 B4-green; do skip "$s" "PRE not PASS"; done
  exit 1
fi

# ================================================================== B0
stage_begin B0
have_smi=0; command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > "${RUN}/b0/nvidia_smi_L.txt" 2>&1 && have_smi=1
if [[ ${have_smi} == 1 ]]; then
  nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total,clocks.max.sm,power.limit,mig.mode.current,compute_mode \
    --format=csv,noheader > "${RUN}/b0/gpu.csv" 2>&1
  nvidia-smi > "${RUN}/b0/nvidia_smi.txt" 2>&1
  nvidia-smi -q -d CLOCK,POWER,COMPUTE,PERFORMANCE > "${RUN}/b0/nvidia_smi_q.txt" 2>&1
  nvidia-smi --query-gpu=memory.used --format=csv,noheader > "${RUN}/b0/memory_used_at_start.txt" 2>&1
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv > "${RUN}/b0/compute_apps_at_start.txt" 2>&1
  # clock / throttle trace (VESSL_A100_CONTEXT sec 5-1); falls back to the renamed field
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv -lms 100 > "${RUN}/meta/clk.csv" 2> "${RUN}/meta/clk.err" &
  clk_pid=$!; sleep 2
  if ! kill -0 "${clk_pid}" 2>/dev/null; then
    nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active \
      --format=csv -lms 100 > "${RUN}/meta/clk.csv" 2>> "${RUN}/meta/clk.err" &
    clk_pid=$!
  fi
fi
nvcc --version > "${RUN}/b0/nvcc.txt" 2>&1
df -h /data /io / > "${RUN}/b0/df.txt" 2>&1
{ grep -i "RmProfilingAdminOnly" /proc/driver/nvidia/params 2>/dev/null || echo "(field absent)"; } > "${RUN}/b0/profiling_param.txt"
if [[ -f /data/substrate_history.tsv && -s "${RUN}/b0/gpu.csv" ]]; then   # compare only, no append
  { echo "# last Job line in /data/substrate_history.tsv vs this Workspace (record only)";
    tail -n 1 /data/substrate_history.tsv; cat "${RUN}/b0/gpu.csv"; } > "${RUN}/b0/history_compare.txt"
fi
run_t 300 "${RUN}/logs/b0_python.log" python3 "${HELPER}" b0-python --out "${RUN}/b0/b0_python.json"
if [[ ${DRY} == 1 ]]; then
  skip B0 "dry-run: GPU facts not judged (b0_python.json still records the import path)"
else
  python3 "${HELPER}" score-b0 --gpu-csv "${RUN}/b0/gpu.csv" --py-json "${RUN}/b0/b0_python.json" \
    --status "${RUN}/status/B0.json"
fi
stage_end B0

# ================================================================== B3
r0_dir="${RUN}/b3/r0"; mkdir -p "${r0_dir}"
# ---- B3-R0: %smid label consistency (verbatim tool, same 4 stages as smid_l0_run.sbatch)
stage_begin B3-R0
if [[ ${DRY} == 0 ]] && ! all_pass B0; then skip B3-R0 "B0 not PASS"
elif over_budget; then skip B3-R0 "budget"
else
  run_t 300 "${r0_dir}/selftest_cpu.log" python3 "${R0_TOOL}" --selftest-cpu --tag "${TAG}" --outdir "${r0_dir}"; rc1=$?
  run_t 300 "${r0_dir}/selftest_analyzer.log" python3 "${R0_TOOL}" --selftest-analyzer; rc2=$?
  echo "selftest_cpu=${rc1} selftest_analyzer=${rc2}" > "${r0_dir}/rc.txt"
  if [[ ${rc1} != 0 || ${rc2} != 0 ]]; then
    set_status B3-R0 UNRESOLVED "R0 plumbing selftest failed (cpu=${rc1} analyzer=${rc2}) -- measurement condition, not a result"
  elif [[ ${DRY} == 1 ]]; then
    skip B3-R0 "dry-run: R0 CPU selftests passed; GPU census not run"
  else
    run_t 600 "${r0_dir}/run.log" python3 "${R0_TOOL}" --run --tag "${TAG}" --outdir "${r0_dir}"; rc3=$?
    rc4=-1
    if [[ ${rc3} == 0 ]]; then
      run_t 300 "${r0_dir}/analyze.log" python3 "${R0_TOOL}" --analyze "${r0_dir}/smid_l0_raw_${TAG}.json" \
        --tag "${TAG}" --outdir "${r0_dir}"; rc4=$?
    fi
    echo "run=${rc3} analyze=${rc4}" >> "${r0_dir}/rc.txt"
    python3 "${HELPER}" score-b3 --b3dir "${RUN}/b3" --status-dir "${RUN}/status" --tag "${TAG}" --parts r0
  fi
fi
stage_end B3-R0

# ---- B3-plain / B3a / B3b: grid pairs, realized vs requested, concurrent disjointness
stage_begin B3a
grid_dir="${RUN}/b3/grid"; mkdir -p "${grid_dir}"
if [[ ${DRY} == 1 ]]; then
  for s in B3-plain B3a B3b; do skip "$s" "dry-run: would run b3-grid on $(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["grid"])' "${RUN}/meta/grid.json" 2>/dev/null)"; done
elif ! all_pass B0 B3-R0; then
  for s in B3-plain B3a B3b; do skip "$s" "B0/B3-R0 not PASS (%smid label premise)"; done
elif over_budget; then
  for s in B3-plain B3a B3b; do skip "$s" "budget"; done
else
  run_t 900 "${grid_dir}/b3_grid.log" python3 "${HELPER}" b3-grid --configs "${GRID_CFGS}" \
    --out "${grid_dir}/b3_grid_raw.json" --outdir "${grid_dir}"
  echo "b3-grid rc=$?" >> "${grid_dir}/rc.txt"
  python3 "${HELPER}" score-b3 --b3dir "${RUN}/b3" --status-dir "${RUN}/status" --tag "${TAG}" --parts grid
fi
stage_end B3a

# ---- B3-gran: odd/small requests (descriptive), one process per candidate
stage_begin B3-gran
gran_dir="${RUN}/b3/gran"; mkdir -p "${gran_dir}"
if [[ ${DRY} == 1 ]]; then
  skip B3-gran "dry-run: candidates $(python3 "${HELPER}" gran-list | tr '\n' ' ')"
elif ! all_pass B3-plain; then
  skip B3-gran "B3-plain not PASS (no reference D)"
else
  while read -r gp gd; do
    if over_budget; then echo "budget stop before ${gp}/${gd}" >> "${gran_dir}/rc.txt"; break; fi
    f="${gran_dir}/gran_${gp}_${gd}.json"; touch "${f}.attempted"
    run_t 150 "${gran_dir}/gran_${gp}_${gd}.log" python3 "${HELPER}" b3-gran-one --p "${gp}" --d "${gd}" --out "${f}"
    echo "${gp}/${gd} rc=$?" >> "${gran_dir}/rc.txt"
  done < <(python3 "${HELPER}" gran-list)
  python3 "${HELPER}" score-b3 --b3dir "${RUN}/b3" --status-dir "${RUN}/status" --tag "${TAG}" --parts gran
fi
stage_end B3-gran

# ---- B3c: P0-A verbatim (cudagraph capture/replay x green-context confinement)
stage_begin B3c
p0a_dir="${RUN}/b3/p0a"; mkdir -p "${p0a_dir}"
if [[ ${DRY} == 0 ]] && ! all_pass B0 B3-R0; then skip B3c "B0/B3-R0 not PASS (P0-A prereg sec9-2: R0 is its premise)"
elif over_budget; then skip B3c "budget"
else
  export TRITON_CACHE_DIR="${RUN}/cache/triton_p0a"; mkdir -p "${TRITON_CACHE_DIR}"
  run_t 900 "${p0a_dir}/selftest.log" python3 "${P0A_TOOL}" --selftest; rc1=$?
  echo "selftest=${rc1}" > "${p0a_dir}/rc.txt"
  if [[ ${rc1} != 0 ]]; then
    touch "${p0a_dir}/.attempted"
    set_status B3c UNRESOLVED "P0-A selftest failed (rc=${rc1}) -- plumbing, not a result (p0a sbatch exit 20)"
  elif [[ ${DRY} == 1 ]]; then
    skip B3c "dry-run: P0-A CPU selftest passed; GPU legs not run"
  else
    touch "${p0a_dir}/.attempted"
    run_t 1200 "${p0a_dir}/run.log" python3 "${P0A_TOOL}" --run --tag "${TAG}" --outdir "${p0a_dir}"; rc2=$?
    rc3=-1
    if [[ ${rc2} == 0 ]]; then
      run_t 300 "${p0a_dir}/analyze.log" python3 "${P0A_TOOL}" --analyze "${p0a_dir}/p0a_raw_${TAG}.json" \
        --tag "${TAG}" --outdir "${p0a_dir}"; rc3=$?
    fi
    echo "run=${rc2} analyze=${rc3}" >> "${p0a_dir}/rc.txt"
    python3 "${HELPER}" score-b3 --b3dir "${RUN}/b3" --status-dir "${RUN}/status" --tag "${TAG}" --parts p0a
  fi
  export TRITON_CACHE_DIR="${RUN}/cache/triton"
fi
stage_end B3c

# ================================================================== B4
stage_begin B4
b4="${RUN}/b4"
# argv = results/r2_eval/e2_sticky_prereg/e2_sticky.sbatch:221-228 (OFF arm), port chosen below
SRV_ARGS=(--model-path "${MODEL}" --trust-remote-code --dtype bfloat16
  --attention-backend "${BACKEND}"
  --enable-pdmux --pdmux-config-path "${CFG}"
  --disable-overlap-schedule --chunked-prefill-size -1
  --disable-radix-cache --mem-fraction-static "${MEM_FRACTION}"
  --max-running-requests "${MAX_RUNNING}"
  --context-length "${CTX}" --random-seed "${SERVER_SEED}"
  --host 127.0.0.1)
refdir="${HF_HOME}/hub/models--${MODEL//\//--}"
model_rev="$(cat "${refdir}/refs/main" 2>/dev/null || true)"
{ echo "model=${MODEL}"; echo "refs/main=${model_rev:-MISSING}";
  [[ -n "${model_rev}" && -d "${refdir}/snapshots/${model_rev}" ]] && echo "snapshot=present" || echo "snapshot=MISSING"; } > "${b4}/model.txt"
run_t 300 "${b4}/argcheck.log" python3 "${HELPER}" b4-argcheck -- "${SRV_ARGS[@]}" --port 30000; rc_arg=$?
printf '%q ' python -m sglang.launch_server "${SRV_ARGS[@]}" --port '<PORT>' > "${b4}/launch_command.txt"; echo >> "${b4}/launch_command.txt"

b4_gate_ok=0
if [[ ${SKIP_B4} == 1 ]]; then skip B4 "--skip-b4"
elif [[ ${rc_arg} != 0 ]]; then set_status B4 FAIL "B4 argv/pdmux config check failed (b4/argcheck.log)"
elif [[ ${DRY} == 1 ]]; then skip B4 "dry-run: argv parsed by SGLang CLI + pdmux config loaded; $(tr '\n' ' ' < "${b4}/model.txt")"
elif ! all_pass B0 B3-R0 B3-plain B3a B3b B3c; then skip B4 "an earlier gating stage is not PASS"
elif [[ "$(status_of B3-gran)" == FAIL && ${KEEP_GOING} == 0 ]]; then skip B4 "B3-gran recorded an isolation breach"
elif over_budget; then skip B4 "budget"
elif ! grep -q "snapshot=present" "${b4}/model.txt"; then set_status B4 FAIL "model not staged offline under ${refdir} (b4/model.txt)"
else b4_gate_ok=1
fi

if [[ ${b4_gate_ok} == 1 ]]; then
  srv_port=0
  for try in $(seq 1 50); do          # e2_sticky.sbatch:199-206
    cand=$((39100 + (RANDOM + $$ + try) % 800))
    if ! curl -s -o /dev/null --max-time 1 "http://127.0.0.1:${cand}/health" 2>/dev/null \
       && ! ss -ltn 2>/dev/null | grep -q ":${cand} "; then srv_port=${cand}; break; fi
  done
  if [[ ${srv_port} == 0 ]]; then set_status B4 FAIL "no free port"; else
    # knobs this smoke does NOT use (e2_sticky.sbatch:60-67)
    unset SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN PDMUX_SLO_SCHED PDMUX_SLO_MODE PDMUX_LA_COORD \
          PDMUX_FIXED_DECODE_SM_FILE PDMUX_TRACE_FORCE_PREFILL PDMUX_DUAL_WORKER_TRACE_EVERY \
          PDMUX_TRUE_DUAL_WORKER PDMUX_DUAL_WORKER PDMUX_STICKY_PARTITION
    export PDMUX_TELEMETRY_PATH="${b4}/tel_b4.jsonl"; : > "${PDMUX_TELEMETRY_PATH}"
    export PDMUX_RUN_ID="substrate_probe_${UTC}" PDMUX_WORKLOAD_ID="substrate_probe_b4"
    export PDMUX_R2_POLICY=fixed PDMUX_R2_FIXED_DSM="${FIXED_DSM}"            # e2_sticky.sbatch:213-214
    # DEVIATION from e2 (observation only, default-off module): in-server driver read-out
    export PDMUX_GREEN_READOUT=1 PDMUX_GREEN_READOUT_PATH="${b4}/green_readout.json"
    # per-run JIT caches, as job_entry.sh:150-152
    export TRITON_CACHE_DIR="${RUN}/cache/triton" XDG_CACHE_HOME="${RUN}/cache/xdg" \
           FLASHINFER_WORKSPACE_BASE="${RUN}/cache/flashinfer"
    mkdir -p "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" "${FLASHINFER_WORKSPACE_BASE}"
    echo "+ setsid $(cat "${b4}/launch_command.txt") [PORT=${srv_port}] > b4/server.log" >> "${RUN}/logs/commands.txt"
    setsid python -m sglang.launch_server "${SRV_ARGS[@]}" --port "${srv_port}" > "${b4}/server.log" 2>&1 &
    srv_pid=$!
    health=0
    for _ in $(seq 1 $(( B4_BOOT_TIMEOUT / 5 ))); do
      if [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${srv_port}/health" 2>/dev/null || true)" == 200 ]]; then health=1; break; fi
      kill -0 "${srv_pid}" 2>/dev/null || break
      over_budget && break
      sleep 5
    done
    if [[ ${health} == 1 ]]; then
      touch "${b4}/HEALTH_OK"
      run_t 900 "${b4}/requests.log" python3 "${HELPER}" b4-requests --port "${srv_port}" \
        --out "${b4}/requests.json" --timeout 180
      sleep 2
    fi
    stop_server
    nvidia-smi --query-gpu=memory.used --format=csv,noheader > "${b4}/memory_used_after_teardown.txt" 2>&1
    python3 "${HELPER}" score-b4 --srvlog "${b4}/server.log" --requests "${b4}/requests.json" \
      --health-flag "${b4}/HEALTH_OK" --green "${b4}/green_readout.json" \
      --telemetry "${PDMUX_TELEMETRY_PATH}" --cfg "${CFG}" --status-dir "${RUN}/status"
  fi
fi
[[ -f "${RUN}/status/B4-green.json" ]] || skip B4-green "B4 did not boot a server"
stage_end B4
# exit 1 iff any stage recorded FAIL (the summary is written by the EXIT trap either way)
if grep -lq '"status": "FAIL"' "${RUN}"/status/*.json 2>/dev/null; then exit 1; fi
exit 0
