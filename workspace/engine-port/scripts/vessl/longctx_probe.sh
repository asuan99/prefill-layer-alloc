#!/bin/bash
# longctx_probe.sh -- 128k long-context bring-up probe on the A100 Workspace `pdmux-probe`.
#
# WHAT.  results/cap_char/PROBE_128k_PLAN_2026-10-20.md (items G1-G7) as one script, in the format of
# substrate_probe.sh: every stage writes status/<stage>.json = PASS | FAIL | UNRESOLVED | SKIPPED with the
# mechanical facts behind it; SUMMARY.md tabulates them; meta/run.json; per-step `timeout`; --budget-min
# is a hard cap (stages that no longer fit are SKIPPED).
#   * NO PERFORMANCE JUDGMENT.  TTFT / ITL numbers are descriptive probe values: not citable, only inputs to
#     the Job budget / the long-ctx block design (plan sec 3).  Workspace = bring-up/probe only (operating
#     model sec 1-1); nothing here is a campaign.
#   * Decisions 10-12' (DECISION_headline_G_shapeC_2026-10-19.md): contrast model Llama-3.1-8B rev
#     d04e592bb4f6aa9cfee91e2e20afa771667e1d4b; control axis = synthetic fixed length; G7 = LongBench v1
#     original; HF token only by SSH stdin (stage_llama.sh).
#
# BOOTS (4; each is torn down and GPU memory release is checked before the next):
#   1 nano_pdmux   PD-mux agnostic (benchmarks/configs/pdmux_agnostic_manual.yml), cudagraph ON, ctx 131072,
#                  cap 48 -> G1 (boot, banner, C_mem table) G2 (L x2 singles) G4 (long prefill + 4 decodes)
#                  G6 (L -> TTFT table) G7 (LongBench v1 original vs synthetic, same boot)
#   2 nano_fused   PDMUX_ENGINE_MODE=vanilla (PD-mux off) + --chunked-prefill-size -1 --disable-overlap-schedule
#                  (= fused_sym, M3 PREREG sec 2) -> G3 (same singles)
#   3 llama        fused_sym knobs, --attention-backend flashinfer, ctx 131072 -> G5 (singles x2) and
#                  G5-conc (C_mem(L) simultaneous requests at 64k and 124k, C_mem from the boot banner)
#   4 nano_chunked proxy for fused_tuned: vanilla + --chunked-prefill-size 8192 --enable-mixed-chunk
#                  (+ --disable-overlap-schedule, same as the other arms) -> G3-chunked (1 request per L).
#                  Last on purpose: it is not in the plan's time table and is the first thing the budget skips.
#   Common: radix OFF, mem-fraction 0.82, --max-running-requests CAP --max-mamba-cache-size CAP (Nano),
#   --disable-piecewise-cuda-graph (decode cudagraph stays ON; M3 PREREG arm table), seed 1, input_ids
#   requests, ignore_eos, 64 forced output tokens, input + output <= 131072 asserted per request.
#   Synthetic input: fixed-seed pseudo-text tokenized to EXACTLY L ids (rule recorded in every result json
#   and in meta/run.json; helper SYNTH_RULE).
#
# UPLOAD OF THE LONGBENCH v1 FILE (G7; once, from the local PC; the file is public data, no token needed):
#   local copy: the longctx_stats run kept it at <scratchpad>/longbench_data/lb1/data.zip (see the header of
#   results/cap_char/longctx_stats/LONGCTX_STATS_2026-10-20.txt; re-download = HF THUDM/LongBench rev
#   5e628be450b7e67fb7ae6e201bd6d8f7056f7672 data.zip).  Required sha256:
#   cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64
#     ssh <ws> 'mkdir -p /data/hf/raw/longbench_v1 && cat > /data/hf/raw/longbench_v1/data.zip' < /path/to/data.zip
#   The probe re-checks the sha256 (mismatch -> G7 UNRESOLVED; the other stages still run).  Override the
#   location with --lb-zip.
#
# PREREQUISITES (once per Workspace start; same as substrate_probe.sh):
#   export PATH=/opt/conda/bin:$PATH PDMUX_ROOT=/opt/pdmux SGLANG_ENGINE_DEV=/opt/pdmux/sglang_engine_dev/python
#   PDMUX_PROJECT_ROOT=/opt/pdmux/prefill-layer-alloc bash \
#     /opt/pdmux/prefill-layer-alloc/workspace/engine-port/scripts/vessl/install_runtime.sh \
#     /opt/pdmux/runtime_source_manifest.sha256        # install from the MERGE tree commit
#   and scripts/vessl/stage_llama.sh must have run in `pdmux-build` (Llama rev pinned under /data/hf).
#
# USAGE
#   bash longctx_probe.sh [--dry-run] [--print-plan] [--manifest PATH] [--out-root DIR] [--budget-min N]
#                         [--keep-going] [--cap N] [--lb-zip PATH] [--only nano_pdmux,nano_fused,llama,nano_chunked]
#   --print-plan  no engine, no GPU: prints the boot argv of the four boots and the per-step timeout /
#                 expected-minutes table (the CPU test compares it with the plan).
#   --dry-run     PRE + argv parsing by SGLang's own CLI + model presence; no GPU work.
# OUTPUT  <out-root>/longctx_probe_<UTC>/{SUMMARY.md, summary.json, status/, meta/, nano_pdmux/, nano_fused/,
#         llama/, nano_chunked/, logs/, SHA256SUMS}.  Default out-root: /data/runs; the plan wants it copied
#         to results/cap_char/probe_128k/<run_id>/ (RUN_ID = longctx_probe_<UTC>).  meta/clk.csv = 100 ms clock
#         trace.  The environment is NEVER dumped.
set -uo pipefail

# ------------------------------------------------------------------ args
DRY=0; PLAN=0; KEEP_GOING=0; BUDGET_MIN=120; CAP=48; ONLY=""
MANIFEST="${PDMUX_RUNTIME_MANIFEST:-/opt/pdmux/runtime_source_manifest.sha256}"
OUT_ROOT="/data/runs"
LB_ZIP="${PDMUX_LB1_ZIP:-/data/hf/raw/longbench_v1/data.zip}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --print-plan) PLAN=1 ;;
    --keep-going) KEEP_GOING=1 ;;
    --manifest) MANIFEST="${2:?}"; shift ;;
    --out-root) OUT_ROOT="${2:?}"; shift ;;
    --budget-min) BUDGET_MIN="${2:?}"; shift ;;
    --cap) CAP="${2:?}"; shift ;;
    --lb-zip) LB_ZIP="${2:?}"; shift ;;
    --only) ONLY="${2:?}"; shift ;;
    -h|--help) sed -n '2,70p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

# ------------------------------------------------------------------ fixed factors
NANO="${PDMUX_PROBE_NANO:-nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base}"   # B4 / e2_sticky precedent
LLAMA="meta-llama/Llama-3.1-8B"
LLAMA_REV="d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
BACKEND="flashinfer"; CTX=131072; MEM_FRACTION=0.82; SERVER_SEED=1; OUT_TOKENS=64
BOOT_TIMEOUT="${PDMUX_PROBE_BOOT_TIMEOUT:-1500}"       # seconds; plan: boot 8-20 min
BOOT_MIN_REMAIN_MIN=12                                   # do not start a boot with less budget left
MEM_FREE_MIB=1024                                        # teardown: GPU memory.used must fall below this

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="$(cd "${SELF_DIR}/../.." && pwd)"
PROJECT="$(cd "${ENGINE}/../.." && pwd)"
HELPER="${SELF_DIR}/longctx_probe_helper.py"
AGN_CFG="${ENGINE}/benchmarks/configs/pdmux_agnostic_manual.yml"

# per boot: argv (after `python -m sglang.launch_server`), extra env
COMMON=(--trust-remote-code --dtype bfloat16 --attention-backend "${BACKEND}" --disable-radix-cache
        --mem-fraction-static "${MEM_FRACTION}" --max-running-requests "${CAP}" --context-length "${CTX}"
        --random-seed "${SERVER_SEED}" --disable-piecewise-cuda-graph --host 127.0.0.1)
argv_for() {   # boot -> prints argv words one per line
  case "$1" in
    nano_pdmux)   printf '%s\n' --model-path "${NANO}" "${COMMON[@]}" --enable-pdmux --pdmux-config-path "${AGN_CFG}" \
                    --disable-overlap-schedule --chunked-prefill-size -1 --max-mamba-cache-size "${CAP}" ;;
    nano_fused)   printf '%s\n' --model-path "${NANO}" "${COMMON[@]}" \
                    --disable-overlap-schedule --chunked-prefill-size -1 --max-mamba-cache-size "${CAP}" ;;
    llama)        printf '%s\n' --model-path "${LLAMA}" --revision "${LLAMA_REV}" "${COMMON[@]}" \
                    --disable-overlap-schedule --chunked-prefill-size -1 ;;
    nano_chunked) printf '%s\n' --model-path "${NANO}" "${COMMON[@]}" \
                    --disable-overlap-schedule --chunked-prefill-size 8192 --enable-mixed-chunk \
                    --max-mamba-cache-size "${CAP}" ;;
  esac
}
engine_mode_for() { case "$1" in nano_pdmux) echo pdmux ;; *) echo vanilla ;; esac; }

# per-step hard timeouts (s) and expected minutes "lo-hi" (plan sec 2; chunked boot is NOT in the plan)
declare -A TMO=( [PRE]=600 [boot]="${BOOT_TIMEOUT}" [G2]=1200 [G4]=900 [G7]=900 [G3]=900 [G5]=900 [G5conc]=900
                 [G3chunked]=900 [teardown]=120 )
declare -A EXP_LO=( [PRE]=1 [boot_nano_pdmux]=8 [G2]=5 [G4]=3 [G7]=2 [td]=1 [boot_nano_fused]=8 [G3]=5
                    [boot_llama]=8 [G5]=3 [G5conc]=5 [boot_nano_chunked]=8 [G3chunked]=3 )
declare -A EXP_HI=( [PRE]=2 [boot_nano_pdmux]=20 [G2]=7 [G4]=5 [G7]=3 [td]=1 [boot_nano_fused]=15 [G3]=5
                    [boot_llama]=15 [G5]=5 [G5conc]=10 [boot_nano_chunked]=15 [G3chunked]=5 )
print_plan() {
  local b k
  for b in nano_pdmux nano_fused llama nano_chunked; do
    printf 'ARGV\t%s\tengine_mode=%s\tpython -m sglang.launch_server %s --port <PORT>\n' "$b" "$(engine_mode_for "$b")" \
      "$(argv_for "$b" | tr '\n' ' ')"
  done
  for k in PRE boot G2 G4 G7 G3 G5 G5conc G3chunked teardown; do printf 'TIMEOUT_S\t%s\t%s\n' "$k" "${TMO[$k]}"; done
  local lo=0 hi=0
  for k in PRE boot_nano_pdmux G2 G4 G7 td boot_nano_fused G3 td boot_llama G5 G5conc td boot_nano_chunked G3chunked td; do
    printf 'EXPECT_MIN\t%s\t%s\t%s\n' "$k" "${EXP_LO[$k]}" "${EXP_HI[$k]}"
    lo=$((lo + EXP_LO[$k])); hi=$((hi + EXP_HI[$k]))
  done
  # plan sec 2 subset (3 boots, no chunked proxy): everything except the chunked boot + its probe
  local plo=$((lo - EXP_LO[boot_nano_chunked] - EXP_LO[G3chunked] - EXP_LO[td]))
  local phi=$((hi - EXP_HI[boot_nano_chunked] - EXP_HI[G3chunked] - EXP_HI[td]))
  printf 'EXPECT_TOTAL_MIN\tall_four_boots\t%s\t%s\n' "${lo}" "${hi}"
  printf 'EXPECT_TOTAL_MIN\tplan_subset_three_boots\t%s\t%s\n' "${plo}" "${phi}"
  printf 'BUDGET_MIN\t%s\n' "${BUDGET_MIN}"
}
if [[ ${PLAN} == 1 ]]; then print_plan; exit 0; fi

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
export PDMUX_PROJECT_ROOT="${PROJECT}"

UTC="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="longctx_probe_${UTC}"
if [[ ${DRY} == 1 && ! -d "${OUT_ROOT}" ]]; then OUT_ROOT="${TMPDIR:-/tmp}/longctx_probe_dryrun"; fi
RUN="${OUT_ROOT}/${RUN_ID}$([[ ${DRY} == 1 ]] && echo _dryrun)"
mkdir -p "${RUN}"/{meta,logs,status,nano_pdmux,nano_fused,llama,nano_chunked,cache/triton} || { echo "cannot create ${RUN}" >&2; exit 2; }
export TRITON_CACHE_DIR="${RUN}/cache/triton" XDG_CACHE_HOME="${RUN}/cache/xdg" \
       FLASHINFER_WORKSPACE_BASE="${RUN}/cache/flashinfer"      # per-run JIT caches (job_entry.sh:150-152), shared by the 4 boots
mkdir -p "${XDG_CACHE_HOME}" "${FLASHINFER_WORKSPACE_BASE}"
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
elapsed_min() { echo $(( SECONDS / 60 )); }
over_budget() { (( SECONDS > BUDGET_MIN * 60 )); }
fits() { (( SECONDS + $1 * 60 <= BUDGET_MIN * 60 )); }          # fits <minutes>
skip() { set_status "$1" SKIPPED "$2"; }
run_t() {        # secs logfile cmd...  -> rc (124 = timeout); secs is clamped to the remaining budget (+60 s)
  local secs="$1" log="$2"; shift 2
  local left=$(( BUDGET_MIN * 60 - SECONDS + 60 )); (( left < 60 )) && left=60
  (( secs > left )) && secs=${left}
  echo "+ $* (timeout ${secs}s) > ${log#${RUN}/}" >> "${RUN}/logs/commands.txt"
  timeout -k 30 "${secs}" "$@" > "${log}" 2>&1
}
want() { [[ -z "${ONLY}" || ",${ONLY}," == *",$1,"* ]]; }

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
  echo "run dir: ${RUN}   (RUN_ID ${RUN_ID})"
  exit "${rc}"
}
RUN_T0="$(now)"
trap finish EXIT
trap 'exit 130' INT TERM

# ------------------------------------------------------------------ meta (no env dump)
python3 - "${RUN}/meta/run.json" "${DRY}" "${MANIFEST}" "${PROJECT}" "${KEEP_GOING}" "${BUDGET_MIN}" "${CAP}" \
  "${NANO}" "${LLAMA}" "${LLAMA_REV}" "${CTX}" "${LB_ZIP}" "${RUN_ID}" "${HELPER}" <<'PY'
import hashlib, json, os, platform, subprocess, sys, time
out, dry, man, proj, keep, budget, cap, nano, llama, lrev, ctx, lb, run_id, helper = sys.argv[1:15]
sys.path.insert(0, os.path.dirname(helper))
def sh(*a):
    try: return subprocess.run(a, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return f"ERROR {e!r}"
def sha(p):
    try: return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except OSError: return None
try:
    import longctx_probe_helper as H
    rule, grid, outtok = H.SYNTH_RULE, H.L_GRID, H.OUT_TOKENS
except Exception as e:  # noqa: BLE001
    rule, grid, outtok = f"helper import failed: {e!r}", None, None
vd = os.path.join(proj, "workspace/engine-port/scripts/vessl")
rec = {"run_id": run_id, "mode": "dry-run" if dry == "1" else "gpu",
       "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": platform.node(),
       "commit": sh("git", "-C", proj, "rev-parse", "HEAD"),
       "git_dirty": bool(sh("git", "-C", proj, "status", "--porcelain", "--untracked-files=no")),
       "manifest": man, "manifest_sha256": sha(man), "keep_going": keep == "1", "budget_min": int(budget),
       "cap": int(cap), "context_length": int(ctx), "nano_model": nano, "llama_model": llama,
       "llama_revision": lrev, "lb_zip": lb, "L_grid": grid, "out_tokens": outtok,
       "synthetic_input_rule": rule,
       "script_sha256": {n: sha(os.path.join(vd, n)) for n in ("longctx_probe.sh", "longctx_probe_helper.py")}}
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
  elif [[ ! -f "${AGN_CFG}" || ! -f "${HELPER}" ]]; then
    pre_fail="missing config/helper: ${AGN_CFG} ${HELPER}"
  elif ! run_t 300 "${RUN}/logs/pre_imports.log" python3 -c '
from sglang.srt.distributed.parallel_state import pdmux_role_is_thread_local
assert pdmux_role_is_thread_local() is True
import sglang.srt.multiplex.pdmux_context
print("engine imports OK")'; then
    pre_fail="engine imports failed (logs/pre_imports.log)"
  elif ! run_t 120 "${RUN}/logs/pre_models.log" python3 "${HELPER}" pre-models --hf-home "${HF_HOME}" \
         --nano "${NANO}" --llama "${LLAMA}" --out "${RUN}/meta/models.json"; then
    pre_fail="model snapshots missing/unpinned under ${HF_HOME} (meta/models.json, logs/pre_models.log; run stage_llama.sh)"
  fi
fi
# argv of every boot must parse with SGLang's own CLI (cheap, CPU): a typo here would cost a 15-minute boot
if [[ -z "${pre_fail}" ]]; then
  for b in nano_pdmux nano_fused llama nano_chunked; do
    mapfile -t _av < <(argv_for "$b")
    if ! python3 - "${_av[@]}" --port 30000 > "${RUN}/logs/argcheck_${b}.log" 2>&1 <<'PY'
import argparse, sys
from sglang.srt.server_args import ServerArgs
p = argparse.ArgumentParser(); ServerArgs.add_cli_args(p); ns = p.parse_args(sys.argv[1:])
assert ns.context_length == 131072 and not ns.disable_cuda_graph and ns.disable_radix_cache
print("ok", ns.enable_pdmux, ns.chunked_prefill_size, ns.enable_mixed_chunk, ns.max_mamba_cache_size)
PY
    then pre_fail="boot argv '${b}' rejected by the SGLang CLI (logs/argcheck_${b}.log)"; break; fi
  done
fi
if [[ -n "${pre_fail}" ]]; then set_status PRE FAIL "${pre_fail}"
else set_status PRE PASS "manifest verified (${n_ent} entries), install commit == HEAD, models staged (Llama pinned ${LLAMA_REV:0:12}), 4 boot argv parse"
fi
stage_end PRE
ALL_STAGES=(G1 G2 G4 G6 G7 TD-nano_pdmux G3 TD-nano_fused G5 G5-conc TD-llama G3-chunked TD-nano_chunked)
if [[ "$(status_of PRE)" != PASS ]]; then
  for s in "${ALL_STAGES[@]}"; do skip "$s" "PRE not PASS"; done
  exit 1
fi
if [[ ${DRY} == 1 ]]; then
  for s in "${ALL_STAGES[@]}"; do skip "$s" "dry-run: PRE passed, no GPU work"; done
  print_plan > "${RUN}/meta/plan.tsv"; exit 0
fi

# ------------------------------------------------------------------ GPU facts + clock trace (no judgment)
if command -v nvidia-smi > /dev/null 2>&1; then
  nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total,clocks.max.sm,power.limit,mig.mode.current \
    --format=csv,noheader > "${RUN}/meta/gpu.csv" 2>&1
  nvidia-smi > "${RUN}/meta/nvidia_smi.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv -lms 100 > "${RUN}/meta/clk.csv" 2> "${RUN}/meta/clk.err" &
  clk_pid=$!; sleep 2
  if ! kill -0 "${clk_pid}" 2>/dev/null; then
    nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active \
      --format=csv -lms 100 > "${RUN}/meta/clk.csv" 2>> "${RUN}/meta/clk.err" &
    clk_pid=$!
  fi
fi
gpu_mem_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' '; }
snap_dir() {  # model -> snapshot dir (refs/main)
  local d="${HF_HOME}/hub/models--${1//\//--}"; echo "${d}/snapshots/$(cat "${d}/refs/main" 2>/dev/null)"
}

# ================================================================== one boot: boot -> steps -> teardown
# usage: run_boot <name> <model> <stages-with-steps callback>
boot_server() {   # name -> sets srv_pid srv_port; returns 0 if /health 200
  local b="$1" dir="${RUN}/$1"
  mapfile -t av < <(argv_for "$b")
  srv_port=0
  for try in $(seq 1 50); do
    cand=$((39100 + (RANDOM + $$ + try) % 800))
    if ! curl -s -o /dev/null --max-time 1 "http://127.0.0.1:${cand}/health" 2>/dev/null \
       && ! ss -ltn 2>/dev/null | grep -q ":${cand} "; then srv_port=${cand}; break; fi
  done
  [[ ${srv_port} == 0 ]] && { echo "no free port" > "${dir}/boot_error.txt"; return 1; }
  printf '%q ' python -m sglang.launch_server "${av[@]}" --port '<PORT>' > "${dir}/launch_command.txt"; echo >> "${dir}/launch_command.txt"
  # knobs this probe does NOT use (same list as substrate_probe.sh / e2_sticky.sbatch:60-67)
  unset SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN PDMUX_SLO_SCHED PDMUX_SLO_MODE PDMUX_LA_COORD \
        PDMUX_FIXED_DECODE_SM_FILE PDMUX_TRACE_FORCE_PREFILL PDMUX_DUAL_WORKER_TRACE_EVERY \
        PDMUX_TRUE_DUAL_WORKER PDMUX_DUAL_WORKER PDMUX_STICKY_PARTITION PDMUX_R2_POLICY PDMUX_R2_FIXED_DSM \
        PDMUX_MODEL_PROFILE PDMUX_ENGINE_MODE
  export PDMUX_TELEMETRY_PATH="${dir}/tel.jsonl"; : > "${PDMUX_TELEMETRY_PATH}"
  export PDMUX_RUN_ID="${RUN_ID}" PDMUX_WORKLOAD_ID="longctx_probe_${b}"
  [[ "$(engine_mode_for "$b")" == vanilla ]] && export PDMUX_ENGINE_MODE=vanilla   # PD-mux off (M3 PREREG fused_sym)
  echo "+ setsid $(cat "${dir}/launch_command.txt") [PORT=${srv_port}] > ${b}/server.log" >> "${RUN}/logs/commands.txt"
  setsid python -m sglang.launch_server "${av[@]}" --port "${srv_port}" > "${dir}/server.log" 2>&1 &
  srv_pid=$!
  local waited=0
  for _ in $(seq 1 $(( BOOT_TIMEOUT / 5 ))); do
    if [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${srv_port}/health" 2>/dev/null || true)" == 200 ]]; then
      touch "${dir}/HEALTH_OK"; return 0; fi
    kill -0 "${srv_pid}" 2>/dev/null || return 1
    over_budget && return 1
    sleep 5
  done
  return 1
}

teardown() {   # name -> status TD-<name>
  local b="$1" dir="${RUN}/$1" m="" i
  stage_begin "TD-${b}"
  stop_server
  for i in $(seq 1 24); do m="$(gpu_mem_mib)"; [[ -n "${m}" && "${m}" -lt ${MEM_FREE_MIB} ]] && break; sleep 5; done
  echo "${m:-unknown}" > "${dir}/memory_used_after_teardown_mib.txt"
  if [[ -z "${m}" ]]; then set_status "TD-${b}" UNRESOLVED "nvidia-smi gave no memory.used"
  elif [[ "${m}" -lt ${MEM_FREE_MIB} ]]; then set_status "TD-${b}" PASS "memory.used ${m} MiB < ${MEM_FREE_MIB} MiB after teardown"
  else set_status "TD-${b}" FAIL "memory.used ${m} MiB >= ${MEM_FREE_MIB} MiB 60 s after teardown (leftover process?)"
  fi
  stage_end "TD-${b}"
}

td_ok() { [[ ${KEEP_GOING} == 1 || "$(status_of "TD-$1")" == PASS || "$(status_of "TD-$1")" == SKIPPED ]]; }

# ---------------------------------------------------------------- boot 1: nano_pdmux  (G1 G2 G4 G6 G7)
nano_tok="$(snap_dir "${NANO}")"; llama_tok="$(snap_dir "${LLAMA}")"
stage_begin G1
if ! want nano_pdmux; then for s in G1 G2 G4 G6 G7 TD-nano_pdmux; do skip "$s" "--only"; done
elif ! fits ${BOOT_MIN_REMAIN_MIN}; then
  for s in G1 G2 G4 G6 G7 TD-nano_pdmux; do skip "$s" "budget"; done
else
  d="${RUN}/nano_pdmux"
  if boot_server nano_pdmux; then
    run_t 120 "${d}/boot_facts.log" python3 "${HELPER}" boot-facts --srvlog "${d}/server.log" --arm nano_pdmux \
      --model "${NANO}" --cap "${CAP}" --out "${d}/boot_facts.json"
    python3 - "${d}/boot_facts.json" "${RUN}/status" "${CAP}" <<'PY'
import json, sys, os
sys.path.insert(0, os.environ["PDMUX_PROJECT_ROOT"] + "/workspace/engine-port/scripts/vessl")
import longctx_probe_helper as H
bf = json.load(open(sys.argv[1])); b = bf["banner"]; sig = bf["signatures"]
if sig["cuda_graph_capture_failed"] or sig["eager_fallback"]:
    H.write_status(sys.argv[2], "G1", H.FAIL, "cudagraph capture failure/eager fallback in the boot log (cudagraph-ON not honoured)", bf)
elif not bf["context_len_matches"]:
    H.write_status(sys.argv[2], "G1", H.FAIL, f"server logged context_len={b['context_len_logged']} != 131072", bf)
elif bf["cmem"] is None:
    H.write_status(sys.argv[2], "G1", H.UNRES, "booted and /health OK, but the KV banner was not parsed (C_mem not computed)", bf)
else:
    H.write_status(sys.argv[2], "G1", H.PASS,
                   f"booted PD-mux agnostic ctx 131072 cap {sys.argv[3]}; max_total_num_tokens={b['max_total_num_tokens']}, "
                   f"mamba slots={b['max_mamba_cache_size']}; C_mem table in nano_pdmux/boot_facts.json", bf)
PY
    PYC=(--port "${srv_port}" --srvlog "${d}/server.log" --arm nano_pdmux --model "${NANO}" --tokenizer "${nano_tok}")
    stage_begin G2
    run_t "${TMO[G2]}" "${d}/g2.log" python3 "${HELPER}" probe-single "${PYC[@]}" --stage G2 --out "${d}/g2_single.json"
    python3 "${HELPER}" score-single --in "${d}/g2_single.json" --stage G2 --expect-n 10 --status-dir "${RUN}/status"
    stage_end G2
    stage_begin G4
    if [[ "$(status_of G2)" != PASS && ${KEEP_GOING} == 0 ]] && ! curl -s -o /dev/null --max-time 5 "http://127.0.0.1:${srv_port}/health"; then
      skip G4 "server not alive after G2"
    else
      run_t "${TMO[G4]}" "${d}/g4.log" python3 "${HELPER}" probe-g4 "${PYC[@]}" --out "${d}/g4_mixed.json"
      python3 "${HELPER}" score-g4 --in "${d}/g4_mixed.json" --status-dir "${RUN}/status"
    fi
    stage_end G4
    stage_begin G7
    if [[ ! -f "${LB_ZIP}" ]]; then
      set_status G7 UNRESOLVED "LongBench v1 zip not found at ${LB_ZIP} (upload procedure in the script header)"
    elif ! curl -s -o /dev/null --max-time 5 "http://127.0.0.1:${srv_port}/health"; then
      skip G7 "server not alive"
    else
      run_t "${TMO[G7]}" "${d}/g7.log" python3 "${HELPER}" probe-g7 "${PYC[@]}" --lb-zip "${LB_ZIP}" --out "${d}/g7_longbench.json"
      python3 "${HELPER}" score-g7 --in "${d}/g7_longbench.json" --status-dir "${RUN}/status"
    fi
    stage_end G7
  else
    set_status G1 FAIL "nano_pdmux did not reach /health (boot timeout ${BOOT_TIMEOUT}s, process died, or budget); see nano_pdmux/server.log"
    run_t 120 "${d}/boot_facts.log" python3 "${HELPER}" boot-facts --srvlog "${d}/server.log" --arm nano_pdmux \
      --model "${NANO}" --cap "${CAP}" --out "${d}/boot_facts.json"
    for s in G2 G4 G6 G7; do skip "$s" "G1 not PASS"; done
  fi
  teardown nano_pdmux
fi
stage_end G1

# ---------------------------------------------------------------- boot 2: nano_fused  (G3)
stage_begin G3
if ! want nano_fused; then skip G3 "--only"; skip TD-nano_fused "--only"
elif ! td_ok nano_pdmux && [[ -f "${RUN}/status/TD-nano_pdmux.json" ]]; then skip G3 "previous teardown not PASS (GPU memory not free)"; skip TD-nano_fused "not booted"
elif ! fits ${BOOT_MIN_REMAIN_MIN}; then skip G3 "budget"; skip TD-nano_fused "not booted"
else
  d="${RUN}/nano_fused"
  if boot_server nano_fused; then
    PYC=(--port "${srv_port}" --srvlog "${d}/server.log" --arm nano_fused --model "${NANO}" --tokenizer "${nano_tok}")
    run_t 120 "${d}/boot_facts.log" python3 "${HELPER}" boot-facts --srvlog "${d}/server.log" --arm nano_fused \
      --model "${NANO}" --cap "${CAP}" --out "${d}/boot_facts.json"
    run_t "${TMO[G3]}" "${d}/g3.log" python3 "${HELPER}" probe-single "${PYC[@]}" --stage G3 --reps 1 --out "${d}/g3_single.json"
    python3 "${HELPER}" score-single --in "${d}/g3_single.json" --stage G3 --expect-n 5 --status-dir "${RUN}/status"
  else
    set_status G3 FAIL "nano_fused did not reach /health; see nano_fused/server.log"
  fi
  teardown nano_fused
fi
stage_end G3

# ---------------------------------------------------------------- boot 3: llama  (G5, G5-conc)
stage_begin G5
if ! want llama; then skip G5 "--only"; skip G5-conc "--only"; skip TD-llama "--only"
elif [[ -f "${RUN}/status/TD-nano_fused.json" ]] && ! td_ok nano_fused; then skip G5 "previous teardown not PASS"; skip G5-conc "previous teardown not PASS"; skip TD-llama "not booted"
elif ! fits ${BOOT_MIN_REMAIN_MIN}; then skip G5 "budget"; skip G5-conc "budget"; skip TD-llama "not booted"
else
  d="${RUN}/llama"
  if boot_server llama; then
    PYC=(--port "${srv_port}" --srvlog "${d}/server.log" --arm llama --model "${LLAMA}" --tokenizer "${llama_tok}")
    run_t 120 "${d}/boot_facts.log" python3 "${HELPER}" boot-facts --srvlog "${d}/server.log" --arm llama \
      --model "${LLAMA}" --cap "${CAP}" --out "${d}/boot_facts.json"
    run_t "${TMO[G5]}" "${d}/g5.log" python3 "${HELPER}" probe-single "${PYC[@]}" --stage G5 --out "${d}/g5_single.json"
    python3 "${HELPER}" score-single --in "${d}/g5_single.json" --stage G5 --expect-n 10 --status-dir "${RUN}/status"
    stage_begin G5-conc
    if ! curl -s -o /dev/null --max-time 5 "http://127.0.0.1:${srv_port}/health"; then skip G5-conc "server not alive after G5"
    else
      run_t "${TMO[G5conc]}" "${d}/g5conc.log" python3 "${HELPER}" probe-conc "${PYC[@]}" --boot-facts "${d}/boot_facts.json" \
        --out "${d}/g5_conc.json"
      python3 "${HELPER}" score-conc --in "${d}/g5_conc.json" --stage G5-conc --expect-n 0 --status-dir "${RUN}/status"
    fi
  else
    set_status G5 FAIL "llama did not reach /health; see llama/server.log"
    skip G5-conc "G5 boot failed"
  fi
  teardown llama
fi
stage_end G5
[[ -n "${T0[G5-conc]:-}" ]] && stage_end G5-conc

# G6: the L -> TTFT table (descriptive); needs only the data above
stage_begin G6
python3 "${HELPER}" score-g6 --nano "${RUN}/nano_pdmux/g2_single.json" --llama "${RUN}/llama/g5_single.json" --status-dir "${RUN}/status"
stage_end G6

# ---------------------------------------------------------------- boot 4: nano_chunked  (G3-chunked; last, first to be skipped)
stage_begin G3-chunked
if ! want nano_chunked; then skip G3-chunked "--only"; skip TD-nano_chunked "--only"
elif [[ -f "${RUN}/status/TD-llama.json" ]] && ! td_ok llama; then skip G3-chunked "previous teardown not PASS"; skip TD-nano_chunked "not booted"
elif ! fits ${BOOT_MIN_REMAIN_MIN}; then skip G3-chunked "budget"; skip TD-nano_chunked "not booted"
else
  d="${RUN}/nano_chunked"
  if boot_server nano_chunked; then
    PYC=(--port "${srv_port}" --srvlog "${d}/server.log" --arm nano_chunked --model "${NANO}" --tokenizer "${nano_tok}")
    run_t 120 "${d}/boot_facts.log" python3 "${HELPER}" boot-facts --srvlog "${d}/server.log" --arm nano_chunked \
      --model "${NANO}" --cap "${CAP}" --out "${d}/boot_facts.json"
    run_t "${TMO[G3chunked]}" "${d}/g3c.log" python3 "${HELPER}" probe-single "${PYC[@]}" --stage G3-chunked --reps 1 --out "${d}/g3c_single.json"
    python3 "${HELPER}" score-single --in "${d}/g3c_single.json" --stage G3-chunked --expect-n 5 --status-dir "${RUN}/status"
  else
    set_status G3-chunked FAIL "nano_chunked did not reach /health; see nano_chunked/server.log"
  fi
  teardown nano_chunked
fi
stage_end G3-chunked

# exit 1 iff any stage recorded FAIL (the summary is written by the EXIT trap either way)
if grep -lq '"status": "FAIL"' "${RUN}"/status/*.json 2>/dev/null; then exit 1; fi
exit 0
