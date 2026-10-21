#!/bin/bash
# compo_probe.sh -- composition-variant BOOT probe (non-measurement) for decision 24, A100 x1.
#
# WHAT.  results/anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md, decision 24 ("선행: 변형·dummy
# 가중치의 부팅 프로브(비측정)").  Can Nemotron-Nano-9B-v2-Base (rev dc0661c829b14e5b9246c05cfa89094a0875e052)
# with Mamba layers turned into attention by `hybrid_override_pattern` boot under SGLang v0.5.10 with
# `--load-format dummy`, at the operating point (cudagraph ON, PD-mux agnostic)?  Format of longctx_probe.sh:
# every stage writes status/<stage>.json = PASS | FAIL | UNRESOLVED | SKIPPED with its mechanical facts;
# SUMMARY.md tabulates them; meta/run.json; per-step `timeout`; --budget-min is a hard cap (stages that no
# longer fit are SKIPPED); GPU memory release is checked after every boot; the environment is NEVER dumped.
#   * NO PERFORMANCE JUDGMENT.  Banner numbers and request times are probe values: NOT citable.  Dummy
#     outputs are meaningless; correctness is checked only on the original model (decision 24).
#   * Telemetry OFF in every boot (decision 22: judged measurements run with instrumentation off; this
#     probe measures nothing, so no ON/OFF pair is needed).
#
# BOOTS (in order; each torn down before the next):
#   real_orig          real weights, original pattern (n_attn 4), PD-mux agnostic
#   dummy_a4           --load-format dummy, original pattern (variant dir a4, pattern byte-identical)
#   dummy_a13/a22/a31  --load-format dummy, 13/22/31 attention layers (variants a13/a22/a31, make_variant.py)
#   dummy_a31_retry    only if BOOT-dummy_a31 FAILs: a31 has 0 Mamba layers, so --max-mamba-cache-size may be
#                      meaningless/rejected -> same argv WITHOUT that flag, once
#   dummy_a4_vanilla   PD-mux off (PDMUX_ENGINE_MODE=vanilla, no --enable-pdmux = fused_sym knobs), dummy a4
#   Common: ctx 73728, mem-fraction 0.6, radix OFF, cap 128 (--max-running-requests 128
#   --max-mamba-cache-size 128), --chunked-prefill-size -1 --disable-overlap-schedule
#   --disable-piecewise-cuda-graph (decode cudagraph ON), flashinfer, bf16, seed 1.
#   PD-mux boots: --enable-pdmux --pdmux-config-path benchmarks/configs/pdmux_agnostic_manual.yml.
# PER BOOT: BOOT-<b> (health, load_format, ctx, cudagraph bs list, PD-mux stream-group line) -> REQ-<b>
#   (8 concurrent requests, 2048 synthetic input ids, 64 forced output tokens, greedy, ignore_eos; + 1
#   return_logprob request: 64 finite logprobs = no NaN/inf from dummy weights) -> PRED-<b> (banner weight
#   bytes / k / s within +-2 % of the variant's [유도] prediction, banner-implied layer counts == variant)
#   -> TD-<b> (GPU memory.used < 1024 MiB).
# VARIANTS stage (CPU, before any boot): make_variant.py builds a4/a13/a22/a31 under <run>/variants from the
#   pinned snapshot (no weight files) and SGLang's own get_config(trust_remote_code=True) must read each
#   pattern and Mamba state size as predicted.
#
# PREREQUISITES (once per Workspace/Job start): install_runtime.sh <manifest> (see longctx_probe.sh header);
#   the Nano snapshot (with safetensors) under $HF_HOME pinned to dc0661c8...
#
# USAGE
#   bash compo_probe.sh [--dry-run] [--print-plan] [--manifest PATH] [--out-root DIR] [--budget-min N]
#                       [--keep-going] [--cap N] [--only real_orig,dummy_a4,...]
#   --print-plan  no engine, no GPU: boot argv of every boot (+ the a31 retry) and the timeout /
#                 expected-minutes table (CPU test: tests/test_compo_variant.py).
#   --dry-run     PRE + VARIANTS + argv parsing by SGLang's own CLI; no GPU work.
# OUTPUT  <out-root>/compo_probe_<UTC>/{SUMMARY.md, summary.json, status/, meta/, variants/, <boot>/, logs/,
#         SHA256SUMS}.  Default out-root /data/runs.
set -uo pipefail

# ------------------------------------------------------------------ args
DRY=0; PLAN=0; KEEP_GOING=0; BUDGET_MIN=90; CAP=128; ONLY=""
MANIFEST="${PDMUX_RUNTIME_MANIFEST:-/opt/pdmux/runtime_source_manifest.sha256}"
OUT_ROOT="/data/runs"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --print-plan) PLAN=1 ;;
    --keep-going) KEEP_GOING=1 ;;
    --manifest) MANIFEST="${2:?}"; shift ;;
    --out-root) OUT_ROOT="${2:?}"; shift ;;
    --budget-min) BUDGET_MIN="${2:?}"; shift ;;
    --cap) CAP="${2:?}"; shift ;;
    --only) ONLY="${2:?}"; shift ;;
    -h|--help) sed -n '2,46p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

# ------------------------------------------------------------------ fixed factors
NANO="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
NANO_REV="dc0661c829b14e5b9246c05cfa89094a0875e052"
BACKEND="flashinfer"; CTX=73728; MEM_FRACTION=0.6; SERVER_SEED=1
BOOT_TIMEOUT="${PDMUX_PROBE_BOOT_TIMEOUT:-1200}"       # seconds
BOOT_MIN_REMAIN_MIN=8                                    # do not start a boot with less budget left
MEM_FREE_MIB=1024

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="$(cd "${SELF_DIR}/../.." && pwd)"
PROJECT="$(cd "${ENGINE}/../.." && pwd)"
HELPER="${SELF_DIR}/compo_probe_helper.py"
MAKE_VARIANT="${ENGINE}/scripts/compo/make_variant.py"
AGN_CFG="${ENGINE}/benchmarks/configs/pdmux_agnostic_manual.yml"
VAR_ROOT="<RUN>/variants"                                # set to the run dir below (print-plan shows the placeholder)
BOOTS=(real_orig dummy_a4 dummy_a13 dummy_a22 dummy_a31 dummy_a4_vanilla)

COMMON=(--trust-remote-code --dtype bfloat16 --attention-backend "${BACKEND}" --disable-radix-cache
        --mem-fraction-static "${MEM_FRACTION}" --max-running-requests "${CAP}" --context-length "${CTX}"
        --random-seed "${SERVER_SEED}" --disable-piecewise-cuda-graph --chunked-prefill-size -1
        --disable-overlap-schedule --host 127.0.0.1)
PDMUX=(--enable-pdmux --pdmux-config-path "${AGN_CFG}")
MAMBA_CAP=(--max-mamba-cache-size "${CAP}")
DUMMY=(--load-format dummy)
argv_for() {   # boot -> argv words, one per line (after `python -m sglang.launch_server`)
  case "$1" in
    real_orig)        printf '%s\n' --model-path "${NANO}" --revision "${NANO_REV}" "${COMMON[@]}" "${PDMUX[@]}" "${MAMBA_CAP[@]}" ;;
    dummy_a4)         printf '%s\n' --model-path "${VAR_ROOT}/a4"  "${COMMON[@]}" "${PDMUX[@]}" "${MAMBA_CAP[@]}" "${DUMMY[@]}" ;;
    dummy_a13)        printf '%s\n' --model-path "${VAR_ROOT}/a13" "${COMMON[@]}" "${PDMUX[@]}" "${MAMBA_CAP[@]}" "${DUMMY[@]}" ;;
    dummy_a22)        printf '%s\n' --model-path "${VAR_ROOT}/a22" "${COMMON[@]}" "${PDMUX[@]}" "${MAMBA_CAP[@]}" "${DUMMY[@]}" ;;
    dummy_a31)        printf '%s\n' --model-path "${VAR_ROOT}/a31" "${COMMON[@]}" "${PDMUX[@]}" "${MAMBA_CAP[@]}" "${DUMMY[@]}" ;;
    dummy_a31_retry)  printf '%s\n' --model-path "${VAR_ROOT}/a31" "${COMMON[@]}" "${PDMUX[@]}" "${DUMMY[@]}" ;;
    dummy_a4_vanilla) printf '%s\n' --model-path "${VAR_ROOT}/a4"  "${COMMON[@]}" "${MAMBA_CAP[@]}" "${DUMMY[@]}" ;;
  esac
}
engine_mode_for() { case "$1" in dummy_a4_vanilla) echo vanilla ;; *) echo pdmux ;; esac; }
is_dummy() { [[ "$1" == dummy_* ]]; }
variant_of() { case "$1" in real_orig|dummy_a4|dummy_a4_vanilla) echo a4 ;; dummy_a13) echo a13 ;; dummy_a22) echo a22 ;; dummy_a31*) echo a31 ;; esac; }

declare -A TMO=( [PRE]=600 [VARIANTS]=300 [boot]="${BOOT_TIMEOUT}" [REQ]=600 [teardown]=120 )
declare -A EXP_LO=( [PRE]=1 [VARIANTS]=1 [boot]=2 [boot_real]=3 [REQ]=1 [PRED]=0 [td]=1 )
declare -A EXP_HI=( [PRE]=3 [VARIANTS]=2 [boot]=8 [boot_real]=10 [REQ]=2 [PRED]=0 [td]=1 )
print_plan() {
  local b k lo=0 hi=0
  for b in "${BOOTS[@]}"; do
    printf 'ARGV\t%s\tengine_mode=%s\tpython -m sglang.launch_server %s --port <PORT>\n' "$b" "$(engine_mode_for "$b")" \
      "$(argv_for "$b" | tr '\n' ' ')"
  done
  printf 'ARGV_RETRY\t%s\tengine_mode=%s\tpython -m sglang.launch_server %s --port <PORT>\n' dummy_a31_retry \
    "$(engine_mode_for dummy_a31_retry)" "$(argv_for dummy_a31_retry | tr '\n' ' ')"
  for k in PRE VARIANTS boot REQ teardown; do printf 'TIMEOUT_S\t%s\t%s\n' "$k" "${TMO[$k]}"; done
  for k in PRE VARIANTS; do
    printf 'EXPECT_MIN\t%s\t%s\t%s\n' "$k" "${EXP_LO[$k]}" "${EXP_HI[$k]}"; lo=$((lo + EXP_LO[$k])); hi=$((hi + EXP_HI[$k]))
  done
  for b in "${BOOTS[@]}" dummy_a31_retry; do
    local bk=boot; [[ "$b" == real_orig ]] && bk=boot_real
    local blo=$((EXP_LO[$bk] + EXP_LO[REQ] + EXP_LO[PRED] + EXP_LO[td])) bhi=$((EXP_HI[$bk] + EXP_HI[REQ] + EXP_HI[PRED] + EXP_HI[td]))
    printf 'EXPECT_MIN\t%s\t%s\t%s\n' "$b" "${blo}" "${bhi}"
    if [[ "$b" == dummy_a31_retry ]]; then retry_lo=${blo}; retry_hi=${bhi}; fi
    lo=$((lo + blo)); hi=$((hi + bhi))
  done
  printf 'EXPECT_TOTAL_MIN\tall_boots_with_a31_retry\t%s\t%s\n' "${lo}" "${hi}"
  printf 'EXPECT_TOTAL_MIN\tno_retry\t%s\t%s\n' "$((lo - retry_lo))" "$((hi - retry_hi))"
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
# state dtypes must stay at the SGLang defaults the predictions assume (conv bf16, ssm fp32)
unset SGLANG_MAMBA_SSM_DTYPE SGLANG_MAMBA_CONV_DTYPE

UTC="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="compo_probe_${UTC}"
if [[ ${DRY} == 1 && ! -d "${OUT_ROOT}" ]]; then OUT_ROOT="${TMPDIR:-/tmp}/compo_probe_dryrun"; fi
RUN="${OUT_ROOT}/${RUN_ID}$([[ ${DRY} == 1 ]] && echo _dryrun)"
mkdir -p "${RUN}"/{meta,logs,status,variants,cache/triton} || { echo "cannot create ${RUN}" >&2; exit 2; }
for b in "${BOOTS[@]}" dummy_a31_retry; do mkdir -p "${RUN}/${b}"; done
VAR_ROOT="${RUN}/variants"
export TRITON_CACHE_DIR="${RUN}/cache/triton" XDG_CACHE_HOME="${RUN}/cache/xdg" \
       FLASHINFER_WORKSPACE_BASE="${RUN}/cache/flashinfer" HF_MODULES_CACHE="${RUN}/cache/hf_modules"   # per-run JIT / remote-code caches
mkdir -p "${XDG_CACHE_HOME}" "${FLASHINFER_WORKSPACE_BASE}" "${HF_MODULES_CACHE}"
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
over_budget() { (( SECONDS > BUDGET_MIN * 60 )); }
fits() { (( SECONDS + $1 * 60 <= BUDGET_MIN * 60 )); }
skip() { set_status "$1" SKIPPED "$2"; }
skip_boot() { local s; for s in BOOT REQ PRED TD; do skip "${s}-$1" "$2"; done; }
run_t() {        # secs logfile cmd...  -> rc (124 = timeout); clamped to the remaining budget (+60 s)
  local secs="$1" log="$2"; shift 2
  local left=$(( BUDGET_MIN * 60 - SECONDS + 60 )); (( left < 60 )) && left=60
  (( secs > left )) && secs=${left}
  echo "+ $* (timeout ${secs}s) > ${log#"${RUN}"/}" >> "${RUN}/logs/commands.txt"
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
  "${NANO}" "${NANO_REV}" "${CTX}" "${MEM_FRACTION}" "${RUN_ID}" "${HELPER}" "${MAKE_VARIANT}" <<'PY'
import hashlib, json, os, platform, subprocess, sys, time
out, dry, man, proj, keep, budget, cap, nano, rev, ctx, memf, run_id, helper, mv = sys.argv[1:15]
def sh(*a):
    try: return subprocess.run(a, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return f"ERROR {e!r}"
def sha(p):
    try: return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except OSError: return None
try:
    sys.path.insert(0, os.path.dirname(helper))
    import compo_probe_helper as H
    rule, n_req, in_len, out_tok = H.SYNTH_RULE, H.N_REQ, H.IN_LEN, H.OUT_TOKENS
    swap = {"name": H.MV.RULE_NAME, "formula": H.MV.RULE_FORMULA}
except Exception as e:  # noqa: BLE001
    rule, n_req, in_len, out_tok, swap = f"helper import failed: {e!r}", None, None, None, None
vd = os.path.dirname(helper)
rec = {"run_id": run_id, "mode": "dry-run" if dry == "1" else "gpu", "kind": "compo_probe (non-measurement)",
       "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": platform.node(),
       "commit": sh("git", "-C", proj, "rev-parse", "HEAD"),
       "git_dirty": bool(sh("git", "-C", proj, "status", "--porcelain", "--untracked-files=no")),
       "manifest": man, "manifest_sha256": sha(man), "keep_going": keep == "1", "budget_min": int(budget),
       "cap": int(cap), "context_length": int(ctx), "mem_fraction_static": float(memf),
       "nano_model": nano, "nano_revision": rev, "swap_rule": swap, "n_req": n_req, "in_len": in_len,
       "out_tokens": out_tok, "synthetic_input_rule": rule, "telemetry": "off (decision 22)",
       "script_sha256": {os.path.basename(p): sha(p) for p in
                         (os.path.join(vd, "compo_probe.sh"), helper, mv, os.path.join(vd, "longctx_probe_helper.py"))}}
json.dump(rec, open(out, "w"), indent=2)
PY

# ================================================================== PRE
stage_begin PRE
pre_fail=""
inst="${MANIFEST}.install.txt"
if [[ ! -d "${SGLANG_ENGINE_DEV}/sglang" ]]; then
  pre_fail="SGLANG_ENGINE_DEV tree missing: ${SGLANG_ENGINE_DEV}"
elif [[ ! -s "${MANIFEST}" ]]; then
  pre_fail="install manifest missing: ${MANIFEST} (run install_runtime.sh <manifest> first)"
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
  elif [[ ! -f "${AGN_CFG}" || ! -f "${HELPER}" || ! -f "${MAKE_VARIANT}" ]]; then
    pre_fail="missing config/helper: ${AGN_CFG} ${HELPER} ${MAKE_VARIANT}"
  elif ! run_t 300 "${RUN}/logs/pre_imports.log" python3 -c '
from sglang.srt.distributed.parallel_state import pdmux_role_is_thread_local
assert pdmux_role_is_thread_local() is True
import sglang.srt.multiplex.pdmux_context
from sglang.srt.configs.load_config import LoadFormat
assert LoadFormat("dummy") == LoadFormat.DUMMY
print("engine imports OK, LoadFormat.DUMMY present")'; then
    pre_fail="engine imports failed (logs/pre_imports.log)"
  fi
fi
if [[ -n "${pre_fail}" ]]; then set_status PRE FAIL "${pre_fail}"
else set_status PRE PASS "manifest verified (${n_ent} entries), install commit == HEAD, thread-local role patch, LoadFormat.DUMMY"
fi
stage_end PRE
ALL_STAGES=(VARIANTS)
for b in "${BOOTS[@]}" dummy_a31_retry; do ALL_STAGES+=("BOOT-${b}" "REQ-${b}" "PRED-${b}" "TD-${b}"); done
if [[ "$(status_of PRE)" != PASS ]]; then
  for s in "${ALL_STAGES[@]}"; do skip "$s" "PRE not PASS"; done
  exit 1
fi

# ================================================================== VARIANTS (CPU)
stage_begin VARIANTS
run_t "${TMO[VARIANTS]}" "${RUN}/logs/variants.log" python3 "${HELPER}" variants --hf-home "${HF_HOME}" \
  --n-attn 4,13,22,31 --out-root "${VAR_ROOT}" --out "${RUN}/meta/variants.json" --status-dir "${RUN}/status"
[[ "$(status_of VARIANTS)" == MISSING ]] && set_status VARIANTS FAIL "helper produced no status (timeout/crash; logs/variants.log)"
stage_end VARIANTS
# argv of every boot must parse with SGLang's own CLI (cheap): a typo here would cost a boot
argfail=""
if [[ "$(status_of VARIANTS)" == PASS ]]; then
  for b in "${BOOTS[@]}" dummy_a31_retry; do
    mapfile -t _av < <(argv_for "$b")
    if ! python3 - "$(is_dummy "$b" && echo 1 || echo 0)" "${_av[@]}" --port 30000 > "${RUN}/logs/argcheck_${b}.log" 2>&1 <<'PY'
import argparse, sys
from sglang.srt.server_args import ServerArgs
want_dummy = sys.argv[1] == "1"
p = argparse.ArgumentParser(); ServerArgs.add_cli_args(p); ns = p.parse_args(sys.argv[2:])
assert ns.context_length == 73728 and not ns.disable_cuda_graph and ns.disable_radix_cache
assert (str(ns.load_format) == "dummy") == want_dummy, ns.load_format
print("ok", ns.enable_pdmux, ns.load_format, ns.max_mamba_cache_size)
PY
    then argfail="boot argv '${b}' rejected by the SGLang CLI (logs/argcheck_${b}.log)"; break; fi
  done
fi
if [[ "$(status_of VARIANTS)" != PASS || -n "${argfail}" ]]; then
  why="${argfail:-VARIANTS not PASS}"
  [[ -n "${argfail}" ]] && set_status VARIANTS FAIL "${argfail}"
  for s in "${ALL_STAGES[@]:1}"; do skip "$s" "${why}"; done
  exit 1
fi
if [[ ${DRY} == 1 ]]; then
  for s in "${ALL_STAGES[@]:1}"; do skip "$s" "dry-run: PRE + VARIANTS + argv parse passed, no GPU work"; done
  print_plan > "${RUN}/meta/plan.tsv"; exit 0
fi

# ------------------------------------------------------------------ GPU facts + clock trace (no judgment)
if command -v nvidia-smi > /dev/null 2>&1; then
  nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total,clocks.max.sm,power.limit,mig.mode.current \
    --format=csv,noheader > "${RUN}/meta/gpu.csv" 2>&1
  nvidia-smi > "${RUN}/meta/nvidia_smi.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active \
    --format=csv -lms 100 > "${RUN}/meta/clk.csv" 2> "${RUN}/meta/clk.err" &
  clk_pid=$!
fi
gpu_mem_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' '; }
nano_snap="${HF_HOME}/hub/models--${NANO//\//--}/snapshots/${NANO_REV}"

# ================================================================== one boot
boot_server() {   # boot -> sets srv_pid srv_port; returns 0 if /health 200
  local b="$1" dir="${RUN}/$1" cand try
  mapfile -t av < <(argv_for "$b")
  srv_port=0
  for try in $(seq 1 50); do
    cand=$((39100 + (RANDOM + $$ + try) % 800))
    if ! curl -s -o /dev/null --max-time 1 "http://127.0.0.1:${cand}/health" 2>/dev/null \
       && ! ss -ltn 2>/dev/null | grep -q ":${cand} "; then srv_port=${cand}; break; fi
  done
  [[ ${srv_port} == 0 ]] && { echo "no free port" > "${dir}/boot_error.txt"; return 1; }
  printf '%q ' python -m sglang.launch_server "${av[@]}" --port '<PORT>' > "${dir}/launch_command.txt"; echo >> "${dir}/launch_command.txt"
  # knobs this probe does NOT use (same list as longctx_probe.sh / engine_gate_driver.py UNSET_KNOBS); telemetry OFF
  unset SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN PDMUX_SLO_SCHED PDMUX_SLO_MODE PDMUX_LA_COORD \
        PDMUX_FIXED_DECODE_SM_FILE PDMUX_TRACE_FORCE_PREFILL PDMUX_DUAL_WORKER_TRACE_EVERY \
        PDMUX_TRUE_DUAL_WORKER PDMUX_DUAL_WORKER PDMUX_STICKY_PARTITION PDMUX_R2_POLICY PDMUX_R2_FIXED_DSM \
        PDMUX_MODEL_PROFILE PDMUX_ENGINE_MODE PDMUX_ADMISSION_PATH PDMUX_DECODE_PROBE_PATH \
        PDMUX_DECODE_PROBE_GPU_EVENTS PDMUX_TELEMETRY_PATH PDMUX_PHASE_EVENTS CUDA_LAUNCH_BLOCKING
  export PDMUX_RUN_ID="${RUN_ID}" PDMUX_WORKLOAD_ID="compo_probe_${b}"
  [[ "$(engine_mode_for "$b")" == vanilla ]] && export PDMUX_ENGINE_MODE=vanilla   # PD-mux off (fused_sym)
  echo "+ setsid $(cat "${dir}/launch_command.txt") [PORT=${srv_port}] > ${b}/server.log" >> "${RUN}/logs/commands.txt"
  setsid python -m sglang.launch_server "${av[@]}" --port "${srv_port}" > "${dir}/server.log" 2>&1 &
  srv_pid=$!
  for _ in $(seq 1 $(( BOOT_TIMEOUT / 5 ))); do
    if [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${srv_port}/health" 2>/dev/null || true)" == 200 ]]; then
      touch "${dir}/HEALTH_OK"; return 0; fi
    kill -0 "${srv_pid}" 2>/dev/null || return 1
    over_budget && return 1
    sleep 5
  done
  return 1
}

teardown() {   # boot -> status TD-<boot>
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

run_boot() {   # boot -> BOOT/REQ/PRED/TD statuses
  local b="$1" d="${RUN}/$1" healthy=0 pd=1 dm=0
  [[ "$(engine_mode_for "$b")" == vanilla ]] && pd=0
  is_dummy "$b" && dm=1
  stage_begin "BOOT-${b}"
  boot_server "$b" && healthy=1
  run_t 120 "${d}/boot_facts.log" python3 "${HELPER}" boot-facts --srvlog "${d}/server.log" --boot "$b" \
    --healthy "${healthy}" --want-dummy "${dm}" --want-pdmux "${pd}" --out "${d}/boot_facts.json" --status-dir "${RUN}/status"
  [[ "$(status_of "BOOT-${b}")" == MISSING ]] && set_status "BOOT-${b}" UNRESOLVED "boot-facts helper produced no status"
  stage_end "BOOT-${b}"
  stage_begin "REQ-${b}"
  if [[ ${healthy} == 1 ]]; then
    run_t "${TMO[REQ]}" "${d}/requests.log" python3 "${HELPER}" requests --port "${srv_port}" --srvlog "${d}/server.log" \
      --boot "$b" --tokenizer "${nano_snap}" --out "${d}/requests.json" --status-dir "${RUN}/status"
    [[ "$(status_of "REQ-${b}")" == MISSING ]] && set_status "REQ-${b}" FAIL "request helper produced no status (timeout ${TMO[REQ]}s or crash; ${b}/requests.log)"
  else
    skip "REQ-${b}" "server not healthy"
  fi
  stage_end "REQ-${b}"
  stage_begin "PRED-${b}"
  if [[ ${healthy} == 1 ]]; then
    python3 "${HELPER}" pred --boot "$b" --variant-json "${VAR_ROOT}/$(variant_of "$b")/variant.json" \
      --boot-facts "${d}/boot_facts.json" --out "${d}/pred.json" --status-dir "${RUN}/status" > "${d}/pred.log" 2>&1
    [[ "$(status_of "PRED-${b}")" == MISSING ]] && set_status "PRED-${b}" UNRESOLVED "pred helper produced no status (${b}/pred.log)"
  else
    skip "PRED-${b}" "server not healthy"
  fi
  stage_end "PRED-${b}"
  teardown "$b"
}

prev_td=""
for b in "${BOOTS[@]}"; do
  if ! want "$b"; then skip_boot "$b" "--only"; [[ "$b" == dummy_a31 ]] && skip_boot dummy_a31_retry "--only"; continue; fi
  if [[ -n "${prev_td}" && ${KEEP_GOING} == 0 && "$(status_of "TD-${prev_td}")" == FAIL ]]; then
    skip_boot "$b" "previous teardown FAIL (GPU memory not free)"; [[ "$b" == dummy_a31 ]] && skip_boot dummy_a31_retry "not attempted"; continue
  fi
  if ! fits ${BOOT_MIN_REMAIN_MIN}; then
    skip_boot "$b" "budget"; [[ "$b" == dummy_a31 ]] && skip_boot dummy_a31_retry "budget"; continue
  fi
  run_boot "$b"; prev_td="$b"
  if [[ "$b" == dummy_a31 ]]; then
    if [[ "$(status_of BOOT-dummy_a31)" == FAIL && "$(status_of TD-dummy_a31)" != FAIL ]] && fits ${BOOT_MIN_REMAIN_MIN}; then
      echo "BOOT-dummy_a31 FAIL -> one retry without --max-mamba-cache-size"
      run_boot dummy_a31_retry; prev_td=dummy_a31_retry
    else
      skip_boot dummy_a31_retry "not needed (BOOT-dummy_a31 $(status_of BOOT-dummy_a31)) or no budget / teardown FAIL"
    fi
  fi
done

# exit 1 iff any stage recorded FAIL other than the a31 first attempt when its retry booted
fails=$(grep -l '"status": "FAIL"' "${RUN}"/status/*.json 2>/dev/null | xargs -r -n1 basename | sed 's/\.json$//')
if [[ "$(status_of BOOT-dummy_a31_retry)" == PASS ]]; then fails=$(grep -vx 'BOOT-dummy_a31' <<< "${fails}"); fi
if [[ -n "${fails//[[:space:]]/}" ]]; then exit 1; fi
exit 0
