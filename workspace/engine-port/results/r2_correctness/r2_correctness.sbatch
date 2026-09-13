#!/bin/bash
#SBATCH --job-name=r2corr
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/%x_%j.err
#
# R2 GPU CORRECTNESS GATE -- does PDMUX_TRUE_DUAL_WORKER=1 compute the same
# function as the legacy single-thread PD-mux loop, at the operating point?
#
# ★ INVARIANT UNDER TEST
#     greedy(prompt | true_dual fixed D) == greedy(prompt | legacy fixed D)
#   token ids, every prompt, server seed identical, cudagraph ON in both.
#   The arm pair is exactly the P2 / Claim D pair as scripts/r2_eval/
#   policy_adapter.sh builds it (legacy:fixed vs true_dual:fixed), on the
#   canonical R2 state set benchmarks/configs/pdmux_r2.yml.
#
# ★ BOOTS  (default order L TD L TD -> labels L1 TD1 L2 TD2)
#   The second L and TD boots are the NULL CONTROL for every equality tier.
#
# ★ TIERS AND VERDICT -- rule v2, fixed 2026-09-11 before any true-dual boot
#   has produced an output.  The NORMATIVE text (per-boot checks B1-B8,
#   O-tier confirmation O1-O4, the five verdict lines in precedence order,
#   and why the rule is not chosen after the fact) is the header of
#   r2_correctness_check.py; this job copies that exact text into
#   job_<id>/verdict_rule.txt and records the checker's sha256 in
#   provenance.txt, so the rule a result was scored under is pinned to it.
#   Summary (non-normative):
#     S  16 requests one at a time                       -> equality gate
#     O  8 overlap probes, each sent while one long background request
#        decodes, so the probe prefill runs on the D division concurrently
#        with that decode; probe token ids compared      -> equality gate
#     C  32 staggered concurrent requests                -> diagnostic only
#        (job 907032: legacy-vs-legacy 7/32 differ, so the pre-declared
#        null-control condition can never hold for C)
#   FAIL > NO_VERDICT_INFRA > INCONCLUSIVE > NO_VERDICT_UNREALIZED > PASS.
#   D realization is pinned by the driver: PDMUX_GREEN_READOUT=1 in every
#   scored boot (startup-only observation, both arms) writes
#   green_<label>.json, and the checker requires the D division's green
#   contexts to report smCount 108-D / D -- not just the stream-group label.
#
# ★ CUDAGRAPH MUST BE ON AND MUST STAY ON
#   No --disable-cuda-graph.  --decode-log-interval 1 makes every decode step
#   log "cuda graph: True/False"; the checker requires True on every Decode
#   line of every boot.  In true-dual mode every decode run_batch is issued
#   from the decode worker thread (multiplexing_mixin.py event_loop_pdmux,
#   decode_future), so "all True" there means graph REPLAY happened on that
#   thread -- capture at boot alone would not show it.  The same signature
#   separates cgON/cgOFF on existing data (results/cudagraph_probe/
#   srv_pdmux_cgON_847391.log 98/98 True; srv_pdmux_cgOFF_847391.log 0/41).
#
# ★ DEVIATIONS FROM scripts/r2_eval/engine_bench_runner.sh (deliberate)
#   --context-length $CTX (see R2C_CTX below: DERIVED from the served model's
#   own config, not a literal, because the runner default 16384 exceeds
#   Zamba2-2.7B max_position_embeddings=4096 and SGLang refuses to boot without
#   SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN); --decode-log-interval 1.
#   PDMUX_TRACE_FORCE_PREFILL=1 (observation only, both arms) so the
#   checker can see prefill/decode overlap in telemetry;
#   PDMUX_GREEN_READOUT=1 (startup-only driver read-out, both arms).
#
# ★ WHAT A PASS DOES NOT MEAN
#   Not a performance result, not Claim D.  It certifies that the two
#   architectures agree on outputs at this split (S: plain stream groups;
#   O: probe prefill on the D division under overlap) and that cudagraph
#   stayed on.  Other splits (D16/D24/D34), generic/hybrid policies, sampling
#   (T>0), concurrent batch compositions beyond the O protocol, and other
#   models are not covered.
#
# Cost estimate (revised after job 907032, 2026-09-11): 907032 took 7m13s with
# both TD boots dying ~30 s after their first log line; a legacy boot spans
# ~50 s of server log (boot + S + C) plus ~45 s of launch/teardown.  The O
# tier adds 8 probes x ~2.5-3 s ~= 25 s per boot (~1.7 min over 4 boots).
# Total ~= warmup 1.5 min + 4 x ~2 min ~= 10-11 min => ~0.18-0.2 GPU-h;
# hard cap 1.25 GPU-h (wall limit).
#
# usage:
#   sbatch workspace/engine-port/results/r2_correctness/r2_correctness.sbatch
#   R2C_DSM=24 sbatch ...          # other split (must exist in pdmux_r2.yml)
#   R2C_ORDER="L TD TD L" sbatch ...
#   R2C_NUM_KV_SPLITS=2 sbatch ... # X1 sensitivity positive control: forces the
#                                  # decode-attention KV-split count for EVERY
#                                  # boot (CLI only, no env var, so the value is
#                                  # visible in the server args dump and checked
#                                  # by SAME_ACROSS_BOOTS).  Unset = server
#                                  # default 8 + kernel heuristic = job 907100.
#                                  # Pre-registered in x1_prereg/ (rev2).
#   R2C_BASELINE_JOB=.../job_907100 sbatch ...  # also run the DIAGNOSTIC
#                                  # cross-job token comparison at the end.
#   R2C_TRACE_FORCE_PREFILL=0 sbatch ...  # X3 runner-configuration
#                                  # certification: drop the forced telemetry
#                                  # samples, which is what the r2_eval runner
#                                  # actually does.  NOTE the forced samples are
#                                  # ALSO the only instrument that makes the O
#                                  # tier observable (prefill is ~4.9% of wall
#                                  # time but 0.07% of SCHEDULED samples,
#                                  # multiplexing_mixin.py:545-551), so with 0
#                                  # the O-tier confirmation is expected to fail
#                                  # and the verdict to be NO_VERDICT_UNREALIZED
#                                  # BY CONSTRUCTION.  Pre-registered in
#                                  # x3_prereg/.
#   R2C_CTX=16384 sbatch ...       # --context-length for every boot.  DEFAULT =
#                                  # the served model's own derived limit
#                                  # (`python -m pdmux_eval.context_limit
#                                  # --model "$MODEL"`, the same module
#                                  # scripts/r2_eval/r2_eval.sbatch pre-flights
#                                  # the campaign with), NOT a literal.
#                                  # Zyphra/Zamba2-2.7B -> 4096, so the default
#                                  # reproduces jobs 907100/907456 exactly;
#                                  # pinned by tests/test_r2_correctness_ctx.py.
#                                  # Set it explicitly for a long-context model:
#                                  # NVIDIA-Nemotron-Nano-9B-v2-Base derives
#                                  # 131072, which is NOT the context length the
#                                  # r2_eval campaign resolves for the W1-W9
#                                  # traces, and a correctness gate should
#                                  # certify the operating point that is
#                                  # actually measured.
#   R2C_ATTN_BACKEND=flashinfer sbatch ...   # --attention-backend.  DEFAULT
#                                  # triton = the 907100/907456 value.  REQUIRED
#                                  # for NemotronH: SGLang asserts
#                                  # "NemotronHForCausalLM does not support
#                                  # triton attention backend" while parsing
#                                  # server args, so the boot dies immediately.
#                                  # Switching the backend changes the kernel,
#                                  # so arms measured under different values are
#                                  # NOT comparable -- this is a campaign-design
#                                  # choice, deliberately left to the caller.

set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
PROJECT=/scratch/ehmoon/whlee/prefill-layer-alloc
cd "$PROJECT" || exit 2

HERE=$PROJECT/workspace/engine-port/results/r2_correctness
CONFIG=$PROJECT/workspace/engine-port/benchmarks/configs/pdmux_r2.yml
OUT="$HERE/job_${SLURM_JOB_ID:-local}"; mkdir -p "$OUT"

export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache"; mkdir -p "$TRITON_CACHE_DIR"

MODEL="${R2C_MODEL:-Zyphra/Zamba2-2.7B}"
DSM="${R2C_DSM:-44}"
SEED="${R2C_SEED:-1}"
ORDER="${R2C_ORDER:-L TD L TD}"
NKVS="${R2C_NUM_KV_SPLITS:-}"
# `triton` is the 907100/907456 value and stays the DEFAULT so those jobs keep
# reproducing.  It is a knob because it is not a free choice: SGLang refuses
# NemotronHForCausalLM on the triton backend outright (server_args.py
# `_handle_mamba_radix_cache` branch: "does not support triton attention
# backend, as the first layer might not be an attention layer"), so a NemotronH
# boot needs `R2C_ATTN_BACKEND=flashinfer`, which is what
# results/longctx_conflict/probes/c_capacity.sbatch already uses for that model.
# Changing it is a MEASUREMENT-DESIGN decision, not a porting detail: the kernel
# changes, so results are not comparable across backends.
ABE="${R2C_ATTN_BACKEND:-triton}"
# 1 keeps the 907100/907456 behaviour (forced prefill-in-flight telemetry); 0 is
# the r2_eval runner's own configuration.  Either way the SCHEDULED sample grid
# is bit-identical -- forced records are strictly additional and tagged
# `trace_forced=true` (multiplexing_mixin.py:552-556).
TFP="${R2C_TRACE_FORCE_PREFILL:-1}"

# Start from a clean PD-mux environment (policy_adapter.sh does the same).
for var in $(env | grep -o '^PDMUX_[A-Z0-9_]*' || true); do unset "$var"; done
unset SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN
# X1 (x1_prereg rev2, condition D4(a)) deliberately perturbs the KV-split count
# through the CLI flag ONLY, because a CLI flag is observable in the server args
# dump and enforced equal across boots by the checker, while this env var's
# effect is not observable at all.  Clear it so an inherited value cannot
# silently change the perturbation that the pre-registration computed.
unset SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS

echo "=== env sync ==="
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh \
  "$OUT/runtime_source_manifest.sha256" || exit 2

# --- BEGIN r2c ctx resolution (extracted verbatim by tests/test_r2_correctness_ctx.py) ---
# `--context-length` used to be the literal 4096 on the server command line, so
# pointing R2C_MODEL at a model whose max_position_embeddings differs silently
# served a context the model does not declare (X3 audit finding D1).
# Resolution order: R2C_CTX, else the model's own derived limit via
# benchmarks/pdmux_eval/context_limit.py -- the same module
# scripts/r2_eval/r2_eval.sbatch pre-flights the campaign with, so the
# correctness gate and the measurement resolve context the same way.
# No --trace is passed: this harness generates its own prompts, so only the
# model limit is knowable here.  A failing derivation is fatal, never a silent
# fallback: a wrong context length is exactly what produced defect D1.
export PYTHONPATH="$PROJECT/workspace/engine-port/benchmarks${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${R2C_CTX:-}" ]; then
  CTX="$R2C_CTX"; CTX_SRC="R2C_CTX"
else
  CTX="$(python -m pdmux_eval.context_limit --model "$MODEL")" || {
    echo "ERROR: could not derive a context length for $MODEL" >&2; exit 2; }
  CTX_SRC="derived from $MODEL config"
fi
case "$CTX" in ''|*[!0-9]*) echo "ERROR: bad context length '$CTX'" >&2; exit 2;; esac
echo "context_length=$CTX ($CTX_SRC)"
# --- END r2c ctx resolution ---

COMMIT=$(git -C "$PROJECT" rev-parse HEAD)
{
  echo "job=${SLURM_JOB_ID:-local} node=$(hostname) date=$(date -Is)"
  echo "commit=$COMMIT"
  echo "src_dirty:"; git -C "$PROJECT" status --porcelain -- workspace/engine-port/src
  nvidia-smi -L
  python -c "from sglang.srt.distributed.parallel_state import pdmux_role_is_thread_local as f; print('thread_local_role_patch', f())"
  echo "harness sha256:"; sha256sum "$HERE/r2_correctness_check.py" "$HERE/r2_correctness_client.py" "$HERE/r2_correctness.sbatch"
  echo "model=$MODEL dsm=$DSM seed=$SEED order=[$ORDER] config=$CONFIG"
  echo "num_kv_splits_cli=${NKVS:-<server default>} baseline_job=${R2C_BASELINE_JOB:-<none>}"
  echo "trace_force_prefill=$TFP (907100/907456 used 1; r2_eval runner uses 0)"
  echo "context_length=$CTX ($CTX_SRC) attention_backend=$ABE"
  # The pre-registration's arithmetic assumes this SM count and a clean
  # SGLANG_TRITON_* environment; record both rather than assuming them.
  echo "env (SGLANG|R2C):"; env | grep -E '^(SGLANG|R2C)_' | sort || echo "  (none)"
  python -c "import torch; p=torch.cuda.get_device_properties(0); print('multi_processor_count', p.multi_processor_count, p.name)"
  for pre in "$HERE"/*_prereg; do
    [ -d "$pre" ] || continue
    echo "pre-registration sha256 ($(basename "$pre")):"
    sha256sum "$pre"/*.md "$pre"/*.py "$pre"/*.txt 2>/dev/null
  done
} > "$OUT/provenance.txt" 2>&1
cat "$OUT/provenance.txt"
# Pin the exact verdict rule this job is scored under (normative text = the
# checker's module header, from the rule banner to its closing rule line).
python3 -c "import ast,sys; d=ast.get_docstring(ast.parse(open(sys.argv[1]).read())); i=d.index('VERDICT RULE v2'); print(d[d.rfind(chr(10), 0, i)+1:])" \
  "$HERE/r2_correctness_check.py" > "$OUT/verdict_rule.txt" || exit 2

server_cmd () {   # $1 = port
  echo python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code \
    --dtype bfloat16 --attention-backend "$ABE" --disable-radix-cache \
    --mem-fraction-static 0.82 --max-running-requests 48 --context-length "$CTX" \
    --random-seed "$SEED" --host 127.0.0.1 --port "$1" \
    ${NKVS:+--triton-attention-num-kv-splits $NKVS} \
    --enable-pdmux --pdmux-config-path "$CONFIG" --chunked-prefill-size -1 \
    --disable-overlap-schedule --decode-log-interval 1
}

wait_health () {  # $1 = port, $2 = pid, $3 = max polls (5 s each)
  for _ in $(seq 1 "$3"); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$1/health" 2>/dev/null || true)" = 200 ] && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

stop_server () {  # $1 = port, $2 = pid
  kill "$2" 2>/dev/null; sleep 5
  pkill -9 -f "launch_server.*--port $1" 2>/dev/null; sleep 15
}

BASEPORT=$((30000 + (${SLURM_JOB_ID:-0} % 5000) * 6))

# Warm the import + weight page cache on a throwaway boot so the first scored
# boot does not time out (cp_baseline job 902397 lost two arms to exactly
# that; its BOOT_FAILED was a harness artifact, not an engine fact).
echo "=== warmup boot (not scored) ==="
env PDMUX_R2_POLICY=fixed PDMUX_R2_FIXED_DSM="$DSM" \
  $(server_cmd "$BASEPORT") > "$OUT/srv_warmup.log" 2>&1 &
WPID=$!
wait_health "$BASEPORT" "$WPID" 180 && echo "  warmup up" || echo "  warmup did not come up (continuing)"
stop_server "$BASEPORT" "$WPID"

: > "$OUT/boots.txt"
n_l=0; n_td=0; i=0
for ARM in $ORDER; do
  i=$((i + 1))
  case "$ARM" in
    L)  n_l=$((n_l + 1)); LABEL="L$n_l"; TD_ENV="" ;;
    TD) n_td=$((n_td + 1)); LABEL="TD$n_td"; TD_ENV="PDMUX_TRUE_DUAL_WORKER=1" ;;
    *)  echo "bad arm $ARM"; continue ;;
  esac
  PORT=$((BASEPORT + i))
  echo ""
  echo "########## boot=$LABEL arm=$ARM port=$PORT ##########"
  echo -n "$LABEL " >> "$OUT/boots.txt"
  env PDMUX_R2_POLICY=fixed PDMUX_R2_FIXED_DSM="$DSM" \
    PDMUX_TELEMETRY_PATH="$OUT/tel_${LABEL}.jsonl" \
    PDMUX_RUN_ID="r2corr_${SLURM_JOB_ID:-local}_${LABEL}" \
    PDMUX_WORKLOAD_ID="r2corr_seed${SEED}" \
    PDMUX_TRACE_FORCE_PREFILL="$TFP" PDMUX_ENGINE_COMMIT="$COMMIT" \
    PDMUX_GREEN_READOUT=1 PDMUX_GREEN_READOUT_PATH="$OUT/green_${LABEL}.json" $TD_ENV \
    $(server_cmd "$PORT") > "$OUT/srv_${LABEL}.log" 2>&1 &
  PID=$!
  if ! wait_health "$PORT" "$PID" 96; then
    echo "BOOT_FAILED boot=$LABEL" | tee -a "$OUT/BOOT_FAILURES.txt"
    grep -n "Traceback\|Error" "$OUT/srv_${LABEL}.log" | tail -5
    stop_server "$PORT" "$PID"
    continue
  fi
  grep -oE "disable_cuda_graph=[A-Za-z]+|random_seed=[0-9]+|enable_pdmux=[A-Za-z]+" \
    "$OUT/srv_${LABEL}.log" | sort -u | tr '\n' ' '; echo
  grep -m1 "PD-Multiplexing enabled" "$OUT/srv_${LABEL}.log"
  grep -m1 "true dual-worker enabled" "$OUT/srv_${LABEL}.log" || echo "  (legacy loop: no true-dual banner)"
  timeout 900 python3 "$HERE/r2_correctness_client.py" --port "$PORT" \
    --boot "$LABEL" --out "$OUT/gen_${LABEL}.json" --n-probes 8 \
    || echo "CLIENT_FAILED boot=$LABEL" | tee -a "$OUT/BOOT_FAILURES.txt"
  kill -0 "$PID" 2>/dev/null || echo "SERVER_DIED_DURING_CLIENT boot=$LABEL" | tee -a "$OUT/BOOT_FAILURES.txt"
  stop_server "$PORT" "$PID"
done

echo ""
echo "=== correctness check ==="
python3 "$HERE/r2_correctness_check.py" "$OUT" --dsm "$DSM" | tee "$OUT/verdict.txt"

# DIAGNOSTIC ONLY (never a verdict): cross-job token comparison against a
# baseline job, pre-registered in x1_prereg/ rev2 §3.  Runs after the verdict so
# it can never influence it.
if [ -n "${R2C_BASELINE_JOB:-}" ] && [ -d "${R2C_BASELINE_JOB}" ]; then
  echo ""
  echo "=== cross-job diagnostic (not a verdict) ==="
  python3 "$HERE/x1_prereg/x1_cross_job_compare.py" "${R2C_BASELINE_JOB}" "$OUT" \
    2>&1 | tee "$OUT/cross_job_compare.txt"
fi
echo "R2CORR_DONE_${SLURM_JOB_ID:-local}"
