#!/bin/bash
#SBATCH --job-name=r2corr
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
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
#   PDMUX_GREEN_READOUT=1 (startup-only driver read-out, both arms);
#   PDMUX_MEM_TELEMETRY=1 (observation only, both arms, engine default OFF)
#   so each runtime_snapshot carries current/peak/reserved allocator bytes and
#   the realised grad-guard state of the role workers.  Added 2026-09-13 for
#   the true-dual OOM decomposition round; it publishes no timing number.
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
# Total ~= warmup 1.5 min + 4 x ~2 min ~= 10-11 min => ~0.18-0.2 GPU-h.
#
# WALL LIMIT 02:30:00 (was 01:15:00 until 2026-09-13).  Raising a wall cap
# changes no measured quantity -- only queue wait -- and putting it in the
# DIRECTIVE removes the single human failure point of having to remember
# `sbatch --time=...`.  Budgets: 9B + R2C_INSTRUMENT=1 is ~28-35 min typical;
# the realistic worst (all three read-outs hit their timeouts, boots normal) is
# ~69 min; the STRUCTURAL worst, where every timeout in the script fires, is
# ~161 min = warm-up health 900 s + read-outs 3000 s + 4 x (health 480 s +
# client 900 s + teardown 20 s) + ~180 s overhead, i.e. 02:30:00 does NOT cover
# it.  That branch is registered in newpair_prereg/ sec 9(f), not defended
# against here: shortening the scored-path timeouts would change the default
# path.
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
#   R2C_INSTRUMENT=1 sbatch ...    # ALSO run the three read-outs the lambda0
#                                  # rule-layer audit asked for (results/r2_eval/
#                                  # lambda0_prereg/VERDICT_lambda0_rules_
#                                  # 2026-09-13.md sec 3-8): the boot banner
#                                  # (I1), a (256,512) concurrency-1 floor probe
#                                  # (I2), and two `--request-rate inf
#                                  # --max-concurrency 64` saturation ceilings
#                                  # (I3a/I3b).  ZERO additional boots: they run
#                                  # against the WARM-UP server, which boots.txt
#                                  # never names, so nothing the scorer opens is
#                                  # touched.  DEFAULT 0 = the exact command
#                                  # sequence jobs 907100/907456 ran.
#                                  # Pre-registered in newpair_prereg/.

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
# lambda0-audit instrumentation, DEFAULT OFF.  0 reproduces the 907100/907456
# command sequence exactly (not one extra process is started); 1 adds the three
# read-outs, all of them on the warm-up boot.  See the block below for why the
# warm-up boot is the only place they can go without touching the verdict.
INSTR="${R2C_INSTRUMENT:-0}"
# Allocator telemetry, DEFAULT ON IN THIS HARNESS (engine default is OFF).
# Every scored boot of every arm gets the same value, so the memory fields
# appear in legacy and true-dual records at the same cadence; a per-arm value
# would make the instrument itself the asymmetry it is meant to measure.  The
# WARM-UP boot is deliberately NOT given it: that boot runs without
# PDMUX_TELEMETRY_PATH, so the flag would be inert there anyway, and its argv
# is pinned byte-for-byte by tests/test_r2_correctness_instrument.py.
# ★This adds ~one allocator stat read per emitted runtime_snapshot to BOTH
#  arms.  It has no paired on/off measurement yet, so no run that publishes a
#  timing number may use it until one exists; this harness publishes none.
MEMTEL="${R2C_MEM_TELEMETRY:-1}"
SGPT_RAW="${R2C_SGPT_RAW:-$HF_HOME/raw/ShareGPT_V3_unfiltered_cleaned_split.json}"

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
  # Realised instrument state, not just the target.  `worker_grad_guard` is NOT
  # settable from this harness on purpose -- the engine default is the
  # registered value, and each boot's telemetry also reports what the worker
  # thread actually observed, so the two channels can disagree visibly.
  echo "mem_telemetry=$MEMTEL worker_grad_guard=<engine default> (907959 had neither)"
  echo "context_length=$CTX ($CTX_SRC) attention_backend=$ABE"
  echo "instrument=$INSTR sgpt_raw=$SGPT_RAW"
  # The manifest hashes sglang's flashinfer_backend.py (entry 25); the wheel it
  # calls into is a separate artifact and is not hashed anywhere else.
  # grep the marker, not `tail -1`: the import emits warnings on stderr and the
  # interleaving is not deterministic.
  python -c "import flashinfer; print('flashinfer_version', getattr(flashinfer, '__version__', '?'), flashinfer.__file__)" 2>&1 \
    | grep -m1 flashinfer_version || echo "flashinfer_version UNAVAILABLE"
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

# --- BEGIN r2c scope guard (added 2026-09-14 after job 908020) ---
# FAIL-CLOSED SUBSTRATE CHECK.  Job 908020 burned 0.15250 GPU-h on
# (Zamba2-2.7B, triton, ctx 4096) while its pre-registration had registered
# (NemotronH-Nano-9B-v2, flashinfer, ctx 16384).  Nothing malfunctioned: the
# registered submit command carried no R2C_* prefix, and the DEFAULTS above
# (`:177` Zamba2, `:191` triton, and a ctx derived from whatever model won)
# are a different substrate.  A pre-registration whose only registered
# execution procedure deterministically produces the wrong substrate is a
# submission-blocking defect, and three lines of defence (author, two
# rule-layer audits, submitter) all missed it.
#
# WHY THE EXPECTATION IS REQUIRED RATHER THAN DEFAULTED.  Defaulting
# R2C_EXPECT_* to one campaign's tuple would privilege that campaign inside a
# shared harness and would silently drift from the pre-registration's scope
# table.  Requiring them makes every submission STATE the substrate it
# believes it is measuring, so a forgotten `R2C_MODEL=` is caught by the
# mismatch and a forgotten expectation is caught by the unset check.  Both
# failure directions exit 2 BEFORE the first boot, so a wrong substrate costs
# seconds instead of a whole job.  No programmatic caller exists (the three
# repo references to this file are comments), so nothing automated breaks.
#
# WHAT IT COMPARES: the RESOLVED values actually about to be served -- $MODEL
# and $ABE as they will enter `server_cmd`, and $CTX after the derivation
# block above -- not the raw env.  That is the target/realized distinction
# gate #176 asks for: a typo in R2C_MODEL changes $MODEL and is caught.
scope_fail=0
for _v in R2C_EXPECT_MODEL R2C_EXPECT_BACKEND R2C_EXPECT_CTX; do
  eval "_val=\${$_v:-}"
  [ -n "$_val" ] || { echo "SCOPE_GUARD: $_v is not set" >&2; scope_fail=1; }
done
if [ "$scope_fail" = 0 ]; then
  [ "$MODEL" = "$R2C_EXPECT_MODEL" ] || { echo "SCOPE_GUARD: model '$MODEL' != expected '$R2C_EXPECT_MODEL'" >&2; scope_fail=1; }
  [ "$ABE" = "$R2C_EXPECT_BACKEND" ] || { echo "SCOPE_GUARD: backend '$ABE' != expected '$R2C_EXPECT_BACKEND'" >&2; scope_fail=1; }
  [ "$CTX" = "$R2C_EXPECT_CTX" ] || { echo "SCOPE_GUARD: context_length '$CTX' != expected '$R2C_EXPECT_CTX'" >&2; scope_fail=1; }
fi
{
  echo "scope_guard: model=$MODEL backend=$ABE ctx=$CTX"
  echo "scope_guard expected: model=${R2C_EXPECT_MODEL:-<unset>} backend=${R2C_EXPECT_BACKEND:-<unset>} ctx=${R2C_EXPECT_CTX:-<unset>}"
  echo "scope_guard: $([ "$scope_fail" = 0 ] && echo OK || echo MISMATCH)"
} | tee -a "$OUT/provenance.txt"
if [ "$scope_fail" != 0 ]; then
  echo "SCOPE_GUARD: refusing to boot (no GPU work performed)" >&2
  exit 2
fi
# --- END r2c scope guard ---

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
# --- BEGIN r2c warmup launch (extracted verbatim by tests/test_r2_correctness_instrument.py) ---
# D8: when the read-outs run here, their numbers are quoted as "measured at
# D44", so the warm-up boot's SM split must be shown REALIZED and not merely
# targeted -- this project has been bitten twice by target != realized (the
# Stage 0 "D108 anchor was actually 16 SM").  PDMUX_GREEN_READOUT is the same
# startup-only driver read-out every scored boot already takes.  It is added
# ONLY when the read-outs run: with R2C_INSTRUMENT=0 the variable is empty and
# word-splits to nothing, so the launched argv is byte-identical to the one
# jobs 907100/907456 ran.  `green_warmup.json` is not of the form
# green_<label>.json for any label the scored loop can emit (L<n>/TD<n>), and
# it is written under instrument/, so the scorer cannot open it either way.
WARM_GREEN=""
if [ "$INSTR" = "1" ]; then
  mkdir -p "$OUT/instrument"
  WARM_GREEN="PDMUX_GREEN_READOUT=1 PDMUX_GREEN_READOUT_PATH=$OUT/instrument/green_warmup.json"
fi
env PDMUX_R2_POLICY=fixed PDMUX_R2_FIXED_DSM="$DSM" $WARM_GREEN \
  $(server_cmd "$BASEPORT") > "$OUT/srv_warmup.log" 2>&1 &
WPID=$!
# --- END r2c warmup launch ---
if wait_health "$BASEPORT" "$WPID" 180; then
  WARM_OK=1; echo "  warmup up"
else
  WARM_OK=0; echo "  warmup did not come up (continuing)"
fi

# --- BEGIN r2c instrumentation (extracted verbatim by tests/test_r2_correctness_instrument.py) ---
# The three read-outs of lambda0 audit sec 3-8, at ZERO additional scored boots.
#
# WHY THE WARM-UP BOOT IS THE ONLY SAFE HOST.  r2_correctness_check.py opens
# exactly five things per label in boots.txt: gen_<label>.json, srv_<label>.log,
# tel_<label>.jsonl, green_<label>.json (main(), and boots.txt itself).  The
# warm-up boot is never written to boots.txt, and it is started WITHOUT
# PDMUX_TELEMETRY_PATH, so it emits no telemetry at all
# (multiplexing_mixin.py resolves an empty trace path to "no writer").  Nothing
# below can therefore reach the verdict.  Attaching the same load to a SCORED
# boot would NOT be safe, because three of the per-boot checks are WHOLE-FILE
# predicates over that boot's artifacts: B3 requires "cuda graph: True" on EVERY
# Decode line of srv_<label>.log, B5 requires EVERY controller_decision to be
# fixed/unlimited, and B8 requires EVERY prefill-and-decode snapshot in
# tel_<label>.jsonl to sit on the D division.  Extra load adds rows to all three.
#
# NOT a verdict input, in either direction: a failed or timed-out read-out is a
# measurement failure and is reported as one (lesson 21 / gate #21).
if [ "$INSTR" = "1" ] && [ "$WARM_OK" = "1" ]; then
  IOUT="$OUT/instrument"; mkdir -p "$IOUT"
  echo "=== instrumentation (R2C_INSTRUMENT=1; warm-up server, NOT scored) ==="
  # Line offsets into the (single, shared) warm-up server log, so each
  # read-out's slice of the engine's own decode lines can be recovered offline.
  i_mark () { echo "$1 $(wc -l < "$OUT/srv_warmup.log" | tr -d " ")" >> "$IOUT/I_log_offsets.txt"; }
  : > "$IOUT/I_log_offsets.txt"
  # I1 -- RECORDED, never judged.  ★The banner's `max_mamba_cache_size: 48` /
  # `max_running_requests=48` are NOT measurements: with --disable-radix-cache
  # and --max-running-requests 48 the engine ASSIGNS the first
  # (model_runner_kv_cache_mixin.py:223-230) and pins the second
  # (:857-876 with :390-392 returning ratio 1), independently of ctx and
  # mem-fraction; job 905835's 48 is the same identity.  What this read-out
  # actually measures is `max_total_num_tokens` and `available_gpu_mem`, which
  # come from mem_get_info at runtime -- the pre-registered forecast is F2'.
  # See newpair_prereg/ NP-7' before quoting any of this.
  grep -hE "Mamba Cache is allocated|KV Cache is allocated|max_total_num_tokens=" \
    "$OUT/srv_warmup.log" | tee "$IOUT/I1_banner_warmup.txt"
  # D8: the split the read-outs below actually ran on, REALIZED not targeted.
  # `sm_counts` is the engine's target table; green_warmup.json is what the
  # driver reports the green contexts really got.
  grep -m1 "sm_counts (prefill_sm, decode_sm)" "$OUT/srv_warmup.log" \
    | tee -a "$IOUT/I1_banner_warmup.txt"
  if [ -s "$IOUT/green_warmup.json" ]; then
    python3 -c "
import json, sys
for g in json.load(open(sys.argv[1])).get('groups', []):
    print('  green idx=%s target=(%s,%s) realized=(%s,%s) null=(%s,%s)' % (
        g.get('stream_index'), g.get('target_prefill_sm'), g.get('target_decode_sm'),
        ((g.get('prefill') or {}).get('green_sm') or {}).get('smCount'),
        ((g.get('decode') or {}).get('green_sm') or {}).get('smCount'),
        (g.get('prefill') or {}).get('green_ctx_is_null'),
        (g.get('decode') or {}).get('green_ctx_is_null')))
" "$IOUT/green_warmup.json" | tee "$IOUT/I1_green_warmup.txt"
  else
    echo "WARMUP_GREEN_READOUT_MISSING" | tee "$IOUT/I1_green_warmup.txt"
  fi
  # I2 -- shape-A floor: 8 x (256 in, 512 out) strictly one at a time, the same
  # recipe probe C used for its discarded warm-up (c_capacity.sbatch:139-145).
  # Gives the 256-token prefill TTFT floor and the B=1 ITL.
  i_mark I2_start
  timeout 600 python -m sglang.bench_serving --backend sglang --model "$MODEL" \
    --host 127.0.0.1 --port "$BASEPORT" \
    --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
    --random-input-len 256 --random-output-len 512 --random-range-ratio 1.0 \
    --num-prompts 8 --max-concurrency 1 --request-rate inf --seed 7 \
    --output-details --output-file "$IOUT/I2_shapeA_conc1.jsonl" \
    > "$IOUT/I2_shapeA_conc1.log" 2>&1
  echo "I2_rc=$?"; i_mark I2_end
  grep -E "Successful requests|Benchmark duration|Total input tokens|Total generated tokens|Mean TTFT|Median TTFT|Median ITL|Concurrency:" \
    "$IOUT/I2_shapeA_conc1.log"
  # I3 -- closed-loop saturation ceiling for the two shapes the first campaign
  # needs ((256,512) = W3 and W4's decode phase; (8192,64) = W4's prefill
  # phase).  `--request-rate inf` makes get_request skip every sleep
  # (bench_serving.py:943-945), so there is NO Poisson realisation and the
  # arrival-realisation factor 1/Ebar cannot appear.
  # ★ THIS IS AN UPPER-BOUND PROBE, NOT lambda*.  Closed-loop 64 is not
  # open-loop Poisson: probe C's saturated d44 cell reported Concurrency 24.08
  # with 22.75 of it merely waiting, so a closed loop can hold more admitted
  # work and read a HIGHER ceiling.  The client-bottleneck check is NOT the
  # `Concurrency:` line -- see the grep after the loop for why.
  for I_SPEC in "I3a_shapeA 256 512 900" "I3b_shapeB 8192 64 1500"; do
    read -r I_NAME I_IN I_OUTLEN I_TMO <<< "$I_SPEC"
    i_mark "${I_NAME}_start"
    timeout "$I_TMO" python -m sglang.bench_serving --backend sglang --model "$MODEL" \
      --host 127.0.0.1 --port "$BASEPORT" \
      --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
      --random-input-len "$I_IN" --random-output-len "$I_OUTLEN" --random-range-ratio 1.0 \
      --num-prompts 300 --request-rate inf --max-concurrency 64 --seed 41 \
      --output-details --output-file "$IOUT/${I_NAME}.jsonl" \
      > "$IOUT/${I_NAME}.log" 2>&1
    echo "${I_NAME}_rc=$?"; i_mark "${I_NAME}_end"
    grep -E "Successful requests|Benchmark duration|Request throughput|Total input tokens|Total generated tokens|Concurrency:|Median TTFT|P99 TTFT|Median ITL" \
      "$IOUT/${I_NAME}.log"
  done
  # ★ CLIENT-BOTTLENECK CHECK (confound #7).  NOT the client's `Concurrency:`
  # line: under `--max-concurrency 64` that metric counts requests from send to
  # completion, queueing included, so a closed loop pins it near 64 BY
  # CONSTRUCTION and it carries no information (probe C's OPEN-loop saturated
  # d44 cell read 24.08 with 22.75 of it merely waiting).  The informative
  # quantity is the ENGINE's own admitted batch: if `#running-req` reaches the
  # mamba-cache cap the engine is the binding constraint; if it plateaus well
  # below while 64 requests are outstanding, the bottleneck is upstream.
  grep -oE "#running-req: [0-9]+" "$OUT/srv_warmup.log" | awk '{print $2}' \
    | sort -n | tail -1 > "$IOUT/I3_max_running_req.txt"
  echo "engine max #running-req during read-outs: $(cat "$IOUT/I3_max_running_req.txt")"
  cat "$IOUT/I_log_offsets.txt"
elif [ "$INSTR" = "1" ]; then
  echo "INSTRUMENT_SKIPPED_WARMUP_DOWN"   # measurement failure, not a verdict
fi
# --- END r2c instrumentation ---

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
    PDMUX_MEM_TELEMETRY="$MEMTEL" \
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

# I1, second pass: the same banner from EVERY boot, warm-up and scored alike.
# Pure `grep` over logs that already exist; it starts no process on the GPU and
# writes only under instrument/, which the scorer never opens.
if [ "$INSTR" = "1" ]; then
  mkdir -p "$OUT/instrument"
  echo ""
  echo "=== instrumentation I1: boot banners, every boot (read-only) ==="
  for I_LOG in "$OUT"/srv_*.log; do
    [ -e "$I_LOG" ] || continue
    echo "--- $(basename "$I_LOG")"
    grep -hE "Mamba Cache is allocated|KV Cache is allocated|max_total_num_tokens=" "$I_LOG"
  done | tee "$OUT/instrument/I1_boot_banners.txt"
fi

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
