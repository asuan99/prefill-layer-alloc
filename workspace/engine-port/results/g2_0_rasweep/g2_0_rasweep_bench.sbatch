#!/bin/bash
#SBATCH --job-name=g20ra
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:50:00
#SBATCH --comment=pytorch
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# ============================================================================
# G2.0 rA CONFIRMATION SWEEP -- claims-auditor pre-registered stage-2 for the
# vector-1 (short-ctx disjoint) claim. Derived verbatim (phase/drain/scorer
# logic unchanged) from g2_0_decliff/g2_0_decliff_bench.sbatch.
#
# Context: g2_0_decliff stage-1 found rA2 (only tested off-cliff Phase-A
# rate) has NO disjoint (static d54 covers both phases), but the onset of
# d54-exclusion (~rate 3.0-3.5) was itself unmeasured/n=3/bimodal there --
# so "vector-1 settled" could not yet be claimed. This sweep densely and
# adequately-powered fills that transition band.
#
# Design (auditor recipe, exact):
#   rate_A in {2.25, 2.5, 2.75, 3.0, 3.25}   (5 rates, float --request-rate)
#   split  in {d16, d34, d44, d54}            (4 splits; d34 added for a
#                                               >=2-SM-step gap vs d16/d44/d54)
#   n=6 per (rate x split) cell, EACH REP A SEPARATE SERVER BOOT (no shared
#   state across reps).
#   = 5 x 4 x 6 = 120 jobs total.
# Phase B held fixed at rB=4/OB=512 (same as g2_0_decliff/full/hard) --
# single-variable=rate_A only, all else (seed policy, backend, cudagraph,
# commit, manifest) identical.
#
# manifest: byte-identical to g2_0_full/g2_0_hard/g2_0_decliff
# (workspace/engine-port/results/g2_0_rasweep/runtime_source_manifest.sha256,
# diffed clean against g2_0_decliff's copy at submission time -- confirmed
# 2026-07-25).
#
#   arg1 MODE (d16|d24|d34|d44|d54|d64|d74) ; arg2 REP ; arg3 RATE_A ; arg4 RATE_B
# Env overrides: ROUNDS (default 1), NPROMPT (default 32), LA/OA/LB/OB
# (workload sizes; default OB=512, unchanged from g2_0_full/hard baseline).
#
# Telemetry: --output-details is on (as in g2_0_decliff), so the per-request
# ttfts[], per-token itls[][] (raw, un-aggregated), input_lens[], and
# output_lens[] arrays land in the JA/JB jsonl untouched -- sufficient for
# result-analyst to do output-invariant ITL rescoring (first-K exclusion,
# tail percentiles) and TTFT-vs-ITL fail-cause decomposition WITHOUT
# rerunning (gate #8: rescore from raw, never re-tune-then-compare). This
# script's own G20RA_RESULT/G20RA_PCT stdout lines are a LEGACY/orientation
# scorer only, not the analysis of record.
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/g2_0_rasweep
CFGDIR=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="/scratch/ehmoon/whlee/prefill-layer-alloc/triage/.triton_cache_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

MODE="${1:-d24}"; REP="${2:-1}"; RATE_A="${3:-3}"; RATE_B="${4:-4}"
MODEL=Zyphra/Zamba2-2.7B; CTX=4096; NPROMPT="${NPROMPT:-32}"; ROUNDS="${ROUNDS:-1}"
LA="${LA:-2048}"; OA="${OA:-32}"; NA="${NA:-$NPROMPT}"
LB="${LB:-2048}"; OB="${OB:-512}"; NB="${NB:-$NPROMPT}"

case "$MODE" in
  d16|d24|d34|d44|d54|d64|d74) CFG="$CFGDIR/pdmux_${MODE}.yml"; unset PDMUX_SLO_SCHED PDMUX_SLO_MODE ;;
  *) echo "bad MODE=$MODE (this sweep is static-split only, no bind/agn arms)"; exit 1 ;;
esac

PORT=$((36500 + SLURM_JOB_ID % 150))
TAG="${MODE}_${REP}_rA${RATE_A}B${RATE_B}_OB${OB}_${SLURM_JOB_ID}"
SRVLOG="$OUT/g20ra_srv_${TAG}.log"; BENCHLOG="$OUT/g20ra_bench_${TAG}.log"
JA="$OUT/g20raA_${TAG}.jsonl"; JB="$OUT/g20raB_${TAG}.jsonl"; : > "$JA"; : > "$JB"
echo "G20RASWEEP TAG=$TAG MODE=$MODE rA=$RATE_A rB=$RATE_B OB=$OB rounds=$ROUNDS nprompt=$NPROMPT port=$PORT job=$SLURM_JOB_ID triton_cache=$TRITON_CACHE_DIR"; cat "$CFG"

python -m sglang.launch_server --model-path $MODEL --trust-remote-code --dtype bfloat16 \
  --attention-backend triton --disable-radix-cache \
  --mem-fraction-static 0.82 --max-running-requests 48 --context-length $CTX \
  --enable-pdmux --pdmux-config-path "$CFG" --chunked-prefill-size -1 --disable-overlap-schedule \
  --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!; ok=0
for i in $(seq 1 180); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; break; }; sleep 5
done
echo "boot_ok=$ok"
if [ "$ok" = 1 ]; then
  curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
    -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":6,"temperature":0}}' | grep -oE '"text":"[^"]*"' | head -1 | sed 's/^/SANITY /'
  for round in $(seq 1 $ROUNDS); do
    echo "== round $round PHASE A (in${LA}/o${OA} @${RATE_A}) =="
    python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
      --dataset-name random-ids --random-input-len $LA --random-output-len $OA --random-range-ratio 1.0 \
      --num-prompts $NA --request-rate $RATE_A --output-details --output-file "$JA" 2>>"$BENCHLOG" | grep -E "Median TTFT|Median TPOT"
    # --- drain/settle: try the exact running==0/queued==0 scheduler line
    # first every poll; if the log has simply stopped emitting new batch
    # lines (server idle, nothing to report) for 2 consecutive polls,
    # accept that as idle too (drain_ok=2) instead of spinning to the
    # poll cap and reporting a stale drain_ok=0 despite true idleness. ---
    drain_ok=0; qline=""; prev_cnt=-1; stable=0
    for di in $(seq 1 30); do
      cnt=$(grep -cE "Prefill batch|Decode batch" "$SRVLOG" 2>/dev/null || echo 0)
      qline=$(grep -E "Prefill batch|Decode batch" "$SRVLOG" | tail -1)
      running=$(echo "$qline" | grep -oE "#running-req: [0-9]+" | grep -oE "[0-9]+" | head -1)
      queued=$(echo "$qline" | grep -oE "#queue-req: [0-9]+" | grep -oE "[0-9]+" | head -1)
      if [ "${running:-1}" = "0" ] && [ "${queued:-1}" = "0" ]; then drain_ok=1; break; fi
      if [ "$cnt" = "$prev_cnt" ]; then
        stable=$((stable+1))
        if [ "$stable" -ge 2 ]; then drain_ok=2; break; fi
      else
        stable=0
      fi
      prev_cnt=$cnt
      sleep 1
    done
    sleep 3  # fixed settle margin beyond scheduler-idle detection
    echo "DRAIN_STATUS round=$round drain_ok=$drain_ok last_sched_line=[$qline]"
    echo "== round $round PHASE B (in${LB}/o${OB} @${RATE_B}) =="
    python -m sglang.bench_serving --backend sglang --model $MODEL --host 127.0.0.1 --port $PORT \
      --dataset-name random-ids --random-input-len $LB --random-output-len $OB --random-range-ratio 1.0 \
      --num-prompts $NB --request-rate $RATE_B --output-details --output-file "$JB" 2>>"$BENCHLOG" | grep -E "Median TTFT|Median TPOT"
  done
else echo "BOOT FAILED"; tail -20 "$SRVLOG"|grep -viE "pynvml|Future"; fi
kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 2
python3 - "$JA" "$JB" "$TAG" "${RATE_A}/${RATE_B}" <<'PY'
import json,sys
def pct(a,q):
    if not a: return 0.0
    a=sorted(a); return a[min(len(a)-1,int(q*len(a)))]
def load(f):
    g=0;dur=0.0;tts=[];itls=[]
    for ln in open(f):
        ln=ln.strip()
        if not ln: continue
        o=json.loads(ln); tt=o.get("ttfts") or []; il=o.get("itls") or []; d=o.get("duration") or 0
        dur+=d   # SUM across rounds (gate #7) -- NOT max
        for i,t in enumerate(tt):
            m=(sum(il[i])/len(il[i])) if (i<len(il) and il[i]) else 9.0
            tts.append(t); itls.append(m*1000.0)
            if t<=3.0 and m<=0.06: g+=1
    return g,dur,tts,itls
ja,jb,tag,rate=sys.argv[1:5]
gA,dA,ttA,ilA=load(ja); gB,dB,ttB,ilB=load(jb)
gpA=gA/dA if dA else 0; gpB=gB/dB if dB else 0; gpC=(gA+gB)/(dA+dB) if (dA+dB) else 0
print(f"G20RA_RESULT tag={tag} rate={rate} | phaseA(prefill) gp={gpA:.3f} | phaseB(decode) gp={gpB:.3f} | COMBINED={gpC:.3f} | min-phase={min(gpA,gpB):.3f} | durA={dA:.1f} durB={dB:.1f}")
print("NOTE: this scorer is LEGACY mean-ITL<=60ms (secondary only), included for")
print("continuity with g2_0_decliff/full/hard logs. THIS SBATCH DOES NOT JUDGE")
print("FEASIBILITY OR DISJOINTNESS. Canonical scoring (fail-cause decomposition,")
print("output-invariant tail-ITL rescoring from raw ttfts/itls/input_lens) is")
print("done by result-analyst directly from the jsonl.")
def pl(nm,tt,il):
    return (f"G20RA_PCT tag={tag} {nm} | TTFTs p50/p90/p95/p99={pct(tt,.5):.2f}/{pct(tt,.9):.2f}/{pct(tt,.95):.2f}/{pct(tt,.99):.2f} "
            f"| ITLms p50/p95/p99={pct(il,.5):.1f}/{pct(il,.95):.1f}/{pct(il,.99):.1f}")
print(pl("A",ttA,ilA)); print(pl("B",ttB,ilB))
PY
echo "DONE_${TAG}"
rm -rf "$TRITON_CACHE_DIR"
