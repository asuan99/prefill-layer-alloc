#!/bin/bash
#SBATCH --job-name=g20dc
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:50:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# ============================================================================
# G2.0 DE-CLIFF sweep -- STAGE 1 (capacity), derived verbatim from
# g2_0_hard_bench.sbatch. Context: claims-auditor re-analysis found the
# g2_0_full/g2_0_hard disjoint verdict (feasible-A={d16,d44} vs
# feasible-B={d54}) was measured ON the Phase-A 3s TTFT cliff (bimodal
# collapse, e.g. rA5/d54-A 0.852 with one run collapsing to 0.531) --
# ill-posed per gate #6 (capacity before feasibility). Before any feasibility
# re-judgment we must first find a Phase-A rate that is de-cliffed: p90 TTFT
# well under the 3s SLO (target <~2s) with ~0 spread across reps/splits.
#
# This sbatch ONLY varies Phase-A rate (RATE_A in {2,3,4}, float-capable via
# --request-rate) and split (d16/d44/d54 -- the splits straddling the
# g2_0_full/hard cliff). Phase B stays at the ORIGINAL g2_0_full/hard
# defaults (rB=4, OB=512) since Phase-A de-cliffing is independent of Phase
# B; the existing drain/settle logic (incl. the drain_ok=2 idle-log
# acceptance path from g2_0_hard) still runs between phases so Phase-B
# backlog cannot leak into the NEXT round's Phase-A measurement, and so
# Phase-A telemetry in this campaign is not contaminated by carryover.
#
# Two-stage plan (see .campaign_drv.sh for the actual matrix):
#   1a. coarse locate: n=3 across rA in {2,3,4} x split in {d16,d44,d54}
#       (9 conditions x 3 = 27 jobs) to find which rA looks de-cliffed
#       (p90 << 3s, low spread) vs still cliff-adjacent.
#   1b. confirm: n>=6 at the de-cliffed rA candidate(s) identified from 1a,
#       same 3 splits, to pin bimodal-collapse frequency with real power.
# Stage-2 (output-invariant ITL + >=2-SM-step gap feasibility re-judgment at
# the confirmed de-cliff rate) is a SEPARATE submission by result-analyst
# after stage-1 lands -- this script does NOT judge feasibility.
#
# manifest: byte-identical to g2_0_full/g2_0_hard
# (workspace/engine-port/results/g2_0_decliff/runtime_source_manifest.sha256,
# diffed clean against g2_0_hard's copy at submission time).
#
#   arg1 MODE (d16|d24|d34|d44|d54|d64|d74) ; arg2 REP ; arg3 RATE_A ; arg4 RATE_B
# Env overrides: ROUNDS (default 1), NPROMPT (default 32), LA/OA/LB/OB
# (workload sizes; default OB=512, unchanged from g2_0_full/hard baseline).
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/g2_0_decliff
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
SRVLOG="$OUT/g20dc_srv_${TAG}.log"; BENCHLOG="$OUT/g20dc_bench_${TAG}.log"
JA="$OUT/g20dcA_${TAG}.jsonl"; JB="$OUT/g20dcB_${TAG}.jsonl"; : > "$JA"; : > "$JB"
echo "G20DECLIFF TAG=$TAG MODE=$MODE rA=$RATE_A rB=$RATE_B OB=$OB rounds=$ROUNDS nprompt=$NPROMPT port=$PORT job=$SLURM_JOB_ID triton_cache=$TRITON_CACHE_DIR"; cat "$CFG"

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
print(f"G20DC_RESULT tag={tag} rate={rate} | phaseA(prefill) gp={gpA:.3f} | phaseB(decode) gp={gpB:.3f} | COMBINED={gpC:.3f} | min-phase={min(gpA,gpB):.3f} | durA={dA:.1f} durB={dB:.1f}")
print("NOTE: this scorer is LEGACY mean-ITL<=60ms (secondary only), included for")
print("continuity with g2_0_full/hard logs. STAGE-1 PURPOSE IS CAPACITY/DE-CLIFF")
print("ONLY (Phase-A TTFT p50/p90/p99 spread) -- no feasibility judgment here.")
print("Canonical scoring is done by result-analyst directly from the jsonl")
print("ttfts/itls/input_lens fields.")
def pl(nm,tt,il):
    return (f"G20DC_PCT tag={tag} {nm} | TTFTs p50/p90/p95/p99={pct(tt,.5):.2f}/{pct(tt,.9):.2f}/{pct(tt,.95):.2f}/{pct(tt,.99):.2f} "
            f"| ITLms p50/p95/p99={pct(il,.5):.1f}/{pct(il,.95):.1f}/{pct(il,.99):.1f}")
print(pl("A",ttA,ilA)); print(pl("B",ttB,ilB))
PY
echo "DONE_${TAG}"
rm -rf "$TRITON_CACHE_DIR"
