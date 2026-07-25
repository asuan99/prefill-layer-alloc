#!/bin/bash
#SBATCH --job-name=g20hard
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
# G2.0 HARDENING sweep -- derived verbatim from g2_0_full_bench.sbatch (jobs
# 863558-863567 pre + the n>=4 full sweep whose verdict is
# g2_0_full/disjoint_verdict_2026-07-24.md: DISJOINT real but razor-thin --
# feasible-A={d16,d44} (decode<=44 SM) vs feasible-B={d54} (decode>=54 SM)
# cross in the single SM-step between d44/d54, and d54-Phase-A sits ON the
# 3s TTFT cliff (0.852 frac_good, one collapse to 0.531). This sweep runs
# TWO axes to move the disjoint off the razor edge (gate #8: re-run, do not
# re-score):
#   Axis 1 (widen the gap): make Phase B heavier (OB env override, default
#     still 512, campaign driver passes 1024) so decode-SM needed for ITL<=
#     50ms shifts up past d54, towards d64/d74 -- widening feasible-B away
#     from feasible-A. New pdmux_d64.yml ([44,64,0]) / pdmux_d74.yml
#     ([34,74,0]) added under slo_sched/ alongside the existing d16..d54.
#   Axis 2 (de-cliff d54/d44 Phase A): same OB=512 as the original full
#     sweep, additional reps at d54/d44 (n>=6 total) at rA5 (matches
#     existing 4 reps) AND at a slightly sub-cliff rA4 to pin whether
#     d54-A is really ~0.85 or <0.5, and whether the fragility is rate-
#     cliff-adjacency rather than a real regime.
# Only ONE addition vs g2_0_full_bench.sbatch beyond the new d64/d74 case
# arms: the drain/idle poll gets a second acceptance path (drain_ok=2) when
# the server scheduler log stops emitting new "Prefill batch"/"Decode
# batch" lines for 2 consecutive 1s polls (no new batch => nothing running
# to report a fresh running/queue count, which is why drain_ok=0 fired
# every time in g2_0_full despite zero measured carryover). This is a modest
# fix, not a rewrite -- the original running==0/queued==0 line match is
# still tried first every poll.
#   arg1 MODE (d16|d24|d34|d44|d54|d64|d74) ; arg2 REP ; arg3 RATE_A ; arg4 RATE_B
# Env overrides: ROUNDS (default 1), NPROMPT (default 32), LA/OA/LB/OB
# (workload sizes; axis-1 caller sets OB=1024).
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
OUT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/g2_0_hard
CFGDIR=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="/scratch/ehmoon/whlee/prefill-layer-alloc/triage/.triton_cache_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

MODE="${1:-d24}"; REP="${2:-1}"; RATE_A="${3:-5}"; RATE_B="${4:-4}"
MODEL=Zyphra/Zamba2-2.7B; CTX=4096; NPROMPT="${NPROMPT:-32}"; ROUNDS="${ROUNDS:-1}"
LA="${LA:-2048}"; OA="${OA:-32}"; NA="${NA:-$NPROMPT}"
LB="${LB:-2048}"; OB="${OB:-512}"; NB="${NB:-$NPROMPT}"

case "$MODE" in
  d16|d24|d34|d44|d54|d64|d74) CFG="$CFGDIR/pdmux_${MODE}.yml"; unset PDMUX_SLO_SCHED PDMUX_SLO_MODE ;;
  *) echo "bad MODE=$MODE (this sweep is static-split only, no bind/agn arms)"; exit 1 ;;
esac

PORT=$((36500 + SLURM_JOB_ID % 150))
TAG="${MODE}_${REP}_rA${RATE_A}B${RATE_B}_OB${OB}_${SLURM_JOB_ID}"
SRVLOG="$OUT/g20h_srv_${TAG}.log"; BENCHLOG="$OUT/g20h_bench_${TAG}.log"
JA="$OUT/g20hA_${TAG}.jsonl"; JB="$OUT/g20hB_${TAG}.jsonl"; : > "$JA"; : > "$JB"
echo "G20HARD TAG=$TAG MODE=$MODE rA=$RATE_A rB=$RATE_B OB=$OB rounds=$ROUNDS nprompt=$NPROMPT port=$PORT job=$SLURM_JOB_ID triton_cache=$TRITON_CACHE_DIR"; cat "$CFG"

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
print(f"G20H_RESULT tag={tag} rate={rate} | phaseA(prefill) gp={gpA:.3f} | phaseB(decode) gp={gpB:.3f} | COMBINED={gpC:.3f} | min-phase={min(gpA,gpB):.3f} | durA={dA:.1f} durB={dB:.1f}")
print("NOTE: this scorer is LEGACY mean-ITL<=60ms (secondary only). Canonical")
print("scoring (TTFT<=3000ms AND per-request token-ITL p95<=50ms) is done by")
print("result-analyst directly from the jsonl ttfts/itls/input_lens fields.")
def pl(nm,tt,il):
    return (f"G20H_PCT tag={tag} {nm} | TTFTs p50/p95/p99={pct(tt,.5):.2f}/{pct(tt,.95):.2f}/{pct(tt,.99):.2f} "
            f"| ITLms p50/p95/p99={pct(il,.5):.1f}/{pct(il,.95):.1f}/{pct(il,.99):.1f}")
print(pl("A",ttA,ilA)); print(pl("B",ttB,ilB))
PY
echo "DONE_${TAG}"
rm -rf "$TRITON_CACHE_DIR"
