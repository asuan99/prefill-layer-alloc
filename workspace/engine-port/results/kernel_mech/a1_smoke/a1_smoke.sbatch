#!/bin/bash
#SBATCH --job-name=a1smoke
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech/a1_smoke
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech/a1_smoke/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech/a1_smoke/%x_%j.err
# ============================================================================
# A1 rev2 sec 2.3 SMOKE = the boot the design calls B-S16'.  NOT an experiment.
#
# ★It produces NO SCORED VERDICT.  Neither rule is applied to it; nothing here
# is attributed, joined or labelled.  That is why it can be bought while the
# rules-layer audit stands at NO-GO -- it is a plumbing smoke (gate #25), and
# gate #34's rules-before-harness order is about the SCORING harness.
#
# It buys exactly the three things the 2nd audit showed were still unbought
# after job 872800 already established Ha8 + sticky (boot, sticky log,
# correctness PASS, realized 1.0000 on ShareGPT):
#
#   (a) does E1_DECODE_REALIZED(16) >= 0.99 hold in the BATCH-SYNCHRONOUS
#       workload?  ★All 21 boots behind that threshold are ShareGPT open-loop,
#       a population with no round boundary -- and this cell has two rounds by
#       construction.  sec 5.1a registers that this is the only place that gap
#       is bought, and that a miss re-opens the registered value rather than
#       the run.
#   (b) is the Q3 channel (`decode_iterations`) non-zero here?  The decision
#       document fixed that channel on existing telemetry; this is the first
#       time it is read on A1's own cell.
#   (c) PDMUX_GREEN_READOUT's FIRST EXECUTION ON A BOOT -- it has unit tests
#       and 0 boot runs.
#
# ★And it is not spent: this boot IS K1's nsys-OFF leg (sec 2.2a).
#
# NO nsys.  NO performance statement.  Two rounds, 16 requests each.
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
HERE=$ROOT/workspace/engine-port/results/kernel_mech/a1_smoke
E1=$ROOT/workspace/engine-port/results/s8_frontier
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache_a1s_${SLURM_JOB_ID:-local}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

"$ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh" \
  "$HERE/runtime_source_manifest.sha256" || { echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }

ARM=Ha8
MODEL="Zyphra/Zamba2-7B-Instruct"
BACKEND=triton
CELL=d16; DSM=16
CFG="$E1/pdmux_e1_${CELL}.yml"
# ★B6 / rev8 cell: batch-synchronous NP = CONC = 16, out = 128, L = 1024.
NP=16; CONC=16; INLEN=1024; OUTLEN=128; ROUNDS=2
CAP=48
# ★step 3 (knob purity) R1: the mamba pool is PINNED and the banner recorded.
MAMBA_POOL=48
CTXCAP=$(( INLEN + OUTLEN + 512 ))

JOBTAG="${SLURM_JOB_ID:-local}"
RUNID="a1smoke_${ARM}_${CELL}_${JOBTAG}"
RESULT="$HERE/${RUNID}_result.txt"; : > "$RESULT"
SRVLOG="$HERE/${RUNID}_srv.log"
TELE="$HERE/${RUNID}_telemetry.jsonl"; : > "$TELE"
GREEN="$HERE/${RUNID}_green_readout.json"

echo "=== A1 SMOKE (B-S16') arm=$ARM cell=$CELL D=$DSM NP=$NP CONC=$CONC L=$INLEN out=$OUTLEN rounds=$ROUNDS job=$JOBTAG ===" | tee -a "$RESULT"
echo "    NO nsys · NO scored verdict · buys (a) realized in batch-sync (b) decode_iterations non-zero (c) green read-out first boot" | tee -a "$RESULT"
[ -f "$CFG" ] || { echo "MISSING_CONFIG $CFG" | tee -a "$RESULT"; exit 4; }

PORT=$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")

export PDMUX_TELEMETRY_PATH="$TELE"
export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="a1_smoke_${ARM}_${CELL}"
export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=$DSM
export PDMUX_TRACE_FORCE_PREFILL=0
export PDMUX_STICKY_PARTITION=1
# ★(c) first boot execution of the green read-out.
export PDMUX_GREEN_READOUT=1
export PDMUX_GREEN_READOUT_PATH="$GREEN"

python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype bfloat16 \
  --attention-backend $BACKEND \
  --enable-pdmux --pdmux-config-path "$CFG" \
  --disable-piecewise-cuda-graph \
  --disable-overlap-schedule --chunked-prefill-size -1 --disable-radix-cache \
  --mem-fraction-static 0.80 \
  --max-running-requests $CAP --max-mamba-cache-size $MAMBA_POOL \
  --context-length $CTXCAP \
  --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 180); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED" | tee -a "$RESULT"; break; }
  sleep 5
done
echo "boot_ok=$ok" | tee -a "$RESULT"
grep -c "sticky partition ENABLED" "$SRVLOG" | sed "s/^/  STICKY_LOG_LINES: /" | tee -a "$RESULT"
# step 3 R3: the banner triple, recorded per cell.
grep -E "Mamba Cache is allocated|max_mamba_cache_size|KV Cache is allocated|max_total_num_tokens|max_running_requests" \
     "$SRVLOG" | head -6 | sed "s/^/  BANNER: /" | tee -a "$RESULT"
# (c) the read-out fired at startup
if [ -s "$GREEN" ]; then
  echo "  GREEN_READOUT_WRITTEN=1" | tee -a "$RESULT"
  python3 -c "
import json
d = json.load(open('$GREEN'))
print('  green groups:', len(d['groups']), 'sticky_target_index:', d['sticky_target_index'])
for g in d['groups']:
    dec = g.get('decode', {})
    print(f\"    idx={g['stream_index']} target=(p{g['target_prefill_sm']},d{g['target_decode_sm']}) \"
          f\"sticky={g['is_sticky_target']} decode_green_sm={dec.get('green_sm')} \"
          f\"null={dec.get('green_ctx_is_null')} err={dec.get('error','-')}\")
" | tee -a "$RESULT"
else
  echo "  GREEN_READOUT_WRITTEN=0  <-- (c) FAILED" | tee -a "$RESULT"
fi

if [ "$ok" = 1 ]; then
  for R in $(seq 1 $ROUNDS); do
    echo "--- round $R ---" | tee -a "$RESULT"
    python -m sglang.bench_serving --backend sglang --model "$MODEL" \
      --host 127.0.0.1 --port $PORT \
      --dataset-name random-ids --random-input-len $INLEN --random-output-len $OUTLEN \
      --random-range-ratio 1.0 --num-prompts $NP --max-concurrency $CONC \
      --request-rate inf --seed $R --output-file "$HERE/${RUNID}_r${R}.jsonl" \
      > "$HERE/${RUNID}_r${R}_bench.log" 2>&1
    echo "  round $R rc=$?" | tee -a "$RESULT"
  done
fi
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null

# ---- (a) realized, canonical estimator · (b) the Q3 channel ----------------
python3 - "$TELE" "$DSM" <<'PY' | tee -a "$RESULT"
import json, sys
sys.path.insert(0, "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/s8_frontier")
tele, dsm = sys.argv[1], int(sys.argv[2])
try:
    from e1_pin_check import compute_decode_realized
    r = compute_decode_realized(tele, dsm)
    hist = ", ".join(f"D{k}:{v:.2f}s" for k, v in
                     sorted(r["decode_hist"].items(), key=lambda kv: -kv[1])[:4])
    print(f"  (a) E1_DECODE_REALIZED D={dsm} = {r['decode_realized_frac']:.4f} "
          f"(t_decode_active={r['t_decode_active_s']:.1f}s)  hist: {hist}")
    print(f"      GATE >= 0.99 -> {'PASS' if r['decode_realized_frac'] >= 0.99 else 'FAIL'}")
except Exception as exc:
    print(f"  (a) realized UNCOMPUTABLE: {exc!r}")

first = last = None
n = 0
phases = {}
for line in open(tele):
    try:
        rec = json.loads(line)
    except ValueError:
        continue
    if rec.get("event") != "runtime_snapshot":
        continue
    n += 1
    phases[rec.get("phase")] = phases.get(rec.get("phase"), 0) + 1
    v = int(rec.get("decode_iterations", 0))
    if first is None:
        first = v
    last = v
span = (last - first) if first is not None else None
print(f"  (b) decode_iterations {first} -> {last}  span={span}  "
      f"snapshots={n} phases={phases}")
print(f"      GATE span > 0 -> {'PASS' if span else 'FAIL'}")
PY
echo "=== A1 SMOKE DONE job=$JOBTAG ===" | tee -a "$RESULT"
