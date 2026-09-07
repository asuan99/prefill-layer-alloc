#!/bin/bash
#SBATCH --job-name=af1
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/cp_baseline/%x_%A_%a.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/cp_baseline/%x_%A_%a.err
#
# AF-1 -- the UNCONTENDED FLOOR probe.  Rule layer GO'd on the 6th pass
# (`audit_af1_rules_6th_2026-09-06/VERDICT.md`); this is stage 2 of gate #34.
#
# ★ WHAT THIS BUYS (pre-registration sec 5.1, "이 프로브가 실제로 사는 것"):
#     (i)  a test of the five registered predictions -- prediction 2 is ALREADY
#          falsified at GPU 0 and is recorded as such, not corrected;
#     (ii) the PRUNE LIST -- which of CP-2's 36 ladder points are physically
#          unreachable;
#     (iii) one bit of arm MFU heterogeneity (the 53.6% threshold);
#     (iv) the FIRST measurement of this axis's between-boot SD on this model.
#
# ★ WHAT IT IS NOT: not a policy comparison, not goodput, not an arm ranking.
#   Concurrency is 1 and no arrival process exists.  "chunked prefill vs PD-mux
#   which is better" cannot come out of this probe (sec 8, forbidden sentences).
#
# One array task = one BOOT ROUND: a throwaway warm-up boot, then the three arms
# in this task's registered rotation (`ARM_ORDER_BY_JOB`), each booted alone and
# measured at concurrency 1.  Arm contrast is therefore WITHIN a node and the
# between-boot SD is ACROSS tasks (sec 3.1).
#
# usage:  python3 af1_prompts.py --seed <s> --out prompts_seed<s>.json   (x6, CPU)
#         sbatch --array=0-5 af1_probe.sbatch

set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc || exit 2

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/cp_baseline
SLO=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
TASK=${SLURM_ARRAY_TASK_ID:-0}
OUT="$ROOT/af1_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${TASK}"; mkdir -p "$OUT"

export HF_HOME=/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$OUT/.triton_cache"; mkdir -p "$TRITON_CACHE_DIR"
unset PDMUX_CHUNK_PROBE_PATH PDMUX_SLO_SCHED PDMUX_SLO_MODE PDMUX_TRUE_DUAL_WORKER

MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
BACKEND="flashinfer"
CTX=32768

# The rotation and the seed come from the registered constants, read out of the
# predicate module so this script cannot drift from the pre-registration.
read -r ARM1 ARM2 ARM3 <<< "$(python3 -c "
import sys; sys.path.insert(0,'$ROOT'); import af1_predicates as P
print(' '.join(P.ARM_ORDER_BY_JOB[$TASK]))")"
SEED=$(python3 -c "
import sys; sys.path.insert(0,'$ROOT'); import af1_predicates as P
print(P.BOOT_SEEDS[$TASK])")
PROMPTS="$ROOT/prompts_seed${SEED}.json"
[ -f "$PROMPTS" ] || { echo "MISSING $PROMPTS -- run af1_prompts.py first"; exit 2; }

echo "AF1 task=$TASK node=$SLURMD_NODENAME seed=$SEED order=[$ARM1 $ARM2 $ARM3]"
echo "AF1 model=$MODEL backend=$BACKEND ctx=$CTX job=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
python3 -c "
import json,sys; sys.path.insert(0,'$ROOT')
s=json.load(open('$PROMPTS'))
print('AF1 strata:', [(x['q'], x['prompt_tokens']) for x in s['strata']])"

echo "=== env sync ==="
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh \
  "$OUT/runtime_source_manifest.sha256" || exit 2

arm_flags () {
  case "$1" in
    plain)  echo "--chunked-prefill-size -1" ;;
    cp2048) echo "--chunked-prefill-size 2048" ;;
    d44)    echo "--enable-pdmux --pdmux-config-path $SLO/pdmux_d44.yml --chunked-prefill-size -1 --disable-overlap-schedule" ;;
    *) echo "__BAD__" ;;
  esac
}

boot () {   # $1=arm  $2=port  $3=logfile  $4=ctx
  local FLAGS; FLAGS=$(arm_flags "$1")
  [ "$FLAGS" = "__BAD__" ] && return 2
  python3 -m sglang.launch_server --model-path "$MODEL" --trust-remote-code \
    --dtype bfloat16 --attention-backend "$BACKEND" $FLAGS \
    --disable-piecewise-cuda-graph --disable-radix-cache --mem-fraction-static 0.82 \
    --max-running-requests 8 --context-length "$4" \
    --host 127.0.0.1 --port "$2" > "$3" 2>&1 &
  echo $!
}

wait_health () {   # $1=port  $2=pid
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$1/health 2>/dev/null||true)" = 200 ] && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

# Warm-up boot: NOT scored.  job 902397 lost its first arm to a cold-start
# timeout and that BOOT_FAILED was a harness artefact, not a model fact
# (`correctness_gate.sbatch:65-67`).  The arm is fixed to `plain` (sec 3), and
# the residual asymmetry that creates is registered there.
echo "=== warmup boot (plain, not scored) ==="
WPORT=$((41000 + RANDOM % 400))
WPID=$(boot plain "$WPORT" "$OUT/srv_warmup.log" 4096)
wait_health "$WPORT" "$WPID" && echo "  warmup ok" || echo "  warmup did not come up (continuing)"
kill "$WPID" 2>/dev/null; sleep 5; pkill -9 -f "launch_server.*--port $WPORT" 2>/dev/null; sleep 15

BOOTED=""
for ARM in "$ARM1" "$ARM2" "$ARM3"; do
  PORT=$((40000 + RANDOM % 500))
  SRVLOG="$OUT/srv_${ARM}.log"
  echo ""
  echo "########## arm=$ARM port=$PORT flags=[$(arm_flags "$ARM")] ##########"
  PID=$(boot "$ARM" "$PORT" "$SRVLOG" "$CTX")
  if wait_health "$PORT" "$PID"; then
    echo "  boot_ok=1"
    # MANDATORY OUTPUT: server-args echo (sec 3.2, "항상")
    grep -oE "chunked_prefill_size=[-0-9]+|enable_pdmux=[A-Za-z]+|disable_piecewise_cuda_graph=[A-Za-z]+|disable_overlap_schedule=[A-Za-z]+|attention_backend='[a-z]+'|context_length=[0-9]+" \
      "$SRVLOG" | sort -u | tr '\n' ' ' | tee "$OUT/args_${ARM}.txt"; echo
    python3 "$ROOT/af1_client.py" --port "$PORT" --arm "$ARM" --seed "$SEED" \
      --prompts "$PROMPTS" --node "${SLURMD_NODENAME:-unknown}" \
      --out "$OUT/boot_${ARM}.json" && BOOTED="$BOOTED $ARM"
  else
    echo "BOOT_FAILED arm=$ARM" | tee -a "$OUT/BOOT_FAILURES.txt"
    tail -12 "$SRVLOG" | grep -viE "pynvml|Future|^\s*$" | tail -4
  fi
  # MANDATORY OUTPUT: realised partition readout -- `d44` ONLY.  `plain` and
  # `cp2048` have no `--enable-pdmux`, so partition telemetry is PRINCIPALLY
  # ABSENT for them (sec 3.2; canon `CONSENSUS.md` sec 1-1).  The emitter's field
  # names are `current_decode_sms` / `target_decode_sms`, not `decode_sms`
  # (`multiplexing_mixin.py:446`).
  if [ "$ARM" = "d44" ]; then
    # (a) CONFIGURED groups -- the engine prints the whole `sm_counts` table at
    # boot.  Smoke 904845_0 showed the per-step fields below can legitimately be
    # empty (no group switch ever happens at concurrency 1), which left the
    # artifact with no record of what the arm was even set up to do.  ADDITIVE
    # repair (addenda sec E): keep (b) unchanged, record the config line too.
    grep -oE "PD-Multiplexing enabled with .*" "$SRVLOG" \
      | sed 's/^/CONFIGURED /' | sort -u | tee "$OUT/partition_${ARM}.txt"
    # (b) REALISED per-step readout (unchanged emitter fields, `multiplexing_mixin.py:446`)
    grep -oE "current_decode_sms=[0-9]+|target_decode_sms=[0-9]+|smCount=[0-9]+|manual_divisions[^ ]*" \
      "$SRVLOG" | sort | uniq -c | sed 's/^/REALISED /' | tee -a "$OUT/partition_${ARM}.txt"
    grep -q "^REALISED" "$OUT/partition_${ARM}.txt" \
      || echo "REALISED NO_GROUP_SWITCH_TELEMETRY (no switch fired; expected at concurrency 1 -- addenda sec E)" \
         | tee -a "$OUT/partition_${ARM}.txt"
  else
    echo "PRINCIPALLY_ABSENT (no --enable-pdmux; canon CONSENSUS sec 1-1)" > "$OUT/partition_${ARM}.txt"
  fi
  kill "$PID" 2>/dev/null; sleep 5; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 15
done

python3 - "$OUT" "$TASK" "$SEED" "${SLURMD_NODENAME:-unknown}" "$BOOTED" <<'PY'
import json, sys
out, task, seed, node, booted = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5].split()
json.dump({"task": task, "seed": seed, "node": node, "arms_scored": booted},
          open(f"{out}/round.json", "w"))
print("AF1 round summary:", {"task": task, "node": node, "arms_scored": booted})
PY
echo "AF1_DONE_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${TASK}"
