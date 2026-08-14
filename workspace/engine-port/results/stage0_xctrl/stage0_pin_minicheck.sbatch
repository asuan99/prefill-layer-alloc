#!/bin/bash
#SBATCH --job-name=s0pin
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=00:40:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
# ============================================================================
# STAGE 0 PIN MINI-CHECK (gate before the full sweep)
#
# The d16 pin is ALREADY proven (smoke: telemetry target_decode_sms hist {16:90}).
# The RISK is the d92 arm: it needs a 2-division config (pdmux_d92.yml) to pass the
# _build_r2_policy decode_states guard (multiplexing_mixin.py:139-142) while still
# pinning to 92 via PDMUX_R2_FIXED_DSM. This mini-check BOOTS d44 and d92 (M arm,
# ctx=4096, keepalive ON) and asserts telemetry target_decode_sms == 44 / 92.
# If d92 does not pin to 92 the whole d92 column is INVALID -> STOP.
#
# Pass condition: for each of {44,92}, boot_ok=1 AND target_decode_sms hist == {D}.
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate

ROOT=/scratch/ehmoon/whlee/prefill-layer-alloc
HERE=$ROOT/workspace/engine-port/results/stage0_xctrl
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/workspace/engine-port/triage/.triton_cache"
bash $ROOT/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh || {
  echo "SYNC_ENGINE_TREE_FAILED"; exit 3; }

MODEL="${MODEL:-$ROOT/hf_cache/mamba2-2.7b-sglang}"   # M arm (cheap, local, self-contained tokenizer)
CTX="${CTX:-4096}"; CTXCAP=$(( CTX + 512 ))
DTYPE="${DTYPE:-bfloat16}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

PF="$HERE/s0pin_prompt.txt"
python3 - "$(( CTX / 8 ))" "$PF" <<'PY'
import sys
open(sys.argv[2], "w").write("the robot walked slowly across the red planet " * max(1, int(sys.argv[1])))
PY

RESULT="$HERE/s0pin_${SLURM_JOB_ID:-local}_result.txt"; : > "$RESULT"
declare -A CFG=( [44]="$HERE/pdmux_d44.yml" [92]="$HERE/pdmux_d92.yml" )
FAILED=0

for D in 44 92; do
  cfg="${CFG[$D]}"
  PORT=$((37000 + RANDOM % 400))
  RUNID="s0pin_D${D}_${SLURM_JOB_ID:-local}"
  SRVLOG="$HERE/${RUNID}_srv.log"
  export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"
  export PDMUX_RUN_ID="$RUNID"; export PDMUX_WORKLOAD_ID="pin_D${D}"
  export PDMUX_R2_POLICY=fixed; export PDMUX_R2_FIXED_DSM=$D

  echo "=== PIN CHECK decode-SM=$D cfg=$(basename $cfg) port=$PORT ===" | tee -a "$RESULT"
  python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dtype "$DTYPE" \
    --attention-backend triton \
    --enable-pdmux --pdmux-config-path "$cfg" \
    --disable-piecewise-cuda-graph \
    --disable-overlap-schedule --chunked-prefill-size -1 --disable-radix-cache \
    --mem-fraction-static 0.80 --max-running-requests 48 --context-length $CTXCAP \
    --host 127.0.0.1 --port $PORT > "$SRVLOG" 2>&1 &
  SRV=$!
  ok=0
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null||true)" = 200 ] && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || { echo "SERVER_DIED D=$D (guard may have raised -- check srv.log)" | tee -a "$RESULT"; break; }; sleep 5
  done
  echo "boot_ok=$ok D=$D" | tee -a "$RESULT"

  if [ "$ok" = 1 ]; then
    RAW="$HERE/${RUNID}_raw.jsonl"
    python3 "$HERE/stage0_client.py" --url "http://127.0.0.1:$PORT" \
      --prompt-file "$PF" --conc 16 --out-tokens 16 \
      --keepalive-prefill 2 --raw-out "$RAW" --tag "pin_D${D}" 2>>"$RESULT" | tee -a "$RESULT"
    RC=$(python3 - "$PDMUX_TELEMETRY_PATH" "$D" <<'PY'
import json,sys,collections
path,expect=sys.argv[1],int(sys.argv[2])
c=collections.Counter()
for ln in open(path):
    try:e=json.loads(ln)
    except Exception:continue
    if "target_decode_sms" in e: c[e.get("target_decode_sms")]+=1
hist=dict(c)
print("ACTUAL_target_decode_sms_hist:", hist or "NONE")
ok = bool(hist) and set(hist)=={expect}
print(f"PIN_CHECK D={expect}: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
) ; PINRC=$?
    echo "$RC" | tee -a "$RESULT"
    [ "$PINRC" != 0 ] && FAILED=1
  else
    echo "BOOT_FAILED D=$D -> guard/capture failure. srv.log tail:" | tee -a "$RESULT"
    tail -30 "$SRVLOG" | grep -viE "pynvml|Future" | tee -a "$RESULT"
    FAILED=1
  fi
  kill $SRV 2>/dev/null; pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null; sleep 3
done

if [ "$FAILED" = 0 ]; then
  echo "PIN_MINICHECK_PASS: d44 and d92 both pin correctly -> full sweep is valid" | tee -a "$RESULT"
else
  echo "PIN_MINICHECK_FAIL: do NOT run the full sweep until fixed" | tee -a "$RESULT"; exit 2
fi
echo "DONE"
