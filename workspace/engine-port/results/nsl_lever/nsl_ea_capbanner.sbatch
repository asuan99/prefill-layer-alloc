#!/bin/bash
#SBATCH --job-name=nsl-ea-capbanner
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/nsl_lever/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/nsl_lever/%x_%j.err
# ============================================================================
# E-A -- cap <-> mamba pool <-> KV budget, read from the ENGINE'S OWN BANNER.
#
# * CANON: PREREG_NSL_EA_CAPBANNER_2026-08-25.md.  Where they differ, it wins.
#
# * NO REQUESTS ARE SENT.  Each cell boots the server, waits for /health, reads
#   three banner values, and kills it.  There is no client, no workload, no
#   trace.  By construction this job CANNOT produce a latency, a throughput or
#   a goodput number -- there is nothing in its output that could be quoted as
#   a performance result.
#
# * It does NOT answer whether the cap binds (that predicate is invalid --
#   capbind audit B1) and it does NOT close B3.  It converts a code-derived
#   identity into a measurement, which is what step (3) needs.
#
# * The realized cap is read from the ENGINE banner, never from the CLI flag
#   (capbind audit B7; standing lesson: a pin is verified by the realized
#   value, not the target).
#
# Registered price: wall-clock ceiling 0.25 GPU-hr (15 min x 1 non-exclusive GPU).
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc

D=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/nsl_lever
S=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/slo_sched
J="${SLURM_JOB_ID:-local}"
MODEL=Zyphra/Zamba2-2.7B
CTX=4096
CFG="$S/pdmux_d44.yml"
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$D/.triton_cache_ea_${J}"
mkdir -p "$TRITON_CACHE_DIR"

echo "############# E-A cap banner  job=$J #############"
nvidia-smi --query-gpu=name,driver_version --format=csv || true
python3 -c "import torch; print('torch', torch.__version__)"
git -C /scratch/ehmoon/whlee/prefill-layer-alloc rev-parse HEAD || true
[ -f "$CFG" ] || { echo "pdmux config missing -> MEASUREMENT ABSENT"; exit 40; }

boot_cell () {                       # $1 = tag   $2 = cap   $3 = extra args
  local TAG="$1" CAP="$2" EXTRA="$3"
  local LOG="$D/ea_${TAG}_${J}_srv.log"
  local PORT
  PORT=$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
  echo "=== cell $TAG : --max-running-requests $CAP $EXTRA (port $PORT) ==="
  local T0=$SECONDS
  python -m sglang.launch_server --model-path $MODEL --trust-remote-code --dtype bfloat16 \
    --attention-backend triton --disable-radix-cache \
    --mem-fraction-static 0.82 --max-running-requests "$CAP" --context-length $CTX \
    --enable-pdmux --pdmux-config-path "$CFG" --chunked-prefill-size -1 \
    --disable-overlap-schedule $EXTRA \
    --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
  local PID=$!
  local OK=0
  for i in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null || true)" = 200 ] \
      && { OK=1; break; }
    kill -0 $PID 2>/dev/null || { echo "  SERVER DIED (cell $TAG) -- see $LOG"; break; }
    sleep 3
  done
  echo "  boot_s=$((SECONDS-T0)) health_ok=$OK"
  # NO REQUEST IS SENT.  Read the banner and stop.
  kill $PID 2>/dev/null
  pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
  sleep 8
  grep -m1 -o "max_total_num_tokens=[0-9]*" "$LOG" || echo "  (no max_total_num_tokens)"
  grep -m1 -o "max_running_requests=[0-9]*" "$LOG" || echo "  (no max_running_requests)"
  grep -m1 -o "max_mamba_cache_size: *[0-9]*" "$LOG" || echo "  (no max_mamba_cache_size)"
}

boot_cell c24    24 ""
boot_cell c48    48 ""
boot_cell c96    96 ""
boot_cell c48m96 48 "--max-mamba-cache-size 96"

echo "=== scoring against the registered rule ==="
python3 "$D/nsl_ea_analyze.py" --dir "$D" --tag "$J" --out "$D/nsl_ea_banner_${J}.json"
RC=$?
echo "analyze rc=$RC"
[ "$RC" -ne 0 ] && { echo "SCORING FAILED -> no verdict exists for this job."; exit 41; }

echo "############# done job=$J #############"
echo "* CITE nsl_ea_banner_${J}.json ONLY (gate #56)."
echo "* NO REQUESTS WERE SENT.  This job cannot produce a performance number."
echo "* It does NOT answer whether the cap binds, and it does NOT close B3."
