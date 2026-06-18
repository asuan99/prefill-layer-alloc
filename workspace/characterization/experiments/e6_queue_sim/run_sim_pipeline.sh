#!/bin/bash
# =============================================================================
# run_sim_pipeline.sh — user-runnable queue-sim measurement + analysis pipeline.
#
# The simulator needs a SINGLE-CHUNK LUT (1 prefill chunk ∥ 1 decode step), i.e.
# E5 full mode with --prefill-tokens = chunk(256) and --decode-protect (so the
# green_ctx_protect rows exist). This script (1) submits that LUT job, then
# (2) runs the λ×policy queue sim on the produced LUT.
#
# Usage (always launch under `env -u BASH_ENV` — cluster lmod breaks bash):
#   env -u BASH_ENV bash run_sim_pipeline.sh submit     # submit the single-chunk LUT job
#   env -u BASH_ENV bash run_sim_pipeline.sh analyze     # run the sim (after the job is done)
#   env -u BASH_ENV bash run_sim_pipeline.sh auto        # submit, poll until done, then analyze
#
# Tunables (env):
#   MODELS="zamba2_2.7b zamba2_7b"   # ≤4 (QOS limit); LUT generated per model
#   PF=attn  DEC=ssm                 # kernel-pair to study (2.7b exception = attn×ssm)
#   SLO=1.0                          # ITL SLO target (ms)
#   LAMBDAS="0.02 0.05 0.1 0.2 0.5 1.0 2.0"   # arrival-rate sweep (req/ms)
# =============================================================================
set -euo pipefail

REPO="/scratch/$USER/whlee/prefill-layer-alloc"
CHAR="$REPO/workspace/characterization"
PY="$REPO/bin/python"
SUBMIT="$CHAR/experiments/slurm/submit_size_sweep.sh"
LUTDIR="$CHAR/results_v2/e5_sim"          # single-chunk LUT lives here (separate from e5/e5_dp)
DEVICE="a100_sxm4_80gb"

MODELS="${MODELS:-zamba2_2.7b zamba2_7b}"
PF="${PF:-attn}"; DEC="${DEC:-ssm}"; SLO="${SLO:-1.0}"
LAMBDAS="${LAMBDAS:-0.02 0.05 0.1 0.2 0.5 1.0 2.0}"
CMD="${1:-auto}"

submit_lut() {
  mkdir -p "$LUTDIR"
  echo "[sim-pipeline] submitting single-chunk LUT (tokens=256, decode-protect) for: $MODELS"
  MODELS="$MODELS" env -u BASH_ENV bash "$SUBMIT" e5 -- \
      --prefill-mode full --prefill-tokens 256 --decode-protect --output-dir "$LUTDIR"
}

analyze() {
  local n=0
  for m in $MODELS; do
    local lut="$LUTDIR/serving_coexec_full_${m}_${DEVICE}.csv"
    if [ ! -f "$lut" ]; then
      echo "[sim-pipeline] LUT missing: $lut  (run 'submit' first / wait for the job)"; continue
    fi
    echo "[sim-pipeline] === queue sim: $m  pf=$PF×dec=$DEC  SLO≤${SLO}ms ==="
    env -u BASH_ENV "$PY" -m experiments.e6_queue_sim.run_queue_sim \
        --lut "$lut" --pf-layer "$PF" --dec-layer "$DEC" --slo-ms "$SLO" --lambdas $LAMBDAS
    n=$((n + 1))
  done
  [ "$n" -gt 0 ] || { echo "[sim-pipeline] no LUTs analyzed."; return 1; }
}

poll_done() {
  local jid="$1"
  echo "[sim-pipeline] waiting for job $jid (poll 120s)..."
  while squeue -j "$jid" -h 2>/dev/null | grep -q .; do sleep 120; done
  echo "[sim-pipeline] job $jid left the queue."
  sacct -j "$jid" --format=JobID,State,Elapsed,End -n 2>/dev/null | head
}

cd "$CHAR"
case "$CMD" in
  submit)  submit_lut ;;
  analyze) analyze ;;
  auto)
    OUT="$(submit_lut)"; echo "$OUT"
    JID="$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+' | head -1)"
    [ -n "$JID" ] || { echo "[sim-pipeline] submit failed (QOS limit? reduce MODELS)"; exit 2; }
    poll_done "$JID"
    analyze ;;
  *) echo "usage: env -u BASH_ENV bash $0 <submit|analyze|auto>"; exit 2 ;;
esac
