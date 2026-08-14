#!/bin/bash
# =============================================================================
# run_pipeline.sh — run the WHOLE v2 size sweep under a tight SLURM job limit.
#
#   # survive disconnect — pick ONE:
#   DETACH=1 MAXQ=2 env -u BASH_ENV bash experiments/slurm/run_pipeline.sh   # self-detaches, logs to file
#   tmux new -s v2 ; MAXQ=2 env -u BASH_ENV bash experiments/slurm/run_pipeline.sh
#
# This is a LONG-running controller (the A100 queue serializes jobs over hours/
# days under QOS aanv8: MaxJobs=2, MaxSubmit=4). It MUST survive logout — running
# it bare in the login shell will be SIGHUP-killed when the session drops.
#
# Why a throttling babysitter (not --dependency):
#   This QOS caps how many jobs a user may have queued (QOSMaxSubmitJobPerUser /
#   MaxJobs). A dependency DAG puts every job PENDING at once → exceeds the cap.
#   Instead this script trickles one (experiment × model) job at a time, keeping
#   at most MAXQ of OUR jobs in the queue, waiting for slots to free. It also
#   retries a submission that is refused for a limit reason, so it works even if
#   MAXQ is set above the real cap.
#
#   BASH_ENV note: this cluster's lmod BASH_ENV breaks non-interactive bash, so
#   every job runs via `--wrap "env -u BASH_ENV bash -c '…'"` (--wrap runs under
#   /bin/sh, which is immune) and you must launch THIS script with `env -u …`.
#
# Flow (one command re-runs the WHOLE study):
#   E0  (inline, cpu, instant)
#   phase 1: e1,e2,e3,e5 × {4 models}         throttled to MAXQ (gate-independent)
#   gates (inline, cpu) → verdicts/{g0,g1}.json
#   phase 2: e4 × {4 models}                  only if G1 authorises (or FORCE_E4)
#   viz   (inline, cpu) → results_v2/figures/
#
# Knobs: MAXQ (default 2)  POLL secs (default 30)  A100_PART  MODELS  FORCE_E4=1
# =============================================================================

set -uo pipefail   # NOT -e: the polling loops handle their own errors

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
CHAR_DIR="$REPO_ROOT/workspace/characterization"
A100_PART="${A100_PART:-amd_a100nv_8}"
LOG="$REPO_ROOT/logs"; mkdir -p "$LOG"
MAXQ="${MAXQ:-2}"
POLL="${POLL:-30}"
MODELS="${MODELS:-zamba2_1.2b zamba2_2.7b falcon_h1_1.5b falcon_h1_3b}"
SLURM_USER_SELECTOR="${SLURM_AGENT_SLURM_USER:-${SLURM_USER_ID:-$(id -u)}}"
ACT="source $REPO_ROOT/bin/activate 2>/dev/null || true; cd $CHAR_DIR"

# --- optional self-detach: survive SSH disconnect WITHOUT tmux (DETACH=1) -----
# Re-exec under a new session (setsid) detached from the terminal, logging to a
# file, then return the shell. Guards against re-exec loops via _V2_DETACHED.
# Keeps BASH_ENV unset in the child (this cluster's lmod BASH_ENV breaks bash).
if [ "${DETACH:-0}" = "1" ] && [ -z "${_V2_DETACHED:-}" ]; then
  mkdir -p "$LOG"
  RUNLOG="$LOG/run_pipeline_$(date +%Y%m%d_%H%M%S).log"
  echo "[detach] launching detached pipeline; it will survive logout."
  _V2_DETACHED=1 MAXQ="$MAXQ" POLL="$POLL" MODELS="$MODELS" A100_PART="$A100_PART" \
    FORCE_E4="${FORCE_E4:-0}" SERIAL="${SERIAL:-0}" \
    setsid env -u BASH_ENV bash "$0" >"$RUNLOG" 2>&1 </dev/null &
  echo "[detach] PID $!   log: $RUNLOG"
  echo "[detach] monitor:  tail -f $RUNLOG       stop:  kill $!  (or scancel your v2- jobs)"
  exit 0
fi

script_of() { case "$1" in
  e1) echo experiments/e1_prefill_decomp/run_component_sweep.py ;;
  e2) echo experiments/e2_sm_saturation/run_batch_swept_sweep.py ;;
  e3) echo experiments/e3_decode_floor/run_decode_floor.py ;;
  e4) echo experiments/e4_concurrent/run_concurrent_ab.py ;;
  e5) echo experiments/e5_serving/run_serving_coexec.py ;;
esac; }
time_of() { case "$1" in e2|e5) echo 08:00:00 ;; *) echo 04:00:00 ;; esac; }

# count OUR jobs currently in the queue (prefix v2-; -r expands any arrays)
inflight() {
  local n
  n=$(squeue -u "$SLURM_USER_SELECTOR" -h -r -o '%80j' 2>/dev/null | grep -c '^v2-' || true)
  echo "${n:-0}"
}
wait_slot() { while [ "$(inflight)" -ge "$MAXQ" ]; do sleep "$POLL"; done; }
wait_all()  { while [ "$(inflight)" -gt 0 ];  do sleep "$POLL"; done; }

# submit one (exp, model) job; block until SLURM accepts it (retry on limit)
submit_job() {
  local exp="$1" model="$2"
  local name="v2-$exp-$model" script t inner out
  script="$(script_of "$exp")"; t="$(time_of "$exp")"
  inner="$ACT; echo \"$exp / $model\"; python $script --models $model"
  while :; do
    wait_slot
    out=$(sbatch --parsable --job-name="$name" --partition="$A100_PART" --gres=gpu:1 \
            --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --time="$t" \
            --comment="field=efficientai;appl=pytorch" \
            --output="$LOG/${name}_%j.log" --error="$LOG/${name}_%j.err" \
            --wrap "env -u BASH_ENV bash -c '$inner'" 2>"$LOG/${name}_agent.err")
    rc=$?
    if [ "$rc" -eq 0 ]; then echo "  submitted $name (job $out)"; return 0; fi
    err=$(<"$LOG/${name}_agent.err")
    if echo "$err" | grep -qiE "limit|qos|assocmax|policy"; then
      echo "  [throttle] $name held (queue full) — retry in ${POLL}s"; sleep "$POLL"
    else
      echo "  ERROR submitting $name: $err"; return 1
    fi
  done
}

echo "=== v2 pipeline (throttled, MAXQ=$MAXQ, poll=${POLL}s) ==="
echo "    models: $MODELS"

# --- E0 inline ---------------------------------------------------------------
echo "[E0] inline ..."
env -u BASH_ENV bash -c "$ACT; python experiments/e0_analytical/run_wave_table.py" \
  >/dev/null 2>&1 && echo "  E0 done" || echo "  E0 failed (non-fatal)"

# --- phase 1: e1,e2,e3,e5 per model (gate-independent measurements) ----------
echo "[phase 1] submitting e1/e2/e3/e5 for each model (throttled) ..."
for m in $MODELS; do for e in e1 e2 e3 e5; do submit_job "$e" "$m"; done; done
echo "[phase 1] all submitted; waiting for completion ..."
wait_all
echo "[phase 1] done."

# --- gates inline ------------------------------------------------------------
echo "[gates] adjudicating G0/G1 ..."
env -u BASH_ENV bash -c "$ACT; python experiments/gates/adjudicate.py" || true

# --- phase 2: e4 only if G1 = BW_MECHANISM ----------------------------------
G1="$REPO_ROOT/workspace/characterization/results_v2/verdicts/g1_verdict.json"
if grep -qE '"verdict": "(ASYMMETRY_PRESENT|BW_MECHANISM)"' "$G1" 2>/dev/null || [ "${FORCE_E4:-0}" = "1" ]; then
  echo "[phase 2] G1=BW_MECHANISM (or FORCE_E4) → submitting e4 per model ..."
  for m in $MODELS; do submit_job e4 "$m"; done
  wait_all
  echo "[phase 2] done."
else
  echo "[phase 2] skipped — G1 is not BW_MECHANISM (see $G1). Use FORCE_E4=1 to override."
fi

# --- viz inline: regenerate all figures from the fresh results ---------------
echo "[viz] generating figures ..."
env -u BASH_ENV bash -c "$ACT; python experiments/viz/plot_results.py" \
  >/dev/null 2>&1 && echo "  figures -> results_v2/figures/" || echo "  viz failed (non-fatal)"

echo "=== pipeline complete. results in results_v2/ ; verdicts in results_v2/verdicts/ ; figures in results_v2/figures/ ==="
