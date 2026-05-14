#!/bin/bash
# =============================================================================
# run_pipeline.sh — Full stage1 → stage2 → stage3 SLURM job dependency chain
#
# USAGE
#   bash slurm/run_pipeline.sh [MODEL] [DEVICE]
#   bash slurm/run_pipeline.sh all     [DEVICE]   # both models
#
#   MODEL  : zamba2 | falcon_h1 | all   (default: zamba2)
#   DEVICE : hardware key from configs/hardware.yaml, or 'auto'  (default: auto)
#
# EXAMPLES
#   bash slurm/run_pipeline.sh                      # zamba2, auto
#   bash slurm/run_pipeline.sh falcon_h1            # falcon_h1, auto
#   bash slurm/run_pipeline.sh all                  # both models
#   bash slurm/run_pipeline.sh zamba2 a100_80gb     # explicit hardware key
#
# DEPENDENCY GRAPH (single model)
#   stage1 ──► stage2 ──► stage3
#
# DEPENDENCY GRAPH (all models)
#   stage1[zamba2]  ──────────────────► stage2[zamba2] ──► stage3[zamba2]
#   stage1[falcon_h1] ──► (after z_s2) ► stage2[falcon_h1] ──► stage3[falcon_h1]
#
#   stage1 runs in parallel across models.
#   stage2[falcon_h1] waits for BOTH stage1[falcon_h1] AND stage2[zamba2] to
#   finish — this prevents concurrent compute_decision_matrix.py writes.
#
# MONITORING
#   squeue -u $USER
#   tail -f logs/stage1_<JID>.log
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."         # repo root regardless of invocation path

MODEL=${1:-zamba2}
DEVICE=${2:-auto}

# ---------------------------------------------------------------------------
# Helper: sbatch wrapper that returns job ID (--parsable)
# Usage: sbatch_job [sbatch_options...] -- script [script_args...]
# ---------------------------------------------------------------------------
sbatch_job() {
    local opts=() script="" args=()
    local past_sep=0
    for arg in "$@"; do
        if [[ $past_sep -eq 0 && "$arg" == "--" ]]; then
            past_sep=1
        elif [[ $past_sep -eq 0 ]]; then
            opts+=("$arg")
        elif [[ -z "$script" ]]; then
            script="$arg"
        else
            args+=("$arg")
        fi
    done
    sbatch --parsable "${opts[@]}" "$script" "${args[@]}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
mkdir -p logs

echo "============================================================"
echo "  Prefill-Layer-Alloc  |  Full Pipeline Submission"
echo "  model  = $MODEL"
echo "  device = $DEVICE"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

if [ "$MODEL" = "all" ]; then
    # ── Both models ──────────────────────────────────────────────────────────
    # stage1 for both runs in parallel.
    # stage2[falcon_h1] is serialized behind stage2[zamba2] so that
    # compute_decision_matrix.py is never running concurrently.

    echo ""
    echo "Mode: both models (stage1 parallel, stage2/3 serialized per model)"

    # --- zamba2 chain (no extra deps) ---
    echo ""
    echo "[zamba2 chain]"

    jid_z1=$(sbatch_job --job-name "s1-zamba2" \
        -- slurm/run_stage1.sh "zamba2" "$DEVICE")
    echo "  [stage1] job=$jid_z1"

    jid_z2=$(sbatch_job --job-name "s2-zamba2" \
        --dependency="afterok:${jid_z1}" \
        -- slurm/run_stage2.sh "zamba2" "$DEVICE")
    echo "  [stage2] job=$jid_z2  dep=afterok:${jid_z1}"

    jid_z3=$(sbatch_job --job-name "s3-zamba2" \
        --dependency="afterok:${jid_z2}" \
        -- slurm/run_stage3.sh "zamba2" "$DEVICE")
    echo "  [stage3] job=$jid_z3  dep=afterok:${jid_z2}"
    echo "  chain  : $jid_z1 → $jid_z2 → $jid_z3"

    # --- falcon_h1 chain (stage1 parallel with zamba2; stage2 after zamba2's stage2) ---
    echo ""
    echo "[falcon_h1 chain]"

    jid_f1=$(sbatch_job --job-name "s1-falcon_h1" \
        -- slurm/run_stage1.sh "falcon_h1" "$DEVICE")
    echo "  [stage1] job=$jid_f1  (parallel with zamba2 stage1)"

    # wait for BOTH falcon stage1 AND zamba2 stage2 — avoids decision_matrix conflict
    jid_f2=$(sbatch_job --job-name "s2-falcon_h1" \
        --dependency="afterok:${jid_f1}:${jid_z2}" \
        -- slurm/run_stage2.sh "falcon_h1" "$DEVICE")
    echo "  [stage2] job=$jid_f2  dep=afterok:${jid_f1}:${jid_z2}"

    jid_f3=$(sbatch_job --job-name "s3-falcon_h1" \
        --dependency="afterok:${jid_f2}" \
        -- slurm/run_stage3.sh "falcon_h1" "$DEVICE")
    echo "  [stage3] job=$jid_f3  dep=afterok:${jid_f2}"
    echo "  chain  : $jid_f1 → $jid_f2 → $jid_f3"

    echo ""
    echo "============================================================"
    echo "  Submitted 6 jobs"
    echo ""
    printf "  %-12s  %-10s  %s\n" "stage"       "model"      "job_id"
    printf "  %-12s  %-10s  %s\n" "stage1"      "zamba2"     "$jid_z1"
    printf "  %-12s  %-10s  %s\n" "stage1"      "falcon_h1"  "$jid_f1"
    printf "  %-12s  %-10s  %s\n" "stage2"      "zamba2"     "$jid_z2"
    printf "  %-12s  %-10s  %s\n" "stage2"      "falcon_h1"  "$jid_f2"
    printf "  %-12s  %-10s  %s\n" "stage3"      "zamba2"     "$jid_z3"
    printf "  %-12s  %-10s  %s\n" "stage3"      "falcon_h1"  "$jid_f3"
    echo "============================================================"

else
    # ── Single model ─────────────────────────────────────────────────────────
    echo ""
    echo "Mode: single model ($MODEL)"

    jid1=$(sbatch_job --job-name "s1-${MODEL}" \
        -- slurm/run_stage1.sh "$MODEL" "$DEVICE")
    echo "  [stage1] job=$jid1"

    jid2=$(sbatch_job --job-name "s2-${MODEL}" \
        --dependency="afterok:${jid1}" \
        -- slurm/run_stage2.sh "$MODEL" "$DEVICE")
    echo "  [stage2] job=$jid2  dep=afterok:${jid1}"

    jid3=$(sbatch_job --job-name "s3-${MODEL}" \
        --dependency="afterok:${jid2}" \
        -- slurm/run_stage3.sh "$MODEL" "$DEVICE")
    echo "  [stage3] job=$jid3  dep=afterok:${jid2}"

    echo ""
    echo "============================================================"
    echo "  Submitted 3 jobs  ($MODEL)"
    echo ""
    printf "  %-12s  %-10s  %s\n" "stage"   "model"    "job_id"
    printf "  %-12s  %-10s  %s\n" "stage1"  "$MODEL"   "$jid1"
    printf "  %-12s  %-10s  %s\n" "stage2"  "$MODEL"   "$jid2"
    printf "  %-12s  %-10s  %s\n" "stage3"  "$MODEL"   "$jid3"
    echo "  chain : $jid1 → $jid2 → $jid3"
    echo "============================================================"
fi

echo ""
echo "  Monitor : squeue -u $USER"
echo "  Logs    : ls logs/"
echo "  Cancel  : scancel \$(squeue -u $USER -h -o '%i' | tr '\n' ' ')"
