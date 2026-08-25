#!/bin/bash
#SBATCH --job-name=a0-stage0ppp
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --chdir=/scratch/ehmoon/whlee/prefill-layer-alloc
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech/stage0ppp/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech/stage0ppp/%x_%j.err
# ============================================================================
# Stage 0''' A0 -- does nsys emit green-context graph-NODE kernels in a form
# that can be attributed and joined?
#
# * CANON: PREREG_STAGE0PPP_A0_2026-08-24.md (rev4).  Where they differ, the
#   prereg wins; where the prereg and the rule file differ, the RULE wins.
#
# * NO SERVER IS STARTED.  No model, no request.  This launches spin kernels
#   and reads nsys rows.  It produces NO latency, NO throughput, NO goodput --
#   by construction nothing here can be quoted as a performance number.
#
# * NOT EXCLUSIVE, and that is registered: CUDA API/kernel tracing does not
#   require --exclusive or --constrain=hwperf (prereg sec6, confirmed against
#   nsys --help three times).  Using exclusive would misprice this by up to 8x.
#
# * TRANSFER GAP (prereg sec0-1): this measures a SYNTHETIC spin-kernel graph.
#   A POSITIVE DOES NOT TRANSFER to the sglang decode graph -- that is A1.
#   Every artefact carries substrate="synthetic_probe_graph".
#
# * The rules-layer self-test is CPU-only and is NOT run here: it takes ~5 min
#   and would burn a quarter of the registered budget on a CPU workload.  It
#   runs on the login node before submission; this job only verifies that the
#   rule file is byte-identical to what was audited.
#
# Registered price: wall-clock ceiling 0.33 GPU-hr (20 min x 1 non-exclusive GPU).
# ============================================================================
set -uo pipefail
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
export PS1="" PATH="/apps/cuda/13.0.2/bin:$HOME/.local/bin:$PATH"
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc

D=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech/stage0ppp
J="${SLURM_JOB_ID:-local}"
export PYTHONUNBUFFERED=1
# $HOME is at its file quota; a default ~/.triton cache aborts compilation with
# "Disk quota exceeded" before a single kernel is built (observed 2026-08-14).
export TRITON_CACHE_DIR="$D/.triton_cache_a0_${J}"
mkdir -p "$TRITON_CACHE_DIR"

RULE_SHA_REGISTERED=583c4ab08a1f2194d63178ac1dab40aa9625e601bc0f72d6ee516d8219082652

echo "############# Stage 0ppp A0  job=$J #############"
nvidia-smi --query-gpu=name,driver_version,compute_mode,persistence_mode --format=csv || true
python3 -c "import torch; print('torch', torch.__version__)"
git -C /scratch/ehmoon/whlee/prefill-layer-alloc rev-parse HEAD || true
nsys --version || { echo "nsys missing -> MEASUREMENT ABSENT"; exit 30; }

echo "=== permissions (prereg sec5 / audit X10) ==="
cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "perf_event_paranoid: unreadable"
grep -i RestrictProfiling /proc/driver/nvidia/params 2>/dev/null || echo "RestrictProfilingToAdminUsers: unreadable"

echo "=== rule integrity (the audited file, byte for byte) ==="
GOT=$(sha256sum "$D/stage0ppp_a0_rule.py" | cut -d' ' -f1)
if [ "$GOT" != "$RULE_SHA_REGISTERED" ]; then
  echo "RULE FILE CHANGED ($GOT != $RULE_SHA_REGISTERED)"
  echo "This is a MEASUREMENT condition, not a result. Refusing to run."
  exit 31
fi
echo "rule sha256 OK"
# B14: rev4 pinned only the rule.  The files this audit round actually faulted
# -- probe and adapter -- were unpinned, which is false reassurance.
sha256sum "$D/stage0ppp_a0_probe.py" "$D/stage0ppp_a0_analyze.py" \
          "$D/stage0ppp_a0_rule.py"

run_leg () {                      # $1 = node|graph   $2 = granularity argument
  local G="$1" GARG="$2"
  echo "=== nsys run: granularity=$G ($GARG) ==="
  nsys profile --trace=cuda --sample=none --cpuctxsw=none --gpu-metrics-devices=none \
       --cuda-graph-trace="$GARG" -o "$D/a0_${G}_${J}" --force-overwrite true \
       python3 "$D/stage0ppp_a0_probe.py" --granularity "$G" \
               --out "$D/a0_raw_${G}_${J}.json" --tag "$J"
  local RC=$?
  echo "nsys rc=$RC"
  [ "$RC" -ne 0 ] && return $RC
  nsys export --type sqlite --force-overwrite true \
       -o "$D/a0_${G}_${J}.sqlite" "$D/a0_${G}_${J}.nsys-rep" || return 32
  # sec5/N6: the export axis is judged by nsys' own loss warnings
  # --force-export=true: `nsys export` has already written the .sqlite, and
  # `nsys stats` refuses a pre-existing export that is older than the .nsys-rep
  # ("Use --force-export=true").  Job 892554 hit exactly that: the trace was
  # perfect on every axis and the run still scored TRACE_TRUNCATED, because the
  # adapter correctly treats a failed `nsys stats` as an unknown export state
  # rather than as "no warnings" (harness audit B10).  That is gate #21 working
  # -- a harness ordering bug became a MEASUREMENT condition, not a verdict.
  nsys stats --force-export=true --report cuda_gpu_kern_sum "$D/a0_${G}_${J}.nsys-rep" \
       > "$D/a0_stats_${G}_${J}.txt" 2>&1
  echo $? > "$D/a0_statsrc_${G}_${J}.txt"    # B10: stats failure != "no warnings"
  grep -iE "dropped|lost|truncat|incomplete" "$D/a0_stats_${G}_${J}.txt" \
       > "$D/a0_warn_${G}_${J}.txt" 2>/dev/null || true
  return 0
}

echo "=== leg 0: plumbing smoke, NO nsys, <= 60 s (lesson #25) ==="
# The harness audit found the probe could not build a single kernel and could
# not capture a graph -- both would have burned the budget for a MEASUREMENT
# label.  Prove the plumbing before the profiler is ever attached.
timeout 120 python3 "$D/stage0ppp_a0_probe.py" --granularity node \
        --out "$D/a0_smoke_${J}.json" --tag "smoke-$J"
RC=$?
if [ "$RC" -ne 0 ]; then
  echo "SMOKE FAILED (rc=$RC) -> plumbing, not a result. Refusing to profile."
  exit 36
fi
echo "smoke ok"

# node granularity, EXPLICIT: driver >= 11.7 defaults to whole-graph, which
# would silently give the wrong thing for the primary estimand (sec4).
run_leg node node || { echo "node leg failed -> MEASUREMENT ABSENT"; exit 33; }
# B6 (harness audit): the `graph` run is NOT taken in this submission.  Its
# scorer (companion / E1-(b)) was never wired -- the second nsys run cost
# budget and produced nothing.  Rather than build a scorer against a schema
# that cannot be exercised here, sec4 moves to A1, where the engine substrate
# makes E1-(b) meaningful anyway (sec0-1: a synthetic-graph positive does not
# transfer).

echo "=== scoring against the registered rule ==="
python3 "$D/stage0ppp_a0_analyze.py" \
  --raw-node    "$D/a0_raw_node_${J}.json"   --sqlite-node  "$D/a0_node_${J}.sqlite" \
  --stats-warn-node  "$(cat "$D/a0_warn_node_${J}.txt" 2>/dev/null)" \
  --stats-rc-node    "$(cat "$D/a0_statsrc_node_${J}.txt" 2>/dev/null || echo 0)" \
  --out "$D/a0_verdict_${J}.json" --tag "$J"
RC=$?
echo "analyze rc=$RC"
[ "$RC" -ne 0 ] && { echo "SCORING FAILED -> no verdict exists for this job."; exit 35; }
[ ! -s "$D/a0_verdict_${J}.json" ] && { echo "SCORING PRODUCED NO .json"; exit 35; }

echo "############# done job=$J #############"
echo "* CITE a0_verdict_${J}.json ONLY (gate #56; the .out is a console transcript)."
echo "* Every outcome here is a TOOL-VALIDITY statement on a SYNTHETIC graph."
echo "  PRIMARY_ESTIMAND_UNCONSTRUCTIBLE does NOT close the kernel_mech track"
echo "  (four survival paths remain).  A positive does NOT transfer to the engine."
