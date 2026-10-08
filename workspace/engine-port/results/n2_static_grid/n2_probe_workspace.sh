#!/bin/bash
# N2(3-a) U-2 (a) calibration probe -- entry for a VESSL A100 80GB **Workspace** (bring-up / probe only; never a Job
# measurement, never cited: N2C-35, N2C-40). PREREG_N2_3A rev12 §14-2 "프로브 하네스 사양", H-1…H-7, R11-7.
#
# usage (inside the Workspace, on the FREEZE COMMIT checkout, clean tree):
#   bash workspace/engine-port/results/n2_static_grid/n2_probe_workspace.sh <probe_root> [--freeze <FREEZE_N2_PROBE.json>]
#        [--deadline-s <s>]
# <probe_root> receives probe_attempt_<n>/ (n chosen by the runner; never overwritten). After the run, copy the WHOLE
# probe_attempt_<n>/ back unchanged; the official post_probe_gate runs on the clean freeze checkout with that directory
# as its working directory (§14-2 (c)).
set -uo pipefail

ROOT="${PDMUX_PROJECT_ROOT:?set PDMUX_PROJECT_ROOT to the freeze-commit checkout of prefill-layer-alloc}"
HERE="$ROOT/workspace/engine-port/results/n2_static_grid"
ENG="$ROOT/workspace/engine-port"
DATA="${PDMUX_DATA:-/data}"
PROBE_ROOT="${1:?usage: n2_probe_workspace.sh <probe_root> [runner args]}"; shift
export PATH="/opt/conda/bin:$PATH"
export PDMUX_ROOT="${PDMUX_ROOT:-/opt/pdmux}" SGLANG_ENGINE_DEV="${SGLANG_ENGINE_DEV:-/opt/pdmux/sglang_engine_dev/python}"
export HF_HOME="${HF_HOME:-$DATA/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$PROBE_ROOT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$PROBE_ROOT/workspace_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
fail () { echo "$1"; exit "$2"; }

# 0. the tree must be the freeze commit with no tracked change (the frozen harness is what runs, H-2 / H-3)
HEAD=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null) || fail "ABORT_GIT not a git checkout: $ROOT" 2
DIRTY=$(git -C "$ROOT" status --porcelain --untracked-files=no | grep -c . || true)
echo "head=$HEAD tracked_dirty=$DIRTY"
[ "$DIRTY" = 0 ] || fail "ABORT_DIRTY tracked changes in the checkout -- the probe must run the committed harness" 2

# 1. engine tree: public image + the commit's engine code (sync -> manual edits -> manifest), idempotent
bash "$ENG/scripts/vessl/install_runtime.sh" "$PROBE_ROOT/runtime_source_${STAMP}.sha256" || fail "ABORT_INSTALL" 2
grep -q "def pdmux_role_is_thread_local" "$SGLANG_ENGINE_DEV/sglang/srt/distributed/parallel_state.py" \
  || fail "ABORT_ENGINE thread-local role patch missing (correctness gate 3)" 3

# 2. substrate B0 (same rule as N3 check_substrate) and model revision
B0_CSV=$(nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,mig.mode.current \
         --format=csv,noheader,nounits 2>/dev/null | head -1)
B0_SM=$(python3 -c 'import torch;print(torch.cuda.get_device_properties(0).multi_processor_count)' 2>/dev/null || echo NA)
python3 "$ENG/results/n3_capacity/n3_plan.py" --check-substrate "${B0_CSV:-NONE}" "$B0_SM" \
  || fail "ABORT_SUBSTRATE ($B0_CSV, sm=$B0_SM)" 3
REV=$(cat "$HF_HOME/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base/refs/main" 2>/dev/null || echo ABSENT)
[ "$REV" = "dc0661c829b14e5b9246c05cfa89094a0875e052" ] || fail "ABORT_MODEL_REVISION refs/main=$REV" 3
python3 -c 'import sgl_kernel.spatial' 2>/dev/null || echo "WARN sgl_kernel.spatial import failed (Workspace needed apt libnuma1 on 2026-10-04)"

# 3. the probe (one attempt; the runner refuses a new attempt if the H-7 stop rule fires)
PLAN="$HERE/PROBE_PLAN.json"
[ -f "$PLAN" ] || fail "ABORT_PLAN $PLAN missing -- emit it locally (n2_probe_run.py --emit-plan) and commit it with the freeze" 3
python3 "$HERE/n2_probe_run.py" --probe-root "$PROBE_ROOT" --plan "$PLAN" --python python "$@"
RC=$?

# 4. integrity record of every attempt directory (fetch must preserve these bytes)
for d in "$PROBE_ROOT"/probe_attempt_*; do
  [ -d "$d" ] || continue
  [ -f "$d/SHA256SUMS" ] && continue
  (cd "$d" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
done
echo "N2_PROBE_DONE rc=$RC head=$HEAD"
exit $RC
