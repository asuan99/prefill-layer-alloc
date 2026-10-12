#!/bin/bash
# M3 Job (PREREG_MULTIGPU rev6 §4): capacity + 1st estimand G_sym on VESSL resourcespec-a100x8.  job_entry.sh target.
#   launch.sh --campaign multigpu --target workspace/engine-port/results/multigpu/m3_job.sh \
#             --env PDMUX_M2G_PASS_SHA256=<sha256 of the committed M2G_PASS_TABLE.json>
#   (PDMUX_M2G_PASS_TABLE overrides the table path; default = results/multigpu/M2G_PASS_TABLE.json, COMMITTED before submit.)
# Order (§4-1): GPU deviation block -> N1(1) -> TP8 -> DP8 -> TP2 -> DP2 -> TP4 -> DP4 -> NUMA -> nsys -> N1 round 2.
# Hard cap 30 h (mg_const.HARD_CAP_S["m3"]).  NOTE: the budget dry-run (mg_plan.py m3-budget) is ~30 h at lambda = prior and
# 33 h at lambda = 0.7 prior for TP -- ask the user about block splitting BEFORE submitting (PREREG §6-1).
set -uo pipefail
KIND=m3
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/mg_job_common.sh"
TABLE="${PDMUX_M2G_PASS_TABLE:-$HERE/$(K M2G_PASS_TABLE)}"
COMMITTED_SHA_DRY="$HERE/$(K M2G_PASS_TABLE).sha256"
SHA="${PDMUX_M2G_PASS_SHA256:-}"
if [ "$DRYRUN" != 1 ]; then
  # F17: THREE digests must agree -- env PDMUX_M2G_PASS_SHA256, the COMMITTED results/multigpu/M2G_PASS_TABLE.json.sha256, sha256 of
  # the table file bytes -- and that committed .sha256 must be tracked and unmodified in the checked-out tree.
  COMMITTED_SHA="$HERE/$(K M2G_PASS_TABLE).sha256"
  if ! git -C "$ROOT" ls-files --error-unmatch "$COMMITTED_SHA" > /dev/null 2>&1 \
     || ! git -C "$ROOT" diff --quiet HEAD -- "$COMMITTED_SHA" 2>/dev/null; then
    fail "ABORT_M2G_NOT_PASSED committed M2G_PASS_TABLE.json.sha256 is not tracked/clean in the tree ($COMMITTED_SHA)" 15
  fi
  python3 "$HERE/mg_plan.py" check-pass-table "$TABLE" "$SHA" "$COMMITTED_SHA" | tee "$OUT/PASS_TABLE_CHECK.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "ABORT_M2G_NOT_PASSED table '$TABLE': file sha256 / env PDMUX_M2G_PASS_SHA256 ('$SHA') / committed .sha256 disagree or table missing" 15
  cp "$TABLE" "$OUT/M2G_PASS_TABLE.used.json"
fi
DRIVE=(python3 "$HERE/mg_drive.py" m3 --out "$OUT" --t0-epoch "$T0" --hard-cap-s "$HARD_CAP_S")
if [ "$DRYRUN" = 1 ]; then
  DRIVE+=(--dry-run)
  [ -f "$TABLE" ] && [ -n "$SHA" ] && DRIVE+=(--pass-table "$TABLE" --pass-sha256 "$SHA" --committed-sha-file "$COMMITTED_SHA_DRY")
else
  DRIVE+=(--pass-table "$TABLE" --pass-sha256 "$SHA" --committed-sha-file "$COMMITTED_SHA")
fi
"${DRIVE[@]}"; rc=$?
python3 "$HERE/mg_plan.py" m3-budget > "$OUT/BUDGET_DRYRUN.txt" 2>&1 || true
rm -rf "$HERE/__pycache__" "$ENG/scripts/multigpu/__pycache__" "$ENG/benchmarks/pdmux_eval/__pycache__" 2>/dev/null
exit "$rc"
