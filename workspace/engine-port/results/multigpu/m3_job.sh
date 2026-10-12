#!/bin/bash
# M3 Job (PREREG_MULTIGPU rev10 §4, §15-§17): capacity + 1st estimand G_sym.  ONE Job = ONE N.  job_entry.sh target.
#   PDMUX_VESSL_SPEC=<Job spec> launch.sh --campaign multigpu --target workspace/engine-port/results/multigpu/m3_job.sh \
#       --env PDMUX_MG_JOB=J4 --env PDMUX_M2G_PASS_SHA256=<sha256 of the committed M2G_PASS_TABLE_J4.json> \
#       --env PDMUX_M2G_PASS_SHA256_J2=<sha256 of the committed M2G_PASS_TABLE_J2.json>      (J4/J8 only: G0(ii) result for the TP axis)
#   Jobs: J1 resourcespec-a100x2 (GPU 0 only, N=1) · J2 x2 · J4 x4 · J8 x8.  Hard cap: J1 10 h, J2/J4/J8 14 h (mg_const.HARD_CAP_S["m3"]).
#   (PDMUX_M2G_PASS_TABLE / PDMUX_M2G_PASS_TABLE_J2 override the table paths; default results/multigpu/M2G_PASS_TABLE_<J>.json, COMMITTED before submit.)
# Order (§16-3): [J8: GPU-deviation block, 8 servers concurrent] -> N=1 anchor boots (fused_sym, agnostic; J2/J4/J8) -> the Job's TP/DP configs in
#   the REGISTERED order (derivation in mg_const.config_order_types; J2=J4=J8=[DP,TP]).  No N=1 round 2, no NUMA block, no nsys block.
# Tables (§15-2): the Job's OWN table (3-way digest check; failure -> ABORT_M2G_NOT_PASSED) and, for J4/J8, J2's table for G0(ii).  A missing or
#   invalid J2 table makes the Job's TP configs NOT_TESTED; DP proceeds (no ABORT).  J1 and J2 need no extra table.
# The cross-Job labels (JOB_EFFECT, trends, P-DP) are produced AFTER fetch: mg_label.py --jobs J1=<dir> J2=<dir> J4=<dir> J8=<dir>.
set -uo pipefail
KIND=m3
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/mg_job_common.sh"
TNAME="M2G_PASS_TABLE_${JOB}.json"
TABLE="${PDMUX_M2G_PASS_TABLE:-$HERE/$TNAME}"
COMMITTED_SHA="$HERE/$TNAME.sha256"
SHA="${PDMUX_M2G_PASS_SHA256:-}"
check_committed () {   # $1 = committed .sha256 path
  git -C "$ROOT" ls-files --error-unmatch "$1" > /dev/null 2>&1 && git -C "$ROOT" diff --quiet HEAD -- "$1" 2>/dev/null
}
DRIVE=(python3 "$HERE/mg_drive.py" m3 --job "$JOB" --out "$OUT" --t0-epoch "$T0" --hard-cap-s "$HARD_CAP_S")
if [ "$DRYRUN" != 1 ]; then
  # F17: THREE digests must agree -- env PDMUX_M2G_PASS_SHA256, the COMMITTED .sha256 beside the harness, sha256 of the table file bytes --
  # and that committed .sha256 must be tracked and unmodified in the checked-out tree.
  check_committed "$COMMITTED_SHA" || fail "ABORT_M2G_NOT_PASSED committed $TNAME.sha256 is not tracked/clean in the tree ($COMMITTED_SHA)" 15
  python3 "$HERE/mg_plan.py" check-pass-table "$TABLE" "$SHA" "$COMMITTED_SHA" | tee "$OUT/PASS_TABLE_CHECK.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail "ABORT_M2G_NOT_PASSED table '$TABLE': file sha256 / env PDMUX_M2G_PASS_SHA256 ('$SHA') / committed .sha256 disagree or table missing" 15
  # R5: belt and braces beside mg_plan.load_pass_table -- the G5(a) CPU unit test must be PASS in the table
  python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("g5a_unit_test")=="PASS" else 1)' "$TABLE" \
    || fail "ABORT_M2G_NOT_PASSED table '$TABLE': g5a_unit_test != PASS" 15
  cp "$TABLE" "$OUT/M2G_PASS_TABLE.used.json"
  DRIVE+=(--pass-table "$TABLE" --pass-sha256 "$SHA" --committed-sha-file "$COMMITTED_SHA")
  if [ "$JOB" = J4 ] || [ "$JOB" = J8 ]; then
    J2TABLE="${PDMUX_M2G_PASS_TABLE_J2:-$HERE/M2G_PASS_TABLE_J2.json}"
    J2SHA="${PDMUX_M2G_PASS_SHA256_J2:-}"
    J2COMMITTED="$HERE/M2G_PASS_TABLE_J2.json.sha256"
    if check_committed "$J2COMMITTED" && python3 "$HERE/mg_plan.py" check-pass-table "$J2TABLE" "$J2SHA" "$J2COMMITTED" > "$OUT/J2_TABLE_CHECK.txt" 2>&1; then
      DRIVE+=(--j2-pass-table "$J2TABLE" --j2-pass-sha256 "$J2SHA" --j2-committed-sha-file "$J2COMMITTED")
    else
      echo "J2 table unusable (untracked/dirty .sha256, missing, or digest mismatch) -> TP configs NOT_TESTED, DP proceeds" | tee -a "$OUT/J2_TABLE_CHECK.txt"
    fi
  fi
else
  DRIVE+=(--dry-run)
  [ -f "$TABLE" ] && [ -n "$SHA" ] && DRIVE+=(--pass-table "$TABLE" --pass-sha256 "$SHA" --committed-sha-file "$COMMITTED_SHA")
fi
"${DRIVE[@]}"; rc=$?
python3 "$HERE/mg_plan.py" m3-budget --job "$JOB" > "$OUT/BUDGET_DRYRUN.txt" 2>&1 || true
rm -rf "$HERE/__pycache__" "$ENG/scripts/multigpu/__pycache__" "$ENG/benchmarks/pdmux_eval/__pycache__" 2>/dev/null
exit "$rc"
