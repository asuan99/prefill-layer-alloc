#!/bin/bash
# M2g Job (PREREG_MULTIGPU rev6 §3): correctness gate on VESSL resourcespec-a100x8.  job_entry.sh target.
#   launch.sh --campaign multigpu --target workspace/engine-port/results/multigpu/m2g_job.sh   (dry-run first; --submit only after approval)
#   env: PDMUX_VESSL_SPEC=resourcespec-a100x8 PDMUX_VESSL_ENV=workspace/engine-port/scripts/vessl/vessl.multigpu.env
# Output: results/multigpu/m2g_<job>/{M2G_PASS_TABLE.json(+.sha256), gates/, boots/, cells/, DRIVER_*}
# Hard cap 8 h (mg_const.HARD_CAP_S["m2g"]) enforced by mg_drive.py (deadline passed to every boot).
#
# Order (F1): G5(a) CPU test -> G3 (run-g3 per TP, BEFORE any serving boot) -> serving boots (incl. the G5(a) GPU boot) -> G1 exposure +
# score-g1/g2 -> G5(a) GPU descriptive -> M2G_PASS_TABLE.  Job path = mg_drive.run_m2g + assemble_pass_table (M2gRun is the reference only).
# ENGINE-PORTER INTERFACE (scripts/multigpu/m2g_correctness_gate.py; verified present 2026-10-12, CPU-tested by engine-porter):
#   serve-g1 --endpoint URL --out DIR --config CFG --arm ARM    run on the LIVE server of each boot (mg_plan.m2g_boots `cmd` steps)
#   serve-g2 --endpoint URL --out DIR --config CFG --repeats K
#   score-g1/score-g2 --out DIR --config CFG ; score-g3 --out DIR --tp N     (mg_drive.run_m2g calls them after the boots)
#   build_pass_table / score_g0_* / score_g3b / score_g4 / score_g5b / score_g5c_memory / check_contamination are IMPORTED by
#   mg_drive.assemble_pass_table: the §3-1 mapping exists in exactly one place.
# A missing artifact leaves its gate NOT_TESTED, which §3-1 treats like FAIL for the range it covers.
set -uo pipefail
KIND=m2g
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/mg_job_common.sh"
DRIVE=(python3 "$HERE/mg_drive.py" m2g --out "$OUT" --t0-epoch "$T0" --hard-cap-s "$HARD_CAP_S")
[ "$DRYRUN" = 1 ] && DRIVE+=(--dry-run)
"${DRIVE[@]}"; rc=$?
if [ "$rc" -eq 0 ] && [ -f "$OUT/$(K M2G_PASS_TABLE)" ]; then
  echo "M2G_PASS_TABLE sha256: $(cat "$OUT/$(K M2G_PASS_TABLE).sha256")  (register it as PDMUX_M2G_PASS_SHA256 for the M3 submit)"
fi
rm -rf "$HERE/__pycache__" "$ENG/scripts/multigpu/__pycache__" "$ENG/benchmarks/pdmux_eval/__pycache__" 2>/dev/null
exit "$rc"
