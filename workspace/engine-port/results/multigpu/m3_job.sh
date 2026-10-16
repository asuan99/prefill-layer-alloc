#!/bin/bash
# M3 Job (PREREG_MULTIGPU rev10 §4, §15-§17): capacity + 1st estimand G_sym.  ONE Job = ONE N.  job_entry.sh target.
#   PDMUX_VESSL_SPEC=<Job spec> launch.sh --campaign multigpu --target workspace/engine-port/results/multigpu/m3_job.sh \
#       --env PDMUX_MG_JOB=J4 --env PDMUX_M2G_PASS_SHA256=<sha256 of the committed M2G_PASS_TABLE_J4.json> \
#       --env PDMUX_M2G_PASS_SHA256_J2=<sha256 of the committed M2G_PASS_TABLE_J2.json>      (J4/J8 only: G0(ii) result for the TP axis)
#       --env PDMUX_HEARTBEAT_S=0                                                              (rev12 §19-5: else ABORT_LOG_KNOB)
#   Jobs: J1 resourcespec-a100x2 (GPU 0 only, N=1) · J2 x2 · J4 x4 · J8 x8.  Hard cap (rev12 U3): J1 14 h, J2/J4/J8 16 h (mg_const.HARD_CAP_S["m3"]).
#   rev12 prerequisites on the cluster volume: $PDMUX_DATA/datasets/arxsum_x/arxsum_x_tokens_v1.bin (mg_const.X_TOKENS_SHA256); committed
#   results/multigpu/PREFLIGHT_PASS.json(+.sha256) from the preflight Job on the final POST (§19-6).
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
check_committed () {   # $1 = committed .sha256 path: tracked at HEAD with exactly these bytes (harness audit r1 R-H1: in the Job results/ is
  # a symlink -- job_entry.sh -- and `git diff --quiet HEAD` reports every file behind it as deleted; mg_plan.committed_matches_head)
  python3 "$HERE/mg_plan.py" committed-clean "$ROOT" "$1" > /dev/null 2>&1
}
# ---- rev12 gates (after the shared preflight; exit 18 ABORT_LOG_KNOB · 19 ABORT_PREFLIGHT_NOT_PASSED · 20 ABORT_DATASET · 21 ABORT_CONFIG_DRIFT)
# §19-5: job_entry.sh records the log knobs in meta/log_mirror.txt and unsets them; PDMUX_HEARTBEAT_S must have been 0.
LK="$DATA/runs/$JOBID/meta/log_mirror.txt"
if [ -f "$LK" ]; then cp "$LK" "$OUT/LOG_KNOBS.txt"; else echo "log_mirror=ABSENT heartbeat_s=ABSENT" > "$OUT/LOG_KNOBS.txt"; fi
HB=$(tr ' ' '\n' < "$OUT/LOG_KNOBS.txt" | awk -F= '$1=="heartbeat_s"{print $2}')
[ "$HB" = "$(K LOG_HEARTBEAT_REQUIRED)" ] || dry_or_abort "ABORT_LOG_KNOB PDMUX_HEARTBEAT_S=$HB (registered $(K LOG_HEARTBEAT_REQUIRED): launch.sh --env PDMUX_HEARTBEAT_S=0)" 18
# §19-6 (i)-(iii): PREFLIGHT_PASS committed + digest, same image as the preflight Job, empty diff of the 7 gate paths since the POST.
python3 "$HERE/mg_plan.py" preflight-gate "$ROOT" "$HERE/$(K PREFLIGHT_PASS_FILE)" "${PDMUX_IMAGE:-}" > "$OUT/PREFLIGHT_GATE.txt" 2>&1 \
  || dry_or_abort "$(head -c 600 "$OUT/PREFLIGHT_GATE.txt")" 19
# §19-2: the registered shape-X token file on the cluster volume (sha256), before any GPU work.
XTOK="$DATA/$(K X_TOKENS_REL)"
if [ -f "$XTOK" ] && [ "$(sha256sum "$XTOK" | cut -d' ' -f1)" = "$(K X_TOKENS_SHA256)" ]; then echo "x_tokens=$XTOK sha256 OK" > "$OUT/X_TOKENS_CHECK.txt"
else echo "x_tokens=$XTOK absent or sha256 mismatch" > "$OUT/X_TOKENS_CHECK.txt"; dry_or_abort "ABORT_DATASET shape-X token file $XTOK absent or sha256 != $(K X_TOKENS_SHA256)" 20; fi
# §19-3 R-4: --mamba-ssm-dtype from the FIXED revision's config.json (key, else float32; torch_dtype never), resolved once per Job.
MCFG="$MODEL_DIR/snapshots/$(K MODEL_REVISION)/config.json"
if DT=$(python3 -c 'import sys; sys.path.insert(0, sys.argv[2]); import mg_boot as B; v, src = B.resolve_mamba_ssm_dtype(sys.argv[1]); print(v); print(src, file=sys.stderr)' \
        "$MCFG" "$ENG/scripts/multigpu" 2> "$OUT/MAMBA_SSM_DTYPE_SOURCE.txt"); then :
elif [ "$DRYRUN" = 1 ]; then DT=$(K MAMBA_SSM_DTYPE_DEFAULT); echo "DRYRUN_NOTE model config absent; mamba_ssm_dtype placeholder $DT" | tee -a "$OUT/DRYRUN_NOTES.txt"
else fail "ABORT_CONFIG_DRIFT cannot read $MCFG to resolve mamba_ssm_dtype" 21; fi
echo "mamba_ssm_dtype=$DT config=$MCFG" > "$OUT/MAMBA_SSM_DTYPE.txt"
DRIVE=(python3 "$HERE/mg_drive.py" m3 --job "$JOB" --out "$OUT" --t0-epoch "$T0" --hard-cap-s "$HARD_CAP_S"
       --mamba-ssm-dtype "$DT" --x-tokens "$XTOK" --x-tokens-sha256 "$(K X_TOKENS_SHA256)" --clk-csv "$DATA/runs/$JOBID/meta/clk.csv")
if [ "$DRYRUN" != 1 ]; then
  # F17: THREE digests must agree -- env PDMUX_M2G_PASS_SHA256, the COMMITTED .sha256 beside the harness, sha256 of the table file bytes --
  # and that committed .sha256 must be tracked at HEAD with the same bytes (check_committed; Job-layout proof, R-H1).
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
