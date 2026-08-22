| # | file:line | kind | arm-pinned | text |
|---|---|---|---|---|
| 1 | `g16_grid.sbatch:168` | ARM_COUNT | - | `ARM_ORDER="${G18_ARMS:-$(python3 "$HERE/g16_arm_order.py" 1)}"` |
| 2 | `g16_grid.sbatch:169` | ARM_COUNT | - | `N_EXPECT_ARMS=$(echo "$ARM_ORDER" | wc -w)` |
| 3 | `g16_grid.sbatch:171` | REF | - | `MIN_SNAPSHOTS=20` |
| 4 | `g16_grid.sbatch:192` | ARM_COUNT | - | `ARM_ORDER="d44 d64 d74"` |
| 5 | `g16_grid.sbatch:193` | ARM_COUNT | - | `N_EXPECT_ARMS=3` |
| 6 | `g16_grid.sbatch:196` | REF | - | `MIN_SNAPSHOTS=5` |
| 7 | `g16_grid.sbatch:206` | ARM_COUNT | - | `ARM_ORDER="$(python3 "$HERE/g16_arm_order.py" "$BLOCK_IDX")" \` |
| 8 | `g16_grid.sbatch:208` | ARM_COUNT | - | `N_EXPECT_ARMS=7` |
| 9 | `g16_grid.sbatch:211` | REF | - | `MIN_SNAPSHOTS=20` |
| 10 | `g16_grid.sbatch:220` | HARD_EXIT | - | `exit 1` |
| 11 | `g16_grid.sbatch:225` | ARM_COUNT | - | `if [ "$N_GOT_ARMS" != "$N_EXPECT_ARMS" ]; then` |
| 12 | `g16_grid.sbatch:226` | ARM_COUNT | - | `echo "ARM_ORDER_COUNT_MISMATCH got=$N_GOT_ARMS want=$N_EXPECT_ARMS order=[$ARM_ORDER]"` |
| 13 | `g16_grid.sbatch:227` | HARD_EXIT | - | `exit 1` |
| 14 | `g16_grid.sbatch:346` | REF | - | `for suffix in LO.jsonl HI.jsonl telemetry.jsonl; do` |
| 15 | `g16_grid.sbatch:381` | EXPORT | - | `G16_TELEM_RC="${TELEM_RC:-}" \` |
| 16 | `g16_grid.sbatch:456` | REF | - | `"telem_rc": _f("G16_TELEM_RC", int),` |
| 17 | `g16_grid.sbatch:468` | GATE_REJECT | - | `BOOTS_FAILED=0` |
| 18 | `g16_grid.sbatch:472` | SMOKE_ITEM | - | `SMOKE1=""; SMOKE2=""; SMOKE2B=""; SMOKE3=""; SMOKE4=""; SMOKE7=""` |
| 19 | `g16_grid.sbatch:473` | SMOKE_ITEM | - | `SMOKE9=""; SMOKE9A=""; SMOKE9B=""; SMOKE10=""` |
| 20 | `g16_grid.sbatch:540` | GATE_REJECT | - | `*) echo "BAD_ARM=$ARM"; BOOTS_FAILED=$((BOOTS_FAILED + 1)); FAILED_ARMS="$FAILED_ARMS $ARM` |
| 21 | `g16_grid.sbatch:544` | GATE_REJECT | - | `BOOTS_FAILED=$((BOOTS_FAILED + 1)); FAILED_ARMS="$FAILED_ARMS $ARM"` |
| 22 | `g16_grid.sbatch:607` | EXPORT | - | `export PDMUX_TELEMETRY_PATH="$HERE/${RUNID}_telemetry.jsonl"; : > "$PDMUX_TELEMETRY_PATH"` |
| 23 | `g16_grid.sbatch:615` | EXPORT | - | `TELEM_RC=""; FIELDS_LO_RC=""; FIELDS_HI_RC=""; PIN_RC=""; CTRLSUMMARY_RC=""` |
| 24 | `g16_grid.sbatch:648` | GATE_REJECT | - | `BOOTS_FAILED=$((BOOTS_FAILED + 1)); FAILED_ARMS="$FAILED_ARMS $ARM"` |
| 25 | `g16_grid.sbatch:691` | GATE_REJECT | - | `BOOTS_FAILED=$((BOOTS_FAILED + 1)); FAILED_ARMS="$FAILED_ARMS $ARM"` |
| 26 | `g16_grid.sbatch:701` | REF | - | `python3 "$HERE/g16_assert.py" runtime_snapshots "$PDMUX_TELEMETRY_PATH" "$MIN_SNAPSHOTS"` |
| 27 | `g16_grid.sbatch:702` | EXPORT | - | `TELEM_RC=$?` |
| 28 | `g16_grid.sbatch:752` | REF | - | `python3 - "$PDMUX_TELEMETRY_PATH" "$CTRLSUMMARY_JSON" <<PYEOF` |
| 29 | `g16_grid.sbatch:801` | REF | - | `ARTIFACT_OK=1` |
| 30 | `g16_grid.sbatch:802` | GATE_REJECT | - | `[ "$TELEM_RC" -eq 0 ] || ARTIFACT_OK=0` |
| 31 | `g16_grid.sbatch:803` | GATE_REJECT | - | `[ "$FIELDS_LO_RC" -eq 0 ] || ARTIFACT_OK=0` |
| 32 | `g16_grid.sbatch:804` | GATE_REJECT | - | `[ "$FIELDS_HI_RC" -eq 0 ] || ARTIFACT_OK=0` |
| 33 | `g16_grid.sbatch:807` | REF | - | `"telemetry_rc=$TELEM_RC fields_lo_rc=$FIELDS_LO_RC fields_hi_rc=$FIELDS_HI_RC" \` |
| 34 | `g16_grid.sbatch:808` | REF | - | `"pin_rc=$PIN_RC ctrlsummary_rc=$CTRLSUMMARY_RC artifact_ok=$ARTIFACT_OK" \` |
| 35 | `g16_grid.sbatch:809` | REF | - | `"lo=$JLO hi=$JHI telemetry=$PDMUX_TELEMETRY_PATH run_id=$RUNID"` |
| 36 | `g16_grid.sbatch:811` | REF | - | `if [ "$ARTIFACT_OK" != 1 ]; then` |
| 37 | `g16_grid.sbatch:812` | REF | - | `echo "ARTIFACT_INVALID arm=$ARM block=$BLOCK_TAG boot=$BOOT_IDX telemetry_rc=$TELEM_RC" \` |
| 38 | `g16_grid.sbatch:815` | GATE_REJECT | - | `BOOTS_FAILED=$((BOOTS_FAILED + 1)); FAILED_ARMS="$FAILED_ARMS $ARM"` |
| 39 | `g16_grid.sbatch:829` | SMOKE_ITEM | d64 | `[ "$PIN_RC" -eq 0 ] && SMOKE2B=PASS || SMOKE2B=FAIL` |
| 40 | `g16_grid.sbatch:837` | SMOKE_ITEM | d74 | `[ "$TELEM_RC" -eq 0 ] && SMOKE1=PASS || SMOKE1=FAIL` |
| 41 | `g16_grid.sbatch:838` | SMOKE_ITEM | d74 | `[ "$PIN_RC" -eq 0 ] && SMOKE2=PASS || SMOKE2=FAIL` |
| 42 | `g16_grid.sbatch:839` | SMOKE_ITEM | d74 | `if [ "$FIELDS_LO_RC" -eq 0 ] && [ "$FIELDS_HI_RC" -eq 0 ]; then SMOKE3=PASS; else SMOKE3=F` |
| 43 | `g16_grid.sbatch:840` | SMOKE_ITEM | d74 | `case "$RUNID" in *smoke1_d74_boot1*) SMOKE4=PASS ;; *) SMOKE4=FAIL ;; esac` |
| 44 | `g16_grid.sbatch:841` | REF | d74 | `echo "G16_SMOKE_CHECK 1_TELEMETRY_RUNTIME_SNAPSHOTS=$SMOKE1 min=$MIN_SNAPSHOTS"` |
| 45 | `g16_grid.sbatch:850` | SMOKE_ITEM | d74 | `[ -n "$D74_WALLTIME_S" ] && SMOKE7=PASS || SMOKE7=FAIL` |
| 46 | `g16_grid.sbatch:851` | SMOKE_ITEM | d74 | `if [ -n "$D74_BOOT_S" ] && [ -n "$D74_BENCH_S" ]; then SMOKE10=PASS; else SMOKE10=FAIL; fi` |
| 47 | `g16_grid.sbatch:875` | SMOKE_ITEM | d74 | `SMOKE9_COUNT=$(python3 - "$PDMUX_TELEMETRY_PATH" <<'PYEOF'` |
| 48 | `g16_grid.sbatch:887` | REF | d74 | `if e.get("event") == "runtime_snapshot" and e.get("phase") == "benchmark" and "decode_sms"` |
| 49 | `g16_grid.sbatch:892` | SMOKE_ITEM | d74 | `SMOKE9_COUNT_RC=$?` |
| 50 | `g16_grid.sbatch:894` | SMOKE_ITEM | d74 | `''|*[!0-9]*) SMOKE9A=FAIL; SMOKE9_COUNT="${SMOKE9_COUNT:-<no_output>}" ;;` |
| 51 | `g16_grid.sbatch:895` | REF | d74 | `*) if [ "$SMOKE9_COUNT_RC" -eq 0 ] && [ "$SMOKE9_COUNT" -ge "$MIN_SNAPSHOTS" ]; then` |
| 52 | `g16_grid.sbatch:896` | SMOKE_ITEM | d74 | `SMOKE9A=PASS` |
| 53 | `g16_grid.sbatch:898` | SMOKE_ITEM | d74 | `SMOKE9A=FAIL` |
| 54 | `g16_grid.sbatch:909` | SMOKE_ITEM | d74 | `SMOKE9_RES_OUT=$(python3 - "$CTRLSUMMARY_JSON" <<'PYEOF'` |
| 55 | `g16_grid.sbatch:946` | SMOKE_ITEM | d74 | `SMOKE9B_RC=$?` |
| 56 | `g16_grid.sbatch:948` | SMOKE_ITEM | d74 | `SMOKE9_REASON="${_s9[0]-}"; [ -n "$SMOKE9_REASON" ] || SMOKE9_REASON="no_output"` |
| 57 | `g16_grid.sbatch:949` | SMOKE_ITEM | d74 | `SMOKE9_RES_SUM="${_s9[1]-}"; [ -n "$SMOKE9_RES_SUM" ] || SMOKE9_RES_SUM="n/a"` |
| 58 | `g16_grid.sbatch:950` | SMOKE_ITEM | d74 | `SMOKE9_RESIDENCY="${_s9[2]-}"; [ -n "$SMOKE9_RESIDENCY" ] || SMOKE9_RESIDENCY="{}"` |
| 59 | `g16_grid.sbatch:951` | SMOKE_ITEM | d74 | `if [ "$SMOKE9B_RC" -eq 0 ] && [ "$SMOKE9_REASON" = ok ]; then SMOKE9B=PASS; else SMOKE9B=F` |
| 60 | `g16_grid.sbatch:952` | SMOKE_ITEM | d74 | `if [ "$SMOKE9A" = PASS ] && [ "$SMOKE9B" = PASS ]; then SMOKE9=PASS; else SMOKE9=FAIL; fi` |
| 61 | `g16_grid.sbatch:954` | REF | d74 | `"runtime_snapshot_benchmark_count=$SMOKE9_COUNT min=$MIN_SNAPSHOTS" \` |
| 62 | `g16_grid.sbatch:955` | REF | d74 | `"recount_rc=$SMOKE9_COUNT_RC assert_telem_rc=$TELEM_RC"` |
| 63 | `g16_grid.sbatch:965` | GATE_REJECT | - | `echo "G16_BLOCK_DONE block=$BLOCK_TAG arms_total=$N_GOT_ARMS BOOTS_FAILED=$BOOTS_FAILED" \` |
| 64 | `g16_grid.sbatch:969` | SMOKE_ITEM | - | `[ "$BOOTS_FAILED" -eq 0 ] && SMOKE5=PASS || SMOKE5=FAIL` |
| 65 | `g16_grid.sbatch:970` | SMOKE_ITEM | - | `if [ -s "$MANIFEST" ] && [ "$GIT_COMMIT" != "unknown" ]; then SMOKE6=PASS; else SMOKE6=FAI` |
| 66 | `g16_grid.sbatch:971` | GATE_REJECT | - | `echo "G16_SMOKE_CHECK 5_BOOTS_FAILED=$SMOKE5 value=$BOOTS_FAILED"` |
| 67 | `g16_grid.sbatch:987` | SMOKE_OVERALL | - | `echo "G16_SMOKE_OVERALL=PASS (sec9 conditions 1-7 + addendum A-7 items 8/9/10 satisfied)"` |
| 68 | `g16_grid.sbatch:989` | SMOKE_OVERALL | - | `echo "G16_SMOKE_OVERALL=FAIL (>=1 condition not satisfied -- do NOT submit the full campai` |
| 69 | `g16_analyze.py:18` | REF | - | `(26 telemetry files) whose selection script is not preserved in` |
| 70 | `g16_analyze.py:937` | REF | - | `Adoption: ``status == 'completed' AND telem_rc == 0 AND fields_rc.lo == 0` |
| 71 | `g16_analyze.py:939` | REF | - | `ran BEFORE the ``ARTIFACT_OK`` judgment (N1); the harness has since been` |
| 72 | `g16_analyze.py:971` | ADOPT | - | `and sidecar.get("telem_rc") == 0` |
| 73 | `g16_analyze.py:977` | REF | - | `"telem_rc": sidecar.get("telem_rc"),` |
| 74 | `g16_analyze.py:1565` | REF | - | `"telemetry is not a free observer: the 2026-07-15/18 grid ran WITHOUT "` |
| 75 | `g16_assert.py:5` | REF | - | `telemetry file).` |
| 76 | `g16_assert.py:10` | REF | - | `* H2 -> H2': "telemetry file non-empty" (equivalent to "did you export the` |
| 77 | `g16_assert.py:11` | REF | - | `env var") is replaced with "telemetry has >= N events with` |
| 78 | `g16_assert.py:12` | REF | - | `event=='runtime_snapshot' and phase=='benchmark' and 'decode_sms' in` |
| 79 | `g16_assert.py:14` | REF | - | `telemetry file containing e.g. only `controller_decision` events.` |
| 80 | `g16_assert.py:38` | REF | - | `4. telemetry with 0 runtime_snapshot(phase=benchmark, has decode_sms) events` |
| 81 | `g16_assert.py:58` | ASSERT_FN | - | `def assert_runtime_snapshots(path: str, min_count: int) -> tuple[bool, str]:` |
| 82 | `g16_assert.py:61` | REF | - | `return False, f"telemetry file missing: {path}"` |
| 83 | `g16_assert.py:63` | REF | - | `return False, f"telemetry file empty (0 bytes): {path}"` |
| 84 | `g16_assert.py:78` | REF | - | `and e.get("event") == "runtime_snapshot"` |
| 85 | `g16_assert.py:85` | REF | - | `f"runtime_snapshot(phase=benchmark, has decode_sms) count={n_snapshots} "` |
| 86 | `g16_assert.py:86` | REF | - | `f"< min={min_count} (telemetry lines={n_lines}): {path}"` |
| 87 | `g16_assert.py:89` | REF | - | `f"lines={n_lines} runtime_snapshot_benchmark_count={n_snapshots} "` |
| 88 | `g16_assert.py:202` | REF | - | `rs = sub.add_parser("runtime_snapshots", help="H2' telemetry assert")` |
| 89 | `g16_assert.py:217` | REF | - | `if args.cmd == "runtime_snapshots":` |
| 90 | `g16_assert.py:218` | ASSERT_FN | - | `ok, detail = assert_runtime_snapshots(args.path, args.min_count)` |
| 91 | `g16_assert.py:255` | ASSERT_FN | - | `ok, detail = assert_runtime_snapshots("/nonexistent/path/g16_does_not_exist.jsonl", 1)` |
| 92 | `g16_assert.py:256` | REF | - | `check("runtime_snapshots missing file -> ok", ok, False)` |
| 93 | `g16_assert.py:261` | ASSERT_FN | - | `ok, detail = assert_runtime_snapshots(empty_path, 1)` |
| 94 | `g16_assert.py:262` | REF | - | `check("runtime_snapshots empty file -> ok", ok, False)` |
| 95 | `g16_assert.py:272` | REF | - | `fh.write(json.dumps({"event": "runtime_snapshot", "phase": "warmup", "decode_sms": 44}) + ` |
| 96 | `g16_assert.py:273` | REF | - | `fh.write(json.dumps({"event": "runtime_snapshot", "phase": "benchmark"}) + "\n")  # no dec` |
| 97 | `g16_assert.py:275` | ASSERT_FN | - | `ok, detail = assert_runtime_snapshots(no_snapshot_path, 1)` |
| 98 | `g16_assert.py:277` | REF | - | `"SYNTHETIC#4 runtime_snapshot(phase=benchmark,decode_sms) count=0 -> ok",` |
| 99 | `g16_assert.py:290` | REF | - | `{"event": "runtime_snapshot", "phase": "benchmark", "decode_sms": 44,` |
| 100 | `g16_assert.py:296` | ASSERT_FN | - | `ok, detail = assert_runtime_snapshots(enough_path, 5)` |
| 101 | `g16_assert.py:297` | REF | - | `check("runtime_snapshots count=5 min=5 -> ok", ok, True)` |
| 102 | `g16_assert.py:298` | ASSERT_FN | - | `ok, detail = assert_runtime_snapshots(enough_path, 6)` |
| 103 | `g16_assert.py:299` | REF | - | `check("runtime_snapshots count=5 min=6 -> ok", ok, False)` |
| 104 | `g16_n3_dryrun_check.py:12` | REF | - | `2. Leaves PDMUX_TELEMETRY_PATH / PDMUX_RUN_ID / PDMUX_WORKLOAD_ID set to` |
| 105 | `g16_n3_dryrun_check.py:57` | REF | - | `GPU 0, no server/telemetry/torch dependency -- pure bash + text.` |
| 106 | `g16_n3_dryrun_check.py:90` | REF | - | `"PDMUX_TELEMETRY_PATH": "/tmp/STALE_LEAKED_TELEMETRY_PATH.jsonl",` |
| 107 | `g16_n3_dryrun_check.py:95` | REF | - | `NEEDED = {"PDMUX_TELEMETRY_PATH", "PDMUX_RUN_ID", "PDMUX_WORKLOAD_ID"}` |

SMOKE satisfiability with arms=['d44']:

| item | pinned to | depends on | site runs | can ever PASS |
|---|---|---|---|---|
| `SMOKE1` | d74 | SMOKE2,SMOKE2B,SMOKE3,SMOKE4,SMOKE7 | no | **NO** |
| `SMOKE10` | d74 | SMOKE9,SMOKE9A,SMOKE9B | no | **NO** |
| `SMOKE2` | d74 | SMOKE1,SMOKE2B,SMOKE3,SMOKE4,SMOKE7 | no | **NO** |
| `SMOKE2B` | d64 | SMOKE1,SMOKE2,SMOKE3,SMOKE4,SMOKE7 | no | **NO** |
| `SMOKE3` | d74 | SMOKE1,SMOKE2,SMOKE2B,SMOKE4,SMOKE7 | no | **NO** |
| `SMOKE4` | d74 | SMOKE1,SMOKE2,SMOKE2B,SMOKE3,SMOKE7 | no | **NO** |
| `SMOKE5` | - | - | yes | YES |
| `SMOKE6` | - | - | yes | YES |
| `SMOKE7` | d74 | SMOKE1,SMOKE2,SMOKE2B,SMOKE3,SMOKE4 | no | **NO** |
| `SMOKE9` | d74 | SMOKE10,SMOKE9A,SMOKE9B | no | **NO** |
| `SMOKE9A` | d74 | SMOKE10,SMOKE9,SMOKE9B,SMOKE9_COUNT | no | **NO** |
| `SMOKE9B` | d74 | SMOKE10,SMOKE9,SMOKE9A,SMOKE9B_RC,SMOKE9_REASON | no | **NO** |
| `SMOKE9B_RC` | d74 | - | no | **NO** |
| `SMOKE9_COUNT` | d74 | SMOKE9A | no | **NO** |
| `SMOKE9_COUNT_RC` | d74 | - | no | **NO** |
| `SMOKE9_REASON` | d74 | - | no | **NO** |
| `SMOKE9_RESIDENCY` | d74 | - | no | **NO** |
| `SMOKE9_RES_OUT` | d74 | - | no | **NO** |
| `SMOKE9_RES_SUM` | d74 | - | no | **NO** |
