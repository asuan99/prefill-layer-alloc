"""N3 capacity campaign (PREREG_N3_CAPACITY_2026-10-04.md): plan, runner, estimators, analyzer.

Every behavioural claim has a negative control (lesson 53: a check that also passes on the broken
variant certifies nothing; lesson 254: an equivalent mutant is recorded, not hidden).

  registration      digests/mirror/config/engine manifest match.  Control: a one-byte copy drifts;
                    a config whose decode SMs lack 44 fails (the CONFIG_CANDIDATES §1 silent mis-arm).
  units             floor first, bridge PHASE_EVENTS OFF, observer ABBA adjacent, every grid
                    (rate, seed) scored once, rates absolute.  Control: perturbed rows are detected.
  fail-closed       each check in the runner is wired to its abort.  Control: deleting an abort
                    line makes the structural detector fail.
  green read-out    A-agn layout + null sticky passes.  Control: the homog5 read-out of λ0 VESSL
                    (index 2 = (64,44)) and a non-null sticky_target_index both fail.
  floor             rev6: formula + REALISED-evidence controls (span_end stream, decode inside the
                    span, unpaired end, dropped events, n_new) and the Job-79999aa entry-label race
                    admitted (n3_plan selftest, T-H3, TestFloorRev6RealJob).
  estimators        pair-keyed == sealed index-keyed E2 tool on the λ0 VESSL archive (7/7; skipped
                    when the git-ignored archive is absent).  Controls: wrong pair, index remap.
  analyzer          labels/CI/stationarity/observer on synthetic campaigns with known answers.
  runner            allowlist / test-knob / shape aborts; dry-run units + env; budget enforcer
                    (both kill layers removed => overrun; one removed => equivalent mutant);
                    resume (and its digest control).

job layout        rev5: results/ as a symlink (VESSL job_entry.sh) keeps the engine root on the
                    clone.  Control: the pre-rev5 `.resolve()` line restored fails (MISSING /
                    ModuleNotFoundError), i.e. the Job a4b0c2e failure is reproduced.

Classes TestRegistration, TestUnits, TestFailClosedWiring are fast and need no sglang: the Job
preflight runs them against the installed tree.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ENG = Path(__file__).resolve().parents[1]
ROOT = ENG.parents[1]
HERE = ENG / "results" / "n3_capacity"
RUNNER = HERE / "n3_capacity.sh"
BASH = "/usr/bin/bash"
LAM0V = ENG / "results/r2_eval/lambda0_vessl/lam0v_pdmux-lambda0-vessl-1d47d3e-20261004084945"

sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import n3_plan as P  # noqa: E402
import n3_realized_mix as M  # noqa: E402


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PDMUX_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra)
    return env


class TestRegistration(unittest.TestCase):
    def test_live_repo_matches_registered_digests(self):
        self.assertEqual(P.verify_digests(ENG), [])
        self.assertEqual(P.mirror_check(ENG), [])

    def test_control_one_byte_drift_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            for rel in P.registered_files(ENG):
                (Path(td) / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(ENG / rel, Path(td) / rel)
            self.assertEqual(P.verify_digests(Path(td)), [])
            p = Path(td) / P.CONFIG_REL
            p.write_text(p.read_text() + " ")
            self.assertEqual(len(P.verify_digests(Path(td))), 1)

    def test_config_passes_and_fixed_dsm_control(self):
        cfg = ENG / P.CONFIG_REL
        self.assertEqual(P.check_config(cfg), [])
        self.assertEqual(P.stream_groups(cfg)[4], P.TARGET_PAIR)
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "c.yml"
            bad.write_text(cfg.read_text().replace("[64, 44, 36]", "[54, 54, 36]"))
            msgs = P.check_config(bad)
            self.assertTrue(any("PDMUX_R2_FIXED_DSM=44 not in" in m for m in msgs), msgs)
            many = Path(td) / "m.yml"
            many.write_text(cfg.read_text().replace("sm_group_num: 6", "sm_group_num: 9"))
            self.assertTrue(P.check_config(many))

    def test_engine_manifest_is_908623_plus_one_registered_file(self):
        ref = LAM0V / "runtime_source_manifest.sha256"
        if not ref.exists():
            self.skipTest("λ0 VESSL manifest absent")
        d = P.engine_check(ref)
        self.assertEqual(len(d), 1, d)
        self.assertTrue(d[0].startswith("sglang/srt/multiplex/multiplexing_mixin.py"), d)
        with tempfile.TemporaryDirectory() as td:
            mine = Path(td) / "m.sha256"
            mine.write_text("".join(f"{h}  /opt/x/python/{k}\n" for k, h in P.ENGINE_MANIFEST.items()))
            self.assertEqual(P.engine_check(mine), [])
            mine.write_text(mine.read_text().replace(P.ENGINE_MANIFEST["sglang/srt/server_args.py"], "0" * 64))
            self.assertEqual(len(P.engine_check(mine)), 1)

    def test_green_readout_check_and_controls(self):
        cfg = ENG / P.CONFIG_REL
        groups = []
        for i, (p, d) in enumerate(P.stream_groups(cfg)):
            div = 1 <= i <= 4

            def side(sm):
                return ({"green_ctx_is_null": False, "green_sm": {"smCount": sm}} if div
                        else {"green_ctx_is_null": True})
            groups.append({"stream_index": i, "target_prefill_sm": p, "target_decode_sm": d,
                           "is_sticky_target": False, "prefill": side(p), "decode": side(d)})
        with tempfile.TemporaryDirectory() as td:
            g = Path(td) / "g.json"
            g.write_text(json.dumps({"sticky_target_index": None, "groups": groups}))
            self.assertEqual(P.check_green(g, cfg), [])
            g.write_text(json.dumps({"sticky_target_index": 4, "groups": groups}))
            self.assertTrue(P.check_green(g, cfg))
            groups[4]["decode"]["green_sm"]["smCount"] = 42
            g.write_text(json.dumps({"sticky_target_index": None, "groups": groups}))
            self.assertTrue(P.check_green(g, cfg))
        homog5 = LAM0V / "green_a_r0.json"
        h5cfg = ENG / "results/longctx_conflict/probes/pdmux_homog5.yml"
        if homog5.exists() and h5cfg.exists():
            # positive control on a REAL driver read-out: it passes against its own config ...
            self.assertEqual(P.check_green(homog5, h5cfg), [])
            # ... and fails against the A-agn layout
            self.assertTrue(P.check_green(homog5, cfg), "homog5 layout must not pass as A-agn")

    def test_plan_selftest(self):
        P.selftest()


class TestUnits(unittest.TestCase):
    def test_structure(self):
        for shape in ("A", "B"):
            us = [P.parse_row(r) for r in P.units(shape)]
            self.assertEqual(us[0]["name"], "floor_head")
            self.assertEqual(us[-1]["name"], "floor_tail")
            first_arm = next(i for i, u in enumerate(us) if u["kind"] != "floor")
            self.assertEqual(first_arm, 1, "floor must be the first unit (P-2)")
            for u in us:
                if u["kind"] in ("bridge", "obs_off"):
                    self.assertEqual(u["phase_events"], 0, u)
                else:
                    self.assertEqual(u["phase_events"], 1, u)
            sc = P.scored_cells(shape)
            s = P.SHAPES[shape]
            self.assertEqual(set(sc), {(r, sd) for r in s["rates"] for sd in s["seeds"]})
            br = [u for u in us if u["kind"] == "bridge"]
            self.assertEqual([(u["rate"], u["np"], u["seed"]) for u in br],
                             [(s["bridge"]["rate"], s["bridge"]["n"], sd) for sd in (4386, 4162)])

    def test_rates_are_registered_absolute_literals(self):
        self.assertEqual(P.SHAPES["B"]["rates"], (0.10, 0.15, 0.22, 0.30, 0.38, 0.45, 0.53, 0.61, 0.69, 0.77))
        self.assertEqual(P.SHAPES["A"]["rates"], (1.35, 1.55, 1.80, 2.10, 2.40, 2.70, 3.00))
        self.assertEqual(P.grid_ns("B"), [20, 20, 22, 30, 38, 45, 53, 61, 69, 77])
        self.assertEqual(P.grid_ns("A"), [122, 140, 162, 189, 216, 243, 270])

    def test_seeds_are_scorer_top4_and_control(self):
        for shape in ("A", "B"):
            top, _ = P.ranked_seeds(shape)
            self.assertEqual(tuple(top), P.SHAPES[shape]["seeds"])
        # control: λ0's seeds fail the Ebar guard at B's small N (why seeds are re-ranked)
        self.assertTrue(any(not (0.95 <= 1 / P.LP.ebar(4162, n) <= 1.05) for n in P.grid_ns("B")))

    def test_r1_budget_cut_removes_highest_rate_first(self):
        for shape in ("A", "B"):
            us = [P.parse_row(r) for r in P.units(shape)]
            grid = [u for u in us if u["kind"] == "grid"]
            self.assertEqual(grid[-1]["rate"], max(P.SHAPES[shape]["rates"]), "last unit must be the top rate")
            self.assertEqual(grid[0]["rate"], max(P.SHAPES[shape]["rates"]))
            for i in range(4):
                rr = [u["rate"] for u in us if u["block"] == f"round{i + 1}"]
                self.assertEqual(rr, sorted(rr, reverse=(i % 2 == 0)), f"round{i + 1}")
            # control: the rev1 order (round 1,3 ascending) ends on the LOWEST rate
            rest = [r for r in P.SHAPES[shape]["rates"] if r != P.SHAPES[shape]["obs_rate"]]
            rev1_last_round = list(reversed(rest))           # rev1: round 4 (i=3) descending
            self.assertEqual(rev1_last_round[-1], min(rest))

    def test_r1_truncation_keeps_lowest_rates_complete(self):
        # simulate a cut of the last unit (and of the last 3): no rate below the cut set loses a seed
        for shape in ("A", "B"):
            us = [P.parse_row(r) for r in P.units(shape) if P.parse_row(r)["kind"] in ("grid", "obs_on")]
            for cut in (1, 3):
                kept = us[:-cut]
                lost = {u["rate"] for u in us[-cut:]}
                n = {}
                for u in kept:
                    n[u["rate"]] = n.get(u["rate"], 0) + 1
                lowest_short = min(r for r in P.SHAPES[shape]["rates"] if n.get(r, 0) < 4)
                self.assertEqual(lowest_short, min(lost), (shape, cut))
                self.assertGreater(lowest_short, min(P.SHAPES[shape]["rates"]))

    def test_observer_pairs_adjacent_abba(self):
        for shape in ("A", "B"):
            ob = [P.parse_row(r) for r in P.units(shape) if P.parse_row(r)["block"] == "observer"]
            for i in range(0, 8, 2):
                a, b = ob[i], ob[i + 1]
                self.assertEqual((a["rate"], a["seed"]), (b["rate"], b["seed"]))
                self.assertEqual({a["kind"], b["kind"]}, {"obs_on", "obs_off"})

    def test_budget_layers(self):
        for shape in ("A", "B"):
            n, w = P.simulate(shape, "normal"), P.simulate(shape, "worst")
            self.assertTrue(n["all_admitted"])
            # rev7: shape B re-estimated from measurement (normal corner 4.24 GPU-h, was 3.37); A unchanged
            self.assertLessEqual(n["gpu_h"], 4.3 if shape == "B" else 3.5)
            self.assertGreaterEqual(w["admitted"], P.REV7_B_WORST_MIN_ADMITTED if shape == "B" else w["units"] - 1)


class TestFailClosedWiring(unittest.TestCase):
    PAIRS = (('python3 "$HELPER" --verify-digests', "ABORT_REG_DRIFT"),
             ('python3 "$HELPER" --check-config', "ABORT_CONFIG"),
             ('python3 "$HELPER" --engine-check', "ABORT_ENGINE_TREE_DRIFT"),
             ('grep -q "def pdmux_role_is_thread_local"', "ABORT_ENGINE"),
             ('grep -q "sticky partition ENABLED"', "ABORT_STICKY"),
             ('python3 "$HELPER" --check-green', "ABORT_GREEN"),
             ('python3 "$HELPER" --floor', "ABORT_FLOOR"),
             ('! tk_valid "$OUT/T_K.json"', "ABORT_FLOOR"),
             ('MMB=$(grep -o', "ABORT_P3"),
             ('python3 "$HELPER" --check-substrate', "ABORT_SUBSTRATE"),
             ('python3 - "$OUT/warmup_floor_head.jsonl"', "ABORT_CG2"))

    @staticmethod
    def _call(abort):
        # the abort must be the FIRST WORD of a `fail` / `dry_or_abort` call (an echo of the same
        # text, or the text in a comment, does not count)
        return re.compile(r'(?:\bfail|\bdry_or_abort) "' + re.escape(abort) + r'[ "]')

    @classmethod
    def _wired(cls, src, check, abort, window=900):
        try:
            i = src.index(check)
        except ValueError:
            return False
        return bool(cls._call(abort).search(src[i:i + window]))

    def test_every_check_is_wired_to_its_abort(self):
        src = RUNNER.read_text()
        for check, abort in self.PAIRS:
            self.assertTrue(self._wired(src, check, abort), check)

    def test_control_each_abort_branch_neutralised_is_detected(self):
        """Per-branch mutation: turn THAT branch's `fail`/`dry_or_abort "ABORT_X` into `echo "ABORT_X`
        (message text kept, abort gone).  The detector must fail for every one of the 11 branches."""
        src = RUNNER.read_text()
        for check, abort in self.PAIRS:
            i = src.index(check)
            seg = src[i:i + 900]
            mseg = re.sub(r'(?:\bfail|\bdry_or_abort) ("' + re.escape(abort) + r'[ "])', r'echo \1', seg)
            self.assertNotEqual(mseg, seg, f"mutation did not apply for {abort}")
            mut = src[:i] + mseg + src[i + 900:]
            self.assertFalse(self._wired(mut, check, abort), f"neutralised {abort} still looks wired")

    @staticmethod
    def _order_ok(src):
        body = src[src.index('for U in "${UNITS[@]}"; do\n  read -r NAME'):]
        try:
            tk = body.index('! tk_valid "$OUT/T_K.json"')
            adm = body.index('if [ "$REM" -lt "$NEED" ]; then')
        except ValueError:
            return False
        boot = body.index('setsid "${SRV[@]}"')
        return tk < boot and adm < boot

    def test_floor_gate_and_admission_precede_boot(self):
        self.assertTrue(self._order_ok(RUNNER.read_text()))

    def test_control_removed_admission(self):
        src = RUNNER.read_text()
        mut = re.sub(r'  if \[ "\$REM" -lt "\$NEED" \]; then\n.*?\n  fi\n', "", src, count=1, flags=re.S)
        self.assertNotEqual(mut, src)
        self.assertFalse(self._order_ok(mut))

    def test_one_per_boot_knob_and_fixed_arm(self):
        src = RUNNER.read_text()
        self.assertIn('if [ "$PEV" = 1 ]; then export PDMUX_PHASE_EVENTS=1; else unset PDMUX_PHASE_EVENTS; fi', src)
        self.assertIn('export PDMUX_R2_POLICY="$R2_POLICY"', src)
        self.assertIn('export PDMUX_R2_FIXED_DSM="$FIXED_DSM"', src)
        self.assertNotIn("export PDMUX_STICKY_PARTITION", src)
        self.assertIn("PDMUX_STICKY_PARTITION", src.split("unset SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN")[1][:400])
        self.assertNotIn("--disable-cuda-graph", src)
        self.assertNotIn("--disable-piecewise-cuda-graph", src)

    def test_cap_and_constants_from_helper_not_env(self):
        src = RUNNER.read_text()
        self.assertIn('HARD_CAP_S=$(K "HARD_CAP_S_${SHAPE}")', src)
        self.assertIn('[ -n "${PDMUX_N3_TEST_HARD_CAP_S:-}" ] && HARD_CAP_S=', src)
        self.assertIn("PDMUX_N3_TEST_HARD_CAP_S PDMUX_N3_TEST_FAKE_BENCH_S", src.split('if [ "$DRYRUN" != 1 ]; then')[1][:300])


class TestPortAndTeardown(unittest.TestCase):
    """E2 diagnosis (A14 DRAFT O1-O4) applied to the N3 runner.  O2 mutation: the OLD check
    (`ss -ltn`, LISTEN only) misses a TIME_WAIT socket on the candidate port; the NEW check
    (`ss -Htan`, every state) catches it.  O1: the listen window must sit outside the kernel
    ephemeral range, and an overlapping range aborts (ABORT_PORT_RANGE)."""

    def setUp(self):
        self.src = RUNNER.read_text()

    def _fake_ss(self, td, port):
        # `ss -ltn` (old) -> LISTEN sockets only (none on `port`); `ss -Htan` (new) -> every
        # state, including a TIME-WAIT socket whose LOCAL port is `port` (the E2 boot-10 case).
        ss = Path(td) / "ss"
        ss.write_text("#!/bin/bash\n"
                      "case \"$*\" in\n"
                      f"  *-Htan*) echo 'TIME-WAIT 0 0 127.0.0.1:{port} 127.0.0.1:39001';;\n"
                      "  *-ltn*) echo 'State Recv-Q Send-Q Local Address:Port Peer Address:Port';"
                      " echo 'LISTEN 0 4096 127.0.0.1:22 0.0.0.0:*';;\n"
                      "esac\n")
        ss.chmod(0o755)
        return {**os.environ, "PATH": f"{td}:{os.environ['PATH']}"}

    def test_o2_new_check_sees_time_wait_old_check_misses_it(self):
        line = next(ln for ln in self.src.splitlines() if ln.startswith("port_in_use () {"))
        with tempfile.TemporaryDirectory() as td:
            env = self._fake_ss(td, 20123)
            new = subprocess.run([BASH, "-c", f"{line}\nport_in_use 20123"], env=env)
            self.assertEqual(new.returncode, 0, "new check must report the TIME_WAIT port as in use")
            free = subprocess.run([BASH, "-c", f"{line}\nport_in_use 20124"], env=env)
            self.assertEqual(free.returncode, 1, "an unrelated port must be free")
            old = subprocess.run([BASH, "-c", 'ss -ltn 2>/dev/null | grep -q ":20123 "'], env=env)
            self.assertEqual(old.returncode, 1, "control: the old LISTEN-only check misses TIME_WAIT")
        self.assertNotIn("ss -ltn", self.src)
        self.assertIn('if ! port_in_use "$CAND"; then PORT=$CAND; break; fi', self.src)

    def _range_block(self, src):
        i = src.index("PORT_LO=")
        j = src.index("\nfi\n", i) + 4
        return src[i:j]

    def _run_range(self, block, lo, hi):
        with tempfile.TemporaryDirectory() as td:
            rng = Path(td) / "range"
            rng.write_text(f"{lo}\t{hi}\n")
            b = block.replace("/proc/sys/net/ipv4/ip_local_port_range", str(rng))
            script = f'OUT={td}\nfail () {{ echo "$1"; exit "$2"; }}\n{b}\necho RANGE_OK\n'
            r = subprocess.run([BASH, "-c", script], capture_output=True, text=True)
            return r.returncode, r.stdout

    def test_o1_window_outside_ephemeral_and_overlap_aborts(self):
        block = self._range_block(self.src)
        self.assertIn("PORT_LO=20000; PORT_SPAN=800", block)
        rc, out = self._run_range(block, 32768, 60999)          # Linux default / λ0 VESSL host
        self.assertEqual(rc, 0, out)
        self.assertIn("RANGE_OK", out)
        rc, out = self._run_range(block, 10000, 30000)          # an overlapping host config
        self.assertEqual(rc, 5, out)
        self.assertIn("ABORT_PORT_RANGE", out)
        # control: the OLD window (39100+800) is inside the default ephemeral range -> must abort
        old = block.replace("PORT_LO=20000", "PORT_LO=39100")
        rc, out = self._run_range(old, 32768, 60999)
        self.assertEqual(rc, 5, out)
        self.assertNotIn("39100 +", self.src)
        self.assertIn("CAND=$((PORT_LO + (RANDOM + $$ + try) % PORT_SPAN))", self.src)

    def test_o3_teardown_kills_renamed_children(self):
        body = self.src[self.src.index("teardown () {"):self.src.index("\n}\n", self.src.index("teardown () {"))]
        self.assertIn('pkill -9 -f "launch_server.*--port $2"', body)
        self.assertIn('pkill -9 -f "^sglang::"', body)

    def test_o4_settle_before_port_and_not_a_gate(self):
        body = self.src[self.src.index('for U in "${UNITS[@]}"; do\n  read -r NAME'):]
        self.assertLess(body.index('[ "$DRYRUN" = 1 ] || settle "$NAME"'), body.index("PORT=0"))
        settle = self.src[self.src.index("settle () {"):self.src.index("\n}\n", self.src.index("settle () {"))]
        self.assertNotIn("fail ", settle)
        self.assertNotIn("exit", settle)
        self.assertIn("seq 1 36", settle)                      # 36 x 5 s = 180 s


class TestEstimators(unittest.TestCase):
    def test_synthetic(self):
        self.assertEqual(M.selftest_synthetic(), 0)

    def test_d1_empty_telemetry_is_unverifiable(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tel.jsonl"
            p.write_text("")                              # the runner pre-creates this file
            ai = M.arm_identity(M.read_events(p))
            self.assertFalse(ai["ok"])
            self.assertTrue(ai["unverifiable"])

    def test_d1_on_cell_without_decode_events_is_unverifiable(self):
        snap = {"event": "runtime_snapshot", "phase": "benchmark", "decode_running_batch_size": 2,
                "prefill_sms": 64, "decode_sms": 44, "timestamp_monotonic_s": 1.0}
        self.assertFalse(M.arm_identity([snap], expect_decode_iterations=True)["ok"])
        # control: OFF / R7-marked cell -> the snapshot leg alone is evidence
        self.assertTrue(M.arm_identity([snap], expect_decode_iterations=False)["ok"])

    @unittest.skipUnless((LAM0V / "tel_a_r4.jsonl").exists(), "git-ignored λ0 VESSL telemetry absent")
    def test_archive_positive_control_7_of_7(self):
        self.assertEqual(M.selftest_archive(LAM0V), 0)

    @unittest.skipUnless((LAM0V / "tel_b_r3.jsonl").exists(), "git-ignored λ0 VESSL telemetry absent")
    def test_control_index_keyed_reading_of_the_new_layout_is_wrong(self):
        # on the A-agn layout (64,44) sits at index 4: reading index 2 (E2 convention) counts
        # (84,24) instead.  Simulate by relabelling pairs and show the pair key follows the pair.
        ev = M.read_events(LAM0V / "tel_b_r3.jsonl")
        a = M.snapshot_estimators(ev, (64, 44))["e_cnt"]["pct"]
        self.assertGreater(a, 80.0)
        self.assertLess(M.snapshot_estimators(ev, (84, 24))["e_cnt"]["pct"], 1.0)


class TestP3Tolerance(unittest.TestCase):
    """rev4 (prereg §2-3 R6): P3 option (ii), same rule and constants as E2 A14 §3-1.  The helper
    is tested directly, then the runner's REAL P3 snippet is executed with stubs (behavioural)."""
    E2_JOB1 = [["48", "2722025"]] * 9 + [["48", "2722022"]]
    SIX = [["48", "2721290"]] * 9
    BANNER = "x max_mamba_cache_size: {m} y\nz max_total_num_tokens={t} w\n"
    # the rev3 strict branch (commit 69fbbb0), kept literally for the "what (ii) changes" control
    STRICT_R3 = (
        '  MMB=$(grep -o "max_mamba_cache_size: *[0-9]*" "$SRVLOG" | tail -1 | grep -o "[0-9]*$")\n'
        '  MTT=$(grep -o "max_total_num_tokens=[0-9]*" "$SRVLOG" | tail -1 | grep -o "[0-9]*$")\n'
        '  echo "${MMB:-NONE} ${MTT:-NONE}" >> "$OUT/P3_KEYS.txt"\n'
        '  if [ "${MMB:-NONE}" != "48" ] || [ "$(sort -u "$OUT/P3_KEYS.txt" | wc -l)" -gt 1 ]; then\n'
        '    teardown "$PID" "$PORT"; fail "ABORT_P3 strict" 8\n'
        '  fi\n')

    def test_constants(self):
        self.assertEqual((P.P3_MAMBA_EXPECT, P.P3_TOKEN_FLOOR, P.P3_REL_TOL), (48, 786432, 1e-4))
        self.assertEqual((P.P3_MAX_RUNNING, P.P3_CTX), (48, 16384))
        self.assertTrue(P.p3_verdict([["48", "786432"]], 48, 16384)[0])
        self.assertFalse(P.p3_verdict(self.E2_JOB1, 48, 8192)[0])
        # same rule as E2 A14 r2 helper (read-only import of the sealed E2 file, no bytecode)
        sys.path.insert(0, str(ENG / "results/r2_eval/e2_sticky_vessl"))
        try:
            import e2_vessl_plan_r2 as V2
        finally:
            sys.path.pop(0)
        self.assertEqual((V2.P3_MAMBA_EXPECT, V2.P3_TOKEN_FLOOR, V2.P3_REL_TOL),
                         (P.P3_MAMBA_EXPECT, P.P3_TOKEN_FLOOR, P.P3_REL_TOL))
        for rows in (self.E2_JOB1, self.SIX, [["48", "2721290"], ["48", "2722025"]],
                     [["47", "2722025"]], [["48", "786431"]], [["48", "NONE"]], []):
            self.assertEqual(P.p3_verdict(rows, 48, 16384)[0], V2.p3_verdict(rows, 48, 16384)[0], rows)

    def test_e2_job1_keys_pass_and_mutants_fail(self):
        ok, line = P.p3_verdict(self.E2_JOB1, 48, 16384)
        self.assertTrue(ok, line)
        self.assertIn("rel=1.102e-06", line)
        mutants = {"mamba 47/48": [["48", "2722025"], ["47", "2722025"]],
                   "mamba all 47": [["47", "2722025"]] * 3,
                   "tokens 0.8% apart": [["48", "2722025"], ["48", str(round(2722025 * 0.992))]],
                   "786431": [["48", "786431"]] * 2,
                   "banner missing": [["48", "2722025"], ["NONE", "NONE"]],
                   "5g vs 6g pool (rel 2.70e-4)": [["48", "2721290"], ["48", "2722025"]]}
        for name, rows in mutants.items():
            ok, line = P.p3_verdict(rows, 48, 16384)
            self.assertFalse(ok, name)
            self.assertTrue(line.startswith("FAIL "), (name, line))

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "k.txt"
            p.write_text("".join(f"{a} {b}\n" for a, b in self.E2_JOB1))
            cmd = [sys.executable, "-B", str(HERE / "n3_plan.py"), "--p3-verdict", str(p), "48", "16384"]
            r = subprocess.run(cmd, capture_output=True, text=True, env=_clean_env())
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertTrue(r.stdout.startswith("PASS "))
            p.write_text("48 2722025\n47 2722025\n")
            r = subprocess.run(cmd, capture_output=True, text=True, env=_clean_env())
            self.assertEqual(r.returncode, 8)
            self.assertTrue(r.stdout.startswith("FAIL "))

    @staticmethod
    def _snippet(src):
        a = src.index('  MMB=$(grep -o "max_mamba_cache_size')
        b = src.index("  esac\n", a) + len("  esac\n")
        return src[a:b]

    def _run(self, snippet, prior_rows, banner, helper=None):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "P3_KEYS.txt").write_text("".join(f"{a} {b}\n" for a, b in prior_rows))
            (out / "srv.log").write_text(banner)
            h = helper or str(HERE / "n3_plan.py")
            script = (f'OUT="{td}"; NAME=X; SRVLOG="{td}/srv.log"; HELPER="{h}"; MAX_RUNNING=48; CTX=16384; PID=1; PORT=1\n'
                      'export PYTHONDONTWRITEBYTECODE=1\n'
                      'teardown () { echo TEARDOWN >> "$OUT/td.txt"; }\n'
                      'fail () { echo "$1"; exit "$2"; }\n' + snippet + "\necho P3_PASSED\n")
            r = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=60, env=_clean_env())
            td_called = (out / "td.txt").exists()
            verdicts = (out / "P3_VERDICTS.txt").read_text() if (out / "P3_VERDICTS.txt").exists() else ""
            banner_rec = (out / "banner_X.txt").read_text() if (out / "banner_X.txt").exists() else ""
            return r.returncode, r.stdout, td_called, verdicts, banner_rec

    def test_runner_snippet(self):
        snip = self._snippet(RUNNER.read_text())
        rc, out, td, v, b = self._run(snip, self.E2_JOB1[:9], self.BANNER.format(m=48, t=2722022))
        self.assertEqual(rc, 0, out)
        self.assertIn("P3_PASSED", out)
        self.assertIn("X P3 rc=0 PASS ", v)
        self.assertFalse(td)
        self.assertEqual(b, "max_mamba_cache_size=48 max_total_num_tokens=2722022\n")   # per-cell banner kept
        for name, prior, (m, t) in (("mamba 47/48", self.SIX, (47, 2721290)),
                                    ("all 47", [["47", "2721290"]] * 3, (47, 2721290)),
                                    ("0.8% drift", self.SIX, (48, round(2721290 * 0.992))),
                                    ("786431", [], (48, 786431)),
                                    ("5g pool in a 6g Job", self.SIX, (48, 2722025))):
            rc, out, td, v, _ = self._run(snip, prior, self.BANNER.format(m=m, t=t))
            self.assertEqual(rc, 8, (name, out))
            self.assertIn("ABORT_P3", out)
            self.assertTrue(td, name)                        # server torn down before exit
            self.assertIn("X P3 rc=8 FAIL", v)
        rc, out, td, v, b = self._run(snip, self.SIX, "no banner here\n")              # banner missing
        self.assertEqual(rc, 8, out)
        self.assertIn("NONE NONE", b.replace("max_mamba_cache_size=", "").replace("max_total_num_tokens=", ""))
        rc, out, _, v, _ = self._run(snip, self.SIX, self.BANNER.format(m=48, t=2721290),
                                     helper="/nonexistent/n3_plan.py")                 # helper absent: fail-closed
        self.assertEqual(rc, 8, out)
        self.assertIn("EMPTY", v)
        with tempfile.TemporaryDirectory() as td:
            liar = Path(td) / "liar.py"                     # prints PASS but exits non-zero
            liar.write_text('import sys\nprint("PASS fake")\nsys.exit(1)\n')
            liar0 = Path(td) / "liar0.py"                   # exits 0 but prints no PASS line
            liar0.write_text('print("OK")\n')
            noisy = Path(td) / "noisy.py"                   # warning on stderr, then a real PASS
            noisy.write_text('import sys\nprint("warning: x", file=sys.stderr)\nprint("PASS ok")\n')
            rc, out, *_ = self._run(snip, self.SIX, self.BANNER.format(m=48, t=2721290), helper=str(liar))
            self.assertEqual(rc, 8, out)
            rc, out, *_ = self._run(snip, self.SIX, self.BANNER.format(m=48, t=2721290), helper=str(liar0))
            self.assertEqual(rc, 8, out)
            rc, out, *_ = self._run(snip, self.SIX, self.BANNER.format(m=48, t=2721290), helper=str(noisy))
            self.assertEqual(rc, 0, out)

    def test_control_neutralised_case_branch_is_detected(self):
        src = RUNNER.read_text()
        m = src.replace('    0:PASS\\ *) ;;\n    *) teardown', '    *) ;;\n    NEVER) teardown', 1)
        self.assertNotEqual(m, src, "mutation did not apply")
        rc, out, *_ = self._run(self._snippet(m), self.SIX, self.BANNER.format(m=47, t=2721290))
        self.assertEqual(rc, 0, "neutralised P3 still aborted -- the test is vacuous")

    def test_rev3_strict_branch_aborts_on_e2_job1_keys(self):
        # what (ii) changes: the rev3 strict rule aborts on exactly the E2 first-Job keys
        rc, out, *_ = self._run(self.STRICT_R3 + "  case x in x) ;; esac\n", self.E2_JOB1[:9],
                                self.BANNER.format(m=48, t=2722022))
        self.assertEqual(rc, 8, out)
        self.assertNotIn('sort -u "$OUT/P3_KEYS.txt" | wc -l)" -gt 1 ]', RUNNER.read_text())


class TestAnalyzer(unittest.TestCase):
    def test_selftest(self):
        import n3_analyze as N
        N.selftest()


class _RunnerBase(unittest.TestCase):
    def _run(self, td, shape="B", job="t", timeout=180, script=RUNNER, **env):
        e = _clean_env(PDMUX_PROJECT_ROOT=str(ROOT), PDMUX_DATA=str(td), PDMUX_IO=str(td),
                       PDMUX_JOB_NAME=job, PDMUX_N3_TEST_OUT_ROOT=str(td), PDMUX_N3_SHAPE=shape, **env)
        r = subprocess.run([BASH, str(script)], env=e, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr


class TestRunnerEnv(_RunnerBase):
    def test_unregistered_knob_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            rc, out = self._run(td, PDMUX_N3_DRYRUN="1", PDMUX_STICKY_PARTITION="1")
            self.assertEqual(rc, 2, out)
            self.assertIn("ABORT_ENV unregistered knob PDMUX_STICKY_PARTITION", out)

    def test_ttft_slo_knob_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            rc, out = self._run(td, PDMUX_N3_DRYRUN="1", PDMUX_TTFT_SLO_MS="3000")
            self.assertEqual(rc, 2, out)

    def test_test_knob_without_dryrun_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            rc, out = self._run(td, PDMUX_N3_TEST_HARD_CAP_S="700")
            self.assertEqual(rc, 2, out)
            self.assertIn("ABORT_TEST_KNOB_IN_MEASUREMENT", out)

    def test_shape_must_be_registered(self):
        with tempfile.TemporaryDirectory() as td:
            rc, out = self._run(td, shape="C", PDMUX_N3_DRYRUN="1")
            self.assertEqual(rc, 2, out)

    def test_inherited_phase_events_allowed_and_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            rc, out = self._run(td, PDMUX_N3_DRYRUN="1", PDMUX_N3_TEST_SKIP_PREFLIGHT="1", PDMUX_PHASE_EVENTS="1")
            self.assertEqual(rc, 0, out)
            cmds = (Path(td) / "n3b_t" / "DRYRUN_COMMANDS.txt").read_text()
            blocks = cmds.split("UNIT ")[1:]
            self.assertEqual(len(blocks), len(P.units("B")))
            for b in blocks:
                name = b.split()[0]
                kind = re.search(r"kind=(\S+)", b).group(1)
                want = "<unset>" if kind in ("bridge", "obs_off") else "1"
                self.assertIn(f"PDMUX_PHASE_EVENTS={want}", b, name)
                self.assertIn("PDMUX_STICKY_PARTITION=<unset>", b, name)
                self.assertIn("PDMUX_R2_FIXED_DSM=44", b, name)
                self.assertIn("pdmux_n3_A_agn.yml", b, name)


class TestR2R7Runner(_RunnerBase):
    def test_r2_dryrun_placeholder_never_admits_an_arm_unit(self):
        # dry-run: the floor placeholder is written, and EVERY arm unit hits the P-2/R2 floor gate
        with tempfile.TemporaryDirectory() as td:
            rc, out = self._run(td, PDMUX_N3_DRYRUN="1", PDMUX_N3_TEST_SKIP_PREFLIGHT="1")
            self.assertEqual(rc, 0, out)
            n_arm = len(P.units("B")) - 1
            self.assertEqual(out.count("DRYRUN_NOTE ABORT_FLOOR no valid T_K.json before arm unit"), n_arm)
            tk = Path(td) / "n3b_t" / "T_K.json"
            r = subprocess.run([sys.executable, "-B", str(HERE / "n3_plan.py"), "--tk-valid", str(tk)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 1, r.stdout)
            self.assertIn("dry-run placeholder", r.stdout)

    def test_r2_runner_uses_the_strict_check(self):
        src = RUNNER.read_text()
        self.assertIn('tk_valid () { python3 "$HELPER" --tk-valid "$1"', src)
        self.assertIn('python3 "$HELPER" --tk-valid "$PREV/T_K.json"', src)
        self.assertNotIn(".get('valid') else 1", src)

    def test_r7_detection_after_teardown_writes_marker(self):
        src = RUNNER.read_text()
        loop = src[src.index('for U in "${UNITS[@]}"; do\n  read -r NAME'):]
        i = loop.index('grep -q "PDMUX_PHASE_EVENTS disabled after failure" "$SRVLOG"')
        self.assertGreater(i, loop.index('timeout -k 30 "${CAP}s" "${BENCH[@]}"'))
        self.assertIn('> "$OUT/phase_events_disabled_${NAME}"', loop[i:i + 400])
        # the literal must match the engine's warning text
        mixin = (ENG / "src/multiplex/multiplexing_mixin.py").read_text()
        self.assertIn('"PDMUX_PHASE_EVENTS disabled after failure: %s"', mixin)


class TestAbortBranchesBehaviour(unittest.TestCase):
    """Behavioural (not textual) controls for the pre-boot aborts: the runner is executed on a
    minimal copy of the tree with ONE thing broken and must exit with the registered code."""

    def _tree(self, td):
        root = Path(td) / "repo"
        eng = root / "workspace/engine-port"
        shutil.copytree(HERE, eng / "results/n3_capacity",
                        ignore=shutil.ignore_patterns("n3a_*", "n3b_*", "__pycache__"))
        for rel in ("results/r2_eval/lambda0_prereg/lambda0_plan.py", P.ANALYZE_REL):
            (eng / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ENG / rel, eng / rel)
        for rel in P.MIRROR_OF.values():
            (eng / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ENG / rel, eng / rel)
        return root, eng

    def _run(self, root, td):
        e = _clean_env(PDMUX_PROJECT_ROOT=str(root), PDMUX_DATA=str(td), PDMUX_IO=str(td),
                       PDMUX_JOB_NAME="tb", PDMUX_N3_TEST_OUT_ROOT=str(td), PDMUX_N3_SHAPE="B",
                       PDMUX_N3_DRYRUN="1", PDMUX_N3_TEST_SKIP_PREFLIGHT="1")
        runner = root / "workspace/engine-port/results/n3_capacity/n3_capacity.sh"
        r = subprocess.run([BASH, str(runner)], env=e, capture_output=True, text=True, timeout=180)
        return r.returncode, r.stdout + r.stderr

    def test_unbroken_copy_runs(self):
        with tempfile.TemporaryDirectory() as td:
            root, _ = self._tree(td)
            rc, out = self._run(root, td)
            self.assertEqual(rc, 0, out[-2000:])

    def test_reg_drift_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            root, eng = self._tree(td)
            (eng / P.ANALYZE_REL).write_text((eng / P.ANALYZE_REL).read_text() + "\n# drift\n")
            rc, out = self._run(root, td)
            self.assertEqual(rc, 3, out[-2000:])
            self.assertIn("ABORT_REG_DRIFT", out)

    def test_mirror_drift_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            root, eng = self._tree(td)
            m = eng / "src/multiplex/multiplexing_mixin.py"
            m.write_text(m.read_text() + "\n# drift\n")
            rc, out = self._run(root, td)
            self.assertEqual(rc, 3, out[-2000:])
            self.assertIn("ABORT_REG_DRIFT", out)

    def test_config_without_fixed_dsm_aborts(self):
        # re-register the digest of a config whose (64,44) row became (54,54): the digest gate passes,
        # so only the FIXED_DSM membership check can stop it
        with tempfile.TemporaryDirectory() as td:
            root, eng = self._tree(td)
            cfg = eng / P.CONFIG_REL
            cfg.write_text(cfg.read_text().replace("[64, 44, 36]", "[54, 54, 36]"))
            plan = eng / "results/n3_capacity/n3_plan.py"
            plan.write_text(plan.read_text().replace(P.CONFIG_DIGEST, P.sha256(cfg)))
            rc, out = self._run(root, td)
            self.assertEqual(rc, 3, out[-2000:])
            self.assertIn("ABORT_CONFIG", out)
            self.assertIn("PDMUX_R2_FIXED_DSM=44 not in", out)

    def test_missing_prereg_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            root, eng = self._tree(td)
            (eng / "results/n3_capacity/PREREG_N3_CAPACITY_2026-10-04.md").unlink()
            rc, out = self._run(root, td)
            self.assertEqual(rc, 3, out[-2000:])
            self.assertIn("ABORT_REG pre-registration missing", out)


class TestBudgetEnforcer(_RunnerBase):
    """cap 620 s - tail 600 s => deadline ~20 s after start; the first fake bench sleeps 60 s."""
    CAP, FAKE, LIMIT = "620", "60", 150

    def _go(self, script):
        with tempfile.TemporaryDirectory() as td:
            t0 = time.monotonic()
            try:
                rc, out = self._run(td, script=script, timeout=self.LIMIT, PDMUX_N3_DRYRUN="1",
                                    PDMUX_N3_TEST_SKIP_PREFLIGHT="1", PDMUX_N3_TEST_HARD_CAP_S=self.CAP,
                                    PDMUX_N3_TEST_FAKE_BENCH_S=self.FAKE)
            except subprocess.TimeoutExpired:
                subprocess.run(["pkill", "-f", "n3_fake_bench"], check=False)
                subprocess.run(["pkill", "-f", "sleep 100000"], check=False)
                return None, time.monotonic() - t0, ""
            fake = (Path(td) / "n3b_t" / "FAKE_BENCH.txt")
            return rc, time.monotonic() - t0, fake.read_text() if fake.exists() else ""

    def test_overrunning_unit_is_cut(self):
        rc, wall, fake = self._go(RUNNER)
        self.assertEqual(rc, 0)
        self.assertLess(wall, 50.0)
        self.assertRegex(fake, r"fake_rc=(124|143|137) unit=floor_head")
        self.assertEqual(fake.count("fake_rc="), 1, fake)

    def test_control_both_kill_layers_removed_overruns(self):
        m = RUNNER.read_text().replace('timeout -k 2 "${BT}s" bash -c', 'bash -c')
        m = m.replace('  while [ "$(remaining)" -gt 0 ]; do sleep 5; done\n', '  sleep 100000\n')
        self.assertEqual(m.count("sleep 100000"), 1)
        with tempfile.TemporaryDirectory() as t2:
            mut = Path(t2) / "n3_mutant.sh"
            mut.write_text(m)
            rc, wall, _ = self._go(mut)
        self.assertGreaterEqual(wall, float(self.FAKE), "mutant still cut -- the test would be vacuous")

    def test_equivalent_mutant_watchdog_removed_still_cuts(self):
        m = RUNNER.read_text().replace('  while [ "$(remaining)" -gt 0 ]; do sleep 5; done\n', '  sleep 100000\n')
        with tempfile.TemporaryDirectory() as t2:
            mut = Path(t2) / "n3_mutant.sh"
            mut.write_text(m)
            rc, wall, _ = self._go(mut)
        self.assertEqual(rc, 0)
        self.assertLess(wall, 50.0)


class TestResume(_RunnerBase):
    def _prev(self, td, registration, tk_valid=True):
        p = Path(td) / "data/runs/prevjob/results/n3_capacity/n3b_prevjob"
        p.mkdir(parents=True)
        (Path(td) / "data/runs/prevjob/meta").mkdir(parents=True)
        (p / "UNITS.txt").write_text("\n".join(P.units("B")) + "\n")
        (p / "REGISTRATION_SHA256.txt").write_text(registration)
        tk = {"valid": tk_valid, "floor_s": 0.9, "ttfts_s": [0.9] * P.FLOOR_N,
              "T_k_s": {"3": 0.9 * 3, "5": 0.9 * 5, "10": 0.9 * 10}}
        if tk_valid == "dryrun":
            tk = {"valid": True, "dryrun": True, "floor_s": 1.0, "T_k_s": {"3": 3.0, "5": 5.0, "10": 10.0}}
        (p / "T_K.json").write_text(json.dumps(tk))
        (p / "bench_br_b_r3_4386.jsonl").write_text("{}\n")
        (p / "unit_br_b_r3_4386.complete").write_text("x\n")
        (Path(td) / "data/runs/prevjob/meta/substrate.json").write_text(
            '{"gpu": {"uuid": "GPU-prev", "driver_version": "580.105.08"}}')

    def _go(self, td, job):
        e = _clean_env(PDMUX_PROJECT_ROOT=str(ROOT), PDMUX_DATA=f"{td}/data", PDMUX_IO=f"{td}/io",
                       PDMUX_JOB_NAME=job, PDMUX_N3_TEST_OUT_ROOT=f"{td}/out", PDMUX_N3_SHAPE="B",
                       PDMUX_N3_DRYRUN="1", PDMUX_N3_TEST_SKIP_PREFLIGHT="1", PDMUX_N3_RESUME_FROM="prevjob")
        r = subprocess.run([BASH, str(RUNNER)], env=e, capture_output=True, text=True, timeout=300)
        return r.returncode, r.stdout + r.stderr, Path(td) / "out" / f"n3b_{job}"

    def test_resume_and_its_controls(self):
        with tempfile.TemporaryDirectory() as td:
            self._prev(td, "0" * 64 + "  n3_plan.py\n")
            rc, out, _ = self._go(td, "j1")
            self.assertEqual(rc, 2, out)
            self.assertIn("ABORT_RESUME registration digests differ", out)
            reg = (Path(td) / "out" / "n3b_j1" / "REGISTRATION_SHA256.txt").read_text()
            shutil.rmtree(Path(td) / "data/runs/prevjob")
            self._prev(td, reg, tk_valid=False)
            rc, out, _ = self._go(td, "j2")
            self.assertEqual(rc, 2, out)
            self.assertIn("no valid T_K.json", out)
            shutil.rmtree(Path(td) / "data/runs/prevjob")
            self._prev(td, reg, tk_valid="dryrun")        # R2: a dry-run Job's placeholder T_K
            rc, out, _ = self._go(td, "j2b")
            self.assertEqual(rc, 2, out)
            self.assertIn("dry-run placeholder", out)
            shutil.rmtree(Path(td) / "data/runs/prevjob")
            self._prev(td, reg)
            rc, out, o = self._go(td, "j3")
            self.assertEqual(rc, 0, out)
            self.assertIn("SKIP_DONE unit=br_b_r3_4386", out)
            self.assertIn("br_b_r3_4386\tprevjob\tGPU-prev", (o / "RESUMED_UNITS.tsv").read_text())
            self.assertEqual(json.loads((o / "T_K.json").read_text())["floor_s"], 0.9)


# =============================================================================================
# Harness-layer audit 2026-10-05 (VERDICT_n3_harness_2026-10-05.md): H-1 repair + N3C-18 shrink.
# Every test below cuts the REAL runner text (no copy of the branch) and executes it with stubs.
# =============================================================================================
PROBE = HERE / "probe_6group_2026-10-05"
CFG = ENG / P.CONFIG_REL


def _line(src, prefix):
    """the runner's own one-line definition starting with `prefix` (fail, dry_or_abort, ...)"""
    return next(ln for ln in src.splitlines() if ln.startswith(prefix))


def _cut(src, start, end):
    """runner text from `start` through the first `end` after it (both literal)"""
    a = src.index(start)
    b = src.index(end, a) + len(end)
    return src[a:b]


def _neutralise(snippet, abort):
    """`fail "ABORT_X` / `dry_or_abort "ABORT_X` -> `echo "ABORT_X` (text kept, abort gone)"""
    m = re.sub(r'(?:\bfail|\bdry_or_abort) ("' + re.escape(abort) + r'[ "])', r'echo \1', snippet)
    assert m != snippet, f"neutralisation did not apply for {abort}"
    return m


class _SnippetBase(unittest.TestCase):
    """Runs a cut of the runner under `set -uo pipefail` with the runner's real `fail`/`dry_or_abort`
    and a recording `teardown` stub.  The script is written to a FILE (not `bash -c`) so that a
    `pkill -f` inside it can never match the test's own command line."""

    def setUp(self):
        self.src = RUNNER.read_text()

    def _exec(self, td, body, *, defs=(), env_vars=None, timeout=120):
        v = {"OUT": td, "NAME": "X", "SRVLOG": f"{td}/srv.log", "HELPER": str(HERE / "n3_plan.py"),
             "CFG": str(CFG), "PID": "1", "PORT": "1", "JOBID": "tj", "SHAPE": "B", "DRYRUN": "0"}
        v.update(env_vars or {})
        pre = "".join(f'{k}="{val}"\n' for k, val in v.items())
        script = ("set -uo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n" + pre +
                  'teardown () { echo TEARDOWN >> "$OUT/td.txt"; }\n' +
                  _line(self.src, "fail () {") + "\n" + _line(self.src, "dry_or_abort () {") + "\n" +
                  "".join(d + "\n" for d in defs) + body + "\necho SNIPPET_PASSED\n")
        sp = Path(td) / "snippet.sh"
        sp.write_text(script)
        t0 = time.monotonic()
        r = subprocess.run([BASH, str(sp)], capture_output=True, text=True, timeout=timeout, env=_clean_env())
        return r.returncode, r.stdout + r.stderr, (Path(td) / "td.txt").exists(), time.monotonic() - t0


class TestCG2Snippet(_SnippetBase):
    """T-H6 (required, lesson 53): H-1 repair of the CG2 fail-open.  The runner's REAL CG2 block is
    executed on the REAL probe warm-up output (`warmup_A2_warm.jsonl`, 8 greedy texts).
      8/8 identical -> rc 0 and CG2_DONE=1;  1 text differs -> 9;  ON input absent -> 9;  ON empty -> 9.
    Control: the REVERTED condition (pre-repair `&& [ -s "$OUT/warmup_floor_head.jsonl" ]`, HEAD
    3604269) turns "ON absent" back into rc 0 / CG2_DONE=0 -- so the "absent -> 9" assertion fails on
    the broken variant and is not an identity."""
    START = '  if [ "$PEV" = 0 ] && [ "$CG2_DONE" = 0 ]'
    END = "    CG2_DONE=1\n  fi\n"
    OLD_COND = '  if [ "$PEV" = 0 ] && [ "$CG2_DONE" = 0 ] && [ -s "$OUT/warmup_floor_head.jsonl" ]; then\n'

    def _reverted(self, src):
        m = re.sub(r'  if \[ "\$PEV" = 0 \] && \[ "\$CG2_DONE" = 0 \]; then\n'
                   r'    if \[ ! -s "\$OUT/warmup_floor_head\.jsonl" \]; then\n.*?\n    fi\n',
                   lambda _m: self.OLD_COND, src, count=1, flags=re.S)
        self.assertNotEqual(m, src, "revert mutation did not apply")
        return m

    def _go(self, snippet, on, off):
        """on/off: None = file absent, "" = empty file, list = generated_texts"""
        with tempfile.TemporaryDirectory() as td:
            base = json.loads((PROBE / "warmup_A2_warm.jsonl").read_text().splitlines()[0])
            for fn, texts in (("warmup_floor_head.jsonl", on), ("warmup_X.jsonl", off)):
                if texts is None:
                    continue
                (Path(td) / fn).write_text("" if texts == "" else json.dumps(dict(base, generated_texts=texts)) + "\n")
            rc, out, td_called, _ = self._exec(td, snippet + 'echo "CG2_DONE=$CG2_DONE"\n',
                                               env_vars={"PEV": "0", "CG2_DONE": "0"})
            cg2 = (Path(td) / "CG2.txt").read_text() if (Path(td) / "CG2.txt").exists() else ""
            return rc, out, td_called, cg2

    def test_t_h6_repaired_block(self):
        snip = _cut(self.src, self.START, self.END)
        self.assertNotIn('] && [ -s "$OUT/warmup_floor_head.jsonl" ]', snip)
        texts = json.loads((PROBE / "warmup_A2_warm.jsonl").read_text().splitlines()[0])["generated_texts"]
        self.assertEqual(len(texts), 8)
        rc, out, td, cg2 = self._go(snip, texts, texts)
        self.assertEqual(rc, 0, out)
        self.assertIn("CG2_DONE=1", out)
        self.assertIn("CG2_PASS on=8 off=8 identical=True", cg2)
        self.assertFalse(td)
        one_off = list(texts)
        one_off[3] = one_off[3] + " "
        rc, out, td, cg2 = self._go(snip, texts, one_off)
        self.assertEqual(rc, 9, out)
        self.assertIn("ABORT_CG2 PHASE_EVENTS ON vs OFF", out)
        self.assertIn("CG2_FAIL", cg2)
        self.assertTrue(td)
        for name, on in (("ON absent", None), ("ON empty", "")):
            rc, out, td, cg2 = self._go(snip, on, texts)
            self.assertEqual(rc, 9, (name, out))
            self.assertIn("ABORT_CG2 ON-side input warmup_floor_head.jsonl missing/empty -- CG2 not evaluable "
                          "(measurement failure, not evidence of an observer effect)", out, name)
            self.assertTrue(td, f"{name}: server must be torn down before exit")
            self.assertNotIn("CG2_DONE=", out, name)
            self.assertEqual(cg2, "", name)

    def test_t_h6_control_reverted_condition_is_fail_open(self):
        mut = self._reverted(self.src)
        snip = _cut(mut, self.START, self.END)
        self.assertTrue(snip.startswith(self.OLD_COND), "mutant must carry the pre-repair condition")
        texts = json.loads((PROBE / "warmup_A2_warm.jsonl").read_text().splitlines()[0])["generated_texts"]
        for on in (None, ""):
            rc, out, td, cg2 = self._go(snip, on, texts)
            # the broken variant passes silently: the repaired assertion (rc 9) would FAIL here
            self.assertEqual(rc, 0, out)
            self.assertIn("CG2_DONE=0", out)
            self.assertNotEqual(rc, 9)
            self.assertFalse(td)
        rc, out, *_ = self._go(snip, texts, texts)          # mutant is otherwise the same gate
        self.assertEqual(rc, 0, out)
        self.assertIn("CG2_DONE=1", out)


class TestBootBranchSnippets(_SnippetBase):
    """N3C-18 shrink (harness verdict §11, LOCAL gates only -- not in the Job preflight list §8-1).
    Each boot-time abort branch is cut from the REAL runner and executed on stub/probe inputs, with a
    neutralised-branch control (lesson 53).

    Equivalent / partially equivalent mutants (lesson 254, disclosed, not hidden):
      E-1 (T-H5) floor boot failure: deleting `if [ "$KIND" = floor ]; then fail "ABORT_FLOOR floor boot
          failed ...` is EQUIVALENT at Job level -- the next arm unit hits the P-2 T_K gate and exits 7
          (different message, same exit code, no arm cell).  Detectable only on a floor-only unit list.
      E-2 (T-H1) STICKY: neutralising the srv-log STICKY check is PARTIALLY equivalent -- the GREEN read-out
          exits 6 if (and only if) the read-out carries a non-null `sticky_target_index`.  Shown here on a
          stub read-out; that the engine's read-out actually carries the index when sticky is ON is
          NOT verified on GPU.
      E-3 (T-H7) hard cap: removing ONLY the bench `timeout` is EQUIVALENT under the runner's watchdog
          (the watchdog TERMs the bench -> rc 143 + BUDGET_HARDCAP_FIRED -> `budget_killed_<unit>` is still
          written).  Without the watchdog the same mutant is NOT cut (shown).  "Admission only removed"
          stays textual (TestFailClosedWiring.test_control_removed_admission), not executed here.
    Inputs are stubs or Workspace probe files; none of this shows that the branches fire on a real
    server's output (N3H-1 / N3H-6 scope)."""

    # ----------------------------------------------------------------------------- T-H1 STICKY
    STICKY = ('  if grep -q "sticky partition ENABLED" "$SRVLOG"; then\n', "  fi\n")

    def _engine_sticky_line(self):
        mixin = (ENG / "src/multiplex/multiplexing_mixin.py").read_text()
        m = re.search(r'"(PD-mux sticky partition ENABLED: [^"]*)"', mixin)
        self.assertIsNotNone(m, "engine literal not found -- runner grep would be untethered")
        return f"[2026-10-05 00:00:00] {m.group(1)}decode is busy (fixed target index=4); ...\n"

    def _sticky(self, snippet, log):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "srv.log").write_text(log)
            return self._exec(td, snippet)

    def test_t_h1_sticky(self):
        snip = _cut(self.src, *self.STICKY)
        probe_log = (PROBE / "srv_A2_warm.log").read_text(errors="replace")
        rc, out, td, _ = self._sticky(snip, probe_log)
        self.assertEqual(rc, 0, out)
        self.assertFalse(td)
        rc, out, td, _ = self._sticky(snip, probe_log + self._engine_sticky_line())
        self.assertEqual(rc, 6, out)
        self.assertIn("ABORT_STICKY", out)
        self.assertTrue(td)
        rc, out, *_ = self._sticky(_neutralise(snip, "ABORT_STICKY"), probe_log + self._engine_sticky_line())
        self.assertEqual(rc, 0, "neutralised STICKY still aborted -- the test is vacuous")

    # ------------------------------------------------------------------------------ T-H2 GREEN
    GREEN = ('  python3 "$HELPER" --check-green', "6; }\n")

    def _green(self, snippet, green):
        with tempfile.TemporaryDirectory() as td:
            shutil.copy(green, Path(td) / "green_X.json")
            return self._exec(td, snippet)

    def test_t_h2_green(self):
        snip = _cut(self.src, *self.GREEN)
        self.assertIn("ABORT_GREEN", snip)
        rc, out, td, _ = self._green(snip, PROBE / "green_A2_warm.json")
        self.assertEqual(rc, 0, out)
        self.assertFalse(td)
        rc, out, td, _ = self._green(snip, PROBE / "green_H5.json")
        self.assertEqual(rc, 6, out)
        self.assertIn("ABORT_GREEN", out)
        self.assertTrue(td)
        with tempfile.TemporaryDirectory() as t2:
            g = json.loads((PROBE / "green_A2_warm.json").read_text())
            self.assertIsNone(g["sticky_target_index"])
            g["sticky_target_index"] = 4
            sticky = Path(t2) / "g.json"
            sticky.write_text(json.dumps(g))
            rc, out, td, _ = self._green(snip, sticky)
            self.assertEqual(rc, 6, out)
            # E-2: STICKY neutralised + engine line in the log; GREEN (sticky_target_index 4) still exits 6
            both = _neutralise(_cut(self.src, *self.STICKY), "ABORT_STICKY") + snip
            with tempfile.TemporaryDirectory() as t3:
                shutil.copy(sticky, Path(t3) / "green_X.json")
                (Path(t3) / "srv.log").write_text(self._engine_sticky_line())
                rc, out, *_ = self._exec(t3, both)
                self.assertEqual(rc, 6, out)
                self.assertIn("ABORT_GREEN", out)
        rc, out, *_ = self._green(_neutralise(snip, "ABORT_GREEN"), PROBE / "green_H5.json")
        self.assertEqual(rc, 0, "neutralised GREEN still aborted -- the test is vacuous")

    # ------------------------------------------------------------------------------ T-H3 floor
    FLOOR = ('  if [ "$KIND" = floor ] || [ "$KIND" = floor_tail ]; then\n    # P-2',
             '  if budget_fired; then STOP=budget')

    def _floor_inputs(self, td, mutate=None):
        """bench: 16 completed concurrency-1 B requests; telemetry (rev6): the probe's REAL
        prefill_span_start/prefill_span_end pairs (all entry (108,0,0,1,0), end stream 0), replayed to
        FLOOR_N + 9 spans 2 s apart, each followed by its decode iterations on the plain decode group
        (outside the span).  `mutate(ctx)` gets ctx = {start, end, events} of the LAST span."""
        tt = [0.9 + 0.001 * i for i in range(P.FLOOR_N)]
        (Path(td) / "bench_X.jsonl").write_text(json.dumps(
            {"completed": P.FLOOR_N, "errors": [""] * P.FLOOR_N, "ttfts": tt,
             "total_input_tokens": P.FLOOR_N * P.SHAPES["B"]["input_len"], "max_concurrency": 1,
             "duration": 2.0 * (P.FLOOR_N - 1) + 0.9 + 0.5}) + "\n")     # W-5 (default length): span window + 0.5 s decode tail
        recs = [json.loads(ln) for ln in (PROBE / "tel_A2_warm.jsonl").read_text().splitlines() if ln.strip()]
        st_real = [e for e in recs if e["event"] == "prefill_span_start"]
        en_real = {e["span_seq"]: e for e in recs if e["event"] == "prefill_span_end"}
        self.assertTrue(st_real and all((e["prefill_sms"], e["decode_sms"], e["running_bs"], e["n_new"],
                                         e["queue_len"]) == (108, 0, 0, 1, 0) for e in st_real))
        self.assertTrue(all(en_real[e["span_seq"]]["stream_idx"] == P.FLOOR_STREAM_IDX for e in st_real))
        dec_real = next(e for e in recs if e["event"] == "decode_iteration")
        events, ctx = [], None
        for i in range(P.FLOOR_N + 9):
            src = st_real[i % len(st_real)]
            t0 = 100.0 + 2.0 * i
            st = dict(src, span_seq=i + 1, timestamp_monotonic_s=t0, dropped_events=0)
            en = dict(en_real[src["span_seq"]], span_seq=i + 1, timestamp_monotonic_s=t0 + 0.9, dropped_events=0)
            events += [st, en]
            events += [dict(dec_real, iter_seq=5 * i + j, timestamp_monotonic_s=t0 + 1.0 + 0.1 * j, dropped_events=0)
                       for j in range(5)]
            ctx = {"start": st, "end": en, "events": events}
        if mutate:
            mutate(ctx)
        (Path(td) / "tel_X.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))

    def _floor(self, snippet, mutate=None, prior_tk=None, r7_marker=False):
        with tempfile.TemporaryDirectory() as td:
            self._floor_inputs(td, mutate)
            if prior_tk is not None:
                (Path(td) / "T_K.json").write_text(prior_tk)
            if r7_marker:   # the runner's R7 detection file (PHASE_EVENTS disabled mid-boot)
                (Path(td) / "phase_events_disabled_X").write_text("PDMUX_PHASE_EVENTS disabled after failure\n")
            rc, out, td_called, _ = self._exec(td, snippet, env_vars={
                "KIND": "floor", "CUT": "", "PDMUX_TELEMETRY_PATH": f"{td}/tel_X.jsonl"})
            files = {f.name: f.read_text() for f in Path(td).iterdir() if f.is_file()}
            return rc, out, files

    # rev6 mutations on the LAST span: realised off stream 0 / a decode iteration inside the span
    @staticmethod
    def _mut_end_stream(ctx):
        ctx["end"]["stream_idx"] = 4

    @staticmethod
    def _mut_decode_inside(ctx):
        ctx["events"].append(dict(ctx["events"][-1], iter_seq=9999, stream_idx=3, prefill_sms=64, decode_sms=44,
                                  timestamp_monotonic_s=ctx["start"]["timestamp_monotonic_s"] + 0.4))

    @staticmethod
    def _mut_entry_race(ctx):
        # Job 79999aa span 6: entry label written before adjust_stream_groups; realised stream 0
        ctx["start"].update(prefill_sms=0, decode_sms=108, stream_idx=5, running_bs=1)

    def test_t_h3_floor(self):
        snip = _cut(self.src, *self.FLOOR)[:-len(self.FLOOR[1])]
        rc, out, f = self._floor(snip)
        self.assertEqual(rc, 0, out)
        tk = json.loads(f["T_K.json"])
        self.assertEqual(P.tk_problems(tk), [], tk)
        self.assertIn("unit_X.complete", f)
        for name, mut in (("last span realised off stream 0", self._mut_end_stream),
                          ("decode_iteration inside the last span", self._mut_decode_inside)):
            rc, out, f = self._floor(snip, mut)
            self.assertEqual(rc, 7, (name, out))
            self.assertIn("ABORT_FLOOR floor invalid", out)
            self.assertFalse(json.loads(f["T_K.json"])["valid"], name)
            self.assertNotIn("unit_X.complete", f, name)
        # rev6 positive control: the entry-label race (rev5 ABORT_FLOOR, Job 79999aa) is a VALID floor
        rc, out, f = self._floor(snip, self._mut_entry_race)
        self.assertEqual(rc, 0, out)
        tk = json.loads(f["T_K.json"])
        self.assertTrue(tk["valid"] and tk["entry_label_off_count"] == 1, tk)
        self.assertIn("unit_X.complete", f)
        # resume path: a T_K.json already exists (copied from the first Job) -> re-measured floor goes
        # to T_K_RESUME_<job>.json, never overwrites T_K.json and never aborts (diagnostic)
        prior = json.dumps({"frozen": "first-job"})
        rc, out, f = self._floor(snip, self._mut_end_stream, prior_tk=prior)
        self.assertEqual(rc, 0, out)
        self.assertNotIn("ABORT_FLOOR", out)
        self.assertEqual(f["T_K.json"], prior)
        self.assertFalse(json.loads(f["T_K_RESUME_tj.json"])["valid"])
        rc, out, f = self._floor(snip, prior_tk=prior)
        self.assertEqual(rc, 0, out)
        self.assertTrue(json.loads(f["T_K_RESUME_tj.json"])["valid"])
        self.assertEqual(f["T_K.json"], prior)
        rc, out, _ = self._floor(_neutralise(snip, "ABORT_FLOOR"), self._mut_end_stream)
        self.assertEqual(rc, 0, "neutralised floor abort still exited -- the test is vacuous")
        # rev6 audit R-1: R7 marker (PHASE_EVENTS disabled mid-boot) => ABORT_FLOOR even though floor_from
        # says valid (floor spans unobserved); the diagnostic T_K.json is still written
        rc, out, f = self._floor(snip, r7_marker=True)
        self.assertEqual(rc, 7, out)
        self.assertIn("ABORT_FLOOR PHASE_EVENTS disabled mid-boot", out)
        self.assertTrue(json.loads(f["T_K.json"])["valid"], "floor_from itself saw a valid boot (the abort is R-1's)")
        self.assertNotIn("unit_X.complete", f)
        rc, out, _ = self._floor(_neutralise(snip, "ABORT_FLOOR PHASE_EVENTS"), r7_marker=True)
        self.assertEqual(rc, 0, "neutralised R-1 abort still exited -- the test is vacuous")
        # R7 marker on the RESUME path (TKOUT != T_K.json) is diagnostic only: no abort
        rc, out, f = self._floor(snip, prior_tk=prior, r7_marker=True)
        self.assertEqual(rc, 0, out)
        self.assertEqual(f["T_K.json"], prior)

    # ------------------------------------------------------------------------- T-H4 T_K gate
    TKGATE = ('  if [ "$KIND" != floor ] && ! tk_valid "$OUT/T_K.json"; then\n', "  fi\n")

    def _tk(self, snippet, kind, dryrun="0", tk=None):
        with tempfile.TemporaryDirectory() as td:
            if tk is not None:
                (Path(td) / "T_K.json").write_text(json.dumps(tk))
            return self._exec(td, snippet, defs=(_line(self.src, "tk_valid () {"),),
                              env_vars={"KIND": kind, "DRYRUN": dryrun})

    def test_t_h4_tk_gate_non_dryrun(self):
        snip = _cut(self.src, *self.TKGATE)
        good = {"valid": True, "floor_s": 0.9, "ttfts_s": [0.9] * P.FLOOR_N,
                "T_k_s": {str(k): 0.9 * k for k in P.K_FLOOR}}
        rc, out, *_ = self._tk(snip, "grid")
        self.assertEqual(rc, 7, out)
        self.assertIn("ABORT_FLOOR no valid T_K.json before arm unit X", out)
        rc, out, *_ = self._tk(snip, "grid", tk={"valid": True, "dryrun": True, "floor_s": 1.0,
                                                  "T_k_s": {"3": 3.0, "5": 5.0, "10": 10.0}})
        self.assertEqual(rc, 7, "dry-run placeholder must not admit an arm unit outside DRYRUN")
        rc, out, *_ = self._tk(snip, "grid", tk=good)
        self.assertEqual(rc, 0, out)
        rc, out, *_ = self._tk(snip, "floor")
        self.assertEqual(rc, 0, out)
        rc, out, *_ = self._tk(snip, "grid", dryrun="1")              # control: DRYRUN only notes
        self.assertEqual(rc, 0, out)
        self.assertIn("DRYRUN_NOTE ABORT_FLOOR", out)
        rc, out, *_ = self._tk(_neutralise(snip, "ABORT_FLOOR"), "grid")
        self.assertEqual(rc, 0, "neutralised T_K gate still aborted -- the test is vacuous")

    # --------------------------------------------------------------- T-H5 floor boot failure
    BOOT = ('  if ! wait_health "$PORT" "$PID"; then\n', "    continue\n  fi\n")
    FLOOR_BOOT_LINE = '    if [ "$KIND" = floor ]; then fail "ABORT_FLOOR floor boot failed'

    def _boot(self, boot_snip, units):
        body = ('for U in ' + " ".join(f'"{u}"' for u in units) + '; do\n  read -r NAME KIND <<< "$U"\n'
                + _cut(self.src, *self.TKGATE) + boot_snip + "done\n")
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "srv.log").write_text("boom\n")
            return self._exec(td, body, defs=(_line(self.src, "tk_valid () {"),
                                              _line(self.src, "budget_fired () {"),
                                              "wait_health () { return 1; }"),
                              env_vars={"FAIL_STREAK": "0", "STOP": "", "T_ADMIT": "0",
                                        "SRVLOG": "$OUT/srv.log"})

    def test_t_h5_floor_boot_failure_and_equivalent_mutant(self):
        snip = _cut(self.src, *self.BOOT)
        self.assertIn(self.FLOOR_BOOT_LINE, snip)
        rc, out, td, _ = self._boot(snip, ["floor_head floor"])
        self.assertEqual(rc, 7, out)
        self.assertIn("ABORT_FLOOR floor boot failed", out)
        self.assertTrue(td)
        rc, out, *_ = self._boot(snip, ["br grid"])                  # an arm boot failure is not an abort
        self.assertIn("ABORT_FLOOR no valid T_K.json", out)          # (it is stopped earlier by the T_K gate)
        mut = re.sub(r'    if \[ "\$KIND" = floor \]; then fail "ABORT_FLOOR floor boot failed[^\n]*\n', "", snip)
        self.assertNotEqual(mut, snip)
        # control: on a floor-only unit list the mutant no longer aborts (the test is not vacuous) ...
        rc, out, *_ = self._boot(mut, ["floor_head floor"])
        self.assertEqual(rc, 0, out)
        self.assertIn("BOOT_FAILED unit=floor_head", out)
        # ... E-1: on the registered order (floor then an arm unit) the mutant still exits 7 via the T_K gate
        rc, out, *_ = self._boot(mut, ["floor_head floor", "br_b_r3_4386 bridge"])
        self.assertEqual(rc, 7, out)
        self.assertNotIn("ABORT_FLOOR floor boot failed", out)
        self.assertIn("ABORT_FLOOR no valid T_K.json before arm unit br_b_r3_4386", out)

    # --------------------------------------------------------------- T-H7 bench timeout / hard cap
    BENCHCUT = ('  echo "bench_start $(mono)"',
                '    echo "BENCH_KILLED_BY_BUDGET unit=$NAME rc=$BRC" | tee -a "$OUT/BUDGET_SKIPPED.txt"\n  fi\n')
    TIMEOUT_LINE = 'timeout -k 30 "${CAP}s" "${BENCH[@]}"'

    def _watchdog(self, src):
        return _cut(src, '(\n  while [ "$(remaining)" -gt 0 ]; do sleep 5; done\n', "WD_PID=$!\n")

    def _bench(self, snippet, sleep_s, deadline_in_s, watchdog=False):
        tag = f"n3_fake_bench_th7_{os.getpid()}"
        body = (f"DEADLINE=$(( $(date +%s) + {deadline_in_s} ))\n"
                f"BENCH=(bash -c 'exec -a {tag} sleep \"$1\"' _ {sleep_s})\n"
                + (self._watchdog(self.src) if watchdog else "WD_PID=\n")
                + snippet + 'echo "BRC=$BRC"\n[ -n "$WD_PID" ] && kill "$WD_PID" 2>/dev/null\n')
        with tempfile.TemporaryDirectory() as td:
            try:
                rc, out, _, wall = self._exec(
                    td, body, timeout=90,
                    defs=(_line(self.src, "remaining () {"), _line(self.src, "cap_s () {"),
                          _line(self.src, "budget_fired () {"), "mono () { echo 0.0; }", "cpu_snap () { :; }"),
                    env_vars={"RETR0": "0"})
            finally:
                subprocess.run(["pkill", "-f", tag], check=False)
            marker = (Path(td) / "budget_killed_X").exists()
            return rc, out, marker, wall

    def test_t_h7_bench_timeout_writes_marker(self):
        snip = _cut(self.src, *self.BENCHCUT)
        self.assertIn(self.TIMEOUT_LINE, snip)
        rc, out, marker, wall = self._bench(snip, 30, 2)
        self.assertEqual(rc, 0, out)
        self.assertIn("BRC=124", out)
        self.assertIn("BENCH_KILLED_BY_BUDGET unit=X rc=124", out)
        self.assertTrue(marker)
        self.assertLess(wall, 20.0)
        rc, out, marker, _ = self._bench(snip, 0, 60)                # control: a bench that finishes
        self.assertIn("BRC=0", out)
        self.assertFalse(marker, "marker must not be unconditional")

    def test_t_h7_timeout_only_removed(self):
        mut = _cut(self.src, *self.BENCHCUT).replace(self.TIMEOUT_LINE, '"${BENCH[@]}"')
        self.assertNotIn("timeout -k 30", mut)
        # without the watchdog the mutant is NOT cut: the bench runs to its end, no marker
        rc, out, marker, wall = self._bench(mut, 6, 2)
        self.assertIn("BRC=0", out)
        self.assertFalse(marker)
        self.assertGreaterEqual(wall, 6.0)
        # E-3: with the runner's REAL watchdog the mutant is still cut (rc 143) and the marker written
        rc, out, marker, wall = self._bench(mut, 40, 2, watchdog=True)
        self.assertIn("BRC=143", out)
        self.assertTrue(marker, "watchdog-only path must still write budget_killed_<unit>")
        self.assertLess(wall, 30.0)


class TestFloorRev6RealJob(unittest.TestCase):
    """rev6 on the REAL shape-B 2nd-submission floor boot (Job pdmux-n3-capacity-79999aa-20261006081638,
    exit 7 ABORT_FLOOR under rev5): 28 spans, span 6 entry label (0,108)/running_bs 1, every
    prefill_span_end on stream 0, no decode_iteration inside any span.  rev6 must accept it and the rev5
    predicate (every prefill_span_start at (108,0)) must reject it -- otherwise rev6 changed nothing.
    The bench/telemetry jsonl are git-ignored: skipped when absent."""
    JOB = HERE / "n3b_pdmux-n3-capacity-79999aa-20261006081638"

    def setUp(self):
        self.bench = self.JOB / "bench_floor_head.jsonl"
        self.tel = self.JOB / "tel_floor_head.jsonl"
        if not (self.bench.exists() and self.tel.exists()):
            self.skipTest("Job 79999aa floor jsonl absent (git-ignored raw data)")

    def test_rev6_accepts_the_job_floor_that_rev5_rejected(self):
        f = P.floor_from(self.bench, self.tel, "B")
        self.assertTrue(f["valid"], f)
        self.assertEqual(f["n_spans"], 28)
        self.assertEqual(f["entry_label_off_count"], 1, f)       # the race, descriptive only
        self.assertEqual(f["end_stream_idx_set"], [0], f)
        self.assertEqual(P.tk_problems(f), [])
        # audit R-2/R-3 on the real boot: recorder live after all 16 floor spans, race count 0 in the floor
        # window (the single race was warm-up span 6), 16 admission gaps all present
        self.assertEqual(f["floor_entry_race_count"], 0, f)
        self.assertEqual(len(f["floor_admission_gap_ms"]), P.FLOOR_N)
        self.assertTrue(all(g is not None and g > 0 for g in f["floor_admission_gap_ms"]), f["floor_admission_gap_ms"])

    def test_rev6_r2_rejects_the_job_data_without_decode_after_a_floor_span(self):
        evs = [json.loads(ln) for ln in self.tel.read_text().splitlines() if ln.strip()]
        last_start = [e for e in evs if e["event"] == "prefill_span_start"][-1]["timestamp_monotonic_s"]
        evs = [e for e in evs if not (e["event"] == "decode_iteration" and e["timestamp_monotonic_s"] > last_start)]
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td) / "t.jsonl"
            tp.write_text("".join(json.dumps(e) + "\n" for e in evs))
            f = P.floor_from(self.bench, tp, "B")
        self.assertFalse(f["valid"], f)
        self.assertTrue(any("not shown live" in p for p in f["problems"]), f)
        # the sealed rev5 T_K.json of that Job (tracked) was invalid for exactly this reason
        tk5 = json.loads((self.JOB / "T_K.json").read_text())
        self.assertFalse(tk5["valid"])
        self.assertEqual(tk5["problems"], ["1 prefill spans not at (108, 0)"])
        starts = [json.loads(ln) for ln in self.tel.read_text().splitlines()
                  if ln.strip() and '"prefill_span_start"' in ln]
        self.assertTrue(any((e["prefill_sms"], e["decode_sms"]) != P.FLOOR_PAIR for e in starts),
                        "rev5 predicate does not fire on this boot -- the test is vacuous")

    def test_rev6_still_rejects_a_realised_violation_on_the_job_data(self):
        evs = [json.loads(ln) for ln in self.tel.read_text().splitlines() if ln.strip()]
        ends = [e for e in evs if e["event"] == "prefill_span_end"]
        ends[-1]["stream_idx"] = 3
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td) / "t.jsonl"
            tp.write_text("".join(json.dumps(e) + "\n" for e in evs))
            f = P.floor_from(self.bench, tp, "B")
        self.assertFalse(f["valid"], f)
        self.assertTrue(any("realised off stream" in p for p in f["problems"]), f)


class TestJobSymlinkLayout(unittest.TestCase):
    """rev5 (2026-10-06): inside a VESSL Job, job_entry.sh moves engine-port/results/ to
    /data/runs/<job>/results and leaves a symlink.  The helpers must derive the engine root from
    the path they were invoked by, not from the resolved file (Job
    pdmux-n3-capacity-a4b0c2e-20261006064114 aborted on `analyze.py: MISSING`).
    Control (lesson 53): the same layout with the pre-rev5 `.resolve()` line restored must fail."""

    def _layout(self, td: Path, revert: bool) -> Path:
        eng = td / "clone" / "workspace" / "engine-port"
        eng.mkdir(parents=True)
        for d in ("benchmarks", "src"):
            (eng / d).symlink_to(ENG / d)
        real = td / "data" / "runs" / "job" / "results"
        shutil.copytree(HERE, real / "n3_capacity",
                        ignore=shutil.ignore_patterns("n3a_*", "n3b_*", "vessl", "probe_*", "__pycache__"))
        shutil.copytree(ENG / "results/r2_eval/lambda0_prereg", real / "r2_eval/lambda0_prereg",
                        ignore=shutil.ignore_patterns("__pycache__"))
        (eng / "results").symlink_to(real)
        if revert:
            for f in ("n3_plan.py", "n3_analyze.py"):
                q = real / "n3_capacity" / f
                t = q.read_text()
                new = "HERE = Path(os.path.abspath(__file__)).parent\n"
                self.assertEqual(t.count(new), 1, f)
                q.write_text(t.replace(new, "HERE = Path(__file__).resolve().parent\n"))
        return eng

    def _run(self, eng: Path, *args):
        return subprocess.run([sys.executable, *map(str, args)], cwd=eng.parents[1], env=_clean_env(),
                              capture_output=True, text=True, timeout=300)

    def test_symlinked_results_layout(self):
        with tempfile.TemporaryDirectory() as td:
            eng = self._layout(Path(td), revert=False)
            h = eng / "results" / "n3_capacity"
            r = self._run(eng, h / "n3_plan.py", "--selftest")
            self.assertEqual(r.returncode, 0, r.stdout[-800:] + r.stderr[-800:])
            self.assertIn("N3 PLAN SELFTEST OK", r.stdout)
            r = self._run(eng, h / "n3_analyze.py", "--help")   # imports the canonical analyze.py
            self.assertEqual(r.returncode, 0, r.stderr[-800:])

    def test_control_resolve_fails_in_symlinked_layout(self):
        with tempfile.TemporaryDirectory() as td:
            eng = self._layout(Path(td), revert=True)
            h = eng / "results" / "n3_capacity"
            r = self._run(eng, h / "n3_plan.py", "--selftest")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("benchmarks/pdmux_eval/analyze.py: MISSING", r.stderr)
            r = self._run(eng, h / "n3_analyze.py", "--help")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("No module named 'analyze'", r.stderr)



class TestRev7Budget(unittest.TestCase):
    """rev7 (2026-10-07, DRAFT before delta audit): shape-B budget constants re-estimated from the
    3rd-round Job's own TIMING.tsv (admit -> next admit, tracked file).  Controls (lesson 53): putting
    the rev6 constants back (overhead 121.0 / no bench-fixed term / floor bench 40 / cap 14400) must
    FAIL the forecast-vs-measurement tests below, i.e. the mutation is detected."""
    JOB = HERE / "n3b_pdmux-n3-capacity-38ed3f4-20261006092813"
    REV6 = {"overhead_s": 121.0, "bench_fixed_s": 0.0, "floor_bench_s": 40.0}

    def setUp(self):
        t = self.JOB / "TIMING.tsv"
        if not t.exists():
            self.skipTest("Job 38ed3f4 TIMING.tsv absent")
        rows = [ln.split("\t") for ln in t.read_text().splitlines()[1:] if ln.strip()]
        self.rows = rows
        self.units = {P.parse_row(r)["name"]: (r, P.parse_row(r)) for r in P.units("B")}
        self.tot = {rows[i][0]: int(rows[i + 1][1]) - int(rows[i][1]) for i in range(len(rows) - 1)}
        self.tot.pop("floor_head")          # cold first boot: modelled by COLD_FIRST_BOOT_EXTRA_S

    def _pred(self, name):
        return P.unit_cost_s("B", self.units[name][0], "normal")

    def _median_rel_err(self):
        errs = sorted((self._pred(n) - t) / t for n, t in self.tot.items())
        return errs[len(errs) // 2]

    def _mutate(self, **kw):
        old = {k: P.SHAPES["B"][k] for k in kw}
        P.SHAPES["B"].update(kw)
        return old

    def test_forecast_matches_measured_unit_totals(self):
        self.assertEqual(len(self.tot), 41)
        self.assertLess(abs(self._median_rel_err()), 0.03, self._median_rel_err())
        worst = max(abs(self._pred(n) - t) / t for n, t in self.tot.items())
        self.assertLess(worst, 0.12, worst)

    def test_every_measured_unit_is_covered_by_its_admission_need(self):
        for n, t in self.tot.items():
            self.assertGreaterEqual(P.need_s("B", n), t, (n, t, P.need_s("B", n)))

    def test_rev6_constants_are_detected(self):
        old = self._mutate(**self.REV6)
        try:
            self.assertGreater(abs(self._median_rel_err()), 0.15)        # forecast ~ -25 %
            self.assertTrue(any(P.need_s("B", n) < t for n, t in self.tot.items()),
                            "rev6 constants must under-cover some measured unit")
        finally:
            P.SHAPES["B"].update(old)

    def test_each_constant_reversion_alone_is_detected(self):
        for k, v in self.REV6.items():
            if k == "floor_bench_s":
                continue                      # floor units are not in the 41 warm totals: see test_floor_bench
            old = self._mutate(**{k: v})
            try:
                self.assertGreater(abs(self._median_rel_err()), 0.02, k)
            finally:
                P.SHAPES["B"].update(old)

    def test_floor_bench_constant_vs_measured_floor_window(self):
        fh = [r for r in self.rows if r[0] == "floor_head"][0]
        measured = int(fh[4]) - int(fh[3])
        fu = self.units["floor_head"][1]
        self.assertLessEqual(abs(P.bench_s("B", fu) - measured), 3.0)
        old = self._mutate(floor_bench_s=40.0)
        try:
            self.assertGreater(abs(P.bench_s("B", fu) - measured), 3.0)        # rev6 value detected
        finally:
            P.SHAPES["B"].update(old)

    def test_normal_corner_admits_48_of_48_and_cap_tolerates_slowdown(self):
        n = P.simulate("B", "normal")
        self.assertTrue(n["all_admitted"], n)
        self.assertGreater(n["gpu_h"], 4.0)                    # honest: no longer fits 4.0 h
        self.assertLessEqual(P.PREFLIGHT_ALLOW_S + P.COLD_FIRST_BOOT_EXTRA_S, 400)
        # uniform 5 % slowdown of every unit (Job-level shift scale in E2VR-3 is 3.6-4.1 %) still admits 48/48
        t = P.PREFLIGHT_ALLOW_S + P.COLD_FIRST_BOOT_EXTRA_S
        lim = P.hard_cap("B") - P.TAIL_RESERVE_S
        # R-5: the tolerance is a property of the chosen cap (16200 -> +2.0 %, 17100 -> +8.2 %, 18000 -> +14.3 %)
        f = min(1.05, 0.99 * P.max_uniform_slowdown("B"))
        for r in P.units("B"):
            self.assertGreaterEqual(lim - t, P.need_s("B", P.parse_row(r)["name"]))
            t += f * P.unit_cost_s("B", r, "normal")

    def test_cap_reversion_is_detected(self):
        old = P.HARD_CAP_S_B
        P.HARD_CAP_S_B = 14400
        try:
            self.assertFalse(P.simulate("B", "normal")["all_admitted"])
        finally:
            P.HARD_CAP_S_B = old

    def test_cap_is_per_shape_and_a_cap_unchanged(self):
        """R-4: the rev7 cap does not touch shape A (A stays 14,400); the runner reads the per-shape constant."""
        self.assertEqual((P.hard_cap("A"), P.hard_cap("B")), (14400, 17100))
        self.assertEqual(P.budget_table("A")["hard_cap"]["cell_section_s"], 14400)
        self.assertEqual(P.budget_table("B")["hard_cap"]["cell_section_s"], 17100)
        out = subprocess.run([sys.executable, str(HERE / "n3_plan.py"), "--const", "HARD_CAP_S_A"], capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "14400")
        old = P.HARD_CAP_S_B
        P.HARD_CAP_S_B = 99999                              # mutation: B cap leaking into A is detected
        try:
            self.assertEqual(P.hard_cap("A"), 14400)
        finally:
            P.HARD_CAP_S_B = old

    def test_stress_floor_is_derived_per_cap_and_pinned_table_is_correct(self):
        """R-5: the fixed point is keyed by the cap; every alternative cap's pinned value equals simulate()."""
        old = P.HARD_CAP_S_B
        try:
            for cap, want in P.REV7_B_WORST_MIN_ADMITTED_BY_CAP.items():
                P.HARD_CAP_S_B = cap
                self.assertEqual(P.simulate("B", "worst")["admitted"], want, cap)
            self.assertEqual(P.REV7_B_WORST_MIN_ADMITTED_BY_CAP, {16200: 41, 17100: 44, 18000: 46})
            self.assertEqual(P.REV7_B_WORST_MIN_ADMITTED, P.REV7_B_WORST_MIN_ADMITTED_BY_CAP[17100])
            P.HARD_CAP_S_B = 16500                          # a cap with no pinned value must not be silently accepted
            self.assertNotIn(P.HARD_CAP_S_B, P.REV7_B_WORST_MIN_ADMITTED_BY_CAP)
        finally:
            P.HARD_CAP_S_B = old

    def test_independent_component_predicates_catch_the_compensating_mutation(self):
        """Audit Q6: overhead 193 + bench-fixed 0 keeps every total-time test green (total unchanged).  The
        overhead and the bench-window residual are therefore tested SEPARATELY against the TIMING.tsv columns."""
        resid, ovh = [], []
        for i, r in enumerate(self.rows):
            u = self.units[r[0]][1]
            if u["kind"] == "floor" or i + 1 >= len(self.rows):
                continue
            win = int(r[4]) - int(r[3])
            resid.append(win - (P.bench_s("B", u) - P.SHAPES["B"]["bench_fixed_s"]))
            ovh.append(int(self.rows[i + 1][1]) - int(r[1]) - win)
        resid.sort()
        ovh.sort()
        self.assertLessEqual(abs(resid[len(resid) // 2] - P.SHAPES["B"]["bench_fixed_s"]), 3.0)
        self.assertLessEqual(abs(ovh[len(ovh) // 2] - P.SHAPES["B"]["overhead_s"]), 8.0)

    def test_compensating_mutation_is_killed_by_component_predicates(self):
        old = self._mutate(overhead_s=193.0, bench_fixed_s=0.0)
        try:
            self.assertLess(abs(self._median_rel_err()), 0.03)            # total-time test is blind to it (equivalent there)
            with self.assertRaises(AssertionError):
                self.test_independent_component_predicates_catch_the_compensating_mutation()
        finally:
            P.SHAPES["B"].update(old)

    def test_bench_fixed_sample_statement_reproduces(self):
        """R-6: n = 42 non-floor units (32 grid, 8 observer, 2 bridge); medians 22.02 / 24.33 / 20.83; overall 22.48 s."""
        import statistics
        base = lambda u: P.bench_s("B", u) - P.SHAPES["B"]["bench_fixed_s"]
        by = {}
        for r in self.rows:
            u = self.units[r[0]][1]
            if u["kind"] != "floor":
                by.setdefault({"obs_on": "obs", "obs_off": "obs"}.get(u["kind"], u["kind"]), []).append((int(r[4]) - int(r[3])) - base(u))
        self.assertEqual({k: len(v) for k, v in by.items()}, {"bridge": 2, "obs": 8, "grid": 32})
        allv = [x for v in by.values() for x in v]
        self.assertEqual(len(allv), 42)
        self.assertAlmostEqual(statistics.median(allv), 22.48, places=2)
        self.assertAlmostEqual(statistics.median(by["grid"]), 22.02, places=2)
        self.assertAlmostEqual(statistics.median(by["obs"]), 24.33, places=2)
        self.assertAlmostEqual(statistics.median(by["bridge"]), 20.83, places=2)
        self.assertTrue(15.0 <= P.SHAPES["B"]["bench_fixed_s"] <= 30.0)

    def test_budget_table_notes_are_not_stale(self):
        for shape in ("A", "B"):
            note = P.budget_table(shape)["note"]
            self.assertNotIn("600 s is UNMEASURED", note)
            self.assertNotIn("per-boot overheads are λ0 VESSL measurements", note)
        self.assertIn("38ed3f4", P.budget_table("B")["note"])
        self.assertIn("UNMEASURED on the N3", P.budget_table("A")["note"])

    def test_shape_a_constants_untouched_by_rev7(self):
        a = P.SHAPES["A"]
        self.assertEqual((a["overhead_s"], a["floor_bench_s"], a["bench_fixed_s"]), (160.1, 110.0, 0.0))
        self.assertTrue(P.simulate("A", "normal")["all_admitted"])

    def test_units_unchanged_by_rev7(self):
        """rev7 touches budget constants only: UNITS.txt must stay byte-identical to the 3rd-round Job's."""
        u = self.JOB / "UNITS.txt"
        self.assertEqual("\n".join(P.units("B")) + "\n", u.read_text())

    def test_resume_registration_digest_gate_blocks_a_rev7_resume_from_rev6(self):
        """Statement of fact the notes rely on: the resume digest filter covers n3_*.py and PREREG_*,
        so any rev7 edit makes ABORT_RESUME fire against the rev6 Job (no resume without a new clause)."""
        src = RUNNER.read_text()
        self.assertIn('grep -E "n3_|PREREG_|VERDICT_|pdmux_n3|analyze.py|lambda0_plan"', src)
        reg = (self.JOB / "REGISTRATION_SHA256.txt").read_text()
        self.assertRegex(reg, r"n3_plan\.py")
        self.assertNotEqual(P.sha256(HERE / "n3_plan.py"),
                            [ln.split()[0] for ln in reg.splitlines() if ln.rstrip().endswith("n3_plan.py")][0])


class TestW5Rev7(unittest.TestCase):
    """rev7 W-5 (length predicate; modes off / length / contain_length; default length -- user decision (iv) 2026-10-07).  On the real Job-38ed3f4 floor boot the
    gap (duration - span window) is 0.88 s; a stray span (the bench_serving test request 2.33 s earlier) in
    the last FLOOR_N breaks it.  Raw jsonl is git-ignored: skipped when absent."""
    JOB = HERE / "n3b_pdmux-n3-capacity-38ed3f4-20261006092813"

    def setUp(self):
        self.bench = self.JOB / "bench_floor_head.jsonl"
        self.tel = self.JOB / "tel_floor_head.jsonl"
        if not (self.bench.exists() and self.tel.exists()):
            self.skipTest("Job 38ed3f4 floor jsonl absent (git-ignored raw data)")

    def test_default_is_length_and_recorded_in_rule_and_tk_json(self):
        """User decision (iv) 2026-10-07: W5_MODE = "length".  The default must be length, the floor `rule` must
        say so, and the T_K.json written by the CLI must carry it.  Mutation: default off -> rule is "rev6"."""
        self.assertEqual(P.W5_MODE, "length")
        self.assertTrue(P.W5_ENABLED)
        f = P.floor_from(self.bench, self.tel, "B")
        self.assertTrue(f["valid"], f)
        self.assertEqual((f["rule"], f["w5_mode"], f["w5_enabled"]), ("rev7+W5:length", "length", True))
        self.assertAlmostEqual(f["w5_window_gap_s"], 0.881, places=2)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "T_K.json"
            r = subprocess.run([sys.executable, str(HERE / "n3_plan.py"), "--floor", "B", str(self.bench), str(self.tel), str(out)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            tk = json.loads(out.read_text())
        self.assertEqual(tk["rule"], "rev7+W5:length")
        self.assertTrue(tk["valid"])
        self.assertEqual(P.tk_problems(tk), [])
        off = P.floor_from(self.bench, self.tel, "B", w5="off")
        self.assertEqual(off["rule"], "rev6")                  # the mutated default would be detected here
        self.assertNotEqual(off["rule"], f["rule"])

    def _with_dropped_first_floor_span(self, td):
        evs = [json.loads(ln) for ln in self.tel.read_text().splitlines() if ln.strip()]
        starts = [e for e in evs if e["event"] == "prefill_span_start"]
        seq = starts[-P.FLOOR_N]["span_seq"]
        evs2 = [e for e in evs if not (e.get("span_seq") == seq and e["event"] in ("prefill_span_start", "prefill_span_end"))]
        tp = Path(td) / "t.jsonl"
        tp.write_text("".join(json.dumps(e) + "\n" for e in evs2))
        return tp

    def test_modes_rule_field_and_real_floor(self):
        """R-1: three modes; `rule` records which; all accept the real Job-38ed3f4 floor."""
        self.assertEqual(P.W5_MODES, ("off", "length", "contain_length"))
        for mode, rule in (("off", "rev6"), ("length", "rev7+W5:length"), ("contain_length", "rev7+W5:contain_length")):
            f = P.floor_from(self.bench, self.tel, "B", w5=mode)
            self.assertTrue(f["valid"], (mode, f["problems"]))
            self.assertEqual((f["rule"], f["w5_mode"]), (rule, mode))
        with self.assertRaises(ValueError):
            P.floor_from(self.bench, self.tel, "B", w5="containment")

    def test_original_containment_would_not_reject_the_test_request_but_length_does(self):
        """R-1 disclosure as a test: with the bench_serving test request inside the last FLOOR_N spans, the marks
        window (CLOCK_MONOTONIC, wraps the whole bench process) still contains every span start, so the ORIGINAL
        containment predicate passes; the rev7 length predicate rejects (gap < 0)."""
        with tempfile.TemporaryDirectory() as td:
            tp = self._with_dropped_first_floor_span(td)
            f = P.floor_from(self.bench, tp, "B", w5="contain_length")
            self.assertFalse(f["valid"], f)
            self.assertFalse(any(x.startswith("W-5 containment") for x in f["problems"]), f["problems"])   # containment passes
            self.assertTrue(any(x.startswith("W-5: bench duration") for x in f["problems"]), f["problems"])  # length rejects
            self.assertLess(f["w5_window_gap_s"], 0.0)
            self.assertFalse(any(x.startswith("W-5") for x in P.floor_from(self.bench, tp, "B", w5="off")["problems"]))

    def test_containment_mode_rejects_spans_outside_marks_window_and_is_fail_closed(self):
        marks = {k: float(v) for k, v in (ln.split() for ln in (self.JOB / "marks_floor_head.txt").read_text().splitlines())}
        with tempfile.TemporaryDirectory() as td:
            mp = Path(td) / "marks.txt"
            mp.write_text(f"bench_start {marks['bench_start'] + 25.0}\nbench_end {marks['bench_end']}\n")   # mutation: window starts late
            f = P.floor_from(self.bench, self.tel, "B", w5="contain_length", marks_path=mp)
            self.assertFalse(f["valid"], f)
            self.assertTrue(any(x.startswith("W-5 containment") for x in f["problems"]), f["problems"])
            # same mutated marks are not read in length mode (no containment problem, still valid)
            self.assertTrue(P.floor_from(self.bench, self.tel, "B", w5="length", marks_path=mp)["valid"])
            fm = P.floor_from(self.bench, self.tel, "B", w5="contain_length", marks_path=Path(td) / "absent.txt")
            self.assertFalse(fm["valid"])
            self.assertTrue(any("marks file missing" in x for x in fm["problems"]), fm["problems"])

    def test_enabled_accepts_the_real_floor(self):
        f = P.floor_from(self.bench, self.tel, "B", w5=True)
        self.assertTrue(f["valid"], f)

    def test_enabled_rejects_a_stray_span_but_disabled_does_not(self):
        evs = [json.loads(ln) for ln in self.tel.read_text().splitlines() if ln.strip()]
        starts = [e for e in evs if e["event"] == "prefill_span_start"]
        # control: drop the first floor span's start+end so the last-16 window absorbs the earlier test span
        seq = starts[-P.FLOOR_N]["span_seq"]
        evs2 = [e for e in evs if not (e.get("span_seq") == seq and e["event"] in ("prefill_span_start", "prefill_span_end"))]
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td) / "t.jsonl"
            tp.write_text("".join(json.dumps(e) + "\n" for e in evs2))
            on = P.floor_from(self.bench, tp, "B", w5=True)
            off = P.floor_from(self.bench, tp, "B", w5=False)
        self.assertFalse(on["valid"], on)
        self.assertTrue(any(p.startswith("W-5") for p in on["problems"]), on)
        self.assertFalse(any(p.startswith("W-5") for p in off["problems"]), off)
        self.assertLess(on["w5_window_gap_s"], 0.0)          # window now exceeds the bench duration


if __name__ == "__main__":
    unittest.main()
