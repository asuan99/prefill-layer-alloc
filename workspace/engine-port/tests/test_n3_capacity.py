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
  floor             formula + (108,0)/running_bs controls (n3_plan selftest).
  estimators        pair-keyed == sealed index-keyed E2 tool on the λ0 VESSL archive (7/7; skipped
                    when the git-ignored archive is absent).  Controls: wrong pair, index remap.
  analyzer          labels/CI/stationarity/observer on synthetic campaigns with known answers.
  runner            allowlist / test-knob / shape aborts; dry-run units + env; budget enforcer
                    (both kill layers removed => overrun; one removed => equivalent mutant);
                    resume (and its digest control).

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
            self.assertLessEqual(n["gpu_h"], 3.5)
            self.assertGreaterEqual(w["admitted"], w["units"] - 1)


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
        self.assertIn("HARD_CAP_S=$(K HARD_CAP_S)", src)
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


if __name__ == "__main__":
    unittest.main()
