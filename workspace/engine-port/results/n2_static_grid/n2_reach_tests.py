"""N2(3-a) reach tests -- PREREG_N2_3A rev16 §8-6 label-reachability table, Job-PROFILE labels (§15 `n2_reach_tests`).
CPU only, HARNESS-SEAL file. Run: PYTHONDONTWRITEBYTECODE=1 python3 -m unittest n2_reach_tests -v

Every PROFILE-side row of §8-6 is produced from SYNTHETIC Job directories (the exact on-disk format of n2_job.py:
marks / tel / srv log / wall / client / green / greedy / boot record / JOB_LEDGER) and pushed through the real
n2_window -> n2_analyze path (which uses the rule-sealed cell_windows). Each contract test has a named harness mutant
in n2_job_mutants.py that must make it fail (lesson 53); a no-mutation control runs first.

Synthetic shortcut (declared): after the 16-iteration trim + 4 the alone segment's ctx is fast-forwarded by one
valid snapshot to c - 75 (below the band [c - 64, c + 63]) so a cell needs ~170 iterations instead of up to ~4,400;
the start ctx (which decides CTX_UNREALIZED) and the band / leader-finish geometry (which decide WINDOW_SHORT) are
exact. Iteration ctx = the median request's length m_j; the leader is m_j + lead_max; an iteration has bs = b while
the leader's length <= c_in + O (the iteration that produces the leader's O-th token still has bs = b).
EA rows of §8-6 that need co-loaded cells (D-3, D-7, R4-2, R4-3, R4-4, CO_WINDOW_SHORT 127/128) are bound by the
rule-sealed reducer tests (TestGeneralizedWindows) or deferred to the Job-EA harness; the pure EA rules implemented
here (coloc_label D-5, signature S1 R3-5 / job label D-4 (ii)) are tested below.
"""
import json
import math
import os
import shutil
import tempfile
import time
import unittest

import n2_analyze as A
import n2_plan
import n2_signature as SG
import n2_window as NW

REG = n2_plan.load_registered()
UNITS = n2_plan.units("PROFILE", REG)
WALL_OFF = 1.79e9                     # synthetic wall = monotonic + WALL_OFF (logs in UTC, tz_gmtoff_s = 0)
PAIR = NW.PAIR
TAU_D44 = {1: 16.0, 8: 18.0, 32: 28.0, 48: 36.0}
CTX_F = {512: 0.9, 2048: 1.0, 8192: 1.25}
D_F = {16: 1.6, 24: 1.35, 34: 1.15, 44: 1.0, 108: 0.75}
LOG_EVERY = 20
IDX_OF = {v: n2_plan.IDX[k] for k, v in PAIR.items()}


def step_ms(D, b, c):
    return TAU_D44[b] * CTX_F[c] * D_F[D]


def _log(ts_mono, text):
    return f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ts_mono + WALL_OFF))}] {text}"


class Cell:
    """Knobs of one synthetic cell (defaults = a clean VALID cell)."""

    def __init__(self, **kw):
        self.lead_med = kw.get("lead_med", 0)
        self.lead_max = kw.get("lead_max", 0)
        self.pe_off = kw.get("pe_off", 0)                  # window iterations on the wrong pair
        self.prefill_in_window = kw.get("prefill_in_window", False)
        self.seq_gap = kw.get("seq_gap", False)
        self.dropped = kw.get("dropped", 0)
        self.cg_false = kw.get("cg_false", False)
        self.retract = kw.get("retract", False)
        self.bench_bad = kw.get("bench_bad", False)
        self.factor = kw.get("factor", 1.0)                # S-per multiplier (D-sensitivity / F-N2-6 scenarios)
        self.short = kw.get("short", False)                # the run stops before the band ends
        self.ctx_c_in = kw.get("ctx_c_in")                 # telemetry-only start length (D-2 scenario)
        self.fast_forward = kw.get("fast_forward", True)
        self.stall_x = kw.get("stall_x")                     # one in-band step `stall_x` times longer (iter_seq kept)


class SynthJob:
    def __init__(self, out, units=UNITS):
        self.out, self.units = out, units
        self.cell_kw = {}           # cell_id -> Cell kwargs
        self.boot_kw = {}           # boot -> dict(r7=bool, no_record_after=bool, wiring_bad=bool, greedy_diverge)
        self.ledger_status = {}     # boot -> status
        self.obs_factor = {}        # round -> OBS C-itl factor (observer effect)
        self.on_skip = set()
        self.ledger_top = {}        # extra JOB_LEDGER.json top-level fields (stopped / job_abort, H17-2)
        self.ledger_tail = []       # extra ledger rows (e.g. the runner's {"job_abort": ...} row)

    # ------------------------------------------------------------------ writers
    def write(self):
        os.makedirs(self.out, exist_ok=True)
        t = 1000.0
        rows = []
        for u in self.units:
            st = self.ledger_status.get(u["name"], "ADMITTED")
            rows.append(dict(unit=u["name"], status=st, need_s=n2_plan.need_s("PROFILE", REG, u["name"]),
                             remaining_s=1e6, n_cells=len(u["cells"])))
            if st != "ADMITTED":
                continue
            t = self._boot(u, t) + 50.0
        with open(os.path.join(self.out, "JOB_LEDGER.json"), "w") as fh:
            json.dump(dict(dict(job="PROFILE", ledger=rows + self.ledger_tail, stopped=None, cap_s=36700.0),
                           **self.ledger_top), fh)

    def _boot(self, u, t):
        name, bk = u["name"], self.boot_kw.get(u["name"], {})
        recs, marks, wall, client, log = [], [], [], [], []
        dropped = [0]
        seq = [0, 0]                  # iter_seq, span_seq
        log.append("[fake] Capture cuda graph bs [1, 2, 4, 8, 12, 16, 24, 32, 40, 48]")
        if u["sticky"]:
            log.append(f"PD-mux sticky partition ENABLED: ... (fixed target index={n2_plan.STICKY_IDX[u['D']]})")
        recs.append(dict(event="phase_marker", phase="startup", timestamp_monotonic_s=t, dropped_events=0))
        marks.append(f"boot_plan {u['D']} {len(u['cells'])}")
        marks.append(f"warmup_start {name} {t + 0.1!r}")
        marks.append(f"warmup_end {name} {t + 0.2!r}")
        t += 1.0
        for cell in u["cells"]:
            ck = Cell(**self.cell_kw.get(cell["cell_id"], {}))
            t = self._cell(u, cell, ck, t, recs, marks, wall, client, log, dropped, seq) + 2.5   # no shared second
        if bk.get("r7"):
            log.append(_log(t, "PDMUX_PHASE_EVENTS disabled after failure: RuntimeError('x')"))
        if not bk.get("no_record_after"):
            recs.append(dict(event="runtime_snapshot", phase="benchmark", timestamp_monotonic_s=t + 1.0,
                             decode_iterations=seq[0], active_decode_sequences=0, decode_running_batch_size=0,
                             context_p50=1, prefill_sms=108, decode_sms=0, prefill_queue_depth=0,
                             dropped_events=dropped[0]))
        p = lambda k, ext: os.path.join(self.out, f"{k}_{name}.{ext}")
        with open(p("tel", "jsonl"), "w") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in recs))
        with open(p("marks", "txt"), "w") as fh:
            fh.write("\n".join(marks) + "\n")
        with open(p("wall", "txt"), "w") as fh:
            fh.write("\n".join(wall) + "\n")
        with open(p("client", "jsonl"), "w") as fh:
            fh.write("".join(json.dumps(c) + "\n" for c in client))
        with open(p("srv", "log"), "w") as fh:
            fh.write("\n".join(log) + "\n")
        wiring = dict(log_problems=["bad"] if bk.get("wiring_bad") else [], green_problems=[])
        with open(p("boot", "json"), "w") as fh:
            json.dump(dict(name=name, tz_gmtoff_s=0, wiring=wiring), fh)
        if u["greedy"]:
            outs = [dict(text=f"t{k}" + ("x" if bk.get("greedy_diverge") else ""),
                         output_ids=[k, k + 1 + (1 if bk.get("greedy_ids_diverge") else 0)]) for k in range(6)]
            if "greedy_ids" in bk:                                 # H17-1: None / [] / missing key on output k
                for k, v in bk["greedy_ids"].items():
                    if v == "drop":
                        outs[k].pop("output_ids")
                    else:
                        outs[k]["output_ids"] = v
            with open(p("greedy", "json"), "w") as fh:
                json.dump(dict(boot=name, outputs=outs), fh)
        return t

    def _cell(self, u, cell, ck, t0, recs, marks, wall, client, log, dropped, seq):
        b, c = cell["b"], cell["c"]
        c_in = cell["c_in"]
        O = cell["O"]
        D = u["alone_D"]
        st = step_ms(D if u["kind"] != "OBS" else 44, b, c) * ck.factor / 1000.0
        pe = u["phase_events"]
        alone_pair = PAIR[u["alone_D"]]
        wrong = PAIR[108] if alone_pair != PAIR[108] else PAIR[u["D"]]
        cid = cell["cell_id"]
        wall.append(f"cell_start {cid} {t0 - 0.0005 + WALL_OFF!r} {t0 - 0.0005!r}")
        marks.append(f"cell_start {cid} {u['D']} {b} {c} {t0!r}")
        marks.append(f"load_call_start call-{cid} {t0 + 0.0001!r}")
        seq[1] += 1
        if pe:
            recs.append(dict(event="prefill_span_start", span_seq=seq[1], timestamp_monotonic_s=t0 + 0.001,
                             extend_tokens=b * c_in, n_new=b, running_bs=0, prefill_sms=108, decode_sms=0,
                             dropped_events=dropped[0]))
        log.append(_log(t0 + 0.001, "Prefill batch, #new-seq: 1, cuda graph: False, input throughput (token/s): 0"))
        t = t0 + 0.002
        # a few in-flight burst iterations (excluded from every alone window)
        for k in range(3):
            seq[0] += 1
            recs.append(dict(event="decode_iteration", iter_seq=seq[0], t_launch=t, t_sync=t + st * 0.8, bs=max(1, b - 1),
                             prefill_in_flight=True, prefill_sms=PAIR[u["D"]][0], decode_sms=PAIR[u["D"]][1],
                             stream_idx=1, dropped_events=dropped[0]))
            t += st
        if pe:
            recs.append(dict(event="prefill_span_end", span_seq=seq[1], timestamp_monotonic_s=t, n_done=b,
                             running_bs=0, dropped_events=dropped[0]))
        t += 0.0005
        m = (ck.ctx_c_in if ck.ctx_c_in is not None else c_in) + 1 + ck.lead_med   # median length, 1st alone iter.
        stop_m = c + 71 + 4                             # client cancel (lower median >= c + 71) + 4 in flight
        if ck.short:
            stop_m = c + 30
        req_times = [[] for _ in range(b)]
        j = 0
        win_lo, win_hi = c - 64, c + 63
        mid_window_done = False
        while m <= stop_m:
            if ck.fast_forward and j == 20 and m < c - 75:
                m = c - 75
            seq[0] += 1
            if ck.seq_gap and m == c and not mid_window_done:
                seq[0] += 1                             # a missing iter_seq inside the band (recorder gap)
                mid_window_done = True
            leader = m + ck.lead_max
            bs = b if (leader <= c_in + O or b == 1) else b - 1
            in_band = win_lo <= m <= win_hi
            pair = wrong if (in_band and win_lo + 10 <= m < win_lo + 10 + ck.pe_off) else alone_pair
            if j == 0 or (j % 16 == 0 and bs == b) or (ck.fast_forward and j == 20):
                recs.append(dict(event="runtime_snapshot", phase="benchmark", timestamp_monotonic_s=t - 0.0001,
                                 decode_iterations=seq[0], active_decode_sequences=bs, decode_running_batch_size=bs,
                                 context_p50=m, context_p95=m + ck.lead_max // 2, context_max=m + ck.lead_max,
                                 prefill_sms=alone_pair[0], decode_sms=alone_pair[1], prefill_queue_depth=0,
                                 dropped_events=dropped[0]))
            if ck.dropped and m == c + 10:
                dropped[0] += ck.dropped
            if pe:
                recs.append(dict(event="decode_iteration", iter_seq=seq[0], t_launch=t, t_sync=t + st * 0.8, bs=bs,
                                 prefill_in_flight=False, prefill_sms=pair[0], decode_sms=pair[1],
                                 stream_idx=IDX_OF[pair],
                                 prefill_kernel_done_wait=False, dropped_events=dropped[0]))
                if ck.prefill_in_window and m == c:
                    seq[1] += 1
                    recs.append(dict(event="prefill_span_start", span_seq=seq[1], timestamp_monotonic_s=t + 0.0001,
                                     extend_tokens=100, n_new=1, dropped_events=dropped[0]))
            obs = self.obs_factor.get(u["round"], 1.0) if u["kind"] == "OBS" else 1.0
            for q in range(bs):
                req_times[q].append(t + st * 0.8 * obs + 0.001)
            if seq[0] % LOG_EVERY == 0:
                cg = "False" if ck.cg_false else "True"
                rt = ", #retracted_reqs: 1" if ck.retract else ""
                log.append(_log(t, f"Decode batch, #running-req: {bs}{rt}, cuda graph: {cg}, gen throughput: 0"))
            t += st * obs * (ck.stall_x if (ck.stall_x and m == c) else 1.0)
            m += 1
            j += 1
        t_end = t + 0.05
        marks.append(f"load_call_end call-{cid} {t_end - 0.01!r}")
        marks.append(f"cell_end {cid} {t_end!r}")
        marks.append(f"fix_s {cid} 0.1")
        wall.append(f"cell_end {cid} {t_end + 0.0005 + WALL_OFF!r} {t_end + 0.0005!r}")
        reqs = []
        for q in range(b):
            reqs.append(dict(rid=f"{cid}-d{q}", n=len(req_times[q]), times=req_times[q], prompt_tokens=c_in,
                             completion_tokens=len(req_times[q]), finish=None, done=False, cancelled=True))
        if ck.bench_bad:
            reqs = reqs[:-1] if b > 1 else [dict(reqs[0], prompt_tokens=c_in + 1)]
        client.append(dict(cell_id=cid, b=b, c=c, c_in=c_in, O=O, cancel_fired=True, t_cancel=t_end - 0.02,
                           requests=reqs))
        recs.append(dict(event="runtime_snapshot", phase="benchmark", timestamp_monotonic_s=t_end - 0.001,
                         decode_iterations=seq[0], active_decode_sequences=0, decode_running_batch_size=0,
                         context_p50=1, prefill_sms=108, decode_sms=0, prefill_queue_depth=0,
                         dropped_events=dropped[0]))
        return t_end


def run(job):
    job.write()
    return A.analyze(job.out, reg=REG)


def cid(boot, b, c):
    return f"{boot}-b{b}-c{c}"


def rep(res, boot, b, c):
    return next(r for r in res["replicates"] if r["cell_id"] == cid(boot, b, c))


class _Base(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="n2reach_")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def job(self, units=UNITS):
        return SynthJob(self.td, units)


# ======================================================================================================== full Job
class TestFullJob(_Base):
    @classmethod
    def setUpClass(cls):
        cls.ctd = tempfile.mkdtemp(prefix="n2reach_full_")
        cls.res = run(SynthJob(cls.ctd))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.ctd, ignore_errors=True)

    def test_profile_complete(self):                               # §8-6 VALID / POINT_REALIZED / COMPLETE rows
        r = self.res
        self.assertEqual(r["profile_label"], "PROFILE_COMPLETE")
        self.assertEqual({v["label"] for v in r["D_labels"].values()}, {"D_PROFILE_COMPLETE"})
        self.assertEqual(sum(1 for p in r["points"].values() if p["label"] == "POINT_REALIZED"), 60)
        self.assertEqual(r["neg_control"]["label"], "NEG_CONTROL_HELD")
        self.assertEqual(r["neg_control"]["n"], 8 * 12)

    def test_108_pools_off16_and_off44(self):                     # §9-2 R-1: n = 8 unconditionally
        for b in n2_plan.B_GRID:
            for c in n2_plan.C_GRID:
                p = self.res["points"][f"108,{b},{c}"]
                self.assertEqual((p["n"], p["fixed16_n"], p["fixed44_n"]), (8, 4, 4))
        self.assertEqual(self.res["points"]["44,8,2048"]["n"], 4)

    def test_point_values_are_replicate_medians(self):            # §6-3
        p = self.res["points"]["44,32,2048"]
        self.assertAlmostEqual(p["itl_p50_ms"], step_ms(44, 32, 2048), places=6)
        self.assertEqual(p["measured_steps"], 4 * 128)
        self.assertAlmostEqual(p["residual_p95_ms"], max(p["replicate_p95"]) - A.statistics.median(p["replicate_p95"]))

    def test_observer_ok_and_correctness_and_wiring(self):
        self.assertEqual(self.res["observer"]["verdict"], "OBSERVER_OK")
        self.assertEqual({v["label"] for v in self.res["sticky_correctness"].values()},
                         {f"STICKY_CORRECTNESS_BYTE_IDENTICAL_D{d}" for d in (16, 24, 34, 44)})
        self.assertEqual({v["label"] for v in self.res["sticky_wired"].values()},
                         {f"STICKY_WIRED_D{d}" for d in (16, 24, 34, 44)})

    def test_dsens_ge3_everywhere_and_forecasts(self):            # §9-6 D_SENS_GE3, F-N2-D3 HELD
        rows = self.res["dsens"]["rows"]
        self.assertEqual(len(rows), 60)
        self.assertEqual({r["label"] for r in rows}, {"D_SENS_GE3"})
        r0 = next(r for r in rows if r["pair"] == [16, 44] and (r["b"], r["c"]) == (8, 2048))
        self.assertAlmostEqual(r0["x"][0], math.log(1.6), places=6)
        self.assertEqual(r0["n"], 4)
        self.assertIsNotNone(r0["ci"])
        self.assertEqual(self.res["forecasts"]["F-N2-D3"], "HELD")
        self.assertEqual(self.res["forecasts"]["F-N2-6"], "HELD")
        self.assertEqual(self.res["forecasts"]["F-N2-7P"], "HELD")
        self.assertEqual(self.res["forecasts"]["F-N2-5"], "HELD")

    def test_blind_and_window_descriptives(self):                  # H15-4: window scope only, no co-located stat
        self.assertFalse(self.res["blind"]["coloc_statistics_computed"])
        w = self.res["windows"][0]
        self.assertEqual(w["a2_pair_w"]["convention"].split()[0], "A2-pair-W")

    def test_n2c75_report_zero(self):
        rep_ = self.res["n2c_75_report"]
        self.assertEqual((rep_["zero_weight_windows"], rep_["endpoint_prefill_events"]), (0, 0))
        self.assertEqual(rep_["windows_checked"], 8 * 12 + 16 * 12)

    def test_profile_csv_rows(self):
        p = os.path.join(self.ctd, "pts.csv")
        A.profile_csv({tuple(int(x) for x in k.split(",")): v for k, v in self.res["points"].items()}, p)
        with open(p) as fh:
            self.assertEqual(len(fh.read().strip().splitlines()), 61)

    def test_d_table(self):                                          # §12-5 (descriptive)
        rows = self.res["d_table"]["rows"]
        self.assertEqual(len(rows), 12 * 4)
        self.assertTrue(all(r["target_ms"] == r["T_ms"] - 0.1 * r["T_ms"] for r in rows))


# ======================================================================================================== §8-6 rows
U1_44 = [u for u in UNITS if u["round"] == 1 and u["cost_key"] in ("ON44", "OFF44")]


class TestReplicateRows(_Base):
    def one(self, units, **cells):
        j = self.job(units)
        for k, v in cells.items():
            j.cell_kw[k] = v
        return j

    def test_valid_spread_cell_and_x_rule_boundaries(self):        # §6-2 R4-5 / §15 X rule (exact registered c_in)
        u = [x for x in UNITS if x["name"] == "r1_ON44"]
        cell = next(cc for cc in u[0]["cells"] if (cc["b"], cc["c"]) == (48, 2048))
        L = cell["c"] - 81 - cell["c_in"]                          # largest lead_med with start ctx <= c - 80
        M = cell["c_in"] + cell["O"] - cell["c"] - 63              # largest lead_max keeping bs == b through c + 63
        s_hat = REG["derived"]["profile"]["44,48,2048"]["s_hat"]
        self.assertEqual(L, math.ceil(s_hat) + 64)
        self.assertGreaterEqual(M, math.floor(s_hat) + 64)
        k = cell["cell_id"]
        for kw, want in ((dict(lead_med=L, lead_max=M), "VALID"),
                         (dict(lead_med=L + 1), "CTX_UNREALIZED"),
                         (dict(lead_max=M + 1), "WINDOW_SHORT")):
            j = self.one(u, **{k: dict(kw, fast_forward=False)})
            res = run(j)
            self.assertEqual(rep(res, "r1_ON44", 48, 2048)["status"], want, kw)
            shutil.rmtree(self.td); os.makedirs(self.td)

    def test_d2_x96_vs_x64(self):                                  # D-2: X = 64 -> CTX_UNREALIZED, 96 -> VALID
        u = [x for x in UNITS if x["name"] == "r1_ON44"]
        k = cid("r1_ON44", 8, 2048)
        for x, want in ((96, "VALID"), (64, "CTX_UNREALIZED"), (145, "VALID")):
            res = run(self.one(u, **{k: dict(ctx_c_in=2048 - x, fast_forward=False)}))
            self.assertEqual(rep(res, "r1_ON44", 8, 2048)["status"], want, x)
            shutil.rmtree(self.td); os.makedirs(self.td)

    def test_measurement_rows(self):                               # BENCH / RETRACT / RECORDER / CG rows
        u = [x for x in UNITS if x["name"] == "r1_ON44"]
        cells = {cid("r1_ON44", 1, 512): dict(bench_bad=True), cid("r1_ON44", 1, 2048): dict(retract=True),
                 cid("r1_ON44", 1, 8192): dict(dropped=1), cid("r1_ON44", 8, 512): dict(seq_gap=True),
                 cid("r1_ON44", 8, 2048): dict(cg_false=True), cid("r1_ON44", 8, 8192): dict(pe_off=2),
                 cid("r1_ON44", 32, 512): dict(pe_off=1), cid("r1_ON44", 32, 2048): dict(prefill_in_window=True),
                 cid("r1_ON44", 32, 8192): dict(short=True)}
        res = run(self.one(u, **cells))
        want = {(1, 512): "BENCH_INVALID", (1, 2048): "RETRACT", (1, 8192): "RECORDER_UNPROVEN",
                (8, 512): "RECORDER_UNPROVEN", (8, 2048): "CG_UNPROVEN", (8, 8192): "D_UNREALIZED",
                (32, 512): "VALID", (32, 2048): "D_UNREALIZED", (32, 8192): "WINDOW_SHORT",
                (48, 512): "VALID"}
        for (b, c), w in want.items():
            self.assertEqual(rep(res, "r1_ON44", b, c)["status"], w, (b, c))
        self.assertIn("Δdropped", rep(res, "r1_ON44", 1, 8192)["reason"])
        self.assertIn("§7-1 (c)", rep(res, "r1_ON44", 32, 2048)["reason"])

    def test_stall_is_valid_and_counted(self):                     # §7-1 (R) rev14: a 13x stall (iter_seq contiguous)
        u = [x for x in UNITS if x["name"] == "r1_ON44"]             # is VALID + stall count 1 (10x is descriptive)
        res = run(self.one(u, **{cid("r1_ON44", 8, 2048): dict(stall_x=13.0)}))
        r = rep(res, "r1_ON44", 8, 2048)
        self.assertEqual(r["status"], "VALID")
        self.assertEqual((r["stats"]["stalls_gt10"], r["stats"]["stalls_gt5"]), (1, 1))

    def test_no_record_after_last_cell(self):                      # F-6
        u = [x for x in UNITS if x["name"] == "r1_ON44"]
        j = self.job(u)
        j.boot_kw["r1_ON44"] = dict(no_record_after=True)
        res = run(j)
        last = u[0]["cells"][-1]
        self.assertEqual(rep(res, "r1_ON44", last["b"], last["c"])["status"], "RECORDER_UNPROVEN")
        self.assertIn("F-6", rep(res, "r1_ON44", last["b"], last["c"])["reason"])
        first = u[0]["cells"][0]
        self.assertEqual(rep(res, "r1_ON44", first["b"], first["c"])["status"], "VALID")

    def test_r7_is_boot_level(self):                               # D-6: every cell of the boot, incl. earlier ones
        u = [x for x in UNITS if x["name"] == "r1_ON44"]
        j = self.job(u)
        j.boot_kw["r1_ON44"] = dict(r7=True)
        res = run(j)
        self.assertEqual({r["status"] for r in res["replicates"] if r["boot"] == "r1_ON44"}, {"RECORDER_UNPROVEN"})

    def test_cg_boundary_seconds(self):                            # §3-2 R-5: [ceil(start), floor(end) - 1]
        off = 1790000000.0
        boot = dict(cells={"x": {"cell_start": ((44, 1, 512), 100.2), "cell_end": 101.9}},
                    wall={("cell_start", "x"): (100.2 + off, 100.2)},
                    log_lines=[(off + 100, "Decode batch, cuda graph: True"), (off + 101, "Decode batch, cuda graph: True")])
        self.assertEqual(NW.cg_evidence(boot, "x"), (False, 0, 0))          # range [101, 100]: empty -> unproven
        boot["cells"]["x"]["cell_end"] = 103.9                             # range [101, 102]
        self.assertEqual(NW.cg_evidence(boot, "x"), (True, 1, 0))
        boot["log_lines"].append((off + 102, "Decode batch, cuda graph: False"))
        self.assertEqual(NW.cg_evidence(boot, "x"), (False, 2, 1))
        boot["log_lines"].append((off + 102, "Prefill batch, cuda graph: False"))   # F-5: ignored
        self.assertEqual(NW.cg_evidence(boot, "x"), (False, 2, 1))
        boot["log_lines"].append((off + 103, "Decode batch, cuda graph: False"))    # boundary second: excluded
        self.assertEqual(NW.cg_evidence(boot, "x"), (False, 2, 1))

    def test_n2c75_report_fields(self):                             # N2C-75 counts come from the Job raw data
        u = [x for x in UNITS if x["name"] == "r1_ON44"]
        run(self.job(u))
        boot = NW.load_boot(self.td, "r1_ON44")
        cell = dict(u[0]["cells"][0])
        r0 = NW.profile_replicate(boot, u[0], cell)
        self.assertEqual((r0["status"], r0["n2c75_zero_weight"], r0["n2c75_endpoint_prefill_events"]), ("VALID", False, 0))
        t_end = r0["_window"][-1]["t_sync"]
        boot["records"].append(dict(event="prefill_span_start", span_seq=999999, timestamp_monotonic_s=t_end,
                                    extend_tokens=10, n_new=1, dropped_events=0))
        r1 = NW.profile_replicate(boot, u[0], cell)
        self.assertEqual((r1["status"], r1["n2c75_endpoint_prefill_events"]), ("D_UNREALIZED", 1))

    def test_prefill_false_lines_ignored(self):                    # F-5: only Decode batch lines count
        u = [x for x in UNITS if x["name"] == "r1_ON44"]
        res = run(self.job(u))
        boot = NW.load_boot(self.td, "r1_ON44")
        self.assertTrue(any("Prefill batch" in t and "cuda graph: False" in t for _s, t in boot["log_lines"]))
        self.assertEqual(rep(res, "r1_ON44", 1, 512)["status"], "VALID")


class TestLabelRows(_Base):
    def test_not_attempted_budget_tail(self):                      # §14-3 / N2C-17: r4 tail ON44 refused
        j = self.job()
        for name in ("r4_ON44", "r4_OBS44"):
            j.ledger_status[name] = "NOT_ATTEMPTED_BUDGET"
        res = run(j)
        self.assertEqual(rep(res, "r4_ON44", 8, 2048)["status"], "NOT_ATTEMPTED_BUDGET")
        self.assertEqual({res["points"][f"44,{b},{c}"]["label"] for b in n2_plan.B_GRID for c in n2_plan.C_GRID},
                         {"UNRESOLVED_N_BELOW_4"})
        self.assertEqual(res["D_labels"]["44"]["label"], "D_PROFILE_PARTIAL")
        self.assertEqual(res["profile_label"], "PROFILE_PARTIAL")
        self.assertEqual(res["observer"]["verdict"], "OBSERVER_UNRESOLVED_N_BELOW_4")
        self.assertEqual(res["forecasts"]["F-N2-7P"], "FALSIFIED")
        self.assertEqual({m["unit"] for m in res["missing_boots"]}, {"r4_ON44", "r4_OBS44"})
        self.assertEqual(res["D_labels"]["44"]["attempted"], 36)   # A excludes NOT_ATTEMPTED_BUDGET

    def test_neg_control_failed_population_not_valid(self):        # §11-3 (i)
        j = self.job()
        j.cell_kw[cid("r3_OFF16", 8, 2048)] = dict(pe_off=13)     # 115/128 = 0.898 at (0,108)
        res = run(j)
        self.assertEqual(rep(res, "r3_OFF16", 8, 2048)["status"], "D_UNREALIZED")
        self.assertEqual(res["neg_control"]["label"], "NEG_CONTROL_FAILED")
        self.assertEqual(res["profile_label"], "PROFILE_NOT_BUILT")

    def test_neg_control_on_leak_is_d_channel(self):              # §11-3 (iii): ON leak -> D_UNREALIZED only
        j = self.job()
        j.cell_kw[cid("r2_ON16", 8, 2048)] = dict(pe_off=2)
        res = run(j)
        self.assertEqual(rep(res, "r2_ON16", 8, 2048)["status"], "D_UNREALIZED")
        self.assertEqual(res["neg_control"]["label"], "NEG_CONTROL_HELD")
        self.assertEqual(res["points"]["16,8,2048"]["label"], "UNRESOLVED_N_BELOW_4")
        self.assertEqual(res["D_labels"]["16"]["label"], "D_PROFILE_PARTIAL")
        self.assertEqual(res["profile_label"], "PROFILE_PARTIAL")

    def test_neg_control_unresolved(self):                         # §11-3 (ii): every OFF boot R7
        j = self.job()
        for u in UNITS:
            if u["kind"] == "OFF":
                j.boot_kw[u["name"]] = dict(r7=True)
        res = run(j)
        self.assertEqual(res["neg_control"]["label"], "NEG_CONTROL_UNRESOLVED")
        self.assertEqual(res["profile_label"], "PROFILE_NOT_BUILT")

    def test_108_pooling_independent_of_f_n2_6(self):              # §9-2 D-1 inverse construction
        j = self.job()
        for u in UNITS:
            if u["kind"] == "OFF" and u["D"] == 16:
                for cc in u["cells"]:
                    j.cell_kw[cc["cell_id"]] = dict(factor=1.05)
        j.cell_kw[cid("r2_OFF16", 8, 2048)] = dict(factor=1.05, bench_bad=True)
        res = run(j)
        p = res["points"]["108,8,2048"]
        self.assertEqual((p["label"], p["n"]), ("POINT_REALIZED", 7))
        self.assertEqual(res["forecasts"]["F-N2-6"], "FALSIFIED")

    def test_d_realization_failed(self):                           # ON16: half of the attempted replicates D_UNREAL.
        j = self.job()
        for u in UNITS:
            if u["kind"] == "ON" and u["D"] == 16:
                for cc in u["cells"][:6]:
                    j.cell_kw[cc["cell_id"]] = dict(pe_off=3)
        res = run(j)
        self.assertEqual(res["D_labels"]["16"]["label"], "D_REALIZATION_FAILED")
        self.assertEqual(res["D_labels"]["16"]["d_unrealized"], 24)

    def test_observer_fail(self):
        j = self.job()
        for r in (1, 2, 3, 4):
            j.obs_factor[r] = 0.9                                 # OBS (PHASE_EVENTS off) 10 % faster -> ln(ON/OFF) > 0
        res = run(j)
        self.assertEqual(res["observer"]["verdict"], "OBSERVER_FAIL")

    def test_observer_ci_must_contain_zero(self):                  # |mean| < ln 1.03 but CI excludes 0 -> FAIL
        j = self.job()
        for r in (1, 2, 3, 4):
            j.obs_factor[r] = 0.99
        res = run(j)
        cell = res["observer"]["cells"]["1,2048"]
        self.assertLess(abs(cell["mean"]), A.LN3)
        self.assertGreater(cell["ci"]["ci95_low"], 0.0)
        self.assertEqual(res["observer"]["verdict"], "OBSERVER_FAIL")

    def test_sticky_correctness_diverged_and_wiring(self):
        j = self.job()
        j.boot_kw["r1_ON24"] = dict(greedy_diverge=True)
        j.boot_kw["r3_ON34"] = dict(wiring_bad=True)
        j.boot_kw["r1_ON34"] = dict(greedy_ids_diverge=True)      # token ids differ, text identical
        res = run(j)
        self.assertEqual(res["sticky_correctness"]["24"]["label"], "STICKY_CORRECTNESS_DIVERGED_D24")
        self.assertEqual(res["sticky_correctness"]["34"]["label"], "STICKY_CORRECTNESS_DIVERGED_D34")
        self.assertEqual(res["sticky_correctness"]["16"]["label"], "STICKY_CORRECTNESS_BYTE_IDENTICAL_D16")
        self.assertEqual(res["sticky_wired"]["34"]["label"], "STICKY_WIRING_NOT_ESTABLISHED_D34")

    def test_sticky_correctness_output_ids_absent(self):            # §18-2-d #4 (H17-1): None == None never passes
        j = self.job()
        none_all = {k: None for k in range(6)}
        j.boot_kw["r1_OFF16"] = dict(greedy_ids=none_all)                # reference side: every D is affected
        res = run(j)
        for D in (16, 24, 34, 44):
            self.assertEqual(res["sticky_correctness"][str(D)]["label"], f"STICKY_CORRECTNESS_UNRESOLVED_D{D}")
            self.assertEqual(res["sticky_correctness"][str(D)]["why"], "output_ids absent")
        j = self.job()
        j.boot_kw["r1_OFF16"] = dict(greedy_ids=none_all)
        j.boot_kw["r1_ON44"] = dict(greedy_ids=none_all)                 # both sides None (text equal)
        j.boot_kw["r1_ON24"] = dict(greedy_ids={3: []})                  # empty list on one output
        j.boot_kw["r1_ON16"] = dict(greedy_ids={5: "drop"})              # key missing on one output
        res = run(j)
        self.assertEqual(res["sticky_correctness"]["44"]["label"], "STICKY_CORRECTNESS_UNRESOLVED_D44")
        j = self.job()
        j.boot_kw["r1_ON24"] = dict(greedy_ids={3: []})
        j.boot_kw["r1_ON16"] = dict(greedy_ids={5: "drop"})
        res = run(j)
        self.assertEqual(res["sticky_correctness"]["24"]["label"], "STICKY_CORRECTNESS_UNRESOLVED_D24")
        self.assertEqual(res["sticky_correctness"]["16"]["label"], "STICKY_CORRECTNESS_UNRESOLVED_D16")
        self.assertEqual(res["sticky_correctness"]["44"]["label"], "STICKY_CORRECTNESS_BYTE_IDENTICAL_D44")

    def test_f_n2_7p_non_budget_stop_unresolved(self):              # §18-2-d #10 (H17-2)
        for top, tail in ((dict(stopped="job_abort"), []), (dict(stopped="vd_abort"), []),
                          (dict(job_abort=dict(code=3, tag="ABORT_CG_CANARY", msg="x")), []),
                          ({}, [dict(job_abort=dict(code=6, tag="ABORT_STICKY", msg="x"))])):
            j = self.job()
            j.ledger_top, j.ledger_tail = top, tail
            self.assertEqual(run(j)["forecasts"]["F-N2-7P"], "UNRESOLVED", (top, tail))
        j = self.job()
        j.ledger_top = dict(stopped="budget")
        self.assertEqual(run(j)["forecasts"]["F-N2-7P"], "HELD")       # budget stop with all 28 admitted
        j = self.job()
        j.ledger_top = dict(stopped="budget", job_abort=None)
        j.ledger_status["r4_OBS44"] = "NOT_ATTEMPTED_BUDGET"
        self.assertEqual(run(j)["forecasts"]["F-N2-7P"], "FALSIFIED")

    def test_dsens_label_variety(self):                            # §9-6: WITHIN / INVERTED / UNRESOLVED / n = 3
        j = self.job()
        for u in UNITS:
            if u["kind"] == "ON" and u["D"] == 24:
                for cc in u["cells"]:
                    if (cc["b"], cc["c"]) == (8, 512):            # D24 == D16 -> (16,24) WITHIN
                        j.cell_kw[cc["cell_id"]] = dict(factor=1.6 / 1.35)
                    if (cc["b"], cc["c"]) == (8, 2048):           # D24 slower than D16 -> (16,24) INVERTED
                        j.cell_kw[cc["cell_id"]] = dict(factor=1.8 / 1.35)
                    if (cc["b"], cc["c"]) == (1, 512):            # mixed -> UNRESOLVED
                        j.cell_kw[cc["cell_id"]] = dict(factor=(1.8 if u["round"] in (1, 2) else 1.0) / 1.35)
        j.cell_kw[cid("r3_ON24", 1, 2048)] = dict(bench_bad=True)  # n = 3
        res = run(j)
        lab = {(r["b"], r["c"]): r["label"] for r in res["dsens"]["rows"] if r["pair"] == [16, 24]}
        self.assertEqual(lab[(8, 512)], "D_SENS_WITHIN_3")
        self.assertEqual(lab[(8, 2048)], "D_SENS_INVERTED_GE3")
        self.assertEqual(lab[(1, 512)], "D_SENS_UNRESOLVED")
        self.assertEqual(lab[(1, 2048)], "UNRESOLVED_N_BELOW_4")
        self.assertEqual(res["forecasts"]["F-N2-D3"], "FALSIFIED")


class TestPureRules(unittest.TestCase):
    def test_dsens_unpaired_rejected(self):
        with self.assertRaises(ValueError):
            A.dsens_x({1: 10.0, 2: 10.0}, {1: 9.0, 3: 9.0})
        self.assertEqual(A.dsens_label([0.05] * 3), "UNRESOLVED_N_BELOW_4")
        self.assertEqual(A.dsens_label([0.05, 0.05, 0.05, -0.05]), "D_SENS_UNRESOLVED")      # 4/4, not majority
        self.assertEqual(A.dsens_label([0.05, 0.05, 0.05, 0.02]), "D_SENS_UNRESOLVED")
        self.assertEqual(A.dsens_label([0.01, -0.02, 0.0, 0.029]), "D_SENS_WITHIN_3")

    def test_round_p50_108_pooled(self):
        mk = lambda D, r, iv, st="VALID": dict(kind="PROFILE", alone_D=D, b=8, c=2048, round=r, status=st,
                                               stats=dict(intervals=iv, s_per_p50=NW.quantile(iv, 0.5)))
        reps = [mk(108, 1, [1.0, 2.0, 3.0]), mk(108, 1, [10.0, 11.0]), mk(108, 2, [4.0, 5.0]),
                mk(108, 3, [7.0], st="D_UNREALIZED")]
        v = A.round_p50(reps, 108, 8, 2048)
        self.assertEqual(v, {1: 3.0, 2: 4.5})                      # pooled [1,2,3,10,11] -> 3; one boot alone ok

    def test_coloc_label_d5(self):                                  # D-5 4/4 rule (Job-EA, pure)
        self.assertEqual(A.coloc_label([0.0, 0.0, 0.01, 0.035]), "COLOC_UNRESOLVED")
        self.assertEqual(A.coloc_label([0.05] * 4), "COLOC_SLOWER_GE3")
        self.assertEqual(A.coloc_label([-0.05] * 4), "COLOC_FASTER_GE3")
        self.assertEqual(A.coloc_label([0.01, -0.01, 0.02, 0.0]), "COLOC_WITHIN_3")
        self.assertEqual(A.ds3_summary(["COLOC_WITHIN_3"] * 8), "DS3_WITHIN_3_ALL")
        self.assertEqual(A.ds3_summary(["COLOC_WITHIN_3"] * 7 + ["COLOC_SLOWER_GE3"]), "DS3_MISMATCH_OBSERVED_1")

    def test_signature_s1_b_finish_not_joined(self):                # R3-5 + D-4 (ii) 31 of 32
        recs = [dict(event="prefill_span_start", span_seq=1, timestamp_monotonic_s=1.0, extend_tokens=16384, n_new=2,
                     running_bs=8),
                dict(event="prefill_span_end", span_seq=1, timestamp_monotonic_s=2.0, n_done=2, running_bs=8),
                dict(event="decode_iteration", t_launch=2.1, bs=7)]
        s = SG.signature(recs, 0.5, 8, [], [])
        self.assertEqual((s["signature"], s["s1"]), ("NOT_JOINED", ["B_FINISH"]))
        recs[-1]["bs"] = 10
        self.assertEqual(SG.signature(recs, 0.5, 8, [], [])["signature"], "JOINED")
        pop = [dict(signature="JOINED")] * 31 + [dict(signature="NOT_JOINED")]
        self.assertEqual(SG.job_label(pop)[0], "COLOAD_JOINED_31_of_32")
        self.assertEqual(SG.job_label([])[0], "COLOAD_UNRESOLVED")

    def test_paired_ci_real_signature(self):
        ci = A.paired_ci({1: 0.0, 2: 0.1, 3: 0.0, 4: 0.1}, {1: 0.1, 2: 0.2, 3: 0.1, 4: 0.2})
        self.assertAlmostEqual(ci["mean_effect"], 0.1, places=9)
        self.assertEqual(ci["pairs"], 4.0)
        self.assertIsNone(A.paired_ci({1: 0.0}, {1: 0.1}))

    def test_quantile_is_numpy_linear(self):
        xs = [3.0, 1.0, 7.0, 2.0, 10.0]
        for q in (0.5, 0.95, 0.99):
            self.assertAlmostEqual(NW.quantile(xs, q), A.ANA.percentile(xs, q))
        try:
            import numpy as np
            self.assertAlmostEqual(NW.quantile(xs, 0.95), float(np.percentile(xs, 95)))
        except ImportError:
            pass

    def test_ctx_mismatch(self):                                     # §10-2 (Job-EA)
        self.assertFalse(NW.ctx_mismatch(2300, 2000, 2400, 2048))
        self.assertTrue(NW.ctx_mismatch(2400 + 205, 2000, 2400, 2048))
        self.assertFalse(NW.ctx_mismatch(2400 + 204, 2000, 2400, 2048))


if __name__ == "__main__":
    unittest.main()
