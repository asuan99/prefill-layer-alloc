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


# ===================================================================================================== Job-EA rows
UNITS_EA = n2_plan.units("EA", REG)


def ea_unit(kind, d, r=1):
    return next(u for u in UNITS_EA if u["name"] == f"r{r}_{kind}{d}")


def ea_cell_boot(kind="ON", d=44, b=8, c=2048, r=1, co_iters=20, joined_at=None, s2_bs=None, comp="ok",
                 payload_ct=1, abort=None, co_pair=None, a2_alone=200, alone_tau=0.02):
    """ONE synthetic E-alpha cell in the exact in-memory boot format of n2_window.load_boot (records, parsed marks,
    client record, wall pairs, log lines): burst span -> alone iterations through the A1 band -> coload_sent at the
    client trigger (ctx c + 71) -> ceil(K / 2) co spans (n_new 2, last 1) with `co_iters` co-located iterations each
    -> `a2_alone` alone iterations. Knobs: joined_at (an iteration right after that co span end with bs = b + 1, S1),
    s2_bs (one pre-filter iteration after the last co span with bs = s2_bs, S2), comp ('ok' / 'sum' = one co request
    missing / 'big' = an n_new 3 span), payload_ct (co completion_tokens), abort (a cell_abort reason)."""
    u = ea_unit(kind, d, r)
    cell = next(x for x in u["cells"] if (x["b"], x["c"]) == (b, c))
    K, c_in, cid = cell["K"], cell["c_in"], cell["cell_id"]
    pi_alone = PAIR[108] if kind == "OFF" else PAIR[d]
    pi_co = co_pair or PAIR[d]
    t0 = 1000.0
    recs, seq, span = [], [0], [0]
    t = [t0 + 0.05]

    def it(pair, in_flight, bs=b):
        seq[0] += 1
        recs.append(dict(event="decode_iteration", iter_seq=seq[0], t_launch=t[0], t_sync=t[0] + alone_tau / 2, bs=bs,
                         stream_idx=IDX_OF[pair], prefill_sms=pair[0], decode_sms=pair[1], prefill_in_flight=in_flight,
                         prefill_kernel_done_wait=False, dropped_events=0, timestamp_monotonic_s=t[0]))
        if seq[0] % 32 == 1 and bs == b:
            recs.append(dict(event="runtime_snapshot", phase="benchmark", timestamp_monotonic_s=t[0] + alone_tau / 4,
                             decode_iterations=seq[0], active_decode_sequences=b, decode_running_batch_size=b,
                             context_p50=c_in + seq[0], prefill_sms=pair[0], decode_sms=pair[1],
                             prefill_queue_depth=0, dropped_events=0))
        t[0] += alone_tau

    def spn(n_new, tokens, bs_run, qlen):
        span[0] += 1
        recs.append(dict(event="prefill_span_start", span_seq=span[0], timestamp_monotonic_s=t[0], n_new=n_new,
                         extend_tokens=tokens, running_bs=bs_run, queue_len=qlen, dropped_events=0,
                         prefill_sms=pi_co[0], decode_sms=pi_co[1]))
        t[0] += 0.001
        return span[0]

    def spn_end(s, n_done, bs_run, qlen):
        recs.append(dict(event="prefill_span_end", span_seq=s, timestamp_monotonic_s=t[0], n_done=n_done,
                         running_bs=bs_run, queue_len=qlen, dropped_events=0))
        t[0] += 0.001
    s = spn(b, b * c_in, 0, 0)
    spn_end(s, b, 0, 0)
    while c_in + seq[0] < c + TRIGGER:                                # alone until the client trigger ctx
        it(pi_alone, False)
    t_co = t[0]
    t[0] += 0.002
    groups = [2] * (K // 2) + ([1] if K % 2 else [])
    if comp == "sum":
        groups = groups[:-1]                                              # one co request never spans (sum < K)
    if comp == "big":
        groups = [3] + groups[2:] + ([1] if sum([3] + groups[2:]) < K else [])
    left = K
    for k, n in enumerate(groups):
        left -= n
        s = spn(n, n * 8192, b, left)
        for _ in range(co_iters):
            it(pi_co, True)
        spn_end(s, n, b, left)
        if joined_at == k:
            it(pi_co, True, bs=b + n)                                     # merged co request decodes (S1)
    if s2_bs is not None:
        it(pi_alone, False, bs=s2_bs)
    for _ in range(a2_alone):
        it(pi_alone, False)
    t_end = t[0] + 0.05
    recs.append(dict(event="runtime_snapshot", phase="benchmark", timestamp_monotonic_s=t_end + 1.0,
                     decode_iterations=seq[0], active_decode_sequences=0, decode_running_batch_size=0, context_p50=1,
                     prefill_sms=108, decode_sms=0, prefill_queue_depth=0, dropped_events=0))
    marks = [f"boot_plan {d} 4", f"cell_start {cid} {d} {b} {c} {t0!r}", f"coload_sent {cid} {t_co!r}"]
    if abort:
        marks.append(f"cell_abort {cid} {abort} {t_end - 0.01!r}")
    marks.append(f"cell_end {cid} {t_end!r}")
    cells, _calls, plan, aborts, issues = NW.RED.parse_marks(marks, {})
    assert not issues, issues
    log = [(math.floor(t0 + WALL_OFF) + k, f"Decode batch, #running-req: {b}, cuda graph: True")
           for k in range(int(t_end - t0) + 2)]
    client = dict(cell_id=cid, cancel_fired=True, requests=[dict(rid=f"{cid}-d{i}", n=10, prompt_tokens=c_in, done=False,
                                                                 finish=None) for i in range(b)],
                  co_requests=[dict(rid=f"{cid}-co{j}", n=1, completion_tokens=payload_ct, done=True, finish="length")
                               for j in range(K)])
    boot = dict(name=u["name"], records=recs, cells=cells, client={cid: client}, log="", log_lines=log,
                wall={("cell_start", cid): (t0 + WALL_OFF, t0), ("cell_end", cid): (t_end + WALL_OFF, t_end)},
                boot={}, cpu="")
    return boot, u, cell


TRIGGER = 71


class TestEAReplicateRows(unittest.TestCase):                                # §9-1 EALPHA rows, §7-2 (c)(f)
    def test_on_and_off_clean_cells_valid(self):                           # D-3: OFF A1/A2 (0,108), C (64,44)
        for kind in ("ON", "OFF"):
            boot, u, cell = ea_cell_boot(kind)
            r = NW.ealpha_replicate(boot, u, cell)
            self.assertEqual(r["windows"], {"A1": "VALID", "A2": "VALID", "C": "VALID"}, (kind, r["details"]))
            self.assertEqual(r["signature"]["signature"], "NOT_JOINED")
            self.assertEqual(r["desc"]["co_spans"], 8)
            self.assertEqual(r["desc"]["composition_problems"], [])
        self.assertEqual(r["values"]["A1"]["pe_iter_idx5"], 1.0)

    def test_c_value_window_first_128(self):                                # §6-1 (A'): C = first 128 co-located
        boot, u, cell = ea_cell_boot("ON", co_iters=25)
        r = NW.ealpha_replicate(boot, u, cell)
        self.assertEqual((r["desc"]["n_colocated_all"], r["window_len"]["C"]), (200, 128))

    def test_co_span_composition(self):                                     # §7-2 (c): sum n_new == K, n_new <= 2
        for comp in ("sum", "big"):
            boot, u, cell = ea_cell_boot("ON", comp=comp)
            r = NW.ealpha_replicate(boot, u, cell)
            self.assertEqual(r["windows"]["C"], "WINDOW_SHORT", comp)
            self.assertIn("co span composition", r["details"]["C"])
            self.assertEqual(r["windows"]["A1"], "VALID")

    def test_joined_s1_and_s2_prefilter(self):                              # §7-2 (f) + §9-1 JOINED -> BS_UNREALIZED
        boot, u, cell = ea_cell_boot("OFF", joined_at=2)
        r = NW.ealpha_replicate(boot, u, cell)
        self.assertEqual(r["signature"]["signature"], "JOINED")
        self.assertEqual(r["windows"]["C"], "BS_UNREALIZED")
        self.assertIn("JOINED", r["details"]["C"])
        boot, u, cell = ea_cell_boot("OFF", s2_bs=9)                       # bs > b only AFTER the last co span
        r = NW.ealpha_replicate(boot, u, cell)
        self.assertEqual(r["signature"]["s2_bs_gt_b"], 1)                  # pre-filter A2 region (filtered A2 = 0)
        self.assertEqual(r["windows"]["C"], "BS_UNREALIZED")

    def test_payload_rows(self):                                            # F3 -> COLOAD_PAYLOAD_INVALID
        boot, u, cell = ea_cell_boot("ON", payload_ct=2)
        self.assertEqual(NW.ealpha_replicate(boot, u, cell)["status"], "COLOAD_PAYLOAD_INVALID")
        boot, u, cell = ea_cell_boot("ON", abort="coload_payload_invalid")
        self.assertEqual(NW.ealpha_replicate(boot, u, cell)["status"], "COLOAD_PAYLOAD_INVALID")
        boot, u, cell = ea_cell_boot("ON", abort="client_exception")
        self.assertEqual(NW.ealpha_replicate(boot, u, cell)["status"], "BENCH_INVALID")

    def test_coloc_pair_leak_is_coloc_unrealized(self):                    # §7-2 (b) 0.95 over bs == b in W_co
        boot, u, cell = ea_cell_boot("OFF", co_pair=PAIR[108])
        r = NW.ealpha_replicate(boot, u, cell)
        self.assertEqual(r["windows"]["C"], "CO_WINDOW_SHORT")             # first-128 filter needs the target pair


def ea_rep(kind, rnd, a1=(10.0, 2000.0), a2=(10.5, 2400.0), cmed=10.6, xc=2200.0, st=None, d=44, b=8, c=2048):
    """A replicate dict as n2_window.ealpha_replicate returns it (only the fields rho assembly reads)."""
    w = {**dict(A1="VALID", A2="VALID", C="VALID"), **(st or {})}
    return dict(cell_key=f"{d},{b},{c}", boot_kind=kind, round=rnd, boot=f"r{rnd}_{kind}{d}", windows=w,
                status="MEASURED", values=dict(A1=dict(s_per_median=a1[0], ctx_mid=a1[1]),
                                               A2=dict(s_per_median=a2[0], ctx_mid=a2[1]),
                                               C=dict(s_per_median=cmed, ctx_mid=xc)), desc={})


class TestRhoAssembly(unittest.TestCase):                                    # H15-3: rho_off across two boots
    def test_value_is_c_median_over_on_a_on(self):
        on, off = ea_rep("ON", 1), ea_rep("OFF", 1, a1=(99.0, 1.0), a2=(99.0, 1.0), cmed=11.0, xc=2300.0)
        p = A.rho_pair(on, off, 2048)                                       # OFF A1/A2 never enter rho_off (D-3)
        a_on = NW.RED.a_on_interp(10.0, 2000.0, 10.5, 2400.0, 2300.0)
        self.assertTrue(p["valid"] and p["aon_contained"])
        self.assertAlmostEqual(p["ln_rho"], math.log(11.0 / a_on))
        self.assertAlmostEqual(A.rho_pair(on, on, 2048)["ln_rho"], math.log(10.6 / NW.RED.a_on_interp(
            10.0, 2000.0, 10.5, 2400.0, 2200.0)))

    def test_cross_boot_containment_is_recorded_not_a_label_input(self):   # §10-2 extrapolation within 0.1 c
        on, off = ea_rep("ON", 1), ea_rep("OFF", 1, cmed=11.0, xc=2500.0)          # 100 > 0 outside, < 0.1 c = 204.8
        p = A.rho_pair(on, off, 2048)
        self.assertTrue(p["valid"])
        self.assertFalse(p["aon_contained"])
        self.assertAlmostEqual(p["a_on"], 10.0 + 0.5 * (2500.0 - 2000.0) / 400.0)   # the same line, extrapolated
        with self.assertRaises(NW.RED.MeasurementRejected):                 # the sealed function itself refuses
            NW.RED.a_on_interp(10.0, 2000.0, 10.5, 2400.0, 2500.0)
        reps = [ea_rep("ON", r) for r in (1, 2, 3, 4)] + [ea_rep("OFF", r, cmed=11.0, xc=2500.0 if r == 1 else 2200.0)
                                                         for r in (1, 2, 3, 4)]
        cr = A.ea_cells(reps)["44,8,2048"]["rho_off"]
        self.assertEqual((cr["n"], cr["aon_containment_outside"]), (4, 1))
        self.assertEqual(cr["missing_hist"]["AON_CONTAINMENT(valid, extrapolated)"], 1)
        self.assertEqual(cr["label"], "COLOC_SLOWER_GE3")
        self.assertEqual(cr["label_if_containment_rejects"], "UNRESOLVED_N_BELOW_4")
        self.assertAlmostEqual(cr["aon_containment_rate"], 0.25)

    def test_ctx_mismatch_and_validity_order(self):                        # §9-1 rho_off_valid order
        on = ea_rep("ON", 1)
        self.assertEqual(A.rho_pair(on, ea_rep("OFF", 1, xc=2400.0 + 205.0), 2048)["reason"], "CTX_MISMATCH")
        self.assertTrue(A.rho_pair(on, ea_rep("OFF", 1, xc=2400.0 + 204.0), 2048)["valid"])
        bad_on = ea_rep("ON", 1, st=dict(A2="WINDOW_SHORT"))
        p = A.rho_pair(bad_on, ea_rep("OFF", 1, xc=9999.0), 2048)              # A_on invalid wins over CTX_MISMATCH
        self.assertTrue(p["reason"].startswith("A_on invalid"))
        p = A.rho_pair(on, ea_rep("OFF", 1, st=dict(C="CO_WINDOW_SHORT"), xc=9999.0), 2048)
        self.assertEqual(p["reason"], "C CO_WINDOW_SHORT")
        self.assertEqual(A.rho_pair(None, ea_rep("OFF", 1), 2048)["reason"], "replicate missing")

    def test_off_a1_a2_do_not_gate_rho_off(self):                           # D-3: OFF A1/A2 are descriptive
        reps = [ea_rep("ON", r) for r in (1, 2, 3, 4)] + [ea_rep("OFF", r, st=dict(A1="D_UNREALIZED", A2="WINDOW_SHORT"),
                                                         cmed=10.0) for r in (1, 2, 3, 4)]
        cr = A.ea_cells(reps)["44,8,2048"]
        self.assertEqual(cr["rho_off"]["n"], 4)
        self.assertEqual(cr["rho_off"]["label"], "COLOC_WITHIN_3")

    def test_labels_ci_and_ds3(self):                                       # §10-3 4/4 + descriptive CI, §10-4
        reps = []
        for r in (1, 2, 3, 4):
            reps += [ea_rep("ON", r, cmed=10.25 * 1.05), ea_rep("OFF", r, cmed=10.25 * (1.10 + 0.01 * r))]
        cr = A.ea_cells(reps)["44,8,2048"]
        self.assertEqual(cr["rho_off"]["label"], "COLOC_SLOWER_GE3")
        self.assertEqual(cr["rho_on"]["label"], "COLOC_SLOWER_GE3")
        self.assertEqual(cr["rho_off"]["ci"]["pairs"], 4.0)
        self.assertAlmostEqual(cr["rho_off"]["ci"]["mean_effect"], statistics_mean(cr["rho_off"]["ln_rho"]))
        reps[1] = dict(reps[1], windows=dict(reps[1]["windows"], C="COLOC_UNREALIZED"))
        self.assertEqual(A.ea_cells(reps)["44,8,2048"]["rho_off"]["label"], "UNRESOLVED_N_BELOW_4")
        labels = ["COLOC_WITHIN_3"] * 7 + ["COLOC_FASTER_GE3"]
        self.assertEqual(A.ds3_summary(labels), "DS3_MISMATCH_OBSERVED_1")


def statistics_mean(xs):
    return sum(xs) / len(xs)


class TestEAJobLabels(unittest.TestCase):                                    # NEG_CONTROL_EA, COLOAD, forecasts
    def _r(self, kind, cid, a1=1.0, a2=1.0, st=None, meas="MEASURED", co=True, sig="NOT_JOINED", s3=("MERGED",)):
        w = {**dict(A1="VALID", A2="VALID", C="VALID"), **(st or {})}
        return dict(cell_id=cid, cell_key="44,8,2048", boot_kind=kind, round=1, status=meas, windows=w,
                    window_len=dict(A1=128, A2=128, C=128), co_span_present=co,
                    values=dict(A1=dict(pe_iter_idx5=a1), A2=dict(pe_iter_idx5=a2)),
                    signature=dict(signature=sig, s3=list(s3)), desc=dict(pe_coloc_iter=1.0))

    def test_neg_control_ea(self):                                          # §11-3 on the OFF E-alpha windows
        ok = [self._r("OFF", "a"), self._r("ON", "b", a1=0.0)]               # ON windows never enter
        self.assertEqual(A.neg_control_ea(ok)["label"], "NEG_CONTROL_EA_HELD")
        bad = [self._r("OFF", "a", a2=0.90, st=dict(A2="D_UNREALIZED"))]    # population is not the VALID set
        self.assertEqual(A.neg_control_ea(bad)["label"], "NEG_CONTROL_EA_FAILED")
        rec = [self._r("OFF", "a", a1=0.5, st=dict(A1="RECORDER_UNPROVEN", A2="RECORDER_UNPROVEN"))]
        self.assertEqual(A.neg_control_ea(rec)["label"], "NEG_CONTROL_EA_UNRESOLVED")
        self.assertEqual(A.neg_control_ea([self._r("OFF", "a", meas="RETRACT")])["label"], "NEG_CONTROL_EA_UNRESOLVED")

    def test_coload_population_is_measured_set(self):                       # D-4 (ii): not the VALID set
        pop = [self._r("ON", f"c{i}", sig="JOINED", st=dict(C="BS_UNREALIZED")) for i in range(31)]
        pop.append(self._r("OFF", "x"))
        pop.append(self._r("OFF", "y", meas="CG_UNPROVEN", sig="JOINED"))   # outside M
        pop.append(self._r("OFF", "z", co=False, sig="JOINED"))             # no co span: outside M
        col = A.coload_label(pop)
        self.assertEqual((col["label"], col["k"], col["n"]), ("COLOAD_JOINED_31_of_32", 31, 32))

    def test_f7e_and_f8(self):
        led = dict(ledger=[dict(unit=u["name"], status="ADMITTED") for u in UNITS_EA], stopped=None, job_abort=None)
        self.assertEqual(A._f7(led, 16), "HELD")
        self.assertEqual(A._f7(dict(led, stopped="crash"), 16), "UNRESOLVED")
        self.assertEqual(A._f7(dict(led, ledger=led["ledger"][:15]), 16), "FALSIFIED")
        cells = A.ea_cells([])
        f = A.forecasts_ea(cells, [], dict(label="COLOAD_NOT_JOINED", s3_merged=3, s3_b=1, s3_total=4), led, 16)
        self.assertEqual((f["F-N2-8"], f["F-N2-7E"], f["F-N2-9"]), ("HELD", "HELD", "UNRESOLVED"))
        f = A.forecasts_ea(cells, [], dict(label="COLOAD_NOT_JOINED", s3_merged=0, s3_b=4, s3_total=4), led, 16)
        self.assertEqual(f["F-N2-8"], "FALSIFIED")
        f = A.forecasts_ea(cells, [], dict(label="COLOAD_JOINED_1_of_4", s3_merged=4, s3_b=0, s3_total=4), led, 16)
        self.assertEqual(f["F-N2-8"], "FALSIFIED")



# ===================================================================================================== H18-1 / H18-4
A_ON_AT = {2100.0: 10.125, 2200.0: 10.25, 2300.0: 10.375}       # the default ea_rep A1 / A2 line at these ctx


def ea_fc_rep(kind, rnd, key, ln=math.log(1.05), pe=1.0, st=None, half_gap=0.0):
    """One E-alpha replicate for the forecast tests: C at ctx 2200 with ln rho = `ln` against the ON line, desc with
    pe_coloc_iter = `pe` and front / back halves (ctx 2100 / 2300) whose ln rho differ by `half_gap`."""
    d, b, c = key
    r = ea_rep(kind, rnd, cmed=A_ON_AT[2200.0] * math.exp(ln), st=st, d=d, b=b, c=c)
    r["desc"] = dict(pe_coloc_iter=pe,
                     front=dict(n=64, ctx_mid=2100.0, s_per_median=A_ON_AT[2100.0] * 1.05),
                     back=dict(n=64, ctx_mid=2300.0, s_per_median=A_ON_AT[2300.0] * 1.05 * math.exp(half_gap)))
    return r


def ea_fc_job(over=None):
    """All 8 E-alpha cells x rounds 1-4 x {ON, OFF}; over[(key, kind, rnd)] = ea_fc_rep kwargs (None = replicate
    absent)."""
    over = over or {}
    reps = []
    for key in A.EA_CELL_KEYS:
        for rnd in (1, 2, 3, 4):
            for kind in ("ON", "OFF"):
                kw = over.get((key, kind, rnd), {})
                if kw is not None:
                    reps.append(ea_fc_rep(kind, rnd, key, **kw))
    return reps


COL_HELD = dict(label="COLOAD_NOT_JOINED", s3_merged=3, s3_b=1, s3_total=4)


def fc(reps):
    cells = A.ea_cells(reps)
    led = dict(ledger=[dict(unit=u["name"], status="ADMITTED") for u in UNITS_EA], stopped=None, job_abort=None)
    return A.forecasts_ea(cells, reps, COL_HELD, led, 16), cells


class TestEAForecasts(unittest.TestCase):                                    # H18-1: §18-2-d #26, every branch
    K = A.EA_CELL_KEYS

    def test_clean_job_all_held(self):
        f, _ = fc(ea_fc_job())
        self.assertEqual({k: f[k] for k in ("F-N2-1", "F-N2-2", "F-N2-3", "F-N2-9")},
                         {"F-N2-1": "HELD", "F-N2-2": "HELD", "F-N2-3": "HELD", "F-N2-9": "HELD"})

    def test_f1_branches(self):                                             # OFF MEASURED pe_coloc_iter, any < 0.99
        k0, k1 = self.K[0], self.K[1]
        self.assertEqual(fc(ea_fc_job({(k0, "OFF", 2): dict(pe=0.99)}))[0]["F-N2-1"], "HELD")      # boundary: 0.99 ok
        self.assertEqual(fc(ea_fc_job({(k0, "OFF", 2): dict(pe=0.989)}))[0]["F-N2-1"], "FALSIFIED")
        self.assertEqual(fc(ea_fc_job({(k0, "ON", r): dict(pe=0.5) for r in (1, 2, 3, 4)}))[0]["F-N2-1"],
                         "HELD")                                            # ON replicates are not the population
        miss = {(k1, "OFF", r): dict(pe=None) for r in (1, 2, 3, 4)}        # a cell without any value
        self.assertEqual(fc(ea_fc_job(miss))[0]["F-N2-1"], "UNRESOLVED")
        self.assertEqual(fc(ea_fc_job({**miss, (k0, "OFF", 1): dict(pe=0.5)}))[0]["F-N2-1"], "FALSIFIED")
        notmeas = ea_fc_job()                                               # a non-MEASURED OFF replicate's value
        for r in notmeas:                                                    # never enters
            if r["cell_key"] == "%d,%d,%d" % k0 and r["boot_kind"] == "OFF" and r["round"] == 3:
                r.update(status="RETRACT", desc=dict(r["desc"], pe_coloc_iter=0.1))
        self.assertEqual(fc(notmeas)[0]["F-N2-1"], "HELD")

    def test_f2_branches(self):                                             # both-valid rounds only, ln 1.03
        k0, k1 = self.K[0], self.K[1]
        lo, hi = math.log(1.05), math.log(1.05) + math.log(1.03)
        self.assertEqual(fc(ea_fc_job({(k0, "OFF", 2): dict(ln=hi + 1e-6)}))[0]["F-N2-2"], "FALSIFIED")
        self.assertEqual(fc(ea_fc_job({(k0, "OFF", 2): dict(ln=hi - 1e-6)}))[0]["F-N2-2"], "HELD")
        self.assertGreater(hi - 1e-6, lo)
        # a round whose rho_on is invalid (ON C not VALID) does not enter -- even with a huge rho_off
        one = {(k0, "ON", 2): dict(st=dict(C="WINDOW_SHORT")), (k0, "OFF", 2): dict(ln=5.0)}
        f, cells = fc(ea_fc_job(one))
        self.assertTrue(cells["%d,%d,%d" % k0]["rho_off"]["pairs"]["2"]["valid"])
        self.assertFalse(cells["%d,%d,%d" % k0]["rho_on"]["pairs"]["2"]["valid"])
        self.assertEqual(f["F-N2-2"], "HELD")
        # a cell with no both-valid round -> UNRESOLVED (rho_off valid everywhere, rho_on nowhere)
        none = {(k1, "ON", r): dict(st=dict(C="CO_WINDOW_SHORT")) for r in (1, 2, 3, 4)}
        self.assertEqual(fc(ea_fc_job(none))[0]["F-N2-2"], "UNRESOLVED")
        self.assertEqual(fc(ea_fc_job({**none, (k0, "OFF", 1): dict(ln=1.0)}))[0]["F-N2-2"], "FALSIFIED")

    def _f3(self, h, u):
        """h cells with rho_on n = 4 and 4/4 > 0; u cells with n < 4 (3 positive rounds); the rest n = 4 with one
        round <= 0."""
        over = {}
        for i, key in enumerate(self.K):
            if i < h:
                continue
            if i < h + u:
                over[(key, "ON", 4)] = dict(st=dict(C="WINDOW_SHORT"))
            else:
                over[(key, "ON", 4)] = dict(ln=-0.01)
        return fc(ea_fc_job(over))[0]["F-N2-3"]

    def test_f3_boundaries(self):                                           # h >= 6 / h + u < 6 / else
        self.assertEqual(self._f3(8, 0), "HELD")
        self.assertEqual(self._f3(6, 0), "HELD")
        self.assertEqual(self._f3(6, 2), "HELD")
        self.assertEqual(self._f3(5, 1), "UNRESOLVED")
        self.assertEqual(self._f3(5, 0), "FALSIFIED")
        self.assertEqual(self._f3(4, 2), "UNRESOLVED")
        self.assertEqual(self._f3(4, 1), "FALSIFIED")
        self.assertEqual(self._f3(0, 0), "FALSIFIED")
        zero = {(key, "ON", 1): dict(ln=0.0) for key in self.K[:3]}         # ln rho_on = 0 is not > 0
        self.assertEqual(fc(ea_fc_job(zero))[0]["F-N2-3"], "FALSIFIED")

    def test_f9_branches(self):                                             # every valid pair with both halves
        k0 = self.K[0]
        self.assertEqual(fc(ea_fc_job({(k0, "OFF", 3): dict(half_gap=math.log(1.03) + 1e-6)}))[0]["F-N2-9"],
                         "FALSIFIED")
        self.assertEqual(fc(ea_fc_job({(k0, "OFF", 3): dict(half_gap=math.log(1.03) - 1e-6)}))[0]["F-N2-9"], "HELD")
        self.assertEqual(fc(ea_fc_job({(k0, "ON", 3): dict(half_gap=-(math.log(1.03) + 1e-6))}))[0]["F-N2-9"],
                         "FALSIFIED")                                       # rho_on halves count too (|.|)
        bad = ea_fc_job({(k0, "OFF", 3): dict(half_gap=1.0, st=dict(C="WINDOW_SHORT"))})
        self.assertEqual(fc(bad)[0]["F-N2-9"], "HELD")                      # an invalid pair's halves never enter
        nohalf = ea_fc_job()
        for r in nohalf:
            r["desc"].pop("back")
        self.assertEqual(fc(nohalf)[0]["F-N2-9"], "UNRESOLVED")


class TestEALabelText(unittest.TestCase):                                     # H18-4: N2C-89 per judgment quantity
    def test_containment_rate_and_strict_label_per_rho(self):
        k0 = A.EA_CELL_KEYS[0]
        key = "%d,%d,%d" % k0
        reps = ea_fc_job()
        for r in reps:                                                       # OFF round 1 C outside [A1, A2]
            if r["cell_key"] == key and r["boot_kind"] == "OFF" and r["round"] == 1:
                r["values"]["C"]["ctx_mid"] = 2440.0
        cells = A.ea_cells(reps)
        ro, rn = cells[key]["rho_off"], cells[key]["rho_on"]
        self.assertEqual((ro["aon_containment_rate"], ro["label_if_containment_rejects"]), (0.25, "UNRESOLVED_N_BELOW_4"))
        self.assertEqual((rn["aon_containment_rate"], rn["label_if_containment_rejects"]), (0.0, "COLOC_SLOWER_GE3"))
        res = dict(job_name="t", gpu_uuid="G", rule_seal_digest="0" * 64, cells=cells, replicates=[], K=REG["K"],
                   kappa_star_worst={}, ds3="DS3_UNRESOLVED", coload=dict(label="COLOAD_NOT_JOINED", k=0, n=1,
                                                                          s3_merged=0, s3_total=0),
                   neg_control_ea=dict(label="NEG_CONTROL_EA_HELD", n=1, min_pe_idx5=1.0), forecasts={})
        txt = A.render_text_ea(res)
        block = txt.split(f"E-α 셀 ({k0[0]}, {k0[1]}, {k0[2]})")[1].split("E-α 셀 (")[0]
        self.assertIn("ρ_off 외삽 짝(AON_CONTAINMENT) 1/4, aon_containment_rate = 0.250, 엄격 포함 시 라벨 "
                      "label_if_containment_rejects = `UNRESOLVED_N_BELOW_4`", block)
        self.assertIn("ρ_on 외삽 짝(AON_CONTAINMENT) 0/4, aon_containment_rate = 0.000, 엄격 포함 시 라벨 "
                      "label_if_containment_rejects = `COLOC_SLOWER_GE3`", block)
        req = [ln for ln in txt.splitlines() if ln.startswith("필수 병기(§16-2)")]
        self.assertEqual(len(req), 1)
        self.assertIn("N2C-87", req[0])
        self.assertIn("N2C-89", req[0])
        empty = A.ea_cells([])[key]["rho_off"]                              # no valid pair: rate NA, not 0
        self.assertIsNone(empty["aon_containment_rate"])
        self.assertIn("aon_containment_rate = NA", A.render_text_ea(dict(res, cells=A.ea_cells([]))))


if __name__ == "__main__":
    unittest.main()
