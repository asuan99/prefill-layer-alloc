#!/usr/bin/env python3
"""he0_followup.py self-test v2 + recomputation of two refuted FOLLOWUP numbers (GPU 0, 2026-10-02, result-analyst).

Why (AUDIT_FOLLOWUP_VERDICT_2026-10-02.md section 4): the self-test inside ``he0_followup.py`` is a partial identity
-- "C1 exact mixture R=0" and the "drop component" mutant never call ``c1()``; the ">= -> >" mutant changes the
argument, not the code under test; ``c1``, ``running_and_queue`` occupancy, ``c4_run``, the ``stall_events``
classifier, ``index_paired_attribution`` and ``zero_transition_contrast`` are untested.

This file does NOT modify ``he0_followup.py``.  Code under test is the module itself.  Mutants are produced by
exact-once source substitution of the WHOLE module text executed in a fresh namespace (so callers such as
``c4_run`` resolve the mutated callee); the unsubstituted re-exec is checked to give identical outputs.

=========================================================================================
PRE-REGISTERED (fixed before the first run of this file)
=========================================================================================
Synthetic: he0_selftest_v2.synth_engine (HI rate 12, D44, delay=real), extended with default-off server-stall
injection and synthetic server log lines (Decode every 40 iterations / POST at receipt / Prefill at admission,
1-s timestamps).  Stall start 5.3 s after the first send, duration 3.5 s.
T-S  stall_events classifier.  For each scenario, among events overlapping the injected window take the one with
     the most gapped requests; PASS iff its class prefix (text before "(") equals the expected one:
       server_wide   (scheduler + HTTP silent)        -> SERVER_WIDE
       server_sched  (scheduler silent, HTTP alive)   -> SERVER_SCHEDULER
       client_recv   (client receive stall only)      -> CLIENT_RECEIVE
       client_loop   (client send + receive stall)    -> CLIENT
     and for no_stall / client_send_only: PASS iff NO event overlaps the window.
T-C1 c1() on a hand-built known mixture: 4 static arms, 4 runs each, deviations (-0.1,+0.1,+0.05,-0.05) around
     a known base per metric (pooled SD = sqrt(0.025/3)); bind_gate runs = exact residency-weighted mixture (R=0),
     bind_nogate = mixture - 3 pooled SD, slo = mixture + 3 pooled SD.  PASS iff pooled SD, mean_R/SD (0, -3, +3)
     match to 1e-9 for every metric and both placements, verdict labels are positioning / intrinsic / weakens,
     and exclude=[job] drops that job.
T-OCC running_and_queue on a hand-built 3-request reconstruction with known step functions:
     occupancy>=3 = 0.5 s, >=2 = 0.7 s; running>=3 = 0.3 s; queue>=1 = 0.25 s (tolerance 1e-9).
T-C4 c4_run on the synthetic HI round vs synthetic server truth (occupancy = admitted and not finished):
     |recon occupancy>=46 time share - truth| <= 0.05.
T-ZT zero_transition_contrast on hand-built runs: diff_pct_of_d24 exact (1e-9) and only 0-transition runs used.
T-IPA index_paired_attribution on a hand-built 10-slot round: pre/touched/post slot counts and target failures exact.
Mutants (KILLED iff any test that exercises the mutated function fails):
  M-S1 swap client/server top branch   : "if dec_in >= 0.5 * exp_dec:" -> "if dec_in < 0.5 * exp_dec:"
  M-S2 swap SERVER_WIDE <-> SERVER_SCHEDULER labels
  M-S3 swap CLIENT <-> CLIENT_RECEIVE labels
  M-O  occupancy start s -> e (running_and_queue occ step function)
  M-C1a c1 static arm mean -> max
  M-C1b c1 static arm means permuted (16->24->34->44->16)
Overall: every control PASS and every mutant KILLED -> self-test PASS.  Any SURVIVED is reported as-is.

HARNESS CORRECTION (after first run, before any result was used): the first run's control FAILED one T-S
scenario (client_recv -> got CLIENT).  With the rule above, every mutant then "failed T-S" trivially, so kills
credited via T-S were spurious.  Mutation scoring is therefore done per UNIT (each T-S scenario is a unit, each
other test is a unit): a mutant is KILLED iff some unit that PASSES in the control FAILS under the mutant.  The
pre-registered overall verdict is unchanged in definition (control must pass every unit -> it does not).
POST-HOC scenario (added after seeing the failure, reported separately, not part of the overall verdict):
  client_recv_after_sends : receive stall placed after the last send (22.3 s) -> expected CLIENT_RECEIVE.

Recomputation (section 1 of the verdict): zero-transition bind+GATE (856930/856941/857112) vs d24, with and
without 859006 (FOLLOWUP's own exclusion); d44 gpu39 (852342) vs gpu38 (856889-856891).  HI phase, durations
summed (analyze.load_bench_serving_rounds).  Welch t / df computed here (analyze.py has no Welch helper; formula
is the textbook one, stated for audit).  Unpaired bootstrap via analyze.unpaired_bootstrap_ci (seed 1, 10000).
=========================================================================================

Usage: python3 he0_followup_selftest_v2.py [--output he0_followup_selftest_v2_2026-10-02.json]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import he0_realized as H  # noqa: E402
import he0_followup as F  # noqa: E402
import he0_selftest_v2 as ST  # noqa: E402
from he0_realized import SRC, A, length, union, intersect, clip  # noqa: E402

STALL_T0 = 5.3
STALL_DUR = 3.5
C4_TOL = 0.05
F_SRC = Path(F.__file__).read_text()


def module_variant(subs):
    src = F_SRC
    for old, new in subs:
        c = src.count(old)
        assert c == 1, ("substitution must match exactly once", old, c)
        src = src.replace(old, new)
    ns = {"__name__": "he0_followup_variant", "__file__": F.__file__}
    exec(compile(src, "<he0_followup variant>", "exec"), ns)
    return ns


MUTANTS = {
    "M-S1_swap_client_server_branch": [("if dec_in >= 0.5 * exp_dec:", "if dec_in < 0.5 * exp_dec:")],
    "M-S2_swap_SERVER_WIDE_SCHEDULER": [('cls = "SERVER_WIDE(', 'cls = "@@SW('), ('cls = "SERVER_SCHEDULER(', 'cls = "SERVER_WIDE('),
                                        ('cls = "@@SW(', 'cls = "SERVER_SCHEDULER(')],
    "M-S3_swap_CLIENT_CLIENT_RECEIVE": [('cls = "CLIENT(', 'cls = "@@CL('), ('cls = "CLIENT_RECEIVE(', 'cls = "CLIENT('),
                                        ('cls = "@@CL(', 'cls = "CLIENT_RECEIVE(')],
    "M-O_occupancy_start_s_to_e": [("occ = step_fn([(s[bidx[i]], +1) for i in range(n)] + [(max(fin[i], s[bidx[i]]), -1) for i in range(n)])",
                                    "occ = step_fn([(e[bidx[i]], +1) for i in range(n)] + [(max(fin[i], e[bidx[i]]), -1) for i in range(n)])")],
    "M-C1a_static_mean_to_max": [("means = {d: statistics.fmean([x[2] for x in v]) for d, v in dv.items()}",
                                  "means = {d: max(x[2] for x in v) for d, v in dv.items()}")],
    "M-C1b_static_means_permuted": [("pred = sum(wd * means[d] for d, wd in w.items())",
                                     "pred = sum(wd * means[{16: 24, 24: 34, 34: 44, 44: 16}[d]] for d, wd in w.items())")],
}


# ----------------------------------------------------------------------------------- synthetic R
def synth_R(**kw):
    sy = ST.synth_engine("HI", 44, "real", **kw)
    rec = H.reconstruct_round(sy["batches"], sy["A_rel"], sy["ttft"], sy["itls"], sy["duration"])
    rec["A_rel"] = sy["A_rel"]
    rd = {"phase": "HI", "round": 0, "o": None, "batches": sy["batches"], "rec": rec, "ttft": sy["ttft"], "itls": sy["itls"]}
    R = {"job": -1, "S": {"ctl": [], "feas": [], "prefill": [], "decode": [(t, r) for t, r, _g, _q in sy["decode_lines"]]},
         "rounds": [rd], "post": sy["post"], "dthr": sy["decode_lines"], "eng": sy["eng"], "node": "synth"}
    return R, sy


def stall_scenarios():
    base = ST.synth_engine("HI", 44, "real")
    s0 = float(base["send"][0]); t0 = s0 + STALL_T0
    idx = int(np.searchsorted(base["send"], t0))
    return {
        "no_stall": ({}, None, (t0, t0 + STALL_DUR)),
        "server_wide": ({"server_stall": (STALL_T0, STALL_DUR, "wide")}, "SERVER_WIDE", (t0, t0 + STALL_DUR)),
        "server_sched": ({"server_stall": (STALL_T0, STALL_DUR, "sched")}, "SERVER_SCHEDULER", (t0, t0 + STALL_DUR)),
        "client_recv": ({"recv_stall": (t0, STALL_DUR)}, "CLIENT_RECEIVE", (t0, t0 + STALL_DUR)),
        "client_loop": ({"recv_stall": (t0, STALL_DUR), "send_stall": (idx, STALL_DUR)}, "CLIENT", (t0, t0 + STALL_DUR)),
        "client_send_only": ({"send_stall": (idx, STALL_DUR)}, None, (t0, t0 + STALL_DUR)),
        "POSTHOC_client_recv_after_sends": ({"recv_stall": (s0 + 22.3, STALL_DUR)}, "CLIENT_RECEIVE", (s0 + 22.3, s0 + 22.3 + STALL_DUR)),
    }


_SR_CACHE = {}


def t_stall(ns):
    out = {}
    for name, (kw, expect, win) in stall_scenarios().items():
        if name not in _SR_CACHE:
            _SR_CACHE[name] = synth_R(**kw)
        R, _sy = _SR_CACHE[name]
        evs = ns["stall_events"](R)
        ov = [e for e in evs if min(e["t_start"] + e["dur_s"], win[1] + 1.0) > max(e["t_start"], win[0] - 1.0)]
        if expect is None:
            ok = not ov
            got = [e["class"].split("(")[0] for e in ov]
        else:
            best = max(ov, key=lambda e: e["n_req_gapped"]) if ov else None
            got = best["class"].split("(")[0] if best else None
            ok = got == expect
        out[name] = {"expected": expect, "got": got, "pass": ok, "n_events_total": len(evs),
                     "event": None if expect is None or not ov else {k: best[k] for k in ("dur_s", "n_req_gapped", "frac_active_gapped",
                                                                                          "interior_log_seconds", "decode_lines_interior",
                                                                                          "decode_lines_expected_at_round_rate", "post_lines_interior",
                                                                                          "max_http_receipt_lag_s_of_requests_sent_during_gap",
                                                                                          "client_send_lateness_jump_s")}}
    return out


# ----------------------------------------------------------------------------------- C1 known mixture
METRICS = [f"{ph}:{m}" for ph, m in F.C1_METRICS]
DEV = (-0.1, 0.1, 0.05, -0.05)
PSD = math.sqrt(sum(d * d for d in DEV) / 3)
BASE = {16: 1.0, 24: 2.0, 34: 3.0, 44: 4.0}


def c1_fixture():
    runs_by_arm, gp = {}, {}
    job = 1000
    def put(jb, val_fn):
        gp[jb] = {}
        for ph, m in F.C1_METRICS:
            gp[jb].setdefault(ph, {})[m] = val_fn(ph, m)
    for d in (16, 24, 34, 44):
        arm = f"d{d}"; runs_by_arm[arm] = []
        for k, dv in enumerate(DEV):
            job += 1
            runs_by_arm[arm].append({"job": job, "node": "gpuA" if k % 2 else "gpuB", "rounds": []})
            put(job, lambda ph, m, d=d, dv=dv: BASE[d] + (0.01 * len(m) if ph == "HI" else 0.0) + dv)
    mixes = [{24: 7.5, 34: 2.5}, {24: 5.0, 34: 5.0}, {16: 3.0, 24: 3.0, 44: 4.0}]
    shift = {"bind_gate": 0.0, "bind_nogate": -3.0, "slo": 3.0}
    for arm in H.DYNAMIC:
        runs_by_arm[arm] = []
        for mix in mixes:
            job += 1
            rounds = [{"phase": ph, "co_split_s": {"earliest": {str(d): s for d, s in mix.items()},
                                                   "latest": {str(d): s for d, s in mix.items()}}} for ph in ("LO", "HI")]
            runs_by_arm[arm].append({"job": job, "node": "gpuA", "rounds": rounds})
            T = sum(mix.values())
            put(job, lambda ph, m, mix=mix, T=T, a=arm: sum(s / T * (BASE[d] + (0.01 * len(m) if ph == "HI" else 0.0)) for d, s in mix.items()) + shift[a] * PSD)
    return runs_by_arm, gp


def t_c1(ns):
    rba, gp = c1_fixture()
    o = ns["c1"](rba, gp)
    fails = []
    for key in METRICS:
        if abs(o["pooled_sd"][key]["pooled_sd"] - PSD) > 1e-9:
            fails.append(("psd", key))
        for arm, z_exp, lab in (("bind_gate", 0.0, "positioning"), ("bind_nogate", -3.0, "intrinsic"), ("slo", 3.0, "weakens")):
            for place in ("earliest", "latest"):
                pp = o["arms"][arm][key][place]
                if abs(pp["mean_R_over_pooledSD"] - z_exp) > 1e-9 or not pp["verdict_by_arm_mean"].startswith(lab):
                    fails.append((arm, key, place, pp["mean_R_over_pooledSD"], pp["verdict_by_arm_mean"]))
    ex_job = rba["bind_gate"][0]["job"]
    o2 = ns["c1"](rba, gp, exclude=[ex_job])
    n2 = o2["arms"]["bind_gate"][METRICS[0]]["earliest"]["n"]
    if n2 != 2 or ex_job in [r["job"] for r in o2["arms"]["bind_gate"][METRICS[0]]["earliest"]["runs"]]:
        fails.append(("exclude", n2))
    return {"pass": not fails, "n_fail": len(fails), "first_fails": [str(x) for x in fails[:5]],
            "bind_gate_HI_legacy_z": o["arms"]["bind_gate"]["HI:legacy_mean_itl_goodput_req_s"]["earliest"]["mean_R_over_pooledSD"]}


# ----------------------------------------------------------------------------------- occupancy / C4
def t_occ(ns):
    rec = {"a": [0.0, 0.1, 0.2], "s_pt": [0.05, 0.3], "e": [0.15, 0.5], "bidx": [0, 1, 1], "fin": [1.0, 0.8, 1.2]}
    run, que, occ = ns["running_and_queue"](rec, None)
    tw = ns["time_where"]
    got = {"occ_ge3": tw(*occ, 0.0, 2.0, lambda c: c >= 3), "occ_ge2": tw(*occ, 0.0, 2.0, lambda c: c >= 2),
           "run_ge3": tw(*run, 0.0, 2.0, lambda c: c >= 3), "que_ge1": tw(*que, 0.0, 2.0, lambda c: c >= 1)}
    exp = {"occ_ge3": 0.5, "occ_ge2": 0.7, "run_ge3": 0.3, "que_ge1": 0.25}
    return {"pass": all(abs(got[k] - exp[k]) < 1e-9 for k in exp), "got": got, "expected": exp}


def t_c4(ns):
    R, sy = _SR_CACHE.get("no_stall") or synth_R()
    o = ns["c4_run"](R)
    # synthetic server truth: request i occupies a cap slot from its batch admission to its last token
    bid = {}
    for k, (_T, S, _R, _Q) in enumerate(sy["batches"]):
        for i in S:
            bid[i] = k
    ev = []
    for i in range(sy["n"]):
        ev += [(sy["s_true"][bid[i]], +1), (sy["gen"][i][-1], -1)]
    times, cnt = F.step_fn(ev)
    w0, w1 = R["rounds"][0]["rec"]["window"]
    truth = {thr: F.time_where(times, cnt, w0, w1, lambda c, t=thr: c >= t) / (w1 - w0) for thr in (46, 47, 48)}
    rec_v = {thr: o[f"recon_time_w_occupancy_ge{thr}"] for thr in (46, 47, 48)}
    return {"pass": abs(rec_v[46] - truth[46]) <= C4_TOL, "recon_occ_share": rec_v, "truth_occ_share": truth,
            "recon_running_ge46": o["recon_time_w_running_ge46"]}


# ----------------------------------------------------------------------------------- zero-transition / IPA
def t_zt(ns):
    keys = ("legacy_mean_itl_goodput_req_s", "slo_goodput_req_s", "slo_goodput_itl65", "ttft_p50_ms", "ttft_p95_ms",
            "token_itl_p50_ms", "token_itl_p95_ms", "token_itl_p99_ms")
    rba = {"bind_gate": [{"job": 1, "node": "x", "n_ctl_transitions": 0}, {"job": 2, "node": "x", "n_ctl_transitions": 0},
                         {"job": 3, "node": "x", "n_ctl_transitions": 4}],
           "d24": [{"job": 11, "node": "y"}, {"job": 12, "node": "y"}]}
    val = {1: 2.0, 2: 4.0, 3: 100.0, 11: 2.0, 12: 2.0}
    gp = {j: {"HI": {k: v for k in keys}} for j, v in val.items()}
    o = ns["zero_transition_contrast"](rba, gp)
    exp = 100.0 * (3.0 - 2.0) / 2.0
    ok = abs(o["legacy_mean_itl_goodput_req_s"]["diff_pct_of_d24"] - exp) < 1e-9 and len(o["zero_transition_bind_gate_jobs"]) == 2
    return {"pass": ok, "diff_pct": o["legacy_mean_itl_goodput_req_s"]["diff_pct_of_d24"], "expected": exp}


def t_ipa(ns):
    def rnd_(n_fail_touched):
        a = [float(i) for i in range(10)]
        tok = [[a[i] + 0.1, a[i] + 0.6] for i in range(10)]  # lifetime [a_i, a_i + 0.6]
        ttft = [0.1] * 10; itls = [[0.02]] * 10
        if n_fail_touched:
            for i in (4, 5):
                ttft[i] = 5.0  # TTFT SLO 3 s violated
        return {"phase": "HI", "round": 0, "rec": {"a": a, "tok_times": tok}, "itls": itls, "ttft": ttft}
    all_R = {7: {"rounds": [rnd_(True)]}, 8: {"rounds": [rnd_(False)]}}
    evs = [{"n_req_gapped": 10, "frac_active_gapped": 1.0, "phase": "HI", "round": 0, "t_start": 4.3, "dur_s": 1.5}]
    o = ns["index_paired_attribution"](all_R, 7, [8], evs)["by_phase"]["HI"]
    # lifetimes [i, i+0.6]; stall [4.3, 5.8] touches i=4 ([4,4.6]) and i=5 ([5,5.6]); pre i=0..3; post i=6..9
    exp = {"pre": (4, 0), "touched": (2, 2), "post": (4, 0)}
    got = {c: (o[c]["n_slots"], o[c]["target_fail_legacy"]) for c in exp if c in o}
    return {"pass": got == exp and all(o[c]["peer_mean_fail_legacy"] == 0 for c in exp), "got": got, "expected": exp}


# ----------------------------------------------------------------------------------- harness
def run_all(ns):
    return {"T-S": t_stall(ns), "T-C1": t_c1(ns), "T-OCC": t_occ(ns), "T-C4": t_c4(ns), "T-ZT": t_zt(ns), "T-IPA": t_ipa(ns)}


def passed(res):
    """Per-unit pass map; POSTHOC units are kept separate from the pre-registered verdict."""
    u = {}
    for k, v in res.items():
        if k == "T-S":
            for name, x in v.items():
                u[f"T-S:{name}"] = x["pass"]
        else:
            u[k] = v["pass"]
    return u


# ----------------------------------------------------------------------------------- recomputation
def welch(a, b):
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    t = (ma - mb) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return t, df


def tag_of(job):
    return list(SRC.glob(f"sgptvsrv_*_{job}.log"))[0].name[len("sgptvsrv_"):-4]


def recompute():
    gp = {}
    def g(job):
        if job not in gp:
            gp[job] = F.goodput_ext({"tag": tag_of(job)})
        return gp[job]
    zt = [856930, 856941, 857112]
    d24_all = [852341, 859005, 859006, 859007]
    d24_x = [j for j in d24_all if j != 859006]
    keys = ("legacy_mean_itl_goodput_req_s", "slo_goodput_itl65", "ttft_p50_ms", "token_itl_p95_ms", "slo_goodput_req_s")
    out = {"zero_transition_vs_d24": {}}
    for label, d24 in (("with_859006(FOLLOWUP_as_published)", d24_all), ("without_859006(FOLLOWUP_own_exclusion)", d24_x)):
        row = {}
        for k in keys:
            a = [g(j)["HI"][k] for j in zt]; b = [g(j)["HI"][k] for j in d24]
            t, df = welch(a, b)
            ci = A.unpaired_bootstrap_ci(b, a)
            row[k] = {"zero_tr_mean": statistics.fmean(a), "zero_tr_sd": statistics.stdev(a), "d24_mean": statistics.fmean(b),
                      "d24_sd": statistics.stdev(b), "n": [len(a), len(b)],
                      "diff_pct_of_d24": 100 * (statistics.fmean(a) - statistics.fmean(b)) / statistics.fmean(b),
                      "welch_t": t, "welch_df": df, "unpaired_boot_ci95_effect": [ci["ci95_low"], ci["ci95_high"]]}
        out["zero_transition_vs_d24"][label] = row
    g39 = [g(852342)["HI"]["legacy_mean_itl_goodput_req_s"]]
    g38 = [g(j)["HI"]["legacy_mean_itl_goodput_req_s"] for j in (856889, 856890, 856891)]
    out["d44_node_contrast_HI_legacy"] = {"gpu39_852342": g39[0], "gpu38_mean": statistics.fmean(g38), "gpu38_sd": statistics.stdev(g38),
                                          "gpu38_minus_gpu39_pct": 100 * (statistics.fmean(g38) - g39[0]) / g39[0], "n": [1, 3]}
    fj = json.loads((HERE / "he0_followup_2026-10-02.json").read_text())
    sens = fj["C1"]["sensitivity_exclude_runs_with_HI_global_stall_ge_3s"]
    bg = sens["arms"]["bind_gate"]["HI:legacy_mean_itl_goodput_req_s"]["earliest"]
    psd = sens["pooled_sd"]["HI:legacy_mean_itl_goodput_req_s"]["pooled_sd"]
    out["followup_1pct_claim_inputs(from he0_followup_2026-10-02.json)"] = {
        "excluded": sens["excluded_jobs"], "ci95_low_over_SD": bg["mean_R_ci95_over_pooledSD"][0], "pooled_sd": psd,
        "ci95_low_req_s": bg["mean_R_ci95"][0]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(HERE / "he0_followup_selftest_v2_2026-10-02.json"))
    args = ap.parse_args()
    ctl_ns = module_variant([])
    res_orig = run_all(vars(F))
    res_ctl = run_all(ctl_ns)
    same = json.dumps(F.rnd(res_orig), sort_keys=True, default=str) == json.dumps(F.rnd(res_ctl), sort_keys=True, default=str)
    pc = passed(res_ctl)
    print("[harness] re-exec copy identical to imported module:", same)
    for k, v in pc.items():
        print(f"[control] {k:40s} {'PASS' if v else 'FAIL'}")
    for name, r in res_ctl["T-S"].items():
        print(f"      T-S {name:16s} expected={r['expected']} got={r['got']} -> {'PASS' if r['pass'] else 'FAIL'}")
    print(f"      T-C4 recon occ>=46 {res_ctl['T-C4']['recon_occ_share'][46]:.4f} truth {res_ctl['T-C4']['truth_occ_share'][46]:.4f}")
    mut = {}
    for m, subs in MUTANTS.items():
        r = run_all(module_variant(subs))
        pm = passed(r)
        failed = [k for k, v in pm.items() if pc[k] and not v]
        mut[m] = {"failed_tests": failed, "verdict": "KILLED" if failed else "SURVIVED",
                  "T-S_got": {n: x["got"] for n, x in r["T-S"].items()}}
        print(f"[mutant] {m:34s} -> {mut[m]['verdict']} (failed: {failed})")
    prereg_units = {k: v for k, v in pc.items() if "POSTHOC" not in k}
    overall = all(prereg_units.values()) and same and all(v["verdict"] == "KILLED" for v in mut.values())
    print("SELF-TEST v2", "PASS" if overall else "FAIL")
    rc = recompute()
    for lab, row in rc["zero_transition_vs_d24"].items():
        print("== 0-transition bind+GATE vs d24", lab)
        for k, v in row.items():
            print(f"   {k:32s} diff {v['diff_pct_of_d24']:+.2f}%  Welch t {v['welch_t']:+.2f} df {v['welch_df']:.1f}  boot CI {v['unpaired_boot_ci95_effect']}")
    print("== d44 node contrast", rc["d44_node_contrast_HI_legacy"])
    print("== FOLLOWUP 1% inputs", rc["followup_1pct_claim_inputs(from he0_followup_2026-10-02.json)"])
    out = {"generated": "2026-10-02", "script": Path(__file__).name, "harness_copy_identical": same,
           "control": res_ctl, "control_pass": pc, "mutants": mut, "overall_pass": overall, "recompute": rc}
    Path(args.output).write_text(json.dumps(F.rnd(out), indent=1, default=str, ensure_ascii=False))
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
