#!/usr/bin/env python3
"""S0R_REPLICATION_2026-08-03.py -- INDEPENDENT replication of the "E1 SPLIT
population is bimodal" reframing, executed per
`PREREG_S0R_MODE_2026-08-03.md`.

STATUS: **DIAGNOSTIC ONLY** (prereg sec 5).  No performance claim, no
`G_LEVER`/`G_FLAT`, no adjudication of DESIGN.md sec 4.3.13 sec 0's (i)/(ii),
no canon edit.

WHO WROTE THIS
--------------
result-analyst, 2026-08-03.  Not claims-auditor (who produced the reframing),
not the main session (who wrote `PREREG_S0_AXIS_2026-08-03.md` /
`s0_axis_check.py` and this pre-registration).  Per prereg sec 1 this is a
*partial* independence guarantee: the estimator in sec 2 is the auditor's
proposal and falsifiers 4-5 are the main session's; only the execution is
independent.

INDEPENDENCE LEDGER (prereg sec 1: declare every reused primitive, and why)
--------------------------------------------------------------------------
REUSED from `m3_conditional.py` (the *audited* module where the disputed label
lives).  Re-implementing any of these would change the object under study and
break the comparison, which is exactly what the prereg forbids:

  R1 `label_probe`     -- the whole E1 labelling pass.  It defines `split_frac`
                          (duration-weighted share of an ITL interval spent at
                          `decode_sms == D`), the `a_free` flag, and the
                          client<->telemetry alignment.  THE OBJECT OF STUDY.
  R2 `tokens`          -- the primary population selector, `a_free_only=True`
                          (prereg sec 2 "unit").
  R3 `sel_split` / `sel_unsplit` -- the label bands 0.90 / 0.10 (prereg sec 2
                          "label"; explicitly not a free parameter).
  R4 `pctl`            -- percentile interpolation rule (matches the harness
                          and `s0dc_client._percentile`, so my C2 gate G-B can
                          check against the client's own recorded p50).
  R5 (transitively, inside R1) `load_telemetry`, `replay_arrivals`, `align`,
                          and label_probe's inner `frac`.

NOT reused (written fresh here):
  N1 the mode estimator (prereg sec 2) and every slow-/fast-share statistic;
  N2 the entire C2 (865493) reader: result.txt anchor parser, raw-jsonl ITL
     reconstruction, telemetry snapshot loader, `wfrac` step-function
     integrator, batch weighting;
  N3 the lag-shift driver (row 5), the per-block scatter machinery, all
     reporting.
  N4 NOTHING from `s0_axis_check.py`, `c2_anchor.py`, `e1_analyze.py`, or any
     scratchpad script was read into this program.  (`c2_anchor.py` was read by
     the analyst only to learn *what estimand* the number 28.79 names -- a
     batch-bin-matched p50 -- so that the row-1 comparand is not silently a
     different quantity; no code or value is imported from it.  The row-1
     primary comparand here is recomputed from raw artifacts by N2.)

TWO INSTRUMENT GATES (mine, run before anything else)
-----------------------------------------------------
  G-A  my `wfrac` (used for the C2 side) must reproduce `label_probe`'s
       `split_frac` ELEMENT-WISE on an E1 probe.  Without this, a C2/E1
       difference could be my integrator rather than the data.
  G-B  my C2 ITL reconstruction must reproduce `s0dc_client`'s own recorded
       `itl_samples` and `itl_ms_p50` per rep.  This validates the C2 reader
       against the producer, not against another analysis script.

ESTIMATOR (prereg sec 2, fixed before running; NOT tuned here)
--------------------------------------------------------------
  unit   : one ITL interval = one emitted token, `a_free_only=True`.
  label  : split_frac >= 0.90 (SPLIT) / <= 0.10 (UNSPLIT108).
  mode   : argmax of a 0.25 ms histogram over ITL in (1.15*p50, 60] ms.
  aggreg.: pooled over the 8 blocks, per-block reported alongside.

Two under-specifications in the prereg, resolved HERE, BEFORE running, and both
reported with their alternative so the choice is auditable:
  D1 histogram bin origin.  PRIMARY = a global grid anchored at 0.0 ms
     (bin k = [0.25k, 0.25(k+1)) ), so that SPLIT and UNSPLIT modes land on the
     SAME grid and rows 3/4 are comparisons of like with like.  SENSITIVITY =
     grid anchored at the window edge 1.15*p50.  Mode point estimate = bin
     centre; ties -> lowest bin.
  D2 "slow share".  PRIMARY = in-window mass, #(1.15*p50 < ITL <= 60)/n, which
     is the support the prereg's estimator is defined on and the support row 4
     names ("no second mode above 5% share in (1.15*p50, 60]").  COMPANION =
     unbounded mass #(ITL > 1.15*p50)/n.  Both are printed everywhere.

USAGE
  python3 S0R_REPLICATION_2026-08-03.py --cache /path/to/s0r_cache.pkl
"""
import argparse
import collections
import importlib.util
import json
import math
import os
import pickle
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCALEUP = os.path.join(os.path.dirname(HERE), "s8_scaleup")
E1_JOB = "872077"
C2_JOB = "865493"

# ---- prereg sec 2 constants (do not tune) --------------------------------
BINW = 0.25          # ms
HI_MS = 60.0         # ms, window ceiling
LO_FAC = 1.15        # window floor = LO_FAC * p50
NEG_CTRL_SHARE = 0.05  # row 4 threshold
ROW1_BAND = (0.95, 1.05)
ROW3_BAND = (0.97, 1.03)
HA8_BAND = (0.75, 0.90)   # prereg: consistency check with a known answer
LAGS = [round(x, 2) for x in np.arange(-0.40, 0.4001, 0.05)]

E1_ARMS = ("T8", "Ha8")
E1_CELLS = ("d16", "d24", "d44", "d54")
BLOCKS = tuple(range(1, 9))

# ---- C2 client protocol constants (from s0dc_client.py + the sbatch header) -
C2_WARMUP_S = 20.0
C2_MEASURE_S = 60.0
C2_REPS = (1, 2, 3, 4)

# ------------------------------------------------------------------ m3 import
_spec = importlib.util.spec_from_file_location(
    "m3_conditional", os.path.join(HERE, "m3_conditional.py"))
m3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m3)                                            # R1-R5

# memoise telemetry loading only (no logic change) so the row-5 lag sweep can
# relabel 136 times without re-parsing 15 MB of JSON each time.
_TEL = {}
_orig_load = m3.load_telemetry


def _load_memo(path):
    if path not in _TEL:
        _TEL[path] = _orig_load(path)
    return _TEL[path]


m3.load_telemetry = _load_memo


# =========================================================== N1: statistics
def hist_mode(vals, lo, hi, anchor=0.0, binw=BINW):
    """argmax of a `binw` histogram over (lo, hi]; returns (centre, n_bin, n_win)."""
    sel = [v for v in vals if lo < v <= hi]
    if not sel:
        return (float("nan"), 0, 0)
    cnt = collections.Counter(int(math.floor((v - anchor) / binw)) for v in sel)
    kbest = max(sorted(cnt), key=lambda k: cnt[k])      # ties -> lowest bin
    return (anchor + (kbest + 0.5) * binw, cnt[kbest], len(sel))


def pop_stats(vals):
    """Everything rows 1-5 need about one labelled population."""
    n = len(vals)
    if n == 0:
        return None
    p50 = m3.pctl(vals, 0.50)
    lo = LO_FAC * p50
    n_win = sum(1 for v in vals if lo < v <= HI_MS)
    n_above = sum(1 for v in vals if v > lo)
    m_slow, nb_slow, _ = hist_mode(vals, lo, HI_MS, anchor=0.0)
    m_slow_alt, _, _ = hist_mode(vals, lo, HI_MS, anchor=lo)
    m_fast, nb_fast, _ = hist_mode(vals, 0.0, lo, anchor=0.0)
    m_glob, _, _ = hist_mode(vals, 0.0, HI_MS, anchor=0.0)
    return dict(n=n, p50=p50, p95=m3.pctl(vals, 0.95), mean=float(np.mean(vals)),
                lo=lo, share_win=n_win / n, share_above=n_above / n,
                mode_slow=m_slow, mode_slow_alt=m_slow_alt, n_mode_slow=nb_slow,
                mode_fast=m_fast, n_mode_fast=nb_fast, mode_global=m_glob,
                n_win=n_win)


def band_mass(vals, centre, rel=0.15):
    """share of `vals` within +/- rel of `centre` (row 4 shape probe)."""
    if not vals or not np.isfinite(centre):
        return float("nan")
    lo, hi = centre * (1 - rel), centre * (1 + rel)
    return sum(1 for v in vals if lo <= v <= hi) / len(vals)


def msd(xs):
    xs = [x for x in xs if np.isfinite(x)]
    if not xs:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return (float(np.mean(xs)), sd, min(xs), max(xs))


def fmt_msd(xs, w=7, p=2):
    m, sd, lo, hi = msd(xs)
    return f"{m:{w}.{p}f} +/-{sd:{w-1}.{p}f} [{lo:.{p}f},{hi:.{p}f}]"


# ============================================================ E1 labelling
def build_e1(cache):
    if cache and os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    O = {}
    for arm in E1_ARMS:
        for cell in E1_CELLS:
            for b in BLOCKS:
                O[(arm, cell, b)] = m3.label_probe(HERE, arm, cell, b, b, E1_JOB)
                print(f"  labelled {arm} {cell} b{b}", file=sys.stderr)
    if cache:
        pickle.dump(O, open(cache, "wb"))
    return O


def e1_pop(O, arm, cell, blocks=BLOCKS, sel=None):
    sel = sel or m3.sel_split
    out = []
    for b in blocks:
        out += m3.tokens(O[(arm, cell, b)]["recs"], sel)               # R2/R3
    return out


# ================================================= N2: the C2 (865493) reader
def c2_anchors(arm):
    """tag -> t0_monotonic_s, parsed from the campaign's own result.txt."""
    path = os.path.join(SCALEUP, f"s8_deconf_{arm}_C1024_{C2_JOB}_result.txt")
    out = {}
    for line in open(path):
        line = line.strip()
        if not (line.startswith("{") and '"tag"' in line):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        out[o["tag"]] = o
    return out


def c2_snapshots(arm, cell):
    """(ts, decode_sms, decode_running_batch_size, prefill_active_batch_size)
    as numpy arrays sorted by absolute monotonic timestamp."""
    path = os.path.join(SCALEUP,
                        f"s8_deconf_{arm}_C1024_{cell}_{C2_JOB}_telemetry.jsonl")
    rows = []
    for line in open(path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "runtime_snapshot":
            continue
        rows.append((e["timestamp_monotonic_s"],
                     e.get("decode_sms") if e.get("decode_sms") is not None else -1,
                     e.get("decode_running_batch_size", 0),
                     e.get("prefill_active_batch_size", 0)))
    rows.sort(key=lambda r: r[0])
    return (np.array([r[0] for r in rows]),
            np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows], dtype=float),
            np.array([r[3] for r in rows], dtype=float))


def wfrac(ct, mask, a, b, extra=None):
    """Duration-weighted share of [a,b] in which `mask` holds, over the snapshot
    step function.  Deliberately the SAME arithmetic (including the trailing
    `len(ct)-1` truncation) as m3_conditional.label_probe's inner `frac`;
    gate G-A asserts element-wise equality.  If `extra` is an array, also
    returns its duration-weighted mean over the same support."""
    if b <= a:
        return (0.0, float("nan")) if extra is not None else 0.0
    i0 = max(np.searchsorted(ct, a, side="right") - 1, 0)
    i1 = max(np.searchsorted(ct, b, side="right") - 1, 0)
    tot = hit = acc = 0.0
    for k in range(i0, min(i1 + 1, len(ct) - 1)):
        s = max(a, ct[k])
        e = min(b, ct[k + 1])
        if e <= s:
            continue
        tot += e - s
        if mask[k]:
            hit += e - s
        if extra is not None:
            acc += extra[k] * (e - s)
    f = (hit / tot) if tot > 0 else 0.0
    if extra is not None:
        return (f, (acc / tot) if tot > 0 else float("nan"))
    return f


def c2_cell(arm, cell):
    """Per-token, partition-labelled C2 population for one (arm, cell).

    Returns per-rep dicts with recs = [(itl_ms, split_frac, un108_frac,
    mean_decode_batch)].  ITL selection replicates s0dc_client.py:178-186
    exactly (gap i>=2, later chunk inside [warmup, warmup+measure]), so gate G-B
    can compare against the client's own recorded summary.
    """
    D = int(cell[1:])
    anc = c2_anchors(arm)
    ts, dsm, drb, pab = c2_snapshots(arm, cell)
    m_split = (dsm == D)
    m_un = (dsm == 108)
    lo, hi = C2_WARMUP_S, C2_WARMUP_S + C2_MEASURE_S
    reps = []
    for rep in C2_REPS:
        tag = f"deconf_{arm}_C1024_{cell}_r{rep}"
        summ = anc.get(tag)
        if summ is None or "t0_monotonic_s" not in summ:
            reps.append(dict(rep=rep, tag=tag, missing=True, recs=[], summ=summ))
            continue
        t0 = summ["t0_monotonic_s"]
        path = os.path.join(
            SCALEUP, f"s8_deconf_{arm}_C1024_{cell}_{C2_JOB}_rep{rep}_raw.jsonl")
        recs = []
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            t = o["chunk_times_s"]
            for i in range(2, len(t)):
                if not (lo <= t[i] <= hi):
                    continue
                a, b = t0 + t[i - 1], t0 + t[i]
                sf, mb = wfrac(ts, m_split, a, b, extra=drb)
                uf = wfrac(ts, m_un, a, b)
                recs.append(((t[i] - t[i - 1]) * 1000.0, sf, uf, mb))
        reps.append(dict(rep=rep, tag=tag, missing=False, recs=recs, summ=summ))
    return reps


# ====================================================================== gates
def gate_A(O, arm="T8", cell="d16", block=1):
    """my wfrac == label_probe's inner frac, element-wise, on an E1 probe."""
    p = O[(arm, cell, block)]
    tag = f"e1m3_{arm}_{cell}_b{block}_{E1_JOB}"
    o = json.loads(open(f"{HERE}/{tag}_s{block}.jsonl").read().strip().split("\n")[0])
    t0b, rows = m3.load_telemetry(f"{HERE}/{tag}_telemetry.jsonl")
    tt, il = o["ttfts"], o["itls"]
    err = o.get("errors") or []
    arr = m3.replay_arrivals(block, m3.RATE_DEFAULT, len(tt))            # R5
    ct = np.array([r[0] - t0b - p["lag"] for r in rows])
    dsm = np.array([r[1] if r[1] is not None else -1 for r in rows])
    mask = (dsm == p["Dsm"])
    nw = int(math.ceil(m3.WARMUP_S * m3.RATE_DEFAULT))
    keep = [i for i in range(nw, len(tt)) if (err[i] if i < len(err) else "") == ""]
    mine, k = [], 0
    for i in keep:
        if not il[i]:
            continue
        t = arr[i] + tt[i]
        for dd in il[i]:
            mine.append(wfrac(ct, mask, t, t + dd))
            t = t + dd
    theirs = [r[2] for r in p["recs"]]
    n = min(len(mine), len(theirs))
    dmax = max(abs(mine[j] - theirs[j]) for j in range(n)) if n else float("nan")
    return dict(n_mine=len(mine), n_theirs=len(theirs), max_abs_diff=dmax,
                ok=(len(mine) == len(theirs) and dmax < 1e-12))


def gate_B(reps):
    """my C2 ITL reconstruction == s0dc_client's own recorded summary."""
    out = []
    for r in reps:
        if r["missing"]:
            out.append((r["tag"], None, None, None, None, "NO_ANCHOR"))
            continue
        v = [x[0] for x in r["recs"]]
        mine_n, mine_p50 = len(v), m3.pctl(v, 0.50)                       # R4
        their_n = r["summ"]["itl_samples"]
        their_p50 = r["summ"]["itl_ms_p50"]
        ok = (mine_n == their_n) and abs(mine_p50 - their_p50) < 5e-3
        out.append((r["tag"], mine_n, their_n, mine_p50, their_p50,
                    "PASS" if ok else "FAIL"))
    return out


# ==================================================================== rows
def sec(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def report_row4(O, out_arms=("T8",)):
    """NEGATIVE CONTROL -- run first, per the prereg's emphasis."""
    sec("[ROW 4 *] NEGATIVE CONTROL: same mode estimator on the UNSPLIT108 pop.")
    print("  falsifier: UNSPLIT also carries a ~31 ms mode at ~9% share")
    print("  -> primary read = in-window share (1.15*p50, 60] must be <= 5%\n")
    for arm in out_arms:
        for cell in E1_CELLS:
            SP = e1_pop(O, arm, cell, sel=m3.sel_split)
            UN = e1_pop(O, arm, cell, sel=m3.sel_unsplit)
            s, u = pop_stats(SP), pop_stats(UN)
            print(f"  {arm} {cell}")
            print(f"    SPLIT   n={s['n']:6d} p50={s['p50']:6.2f} cut={s['lo']:6.2f} "
                  f"slow-mode={s['mode_slow']:6.2f} share_win={s['share_win']*100:5.2f}% "
                  f"share_above={s['share_above']*100:5.2f}%")
            print(f"    UNSPLIT n={u['n']:6d} p50={u['p50']:6.2f} cut={u['lo']:6.2f} "
                  f"slow-mode={u['mode_slow']:6.2f} share_win={u['share_win']*100:5.2f}% "
                  f"share_above={u['share_above']*100:5.2f}%")
            print(f"    mass within +/-15% of SPLIT's slow mode "
                  f"({s['mode_slow']:.2f} ms):  SPLIT={band_mass(SP, s['mode_slow'])*100:5.2f}%"
                  f"   UNSPLIT={band_mass(UN, s['mode_slow'])*100:5.2f}%")
            verdict = ("UNIMODAL (share<=5%)" if u["share_win"] <= NEG_CTRL_SHARE
                       else "SECOND MODE PRESENT (share>5%) -> ROW 4 FIRES")
            print(f"    -> {verdict}")
            # per-block scatter of the UNSPLIT in-window share
            pb = []
            for b in BLOCKS:
                ub = m3.tokens(O[(arm, cell, b)]["recs"], m3.sel_unsplit)
                st = pop_stats(ub)
                pb.append(st["share_win"] * 100 if st else float("nan"))
            print(f"    per-block UNSPLIT share_win %: {fmt_msd(pb, 6, 2)}")
    sec("[ROW 4b] SHAPE: 1 ms histograms, T8 (share of population per bin, %)")
    for cell in E1_CELLS:
        SP = e1_pop(O, "T8", cell, sel=m3.sel_split)
        UN = e1_pop(O, "T8", cell, sel=m3.sel_unsplit)
        print(f"  T8 {cell}   (rows: SPLIT / UNSPLIT; bins are 1 ms, 0..48 ms)")
        for name, v in (("SPLIT ", SP), ("UNSPL ", UN)):
            h = collections.Counter(int(x) for x in v if x < 48)
            line = " ".join(f"{100*h.get(k,0)/len(v):4.1f}" for k in range(0, 48, 1))
            print(f"    {name}{line}")
        print("    bin idx:  " + " ".join(f"{k:4d}" for k in range(0, 48, 1)))


def report_row4c(O):
    """Row 4 asks a yes/no ("is UNSPLIT unimodal?").  The answer turned out to
    be quantitative, so this reports the quantity: how much is the slow bump
    ENRICHED in the SPLIT class rather than absent from the UNSPLIT class, in
    absolute counts, plus whether the slow tokens of BOTH classes sit against
    prefill activity.  Added after the first run; changes no pre-registered
    quantity, uses the same cut (1.15*p50 of the SPLIT population) for every
    sub-population so the shares are commensurable.

    CAVEAT on the prefill-overlap column: 872077 ran with
    PDMUX_TRACE_FORCE_PREFILL=0 and CONSENSUS sec 1-27 records that this label
    is UNDER-SAMPLED in this job.  Directional only.
    """
    sec("[ROW 4c] ENRICHMENT, not exclusivity: absolute counts + prefill assoc.")
    print("  cut = 1.15 * p50(SPLIT pop) of that cell; a_free population only.")
    print(f"  {'arm':>4} {'cell':>5} {'cut':>6} | {'n_all':>7} {'slow_all':>8} "
          f"{'base%':>6} | {'n_SP':>6} {'slow_SP':>7} {'SP%':>6} {'enrich':>7} "
          f"| {'n_UN':>7} {'slow_UN':>8} {'UN%':>6} {'enrich':>7}")
    for arm in E1_ARMS:
        for cell in E1_CELLS:
            allr = [r for b in BLOCKS for r in O[(arm, cell, b)]["recs"] if r[1]]
            SP = e1_pop(O, arm, cell, sel=m3.sel_split)
            cut = LO_FAC * m3.pctl(SP, 0.50)
            def slow(v):
                return sum(1 for x in v if cut < x <= HI_MS)
            av = [r[0] for r in allr]
            uv = e1_pop(O, arm, cell, sel=m3.sel_unsplit)
            b_all = slow(av) / len(av)
            b_sp, b_un = slow(SP) / len(SP), slow(uv) / len(uv)
            print(f"  {arm:>4} {cell:>5} {cut:6.2f} | {len(av):7d} {slow(av):8d} "
                  f"{b_all*100:6.2f} | {len(SP):6d} {slow(SP):7d} {b_sp*100:6.2f} "
                  f"{b_sp/b_all:7.2f} | {len(uv):7d} {slow(uv):8d} {b_un*100:6.2f} "
                  f"{b_un/b_all:7.2f}")
    print("\n  prefill association (share of tokens with prefill_overlap_frac > 0;")
    print("  UNDER-SAMPLED instrument, see CONSENSUS sec 1-27 -- directional only)")
    print(f"  {'arm':>4} {'cell':>5} | {'SP fast':>8} {'SP slow':>8} | "
          f"{'UN fast':>8} {'UN slow':>8}")
    for arm in E1_ARMS:
        for cell in E1_CELLS:
            allr = [r for b in BLOCKS for r in O[(arm, cell, b)]["recs"] if r[1]]
            SP = e1_pop(O, arm, cell, sel=m3.sel_split)
            cut = LO_FAC * m3.pctl(SP, 0.50)
            def sh(pred):
                v = [r for r in allr if pred(r)]
                return (sum(1 for r in v if r[3] > 0) / len(v)) if v else float("nan")
            print(f"  {arm:>4} {cell:>5} | "
                  f"{sh(lambda r: m3.sel_split(r) and r[0] <= cut)*100:8.1f} "
                  f"{sh(lambda r: m3.sel_split(r) and cut < r[0] <= HI_MS)*100:8.1f} | "
                  f"{sh(lambda r: m3.sel_unsplit(r) and r[0] <= cut)*100:8.1f} "
                  f"{sh(lambda r: m3.sel_unsplit(r) and cut < r[0] <= HI_MS)*100:8.1f}")


def report_diag(O):
    """Why my slow-SHARE levels sit below the audit's parenthetical numbers.

    Added AFTER the first run, and it changes no pre-registered quantity: it
    only probes which *population* the audit's 8.95/13.60/23.62/29.38 could
    have come from, since the prereg pinned the mode estimator but not the
    share definition (my D2).  Reported so the divergence is explained rather
    than smoothed over.
    """
    sec("[DIAG] population sensitivity of the slow SHARE (not a prereg row)")
    print("  audit parenthetical (row 2): 8.95 / 13.60 / 23.62 / 29.38 %")
    print(f"  {'arm':>4} {'cell':>5} {'population':>16} {'n':>7} {'p50':>6} "
          f"{'cut':>6} {'share_win%':>11} {'share_above%':>13} {'slow_mode':>10}")
    for arm in E1_ARMS:
        for cell in E1_CELLS:
            for name, afo in (("a_free_only", True), ("all tokens", False)):
                v = []
                for b in BLOCKS:
                    v += m3.tokens(O[(arm, cell, b)]["recs"], m3.sel_split,
                                   a_free_only=afo)
                s = pop_stats(v)
                print(f"  {arm:>4} {cell:>5} {name:>16} {s['n']:7d} {s['p50']:6.2f} "
                      f"{s['lo']:6.2f} {s['share_win']*100:11.2f} "
                      f"{s['share_above']*100:13.2f} {s['mode_slow']:10.2f}")


def report_row123(O, C2, out=None):
    sec("[ROW 1] E1 slow mode / C2 SPLIT p50, per cell")
    print("  prereg band [0.95,1.05] AND flat in D.")
    print("  C2 comparand PRIMARY = per-cell SPLIT p50 (all batches, pooled over")
    print("  4 reps = pseudo-replication, n_indep=1).  COMPANION = batch-bin-")
    print("  matched p50 at the E1 modal decode batch (that is the estimand the")
    print("  number 28.79 names).\n")
    rows = []
    for arm in E1_ARMS:
        for cell in E1_CELLS:
            SP = e1_pop(O, arm, cell, sel=m3.sel_split)
            s = pop_stats(SP)
            c = C2.get((arm, cell))
            if c is None or c["n_split"] == 0:
                print(f"  {arm} {cell}: E1 slow-mode={s['mode_slow']:6.2f} ms | "
                      f"C2 UNAVAILABLE ({'no telemetry anchor' if c is None else 'no SPLIT tokens'})")
                continue
            r = s["mode_slow"] / c["sp_p50"]
            r_alt = s["mode_slow_alt"] / c["sp_p50"]
            r_bin = (s["mode_slow"] / c["sp_p50_binmatch"]
                     if np.isfinite(c["sp_p50_binmatch"]) else float("nan"))
            # per-block scatter of the E1 slow mode -> ratio scatter
            pb = []
            for b in BLOCKS:
                sb = pop_stats(m3.tokens(O[(arm, cell, b)]["recs"], m3.sel_split))
                pb.append(sb["mode_slow"] / c["sp_p50"] if sb else float("nan"))
            n_pb = sum(1 for x in pb if np.isfinite(x))
            flag = "in-band" if ROW1_BAND[0] <= r <= ROW1_BAND[1] else "OUT-OF-BAND"
            print(f"  {arm} {cell}: E1 slow-mode={s['mode_slow']:6.2f} "
                  f"(alt-anchor {s['mode_slow_alt']:6.2f})  C2 SPLIT p50="
                  f"{c['sp_p50']:6.2f} (n={c['n_split']}, {c['n_reps']} reps)"
                  f"  ratio={r:5.3f} [{flag}]")
            print(f"          alt-anchor ratio={r_alt:5.3f} | batch-matched "
                  f"comparand p50={c['sp_p50_binmatch']:6.2f} (bin={c['binmatch']}, "
                  f"n={c['n_binmatch']}) ratio={r_bin:5.3f}")
            print(f"          per-block ratio (n={n_pb}/8): {fmt_msd(pb, 6, 3)}")
            rows.append((arm, cell, r))
    for arm in E1_ARMS:
        rr = [(int(c[1:]), r) for a, c, r in rows if a == arm]
        if len(rr) >= 2:
            rr.sort()
            trend = "MONOTONE" if (all(rr[i][1] < rr[i+1][1] for i in range(len(rr)-1))
                                   or all(rr[i][1] > rr[i+1][1] for i in range(len(rr)-1))) else "non-monotone"
            print(f"  {arm} ratio vs D: " + "  ".join(f"d{d}={v:.3f}" for d, v in rr)
                  + f"   -> {trend}")

    sec("[ROW 2] T8 slow-mode SHARE rises monotonically in D?")
    print("  audit: 8.95 -> 13.60 -> 23.62 -> 29.38 % for d16/d24/d44/d54")
    print("  blocks share the seed across cells, so the block index pairs; the")
    print("  adjacent-pair test below is paired-within-block (n=8).\n")
    for arm in E1_ARMS:
        per_cell_pb = {}
        for cell in E1_CELLS:
            pooled = pop_stats(e1_pop(O, arm, cell, sel=m3.sel_split))
            pb_w, pb_a = [], []
            for b in BLOCKS:
                sb = pop_stats(m3.tokens(O[(arm, cell, b)]["recs"], m3.sel_split))
                pb_w.append(sb["share_win"] * 100 if sb else float("nan"))
                pb_a.append(sb["share_above"] * 100 if sb else float("nan"))
            per_cell_pb[cell] = (pb_w, pb_a)
            print(f"  {arm} {cell}: pooled share_win={pooled['share_win']*100:6.2f}% "
                  f"share_above={pooled['share_above']*100:6.2f}%   "
                  f"per-block share_win: {fmt_msd(pb_w, 6, 2)}")
        for lo_c, hi_c in zip(E1_CELLS[:-1], E1_CELLS[1:]):
            d = [per_cell_pb[hi_c][0][i] - per_cell_pb[lo_c][0][i]
                 for i in range(len(BLOCKS))]
            m, sd, mn, mx = msd(d)
            npos = sum(1 for x in d if x > 0)
            print(f"  {arm} paired {hi_c}-{lo_c}: {m:+6.2f} +/-{sd:5.2f} pp "
                  f"({npos}/8 blocks positive, range [{mn:+.2f},{mx:+.2f}])")

    sec("[ROW 3] T8 fast mode of SPLIT vs the same job's UNSPLIT population")
    print("  prereg band [0.97,1.03]; audit 11.06 vs 10.97 = 1.011\n")
    for arm in E1_ARMS:
        for cell in E1_CELLS:
            SP = e1_pop(O, arm, cell, sel=m3.sel_split)
            UN = e1_pop(O, arm, cell, sel=m3.sel_unsplit)
            s, u = pop_stats(SP), pop_stats(UN)
            r_mode = s["mode_fast"] / u["mode_global"]
            r_p50 = s["p50"] / u["p50"]
            flag = "in-band" if ROW3_BAND[0] <= r_mode <= ROW3_BAND[1] else "OUT-OF-BAND"
            pb = []
            for b in BLOCKS:
                sb = pop_stats(m3.tokens(O[(arm, cell, b)]["recs"], m3.sel_split))
                ub = pop_stats(m3.tokens(O[(arm, cell, b)]["recs"], m3.sel_unsplit))
                pb.append(sb["mode_fast"] / ub["mode_global"]
                          if (sb and ub) else float("nan"))
            print(f"  {arm} {cell}: SPLIT fast-mode={s['mode_fast']:6.2f}  "
                  f"UNSPLIT mode={u['mode_global']:6.2f}  ratio={r_mode:5.3f} [{flag}]"
                  f"   (p50/p50={r_p50:5.3f})   per-block: {fmt_msd(pb, 6, 3)}")


def report_row5(O, arm="T8", cells=("d16", "d44")):
    sec("[ROW 5 *] LAG SWEEP: delta in [-0.40,+0.40] s, 0.05 s steps")
    print("  falsifier: if argmax slow-share reaches >= 0.90 the 'impurity' is a")
    print("  clock artifact and m3.align -- not the label -- is the defect.")
    print("  audit: argmax at delta ~ +0.05..+0.10 s, share 13.6-14.9% vs 8.95% at 0\n")
    base = {(a, c, b): O[(a, c, b)]["lag"] for (a, c, b) in O}
    orig_align = m3.align

    def relabel(cell, d):
        SP = []
        for b in BLOCKS:
            lag0 = base[(arm, cell, b)]
            m3.align = (lambda rows, t0b, arr, tt, il, _l=lag0 + d:
                        (_l, float("nan")))
            try:
                p = m3.label_probe(HERE, arm, cell, b, b, E1_JOB)
            finally:
                m3.align = orig_align
            SP += m3.tokens(p["recs"], m3.sel_split)
        return SP

    for cell in cells:
        print(f"  --- {arm} {cell} (pooled over 8 blocks) ---")
        # delta = 0 first, so the fixed-cut column uses the as-measured cut
        fixed_cut = pop_stats(relabel(cell, 0.0))["lo"]
        print(f"  fixed cut (1.15*p50 at delta=0) = {fixed_cut:.2f} ms")
        print(f"  {'delta_s':>8} {'n_split':>8} {'p50':>7} {'share_win%':>11} "
              f"{'share_above%':>13} {'slow_mode':>10} {'fixedcut%':>10}")
        best = (None, -1.0)
        for d in LAGS:
            SP = relabel(cell, d)
            s = pop_stats(SP)
            fc = sum(1 for v in SP if fixed_cut < v <= HI_MS) / len(SP)
            print(f"  {d:+8.2f} {s['n']:8d} {s['p50']:7.2f} {s['share_win']*100:11.2f} "
                  f"{s['share_above']*100:13.2f} {s['mode_slow']:10.2f} {fc*100:10.2f}")
            if s["share_win"] > best[1]:
                best = (d, s["share_win"])
        print(f"  -> argmax(share_win) at delta={best[0]:+.2f} s, share={best[1]*100:.2f}% "
              f"({'FIRES: >=90%' if best[1] >= 0.90 else 'does not reach 90% -> falsifier not fired'})")
    m3.align = orig_align


def report_context(O, C2):
    sec("[CONTEXT] alignment quality, label yield, C2 label yield")
    print(f"  {'arm':>4} {'cell':>5} {'blk':>3} {'lag_s':>8} {'align_r':>8} "
          f"{'n_rec':>7} {'n_split':>8} {'n_unspl':>8} {'w_time':>7}  flag")
    for arm in E1_ARMS:
        for cell in E1_CELLS:
            for b in BLOCKS:
                p = O[(arm, cell, b)]
                ns = len(m3.tokens(p["recs"], m3.sel_split))
                nu = len(m3.tokens(p["recs"], m3.sel_unsplit))
                fl = "" if p["align_r"] >= m3.ALIGN_R_MIN else "ALIGN-WEAK"
                print(f"  {arm:>4} {cell:>5} {b:>3} {p['lag']:8.2f} {p['align_r']:8.3f} "
                      f"{len(p['recs']):7d} {ns:8d} {nu:8d} {p['w_time']:7.3f}  {fl}")
    print()
    print(f"  C2 label yield  {'arm':>4} {'cell':>5} {'n_tok':>8} {'SPLIT':>8} "
          f"{'UN108':>8} {'AMBIG':>8} {'sp_p50':>8} {'un_p50':>8} {'all_p50':>8}")
    for (arm, cell), c in sorted(C2.items()):
        print(f"                  {arm:>4} {cell:>5} {c['n_tot']:8d} {c['n_split']:8d} "
              f"{c['n_un']:8d} {c['n_amb']:8d} {c['sp_p50']:8.2f} {c['un_p50']:8.2f} "
              f"{c['all_p50']:8.2f}")


# ==================================================================== driver
C2_SKIPPED = []


def collect_c2(O):
    """C2 populations for every (arm, cell) that has a client anchor, plus the
    batch-matched companion at the E1 modal decode batch."""
    out = {}
    for arm in E1_ARMS:
        for cell in E1_CELLS:
            if cell == "d54":
                continue                      # 865493 has no d54 cell
            try:
                reps = c2_cell(arm, cell)
            except FileNotFoundError as exc:
                print(f"  [C2] {arm} {cell}: MISSING FILE {exc}", file=sys.stderr)
                continue
            gb = gate_B(reps)
            recs = [r for rp in reps for r in rp["recs"]]
            if not recs:
                C2_SKIPPED.append((arm, cell,
                                   [r["tag"] for r in reps if r["missing"]]))
                continue
            sp = [r[0] for r in recs if r[1] >= m3.SPLIT_HI]
            un = [r[0] for r in recs if r[2] >= m3.SPLIT_HI]
            amb = len(recs) - len(sp) - len(un)
            # batch-matched companion: bin the duration-weighted mean decode
            # batch to int and take the bin nearest the E1 modal decode batch.
            bybin = collections.defaultdict(list)
            for it, sf, uf, mb in recs:
                if sf >= m3.SPLIT_HI and np.isfinite(mb):
                    bybin[int(round(mb))].append(it)
            e1b = e1_modal_batch(O, arm, cell)
            binm = (min(bybin, key=lambda k: (abs(k - e1b), k)) if bybin else None)
            out[(arm, cell)] = dict(
                n_tot=len(recs), n_split=len(sp), n_un=len(un), n_amb=amb,
                sp_p50=m3.pctl(sp, .50) if sp else float("nan"),
                sp_p95=m3.pctl(sp, .95) if sp else float("nan"),
                un_p50=m3.pctl(un, .50) if un else float("nan"),
                all_p50=m3.pctl([r[0] for r in recs], .50),
                n_reps=sum(1 for rp in reps if not rp["missing"]),
                binmatch=binm, e1_modal_batch=e1b,
                n_binmatch=len(bybin[binm]) if binm is not None else 0,
                sp_p50_binmatch=(m3.pctl(bybin[binm], .50)
                                 if binm is not None else float("nan")),
                gateB=gb)
    return out


_E1_BATCH = {}


def e1_modal_batch(O, arm, cell):
    """median decode batch during SPLIT-labelled E1 intervals (for the
    batch-matched C2 companion only)."""
    key = (arm, cell)
    if key in _E1_BATCH:
        return _E1_BATCH[key]
    vals = []
    for b in BLOCKS:
        p = O[(arm, cell, b)]
        tag = f"e1m3_{arm}_{cell}_b{b}_{E1_JOB}"
        t0b, rows = m3.load_telemetry(f"{HERE}/{tag}_telemetry.jsonl")
        ct = np.array([r[0] - t0b - p["lag"] for r in rows])
        dsm = np.array([r[1] if r[1] is not None else -1 for r in rows])
        drb = np.array([r[2] for r in rows], dtype=float)
        mask = (dsm == p["Dsm"])
        # sample the decode batch wherever the target partition is realized
        v = drb[mask]
        vals += [x for x in v if x > 0]
    _E1_BATCH[key] = int(round(float(np.median(vals)))) if vals else 1
    return _E1_BATCH[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--only", default="all",
                    choices=["all", "row4", "rows123", "row5", "context", "diag", "row4c"])
    a = ap.parse_args()

    print("S0-R INDEPENDENT REPLICATION -- DIAGNOSTIC ONLY (prereg sec 5).")
    print(f"  E1 job {E1_JOB} ({HERE})")
    print(f"  C2 job {C2_JOB} ({SCALEUP})")
    print("  Reused primitives (prereg sec 1): m3_conditional.label_probe, .tokens,")
    print("  .sel_split/.sel_unsplit, .pctl (+ transitively load_telemetry,")
    print("  replay_arrivals, align, label_probe's inner frac).  Everything else")
    print("  (mode estimator, C2 reader, lag driver, reporting) is new here.")
    print("  Bin origin PRIMARY=0.0 ms grid; alt-anchor at 1.15*p50 reported too.")
    print("  Slow share PRIMARY=in-window (1.15*p50,60]; unbounded reported too.")

    O = build_e1(a.cache)

    sec("[GATE G-A] my wfrac == label_probe's split_frac, element-wise (T8 d16 b1)")
    g = gate_A(O)
    print(f"  n_mine={g['n_mine']} n_theirs={g['n_theirs']} "
          f"max_abs_diff={g['max_abs_diff']:.3e} -> {'PASS' if g['ok'] else 'FAIL'}")

    C2 = collect_c2(O)

    sec("[GATE G-B] my C2 ITL reconstruction == s0dc_client's own summary")
    for (arm, cell), c in sorted(C2.items()):
        for tag, mn, tn, mp, tp, st in c["gateB"]:
            if st == "NO_ANCHOR":
                print(f"  {tag:>28}: NO CLIENT ANCHOR -> rep excluded")
            else:
                print(f"  {tag:>28}: n {mn} vs {tn} | p50 {mp:.4f} vs {tp:.4f} -> {st}")
    for arm, cell, tags in C2_SKIPPED:
        print(f"  {arm} {cell}: ALL REPS LACK t0_monotonic_s -> C2 comparand not "
              f"reconstructible ({', '.join(tags)})")

    if a.only in ("all", "row4"):
        report_row4(O)
    if a.only in ("all", "row4c"):
        report_row4c(O)
    if a.only in ("all", "diag"):
        report_diag(O)
    if a.only in ("all", "rows123"):
        report_row123(O, C2)
    if a.only in ("all", "context"):
        report_context(O, C2)
    if a.only in ("all", "row5"):
        report_row5(O)

    print("\nDIAGNOSTIC ONLY -- replicate / fail-to-replicate per prereg row. "
          "No performance\nclaim, no G_LEVER/G_FLAT, no (i)/(ii) adjudication, "
          "no canon edit.")


if __name__ == "__main__":
    main()
