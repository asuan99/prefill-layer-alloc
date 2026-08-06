#!/usr/bin/env python
"""Analyzer for decode_knee_vs_ctx_v2 (P5 re-measurement, 2026-08-04).

GATES FIRST, NUMBERS SECOND.  This script prints per-cell values only after the
instrumentation gates pass, and prints NO knee / NO composition verdict if any
gate fails or cannot be scored.  The verdict on what the numbers mean is
result-analyst's, not this script's -- it reports, it does not conclude.

  GATE C (closure).  sum(disjoint buckets) / (independent event pair around the
  whole layer loop) >= 0.85 for every cell (calibrated 2026-08-05 from job 873617
  smoke: healthy population min/p50/max = 0.9390/0.9403/0.9435 vs legacy defective
  bucket recomputed = 0.6997; 0.85 sits in the gap, not at the healthy floor).
  This is the runtime check that job 858811 lacked; its "attn" bucket covered
  only the RadixAttention core.

  GATE N (negative control).  Mamba2 decode is a fixed-size recurrent update:
  per-mamba-layer time must be O(1) in ctx AT FIXED BATCH.  Scored per
  (realised bs) group -- see D4 below.

Also reported: `per_attn` (new bucket = the whole attention module) next to
`per_attn_core` (the pre-2026-08-04 bucket = RadixAttention core only), so the
magnitude of defect 1 is visible rather than inferred.

------------------------------------------------------------------------------
TOOLING FIXES 2026-08-05 (v2.1) -- mechanical only, NO re-measurement design
choice is made here.  Source of the defect list: results/r0c/P5_GATES_BATCH1_
2026-08-05.md (result-analyst, UNAUDITED) sections 2.1/2.3/2.5/6.3.

  D1  The analyzer used to print `GATE C: PASS` and then `GATES: closure=FAIL`,
      because it summed the harness' `ARMDONE closure_fail=` counter, and that
      counter was greped with `mode=$SM@${CTX}r`, which ALSO matches the
      discarded warm-up tag `rw`.  Warm-up blocks are now EXCLUDED from scoring
      and REPORTED SEPARATELY (they retain diagnostic value).
      * Side fact worth keeping (P5 report section 4): the one warm-up failure in job
        873783 is `mode=44@1024rw blk=0 closure=0.7119`, and the legacy defective
        recompute is 0.6997.  So ~0.70 has AT LEAST TWO PRODUCERS (a mis-bucketed
        span set, and a one-off host stall).  GATE C's discriminating power
        against defect 1 therefore rests on the deficit being PERSISTENT, not on
        the value ~0.70 being diagnostic.
  D2  `mamba_spread` was max/min over REPS, i.e. identically 1.00 at n_reps=1
      (methodology gate #6: a test that always passes is an identity, not
      evidence).  It is now `rep_spread`, printed as `N/A` when n_reps < 2, and
      the within-cell block-to-block spread is a SEPARATE, differently named
      field (`blk_spr%`).  The two quantities never share a name.
  D3  CELL `bs` was `statistics.mode`, which silently labelled a
      6 x bs=19 + 8 x bs=13 mixture as `bs=13`.  The per-cell line now carries
      `bs_mode`, `bs_set` (every realised value with its block count) and
      `bs_med`.  No single-value batch field is printed alone.
  D4  GATE N confounded ctx with realised batch: within one ctx16384 cell, bs
      19->13 alone moves per_mamba by 1.069/1.104/1.139/1.163/1.180 (full/44/24/
      16/8), i.e. 3 of 5 arms cross the 1.15 threshold on batch alone.  The gate
      is now CONDITIONAL ON (ctx, realised bs): only cells with the SAME realised
      bs are compared across ctx.  A group with fewer than 2 distinct ctx is
      `UNSCOREABLE` -- neither PASS nor FAIL.
      * The threshold stays 1.15.  It must NOT be re-tuned against batch 1.
      * The same O(1)-in-ctx check is now ALWAYS run on the complement buckets
        `per_mlp` and `per_other` (methodology gate #10) and printed alongside.
        In batch 1 `per_mlp` is clean (1.002-1.004) while `per_other` fails in
        the same rank order as mamba -- that contrast is what separates "the
        mamba bucket is broken" from "cross-bucket attribution is broken", and
        without it the gate cannot be told apart from estimator insensitivity.
        `per_attn` is printed as a positive control (it MUST move with ctx).
  D5  Host-boundness OBSERVABLES are printed (per-subcell `fwd_ms`, the minimum
      `fwd_ms` over arms at the same (ctx, bs), and their ratio).  These are
      VALUES ONLY.  This script applies NO admissibility cut and drops no cell:
      any host-boundness criterion has to be pre-registered and audited before
      batch 2 (methodology gate #8).

The raw artifacts are never modified; this script only re-reads them.

Usage:  python analyze_decode_knee_vs_ctx_v2.py [JOBID] [--negctrl-max 1.15]
"""
import argparse
import glob
import os
import re
import statistics
import sys

GATE_C_MIN_DEFAULT = 0.85
# Pre-registered before batch 1.  Do NOT re-tune against observed data (#8).
GATE_N_MAX_DEFAULT = 1.15

BLOCK = re.compile(
    r"BLOCK ctx=(?P<ctx>\d+) rep=(?P<rep>\d+) sm=(?P<sm>\S+) round=(?P<round>\d+) .*?"
    r"blk=(?P<blk>\d+) blk_n=(?P<blk_n>\d+) dirty=(?P<dirty>\d+) "
    r"shape=(?P<bs>\d+)x(?P<ntok>\d+) "
    r"closure=(?P<closure>[\d.eE+-]+) closure_min=(?P<closure_min>[\d.eE+-]+) "
    r"fwd_ms=(?P<fwd>[\d.eE+-]+) \| "
    r"b_attn=(?P<b_attn>[\d.eE+-]+) b_mlp=(?P<b_mlp>[\d.eE+-]+) "
    r"b_other=(?P<b_other>[\d.eE+-]+) b_mamba=(?P<b_mamba>[\d.eE+-]+) "
    r"b_attn_core=(?P<b_attn_core>[\d.eE+-]+) \| "
    r"b_per_attn=(?P<per_attn>[\d.eE+-]+) b_per_mamba=(?P<per_mamba>[\d.eE+-]+) "
    r"b_per_attn_core=(?P<per_attn_core>[\d.eE+-]+)"
)
HDR = re.compile(r"HEADER (\S+?)=(.*)")
# mode tag = "<sm>@<ctx>r<round>", round == "w" for the DISCARDED warm-up round.
CLOSURE_FAIL = re.compile(
    r"ZBLT_CLOSURE_FAIL mode=(?P<sm>[^@\s]+)@(?P<ctx>\d+)r(?P<round>\w+)\s"
    r".*?closure=(?P<closure>[\d.eE+-]+)"
)
SRVLOG_NAME = re.compile(r"deckneectx2_srv_C(?P<ctx>\d+)_r(?P<rep>\d+)_(?P<job>\d+)\.log$")


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def parse_block(line):
    """One BLOCK record -> dict, or None if the line is not a BLOCK line."""
    m = BLOCK.search(line)
    if not m:
        return None
    g = m.groupdict()
    out = {k: int(g[k]) for k in ("ctx", "rep", "round", "blk", "blk_n", "dirty", "bs", "ntok")}
    out["sm"] = g["sm"]
    for k in ("closure", "closure_min", "fwd", "b_attn", "b_mlp", "b_other", "b_mamba",
              "b_attn_core", "per_attn", "per_mamba", "per_attn_core"):
        out[k] = float(g[k])
    return out


def parse_kv_line(line, tag):
    """`TAG k=v k=v ...` -> {k: v(str)}; None if the line does not start with TAG."""
    if not line.startswith(tag + " "):
        return None
    kv = {}
    for tok in line.split()[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k] = v
    return kv


def parse_closure_fail(line):
    """A server-log ZBLT_CLOSURE_FAIL line -> dict with `warmup` flag, else None.

    D1: the warm-up round carries the tag `rw`; measured rounds carry `r<digit>`.
    The harness' old grep pattern (`...r`) matched both, which is exactly why the
    gate contradicted itself.
    """
    m = CLOSURE_FAIL.search(line)
    if not m:
        return None
    g = m.groupdict()
    return dict(sm=g["sm"], ctx=int(g["ctx"]), round=g["round"],
                warmup=(g["round"] == "w"), closure=float(g["closure"]))


def load(jobid, here):
    """Read result files (+ their server logs).  Read-only: nothing is written."""
    pat = os.path.join(here, f"deckneectx2_result_C*_r*_{jobid or '*'}.txt")
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f"no result files matching {pat}")
    blocks, headers, armdone, cell_lines, warmup_blocks = [], {}, {}, {}, {}
    for f in files:
        for ln in open(f):
            ln = ln.rstrip("\n")
            m = HDR.match(ln.strip())
            if m:
                headers.setdefault(m.group(1), set()).add(m.group(2))
            rec = parse_block(ln)
            if rec:
                blocks.append(rec)
                continue
            kv = parse_kv_line(ln, "ARMDONE")
            if kv:
                armdone[(int(kv["ctx"]), kv["sm"], int(kv["rep"]))] = kv
                continue
            kv = parse_kv_line(ln, "WARMUP")
            if kv:
                warmup_blocks[(int(kv["ctx"]), kv["sm"], int(kv["rep"]))] = int(kv.get("blocks", 0))
                continue
            kv = parse_kv_line(ln, "CELL")
            if kv and kv.get("STATUS") == "OK":
                cell_lines[(int(kv["ctx"]), kv["sm"], int(kv["rep"]))] = kv

    # server-side closure failures, classified into measured vs warm-up (D1)
    srv = {}
    for lf in sorted(glob.glob(os.path.join(here, "deckneectx2_srv_C*_r*_*.log"))):
        m = SRVLOG_NAME.search(lf)
        if not m or (jobid and m.group("job") != str(jobid)):
            continue
        rep = int(m.group("rep"))
        for ln in open(lf, errors="replace"):
            cf = parse_closure_fail(ln)
            if cf:
                srv.setdefault((cf["ctx"], cf["sm"], rep), []).append(cf)
    return files, blocks, headers, armdone, cell_lines, warmup_blocks, srv


def resolve_closure_fails(armdone, srv, srv_files_found):
    """Per (ctx, sm, rep): measured-round and warm-up closure-fail counts (D1).

    Resolution order:
      1. `closure_fail_meas=` / `closure_fail_warmup=` on the ARMDONE line
         (harness >= 2026-08-05, which greps the round-scoped patterns).
      2. the server log, classified by mode tag (works for batch-1 artifacts,
         whose ARMDONE counter is round-ambiguous).
      3. otherwise AMBIGUOUS -- the legacy counter cannot be split, so the
         server-side component of GATE C is UNSCOREABLE (never silently PASS).
    """
    out = {}
    for key, kv in sorted(armdone.items()):
        if "closure_fail_meas" in kv:
            out[key] = dict(measured=int(kv["closure_fail_meas"]),
                            warmup=int(kv.get("closure_fail_warmup", 0)),
                            source="armdone_round_scoped")
            continue
        legacy = int(kv.get("closure_fail", 0))
        if key in srv or srv_files_found:
            recs = srv.get(key, [])
            meas = sum(1 for r in recs if not r["warmup"])
            warm = sum(1 for r in recs if r["warmup"])
            # The legacy pattern matches BOTH round classes, so it must equal the
            # sum of the two.  If it does not, the log scan and the harness counter
            # disagree and neither may be trusted silently.
            if meas + warm != legacy:
                out[key] = dict(measured=None, warmup=None, source="AMBIGUOUS",
                                legacy_count=legacy, srvlog_measured=meas,
                                srvlog_warmup=warm)
            else:
                out[key] = dict(measured=meas, warmup=warm, source="srvlog",
                                legacy_count=legacy)
        elif legacy == 0:
            out[key] = dict(measured=0, warmup=0, source="legacy_zero", legacy_count=0)
        else:
            out[key] = dict(measured=None, warmup=None, source="AMBIGUOUS",
                            legacy_count=legacy)
    return out


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def keep_steady(recs):
    """Steady-state selection, identical to the harness' own CELL rule:
    drop every `dirty` block and the first block of each round."""
    first = {}
    for r in recs:
        first[r["round"]] = min(first.get(r["round"], 10 ** 9), r["blk"])
    return [r for r in recs if r["dirty"] == 0 and r["blk"] != first[r["round"]]]


def summarize(recs, n_attn, n_mamba):
    """Medians over a set of blocks.  Ratios of `per_mlp`/`per_other` are
    invariant to the divisor, so the per-module normalisation only affects the
    printed magnitude, never a gate verdict."""
    med = lambda k: statistics.median([r[k] for r in recs])
    pm = [r["per_mamba"] for r in recs]
    pm_med = statistics.median(pm)
    return dict(
        n=len(recs),
        closure=med("closure"),
        closure_min=min(r["closure_min"] for r in recs),
        fwd=med("fwd"),
        per_attn=med("per_attn"),
        per_attn_core=med("per_attn_core"),
        per_mamba=pm_med,
        per_mlp=statistics.median([r["b_mlp"] / n_attn for r in recs]),
        per_other=statistics.median([r["b_other"] / n_mamba for r in recs]),
        unbkt=med("fwd") * (1.0 - med("closure")),
        blk_spread_pct=(max(pm) - min(pm)) / pm_med * 100.0 if pm_med > 0 else float("nan"),
        bs_set=sorted({r["bs"] for r in recs}),
    )


METRICS = ("closure", "fwd", "per_attn", "per_attn_core", "per_mamba", "per_mlp",
           "per_other", "unbkt")


def batch_mixture(recs):
    """D3: the realised batch of a cell is a MIXTURE, not a scalar.
    Returns mode, the full {bs: n_blocks} map, and the median."""
    bss = [r["bs"] for r in recs]
    counts = {}
    for b in bss:
        counts[b] = counts.get(b, 0) + 1
    return dict(bs_mode=statistics.mode(bss),
                bs_counts=dict(sorted(counts.items(), reverse=True)),
                bs_med=statistics.median(bss),
                mixed=len(counts) > 1)


def fmt_bs_set(counts):
    return ",".join(f"{b}x{n}" for b, n in counts.items())


def rep_spread(cells, ctx, sm, metric="per_mamba"):
    """D2: max/min ACROSS REPS.  Returns None (printed as `N/A`) when fewer than
    two reps exist, because at n_reps=1 it is identically 1.00 and would be read
    as stability (methodology gate #6).  The within-cell block-to-block spread is
    a different quantity and carries a different name (`blk_spread_pct`)."""
    vals = [rec[metric] for (c, s, _), rec in cells.items()
            if c == ctx and s == sm and rec.get("status") == "OK"]
    if len(vals) < 2 or min(vals) <= 0:
        return None
    return max(vals) / min(vals)


def build_cells(blocks, n_attn, n_mamba):
    """(ctx, sm, rep) -> per-cell summary over ALL kept blocks (mixture pooled)."""
    per_key = {}
    for r in blocks:
        per_key.setdefault((r["ctx"], r["rep"], r["sm"]), []).append(r)
    cells = {}
    for (ctx, rep, sm), recs in per_key.items():
        kept = keep_steady(recs)
        if not kept:
            cells[(ctx, sm, rep)] = dict(status="NO_STEADY_BLOCKS", n_raw=len(recs))
            continue
        s = summarize(kept, n_attn, n_mamba)
        s.update(batch_mixture(kept))
        s.update(status="OK", n_raw=len(recs))
        cells[(ctx, sm, rep)] = s
    return cells


def build_subcells(blocks, n_attn, n_mamba):
    """(ctx, sm, realised bs) -> summary, pooled over reps as the median of the
    per-rep sub-cell medians (D4: the gate's unit of comparison)."""
    per_key = {}
    for r in blocks:
        per_key.setdefault((r["ctx"], r["rep"], r["sm"]), []).append(r)
    per_rep = {}
    for (ctx, rep, sm), recs in per_key.items():
        for r in keep_steady(recs):
            per_rep.setdefault((ctx, sm, r["bs"]), {}).setdefault(rep, []).append(r)
    sub = {}
    for key, by_rep in per_rep.items():
        rep_sums = {rep: summarize(recs, n_attn, n_mamba) for rep, recs in by_rep.items()}
        agg = dict(n=sum(s["n"] for s in rep_sums.values()), n_reps=len(rep_sums))
        for m in METRICS:
            vals = [s[m] for s in rep_sums.values()]
            agg[m] = statistics.median(vals)
            # D2: spread ACROSS REPS is only defined for n_reps >= 2; at n_reps=1
            # max/min is identically 1.00 and must not be printed as stability.
            agg[m + "_rep_spread"] = (max(vals) / min(vals)) if len(vals) > 1 and min(vals) > 0 else None
        agg["blk_spread_pct"] = statistics.median([s["blk_spread_pct"] for s in rep_sums.values()])
        sub[key] = agg
    return sub


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------
def gate_n_conditional(subcells, metric, threshold, sms):
    """GATE N, conditional on (ctx, realised bs) -- D4.

    Only sub-cells with the SAME realised bs are compared across ctx.  A bs group
    spanning fewer than 2 ctx values is UNSCOREABLE (not PASS, not FAIL).

    Arm verdict:
      VIOLATION    some scored group exceeds `threshold`
      UNSCOREABLE  no group of this arm could be scored at all
      OK_PARTIAL   scored and clean, but some ctx of this arm never entered ANY
                   scored comparison (its realised batch is unique to it), so
                   that ctx column is simply NOT covered by the gate
      OK           scored, clean, and every ctx of this arm is covered
    Coverage is counted in ctx, not in bs groups: a stray one-off bs value at a
    ctx that is already covered by another group does not make the arm partial.
    """
    results = []
    for sm in sms:
        by_bs = {}
        for (ctx, s, bs), rec in subcells.items():
            if s != sm:
                continue
            by_bs.setdefault(bs, []).append((ctx, rec[metric], rec["n"]))
        groups = []
        for bs in sorted(by_bs, reverse=True):
            pts = sorted(by_bs[bs])
            if len({c for c, _, _ in pts}) < 2:
                groups.append(dict(bs=bs, points=pts, ratio=None, verdict="UNSCOREABLE"))
                continue
            vals = [v for _, v, _ in pts]
            lo, hi = min(vals), max(vals)
            ratio = hi / lo if lo > 0 else float("nan")
            groups.append(dict(bs=bs, points=pts, ratio=ratio,
                               verdict="VIOLATION" if ratio > threshold else "OK"))
        all_ctx = {c for g in groups for c, _, _ in g["points"]}
        covered = {c for g in groups if g["verdict"] in ("OK", "VIOLATION")
                   for c, _, _ in g["points"]}
        uncovered = sorted(all_ctx - covered)
        if any(g["verdict"] == "VIOLATION" for g in groups):
            verdict = "VIOLATION"
        elif not covered:
            verdict = "UNSCOREABLE"
        elif uncovered:
            verdict = "OK_PARTIAL"
        else:
            verdict = "OK"
        results.append(dict(sm=sm, groups=groups, verdict=verdict,
                            uncovered_ctx=uncovered))
    return results


def combine_gate_n(results):
    """PASS only if every arm is fully scored and clean.  An arm that could not be
    scored, or whose ctx coverage is partial, makes the gate UNSCOREABLE -- never
    a silent PASS over a ctx column that was never compared to anything."""
    if not results:
        return "UNSCOREABLE"
    if any(r["verdict"] == "VIOLATION" for r in results):
        return "FAIL"
    if any(r["verdict"] in ("UNSCOREABLE", "OK_PARTIAL") for r in results):
        return "UNSCOREABLE"
    return "PASS"


def gate_c(cells, resolved, threshold):
    """Cell-median closure >= threshold AND zero MEASURED-round server-side
    closure failures.  Warm-up failures are excluded from scoring (D1)."""
    bad = []
    for key, rec in sorted(cells.items()):
        if rec.get("status") != "OK":
            bad.append((key, "no steady blocks"))
        elif rec["closure"] < threshold:
            bad.append((key, f"closure_med={rec['closure']:.4f}"))
    meas = [(k, v["measured"]) for k, v in resolved.items() if v["measured"]]
    ambiguous = [k for k, v in resolved.items() if v["source"] == "AMBIGUOUS"]
    if bad or meas:
        verdict = "FAIL"
    elif ambiguous:
        verdict = "UNSCOREABLE"
    else:
        verdict = "PASS"
    return verdict, bad, meas, ambiguous


def sm_key(sm):
    return 108 if sm == "full" else int(sm)


def host_boundness_rows(subcells):
    """D5: values only.  `fwd_min` is the minimum fwd_ms over ARMS at the same
    (ctx, realised bs).  NO cell is filtered and NO cut-off is applied here."""
    floor = {}
    for (ctx, sm, bs), rec in subcells.items():
        k = (ctx, bs)
        floor[k] = min(floor.get(k, float("inf")), rec["fwd"])
    rows = []
    for (ctx, sm, bs), rec in sorted(subcells.items(),
                                     key=lambda kv: (kv[0][0], sm_key(kv[0][1]), -kv[0][2])):
        f = floor[(ctx, bs)]
        rows.append(dict(ctx=ctx, sm=sm, bs=bs, n=rec["n"], fwd=rec["fwd"],
                         fwd_min=f, fwd_ratio=rec["fwd"] / f if f > 0 else float("nan"),
                         unbkt=rec["unbkt"], closure=rec["closure"]))
    return rows


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("jobid", nargs="?", default=None)
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--closure-min", type=float, default=GATE_C_MIN_DEFAULT)
    ap.add_argument("--negctrl-max", type=float, default=GATE_N_MAX_DEFAULT,
                    help="max allowed max/min across ctx AT FIXED realised bs "
                         "(pre-registered 1.15; do not re-tune against observed data)")
    ap.add_argument("--n-attn", type=int, default=9, help="Zamba2-2.7B: 9 hybrid layers")
    ap.add_argument("--n-mamba", type=int, default=54, help="Zamba2-2.7B: 54 mamba layers")
    a = ap.parse_args(argv)

    files, blocks, headers, armdone, cell_lines, warmup_blocks, srv = load(a.jobid, a.dir)
    cells = build_cells(blocks, a.n_attn, a.n_mamba)
    subcells = build_subcells(blocks, a.n_attn, a.n_mamba)
    resolved = resolve_closure_fails(armdone, srv, srv_files_found=bool(
        glob.glob(os.path.join(a.dir, f"deckneectx2_srv_C*_r*_{a.jobid or '*'}.log"))))

    n_reps = len({rep for _, _, rep in cells})
    print(f"# files={len(files)}  cells={len(cells)}  blocks_parsed={len(blocks)}  n_reps={n_reps}")
    for k in ("sm_order", "block_every", "closure_min", "cudagraph", "seed_base",
              "zamba2_sha256", "git_head"):
        if k in headers:
            vals = sorted(headers[k])
            print(f"# {k}: {vals if len(vals) > 1 else vals[0]}")

    ctxs = sorted({c for c, _, _ in cells})
    sms = sorted({s for _, s, _ in cells}, key=lambda s: 108 if s == "full" else int(s))

    # ---- cross-check against the harness' own CELL lines (parser drift guard)
    dev = []
    for key, kv in cell_lines.items():
        rec = cells.get(key)
        if not rec or rec.get("status") != "OK":
            continue
        for fld, col in (("closure", "closure_med"), ("fwd", "fwd_ms"),
                         ("per_attn", "per_attn"), ("per_mamba", "per_mamba")):
            if col in kv:
                ref = float(kv[col])
                if ref:
                    dev.append(abs(rec[fld] - ref) / ref)
    if dev:
        print(f"# crosscheck vs harness CELL lines: max rel dev = {max(dev):.2e} "
              f"over {len(dev)} values (recomputed from BLOCK lines)")

    # ---- GATE C ----------------------------------------------------------
    c_verdict, bad_closure, meas_fail, ambiguous = gate_c(cells, resolved, a.closure_min)
    print(f"\nGATE C (closure >= {a.closure_min}): {c_verdict}")
    for key, why in bad_closure:
        print(f"  ctx={key[0]} sm={key[1]} rep={key[2]}: {why}")
    for key, n in meas_fail:
        print(f"  ctx={key[0]} sm={key[1]} rep={key[2]}: {n} MEASURED-round "
              f"server-side ZBLT_CLOSURE_FAIL")
    for key in ambiguous:
        print(f"  ctx={key[0]} sm={key[1]} rep={key[2]}: legacy round-ambiguous "
              f"closure_fail counter and no server log -> not scoreable")
    src = sorted({v["source"] for v in resolved.values()})
    print(f"  measured-round closure failures: {sum(v['measured'] or 0 for v in resolved.values())}"
          f"   (source: {','.join(src) if src else 'none'})")

    # D1: warm-up is EXCLUDED from scoring but REPORTED -- it is a diagnostic.
    warm = [(k, v["warmup"]) for k, v in sorted(resolved.items()) if v["warmup"]]
    tot_warm_blocks = sum(warmup_blocks.values())
    print(f"  warm-up (tag `rw`, discarded by construction, NOT scored): "
          f"{sum(n for _, n in warm)} closure failures over {tot_warm_blocks} warm-up blocks")
    for key, n in warm:
        vals = ",".join(f"{r['closure']:.4f}"
                        for r in srv.get((key[0], key[1], key[2]), []) if r["warmup"])
        print(f"    ctx={key[0]} sm={key[1]} rep={key[2]}: n={n} closure={vals}")
    if warm:
        print("    NOTE: a warm-up closure near ~0.70 coincides with the legacy defective "
              "recompute (0.6997), so ~0.70 has at least two producers (mis-bucketed span "
              "set; one-off host stall). GATE C's power against defect 1 rests on the "
              "deficit being PERSISTENT, not on the value.")

    # ---- per-cell table --------------------------------------------------
    print("\n# per-cell medians over kept blocks (ms); per_attn = WHOLE attention module,")
    print("# per_attn_core = pre-2026-08-04 bucket (RadixAttention core only)")
    print("# bs_set = every realised batch size with its block count (D3: the cell may be a MIXTURE)")
    print(f"{'ctx':>6} {'sm':>5} {'rep':>3} {'nblk':>5} {'bs_mode':>7} {'bs_med':>6} "
          f"{'bs_set':>12} {'clo':>6} {'fwd_ms':>8} {'per_attn':>9} {'core':>8} "
          f"{'per_mamba':>10} {'blk_spr%':>8} {'rep_spread':>10}")
    for ctx in ctxs:
        for sm in sms:
            for rep in sorted({r for _, _, r in cells}):
                rec = cells.get((ctx, sm, rep))
                if not rec:
                    continue
                if rec.get("status") != "OK":
                    print(f"{ctx:>6} {sm:>5} {rep:>3}  {rec['status']}")
                    continue
                # D2: rep_spread is across REPS; identity-valued at n_reps=1 -> N/A.
                spread = rep_spread(cells, ctx, sm)
                rs = "N/A" if spread is None else f"{spread:.3f}"
                print(f"{ctx:>6} {sm:>5} {rep:>3} {rec['n']:>5} {rec['bs_mode']:>7} "
                      f"{rec['bs_med']:>6.1f} {fmt_bs_set(rec['bs_counts']):>12} "
                      f"{rec['closure']:>6.3f} {rec['fwd']:>8.3f} {rec['per_attn']:>9.4f} "
                      f"{rec['per_attn_core']:>8.4f} {rec['per_mamba']:>10.4f} "
                      f"{rec['blk_spread_pct']:>8.2f} {rs:>10}")
    if n_reps < 2:
        print("# rep_spread = N/A: max/min over reps is identically 1.00 at n_reps=1 "
              "(methodology gate #6). Within-cell block spread is blk_spr% -- a DIFFERENT quantity.")

    # ---- GATE N (conditional on realised bs) + complement controls -------
    print(f"\nGATE N (per-bucket O(1) in ctx AT FIXED realised bs; max/min <= {a.negctrl_max})")
    print("  scored per (ctx, realised bs); a bs group spanning <2 ctx is UNSCOREABLE.")
    print("  complement negative controls (per_mlp, per_other) are ALWAYS scored alongside")
    print("  (methodology gate #10); per_attn is a POSITIVE control that must move.")
    gate_n_results = gate_n_conditional(subcells, "per_mamba", a.negctrl_max, sms)
    controls = {m: gate_n_conditional(subcells, m, a.negctrl_max, sms)
                for m in ("per_mlp", "per_other", "per_attn")}

    # For the POSITIVE control the polarity is inverted: exceeding the threshold is
    # the expected, healthy outcome, so it is not printed as "VIOLATION".
    POS = {"VIOLATION": "MOVES(ok)", "OK": "FLAT(suspect)", "OK_PARTIAL": "FLAT(suspect)"}

    def show(tag, results, gated):
        print(f"\n  [{tag}] {'GATED' if gated else 'informational only'}")
        for r in results:
            for g in r["groups"]:
                detail = " ".join(f"c{c}={v:.4f}(n={n})" for c, v, n in g["points"])
                ratio = "   n/a" if g["ratio"] is None else f"{g['ratio']:6.3f}"
                v = g["verdict"] if gated else POS.get(g["verdict"], g["verdict"])
                print(f"    sm={r['sm']:>5} bs={g['bs']:>3} max/min={ratio} "
                      f"{v:<13} {detail}")
            note = ""
            if r["uncovered_ctx"]:
                note = ("   (ctx " + ",".join(str(c) for c in r["uncovered_ctx"])
                        + " has no comparable same-bs cell -> NOT covered)")
            v = r["verdict"] if gated else POS.get(r["verdict"], r["verdict"])
            print(f"    sm={r['sm']:>5} ARM VERDICT: {v}{note}")

    show("per_mamba  (GATE N)", gate_n_results, True)
    show("per_mlp    (complement negative control)", controls["per_mlp"], True)
    show("per_other  (complement negative control)", controls["per_other"], True)
    show("per_attn   (positive control: MUST move with ctx)", controls["per_attn"], False)

    def tally(results):
        counts = {}
        for r in results:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        return " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))

    n_verdict = combine_gate_n(gate_n_results)
    print(f"\n  GATE N per_mamba = {n_verdict}   [arms: {tally(gate_n_results)}]")
    for m in ("per_mlp", "per_other"):
        print(f"  complement control {m} = {combine_gate_n(controls[m])}   "
              f"[arms: {tally(controls[m])}]")
    print("  (a complement control that fails WITH per_mamba points at cross-bucket "
          "attribution, not at the mamba bucket; one that stays clean while per_mamba "
          "fails is what distinguishes a real violation from estimator insensitivity.)")

    # cross-arm sanity that 858811 also failed: full(108 SM) must not be slower
    # than a small partition on the SAME ctx AND the same realised bs.
    for ctx in ctxs:
        for bs in sorted({b for (c, _, b) in subcells if c == ctx}, reverse=True):
            if (ctx, "full", bs) in subcells and (ctx, "44", bs) in subcells:
                r = subcells[(ctx, "full", bs)]["per_mamba"] / subcells[(ctx, "44", bs)]["per_mamba"]
                if r > 1.0:
                    print(f"  NOTE ctx={ctx} bs={bs}: per_mamba full/sm44 = {r:.2f} "
                          f"(>1 means 108 SM is SLOWER than 44 SM -- physically suspect)")

    # ---- D5: host-boundness observables (VALUES ONLY) --------------------
    print("\n# HOST-BOUNDNESS OBSERVABLES (D5) -- values only. NO admissibility cut is applied")
    print("# and no cell is dropped here; any such criterion must be pre-registered and")
    print("# audited before batch 2 (methodology gate #8). fwd_min = min fwd_ms over ARMS")
    print("# at the same (ctx, realised bs); unbkt_ms = fwd_ms * (1 - closure).")
    print(f"{'ctx':>6} {'sm':>5} {'bs':>3} {'n':>3} {'fwd_ms':>9} {'fwd_min':>9} "
          f"{'fwd/min':>8} {'unbkt_ms':>9}")
    for row in host_boundness_rows(subcells):
        print(f"{row['ctx']:>6} {row['sm']:>5} {row['bs']:>3} {row['n']:>3} "
              f"{row['fwd']:>9.3f} {row['fwd_min']:>9.3f} {row['fwd_ratio']:>8.3f} "
              f"{row['unbkt']:>9.3f}")

    print(f"\nGATES: closure={c_verdict} negctrl={n_verdict}")
    if c_verdict != "PASS" or n_verdict != "PASS":
        print("\nNO COMPOSITION / KNEE REPORTED: at least one instrumentation gate did not "
              "pass, so per-type numbers from this run are not a measurement.")
        return 1

    # descriptive only -- no verdict, no policy claim
    print(f"\n# step composition (descriptive; n_attn={a.n_attn} n_mamba={a.n_mamba}, "
          "eager/no-cudagraph)")
    print(f"{'ctx':>6} {'sm':>5} {'bs':>3} {'attn_share_of_bucketed':>24}")
    for (ctx, sm, bs), rec in sorted(subcells.items()):
        tot = rec["per_attn"] * a.n_attn + rec["per_mamba"] * a.n_mamba
        print(f"{ctx:>6} {sm:>5} {bs:>3} {rec['per_attn'] * a.n_attn / tot:>24.3f}")
    print("\n(Interpretation, knee selection and any policy statement are "
          "result-analyst's call, not this script's.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
