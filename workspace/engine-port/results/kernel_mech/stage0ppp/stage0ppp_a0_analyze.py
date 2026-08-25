#!/usr/bin/env python3
"""Stage 0''' A0 -- THE ADAPTER.  sqlite + raw JSON -> World -> registered rule.

This file owns ONLY the mapping from measurement to the World axes.  It does not
own the decision: it imports `score()`/`companion()` from stage0ppp_a0_rule.py,
which is the registered canon (prereg sec3).  If the two ever disagree, the rule
wins -- so the rule is imported, never re-implemented here (the P0-A precedent).

FAIL-CLOSED: every axis this file cannot determine from the artefacts maps to a
MEASUREMENT condition, never to a substantive value.  A missing sqlite column is
a fact about the export, not about the driver (gate #21).
"""
import argparse, hashlib, json, os, sqlite3, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage0ppp_a0_rule as RULE

KERN = "CUPTI_ACTIVITY_KIND_KERNEL"
RUNT = "CUPTI_ACTIVITY_KIND_RUNTIME"
NEEDED = ("start", "end", "streamId", "correlationId", "graphNodeId")


def _cols(cx, table):
    try:
        return {r[1] for r in cx.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def _rows(cx, names):
    """Kernel rows joined to their demangled name, one dict per row."""
    c = _cols(cx, KERN)
    missing = [n for n in NEEDED if n not in c]
    if "demangledName" not in c:
        missing.append("demangledName")
    if _cols(cx, "StringIds") == set():
        missing.append("StringIds(table)")
    # B8: a MISSING greenContextId column is a fact about the export schema.
    # rev4 let it fall through as ctx="null" and scored a substantive label --
    # nsys itself errors with "does not contain 'greenContextId' column", so
    # this world is real.
    if "greenContextId" not in c:
        missing.append("greenContextId")
    if missing:
        return None, f"kernel table lacks {missing}"
    green = "greenContextId"
    ctxc = "contextId" if "contextId" in c else None
    sel = ["k.start", "k.end", "k.streamId", "k.correlationId", "k.graphNodeId",
           (f"k.{green}" if green else "NULL"), (f"k.{ctxc}" if ctxc else "NULL"),
           "s.value"]
    q = (f"SELECT {', '.join(sel)} FROM {KERN} k "
         f"LEFT JOIN StringIds s ON s.id = k.demangledName")
    out = []
    try:
        cur = cx.execute(q)
    except sqlite3.Error as e:      # B9: never a traceback, always a label
        return None, f"kernel query failed: {e}"
    for st, en, sid, cid, gnid, gctx, ctx, nm in cur:
        nm = nm or ""
        for tag in names:
            if tag in nm:
                out.append(dict(start=st, end=en, stream=sid, corr=cid,
                                node=gnid or 0, gctx=gctx, ctx=ctx, leg=tag))
                break
    return out, None


def _launch_rows(cx):
    """RUNTIME launch rows, split into REPLAY-time and CAPTURE-time.

    Parent E5: a correlationId that joins only to a capture-time individual
    launch has the field populated and is still useless for membership.  The
    two sets must be kept apart or `KSET_JOINS_CAPTURE_NOT_REPLAY` can never
    fire and the world is silently scored as if the join worked.
    """
    c = _cols(cx, RUNT)
    if not {"start", "correlationId"} <= c:
        return None, None
    q = (f"SELECT r.start, r.correlationId, s.value FROM {RUNT} r "
         f"LEFT JOIN StringIds s ON s.id = r.nameId")
    rep, cap = set(), set()
    try:
        cur = cx.execute(q)
    except sqlite3.Error:           # B9
        return None, None
    for st, cid, nm in cur:
        nm = nm or ""
        if "GraphLaunch" in nm:
            rep.add(cid)
        elif "LaunchKernel" in nm:
            cap.add(cid)
    return rep, cap


def _green_axis(raw):
    """prereg sec5 / N6: judged by the PROBE's driver readout, never by nsys."""
    if raw.get("green_create") != "ok":
        return "absent", "green_create failed"
    gr = raw.get("green_readout") or {}
    if "green_ctx_is_null" not in gr:
        return None, f"green readout absent -> fail-closed ({gr.get('error')})"
    if gr["green_ctx_is_null"]:
        return "absent", "stream carries no green context"
    tgt = (raw.get("green_target") or {}).get("decode_sm")
    got = (gr.get("green_sm") or {}).get("smCount")
    if tgt is None or got is None:
        return None, "no realized green SM count -> fail-closed"
    return ("matched" if int(got) == int(tgt) else "mismatched"), f"realized={got} target={tgt}"


def build(raw, sqlite_path, stats_warn, stats_ok=True):
    """raw JSON + sqlite -> (World kwargs, diagnostics).  Fail-closed."""
    d = {"diag": {}}
    if not raw or not raw.get("legs"):
        return None, {"reason": "probe raw absent"}
    if not os.path.exists(sqlite_path):
        return dict(run_ok=True, export="fail"), {"reason": "sqlite absent"}
    cx = sqlite3.connect(sqlite_path)
    rows, err = _rows(cx, ("a0_l1_eager_full", "a0_l2_graph_full",
                           "a0_l2p_graph_full", "a0_l3_eager_green",
                           "a0_l4_graph_green"))
    if rows is None:
        return dict(run_ok=True, export="fail"), {"reason": err}
    rep_corr, cap_corr = _launch_rows(cx)

    L4 = [r for r in rows if r["leg"] == "a0_l4_graph_green" and r["node"]]
    L1 = [r for r in rows if r["leg"] == "a0_l1_eager_full"]
    L3 = [r for r in rows if r["leg"] == "a0_l3_eager_green"]
    # B3: rev4 split L2/L2' by position around the L4 rows, so an EMPTY L4 --
    # the registered decisive negative -- made L2' look empty and scored
    # TRACE_TRUNCATED.  The harness erased the answer and then forbade stating
    # it.  L2' now carries its own kernel name and nothing is derived from L4.
    L2 = [r for r in rows if r["leg"] == "a0_l2_graph_full" and r["node"]]
    L2p = [r for r in rows if r["leg"] == "a0_l2p_graph_full" and r["node"]]
    d["diag"] = {"n_rows": len(rows), "L1": len(L1), "L2": len(L2),
                 "L3": len(L3), "L4": len(L4), "L2p": len(L2p),
                 "eager_contamination": sum(1 for r in rows
                                            if r["leg"].startswith("a0_l4") and not r["node"])}

    green, why = _green_axis(raw)
    d["diag"]["green_why"] = why
    if green is None:
        # B15: rev4 labelled this export="fail", mis-attributing a probe-side
        # channel failure to the trace.  It is a probe condition.
        return dict(run_ok=True, green="absent", export="ok", l1="ok", l2="ok",
                    l3="ok", l2_post="ok", capture="ok",
                    profile="empty", ctx="null", stream="mismatch",
                    join_target="none", join_rate=0.0, l2_join="fail"), \
               {"reason": why, **d}

    cap4 = (raw["legs"].get("L4") or {}).get("L4_capture")
    cap2 = (raw["legs"].get("L2") or {}).get("L2_capture")
    capture = ("fail_all" if cap2 == "fail"
               else "fail_green_only" if cap4 == "fail" else "ok")

    # expectation is MEASURED in L2, per replay, via the graph-launch join
    # B7: rev4 set l2_join="ok" whenever ANY GraphLaunch row existed anywhere
    # and L2 had any node row -- so EXPECTATION_UNVERIFIABLE (the R6 repair)
    # was unreachable.  The join must be measured ON L2.
    l2_join, expected = "fail", None
    per = {}
    for r in L2:
        if r["corr"] in (rep_corr or set()):
            per[r["corr"]] = per.get(r["corr"], 0) + 1
    if per:
        l2_join = "ok"
        v = sorted(per.values())
        expected = v[len(v) // 2]
    d["diag"]["l2_joined_replays"] = len(per)
    d["diag"]["expected_nodes_per_replay"] = expected
    d["diag"]["expectation_basis"] = "median_per_replay" if l2_join == "ok" else "unavailable"

    # profile: per-replay vector on L4 (X5) -- uniform vs tail vs excess
    profile = "empty"
    if L4 and not any(r["node"] for r in L4):
        profile = "eager_only"
    elif expected and L4:
        by = {}
        for r in L4:
            by[r["corr"]] = by.get(r["corr"], 0) + 1
        seq = [by.get(k, 0) for k in sorted(by)]
        frac = len(L4) / float(expected * RULE.REPLAYS_EXPECTED)
        # B4: rev4 bucketed by adapter thresholds (0.995/0.905/0.885) that did
        # not agree with PROFILE_FRAC -- a measured frac of exactly 0.900 was
        # scored on the WRONG side of Q1_FRAC.  And `tail_missing` was
        # `len(seq) < 20`, which on a one-node graph swallowed every partial
        # world, so Q1_FRAC was never evaluated at all.  Now: the tail test
        # requires a genuine SUFFIX deficit, and everything else is placed by
        # the measured frac against the registered constants themselves.
        tail = (len(seq) < RULE.REPLAYS_EXPECTED
                or (len(seq) >= 2 and seq[-1] < max(seq)))
        if frac > 1.0:
            profile = "excess"
        elif tail and frac < 1.0:
            profile = "tail_missing"
        elif frac >= 1.0:
            profile = "full"
        elif frac >= RULE.Q1_FRAC:
            profile = "just_above"     # >= 0.90 -> Q1 passes
        elif frac >= 0.885:
            profile = "just_below"     # < 0.90 -> Q1 fails
        else:
            profile = "uniform_short"
        d["diag"]["profile_reason"] = f"frac={frac:.4f} tail={tail} seq={seq[:5]}"
        d["diag"]["frac"] = frac
        d["diag"]["per_replay"] = seq
    elif L4 and not any(r["node"] for r in L4):
        profile = "eager_only"

    # B5: rev4 mapped `parent` to "two or more distinct green ids", so
    # ATTRIBUTION_DISCONFIRMED fired on an unregistered event and never on the
    # registered one.  nsys' own reports use NULLIF(greenContext, 0), i.e. 0
    # ENCODES "not green" -- so gctx in {NULL, 0} IS the disconfirming
    # evidence, corroborated by the row carrying the same contextId as the
    # full-GPU legs.  The contextId column was already being selected and
    # thrown away.
    gset = {r["gctx"] for r in L4}
    real = gset - {None, 0}
    parent_ctx = {r["ctx"] for r in rows
                  if r["leg"] in ("a0_l1_eager_full", "a0_l2_graph_full")}
    l4_ctx = {r["ctx"] for r in L4}
    d["diag"]["ctx_multivalued"] = len(real) > 1
    if len(real) == 1:
        ctx = "distinct"
    elif len(real) > 1:
        ctx = "null"               # outside the modelled space; see diagnostics
    elif l4_ctx and parent_ctx and l4_ctx == parent_ctx:
        ctx = "parent"             # positive evidence: these rows are NOT green
    else:
        ctx = "zero" if gset <= {0, None} and 0 in gset else "null"
    s4 = {r["stream"] for r in L4}
    s3 = {r["stream"] for r in L3}
    stream = "match" if (s4 and s3 and s4 == s3) else "mismatch"

    jrep = sum(1 for r in L4 if r["corr"] in (rep_corr or set()))
    jcap = sum(1 for r in L4 if r["corr"] in (cap_corr or set()))
    if not L4 or (jrep == 0 and jcap == 0):
        join_target, join_rate = "none", 0.0
    elif jrep == 0:
        join_target, join_rate = "capture_only", 0.0      # parent E5
    else:
        join_target, join_rate = "replay", jrep / float(len(L4))
    d["diag"].update(dict(ctx_values=[str(x) for x in gset], stream_l4=list(s4),
                          stream_l3=list(s3), join_rate=join_rate,
                          joined_replay=jrep, joined_capture=jcap,
                          n_replay_corr=len(rep_corr or ()),
                          n_capture_corr=len(cap_corr or ())))

    return dict(run_ok=True,
                export=("partial" if stats_warn else
                        "partial" if not stats_ok else "ok"),
                l1="ok" if L1 else "zero", l2="ok" if L2 else "zero",
                l3="ok" if L3 else "zero", l2_post="ok" if L2p else "zero",
                capture=capture, green=green, profile=profile, ctx=ctx,
                stream=stream, join_target=join_target, join_rate=join_rate,
                l2_join=l2_join), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-node", required=True)
    ap.add_argument("--sqlite-node", required=True)
    # B6: the `graph` granularity run is not taken in this submission (sec4
    # moved to A1).  These stay accepted-but-optional so a later run that does
    # take it needs no interface change.
    ap.add_argument("--raw-graph", default=None)
    ap.add_argument("--sqlite-graph", default=None)
    ap.add_argument("--stats-warn-node", default="")
    ap.add_argument("--stats-rc-node", default="0")
    ap.add_argument("--stats-rc-graph", default="0")
    ap.add_argument("--stats-warn-graph", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="local")
    a = ap.parse_args()

    def _load(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None

    rn = _load(a.raw_node)
    rg = _load(a.raw_graph) if a.raw_graph else None
    kw, diag = build(rn, a.sqlite_node, bool(a.stats_warn_node.strip()),
                     stats_ok=(a.stats_rc_node == "0"))
    if kw is None:
        label, basis = RULE.ABSENT, None
        world = None
    else:
        world = RULE.World(**kw)
        label, basis = RULE.score(world)

    verdict = {"kind": "stage0ppp_a0_verdict", "tag": a.tag,
               "rule_rev": RULE.RULE_REV,
               "rule_sha256": hashlib.sha256(
                   open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "stage0ppp_a0_rule.py"), "rb").read()).hexdigest(),
               "analyzer_sha256": hashlib.sha256(
                   open(os.path.abspath(__file__), "rb").read()).hexdigest(),
               "substrate": "synthetic_probe_graph",
               "world": kw, "diagnostics": diag,
               "label": label, "basis": basis,
               "raw_node": rn, "raw_graph": rg,
               "companion_scored": False,
               "companion_note": ("granularity=graph not taken in this run; "
                                  "E1-(b) deferred to A1 (harness audit B6)"),
               "NOT_A_PERFORMANCE_RESULT": (
                   "tool-validity probe; no server, no model, no request. "
                   "A positive does NOT transfer to the engine decode graph "
                   "(prereg sec0-1)."),
               }
    with open(a.out, "w") as f:
        json.dump(verdict, f, indent=1, sort_keys=True, default=str)
    print(f"[a0-analyze] label={label} basis={basis}")
    print(f"[a0-analyze] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
