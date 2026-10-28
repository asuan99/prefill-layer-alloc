#!/usr/bin/env python3
"""C2: VESSL runs -- how often did a drain (adjust path: decode-empty / merge / admission, or SLO/R2 switch)
wait on IN-FLIGHT prefill, and how long could it have lasted?

Part A (ground truth, PHASE_DIAG `drain` events, engine 3858f129-equivalent tree 71f5006):
    results/engine_gate/pdmux-policy-gate-smoke-71f5006-20261028064810/{on,on_slo,static_ns}/tel.jsonl
Part B (reconstruction from decode_iteration + prefill_span_start/end, no drain events):
    N3 (n3_capacity/n3b_*), N2 (n2_static_grid/n2p_*), policy smoke stage2 (cap_char/policy_runs/*/smoke/*),
    and Part-A files again (to validate the estimator against the ground truth).

Estimator (code facts, multiplexing_mixin.py event_loop_pdmux, identical drain structure in all commits used):
  * decode_iteration is emitted only when decode ran; a decode-EMPTY interval leaves no record.
  * B2 in-flight drain <=> decode batch empties while a split prefill is in flight. Signature on a span
    (t0=span_start, t1=span_end, r0=running_bs at start): r0 > 0, the old decode requests are all gone at merge
    (first decode_iteration after t1 has bs <= n_done), and decode stopped before t1.
    Upper bound of that drain = gap = t1 - t_sync(last decode_iteration inside the span)  [host clock];
    it also contains the empty-loop spin, the remaining chunk issue and the <=1-iteration span_end observation lag.
  * B1-type (switch while chunks in flight): a split_transition (R2 policy) timestamp that falls AFTER the first
    in-span decode_iteration of a span that is not yet in kernel-done-wait.  A transition before the first in-span
    decode_iteration is on the admission iteration (prefill stream empty).
"""
import glob, json, math, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent
N_LAYERS = 56          # Nemotron-Nano-9B-v2 [cited lockstep_diag.py:57]
BUDGET = 65536


def q(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def dist(xs):
    return dict(n=len(xs), p50=q(xs, .5), p95=q(xs, .95), max=(max(xs) if xs else None),
                sum=(sum(xs) if xs else 0.0))


def load(path):
    it, st, en, tr, dr = [], {}, {}, [], []
    for ln in open(path):
        if not any(k in ln for k in ("decode_iteration", "prefill_span_", "split_transition", '"drain"')):
            continue
        d = json.loads(ln)
        e = d.get("event")
        if e not in ("decode_iteration", "prefill_span_start", "prefill_span_end", "split_transition", "drain"):
            continue
        if d.get("phase") not in (None, "benchmark"):
            continue
        if e == "decode_iteration":
            it.append(d)
        elif e == "prefill_span_start":
            st.setdefault(d["span_seq"], d)
        elif e == "prefill_span_end":
            en.setdefault(d["span_seq"], d)
        elif e == "split_transition":
            tr.append(d["timestamp_monotonic_s"])
        else:
            dr.append(d)
    it.sort(key=lambda x: x["t_launch"])
    spans = []
    for s, a in sorted(st.items()):
        b = en.get(s)
        if b is None:
            continue
        spans.append(dict(seq=s, t0=a["timestamp_monotonic_s"], t1=b["timestamp_monotonic_s"], r0=int(a["running_bs"]),
                          ext=int(a["extend_tokens"]), n_done=int(b.get("n_done", 0))))
    return it, spans, tr, dr


def n_chunks(ext):
    fc = max(1, BUDGET // ext) if ext > 0 else N_LAYERS
    return math.ceil(N_LAYERS / min(fc, N_LAYERS))


def analyse(path):
    it, spans, tr, dr = load(path)
    tl = [x["t_launch"] for x in it]
    import bisect
    res = dict(file=str(Path(path).relative_to(RES)), n_spans=len(spans), n_iters=len(it),
               n_spans_r0_zero=0, b2=[], n_split_transition=len(tr), tr_admission=0, tr_midspan=0, tr_outside=0)
    for sp in spans:
        i0 = bisect.bisect_right(tl, sp["t0"])
        i1 = bisect.bisect_left(tl, sp["t1"])
        inside = it[i0:i1]
        # B1-type proxy for ANY controller (incl. PDMUX_SLO_SCHED, which emits no split_transition):
        # stream_idx change between two consecutive in-span decode iterations while a prefill is in flight.
        for a, b in zip(inside, inside[1:]):
            if a["stream_idx"] != b["stream_idx"] and a["prefill_in_flight"] and b["prefill_in_flight"]:
                res.setdefault("idx_change_inflight", []).append(dict(
                    span=sp["seq"], frm=a["stream_idx"], to=b["stream_idx"],
                    gap_ms=(b["t_launch"] - a["t_sync"]) * 1000.0,
                    chunks_issued_before=sum(1 for x in inside if x["t_launch"] <= a["t_launch"]
                                             and x["prefill_in_flight"] and not x["prefill_kernel_done_wait"])))
        if sp["r0"] == 0:
            res["n_spans_r0_zero"] += 1
            continue
        after = it[i1] if i1 < len(it) else None
        if after is None:
            continue
        emptied = after["bs"] <= sp["n_done"]
        if not emptied:
            continue
        if not inside:
            # r0 counted finished-but-unfiltered requests (span_start is emitted before update_running_batch):
            # decode was empty from the admission iteration -> adjust drain on an EMPTY prefill stream, not B2-inflight.
            res["n_empty_from_admission"] = res.get("n_empty_from_admission", 0) + 1
            continue
        last_sync = inside[-1]["t_sync"] if inside else sp["t0"]
        gap = (sp["t1"] - last_sync) * 1000.0
        issued = sum(1 for x in inside if x["prefill_in_flight"] and not x["prefill_kernel_done_wait"])
        kdw = bool(inside[-1]["prefill_kernel_done_wait"]) if inside else False
        res["b2"].append(dict(span=sp["seq"], ext=sp["ext"], n_chunks=n_chunks(sp["ext"]), iters_inside=len(inside),
                              chunks_issued_before_empty=issued, all_issued=kdw, gap_ms=gap,
                              span_ms=(sp["t1"] - sp["t0"]) * 1000.0))
    # split transitions
    for t in tr:
        owner = next((sp for sp in spans if sp["t0"] <= t <= sp["t1"]), None)
        if owner is None:
            res["tr_outside"] += 1
            continue
        i0 = bisect.bisect_right(tl, owner["t0"])
        first = it[i0]["t_launch"] if i0 < len(it) else None
        if first is None or t < first:
            res["tr_admission"] += 1
        else:
            res["tr_midspan"] += 1
    res["drain_events"] = dr
    return res


def drain_table(dr):
    rows = {}
    for d in dr:
        if d["reason"] == "slo_switch":
            cat = "slo_switch"
        elif not d["split_present"]:
            cat = "adjust_merge(no split)"
        elif d["running_bs"] == 0 and (d["split_index"] or d["prefill_kernel_done_wait"] or d["prefill_chunks_unresolved"]):
            cat = "adjust_decode_empty_INFLIGHT"
        elif d["split_index"] == 0 and not d["prefill_kernel_done_wait"]:
            cat = "adjust_admission(split_index 0)"
        else:
            cat = "adjust_other"
        r = rows.setdefault(cat, dict(prefill_sync_ms=[], decode_sync_ms=[], unresolved=[], split_index=[], kdw=0))
        r["prefill_sync_ms"].append(d["prefill_sync_ms"]); r["decode_sync_ms"].append(d["decode_sync_ms"])
        r["unresolved"].append(d["prefill_chunks_unresolved"]); r["split_index"].append(d["split_index"])
        r["kdw"] += int(bool(d["prefill_kernel_done_wait"]))
    return {k: dict(n=len(v["prefill_sync_ms"]), prefill_sync_ms=dist(v["prefill_sync_ms"]),
                    decode_sync_ms=dist(v["decode_sync_ms"]), chunks_unresolved=sorted(v["unresolved"]),
                    split_index=sorted(x if x is not None else -1 for x in v["split_index"]), kdw=v["kdw"])
            for k, v in rows.items()}


def main():
    groups = {
        "GATE_71f5006(drain truth)": sorted(glob.glob(str(RES / "engine_gate/pdmux-policy-gate-smoke-71f5006-*/*/tel.jsonl"))),
        "SMOKE2_71f5006": sorted(glob.glob(str(RES / "cap_char/policy_runs/pdmux-policy-gate-smoke-71f5006-*/smoke/*/tel.jsonl"))),
        "N3_38ed3f4(job3)": sorted(glob.glob(str(RES / "n3_capacity/n3b_*38ed3f4*/tel_*.jsonl"))),
        "N3_93c3cca(job4)": sorted(glob.glob(str(RES / "n3_capacity/n3b_*93c3cca*/tel_*.jsonl"))),
        "N3_79999aa": sorted(glob.glob(str(RES / "n3_capacity/n3b_*79999aa*/tel_*.jsonl"))),
        "N2EA_7e09cc3": sorted(glob.glob(str(RES / "n2_static_grid/n2p_*7e09cc3*/tel_*.jsonl"))),
        "N2PROF_a5de2d6": sorted(glob.glob(str(RES / "n2_static_grid/n2p_*a5de2d6*/tel_*.jsonl"))),
    }
    out = {"params": dict(N_LAYERS=N_LAYERS, BUDGET=BUDGET), "groups": {}}
    for g, files in groups.items():
        per = [analyse(f) for f in files]
        b2 = [b for r in per for b in r["b2"]]
        allspans = sum(r["n_spans"] for r in per)
        gs = dict(n_files=len(per), n_spans=allspans, n_spans_r0_zero=sum(r["n_spans_r0_zero"] for r in per),
                  n_empty_from_admission=sum(r.get("n_empty_from_admission", 0) for r in per),
                  n_b2_inflight=len(b2), frac_b2_of_spans=(len(b2) / allspans if allspans else None),
                  b2_all_chunks_issued=sum(1 for b in b2 if b["all_issued"]),
                  b2_gap_ms=dist([b["gap_ms"] for b in b2]),
                  b2_gap_over_span=dist([b["gap_ms"] / b["span_ms"] for b in b2 if b["span_ms"] > 0]),
                  b2_chunks_issued=dist([b["chunks_issued_before_empty"] for b in b2]),
                  b2_n_chunks=dist([b["n_chunks"] for b in b2]),
                  idx_change_inflight=dict(n=sum(len(r.get("idx_change_inflight", [])) for r in per),
                                           gap_ms=dist([x["gap_ms"] for r in per for x in r.get("idx_change_inflight", [])]),
                                           rows=[x for r in per for x in r.get("idx_change_inflight", [])][:20]),
                  split_transition=dict(n=sum(r["n_split_transition"] for r in per),
                                        admission=sum(r["tr_admission"] for r in per),
                                        midspan=sum(r["tr_midspan"] for r in per),
                                        outside_span=sum(r["tr_outside"] for r in per)),
                  per_file=[dict(file=r["file"], n_spans=r["n_spans"], n_b2=len(r["b2"]),
                                 b2_gap_ms_sum=round(sum(b["gap_ms"] for b in r["b2"]), 1),
                                 b2=[{k: (round(v, 2) if isinstance(v, float) else v) for k, v in b.items()} for b in r["b2"]])
                            for r in per])
        dr = [d for r in per for d in r["drain_events"]]
        if dr:
            gs["drain_truth"] = drain_table(dr)
            gs["drain_truth_per_file"] = {r["file"]: drain_table(r["drain_events"]) for r in per if r["drain_events"]}
        out["groups"][g] = gs
        print(f"== {g}: files={gs['n_files']} spans={allspans} r0=0:{gs['n_spans_r0_zero']} "
              f"emptyAdm={gs['n_empty_from_admission']} B2-inflight={len(b2)} ({gs['frac_b2_of_spans']}) all_issued={gs['b2_all_chunks_issued']} "
              f"gap_ms={gs['b2_gap_ms']} gap/span={gs['b2_gap_over_span']} transitions={gs['split_transition']} "
              f"idx_change_inflight={gs['idx_change_inflight']['n']} {gs['idx_change_inflight']['gap_ms']} "
              f"b2_chunks_issued={gs['b2_chunks_issued']} b2_n_chunks={gs['b2_n_chunks']}")
        if dr:
            for k, v in gs["drain_truth"].items():
                print("   drain", k, v)
    Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "c2_vessl_adjust_inflight.json").write_text(
        json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
