#!/usr/bin/env python3
"""C1: HE0 (KISTI, Zamba2-2.7B, sharegpt_vary_bench rate 3<->12) -- controller switches vs in-flight prefill.

Data actually available (no PDMUX_TELEMETRY_PATH in sharegpt_vary_bench.sbatch; migration bundle has only client jsonl):
  server log  sgptvsrv_<tag>.log : 'Prefill batch' (one per admitted split-prefill batch, 1 s timestamps, #new-token,
                                   #running-req), 'Decode batch' (every 40 decode iterations, gen throughput),
                                   'SLO-BIND a->b' / 'SLO-SCHED a->b' (controller decision change), 'SLO-CTLCOST n=' (every 200 evals).
Code facts used (HE0 tree fdbf5d1 event_loop_pdmux; same structure in 3858f129):
  * controller evaluated only when running non-empty AND split_prefill_batch AND not wait_prefill_kernel_done,
    i.e. on the admission iteration (0 chunks issued) and on iterations 1..n_chunks-1 of a span.
  * n_chunks = ceil(L / max(1, budget // extend_tokens)), budget 65536 (pdmux_*.yml), L = 54 (Zamba2-2.7B).
  * a switch on the admission iteration drains an EMPTY prefill stream (previous span merged only after its
    exe_done event query() returned True); only a switch on iteration k>=1 of a multi-chunk span can wait on
    in-flight prefill chunks.
Scheduler log lines are written by one process, so their file order == execution order. A switch line belongs to
the most recent preceding 'Prefill batch' line (the span in progress).
Decode-gap proxy: window step = #running-req / gen_throughput (s per decode iteration) of the first 'Decode batch'
line after the switch, minus median of 3 windows before + 3 after; excess_ms = 40 * diff. Same statistic computed for
control windows (first Decode line after a 'Prefill batch' line with no switch in between).
"""
import json, math, re, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent
SRC = RES / "slo_sched"
REAL = RES / "he0_contamination_2026-10-01" / "he0_realized_2026-10-01.json"
L_LAYERS = 54          # Zamba2-2.7B num_hidden_layers [cited]
BUDGET = 65536         # split_forward_token_budget in pdmux_{d16..d44,slo}.yml [code]
LOG_INTERVAL = 40      # sglang decode_log_interval default [code]

TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (.*)$")
PB = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+),.*#running-req: (\d+), #queue-req: (\d+)")
DB = re.compile(r"Decode batch, #running-req: (\d+),.*gen throughput \(token/s\): ([\d.]+)")
SW = re.compile(r"SLO-(BIND|SCHED) (\d+)->(\d+)")
CC = re.compile(r"SLO-CTLCOST n=(\d+)")


def n_chunks(tok):
    fc = max(1, BUDGET // tok) if tok > 0 else L_LAYERS
    return math.ceil(L_LAYERS / min(fc, L_LAYERS))


def parse(path):
    ev = []
    for ln in open(path, errors="replace"):
        m = TS.match(ln.rstrip("\n"))
        if not m:
            continue
        body = m.group(2)
        if (p := PB.search(body)):
            ev.append(("P", int(p.group(2)), int(p.group(3)), int(p.group(1))))
        elif (d := DB.search(body)):
            ev.append(("D", int(d.group(1)), float(d.group(2))))
        elif (s := SW.search(body)):
            ev.append(("S", int(s.group(2)), int(s.group(3)), s.group(1)))
        elif (c := CC.search(body)):
            ev.append(("C", int(c.group(1))))
    return ev


def window_excess(ev, i):
    """excess ms of the first Decode window at/after index i vs neighbour windows."""
    didx = [k for k, e in enumerate(ev) if e[0] == "D"]
    nxt = [k for k in didx if k > i]
    if not nxt:
        return None
    k0 = nxt[0]
    pos = didx.index(k0)
    def step(k):
        e = ev[k]
        return e[1] / e[2] if e[2] > 0 and e[1] > 0 else None
    s0 = step(k0)
    nb = [step(didx[j]) for j in list(range(pos - 3, pos)) + list(range(pos + 1, pos + 4)) if 0 <= j < len(didx)]
    nb = [x for x in nb if x is not None]
    if s0 is None or len(nb) < 4:
        return None
    return LOG_INTERVAL * (s0 - statistics.median(nb)) * 1000.0


def q(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def b1_bound(runs):
    """Upper bound of in-flight drain per run and the HI-phase requests it could push over the TTFT SLO.
    Drain at iteration k of an n-chunk span waits <= k/n * W <= (n-1)/n * W; W <= max(admission->first-token) of
    that arm (he0_realized q4, both phases) [derived]. A stall s can raise any later TTFT by <= s, so the requests
    whose TTFT lies in (SLO - s_run, SLO] bound the TTFT-side goodput loss [derived, conservative]. ITL side:
    each stall adds <= 1 gap to <= max_running(48) requests."""
    sys.path.insert(0, str(RES.parent / "benchmarks"))
    from pdmux_eval import analyze as A
    real = json.load(open(REAL))
    slo_ttft = float(real["slo"]["ttft_ms"])
    rows = []
    for rr in runs:
        multi = [sw for sw in rr["switches"] if sw["cls"].startswith("multi")]
        if rr["arm"] not in ("bind_gate", "bind_nogate", "slo"):
            continue
        q4 = real["q4"][rr["arm"]]
        wmax = max(q4["LO"]["exec_mean_ms_prefill_cores"]["max"], q4["HI"]["exec_mean_ms_prefill_cores"]["max"])
        s_run = sum((sw["n_chunks"] - 1) / sw["n_chunks"] * wmax for sw in multi)
        reqs, dur = A.load_bench_serving_rounds(SRC / f"sgptvHi_{rr['tag']}.jsonl")
        tt = [r.ttft_ms for r in reqs if r.ttft_ms is not None]
        near = sum(1 for t in tt if slo_ttft - s_run < t <= slo_ttft) if s_run > 0 else 0
        rows.append(dict(arm=rr["arm"], job=rr["job"], n_multi_switch=len(multi), w_max_ms=round(wmax, 1),
                         stall_bound_s=round(s_run / 1000.0, 3), hi_requests=len(tt),
                         hi_req_ttft_within_bound_of_slo=near,
                         frac=round(near / len(tt), 4) if tt else None,
                         itl_req_touched_bound=48 * len(multi)))
    return rows


def main():
    runs = json.load(open(REAL))["runs"]
    out = {"params": dict(L=L_LAYERS, budget=BUDGET, chunk_threshold_tokens=math.floor(BUDGET / L_LAYERS) + 1),
           "runs": []}
    for r in runs:
        tag, arm = r["tag"], r["arm"]
        ev = parse(SRC / f"sgptvsrv_{tag}.log")
        P = [e for e in ev if e[0] == "P"]
        nch = [n_chunks(e[1]) for e in P]
        pred_evals = sum(n for e, n in zip(P, nch) if e[2] > 0)
        ctl = [e[1] for e in ev if e[0] == "C"]
        sw_rows, ctrl_excess_multi, ctrl_excess_single = [], [], []
        cur = None
        for i, e in enumerate(ev):
            if e[0] == "P":
                cur = (i, e)
                # control: no switch before next Decode line
                nxt_d = next((k for k in range(i + 1, len(ev)) if ev[k][0] in "DS"), None)
                if nxt_d is not None and ev[nxt_d][0] == "D":
                    x = window_excess(ev, i)
                    if x is not None:
                        (ctrl_excess_multi if n_chunks(e[1]) >= 2 else ctrl_excess_single).append(x)
            elif e[0] == "S":
                if cur is None:
                    sw_rows.append(dict(kind=e[3], frm=e[1], to=e[2], span_tokens=None, n_chunks=None,
                                        running=None, cls="no_span_before", excess_ms=window_excess(ev, i)))
                    continue
                pi, pe = cur
                nc = n_chunks(pe[1])
                d_between = sum(1 for k in range(pi + 1, i) if ev[k][0] == "D")
                cls = "admission_iter_empty_prefill_stream" if nc == 1 else "multi_chunk_possible_inflight"
                sw_rows.append(dict(kind=e[3], frm=e[1], to=e[2], span_tokens=pe[1], n_chunks=nc, running=pe[2],
                                    decode_lines_between=d_between, cls=cls, excess_ms=window_excess(ev, i)))
        rr = dict(arm=arm, job=r["job"], tag=tag, n_prefill_batches=len(P),
                  n_batches_running0=sum(1 for e in P if e[2] == 0),
                  frac_batches_multichunk=(sum(1 for n in nch if n >= 2) / len(nch)) if nch else None,
                  max_chunks=max(nch) if nch else None,
                  pred_controller_evals=pred_evals, ctlcost_last_n=(ctl[-1] if ctl else None),
                  n_switch=len(sw_rows),
                  n_switch_admission_only=sum(1 for s in sw_rows if s["cls"].startswith("admission")),
                  n_switch_multichunk=sum(1 for s in sw_rows if s["cls"].startswith("multi")),
                  switches=sw_rows,
                  ctrl_excess_ms_single=dict(n=len(ctrl_excess_single), p50=q(ctrl_excess_single, .5),
                                             p95=q(ctrl_excess_single, .95)),
                  ctrl_excess_ms_multi=dict(n=len(ctrl_excess_multi), p50=q(ctrl_excess_multi, .5),
                                            p95=q(ctrl_excess_multi, .95)),
                  ctrl_multi_raw=sorted(round(x, 1) for x in ctrl_excess_multi),
                  switch_window_rank_in_multi_ctrl=[
                      round(sum(1 for c in ctrl_excess_multi if c < sw["excess_ms"]) / len(ctrl_excess_multi), 3)
                      for sw in sw_rows if sw["excess_ms"] is not None and sw["cls"].startswith("multi")
                      and ctrl_excess_multi],
                  prefill_only_s_rounds=[round(x["frac"]["pt"]["prefill_only_s"], 4) for x in r["rounds"]],
                  dur_s_rounds=[round(x["frac"]["pt"]["W_s"], 2) for x in r["rounds"]])
        out["runs"].append(rr)
    # aggregate by arm
    agg = {}
    for rr in out["runs"]:
        a = agg.setdefault(rr["arm"], dict(n_runs=0, switches=[], n_sw=[], n_multi=0, n_adm=0, fm=[], ex_sw=[],
                                           ex_c1=[], ex_cm=[], evals=[], pfo=[]))
        a["n_runs"] += 1
        a["n_sw"].append(rr["n_switch"]); a["n_multi"] += rr["n_switch_multichunk"]; a["n_adm"] += rr["n_switch_admission_only"]
        a["fm"].append(rr["frac_batches_multichunk"])
        a["ex_sw"] += [s["excess_ms"] for s in rr["switches"] if s["excess_ms"] is not None and s["cls"].startswith("multi")]
        a["evals"].append((rr["pred_controller_evals"], rr["ctlcost_last_n"]))
        a["pfo"].append(sum(rr["prefill_only_s_rounds"]))
        a.setdefault("ranks", []).extend(rr["switch_window_rank_in_multi_ctrl"])
        a.setdefault("r0", []).append(rr["n_batches_running0"])
        a.setdefault("nb", []).append(rr["n_prefill_batches"])
    summ = {}
    for arm, a in agg.items():
        summ[arm] = dict(n_runs=a["n_runs"], switches_per_run=a["n_sw"],
                         switches_mean=statistics.fmean(a["n_sw"]),
                         switches_sd=statistics.stdev(a["n_sw"]) if len(a["n_sw"]) > 1 else 0.0,
                         n_switch_admission_only=a["n_adm"], n_switch_multichunk=a["n_multi"],
                         frac_batches_multichunk_range=[min(a["fm"]), max(a["fm"])],
                         excess_ms_multichunk_switch_windows=sorted(round(x, 1) for x in a["ex_sw"]),
                         pred_evals_vs_ctlcost=a["evals"],
                         switch_window_rank_in_same_run_multichunk_ctrl=sorted(a["ranks"]),
                         n_batches_running0_per_run=a["r0"], n_batches_per_run=a["nb"],
                         prefill_only_s_sum_per_run_range=[round(min(a["pfo"]), 3), round(max(a["pfo"]), 3)])
    out["summary_by_arm"] = summ
    out["b1_bound"] = b1_bound(out["runs"])
    for row in out["b1_bound"]:
        print("B1 bound", row)
    Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "c1_he0_switch_inflight.json").write_text(
        json.dumps(out, indent=1, default=str))
    for arm, s in summ.items():
        print(arm, {k: v for k, v in s.items() if k not in ("pred_evals_vs_ctlcost",)})
        print("   evals pred vs ctlcost:", s["pred_evals_vs_ctlcost"])


if __name__ == "__main__":
    main()
