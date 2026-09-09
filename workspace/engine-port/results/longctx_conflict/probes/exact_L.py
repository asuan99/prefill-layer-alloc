#!/usr/bin/env python3
"""Exact client-side phase decomposition from bench_serving --output-details.

Little's law, integral form: the time-average population in a state equals
(total time all requests spent in that state) / (observation window).  Needs
NO arrival timestamps and NO telemetry sampling, so it is immune to the
snapshot-interval imputation artifact (CONSENSUS §3 item 120, gate #103).

  L_decode = sum_i sum(itls_i) / duration      # first token -> last token
  L_pre    = sum_i ttft_i      / duration      # arrival -> first token (queue+prefill)
  L_total  = L_pre + L_decode                  # must equal reported `concurrency`
"""
import json, sys

ITL_SOLO = 0.01303689880296588   # P2 job 905712, ctx 8192
FLOOR_REF = 0.9104067257139832   # P2 job 905712, ctx 8192

def analyze(path, label):
    d = json.loads(open(path).read().strip().split("\n")[0])
    dur = d["duration"]
    ttfts = [t for t, e in zip(d["ttfts"], d["errors"]) if e == ""]
    itls = [x for x, e in zip(d["itls"], d["errors"]) if e == ""]
    sum_dec = sum(sum(v) for v in itls)
    sum_pre = sum(ttfts)
    L_dec, L_pre = sum_dec / dur, sum_pre / dur
    out_len = d["random_output_len"]
    R = FLOOR_REF / (FLOOR_REF + out_len * ITL_SOLO)
    itl_p50 = d["median_itl_ms"] / 1000.0
    mu_solo = 1.0 / FLOOR_REF
    mu_ach = d["request_throughput"]
    s_D = mu_solo / mu_ach
    pred = ((1 - R) / R) * (itl_p50 / ITL_SOLO) / s_D
    print(f"--- {label}")
    print(f"  offered={d['request_rate']}  achieved={mu_ach:.4f} req/s  ach/off={mu_ach/d['request_rate']:.3f}"
          f"  duration={dur:.1f}s  completed={d['completed']}")
    print(f"  TTFT p50={d['median_ttft_ms']/1000:.2f}s (= {d['median_ttft_ms']/1000/FLOOR_REF:.1f}x floor_ref)"
          f"   ITL p50={itl_p50*1000:.2f}ms (= {itl_p50/ITL_SOLO:.2f}x itl_solo)")
    print(f"  L_total(reported concurrency) = {d['concurrency']:.3f}")
    print(f"  L_pre  (queue+prefill)        = {L_pre:.3f}")
    print(f"  L_decode (EXACT, client-side) = {L_dec:.3f}      <-- decode population")
    print(f"  check L_pre+L_dec = {L_pre+L_dec:.3f} vs reported {d['concurrency']:.3f}"
          f"  (rel err {abs(L_pre+L_dec-d['concurrency'])/d['concurrency']:.2e})")
    print(f"  R = {R:.4f}   C_max=(1-R)/R = {(1-R)/R:.3f}   prefill slowdown s(D)={s_D:.3f}")
    print(f"  model pred L_decode = C_max*(itl/itl_solo)/s(D) = {pred:.3f}"
          f"   -> obs/pred = {L_dec/pred:.3f}")

P = "/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/p1_905707"
analyze(f"{P}/bench_d44.jsonl", "P1 d44 (prefill 64 SM)  ctx8192/out96  offered 1.0")
analyze(f"{P}/bench_d92.jsonl", "P1 d92 (prefill 16 SM)  ctx8192/out96  offered 1.0")
