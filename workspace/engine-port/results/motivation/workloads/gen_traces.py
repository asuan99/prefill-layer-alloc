#!/usr/bin/env python3
"""Motivation trace generator (CPU only).  Registered seeds, rate-free masters, paired interference traces.

Master stream of (workload, seed): N unit-rate Poisson arrival times + a pool permutation order.  A rung (rate, duration) is the
prefix of the master with t / rate < duration, so every rung sees the same request sequence (common random numbers; seed key does not
hold the rate).  Master hashes are registered in trace_registry.json before any rate is chosen; a rung hash is reproducible from
(workload, seed, rate, duration) and is recorded once the rate grid is fixed.  Records follow ph_trace (record/arrival_s/input_tokens/
output_tokens/ignore_eos); doc_id is the token-file record id (absent for sharegpt_first, whose prompt is the [1]*n convention).

Interference cell (paired), all from one seed:
  short stream = the sharegpt_first master prefix at the fixed short rate (byte-identical records with and without the long stream)
  long stream  = n_long LooGLE records, n_long = round(share/(1-share) * short_rate * duration) requests, share = 0.05
                 arrival times are Poisson conditioned on the count: t_k = duration * S_k / S_{n_long+1}, S = unit exponential sums
                 (fixes the count; avoids the ~14% count noise of 48 expected long arrivals in 300 s, workload_comparison.html)
  with_long / without_long traces differ only by the long records; the short-stream hash is identical (checked below).

  python3 -I gen_traces.py seal                           write trace_registry.json (needs lengths/*.json)
  python3 -I gen_traces.py check                          recompute and compare with the registry
  python3 -I gen_traces.py emit --workload W --seed S --rate R --duration T --out F
  python3 -I gen_traces.py emit-interference --seed S --short-rate R --duration T --with-long 0|1 --out F
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mw_lib as L  # noqa: E402
import ph_trace as PH  # noqa: E402

MASTER_N = {"sharegpt_first": 20000, "openchat_rounds": 20000, "arxiv_summ": 5000, "loogle_longdep": 2000}
ARXIV_MAX_NEW = 1024                                # natural-output cap, decision 47 convention
LONG_SHARE = 0.05                                   # long requests / all requests of the interference cell
PROVISIONAL_SHORT_RATES = (2.0, 3.0, 4.0)           # registered ONLY as examples; the rate grid is fixed after the pre-ladder
INTERFERENCE_DURATION = 300.0
SCHEMA = "pdmux-motivation-traces/v1"


def master(workload, seed, pool):
    """(unit_times, pool_indices) of length MASTER_N[workload]; the seed key excludes the rate."""
    n = MASTER_N[workload]
    key = ("mw-master", workload, seed)
    r = PH._rng("arr", *key)
    t, times = 0.0, []
    for _ in range(n):
        t += r.expovariate(1.0)
        times.append(t)
    r = PH._rng("pick", *key)
    idx = []
    while len(idx) < n:                              # permutation passes: no repeats within a pass, prefix-stable in n
        p = list(range(len(pool)))
        r.shuffle(p)
        idx += p
    return times, idx[:n]


def request(workload, row, arrival_s, k, prefix, phase):
    rid, ni, no = row
    rec = {"record": "request", "request_id": f"{prefix}-{k:05d}", "phase": phase, "arrival_s": round(arrival_s, 6),
           "input_tokens": ni, "output_tokens": no, "ignore_eos": True}
    if workload == "arxiv_summ":
        rec.update(output_tokens=ARXIV_MAX_NEW, ignore_eos=False)
    if rid is not None:
        rec["doc_id"] = rid
    return rec


def master_records(workload, seed, pool):
    times, idx = master(workload, seed, pool)
    return [request(workload, pool[i], t, k, "m", workload) for k, (t, i) in enumerate(zip(times, idx))]


def segment(tag, phase, requests, **extra):
    return {"segment_id": tag, "phase": phase, "round": 0, "requests": requests, **extra}


def instantiate(workload, seed, pool, rate, duration=None, n=None):
    """One rung: scale the master times by 1/rate and keep the prefix inside the window (or the first n requests)."""
    if (duration is None) == (n is None) or rate <= 0:
        raise ValueError("rate > 0 and exactly one of duration / n")
    times, idx = master(workload, seed, pool)
    if n is not None and n > len(times):
        raise ValueError("n exceeds the registered master")
    if duration is not None and times[-1] / rate < duration:
        raise ValueError("duration exceeds the registered master at this rate")
    reqs, prefix = [], f"{workload}-s{seed}"
    for k, (t, i) in enumerate(zip(times, idx)):
        at = t / rate
        if (n is not None and k >= n) or (duration is not None and at >= duration):
            break
        reqs.append(request(workload, pool[i], at, k, prefix, workload))
    return [segment(f"{workload}-s{seed}-r{rate}", workload, reqs, rate=rate, seed=seed)]


def long_stream(seed, pool, n_long, duration):
    key = ("mw-master", "loogle_long_interference", seed)
    r = PH._rng("arr", *key)
    s = []
    t = 0.0
    for _ in range(n_long + 1):
        t += r.expovariate(1.0)
        s.append(t)
    r = PH._rng("pick", *key)
    idx = list(range(len(pool)))
    r.shuffle(idx)
    if n_long > len(idx):
        raise ValueError("n_long exceeds the LooGLE pool")
    return [(duration * s[k] / s[n_long], idx[k]) for k in range(n_long)]


def interference(seed, short_pool, long_pool, short_rate, duration, with_long):
    """-> (segments, short_records).  short_records carry class=short and are identical for with_long = 0 and 1."""
    if not 0 < LONG_SHARE < 1:
        raise ValueError("share")
    short = instantiate("sharegpt_first", seed, short_pool, short_rate, duration=duration)[0]["requests"]
    for r in short:
        r["request_class"] = "short"
    reqs = list(short)
    n_long = int(round(LONG_SHARE / (1 - LONG_SHARE) * short_rate * duration))
    if with_long:
        for k, (t, i) in enumerate(long_stream(seed, long_pool, n_long, duration)):
            r = request("loogle_longdep", long_pool[i], t, k, f"long-s{seed}", "LONG")
            r["request_class"] = "long"
            reqs.append(r)
    reqs.sort(key=lambda r: (r["arrival_s"], r["request_id"]))
    tag = f"interference-s{seed}-r{short_rate}-{'with' if with_long else 'without'}"
    seg = segment(tag, "INTERFERENCE", reqs, seed=seed, short_rate=short_rate, duration_s=duration, long_share=LONG_SHARE,
                  n_long=n_long if with_long else 0)
    return [seg], short


def short_hash(records):
    return PH.trace_hash([segment("x", "short", [{k: v for k, v in r.items() if k != "request_class"} for r in records])])


def load_pools():
    return {w: L.load_pool(w) for w in L.WORKLOADS}


def seal_data():
    pools = load_pools()
    masters = {}
    for w in L.WORKLOADS:
        masters[w] = {}
        for s in L.SEEDS + L.SPARE_SEEDS:
            masters[w][str(s)] = PH.trace_hash([segment("m", w, master_records(w, s, pools[w]))])
    pairs = {}
    for s in L.SEEDS:
        for rate in PROVISIONAL_SHORT_RATES:
            with_seg, sa = interference(s, pools["sharegpt_first"], pools["loogle_longdep"], rate, INTERFERENCE_DURATION, 1)
            wo_seg, sb = interference(s, pools["sharegpt_first"], pools["loogle_longdep"], rate, INTERFERENCE_DURATION, 0)
            if short_hash(sa) != short_hash(sb):
                raise SystemExit("pairing broken: short stream differs")
            pairs[f"{s}|{rate}"] = {"with_long": PH.trace_hash(with_seg), "without_long": PH.trace_hash(wo_seg),
                                    "short_stream": short_hash(sa), "n_short": len(sa), "n_long": with_seg[0]["n_long"]}
    return {"schema": SCHEMA, "seeds": list(L.SEEDS), "spare_seeds": list(L.SPARE_SEEDS), "master_n": MASTER_N,
            "arrival": "unit-rate Poisson master, rung = t/rate prefix (ph_trace._rng key excludes the rate)",
            "interference": {"long_share": LONG_SHARE, "duration_s": INTERFERENCE_DURATION,
                             "provisional_short_rates": list(PROVISIONAL_SHORT_RATES),
                             "provisional": "example rates only; the rate grid is fixed by the preregistration after the pre-ladder",
                             "long_arrivals": "Poisson conditioned on count: t_k = duration * S_k / S_(n_long+1)"},
            "master_hashes": masters, "provisional_interference_pairs": pairs,
            "generator_sha256": L.sha256_file(HERE / "gen_traces.py"), "lib_sha256": L.sha256_file(HERE / "mw_lib.py")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seal")
    sub.add_parser("check")
    e = sub.add_parser("emit")
    e.add_argument("--workload", required=True, choices=L.WORKLOADS)
    e.add_argument("--seed", type=int, required=True)
    e.add_argument("--rate", type=float, required=True)
    g = e.add_mutually_exclusive_group(required=True)
    g.add_argument("--duration", type=float)
    g.add_argument("--n", type=int)
    e.add_argument("--out", required=True)
    f = sub.add_parser("emit-interference")
    f.add_argument("--seed", type=int, required=True)
    f.add_argument("--short-rate", type=float, required=True)
    f.add_argument("--duration", type=float, default=INTERFERENCE_DURATION)
    f.add_argument("--with-long", type=int, choices=(0, 1), required=True)
    f.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "seal":
        reg = seal_data()
        (HERE / "trace_registry.json").write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
        print("sealed", L.sha256_file(HERE / "trace_registry.json"))
        return 0
    if a.cmd == "check":
        reg = json.load(open(HERE / "trace_registry.json"))
        new = seal_data()
        bad = [k for k in ("master_hashes", "provisional_interference_pairs") if reg[k] != new[k]]
        print("trace_registry check:", "OK" if not bad else f"MISMATCH {bad}")
        return 1 if bad else 0
    pools = load_pools()
    if a.cmd == "emit":
        segs = instantiate(a.workload, a.seed, pools[a.workload], a.rate, duration=a.duration, n=a.n)
    else:
        segs, _ = interference(a.seed, pools["sharegpt_first"], pools["loogle_longdep"], a.short_rate, a.duration, bool(a.with_long))
    h = PH.write_trace(segs, a.out)
    print(json.dumps({"out": a.out, "trace_hash": h, "n_requests": sum(len(s["requests"]) for s in segs)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
