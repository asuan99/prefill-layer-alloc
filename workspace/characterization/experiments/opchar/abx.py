"""abx.py — audit C3 decider: 104 green vs 106 green vs 108 plain, interleaved in ONE process.

Claim under test (AUDIT_opchar_2026-10-13.md C3): the decode GEMM drop between 104-green and
108-plain is wave quantization of a 105-CTA split-K GEMM (b=1), not a green-vs-plain effect.
Prediction registered before running: on a 106-SM green partition the 105-CTA GEMMs fit in ONE
wave, so their time ≈ the 108-plain value (o_proj ~0.045 ms); staying at the 104-green value
(~0.066 ms) refutes the claim. 1-wave controls (in_proj grid 89, up_proj grid 62) should match
across all three arms. Prefill L=1024 up_proj (grid 496) / in_proj (grid 712) are recorded too.

Design: all three partitions exist simultaneously (green pairs (104,4) and (106,2) + one plain
non-default stream); each arm captures its OWN CUDA graph on its own stream (decode) or runs eager
(prefill); R rounds, arm order shuffled per round with a fixed seed; n_per_round timed calls per
(arm, op) per round. A graph census per arm records the launch actually captured there.
"""
from __future__ import annotations

import random
import statistics
from typing import Dict, List

ARMS = (("green104", 104), ("green106", 106), ("plain108", 108))
DECODE_OPS = ("*:o_proj", "M:out_proj", "-:down_proj", "M:in_proj", "-:up_proj")
PREFILL_OPS = ("-:up_proj", "M:in_proj", "M:out_proj")


def open_arms(device_sm: int = 108):
    import torch
    from sgl_kernel import spatial
    from experiments.opchar.worker import _green_readout, realized_sm
    dev = torch.cuda.current_device()
    arms, keep = {}, []
    for name, n in ARMS:
        if n >= device_sm:
            arms[name] = {"stream": torch.cuda.Stream(), "n_req": n, "n_realized": device_sm, "mode": "plain"}
            continue
        sa, sb = spatial.create_greenctx_stream_by_value(n, device_sm - n, dev)
        keep.append(sb)
        rd = _green_readout().green_sm_readout(int(sa.cuda_stream))
        arms[name] = {"stream": sa, "n_req": n, "n_realized": realized_sm(rd), "mode": "green"}
    return arms, keep


def summarize(samples: Dict[tuple, List[float]]) -> List[dict]:
    """(arm, phase, op) -> pooled median / p10 / p90 over all rounds (CPU, unit-tested)."""
    out = []
    for (arm, phase, op), v in sorted(samples.items()):
        s = sorted(v)
        q = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]
        out.append({"arm": arm, "phase": phase, "op": op, "n": len(s), "ms_median": statistics.median(s),
                    "ms_p10": q(0.10), "ms_p90": q(0.90)})
    return out


def run(cfg, rounds: int = 10, n_per_round: int = 10, inner: int = 10, seed: int = 20261013,
        decode_B: int = 128, ctx: int = 2048, prefill_L: int = 1024):
    import torch
    from experiments.opchar import builders as Bld
    from experiments.opchar import graph_census as GC
    arms, keep = open_arms()
    dec, _ = Bld.build(cfg, "decode", decode_B, ctx=ctx)
    pre, _ = Bld.build(cfg, "prefill", prefill_L)
    dec = {s.key: f for s, f in dec if s.key in DECODE_OPS}
    pre = {s.key: f for s, f in pre if s.key in PREFILL_OPS}
    graphs, census = {}, []
    for name, a in arms.items():
        st = a["stream"]
        for op, fn in dec.items():
            with torch.cuda.stream(st):
                fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=st):
                for _ in range(inner):
                    fn()
            graphs[(name, op)] = g
            for r in GC.census_rows(GC.capture_census(fn, st), int(a["n_realized"] or a["n_req"])):
                census.append({"arm": name, "phase": "decode", "op": op, **r})
        for op, fn in pre.items():
            for r in GC.census_rows(GC.capture_census(fn, st), int(a["n_realized"] or a["n_req"])):
                census.append({"arm": name, "phase": "prefill", "op": op, **r})
    torch.cuda.synchronize()
    rng = random.Random(seed)
    samples: Dict[tuple, List[float]] = {}
    order_log = []
    for rnd in range(rounds):
        order = list(arms)
        rng.shuffle(order)
        order_log.append(order)
        for name in order:
            st = arms[name]["stream"]
            for op in dec:
                samples.setdefault((name, "decode", op), []).extend(
                    _time(lambda: graphs[(name, op)].replay(), st, n_per_round, inner))
            for op, fn in pre.items():
                samples.setdefault((name, "prefill", op), []).extend(_time(fn, st, n_per_round, 1))
    meta = {"arms": {k: {kk: vv for kk, vv in v.items() if kk != "stream"} for k, v in arms.items()},
            "rounds": rounds, "n_per_round": n_per_round, "inner": inner, "seed": seed,
            "order": order_log, "decode_B": decode_B, "ctx": ctx, "prefill_L": prefill_L}
    return summarize(samples), census, meta


def _time(fn, stream, n, inner):
    import torch
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            e0.record(stream)
            fn()
            e1.record(stream)
        torch.cuda.synchronize()
        out.append(e0.elapsed_time(e1) / inner)
    return out
