"""Green-context allocation probe: is `decode_sm` honored, or just the remainder?

Every PD-mux division used so far sums to exactly 108
(`[92,16]`, `[64,44]`, `[16,92]`), so the existing runs cannot distinguish

  (H_honored)   create_greenctx_stream_by_value(a, b) gives stream B exactly b SMs
                and leaves 108-a-b SMs idle, from
  (H_remainder) stream B gets whatever A did not take (108-a), so b is ignored.

The distinction decides whether the Stage 0 de-confound (hold prefill SM fixed,
sweep decode SM, remainder idle) is expressible in config alone or needs an
engine patch.

Method: green-context streams have no public SM-count getter, so measure it.
Run a compute-bound bf16 matmul loop on each stream and read the achieved
throughput; matmul scales close to linearly in SMs, so effective_sm is
recovered by scaling against a full-device (108 SM) baseline.

Discriminator, reading effective SMs of stream B:
  (16,44) vs (64,44)  -> equal   => b honored          (sum<108 usable)
  (16,44) vs (16,92)  -> equal   => b is the remainder (sum<108 NOT usable)
"""

import json
import os
import sys

import torch
from sgl_kernel import spatial

DEV = 0
# Big enough to be solidly compute-bound, small enough that a 16-SM partition
# still finishes quickly.
N = 4096
ITERS = 30
WARMUP = 10
FLOP_PER_MM = 2.0 * N * N * N


def bench(stream, tag):
    """Achieved TFLOP/s for a bf16 matmul loop issued on `stream`."""
    a = torch.randn(N, N, device=f"cuda:{DEV}", dtype=torch.bfloat16)
    b = torch.randn(N, N, device=f"cuda:{DEV}", dtype=torch.bfloat16)
    ctx = torch.cuda.stream(stream) if stream is not None else torch.cuda.stream(
        torch.cuda.current_stream(DEV)
    )
    with ctx:
        for _ in range(WARMUP):
            a @ b
        torch.cuda.synchronize(DEV)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream) if stream is not None else start.record()
        for _ in range(ITERS):
            a @ b
        end.record(stream) if stream is not None else end.record()
        torch.cuda.synchronize(DEV)
    ms = start.elapsed_time(end)
    tflops = (FLOP_PER_MM * ITERS) / (ms * 1e-3) / 1e12
    del a, b
    torch.cuda.empty_cache()
    return {"tag": tag, "ms": round(ms, 3), "tflops": round(tflops, 2)}


def main():
    total_sm = spatial.get_sm_available(DEV)
    out = {"total_sm_available": total_sm, "n": N, "iters": ITERS, "pairs": []}

    # Full-device baseline: converts TFLOP/s into an effective SM count.
    base = bench(None, "baseline_default_stream")
    out["baseline"] = base
    print(f"baseline (default stream, {total_sm} SM): {base['tflops']} TFLOP/s",
          flush=True)

    # (prefill_sm, decode_sm) exactly as manual_divisions feeds them.
    pairs = [
        (92, 16),   # existing d16   config, sums to 108
        (64, 44),   # existing d44   config, sums to 108
        (16, 92),   # existing d92   config, sums to 108
        (16, 44),   # PROBE: sums to 60  -> honored? or B gets 92?
        (16, 16),   # PROBE: sums to 32  -> honored? or B gets 92?
        (44, 16),   # PROBE: sums to 60  -> honored? or B gets 64?
    ]

    for (a_sm, b_sm) in pairs:
        rec = {"requested": [a_sm, b_sm], "sum": a_sm + b_sm}
        try:
            sa, sb = spatial.create_greenctx_stream_by_value(a_sm, b_sm, DEV)
        except Exception as exc:  # creation itself may reject sum<108
            rec["error"] = f"{type(exc).__name__}: {exc}"
            out["pairs"].append(rec)
            print(f"({a_sm},{b_sm}) CREATE FAILED: {rec['error']}", flush=True)
            continue

        # Benchmark each partition alone; the other stream stays idle so the
        # measurement reflects that partition's own SM allotment.
        ra = bench(sa, f"A_{a_sm}")
        rb = bench(sb, f"B_{b_sm}")
        for side, r, req in (("A", ra, a_sm), ("B", rb, b_sm)):
            eff = total_sm * r["tflops"] / base["tflops"]
            rec[side] = {
                "requested_sm": req,
                "tflops": r["tflops"],
                "effective_sm_est": round(eff, 1),
            }
        out["pairs"].append(rec)
        print(
            f"({a_sm},{b_sm}) sum={a_sm + b_sm:3d} | "
            f"A req {a_sm:3d} -> eff {rec['A']['effective_sm_est']:6.1f} | "
            f"B req {b_sm:3d} -> eff {rec['B']['effective_sm_est']:6.1f}",
            flush=True,
        )
        del sa, sb

    dest = sys.argv[1] if len(sys.argv) > 1 else "greenctx_alloc_probe.json"
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)

    # Verdict on the discriminator.
    def eff_b(pair):
        for p in out["pairs"]:
            if p["requested"] == list(pair) and "B" in p:
                return p["B"]["effective_sm_est"]
        return None

    b_16_44, b_64_44, b_16_92 = eff_b((16, 44)), eff_b((64, 44)), eff_b((16, 92))
    print("\n=== VERDICT ===", flush=True)
    if b_16_44 is None:
        print("SUM_LT_108_REJECTED: green ctx refused a partition pair summing "
              "to <108 -> fixed-prefill de-confound needs an engine change.",
              flush=True)
    elif b_64_44 and abs(b_16_44 - b_64_44) < 0.15 * b_64_44:
        print(f"DECODE_SM_HONORED: B(16,44) eff={b_16_44} ~= B(64,44) "
              f"eff={b_64_44} -> sum<108 leaves SMs idle; the fixed-prefill "
              f"de-confound is CONFIG-ONLY.", flush=True)
    elif b_16_92 and abs(b_16_44 - b_16_92) < 0.15 * b_16_92:
        print(f"DECODE_SM_IS_REMAINDER: B(16,44) eff={b_16_44} ~= B(16,92) "
              f"eff={b_16_92} -> requested decode_sm ignored; de-confound "
              f"needs an engine change.", flush=True)
    else:
        print(f"INCONCLUSIVE: B(16,44)={b_16_44} B(64,44)={b_64_44} "
              f"B(16,92)={b_16_92}", flush=True)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
