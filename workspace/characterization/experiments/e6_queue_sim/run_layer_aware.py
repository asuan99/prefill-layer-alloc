"""Does LAYER-TYPE-AWARE PD-mux beat layer-AGNOSTIC PD-mux? (the original thesis's live form)

A hybrid forward = n_a attn-layers + n_s ssm-layers. Decode cost is dominated by the
(memory-bound, no-GQA-expensive) attn-decode; ssm-decode is O(1) cheap.

  co_schedule         : every layer shares SMs (two_stream)   — no reservation
  agnostic_protect    : every layer reserves decode its floor — wastes SMs on cheap ssm layers
  layer_aware_protect : reserve ONLY attn layers; ssm layers share (prefill gets the SMs back)

Hypothesis: layer_aware keeps the SLO-critical attn-decode ITL bounded (same as agnostic) but
gives prefill ~full SMs during the (many) ssm layers → less starvation → higher goodput. If so,
layer-type awareness improves PD-mux — esp. for mostly-ssm hybrids (zamba2: 83% ssm layers).

Full-model step cost = Σ_layers (per-layer-type stream time). Per-type values from the E5
opt LUT (pf=attn×dec=attn and pf=ssm×dec=ssm cells; two_stream & green_ctx_protect backends).
GPU-free. Uses the decoupled scheduler (decode ITL bounded on reserved SMs).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))
for _p in (_CHAR, os.path.abspath(os.path.join(_here, "..", "..", ".."))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.e6_queue_sim.workload import gen_requests
from experiments.e6_queue_sim.simulator import simulate
from experiments.e6_queue_sim.run_queue_sim import metrics

# per-policy backend used for each layer type
_POLICY = {
    "fused":               {"attn": "two_stream",       "ssm": "two_stream"},   # vLLM default: prefill+decode in ONE forward (sync)
    "co_schedule":         {"attn": "two_stream",       "ssm": "two_stream"},
    "agnostic_protect":    {"attn": "green_ctx_protect", "ssm": "green_ctx_protect"},
    "layer_aware_protect": {"attn": "green_ctx_protect", "ssm": "two_stream"},
}

# fused/co_schedule = sync (decode coupled to the forward); the *_protect = decoupled (decode on reserved SMs)
_SCHED = {"fused": "sync", "co_schedule": "decoupled",
          "agnostic_protect": "decoupled", "layer_aware_protect": "decoupled"}


def _nearest(b, grid):
    return min(grid, key=lambda g: abs(g - b))


class FullModelLM:
    """Sums per-layer-type stream times into a full-forward (prefill, decode) cost."""

    def __init__(self, lut_csv, n_attn, n_ssm):
        import pandas as pd
        df = pd.read_csv(lut_csv, skiprows=1)
        df.columns = [c.split("__")[0] for c in df.columns]
        df = df[df.status == "ok"].copy()
        for c in ("decode_batch", "prefill_stream_ms", "decode_stream_ms",
                  "solo_prefill_ms", "solo_decode_ms", "context_len"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        self.n = {"attn": n_attn, "ssm": n_ssm}
        # tbl[(backend, layer_type)] -> {B -> (pstream, dstream, solo_p, solo_d)}
        self.tbl, self.green_best = {}, {}
        for _, r in df.iterrows():
            lt = r.prefill_layer
            if r.prefill_layer != r.decode_layer:        # full-model layers are single-type
                continue
            bk = "green_ctx" if r.backend == "green_ctx" else r.backend
            row = (r.prefill_stream_ms, r.decode_stream_ms, r.solo_prefill_ms, r.solo_decode_ms)
            self.tbl.setdefault((bk, lt), {})[int(r.decode_batch)] = row
        # green_ctx_protect may be absent (no_room) → fall back to two_stream for that type
        self.fused_decode_frac = 0.0  # unused; for simulate compat

    def _get(self, backend, lt, b):
        g = self.tbl.get((backend, lt))
        if not g:
            g = self.tbl.get(("two_stream", lt))            # fallback
        if not g:
            return (0.0, 0.0, 0.0, 0.0)
        return g[_nearest(max(b, 1), list(g))]

    def streams(self, policy, pf, dec, b, ctx):
        be = _POLICY[policy]
        ps = ds = sp = sd = 0.0
        for lt in ("attn", "ssm"):
            p, d, s_p, s_d = self._get(be[lt], lt, b)
            k = self.n[lt]
            ps += k * p; ds += k * d; sp += k * s_p; sd += k * s_d
        return ps, ds, sp, sd

    def step_ms(self, policy, pf, dec, b, ctx, prefill_active):
        """Sync-scheduler step time (used by `fused`). The fused mixed-batch forward runs
        prefill(P)+decode(B) over (P+B) rows in ONE pass; measured fusion_saving≈0, so the
        full-model step ≈ Σ_layers(solo_prefill + solo_decode) = sp + sd when prefill is in
        the batch, else sd. Decode ITL = this step (coupled) — that's fused's disadvantage."""
        ps, ds, sp, sd = self.streams(policy, pf, dec, b, ctx)
        return (sp + sd if prefill_active else sd), ""


def main():
    p = argparse.ArgumentParser(description="layer-aware vs agnostic PD-mux")
    p.add_argument("--model", default="zamba2_2.7b")
    p.add_argument("--lut", default=None, help="E5 opt LUT (default e5_sim_b8_opt)")
    p.add_argument("--lambdas", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    p.add_argument("--slo-ms", type=float, default=1.0)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--prefill-budget", type=int, default=8)
    p.add_argument("--n-requests", type=int, default=400)
    p.add_argument("--prompt-lens", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    p.add_argument("--output-lens", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    from shared.loaders import get_model_config
    cfg = get_model_config(a.model)
    L = cfg["num_layers"]
    fa = cfg.get("attention", {}).get("attn_layer_fraction", 1.0)
    fs = cfg.get("ssm", {}).get("ssm_layer_fraction", 1.0)
    n_a, n_s = max(1, round(L * fa)), max(1, round(L * fs))
    lut = a.lut or str(Path(_CHAR) / "results_v2" / "e5_sim_b8_opt" /
                       f"serving_coexec_full_{a.model}_a100_sxm4_80gb.csv")
    lm = FullModelLM(lut, n_a, n_s)

    print(f"=== layer-aware PD-mux: {a.model}  ({L} layers = {n_a} attn + {n_s} ssm)  "
          f"budget={a.prefill_budget} SLO(ITL)≤{a.slo_ms}ms ===")
    print(f"{'policy':22}{'λ':>6}{'ITL_p99':>9}{'TTFT_p99':>10}{'SLO_att':>8}{'tok/s':>9}{'GOODPUT':>9}")
    rows = []
    for lam in a.lambdas:
        good = {}
        for pol in ("fused", "co_schedule", "agnostic_protect", "layer_aware_protect"):
            reqs = gen_requests(a.n_requests, lam, a.prompt_lens, a.output_lens, a.chunk, a.seed)
            reqs, st = simulate(reqs, pol, lm, "full", "full", a.ctx,
                                prefill_budget=a.prefill_budget, scheduler=_SCHED[pol])
            m = metrics(reqs, st["makespan_ms"], a.slo_ms)
            good[pol] = m["goodput_tok_s"]
            rows.append({"model": a.model, "policy": pol, "lambda_req_ms": lam, "slo_ms": a.slo_ms, **m})
            print(f"{pol:22}{lam:6.2f}{m['itl_p99']:9.3f}{m['ttft_p99']:10.1f}"
                  f"{m['slo_attain']:8.3f}{m['throughput_tok_s']:9.0f}{m['goodput_tok_s']:9.0f}")
        la, fu, ag = good["layer_aware_protect"], good["fused"], good["agnostic_protect"]
        vs = lambda x: ("—" if x == 0 else f"{la/x:.2f}x")
        print(f"  → layer_aware {la:.0f}  vs fused(vLLM) {fu:.0f} [{vs(fu)}]  vs agnostic {ag:.0f} [{vs(ag)}]")
        print("  " + "-" * 60)

    out = Path(_CHAR) / "results_v2" / "e6" / f"layer_aware_{a.model}_b{a.prefill_budget}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
