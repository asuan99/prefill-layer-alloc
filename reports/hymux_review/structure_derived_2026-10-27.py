#!/usr/bin/env python3
"""Recompute the four 'derived' numbers of the 2026-10-27 structure hand-over from boot_facts.json.

Companion of STRUCTURE_batching_admission_2026-10-27.md (section 3).  CPU only, read only, no GPU.

Inputs (measured = boot banner values written by the 128k long-context probe, A100-SXM4 80GB,
Nemotron-Nano-9B-v2-Base, ctx 131072, mem-fraction-static 0.6, cap 48, radix OFF):
  workspace/engine-port/results/cap_char/probe_128k/longctx_probe_20261020T034615Z/nano_pdmux/boot_facts.json
  workspace/engine-port/results/cap_char/probe_128k/longctx_probe_20261020T034615Z/llama/boot_facts.json
NOTE: pool_bytes / slot_bytes / kv_bytes_per_token in those files are themselves computed by the probe from banner
GiB values printed with 2 decimals (pool = (kv_k + kv_v + conv + ssm) GiB, slot = (conv + ssm) GiB / (cap + 1),
kv/token = (kv_k + kv_v) GiB / kv_cache_tokens).  The c_mem column is a formula output, not a measured concurrency.

Every number printed here is [derived].  The analytic cross-check uses
workspace/engine-port/scripts/costmodel/analytic_terms.py (decision 27), never hand arithmetic.

Usage:  python3 -I reports/hymux_review/structure_derived_2026-10-27.py [--json]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "workspace/engine-port/results/cap_char/probe_128k/longctx_probe_20261020T034615Z"
NANO = RUN / "nano_pdmux" / "boot_facts.json"
LLAMA = RUN / "llama" / "boot_facts.json"
ANALYTIC = REPO / "workspace/engine-port/scripts/costmodel/analytic_terms.py"

# Values stated in the hand-over (chat session, unaudited) -- the things being checked.
CHAT = {
    "slot_in_kv_tokens": 8.8e3,        # "slot 1개 ~ KV 8.8k 토큰"
    "ceiling_at_L0": 230,              # "문맥 0 동시성 천장 ~ 230"
    "cap48_boundary_tokens": 33e3,     # "cap 48 slot-bound <-> memory-bound 경계 ~ L+O 33k"
    "c6_breakeven_tokens": 8.8e3,      # C6 "손익분기 L+O ~ 8.8k"
    "c_mem": {"16384": 80, "32768": 48, "65536": 27, "98304": 18, "126976": 14},
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_analytic():
    spec = importlib.util.spec_from_file_location("analytic_terms", ANALYTIC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analytic_terms"] = mod  # dataclass processing looks the module up
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def c_mem(pool: float, slot: float, kv: float, L: int) -> int:
    """cmem.formula verbatim: floor(pool_bytes / (slot_bytes + kv_bytes_per_token * L))."""
    return math.floor(pool / (slot + kv * L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    nano = json.loads(NANO.read_text())
    llama = json.loads(LLAMA.read_text())
    cm = nano["cmem"]
    assert cm["formula"] == "floor(pool_bytes / (slot_bytes + kv_bytes_per_token * L))", cm["formula"]
    pool, slot, kv = float(cm["pool_bytes"]), float(cm["slot_bytes"]), float(cm["kv_bytes_per_token"])
    cap = int(nano["cap"])
    kv_tokens = int(nano["banner"]["kv_cache_tokens"])
    kv_llama = float(llama["cmem"]["kv_bytes_per_token"])

    out: dict = {"inputs": {
        "nano_boot_facts": str(NANO.relative_to(REPO)), "nano_sha256": sha256(NANO),
        "llama_boot_facts": str(LLAMA.relative_to(REPO)), "llama_sha256": sha256(LLAMA),
        "pool_bytes": pool, "slot_bytes": slot, "kv_bytes_per_token": kv, "cap": cap,
        "kv_cache_tokens": kv_tokens, "llama_kv_bytes_per_token": kv_llama,
        "banner_gib": {k: nano["banner"][k] for k in ("kv_k_gib", "kv_v_gib", "mamba_conv_gib", "mamba_ssm_gib")},
    }}

    # 0. c_mem table: recompute and compare with the stored column.
    table = {}
    for L, rec in sorted(cm["by_L"].items(), key=lambda kv_: int(kv_[0])):
        val = c_mem(pool, slot, kv, int(L))
        table[L] = {"c_mem_recomputed": val, "c_mem_stored": rec["c_mem"], "match": val == rec["c_mem"],
                    "kv_token_bound_stored": rec["kv_token_bound"],
                    "kv_token_bound_recomputed": kv_tokens // int(L),
                    "physical_min_cap_kvbound": min(cap, kv_tokens // int(L)),
                    "chat_value": CHAT["c_mem"].get(L)}
    out["c_mem_table"] = table

    # C5 check point: arXiv-Sum (Nano tokenizer, kept <= 16384) mean input 7116 + mean output 203
    # [measured, results/handoff_review_2026-10-15/arxsum/ARXSUM_STATS_2026-10-15.txt]; max kept in+out 16744 is cited
    # from results/dp_routing/PLAN_DP_routing_per_gpu_partition_2026-10-26.md:37.
    arx = {"L_mean_in_plus_out": 7116 + 203, "L_max_in_plus_out": 16744}
    out["c5_arxsum"] = {k: {"L": v, "c_mem_fungible": c_mem(pool, slot, kv, v), "kv_only_bound": kv_tokens // v,
                            "physical_min_cap_kvbound": min(cap, kv_tokens // v)} for k, v in arx.items()}

    # 1. slot expressed in Nano KV tokens  (= s / k, analytic L_star)
    d1 = slot / kv
    # 2. concurrency ceiling at L = 0 under a fungible pool
    d2_exact = pool / slot
    d2_formula = c_mem(pool, slot, kv, 0)
    # 3. cap-48 boundary: largest L with c_mem(L) >= cap  (fungible pool)  and  KV-pool-only boundary (physical split)
    L_bound_fungible = (pool / cap - slot) / kv
    L_int = math.floor(L_bound_fungible)
    assert c_mem(pool, slot, kv, L_int) >= cap and c_mem(pool, slot, kv, L_int + 1) < cap
    L_bound_physical = kv_tokens / cap
    # 4. C6 break-even vs a transformer: s / (k_T - k), with k_T from the Llama-3.1-8B boot of the same Job
    d4_vs_llama = slot / (kv_llama - kv)

    # analytic cross-check (decision 27): s, k, L* for Nano; s/(k_T - k) with the Llama-3.1-8B KV shape
    at = load_analytic()
    a = at.nano9b_v2()
    s_an, k_an, Lstar_an = at.state_bytes_per_request(a), at.kv_bytes_per_token(a), at.L_star(a)
    # Llama-3.1-8B KV per token: 32 layers x 2 x 8 kv-heads x 128 x bf16 -> same closed form as kv_bytes_per_token
    llama_like = at.Arch(d=4096, n_q=32, n_kv=8, d_h=128, n_h=0, p=0, G=0, N=0, K=0, Q=1, d_ff=14336, V=128256,
                         n_A=32, n_M=0, n_F=32, mlp_gated=True)
    kT_an = at.kv_bytes_per_token(llama_like)
    a31 = at.nano9b_v2(n_A=31, n_M=0)
    k31_an = at.kv_bytes_per_token(a31)

    out["derived"] = {
        "D1_slot_in_kv_tokens": {"value": d1, "chat": CHAT["slot_in_kv_tokens"],
                                 "analytic_L_star": Lstar_an, "analytic_s_bytes": s_an, "analytic_k_bytes": k_an,
                                 "boot_vs_analytic_s_rel": slot / s_an - 1, "boot_vs_analytic_k_rel": kv / k_an - 1},
        "D2_ceiling_L0": {"value_exact": d2_exact, "value_floor": d2_formula, "chat": CHAT["ceiling_at_L0"],
                          "realized_slots_in_this_boot": cap + 1},
        "D3_cap48_boundary": {"fungible_pool_L": L_bound_fungible, "fungible_pool_L_int": L_int,
                              "physical_kv_pool_L": L_bound_physical, "chat": CHAT["cap48_boundary_tokens"]},
        "D4_c6_breakeven": {"chat": CHAT["c6_breakeven_tokens"],
                            "chat_quantity_is": "s/k = L* (slot vs Nano's own KV), not hybrid-vs-transformer",
                            "boot_s_over_kLlama_minus_k": d4_vs_llama,
                            "analytic_s_over_kLlama_minus_k": s_an / (kT_an - k_an),
                            "analytic_s_over_k31_minus_k": s_an / (k31_an - k_an),
                            "analytic_kLlama": kT_an, "analytic_k31": k31_an},
        "analytic_version": at.VERSION,
    }

    if args.json:
        print(json.dumps(out, indent=1, sort_keys=True))
        return 0
    print("[inputs, measured via boot banner]", f"pool={pool:.2f}B slot={slot:.2f}B kv/token={kv:.3f}B cap={cap}",
          f"kv_cache_tokens={kv_tokens} llama kv/token={kv_llama:.3f}B")
    print("[derived] c_mem table (L: recomputed / stored / chat ; KV-only bound ; physical min(cap, KV bound))")
    for L, r in table.items():
        print(f"  {L:>6}: {r['c_mem_recomputed']:>3} / {r['c_mem_stored']:>3} / {r['chat_value']}"
              f" ; {r['kv_token_bound_recomputed']:>3} ; {r['physical_min_cap_kvbound']:>3}  match={r['match']}")
    d = out["derived"]
    print(f"[derived] D1 slot/kv = {d1:.1f} tokens (chat 8.8k); analytic L* = {Lstar_an:.1f};"
          f" boot s vs analytic s {d['D1_slot_in_kv_tokens']['boot_vs_analytic_s_rel']:+.4%},"
          f" boot k vs analytic k {d['D1_slot_in_kv_tokens']['boot_vs_analytic_k_rel']:+.4%}")
    print(f"[derived] D2 pool/slot = {d2_exact:.2f} (floor {d2_formula}; chat 230); slots realized in this boot = {cap + 1}")
    print(f"[derived] D3 cap-48 boundary: fungible-pool L = {L_bound_fungible:.1f} (int {L_int});"
          f" physical KV pool L = kv_cache_tokens/cap = {L_bound_physical:.1f} (chat 33k)")
    print(f"[derived] D4 hybrid-vs-transformer break-even s/(k_T-k): boot Llama-3.1-8B = {d4_vs_llama:.1f};"
          f" analytic (Llama shape) = {s_an / (kT_an - k_an):.1f}; analytic a31 = {s_an / (k31_an - k_an):.1f}"
          f" (chat C6 8.8k = s/k, a different quantity)")
    for k, r in out["c5_arxsum"].items():
        print(f"[derived] C5 arXiv-Sum {k} L={r['L']}: c_mem={r['c_mem_fungible']}, KV-only bound={r['kv_only_bound']},"
              f" physical min(cap, KV)={r['physical_min_cap_kvbound']}")
    print(f"[analytic] {at.VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
