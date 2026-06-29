"""Affirmative-case figures for the layer-type-AWARE hypothesis (the user's original thesis).

Shows, from the full-model sim (run_layer_aware machinery), that reserving SMs ONLY for the
expensive attn-decode layers (layer_aware_protect) and releasing them during the many cheap
ssm-decode layers beats both layer-agnostic reservation and plain co-scheduling — for TEMPORAL
hybrids at a realistic per-token SLO.

Panels: (A) goodput vs SLO with the layer_aware-wins region shaded; (B) the throughput/TTFT/ITL
tradeoff; (C) the mechanism (per-layer SM allocation); (D) temporal vs spatial applicability.
Output: reports/figures/layer_aware_*.png
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))
for _p in (_CHAR, os.path.abspath(os.path.join(_here, "..", "..", ".."))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from experiments.e6_queue_sim.run_layer_aware import FullModelLM, _POLICY, _SCHED
from experiments.e6_queue_sim.workload import gen_requests
from experiments.e6_queue_sim.simulator import simulate
from experiments.e6_queue_sim.run_queue_sim import metrics
from shared.loaders import get_model_config

MODEL = "zamba2_2.7b"
BUDGET = 8
LAM = 0.5
CTX = 4096
POLICIES = ["fused", "co_schedule", "agnostic_protect", "layer_aware_protect"]
LABELS = {"fused": "fused\n(vLLM default)",
          "co_schedule": "co_schedule\n(two-stream, no reserve)",
          "agnostic_protect": "agnostic_protect\n(reserve every layer)",
          "layer_aware_protect": "layer_aware_protect\n(reserve attn layers only)"}
COL = {"fused": "#c5221f", "co_schedule": "#9aa0a6", "agnostic_protect": "#e8710a", "layer_aware_protect": "#1a73e8"}

cfg = get_model_config(MODEL)
L = cfg["num_layers"]
n_a = max(1, round(L * cfg.get("attention", {}).get("attn_layer_fraction", 1.0)))
n_s = max(1, round(L * cfg.get("ssm", {}).get("ssm_layer_fraction", 1.0)))
lut = str(Path(_CHAR) / "results_v2" / "e5_sim_b8_opt" /
          f"serving_coexec_full_{MODEL}_a100_sxm4_80gb.csv")
lm = FullModelLM(lut, n_a, n_s)

# simulate each policy ONCE, then sweep SLO analytically via metrics()
sim = {}
for pol in POLICIES:
    reqs = gen_requests(400, LAM, [512, 1024, 2048, 4096], [64, 128, 256], 256, 0)
    reqs, st = simulate(reqs, pol, lm, "full", "full", CTX, prefill_budget=BUDGET, scheduler=_SCHED[pol])
    sim[pol] = (reqs, st["makespan_ms"])

SLO = np.arange(25, 121, 2.5)
good = {p: np.array([metrics(sim[p][0], sim[p][1], s)["goodput_tok_s"] for s in SLO]) for p in POLICIES}
# scalar summaries (ITL/throughput are SLO-independent; take at a loose SLO)
summ = {p: metrics(sim[p][0], sim[p][1], 100.0) for p in POLICIES}

fig = plt.figure(figsize=(13, 9.5))
gs = fig.add_gridspec(2, 1, hspace=0.5)

# ---- Panel A: goodput vs SLO ----
axA = fig.add_subplot(gs[0, 0])
for p in POLICIES:
    axA.plot(SLO, good[p] / 1000, color=COL[p], lw=2.4,
             label=LABELS[p].replace("\n", " "), marker="o", ms=3, markevery=4)
# shade ONLY the SLO points where layer_aware is strictly best (non-contiguous-safe)
la, ag, co = good["layer_aware_protect"], good["agnostic_protect"], good["co_schedule"]
win = (la > ag * 1.01) & (la > co * 1.01)
if win.any():
    axA.fill_between(SLO, 0, 1.45, where=win, color="#1a73e8", alpha=0.09, step="mid")
    # annotate in the robust 2x zone (where la ~= 2*ag)
    robust = win & (la > ag * 1.8)
    if robust.any():
        axA.text(0.97, 0.62, "layer_aware\nbest (~2x)", transform=axA.transAxes, color="#1a73e8",
                 ha="right", va="center", fontsize=10, fontweight="bold")
        axA.text(SLO[robust].min(), 0.05, f"SLO >= {SLO[robust].min():.0f}ms", color="#1a73e8",
                 ha="center", va="bottom", fontsize=8)
axA.set_xlabel("per-token SLO (TBT, ms)")
axA.set_ylabel("goodput@SLO  (k tok/s)")
axA.set_title("(A) goodput vs SLO -- zamba2_2.7b (9 attn + 45 ssm)", fontweight="bold")
axA.legend(fontsize=8.3, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4,
           framealpha=0.95)   # below the panel (4-way legend covers data if placed inside)
axA.grid(alpha=0.3)
axA.set_ylim(0, 1.45)

# ---- Panel B: twin-axis tradeoff. throughput bars (left, linear) + TTFT & ITL lines
#      (right, LOG ms) so the two latencies separate by ~3 decades and each policy
#      difference is readable (the old linear right-axis put TTFT[s] & ITL[ms] both in 37-85). ----
axB = fig.add_subplot(gs[1, 0])
x = np.arange(len(POLICIES))
thr = [summ[p]["throughput_tok_s"] for p in POLICIES]
ttft = [summ[p]["ttft_p99"] for p in POLICIES]          # ms
itl = [summ[p]["itl_p99"] for p in POLICIES]            # ms
short = {"fused": "fused", "co_schedule": "co_schedule", "agnostic_protect": "agnostic", "layer_aware_protect": "layer_aware"}
axB.bar(x, thr, 0.6, color=[COL[p] for p in POLICIES], alpha=0.55, edgecolor="k", linewidth=0.5, zorder=1)
for i in range(len(POLICIES)):
    axB.text(i, thr[i] + 25, f"{thr[i]:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
             color=COL[POLICIES[i]])
axB.set_xticks(x); axB.set_xticklabels([short[p] for p in POLICIES], fontsize=9)
axB.set_ylabel("throughput (tok/s)  [bars]")
axB.set_ylim(0, max(thr) * 1.18)
axB.set_title("(B) tradeoff: throughput (bars) vs TTFT & decode-ITL (lines, log)", fontweight="bold")
axB.grid(axis="y", alpha=0.25)

axB2 = axB.twinx()
axB2.set_yscale("log")
axB2.plot(x, ttft, "--o", color="#202124", lw=1.8, ms=7, zorder=3, label="TTFT p99")
axB2.plot(x, itl, ":s", color="#c5221f", lw=1.8, ms=7, zorder=3, label="decode ITL p99")
for i in range(len(POLICIES)):
    axB2.annotate(f"{ttft[i]/1000:.0f}s", (i, ttft[i]), textcoords="offset points", xytext=(0, 9),
                  ha="center", fontsize=8.5, color="#202124", fontweight="bold")
    axB2.annotate(f"{itl[i]:.0f}ms", (i, itl[i]), textcoords="offset points", xytext=(0, -13),
                  ha="center", fontsize=8.5, color="#c5221f", fontweight="bold")
axB2.set_ylabel("latency (ms, log)  [lines]")
axB2.set_ylim(20, 5e5)
axB2.legend(fontsize=8.5, loc="center right", framealpha=0.95)

fig.suptitle("Layer-type-AWARE reservation improves PD-mux (full-model sim, zamba2_2.7b)",
             fontsize=13.5, fontweight="bold", y=0.96)
out = Path(_CHAR).parents[1] / "reports" / "figures"
out.mkdir(parents=True, exist_ok=True)
f1 = out / "layer_aware_result.png"
fig.savefig(f1, dpi=130, bbox_inches="tight")
print(f"wrote {f1}")

# ---- separate Panel D: temporal vs spatial applicability ----
figD, (axd1, axd2) = plt.subplots(1, 2, figsize=(12, 2.6))
for ax, name, na, ns, kind, ok in [
        (axd1, "zamba2 (temporal)", 9, 45, "attn/ssm in SEPARATE layers", True),
        (axd2, "falcon_h1 (spatial)", 0, 0, "attn+ssm PARALLEL in every layer", False)]:
    if ok:
        attn_pos = set(np.linspace(1, 53, 9).round().astype(int).tolist())
        for i in range(54):
            c = "#e8710a" if i in attn_pos else "#1a73e8"
            ax.add_patch(plt.Rectangle((i, 0), 0.92, 1, color=c, alpha=0.85))
        ax.set_xlim(-1, 55)
        ax.text(27, -0.6, "-> attn layers separable for reservation => layer_aware APPLIES (OK)", ha="center", color="#1a73e8", fontsize=9, fontweight="bold")
    else:
        for i in range(32):
            ax.add_patch(plt.Rectangle((i, 0.05), 0.92, 0.45, color="#e8710a", alpha=0.85))
            ax.add_patch(plt.Rectangle((i, 0.52), 0.92, 0.45, color="#1a73e8", alpha=0.7))
        ax.set_xlim(-1, 33)
        ax.text(16, -0.6, "-> every layer is attn+ssm together => cannot separate, layer_aware N/A", ha="center", color="#9aa0a6", fontsize=9, fontweight="bold")
    ax.set_ylim(-0.9, 1.2); ax.axis("off")
    ax.set_title(f"{name} — {kind}", fontsize=10, fontweight="bold")
figD.suptitle("(D) Applicability -- layer_aware is limited to TEMPORAL hybrids", fontsize=12, fontweight="bold", y=1.04)
f2 = out / "layer_aware_applicability.png"
figD.savefig(f2, dpi=130, bbox_inches="tight")
print(f"wrote {f2}")

# print the numbers for the report
print("\n=== data for report ===")
for p in POLICIES:
    s = summ[p]
    print(f"{p:22} thr={s['throughput_tok_s']:.0f} ttft_p99={s['ttft_p99']:.0f}ms itl_p99={s['itl_p99']:.1f}ms")
print("win SLO range:", SLO[win].min() if win.any() else None, "-", SLO[win].max() if win.any() else None)
for s in (40, 50, 60, 100):
    j = int(np.argmin(abs(SLO - s)))
    print(f"  SLO~{s}: co={good['co_schedule'][j]:.0f} ag={good['agnostic_protect'][j]:.0f} la={good['layer_aware_protect'][j]:.0f}")
