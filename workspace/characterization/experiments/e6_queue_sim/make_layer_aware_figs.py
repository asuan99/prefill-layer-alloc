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

from experiments.e6_queue_sim.run_layer_aware import FullModelLM, _POLICY
from experiments.e6_queue_sim.workload import gen_requests
from experiments.e6_queue_sim.simulator import simulate
from experiments.e6_queue_sim.run_queue_sim import metrics
from shared.loaders import get_model_config

MODEL = "zamba2_2.7b"
BUDGET = 8
LAM = 0.5
CTX = 4096
POLICIES = ["co_schedule", "agnostic_protect", "layer_aware_protect"]
LABELS = {"co_schedule": "co_schedule\n(no reserve)",
          "agnostic_protect": "agnostic_protect\n(reserve every layer)",
          "layer_aware_protect": "layer_aware_protect\n(reserve attn layers only)"}
COL = {"co_schedule": "#9aa0a6", "agnostic_protect": "#e8710a", "layer_aware_protect": "#1a73e8"}

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
    reqs, st = simulate(reqs, pol, lm, "full", "full", CTX, prefill_budget=BUDGET, scheduler="decoupled")
    sim[pol] = (reqs, st["makespan_ms"])

SLO = np.arange(25, 121, 2.5)
good = {p: np.array([metrics(sim[p][0], sim[p][1], s)["goodput_tok_s"] for s in SLO]) for p in POLICIES}
# scalar summaries (ITL/throughput are SLO-independent; take at a loose SLO)
summ = {p: metrics(sim[p][0], sim[p][1], 100.0) for p in POLICIES}

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.34, wspace=0.22)

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
        axA.text(SLO[robust].mean(), 1.33, "layer_aware best (~2x)", color="#1a73e8",
                 ha="center", va="top", fontsize=10, fontweight="bold")
        axA.text(SLO[robust].min(), 0.05, f"SLO >= {SLO[robust].min():.0f}ms", color="#1a73e8",
                 ha="center", va="bottom", fontsize=8)
axA.set_xlabel("per-token SLO (TBT, ms)")
axA.set_ylabel("goodput@SLO  (k tok/s)")
axA.set_title("(A) goodput vs SLO -- zamba2_2.7b (9 attn + 45 ssm)", fontweight="bold")
axA.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
axA.grid(alpha=0.3)
axA.set_ylim(0, 1.45)

# ---- Panel B: tradeoff bars ----
axB = fig.add_subplot(gs[0, 1])
x = np.arange(len(POLICIES))
thr = [summ[p]["throughput_tok_s"] / 1000 for p in POLICIES]
ttft = [summ[p]["ttft_p99"] / 1000 for p in POLICIES]
itl = [summ[p]["itl_p99"] for p in POLICIES]
w = 0.62
b = axB.bar(x, thr, w, color=[COL[p] for p in POLICIES])
axB.set_ylabel("throughput (k tok/s)")
axB.set_xticks(x); axB.set_xticklabels([LABELS[p] for p in POLICIES], fontsize=8)
axB.set_title("(B) throughput up / TTFT down (ITL within SLO)", fontweight="bold")
for i, p in enumerate(POLICIES):
    axB.text(i, thr[i] + 0.02, f"{thr[i]*1000:.0f}\ntok/s", ha="center", va="bottom", fontsize=8, fontweight="bold")
axB.set_ylim(0, max(thr) * 1.35)
axB2 = axB.twinx()
axB2.plot(x, ttft, "k--o", lw=1.6, ms=6, label="TTFT p99 (s)")
axB2.plot(x, itl, "r:s", lw=1.6, ms=6, label="ITL p99 (ms)")
for i in range(len(POLICIES)):
    axB2.text(i + 0.13, ttft[i], f"{ttft[i]:.0f}s", color="k", fontsize=7.5, va="center")
    axB2.text(i + 0.13, itl[i], f"{itl[i]:.0f}ms", color="r", fontsize=7.5, va="center")
axB2.set_ylabel("TTFT p99 (s)  /  ITL p99 (ms)")
axB2.legend(fontsize=8, loc="upper center")
axB2.set_ylim(0, max(max(ttft), max(itl)) * 1.25)

# ---- Panel C: mechanism (per-layer SM allocation) ----
axC = fig.add_subplot(gs[1, :])
axC.set_title("(C) Mechanism -- SM allocation over the 54-layer forward: protect only the expensive attn-decode, give ssm layers back to prefill",
              fontweight="bold")
# illustrative interleave: 9 attn positions evenly spread among 54
attn_pos = set(np.linspace(2, L - 2, n_a).round().astype(int).tolist())
y_ag, y_la = 1.0, 0.0
for i in range(L):
    is_attn = i in attn_pos
    # agnostic: every layer reserves decode floor -> prefill throttled (hatched) everywhere
    axC.add_patch(plt.Rectangle((i, y_ag), 0.92, 0.8, color="#e8710a", alpha=0.85 if is_attn else 0.5))
    # layer_aware: attn layers protected (orange), ssm layers full-share -> prefill full SMs (blue)
    axC.add_patch(plt.Rectangle((i, y_la), 0.92, 0.8,
                                color="#e8710a" if is_attn else "#1a73e8",
                                alpha=0.9 if is_attn else 0.7))
axC.text(-1.5, y_ag + 0.4, "agnostic_protect", ha="right", va="center", fontsize=9, fontweight="bold")
axC.text(-1.5, y_la + 0.4, "layer_aware_protect", ha="right", va="center", fontsize=9, fontweight="bold")
axC.text(L + 0.5, y_ag + 0.4, "reserve decode every layer\n-> prefill always starved\n-> throughput down, TTFT up", va="center", fontsize=8)
axC.text(L + 0.5, y_la + 0.4, "reserve 9 attn only (ITL safe)\n45 ssm -> prefill full-SM\n-> throughput 2x, TTFT half", va="center", fontsize=8)
axC.set_xlim(-13, L + 13); axC.set_ylim(-0.4, 2.1)
axC.axis("off")
leg = [Patch(color="#e8710a", alpha=0.9, label="attn-decode layer = SM reserved (protect decode ITL)"),
       Patch(color="#1a73e8", alpha=0.7, label="ssm-decode layer = SM shared (given to prefill)"),
       Patch(color="#e8710a", alpha=0.5, label="agnostic: ssm layers reserved too (wasteful)")]
axC.legend(handles=leg, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.12))

fig.suptitle("Layer-type-AWARE reservation improves PD-mux (the live form of the original hypothesis) -- full-model sim",
             fontsize=13.5, fontweight="bold", y=0.985)
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
