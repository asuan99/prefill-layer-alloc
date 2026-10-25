#!/usr/bin/env python3
"""Observed no-contention prefill TTFT floor (M3 J1 N=1, fused_sym boot1 FLOOR.json; audit CONFIRMED, scope: L 1k-16k,
Nano-9B-v2, 108 SM, A100) against the analytic-v1 ideal-roofline lower bound (same model constants).
Observed = measured; ideal = [derived].  The gap is NOT decomposed here beyond what the ideal operator split allows."""
import sys, json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]
# reuse mk()/t_ops() from the scaling script (everything before its plotting section)
exec((HERE / "scaling_batch_seqlen.py").read_text().split("a = T.nano9b_v2(); pre")[0])
J1 = (ROOT / "workspace/engine-port/results/multigpu/manual_retrieval/pdmux-multigpu-2106f42-20261018074950/results/multigpu/"
      "m3_pdmux-multigpu-2106f42-20261018074950/boots/m3_N1_fused_sym_boot1/FLOOR.json")
d = json.loads(J1.read_text()); Lo = np.array(sorted(int(k) for k in d["points"])); To = np.array([d["points"][str(l)]["floor_s"] for l in Lo])
a = T.nano9b_v2(); pre, dec, pm, _ = mk(a)
def classes(L):
    A = a.n_A*t_ops(pre("A", L)); Mo = pre("M", L); Mg = a.n_M*t_ops(Mo[:2]); Ms = a.n_M*t_ops(Mo[2:]); F = a.n_F*t_ops(pre("F", L))
    head = max(2*a.d*a.V/PEAK, a.d*a.V*a.beta/BW)          # LM head once; weight read dominates
    return dict(A=A, Mg=Mg, Ms=Ms, F=F, head=head)
Lg = np.linspace(512, 16896, 60)
C = {k: np.array([classes(int(L))[k] for L in Lg]) for k in ("A", "Mg", "Ms", "F", "head")}
Tideal = sum(C.values()); Tid_o = np.array([sum(classes(int(L)).values()) for L in Lo])
slope, icpt = np.polyfit(Lo, To, 1)
fig, ax = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
cols = {"A": "#0072B2", "Mg": "#D55E00", "Ms": "#CC79A7", "F": "#999999", "head": "#E69F00"}
lab = {"A": "attention (proj+core)", "Mg": "Mamba2 in/out_proj GEMM", "Ms": "Mamba2 conv+SSD scan", "F": "MLP", "head": "LM head weight read"}
ax[0].stackplot(Lg, *[C[k]*1e3 for k in ("Mg", "F", "Ms", "A", "head")], colors=[cols[k] for k in ("Mg", "F", "Ms", "A", "head")],
                labels=[lab[k] for k in ("Mg", "F", "Ms", "A", "head")], alpha=.55)
ax[0].plot(Lg, Tideal*1e3, color="#222", lw=1.5, label="ideal total (analytic-v1)")
ax[0].scatter(Lo, To*1e3, s=60, color="#111", zorder=5, label="observed floor (J1, n=16/pt)")
ax[0].plot(Lg, (icpt + slope*Lg)*1e3, color="#111", ls="--", lw=1, label=f"fit {icpt*1e3:.1f} ms + {slope*1e3:.3f} ms/token")
ax[0].set_xlabel("input length L (tokens)"); ax[0].set_ylabel("TTFT floor (ms)"); ax[0].legend(frameon=False, fontsize=8.5, loc="upper left")
ax[0].set_title("(a) Observed vs ideal-roofline lower bound", loc="left", fontsize=10.5); ax[0].grid(alpha=.2)
ax[1].plot(Lo, To/Tid_o, "o-", color="#111", lw=1.8); ax[1].axhline(1, color="#999", ls=":")
for l, r in zip(Lo, To/Tid_o): ax[1].annotate(f"{r:.2f}×", (l, r), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
ax[1].set_ylim(0.8, 2.8); ax[1].set_xlabel("input length L"); ax[1].set_ylabel("observed ÷ ideal")
ax[1].set_title("(b) Gap factor (1.0 = roofline reached)", loc="left", fontsize=10.5); ax[1].grid(alpha=.2)
Lx = Lo
ax[2].plot(Lx, To/Lx*1e6, "o-", color="#111", lw=1.8, label="observed  T/L")
ax[2].plot(Lx, (To - icpt)/Lx*1e6, "s--", color="#111", lw=1.2, label="observed (T − intercept)/L")
ax[2].plot(Lg, Tideal/Lg*1e6, color="#0072B2", lw=2, label="ideal  T/L")
ax[2].set_ylim(0, 140); ax[2].set_xlabel("input length L"); ax[2].set_ylabel("µs per token"); ax[2].legend(frameon=False, fontsize=9)
ax[2].set_title("(c) Per-token cost: intercept vs slope", loc="left", fontsize=10.5); ax[2].grid(alpha=.2)
fig.suptitle("Prefill, Nemotron-Nano-9B-v2, 108 SM A100: measured floor vs analytic-v1 ideal  (no contention, chunked-prefill −1)", x=0.01, ha="left", fontsize=11.5)
out = HERE / "prefill_observed_vs_ideal.png"; fig.savefig(out, dpi=135)
print("fit intercept ms", round(icpt*1e3, 2), "slope ms/tok", round(slope*1e3, 4))
for l, t, ti in zip(Lo, To, Tid_o): print(l, "obs ms", round(t*1e3, 1), "ideal ms", round(ti*1e3, 1), "ratio", round(t/ti, 2), "(T-9ms)/L us", round((t-0.009)/l*1e6, 1), "ideal us/tok", round(ti/l*1e6, 1))
sh = classes(4096); s = sum(sh.values()); print("ideal share @4k", {k: round(v/s, 3) for k, v in sh.items()}, "head ms", round(sh["head"]*1e3, 2))
print("implied uniform eta at slope:", round((sum(classes(16384).values())-sum(classes(1024).values()))/15360/slope, 3))
print(out)
