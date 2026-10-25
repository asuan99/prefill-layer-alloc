#!/usr/bin/env python3
"""Hold-out check of the J1-calibrated prefill model at L=32k, 64k.  Calibration points: J1 FLOOR.json (L 1k-16k, audited scope).
Hold-out points: 128k-probe fused_sym(chunk -1) TTFT -- DESCRIPTIVE, citation-forbidden (RESULT_probe_128k), n=1 per run, 2 runs,
Workspace, out_tokens=64, mem-fraction 0.82 / 0.6.  Ideal = analytic-v1 [derived]."""
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]
exec((HERE / "scaling_batch_seqlen.py").read_text().split("a = T.nano9b_v2(); pre")[0])
E = ROOT / "workspace/engine-port/results"
J1 = E / ("multigpu/manual_retrieval/pdmux-multigpu-2106f42-20261018074950/results/multigpu/m3_pdmux-multigpu-2106f42-20261018074950/"
          "boots/m3_N1_fused_sym_boot1/FLOOR.json")
d = json.loads(J1.read_text()); Lo = np.array(sorted(int(k) for k in d["points"])); To = np.array([d["points"][str(l)]["floor_s"] for l in Lo])
P = E / "cap_char/probe_128k"; hold = {}
for run, tag in (("longctx_probe_20261020T022851Z", "run1 (mem 0.82)"), ("longctx_probe_20261020T034615Z", "run2 (mem 0.6)")):
    r = json.loads((P / run / "nano_fused/g3_single.json").read_text())
    hold[tag] = [(x["L"], x["ttft_s"]) for x in r["records"] if x["ok"] and x["L"] in (32768, 65536)]
a = T.nano9b_v2(); pre, dec, pm, _ = mk(a)
ideal = lambda L: pm(int(L)); core = lambda L: a.n_A*sum(max(f/PEAK, b/BW) for f, b in pre("A", int(L))[2:3])
slope, icpt = np.polyfit(Lo, To, 1); eta_inv = np.mean(To/np.array([ideal(l) for l in Lo]))   # uniform scale
Lg = np.linspace(1000, 70000, 140)
fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.4), constrained_layout=True)
ax[0].plot(Lg, [ideal(l)*1e3 for l in Lg], color="#0072B2", lw=1.8, label="ideal roofline (analytic-v1)")
ax[0].plot(Lg, [(icpt + eta_inv*ideal(l))*1e3 for l in Lg], color="#D55E00", lw=2, label=f"ideal × {eta_inv:.2f} + {icpt*1e3:.1f} ms (uniform η, J1 calibrated)")
ax[0].plot(Lg, (icpt + slope*Lg)*1e3, color="#777", lw=1.4, ls="--", label="J1 linear fit extrapolated")
ax[0].scatter(Lo, To*1e3, s=55, color="#111", zorder=5, label="J1 observed (calibration, audited scope)")
mk_ = {"run1 (mem 0.82)": "^", "run2 (mem 0.6)": "v"}
for tag, pts in hold.items():
    ax[0].scatter([p[0] for p in pts], [p[1]*1e3 for p in pts], s=90, marker=mk_[tag], facecolor="white", edgecolor="#111", linewidth=2, zorder=6,
                  label=f"probe hold-out {tag} — descriptive, uncited")
ax[0].set_xlabel("input length L (tokens)"); ax[0].set_ylabel("TTFT (ms)"); ax[0].legend(frameon=False, fontsize=8.5, loc="upper left"); ax[0].grid(alpha=.2)
ax[0].set_title("(a) Calibrated on 1k–16k; hold-out at 32k, 64k", loc="left", fontsize=10.5)
# (b) relative error of the two predictors on hold-out
rows = []
for tag, pts in hold.items():
    for L, t in pts: rows.append((L, tag, t, (icpt + eta_inv*ideal(L)), (icpt + slope*L)))
x = np.arange(len(rows)); w = .36
e1 = [(r[2]/r[3]-1)*100 for r in rows]; e2 = [(r[2]/r[4]-1)*100 for r in rows]
ax[1].bar(x - w/2, e1, w, color="#D55E00", label="vs uniform-η scaled ideal"); ax[1].bar(x + w/2, e2, w, color="#777", label="vs linear extrapolation")
for xi, v in zip(x, e1): ax[1].text(xi - w/2, v + .4, f"{v:+.0f}%", ha="center", fontsize=9)
for xi, v in zip(x, e2): ax[1].text(xi + w/2, v + .4, f"{v:+.0f}%", ha="center", fontsize=9)
ax[1].axhline(0, color="#333", lw=1); ax[1].set_xticks(x); ax[1].set_xticklabels([f"{r[0]//1024}k\n{r[1].split()[0]}" for r in rows])
ax[1].set_ylabel("observed ÷ predicted − 1  (%)"); ax[1].legend(frameon=False, fontsize=9); ax[1].grid(alpha=.2, axis="y")
ax[1].set_title("(b) Prediction error at hold-out points (positive = observed slower)", loc="left", fontsize=10.5)
fig.suptitle("Does the J1-calibrated prefill model extrapolate?  Nano-9B-v2, fused_sym chunk −1, A100", x=0.01, ha="left", fontsize=11.5)
out = HERE / "prefill_extrapolation.png"; fig.savefig(out, dpi=135)
print("eta_inv", round(eta_inv, 3), "icpt ms", round(icpt*1e3, 2), "slope", round(slope*1e3, 4))
for L, tag, t, p1, p2 in rows: print(L, tag, "obs", round(t, 3), "uniform-eta pred", round(p1, 3), f"({(t/p1-1)*100:+.1f}%)", "linear pred", round(p2, 3), f"({(t/p2-1)*100:+.1f}%)")
for L in (16384, 32768, 65536): print(L, "ideal s", round(ideal(L), 3), "ideal attn core s", round(core(L), 3), "core share", round(core(L)/ideal(L), 3))
print(out)
