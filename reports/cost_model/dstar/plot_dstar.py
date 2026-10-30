#!/usr/bin/env python3
"""Figure for PROPOSAL_dstar_model_2026-10-30.md: u(D), and model lam_cap(D) vs measured capacity brackets (W1, W3).

Reads dstar_result.json (model output). Measured brackets are copied from RESULT_cap1_2026-10-28.md / RESULT_cap1r_2026-10-29.md
(W1, burst 120-request cells) and RESULT_cap2_sustain_2026-10-29.md (W3); they are descriptive values, not verdicts.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
D_GRID = [16, 24, 34, 44, 64]
# (last pass, first fail) req/s; None = not bracketed on that side. W1: tightest of CAP1 / CAP1-R.
W1_MEAS = {16: (1.661, 2.242), 24: (4.086, 5.516), 34: (6.3434, 7.29491), 44: (7.29491, 8.389146), 64: (7.29491, 8.389146)}
W3_MEAS = {16: (0.3375, 0.50625), 24: (0.225, 0.3375), 34: (0.225, 0.3375), 44: (0.15, 0.225), 64: (0.0, 0.1)}
TAG = {"ITL": "I", "SLOT": "S", "TTFT": "T"}


def style(ax, title, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_xlabel("decode SM D (static, co-located target)", color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def cap_panel(ax, w, meas, title, legend_loc="upper left"):
    xs = D_GRID
    mid = [w["lam_cap_rho_mid"][str(d)] for d in xs]
    lo = [w["per_D"][str(d)]["rho"]["1.19"]["lam_cap"] for d in xs]
    hi = [w["per_D"][str(d)]["rho"]["1.13"]["lam_cap"] for d in xs]
    off = 1.6
    for i, d in enumerate(xs):
        a, b = meas[d]
        ax.plot([d + off, d + off], [a, b], color=ORANGE, linewidth=2, solid_capstyle="round")
        ax.plot([d + off], [a], marker="o", markersize=8, color=ORANGE, markeredgecolor=SURF, markeredgewidth=2)
        ax.plot([d - off, d - off], [lo[i], hi[i]], color=BLUE, linewidth=2)
        ax.plot([d - off], [mid[i]], marker="s", markersize=8, color=BLUE, markeredgecolor=SURF, markeredgewidth=2)
        bind = w["per_D"][str(d)]["rho"]["1.16"]["binding"]
        ax.annotate("".join(TAG.get(k, "?") for k in bind), (d - off, mid[i]), textcoords="offset points",
                    xytext=(-12, 6), color=MUTED, fontsize=8)
    ax.set_xticks(xs)
    ax.plot([], [], marker="s", color=BLUE, linewidth=2, label="model lam_cap (rho 1.13-1.19), steady state")
    ax.plot([], [], marker="o", color=ORANGE, linewidth=2, label="measured: last pass -> first fail")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc=legend_loc)


def main():
    r = json.load(open(os.path.join(HERE, "dstar_result.json")))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURF)
    hw = r["hardware"]
    ax = axes[0]
    style(ax, "(a) effective HBM fraction u(D) = a0 / a(D)", "u (fraction of 2.039 TB/s)")
    ds = [row["D"] for row in hw["rows"]]
    us = [row["u"] for row in hw["rows"]]
    f = hw["fit"]
    fx = [0, f["D_knee"], 108]
    fy = [0, f["beta_per_sm"] * f["D_knee"], f["beta_per_sm"] * f["D_knee"] + f["gamma_per_sm"] * (108 - f["D_knee"])]
    ax.plot(fx, fy, color=MUTED, linewidth=1.2, linestyle="--")
    ax.plot(ds, us, marker="o", markersize=8, color=BLUE, linewidth=0, markeredgecolor=SURF, markeredgewidth=2)
    ax.annotate(f"knee ~ D{f['D_knee']:.0f}\n{f['per_sm_bw_GBps_below_knee']:.1f} GB/s per SM below,\n"
                f"slope x{f['slope_ratio']:.2f} above", (f["D_knee"], fy[1]), textcoords="offset points",
                xytext=(8, -42), color=MUTED, fontsize=8)
    ax.set_xticks([16, 24, 34, 44, 64, 108])
    ax.set_ylim(0, 0.7)
    ax.set_xlim(0, 112)
    style(axes[1], "(b) W1 ShareGPT: capacity vs D", "req/s")
    cap_panel(axes[1], r["workloads"]["W1"], W1_MEAS, "")
    style(axes[2], "(c) W3 arXiv-Sum 16k: capacity vs D", "req/s")
    cap_panel(axes[2], r["workloads"]["W3"], W3_MEAS, "", legend_loc="upper right")
    axes[2].set_ylim(0, 0.68)
    fig.text(0.01, 0.005, "Labels above model marks = binding constraint (I = ITL, S = slot cap, T = TTFT). "
             "W1 measured cells are 120-request bursts\n(burst > steady state; sustained D44/D64 completion ~3.0 req/s). "
             "Post-hoc comparison: the model was written after these capacities were seen.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = os.path.join(HERE, "dstar_capacity.png")
    fig.savefig(out, dpi=150, facecolor=SURF)
    print(out)


if __name__ == "__main__":
    main()
