#!/usr/bin/env python
"""WHY dynamic loses to static — it is NOT overhead; it is a positioning failure.

The clean 'overhead > gain everywhere' story is already refuted (switch cost ~0: CONSENSUS
§1-8; controller CPU 0.014%: §1-12). So the loss must be a design flaw. This pins WHERE.

The failure, from the controller's own logs (bind, variance trace):
  1. The optimum is dec_sm=44 (decode-heavy; wins BOTH throughput and goodput).
  2. The controller HOVERS at dec_sm 16-24 (mean 22) and NEVER reaches 44 -- it is decode-STARVED.
     Why: it is REACTIVE (gives decode SM only after TPOT spikes) + SYMMETRIC (balances the two
     slacks), so the moment decode is briefly fine it pulls SM back to prefill. It structurally
     cannot accumulate to the decode-heavy optimum.
  3. Risk/reward: moving prefill-ward gains <=2.3% (LO phase is split-insensitive) but risks
     <=41.5% (HI mispositioning). Every switch is an 18:1 bad bet -- and it is NOT the switch
     COST, it is the POSITION the switch moves to.
  4. Every fix (anchor decode-heavy / asymmetric penalty / stop moving) converges the controller
     toward 'sit at 44' = the best static. The gate (a ratchet that stops moving) is the best
     dynamic (settles at 34) but undershoots. There is no regime where moving beats sitting at 44
     (§1-13: LO split-insensitive, HI wants 44 which is already the static choice).

Data: results/slo_sched (bind trajectories jobs 856963/4), per-phase spreads from §1-13.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# controller dec_sm trajectories (from SLO-BIND log lines)
TRAJ = {
    "clean (goodput 3.10)":     [16, 24, 16, 24, 34, 24, 16, 24],
    "collapsed (goodput 2.41)": [34, 24, 16, 24, 16, 24, 16, 24, 16, 24],
}
OPT = 44  # dec_sm of d44, optimal on both throughput and goodput

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4))
fig.suptitle("WHY dynamic loses to static — a POSITIONING failure, not an overhead failure\n"
             "(switch cost ≈0 §1-8; controller CPU 0.014% §1-12) — the controller sits at the wrong split, it doesn't pay to move there",
             fontsize=12, fontweight="bold")

# ── panel 1: trajectory never reaches the optimum ──
a = ax[0]
for (lab, seq), c in zip(TRAJ.items(), ["#0077b6", "#c1121f"]):
    a.plot(range(len(seq)), seq, "o-", color=c, lw=2, ms=8, label=lab)
a.axhline(OPT, color="#1a7a3a", lw=2.5, ls="--")
a.text(0.1, OPT+0.8, "OPTIMUM  dec_sm = 44 (d44)  — wins BOTH throughput & goodput", fontsize=9, color="#1a7a3a", fontweight="bold")
a.axhspan(14, 26, color="#c1121f", alpha=.10)
a.text(4.5, 19.5, "controller HOVERS here\n(dec_sm 16–24, mean 22)\n= decode-STARVED", fontsize=9.5, color="#c1121f", fontweight="bold", ha="center")
a.annotate("", xy=(7.3, 44), xytext=(7.3, 24), arrowprops=dict(arrowstyle="<->", color="gray", lw=1.6))
a.text(7.45, 34, "never\ncloses\nthis gap", fontsize=8.5, color="gray", fontweight="bold")
a.set_ylim(12, 50); a.set_xlabel("switch # (controller decision sequence)")
a.set_ylabel("dec_sm the controller chose")
a.set_title("① it never reaches the optimum\nreactive + symmetric ⇒ pulls back to prefill whenever TPOT recovers", fontsize=10.3, fontweight="bold")
a.grid(alpha=.3); a.legend(fontsize=8.5, loc="center right")

# ── panel 2: the 18:1 bad bet ──
a = ax[1]
gain = 2.3; loss = 41.5
bars = a.bar(["max GAIN\nfrom moving\n(LO phase)", "max LOSS\nfrom mispositioning\n(HI phase)"],
             [gain, loss], color=["#2a9d8f", "#c1121f"], alpha=.9, width=.55)
for b, v in zip(bars, [gain, loss]):
    a.text(b.get_x()+b.get_width()/2, v+0.8, f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")
a.annotate("", xy=(1, 41.5), xytext=(0, 2.3), arrowprops=dict(arrowstyle="->", color="k", lw=1.6, ls=":"))
a.text(0.5, 26, "moving is an\n18 : 1\nBAD BET", fontsize=13, color="#c1121f", fontweight="bold", ha="center",
       bbox=dict(boxstyle="round", fc="white", ec="#c1121f", lw=1.5))
a.set_ylabel("goodput change (% of base)")
a.set_title("② WHY moving doesn't pay (asymmetry, §1-6/13)\nLO is split-insensitive; HI mispositioning is catastrophic", fontsize=10.3, fontweight="bold")
a.set_ylim(0, 48); a.grid(alpha=.3, axis="y")

# ── panel 3: every fix → static ──
a = ax[2]
# less movement / more anchored -> closer to optimum
variants = ["bind\n(oscillates,\nmean sm 22)", "bind+GATE\n(ratchets,\nstops at 34)", "d44 STATIC\n(never moves,\nsm 44)"]
sit = [22, 34, 44]         # where it sits
gpv = [2.934, 3.132, 3.220]
cols = ["#c1121f", "#e9a13b", "#1a7a3a"]
ax2 = a
b = ax2.bar(variants, gpv, color=cols, alpha=.9, width=.6)
for bi, g, s in zip(b, gpv, sit):
    ax2.text(bi.get_x()+bi.get_width()/2, g+0.01, f"gp {g:.3f}", ha="center", fontsize=9.5, fontweight="bold")
    ax2.text(bi.get_x()+bi.get_width()/2, 2.72, f"sits@{s}", ha="center", fontsize=8.5, color="white", fontweight="bold")
ax2.axhline(3.220, color="#1a7a3a", ls="--", lw=1.3)
ax2.annotate("the less it moves,\nthe closer to optimum\n⇒ the 'fix' is to become STATIC",
             (1, 3.05), fontsize=9, color="#555", fontweight="bold", ha="center")
ax2.set_ylim(2.7, 3.28); ax2.set_ylabel("goodput (req/s)")
ax2.set_title("③ every fix converges to static\ndynamic can at best TIE (by not moving), never beat", fontsize=10.3, fontweight="bold")
ax2.grid(alpha=.3, axis="y")

fig.tight_layout(rect=[0, 0.03, 1, 0.9])
fig.text(0.5, 0.008,
         "The loss is not the cost of switching — it is that the reactive/symmetric controller sits at a decode-starved split (mean 22) instead of the decode-heavy optimum (44), "
         "and no switch it can make pays off (LO gain 2.3% ≪ HI risk 41.5%). Fixing the design = anchoring at 44 = static. There is no moving optimum to justify dynamism (§1-13).",
         ha="center", fontsize=8.6, style="italic")
fig.savefig("why_dynamic_loses.png", dpi=150)
print("wrote why_dynamic_loses.png")
