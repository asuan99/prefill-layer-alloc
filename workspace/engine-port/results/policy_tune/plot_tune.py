import json, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1])); cfgs = sorted(d["configs"], key=lambda c: -d["configs"][c]["mean_goodput"])
fig, ax = plt.subplots(1, 3, figsize=(14, 4)); x = range(len(cfgs))
for s, off, col in ((101, -0.2, "#4c78a8"), (102, 0.2, "#f58518")):
    ax[0].bar([i + off for i in x], [d["configs"][c]["per_seed"][str(s)] for c in cfgs], 0.4, label=f"seed {s}", color=col)
top = max(d["configs"][c]["mean_goodput"] for c in cfgs); ax[0].axhline(top * 0.99, ls="--", c="gray", lw=1)
ax[0].set_ylim(1.9, 2.3); ax[0].set_xticks(list(x)); ax[0].set_xticklabels(cfgs, rotation=30, fontsize=7); ax[0].set_ylabel("TRUE goodput (req/s)"); ax[0].legend(fontsize=7)
ax[0].set_title("TUNE goodput; dashed = 1% tie band (descriptive)", fontsize=9)
ax[1].bar([i - 0.2 for i in x], [d["configs"][c]["mean_LO"] for c in cfgs], 0.4, label="LO"); ax[1].bar([i + 0.2 for i in x], [d["configs"][c]["mean_HI"] for c in cfgs], 0.4, label="HI")
ax[1].set_xticks(list(x)); ax[1].set_xticklabels(cfgs, rotation=30, fontsize=7); ax[1].set_title("phase goodput (mean of 2 seeds)", fontsize=9); ax[1].legend(fontsize=7)
for s, off in ((101, -0.2), (102, 0.2)):
    ax[2].bar([i + off for i in x], [next(v["switch_count"] for v in d["runs"].values() if v["cfg"] == c and v["seed"] == s) for c in cfgs], 0.4, label=f"seed {s}")
ax[2].set_xticks(list(x)); ax[2].set_xticklabels(cfgs, rotation=30, fontsize=7); ax[2].set_title("switch_count (controller decision changes)", fontsize=9); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=130)
