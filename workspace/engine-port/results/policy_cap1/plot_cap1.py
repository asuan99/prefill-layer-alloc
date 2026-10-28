import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1])); out = sys.argv[2]
per = d["addendum"]["per_D"]; Ds = sorted(per, key=int)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i, D in enumerate(Ds):
    v = per[D]; cap = v["capacity"]; ff = v["first_fail"]
    hi = (ff - cap) if ff else 0
    ax[0].bar(i, cap, yerr=[[0], [hi]], capsize=4, color="#4c78a8")
    if ff is None: ax[0].annotate("not bracketed\n(top rung passes)", (i, cap), ha="center", va="bottom", fontsize=7)
ax[0].set_xticks(range(len(Ds))); ax[0].set_xticklabels([f"D{D}" for D in Ds])
ax[0].set_ylabel("last passing rate (req/s); bar top = lambda*_D, whisker = bracket to first fail")
ax[0].set_title("CAP1 W1 static capacity (descriptive; N=2 seeds x 120 req)", fontsize=9)
for D in Ds:
    a = per[D]["attainment"]; xs = sorted(a, key=float)
    ax[1].plot([float(x) for x in xs], [a[x] for x in xs], "o-", label=f"D{D}", ms=3)
ax[1].axhline(0.9, ls="--", c="gray"); ax[1].set_xscale("log"); ax[1].set_xlabel("offered rate (req/s)"); ax[1].set_ylabel("goodput attainment (TTFT<=3000ms & ITL p95<=60ms)")
ax[1].legend(fontsize=7); ax[1].set_title("attainment vs rate; dashed = 0.90 capacity threshold", fontsize=9)
plt.tight_layout(); plt.savefig(out, dpi=130)
