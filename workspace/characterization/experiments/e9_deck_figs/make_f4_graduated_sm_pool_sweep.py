"""F4. graduated_sm_pool_sweep (decode-side)
graduated(R0b) per-type (mamba SM / attn SM) 조합 스윕 — 전부 agnostic 기준선에 못 미침.
값 전부 measured: results/r0b/r0b_summary.csv (jobs 835044-835209), @in3600/o32 rate2.
agnostic 기준: results/r0c/_rows_agn_rep1_835303.csv (42.5ms, goodput 정상)."""
import matplotlib.pyplot as plt
from _deck import EP, AGN, LA, FUSED, read_rows, tpot_at, save

SUM = f"{EP}/r0b/r0b_summary.csv"
AGN_TPOT = tpot_at(f"{EP}/r0c/_rows_agn_rep1_835303.csv", 2)   # 42.51

def r0b(label):  # r0b_summary rate2 tpot + goodput
    import csv
    for row in csv.reader(open(SUM)):
        if len(row) > 16 and row[1] == label and row[7] == "2":
            return float(row[13]), int(float(row[15]))   # tpot50, good_count
    raise KeyError(label)

combos = [("96 / 96", "g_96_96"), ("84 / 84", "g_84_84"), ("72 / 72", "g_72_72"),
          ("54 / 84", "g_54_84"), ("54 / 16", "g_54_16")]   # (mamba SM / attn SM)
data = [(lab, *r0b(key)) for lab, key in combos]
data.sort(key=lambda t: t[1])   # tpot 오름차순 → 나은 것 위로 (barh y=0 바닥)

fig, ax = plt.subplots(figsize=(9.5, 4.6))
ys = range(len(data)); vals = [d[1] for d in data]
cols = [AGN if v < AGN_TPOT * 1.15 else FUSED for v in vals]  # 붕괴(starved)=red
ax.barh(list(ys), vals, color=cols, edgecolor="white", height=0.6, zorder=3)
ax.set_yticks(list(ys)); ax.set_yticklabels([f"{d[0]}\n(mamba/attn SM)" for d in data], fontsize=9)
ax.axvline(AGN_TPOT, color=LA, lw=2.6, zorder=4)
ax.text(AGN_TPOT + 2, len(data) - 0.4, f"agnostic {AGN_TPOT:.0f} ms\n(~96 SM, dynamic)",
        color=LA, fontsize=9.5, fontweight="bold", va="top")
for i, (lab, tp, gc) in enumerate(data):
    note = "goodput 0 (starved)" if gc == 0 else f"goodput ok"
    ax.text(tp + 2, i, f"{tp:.0f} ms · {note}", va="center", fontsize=9, fontweight="bold")
ax.set_xlim(0, max(vals) * 1.22)
ax.set_xlabel("decode TPOT p50 (ms)  —  in3600/o32, rate 2  [lower = better]", fontsize=10)
ax.set_title("F4 · graduated per-type sweep: every (mamba/attn SM) combo loses to agnostic\n"
             "≥84 SM merely matches (44-45 ms); <84 SM collapses (goodput 0) — all measured (R0b)",
             fontsize=11, fontweight="bold")
ax.grid(axis="x", alpha=0.3, zorder=0)
fig.tight_layout()
save(fig, "f4_graduated_sm_pool_sweep")
