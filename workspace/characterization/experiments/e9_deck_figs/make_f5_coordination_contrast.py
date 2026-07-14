"""F5. coordination_contrast (decode-side)
같은 54 decode SM인데 조율 유무로 42ms(coordinated) vs 121ms(uncoordinated) —
"SM 부족이 아니라 조율 문제"라는 핵심 반박.
- coordinated   = agnostic pdmux 쌍(decode 54 · prefill 상보 54, 배타적 파티션)
                  results/r0c/_rows_agn_rep1_835303.csv rate2 = 42.5 ms [measured]
- uncoordinated = 독립 green-ctx 54/54 (pdmux prefill과 겹침 → 경합, pass C)
                  results/r0b/r0b_summary.csv g_54_54 rate2 = 121.8 ms [measured, job 835044]
(교차 확인: r_series_status.md §R0b — "42ms vs 121ms".)"""
import csv
import matplotlib.pyplot as plt
from _deck import EP, AGN, FUSED, tpot_at, save

COORD = tpot_at(f"{EP}/r0c/_rows_agn_rep1_835303.csv", 2)     # 42.51
UNCOORD = next(float(r[13]) for r in csv.reader(open(f"{EP}/r0b/r0b_summary.csv"))
               if len(r) > 13 and r[1] == "g_54_54" and r[7] == "2")  # 121.81

fig, ax = plt.subplots(figsize=(6.4, 5.0))
xs = [0, 1]
bars = ax.bar(xs, [COORD, UNCOORD], width=0.58, color=[AGN, FUSED], edgecolor="white", zorder=3)
ax.set_xticks(xs)
ax.set_xticklabels(["coordinated\n(agnostic pdmux pair)\ndecode 54 · prefill 54",
                    "uncoordinated\n(independent green-ctx)\ndecode 54, overlaps prefill"], fontsize=9.5)
for b, v in zip(bars, [COORD, UNCOORD]):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f} ms", ha="center", fontsize=12, fontweight="bold")
ax.annotate(f"same 54 SM\n{UNCOORD/COORD:.1f}× worse", xy=(1, UNCOORD * 0.5), xytext=(0.5, UNCOORD * 0.72),
            ha="center", fontsize=10.5, fontweight="bold", color=FUSED,
            arrowprops=dict(arrowstyle="-", color="0.7", lw=0))
ax.set_ylim(0, UNCOORD * 1.16); ax.set_ylabel("decode TPOT (ms)", fontsize=10)
ax.set_title("F5 · same 54 decode SM — it's coordination, not SM starvation\n"
             "coordinated 42 ms vs uncoordinated 121 ms (both measured)", fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3, zorder=0)
fig.tight_layout()
save(fig, "f5_coordination_contrast")
