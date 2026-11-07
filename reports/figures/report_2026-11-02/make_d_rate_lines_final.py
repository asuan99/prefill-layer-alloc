#!/usr/bin/env python3
"""Report figure.  Both panels use post-hoc predicates, applied to every arm in the panel: short input ITL p95 <= 80 ms (registered: 60),
long input registered predicate + largest token gap <= 1 s.  Values from stock_quick/sq_tables.json, sq_maxitl.json.  Descriptive."""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import style
c = style.apply("light")
plt.rcParams["font.family"] = "Noto Sans CJK JP"
SQ = os.path.abspath(os.path.join(HERE, "../../../workspace/engine-port/results/stock_quick"))
T = json.load(open(os.path.join(SQ, "sq_maxitl.json")))
A = json.load(open(os.path.join(SQ, "sq_tables.json")))
for arm, tab in T["W1"].items():
    for r, v in tab.items():
        v["itl80"] = A["W1"][arm][r]["attain"]["80"]
PANELS = [("W1", "itl80", "짧은 입력: ShareGPT, TTFT 3 s, ITL p95 80 ms(사후 문턱)", [5.5, 7.3, 9.6, 12.8, 19.4]),
          ("W3", "with_gap", "긴 입력: arXiv 16k, TTFT 2.8 s, ITL p95 60 ms, 최대 토큰 정지 1초 이내(사후 조건)", [0.10, 0.15, 0.23, 0.34, 0.51, 0.76])]
Ds, MARK = [16, 24, 34, 44, 64], ["o", "s", "^", "D", "v"]
def line(tab, key):
    pts = sorted((float(r), v[key]) for r, v in tab.items())
    return [p[0] for p in pts], [p[1] for p in pts]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), sharey=True)
for ax, (w, key, title, ticks) in zip(axes, PANELS):
    for i, d in enumerate(Ds):
        x, y = line(T[w][f"D{d}"], key)
        ax.plot(x, y, color=c["series"][i], marker=MARK[i], ms=5, lw=1.6, mec=c["surface"], mew=.8, alpha=.85, label=f"PD-mux D{d}", zorder=3 + i)
    x, y = line(T[w]["SG-stock"], key)
    ax.plot(x, y, color=c["ink"], marker="X", ms=8, lw=2.4, mec=c["surface"], mew=.8, label="SGLang 기본", zorder=10)
    ax.axhline(0.90, color=c["muted"], ls="--", lw=1, zorder=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("도착률 (req/s, 로그 축)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-0.03, 1.05)
axes[0].set_ylabel("SLO 달성 요청 비율")
axes[0].text(19.4, 0.915, "달성률 0.90", ha="right", fontsize=8.5, color=c["muted"])
axes[0].legend(frameon=False, loc="center right", fontsize=8.5, bbox_to_anchor=(0.62, 0.3), ncol=2)
fig.suptitle("SLO 달성률: SGLang 기본과 static PD-mux 분할", fontsize=12.5, x=0.01, ha="left")
fig.text(0.01, 0.012, "Nano-9B-v2, VESSL A100, seed 2개. 두 패널 모두 결과를 본 뒤 정한 기준을 모든 선에 똑같이 적용했다. 등록 기준(ITL p95 60 ms, 정지 조건 없음)에서는 짧은 입력의 SGLang이 5.5–7.3 req/s에서 0.16–0.24이고\n"
                     "(ITL p95 약 65 ms), 긴 입력의 SGLang은 static 분할과 같거나 높다. SGLang과 PD-mux는 다른 Job·GPU의 짧은 버스트 측정이고 설정도 달라 순서와 방향만 읽는다.", fontsize=8.5, color=c["muted"])
fig.tight_layout(rect=(0, 0.09, 1, 0.95))
fig.savefig(os.path.join(HERE, "fig_d_rate_lines_final.png"), dpi=170)
