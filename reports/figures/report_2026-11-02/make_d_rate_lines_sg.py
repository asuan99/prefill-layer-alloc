#!/usr/bin/env python3
"""SLO attainment vs arrival rate: existing static PD-mux D lines (CAP1-R / CAP2) + default SGLang (stock_quick Job).
All from stock_quick/sq_tables.json.  Descriptive: different Jobs/GPUs, 2 seeds."""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import style
c = style.apply("light")
plt.rcParams["font.family"] = "Noto Sans CJK JP"
T = json.load(open(os.path.abspath(os.path.join(HERE, "../../../workspace/engine-port/results/stock_quick/sq_tables.json"))))
PANELS = [("W1", "짧은 입력: 채팅형 ShareGPT, 셀당 240요청", [5.5, 7.3, 9.6, 12.8, 19.4]),
          ("W3", "긴 입력: arXiv 요약 16k, 셀당 96요청", [0.10, 0.15, 0.23, 0.34, 0.51, 0.76])]
Ds = [16, 24, 34, 44, 64]
MARK = ["o", "s", "^", "D", "v"]
def line(tab, itl="60"):
    pts = sorted((float(r), v["attain"][itl]) for r, v in tab.items())
    return [p[0] for p in pts], [p[1] for p in pts]
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, (w, title, ticks) in zip(axes, PANELS):
    for i, d in enumerate(Ds):
        x, y = line(T[w][f"D{d}"])
        ax.plot(x, y, color=c["series"][i], marker=MARK[i], ms=5, lw=1.6, mec=c["surface"], mew=.8, alpha=.85, label=f"PD-mux D{d}", zorder=3 + i)
    x, y = line(T[w]["SG-stock"])
    ax.plot(x, y, color=c["ink"], marker="X", ms=8, lw=2.4, mec=c["surface"], mew=.8, label="SGLang 기본", zorder=10)
    if w == "W1":
        x, y = line(T[w]["SG-stock"], "80")
        ax.plot(x, y, color=c["ink"], marker="X", ms=6, lw=1.4, ls=":", label="SGLang 기본 (사후 문턱 ITL 80 ms)", zorder=9)
    ax.axhline(0.90, color=c["muted"], ls="--", lw=1, zorder=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("도착률 (req/s, 로그 축)")
    ax.set_title(title, fontsize=10.5)
    ax.set_ylim(-0.03, 1.05)
axes[0].set_ylabel("SLO 달성 요청 비율")
axes[0].text(19.4, 0.915, "달성률 0.90", ha="right", fontsize=8.5, color=c["muted"])
h, l = axes[0].get_legend_handles_labels()
axes[1].legend(h, l, frameon=False, loc="lower left", fontsize=8.5)
fig.suptitle("SLO 달성률: SGLang 기본과 static PD-mux 분할", fontsize=12.5, x=0.01, ha="left")
fig.text(0.01, 0.012, "Nano-9B-v2, VESSL A100, seed 2개. SLO: 짧은 입력 TTFT 3 s, 긴 입력 TTFT 2.8 s, 실선은 모두 요청 내부 ITL p95 60 ms. 짧은 입력의 SGLang은 ITL p95가 약 65 ms로 문턱 바로 위에 몰려 값이 문턱에 민감하다.\n"
                     "SGLang과 PD-mux는 다른 Job·GPU의 짧은 버스트 측정이고 설정도 다르다(PD-mux: mem 0.6, overlap OFF, chunked prefill 끔). 순서와 방향만 읽는다.\n"
                     "긴 입력의 p95는 토큰 정지를 숨긴다: SGLang은 요청당 최대 정지 중앙 0.6–5 s, PD-mux는 100 ms 초과 정지 없음.", fontsize=8.5, color=c["muted"])
fig.tight_layout(rect=(0, 0.12, 1, 0.95))
fig.savefig(os.path.join(HERE, "fig_d_rate_lines_sglang.png"), dpi=170)
