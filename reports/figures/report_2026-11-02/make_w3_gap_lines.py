#!/usr/bin/env python3
"""Long input (W3): SLO attainment with the registered predicate vs with a post-hoc 'largest token gap <= 1 s' added.
Values from stock_quick/sq_maxitl.json.  Descriptive: different Jobs/GPUs, 2 seeds."""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import style
c = style.apply("light")
plt.rcParams["font.family"] = "Noto Sans CJK JP"
T = json.load(open(os.path.abspath(os.path.join(HERE, "../../../workspace/engine-port/results/stock_quick/sq_maxitl.json"))))["W3"]
Ds, MARK = [16, 24, 34, 44, 64], ["o", "s", "^", "D", "v"]
ticks = [0.10, 0.15, 0.23, 0.34, 0.51, 0.76]
def line(tab, key):
    pts = sorted((float(r), v[key]) for r, v in tab.items())
    return [p[0] for p in pts], [p[1] for p in pts]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), sharey=True)
for ax, key, title in [(axes[0], "base", "등록 기준: TTFT 2.8 s, 요청 내부 ITL p95 60 ms"), (axes[1], "with_gap", "위 기준 + 최대 토큰 정지 1초 이내 (사후 조건)")]:
    for i, d in enumerate(Ds):
        x, y = line(T[f"D{d}"], key)
        ax.plot(x, y, color=c["series"][i], marker=MARK[i], ms=5, lw=1.6, mec=c["surface"], mew=.8, alpha=.85, label=f"PD-mux D{d}", zorder=3 + i)
    x, y = line(T["SG-stock"], key)
    ax.plot(x, y, color=c["ink"], marker="X", ms=8, lw=2.4, mec=c["surface"], mew=.8, label="SGLang 기본", zorder=10)
    ax.axhline(0.90, color=c["muted"], ls="--", lw=1, zorder=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("도착률 (req/s, 로그 축)")
    ax.set_title(title, fontsize=10.5)
    ax.set_ylim(-0.03, 1.05)
axes[0].set_ylabel("SLO 달성 요청 비율")
axes[1].text(0.76, 0.915, "달성률 0.90", ha="right", fontsize=8.5, color=c["muted"])
axes[0].legend(frameon=False, loc="lower left", fontsize=8.5)
fig.suptitle("긴 입력(arXiv 16k): 최대 토큰 정지를 함께 보면 SGLang 기본의 순서가 뒤집힌다", fontsize=12.5, x=0.01, ha="left")
fig.text(0.01, 0.012, "Nano-9B-v2, VESSL A100, seed 2개, 셀당 96요청. PD-mux 선은 두 패널에서 같다(최대 정지 131 ms 이하). 1초는 결과를 본 뒤 고른 문턱이며, 2초로 올리면 0.10–0.23 req/s에서 SGLang이 0.93–0.97이 된다.\n"
                     "SGLang과 PD-mux는 다른 Job·GPU의 짧은 버스트 측정이고 설정도 다르다(PD-mux: mem 0.6, overlap OFF, chunked prefill 끔). 순서와 방향만 읽는다.", fontsize=8.5, color=c["muted"])
fig.tight_layout(rect=(0, 0.09, 1, 0.95))
fig.savefig(os.path.join(HERE, "fig_w3_gap_lines.png"), dpi=170)
