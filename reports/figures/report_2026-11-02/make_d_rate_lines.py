#!/usr/bin/env python3
"""Line view of the same points as make_d_rate_heatmap.py: SLO attainment vs arrival rate, one line per static D.
Descriptive. Different Jobs/GPUs: read D order inside each panel only."""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import style
c = style.apply("light")
plt.rcParams["font.family"] = "Noto Sans CJK JP"
R = os.path.abspath(os.path.join(HERE, "../../../workspace/engine-port/results"))
PANELS = [("policy_cap1r/cap1r_analysis.json", "짧은 입력: 채팅형 ShareGPT (입력 q90 ≈ 686 토큰), 셀당 240요청", [5.5, 7.3, 9.6, 12.8, 19.4]),
          ("policy_cap2/cap2_analysis.json", "긴 입력: arXiv 요약 16k (입력 q90 ≈ 12.5k 토큰), 셀당 96요청", [0.10, 0.15, 0.23, 0.34, 0.51, 0.76])]
Ds = [16, 24, 34, 44, 64]
MARK = ["o", "s", "^", "D", "v"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, (f, title, ticks) in zip(axes, PANELS):
    cells = list(json.load(open(os.path.join(R, f)))["cells"].values())
    for i, d in enumerate(Ds):
        pts = sorted((x["rate"], x["attain"]) for x in cells if x["D"] == d)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c["series"][i], marker=MARK[i], ms=6, lw=2,
                mec=c["surface"], mew=1, label=f"D{d}", zorder=3 + i)
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
axes[1].legend(frameon=False, loc="lower left", title="decode SM 수", fontsize=9, title_fontsize=9)
fig.suptitle("static 분할의 D별 SLO 달성률: 짧은 입력과 긴 입력에서 유리한 D가 달라진다", fontsize=12.5, x=0.01, ha="left")
fig.text(0.01, 0.012, "Nano-9B-v2, VESSL A100, seed 2개, 전환 0회. 짧은 입력 SLO: TTFT 3 s, 긴 입력 SLO: TTFT 2.8 s, 둘 다 ITL p95 60 ms. 짧은 입력에서 D34·D44·D64 선은 거의 겹친다.\n"
                     "두 패널은 다른 Job·GPU의 짧은 버스트 측정이라 패널 사이 크기를 비교하지 않고, 각 패널 안의 D 순서만 읽는다.", fontsize=8.5, color=c["muted"])
fig.tight_layout(rect=(0, 0.09, 1, 0.95))
fig.savefig(os.path.join(HERE, "fig_d_rate_lines.png"), dpi=170)
