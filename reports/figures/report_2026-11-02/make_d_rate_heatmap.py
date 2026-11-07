#!/usr/bin/env python3
"""All measured static D x rate points, short (W1 ShareGPT, CAP1-R) vs long (W3 arXiv 16k, CAP2) input.
Descriptive. Different Jobs/GPUs: read each panel on its own; do not compare magnitudes across panels."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import style
c = style.apply("light")
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.grid"] = False
R = os.path.abspath(os.path.join(HERE, "../../../workspace/engine-port/results"))
PANELS = [("policy_cap1r/cap1r_analysis.json", "짧은 입력: 채팅형 ShareGPT (입력 q90 ≈ 686 토큰)", "셀당 240요청"),
          ("policy_cap2/cap2_analysis.json", "긴 입력: arXiv 요약 16k (입력 q90 ≈ 12.5k 토큰)", "셀당 96요청")]
cmap = LinearSegmentedColormap.from_list("att", ["#f4f3ee", "#a9c9ee", c["series"][0]])
Ds = [16, 24, 34, 44, 64]
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw=dict(width_ratios=[1, 1]))
for ax, (f, title, note) in zip(axes, PANELS):
    cells = json.load(open(os.path.join(R, f)))["cells"].values()
    rates = sorted({x["rate"] for x in cells})
    M = np.full((len(rates), len(Ds)), np.nan)
    for x in cells:
        M[rates.index(x["rate"]), Ds.index(x["D"])] = x["attain"]
    assert not np.isnan(M).any(), f
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, origin="lower", aspect="auto")
    for i in range(len(rates)):
        for j in range(len(Ds)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v >= 0.75 else c["ink"], fontweight="bold" if v >= 0.90 else "normal")
            if v >= 0.90:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=c["ink"], lw=1.6))
    ax.set_xticks(range(len(Ds)), [f"D{d}" for d in Ds])
    ax.set_yticks(range(len(rates)), [f"{r:.2f}" for r in rates])
    ax.set_xlabel("decode에 준 SM 수 D (전체 108)")
    ax.set_ylabel("도착률 (req/s)")
    ax.set_title(f"{title}\n{note}", fontsize=10.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
fig.suptitle("static 분할의 측정 지점 전체: 칸 안 숫자 = SLO 달성 요청 비율", fontsize=12.5, x=0.01, ha="left")
fig.text(0.01, 0.012, "굵은 테두리 = 달성률 0.90 이상(테두리 없는 0.90은 0.896). Nano-9B-v2, VESSL A100, seed 2개, 전환 0회. 짧은 입력 SLO: TTFT 3 s, 긴 입력 SLO: TTFT 2.8 s, 둘 다 ITL p95 60 ms.\n"
                     "두 패널은 다른 Job·GPU의 짧은 버스트 측정이라 패널 사이 크기를 비교하지 않고, 각 패널 안의 D 순서만 읽는다.", fontsize=8.5, color=c["muted"])
fig.tight_layout(rect=(0, 0.08, 1, 0.95))
fig.savefig(os.path.join(HERE, "fig_d_rate_heatmap.png"), dpi=170)
