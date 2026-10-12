#!/usr/bin/env python3
"""opchar 예상 도식(MOCKUP) — Bullet(ASPLOS'26) Fig 2 / Table 2 / Fig 3의 Hybrid LLM 판.

이 그림은 **측정 결과가 아니다**. 막대 길이·곡선 모양은 레이아웃을 보여 주기 위한
임의 값이며, 축 눈금 수치를 일부러 지웠다. 인용·분석 근거로 쓰지 말 것.
계획: ~/.claude/plans/figure2-table2-linear-cherny.md (2026-10-12 승인).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

OUT = Path(__file__).resolve().parent / "opchar_mockup_2026-10-12.png"


def _korean_font() -> None:
    for name in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = name
            return


def _watermark(ax, c) -> None:
    ax.text(0.5, 0.5, "MOCKUP", transform=ax.transAxes, ha="center", va="center",
            fontsize=42, color=c["muted"], alpha=0.13, rotation=20, zorder=0,
            fontweight="bold")


def panel_breakdown(ax, c) -> None:
    """(A-a) 연산자별 실행시간 비중 — L 축 4개, 모양만."""
    ops = ["in_proj", "conv1d", "SSD scan", "out_proj", "MLP", "qkv", "attn core",
           "o_proj", "norm/elem"]
    colors = [c["series"][0], c["series"][0], c["series"][1], c["series"][0],
              c["series"][2], c["series"][4], c["series"][3], c["series"][4], c["muted"]]
    hatches = ["", "..", "", "//", "", "", "", "//", ""]
    # 임의 모양: L이 길수록 attn core 비중이 커지는 '가설'만 표현
    shares = {
        "1k":  [14, 3, 16, 9, 40, 3, 4, 3, 8],
        "2k":  [13, 3, 16, 9, 38, 3, 7, 3, 8],
        "4k":  [12, 3, 15, 8, 35, 3, 13, 3, 8],
        "16k": [9, 2, 13, 6, 26, 2, 33, 2, 7],
    }
    ys = np.arange(len(shares))
    for y, (L, s) in zip(ys, shares.items()):
        s = np.array(s, float) / sum(s)
        left = 0.0
        for w, col, h, op in zip(s, colors, hatches, ops):
            ax.barh(y, w, left=left, color=col, hatch=h, edgecolor=c["surface"],
                    linewidth=1.2, height=0.62)
            if w > 0.07:
                ax.text(left + w / 2, y, op, ha="center", va="center", fontsize=7.5,
                        color="white")
            left += w
    ax.set_yticks(ys, [f"L={k}" for k in shares])
    ax.set_xticks([0, 1], ["0", "1"])
    ax.set_xlabel("정규화 실행시간 (실측: CUPTI 커널 트레이스 → 연산자 분류)")
    ax.set_title("A(a) 연산자별 시간 비중 — Nemotron-H prefill, 108 SM", loc="left", fontsize=10)
    ax.grid(False)
    from matplotlib.patches import Patch
    seen = {}
    for op, col, h in zip(ops, colors, hatches):
        seen.setdefault(op, Patch(facecolor=col, hatch=h, edgecolor=c["surface"], label=op))
    ax.legend(handles=list(seen.values()), frameon=False, fontsize=7, ncol=5,
              loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.text(0, -0.52, "가설(측정으로 확인할 점): L↑ → attn core 비중↑. attention 층이 적어 교차점이 "
            "Zamba2(≈2.75k)보다 뒤로 밀릴 수 있음", transform=ax.transAxes, fontsize=7.5,
            color=c["secondary"])
    _watermark(ax, c)


def panel_util(ax, c) -> None:
    """(A-b/c) 유도 util — 빗금 = derived, 수치 눈금 없음."""
    ops = ["in_proj", "conv", "scan", "out", "qkv", "attn", "o", "MLP", "norm"]
    rng = np.random.default_rng(0)
    x = np.arange(len(ops))
    width = 0.26
    for i, (L, col) in enumerate(zip(["1k", "2k", "16k"], c["series"][:3])):
        h = rng.uniform(0.15, 0.8, len(ops))
        ax.bar(x + (i - 1) * width, h, width, color=col, hatch="///",
               edgecolor=c["surface"], label=f"L={L}")
    ax.axhline(1.0, ls="--", color=c["secondary"], lw=1)
    ax.text(len(ops) - 0.5, 1.02, "천장 (compute: 312 TF / BW: memcpy 실측)", ha="right",
            fontsize=7.5, color=c["secondary"])
    ax.set_xticks(x, ops, fontsize=8)
    ax.set_yticks([0, 1], ["0", "천장"])
    ax.set_ylim(0, 1.15)
    ax.set_title("A(b)(c) compute / BW util — 유도값(분석 FLOP·byte ÷ 실측 시간)",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left")
    ax.text(0, -0.30, "VESSL은 ncu 카운터 차단(ERR_NVGPUCTRPERM) → Bullet식 하드웨어 실측 % 불가. "
            "빗금 = derived", transform=ax.transAxes, fontsize=7.5, color=c["critical"])
    _watermark(ax, c)


def panel_wave(ax, c) -> None:
    """(B) Table 2 대응 — 칸은 전부 '?'."""
    ax.axis("off")
    cols = ["in_proj", "conv1d", "SSD scan", "attn core", "o_proj", "MLP", "층 합"]
    rows = ["L=1k @108", "L=16k @108", "L=4k @44", "L=4k @16"]
    cell = [["?"] * len(cols) for _ in rows]
    tab = ax.table(cellText=cell, rowLabels=rows, colLabels=cols, loc="center",
                   cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1, 1.5)
    for (r, k), cl in tab.get_celld().items():
        cl.set_edgecolor(c["axis"])
        cl.set_facecolor(c["surface"])
        cl.get_text().set_color(c["ink"])
        if r in (3, 4):
            cl.set_facecolor(c["band"])
    ax.set_title("B wave quantization SM 유휴율 % (유도값: 실제 grid·blocks/SM → idle=(N−tail)/(N·w))",
                 loc="left", fontsize=10)
    ax.text(0.0, -0.08, "음영 행 = Bullet에 없는 축(분할 SM에서의 tail). SSD scan 등 SM 수를 읽는 "
            "커널은 분할 SM에서 식이 달라질 수 있음", transform=ax.transAxes, fontsize=7.5,
            color=c["secondary"])
    _watermark(ax, c)


def panel_scaling(ax, c, decode: bool) -> None:
    """(C) 108 SM 대비 speedup — 곡선 모양만."""
    r = np.linspace(16 / 108, 1, 60)
    ax.plot(r, r, ls=":", color=c["muted"], lw=1.2)
    ax.text(0.30, 0.20, "선형", color=c["muted"], fontsize=8)
    if not decode:
        curves = [("SSD scan (조기 포화?)", 1 - np.exp(-r / 0.18), c["series"][1]),
                  ("GEMM in_proj/MLP", r ** 0.85, c["series"][0]),
                  ("attn core", r ** 1.0 * 0.98, c["series"][3])]
        title = "C(a) prefill 연산자별 (L=4k 패널; 1k/16k/32k 반복)"
    else:
        curves = [("mamba step", 1 - np.exp(-r / 0.25), c["series"][1]),
                  ("attn decode", 1 - np.exp(-r / 0.4), c["series"][3]),
                  ("MLP (GEMV)", 1 - np.exp(-r / 0.3), c["series"][2])]
        title = "C(b) decode 층 종류별 1 step (ctx 2048; batch 32–128 선 반복)"
    for lab, y, col in curves:
        y = y / y[-1]
        ax.plot(r, y, color=col, lw=2, label=lab)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xticks([16 / 108, 1], ["16/108", "108/108"])
    ax.set_yticks([0, 1], ["0", "1 (=108 SM)"])
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("SM 비율 (green context, 4 SM 최소·2 SM 단위)")
    ax.set_ylabel("speedup")
    ax.set_title(title, loc="left", fontsize=10)
    _watermark(ax, c)


def main() -> None:
    c = style.apply("light")
    _korean_font()
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 0.8, 1.0], hspace=0.75, wspace=0.22)
    panel_breakdown(fig.add_subplot(gs[0, 0]), c)
    panel_util(fig.add_subplot(gs[0, 1]), c)
    panel_wave(fig.add_subplot(gs[1, :]), c)
    panel_scaling(fig.add_subplot(gs[2, 0]), c, decode=False)
    panel_scaling(fig.add_subplot(gs[2, 1]), c, decode=True)
    fig.suptitle("opchar 예상 도식 (MOCKUP — 측정 결과 아님, 모양·축 구성만) · "
                 "Bullet Fig 2 / Table 2 / Fig 3의 Hybrid LLM 판", fontsize=12, x=0.01,
                 ha="left", color=c["critical"])
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
