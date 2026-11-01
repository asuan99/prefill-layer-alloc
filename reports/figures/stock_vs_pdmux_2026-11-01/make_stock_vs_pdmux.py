#!/usr/bin/env python3
"""Stock serving (fused SGLang) vs static PD-mux — what has actually been measured.

Descriptive only: every number is transcribed from
  workspace/engine-port/results/stock_baseline/STOCK_METRICS_TABLE.csv (Phase A, 2026-11-01)
and no new verdict is produced.  Figure 01 (KISTI P1-opint, n=5) already exists at
reports/figures/current_2026-10-21/01_claimG_p1opint.png and is not redrawn here.

  fig_a  VESSL · Nano-9B-v2 · M3 J1 LO cells (N=1 Job, seed 2) — fused / fused_sym / agnostic / d44
  fig_b  evidence map: which baseline x which substrate has serving data, and at what grade
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import style  # noqa: E402

CSV = HERE.parents[2] / "workspace/engine-port/results/stock_baseline/STOCK_METRICS_TABLE.csv"
OUT = HERE


def load_m3():
    rows = {}
    with open(CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] != "m3_j1_N1" or r["round"] != "pooled":
                continue
            if r["rate_or_rung"] not in ("LO", "LOx"):
                continue
            shape = r["workload"].split(" ")[1]  # A / B / X
            if shape == "X":
                continue  # X_CORRECTNESS_FAIL — ratios undefined, not drawn
            rows[(shape, r["arm"])] = dict(
                req=float(r["req_per_s"]),
                ttft95=float(r["ttft_p95_s"]) * 1e3,
                itl95=float(r["itl_req_p95_median_s"]) * 1e3,
                tpot50=float(r["tpot_p50_s"]) * 1e3,
            )
    return rows


ARMS = [("fused", "stock SGLang (chunk 8192, overlap ON)"),
        ("fused_sym", "stock SGLang (chunk −1, overlap OFF)"),
        ("agnostic", "PD-mux agnostic"),
        ("d44", "PD-mux static D44")]


def fig_a(c, rows):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    shapes = [("A", "shape A  in256 / out512  (LO)"), ("B", "shape B  in8192 / out64  (LOx)")]
    metrics = [("ttft95", "TTFT p95 (ms)", "log"), ("itl95", "요청 내부 ITL p95 중앙 (ms)", "linear"),
               ("req", "req/s", "linear")]
    w = 0.19
    cols = [c["series"][1], c["series"][3], c["series"][0], c["series"][2]]
    for ax, (key, yl, scale) in zip(axes, metrics):
        x = np.arange(len(shapes))
        for j, ((arm, lab), col) in enumerate(zip(ARMS, cols)):
            vals = [rows[(s, arm)][key] for s, _ in shapes]
            bars = ax.bar(x + (j - 1.5) * w, vals, w * 0.92, color=col, label=lab)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}" if v >= 10 else f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6.8, color=c["ink"])
        ax.set_xticks(x, [s for _, s in shapes], fontsize=8)
        ax.set_ylabel(yl)
        if scale == "log":
            ax.set_yscale("log"); style.no_minor_labels(ax, "y")
        if key == "itl95":
            ax.axhline(60, ls=":", color=c["critical"], lw=1)
            ax.text(1.42, 60, " ITL SLO 60 ms", va="bottom", ha="right", fontsize=7, color=c["critical"])
        if key == "ttft95":
            ax.axhline(3000, ls=":", color=c["critical"], lw=1)
            ax.text(1.42, 3000, " TTFT SLO 3 s", va="bottom", ha="right", fontsize=7, color=c["critical"])
        style.grade_chip(ax, "N=1 Job · seed 2 · 기술값", c, "warning")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("VESSL A100 · Nemotron-Nano-9B-v2 · stock SGLang(fused) 대 PD-mux — M3 J1 LO 셀 (사다리 아래, 용량 아래)",
                 fontsize=11.5, x=0.01, ha="left")
    style.footer(fig, "현 기판·현 모델에서 stock 구조와 PD-mux가 같은 Job에 있는 유일한 서빙 원자료. 셀당 seed 2, Job 1개 → CI·검정력 없음, "
                      "arm 순위 서술 금지(설계 입력용). HI 셀 TTFT는 인용 금지(C:8792), shape X는 X_CORRECTNESS_FAIL로 미정의. "
                      "fused는 미조율(SGLang 기본값). 출처: stock_baseline/RESULT_stock_baseline_phaseA_2026-11-01.md §2, STOCK_METRICS_TABLE.csv.", c)
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    return fig


# evidence map — transcribed from RESULT_stock_baseline_phaseA §1·§2·§5 and CONSENSUS §1-1
CELLS = {
    # (baseline row, substrate col): (text, kind)
    ("stock SGLang fused\n(미조율 = 기본값)", "KISTI A100\nZamba2-2.7B"): ("n=5 paired\nPD-mux +40.5% / +185.8%\n(rate 2·3, 정본 §1-1)\nZamba 배제 → 인용만", "good"),
    ("stock SGLang fused\n(미조율 = 기본값)", "KISTI A100\nGranite-4.0-h-micro"): ("n=5 paired\nPD-mux +11.7% / +27.0%\n(rate 3·4, 정본 §1-1)", "good"),
    ("stock SGLang fused\n(미조율 = 기본값)", "VESSL A100\nNano-9B-v2"): ("M3 J1 · N=1 · seed 2\n기술값만, 순위 금지", "warning"),
    ("stock SGLang fused\n(조율: chunk·overlap·\nmixed 튜닝)", "KISTI A100\nZamba2-2.7B"): ("없음", "critical"),
    ("stock SGLang fused\n(조율: chunk·overlap·\nmixed 튜닝)", "KISTI A100\nGranite-4.0-h-micro"): ("없음", "critical"),
    ("stock SGLang fused\n(조율: chunk·overlap·\nmixed 튜닝)", "VESSL A100\nNano-9B-v2"): ("없음 → Phase B\n(결정 62, 2차)", "critical"),
    ("stock vLLM", "KISTI A100\nZamba2-2.7B"): ("0.22.1 · n=1 · 2026-07\n이력 전용(deprecated/)", "serious"),
    ("stock vLLM", "KISTI A100\nGranite-4.0-h-micro"): ("없음", "critical"),
    ("stock vLLM", "VESSL A100\nNano-9B-v2"): ("없음 → Phase B\n(VL-stock, 최신 stable)", "critical"),
    ("PD-mux static\n(D 격자, 비sticky)", "KISTI A100\nZamba2-2.7B"): ("HE0 캠페인 n≥4\nd44 best (관대 SLO)", "good"),
    ("PD-mux static\n(D 격자, 비sticky)", "KISTI A100\nGranite-4.0-h-micro"): ("P1-opint agnostic만\n(static D 격자 없음)", "warning"),
    ("PD-mux static\n(D 격자, 비sticky)", "VESSL A100\nNano-9B-v2"): ("R Job n=4 (구 셀)\nD34–64 고원, 1.5% 안\nCAP/TUNE/H 기술값", "warning"),
}
ROWS = ["stock SGLang fused\n(미조율 = 기본값)", "stock SGLang fused\n(조율: chunk·overlap·\nmixed 튜닝)", "stock vLLM", "PD-mux static\n(D 격자, 비sticky)"]
COLS = ["KISTI A100\nZamba2-2.7B", "KISTI A100\nGranite-4.0-h-micro", "VESSL A100\nNano-9B-v2"]


def fig_b(c):
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(0, len(COLS)); ax.set_ylim(0, len(ROWS)); ax.invert_yaxis()
    ax.set_xticks(np.arange(len(COLS)) + 0.5, COLS, fontsize=9)
    ax.set_yticks(np.arange(len(ROWS)) + 0.5, ROWS, fontsize=8.5)
    ax.xaxis.tick_top(); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    for i, r in enumerate(ROWS):
        for j, col in enumerate(COLS):
            text, kind = CELLS[(r, col)]
            face = {"good": "#e3f4e3", "warning": "#fff3d6", "serious": "#fde7dc", "critical": "#f6dddd"}[kind]
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.05), 0.94, 0.9, facecolor=face, edgecolor=c["axis"], lw=0.8))
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=7.8, color=c["ink"], linespacing=1.35)
    ax.set_title("서빙 측정 지도 — 대조군 × 기판·모델 (2026-11-01 현재)", loc="left", fontsize=11.5, pad=28)
    style.footer(fig, "녹색 = n≥4 paired 서빙 측정(정본 인용 가능, 범위 한정) · 황색 = 기술값(N=1 또는 순위 금지) · 주황 = 이력 전용 · 적색 = 없음. "
                      "'PD-mux가 vLLM을 이긴다/진다'는 어느 셀에도 근거가 없다. 다른 열의 수치를 섞어 비교하지 않는다(게이트 1). "
                      "출처: stock_baseline/RESULT_stock_baseline_phaseA_2026-11-01.md §1·§2·§5, PLAN §1 표 V1, CONSENSUS §1-1, policy_implfix RESULT §6.", c)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig


def main():
    rows = load_m3()
    for mode in ("light", "dark"):
        c = style.apply(mode)
        plt.rcParams["font.family"] = ["Noto Sans CJK KR", "Noto Sans CJK JP", "DejaVu Sans"]
        for name, fn in (("fig_a_nano_m3j1_lo_cells", lambda c: fig_a(c, rows)), ("fig_b_evidence_map", fig_b)):
            fig = fn(c)
            fig.savefig(OUT / f"{name}.{mode}.png", dpi=170)
            plt.close(fig)
    print("ok", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
