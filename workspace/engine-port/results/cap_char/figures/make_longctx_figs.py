#!/usr/bin/env python3
"""cap 블록 + long-ctx 블록 설계 그림 4장 (2026-10-20, main, 측정 0 — [측정] 길이 표 + [유도] 산술).

usage: python3 -I make_longctx_figs.py   (저장소 어디서 실행해도 됨; 경로는 이 파일 기준)

F1 longctx_input_len_ecdf.png   — 입력 길이 분포(ECDF): LongBench v1, v2, InfiniteBench 자연문서 [측정] + arXiv-Sum 분위 [측정]
F2 longctx_docs_per_target.png  — 목표 길이별 쓸 수 있는 고유 문서: ±10% 선별 vs 목표 길이 절단 [측정]
F3 longctx_mem_per_req.png      — 요청당 메모리와 메모리 상한 동시 수 C_mem(L): Nano vs Llama [유도]
(F4 span 정지 그림은 뺐다: 단순 token-layer 산술이 J1 PD-mux arm ITL(25–37 ms)과 맞지 않아 기전 모형이 미확인 — 프로브에서 실측)
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../../.."))
sys.path.insert(0, os.path.join(REPO, "reports", "figures"))
import style  # noqa: E402  (validated reference palette, fixed slot order)

C = style.apply("light")
S = C["series"]
INK, SEC, MUT = C["ink"], C["secondary"], C["muted"]
TSV = os.path.join(HERE, "..", "longctx_stats", "longctx_lengths.tsv")

IB_SYNTH = {"passkey", "number_string", "kv_retrieval", "math_find", "math_calc", "code_run"}
K = 1024
TARGETS = [32 * K, 64 * K, 96 * K, 128 * K]
L_STAR = 8762            # Nano s/k = 136.9 MiB / 16 KiB [유도]
NATIVE = 131072          # Nano·Llama native context


def load():
    rows = list(csv.DictReader(open(TSV), delimiter="\t"))
    for r in rows:
        r["nano"], r["llama"] = int(r["nano"]), int(r["llama"])
    return rows


def natural(r):
    return r["meta"] != "zh" and not (r["dataset"] == "ib" and r["task"] in IB_SYNTH)


def kfmt(x, _=None):
    return f"{x / K:.0f}k" if x >= K else f"{x:.0f}"


def kticks(ax, ticks):
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: (f"{x / K / 1024:.0f}M" if x >= K * K else f"{x / K:.0f}k")))


def refline(ax, x, label, y=0.97, ha="left"):
    ax.axvline(x, color=MUT, lw=1, ls="--", zorder=1)
    ax.text(x * (1.04 if ha == "left" else 0.96), y, label, color=SEC, fontsize=7.5,
            ha=ha, va="top", transform=ax.get_xaxis_transform())


def save(fig, name):
    out = os.path.join(HERE, name)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(out)


def f1(rows):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    groups = [("LongBench v1 (+E)", lambda r: r["dataset"] == "lb1" and natural(r), S[0]),
              ("LongBench v2", lambda r: r["dataset"] == "lb2" and natural(r), S[1]),
              ("InfiniteBench (natural docs)", lambda r: r["dataset"] == "ib" and natural(r), S[2])]
    pos = {"LongBench v1 (+E)": (1.08, -0.06, "left"), "LongBench v2": (0.55, 0.30, "right"),
           "InfiniteBench (natural docs)": (1.9, -0.18, "left")}
    for name, sel, col in groups:
        seen, xs = set(), []
        for r in rows:
            if sel(r) and r["ctx_sha16"] not in seen:
                seen.add(r["ctx_sha16"]); xs.append(r["nano"])
        xs.sort(); n = len(xs)
        ax.step(xs, [(i + 1) / n for i in range(n)], where="post", color=col, lw=2, zorder=3)
        i50 = xs[n // 2]
        ax.plot([i50], [0.5], "o", ms=7, color=col, mec=C["surface"], mew=2, zorder=4)
        fx, dy, ha = pos[name]
        ax.text(i50 * fx, 0.5 + dy, f"{name}\nunique docs {n}, p50 {kfmt(i50)}",
                color=INK, fontsize=7.5, va="top", ha=ha)
    # arXiv-Sum: only quantiles are stored (ARXSUM_STATS, kept <=16k) -> p10-p90 bar + p50 dot
    p10, p50, p90 = 2877, 6484, 12369
    ax.plot([p10, p90], [0.08, 0.08], color=S[3], lw=5, solid_capstyle="round", zorder=3)
    ax.plot([p50], [0.08], "o", ms=8, color=S[3], mec=C["surface"], mew=2, zorder=4)
    ax.text(p10, 0.12, "arXiv-Sum (standard cell, <=16k): p10-p90 bar, p50 dot", color=INK,
            fontsize=7.5, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(1000, 2_000_000)
    for x, lab in ((L_STAR, "L* 8.8k\n(KV = state)"), (32 * K, "32k"), (64 * K, "64k"),
                   (96 * K, "96k"), (NATIVE, "128k native\nlimit")):
        refline(ax, x, lab, y=0.99)
    kticks(ax, [K, 4 * K, 16 * K, 64 * K, 256 * K, K * K])
    ax.set_xlabel("input tokens (context + question, Nano-9B-v2 tokenizer, log scale)")
    ax.set_ylabel("cumulative share of unique docs")
    ax.set_ylim(0, 1.02)
    ax.set_title("Where each source sits on the length axis (zh tasks and InfiniteBench synthetic tasks excluded)",
                 loc="left")
    save(fig, "longctx_input_len_ecdf.png")


def f2(rows):
    pool = [r for r in rows if natural(r)]
    sel, trunc = [], []
    for T in TARGETS:
        sel.append(len({r["ctx_sha16"] for r in pool if 0.9 * T <= r["nano"] <= 1.1 * T}))
        trunc.append(len({r["ctx_sha16"] for r in pool if r["nano"] >= T}))
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.36
    xs = range(len(TARGETS))
    b1 = ax.bar([x - w / 2 - 0.01 for x in xs], sel, w, color=S[0], label="select docs within T +/-10%")
    b2 = ax.bar([x + w / 2 + 0.01 for x in xs], trunc, w, color=S[1], label="truncate docs >= T to exactly T")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 15, f"{int(b.get_height())}",
                    ha="center", va="bottom", fontsize=8, color=INK)
    for y, lab in ((128, "128 = 4 seeds x 32 req (docs shared across rates)"),
                   (384, "384 = 4 seeds x 3 rates x 32 req (no reuse)")):
        ax.axhline(y, color=MUT, lw=1, ls="--", zorder=1)
        ax.text(-0.45, y + 12, lab, color=SEC, fontsize=7.5, ha="left", va="bottom")
    ax.set_xticks(list(xs), ["32k", "64k", "96k", "128k\n(+/-10% bin crosses the\n131,072 native limit)"])
    ax.set_xlabel("target input length T (Nano tokenizer)")
    ax.set_ylabel("unique documents available")
    ax.set_title("Long-ctx block: +/-10% selection runs out at 64k-128k; truncation keeps hundreds", loc="left")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.92), fontsize=8)
    ax.set_ylim(0, max(trunc) * 1.15)
    save(fig, "longctx_docs_per_target.png")


def f3():
    GiB, MiB, KiB = 1024 ** 3, 1024 ** 2, 1024
    s_nano, k_nano, k_llama = 136.9 * MiB, 16 * KiB, 128 * KiB
    avail_nano = 48.3 * GiB      # KV 41.9 + state(cap48) 6.42 GiB, 저장소 실측 합 [측정 근거, 합산은 유도]
    avail_llama = 49.9 * GiB     # 64.9 GiB pool - 15.0 GiB nominal weights [유도, 부팅 로그로 확인 필요]
    Ls = [int(1000 * 1.06 ** i) for i in range(0, 120) if 1000 * 1.06 ** i <= NATIVE]
    mem_n = [(s_nano + k_nano * L) / GiB for L in Ls]
    mem_l = [(k_llama * L) / GiB for L in Ls]
    cm_n = [avail_nano / (s_nano + k_nano * L) for L in Ls]
    cm_l = [avail_llama / (k_llama * L) for L in Ls]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, yn, yl, ylab, title in (
            (a1, mem_n, mem_l, "memory per request (GiB)", "Per-request memory = s + k*L"),
            (a2, cm_n, cm_l, "requests that fit in memory, C_mem(L)", "Memory ceiling on concurrency")):
        ax.plot(Ls, yn, color=S[0], lw=2)
        ax.plot(Ls, yl, color=S[1], lw=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        kticks(ax, [K, 4 * K, 16 * K, 64 * K, 128 * K])
        ax.set_xlabel("total context per request L = input + output (tokens, log)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        ax.text(Ls[-1], yn[-1], "  Nemotron-Nano-9B-v2\n  (s 137 MiB, k 16 KiB)", color=INK, fontsize=7.5,
                va="center", ha="left")
        ax.text(Ls[-1], yl[-1], "  Llama-3.1-8B\n  (s 0, k 128 KiB)", color=INK, fontsize=7.5,
                va="center", ha="left")
        refline(ax, L_STAR, "L* 8.8k", y=0.99)
        refline(ax, 1200, "L_x 1.2k\n(curves cross)", y=0.99)
    a2.axhline(48, color=MUT, lw=1, ls=":")
    a2.text(1050, 52, "cap 48 (inherited constant)", color=SEC, fontsize=7.5)
    for ax in (a1, a2):
        ax.set_xlim(1000, NATIVE * 3.2)
    fig.suptitle("Derived from config + boot-log sums, not measured in serving  -  A100 80GB, mem-fraction 0.82",
                 x=0.01, ha="left", fontsize=8, color=SEC, y=1.0)
    fig.tight_layout()
    save(fig, "longctx_mem_per_req.png")


if __name__ == "__main__":
    rows = load()
    f1(rows); f2(rows); f3()
