#!/usr/bin/env python3
"""Prefill (input) and decode (output) token distributions of the four motivation length pools.

Pool statistics only: no serving result, no new verdict.  Reads the pools through mw_lib.load_pool and writes
workload_token_split.json and figures/workload_token_split.png next to this file.
Decode steps = n_out - 1, because the first output token comes from the last prefill forward.
arXiv output here is the abstract length (main condition, user decision 2026-11-10); the 1,024-token variant is a constant.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "workspace/engine-port/results/motivation/workloads"))
import mw_lib as M  # noqa: E402

POOLS = [("sharegpt_first", "ShareGPT 첫 발화"), ("openchat_rounds", "OpenChat 다중 턴"), ("arxiv_summ", "arXiv 요약"), ("loogle_longdep", "LooGLE")]
COLORS = {"sharegpt_first": "#2a78d6", "openchat_rounds": "#eb6834", "arxiv_summ": "#1baf7a", "loogle_longdep": "#eda100"}
STYLES = {"sharegpt_first": "-", "openchat_rounds": "--", "arxiv_summ": "-.", "loogle_longdep": ":"}
PREFILL, DECODE = "#6250d6", "#e87ba4"
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLOCK = 1312       # vLLM attention block logged for the Nemotron configuration in the short check Job
LONG_SHARE = 0.05  # LooGLE share of requests in the interference cell


def stats(v):
    return {k: x for k, x in M.dist(v).items() if k in ("n", "mean", "cv", "min", "max", "p25", "p50", "p75", "p90", "p95", "p99")}


def summarize(rows):
    i = [r[1] for r in rows]
    o = [r[2] for r in rows]
    si, so = sum(i), sum(o)
    top = sorted(i, reverse=True)[: max(1, len(i) // 10)]
    return dict(
        input=stats(i), output=stats(o), decode_steps_mean=round(sum(x - 1 for x in o) / len(o), 1),
        prefill_tokens_total=si, decode_tokens_total=so, prefill_token_share=round(si / (si + so), 4), decode_token_share=round(so / (si + so), 4),
        tokens_per_output_token=round(si / so, 2), median_output_over_input=round(M.q(sorted(b / a for a, b in zip(i, o)), 50), 3),
        frac_output_gt_input=round(sum(b > a for a, b in zip(i, o)) / len(i), 4), frac_output_is_1=round(sum(b == 1 for b in o) / len(o), 4),
        top10pct_requests_share_of_prefill=round(sum(top) / si, 4), frac_total_le_block=round(sum(a + b <= BLOCK for a, b in zip(i, o)) / len(i), 4),
    )


def cdf(ax, v, key, label):
    xs = sorted(v)
    n = len(xs)
    ax.plot(xs, [(k + 1) / n for k in range(n)], color=COLORS[key], ls=STYLES[key], lw=2, label=label, solid_capstyle="round")


def style(ax, xlabel):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_xlim(1, 100000)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    pools = {k: M.load_pool(k) for k, _ in POOLS}
    out = {k: summarize(pools[k]) for k in pools}
    sg, lg = out["sharegpt_first"], out["loogle_longdep"]
    i = (1 - LONG_SHARE) * sg["input"]["mean"] + LONG_SHARE * lg["input"]["mean"]
    o = (1 - LONG_SHARE) * sg["output"]["mean"] + LONG_SHARE * lg["output"]["mean"]
    mix = dict(input_mean=round(i, 1), output_mean=round(o, 1), prefill_token_share=round(i / (i + o), 4), decode_token_share=round(o / (i + o), 4),
               loogle_share_of_prefill=round(LONG_SHARE * lg["input"]["mean"] / i, 4), loogle_share_of_decode=round(LONG_SHARE * lg["output"]["mean"] / o, 4))
    (HERE / "workload_token_split.json").write_text(json.dumps(dict(
        status="pool statistics, not a serving result", block_tokens_vllm_nemotron_config=BLOCK, arxiv_output="abstract length; variant fixed 1024",
        workloads=out, interference_mix_sharegpt_plus_5pct_loogle=mix), ensure_ascii=False, indent=1))

    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    fig = plt.figure(figsize=(12, 7.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.65], hspace=0.42, wspace=0.08, left=0.135, right=0.975, top=0.86, bottom=0.075)
    a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    for k, name in POOLS:
        cdf(a, [r[1] for r in pools[k]], k, name)
        cdf(b, [r[2] for r in pools[k]], k, name)
    style(a, "요청당 입력 토큰 (prefill)")
    style(b, "요청당 출력 토큰 (decode)")
    b.tick_params(labelleft=False)
    a.set_ylabel("요청 누적 비율", color=MUTED, fontsize=10)
    b.axvline(1024, color=MUTED, lw=1, ls=(0, (2, 3)))
    b.text(1130, 0.06, "arXiv 변형\n1,024 고정", fontsize=9, color=MUTED, va="bottom")
    for ax, k, x, y, ha in ((a, "sharegpt_first", 55, .42, "left"), (a, "openchat_rounds", 500, .36, "right"), (a, "arxiv_summ", 5200, .12, "left"), (a, "loogle_longdep", 24000, .30, "left"),
                            (b, "loogle_longdep", 2.4, .36, "left"), (b, "arxiv_summ", 150, .55, "right"), (b, "openchat_rounds", 520, .55, "left"), (b, "sharegpt_first", 45, .12, "right")):
        ax.text(x, y, dict(POOLS)[k], fontsize=9.5, color=TEXT, ha=ha, bbox=dict(fc=SURFACE, ec="none", pad=1.2, alpha=.9))
    a.set_title("prefill 입력 길이", loc="left", fontsize=12, color=TEXT, fontweight="bold")
    b.set_title("decode 출력 길이", loc="left", fontsize=12, color=TEXT, fontweight="bold")

    rows = [(dict(POOLS)[k], out[k]["prefill_token_share"], out[k]["decode_token_share"]) for k, _ in POOLS]
    rows.append(("ShareGPT + LooGLE 5%", mix["prefill_token_share"], mix["decode_token_share"]))
    rows.reverse()
    y = range(len(rows))
    c.barh(y, [r[1] * 100 for r in rows], color=PREFILL, height=.62, label="prefill", edgecolor=SURFACE, linewidth=2)
    c.barh(y, [r[2] * 100 for r in rows], left=[r[1] * 100 for r in rows], color=DECODE, height=.62, label="decode", edgecolor=SURFACE, linewidth=2)
    c.set_yticks(list(y), [r[0] for r in rows], fontsize=9.5, color=TEXT)
    c.set_xlim(0, 113)
    c.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    c.tick_params(colors=MUTED, length=0, labelsize=9)
    c.tick_params(axis="y", colors=TEXT)
    c.grid(True, axis="x", color=GRID, lw=.8)
    c.set_axisbelow(True)
    for s in c.spines.values():
        s.set_visible(False)
    for k, r in zip(y, rows):
        c.text(101, k, f"decode {r[2] * 100:.1f}%", va="center", fontsize=9.5, color=TEXT)
    c.set_title("전체 토큰 중 prefill과 decode 비중", loc="left", fontsize=12, color=TEXT, fontweight="bold")
    c.legend(loc="upper right", bbox_to_anchor=(1, 1.3), ncol=2, frameon=False, fontsize=9.5)

    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(.975, .985), handlelength=3)
    fig.text(.135, .955, "워크로드별 prefill·decode 토큰 분포", fontsize=16, color=TEXT, fontweight="bold")
    fig.text(.135, .925, "길이 풀 집계, 서빙 결과 아님. 가로축은 로그, arXiv 출력은 초록 길이", fontsize=10, color=MUTED)
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures/workload_token_split.png", dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
