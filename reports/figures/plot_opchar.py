#!/usr/bin/env python3
"""opchar figures + tables from one Job's data dir (Bullet ASPLOS'26 Fig 2 / Table 2 / Fig 3 analogue).

    python3 reports/figures/plot_opchar.py <run_dir>     # run_dir = results/opchar/<mode>_<job>

CHARACTERIZATION ONLY. Every number is an isolated operator kernel on one SM partition of one
A100 (VESSL), median of n timed repetitions inside ONE Job — descriptive, not a serving or
policy result, and not comparable with KISTI-substrate numbers.

Outputs (next to this script): opchar_fig2_<job>.png, opchar_fig3_<job>.png, opchar_memcpy_<job>.png,
and <run_dir>/ANALYSIS/{fig2_share.csv, fig2_util.csv, table2_idle.csv, table2_idle.md, fig3_speedup.csv}.
"""
from __future__ import annotations

import csv
import glob
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path[:0] = [str(HERE), str(REPO / "workspace"), str(REPO / "workspace" / "characterization")]

from experiments.common.wave_model import gemm_tile_blocks, wave_quantization  # noqa: E402
from experiments.opchar import ops_spec as S  # noqa: E402

PEAK_TFLOPS_BF16 = 312.0          # A100 dense bf16 spec sheet (derived util denominator)
GEMM_TILE = (128, 256)            # ASSUMED cuBLAS tile (reproduces Bullet Table 2; not observed)
FULL = 108

OP_ORDER = ["M:in_proj", "M:conv1d", "M:conv_update", "M:ssd_scan", "M:ssm_update", "M:gated_norm",
            "M:out_proj", "*:qkv_proj", "*:kv_write", "*:attn", "*:o_proj", "-:up_proj", "-:relu2",
            "-:down_proj", "norm"]
LABEL = {"M:in_proj": "M in_proj", "M:conv1d": "M conv1d", "M:conv_update": "M conv_upd",
         "M:ssd_scan": "M SSD scan", "M:ssm_update": "M SSM update", "M:gated_norm": "M gated_norm",
         "M:out_proj": "M out_proj", "*:qkv_proj": "A qkv", "*:kv_write": "A kv_write",
         "*:attn": "A attn", "*:o_proj": "A o_proj", "-:up_proj": "MLP up", "-:relu2": "MLP relu²",
         "-:down_proj": "MLP down", "norm": "RMSNorm (all)"}


def read_labeled(path):
    with open(path) as f:
        rows = list(csv.reader(f))[1:]
    head = [c.split("__")[0] for c in rows[0]]
    return [dict(zip(head, r)) for r in rows[1:]]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(run_dir: Path):
    data = run_dir / "data"
    ops = [r for p in glob.glob(str(data / "ops_*.csv")) for r in read_labeled(p)]
    launch = [r for p in glob.glob(str(data / "launch_*.csv")) for r in read_labeled(p)]
    mem = [r for p in glob.glob(str(data / "memcpy_*.csv")) for r in read_labeled(p)]
    return ops, launch, mem


def best_ms(rows):
    """phase,size,sm,op -> ms. Decode prefers CUDA-graph timing (engine decode is graph-captured)."""
    out = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        key = (r["phase"], int(r["size"]), int(r["n_sm_req"]), r["op"])
        ms = fnum(r["ms_median"])
        pref = 1 if (r["phase"] == "decode") == (r["timing_mode"] == "graph") else 0
        if key not in out or pref > out[key][1]:
            out[key] = (ms, pref, r["timing_mode"])
    return out


def merge_norm(key):
    return "norm" if key.endswith(":norm") else key


def shares(ms, phase, size, counts, sm=FULL):
    per = {k[3]: v[0] for k, v in ms.items() if k[0] == phase and k[1] == size and k[2] == sm}
    if not per:
        return {}
    sh = S.model_shares(per, counts)
    out = defaultdict(float)
    for k, v in sh.items():
        out[merge_norm(k)] += v
    return dict(out)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def table2(launch, ms, cfg, sizes, sms):
    """Idle % per op: Triton ops from RECORDED launches (segment-weighted), GEMMs from ASSUMED tile."""
    by = defaultdict(list)
    for r in launch:
        if r["phase"] == "prefill":
            by[(int(r["size"]), int(r["n_sm_req"]), r["op"])].append(r)
    out = []
    for L in sizes:
        spec = {o.key: o for ops in S.prefill_ops(cfg, L).values() for o in ops}
        for sm in sms:
            row = {"L": L, "sm": sm}
            for key, o in spec.items():
                if o.category == "gemm":
                    m, n = o.shape["m"], o.shape["n"]
                    row[key] = 100 * wave_quantization(gemm_tile_blocks(m, n, *GEMM_TILE), sm)["idle_ratio"]
                    row[key + "#src"] = "assumed-tile"
                    continue
                recs = by.get((L, sm, key), [])
                num = den = 0.0
                for r in recs:
                    idle, seg = fnum(r["idle_ratio"]), fnum(r["segment_ms_sum"]) or 0.0
                    if idle is not None and seg > 0:
                        num += idle * seg
                        den += seg
                if den > 0:
                    row[key] = 100 * num / den
                    row[key + "#src"] = "triton-launch"
            out.append(row)
    return out


def main(run_dir):
    run_dir = Path(run_dir)
    job = run_dir.name
    cfg = S.cfg_from_models_yaml("nemotron_nano_9b_v2")
    counts = S.layer_counts(cfg.pattern)
    ops, launch, mem = load(run_dir)
    ms = best_ms(ops)
    an = run_dir / "ANALYSIS"
    sizes_p = sorted({k[1] for k in ms if k[0] == "prefill"})
    sizes_d = sorted({k[1] for k in ms if k[0] == "decode"})
    sms = sorted({k[2] for k in ms})
    memcpy = {int(r["n_sm_req"]): fnum(r["gbps"]) for r in mem}
    bw_ceiling = memcpy.get(FULL) or max(memcpy.values() or [2039.0])

    # ---- Fig 2(a) shares (model-level, layer-count weighted, 108 SM)
    share_rows = []
    for phase, sizes in (("prefill", sizes_p), ("decode", sizes_d)):
        for sz in sizes:
            for op, v in shares(ms, phase, sz, counts).items():
                share_rows.append({"phase": phase, "size": sz, "op": op, "share": v})
    write_csv(an / "fig2_share.csv", share_rows)

    # ---- Fig 2(b)(c) derived util at 108 SM
    spec_cache = {}
    util_rows = []
    for (phase, sz, sm, op), (t, _, mode) in ms.items():
        if sm != FULL or not t:
            continue
        key = (phase, sz)
        if key not in spec_cache:
            d = S.prefill_ops(cfg, sz) if phase == "prefill" else S.decode_ops(cfg, sz, 2048)
            spec_cache[key] = {o.key: o for v in d.values() for o in v}
        o = spec_cache[key][op]
        util_rows.append({"phase": phase, "size": sz, "op": op, "ms": t, "timing": mode,
                          "compute_util_derived": o.flops / (t / 1e3) / 1e12 / PEAK_TFLOPS_BF16,
                          "bw_util_derived": o.bytes_total / (t / 1e3) / 1e9 / bw_ceiling})
    write_csv(an / "fig2_util.csv", util_rows)

    # ---- Table 2
    t2 = table2(launch, ms, cfg, [L for L in sizes_p if L in (1024, 2048, 4096, 16384)] or sizes_p,
                [s for s in sms if s in (108, 44, 16)])
    write_csv(an / "table2_idle.csv", t2)
    cols = ["M:in_proj", "M:conv1d", "M:ssd_scan", "M:out_proj", "*:qkv_proj", "*:o_proj",
            "-:up_proj", "-:down_proj"]
    with open(an / "table2_idle.md", "w") as f:
        f.write(f"# Table 2 analogue — wave-quantization SM idle % (DERIVED), job {job}\n\n")
        f.write("Triton ops (conv1d, SSD scan): recorded launch grid/regs/smem, segment-time weighted. "
                f"GEMMs: ASSUMED {GEMM_TILE[0]}x{GEMM_TILE[1]} tiles, b=1 (not observed). "
                "Attention/norm/elementwise not covered (no CUPTI on this image).\n\n")
        f.write("| L | SM | " + " | ".join(LABEL.get(c, c) for c in cols) + " |\n")
        f.write("|---|---|" + "---|" * len(cols) + "\n")
        for r in t2:
            cells = []
            for c in cols:
                v = r.get(c)
                tag = "" if r.get(c + "#src") == "triton-launch" else "*"
                cells.append("—" if v is None else f"{v:.1f}{tag}")
            f.write(f"| {r['L']} | {r['sm']} | " + " | ".join(cells) + " |\n")
        f.write("\n`*` = assumed GEMM tile.\n")

    # ---- Fig 3 speedups
    sp_rows = []
    for (phase, sz, sm, op), (t, _, mode) in ms.items():
        base = ms.get((phase, sz, FULL, op))
        if base and t:
            sp_rows.append({"phase": phase, "size": sz, "sm": sm, "op": op, "speedup_vs_108": base[0] / t,
                            "timing": mode})
    write_csv(an / "fig3_speedup.csv", sp_rows)

    ns = sorted({int(r["n"]) for r in ops if r["status"] == "ok" and r["n"]})
    n_txt = f"n={ns[0]}" if len(ns) == 1 else f"n={ns[0]}–{ns[-1]}"
    plot(job, cfg, counts, ms, share_rows, util_rows, sp_rows, memcpy, sizes_p, sizes_d, sms, n_txt)
    print(f"analysis -> {an}")


def plot(job, cfg, counts, ms, share_rows, util_rows, sp_rows, memcpy, sizes_p, sizes_d, sms, n_txt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import style
    c = style.apply("light")
    for name in ("Noto Sans CJK KR", "Noto Sans CJK JP"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = name
            break
    cat_color = {"M": c["series"][1], "*": c["series"][3], "-": c["series"][2], "n": c["muted"]}
    hatch_cycle = ["", "//", "..", "xx", "\\\\", "++", "oo"]

    def op_style(op, idx):
        layer = "n" if op == "norm" else op[0]
        return cat_color[layer], hatch_cycle[idx % len(hatch_cycle)]

    foot = (f"VESSL A100 SXM4 80GB · Nemotron-Nano-9B-v2 · isolated operator kernels (engine kernels, "
            f"random weights) · {n_txt} median · job {job} · CHARACTERIZATION, not serving")

    # ---------------- Fig 2
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, phase, sizes in ((axes[0, 0], "prefill", sizes_p), (axes[1, 0], "decode", sizes_d)):
        present = [op for op in OP_ORDER if any(r["op"] == op and r["phase"] == phase for r in share_rows)]
        for yi, sz in enumerate(sizes):
            left = 0.0
            sh = {r["op"]: r["share"] for r in share_rows if r["phase"] == phase and r["size"] == sz}
            for i, op in enumerate(present):
                w = sh.get(op, 0.0)
                col, h = op_style(op, i)
                ax.barh(yi, w, left=left, color=col, hatch=h, edgecolor=c["surface"], height=0.66,
                        label=LABEL[op] if yi == 0 else None)
                if w > 0.06:
                    ax.text(left + w / 2, yi, f"{100 * w:.0f}%", ha="center", va="center", fontsize=7,
                            color="white")
                left += w
        lab = "L=" if phase == "prefill" else "B="
        ax.set_yticks(range(len(sizes)), [f"{lab}{s}" for s in sizes])
        ax.set_xlim(0, 1)
        ax.set_xlabel("모델 전체 시간 비중 (층별 시간 × 층 수 27M/4A/25MLP, 108 SM)")
        ax.set_title(f"(a) 연산자별 실행시간 비중 — {phase}" + (" (ctx 2048, CUDA graph)" if phase == "decode" else ""),
                     loc="left", fontsize=10)
        ax.grid(False)
        ax.legend(frameon=False, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    for ax, metric, title in ((axes[0, 1], "compute_util_derived", "(b) compute util — 유도값 (분석 FLOP ÷ 시간 ÷ 312 TF)"),
                              (axes[1, 1], "bw_util_derived", "(c) BW util — 유도값 (분석 byte ÷ 시간 ÷ memcpy 108SM 실측)")):
        pick = [s for s in sizes_p if s in (1024, 4096, 16384)] or sizes_p
        ops_here = [op for op in OP_ORDER if op != "norm" and any(r["op"] == op and r["phase"] == "prefill" for r in util_rows)] + ["M:norm"]
        x = list(range(len(ops_here)))
        wdt = 0.8 / max(1, len(pick))
        for j, sz in enumerate(pick):
            vals = []
            for op in ops_here:
                r = next((u for u in util_rows if u["phase"] == "prefill" and u["size"] == sz and u["op"] == op), None)
                vals.append(r[metric] if r else 0)
            ax.bar([xi + (j - (len(pick) - 1) / 2) * wdt for xi in x], vals, wdt, color=c["series"][j % 4],
                   hatch="///", edgecolor=c["surface"], label=f"prefill L={sz}")
        ax.axhline(1.0, ls="--", color=c["secondary"], lw=1)
        ax.set_xticks(x, [LABEL.get(op, "RMSNorm") for op in ops_here], rotation=40, ha="right", fontsize=7.5)
        ax.set_ylabel("천장 대비 비율")
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle("Fig 2 대응 — 연산자 단위 시간 비중과 유도 활용률 (빗금 = derived, 하드웨어 카운터 아님)",
                 x=0.01, ha="left", fontsize=12)
    fig.text(0.01, 0.005, foot, fontsize=7, color=c["muted"])
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(HERE / f"opchar_fig2_{job}.png", dpi=140)
    plt.close(fig)

    # ---------------- Fig 3
    panels = [("prefill", s) for s in sizes_p if s in (1024, 4096, 16384, 32768)] + \
             [("decode", s) for s in sizes_d if s in (32, 128)]
    ncol = 3
    nrow = math.ceil(len(panels) / ncol) or 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 4.2 * nrow), squeeze=False)
    for ax, (phase, sz) in zip(axes.flat, panels):
        ratio = [s / FULL for s in sms]
        ax.plot([16 / FULL, 1], [16 / FULL, 1], ls=":", color=c["muted"], lw=1)
        present = [op for op in OP_ORDER if op != "norm" and any(r["op"] == op and r["phase"] == phase and r["size"] == sz for r in sp_rows)] + ["M:norm"]
        for i, op in enumerate(present):
            pts = sorted((r["sm"] / FULL, r["speedup_vs_108"]) for r in sp_rows
                         if r["phase"] == phase and r["size"] == sz and r["op"] == op)
            if not pts:
                continue
            col, _ = op_style("norm" if op == "M:norm" else op, i)
            ls = ["-", "--", "-.", ":"][i % 4]
            ax.plot(*zip(*pts), ls=ls, marker="o", ms=3, color=col, lw=1.6, label=LABEL.get(op, "RMSNorm"))
        ax.set_ylim(0, 1.08)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("SM 비율 (green partition / 108)")
        ax.set_ylabel("speedup (108 SM = 1)")
        lab = f"L={sz}" if phase == "prefill" else f"B={sz}, ctx 2048, graph"
        ax.set_title(f"{phase} {lab}", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower right")
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle("Fig 3 대응 — 연산자별 SM 비율 대비 speedup (격리 실행, 점선 = 선형)", x=0.01, ha="left", fontsize=12)
    fig.text(0.01, 0.005, foot, fontsize=7, color=c["muted"])
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(HERE / f"opchar_fig3_{job}.png", dpi=140)
    plt.close(fig)

    # ---------------- memcpy (Bullet Fig 8a analogue)
    if memcpy:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        xs = sorted(memcpy)
        ax.plot(xs, [memcpy[x] for x in xs], marker="o", color=c["series"][0])
        ax.set_xlabel("SM 수 (green partition)")
        ax.set_ylabel("device-to-device copy GB/s")
        ax.set_title("memcpy 대역폭 vs SM (Bullet Fig 8a 대응)", loc="left", fontsize=10)
        fig.text(0.01, 0.005, f"job {job} · {n_txt} median · 1 GiB bf16 copy", fontsize=7, color=c["muted"])
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(HERE / f"opchar_memcpy_{job}.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1])
