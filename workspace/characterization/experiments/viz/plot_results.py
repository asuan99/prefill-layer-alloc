"""
plot_results.py — visualize the v2 results in results_v2/ (E0–E4 + verdicts).

Reads the labeled CSVs (strips the `# value_kind` line and the `__measured/
__derived/__metadata` header suffixes) and writes PNGs to results_v2/figures/.
Every panel is guarded: if an experiment's CSVs are missing it is skipped with a
note, so this runs at any pipeline stage (E0-only, after E2, etc.).

Headline figures:
  E0  grid_sat_sm vs batch        — predicted batch where asymmetry dies
  E1  scan_share vs batch         — G0 (is scan the dominant component?)
  E2  sat_sm(CI) vs batch + E0    — G1 (BW=flat vs GRID=rises toward total_sm)
  E2  bw_util@sat vs batch        — G1 (does HBM saturate? 85% line)
  E3  decode floor vs batch       — Memory Gap (decode as donor)
  E4  decode inflation by type    — bandwidth interference (if E4 ran)

Run with the project venv (has matplotlib/pandas):
  source $REPO_ROOT/bin/activate
  python experiments/viz/plot_results.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
_CHAR = os.path.abspath(os.path.join(_here, "..", ".."))

# consistent per-model styling (size sweep)
MODEL_ORDER = ["zamba2_1.2b", "zamba2_2.7b", "falcon_h1_1.5b", "falcon_h1_3b"]
MODEL_COLOR = {
    "zamba2_1.2b": "#1f77b4", "zamba2_2.7b": "#4c9be8",
    "falcon_h1_1.5b": "#d62728", "falcon_h1_3b": "#ff9896",
}
TOTAL_SM = 108


def load_csv(path: str) -> pd.DataFrame:
    """Read a labeled CSV: skip the value_kind comment, strip __label suffixes."""
    df = pd.read_csv(path, skiprows=1)
    df.columns = [c.split("__")[0] for c in df.columns]
    return df


def _concat(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat([load_csv(f) for f in files], ignore_index=True)


def _models_in(df) -> list:
    return [m for m in MODEL_ORDER if m in set(df.get("model", []))]


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


# ---------------------------------------------------------------------------
# E0
# ---------------------------------------------------------------------------

def plot_e0(rdir, out):
    wt = os.path.join(rdir, "e0", "wave_table_*.csv")
    files = sorted(glob.glob(wt))
    if not files:
        print("E0: no wave_table — skip"); return
    df = load_csv(files[0])
    df = df[df["chunk_granularity"].astype(str) == "256"]
    # grid_sat_sm vs batch (per model); one point per (model,batch) — collapse seq/sm
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in _models_in(df):
        g = (df[df["model"] == m].groupby("batch")["grid_sat_sm"].first().reset_index())
        ax.plot(g["batch"], g["grid_sat_sm"], "o-", color=MODEL_COLOR[m], label=m)
    ax.axhline(TOTAL_SM, color="k", ls="--", lw=1, label=f"total_sm={TOTAL_SM}")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("batch"); ax.set_ylabel("grid_sat_sm  (derived)")
    ax.set_title("E0: SSM grid saturation SM vs batch (chunk=256)\n"
                 "above the dashed line → grid fills all SMs → asymmetry dead (GRID)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    _save(fig, out, "e0_grid_sat_vs_batch.png")

    # temporal overhead budget bars
    bf = sorted(glob.glob(os.path.join(rdir, "e0", "temporal_overhead_budget_*.csv")))
    if bf:
        b = load_csv(bf[0])
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(b)); w = 0.38
        ax.bar(x - w/2, b["per_swap_overhead_pct"], w, label="per-swap %", color="#1f77b4")
        ax.bar(x + w/2, b["full_model_overhead_pct"], w, label="full-model %", color="#ff7f0e")
        ax.set_xticks(x); ax.set_xticklabels(b["model"], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("overhead % (derived)")
        ax.set_title("E0: temporal-policy swap overhead budget\n"
                     f"(swap≈{b['swap_us'].iloc[0]:.1f}µs measured; per-layer compute ASSUMED)")
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
        _save(fig, out, "e0_overhead_budget.png")


# ---------------------------------------------------------------------------
# E1 — G0
# ---------------------------------------------------------------------------

def plot_e1(rdir, out):
    df = _concat(os.path.join(rdir, "e1", "component_sweep_*.csv"))
    if df.empty:
        print("E1: no component_sweep — skip"); return
    ok = df[df["status"] == "ok"].copy()
    # scan_share vs batch (chunk=256)
    s = ok[(ok["chunk_granularity"].astype(str) == "256") & (ok["component"] == "scan")]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in _models_in(s):
        g = s[s["model"] == m].groupby("batch")["scan_share_pct"].mean().reset_index()
        ax.plot(g["batch"], g["scan_share_pct"], "o-", color=MODEL_COLOR[m], label=m)
    ax.axhline(30, color="k", ls="--", lw=1, label="G0 threshold 30%")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch"); ax.set_ylabel("scan_share_pct (derived)")
    ax.set_title("E1 / G0: SSM scan share of the layer vs batch (chunk=256)")
    ax.set_ylim(0, 100); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _save(fig, out, "e1_scan_share_vs_batch.png")

    # component latency breakdown at batch=8, chunk=256
    sub = ok[(ok["chunk_granularity"].astype(str) == "256") & (ok["batch"] == 8)
             & (ok["component"].isin(["in_proj", "scan", "out_proj"]))]
    if not sub.empty:
        comps = ["in_proj", "scan", "out_proj"]
        models = _models_in(sub)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bottom = np.zeros(len(models))
        colors = {"in_proj": "#9467bd", "scan": "#2ca02c", "out_proj": "#8c564b"}
        for c in comps:
            vals = [sub[(sub["model"] == m) & (sub["component"] == c)]["latency_ms"].mean()
                    for m in models]
            vals = np.nan_to_num(vals)
            ax.bar(models, vals, bottom=bottom, label=c, color=colors[c])
            bottom += vals
        ax.set_ylabel("latency_ms (measured)")
        ax.set_title("E1: prefill component latency breakdown (batch=8, chunk=256)")
        ax.legend(fontsize=8); ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        _save(fig, out, "e1_component_breakdown.png")


# ---------------------------------------------------------------------------
# E2 — G1 (the headline)
# ---------------------------------------------------------------------------

def plot_e2(rdir, out):
    ci = _concat(os.path.join(rdir, "e2", "saturation_ci_*.csv"))
    if not ci.empty:
        s = ci[(ci["layer_type"] == "ssm") & (ci["chunk_granularity"].astype(str) == "256")]
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for m in _models_in(s):
            g = s[s["model"] == m].sort_values("batch")
            yerr = np.vstack([g["sat_sm_point"] - g["sat_sm_ci_low"],
                              g["sat_sm_ci_high"] - g["sat_sm_point"]])
            ax.errorbar(g["batch"], g["sat_sm_point"], yerr=yerr, fmt="o-",
                        color=MODEL_COLOR[m], capsize=3, label=f"{m} (measured sat_sm±CI)")
            ax.plot(g["batch"], np.minimum(g["grid_sat_sm_e0"], TOTAL_SM), ":",
                    color=MODEL_COLOR[m], alpha=0.7,
                    label=f"{m} (E0 grid pred, clamped)")
        ax.axhline(TOTAL_SM, color="k", ls="--", lw=1, label=f"total_sm={TOTAL_SM}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("batch"); ax.set_ylabel("saturation SM")
        ax.set_title("E2 / G1: SSM saturation SM vs batch (chunk=256)\n"
                     "flat & batch-invariant → BW mechanism; rises toward total_sm → GRID")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        _save(fig, out, "e2_saturation_vs_batch.png")

        # bw util at saturation
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for m in _models_in(s):
            g = s[s["model"] == m].sort_values("batch")
            ax.plot(g["batch"], g["bw_util_pct_at_sat"], "o-", color=MODEL_COLOR[m], label=m)
        ax.axhline(85, color="k", ls="--", lw=1, label="BW-bound ~85%")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("batch"); ax.set_ylabel("BW util % at saturation (measured/derived)")
        ax.set_title("E2 / G1: HBM utilization at the saturation point (chunk=256)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        _save(fig, out, "e2_bwutil_at_sat.png")

    # SM-sweep saturation curves (throughput vs SM), zamba2_1.2b ssm chunk=256
    sw = _concat(os.path.join(rdir, "e2", "sm_sweep_*.csv"))
    if not sw.empty:
        sw = sw[sw["status"] == "ok"].copy()
        sub = sw[(sw["layer_type"] == "ssm") & (sw["chunk_granularity"].astype(str) == "256")]
        for m in _models_in(sub):
            d = sub[sub["model"] == m]
            if d.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for b in sorted(d["batch"].unique()):
                g = d[d["batch"] == b].sort_values("sm_count")
                tput = 1.0 / g["latency_ms"]
                ax.plot(g["sm_count"], tput / tput.max(), "o-", label=f"batch={b}")
            ax.set_xlabel("SM count"); ax.set_ylabel("norm. throughput (1/latency)")
            ax.set_title(f"E2: SSM SM-scaling curves — {m} (chunk=256)")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            _save(fig, out, f"e2_smcurve_{m}.png")
            break  # one representative model to keep the figure set small


# ---------------------------------------------------------------------------
# E2 — attn vs ssm RESOURCE ASYMMETRY (the layer-alloc thesis)
# ---------------------------------------------------------------------------

def plot_asymmetry(rdir, out):
    ci = _concat(os.path.join(rdir, "e2", "saturation_ci_*.csv"))
    if ci.empty:
        print("asymmetry: no E2 CI — skip"); return
    ci = ci[ci["chunk_granularity"].astype(str) == "256"]
    models = _models_in(ci)

    # (1) sat_sm: SSM vs ATTN per model — the exploitable gap
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, m in zip(axes.ravel(), models):
        s = ci[(ci.model == m) & (ci.layer_type == "ssm")].sort_values("batch")
        a = ci[(ci.model == m) & (ci.layer_type == "attn") & (ci.context_len == 0)].sort_values("batch")
        for d, lab, c in [(s, "SSM", "#2ca02c"), (a, "Attn", "#d62728")]:
            ye = np.vstack([d.sat_sm_point - d.sat_sm_ci_low, d.sat_sm_ci_high - d.sat_sm_point])
            ax.errorbar(d.batch, d.sat_sm_point, yerr=ye, fmt="o-", color=c, capsize=3, label=lab)
        # shade the asymmetry gap
        if len(s) and len(a):
            bm = sorted(set(s.batch) & set(a.batch))
            sv = [s[s.batch == b].sat_sm_point.iloc[0] for b in bm]
            av = [a[a.batch == b].sat_sm_point.iloc[0] for b in bm]
            ax.fill_between(bm, sv, av, color="grey", alpha=0.25, label="asymmetry (free SM)")
        ax.axhline(TOTAL_SM, color="k", ls="--", lw=0.8)
        ax.set_xscale("log", base=2); ax.set_title(m, fontsize=10)
        ax.set_ylabel("saturation SM"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("batch")
    fig.suptitle("E2: Attn vs SSM saturation SM — resource asymmetry (chunk=256)\n"
                 "Attn saturates at more SMs than SSM ⇒ give SSM fewer, Attn more",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out, "e2_asymmetry_satsm.png")

    # (2) bw_util: the asymmetry is in the MECHANISM (grid SSM vs BW/compute Attn)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for m in models:
        s = ci[(ci.model == m) & (ci.layer_type == "ssm")].groupby("batch")["bw_util_pct_at_sat"].mean()
        a = ci[(ci.model == m) & (ci.layer_type == "attn") & (ci.context_len == 0)].groupby("batch")["bw_util_pct_at_sat"].mean()
        ax.plot(s.index, s.values, "o-", color=MODEL_COLOR[m], alpha=0.9, label=f"{m} SSM")
        ax.plot(a.index, a.values, "s--", color=MODEL_COLOR[m], alpha=0.6, label=f"{m} Attn")
    ax.axhline(85, color="k", ls=":", lw=1, label="BW-bound ~85%")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("batch"); ax.set_ylabel("BW util % at saturation (log)")
    ax.set_title("E2: saturation mechanism is asymmetric\n"
                 "SSM low util (grid-limited) vs Attn high util (BW/compute-limited)\n"
                 "[NOTE: absolute BW% unreliable — attn_bytes overcounts cached KV, "
                 "ssm undercounts state]", fontsize=9)
    ax.legend(fontsize=6, ncol=2); ax.grid(True, which="both", alpha=0.3)
    _save(fig, out, "e2_mechanism_bwutil.png")


# ---------------------------------------------------------------------------
# E3 — decode floor / Memory Gap
# ---------------------------------------------------------------------------

def plot_e3(rdir, out):
    fl = _concat(os.path.join(rdir, "e3", "decode_floor_*.csv"))
    if fl.empty:
        print("E3: no decode_floor — skip"); return
    for lt, ctx in [("ssm", None), ("attn", 4096)]:
        d = fl[fl["layer_type"] == lt]
        if ctx is not None and "context_len" in d:
            d = d[d["context_len"] == ctx]
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for m in _models_in(d):
            g = d[d["model"] == m].sort_values("batch")
            g = g.groupby("batch")["floor_sm_point"].mean().reset_index()
            ax.plot(g["batch"], g["floor_sm_point"], "o-", color=MODEL_COLOR[m], label=m)
        ax.axhline(TOTAL_SM, color="k", ls="--", lw=1, label=f"total_sm={TOTAL_SM}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("decode batch"); ax.set_ylabel("decode floor SM (derived)")
        suff = f" ctx={ctx}" if ctx else ""
        ax.set_title(f"E3: decode SM floor vs batch — {lt}{suff}\n"
                     "low & flat → Memory Gap → decode is a donor")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        _save(fig, out, f"e3_decode_floor_{lt}.png")


# ---------------------------------------------------------------------------
# E4 — concurrent (only if it ran)
# ---------------------------------------------------------------------------

def plot_e4(rdir, out):
    pp = _concat(os.path.join(rdir, "e4", "concurrent_pp_*.csv"))
    di = _concat(os.path.join(rdir, "e4", "decode_interference_*.csv"))
    if pp.empty and di.empty:
        print("E4: no data — skip (runs only if G1 authorizes / FORCE_E4)")
        return

    # (1) prefill+prefill concurrent throughput: uniform vs layer-aware split
    if not pp.empty:
        pp = pp[pp["status"] == "ok"]
        models = _models_in(pp); modes = sorted(pp["budget_mode"].unique())
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        x = np.arange(len(models)); w = 0.8 / max(1, len(modes))
        for i, md in enumerate(modes):
            vals = [pp[(pp.model == m) & (pp.budget_mode == md)]["throughput_gain"].mean()
                    for m in models]
            ax.bar(x + (i - (len(modes) - 1) / 2) * w, np.nan_to_num(vals), w, label=md)
        ax.axhline(1.0, color="k", ls="--", lw=1, label="no gain (=1.0)")
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("prefill-prefill throughput_gain (derived)")
        ax.set_title("E4 (a): concurrent prefill throughput — uniform vs layer-aware SM split\n"
                     ">1 = aware split helps; overlap is small (overlap_ratio≈1)")
        ax.set_ylim(0.85, 1.15); ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
        _save(fig, out, "e4_pp_throughput.png")

    # (2) decode latency inflation by concurrent prefill TYPE, per budget mode
    if not di.empty:
        di = di[di["status"] == "ok"]
        modes = sorted(set(di["budget_mode"]))
        fig, axes = plt.subplots(1, len(modes), figsize=(6.5 * len(modes), 4.6), sharey=True)
        if len(modes) == 1:
            axes = [axes]
        models = _models_in(di)
        col = {"ssm": "#2ca02c", "attn": "#d62728"}
        for ax, md in zip(axes, modes):
            d = di[di["budget_mode"] == md]
            x = np.arange(len(models)); w = 0.38
            for i, lt in enumerate(["ssm", "attn"]):
                vals = [d[(d.model == m) & (d.prefill_layer_type == lt)]
                        ["decode_inflation_pct"].mean() for m in models]
                ax.bar(x + (i - 0.5) * w, np.nan_to_num(vals), w, label=f"prefill={lt}", color=col[lt])
            psm = d["prefill_sm"].iloc[0] if len(d) else "?"
            dsm = d["decode_sm"].iloc[0] if len(d) else "?"
            ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
            ax.set_title(f"{md}\n(prefill={psm} SM / decode={dsm} SM)", fontsize=9)
            ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)
        axes[0].set_ylabel("decode latency inflation % (derived)")
        fig.suptitle("E4 (b): decode inflation by concurrent prefill type — "
                     "SSM ≥ Attn ⇒ layer-type-dependent bandwidth interference",
                     fontweight="bold", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        _save(fig, out, "e4_decode_interference.png")


# ---------------------------------------------------------------------------
# Serving re-interpretation: prefill+decode overlap + free-SM budget vs batch
# ---------------------------------------------------------------------------

def plot_serving(rdir, out):
    pp = _concat(os.path.join(rdir, "e4", "concurrent_pp_*.csv"))
    di = _concat(os.path.join(rdir, "e4", "decode_interference_*.csv"))
    fl = _concat(os.path.join(rdir, "e3", "decode_floor_*.csv"))
    if di.empty or pp.empty:
        print("serving: needs E4 pp+di — skip"); return
    pp = pp[pp.status == "ok"]; di = di[di.status == "ok"]
    models = _models_in(di)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # Panel A: SSM-prefill + decode concurrent (50/50) vs sequential full-SM.
    # Only ssm is a valid reconstruction: the scan ignores context_len, so its
    # solo (pp, ctx=0) and concurrent (di, ctx=4096) prefill are the same work.
    # (attn is excluded — pp solo_b has ctx=0 but di runs attn with ctx=4096.)
    ax = axes[0]; x = np.arange(len(models))
    sp = []
    for m in models:
        pr = pp[(pp.model == m) & (pp.budget_mode == "uniform_50_50")]
        pf_solo = pr["solo_a_ms"].mean() if len(pr) else np.nan      # a = ssm
        d = di[(di.model == m) & (di.budget_mode == "uniform_50_50")
               & (di.prefill_layer_type == "ssm")]
        if not len(d) or np.isnan(pf_solo):
            sp.append(np.nan); continue
        tseq = pf_solo + d.decode_solo_ms.iloc[0]
        tconc = max(d.prefill_concurrent_ms.iloc[0], d.decode_concurrent_ms.iloc[0])
        sp.append(tseq / tconc if tconc else np.nan)
    bars = ax.bar(x, np.nan_to_num(sp), 0.6, color="#2ca02c")
    for xi, v in zip(x, sp):
        if not np.isnan(v):
            ax.text(xi, v + 0.01, f"{v:.2f}x", ha="center", fontsize=8)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="no gain (sequential)")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("SSM-prefill+decode speedup (concurrent 50/50 ÷ sequential)")
    ax.set_ylim(0, 1.6)
    ax.set_title("(A) SSM-prefill + DECODE overlap — the serving scenario\n"
                 "vs strict-sequential (upper bound; MPS/two-stream baseline TODO)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    # Panel B: decode free-SM budget (108 - floor) vs decode batch
    ax = axes[1]
    if not fl.empty:
        sub = fl[(fl.layer_type == "ssm") & (fl.context_len == 4096)]
        for m in _models_in(sub):
            g = sub[sub.model == m].groupby("batch")["floor_sm_point"].mean().reset_index()
            ax.plot(g.batch, TOTAL_SM - g.floor_sm_point, "o-", color=MODEL_COLOR[m], label=m)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("decode batch (concurrent requests)")
        ax.set_ylabel("free SM budget for prefill  (108 − decode floor)")
        ax.set_title("(B) free-SM budget shrinks as decode batch grows\n"
                     "large batch → decode fills all SMs → spatial-split window closes", fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("E4 re-interpreted for serving (multi-request): overlap helps at "
                 "low/moderate decode batch; the window closes at high batch",
                 fontweight="bold", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out, "serving_overlap_and_budget.png")


# ---------------------------------------------------------------------------
# E5 — serving prefill+decode coexistence (the real serving answer)
# ---------------------------------------------------------------------------

_PF_ROWS = ["ssm", "attn"]            # prefill layer types (heatmap rows)
_DEC_COLS = ["attn", "ssm"]           # decode layer types  (heatmap cols)
_CELL_STYLE = {("ssm", "attn"): ("#2ca02c", "o-"), ("ssm", "ssm"): ("#98df8a", "o--"),
               ("attn", "attn"): ("#d62728", "s-"), ("attn", "ssm"): ("#ff9896", "s--")}


def plot_e5(rdir, out):
    df = _concat(os.path.join(rdir, "e5", "serving_coexec_*.csv"))
    if df.empty:
        print("E5: no serving_coexec — skip (run e5)"); return
    df = df[df["status"] == "ok"].copy()
    if "decode_layer" not in df.columns:       # back-compat with single-cell E5
        df["decode_layer"] = "attn"
    ctxs = sorted(df["context_len"].dropna().unique())
    ctx = ctxs[len(ctxs) // 2] if ctxs else None
    d0 = df[df["context_len"] == ctx] if ctx is not None else df
    models = _models_in(d0)

    def ts_speedup(sm, pl, dl, b):
        r = sm[(sm.prefill_layer == pl) & (sm.decode_layer == dl)
               & (sm.decode_batch == b) & (sm.backend == "two_stream")]
        return float(r["speedup_vs_seq"].iloc[0]) if len(r) else np.nan

    # (A) 2×2 overlap matrix per model — two_stream speedup at the smallest batch
    fig, axes = plt.subplots(1, len(models), figsize=(4.3 * len(models), 4.2))
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        sm = d0[d0.model == m]
        bmin = min(sm.decode_batch.unique()) if len(sm) else None
        M = np.array([[ts_speedup(sm, pl, dl, bmin) for dl in _DEC_COLS] for pl in _PF_ROWS])
        im = ax.imshow(M, cmap="RdYlGn", vmin=0.8, vmax=2.0, aspect="auto")
        ax.set_xticks(range(len(_DEC_COLS))); ax.set_xticklabels([f"dec={d}" for d in _DEC_COLS], fontsize=8)
        ax.set_yticks(range(len(_PF_ROWS))); ax.set_yticklabels([f"pf={p}" for p in _PF_ROWS], fontsize=8)
        for i in range(len(_PF_ROWS)):
            for j in range(len(_DEC_COLS)):
                v = M[i, j]
                ax.text(j, i, "—" if np.isnan(v) else f"{v:.2f}x", ha="center", va="center",
                        fontsize=10, fontweight="bold")
        ax.set_title(f"{m}\n(two_stream speedup @ db={bmin})", fontsize=9)
    fig.suptitle("E5: prefill×decode overlap matrix (two_stream speedup vs sequential, max-overlap batch)\n"
                 "best pairing = compute/SM-heavy prefill ⊗ BW-bound decode",
                 fontweight="bold", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, out, "e5_overlap_matrix.png")

    # (B) speedup vs batch, one line per (prefill,decode) cell, per model
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, m in zip(axes.ravel(), models):
        sm = d0[d0.model == m]
        batches = sorted(sm.decode_batch.unique())
        for pl in _PF_ROWS:
            for dl in _DEC_COLS:
                color, style = _CELL_STYLE[(pl, dl)]
                ys = [ts_speedup(sm, pl, dl, b) for b in batches]
                if not all(np.isnan(ys)):
                    ax.plot(batches, ys, style, color=color, label=f"pf={pl}/dec={dl}")
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.set_xscale("log", base=2); ax.set_title(m, fontsize=10)
        ax.set_ylabel("two_stream speedup vs sequential"); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("decode batch")
    fig.suptitle("E5: overlap speedup vs decode batch, by prefill×decode cell\n"
                 "window closes as decode fills the GPU; cell = which pairing overlaps best",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, out, "e5_speedup_vs_batch.png")

    # (C) spatial-partition-SPECIFIC gain (best over cells & fracs) = two_stream/green_ctx
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for m in models:
        sm = d0[d0.model == m]
        batches = sorted(sm.decode_batch.unique())
        ys = []
        for b in batches:
            ratios = []
            for pl in _PF_ROWS:
                for dl in _DEC_COLS:
                    cell = sm[(sm.prefill_layer == pl) & (sm.decode_layer == dl)
                              & (sm.decode_batch == b)]
                    ts = cell[cell.backend == "two_stream"]["concurrent_ms"]
                    gc = cell[cell.backend == "green_ctx"]["concurrent_ms"]
                    if len(ts) and len(gc) and gc.min() > 0:
                        ratios.append(ts.iloc[0] / gc.min())   # best green over fracs
            ys.append(max(ratios) if ratios else np.nan)        # best cell
        ax.plot(batches, ys, "o-", color=MODEL_COLOR[m], label=m)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="no spatial benefit (=two_stream)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("decode batch"); ax.set_ylabel("BEST green_ctx ÷ two_stream  (>1 = partition helps)")
    ax.set_title("E5: spatial-partition gain over two-stream — best over all cells/fracs\n"
                 "(if even the best cell is ≤1, Green Context never beats naive co-scheduling)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _save(fig, out, "e5_spatial_specific_gain.png")


# ---------------------------------------------------------------------------
# Verdict summary panel
# ---------------------------------------------------------------------------

def plot_verdicts(rdir, out):
    lines = []
    for g in ("g0", "g1"):
        p = os.path.join(rdir, "verdicts", f"{g}_verdict.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        lines.append(f"{d['gate']}: {d['verdict']}")
        for m in d.get("per_model", []):
            if isinstance(m, dict) and "model" in m:
                lines.append(f"    {m['model']}: {m['verdict']}")
        lines.append(f"  action: {d.get('recommended_action','')}")
        lines.append("")
    if not lines:
        return
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(lines) + 1))
    ax.axis("off")
    ax.text(0.01, 0.99, "\n".join(lines), va="top", ha="left",
            fontfamily="monospace", fontsize=10)
    ax.set_title("v2 gate verdicts", loc="left", fontweight="bold")
    _save(fig, out, "verdicts_summary.png")


def main():
    ap = argparse.ArgumentParser(description="Plot v2 results")
    ap.add_argument("--results-dir", default=os.path.join(_CHAR, "results_v2"))
    ap.add_argument("--out", default=os.path.join(_CHAR, "results_v2", "figures"))
    args = ap.parse_args()
    print(f"v2 visualization  (results={args.results_dir}, out={args.out})")
    for fn in (plot_e0, plot_e1, plot_e2, plot_asymmetry, plot_e3, plot_e4,
               plot_serving, plot_e5, plot_verdicts):
        try:
            fn(args.results_dir, args.out)
        except Exception as e:  # noqa: BLE001
            print(f"  {fn.__name__} failed: {type(e).__name__}: {e}")
    print("done.")


if __name__ == "__main__":
    main()
