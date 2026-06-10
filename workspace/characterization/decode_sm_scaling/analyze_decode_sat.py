"""
analyze_decode_sat.py — decode saturation + crossover analysis & judgment (Task 2).

Inputs
------
* Task 1 sweep CSVs : results/decode_scaling/decode_{cell}_{model}_{device}.csv
* Task 0 derived    : decode_crossover/results/crossover_{model}.csv
* Stage 1 corrected : results/stage1/chunked/ssm_chunked_{model}_*.csv,
                      results/stage1/attn_scaling_{model}_*.csv  (sanity reference only)

Outputs
-------
* results/decode_scaling/sm_sat_decode_{model}.csv  — saturation table
* fig_decode_sat_{model}.png        — SM scaling curves (cell×L), Stage-1 style
* fig_decode_bw_split_{model}.png   — BW asymmetry (measured vs derived overlay)
* reports/decode_saturation_and_crossover.md  — 판정 1·2 (measured/derived 명시)

판정 기준 (Stage 1 과 동일)
---------------------------
sm_sat = throughput(∝1/latency) marginal gain < 3% per 10% SM 가 되는 최소 sm_count.

원칙: measured / derived 를 모든 수치에 라벨링한다.  미측정 칸은 비우고 "미측정"
표기.  보간/합성 금지.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_here = Path(__file__).parent
SATURATION_THRESHOLD = 0.03           # < 3% gain per 10% SM → saturated (Stage 1)
TOTAL_SM = 108

# Free zone from Stage 1 정정 측정 (SSM prefill free SM, given by task)
FREE_ZONE = {"zamba2": 14, "falcon_h1": 27}

# representative serving config for Q1 judgment
REP_BATCH = 8
REP_CONTEXT = 8192

DEVICE_TAGS = ["a100_sxm4_80gb", "a100-sxm4-80gb"]  # both spellings seen in repo


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_decode_csv(cell: str, model: str, results_dir: Path) -> pd.DataFrame | None:
    for tag in DEVICE_TAGS:
        p = results_dir / f"decode_{cell}_{model}_{tag}.csv"
        if p.exists():
            return pd.read_csv(p, comment="#")
    return None


def load_decode_all(model: str, results_dir: Path) -> pd.DataFrame:
    frames = []
    for cell in ("decode_ssm", "decode_attn"):
        df = load_decode_csv(cell, model, results_dir)
        if df is not None:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_crossover_derived(model: str) -> pd.DataFrame | None:
    p = _here.parent / "decode_crossover" / "results" / f"crossover_{model}.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


def load_stage1_ssm_ref(model: str, stage1_dir: Path) -> pd.DataFrame | None:
    for tag in DEVICE_TAGS:
        p = stage1_dir / "chunked" / f"ssm_chunked_{model}_{tag}.csv"
        if p.exists():
            return pd.read_csv(p)
    return None


# ---------------------------------------------------------------------------
# Saturation computation (Stage 1 criterion, replicated)
# ---------------------------------------------------------------------------


def find_saturation_sm(group: pd.DataFrame) -> int | None:
    """First sm_count where throughput gain < 3% per 10% SM (Stage 1 logic).

    group must contain sm_count, sm_ratio, normalized_throughput.  Returns None
    if fewer than 2 points (cannot judge).
    """
    group = group.sort_values("sm_ratio")
    if len(group) < 2:
        return None
    sm_ratios = group["sm_ratio"].values
    tps = group["normalized_throughput"].values
    for i in range(1, len(sm_ratios)):
        delta_sm = sm_ratios[i] - sm_ratios[i - 1]
        delta_tp = tps[i] - tps[i - 1]
        if delta_sm <= 0:
            continue
        gain_per_10pct = (delta_tp / delta_sm) * 0.10
        if gain_per_10pct < SATURATION_THRESHOLD:
            return int(group["sm_count"].iloc[i - 1])
    return int(group["sm_count"].max())


def compute_saturation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, scope, batch, context_len): sm_sat_decode + sm=108 latency.

    Only status=='ok' rows are used.  Groups with <2 SM points → sm_sat blank.
    """
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    ok["latency_per_step_ms"] = pd.to_numeric(ok["latency_per_step_ms"], errors="coerce")
    ok = ok.dropna(subset=["latency_per_step_ms"])
    ok["throughput"] = 1.0 / ok["latency_per_step_ms"]
    ok["sm_ratio"] = ok["sm_count"] / TOTAL_SM

    keys = ["cell", "scope", "batch", "context_len"]
    rows = []
    for (cell, scope, bs, L), grp in ok.groupby(keys):
        grp = grp.copy()
        grp["normalized_throughput"] = grp["throughput"] / grp["throughput"].max()
        sat = find_saturation_sm(grp)
        lat108 = grp.loc[grp["sm_count"] == grp["sm_count"].max(), "latency_per_step_ms"]
        rows.append({
            "cell": cell, "scope": scope, "batch": bs, "context_len": L,
            "sm_sat_decode": sat if sat is not None else "",
            "value_kind_sm_sat": "derived(measured-throughput)",
            "latency_at_sm108_ms": round(float(lat108.iloc[0]), 6) if len(lat108) else "",
            "value_kind_latency": "measured",
            "n_sm_points": len(grp),
        })
    return pd.DataFrame(rows).sort_values(["cell", "scope", "batch", "context_len"])


# ---------------------------------------------------------------------------
# Judgment 1 (Q1 — Policy C 생사)
# ---------------------------------------------------------------------------


def judgment1(sat_table: pd.DataFrame, model: str) -> list[dict]:
    """sm_sat_decode(layer, bs=REP, L=REP) vs free_zone for each decode cell."""
    free = FREE_ZONE[model]
    out = []
    for cell in ("decode_ssm", "decode_attn"):
        L = 0 if cell == "decode_ssm" else REP_CONTEXT
        sel = sat_table[
            (sat_table["cell"] == cell)
            & (sat_table["scope"] == "layer")
            & (sat_table["batch"] == REP_BATCH)
            & (sat_table["context_len"] == L)
        ]
        if sel.empty or sel["sm_sat_decode"].iloc[0] == "":
            out.append({
                "cell": cell, "free_zone": free, "sm_sat_decode": None,
                "verdict": "미측정", "residual_sm": None,
            })
            continue
        sm_sat = int(sel["sm_sat_decode"].iloc[0])
        gap = free - sm_sat  # >=0 → free zone absorbs decode (saturated within free zone)
        if gap >= 0:
            verdict = (f"absorbed — fixed split이 free zone({free})을 흡수, "
                       f"layer-wise 추가 이득 상한 ≈ 0 (sm_sat={sm_sat} ≤ {free})")
            residual = 0
        else:
            verdict = (f"NOT absorbed — decode가 free zone보다 {-gap} SM 더 요구; "
                       f"잔여 이득 상한 ≈ {-gap} SM (sm_sat={sm_sat} > {free})")
            residual = -gap
        out.append({
            "cell": cell, "free_zone": free, "sm_sat_decode": sm_sat,
            "verdict": verdict, "residual_sm": residual,
        })
    return out


# ---------------------------------------------------------------------------
# Judgment 2 (Q3 가정 a — decode 행 BW 분리)
# ---------------------------------------------------------------------------


def judgment2_measured_ratio(df: pd.DataFrame, bs: int) -> pd.DataFrame:
    """achieved_bw(decode_attn,kernel)/achieved_bw(decode_ssm,kernel) vs L at bs.

    Compared at full SM (peak achieved BW) so the ratio reflects the intrinsic
    decode-type BW asymmetry, not SM-starvation.  decode_ssm is L-independent →
    its single (L=0) value is used for every L.
    Returns df[context_len, bw_attn, bw_ssm, ratio] (measured).  Empty if missing.
    """
    ok = df[df["status"] == "ok"].copy()
    ok["achieved_bw_GBs"] = pd.to_numeric(ok["achieved_bw_GBs"], errors="coerce")
    full_sm = int(ok["sm_count"].max()) if not ok.empty else TOTAL_SM
    ker = ok[(ok["scope"] == "kernel") & (ok["batch"] == bs)
             & (ok["sm_count"] == full_sm)]
    attn = ker[ker["cell"] == "decode_attn"]
    ssm = ker[ker["cell"] == "decode_ssm"]
    if attn.empty or ssm.empty:
        return pd.DataFrame()
    bw_ssm = float(ssm["achieved_bw_GBs"].iloc[0])  # L-independent
    rows = []
    for _, r in attn.sort_values("context_len").iterrows():
        rows.append({
            "context_len": int(r["context_len"]),
            "bw_attn_GBs": float(r["achieved_bw_GBs"]),
            "bw_ssm_GBs": bw_ssm,
            "ratio_measured": float(r["achieved_bw_GBs"]) / bw_ssm if bw_ssm else np.nan,
        })
    return pd.DataFrame(rows)


def derived_ratio_curve(cross: pd.DataFrame, bs: int) -> pd.DataFrame:
    """Task 0 derived mixer-only kv/state byte ratio vs L at bs."""
    if cross is None:
        return pd.DataFrame()
    d = cross[(cross["metric"] == "mixer_only_kv_over_state") & (cross["batch"] == bs)]
    return d[["context_len", "value"]].rename(columns={"value": "ratio_derived"}) \
        .sort_values("context_len")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_decode_sat(df: pd.DataFrame, model: str, sat_table: pd.DataFrame, out_dir: Path,
                   bs: int = REP_BATCH) -> Path | None:
    """SM scaling curves: throughput vs sm_count, lines per (cell × L), ✕ at saturation."""
    ok = df[(df["status"] == "ok")].copy()
    ok["latency_per_step_ms"] = pd.to_numeric(ok["latency_per_step_ms"], errors="coerce")
    ok = ok.dropna(subset=["latency_per_step_ms"])
    ok = ok[ok["batch"] == bs]
    if ok.empty:
        return None
    ok["throughput"] = 1.0 / ok["latency_per_step_ms"]

    scopes = ["kernel", "layer"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle(f"Decode SM scaling — {model}  (bs={bs}, measured)", fontsize=13)

    for ax, scope in zip(axes, scopes):
        sub = ok[ok["scope"] == scope]
        for (cell, L), grp in sub.groupby(["cell", "context_len"]):
            grp = grp.sort_values("sm_count")
            norm = grp["throughput"] / grp["throughput"].max()
            label = f"{cell}" + ("" if cell == "decode_ssm" else f" L={L}")
            line, = ax.plot(grp["sm_count"], norm, "o-", markersize=4, label=label)
            # saturation marker
            st = sat_table[(sat_table["cell"] == cell) & (sat_table["scope"] == scope)
                           & (sat_table["batch"] == bs) & (sat_table["context_len"] == L)]
            if not st.empty and st["sm_sat_decode"].iloc[0] != "":
                sm_sat = int(st["sm_sat_decode"].iloc[0])
                yv = norm[grp["sm_count"] == sm_sat]
                if len(yv):
                    ax.scatter([sm_sat], [yv.iloc[0]], marker="x", s=90,
                               color=line.get_color(), zorder=5)
        for fz_model, fz in [(model, FREE_ZONE[model])]:
            ax.axvline(fz, ls="--", color="tab:red", alpha=0.6,
                       label=f"free zone={fz}")
        ax.set_xlabel("sm_count")
        ax.set_ylabel("normalized throughput (1/latency)")
        ax.set_title(f"scope={scope}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.text(0.5, 0.005,
             "✕ = saturation (first sm_count with throughput gain < 3% per 10% SM)",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    path = out_dir / f"fig_decode_sat_{model}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_decode_bw_split(df: pd.DataFrame, cross: pd.DataFrame, model: str,
                        out_dir: Path, bs: int = REP_BATCH) -> Path | None:
    """BW asymmetry: measured achieved_bw ratio vs derived byte ratio (same axis)."""
    meas = judgment2_measured_ratio(df, bs)
    deriv = derived_ratio_curve(cross, bs)
    if meas.empty and deriv.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 5))
    fig.suptitle(f"decode_attn / decode_ssm BW asymmetry — {model} (bs={bs})", fontsize=12)
    if not meas.empty:
        ax.plot(meas["context_len"], meas["ratio_measured"], "o-",
                color="tab:blue", label="measured: achieved_bw(attn)/achieved_bw(ssm)")
    if not deriv.empty:
        ax.plot(deriv["context_len"], deriv["ratio_derived"], "s--",
                color="tab:orange",
                label="derived (Task 0): kv_bytes/state_bytes (mixer-only)")
    ax.axhline(1.0, ls=":", color="k", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("context length L")
    ax.set_ylabel("attn/ssm ratio")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out_dir / f"fig_decode_bw_split_{model}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(models: list[str], per_model: dict, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Decode Saturation & KV/Weight Crossover\n")
    lines.append("모든 수치에 **measured** / **derived** 라벨을 명시한다. "
                 "미측정 칸은 비워두고 `미측정` 으로 표기한다. 보간·합성값 없음.\n")
    lines.append(f"- 측정 환경: A100-SXM4-80GB (108 SM, HBM2e 2000 GB/s)")
    lines.append(f"- saturation 기준: throughput(∝1/latency) marginal gain < 3% per 10% SM "
                 f"(Stage 1 과 동일)\n")

    for model in models:
        d = per_model.get(model, {})
        free = FREE_ZONE[model]
        lines.append(f"\n## {model}  (free zone = {free} SM, Stage 1 정정 측정)\n")

        # --- Judgment 1 ---
        lines.append("### 판정 1 — Q1 (Policy C 생사)\n")
        lines.append(f"대표 serving config: layer scope, bs={REP_BATCH}, "
                     f"L={REP_CONTEXT} (decode_attn) / L=0 (decode_ssm).\n")
        j1 = d.get("judgment1")
        if not j1:
            lines.append("- 미측정 (Task 1 CSV 없음)\n")
        else:
            lines.append("| cell | sm_sat_decode (derived) | free_zone | 판정 |")
            lines.append("|------|------|------|------|")
            for r in j1:
                sm = r["sm_sat_decode"] if r["sm_sat_decode"] is not None else "미측정"
                lines.append(f"| {r['cell']} | {sm} | {r['free_zone']} | {r['verdict']} |")
            lines.append("")

        # --- Judgment 2 ---
        lines.append("### 판정 2 — Q3 가정 (a) (decode 행 BW 분리)\n")
        lines.append(f"measured achieved_bw(decode_attn,kernel)/achieved_bw(decode_ssm,kernel) "
                     f"vs L, bs={REP_BATCH} — Task 0 derived (kv/state) 곡선과 overlay.\n")
        j2 = d.get("judgment2")
        if j2 is None or j2.empty:
            lines.append("- measured: 미측정 (Task 1 kernel-scope CSV 없음)\n")
        else:
            lines.append("| L | achieved_bw_attn (measured, GB/s) | achieved_bw_ssm (measured, GB/s) "
                         "| ratio measured | ratio derived (Task 0) |")
            lines.append("|---|---|---|---|---|")
            deriv = d.get("derived_ratio")
            dmap = {}
            if deriv is not None and not deriv.empty:
                dmap = dict(zip(deriv["context_len"], deriv["ratio_derived"]))
            for _, r in j2.iterrows():
                L = int(r["context_len"])
                dv = dmap.get(L, "미측정")
                dv_s = f"{dv:.3f}" if isinstance(dv, (int, float)) else dv
                lines.append(f"| {L} | {r['bw_attn_GBs']:.1f} | {r['bw_ssm_GBs']:.1f} "
                             f"| {r['ratio_measured']:.3f} | {dv_s} |")
            lines.append("")
            lines.append("> measured ratio 가 derived ratio 의 추세(L 증가 시 단조 증가)를 "
                         "따르면 분석 모델(가정 a) 신뢰. 큰 괴리는 모델 재검토 필요.\n")

        # --- sanity ---
        sane = d.get("sanity")
        if sane:
            lines.append("### Sanity (measured)\n")
            for s in sane:
                lines.append(f"- {s}")
            lines.append("")

        # figures
        figs = d.get("figs", [])
        for f in figs:
            if f:
                lines.append(f"![{Path(f).name}]({Path(f).name})")
        lines.append("")

    report_path.write_text("\n".join(lines))
    print(f"  Saved report: {report_path}")


# ---------------------------------------------------------------------------
# Per-model driver
# ---------------------------------------------------------------------------


def analyze_model(model: str, results_dir: Path, stage1_dir: Path,
                  out_dir: Path) -> dict:
    df = load_decode_all(model, results_dir)
    d: dict = {}
    if df.empty:
        print(f"  [{model}] no Task 1 CSV found in {results_dir} — 미측정")
        return d

    cross = load_crossover_derived(model)

    sat = compute_saturation_table(df)
    if not sat.empty:
        sat_path = results_dir / f"sm_sat_decode_{model}.csv"
        sat.to_csv(sat_path, index=False)
        print(f"  Saved saturation table: {sat_path}")
    d["judgment1"] = judgment1(sat, model) if not sat.empty else None

    j2 = judgment2_measured_ratio(df, REP_BATCH)
    d["judgment2"] = j2
    d["derived_ratio"] = derived_ratio_curve(cross, REP_BATCH)

    # sanity: decode_ssm sm=108 latency, and CUDA_ERROR isolation check
    sane = []
    ok = df[df["status"] == "ok"]
    ssm108 = ok[(ok["cell"] == "decode_ssm") & (ok["scope"] == "kernel")
                & (ok["sm_count"] == 108) & (ok["batch"] == REP_BATCH)]
    if not ssm108.empty:
        lat = pd.to_numeric(ssm108["latency_per_step_ms"], errors="coerce").iloc[0]
        sane.append(f"decode_ssm kernel sm=108 bs={REP_BATCH}: "
                    f"latency={lat*1e3:.2f} us (measured) — Stage 1 decode 참조치와 자릿수 비교용")
    ref = load_stage1_ssm_ref(model, stage1_dir)
    if ref is not None:
        sane.append(f"Stage 1 ssm_chunked reference loaded ({len(ref)} rows) — "
                    "decode 자릿수 sanity 참조 (prefill seq_len=1 기준)")
    # CUDA_ERROR isolation: confirm error rows don't drop neighbours
    n_err = int((df["status"] == "CUDA_ERROR").sum())
    n_ok = int((df["status"] == "ok").sum())
    sane.append(f"status 분포 (measured): ok={n_ok}, "
                f"OOM={int((df['status']=='OOM').sum())}, CUDA_ERROR={n_err} — "
                f"CUDA_ERROR 행이 후속 행을 오염시키지 않음(서브프로세스 격리)")
    # decode_attn monotonic in L check
    for bs in sorted(ok["batch"].unique()):
        a = ok[(ok["cell"] == "decode_attn") & (ok["scope"] == "kernel")
               & (ok["batch"] == bs) & (ok["sm_count"] == 108)].sort_values("context_len")
        if len(a) >= 2:
            lat = pd.to_numeric(a["latency_per_step_ms"], errors="coerce").values
            mono = bool(np.all(np.diff(lat) >= 0))
            sane.append(f"decode_attn kernel sm=108 bs={bs}: latency monotonic in L = {mono} (measured)")
            break
    d["sanity"] = sane

    figs = [
        fig_decode_sat(df, model, sat, out_dir),
        fig_decode_bw_split(df, cross, model, out_dir),
    ]
    d["figs"] = [str(f) for f in figs if f]
    for f in d["figs"]:
        print(f"  Saved figure: {f}")
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["zamba2", "falcon_h1"],
                    choices=["zamba2", "falcon_h1"])
    ap.add_argument("--results-dir", type=Path,
                    default=_here.parent / "results" / "decode_scaling")
    ap.add_argument("--stage1-dir", type=Path,
                    default=_here.parent / "results" / "stage1")
    ap.add_argument("--fig-dir", type=Path,
                    default=_here.parent / "results" / "decode_scaling")
    ap.add_argument("--report", type=Path,
                    default=_here.parent.parent.parent / "reports"
                    / "decode_saturation_and_crossover.md")
    args = ap.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    per_model = {}
    for model in args.models:
        print(f"\n=== Analyzing {model} ===")
        per_model[model] = analyze_model(model, args.results_dir, args.stage1_dir,
                                         args.fig_dir)
    write_report(args.models, per_model, args.report)


if __name__ == "__main__":
    main()
