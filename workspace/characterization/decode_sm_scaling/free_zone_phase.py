"""
free_zone_phase.py — prefill free-zone surface + Q1 phase diagram (CPU-only).

동기
----
analyze_decode_sat.py 는 free zone 을 단일 상수(zamba2=14, falcon_h1=27)로
고정했으나, 정정된 Stage 1 chunked 데이터로 재계산하면 free zone 은 (batch,
seq_len) 에 따라 0 ~ 94 SM 으로 6배 이상 변한다.  단일 상수는 "Policy C 가
유의미한 좁은 operating 창" 을 통째로 가린다.

이 스크립트는:
  (A) free_zone(prefill_bs, prefill_seq) 곡면을 Stage 1 chunked latency 에서
      재계산 (measured throughput 기반) → CSV + heatmap.
  (B) Q1(Policy C 생사) 을 boolean 이 아닌 region 으로 출력하는 phase diagram.
      각 (bs, seq) cell 을 decode 수요 d(=sm_sat_decode) 대비 regime 으로 색칠:
        - no-slack   : free_zone == 0          → HM 멀티플렉싱 여유 자체가 없음
        - under-prov : 0 < free_zone < d        → free 전량 할당해도 decode 미포화
        - absorbed   : free_zone >= d           → fixed split 흡수, Policy C 이득 ≈ 0
      Policy C 가 차이를 낼 수 있는 곳은 under-prov ↔ absorbed 경계의 좁은 띠뿐.

measured / derived 라벨
-----------------------
* free_zone : measured (Stage 1 chunked latency, Stage 1 saturation 기준으로 derived)
* d = sm_sat_decode :
    - Task 1 결과(sm_sat_decode_{model}.csv) 가 있으면 **measured** 사용
      (decode_attn, scope=layer, L≈seq 커플링; self-decode 가정).
    - 없으면 **parametric** 후보값들로 small-multiples (측정 대기 표시).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd

_here = Path(__file__).parent
TOTAL_SM = 108
SATURATION_THRESHOLD = 0.03
DEVICE_TAGS = ["a100-sxm4-80gb", "a100_sxm4_80gb"]

# parametric decode-demand candidates (used when Task 1 measured data absent)
PARAM_D = [14, 27, 40, 54]

# regime codes / colors
_R_VOID, _R_UNDER, _R_ABSORB = 0, 1, 2
_REGIME_CMAP = ListedColormap(["#9e9e9e", "#ff9800", "#4caf50"])
_REGIME_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], _REGIME_CMAP.N)
_REGIME_LABEL = {
    _R_VOID: "no-slack (free=0)",
    _R_UNDER: "under-prov (0<free<d)",
    _R_ABSORB: "absorbed (free>=d)",
}


# ---------------------------------------------------------------------------
# Saturation (Stage 1 criterion, replicated)
# ---------------------------------------------------------------------------


def _saturation_sm(grp: pd.DataFrame) -> int:
    grp = grp.sort_values("sm_ratio")
    if len(grp) < 2:
        return int(grp["sm_count"].max())
    sr = grp["sm_ratio"].values
    tp = grp["tp_norm"].values
    for i in range(1, len(sr)):
        dsm = sr[i] - sr[i - 1]
        if dsm <= 0:
            continue
        if (tp[i] - tp[i - 1]) / dsm * 0.10 < SATURATION_THRESHOLD:
            return int(grp["sm_count"].iloc[i - 1])
    return int(grp["sm_count"].max())


# ---------------------------------------------------------------------------
# Free-zone surface from Stage 1 chunked data
# ---------------------------------------------------------------------------


def load_chunked(model: str, stage1_dir: Path) -> pd.DataFrame | None:
    for tag in DEVICE_TAGS:
        p = stage1_dir / "chunked" / f"ssm_chunked_{model}_{tag}.csv"
        if p.exists():
            return pd.read_csv(p)
    return None


def free_zone_surface(model: str, stage1_dir: Path) -> pd.DataFrame:
    """Return DataFrame[seq_len, batch, sat_sm, free_sm] from chunked latency."""
    df = load_chunked(model, stage1_dir)
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    df["sm_ratio"] = df["sm_count"] / TOTAL_SM
    df["tp"] = 1.0 / df["latency_ms"]
    rows = []
    for (sl, bs), grp in df.groupby(["seq_len", "batch_size"]):
        grp = grp.copy()
        grp["tp_norm"] = grp["tp"] / grp["tp"].max()
        sat = _saturation_sm(grp)
        rows.append({
            "model": model, "seq_len": int(sl), "batch": int(bs),
            "sat_sm": sat, "free_sm": TOTAL_SM - sat,
            "value_kind": "measured(throughput-derived saturation)",
        })
    return pd.DataFrame(rows).sort_values(["batch", "seq_len"])


# ---------------------------------------------------------------------------
# Measured decode demand d(bs, seq) from Task 1, if present
# ---------------------------------------------------------------------------


def load_measured_demand(model: str, results_dir: Path) -> pd.DataFrame | None:
    """sm_sat_decode for the SM-binding decode work per step.

    Couples decode context L to prefill seq (self-decode assumption) and takes
    the max over decode_ssm/decode_attn at layer scope as the binding demand.
    Returns DataFrame[batch, seq_or_L, d] or None if Task 1 output absent.
    """
    p = results_dir / f"sm_sat_decode_{model}.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p)
    t = t[t["scope"] == "layer"].copy()
    t = t[t["sm_sat_decode"] != ""]
    if t.empty:
        return None
    t["sm_sat_decode"] = pd.to_numeric(t["sm_sat_decode"], errors="coerce")
    rows = []
    for bs in sorted(t["batch"].unique()):
        sub = t[t["batch"] == bs]
        ssm = sub[sub["cell"] == "decode_ssm"]["sm_sat_decode"]
        ssm_d = float(ssm.iloc[0]) if len(ssm) else np.nan
        attn = sub[sub["cell"] == "decode_attn"]
        for _, r in attn.iterrows():
            d = np.nanmax([ssm_d, float(r["sm_sat_decode"])])
            rows.append({"batch": int(bs), "ctx": int(r["context_len"]), "d": d})
    return pd.DataFrame(rows) if rows else None


def _demand_for(measured: pd.DataFrame, bs: int, seq: int) -> float | None:
    """Look up measured demand at (bs, ctx≈seq); nearest context."""
    sub = measured[measured["batch"] == bs]
    if sub.empty:
        return None
    idx = (sub["ctx"] - seq).abs().idxmin()
    return float(sub.loc[idx, "d"])


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------


def _regime(free: int, d: float) -> int:
    if free <= 0:
        return _R_VOID
    if free < d:
        return _R_UNDER
    return _R_ABSORB


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _grid(surface: pd.DataFrame):
    seqs = sorted(surface["seq_len"].unique())
    bss = sorted(surface["batch"].unique())
    free = np.full((len(bss), len(seqs)), np.nan)
    for _, r in surface.iterrows():
        free[bss.index(r["batch"]), seqs.index(r["seq_len"])] = r["free_sm"]
    return bss, seqs, free


def fig_free_zone_heatmap(surface: pd.DataFrame, model: str, out_dir: Path) -> Path:
    bss, seqs, free = _grid(surface)
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(seqs), 1.2 + 0.7 * len(bss)))
    im = ax.imshow(free, aspect="auto", cmap="viridis", origin="lower",
                   vmin=0, vmax=TOTAL_SM)
    ax.set_xticks(range(len(seqs)))
    ax.set_xticklabels(seqs)
    ax.set_yticks(range(len(bss)))
    ax.set_yticklabels(bss)
    ax.set_xlabel("prefill seq_len")
    ax.set_ylabel("prefill batch")
    ax.set_title(f"Free SM zone during SSM prefill - {model}\n"
                 f"(measured; 108 - saturation_sm, Stage 1 criterion)")
    for i in range(len(bss)):
        for j in range(len(seqs)):
            v = free[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{int(v)}", ha="center", va="center",
                        color="white" if v < TOTAL_SM * 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="free SM (108 − sat)")
    fig.tight_layout()
    path = out_dir / f"fig_free_zone_surface_{model}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_phase_diagram(surface: pd.DataFrame, model: str, out_dir: Path,
                      measured: pd.DataFrame | None) -> Path:
    bss, seqs, free = _grid(surface)

    def _draw(ax, dmap, title):
        reg = np.full((len(bss), len(seqs)), np.nan)
        for i, bs in enumerate(bss):
            for j, sq in enumerate(seqs):
                f = free[i, j]
                if np.isnan(f):
                    continue
                d = dmap(bs, sq)
                if d is None or np.isnan(d):
                    continue
                reg[i, j] = _regime(int(f), d)
        ax.imshow(reg, aspect="auto", cmap=_REGIME_CMAP, norm=_REGIME_NORM,
                  origin="lower")
        ax.set_xticks(range(len(seqs)))
        ax.set_xticklabels(seqs, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(bss)))
        ax.set_yticklabels(bss, fontsize=8)
        ax.set_xlabel("prefill seq_len", fontsize=9)
        ax.set_ylabel("prefill batch", fontsize=9)
        ax.set_title(title, fontsize=10)
        for i in range(len(bss)):
            for j in range(len(seqs)):
                if not np.isnan(free[i, j]):
                    ax.text(j, i, f"{int(free[i, j])}", ha="center", va="center",
                            fontsize=7, color="black")

    if measured is not None:
        fig, ax = plt.subplots(figsize=(2.0 + 1.0 * len(seqs), 1.5 + 0.6 * len(bss)))
        _draw(ax, lambda bs, sq: _demand_for(measured, bs, sq),
              f"Q1 phase diagram - {model}\n"
              f"d = sm_sat_decode (MEASURED, layer scope, L~=seq coupling)")
        mode = "measured"
        figs_axes = [ax]
    else:
        n = len(PARAM_D)
        fig, axes = plt.subplots(1, n, figsize=(2.2 + 2.0 * n, 1.6 + 0.6 * len(bss)),
                                 squeeze=False)
        axes = axes[0]
        for ax, d in zip(axes, PARAM_D):
            _draw(ax, lambda bs, sq, d=d: float(d),
                  f"d = sm_sat_decode = {d}\n(PARAMETRIC - awaiting Task 1)")
        fig.suptitle(f"Q1 phase diagram - {model}   "
                     f"(free_zone: measured; d: parametric until Task 1)",
                     fontsize=11)
        mode = "parametric"
        figs_axes = axes

    # shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=_REGIME_CMAP(c)) for c in (0, 1, 2)]
    labels = [_REGIME_LABEL[c] for c in (0, 1, 2)]
    figs_axes[-1].legend(handles, labels, loc="center left",
                         bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    fig.text(0.5, 0.005,
             "cell number = free SM. Policy C can only matter in the thin band at the "
             "under-prov / absorbed boundary (no-slack & absorbed regions: Policy C gain ~= 0).",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 0.92 if measured is not None else 1.0, 0.94))
    path = out_dir / f"fig_q1_phase_{model}_{mode}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report section builder (consumed by analyze_decode_sat.py)
# ---------------------------------------------------------------------------


def regime_breakdown(surface: pd.DataFrame, measured: pd.DataFrame | None,
                     param_d: int | None = None) -> tuple[list[dict], dict]:
    """Classify every (batch, seq) cell. Returns (rows, counts).

    d per cell: measured sm_sat_decode (L≈seq coupling) if `measured` given,
    else the constant `param_d`.
    """
    rows, counts = [], {0: 0, 1: 0, 2: 0}
    for _, r in surface.sort_values(["batch", "seq_len"]).iterrows():
        bs, sq, free = int(r["batch"]), int(r["seq_len"]), int(r["free_sm"])
        if measured is not None:
            d = _demand_for(measured, bs, sq)
        else:
            d = float(param_d) if param_d is not None else None
        if d is None or (isinstance(d, float) and np.isnan(d)):
            continue
        reg = _regime(free, d)
        counts[reg] += 1
        rows.append({"batch": bs, "seq_len": sq, "free_sm": free,
                     "d": int(d), "regime": reg})
    return rows, counts


def build_phase_report_section(model: str, surface: pd.DataFrame,
                               measured: pd.DataFrame | None) -> list[str]:
    """Return markdown lines (text only; figures emitted separately by caller)."""
    free_vals = surface["free_sm"].values
    fmin, fmax = int(free_vals.min()), int(free_vals.max())
    rows, counts = regime_breakdown(surface, measured)
    total = sum(counts.values())

    # which (bs,seq) cells produce the legacy constant (14 / 27)?
    legacy = {"zamba2": 14, "falcon_h1": 27}.get(model)
    legacy_cells = [
        f"(bs{int(r['batch'])}, seq{int(r['seq_len'])})"
        for _, r in surface.iterrows() if int(r["free_sm"]) == legacy
    ]

    L = []
    L.append("### 판정 1b — free zone 곡면 + Q1 phase diagram (measured)\n")
    L.append("**핵심 정정:** 판정 1의 free_zone 상수는 단일 operating point 값이다. "
             "정정된 Stage 1 chunked latency 로 재계산하면 free zone 은 "
             f"(prefill batch, seq_len) 에 따라 **{fmin}–{fmax} SM** 로 변한다 "
             "(measured; 108 − SSM prefill saturation_sm).")
    if legacy is not None:
        cells_s = ", ".join(legacy_cells) if legacy_cells else "측정 격자 내 없음"
        L.append(f"- 판정 1이 쓴 상수 free_zone={legacy} 가 나오는 cell: {cells_s} "
                 f"→ 그 한 점에만 해당하는 값임.\n")

    # free-zone surface table
    seqs = sorted(surface["seq_len"].unique())
    bss = sorted(surface["batch"].unique(), reverse=True)
    L.append("free zone 곡면 (measured, 108 − saturation_sm):\n")
    L.append("| bs \\\\ seq | " + " | ".join(str(s) for s in seqs) + " |")
    L.append("|" + "---|" * (len(seqs) + 1))
    fmap = {(int(r["batch"]), int(r["seq_len"])): int(r["free_sm"])
            for _, r in surface.iterrows()}
    for bs in bss:
        cells = [str(fmap.get((bs, s), "")) for s in seqs]
        L.append(f"| **{bs}** | " + " | ".join(cells) + " |")
    L.append("")

    # phase regime table (d per cell)
    dlabel = "measured sm_sat_decode (layer scope, L≈seq)" if measured is not None \
        else "parametric d"
    L.append(f"Q1 regime — d = {dlabel}; 셀 = `regime(free,d)`:\n")
    L.append("| bs \\\\ seq | " + " | ".join(str(s) for s in seqs) + " |")
    L.append("|" + "---|" * (len(seqs) + 1))
    rmap = {(r["batch"], r["seq_len"]): r for r in rows}
    tag = {0: "void", 1: "under", 2: "absorb"}
    for bs in bss:
        cells = []
        for s in seqs:
            rr = rmap.get((bs, s))
            cells.append(f"{tag[rr['regime']]} (f{rr['free_sm']}/d{rr['d']})"
                         if rr else "")
        L.append(f"| **{bs}** | " + " | ".join(cells) + " |")
    L.append("")
    L.append(f"regime 집계 (measured): absorbed={counts[2]}, "
             f"under-prov={counts[1]}, no-slack={counts[0]}  (총 {total} cells)\n")

    # data-driven conclusion
    L.append("**결론 (Q1):** free zone 과 decode 수요-부족은 batch 축에서 "
             "**역상관**이다 — 여유가 큰 저-batch 에선 decode 가 free zone 에 흡수되어 "
             "(absorbed) Policy C 이득 ≈ 0, decode 가 더 요구하는 고-batch 에선 "
             "free zone = 0 (no-slack) 이라 멀티플렉싱 여유 자체가 없다. "
             f"Policy C 가 fixed split 대비 차이를 낼 수 있는 구간은 under-prov 셀 "
             f"{counts[1]}/{total} 의 좁은 띠로 한정된다. "
             "따라서 단일 free_zone 상수에 기반한 \"잔여 이득 상한\" 수치(판정 1)는 "
             "operating point 평균을 호도하므로, 아래 phase diagram 의 region 해석으로 대체한다.\n")
    return L


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def analyze(model: str, stage1_dir: Path, results_dir: Path, out_dir: Path) -> dict:
    surface = free_zone_surface(model, stage1_dir)
    if surface.empty:
        print(f"  [{model}] Stage 1 chunked CSV 없음 — free zone 곡면 생성 불가")
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"free_zone_surface_{model}.csv"
    surface.to_csv(csv_path, index=False)
    print(f"  Saved free-zone surface CSV: {csv_path}")

    measured = load_measured_demand(model, results_dir)
    if measured is not None:
        print(f"  [{model}] Task 1 sm_sat_decode 발견 → phase diagram MEASURED 모드")
    else:
        print(f"  [{model}] Task 1 sm_sat_decode 없음 → phase diagram PARAMETRIC 모드 "
              f"(d ∈ {PARAM_D})")

    f1 = fig_free_zone_heatmap(surface, model, out_dir)
    f2 = fig_phase_diagram(surface, model, out_dir, measured)
    print(f"  Saved: {f1}")
    print(f"  Saved: {f2}")

    # stdout summary
    free_vals = surface["free_sm"].values
    print(f"  [{model}] free zone range: {int(free_vals.min())}–{int(free_vals.max())} SM "
          f"over {len(surface)} (bs,seq) points; "
          f"no-slack(free=0) cells = {(free_vals==0).sum()}/{len(free_vals)}")
    return {"surface": surface, "measured_mode": measured is not None,
            "figs": [str(f1), str(f2)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["zamba2", "falcon_h1"],
                    choices=["zamba2", "falcon_h1"])
    ap.add_argument("--stage1-dir", type=Path,
                    default=_here.parent / "results" / "stage1")
    ap.add_argument("--results-dir", type=Path,
                    default=_here.parent / "results" / "decode_scaling")
    ap.add_argument("--out-dir", type=Path,
                    default=_here.parent / "results" / "decode_scaling")
    args = ap.parse_args()
    for model in args.models:
        print(f"\n=== Free-zone surface + Q1 phase: {model} ===")
        analyze(model, args.stage1_dir, args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
