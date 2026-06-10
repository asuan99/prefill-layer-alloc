#!/usr/bin/env python
"""
compute_crossover.py — decode KV/weight crossover analysis (Task 0, GPU-free).

Purpose
-------
Quantify, purely from model config, the per-decode-step HBM read traffic of a
single transformer layer and answer two questions for the HM study:

  Q3 가정 (a) — decode_attn vs decode_ssm BW 비대칭 검증의 전제 조건:
    decode_attn 의 KV read 가 weight read 에 묻히지 않을 만큼 context 가 길어야
    한다.  "충분히 길다" 의 경계가 곧 crossover context 길이 L*.

  L* 이 현실적 workload (≤32K) 밖이면 decode-type 축 주장은 조건을 크게 제한
  해야 한다.  L* 이 그 안이면 가정 (a) 는 성립 가능.

모든 수치는 **derived** (config 로부터 계산).  이 Task 에 measured 값은 없다 —
모든 CSV 행에 value_kind=derived 를 명시한다.

가정
----
* bf16 = 2 bytes (BYTES_PER_ELEM).
* decode 1 step, q_len = 1.
* weight_bytes 는 mixer projection 가중치 (decode step 당 HBM read).  batch 와
  무관 (batch 로 amortize 됨).
* Zamba2 의 shared attention block (13 hybrid layer 가 weight 공유) 는
  "메모리 상주" 가 아니라 "decode step 당 read" 가 기준이므로, 공유 여부와
  무관하게 **레이어 실행 횟수만큼** 카운트한다.

per-decode-step, per-layer 읽기 바이트 (mixer 기준):
    weight_bytes(layer_type) = (mixer projection 파라미터 수) × 2
    kv_bytes(L, bs)          = bs × L × n_kv_heads × head_dim × 2(K,V) × 2 bytes
    state_bytes(bs)          = bs × n_heads × head_dim × d_state × 2(R/W) × 2 bytes
    conv_state_bytes(bs)     = bs × d_inner_conv × d_conv × 2 bytes

출처
----
모든 상수는 HF config.json 에서 가져왔다 (hf_cache 스냅샷 경로는 각 ModelSpec
주석 참조).  HF 다운로드가 불가한 환경에서도 재현 가능하도록 하드코딩했다.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display on login/compute node
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BYTES_PER_ELEM = 2  # bf16
VALUE_KIND = "derived"  # this whole Task produces only derived numbers

# Sweep grids (per task spec)
BATCH_GRID = [1, 4, 8, 16, 32]
# L grid for R(L,bs): 512 .. 131072 in log-2 steps
L_GRID = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
R_THRESHOLD = 2.0  # report minimal L where R(L,bs) >= 2.0
WORKLOAD_L_MAX = 32768  # "현실적 workload" 상한 (Task 1 sweep 격자 최대값)


# ---------------------------------------------------------------------------
# Model specification (config-derived; bf16)
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """All config-derived dimensions needed for crossover math.

    Per-layer weight-parameter counts are computed transparently in the
    ``*_mixer_params`` methods below so the derivation is auditable.
    """

    name: str
    hidden_size: int
    num_layers: int

    # Attention branch
    attn_num_heads: int
    attn_num_kv_heads: int
    attn_head_dim: int
    n_attn_layers: int          # how many layer executions read attn weights+KV / step

    # SSM (Mamba-2) branch
    ssm_n_heads: int
    ssm_head_dim: int
    ssm_d_state: int
    ssm_n_groups: int
    ssm_d_conv: int
    n_ssm_layers: int           # how many layer executions read ssm weights+state / step

    # description of layer-count bookkeeping (for the report / provenance)
    layer_note: str = ""

    # ---- derived intermediate dims -------------------------------------

    @property
    def attn_hidden(self) -> int:
        return self.attn_num_heads * self.attn_head_dim

    @property
    def kv_hidden(self) -> int:
        return self.attn_num_kv_heads * self.attn_head_dim

    @property
    def ssm_d_inner(self) -> int:
        return self.ssm_n_heads * self.ssm_head_dim

    @property
    def ssm_conv_channels(self) -> int:
        # depthwise conv1d acts on x + B + C channels
        return self.ssm_d_inner + 2 * self.ssm_n_groups * self.ssm_d_state

    # ---- per-layer mixer parameter counts ------------------------------

    def attn_mixer_params(self) -> int:
        """Q/K/V/O projection parameters for ONE attention-layer execution.

        q: H × attn_hidden,  k: H × kv_hidden,  v: H × kv_hidden,
        o: attn_hidden × H.  Bias-free (config: *_bias=false / add_bias_linear=false).
        """
        H = self.hidden_size
        q = H * self.attn_hidden
        k = H * self.kv_hidden
        v = H * self.kv_hidden
        o = self.attn_hidden * H
        return q + k + v + o

    def ssm_mixer_params(self) -> int:
        """Mamba-2 mixer parameters for ONE ssm-layer execution.

        in_proj : H × (2*d_inner + 2*n_groups*d_state + n_heads)
                  (z, x, B, C, dt)
        conv1d  : conv_channels × d_conv (depthwise; negligible but counted)
        out_proj: d_inner × H
        + A_log, D, dt_bias (n_heads each) — negligible, counted for completeness.
        """
        H = self.hidden_size
        d_inner = self.ssm_d_inner
        in_proj_out = (
            2 * d_inner
            + 2 * self.ssm_n_groups * self.ssm_d_state
            + self.ssm_n_heads
        )
        in_proj = H * in_proj_out
        conv1d = self.ssm_conv_channels * self.ssm_d_conv
        out_proj = d_inner * H
        small = 3 * self.ssm_n_heads  # A_log, D, dt_bias
        return in_proj + conv1d + out_proj + small

    # ---- per-decode-step byte formulas ---------------------------------

    def attn_weight_bytes(self) -> int:
        return self.attn_mixer_params() * BYTES_PER_ELEM

    def ssm_weight_bytes(self) -> int:
        return self.ssm_mixer_params() * BYTES_PER_ELEM

    def kv_bytes(self, L: int, bs: int) -> int:
        """KV-cache read for one attn layer: bs × L × n_kv_heads × head_dim × 2(K,V) × 2B."""
        return bs * L * self.attn_num_kv_heads * self.attn_head_dim * 2 * BYTES_PER_ELEM

    def state_bytes(self, bs: int) -> int:
        """SSM recurrent-state read+write: bs × n_heads × head_dim × d_state × 2 × 2B (L-무관)."""
        return (
            bs
            * self.ssm_n_heads
            * self.ssm_head_dim
            * self.ssm_d_state
            * 2
            * BYTES_PER_ELEM
        )

    def conv_state_bytes(self, bs: int) -> int:
        """SSM conv-state read: bs × conv_channels × d_conv × 2B (L-무관)."""
        return bs * self.ssm_conv_channels * self.ssm_d_conv * BYTES_PER_ELEM


# ---------------------------------------------------------------------------
# Model registry (hardcoded from HF config.json; sources noted)
# ---------------------------------------------------------------------------

# Source: hf_cache/hub/models--Zyphra--Zamba2-7B-Instruct/.../config.json
#   hidden_size=3584, num_hidden_layers=81,
#   attention_head_dim=224, num_attention_heads=32, num_key_value_heads=32,
#   n_mamba_heads=112, mamba_headdim=64, mamba_d_state=64, mamba_ngroups=2,
#   mamba_d_conv=4, hybrid_layer_ids=[13 entries].
# Zamba2: every layer carries a Mamba mixer (81); the shared attention block is
# executed only in the 13 hybrid layers.  Per task: count per-execution, so
# n_attn_layers = 13 (shared weights re-read each execution), n_ssm_layers = 81.
ZAMBA2 = ModelSpec(
    name="zamba2",
    hidden_size=3584,
    num_layers=81,
    attn_num_heads=32,
    attn_num_kv_heads=32,
    attn_head_dim=224,
    n_attn_layers=13,
    ssm_n_heads=112,
    ssm_head_dim=64,
    ssm_d_state=64,
    ssm_n_groups=2,
    ssm_d_conv=4,
    n_ssm_layers=81,
    layer_note="81 Mamba mixers; 13 shared-attn executions (counted per-execution).",
)

# Source: hf_cache/hub/models--tiiuae--Falcon-H1-7B-Instruct/.../config.json
#   hidden_size=3072, num_hidden_layers=44,
#   head_dim=128, num_attention_heads=12, num_key_value_heads=2 (GQA),
#   mamba_n_heads=24, mamba_d_head=128, mamba_d_state=256, mamba_n_groups=1,
#   mamba_d_conv=4, mamba_d_ssm=3072.
# Falcon-H1: all 44 layers run attn + ssm branches in parallel → both counts = 44.
FALCON_H1 = ModelSpec(
    name="falcon_h1",
    hidden_size=3072,
    num_layers=44,
    attn_num_heads=12,
    attn_num_kv_heads=2,
    attn_head_dim=128,
    n_attn_layers=44,
    ssm_n_heads=24,
    ssm_head_dim=128,
    ssm_d_state=256,
    ssm_n_groups=1,
    ssm_d_conv=4,
    n_ssm_layers=44,
    layer_note="44 parallel attn+ssm layers; both branches in every layer.",
)

MODELS = {"zamba2": ZAMBA2, "falcon_h1": FALCON_H1}


# ---------------------------------------------------------------------------
# Crossover computations
# ---------------------------------------------------------------------------


def layer_crossover_L(spec: ModelSpec, bs: int) -> float:
    """Solve kv_bytes(L, bs) = attn_weight_bytes  →  L*(bs).

    L* = W_attn / (bs × n_kv_heads × head_dim × 2 × 2B)
    """
    per_token_per_batch = (
        spec.attn_num_kv_heads * spec.attn_head_dim * 2 * BYTES_PER_ELEM
    )
    return spec.attn_weight_bytes() / (bs * per_token_per_batch)


def forward_ratio_R(spec: ModelSpec, L: int, bs: int) -> float:
    """R(L,bs) = Σ_attn (weight+kv) / Σ_ssm (weight+state+conv_state).

    Mixer-level: weight = mixer projection weights; ssm state includes recurrent
    state + conv state (both L-independent).
    """
    attn_numer = spec.n_attn_layers * (spec.attn_weight_bytes() + spec.kv_bytes(L, bs))
    ssm_denom = spec.n_ssm_layers * (
        spec.ssm_weight_bytes() + spec.state_bytes(bs) + spec.conv_state_bytes(bs)
    )
    return attn_numer / ssm_denom


def min_L_for_R(spec: ModelSpec, bs: int, threshold: float) -> float | None:
    """Smallest L (continuous) with R(L,bs) >= threshold, or None if unreachable.

    R is monotone increasing in L (KV grows, everything else fixed).  Solve
    analytically:
        threshold = n_attn*(W_a + kv(L)) / D_ssm
        kv(L) = bs*L*kv_per_token  →  L = (threshold*D_ssm/n_attn - W_a) / (bs*kv_per_token)
    """
    W_a = spec.attn_weight_bytes()
    D_ssm = spec.n_ssm_layers * (
        spec.ssm_weight_bytes() + spec.state_bytes(bs) + spec.conv_state_bytes(bs)
    )
    kv_per_token = spec.attn_num_kv_heads * spec.attn_head_dim * 2 * BYTES_PER_ELEM
    target_kv = threshold * D_ssm / spec.n_attn_layers - W_a
    if target_kv <= 0:
        return 0.0  # already >= threshold at L=0
    L = target_kv / (bs * kv_per_token)
    return L


def mixer_only_bytes_ratio(spec: ModelSpec, L: int, bs: int) -> float:
    """decode_attn vs decode_ssm per-layer mixer-only byte ratio (weight excluded).

    kv_bytes(L,bs) / (state_bytes(bs) + conv_state_bytes(bs))
    """
    ssm_movement = spec.state_bytes(bs) + spec.conv_state_bytes(bs)
    return spec.kv_bytes(L, bs) / ssm_movement


def per_layer_bytes_ratio_with_weight(spec: ModelSpec, L: int, bs: int) -> float:
    """decode_attn vs decode_ssm per-layer byte ratio INCLUDING mixer weights."""
    attn = spec.attn_weight_bytes() + spec.kv_bytes(L, bs)
    ssm = spec.ssm_weight_bytes() + spec.state_bytes(bs) + spec.conv_state_bytes(bs)
    return attn / ssm


# ---------------------------------------------------------------------------
# CSV emission
# ---------------------------------------------------------------------------


def write_csv(spec: ModelSpec, out_dir: Path) -> Path:
    """Emit crossover_{model}.csv with all derived rows (one schema, tagged metric)."""
    path = out_dir / f"crossover_{spec.name}.csv"
    rows: list[dict] = []

    # --- block 1: layer-unit crossover L*(bs) ---
    for bs in BATCH_GRID:
        Lstar = layer_crossover_L(spec, bs)
        rows.append(
            {
                "metric": "layer_crossover_Lstar",
                "batch": bs,
                "context_len": "",
                "value": round(Lstar, 2),
                "unit": "tokens",
                "value_kind": VALUE_KIND,
                "note": "kv_bytes(L,bs)==attn_mixer_weight_bytes",
            }
        )

    # --- block 2: forward-unit BW asymmetry R(L,bs) ---
    for bs in BATCH_GRID:
        for L in L_GRID:
            rows.append(
                {
                    "metric": "forward_ratio_R",
                    "batch": bs,
                    "context_len": L,
                    "value": round(forward_ratio_R(spec, L, bs), 6),
                    "unit": "ratio_attn_over_ssm_bytes",
                    "value_kind": VALUE_KIND,
                    "note": "sum_attn(weight+kv)/sum_ssm(weight+state+conv)",
                }
            )

    # --- block 3a: per-layer mixer-only ratio (weight excluded) ---
    for bs in BATCH_GRID:
        for L in L_GRID:
            rows.append(
                {
                    "metric": "mixer_only_kv_over_state",
                    "batch": bs,
                    "context_len": L,
                    "value": round(mixer_only_bytes_ratio(spec, L, bs), 6),
                    "unit": "ratio_kvbytes_over_statebytes",
                    "value_kind": VALUE_KIND,
                    "note": "kv_bytes/(state_bytes+conv_state_bytes); weight excluded",
                }
            )

    # --- block 3b: per-layer ratio WITH weight ---
    for bs in BATCH_GRID:
        for L in L_GRID:
            rows.append(
                {
                    "metric": "per_layer_ratio_with_weight",
                    "batch": bs,
                    "context_len": L,
                    "value": round(per_layer_bytes_ratio_with_weight(spec, L, bs), 6),
                    "unit": "ratio_attn_over_ssm_bytes",
                    "value_kind": VALUE_KIND,
                    "note": "(attn_w+kv)/(ssm_w+state+conv)",
                }
            )

    # --- block 4: judgment outputs (L*, min L for R>=2) ---
    for bs in BATCH_GRID:
        rows.append(
            {
                "metric": "judgment_Lstar_layer",
                "batch": bs,
                "context_len": "",
                "value": round(layer_crossover_L(spec, bs), 2),
                "unit": "tokens",
                "value_kind": VALUE_KIND,
                "note": f"in_workload(<= {WORKLOAD_L_MAX})="
                + str(layer_crossover_L(spec, bs) <= WORKLOAD_L_MAX),
            }
        )
        minL = min_L_for_R(spec, bs, R_THRESHOLD)
        rows.append(
            {
                "metric": "judgment_minL_R_ge_2",
                "batch": bs,
                "context_len": "",
                "value": round(minL, 2) if minL is not None else "",
                "unit": "tokens",
                "value_kind": VALUE_KIND,
                "note": f"smallest L with R>= {R_THRESHOLD}; "
                + (
                    f"in_workload={minL <= WORKLOAD_L_MAX}"
                    if minL is not None
                    else "unreachable"
                ),
            }
        )

    fieldnames = [
        "metric",
        "batch",
        "context_len",
        "value",
        "unit",
        "value_kind",
        "note",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_plot(spec: ModelSpec, out_dir: Path) -> Path:
    """Three-panel figure: L*(bs), R(L,bs), mixer-only ratio."""
    path = out_dir / f"crossover_{spec.name}.png"
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(
        f"Decode KV/weight crossover — {spec.name}  (derived, bf16)", fontsize=13
    )

    # Panel 1: L*(bs)
    ax = axes[0]
    Lstars = [layer_crossover_L(spec, bs) for bs in BATCH_GRID]
    ax.plot(BATCH_GRID, Lstars, "o-", color="tab:blue")
    ax.axhline(
        WORKLOAD_L_MAX,
        ls="--",
        color="tab:red",
        label=f"workload max L={WORKLOAD_L_MAX}",
    )
    for bs, Ls in zip(BATCH_GRID, Lstars):
        ax.annotate(f"{Ls:.0f}", (bs, Ls), textcoords="offset points", xytext=(0, 6),
                    fontsize=8, ha="center")
    ax.set_xlabel("batch size")
    ax.set_ylabel("L*  (context tokens)")
    ax.set_title("Layer-unit crossover L*\n(KV read == attn mixer weight read)")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(BATCH_GRID)
    ax.set_xticklabels(BATCH_GRID)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 2: R(L,bs)
    ax = axes[1]
    for bs in BATCH_GRID:
        ys = [forward_ratio_R(spec, L, bs) for L in L_GRID]
        ax.plot(L_GRID, ys, "o-", label=f"bs={bs}", markersize=3)
    ax.axhline(R_THRESHOLD, ls="--", color="k", alpha=0.6, label=f"R={R_THRESHOLD}")
    ax.axvline(WORKLOAD_L_MAX, ls=":", color="tab:red", alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length L")
    ax.set_ylabel("R = Σattn(w+kv) / Σssm(w+state)")
    ax.set_title("Forward-unit BW asymmetry R(L,bs)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 3: mixer-only ratio (kv / state)
    ax = axes[2]
    for bs in BATCH_GRID:
        ys = [mixer_only_bytes_ratio(spec, L, bs) for L in L_GRID]
        ax.plot(L_GRID, ys, "o-", label=f"bs={bs}", markersize=3)
    ax.axhline(1.0, ls="--", color="k", alpha=0.6, label="ratio=1")
    ax.axvline(WORKLOAD_L_MAX, ls=":", color="tab:red", alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("context length L")
    ax.set_ylabel("kv_bytes / state_bytes  (mixer-only)")
    ax.set_title("decode_attn vs decode_ssm\nmixer-only byte ratio (weight excluded)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Stdout judgment report
# ---------------------------------------------------------------------------


def print_judgment(spec: ModelSpec) -> None:
    print(f"\n{'='*72}")
    print(f"  CROSSOVER JUDGMENT — {spec.name}   (all values derived, bf16)")
    print(f"  {spec.layer_note}")
    print(f"{'='*72}")
    print(
        f"  attn mixer weight = {spec.attn_weight_bytes()/1e6:8.2f} MB"
        f"   ({spec.attn_mixer_params():,} params)"
    )
    print(
        f"  ssm  mixer weight = {spec.ssm_weight_bytes()/1e6:8.2f} MB"
        f"   ({spec.ssm_mixer_params():,} params)"
    )
    print(
        f"  KV per (token,batch) = {spec.attn_num_kv_heads*spec.attn_head_dim*2*BYTES_PER_ELEM:,} B"
        f"   (n_kv_heads={spec.attn_num_kv_heads}, head_dim={spec.attn_head_dim})"
    )
    print(
        f"  SSM state per batch  = {spec.state_bytes(1):,} B"
        f"   (+ conv {spec.conv_state_bytes(1):,} B)"
    )
    print()
    print(f"  {'bs':>4} | {'L* (tokens)':>14} | in_workload? | {'minL R>=2':>12} | in_wl?")
    print(f"  {'-'*4}-+-{'-'*14}-+--------------+-{'-'*12}-+-------")
    for bs in BATCH_GRID:
        Lstar = layer_crossover_L(spec, bs)
        minL = min_L_for_R(spec, bs, R_THRESHOLD)
        in_wl_star = "yes" if Lstar <= WORKLOAD_L_MAX else "NO (>32K)"
        if minL is None:
            minL_s, in_wl_R = "unreachable", "-"
        else:
            minL_s = f"{minL:.0f}"
            in_wl_R = "yes" if minL <= WORKLOAD_L_MAX else "NO (>32K)"
        print(f"  {bs:>4} | {Lstar:>14.1f} | {in_wl_star:<12} | {minL_s:>12} | {in_wl_R}")
    print(f"{'='*72}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["zamba2", "falcon_h1"],
                    choices=list(MODELS))
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.models:
        spec = MODELS[name]
        csv_path = write_csv(spec, args.out_dir)
        png_path = make_plot(spec, args.out_dir)
        print_judgment(spec)
        print(f"  → CSV : {csv_path}")
        print(f"  → PNG : {png_path}")


if __name__ == "__main__":
    main()
