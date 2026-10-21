#!/usr/bin/env python3
"""지금 보고 가능한 결과 그림 6장 (2026-10-21). 새 판정 없음 — 값은 정본·원자료에서 그대로 가져온다.

출처는 그림마다 하단 주석과 SOURCES에 둔다. 실행: python3 make_current_figs.py → current_2026-10-21/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
R = REPO / "workspace/engine-port/results"
OUT = HERE / "current_2026-10-21"
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

# ---- 1. Claim G: P1-opint (CONSENSUS §1-1; P1_OPINT_RESULT_2026-08-05.md §3.1·§3.2 정본 p95 지표) ----
P1 = {  # rate: (fused mean, fused SD, PD-mux mean, PD-mux SD, cliff)
    "Zamba2-2.7B": {2: (1.314, .056, 1.847, .159, 0), 3: (1.032, .268, 2.949, .354, 0),
                    4: (.562, .373, 3.376, .729, 1), 6: (.221, .339, 1.621, .665, 1)},
    "Granite-4.0-h-micro": {2: (1.904, .256, 1.968, .369, 0), 3: (2.619, .258, 2.924, .253, 0),
                            4: (3.103, .304, 3.943, .292, 0), 6: (2.021, .692, 5.557, .597, 0)},
}
P1_HEAD = {("Zamba2-2.7B", 2): "+40.5%", ("Zamba2-2.7B", 3): "+185.8%", ("Granite-4.0-h-micro", 4): "+27.0%"}

# ---- 2. HE0 (CONSENSUS §1-7, §1-17) ----
HE0 = [("d44", 3.220, .013), ("d34", 3.171, .025), ("bind+GATE", 3.132, .019), ("d24", 3.039, .130),
       ("SLO-aware", 2.964, .025), ("bind", 2.934, .306), ("d16", 2.846, .055)]
TIGHT = [("d44", 73.2, 4.8), ("d34", 49.6, 3.9), ("bind+GATE", 44.3, 2.7), ("bind", 40.6, 0.4)]

# ---- 3. layer-type (CONSENSUS §1-3) ----
LT = [("층 경계 전환 없음", 42, ""), ("층 경계마다\nSM 전환", 124, ""), ("전환 최적화 후", 85, "약 ")]

# ---- 4. CM-2 floor (M3 J1 FLOOR.json; 감사 AUDIT_J1_N1_analysis_2026-10-19.md) ----
J1 = R / "multigpu/manual_retrieval/pdmux-multigpu-2106f42-20261018074950/results/multigpu/m3_pdmux-multigpu-2106f42-20261018074950/boots"

# ---- 5. CM-5 요청당 메모리 (부팅 로그 실측 s·k; COSTMODEL CM-5) ----
S_BYTES, K_BYTES = 143.5e6, 16382

# ---- 6. 계측 관측자 효과 (DECOMP_RESULT.json; 감사 r1 §1) ----
DECOMP = R / "engine_gate/pdmux-engine-gate-decomp-d1f2ddb-20261021022225/DECOMP_RESULT.json"


def foot(fig, text, c):
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=7.5, color=c["muted"], wrap=True)


def fig_p1(c):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    w = 0.38
    for ax, (model, d) in zip(axes, P1.items()):
        rates = sorted(d)
        x = np.arange(len(rates))
        for j, (lab, col, i) in enumerate([("fused (미조율)", c["series"][1], 0), ("PD-mux (agnostic)", c["series"][0], 2)]):
            m = [d[r][i] for r in rates]
            sd = [d[r][i + 1] for r in rates]
            bars = ax.bar(x + (j - .5) * w, m, w * 0.94, yerr=sd, color=col, label=lab,
                          error_kw=dict(ecolor=c["secondary"], lw=1, capsize=3))
            for k, r in enumerate(rates):
                if d[r][4]:
                    bars[k].set_hatch("///")
                    bars[k].set_alpha(0.55)
        for k, r in enumerate(rates):
            h = P1_HEAD.get((model, r))
            if h:
                top = max(d[r][0] + d[r][1], d[r][2] + d[r][3])
                ax.text(x[k], top + 0.12, h, ha="center", fontsize=10, fontweight="bold", color=c["ink"])
        ax.set_xticks(x, [f"rate {r}" + (" ⚠" if d[r][4] else "") for r in rates])
        ax.set_title(model, fontsize=11)
        ax.set_ylabel("goodput (req/s)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Claim G — PD-mux 활성화의 꼬리 SLO goodput (A100, cudagraph-ON, n=5)", fontsize=12, x=0.01, ha="left")
    foot(fig, "goodput = TTFT ≤ 3 s ∧ 요청 내부 token-ITL p95 ≤ 60 ms. 굵은 숫자 = 정본 headline 셀. ⚠/빗금 = TTFT 절벽 위 셀(headline 아님). "
              "분모는 미조율 fused, --enable-pdmux 3-플래그 묶음 처치 → 기전 귀속 미확립.\n"
              "출처: P1_OPINT_RESULT_2026-08-05.md §3.1–3.2, CONSENSUS §1-1. 등급: 부분 지지(모델·워크로드 한정).", c)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    return fig


def fig_he0(c):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1.6, 1]))
    static = {"d44", "d34", "d24", "d16"}
    for ax, data, yl, ttl in [(a1, HE0, "goodput (req/s)", "관대 SLO (3 s / 60 ms), 변화 trace rate 3↔12"),
                              (a2, TIGHT, "SLO attainment (%)", "tight SLO (chat 300 / 50 ms), rate 8")]:
        names = [n for n, _, _ in data]
        cols = [c["series"][0] if n in static else c["series"][1] for n in names]
        ax.bar(names, [m for _, m, _ in data], 0.6, yerr=[s for _, _, s in data], color=cols,
               error_kw=dict(ecolor=c["secondary"], lw=1, capsize=3))
        ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=10)
        ax.tick_params(axis="x", labelsize=8.5)
    a1.set_ylim(2.6, 3.3)
    a1.bar([0], [0], color=c["series"][0], label="static 분할")
    a1.bar([0], [0], color=c["series"][1], label="동적 제어")
    a1.legend(frameon=False, loc="upper right")
    a1.annotate("d44 − bind+GATE = 0.088\n(5.4 pooled σ)", xy=(2, 3.15), xytext=(3.3, 3.22), fontsize=8.5,
                color=c["secondary"], arrowprops=dict(arrowstyle="-", color=c["muted"]))
    fig.suptitle("HE0 — 단일 GPU 동적 제어는 best-static(d44)을 넘지 못한다 (n ≥ 4)", fontsize=12, x=0.01, ha="left")
    foot(fig, "왼쪽 y축은 2.6에서 시작(차이를 보이기 위함). 오른쪽은 SLO로 컨트롤러를 재튜닝해 직접 측정(재스코어 아님).\n"
              "출처: CONSENSUS §1-7·§1-17. 범위: 변화 trace·시험한 두 SLO 점 한정, 크기는 HI phase −6.8~−6.9%로 축소(2026-10-02).", c)
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    return fig


def fig_lt(c):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    names = [n for n, _, _ in LT]
    vals = [v for _, v, _ in LT]
    ax.bar(names, vals, 0.5, color=[c["series"][0], c["series"][1], c["series"][1]])
    for i, (_, v, pre) in enumerate(LT):
        ax.text(i, v + 3, f"{pre}{v} ms", ha="center", fontsize=11, fontweight="bold", color=c["ink"])
    ax.set_ylabel("TPOT (ms)")
    ax.set_ylim(0, 140)
    fig.suptitle("layer-type 런타임 정책 — 층 종류별 SM 전환은 decode TPOT를 악화", fontsize=11.5, x=0.01, ha="left")
    foot(fig, "A100 green-context에서 층 경계 SM 전환이 sub-step drain·동기화를 일으킴(서빙 직접 측정). 시뮬레이터 per-layer 이득(1.37–2.02×)은 전이되지 않음.\n"
              "출처: CONSENSUS §1-3·§1-15, PROJECT_STATUS 요약(42→124 ms, 최적화 후 약 85 ms). 등급: 서빙(Claim B 강한 지지).", c)
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    return fig


def fig_floor(c):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    allL, allT = [], []
    for j, (arm, lab) in enumerate([("fused_sym", "fused_sym"), ("agnostic", "PD-mux agnostic")]):
        d = json.loads((J1 / f"m3_N1_{arm}_boot1/FLOOR.json").read_text())
        L = sorted(int(k) for k in d["points"])
        T = [d["points"][str(l)]["floor_s"] * 1000 for l in L]
        allL += L
        allT += T
        if j == 0:
            ax.plot(L, T, marker="o", ms=13, lw=0, markerfacecolor="none", markeredgecolor=c["series"][0], markeredgewidth=2, label=lab)
        else:
            ax.plot(L, T, marker="s", ms=6, lw=0, color=c["series"][1], label=lab)
    xs = np.linspace(0, 17000, 50)
    ax.plot(xs, 9 + 0.111 * xs, lw=2, color=c["muted"], ls="--", label="정본 식 9 ms + 0.111 ms/토큰 × L")
    ax.set_xlabel("입력 길이 L (토큰)")
    ax.set_ylabel("무경합 TTFT floor (ms)")
    ax.set_xlim(0, 17500)
    ax.set_ylim(0, 2000)
    ax.legend(frameon=False, loc="upper left")
    fig.suptitle("CM-2 — prefill 시간은 L에 선형 (Nemotron-Nano-9B-v2, 108 SM)", fontsize=11.5, x=0.01, ha="left")
    foot(fig, "점 = M3 J1 boot1 FLOOR.json 원자료(길이당 16회, floor_s). 두 arm이 거의 겹침(차 ≤ 0.2%). chunked-prefill −1, mem 0.82, VESSL A100.\n"
              "출처: AUDIT_J1_N1_analysis_2026-10-19.md(CONFIRMED, 범위 한정: L 1k–16k·무경합), COSTMODEL CM-2.", c)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    return fig


def fig_mem(c):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    L = np.linspace(0, 32768, 200)
    kv = K_BYTES * L / 1e6
    ax.fill_between(L, 0, S_BYTES / 1e6, color=c["series"][2], alpha=0.85, lw=0, label="고정 state s = 143.5 MB")
    ax.fill_between(L, S_BYTES / 1e6, S_BYTES / 1e6 + kv, color=c["series"][0], alpha=0.85, lw=0,
                    label="KV k·L (k = 16,382 B/토큰)")
    ls = S_BYTES / K_BYTES
    ax.axvline(ls, color=c["ink"], lw=1, ls=":")
    ax.text(ls + 400, 520, f"L* = s/k ≈ {ls/1000:.1f}k\n(이보다 짧으면 state가 지배)", fontsize=9, color=c["secondary"])
    ax.set_xlabel("요청 길이 L (토큰)")
    ax.set_ylabel("요청당 메모리 (MB)")
    ax.set_xlim(0, 32768)
    ax.set_ylim(0, 700)
    ax.legend(frameon=False, loc="upper left")
    fig.suptitle("CM-5 — hybrid 요청당 메모리 = 고정 state + KV (Nemotron-Nano-9B-v2)", fontsize=11.5, x=0.01, ha="left")
    foot(fig, "s·k = 부팅 로그 실측(2줄). 경계 L* ≈ 8.8k는 s/k 산술 [유도, GB 단위; GiB면 9.4k] — 경계 자체는 실측 아님.\n"
              "radix OFF에서 동시 요청 상한 = mamba 슬롯 수(cap). 출처: COSTMODEL CM-5, 128k 프로브 RESULT.", c)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    return fig


def fig_obs(c):
    d = json.loads(DECOMP.read_text())["checks"]["metrics"]
    rows = [("TEL 묶음\n(telemetry+phase)", "TEL"), ("I-4 증분\n(CUDA event)", "I4_increment"),
            ("I-5\n(admission)", "I5"), ("I-6\n(cudagraph pad)", "I6")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (met, ttl) in zip(axes, [("lambda_thr", "처리 용량 λ (HI 셀 achieved)"), ("itl_p95", "요청 ITL p95 (LO 셀)")]):
        for i, (lab, key) in enumerate(rows):
            e = d[met][key]
            lo, hi = e["ci95_pct"]
            flagged = e["flag"] == "FLAGGED"
            col = c["series"][1] if flagged else c["muted"]
            ax.plot([lo, hi], [i, i], color=col, lw=2.5)
            ax.plot([e["pct"]], [i], "o", ms=9, color=col, markeredgecolor=c["surface"], markeredgewidth=1.5)
            ax.text(hi + 0.15, i, f"{e['pct']:+.1f}%" + (" (CI∌0)" if flagged else ""), va="center", fontsize=9,
                    color=c["ink"] if flagged else c["secondary"])
        ax.axvline(0, color=c["axis"], lw=1)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.invert_yaxis()
        ax.set_xlabel("OFF 대비 변화 (%), 95% CI")
        ax.set_title(ttl, fontsize=10)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    fig.suptitle("계측 자체가 측정을 움직인다 — 계측별 관측자 효과 (agnostic, shape A, n=4)", fontsize=12, x=0.01, ha="left")
    foot(fig, "주황 = CI가 0을 배제(FLAGGED), 회색 = CI∋0(저검정력). TEL은 경로+PHASE_EVENTS 묶음이라 두 knob 몫 미분리. 단일 Job·shape A 점추정이며 다른 arm·부하의 보정 계수가 아님.\n"
              "결과 → 결정 22: 판정 측정은 전 arm 계측 OFF. 출처: DECOMP_RESULT.json, AUDIT_telemetry_asymmetry_past_results_2026-10-21.md §1·§3.", c)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    return fig


FIGS = [("01_claimG_p1opint", fig_p1), ("02_he0_static_vs_dynamic", fig_he0), ("03_layertype_tpot", fig_lt),
        ("04_cm2_prefill_floor", fig_floor), ("05_cm5_memory_per_request", fig_mem), ("06_instrument_observer_effect", fig_obs)]


def main():
    OUT.mkdir(exist_ok=True)
    c = style.apply("light")
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    for slug, fn in FIGS:
        fig = fn(c)
        p = OUT / f"{slug}.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        print(p)


if __name__ == "__main__":
    main()
