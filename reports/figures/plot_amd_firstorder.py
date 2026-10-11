#!/usr/bin/env python3
"""AMD 1차 계산 그림 (2026-10-11). 값·공식 모두 amd_firstorder_calc.py의 함수를 호출한다(인라인 재구현 없음).
좌: 모델별 decode step당 crossing 지연 상한 vs L0 (B=32, NVLink 실효 200 GB/s · PCIe 25 GB/s)
우: ctx별 요청당 KV(attention 층) vs mamba state 바이트 — 어느 모델에서 어느 쪽이 binder인가
출력: reports/figures/amd_firstorder_2026-10-11.png (커널 시간 0건, 예측 아님 — 가정 하 가산 비용)."""
import os, sys, importlib.util
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]  # 한글 글리프
matplotlib.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
spec = importlib.util.spec_from_file_location("calc", os.path.join(ROOT, "workspace/engine-port/results/amd_track/amd_firstorder_calc.py"))
calc = importlib.util.module_from_spec(spec); spec.loader.exec_module(calc)
sys.path.insert(0, HERE)
from style import LIGHT as C

MODELS = ["nemotron_h_8b", "zamba2_2.7b", "zamba2_7b", "granite_4_h_micro"]
SHORT = {"nemotron_h_8b": "Nemotron-H-8B", "zamba2_2.7b": "Zamba2-2.7B", "zamba2_7b": "Zamba2-7B", "granite_4_h_micro": "Granite-4-h-micro"}
B = 32
L0S = [1, 10, 65, 130, 300]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=C["surface"])
for ax in (ax1, ax2):
    ax.set_facecolor(C["plane"]); ax.grid(True, color=C["grid"], lw=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_color(C["axis"])
# 좌
for i, k in enumerate(MODELS):
    m = calc.MODELS[k]; n = 2 * m["n_attn_calls"]
    for fab, bw, ls in [("NVLink 200 GB/s", 200e9, "-"), ("PCIe 25 GB/s", 25e9, "--")]:
        y = [calc.crossing_ms(m, B, bw, l0) for l0 in L0S]   # 공식은 calc 한 곳에만(감사 (vii))
        ax1.plot(L0S, y, ls, color=C["series"][i], lw=2, marker="o", ms=4,
                 label=f"{SHORT[k]} ({n}회/step) · {fab}")
ax1.axhline(calc.TPOT_YARDSTICK_MS * 0.05, color=C["warning"], lw=1.2)
ax1.text(150, calc.TPOT_YARDSTICK_MS * 0.05 * 0.72, "TPOT 12 ms 눈금의 5%", color=C["secondary"], fontsize=8)
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("crossing 1회 지연 바닥 L0 (µs) — V1 실측 대상"); ax1.set_ylabel("step당 가산 지연 (ms, 가정 하·경계 아님), B=32")
ax1.set_title("L-B: decode step당 GPU 간 crossing 비용 (2×attention 호출)", fontsize=10)
ax1.legend(fontsize=7, loc="upper left", frameon=False, ncol=1)
# 우
ctxs = calc.CTXS
w = 0.18
for i, k in enumerate(MODELS):
    m = calc.MODELS[k]; kvt = calc.kv_bytes_per_token(m); ssm, conv = calc.state_bytes_per_req(m)
    xs = [j + (i - 1.5) * w for j in range(len(ctxs))]
    ax2.bar(xs, [kvt * L / calc.MiB for L in ctxs], w, color=C["series"][i], label=f"{SHORT[k]} KV/req")
    ax2.plot([j + (i - 1.5) * w for j in range(len(ctxs))], [(ssm + conv) / calc.MiB] * len(ctxs), "_",
             color=C["ink"], ms=12, mew=2)
ax2.plot([], [], "_", color=C["ink"], ms=12, mew=2, label="mamba state/req (L-불변)")
ax2.set_xticks(range(len(ctxs))); ax2.set_xticklabels([f"ctx {L}" for L in ctxs])
ax2.set_yscale("log"); ax2.set_ylabel("요청당 바이트 (MiB)")
ax2.set_title("P→D 핸드오프 binder: KV(attention 층, L-스케일) vs state(B-스케일)", fontsize=10)
ax2.legend(fontsize=7, frameon=False, loc="upper left")
fig.suptitle("AMD 1차 계산 — 구조 상수 × 대역폭·L0 가정, 커널 시간 0건 (가산 비용, 경계도 예측도 아님) · 2026-10-11 rev2", fontsize=10, color=C["secondary"])
fig.tight_layout()
out = os.path.join(HERE, "amd_firstorder_2026-10-11.png")
fig.savefig(out, dpi=160, facecolor=C["surface"]); print("wrote", out)
