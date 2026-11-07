#!/usr/bin/env python3
"""Progress-report figures (2026-11-02). Descriptive only; numbers transcribed
from STOCK_METRICS_TABLE.csv (P1-opint rep rows) and RESULT_policy_implfix §6."""
import csv, os, statistics as st, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import style
c = style.apply("light")
plt.rcParams["font.family"] = "Noto Sans CJK JP"
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
CSV = os.path.join(ROOT, "workspace/engine-port/results/stock_baseline/STOCK_METRICS_TABLE.csv")

# --- A. baseline: SGLang default (plain) vs PD-mux default (agnostic), rep mean +- sd
rows = [r for r in csv.DictReader(open(CSV)) if r["source"] == "p1_opint" and r["round"].startswith("rep")]
cells = [("granite", "2"), ("granite", "3"), ("granite", "4"), ("zamba", "2"), ("zamba", "3")]
pct = {("granite", "3"): "+11.7%", ("granite", "4"): "+27.0%", ("zamba", "2"): "+40.5%", ("zamba", "3"): "+185.8%"}
def g(m, rate, arm):
    v = [float(r["goodput_req_per_s"]) for r in rows if m in r["model"] and r["rate_or_rung"] == rate and r["arm"] == arm]
    return st.mean(v), st.stdev(v), len(v)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [3, 2]})
for ax, m, title in [(axes[0], "granite", "Granite-4.0-h-micro"), (axes[1], "zamba", "Zamba2-2.7B")]:
    rates = [r for mm, r in cells if mm == m]
    for i, rate in enumerate(rates):
        for j, (arm, lab, col) in enumerate([("plain", "SGLang 기본", c["series"][1]), ("agnostic", "PD-mux 기본 설정", c["series"][0])]):
            mu, sd, n = g(m, rate, arm)
            assert n == 5, (m, rate, arm, n)
            ax.bar(i + (j - 0.5) * 0.38, mu, 0.36, color=col, label=lab if i == 0 else None)
            ax.errorbar(i + (j - 0.5) * 0.38, mu, yerr=sd, color=c["secondary"], capsize=3, lw=1)
        top = max(g(m, rate, a)[0] + g(m, rate, a)[1] for a in ("plain", "agnostic"))
        if (m, rate) in pct:
            ax.text(i, top + 0.12, pct[(m, rate)], ha="center", fontsize=11, fontweight="bold", color=c["ink"])
    ax.set_xticks(range(len(rates)), [f"도착률 {r} req/s" for r in rates])
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, 4.6)
axes[0].set_ylabel("goodput (req/s)")
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("SGLang 기본 대비 PD-mux 기본 설정의 꼬리 SLO goodput (A100, cudagraph ON, 5회 평균 ± 표준편차)", fontsize=12, x=0.01, ha="left")
fig.text(0.01, 0.01, "goodput = TTFT ≤ 3 s 이고 요청 내부 ITL p95 ≤ 60 ms인 요청의 처리율. 입력 2,000 / 출력 96 토큰. 2026년 8월, 이전 기판(KISTI) 측정.", fontsize=8.5, color=c["muted"])
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
fig.savefig(os.path.join(HERE, "fig_baseline_goodput.png"), dpi=170)

# --- B. static D vs goodput with sealed prediction bands (RESULT_policy_implfix §6)
D = [16, 24, 34, 44, 64]
G = [0.868, 1.611, 2.166, 2.199, 2.188]
H = [0.118, 0.031, 0.111, 0.108, 0.121]  # t(3) 95% CI half width
g44 = G[3]
bands = {16: (0.0, 0.30, False), 24: (0.55, 0.85, True), 34: (0.93, 1.01, True), 64: (0.98, 1.08, True)}
fig, ax = plt.subplots(figsize=(7.2, 4.3))
for d, (lo, hi, hit) in bands.items():
    col = c["good"] if hit else c["critical"]
    ax.add_patch(plt.Rectangle((d - 2.6, lo * g44), 5.2, (hi - lo) * g44, color=col, alpha=0.18, lw=0))
    y, va = {16: (0.33, "center"), 24: (lo * g44 - 0.06, "top"), 34: (lo * g44 - 0.06, "top"), 64: (hi * g44 + 0.04, "bottom")}[d]
    ax.text(d, y, "예측 띠 적중" if hit else "예측 띠\n빗나감", ha="center", va=va, fontsize=9, color=col)
ax.plot(D, G, color=c["ink"], lw=2, zorder=3)
ax.errorbar(D, G, yerr=H, fmt="o", ms=8, color=c["ink"], capsize=4, zorder=4, label="측정 (seed 4개 평균, 95% CI)")
ax.annotate("기준점 D44", (44, g44), xytext=(44, 1.55), ha="center", fontsize=9, color=c["secondary"],
            arrowprops=dict(arrowstyle="-", color=c["muted"]))
ax.set_xticks(D)
ax.set_xlabel("decode에 준 SM 수 D (전체 108)")
ax.set_ylabel("goodput (req/s)")
ax.set_ylim(0, 2.6)
ax.legend(frameon=False, loc="lower right")
ax.set_title("Nano-9B-v2 static 분할: 측정 대 사전 봉인 예측 띠", fontsize=12, loc="left")
fig.text(0.01, 0.01, "띠 = 측정 전에 봉인한 G_D/G_44 예측 범위 × 측정 G_44. VESSL A100 1장, ShareGPT 부하. 과부하 구간을 포함한 셀이라 D 사이 배율은 인용하지 않는다.", fontsize=8, color=c["muted"])
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(HERE, "fig_static_D_prediction.png"), dpi=170)
print("ok")
