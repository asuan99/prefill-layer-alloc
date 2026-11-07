#!/usr/bin/env python3
"""HE0 (KISTI A100, Zamba2-2.7B) static vs dynamic + SGLang-default ESTIMATE (not measured).
Estimate = G_d44(HE0) x [G_SGLang / G_PD-mux] from P1-opint Zamba2 (rate 2 and rate 3, rep means)."""
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
# CONSENSUS §1-7·§1-17 (same values as make_current_figs.py)
HE0 = [("d44", 3.220, .013), ("d34", 3.171, .025), ("bind+GATE", 3.132, .019), ("d24", 3.039, .130),
       ("SLO-aware", 2.964, .025), ("bind", 2.934, .306), ("d16", 2.846, .055)]
TIGHT = [("d44", 73.2, 4.8), ("d34", 49.6, 3.9), ("bind+GATE", 44.3, 2.7), ("bind", 40.6, 0.4)]
STATIC = {"d44", "d34", "d24", "d16"}

rows = [r for r in csv.DictReader(open(CSV)) if r["source"] == "p1_opint" and r["round"].startswith("rep") and "zamba" in r["model"]]
def m(rate, arm): return st.mean(float(r["goodput_req_per_s"]) for r in rows if r["rate_or_rung"] == rate and r["arm"] == arm)
ratios = {rate: m(rate, "plain") / m(rate, "agnostic") for rate in ("2", "3")}
g44 = HE0[0][1]
est_hi, est_lo = g44 * ratios["2"], g44 * ratios["3"]
est_mid = (est_hi + est_lo) / 2
print({k: round(v, 3) for k, v in ratios.items()}, round(est_lo, 2), round(est_hi, 2))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[1.75, 1]))
# conservative single value: the smaller PD-mux gain (rate-2 ratio)
est = est_hi
bars = [("SGLang 기본*", est, 0, c["muted"])] + [(n, v, s, c["series"][0] if n in STATIC else c["series"][1]) for n, v, s in HE0]
xs = range(len(bars))
a1.bar(xs, [b[1] for b in bars], 0.6, color=[b[3] for b in bars])
a1.errorbar(list(xs)[1:], [b[1] for b in bars][1:], yerr=[b[2] for b in bars][1:], fmt="none", ecolor=c["secondary"], lw=1, capsize=3)
a1.set_xticks(list(xs), [b[0] for b in bars])
a1.tick_params(axis="x", labelsize=8.5)
a1.set_ylim(2.0, 3.45)
a1.set_ylabel("goodput (req/s)")
a1.set_title("관대 SLO (3 s / 60 ms), 변화 trace rate 3↔12", fontsize=10)
a1.bar([0], [0], color=c["muted"], label="SGLang 기본")
a1.bar([0], [0], color=c["series"][0], label="static 분할")
a1.bar([0], [0], color=c["series"][1], label="동적 제어")
a1.legend(frameon=False, loc="upper right", fontsize=8.5, ncol=3)
tn = [n for n, _, _ in TIGHT]
a2.bar(tn, [v for _, v, _ in TIGHT], 0.6, yerr=[s for _, _, s in TIGHT],
       color=[c["series"][0] if n in STATIC else c["series"][1] for n in tn],
       error_kw=dict(ecolor=c["secondary"], lw=1, capsize=3))
a2.set_ylabel("SLO 달성률 (%)")
a2.set_ylim(0, 100)
a2.set_title("엄격 SLO (chat 300 / 50 ms), rate 8", fontsize=10)
a2.tick_params(axis="x", labelsize=8.5)

fig.suptitle("Zamba2-2.7B (KISTI A100): SGLang 기본 대비 static·동적 제어 goodput", fontsize=12, x=0.01, ha="left")
fig.text(0.01, 0.015, "static·동적: n ≥ 4, 평균 ± 표준편차. 왼쪽 y축은 2.0에서 시작. * 같은 모델 고정 도착률 실험의 SGLang 기본/PD-mux 비(0.71)로 환산한 추정값.",
         fontsize=8, color=c["muted"])
fig.tight_layout(rect=(0, 0.05, 1, 0.94))
fig.savefig(os.path.join(HERE, "fig_he0_with_sglang_estimate.png"), dpi=170)
