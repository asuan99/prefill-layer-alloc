#!/usr/bin/env python3
"""Render the D* proposal section (§8b) of the axes map from dstar_model output.

Writes the block between <!-- DSTAR:BEGIN --> and <!-- DSTAR:END --> in reports/anchor/axes_map.html (inserting it right
after <!-- ANALYTIC:END --> the first time). Numbers come from reports/cost_model/dstar/dstar_result.json; measured
capacity brackets and failure-reason fractions are descriptive values copied below with their sources. Do not hand-edit
the block -- rerun this script.

  python3 workspace/engine-port/scripts/costmodel/render_dstar_section.py [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
RESULT = os.path.join(ROOT, "reports", "cost_model", "dstar", "dstar_result.json")
PAGE = os.path.join(ROOT, "reports", "anchor", "axes_map.html")
BEGIN, END, AFTER = "<!-- DSTAR:BEGIN -->", "<!-- DSTAR:END -->", "<!-- ANALYTIC:END -->"
D_GRID = ["16", "24", "34", "44", "64"]

# Measured, descriptive. W1: tightest of CAP1 (policy_cap1/RESULT_cap1_2026-10-28.md) and CAP1-R
# (policy_cap1r/RESULT_cap1r_2026-10-29.md); W3: CAP2 (policy_cap2/RESULT_cap2_sustain_2026-10-29.md).
MEAS = {
    "W1": {"16": "[1.66, 2.24)", "24": "[4.09, 5.52)", "34": "[6.34, 7.29)", "44": "[7.29, 8.39)", "64": "[7.29, 8.39)"},
    "W3": {"16": "[0.3375, 0.506)", "24": "[0.225, 0.3375)", "34": "[0.225, 0.3375)", "44": "[0.15, 0.225)",
           "64": "&lt; 0.1"},
}
# First-failing cell, share of requests failing each predicate (raw segments re-tally, PROPOSAL_dstar_model §4).
FAIL = {
    "W1": {"16": "ITL 0.82", "24": "ITL 0.65", "34": "TTFT 0.10", "44": "TTFT 0.30", "64": "TTFT 0.29"},
    "W3": {"16": "ITL 0.41 · TTFT 0.08", "24": "TTFT 0.12", "34": "TTFT 0.20", "44": "TTFT 0.10", "64": "TTFT 0.20"},
}
BIND_KO = {"ITL": "ITL", "SLOT": "슬롯", "TTFT": "TTFT"}


def f(x, n=2):
    return f"{x:,.{n}f}"


def render(r: dict) -> str:
    hw = r["hardware"]
    fit = hw["fit"]
    rows_u = "".join(
        f"<tr><td>{row['D']}</td><td>{f(row['a_ms'])}</td><td>{f(row['u'], 3)}</td><td>{f(row['u_per_sm'], 4)}</td></tr>"
        for row in hw["rows"])
    kap = " / ".join(f"{float(v):.3f}" for v in hw["kappa"].values())

    def cap_rows(w):
        W = r["workloads"][w]
        out = []
        for d in D_GRID:
            pr = W["per_D"][d]["rho"]
            lo, mid, hi = pr["1.19"]["lam_cap"], pr["1.16"]["lam_cap"], pr["1.13"]["lam_cap"]
            bind = " · ".join(BIND_KO.get(b, b) for b in pr["1.16"]["binding"])
            out.append(f"<tr><td>{w}</td><td>D{d}</td><td>{mid:.3g} <small>({lo:.3g}–{hi:.3g})</small></td>"
                       f"<td>{bind}</td><td>{MEAS[w][d]}</td><td>{FAIL[w][d]}</td></tr>")
        return "".join(out)

    win = r["windows"]
    cs1, cs3 = win["W1"]["co_step_at_cap_ms"], win["W3"]["co_step_at_cap_ms"]
    co1 = " / ".join(f"{cs1[d]:.1f}" for d in D_GRID)
    co3 = " / ".join(f"{cs3[d]:.1f}" for d in D_GRID)
    ev = r["eval_prediction"]["1.16"]
    ev_rows = "".join(
        f"<tr><td>D{d}</td><td>{ev[d]['LO_good_frac']:.3f}</td><td>{ev[d]['HI_ratio_to_D44']:.3f}</td>"
        f"<td>{ev[d]['total_ratio_to_D44']:.3f}</td></tr>" for d in D_GRID)
    w1, w3 = r["workloads"]["W1"], r["workloads"]["W3"]

    return f"""{BEGIN}
<section id="dstar">
  <style>#dstar td{{font-size:.84rem}}#dstar td small{{color:var(--muted);font-family:var(--f-mono);font-size:.72rem}}#dstar h3{{margin-top:4px}}</style>
  <h2>D* 수식 — static decode SM을 식으로 <span class="sec">§8b · 제안 · 결정 61 (2026-10-30)</span></h2>
  <p class="note"><b>제안이며 정본이 아닙니다.</b> §8 고정본(analytic-v1)은 바꾸지 않고 사전으로 불러 씁니다. 식은 CAP1·CAP1-R·CAP2·TUNE 결과를 본 뒤 썼으므로 아래 대조는 <b>사후 재현</b>입니다. 예측으로 셀 수 있는 것은 EVAL 사전 예측(봉인 커밋 <code>113de504</code>)뿐이고, EVAL은 결정 60에 따라 implfix gate 뒤 짝 실험으로 보류됐습니다. 근거 <code>reports/cost_model/PROPOSAL_dstar_model_2026-10-30.md</code>, 계산 <code>scripts/costmodel/dstar_model.py</code>(이 절은 <code>render_dstar_section.py</code>가 다시 씁니다).</p>

  <h3>세 층</h3>
  <div class="tbl"><table><thead><tr><th>층</th><th>무엇</th><th>값 · 출처</th></tr></thead><tbody>
    <tr><td>하드웨어·모델 <span class="chip g-derived">derived</span></td><td><span class="fx">a₀ = 가중치 바이트 / BW</span></td><td>{f(hw['a0_ms_derived'], 2)} ms (Nano, {hw['hbm_bw'] / 1e12:.3f} TB/s) · §8 표</td></tr>
    <tr><td>비용 항 <span class="chip g-scoped">scoped</span></td><td><span class="fx">T<sub>dec</sub>(D, b, c)</span> · <span class="fx">T2(L; 108)</span> · <span class="fx">T2′(32k; P)</span> · <span class="fx">ρ</span></td><td>T1 sticky D 5개 + 108u(F1/F2/F3, 두 round 평균) · T2 108v · ρ 1.13–1.19(동거·전력·관측자 총효과) · <code>RESULT_costterms_2026-10-26.md</code> §2–§5</td></tr>
    <tr><td>서빙 <span class="chip g-open">open</span></td><td>비sticky lockstep PD-mux: 동거 중에만 D, 그 밖엔 108 SM</td><td>아래 식, 가정 A1–A9</td></tr>
  </tbody></table></div>

  <h3>기호</h3>
  <div class="tbl"><table><thead><tr><th>기호</th><th>의미</th></tr></thead><tbody>
    <tr><td>D, P = 108 − D</td><td>동거 중 decode SM(목표 라벨) / prefill SM</td></tr>
    <tr><td>u(D) = a₀ / a(D)</td><td>D SM이 끌어내는 실효 HBM 비율(a(D) = T1 절편, 고정비 포함)</td></tr>
    <tr><td>κ(D)</td><td>T2′(32k; P)·P / (T2(32k; 108)·108) — prefill이 P에 반비례하는 정도</td></tr>
    <tr><td>f</td><td>동거 비율 = prefill이 도는 시간 비율</td></tr>
    <tr><td>b, b95</td><td>decode 점유(Little), b + 1.645√b (≤ cap)</td></tr>
    <tr><td>cap, S<sub>ITL</sub>, S<sub>TTFT</sub></td><td>max_running_requests 48, 60 ms, W1 3,000 ms · W3 2,814.84 ms</td></tr>
  </tbody></table></div>

  <h3>서빙 식</h3>
  <div class="tbl"><table><thead><tr><th>항</th><th>식</th></tr></thead><tbody>
    <tr><td>prefill 요청 하나</td><td><span class="fx">S<sub>pre</sub>(L) = κ · T2(L; 108) · 108 / P</span></td></tr>
    <tr><td>동거 비율</td><td><span class="fx">f = min(1, λ · E[S<sub>pre</sub>])</span></td></tr>
    <tr><td>decode step</td><td><span class="fx">T<sub>co</sub> = ρ · T<sub>dec</sub>(D, b, c)</span> · <span class="fx">T<sub>solo</sub> = T<sub>dec</sub>(108u, b, c)</span></td></tr>
    <tr><td>실효 step · 점유</td><td><span class="fx">T<sub>eff</sub> = f·T<sub>co</sub> + (1−f)·T<sub>solo</sub></span> · <span class="fx">b = λ · E[R] · T<sub>eff</sub></span> (고정점)</td></tr>
    <tr><td>ITL 제약</td><td><span class="fx">step(b95) ≤ S<sub>ITL</sub></span> <small>(f &gt; 5%면 T<sub>co</sub>, 아니면 T<sub>solo</sub>)</small></td></tr>
    <tr><td>슬롯 제약</td><td><span class="fx">λ · (E[R]·T<sub>eff</sub> + E[S<sub>pre</sub>] + W) ≤ cap</span></td></tr>
    <tr><td>TTFT 제약</td><td><span class="fx">S<sub>pre</sub>(L<sub>q90</sub>) + W ≤ S<sub>TTFT</sub></span>, <span class="fx">W = λE[S²] / (2(1 − λE[S]))</span> <small>M/G/1 평균 대기</small></td></tr>
    <tr><td>용량 · D*</td><td><span class="fx">λ<sub>cap</sub>(D)</span> = 세 제약을 모두 만족하는 최대 λ · <span class="fx">D* = argmax<sub>D</sub> λ<sub>cap</sub>(D)</span></td></tr>
  </tbody></table></div>

  <h3>두 벽 — 대기열 이전의 닫힌 형태</h3>
  <div class="tbl"><table><thead><tr><th>벽</th><th>식</th><th>W1 ShareGPT</th><th>W3 arXiv-Sum 16k</th></tr></thead><tbody>
    <tr><td>아래 벽 (decode가 ITL을 지키는 최소 D)</td><td><span class="fx">ρ · T<sub>dec</sub>(D, b<sub>peak</sub>, c) ≤ S<sub>ITL</sub></span></td><td>D{win['W1']['lower_wall_D_at_cap']['1.16']} (b = cap 48) · 동거 step D16–64: {co1} ms</td><td>cap 벽 해당 없음(capacity에서 b ≈ 17) · 동거 step @48: {co3} ms</td></tr>
    <tr><td>위 벽 (prefill이 TTFT를 지키는 최대 D)</td><td><span class="fx">D ≤ 108 − κ · T2(L<sub>q90</sub>) · 108 / S<sub>TTFT</sub></span></td><td>{win['W1']['upper_wall_D']:.1f} (L<sub>q90</sub> {win['W1']['L_q90']:,})</td><td><b>{win['W3']['upper_wall_D']:.1f}</b> (L<sub>q90</sub> {win['W3']['L_q90']:,}) → D64 불가</td></tr>
  </tbody></table></div>
  <p class="lede" style="font-size:.84rem">W1에서는 아래 벽이 D24와 D34 사이에서 닫히고, 그 위는 슬롯 상한과 108 SM decode step이 정하는 고원입니다. D44는 이 고원 안의 한 점으로 설명됩니다.</p>

  <h3>하드웨어 층 — u(D)</h3>
  <div class="tbl"><table><thead><tr><th>D</th><th>a(D) ms</th><th>u</th><th>u / SM</th></tr></thead><tbody>{rows_u}</tbody></table></div>
  <p class="lede" style="font-size:.84rem">원점을 지나는 2구간 적합(6점): 무릎 D ≈ {fit['D_knee']:.0f}, 무릎 아래 SM당 {fit['per_sm_bw_GBps_below_knee']:.1f} GB/s, 무릎 위 기울기 ×{fit['slope_ratio']:.2f}. κ(D16/24/34/44/64) = {kap}: prefill은 동거 상태에서도 P에 거의 정확히 반비례합니다. decode는 무릎 뒤 포화하고 prefill은 선형이라 분할 최적이 생깁니다. W1의 D*는 이 무릎이 아니라 ITL 벽과 슬롯 고원이 정했습니다.</p>

  <h3>사후 재현 — 측정 용량 대비 (예측 아님)</h3>
  <div class="tbl"><table><thead><tr><th>W</th><th>D</th><th>모델 λ<sub>cap</sub> (ρ 1.16, 띠 1.19–1.13)</th><th>모델 구속</th><th>측정 통과 → 첫 실패</th><th>첫 실패 셀 실패 사유(요청 비율)</th></tr></thead><tbody>{cap_rows('W1')}{cap_rows('W3')}</tbody></table></div>
  <p class="lede" style="font-size:.84rem">맞은 것: 구속 제약 종류 W1 5/5·W3 4/5, W1 순서, 워크로드별 D* 방향(W1 큰 D, W3 작은 D), W3 D64 불가. 틀린 것: W3 상위 순서(모델 D24 &gt; D34 &gt; D16, 실측 D16 &gt; D24 = D34; TTFT 구속 셀에서 1.2–2.6배 낙관, A7 평균 대기). W1 크기는 비교하지 않습니다(측정은 120요청 버스트, 모델은 정상상태; 지속 600 s D44·D64 완료율 약 3.0 req/s). 입력: W1 E[R] {w1['ER']:.0f}·ctx {w1['ctx_decode']:.0f}, W3 E[R] {w3['ER']:.0f}·ctx {w3['ctx_decode']:.0f}.</p>

  <h3>EVAL 사전 예측 (봉인 113de504 · 보류 중)</h3>
  <div class="tbl"><table><thead><tr><th>D</th><th>LO 좋은 요청 비율</th><th>HI goodput / D44</th><th>전체 goodput / D44</th></tr></thead><tbody>{ev_rows}</tbody></table></div>
  <p class="lede" style="font-size:.84rem">등록 예측 E-1…E-8(띠 포함)은 PROPOSAL §5에 있습니다. 가장 강한 예측은 static D24의 HI goodput ≈ 0(배치가 차면 동거 step이 60 ms를 넘음)입니다. D44 절대 goodput은 TUNE을 본 뒤라 채점하지 않습니다. 셀 rate가 다시 정해지면 봉인된 모델 코드에 새 rate만 넣어 addendum으로 다시 봉인합니다.</p>

  <h3>가정</h3>
  <ul class="ban">
    <li>A1 sticky-D 무경합 T1을 ρ로 동거에 옮김 · A2 a(D)에 고정비 포함 · A3 W1의 L &lt; 1k는 T2 격자 밖 외삽<span class="sp">PROPOSAL §6</span></li>
    <li>A4 prefill 배치 상각 없음 · A5 f = λE[S] · A6 ITL p95 규칙은 f &gt; 5% 경성 문턱<span class="sp">PROPOSAL §6</span></li>
    <li>A7 TTFT는 q90 서비스 + 평균 대기(꼬리 과소) · A8 ρ는 D·b 무관 띠(CM-4: D 순서 서술 금지) · A9 EVAL HI는 슬롯 포화, drain 제외<span class="sp">PROPOSAL §6</span></li>
  </ul>
</section>
{END}"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the page block differs from a fresh render")
    a = ap.parse_args(argv)
    block = render(json.load(open(RESULT)))
    page = open(PAGE, encoding="utf-8").read()
    if BEGIN in page:
        new = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda m: block, page, flags=re.S)
    else:
        if AFTER not in page:
            sys.exit("ANALYTIC:END marker not found")
        new = page.replace(AFTER, AFTER + "\n\n" + block, 1)
    if a.check:
        sys.exit(0 if new == page else 1)
    with open(PAGE, "w", encoding="utf-8") as fh:
        fh.write(new)
    print("wrote", PAGE)


if __name__ == "__main__":
    main()
