#!/usr/bin/env python3
"""Render the fixed analytic cost formulas (decision 27) as HTML tables into the axes map artifact.

Rewrites the block between <!-- ANALYTIC:BEGIN --> and <!-- ANALYTIC:END --> in reports/anchor/axes_map.html.
All numbers come from analytic_terms.py, so the page cannot drift from the code.  Run after any formula change:
    python3 workspace/engine-port/scripts/costmodel/render_formula_tables.py
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analytic_terms as t  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
PAGE = os.path.join(REPO, "reports", "anchor", "axes_map.html")
BEGIN, END = "<!-- ANALYTIC:BEGIN -->", "<!-- ANALYTIC:END -->"


def n(x, unit=""):
    if isinstance(x, float) and x == float("inf"):
        return "∞"
    if isinstance(x, int) or (isinstance(x, float) and x.is_integer()):
        return f"{int(x):,}{unit}"
    return f"{x:,.3g}{unit}"


def human_bytes(b):
    return f"{b/1e6:,.1f} MB" if b >= 1e6 else f"{b:,.0f} B"


def tbl(head, rows, cls="tbl"):
    h = "".join(f"<th>{c}</th>" for c in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="{cls}"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def render() -> str:
    a = t.nano9b_v2()
    pf = t.prefill_flops(a, 65536)
    dc = t.decode_coefficients(a, 2.039e12)
    dstep = t.decode_step(a, 32, 8192)
    F = lambda s: f'<span class="fx">{s}</span>'  # noqa: E731

    sym = tbl(["기호", "의미", "Nano-9B-v2"], [
        ["d", "hidden", n(a.d)], ["L / c / b", "prefill 길이 / decode 문맥 / 배치", "—"],
        ["n<sub>q</sub>, n<sub>kv</sub>, d<sub>h</sub>", "query head, KV head, head 차원", f"{a.n_q}, {a.n_kv}, {a.d_h}"],
        ["n<sub>h</sub>, p, d<sub>in</sub>=n<sub>h</sub>p", "Mamba head, head 차원, 내부 차원", f"{a.n_h}, {a.p}, {n(a.d_in)}"],
        ["G, N, K, Q", "SSM group, state 차원, conv 커널, SSD chunk", f"{a.G}, {a.N}, {a.K}, {a.Q}"],
        ["d<sub>conv</sub>=d<sub>in</sub>+2GN", "conv 채널", n(a.conv_dim)], ["d<sub>ff</sub>, V", "MLP 중간 차원, 어휘", f"{n(a.d_ff)}, {n(a.V)}"],
        ["n<sub>A</sub>, n<sub>M</sub>, n<sub>F</sub>", "attention / Mamba / MLP 층 수", f"{a.n_A} / {a.n_M} / {a.n_F}"],
        ["β, β<sub>s</sub>", "원소 바이트 bf16 / SSM state fp32(SGLang)", f"{a.beta} / {a.beta_ssm}"],
    ])
    layer = tbl(["층", "파라미터", "FLOPs / 토큰(투영)", "prefill 추가 FLOPs", "decode 추가 FLOPs / 토큰", "상주 메모리", "decode 트래픽 / step"], [
        ["Attention (GQA)", F("W<sub>A</sub> = d·d<sub>h</sub>(n<sub>q</sub>+2n<sub>kv</sub>) + n<sub>q</sub>d<sub>h</sub>·d") + f"<br><small>{n(t.params_attn(a))}</small>",
         F("2W<sub>A</sub>"), F("2·n<sub>q</sub>·d<sub>h</sub>·L²") + " <small>(요청 전체, causal)</small>", F("4·n<sub>q</sub>·d<sub>h</sub>·c"),
         F("k<sub>A</sub> = 2·n<sub>kv</sub>·d<sub>h</sub>·β") + " /토큰", F("2W<sub>A</sub> + b·c·k<sub>A</sub>")],
        ["Mamba2 (SSD)", F("W<sub>M</sub> = d(2d<sub>in</sub>+2GN+n<sub>h</sub>) + d<sub>in</sub>·d") + f"<br><small>{n(t.params_mamba_matmul(a))}</small>",
         F("2W<sub>M</sub> + 2K·d<sub>conv</sub>"), F("2GQN + 2n<sub>h</sub>Qp + 4n<sub>h</sub>pN") + f" /토큰<br><small>{n(t.flops_ssd_prefill_per_token(a))}</small>",
         F("5·d<sub>in</sub>·N") + f"<br><small>{n(t.flops_ssm_decode_per_token(a))}</small>",
         F("s<sub>M</sub> = n<sub>h</sub>pN·β<sub>s</sub> + d<sub>conv</sub>(K−1)·β") + " /요청<br><small>길이 무관</small>", F("2W<sub>M</sub> + b·2·s<sub>M</sub>")],
        ["MLP (ReLU², 비게이트)", F("W<sub>F</sub> = 2·d·d<sub>ff</sub>") + f"<br><small>{n(t.params_mlp(a))} (게이트형 3·d·d<sub>ff</sub>)</small>",
         F("2W<sub>F</sub>"), "—", "—", "—", F("2W<sub>F</sub>")],
        ["LM head", F("d·V"), F("2dV") + " <small>샘플링 토큰만</small>", "prefill당 1회", "매 토큰", "—", F("β·d·V")],
    ])
    whole = tbl(["모델 전체", "식", "Nano-9B-v2 [유도]"], [
        ["요청당 고정 state", F("s = n<sub>M</sub>·s<sub>M</sub>"), human_bytes(t.state_bytes_per_request(a))],
        ["KV / 토큰", F("k = n<sub>A</sub>·k<sub>A</sub>"), f"{n(t.kv_bytes_per_token(a))} B"],
        ["state = KV 경계", F("L* = s / k"), f"≈ {n(round(t.L_star(a)))} 토큰"],
        ["prefill FLOPs", F("F<sub>pre</sub>(L) = α·L + γ·L² + 2dV"), f"α = {pf['linear_per_token']/1e9:.3f} GFLOP/토큰, γ = {n(pf['quad_coeff'])}"],
        ["이차항 = 선형항", F("L<sub>q</sub> = α / γ"), f"≈ {pf['L_quad_eq_linear']/1e3:.0f}k (64k에서 {pf['quad_share_at_L']*100:.0f}%)"],
        ["decode 하한", F("T<sub>dec</sub> ≥ a₀ + b₀·b + c₀·b·c"), "PREREG T1 함수형의 물리적 하한"],
        ["a₀ 가중치", F("β(n<sub>A</sub>W<sub>A</sub>+n<sub>M</sub>W<sub>M</sub>+n<sub>F</sub>W<sub>F</sub>+dV) / BW"), f"{dc['a0_s']*1e3:.2f} ms (A100 2.039 TB/s)"],
        ["b₀ SSM state", F("2s / BW"), f"{dc['b0_s_per_req']*1e3:.3f} ms / 요청"],
        ["c₀ KV", F("k / BW"), f"{dc['c0_s_per_req_token']*1e9:.1f} ns / (요청·토큰)"],
        ["state 항 = KV 항", F("c* = 2s / k"), f"≈ {dstep['c_state_eq_kv']/1e3:.1f}k"],
        ["b=32, c=8k 한 step", "가중치 + state + KV 바이트", f"{dstep['bytes_total']/1e9:.1f} GB, {dstep['intensity_flop_per_byte']:.1f} FLOP/B (메모리 바운드)"],
    ])
    comp_rows = []
    for r in t.table():
        comp_rows.append([r["variant"], f"{r['n_A']} / {r['n_M']}", human_bytes(r["s_bytes"]), n(r["k_bytes_per_token"]),
                          n(round(r["L_star"])), f"{r['L_quad_eq_linear']/1e3:.1f}k", f"{r['quad_share_at_64k']:.2f}",
                          f"{r['decode_c_state_eq_kv']/1e3:.1f}k" if r["decode_c_state_eq_kv"] else "0"])
    comp = tbl(["변형", "n<sub>A</sub> / n<sub>M</sub>", "s", "k (B/토큰)", "L*", "L<sub>q</sub>", "64k 이차항 비중", "c*"], comp_rows)
    return (f'{BEGIN}\n<section id="formulas">\n'
            f'  <h2>비용 수식 (고정본) <span class="sec">§8 · {t.VERSION} · 결정 27</span></h2>\n'
            '  <p class="lede">Attention / Mamba2 / MLP 층의 FLOPs, 상주 메모리, decode 트래픽 닫힌 식입니다. 비용 모델은 이 식을 사전(prior)으로 두고 측정이 효율·PD-mux 항을 덧붙입니다. '
            '정본 문서는 <code>reports/cost_model/ANALYTIC_COST_FORMULAS.md</code>, 수치는 <code>scripts/costmodel/analytic_terms.py</code>가 계산해 이 표를 다시 씁니다(손으로 고치지 않음). 모든 값은 [유도]이며 측정이 아닙니다. '
            'FLOPs는 곱셈·덧셈을 각각 1로 셉니다.</p>\n'
            f'  <h3>기호</h3>\n  {sym}\n  <h3>층별 식</h3>\n  {layer}\n  <h3>모델 전체</h3>\n  {whole}\n'
            f'  <h3>같은 골격 조성 변형 (결정 24)</h3>\n  {comp}\n'
            '  <p class="lede" style="font-size:.84rem">실측 대조: 파라미터·s·k가 부팅 배너와 일치하고, J1 무경합 floor 기울기 0.111 ms/토큰은 α 기준 실효 약 141 TFLOP/s(피크의 약 45%)입니다. '
            'k와 c₀는 n<sub>A</sub>에 정확히 비례하고(J-H 예측 P3), γ/α는 n<sub>A</sub>에 따라 커집니다(P4).</p>\n'
            f'</section>\n{END}')


def main() -> int:
    s = open(PAGE, encoding="utf-8").read()
    block = render()
    if BEGIN in s:
        s = re.sub(re.escape(BEGIN) + ".*?" + re.escape(END), lambda m: block, s, flags=re.S)
    else:
        s = s.replace('<section id="docs">', block + '\n\n<section id="docs">', 1)
    open(PAGE, "w", encoding="utf-8").write(s)
    print(f"rendered {t.VERSION} into {os.path.relpath(PAGE, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
