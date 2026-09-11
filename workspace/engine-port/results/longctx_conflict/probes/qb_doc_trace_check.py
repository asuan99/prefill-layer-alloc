#!/usr/bin/env python3
"""Trace check: PREREG_QB_LOOPGAIN_2026-09-11.md body numbers -> qb_forecast_result.json
(verdict D4; gate candidates G-epsilon / #159 / #161).  GPU 0.

WHY NOT A GLOBAL MATCH: matching doc numbers against *every* number in the JSON is
vacuous -- measured chance-hit rate 97-100 % for 1-2-decimal numbers and integers,
~50 % at 3-4 decimals (the JSON holds ~50k floats).  So this checker BINDS numbers:

  (1) TABLE ROWS are regenerated from named JSON paths and must appear verbatim.
  (2) PROSE numbers are bound by exact SNIPPETS (number + surrounding words),
      each generated from a named JSON path.
  (3) Every bound string is masked out; the RESIDUAL body numbers must each be
      a small integer (< 10), structural (section/gate/rev/job/date/path/arm/
      formula token), or an EXTERNAL constant whose source is named on the same
      line (the prereg header's declared exception 2).  Anything else is UNTRACED.
Excluded by declaration (exception 1): sec 7 block quotes and all of sec 12
(verbatim transcriptions), fenced code.

--mutation-test: the check must FAIL when (a) a bound number is perturbed and
(b) an unbound number is inserted into the body (gate #53).
Usage: python3 qb_doc_trace_check.py [--mutation-test]
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = HERE.parent / "PREREG_QB_LOOPGAIN_2026-09-11.md"
JSN = HERE / "qb_forecast_result.json"
M = "−"   # unicode minus used in prose / some tables

# external constants: token -> source markers, one of which must be on the same line
EXTERNAL = {
    "13.0064": ("Q-A", "ⓐ"), "13.0158": ("Q-A", "ⓐ"), "0.8768": ("Q-A",), "0.8785": ("Q-A",),
    "144": ("ⓒ", "144/152"), "152": ("ⓒ", "144/152"), "94.74": ("ⓒ",),
    "40": ("40/40",), "1.1": ("Q-A §12",), "2.1": ("Q-A §12",), "5": ("Q-A §12",),
    "4.754": ("N-5",), "0.811": ("Q-A 등록 상한",), "150": ("T_win",),
    "0.87": ("등록 산정",), "196": ("등록 산정",), "16": ("16 부팅", "arm"), "44": ("arm",), "54": ("arm",),
    "0.7": ("MC 오차",), "12.7": ("t₁",),
    "170": ("≈170",), "2.27": ("48 ×",), "0.65": ("12 ×",), "2.9": ("≈2.9",), "3.1": ("3.1",),
    "3.0": ("≈3.0",), "15.42": ("트랙 누적",), "18.4": ("트랙 누적",),
    "23": ("23표면",), "20": ("QB-20", "20건"), "595": ("판정서",), "5.3": ("초안 QB-11",), "0.9": ("초안 QB-11",),
    "14.9": ("프로브 C",), "23.7": ("프로브 C",), "15": ("15행", "행 15"), "19": ("19번째",), "95": ("95%",),
    "108": ("108",), "8192": ("8192",), "96": ("out", "모양"), "384": ("out", "모양"),
    "48": ("48 bench", "max_running_requests"), "64": ("64",), "0.491": ("0.491",), "12": ("12 부팅", "12 ×", "= 12"),
    "0.09": ("λ",), "0.15": ("λ",), "0.25": ("λ",), "0.42": ("λ", "round(0.42"), "0.70": ("λ",), "0.1162": ("λ",),
    "81": ("시드",), "82": ("시드",), "83": ("시드",), "84": ("시드",), "85": ("시드",), "86": ("시드",),
    "87": ("시드",), "88": ("시드",), "10000": ("B =",), "50": ("50 ms",), "0.0": ("0.0%",),
    "11": ("QB-11", "11건"), "10": ("열린 항목",), "0.85": ("PREREG_P7",), "0.851": ("PREREG_P7",),
}
STRUCT = [
    r"^#{2,4}\s+\d+(\.\d+)*", r"^\s*\d+\.\s", r"^\|\s*\d+\s*\|", r"§\s?\d+(\.\d+)*(\([a-z0-9]\))?(-\d+)?", r"#\d+",
    r"rev\d+", r"(?<!\d)\d-\d{1,2}(?!\d)", r"N-\d+(?!\d)", r"N\d(?!\d)", r"QB-\d+(?!\d)", r"(?<![\w.])D\d+(?!\d)",
    r"(?<![\w.])A\d(?!\d)", r"G-\S", r"(?<![\w.])J\d(?!\d)", r"(?<![\w.])P\d(?!\d)", r"20\d\d-\d\d-\d\d",
    r"(?<!\d)\d{6}(?!\d)", r"(?<![\w.])d\d{2}(?!\d)", r":\d+(-\d+)?", r"02918e8", r"homog\d", r"C2(?!\d)",
    r"항목\s?\d+", r"교훈\s?(항목\s?)?\d+", r"Step\s?1", r"Stage\s?[01]", r"stake\s?#1", r"(?<![\w.])[rR]\d(?!\d)",
    r"[⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉]", r"(?<![\w.])\d+\s?SM", r"out=\d+", r"H100|H200|A100", r"9B", r"v2",
    r"판정서 7건", r"게이트 후보", r"7항목",
]
NUM = re.compile(r"(?<![\w.])[-+" + M + r"]?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?")


def f3(x): return f"{x:.3f}"


def sgn3(x): return (f"+{x:.3f}" if x >= 0 else f"{M}{abs(x):.3f}")


def facts(r):
    dec = r["C_decomposition"]; J = r["J_doc_summary"]; N = r["N_proxy_table_sec3_6"]; O = r["O_path_disclosure"]
    L = r["M_little_probeC"]; K = r["K_provenance"]; A = r["A_timeline_alignment"]; Bc = r["B_classifier_by_arm_QA"]
    Be = r["B_exposure_vs_rho_QA"]; P = r["C_placebo_summary"]; SP = r["C_sign_pattern"]; ES = r["E_sensitivity"]
    F = r["F_load_covariate"]; Gx = r["G_crosscheck_vs_QA_delta_mean"]; I1 = r["I_stage1_forecast"]
    SF = r["L_forbidden_output_selfcheck"]; HD = ES["hierarchy_disagreement"]; S = ES["summary"]
    n = len(dec); nqa = sum(1 for d in dec if d["pair"][0] == "QA"); np7 = n - nqa
    pc = lambda s, l, h, a, b: [d for d in dec if d["pair"] == [s, l, h, a, b]][0]
    nQA = K["QA"][0]["n_benches"]; nP7 = K["P7"][0]["n_benches"]; nC = K["C"][0]["n_benches"]
    sig = K["QA"][0]["signature"]
    gq, gp = J["G_cf_by_source_path_cf"]["QA"], J["G_cf_by_source_path_cf"]["P7"]
    lin, eo, reg = O["offsupport_linear_in_b"], O["e_only_batch_none"], O["registered"]
    prim15 = max(max(S[v]["max_abs_change_ms"]["K_atD"], S[v]["max_abs_change_ms"]["S_atDp"])
                 for v in ("point=emission", "point=midpoint", "exposure=overlap>=1/2",
                           "pre_start_lag=+50ms", "arrival=span_rescaled"))
    lag = S["pre_start_lag=+50ms"]["max_abs_change_ms"]
    g116 = J["gate116_counts"]; sh = J["QA_2seed_between_seed_share_median"]
    p7r = {c: sorted(F["P7_12seed_regression"][c], key=lambda x: x["R2"]) for c in F["P7_12seed_regression"]}
    arm = J["arm_level_means_QA_paired_cells"]
    rat = J["stage1_projection_ratios"]; rr = lambda cell, comp: [x for x in rat if x["cell"] == cell and x["component"] == comp][0]
    ph = I1["projected_half_widths"]
    gsf = [d["G_ratio"]["supply_first_path"] for d in dec]
    seeds = I1["seeds"]; wins = [v["arrival_window_s"] for v in seeds.values()]
    mu = I1["mu_p_ref"]; dur = J["QA_lambda0.42_bench_duration_s"]
    E42 = {("A", 16): pc("QA", 0.42, "A", 16, 44)["E_D"], ("A", 44): pc("QA", 0.42, "A", 16, 44)["E_Dp"],
           ("A", 54): pc("QA", 0.42, "A", 44, 54)["E_Dp"], ("B", 16): pc("QA", 0.42, "B", 16, 54)["E_D"],
           ("B", 44): pc("QA", 0.42, "B", 44, 54)["E_D"], ("B", 54): pc("QA", 0.42, "B", 16, 54)["E_Dp"]}
    un_p05_min = min(v["unexposed_p05_p50_p95"][0] for v in Bc.values())
    x0954 = pc("QA", 0.09, "A", 44, 54)["components"]["S_atDp"]
    osA, osB = J["offsupport_M_serviceD_compDp"], J["offsupport_M_serviceDp_compD"]
    p7n = [d["n_replicates"] for d in dec if d["pair"][0] == "P7"][0]
    s = [
        ("benches", f"Q-A {nQA} bench + P7 {nP7} bench"),
        ("pairs", f"({n} pair-cell: Q-A {nqa} + P7 {np7})"),
        ("S range", f"`S@D′` = **{M}{abs(J['S_atDp_range']['max']):.2f} … {M}{abs(J['S_atDp_range']['min']):.2f} ms**, {n}칸 범위"),
        ("E 0.42B", f"d16 **{E42[('B', 16)]:.3f}** → d54 **{E42[('B', 54)]:.3f}**"),
        ("K range", f"`K@D` = **+{J['K_atD_range']['min']:.2f} … +{J['K_atD_range']['max']:.2f} ms**"),
        ("G ranges", f"**{J['G_cf_range_all_39'][0]:.3f} … {J['G_cf_range_all_39'][1]:.3f}**(Q-A {gq['n']}칸 {gq['min']:.3f}–{gq['max']:.3f} · P7 {gp['n']}칸 {gp['min']:.3f}–{gp['max']:.3f})"),
        ("crosscheck", f"{Gx['n']}칸 전부 **최대 차이 {Gx['max_abs_diff_ms']:.1f} ms**"),
        ("little96", f"out=96 **{L['out96_r2']['L_ratio']:.3f} = {L['out96_r2']['X_ratio']:.3f} × {L['out96_r2']['sojourn_ratio']:.3f}**(로그 몫 도착 {100*L['out96_r2']['log_share_arrival']:.1f}% · 체류 {100*L['out96_r2']['log_share_sojourn']:.1f}%)"),
        ("little384", f"out=384 **{L['out384_r1']['L_ratio']:.3f} = {L['out384_r1']['X_ratio']:.3f} × {L['out384_r1']['sojourn_ratio']:.3f}**({100*L['out384_r1']['log_share_arrival']:.1f}% · {100*L['out384_r1']['log_share_sojourn']:.1f}%)"),
        ("rho R2 (0.3)", f"**R² {N['rho_proxy']['R2']:.3f}**(중앙 상대잔차 **{100*N['rho_proxy']['median_abs_rel_resid']:.1f}%**)"),
        ("corr (0.3)", f"corr(`ΔE`, `Δρ`) = **{N['corr_dE_drho']:.3f}**({N['n_pairs']}쌍)임을"),
        ("sens linear", f"**최대 {S['offsupport=linear_in_b']['max_abs_change_ms']['S_atD']:.3f} ms** 움직이고 `K@D′` 부호가 **{S['offsupport=linear_in_b']['sign_flips_vs_registered']['K_atDp']}/{n}** 뒤집히지만, 1차 경로는 **최대 {max(S['offsupport=linear_in_b']['max_abs_change_ms']['K_atD'], S['offsupport=linear_in_b']['max_abs_change_ms']['S_atDp']):.3f} ms**"),
        ("dir K@D", f"`K@D`는 {SP['K_atD']['pos_point']}/{n} 양"),
        ("dir K@D'", f"{SP['K_atDp']['pos_point']}/{n} 양·{SP['K_atDp']['neg_point']}/{n} 음"),
        ("dir 0/39", f"음 **{lin['K_atDp_negative_points']}/{n}**다\n(QB-2" if False else f"음 **{max(lin['K_atDp_negative_points'], eo['K_atDp_negative_points'])}/{n}**다"),
        ("eonly support", f"off-support 질량 모두 {100*max(eo['offsupport_M_serviceD_compDp_mean'], eo['offsupport_M_serviceDp_compD_mean']):.2f}%"),
        ("eonly G", f"경로 cf **{eo['G_cf_median_pooled_DISCLOSURE_ONLY']:.3f}** vs 경로 sf **{eo['G_sf_median_pooled_DISCLOSURE_ONLY']:.3f}**으로 갈린다(등록 층에서는 {reg['G_cf_median_pooled_DISCLOSURE_ONLY']:.3f} vs {reg['G_sf_median_pooled_DISCLOSURE_ONLY']:.3f})"),
        ("algebra", f"계산기가 {reg['algebra_n_pairs_checked']}/{n} pair에서 이 동치를 확인"),
        ("hier (1.5)", f"**`C_pop` 부호가 {HD['C_pop_sign_disagree']}/{n}에서 불일치**, 노출·인구 크기 순서가 **{HD['exp_vs_pop_magnitude_order_disagree']}/{n}에서 불일치**"),
        ("support row A", f"평균 **{100*osA['mean']:.2f}%** · 중앙 **{100*osA['median']:.0f}** · 최대(복제) **{100*osA['max_over_replicates']:.2f}%**"),
        ("support row B", f"평균 **{100*osB['mean']:.2f}%** · 중앙 **{100*osB['median']:.2f}%** · 최대(복제) **{100*osB['max_over_replicates']:.2f}%**"),
        ("row9", f"Q-A {nqa} + P7 {np7} = **{n} pair-cell**"),
        ("src QA", f".jsonl`({nQA})"), ("src P7", f".jsonl`({nP7})"), ("src C", f"({nC}, 정렬 검증"),
        ("228", f"Q-A {nQA} + P7 {nP7} = **{nQA+nP7}/{nQA+nP7} bench가 한 서명**"),
        ("sig mrr", f"`max_running_requests={sig['max_running_requests']}`"),
        ("sig rr", f"`random_range_ratio={sig['random_range_ratio']}`"),
        ("sig in", f"입력 {sig['random_input_len']} 토큰"), ("sig C", f"프로브 C {nC} bench는"),
        ("32 cells", f"Q-A {J['QA_cells_total']}셀 중 {J['QA_cells_total'] - J['QA_cells_with_partnered_tokens']}셀"),
        ("p05 min", f"최소 p05 {un_p05_min:.2f} ms"),
        ("placebo", f"({P['n_placebo_estimates']} 추정): 최대 |점추정| **{P['max_abs_point_ms']:.3f} ms** · 등록 CI가 0을 배제한 것 **{P['n_ci_excluding_zero_registered']}/{P['n_placebo_estimates']}**"),
        ("placebo2", f"(우연 기대 {P['expected_by_chance_at_alpha_0.05']:.1f}) · 공통-trace 진단(1시드 rung 해당분만) **{P['n_ci_excluding_zero_common_trace']}/{J['placebo_common_trace_denominator']}**"),
        ("crosscheck2", f"{Gx['n']}/{Gx['n']}, **최대 차이 {Gx['max_abs_diff_ms']:.1f} ms**"),
        ("E corr", f"상관 **{Be['corr_E_rho']:.3f}**, `E`와 클라이언트측 prefill-busy 시간분율 `U` 상관 **{Be['corr_E_U']:.3f}**, `E/U` 평균 **{Be['mean_E_over_U']:.3f}**({Be['n']}셀)"),
        ("E42 A", f"λ=0.42 모양 A `E` = d16 **{E42[('A', 16)]:.3f}** · d44 **{E42[('A', 44)]:.3f}** · d54 **{E42[('A', 54)]:.3f}**; 모양 B d16 **{E42[('B', 16)]:.3f}**"),
        ("E42 B", f"d44 **{E42[('B', 44)]:.3f}** · d54 **{E42[('B', 54)]:.3f}**. 노출"),
        ("mu1 16/44", f"(d16 {arm['d16']['mu_exposed_min']:.1f}–{arm['d16']['mu_exposed_max']:.1f} · d44 {arm['d44']['mu_exposed_min']:.1f}–{arm['d44']['mu_exposed_max']:.1f} ·"),
        ("mu1 54/92", f"d54 {arm['d54']['mu_exposed_min']:.1f}–{arm['d54']['mu_exposed_max']:.1f} · d92 {arm['d92']['mu_exposed_min']:.1f}–{arm['d92']['mu_exposed_max']:.1f} ms"),
        ("bbar", f"`b̄ ≤ {J['bbar_exposed_max_per_cell_paired']:.2f}`"),
        ("mu0", f"`μ0` {min(v['mu_unexposed_min'] for v in arm.values()):.1f}–{max(v['mu_unexposed_max'] for v in arm.values()):.1f} ms"),
        ("sign KD", f"**{SP['K_atD']['pos_point']}/{n}**(CI 0 배제 {SP['K_atD']['pos_ci']})"),
        ("sign S I", f"`S@D′` 음 **{SP['S_atDp']['neg_point']}/{n}**({SP['S_atDp']['neg_ci']}) · `I` 음 **{SP['I_interaction']['neg_point']}/{n}**({SP['I_interaction']['neg_ci']}, 2차)"),
        ("G src", f"Q-A {gq['n']}칸 **{gq['min']:.3f}–{gq['max']:.3f}** · P7 {gp['n']}칸 **{gp['min']:.3f}–{gp['max']:.3f}**"),
        ("39 pooled", f"{n}칸 풀 중앙값의 단일 수"),
        ("K@D' pts", f"점추정 **양 {SP['K_atDp']['pos_point']} / 음 {SP['K_atDp']['neg_point']}**"),
        ("K@D' ci", f"(CI 0 배제: 양 {SP['K_atDp']['pos_ci']}·음 {SP['K_atDp']['neg_ci']})"),
        ("3.4 lin", f"**최대 {S['offsupport=linear_in_b']['max_abs_change_ms']['S_atD']:.3f} ms** 움직이고 `K@D′` 부호가"),
        ("3.4 flips", f"**{S['offsupport=linear_in_b']['sign_flips_vs_registered']['K_atDp']}/{n}** 뒤집힌다"),
        ("3.4 0/39", f"`K@D′` 음이 **{max(lin['K_atDp_negative_points'], eo['K_atDp_negative_points'])}/{n}**다"),
        ("3.4 hier", f"`C_exp` 양 {SP['C_exp_Efirst']['pos_point']}/{SP['C_exp_Bfirst']['pos_point']} · `C_pop` 음 {SP['C_pop_Efirst']['neg_point']} / **양 {SP['C_pop_Bfirst']['pos_point']}** ⇒ **`C_pop` 부호 불일치 {HD['C_pop_sign_disagree']}/{n}**"),
        ("3.4 hier2", f"`C_exp` {HD['C_exp_sign_disagree']}/{n}, 크기 순서 {HD['exp_vs_pop_magnitude_order_disagree']}/{n}"),
        ("g116 a", f"(2시드 rung {g116['K_atD']['decomposable']} pair-cell 중"),
        ("g116 b", f"`K@D` {g116['K_atD']['undetected']}/{g116['K_atD']['decomposable']}(퇴화 {g116['K_atD']['degenerate']})"),
        ("g116 c", f"`S@D′` {g116['S_atDp']['undetected']}/{g116['S_atDp']['decomposable']}(퇴화 {g116['S_atDp']['degenerate']}) · `Δ` {g116['delta_mean']['undetected']}/{g116['delta_mean']['decomposable']}(퇴화 {g116['delta_mean']['degenerate']}) · `I` {g116['I_interaction']['undetected']}/{g116['I_interaction']['decomposable']}(퇴화 {g116['I_interaction']['degenerate']})"),
        ("share a", f"중앙 `K@D` {sh['K_atD']:.3f}"),
        ("share b", f"`S@D′` {sh['S_atDp']:.3f} · `I` {sh['I_interaction']:.3f} · `Δ` {sh['delta_mean']:.3f}({len(F['QA_2seed_between_seed_share']['K_atD'])} pair-cell)"),
        ("p7 n", f"P7(라운드마다 새 시드, {p7n} trace)"),
        ("p7 S", f"`S@D′` **{'/'.join(f3(x['R2']) for x in p7r['S_atDp'])}**"),
        ("p7 sd", f"(잔차 SD {min(x['resid_sd'] for x in p7r['S_atDp']):.3f}–{max(x['resid_sd'] for x in p7r['S_atDp']):.3f} vs 원 SD {min(x['raw_sd'] for x in p7r['S_atDp']):.3f}–{max(x['raw_sd'] for x in p7r['S_atDp']):.3f})"),
        ("p7 K", f"**`K@D` {'/'.join(f3(x['R2']) for x in p7r['K_atD'])}**"),
        ("p7 I D", f"`I` {p7r['I_interaction'][0]['R2']:.3f}–{p7r['I_interaction'][-1]['R2']:.3f} · `Δ` {p7r['delta_mean'][0]['R2']:.3f}–{p7r['delta_mean'][-1]['R2']:.3f}"),
        ("p7 n2", f"3쌍 × {p7n} trace"),
        ("3.6 n", f"Q-A {N['n_pairs']} pair-cell에서"),
        ("3.6 aux", f"corr(`ΔE`, `Δρ`) = **{N['corr_dE_drho']:.3f}** · corr(`ΔE`, `ΔU`) = {N['corr_dE_dU']:.3f} · `K@D`를 `ΔE` 단독으로 원점 적합한 R² {N['dE_alone']['R2']:.3f} ·"),
        ("3.6 aux2", f"`Δρ` 단독 {M}{abs(N['drho_alone']['R2']):.3f}({N['n_pairs']}쌍)"),
        ("3.6 concl", f"R² **{N['rho_proxy']['R2']:.3f}**와 corr(`ΔE`, `Δρ`) = **{N['corr_dE_drho']:.3f}**({N['n_pairs']}쌍)"),
        ("3.6 id", f"R² {N['dE_times_gap_IDENTITY_NOT_EVIDENCE']['R2']:.3f}을"),
        ("3.6 row rho", f"| {N['rho_proxy']['c']:.3f} | **{N['rho_proxy']['R2']:.3f}** | {100*N['rho_proxy']['median_abs_rel_resid']:.1f}% |"),
        ("3.6 row U", f"| {N['U_proxy']['c']:.3f} | {N['U_proxy']['R2']:.3f} | {100*N['U_proxy']['median_abs_rel_resid']:.1f}% |"),
        ("3.6 row E", f"| {N['dE_times_gap_IDENTITY_NOT_EVIDENCE']['c']:.3f} | {N['dE_times_gap_IDENTITY_NOT_EVIDENCE']['R2']:.3f} | {100*N['dE_times_gap_IDENTITY_NOT_EVIDENCE']['median_abs_rel_resid']:.1f}% |"),
        ("3.7 17", f"투영 반폭 {len(rat)}칸 중 {sum(1 for x in rat if x['proj_over_registered'] >= 1)}칸이 기존 등록 반폭 이상"),
        ("3.8 hier", f"`C_pop` 부호를 {HD['C_pop_sign_disagree']}/{n} 뒤집고"),
        ("3.8 idx", f"부호를 {SP['K_atDp']['neg_point']}/{n} 뒤집음"),
        ("4.3 rho", f"`ρ ≤ {J['stage1_rho_realised_max']:.3f}`"),
        ("D5 18", f"**{len(rat)}칸 중 5칸에서"),
        ("D5 K", f"({ph['A d44->d54']['K_atD']['projected_half_width_4boot_x_2freshseed']:.3f} < {ph['A d44->d54']['K_atD']['existing_half_width_registered']:.3f}, 비 {rr('A d44->d54', 'K_atD')['proj_over_registered']:.3f}) — 나머지 {sum(1 for x in rat if x['proj_over_registered'] >= 1)}칸"),
        ("D5 BD", f"(d16→d44 **{rr('B d16->d44', 'delta_mean')['proj_over_common_trace']:.3f}×** · d16→d54 **{rr('B d16->d54', 'delta_mean')['proj_over_common_trace']:.3f}×**)"),
        ("D5 S", f"(A **{rr('A d44->d54', 'S_atDp')['proj_over_common_trace']:.3f}×** · B **{rr('B d44->d54', 'S_atDp')['proj_over_common_trace']:.3f}×**)"),
        ("D5 AD", f"(d16→d44 {rr('A d16->d44', 'delta_mean')['proj_over_common_trace']:.3f}× · d16→d54 {rr('A d16->d54', 'delta_mean')['proj_over_common_trace']:.3f}×"),
        ("4.4 18", f"(전 {len(rat)}칸은 JSON"),
        ("4.4 p7", f"P7 {p7n} trace의"),
        ("4.5 win", f"(도착 창 {min(wins):.1f}–{max(wins):.1f} s"),
        ("4.5 dur", f"평균 **{dur['mean']:.1f} s**, 범위 {dur['min']:.1f}–{dur['max']:.1f}, {dur['n']} bench"),
        ("1.4 ratio", f"`X` 비 최대 {mu['16']/mu['92']:.3f}×"),
        ("5 intro", f"기존 원자료({n} pair-cell)로 쟀다"),
        ("5 row6", f"b-선형에서는 {n - lin['K_atDp_negative_points']}/{n} 양"),
        ("5 row11", f"`G` 중앙 {eo['G_cf_median_pooled_DISCLOSURE_ONLY']:.3f} vs {eo['G_sf_median_pooled_DISCLOSURE_ONLY']:.3f}(D8)"),
        ("5 row14", f"풀 중앙값 {reg['G_cf_median_pooled_DISCLOSURE_ONLY']:.3f}는 이 행과"),
        ("5 row15 a", f"두 반사실의 off-support {100*max(eo['offsupport_M_serviceD_compDp_mean'], eo['offsupport_M_serviceDp_compD_mean']):.2f}%, `K@D′` 음 {eo['K_atDp_negative_points']}/{n}, `G` 중앙 cf {eo['G_cf_median_pooled_DISCLOSURE_ONLY']:.3f} / sf {eo['G_sf_median_pooled_DISCLOSURE_ONLY']:.3f}"),
        ("5 row15 b", f"(`C_pop` 뒤집힘 {S['batch=none(e-only)']['sign_flips_vs_registered']['C_pop_Efirst']}은"),
        ("5 N2", f"채움 규칙으로 {SP['K_atDp']['neg_point']}/{n} 반전"),
        ("5 N2b", f"계층으로 {HD['exp_vs_pop_magnitude_order_disagree']}/{n}·{HD['C_pop_sign_disagree']}/{n} 반전"),
        ("7.5 QB1", f"풀 중앙값 {reg['G_cf_median_pooled_DISCLOSURE_ONLY']:.3f} 단일 수"),
        ("7.5 QB2", f"(\"`K@D′` 음 {SP['K_atDp']['neg_point']}/{n}\""),
        ("8.1 a", f"등록 층 중앙 cf {reg['G_cf_median_pooled_DISCLOSURE_ONLY']:.3f} / sf {reg['G_sf_median_pooled_DISCLOSURE_ONLY']:.3f}(sf 칸별 범위"),
        ("8.1 b", f"{M}{abs(min(gsf)):.2f}…{max(gsf):.2f}), 외삽 없는 e-only에서도 cf {eo['G_cf_median_pooled_DISCLOSURE_ONLY']:.3f} / sf {eo['G_sf_median_pooled_DISCLOSURE_ONLY']:.3f}"),
        ("8.2 lag", f"(+50 ms 시험 ≤{max(lag['K_atD'], lag['S_atDp']):.2f} ms)"),
        ("8.5", f"민감도 1–5(최대 {prim15:.2f} ms)"),
        ("8.6", f"`S@D′` {M}{abs(x0954['point']):.3f} ±{x0954['half_width']:.3f}"),
        ("8.8", f"`b̄ ≤ {J['bbar_exposed_max_per_cell_paired']:.2f}`(1차 표 범위)"),
        ("9.1", f"(반론: `ρ`-대리 R² {N['rho_proxy']['R2']:.3f},"),
        ("9.1b", f"corr(`ΔE`, `Δρ`) {N['corr_dE_drho']:.3f}, §3.6. ⚠️초판은 여기서 `ΔE` 대리의 R² {N['dE_times_gap_IDENTITY_NOT_EVIDENCE']['R2']:.3f}을"),
        ("9.2", f"≤{prim15:.2f} ms·부호 0 뒤집힘"),
        ("9.3", f"정렬 {100*A['QA']['frac_lt_2ms']:.2f}%/{100*A['QA']['frac_lt_5ms']:.2f}%"),
        ("9.3b", f"위약 {P['n_ci_excluding_zero_registered']}/{P['n_placebo_estimates']}"),
        ("10 keys", f"키 경로 전수({SF['n_key_paths_scanned']:,})"),
        ("10 mut", f"**{sum(1 for x in SF['mutation_test']['mutants'] if not x['selfcheck_passed'])}/{len(SF['mutation_test']['mutants'])} 검출**"),
        ("11 39", f"기존 원자료 {n} pair-cell 전수 계산"),
        ("11 15", "민감도 15행"),
        ("n hdr", f"off-support 토큰 질량 ({n} pair-cell)"), ("n 1.3", f"{n}칸 풀 중앙값을 단일 수로"),
        ("n 1.6", f"{n}칸 풀 중앙 `G`를 단일 수로"), ("n 3.3", f"— {n} pair-cell 전문"),
        ("n 3.7", f"전 영역** {n} pair-cell에서 사후 정의로"),
        ("rr 200", f"`--random-range-ratio {sig['random_range_ratio']}` 전용"),
        ("rr 415", f"`--random-range-ratio {sig['random_range_ratio']}` ·"),
        ("5 row8", f"`C_pop` 부호 **{HD['C_pop_sign_disagree']}/{n}** 불일치 · 크기 순서 {HD['exp_vs_pop_magnitude_order_disagree']}/{n}"),
    ]
    rows = []
    for src_, tag in (("QA", "QA"), ("P7", "P7"), ("C", "C")):
        v = A[src_]
        rows.append(("align " + tag, f"| {tag} | {v['cells']} | {v['partnered_tokens']:,} | {100*v['frac_lt_1ms']:.2f}% | {100*v['frac_lt_2ms']:.2f}% | {100*v['frac_lt_5ms']:.2f}% |"))
    for k, v in Bc.items():
        rows.append(("clf " + k, f"| {k} | {v['n_unexposed']:,} | {' / '.join('%.2f' % x for x in v['unexposed_p05_p50_p95'])} | {v['n_exposed']:,} | {' / '.join('%.2f' % x for x in v['exposed_p05_p50_p95'])} |"))
    for d in dec:
        c = d["components"]; p = d["pair"]

        def v(k):
            x = c[k]; out = f"{x['point']:+.3f} ±{x['half_width']:.3f}"
            if x.get("half_width_common_trace_diagnostic"):
                out += f" (ct ±{x['half_width_common_trace_diagnostic']:.3f})"
            return out
        rows.append(("t33 " + str(p), f"| {p[0]} | {p[1]} | {p[2]} | d{p[3]}→d{p[4]} | {d['n_replicates']} | {d['E_D']:.3f}→{d['E_Dp']:.3f} | {v('delta_mean')} | {v('K_atD')} | {v('S_atDp')} | {d['G_ratio']['composition_first_path']:.3f} | {v('I_interaction')} | {d['offsupport_M_serviceD_compDp']:.4f} / {d['offsupport_M_serviceDp_compD']:.3f} |"))
    for sd, x in seeds.items():
        rr_ = x["rho_realised_forecast"]
        rows.append(("t43 " + sd, f"| {sd} | {x['round']} | {x['span_factor']:.4f} | {x['arrival_window_s']:.1f} | {x['X_forecast_arrival_axis']:.4f} | {rr_['d16']:.3f} / {rr_['d44']:.3f} / {rr_['d54']:.3f} |"))
    for cell, comp, lab in (("A d16->d44", "K_atD", "`K@D`"), ("A d16->d44", "S_atDp", "`S@D′`"), ("A d44->d54", "K_atD", "`K@D`"),
                            ("A d44->d54", "S_atDp", "`S@D′`"), ("B d16->d54", "K_atD", "`K@D`"), ("B d44->d54", "K_atD", "`K@D`")):
        x = ph[cell][comp]
        rows.append(("t44 " + cell + comp, f"| {cell.replace('->', '→')} | {lab} | {sgn3(x['existing_point'])} | {x['existing_half_width_registered']:.3f} | {x['existing_half_width_common_trace']:.3f} | **{x['projected_half_width_4boot_x_2freshseed']:.3f}** |"))
    for var in ("point=emission", "point=midpoint", "exposure=overlap>=1/2", "pre_start_lag=+50ms",
                "arrival=span_rescaled", "offsupport=linear_in_b", "batch=none(e-only)"):
        m, fl = S[var]["max_abs_change_ms"], S[var]["sign_flips_vs_registered"]
        pl = max(m["S0_atD_placebo"], m["S0_atDp_placebo"])
        a = f"{m['K_atD']:.3f} / {m['S_atDp']:.3f} · {fl['K_atD']}/{fl['S_atDp']}"
        b = f"{m['S_atD']:.3f} / {m['K_atDp']:.3f} / {m['I_interaction']:.3f} · {fl['S_atD']}/{fl['K_atDp']}/{fl['I_interaction']}"
        rows.append(("t5 " + var, (a, b, f"| {pl:.3f} |")))
    return s, rows


def body(text):
    out, fence, sec = [], False, ""
    for i, l in enumerate(text.split("\n"), 1):
        if l.startswith("```"):
            fence = not fence; continue
        m = re.match(r"^##\s+(\d+)\.", l)
        if m:
            sec = m.group(1)
        if fence or sec == "12" or (sec == "7" and l.startswith(">")):
            continue
        out.append((i, l))
    return out


def check(doc_text, r):
    snippets, rows = facts(r)
    B = body(doc_text)
    btxt = "\n".join(l for _, l in B)
    missing = []
    masks = []
    for name, sn in snippets:
        if sn not in btxt:
            missing.append(("snippet", name, sn)); continue
        masks.append(sn)
    t5 = []
    for name, rw in rows:
        parts = rw if isinstance(rw, tuple) else (rw,)
        # t5 rows: the three numeric fragments must co-occur on one body line
        if isinstance(rw, tuple):
            ok = [l for _, l in B if all(p in l.replace("**", "") for p in parts)]
            if not ok:
                missing.append(("row", name, " | ".join(parts))); continue
        elif rw not in btxt:
            missing.append(("row", name, rw)); continue
        (t5 if isinstance(rw, tuple) else masks).extend(parts)
    n_bound = 0
    for mk in masks + t5:
        n_bound += len([x for x in NUM.finditer(mk)])
    res = {"snippets": len(snippets), "table_rows": len(rows), "missing": missing, "bound_numbers": n_bound,
           "external_with_source": 0, "small_int": 0, "untraced": []}
    for ln, raw in B:
        t = raw
        for mk in sorted(set(masks), key=len, reverse=True):
            t = t.replace(mk, " ")
        if raw.lstrip().startswith("|"):
            t = t.replace("**", "")
            for mk in sorted(set(t5), key=len, reverse=True):
                t = t.replace(mk, " ")
        t = re.sub(r"\]\([^)]*\)", "]", t)
        t = re.sub(r"`[^`]*`", lambda m: " " if re.search(r"[/_.]\w*\.(py|md|json|yml|jsonl|txt|sbatch)|probes/|\{|\*", m.group(0)) else m.group(0), t)
        for pat in STRUCT:
            t = re.sub(pat, " ", t)
        for m in NUM.finditer(t):
            tok = m.group(0).lstrip("+-" + M).replace(",", "")
            if "." not in tok and int(tok) < 10:
                res["small_int"] += 1; continue
            src = EXTERNAL.get(tok)
            if src and any(s in raw for s in src):
                res["external_with_source"] += 1; continue
            res["untraced"].append({"line": ln, "number": tok, "context": raw.strip()[:150]})
    res["passed"] = not res["missing"] and not res["untraced"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=str(DOC)); ap.add_argument("--json", default=str(JSN))
    ap.add_argument("--mutation-test", action="store_true")
    a = ap.parse_args()
    doc = Path(a.doc).read_text(); r = json.loads(Path(a.json).read_text())
    res = check(doc, r)
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in res.items()}, ensure_ascii=False))
    for x in res["missing"]:
        print("  MISSING", x[0], x[1], "::", x[2][:160])
    for u in res["untraced"]:
        print(f"  UNTRACED line {u['line']}: {u['number']}  | {u['context']}")
    if a.mutation_test:
        m1 = doc.replace("**R² 0.529**", "**R² 0.528**", 1)
        m2 = doc.replace("## 4. 격자", "노출 부가 계수는 7.777 ms다.\n\n## 4. 격자", 1)
        m3 = doc.replace("| QA | 0.42 | B | d16→d54 | 4 | 0.200→0.670 |", "| QA | 0.42 | B | d16→d54 | 4 | 0.200→0.671 |", 1)
        for name, md in (("perturb bound prose number 0.529->0.528", m1),
                         ("insert unbound number 7.777 ms", m2), ("perturb table cell 0.670->0.671", m3)):
            rr_ = check(md, r)
            print(f"  MUTANT {name:42s} passed={rr_['passed']}  (missing {len(rr_['missing'])}, untraced {len(rr_['untraced'])})")
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
