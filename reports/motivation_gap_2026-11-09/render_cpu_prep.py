#!/usr/bin/env python3
"""Render cpu_prep_2026-11-10.html from the workload manifest, the trace registry and hybrid_term_cells.json.  CPU only.
Usage: python3 -I render_cpu_prep.py --cells hybrid_term_cells.json --out cpu_prep_2026-11-10.html
Numbers come from the input files; the static text holds the record pointers for the retracted per-layer decode jobs and the vLLM code reading."""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WL = ROOT / "workspace" / "engine-port" / "results" / "motivation" / "workloads"
QS = ("p1", "p5", "p25", "p50", "p75", "p90", "p95", "p99")
esc = html.escape

NAME = {"sharegpt_first": "사용자 첫 발화 ShareGPT", "openchat_rounds": "openchat_sharegpt4 라운드",
        "arxiv_summ": "arXiv 요약 16k", "loogle_longdep": "LooGLE 장의존 QA"}


def dist_row(label, d):
    cells = "".join(f"<td>{d[q]:,}</td>" for q in QS)
    return f"<tr><th>{esc(label)}</th><td>{d['n']:,}</td><td>{d['mean']:,.1f}</td><td>{d['cv']}</td><td>{d['min']:,}</td>{cells}<td>{d['max']:,}</td></tr>"


def workload_section(man):
    out = []
    for w, part in man["workloads"].items():
        extra = ""
        if w == "arxiv_summ":
            o = part["observed_natural_output_cap2"]
            extra = (f"<p class=note>자연 출력(CAP2 관측, ignore_eos=False·상한 1,024, 요청 {o['n']:,}건): 평균 {o['mean']:,.1f}, "
                     f"p50 {o['p50']:,}, 1,024 도달 비율 {o['frac_at_cap_1024']:.1%}. 표의 '출력' 행은 초록 토큰 길이(참조값)이며 서빙 출력이 아니다.</p>")
        tf = part.get("token_file")
        tfs = f"토큰 파일 {tf['bytes'] / 1e6:.1f} MB, sha256 {tf['sha256'][:16]}…" if tf else f"토큰 파일: {part['source'].get('token_file_sha256', '')[:16]}… (기존)"
        out.append(
            f"<h3>{NAME[w]} <code>{w}</code></h3><p class=meta>{esc(part['rule'])}<br>출력: {esc(part['output_mode'])}. 프롬프트: {esc(part['prompt'])}. {tfs}</p>"
            f"<table><tr><th></th><th>n</th><th>평균</th><th>CV</th><th>최소</th>{''.join(f'<th>{q}</th>' for q in QS)}<th>최대</th></tr>"
            f"{dist_row('입력(토큰)', part['input'])}{dist_row('출력(토큰)', part['output'])}</table>{extra}")
    return "\n".join(out)


def cells_section(res):
    s = res["summary"]
    rows = res["rows"]
    keep = [r for r in rows if r["differs_corrected"] or r["differs_predictpy"]]
    tr = ""
    for r in keep:
        lam = r["lam_full_by_D"]
        tr += (f"<tr><td>{NAME[r['workload']]}</td><td>{esc(r['band'])}</td><td>{r['n']:,}</td><td>{r['cap']}</td><td>{r['rho']}</td>"
               f"<td>{r['ER']:,.0f}</td><td>{r['ctx_decode']:,.0f}</td><td>{r['D_full']}</td><td>{r['D_nostate_corrected']}</td><td>{r['D_nostate_predictpy']}</td>"
               f"<td class=num>{' / '.join(f'{lam[d]:.2f}' for d in ('16', '24', '34', '44', '64'))}</td><td>{', '.join(r['binding_full'])}</td></tr>")
    whole = [r for r in rows if r["band"].startswith("whole") and r["rho"] == 1.16 and r["cap"] == 48]
    wr = "".join(f"<tr><td>{NAME[r['workload']]}</td><td>{esc(r['band'])}</td><td>{r['ER']:,.0f}</td><td>{r['ctx_decode']:,.0f}</td><td>{r['D_full']}</td><td>{r['D_nostate_corrected']}</td>"
                 f"<td>{r['D_nostate_predictpy']}</td><td>{r['lam_full']:.2f}</td><td>{'영(0) — 모델 적용 불가' if r['degenerate_all_zero'] else ('갈림' if r['differs_corrected'] else '같음')}</td></tr>" for r in whole)
    oc = next(r for r in whole if r["workload"] == "openchat_rounds")
    lam = oc["lam_full_by_D"]
    note = (f"<p class=note>읽는 법: 갈리는 셀은 모두 항을 빼면 D*가 격자 한 칸 작아지는 방향이다(입력 약 256–8k 구간). 워크로드 전체 셀에서 안정적으로 갈리는 것은 openchat 라운드뿐이다: "
            f"포함 모델 D*={oc['D_full']}, 제거 모델 D*={oc['D_nostate_corrected']}, 포함 모델 안에서 λ_cap은 D34 {lam['34']:.2f}, D24 {lam['24']:.2f} req/s. "
            f"사용자 첫 발화 ShareGPT와 초록 길이 arXiv 요약은 갈리지 않고, 관측 자연 출력 arXiv 요약은 ρ 1.13에서만 갈리며 λ 차이가 0.01이다. 격자가 성겨 차이는 한 칸 이내로만 읽는다.</p>")
    return (note + f"<p>전체 {s['n_cells']}셀(워크로드·입력 대역 × cap {{48, 128}} × ρ {{1.13, 1.16, 1.19}}) 중 hybrid 항을 빼면 D*가 달라지는 셀: "
            f"보정 방식 <b>{s['n_differs_corrected']}</b>, predict.py 방식 {s['n_differs_predictpy']} (이 가운데 보정 방식에서는 안 갈리는 셀 {s['predictpy_differ_cells_not_in_corrected']}). "
            f"람다 상한이 전 D에서 0인 퇴화 셀 {s['degenerate_cells']}.</p>"
            "<p class=note>predict.py 방식은 F3 형태(D24)의 상태 항을 제거하지 않아 D24가 다른 D에 비해 불리해진다. 보정 방식은 모든 모드에서 b·b0(D)를 빼고 a(D)로 하한을 건다.</p>"
            f"<h3>워크로드 전체 셀 (cap 48, ρ 1.16)</h3><table><tr><th>워크로드</th><th>셀</th><th>E[R]</th><th>문맥</th><th>D*(hybrid 항 포함)</th><th>D*(항 제거, 보정)</th><th>D*(predict.py)</th><th>λ_cap</th><th>판정</th></tr>{wr}</table>"
            f"<h3>D*가 갈리는 셀 전체</h3><table><tr><th>워크로드</th><th>셀</th><th>n</th><th>cap</th><th>ρ</th><th>E[R]</th><th>문맥</th><th>D*(포함)</th><th>D*(제거, 보정)</th><th>D*(predict.py)</th><th>λ_cap(D16/24/34/44/64)</th><th>구속</th></tr>{tr}</table>")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    man = json.load(open(WL / "manifest.json"))
    reg = json.load(open(WL / "trace_registry.json"))
    res = json.load(open(a.cells))
    pair = reg["provisional_interference_pairs"]["20261111|3.0"]
    pool_bytes = sum(p.get("token_file", {}).get("bytes", 0) for p in man["workloads"].values())
    html_doc = f"""<!doctype html><html lang=ko><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>CPU 준비 결과</title>
<style>:root{{--bg:#fbfbf9;--fg:#1d2124;--muted:#5d6670;--line:#d9dde1;--acc:#245a7c;--warn:#8a4a00}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#16181b;--fg:#e7e9ec;--muted:#9aa3ad;--line:#333941;--acc:#7fb7dc;--warn:#e0a860}}}}
:root[data-theme=dark]{{--bg:#16181b;--fg:#e7e9ec;--muted:#9aa3ad;--line:#333941;--acc:#7fb7dc;--warn:#e0a860}}
body{{background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,sans-serif;max-width:980px;margin:0 auto;padding:24px 16px}}
h1{{font-size:24px}}h2{{font-size:19px;border-bottom:1px solid var(--line);padding-bottom:4px;margin-top:32px}}h3{{font-size:15.5px;margin:18px 0 4px}}
table{{border-collapse:collapse;display:block;overflow-x:auto;font-size:13px;margin:6px 0}}td,th{{border:1px solid var(--line);padding:3px 6px;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}
code{{font-size:.92em}}.meta{{color:var(--muted);font-size:13px;margin:2px 0 6px}}.note{{color:var(--warn);font-size:13.5px}}.tag{{color:var(--acc);font-weight:600}}li{{margin:3px 0}}</style>
<h1>Motivation 실험 CPU 준비 결과</h1><p class=meta>2026-11-10, GPU·VESSL 업로드 0. 측정 결과가 아니다. 워크로드 준비·철회 기록 수집·예측·코드 확인을 모았다. 예측 절은 인용 금지다.</p>

<h2>워크로드 길이 분포와 구성 카드 값</h2>
<p>토크나이저: Nano-9B-v2 <code>tokenizer.json</code> sha256 {man['tokenizer_sha256'][:16]}… (HF 캐시의 Base 스냅샷과 동일). 분위수 규칙: nearest rank. 재생성은 <code>workspace/engine-port/results/motivation/workloads/build_pools.py</code>, 전 항목 CPU.</p>
{workload_section(man)}
<h3>검증</h3><ul>
<li>사용자 첫 발화 ShareGPT: 원본 재다운로드 sha256 {man['workloads']['sharegpt_first']['source']['sha256'][:16]}… = Job 기록(sharegpt.sha256), 풀 sha256 {man['workloads']['sharegpt_first']['pool_sha256'][:16]}… = 기존 w1_lengths.json 바이트 일치.</li>
<li>openchat 라운드: 이전 점검의 p50 1597·(입력, 출력) 쌍 전체와 일치(<code>--check</code>).</li>
<li>arXiv 요약: 기존 토큰 파일 재검증 통과(sha256 일치). 합계 문서 5,906건.</li>
<li>LooGLE: 입력 p50 35,187·최소 13,459·답 p50 7은 이전 비교표와 일치. 96k 부트 조건(입력 ≤ 94,800)으로 5건(0.45%) 제외.</li></ul>

<h2>도착 trace와 간섭 셀 짝 설계</h2>
<ul>
<li><b>등록 seed</b> {', '.join(map(str, reg['seeds']))} (예비 {', '.join(map(str, reg['spare_seeds']))}). 마스터 해시는 <code>trace_registry.json</code>에 봉인되어 있고, rung은 (워크로드, seed, rate, 시간)으로 재현된다. 마스터의 도착 시각은 rate와 무관한 단위 Poisson이고 요청 순서는 풀 순열이므로, 모든 rung이 같은 요청 순서를 쓴다.</li>
<li><b>간섭 셀</b>: 같은 seed에서 짧은 스트림(사용자 첫 발화 ShareGPT 마스터의 prefix)을 고정하고, LooGLE 스트림을 전체 요청의 5%가 되도록 얹는다. 긴 요청 도착은 개수 고정 Poisson(n_long = round(0.05/0.95 · λ_short · T), t_k = T·S_k/S_(n+1))이라 300초당 도착 건수 잡음이 없다. <code>with_long</code>과 <code>without_long</code> trace의 차이는 긴 요청 레코드뿐이며, 짧은 스트림 해시가 일치함을 검사한다(<code>gen_traces.py seal</code>, 테스트 10개 통과).</li>
<li>예시(잠정 rate 3.0 req/s, seed 20261111): 짧은 {pair['n_short']}건·긴 {pair['n_long']}건, 짧은 스트림 해시 {pair['short_stream'][:16]}…, with {pair['with_long'][:16]}…, without {pair['without_long'][:16]}…. <span class=note>잠정 rate 격자는 사전 사다리 후 봉인한다.</span></li>
<li>결과는 요청 부류(짧은/긴)별로만 집계하고, 합산 attainment는 주 지표로 쓰지 않는다.</li></ul>

<h2>클러스터 볼륨 배치 필요 항목 (업로드 안 함)</h2>
<table><tr><th>항목</th><th>현재 위치</th><th>볼륨 경로(제안)</th><th>크기</th><th>상태</th></tr>
<tr><td>Nano tokenizer·ShareGPT V3 원본</td><td>HF 캐시, 로컬 재다운로드 확인</td><td>/data/hf/… (기존)</td><td>원본 672.8 MB</td><td>이미 있음(sharegpt.sha256의 경로)</td></tr>
<tr><td>arXiv 요약 토큰 파일</td><td>기존 Job의 볼륨 경로</td><td>/data/datasets/arxsum_x/arxsum_x_tokens_v1.bin</td><td>168.3 MB</td><td>이미 있음(rh7 기록)</td></tr>
<tr><td>openchat 라운드 토큰 파일 (새로)</td><td>로컬 scratchpad</td><td>/data/datasets/motivation_wl/openchat_rounds_tokens_v1.bin</td><td>{man['workloads']['openchat_rounds']['token_file']['bytes'] / 1e6:.1f} MB</td><td>배치 필요</td></tr>
<tr><td>LooGLE 토큰 파일 (새로)</td><td>로컬 scratchpad</td><td>/data/datasets/motivation_wl/loogle_longdep_tokens_v1.bin</td><td>{man['workloads']['loogle_longdep']['token_file']['bytes'] / 1e6:.1f} MB</td><td>배치 필요</td></tr></table>
<p class=meta>새로 놓을 파일 합계 {pool_bytes / 1e6:.1f} MB. 길이표·해시 목록은 저장소에 추적된다(하네스와 함께 Job에 들어감).</p>

<h2>철회된 층별 decode 결과의 원래 결함 (KISTI, Zamba2-2.7B, triton·no-cudagraph)</h2>
<p class=meta>새 층 종류 측정 설계용 사실 목록. 판정은 원 기록을 옮겼다. 줄 번호는 이 커밋 기준이다.</p>
<ul>
<li><b>전 기간 구조 문제</b>: 층별 계측(<code>_zt</code>)은 Python forward 안에 있어 cudagraph 재생 시 실행되지 않는다(<code>handoff-report/session_handoff_2026-08-13.md:150-153</code>). 이 측정은 정의상 운영점(cudagraph ON) 밖이다. Nano·Granite에는 리워크 계측이 이식되지 않았다(같은 문서 146행).</li>
<li><b>835571</b> (2026-07-07, ctx 3600, 1 boot, SM {{full, 44, 24, 16, 8}}, <code>results/r0c/knee_result_835571.txt</code>): 수정 전 계측(attn 버킷은 RadixAttention 코어만, mamba는 mixer 전체) — <code>reports/prefill_attn_ssm_length_review_2026-10-01.md:34</code>; full-SM arm이 항상 첫 순서이고 warm-up이 full-SM mamba 시간을 부풀린다(같은 문서 62행; <code>r0c_knee_isolate.sbatch:52</code>, 러닝평균 + <code>tail -1</code>: 60행); 반복 없음(arm당 1행). 판정: 무효 (<code>results/anchor_audit/VERDICT_claimE_first_arrow_2026-10-19.md:24</code>).</li>
<li><b>858811</b> (2026-07-18, ctx {{256, 1024, 4096, 16384}} 각 1 boot): 같은 비대칭 버킷과 러닝평균, full이 항상 첫 arm(<code>results/r0c/decode_knee_vs_ctx.sbatch:55</code> <code>SM_LIST=(full 44 24 16 8)</code>, 100행 <code>tail -1</code>); 감사가 물리 불변량 위반을 확인(sm44에서 mamba 1.98× 산포, full이 sm44보다 2.34× 느림, ctx256 65.655 vs 28.057) — <code>reports/layertype_dynamic_NEGATIVE_2026-08-04.md:72-80</code>. 결함 3건 정리와 재작업: <code>workspace/engine-port/env/dev_tree_edits.md:212-244</code>. 판정: 무효(같은 VERDICT).</li>
<li><b>873783</b> (2026-08-05, 수정 후 계측, arm 순서 셔플, warm-up 폐기; <code>results/r0c/decode_knee_vs_ctx_v2.sbatch:81</code>): GATE C 20/20 PASS, <b>GATE N FAIL</b> (bs=32 부분집합에서도 full 2.780 / sm44 1.971 / sm24 1.276). 기전은 host-paced 셀(<code>fwd_ms</code>가 device work 2–8× 달라도 44–46 ms 고정); 고-SM·단-ctx에서 per_mamba가 SM에 비단조. GATE N 자체도 ctx 16384에서 실현 bs 혼합(19/13)으로 잘못 지정됐다. n_indep = 1(rep1만), G6 NO-GO로 배치 2·3 미제출, 결과 미감사(UNAUDITED). 근거: <code>results/r0c/P5_GATES_BATCH1_2026-08-05.md:1-16, 71, 129, 183-191, 326, 330</code>; <code>handoff-report/session_handoff_2026-08-06.md:111-117</code>.</li>
<li><b>요약</b>: 수정 전 계측(버킷 비대칭, 러닝평균)은 835571·858811에 해당하고, full-SM 첫 순서도 이 둘(873783은 셔플로 해소), n=1은 세 건 모두(835571·858811은 ctx·arm당 1 boot, 873783은 n_indep=1), GATE N FAIL은 873783이다. 계측 게이트를 통과한 decode 층별 유효 측정은 0건이다.</li></ul>

<h2>hybrid 항이 D 선택을 바꾸는 셀 <span class=tag>예측, 인용 금지</span></h2>
<p class=meta>dstar_model 제안 r0(결정 61, 정본 아님) + analytic_terms. 실측 아님. 재생성: <code>reports/motivation_gap_2026-11-09/hybrid_term_cells.py</code>. 항 제거는 decode 상태 트래픽 b0(D) = b0/u(D). 입력 길이 대역은 실제 풀에서 잘랐다. LooGLE 단독은 TTFT 3 s로 전 D 불가라 퇴화 처리한다.</p>
{cells_section(res)}

<h2>vLLM 0.31.0에서 a31 조성 변형 로드 가능성 (코드 확인, 미실행)</h2>
<ul>
<li>a31은 <code>make_variant.py</code>가 Nano config의 <code>hybrid_override_pattern</code>만 바꾼다(27개 Mamba 층 전부 → <code>*</code>, 31 attention · 0 Mamba, MLP 25층). 이미지 <code>ghcr.io/asuan99/sglang-vllm-runtime:0.31.0-09df5f34</code>의 <code>/opt/vllm-venv</code> 코드를 읽어 확인했다. <b>실제 로드는 하지 않았다</b>(GPU 없음, 이미지 안에서 코드 실행도 안 함).</li>
<li><b>로드 경로에 Mamba 층 존재를 요구하는 검사가 없다</b>: 레지스트리 <code>NemotronHForCausalLM</code>→<code>nemotron_h.py</code>; 층 생성은 패턴 문자별(<code>nemotron_h.py:596-605</code>, <code>*</code> 허용); config 검증은 길이·문자 집합만(<code>transformers_utils/configs/nemotron_h.py</code>). <code>transformers</code>의 NemotronHConfig도 패턴을 층 목록으로 변환해 받는다(<code>configuration_nemotron_h.py:157-165</code>). KV spec은 attention 모듈에서 수집하므로 31개 attention spec만 생긴다(<code>gpu_model_runner.py:7435-7455</code>, MambaSpec 없음).</li>
<li><b>주의 1 — block size</b>: <code>NemotronHForCausalLM</code>은 클래스 속성으로 <code>IsHybrid</code>이며 config의 <code>mamba_*</code> 값은 남아 있어 <code>_align_hybrid_block_size</code>(<code>platforms/interface.py:835-1016</code>)가 그대로 돈다. 실제 Nano 부팅의 기록은 attention block size 1,312 토큰(Phase B vLLM <code>server.log</code>, interface.py:989)이다. 이 코드 경로대로면 a31도 같은 1,312 토큰이 될 것으로 유도된다(미실행). a31은 Mamba 상태가 없는데 attention 할당 단위만 1,312 토큰 × 31층이 되어 짧은 요청이 한 블록(약 {1312 * 126976 / 1e6:.0f} MB, k = 126,976 B/토큰은 make_variant 값)을 차지한다.</li>
<li><b>주의 2 — 이 값은 SGLang과 다른 축</b>: compo_probe의 SGLang 부팅에서 a31은 정상 부팅·배너 일치(<code>results/cap_char/compo_probe/RESULT_compo_probe_2026-10-21.md:31-36</code>). vLLM·a31 비교를 쓰려면 vLLM 쪽 block 단위 차이를 공시하거나 끌 방법(변형 config에서 mamba 필드 0 처리, 미검증)을 정해야 한다.</li>
<li>판단: 코드상 로드 가능, 단 실측 부팅(a31 dummy) 전까지는 '로드 확인'이 아니다.</li></ul>
<p class=meta>생성: reports/motivation_gap_2026-11-09/render_cpu_prep.py</p>
</html>"""
    Path(a.out).write_text(html_doc, encoding="utf-8")
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
