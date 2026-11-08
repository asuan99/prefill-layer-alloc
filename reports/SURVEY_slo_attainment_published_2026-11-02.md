# 문헌의 SLO attainment 값 조사 — ShareGPT·arXiv 요약류 (2026-11-02)

> ★결과 감사 r1(2026-11-08) GO-with-edits — 판정서 `reports/audit/2026-11-08_unaudited_r1/AUDIT_slo_survey_and_sq_maxitl_r1_2026-11-08.md`, 수정 전 인용 금지 목록 적용. 본문은 고치지 않았다.

> **지위**: 문헌 조사 메모. GPU 0 · 새 측정 0 · 성능 판정 0 · claim 등급 변경 0 · 정본 미수정. 미감사.
> **목적**: 우리 측정값(Nemotron-Nano-9B-v2, A100 1장)이 문헌 값과 어떻게 비교되는지 감 잡기. 순위 주장용이 아니다.
> **확인 범위**: 아래 값은 전부 1차 출처 PDF(arXiv 또는 로컬 `Papers/`, ORCA는 USENIX PDF)의 본문·표·그림을 직접 열어 옮겼다. 검색 스니펫은 근거로 쓰지 않았다. Bullet·MuxWise는 로컬 camera-ready 기준이다.
> **논문 수치는 저자 주장이다.** 재현하지 않았다.

## 값 표기 규칙

| 표기 | 뜻 |
|---|---|
| `[표]` | 논문 표의 숫자 |
| `[본문]` | 본문에 적힌 숫자 |
| `[그림 인쇄]` | 그림 안에 숫자로 인쇄된 값 |
| `[그림 판독, 근사]` | 축을 눈으로 읽은 값. 오차는 대략 눈금 간격의 절반으로 본다 |
| `[유도]` | 위 값을 이 메모가 나눈 값(총 rate ÷ GPU 수 등). 논문에 없다 |
| 미확인 | 원문에서 찾지 못했다 |
| 미보고 | 원문을 읽었고 그 양을 보고하지 않는다 |

## 0. 결론

1. **attainment 문턱 0.90은 문헌의 주류 형식이다.** DistServe, Bullet, SLOs-Serve, PolyServe, LoongServe가 90%에서 용량을 읽는다. 99%는 Niyama, MuxWise(토큰 백분위), DistServe 부록이 쓴다.
2. **가장 가까운 비교 대상은 Bullet(ShareGPT·arXiv, A100 1장, 8B)과 Sarathi-Serve(arXiv 입력 필터 16k, A100 1장, 7B)다.** 그래도 술어가 달라 같은 양이 아니다.
3. **숫자상 우리 값은 이 둘의 절반 안팎이다.** W1 λ\* 7.29는 Bullet 시스템 약 18.3의 약 0.4배, Bullet 그림의 vLLM 약 10의 약 0.7배다. W3 λ\* 0.3375는 Bullet 약 0.72, Sarathi strict 약 0.62의 약 0.5배다. **그러나 이 비로 "우리가 느리다/빠르다"를 말할 수 없다.** ITL 술어는 우리가 조이고(60 ms 대 150–175 ms), W1 TTFT는 우리가 풀어 두었으며, 모델도 hybrid로 다르다.
4. **우리 W1 7.29는 문헌 용량과 같은 종류의 값이 아닐 수 있다.** 120요청 셀의 버스트 값이고, 같은 rate의 600 s 지속 블록은 첫 60 s 창 뒤 attainment 0이다. 문헌은 대부분 런 길이를 적지 않아 같은 약점이 있는지 판단할 수 없다.
5. **우리 W3 0.3375는 통과 여유가 1요청이다.** 감사 r1 기준 문턱·floor를 조금 바꾸면 0.225–0.506으로 움직인다. 단일 값으로 비교하지 않는다.
6. **공정한 비교에 빠진 것이 하나 있다.** 같은 SLO로 채점한 우리 값이다. static arm은 SLO를 입력으로 받지 않으므로, 우리 원자료를 Bullet SLO로 재채점하는 CPU 분석이 가능해 보인다(§5). 이번에는 하지 않았다.

## 1. 우리 값 (비교 기준)

출처: `workspace/engine-port/results/policy_cap1r/RESULT_cap1r_2026-10-29.md`, `policy_cap2/RESULT_cap2_sustain_2026-10-29.md`, `policy_cap2/RECHECK_sustain_throughput_2026-11-02.md`, 같은 디렉터리 `AUDIT_*_r1_*.md`, `cap_char/PREREG_HD_HE0cell_rev0_2026-10-28.md`. 전부 기술값이며 감사 r1은 GO-with-caveats다.

| 항목 | W1 (ShareGPT) | W3 (arXiv 요약 16k) |
|---|---|---|
| 모델·기판 | Nemotron-Nano-9B-v2, A100-SXM 80GB 1장(108 SM), VESSL, SGLang 공개 PD-mux 루프 static 비sticky | 같음 |
| 입력 | V3 unfiltered, context 캡 4000. 프롬프트 p50 151 · p90 879 · p99 2,523 (전수 센서스, **Zamba2 토크나이저**, `cp_baseline/RESULT_SHAREGPT_CENSUS_2026-09-02.md`) | ≤16,384 필터, Nano 토크나이저 p50 6,484 |
| 출력 | ShareGPT 응답 길이, `ignore_eos` 고정 길이. 위 센서스 p50 131 · 평균 223 | 자연 출력, 평균 203, 상한 1024 |
| 요청 술어 | TTFT ≤ 3,000 ms ∧ 요청 내부 ITL p95 ≤ 60 ms | TTFT ≤ 2,814.84 ms(3 × floor 938.28 ms) ∧ 요청 내부 ITL p95 ≤ 60 ms |
| 셀 크기 | 120 req × seed 2 = 240 | 48 req × seed 2 = 96 |
| attainment 곡선(best D) | D44: 5.516→1.00 · 6.343→1.00 · **7.295→0.971** · 8.389→0.700 · 11.095→0.758 · 19.405→0.542 | D16: 0.1→0.990 · 0.225→0.990 · **0.3375→0.906** · 0.50625→0.542 · 0.759→0.083 |
| 0.90 문턱 λ\* | **7.29491 req/s**(D44·D64 동률, 다음 격자 8.389 실패) | **0.3375 req/s**(D16, 다음 격자 0.50625 실패) |
| 정상성 | 용량 셀 두 seed 모두 `UNSTABLE`(마지막 1/3 TTFT 중앙값이 처음 1/3의 16–31배) | 용량 셀 두 seed `STATIONARY` |
| 지속성 | 7.29 req/s 600 s 블록: 첫 60 s 창 attainment 0.28–0.36, 이후 전 창 0. 과부하 포화 완료율 5.06–5.34 req/s(goodput 아님) | 지속 블록 없음 |
| 취약성 | CAP1(다른 GPU, 다른 격자)에서는 7.447이 0.933·0.946 통과. 격자·Job 간 값이 움직인다 | 87/96, 문턱 86.4. seed별 40/48·47/48. floor 평균 사용 시 0.225, 문턱 0.875에서 0.50625(D24) |
| baseline 대비 | 없음. λ\*는 static D 중 최댓값이고 stock SGLang·vLLM 대비 값은 아직 미측정(Phase B 대기) | 같음 |

**인용 제한(감사 문안 승계)**: "Nano W1 용량은 7.3 req/s다", "W3 용량은 0.34 req/s", "지속 처리율 5.2 req/s"는 금지 문안이다. 이 메모의 7.29·0.3375는 **문헌 좌표에 놓아 보려는 참고값**이며 크기 주장이 아니다. 지속 완료율 "약 3.0"은 단위 오류로 판명돼 쓰지 않는다(RECHECK).

**99% 문턱**: W1 D44는 격자상 6.343까지 1.00, 7.295에서 0.971이다. W3는 셀당 96요청이라 95/96 = 0.9896으로 99%를 판정할 해상도가 없다.

**지속 용량**: 미측정이다. 포화 완료율 5.06–5.34로 보아 W1 지속 용량이 그 위에 있기는 어렵다고 추정하지만(유도), 포화 완료율이 부하에 무관하다는 보장은 없다.

## 2. 문헌 표 — ShareGPT·arXiv 요약류를 평가한 논문

"용량"은 attainment 문턱에서 지속 가능한 rate를 뜻한다. 다중 GPU는 `[유도]`로 GPU당 값을 병기한다.

| 논문 (연도·학회·URL) | 워크로드 | 모델 | GPU·수 | SLO 정의 | attainment 값 (어느 부하) | 문턱과 GPU당 용량 req/s | baseline 대비 | 값 출처 |
|---|---|---|---|---|---|---|---|---|
| **Bullet** (2026 ASPLOS, [arXiv 2504.19516](https://arxiv.org/abs/2504.19516), 로컬 camera-ready) | **ShareGPT** (입력 대략 1k 미만, Fig.14 판독) | Llama3.1-8B | A100-80GB ×1 | 요청이 norm. TTFT ≤ 3.0 ms **그리고** TPOT ≤ 150 ms. 평균 TPOT형, 요청 단위 결합. 정규화 분자는 원문에서 미확정 | 10 req/s에서 Bullet 약 1.0, 20 req/s 약 0.8, 25 req/s 약 0.6 `[그림 판독, 근사]` | **90%**: Bullet 약 18.3 · Nanoflow 약 17.2 · SGLang-2048 약 15.1 · SGLang-1024 약 12.2 · vLLM 약 10 `[그림 판독, 근사]`. ×1이라 그대로 GPU당 | SGLang-1024 대비 1.49× · Nanoflow 대비 SLO compliance 5.2%↑ `[본문]` | Fig.15d 상단, Table 4, §4.2.1 |
| 〃 | **arXiv-Summary** (입력 약 10k–20k, Fig.14 판독, 출력 대략 수백) | 〃 | 〃 | norm. TTFT ≤ 1.5 ms, TPOT ≤ 175 ms | 0.5·0.6·0.7에서 Bullet 약 1.0, 0.8에서 약 0.47, 0.9에서 약 0.3 `[그림 판독, 근사]` | **90%**: Bullet 약 0.72 · Nanoflow 약 0.72(겹침) · vLLM 약 0.64 · SGLang 약 0.52 `[그림 판독, 근사]` | 전 워크로드 평균 throughput 1.09×, 최대 1.20× `[본문]`(attainment 값 아님) | Fig.15d 하단, Table 4 |
| **MuxWise** (2026 ASPLOS, [arXiv 2504.14489](https://arxiv.org/abs/2504.14489), 로컬 camera-ready) | **ShareGPT** (입력 4/226/1024) | Llama-70B | 서버 명시 없음. 주 시험대 A100 ×8, TP8 | **P99 TBT ≤ 100 ms**(토큰 단위 백분위). TTFT는 SLO 아님 | attainment 곡선 대신 P99 곡선. 19 req/s에서 P99 TTFT 약 2.2 s `[그림 판독, 근사]` | P99 기준 19 req/s(baseline 10·11·13·2) `[그림 판독, 근사]`, 본문 비와 일치. GPU당 약 2.4 `[유도]` | Chunked 1.9× · NanoFlow 1.73× · LoongServe 9.5× · SGLang-PD 1.46× `[본문]` | Fig.17, §4.3 |
| 〃 | **ShareGPT** | Llama-8B | A100 ×1 | TBT 50 ms 목표. 단일 GPU 절대값 미보고 | 미보고 | 절대 용량 미보고 | Chunked 대비 goodput 1.2× `[본문]` | §4.3.1 |
| 〃 | LooGLE (arXiv류 아님, 입력 3380/30k/81k) | Llama-70B | 위와 같음 | P99 TBT ≤ 100 ms | 미보고(곡선만) | 약 0.24 `[그림 판독, 근사]`. GPU당 약 0.03 `[유도]` | 1.71× · 2× · 1.33× · 2× `[본문]` | Fig.17 |
| **Sarathi-Serve** (2024 OSDI, [arXiv 2403.02310](https://arxiv.org/abs/2403.02310)) | **openchat_sharegpt4** (프롬프트 중앙 1730, p90 5696, 출력 중앙 415, 합계 ≤8192) | Mistral-7B | A100 ×1 | **P99 TBT**(토큰 단위) strict 100 ms / relaxed 500 ms `[표]`. TTFT SLO 없음. 지속 조건: 중앙 스케줄링 지연 ≤ 2 s | attainment 미보고 | 문턱 없음(P99 형식). strict: Sarathi 약 2.0 · vLLM 약 0.77 · Orca 약 0.72. relaxed: 약 2.3 · 1.55 · 1.05 `[그림 판독, 근사]` | Orca 대비 2.78×(S)·2.15×(R) `[그림 인쇄]` | Fig.10a, Table 1–3 |
| 〃 | **arxiv_summarization** (프롬프트 중앙 7059, p90 12985, 출력 중앙 208, 합계 ≤16384) | Mistral-7B | A100 ×1 | 위와 같음 | attainment 미보고 | strict: Sarathi 약 0.62 · vLLM 약 0.44 · Orca 약 0.34. relaxed: 약 0.77 · 0.47 · 0.39 `[그림 판독, 근사]` | Orca 대비 1.82×(S)·1.97×(R) `[그림 인쇄]` | Fig.10b |
| 〃 | 같은 두 입력 | Yi-34B | A100 ×2(TP2) | strict 200 ms / relaxed 1 s `[표]` | 미보고 | sharegpt4 strict 약 1.07, arxiv strict 약 0.27(총 rate) `[그림 판독, 근사]` | Orca 대비 4.00×·1.69× `[그림 인쇄]`, vLLM 대비 최대 3.7× `[본문]` | Fig.10 |
| **DistServe** (2024 OSDI, [arXiv 2401.09670](https://arxiv.org/abs/2401.09670)) | **ShareGPT** (입력 평균 755.5, 출력 200.3) | OPT-13B | A100-80GB, 4노드 32장 중 배치 | TTFT 0.25 s **그리고** TPOT 0.1 s `[표]`. 요청 단위 평균 TPOT | 1 req/s/GPU: DistServe 약 100, vLLM·DeepSpeed 약 85 %. 3 req/s/GPU: 약 75 · 55 · 60 % `[그림 판독, 근사]` | **90%**: DistServe 약 2.3 · vLLM 약 0.5 · DeepSpeed 약 0.3 per GPU `[그림 판독, 근사]` | vLLM 대비 2.0–4.6× · DeepSpeed 대비 1.6–7.4× (3모델 범위) `[본문]` | Fig.8a, Table 1, §6.2 |
| 〃 | **ShareGPT** | OPT-66B | 〃 | 2.5 s, 0.15 s `[표]` | 그림 곡선 | **90%**: 약 0.95 · vLLM 약 0.35 · DeepSpeed 약 0.6 `[그림 판독, 근사]` | 위 범위에 포함 | Fig.8b |
| 〃 | **ShareGPT** | OPT-175B | 〃 | 4.0 s, 0.2 s `[표]` | 그림 곡선 | **90%**: 약 0.2 · vLLM 약 0.1 `[그림 판독, 근사]`. **99%** 결과는 부록 Fig.13–14(값 미판독) | 99%: vLLM 대비 3×–8× `[본문]` | Fig.8c, App.C |
| 〃 | LongBench 요약 (입력 평균 1738, **2048 절단**) | OPT-66B | 〃 | 15 s, 0.15 s `[표]` | 그림 곡선 | **90%**: 약 0.53 · DeepSpeed 약 0.3 · vLLM 약 0.12 `[그림 판독, 근사]` | vLLM 4.3× · DeepSpeed 1.8× `[본문]` | Fig.9b |
| **SLOs-Serve** ([arXiv 2504.08784](https://arxiv.org/abs/2504.08784) v1, 2025, 게재처 미확인) | ChatBot: **ShareGPT** 입력 평균 763, p99 1591 + Azure 도착 | OPT-7B | A100 40GB ×4, rate는 GPU 수로 정규화 | "Max TTFT slowdown" 무부하의 3×/5×, "Max TPOT" 50/100 ms `[표]`. ChatBot은 Table 1에 loose/loose. **요청 단위 결합** | 위반율 곡선 | **90%(위반 <10%)**: 시스템 5.9, baseline 표기 1.7·2.5 `[그림 인쇄]` | 전체 geo-mean 2.2× `[본문]` | Fig.9, Table 1–4 |
| 〃 | ChatBot | OPT-13B | 〃 | 〃 | 〃 | 시스템 2.2, baseline .88·1.2·1.6 `[그림 인쇄]` | ⚠ §4 | Fig.9 |
| 〃 | Summarizer: **Arxiv Summary** 입력 평균 **1333**, p99 1946 | OPT-7B | 〃 | Table 1 tight prefill / loose decode | 〃 | 시스템 5.8, baseline 2.6 `[그림 인쇄]` | ⚠ §4 | Fig.9 |
| 〃 | Summarizer | OPT-13B | 〃 | 〃 | 〃 | 시스템 1.9, baseline .58·.95·1.6 `[그림 인쇄]` | ⚠ §4 | Fig.9 |
| **PolyServe** ([arXiv 2507.17769](https://arxiv.org/abs/2507.17769) v1, 2025, 게재처 미확인) | **sharegpt** (입력 p50 36, p90 818, p99 3421. 출력 p50 280) | Llama3.1-8B | **시뮬레이터**, H200 프로파일, 20 인스턴스 | 데드라인형(DSLO): i번째 토큰 ≤ TTFT + i×TPOT. TTFT {300, 500, 1000} ms, TPOT {20, 30, 50, 100} ms 혼합 | 곡선 | **90%**: PD-PolyServe 약 740 · CO-PolyServe 약 640 · CO-Chunk 약 625 총 rate `[그림 판독, 근사]`. 인스턴스당 약 37·32·31 `[유도]`. **실측 아님** | 전 trace 평균 1.23×·1.18× `[본문]` | Fig.6, Table 1, §5.2 |
| **Niyama** ([arXiv 2503.22562](https://arxiv.org/abs/2503.22562) v1, 2025, 게재처 미확인) | **ShareGPT** (프롬프트 p50 1730, p90 5696, 출력 p50 415). 요청을 3등분해 QoS 배정 | Llama3-8B A100×1 / Qwen-7B A100×2. Fig.7a 모델 미명시 | A100 | Q1: TTFT 6 s, TBT 50 ms, Q2·Q3: TTLT 600·1800 s `[표]`. 데드라인형 | 미보고 | **99%(위반 ≤1%)**: 50 QPS에 Niyama 66장, SOTA 76장 `[그림 인쇄]` → GPU당 약 0.76·0.66 `[유도]`, QoS 혼합 | ShareGPT에서 GPU 76→66 `[그림 인쇄]`. 전 데이터셋 13–32% `[본문]` | Fig.7a, Table 1–2, §4.1 |
| **LoongServe** (2024 SOSP, [arXiv 2404.09526](https://arxiv.org/abs/2404.09526)) | **ShareGPT** (시퀀스 4–2.3K) | LWM-1M-Text (Llama-2-7B 구조) | A800 80GB ×8 | 무경합 추론 지연의 25×, normalized latency `[본문]`, P90 goodput | 약 26 req/s까지 약 100 % `[그림 판독, 근사]` | **90%**: 약 26.5, elastic scale-up 제외 약 9.3 총 rate `[그림 판독, 근사]`. GPU당 약 3.3·1.2 `[유도]` | 위 ablation 2.87× `[본문]` | Fig.13a, §7.4 |
| **Mooncake** (2025 FAST, [arXiv 2407.00079](https://arxiv.org/abs/2407.00079) v5) | **ArXiv Summarization** (입력 평균 8088, 출력 229, 캐시 ~0 %) | LLaMA2-70B 구조 dummy | A800 ×8 노드 4개 | **P90** TTFT 10×, **P90** TBT 5× 무부하 대비, 값은 1.0으로 정규화해 절대값 미공개 | attainment 미보고(P90 곡선) | P90 TTFT가 1.0에 닿는 점: 3P+1D 약 3.0 `[그림 판독, 근사]`, GPU당 약 0.09 `[유도]` | vLLM-[4M] 대비 throughput +20 % `[본문]` | Fig.11, Table 2, §8.1 |
| **QLM** (2024 SoCC, [arXiv 2407.00047](https://arxiv.org/abs/2407.00047)) | **ShareGPT** 3,500요청 | Vicuna-13B(Fig.10) | A100 50장 | **p99 TTFT만**: Interactive 20 s, Batch-1 1 min, Batch-2 1 h. ITL SLO 없음 | SLOs Met, 0.5K: QLM 약 95 · EDF 약 75 · vLLM 약 10 %. 2K: 약 53·45·10 %. 8K 이상 모두 약 10 % `[그림 판독, 근사]` | 문턱 미사용. 90 % 도달점은 0.5K–1K 사이(약 87 %가 1K). 이 rate를 GPU로 나누면 약 10 `[유도]`지만 배치 대기를 허용해 **용량으로 읽으면 안 된다** | SLO attainment 40–90 % 향상 `[본문]` | Fig.9–10, §8.1 |
| **NanoFlow** (후보 외 추가, 2025, [arXiv 2408.12757](https://arxiv.org/abs/2408.12757)) | **ShareGPT** | LLaMA-2-70B | A100 ×8 | **평균** normalized latency ≤ 200 ms/token (전 요청 평균) | attainment 미보고 | 문턱 없음. 16.3, TensorRT-LLM 10.5 `[그림 인쇄]`, GPU당 약 2.0·1.3 `[유도]` | 전 데이터셋 예: LMSys 1.64× `[본문]` | Fig.8c, §6 |

## 3. 후보 중 attainment·해당 워크로드가 없는 논문

| 논문 | 무엇을 확인했나 | 판정 |
|---|---|---|
| **vLLM** (2023 SOSP, [arXiv 2309.06180](https://arxiv.org/abs/2309.06180)) | ShareGPT, OPT-13B 1 GPU · 66B 4 GPU · 175B 8 GPU. normalized latency 곡선. Orca(Oracle) 대비 1.7–2.7×, Orca(Max) 대비 2.7–8× `[본문]`(Fig.12) | SLO·attainment **미보고**. attain 언급 0건 |
| **ORCA** (2022 OSDI, [USENIX](https://www.usenix.org/conference/osdi22/presentation/yu)) | ShareGPT·SLO·attain 언급 0건. 지연-처리량 곡선 | **해당 없음** |
| **Llumnix** (2024 OSDI로 알고 있으나 PDF 표기 미확인, [arXiv 2406.03243](https://arxiv.org/abs/2406.03243)) | ShareGPT(GPT4) 사용, A10, P99 first-token 최대 15× 개선 `[본문]` | SLO 임계·attainment **미보고** |
| **Splitwise** (2024 ISCA, [arXiv 2311.18677](https://arxiv.org/abs/2311.18677)) | Azure coding·conversation trace. P50·P90·P99 × TTFT·TBT·E2E 9개 전부 충족 조건(Table VI) | ShareGPT·arXiv 없음. 전체 충족형이라 attainment 곡선 없음 |
| **DynamoLLM** (2025 HPCA, [arXiv 2408.00741](https://arxiv.org/abs/2408.00741)) | Azure trace, 에너지 목적, P99, 단독 5×(Table IV) | ShareGPT·arXiv·attainment **없음** |
| **DuetServe** (PDF 각주상 ICML 2026, [arXiv 2511.04791](https://arxiv.org/abs/2511.04791) v2) | Azure Code·Conv·Mooncake, Qwen3-8B, H100. 평균 TBT·throughput 곡선. 1.1–1.3× `[본문]` | ShareGPT·arXiv·attainment **없음** |
| **TetriServe** ([arXiv 2510.01565](https://arxiv.org/abs/2510.01565)) | **DiT 이미지 생성** 서빙. LLM이 아니다 | **범위 밖** |

표 미포함: LayerKV(ShareGPT 언급 있음, 평가 미독), GenZ, Medha, Nexus.

## 4. 원문을 읽다 발견한 주의점

- **SLOs-Serve 본문과 그림 불일치 가능성**: §6.1 본문은 ChatBot·Summarizer 개선을 1.41–1.70×, 1.14–1.17×라 적는데, Fig.9의 OPT-7B는 5.9 대 2.5, 5.8 대 2.6으로 2배가 넘는다. 범례와 인쇄값의 대응을 잘못 읽었을 가능성도 있어 판단하지 않는다. 이 논문의 개선 비는 인용 전 재확인이 필요하다.
- **"ShareGPT"는 한 워크로드가 아니다.** 같은 이름의 입력 중앙값·평균이 36(PolyServe p50)부터 1730(Sarathi·Niyama p50)까지 50배 가까이 벌어진다. 우리는 p50 151이다.
- **"arXiv 요약"도 하나가 아니다.** 입력이 평균 1333(SLOs-Serve), 중앙 7059(Sarathi), 평균 8088(Mooncake), 약 10k–20k(Bullet 판독), 우리 p50 6484다. 앞의 두 번째 부류부터가 우리와 견줄 만하다.
- **MuxWise Fig.15의 attainment 곡선은 ShareGPT가 아니라 Tool&Agent다.** ShareGPT 값은 Fig.17의 P99 곡선에서만 읽을 수 있다.
- **선행 메모 미해결 항목 갱신**: DistServe 90 %는 이번에 원문 확인했다(§6.1 "particularly interested in an SLO attainment of 90%"). 따라서 `slo_definition_rationale_2026-10-04.md`의 "원문 미확인 DistServe 90 %"는 해소 가능하다(정본 수정은 하지 않았다). Bullet 정규화 분자는 여전히 미확정이다(§4.1은 "normalized input/generation latency"라고만 적는다).
- **Sarathi Fig.10·11의 배수 라벨은 Orca 대비 값이다**(인쇄 막대에서 역산해 일치). 초록의 "2.6×, 3.7×"는 vLLM 대비다.

## 5. 비교 가능·불가능

| 비교 | 가능 여부 | 이유 |
|---|---|---|
| 우리 W1 대 Bullet ShareGPT | **크기 자릿수까지만** | 같은 A100 1장·같은 90%·유사 입력(V3 계열로 보임, 동일 파일 여부 미확인). 그러나 술어(ITL p95 60 ms 대 평균 TPOT 150 ms, 절대 TTFT 3 s 대 정규화 3 ms), hybrid 대 attention, 동적 대 static, 런 길이가 다르다 |
| 우리 W3 대 Sarathi arxiv_summarization (Mistral-7B) | **크기 자릿수까지만** | 같은 A100 1장, 같은 16k 필터. 입력 중앙값 6484 대 7059로 가깝다. Sarathi에는 TTFT 임계가 사실상 없고(중앙 스케줄링 지연 ≤2 s) 토큰 합산 P99 TBT 100 ms를 쓴다 |
| 우리 W3 대 Bullet arXiv-Sum | **약함** | 입력 분포가 약 2배 길다(판독). TTFT 임계가 ms/token으로 달라 긴 입력일수록 크게 풀린다 |
| GPU당 값으로 다중 GPU 논문과 비교 | **불가** | 모델 크기(OPT-13B 대 70B 등), 병렬화(TP·SP·PD 분리)가 다르다. `[유도]` 값은 그 논문 안의 비교용이다 |
| 다른 논문의 개선 배수 대 우리 값 | **불가** | 분모 baseline이 논문마다 다르고 우리 arm에는 대응 baseline이 없다. 같은 Bullet 안에서도 SGLang 비 1.49×는 Nanoflow 비 약 1.06×와 다르다 |
| 저부하 attainment 절대값 | **의미 약함** | 거의 모든 논문이 저부하에서 약 100 %다. 정보는 무너지는 지점에 있다 |
| 용량 위 attainment 절대값(우리 0.52–0.78 고원 포함) | **불가** | 짧은 셀에서는 큐가 끝까지 자라기 전에 측정이 끝난다. 우리 지속 블록이 0인 것이 그 증거다. 문헌 곡선의 런 길이는 Bullet·DistServe·MuxWise·SLOs-Serve에서 **미확인**이다 |
| 시뮬레이션(PolyServe) 대 실측 | **불가** | 게이트 1 |
| 90 % 대 99 % 값 | **불가** | 곡선이 가팔라 문턱이 값을 바꾼다 |
| 같은 SLO로 우리 재채점 후 Bullet 값과 비교 | **가장 나은 비교지만 미수행** | static arm은 SLO를 입력으로 받지 않아 재채점이 게이트 8에 걸리지 않는다는 기존 판단이 있다. 단 W1 7.29 지속성, 입력 길이 차이는 남는다. 제안일 뿐이며 이번 메모에는 수치가 없다 |

### SLO 조임 방향 요약 (A100 1장 후보)

| 항목 | 우리 | Bullet | Sarathi-Serve(7B) |
|---|---|---|---|
| ShareGPT TTFT | 3 s 절대 | 3 ms × 입력 토큰(분자 미확정. 151토큰이면 약 0.45 s `[유도]`) → **Bullet이 조임** | 임계 없음 → **우리가 조임** |
| arXiv TTFT | 2.8 s 절대(p50 입력 환산 약 0.43 ms/토큰 `[유도]`) | 1.5 ms × 입력 토큰(13k면 약 20 s `[유도]`) → **우리가 조임** | 임계 없음 → **우리가 조임** |
| decode 지연 | 요청 내부 ITL p95 ≤ 60 ms | 평균 TPOT ≤ 150·175 ms → **우리가 대체로 조임**(단 아래 단락) | 토큰 합산 P99 ≤ 100 ms → **방향 불명** |
| 입력 비교 | V3, p50 151 | V3 추정, 입력 대부분 1k 미만 | sharegpt4, p50 1730 → Sarathi 입력이 김 |

위 표로 볼 때 W3의 낮은 λ\*는 2.8 s TTFT 때문일 가능성이 있다. **이는 가설이고 측정이 아니다.** floor 938 ms의 3배인데 입력 길이에 따라 여유가 1.6×–14×로 갈린다는 감사 지적과 같은 방향이다.

## 6. 우리 술어와 TPOT 평균의 차이

TPOT는 요청 하나의 출력 구간 평균 `(E2E − TTFT) / (출력 − 1)`이고, 우리 술어는 그 요청의 토큰 간격 분포에서 95번째 백분위 값이다. 둘은 서로 다른 것을 보고 **어느 쪽이 항상 더 엄격하지 않다.** 평균은 간격 **총합**에 민감해 큰 정지 몇 번으로 크게 오르지만 나머지 간격이 그 값 아래인지는 묻지 않는다. p95는 상위 5 % 간격을 크기와 무관하게 버리므로 출력 64토큰(간격 63개)에서는 정지 약 3회를 보지 못하고, 대신 간격 전반의 흔들림에는 평균보다 민감하다. 저장소 기술값(M3 J1, Job 1개·seed 2, 순위 서술 금지, `stock_baseline/RESULT_stock_baseline_phaseA_2026-11-01.md` §2)에 양방향 사례가 모두 있다. fused·shape B는 TPOT p50 약 41 ms인데 요청 내부 ITL p95 중앙은 약 16 ms이고, agnostic·shape A는 TPOT p50 약 21 ms인데 ITL p95 중앙은 35 ms다. 따라서 "우리 60 ms는 문헌 150 ms보다 2.5배 엄격하다"는 단순 환산은 틀릴 수 있다. 또한 Sarathi·MuxWise의 P99 TBT는 **모든 토큰을 합친** 백분위라 긴 출력 요청에 가중되며 요청별 지시함수가 아니다. 우리 attainment 0.90은 "요청의 90 %가 각자의 p95를 지킨다"는 뜻이고 P90 TPOT와도, P99 TBT와도 다르다.

## 7. 확인하지 못한 것

- 대부분 논문의 런 길이·요청 수(Bullet, DistServe, MuxWise, SLOs-Serve). 예외: Niyama 4 h(Fig.7b), NanoFlow 5분, QLM 3,500요청, PolyServe 시뮬 300,000.
- Bullet ShareGPT와 우리 파일의 동일 여부, Bullet `ignore_eos` 여부, TTFT 정규화 분자.
- MuxWise Fig.17의 서버(문맥상 A100 ×8, 명시 없음), 단일 GPU 절대 용량.
- DistServe 99 % 부록 값, Yi-34B 이상 Sarathi 세부, QLM 다른 모델 곡선.
- SLOs-Serve·Niyama·PolyServe 게재처, Llumnix PDF 내 학회 표기. Mooncake FAST'25·LoongServe SOSP'24·Splitwise ISCA'24·DynamoLLM HPCA'25는 선행 메모 승계이며 이번에 학회 페이지를 재확인하지 않았다.
- W1 입력 분위수는 Zamba2 토크나이저 센서스다. Nano 토크나이저 값은 이 메모에서 재지 않았다.
