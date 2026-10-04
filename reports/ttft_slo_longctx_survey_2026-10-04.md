# TTFT/TPOT SLO 설정 관행 조사 — long-context 중심 (2026-10-04)

> **지위**: venue-strategist가 읽기와 웹 조사로 만든 조사 보고를 메인 세션이 옮겼다. GPU 0. 성능 판정 0. 평가 방법론 문서이며 성능 주장이 아니다.
> **선행 조사**: `workspace/engine-port/results/cp_baseline/RELATEDWORK_LONGCTX_SLO_2026-09-07.md`, `reports/serving_slo_survey.md`, `reports/longcontext_trace_plan.md` §4.3. 이번 조사는 이들을 보강한다. 1차 출처 재확인과 Bullet·DynamoLLM·SLOs-Serve·MLPerf·DuetServe·Nexus·Medha 추가가 새로 한 일이다.

## 0. 결론

**long-context TTFT SLO에 합의된 단일 기준은 없다.**
- DistServe는 원문에서 "이 응용들에 쓸 수 있는 SLO 설정이 존재하지 않아 경험적으로 정했다"고 쓴다.
- MLPerf 규칙 문서에도 제약값의 근거 서술이 없다.

문헌이 반복해서 쓰는 **방법**은 네 가지다.
- (a) 무부하 단일요청 지연 × 배수
- (b) 입력 토큰당 정규화 TTFT
- (c) 고정 절대 제약(MLPerf)
- (d) 응용·길이 버킷별 절대표

long-context 계열에서는 (a)가 사실상 주류다. 단 배수 값 자체를 정당화한 논문은 찾지 못했다.

## 1. 표

"이번 확인" = 이번 조사에서 원문을 직접 봤다. "선행 메모" = 2026-09-07 조사의 값을 승계했고 재확인하지 않았다.

| 논문/벤치 | 모델·GPU | 입력 길이 | TTFT SLO | TPOT/TBT SLO | 방식 | 출처 |
|---|---|---|---|---|---|---|
| **Bullet** (ASPLOS'26) | Llama3.1-8B, **A100 1장** | ShareGPT / Azure-Code / **arXiv-Summary**(약 20k까지) | 정규화 TTFT 3.0 / 1.5 / **1.5 ms/tok** | 150 / 200 / **175 ms** | (b) LoongServe·NanoFlow 정규화 승계, P90 attainment | `Papers/Bullet.pdf` Table 4, §4.1 (이번 확인) |
| **MuxWise** (ASPLOS'26) | Llama-8B/70B, Qwen3-235B, H100/H200 | ShareGPT, **LooGLE 3.4k/30k/81k** | **없음**(특성화에만 400 ms, 평가는 TTFT/token 보조) | **50 ms(8B) / 100 ms(70B)**, P99 | (c) TBT만 | `Papers/Muxwise.pdf` §2.2, §4 (이번 확인) |
| **MLPerf Inference** | 제출자 | 8B 요약 ≤2048 / 70B ≤1024 / **405B ≤20000(평균 9.4k)** / R1 / GPT-OSS | server/interactive: 8B 2000/500 ms · 70B 2000/450 · **405B 6000/4500** · R1 2000/1500 · GPT-OSS 3000/2000 | 100/30 · 200/40 · **175/80** · 80/15 · 80/20 ms | (c) 고정 절대, P99. 근거는 정성 문장뿐 | inference_rules.adoc, MLCommons 블로그 (이번 확인) |
| **DynamoLLM** (HPCA'25) | Llama2-70B 등 | 길이 버킷 | **<256: 250 ms, <1024: 400 ms, ≤8192: 2000 ms** | 100 ms | (a)+(d): 단독 지연의 5×, P99 | arXiv 2408.00741 Table IV (이번 확인) |
| **SLOs-Serve** | OPT 등, A100 40GB ×4 | 요약 평균 1.3k | 무부하 대비 **tight 3× / loose 5×** | 50 / 100 ms | (a), 용량 = 위반 <10%인 최대 부하 | arXiv 2504.08784 Table 3–4 (이번 확인) |
| **Splitwise** (ISCA'24) | DGX-A100 | coding 1.5k / conv 1.0k | **P50 2× / P90 3× / P99 6×** | 1.25× / 1.5× / 5× | (a) 무경합 기준 | arXiv 2311.18677 Table VI (이번 확인) |
| **Mooncake** (FAST'25) | 70B급, A800 ×8 | 실 trace 평균 7.6k, 시뮬 16k–128k | **P90 10×**, 실험은 절대 30 s | P90 5× / 0.1 s | (a)+(c) | arXiv 2407.00079v3 (이번 확인) |
| **LoongServe** (SOSP'24) | 7B급, A800 ×8 | 2.7k–497k | 무경합의 **25×**, normalized input latency | normalized output | (a)+(b) | arXiv 2404.09526v2 (이번 확인) |
| **DistServe** (OSDI'24) | OPT 13–175B | 요약 LongBench(**2048로 절단**) | chat 0.25/2.5/4.0 s, code 0.125 s, **요약 15 s** | 0.1–0.2 s | (d) 경험값 × SLO scale | arXiv 2401.09670v3 Table 1 (이번 확인) |
| **Etalon(Metron)** | — | — | D_p = `predict_prefill_time(P) + slack`, 단독 프로파일 곡선 fit | 응용별 | (a)의 길이 함수판. 단순 길이 정규화 비판 | arXiv 2407.07000 §3.1 (이번 확인) |
| Sarathi-Serve (OSDI'24) | Mistral-7B 등 | 4k | 없음 | P99 TBT = iteration의 5×/25× | (a) TBT만 | 선행 메모 |
| DuetServe / Nexus / Medha | — | 평균 12k / 3.8–5.9k / 128k–1M | 수치 없음(상대 slack·deadline) | DuetServe "typical 100 ms TBT" | 상대 | fetch 요약 기준(본문 전수 아님) |
| TetriInfer / Infinite-LLM / Llumnix | — | — | 임계 없음 | — | 없음 | 이번 확인(부재 확인 수준) |
| vLLM `bench serve --goodput` | — | — | 사용자 지정, 기본값 없음 | 동일 | 도구가 정하지 않음 | docs.vllm.ai (이번 확인) |
| GenZ / Niyama / LayerKV | — | 요약 15k 등 | 2 s / 6 s / 3 s | 20 / 50 / 200 ms | (d)/(c)/(c) | 선행 메모(미재확인) |

## 2. 유형별 정리

- **(a) 무부하 배수**: long-context 계열의 주류다.
  - 분모가 그 요청 자신의 단독 지연이므로 길이 정규화가 자동으로 된다.
  - 배수는 3×(tight), 5×(P99), 6×(Splitwise P99), 10×(Mooncake P90), 25×(LoongServe)에 분포한다. 정당화 서술은 없다.
- **(b) 토큰당 정규화**: prefill이 길이에 대략 선형이라고 암묵적으로 가정한다. Etalon은 이 방식이 대기 지연까지 정규화해 짧은 요청에 불리하다고 비판한다.
- **(c) MLPerf 고정**: 인용력이 가장 높다. 값은 길이가 아니라 모델 규모와 시나리오로 늘린다.
- **(d) 버킷표**: DynamoLLM 표는 (a) 5×를 버킷 단위로 굳힌 것이다. GenZ는 길이가 아니라 응용이 값을 정한다.
- **배제형**: Sarathi, Llumnix, 그리고 가장 가까운 선행인 MuxWise는 TTFT를 SLO에서 빼고 TBT만 묶는다. top-tier에서도 허용되는 선택이다.
- **attainment 기준**: P99(MLPerf, DynamoLLM), P90/P99 분리(Mooncake, Splitwise), 위반 <10%(SLOs-Serve), P90 attainment(Bullet).

## 3. shape B(8192/64, Nemotron-Nano-9B-v2, VESSL A100 1장)에 대한 함의

### 3-1. 문헌 좌표 환산 (참조 좌표일 뿐, 이식 근거 아님)

| 후보 | 8192 환산 | 근거 | 우리 실측(0.45 req/s: p50 1.6 s, p95 6.9 s) 대비 |
|---|---|---|---|
| 약 2 s | 2 s | DynamoLLM ≤8192 버킷, MLPerf server 2 s | p50 바로 위 → 저부하에서도 cliff |
| 약 6 s | 6 s | MLPerf 405B P99 | p95 바로 아래 → cliff |
| 약 12 s | 1.5 ms × 8192 ≈ **12.3 s** | **Bullet arXiv-Sum**(같은 A100 1장, 8B급, 장문 요약) | 저부하는 거의 전부 통과 → λ\* 근처에서만 판별 |
| 15–30 s | — | DistServe 요약(입력 ≤2k), Mooncake 실배포 | 판별력 거의 없음 |

### 3-2. 무부하 배수 방식과 우리 게이트의 관계

1. **분모는 정책 비의존 기준 config에서 잰다.**
   - 조건은 단일요청·무경합·비분할, cudagraph ON이다. split arm에서 재면 교락이 생긴다.
   - 저장소에 shape B 무부하 TTFT 실측은 **없다**. λ\*(B) 역산으로 약 1.4 s 이하로 **추정**되지만 인용하면 안 된다.
   - 별도 calibration(n≥4)으로 재고 사전등록에 동결한다(`longcontext_trace_plan.md` §4.3 "prefill_floor는 반드시 측정").
2. **재스코어 금지(게이트 8)**
   - static arm: SLO를 입력으로 받지 않는다. **사전등록한 배수 사다리**로 같은 telemetry를 채점하는 것은 DistServe SLO-scale 관행과 같다.
   - SLO를 입력으로 받는 컨트롤러: SLO 단계마다 재튜닝하고 따로 실행해야 한다.
   - 이 구분은 조사자의 해석이므로 **claims-auditor 확인이 필요하다**.
3. **Metric cliff(게이트 6)는 배수 방식으로도 사라지지 않는다.**
   - 단일 rate goodput 대신 **사다리 단계별 attainment-vs-rate 곡선과 "attainment ≥90%를 만족하는 최대 rate"**(bootstrap CI)를 보고한다. SLOs-Serve·DistServe가 쓰는 산출물 형태다.

### 3-3. 16k–64k 확장

- 한 런에 한 길이만 쓰면(현 결정: 혼합 trace 금지) 길이별 floor(L)에 같은 k를 곱하면 된다. DynamoLLM과 같은 버킷형 (a)다.
- 한 런 안에서 길이가 퍼지면 요청별 `k × floor(L_i)`가 필요하다. floor(L)은 실측점에 곡선을 맞춰 만들고(Etalon), 선형 폐형식은 가정하지 않는다.
- hybrid prefill의 길이 선형성은 미실측이다.
- 64k는 context-length를 바꿔야 한다.

### 3-4. ITL

- 50 ms p95는 MuxWise 8B TBT와 같고, MLPerf 8B server(100 ms)와 interactive(30 ms) 사이다.
- shape B 저부하 ITL p95는 22–24 ms다. 따라서 TTFT가 구속 술어라고 **예상**한다(측정 아님).

### 3-5. 권고와 대안 (조사자)

- **권고 — 무부하 배수 사다리**
  - 비분할 기준 config에서 shape B floor를 n≥4로 실측한다.
  - TTFT SLO = {3×, 5×, 10×} × floor로 사전등록한다.
  - ITL은 50 ms p95를 유지한다.
  - 산출물은 단계별 attainment-vs-rate 곡선과 90% 용량이다.
  - 절대 환산값을 Bullet 12.3 s, MLPerf 6 s 대비 위치와 함께 적는다.
  - 리스크: (i) floor가 config에 의존한다. (ii) 배수 자체는 문헌도 근거를 대지 않으므로 "업계 기준"이 아니라 "선행 관행 범위"라고 써야 한다. (iii) calibration GPU 시간이 든다.
- **대안 1 — Bullet 1.5 ms/tok(8k에서 약 12.3 s)**
  - 장점: 가장 가까운 선행과 직접 비교된다.
  - 리스크: 9B hybrid에서는 관대해 판별력이 부족하다. Etalon 비판이 적용된다. Bullet의 분자가 TTFT인지 prefill 시간인지 원문으로 확정하지 못했다.
- **대안 2 — MLPerf 고정 P99 6 s**
  - 장점: 인용력이 높다.
  - 리스크: 405B·다중 GPU 조건의 값이다. 우리 p95 바로 아래라 cliff 위험이 가장 크다.

## 4. 확인하지 못한 것

- **추정**: floor 약 1.4 s 이하, hybrid prefill 선형성, ITL 비구속.
- **원문 미확인**: Bullet 정규화의 분자, DistServe 90% 목표, DynamoLLM GPU 구성, DuetServe·Nexus·Medha 본문 전수, SGLang PD-mux SLO 기본값, Sarathi·GenZ·Niyama·LayerKV 수치.
- MLPerf 버전과 ASPLOS'26 게재 여부는 변동 가능성이 있다.

## Sources

MLCommons inference_rules.adoc · MLPerf v5.0/v5.1 블로그 · DynamoLLM 2408.00741 · SLOs-Serve 2504.08784 · Splitwise 2311.18677 · Mooncake 2407.00079 · LoongServe 2404.09526 · DistServe 2401.09670 · Etalon 2407.07000 · DuetServe 2511.04791 · Nexus 2507.06608 · Medha 2409.17264 · TetriInfer 2401.11181 · Infinite-LLM 2401.02669 · Llumnix 2406.03243 · vLLM bench serve 문서 · NVIDIA NIM 벤치 문서 · Bullet 2504.19516 / MuxWise 2504.14489(로컬 PDF `Papers/`).
