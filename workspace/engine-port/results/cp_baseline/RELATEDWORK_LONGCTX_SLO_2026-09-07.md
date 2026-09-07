# Related work — "기존 LLM 서빙 연구는 TTFT/TPOT SLO 임계를 **어떻게 정했는가**", 특히 long-context

작성: 2026-09-07 (venue-strategist, 조사 1회). 대상 트랙: `cp_baseline`.
**전제 문서**: `reports/serving_slo_survey.md`(2026-07-18) · `reports/longcontext_trace_plan.md` §4.3
(2026-07-25 venue-strategist 교정). 이 문서는 그 둘을 **중복하지 않고 위에 쌓는다**.

**출처 규율**: 아래 인용문은 전부 논문 본문(또는 초록) 문장이다. 우리 추정/해석은 `[우리 해석]`,
확인 못 한 것은 `[미확인]`으로 표시했다. 웹 인용은 2026-09-07 접근분이며, 수치·날짜는
**검증 필요**(모델 지식 컷오프 이후 변동 가능, 특히 2026년 arXiv ID들).

---

## 0. 한 줄

★**long-context 서빙에는 Spheron식 "절대 SLO 표"가 문헌에 존재하지 않는다** — 사용자의 판단이
맞다. 문헌이 실제로 쓰는 것은 셋 중 하나다: **(1) 응용 class별로 크게 완화한 절대 상수**
(요약 15s, Mooncake 30s, MLPerf 405B 6s), **(2) 그 요청 자신의 무경쟁 실행 지연의 배수**
(Splitwise 2×/3×/6×, Mooncake 10×, LoongServe 25×, Sarathi-Serve 5×/25×), **(3) TTFT를 SLO에서
아예 빼고 TBT/throughput만 구속**(Sarathi-Serve, LoongServe, Llumnix, long-ctx 커널 논문 전부).
길이-의존 TTFT deadline을 **명시적으로 처방한 논문은 Etalon/Metron 하나뿐**이며, 그 논문은
"정적 TTFT SLO는 실용적이지 않다"고 직접 쓴다.

★**우리 자신의 문서 오류 1건 발견** — §6 참조 (`serving_slo_survey.md` §2의 DistServe
"SLO scale" 서술이 논문 본문과 다르다).

---

## 1. SLO 설정 방법의 분류 (요청한 (a)–(f))

### (a) 절대 상수 — 근거를 대는가?

**대부분 안 댄다. 그리고 그 사실을 논문이 직접 인정한 사례가 있다.**

- **DistServe** (OSDI'24, [arXiv 2401.09670](https://arxiv.org/html/2401.09670v3)) — 본문:
  > "we set the SLOs empirically based on their service target because there exists no available
  > SLO settings for these applications as far as we know."

  Table 1(논문 본문 수치):

  | Application | Model | TTFT | TPOT | Dataset |
  |---|---|---|---|---|
  | Chatbot | OPT-13B | 0.25s | 0.1s | ShareGPT |
  | Chatbot | OPT-66B | 2.5s | 0.15s | ShareGPT |
  | Chatbot | OPT-175B | 4.0s | 0.2s | ShareGPT |
  | Code Completion | OPT-66B | 0.125s | 0.2s | HumanEval |
  | **Summarization** | OPT-66B | **15s** | 0.15s | **LongBench** |

  ⇒ ★**long-context(요약/LongBench) 행의 TTFT가 15s** — 이것이 top-tier 논문이 내놓은 사실상
  유일한 "long-ctx 절대 TTFT" 좌표다. 그리고 **모델 크기에 따라 TTFT를 0.25→2.5→4.0s로 올린다**
  (= 절대값이지만 "실행 시간에 비례하게" 손으로 조정한 것). `[우리 해석]` 이는 (b)의 비공식판이다.

- **LayerKV** ([arXiv 2410.00428](https://arxiv.org/html/2410.00428)) — 본문:
  > "the TTFT SLO is set to 3000 ms and the TPOT SLO to 200 ms for each request"

  context 128 → 16k를 스윕하면서 **길이 무관 고정 임계**를 쓴다. 같은 논문이
  > "TTFT exhibits a quadratic increase as context length extends, while TPOT scales linearly"

  라고 쓰면서도 SLO는 고정이다. `[우리 해석]` = 길이가 길어지면 구조적으로 위반이 늘어나는
  설계이며, 논문은 그것을 **시스템 최적화로 줄일 문제**로 프레이밍한다(불가능 영역으로 보지 않음).

- **Niyama** ([arXiv 2503.22562](https://ar5iv.labs.arxiv.org/html/2503.22562), 2025) — Table 2:
  Q1(interactive) TTFT **6s** / TBT **50ms**, Q2(non-interactive) TTLT **600s**, Q3 TTLT **1800s**.
  deadline 모형은 `D_first = t_arrival + SLO_TTFT`, `D_n = t_arrival + SLO_TTFT + (n-1)·SLO_TBT`.
  본문에 **baseline 실행시간 대비라는 서술은 없다**(절대값). 요약 시나리오 TTFT 2s 언급 있음.

- **MuxWise** ([arXiv 2504.14489](https://arxiv.org/html/2504.14489) — ★우리 external prior art) —
  > "We set the TBT SLO target to 50ms for Llama3-8B and 100ms for Llama3-70B, following prior works."

  characterization에서 TTFT 400ms 사용. 워크로드 입력길이(논문 수치, min/median/max 형식):
  ShareGPT `4/226/1024`, **LooGLE `3380/30k/81k`**, OpenThoughts `311/709/4633`, multi-turn 재사용
  context `0/4496/120k`. ⇒ ★**가장 가까운 선행이 이미 long-ctx(81k)까지 밟았고, SLO는 절대
  상수 + "following prior works"로 정했다.**

- **Bullet** ([arXiv 2504.19516](https://arxiv.org/html/2504.19516), ASPLOS'26으로 보임 —
  [저자 페이지 PDF](https://xianweiz.github.io/doc/papers/26asplos_bullet.pdf), `[미확인]` 정확한 venue/연도)
  — SLO 준수율을 보고하나 임계 설정 근거는 이 조사에서 **확인 못 함** `[미확인]`.

- ★**MLPerf Inference** — 커뮤니티가 합의한 유일한 **표준화된 절대 표**. server 시나리오
  latency constraint(2차 출처 기반, `[미확인]` 공식 rules 대조 필요):
  Llama2-70B **TTFT 2000ms / TPOT 200ms**; Llama2-70B **Interactive TTFT 450ms / TPOT 40ms**;
  Llama3.1-405B **P99 TTFT 6s / P99 TPOT 175ms**; v6.0 DeepSeek-R1 interactive **P99 TTFT ≤1.5s /
  TPOT ≤15ms**.
  ⇒ ★**MLPerf도 "길이"가 아니라 "모델 크기 + 대화형/비대화형 시나리오"로 TTFT를 늘린다.**
  (출처: [MLCommons v5.0](https://mlcommons.org/2025/04/llm-inference-v5/) ·
  [v6.0 GPT-OSS/DeepSeek 업데이트](https://mlcommons.org/2026/03/mlperf-inference-gpt-oss/) ·
  [NVIDIA v5.0 블로그](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/))

### (b) 무경쟁(unloaded) 실행 지연의 배수 — ★long-context의 사실상 표준

- **Splitwise** (ISCA'24, [arXiv 2311.18677](https://arxiv.org/html/2311.18677v2)) — Table VI를
  "a request running on DGX-A100 under no contention" 대비 slowdown으로 정의:
  **TTFT P50 2× / P90 3× / P99 6×**, **TBT P50 1.25× / P90 1.5× / P99 5×**.
  워크로드(논문 수치): coding 프롬프트 중앙값 **1500 tok**·출력 중앙값 **13 tok**;
  conversation 프롬프트 중앙값 **1020 tok**·출력 중앙값 **129 tok**.
  본문: "TTFT for both models grows almost linearly as the prompt size increases".
  ⇒ ★**denominator가 요청마다 다르므로 이 SLO는 *암묵적으로 길이-정규화*되어 있다.** 우리가
  하려는 "무경쟁 바닥으로 정규화"와 **같은 계보**다.

- **Mooncake** (FAST'25 / ToS 2025, [arXiv 2407.00079](https://arxiv.org/html/2407.00079v3)) —
  정규화 임계 **TTFT_P90 = 10×, TBT_P90 = 5×**(단일요청 성능 대비), 실배포 실험에선
  절대 **TTFT 상한 30초 / TBT 0.1초**. production trace 평균 입력 **7,590 tok**·출력 **182 tok**
  (입출력비 ≈720). 시뮬 long-ctx: 16k/32k/64k/**128k** 입력 × 512 출력.
  ⇒ ★**실배포 TTFT SLO가 30초.** long-context 서빙의 실제 절대 좌표는 여기다.

- **LoongServe** (SOSP'24, [arXiv 2404.09526](https://arxiv.org/html/2404.09526v2)) —
  > "set SLO to 25× of the inference latency"

  + 보고 지표 자체가 길이 정규화: "normalized per-token latency … end-to-end latency divided by
  their sequence lengths", "**normalized input latency, i.e., the mean of prefill phase time divided
  by the input lengths**", "normalized output latency …", 그리고 "the maximum throughput under this SLO".
  데이터셋 길이: ShareGPT `4–2.3K`, L-Eval `2.7K–210.5K`, LV-Eval `15.1K–497.3K`.
  ⇒ ★**SOSP 논문이 497K 토큰까지 다루면서 절대 TTFT 임계를 전혀 쓰지 않는다.**

- **Sarathi-Serve** (OSDI'24, [arXiv 2403.02310](https://arxiv.org/html/2403.02310v2)) —
  > "P99 TBT to be equal to 5× and 25× the execution time of a decode iteration for a request
  > (with prefill length of 4k and 32 batch size)"

  → Table 3 절대환산: Mistral-7B 0.1/0.5s, Yi-34B 0.2/1.0s, LLaMA2-70B 1.0/5.0s, Falcon-180B 1.0/5.0s.
  **TTFT SLO는 없다**: "We focus on the median value for the TTFT since this metric is obtained only
  once per user request." ⇒ (c)절 아님, **아래 S3 유형의 원조**.

- **DistServe "SLO scale"** — ★**주의(정정 대상)**: 논문 본문은
  > "We fix the rate and then linearly scale the two latency requirements in Table 1 simultaneously
  > using a parameter called SLO Scale."

  즉 **곱해지는 base는 Table 1의 경험적 절대값**이지 무경쟁 실행 지연이 아니다. 무경쟁 지연을
  분모로 쓰는 쪽은 Splitwise/Mooncake/LoongServe다. → §6 정정 항목.

### (c) 인간 지각·응용 근거

- **Andes** ([arXiv 2404.16283](https://arxiv.org/abs/2404.16283), 2024) — QoE를 실제 token
  delivery timeline vs **기대 TDT**(expected TTFT + expected token delivery speed)로 정의하고,
  TDS를 **"the user's typical reading speed"**에서 끌어온다. 핵심 주장: "generating text too fast
  (than user reading speed) does not yield QoE benefits". `[미확인]` 최종 venue.
- **GenZ** ([arXiv 2406.01698](https://arxiv.org/html/2406.01698)) — Table III(논문 수치):

  | Use case | in/out tok | TTFT/TPOT |
  |---|---|---|
  | Question Answering | 1000/200 | 0.2s / 10ms |
  | Chat Services | 3000/1000 | 0.2s / 10ms |
  | **QA + RAG** | **10000**/200 | **0.4s** / 10ms |
  | **Text Summarization** | **15000**/1000 | **2s** / 20ms |
  | **Code Generation** | **20000**/50 | **0.5s** / 20ms |

  본문은 Table III 값의 근거를 대지 않으나, AI-assistant 절에서 출력 속도만 인간 기준으로 정당화:
  "we assume it can generate output at the rate at which humans read or listen, which amounts to
  **300 words per minute**".
  ⇒ ★**결정적 관찰**: 입력 20000 tok의 code generation이 TTFT 0.5s인데 15000 tok 요약은 2s다 —
  **길이와 TTFT 예산이 단조하지 않다**. 즉 이 표조차 "길이의 함수"가 아니라 **응용의 함수**다.
- (비-피어리뷰) Spheron 2026 가이드 = 우리가 이미 쓴 좌표. 학술 인용가치 낮음(§5 리스크 참조).

### (d) 베이스라인 시스템 대비 상대값 / SLO 없음

- **Llumnix** (OSDI'24, [arXiv 2406.03243](https://arxiv.org/html/2406.03243v1)) — 평가에서
  **명시적 SLO 임계를 정의하지 않는다**. mean·P99 지연을 보고하고 baseline 대비 배수
  ("up to 15× improvement")로 주장한다. `[우리 해석]` OSDI 논문도 SLO 임계 없이 통과한다는
  선례 — "SLO 좌표가 없으면 논문이 안 된다"는 우리 트랙의 암묵 가정은 틀렸다.
- **long-context 커널/알고리즘 계열 전부**: MInference 1.0(NeurIPS'24), DuoAttention(ICLR'25),
  Star Attention(NVIDIA), InfiniteHiP — 전부 **SLO 임계 없이** 속도향상 배수 + 정확도만 보고한다.
  (예: MInference "reduces inference latency by up to 10× for pre-filling on an A100").
  ⇒ ★**long-context 논문이 SLO 표를 안 쓰는 것은 결함이 아니라 관행이다.**

### (e) SLO를 sweep하고 곡선을 보고 (임계 고정 회피)

- **DistServe**: SLO Scale 스윕 + "SLO attainment" 곡선 (위 인용).
- **Vidur / Vidur-Search** (MLSys'24, [arXiv 2405.05465](https://arxiv.org/abs/2405.05465)) —
  TTFT/TBT SLO를 **제약**으로 두고 parallelism·스케줄러·GPU SKU를 탐색해 **QPS/$**를 최대화.
  capacity 정의: "maximum queries per second that it can support without the queuing delay blowing
  up", 구체적으로 **P99 scheduling delay < 5초**로 구속.
- ★**Human-Less LLM Serving** ([arXiv 2606.20577](https://arxiv.org/html/2606.20577v1), 2026) —
  **throughput-SLA frontier**를 정의:
  > "the set of (SLO, throughput) pairs achievable by a serving system under a fixed workload:
  > for each SLO value, the maximum output token throughput the system can sustain while meeting the SLO."

  그리고 **"human tax"가 context와 함께 커진다**고 정량화: "At 8K context, the throughput gap … is
  15% … At 32K the gap widens to 40% … At 64K it reaches 60%". 하드웨어 Qwen-2.5-32B FP16 /
  8×H20 TP. ★**단, 무경쟁 지연 바닥을 입력 길이의 함수로 재지는 않는다**(§4 참조).
- **Beyond the Buzz: A Pragmatic Take on Inference Disaggregation**
  ([arXiv 2506.05508](https://arxiv.org/pdf/2506.05508)) — 같은 frontier 방법론 사용.
  `[미확인]` 본문 수치·길이 체제(PDF 텍스트 추출 실패).

### (f) 길이 정규화 SLO — **쓰는 논문이 있다. 단 두 갈래이고 서로 다르다.**

| 갈래 | 정의 | 누가 | 축 |
|---|---|---|---|
| **출력 정규화**(원조) | end-to-end latency ÷ **output** length | **Orca**(OSDI'22), **vLLM**(SOSP'23) 이후 표준 "normalized latency" | **decode** 축 — 우리 질문과 무관 |
| **입력 정규화** | prefill phase time ÷ **input** length ("normalized input latency") | **LoongServe**(SOSP'24) | ★**prefill 축 — 우리가 찾던 것** |
| **길이-의존 deadline 곡선** | `D_i = D_p + i·D_d`, **`D_p` = 프롬프트 길이의 함수(프로파일링으로 fitting)** | ★**Etalon/Metron**([arXiv 2407.07000](https://arxiv.org/html/2407.07000v2)) | prefill+decode |
| **요청별 무경쟁 지연 배수** | 관측 ÷ 그 요청의 no-contention 지연 | Splitwise·Mooncake·LoongServe·Sarathi-Serve | 암묵적 길이 정규화 |

★**Etalon/Metron이 이 질문에 대한 문헌의 가장 직접적인 답**이다(본문 인용):
> "TTFT is oblivious of prompt length"
> "defining a static Service Level Objective (SLO) on TTFT as a measure for user-facing
> responsiveness of the system is not practical."
> "The prefill time increases quadratically in the size of input prompt. Therefore, setting a
> static value for `D_p` is impractical." → **프로파일링으로 `D_p(L)` 곡선을 fitting하라고 처방.**

`[미확인]` Etalon의 최종 venue(arXiv 2407.07000, 구제목 Metron; Microsoft Research 게재 기록 있음,
MLSys 확정 여부 확인 못 함).

⇒ 이 항목이 `longcontext_trace_plan.md` §4.3의 2026-07-25 판정("연속 길이-정규화 임계
`a+b·L`은 학회 통용 방법론이 아니다 = ad-hoc")을 **부분적으로 뒤집는다**: 계보가 **있다**
(Etalon). 다만 Etalon도 `a+b·L` 같은 **선형 폐형식이 아니라 "프로파일링 곡선 fitting"**을
처방하므로, §4.3의 실무 처방(절대 class SLO + slowdown 병기)은 여전히 안전하다.

---

## 2. long-context 서빙 논문들이 실제로 쓴 것 (표)

| 논문 | venue/연도 | 입력 길이 체제(논문 수치) | SLO를 어떻게 정했나 | TTFT 취급 |
|---|---|---|---|---|
| **DistServe** | OSDI'24 | ShareGPT/HumanEval + **LongBench**(요약) | 경험적 절대값 + SLO-Scale 곱셈 스윕 | **고정 절대, 단 요약은 15s로 완화** |
| **Sarathi-Serve** | OSDI'24 | prefill 4k 기준 SLO 산정, 데이터셋 mixed | **TBT = decode iteration 실행시간의 5×/25×** | ★**SLO에서 제외**. median만 보고 |
| **Splitwise** | ISCA'24 | coding p50 1500 / conv p50 1020 tok | **무경쟁 DGX-A100 대비 slowdown 배수** | 2×/3×/6× (P50/P90/P99) = 암묵 길이정규화 |
| **Llumnix** | OSDI'24 | ShareGPT류 | **SLO 임계 없음** | 임계 없음, mean/P99 보고 |
| **LoongServe** | SOSP'24 | ShareGPT 4–2.3K / L-Eval 2.7K–**210.5K** / LV-Eval 15.1K–**497.3K** | **SLO = 추론 지연의 25×** | ★TTFT 대신 **normalized input latency**(prefill÷입력길이) |
| **Mooncake** | FAST'25 | production 평균 입력 **7,590 tok**; 시뮬 16k–**128k** | **TTFT_P90 10× / TBT_P90 5×** (+ 실배포 절대 **30s / 100ms**) | 완화된 절대 + 배수 병용 |
| **Vidur** | MLSys'24 | (시뮬) | TTFT/TBT를 탐색 **제약**으로, capacity=P99 sched delay<5s | 제약, 곡선/탐색 |
| **Etalon/Metron** | arXiv'24 `[venue 미확인]` | 장프롬프트 포함 | ★**fluidity-index**, `D_p`를 길이의 함수로 프로파일 fitting | ★**정적 TTFT SLO를 명시적으로 기각** |
| **Andes** | arXiv'24 `[venue 미확인]` | 텍스트 스트리밍 | 기대 TTFT + 인간 **읽기 속도** 기반 TDS | QoE 곡선 |
| **Niyama** | arXiv'25 (MSR) | p50/p90 프롬프트 표 존재 | 절대 QoS class (TTFT 6s / TBT 50ms / TTLT 600·1800s) | 고정 절대, 길이 무관 |
| **LayerKV** | arXiv'24 | context **128 → 16k** 스윕 | 절대 **TTFT 3000ms / TPOT 200ms** | 고정 절대(길이 무관) → 길수록 위반 증가를 *최적화 문제*로 취급 |
| **MuxWise** | arXiv'25 (우리 prior art) | ShareGPT 1024 / **LooGLE 81k** / 재사용 120k | 절대 TBT 50·100ms "following prior works", TTFT 400ms(특성화) | 고정 절대 |
| **Bullet** | ASPLOS'26 `[미확인]` | `[미확인]` | `[미확인]` | `[미확인]` |
| **Helix Parallelism** | arXiv'25 (NVIDIA) | **1M 토큰 컨텍스트**(시뮬, DeepSeek-R1 671B FP4 / GB200 NVL72) | "latency budget"·**minimum achievable token-to-token latency** | TTFT 아님 — **TTL(decode) 예산** 중심 |
| **MInference / DuoAttention / Star Attention / InfiniteHiP** | NeurIPS'24 / ICLR'25 / arXiv / arXiv | 128K–3M | **SLO 없음** | 속도향상 배수 + 정확도만 |
| **Marconi** (hybrid Mamba-Transformer prefix caching) | MLSys'25 ([arXiv 2411.19379](https://arxiv.org/abs/2411.19379)) | hybrid long-ctx | **SLO 없음** (TTFT 절감 617ms 보고) | 상대 개선 |
| **ServeGen** (Alibaba) | NSDI'26 ([arXiv 2505.09999](https://arxiv.org/html/2505.09999v2)) | 입력=Pareto+Log-normal 혼합, 출력=Exponential; 10M-context 모델 1종 존재 | SLO 설정 논문 아님 — 워크로드 생성. "naive 생성 대비 **50% under-provisioning** 회피" | — |
| **TraceLab** (coding agent) | arXiv 2026 ([2606.30560](https://arxiv.org/html/2606.30560v2)) | ★Claude prefix **p50 126,180 / p90 467,082 / p99 918,111 tok**; append p50 857; 출력 p50 252 | **SLO 제안 없음**, 관측 지연만(“median residual TTFT … 1.5 s to 2.9 s”, 생성 median 5.7s/p90 22.2s) | 관측치만 |
| **Human-Less LLM Serving** | arXiv 2026 ([2606.20577](https://arxiv.org/html/2606.20577v1)) | 8K/32K/64K | ★**throughput-SLA frontier**(SLO 스윕) | 임계 대신 frontier |
| **Preble** | `[미확인]`(ICLR/arXiv 2407.00023) | 긴 프롬프트·프리픽스 공유 | avg/p99 latency·TTFT·TPOT 상대 비교 `[미확인 세부]` | `[미확인]` |
| **NanoFlow** | OSDI'25 `[미확인 세부]` | `[미확인]` | `[미확인]` | `[미확인]` |
| **vLLM / Orca** | SOSP'23 / OSDI'22 | short | **normalized latency**(E2E÷출력길이) | TTFT 임계 없음 |

---

## 3. ★핵심 질문: 프롬프트가 길어질 때 TTFT SLO를 어떻게 취급하는가

문헌에서 **정확히 네 가지 전략**만 관찰된다.

**S1 — 길이 무관 고정 TTFT를 유지하되, long-ctx "응용 class"에는 훨씬 큰 상수를 준다.**
DistServe 요약 **15s**, Mooncake 실배포 **30s**, Niyama 요약 **2s**, MLPerf 405B **6s**,
LayerKV **3s**(128–16k 전 구간 공통), "typical serving requirement of **10 seconds**"(128K 맥락,
2차 출처 `[미확인]`).
⇒ **긴 요청이 구조적으로 위반**하는 것을 인정하지 않고 **class 단위로 예산을 키워 회피**한다.
★인터랙티브 좌표(100–400ms)를 long-ctx에 그대로 들이대는 논문은 **하나도 못 찾았다.**

**S2 — SLO를 그 요청 자신의 무경쟁 실행 지연의 배수로 정의(= 암묵적 길이 정규화).**
Splitwise(2×/3×/6×), Mooncake(10×/5×), LoongServe(25×), Sarathi-Serve(TBT 5×/25×).
⇒ ★**long-context에서 절대 TTFT SLO를 "그대로 쓰는 것"은 문헌에서 정당화되지 않는다.
정당화되는 것은 배수화(S2)거나 class 완화(S1)다.**

**S3 — TTFT를 SLO에서 아예 뺀다.** Sarathi-Serve(명시적으로 median만 보고), LoongServe(정규화
지연 + 25× 하 최대 처리량), Llumnix(임계 없음), long-ctx 커널 논문 전부.
⇒ **top-tier에서 완전히 허용되는 선택지.** 우리 트랙이 "SLO 좌표를 못 고르면 막힌다"고 본 것은
문헌 근거가 없다.

**S4 — 길이-의존 deadline 곡선을 프로파일링으로 만든다.** **Etalon/Metron 단독.**
"setting a static value for `D_p` is impractical" → `D_p(L)`를 fitting.

★**종합 답**: long-context에서 절대 TTFT SLO를 그대로 쓰는 것은 **문헌이 정당화하지 않으며,
실제로 아무도 안 한다**(하는 경우는 임계를 3s–30s로 키운 뒤다). 우리 트랙이 감사에서 받은
死因("좌표와 프롬프트 길이 분위수의 짝짓기가 성립하지 않는다")은 **문헌과 정합적인 진단**이다.

---

## 4. "우리가 하려는 것"이 이미 있는가

우리 형태 = **"무경쟁 지연 바닥을 길이의 함수로 실측 → 어떤 부하·정책으로도 달성 불가능한
SLO 영역을 *필요조건*으로 확정"**.

| 선행 | 어디까지 했나 | 남긴 것 |
|---|---|---|
| **Splitwise** (ISCA'24) | ★**무경쟁 A100에서 TTFT vs 프롬프트 길이를 실측**하고 그것을 **SLO의 분모로 사용**. "TTFT … grows almost linearly as the prompt size increases" | 곡선을 **SLO 분모**로만 씀. **불가능 영역(feasibility screen)으로 프레이밍하지 않음** |
| **Etalon/Metron** | ★**정적 TTFT SLO가 실용적이지 않다는 주장 + `D_p(L)` 프로파일 처방**을 이미 함 | **모델별 바닥 곡선을 공표하지 않음**. hybrid(SSM) 모델·SM 분할 미다룸. "불가능 영역"이라는 판정 산출물 없음 |
| **Floor First: "Think Before You Grid-Search"** ([arXiv 2607.05876](https://arxiv.org/abs/2607.05876), 2026-07-07, 단독저자, venue 없음) | ★이름이 가장 가깝다. 초록 본문: decode step을 **5차원 자원 벡터**(HBM bytes, FLOPs, network bytes/messages, KV capacity)로 모델링, "summing within a resource and maximizing across resources gives an optimistic floor, the plain sum a pessimistic one", 측정치가 `[max, sum]` 구간 어디에 떨어지는지로 overlap 품질을 읽음. 사례연구 = 671B MoE/MLA on 16×H20 | ★**해석적 floor**이지 **실측 무경쟁 바닥이 아니다**. **decode step** 대상이며 **prefill/TTFT×길이 스윕이 아니다**. SLO 불가능 영역 판정 아님. `[미확인]` 본문(초록만 확인; 1차 웹요약이 "no-contention 측정·입력길이 함수"라고 했으나 **초록과 불일치** — 본문 대조 전엔 인용 금지) |
| **GenZ** (arXiv'24) | use-case별 TTFT/TPOT 목표 하 **플랫폼 요구사항 탐색**. Mamba 포함 | 본문에 **"SLO X를 context L에서 만족하려면 N GPU 필요"류 feasibility bound 진술 없음**(descriptive) |
| **LLM-Viewer / "LLM Inference Unveiled"** ([arXiv 2402.16363](https://arxiv.org/abs/2402.16363)) | roofline으로 연산/메모리 병목·이론 하한 | **op/layer 수준**. 서빙 SLO 달성가능성 영역 아님 |
| **Vidur / Vidur-Search** (MLSys'24) | SLO 제약 하 **config 탐색**, capacity 정의(P99 sched delay<5s) | **탐색 결과 = 최적 config**. "어떤 config로도 불가능"은 부산물일 뿐 산출물이 아님. 시뮬 기반 |
| **Human-Less LLM Serving** (arXiv 2026) | ★**throughput-SLA frontier** + context별 human tax(8K 15% / 32K 40% / 64K 60%) | **무경쟁 바닥을 길이 함수로 재지 않음**. 대상은 "지연 제약이 throughput에 물리는 세금"이지 "달성 불가능 SLO 영역"이 아님 |
| **LLMServingSim** | `[미확인]` — 이번 조사에서 확인 못 함 | — |

★**판정**: **정확히 그 형태의 기여는 못 찾았다.** 그러나 **조각은 전부 이미 있다** —
곡선 측정(Splitwise), 필요성 주장과 처방(Etalon), floor-first 어휘(2607.05876), frontier
방법론(Human-Less). 남은 빈틈은 **"조각들을 하나의 판정 산출물로 묶고, hybrid(attn+SSM)
모델과 SM-분할 PD-mux 기판에서 실측한다"**는 **좁은 슬라이스**다. `[우리 해석]` 이것은
"새 문제 발견"이 아니라 **"알려진 관행의 정량화"**다.

---

## 5. 정직한 fit 평가

**(i) 빈 자리인가** — **부분적으로만.** 위 표대로 "정적 TTFT SLO는 long-ctx에서 부적절"은
**이미 출판된 주장**(Etalon)이고, "무경쟁 지연으로 정규화"는 **이미 표준 관행**(Splitwise/
Mooncake/LoongServe). 우리가 더할 수 있는 것은 **(1) hybrid attn+SSM 모델에 대한 바닥 곡선**,
**(2) 그것을 SLO 좌표 스크리닝의 *필요조건*으로 형식화**, **(3) SM 분할(green-context) 기판에서
"분할이 바닥을 얼마나 올리는가"**. (3)이 가장 우리 고유하지만, **그건 이미 우리 트랙의 다른
결론(SM 민감도 C2 계열)과 겹친다.**

**(ii) 시스템 학회 논문 크기인가** — **아니다(현재 증거로는).**

| 학회 | fit | 왜 |
|---|---|---|
| OSDI/SOSP/NSDI | **약** | constructive system 없음. 우리 정본은 negative(layer-aware 死·동적 열위). "프로파일링해서 표를 만들었다"는 rejection 사유가 된다. Llumnix가 SLO 임계 없이 통과한 선례는 **SLO 논의 자체가 기여로 안 쳐진다**는 뜻이기도 하다 |
| EuroSys/ATC | **약–중** | characterization+guideline 트랙 존재. 단 **1 GPU·1 모델·1 노드·동시성 1·≤8192 tok**로는 일반화 주장 불가 |
| **MLSys** | ★**중** | 모델 특성화+서빙 정책에 관대. hybrid(Mamba/SSD) 서빙 특성화는 Marconi(MLSys'25) 선례가 있어 **주제 적합성은 확인됨**. 단 지금 크기로는 **논문 1편이 아니라 섹션 1개** |
| ISCA/MICRO/HPCA | **약** | SM 분할 기전 자체는 fit하나, SLO 임계 설정 논의는 아키텍처 학회의 보상 대상이 아님 |
| SC | **약** | 단일 GPU 미시, 규모 없음 |
| ★**워크숍** (HotInfra/EuroMLSys/MLArchSys/HotCarbon류) | ★**강** | "long-ctx SLO 실현가능성 하한 + hybrid 모델 바닥 곡선"은 **정확히 워크숍/단문 크기**. 조기 피드백·선점에 유리 |

**(iii) 결론** — **현재 증거로는 워크숍/단문 크기이며, 본 논문에서는 *방법론 섹션*으로
들어가는 것이 정직하다.** 구체적으로: "우리는 SLO 좌표를 외부 표에서 빌리지 않고, 무경쟁
바닥 곡선으로 **필요조건 스크린**을 만든 뒤 그 위에서만 정책을 비교했다"는 **평가 방법론
정당화**로 쓰는 것이 가장 강하고 가장 정직하다. 이는 CLAUDE.md 게이트 #6(metric cliff 회피)·
게이트 #1(정책 주장은 서빙 실증)과 정합하며, 새 성능 주장을 만들지 않는다.

### 이 각도를 논문 크기로 키우려면 필요한 것 (roadmap 언어로)

- **길이 축**: 8,192 초과(16K/32K/64K) 실측. **P6/W5**(1K/8K context 교대, 지원 시 16K)가 이미
  예고한 축. 모델 max ctx·rope 외삽 제약을 먼저 확정(→ `longcontext_trace_plan.md` §4.1의
  하드 블로커 재확인, Nemotron-Nano-9B-v2에 대해 미확인).
- **동시성 축**: 동시성 1 바닥은 **필요조건일 뿐**. 부하 하 바닥 상승을 재려면 **W7/W8/W9**
  (0.20/0.90·0.95/1.10·1.25·1.50 λ*)로 λ 축을 붙여야 "달성 가능 frontier"가 된다.
- **모델 축**: hybrid vs 동급 pure-Transformer 대조가 있어야 "hybrid 고유"라고 말할 수 있다.
  단 **게이트 #83(엔진 강제 교락: Nemotron-H+triton 부팅거부 / Zamba2+flashinfer 스케줄러 사망)**
  때문에 "모델 고정·백엔드만 변경" 셀은 존재 불가 — 대조 설계 시 선반영.
- **정책 축(있어야 논문이 됨)**: 바닥 곡선이 **어떤 결정을 바꾸는가**. 후보 = **B1(one global
  static) vs B2(per-workload offline best static oracle)** 선택이 길이 체제에 따라 뒤집히는가.
  뒤집히지 않으면 그 자체가 "decode-heavy static 고정" 정본의 **범위 확장 증거**(positive가
  아니라 범위 확정).

### 정직한 리스크

- **R1 (가장 큼)**: 동시성 1 바닥은 **필요조건**이지 달성가능 frontier가 아니다. 리뷰어의 첫
  질문은 "그래서 이 스크린이 어떤 잘못된 결론을 막아줬나"다. **답을 미리 준비 못 하면 기여가 아니다.**
- **R2**: 바닥 자체가 **config 의존**이다. 이 저장소가 이미 측정한 것 — `--chunked-prefill-size`가
  prefill 예산과 piecewise CUDA graph 캡처 범위를 **동시에** 결정한다(F3 관측). 따라서 "바닥"은
  단일 수가 아니라 **(엔진 config, cudagraph 상태) 위의 함수**이며, 등록 없이 "floor"라 부르면
  게이트 #80(출처 허위)·#61/72(라벨 오독) 계열 재발이다.
- **R3**: Spheron 가이드는 **비-피어리뷰 블로그**다. 논문에서 이것을 SLO 근거로 쓰면 즉시 공격
  대상이 된다. 학술 인용은 **MLPerf(표준) + DistServe Table 1 + Splitwise Table VI**로 갈아타야 한다.
- **R4**: `2607.05876`(Floor First)은 **이름이 우리 주장과 충돌**한다. 본문을 읽고 차이를 명시하지
  않으면 "이미 있다"로 처리될 수 있다. **선점 위험이 실재**한다(2026-07 arXiv).
- **R5**: 이 각도를 **positive constructive win으로 포장하면 claims-auditor에 걸린다.** 이것은
  측정 규율이지 성능 주장이 아니다. 정본과 충돌하는 framing(예: "동적 제어가 long-ctx에서는
  이긴다")은 이 문서로 뒷받침되지 않는다.

---

## 6. ★우리 저장소 정정 필요 항목 (doc-steward 이관)

1. **`reports/serving_slo_survey.md` §2** — "DistServe는 **'SLO scale' = 무경쟁 단일 요청 실행
   지연의 배수**로 SLO를 정의하고"는 **논문 본문과 불일치**. 본문은
   "linearly scale the two latency requirements **in Table 1**"(= 경험적 절대값의 배수).
   무경쟁 지연을 분모로 쓰는 것은 **Splitwise/Mooncake/LoongServe**다. → 문장 교체 + 출처 이관.
   `[검증 필요]` arXiv HTML v3 기준. 카메라레디 PDF 대조 권고.
2. **`reports/longcontext_trace_plan.md` §4.3** — "(i) 연속 길이-정규화 임계는 학회 통용
   방법론이 아니다 = ad-hoc"은 **부분 정정 필요**: 계보가 있다(**Etalon/Metron**이 `D_p(L)`
   프로파일 fitting을 명시적으로 처방, LoongServe가 normalized input latency를 사용).
   단 **선형 폐형식 `a+b·L`**에 대한 기각은 유효하며, 처방(절대 class SLO + slowdown 병기)도 유효.
3. `serving_slo_survey.md` §1의 Spheron 표는 **비-피어리뷰**임을 문서에 배너로 명시할 것(§5 R3).

---

## 7. 출처 목록 (2026-09-07 접근)

- DistServe, OSDI'24 — https://arxiv.org/html/2401.09670v3 · https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
- Sarathi-Serve, OSDI'24 — https://arxiv.org/html/2403.02310v2 · https://www.usenix.org/conference/osdi24/presentation/agrawal
- Splitwise, ISCA'24 — https://arxiv.org/html/2311.18677v2
- Llumnix, OSDI'24 — https://arxiv.org/html/2406.03243v1
- LoongServe, SOSP'24 — https://arxiv.org/html/2404.09526v2 · https://dl.acm.org/doi/10.1145/3694715.3695948
- Mooncake, FAST'25 / ACM ToS 2025 — https://arxiv.org/html/2407.00079v3 · https://dl.acm.org/doi/10.1145/3773772
- Vidur, MLSys'24 — https://arxiv.org/abs/2405.05465
- Etalon (구 Metron) — https://arxiv.org/html/2407.07000v2 · https://arxiv.org/abs/2407.07000 `[venue 미확인]`
- Andes — https://arxiv.org/abs/2404.16283 `[venue 미확인]`
- Niyama — https://ar5iv.labs.arxiv.org/html/2503.22562 · https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/niyama.pdf
- LayerKV — https://arxiv.org/html/2410.00428
- GenZ — https://arxiv.org/html/2406.01698
- LLM-Viewer / LLM Inference Unveiled — https://arxiv.org/abs/2402.16363
- Marconi, MLSys'25 — https://arxiv.org/abs/2411.19379 · https://proceedings.mlsys.org/paper_files/paper/2025/hash/7c180af017258d239bac6248d1eb26ac-Abstract-Conference.html
- ServeGen, NSDI'26 — https://arxiv.org/html/2505.09999v2 · https://www.usenix.org/conference/nsdi26/presentation/xiang-servegen
- TraceLab — https://arxiv.org/html/2606.30560v2 · https://syfi.cs.washington.edu/blog/2026-06-25-tracelab/
- Human-Less LLM Serving — https://arxiv.org/html/2606.20577v1
- Think Before You Grid-Search (Floor First) — https://arxiv.org/abs/2607.05876 `[본문 미확인]`
- Beyond the Buzz (disaggregation) — https://arxiv.org/pdf/2506.05508 `[본문 미확인]`
- MuxWise — https://arxiv.org/html/2504.14489
- Bullet — https://arxiv.org/html/2504.19516 · https://xianweiz.github.io/doc/papers/26asplos_bullet.pdf `[venue/SLO 미확인]`
- Helix Parallelism — https://arxiv.org/pdf/2507.07120
- MInference 1.0, NeurIPS'24 — https://neurips.cc/virtual/2024/poster/94208
- MLPerf Inference — https://mlcommons.org/2025/04/llm-inference-v5/ · https://mlcommons.org/2026/03/mlperf-inference-gpt-oss/ · https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/ `[공식 rules 대조 필요]`
- Revisiting SLO and Goodput Metrics — https://arxiv.org/html/2410.14257v1 (기존 조사분)
- Orca, OSDI'22 — https://www.usenix.org/system/files/osdi22-yu.pdf · vLLM, SOSP'23 — https://arxiv.org/pdf/2309.06180
