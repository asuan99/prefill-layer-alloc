# 실 서빙 goodput SLO 관행 조사 — "우리 3s는 어느 regime이었나"

조사: 2026-07-18. 동기: §1-16(HE0가 SLO 엄격도 의존)의 후속 — "옳은 SLO는 응용이 정한다"면, **실무 관행이 우리 tight-SLO regime(동적 승) 안에 있나 밖에 있나**를 확인해야 발견의 실무적 유의미성이 정해진다.
정본 [CONSENSUS.md](CONSENSUS.md) §1-16 · 재계측 [`../results/slo_sched/lengthnorm_slo_reanalysis.md`](../workspace/engine-port/results/slo_sched/lengthnorm_slo_reanalysis.md).

---

## 0. 한 줄

★**우리 정본 SLO(TTFT 3s)는 인터랙티브 서빙 어디에도 안 맞는 "배치 async" 값이었다.** 프로덕션 인터랙티브 SLO(chat 300ms · voice 150ms · code 100ms · RAG 400ms)는 **전부 우리가 "tight = 동적 승"이라 부른 regime(TTFT≲0.5s) 안**에 있다.
⇒ §1-16의 tight-SLO 발견은 실무적 예외가 아니라 **인터랙티브 서빙의 주류**이고, static이 이기던 3s는 **비인터랙티브 배치 한정**이었다.

---

## 1. 프로덕션 TTFT/ITL SLO budget (P99 기준)

Spheron 2026 SLO Engineering 가이드의 use-case별 표(P99):

| 응용 | TTFT P99 | ITL P99 | 우리 sweep상 위치 |
|---|---|---|---|
| Code completion (inline) | **100ms** | 25ms | 동적 승 (≪0.5s) |
| Voice agent | **150ms** | 30ms | 동적 승 |
| Interactive chat | **300ms** | 50ms | ★**교체 임계 = bind+GATE 승** (우리 0.335s) |
| Code completion (panel) | 300ms | 50ms | 동적 승 |
| RAG-augmented chat | 400ms | 80ms | 동적 승/대등 |
| **Batch agent (async)** | **3,000ms** | 200ms | ★**우리 정본 값 = static 승** |

- 다른 소스와 정합: chatbot 500ms–2s / voice·code <200ms / code inline <100ms (BentoML, Softcery, AIonX 등).
- **percentile 표준 = P99** ("배치 간섭 스파이크·KV eviction·GC를 평균은 숨기고 P99만 잡는다"). goodput/attainment도 통상 **P90–P99**.

★**결정적 대응**: 우리 정본 SLO = **TTFT 3000ms / TPOT 60ms**. TTFT 3000ms는 표에서 **정확히 "Batch agent (async)" 행**이다. 반면 TPOT 60ms는 **interactive chat(ITL 50ms) 수준** — 즉 우리는 **TPOT는 인터랙티브인데 TTFT만 배치인 비일관 SLO**를 써왔고, 그 TTFT 관대함이 §1-16의 "static 지배"를 만들었다.

## 2. goodput·SLO 설정의 표준 관행

- **goodput 정의(DistServe, 이후 표준)**: "SLO를 만족하며 지속 가능한 최대 req/s" = TTFT·TPOT 둘 다 임계 이하인 요청의 처리율. 우리 goodput 정의와 동일.
- ★**SLO를 *sweep*하는 것이 표준 방법론**: DistServe는 **"SLO scale" = 무경쟁 단일 요청 실행 지연의 배수**로 SLO를 정의하고, scale을 조이며(loose→tight) goodput 곡선을 측정한다.
  > ⚠️**정정(2026-09-07, 1차 출처 확인)** — 이 줄의 *"무경쟁 단일 요청 실행 지연의 배수"* 는 **DistServe 본문과 불일치한다.** 본문(arXiv 2401.09670v3, 평가절): *"We fix the rate and then **linearly scale the two latency requirements in Table 1** simultaneously using a parameter called SLO Scale."* ⇒ DistServe의 scale은 **Table 1의 절대값**(Chatbot OPT-13B 0.25s/0.1s · OPT-66B 2.5s/0.15s · OPT-175B 4.0s/0.2s · Code Completion 0.125s/0.2s · **Summarization 15s/0.15s[LongBench]**)을 곱한다. 같은 논문은 *"we set the SLOs **empirically** … because **there exists no available SLO settings** for these applications as far as we know"* 라고 스스로 적는다. **무경쟁 지연을 분모로 쓰는 것은 Splitwise·Mooncake·LoongServe**다 — Splitwise(ISCA'24)는 *"slowdown compared to a request running on DGX-A100 under **no contention**"* 로 TTFT 2×/3×/6× · TBT 1.25×/1.5×/5×(P50/P90/P99)를 쓴다. 나머지 문장(sweep이 표준·엄격할수록 승자 변동)은 유효하다. 상세 = `workspace/engine-port/results/cp_baseline/RELATEDWORK_LONGCTX_SLO_2026-09-07.md`. ⇒ **우리 fixed-tight sweep(§1-16)과 동형.** 우리 실수는 방법이 아니라 **단일 점(3s)만 측정하고 그것을 "확정"이라 부른 것**.
- ★**"엄격할수록 승자가 바뀐다"는 알려진 현상**: DistServe도 **SLO가 엄격해질수록 disaggregation(구조 분리)이 colocation보다 유리**해진다고 보고. 우리 발견(**엄격할수록 동적 split 제어가 static보다 유리**)은 같은 부류 — **엄격한 SLO가 시스템의 반응성/구조 이점을 드러낸다**는 일반 패턴의 한 사례.
- **길이-aware provisioning 권고**: "SLO를 median prompt length에 맞춰 잡지 말고 **P99 prompt length로 prefill budget을 산정**하라"(prefill은 length에 초선형). ⇒ 사용자의 원래 직관(길이 반영)은 **capacity provisioning 층위에선 실무 지지**. 단 §1-16 robustness는 **per-request SLO 순위 반전의 축은 길이-비례성이 아니라 절대 엄격도**임을 보였다 — 두 층위는 별개(provisioning=길이 / 순위=엄격도).

## 3. goodput 메트릭 자체에 대한 비판 (우리 §1-14와 공명)

"Revisiting SLO and Goodput Metrics in LLM Serving"(arXiv 2410.14257):
- **goodput은 abandonment를 유도**: SLO 살짝 넘긴 요청을 버리는 게 지표상 유리해짐(부분 기여 무시).
- **token-level SLO(TBT)가 과도하게 restrictive**: 버퍼된 토큰이 있으면 간헐적 stall은 무해한데 벌준다.
- 제안: 첫 토큰 기준 deadline `dᵢ = V×i`(V=사용자 정보처리 속도) + "smooth goodput".

⇒ 우리 **§1-14(임계 지시함수 goodput의 메트릭 절벽 = 과부하서 런길이 의존·ill-posed)** 와 독립적으로 같은 방향: **threshold-goodput은 취약한 지표**. 우리 tight-SLO 재계측도 이 취약성의 다른 단면(SLO 위치가 순위를 가름).

## 4. 결론 — §1-16의 실무적 지위

| | |
|---|---|
| **우리 정본 3s는 뭐였나** | **비인터랙티브 배치(async) SLO.** 인터랙티브 어디에도 안 맞음 |
| **인터랙티브 주류(chat/voice/code/RAG)** | **전부 TTFT≲0.5s = §1-16의 "동적 승/대등" regime** |
| **⇒ §1-16의 지위** | 실무적 예외가 아니라 **주류**. "decode-heavy static 지배"는 **배치 서빙 한정 권고**로 좁혀야 함 |
| **동적 제어의 값어치** | 인터랙티브 SLO에선 **성능 이득**(§1-16), 배치에선 견고성뿐(§1-11) |

★**단 정직한 유보**: §1-16는 여전히 **Zamba2 short-ctx·ShareGPT** 재계측이고, 인터랙티브 TTFT 100–400ms를 **직접 서빙으로 attain하는 측정은 안 했다**(재분석은 3s 벤치 데이터를 재스코어). 특히 현 bind+GATE는 TTFT≤3s 가정으로 튜닝됨 ⇒ **인터랙티브 SLO를 목적함수로 한 컨트롤러 재튜닝 + P99-attainment 지표 재측정**이 다음 단계(CONSENSUS §5-7).

## 5. 출처

- [Spheron — LLM Inference SLO Engineering (TTFT/ITL/P99 budgets), 2026](https://www.spheron.network/blog/llm-inference-slo-ttft-itl-latency-budget-guide-2026/)
- [DistServe (goodput, SLO scale, prefill-decode disaggregation), arXiv 2401.09670](https://arxiv.org/pdf/2401.09670) · [Hao AI Lab blog](https://haoailab.com/blogs/distserve/)
- [Revisiting SLO and Goodput Metrics in LLM Serving, arXiv 2410.14257](https://arxiv.org/html/2410.14257v1)
- [BentoML — Key metrics for LLM inference](https://bentoml.com/llm/llm-inference-basics/llm-inference-metrics) · [Anyscale — LLM latency/throughput metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics)
- [Softcery — Choosing an LLM for Voice Agents](https://softcery.com/lab/ai-voice-agents-choosing-the-right-llm) · [Hamming AI — Voice AI Latency](https://hamming.ai/resources/voice-ai-latency-whats-fast-whats-slow-how-to-fix-it)
