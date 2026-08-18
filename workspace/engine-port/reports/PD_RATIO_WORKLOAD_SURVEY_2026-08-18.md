# 서빙 워크로드 prefill:decode 비 — 축별 문헌 조사

2026-08-18 · venue-strategist 조사 · 메인 세션 저장 · GPU **0** · 새 성능 판정 **0건**
· **정본 아님** · claims-auditor 미실행

> 축 정의는 [`WORKLOAD_AXES_2026-08-18.md`](WORKLOAD_AXES_2026-08-18.md) 참조
> (① 토큰 수 · ② 순차 forward step · ③ FLOPs · ④ 병목/시간).

## 0. 한 줄

**"한 방향으로 움직이지 않는다."** 축 ①에서 세그먼트 분산이 **5자릿수**로 벌어졌고
(OpenThoughts 0.085:1 ↔ LooGLE 2000:1), 축 ④에서는 **거의 모든 세그먼트가 여전히 decode 지배**다.
데이터가 지지하는 "이동"은 (a) 입력 절대길이 증가, (b) prefix-cache 후 유효 prefill 축소,
(c) reasoning 출력 폭증 셋인데 **서로 반대 방향**이다.

## 1. 축이 갈리는 것을 문헌 자신이 말한 사례 (이 조사의 존재 이유)

**Splitwise(ISCA'24)** — 같은 논문 안에서:
- 축 ①: Azure coding prompt median **1500** / output median **13** (115:1 prefill-heavy)
- 축 ④: *"most of the E2E time is spent running the token phase. **This holds true even for the
  coding trace**, where prompt sizes are large and generated tokens few."*
- 축 ①↔④ 환산: *"1500 input tokens takes the same time as token phase with only 6 output tokens"*(BLOOM-176B)

**Sarathi-Serve(OSDI'24)**: *"the cost of linear operation for **1 decode token is nearly same as
128 prefill tokens**"*(Mistral-7B/A100) — 축③④ 환산 계수. ⚠️모델·HW·batch 의존, **이식 금지**.

## 2. 세그먼트별 (모든 수치에 축 라벨)

| 세그먼트 | 출처 | 수치 | 축 |
|---|---|---|---|
| chat | MuxWise/Drift(ASPLOS'26) T1 | ShareGPT in **4/226/1024**, out **4/195/1838** | ① |
| chat | Sarathi T2 | `openchat_sharegpt4` prompt median **1730**, out **415** | ① |
| chat | Splitwise | Azure conv prompt **1020** / out **129** | ① |
| chat | DuetServe T1 | Azure-Conv **1155/211** | ① |
| 코드 | Splitwise · DuetServe | **1500/13** · **2047/28** | ① |
| RAG | Sarathi T2 | `arxiv_summarization` **7059/208** | ① |
| RAG | MuxWise T1 | LooGLE in **3380/30k/81k**, out **2/15/326** | ① |
| RAG | POD-Attention(ASPLOS'25) | 사내 mean ctx **10.5K**, P:D **0–40**; 16K ctx서 attention *">60% of total inference time"* | ①·④ |
| agentic | MuxWise T1 | Mooncake conv **891/7538/123k** in, **1/342/20000** out, reused **4496** | ① |
| agentic | DuetServe T1 | Mooncake **12035/343** | ① |
| ★agentic | Agentic AI Workload Characteristics | raw in:out **53.9–559.8×**, **append:out 1.5–7.3×**, cache-hit **84.6–99.5%** | ① |
| ★★agentic | 〃 | ***"decode accounts for 91.0–98.6% of LLM time, prefill only 1.4–9.0%"*** | ④ |
| ★agentic | TraceLab | step당 median **119K prefix / 875 append / 214 output** | ① |
| ★agentic | 〃 | *"tool calls cut LLM generation into multiple steps, making each step's output shorter than traditional reasoning workloads"* | ①·② |
| ★reasoning | MuxWise T1 | OpenThoughts in **311/709/4633**, out **684/8374/32k** = **0.085:1** | ① |
| reasoning | 〃 §4.3 | *"On OpenThoughts, the system spends most of the time in the decode phase"* | ④ |
| reasoning | ServeGen(NSDI'26, Alibaba 3.54B req) | *"reason lengths ... on average **4× longer than answer lengths**"* | ① |
| 프로덕션 | DeepSeek V3/R1(24h) | in **608B**(cache hit **342B**) / out **168B**; prefill 4노드 : decode **18노드** | ①·③④ |

★**"ShareGPT"는 단일 숫자가 아니다** — median prompt가 **226 ↔ 1730**(7.6×)로 갈린다.
원인은 **turn flattening 규약**: SGLang 로더(우리 것)는 **첫 2턴만** 쓰고, Sarathi는 라운드를
별도 요청으로 만든다. 우리는 추가로 `--sharegpt-context-len 4000` 필터가 걸려 있다.
⇒ **데이터셋 이름은 워크로드를 특정하지 못한다.**

## 3. 축 ① 좌표 지도 (우리 대비)

| 문헌 점 | in:out | 우리(1.44) 대비 |
|---|---|---|
| OpenThoughts | 0.085 | **17× decode 쪽** |
| MuxWise ShareGPT (226/195) | 1.16 | ★**거의 동일 — 유일한 동일 영역** |
| ★우리 G16 앵커 (341/237) | **1.44** | — |
| Sarathi sharegpt4 | 4.2 | 2.9× prefill |
| Azure-Conv | 5.5 | 3.8× |
| Splitwise conv | 7.9 | 5.5× |
| arxiv_summ | 34 | 24× |
| Mooncake conv | 35 | 24× |
| ★우리 he2 phase A (**3600/32**) | **112.5** | 78× |
| Azure-Code | 73 | 51× |
| LooGLE | 2000 | 1400× |

★**우리 격자는 1.44와 112.5 두 점만 밟았고 그 사이(4–35)가 완전 공백**인데, 문헌 워크로드
대다수가 정확히 거기 산다.
⚠️조사 초안은 he2 phase A를 2048/32(=64)로 적었으나 **메인 세션 확인 결과 `he2_bench.sbatch:35`
의 `LA=3600, OA=32` = 112.5**다. Azure-Code(73)와 "사실상 동일"이 아니라 **1.5× 더 prefill-heavy**다.

## 4. ★질문 4: "prefill·decode 동시 활성 시간 분율"

**직접 보고한 논문은 없다**(PD-mux 계열 7편 본문까지 조사: Bullet · MuxWise/Drift · Semi-PD ·
Nexus · DuetServe · POD-Attention · TaiChi).

**가장 값어치 있는 외부 앵커**: Agentic AI Workload Characteristics —
*"decode 91.0–98.6% of LLM time, prefill 1.4–9.0%"*. prefill이 활성이 아니면 동거가 불가능하므로
**1.4–9.0%가 동거의 상한**이다.
⚠️caveat: 그쪽은 prefix-cache ON·저경합(*"no-thrashing cache regime"* 명시), 우리는
`--disable-radix-cache`·포화 근처. **같은 눈금으로 나란히 쓰지 말고 방향만 인용.**

⚠️★**함정**: MuxWise의 **bubble ratio 7.7%**(= CUDA stream이 어떤 커널에도 점유되지 않은 시간)가
우리 d16 노출 7.7%와 **숫자가 우연히 같다.** 전혀 다른 양이며 **교차 인용 절대 금지.**

## 5. ★근거가 약한 것 (서사 vs 데이터)

| 주장 | 상태 |
|---|---|
| "워크로드가 decode 쪽으로 이동 중" | ⚠️**종단 측정 없음.** 유일한 종단 데이터(OpenRouter 100T 토큰)는 prompt **4×** vs completion **2.7×** ⇒ 축①로는 **오히려 prefill 쪽 1.5×**. **서사와 데이터가 반대** |
| "reasoning이 서빙을 decode-heavy로" | 데이터셋 수준 ✔ / **요청 수준 ✘**(TraceLab: tool-call이 CoT를 분할, step당 median out 214) |
| "agentic = long-prompt 워크로드" | ✘ **반증**(캐시 후 decode 91–98.6%) |
| OpenRouter 1.5K→6K | **app-mix 교락 치명적**(programming 11%→50%), 세그먼트 내 변화인지 분해 불가 |
| ServeGen "1.63×/1.46×" | ⚠️**기간 내 변동폭이지 추세 아님** |
| "18개월 전 대비 출력 10–50×" | **서사**(블로그). **인용 금지** |
| "prefill은 max batch서도 5% 미만" | 블로그, 출처 추적 불가. **인용 금지** |
| Mooncake "in:out ≈ 720" | 웹 요약 경유, **원문 대조 미완** |
| DeepSeek node-time 3.1:1 | ★**조사자 산술**(그들 주장 아님) |
| arXiv 2605.26297 · 2606.30560 · 2511.04791 · vLLM 블로그 | **날짜·판본 재검증 필요**(조사자 지식 컷오프 근접) |
| MuxWise 이름 | arXiv v1(2504.14489)은 **"Drift"**, ASPLOS'26판은 **"MuxWise"** — 판본 명시 필요 |

## 6. 주요 출처

Splitwise(ISCA'24) · DistServe(OSDI'24) · Sarathi-Serve(OSDI'24) · POD-Attention(ASPLOS'25) ·
MuxWise/Drift(ASPLOS'26, arXiv 2504.14489) · Bullet(ASPLOS'26) · DuetServe(arXiv 2511.04791) ·
Nexus(arXiv 2507.06608) · TaiChi(arXiv 2508.01989) · Mooncake(FAST'25) · ServeGen(NSDI'26) ·
Agentic AI Workload Characteristics(arXiv 2605.26297) · TraceLab(arXiv 2606.30560) ·
Preble(arXiv 2407.00023) · DeepSeek V3/R1 Inference System Overview · OpenRouter State of AI 2025
