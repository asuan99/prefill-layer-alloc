# 정량 지표·논의·부하 의존성 종합 설명

작성일: 2026-06-22 · 목적: 본 프로젝트에서 진행된 논의를 **정량 지표 기반**으로 정리하고, 각 metric의 의미와 **부하 축(모델 크기·배치·시퀀스 길이)이 그 지표·추세 차이를 어떻게 만드는가**를 설명.
관련 보고서: [layer_aware_benefit](layer_aware_benefit_report.md) · [7B 검증](layer_aware_7b_verification.md) · [vllm_validation](vllm_validation.md) · [queue_simulator_design](queue_simulator_design.md)

---

## Part 1 — 정량 지표 사전 (각 metric이 무엇을 의미하나)

| metric | 정의 | 무엇을 말하나 | 본 프로젝트 대표값 |
|---|---|---|---|
| **TTFT** (Time-To-First-Token) | 도착→첫 토큰까지 = prefill 지연 + 큐 대기 | prefill 응답성. 부하↑면 큐로 폭증 | vLLM 2.7b: 182ms(RR2)→6375ms(포화) |
| **TPOT / ITL / TBT** (per-token decode 지연) | 토큰 사이 간격 = 한 decode step 시간 | **SLO의 핵심**. 대화형 품질 결정 | vLLM 2.7b: 11.9ms(저부하)→80ms(포화) |
| **throughput** (tok/s) | 시스템 총 출력률 | 용량 | sim 2.7b co=752, 1.2b=1302, 7b=375 |
| **goodput@SLO** = thr × SLO_attain | **SLO 만족한** 토큰만 센 throughput | *유일하게 옳은* serving 지표(빠르지만 SLO 위반=무가치) | 2.7b layer_aware 1268 vs agnostic 629 |
| **decode floor** (SM) | decode를 solo 속도 근처로 유지하는 *최소 SM* | **"decode 보호 비용"**. 작을수록 싸게 보호→prefill에 SM 환원 여지↑ | attn b1: SLM 54 / 7B **68** |
| **HBM BW util %** (flat_region) | 커널이 쓰는 HBM 대역폭 비율 | **메모리-바운드 여부**. 높으면 SM 더 줘도 안 빨라짐 | attn-decode 40–70% / **ssm-decode ~0.01%** |
| **fusion_saving** = (prefill+decode)−fused | prefill·decode를 한 forward로 fuse 시 절약 | ≈0 ⇒ fused가 분리 실행 대비 공짜 아님 | 실측 ≈ 0 |
| **decode inflation %** | 부분 SM에서 decode가 solo 대비 느려지는 정도 | 분할 시 decode 손해 | 7B attn@54SM: +67~97% |
| **prefill SM-sensitivity** | SM을 더 줄 때 prefill 가속비 | **layer_aware 이득의 원천**(환원 효과) | 2.7b 6.2×(14→108SM) / 7B 1.02× |
| **la/agnostic ratio** | layer_aware ÷ agnostic goodput | layer-type 인지의 순이득 | 1.2b 1.37× / 2.7b 2.0× / 7b 1.06× |

---

## Part 2 — 어떤 논의들이 있었나 (정량 근거 + 판정)

### D1. layer-type 공간 *분할*이 co-schedule을 이기나 → **死**
green_ctx(정적 SM 분할)가 two_stream(동적 공유)을 **1576셀 중 0번** 이김(max attain 0.987/0.996). *근거 지표*: 자원 비대칭(attn은 compute, ssm은 memory)이 존재해도 그게 **lever가 아니다** — 분할은 양쪽을 다 굶길 뿐. compute⊥memory(§3.2)라 한 자원 나눠도 다른 축 경합 안 풂.

### D2. prefill/decode *PD-mux*의 이득 구간 → **{SLO < decode ITL}**
*근거 지표*: decode ITL이 SLO보다 크면(=decode가 SLO 위반) decode를 보호(예약)해 살릴 수 있음. ITL<SLO면 이미 만족이라 예약은 prefill만 굶겨 손해. 그래서 **goodput@SLO**로 봐야 보임(raw throughput으론 안 보임 — 이게 결론을 6번 뒤집은 원인).

### D3. fused(vLLM 기본) vs 분할 → **fusion_saving ≈ 0**(실측)
*근거 지표*: 진짜 fused 커널 측정 시 (prefill_only+decode_only)−fused ≈ 0. mixer/KV traffic이 지배해 weight 공유 절약 미미 ⇒ "fused가 공짜로 decode를 태운다"는 낙관 기각.

### D4. GQA가 decode 비용을 16× 가른다? → **반증(vLLM 실측)**
*근거 지표*: **per-attn-layer** decode 비율은 zamba2(no-GQA)/falcon(GQA) = **10.6×(B8)~21.7×(B256)** — 맞음. 그러나 **full-model** decode ITL 비율은 **1.3×**(ssm 레이어가 지배). vLLM 실측이 1.2–1.6×로 확인 ⇒ **decode ITL은 GQA/KV-read가 아니라 모델 전체 weight·ssm-state memory traffic이 결정.** "16× wide/narrow 이득 구간"은 single-layer artifact.

### D5. layer-type-aware *예약*은 PD-mux를 개선하나 → **生(SLM)**
*근거 지표*: 비싼 attn-decode 레이어만 예약, 싼 ssm 레이어는 prefill 환원 ⇒ 2.7b서 **la/agnostic 2.0× goodput**(throughput 1268 vs 629, TTFT 절반, decode ITL 43.5ms로 SLO 내). 단 temporal 하이브리드·SLM 한정.

### D6. 크기 추세 → **역-U자, 2.7B peak**
*근거 지표*: la/agnostic = 1.2b **1.37×** → 2.7b **2.0×** → 7b **1.06×**. 단조 아님(Part 3-A에서 이유).

---

## Part 3 — 부하 축이 지표·추세 차이를 만드는 메커니즘

### A. 모델 크기 (1.2B → 2.7B → 7B)

| 지표 | 1.2B | 2.7B | 7B | 추세 |
|---|--|--|--|--|
| attn-decode floor(SM, b1) | 54 | 54 | **68** | 크기↑ → 보호 비용↑ |
| prefill SM-sensitivity | 1.41×(@54) | **6.2×**(@14) | **1.02×** | 中에서 최대 |
| full-model decode step(b8) | — | ~28ms | **~51ms** | 크기↑ → decode 지배↑ |
| co throughput | 1302 | 752 | 375 | 크기↑ → 용량↓ |
| **la/agnostic** | 1.37× | **2.0×** | 1.06× | **역-U(2.7B peak)** |

**왜 역-U인가** — layer_aware 이득 = (싼 attn 보호 가능) × (ssm서 prefill 환원 효과). 두 전제가 크기에 반대로 움직인다:
- **7B(1.06×)**: attn-decode가 54 SM로 미포화(+67~97%, floor 68↑)라 *싸게 보호 못 함*; prefill이 측정 구간서 SM-무감각(1.02×)이라 *환원 이득 없음*; 게다가 decode-step 지배(51ms)라 prefill lever leverage 작음 → **lever 부재.**
- **1.2B(1.37×)**: 보호는 싸지만(floor 54), 모델이 작아 agnostic이 prefill을 *덜 굶김*(goodput agnostic 1376 > co 1302) → *환원할 여지 자체가 작음* → 이득 축소.
- **2.7B(2.0×, sweet-spot)**: 보호 싸고(저배치 floor 54) + agnostic이 prefill을 강하게 굶김(629 < co 752, 14 SM까지 → 6.2× 회수) → **환원 효과 최대.**

### B. 배치 (decode_batch B = 동시 decode 요청 수)

| 지표 | B=1 | B=8 | B=64 | B=256 | 메커니즘 |
|---|--|--|--|--|--|
| attn-decode floor(2.7b, SM) | 54 | 68 | 108 | 108 | 배치↑ → 보호에 GPU 전체 필요 → **분할 여지 소멸** |
| per-layer attn-decode(2.7b, ms) | 0.082 | 0.59 | 4.16 | 15.7 | KV read × 배치 (메모리-바운드) |
| **GQA 비율**(zamba2/falcon, attn) | 1.7× | 10.6× | 21.1× | 21.7× | 배치↑ → no-GQA KV 폭증 |
| full-model decode ITL(2.7b, ms) | 24 | 29 | 74 | 253 | 위 합산 |

**왜** — decode는 **메모리-바운드**(토큰마다 KV+weight를 HBM서 읽음). 배치↑면 (a)KV read가 배치배 늘어 비용↑, (b)포화에 필요한 SM↑(floor가 b16서 108=전체) → **고배치에선 decode 보호=GPU 독점**이라 prefill과 공유 불가 → 분할/예약 무력. GQA는 토큰당 KV를 줄이므로 배치↑서 no-GQA와 격차가 벌어진다(1.7→21×). 반면 **ssm-decode는 BW util ~0.01%**(O(1) state)라 배치에 거의 무감각 → 싼 채로 유지.

### C. 시퀀스 길이 (context_len = KV 길이)

| 지표 | ctx 1024 | 4096 | 16384 | 메커니즘 |
|---|--|--|--|--|
| **ssm-decode floor**(2.7b, b1, SM) | 27 | 14 | 27 | **거의 불변** — SSM state는 O(1) |
| ssm-decode BW util | ~0.002% | ~0.002% | ~0.003% | context 무관(고정 크기 state) |
| ssm-decode floor(b16) | 68 | 68 | 94 | 고배치서만 약간↑ |
| attn-decode KV read | ∝ ctx | ∝ ctx | ∝ ctx | **O(L)** — KV가 길이배 |

**왜 — 이게 hybrid가 존재하는 근본 비대칭이다.** Attention decode는 길이 L의 KV 전체를 읽어 **O(L)**(context↑ → 비용↑ → ITL↑ → 이득 구간 {SLO<ITL} 넓어짐). SSM decode는 고정 크기 recurrent state만 갱신해 **O(1)**(context 무관, BW ~0.01%). 즉:
- **긴 context = attn-decode가 더 비싸짐** → PD-mux/예약의 이득 구간이 *원리상* 넓어짐.
- 단 **attn-decode floor는 이미 108(전체)로 포화**(고배치)라, context가 길어도 "보호 가능 여부"는 안 바뀜 — KV가 커지면 오히려 메모리 벽에 더 막힘.
- **그래서 layer_aware의 핵심**: ssm 레이어가 context-무관하게 싸다는 점(O(1))을 이용해 그 시간을 prefill에 환원. context가 길수록 attn KV는 커지지만 ssm은 그대로라 layer-type 비대칭이 더 또렷해진다.

> **부하 축 요약:** decode 비용/보호 비용을 키우는 건 **배치(KV read × B)**와 **context(attn O(L))**이고, 보호를 *싸게* 만드는 건 **작은 모델(floor 낮음)**과 **ssm 레이어(O(1), BW~0%)**이다. layer_aware 이득은 이 둘이 만나는 **SLM·저~중배치·temporal 하이브리드**에서 최대(2.7B), 모델이 크거나(7B: 보호 비싸짐+prefill 무감각) 너무 작으면(1.2B: 환원 여지↓) 약해진다.
