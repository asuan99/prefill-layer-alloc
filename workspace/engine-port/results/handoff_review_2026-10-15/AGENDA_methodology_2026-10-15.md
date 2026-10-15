# 방법론 안건 상세 — n의 의미 · goodput 정의 · 지표 영향 요인 · 결정 안건 8개 (2026-10-15)

> 지위: main 트랙 검토 메모(측정 0, 새 판정 0). `REVIEW_handoff_prompt_2026-10-15.md` §4를 대체·확장한다.
> 근거는 정본·PREREG·코드를 읽은 결과다. [C]=문서/코드에 적힌 사실, [I]=이 메모의 추론.
> 경로 약어: `R/`=`workspace/engine-port/results/`, `S/`=`sglang_engine_dev/python/sglang/srt/`.

---

## 0. 한 줄 요약

- **goodput 정의는 이미 고정돼 있다**: 요청 술어 `TTFT ≤ k·floor(L) ∧ P95(요청 내부 ITL) ≤ 50 ms`(A′, 2026-10-04 사용자 승인). 판정 단위는 attainment 90% 용량이다.
  - 따라서 프롬프트의 "TTFT 4–15 s · mean TPOT"은 정본과 충돌한다. 새로 정할 것은 정의가 아니라 **혼합 길이 워크로드에 A′를 어떻게 적용하는가**다.
- **VESSL 이후 측정은 전부 "한 Job 안, seed(=boot) 4개, 고정 rate 또는 닫힌 burst"였다.** Job 수준 반복은 0회, 변화 trace도 0회다.
  - 지금까지의 n=4는 "이 GPU, 이 시점"에 대한 n이다. 운영 모델이 요구한 "n은 여러 Job으로"는 한 번도 실행되지 않았다.
- **단일 GPU fused arm은 VESSL에서 한 번도 돌지 않았다.** G0(PD-mux vs fused)는 이양 후 첫 fused 대조이므로 baseline 정의를 지금 정해야 한다.

---

## 1. VESSL 이후 실험들이 이 쟁점을 어떻게 다뤘나

| 캠페인 (날짜) | 부하·도착 | n과 검정 | 용량·cliff | SLO | 전력·클럭 | 상태 |
|---|---|---|---|---|---|---|
| λ0 (10-04) | random shape A/B, 고정 rate 사다리 | 최상단 rung n=4. boot·seed·시점 미분리 | 처리율 포화만 봄 (SLO 용량 아님) | 없음 | cap 노출 A 27%·B 49%, 귀속 금지(LVR-2) | 死因 0, 판정 0 |
| E2 sticky r1/r2 (10-04/06) | 고정 rate Poisson | seed 4개, arm짝 차분 bootstrap. **"4/4 = 부호검정 1/16 → p<0.05 금지"**(E2VR-5). Job 간 수준 +4% 이동 → Job 내 짝 차분만 인용(E2VR-3) | 48-구속 achieved라 시스템 용량 아님 | 없음 | — | 라벨 CONFIRMED, 정책 판정 아님 |
| N3 shape B r1–r4 (10-06~08) | random 8192/64, 절대 rate 10점, Poisson | rate당 seed 4개(셀당 1 boot). bootstrap 2000회, **"재표집 공간 35개" 공시**(N3VR-11) | 첫 실패 규칙(attainment<0.90), 정상성 S1–S3(S3=큐 성장), cliff zone | A′ (k×floor, ITL 50 ms) | 창별 cap 노출 11.5–47.4% 병기 | 라벨 CONFIRMED. **k=3 용량 셀이 cliff 구간 안(문턱과 1요청 차)** → 게이트 6 미닫힘 |
| N2 3-a PROFILE/EA (10-11~13) | 닫힌 burst(b개 동시 송신) | boot당 짝, 셀당 n=4. 라벨은 4/4 부호, bootstrap은 기술값. "64"는 짝 64개 | goodput 없음 | — | clk 100 ms, power 20 ms. **C 창 64/64 cap 노출 → ρ 크기 CONFOUNDED** | 死因 0 |
| E-1 cap (prereg만) | **변화 trace**(ShareGPT 200, 3↔12, 3라운드) | **블록=Job=노드, n≥4 Job** | — | HE0 술어 | clk 공시 | **미실행** |
| multigpu M3 (미제출) | 고정 rate HI/LO | 같은 boot 안 seed 4개. "boot 산포 미포함" 명시 | 큐 성장 g≥0.10/≤0.05 | 기술값 | GPU별 clk | 대칭 fused(`fused_sym`)를 1차 분모로 둔 유일한 설계 |
| opchar (10-12/13) | 격리 커널 | n=30 중앙값 | — | — | 74 SM 이상에서 cap 이벤트 | 특성화 전용 |

해석 규약으로 굳은 것 [C]:

- **게이트 2**: "정책 비교가 아니면 해당 없음"으로 일관되게 처리했다(`PREREG_N2_3A:170`, `RESULT_n3_B_r3:317`, E2 `:114`). 다만 "그 점을 정책 근거로 쓰는 것은 금지"를 함께 적는다.
- **Job 효과**: 같은 seed·같은 채점 경로에서도 Job(GPU) 효과가 공통 격자 36/36 셀에 한 방향으로 나타났다. 그래서 라벨 일치를 재현으로 쓰지 않고, 비교는 같은 Job 안에서만 한다(게이트 #262, `PROJECT_STATUS.md:101`).
- **기판 공변 검사**: 처치와 기판 변수가 창 단위로 공변하는지 먼저 검사한다(#268, N2 Job-EA 계기).
  - 클럭·전력 **수집**은 기본값이다(clk.csv, 10-08부터 전력 샘플러 opt-out).
  - 그 **분석**은 기본이 아니었다. N2에서는 결과가 나온 뒤에야 노출 공변이 발견됐다.

---

## 2. "n"이 의미하는 것 — 반복의 층위

### 2.1 층위 정의

한 측정값은 여러 층의 변동을 동시에 품는다. "n=4"는 **어느 층을 4번 반복했는가**를 말하지 않으면 뜻이 없다.

| 층위 | 단위 | 이 층을 반복하면 잡히는 변동 | 이 프로젝트에서 실제 수 |
|---|---|---|---|
| L0 요청 | 한 run 안의 요청 수백–수천 개 | 요청 길이·도착 간격 | 많음. **서로 독립 아님**(같은 대기열을 공유) → n으로 세면 의사반복 |
| L1 창/짝 | 시간 창, arm 짝 | 창 사이 부하 변동 | N2 64짝 (셀당 4) |
| L2 seed | 도착 순서·길이 표본 난수 | 워크로드 표본 변동 | 셀당 4 |
| L3 boot | 서버 재기동 | cudagraph 캡처, 메모리 배치, 커널 autotune, 초기 상태 | VESSL에서는 seed = boot (셀당 1 boot) |
| L4 Job | VESSL Job = 노드/GPU/드라이버/동거 부하/전력 환경 | GPU 개체 차이, 노드 열·전력 상태, 시점 | **항상 1** |
| L5 기판 | A100 SXM vs 다른 세대 | 아키텍처 | 1 (A100) |

### 2.2 층위가 주장 범위를 정한다

- **같은 Job 안 arm 짝 차분 n=4(L2/L3)**: "이 GPU, 이 시점에서 arm A가 B보다 낫다"까지만 말할 수 있다. 이것은 E2VR-3과 #262가 사실상 내린 결론이다.
- **"VESSL A100에서 A가 B보다 낫다"**: L4 반복이 있어야 한다. 근거는 두 가지다.
  - Job 간 +4% 수준 이동(E2)
  - 36/36 일방향 Job 효과(#262)
  - 즉 σ_Job이 3% headline 문턱(게이트 3)과 같은 크기다. Job 1개로는 효과와 Job을 구분할 수 없다.
- **"hybrid 모델 일반에서"**: 모델 축 반복이 필요하다. 이것은 G2의 몫이다.

### 2.3 작은 n의 통계적 한계 (계산 가능한 사실)

| 설계 | 가능한 최소 p | 의미 |
|---|---|---|
| 짝 4개, 부호검정 (단측) | 1/16 = 0.0625 | 4/4 같은 부호여도 p<0.05 불가 (E2VR-5) |
| 짝 5개, 부호/순열 (양측) | 2/32 = 0.0625 | 5/5여도 양측 p<0.05 불가 |
| 짝 6개, 부호 (양측) | 2/64 ≈ 0.031 | 6/6이면 도달 가능 |
| n=4 bootstrap | 서로 다른 재표집 C(7,4)=35개 | CI 끝점이 거칠다 (N3VR-11) |
| n=5 percentile bootstrap | 실제 coverage ≈ 0.84 (2026-08-06 정정) | 명목 95% CI가 아니다 |

⇒ 현재 규칙 "n≥4"는 **판정 가능성**을 보장하지 않는다. 그래서 VESSL 시기 판정은 전부 "4/4 부호 규칙"이라는 **사전등록된 결정 규칙**으로 내려졌고, p값은 쓰지 않았다. 이 방식 자체는 정직하다. 다만 결정 규칙의 오류율이 공시되지 않는다.

### 2.4 제안 (결정 안건 2)

- **(a) 2층 설계를 기본으로 한다.**
  - Job 안에서는 arm을 블록 무작위 순서로 섞고, seed는 4 이상으로 둔다.
  - Job은 **3개 이상** 반복하고, 노드(hostname·GPU UUID)를 공변량으로 기록한다.
  - 판정 = Job 내 짝 차분의 부호가 **모든 Job에서 일치** + Job 수준 효과의 t-CI(자유도 J−1)가 0을 배제. Job이 3개면 t 임계는 4.30이라 보수적이지만 정직하다.
- **(b)** 게이트 3의 문구를 "n≥4"에서 "**층위를 명시한 n**"으로 바꾼다. 예: "seed≥4 × Job≥3, 판정은 Job 수준".
- **(c)** 비용: G0를 Job 3개로 나누면 각 Job은 짧게 유지하고 셀은 묶는다. 운영 모델 원칙 3과 일치한다. GPU 사용량 계산은 설계 단계에서 제시한다.

---

## 3. goodput 정의 검토 — 이미 고정된 것과 열린 것

### 3.1 고정된 정의 [C]

`analyze.py:34-38`:

```
good_i     = [TTFT_i ≤ T_TTFT(L_i)] ∧ [P95(ITL_i) ≤ T_ITL]      # 요청 단위, ITL은 그 요청의 토큰 간격(첫 토큰 제외)
goodput    = Σ good_i / Σ_round duration_round                    # req/s, duration은 라운드 합산(게이트 7)
attainment = Σ good_i / N
용량(k)    = attainment(r,k) ≥ 0.90을 만족하는 등록 rate 중, 첫 실패 rate 바로 아래 (N3 첫 실패 규칙)
```

- ITL이 빈 요청(출력 ≤1 토큰, 실패 요청)은 `P95([])=NaN`이므로 **불합격**이다. 반면 legacy mean 술어는 0으로 채워 **합격**시킨다. 두 술어의 처리가 반대다.
- secondary: `TTFT ≤ T ∧ mean(ITL_i) ≤ T_ITL`.
- 출처와 이유: `EXPERIMENT_ROADMAP.md:1010-1011`(R1 재분석 때 "혼용 금지"). Granite 부호 반전 사례에서는 mean 술어 셀이 위반 0건이라 **goodput ≡ throughput 항등식**이 됐고, 그래서 강등 근거로 채택하지 않았다(`PROJECT_STATUS.md:6685-6697`).

### 3.2 TTFT SLO = A′ (2026-10-04) [C]

출처: `reports/slo_definition_rationale_2026-10-04.md`, `AUDIT_slo_method_2026-10-04.md`.

- `T_TTFT = k · floor(L)`, 결정점은 **k ∈ {3, 5, 10}**(Holm 보정)이고 k∈[1,25] 곡선은 기술 통계다.
- `floor(L)` = 같은 Job 첫 블록에서 무경합(108 SM 전부 prefill)으로 잰 단일 요청 TTFT 16개의 중앙값이다. 초 단위로 동결한다.
- T는 **처치가 아니라 판독 눈금**이다. static arm은 SLO를 읽지 않으므로 사다리로 채점해도 재스코어가 아니다(게이트 8 비위반). 정책 arm은 ITL 50 ms에 고정한다.
- 우세 판정 = 세 k점에서 부호가 같고 CI가 0을 배제.
- 금지 문안: "업계 기준", 임계값 없이 "사용자 SLO 만족", "길이 정규화로 길이 간 크기 비교".
- 절대값(DistServe 15 s, MLPerf 6 s, Bullet 12.3 s)은 **환산 좌표로만** 쓴다. `longcontext_trace_plan.md:195`의 "15 s Primary"는 A′로 대체됐다.
- 실측 예(N3 shape B, 8192/64): floor 0.907 s → T = 2.72 / 4.53 / 9.07 s. 프롬프트의 "4–15 s"가 이 사다리와 대략 겹치는 것은 우연이 아니다. 8k 입력의 floor × 5–15 정도다 [I].

### 3.3 열린 것 (결정 안건 3·4)

1. **혼합 길이 적용.** arXiv-Sum이나 Mooncake처럼 L_i가 요청마다 다를 때는 rationale이 이미 "요청별 k·floor(L_i), floor 곡선은 실측점에 맞춤(Etalon)"으로 방향을 정했다. 남은 결정은 세 가지다.
   - **floor 곡선 측정점**: arXiv-Sum 16k라면 1k–16k에 몇 점을 둘지. 제안 {1k, 2k, 4k, 8k, 12k, 16k} [I].
   - **곱셈형 `k·floor(L)` vs 가산형 `floor(L)(1+α)+β`** (`rationale:127-131` "재검토"). 곱셈형은 긴 요청에 절대 여유를 더 준다.
   - 128k까지 가는 Mooncake 꼬리의 floor(128k 단일 요청 프로브, main 결정 대기).
2. **ITL 50 ms는 "관행값이지 사용자 기준이 아니다"**(`rationale:146`).
   - arXiv-Sum(출력 ~200)에서는 요청당 ITL 표본 ~200개라 P95가 안정적이다.
   - 하지만 Mooncake toolagent(출력 p50 30)나 LooGLE(출력 ~15)에서는 요청당 표본이 적어 P95가 사실상 최댓값이 된다. 그러면 단 한 번의 prefill stall이 그 요청을 떨어뜨린다 [I]. **짧은 출력 워크로드에는 ITL 술어의 통계적 성질이 다르다**는 점을 사전등록에 명시할 필요가 있다.
3. **실패·abort 처리 불일치.** analyze.py는 ITL이 비면 불합격, N3는 errors>0이면 셀 무효다. G0 사전등록에서 하나로 정한다. 제안: 셀 무효 + 건수 공시 [I].
4. **보고 단위.** 판정 = attainment 90% 용량(N3 방식)이고, goodput req/s는 고정 rate에서의 보조값이다. 프롬프트의 "rate별 goodput 곡선"은 attainment 곡선 + 용량점으로 바꿔 그린다.
5. **프롬프트 문구 교체**: "goodput@SLO, TTFT SLO 4–15 s, TPOT SLO" → "A′ 사다리(k=3/5/10) 용량 + ITL p95 50 ms 술어, 절대 초 값은 환산 좌표로 병기".

---

## 4. 시스템 지표(TTFT · ITL/TPOT · throughput)에 영향을 주는 요인

> ★정정(같은 날): span과 ITL의 인과 관계, 그리고 "stream" 표기(= green context에 묶인 stream)는 `DESIGN_span_and_integration_2026-10-15.md` §2가 우선한다.
> span 하나의 정지 크기는 입력 길이와 대체로 무관하다(≈65,536 token-layer 예산). 길이가 늘리는 것은 정지 횟수와, 그 정지에 걸리는 토큰 비율이다.

### 4.1 구조: 무엇이 언제 정해지나

```
[사전(정적) — 서빙 전에 값이 확정]
  M 모델 구조 ──┐  (층 종류·수, KV B/token, state B/seq, 파라미터)
  dtype ────────┤  (mamba_ssm_dtype)
  서버 플래그 ──┤  (mem_fraction_static, max_running_requests, max_prefill_tokens, split_forward_token_budget,
                │   cudagraph bs 목록, radix, backend; PD-mux가 chunk=-1·overlap off 강제)
  SM 분할 표 ───┤  (manual_divisions / divide_sm)
  하드웨어 ─────┘  (108 SM, 전력 한도 — VESSL에서 조작 불가)
        │ 정적 요인은 "가능한 런타임 상태의 범위"를 정한다
        ▼
[런타임 — 요청 흐름과 엔진 상태가 매 iteration 결정]
  도착(U) → 대기열 → admission → prefill 배치 형성 → span 분할 실행 ─┐→ TTFT
                                                              │ (stall)
  running batch(bs, ctx 분포 = U×E) → cudagraph 패딩 → decode step ─┴→ ITL/TPOT
  분할 선택(전환 시점) · 동거 간섭 · 클럭 하락 · retract ───────────────→ 둘 다
  완료 수 / 합산 duration ──────────────────────────────────────────→ throughput
```

**모델 구조는 런타임 전에 다 계산할 수 있다.**

- Nano-9B-v2: 56층 = Mamba 27 + attn 4 + MLP 25, KV 16 KiB/token, state 136.9 MiB/요청 (fp32).
- 그래서 "요청 하나가 얼마의 메모리·FLOP을 쓰는가"는 roofline으로 사전 계산된다(`costmodel_a100/`).
- 반면 **같은 모델·같은 플래그에서도 지표를 바꾸는 것은 런타임 상태**다. 이것이 cost model의 E×U 부분이다.

### 4.2 정적 요인 (요약; 런타임 상태의 범위를 정함)

| 요인 | 무엇을 정하나 | 지표 영향 | 확정 정도 |
|---|---|---|---|
| 층 구성·KV·state | 요청당 메모리, decode에서 attn ∝ bs·ctx / SSM ∝ bs | ITL·용량 | CM-1 형태 일부. s_attn ≠ s_ssm은 **미입증** |
| `mamba_ssm_dtype` | state 크기 → 슬롯 수 | admission → TTFT | 미측정. **명시 고정 필요** |
| `max_running_requests` | **= mamba 슬롯 수**(radix OFF, `kv_cache_mixin.py:224-229` 확인). decode bs 상한 | bs↑ → throughput↑·ITL↑, 너무 낮으면 W_queue↑ | CM-5 코드 사실 |
| `mem_fraction_static` | KV 풀 토큰 수 | admission(KV 막힘)·retract | CM-5 형태 |
| `max_prefill_tokens` (기본 16384) | prefill 배치 크기. **첫 요청은 초과해도 무조건 수용** | S_prefill↑ vs W_queue↓, stall 길이 | 미연구 |
| `split_forward_token_budget` (65536) | span 하나의 층 수 = 65536 // 배치 토큰 (`multiplexing_mixin.py:1415-1421` 확인) | stall 단위·경계 빈도 | 82개 config 전부 기본값, 스윕 0 |
| cudagraph bs 목록 | decode bs를 다음 값으로 패딩, 목록 밖이면 eager | ITL | 운영점 ON, **패딩·hit 미기록** |
| PD-mux 강제 플래그 | chunk=-1, overlap off | un-chunked prefill → ITL 꼬리 | §1-24 기판 전제 |
| SM 분할 표 | D(decode SM) ↔ 108−D(prefill SM) | D↑ → T_dec↓, S_prefill↑, stall↑ | C2 레버 존재(2.36–2.91×) |
| 하드웨어·전력 | 클럭 | 전 단계 시간 | VESSL에서 cap 노출 확인, `-pl` 불가 |

**arXiv-Sum에서의 구체 값** (Nano-9B-v2, 56층) [I, 코드 식 대입]:

- 입력 7k 요청 1개짜리 prefill 배치 → span당 65536 // 7000 = 9층 → **약 7 span**
- 16k 배치(요청 2개 또는 긴 요청 1개) → 4층/span → **14 span**
- `max_prefill_tokens` 16384 → 평균 7.1k 기준 prefill 배치당 요청 ~2개
- prefill은 **한 번에 한 배치만**, 끝나야 다음을 받는다(`event_loop_pdmux`) → 긴 요청이 대기열 앞을 막는다(HOL)

### 4.3 런타임 요인 — 지표에 닿는 지점

**TTFT 분해** [C] (`ttft_measurement_definition_review_2026-09-28.md:106-110`, anchor A9):
`TTFT = L_client + L_ingress + W_queue + S_prefill + L_egress`

| 런타임 요인 | 닿는 항 | 기전 | 관측 |
|---|---|---|---|
| 도착 burst·대기열 길이 | W_queue | 큐 적체 | `queue_len`, `oldest_prefill_age_ms` |
| admission 차단 (cap = 슬롯 / KV / R2 latch) | W_queue | **entanglement 사슬**: decode 굶김 → ITL↑ → batch 정체 → 슬롯 안 풀림 → admission 차단 → TTFT 폭발 (§1-4, TTFT 차이의 ≥95%가 대기열) | `r2_admission_*`. **거절 사유(KV/cap/slot) 미기록** |
| prefill 배치 형성 (1개씩, 탐욕적) | W_queue + S_prefill | 긴 요청 HOL, 배치 크기 → span 수 | `n_new`, `extend_tokens`(PHASE_EVENTS 켠 때만) |
| span 실행 (108−D SM) + span 사이 decode 대기 + 경계 drain | S_prefill | prefill SM이 줄면 길어짐 | `prefill_span_start/end`는 **배치 단위**. span별 GPU 시간 없음 |
| retract | W_queue 오염 | SSM은 state 재계산 → 전량 재-prefill | 분해 표에서 제외, 건수만. **텔레메트리 이벤트 없음** |

**ITL 분해** [C] (§1-24, CM-4):
`ITL_token = T_dec(D, bs_pad, ctx) × ρ(동거) × (클럭 보정) + [prefill로 인한 정지]`

| 런타임 요인 | 기전 | 확정 정도 | 관측 |
|---|---|---|---|
| decode bs → 패딩 bs | T_dec ↑ (bs) | CM-1 | `decode_iteration.bs`. 패딩 bs 미기록 |
| ctx 분포 (U×E) | attn 항 ∝ Σctx | §1-5 방향만 | 32 sync마다 표본뿐. **iteration 단위 없음** |
| 분할 D (전환 시점에 선택) | D↑ → T_dec↓ | C2 | `decode_iteration.prefill_sms/decode_sms` |
| prefill로 인한 정지 | un-chunked span이 decode 반복을 막음. **ITL 꼬리의 지배항**, 크기 D에 단조↑ (d24 168 ms → d92 866 ms) | §1-24 구조 확정, 크기는 seed 1개 | `t_sync − t_launch`에 prefill CPU launch가 섞임 → **오염** |
| 동거 간섭 ρ | prefill과 동시에 돈 decode step이 더 김 | CM-4 존재 확인, 크기는 전력과 교락 | `prefill_in_flight` |
| 클럭·전력 | 전 단계 감속 | Job-EA 64/64 노출 | `clk.csv` (epoch 시간축 ↔ 텔레메트리 monotonic **정렬 필요**) |

**throughput**은 대략 min(prefill 처리 능력 @ 108−D SM, decode 처리 능력 @ D SM, 슬롯/KV 상한)으로 묶인다 [I].

- arXiv-Sum은 prefill 쪽이 묶이는 구간이다. 유도값: 전 GPU 포화 ~1.2–1.5 req/s, PD-mux에서는 그보다 낮음.
- decode 쪽 동시 요청은 ~9개로 작다(유도).
- ⇒ arXiv-Sum에서 PD-mux 용량을 정하는 것은 주로 **D 선택과 prefill 정지**이고, hybrid decode floor가 아니다.

### 4.4 관측 공백 — G0 전에 계측을 보강할지 (결정 안건 9)

우선순위 순 [I]:

1. **span 단위 prefill GPU 시간과 층 범위.** S_prefill과 stall을 분해하려면 필요하다.
2. **decode 커널만의 GPU 시간(CUDA event).** 지금 `t_launch → t_sync`에는 prefill launch가 섞여 CM-1과 CM-4를 가를 수 없다.
   - 프롬프트의 "CUDA event latency"는 여기에 정확히 쓰인다. headline이 아니라 **분해용 계측**이다.
3. **cudagraph 패딩 bs·hit 여부.**
4. **iteration 단위 Σctx / ctx p95.**
5. **admission 거절 사유와 retract 이벤트.** CM-5 X-6(무엇이 먼저 묶이나)의 직접 증거가 된다.
6. **legacy 자동 분할 전환 이벤트.**
7. **clk.csv ↔ 텔레메트리 시간축 정렬.** F-1 전력 분리에 필요하다.

이 계측 보강은 엔진 패치다. engine-porter의 correctness gate와 텔레메트리 스키마 개정이 선행된다.

---

## 5. 결정 안건 (각각 구체화)

| # | 안건 | 지금까지의 처리 | 선택지 | 권고 |
|---|---|---|---|---|
| 1 | **게이트 2 적용 범위** | VESSL 측정 전부 "정책 비교 아님 → 해당 없음 + 정책 근거 사용 금지"로 처리. 변화 trace 실행 0회 | (a) 현 관행 명문화 (b) G0에도 변화 trace 추가 | **(a)**. 단 G0 결과 문장에 "정적 시스템 비교, 정상상태" 범위를 명시. 실 trace(Mooncake) 재생 시 rate 축 = **시간축 압축 배율**로 정의하고, 정상성 S1–S3를 적용하지 않는 별도 판정 규칙을 5-a 사전등록에서 정함 |
| 2 | **n의 층위와 검정** | 실제는 Job 내 seed=boot 4개, Job 수준 n=1. 판정 = 4/4 부호 규칙 | (a) 현행 유지 (b) seed≥4 × Job≥3 2층 설계 (c) seed 6개 단일 Job | **(b)**. "VESSL A100에서"를 주장하려면 Job 반복이 필수(σ_Job ≈ headline 문턱). 비용은 Job을 짧게 나눠 흡수 |
| 3 | **혼합 길이 TTFT SLO** | A′ 고정. 혼합 길이 적용은 방향만 있음 | floor 곡선 측정점 · 곱셈형 vs 가산형 · 128k 프로브 | 요청별 k·floor(L_i), floor 6점 실측(같은 Job 첫 블록), 곱셈형 유지(rationale 기본). 4–15 s는 환산 좌표 |
| 4 | **ITL 술어** | p95 50 ms 고정, mean은 secondary | 50 ms 유지 / 사다리 {40,50,60,65} 기술 병기 / 짧은 출력 워크로드 별도 규약 | 50 ms 유지 + static arm 사다리 병기(N3 관행). 출력 짧은 워크로드는 요청당 ITL 표본 수를 공시 |
| 5 | **fused baseline 정의** | 단일 GPU에서는 VESSL 측정 0회. 정본은 "미조율 fused", T2(조율 공간 소진) 주장 금지. multigpu만 `fused_sym` | (a) 기본 fused만 (b) 기본 + 3-플래그 분해 arm(fused + chunk −1 + overlap off) (c) (b) + 사전등록된 유한 E-D 스윕 | **(c)**. headline은 "조율한 fused 대비"로 두고, 기본 fused는 선행 연구 정렬용. E-D 노브: chunk {2048, 4096}, `max_running_requests`, `num-continuous-decode-steps`, `prefill-max-requests`. 스윕은 같은 SLO(A′)로 튜닝(게이트 8) |
| 6 | **headline 위계** | 서빙 술어 중심 | — | 1위 attainment 90% 용량(k=3/5/10) → 2위 고정 rate의 TTFT/ITL 분포(p50/p95/p99) → 3위 CUDA event 분해(기전) |
| 7 | **예측-측정 대조** | 없음 | roofline B_max를 무엇과 비교할지 | roofline B_max(KV/state 용량)와 엔진 실효 상한(부팅 로그의 KV 풀 토큰, `max_running_requests` = 슬롯)을 **별개 열**로 기록. 층 비중 예측은 서빙에서 측정 불가 → opchar 격리 커널과만 대조 |
| 8 | **전력·클럭** | 수집 기본, 분석 비기본. #268은 사후에 생김 | — | 사전등록에 **arm × 창 cap 노출 공변 검사**를 판정 전 단계로 넣는다(#268 이행). 공변이 발견되면 그 셀은 크기 인용 금지 |
| 9 | **계측 보강** | span·decode GPU 시간 미분리 | G0 전에 보강 / G0는 현 계측, 기전은 후속 | 1·2·5번(4.4)은 G0 전에 보강 권고. PD-mux 이득의 기전("prefill 정지")을 G0에서 바로 분해할 수 있다 |

### 안건 간 의존

- 2(Job 반복)는 G0 Job 규모를 정한다.
- 5(fused 정의)는 G0 arm 수를 정한다.
- 9(계측)는 엔진 패치 일정을 정한다.
- ⇒ 이 셋이 G0 사전등록의 크기를 함께 결정한다.
- 3·4는 floor 측정 블록 하나로 같은 Job 안에서 해결된다.
