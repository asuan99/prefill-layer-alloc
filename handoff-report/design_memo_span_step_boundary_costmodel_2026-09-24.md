# 설계 메모 — prefill span / decode step 경계에서의 cost-model 기반 분할 결정 (2026-09-24, rev2)

> **지위**: 설계 메모(사전등록 아님). **GPU 지출 0 · 새 측정 0 · 새 성능 판정 0.** 연구 결론·Claim 등급·게이트 불변.
> 사용자 제안(2026-09-24): "whole-phase 분할이 아니라 prefill은 layer-wise, decode는 step-wise로 재분할 경계를 두고,
> 경계에서 요청 상황 조건에 따라 자원을 재분배하거나 유지한다. phase/layer별로 어떤 차이를 보여야 할지 정리."
>
> **rev2(같은 날)**: rev1에 대한 claims-auditor 적대 감사(read-only, 코드 기준 커밋 `a9cd8dd` 소스 = dev tree 바이트 동일) 결과
> 교정 R1–R10을 본문에 반영하고 누락 술어 P0–P12를 §4에 추가했다. 감사 판정: Q1 코드 사실 PLAUSIBLE(3개 하위 주장 REFUTED),
> Q2 "Claim E만 미시험" REFUTED, Q3 인용 PLAUSIBLE(범위 표기 누락 5건), Q4 술어 NOT-YET-SUPPORTED, Q5 교정 10건 필수.
> ⇒ **이 메모는 아직 사전등록 근거로 쓸 수 없다.** §5에 P0 부분 실측(아카이브 로그 재집계, GPU 0)을 추가했다.

## 1. 코드 사실 (`workspace/engine-port/src/multiplex/multiplexing_mixin.py`)

- **R1** `event_loop_pdmux`(legacy·true-dual 공통)의 루프 1회 = decode 1 step ∥ prefill 1 span **발행**. 이는 CPU 루프 경계이며
  GPU 완료 경계가 아니다(legacy는 decode만 매 반복 synchronize[`:1494`], prefill 스트림은 최종 span까지 비동기 — 중간 전환은
  `prefill_stream.synchronize()`로 발행된 span 전부를 드레인한다). `PDMUX_LA_COORD=1`(`event_loop_pdmux_coord`)에서는
  성립하지 않는다. span 크기 `forward_count = max(1, split_forward_token_budget // extend_num_tokens)`(`:1416–1427`)이고
  `split_forward_token_budget=65536`(전 yml 79개, `pdmux_context.py:20`) 하에서 `extend_num_tokens ≤ 65536/num_hidden_layers`인
  prefill은 span 1개(= layer-wise 경계 없음)다. Nano-9B-v2는 `num_hidden_layers=56`(HF config, 리비전 `dc0661c8…`) ⇒ 문턱 ≈1170 토큰.
- **R2** v7(`PDMUX_SLO_SCHED`)은 prefill이 비행 중인 동안에만(`:1312–1317`) 매 span `_slo_decide_idx()`를 호출하고(dwell 3,
  `:1129–1142`), R2는 매 span `_r2_decide_idx()`를 호출하나 `CoarseGrainedController`가 `max(4 iter, 100 ms)` cadence·
  `max(8 step, 200 ms)` dwell로 재평가를 제한한다(`controller.py:112–115,168–179`; env 노브 없음). prefill이 없는 decode step은
  결정 지점이 아니며, prefill 종료 후에는 이벤트 경로(`adjust_stream_groups`의 `manual_divisions` bs-임계값, `:1175–1182`)가
  컨트롤러 선택을 덮어쓴다(sticky+fixed 예외, `:1168–1175`) ⇒ 제안의 '유지'는 prefill 생애주기를 넘어 유지되지 않는다.
  (정본 관측: decode-busy 가중 시간의 86–89%가 prefill 유휴 구간 — E2C-21, shape A.)
- **R3** decode CUDA graph는 `f"{stream_idx}_{bs}"`별로 전 stream group에 캡처(dev tree `cuda_graph_runner.py:798,811–816`,
  재생 키 `:681`) ⇒ step 경계 전환은 **decode** cudagraph와 양립. split prefill은 eager(`model_runner.py:2866`)다.
  파티션 수 × bs 수만큼 graph 메모리가 늘어난다(미측정). true-dual worker 스레드의 전역 stream idx 읽기 안전성은 미검증.
- **R4** `PDMUX_SLO_SPAN_TYPE`(`:1430–1437`): prefill span을 attn/ssm type-run 경계에서 자름 — `la_coord_windows`는
  `zamba2.py`에만 있어 NemotronH에서는 무동작(silent no-op).

## 2. 정본 대조

| | 死 per-layer-type (postmortem 시도 C·F·H, decode-side) | 제안 = 현 엔진 경계 구조 |
|---|---|---|
| 전환 위치 | 한 forward 안 attn↔mamba 경계마다(decode 1 step이 19 윈도우) | prefill span 발행 경계 = decode step 경계(prefill 비행 중에만) |
| 나누는 것 | layer 종류별 SM 수 | prefill:decode 비율(파티션 1개) |
| cudagraph | 비양립 → eager 강제 → 운영점 진입 불가 | decode는 양립, prefill은 원래 eager |
| **R5** 실행 결과 | coordinated TPOT 42→124 ms(OPT 85 ms; 드레인은 갭의 ~47%, 나머지는 오버랩 손실+cudagraph 비양립 — 전환 비용 단독 수치 아님) | 전환 비용 미측정: 운영점 `s ≤ 0.04 ms`(CONSENSUS §1 신규 행 34)는 **컨트롤러 off·agnostic 경로·green→green 0건·decode_bs 1–23**의 상한이므로 이 제안의 컨트롤러 구동·중간-prefill·green→green 전환에 **이식 금지** |

- **R6** 죽은 형태에는 decode-side(C·F·H)뿐 아니라 prefill-side(D·E, lever 부재 — no-cudagraph micro)가 있고, prefill
  현재-layer-type으로 prefill:decode 비율을 span 경계에서 옮기는 **PF(`PDMUX_LA_COORD_PF`, `reports/research_arc.md:192` #9b /
  `:216` E5)는 서빙이나 no-cudagraph·n=1·변수 동시 변경으로 生死 판정 불가**다(`reports/policy_comparison.md:106`). 따라서
  결정이 **현재 span의 layer type의 함수**이면 PF 계열(정본상 layer-type 정책, 全형태 死 쪽)이고, **남은 phase 전체의 집계
  조성만의 함수**여야 postmortem §3의 생존 형태(whole-phase floor)다. 구현된 Claim E(`profile.py:1–6`)는 조성을 쓰지 않고
  full-model decode 곡선을 보간한다(`attention_layers`/`ssm_layers`는 메타데이터) ⇒ **'조성 기반 cost model'과 prefill-span 시간
  모델(A1)은 Claim E의 확장이며 별도 사전등록 대상이다.**

**걸리는 기존 결과**
1. **HE0**: 이 경계 구조 위의 reactive 규칙(SLO-aware v7·binding-first·gate)은 best decode-heavy static을 못 넘음(n≥4,
   관대·tight SLO). 원인 = positioning + entanglement, switch 비용 아님(CONSENSUS §1 항목17). **R10** HE0이 확정하는 것은
   single-worker·SM-split·reactive 계열뿐(§1-17 스코프) — 달성 가능한 천장은 별개 명제.
2. **R7** type-aware span sizing은 net-negative 관측(TTFT 2–4×↑; SUPERSEDED 문서 `policy_comparison.md`, no-cudagraph·Zamba2 한정,
   `system_vs_engine_vs_sim.md:129` 분류 = substrate-tinged)이며 NemotronH에서는 해당 플래그가 무동작이다.
3. **R8** Step D `PDMUX_SLO_LFF` = HD0(n=3, underpowered), Step F `sat_predict`도 시험됨(n 미기재) — 둘 다 single-worker·구 기판/빌드.
   CONSENSUS §1 항목17 각주: "offline 모델-프로파일 기반 decode-floor 예측(Claim E)만 미시험".

⇒ 제안에서 미검증으로 남는 것: (i) 구현된 Claim E(full-model decode-floor 보간, B6 true-dual)와 (ii) 그 **확장**인 조성 기반
cost model(별도 사전등록), (iii) PF 계열(현재-span type 조건)은 판정 불가 상태로 남아 있으나 정본상 layer-type 정책 쪽이다.
"layer-type 런타임 정책의 부활"이 아니라 "모델 프로파일 기반 step-경계 분할 제어"로 서술하되, 결정 입력이 현재-span type이면 그 서술은 거짓이 된다.

## 3. 보여야 할 차이 (앞이 불성립이면 뒤는 무의미)

### A. prefill (span 단위)
- **A1 span 시간 예측력** — **R9** [no-cudagraph micro · `--attention-backend triton` + Zamba2-2.7B 한정 · n_indep=1 · Nemotron-H 계열
  이전 금지] Diff A는 L 따라 크게 열리고, Diff B는 L=256 밴드 [1.24, 1.34]·L=512 1.198·L≥1024 ≈1.0(정책 단위 `R_policy`≈1.03,
  CONSENSUS §1-3). ⇒ layer type의 역할은 "SM을 몇 개"가 아니라 "남은 prefill 시간 예측". ★감사: **단일 모델에선 식별 불가** —
  조성은 모델 상수라 "조성 포함 vs 미포함"은 모델별 프로파일 유무의 재매개화일 뿐이다. **모델 hold-out**(조성이 다른 ≥2–3개 모델로
  적합, 미사용 모델 예측)이 있어야 식별된다. prefill span 수준은 부분 식별 가능하나 §5대로 짧은 입력은 span이 1개로 퇴화한다.
- **A2 intra-prefill 재결정 가치** — ★감사: **서술대로는 검정 불가**. v7(매 span)과 이벤트 경로는 규칙과 cadence가 **동시에** 다르다
  (교락 #10). R2 cadence는 하드코딩(`controller.py:112–115`)이라 같은 규칙에서 cadence만 바꾸려면 엔진 패치가 필요하다. 또
  prefill이 대부분 1 span이면 도메인이 비어 있다(§5, N3 위험) ⇒ P0 선행.

### B. decode (step 단위)
- **B1 decode floor의 조성 의존 이동**: attn decode ∝ ctx, SSM decode O(1) ⇒ floor(ctx, bs)의 이동 형태가 모델 조성에 좌우.
  ⚠ 기존 "knee 16→44→108→108"·"attn 비중 5%→79%"·"ctx256 1.1× SM-free"는 **인용 금지** [CS-OK: 금지 사실을 적기 위한 인용]
  (계측 결함 3건, CONSENSUS §1 항목5 각주), no-cudagraph 값이며 운영점에선 HE2가 decode non-binding 관측(§1-5 문구, HE2 수치
  자체는 인용 금지) ⇒ **VESSL cudagraph-ON 재측정 필수**. A1과 같은 이유로 조성 효과 주장에는 모델 hold-out이 필요하다.
- **B2 (교체)** 예측 위치의 goodput이 VESSL 실측 best static의 paired CI 안(위치 일치가 아니라 성능 기준; d44 등 KISTI 값 대입 금지).
  **성공해도 이것은 offline static 선택 주장이지 step-경계 동적 제어 주장이 아니다**(HE0: 모든 수정이 static으로 수렴). B2는
  positioning만 다루며 HE0 기전의 다른 절반인 **entanglement**(decode 굶김→ITL↑→batch 정체→admission 차단)는 P10이 다룬다.

### C. 경계·정책
- **C1 전환 비용**: 직접(드레인) + 간접(드레인 중 반대 스트림 bubble). 0.04 ms 대입 금지 — P12로 새 부류를 직접 측정.
- **C2 유지/전환 분포**: `switch_count`·split 체류 분포·dwell 병기(방법론 게이트 5).
- **C3 (교체, R10)** Claim E 채택 규칙대로 B6(true-dual hybrid) vs B1(global static)·B5(true-dual generic) 같은 job 내 교차배치,
  변화 trace는 VESSL 실측 λ\*의 분수로 정의(KISTI 'rate 3↔12' 이식 금지 — Nano-9B에선 λ\*(A)≈3.05·λ\*(B)≈0.696 req/s[KISTI, n=2]라
  rate 12는 ≈4× 과부하), 독립 job ≥2 × n≥4, SLO별 재튜닝, headline = effect ≥3% AND paired CI 하한 >0.
- **C4 (교체)** Step D/F와의 차별화는 cross-build 비교가 된다 ⇒ 공정한 대조는 **같은 job 내 B6 vs B5**.

## 4. 추가 술어 P0–P12 (감사 제시, 역사 번호는 괄호)

- **P0 도메인 존재**: prefill당 span 수 분포 + decode-active 시간 중 prefill-in-flight 비율(residency), VESSL 실측·워크로드 고정.
  대부분 1 span이거나 residency가 낮으면 A2·B·C 전부 퇴화(N3). → §5에 KISTI 아카이브 기반 부분 결과.
- **P1 realized vs target**: telemetry의 실현 split만 인정(Stage 0/λ0 교훈). decode-only fallback(`adjust_stream_groups:1188–1200`)·
  sticky 여부 등록.
- **P2 아키텍처 선언**: single-worker(HE0 도메인) vs true-dual(Claim E 도메인). 둘을 동시에 바꾼 비교는 해석 불가(#10).
- **P3 용량 먼저**: VESSL에서 모델·shape별 λ\* 측정 후 변화 trace를 λ\* 분수로 정의. TTFT≈SLO metric cliff 회피(#5, 게이트 6).
- **P4 SLO 재튜닝**: cost-model 정책의 margin/confidence와 static 비교군 모두 SLO별 재튜닝, 기존 run 재스코어 금지(#6, 게이트 8).
- **P5 n의 독립성**: 한 job 내 다중 boot ≠ 독립 run(인용정지 B3C-3). arm은 job 내 교차배치 + ≥2 job 복제(model≡node≡job 앨리어스, #3).
- **P6** 라운드 duration 합산(#4), goodput 술어(TTFT ∧ 요청내부 ITL p95), switch_count·dwell·p50/p95/p99(게이트 4·5·7).
- **P7 운영점 핀**: 전 arm 실현 로그에서 decode cudagraph-ON·prefill eager·NemotronH backend=flashinfer·동일 commit/manifest SHA·
  seed·trace hash. `SPAN_TYPE`·`LA_COORD`는 OFF로 등록(NemotronH에서 SPAN_TYPE 무동작, #8).
- **P8 관측자 효과**: telemetry는 무료 관측자가 아니다(CONSENSUS §3 항목61/게이트 #41). 특성화·적합 run과 전 arm의 telemetry 구성 동일.
- **P9 프로파일 hold-out**: cost model 적합 trace/워크로드와 평가 trace 분리. 조성 주장엔 모델 hold-out 추가(A1).
- **P10 entanglement 지표**: B6·B5 arm에서 예측 floor 미만 체류 비율, ITL→running-batch→admission 차단 연쇄.
- **P11 상금 크기 선측정**: B2(per-workload oracle static) − B1 갭. 3% 미만이면 Claim E는 원리상 headline 불가(N3). B8이 천장.
- **P12 새 부류 전환 비용**: green→green·중간-prefill 드레인(대기 span 포함)을 직접 측정. 0.04 ms 대입 금지.
- **기판**: VESSL 수치를 KISTI 결론과 섞지 않음 + 노드 단독 점유 여부 등록(`vessl_execution_plan_2026-09-24.md` §1).

## 5. P0 부분 실측 — prefill당 span 수 (KISTI 아카이브 재집계, GPU 0, 2026-09-24)

**방법**: λ0 job 908623(KISTI A100, NemotronH Nano-9B-v2, `pdmux_homog5.yml`, budget 65536)의 서버 로그 `srv_*.log`에서
`Prefill batch, #new-seq, #new-token` 줄을 모두 읽고(#new-token < 64인 부팅/웜업 프로브 제외), 엔진 공식 그대로
`spans = ceil(56 / max(1, 65536 // new_token))`을 적용했다. `#new-token` = `extend_num_tokens`로 가정(캐시 hit 0 워크로드).

| cell | prefill 배치 | seq/배치 평균(최대) | 토큰/배치 p50(최대) | span=1 비율 | span 분포 |
|---|---:|---|---|---:|---|
| a_r0 | 662 | 1.01 (2) | 256 (512) | **100%** | {1: 662} |
| a_r1 | 1245 | 1.02 (3) | 256 (768) | **100%** | {1: 1245} |
| a_r2 | 394 | 1.04 (2) | 256 (512) | **100%** | {1: 394} |
| a_r3 | 378 | 1.08 (2) | 256 (512) | **100%** | {1: 378} |
| a_r4 | 361 | 1.14 (2) | 256 (512) | **100%** | {1: 361} |
| a_r4_s2 | 306 | 1.34 (4) | 256 (1024) | **100%** | {1: 306} |
| b_r0 | 79 | 1.14 (2) | 8192 (16384) | 0% | {7: 68, 14: 11} |
| b_r1 | 90 | 1.33 (2) | 8192 (16384) | 0% | {7: 60, 14: 30} |
| b_r2 | 82 | 1.83 (2) | 16384 (16384) | 0% | {7: 14, 14: 68} |
| b_r3 | 111 | 1.89 (2) | 16384 (16384) | 0% | {7: 12, 14: 99} |
| b_r3_s2 | 111 | 1.89 (2) | 16384 (16384) | 0% | {7: 12, 14: 99} |

**읽는 법 (판정 아님)**
- **shape A(입력 256)**: 관측된 모든 prefill 배치(최대 1024 토큰)가 문턱 ≈1170 미만 ⇒ **prefill 내부 layer-wise 경계가 한 번도 없다**.
  이 shape에서 "prefill layer-wise 재결정"(A2)과 prefill-span 시간 모델(A1)의 도메인은 **공집합**이다. 경계는 prefill 배치 사이와
  decode step뿐이다.
- **shape B(입력 8192)**: 모든 배치가 7 또는 14 span ⇒ 중간-prefill 결정 지점이 존재한다(배치당 6 또는 13개).
- 따라서 제안 구조의 prefill 측 의미는 **긴 입력 워크로드에 한정**된다. 짧은 입력에서 layer-wise 경계를 만들려면 budget을 낮춰야
  하는데, 그것은 span 재진입 비용이라는 새 변수를 들이며(type-aware span sizing의 TTFT 2–4×↑가 같은 기전의 선례[R7 범위 한정]) 별도 측정 대상이다.
- ⚠ 한계: (i) KISTI 기판·λ0 설정·합성 고정길이 shape 두 개뿐 — VESSL/다른 trace로 일반화 금지; (ii) 배치 크기는 스케줄러 산물이라
  부하에 따라 변한다(표의 seq/배치가 rate에 따라 증가); (iii) **residency(P0 후반부)는 이 재집계에 없다** — decode-active 시간 중
  prefill-in-flight 비율은 telemetry 시간가중 재집계가 필요하며, 정본 E2C-21(shape A에서 decode-busy 가중 시간의 86–89%가
  prefill 유휴)이 방향을 보여 준다.

## 6. 순서 (rev2)
0. P0 완성(residency 재집계, CPU) + P11 상금 크기 확인(기존 자료로 가능한 범위, CPU).
1. VESSL 특성화 측정(GPU 소량, 정책 주장 아님): B1 decode step-SM 곡선(cudagraph-ON, ctx×bs), P12 전환 비용·bubble, (긴 입력 한정) A1 span-SM 곡선.
2. cost model 적합 + hold-out 예측 오차(CPU): 모델 hold-out 포함(P9).
3. 정책 캠페인: 사전등록 + 규칙층 감사 후 C3(B6 vs B1·B5).

기판 주의: 1–3의 모든 수치는 VESSL A100 기판 수치이며 KISTI 결론과 섞지 않는다(CLAUDE.md 게이트 1 연장).

## 7. P0 후반부 — residency·decode-only bs·prefill 중 조건 변화 (KISTI 아카이브 재집계, GPU 0, 2026-09-24)

**자료·규약**: λ0 job 908623 `lam0_908623/tel_*.jsonl`(KISTI A100, `PDMUX_R2_POLICY=fixed` D44, sticky OFF, probe 없음 ⇒ `probe_end=None`).
술어는 `e2_realized_mix.py`의 등록 리터럴 재사용: snapshot = `runtime_snapshot` ∧ `phase≠startup`, decode-busy =
`decode_running_batch_size>0`, prefill-in-flight = `prefill_active_batch_size>0`. 가중 3종: **cnt**(균등), **time**(다음 스냅숏까지 간격,
무상한), **iter**(`decode_iterations` 증가분, 왼쪽 busy). residency = decode-busy 중 prefill-in-flight 비율. ★정본 규약상
(E2C-8′) 세 가중이 어긋나는 열은 **범위로만** 서술하고 단일값 인용 금지 — 아래 shape B는 cnt↔time 차가 크다(비균등 표본 격자).

| cell | busy n | residency % cnt / time / iter | decode-only bs p50 / p90 / max (iter 가중) | decode-only 구간 stream idx (iter) | prefill 에피소드 n · 지속 p50 | 에피소드당 decode-bs 변화 평균 · 변화≥1 비율 |
|---|---:|---|---|---|---|---|
| a_r0 | 2677 | 2.9 / 3.1 / 2.9 | 8 / 12 / 20 | idx4 ≈98% | 85 · <1 스냅숏 | 0.02 · 2.4% |
| a_r1 | 2667 | 3.7 / 4.0 / 3.8 | 16 / 23 / 35 | idx4 ≈97% | 103 · <1 | 0.06 · 4.9% |
| a_r2 | 610 | 3.9 / 5.3 / 4.0 | 18 / 48 / 48 | idx4 ≈96% | 34 · <1 | 0.03 · 2.9% |
| a_r3 | 559 | 5.0 / 7.4 / 5.1 | 8 / 48 / 48 | idx4 ≈96% | 35 · <1 | 0.09 · 8.6% |
| a_r4 | 555 | 5.8 / 9.3 / 5.8 | 7 / 48 / 48 | idx4 ≈97% | 33 · <1 | 0.24 · 18.2% |
| a_r4_s2 | 557 | 6.3 / 10.5 / 6.4 | 5 / 48 / 48 | idx4 ≈98% | 36 · <1 | 0.08 · 8.3% |
| b_r0 | 313 | 43.1 / 66.2 / 45.6 | 1 / 1 / 2 | idx4 ≈92% | 55 · 0.83 s | 0.18 · 14.5% |
| b_r1 | 356 | 66.9 / 85.0 / 69.7 | 1 / 1 / 2 | idx4 ≈99% | 34 · 0.86 s | 0.53 · 32.4% |
| b_r2 | 324 | 87.0 / 96.3 / 88.6 | 1 / 1 / 2 | idx4 ≈94% | 12 · 0.84 s | 0.25 · 8.3% |
| b_r3 | 441 | 90.7 / 97.6 / 92.3 | 1 / 2 / 2 | idx4 ≈97% | 11 · 0.84 s | 0.18 · 9.1% |
| b_r3_s2 | 441 | 90.7 / 97.5 / 92.1 | 1 / 1 / 2 | idx4 100% | 11 · 0.83 s | 0.18 · 9.1% |

(idx4 = `(0,108)` 비분할, idx2 = D44. 에피소드 = prefill-in-flight 스냅숏의 최대 연속 구간 — 연달은 prefill이 합쳐질 수 있고 스냅숏
격자가 비균등이라 **짧은 prefill은 스냅숏 0–1개로만 보인다**[shape A 지속 p50 = 0 s는 "측정 불가 수준으로 짧다"는 뜻].)

**읽는 법 (판정 아님, KISTI 기판·합성 고정길이 shape 2개·fixed D44 한정)**
- **shape A**: decode-busy 시간의 **약 90–97%가 prefill 유휴(decode-only)** — 정본 E2C-21(86–89%, sticky 대조 문맥)과 같은 방향
  (규약이 달라 수치 대조는 하지 않음). 그 구간의 decode는 거의 전부 `(0,108)`에서 bs p50 5–18(p90 최대 48 = `max_running`)로 돈다.
  ⇒ sticky/'유지'를 켜면 **decode 반복의 ~90% 이상이 108→44 SM으로 바뀌고**, 이득(전환 제거)은 셀당 prefill 에피소드 33–103회에서만 생긴다.
  손익분기 `idle_share × Δstep × N_iter < N_prefill × (전환비용+spike)`에서 우변 횟수가 좌변 반복 수의 약 1/100 이하라, sticky가
  이기려면 **Δstep(44 vs 108, bs 5–48, cudagraph-ON)이 전환비용+spike의 약 1% 이하**여야 한다 — Δstep 미측정이라 예측일 뿐이다.
- **shape B**: residency가 높다(범위 43–98%, 가중에 따라 크게 다름) — decode는 bs 1–2로 작고 prefill(에피소드 ~0.84 s)이 거의 항상 돈다.
  **prefill 1회 동안 decode 조건(bs)이 바뀌는 경우는 에피소드의 8–32%, 평균 0.18–0.53회**뿐. ⇒ span 경계 재결정의 기회가 드물고
  decode 수요가 작아, 이 shape에서의 결정은 사실상 정적(prefill-heavy)이다.
- ⇒ 두 합성 shape는 제안 구조가 겨냥하는 상황(긴 prefill이 도는 **동안** decode 수요가 크게 변함)을 **어느 쪽도 만들지 않는다**:
  A는 prefill이 짧고 드물며, B는 decode가 작다. H-Policy가 명시한 도메인("시간적으로 상보적인 near-saturation workload")에 해당하는
  **혼합 trace(긴 prefill + 많은 동시 decode, 부하 변화)**가 있어야 A2·B·C가 비퇴화한다 — 이 trace의 설계·용량(λ\*) 측정이 선행 과제.
- 한계: telemetry 표본 격자가 비균등(prefill-in-flight 표본 과소 가능 — `PDMUX_TRACE_FORCE_PREFILL` 미사용)이라 residency 절대값은
  가중 규약 의존. 스크립트는 세션 scratchpad `p0_residency.py`(저장소 미등록) — 인용 전 저장소 등록·selftest 필요.
