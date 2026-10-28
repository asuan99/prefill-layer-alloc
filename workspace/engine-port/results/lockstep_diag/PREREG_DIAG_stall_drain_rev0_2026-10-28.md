# 사전등록 rev0.2 (DIAG) — lockstep 정지 기전·동적 arm drain 대기 진단 Job, 2026-10-28

> **봉인 상태: 미봉인 초안 rev0.2**(하네스층 감사 r1 `AUDIT_harness_diag_r1_2026-10-28.md` 死因 K1·K2, MAJOR M1–M8, MINOR 수리 + 결정 49; 수리 위치 대조표 §13). 계측 필드 이름은 감사서 표기를 쓰며 **engine-porter 커밋 후 실제 이름에 맞춘다**(코드는 이 문서가 건드리지 않는다). 문서 작업(CPU)만 했다. 측정 0, GPU 0, `launch.sh` 실행 0(dry-run 포함), 성능·정책 판정 0. 하네스 코드는 쓰지 않았다(감사 뒤).
> 지위: **진단 Job**. 정책 주장이 아니다. 이 Job의 어떤 라벨도 goodput·정책 순위·HE0 판정으로 인용하지 않는다. Nano HE0 핵심 셀의 정책 비교는 별도 PREREG `../cap_char/PREREG_HD_HE0cell_rev0_2026-10-28.md`가 맡는다(결정 40). **rev0.1(2026-10-28): 결정 46으로 Nano 단일 모델 진단 Job이다**(Zamba 배제, 결정 45 대체).
> 근거: 진단 `DIAG_lockstep_stall_share_2026-10-27.md` §5 · 감사 `../anchor_audit/VERDICT_lockstep_diag_claims_r1_2026-10-27.md`("해소 실험"·P1–P5 허용/금지 문안) · 사용자 결정 39–45(`tracks/main.md`; 결정 46 = Zamba 배제·결정 45 대체) · 형식 `../cap_char/PREREG_COSTMODEL_TERMS_MEAS_rev3_2026-10-20.md`(봉인 S1′ 형식).
> 수치 라벨: [측정](출처 명시) · [유도](식 대입·시간 모형, 측정 아님) · [코드] · [결정] · [제안](이 문서가 고른 값, 봉인 시 사용자 확인) · [미정](사용자 결정 또는 선행 작업 몫) · **[봉인 시 기입]**.

## 0. 무엇을 재고 무엇을 하지 않는가

**묻는 것(진단 §5, 감사 "해소 실험")**
1. 정지가 있는 반복에서 **host 차단 시간 H**가 (a) prefill chunk의 GPU 시간(lockstep-GPU), (b) chunk의 host 발행 시간(CPU-launch), (c) decode GPU 시간 자체(동거 간섭) 중 무엇을 따르는가.
2. 동적 arm(reactive bind+GATE)이 전환마다 치르는 **drain**(`adjust` 경로의 `prefill_stream.synchronize()` + `decode_stream.synchronize()`)의 누적 대기가 TTFT·ITL에서 차지하는 몫. 정지가 HE0를 바꿀 수 있는 유일한 경로가 "arm별로 다른 drain"이라는 감사 판정(P2)의 열린 구멍을 겨냥한다. 전환 비용 s ≤ 0.04 ms(§1-34)는 이 경로로 옮겨 쓸 수 없다.
3. PHASE 계측의 관측자 비용(PHASE OFF 대칭 arm).

**하지 않는 것**
- 정책 판정, goodput 비교, HE0·C2·layer-type 死의 등급 변경, "정지가 HE0의 원인/무관" 귀속(감사 금지).
- 모델·backend 교락(게이트 #83)의 해소. 이 Job은 Nano(flashinfer) 단일 모델이라 모델·backend 교락은 **시험 대상이 아니다.** Nano 결과를 다른 모델·backend(특히 KISTI Zamba2·triton)로 옮기지 않는다.
- KISTI 수치와의 비교·결합(기판 다름: 감사 P1 금지).

## 1. 선행 조건 (전부 충족돼야 봉인)

| # | 선행 조건 | 상태 | 막는 것 |
|---|---|---|---|
| P-1 | **계측 커밋 해시 확정 + GPU 정확성 gate PASS.** 계측 = 진단 §5 (i) `prefill_chunk` 이벤트(`span_seq, split_index, forward_count, t_issue_start, t_issue_end` host + prefill stream CUDA event 쌍 GPU ms, query 비차단 회수), (ii) `drain` 이벤트(사유, 두 synchronize 각각의 host ms, **진행 중 prefill chunk 수**), (iii) `prefill_span_start`의 `rids`, (iv) I-4 `decode_fwd_gpu_ms`를 같은 boot에서 함께, (v) client 토큰 절대 CLOCK_MONOTONIC. 구현 중(engine-porter), 기본 OFF env. | **[봉인 시 기입: 계측 커밋 해시 / gate 통과값 / ENGINE_DIGEST]** | 봉인 |
| P-2 | 계측 env: 진단 계측 총괄 env = `PDMUX_PHASE_DIAG`(기본 OFF, `PDMUX_PHASE_EVENTS=1` 필수), 클라이언트 절대 단조시각 = `PDMUX_LOADGEN_ABS_MONO`(래퍼 `abs_mono_loadgen.py`). chunk·drain·rids·launch/GPU-bound 플래그의 세부 필드 이름은 감사서 표기 — `prefill_chunk{t_issue_start, t_issue_end, forward_count, gpu_ms, start_done_at_issue_start, end_done_at_issue_end}`, `drain{reason, stream_idx_from, stream_idx_to, host_ms, in_flight_chunks}`, `prefill_span_start.rids` — **engine-porter 커밋 후 실제 이름에 맞춘다**. | 부분 확정 | 봉인 |
| P-3 | gate 내용(engine-porter 설계, 이 문서는 요구만 둔다): 계측 ON/OFF에서 greedy 출력 토큰 동일성, 계측이 새 동기화(`synchronize`·`.item()`·블로킹 `query` 대기)를 만들지 않음, CUDA event 쌍의 시각 정합. 통과가 PASS이고 "특성화 종결"은 PASS가 아니다(결정 22 선례). | 미정 | 봉인 |
| P-6 | 동적 arm에 쓰는 Nano 튜닝 구성과 Nano 용량 rate는 정책 PREREG(`PREREG_HD_HE0cell_rev0`)의 Job-CAP1·Job-TUNE 결과(addendum)에서 온다. D-3은 그 뒤에만 제출한다. | 대기 | D-3 |
| P-7 | 하네스(신규 구현 항목 §9)는 감사 뒤 작성. | 대기 | 봉인 |
| P-8 | **엔진 digest 단일 최종 트리 핀(감사 M8)**: 진단 Job과 정책 Job(`PREREG_HD_HE0cell_rev0.2` Q-7)은 **같은 최종 트리 하나**의 digest로 핀한다. K1 수리(launch/GPU-bound 플래그)가 digest를 한 번 더 바꾸므로 값은 `PENDING_FINAL_TREE`(자리표시, 전달 후 기입). | 대기 | 봉인 |
| P-9 | CPU 재현 로그: `test_phase_diag`·`test_engine_gate_diag`는 이 머신에서 `sgl_kernel` 부재로 import 실패 — 감사자 CPU 재현 불가. 봉인 전 `sgl_kernel`이 있는 환경의 두 시험·변이 실행 로그를 첨부한다. | 대기 | 봉인 |

## 2. 측정 대상 arm

### 2-1. 정지 관측 격자 (저부하 고정 rate, PHASE ON + decode probe ON)

| 축 | Nano-9B-v2-Base (flashinfer) |
|---|---|
| 모델 revision | `dc0661c…`(비용 항 PREREG와 동일, 봉인 시 전체 해시 기입) |
| D (decode SM) = prefill SM | D {24, 44, 92} = prefill {84, 64, 16} (A100 108 SM, 실현 SM은 realized로 검증 — §6) |
| 프롬프트 길이 | {2k, 8k} |
| split 예산(`split_forward_token_budget`) | {8192, 65536} |
| 셀 수 | 3 × 2 × 2 = 12 |

- 길이는 고정(분산 없음): 입력 2k 또는 8k 토큰, **출력 512 고정**(`ignore_eos`) [제안 — decode batch를 span 동안 비우지 않기 위한 값]. radix OFF, cudagraph ON, `--mem-fraction-static` 0.6(Nano, 결정 13 승계), max-running 48, backend·cap·KV capacity는 모델별 고정.
- **저부하 고정 rate**: 도착 Poisson, 입력 2k **0.5 req/s·60 요청**, 8k **0.1 req/s·40 요청** [감사 r1 권고 채택: O-DX4 닫힘]. 근거 = 정지 관측이 목적이고 goodput 판정이 아니므로 용량 이하(Nano 8k shape B SLO 용량 0.15–0.22 req/s [측정, N3 허용 문안 구간; 인용은 구간·병기와 함께만]의 아래)에 둔다. 큐 길이는 이벤트에 기록한다 [코드, `queue_len`].
- 셀당 span ≥ 40개(8k) / ≥ 60개(2k) + 꼬리 30 s. 반복 = **seed 4개**(O-DX2 닫힘: n = 4; 트레이스 해시·workload seed·server seed가 seed마다 고정, 같은 Job 안 모든 셀에 동일 시퀀스).
- **Job 분할(감사 권고)**: D-1을 프롬프트 길이별 두 Job으로 나눈다 — D-1a(2k), D-1b(8k), **Job마다 OFF boot 1개**. static 모드는 정책 PREREG와 같은 값(결정 49): **전 Job static arm 비sticky**(`PDMUX_R2_POLICY=fixed`, `PDMUX_STICKY_PARTITION` 없음); "D"는 동거 중 목표값이며 실현 할당이 아니다.
- **PHASE OFF 대칭 arm**(관측자 비용, D-1은 **기술값**, 감사 M5): Job마다 1 boot, 해당 Job의 (D44, 예산 65536) 구성에서 `PDMUX_PHASE_EVENTS`·`PDMUX_PHASE_DIAG`·decode probe·admission probe 전부 OFF(OFF가 끄는 범위 등록; 그 외 telemetry 경로는 동일), 프롬프트는 그 Job의 길이. 관측 지표는 client 측(TTFT, 토큰 간격 p50/p95/p99)만. 같은 Job 안의 ON 셀과 짝으로만 비교한다(다른 Job 비교 금지).

### 2-2. 동적 arm drain 블록 (Nano, HE0형 변화 trace 핵심 셀)

- 트레이스·rate·SLO·rate 환산·cliff 규칙은 `PREREG_HD_HE0cell_rev0` §3(정책 PREREG)을 **그대로 가져온다**: KISTI `sharegpt_vary_bench.sbatch` 구조(구간당 200 요청, LO/HI 교대 3 라운드, 구간 사이 drain), rate는 Nano λ\*\_best-static 비례(LO/HI = 1/4). 이 문서에서 다시 정의하지 않는다 — 정의가 갈라지면 정책 PREREG가 이긴다.
- arm 3개, 같은 Job(D-3) 안에서 블록 무작위 순서, 모두 seed 4개:
  1. **reactive bind+GATE**, 계측 ON — 정책 PREREG의 Job-TUNE이 고른 Nano 재튜닝 구성.
  2. **best decode-heavy static** = **D44 사전 지정, 비sticky**, 계측 ON [감사 r1 권고 채택: O-DX3 닫힘]. 이 블록의 목적은 drain 대기 측정이며 static 선정이 아니다.
  3. **PHASE OFF 대칭 arm**: bind+GATE, 계측 전부 OFF(같은 구성·seed). 관측자 비용 라벨은 **이 블록(seed별 재부팅)에서만** 낸다(M5).
  4. **"PHASE\_EVENTS ON · DIAG OFF" arm**(감사 M8, D2 공백): bind+GATE, `PDMUX_PHASE_EVENTS=1` + `PDMUX_PHASE_DIAG` 없음. ON(DIAG)과 OFF 사이 관측자 비용을 PHASE 계층과 DIAG 계층으로 가른다.
  static arm의 OFF 짝은 두지 않는다(잃는 범위 §4 X-3).
- 동일 boot 조건: mem-fraction·max-running·cudagraph·config yml·`split_forward_token_budget`가 세 arm에서 같다. 동적 arm만 `PDMUX_SLO_*` env가 추가된다.

### 2-3. 셀 순서·무작위화

각 Job 안에서 boot 순서를 블록 무작위(seed는 계획 해시에서 파생, 봉인 시 기입)로 한다. boot 안에서는 프롬프트 길이 순서를 seed마다 교대. warm-up(60 s) / correctness(첫 요청 출력 토큰 길이 검사) / benchmark phase를 분리하고, warm-up 구간은 계측은 켜되 판별량에서 제외한다.

## 3. 계측과 판별량

### 3-1. 이벤트와 시각 기준

진단 §1 표 승계(`prefill_span_start/_end`, `decode_iteration`, 시각 = host CLOCK_MONOTONIC) + P-1 신규 이벤트. **GPU 시각은 CUDA event 쌍의 경과 ms로만 쓰고 host 시각과 섞어 한 축에 놓지 않는다.** `prefill_span_end`는 GPU 완료 이후 최대 1 루프 반복 늦게 관측된다(진단 §1). 관측자 공시(MINOR): chunk begin poll/emit 코드(`multiplexing_mixin.py` chunk 발행 구간)가 `[t_launch, t_sync]` 안에 있어 **관측자 비용이 H에 포함된다** — chunk GPU 시간은 span 구간 차가 아니라 event 쌍으로 잰다(진단 §5 공백 2).

### 3-2. 판별량 A — 정지 반복의 host 차단 분해 [동등성 여유를 데이터 전에 등록; rev0.2 K1·K2·M4]

**왜 바꿨나(감사 K1)**: 루프 순서가 decode 발행(`t_launch`) → chunk 발행(`t_issue_start…t_issue_end`) → decode sync(`t_sync`)라서 `[t_issue_start, t_issue_end] ⊂ [t_launch, t_sync]`, 즉 H = t_sync − t_launch ≥ X2(chunk host 발행 시간)가 **정의상** 성립하고 H ≈ max(X2 + ε, X3 − δ)다. H를 X2에 회귀하면 기전과 무관하게 b2 = 1이 나오므로(엔진 회계 항등식) rev0.1의 `FOLLOWS_X2`는 판별량이 아니었다. 따라서 X2는 **회계 항**으로만 보고하고 판별은 아래 잔차로 한다.

- 변수(반복 i = 한 `decode_iteration`): H\_i = t\_sync − t\_launch; X1 = 그 반복에 진행 중인 chunk의 GPU 시간(event 쌍); X2 = t\_issue\_end − t\_issue\_start; X3 = `decode_fwd_gpu_ms`; F = `forward_count`.
- **판별량(등록)**: R\_i = t\_sync − t\_issue\_end(마지막 chunk 발행이 끝난 뒤의 host 대기), Z\_i = H\_i − max(X2, X3). **R과 Z 각각을 X1·X3·F에 동시 OLS**(절편 포함). X2는 설명변수에 넣지 않고 "회계 항"으로 H 중 X2가 차지하는 몫(H − R)만 기술값으로 보고한다.
- **반복 분류(비차단 플래그 2개, 계측이 `query()`로 기록)**: `start_done_at_issue_start`(chunk 시작 event가 발행 시작 시점에 이미 완료 = prefill stream이 비어 있었음), `end_done_at_issue_end`(끝 event가 발행 종료 시점에 이미 완료 = GPU가 host를 기다림). `end_done_at_issue_end` = 참이면 **LAUNCH\_BOUND**(host 발행이 병목), 거짓이면 **GPU\_BOUND**(발행 뒤에도 GPU가 남음). 분류별로 따로 회귀하고 분류별 반복 수를 보고한다. 풀 전체 라벨은 두 분류가 일치할 때만 낸다.
- **정지 반복 집합 S**: 간격(연속 `t_sync` 차)이 같은 boot·같은 bs의 비-span 반복 중앙값의 3배 초과(진단 §3-4 정의, 민감도 2배·5배). **주 추정은 span과 겹친 모든 반복**, S는 부차.
- **짝짓기 키(M4)**: X1 = 같은 `decode_iter_seq`의 chunk 레코드; X3 = host bracket 종료 ≤ 그 반복의 t\_launch인 **직전** probe 레코드. 조인 실패율을 보고하고 5% 초과면 `UNRESOLVED_JOIN`.
- 추정: (cell, seed) 군집 표준오차 + 군집 수준 t 구간(percentile bootstrap 금지). 공선성 VIF > 5 → `UNRESOLVED_COLLINEAR`.
- **동등성 여유 [제안, 봉인 시 확인]**: 기울기 b의 95% 구간이 [0.7, 1.3]에 들면 `FOLLOWS`, [−0.05, +0.05]에 들면 `FLAT`, 그 외 `PARTIAL`. 근거 공시(MINOR): 이 여유는 **관례**이며 해상도 참조(보관 N3·N2·T3의 span 시간 기울기 0.0015, 95% 구간 [0.0005, 0.0027] [측정, 진단 §3-1, 그 run들 한정])의 약 20배로 느슨하다.
- **라벨 우선순위(K2, 고정 순서)**: ① `STALL_ABSENT` — S가 반복의 1e-4 미만이고 최대 정지 간격 < 5 × 중앙값(N3 Job 4 선례 [측정]) → **기울기 판별을 하지 않고 주 추정(R·Z·회계 몫)을 보고만 한다**, ② `UNRESOLVED_FEW` — S 30개 미만 또는 군집 < 6 또는 seed < 3, ③ `UNRESOLVED_COLLINEAR`/`UNRESOLVED_JOIN`, ④ 기울기 라벨 — `STALL_FOLLOWS_CHUNK_GPU`(R·Z가 X1에 FOLLOWS ∧ X3에 FLAT), `STALL_FOLLOWS_DECODE_GPU`(X3 FOLLOWS ∧ X1 FLAT), `STALL_HOST_ISSUE_ACCOUNTED`(Z ≈ 0, H가 X2로 설명), `STALL_MIXED`(그 외). 앞 번호가 성립하면 뒤 번호는 매기지 않는다.
- 부가(판정 아님): 정지가 span의 첫 chunk에 몰렸는지(N3 Job 3의 첫-chunk host 차단 0.66–1.34 s, 원인 미해결 [측정]).

### 3-3. 판별량 B — drain 겹침과 TTFT·ITL 몫 (세 arm 모두, 감사 M1·M2·M3)

- **static arm에도 drain이 있다(rev0.1 정정)**: `adjust` 경로의 drain(`prefill_stream.synchronize()` + `decode_stream.synchronize()`)은 static 모드에서 **prefill 경계마다** 발화한다(`multiplexing_mixin.py` 주석 "EVERY prefill boundary"); 동적 arm에서는 SLO 분기가 이를 억제한다. 따라서 B는 D-3의 **세 arm(동적 ON, static D44 ON, 동적 OFF는 client 지표만)** 에서 계산하고, **seed별 동적 − static 차**를 기술값으로 낸다. static의 drain이 span마다 ≈ 0이면 그 사실이 결과다.
- 전환 집계는 `stream_idx_from ≠ stream_idx_to`로 한다: SLO\_SCHED 아래 컨트롤러 결정도 `adjust_stream_groups`에서 수행돼 reason = `adjust`로 찍히기 때문이다. drain ms에는 `update_decode_attn_backend` 갱신 시간이 **포함되지 않는다**(공시).
- 요청 r의 `drain_overlap_in_TTFT(r)` = r의 [도착, 첫 토큰]과 겹친 drain 시간 합, `share_TTFT(r)`; ITL p95/p99 꼬리 토큰 중 drain과 겹친 비율 `share_ITLtail`. **겹침은 손실이 아니다**: sync 대기 중에도 GPU prefill이 진행된다. 이 몫을 "drain 비용"으로 읽지 않는다.
- **겹침 허용창(M3)**: client–server 지연 보정으로 겹침 판정 창을 lag p99만큼 넓힌다. lag = `token_t_mono[0]` − 같은 rid의 `prefill_span_end` 분포를 gate와 본측정에서 p50/p99로 보고; 민감도 0·2 × p99 병기. 창은 데이터 전에 "lag p99"로 등록한다.
- **등록 라벨(이름 변경)**: `DRAIN_OVERLAP_MATERIAL` = seed 중앙값 `share_TTFT` 평균 구간 하한 ≥ 5% 또는 ITL p99 꼬리 토큰의 25% 이상이 겹침; `DRAIN_OVERLAP_NEGLIGIBLE` = 두 지표 구간 상한 < 1% / < 2.5%; 그 외 `DRAIN_OVERLAP_UNRESOLVED`. 구간은 seed 수준 t(n−1), n = 4(t(3) = 3.182), percentile bootstrap 금지(정본 §3 항목56). 이 라벨은 겹침의 기술이며 정책·원인 주장이 아니다.

### 3-4. 판별량 C — 관측자 비용 (감사 M5)

OFF 짝 비 `ln(OFF/ON)`을 seed 수준 t(n−1) 95% 구간으로 보고한다. **D-1(두 Job)의 ON/OFF는 boot 1개씩이라 boot 효과와 교락 → 기술값으로 강등**, 등록 라벨(`OBSERVER_WITHIN_3PCT`/`OBSERVER_COST_PRESENT`/`OBSERVER_UNRESOLVED`, 허용폭 ±3%)은 **D-3(seed별 재부팅)에서만** 낸다. D-3은 ON(DIAG) / "PHASE\_EVENTS만" / OFF 세 수준이므로 계층별 비용을 가른다. OFF가 끄는 범위는 §2-1에 등록했다. 기존 ρ\_on 1.156–1.193의 분해를 대체하지 않는다(감사 P1: 합성값).

### 3-5. 공변량 (모든 판별량에 병기)

`substrate.json`(GPU UUID·드라이버·호스트)·`clk_summary.json`(클럭·throttle 비율, 클럭 고정 불가)·bench 창 power cap 노출·`SUBSTRATE_WARNING`. Job 간 비교는 하지 않는다.

## 4. 측정점 삭제 원칙 (결정 30 형식 — 데이터 전 전수표)

원칙(고정): (P) 물리적 불가 또는 (R) 더 강한 증거로 대체될 때만 삭제, (C) 설정 제약이면 별도 boot·보정으로 살리거나 잃는 범위를 공시.

### 4-1. 이번에 포함한 것 (사용자 지시 대비 변경 없음)
Nano {D24, D44, D92} × {2k, 8k} × {8192, 65536}, 동적/static/OFF.

### 4-2. 삭제·미포함

| # | 측정점 | 이유 | 분류 | 상태 | 잃는 범위·대안 |
|---|---|---|---|---|---|
| X-1 | **Zamba 계열 전체(Zamba2-7B 정지 격자 6 ON 셀 + OFF 1, Zamba2-2.7B)와 D-2 Job** | **결정 46(2026-10-28, 사용자)**: Zamba(2.7B·7B 모두) 실험 대상에서 배제, 재실행하지 않음. 배경: 모델 context 4096으로 8k 불가 + 이전 배제 이력. 결정 45(Zamba2-7B arm)는 46으로 대체됨 | (R) 사용자 배제 결정 | 삭제 | 잃는 범위: **KISTI Zamba2 정지의 기전과 그 모델/backend 의존성은 이 Job으로 판별 불가.** HE0 범위 문장("phase 계측 0이라 lockstep 정지의 기여는 판단할 수 없다")은 그대로 유지된다. 복구 비용: 해당 없음(재실행 안 함) |
| X-3 | static arm의 PHASE OFF 짝(D-3) | 사용자 지시가 OFF 대칭 arm 1개 | (C) 설정·예산 — 허용 사유 아님(공시) | 삭제 | 잃는 범위: static ON/OFF 차를 동적 ON/OFF 차와 같은 Job에서 비교하지 못함(관측자 비용의 arm 의존성 미측정). 대안: D-3에 static OFF boot 1개(seed 4)(n=4) 추가 시 +1.3 GPU-h [유도] |
| X-4 | 예산 중간값(16384·32768), D 중간값(34·64) | 사용자 지시 격자가 양 끝(SPEC R13-3 격자의 양 끝) | (C) 설정 선택 | 미포함 | 잃는 범위: 기울기의 선형성 검사. 3점 D 격자로 b\_j의 선형 형태 가정만 가능, 비선형 반응(knee)은 못 봄 |
| X-5 | KISTI 설정(ShareGPT ~2k, triton, mem 0.82) 재현 셀 | 기판·모델·backend 다름; 결정 46으로 Zamba 배제 | (R) 사용자 결정 + 비교 불가 | 삭제 | 감사 P1 선택 편향 (2): "KISTI 설정 0셀"은 이 Job 후에도 유지된다 |

## 5. 공시 (모든 결과 문장에 붙임)

- **단일 모델·단일 backend(Nano, flashinfer)**: 게이트 #83의 모델·backend 교락은 이 Job에서 시험 대상이 아니다. 결과는 Nano·flashinfer 범위이며 KISTI Zamba2·triton의 정지에 대해 말하지 않는다.
- 계측 ON 조건이다(OFF arm 1개 제외). "PHASE ON 조건에서"로 시작한다.
- Job 간 기판(GPU·드라이버)이 다르면 Job 간 값을 한 풀로 합치지 않는다. 한 Job 안의 arm끼리만 비교한다.
- n = 4 seed, 정지 사건은 희귀할 수 있어 `UNRESOLVED_FEW`가 정상 결과일 수 있다.

## 6. 규약 준수

- 모든 arm 동일 trace hash·workload seed·server seed(Job 내), 실행 순서 randomize.
- cudagraph ON(운영점), backend·KV capacity·max-running 고정. GPU clock/power는 VESSL에서 고정 불가 — `clk.csv`·`substrate.json` 기록, arm 간 동일성 검사. no-cudagraph면 하한 표시.
- **realized-SM 검증(M6: 단위 = span)**: 각 span의 `prefill_sms`·`decode_sms`(span\_start 이벤트 [코드])를 실현 SM으로 기록한다. 요청 D와 실현 SM이 다른 span, 또는 decode가 비어 있던 span(8k 저부하에서 약 7% [유도])은 **그 span만 제외**하고, 셀은 **실현률 ≥ 95%**일 때만 유지한다(미만이면 `D_NOT_REALIZED`로 그 셀 라벨 제외). 비sticky static은 prefill 휴지 중 108 SM으로 돌아가므로 "D"는 목표값이다(결정 49).
- secret을 Job에 넣지 않는다. 결과·사전등록·엔진 코드를 이미지에 넣지 않는다.

## 7. Job 구성, 시간, GPU-h [유도 — 측정 아님, Job 크기 결정용]

**시간 모형**(비용 항 PREREG rev3 §7-1 승계): boot 200 s + warm-up 60 s = 4.3 min/boot, 셀 오버헤드 15 s, 셀 시간 = 요청 수/rate + 꼬리 30 s + 15 s. 2k 셀 (60/0.5 = 120 s) + 45 = 2.75 min, 8k 셀 (60/0.2 = 300 s) + 45 = 5.75 min. HE0형 1 run(3 라운드) ≈ 15 min(Nano λ\*를 임의 3 req/s로 둔 모형, 정책 PREREG §7 참조) + boot 4.3 min(재부팅 run). **모든 값은 측정 전이고 λ\* 의존 항은 용량 측정 뒤 갱신한다.**

| Job | 내용 | 스펙 | boot | 셀/run | 예상 h [유도] | 하드캡(×1.2) [제안] | 사용 GPU 시간(사용률) |
|---|---|---|---|---|---|---|---|
| **D-1a** Nano 정지 격자 2k | 6 ON 셀(boot마다 2k × seed 4) + OFF 대칭 1 boot(D44·65536) | `resourcespec-a100x1` | 7 | 7 × 4 × 2.75 min | 1.78 | 2.1 | 77 min / 107 min = 72% |
| **D-1b** Nano 정지 격자 8k | 6 ON 셀(boot마다 8k × seed 4, 0.1 req/s·40 요청 = 7.4 min/셀) + OFF 1 boot | 〃 | 7 | 7 × 4 × 7.4 min | 3.97 | 4.8 | 208 min / 238 min = 87% |
| **D-3** 동적 drain 블록 | 4 arm × seed 4 = 16 run: bind+GATE ON(DIAG) · static D44 ON(비sticky) · bind+GATE OFF · bind+GATE "PHASE\_EVENTS만". 동적 3 arm은 seed마다 재부팅(12 boot), static 1 boot | 〃 | 13 | 16 × 15 min | 4.93 | 5.9 | 240 min / 296 min = 81% |
| 합계 | arm-cell 18(정지 격자 ON 12 + OFF 2 + D-3 4) | | 27 | | **약 10.7 GPU-h** | 약 12.8 | |

- **같은 비교의 arm은 한 Job 안**: 비교 단위 = (D, 예산)와 ON/OFF 짝 → 길이별 Job 안에서 완결(D-1a, D-1b), (동적/static/OFF) → D-3. 길이 간 비교는 Job이 달라 기술 병기다(기판 교락).
- **Job당 ≤ 6 h**: 모든 Job 하드캡 포함 ≤ 5.9 h. D-1을 길이별로 나눈 이유: 8k 셀이 40 요청 × 0.1 req/s로 길어져 단일 Job이면 5.25 h(하드캡 6.3 h > 6 h)가 되기 때문(감사 권고). 비용: 부팅 7개 추가로 합계 +0.5 GPU-h.
- **사용률(GPU Job 적정 크기)**: 모두 `resourcespec-a100x1`(동시 사용 GPU 1). boot 구간에는 서빙하지 않으므로 사용률은 셀 시간 비율(72–87%). D-1a는 boot 지배로 낮다.
- 재개: 셀 단위 DONE 마커·`vessl_jobs.tsv`·sha 승계. 자동 재시도 없음, 재실행은 새 job name(실패 셀만 골라 재실행한 사실을 결과에 적는다).
- 이 Job이 채우는 COSTMODEL 칸: **CM-4(동거 간섭·정지 항의 기전)** — D-3의 drain 몫은 CM-4의 "정지 항" 보조 입력. 정책 칸 아님.

## 8. 순서

1. 계측 구현(engine-porter) → P-1·P-3 gate → 계측 커밋 해시 확정.
2. 하네스 구현 → **하네스층 감사 1회** → 봉인(§10 체크리스트) → `launch.sh` dry-run(명령·job name·`$/h` 표시) → **사용자 승인** → `--submit`. D-1a·D-1b는 독립 제출 가능. D-3은 정책 PREREG의 Job-CAP1·Job-TUNE 회수 뒤.
3. `fetch.sh`(DONE + `sha256sum -c`) 통과가 "완료". result-analyst가 §3 판별량 계산.

## 9. 신규 하네스 구현 항목 (감사 뒤 작성, 이 문서에서는 코드 0)

① 격자·반복·블록 무작위 계획 생성기와 계획 해시 ② 저부하 고정 rate 클라이언트(seed 고정, 토큰 절대 CLOCK_MONOTONIC) ③ 계측 env 매트릭스(ON/OFF arm이 OFF에서 env 누출 없음 검사) ④ 판별량 A 분석(R·Z 잔차 회귀·launch/GPU-bound 분류·군집 구간·VIF·짝짓기 키·조인 실패율·K2 라벨 순서 — 가짜 데이터로 라벨 경계 시험, 라벨러는 역변이에서 실패해야) ⑤ drain 몫 계산(요청 ID 대응, rids) ⑥ OFF 짝 비 계산 ⑦ realized-SM 검사. 변이 시험(예: OFF arm에 계측 env를 남김, X1 열을 X2로 바꿔치기)이 검출돼야 한다.

## 10. 봉인 체크리스트

| # | 항목 | 상태 |
|---|---|---|
| 1 | P-1…P-3, P-6…P-9 충족 | 대기 |
| 2 | 하네스 구현·CPU 시험·변이 | 대기 |
| 3 | 하네스층 감사 1회 | 대기 |
| 4 | 봉인(규칙·하네스·엔진·config digest, 이 문서의 [봉인 시 기입] 칸) | 대기 |
| 5 | CPU preflight → `launch.sh` dry-run → 사용자 승인 → `--submit`(이 문서 작성 시점에는 dry-run도 하지 않음) | 대기 |
| 6 | 장부류(변이 장부)는 봉인 바이트 확정 뒤 1회 | 규칙 |

## 11. 열린 결정 [미정] (rev0.2: 감사 r1 권고값 채택으로 닫힘)

**닫힘**: O-DX2 n = 4 · O-DX3 static = D44 사전 지정·비sticky(결정 49) · O-DX4 8k rate 0.1 req/s·40 요청 · 동등성 여유 유지(잔차 판별량 R·Z에 적용) · `DRAIN_OVERLAP_*` 임계 유지·라벨 이름 변경 · 겹침 허용창 = lag p99 · realized 판정 단위 = span(셀 실현률 ≥ 95%) · digest 단일 최종 트리.
**남은 선행**: P-1(계측 커밋·gate PASS) · P-2 필드 이름 확정 · P-8 최종 트리 digest(`PENDING_FINAL_TREE`) · P-9 sgl_kernel 환경 시험 로그 · P-6(D-3은 정책 CAP1·TUNE 뒤).
**gate 요구(engine-porter 몫, 감사 M7)**: (a) ON boot에서 `torch.cuda.Stream.synchronize`·`Event.synchronize`·`Tensor.item`을 감싸 diag 블록 안 호출 횟수 == 0 요구 (b) D1 동시성 경우 출력은 기록만 (c) D6 UNRESOLVED 규칙 — D-1은 허용, D-3은 첫 boot에서 slo\_switch 정합(D3 + D6) PASS 필수 (d) 변이 추가: 해결 전 pool 반환, MAX\_PENDING 초과 드롭 미집계.

## 12. 라벨과 금지 문안

라벨: `STALL_ABSENT`·`STALL_FOLLOWS_CHUNK_GPU`·`STALL_HOST_ISSUE_ACCOUNTED`·`STALL_FOLLOWS_DECODE_GPU`·`STALL_MIXED`·`UNRESOLVED_FEW`·`UNRESOLVED_COLLINEAR`·`UNRESOLVED_JOIN`·`D_NOT_REALIZED`·`DRAIN_OVERLAP_MATERIAL`·`DRAIN_OVERLAP_NEGLIGIBLE`·`DRAIN_OVERLAP_UNRESOLVED`·`OBSERVER_WITHIN_3PCT`·`OBSERVER_COST_PRESENT`·`OBSERVER_UNRESOLVED`.

**금지 문안(감사 `VERDICT_lockstep_diag_claims_r1` P1–P5 승계 + 이 PREREG)**
- "VESSL 기판에는 lockstep 정지가 없다"(한정 생략)·"L\_stall = 0"·"ρ\_on ≈ 1.17이 동거 간섭의 크기다"·"정지가 goodput에 영향 없다"·이 수치를 KISTI 수치와 비교·결합.
- "HE0은 lockstep 정지 기판의 산물이다"·"정지 없는 기판에서는 동적 제어가 이길 수 있다"·"layer-type 死는 KISTI 정지 기판 한정이다"·"VESSL·Nano에서 HE0이 뒤집힌다/유지된다"·"정지가 HE0를 설명/반증한다".
- "현 구현은 매 반복 lockstep이라 ITL이 span에 묶인다"(범위 없이)·"KISTI Ha8 정지는 이 Job의 기전과 같다/다르다".
- "C2 꼬리는 lockstep 정지에 오염됐다"·"C2 p50은 정지 없는 decode step 시간이다"·Job 865493 d24–d92 다리 인용.
- 다른 모델·backend(특히 Zamba2·triton)로의 전이 서술.
- `DRAIN_OVERLAP_MATERIAL`/`DRAIN_OVERLAP_NEGLIGIBLE`을 "동적 제어가 진 이유/이길 수 있는 이유"로 쓰는 것, drain 몫을 static과의 goodput 차이로 환산하는 것.
- 동등성 여유가 등록된 판별량 A 라벨 이전의 사후 진단 수치(진단 §3)를 "등록된 P4 판별 예측"으로 인용하는 것(S-6/K8: P4는 동등성 여유가 미등록이었다).
- PHASE ON 접두 없는 인용, "계측은 영향이 없다", OFF 짝 비를 다른 arm·모델·부하의 보정 계수로 쓰는 것(결정 22 추가 금지 승계).
- Job 간 값을 한 표에서 비교하는 것. Zamba 결과가 있는 듯한 서술.

- "static arm에는 drain이 없다"/"drain은 동적 arm에만 있다"(감사 M1: static 모드에서도 prefill 경계마다 발화) · drain 겹침 몫을 "drain 비용/손실"로 읽는 것(M2: sync 대기 중 GPU prefill 진행) · drain ms가 backend 갱신 시간을 포함한다는 서술 · 동적 전환을 reason = `adjust` 행 수로 세는 것.
- "H가 X2(chunk host 발행 시간)를 따른다 ⇒ CPU-launch 기전"(K1: H ≥ X2는 루프 순서의 회계 항등식) · ABSENT인 셀의 기울기를 "기전 판별"로 인용 · D-1 ON/OFF 짝(boot 1개씩)을 `OBSERVER_*` 라벨로 인용 · 비sticky static의 "D"를 실현 할당으로 서술("static D는 동거 중 목표값이며 실현 할당이 아니다").

**허용 문안 틀**: "VESSL A100-SXM4 · &lt;모델&gt; · &lt;backend&gt; · cudagraph ON · &lt;D&gt; · &lt;예산&gt; · &lt;프롬프트&gt; · PHASE ON 조건의 &lt;Job&gt;에서, 정지 반복의 host 차단은 &lt;라벨&gt;이었다(기울기 구간 …, n = 4 seed). 정책 판정이 아니다."


## 13. rev0.2 감사 r1 → 수리 위치 대조표

| 감사 항목 | 수리 | 위치 |
|---|---|---|
| K1 판별량 A가 회계 항등식 | R = t\_sync − t\_issue\_end, Z = H − max(X2, X3)를 X1·X3·F에 회귀, X2는 회계 항, `start_done_at_issue_start`·`end_done_at_issue_end`로 LAUNCH/GPU-bound 분류, 해당 금지 문안 | §3-2, §12, P-2 |
| K2 라벨 우선순위 | ABSENT → FEW → COLLINEAR/JOIN → 기울기 고정, ABSENT는 주 추정 보고만 | §3-2 |
| M1 "static에 drain 없음" 거짓 | 판별량 B를 세 arm에 계산, 동적−static 차 기술값, D-1·D-3 static 모드 = 비sticky 등록, 금지 문안 | §2-1, §3-3, §12 |
| M2 겹침 ≠ 손실 | 라벨 `DRAIN_OVERLAP_*`, 손실 해석 금지, 전환 집계 = `stream_idx_from ≠ to`, backend 갱신 미포함 공시 | §3-3, §12 |
| M3 client–server 지연 | lag 분포 보고 + 겹침 허용창 = lag p99(민감도 0·2×) | §3-3 |
| M4 짝짓기 키 | X1 = 같은 `decode_iter_seq`, X3 = 직전 probe, 조인 실패율 5% | §3-2 |
| M5 관측자 비용 boot 미반복 | D-1 기술값 강등, 라벨은 D-3(seed별 재부팅)에서만, OFF 범위 등록 | §2-1, §3-4 |
| M6 realized-SM 단위 | span 단위 제외 + 셀 실현률 ≥ 95% | §6 |
| M7 gate 공백 | (a)–(d) gate 요구 등록(코드는 engine-porter) | §11 |
| M8 digest 핀 | 정책·진단 같은 최종 트리 하나, P-8 `PENDING_FINAL_TREE`, "PHASE\_EVENTS만" arm 추가 | P-8, §2-2 |
| MINOR 헤더 rev 표기 | rev0.2로 통일 | 1행 |
| MINOR P-2 TBD | `PDMUX_PHASE_DIAG` + 필드명은 감사서 표기, 커밋 후 맞춤 | P-2 |
| MINOR X-5 Zamba2-7B 잔여 | 결정 46 반영 문구로 교체 | §4 |
| MINOR "모델과 backend가 함께 다르므로" | 문안 삭제(허용 문안 틀) | §12 |
| MINOR FLAT 여유 근거 | "관례" 공시 | §3-2 |
| MINOR 관측자 비용이 H에 포함 | 공시 추가 | §3-1 |
| 열린 [미정] 닫기 | O-DX2·3·4 등 채택 | §11 |
| Job 분할·비용 재산정 | D-1a/D-1b/D-3, 약 10.7 GPU-h | §7 |
