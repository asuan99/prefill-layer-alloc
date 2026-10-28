# 사전등록 rev0.1.1 (DIAG) — lockstep 정지 기전·동적 arm drain 대기 진단 Job, 2026-10-28

> **봉인 상태: 미봉인 초안 rev0.** 문서 작업(CPU)만 했다. 측정 0, GPU 0, `launch.sh` 실행 0(dry-run 포함), 성능·정책 판정 0. 하네스 코드는 쓰지 않았다(감사 뒤).
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
| P-2 | 계측 env 이름 확정 — 자리표시: chunk 이벤트 env = `TBD(engine-porter)` · drain 이벤트 env = `TBD(engine-porter)` · rids env = `TBD(engine-porter)`. 기존 env `PDMUX_PHASE_EVENTS`·`PDMUX_DECODE_PROBE_PATH`는 [코드] `multiplexing_mixin.py:2129-2302`·`decode_probe.py`. 봉인 때 실제 이름으로 치환한다. | 미정 | 봉인 |
| P-3 | gate 내용(engine-porter 설계, 이 문서는 요구만 둔다): 계측 ON/OFF에서 greedy 출력 토큰 동일성, 계측이 새 동기화(`synchronize`·`.item()`·블로킹 `query` 대기)를 만들지 않음, CUDA event 쌍의 시각 정합. 통과가 PASS이고 "특성화 종결"은 PASS가 아니다(결정 22 선례). | 미정 | 봉인 |
| P-6 | 동적 arm에 쓰는 Nano 튜닝 구성과 Nano 용량 rate는 정책 PREREG(`PREREG_HD_HE0cell_rev0`)의 Job-CAP1·Job-TUNE 결과(addendum)에서 온다. D-3은 그 뒤에만 제출한다. | 대기 | D-3 |
| P-7 | 하네스(신규 구현 항목 §9)는 감사 뒤 작성. | 대기 | 봉인 |

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
- **저부하 고정 rate**: 도착 Poisson, 입력 2k 0.5 req/s, 8k 0.2 req/s [제안]. 근거 = span 간 겹침을 줄여 정지를 관측하는 데 목적이 있고 goodput 판정이 아니므로 용량 이하로 둔다(Nano 8k shape B의 SLO 용량 0.15–0.22 req/s [측정, N3 감사 허용 문안 구간; 인용은 구간·병기와 함께만]; 8k 0.2 req/s는 그 용량 근방이라 정지 관측에 큐잉이 섞일 수 있음 — 그래서 큐 길이를 이벤트에 기록한다 [코드, `queue_len`]). 8k rate는 [미정 — 봉인 전 Nano 8k에서 TTFT 중앙값 ≤ 2× 단독 floor 확인].
- 셀당 **60 요청**(span 60개 이상) + 꼬리 30 s. 반복 = seed 4개(트레이스 해시·workload seed·server seed가 seed마다 고정, 같은 Job 안 모든 셀에 동일 시퀀스 적용).
- **PHASE OFF 대칭 arm**(관측자 비용): Job마다 1 boot, 해당 Job의 (D44, 예산 65536) 구성에서 PHASE·chunk·drain 계측 전부 OFF(decode probe도 OFF), 프롬프트는 그 Job이 가진 전부. 관측 지표는 client 측(TTFT, 토큰 간격 p50/p95/p99)만. 같은 Job 안의 ON 셀과 짝으로만 비교한다(다른 Job 비교 금지).

### 2-2. 동적 arm drain 블록 (Nano, HE0형 변화 trace 핵심 셀)

- 트레이스·rate·SLO·rate 환산·cliff 규칙은 `PREREG_HD_HE0cell_rev0` §3(정책 PREREG)을 **그대로 가져온다**: KISTI `sharegpt_vary_bench.sbatch` 구조(구간당 200 요청, LO/HI 교대 3 라운드, 구간 사이 drain), rate는 Nano λ\*\_best-static 비례(LO/HI = 1/4). 이 문서에서 다시 정의하지 않는다 — 정의가 갈라지면 정책 PREREG가 이긴다.
- arm 3개, 같은 Job(D-3) 안에서 블록 무작위 순서, 모두 seed 4개:
  1. **reactive bind+GATE**, 계측 ON — 정책 PREREG의 Job-TUNE이 고른 Nano 재튜닝 구성.
  2. **best decode-heavy static**, 계측 ON — [제안] D44(KISTI HE0 승자 D, 사전 지정). 정책 PREREG의 HE0 평가 결과가 먼저 나오면 그 사후 best static으로 바꿀지는 [미정, D-3 제출 전 사용자 확인]. 어느 쪽이든 이 블록의 목적은 drain 대기 측정이며 static 선정이 아니다.
  3. **PHASE OFF 대칭 arm**: bind+GATE, 계측 OFF(같은 구성, 같은 seed). 관측자 비용은 **동적 arm 한정**이다. static arm의 OFF 짝은 두지 않는다(잃는 범위: static의 관측자 비용을 이 Job 안에서 못 잼 — §4-3).
- 동일 boot 조건: mem-fraction·max-running·cudagraph·config yml·`split_forward_token_budget`가 세 arm에서 같다. 동적 arm만 `PDMUX_SLO_*` env가 추가된다.

### 2-3. 셀 순서·무작위화

각 Job 안에서 boot 순서를 블록 무작위(seed는 계획 해시에서 파생, 봉인 시 기입)로 한다. boot 안에서는 프롬프트 길이 순서를 seed마다 교대. warm-up(60 s) / correctness(첫 요청 출력 토큰 길이 검사) / benchmark phase를 분리하고, warm-up 구간은 계측은 켜되 판별량에서 제외한다.

## 3. 계측과 판별량

### 3-1. 이벤트와 시각 기준

진단 §1 표 승계(`prefill_span_start/_end`, `decode_iteration`, 시각 = host CLOCK_MONOTONIC) + P-1 신규 이벤트. **GPU 시각은 CUDA event 쌍의 경과 ms로만 쓰고 host 시각과 섞어 한 축에 놓지 않는다.** `prefill_span_end`는 GPU 완료 이후 최대 1 루프 반복 늦게 관측된다(진단 §1) — chunk GPU 시간은 span 구간 차가 아니라 event 쌍으로 잰다(진단 §5 공백 2).

### 3-2. 판별량 A — 정지 반복에서 H가 무엇을 따르는가 [동등성 여유를 데이터 전에 등록]

- 단위: 반복 i = 한 `decode_iteration`(발행~`decode_stream.synchronize()` 종료). H\_i = `t_sync − t_launch`(host 차단). 설명 변수: X1\_i = 그 반복에서 진행 중인 prefill chunk의 GPU 시간(event 쌍), X2\_i = 그 반복의 chunk host 발행 시간(`t_issue_end − t_issue_start`), X3\_i = 같은 반복 decode forward의 GPU 시간(`decode_fwd_gpu_ms`).
- **정지 반복 집합 S**: 간격(= 연속 `t_sync` 차)이 같은 boot·같은 bs의 비-span 반복 중앙값의 **3배 초과**(진단 §3-4와 같은 정의, 민감도 2배·5배 병기). S는 판별의 부차 집합이고, **주 추정은 span과 겹친 모든 반복**(power를 위해)에서 한다.
- 추정: H를 X1, X2, X3에 **동시 OLS**(절편 포함), (cell, seed) 군집 표준오차 + 군집 수준 t 구간(군집 수가 적으면 t(군집−1), percentile bootstrap 금지). 격자가 X1을 (D, 프롬프트)로, X2를 (예산 → `forward_count`)로 독립적으로 움직이게 설계했다. 공선성: 변수 분산팽창계수 VIF > 5면 해당 비교는 `UNRESOLVED_COLLINEAR`.
- **동등성 여유 [제안, 봉인 시 확인]**: 기울기 b\_j(ms/ms)의 95% 구간이
  - [0.7, 1.3]에 완전히 들면 "H가 X\_j를 1:1로 따른다"(`FOLLOWS_Xj`),
  - [−0.05, +0.05]에 완전히 들면 "X\_j에 반응하지 않는다"(`FLAT_Xj`; 해상도 참조 — 보관 N3·N2·T3의 span 시간 기울기 0.0015, 95% 구간 [0.0005, 0.0027] [측정, 진단 §3-1, N3·N2-EA·T3 한정]. 여유 0.05는 그 해상도의 약 20배로 느슨하다),
  - 그 외 `PARTIAL_Xj`(둘 다 아님).
- **셀 집합별 라벨**(모델 × D × 예산 × 프롬프트 한 셀 단위가 아니라 (모델, 프롬프트) 풀에서; 셀별 군집 수 부족을 막기 위해):
  - `STALL_ABSENT`: S 크기가 반복의 1e-4 미만이고 최대 정지 간격 < 5 × 중앙값(N3 Job 4 선례 [측정]). H의 기울기 판별은 하지 않는다.
  - `STALL_FOLLOWS_CHUNK_GPU`: FOLLOWS\_X1 ∧ FLAT\_X2 ∧ FLAT\_X3.
  - `STALL_FOLLOWS_HOST_ISSUE`: FOLLOWS\_X2 ∧ FLAT\_X1 ∧ FLAT\_X3.
  - `STALL_FOLLOWS_DECODE_GPU`: FOLLOWS\_X3 ∧ FLAT\_X1 ∧ FLAT\_X2.
  - `STALL_MIXED`: 둘 이상이 FOLLOWS 또는 PARTIAL.
  - `UNRESOLVED_FEW`: S가 30개 미만이거나 군집 < 6 또는 seed < 3에서 S가 나온 것.
- 부가(판정 아님): 정지가 나온 반복이 **span의 첫 chunk에 몰려 있는가**(N3 Job 3의 첫-chunk host 차단 0.66–1.34 s, 원인 미해결 [측정]) — 첫 chunk 반복과 나머지의 H 분포를 따로 보고한다.

### 3-3. 판별량 B — 동적 arm drain 누적 대기의 TTFT·ITL 몫 (D-3)

- drain 이벤트 구간 \[t0, t1\](두 synchronize 합, 사유, 진행 chunk 수). 요청 r의 `drain_in_TTFT(r)` = r의 \[도착, 첫 토큰\]과 겹친 drain 시간 합, `share_TTFT(r) = drain_in_TTFT(r) / TTFT(r)`. 토큰 간격 중 drain 구간과 겹친 것의 비율을 ITL p95/p99 꼬리 토큰에 대해 센다(`share_ITLtail`).
- 보고: seed별 `Σ drain ms`, drain 횟수·길이 분포(진행 chunk 수별), LO/HI 구간별, 요청 TTFT 몫 분위수(p50/p95), `share_ITLtail`. static arm에는 drain이 없으므로 대조는 동적 OFF arm(관측자) 대 ON arm의 client 측 지표 차(§3-4)와, 같은 Job static ON arm의 같은 지표(무 drain 기준).
- **등록 라벨**(데이터 전, [제안]): `DRAIN_MATERIAL` = seed 중앙값 `share_TTFT` 평균 ≥ 5%(구간 하한이 5% 이상) 또는 ITL p99 꼬리 토큰의 25% 이상이 drain과 겹침; `DRAIN_NEGLIGIBLE` = 두 지표 모두 95% 구간 상한 < 1%(토큰 겹침은 < 2.5%); 그 외 `DRAIN_UNRESOLVED`. 구간은 seed 수준 t(n−1)(n = 4, t(3) = 3.182), percentile bootstrap 금지(정본 §3 항목56).
- 이 라벨은 "동적 arm이 drain에서 잃는 TTFT·ITL 몫"의 기술이다. `DRAIN_MATERIAL`이어도 "동적 제어가 질 이유"로, `DRAIN_NEGLIGIBLE`이어도 "정지는 HE0와 무관"으로 쓰지 않는다(감사 P2 금지).

### 3-4. 판별량 C — 관측자 비용

같은 Job, 같은 seed의 OFF 대 ON 짝에서 client 측 TTFT·토큰 간격 p50/p95/p99의 짝 비 `ln(OFF/ON)`을 seed 수준 t(n−1) 95% 구간으로 보고한다. 등록 허용폭 ±3%(CLAUDE.md 게이트 3과 정합) 안이면 `OBSERVER_WITHIN_3PCT`, 구간이 ±3% 밖이면 `OBSERVER_COST_PRESENT`, 걸치면 `OBSERVER_UNRESOLVED`. 이 라벨은 ON 조건 수치에 "PHASE ON 조건에서" 접두를 붙일지의 근거일 뿐이며, 기존 ρ\_on 1.156–1.193의 분해(동거·전력·관측자)를 대체하지 않는다(감사 P1: 합성값).

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
| X-5 | KISTI 설정(Zamba2-2.7B ShareGPT ~2k, triton, 0.82 mem) 재현 셀 | 기판 다름·모델 다름 | (R) 비교 불가 | 삭제 | 감사 P1 선택 편향 (2): "KISTI 설정 0셀"은 이 Job 후에도 일부 유지(Zamba2-7B/triton/2k만 근접) |

## 5. 공시 (모든 결과 문장에 붙임)

- **단일 모델·단일 backend(Nano, flashinfer)**: 게이트 #83의 모델·backend 교락은 이 Job에서 시험 대상이 아니다. 결과는 Nano·flashinfer 범위이며 KISTI Zamba2·triton의 정지에 대해 말하지 않는다.
- 계측 ON 조건이다(OFF arm 1개 제외). "PHASE ON 조건에서"로 시작한다.
- Job 간 기판(GPU·드라이버)이 다르면 Job 간 값을 한 풀로 합치지 않는다. 한 Job 안의 arm끼리만 비교한다.
- n = 4 seed, 정지 사건은 희귀할 수 있어 `UNRESOLVED_FEW`가 정상 결과일 수 있다.

## 6. 규약 준수

- 모든 arm 동일 trace hash·workload seed·server seed(Job 내), 실행 순서 randomize.
- cudagraph ON(운영점), backend·KV capacity·max-running 고정. GPU clock/power는 VESSL에서 고정 불가 — `clk.csv`·`substrate.json` 기록, arm 간 동일성 검사. no-cudagraph면 하한 표시.
- **realized-SM 검증**: 요청 D ≠ 실현 SM일 수 있다(생성 성공 ≠ 요청 수신). 각 boot의 `prefill_sms`·`decode_sms`(span_start 이벤트 [코드])로 실현 SM을 기록하고, 요청 SM과 다르면 셀을 `D_NOT_REALIZED`로 표시해 해당 D 라벨에서 제외한다(Stage 0 교훈: pin은 realized로 검증).
- secret을 Job에 넣지 않는다. 결과·사전등록·엔진 코드를 이미지에 넣지 않는다.

## 7. Job 구성, 시간, GPU-h [유도 — 측정 아님, Job 크기 결정용]

**시간 모형**(비용 항 PREREG rev3 §7-1 승계): boot 200 s + warm-up 60 s = 4.3 min/boot, 셀 오버헤드 15 s, 셀 시간 = 요청 수/rate + 꼬리 30 s + 15 s. 2k 셀 (60/0.5 = 120 s) + 45 = 2.75 min, 8k 셀 (60/0.2 = 300 s) + 45 = 5.75 min. HE0형 1 run(3 라운드) ≈ 15 min(Nano λ\*를 임의 3 req/s로 둔 모형, 정책 PREREG §7 참조) + boot 4.3 min(재부팅 run). **모든 값은 측정 전이고 λ\* 의존 항은 용량 측정 뒤 갱신한다.**

| Job | 내용 | 스펙 | boot | 셀/run | 예상 h [유도] | 하드캡(×1.2) [제안] | 사용 GPU 시간(사용률) |
|---|---|---|---|---|---|---|---|
| **D-1** Nano 정지 격자 | 12 ON 셀(6 boot, 각 boot에 2k·8k × seed 4) + OFF 대칭 1 boot(D44·65536, 2k·8k × seed 4) | `resourcespec-a100x1` | 7 | 7 × 4 × (2.75 + 5.75) min | 4.47 | 5.4 | 셀 3.97 h / 4.47 h = 89% |
| **D-3** 동적 drain 블록 | 3 arm(bind+GATE ON, best static ON, bind+GATE OFF) × seed 4 = 12 run. 동적 arm은 컨트롤러 상태가 boot에 남으므로 seed마다 재부팅(8 boot), static은 1 boot에서 seed 4개 | `resourcespec-a100x1` | 9 | 12 × 15 min | 3.65 | 4.4 | 3.0 h / 3.65 h = 82% |
| 합계 | arm-cell 16(정지 격자 ON 12 + OFF 1 + 동적/static/OFF 3) | | 16 | | **약 8.1 GPU-h** | 약 9.8 | |

- **같은 비교의 arm은 한 Job 안**: 비교 단위 = (D, 예산, 프롬프트)와 ON/OFF 짝 → D-1 한 Job, 동적/static/OFF → D-3 한 Job.
- **Job당 ≤ 6 h**: 두 Job 모두 하드캡 포함 ≤ 5.4 h. D-1이 rev3의 "기대 ≤ 4.4 h" 관례를 4.47 h로 넘는다 — 반복을 seed 3으로 줄이면 3.4 h(`+` 잃는 범위: n = 3 → t(2) = 4.30, 구간 확대) [미정, O-DX2, 초안은 n = 4 유지].
- **사용률(GPU Job 적정 크기)**: 모두 GPU 1장 사용 스펙 `resourcespec-a100x1`(동시 사용 GPU 1). boot 구간(약 4.3 min)에 GPU가 서빙하지 않으므로 사용률은 셀 시간 비율(D-1 89%, D-3 82%).
- 재개: 셀 단위 DONE 마커·`vessl_jobs.tsv`·sha 승계. 자동 재시도 없음, 재실행은 새 job name(실패 셀만 골라 재실행한 사실을 결과에 적는다).
- 이 Job이 채우는 COSTMODEL 칸: **CM-4(동거 간섭·정지 항의 기전)** — D-3의 drain 몫은 CM-4의 "정지 항" 보조 입력. 정책 칸 아님.

## 8. 순서

1. 계측 구현(engine-porter) → P-1·P-3 gate → 계측 커밋 해시 확정.
2. 하네스 구현 → **하네스층 감사 1회** → 봉인(§10 체크리스트) → `launch.sh` dry-run(명령·job name·`$/h` 표시) → **사용자 승인** → `--submit`. D-1은 독립 제출 가능. D-3은 정책 PREREG의 Job-CAP1·Job-TUNE 회수 뒤.
3. `fetch.sh`(DONE + `sha256sum -c`) 통과가 "완료". result-analyst가 §3 판별량 계산.

## 9. 신규 하네스 구현 항목 (감사 뒤 작성, 이 문서에서는 코드 0)

① 격자·반복·블록 무작위 계획 생성기와 계획 해시 ② 저부하 고정 rate 클라이언트(seed 고정, 토큰 절대 CLOCK_MONOTONIC) ③ 계측 env 매트릭스(ON/OFF arm이 OFF에서 env 누출 없음 검사) ④ 판별량 A 분석(동시 OLS·군집 구간·VIF·동등성 라벨 — 가짜 데이터로 라벨 경계 시험, 라벨러는 역변이에서 실패해야) ⑤ drain 몫 계산(요청 ID 대응, rids) ⑥ OFF 짝 비 계산 ⑦ realized-SM 검사. 변이 시험(예: OFF arm에 계측 env를 남김, X1 열을 X2로 바꿔치기)이 검출돼야 한다.

## 10. 봉인 체크리스트

| # | 항목 | 상태 |
|---|---|---|
| 1 | P-1…P-3, P-6, P-7 충족 | 대기 |
| 2 | 하네스 구현·CPU 시험·변이 | 대기 |
| 3 | 하네스층 감사 1회 | 대기 |
| 4 | 봉인(규칙·하네스·엔진·config digest, 이 문서의 [봉인 시 기입] 칸) | 대기 |
| 5 | CPU preflight → `launch.sh` dry-run → 사용자 승인 → `--submit`(이 문서 작성 시점에는 dry-run도 하지 않음) | 대기 |
| 6 | 장부류(변이 장부)는 봉인 바이트 확정 뒤 1회 | 규칙 |

## 11. 열린 결정 [미정]

- O-DX2 D-1 반복 n = 4(초안) 대 3.
- O-DX3 D-3 static arm = D44 사전 지정(초안) 대 정책 PREREG HE0 평가의 사후 best static.
- O-DX4 8k rate 0.2 req/s의 큐잉 영향(봉인 전 확인).
- 동등성 여유 [0.7, 1.3]·[−0.05, 0.05]·`DRAIN_*` 임계(5%·25%·1%·2.5%) — 모두 [제안], 봉인 시 사용자 확인.

## 12. 라벨과 금지 문안

라벨: `STALL_ABSENT`·`STALL_FOLLOWS_CHUNK_GPU`·`STALL_FOLLOWS_HOST_ISSUE`·`STALL_FOLLOWS_DECODE_GPU`·`STALL_MIXED`·`UNRESOLVED_FEW`·`UNRESOLVED_COLLINEAR`·`D_NOT_REALIZED`·`DRAIN_MATERIAL`·`DRAIN_NEGLIGIBLE`·`DRAIN_UNRESOLVED`·`OBSERVER_WITHIN_3PCT`·`OBSERVER_COST_PRESENT`·`OBSERVER_UNRESOLVED`.

**금지 문안(감사 `VERDICT_lockstep_diag_claims_r1` P1–P5 승계 + 이 PREREG)**
- "VESSL 기판에는 lockstep 정지가 없다"(한정 생략)·"L\_stall = 0"·"ρ\_on ≈ 1.17이 동거 간섭의 크기다"·"정지가 goodput에 영향 없다"·이 수치를 KISTI 수치와 비교·결합.
- "HE0은 lockstep 정지 기판의 산물이다"·"정지 없는 기판에서는 동적 제어가 이길 수 있다"·"layer-type 死는 KISTI 정지 기판 한정이다"·"VESSL·Nano에서 HE0이 뒤집힌다/유지된다"·"정지가 HE0를 설명/반증한다".
- "현 구현은 매 반복 lockstep이라 ITL이 span에 묶인다"(범위 없이)·"KISTI Ha8 정지는 이 Job의 기전과 같다/다르다".
- "C2 꼬리는 lockstep 정지에 오염됐다"·"C2 p50은 정지 없는 decode step 시간이다"·Job 865493 d24–d92 다리 인용.
- 다른 모델·backend(특히 Zamba2·triton)로의 전이 서술.
- `DRAIN_MATERIAL`/`DRAIN_NEGLIGIBLE`을 "동적 제어가 진 이유/이길 수 있는 이유"로 쓰는 것, drain 몫을 static과의 goodput 차이로 환산하는 것.
- 동등성 여유가 등록된 판별량 A 라벨 이전의 사후 진단 수치(진단 §3)를 "등록된 P4 판별 예측"으로 인용하는 것(S-6/K8: P4는 동등성 여유가 미등록이었다).
- PHASE ON 접두 없는 인용, "계측은 영향이 없다", OFF 짝 비를 다른 arm·모델·부하의 보정 계수로 쓰는 것(결정 22 추가 금지 승계).
- Job 간 값을 한 표에서 비교하는 것. Zamba 결과가 있는 듯한 서술.

**허용 문안 틀**: "VESSL A100-SXM4 · &lt;모델&gt; · &lt;backend&gt; · cudagraph ON · &lt;D&gt; · &lt;예산&gt; · &lt;프롬프트&gt; · PHASE ON 조건의 &lt;Job&gt;에서, 정지 반복의 host 차단은 &lt;라벨&gt;이었다(기울기 구간 …, n = 4 seed). 모델과 backend가 함께 다르므로 모델 간 차이를 귀속하지 않는다. 정책 판정이 아니다."
