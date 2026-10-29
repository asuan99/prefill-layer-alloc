# 사전등록 rev2 addendum (DIAG, **미봉인 초안**) — 영역 판별(M1·경계 셀)·B1 크기·M5/M6, 2026-10-29

> **상태: 미봉인 초안(문서 작업만, 결정 57 ②).** 측정 0, GPU 0, `launch.sh` 실행 0(dry-run 포함), 제출 0, 성능·정책 판정 0. 봉인된 `PREREG_DIAG_stall_drain_rev0_2026-10-28.md`(rev1)와 `SEALED_rev1.txt`는 **수정하지 않았다.** 이 문서는 rev1 위에 얹는 **추가분만** 정의하고, rev1의 판별량 B·C 라벨·규칙·금지 문안·n=4 seed·Nano 단일 모델 범위를 그대로 상속한다(충돌 시 rev1 우선, 단 §9 금지 문안은 추가).
> **엔진 = rev1과 같은 digest 하나**: `65cbb49d5caa630bf418520a8c09c664ab304cfbda4ce5350c56f6ae82e6b777`(= 커밋 `3858f129`, rev1 P-8). **branch 코드(`impl-defects-e123`) 없음.** 따라서 이 문서가 쓰는 계측은 main에 이미 있는 `PDMUX_PHASE_EVENTS` · `PDMUX_PHASE_DIAG`(chunk/drain) · decode_probe(`decode_fwd_gpu_ms`)뿐이다. 이것으로 못 재는 것은 §6에서 연기로 명시한다.
> 근거: 작업 루트 `PDMUX_IMPL_FIX_PLAN_2026-10-28.md` Part A3(M1–M7)·A5 · `../impl_defects/PHASE0_CPU_CHECK_2026-10-28.md`, `phase0_c1_c2.md`, `phase0_c3.md` · 감사 `../impl_defects/AUDIT_scope_loop_dependence_r1_2026-10-29.md` §3 실험 2(B1 크기)·실험 4(CM-4 영역) · rev1 감사 r1·r2.
> 라벨: [측정] · [유도](식·모형, 측정 아님) · [코드](`multiplexing_mixin.py`@`3858f129` 직접 확인) · [가정] · [제안](이 문서가 고른 값, 봉인 시 사용자 확인) · [미정] · **[봉인 시 기입]**.
> **라벨 정책(결정 50 승계)**: 이 addendum은 **기전·정책 라벨을 새로 만들지 않는다.** 새 산출물은 (i) 기술값 표, (ii) 양성 대조 통과/실패 표시, (iii) 관문 지표(CONTINUE / STOP / UNMEASURED — 다음 Job 제안용이며 주장 아님)뿐이다. 셀 단위(군집 seed 4 < 6) 라벨은 내지 않는다.

## 0. 무엇을 추가하는가 (rev1 대비)

| # | 추가 | 종류 | 셀·boot 변화 | 막힌 질문(계획 A3) |
|---|---|---|---|---|
| A-1 | **M1 셀**: 2k × `split_forward_token_budget` 512 × D44 static(비sticky), PHASE+DIAG+probe ON | 신규 셀 | D-1a에 **+1 boot** (4 seed) | 박자 영역(r<1)이 한 번이라도 실현되는가 (phase0 C3: 기존 측정 전부 r≫1 또는 span 1개) |
| A-2 | **영역 판별 분석**: rev1 격자 12셀 + M1에 span GPU–decode step 짝짓기 | 분석만 | 0 | 경계 4셀(budget 8192 × D24/D44)의 영역 |
| A-3 | **B1 크기 블록 D-4**: 8k 프롬프트에서 동적 arm이 진행 중 span과 겹쳐 전환하는 구성 | 신규 Job | 신규 Job 1개, 8 boot | B1(전환 시 in-flight drain)의 빈도 × drain ms |
| A-4 | **M6**: decode가 빈 뒤 D SM 유휴(adjust drain, span 진행 중) | 분석만 | 0 | B2 크기 |
| A-5 | **M5**: prefill 진행 중 도착 요청의 입장 대기 | 분석만 | 0 | B3 크기 |
| 연기 | M2·M3·M4·M7(완전판) | — | — | §6 |

rev1의 12 격자 셀·D-3·OFF 대칭 arm·B·C 라벨은 **변하지 않는다.** A-2/A-4/A-5는 rev1 셀에서 나오는 같은 telemetry를 다시 해석하는 것이라 boot·셀 시간이 늘지 않는다.

## 1. 선행·계측 필드 대조 (코드 기준 — rev1 표기와 실제 이름)

rev1/계획서가 쓴 이름과 `3858f129` 실제 이름이 다르다. **이 addendum은 실제 이름을 쓴다**(rev1 P-2 "필드 어긋나면 코드 기준 정정" 조항에 따른 해석이며 rev1 본문은 건드리지 않는다).

| 개념 | rev1/계획서 표기 | 실제 필드 [코드] | 비고 |
|---|---|---|---|
| chunk GPU 시간 | `gpu_ms` | **`chunk_gpu_ms`** (`prefill_chunk` 이벤트, event 쌍 resolve 후에만 emit — 미해결·MAX_PENDING 초과 폐기 chunk는 이벤트 없음, `n_chunks_dropped` 집계) | 측정 구간 = 같은 prefill stream의 start/end event 사이. launch-bound면 발행 공백을 포함하므로 GPU 순수 시간의 **상한**이다 |
| chunk 층 위치 | `split_index` | `split_index`(발행 **전** 첫 층, `num_layers`와 함께) · `forward_count` | `batch.split_index`는 발행 뒤 `next_split_index`로 갱신된다(`:1480,:1513`) → drain 시점의 `split_index` = **다음에 발행할 층**, 남은 미발행 층 = `num_layers − split_index` |
| decode 짝 | `decode_iter_seq` | `prefill_chunk.decode_iter_seq == decode_iteration.iter_seq == decode_forward.decode_iter_seq`, `decode_launched` | rev1 §3-2 짝짓기·폴백·두 실패율 규칙 그대로 |
| decode step GPU | `decode_fwd_gpu_ms` | decode_probe `decode_forward.decode_fwd_gpu_ms` | 동거(co-located) 구간 값 = 같은 반복에 chunk가 발행된 `decode_launched=True` 반복 |
| drain in-flight | `in_flight_chunks` | **`prefill_chunks_unresolved`** (drain 시작 시 non-blocking poll 뒤 event가 아직 안 끝난 chunk 수) · `split_present` · `prefill_kernel_done_wait` | "미해결 > 0" = poll 시점에 GPU가 아직 안 끝낸 chunk가 있음. 폐기된 chunk(MAX_PENDING)는 세지 않는다(과소 방향) |
| drain 시간 | `host_ms` | `prefill_sync_ms`, `decode_sync_ms`, `t_drain_start/_prefill_sync_end/_drain_end`(host perf_counter), `reason`, `stream_idx_from/_to`, `prefill_sms_from/_to`, `decode_sms_from/_to`, `running_bs`, `queue_len`, `span_seq`, `decode_iter_seq` | 전환 집계는 `stream_idx_from ≠ stream_idx_to`(rev1 §3-3) |
| 입장 | — | `prefill_span_start{span_seq, n_new, extend_tokens, running_bs, queue_len, cap, stream_idx, prefill_sms, decode_sms, rids(DIAG)}` + telemetry `timestamp_monotonic_s` | **서버 쪽 대기열 진입 시각은 기록되지 않는다**(M5 한계, §4-3) |
| 클라이언트 시각 | — | `PDMUX_LOADGEN_ABS_MONO` 래퍼의 요청 송신·토큰 CLOCK_MONOTONIC | server `perf_counter`와 같은 호스트 단조 시계 [가정: 같은 컨테이너, 봉인 시 gate에서 확인] |

선행 조건은 rev1 P-1…P-9와 같다. 추가 선행: **P-A1** — GPU gate(`engine_gate_diag.sh`)에 "budget 512 boot smoke"(config가 512를 받아 부팅하고 `forward_count==1` span이 나오는지) 1건 추가를 engine-porter에게 요구한다 [제안]. 512가 거부되면 A-1 셀은 `PDMUX` budget 값 {512} → {1}(= `max(1, budget // extend)` 하한에 닿는 최소 유효 값)으로 대체한다(영역 판별 목적은 `forward_count==1`, 즉 층 하나 = span 하나다).

## 2. A-1 + A-2 — 영역 판별 (M1 셀과 경계 4셀)

### 2-1. M1 셀 정의

- **구성**: Nano · flashinfer · cudagraph ON · radix OFF · mem 0.6 · max-running 48(rev1 §2-1과 동일), **2k 프롬프트 · 출력 512 고정 · budget 512 · D44=prefill 64 · static 비sticky**(`PDMUX_R2_POLICY=fixed`, sticky 없음), 계측 ON(PHASE_EVENTS+DIAG+probe). rate = rev1 2k 저부하 **0.5 req/s·60 요청**, seed 4, 같은 D-1a Job 안에서 블록 무작위 boot 순서에 포함. OFF 짝 없음(rev1 §2-1의 OFF arm은 (D44, 65536) 하나뿐이며 그대로).
- **예측 [유도, phase0_c3 §3·§4.2 — 측정 아님]**: `forward_count = max(1, 512 // 2048) = 1` → span 56개(층당 1개), 가장 긴 span(mamba/MLP 층) ≈ 7.6 ms, 동거 decode step ≈ 20.5 ms × ρ 1.16 = 23.8 ms, **r ≈ 0.32**(박자 띠 0.67 밑, ρ 범위 전체에서도 박자). 이를 뒤집으려면 span 모형이 3배 넘게 과소추정이어야 한다. 이 예측값을 데이터 전에 등록한다(§2-3 모형 점검 규칙의 기준).
- 부수 예측 [유도]: 박자 영역에서 prefill 완료 ≈ 56 × 23.8 ms ≈ 1.3 s(GPU 일 ≈ 0.40 s보다 김). 도착 0.5 req/s라 prefill 점유율 ≈ 0.65 → 입장 대기가 생긴다(M5에 표본 제공). 셀 시간 모형(120 s 도착 + 꼬리 30 s)은 이 큐 지연(M/G/1 대략 1–2 s)을 흡수한다.

### 2-2. 판별량 R (영역 지표) — 정의

대상 셀: rev1 격자 12셀 + M1 셀 = 13셀. 기술값 표만 낸다(라벨 없음).

1. **chunk–decode 짝**: `prefill_chunk` 레코드 중 `decode_launched=True`이고 같은 `decode_iter_seq`의 `decode_forward.decode_fwd_gpu_ms`가 조인되는 것만 쓴다(rev1 §3-2 조인 규칙·두 실패율 병기). 키 폴백 쓴 레코드는 따로 표시하고 주 통계에서 **제외**한다(민감도로 포함 병기).
2. **q = chunk_gpu_ms / decode_fwd_gpu_ms** (짝 단위). 분포(p50/p95/p99)를 셀별·seed별로 낸다.
3. **r_span = (span 안 최대 `chunk_gpu_ms`) / (같은 span의 짝지어진 `decode_fwd_gpu_ms` 중앙값)**. span = 같은 `span_seq`. 이것이 phase0 C3의 r(최장 span ÷ 동거 step × ρ)의 **직접 측정판**이다(ρ는 분모 측정값에 이미 포함되므로 곱하지 않는다). 셀 값 = span r_span의 중앙값, seed 4개의 값과 범위 병기.
4. **영역 기술 분류(라벨 아님, [제안] 띠 승계 phase0_c3 A5)**: span r_span ≤ 0.67 → "박자 후보", ≥ 1.5 → "적체 후보", 그 사이 → "경계". 각 분류의 span 비율을 셀별로 보고한다. 셀 요약은 비율 표이지 단일 영역 선언이 아니다.
5. **보조 신호(corroboration, 독립 필드)**: (a) `start_done_at_issue_start` 참 비율(참 = 발행 시작 때 prefill stream이 비어 있었음 = 적체 없음), (b) span당 마지막 chunk 발행 뒤 순수 WAIT 반복(`decode_iteration.prefill_kernel_done_wait=True`) 수의 중앙값 — N3 49, N2-EA 28 [측정, phase0_c3 §4.3] 기준으로 대조, (c) `end_done_at_issue_end` 참 비율. rev1 §3-2의 "비율만 보고, 규칙 적용 없음" 원칙을 지킨다.
6. **측정기 편향 공시**: `chunk_gpu_ms`는 launch-bound면 발행 공백을 포함해 순수 GPU 시간보다 크다(§1). 층당 1 span(M1, 8k×8192)에서는 launch 공백 비중이 가장 커서 r을 **과대**평가할 수 있다 → 박자 판정에 불리한 방향이다(M1 양성 대조 통과는 보수적). chunk begin poll/emit이 `[t_launch,t_sync]` 안에 있는 관측자 비용(rev1 §3-1 공시)도 그대로다.

### 2-3. 양성·음성 대조와 모형 점검 [제안]

| ID | 대조 | 통과 기준 [제안] | 실패 시 |
|---|---|---|---|
| PC-M1 | **양성**: M1 셀에서 `chunk_gpu_ms` < 동거 `decode_fwd_gpu_ms` | M1 셀의 풀(seed 4) r_span 중앙값 < 0.67 **그리고** 짝 q 중앙값 < 0.67, 그리고 적체 후보 span 비율 < 10% | r 중앙값 ∈ [0.67,1.0): "경계". ≥ 1.0: **박자 영역이 budget 512만으로는 실현되지 않았다**(발행 깊이 제한 E2 필요 → branch). 어느 쪽이든 CM-4 "적체 영역 관측으로 범위 한정" 문구는 유지되고 "박자에서도 무반응"으로 확장하지 않는다 |
| NC-65536 | **음성(측정기 검증)**: (D44, 65536) 2k·8k 셀은 r ≥ 5 예측 | r_span 중앙값 ≥ 1.5 | 통과 못 하면 r 측정기가 적체를 구분하지 못한다 → A-2 전체 `UNRESOLVED_ESTIMATOR`로 표시, 영역 서술 금지 |
| 모형 점검 | phase0_c3 예측 r(M1 0.32; 8192×D24 2k 0.67/8k 0.75; 8192×D44 2k 1.26/8k 1.44; 65536·D92 셀 r 5.0–59)과 r_obs의 비 | r_obs/r_pred ∈ [0.5, 2.0] | 밖이면 "span 시간 모형 오차"로 기술(η·T2 형태 A1–A5 재검토 입력). 판정 아님 |
| 짝 조인 | 짝짓기 성공률 | rev1 §3-2 두 실패율 병기, 조인 성공 span이 셀당 < 30개면 그 셀 r을 "표본 부족"으로 표시 | — |

### 2-4. 경계 4셀(budget 8192 × D24/D44, 2k·8k)의 영역 판별 분석

- 이 4셀은 rev1 격자에 **이미 있다**(셀 추가 0). A-2는 위 §2-2를 적용해 span별 분류 비율을 보고하는 것이 전부다.
- 필드 충분성: `chunk_gpu_ms`(event 쌍)·`decode_fwd_gpu_ms`·`decode_iter_seq` 짝 — **모두 있다.** 부족한 것: (i) 배치 안 요청 수가 1이 아니면 `extend_tokens`가 합이라 `forward_count`가 달라진다(`n_reqs`·`extend_tokens`로 층화 가능, 층화 보고) (ii) launch-bound 분리는 불가(§2-2-6).
- 예측(데이터 전 등록): 2k×D24 r=0.67·8k×D24 0.75(경계, ρ 상단에서 박자 쪽) · 2k×D44 1.26·8k×D44 1.44(경계, 적체 쪽). 이 4셀의 목적은 영역 지형 한 칸을 **분류 비율로 채우는 것**이며, 실제 결과가 예측과 반대여도 (정책 의미가 없는) 기술값이다.

## 3. A-3 — B1 크기 블록 (신규 Job D-4)

### 3-1. 왜 새 구성이 필요한가 [측정·코드]

- phase0 C2: VESSL 기존 run에서 span 중간 전환 0건(fixed 정책/anchor 고정 dyn이라 **기회 없는 0**). 전환 중 drain 53건 전부 빈 prefill stream, 최대 0.153 ms.
- 코드: 컨트롤러는 `_slo_on ∧ running 비어 있지 않음 ∧ (split_prefill_batch 또는 recheck) ∧ not wait_prefill_kernel_done`일 때만 평가된다(`:1315-1319`). budget 65536·8k 프롬프트면 56층 ÷ 8층/chunk = 7 chunk가 ≈7 반복에 다 발행되고 WAIT에 들어가므로 **평가 기회는 span당 ≤ 6반복**이다. budget 8192·8k면 `forward_count==1`이라 56 chunk가 반복당 1개씩 발행되어 평가 기회가 **span당 ≈ 55반복**(약 9배) [유도]. 이것이 main 코드만으로 B1 기회를 증폭하는 유일한 레버다(`budget`은 boot 구성).
- 따라서 B1 = 비율(operating point에서 in-flight 전환이 얼마나 자주 일어나는가) × 비용(일어났을 때 drain ms)로 분해해 두 구성에서 따로 잰다.

### 3-2. 구성

- **Job D-4**(rev1 D-3과 분리 — D-3 하드캡 5.9 h에 더하면 6 h 초과, §7). 동적 arm 하나만(bind+GATE, 정책 PREREG Job-TUNE이 고른 Nano 재튜닝 구성, `PDMUX_SLO_*` env는 D-3 arm 1과 동일 바이트). **P-6 승계: D-4도 Job-CAP1·Job-TUNE 회수 뒤에만 제출.** 계측 ON(PHASE_EVENTS+DIAG+probe). static/OFF 짝 없음(§3-6 잃는 범위).
- **워크로드 W-B1** [제안]: **8k 고정 프롬프트(random 8192)·출력 512·`ignore_eos`**, rate 교대 HI/LO(LO = HI/4, 정책 PREREG 비율), **HI = 0.15 req/s**(Nano 8k shape B SLO 용량 0.15–0.22 req/s [측정; 구간·병기 조건으로만 인용]의 하단, cliff 회피) × 24 요청(160 s), **LO = 0.0375 req/s × 6 요청**(160 s), 2 라운드 + 꼬리 30 s ≈ 11.4 min/run. HI/LO 절대값은 정책 PREREG의 용량 환산(Nano λ\*, addendum)이 나오면 그 비례 규칙으로 **갱신**한다(비율 1/4 유지). trace hash·workload seed·server seed는 seed마다 고정.
- **boot 8개** (seed 4 × 2 구성, boot마다 seed 1개 → 재부팅):
  - **B1-op**: budget **65536**(정책 PREREG의 운영 구성) × W-B1 — "운영점에서 in-flight 전환이 얼마나 자주 나는가".
  - **B1-max**: budget **8192**(기회 증폭, rev1 격자에 이미 있는 값) × W-B1 — "일어났을 때 drain이 얼마나 긴가".
  - 두 구성의 boot 순서는 블록 무작위(계획 해시 파생), 같은 Job 안.
- 공통 고정: cudagraph ON, backend flashinfer, mem 0.6, max-running 48, radix OFF, split config yml은 budget 한 값만 다르다(그 외 같은 바이트).

### 3-3. B1 분석 정의

drain 이벤트 중 **전환 drain = `stream_idx_from ≠ stream_idx_to`**(rev1 §3-3; reason `slo_switch` 또는 SLO 하에서 `adjust`로 찍힌 것 포함).

| 지표 | 정의 | 필드 |
|---|---|---|
| 전환 수 | seed·구성별 전환 drain 개수 | drain |
| **in-flight 전환** | 전환 drain 중 `split_present ∧ prefill_chunks_unresolved ≥ 1` | drain |
| **in-flight 전환 비율** `ρ_if` | in-flight 전환 / 전환 전체 (풀 개수 + seed별 값, Wilson 구간은 기술용) | drain |
| 보조 정의(일치 점검) | `prefill_sync_ms > 1.0 ms` [제안; 빈 stream 상한 0.153 ms [측정] 위 여유] | drain |
| **기회 수** | `decode_iteration`에서 `prefill_in_flight=True ∧ prefill_kernel_done_wait=False`인 반복 수(= 컨트롤러 평가 기회의 상계) | decode_iteration |
| 기회당 in-flight 전환율 | in-flight 전환 수 / 기회 수 (B1-op vs B1-max 비교, 증폭 배율 실측) | 둘 |
| **drain ms** | `prefill_sync_ms + decode_sync_ms`(총), 두 항 각각. 분포 p50/p95/p99 | drain |
| 깊이 공변량(M7 부분) | `prefill_chunks_unresolved`, 미발행 층 `num_layers − split_index` | drain |
| 회귀 입력 [유도] | in-flight drain에서 `prefill_sync_ms` ~ 직전 `prefill_chunks_unresolved`개 chunk의 `chunk_gpu_ms` 합(같은 span의 최신 chunk들, resolve된 것) — 상한 대조용 | drain + chunk |
| decode step 기준 | 같은 구성 `decode_launched=True` 반복의 `decode_fwd_gpu_ms` 중앙값 `s_dec`(동거 step). 보조: `decode_iteration`의 `t_sync−t_launch` 중앙값 | probe, decode_iteration |
| **drain/step 비** | 전환 drain 총 ms 중앙값 ÷ `s_dec` (전체 전환 기준 `ρ_all`, in-flight 조건부 `ρ_if_med`) | — |
| TTFT/ITL 몫 | rev1 §3-3 `share_TTFT`·`share_ITLtail`(lag p99 창)을 같은 정의로 계산 — **손실이 아니라 겹침**이다 | drain, rids, client |

- 이벤트 폐기·미집계 공시: `n_chunks_dropped`, `phase_diag_close` 카운트를 구성별로 보고. 폐기가 있으면 `prefill_chunks_unresolved`는 하한이다.
- 전환 직후 decode 간격(전환이 ITL에 미친 효과)은 이 계측으로 **직접 못 잰다**(전환 직전·직후 반복의 동거 상태가 섞임) — `decode_iteration` 간격을 전환 전후 5반복 창으로 기술값만 보고하고 귀속하지 않는다.

### 3-4. 양성 대조와 안 켜질 때 [제안]

| ID | 기준 | 통과 | 실패 시 처리 |
|---|---|---|---|
| **PC-B1** | B1-max 풀(seed 4)에서 **in-flight 전환 ≥ 20개** 그리고 **seed 4개 중 ≥ 3개에서 각각 ≥ 1개** | B1 크기 지표를 "측정됨"으로 보고 | **`B1_NOT_EXERCISED`(기술 표시)**: B1 크기는 **미관측**이며 "작다/무시 가능"으로 읽지 않는다. 관문 G-B1은 `UNMEASURED`. 안 켜진 이유를 분해 보고: ① 전환 총수(전환이 거의 없었는가), ② 기회 수, ③ 전환이 전부 admission 시점(빈 stream)이었는가. 이어서 후속은 **branch의 강제 전환 probe**(`impl-defects-e123`/diag v2)로 넘긴다(§6) — main digest로 더 강제할 방법은 없다 |
| PC-B1-tot | 전환 총수 ≥ 30(B1-max 풀) | `ρ_if`를 Wilson 구간과 함께 보고 | 미만이면 `ρ_if`는 `UNDERPOWERED`(표 표시만) |
| PC-B1-amp | B1-max의 기회당 in-flight 전환율 vs B1-op | 증폭 배율 보고(기대 ~9배 [유도]) | 배율 ≈ 1이면 "기회 증폭이 실제로 작동하지 않음" 공시 |
| PC-B1-op | B1-op 풀에서 in-flight 전환 ≥ 1 | `ρ_if`(operating) 보고 | 0건이면 B1-op의 `ρ_if`는 **상한만**: Wilson 상한(전환 총수 기준)과 기회 수를 병기, "0건"을 "0"으로 쓰지 않는다 |

### 3-5. 관문 G-B1·G-M6 (계획 A5 step 4 문턱 — 모두 [제안], 사용자 확인 필요)

계획 A5 step 4의 세 문턱을 그대로 등록한다. 판정은 **진단 Job의 관문 지표**이며 정책 주장이 아니다. 결과는 사용자 결정으로만 정책 Job (나)에 `time + inflight 2 + handoff` arm 추가를 제안하는 입력이 된다(결정 38: 데이터가 기다리는 곳에만 설계를 얹는다, §8).

| 관문 | 지표 | 문턱 [제안, 계획 A5-4] | 평가 위치 |
|---|---|---|---|
| **G-B1a** | 전환당 drain 총 ms 중앙값 ÷ `s_dec` (`ρ_all`) | **> 2** | D-4 B1-op 풀(운영 구성) 우선, D-3 DIAG ON 풀(rev1)도 병기 |
| **G-B1b** | in-flight 전환 비율 `ρ_if` | **> 10%** | B1-op 풀 우선, D-3 병기 |
| **G-M6** | M6 유휴 / prefill 시간 | **> 5%** | §4-2, D-1b/a 정적 셀 풀 + D-3/D-4 동적 |

- 관문 결과 상태: `CONTINUE`(문턱 초과) / `STOP`(모두 측정됐고 모두 문턱 이하) / `UNMEASURED`(양성 대조 실패 또는 표본 부족). **`UNMEASURED`는 `STOP`으로 간주하지 않는다**: 계획 A5의 "중단(모두 미만 → E1–E3 기본 OFF 보관, 'B1은 결과를 좌우하지 않았다' 공시)" 분기는 모든 항이 `측정됨`일 때만 쓴다.
- **구성 범위 규칙**: 문턱 초과가 **B1-max에서만** 나오고 B1-op/D-3에서는 아니면 상태는 `CONTINUE_AMPLIFIED_ONLY`로 따로 표기한다(증폭 구성은 운영점이 아니다). 이때 정책 Job arm 추가 여부는 사용자 결정이다.
- 문턱값 자체가 [제안]인 이유: 계획서 값이며 근거 데이터가 이 Job 이전에 없다. HE0 핵심 비교의 격차(6.8%)를 B1 상한 ≤0.72%가 닫지 못한다는 phase0 C1′ 정보와 정합하는지는 봉인 시 사용자가 확인한다.

### 3-6. 잃는 범위 (B1)

- 동적 arm **1개 구성**(bind+GATE tuned)만. no-gate·slo(KISTI에서 전환이 훨씬 잦았던 arm: run당 11·15회 [측정, phase0 C1])는 측정하지 않는다 → B1이 가장 큰 arm에서의 크기는 못 본다. 추가 시 D-4에 boot +8(+2 h)이 필요 [미정, §10].
- B1-max(budget 8192, 8k random)는 **운영 구성이 아니다**: drain 깊이 분포가 65536과 다르다(깊이 얕고 span당 chunk 56개). 비용 항(`drain ms`)의 깊이 의존성은 `prefill_chunks_unresolved` 공변량으로만 외삽한다.
- static 대조 없음: 같은 Job의 static drain 기준은 rev1 D-3 static boot(HE0 trace)에 있으나 Job이 달라 비교하지 않는다. B1은 동적 arm 안 기술값이다.
- Nano·flashinfer·A100-SXM4 한정. KISTI Zamba2 수치(전환 수 등)와 결합하지 않는다.

## 4. A-4·A-5 — M6·M5 (기존 필드 분석)

### 4-1. 공통 전제

대상: rev1 D-1a/D-1b ON 셀(static 비sticky, DIAG ON)과 D-3·D-4의 DIAG ON arm. 비sticky static에서 `adjust` drain은 분할이 바뀔 때만 발화한다(rev1 §3-3 정정 승계). 기판(Job)이 다른 값을 한 풀로 합치지 않는다.

### 4-2. M6 — decode가 빈 뒤 D SM 유휴 (B2 크기)

- **B2 drain 정의**: `reason == "adjust" ∧ running_bs == 0(drain 시작 상태) ∧ split_present ∧ prefill_chunks_unresolved ≥ 1`. (코드: 이미 발행된 span만 기다린다. 미발행 층은 drain 뒤 idx 0에서 돈다 — phase0 §0 정정.)
- **D SM 유휴 구간**: 시작 = 직전 `decode_iteration`의 `t_sync`(decode 배치가 비는 반복의 끝; 그 반복이 마지막 decode), 끝 = `t_drain_end`. 유휴 ms = `t_drain_end − t_sync(last decode)`; 포함 항: drain 시작 전 루프 지연 + `prefill_sync_ms` + `decode_sync_ms`. 이 구간 동안 decode SM 컨텍스트는 일이 없고 이미 발행된 chunk만 P SM에서 돈다. **단 drain 뒤 미발행 층은 idx 0(108 SM)에서 돌아 D SM 유휴가 더는 아니다**(phase0 §0).
- **M6 비율** = Σ(유휴 ms) / Σ(prefill 시간). prefill 시간 = 같은 풀 span들의 `prefill_span_end.timestamp − prefill_span_start.timestamp` 합(span_end는 최대 1반복 늦게 관측, rev1 §3-1). 분모를 (i) 셀의 모든 span, (ii) decode 존재 span으로 나눈 두 버전을 병기한다. SM 가중판: 유휴 ms × `decode_sms_from`을 `prefill_sms`·span ms로 나눈 값은 기술 보조.
- **양성 대조 PC-M6** [제안]: D-1b 정적 셀 풀(**D{24,44} × budget {8192,65536} 4 구성 × seed 4 = 16 셀-run**; D92 셀은 보고하되 PC 집계에서 제외)에서 B2 drain **≥ 10개**. 기대 [유도]: 8k·0.1 req/s·b≈1이라 decode-empty가 span의 약 7%(rev1 §6), 셀당 span ≈ 40 × 4 seed. 못 켜지면 **`B2_NOT_EXERCISED`**: M6은 미관측(0이 아님), G-M6 `UNMEASURED`; N3 span의 44.8–45.8%가 decode-empty라는 phase0 [유도 상한]은 양성 실측이 아니라 상한으로 남는다.
- 관문: G-M6 = (풀 Σ유휴 / Σprefill 시간) > 5% [제안].
- 공시: 이 지표는 TTFT 이득이 아니다(phase0 §2-1: B2 비용은 ITL≈0, TTFT에만 있을 수 있음). "쌓인 span이 P SM만 쓰고 D SM이 노는 시간"의 크기이며, 이를 TTFT 손실·goodput으로 환산하지 않는다.

### 4-3. M5 — prefill 진행 중 도착한 요청의 입장 대기 (B3 크기)

- 코드 사실: `split_prefill_batch`가 있으면 새 prefill을 입장시키지 않는다(`:1218-1238`); 선점 없음.
- **요청별 대기**: `t_send`(client abs mono) → `t_admit`(그 요청 rid를 포함한 `prefill_span_start`의 telemetry 시각). `W_adm(r) = t_admit − t_send`(상계: HTTP·토크나이즈·루프 지연 포함).
- **차단 대기** `W_blk(r)` = `[t_send, t_admit]` 중 **다른 span의 in-flight 구간** `[span_start_k, span_end_k]`(k ≠ r의 span)와 겹친 시간의 합. 비차단 기준선 `W_base` = in-flight 구간과 겹치지 않는 요청의 `W_adm` 분포(루프 틱+토크나이즈 오차의 대리). 보고: `W_blk`의 p50/p95/p99, TTFT 대비 몫(`W_blk / TTFT`), 도착 시 in-flight였던 요청 비율(`share_blocked`), `queue_len`(span_start 시점) 분포.
- **양성 대조 PC-M5** [제안]: 8k 셀 풀과 M1 셀에서 **in-flight 중 도착해 입장이 지연된 요청 ≥ 30개**(M1은 prefill 점유율 ≈0.65 [유도]라 기대 풀 크기 충분). 미충족이면 M5는 `UNMEASURED_FEW`로 표시.
- **한계(데이터 전 등록)**: ① 서버 대기열 진입 시각이 기록되지 않는다 → `W_adm`은 상계이고 `W_base`로만 보정한다. ② client↔server 시계 정합은 rev1 §3-3 lag 분포(p50/p99)로 창을 넓혀 민감도(0·2×p99) 병기. ③ "차단이 없었다면"의 반사실(동시 두 prefill 허용)은 이 계측으로 알 수 없다 — M5는 **대기의 크기**이지 손실이 아니다. ④ 입장 순서는 SGLang 스케줄러 정책(FIFO 등)의 영향도 함께 반영한다.
- 관문: M5에는 계획 A5 문턱이 없다. 문턱을 새로 만들지 않는다(보고만).

## 5. 라벨·산출물 요약

- 새 **라벨 0**. 산출물 표: (1) 영역 지표 표 13셀(§2), (2) PC/NC 대조 통과표(PC-M1, NC-65536, PC-B1 계열, PC-B1-op, PC-M6, PC-M5), (3) B1 크기 표(`ρ_if`, drain ms, 기회당 전환율, `ρ_all`) (4) M6·M5 표 (5) 관문 지표 3개의 상태.
- rev1의 `DRAIN_OVERLAP_*`·`OBSERVER_*`는 D-3(DIAG ON/PHASE-only/OFF) 풀에서만 내며 D-4와 섞지 않는다. D-4의 `share_TTFT` 등은 기술값이다(군집 seed 4, 단일 arm, 짝 없음).

## 6. 연기 — main digest로 못 재는 항목 (branch `impl-defects-e123`, diag v2)

| 항목 | main 계측으로 되는 것 | 연기 이유 | 연기 대상 |
|---|---|---|---|
| **M2** span GPU 시간 ~ 층 종류별 회귀(η_X) | fc=1 셀(M1, 8k×8192)의 `chunk_gpu_ms`는 `split_index`로 층 종류에 매핑 가능 — **원자료는 어차피 telemetry에 남는다.** 회귀·η_X 추정은 **이 addendum에서 등록하지 않는다** | (i) 소비처 E1(`PDMUX_SPAN_POLICY=time`, η_X 설정값)이 branch에만 있다 (ii) 설계가 없다: launch-bound 공백(§2-2-6)·동거 decode 간섭(ρ)이 `chunk_gpu_ms`에 섞여 층 종류 효과와 분리되지 않는다 — 등록 없는 사후 회귀는 금지(게이트 3 방향). diag v2 PREREG에서 층 종류 균형·간섭 분리 설계를 먼저 등록한 뒤 이 원자료를 쓴다 | diag v2 |
| **M3** span 구성별 ρ(decode step이 겹친 span 종류에 따른 간섭) | `decode_iter_seq` 조인으로 단편적 관측만 가능(층 종류 = M2 매핑 의존) | M2의 층 종류 매핑·η_X가 선행이며, 충분한 층 종류 교차 표본은 fc=1 셀 2개뿐이라 표본·교락(층 위치 순서 = 층 종류 순서) 문제가 설계 없이는 해결 안 됨 | diag v2 |
| **M4** 반복별 host 시간 분해(수신·입장·span 발행·decode sync 대기·결과 처리) | 일부 단편(`issue_host_ms`, `t_launch/t_sync`, drain host ms)뿐 | 분해 필드가 **main에 없다**(계획 A3/E5: 새 계측). 필드를 추가하면 엔진 digest가 바뀌어 rev1 P-8 위반 | branch(E5) |
| **M7** 전환·drain 시점의 발행 깊이 | **부분 가능**: drain 이벤트의 `prefill_chunks_unresolved`(미완료 chunk 수), `num_layers − split_index`(미발행 층)을 §3-3 공변량으로 쓴다(별도 청구 없음) | 완전판(미완료 **span** 수, handoff 시 옮길 남은 층, handoff 플래그)은 E3/E5 필드가 필요하다. 이 addendum이 쓰는 부분 M7은 "chunk 단위·poll 시점 근사"이며 M7 전체를 닫지 않는다 | branch(E5) |

연기가 막는 것: E1(시간 기준 span)의 η_X 설정, E3 handoff 설계 가치 판단(=B1 크기에 의존, D-4가 일부 채움), B5 크기. 이 항목들은 **관문(G-B1·G-M6)이 CONTINUE일 때만** 기다릴 가치가 생긴다.

## 7. Job 구성·시간·GPU-h·사용률 [유도 — 측정 아님]

rev1 §7 시간 모형 승계(boot 200 s + warm-up 60 s = 4.3 min/boot, 셀 오버헤드 15 s, 2k 셀 2.75 min, 8k 셀 5.75 min).

| Job | rev1 | addendum 추가 | 합 | 예상 h | 하드캡(×1.2) | 사용 GPU 시간(사용률) |
|---|---|---|---|---|---|---|
| **D-1a** (2k) | 7 boot, 1.78 h | **+1 boot**(M1): 4.3 + 4 × 2.75 = 15.3 min = +0.26 h | 8 boot | **2.04** | 2.4 | 88 min / 122 min = **72%** |
| **D-1b** (8k) | 7 boot, 3.97 h | 0 (M6/M5/영역은 분석) | 7 boot | 3.97 | 4.8 | 87% (rev1) |
| **D-3** | 13 boot, 4.93 h | 0 | 13 boot | 4.93 | 5.9 | 81% (rev1) |
| **D-4** (신규, B1) | — | 8 boot × (4.3 boot + 11.4 run) = 125.6 min | 8 boot | **2.09** | 2.5 | 91 min / 126 min = **73%** |
| 합계 | 10.7 h, 27 boot | **+2.35 GPU-h**, +9 boot | | **약 13.0 GPU-h** | 약 15.6 | |

- 전부 `resourcespec-a100x1`(동시 사용 GPU 1). boot 구간은 서빙이 없어 사용률은 셀 시간 비율(72–73%)이며 rev1 D-1a와 같은 boot 지배 구조다. Job당 ≤ 6 h, 모든 Job 하드캡 ≤ 5.9 h(rev1 D-3) 유지.
- D-4 크기 절약안(선택 [미정]): B1-op 4 boot를 생략하고 B1-max만(4 boot, 62 min = 1.0 h): 운영점 `ρ_if` 정보를 잃는다(§3-6과 별도). **이 경우 G-B1b는 B1-max만 보고 `CONTINUE_AMPLIFIED_ONLY`/`UNMEASURED`만 가능** — 비권고.
- 재개 단위·`vessl_jobs.tsv`·`fetch.sh`(DONE+sha256)·자동 재시도 없음: rev1 §7 동일.
- 이 Job들이 채우는 COSTMODEL 칸: **CM-4(동거 간섭·정지 항 — 영역 r, B1·B2·B3 drain 항)**, 부수로 **CM-2(span 시간 `T_span`의 직접 측정, `chunk_gpu_ms`)**. 정책 칸 아님.

## 8. 결정 38 정당화 — 각 셀이 기다리는 데이터

결정 38(측정 우선, 설계 쌓지 않기): 새 설계는 "기다리는 데이터가 있을 때"만. 이 addendum에서 각 요소가 기다리는 데이터:

| 요소 | 기다리는 데이터 | 안 재면 막히는 것 |
|---|---|---|
| M1 셀 | 박자 영역(r<1) 첫 직접 관측. 기존 측정은 전부 r≫1 또는 span 1개(phase0 C3) | CM-4 "동거 간격이 span 시간에 반응하지 않음"의 범위 문장(적체 한정으로 좁힐지, 박자에서도 확인할지); 감사 r1 실험 4 |
| 경계 4셀 분석 | 모형이 가르지 못한 4셀(r 0.67–1.44)의 영역 | 현 diag 격자의 "박자 셀 0" 주장 확정 여부 |
| D-4 B1 | B1의 비율×비용 크기(지금은 상한 ≤0.72%만, VESSL 0건은 기회 없는 0) | 계획 A5 step 4 관문 G-B1; E3 handoff arm이 정책 Job (나)에 필요한지 |
| M6 | B2(decode-empty drain 중 D SM 유휴)의 직접 크기(지금은 상한 p50 849 ms 유도뿐) | G-M6; E2(발행 깊이 제한)가 E3보다 우선인지(phase0 §2-2) |
| M5 | B3(입장 직렬)의 대기 크기 | 선점(E4) 설계 의뢰 가치 (E4는 이번 범위 밖, 설계 메모만) |

이 중 어느 것도 데이터 없이 엔진 설계를 확장하지 않는다. 관문이 `STOP`이거나 양성 대조가 실패하면 branch 작업(E1–E3)은 **기본 OFF 보관**이며 추가 설계를 쌓지 않는다.

## 9. 금지·허용 문안 (rev1 §12에 **추가**)

**추가 금지**
- "박자 영역에서도 동거 간격이 span 시간에 반응하지 않는다/한다"를 PC-M1 실패 또는 표본 부족 상태에서 쓰는 것; "CM-4 일반형이 반증/확정됐다" (기울기 판정은 이 addendum에 등록되지 않았다).
- `B1_NOT_EXERCISED`·`B2_NOT_EXERCISED`·`UNMEASURED`를 "B1/B2는 무시 가능/작다/없다"로 읽는 것; "VESSL에는 B1이 없다". 0건은 기회 수와 함께만 쓴다.
- B1-max(budget 8192) 수치를 운영점(65536)의 B1 크기로 쓰는 것; B1-max와 B1-op를 한 풀로 합치는 것.
- `W_blk`·M6 유휴·`share_TTFT`를 "손실/비용/TTFT 이득/goodput"으로 환산하는 것. 관문 상태를 "구현 결함이 HE0를 좌우했다/좌우하지 않았다"로 쓰는 것(그 공시 문구는 정책 PREREG와 사용자 결정 몫).
- r_obs를 정책 순위·static/dynamic 비교의 근거로 쓰는 것; 13셀 r을 Job 간 비교 표에 섞는 것.
- 관문 `CONTINUE`를 "handoff/time-span이 이긴다/필요하다"로 쓰는 것(arm 추가 제안일 뿐).
- 연기 항목(M2·M3·M4·M7 완전판)의 수치를 이 Job 결과로 인용하는 것.

**허용 문안 틀(추가)**: "VESSL A100-SXM4 · Nano-9B-v2-Base · flashinfer · cudagraph ON · PHASE+DIAG ON 조건의 <Job>·<셀>에서, 짝지어진 chunk GPU 시간 대 동거 decode step 비 r_span의 중앙값은 <값>(seed 범위 <…>)이었다. 기술값이며 영역 라벨·정책 판정이 아니다." / "…양성 대조 PC-<id>는 <통과/실패>였다. 실패라면 해당 크기는 미관측이다."

## 10. 하네스 추가 항목·봉인 체크·열린 결정

**rev1 §9 하네스에 추가(코드 0, 감사 뒤 작성)**: ⑧ 영역 지표 집계(짝 조인·r_span·분류 비율·보조 신호·PC/NC 대조) ⑨ B1 집계(전환 분류, in-flight, 기회 수, drain/step 비, 회귀 입력) ⑩ M6·M5 집계(유휴 구간 구성, `W_blk` 겹침 계산) ⑪ W-B1 클라이언트(HI/LO, seed 고정, abs-mono) ⑫ budget 512 config·D-4 config yml(budget만 다른 바이트 동일 쌍) 및 env 누출 검사. **변이 시험**: (a) r_span 분자를 `issue_host_ms`로 바꿔치기, (b) 분자·분모를 서로 다른 `decode_iter_seq`로 짝지음, (c) in-flight 판정에서 `prefill_chunks_unresolved`를 무시, (d) 유휴 구간 시작을 `t_drain_start`로 바꿈 — 모두 검출돼야 한다.

| # | 봉인 체크 항목 | 상태 |
|---|---|---|
| 1 | rev1 P-1…P-9 + **P-A1**(512 smoke gate 추가, engine-porter) | 대기 |
| 2 | 하네스 ⑧–⑫ 구현·CPU 시험·변이 | 대기 |
| 3 | addendum delta(§2–§4 정의·§3-5 문턱)에 대한 하네스층 감사 | [미정 — rev1은 감사 상한 소진(결정 50)이었다. 새 정의 영역이라 delta 한정 1회 권고, 또는 사용자 면제] |
| 4 | 봉인(`SEALED_rev2.txt`는 **봉인 시점에 새로 만든다**, 지금은 만들지 않음) | 대기 |
| 5 | CPU preflight → `launch.sh` dry-run → 사용자 승인 → `--submit` (D-1a 재제출 규모 확인, D-4는 P-6 뒤) | 대기 |

**열린 결정 [미정]**
1. G-B1a/b·G-M6 문턱 3개 확인(계획 A5 값). 특히 `ρ_all`의 분모를 `decode_fwd_gpu_ms` 중앙값으로 한 것과 B1-op 우선 평가.
2. PC 기준값(PC-B1 ≥20 in-flight·≥3 seed, PC-M6 ≥10, PC-M5 ≥30, PC-M1 r<0.67) — 전부 [제안].
3. D-4 규모(8 boot 권고 / 4 boot 절약안 / no-gate·slo arm 추가 +2 h).
4. addendum 감사 1회 여부(체크 #3).
5. W-B1 HI/LO 절대값: 정책 PREREG Job-CAP1 용량 회수 뒤 비율 1/4로 갱신(그 전엔 0.15/0.0375).
6. A-1을 D-1a에 붙일지 별도 소형 Job(1 boot, 0.26 h)으로 낼지 — 같은 Job 안 대조가 필요한 것은 (D44, 512) vs (D44, 65536)의 NC-65536 비교이므로 D-1a 포함을 권고.

## 11. 잃는 범위 (전체)

- M1은 2k·D44·budget 512 **한 점**이다. 8k의 fc=1 점은 rev1 격자의 8192 셀이 대리한다(budget 8192 ÷ 8192 = fc 1). D 축의 박자 점(D24/D92)·budget 중간값은 없다(rev1 X-4와 같은 사유).
- 경계 4셀은 분류 비율만 낸다. 영역을 가르는 모형 오차 요인(A1–A5)은 분리되지 않는다.
- B1은 §3-6, M6은 정적 non-sticky 셀(+동적 arm)만, M5는 서버 enqueue 시각 부재(상계).
- rev1이 잃은 범위(Zamba 제외·KISTI 설정 재현 0셀·static OFF 짝 없음)는 그대로다.
- 이 Job들은 **branch 코드가 없으므로 E1(시간 span)·E2(깊이 제한)·E3(handoff) 효과를 시험하지 않는다.** "구현 결함을 고쳐도 변하지 않는다/변한다"는 이 addendum에서 말할 수 없다.
