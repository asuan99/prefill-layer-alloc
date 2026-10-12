# 규칙층 델타 감사 판정서 — PREREG_MULTIGPU_2026-10-12 rev2 (M2.5)

> 감사자: claims-auditor(읽기 전용, GPU 0), 2026-10-12. 판정문 요지를 메인 세션이 저장했다(표 수치·처방·caveat 문안은 원문 그대로).
> 직전 판정서 표·처방은 등록 상수로 쓰지 않고 재계산했다(게이트 #110).

## 0. 판정 — `MIXED`

| 범위 | 등급 | 이유 |
|---|---|---|
| **M2g 규칙(§3)** | **`GO-with-caveats` 유지** | rev2 M2g 변경에서 반전 없음. 하네스층 불일치 1건 caveat(MGP-23) |
| **M3 규칙(§4)** | **`NO-GO`** | D2 해소. D1은 형태만 바뀌어 재발. 새 死因 **D3**(N3+N2), **D4**(N3), **D5**(N3), **D6**(N2, 경미). D3·D4·D5 각각만으로 G의 인용 가능 정의역이 ∅이거나 N의 감소함수. 수리는 모두 GPU 없이 가능 |
| M4 | 판정 안 함 | — |

"M2.5 통과"·"M3 거의 GO"로 요약하지 않는다.

## 1. 직전 지적 처리
| 항목 | 상태 |
|---|---|
| D1 | 형태만 바뀌어 재발: 대체 술어가 포화가 아니라 PD-mux 입장 기계의 항등식을 잰다(D3·D4) |
| D2 | 해소(규칙층). agnostic 분할 idx {1,2}, `pdmux_r2.yml` D44 = idx 4 ∈ {1..4}. idx==2 변이 검사는 하네스층 |
| MGP-1 | 해소. 하네스 불일치 1건(MGP-23) |
| MGP-10 | 해소 |

## 2. 코드로 확인한 사실 (dev tree `sglang_engine_dev_multigpu/python/sglang/srt`)
- 줄 형식: Prefill `observability/scheduler_metrics_mixin.py:389-395`, Decode `:608`, `:657`.
- 로그 rank: `is_stats_logging_rank = attn_tp_rank == 0`(`:119`). TP는 rank 0만(접두 ` TP0`, `managers/scheduler.py:3593-3594`). DP(dp-attention 없음)는 복제본별, 접두 ` DP{k}`(`:3585-3586`).
- 빈도: Decode 줄은 `forward_ct_decode % decode_log_interval == 0`(`:533-534`), 기본 **40**(`server_args.py:418`). runner는 `--decode-log-interval`을 넘기지 않음(`:136-163`). r2c(KISTI 907959)는 1. Prefill 줄은 배치마다(`scheduler_output_processor_mixin.py:328`). 타임스탬프 초 해상도(`utils/common.py:1136-1141`, `SGLANG_LOG_MS` 미설정 시).
- Prefill 줄 `#running-req`는 merge 전 수(`scheduler.py:2546`), 입장은 running < cap일 때만(`:2309-2313`, `:2436-2438`) → **cap 구속에서 Prefill 줄은 구조적으로 cap 미도달**(KISTI 907959 I3a 24/24줄 최대 47, Zamba2 r0a fused 445/445줄 최대 47).
- PD-mux 입장 기계(`multiplexing_mixin.py`): 분할 prefill 진행 중 새 입장 없음(`:1219-1220`), 입장(`:1284-1290`)이 finished 필터(`:1293`)보다 먼저, 진행 중 prefill 요청은 merge(`:1528-1534`) 전까지 `running_batch` 밖 → 입장 사건마다 Decode 줄이 cap−k로 최소 2–3 반복(KISTI 907959 1-요청 입장 6/6건 정확히 3 반복).
- fused: 완료 → filter → prefill 반복(decode 없음) → merge → decode at cap. Decode 줄은 "입장 가능한 대기열이 없을 때"만 표본(코드 추론).
- fused cap 도달성: radix OFF면 mamba cache = max_running_requests(`model_runner_kv_cache_mixin.py:224-229`), `_resolve_max_num_reqs` = min(cap, ≥2048)(`:860-876`). cudagraph max bs A100 tp<4 256, tp≥4 512(`server_args.py:1196-1205`) → cap 96/192/384 덮음.

## 3. 반전 시험 (감사자 예보 모형, 측정 아님)
A 모형: 반복 단위 이산, τ 25.6 ms(KISTI 907959 cap 48, 39 Decode줄/s)·35 ms, OSL 512, C = 48N, Poisson HI 1.6·N·2.99 / LO 0.6·N·2.99, 창 170 s(웜 60 s 뒤), seed 0–3, PD-mux 입장 구멍 b = max(2,k)(1-요청 cap−1 3줄 재현). B 모형: 단일 prefill 서버(배치 ≤ 2), S = ρ\_arm/x\_lo, τ\_B 37 ms, 간격 40.

| 자유 표면 | 판정 변화 | 근거 |
|---|---|---|
| 샘플 줄 종류(문언 Decode+Prefill ↔ Decode만) | **반전**(A HI fused) | 문언·간격 40: fused 0.23/0.16/0.11/0.11–0.14(N=1/2/4/8), PD-mux 0.16–0.17/0.08/0.03–0.04/0.04–0.12 → 전부 < 0.80. Decode만 fused 1.00. 실측 Zamba2 r0a 간격 40 Prefill:Decode fused 445:68, agnostic 277:152 → 상한 0.133/0.354. KISTI 907959 간격 40 모사: 문언 0.587, Decode만 0.732 → D3 |
| decode_log_interval(미등록) | **반전** | fused N=1 0.23(K=40) ↔ 0.92–0.93(K=1), TP8 0.11–0.14 ↔ 0.83–0.85 → D3 |
| `#running-req` = cap 술어(Decode만) | **반전**: PD-mux A `HI_SATURATED` 도달 불가, 정의 여부가 N 감소함수 | PD-mux N=1 0.73–0.77 / TP2 0.57–0.65 / TP4 0.36–0.59 / TP8 0.24–0.38, fused 1.00. X=0.70도 N=1만 통과. 실측 KISTI 907959 q>0 조건부 0.848 → D3 |
| Y = 0.90 at x\_lo, shape B | **반전**: agnostic B `LO_UNSATURATED` 도달 불가 | PD-mux q==0: ρ 0.6 → 0.77–0.83, 0.7 → 0.57–0.78, 0.8 → 0.47–0.66, M/D/1 ρ=0.6 → 0.729. fused ≈1. 실측 Zamba2 r0a agnostic running ≤ 7/48인데 q>0 76/152 = 0.50, fused 4/68 = 0.06 → D4 |
| Y = 0.90, A LO | TP8 경계(0.84–0.91) | D3 이후 실익 없음 |
| G5(c) 정확 일치 | **반전**: 모든 config `MEMORY_MISMATCH` → 인용 정의역 ∅ | Zamba2 r0a fused 327637 vs PD-mux 327536(−0.031%), boot 지터 857477 295520/295521, Nemotron 93c3cca 2721290 vs 2721287 → D5 |
| FRONTEND `ps %cpu` | **반전**: 발화 불가 | 수명 평균 상한 170/(170+180–300) = 0.36–0.49 < 0.90 → D6 |
| X 수치 | D3 포함 | 0.50도 TP8 PD-mux 구제 못 함 |
| DP 모든 복제본 | D3 포함 | — |
| 최소 20줄 | 반전 없음 | HI Decode 줄 122–166, B LO ≈110 |
| HI/LO boot 분리 | 반전 없음 → MGP-21 | — |
| 공동 확장·x\_lo | D3 포함, 그 외 MGP-20 | D3 아래 A에서 확장 항상 발화, boot 21회 추가 |
| `M2G_PASS_TABLE` digest | MGP-22 | 등록값은 M2g 전 알 수 없음 |
| seed 색인 공동 재표집 | MGP-24 | — |
| N\_max | MGP-25 | 정의역이 술어로 선택 |
| G0(i) 1800 s vs runner 180×2 s | MGP-23 | `engine_bench_runner.sh:204-213` |

## 4. 死因과 처방

- **D3 (N3+N2)**: shape A HI 술어가 포화가 아니라 입장 기계를 잰다. Prefill 줄 포함 시 모든 arm·N 도달 불가(N3), Decode만이어도 PD-mux는 입장 사건마다 cap−k 구멍, 완료율 ∝ cap/OSL ∝ N(TP) → TP N≥2 PD-mux 구조적 탈락, 정의역이 N의 감소함수. 두 독해 사이 fused 라벨 반전(N2).
- **D4 (N3)**: shape B LO 술어가 arm마다 다른 것을 표본(PD-mux 시간 균등 vs fused 사건 조건부) → agnostic `LO_UNSATURATED` 도달 불가. D3와 합치면 G 어디서도 미정의, P-DP 출력공간에 "거짓" 없음.
- **P-D3/D4 (GPU 0)**: (1) arm 대칭 대기열 증가 통계 g = (송신 창 마지막 1/3 평균 `#queue-req` − 첫 1/3 평균) / ((2T/3) × realized 도착률), Decode 줄만, DP 복제본별(복제본 도착률 = 전체/N). `HI_SATURATED` ⇔ 4 seed 모두 g ≥ 0.10, `LO_UNSATURATED` ⇔ 2 seed 모두 g ≤ 0.05. running==cap은 기술치. (2) `decode_log_interval`(=40)과 `SGLANG_LOG_MS=1` 등록. (3) 변이 3개 KILLED: "running==cap 술어", "Prefill 줄 포함", "K=1 vs K=40 라벨 불변 위반"(PD-mux 합성 로그).
  실현가능성(감사자 모형 예보, 등록 상수 아님): HI g A fused 0.26–0.40, PD-mux 0.21–0.30, B 1.6× 0.25–0.44. LO g A ≈0, B ρ 0.6–0.8 ≤ 0.031, ρ 0.9 ≤ 0.037. 문턱 [0.04, 0.20] 어디서도 모형상 반전 없음. g는 송신 창 안 서버측 대기열이라 drain 꼬리 미포함 → 교훈 9 항등식 아님. 저자가 재도출하고 델타 감사에서 재반전 시험.
- **D5 (N3)**: G5(c) 정확 일치 결정적 실패. **P-D5**: `MEMORY_MISMATCH`는 (i) mamba cache ≠ effective cap(정확), (ii) `max_total_num_tokens` < effective cap × (ISL+OSL)(A 36,864 / B 396,288 토큰/인스턴스 vs 실측 ≈2.72 M, 비구속), (iii) arm 간 상대차 > 1%(관측 0.031%) 중 하나일 때만. 그 외 기술치. 프로세스 검사는 `--query-compute-apps=gpu_uuid,pid,used_memory`, M2g 초두 가시성 프로브, 안 보이면 NVML `memory.used` GPU별 비교 폴백.
- **D6 (N2, 경미)**: `ps %cpu` 수명 평균. **처방**: `/proc/<pid>/stat` utime+stime 1 s 델타, 대상 = launch_server 주 프로세스(TokenizerManager), detokenizer, DP controller.

## 5. caveat (문자 그대로 승계)
- **MGP-17**: "`#running-req` = effective cap 또는 `#queue-req` = 0 비율로 낸 `HI_SATURATED`·`LO_UNSATURATED`·`KNEE_*` 라벨은 인용하지 않는다. PD-mux에서는 분할 prefill 진행 중인 요청이 `#running-req`에서 빠지고, 새 요청이 진행 중 prefill 뒤에 대기하기 때문이다."
- **MGP-18**: "arm 간 `max_total_num_tokens` 차이(KISTI Zamba2 실측 −0.031%)를 G의 원인으로 쓰지 않으며, KV 비구속 조건에서 그 차이만으로 G가 무효라고도 쓰지 않는다."
- **MGP-19**: "`ps %cpu`(수명 평균)로 잰 프론트엔드 지표는 인용하지 않는다. `FRONTEND_SATURATED`가 붙지 않은 것은 구간 CPU 델타로 쟀을 때만 '프론트엔드 비병목'의 근거가 된다."
- **MGP-20**: "x_lo는 fused HI 중앙값 기준이라 PD-mux arm의 LO 부하율은 0.6/G다. G < 1인 셀의 PD-mux `KNEE_NOT_BRACKETED`를 'PD-mux가 불안정하다'로 쓰지 않는다."
- **MGP-21**: "HI와 LO는 서로 다른 boot다. `KNEE_BRACKETED`는 boot 두 개에 걸친 라벨이며, HI boot 자체의 비포화 구간을 보증하지 않는다."
- **MGP-22**: "M3가 대조하는 `M2G_PASS_TABLE.json` sha256 '등록값'은 M2g 회수(fetch.sh DONE+sha256) 뒤 M3 dry-run 계획에 고정한 값이다. 고정 전 M3 실행 결과는 인용하지 않는다."
- **MGP-23**: "runner health 대기(180×2 s)가 등록 G0(i) 1800 s보다 짧은 상태에서 난 boot 실패는 G0(i) FAIL이 아니라 `NOT_TESTED`다."
- **MGP-24**: "기울기 bootstrap의 seed 색인 공동 재표집에서 N 간 결합은 같은 trace가 아니다. 의미 있는 짝은 같은 N 안의 arm 짝뿐이다."
- **MGP-25**: "N_max < 8인 크기·추세 라벨은 N_max와 각 미정의 N의 미정의 사유를 병기하지 않으면 인용하지 않는다(정의역이 라벨 규칙에 의해 선택될 수 있다)."
- MGP-0…MGP-16 유지. MGP-0의 "M3 `NO-GO`"는 rev2에도 그대로 적용.

## 6. 다음 gate
rev3에 P-D3/D4·P-D5·P-D6 반영(CPU만) → 델타 감사(범위: D3–D6, g 0.10/0.05·허용 오차 1% 반전 시험). M2g는 MGP-1…5·MGP-23 승계 조건으로 하네스 작성·하네스층 감사 진입 가능. M2g Job 제출은 M3 rev3 판정과 별개로 사용자 승인 사항.
기판 고지(#11): 구조 사실(입장 구멍 3 반복, Prefill 줄 < cap)은 엔진 코드가 정하고 KISTI 로그로 확인. 비율 수치는 VESSL 값이 아닌 예보. VESSL λ0 서버 원로그는 로컬에 없음.
