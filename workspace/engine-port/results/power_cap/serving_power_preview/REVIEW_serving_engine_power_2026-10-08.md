# 검토 — 서빙 엔진에서 파티션·phase별 전력을 확인할 수 있는가 (2026-10-08)

**등급: 검토·설계 메모(GPU 0). 새 판정 0건. 정본 미편집.** 근거: 합성 부하 프로브 `../partition_power/RESULT_partition_power_2026-10-08.md`,
센서 프로브 `../sensor_bw/RESULT_sensor_bandwidth_2026-10-08.md`, 이 디렉터리의 기존 데이터 미리보기 `preview_serving_power.py`.

## 1. 기존 서빙 데이터로 본 미리보기 (0 GPU-h)

N3 shape B 3회차 Job(Nemotron-H, 8192→64, D44 sticky-OFF, PHASE_EVENTS=1 유닛 37개)에서 100 ms 보드 전력 샘플 50,000개를 엔진 상태로 분류했다.
엔진 상태는 per-step telemetry의 `prefill_in_flight`이고, 시계 정렬은 유닛별 ≤ 52 ms, 센서 지연은 0–180 ms 범위에서 보정했다.

| 엔진 상태(그 100 ms 동안) | 샘플 수 | 평균 전력 | SW power cap(0x4) 비율 | 실현 decode 분할 |
|---|---|---|---|---|
| 유휴 | 20,642 | 89 W | 0.00 | — |
| decode만(prefill 없음) | 4,962 | 265–296 W | 0.02–0.11 | (0,108) 비분할 |
| prefill과 동거 | 9,407 | 365–369 W | **0.92–0.97** | (64,44) |
| decode 없음(prefill만·틈) | 14,910 | 251–263 W | 0.29–0.33 | — |

(범위 = 센서 지연 보정값 0/60/120/180 ms에 따른 변화.)

- **서빙에서도 phase별 전력은 보인다.** 요청마다 prefill(350–400 W) → decode(~290 W) → 유휴(89 W) 순서가 100 ms 해상도로도 드러난다.
  동거 상태에서는 SW power cap이 92–97 % 걸린다. 합성 프로브의 "둘 다 바쁘면 400 W에 붙는다"와 같은 방향이다.
- **이 데이터로는 파티션 효과를 분리할 수 없다.** 이 arm(sticky OFF)에서 decode만 = (0,108) 비분할, 동거 = (64,44)로 **상태와 실현 분할이 완전히 겹친다**(감사 지적, 교락).
  shape B라 batch가 1–2뿐이고, 전력은 100 ms 폴링 스냅샷이다(주기 부하에서 aliasing 위험).

## 2. 서빙 엔진 확인에 필요한 것

| 필요 | 현재 | 조치 |
|---|---|---|
| 20 ms 전력 | `job_entry.sh`가 `meta/power_samples.csv` 기록(커밋 `2196642`, 2026-10-08) | 이후 제출 Job부터 적용 |
| 20 ms SM 클럭(cap 하강 확인) | `meta/sm_clk_samples.csv` 시도(NVML `PROCESSOR_CLK_SAMPLES`), 로컬 GPU 미지원 · A100 미확인 | 미지원이면 100 ms `clk.csv`로 대체 |
| 시계 정렬 | srv.log 짝 맞춤으로 유닛별 ≤ 52 ms | runner가 bench 직전에 `(time.monotonic(), time.time())` 쌍을 한 번 기록하면 µs 단위. 엔진 수정 불필요 |
| 상태 ≠ 분할 분리 | sticky OFF에서는 겹침 | **sticky ON 고정 분할**에서 decode만 / prefill만 / 동거를 같은 (P,D)로 측정 |
| 동거 중 파티션 귀속 | 보드 전력 하나 + cap 포화 | 불가(합성 프로브 판정 3). 배제 상태(한쪽만)에서만 귀속, 동거는 "합 대비 몇 배"로 기술 |

## 3. 권고 경로

**A. N2(3-a) 정적 격자에 얹기 — 추가 GPU 0.** N2는 sticky ON으로 D ∈ {16, 24, 34, 44}와 비분할 108에서 **decode만** 단독 프로파일을 PHASE_EVENTS=1로 잰다.
`job_entry.sh` 샘플러를 커밋한 뒤 N2를 제출하면 "분할 크기별 decode-only 전력 + batch(1/8/32/48)·context별 J/step"이 부산물로 생긴다.
- 조건: N2는 등록 감사 중(rev10)이다. 샘플러는 등록 스크립트를 바꾸지 않지만 Job 안에 백그라운드 프로세스(10 Hz NVML 조회)를 하나 더한다.
  따라서 **기판 변경으로 N2 감사에 고지**해야 한다(조용히 얹지 않는다). 판정 입력이 아니고 기술 산출물이다.

**B. 전용 서빙 관측 Job — 약 1 h(추정).** A가 못 채우는 prefill만·동거 상태를 같은 고정 분할에서 잰다.
- 모델 Nemotron-H(사다리 규칙), shape B(필요 시 A), sticky ON 고정 분할 (P,D) ∈ {(92,16), (64,44)}와 sticky OFF 비분할 대조.
- boot 하나 안에서 순서대로: ① decode만(짧은 prompt·긴 출력, 동시 요청 수 고정) ② prefill만(긴 prompt·출력 1 토큰) ③ 동거(shape B 고정 rate, cliff 밖) ④ 토글(decode 부하 위에 prefill 요청을 주기적으로 주입).
- seed 2 × 3 arm = 6 boot. 20 ms 전력·클럭, µs 정렬, PHASE_EVENTS=1.
- 산출(기술): 상태별 보드 전력, 동거/(prefill만 + decode만) 비율, cap 노출, J/prefill-token, J/decode-step(bs별), 동거 시 처리량 하락.
- 기반: E2 VESSL runner(`results/r2_eval/e2_sticky_vessl/e2_sticky_vessl_r2.sh`, 같은 Job 안 sticky ON/OFF arm 배치 선례)를 포크한다. 프로젝트 규율상 사전등록 + 규칙층 감사 → dry-run → 승인 → 제출 순서를 따른다.

## 4. 주장 범위 (미리 고정)

- 쓸 수 있는 것: "서빙 중 phase(prefill/decode)별 보드 전력·에너지", "고정 분할에서 한쪽만 돌 때의 전력", "동거 시 cap 포화와 가산성 붕괴의 크기".
- 쓸 수 없는 것: 동거 중 파티션별 전력, "SM 분할 = 전력 분할", 기판 간 수치 혼합, 정책 주장(게이트 1·2·3).

## 5. 사용자 결정

1. ~~`job_entry.sh` 샘플러 변경을 커밋할지~~ — 2026-10-08 사용자 "A 진행"으로 커밋 `2196642`.
2. ~~A(N2에 얹기)~~ — 진행. 고지 `results/n2_static_grid/NOTICE_substrate_power_sampler_2026-10-08.md`가 N2 rev12부터 §3-3·N2C-61로 반영됨(샘플러 on/off를 `substrate.json`에 기록, 영향 주장 금지). 전력 분석은 N2 Job 회수 뒤.
3. B(전용 Job) 사전등록을 시작할지.

## 그림

- `figures/preview_power_by_state.png`: 기존 N3 데이터의 엔진 상태별 전력 분포(왼쪽 위), batch별(오른쪽 위), 요청 단위 시계열(아래)
