# 결과 감사: N3(게이트 #6 SLO 용량 격자) shape B 3회차 Job `pdmux-n3-capacity-38ed3f4-20261006092813` (claims-auditor, 2026-10-06, 읽기 전용, GPU 0)

> 감사 대상은 원자료와 등록 문서뿐이다. result-analyst의 `RESULT_n3_B_r3_2026-10-06.md`는 감사 시점에 읽지 않았다. 결과 문서는 이 판정서 §9의 허용·금지 문안과 대조해야 한다(메인 세션 후속 작업).
> 감사 기준 HEAD `0c7b3b8`(회수 커밋). 등록 봉인 커밋 `38ed3f4`(rev6). 변이·재채점은 전부 scratchpad 사본(`…/scratchpad/aud_r3/`)에서 했고 저장소 파일은 이 판정서 외에 바꾸지 않았다.

## 최종 판정

**死因 0. 등록 라벨은 기계적 산물로 `CONFIRMED`다(`N3_RESULT.json`·`T_K.json` 바이트 재현). 해석 범위는 아래 caveat N3VR-1…11로 좁힌다. 게이트 #6은 닫히지 않는다.**

| 항목 | 판정 |
|---|---|
| Q1 라벨 기계적 재현 · 등록 봉인 | `CONFIRMED`: 바이트 동일, 봉인 15/15 일치, 회수 sha 639/639 |
| Q2 등록대로 집행(실행된 43 unit) | `CONFIRMED`: 집행 이탈 0건. 미시도 5 unit은 등록된 입장 규칙이 정상 작동한 결과다 |
| Q3 예산 모형(§8-3 보통 48/48·최악 47/48, §8-7 최악 46/48) | **예보 `REFUTED`**(실측 43/48, 등록 예보 실패). 라벨 오염 0. k=3 라벨·CI 영향 0. k=5/10 `UNRESOLVED`는 등록된 측정 부족 경로다 |
| Q4 k=3 `CAPACITY_BRACKETED_3` 0.15 [0.15, 0.22) | 라벨은 `CONFIRMED`(기계적). "용량 0.15 req/s" 단독 인용은 `REFUTED`. 용량 추정치로서는 `PLAUSIBLE(조건부)`: 요청 수 절벽(80건 중 2건 여유), 등록 절벽 구간 안, 버스트 실현에 조건부 |
| Q5 rev6 floor 첫 실가동 | floor 유효 `CONFIRMED`. 첫 간격 1020.9 ms가 bench_serving 시험 요청 뒤 1 s 대기라는 것 `CONFIRMED`. N3C-30 편향은 이 Job floor 창에서 관측되지 않았다(`CONFIRMED(scoped)`). 단 입장 라벨 경쟁은 드물지 않다(새 사실, N3VR-7) |
| Q6 관측자·CG2·arm identity·retract·정상성·기판 | 전부 `CONFIRMED`(기계적). 해석 caveat N3VR-8·9·10 |
| Q7 §9 예보 | F-B1 `HELD`, F-B2 `HELD`, F-A1·F-A2는 shape A 미실행이라 해당 없음. 예산 예보는 위 Q3 |
| Q8 금지 문안 | §9 목록 |

---

## 0. 독립 실측

- **회수 무결성**: `vessl/…/SHA256SUMS` 639행 = 메타·로그 13 + OUT 626, **639/639 OK**. `DONE` `exit_code=0`, `meta/exit_code` 0, 컨테이너 09:28:39Z→13:18:56Z(`cost.txt` 3.838 h, 하한 $5.68). k8s Duration은 3h54m20s다.
- **등록 봉인**: `REGISTRATION_SHA256.txt`의 15개 digest를 `git show 38ed3f4:<path> | sha256sum`과 대조했다. **15/15 일치**, 로컬 작업 트리 바이트도 15/15 같다. `# tree: 38ed3f4… uncommitted_outside_results=0`.
- **기판**: `substrate.json` A100-SXM4-80GB `GPU-c36b3c0c-d3dc-55d4-ec0f-f171016325b9`, driver 580.105.08, cc 8.0, MIG off, 이미지 `sglang-runtime@1e0d335b…`, 커밋 `38ed3f4`. 43 boot가 모두 이 GPU에서 돌았다(Job 1개). `SUBSTRATE_WARNING`은 0건이다.
  - 부수 사실: 2회차 Job(`…79999aa…`)도 같은 GPU UUID였다(1회차는 `GPU-1d3fda4c…`). 이 사실은 N3C-28을 완화하지 않는다.
- **Job env**: 식별자와 `PDMUX_N3_SHAPE=B`뿐이다. `KNOBS.txt` = 상속 `PDMUX_PHASE_EVENTS` unset.

## 1. Q1 — 라벨 기계적 재현 (`CONFIRMED`)

- `n3_analyze.py --out-dir <사본> --shape B --clk <회수 meta/clk.csv> --json-out …`를 로컬 miniconda `python3`로 돌렸다(18 s). 산출이 `N3_RESULT.json`과 **바이트 동일**하다(sha `f6894b99…`, `cmp` 통과). 결정 필드, 기술 필드, NaN 전부 같다.
- `n3_plan.py --floor B bench_floor_head.jsonl tel_floor_head.jsonl` 산출도 `T_K.json`과 **바이트 동일**하다. floor_s 0.9049716886365786 s, T_3/5/10 = 2.714915/4.524858/9.049717 s.
- `n3_plan.py --units B` 출력이 `UNITS.txt`와 바이트 동일하다(48 unit).

## 2. Q2 — 등록 집행 (`CONFIRMED`, 이탈 0)

| 검사 | 결과 |
|---|---|
| unit 순서 | `TIMING.tsv` 43행 = `UNITS.txt` 앞 43행(diff 0). floor_head → bridge 4386·4162 → observer ABBA(ON,OFF/OFF,ON/ON,OFF/OFF,ON) → round1 내림 → round2 오름 → round3 내림 → round4 오름(0.45까지) |
| seed·rate·N | 42개 bench 로그의 `Namespace`에서 seed·`request_rate`·`num_prompts`가 unit 이름·등록과 42/42 일치. `cell_validity` 무효 셀은 미시도 5건(`bench artefact absent`)뿐이다 |
| PHASE_EVENTS 배치 | telemetry에 `prefill_span_start`가 있는 boot 37개 = floor_head + grid/obs_on 36. 0건인 boot 6개 = bridge 2 + obs_off 4. 등록 배치와 같다. R7 표식·`disabled after failure` 0건 |
| 서버 명령줄 | `server_args`를 port만 가리고 비교하면 43/43 동일하다. flashinfer, `disable_cuda_graph=False`·`disable_piecewise_cuda_graph=False`(**cudagraph ON**), radix off, chunked −1, overlap off, mem 0.82, max_running 48, context 16384, seed 1, `sm_group_num=8`, A-agn config 경로 |
| sticky·green | `sticky partition ENABLED` 0/43, `GREEN OK` 43/43 |
| P3 (ii) | `P3_VERDICTS.txt` 43행 전부 PASS, mamba 48, 토큰 43/43 = **2,721,290**(distinct 1, rel 0). E2VR-1과 달리 이 Job의 라벨 존재는 P3 (ii) 허용오차에 의존하지 않는다 |
| CG2 | `CG2_PASS on=8 off=8 identical=True`(floor_head 대 첫 OFF boot `br_b_r3_4386`). `cg2_all_boots` distinct ON 1·OFF 1 |
| O1–O4 | `ip_local_port_range` 32768–60999. 포트 43/43이 창 20000–20799 안이고 `tries=1`이다. teardown에 `^sglang::` pkill이 있다. `SETTLE.txt` 43/43 `settle_polls=1`·used 0 MiB, NVML used 804,519,936 B로 43/43 동일하다 |
| retract | `#retracted_reqs` 로그 줄 0건. `retract_lines_before_bench=0`도 43/43이다. 단 이 로그가 `event_loop_pdmux` 경로에서도 찍히는지는 미실측이다(§16) |
| arm identity(R3) | 무효 셀 0. 채점 셀과 OFF·bridge 셀에서 비표적 분할 decode iteration·busy 스냅숏이 0건이다 |
| tail | `N3_DONE_B… remaining_s=-17`. 분석은 셀 마감 17 s 뒤에 끝났지만 tail 예약 600 s 안이다. 셀 구간 kill 층은 발화하지 않았다(`cut=none` 43/43, `budget_killed_*` 없음) |

## 3. Q3 — 예산 미달(5 unit 미시도): 등록 예보 실패

### 3-1. 사실

- `BUDGET.txt`: `remaining_at_loop_end_s=31`. 입장 거부는 `g_r053_4589`(need 253 s, 남은 31 s)이고, 이후 `g_r061/069/077_4589`와 `floor_tail`은 미시도다. 거부는 unit 시작 전에 났으므로 **잘린 bench가 채점에 들어가지 않았다**.
- 등록 예보: §8-3 보통 코너 12,126 s 48/48, 최악 코너 13,792 s 47/48. §8-7 "+1.45 s/boot면 최악 46/48". 실측은 **43/48**이다. 등록된 최악 코너보다 나쁘다. 예보 `REFUTED`다.

### 3-2. 분해(`TIMING.tsv`, unit = 입장 → 다음 입장)

| 구성 | 실측(grid/obs 39 unit 중앙) | 등록 모형(`n3_plan.SHAPES["B"]`) | 차 |
|---|---|---|---|
| preflight(t0 → 첫 입장) | **224 s** | `PREFLIGHT_ALLOW_S` 600 s | −376 s(여유) |
| 첫 boot(cold) | floor_head boot 162 s(warm 중앙 104 s 대비 +58 s) | `COLD_FIRST_BOOT_EXTRA_S` +92.1 s | −34 s |
| bench 밖 부대비용 | **170 s** = boot 104 + health→bench 42 + bench 뒤 17 + unit 간 7 | `overhead_s` 121.0 s(최악 ×1.25 = 151.25) | **+49 s(+40%)**, 최악보다도 +19 s |
| bench | 예: 0.30 → 120–128 s(모형 99.7), 0.10 → 213–227 s(모형 193.0), 0.15 → 149–158 s(모형 129.7). 중앙 초과 +22 s | `max((N−1)/rate, N/λ_ref) + drain 3 s` | **+22 s** |
| unit 합 | 입장→다음 입장 중앙 **293 s**(평균 307.7). 입장→종료 중앙 285.5 s | 보통 ≈220–236 s, 최악 need 250–294 s | ≈ +70 s/unit |

- 누적: unit당 약 +70 s × 43 ≈ +3,000 s다. preflight 여유 376 s로는 메우지 못했다.
- 모형이 빗나간 지점:
  - (i) **부대비용 상수 121.0 s는 λ0 VESSL(5그룹, PHASE_EVENTS 미사용) unit에서 쟀다.** A13 §5-1 참고 분해로 B의 boot+warm-up은 95–97 s였다. N3에서는 boot+health→bench가 146 s다(+≈50 s). 이 50 s가 green read-out·P3 helper·settle·cpu.stat·`--output-details`·6그룹 캡처·PHASE_EVENTS 기록 중 어디서 나는지는 `TIMING.tsv` 해상도로 가를 수 없다(미확정).
  - (ii) bench 모형에는 bench_serving 자체 시험 요청과 1 s 대기, 프롬프트 토큰화, 마지막 요청 완료 시간이 없다. drain 3 s도 작다.
  - (iii) **§8-7의 "상수를 올릴 근거가 프로브에 없다"는 boot 벽시계(59 s)만 잰 결론이었다.** 상수가 담는 unit 부대비용(boot + warm-up + 사후 + 간격)은 재지 않았다. 등록이 이미 "1:1 비교 아님"이라고 적었지만, 그 단서가 예산 판단으로 번역되지 않았다.

### 3-3. 라벨 영향과 결측 구조(결측은 rate와 상관됨 — 등록 R1 설계대로)

- **k=3**: 첫 실패 규칙이 읽는 rate는 0.10·0.15·0.22이고 모두 n=4다. 결측(≥0.53)은 점 라벨에 닿지 않는다.
  - bootstrap은 등록 함수가 **전 rate 교집합 seed**를 쓰므로 4589가 빠져 `n_paired_seeds`=3이다(N3C-12: 라벨 n=4와 함께 적는다).
  - 진단: 4 seed를 0.10–0.45 rate로 같은 함수에 넣으면 CI는 [0.15, 0.22]로 **같다**. 모든 seed가 0.30에서 실패(≤26/30)하므로 재표집 용량이 0.53 이상 셀에 의존하지 않는다.
  - 즉 결측은 k=3 CI도 편향시키지 않는다. 단 이 4-seed 값은 등록 산출물이 아니라 진단이다.
- **k=5** `UNRESOLVED_N_BELOW_4_5`(missing_at 0.53): 등록 라벨이고 측정 부족이다.
  - 감사 산술(라벨 아님): 0.53의 3 seed 통과는 133/159다. 결측 셀이 53/53이어도 (133+53)/212 = **0.877 < 0.90**이다.
  - 따라서 결측 셀은 k=5의 첫 실패 위치를 바꿀 수 없고, n 규칙만 충족시킨다.
- **k=10** `UNRESOLVED_N_BELOW_4_10`(missing_at 0.53): 결측 셀이 라벨을 **정한다**.
  - 0.53이 통과하려면 4589 ≥ 32/53, 0.61은 ≥ 38/61, 0.69는 ≥ 52/69가 필요하다. 0.77은 상한 0.614라 실패가 확정이다.
  - 그래서 가능한 용량은 0.45·0.53·0.61·0.69 중 어디든이다(다른 seed의 0.69는 63/65/69 of 69 — 추정 금지).
- **기술 곡선**: `capacity_curve` k ∈ [5, 25]의 81점이 전부 `UNRESOLVED_N_BELOW_4`다. A′ 틀의 "연속 곡선 공개"(P-1) 역할은 shape B에서 k < 5 구간만 수행됐다.

### 3-4. "재개 Job으로 채우면 된다"의 성립 조건과 교락(처방 아님 — 사용자 결정)

- **규약상 가능**(§8-4, 러너 `RESUME_FROM` 분기). 조건은 셋이다.
  - `UNITS.txt` 바이트 동일, 등록 digest 동일(봉인 파일을 바꾸면 불가), 이 Job `T_K.json` 유효(`--tk-valid`) — 셋 다 현재 충족 가능하다.
  - 이전 Job 디렉터리가 `$DATA/runs/<이 Job>/results/n3_capacity/n3b_<이 Job>`에 **남아 있어야 한다**. VESSL 클러스터 볼륨에 보존됐는지는 이 감사가 확인하지 못했다.
- 재개 Job은 floor를 다시 재지만(`T_K_RESUME_<job>.json`, 진단) 채점은 이 Job의 초 값으로 한다. OFF boot가 없어 CG2를 다시 평가하지 않는다. 복사된 unit은 `P3_KEYS`에 들어가지 않는다.
- **교락(#11, E2VR-3)**: 재개 셀은 다른 Job·(대개) 다른 GPU·다른 시점이다. VESSL Job 간 같은 셀 수준 이동은 이미 +3.6–4.1%가 관측됐다(E2VR-3).
  - k=10은 재개 셀이 라벨을 직접 정하므로 **Job 혼합이 라벨 안에 들어간다**.
  - k=5는 재개 셀이 점 라벨을 바꿀 수 없다(3-3). 그래도 bootstrap CI는 Job 혼합 셀로 계산된다.
  - k=3은 무영향이다.
- 대안(새 Job 전체 재측정, 예산 상수 재추정 후 rev7 등)과 그 선택은 **사용자 결정**이다. 이 감사는 처방하지 않는다.

## 4. Q4 — k=3 라벨의 metric cliff 노출(민감도 진단, 판정 아님)

원자료는 이 Job의 bench jsonl이다. 판정 함수는 등록 `n3_analyze.capacity_rule`·`pooled`·`bootstrap_capacity` 그대로이며, 새 추정량은 만들지 않았다.

| 축 | 결과 |
|---|---|
| per-seed 통과(k=3) | 0.10: 20,19,20,19 of 20(78/80 = 0.975) · **0.15: 18,18,19,18 of 20(73/80 = 0.9125)** · 0.22: 20,17,18,20 of 22(75/88 = 0.852) |
| 요청 수 여유 | 0.15에서 **2건**만 더 실패하면(71/80 = 0.8875) 용량이 0.10으로 내려간다. 0.22가 통과하려면 5건이 더 통과해야 한다(80/88) |
| 동률 | 짝 seed 3개(2084·3704·4861)가 0.15에서 정확히 18/20 = 0.90이다 → bootstrap attainment CI [0.90, 0.90]. 하단 0.15는 등록 동률 규약("< 0.90만 실패")으로만 통과한다 |
| 등록 절벽 구간 | `cliff_zone["3"]` = 0.15–0.53. **용량 셀 0.15와 실패 셀 0.22가 모두 Z_B1(3) 안**이다(§3-2: 5-a 헤드라인 비교점 불가) |
| T(floor) 균일 섭동 | 라벨이 0.15로 유지되는 T 범위는 **[2.371, 3.140) s**, 즉 floor −12.7% … +15.7%다. floor ±2%·±5%에서 라벨·attainment 불변(0.15에서 TTFT 2,442 ms와 2,818 ms 사이에 빈 띠가 있다) |
| leave-one-seed-out(n=3, 진단) | 네 경우 모두 0.15 |
| 도착 순서 반분(진단, 등록 아님) | 전반만 쓰면 0.10, 후반만 쓰면 **0.30**이다. ≤0.22 실패 22건 중 20건이 전반이다. 앞 요청이 시스템에 남아 있을 때 도착한 수가 전반/후반 12/4(0.10)·14/6(0.15)·19/11(0.22)이고, 겹친 도착의 TTFT 중앙값은 전·후반이 비슷하다(0.15: 2,296 vs 2,230 ms) → 선택된 seed의 Poisson 실현(버스트가 전반에 몰림)으로 설명된다. 같은 seed는 rate마다 같은 지수 난수열의 척도 변환이다(`arrival_offsets`) |
| 정상성 | 0.15: S1 0.80, S2(k=3) −0.125, S3 −0.020 → `STATIONARY`. 0.22: S1 0.70, S2 −0.25 → `STATIONARY`. S2는 후반 악화만 발화하므로 초기 집중은 잡지 않는다 |

결론:
- "용량 0.15 req/s" **단독** 인용은 `REFUTED`다. 등록 §11 허용형은 구간·CI·T_K·구속 설정·실현 분할·정상성·관측자·CG2·power cap·Job/GPU를 함께 요구하고, 위 절벽 노출은 그 허용형 안에서도 병기를 요구한다(N3VR-1·2·3).
- 라벨은 T에 대해서는 견고하고, 요청 수와 버스트 실현에 대해서는 얇다. 이 노출은 등록 자유 표면 §12-1(작은 N)·§12-3(seed 재채점)·N3C-7·N3C-8이 예고한 범위 안이다.

## 5. Q5 — rev6 floor 규칙 첫 실가동

- **floor 유효**(`CONFIRMED`): `problems` [], 28 span 전부 `prefill_span_end.stream_idx` 0, span 안 decode_iteration 0, `dropped_events_max` 0, `floor_entry_race_count` 0.
- **첫 간격 1020.9 ms**(`CONFIRMED`): telemetry 순서상 span 12는 bench 창 안의 bench_serving 시험 요청이다(앞 간격 20,013.6 ms). span 13(첫 floor 요청)의 앞 간격 1020.9 ms는 그 시험 요청 뒤 1 s 대기다. warm-up 쪽 span 4(1019.8 ms)도 같은 구조다. 나머지 15개 간격은 16.1–19.2 ms다.
- **N3C-30 지연 오염**: 이 Job의 floor 창에는 없었다(`CONFIRMED(scoped)`). floor TTFT 0.9015–0.9115 s, `dispersion_max_over_min` 1.011, floor 창 입장 경쟁 0이다.
- **새 사실(입장 라벨 경쟁은 드물지 않다)**:
  - `entry_label_off_count` 1은 floor_head warm-up의 span_seq 6이다.
  - 같은 위치(span_seq 6, warm-up 세 번째 프롬프트)의 입장 라벨 (0,108)·running_bs 1이 **이 Job의 PHASE_EVENTS ON boot 37개 중 36개**(예외 `g_r045_4589`)에서 나타났다.
  - 그 앞 간격은 85–1,445 ms였고, 해당 warm-up 요청 TTFT는 1.39–2.33 s였다(floor_head 1.39 s, 다른 warm-up 요청 ≈0.905 s).
  - 2회차 Job도 span_seq 6이었다.
  - 따라서 2회차 진단의 "드문 경쟁(1/28)"·"rev5 재제출 통과 확률 ≈40%" 어림은 이 Job 자료와 맞지 않는다. rev5였다면 이 Job도 거의 확실히 `ABORT_FLOOR`였다. 위치 고정의 원인은 **미확정**이다(N3VR-7).

## 6. Q6 — 관측자 효과·CG2·arm identity·정상성·기판

- **관측자**(`OBSERVER_OK`, n=4, rate 0.30):
  - TTFT p50 ln(ON/OFF) 평균 +1.20%, CI [−0.38%, +4.01%].
  - ITL 평균 +1.65%, CI [**−0.07%**, +3.39%]. 하한이 0에 붙어 있다.
  - TTFT +4%(CI 상단)는 균일 T 축소로 환산해도 T_3 = 2.61 s로 라벨 띠 [2.371, 3.140) 안이다 → k=3 라벨은 뒤집히지 않는다(진단). "PHASE_EVENTS 무영향"은 쓰지 않는다(N3VR-8).
- **CG2**: 위 §2. 의미는 N3C-10 범위(동시성 1 비분할 경로)뿐이다.
- **arm identity**: 위반·unverifiable 0. 이는 `invalid_cells` 근거 인용이다(N3C-20).
- **기판**:
  - bench 창 SW power cap 노출: 0.10 8.8–9.3%, **0.15 13.4–13.7%**, **0.22 16.3–18.7%**, 0.77 41–45%, bridge 46.9%. 평균 SM MHz 1,380–1,407(최대 1,410).
  - `clk_summary` 전 구간 throttle 표본 21,439/137,581(15.6%, 유휴·부팅 포함).
  - cgroup quota 11 CPU(`cpu.max 1100000 100000`) vs `nproc` 96(교훈 268). bench 창 CPU 사용 1.13–1.30 코어, 창당 throttled period 7–10회(합 359/62,922 = 0.57%, rate 무관).
  - 어느 것도 판정 입력이 아니다(§2-4). 개별 요청 단위 영향은 배제하지 못한다(N3VR-2·8).
- **floor_tail 미시도**: Job 앞뒤 floor 비교(클럭 drift 진단)가 없다.
- **실현 분할**(필수 병기, §7-4):
  - 용량 셀 0.15 seed 4개: E-cnt 13.79–18.80(평균 15.88) / E-time 23.80–31.17(27.51) / E-iter 15.72–20.85(17.85) / E-qcond 80.0–100(91.88), 쌍 (64,44) A2 규약.
  - 같은 셀 W-idx (0,108) 0.821–0.835 · (108,0) 0.099–0.112 · (64,44) 0.053–0.073, X-tok 19.5–27.4(22.99), cap-bound 0, `dropped_events` 0.
  - 실패 셀 0.22: E-cnt 15.32–19.66(18.17) / E-time 28.65–39.64(34.73) / E-iter 17.41–22.74(20.59) / E-qcond 80–100(90.0), W-idx (0,108) 0.736–0.753 · (108,0) 0.130–0.145 · (64,44) 0.111–0.130, X-tok 22.2–29.6(26.85), `dropped_events` 0.
  - **target-D44 arm의 용량 셀에서 decode-busy 시간 중 (64,44) 점유는 1/4 남짓이다** — λ\*\_SLO(3)는 D44 측정이 아니다(§1-25/λ0R 계열 재현, N3VR-9).
- **ITL**: F-B2 `HELD`(요청별 ITL p95 > 50 ms 0건). `ITL_BOUND_3` false(ITL 40/50/60/65 ms 사다리 전부 0.15).
  - 단 0.10–0.15 셀에서 단일 토큰 간격 > 100 ms 요청이 rate당 3–5개였고, 최대 1,372 ms였다(N3VR-10).

## 7. Q7 — §9 반증 가능 예보 대조(판정 아님, 등록 의무)

| 예보 | 결과 | 근거 |
|---|---|---|
| F-B1 저부하 희석 | `HELD` | E-time(0.10) seed 평균 21.11% ≤ 40%, rate ≤0.45 Spearman 1.0(21.1→27.5→34.7→46.9→57.6→67.1) |
| F-B2 ITL 비구속 | `HELD` | 요청별 ITL p95 > 50 ms 0건 |
| F-A1·F-A2 | 해당 없음 | shape A Job 미실행 |
| §8-3 예산(보통 48/48·최악 47/48) | **틀림** | 실측 43/48 |
| §8-7 "+1.45 s/boot면 최악 46/48, 보통 48/48 불변" | **틀림** | 실측 43/48. 원인은 캡처가 아니라 unit 부대비용 +49 s와 bench +22 s |
| 2회차 진단 "경쟁 1/28·rev5 통과 ≈40%"(SUBMISSIONS, 계획용 어림) | **틀림** | 이 Job 36/37 boot의 warm-up span_seq 6 |
| §12-4 설계 사전값(k=10 절벽 0.62–0.85) | 인용 금지(N3C-13) — 대조하지 않는다 | — |

## 8. 반전 시험 표

원자료는 이 Job의 bench·telemetry jsonl과 `T_K.json`이다. 판정 함수는 등록 `n3_analyze.capacity_rule`/`bootstrap_capacity`(X=0.90, `< 0.90` 실패, 2000 표본·seed 1)다. 새 격자·추정량은 없다. "진단" 행은 등록이 허용하지 않는 조작이라 판정을 바꿀 수 없고, 노출 기록용이다.

| 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 |
|---|---|---|---|
| T_3(floor 측정 산포) | floor ±2%, ±5%, 관측자 CI 상단 +4% | 없음 | 라벨 유지 띠 T ∈ [2.371, 3.140) s(−12.7%…+15.7%) |
| seed 부분집합 | leave-one-out 4종(진단) | 없음(k=3 0.15) | — |
| bootstrap seed 집합 | 등록 3(교집합) ↔ 4(rate ≤0.45, 진단) | CI 불변 [0.15, 0.22] | 35 multiset 중 30개 0.15, 5개 0.22 |
| 동률 규약 | 등록 `<0.90 실패`(고정, 자유 아님) ↔ `≤0.90 실패`(진단) | 점 라벨 불변(0.9125 > 0.90). 3-seed bootstrap 하단만 0.10으로 | 3 seed가 정확히 18/20 |
| ITL SLO | 40–65 ms 사다리(등록 기술) | 없음 | `ITL_BOUND_3` false |
| 예산 결측(rate 상관) | k=3 / k=5 / k=10 | k=3 없음. k=5 결측 셀 무관(상한 0.877). k=10 결측 셀이 0.45–0.69를 정함 | §3-3 |
| P3 (ii) ↔ 엄격 | 43 boot 토큰 distinct 1 | 없음 | 2,721,290 × 43 |
| 정상성 문턱 | S2 0.10(단측) | 없음(초기 집중은 음의 S2) | 0.15 −0.125, 0.22 −0.25 |
| 도착 순서 반분(진단, 등록 아님) | 전반 ↔ 후반 | **0.10 ↔ 0.30** | 실패 22건 중 20건 전반 |

**등록 범위 안에서 반전을 만들지 못했다 → 死因 0.** 반분 행의 반전은 등록이 허용하지 않는 조작이라 死因이 아니라 caveat(N3VR-3)다. 결측 행은 등록된 `UNRESOLVED` 경로 그대로다.

## 9. 정본 반영 문안

**허용 문안(정본 `PROJECT_STATUS.md`·`CONSENSUS.md`가 문자 그대로 옮길 것)**:
> "N3 shape B 3회차 Job `pdmux-n3-capacity-38ed3f4-20261006092813`(VESSL A100-SXM4-80GB `GPU-c36b3c0c-d3dc-55d4-ec0f-f171016325b9`, 사전등록 rev6 봉인 `38ed3f4`, 43/48 boot 완료 — 예산 입장 규칙으로 round 4 seed 4589의 0.53·0.61·0.69·0.77과 floor_tail 미시도, 등록 예산 예보[보통 48/48] 실패; 라벨 바이트 재현)은 target-D44 sticky-OFF B1의 λ\*\_SLO(k=3) — shape B(8192/64), 라벨 `CAPACITY_BRACKETED_3`, 구간 [0.15, 0.22) req/s(절대), seed 짝 bootstrap 95% [0.15, 0.22](짝 seed n=3; 라벨 n=4), T_3 = 3 × floor = 2.715 s(floor 0.905 s, 같은 Job 첫 블록, rev6 실현 증거 판정 유효), ITL SLO 50 ms, attainment ≥ 90%(0.15: 73/80 = 0.9125, 0.22: 75/88 = 0.852 — 2요청 여유, 두 셀 모두 등록 절벽 구간 안) — 을 냈다. 구속 설정: cap 48 = mamba 48, `max_prefill_tokens` 16384, `max_total_num_tokens` 2,721,290(43 boot 동일). 용량 셀 실현 분할(쌍 (64,44), A2 규약, seed 범위): E-cnt 13.8–18.8% / E-time 23.8–31.2% / E-iter 15.7–20.9% / E-qcond 80–100%, W-idx (0,108) 0.82–0.84·(108,0) 0.10–0.11·(64,44) 0.05–0.07, X-tok 19.5–27.4%, dropped_events 0 — D44 측정이 아니다. 정상성 S1–S3 `STATIONARY`, 관측자 `OBSERVER_OK`(rate 0.30 한 점), CG2 PASS, bench 창 power cap 노출 13.4–13.7%. k=5·k=10은 `UNRESOLVED_N_BELOW_4`(missing_at 0.53, n=3 — 예산 미시도로 인한 측정 부족이며 '용량 없음'·'구별 불가'가 아니다). best static 판정 아님. 게이트 #6은 닫히지 않는다(shape A 미실행, k=5/10 미해결, 5-a 전제 미판정)."

**필수 병기**: N3C-1·2·5·7·8·10·11·12·13·14·15·17·20·21·22·29·30·31, DS-1·DS-4, 아래 N3VR-1…11.

**금지 문안**:
1. "게이트 #6을 닫았다 / 해소했다 / 충족했다"(N3C-1, §14). "N3 B 완료"를 게이트 상태 변경으로 쓰는 것.
2. "N3 B1 용량 = 0.15 req/s", "λ\*\_SLO(B) = 0.15"(구간·CI·T_K·병기 없는 단독 수치, 0.15를 견고한 용량으로 쓰는 것). "D44 용량", "D44에서의 SLO 용량"(조항 2, DS-1).
3. "k=5/k=10 용량 없음", "k=5/10에서 구별 불가", "k=5 용량 0.45", "k=10 용량 ≈ 0.69"(n=3 값이나 감사 산술을 라벨로 쓰는 것, 게이트 #21).
4. "VESSL λ\*\_SLO를 KISTI에 이식/비교", "λ\*\_SLO(3) ≈ 0.22 × λ\*(B)"류 비율 표현(DS-4, LV-3(c)), 기판 간 풀링.
5. "best static", "B1이 최적", "N3가 정책/D를 비교했다"(조항 10).
6. "floor 0.905 s = 2회차 0.907 s 재현", "같은 GPU라서 floor가 재현됐다"(N3C-28).
7. "입장 라벨 경쟁 발생률은 1/28 / 36/37", "경쟁은 드물다", "rev5였어도 ~40% 통과", "floor는 이 정지에 면역이다"(N3C-31, N3VR-7).
8. "rev6는 floor 값을 바꾸지 않는 순수 라벨 수리다"(N3C-30), "`prefill_span_end.stream_idx == 0`이므로 (108,0)에서 실현"(N3C-29).
9. "PHASE_EVENTS는 결과에 영향이 없다"(OBSERVER_OK의 확대, N3C-6 형식), "CG2 ⇒ 부하 중 출력 동치"(N3C-10).
10. "ITL은 무관하다 / decode는 멈추지 않는다"(F-B2·`ITL_BOUND` false의 확대, N3C-17, N3VR-10).
11. "결측 셀을 재개 Job으로 채우면 같은 Job 결과가 된다", 재개 셀을 `RESUMED_UNITS.tsv`·노드 병기 없이 라벨에 넣는 것(§8-4, #11, E2VR-3).
12. "예산 모형이 보수적이었다 / 최악 코너가 상한이었다"(§8-3·§8-7 예보 실패), "6그룹 프로브가 예산 상수를 검증했다".
13. "N3가 STATIONARY를 확인했으므로 초기 과도가 없다"(N3VR-3).
14. "bridge가 새 config와 homog5가 같은 arm임을 보였다"(§4-2). bridge achieved 0.6891/0.6903은 λ0 VESSL n=4 0.6901–0.6910 옆의 기술값일 뿐이다.
15. "O1–O4가 토큰 이탈을 해결했다"(N3C-21 — '이 Job에서는 이탈이 관측되지 않았다'만 허용), "P3가 boot 간 KV 풀이 같음을 확인했다"(N3C-22).
16. shape A·B 용량 크기 비교, 길이 정규화 비교(P-6). N3 셀을 5-a/5-b arm 비교 짝으로 쓰는 것(N3C-4).
17. N3C-1…33의 각 금지 문안 전부(승계, 이 판정서가 완화하지 않는다).

**불변 선언**: 게이트 #6(SLO 용량 — 부분 측정, 미닫힘), λ0R-8(iii)(E2C-5), HE0, layer-type 死, 정책 순위, Claim D/E(미검증), P2 블로커 ②③, 게이트 #13/#16, C2 인용정지는 모두 **불변**이다. 새 성능 판정은 0건이다.

## 10. 정정 R-n

- **R-1**: 지시문의 "unit당 실측 ≈285 s"는 입장→종료 중앙값(285.5 s)이다. 입장 규칙의 회계 단위인 입장→다음 입장은 중앙 293 s(평균 307.7 s)다. 둘 다 need 250–253 s(최악 코너)를 넘는다.
- **R-2**: 지시문은 `entry_label_off` 1건을 "warm-up" 1회성으로 요약했다. 이 Job 전체 ON boot에서는 36/37이다(§5). floor_head boot 기준 1건이라는 수치 자체는 맞다.
- **R-3**: 지시문의 0.15 attainment CI "[0.9,0.9]?"는 맞다. 다만 그것은 짝 seed 3개가 모두 정확히 18/20이라 생긴 퇴화 구간이다(동률 규약 의존).

## 11. 신규 caveat (인용 금지 문장 형태, 결과 문서·정본 문자 승계)

- **N3VR-1 (k=3 요청 수 절벽)**: "k=3 라벨의 하단(0.15 통과)은 80요청 중 2건 여유다(73/80 = 0.9125, 71/80이면 용량 0.10). 짝 seed 3개가 정확히 18/20 = 0.90으로 문턱 위에 동률로 앉아 있고, bootstrap의 0.15 attainment CI [0.90, 0.90]은 동률 규약('< 0.90만 실패')으로만 통과한다. 용량 셀 0.15와 실패 셀 0.22는 모두 등록 절벽 구간 Z_B1(3) 안이므로 5-a 헤드라인 비교점으로 쓰지 않는다. '0.15 req/s'를 단독·견고한 용량으로 인용하지 않는다."
- **N3VR-2 (T 견고성의 범위)**: "k=3 라벨은 T_3를 [2.371, 3.140) s(floor −12.7%…+15.7%)에서 균일하게 움직여도 0.15다. 이것은 균일한 T 변화에 대한 견고성일 뿐이다. 개별 요청의 지연 사건(단일 토큰 간격 최대 1.37 s, cgroup 쓰로틀 bench 창당 7–10회)에 대한 견고성이 아니다."
- **N3VR-3 (버스트 실현 조건부)**: "k=3 라벨은 seed 4개 × 20–22요청의 버스트 실현에 조건부다. ≤0.22 실패 22건 중 20건이 도착 순서 전반에 있고, 반분만으로 채점하면 0.10(전반)과 0.30(후반)으로 갈린다(진단, 등록 아님). 같은 seed는 rate마다 같은 지수 난수열의 척도 변환이라 rate 간 attainment는 독립 증거가 아니다. S2는 후반 악화만 발화하므로 STATIONARY가 초기 집중의 부재를 뜻하지 않는다."
- **N3VR-4 (예산 예보 실패)**: "등록 예산 모형(§8-3 보통 48/48·최악 47/48, §8-7 최악 46/48)은 틀렸다. 실측은 43/48이다. grid/obs unit은 입장→다음 입장 중앙 293 s로 최악 코너 need 250–253 s를 넘었다. bench 밖 부대비용 중앙 170 s는 상수 121.0 s보다 +40%, bench는 모형보다 중앙 +22 s다. preflight는 224 s(허용 600 s)였다. §8-7 프로브는 boot 벽시계만 쟀고 상수가 담는 unit 부대비용을 재지 않았다. 이 모형을 N3 shape A·재개·후속 캠페인의 예산 문언에 그대로 쓰지 않는다. 라벨 오염은 없다(입장 규칙이 시작 전에 거부했다)."
- **N3VR-5 (UNRESOLVED와 결측 구조)**: "k=5·k=10의 `UNRESOLVED_N_BELOW_4`는 seed 4589 × 0.53–0.77(rate와 상관된 결측)의 예산 미시도로 인한 측정 부족이다. 감사 산술(라벨 아님): k=5에서는 결측 셀이 어떤 값이어도 0.53의 pooled attainment 상한이 0.877이라 첫 실패 위치를 바꿀 수 없다. k=10에서는 결측 셀이 용량을 0.45–0.69 사이 어디로든 정할 수 있다. 기술 곡선 k ∈ [5, 25] 81점이 전부 미해결이다. 'k=5 용량 0.45', 'k=10 용량 ≈ 0.69'로 쓰지 않는다."
- **N3VR-6 (재개 = Job 혼합)**: "§8-4 재개는 규약상 가능하나(이전 Job 디렉터리 보존은 미확인), 재개 셀은 다른 Job·GPU·시점이다(#11, E2VR-3 Job 간 수준 이동 +3.6–4.1%). k=10 라벨은 재개 셀이 직접 정하므로 Job 혼합이 라벨에 들어간다. k=5는 재개 셀이 점 라벨을 바꾸지 못하고 n 규칙만 충족시킨다. 재개·새 Job 전체 재측정·rev7 중 무엇을 할지는 사용자 결정이다."
- **N3VR-7 (입장 라벨 경쟁의 위치 고정)**: "rev6 floor는 이 Job에서 유효했다(floor 창 경쟁 0, 간격 16.1–19.2 ms; 첫 1020.9 ms는 bench_serving 시험 요청 뒤 1 s 대기). 그러나 입장 라벨 경쟁은 이 Job ON boot 37개 중 36개의 warm-up span_seq 6에 나타났다(앞 간격 85–1,445 ms, 그 요청 TTFT 1.39–2.33 s). 2회차 Job도 span_seq 6이었다. 원인과 위치 고정의 이유는 미확정이다. 'floor는 이 정지에 면역', '경쟁 발생률 x', 'rev5였어도 통과 가능'으로 쓰지 않는다."
- **N3VR-8 (관측자·기판)**: "`OBSERVER_OK`는 rate 0.30 한 점에서 TTFT +1.2% [−0.4%, +4.0%], ITL +1.6% [−0.07%, +3.4%](n=4)다. ITL CI 하한이 0에 붙어 있다. 'PHASE_EVENTS 무영향'으로 쓰지 않는다. bench 창 power cap 노출(0.15 13.4–13.7%, 0.22 16.3–18.7%), cgroup quota 11 CPU(bench 창 사용 1.1–1.3 코어, throttled 0.57%), floor_tail 미시도(Job 앞뒤 drift 미진단)는 판정 입력이 아니며 병기만 한다."
- **N3VR-9 (실현 분할)**: "target-D44 arm의 용량 셀(0.15)에서 decode-busy 시간 중 (64,44) 비율은 E-time 23.8–31.2%, 벽시계 W-idx (64,44)는 5.3–7.3%다. decode의 대부분은 (0,108)에서 돌았다. λ\*\_SLO(3)는 D44 SM의 측정이 아니다."
- **N3VR-10 (ITL 비구속의 범위)**: "F-B2 `HELD`와 `ITL_BOUND_3` false는 요청별 ITL p95 > 50 ms가 0건이었다는 뜻이다. 0.10–0.15 셀에서도 단일 토큰 간격 > 100 ms 요청이 rate당 3–5개(최대 1,372 ms) 있었다. 'decode가 멈추지 않는다', 'ITL은 무관하다'로 쓰지 않는다."
- **N3VR-11 (CI의 n)**: "k=3 bootstrap CI [0.15, 0.22]는 짝 seed 3개(교집합)로 계산됐고 라벨은 n=4다(N3C-12). 4 seed 진단(rate ≤0.45)에서도 CI가 같다는 것은 감사 진단이지 등록 산출물이 아니다. n=4 CI의 재표집 공간은 35개뿐이다."

## 12. 반증 실패 기록

- 확인한 것: 봉인 15/15, 회수 639/639, `N3_RESULT.json`·`T_K.json`·`UNITS.txt` 바이트 재현, 순서 43/43, seed·rate·N 42/42, server_args 43/43, P3 43/43(토큰 distinct 1), sticky 0/43, GREEN 43/43, 포트 43/43 1회, settle 43/43, retract 0, R7 0, arm identity 위반 0, floor rev6 술어, T 섭동·LOO·bootstrap seed 집합·ITL 사다리 민감도.
- 라벨을 깨는 경로는 찾지 못했다. 단 아래 실험 전에는 "k=3 용량이 0.15–0.22 사이"라는 해석이 `CONFIRMED`가 아니다(`PLAUSIBLE(조건부)`).
  - **k=3 견고성**: 같은 Job 안에서 0.10·0.15·0.22에 seed를 늘리거나(n ≥ 8) span을 늘려 셀당 요청 수 ≥ 80/seed를 확보하는 등록이 필요하다. 그래야 2요청 절벽과 버스트 실현 의존을 줄인다. 새 사전등록과 규칙층 감사가 필요하다.
  - **k=5·10**: 같은 Job 안에서 0.53–0.77을 n ≥ 4로 측정해야 한다. 재개를 쓰면 Job 혼합을 라벨 옆에 병기해야 한다(N3VR-6). 예산 상수 재추정이 선행돼야 한다(N3VR-4).
  - **입장 라벨 경쟁**: 원인 규명은 엔진 계측 과제다(engine-porter). 이 감사의 판정 범위 밖이다.

## 13. 실현가능성 자기검사(게이트 #113)

- 이 판정서의 처방은 문서 문안과 병기뿐이다. 명령줄·arm·규칙·봉인 바이트 변경은 0건이다. 재개·재측정은 처방하지 않고 사용자 결정으로 남겼다.
- 반전 계산은 등록 함수만 썼다. "진단" 행(반분·4-seed bootstrap·결측 상한)은 판정에 쓰지 않았고 caveat로만 남겼다. 감사 산술 상한(k=5 0.877)은 단순 산술이며 라벨로 승격하지 않았다(새 자유 표면이 되지 않도록 금지 문안 3에 묶었다).
- 재감사 규율(#110): rev6 델타 감사의 N3C-28…33은 원 파일 바이트로 승계했고, 2회차 진단 수치(1/28·40%)는 이 Job 원자료로 독립 재검증했다(틀림, §5).
- §12의 k=3 견고성 처방은 예산과 충돌할 수 있다(N3VR-4: 이번 Job이 이미 예산 미달). 그래서 "처방"이 아니라 CONFIRMED 조건으로만 적었다.

## 관련 경로

- 원자료 OUT: `workspace/engine-port/results/n3_capacity/n3b_pdmux-n3-capacity-38ed3f4-20261006092813/`
- 회수 메타: `workspace/engine-port/results/n3_capacity/vessl/pdmux-n3-capacity-38ed3f4-20261006092813/{meta,logs,SHA256SUMS,DONE,cost.txt}`
- 등록: `workspace/engine-port/results/n3_capacity/PREREG_N3_CAPACITY_2026-10-04.md`(rev6), `VERDICT_n3_rev6_delta_2026-10-06.md`, `VERDICT_n3_rev5_delta_2026-10-06.md`, `VERDICT_n3_rules_rev{1..4}_*.md`, `VERDICT_n3_harness_2026-10-05.md`, `SUBMISSIONS_2026-10-06.md`
- 코드(봉인): `n3_capacity.sh`, `n3_plan.py`, `n3_analyze.py`, `n3_realized_mix.py`, `configs/pdmux_n3_A_agn.yml`
- 감사 재계산(scratchpad, 저장소 밖): `/tmp/claude-1000/-home-wonho-Experiments-KISTI/326fdd2e-1861-4666-bedf-03eb9efcd724/scratchpad/aud_r3/{N3_local.json, TK_local.json, units_local.txt, percell.py, sens.py, boot4.py, halves.py, replay.py, early.py, warm_race.py, splitcold.py, stall.py}`
