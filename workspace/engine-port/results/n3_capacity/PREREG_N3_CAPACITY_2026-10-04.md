# 사전등록 — N3: 게이트 #6 SLO 용량 격자 (Nemotron-Nano-9B-v2, VESSL A100) rev5 (2026-10-06)

> **지위**: 사전등록 **rev5**(하네스 결함 수리, 규칙 불변). rev5 델타 감사(규칙층+하네스층 합본) `VERDICT_n3_rev5_delta_2026-10-06.md` = `GO-with-caveats`(死因 0, N3C-23…27, 필수 수리 R-1…R-3 반영). 직전 rev4 델타 감사 `VERDICT_n3_rules_rev4_2026-10-05.md` = `GO-with-caveats`(N3C-22), 하네스층 `VERDICT_n3_harness_2026-10-05.md`. 작성 engine-porter(rev1–4)·메인 세션(rev5), GPU 0, 성능 판정 0, Claim 등급 변경 0.
> - **rev4 → rev5 (2026-10-06, 하네스 결함 수리 — 규칙·상수·arm·순서·예산 불변)**: shape B 1회차 Job `pdmux-n3-capacity-a4b0c2e-20261006064114`가 preflight `ABORT_SELFTEST n3_plan`(exit 4, 셀 측정 0·라벨 0)으로 끝났다. 원인: `n3_plan.py`·`n3_analyze.py`가 엔진 루트를 `Path(__file__).resolve()`로 유도했는데, VESSL `job_entry.sh`가 `engine-port/results/`를 `/data/runs/<job>/results` 심볼릭 링크로 바꾸므로 루트가 클론 밖으로 풀려 `benchmarks/pdmux_eval/analyze.py`가 MISSING(채점 단계 `import analyze`도 같은 이유로 실패했을 것). 수리: 두 파일의 `HERE`를 `Path(os.path.abspath(__file__)).parent`(심볼릭 링크 비해석)로 교체. 러너·config·`n3_realized_mix.py`·판정 규칙 바이트 불변. 테스트 `TestJobSymlinkLayout` 2개(심볼릭 링크 레이아웃 통과 + `.resolve()` 복원 변이가 Job과 같은 MISSING/ModuleNotFoundError로 실패). 진단 전문 `SUBMISSIONS_2026-10-06.md`. rev4 봉인 커밋 `a4b0c2e` 대비 diff로 델타 감사를 받는다. 변경 색인 §17-4. 1회차 Job은 셀 측정 0·`T_K.json` 없음이라 부분 집합 재라벨·Job 혼합 문제가 없고, `RESUME_FROM`으로 쓸 수 없다(`ABORT_RESUME`). 재제출은 새 Job·새 job name. caveat N3C-23…27은 rev5 델타 감사 판정서 §9를 문자 그대로 승계한다.
> - **rev3 → rev4**: 사용자 결정(2026-10-05) **R6 = P3 (ii) 허용오차**를 E2 A14 §3-1과 같은 문언·상수로 기입(§2-3)하고, 러너 P3 분기를 엄격 → (ii)로 교체(`n3_plan.p3_verdict`)했다. 6그룹 Workspace 프로브 결과를 예산 상수 검증 근거(판정 아님)로 追記(§8-7, §12-13), N3C-21(O4 해상도, E2VC-14 동형) 수록. rev3 봉인 커밋 `69fbbb0` 대비 diff로 델타 감사를 받는다. 변경 색인은 §17-3.
> - **rev2 → rev3**: rev2 델타 감사 `VERDICT_n3_rules_rev2_2026-10-05.md` = **`GO-with-caveats`**(死因 0, N3C-15…19). 새 결함 **D1(R3 근거 부재 = 통과 → unverifiable로)·D2(R6 전례 문구 정정)만** 수리한다. rev2 봉인 커밋 `81b55c3` 대비 diff로 델타 감사를 받는다. 변경 색인은 §17-2.
> - **rev1 → rev2**: claims-auditor 규칙층 감사 `VERDICT_n3_rules_rev1_2026-10-05.md` = **`GO-with-caveats`**(死因 0, N3C-1…14). 필수 수리 **R1·R2·R3·R4·R5(기본안)·R7·R8 반영**. **R6(P3 정책)은 사용자 결정 대기 — 자리표시자만 둔다(§2-3).** R1·R3는 규칙 변경이라 **델타 감사 대기**. 변경 색인은 §17.
> - **claims-auditor 규칙층 감사 대기**. `GO`(또는 `GO-with-caveats`) 전에는 `launch.sh --submit` 금지.
> - 커밋은 메인 세션이 한다(제출 전 커밋 필수, 조항 11).
> - 사용자 결정(2026-10-04, 권고 승인 — 작업 가정, 사용자가 뒤집으면 rev2): D1 = 후보 A 분할 집합
>   {(92,16),(84,24),(74,34),(64,44)} · D2 = agnostic arm 포함 → **A-agn config** 하나를 N3·5-a·5-b 공통으로 ·
>   D3 = config bridge 블록 포함 · D4 = floor는 PD-mux 활성 + decode 유휴 · D5 = PHASE_EVENTS 전 셀 ON +
>   같은 Job 안 OFF 대조 블록. TTFT SLO 방법론 = **A′**(`reports/AUDIT_slo_method_2026-10-04.md` §3-2 P-1~P-7).
> - 입력 정본: `handoff-report/VERDICT_n4_dsem_rules_2026-10-04.md`(N3 필수 조항 1–11, DS-1…7),
>   `lambda0_prereg/VERDICT_result_lambda0_908623_2026-09-14.md` §7 E1, `lambda0_vessl/{PREREG,VERDICT_*_rules,VERDICT_result}`,
>   `e2_sticky_vessl/PREREG_E2_VESSL_A13_2026-10-04.md`(읽기만), `CONFIG_CANDIDATES_2026-10-04.md`,
>   `reports/slo_definition_rationale_2026-10-04.md`(A′ 정정판), `reports/ttft_measurement_definition_review_2026-09-28.md` §3.1,
>   `reports/cost_model_design_space_2026-10-01.md` A9.
>
> **파일**(모두 `workspace/engine-port/results/n3_capacity/`)
> - `n3_capacity.sh` — 러너(shape당 1 Job, `PDMUX_N3_SHAPE=B|A`).
> - `n3_plan.py` — 등록 상수·unit 목록·예산/입장·preflight 검사·floor→`T_K.json`.
> - `n3_analyze.py` — 결정 규칙(P-3)·기술 통계·정상성·관측자 효과·bridge·예보 판정.
> - `n3_realized_mix.py` — (prefill_sms, decode_sms) 쌍 키 실현 분할 추정량 + PHASE_EVENTS 추정량(기술용).
> - `configs/pdmux_n3_A_agn.yml` — 등록 config(헤더만 "REGISTERED"로 갱신, 분할·임계 행 불변).
> - 테스트: `workspace/engine-port/tests/test_n3_capacity.py`.
>
> **줄 번호 인용 규약**: 이 회차의 새 파일은 미커밋이므로 **코드 텍스트**로 인용한다(λ5A-11과 같은 이유).

---

## 0. 한 줄 요약

target-D44 sticky-OFF B1 arm 하나를, **절대 req/s 격자**(shape B 10점·shape A 7점) × **seed 4개**(셀당 1 boot)로 VESSL A100에서 shape당 1 Job씩 측정한다.
각 Job은 첫 블록에서 floor를 재 **`T_k = k × floor`(k ∈ {3,5,10}, 초)를 파일에 동결한 뒤에만** arm 셀을 돌린다.
산출은 **k별 용량 라벨 + attainment-vs-rate 곡선 + k ∈ [1,25] 연속 곡선(기술)**이다.
이 회차는 **best static을 정하지 않고, 정책을 비교하지 않으며, 게이트 #6을 "닫았다"고 선언하지 않는다**(§11, 조항 10).

**명명(조항 2)**: 이 등록이 내는 값은 **"target-D44 sticky-OFF B1의 λ\*\_SLO(k)"** 다. "D44 용량"·"D44에서의 SLO 용량"은 금지다.

---

## 1. 이 회차가 가르는 것 · 가르지 않는 것

**가르는 것**
- shape별·k별로, 등록 격자 위에서 attainment ≥ 90%가 유지되는 최대 등록 rate(= 용량 라벨)와 그 bootstrap 구간.
- 3-b의 재배정 역할 (i) SLO 용량(게이트 #6, A4 판정서 배정 → N4 판정서 §1-3 재배정)과 (ii) 정상성(S1–S3)을 이 등록이 받는다.

**가르지 않는 것(등록)**
- best static, D 간 비교, 정책 순위, HE0 재판정(DS-7), SM 민감도(DS-2). N3는 arm이 하나다(Holm 없음).
- "λ\*\_SLO가 D44 SM에서의 용량" — DS-1·조항 2. 실현 분할은 §7 추정량으로만 말한다.
- 8192 이외 입력 길이(16k–64k)의 용량 — 미측정 열린 항목(N4 판정서 §1-3 (iii)).
- 노드 간 산포 — shape당 Job 1개, 노드 1개.

---

## 2. arm · config · 구속 설정 · 기판 (조항 1·8·9)

### 2-1. arm (모든 boot 동일, `n3_plan.py` §1 상수)

| 항목 | 값 | 강제 |
|---|---|---|
| 모델 / 리비전 | `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base` / `dc0661c829b14e5b9246c05cfa89094a0875e052` | `ABORT_MODEL_REVISION` |
| 루프 | legacy `event_loop_pdmux`(true-dual 아님, `PDMUX_TRUE_DUAL_WORKER` unset) | unset 목록 |
| 정책 | `PDMUX_R2_POLICY=fixed`, `PDMUX_R2_FIXED_DSM=44` | 러너 상수(env로 못 바꿈) |
| sticky | **unset**. boot마다 (i) 서버 로그에 `sticky partition ENABLED` 없음 (ii) green read-out `sticky_target_index` = null, 모든 group `is_sticky_target` = false | `ABORT_STICKY`·`ABORT_GREEN`(exit 6) |
| config | `configs/pdmux_n3_A_agn.yml`(sha `01675b2d…`), 6 stream group: idx0 (108,0) · idx1 (92,16) · idx2 (84,24) · idx3 (74,34) · **idx4 (64,44)** · idx5 (0,108) | `ABORT_REG_DRIFT` |
| 서버 플래그 | λ0 rev5와 같은 텍스트: flashinfer, `--enable-pdmux`, `--disable-overlap-schedule --chunked-prefill-size -1 --disable-radix-cache --mem-fraction-static 0.82 --max-running-requests 48 --context-length 16384 --random-seed 1`. **cudagraph ON**(`--disable-cuda-graph`·`--disable-piecewise-cuda-graph` 미전달) | 구조 테스트 |
| warm-up | 8요청, 동시성 1, seed 7(λ0 rev5와 같음) + **`--output-details --output-file warmup_<unit>.jsonl` 추가**(CG2 입력, 클라이언트 파일 쓰기뿐) | — |
| 측정 bench | `--dataset-name random --tokenize-prompt --random-range-ratio 1.0 --request-rate R --num-prompts N --seed s --output-details` | — |
| telemetry | 셀마다 `PDMUX_TELEMETRY_PATH`. runtime_snapshot 케이던스는 λ0와 같음(`PDMUX_DUAL_WORKER_TRACE_EVERY` unset) | unset 목록 |
| green read-out | boot마다 `PDMUX_GREEN_READOUT=1` | — |

**preflight config 검사(조항 1, CONFIG_CANDIDATES §1 함의 3)** — `n3_plan.check_config`:
- `PDMUX_R2_FIXED_DSM`(44)가 config 분할 decode SM 집합에 **정확히 한 번** 있어야 한다. 없으면 `ABORT_CONFIG`.
  (sticky OFF fixed는 목표 D가 없을 때 경고만 남기고 현재 인덱스를 유지한다 — 조용히 다른 arm이 된다.)
- `sm_group_num` = 분할 수 + 2, 그리고 ≤ 서버 `--sm-group-num`(미전달 → 기본 8).
- 목표 쌍 (64,44)가 분할에 있고, 첫 임계는 0, 모든 분할의 합이 108.
- 음성 대조(테스트): decode 44를 54로 바꾼 config, `sm_group_num` 9 config는 둘 다 검사에 걸린다.

**임계 열은 이 arm에서 inert다**(CONFIG_CANDIDATES §1·§3: fixed arm은 C7 블록이 인덱스를 정하고, 스캔은 `PDMUX_R2_POLICY` 미설정 런에서만 도달). 임계 0/18/27/36은 5-a/5-b의 agnostic-on-A arm을 위한 것이며 N3 결과에 영향을 주지 않는다 — **코드 독해 + CPU fake 근거, GPU 미실측**.

### 2-2. 같은 arm 조항(조항 1)과 config bridge(D3)

N3 config는 λ0 VESSL·E2(homog5)와 다르다(그룹 6 vs 5, D44 인덱스 4 vs 2, cudagraph 캡처 집합·메모리 잔여 다름).
CONFIG_CANDIDATES §2는 "같은 arm"이 **측정으로 확인되지 않았다**고 판정했다 → 재측정 조항이 발동한다 = **config bridge 블록**(§4-2).
5-a/5-b는 이 config를 바이트 그대로 쓴다. 다른 config를 쓰면 그 캠페인이 재측정 조항을 다시 진다.

### 2-3. 구속 설정 병기(조항 8)

| 구속 | 값 | 출처 | 기록 |
|---|---|---|---|
| running cap | 48 | 서버 플래그 | — |
| mamba state pool | 48 (= cap, `--disable-radix-cache` ∧ `--max-running-requests`) | CONSENSUS §3 항목23 | 배너 `max_mamba_cache_size`, 모든 boot에서 정확히 48 아니면 `ABORT_P3`(R6 (a)) |
| attention KV | `max_total_num_tokens`(6그룹 VESSL Workspace 프로브 2,721,290 — 기술값, 판정 아님, §8-7) | 배너 | rev4 R6 (ii): 모든 boot ≥ 786,432 ∧ boot 간 (max−min)/max ≤ 1e-4, 아니면 `ABORT_P3`(exit 8) |
| prefill batch 토큰 | `max_prefill_tokens` 16384 (shape B는 prefill 1 batch = 8192×최대 2요청) | bench `server_info` | `N3_RESULT.json` `binding_settings` |
| cap-bound 입장 비율 | PHASE_EVENTS `prefill_span_start`의 `running_bs + n_new ≥ cap ∧ queue_len > 0` 비율(E-1 §5-5 정의) | 기술용 | 셀별 `mix_*.json` |

용량 인용에는 이 표를 함께 쓴다(§11).

**R6 — P3 정책 = (ii) 허용오차 (사용자 결정 2026-10-05, rev4 기입).** E2 A14 §3-1(`e2_sticky_vessl/PREREG_E2_VESSL_A14_2026-10-05.md`)과 **같은 문언·같은 상수**다.
- **등록 규칙 (ii)**: 이 Job의 모든 boot에 대해 다음 셋이 모두 성립해야 한다. 하나라도 어기면 `ABORT_P3`(exit 8)다.
  - (a) `max_mamba_cache_size` = **48**, 모든 boot에서 정확히(허용오차 없음).
  - (b) 모든 `max_total_num_tokens` ≥ `MAX_RUNNING × CTX` = 48 × 16,384 = **786,432**.
  - (c) `max_total_num_tokens`의 (max − min) / max ≤ **1e-4**.
  - 배너를 찾지 못한 boot(`NONE`)는 abort다. `MAX_RUNNING`·`CTX`가 등록값(48, 16,384)이 아니어도 abort다.
  - 원 키는 `P3_KEYS.txt`와 셀별 `banner_<unit>.txt`에 그대로 남고, boot마다 판정 줄이 `P3_VERDICTS.txt`에 남는다.
  - 재개 Job에서 복사된 unit은 다른 boot이므로 `P3_KEYS.txt`에 들어가지 않는다(rev3와 같음).
- **집행(A14 r2 러너와 같은 방식)**: 규칙과 상수는 `n3_plan.p3_verdict`(`--p3-verdict P3_KEYS MAX_RUNNING CTX`, exit 0 PASS / 8 FAIL)에 있다. 상수 48 / 786,432 / 1e-4는 `n3_plan`에 고정되고 `test_constants`가 E2 helper와의 동일성까지 고정한다.
  러너는 **exit 0 ∧ stdout이 "PASS "로 시작**할 때만 통과시킨다(stderr는 별도 파일). helper 부재·빈 출력·거짓 PASS(exit ≠ 0)는 abort다(fail-closed).
- **1e-4는 사용자 결정 상수이며 데이터 보정이 아니다**(E2VC-15와 같은 취지). 판별 근거는 두 점뿐이다: E2 1회차 Job 안 이탈 rel 1.102e-6(PASS)과 5↔6그룹 토큰 풀 차 rel 2.70e-4(FAIL).
- **새 사실(A14 규칙층 감사 §1)**: 같은 모델·`--mem-fraction-static 0.82`에서 **6그룹 config의 토큰 풀은 2,721,290**(KISTI 907959 `srv_L1`, 908179 `srv_L2`, N3 Workspace 프로브 A1/A2 — §8-7)이고 5그룹(homog5)은 2,722,025다. 상대차 **2.70e-4**다.
  - N3는 6그룹 **단일 config**(A-agn)다. 그래서 (c)의 1e-4는 **같은 config boot 사이의 비교에만** 적용된다.
  - bridge 블록(§4-2)도 A-agn config로 boot한다. λ0 VESSL(homog5) boot는 이 Job의 `P3_KEYS.txt`에 들어가지 않는다. 그러므로 **config 차이(2.70e-4)는 (c)에 걸리지 않는다.**
  - config가 바뀌면 (c)보다 먼저 `CONFIG_DIGEST`(`ABORT_REG_DRIFT`)가 멈춘다.
- **조이는 방향(교훈 251)**: (b) 하한 786,432는 rev3 엄격 규칙에 없던 실패 조건이다((a) mamba 48은 rev3도 `EXPECT_MAMBA_CACHE`로 강제했다). 토큰 **수준값**(2,721,290)은 강제하지 않는다 → 결과 문서는 `banner_<unit>.txt`를 48 / 2,721,290(프로브 기술값)과 대조해 병기한다.
- **느슨해지는 방향**: 토큰 키가 boot마다 1e-4(≈272 토큰) 안에서 다를 수 있다. 그 표면이 판정량에 닿는 코드 경로는 A14 §3-2가 찾지 못했다(`FixedPolicy.decide`는 occupancy를 읽지 않고, KV 수요 상한 B 48×8,256 = 396,288 ≤ 786,432라 KV는 구속 자원이 아니다). 메모리 배치 차이 같은 2차 경로는 배제를 증명할 수 없다(A14 자기 공시 승계).
  토큰 키가 둘 이상이었으면 결과 문서는 unit별 값과 그 unit의 블록(floor/bridge/observer ON·OFF/grid)을 함께 싣는다(E2VC-12 형식). "P3가 boot 간 KV 풀이 같음을 확인했다"고 쓰지 않는다(E2V-10 형식).
- **전례 공시(E2 boot 11, D2 정정 rev3)**: **다른 config(homog5 5그룹)·다른 arm(sticky ON/OFF)의 E2 VESSL Job**에서 boot 11의 `max_total_num_tokens`가 앞 boot들보다 **3 토큰 적게** 나왔고, 엄격 P3가 Job을 멈췄다. (rev2의 "같은 config·arm" 표현은 거짓이라 철회한다. N3는 A-agn 6그룹·sticky unset이다 — §2-2.)
  - **정황**(`e2_sticky_vessl/PREREG_E2_VESSL_A14_DRAFT_2026-10-04.md`:79–81, 미감사 초안):
    - VESSL boot 26개(λ0 VESSL 15 + 그 E2 Job 11) 중 이탈은 이 1건뿐이다. KISTI 908623의 11 boot는 전부 2,722,025였다.
    - 그 이탈은 **그 Job의 유일한 비정상 종료(포트 충돌 `BOOT_FAILED`) 바로 다음 boot**에서 나왔다.
    - 두 가설은 판별할 수 없다: **H1** 비정상 종료 뒤 드라이버 수준 잔여, **H2** 프로세스 내부 비결정적 소량 할당.
    - N3에 반영한 O3(`^sglang::` 정리)·O4(boot 전 GPU 메모리 baseline 복귀 대기)·O1/O2(포트 충돌 원인 제거)는 **H1 경로를 겨냥**한다(§8-6). H2라면 막지 못한다.
  - N3는 shape B 한 Job에 boot가 48개라, 같은 사건이 독립적으로 일어난다면 노출 기회가 E2(20 boot)보다 많다. H1이 원인이고 O1–O4가 그 경로를 막는다면 노출은 줄어든다. 어느 쪽인지 이 등록은 판정하지 않는다.
  - (ii)는 어느 가설에서도 같은 규모(수 토큰)의 이탈에 Job을 멈추지 않는다. O4가 그 규모의 잔여를 해상하지 못한다는 한계는 N3C-21이다.
- (rev3의 "사용자가 고르면 rev4로 기입" 자리표시자는 이 rev4 기입으로 대체됐다. 문언·상수의 출처는 초안이 아니라 감사받은 A14 확정본 §3-1과 그 규칙층 판정서 `VERDICT_e2_vessl_a14_rules_2026-10-05.md` §1·E2VC-12·14·15다.) **rev4 델타 감사 전에는 제출하지 않는다.**

### 2-4. 기판(조항 9)

- **shape별 1 Job**(E1a = shape B, E1b = shape A). Job 안에서 모든 셀이 같은 노드·같은 GPU UUID다.
- B0 게이트(E2 A13과 같은 규칙, `n3_plan.check_substrate`): A100 80GB, cc 8.0, ≥ 79,000 MiB, driver major ≥ 550, MIG off, SM 108. 아니면 `ABORT_SUBSTRATE`.
- 시계: `perf_counter`·`monotonic`이 `clock_gettime(CLOCK_MONOTONIC)`가 아니면 `ABORT_CLOCK`(서버 이벤트와 bench mark가 같은 시계여야 한다).
- **셀 전후 기록**: bench 직전·직후 `/sys/fs/cgroup/cpu.stat` 스냅숏(`cpu_<unit>.txt`), unit별 epoch(`TIMING.tsv`).
  클럭·전력·스로틀 사유는 job_entry가 100 ms마다 `meta/clk.csv`에 기록한다. 분석기가 bench 창별 SW power cap(0x4) 노출 %와 평균 SM MHz를 낸다(`substrate.clk`).
- 기판 caveat(승계): λ0 VESSL에서 bench 창 power cap 노출 a_r4 27.2%·b_r3 49.4%, cgroup CPU quota 11 vs `nproc` 96(교훈 268). **어느 것도 판정에 넣지 않고 병기한다.**

---

## 3. 부하 격자 — 절대 req/s (조항 7, 조항 5, E-β)

### 3-1. 등록 격자

| shape | (in, out) | 등록 rate (req/s, **절대**) | N/seed | seed |
|---|---|---|---|---|
| **B** (E1a) | (8192, 64) | 0.10, 0.15, 0.22, 0.30, 0.38, 0.45, 0.53, 0.61, 0.69, 0.77 | 20, 20, 22, 30, 38, 45, 53, 61, 69, 77 | 3704, 2084, 4861, 4589 |
| **A** (E1b) | (256, 512) | 1.35, 1.55, 1.80, 2.10, 2.40, 2.70, 3.00 | 122, 140, 162, 189, 216, 243, 270 | 898, 3801, 903, 4364 |

- **N 규칙**: `N = max(N_min, round(rate × span))`, B span 100 s·N_min 20, A span 90 s.
- **seed 규칙**: λ0 rev5 `choose_seeds`와 같은 채점(`lambda0_plan.ebar`로 shape의 N 다중집합 위 max|Ēbar−1|) 상위 4개. `n3_plan.selftest`가 등록 리터럴과 대조하고, 모든 (seed, N)에서 Ēbar 가드 1/Ēbar ∈ [0.95, 1.05]를 확인한다.
  - λ0 seed(4386·4162·251·2630)는 B의 작은 N에서 가드를 깬다(예: 4162는 N=20에서 1/Ēbar = 0.75 — 테스트의 음성 대조). 그래서 재채점했다.
  - 이 채점은 **표본 평균 도착 간격**만 nominal에 맞춘다. 버스트 구조 선택 효과는 자유 표면 §12-3에 공시한다.
- **λ\*는 참조 라벨**: λ\*\_VESSL(A) ≈ 2.99, (B) ≈ 0.690(Job `pdmux-lambda0-vessl-1d47d3e-20261004084945`)은 **예산 모형과 간격 검사에만** 쓴다. 어떤 부하도 "×λ\*"로 정의하지 않는다(DS-4).
- **간격 — 편차 등록(R5 기본안)**: **shape B는 조항 5의 0.1× 간격을 충족하지 않는다.** B의 9간격 중 8개가 0.101–0.116×λ\*\_ref다(최대 0.116×, 브래킷 폭 최대 0.08 req/s). 이것은 등록 상수 `CLIFF_SPACING_FRAC` 0.10과도 어긋난다.
  rev1의 "간격 ≤ 0.12×로 조항 5 충족" 문장은 사실과 달라 철회한다. 이 편차는 N3C-14로 승계한다. shape A는 0.067–0.1003×다.
  `selftest`가 강제하는 0.12× 상한(`MAX_SPACING_FRAC`)은 이 편차의 상한을 고정할 뿐이다. 조항 5 충족의 증거가 아니다.
  B 재격자는 사용자 선택 대안이다(채택 시 델타 감사).
- **출발점과의 관계**:
  - shape B는 908623 §7 E1 격자 {0.10, 0.15, 0.22, 0.30, 0.38, 0.45}를 **그대로 포함**하고, 위로 0.53–0.77을 더했다(k=10 절벽이 0.45 위에 있을 것이라는 설계 사전 추정 §12-4 때문).
  - shape A는 E1 격자 {1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.4}를 **바꿨다**. 이유: E1은 TTFT SLO 3 s를 가정했으나 A′의 T_k = k×floor는 shape A에서 수백 ms 규모로 예상된다(§12-4). 그래서 절벽이 λ\* 아래로 내려온다. 1.8–3.0을 0.30 간격으로 촘촘히 두고, λ\* 위 점(3.4–4.4)은 뺐다.
  - **E-β**(N4 판정서 §4): shape A의 0.36×–0.59×λ\* 구간(λ0 rung a_r0 1.1·a_r1 1.8 사이) **안에 2점** = 1.35, 1.55. 1.8은 격자에 있다.

### 3-2. 5-a용 절벽 가드 규칙 상수(조항 7)

- arm a의 절벽 구간 `Z_a(k) = {r : 0.50 ≤ att_a(r, k) ≤ 0.97}`(`CLIFF_ATT_LO`·`CLIFF_ATT_HI`).
- 5-a 헤드라인 비교점은 **모든 arm의 Z_a 밖**이어야 한다. Z_a 안 점은 곡선으로만 보고한다.
- 절벽 둘레 격자 간격 = 0.10×λ\*\_ref(`CLIFF_SPACING_FRAC`).
- N3는 B1의 Z_B1(k)만 산출한다(`N3_RESULT.json` `cliff_zone`). 5-a 다른 arm의 Z는 5-a가 잰다. 상수는 지금 고정하며 5-a 사전등록이 바꾸려면 새 감사가 필요하다.

---

## 4. Job 구조 · 블록 (shape당 1 Job)

순서(`n3_plan.units`, 구조 테스트 고정):

```
[preflight] → floor_head → bridge×2 → observer 블록(ON/OFF 4쌍, ABBA) → 격자 round1..4 → floor_tail → [tail 분석]
```

| 블록 | boot 수 B / A | PHASE_EVENTS | 채점 | 목적 |
|---|---|---|---|---|
| floor_head | 1 / 1 | ON | — | P-2: T_k 동결 |
| bridge | 2 / 2 | **OFF** | 기술만 | D3 |
| observer | 8 / 8 (ON 4 = 격자 셀) | ON/OFF | ON 4개는 격자 셀 | 게이트 4 |
| 격자 round1–4 | 36 / 24 | ON | **결정 입력** | P-3 |
| floor_tail | 1 / 1 | ON | 진단만 | 클럭 drift 진단 |
| **합계** | **48 / 36** | | | |

### 4-1. floor 블록 (P-2, D4 리터럴)

- **구성**: 측정 셀과 **같은 서버 config·플래그**(PD-mux 활성), warm-up 8요청(동시성 1, seed 7) 뒤, **서로 다른 프롬프트 16개**(`FLOOR_N`, seed 11 → warm-up과 다른 프롬프트, radix cache off), `--max-concurrency 1`, rate 무한(직전 요청 완료 후 다음 요청 송신 → **decode 유휴**).
- **floor = 16개 송신→첫 토큰 TTFT(클라이언트)의 중앙값**(초).
- **실현 분할 실증(PHASE_EVENTS)**: 그 boot의 `prefill_span_start` 이벤트가 ≥ 16개, **전부** (prefill_sms, decode_sms) = (108, 0)이고, 마지막 16개(floor 요청, 순서대로)가 `running_bs = 0 ∧ n_new = 1 ∧ queue_len = 0`이어야 한다.
  하나라도 어기면 floor 무효 → `ABORT_FLOOR`(exit 7). **arm 셀은 하나도 돌지 않는다.**
- **등록 공식**: `T_k = k × floor`, k ∈ {3, 5, 10}, 초 단위로 `T_K.json`에 기록(`n3_plan.floor_from`). 러너는 유효한 `T_K.json`이 없으면 어떤 arm unit도 boot하지 않는다(`ABORT_FLOOR`, 구조 테스트가 순서 고정).
- **채점은 이 초 값으로만**. floor_tail(같은 seed 11 프롬프트)과 재개 Job의 floor는 `T_K_TAIL.json`·`T_K_RESUME_<job>.json` **진단**이며 재채점하지 않는다.
- 코드 근거(독해): decode가 빈 상태의 admission은 `elif` → `adjust_stream_groups` → idx0(CONFIG_CANDIDATES §7 새 블록 2). 이 근거는 측정으로 대체된다(위 실증).

### 4-2. config bridge 블록 (D3)

- λ0 VESSL의 top rung을 새 config로 다시 잰다: shape B Job에서 `b_r3`(offered 1.15, N=200), shape A Job에서 `a_r4`(offered 8.0, N=400), 각각 seed **4386·4162** 1회씩.
- **PHASE_EVENTS OFF**(자유 표면 §12-6): λ0 VESSL과 다른 것을 config(+엔진 1파일의 OFF 경로, +Job·노드)로 한정하기 위해서다.
- 산출(기술만): achieved = completed/duration, 4추정량(쌍 키). λ0 VESSL n=4(achieved, homog5) 옆에 **나란히** 놓는다.
- **판정 아님**: "같은 arm" 판정을 하지 않는다. λ0 §4-4의 "3% ∧ 범위 비중첩" 규칙은 **재사용하지 않는다**(그것은 λ0 기판 비교 규칙이지 N3 estimand가 아니다). 이 블록은 5-a/5-b가 λ0 VESSL 값을 "새 config 옆"에 둘 때의 **인용 자격 기록**이다.

### 4-3. correctness gate 2 (CG2) — PHASE_EVENTS ON vs OFF 출력 동치

- 같은 8 warm-up 프롬프트(seed 7)를 동시성 1·greedy(bench_serving 기본 temperature 0.0)로 생성한 텍스트를, **floor_head(ON)**와 **이 Job에서 처음 실행되는 OFF boot**(보통 첫 bridge) 사이에서 바이트 비교한다.
- 8/8 동일 + 빈 출력 0 → `CG2_PASS`. 아니면 `ABORT_CG2`(exit 9) — 채점 셀 전에 멈춘다.
- ★자기 공시: 동시성 1 greedy가 boot 간 결정론적이라는 것은 가정이다. 불일치는 "계측이 출력을 바꿨다"의 증거가 아니라 **조사 대상**이다(fail-closed, E-1 §5-3 2와 같은 처리).
- 전 boot 기술 요약: ON boot·OFF boot 각각의 서로 다른 출력 튜플 수(`cg2_all_boots`).

### 4-4. 관측자 효과 블록 (D5, 게이트 4)

- 관측 rate: B **0.30**, A **1.80**(둘 다 격자 점). seed 4개 각각 ON(= 그 rate의 격자 셀)·OFF(PHASE_EVENTS unset) 2 boot를 **붙여서**, 순서 ABBA(ON,OFF / OFF,ON / ON,OFF / OFF,ON).
- 지표 2개: 셀 TTFT p50, 요청별 ITL p95의 중앙값. seed 짝마다 ln(ON/OFF).
- **OBSERVER_OK** ⇔ 두 지표 모두 n ≥ 4, |평균 ln 비| < ln 1.03, 95% paired bootstrap CI(`analyze.paired_bootstrap_ci`, 10000 아닌 2000 표본·seed 1)가 0 포함. n < 4면 `OBSERVER_UNRESOLVED_N_BELOW_4`, 그 밖은 `OBSERVER_FAIL`.
- **효과**: 라벨은 어느 경우든 계산한다(채점 셀 전부 ON이라 대칭). FAIL/UNRESOLVED면 모든 N3 인용에 "PHASE_EVENTS ON 조건에서"를 붙이고, PHASE_EVENTS 기반 수치는 "관측자 효과 미검증 하 기술값"으로만 쓴다.
- 한계: "CI가 0 포함"은 부재의 증거가 아니다. 관측 rate 1점에서만 잰다(다른 rate의 관측자 효과는 외삽).

### 4-5. 격자 블록

- round i(seed i)마다 관측 rate를 뺀 나머지 rate 전부를 돈다.
- **round 1·3은 내림차순, 2·4는 오름차순**(R1, rev2). 그래서 마지막 unit(floor_tail 직전)은 **최고 rate**다.
  - 시간 drift 균형은 rev1과 같다(교대).
  - 예산 절단이 나면 **최고 rate부터** n이 3으로 줄어든다. 첫 실패 규칙이 반드시 읽어야 하는 저 rate 셀은 마지막까지 보존된다.
  - rev1 순서에서는 마지막 unit이 최저 rate(B `g_r010_4589`)였다. B 최악 코너 여유가 8 s뿐이라, 절단 1건만으로 세 k 전부가 `UNRESOLVED_N_BELOW_4`가 될 수 있었다(감사 §2-9).
- 테스트: 마지막·첫 격자 unit이 최고 rate인지, round별 정렬 방향, 절단 1·3 unit을 모사했을 때 n < 4가 되는 최저 rate = 잘린 rate 중 최저(> 격자 최저).

---

## 5. 결정 규칙 (P-3, 실행 전 고정 — `n3_analyze.py` 결정 경로)

### 5-1. 요청 술어 (정본, 재정의 없음)

`benchmarks/pdmux_eval/analyze.py`의 `RequestResult.passes(ttft_slo_ms, itl_slo_ms)`(sha `1be49511…` 고정):
**TTFT ≤ T_k ∧ 요청 내부 token-ITL p95 ≤ 50 ms**. `load_bench_serving_rounds`로 읽는다(라운드 1개, duration 합산 규약 그대로).

### 5-2. 셀 유효성

셀(rate, seed)은 다음을 모두 만족할 때만 유효하다. 무효 셀은 그 rate의 n을 줄인다(대체값 없음).
- bench 레코드 1개, `errors` 0, `completed` = N, `total_input_tokens` = N×in, `output_lens` 전부 = out, `random_range_ratio` 1.0, `request_rate` = 등록값.
- 예산 집행기가 bench를 자르지 않았다(`budget_killed_<unit>` 없음).
- **arm 동일성 0건(R3, rev2)**:
  - 채점 셀(격자·obs_on)과 obs_off·bridge 셀에서 다음 둘이 모두 **0건**이어야 한다.
    - `decode_iteration` 이벤트의 (prefill_sms, decode_sms)가 config의 비표적 분할 (92,16)·(84,24)·(74,34) 중 하나. 이 이벤트는 decode sync 때 기록되므로 **전환 뒤 인덱스**다.
    - decode-busy `runtime_snapshot`이 같은 비표적 분할에 있다.
  - 위반 셀은 **무효**다(retract와 같은 처리, 대체값 없음).
  - telemetry가 없어 검사할 수 없는 셀도 무효다(`arm identity unverifiable`).
  - **D1 (rev3) — 근거 부재는 통과가 아니다.** 다음 셀도 `unverifiable` = 무효다(문턱은 0뿐, 새 상수 없음).
    - decode-busy 스냅숏이 0개인 셀(`busy_snapshots_seen == 0`, 예: 러너가 미리 만든 빈 telemetry 파일 — writer가 조용히 실패해도 rev2에서는 0건 = 통과였다).
    - PHASE_EVENTS ON이고 R7 표식이 없는데 `decode_iteration`이 0개인 셀(`decode_iterations_seen == 0`).
    - PHASE_EVENTS OFF 셀(obs_off·bridge)과 R7 표식 셀은 스냅숏 다리만으로 판정한다.
    - 변이 시험: 빈 telemetry 셀 → 무효(그 rate n = 3 → `UNRESOLVED_N_BELOW_4_5`); 스냅숏만 있는 ON 셀 → 무효; 대조로 같은 파일 + R7 표식 → 유효, 스냅숏만 있는 OFF 셀 → 유효.
  - PHASE_EVENTS OFF이거나 boot 중 꺼진 셀은 스냅숏 다리만 증거가 된다(`decode_iterations_seen` 기록).
  - 사전 근거: λ0 VESSL 358,583 레코드에서 비표적 분할 0건(감사 §4 R3). 로컬 셀프테스트로 λ0 7셀이 homog5 비표적 (92,16)·(16,92)에서 0건임을 재확인했다.
  - 근거 경로: 임계 열이 fixed arm에서 inert라는 것이 GPU 미실증(조항 1)이다. 이 조건이 그것을 측정으로 대신한다. 오발화하면 해당 rate가 `UNRESOLVED`로 간다(정직한 쪽).
- **retract 0**(조항 11, X-10): 서버 로그 `#retracted_reqs:` 합이 0. 요청 단위 식별이 이 하네스에서 불가능하므로(클라이언트 기록에 서버 stamp 없음) **retract가 하나라도 있으면 셀 전체를 제외**한다. 사전 예상은 0이다(KV 2.72M 토큰 ≫ 48×8256 ≈ 0.40M — 산술, 미측정).

### 5-3. attainment · 용량 · 라벨

- `attainment(r, k)` = rate r의 **유효 seed 전부를 합친** 통과 비율(seed당 N 같음 → seed 평균과 같다).
- `용량(k)` = **첫 실패 규칙**: 등록 rate를 오름차순으로 읽어, attainment < 0.90인 첫 rate(r_fail) 바로 아래 rate. `X_ATTAIN = 0.90`.
- 라벨(k마다, 문자 그대로):

| 라벨 | 조건 |
|---|---|
| `CAPACITY_BRACKETED_k` | 첫 실패 rate가 있고 그 아래 통과 rate가 있다. 용량 = r_cap, 구간 [r_cap, r_fail) |
| `CAPACITY_BELOW_GRID_k` | 최저 등록 rate가 이미 실패. 용량 < 최저 rate |
| `CAPACITY_ABOVE_GRID_k` | 모든 등록 rate 통과. 용량 ≥ 최고 rate |
| `UNRESOLVED_N_BELOW_4_k` | 규칙이 읽어야 하는 rate(최저 ~ r_fail, 실패가 없으면 전부) 중 유효 seed < 4인 것이 있다. **측정 부족이며 발견이 아니다**(게이트 #21) |

- **T_K 유효성(R2, rev2)**:
  - 러너(`tk_valid`, 재개 검사)와 분석기가 `n3_plan.tk_problems`로 `T_K.json`을 검사한다. 다음 중 하나면 거부한다.
    - `valid` ≠ true
    - **`dryrun` 키 존재**(DRYRUN placeholder)
    - `ttfts_s`가 없거나 16개가 아님
    - floor_s ≠ median(ttfts_s)
    - T_k ≠ k×floor_s
  - 거부되면 arm unit은 boot하지 않고(`ABORT_FLOOR`/`ABORT_RESUME`), 분석기는 채점하지 않는다(`NO_VALID_FLOOR`).
  - rev1은 DRYRUN Job을 재개하면 placeholder T = 3/5/10 s로 채점될 수 있었다(P-2 fail-open).
- **`ITL_BOUND_k`(R4, rev2, 새 상수 없음)**: k별 등록 라벨·용량이 같은 요청을 ITL 65 ms로 재채점한 `itl_ladder_capacity[k{k}_itl65]`의 라벨·용량과 다르면 `ITL_BOUND_k` 플래그를 단다. 이때 라벨은 결합 술어 용량이며 "TTFT 용량"으로 쓰지 않는다(N3C-5).
- **seed 집합 불일치(N3C-12)**: r_fail 위 셀이 빠져 bootstrap의 짝 seed 집합이 라벨의 seed 집합보다 작으면, 두 n을 함께 적는다.
- **비단조 플래그** `non_monotone`: r_fail 위에서 attainment ≥ 0.90인 rate가 있으면 표시한다(라벨은 바꾸지 않는다).
- **CI**: seed 짝 bootstrap(rate 전체에서 seed를 함께 재표집, 2000회, seed 1) → 용량 분포의 2.5/97.5 백분위(격자 값, `-1` = 격자 아래)와 rate별 attainment CI. n = 4면 서로 다른 재표집은 35개뿐이라 구간이 거칠다(감사 #14) — 공시.
- **정상성 게이팅(조항 6)**: 용량 rate 셀의 (rate, k) 정상성 라벨이 `UNSTABLE`이면 `magnitude_citable = false` — 그 셀의 attainment·TTFT 크기는 제출하지 않고 라벨·구간만 쓴다.
- N3는 arm이 하나다. arm 쌍 비교·Holm 보정은 없다(P-3의 다중 arm 부분은 5-a/5-b가 진다).

### 5-4. 정상성 S1–S3 (cost_model A9, ttft_measurement_definition_review §3.1 — 문턱 실행 전 고정)

| 기호 | 정의 | 문턱(발화) |
|---|---|---|
| S1 | seed마다 도착 순서 마지막 1/3 TTFT p50 ÷ 첫 1/3 TTFT p50, seed 평균 | > **1.7**(기존 정의) |
| S2 | seed마다 (후반 위반율 − 전반 위반율), 술어 = (T_k, 50 ms), seed 평균 — **k마다** | > **0.10** |
| S3 | bench mark 창 안 runtime_snapshot(phase ≠ startup)의 표본 순서 대 `prefill_queue_depth` Spearman ρ, seed 평균. 표본 < 10이거나 상수면 NA(비발화) | ≥ **0.50** |

하나라도 발화 → (rate, k) `UNSTABLE`, 아니면 `STATIONARY`. **크기는 STATIONARY 셀만** 보고한다. 원시값은 아티팩트에 남는다.
- 전제: 요청 거부 없음(`--max-queued-requests` 미설정). S3는 엔진 측 독립 계측이고 시간가중이 아니라 순위 기반이다.

---

## 6. 기술 통계 (판정 입력 아님)

- **연속 곡선**: k ∈ [1, 25](0.25 간격) 각 점의 attainment-vs-rate와 그 k의 용량 라벨(`capacity_curve`). ITL 통과 요청의 TTFT/floor 경험 분포 p50/p90/p95/p99(P-3).
- **99% attainment**: k별 X = 0.99 용량 라벨 + 셀별 요청 수·기대 실패 건수(0.01×요청 수) 병기(P-3·P-5).
- **ITL 사다리**(P-5, static arm 재채점은 재스코어 아님 — `FixedPolicy`는 SLO를 읽지 않는다): {40, 50, 60, 65} ms × k {3,5,10}의 attainment와 용량 라벨.
- **구속 술어 분해**: (rate, k)마다 TTFT만 실패 / ITL만 실패 / 둘 다 실패 비율. 요청별 ITL p95 ∈ [45, 55] ms 비율.
- **절벽 구간** Z_B1(k)(§3-2).
- **절대 좌표 표(P-6)**: T_k(초)를 MLPerf 405B P99 6 s(**모델 규모 불일치** 명기)와 Bullet arXiv-Sum 1.5 ms/tok × L(B: 12.3 s, A: 0.384 s, **분자가 TTFT인지 prefill 시간인지 원문 미확인** 명기) 옆에 둔다. 결과 문서에서 채운다.

---

## 7. 실현 분할 · 노출 (조항 3·4, `n3_realized_mix.py`, 기술용)

### 7-1. 쌍 키 4추정량 — E2 ADDENDUM A2·E2C-8′ 규약 리터럴

`e2_realized_mix.py`는 `D44_STREAM_INDEX = 2`(homog5) 하드코딩이라 N3 config(D44 = idx4)에 쓸 수 없고, E2 Job 실행 중이라 봉인이다. 새 모듈은 **같은 네 추정량**을 표적 **(prefill_sms, decode_sms) = (64,44)** 쌍으로 고른다.
- 모집단: `runtime_snapshot`, phase ≠ startup, **파일 전체**(warm-up 포함, E2C-16). probe 없음 → A3 probe_end = None.
- decode-busy = `decode_running_batch_size > 0`(A2 리터럴).
- E-cnt / E-time(다음 비startup 표본까지 간격, 상한 없음, 마지막 표본 제외) / E-iter(왼쪽 busy, Δdecode_iterations 가중) / E-qcond(`prefill_queue_depth > 0`). E-pact는 항등 가드.
- **양성 대조(로컬 제출 게이트)**: λ0 VESSL 아카이브(homog5, D44 = idx2)의 7셀(a_r0, a_r2, a_r4, a_r4_s2, b_r0, b_r3, b_r3_s2)에서 쌍 키 결과가 봉인 E2 도구(읽기 전용 import, 바이트코드 미생성)와 **7/7 비트 동일**(busy_n, 각 num/den, E-time 초, E-pact). 외부 기대값(λ0 VESSL 결과 감사 C9) a_r4 8.66/13.89/8.76/18.33 · b_r3 90.48/97.38/91.68/100.00 재현.
- 음성 대조: 표적 (92,16)으로는 7셀 모두 불일치. 인덱스 재배치(2→4, 4→5, 3→6) 사본에서 쌍 키 값은 불변, E2 도구의 E-cnt는 움직인다(이 파일이 필요한 이유).
- 이 아카이브는 git 밖이라 Job 안에서는 합성 셀프테스트만 돈다(추정량은 판정 입력이 아니다 — λ0 VESSL과 같은 처리).

### 7-2. PHASE_EVENTS 추정량(새 규약, bench mark 창 = 창 안 첫~마지막 작업 이벤트)

| 이름 | 정의 |
|---|---|
| W-dec | decode 벽시계(Σ t_sync − t_launch)의 쌍별 비율 |
| W-idx | **벽시계 인덱스 점유**: 각 작업 이벤트(`prefill_span_start`, `decode_iteration`@t_launch)의 쌍을 다음 작업 이벤트까지 유지한 근사(해상도 = 루프 1회) |
| X-tok | 토큰 가중 노출 = Σ bs·[쌍 = 표적] / Σ bs |
| CO-a | 동거율 정의 (a) = \|PF ∩ DEC\| / \|창\|(호스트 관측 구간) |
| cap-bound | §2-3 |
| pin | 동거 decode iteration(`prefill_in_flight ∧ ¬kernel_done_wait`) 중 표적 쌍 비율 |

**R7 (rev2) — boot 중 PHASE_EVENTS 자동 비활성 검출**:
- 엔진은 이벤트 기록·emit이 실패하면 그 boot의 나머지 동안 PHASE_EVENTS를 **조용히 끈다**(`_phase_events_disable`, 경고 1회).
- 러너는 boot마다 서버 로그의 `PDMUX_PHASE_EVENTS disabled after failure`를 검출해 `phase_events_disabled_<unit>` 표식과 `PHASE_EVENTS_DISABLED.txt`를 남긴다.
- 분석기는 그 셀의 PHASE_EVENTS 기반 기술값(W-dec·W-idx·X-tok·CO-a·cap-bound·pin·X-req)을 **무효**로 하고(`phase_events_voided`), 관측자 효과의 ON/OFF 대칭 짝에서 **제외**한다. 제외로 n < 4가 되면 `OBSERVER_UNRESOLVED_N_BELOW_4`.
- 채점(클라이언트 TTFT/ITL)은 영향받지 않는다. 다만 그 셀은 "PHASE_EVENTS 전 구간 ON" 조건을 만족하지 않았다고 병기한다.

### 7-3. 요청별 노출 X-req (조항 4) — **재구성 기반 기술값**

클라이언트 기록에는 절대 시각이 없다. 그래서 요청 i의 송신 시각을 `a_i = t0 + Σ_{j<i} Exp(1/rate)`(np.random.seed(seed), bench_serving `get_request`의 RNG 재생; λ0 `ebar`와 같은 스트림)로 재구성한다.
- t0 = 창 안 **두 번째** `prefill_span_start`(첫째는 bench_serving 자체 warm-up 요청이고 그 뒤 1 s sleep). 간격 < 1 s면 `UNANCHORED`.
- decode 토큰 시각 = a_i + ttft_i + cumsum(itl_i). 각 토큰을 t_sync ≤ 토큰 시각인 마지막 decode iteration에 배정한다.
- exposure_i = 표적 쌍 iteration에 배정된 decode 토큰 비율. p10/p50/p90·평균·≥0.5 비율, 첫 토큰 잔차(앵커 품질) 병기.
- **"요청별 SM 측정"으로 쓰지 않는다.** 클라이언트 지연(detokenize·HTTP)만큼 배정이 한 iteration 늦을 수 있다.

### 7-4. 필수 병기(조항 3)

λ\*\_SLO(k)를 인용할 때는 그 용량 셀(및 바로 위 실패 셀)의 E-cnt/E-time/E-iter/E-qcond(규약명 포함), W-idx 점유(모든 쌍), X-tok, 최종 `dropped_events`를 함께 적는다. `dropped_events` ≠ 0이면 "표본 손실 하 기술값"(LV-12).

---

## 8. 러너 — 등록 무결성 · 노브 · 예산 · 재개

### 8-1. 등록 무결성 (GPU 0에서 중단)

- `n3_plan.verify_digests`: config(`01675b2d…`), `analyze.py`(`1be49511…`), `lambda0_plan.py`(`2e51280c…`) + **src 미러 9개**가 등록 매니페스트 항목과 같은지. 아니면 `ABORT_REG_DRIFT`.
- `REGISTRATION_SHA256.txt`: 이 사전등록·러너·plan·analyze·mix·config·정본 술어·lambda0_plan + 판정서들, git HEAD, 결과 밖 미커밋 수. digest ≥ 8개 아니면 `ABORT_REG`.
- **엔진 트리 = HEAD 등록 digest 일치**(E2의 "1파일 차이" 규약이 아님): sync 후 manifest 25항목 전부가 `n3_plan.ENGINE_MANIFEST`(= job 908623 manifest + `multiplexing_mixin.py` `ba5835aa…`)와 경로 접미별로 같아야 한다. 아니면 `ABORT_ENGINE_TREE_DRIFT`. thread-local role patch 미적용이면 `ABORT_ENGINE`(correctness gate 3).
- preflight 셀프테스트·단위 테스트(`n3_plan`/`n3_realized_mix`(합성)/`n3_analyze` selftest, `test_phase_events`, `test_green_readout`, `test_n3_capacity`의 TestRegistration·TestUnits·TestFailClosedWiring)를 설치 트리에서 돈다. 실패 → `ABORT_SELFTEST`/`ABORT_PREFLIGHT`.

### 8-2. 노브 허용 목록

- 허용: job_entry/launch 식별자(`PDMUX_COMMIT`·`_JOB_NAME`·`_CAMPAIGN`·`_TARGET`·`_IMAGE`·`_SPEC`·`_ROOT`·`_PROJECT_ROOT`·`_IO`·`_DATA`·`_OPT`), `PDMUX_N3_SHAPE`(A|B만), `PDMUX_N3_RESUME_FROM`, `PDMUX_N3_DRYRUN`, **`PDMUX_PHASE_EVENTS`**(이 등록이 boot마다 스스로 설정하는 노브 — 상속값은 `KNOBS.txt`에 기록 후 unset, ON unit에서만 `=1` export).
- 테스트 전용(`PDMUX_N3_TEST_*`, `PDMUX_TEST_SKIP_SYNC`)은 DRYRUN이 아니면 `ABORT_TEST_KNOB_IN_MEASUREMENT`. `PDMUX_ARRAY_SPEC`는 비어 있어야 한다.
- 그 밖의 `PDMUX_*`(예: `PDMUX_STICKY_PARTITION`, `PDMUX_TTFT_SLO_MS`)는 `ABORT_ENV`(테스트).

### 8-3. 예산 3층 · 하드캡 · 입장 (단가 $1.48/h)

per-boot 부대비용 = λ0 VESSL 실측(E2 A13 §5-1): A 160.1 s · B 121.0 s, 첫 boot cold +92.1 s. preflight 600 s(**N3 미측정**).
bench 시간 모형 = max((N−1)/rate, N/λ\*\_ref) + drain(A 25 s, B 3 s); floor bench A 110 s · B 40 s; bridge = λ0 VESSL 실측 duration. 최악 코너 = 부대비용 ×1.25, 용량 λ\*\_ref×0.8.

| shape | 층 | 컨테이너 초(tail 제외) | GPU-h | 비용 | 입장 |
|---|---|---|---|---|---|
| B | 보통 | 12,126 | **3.37** | **$4.99** | 48/48 |
| B | 최악 코너 | 13,792 | 3.83 | $5.67 | 47/48 (floor_tail만 입장 거부) |
| A | 보통 | 10,612 | **2.95** | **$4.36** | 36/36 |
| A | 최악 코너 | 12,261 | 3.41 | $5.04 | 36/36 |
| 둘 다 | 하드캡(셀 구간) | 14,400 / Job | 4.00 / Job | $5.92 / Job | — |

- 두 Job 보통 합계 ≈ 6.3 GPU-h(≈ $9.35). 블록별 분해는 `BUDGET_TABLE.json`.
- **하드캡 = 스크립트 내부 wall-clock 집행기 세 겹**(λ0/E2 패턴): (1) job_entry `start_utc`부터 14,400 s − tail 600 s가 셀 마감 (2) **입장 규칙**: unit 시작 전 남은 시간 ≥ 그 unit의 최악 코너 비용(`--need`), 아니면 `NOT_ATTEMPTED_BUDGET` 후 이후 unit 미시도 (3) warm-up·bench `timeout -k 30 <남은 시간>` + watchdog(마감에 bench·서버 종료, `BUDGET_HARDCAP_FIRED`). preflight 단계도 `timeout <남은 시간>`, tail 분석은 `timeout <캡 끝까지>`.
- 변이 검증(테스트): 가짜 bench가 마감에서 잘린다; **두 kill 층을 모두 제거**한 변이는 마감을 넘긴다(검출); watchdog만 제거한 변이는 여전히 잘린다(**등가 변이**, 교훈 254 — 의도된 중복).
- **과금 하드캡이 아니다**(LV-13 형식): tail·반출에는 kill 층이 없다. 사용자 승인 문언에 "Job당 보통 ≈3.0–3.4 GPU-h, 셀 구간 캡 4.0 h, 꼬리 비한정"을 명시한다.

### 8-4. 재개

- 자동 재시도 없음. 새 Job에 `--env PDMUX_N3_RESUME_FROM=<이전 job>`.
- 조건: `UNITS.txt` 바이트 동일, 등록 digest 동일, 이전 Job `T_K.json` 유효. 아니면 `ABORT_RESUME`.
- `unit_<name>.complete`가 있는 unit만 복사한다. **`T_K.json`은 이전 Job의 초 값을 그대로 쓴다**(P-2 "초 동결"). 재개 Job은 floor를 다시 재되 `T_K_RESUME_<job>.json` 진단으로만 둔다.
- 재개 결과는 **Job 혼합**이다. `RESUMED_UNITS.tsv`(출처 Job·GPU UUID)를 결과 문서에 반드시 적고, 노드가 다른 셀을 섞었다는 사실을 라벨 옆에 병기한다.

### 8-5. 중단 코드

| 코드 | exit | 조건 |
|---|---|---|
| `ABORT_ENV` / `ABORT_TEST_KNOB_IN_MEASUREMENT` | 2 | §8-2 |
| `ABORT_REG` / `ABORT_REG_DRIFT` / `ABORT_CONFIG` | 3 | §8-1, §2-1 |
| `ABORT_SUBSTRATE` / `ABORT_MODEL_REVISION` / `ABORT_TRACE` / `ABORT_CLOCK` | 3 | §2-4 (DRYRUN에서는 기록만) |
| `ABORT_ENGINE_TREE_DRIFT` / `ABORT_ENGINE` | 3 | §8-1 |
| `ABORT_SELFTEST` / `ABORT_PREFLIGHT` | 4 | §8-1 |
| `ABORT_NO_FREE_PORT` / `ABORT_PORT_RANGE` | 5 | §8-6 O1·O2 |
| `ABORT_STICKY` / `ABORT_GREEN` | 6 | §2-1 (boot마다) |
| `ABORT_FLOOR` | 7 | §4-1 |
| `ABORT_P3` | 8 | §2-3 (boot마다) |
| `ABORT_CG2` | 9 | §4-3 |
| `ABORT_UNIT_COUNT` / `ABORT_RESUME` | 2 | §8-4 |
| boot 실패 2연속 | — | 이후 unit 미시도(floor_head boot 실패는 `ABORT_FLOOR`) |


### 8-6. 포트 선택 · teardown 규약 (E2 진단 반영 — `e2_sticky_vessl/A14_DRAFT_runner_ops_2026-10-04.patch` O1–O4를 동등 반영, 패치 파일은 적용하지 않음)

**진단(다른 engine-porter 회차, E2 Job)**: λ0/E2에서 물려받은 포트 선택은 39100–39899에서 뽑고 `ss -ltn`(LISTEN만)으로 검사했다. 이 범위는 커널 ephemeral 범위(32768–60999) 안이다. 그래서 직전 bench 클라이언트가 쓰고 TIME_WAIT로 남은 포트를 고를 수 있고, 서버 bind가 EADDRINUSE로 실패한다(E2 boot 10 실패, 로컬 재현). N3 초안 러너도 같은 로직이었다.

| ID | 규약 | 판정에 닿는 경로 |
|---|---|---|
| **O1** | 서버 포트 창 = **20000–20799**(ephemeral 범위 밖). 시작 시 `/proc/sys/net/ipv4/ip_local_port_range`·`ip_local_reserved_ports`를 `PORT_RANGE.txt`에 기록한다. 창이 ephemeral 범위와 겹치거나 범위를 읽지 못하면 boot 전에 `ABORT_PORT_RANGE`(exit 5). unit마다 고른 포트와 시도 횟수를 기록한다 | 없음(boot 실패 확률만 바꾼다) |
| **O2** | 후보 포트는 `ss -Htan`의 **모든 TCP 상태**(TIME_WAIT 포함) 소켓의 로컬 포트와 겹치지 않을 때만 쓴다(`port_in_use`) | 없음 |
| **O3** | teardown은 기존 순서(kill, 5 s, `pkill -9 -f "launch_server.*--port P"`, 10 s)를 **그대로 둔 뒤** `pkill -9 -f "^sglang::"`를 더한다. SGLang 자식 프로세스는 setproctitle로 cmdline이 `sglang::scheduler` 등으로 바뀌어 기존 패턴에 안 걸린다. 컨테이너당 서버가 하나라 다른 arm을 칠 수 없다 | 이전 boot의 잔존 프로세스가 GPU를 점유한 채 다음 boot와 겹치는 경로를 닫는다 |
| **O4** | boot 전마다 SGLang 프로세스 0개 ∧ GPU used 메모리 ≤ 첫 boot 전 baseline + 64 MiB가 될 때까지 최대 180 s(5 s × 36) 기다리고, 폴 횟수·used MiB·NVML 바이트를 `SETTLE.txt`에 기록한다. **기록용이며 gate가 아니다**(gate는 `ABORT_P3`). DRYRUN에서는 생략 | 기록만. `max_total_num_tokens` drift가 생기면 장치 수준 잔류와 프로세스 내부 변동을 가르는 재료 |

- 테스트(`TestPortAndTeardown`, Job preflight에도 포함):
  - **O2 변이**: 가짜 `ss`가 후보 포트에 TIME-WAIT 소켓을 보이게 했을 때, 옛 검사(`ss -ltn | grep`)는 놓치고 새 `port_in_use`는 잡는다. 무관한 포트는 free다.
  - **O1 범위**: 기본 범위(32768–60999)에서 통과한다. 겹치는 범위(10000–30000)에서는 `ABORT_PORT_RANGE`가 난다. 대조: 옛 창(39100)은 기본 범위에서 abort된다.
  - **O3**: teardown 본문에 두 pkill이 모두 있다.
  - **O4**: settle이 포트 선택보다 앞에 있고, `fail`/`exit`가 없으며(gate 아님), 36회 폴이다.
- λ0 VESSL과의 bridge 비교에 미치는 영향: 포트 번호와 teardown은 서버 동작(스케줄링·커널)에 닿지 않는다. settle 대기는 boot 사이 유휴 시간만 늘린다. 둘 다 boot당 부대비용에만 들어간다(settle 최대 +180 s, 보통 0–15 s 예상, 미측정).

### 8-7. 6그룹 config Workspace 프로브 (2026-10-05) — **예산 상수 검증 근거, 판정 아님** (rev4 追記)

출처: `probe_6group_2026-10-05/SUMMARY.md`(+ `TIMING_TABLE.tsv`, `srv_*.log`, `SHA256SUMS` 로컬 재검증 OK). Workspace `pdmux-probe`, A100 SXM, 커밋 `69fbbb0` 트리를 `install_runtime.sh`로 설치.
측정이 아니고 캠페인 셀도 아니다. 이미지 digest가 등록 pin과 다르다(`c85fc198` vs `1e0d335b`, SUMMARY DEVIATION 1 — 엔진 코드는 커밋에서 새로 설치). 각 행 n = 1이다.

| 항목 | 6그룹 A-agn (A2 warm) | 5그룹 homog5 (H5 warm) | 차이 |
|---|---|---|---|
| boot 벽시계 | 59.11 s | 60.11 s | −1.0 s |
| cudagraph 캡처 시간 | 14.12 s | 12.67 s | **+1.45 s** |
| 캡처 메모리 | 1.27 GB | 1.07 GB | **+0.20 GB** |
| `max_total_num_tokens` | **2,721,290** | 2,722,025 | −735 (rel 2.70e-4) |
| `max_mamba_cache_size` | 48 | 48 | 0 |
| green read-out | 6/6 realized == target, `sticky_target_index` null | 5/5 | — |

- cold 첫 boot(A1)는 124.31 s(캡처 51.60 s)였고, 토큰·mamba·green은 A2와 같았다.
- 컨테이너 사실: `ip_local_port_range` = 32768–60999(O1 창 20000–20799와 겹치지 않음), `ip_local_reserved_ports` 비어 있음, 3회 teardown 뒤 `ss -Htan` TIME_WAIT 0. cgroup CPU quota 11 vs `nproc` 96.
- **per-boot 부대비용 상수(A 160.1 s · B 121.0 s, §8-3)는 바꾸지 않는다.** 그 상수는 boot + settle + 입장 장부를 포함한 unit 부대비용이고, 프로브는 boot 벽시계만 쟀다. 그래서 1:1 비교가 아니다(SUMMARY가 그렇게 명시한다).
  다만 6그룹 warm boot 벽시계(59.1 s)가 5그룹(60.1 s)보다 길지 않으므로, **"6그룹이라서 상수를 올릴 근거"는 이 프로브에 없다.**
- **캡처 +1.45 s의 입장 영향(공시)**: 부대비용이 boot마다 실제로 +1.45 s 늘었다고 두면, shape B 최악 코너에서 마지막 격자 unit `g_r077_4589`(48개 중 47번째)의 입장 여유 **7.86 s**(`n3_plan.simulate`·`need_s`, rev3 §4-5의 "8 s")가 앞선 46 boot × 1.45 s = **66.7 s**만큼 줄어 **음수(−58.8 s)**가 된다.
  - 그 경우 최악 코너 입장은 47/48 → **46/48**이다(floor_tail은 원래 최악 코너에서 거부, 이번에는 `g_r077_4589`도 거부). 보통 코너는 48/48 그대로다(최소 여유 1,643 → 1,575 s). shape A는 최악 코너에서도 36/36이다(최소 여유 1,539 → 1,488 s).
  - **측정 결과는 왜곡되지 않는다.** 입장 규칙이 unit을 시작 전에 거부하므로(`NOT_ATTEMPTED_BUDGET`), 잘린 bench가 채점에 들어가지 않는다.
  - 대가는 n이다. 0.77의 seed 4589가 빠지면 0.77은 n = 3이다. 첫 실패 규칙이 0.77을 읽어야 하는 k(0.69 이하에서 실패가 없는 k)는 `UNRESOLVED_N_BELOW_4_k`가 된다 — 측정 부족이지 발견이 아니다(N3C-15와 같은 경로).
  - 이 계산은 "+1.45 s가 매 boot 부대비용에 그대로 더해진다"는 가정이다(n = 1 warm 쌍, 실제로는 boot 벽시계가 오히려 1.0 s 짧았다). 예산 상수·입장 규칙은 바꾸지 않는다.
- §16 첫 항목(6그룹 VESSL boot 미측정)은 Workspace에서 **기능**(boot·캡처·green 6/6·토큰 값)이 관측됐다. 그러나 Job 안 측정이 아니므로 그 항목의 fail-closed 확인 경로(`ABORT_GREEN`·`ABORT_P3`)는 그대로 유지된다.

---

## 9. 반증 가능 예보 (조항 11 "저부하 E-time 예보" 포함 — 판정 입력 아님)

`n3_analyze.forecasts`가 결과에서 기계적으로 `HELD`/`FALSIFIED`/`UNRESOLVED`를 낸다(E-time은 seed 4개 평균, 4개 미만이면 UNRESOLVED).

| ID | 예보 | 반증 조건 | 근거(예상일 뿐) |
|---|---|---|---|
| **F-B1** (저부하 희석) | shape B에서 rate ≤ 0.45의 D44 E-time(seed 평균)이 rate에 대해 단조 증가(Spearman ≥ 0.9)하고 **E-time(0.10) ≤ 40%** | 둘 중 하나라도 거짓 | λ0 VESSL b_r0→b_r3 E-time 70.07→97.38(homog5). 0.10 req/s에서 한 요청의 decode(≈1.4 s) 동안 다음 요청이 올 확률 ≈ 1−e^(−0.14) ≈ 13% → 대부분의 decode-busy 시간이 (0,108) — 대략 산술 |
| **F-B2** | shape B 전 셀에서 요청별 ITL p95 > 50 ms인 요청 0개(ITL은 구속 술어가 아니다) | 1개라도 있으면 | λ0 VESSL B 셀 요청별 ITL p95 최대 28.5 ms(설계 재집계, §12-4) |
| **F-A1** (E-β 노출) | shape A의 E-time(1.35)·E-time(1.55)가 [3%, 15%] 안이고 E-time(1.35) ≤ E-time(1.55) ≤ E-time(1.80) | 거짓이면 | λ0 VESSL a_r0 3.22%(1.1)·a_r1 6.14%(1.8) — DS-6 노출 절벽 예보 |
| **F-A2** (약한 예보) | shape A의 2.70 또는 3.00에서 요청별 ITL p95 > 50 ms 요청이 있는 seed가 ≥ 1개 | 두 rate 모두 0개 | §10-2 ITL 여유 0.1–1.3 ms |

예보가 깨져도 용량 라벨은 바뀌지 않는다. F-B1이 깨지면 "shape B 저부하에서 D44 희석이 커진다"(N4 판정서 §1-2 (d))는 **예보로도 쓸 수 없게** 된다.

---

## 10. SLO 값 고정 · ITL 절벽 공시 (조항 5)

### 10-1. 실행 전 고정되는 것

- 결정점 k ∈ {3, 5, 10}, X = 90%, ITL 50 ms, 연속 곡선 k ∈ [1, 25], ITL 사다리 {40, 50, 60, 65}, 99%(기술).
- T_k의 **초 값**은 실행 전에 알 수 없다 — 대신 **공식과 측정 절차**가 실행 전에 고정되고, 초 값은 **같은 Job 첫 블록**에서 정해져 arm 셀 전에 파일로 동결된다(P-2 선택지 (i)). 재량 0.
- 주 k는 없다(P-3).

### 10-2. ITL 절벽 공시 (λ0 VESSL 아카이브 설계 재집계 — 인용 금지, 판정 아님)

| shape | 요청별 ITL p95 (λ0 VESSL, homog5) | 50 ms까지 여유 | HE0 절벽(55–62 ms)과의 거리 | 구속 술어 예상 |
|---|---|---|---|---|
| B | 중앙 21.8–23.8 ms, 최대 28.5 ms(전 rung) | ≥ 21.5 ms(약 1.75배) | 26 ms 이상 | **TTFT**(조항 5 "B의 D44 ITL p95 약 24 ms") |
| A | 1.8: 최대 41.7 ms · **3.0: 중앙 48.7, 최대 49.4 ms** · 포화(4.9·8.0): 최대 49.9 ms | **0.1–1.3 ms(3.0 이상)** | 5–12 ms | 2.4 이상 rate에서 **ITL이 구속 술어가 될 수 있다** |

- ★R4: "ITL로 정해지면"은 `ITL_BOUND_k` 플래그로 **기계적으로** 판정한다(§5-3).
- ★shape A의 50 ms는 **절벽 안**이다. cap 48로 배치가 묶이면 ITL이 49–50 ms 근처에 포화하는데, 50 ms 문턱이 바로 그 위에 있다.
  power cap 노출(λ0 VESSL a_r4 27.2%)이나 노드 차이로 수 % 느려지면 ITL 단독 실패가 대량으로 생길 수 있다 → 예보 F-A2. **E1b의 용량이 ITL로 정해지면 그 사실(구속 술어 분해 §6)을 라벨 옆에 필수 병기**하고, "TTFT 용량"으로 쓰지 않는다.
- ITL 사다리 {60, 65}는 shape A의 이 절벽 의존성을 기술로 보여 준다(재스코어 아님 — static arm).

---

## 11. 인용 규칙 (결과 문서·정본이 문자 그대로 승계)

**P-1 틀 문장(R8, 리터럴 — `AUDIT_slo_method_2026-10-04.md` §3-2 P-1)**:
> "TTFT 임계값 T는 처치가 아니라 판독 눈금이다. 사용자 기준이 공개 자료에 없으므로 이 연구는 단일 T의 만족 여부를 주장하지 않는다. 같은 모델·기판·길이 안에서 arm 간 attainment 차이의 부호가 등록 T 점 전부에서 일치하는지를 판정하고, 연속 곡선을 공개한다."

(N3는 arm이 하나라 "arm 간 부호 일치" 판정을 하지 않는다. N3의 역할은 그 판정이 쓸 B1의 T 눈금·곡선·용량을 등록 T 점 전부에서 공개하는 것이다 — N3C-11.)

**허용형(유일)**:
> target-D44 sticky-OFF B1의 λ\*\_SLO(k=K) — shape S, 라벨 L(구간 [r_lo, r_hi) req/s, 절대), seed 짝 bootstrap 95% [a, b], T_K = K × floor(= t 초, floor는 같은 Job 첫 블록 측정), ITL SLO 50 ms 조건, attainment ≥ 90%. 구속 설정: cap 48 = mamba 48, `max_prefill_tokens` 16384, `max_total_num_tokens` M. 용량 셀 실현 분할 E-cnt/E-time/E-iter/E-qcond(쌍 (64,44), A2 규약), W-idx, X-tok, dropped_events. 정상성 S1–S3 라벨, 관측자 효과 라벨, CG2, bench 창 power cap 노출 %. Job ID·GPU UUID. **best static 판정 아님**.

**금지 문안**:
- "D44 용량", "D44에서의 SLO 용량", "D44 SM으로 서빙할 때의 용량"(조항 2, DS-1)
- "사용자 SLO를 만족"(임계값 명시 없이), "업계 기준", "문헌 최빈값 5×"(P-6)
- (R8, P-6 추가 4개) "cliff 때문에 MLPerf 불채택" · 출처 없는 "10 s 휴리스틱" · 원문 미확인 "DistServe 90%" · "길이 정규화로 길이 간 비교"
- "TTFT 용량" — `ITL_BOUND_k` 셀(N3C-5)
- "N3가 best static을 정했다/B1이 최적이다"(조항 10)
- "게이트 #6을 닫았다" — 결과 감사 뒤에도 그 선언은 **5-a 전제를 충족했는가**의 판정과 함께만(§14)
- "k×λ\*(B) 부하"류의 정규화 표현, "λ\*\_SLO = c × λ\*"를 다른 arm·기판에 이식(DS-4, LV-3(c))
- shape A·B의 용량 크기 비교(길이 간 크기 비교 금지, P-6), 기판 간(KISTI) 수치와 풀링
- bridge를 근거로 "새 config와 homog5는 같은 arm이다"(§4-2)
- `UNRESOLVED_N_BELOW_4_k`를 "용량 없음"·"구별 불가"로 쓰는 것(게이트 #21)

**등록 caveat N3C-1…14 (rules verdict rev1 §5, 문자 그대로 승계)**:
- **N3C-1.** "이 캠페인은 게이트 #6을 닫지 않는다. 게이트 #6 상태 변경은 결과 감사와 5-a 전제 판정 뒤에만 논의한다."
- **N3C-2.** "floor의 (108,0) 실증은 admission 시점 인덱스의 telemetry와, decode 유휴 시 발사 인덱스가 같다는 코드 독해의 결합이다. 발사 분할을 직접 계측한 것이 아니다."
- **N3C-3(재측정 조항).** "N3 B1 수치(λ\*\_SLO 라벨·T_k·실현 분할)는 `pdmux_n3_A_agn.yml`(sha 01675b2d…) 한정이다. 다른 config의 캠페인은 B1을 다시 잰다. bridge 2셀은 λ\*\_VESSL(homog5)을 A-agn으로 이전할 자격을 주지 않는다. 같은 config·arm에서도 노드 간 achieved 차가 3.3–4.8%이기 때문이다(E2 vs λ0 a_r4)."
- **N3C-4.** "N3 셀은 5-a·5-b arm 비교의 짝으로 쓰지 않는다. Job·노드·seed가 다르다(confound 11)."
- **N3C-5.** "shape A 라벨은 결합 술어(TTFT ∧ ITL 50 ms) 용량이다. `ITL_BOUND_k`이면 'TTFT 용량'으로 쓰지 않는다. shape A의 ITL 50 ms는 cap 48 포화 ITL(49–51 ms) 바로 위의 칼날이라, 라벨은 측정 노드에 조건부다(같은 config·arm, offered 3.0에서 ITL≤50 통과율이 노드에 따라 1.000 vs 0.735)."
- **N3C-6.** "OBSERVER_OK는 B 0.30·A 1.80 한 점의 결과다. shape A의 ITL 구속 구간에서 PHASE_EVENTS가 라벨에 영향이 없음을 보이지 않는다. 등록 허용폭 3% 안의 ITL 섭동이 offered 3.0에서 ITL 통과율을 1.000→0.490으로 바꾼다."
- **N3C-7.** "용량 라벨은 등록 span(B 100 s, A 90 s)에 조건부다. 용량 셀이나 r_fail 셀이 UNSTABLE이면 'span 조건부'를 병기한다."
- **N3C-8.** "seed는 실현 평균 도착률을 nominal ±5%에 맞춘 조건부 표본이다. 무조건부 Poisson 대비 대기가 낙관 쪽으로 기울 수 있다(미측정)."
- **N3C-9.** "W-idx와 span 기반 기술값은 span 시작을 전환 이전 인덱스로 기록한다. 부하 셀에서는 기술값이며 '발사 분할 점유'로 쓰지 않는다."
- **N3C-10.** "CG2 PASS는 동시성 1 비분할 경로에서 PHASE_EVENTS가 greedy 출력을 바꾸지 않는다는 것만 뜻한다."
- **N3C-11.** "N3에는 arm 비교가 없다. k별 라벨을 묶은 '모든 k에서 견고'는 판정이 아니다."
- **N3C-12.** "r_fail 위 셀이 빠져 bootstrap seed 집합이 라벨의 seed 집합보다 작으면, 두 n을 함께 적는다."
- **N3C-13.** "λ0 VESSL 셀로 정한 격자 위치와 사전 floor(B 0.90 s, A 45 ms)는 설계 사전값이다. 결과로 인용하지 않는다."
- **N3C-14.** "B 격자 간격은 조항 5의 0.1×를 충족하지 않는다(최대 0.116×λ\*\_ref, 브래킷 폭 최대 0.08 req/s)." (R5 기본안 채택 시)

**등록 caveat N3C-15…19 (rules verdict rev2 §4, 문자 그대로 승계)**:
- **N3C-15.** "round 순서는 예산 절단이 최고 rate부터 빠지게 설계됐다. shape A에서는 상위 rate가 r_fail 후보대와 겹쳐, 절단이 그 k의 라벨을 UNRESOLVED로 만들 수 있다. 이는 측정 부족이지 발견이 아니다."
- **N3C-16.** "각 Job의 첫 grid 셀은 최고 rate다. Job 초반 노드 상태(클럭·power cap)의 비선형 효과는 r_fail이 최고 rate일 때만 라벨에 닿고, 크기는 측정하지 않았다."
- **N3C-17.** "`ITL_BOUND_k` = false는 ITL을 50→65 ms로 완화해도 라벨이 바뀌지 않는다는 뜻일 뿐이다. ITL이 무관하다는 뜻이 아니다."
- **N3C-18.** "boot마다 도는 abort 분기(STICKY·GREEN·P3·floor boot·CG2)의 배선은 텍스트 검사(분기별 중화 변이 11/11)로만 검증됐다. 실행형 검증은 preflight 4분기뿐이다."
- **N3C-19.** "R3 arm 동일성의 사전 근거(λ0 VESSL 11셀 비표적 0건)는 homog5·스냅숏 다리에서만 나왔다. A-agn 임계에서 decode_iteration 다리는 GPU에서 측정된 적이 없다."

**등록 caveat N3C-20 (rules verdict rev3 §4, 문자 그대로 승계 — rev3 수록 누락분을 rev4에서 수록)**:
- **N3C-20.** "`N3_RESULT.json`의 `realized_partition[*].arm_identity.ok`는 decode 다리를 요구하지 않는 기술 기록이다. 셀의 arm 동일성 판정은 `invalid_cells`(분석기 count 재계산)만 근거로 인용한다."

**등록 caveat N3C-21 (rev4, E2 A14 규칙층 판정서 E2VC-14와 동형)**:
- **N3C-21 (O4 해상도).** "O4 settle 정지 조건(`memory.used ≤ base+64 MiB`)은 MiB 해상도라 −3 토큰급(≈40–80 KB) 잔여를 해상하지 못한다. 'O4가 H1 잔여를 막았다/배제했다'고 쓰지 않는다. 실효는 `SETTLE.txt`의 NVML 바이트 기록이며 그 판별력도 미검증이다. 토큰 이탈이 재발하지 않으면 'O1–O4가 토큰 이탈을 해결했다'가 아니라 '이 Job에서는 이탈이 관측되지 않았다'고만 쓴다."

**승계 caveat**: DS-1…DS-7(N4 판정서) 전부, λ0 VESSL LV-2·LV-3·LV-6·LV-12·LV-13 형식, LVR-1…4, λ0R-3(cap 48 구속)·λ0R-8 계열(형식만, KISTI 수치 승계 금지 — LV-5). 특히 **DS-4**: 부하점은 절대 offered rate이고 "arm별 용량 대비 비율"로 쓰지 않는다. **DS-6**: shape A 0.36×–0.59× 부근의 ITL 민감도 전환은 예보다.

---

## 12. 감사자가 봐야 할 자유 표면 (자기 공시)

1. **격자 rate·N 규칙·span**(B 100 s/N_min 20, A 90 s): 예산(≲3–4 GPU-h)과 셀당 요청 수(B 0.10에서 seed당 20, 4 seed 합 80)의 맞교환. 작은 N은 attainment 해상도를 낮춘다(80요청이면 90% 문턱에 실패 8건).
2. **shape A 격자 교체**(E1 원형의 3.4–4.4 제거): λ\* 위 점이 없어서 k=10 용량이 3.0 이상이면 `CAPACITY_ABOVE_GRID_10`이 된다(정직한 라벨). 반대로 넣으면 Job당 +0.35 GPU-h.
3. **seed 재채점**: 평균 도착 간격만 맞춘 seed는 버스트가 적은 표본을 고를 수 있다 → 용량을 약간 낙관 쪽으로 움직일 수 있다(미측정). λ0 seed는 작은 N에서 실현 rate가 최대 25% 어긋나 쓸 수 없었다(§3-1).
4. **설계 사전 추정(인용 금지)**: 격자 위치는 λ0 VESSL 셀(n = 1, homog5)을 **설계 목적으로만** 재집계해 정했다.
   - floor 사전 추정 = λ0 VESSL warm-up(동시성 1, 8요청) TTFT 중앙 B 0.90–0.91 s, A 44–46 ms. floor 측정이 아니다((108,0) 미실증, warm-up).
   - 그 값으로 만든 T_k로 λ0 셀을 재채점하면 k=10 B 절벽이 0.62–0.85 사이, A의 세 k 용량이 1.8–3.0 사이에 온다.
   - **이것은 게이트 8이 금지하는 "재스코어로 λ\*\_SLO를 얻는 것"이 아니라 격자 배치 사전값이다.** 이 수치는 어떤 결과 문서에도 인용하지 않는다. 격자는 절대값으로 실행 전에 고정되므로 이 사전값은 결과의 정의를 바꾸지 않는다. 사전값이 틀리면 라벨이 BELOW/ABOVE_GRID로 나온다.
5. **첫 실패 규칙 vs "최대 통과 rate"**: 비단조일 때 둘이 갈린다. 첫 실패 규칙을 택했고 `non_monotone` 플래그로 공시한다.
6. **bridge를 PHASE_EVENTS OFF로**: D5 "전 셀 ON"의 예외다. λ0 VESSL과의 비교를 config 하나의 차이로 좁히기 위해서다. 그 대가로 bridge 셀은 N3 격자 셀과 계측이 다르다(어차피 판정 입력 아님).
7. **관측 rate 1점(B 0.30, A 1.80)**: 절벽 밖이라 TTFT 잡음이 작은 점을 골랐다. 절벽 안 관측자 효과는 외삽이다.
8. **S2 문턱 0.10, S3 ρ 0.50, S3 최소 표본 10**: 새 상수다. 결과 보기 전 고정. 더 엄격하면 크기 인용 가능 셀이 줄어든다(라벨은 불변).
9. **retract 셀 전체 제외**(요청 단위 제외 불가): 보수적이다. 예상 발생 0.
10. **floor = 16개 중앙값, 분산 문턱 없음**: 분산은 기록만 한다(`dispersion_max_over_min`). 이상값 하나는 중앙값이 흡수한다.
11. **CG2를 warm-up 텍스트로**: warm-up 명령에 출력 파일 인자만 더했다(클라이언트 동작). 동시성 1 greedy의 boot 간 결정성은 가정이다.
12. **엔진 트리 = HEAD 전체 manifest 일치**: 다른 트랙이 manifest 25파일 중 하나라도 바꾸면 이 등록은 돌지 않는다(의도). 그때는 rev2 + 새 digest가 필요하다.
13. **예산 상수**: per-boot 부대비용은 5그룹 config(λ0)에서 쟀다. 6그룹은 캡처 +1–2 s 외삽(CONFIG_CANDIDATES §2). preflight 600 s는 미측정. 최악 코너 배수(1.25, 0.8)는 새 상수.
    - (rev4) 6그룹 Workspace 프로브(§8-7, n = 1, 판정 아님): 캡처 +1.45 s·+0.20 GB, warm boot 벽시계 59.1 s(5그룹 60.1 s). 외삽의 부호·크기와 맞고, **상수를 올릴 근거는 없다.** 상수는 그대로다.
    - 단 +1.45 s가 매 boot 부대비용에 더해진다면 shape B 최악 코너의 마지막 격자 unit(`g_r077_4589`) 여유 7.86 s가 66.7 s 깎여 입장 거부된다(47/48 → 46/48). 측정 왜곡은 없고(입장 규칙이 시작 전에 막는다) 0.77의 n이 3으로 줄 수 있다(§8-7).
14. **입장 규칙 = 최악 코너 비용**: 보통 코너보다 보수적이라 마지막 unit(floor_tail)이 B 최악 코너에서 빠질 수 있다.
15. **X-req 재구성 앵커**: t0 = 창 안 두 번째 prefill span. 요청 0이 즉시 입장한다는 가정. 앵커 품질을 같이 낸다.
16. **CI 표본 2000, seed 1**: n = 4라 재표집 공간이 35개뿐이다. 표본 수를 늘려도 구간은 거의 바뀌지 않는다.
17. **포트 창 20000–20799**(§8-6 O1): ephemeral 범위 밖이라는 것은 VESSL 노드의 `ip_local_port_range`에 달렸다. 실행 시 기록하고 겹치면 abort한다. 다른 서비스가 그 창에서 LISTEN 중이면 O2가 피해 간다. 50번 시도에도 빈 포트가 없으면 `ABORT_NO_FREE_PORT`.
18. **O3 `pkill -9 -f "^sglang::"`**: 컨테이너 안의 모든 `sglang::` 프로세스를 죽인다. N3는 컨테이너당 서버 하나라 안전하다는 전제다.
19. **O4 settle 문턱(+64 MiB, 최대 180 s)과 "gate 아님"**: 180 s 안에 baseline으로 돌아오지 않아도 boot는 진행한다(기록만). 잔류가 실제로 있으면 `ABORT_P3`(`max_total_num_tokens` 불일치)가 잡는다. settle 대기는 예산 모형에 넣지 않았다(최악 +180 s × 48 boot ≈ 2.4 h가 이론상 상한, 입장 규칙이 남은 시간으로 막는다).
20. **R3 허용 0건**: 비표적 분할 decode iteration·busy 스냅숏이 **하나라도** 있으면 셀을 버린다. 관대하게 하면(예: 비율 문턱) arm 동일성이 정의상 흐려진다. 대가는 드문 오발화가 그 rate의 `UNRESOLVED`로 번지는 것이다.
21. **R1 순서 반전**: drift 균형은 같지만, round 1의 첫 셀이 최고 rate(포화에 가까운 셀)라 Job 초반 노드 상태가 고부하 셀에 실린다. rev1과 rate·시점의 결합 방식이 다르다.
22. **R7 처리 범위**: PHASE_EVENTS가 boot 중간에 꺼진 셀도 채점은 유지한다(클라이언트 측정은 무영향). 기술값과 관측자 대칭에서만 빠진다.

---

## 13. 로컬 검증 (GPU 0, 2026-10-04)

- `bash -n n3_capacity.sh` 통과.
- `n3_plan.py --selftest` 통과: seed = 채점 상위 4, Ēbar 가드, unit 구조·ABBA·PHASE_EVENTS 배치, 간격 ≤ 0.12×, 보통 코너 전 unit·최악 코너 floor_tail 외 전 unit 입장, config 검사 + 음성 대조 2(D54·5그룹), 등록 digest + 미러 9개, floor 공식 + 무효화 대조 2((64,44) span, running_bs > 0).
- `n3_realized_mix.py --selftest --require-archive`: 합성 OK + **λ0 VESSL 아카이브 7/7 E2 도구와 비트 동일**, 외부 기대값 재현, (92,16)·인덱스 재배치 대조 통과.
- `n3_analyze.py --selftest`: 라벨·CI·k 의존 절벽·BELOW/ABOVE·n < 4 규칙·ITL 다리·관측자 6% → FAIL·S2 → UNSTABLE·floor 무효 → 미채점·비단조 플래그.
- `tests/test_n3_capacity.py` 52/52(구조·변이 대조·digest·셀프테스트·dry-run·집행기·재개·포트/teardown O1–O4; rev2: R1 순서·절단 모사, R2 placeholder 거부·재개 대조, R3 주입 변이, R4 플래그, R7 표식, ABORT 분기별 변이 11개, 최소 트리 행동 시험 5개; rev3: D1 단위 시험 2개 + mix/analyze 셀프테스트 D1 변이; rev4: `TestP3Tolerance` 6개 — 상수 + E2 helper 동일성, E2 1회차 키 PASS·변이 FAIL, CLI exit 0/8, 러너 실제 P3 조각 실행, case 분기 중화 변이 검출, rev3 엄격 분기 대조).
- (rev4, 2026-10-05) `n3_plan.py --selftest`에 P3 (ii) 판별 추가: E2 1회차 키(2,722,025×9 + 2,722,022) → `PASS rel=1.102e-06`, 6그룹 48 boot 동일 → PASS; mamba 47/48·전부 47·토큰 0.8% 차·786,431·배너 누락·토큰 누락·키 없음·5그룹 대 6그룹 풀(rel 2.70e-4)·미등록 MAX_RUNNING/CTX → 전부 FAIL; 경계(786,432, 상대차 정확히 1e-4) → PASS.
  러너 P3 조각 변이(`TestP3Tolerance.test_runner_snippet`): mamba 47/48·전부 47·0.8% 차·786,431·6그룹 Job 안 5그룹 풀·배너 누락·helper 부재·거짓 PASS(exit ≠ 0)·PASS 없는 exit 0 → exit 8 + teardown; stderr 경고 + 진짜 PASS → 통과. 셀별 `banner_<unit>.txt` 기록 유지 확인.
- 로컬 dry-run(B 48 unit, A 36 unit 명령 생성), `__pycache__` 잔여 0.
- 로컬에서 **돌지 않는 것**: `test_phase_events`·`test_green_readout`(sgl_kernel 없음 — Job preflight에서 돈다), 실제 sync·manifest 대조, GPU 전부.
- 이 문서의 digest가 실행 기록에 들어가므로, 공개 이미지 컨테이너 dry-run 결과는 이 문서가 아니라 engine-porter 보고에 적는다(λ0 VESSL §9와 같은 이유).

## 14. 이 회차가 하지 못하는 것 · 제출 게이트

**하지 못하는 것**
- best static 판정, D 간·정책 간 비교, HE0 재판정, 게이트 #6 "닫음" 선언. 게이트 #6 상태 변경은 **결과 감사 뒤**에만 논의하며, 그때도 "5-a가 이 용량을 부하 기준으로 쓸 수 있는가"(5-a 전제)의 판정과 함께만 쓴다.
- λ0R-8(iii) 정규화 교락 해소(E2C-5 — E2도 N3도 해소하지 않는다).
- 노드 간 산포, 8192 이외 길이, 다른 모델.

**제출 게이트(순서대로)**
1. 이 문서가 claims-auditor 규칙층 `GO`(또는 `GO-with-caveats`)를 받는다. 감사가 요구한 수정은 rev2로 반영하고 델타 감사를 받는다.
2. 하네스층 감사(러너·plan·analyze·테스트, §8-6 포트·teardown 포함).
3. 메인 세션이 커밋한다(사전등록·러너·config 헤더·테스트). 커밋 후 `REGISTRATION_SHA256.txt`가 기록할 digest를 대조한다.
4. CPU 회귀 통과, `check_citation_stops.py` 0위반.
5. 사용자가 `launch.sh --campaign n3_capacity --env PDMUX_N3_SHAPE=B …` dry-run 출력(명령·단가·job name·예산 문언 §8-3)을 보고 승인한다. shape A는 **별도 승인**.
6. E2 Job이 끝난 뒤 제출한다(같은 계정 GPU 동시 사용·결과 혼동 방지 — 운영 규약상 권고, 하드 의존은 없다).

---

## 15. N3 필수 조항 대응표 (`VERDICT_n4_dsem_rules_2026-10-04.md` §"N3 사전등록 필수 조항")

| 조항 | 반영 위치 | 강제 수단 |
|---|---|---|
| 1 arm 정의(fixed/DSM44/sticky unset + read-out null, legacy, cudagraph ON, 5-a/5-b와 같은 config, 재측정 조항) | §2-1·§2-2·§4-2 | `ABORT_STICKY`·`ABORT_GREEN`·`ABORT_CONFIG`·구조 테스트, bridge 블록 |
| 2 명명 | §0·§11 | 금지 문안 |
| 3 4추정량 + 벽시계 점유 | §7-1·§7-2·§7-4 | `mix_<unit>.json`, 필수 병기 |
| 4 요청별 노출, PHASE_EVENTS 대칭 + OFF 대조 | §7-3·§4-4 | per-boot 노브, 관측자 라벨 |
| 5 SLO 실행 전 고정, ITL 절벽 공시, 절벽 둘레 0.1×, n ≥ 4, paired bootstrap | §10·§3-1·§5-3 | 간격 selftest, T_K 순서 |
| 6 정상성 S1–S3, STATIONARY만 크기 | §5-4 | `magnitude_citable` |
| 7 절대 req/s + 5-a 절벽 가드 상수 | §3 | 리터럴 테스트 |
| 8 구속 설정 병기 | §2-3 | `ABORT_P3`, `binding_settings` |
| 9 shape별 1 Job, 셀 전후 clk·cpu.stat, power cap 노출 | §2-4 | `cpu_<unit>.txt`, `substrate.clk` |
| 10 best static 판정 아님 | §0·§1·§14 | 금지 문안 |
| 11 retract 제외, 제출 전 커밋, 저부하 E-time 예보 | §5-2·§14·§9 | 셀 유효성, 제출 게이트, F-B1 |

## 16. 미확인 사항 (실행 전에 닫히지 않는 것)

- VESSL에서 **6그룹 config boot**(green read-out 6/6, 캡처 시간·메모리, `max_total_num_tokens`)는 한 번도 측정되지 않았다(KISTI 907959만). 첫 boot의 `ABORT_GREEN`/`ABORT_P3`가 이를 fail-closed로 확인한다.
- decode 유휴 admission이 실제로 (108,0)인지 — 코드 독해뿐. floor 블록이 측정으로 확인하거나 `ABORT_FLOOR`로 멈춘다.
- 동시성 1 greedy의 boot 간 결정성(CG2 전제).
- `#retracted_reqs:` 로그가 `event_loop_pdmux` 경로에서도 같은 함수로 찍히는지(코드 독해: `scheduler.py` decode 메모리 검사 경로) — 미실측.
- bench_serving random dataset 생성이 numpy 전역 RNG를 소비하지 않는다는 가정(λ0 판정서의 "span RNG 재생 11/11"로 간접 지지) — X-req 재구성에만 영향.
- per-boot 부대비용·preflight 시간(6그룹, PHASE_EVENTS ON, 추가 셀프테스트).

## 17. rev1 → rev2 변경 색인 (claims-auditor 델타 감사용)

| 수리 | 문서 | 코드 | 테스트 |
|---|---|---|---|
| R1 round 순서 반전 | §4-5 | `n3_plan.units` | `TestUnits.test_r1_*` 2개, plan selftest |
| R2 T_K placeholder 거부 | §5-3 | `n3_plan.tk_problems`·`--tk-valid`, 러너 `tk_valid`·재개, `n3_analyze.analyze` | `TestR2R7Runner`, `TestResume`(dryrun 대조), analyze selftest |
| R3 arm 동일성 0건 | §5-2 | `n3_realized_mix.arm_identity`, `n3_analyze.cell_validity` | mix selftest(주입 3·유휴 대조·λ0 7셀), analyze selftest(주입 → UNRESOLVED, 무telemetry → 무효) |
| R4 `ITL_BOUND_k` | §5-3·§10-2 | `n3_analyze.analyze` | analyze selftest(ITL 60 ms → 플래그, 기본 → 무플래그) |
| R5 간격 편차 등록 | §3-1, N3C-14 | 없음 | — |
| R6 P3 정책 | §2-3 자리표시자 | 미변경(엄격) | — |
| R7 PHASE_EVENTS 중도 비활성 | §7-2 | 러너 검출·표식, `n3_analyze.phase_events_disabled`·`summarize_mix`·`observer` | `TestR2R7Runner.test_r7_*`, analyze selftest |
| R8 P-1 리터럴·P-6 4개 | §11 | 없음 | — |
| 하네스: wiring 항등식 근접 | — | — | `TestFailClosedWiring` 분기별 중화 변이 11개, `TestAbortBranchesBehaviour` 5개 |

### 17-2. rev2 → rev3 변경 색인 (기준 커밋 `81b55c3`)

| 수리 | 문서 | 코드 | 테스트 |
|---|---|---|---|
| D1 R3 근거 부재 → unverifiable | §5-2 R3 항목 | `n3_realized_mix.arm_identity`(`expect_decode_iterations`, `unverifiable`), `n3_analyze.cell_validity`(ON ∧ R7 표식 없음이면 decode 다리 요구) | `TestEstimators.test_d1_*` 2개; mix 셀프테스트(빈·스냅숏만 ON → 무효, 스냅숏만 OFF → 유효); analyze 셀프테스트(빈 파일 → UNRESOLVED, ON 스냅숏만 → 무효, R7 표식·OFF 대조 → 유효); 합성 fixture에 두 다리 근거 추가 |
| D2 R6 전례 문구 | §2-3 | 없음 | — |
| N3C-15…19 수록 | §11 | 없음 | — |
| 헤더·로컬 검증 수치 | 머리말, §13 | 없음 | — |

### 17-3. rev3 → rev4 변경 색인 (기준 커밋 `69fbbb0`)

| 변경 | 문서 | 코드 | 테스트 |
|---|---|---|---|
| R6 = P3 (ii) 기입(사용자 결정 2026-10-05, A14 §3-1과 같은 문언·상수) + 6그룹 단일 config 사실(2,721,290 vs 2,722,025 = rel 2.70e-4, boot 간 비교에만 적용, bridge도 같은 config) + 1e-4 사용자 결정 상수 공시 | §2-3 표 2행·R6 블록(rev3 D2 전례 문구 유지) | `n3_plan.p3_verdict`·`p3_verdict_file`·`P3_*` 상수·`--p3-verdict`(exit 0/8)·`p3_selftest`; 러너 P3 분기 엄격 → helper 판정(exit 0 ∧ "PASS " 접두, fail-closed, `P3_VERDICTS.txt`, 셀별 `banner_<unit>.txt` 유지) | `TestP3Tolerance` 6개, plan 셀프테스트 P3 판별 |
| 6그룹 Workspace 프로브 반영(예산 상수 검증 근거, 판정 아님; 상수 불변; 캡처 +1.45 s의 B 최악 코너 입장 영향 공시) | §8-7 신설, §12-13 追記 | 없음(예산 상수·입장 규칙 불변) | — |
| N3C-21(O4 해상도, E2VC-14 동형) | §11 | 없음 | — |
| 헤더·로컬 검증 수치 | 머리말, §13 | 없음 | — |
| N3C-20 수록(rev3 판정서 §4 문안 그대로, rev3 수록 누락분) | §11 | 없음 | — |
| 러너 머리 주석·`n3_plan` docstring 회차 표기 "(rev1)" → "(rev4)"(주석·docstring만, 실행 경로 무변) | — | `n3_capacity.sh`:3, `n3_plan.py`:5 | — |

### 17-4. rev4 → rev5 변경 색인 (기준 커밋 `a4b0c2e`)

| 변경 | 문서 | 코드 | 테스트 |
|---|---|---|---|
| 엔진 루트 유도 `.resolve()` → `os.path.abspath` (Job 레이아웃 = `results/` 심볼릭 링크; 1회차 Job preflight 실패 수리, 판정 경로·상수 불변) | 머리말 rev5 줄 | `n3_plan.py` `HERE`(+`import os`, 주석 4줄), `n3_analyze.py` `HERE`(+`import os`, 주석 4줄) | `TestJobSymlinkLayout` 2개(통과 + `.resolve()` 복원 변이 대조) |
| 1회차 Job 실패 진단 기록 | `SUBMISSIONS_2026-10-06.md` | 없음 | — |
