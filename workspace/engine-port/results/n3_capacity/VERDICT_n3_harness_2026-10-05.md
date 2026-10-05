# 판정서: N3 하네스층 감사 (사전등록 rev4, HEAD `3604269`, rev4 커밋 `23c4175`, claims-auditor, 2026-10-05)

> 읽기 전용, GPU 0. 감사자 최종 보고를 메인 세션이 옮겼다. 성능 판정·Claim 등급 변경 0건.
> 실행 확인: (a) `test_n3_capacity` 52/52 재실행(226.9 s) (b) 러너 실제 텍스트 조각 + stub 입력 bash 실행 (c) 실데이터 대조(프로브 green read-out·telemetry, λ0 VESSL 아카이브). 저장소 미수정.

## 0. 판정: **`GO-with-caveats`** — 死因 0, 필수 수리 1건(H-1, 코드 국소·규칙 변경 아님), 새 caveat N3H-1…6
- **등록 게이트 하나가 실행상 fail-open**(§1-2 H-1): CG2의 ON 측 입력 `warmup_floor_head.jsonl`이 없거나 비면 CG2가 조용히 건너뛰어진다(rc 0, `CG2_DONE=0`, 기록 없음) → 채점 진행. 등록 §4-3은 "아니면 `ABORT_CG2`". 死因이 아닌 이유: 비정상 경로에서만 열리고(floor_head warm-up 실패 ∧ 같은 boot floor bench 16/16 통과), 수리 4행·규칙 불변(E2 하네스 rev2 선례). **정상 경로였다면 死因** → 제출 전 필수.
- N3C-18은 **미해소**이나 서버 없이 실행형 검증이 가능함을 이 감사가 실제로 보였다(§1-1, 처방 §11).

## 1. N3C-18: boot별 abort 분기 실행형 검증
| 분기 | 기존 시험 | 대조 변이 | 이 감사 실행 결과 |
|---|---|---|---|
| STICKY | 텍스트만 | 중화 1 | 실제 엔진 문자열(`multiplexing_mixin.py:451`) 주입 → **exit 6 + TEARDOWN** |
| GREEN(`--check-green`) | helper 행동 시험, 배선 텍스트 | 중화 1 | 프로브 `green_A2_warm.json` → 통과; `green_H5.json` → `ABORT_GREEN` exit 6 |
| P3 | **실행형**(rev4) | 있음 | 재확인 OK |
| floor 판정(`--floor`) | helper 행동, 배선 텍스트 | 중화 1 | 유효 → T_K.json rc 0; 마지막 span (64,44) → exit 7, `valid:false` |
| T_K 게이트 | DRYRUN 노트만, 비-dry 미실행 | 중화 1 | T_K 없음·비-dry → **exit 7**; dry → 노트(대조) |
| floor boot 실패 행 | **시험 없음** | 없음 | 등가 변이(T_K 게이트가 exit 7로 대체), 미공시 → N3H-1 |
| CG2 | 텍스트만 | 중화 1 | 8/8 → PASS; 1개 차이 → exit 9; **ON 부재/빈 파일 → rc 0, CG2_DONE=0(fail-open)** |
STICKY 중화 시 GREEN의 `sticky_target_index`가 exit 6으로 부분 대체(등가, 미공시 → N3H-1).

### 1-2. H-1(필수): CG2 fail-open — 조건 `&& [ -s "$OUT/warmup_floor_head.jsonl" ]`이 ON 측 부재를 "검사 생략"으로 처리; 루프 끝 `CG2_DONE` 확인 없음; 분석기도 CG2 결과를 읽지 않음(`cg2_all_boots` 기술용) → `N3_RESULT.json`만 보면 CG2 미실행이 보이지 않음.

## 2. 등록 → 집행 대응(전수) — 전부 ✓
unit 순서(floor_head→bridge×2→observer ABBA→round1–4→floor_tail, `--units` B 48·A 36) · ABBA ON,OFF/OFF,ON/ON,OFF/OFF,ON(B 0.30·A 1.80) · round 1·3 내림/2·4 오름, seed i=round i · N=max(N_min, round(rate×span)) (B [20,20,22,30,38,45,53,61,69,77], A [122…270]) · seed B (3704,2084,4861,4589)/A (898,3801,903,4364) · bridge b_r3 1.15/N200·a_r4 8.0/N400 seed 4386·4162 PHASE_EVENTS OFF · floor N16 seed 11 `--max-concurrency 1` · PHASE_EVENTS 분기 · 하드캡 14,400/tail 600/`start_utc` · 입장 최악 코너(B 보통 12,126 s/3.37 GPU-h·최악 47/48, A 10,612/2.95·36/36) · timeout `-k 30`·watchdog · settle 36×5 s +64 MiB 비게이트 · 포트 20000–20799·50회·`ss -Htan` · teardown 순서.
**λ0 VESSL 명령줄 diff**: SRV는 `$CFG`만, WARM은 `--output-details --output-file`(CG2용), BENCH 경로만, floor BENCH 동시성 1·rate 없음, env `PDMUX_PHASE_EVENTS` — 전부 의도. **의도 밖**: (1) `export PYTHONDONTWRITEBYTECODE=1` 전역(러너 46행, 서버·bench 상속; λ0 미export) (2) health~warm-up·warm-up~bench 사이 helper 호출(무부하 구간 <1 s 추정, 미측정) → N3H-3. §12-6 "bridge 차이 = config(+OFF, Job·노드)"는 문자 그대로는 거짓 → caveat.

## 3. P-2 floor 블록
`floor_from`은 `prefill_span_start`의 `prefill_sms`·`decode_sms`·`running_bs`·`n_new`·`queue_len`(admission 시점 인덱스, N3C-2 일치). 프로브 A1·A2 telemetry span 11개 전부 (108,0,0,1,0) — 도달 가능. 공식 16 TTFT 중앙값, T_k=k×floor 초, 실패 시 `valid:false`+exit 7(실행 확인). 순서는 `_order_ok`+DRYRUN 노트+비-dry exit 7. `tk_problems` 3곳 ✓. 사소: `output_lens`·`request_rate` 미검사(판정 무관); 재개 Job floor_head boot 실패도 `ABORT_FLOOR`(보수). 예산 반증 실패: 프로브 warm-up 8건 55.38 s → floor 16건 ≈110.8 s = 등록 110 s; 155 s까지 36/36.

## 4. 분석기
정본 `import analyze as A`·`passes()` 사본 없음(sha 1be49511…). 결정 규칙 문언 일치(첫 실패·4라벨·UNRESOLVED 최저~r_fail·`non_monotone` 라벨 불변·`ITL_BOUND_k`). bootstrap 2000·seed 1·공동 재표집; 관측자 paired(2000, seed 1), |mean ln|<ln 1.03 ∧ CI∋0; ITL 사다리·X=0.99 기술용; S1 1.7/S2 0.10/S3 0.50/최소 10. R3 `cell_validity` count 재평가 — 실데이터 양성 대조: λ0 b_r3 → OFF 유효/ON 무효. R7 러너 표식 또는 srv 로그 재검출. retract 정규식 합>0 → 셀 무효. N3C-20 재확인 — 필드명 변경 **비권고**(재봉인 비용 > 이득, 문안이 인용 차단). **미등록 자유 표면 0**(CLI `--clk`·`--json-out` 기술/출력, env 없음). 경계: srv 로그 없으면 retract 0 통과(근거 부재=통과, D1과 비대칭) → N3H-5; CG2 결과 미기재 → H-1·N3H-4.

## 5. digest·재개·fail-closed
`REGISTRATION_SHA256.txt`: prereg·러너·plan·analyze(n3)·mix·config·정본 analyze.py·lambda0_plan·`VERDICT_n3_*.md` 전부·git HEAD·미커밋 수(8개 미만 `ABORT_REG`). Job 내 `--verify-digests`가 막는 것: config·정본 술어·lambda0_plan·src 미러 9개; `--engine-check` 25항목. **못 막는 것**: helper 자신·러너·n3_analyze·mix·prereg·테스트(기록만, E2 동일 설계) → 결과 감사가 `REGISTRATION_SHA256.txt`를 봉인 표와 대조(N3H-2). 재개: UNITS 바이트·digest·이전 T_K 유효 요구, 시험 4종 행동 확인; 원 Job과 재개 사이 `VERDICT_n3_*.md` 추가 시 `ABORT_RESUME`(fail-closed). 허용 목록 밖 exit 2, 테스트 노브 비-DRYRUN exit 2, `PDMUX_PHASE_EVENTS` 상속값 기록 후 unset, 러너 자체 export 노브 주입 시 abort.

## 6. 테스트 항등식 점검(52)
실제 경로 행동 ≈33(Registration 6, Units 5, Port O1·O2 2, Estimators 5, P3 5, Analyzer 1, RunnerEnv 5, R2 dry-run 1, AbortBehaviour 5, Budget 3, Resume 1) · 텍스트·리터럴 ≈12(FailClosedWiring 6, O3·O4 2, R2 strict·R7 2, rates 1, `one_per_boot_knob`) · 사본 1(`STRICT_R3`, 대조 용도 정당). 변이 대조 없는 검사: O3·O4·`one_per_boot_knob`·`cap_and_constants`·`r2_runner_uses_strict`·`r7`·`budget_layers`·ABBA·rates. 하드캡 3겹: 두 층 제거→초과(검출), watchdog만 제거→등가(공시); **미시험·미공시**: timeout만 제거(등가 #2 예상), 입장만 제거(텍스트). **실제 bench `timeout` 행·`budget_killed_` 표식은 실행형 시험 없음(DRYRUN fake 형제 경로)** → N3H-6. 미공시 등가: floor boot 실패 abort, STICKY.

## 7. 운영
OUT → job_entry 심볼릭 `/data/runs/<job>/results/...`, 반출은 `start.marker` 이후 파일만, 재개 `PREV` 경로 일치, T0=`start_utc`. `FLASHINFER_WORKSPACE_BASE` 미검증(판정 무관, 기록만). B·A Job 공유: `/data/hf`(읽기)·`substrate_history.tsv`(append, 드라이버 다르면 `SUBSTRATE_WARNING` 정보용)뿐, 나머지 분리; fetch 병합 충돌 경로 없음. Job 전용 preflight(`test_phase_events`·`test_green_readout`)는 로컬 미실행, 실패 시 `ABORT_PREFLIGHT` 비용만.

## 8. 봉인 sha 표(`git show HEAD:` = 작업 트리, 전부 일치; 경로 `workspace/engine-port/` 기준)
| 파일 | sha256 |
|---|---|
| `results/n3_capacity/n3_capacity.sh` | `7e97e8f7969852f9ab41277dc8e3ce134897d5909f65cf80d1720ab9040b164e` |
| `results/n3_capacity/n3_plan.py` | `bf2e52f17d0742de5a5021707ef8ec439987b7c7d234153465b013087c1aee1d` |
| `results/n3_capacity/n3_analyze.py` | `923a240c70efab60dcf951635d6298f4f1acab8cbfafb26d146d0c47a54efddc` |
| `results/n3_capacity/n3_realized_mix.py` | `3b6f8934c69ad09b23fd4490a882dafbdecef05e7980f51b2679fe062d2e65c3` |
| `results/n3_capacity/configs/pdmux_n3_A_agn.yml` | `01675b2d7fa8ff23bb1cb142ff66fa4dfd0c021722ae9b8e7acedc2f4d38fca1` |
| `results/n3_capacity/PREREG_N3_CAPACITY_2026-10-04.md` | `c36d90468a36b26f79c50a7dd8abdbcfe1435e72c734c95384f1d6838407c7a5` |
| `tests/test_n3_capacity.py` | `69284c4605554cd3e8bb8bf8e6a5424d8a5a21ccbce4fdfbf529dad230d50b5b` |
| `scripts/vessl/job_entry.sh` | `75cc853f8096bbf218df0303b98a038d89ebe5aec2f2862d37b0b283c8beddc6` |
| `scripts/vessl/launch.sh` | `87d449a65b41b0cb7e1abbf2c3996e23392844516054f259e3a9d122c2f306f0` |
| `scripts/vessl/fetch.sh` | `6df5b8b13999165dbc5175bc71b6abc8becb42fb1c1472c1df166d7a736c19df` |
| `benchmarks/pdmux_eval/analyze.py`(정본) | `1be495110ea67c95f275b6c8ab7e7af25d0778c2910bf6a830170dbc078aeaa2` |
| `src/multiplex/multiplexing_mixin.py` | `ba5835aa9f935652860f73c2e38fdb550705f3d462b7ed25015c57d28d2aff8b` |
| `probe_6group_2026-10-05/SHA256SUMS` | `0ac1a93517a1dd67454a15a6646ec2398f0ae52f9f21a8cd59808ea974a69af2` |
(H-1 수리 후 러너·테스트 sha는 갱신된다 — 수리 커밋의 값으로 재봉인.)

## 9. 반증 시험 표
| 자유 표면 | 민 범위 | 판정 변화 | 근거 |
|---|---|---|---|
| CG2 ON 입력 부재 | 없음/빈 파일 | **등록 ABORT_CG2 → 실제 계속** | stub rc 0, CG2_DONE=0 ×2 → H-1 |
| R3 근거 출처 | mix json vs 재계산 | 없음 | λ0 b_r3: OFF 유효/ON 무효 |
| mix `ok` | ON 스냅숏만 | 라벨 불변 | N3C-20 |
| retract 로그 부재 | 삭제 | 통과(정상 반출 비도달) | N3H-5 |
| bootstrap seed 교집합 | 0.77 결손 | 라벨 불변, CI 3 seed | N3C-12 |
| floor bench 예산 A | 110→155 s | 36/36 불변 | 프로브 55.38 s/8건 |
| `PYTHONDONTWRITEBYTECODE` | 켬/끔 | 서빙 경로 없음(독해) | N3H-3 |
| 재개 표식 누락 | — | 없음(srv 로그 재검출) | — |
| STICKY 제거 | 중화 | 부분 등가(GREEN exit 6) | 실행 |
| floor boot 실패 제거 | 중화 | 등가(T_K 게이트 exit 7) | 실행 |
원자료: 프로브 `green_{A2_warm,H5}.json`·`tel_A{1,2}_*.jsonl`·`warmup_client_A2_warm.log`, λ0 `bench/tel/srv_b_r3`, `n3_plan.simulate`. 새 상수·추정량 없음.

## 10. 필수 수리(규칙 변경 아님 → 규칙층 델타 감사 불요, 러너·테스트 diff 국소 확인)
**H-1 (CG2 fail-open)**: CG2 블록 조건에서 `&& [ -s "$OUT/warmup_floor_head.jsonl" ]` 제거, 블록 첫머리에
```
    if [ ! -s "$OUT/warmup_floor_head.jsonl" ]; then
      teardown "$PID" "$PORT"
      fail "ABORT_CG2 ON-side input warmup_floor_head.jsonl missing/empty -- CG2 not evaluable (measurement failure, not evidence of an observer effect)" 9
    fi
```
새 exit 코드·상수 없음(등록 exit 9). **T-H6(필수, 교훈 53)**: 러너 CG2 조각 실행 — 8/8 → rc 0 ∧ CG2_DONE=1; 1개 차이 → 9; ON 부재 → 9; ON 빈 파일 → 9; **되돌린 변이(현 코드)에서 ON 부재 → rc 0이라 시험이 실패해야 함**.

## 11. 권고 시험(N3C-18 축소, 선택·로컬 게이트)
T-H1 STICKY(실제 문자열→6/없음→통과/중화 대조) · T-H2 GREEN(`green_A2_warm`→통과, `green_H5`→6, `sticky_target_index` 4→6) · T-H3 floor(유효→T_K, (64,44)→7, running_bs>0→7, 재개 경로→T_K_RESUME) · T-H4 비-dry T_K 게이트→7 · T-H5 floor boot 실패→7 + 등가 공시 · T-H7 실제 bench `timeout` 조각을 `BENCH=(sleep N)`으로 실행 → `budget_killed_` 생성. preflight 목록(§8-1)에는 넣지 않음.

## 12. 새 caveat(결과 문서·정본 승계)
- **N3H-1.** "boot별 abort 분기 중 P3만 러너 조각 실행형 시험이 있다. STICKY·GREEN·floor·T_K 게이트·CG2는 텍스트 배선 검사(+ H-1 이후 CG2 조각 실행)로 검증됐다. floor boot 실패 abort는 T_K 게이트가, STICKY는 GREEN read-out이 부분적으로 대신하는 등가 변이다. 실제 서버 출력에서 이 분기들이 발화하는 것은 검증되지 않았다." (T-H1…T-H5 추가 시 "P3만"→"STICKY·GREEN·floor·T_K·CG2·P3 조각 실행형(stub 입력)"으로 좁혀 승계; 그 전까지 N3C-18 유지.)
- **N3H-2.** "`--verify-digests`가 Job 안에서 막는 것은 config·정본 술어·lambda0_plan·src 미러뿐이다. 러너·plan·분석기·mix·사전등록 자신은 기록만 된다. 결과 감사가 `REGISTRATION_SHA256.txt`를 하네스 판정서의 봉인 sha 표와 대조하기 전에는 '등록대로 집행됐다'고 쓰지 않는다."
- **N3H-3.** "N3 러너는 `PYTHONDONTWRITEBYTECODE=1`을 전역 export하므로 서버·bench 프로세스가 이를 상속한다(λ0 VESSL 러너는 그러지 않았다). 그 밖에도 boot 전 settle·포트 창·helper 호출 대기(서버 무부하 구간)가 다르다. bridge의 'λ0 VESSL과의 차이는 config(+OFF 경로, Job·노드)뿐'이라는 서술은 이 차이들을 포함해야 한다. 서빙 경로 영향은 코드 독해상 없으나 측정하지 않았다."
- **N3H-4.** "CG2 게이트 결과는 `CG2.txt`에만 있고 `N3_RESULT.json`에는 없다. 재개 Job에서 OFF unit이 모두 복사되면 그 Job 안에서는 CG2가 돌지 않는다. 결과 문서는 'CG2 PASS'를 Job별 `CG2.txt`로만 인용한다. 재개 Job의 TIMING.tsv에는 복사된 unit 행이 없어 그 셀의 bench 창 power cap 노출 %가 산출되지 않는다."
- **N3H-5.** "retract 검사는 서버 로그가 없으면 0으로 통과한다. 'retract 0'은 채점 셀마다 `srv_<unit>.log`가 존재함을 확인한 뒤에만 쓴다."
- **N3H-6.** "N3C-19의 decode_iteration 다리는 Workspace 프로브의 동시성 1 boot 2개(각 4,126 iteration, 비표적 0건)에서만 GPU로 관측됐고, 전부 (0,108)이었다. 분할 전환 경로(부하 셀)의 다리는 GPU에서 관측된 적이 없다. 하드캡의 실제 bench `timeout` 행과 `budget_killed_` 표식, 'timeout만 제거'·'입장만 제거' 변이는 실행형으로 시험되지 않았다."

## 13. 남은 선행조건
1. H-1 + T-H6 반영(러너·테스트 sha 갱신, CG2 블록 국소 diff 확인, 규칙층 재감사 불요).
2. 이 판정서를 **제출 전에 커밋**(untracked는 번들·REGISTRATION에 안 들어감).
3. CPU 회귀 + `check_citation_stops.py` 0위반.
4. 사용자 dry-run 승인(B·A 별도). E2 Job 뒤 제출.
5. (권고) T-H1…T-H5·T-H7; Workspace에서 `test_phase_events`·`test_green_readout` 사전 실행.

## 14. 자기 실현성 검사
H-1은 등록 exit 9와 기존 `teardown`/`fail`만 사용, 새 상수·노브·라벨 없음(교훈 250); 재개 Job도 floor_head를 항상 다시 돌므로 정상 경로를 막지 않음. T-H6는 scratchpad에서 같은 구성으로 실행·대조 성립 확인. N3C-20 필드명·CG2 기록 변경은 재봉인 비용 때문에 일부러 미처방. 직전 판정서 지적(N3C-18·19·20)을 상수로 승격하지 않고 실행·실데이터로 독립 재검증(게이트 #110).

---
## 追記 (메인 세션, 2026-10-05 — H-1 수리·T-H1…T-H7 반영 후)
- engine-porter가 H-1(CG2 fail-closed, §10 문안 그대로)과 T-H6 + 권고 T-H1…T-H5·T-H7을 반영했다. `git diff 3604269 --stat` = `n3_capacity.sh`(+5/−1, CG2 블록만)·`tests/test_n3_capacity.py`(+388) 2파일. 사전등록·plan·analyze·mix·config 바이트 불변. 테스트 **61/61**(52+9), selftest 3/3, citation 0위반.
- T-H6 대조: 정규식으로 되돌린 변이가 HEAD `3604269` 러너와 바이트 동일함을 확인했고, 그 변이에서 ON 부재 → rc 0·`CG2_DONE=0`으로 수리 검사가 실패한다(교훈 53 충족).
- 등가 변이 공시(테스트 docstring에도): E-1 floor boot 실패 abort 삭제 ≈ T_K 게이트(floor-only unit 목록에서만 검출) · E-2 STICKY 중화 ≈ GREEN `sticky_target_index`(stub read-out 한정, GPU 미검증) · E-3 timeout만 제거 ≈ watchdog(`BRC=143`, 표식 기록; watchdog 없으면 비등가). "입장만 제거"는 여전히 텍스트 검사.
- **재봉인 sha(수리 후)**: `n3_capacity.sh` `3cc780dca651419965b3917db392b4af37f96feb5d035bfef56622b393cb0c3f` · `tests/test_n3_capacity.py` `f28ba360237271dfff0830e03af9fa8bd0381010d6cc7ff60f5ed877a7f65a42`. §8 표의 두 행은 이 값으로 대체되며, 나머지 행은 불변.
- **N3H-1 축소 승계(판정서 §12 자체 조건 충족)**: "boot별 abort 분기 STICKY·GREEN·floor·T_K 게이트·CG2·P3는 모두 러너 조각 실행형 시험(stub 입력, 실제 프로브 로그·green json·telemetry 사용)으로 검증됐다. floor boot 실패 abort는 T_K 게이트가, STICKY는 GREEN read-out이 부분적으로 대신하는 등가 변이다(E-1·E-2). 실제 서버 출력에서 이 분기들이 발화하는 것은 검증되지 않았다." N3C-18은 이 범위로 좁힌다.
- 주의: T-H7 watchdog 케이스는 실제 watchdog 블록을 돌려 로컬 머신의 `sglang.bench_serving`/`sglang.launch_server`/`n3_fake_bench` 프로세스에 `pkill`을 보낸다 — 로컬 서빙이 없는 환경에서만 무해.
