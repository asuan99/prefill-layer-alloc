# 세션 핸드오프 — 2026-10-11 (2026-10-07 시작, 10-08·10-09·10-11 속행; VESSL A100 + 로컬 CPU)

## 이번 세션 요약

2026-10-06 2차 핸드오프에서 이어받아 세 갈래를 진행했다. (1) **정리 작업(C)**: doc_facts 회귀 수 필드 구조 수리(게이트 #245 (i)), 잘린 메모리 3파일 복구. (2) **N3(게이트 #6) shape B 4회차**: rev7(예산 상수 재추정·shape별 하드캡·W-5 length) 봉인 → Job 완주(48/48) → 결과 감사(死因 0, `OBSERVER_FAIL`, 3회차와 "재현" `REFUTED`) → 정본 rev91. (3) **N2(3-a) 사전등록**: rev1→rev16, 감사 17회(규칙층·델타·설계 비교·봉인), 사용자 설계 결정(프로파일 Job 먼저·E-α (A′) 고정 K·D 민감도 부속 estimand) → **규칙 봉인(커밋 `fcc0b9e`)** → **Job-PROFILE 하네스 구현(커밋 `1037655`, 미봉인)** → **rev17(§18-2-c 규칙 재개봉 1회차) 진행 중·미커밋**. 새 성능 판정 0건(N3 4회차 라벨 3개는 기계적 산출물), 게이트 #6·HE0·Claim D/E 불변.

## 결정·측정

**사용자 지시·결정(시간순)**
- 2026-10-07 "C를 먼저 진행하고, A,B를 병렬로 진행" → doc_facts 수리 (i) 승인으로 처리.
- 2026-10-07 "권장 사항대로 진행" → N3 rev7: 새 Job 48 unit(rev6 재개 포기), shape B 하드캡 17,100 s, W-5 length; N2: D-8 수리 K 증가, X = ŝ+145, Job-PROFILE/Job-EA 분리, U-2 프로브 실시(이후 철회됨).
- 2026-10-07 "제출 진행" → N3 4회차 Job 제출.
- 2026-10-08 "N2 rev만 증가… 무엇이 바뀌는지 모르겠다", "K 고정의 의의?" → 설계 비교 감사(`REVIEW_ealpha_coload_design_2026-10-08.md`): rev6–rev13의 반복 원인 = "프로브 결과(κ)로 최소 K를 정하는 규칙"이 측정값을 판정 입력으로 만든 것. 권고 (A′).
- 2026-10-08 실행 순서 = **프로파일 Job 먼저, 동거 측정(E-α) 나중**(감사 권고와 반대, 사용자 결정).
- 2026-10-08 사용자 가설(hybrid decode는 attention보다 compute 집약 → SM 추가 의미↑)을 **D 민감도 부속 estimand**(같은 round 안 D 쌍 paired, 4/4 라벨, 예측 F-N2-D1…D3, "attention보다" 금지)로 등록하기로 "진행".
- 2026-10-08 (A′) 채택 "진행": K를 봉인 전 1회 유도해 고정(K_16 33·K_44 15), C 창 = 처음 128 동거 iteration, 프로브→K 장치 판정 경로 제외(U-2 프로브는 선택적 비게이트 smoke).
- 2026-10-11 "다음 세션에서 이어서" → 이 핸드오프.

**N3 shape B 4회차** — Job `pdmux-n3-capacity-93c3cca-20261007064124`(rev7 봉인 `93c3cca`, 제출 `f8414f5`, 회수 `b28676c`, 결과 감사 `02a6431`, 정본 `9831c07`): exit 0, 3.409 h, 48/48 unit, 결측 0. 라벨(기계적 `CONFIRMED`): k=3 `CAPACITY_BRACKETED_3` [0.15, 0.22) bootstrap [0.15, 0.30](실패 셀 0.22가 1요청 차), k=5 [0.45, 0.53)·k=10 [0.69, 0.77)(용량 셀 `UNSTABLE`, 크기 인용 불가). **`OBSERVER_FAIL`**(TTFT +0.41% [+0.05, +0.84]) → 모든 인용에 "PHASE_EVENTS ON 조건에서". 3회차와 다른 GPU, 같은 seed 공통 36셀에서 4회차가 모든 k 36/36 셀 ≥ → **"재현" REFUTED**(Job 효과). 게이트 #6 미닫힘. 허용·금지 문안 = CONSENSUS §1 행 37, N3VR-12…20, 게이트 #262–263.

**N2(3-a)** — 판정서 경로 `workspace/engine-port/results/n2_static_grid/VERDICT_n2_3a_*`·`REVIEW_ealpha_coload_design_2026-10-08.md`.
- rev1–rev5: 측정 설계 결함(D-1…D-8: 108 합산 모순, 산포 0 셀 창 불성립, OFF 단일 π, 항등식 라벨, WITHIN 규칙 모순, R7 범위, E-α 출력 길이, D16 동거 창 검열) 수리. rev6–rev13: 프로브→K 장치 시행 구멍(감사 대부분 GO-with-caveats, 死因 D-9 1회).
- rev14: 프로파일 우선 재구성(A′ K=(33,15), D 민감도, 기록 공백 규칙 = iter_seq 연속).
- rev15·rev16: 낡은 문언·창 함수 일반화(ON/OFF E-α·PROFILE)·봉인 파일 분류·변이 장부 분할.
- **규칙 봉인** `fcc0b9e`: `REGISTRATION_SHA256.txt`, 7 규칙 파일, 봉인 digest `fbb13b15fc63e595e7588eac64fcde71e1dd98d08e96595d46f88ad6b0889512`(n2_static_grid 안에서 `sha256sum <7 files> | sha256sum`), 규칙 시험 141, 코어 변이 192(183 사살·9 등가).
- **Job-PROFILE 하네스** `1037655`(미봉인): n2_plan(ABORT_REG_SOURCE = 등록 파일 바이트 sha + 규칙 봉인 digest), n2_job, n2_capacity.sh(3층 하드캡), n2_window(봉인 `cell_windows` import), n2_realized_mix(λ0 양성 대조), n2_signature, n2_analyze(D 민감도 포함), n2_reach_tests, fake engine 확장. 시험 job 48·reach 36·harness 36, 변이 harness 32/32·JH 76/76 사살, 가짜 엔진 전규모 e2e 296/296 VALID(가짜 엔진은 D 무관 → D 민감도 값 무의미). Job-EA 부분은 스텁.

**기타**: doc_facts 회귀 수 필드를 절차 2 커밋 `27bbae7`의 tests/ 155로 동결(`6113723`). 메모리 3파일(교훈·SLO 트랙·사다리 결정) 2026-10-06 잘림 → 옛 머신 9-16 사본 + 세션 기록 op 18개 재생으로 복구, 원인(`open(p,'w')`와 read 같은 표현식) → 메모리 `memory-file-write-safety`. 복구 중 생긴 stray 커밋 `cffbee4`는 정리됨.

## 코드·문서 변경

**커밋됨**(0c6b0de..1037655, 28 커밋, push 안 함): 위 N3·N2·doc_facts·정본(rev90·rev91) 전부. 다른 세션의 power_cap 커밋(`2196642`, `7a7fa10`, `677111a`, `cea838d`)이 사이에 섞여 있음.

**미커밋 — rev17 진행 중(다음 세션이 이어받음)**:
- `n2_static_grid/PREREG_N2_3A_2026-10-07.md`, `n2_mutants.py`, `test_n2_probe_reduce.py` — 초안 에이전트가 §18-2-c 규칙 재개봉 1회차(rev17)를 편집하던 중. **규칙 봉인 파일이므로 완성·감사 전 커밋 금지.** 세션 종료로 에이전트와 그 코어 변이 실행은 중단된다.
- 다른 세션 작업(건드리지 말 것): `results/power_cap/analysis_2026-10-07/pc_common.py`, `scripts/vessl/job_entry.sh`(+25줄, 샘플러 관련), power_cap 미추적 CSV들, he0/n3 `clk.csv`(관례상 제외).

## 열린 항목 / 다음 세션 시작점

1. **rev17 완성**(§18-2-c 규칙 재개봉 1회차, 하네스 봉인 준비). `git diff -- workspace/engine-port/results/n2_static_grid`로 진행분 확인 후 남은 항목 마무리:
   - H16-1: `test_n2_probe_reduce.py`에 경계 시험 2개(`test_pe_time_zero_weight_is_fail_closed`, `test_alone_prefill_at_window_endpoints` — 하네스 보고 초안 그대로) + `n2_mutants.py` 코어 변이 2개(`_pe_time` 0→1.0, §7-1(c) `<=`→`<`) → N2C-75 n/a.
   - §18-2-a/§15 파일표에 n2_job.py·test_n2_job.py·n2_job_mutants.py·n2_reach_tests.py를 하네스 봉인 칸으로, 시험 수(규칙 143·harness 36·job 48·reach 36).
   - CTX_UNREALIZED 문언을 코드(트림 뒤 첫 iteration)에 맞춤.
   - 하네스 계약 모호점 등록(셀 순서 b 바깥·c 안쪽, 셀 마감 기준 = 셀 구간 시작, G-3(a) c_in, §11-2 greedy 6 프롬프트, bench 유효성, RETRACT 귀속, **G-10 Job 문자 적용(VD 중단 → Job 정지; 정책 성격, 필요하면 사용자 확인)**, 고안 라벨, OBSERVER 우선순위, 예측 운영 정의, §7-3 vs H15-4 → "Job-PROFILE 셀·boot 4추정량은 Job-EA 봉인 뒤", ABORT_REG_SOURCE 두 검사, "취소 OFF 경로 바이트 동일" 정의 또는 삭제, 엔진 manifest = N3 25항목, 로그 TZ).
   - §18-2-b에 config·analyze.py·profile.py digest(n2_plan 상수에서).
   - 규칙 시험 + **코어 변이만** 순차 실행(수 시간 — 세션 종료로 끊긴 전례 다수; 가능하면 세션 초반에 시작) → 새 규칙 봉인 digest 계산 → `n2_plan.RULE_SEAL_DIGEST`/`RULE_FILES` 갱신 → test_n2_job 재실행.
2. **좁은 규칙 델타 감사 + 하네스층 감사**(한 번에) → `REGISTRATION_SHA256.txt` 규칙 digest 갱신 + `## HARNESS SEAL` 절(형식: `rule_seal_digest <hex>`, `n2_plan.HARNESS_FILES` 순서로 `<hex>  <file>`, `harness_digest`) → 하네스 봉인 커밋.
3. **Job-PROFILE dry-run**: `launch.sh --campaign n2_static_grid --env PDMUX_N2_JOB=PROFILE --target workspace/engine-port/results/n2_static_grid/n2_capacity.sh` → 사용자 승인 → `--submit`. 셀 구간 하드캡 36,700 s(PROFILE (b), 기존 승인 범위).
4. 실엔진 첫 확인 사항(첫 boot fail-closed로 막혀 있음): `/generate` 비스트림 `output_ids` 반환, RETRACT 로그 형식, sticky ON 6그룹 cudagraph 로그·green read-out(§19-7/8).
5. Job-EA 전: 하드캡 16,100(b)/24,500(a) 사용자 재승인(E14-1), E-α 하네스(H15-3 ρ_off 조립 등) 구현, 블라인드 규칙(Job-PROFILE 동거 원자료 κ 분석 금지) 준수.
6. N3: shape A 착수 여부(사용자 결정), 4회차 결과 감사 §12 확정 조건(k=3 위치 n≥8, k=10 창 의존, 관측자 n≥8, σ_Job) — 처방 아님.

## 미완·주의

- rev17 미완·미커밋(위). 다음 세션에서 초안 에이전트를 재개할 수 없으면 새 에이전트에 이 핸드오프 "열린 항목 1"을 그대로 지시.
- 하네스는 가짜 엔진에서만 검증됨 — 실엔진 동작 미확인. 하네스 봉인 전 dry-run 불가(스크립트가 `ABORT_HARNESS_UNSEALED`).
- 장시간 감사·변이 실행이 세션 재시작·API 한도로 여러 번 끊김(감사 3회, 초안 2회). 감사는 변이 재실행 없이 대조 위주로 범위를 좁히는 것이 효과적이었음.
- E-α 관련 caveat 다수(N2C-64…75, EAK-1…5)와 블라인드 규칙은 Job-EA 단계에서 적용.
- Workspace `pdmux-probe`(A100)·`pdmux-build`(CPU) 둘 다 standby.

## GPU 실행 기록(VESSL)

| job name | 캠페인 | 커밋 | 이미지 | 상태 | fetch | cost.txt |
|---|---|---|---|---|---|---|
| `pdmux-n3-capacity-93c3cca-20261007064124` | n3_capacity B(rev7, 4회차) | `93c3cca` | `sglang-runtime@1e0d335b…` | succeeded exit 0 | DONE+sha256 708/708 OK (`b28676c`) | 3.409 h |

(다른 세션의 power_cap 프로브 Job `pdmux-sensor-bw-*`·`pdmux-partition-power-*`는 그 트랙 기록 참조.) Workspace: `pdmux-probe`·`pdmux-build` standby.
