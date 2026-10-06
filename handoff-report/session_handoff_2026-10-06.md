# 세션 핸드오프 — 2026-10-06 (2026-10-04 세션에서 이어짐, VESSL A100)

## 이번 세션 요약

2026-10-04 세션의 사다리(N0–N6) 결정과 TTFT SLO 방법 A′ 채택을 이어받아, 두 캠페인을 VESSL에서 실행했다.
**E2 sticky 대조 재실행(A14)**은 20/20 boot로 완주했고, 등록 라벨이 전부 나왔으며 결과 감사는 死因 0이다.
**N3 게이트 #6 SLO 용량 shape B**는 두 번 제출해 두 번 다 셀 측정 0으로 끝났다.
- 1회차: Job 레이아웃(results 심볼릭 링크)에서 엔진 루트가 클론 밖으로 풀려 preflight가 중단됐다. rev5에서 수리했고 델타 감사는 `GO-with-caveats`다.
- 2회차: preflight는 통과했다. floor 판정이 입장 시점 텔레메트리 라벨 1건을 분할 위반으로 읽어 중단됐다. 원인 진단은 끝났고, 수리는 규칙 변경(rev6)이라 **사용자 결정 대기** 상태에서 세션을 닫았다.

## 결정·측정

**사용자 지시(이번 세션)**
- "231 순서로 진행" — E2 재실행 등록(A14)·Workspace 6그룹 프로브·N3 P3 (ii) 순서.
- "E2 r2 재실행 진행 N3 dry-run 진행"
- 권한은 b안(permission rule)으로 처리했다. 사용자가 `launch.sh`·`vesslctl job` allow 규칙을 직접 추가했다.
- **"앞으로 과금에 대한 부분은 설명하지 말고 실험 job들에 대해서 다시 진행"** → 메모리 `no-billing-commentary`.
- "수리 진행 후 실행" — N3 rev5 수리 후 재제출 승인. "실패 원인 분석 진행", "다음 세션에서 진행하자".

**E2 r2 — Job `pdmux-e2-sticky-vessl-22c8c64-20261006020317`**
- 원자료: `workspace/engine-port/results/r2_eval/e2_sticky_vessl/e2v_pdmux-e2-sticky-vessl-22c8c64-20261006020317/`
- 집행: 20/20 boot, exit 0, 컨테이너 1.953 h, P3 (ii) 20/20 PASS. boot 15에서 −4 토큰(rel 1.47e-6) 이탈이 있었다. 포트 재시도 0.
- 라벨(`E2_LABEL.json`, 결과 감사에서 바이트 재현):
  - R1 = `OFF_BASELINE_ABOVE_BAND` — 대조 arm 진술이고 Q1은 미확립이다.
  - R2 a_r4·a_r2 = `LAMBDA_MOVES_DOWN`.
    - a_r4: d̄ −0.905 req/s, CI [−0.912, −0.898], −29.8…−30.3%.
    - a_r2: d̄ −0.689, CI [−0.714, −0.669].
    - 둘 다 n=4, `--max-running-requests 48`에 구속된 achieved이고, 정지 rate 사다리라서 **정책 비교가 아니다**.
  - R3·R4·R4′·R5 PASS. R4″ = `B_RESULT_AXIS_WITHIN`(n=2, "차이 없음" 아님). P13 byte-identical.
- 결과 감사 `VERDICT_result_e2_vessl_r2_2026-10-06.md`: 死因 0, 라벨은 기계적으로 `CONFIRMED`. 신규 caveat E2VR-1…8, 금지 문안 16건, 허용 문안 §9.
  - Q1(sticky의 D44 실현)과 R2의 기전 귀속은 `NOT-YET-SUPPORTED`이다.
  - 불변 선언: 게이트 #6, λ0R-8(iii), HE0, Claim D/E.
- 결과 문서: `RESULT_e2_vessl_r2_2026-10-06.md`(result-analyst). **감사 §9 금지 문안과의 대조는 아직 하지 않았다.**

**N3 shape B 1회차 — `pdmux-n3-capacity-a4b0c2e-20261006064114`**
- exit 4 `ABORT_SELFTEST`, 측정 0.
- 원인: `n3_plan.py`·`n3_analyze.py`가 `Path(__file__).resolve()`로 엔진 루트를 정한다. `job_entry.sh:100-103`이 `results/`를 `/data/runs/<job>/results` 심볼릭 링크로 바꾸므로, 루트가 클론 밖으로 풀려 `analyze.py`를 찾지 못했다(MISSING).
- 수리 rev5(`5c58dbc`): `os.path.abspath`로 바꿨다. 테스트 `TestJobSymlinkLayout` +2(복원 변이 대조 포함).
- 델타 감사 `VERDICT_n3_rev5_delta_2026-10-06.md`: `GO-with-caveats`, 死因 0, N3C-23…27. 필수 수리 R-1…R-3을 반영했다.
- 같은 함정의 선례가 이미 있었다(`e1_plan.py:45-47`, 2026-10-04). 그런데 승계하지 않았다(교훈 18 계열).

**N3 shape B 2회차 — `pdmux-n3-capacity-79999aa-20261006081638`**
- exit 7 `ABORT_FLOOR`, 측정 0. rev5 수리는 작동했다(preflight 전부 통과).
- floor bench는 16/16 완료했다. TTFT p50 906.76 ms, p99 909.99 ms.
- 실패 지점: `floor_from` 조건 (iii) "모든 prefill_span_start가 (108,0)". 28 span 중 1건(warmup 4번째, bench 창 이전)이 (0,108)·`running_bs=1`로 기록됐다.
- 진단(코드 순서 + 텔레메트리, 상세는 `results/n3_capacity/SUBMISSIONS_2026-10-06.md` "2회차 실패 진단"):
  - `event_loop_pdmux`의 순서는 입장+시작 이벤트 기록 → 끝난 요청 제거 → stream 조정 → prefill 발사다. 그래서 시작 이벤트는 **조정 전** 상태를 적는다.
  - 실현 분할은 (108,0)이었다. 근거: span_end stream 0, +15/+47 ms 스냅샷 stream 0, span 안 decode iteration 0건.
  - 직전 decode→prefill 간격은 124.8 ms였다(정상 16–20 ms). 스케줄러 정지 원인은 cgroup CPU 쓰로틀이 후보이나 **미확정**이다.
  - R3 arm identity는 이 라벨을 쓰지 않는다. 영향을 받는 곳은 floor 판정(전 span 라벨과 마지막 16 `running_bs`)뿐이다.

## 코드·문서 변경

**커밋됨**
- `5c58dbc`: n3_plan.py/n3_analyze.py rev5, 사전등록 rev5(머리말·§17-4), `tests/test_n3_capacity.py` +2(63 tests), rev5 델타 판정서, 1회차 회수 메타·preflight 산출물.
- `79999aa`: doc_facts 10차 재발 정정 955→957(PROJECT_STATUS #245, CONSENSUS §3 항목265, kernel_mech DESIGN :382).
- 그 이전 커밋 `7203672`…`a4b0c2e`: E2 1회차 진단, A14 등록, 6그룹 프로브, N3 rev3·rev4·하네스, E2 r2 제출·회수(git log 참조).

**이 핸드오프 커밋에 포함**
- 이 문서.
- `SUBMISSIONS_2026-10-06.md`(2회차 제출 기록 + 2회차 실패 진단), `vessl_jobs.tsv`, 2회차 회수 메타·OUT.
- E2 r2 결과 문서·결과 감사·analysis_2026-10-06.
- doc-steward 정본·메모리 반영분.

**커밋 제외**: 원시 `clk.csv`(관례), `he0_realized_2026-10-01.json`(이번 세션 무관).

## 열린 항목 / 다음 세션 시작점

1. **N3 rev6 결정(사용자)** — floor 판정 입력을 입장 라벨에서 실현 증거로 바꾸는 안을 권고했다.
   - 각 span의 `prefill_span_end.stream_idx == 0` ∧ span 구간 안 decode_iteration 0건.
   - 마지막 16 span은 `n_new==1`·`queue_len==0`을 유지한다.
   - 근거: SLO 방법 감사 P-2가 이미 "realized 분할 (108,0)을 telemetry로 실증"을 요구한다.
   - 대안 두 개는 비권고했다. 엔진 이벤트 위치 이동은 엔진 manifest와 다른 캠페인의 의미를 바꾼다. warmup 제외는 floor bench 안의 같은 경쟁을 막지 못한다.
   - 승인되면: rev6 → claims-auditor 델타 감사 → 커밋 → dry-run → 재제출(새 job name, `RESUME_FROM` 금지).
2. **E2 r2 결과 문서 정합** — `RESULT_e2_vessl_r2_2026-10-06.md`를 결과 감사 §9 금지 문안 16건과 대조해야 한다. result-analyst 보고에 있던 정정 2건(E2VC-12 "양 arm 모두 021" 거짓, HEAD 표기)이 문서에 반영됐는지도 확인한다.
3. **N2(3-a) 사전등록**: E2가 넘겨주는 범위(E2VR-6 가져갈 수 있는 것·없는 것)를 승계해 착수한다.
4. **doc_facts 구조 수리 (i)** — 사용자 승인 대기(10차 재발).
5. N3 shape A는 B 결과 감사 뒤 별도 승인.

## 미완·주의

- 로컬 전체 회귀는 sglang import 불가(`sgl_kernel` 없음)와 PDMUX_ROOT·모델 파일 부재 때문에 838 tests 중 실패 27·오류 14다. 실패 모듈은 N3 파일과 무관하다. `test_n3_capacity` 63/63 OK.
- N3C-26 관련 사실: 2회차 Job에서 preflight unittest 묶음(`test_phase_events`·`test_green_readout` 포함) 46건이 VESSL 이미지에서 60.7 s에 OK(`n3b_pdmux-n3-capacity-79999aa-20261006081638/PREFLIGHT_TESTS.txt`).
- N3C-27: 1회차·2회차가 `/data/substrate_history.tsv`에 측정 0 행을 남겼다. 다음 Job의 `SUBSTRATE_WARNING` 해석에 주의한다.
- 테스트 결함(기존, rev5 무관): `TestBootBranchSnippets.test_t_h1_sticky`·`test_t_h3_floor`는 클린 클론에서 skip이 아니라 error가 난다(git-ignored 프로브 파일 필요).
- 미회수 Job 0. Workspace `pdmux-probe`·`pdmux-build` 둘 다 `standby`.

## GPU 실행 기록(VESSL)

| job name | 캠페인 | 커밋 | 이미지 | 상태 | fetch | cost.txt |
|---|---|---|---|---|---|---|
| `pdmux-e2-sticky-vessl-22c8c64-20261006020317` | e2_sticky_vessl (r2) | `22c8c64` | `sglang-runtime@1e0d335b…` | 완료 exit 0 | DONE+sha256 OK | 1.953 h |
| `pdmux-n3-capacity-a4b0c2e-20261006064114` | n3_capacity B | `a4b0c2e` | 동일 | failed exit 4 | DONE+sha256 OK | 0.016 h |
| `pdmux-n3-capacity-79999aa-20261006081638` | n3_capacity B | `79999aa` | 동일 | failed exit 7 | DONE+sha256 OK | 0.139 h |

Workspace: `pdmux-probe`(A100)·`pdmux-build`(CPU) 모두 standby.

---

# 2차 세션 (2026-10-06, 같은 날 속행) — N3 rev6 승인 → 델타 감사 → 3회차 Job 완주 → 결과 감사 → 정본 rev89

## 이번 세션 요약

catch-up 브리핑의 선택지 A("N3 rev6 승인 → 사전등록 개정·델타 감사·재제출까지 진행")를 사용자가 "a"로 승인했다. floor 판정 입력을 입장 시점 라벨에서 실현 증거로 바꾼 rev6를 작성하고, claims-auditor 델타 감사(`GO-with-caveats`, 死因 0, R-1…R-4)를 받아 반영한 뒤 커밋·제출했다. shape B 3회차 Job은 **완주**(exit 0, 컨테이너 3.838 h)했고 rev6 floor는 실제 Job에서 유효했다. 다만 예산 입장 규칙이 round 4 상위 4 rate + floor_tail을 거부해 **43/48 unit**만 측정됐다(등록 예산 예보 실패). 결과 감사(死因 0, 라벨 바이트 재현 `CONFIRMED`)와 정본 반영(CONSENSUS rev89)까지 끝났다. 새 성능 판정 0건, 게이트 #6은 **부분 측정·미닫힘**, HE0·Claim 등급·λ0R-8(iii) 불변.

## 결정·측정

**사용자 지시**: "a"(catch-up 선택지 A 승인) 한 건. 재제출은 이 승인과 2회차 선례("수리 진행 후 실행")에 따라 dry-run 2회(커밋 전·후) 확인 뒤 `--submit`했다.

**rev6 (커밋 `38ed3f4`)** — `PREREG_N3_CAPACITY_2026-10-04.md` 머리말 rev6 줄·§4-1·§8-5·§11(N3C-28…33 승계)·§12-23·§13·§16·§17-5.
- `n3_plan.floor_from`: (iii) 각 span `prefill_span_end.stream_idx == 0` ∧ 구간 [start, end] 안 `decode_iteration` 0건; (ii) start↔end 짝·`dropped_events == 0`; (v, 감사 R-2) 각 floor span 뒤 decode_iteration ≥ 1(기록기 생존); (vi, W-1) span_seq 유일성; (iv) 마지막 16 span `n_new/queue_len` 1/0 유지, 입장 `running_bs` 제외. 기술값(R-3) `floor_entry_race_count`·`floor_admission_gap_ms`. `FLOOR_STREAM_IDX = 0`(check_green이 boot마다 group 0 → (108,0) 고정).
- 러너: ABORT_FLOOR 메시지 1줄 + (R-1) floor_head boot에 R7 표식(`phase_events_disabled_*`)이 있으면 floor_from 결과와 무관하게 `ABORT_FLOOR`.
- 테스트: `test_t_h3_floor` 변이 교체(실현 증거 + 경쟁 양성 + R-1 대조·중화) + `TestFloorRev6RealJob` 3개(2회차 실자료, git-ignored 부재 시 skip). 전체 모듈 **66/66 OK**. doc_facts 11차 정정 957→960.
- 델타 감사 `VERDICT_n3_rev6_delta_2026-10-06.md`: Q1 CONFIRMED(조건부: end 라벨은 "span 안 decode 0건"과의 논리곱으로만 실현 증거), Q2 fail-open 2건 실증 → R-1·R-2, Q3 지연 측면 REFUTED(입장 라벨 경쟁 = admission 정지 지문, 그 요청 TTFT +106 ms, rev6는 그 요청을 floor 표본에 넣음 — N3C-30), Q4 CONFIRMED(수리 되돌림 M1 3경로 검출). W-5(bench 창 대조)는 입력 추가라 **미반영·사용자 결정**.

**3회차 Job `pdmux-n3-capacity-38ed3f4-20261006092813`** (slug `job-n60kw8bsejuz`, A100 `GPU-c36b3c0c…`, 09:28–13:19Z).
- floor 유효: floor_s 0.905 s, T_3/5/10 = 2.715/4.525/9.050 s, floor 창 경쟁 0(boot 전체 warm-up 1건), dropped 0.
- 43/48 unit; `NOT_ATTEMPTED_BUDGET` g_r053/061/069/077_4589 + floor_tail(입장 시 remaining 31 s < need 253 s). 원인: unit 입장→다음 입장 중앙 293 s vs need 250–253 s(부대비용 중앙 170 s vs 상수 121 s, bench 고정비 ≈20 s 모형 누락). preflight는 224 s(허용 600).
- 라벨(`N3_RESULT.json`, 로컬 재채점 바이트 동일): k=3 `CAPACITY_BRACKETED_3` [0.15, 0.22) req/s, bootstrap [0.15, 0.22](짝 seed n=3), 0.15 attainment 73/80 = 0.9125(문턱 72/80 — **2요청 여유, 절벽 구간**), 0.22 75/88; STATIONARY; k=5·10 `UNRESOLVED_N_BELOW_4`(missing_at 0.53, n=3 — 측정 부족). observer `OBSERVER_OK`(TTFT +1.2% [−0.4, +4.0], ITL +1.6% [−0.07, +3.4]). CG2 PASS, P3 43/43, arm identity 0, retract 경고 0(관측 안 됨), R7 0. 용량 셀 (64,44) E-time 23.8–31.2% — **D44 측정 아님**. throttle 21439/137581 샘플(cgroup quota 11 CPU), bench 창 throttled 0.57%.
- 결과 감사 `VERDICT_result_n3_B_r3_2026-10-06.md`: 死因 0, 신규 caveat **N3VR-1…11**, 금지 문안 17항목, 정정 R-1(unit 시간 기준)·R-2(경쟁은 ON boot **36/37**에서 warm-up span_seq 6 — 2회차 "1/28·~40%" 어림 틀림, 인용 금지)·R-3(0.15 CI [0.90,0.90]은 동률 퇴화). T(floor) 견고 범위 −12.7%…+15.7%. 도착 순서 반분 진단: 전반만 0.10, 후반만 0.30(N3VR-3, 등록 외 조작 — caveat만).
- 결과 문서 `RESULT_n3_B_r3_2026-10-06.md`(감사 대조 배너, 저촉 0).

## 코드·문서 변경 (전부 커밋됨)

| 커밋 | 내용 |
|---|---|
| `38ed3f4` | rev6(n3_plan.py·n3_capacity.sh·prereg·tests)+델타 감사 판정서+doc_facts 11차(PROJECT_STATUS #245·CONSENSUS 항목265·kernel_mech DESIGN) |
| `e502f1e` | 제출 원장(SUBMISSIONS 3회차 절·vessl_jobs.tsv) |
| `0c7b3b8` | 회수 아티팩트 382 files(OUT 추적 파일·vessl meta, clk.csv 제외) |
| `8e1900b` | 결과 문서 + 결과 감사 판정서 |
| `59bb08a` | 정본: PROJECT_STATUS (a′)+게이트 #260–261·"다음 gate" #19 갱신, CONSENSUS rev89(§1 행 36·§3 항목 284–285·§5 항목 19 해소/20 신설), ROADMAP 배너, SUBMISSIONS 정정 |

메모리(저장소 밖): `slo-aware-scheduling-track.md` 2026-10-06(2차) 절, `deconfound-measurement-lessons.md` 항목 272–273, `ladder-decisions-2026-10-04.md` rev6 승인 줄, `MEMORY.md` 헤드라인 갱신 + `no-billing-commentary` 인덱스 줄 추가(1차 세션 누락분).

## 열린 항목 / 다음 세션 시작점

1. **사용자 결정 4건(N3)**: (i) 결측 4 unit(seed 4589 0.53–0.77)+floor_tail을 §8-4 재개 Job으로 채울지(`RESUME_FROM`, T_K 동결, Job 혼합 병기 의무 N3VR-6·E2VR-3, 이전 Job 디렉터리 `/data/runs/<job>` 보존 미확인) vs 새 Job vs 보류 (ii) 예산 상수 재추정(overhead 121→≈170 s, bench 고정비 ≈20 s) = rev7 + 델타 감사 (iii) shape A 착수 (iv) W-5 반영.
2. **N2(3-a) 사전등록** 착수 — E2VR-6 승계 범위(불변).
3. E2 r2 결과 문서 정합 대조(1차 세션 열린 항목 2) — 이번 세션 미착수.
4. doc_facts 구조 수리 (i) — 사용자 승인 대기(11차 재발).
5. 교훈 재발 주의: 2회차 진단의 "1/28·~40%" 어림이 결과 감사에서 36/37로 뒤집힘 — n=1 boot 어림을 계획 근거로 쓸 때도 공시 필요(N3VR-7).

## 미완·주의

- k=3 라벨은 요청 2건 여유의 절벽 위 — 단독 수치 "0.15" 인용 금지(§9 허용형 전문으로만).
- 결과 문서 §12의 unit 시간은 입장→종료 기준(감사 R-1 정정 병기).
- `TestFloorRev6RealJob`은 Job preflight(클린 클론)에서 skip(N3C-32).
- 미회수 Job 0, 실행 중 Job 0. Workspace `pdmux-probe`·`pdmux-build` 둘 다 standby.
- 로컬 전체 회귀는 1차 세션과 같이 sglang 부재로 일부 모듈 실패(N3 무관).

## GPU 실행 기록(VESSL)

| job name | 캠페인 | 커밋 | 이미지 | 상태 | fetch | cost.txt |
|---|---|---|---|---|---|---|
| `pdmux-n3-capacity-38ed3f4-20261006092813` | n3_capacity B(rev6, 3회차) | `38ed3f4` | `sglang-runtime@1e0d335b…` | succeeded exit 0 | DONE+sha256 639/639 OK | 3.838 h |

Workspace: `pdmux-probe`(A100)·`pdmux-build`(CPU) 모두 standby(이번 세션 미사용).
