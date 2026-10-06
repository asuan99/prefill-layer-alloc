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
