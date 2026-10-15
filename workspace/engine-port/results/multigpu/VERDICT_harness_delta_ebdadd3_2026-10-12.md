# 하네스층 국소 재확인(델타) — PREREG_MULTIGPU rev6 하네스 F1–F17, 커밋 ebdadd3

> 감사자: claims-auditor(읽기 전용, GPU 0), 2026-10-12. 요지를 메인 세션이 저장했다(필수 수정·caveat 문안은 원문 그대로).

## 0. 판정
| 범위 | 등급 | 요지 |
|---|---|---|
| **M2g 하네스** | **`GO-with-caveats`** | 死因 0. **R1**(F1 수정이 드러낸 비정상 경로 반전)은 고치기 전 봉인 금지. R2 기록 결손. 나머지 F 처방대로 닫힘 |
| **M3 하네스** | **`GO-with-caveats`(단일 Job 한정)** | 死因 0. 봉인 전 필수 R3–R5(R3·R4 조건부 반전). M2g PASS 표·sha 커밋 뒤에만 봉인. 블록 분할 시 미적용 |

## 1. 독립 재검증
- 테스트 276+engine_digest OK(메인 280과 일치), `mg_label --selftest` PASS, g5a PASS(tree digest `82a5fc5b…`, UNSEALED라 rc 3 정상).
- 변이(이번 수정분) 5/5 KILLED: score_g1 None 분기 삭제, alive_after가 사본 필드 읽기, score_g3 FAIL 우선 삭제, F10 boot2 차단 삭제, F5 CONTAM_UNVERIFIED 기록 삭제.
- ENGINE_DIGEST pathless-v1 = `82a5fc5b5f587464e0df8037b64791900a013f61024d128e451818793526781c` 재현, 직전 VESSL manifest 앞 25개 중 21개 일치, 차이 4개(multiplexing_mixin, telemetry, holb_probe, chunk_probe)는 M2 엔진 커밋 c0e200d로 전부 설명(CONFIRMED).
- m3-forecast·m2g-budget(37 boot 5.58 h) 재생성 일치.

## 2. 반전 시험 핵심
- **R1(반전)**: `G3_COMBO_TIMEOUT_S`=900 × G3로 이미 제외된 TP의 서빙 boot도 전부 실행 × G0(ii) 거부 boot가 꼬리. TP4·TP8 NCCL 초기화 hang 시 누적 7.40 h → DP4부터 입장 거부 → DP4·DP8·g5a·G0(ii) 미실행 → 원인과 무관한 TP2·DP4·DP8 제외. G3≠PASS인 TP{n} 서빙 boot를 건너뛰면 4.11 h 회수.
- score_g3 FAIL 우선, preflight 실패, timeout 하한: 반전 없음.
- G3 timeout 후 고아 worker(torch elastic `start_new_session=True`) → MGH-21. `/health` 판독 → MGH-16.
- F5 NVML: M2g fail-closed, M3는 실행 중 상실 때만 → MGH-19.
- **R3(조건부 반전)**: M3 pair 자신의 arm cell_meta 결손(`memory_not_tested_arms`)을 `defined()`·`disclosures()`가 읽지 않음.
- **R4(조건부 반전)**: config 내 오염이 config 전체 pair를 죽임(직전 F11 처방 불완전, #113).
- ENGINE_DIGEST 정렬: 반전 없음 → MGH-17.
- **R5(M3 반전)**: Job 안 g5a FAIL이 M3 입장을 막지 않음.

## 3. F 종결: F1·F2·F3·F4·F5·F6·F8·F9·F10·F12·F13·F15·F17 닫힘. F7 기전 닫힘 — `82a5fc5b…`로 봉인 가능(불일치 결과는 측정 전 rc 13 fail-closed), F7(a) bring-up 필수 아님. ABORT 시 회수 manifest를 `compare-prefix --n 27`로 대조해 설명된 차이만 재봉인(사용자 결정 + 국소 델타), 컨테이너 값 무검토 재봉인 금지. F11 처방대로 닫힘·잔여 R4. F14 부분(R3). F16 부분(R2).

## 4. 봉인 전 필수 수정
- **R1** G3 verdict ≠ PASS인 TP{n}의 서빙 boot(`m2g_TP{n}_*`, `m2g_g4_TP{n}_d44`)를 `NOT_ATTEMPTED_G3_EXCLUDED`로 기록하고 건너뜀.
- **R2** MANDATORY_REG에 `engine_digest.py`, `sync_engine_tree.sh`, `test_engine_digest.py`, `ENGINE_DIGEST.txt` 추가·문턱 갱신.
- (권고) `run_g3` 뒤 `score-g3` 재호출이 preflight·not_attempted를 지우지 않게, G3 timeout 뒤 `pkill -f m2g_comm_reverse_stress.py`, 서빙 boot 전 NVML 빈 GPU 1회 확인.
- **R3** pair의 arm이 `memory_not_tested_arms`에 있으면 `"<arm>:MEMORY_NOT_TESTED"`로 undefined(pair 밖 arm 결손은 유지).
- **R4** `gpu_contaminated` 범위를 boot의 arm으로, 그 arm을 포함하는 pair만 undefined.
- **R5** `m3_job.sh`/`load_pass_table`이 `table["g5a_unit_test"] == "PASS"` 요구(아니면 ABORT_M2G_NOT_PASSED).

## 5. caveat (문자 그대로 승계)
- **MGH-15** "G3의 조합별 외곽 timeout 900 s는 미등록 하네스 상수다. 진행형 watchdog(120 s)은 반복 루프 진입 뒤에만 무장되므로 초기화(gloo rendezvous·PyNccl·green context 생성) 단계의 hang은 FAIL이 아니라 NOT_TESTED(HARNESS_ERROR)로 기록된다. 'G3 NOT_TESTED = 통신 hang 없음'으로도, 'HARNESS_ERROR = 하네스 결함 확정'으로도 쓰지 않는다."
- **MGH-16** "G1 '비정상 종료 0'과 G3b '서버 생존'은 boot의 마지막 step 직후 1회, `poll()`과 SGLang `/health`(1토큰 생성 프로브, 클라이언트 timeout 10 s) 응답 200으로만 판정한다. step 중간의 정지·재기동은 보지 않는다."
- **MGH-17** "ENGINE_DIGEST(pathless-v1)는 등록 27개 파일 hash의 정렬 집합이다. 등록 밖 파일(이미지의 나머지 SGLang 트리)은 고정하지 않고, 항목 26 `pdmux_context.py`는 봉인 시점에 컨테이너에서 관측된 적이 없다. 'Job 엔진 트리가 로컬과 같다'고 쓰지 않는다 — '등록 27개 파일의 내용 집합이 같다'까지만 쓴다."
- **MGH-18** "M2g 예산 5.58 h는 VESSL boot 부대비용(+49 s/boot, 약 +0.5 h), G3b drain(+0.5–1.1 h), G3 초기화 hang(조합당 최대 900 s, 최대 +2.25 h)을 넣지 않은 모형값이다. 'M2g가 8 h 안에 끝난다'고 쓰지 않는다."
- **MGH-19** "NVML 불가로 오염 검사가 돌지 않은 boot는 M2g에서는 제외(CONTAM_UNVERIFIED)되고 M3에서는 공시만 된다. M3 결과 옆의 CONTAM_UNVERIFIED를 '오염 없음'으로 읽지 않는다."
- **MGH-20** "F15 예보에서 VESSL 부대비용을 보정한 모든 시나리오(S1 vessl_boot 67/70 boot, S2 전부)는 round-2 앵커 3 boot를 전부 잘랐다. 단일 Job M3에서 `ANCHOR_DRIFT` 미발화는 기본적으로 '측정 안 됨'이다(예보이며 측정이 아니다)."
- **MGH-21**(권고 pkill·NVML 확인 미반영 시 승계) "G3 timeout 시 하네스는 launcher의 프로세스 그룹만 종료한다. torch elastic worker는 별도 세션이라 잔존할 수 있고, 그 경우 뒤따르는 boot의 GPU_CONTAMINATED는 외부 오염이 아니라 G3 잔존 프로세스일 수 있다."

## 7. 다음 gate
R1·R2 반영·국소 확인 → ENGINE_DIGEST `82a5fc5b…` 기입·g5a `--record`로 G5A_PASS.json·M2g 봉인 묶음 커밋 → M2g dry-run → 사용자 승인 → 제출 → 회수 → PASS 표·sha 커밋 → R3–R5 → M3 봉인 → 예산 결정 → M3 제출.
