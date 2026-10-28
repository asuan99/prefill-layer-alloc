# 하네스층 감사 r2(최종, 2/2) — PREREG_DIAG_stall_drain rev0.3 + 3858f129 (2026-10-28)

- 감사자: claims-auditor(읽기 전용) · **판정 NO-GO → 사용자 결정 필요**(2회차에도 死因, 3회째 처방 없음)
- 재현: `tests/run_diag_tests_with_stub.sh` 실행, ENGINE_DIGEST 65cbb49d…b777 = P-8, **110 OK, skipped 3**(decode_probe hook-patch 시험), 변이 12 모두 사망(`test_phase_diag.py:589-669`, anchor 유일).

## r1 항목 상태
K1 **OPEN(항등식이 X3로 이동)** · K2 CLOSED(①–④ 순서; ④ 내부 순서·"R·Z" 결합 미등록은 D1에 포함) · M1 CLOSED(`test_static_nonsticky_partition_and_drains`, drain 수 < span 수) · M2·M5·M6 CLOSED · M3 CLOSED(`check_clocks` lag p50/p99) · M4 NEW-ISSUE MINOR · M7 a–d CLOSED(D7 양성 대조 `test_sync_audit_attributes_calls_to_diag_frames`; GPU에서는 래퍼 생존 `total_stream_synchronize>0`만) · M8 CLOSED.

## 死因
- **D1 K1 수리가 항등식을 X2→X3로 옮김(N1+N2)** — r1 감사자 처방이 함께 죽은 사례(게이트 #113 재발). `t_launch`는 decode `run_batch` 뒤(`multiplexing_mixin.py:1405`), sync는 decode stream만(`:1494`) → R ≈ max(X3 − h_d − g − X2, 0) + ε, R의 X3 기울기 = "decode가 발행보다 늦게 끝난 반복 비율". Z = H − max(X2,X3)는 구조적으로 평평. 예보(간섭 0 생성기, scratchpad `sim.py`): backpressure 0.05→0.8에서 `STALL_FOLLOWS_DECODE_GPU`→`STALL_MIXED`(R의 X3 기울기 +1.00→+0.56, X1 기울기 −0.05/−0.20) — 간섭 0인데 DECODE_GPU. "R·Z" 결합 해석(:82)에 따라 같은 데이터가 `FOLLOWS_DECODE_GPU` ↔ `HOST_ISSUE_ACCOUNTED`(Z의 X3 기울기 0.000, |Z| p50 0.15 ms, H p50 19.8 ms). `end_done_at_issue_end`는 chunk 완료를 가를 뿐 decode 대기 영역을 가르지 않음. lockstep-backpressure (a)는 R의 X1 기울기 음수(−0.29)로 `CHUNK_GPU` 도달 불가.
- **D2 라벨 단위 삭제(N3, 이번 diff)**: rev0.1의 "(모델, 프롬프트) 풀에서 라벨" 문장이 rev0.2에서 삭제. 허용 문안 틀(:193)은 셀 단위 → 군집 (cell, seed) 4개 < FEW 기준 6(:82) → 12셀 전부 `UNRESOLVED_FEW` 결정론적. 풀 단위면 군집 24.

## 새 문제(비死因)
- MAJOR: LAUNCH_BOUND 쪽이 빌 때 "두 분류 일치" 규칙(:77) 미정의(큰 extend chunk는 record 직후 query가 거의 항상 False 예상, 미실측).
- MINOR: X3 짝짓기 규칙 이중(§3-2:79 host bracket vs P-2:25 `decode_iter_seq`) · X1 조인 `decode_launched=True` 필터 미등록 · §13:209 "PENDING_FINAL_TREE" 잔여, :211 행 깨짐, P-1:24 "구현 중", §3-3:87 "세 arm"(D-3은 4 arm) · D7 대상 밖 `.cpu()`/`.tolist()`·모듈 함수(`_phase_diag_clock_pair`), D7 boot는 측정 구성 아님.

## 사용자 결정 필요 — 판별량 A 라벨(STALL_*) 봉인 여부
- (권고) A: 판별량 A를 라벨 0 순수 기술 계측으로 강등, B·C 라벨만 봉인(문서 수리, 새 감사 없음)
- B: A 재설계(예: `t_issue_end`에서 decode end event query 플래그, X3를 X1에 회귀해 간섭 직접) — 새 설계·감사 주기
- C: D-1 보류, D-3만 진행

## A 선택 시 승계 caveat(인용 금지)
- "판별량 A의 R·Z 기울기는 루프 회계 항등식(sync는 decode stream만 기다린다)의 산물이며, 정지 기전 (a)/(b)/(c)의 판별로 인용하지 않는다."
- "`end_done_at_issue_end` 분류는 chunk 완료 여부이며 decode 대기 영역을 가르지 않는다."
- "D7 PASS는 GPU에서 래퍼가 살아 있었다는 것과 diag 스택 0회를 뜻할 뿐이며, diag 스택 판별 자체는 CPU 양성 대조로만 확인됐다."
