# 하네스층 한정 델타 재확인 — rev11 D-A·D-B·D-C, 커밋 6f3611c, 2026-10-13

> 감사자: claims-auditor(읽기 전용, GPU 0). 요지를 메인 세션이 저장(필수 수정·caveat 문안 원문 그대로).

## 0. 판정
**M3 하네스 `NO-GO`(6f3611c)** — 死因 1(D-D, N2): D-B 처방(감사자 자신)의 try/finally가 새 fail-open(#113 자기 적용). D-A·D-B(parse/reconstruct)·D-C 닫힘. **M2g `GO-with-caveats` 유지**(D-D 수정 트리에서 M3와 함께 봉인 권고).

## 1. 이행 확인
- D-A 닫힘(raw 대입 = TJS ∧ raw ∈ DIRECTIONAL, raw_label_used는 BROKEN 승격 시만, _syn_noisy 시나리오는 실경로·중간 상태 단언으로 항등식 아님; 단 noise 0.15에서 seed 0–19 중 판별 배치 1/20(seed 4) → MGH-33).
- D-B(parse) 닫힘(4 Job 필수, 조용한 부분집합 rc 3, 명시 생략 시 JOB_NOT_JOINED 공시 → MGH-31 승계).
- D-B(reconstruct) fail-closed(SIGKILL boot는 boot.json 없음 → RECONSTRUCTED_NO_BOOT_RECORD; 로그 결손·pass table 없는 --root는 과대 절단 → MGH-34).
- D-C(i) 닫힘, config 단위 증거가 인과적으로 옳음; 반전 불가 논거: DP 확장 boot 하드캡 기록 시 잔여 < extAB need(N2/4/8 4735/3899/3230 s) < TP G_sym 최소 ≈8.6–9.9k s, 순서 [DP,TP] → TP 점 미정의 → MGH-32.

## 3. 死因 D-D
finally로 예외 시에도 finish()가 driver 레코드를 써서, 예외로 도달 못 한 계획 boot가 not_attempted에 없음 → label_job_dir가 reconstruct_driver를 부르지 않음 → SIGKILL(절단)과 예외(비절단 P9-c)가 갈림, 공시 없음. 실경로 예보(rev.py): J4 DP4 EXT_INCOMPLETE(기동 실패), J8 driver가 DP8 boot2 전 사망, TP 1,1.05,1.10 → finally 경로 TP G_INCREASES citable, 재구성 경로 TP_UNCONTROLLED_TRUNCATION.
**필수 수정**: (1) DRIVER_LOG에 DRIVER_EXCEPTION_FINISH_IN_FINALLY가 있거나 driver 레코드에 예외 표지가 있으면 reconstruct_driver 실행, RECONSTRUCTED_NO_BOOT_RECORD 항목을 boot_id 기준 중복 제거해 driver.not_attempted에 병합(추가만). (2) 그 Job 모든 trend에 JOB_DRIVER_EXCEPTION(J) 공시. (3) 위 시나리오 tmp+label_job_dir+aggregate_jobs 실경로 회귀(TP_UNCONTROLLED_TRUNCATION), "finally 레코드 완결 신뢰" 변이 KILLED, 정상 Job 병합 0건.

## 4. caveat (문자 그대로 승계)
- **MGH-32**: "NOT_ATTEMPTED_EXT_INCOMPLETE의 절단 판정은 config 단위다. 같은 config의 어느 arm 확장 boot에든 하드캡 기록(NOT_ATTEMPTED_BUDGET·partial boot·확장 셀 TRUNCATED)이 있으면 그 config의 모든 boot2 미진입을 절단으로 센다. 기동 실패와 하드캡이 겹친 config는 원인을 가리지 않는다. 이 처리가 TP 라벨을 바꾸지 못한다는 논거는 config 순서 [DP, TP]와 rev11 boot_need_s에 의존한다."
- **MGH-33**: "D-A 회귀 시나리오(S_da_shape_b_tjs_raw_no_trend_is_untested)는 잡음 seed 4로 고른 합성 입력이다(seed 0–19 중 판별 배치 1/20). 실경로 구별력의 증거일 뿐, 실측에서 그 배치가 얼마나 자주 나오는지를 뜻하지 않는다."
- **MGH-34**: "RECONSTRUCTED driver 기록은 DRIVER_LOG.jsonl에 도달 사건이 없는 config를 계획된 것으로 보고, boot 기록이 없으면 절단으로 센다. 로그가 없거나 잘린 Job에서는 M2g에서 제외된 config나 TP_NOT_TESTED가 될 config도 절단으로 셀 수 있다(fail-closed 방향의 과대 절단)."
- MGH-31 승계, MGH-30 미승계, MGH-24″·27·29, MGP-43′·44–48·49′·50′·51–58, MGH-3·21·22·23·26 승계.
