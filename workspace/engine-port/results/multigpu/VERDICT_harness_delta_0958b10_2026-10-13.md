# 하네스층 한정 재확인(D-D) — 커밋 0958b10, 2026-10-13

> 감사자: claims-auditor(읽기 전용, GPU 0, 범위 D-D 고정). 요지를 메인 세션이 저장(caveat 원문 그대로).

## 판정: `GO-with-caveats` — M2g/M3 하네스 봉인 가능, 死因 0
- 처방 (1) 재구성+추가 병합(`mg_label.py:1907-1920`, 표지 `mg_drive.py:656`·`:262`), (2) JOB_DRIVER_EXCEPTION 공시(하위 경로 1곳 누락 → 권고 R-1), (3) Rev11DeltaDD 5건 — trust_finally·no_exception_disclosure 변이 KILLED, 정상 Job 병합 0, `test_driver_finish_writes_exception_marker`는 실제 D.main finally 경로.
- 새 fail-open 탐색(독립 프로브): finish 쓰기 전 예외 → D-B 재구성 fail-closed(na 16), finish 쓰기 후 예외 → exception=True 이미 기록, run_m3 반환 후 예외는 finish 안뿐(도달 불가), run_m3 중간 KeyboardInterrupt → 병합 4건 = 미도달 boot와 일치, 진행 중 boot 사망 → boot.json 없음·spec 있음 → 절단. 반전 없음.
- 권고 R-1(비차단): D-B 분기에서도 DRIVER_LOG의 DRIVER_EXCEPTION_FINISH_IN_FINALLY면 driver.exception=True — **메인 세션 반영**(회귀 `test_r1_no_labels_json_but_exception_logged_still_discloses`, 변이 KILLED) → C-1 소멸.

## caveat (문자 그대로 승계)
- ~~**C-1**~~(R-1 반영으로 소멸).
- **C-2**: "DRIVER_LOG.jsonl은 append 방식이다. 같은 출력 디렉터리에 재실행하면 이전 실행의 예외 표지가 남아 `JOB_DRIVER_EXCEPTION`이 공시될 수 있다. 이 경우는 fail-closed이므로 그 공시를 그 실행에서 예외가 났다는 증거로 인용하지 않는다. Job마다 새 디렉터리를 쓰는 것이 전제다."
- **C-3**: "D-D 수리는 CPU 회귀(합성 estimand + 실제 driver 병합·집계 경로)와 dry-run 프로브로만 검증됐다. 실 GPU Job에서 예외가 난 경우는 관측된 적이 없다."
- 확정 조건(결과 감사): 첫 실 Job 회수 시 driver.exception 유무, 있으면 exception_merged_boot_ids = boot.json 없는 계획 boot 집합 대조.
