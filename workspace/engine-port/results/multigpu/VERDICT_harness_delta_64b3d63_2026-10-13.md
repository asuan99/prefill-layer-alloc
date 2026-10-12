# 하네스층 델타 재확인 — rev11 하네스, 커밋 64b3d63, 2026-10-13

> 감사자: claims-auditor(읽기 전용, GPU 0). 요지를 메인 세션이 저장(필수 수정·caveat 문안 원문 그대로).

## 0. 판정
M2g 하네스 `GO-with-caveats`(死因 0; D-A~D-C가 mg_label을 바꾸므로 수정 후 같은 트리에서 M3와 함께 봉인, 64b3d63 단독 봉인 비권고). **M3 하네스 `NO-GO`(64b3d63)** — 死因 3(D-A, D-B, D-C, 모두 N2). D-A는 rev11 수정 자신이 만든 fail-open(§18 "새 fail-open 없음"과 모순).

## 1. 재확인
M-1 해소(단 D-C), M-2 해소(flags_final 변이 구별력 있음), M-3 해소(단 D-B), M-4 구현 결함(D-A; raw_label_used가 HELD에도 기록되는 오기재 포함), R-1·R-2(MGH-25′ 소멸)·R-3·R-5 안전, R-4 운영 항목 유지. PDP 시나리오 "P_A_no_trend_B_job_sensitive"는 합성 입력에 raw_label이 없어 실경로 미시험.

## 2. 반전
- **D-A**: 비앵커 shape B가 TJS이고 raw NO_TREND면 rev11은 P-DP HELD·TP 공시 [] (P9-c/rev10은 UNTESTED·'P-DP 미검정'). 시나리오: 앵커 J2 1.00 CLEAR, J4 0.90 FIRED, J8 0.80 FIRED; DP-A raw 1,1,.95,.90(NO_TREND, CI [−0.080, 0.006]); DP-B raw 평평 → raw NO_TREND·보정 G_INCREASES → TJS; TP-A 1,1.05,1.10,1.20 G_INCREASES.
- **D-B**: `--jobs`가 임의 부분집합 허용 — J8 생략 시 DP-A 1,1,1,1.10(BROKEN, MGP-13) → 1,1,1(HELD), TP 공시 없이 인용. "driver 기록 필수"로 finish() 전에 죽은 Job은 결합 경로가 없음.
- **D-C**: CUT_REASONS가 NOT_ATTEMPTED_EXT_INCOMPLETE를 사유 불문 포함 → 확장 boot 기동 실패(비절단)도 TP_UNCONTROLLED_TRUNCATION(fail-closed 방향). 직전 M-1 처방에 들어 있던 사유 — 처방자 결함(#113).

## 3. 봉인 전 필수 수정 (M3)
- **D-A**: 비앵커 shape raw 대입을 `raw_label ∈ DIRECTIONAL`일 때로 한정(그 외 TJS 유지 → UNTESTED). "A HELD + B TJS(raw NO_TREND) → UNTESTED" 시나리오를 aggregate_jobs 실경로로 추가, 기존 PDP 시나리오에 raw_label 채움, "raw 무조건 대입" 변이 KILLED, raw_label_used는 BROKEN 승격 시에만 기록.
- **D-B**: `--jobs`는 J1·J2·J4·J8 모두 요구, 생략은 명시 인자(예: `J8=NOT_SUBMITTED`)일 때만, 결과에 `JOB_NOT_JOINED`. driver 없는 Job은 거부 대신 재구성 — DRIVER_LOG.jsonl NOT_ATTEMPTED_* 이벤트를 boot_id regex로 분해, partial_boots_of 적용, 계획에 있으나 boots/<id>/boot.json 없는 main boot는 절단(fail-closed). run_m3를 try/finally로 감싸 finish() 보장. 변이: "부분집합 결합 허용"(J8 생략 시나리오 KILLED), "driver 결측 Job 거부".
- **D-C**: (i) EXT_INCOMPLETE는 같은 cfg 확장 boot에 BUDGET·PARTIAL_BOOT·TRUNCATED_CELL 기록이 있을 때만 절단, 기동 실패는 비절단(P9-c), "EXT_INCOMPLETE 무조건 포함" 변이 KILLED. 또는 (ii) 사용자 결정으로 §18에 "원인 불문 절단" 등록 + MGH-30.

## 4. caveat (문자 그대로 승계)
- **MGH-24″**(MGH-24′ 대체): "rev11 M2g J2·J8의 G4 nsys boot는 이미지에 nsys가 없어 실행하지 않았다(`NOT_RUN_NSYS_ABSENT`). G4 'PASS'는 주 boot의 cudagraph 캡처 성공만 뜻하며 all-reduce 커널 이름 증거는 없다."
- **MGH-27, MGH-29** 승계. **MGH-25′** 소멸. **MGH-28** 미승계.
- **MGH-30**(D-C를 (ii)로 닫을 때만): "확장 미완(NOT_ATTEMPTED_EXT_INCOMPLETE)으로 boot2에 들어가지 않은 config는 원인(하드캡 / 기동 실패)을 가리지 않고 절단으로 처리했다. `TP_UNCONTROLLED_TRUNCATION` 일부는 하드캡 절단이 아닐 수 있으며, 그 경우 등록 §17-2(b)의 비절단 경로였다면 TP 방향을 공시 후 인용할 수 있었다."
- **MGH-31**(D-B 수정 후 명시 생략을 쓸 때만): "`JOB_NOT_JOINED` Job이 있는 결합의 추세는 그 N을 뺀 점으로 계산했다. 빠진 N이 있었다면 P-DP 상태와 TP 인용 가부가 달라질 수 있다. 실측: J8 생략만으로 BROKEN이 HELD로 바뀌는 사례가 있다."
- 승계: MGP-43′, 44–48, 49′, 50′, 51–58, MGH-3, 21, 22, 23, 26.
