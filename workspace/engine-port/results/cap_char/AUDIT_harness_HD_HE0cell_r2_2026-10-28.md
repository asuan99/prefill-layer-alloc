# 하네스층 감사 r2(최종, 2/2) — PREREG_HD_HE0cell rev0.3 + policy_harness (2026-10-28)

- 감사자: claims-auditor(읽기 전용) · 범위: 수리 diff `1576aa55..d4e8773c` · CPU 62 OK, 변이 50 등록·생존 0
- **판정: GO-with-caveats** (死因 0). 이 판정은 정책 결론을 닫지 않는다. n2·n6은 봉인 전 수리 권고(미수리 시 caveat 승계).

## r1 항목 상태
D1 digest CLOSED(65cbb49d, `--sealed-file` 포함) · D2 CAP trace CLOSED(`ph_trace.py:122` 키 `("cap",seed,rate)`, 시험 `test_cap_trace_identical_for_every_arm`, 변이 M32) · D3 sticky CLOSED(코드; 엔진 비sticky 의미는 Q-3 smoke 전 미실측 → caveat 1) · D4 CLOSED(bracket 규칙 결정적; W1 factor 1.35 < h 1.5; 재실행 상한·데이터 사용 규칙 누락 → caveat 2) · M1 CLOSED(`limit=0`이 `trace_loadgen.py:52`까지 전달) · M2 CLOSED(byte offset, `PYTHONUNBUFFERED=1` `ph_runner.py:59`, `idx_after`; 잔여 n1) · M3 CLOSED(boot 12, 변이 M35) · M4 CLOSED(CAP2 상수, HD floor 공변량 `floor_covariate.json`) · M5 CLOSED · M6 CLOSED(범위 축소, S1·S2만, UNSTABLE이 λ* 채택을 막지 않음 → caveat 3) · M7 부분(내용 hash 수리, `cap_trace_problems` `ph_analyze.py:134` 미호출 → n2) · m1–m9 CLOSED(m8 env 덮어쓰기 잔여 → n4) · 빠진 변이 11종 중 10 CLOSED(ENGINE_DIGEST 핀 변이 없음, TEST_MODE에서 블록 건너뜀 → n3).

## 새 문제
- n1 MINOR `ph_runner.py:123` 첫 구간 체류가 float/1초 절단 차로 최대 1 s 음수 가능(기술값, 판정 무영향)
- n2 MINOR `ph_analyze.py:31-43` `cap_addendum`이 CAP arm 간 trace 동일성 미대조(§13-1 등록과 불일치)
- n3 MINOR `ph_job.sh:51` digest 핀 변이 부재(봉인 바이트로 보호)
- n4 MINOR `ph_job.sh:35` SEALED=1에서도 env로 digest 파일 교체 가능
- n5 MINOR `ph_trace.py:122` 대 PREREG §13-1 키 표기 불일치(workload 없음, 독립성 무영향)
- n6 MINOR 실행 실패 처리 규칙 부재 — INVALID boot를 다른 Job `--resume`으로 메우는 경로 미차단(confound 11), PREREG:200 "§3-3" 포인터 오기(실제 §13-1)

## 반전 시험
NONMONOTONE 시 λ* 값·CAP 재실행 횟수/어느 Job λ*: 원자료 0 → caveat · S2 0.20, goodput 0 seed 제외, 체류 음수 오차: 판정 변화 없음.

## caveat (문자 그대로 승계, 인용 금지 형태)
1. "static D는 동거 중 목표값이며 실현 할당이 아니다. 비sticky `R2_POLICY=fixed`가 co-residence 중 D별로 다른 분할을 실제로 적용한다는 것은 Q-3 smoke 전까지 미확인이며, smoke에서 D 간 분할이 같으면 이 봉인은 무효다(CAP1 제출 전 확인)."
2. "CAP 재실행은 bracket 실패당 최대 1회다. 2회째도 실패하면 사용자 결정으로 넘긴다. λ*는 마지막 CAP Job 하나의 데이터로만 정하고 Job 간 셀을 합치지 않는다. NONMONOTONE이면 첫 실패 전 마지막 통과 rate를 쓴다."
3. "`UNSTABLE` 용량 셀에서 나온 λ*로 정한 rate의 결과는 크기를 인용하지 않는다. S3(실현 분할 기반)는 검사하지 않았다."
4. "PHASE·diag 계측 OFF 경로가 비용 항 트리(99bbef91)와 같게 동작한다는 것은 정적 확인이며 측정되지 않았다. 비용 항 수치와 이 Job 수치를 결합하지 않는다."
5. "INVALID 또는 실패한 boot는 그 Job 전체를 새 job name으로 재실행해 채운다. 다른 Job의 arm으로 메우지 않는다(같은 비교의 arm은 한 Job 안)."
6. "H-D의 H는 요청률 가중이라 W1 phase가 지배한다. λ 비 약 10:1이면 W3 비중 약 9%, W3 phase만의 여지가 PRESENT를 만들려면 약 33% 이상 [유도, §7 가정 λ 3/0.3]."
7. "h=1.5는 한 점이다. 원 KISTI HI/λ* 배율은 미상이다."
