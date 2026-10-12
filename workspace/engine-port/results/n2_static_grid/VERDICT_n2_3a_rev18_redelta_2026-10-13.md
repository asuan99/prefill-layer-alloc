# 판정서 — N2(3-a) rev18 좁은 재감사: R18 수리분 + #32 EA 의미 카나리아 + 하네스 H18 수리 (2026-10-13, claims-auditor, 읽기 전용)

> 저장: 메인 세션. 판정·수리·caveat는 문자 그대로, 근거는 요약. 대상: PREREG 작업 트리 `c11baccc…`, 하네스 미커밋 diff 8파일, `n2_mutants.py` `12b2726b…`. 기준: `VERDICT_n2_3a_rev18_delta_ea_harness_2026-10-12.md`. 변이 러너 미실행.

## 0. 판정
| 대상 | 등급 | 死因 | 조건 |
|---|---|---|---|
| 규칙 봉인 갱신(rev18) | **`GO-with-caveats`** | 0 | 봉인 커밋 전 R18-5(문언만, 지정 문장 한정이면 재감사 불필요); 수리 뒤 digest 재계산; 그 바이트에서 규칙 장부 core 재실행(§3) |
| 하네스 정본 장부 진입(H01–H32, JH01–JH184) | **`GO-with-caveats`** | 0 | 규칙 봉인 커밋 뒤, H18-6(옛 PROFILE 절 보관 → UNSEALED 확인) → H18-5(무변이 반복 대조) 다음 |
| 하네스 봉인 발행(PROFILE 재발행 + EA) | **조건부 `GO-with-caveats`** | 0 | 정본 장부 기대값(H 32/32 KILLED; JH 183 KILLED + JH183 `EQUIVALENT-SURVIVED`) 일치 + 발행 뒤 봉인 상태 종단 preflight 통과 |

## 1. 직접 확인 (요약)
R18-1…4 지정 위치 반영, N2C-91…97 문자 단위 일치·각 1회; 2026-10-13 범위 밖 변경(머리말 #32, §18-2-c 폐쇄 표지, #30 추가, #31 확장, #32 구현 세부) 판정 경로 변경 0; #32 = 사용자 결정(H18-R1) 일치, 추가 세부(재읽기 9회, `t0 ≤ ts ≤ t1`)는 거짓 중단을 줄이는 방향; `warmup_ea_canary` 술어 = 셀 경로 `co_span_composition`; 하네스 192 중 8 FAIL = 봉인 상수 대조, 후보 UNSEALED 트리에서 하네스 192/192·규칙 143/143 skip 0; JH 184행 ID 중복 0·앵커 계수 1·기대 kill 183/equivalent 1, H01–H32 앵커 32/32; JH183 손 모사 22/22 OK; JH174·175·177 손 모사 → 의도한 toy 시험이 사살; "장부는 UNSEALED에서" 규칙 타당(코드 확인); 실엔진 근거: Job-PROFILE r4 ON44·OFF44·OFF16 in-flight iteration stream_idx = `n2_plan.IDX[d]` 하나뿐, 1-요청 span extend_tokens 정확, dropped_events 0/28, `--chunked-prefill-size -1 --disable-radix-cache` — ⚠ 이 확인은 블라인드 의무 위반 포함(N2C-98).

## 2. 쟁점 (요지)
① §21-33 자기검사 기록이 다시 낡음(본문은 깨끗, 기록만) → R18-5. ② 카나리아 거짓 중단 표면: {3} 도달 불가(첫 group gather 뒤 2개), co 오인 0, co·decode 혼합 0, 경계 개폐 무관, bs 무관, flush 10 s 대기, drop 시 abort(PROFILE drop 0/28), 후반 boot 발화 시 Job 종료(N2C-101); 셀 판정 영향 0. ③ H18-1·2·4·R1 처방대로. ④ 시간 민감 시험 3개가 모든 JH 행 전체 suite에 들어감 + 러너 꼬리에 사살 시험 이름 없음 → 거짓 사살 가림 가능 → H18-5·N2C-99. ⑤ 옛 PROFILE 절은 규칙 봉인 커밋 뒤 BROKEN → 보관 머리로 이동(H18-6).

## 3. 규칙 장부(core) 재실행 — 필요
이유: `n2_mutants.py` 러너가 `ffb4d6f3…`→`12b2726b…`로 바뀜. 새 러너 `--no-targeted` 행 vs rev17 장부(옛 러너) 행 비교는 항등식이 아니며 "`--no-targeted` = rev17 동작" 회귀 검사다(N2C-97은 같은 러너 안 비교만). 명령(R18-5 뒤 최종 바이트, 규칙 봉인 커밋 전): `PYTHONDONTWRITEBYTECODE=1 python3 n2_mutants.py --no-targeted --jobs 1 --set core ../n3_capacity/n3b_pdmux-n3-capacity-38ed3f4-20261006092813`. 통과: 무변이 대조 skip 정확히 1, 예상 밖 생존·TIMEOUT·BROKEN·.pyc 0, 입력 mtime < 시작; 194행이 rev17 장부와 행별 동일(J 85 = 77/8, T 12, S 97 = 96/1, J28·J29 KILLED); 규칙 시험 143/143 skip 0; 출처에 러너 sha·PREREG 최종 sha·셸 env. rev17 행별 기준은 저장소에 보존(scratchpad 임시). 차이 행이 있으면 봉인 중지·감사.

## 4. 봉인 순서(확정)
1 R18-5 → 2 최종 sha 확정·core 장부 → 3 규칙 봉인 커밋(REGISTRATION 규칙 절, `RULE_SEAL_DIGEST`, `RULE_FILES`) + 같은 커밋 H18-6 → 4 PROFILE·EA 둘 다 `UNSEALED`(BROKEN 아님) 확인 → 5 H18-5 → 6 하네스 정본 장부 `n2_mutants.py --no-targeted --jobs 1 --set harness <n3b>` → `n2_job_mutants.py --no-targeted --jobs 1`(기대 H 32/32, JH 183 KILLED + JH183 EQUIVALENT-SURVIVED, CTRL OK, 입력 = 봉인 바이트, mtime < 시작, env·러너 sha 기록) → 7 PROFILE·EA 절 동시 발행(같은 16파일) → 8 봉인 상태 종단 preflight(규칙 143 + 하네스 192 OK skip 0, `verify_harness_seal('EA') == ('SEALED', [])`, EA dry-run 단계 통과) → 9 Job-EA dry-run.

## 5. 반전 시험 표 (요지)
#32 구성 집합·경계·재읽기 수·idx 출처, 장부 저장소 상태, `_reseal`, core 재실행, 시간 민감 시험 — 반전 자유 표면 0, N1–N4 0.

## 6. 필수 수리 (문자 그대로)
- **R18-5** (a) §21-33 (ii) 결과의 "남은 1곳 … 이 판에서 고치지 않음 …" → "§18-2-c 절차 분류 문장에도 2026-10-13 폐쇄 표지를 붙였다 — 남은 곳 0"(줄 번호 없이) (b) §21-32 #32 행 "(이 판은 규칙 문언만)" → "구현 세부는 2026-10-13 §18-2-d #32에 등록" (c) §21-32에 2026-10-13 변경 색인 1행(머리말 #32 언급, §18-2-c 폐쇄 표지, #30 추가, #31 확장, #32 구현 세부 — 각각 "판정 경로 불변", 출처 이 판정서) (선택) #30 행 구분 부호.
- **H18-5** 하네스 정본 장부 직전 같은 셸 env에서 무변이 `python3 -m unittest test_n2_job n2_reach_tests test_n2_harness` 3회 → 3/3 OK skip 0, HARNESS LEDGER PROVENANCE에 기록; 1회라도 실패하면 장부 미실행·감사.
- **H18-6** 규칙 봉인 커밋에서 a5de2d6 PROFILE 절과 provenance를 `## HARNESS SEAL`로 시작하지 않는 머리로 이동(harness_digest `97334df9…`, 규칙 digest `fc9e6404…`, 16파일 sha, 커밋 `a5de2d6` 문자 그대로 보존), 이동 뒤 두 scope `UNSEALED` 확인.

## 7. Caveat (문자 그대로) — N2C-98…101은 PREREG §20-1에 등재됨(2026-10-13).

## 8. 자기 적용
R18-5 문장 세 곳, H18-5 ≈6 분, H18-6 텍스트 이동, core ≈1.5 h, 하네스 장부 H ≈25 분 + JH 184행. 러너 수정 처방 없음(재개봉 회차 방지). 死因 0, 반전 0. 감사자 블라인드 노출(N2C-98)은 감사자 자신의 행위이며 수치를 옮기지 않음. 새 성능 판정 0건.
