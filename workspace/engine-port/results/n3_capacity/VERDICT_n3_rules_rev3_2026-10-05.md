# 판정서: N3 사전등록 rev3 델타 감사 (D1·D2, claims-auditor, 2026-10-05)

> 읽기 전용, GPU 0. 감사자 최종 보고를 메인 세션이 옮겼다. 성능 판정 0건, Claim 등급 변경 0건.
> 대상: `git diff 81b55c3`(rev2 커밋) — 4개 파일, +102/−10. sha 7/7 일치: PREREG 7548f5a4 · n3_analyze.py 923a240c · n3_realized_mix.py 3b6f8934 · test 4e7c62a8; 불변 n3_capacity.sh 4ffb6350 · n3_plan.py dabcdd6e · config 01675b2d.
> 감사자 재실행: `test_n3_capacity` 46/46(226 s) · analyze·mix 셀프테스트 OK · 추가 프로브(idle 스냅숏만 → `ok=False`).

## 0. 판정: **`GO-with-caveats`**, 死因 0
D1·D2 수리 확인. 새 caveat 1건(N3C-20). 텍스트 정정 2건(T1·T2, 규칙 무변 — 델타 감사 불필요).

## 1. 확인 항목
**(1) D1 — 충족, 항등식 아님, 새 상수 0.** `arm_identity(expect_decode_iterations)` + `unverifiable` 목록(스냅숏 0건 / decode 다리 요구인데 0건), `ok` = 위반 0 ∧ unverifiable 비어 있음. `cell_validity`는 mix json의 `ok`를 읽지 않고 count 필드로 직접 판정; decode 다리는 `phase_events == 1 ∧ ¬R7 표식`일 때만 요구. 변이 4종 실경로 통과(빈 파일→무효·n=3→`UNRESOLVED_N_BELOW_4_5`, 스냅숏만 ON→무효, R7 표식→유효, OFF 스냅숏만→유효). 감사자 프로브: idle 스냅숏만 → `ok=False`("idle은 근거 아님"이 코드로 성립). 오발화 위험 낮음(λ0 11셀 busy 312–2590건; B 최저 rate도 스냅숏 수십 건 예상). **fixture 변경 2곳은 검출력을 약화시키지 않음**(R3 주입은 (84,24) 덧붙임이라 여전히 검출; mix의 표적 스냅숏 prepend는 R3 구간에만 영향, `ev[-1]` 불변; idle 대조는 오히려 의미 획득).
**(2) D2 — 충족.** "다른 config(homog5 5그룹)·다른 arm(sticky ON/OFF)" 정정, A14:79–81 정황 원문 일치, O1–O4가 H1 겨냥, 노출 진술 균형("판정하지 않는다"), R6 자리표시자·결정 전 제출 금지 유지.
**(3) N3C-15…19** §11 문자 일치, §17-2 색인 diff 일치.
**(4) 수리 범위 밖 변경 — 없음**(diff 전수 대조; 모든 hunk가 D1·D2·N3C 수록·헤더·§13 수치).
**(5) mix json 이중 기록 — 혼동 경로 실재, caveat 처리.** `summarize_mix`(:484)가 mix json의 `arm_identity`(러너 호출 기준 `expect_decode_iterations=False`)를 `N3_RESULT.json` `realized_partition[cell]`에 그대로 실음 → "PHASE_EVENTS ON인데 decode_iteration 0건"인 셀이 `realized_partition.*.arm_identity.ok = true`이면서 `invalid_cells`에서는 unverifiable로 다르게 나옴. 판정 경로는 count 재계산이라 라벨 불변, 반전 경로 없음 → **N3C-20**. (선택, 하네스층) 필드명을 `arm_identity_descriptive_expect_decode_false`로 바꾸거나 `ok` 제거.

## 2. 반전 시험(rev3 신규 표면)
| 자유 표면 | 민 범위 | 판정 변화 |
|---|---|---|
| D1 근거 문턱 | 0 고정 | 자유도 없음 |
| decode 다리 요구 조건 | PHASE_EVENTS ON ∧ ¬R7(등록 고정, R7 표식은 엔진 로그) | 자유도 없음 |
| mix `ok` vs count | 두 기록 | 라벨 불변(판정은 count만) |
死因 0.

## 3. 제출 전 텍스트 정정(규칙 무변)
- **T1.** 머리말 :3 "지위: 새 사전등록 **rev2**" → rev3.
- **T2.** §2-3 "사용자가 고르면 **rev3**로 기입" → rev4(rev3는 이 회차가 사용).

## 4. 신규 caveat
- **N3C-20.** "`N3_RESULT.json`의 `realized_partition[*].arm_identity.ok`는 decode 다리를 요구하지 않는 기술 기록이다. 셀의 arm 동일성 판정은 `invalid_cells`(분석기 count 재계산)만 근거로 인용한다."
N3C-1…19, DS·LV·LVR 계열 승계.

## 5. 반증 실패
D1 실경로 작동 · OFF·R7 대조 비오발화 · fixture 검출력 보존 · D2 원문 일치. 실행·결과 감사 전 `CONFIRMED` 아님.

## 6. 남은 제출 선행조건
1. T1·T2 텍스트 정정(감사 불필요).
2. **R6 사용자 결정**(엄격 / 허용오차(ii)). (ii)면 E2 A14 (ii) 감사 선행 → rev4 기입 + 델타 감사. 엄격이면 문구만 확정.
3. **하네스층 감사** + 봉인 sha(§14 게이트 2) — N3C-18(per-boot 5분기 실행형 미검증)·N3C-20 선택 수리 포함.
4. 커밋 → `REGISTRATION_SHA256.txt` 대조 → CPU 회귀 → citation 0위반.
5. (선택) Workspace 6그룹 boot 프로브(≈0.1 GPU-h).
6. `launch.sh` dry-run 사용자 승인(shape A 별도). E2 정리 뒤 제출 권고.

## 7. 실현가능성 자기검사(게이트 #113)
T1·T2 문자 수정, N3C-20 문구뿐, 선택 수리도 판정 경로 밖. 새 상수·자유 표면 0. rev2 지적을 상수로 승격하지 않고 코드 실행·원문 대조로 독립 재검증.

## 관련 파일
`n3_capacity/PREREG_N3_CAPACITY_2026-10-04.md`(:3, §2-3 :99–108, §5-2 D1, §11, §17-2) · `n3_realized_mix.py`(`arm_identity` :374–395) · `n3_analyze.py`(`cell_validity` :63–103, `summarize_mix` :484) · `tests/test_n3_capacity.py`(`test_d1_*`) · `e2_sticky_vessl/PREREG_E2_VESSL_A14_DRAFT_2026-10-04.md`(:79–81).
