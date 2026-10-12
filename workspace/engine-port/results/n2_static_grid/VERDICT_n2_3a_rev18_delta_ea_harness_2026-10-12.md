# 판정서 — N2(3-a) rev18 규칙 재개봉 2회차 델타 감사 + Job-EA 하네스층 감사 (2026-10-12, claims-auditor, 읽기 전용)

> 저장: 메인 세션(2026-10-12). 감사 보고의 판정·수리·caveat는 문자 그대로, 근거 서술은 요약.
> 대상: PREREG rev18(미커밋, sha `eb5ed803…`), 하네스 `b1de10c`, `n2_mutants.py`(`ffb4d6f3…`→`12b2726b…`). 기준: 규칙 봉인 `5e29990`(`fc9e6404…`), 하네스 PROFILE `a5de2d6`(`97334df9…`). 변이 러너 미실행.

## 0. 판정
| 대상 | 등급 | 死因 | 조건 |
|---|---|---|---|
| 규칙 봉인 갱신(rev18) | **`GO-with-caveats`** | 0 | 봉인 커밋 전 R18-1…R18-4(지정 문장 한정이면 재감사 불필요); 수리 뒤 바이트로 digest 재계산 |
| 하네스 봉인 scope EA | **보류** | 0 | H18-1…H18-4 → 규칙 봉인 갱신 → 최종 바이트 정본 장부 뒤 발행 |
| 하네스 봉인 scope PROFILE 재발행 | **`GO-with-caveats`(조건부)** | 0 | EA 절과 같은 장부·같은 바이트에서만, N2C-91 승계 |
현재 바이트 규칙 digest 후보 `b8908f2b…`는 R18 수리 뒤 무효(재계산).

## 1. 감사자 직접 확인
불변 5파일 바이트 일치(바뀐 규칙 파일은 PREREG·n2_mutants.py뿐, 핵심 변이 194종 불변); 규칙 시험 143/143 skip 0; 하네스 175 중 8 FAIL은 전부 봉인 상수 대조(후보 상수 임시 트리에서 174/175, 남은 1개는 PROFILE 절 재발행으로 해소); **`b1de10c` 분석기로 Job-PROFILE a5de2d6 재분석 → `N2_PROFILE_LABELS.txt` 바이트 동일, RESULT 최상위 키 차이 0**(창 단위만, 동거 통계 미계산); 추적 파일 51.

## 2. 쟁점 요지
① 겨냥 러너: rev17과 다를 수 있는 방향은 SURVIVED→KILLED(거짓 사살) 한쪽뿐, `PLAUSIBLE(조건부)`. 규칙 시험에 모듈 전역 변경이 있어 독립성 가정은 공허하지 않음. **규칙 장부 2회 비교는 기대상 항등식(교훈 9)** — N2C-90이 실제로 걸리는 곳은 하네스 장부(특히 JH98–141). `--jobs N`은 행별 대조 없음 + 가짜 엔진 벽시계 의존 → 거짓 사살 경로. 러너 기계장치를 묶는 변이 없음(교훈 53).
② F-N2-1/2/3 시험·변이 없음, F-N2-9는 빈 입력만; F-N2-1 모집단에 C `RECORDER_UNPROVEN` 포함.
③ #25 기술 ρ 무제한 외삽: 라벨 반전 경로 없음, F-N2-9만 영향.
④ #24 boot 간 ρ_off 외삽: 외삽 조건 lag_off − lag_on > W_co,on + 16 ≥ 261, 레버리지 ≤ 0.32((44,8,2048))…≤ 2.07((16,32,8192)); 반전 못 만듦; 외삽 1건이면 엄격 라벨은 항상 `UNRESOLVED_N_BELOW_4`.
⑤ ŝ=0 E-α 셀: A1은 잘리지 않음 `CONFIRMED`(설계 산술); ON A2는 lag + W_co ≤ 2881 = L_lag + N_co,max − 57일 때만 보장 → n 감소만(반전 아님).
⑥ 봉인 순서 성립. PROFILE 재발행 사유는 규칙 digest 변경 + 하네스 바이트 변경(`b1de10c`); 두 절은 같은 16파일 바이트를 나열(2단 봉인은 형식); 재발행 절은 Job-PROFILE 출처 아님. 순서: R18 → 규칙 봉인 커밋(n2_plan 상수 갱신) → H18 → 최종 바이트에서 H·JH 정본 장부 1회 → 그 장부로 PROFILE·EA 절 함께 → EA dry-run.
⑦ #22 코드 = 문언(라벨 불변, 단 측정 실패 부류 `CO_WINDOW_SHORT`가 술어 불성립으로 이동, details가 원상태 지움); #21 fail-closed 타당; #28 기계적 조건만(디렉터리 출처 미검사).
⑧ D6: start/settle 예외 → VD, 라벨 영향 0; 체계적 boot 실패도 F-N2-7E HELD로 갈 수 있음(예보 반전 경로).
⑨ §21-33 낡은 수치 재발 2건: §18-2-c `:1052` "318 중 8", §8-6 `:416`·`:469` 개정 창 열린 서술.
추가: Job-EA 첫 boot fail-closed는 부분적 — 배선·P3·CG_BS·CG 카나리아는 boot마다, **E-α 의미 경로(co span 식별·Σn_new=K·coload_sent·동거 iteration stream_idx)는 첫 boot에서 검사 안 됨**(체계적 결함 시 16 boot 전부 C WINDOW_SHORT, 3–4 h 뒤 분석에서 드러남). 블라인드는 과속방지턱 수준(봉인 파일 임의 생성, 원자료 로컬 존재, `blind_exception=True` Python 호출). EA 흐름 Job 레이아웃 경로(교훈 270) 통과.

## 3. 반전 시험 표 (요지)
§10-2 외삽·#25·#22·#21·#26·#27·#28·#29·#30·동결·ŝ=0 — 반전을 만든 자유 표면 0, N1–N4 해당 0. 변이 러너: SURVIVED→KILLED 방향만, 하네스 장부는 미실행이라 계산 불가.

## 4. 필수 수리·권고 (문자 그대로)
- **R18-1** §18-2-c `:1052` "318 중 8" → 숫자 없는 포인터; §8-6 `:416`·`:469`에 "(rev18: 개정 창 폐쇄 — 이 경로는 사용자 결정 없이는 열리지 않는다)"; §21-33 해당 두 주장 정정, 자기검사를 "문단 단위 숫자 전수 + 개정 관련 어간(개정·제출 전) 전수"로.
- **R18-2** §18-2 하네스 봉인 행: PROFILE 재발행 사유 = 규칙 digest 변경 + 하네스 바이트 변경(`b1de10c`), 두 절은 같은 바이트, 재발행 절은 Job-PROFILE 출처 아님(N2C-91).
- **R18-3** §17-24·§18-2-d #30·§21-32: (i) 하네스 장부(H01–H32, JH 전부) 정본은 `--no-targeted --jobs 1`, 겨냥은 선택 비교 (ii) 모든 봉인 장부 `--jobs 1` (iii) 규칙 장부 2회 비교는 N2C-90 검증 아님(N2C-97).
- **R18-4** §17-26 "감사에 넘김"을 본 판정서 §2 ⑤로 닫음; §20-1에 N2C-91…97 등재.
- **H18-1** F-N2-1·2·3·9 모든 분기 합성 시험 + JH 변이(경계: F-N2-3 h=6→HELD, (5,1)→UNRESOLVED, (5,0)→FALSIFIED; F-N2-2 "둘 다 유효한 round만" 삭제 변이; F-N2-9 halves 차 ≥ ln 1.03 → FALSIFIED, < → HELD).
- **H18-2** 러너 기계장치 JH 변이: (a) `controls.passes(targets)` 삭제 (b) 겨냥 시간 초과를 사살 처리 (c) 겨냥 생존 → 전체 재판정 생략 (d) `jobs>1` 동시 무변이 대조 생략 — 기존 toy 시험으로 사살.
- **H18-3** 최종 바이트에서 H·JH 정본 장부(`--no-targeted --jobs 1`), 출처에 러너 sha(`n2_mutants.py`)·셸 env 기록(≈2.5 h CPU).
- **H18-4** `render_text_ea`에 ρ_off·ρ_on 각각의 `aon_containment_rate`·`label_if_containment_rejects` + 필수 병기 줄에 N2C-87·N2C-89.
- 권고(사용자 결정, 死因 아님): **H18-R1** EA 의미 카나리아(boot warm-up telemetry: co span 짝·`extend_tokens == n_new × 8192`·Σn_new = 3[{1,2}·{1,1,1} 모두 허용], `prefill_in_flight ∧ stream_idx == idx(d)` decode_iteration ≥ 1, 실패 시 `ABORT_EA_CANARY`; 미채택 시 "E-α 경로 첫 실엔진 확인은 Job 뒤 분석" caveat). **H18-R2** `verify_ea_result_seal`이 `N2_EA_RESULT.json` sha를 fetch된 `vessl/<job>/SHA256SUMS`와 대조, 가짜 엔진 산출 거부. **H18-R3** #22 덮어쓰기 전 C 상태를 details에 보존.

## 5. Caveat (문자 그대로)
- **N2C-91** "rev18 규칙 봉인 위에서 재발행된 하네스 봉인 scope PROFILE 절은 Job-EA의 선행 조건일 뿐이며 EA 절과 같은 바이트를 나열한다. Job-PROFILE `pdmux-n2-static-grid-a5de2d6-20261012021243`를 낸 하네스 바이트(harness_digest `97334df9…`, 규칙 digest `fc9e6404…`)를 증명하지 않는다. 감사가 `b1de10c` 분석기로 그 원자료를 재분석해 `N2_PROFILE_RESULT.json`·`N2_PROFILE_LABELS.txt`가 바이트 동일함을 확인했으나, 그것은 그 한 Job·그 분석 경로의 회귀 부재일 뿐이다. 재발행 절이 Job-PROFILE을 봉인했다고 쓰지 않는다."
- **N2C-92** "Job-PROFILE에서 관측된 첫 요청 단독 span 산포(ON44·OFF44 (8, 2048) s ≈ 118–122)는 같은 c_in을 쓰는 E-α (44, 8, 2048)의 등록 설계 보장(s ≤ ŝ + 64 = 64)도 넘는다. 감사 산술로 A1은 이 기전으로 잘리지 않지만, ON A2는 실제 lag + W_co ≤ 2881 = L_lag + N_co,max − 57일 때만 설계상 보장된다. 그 셀의 ON A2 `WINDOW_SHORT`를 무작위 결측이나 D44의 성질로 쓰지 않는다. 이 산술은 PROFILE 수준값을 옮긴 감사 예보이며(N2C-13) 수치로 인용하지 않는다."
- **N2C-93** "F-N2-9는 A_on을 허용 폭 검사 없이 외삽한 앞뒤 절반 ρ로 계산된다. FALSIFIED를 인용할 때 해당 짝의 front/back `aon_contained` 표시를 병기한다. 외삽된 절반이 낀 짝만으로 FALSIFIED가 된 경우 '경과 시간 표류'로 쓰지 않는다. F-N2-1의 모집단은 C가 `RECORDER_UNPROVEN`인 측정 성립 복제를 포함하므로, 그 복제가 결과를 정한 경우 그 사실을 병기한다."
- **N2C-94** "co span 구성 실패로 C가 `WINDOW_SHORT`가 된 복제는 원래 `CO_WINDOW_SHORT`(EAK-3 검열)였을 수 있다. 검열 수를 인용할 때 구성 실패 복제 수를 따로 병기하고, '큰 감속의 검열은 0건'으로 쓰지 않는다."
- **N2C-95** "Job-EA 결과 봉인 파일(`EA_RESULT_SEAL.json`)은 두 결과 파일의 sha256만 묶으며, 그것이 Job 안 분석 산출인지 검사하지 않는다. Job-PROFILE 원자료는 로컬에서 읽을 수 있다. 분석기의 `BlindedScope` 거부를 블라인드가 집행됐다는 증거로 인용하지 않는다(N2C-73과 같은 취지)."
- **N2C-96** "D6 이후 start·settle 실패는 셀 0개 boot에서도 VD boot 중단으로 분류되고(보수 규약), Job은 다음 boot로 진행한다. 그 결측을 값 의존 결측의 증거로 쓰지 않는다. boot 중단이 1건 이상인 Job의 F-N2-7E(·7P) `HELD`를 '등록 예산 모형 성립'으로 인용할 때는 중단 boot 수를 병기한다."
- **N2C-97** "rev18 바이트에서 규칙 봉인 장부의 `--no-targeted`와 겨냥 실행이 행별로 같다는 기록은 기대상 항등이다(rev17 kill 예상 생존 0, 피시험 바이트 불변). 겨냥 러너의 장부 의미 보존(N2C-90) 근거로 인용하지 않는다. 하네스 봉인 장부는 `--no-targeted --jobs 1` 실행만 정본이다."

## 6. 자기 적용
R18 문언만, H18-1·2·4 기존 fixture에 붙는 CPU 시험·변이, H18-3 ≈2.5 h CPU, H18-R1은 구성 {1,2}·{1,1,1} 모두 허용으로 거짓 중단 표면 방지. 이전 처방 상수 승격 안 함(교훈 276). 반전 자유 표면 0, N1–N4 0 → NO-GO 근거 없음. 새 성능 판정 0건.
