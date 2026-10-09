> 결론(한 줄): 규칙 봉인 불가 — 기준 (c) 봉인 파일 집합 정의 결함(R15-1)만. 死因 0, 판정 경로 구멍 0.

# 판정서 — N2(3-a) 사전등록 rev15 좁은 델타 감사 (규칙 봉인 가부) (claims-auditor, 2026-10-09)

> 감사자는 읽기 전용이라 메인 세션이 감사 보고 본문을 그대로 저장했다. GPU 0, 새 성능 판정 0건, Claim 등급 변경 0건.
>
> **대상**: 커밋 `6bd8ea3`. 그 뒤 HEAD `677111a`는 power_cap만 바꿨고, 이 디렉터리의 작업 트리는 `6bd8ea3`과 바이트가 같다.
> - 파일 sha 앞자리: PREREG `328c1962…`, `n2_feasibility.py` `1ee3338d…`, `n2_probe_reduce.py` `f768936a…`, `n2_mutants.py` `a8da9d49…`, `N2_REGISTERED.json` `e7d743c8…`, `PROBE_PLAN.json` `a9fbac5c…`
> - 대조본: `99c2578`
> - 입력 판정서: `VERDICT_n2_3a_rev14_delta_2026-10-09.md`
>
> **수행 내용**(범위를 일부러 좁혔다)
> - 시험: `test_n2_feasibility` + `test_n2_probe_reduce`를 1회 돌렸다. 133/133 통과, skip 0, N3 양성 대조 포함.
> - `--verify-registered` 실행, `derive_fixed_k()` 재계산, 등록 파일을 rev14 판과 필드 단위로 대조했다.
> - 감사 변이 5종(A1–A5)을 돌렸다. 대상은 `n2_probe_reduce.py`의 판정 경로이고, 스크래치 사본에서 `test_n2_probe_reduce` 단일 파일만 썼다. 무변이 대조는 OK였고, 대조에서도 N3는 skip했다.
> - 등록 변이 213종은 재실행하지 않았다. §15·§21-26의 기록을 신뢰했다(지시 범위).
> - 저장소 파일 변경은 0이다. `__pycache__`는 감사 전부터 있었고(10:35·10:44), 감사 실행은 `PYTHONDONTWRITEBYTECODE=1`로 했다.

## 등급: `GO-with-caveats` — 死因 0. 단, 규칙 봉인은 R15-1 수리 뒤에 한다

- **N1(estimand 부재)·N3(예측 도달 불가)·N4(제1원칙 위반) 해당 없음.** rev14 판정이 그대로 유지된다. 이번 델타는 문언, 창 구현 범위, 시험, 변이만 바꿨다.
- **N2(반전) 없음.** rev15가 새로 만든 자유 표면 7개를 등록 범위 끝까지 밀었는데, 예보 범위 안에서 라벨·부호가 바뀐 경우는 0이다(§3 표).
- **봉인을 막는 것은 (c) 하나다(R15-1).** 사용자 기준 (c)인 "봉인 대상 파일 정의의 결함"에 해당한다. NO-GO 사유(N1–N4)는 아니다. 처방은 문언 수정뿐이다.

## 1. 항목별 확인 표

| # | 확인 대상 | 결과 | 근거 |
|---|---|---|---|
| 1 | R14-1 낡은 문언 13곳 | **13/13 해소** | 아래 1-1 |
| 1a | §4-2-5 E-α 취소 208로 단일화 | CONFIRMED | 줄 157 "16 + 128 + 64 = 208"; §8-4 δ 64; §14-2 모형 "+ 208"; §8-6 trace "+144 + 64". 현행 절의 "152"는 폐기 고지 1건뿐 |
| 2 | R14-2 필터 전 iter_seq 연속 | **문언·코드 일치** | §7-1 (R)·§7-2 (R)·§9-1 의사코드(`iters_all.has_gap(window.seq_min, seq_max)`, W_co 범위)와 `_iter_seq_gap(iters, lo, hi)`(필터 전 전체 열)·`cell_windows` 호출부(W·A2: `alone_status`, C: `wco` 범위)가 일치한다. 시험 `test_recorder_continuity_is_pre_filter`는 OFF 누수 → VALID, W_co 공백 → RECORDER, 단독 띠 공백 → W RECORDER를 확인한다. 감사 변이 A1(단독 창 검사를 필터 후로 되돌림)은 사살됐다 |
| 3 | R14-8 일반화 `cell_windows` | **조건부 확인** | 아래 1-2. 창별 π·창별 상태·§9-1 C 우선순위·PROFILE·OFF E-α 경로는 합성 시험이 실제로 묶는다. 단 §7-1 (b) PE-time, §7-1 (c), §7-2 (b) S-per 가중은 구현되지 않았고, §17-21의 미구현 목록에도 없다 → H15-1, N2C-74. 단독 PE 문턱 0.99는 경계가 묶이지 않았다(A5 생존) → H15-2 |
| 4 | R14-3 블라인드 규칙 | **집행은 부분적. 판정 경로 아님** | 아래 1-3 → N2C-73 |
| 5 | R14-6 두 단계 봉인의 파일 집합 | **결함 — 봉인 차단(c)** | §2 R15-1 |
| 6 | R14-4 | CONFIRMED | N2C-14 원문이 rules 판정서와 바이트 동일. N2C-64…72 9건 모두 rev14 판정서 문안이 PREREG에 문자 그대로 포함됨(스크립트 대조) |
| 6 | R14-5 | CONFIRMED | `N2_REGISTERED.json`에 `theta_rho_estimator` 필드 추가(r4 중앙값 5,705.4, 357/30/0, 재현 명령). 머리말·§17-14·§8-6 문언 일치 |
| 6 | R14-7 | CONFIRMED | §16-1 D-민감도 문안 틀(줄 872), §16-3 27a, F-N2-D1·D2 중립 명칭("b-증가"·"c-기울기")과 N2C-70 병기, §9-6 "같은 프롬프트"를 ŝ = 0 칸으로 한정, §12-7 round 간 sd |
| 6 | R14-9 | CONFIRMED(기록 신뢰) | Q05(`test_kappa_parts_uses_value_window_not_all`), Q06·Q07(`test_registered_rc_and_match_are_live`), Q09(`test_derive_fixed_k_follows_its_inputs`) 시험이 있다. 러너는 `CONTROL-FAILED`면 `SystemExit`(n2_mutants 줄 321–322), `N2_N3_SH`는 env → git toplevel → HERE 순으로 찾는다 |
| 6 | H14-2 | CONFIRMED(계산기 쪽) | `verify_registered`가 재생성 바이트와 digest를 비교한다. M13·M14 시험이 있다. 러너 쪽 `ABORT_REG_SOURCE` 구현은 하네스 몫 |
| 6 | H14-3 | CONFIRMED(등록) | G-3가 두 단계(a·b)로 등록됐다: seed 레이블, 길이, span 1+2, sticky 조건 그대로, `warmup_start/end`. smoke 고아 규칙(`n2_probe_reduce.py:647`)과의 충돌은 공시만 됐다 → R15-1 (iii) |
| 6 | H14-4 | CONFIRMED | `detail` = w_co·c·세 손실. 시험 `test_co_window_short_detail`(J19) |
| 6 | H14-5 | CONFIRMED | §15 reach R4-3: Δℓ 60 성립, 70 `WINDOW_SHORT`, δ 8 변이본은 실패해야 함 |
| 7 | K (33, 15) | CONFIRMED | `derive_fixed_k()` == `K_CO` == {16: 33, 44: 15} |
| 7 | 예산 | CONFIRMED | `N2_REGISTERED.json`과 rev14 판의 차이는 `K_derivation`(설명 문자열 1개 추가)과 `calculator_sha256`뿐이다. `budget`·`derived`·`kappa_star`·`limits`는 같다. PROFILE (b) 22,204/33,938/36,700, EA (b) 9,814/15,218/16,100 |
| 7 | `--verify-registered` | CONFIRMED | `{"ok": true}` rc 0(digest = 파일 sha 지정). 이 실행은 자기 대조라 진정성 증거는 아니다(N2C-69 그대로) |

### 1-1. R14-1 13곳 개별 대조

| 위치 | 현행 문언 | 판정 |
|---|---|---|
| §4-2-5 | 208, 152·Δℓ 폐기 고지 | 해소 |
| §14-2 모형 문단 | "E-α (b) = 트리거까지 + L_lag + 실제 W_co + 208", "프로브는 없다" | 해소 |
| §8-3 (2곳) | "rev14 — 프로브 없음; N2C-66", "보정 프로브는 없다" | 해소 |
| §8-5 REAL_CTX | "등록 설계 포락선 그대로(보정 없음)" | 해소 |
| §8-6 trace | rev5 예시 고지 배너, "δ는 고정 허용치" | 해소 |
| §8-6 말미 | "프로브는 재검사의 사유가 아니다" | 해소 |
| §14-1 | "아무도 재지 않는다(rev14)" | 해소 |
| §14-2 하드캡·§18-1 | "K 변경으로 재승인(E14-1; 프로브는 사유가 아니다)" | 해소 |
| §15 R4-3 | δ 64 기준 재정의 | 해소 |
| §17-2 | "10× 정지 문턱은 기술값" | 해소 |
| §19-7·8 | 완화 근거·선택적 smoke | 해소 |
| §16-3 #22 | "rev14는 실제로 프로브 없이 봉인하므로 그대로 적용" | 해소 |
| G-3 | (a)/(b) 두 단계 정합 | 해소 |

**새로 찾은 낡은 문언 2곳**(R14-1 목록 밖, §6-1 R5-7·창 필터 문단 — 봉인 기준 밖, R15-2):
- (i) "`CO_WINDOW_SHORT` 검사(len(W_co) < 128)는 이 집합(필터 전 W_co)에 적용된다". 그런데 §7-2 (a)·§9-1·코드는 **필터 뒤 C 값 창 < 128**이다. 규칙 문장끼리 충돌한다.
- (ii) "구현은 축약기 공유 함수 `extract_cell_windows`이고 `n2_window`가 import한다". rev15에서는 `cell_windows`이고, `extract_cell_windows`는 ON smoke 래퍼다.

§21-27의 정규식 자기검사는 이 두 곳을 잡지 못했다. 교훈 89의 역형이 재발한 것이다.

### 1-2. R14-8 — 합성 시험이 실제로 묶는 경로(코드 독해와 변이)

**묶이는 것**
- OFF E-α: π_alone (0,108), π_coloc (64,44)일 때 전부 VALID(J14). `kappa_parts` 값도 고정돼 있다. ON 래퍼는 OFF 셀을 `ALONE_PE_LOW`로 거부한다.
- PROFILE: ON·OFF 모두 co-load 없이 W만 낸다(J15).
- 필터 전 연속(J16·A1).
- C 우선순위: B_FINISHED가 CO_WINDOW_SHORT보다 먼저(J17).
- `COLOC_UNREALIZED` 0.95 경계 200/211 대 200/210(J18). 분모를 in-flight로 바꾼 변이(A2)도 사살됐다.
- 손실 분해(J19).
- C ≠ C_all(Q05).
- WINDOW_SHORT와 CTX_UNREALIZED 분기, 단독 D_UNREALIZED.
- 비-co span 검사(A3 사살 — 단 smoke 경로 시험으로만 잡혔다).
- A_on을 ctx_mid(C_all)로 계산하는 변이(A4 사살).

**§9-1과의 차이(코드 독해)**
- C 상태 순서가 "no co span → RECORDER"다. no co span이면 W_co가 빈 집합이라 동치다.
- `JOINED`(서명)가 없다. §17-21이 미구현으로 공시했다.

**묶이지 않거나 구현되지 않은 것**
- (a) §7-1 (b) **PE-time**(S-per 가중 ≥ 0.99)과 §7-2 (b) **S-per 가중 ≥ 0.95**. `cell_windows`는 iteration 비율만 계산한다.
- (b) §7-1 (c) 창 안 prefill 부재. PROFILE과 A2에서는 구성상 성립한다(W가 모든 burst 뒤에 온다). 그러나 ON E-α A1에서 트리거 뒤 비-co span이 있으면 A1이 VALID로 남을 수 있다. 그러면 그 ON 복제의 A_on이 ρ_off 짝에 쓰인다(fail-open). 예보 포락선(ŝ + 64) 밖에서만 발화한다.
- (c) 단독 PE 문턱의 경계. 감사 변이 **A5(0.99 → 0.95)가 생존**했다. 시험은 0.906에서만 본다.
- §15 `n2_window` 행은 "n2_window가 덧붙이는 것은 셀 자르기·Δdropped·R7·CTX_MISMATCH·사유 결합뿐"이라고 적고, 같은 줄 시험 열에 "창 안 prefill 1건 → D_UNREALIZED"를 요구한다. 그러면 (a)·(b)를 구현할 자리가 규칙 봉인 파일(`cell_windows`) 수정 말고는 없다. 이것이 R15-1 (iii)의 근거 중 하나다.

### 1-3. R14-3 블라인드의 집행 가능성

- **(가) 비데이터 사유만**과 **(다) 개정 전 값 병기**는 델타 감사가 문서 수준에서 집행할 수 있다.
- **(나) 블라인드**는 분석자의 자기 기록과 산출물 검사뿐이다. 기록되지 않은 열람·계산은 검출할 수 없다.
- 등록 산출물 자체도 블라인드 문언과 긴장한다. §7-3이 요구하는 "셀 구간·boot 전체" 실현 분할 4추정량(E-time·E-iter, 쌍별)은 PROFILE ON boot의 burst 단계, 곧 idx(d) 동거를 포함한다. 그러므로 쌍별 E-time/E-iter에서 동거가 섞인 평균 iteration 시간을 얻을 수 있다(약한 파일럿).
- **판정 경로가 아닌 이유(반전 시험)**: 블라인드를 완전히 깨고 PROFILE 동거 자료를 보더라도, 개정이 바꿀 수 있는 것은 K·C 절단·κ\* 요구뿐이다.
  - K는 검열 경계 κ\*(최악값 3.07/3.21)와 A2 위치에만 닿는다. 예보 κ는 [1.3, 1.6]이다.
  - C 절단은 C의 구성을 K와 무관하게 만든다.
  - C 128 ↔ 전체의 차이는 |ln| ≤ 0.46%로 3% 문턱의 1/6 이하다(rev14 §8, N3 r3·r4 체인 19개).
  - ⇒ 예보 범위 안에서 라벨·부호는 바뀌지 않는다. 남는 것은 서사("K pre-data")의 진정성뿐이다 → N2C-73.

## 2. 봉인 전 필수 수리 (기준 (a)(b)(c) 해당분만)

**R15-1 (기준 (c) — 봉인 대상 파일 정의 결함)**

§18-2의 규칙 봉인 집합은 다음과 같다: PREREG, 계산기, 축약기, `N2_REGISTERED.json`, 시험 2개, `n2_mutants.py`, config·`analyze.py`·`profile.py` digest. 하네스 봉인 집합은 다음과 같다: 러너, plan, window, signature, realized_mix, analyze, reach_tests, "Job용 `n2_marks`·`n2_loadgen` 확장". 결함은 셋이다.

- **(i) 봉인 파일이 봉인 밖 파일에 의존한다.**
  - 규칙 봉인 대상인 `n2_mutants.py`의 무변이 대조와 하네스 변이 32종은 `test_n2_harness.py`, `n2_harness_mutants.py`, `n2_marks.py`, `n2_loadgen.py`, `n2_probe_run.py`, `fake_n2_engine.py`를 실행한다(`n2_mutants.py:25-30`, "the control runs all three"). N3 양성 대조는 `n3_capacity.sh`(`N2_N3_SH`)와 N3 r3 원자료 디렉터리에 의존한다.
  - 이 파일들은 규칙 봉인에 없다. 그중 `n2_marks`·`n2_loadgen`은 하네스 단계에서 "확장"하도록 예정돼 있다.
  - ⇒ 하네스 구현이 진행되면, 규칙 봉인된 변이 장부(213종, 예상 밖 생존 0)는 봉인 파일만으로 재현되지 않는다. 대조가 깨지거나 H 변이의 앵커가 사라진다.
  - §21-26의 "시험 58·75·35 = 168"도 규칙 봉인 시험 133개와 어긋난다.
- **(ii) 어느 봉인에도 속하지 않는 파일이 있다.** `n2_probe_run.py`, `fake_n2_engine.py`, `test_n2_harness.py`, `n2_harness_mutants.py`, `n2_probe_workspace.sh`, `PROBE_PLAN.json`이다. 이 가운데 `n2_marks`·`n2_loadgen`에는 §14-2가 "등록 상수"라 부르는 G-1…G-16(프롬프트 생성기, seed 유도, G-13 트리거·취소 분위)이 산다.
- **(iii) 규칙 봉인 뒤 규칙 파일을 고치는 절차가 없다.** 그런데 그런 수정이 이미 예고돼 있다.
  - H14-3: smoke 고아 규칙의 warm-up 예외. 수정 대상은 `n2_probe_reduce.py:647`, 규칙 봉인 파일이다.
  - 1-2 (a)·(b): §15가 "두 번째 구현을 두지 않는다"고 했으므로 `cell_windows`를 고쳐야 한다.
  - §18-2에 있는 수정 절차는 Job-EA 개정 창과 post-smoke 두 가지뿐이다.

**처방**(문언만 고치면 되고, 시험·변이 재실행은 필요 없다. 실현 가능성: 파일 목록과 절차 문단 추가뿐이다)
- (1) 디렉터리의 모든 파일을 {규칙 봉인 · 하네스 봉인 · smoke 전용 · 이력} 중 **정확히 하나**로 분류하는 표를 §18-2에 둔다. 외부 의존(`n3_capacity.sh`, N3 r3 원자료 경로)은 digest나 경로로 기록한다.
- (2) `n2_mutants.py`의 하네스 행(H01–H32)과 그 의존 파일을 규칙 봉인에 넣거나, 둘 중 하나를 택한다:
  - (a) rev13 하네스 6개 파일을 규칙 봉인에 넣는다.
  - (b) 규칙 봉인 시점의 변이 장부를 "핵심 181 = 규칙 봉인, 하네스 32 = 하네스 봉인에서 재실행"으로 나누고, 시험 수 문언을 133(규칙)/35(하네스)로 정정한다.
- (3) "규칙 봉인 뒤·하네스 봉인 전에 규칙 봉인 파일을 수정해야 하면 §21 변경 색인 + 좁은 델타 감사 + 규칙 봉인 digest 갱신. 하네스 봉인은 자신이 빌드된 규칙 봉인 digest를 기록한다"를 등록한다. 예고된 수정 두 건(H14-3 smoke 예외, `cell_windows` PE-time·§7-1 (c))을 이 절차의 첫 대상으로 명시한다. 또는 (권고) 이 두 건을 지금 반영한 뒤 봉인해, 봉인 직후 개봉을 피한다.

(a) 死因: 0. (b) 판정 경로 구멍(예보 안에서 라벨·부호 변화): 0.

## 3. 반전 시험 표 (rev15에서 새로 생긴 자유 표면)

원자료와 추정량은 rev14 판정서 §8과 같다(N3 r3 `38ed3f4`·r4 `93c3cca`; 설계 재집계, 인용 금지). 이번에 새 원자료 계산은 하지 않았다. 감사 변이는 등록 축약기의 합성 시험 위에서 돌렸다.

| 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 |
|---|---|---|---|
| 블라인드 범위("동거 통계"의 외연) | 완전 위반(PROFILE 동거 자료를 κ로 분석) | 없음. 개정 가능 대상은 K·C 절단·κ\*뿐 | κ\*_worst 3.07/3.21 ≫ 예보 1.6; C 128 ↔ 전체 \|ln\| ≤ 0.46% |
| (가) "비데이터 사유 등"의 열린 목록 | 임의 사유 | 없음(위와 같은 이유) | 같음 |
| 필터 전 연속의 검사 범위(W: 필터된 창의 첫~끝 iter_seq) | 창 밖 구간 공백 미검사 | n만 바뀐다. S-per 쌍은 창 안 연속 쌍뿐이라 값은 불변 | 코드 `s_per_intervals` |
| 단독 PE 문턱 0.99(경계 미결속, A5) | 0.95–1.0 | 없음 | rev14 반전 표 "0.99(0.95–1.0) 판정 변화 없음"; N3 체인 비-in-flight·쌍 불일치 0 |
| PE-time·S-per 가중 0.95 누락 | 누락 ↔ 적용 | 예보 안에서는 없음(누수 0이면 PE-iter = PE-time = 1) | N3 r3·r4 co 체인 69개, 누수 0 |
| §6-1 "len(W_co) < 128" ↔ §9-1 "len(C) < 128" | 두 정의 | 예보 안에서는 없음(필터 손실 0이면 동치) | 같음 |
| §7-1 (c) ON A1 미검사(비-co span 뒤) | 발화 ↔ 미발화 | 예보 밖에서만 발화(ŝ + 64 포락선 위반 필요). 발화해도 A1 값은 단독 iteration이다 | §6-2 X 규칙 |

반전 계산에는 등록 계산기 상수, 등록 축약기, rev14 판정서의 재집계 수치만 썼다. 새 상수는 만들지 않았다.

## 4. 하네스 단계 수리 목록 (봉인을 막지 않음)

- **H15-1**: §7-1 (b) PE-time, §7-1 (c) 창 안 prefill 부재(특히 ON E-α A1 + 비-co span), §7-2 (b) S-per 가중 0.95를 공유 창 구현에 넣는다. R15-1 (3)의 절차를 거치거나 봉인 전에 반영한다. 반영 전에는 N2C-74.
- **H15-2**: 단독 PE 문턱 경계 시험을 추가한다(0.99 바로 아래/위; A5 사살). 같은 형식으로 OFF E-α의 A1 `CTX_UNREALIZED`, A2 RECORDER·D_UNREALIZED 경로도 묶는다.
- **H15-3**: ρ_off 조립을 시험으로 묶는다: C_off 중앙값 ÷ ON 복제 A_on(ctx_mid(C_off)), 다른 boot 사이의 포함 가드, `CTX_MISMATCH`. "A1 < C < A2 구성상 포함" 논증은 같은 boot 안에서만 성립한다. ρ_off는 다른 boot의 ctx_mid를 쓰므로 `AON_CONTAINMENT` 거부율을 결측 히스토그램에 넣는다.
- **H15-4**: `n2_realized_mix`의 PROFILE 셀·boot 단위 출력이 블라인드(§5-4 (나))와 충돌하지 않게 범위를 정한다. 쌍별 E-time/E-iter 비를 내지 않거나, 블라인드 예외로 명시하고 N2C-73을 병기한다.
- **H15-5**: H14-2의 러너 쪽 `ABORT_REG_SOURCE`(digest 상수 보유)를 구현하고 변이로 묶는다.
- **H15-6**: R15-2 — §6-1의 낡은 문언 2곳(1-1 끝)을 고친다. 문언이 규칙끼리 충돌하므로 봉인 편집 때 함께 고치는 것을 권고한다.
- rev14 판정서의 H14-1(하네스 전체 미구현), E14-1(하드캡 16,100 재승인), E14-2·E14-3은 그대로 열려 있다.

## 5. 신규 caveat (문자 그대로 승계)

- **N2C-73**: "Job-EA 개정 창의 블라인드(§5-4 (ii)(나))는 분석자의 자기 기록과 산출물 검사로만 확인된다 — 기록되지 않은 열람·계산은 검출할 수 없고, §7-3의 셀·boot 단위 실현 분할 4추정량(burst 단계 포함)은 등록 산출물 자체에 동거 구간 정보를 담는다. 블라인드 검사 통과를 '아무도 Job-PROFILE 동거 자료를 보지 않았다'의 증거로 인용하지 않으며, 'K는 데이터를 보기 전에 정한 상수'는 개정이 없었을 때만 쓴다."
- **N2C-74**: "규칙 봉인 시점의 공유 창 구현 `cell_windows`는 §7-1 (b)의 PE-iter만 계산하고 PE-time(S-per 가중)·§7-1 (c) 창 안 prefill 부재·§7-2 (b)의 S-per 가중 0.95는 계산하지 않는다. 이를 보완한 판이 감사·봉인되기 전에는 창 상태를 '§7 술어 전부로 판정됐다'고 쓰지 않으며, 합성 시험(`TestGeneralizedWindows`) 통과를 실제 엔진 이벤트열에서 술어가 성립한다는 증거로 인용하지 않는다."

## 6. 감사 변이 결과 (`test_n2_probe_reduce` 단일 파일, 스크래치 사본, 무변이 대조 OK 75 tests / skip 1)

| ID | 변이 | 결과 |
|---|---|---|
| A1 | 단독 창 기록기 검사를 필터 뒤 창으로 | 사살(`test_alone_windows_exclude_in_flight_and_trim` ERROR — 예외로 잡힌 것) |
| A2 | PE_coloc 분모를 in-flight만으로 | 사살(2) |
| A3 | 트리거 뒤 비-co span 검사 제거 | 사살(4 — smoke 경로 시험만) |
| A4 | A_on을 ctx_mid(C_all)에서 보간 | 사살(1, `test_kappa_parts_pinned`) |
| **A5** | `cell_windows` 단독 PE 문턱 0.99 → 0.95 | **생존** → H15-2 (반전 없음, §3) |

---

## 이 보고서의 경로

- 대상 디렉터리: `/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/n2_static_grid/`
- 사전등록: `.../PREREG_N2_3A_2026-10-07.md`
  - 봉인 집합: §18-2(줄 957–970)
  - 블라인드: §5-4(줄 215)
  - 낡은 문언: §6-1(줄 225·227)
  - `n2_window` 행: §15(줄 842)
  - 고정 범위 공시: §17-21
- `.../n2_probe_reduce.py`: `cell_windows` 줄 257–351, smoke 고아 규칙 줄 642–649
- `.../n2_mutants.py`: 하네스 의존 줄 25–30, 대조 중단 줄 321–322
- 선행 판정서: `.../VERDICT_n2_3a_rev14_delta_2026-10-09.md`
- 감사 변이 스크립트(저장소 밖): `/tmp/claude-1000/-home-wonho-Experiments-KISTI/2657f214-f740-4503-8f81-241f78b1e2b2/scratchpad/amut/run.py`
- 시험 출력(저장소 밖): `/tmp/claude-1000/-home-wonho-Experiments-KISTI/2657f214-f740-4503-8f81-241f78b1e2b2/scratchpad/tests.txt`