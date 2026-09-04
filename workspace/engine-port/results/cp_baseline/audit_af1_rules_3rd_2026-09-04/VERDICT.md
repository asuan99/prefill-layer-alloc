# VERDICT — AF-1 rev3 규칙층 감사 (2026-09-04) · cp_baseline 트랙 **9회차**

> **파일 출처 주의**: claims-auditor(적대, read-only)가 작성했으나 그 하네스가 보고서 `.md`
> 작성을 금지해 **메인 세션이 판정 본문을 축자로 옮겨 적은 것**이다. 판정·수치·변이 결과는
> 감사자의 것이고, 메인 세션은 말미의 §"메인 세션 독립 검증" 절만 추가했다.

**판정: `NO-GO`** · 방법론 게이트 #34 1단 · GPU 지출 **0** · job 제출 **0** ·
새 성능 판정 **0건** · 등급 변경 **0건** · 정책 순위 변경 **0건** · 정본 변경 **0건** ·
**저장소 파일 수정 0건**

**감사 대상**(무수정, md5 감사 전후 불변):
`PREREG_AF1_2026-09-04.md`(ae9a8549) · `af1_rule.py`(b4886287, RULE_REV=3) ·
`af1_predicates.py`(5bf81f94, PRED_REV=3) · `af1_selftest.py`(1a7577c9) ·
`af1_expected_labels.json`(be0df3cb, hand 24행 + frozen 600 + fold 75벡터)
스크래치 사본에서 `af1_selftest.py` 재현 → **rc=0, 92 검사 전부 PASS**(55.8 s).

**앞선 회차**: CP-1 rev1·rev2 · CP-0 rev1·rev3 · CP-2 rev1(H1–H8, 81%) · CP-2 rev2(F1–F8, 88%) ·
AF-1 rev1(D1–D10, 80%) · AF-1 rev2(E1–E10, 80%) — **8연속 `NO-GO`**.

---

## ★단일 판정 질문 직답

> *"rev3는 이번엔 지배하는 축에 수리를 설치했는가 — 아니면 축을 합치고(`boot`+`variance`→`coverage`) 새 가드를 둘 앞에 세우면서, 이제 그 새 앞자리 둘이 아무 수리도 받지 못한 축이 되었는가?"*

**둘 다 아니다. 정확한 답은 셋째이고, 그것이 이 판본의 성격이다.**

**(1) rev3는 처음으로 지배하는 축에 수리를 설치했고 그 설치는 실재한다.**
`neutrality_axis`는 이제 strict 스크린에서만 판정하고 `q`를 받는다. 2회차에 **둘 다 SURVIVED**였던
변이가 이번엔 **둘 다 KILLED**다(M20 banded 패스 삭제 · M21 단일 다리 축소). 회귀 회복이고, 2회차가
가장 크게 요구한 두 수리 중 하나가 진짜 이를 얻었다.

**(2) 그런데 지배하는 축에 설치한 것은 *이름*이었지 *처방*이 아니다.**
```
바닥 3 arm 전부 350.0 ms 동일, SD도 동일(1.0)   -> rev2: neutral                    rev3: neutral
바닥 3 arm 전부 350.0 ms 동일, d44의 SD만 1->40  -> rev2: FLOOR_NOT_ARM_NEUTRAL -> b
                                                   rev3: FLOOR_NEUTRALITY_BORDERLINE -> b
```
**바닥이 완전히 같은데 한 arm의 잡음만으로 경로 (a)가 죽는 경로는 그대로 살아 있다.** 라벨 이름만
바뀌었고 처방은 동일하다. `anchor`·`ladder`에서는 `borderline`이 분기를 실제로 가르는데
(`ANCHOR_REGISTERED_LADDER_BORDERLINE` → `a_with_ladder_rederivation`) 지배하는 축에서는 안 가른다.
§4.2("밴드의 부호")는 `anchor`·`ladder`만 다루고 중립성-borderline은 **다루지 않는다**.
`af1_predicates.py:37`은 여전히 현재형으로 *"neutrality is now DECISION-RELEVANT and **SD-free**"* 라
적는다 — rev3에 대해서도 거짓이다.
★자기 신고: **이 처방은 2회차의 나 자신이 준 것이다**(E2 해소 (i)). rev3는 충실히 이행했다.
불충분했던 것은 처방이다 — "축값으로 승격하라"만 적고 "그 축값이 어느 분기를 사야 하는가"를 안 적었다.

**(3) 그리고 같은 패턴이 한 칸 위로 이사했다 — 이제 수리를 못 받은 것은 축이 아니라 *검사*다.**
```
SURVIVED  MDUP    af1_predicates.py의 SECOND `stratum_axis` 정의(rev3 삼중쌍) 삭제
                  -> 파일에 그대로 남아 있던 FIRST 정의(:414, rev2 쌍-비교)가 부활   92 검사 전부 PASS
SURVIVED  MNEUTQ  neutrality_axis가 받은 q를 무시(=rev2 동작)                       92 검사 전부 PASS
```
`af1_predicates.py`에 `def stratum_axis`가 **두 번** 정의돼 있다 — `:414`(rev2판, 죽은 코드) 와
`:501`(rev3판, 삼중쌍). **E2-ii 수리 전체(`q` 인자 + 삼중쌍)에 판별 벡터가 0개이고, 되돌리는 데
필요한 것은 정의 하나 삭제다.**

**(4) 앞자리 둘은 "수리를 못 받은 축"이 아니라 둘로 갈린다.**
`coverage`는 정수 카운트라 밴드·층 처우가 원리적으로 적용되지 않는다(수리 불필요 — E3의 이 부분은
참이다). 그러나 **`screen_incomplete`는 등록 층 하나에서만 판정한다**. 그 결과 2회차 W2가 그대로다:
```
W1-A  부팅 1개 손실(6 중 5)       -> sufficient|neutral|agree|unique|pruned            ✅ 수리됨
W3    p90 셀 한 arm 결측           -> sufficient|unmeasured|...                        ✅ SCREEN_INCOMPLETE
W4    한 arm ITL 미정의            -> sufficient|unmeasured|...                        ✅ SCREEN_INCOMPLETE
W2    p99 셀 한 arm 결측(p90 정상) -> sufficient|neutral|unmeasured|unique|pruned
                                     -> IMPOSSIBLE_WORLD, fork_branch KeyError         ❌ 불변
```
2회차 E3 해소 (iii) *"`_c_one_pass`를 층별로 분리하라"* 는 **이행되지 않았고 §0.2 어디에도 기록되지
않았다**. `_c_one_pass`의 근거 *"네 판독은 한 번의 스크린에서 나온다"* 는 `stratum`에 대해 **구성상
거짓**이다 — `stratum_axis`는 정의부터 두 층을 돈다.

**(5) 그리고 1회차가 가장 크게 산 수리(D2 상 계산)가 지배하는 축에서 다시 열렸다.**
해석적 보조정리 + 40만 표적 draw + 25만 무작위 draw로 확인:
**`neutrality = borderline` ⟹ `anchor ∈ {borderline, multiple}`**(banded 집합은 strict 집합의
부분집합이므로 최악 arm의 banded 집합이 함께 줄어든다).
⇒ `sufficient|borderline|{agree,disagree}|{unique,none}|*` **14 세계**가 스크린의 상 밖인데
실질 라벨 `FLOOR_NEUTRALITY_BORDERLINE`(fork `b`)을 받는다. C11의 *"every substantive label is
reachable"* 이 그 차원에서 **선언 곱집합 열거**로 되돌아갔다 — 1회차 D2가 *"an identity"* 라 부른 것.

⇒ **답: rev3는 "지배당하는 축에만 설치했다"를 부분적으로 고쳤다 — 옳은 수리 둘을 실제로
`neutrality`에 설치했고 나는 그 설치 자체를 깨지 못했다. 그런데 그 두 수리를 지키는 벡터를 같이
설치하지 않았고, 상 계산·층 분리·판별 벡터 셋 다 여전히 `anchor`·`ladder`에만 붙어 있다.
2026-08-26 교훈 "수리는 국소, 주장은 전역"이 아홉 번째로 발화한다 — 이번 형태는 "수리는 지배하는
축에 갔는데, 그 수리를 지키는 검사가 안 따라갔다"이다.**

---

## 합격 기준 채점 (8개)

| # | 기준 | 판정 | 근거 |
|---|---|---|---|
| 1 | E1–E10이 조건 등록인가 | **FAIL (6/10 참)** | **참**: E1(M1·M2·M17 KILLED) · E2-a(M20·M21 KILLED) · E4(M11·M12 KILLED) · E5(M6·M7·M13·M15·M16 KILLED) · E7(엔진 소스 4/4 일치) · E8(복원+벡터). **거짓·부분**: E2-b(J1·J2) · E3(J4) · E6(J7) · E9(J5) · E10(인용한 `C3`가 **무력 증명서**, MARM5 SURVIVED) |
| 2 | L1–L12 응답이 참인가 | **FAIL (8/12 참)** | 참: L2·L3-a·L4-a·L6·L8·L9·L10·L12. 거짓·부분: L1(`IGNORE_EOS` 한 방향 항등 + 신규 `BOOT_SEEDS`도 동일 결함) · L3-b(죽은 선언 불변) · L4-b(`n.isupper()` 필터 탈출, M19c) · L5(§0.3 "수리" ↔ §4.4 "등가 변이", 같은 항목에 두 답) · L7(M25 SURVIVED, 3회차 불변) · L11(**스코프 오류** — `C14`가 `NO-GO` 받은 `cp2r2_predicates.py`를 사다리 정본으로 읽는 것은 doc-steward 소관이 아니라 코드 의존성) |
| 3 | 2회차 패턴 재발 | **FAIL(형태 이동)** | 단일 질문 직답 (2)–(5) |
| 4 | 상수 분류·값 치환·출처 자구 | **FAIL** | ★출처 허위 **0건 — 4판본 연속**(20+ 좌표 전수) · ★값 치환 저항 최고 수준 유지(LABEL 11개 + 코어 술어 3종 전부 KILLED). 그럼에도: DESIGN 8개 중 2개 항등 치환 · `BAND_SD_MULT = 2.0` 출처 **여전히 0건** · `312 TFLOPS` 출처 0건 · 비-대문자 상수 분류 탈출 · §5.1이 §1이 금지한 센서스 토큰 수를 인용 |
| 5 | 라벨 도달가능성 + 설계가 답을 미리 정하지 않는가 + §5.1 반증 가능성 | **FAIL(절반은 샀다)** | 상에서는 14 세계가 불가능(J3) · ★저자 자유도 없음 3판본 연속 · ★§5.1 예측 1은 **진짜로 반증 가능**하나 (i) "0.2%/2% ⇒ 독립 증거"는 **in-sample** (ii) "게이트 86의 나머지 절반" 주장 거짓(J5) (iii) prune 예측이 문서 자신의 ±20%에서 뒤집힘 (iv) ITL 예측 0건인데 예측 1이 의존 |
| 6 | S1–S5 분기 = world key, 라벨 = 산문 | **PASS(조건부)** | ★닫혔다. C8이 가드 목록에서 유도. 조건: M25 SURVIVED · S1/S2의 `n/a`가 `FORK_BRANCHES`에 없어 `fork_branch`가 **KeyError** |
| 7 | 금지 문장 15건 | **FAIL** | 2회차가 요구한 3건은 **정확한 형태로 이행**. 남은 구멍: 15건 중 1건이 **존재할 수 없는 라벨**(`VARIANCE_UNMEASURED`)을 덮고 카운트 고정이 그것을 보호(J8) · `FLOOR_NEUTRALITY_BORDERLINE` 오독 금지 0건 · ITL 다리 금지 0건(J6) · "예측이 맞았으므로 루프라인이 검증됐다" 금지 0건 |
| 8 | 자기검사(92)의 항등식·자기 수리 미검증 | **FAIL** | 변이 31종 → KILLED 24 / SURVIVED 7. 자기 수리 무검증: MDUP·MNEUTQ(E2-ii 전체) · MARM5·MSEED4 · M25 · M19c |

---

## 死因 (런킬러) — 8건

### J1. **E2의 수리는 라벨을 바꿨고 처방은 안 바꿨다** **[SHADOW of 2회차 E2 · 1회차 D1 — 세 번째 개명]**
`anchor`·`ladder`에서는 `borderline`이 분기를 가르는데 `neutrality`에서는 `borderline`과
`arm_specific`이 같은 `b`다. §4.2는 중립성-borderline을 다루지 않는다. `af1_predicates.py:37`의
*"SD-free"*(현재형, 미철회)는 rev3에 대해서도 거짓이다.
**해소(≈6줄 + 문서 4행)**: `FLOOR_NEUTRALITY_BORDERLINE`에 별도 분기(`b_more_boots`)를 주든지,
§4.2·§4.6에 잡음 경로 생존을 적고 `:37`을 철회하라.

### J2. **E2-ii 수리에 판별 벡터가 0개이고 rev2 구현이 죽은 코드로 남아 정의 하나 삭제 거리에 부활** **[SHADOW of 2회차 E2-ii]**
`def stratum_axis`가 `:414`(rev2판)·`:501`(rev3판)에 **두 번**. 등록된 `stratum_axis` 벡터 5개는
전부 `(anchor, ladder)`가 함께 갈리는 세계라 삼중쌍을 판별하지 못한다. 판별 벡터는 구성 가능하다:
`q=0.90` 전 arm `(390,60)`, `q=0.99` plain `(200,20)` / d44 `(390,60)`.
**해소(≈8줄)**: 죽은 정의 삭제 + 위 벡터 + `q`만 다른 `neutrality_axis` 벡터 쌍 + 동명 함수 중복
정의 금지 검사(`ast`).

### J3. **상 계산이 `(anchor, ladder)`에만 걸려 있어 14 세계가 스크린이 만들 수 없는데 실질 라벨을 받는다** **[SHADOW of 1회차 D2 — 지배하는 축에서 재개방]**
보조정리: strict 집합이 전부 같아야 `borderline`이 가능하고, banded 경계 ≥ strict 경계이므로 각 arm의
banded ⊆ 공통 strict `S`. 불일치 ⟹ ∀arm banded ⊊ S. `|S|=1`이면 banded=∅ ⇒ anchor strict `unique` /
banded `none` ⇒ **anchor = borderline**. ⇒ **`neutrality=borderline` ⟹ `anchor ∈ {borderline, multiple}`**.
**해소(≈15줄)**: `reachable_pairs()`를 `(neutrality, anchor, ladder)` 삼중쌍으로 확장하고 `_c_image`가
삼중쌍을 검사하게 하라.

### J4. **E3이 만들어진 이유였던 세계(층 단위 결측)가 여전히 `IMPOSSIBLE_WORLD` + KeyError** **[SHADOW of 2회차 E3 · 1회차 D8 — 세 번째]**
`coverage_axis`는 부팅을 **arm 단위**로 세고 층 단위로 세지 않는다. 한 부팅이 p90을 마치고 p99 전에
죽으면 `coverage=sufficient`인데 p99 바닥이 `None`이다. 2회차 E3 해소 (iii)는 미이행이고 §0.2에
**언급조차 없다**. 부수: §5는 분기 `n/a`를 등록하는데 `FORK_BRANCHES`에 없고 `fork_branch()`가
`COVERAGE_INSUFFICIENT`·`SCREEN_INCOMPLETE` 둘 다에 KeyError를 던진다.
**해소(≈12줄)**: `_c_one_pass` 층별 분리 · `screen_incomplete`를 "어느 층이든 미측정"으로 ·
`FORK_BRANCHES`에 `n/a` · 부팅 카운트를 `(arm, stratum)` 단위로.

### J5. **§5.1의 게이트 86 주장이 basis·척도·표본 셋 다 틀렸다** **[SHADOW of 2회차 E9의 *처방* — 수리가 死因을 만들었다]**
**(a) SD의 basis**: "1.99%/6.13%"의 출처 `RESULT_VPROBE_2026-09-01.md:37`은 **COMBINED goodput Δ의
job-내 페어 SD · 부하 하 · Zamba2-2.7B · arm 2개**다. 같은 문서 `:117`이 CI **[1.13%, 7.42%]**,
`:120`이 *"이식 주장 없음"*, `PREREG_CP2R2:56`이 **"Zamba2 V-probe 값은 이식 불가"** 를 등재한다.
★**같은 문서 §2.2는 `RESULT_VPROBE:49`의 10.4 ms를 정확히 이 이유로 거부한다** — 한 파일의 한 숫자에는
기준을 적용하고 다른 숫자에는 적용하지 않는다.
**(b) 척도와 여유**: 구속 arm(`d44` 850.5 ms)의 가장 가까운 경계는 **사다리 1000 ms 행이고 여유는
+17.6%** — 인용한 "20–100%"의 하단보다 낮다. 비교 대상은 SD가 아니라 밴드 `2·SD` = 4–12% ⇒ 비율
**1.5–4.4×**. 게이트 86이 요구한 **명시 임계 상수가 `af1_predicates.py`에 없다**.
**(c) in-sample**: MFU를 W1의 두 관측에서 역산한 뒤 그 둘을 재현한 잔차를 *"독립 증거"* 라 부르는 것은
1-모수 적합의 in-sample 잔차다. 살아남는 내용은 **두 arm 역산 MFU가 2.3% 안에서 합치**뿐.
**(d) 자기 규칙 위반**: §5.1이 인용한 층 토큰 수를 §1이 금지한다. 그리고 예측 1의 prune 목록은
**+20%에서 뒤집힌다**(4,246→5,095 tok ⇒ d44 1,021 ms ⇒ 1000 ms 행도 탈락) — 그것이 §5.1이 "실제로
사는 것"으로 지목한 산출물이다.
**해소**: 식별력 임계를 상수로 등록 · SD 미측정 명시 · "독립 증거" 정정 · ±20% 감도를 두 번째
예측 라벨로 등록.

### J6. **ITL 다리는 처치를 볼 수 없고 그것은 GPU 0에서 엔진 소스로 유도된다** **[NEW 대상]**
```
pdmux_context.py:124-127  SM_COUNTS = [(108,0)] + divisions + [(0,108)]
multiplexing_mixin.py:952-953  elif not self.running_batch.is_empty():
                                   set_current_stream_idx(self.real_sm_group_num - 1)   # = (0,108)
pdmux_d44.yml  sm_group_num 3 · manual_divisions [[64,44,0]]  => [(108,0),(64,44),(0,108)]
```
동시성 1의 decode 구간에는 경쟁 prefill이 없으므로 위 분기가 **결정적으로** index 2 = **(0,108)** 을
고른다. ⇒ **`d44`의 무경쟁 ITL 바닥은 108 SM에서 측정되고, 연구 대상인 SM 분할은 ITL 다리에 작용하지
않는다.** §9-1은 이것을 경험적 질문으로 적고 완화가 **그 완화를 유도한 코드 경로의 재확인**일 뿐이다.
**해소**: §9-1을 유도된 사실로 재작성(코드 좌표 3개) · ITL 다리를 *"arm-독립 절대 도달가능성 스크린"*
으로 재등록 · ITL 바닥 예측치 등록 · §8에 금지문 1건.

### J7. **"어느 검사가 강제하는가" 칸이 무력 증명서를 강제 근거로 인용하고 자기 규칙을 어긴다** **[SHADOW of 2회차 E6 · 게이트 88]**
```
SURVIVED  MARM5   ARM_ORDER_BY_JOB을 rev2의 5 순열로 되돌림     92 검사 전부 PASS
SURVIVED  MSEED4  BOOT_SEEDS를 CP-2의 4 seed로 되돌림           92 검사 전부 PASS
```
§0.2 E10 행·§0.1 seed 행이 강제 근거로 인용한 `C3`는 *"DESIGN constant moves **no** label"* —
**비강제 인증서**다. 게다가 `DESIGN_SUBST`의 `BOOT_SEEDS` 둘째 값과 `IGNORE_EOS` 둘째 값이
**등록값과 동일**해 한 방향이 항등식이다. §0.1이 스스로 선언한 규칙("칸이 비면 '등록'이라 쓸 수
없다")을 §0.2의 E9·E10 행이 어긴다.
**해소**: 칸 값을 열거형 3개로 제한 · `C3` 인용 전부 후자로 이동 · 항등 치환 2건 교체.

### J8. **rev3가 삭제한 축과 라벨이 원장에 없고, 카운트 고정이 존재할 수 없는 라벨의 금지문을 보호한다** **[SHADOW of 2회차 E8 · 게이트 88]**
`VARIANCE_UNMEASURED`는 rev3에 존재하지 않는데 §8의 15개 금지문 중 하나가 그것을 덮고,
`meta.prereg_counts.prohibitions = 15`가 그것을 지킨다. `af1_predicates.py:261`은 없는 축을 서술한다
(*"The `variance` axis turns that None into a label."*).
**해소**: **삭제 원장** 신설(축 `variance` · 라벨 `VARIANCE_UNMEASURED` · 처방 E3-(iii)) + 행 수 고정 ·
공허한 금지문 정정 · `:37`·`:261` 수정.

---

## 국소 지적 (N1–N12)

| # | 내용 | 계보 |
|---|---|---|
| **N1** | `af1_predicates.py`에 `def stratum_axis` 두 번(:414 죽은 코드 / :501) | NEW(J2 기반) |
| **N2** | 규칙 정본 파일 안 유령·거짓 서술 2건: `:37` "SD-free" · `:261` "`variance` 축" | SHADOW of 2회차 E2·E8 |
| **N3** | `R.PREDICATE_CONSTANTS`·`R.LABEL_CONSTANTS = ()` 죽은 선언, 검사 0건 | SHADOW of 1회차 L1 / 2회차 L3 |
| **N4** | `BAND_SD_MULT = 2.0` · `312 TFLOPS` 출처 0건 | SHADOW of 2회차 criterion 3-c |
| **N5** | `DESIGN_SUBST`의 `IGNORE_EOS`·`BOOT_SEEDS` 항등 치환 | SHADOW of 2회차 L1 |
| **N6** | C1이 `n.isupper()`라 비-대문자 상수는 분류 탈출(M19c) | SHADOW of 2회차 L4 |
| **N7** | `_section`이 헤더 부재 시 `ValueError`로 죽는다 — 차단되나 **명명된 실패 아님** | NEW |
| **N8** | §3.3 "≤0.6 GPU-hr" ↔ §7 "≤0.5" 불일치 | NEW |
| **N9** | 워밍업 arm `plain` 고정 ⇒ `plain`이 1위인 2 job은 동일 설정 부팅 직후에 측정 | NEW(E10의 그늘) |
| **N10** | §5.1 예측 1이 ITL 6열 통과를 암묵 가정하는데 ITL 예측치 0건 | NEW(J6 부수) |
| **N11** | "선언 축 수 = 등록 축 술어 수" 검사 없음(2회차 교훈 4 처방 미이행) | SHADOW of 2회차 교훈 4 |
| **N12** | §0.3 L5("수리") ↔ §4.4("등가 변이") 같은 항목에 두 답 | NEW |

---

## 반증 실패 (S1–S14)

- **S1 ★출처 허위 4판본 연속 0건** — 20+ 좌표 전수 대조 일치(survey 표 6행 · retune plan
  `:77`/`:110`/`:113` · CP-2 rev1 5좌표 · `cp2r2_predicates:47-48` · `server_args:6131` ·
  `correctness_gate:24`/`:65-67` · `multiplexing_mixin:446` · `dual_worker:612`(정확히 1회) ·
  `pdmux_d44.yml` · `RESULT_CORRECTNESS_GATE §2` · **`model.safetensors.index.json` total_size ÷2 =
  8.888e9** · **`sacct` job 902407 = 00:39:53**).
- **S2 ★E1(분기 포함 짝 앵커)은 완전히 실재한다** — M1·M2·M17 전부 KILLED. *"처방을 바꾸려면
  사전등록을 고쳐야 한다"* 가 이제 **참**이다.
- **S3 ★E2의 절반(strict-only)은 실재하고 판별 벡터가 있다** — M20·M21 둘 다 KILLED(2회차엔 둘 다
  SURVIVED). **회귀가 회복된 유일한 항목.**
- **S4 ★E4(survey 좌표)는 닫혔다** — M11·M12 KILLED. C14가 표 셀을 파싱해 좌표 **쌍 집합**을 비교한다.
- **S5 ★E5(삭제 채널)는 닫혔다** — M6·M7·M13·M15·M16 전부 KILLED. `prereg_counts` 역방향 대조와
  `nrows == len(rows)` **등식**이 작동한다.
- **S6 ★코어 스크린은 값·구조 양쪽으로 고정** — MALLARM(∀arm→단일 arm)·MTHRESH(`>`→`>=`)·MSTRATQ
  전부 KILLED. LABEL 상수 11개 값 치환도 전부 KILLED.
- **S7 ★메인 세션이 신고한 검사기 결함 3건은 전부 실재했고 전부 고쳐졌다** — 되돌리면 전부 KILLED:
  `_section` 시작점(원장 카운트 12/10/12 → **0**) · 앵커 정규식(24 → **28**, 표 헤더 4행) ·
  분기 문자클래스(24 → **17**, `n/a`×6 + `cp2_regime_void`×1).
- **S8 ★§4.4의 "등가 변이" 주장은 참이다** — ∀arm 스크린의 통과 집합은 직사각이라 `ok = a·b`이고
  `35 = 5×7`은 만들 수 없다 ⇒ `ok >= total-1` 완화는 **함수적으로 동일**. M18 SURVIVED는 결함이 아니다.
- **S9 ★tie-break 부재 3판본 연속 구조적으로 참** — `anchor_axis`에 정렬·비교·선호 코드 0줄.
- **S10 ★L2·L3-a·L4-a·L6·L8 전부 진짜 이를 얻었다** — M22·M23·M24·M19·M26·M4 전부 KILLED.
  C10c가 반례 세계 `sufficient|borderline|disagree|unique|intact`를 지목한다.
- **S11 인용정지 위반 0건** — 497 added lines, 0 violation / 16 rules OK.
- **S12 ★E7은 진짜 정정이다** — 필드명·`dual_worker:612` 유일 위치·`plain`/`cp2048`의 원리적 부재
  전부 확인(정본 `CONSENSUS §1-1`에 축자로 존재).
- **S13 ★§7 예산 환산은 재현된다** — 902407 = 39:53, 6 부팅 ⇒ 6.65분/부팅; AF-1 4부팅 ⇒ 26.6분 +
  생성 ≈ 28분 × 6 = **2.8 GPU-hr**.
- **S14 ★2회차의 나 자신도 재검증했다 — 내 처방 두 개가 새 死因을 낳았다**(J1·J5). rev3의 잘못이
  아니라 내 처방의 잘못이므로 이번엔 **코드 층까지** 적었다. 루프라인 자체는 재계산으로 재확인
  (108 SM 100% MFU 241.9 ms · 64 SM 408.2 ms · 역산 MFU 0.4900/0.4790) — 결론은 유지하되
  *"독립 증거"* 라는 **내 표현이 과했다**.

---

## 그림자 비율 — **死因 7/8 (88%)**

| 死因 | 근원 | 기전 |
|---|---|---|
| J1 | 2회차 E2 · 1회차 D1 | σ가 세 번째로 **개명** — 라벨만 바뀌고 처방 불변 |
| J2 | 2회차 E2-ii | 수리를 지배하는 축에 설치했으나 **지키는 벡터가 안 따라갔다** |
| J3 | 1회차 D2 | 상 계산이 새 축을 안 덮어 **선언 곱집합 = 항등식** 부활 |
| J4 | 2회차 E3 · 1회차 D8 | 처방 (iii) 미이행 + 무기록 |
| J5 | 2회차 E9의 *처방* | ★**수리가 死因을 만들었다** |
| J6 | (NEW 대상) | *"GPU 0에서 유도되지 않는다"* 의 두 번째 자리 |
| J7 | 2회차 E6 · 게이트 88 | 강제 칸이 **무력 증명서**를 인용 |
| J8 | 2회차 E8 · 게이트 88 | 카운트 고정이 **유령 금지문**을 보호 |

**판정: 2회차 최소 경로 11개 중 여섯을 실제로 이행했고 그 여섯 전부에서 검사가 진짜 이를 얻었다 —
나는 그 여섯을 깨지 못했다. 짝 앵커·survey 좌표·삭제 채널·코어 스크린은 이제 견고하다. 지배하는
축에 옳은 수리를 설치한 것도 이번이 처음이다. 실패한 것은 그 수리를 지키는 장치다.**

---

## AF-1 rev4 최소 경로 (전부 GPU 0)

1. ★★★ **J3(≈15줄)** — `reachable_pairs()`를 삼중쌍으로. J1의 처방이 이 상 위에서 정의되므로 먼저.
2. ★★★ **J2(≈8줄)** — 죽은 정의 삭제 + 판별 벡터 2개 + 동명 함수 중복 금지.
3. ★★★ **J4(≈12줄)** — `_c_one_pass` 층별 분리 · `FORK_BRANCHES`에 `n/a` · 부팅 카운트 `(arm, stratum)`.
4. ★★★ **J1(≈6줄 + 문서 4행)** — 별도 분기 또는 잡음 경로 생존 명기 + `:37` 철회.
5. ★★ **J5(≈30분)** — 식별력 임계 상수 · SD 미측정 명시 · "독립 증거" 정정 · ±20% 감도 등록.
6. ★★ **J6** — §9-1을 유도된 사실로 · ITL 다리 역할 재등록 · §8 금지문.
7. ★★ **J7** — 강제 칸 열거형 3값 · `C3` 인용 제거 · 항등 치환 2건 교체.
8. ★ **J8** — 삭제 원장 신설 · 공허한 금지문 정정 · `:37`·`:261` 수정.
9. ★ **L11 재분류** — `C14`가 `NO-GO` 판본의 술어 파일을 사다리 정본으로 읽는 문제.
10. **N1–N12** — 각 1–5줄.

> **이 프로브의 질문은 여전히 살아 있고 이번 판본에서 실제로 전진했다.** 짝 앵커는 분기를 묶고,
> 삭제 채널은 닫혔고, 코어 스크린은 값·구조 양쪽으로 고정됐고, 루프라인은 처음으로 **반증 가능한
> 예측**이 됐다. **사지 말아야 할 것은 지키는 장치가 없는 수리 둘과, 게이트 86을 채웠다는 주장과,
> 이미 GPU 0에서 답이 나와 있는 ITL 다리다** — 넷 다 GPU 0에서 고쳐진다.

---

## 신규 방법론 교훈 후보

1. ★★★★ **"수리를 지배하는 축에 설치했는가"의 다음 질문은 "그 수리를 검사하는 벡터도 그 축에
   설치했는가"다.** 축을 만들거나 함수에 인자를 추가하는 수리에는 **그 인자만 다른 판별 벡터**를
   같은 커밋에서 요구하라.
2. ★★★★ **직전 판본의 구현을 파일에 남긴 채 새 정의를 아래에 추가하면 수리는 "정의 하나 삭제"
   거리에 있다.** 동명 함수 중복 정의를 자기검사로 금지하라. 게이트 88의 거울상 — 여기서는
   **없어진 것이 아니라 안 없어진 것**이 위험하다.
3. ★★★★ **"어느 검사가 강제하는가" 칸에 *무력 증명서*를 적을 수 있다.** 칸 값을 열거형으로 제한하고
   "이 검사는 이 상수가 라벨을 **움직이지 않음**을 보인다"를 별도 값으로 두어라.
4. ★★★★ **처방을 준 감사자는 다음 회차에 자기 처방부터 재검증하라.** 2회차 E2-(i)·E9-(i)는 충실히
   이행됐고 **둘 다 새 死因을 낳았다**. 처방은 "무엇을 등록하라"만이 아니라 **"그 등록이 어느 결정을
   바꿔야 하는가"** 까지 적어야 한다.
5. ★★★ **다른 캠페인의 SD를 노이즈 척도로 빌리는 것은 basis 검증 대상이다 — 특히 그 출처가
   "이식 불가"를 스스로 등재했을 때.**
6. ★★★ **1-모수 적합의 in-sample 잔차를 "독립 증거"라 부르지 마라.** 살아남는 내용은 **여러 조건의
   역산값이 서로 합치하는가**뿐이다.
7. ★★★ **삭제 원장을 "死因"·"국소 지적"과 나란히 세 번째 원장으로 두어라.**
8. ★★ **카운트 고정은 공허한 항목을 보호한다.** 개수에 더해 **각 항목이 실재하는 라벨/축을
   지시하는지**를 검사하라.

---

## 재실행·신규 실행한 검사 (전부 GPU 0)

```
[1]  af1_selftest.py (스크래치 사본)          -> SELFTEST OK, 92 검사 전부 PASS (55.8 s)
[2]  외부 변이 31종                            -> KILLED 24 / SURVIVED 7
       KILLED  : M1 M2 M17 M20 M21 M11 M12 M6 M7 M13 M15 M16 M22 M23 M24 M19 M26 M4
                 MALLARM MTHRESH MSTRATQ R1 R2 R3
       SURVIVED: MDUP MNEUTQ MARM5 MSEED4 M25 M19c M18(등가 변이 증명됨)
[3]  af1_rule.label() 직접 열거 (600 세계)     -> frozen 600 일치, S1–S5 라벨 산문 일치
[4]  ★도달 가능 IMPOSSIBLE 세계 탐색           -> W1-A/W3/W4 수리 확인, **W2 불변**(KeyError)
[5]  ★중립성 축 SD-의존성 프로브 재실행         -> 바닥 동일·d44 SD만 1->40 => FLOOR_NEUTRALITY_BORDERLINE -> b
[6]  ★상 독립 열거(해석 + 40만 표적 + 25만 무작위) -> neutrality=borderline ⟹ anchor ∈ {borderline, multiple}
                                                  ⇒ 14 세계가 상 밖인데 실질 라벨
[7]  ★루프라인 재계산 + 여유 전수               -> 108SM/1.00 241.9ms · 64SM/1.00 408.2ms
                                                  역산 MFU cp 0.4900 / d44 0.4790 (합치 2.3%)
                                                  최소 여유 = 사다리 1000ms 행 **+17.6%**
                                                  토큰 ×1.2에서 prune 목록 뒤집힘
[8]  ★엔진 소스 직접 대조(ITL 다리)             -> SM_COUNTS=[(108,0),(64,44),(0,108)] · 무경쟁 decode = index 2
[9]  ★원장 34행 코드 검증                       -> 인용 좌표 전수 실재, 강제 칸 2건 거짓
[10] ★DESIGN_SUBST 항등 치환 정적 검사          -> IGNORE_EOS·BOOT_SEEDS 각 1개 항등
[11] grep 정본·출처 (20+ 좌표)                  -> 전부 실재. VARIANCE_UNMEASURED = 유령 금지문
[12] model.safetensors.index.json total_size    -> 17,776,454,656 ÷2 = 8.888e9 (정확 일치)
[13] sacct -j 902407                            -> Elapsed 00:39:53
[14] check_citation_stops.py --file x4          -> 497 added lines, 0 violation / 16 rules OK
[15] check_version_sweep.py                     -> rc=1 (§6.1의 "항등식" 주장 확인)
[16] 원본 md5 5파일                              -> 감사 전후 불변
     ae9a8549 / b4886287 / 5bf81f94 / 1a7577c9 / be0df3cb
```

**돌리지 않은 것(의도)**: `presubmit.py`(게이트 89) · `check_line_citations.py`(항목81) ·
`design_reachability.py`(격자 밖 문제라 답할 수 없다 — 대신 [6]) · `check_doc_facts.py`.

---

## 감사자 자기 신고

- 변이 31종은 전부 스크래치 사본 위에서 돌았고 대상 5파일 md5는 감사 전후 동일하다.
- **`presubmit.py`는 돌리지 않았다**(게이트 89).
- **J5·J6의 산술은 측정이 아니다.** MFU 역산에 쓴 W1 수치는 정본이 arm 비교·순위 인용을 금지한
  표에서 왔다 — 순위가 아니라 **물리 처리량 역산**에만 썼고, ★**rev3가 그 두 숫자를 사전등록 본문의
  예측 근거로 승격시킨 것 자체가 스코프 확대**임을 J5에 적었다.
- **J3의 불가능 세계 주장은 해석적 증명 + 두 독립 표본기로 확인했다.** 무작위 draw의 "never
  produced" 목록 대부분은 **표본 희소성**이므로 死因 근거로 쓰지 않았다.
- **2회차의 나 자신을 재검증했고 내 처방 두 개가 새 死因을 낳았음을 확인했다**(S14).
  2회차가 *"완전히 닫혔다"* 고 기록한 항목 중 **되열린 것은 없다**.
- **J6은 "무가치하다"는 주장이 아니다** — 처방은 다리를 없애는 것이 아니라 **그 다리가 무엇을
  사는지 다시 등록하는 것**이다.
- **성능 판정 0건 · 등급 변경 0건 · 정책 순위 변경 0건.** 불변 배너 전부 유지.

---

## 메인 세션 독립 검증 (2026-09-04, 감사자와 별개로 실행)

死因 네 건이 이 판정의 근간이므로 메인 세션이 직접 재현했다. **넷 다 확인된다.**

- **J2 확인**: `grep -n "^def stratum_axis" af1_predicates.py` → **`414` 와 `501` 두 줄**.
  rev2→rev3 패치가 `neutrality_axis`부터 상 계산 직전까지를 통째로 치환했는데, 구 `stratum_axis`는
  그 구간보다 **앞**에 있어 살아남았다. 메인 세션이 만든 결함이다.
- **J4 확인**: `fork_branch("COVERAGE_INSUFFICIENT")`·`fork_branch("SCREEN_INCOMPLETE")` **둘 다
  `KeyError`**. `FORK_BRANCHES`에 `n/a`가 없다. W2 세계
  `sufficient|neutral|unmeasured|unique|pruned` → **`IMPOSSIBLE_WORLD`**.
- **J3 확인**: 독립 20만 무작위 draw(arm별 바닥·SD 무작위) — `neutrality == "borderline"`인 모든
  draw에서 관측된 `anchor` 값은 **`{borderline}` 하나뿐**. 그리고
  `sufficient|borderline|agree|unique|intact`·`sufficient|borderline|agree|none|empty` 두 세계가
  **실질 라벨을 받는다**(상 밖인데 라벨이 붙는다).
- **J6 확인**: `multiplex/pdmux_context.py:124-127`이 `SM_COUNTS = [(108,0)] + divisions + [(0,108)]`,
  `multiplex/multiplexing_mixin.py:952-953`이 `elif not self.running_batch.is_empty():
  set_current_stream_idx(self.real_sm_group_num - 1)` — 자구 확인. `sm_group_num 3`이므로 index 2 =
  **(0, 108)**. ⇒ 경쟁 prefill이 없는 decode 구간은 **108 SM**에서 돈다. **AF-1의 ITL 다리는 SM
  분할을 볼 수 없고, 그것은 GPU 0에서 유도된다.**
