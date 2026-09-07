# AF-1 규칙층 감사 6회차 — **확인 패스** (감사 아님)

2026-09-06 · claims-auditor(read-only) · **GPU 0** · 성능 판정 **0건** · arm 순위 **0건** ·
등급 변경 **0건** · 정본 변경 **0건** · 정책 순위 변경 **0건**

**판정: `GO`.**

5회차 판정서(`audit_af1_rules_5th_2026-09-04/VERDICT.md`)가 스스로 지정한 범위 —
*"아래 넷을 이행한 rev6는 `GO`다. 다음 감사자는 이 넷만 확인하고 새 스윕을 열지 마라"* — 를
지켰다. 목록 밖 발견은 §"등재 권고"에만 적었고 **판정에 반영하지 않았다.** 5회차가 승계한
예외 둘(**(a)** 결정 규칙[라벨·분기·상·가드]을 건드리는 결함 · **(b)** 새로 등록된 예측이
GPU 0에서 반증) 은 **둘 다 미발화**이며, 그 판정 근거를 아래 §4에 적는다.

**대상 md5(감사 전후 불변, 5파일 전부 사전신고와 일치)**

| 파일 | md5 | 신고 |
|---|---|---|
| `PREREG_AF1_2026-09-04.md` | `174c6a10a1e1d48409901d64f6b4d9dc` | `174c6a10` ✅ |
| `af1_rule.py` (`RULE_REV = 6`) | `4c851dc8166a4e4c6e6e7caa7d44874c` | `4c851dc8` ✅ |
| `af1_predicates.py` (`PRED_REV = 6`) | `a4c5e2106458c3266b415b8159f7d67e` | `a4c5e210` ✅ |
| `af1_selftest.py` | `9e44d0870d634adfbffd7a999172d66c` | `9e44d087` ✅ |
| `af1_expected_labels.json` | `7953a3258a02280bb2a92c5e58ab03c2` | `7953a325` ✅ |

★`presubmit.py`는 실행하지 않았다(게이트 89 — 그 도구는 read-only가 아니다). 저장소
파일은 하나도 만들지 않았고 하나도 고치지 않았다. 변이는 전부 스크래치 격리 사본
(`.../scratchpad/mrun*/w/e/r/cp_baseline/`, 깊이를 맞춰 `reports/serving_slo_survey.md`도 함께
복제)에서 돌렸다. **대조군 먼저.**

---

## 0. 대조군

| 실행 | rc | 단언 |
|---|---|---|
| 정본 위치 `python3 af1_selftest.py` | **0** | **130 PASS · 0 FAIL** |
| 격리 사본 무변이 미러 | **0** | 130 PASS |
| 하네스 내부 대조군 `MNONE` | **0** | `SURVIVED` ✔기대 |

⇒ 경로 아티팩트 없음. 사전신고 *"rc=0, 130 PASS"* 는 정확하다.

---

## 1. 확인할 넷 — **4/4 O**

### 1-1. `PREREG:597`(rev6에서 **:613–617**) — 분기 `b_precision_bound` · 처방 자구 일치 → **O**

```
613: - ❌ *"`FLOOR_NEUTRALITY_BORDERLINE`은 arm들이 실제로 다르다는 뜻이다"* — 그 라벨이 말하는 것은
614:   **잡음이 판정을 정했다**이고 분기는 `b_precision_bound` — *"이 측정 정밀도에서 답이 안 나온
615:   것"* 이다. ★**그 처방은 "부팅을 더 사라"가 아니다**(§5·§6:S4와 자구 일치. `boot_sd`는 표본
616:   SD라 밴드가 `2σ`로 수렴하고 §3.3이 그 구매를 금지한다). `FLOOR_NOT_ARM_NEUTRAL`과 같은 것으로
617:   읽지 마라.
```

자구 일치를 **두 앵커에 대해 대조**했다(축자 grep):

- **§6:S4 앵커**(`PREREG:532`, C7이 축자로 고정하는 짝 앵커):
  *"그 처방은 **부팅을 더 사라가 아니다**(`boot_sd`는 표본 SD라 밴드가 2σ로 수렴한다 — 死因 K3)"*
  → §8:615와 인용부호 유무를 제외하고 **동일 문자열**.
- **§5**(`PREREG:420–423`): *"이 측정 정밀도에서 답이 안 나온 것"* 은 §8:614–615와 **완전 축자 일치**,
  *"`boot_sd`는 표본 SD … §3.3은 그 구매를 금지한다"* 도 압축형이나 같은 자구.

rev5의 문장(*"분기는 `b_more_boots`(부팅을 더 사라)다"*)은 삭제됐다(diff 확인). **한 라벨에 두
분기가 등록된 상태는 해소됐다.**

### 1-2. `PREREG:522`(rev6에서 **:539**) + `af1_predicates.py` `neutrality_axis` docstring → **O**

- `PREREG:538–540`: *"S4 = 중립성 borderline(★자기 분기 **`b_precision_bound`** 를 갖는다 —
  rev4가 이 자리에 붙였던 이름 `b_more_boots`는 5회차 死因 Q1로 **철회됐고 §0.4가 그 삭제를
  센다**…)"* — **rev4 이력으로 명시 + 철회 명시 + 원장 참조.**
- `af1_predicates.py:513–523`(`neutrality_axis`): 현재형 처방을 제거하고
  *"★It is NOT 'buy more boots'. rev4 named that branch `b_more_boots` and 4th-audit K3 showed
  the name was unpurchasable … (5th-audit Q1: this docstring was one of three places where the
  deleted name kept giving the old prescription.)"* 로 갱신.

**잔존 `b_more_boots` 전수 확인**(`grep -n`, 3파일 12개소): `PREREG` 8개소(§0.1 Q1 행 :54 ·
§0.2 K3 행 :64 · §0.2 J1 행 :76 · §0.4 :117 · §0.4 :121 · §5 :421 · §6 :539) ·
`af1_predicates.py` 1개소(:518, K3 인용) · `af1_rule.py` 3개소(:24·:31 = rev5→rev6 변경 이력,
:44 = rev4 K3, :66 = rev3 J1 이력). **전부 원장/이력 문맥이고 현재형 처방 0건.**

### 1-3. `af1_selftest.py` C20 정규식 확장 + **판별식** → **O**

```python
for m in re.finditer(r"`([A-Za-z][A-Za-z0-9_]{2,})`", _section(raw2, sect)):
    t = m.group(1)
    if "_" in t or t[0].isupper():
        toks.add(t)
```

★**5회차가 지정한 판별식을 그대로 집행했다.** 확장된 C20을 **수리 전 문서**(= `HEAD`의 rev5
판, `git show HEAD:…/PREREG_AF1_2026-09-04.md`, md5 `a6e70421`)에 적용:

| 실행 | ghosts |
|---|---|
| 확장 C20 + rev6 `known` → **rev5 문서** | `['b_more_boots']` — **정확히 하나** |
| 확장 C20 + rev5 `known`(SURVEY_POINTS·ARMS·extids 없이) → **rev5 문서** | `['b_more_boots']` — **정확히 하나** |
| 확장 C20 + rev6 `known` → **rev6 문서** | `[]` |

⇒ `af1_rule.py:29–31`이 새로 등록한 문장 *"before it, the widened C20 flags exactly
`b_more_boots` and nothing else"* 는 **GPU 0에서 참**이다. 변이로도 확인:
`MGHOSTBRANCHPROHIB` **KILLED**(`C20 … -- ['b_more_boots']`), `MREVERT597` **KILLED**(동일 검사).
수리 후 `rc=0` / 130 PASS. **판별식 양방향 충족.**

### 1-4. `PREREG:608`(rev6에서 **:627**) — `60.5%` → `53.6%` → **O**

```
625: - ❌ *"`FLOOR_NOT_ARM_NEUTRAL`이 나왔으므로 이 체제에 외부 앵커는 불가능하다"* — …
627:   대로 **파티션 없는 arm이 MFU 53.6%를 넘는가** 하나로 정해진다(rev4의 60.5%는 틀린 층에서
628:   나온 값이고 §5.1 예측 3이 이미 철회했다 — 5회차 등재 권고 1).
```

문서 전체에서 `60.5` 잔존은 **2개소뿐**(:485 §5.1 예측 3 · :627 §8)이고 **둘 다 rev4의 철회된
값이라고 명시**한다. §5.1 예측 3의 문턱 `53.6%` 와 §8이 **일치**한다(4회차 K1이 §9-0에서 지목한
동일 형태의 내부 모순 해소).

---

## 2. 메인 세션 자기신고 재검증 — **3/3 실재**

이 트랙의 규율(*"자기신고는 재검증한다"*)에 따라 신고된 셋을 **직접 다시 돌렸다.**

| 변이 | 무엇을 되돌리나 | 신고 | **본 감사 실측** | 죽인 검사 |
|---|---|---|---|---|
| `MC20NARROW` | C20 정규식을 대문자 전용으로 재축소 | KILLED | **KILLED** rc=1 | `C20b the token scan actually reaches lowercase identifiers` |
| `MC20FILTER` | 필터 `"_" in t or t[0].isupper()` → `t[0].isupper()` | KILLED | **KILLED** rc=1 | `C20b` (동일) |
| `MEXTIDGHOST` | `meta.external_identifiers`에 유령 항목 주입 | KILLED | **KILLED** rc=1 | `C20c every registered external identifier is actually used -- ['zzz_ghost_identifier']` |

★`C20b`·`C20c`가 **실재하는 신규 검사**임을 확인했다(`af1_selftest.py:599–609`, 문서화는
:63–66). `external_identifiers` = **15항**, 미사용 **0항**(전수 확인). traceback 0건 —
셋 다 **크래시가 아니라 명명된 단언 실패**로 죽었다(3회차 N7 규율 유지).

---

## 3. 회귀 확인 — 5회차 KILLED 7종

| 변이 | 5회차 | **본 감사** | 죽인 검사 |
|---|---|---|---|
| `MGHOSTBRANCHPROHIB` | SURVIVED(死因 Q1) | **KILLED** | C20 `['b_more_boots']` |
| `MREVERT597` | (신규) | **KILLED** | C20 `['b_more_boots']` |
| `MFORKB` | KILLED | **KILLED** | C20 `['b_precision_bound']` + `C11 FORK content is pinned row by row` |
| `MPROHIBSWAP` | KILLED | **KILLED** | C20 `['VARIANCE_UNMEASURED']` |
| `MLIMITSWAP` | KILLED | **KILLED** | `C6 every registered limit anchor is still in the document` |
| `MLEDGER04ALL` | KILLED(traceback 0) | **KILLED**(traceback 0) | `C6 … ledger_0_4_rows -- meta=14 document=0` |
| `MPREDCONSTDROP` | KILLED | **KILLED**(충실형) | `C21` |

★`MPREDCONSTDROP` 주석: 본 감사는 두 형태를 돌렸다. **(i)** 파생 import를 **바이트 동일한
리터럴**로 재입력 → `SURVIVED`. **(ii)** 충실형(리터럴 + 상수 하나 누락, 이름의 `DROP`)
→ **`KILLED`**(C21). C21은 `tuple(R.PREDICATE_CONSTANTS) == tuple(P.LABEL_CONSTANTS)` 의
**값 동등** 검사이므로 (i)이 통과하는 것은 **검사 자신이 이미 등재한 한계**다
(`af1_selftest.py:711–712`: *"rev5 made `PREDICATE_CONSTANTS` derived, which stops it DRIFTING
but not someone re-typing it as a literal"*). 따라서 5회차의 `KILLED` 보고와 모순이 아니고
**신규 결함도 아니다** — 다음 회차가 다시 발견하지 않도록 §5-4에 등재한다.

**실변이 12(대조 `MNONE` 제외 11 + `MPREDCONSTDROP2`) → KILLED 11 / SURVIVED 1**
(그 1은 위 (i), 등재된 한계).

---

## 4. 예외 (a)(b) 판정 — **둘 다 미발화**

### (a) 결정 규칙(라벨·분기·상·가드)을 건드리는 결함 — **미해당**

rev5→rev6 diff를 **전수**로 읽었다(`git diff HEAD`, 5파일 117+/22−). `af1_rule.py` 변경분은
`RULE_REV 5→6` 과 docstring뿐이고 `RULES`·`AXES`·`SUBSTANTIVE`·`FORK_BRANCHES`·`INCOHERENCE`
어느 것도 바뀌지 않았다. `af1_predicates.py`는 `PRED_REV` 와 `neutrality_axis` docstring뿐
(**술어 본문 무변경** — :525–535 그대로). `af1_expected_labels.json`은 `prereg_counts` 3개 갱신
(27→30, 34→38, 12→14 — 원장 행 추가에 따른 것, C6이 문서와 대조해 통과) + `external_identifiers`
신설. **결정 규칙 0줄 변경.**

★**게이트 88(삭제는 조건 등록이 아니다) 검사**: rev5→rev6 삭제 라인 **22줄 전수 열거**했고,
전부 (i) 헤더 재명명 1 · (ii) §5.1 한 문장의 확장 교체 1 · (iii) S4 줄 교체 1 · (iv) §8 금지문
2건 교체 4 · (v) 카운트 3 · (vi) docstring/REV 8 · (vii) C20 known/regex 4 로 설명된다.
**rev5가 등록했던 조항 중 rev6에서 사라진 것은 0건.** 금지문 **18**·§0.1 **12**·한계 **9** ·
`limit_anchors` **9** 전부 보존(C6 통과).

### (b) 새로 등록된 예측이 GPU 0에서 반증 — **미해당**

| 새로 등록된 검사 가능 진술 | GPU 0 검정 | 결과 |
|---|---|---|
| `af1_rule.py:31` *"the widened C20 flags exactly `b_more_boots` and nothing else"* | rev5 문서에 확장 C20 적용 | **참** (§1-3) |
| `PREREG:54` *"확장 즉시 잡았다 — 수리가 자기 판별식을 내장한다"* | 동일 | **참** |
| `PREREG:469` 상대차 `2.24 / 2.29 / 2.26%` | 등록 상수로 독립 재계산 | **참** — 2.2393 / 2.2906 / 2.2647% |
| §5.1 등록 예측 1–5 | rev6에서 **무변경**(diff 확인), 5회차 T3이 이미 검정 | **반증 0건** |

독립 재계산(`PARAM_COUNT 8.888227328e9`, p50 2,514 tok, 312 TFLOPS, 64/108):
cp **0.440730** · d44 **0.430861** ⇒ 4자리 반올림 0.4407 / 0.4309 로 §5.1 표와 일치.

---

## 5. 등재 권고 (목록 밖 · **판정에 미반영** · (a)(b) 미해당)

1. ★★**`PREREG:167`(§0.3 Q-권고4 행)이 거짓이다 — 하지 않은 수리를 "수리"로 적었다.**
   행 원문: *"| Q-권고4 | `af1_predicates.py` 변경 이력 헤더가 rev2에서 멈춤 | **수리** —
   rev5·rev6 항목 추가 |"*. 실측: 그 파일의 헤더는 **여전히 `What changed from rev2` /
   `What changed in rev2, from rev1` 둘뿐**이고 rev3·rev4·rev5·rev6 항목이 **하나도 없다**
   (`grep -n "What changed" af1_predicates.py` → :3, :31, :515[함수 내부]). 파일 전체에서
   rev5/rev6를 언급하는 곳은 `neutrality_axis` docstring :522 **한 줄뿐**이고, 그것은
   Q1 수리의 부산물이지 변경 이력 헤더가 아니다. `PRED_REV = 6` 인데 헤더는 rev2 —
   5회차 등재 권고 4가 지적한 상태가 **더 벌어졌다.**
   판정: 표준 게이트 **#67/#70**(*"변경 이력표·검증 칸에는 이미 실행된 검증만 적어라"*) 위반.
   **(a) 미해당**(라벨·분기·상·가드 아님) · **(b) 미해당**(§5.1 예측 아님) ⇒ 판정 미반영.
   **비용 0 처리**: 헤더에 rev3–rev6 항목을 실제로 추가하거나, 행을 **`미해소(등재)`** 로
   강등하라. 어느 쪽이든 §0.3을 인용하는 산출물이 나오기 **전에** 하라.
   ※ 나머지 자기신고 행은 참이다 — Q-권고2(`RESULT_AF1_STRATA_2026-09-04.md` 정정 배너) **실재**
   (diff에 +10줄 확인), Q-권고3(§5.1 분모 고정) **실재**, Q-권고5·6은 `미해소(등재 유지)`로
   정직하게 남겼다.

2. ★★**`C20c`가 닫았다고 선언한 세탁 구멍은 절반만 닫혔다** — `MEXTIDLAUNDER` **SURVIVED**.
   C20c 주석은 *"an unused slot launders any ghost. **Every entry must earn its place.**"*
   라고 등록하지만 구현은 `e not in raw2`, 즉 **문서 어디에든** 등장하면 통과다(§8·§9로
   한정하지 않는다). 본 감사가 만든 변이: `meta.external_identifiers` 에 `b_more_boots` 를
   1줄 추가하고 §8 금지문의 분기를 `b_precision_bound` → `b_more_boots` 로 되돌림
   (= **死因 Q1 그 자체**) ⇒ 자기검사 **rc=0, 130 PASS, 통과**.
   정량: 등록된 15항 중 C20의 실제 스캔이 필요로 하는 것은 **`boot_sd` 하나뿐**이다
   (§8·§9의 백틱 토큰은 총 6개: 대문자 라벨 4 + `b_precision_bound` + `boot_sd`. 나머지 14항은
   C20에 대해 **불활성**이고, `SM_COUNTS`·`manual_divisions` 등은 코드펜스 안이라 백틱 매치가
   안 된다). ⇒ 해치의 14/15가 *"자기 자리를 벌지"* 않는다.
   판정: **(a) 미해당** — rev6 산출물 자체의 결정 규칙은 정합하고(`rev6 ghosts = []`,
   `MGHOSTBRANCHPROHIB`·`MFORKB` 사망), rev6는 rev5보다 **강해졌지** 약해지지 않았다.
   **(b) 미해당** — 이 변이는 `af1_expected_labels.json` + 사전등록의 **2파일 일관 편집**이고,
   그것은 `af1_selftest.py`의 *"What this self-test CANNOT do"* 절과 §9-8이 **이미 등재한
   한계**다. 5회차가 그은 선(*"Q1은 그 한계의 사례가 아니다 — 일관 편집이 아니라 비일관 상태가
   이미 존재했다"*)을 그대로 적용하면 이것은 **일관 편집 쪽**이다 ⇒ 등재만.
   ⚠️단, rev6는 그 한계의 **표면적을 넓혔다**: rev5까지 C20의 `known` 은 코드(rule·predicates)
   에서만 왔는데, 이제 **데이터 파일 1줄**이 산문 토큰을 화이트리스트할 수 있다. 값싼 강화:
   C20c를 *"등록 항목은 §8·§9 스캔이 실제로 필요로 해야 한다"* 로 좁히거나, 해치를 코드로 옮겨라.

3. `PREREG:469`가 반올림한 피연산자로 식을 쓴다 — `|0.4309−0.4407|/0.4407` 을 그대로 계산하면
   **2.2237% ≈ 2.22%** 이고, 등록값 **2.24%** 는 **비반올림** MFU(0.440730/0.430861)에서만 나온다
   (재계산 2.2393%). 세 값 전부 비반올림 기준으로는 정확하다. *"(비반올림 기준)"* 한 마디면 닫힌다.

4. `C21`의 이름(*"the rule's predicate-constant list **IS** the predicate module's"*)이
   구현(값 동등)보다 강하다. 바이트 동일 리터럴 재입력은 통과한다 — **검사 주석이 이미 등재한
   한계**이므로 결함이 아니다. 다음 회차가 이것을 신규 결함으로 재발견하지 않도록 여기 적는다.

5. `C20b`의 감시 토큰이 **정확히 둘**(`b_precision_bound`·`boot_sd`)뿐이다. 비공허성은 확보됐고
   축소 변이는 죽지만(§2), 여유가 얇다. 이 둘 중 하나가 §8·§9를 떠나면 C20b는 n=1 검사가 된다.

6. 5회차 등재 권고 5(`M25` — C8의 세계 지정을 라벨이 아니라 **구조**로 고정)는 rev6에서
   `미해소(등재 유지)` 로 정직하게 승계됐다. 본 감사는 새 스윕을 열지 않았으므로 `M25` 를
   다시 돌리지 않았다.

---

## 6. `GO` 직후 — 하네스 단계 전 등재할 한계

**5회차 판정서 말미의 11항을 그대로 승계한다**(재작성하지 않는다 —
`audit_af1_rules_5th_2026-09-04/VERDICT.md` §"`GO` 직후 — 하네스 단계 전 등재해야 할 한계").
여기에 §5-1(§0.3 Q-권고4 행 정정)과 §5-2(C20c 해치 강화)를 **비용 0 선결 항목**으로 덧붙인다.

---

## 7. 불변 배너 (전부 승계)

HE0 · 정책 순위 · gate #13/#16 *"닫았다"* 금지 · switch-cost *"닫았다"* 금지 ·
C2 인용정지 (a)(b) · `CONSENSUS.md` §1-24. **전부 불변.**

★**원래 질문(chunked prefill vs PD-mux 정책 비교)은 이 회차에서도 0건 측정.** 이 판정서는
규칙층 문자열·변이 확인일 뿐이며 성능·순위·정책에 대해 **아무것도 말하지 않는다.**

---

## 부록 — 재현

```
# 대조군 (정본 위치)
python3 af1_selftest.py            # rc=0, 130 PASS

# 판별식 (수리 전 문서에 확장 C20 적용)
git show HEAD:workspace/engine-port/results/cp_baseline/PREREG_AF1_2026-09-04.md
  → 확장 C20 + rev6 known  ⇒ ghosts == ['b_more_boots']
  → 확장 C20 + rev5 known  ⇒ ghosts == ['b_more_boots']

# 변이 (전부 스크래치 격리 사본, 깊이 w/e/r/cp_baseline + ../../../../reports/)
MNONE SURVIVED · MGHOSTBRANCHPROHIB/MREVERT597/MFORKB/MPROHIBSWAP/MLIMITSWAP/
MLEDGER04ALL/MPREDCONSTDROP2/MC20NARROW/MC20FILTER/MEXTIDGHOST KILLED ·
MPREDCONSTDROP(바이트 동일 리터럴) SURVIVED[등재된 한계] · MEXTIDLAUNDER SURVIVED[§5-2]
```
