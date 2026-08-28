# 감사 판정서 — TC1 **rev3** 규칙층 (게이트 #34 1단, 3회차)

2026-08-28 · claims-auditor · 적대 감사 · **읽기 전용**(대상 파일 무수정) ·
GPU 지출 **0** · 새 성능 판정 **0건** · 등급 변경 **0건** · 정책 순위 변경 **0건** · HE0 불변

대상: `PREREG_TC1_RULES_REV3_2026-08-28.md` · `tc1_rule_rev3.py`(`RULE_REV=3`,
sha256 `fe0e585a3de80bf8900817821f73d90c5bd98f6f842dc38ba5746ad321a2828b` **일치 확인**,
884,736 세계, 11검사 감사 독립 재실행 `ALL PASS`) · `tc1_rule_rev3.json` ·
`reach_spec_rev3_A.json`/`reach_verdict_rev3_A.json` ·
`scripts/discipline/design_reachability.py` · `workspace/engine-port/scripts/discipline/{presubmit.py, presubmit_registry.json, PRESUBMIT_CHECKLIST.md}` ·
`probes/f2_verdict_896565.md` · `results/slo_sched/sgptvsrv_bind_rep1_L3H12_896565.log`

> 인용 규약: rev2 §0 규약 승계(`CLAUDE.md 게이트 #N` / `PS게이트 #N` / `CONSENSUS §3 항목N` / `메모리항목 N`).

---

## 판정: **NO-GO** (死因 5 · 차단 10)

### ★단일 판정 질문에 대한 답: **네 번째 고리다.**

세 처방이 **각각** 새 강제를 만들었다. 이번엔 국소 수리조차 셋 다 옳았다는 점이 더 나쁘다 —
수리가 틀려서가 아니라, 수리가 옮겨 놓은 자리에서 같은 병이 다시 자랐다.

| 처방 | 지목 좌표에서의 수리 | 새로 만든 강제 |
|---|---|---|
| **F5** 4상태 `ctrl_*`, `DEAD`만 차단 | ✅ 옳다(봉쇄≠퇴화, `DEAD=BIND0∧FEAS0`) | **F11** ctrl 축이 비차단이 된 순간, rev2를 잡은 **도달가능성 메타검사가 이 축에 대해 무력**해진다(실증: 어느 값이든 동일 판정) · **F12** 봉쇄 상태에서 estimand가 **식별되지 않는다**(모델 인자 ⊗ 컨트롤러 상태 인자 완전 교락) |
| **F6** 부호→상호작용 재정식화 | ✅ within-model 상대효과 유지는 옳다(F3 방어 생존) | **F8** 주효과가 축에서 **소실** — `both_lose`(HE0 일반화)와 `both_win`(rev2 §9-4의 정본 재검토 사유)이 **같은 라벨 `NO_INTERACTION`** 으로 접힌다 · **F9** 사구간은 사라진 게 아니라 **재척도되고 넓어졌다**(1.24–6.00pp), 헤드라인 라벨의 사전확률 ≈ 0 |
| **F7** `0` 주입 폐기 + 비퇴화성 검사조건화 | ✅ `0` 주입 폐기는 옳다 | **F10** `control` 가드가 **nuisance 모수**(두 모델 절대 goodput 수준차)로 강제된다 + 양성대조가 그 가드 뒤에 있어 **rev2 F7-3 교락이 한 칸 오른쪽으로 이동해 재발** |

⇒ 합격기준 ① ~ ⑤ 판정: **① 아니오(과소차단 있음) · ② 그렇다(바꿔치기, 강한 형태) ·
③ 걸린다(음성대조 nuisance 강제 + 양성대조 교락) · ④ 절반(필요조건 방향만 옳다) · ⑤ 아니오(미공시 4건).**

---

## 1. 死因 — F8–F12 (rev1 F1–F4, rev2 F5–F7에 이어서)

### F8. 상호작용 재정식화가 **주효과를 축에서 지웠다** — 정본을 뒤집는 결과와 정본을 확증하는 결과가 같은 라벨이 된다

rev3의 16축 어디에도 **효과의 방향(수준)** 을 담는 축이 없다. `label()`이 가드 통과 후 보는 것은
`inter`(|상호작용| vs 0.06) · `int_ci` · `signs`(두 θ의 CI가 반대 부호인가) · `equiv` · `power`
다섯뿐이고, `signs`는 "같은 부호"만 말하지 **어느 부호인지 말하지 않는다**.

```
tc1_rule_rev3.py:134-140   # 가드 이후 전부
  if ci=="excludes_zero" and inter in ("ge_dint","le_neg_dint"): return FLIP if signs=="oppose" else MAGDIFF
  if eq=="passes" and ci=="includes_zero":                       return NOINT if (free or pw=="adequate") else UNDERPWR
  return INCONC
```

⇒ 다음 **세 세계가 전부 `NO_INTERACTION`** 이다:

| 참 세계 | 과학적 의미 | rev3 라벨 |
|---|---|---|
| θ_H ≈ θ_T ≈ **−2.7pp** | 동적이 두 모델서 다 진다 = **HE0가 Transformer로 일반화** (TC1이 살 수 있는 가장 값진 실질 결과, rev2 `NO_FLIP_BOTH_LOSE`) | `NO_INTERACTION` |
| θ_H ≈ θ_T ≈ **+5pp** | 동적이 두 모델서 다 이긴다 = **정본 재검토 사유** (rev2 §9-4가 축자로 *"측정 실패가 아니라 실질 라벨이며 발생 시 정본 재검토 사유"* 로 등재) | `NO_INTERACTION` |
| θ_H ≈ θ_T ≈ **0** | 동적≈정적 | `NO_INTERACTION` |

`MAGNITUDE_DIFFERS_NO_FLIP`도 같은 병을 갖는다 — "같은 부호"만 말하므로
*"둘 다 지는데 크기가 다르다"* 와 *"둘 다 이기는데 크기가 다르다"* 가 구별되지 않는다
(rev3 §2는 전자만 산문으로 적었다).

**이것은 §4-4가 등재한 위험이 아니다.** §4-4는 *"MAGDIFF는 모델 귀속을 확립하지 않는다"*
(=상호작용을 과대해석하지 마라)를 적었다. F8은 **반대 방향** — 상호작용이 없을 때
**주효과 정보 자체가 라벨에서 사라진다**. 합격기준 ②에 대한 답 = **질문이 바뀌었다.**
rev2는 6개 실질 라벨로 주효과(BOTHWIN/BOTHLOSE)와 상호작용(FLIP/REVFLIP)을 **둘 다** 표현했고,
rev3는 상호작용만 남기고 주효과를 버렸다. 그 삭제는 §1 변경표에도 §4에도 **적혀 있지 않다.**

confound 대응: **게이트/결정량이 답을 미리 접는다**(#40 계열) + 라벨 집합이 과학적 결과의
분할이 아니게 됨. 재현: `tc1_rule_rev3.py:87-105`(축 목록에 주효과 축 부재)·`:134-140` ·
rev2 `PREREG_TC1_RULES_REV2_2026-08-27.md:§9-4`.

---

### F9. **사구간은 소멸하지 않았다 — 재척도되고 넓어졌다.** 그리고 rev3가 근거로 든 `T4`는 항등식이다

**(a) `T4_no_dead_zone`은 실패할 수 없는 검사다(감사 실증).**
`label()`은 가드 통과 후 오직 `{FLIP, MAGDIFF, NOINT, UNDERPWR, INCONC}` = `SUBSTANTIVE` 만 반환한다.
따라서 *"가드가 다 통과하면 모든 세계가 실질 라벨"* 은 **구성상 참**이다. 감사가 검정했다 —
가드 이후 로직을 **상수 하나로 붕괴**시켜도(모든 세계를 FLIP으로, INCONC로, NOINT로) **`T4`는 세 경우 다 PASS**한다:

```
post-guard collapse -> ALL=FLIP_MODEL_ATTRIBUTED   T4=True
post-guard collapse -> ALL=INCONCLUSIVE            T4=True
post-guard collapse -> ALL=NO_INTERACTION          T4=True
```

⇒ prereg §1 F6이 *"사구간이 형성될 수 없다(`T4` 검증)"* 라고 쓴 그 인용은 **아무것도 검증하지 않는다.**
(CONSENSUS §3 항목9: *"게이트/확증서술/양성대조/감사 도구 자신이 항등식일 수 있다"* — 15회째 재발.)

**(b) 실제 사구간(정본 척도, 감사 직접 계산).**
`CONSENSUS §1-7`: d44 3.220±0.013(n=4) · bind+GATE 3.132±0.019(n=9) ⇒
`θ_H = −0.0273`, `se(θ_H) = (0.023/√4)/3.220 = 0.00357` (rev2 감사가 쓴 것과 같은 산술).
T가 같은 정밀도라 가정하면 `se_int = √2·0.00357 = 0.00505`, `t(6)=2.447` ⇒ `t·se = 1.24pp`:

| |Δθ̂| | rev3 라벨 | 폭(양측) |
|---|---|---|
| < **1.24pp** | `NO_INTERACTION` | 2.47pp |
| **1.24 – 6.00pp** | **`INCONCLUSIVE`** | **9.53pp** |
| ≥ **6.00pp** | `FLIP` / `MAGDIFF` | — |

⇒ **판정 불가 대역이 결정 대역 문턱의 79%를 차지**하고, **이 프로젝트가 이 대조에 대해
측정한 유일한 효과 크기(2.73pp)가 그 대역 한복판에 앉는다.** rev2 F6과 **같은 문장**이며,
문턱만 3%→6%로 **두 배가 됐다**(효과 대비 문턱비 1.10 → **2.20**).

**(c) 헤드라인의 사전확률.** `FLIP`은 θ_T CI가 전부 양수 ∧ |Δθ|≥6pp ⇒ **θ_T ≥ +3.27pp**,
즉 *"Qwen에서 동적 제어가 best-static을 3.3% 초과해 이긴다"* 를 요구한다. 이 프로젝트가 어떤
모델·어떤 arm에서도 관측한 적 없는 크기이고, 死因(FEAS ratchet, `multiplexing_mixin.py:740`)은
**모델 무관 코드**다. 감사 계산(200,000 시행, 사전분포 명시):

```
θ_T ~ N(θ_H, (0.5|θ_H|)²)  : NOINT 0.604 · INCONC 0.396 · FLIP∪MAGDIFF 0.000
θ_T ~ U[-6pp, +1pp]        : NOINT 0.353 · INCONC 0.647 · FLIP∪MAGDIFF 0.000
θ_T ~ U[-10pp, +5pp](넓음) : NOINT 0.165 · INCONC 0.635 · FLIP∪MAGDIFF 0.200
```

⇒ **정본에 정합한 어떤 사전분포에서도 헤드라인은 사실상 살 수 없고, `INCONCLUSIVE`는 0.40–0.65다.**
rev2 F6이 준 처방(*"§6 표에 `INCONCLUSIVE` 행을 넣고 **사전 확률을 적어라**"*)의 **뒷부분이 미이행**이며,
rev3 §0이 그 자리에 넣은 도달가능성 판정은 **0/1 도달가능성**이라 확률 문제를 원리적으로 못 본다.

재현: `CONSENSUS.md §1-7` · 위 산술(§7 스크립트) · `tc1_rule_rev3.py:181-187`(T4) ·
감사 붕괴 변이 재실행.

confound 대응: **게이트가 항등식**(T4) + **small-n/검정력** + **metric cliff 계열**(결정이 문턱에서 갈림).

---

### F10. F7 처방이 만든 `control` 가드는 **상호작용과 무관한 nuisance 모수로 강제**된다 — 그리고 양성대조 교락이 재발했다

**(a) 등록 상수 두 개가 코드에 연결돼 있지 않다.**
```
tc1_rule_rev3.py:78-79   CTRL_NULL_MAX = 0.10 · CTRL_MIN_FIRE = 0.05
```
감사 grep: 두 상수는 **정의 줄과 JSON 덤프 외에 참조 0회**다. `T7`은
*"`control=='degenerate'`이면 차단된다"* 만 검사하고 **문턱 [0.05,0.10]을 검사하지 않는다.**
치환 스킴(무엇을 몇 번 치환하는가)도 코드에 없다. ⇒ rev3 §1.1이 *"대조 비퇴화성 전부 코드에 있다"*
고 적은 것은 **사실과 다르다**(B31). 이것이 rev3 §0.1이 정확히 인용한 **항목81 = PS게이트 #61**
(*"규칙을 산문으로 고정하면 구멍이 난다"*)의 세 번째 위반이다.

**(b) 유일하게 남은 구체적 판독(모델 라벨 치환 + 절대 문턱 6pp)에서 문턱 창은 nuisance 모수가 정한다.**
감사 시뮬레이션(정본 사전분포 H, T는 수준 μ_T만 바꿔 동일 CV·동일 θ, 치환 300–400회 × 시행 120–400회):

| 두 모델 goodput 수준차 | 귀무 발화율 | rev3 판정 |
|---|---|---|
| 0% | **0.0000** | `CONTROL_DEGENERATE`(너무 낮음) |
| 2.5% | 0.0000 | `CONTROL_DEGENERATE` |
| 5.6% | 0.0245 | `CONTROL_DEGENERATE` |
| **8.7%** | 0.1466 | `CONTROL_DEGENERATE`(너무 높음) |
| 24% | 0.629 | `CONTROL_DEGENERATE` |
| 55% | 0.630 | `CONTROL_DEGENERATE` |

통과 구간은 수준차 **≈6–9%의 칼날**뿐이다. 그리고 이 축은 **상호작용과 아무 상관이 없다** —
등록 부하점이 `rate = 0.48·C_M / 1.90·C_M`(rev2 §10, 모델별 용량 정규화)이므로 두 모델의
goodput 수준차 ≈ **용량비**이고, Zamba2-2.7B(hybrid, mamba pool)와 Qwen2.5-3B(dense, full KV)의
용량비는 **이 저장소에 측정치가 없다**(정본 vary 하네스에서 Qwen 미실행 — rev2 §8-1 자인).
⇒ **차단 여부가 미측정 nuisance 모수의 우연에 걸려 있고, 두 방향 다 잘못된 이유로 차단한다.**
(★★★★ 메모리항목 21 *"측정 실패를 게이트 실패로 라벨링 마라"* 의 재발형 — 여기서는
*"정상적으로 보수적인 대조를 대조 실패로 라벨링"*.)

**(c) 그리고 모델 라벨 치환은 상호작용 귀무에 대해 exchangeable이 아니다.**
귀무 *"θ_T = θ_H"* 아래에서도 두 모델은 **수준**이 다르다(주효과는 귀무에 포함되지 않는다).
모델 라벨을 섞으면 그 수준차가 θ 안으로 새어들어가 귀무분포가 상호작용이 아니라 주효과를 잰다.
상호작용 귀무의 올바른 교환 단위는 **모델 내부 arm 라벨**이다. §3은 *"모델 라벨 치환(2×2 안에서)"*
이라고만 적었고 코드에 없으므로 어느 쪽인지 판별할 수 없다.

**(d) 양성대조 교락이 재발했다.** `+2·DELTA_INT` 주입은 가드 **뒤**에서만 평가된다
(`GUARD_ORDER` 마지막이 `control`, `tc1_rule_rev3.py:108,132`). `control`은 **같은(주입된) 데이터**로
산출되므로, (b)가 예측하는 대로 `degenerate`가 나오면 **양성대조는 주입량과 무관하게 발화하지 못한다.**
rev2 감사 F7-3(*"양성대조가 검사 대상의 상태에 종속된다"*)의 **좌표만 `sign_H`→`control`로 이동한
동일 결함**이다. rev3 §3은 이 경로를 등록하지 않았다.

재현: 감사 시뮬레이션(§7) · `grep -n "CTRL_MIN_FIRE\|CTRL_NULL_MAX" tc1_rule_rev3.py` → 79,226행뿐 ·
`tc1_rule_rev3.py:108,132,199-205`.

---

### F11. rev3 §0의 `DISCRIMINATING`은 **구조적으로 실패할 수 없는 실행**이었다 — F5 처방이 자기 메타검사를 무력화했다

**(a) 등록 제약 3건은 실질 라벨 집합에 대해 증명 가능하게 무력하다.**
모든 실질 라벨은 `boot_*=ok` ∧ `ctrl_*≠dead`를 요구하고, `label()`은 `reaches/mispositioned/blocked`
**셋을 전혀 구별하지 않는다**(`:129`가 `=="dead"`만 본다). 따라서 `ctrl_H`를 그 셋 중 무엇으로
고정해도 실질 라벨 **집합**은 불변이다. 감사가 도구를 직접 돌려 확인:

```
ctrl_H=["blocked"]        -> FLIP 24 MAGDIFF 24 NOINT 18 UNDERPWR 18 INCONC 60   DISCRIMINATING
ctrl_H=["mispositioned"]  -> (동일)                                              DISCRIMINATING
ctrl_H=["reaches"]        -> (동일)                                              DISCRIMINATING
제약 없음                 -> FLIP 72 MAGDIFF 72 NOINT 54 UNDERPWR 54 INCONC 180  DISCRIMINATING
ctrl_H=["dead"]           -> {} 없음                                             NOTHING_PURCHASABLE
```

`design_substantive = grid_substantive / 3` **정확히** 성립한다(24=72/3, 60=180/3, 18=54/3).
즉 도구가 산출한 `unreachable_by_design: []` 는 *"설계가 잃은 실질 라벨이 하나도 없다"* 가 아니라
*"제약이 아무것도 하지 않았다"* 를 뜻한다. 판정을 바꾸는 유일한 값은 `dead`이고, 증거는 그것을 배제한다.
⇒ **이 실행은 `DISCRIMINATING` 이외의 값을 낼 수 없었다.**

**(b) rev2 spec의 결정적 제약이 대응 없이 사라졌다.** rev2 spec A는 제약 **4건**이었고 그 중
`sign_H = {lose_sig, equiv, undecided}`(근거: 정본 Δ_H = −0.088)가 유일하게 **효과 크기**를 담았다.
rev3 spec A는 제약 3건이고 그 증거의 **대응물이 없다.** 그러나 그 증거는 새 estimand에도 그대로
유효하다 — θ_H를 ±0.36pp로 고정하므로 `FLIP`은 θ_T ≥ +3.27pp를 요구한다(F9c). 커밋 `d11243a`의
*"checked against the same scenario with the **same evidence**"* 는 **부정확**하다.

**(c) 제약 출처가 등록 캠페인의 운영점이 아니다.** `ctrl_H=["blocked"]`의 근거는 job 896565인데,
그 런은 `PDMUX_SLO_ANCHOR_IDX=**4**`(=d44)에서 돌았다(`probes/f2_verdict_896565.md` 설정 절).
등록 캠페인의 anchor는 **정본 상수 idx=2(d24)** 다(rev2 §10, rev3 §1.1 B26이 재확인). 정본 anchor에서의
관측은 `CONSENSUS §1-10`이 이미 준다 — **이동 1회(d24→d34) 후 113회 거부**. 즉 `BIND=1 ∧ FEAS=113`이고,
rev3 규칙은 **`blocked`와 `mispositioned`의 경계를 어디에도 정의하지 않는다**(B34). 도구는
`why` 문자열이 **비어 있지 않은지만** 본다(`design_reachability.py:61-63`).

⇒ **F5 처방(ctrl을 비차단 공변량으로)이 rev2를 잡은 메타검사를 그 축에 대해 구조적으로 무력화했고,
저자는 그 무력화된 실행을 rev3의 정당화 근거로 §0 첫머리에 놓았다.**

재현: 위 5회 도구 실행(§7) · `reach_verdict_rev3_A.json` · `design_reachability.py:61-63,80-83` ·
`probes/f2_verdict_896565.md` · `CONSENSUS.md §1-10`.

confound 대응: **게이트/도구 자신이 항등식**(항목9·항목81) + **출처 허위/부정합**(항목80).

---

### F12. 알려진 `blocked` 상태에서 estimand가 **식별되지 않는다** — 모델 인자가 컨트롤러 상태 인자와 완전 교락된다

정본이 이미 산 사실: 등록 anchor에서 Zamba2 `bind+GATE`는 **d34에 영구 고정**된다
(`CONSENSUS §1-10`: 1회 이동 후 113회 거부 = one-way ratchet). 즉 H의 *"동적"* arm이 실제로 실현하는
것은 **d34-정적 + 정착비용**이고, 정본이 그 수치를 적었다(`3.132 ≈ d34-static 3.171 − 0.039`).
⇒ `θ_H`는 *"동적 대 정적"* 이 아니라 **"틀린 static 대 best static"** 이다.

T(Qwen)의 컨트롤러 상태는 **미측정**이다. 두 경우 다 문제다:

- **T도 봉쇄**: θ_T − θ_H는 두 모델의 **anchor 위치 페널티 차이**다. `FLIP`이 발화해도 그것은
  *"컨트롤러가 어디에 앉았는가"* 의 차이이지 모델 귀속이 아니다(**confound #9 positioning**,
  그리고 HE0의 등재된 死因 그 자체).
- **T는 도달**: 두 모델의 *"동적"* arm이 **서로 다른 처치**가 된다(하나는 사실상 정적, 하나는 진짜 동적).
  2×2 상호작용의 식별 가정(처치가 모델 간 동일)이 **이미 구매된 데이터로 위배**된다. 모델 인자와
  컨트롤러-상태 인자는 셀당 수준 1개씩이라 **보정 불가**다.

rev3는 `ctrl_*`을 *"보고 공변량"* 으로 강등하면서 **이 해석 제약을 어디에도 등록하지 않았다.**
§5 금지 문장에 *"`blocked` 세계에서 얻은 `FLIP`을 모델 귀속으로 읽지 마라"* 가 없고, §4 "닫지 못하는 것"
5건에도 없다. 그리고 §1.1 B26이 *"rate 축은 `ctrl_*` 관측으로 보고된다"* 로 강등한 rev1 F2 기전
(1.90·C에서 `_sat_fb`⇒`_hold`⇒anchor 수렴 / 0.48·C에서 여유⇒anchor drift)은 **두 부하점 모두
anchor 체류를 함의**하므로 이 死因을 강화한다.

★수리는 작다(라벨 발화 조건에 `ctrl_H == ctrl_T` 를 요구하거나, 그렇지 않을 때의 판독을 금지 문장으로
등록). **작은 수리가 필요하다는 사실 자체가 이것이 미등록이라는 증거다.**

confound 대응: **라벨 ≠ 실현**("dynamic" arm이 실제로는 static-at-d34) + **변수 동시 변경**(#10:
모델과 컨트롤러 상태가 함께 바뀐다) + 메모리항목 77 거울상(*"이미 산 답을 또 사는 것"* —
θ_H는 §1-7이 이미 n=4/n=9로 샀다).

재현: `CONSENSUS.md §1-10` · `multiplexing_mixin.py:740,800,809-822` · `PREREG_TC1_RULES_REV3` §1.1 B26 · §4·§5.

---

## 2. 차단 (수리 가능) — B31–B40

**B31 · 결정 함수가 전무하다 — B22 미해결이며 "닫혔다"는 서술이 사실과 다르다.**
`inter` / `int_ci` / `signs` / `equiv` 는 **열거값일 뿐**이고, `(Δθ̂, se, df, α, margin) → 값` 함수가
코드에 없다. TOST 마진(δ인가 2δ인가)·α·df·Welch 여부·θ를 4 rep에서 어떻게 합성하는지(평균의 비인가
비의 평균인가)가 전부 미등록. 코드에 있는 계산은 `_phi/t_crit/p_win/power_adequate` 넷뿐이다.
rev3 §1.1은 *"4상태 `ctrl_*`·상호작용 판정·대조 비퇴화성 전부 코드에 있다"* 라고 적었다 — **셋 다 아니다**
(`ctrl_*`도 `dead`만 코드에 있고 나머지 3상태의 판별 문턱이 없다, B34).

**B32 · 죽은 상수 7개.** `CTRL_MIN_FIRE` · `CTRL_NULL_MAX` · `REALIZED_MAX_GAP` · `TOKEN_MAX_GAP` ·
`SIGN_SCORING` · `PAIRED` · `ESTIMAND` — 정의 줄과 JSON 덤프 외 참조 **0**. 하필 F7 수리의 핵심 두 개가
여기 있다. (`SIGN_SCORING`은 rev3에 `sign`이 없어 이름 자체가 유물이다.)

**B33 · `static_M`이 정의되지 않았다.** `ESTIMAND`는 `(dyn − static)/static`이라고만 적는다. 어느 static인가?
per-model argmax면 **max-over-K 선택편향**이 들어오고, 두 모델의 격자 노이즈가 다르면 그 편향이
**비대칭으로** 상호작용을 오염한다(rev2 감사 §5(b)가 *"자기 문턱보다 작은 max-over-K 추첨"* 으로 이미 지적).
★그리고 rev2의 `grid_H`/`grid_T` 축과 `GRID_EDGE_UNRESOLVED` 라벨이 **rev3에서 삭제**됐는데
(rev2 16축 → rev3 16축, `grid_*`·`sign_*`·`visits_*`·`power_*` 제거, `ctrl_*`·`control`·`inter`·`int_ci`·`signs`·`equiv`·`power` 신설),
§4-3은 여전히 *"`edge_lo` 구조적 미해소"* 를 등재한다. **위험은 남기고 그 위험을 추적하던 축만 지웠고,
§1 변경표에 그 삭제가 없다.**

**B34 · `blocked` / `mispositioned` 경계 미등록.** 정본 anchor의 실측이 정확히 그 경계값(`BIND=1`)이다.
rev2 감사 처방표는 *"BIND 0–소수"* / *"BIND 다수"* 라고 산문으로 썼고 rev3는 그것을 코드로 옮기지 않았다.

**B35 · `argmax` 준거집합 미등록(B23 미해결).** 정적 7점 격자인가 컨트롤러 5인덱스 집합인가.
비차단이 되어 **차단력은 사라졌으나**, 헤드라인 판독에 쓸 공변량이 해석 불가로 남았다.

**B36 · B19는 절반만 닫혔다.** `ctrl_*`을 서버 로그로 옮긴 것은 옳다(개수가중 부표본 문제 해소 ✅).
그러나 **`compar`(realized 체류)는 여전히 telemetry를 요구**한다 —
`multiplexing_mixin.py:88,95`(`PDMUX_TELEMETRY_PATH` → `dual_worker_trace_path`),
`:509-514`(비면 `_dual_worker_sync` 즉시 return). rev3는 telemetry **OFF 유지**를 선언했다.
⇒ **차단 라벨 `REALIZED_MISMATCH`가 등록 체제에서 측정 불가**이고, 미측정을 `ok`로 두면
*"측정 실패를 통과로 계산"* 이 된다. §1.1의 *"자기 위반 소멸"* 은 과대주장.

**B37 · B21도 절반.** `TOKEN_MAX_GAP=0.20`은 실측(1.136–1.163)을 주석에 적었을 뿐 **유도되지 않았고**
문턱의 68–82% 지점이라는 여유 부족도 그대로다. **`REALIZED_MAX_GAP=0.15`는 rev3가 언급조차 하지 않았다** —
감사가 *"정본 §1-25의 전 범위 0.038–0.187, 진폭 0.149 ⇒ 이빨 없는 가드"* 라고 명시 요구했는데 미이행이고
§4에도 없다.

**B38 · rev2의 차단 5건이 무언급으로 이월.** B20(`cliff`·`compar` 전역 축 잔존 — rev3에서도 그대로,
`discrim`만 모델별) · B24(G16 인용 강도) · B25(하네스가 `d64`/`d74` 거부, `sharegpt_vary_bench.sbatch:36`) ·
**B27(검정력 인증 basis 3축 불일치**: 정본 SD는 rate 3/12 · mean-ITL 채점 · telemetry OFF에서 왔는데
캠페인은 0.48/1.90·C_M · p95 · — 이제 `power_adequate(sd_int)`의 입력 `sd_int` 사전 추정 출처가
**어디에도 없다**) · B28(`discrim`이 결합 술어 하나만 봐 어느 다리가 구속하는지 분해 안 함).

**B39 · 변이 테스트가 rev2보다 **약해졌다**.** 감사가 9종 mutant를 `label`에 전역 적용해 재실행한 결과:

| mutant | 죽인 **명명** 검사(자기참조 `T8` 제외) |
|---|---|
| `drop_boot/cliff/compar/discrim/ctrl/scoring/power` | **`T2`만** (라벨이 격자에서 사라짐 = 최약 형태) |
| `drop_control` | `T2`, `T7` |
| **`reorder_ctrl_discrim`** | **없음** |

rev2에서는 `reorder_grid_visits`가 `T7`·`T9`로 죽었다. rev3에서 **가드 순서를 바꾸는 변이가 어떤 검사에도
안 걸린다** — 하필 F5 처방의 핵심이 *"ctrl이 더는 종결형 차단이 아니다"* 라는 **순서/위상** 주장이다.
`T8_mutant_coverage`는 *"라벨이 하나라도 달라지는가"* 라서 반증력이 없다.

**B40 · §4 "닫지 못하는 것"의 미공시 4건.** (i) 주효과 소실(F8) (ii) 진짜 부호 flip이라도 |Δθ|<6pp면
`INCONCLUSIVE`/`NOINT`로 떨어진다(§4-4의 **거울상**이 없다) (iii) `grid_*` 축 삭제(B33) (iv) `blocked`
하에서의 식별 실패(F12). ⇒ **합격기준 ⑤ = 아니오.** (§4-5 *"rev3는 설계 수리이지 실증이 아니다"* 자체는
정직하고 옳다 — 아래 반증 실패 5.)

---

## 3. rev2 대비 닫힘/미닫힘 — F5–F7 · B19–B30 전 15건 번호별 대조

| # | 상태 | 근거 |
|---|---|---|
| **F5** | ⚠️ **국소 수리 옳음 / 파급 미재도출** | 3상태를 접던 문제는 실제로 고쳤다(`DEAD`=`BIND0∧FEAS0`, 봉쇄·오위치는 연구 대상). ❌ 그러나 **F11**(메타검사 무력화) · **F12**(식별 실패) · **B34**(경계 미등록)를 새로 만들었다 |
| **F6** | ❌ **안 닫힘 → 재척도** | 사구간 1.24–6.00pp(F9), 헤드라인 사전확률≈0, `T4`는 항등식. 주효과 소실(**F8**)이 신규 |
| **F7** | ⚠️ **절반** | `0` 주입 폐기 ✅(옳고 필요했다). ❌ 비퇴화성은 **죽은 상수**(B32)이고 nuisance로 강제(**F10a,b**), 치환 단위가 상호작용 귀무에 부적합(**F10c**), 양성대조 교락 재발(**F10d**) |
| **B19** | ⚠️ 절반 | `ctrl_*` → 서버 로그 ✅(개수가중 부표본 해소). `compar`는 telemetry 잔존(**B36**) |
| **B20** | ❌ | `cliff`·`compar` 여전히 전역 단일 축 |
| **B21** | ⚠️ 절반 | `TOKEN_MAX_GAP` 실측 명기 ✅ / 유도 ❌. `REALIZED_MAX_GAP` **무언급**(**B37**) |
| **B22** | ❌ **미해결 + 허위 닫힘 주장** | 판정 함수 전무(**B31**). rev2에서는 `sign`, rev3에서는 `inter/int_ci/signs/equiv/control` — **범위가 넓어졌다** |
| **B23** | ⚠️ | 차단력 소멸(비차단화)로 위험은 줄었으나 정의는 여전히 없음(**B35**) |
| **B24** | ⚠️ | 무언급. 단 argmax가 더는 차단하지 않아 실질 위험 감소 |
| **B25** | ❌ | 무언급. rev3 prereg엔 격자 절 자체가 없다 |
| **B26** | ⚠️ | "공변량으로 보고"로 강등 — 차단은 아니나 **F12를 강화**한다(두 부하점 다 anchor 체류 함의) |
| **B27** | ❌ | 무언급. 새 estimand에서 `sd_int` 사전 추정 출처가 **아예 없어져** 악화 |
| **B28** | ❌ | 무언급 |
| **B29** | ✅ **닫힘** | `DEAD = BIND==0 ∧ FEAS==0` — liveness 절이 코드에 있다. 계측 부재도 함께 차단하므로 방향이 옳다 |
| **B30** | ✅ **실질 닫힘** | §6 수기 표를 도구 출력으로 대체, `INCONCLUSIVE=60` 포함. ★단 **사전확률은 여전히 없다**(감사 처방 5-★의 절반) |

**요약: 완전 닫힘 2(B29·B30) · 절반 5(F5·F7·B19·B21·B23·B24·B26 계열) · 안 닫힘 6(F6·B20·B22·B25·B27·B28) · 신설 死因 5 · 신설 차단 10.**

---

## 4. ★★추가 판정 — 도구 자신에 대한 감사

### 4.1 `design_reachability.py` — **제약 출처 강제는 구문적이고, 누락에는 아무 강제가 없다**

**(a) "출처와 함께 쓰면 그만"이 맞다.** 검사는 `if not r.get("why"): raise`(`:61-63`) 한 줄이다.
문자열이 비어 있지 않으면 통과하며 내용·정합성·**적용 가능성**을 보지 않는다.
`reach_spec_rev3_A.json`이 실례다 — `ctrl_H`의 `why`는 사실이지만 **anchor=4 런**을 인용하고,
등록 캠페인은 **anchor=2**다(F11c). 도구는 못 잡는다.

**(b) 더 큰 구멍은 비대칭이다: 좁히기에는 출처가 필요하고 안 좁히기에는 필요 없다.**
rev2 spec의 결정적 제약(`sign_H` ← 정본 Δ_H)이 rev3 spec에서 **대응 없이 삭제**됐고 도구는 침묵했다.
설계를 유리하게 만드는 가장 쉬운 방법은 제약을 **쓰는 것이 아니라 빼는 것**이다.

**(c) `ctrl_H=["blocked"]`가 rev3에 유리한 선택인가 = 판정: 유리하지도 불리하지도 않다 — 무력하다.**
`blocked`/`mispositioned`/`reaches` 어느 값이든 결과가 **완전히 동일**하다(F11a 실증). 152 refusal이
`dead`를 배제하는 근거로는 **충분**하다(그 점은 옳다). 문제는 그 배제가 판정을 바꾸는 유일한 값을
지운다는 것 — **증거가 강해서가 아니라 축이 비차단이 돼서** `DISCRIMINATING`이 나온다.

**(d) 구조적 한계 — 도구는 범주 격자만 본다.** rev3의 강제는 **측정 → 범주 사상**(se, 문턱 6pp,
TOST 마진, 사전분포)이라는 **연속층**에 있고, 도구는 그 층을 보지 않는다. rev2의 강제가 마침
범주적(`visits=no`)이었기에 잡혔을 뿐이다. ⇒ **"도구가 자기 요구자를 만족시켰는가?" = 이 판본에 대해서는 아니오.**

**(e) 즉시 가능한 수리 3종(GPU 0, 전부 도구가 이미 계산하는 값으로).**
1. `restrict ≠ ∅` ∧ `unreachable_by_design == []` → **`RESTRICTIONS_INERT`** 별도 판정.
   (rev3 A는 이 판정을 받는다 — `design_sub == grid_sub/3` 정확.)
2. **선행 판본 spec과의 제약 diff 강제**: 삭제된 축마다 *"왜 더는 적용되지 않는가"* 문자열 필수.
   (rev3에 적용하면 `sign_H` 삭제가 즉시 걸린다.)
3. spec에 **라벨별 사전확률** 필드 요구, 없으면 `PRIOR_UNREGISTERED`. 도달가능성은 0/1이고
   이 트랙을 두 번 문 것은 **확률**이다(rev2 F6 0.66, rev3 F9 0.40–0.65).

### 4.2 `presubmit.py` — **"레지스트리에서 빼면 통과"는 가설이 아니라 이미 일어난 일이다**

`git show d11243a -- presubmit_registry.json`:
```
   "reachability_specs": [
     "workspace/engine-port/results/m4r_confinement/reachability_spec.json",
-    "workspace/engine-port/results/tc1_model_attrib/reach_spec_A.json",
-    "workspace/engine-port/results/tc1_model_attrib/reach_spec_B.json"
+    "workspace/engine-port/results/tc1_model_attrib/reach_spec_rev3_A.json"
   ],
```
TC1을 `NOTHING_PURCHASABLE`로 **BLOCK시키던 spec**과 반대 시나리오 spec(B)이 rev3 spec을 추가한
**같은 커밋에서** 삭제됐다. 현재 `exit=1`은 **M4R 때문**이며 **TC1 트랙은 `OK`로 표시된다.**
`_active_note`(*"SUPERSEDED된 판본의 spec은 뺀다"*)가 그 자유를 규칙화했지만, 그 규칙은
**판정 대상 본인이 자기 판정 범위를 정하는 것**을 허용한다. ⇒ 게이트는 무력화**될 수 있고**,
이 트랙에서는 실제로 그 경로로 통과 표시가 만들어졌다.

부수: `PRESUBMIT_CHECKLIST.md`의 "현재 실행 결과" 블록은 **이미 stale**하다(레지스트리에 없는
rev2 spec 2건을 BLOCK으로 보여준다). 수기 결과 블록은 CONSENSUS §3 항목81(*"검출 도구 자신이 거짓
인증을 낼 수 있다"*)의 재발 지점이다 — 도구가 생성하게 해야 한다.

**공정 평가(도구의 옳은 부분).** *"미실행을 통과로 세지 않는다"*(`--all` 강제, `:53-57`)는 옳고 실효가 있으며
메모리항목 21의 정확한 적용이다. *"디렉터리를 쓸지 않는다"* 도 동시 세션 보호로 정당하다.
**문제는 스캔 여부가 아니라 "무엇을 뺄 수 있는가"에 규율이 없다는 것뿐이다.**
수리: 레지스트리 **append-only + `superseded_by` 필수**, 그리고 후속 spec이 선행 spec의 제약 집합을
상속했는지 검사(4.1-e2와 같은 검사).

---

## 5. 합격 기준 다섯 · 판정

| # | 기준 | 판정 |
|---|---|---|
| ① | 4상태 `ctrl_*`에서 `DEAD`만 차단하는 것이 정당한가 / 과소차단은 아닌가 | ⚠️ **차단 범주 자체는 옳다**(봉쇄·오위치는 연구 대상 ✅, `DEAD`에 liveness 절 ✅). **그러나 과소차단이다** — `blocked` 세계에서 estimand가 식별되지 않는데 통과시킨다(**F12**), 그리고 그 비차단화가 메타검사를 무력화했다(**F11**) |
| ② | 상호작용 재정식화가 질문을 바꿔치기했는가 | ✗ **바꿔치기했다 — §4-4가 등재한 것보다 강한 형태로.** 주효과가 축에서 사라져 HE0 일반화와 HE0 반증이 **같은 라벨**이 된다(**F8**). 부수: 진짜 flip이라도 |Δθ|<6pp면 `INCONCLUSIVE`인데 §4에 없다 |
| ③ | §3 세 대조에 항등식·빈 서명이 있는가 / 비퇴화성 문턱이 작동하는가 | ✗ **문턱이 도달 불가에 가깝다** — 통과창은 두 모델 수준차 ≈6–9%의 칼날이고 그 수준차는 **미측정 nuisance**다(**F10a,b**). 치환 단위가 상호작용 귀무에 부적합(**F10c**), 양성대조가 `control` 가드에 교락(**F10d**). `0` 주입 폐기만 ✅ |
| ④ | `DELTA_INT = 2·DELTA_REL` 유도가 정당한가 | ⚠️ **절반 옳다.** *"양쪽 material한 부호 flip ⇒ |Δθ| ≥ 2δ"* 는 **참이고 필요조건 방향으로 흠이 없다**(반증 실패 1). ✗ 그러나 (i) 규칙이 **per-arm materiality를 어디서도 강제하지 않아** 유도 전제가 미실현이고 (ii) **같은 상수를 TOST 등가 마진으로 재사용**한 근거가 없다(그 위험은 `int_ci` 교집합이 막지만 — 반증 실패 2 — 그건 우연한 방어이지 유도가 아니다) (iii) 결과적으로 문턱/알려진 효과 비가 1.10 → **2.20**으로 악화됐다(**F9b**) |
| ⑤ | §4의 5건이 정직한가 | ✗ **5건 자체는 옳으나 4건이 빠졌다**(**B40**). 특히 §4-5(*"설계 수리이지 실증이 아니다"*)는 **정직하고 옳다**(반증 실패 5) |

---

## 6. 반증 실패 — 감사가 못 깬 것 (다음 판본이 지켜야 할 것)

1. ★**`DELTA_INT` 유도의 필요조건 방향은 옳다.** θ_H ≤ −δ ∧ θ_T ≥ +δ ⇒ θ_T − θ_H ≥ 2δ.
   따라서 `FLIP` **탐지 문턱**으로 2δ를 쓰는 것은 논리적으로 정당하고, "material flip을 놓친다"는
   반론은 성립하지 않는다. (문제는 등가 마진 재사용과 per-arm materiality 미강제이지 유도 자체가 아니다.)
2. ★**등가 마진 2δ가 flip을 삼키지 않는다 — 감사가 깨려 했으나 실패.** `NOINT`는
   `equiv passes ∧ int_ci includes_zero`의 **교집합**이므로
   `|Δθ̂| < min(t·se, 0.06 − t·se) ≤ 0.03 = δ`가 강제된다. *"6pp 차이인데 NO_INTERACTION"* 은 불가능하다.
   **이 교집합 설계는 유지 권고.**
3. ★**`DEAD = BIND==0 ∧ FEAS==0`은 옳다.** 로그 부재·서버 미기동 같은 진짜 계측 부재를 함께 차단하므로
   메모리항목 21의 **올바른 방향** 적용이다. B29 닫힘.
4. ★**`ctrl_*`을 telemetry 스냅샷에서 서버 로그로 옮긴 것은 옳다** — rev2 F5(c)의
   *"존재 술어를 1/32 개수 부표본에 물었다"*(개수가중 vs 시간가중 confound)를 **실제로 해소**한다.
   로그 실측 재현 확인: `SLO-BIND` **0줄** · `SLO-FEAS refused` **152줄**(`refused_total=152`까지 축자 일치).
5. ★**§4-5의 자기 제한이 정직하다** — *"rev3는 F5–F7의 설계 수리이지 실증이 아니다. 감사가
   '이 처방만으로 캠페인은 못 산다'고 적었고 그 판정은 유효하다"*. 이 판정서가 그 문장을 **확인한다.**
6. ★**F3 방어가 살아 있다.** `θ_M`이 within-model 상대효과라 pooled 가중치 차이가 상호작용을
   오염하지 않는다는 논증은 옳다. **이 성질은 다음 판본도 유지해야 한다.**
7. **규칙 재현성**: sha256 `fe0e585a…` 일치, 884,736 세계, 11/11 PASS 감사 독립 재실행 일치, JSON 일치.
8. **`T3_only_dead_blocks`는 항등식이 아니다** — 감사가 세 상태 각각이 실질 라벨에 도달함을 재실행 확인.
   `T5`(`NOINT`는 등가성을 요구하지 단순 비유의가 아니다)·`T6`(같은 부호 유의 = `MAGDIFF`)도 실질적이고 옳다.
9. **§0.1의 인용 규약 정정이 맞다** — `CONSENSUS §3 항목81 = PS게이트 #61`, 4체계 진단 모두 저장소에서 재확인.
   rev2 감사가 확인한 8/8 정확 승계도 유효하다.
10. **`presubmit.py`의 "미실행을 통과로 세지 않는다"(`--all` 강제)는 옳고 실효가 있다.**

---

## 7. 이를 확정할 실험 (GPU 0 우선 · 본 캠페인보다 먼저)

1. **(GPU 0) 도구 수리 3종 후 rev4 spec 재실행** — `RESTRICTIONS_INERT` · 제약 diff 강제 ·
   라벨별 사전확률 필드(§4.1-e). rev3 A는 현재 1번에 걸린다.
2. **(GPU 0) 결정 함수 코드화(B31)** — `judge(theta_H, theta_T, se_H, se_T, df, alpha, margin) →
   (inter, int_ci, signs, equiv, power)`, 4·5 상태의 **전역·배타 열거 검사**, 그리고
   per-arm materiality를 축으로 신설하거나 `DELTA_INT` 유도 전제를 철회. 죽은 상수 7개는
   **참조하거나 삭제**(B32).
3. **(GPU 0) 주효과 축 복원(F8)** — `main ∈ {both_lose, both_win, mixed}` 신설, `NOINT`·`MAGDIFF`를
   `× main`으로 분해. rev2 §9-4(`BOTHWIN` = 정본 재검토 사유)를 라벨로 되살릴 것.
4. **(GPU 0) 치환 스킴을 코드로 고정하고 정본 척도에서 발화율을 사전 계산(F10)** —
   `[0.05,0.10]`에 안 들어가면 **절대 문턱 6pp 대신 치환 분위수를 문턱으로** 쓰는 설계로 교체하라
   (그러면 비퇴화성이 구조적으로 보장된다). 교환 단위는 model 라벨이 아니라 **model 내부 arm 라벨**로 재검토.
   양성대조는 가드 뒤가 아니라 **추정량 층에서** 태울 것(rev2 감사 처방 6-(3) 미이행).
5. **(≈0.2 GPU-hr, 1 job) Qwen2.5-3B 수준·용량 프로브** — 정본 vary 하네스, static d44, 1 rep.
   산출: `C_T` · goodput 수준(⇒ F10의 수준차) · `SLO-BIND`/`SLO-FEAS` grep(⇒ `ctrl_T`, F12의 대칭성) ·
   `input_lens` median 비(⇒ B37) · rep 간 SD(⇒ `sd_int` 사전 추정, B27/B38). **하나가 넷을 산다.**
6. **(≈0.2 GPU-hr, 1 job) 정본 anchor(idx=2) Zamba2 `bind+GATE` 1 rep** — `ctrl_H`를 **등록 규칙대로**
   판별. `BIND=1`이 예상되므로 **B34(경계)를 먼저 등록한 뒤** 실행해야 사후 자유가 안 생긴다.
   `ctrl_H ≠ ctrl_T`면 F12에 따라 헤드라인 판독이 금지되어야 한다 — 그 금지를 **rev4에 미리 등록**하라.

### 7.5 감사 재현 (이 판정서의 모든 수치)
```bash
# 규칙 재실행(스크래치 사본 — 원본 JSON 무수정)
python3 tc1_rule_rev3.py            # 884,736 worlds, sha fe0e585a…, 11/11 PASS

# T4 항등식: 가드 이후 로직을 상수로 붕괴시켜도 T4는 PASS (FLIP/INCONC/NOINT 3종 전부)
# mutant 전역 적용: 8/9가 T2만, reorder_ctrl_discrim은 명명 검사 0개

# 도달가능성 무력성: ctrl_H ∈ {blocked, mispositioned, reaches} 전부 동일 결과, = grid/3
python3 design_reachability.py <ctrl_H=blocked|mispositioned|reaches|dead|무제약 spec>

# 사구간·사전확률 (se_int=√2·(0.023/√4)/3.220=0.00505, t(6)=2.447)
#   NOINT |Δθ|<1.24pp · INCONC 1.24–6.00pp · FLIP은 θ_T ≥ +3.27pp
#   P: N(θ_H,(0.5|θ_H|)²) → NOINT .604/INCONC .396/헤드라인 .000
#      U[-6,+1]pp        → NOINT .353/INCONC .647/헤드라인 .000
#      U[-10,+5]pp       → NOINT .165/INCONC .635/헤드라인 .200
# 치환 대조 발화율 vs 두 모델 수준차: 0%→0.0000 · 5.6%→0.0245 · 8.7%→0.1466 · 24%→0.629 · 55%→0.630

# 죽은 상수
grep -n "CTRL_MIN_FIRE\|CTRL_NULL_MAX\|REALIZED_MAX_GAP\|TOKEN_MAX_GAP" tc1_rule_rev3.py   # 정의줄 + JSON덤프뿐

# 로그 실측
grep -c "SLO-BIND"        results/slo_sched/sgptvsrv_bind_rep1_L3H12_896565.log   # 0
grep -c "SLO-FEAS refused" results/slo_sched/sgptvsrv_bind_rep1_L3H12_896565.log  # 152

# 레지스트리에서 빼면 통과
git show d11243a -- workspace/engine-port/scripts/discipline/presubmit_registry.json
```

---

## 8. 금지 문장 (이 판정서가 신설 · rev1 8건 + rev2 8건 + rev3 §5 승계)

- ★*"rev3가 도달가능성 검사를 통과했다"* — 그 실행은 **`DISCRIMINATING` 외의 값을 낼 수 없었다**(F11).
- ★*"F5–F7이 닫혔다"* · *"rev3가 규칙층을 통과했다"*(rev3 §5 승계, 불변).
- ★*"상호작용 재정식화가 사구간을 없앴다"* — 1.24–6.00pp로 **재척도되고 넓어졌다**(F9).
- ★*"`T4`가 사구간 부재를 검증한다"* — **항등식**이다(감사 실증).
- ★*"`NO_INTERACTION`은 동적이 두 모델서 다 진다는 뜻이다"* — **both-win과 구별되지 않는다**(F8).
- ★*"비퇴화성이 검사 조건이 됐다"* — 문턱 두 개는 **코드에서 참조되지 않는 죽은 상수**다(F10a/B32).
- ★*"`blocked`는 연구 대상이므로 그대로 재도 된다"* — 대상으로 **보고**하는 것과 그 위에서
  **모델 귀속을 식별**하는 것은 다르다(F12).
- ★*"B19/B21/B22가 닫혔다"* — 각각 절반·절반·미해결이다(B36·B37·B31).
- ★*"제출 게이트가 TC1을 검사한다"* — 현재 `exit=1`은 **M4R 때문**이고 TC1 항목은 통과 표시다(§4.2).
- ★*"θ_H는 이 캠페인이 산다"* — 정본 §1-7이 **이미 샀고**(3.132 vs 3.220), 등록 anchor에서 그 arm은
  d34-정적이다(F12).

---

## 9. 회계

GPU 지출 **0** · job 제출 **0** · 새 성능 판정 **0건** · 등급 변경 **0건** · 정책 순위 변경 **0건** ·
HE0 **불변** · 대상 파일 **무수정**(규칙 실행은 스크래치 사본에서, 도달가능성 실행은 스크래치 spec/out으로).
감사가 실행한 것: `tc1_rule_rev3.py` 스크래치 재실행(11/11 PASS·sha 대조) · **가드 이후 붕괴 변이 3종 +
mutant 9종 전역 적용 재실행** · `design_reachability.py` **5개 변형 spec 실행**(무력성 증명) ·
사구간/사전확률 산술(200,000 시행) · **치환 대조 시뮬레이션**(수준차 6점 × 시행 120–400 × 치환 300–400) ·
상수 참조 grep · job 896565 서버 로그 재검(0/152) · 코드 정독(`tc1_rule_rev3.py`·`design_reachability.py`·
`presubmit.py`·`multiplexing_mixin.py`) · rev2 규칙 JSON 축 대조 · git 이력 대조(`d11243a`·`9297d2d`) ·
정본 grep(`CONSENSUS §1-7·§1-10`, 항목81/PS게이트 #61).
