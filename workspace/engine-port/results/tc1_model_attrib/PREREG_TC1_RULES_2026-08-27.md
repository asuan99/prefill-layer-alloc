# 사전등록 **TC1 rev1** — 모델 귀속: 이 프로젝트의 negative는 hybrid 때문인가

2026-08-27 · 메인 세션 · ★**규칙층 초안 — 미감사**(게이트 #34 2단 감사의 **1단계 대기**) ·
GPU 지출 **0** · 새 성능 판정 **0건** · 등급 변경 **0건** · 정책 순위 변경 **0건**

규칙 파일: [`tc1_rule.py`](tc1_rule.py) (`RULE_REV=1`, 20,736 세계, 자체검사 18건 전부 PASS,
`rule_sha256 = 598e5ad28523507be8c0927a682c4bc1…`)
· 산출 [`tc1_rule_rev1.json`](tc1_rule_rev1.json)

> ★**이 문서는 제출 승인이 아니다.** 규칙층 감사를 통과하기 전에는 하네스를 쓰지 않고
> job을 제출하지 않는다. 금지 문장: *"TC1이 규칙층을 통과했다"* · *"TC1 설계가 확정됐다"*.

---

## 0. ★인용 규율 — 이 문서를 쓰다 발견한 **게이트 번호 충돌** (신규, 별건 등재 요망)

이 사전등록을 쓰면서 게이트 번호를 전수 검증했고, **저장소에 "방법론 게이트"라는
이름의 번호 체계가 두 개** 있으며 서로 **불일치**한다는 것을 확인했다.

| 체계 | 위치 | 범위 | 예 |
|---|---|---|---|
| **A** | 워크스페이스 루트 `CLAUDE.md` "방법론 게이트 (필수)" | #1–#8 | #3 = n≥4 · #4 = goodput p95 · #6 = metric cliff · #7 = duration 합산 |
| **B** | `PROJECT_STATUS.md` "방법론 게이트" | #1–#80 | #3 = keepalive 설계 · #4 = 집계 단위 · #6 = 독립성 · #7 = 게이트가 항등식인가 |

**두 체계에서 같은 번호가 다른 게이트를 가리킨다.** 그리고 저장소의 사전등록들이
한정어 없이 `게이트 #N`으로 두 체계를 섞어 쓰고 있다.

★**실제 오인용 3건 확인**(전부 이 세션에서 처음 지적):

1. `results/bsweep_regime/PREREG_E1_BSWEEP_REGIME_2026-08-14.md` — `게이트 #3(n≥4)`.
   체계 B의 #3은 keepalive다. (이 사전등록은 규칙층 `NO-GO(F1–F9)`를 받았으나
   **그 감사도 이 오인용을 잡지 않았다**.)
2. `results/nsl_lever/nsl_eb/PREREG_NSL_EB_ATTRIBUTION_2026-08-25.md` — 같은 오인용.
   (이 묶음도 규칙층 `NO-GO`(死因 4건)를 받았고, **역시 잡히지 않았다**.)
3. ★★★**`results/kernel_mech/stage0ppp/stage0ppp_a0_rule.py` 첫 줄** —
   *"THE REGISTERED DECISION RULE, fixed in code (gate #66)"*.
   체계 B의 #66은 *"포크는 …"*(S-6 rev1–rev4, 2026-08-23)이고, 규칙-as-코드 규율은
   **CONSENSUS §3 교훈 항목66**이지 게이트 #66이 아니다.
   ⇒ 이것은 **완주한 실험(A0, job 892556)의 규칙 정본 파일 안**에 있고,
   그 파일의 `rule_sha256`은 `a0_verdict_892556.json`이 고정하고 있다.

★★**그리고 이 문서의 초안이 (3)을 그대로 승계했다** — `tc1_rule.py` rev1 초판이
같은 문장을 복사했다가 이 검증에서 되돌렸다. 즉 **교훈 항목80이 예측한 전파 경로**
(*"출처 허위는 규칙 정본 파일 안이 가장 위험하다 — 다음 판본이 재검증 없이 승계한다"*)가
**실제로 한 세대 만에 발화했다.**

**이 문서의 인용 규약(등록)**: 체계 B(`PROJECT_STATUS.md`)를 기본으로 쓰고,
체계 A를 인용할 때는 **`CLAUDE.md 게이트 #N`**으로 한정어를 붙인다.
그리고 **모든 게이트 인용에 이름을 병기**해 번호가 틀려도 자기교정되게 한다.
번호를 검증하지 못한 규율(변이 테스트 등)은 **번호 없이 이름으로만** 인용한다.

★**별건 등재 요망(이 캠페인 범위 밖)**: (3)의 수정은 A0 규칙 파일을 건드리는 일이고,
그 파일은 **동결 대상**이다(제자리 수정 시 완주 실험의 채점 해시가 무효화된다).
⇒ 코드 수정이 아니라 **doc-steward 판단으로 정오표 배너**를 붙이는 것이 맞다.
`scripts/discipline/check_line_citations.py`는 지문 기반이라 이 오류를 **잡지 못한다**
(줄 내용은 맞고 *번호가 가리키는 대상*이 틀렸다) — 도구 한계 1건 추가.

## 1. 이 캠페인이 존재하는 유일한 이유

`venue_positioning.md` §0.1(2026-07-25)이 지정한 **벡터2의 신규 게이트**를 이행한다. 그 문서는
이 프로젝트의 central negative가 substrate-robustness 축에서 두 갈래라고 판정했다 —
**(A) green-context 종속**(layer-aware 死·cudagraph 비양립, Claim B) vs
**(B) mechanism-independent 후보**(lever-weakness·entanglement·비대칭, Claim A/C).
그리고 **Risk 2(모델 vs substrate 귀속)**를 닫는 값싼 식별 실험으로 다음을 등록했다:

> 순수 Transformer를 기존 pdmux(green-context)에 통과시켜 hybrid와 **같은 green-context +
> 같은 conjunctive-SLO**에서 대조한다. drain 비용은 두 모델에 동일하게 작용 → **상쇄**.

`TC1`은 그 문서 이후 **13개월치 세션이 지나도록 한 번도 실행되지 않았고**, 저장소 전체에서
`PROJECT_STATUS.md` · `CONSENSUS.md` · `CLAIM_EVIDENCE_MATRIX.md` 어디에도 등장하지 않는다
(`EXPERIMENT_ROADMAP.md`의 정의 1건이 전부). 그동안 그 자리에 제출됐던 것은 M3
Transformer-control 대조(job **872077**) 하나뿐인데 **NO VERDICT**로 끝났고
(사유가 "CI 폭 부족"에서 **estimand 미식별**로 확장), ★**그 job이 바로 라벨 희석
(D108 96.2%)이 발견된 job**이다.

⇒ **"이 프로젝트의 negative가 hybrid 때문인가"에 대한 직접 증거는 현재 0건이다.**
그리고 그것은 프로젝트 이름(`prefill-layer-alloc`)과 모델 선택의 정당성이 걸린 질문이다.

### 1.1 ★이 캠페인이 **닫지 못하는** 것 — 먼저 적는다

1. ★★**두 체크포인트의 비교이지 두 아키텍처 계열의 비교가 아니다.** Qwen2.5-3B와
   Zamba2-2.7B는 파라미터 수·층 구성·head 구성·**tokenizer**가 동시에 다르다.
   ★이것은 **C2b를 폐기시킨 바로 그 반론**이다(E4: *"현존 체크포인트로는 통제된 비교가
   불가능하므로 주장 폐기가 정직한 수순 — 추가 실험으로 구제하지 않는다"*).
   **TC1이 그 반론을 피한다고 주장하지 않는다.** 피하는 것은 그중 한 조각뿐이다 —
   C2b는 **arm 간 크기(비)**를 비교했고 TC1은 **모델 내부의 부호**를 비교한다.
   부호 `sign(dynamic − 자기 best static)`는 **모델별 단조 재척도에 불변**이므로
   비(ratio)가 견디지 못하는 종류의 모델 차이를 견딘다. **그러나 면역은 아니다**(§3(f)).
2. **flip이 발화해도 "왜"에는 답하지 않는다.** hybrid 특이성 후보가 최소 둘이고
   TC1은 그 둘을 구별하지 못한다 — (i) 층 타입 구성(창립 가설, 이미 死),
   (ii) ★**admission 용량이 SSM state pool에 묶여 있다**(NSL E-A `ARITHMETIC_CONFIRMED`:
   KV 예산은 cap이 아니라 mamba pool을 따른다 · `CONSENSUS.md` §1-23).
   ⇒ 판정어는 **"모델에 귀속된다"**까지이고 **"층 타입 때문이다"는 금지**한다.
3. **일반화 불가**: 1 hybrid × 1 Transformer × 1 워크로드 × 1 SLO × 1 기판.
   *"hybrid에서는 동적이 진다"* · *"Transformer에서는 동적이 이긴다"* 둘 다 금지.
4. **HE0를 되살리지도 죽이지도 않는다.** HE0는 hybrid arm 내부의 명제이고 이미 n≥4로
   확정돼 있다. TC1이 재현에 실패하면 그것은 **TC1의 측정 실패**로 라벨링한다(게이트 #21 — 측정 실패를 게이트 실패로 라벨링 마라).
5. **Claim B(green-context 종속)를 닫지 않는다.** drain 상쇄 논증은 *두 모델이 같은
   primitive를 쓴다*는 것만 보장하며, 다른 primitive에서 무슨 일이 나는지는 말하지 않는다.

---

## 2. 단 하나의 질문

> **같은 green context · 같은 conjunctive-SLO goodput · 같은 ShareGPT 변화-trace에서,
> `sign(best dynamic − 자기 best static)`이 hybrid arm과 size-matched Transformer arm에서
> 서로 다른가?**

부호가 다르면 flip은 **substrate·메트릭 고정 하에 모델에 귀속**된다.
부호가 같으면 negative는 **메트릭(conjunctive-SLO goodput) + 배포 primitive(green-context)**
탓으로 재프레이밍된다 — 여전히 유효한 기여이지만 **주장의 주어가 바뀐다.**

---

## 3. ★게이트 #40 — 결정량 항등식 사전 점검 (설계보다 먼저)

> 게이트 #40: *"결정량 설계 전 **데이터와 무관하게 참/거짓이 되는 극단 사례가 있는가**를
> 먼저 대수적으로 점검하고 저장소를 grep하라."* — 이 점검에서 **(a)가 설계를 실제로 바꿨다.**

### (a) ★★flip이 **격자 기하학**으로 강제되는가 — **YES. 설계 변경 발생.**

reactive 컨트롤러는 최적점 **주위를 진동**한다. 따라서 어떤 모델의 static argmax가
**격자 끝점**에 있으면, 그 끝점에 못 박은 static은 구조적으로 이길 수밖에 없고
동적은 데이터와 무관하게 진다.

★**정본 Zamba2의 argmax는 `d44`이고, 정본 격자 `{d16,d24,d34,d44}`의 최댓값이다.**
즉 **기존 격자를 그대로 쓰면 hybrid arm의 "동적 패"는 부분적으로 격자가 강제한 것**이고,
Transformer arm의 argmax가 내부에 있으면 flip이 **아키텍처와 무관하게** 발화한다.

**해소**: 격자를 양쪽으로 확장해 **두 모델의 argmax가 모두 내부(interior)**가 되도록 한다.
`pdmux_d{16,24,34,44,54,64,74}.yml`이 이미 저장소에 있다(G16이 만든 상향 확장분).
argmax가 끝점이면 그 방향으로 한 칸 확장해 Stage 1을 재실행하고, 그래도 끝점이면
판정을 **`GRID_EDGE_UNRESOLVED`로 차단**한다(실질 판정 금지).
★**두 모델의 argmax 위치와 내부 여부는 필수 보고 항목**이다.

### (b) `best dynamic > best static`이 **max-over-K 편향**인가 — **YES. 설계 변경 발생.**

정본 격자에서 static은 4 arm, dynamic은 3 arm이다. 귀무에서도 `E[max of 4] > E[max of 3]`
이므로 **비교 자체가 동적에 불리하게 편향**돼 있다(승자의 저주).

**해소**: 양쪽 모두 **사전 지명된 단일 비교자**로 고정하고 **표본 분할**한다.
- 동적 = **`bind+GATE`** 하나. (v7b·bind no-gate는 arm에서 제외 — (c) 참조.)
- static = **그 모델 자신의 Stage 1 argmax** 하나.
- ★**Stage 1(argmax 선정)과 Stage 2(채점)는 서로 다른 rep을 쓴다.** Stage 1 결과는
  Stage 2 제출 **전에** 동결·해시된다. ⇒ 승자의 저주가 채점 표본에 들어오지 않는다.

### (c) 컨트롤러가 **구조적으로 못 움직일** 수 있는가 — **YES. arm 선택이 바뀜.**

v7b는 cudagraph 운영점에서 TPOT가 non-binding이라 prefill 경로가 거의 열리지 않는
**설계 결함**이 이미 확정돼 있다(`CONSENSUS.md` §1). 그 arm을 쓰면 `win=false`가
**데이터와 무관하게** 나온다.

**해소**: (i) 동적 arm은 그 결함이 수정된 **`bind+GATE`(Step E+F+G)** 하나로 고정한다.
(ii) 그래도 못 움직일 수 있으므로 **관측 가능한 술어**를 등록한다 —
`CONTROLLER_DEGENERATE`가 발화하는 조건:

```
switch_count == 0
  OR  체류 분포가 단일 split index에 100% 집중
  OR  결정 호출 중 두 urgency 분기(pf_slack < PF_URGENCY, dec_slack < 1−TPOT_HI)가
      한 번도 발화하지 않은 비율 ≥ 0.99
```
발화 시 그 모델의 부호는 **판정 불가**이며 flip을 차단한다.

### (d) 공통 off-cliff 밴드가 **공집합**일 수 있는가 — 가능. `NO_COMMON_BAND`.

### (e) 메트릭 절벽(CLAUDE.md 게이트 #6) — 가능. `CLIFF_STRADDLE`.
어느 arm에서든 HI phase의 **TTFT p90이 SLO 3 s를 가로지르면** goodput은 ill-posed다
(§S10: 3% 섭동이 2× 신호로 증폭). 용량을 먼저 재고 off-cliff에서 측정한다.

### (f) ★rate를 **절대값으로 맞추면** flip이 부하 아티팩트가 된다 — **설계 변경 발생.**

3B Transformer와 2.7B hybrid는 용량이 다르다. 같은 절대 rate는 **다른 부하 분율**을 뜻하고,
최적 split은 부하의 함수(`CONSENSUS.md` §1-5)이므로 부호가 **부하 때문에** 갈릴 수 있다.

**해소**: ★**rate는 λ가 아니라 λ/용량으로 맞춘다.** Stage 0(용량 스캔)이 모델별 용량
`C_M`을 재고, LO/HI를 `(λ_LO/C_M, λ_HI/C_M)` 고정 분율로 등록한다.
정본 Zamba2 격자의 `(3, 12)`가 기준이고 용량 ≈6.3 req/s이므로 분율은
**LO ≈ 0.48·C, HI ≈ 1.90·C**로 등록한다. Qwen arm의 절대 rate는 그 분율에서 파생된다.

### (g) 저장소 grep — 이 결정량이 이미 반증된 적 있는가

`flip` · `모델 귀속` · `TC1` · `Transformer-control`로 전수 검색했다. 선행 판정은
M3(872077) **NO VERDICT** 하나뿐이며 **estimand 미식별**이 사유였다(라벨 희석).
그 사유는 `PDMUX_STICKY_PARTITION`(구현 완료·correctness gate PASS)으로 해소 가능하다 —
§8 하네스 요건에 **sticky ON 필수**로 등록한다.

---

## 4. Estimand — 정본 하네스에서 **축자 복사** (새로 정의하지 않는다)

출처: `results/slo_sched/sharegpt_vary_bench.sbatch:82-96`.

```python
dur += d                                                    # :92  라운드 duration 합산(CLAUDE.md 게이트 #7 — duration 합산)
m = (sum(I[i])/len(I[i])) if (i<len(I) and I[i]) else 9.0    # :94  요청 내부 MEAN ITL (초)
if t <= 3.0 and m <= 0.06: g += 1                           # :96  conjunctive 술어
goodput = sum(g over rounds) / sum(durations)
```

### 4.1 ★정본 하네스가 CLAUDE.md 게이트 #4(goodput = TTFT ∧ ITL **p95**)를 문자 그대로는 어긴다 — 그리고 이 캠페인이 그것을 **공짜로** 닫는다

**CLAUDE.md 게이트 #4**는 goodput을 *"request TTFT ≤ SLO **AND** request-내부 token-ITL **p95** ≤ SLO"*로
정의하는데, `:94`는 **mean**을 쓴다. 이 진단은 저장소가 **세 번** 냈고 가장 최근이 NSL-1 §7이다.
정본 판정(HE0)은 세 갈래 확인으로 흔들리지 않지만, **TC1은 이것을 GPU 0으로 닫을 수 있다** —
`--output-details`가 요청별 `itls` 배열을 **원자료로 보관**하므로 같은 런에서 두 채점을
전부 재계산할 수 있다.

**등록**:
- **1차 채점 = p95**(CLAUDE.md 게이트 #4 준수). `m_p95 = percentile(I[i], 0.95)`.
- **2차 채점 = mean**(정본 HE0 수치와의 **비교가능성 다리**로만).
- ★두 채점이 **부호 판정에서 어긋나면** 라벨은 `SCORING_DEPENDENT`이고 **실질 판정을
  차단**한다 — 저자가 데이터를 보고 고르는 것을 구조적으로 막는다(NSL 감사 G1의 실패 모드).
- ★**금지 문장**: *"TC1이 p95 SLO로 튜닝된 컨트롤러를 쟀다"*. 컨트롤러의 내부 신호는
  `TPOT-EMA`(평활 평균)이고 코드 변경 없이 p95-aware로 만들 수 없다. 이 성질은
  **두 모델에 동일하게** 걸리므로 부호 비교는 성립하지만, 튜닝 주장은 성립하지 않는다.

### 4.2 per-model 부호 판정 (CLAUDE.md 게이트 #3 · PROJECT_STATUS 게이트 #14)

```
Δ_M = goodput(bind+GATE, M) − goodput(argmax-static, M)
win(M)  ⟺  Δ̂_M ≥ δ_M  ∧  t-CI(95%, n−1)가 0을 배제
lose(M) ⟺  ¬win(M)  ∧  power_M = adequate
tie(M)  ⟺  그 외
δ_M = 0.03 × goodput(argmax-static, M)          # CLAUDE.md 게이트 #3
```
★**t-CI를 쓴다. percentile bootstrap 금지** — 게이트 #14(통계 방법 층 정정)가 n=4에서 coverage 0.798,
한쪽 오류율 ≈10%를 확인했고, 이 결정 규칙은 단방향이라 그 undercoverage가
**한 방향으로만** 편향된다.

**검정력(등록)**: `power_M = adequate ⟺ MDE(80%) ≤ 2 × δ_M`.
설계 시점 산출(H arm, 사전 SD 0.023, base 3.220):
`δ = 0.0966` · `t·se = 0.0366` ⇒ **δ가 구속** · `P(win | Δ=0) ≈ 0` ·
`MDE(80%) = 0.1063 = 3.30%`. ⇒ H arm은 n=4로 **검정력 충분**.
★**T arm의 SD는 미측정이다** — Stage 1이 재서 `power_T`를 채운다. 채워지기 전에는
`UNDERPOWERED_NULL`이 기본값이다.

---

## 5. 격자 · 설계 — 3단계, 각 단계가 다음 단계를 잠근다

| 단계 | 무엇 | 격자 | rep | 산출 | 잠그는 것 |
|---|---|---|---|---|---|
| **0** 용량 | 모델별 off-cliff 용량 `C_M` | rate 스캔 | 2 seed | `C_H`, `C_T`, TTFT p90 곡선 | LO/HI 절대 rate (§3(f)) · `CLIFF_STRADDLE` |
| **1** static 스크린 | 각 모델의 static argmax + SD | `{d16,d24,d34,d44,d54}` × 2 모델 | **n=3** | argmax 위치·내부 여부·per-cell SD | `PDMUX_SLO_ANCHOR_IDX` · `δ_M` · `power_M` · `GRID_EDGE_UNRESOLVED` |
| **2** 채점 | 부호 판정 | {argmax-static, bind+GATE} × 2 모델 | **n=4 (신규 rep)** | `Δ̂_M`, t-CI, switch/dwell | 최종 라벨 |

- ★**Stage 1 결과는 Stage 2 제출 전에 동결·SHA-256 해시**한다. Stage 2는 그 해시를
  아티팩트에 기록한다. (표본 분할 — §3(b))
- ★Stage 1에서 argmax가 격자 끝점이면 **그 방향으로 한 칸 확장**(d64/d74 또는 d8 신규)해
  재실행. 2회차에도 끝점이면 `GRID_EDGE_UNRESOLVED`로 차단한다.
- `bind+GATE`의 anchor는 **그 모델 자신의 Stage 1 argmax**로 등록한다(모델별로 다를 수 있다).
  이것은 SLO 쇼핑이 아니다 — 두 arm에 **동일한 규칙**을 적용하고, 그 규칙이 참조하는
  데이터는 채점 표본과 **분리**돼 있다.

**공통 고정**: `ctx 4096`(Zamba2 상한에 맞춤 · ShareGPT는 98%가 L<2000이라 비구속) ·
`--max-running-requests 48` · `--disable-radix-cache` · `--mem-fraction-static 0.82` ·
`--chunked-prefill-size -1` · `--disable-overlap-schedule` · `--attention-backend triton` ·
**cudagraph ON**(운영점) · ShareGPT v3 `--sharegpt-context-len 4000` · `ROUNDS=3` · `NP=200`.

---

## 6. 결정 규칙 — **코드로 고정** (규칙-as-코드 규율 · §0 참조)

[`tc1_rule.py`](tc1_rule.py) `RULE_REV=1`. **12축 20,736 세계 전수 열거**, 자체검사 17건 PASS.

**라벨(전 12종, iff-분할)**

| 계층 | 라벨 | 뜻 |
|---|---|---|
| 측정 | `MEASUREMENT_ABSENT` · `NO_COMMON_BAND` · `CLIFF_STRADDLE` | ★게이트 #21 — **게이트 실패가 아니라 측정 실패** |
| 설계 | `GRID_EDGE_UNRESOLVED` · `CONTROLLER_DEGENERATE` | 설계상 도달 불가 |
| 채점 | `SCORING_DEPENDENT` | p95와 mean이 부호에서 어긋남 |
| 실질 | `INCONCLUSIVE_TIE` · `UNDERPOWERED_NULL` · **`FLIP_MODEL_ATTRIBUTED`** · `REVERSE_FLIP` · `NO_FLIP_BOTH_WIN` · `NO_FLIP_BOTH_LOSE` | |

**가드 순서도 등록한다** — `(boot, band, cliff, grid, degen, scoring)`.
★자체검사 M5가 이 순서가 **1,728개 라벨을 좌우한다**는 것을 보인다(grid↔degen 교환).
A0 rev4의 死因 N1이 *"분기 순서가 무등록 자유 모수였다"*였으므로 여기서는 명시 등록한다.

**검정력은 null을 지는 라벨만 게이트한다** — `FLIP`은 `power_H`만, `REVFLIP`은 `power_T`만,
`BOTHLOSE`는 둘 다, `BOTHWIN`은 게이트하지 않는다(승리는 검정력 논증이 필요 없다).
자체검사 T6가 이 대응을 **양방향으로** 검증한다.

> ⚠️★**세계 개수는 확률이 아니다.** `tc1_rule_rev1.json`의 `marginals`는 축 격자의 산물이며
> (`MEASUREMENT_ABSENT` 10,368 vs `NO_FLIP_BOTH_LOSE` 1), 어떤 결과가 나올 가능성을
> 뜻하지 않는다. NSL 감사 D3c가 **격자 인공물을 근거로 검사를 삭제**한 사례를 낸 바 있다.

---

## 7. 대조 — 음성 · 양성 (게이트 #9 항등식 계열 · #44 빈 서명 구멍)

### 7.1 음성 대조 — ★**GPU 0, 지금 실행 가능**
정본 Zamba2 변화-trace 반복(n≥4)을 **rep 인덱스로 두 유사-arm(H_a, H_b)으로 분할**해
같은 flip 규칙을 돌린다. **`FLIP`·`REVERSE_FLIP`이 발화하면 안 된다.**
규칙층 자체검사 **T5**가 이 성질을 세계 공간에서 이미 증명했다(같은 부호 6,912 세계, 귀속 0건).
실데이터 버전은 **귀무 발화율의 실측**을 준다 — 게이트 #24(`any()` over n reps의 귀무 발화율 `1−(1−α)ⁿ`)에 걸리지 않게, 가능한 **모든 분할**에 대해 발화율을 보고한다.

### 7.2 양성 대조 — 규칙이 flip을 **실제로 볼 수 있는가**
H arm 실데이터에 `Δ_T := Δ_H + 2δ`를 주입한 합성 T arm을 만들어 규칙에 넣는다.
`FLIP_MODEL_ATTRIBUTED`가 **반드시** 발화해야 한다.
★**빈 서명 구멍 차단**(게이트 #44): 대조는 estimand를 **실제로 실행**해야 한다 —
`itls` 배열이 비어 있으면 통과가 아니라 **실패**로 채점한다.
★**변이 테스트**(교훈 항목53 — 게이트 번호 없음): `δ`를 `2δ`로 되돌린 변이본에서 이 대조는 **반드시 실패**해야 한다.

---

## 8. 하네스 요건 (2단계 감사 대조표 — 목록만, 코드 없음)

1. ★`sharegpt_vary_bench.sbatch`의 `MODEL`/`CTX`를 **인자화**한다(현재 Zamba2 하드코딩).
   ⇒ **TC0는 "완료"가 아니라 "부분 완료"다** — Qwen2.5-3B는 `stage0_xctrl`에서 pdmux로
   부팅된 선례가 있으나(`stage0_pdmux_capture.sbatch:89`), **정본 변화-trace 하네스에서는
   한 번도 돌지 않았다**. 이 정정을 등재한다.
2. ★**`PDMUX_STICKY_PARTITION=1` 필수** — M3(872077)의 estimand 미식별(라벨 희석)이
   재발하지 않도록. **realized 파티션 체류분포를 셀마다 보고**하고 ≥0.90 미달이면
   `MEASUREMENT_ABSENT`.
3. **p95·mean 두 채점을 같은 jsonl에서 산출**하는 분석기(§4.1). 사전등록 분석기가
   **결정 규칙의 모든 항을 계산**해야 한다(게이트 #20 — 계산되지 않는 비교는 사후 비교다).
4. `switch_count` + **split 체류분포** + TTFT/ITL **p50/p95/p99** 병기(CLAUDE.md 게이트 #5).
5. **모델별 `input_lens` 분포를 보고**한다 — tokenizer가 다르므로 같은 *텍스트*가 다른
   *토큰 수*가 된다. 통제 대상이 아니라 **필수 공변량**이다.
6. ★**`--max-mamba-cache-size`는 Zamba2 arm에만 존재한다.** NSL E-A와
   `CONSENSUS.md` §1-23 따름정리에 따라 **`= cap` 규칙**으로 고정한다(절대상수 금지 —
   slot당 비용이 arm마다 달라 새 cross-arm 교락이 된다). Qwen arm에는 해당 항목이 없으며,
   ★이 비대칭은 **제거 대상이 아니라 보고 대상**이다(§1.1-2의 hybrid 특이성 후보 (ii)).
7. ★**두 모델을 같은 배치·같은 트리·같은 telemetry 설정으로 실행**한다.
   게이트 #41(telemetry는 공짜 관찰자가 아니다)에 따라 **2026-07 Zamba2 수치 재사용 금지**.
8. `sync_engine_tree.sh` manifest 해시를 아티팩트에 기록. `--comment="field=efficientai;appl=pytorch"`.

---

## 9. 비용 · 스모크 (게이트 #26 배관 스모크 · #25 진단 필드 미산출)

**스모크 먼저 (제출 조건)** — 1 job, ≈0.3 GPU-hr, ★**채점 판정 0건**:
Qwen2.5-3B가 인자화된 vary 하네스에서 pdmux + cudagraph-ON으로 부팅되고,
`--output-details` jsonl의 `itls`가 비어 있지 않고, `SLO-BIND` 로그가 찍히고,
sticky realized 체류가 산출되는지. ★**그리고 셀당 실제 wall time을 측정**한다.

**본 캠페인(추정)**: Stage 0 ≈4 job · Stage 1 = 2×5×3 = 30 job · Stage 2 = 2×2×4 = 16 job
= **50 job**. 하네스 상한은 `--time=00:50:00`이므로 최악 ≈**42 GPU-hr**,
실측 wall(≈15–20분/job) 기준 ≈**13–17 GPU-hr**. ★**이 폭은 그대로 두지 않는다** —
NSL 감사 G5가 *"비용 3.18배 과소, 감사가 §7이 채우라 한 상수를 직접 측정했다"*를 낸 바 있으므로,
**Stage 2 예산은 스모크가 잰 셀당 wall time에서 재도출**한다. 스모크 전에는 제출하지 않는다.

---

## 10. 손잡이 사전 등재 (게이트 #19 — 사후 지정 셀 이동 방지)

| 손잡이 | 값 | 근거 |
|---|---|---|
| `PDMUX_SLO_MODE` | `binding` + `PDMUX_SLO_FEAS_GATE=1` | §3(c) — v7b 설계 결함 회피 |
| `PDMUX_SLO_ANCHOR_IDX` | **모델별 Stage 1 argmax** | §3(b) 표본 분할 |
| `PDMUX_TPOT_SLO_MS` / `PDMUX_TTFT_SLO_MS` | `60` / `3000` | 정본 E3-vary와 동일 |
| `PDMUX_SLO_FEAS_OCC` / `_MARGIN` / `_DWELL` / `_EMA` | `0.85` / `0.9` / `3` / `0.85` | 엔진 기본값 — **모델별 튜닝 금지** |
| `PDMUX_STICKY_PARTITION` | `1` | §8-2 |
| `max_running_requests` | `48` | 정본 고정. ★**이 캠페인의 손잡이가 아니다** |
| `--max-mamba-cache-size` | `= cap` (Zamba2만) | §8-6 |
| rate LO / HI | `0.48·C_M` / `1.90·C_M` | §3(f) — λ가 아니라 λ/용량 |
| `N_REPS` / `DELTA_REL` | `4` / `0.03` | CLAUDE.md 게이트 #3 |

---

## 11. 금지 문장 (등재)

- *"TC1이 규칙층을 통과했다"* · *"TC1 설계가 확정됐다"* (감사 전)
- *"hybrid에서는 동적이 진다"* · *"Transformer에서는 동적이 이긴다"* (1×1 체크포인트)
- *"flip이 층 타입 때문이다"* (§1.1-2 — 후보가 최소 둘)
- *"TC1이 p95 SLO로 튜닝된 컨트롤러를 쟀다"* (§4.1)
- *"TC0는 완료됐다"* (§8-1 — 정본 하네스에서 미실행)
- *"세계 개수가 결과 가능성을 뜻한다"* (§6)
- *"TC1이 C2b의 반론을 피한다"* (§1.1-1 — 피하는 것은 한 조각뿐)

---

## 12. 1단계(규칙층) 감사에 묻는 것 — **단 하나** + 합격 기준 (게이트 #34 2단 감사 · #37 합격 기준+단일 판정 질문)

> ★**판정 질문**: *"이 결정 규칙은, **어떤 데이터가 오든**, 저자가 사후에 고를 여지를
> 남기지 않는가? 그리고 §3(a)–(g) 중 **놓친 항등식·강제 분기**가 있는가?"*

**합격 기준(감사가 이것들만 보면 된다)**:
1. `tc1_rule.py`의 12축이 실제 실험이 만들 수 있는 세계를 **전부** 덮는가.
   덮지 못하는 축이 있으면 그것이 死因이다.
2. §3의 항등식 점검 (a)–(g)에 **빠진 것**이 있는가. 특히 *"데이터와 무관하게
   `FLIP` 또는 `NO_FLIP_BOTH_LOSE`가 강제되는 극단 사례"*.
3. 가드 순서(§6)가 **답을 미리 정하지** 않는가. M5가 1,728 라벨을 좌우함을 감안해
   등록된 순서가 정당한가.
4. §7의 두 대조 중 어느 하나라도 **항등식**이거나 **빈 서명으로 통과 가능**한가
   (게이트 #9 — 이 저장소에서 14회 재발).
5. §0의 게이트 번호 충돌 보고가 **정확한가**(감사가 직접 대조해 주기 바란다).
6. §1.1의 "닫지 못하는 것" 5건이 **충분히 정직한가** — 특히 §1.1-1(C2b 반론)이
   과소 서술됐는가.

**감사가 답할 필요 없는 것**: 비용·일정·하네스 구현(2단계 소관) · 어느 학회에 맞는가.
