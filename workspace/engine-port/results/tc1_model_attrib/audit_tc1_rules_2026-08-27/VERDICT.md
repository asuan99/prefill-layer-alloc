# 감사 판정서 — TC1 rev1 규칙층 (게이트 #34 1단)

2026-08-27 · claims-auditor · 적대 감사 · **읽기 전용**(대상 파일 무수정) ·
GPU 지출 **0** · 새 성능 판정 **0건** · 등급 변경 **0건** · 정책 순위 변경 **0건**

대상: `PREREG_TC1_RULES_2026-08-27.md` · `tc1_rule.py`(`RULE_REV=1`, sha256
`598e5ad28523507b…`, 20,736 세계) · `tc1_rule_rev1.json`

> ★인용 규약(이 판정서): 체계를 **항상 이름과 함께** 표기한다 —
> `CLAUDE.md #N` / `PS게이트 #N` / `CONSENSUS §3 항목N` / `메모리항목 N`.
> 사전등록 §0이 두 체계만 셌으나 실제로는 **넷**이다(아래 §5).

---

## 판정: **NO-GO** (死因 4 · 차단 18)

**단일 판정 질문에 대한 답**: **아니오.** 이 규칙은 어떤 데이터가 오든 저자에게
사후 선택지를 남긴다. 최소 세 곳에서 명시적으로 남긴다 — (F3) 하네스가 내는
goodput 3종 중 어느 것이 estimand인지 미등록, (§1.1-4 vs 코드) H가 이기면
"측정 실패"인지 "중심 negative가 흔들림"인지 미결, (B5) 검정력 부족이
`INCONCLUSIVE_TIE`인지 `UNDERPOWERED_NULL`인지 미결.
그리고 §3(a)–(g)는 **데이터와 무관하게 `MEASUREMENT_ABSENT`와
`NO_FLIP_BOTH_LOSE`를 강제하는 극단 사례 두 개를 놓쳤다**(F1·F2).

---

## 1. 死因 (설계를 죽이는 것 — 수리 = 재설계)

### F1. 등록된 필수 손잡이 둘이 엔진에서 **상호 배타** ⇒ 동적 다리가 부팅을 못 하고, 우회해도 `MEASUREMENT_ABSENT`가 강제된다

사전등록은 두 가지를 **동시에 필수**로 등록했다.

- §8-2 · §10: `PDMUX_STICKY_PARTITION=1` **필수**(M3 872077 라벨 희석 재발 방지).
- §3(b) · §3(c) · §10: 동적 arm = **`bind+GATE` 하나**(`PDMUX_SLO_MODE=binding`).

코드 사실:

```
results/slo_sched/sharegpt_vary_bench.sbatch:39
  bind) CFG=...; export PDMUX_SLO_SCHED=1 PDMUX_SLO_MODE=binding ...

src/multiplex/multiplexing_mixin.py:298-302
  if os.environ.get("PDMUX_SLO_SCHED"):
      raise RuntimeError(
          "PDMUX_STICKY_PARTITION with PDMUX_SLO_SCHED is undefined: the "
          "SLO controller only decides on prefill spans, and extending "
          "its decisions into decode-only spans changes its dynamics")
```

⇒ **sticky ON + bind = 스케줄러 init에서 `RuntimeError` = 부팅 실패 = `MEASUREMENT_ABSENT`.**

우회(sticky OFF)도 막힌다:

```
src/multiplex/multiplexing_mixin.py:906-914
  if (os.environ.get("PDMUX_SLO_SCHED") and not self.running_batch.is_empty()
      and self.split_prefill_batch):        # ← 컨트롤러는 prefill span에서만 결정
      ...
  if not self.running_batch.is_empty() and (
      self.split_prefill_batch or self.sticky_partition_enabled):  # ← OFF면 앞항만
```

sticky OFF면 decode-only 구간이 **무분할 108 SM**으로 떨어진다. 정본이 이미
크기를 쟀다: `CONSENSUS.md` §1-25 — decode 축 realized 라벨은 **4–19%**
(T8 d16 0.038 → d54 0.093 / Ha8 0.104 → 0.187). §8-2가 등록한 문턱은 **≥0.90**.

**⇒ 삼도논법: sticky ON → 부팅 실패 → `ABSENT` / sticky OFF → 체류 0.04–0.19 <
0.90 → `ABSENT`. 양쪽 다 `MEASUREMENT_ABSENT`이며 데이터와 무관하다.**

§3(g)는 M3의 estimand 미식별이 *"`PDMUX_STICKY_PARTITION`(구현 완료·correctness
gate PASS)으로 해소 가능"*하다고 적었다. 그 grep은 **플래그의 존재**만 확인했고
**그 플래그가 컨트롤러 arm에 대해 정의되지 않았다는 코드 사실**을 확인하지 않았다.
그 사실은 이 저장소가 직접 쓴 예외 메시지 안에 문장으로 들어 있다.

confound 대응: **라벨≠실현**(Stage 0 D108 · M3 872077의 세 번째 판본) +
**게이트가 항등식**(PS게이트 #40 계열: 결정량이 데이터와 무관하게 고정).

재현: 위 3개 파일:줄. 비용 0.

---

### F2. 동적 arm의 anchor를 static argmax에 못 박으면 대조가 붕괴 — `NO_FLIP_BOTH_LOSE`가 **기전적으로** 강제된다

- §5 · §10 등록: `PDMUX_SLO_ANCHOR_IDX` = **그 모델 자신의 Stage 1 argmax**.

컨트롤러의 등록된 거동(`multiplexing_mixin.py:809-822`):

```
if _dwell > 0:            _new = _idx                       # 체류 중 = 이동 없음
elif _hold:               _new = idx±1 → _anchor 로 수렴     # 포화 latch(=20 step)
elif pf_slack < dec_slack and pf_slack < 0.5:  prefill-ward  # urgency 1
elif dec_slack < pf_slack and dec_slack < 0.15: decode-ward  # urgency 2
elif _idx != _anchor:     _new = idx±1 → _anchor 로 drift
else:                     _new = _idx                       # anchor에서 hold
```

`_idx`는 `_anchor`로 초기화된다(`:777`). 그리고 등록된 두 부하점 모두에서
컨트롤러는 anchor를 떠나지 않는다:

- **HI = 1.90·C**(§3(f), 과부하): `_sat_fb = (pf_slack < −0.5) or (dec_slack < −0.5)
  or (pf_slack < 0 and dec_slack < 0)` — 즉 pf_age>4.5s **또는** TPOT>90ms **또는**
  (pf_age>3s ∧ TPOT>60ms). 용량의 1.9배에서 상시 참 ⇒ latch ⇒ **anchor 고정**.
- **LO = 0.48·C**: pf_slack≈0.96 > 0.5, dec_slack≈0.33 > 0.15 ⇒ urgency 둘 다
  미발화 ⇒ `_idx == _anchor` ⇒ **anchor 고정**.

**정본이 이 기전을 이미 측정했다.** `CONSENSUS.md` §1-10:

> bind+GATE **3.132 (n=9)** ≈ **d34-static 3.171 − 0.039(정착 비용)**.
> ★그런데 **틀린 static으로 수렴** — 최적은 d44(3.220).

즉 HE0의 격차 0.088 = **0.049(anchor≠argmax, "틀린 static으로 수렴")** +
**0.039(정착 비용)**. 정본 bind는 anchor 기본값 idx2에서 돌았다
(`lff_bench.sbatch:47` `PDMUX_SLO_ANCHOR_IDX:-2` "anchor idx2=d24").

⇒ **anchor = argmax로 바꾸면 0.049 성분이 설계로 제거되고 Δ ≈ −0.039 =
−1.2%가 남는다. δ = 3%이므로 두 모델 모두 `win=false` ⇒ `lose` ⇒ `BOTHLOSE`.**

FLIP이 뜨려면 Transformer의 컨트롤러가 **자기 anchor보다 +3% 이상** 이겨야 하는데,
등록된 손잡이는 컨트롤러를 anchor에 앉히는 방향으로만 작동한다.

부수 피해: TC1의 H arm은 **HE0가 측정한 arm이 아니다**(다른 anchor). 그런데 §7.1·
§7.2 두 대조는 **정본 anchor 데이터**로 보정된다 ⇒ confound #10(변수 동시 변경).

★§3(a)는 "격자 기하학이 동적을 지게 만든다"를 걱정해 격자를 넓혔는데, 그
처방(anchor=argmax)이 **더 강한 강제**를 새로 만들었다. 항등식 점검이 자기
처방을 다시 점검하지 않은 사례(CONSENSUS §3 항목99 "de-confound 처방 자체가
추정량의 자유 모수일 수 있다"의 재발).

재현: `CONSENSUS.md` §1-10 · `multiplexing_mixin.py:777,809-822` ·
`lff_bench.sbatch:47` · 사전등록 §5 표 마지막 bullet, §10 2행.

---

### F3. estimand 미지정 — 정본 하네스는 goodput을 **세 개** 내는데 규칙은 어느 것인지 등록하지 않았다

```
results/slo_sched/sharegpt_vary_bench.sbatch:100-102
  gpL=gL/dL ; gpH=gH/dH ; gpC=(gL+gH)/(dL+dH)
  print(f"SGPTV_RESULT ... | LO gp={gpL:.3f} | HI gp={gpH:.3f} | COMBINED={gpC:.3f}")
```

사전등록 §4는 `:92` `:94` `:96` 세 줄만 축자 복사하고 `goodput = sum(g)/sum(dur)`
라고만 적는다. **LO / HI / COMBINED 중 어느 것이 `Δ_M`의 피연산자인지 미등록.**

이 선택이 결과를 뒤집은 정본 선례가 있다 — `CONSENSUS.md` §1-19(R1 재분석):
phase-mean **+2.28%** vs pooled trace-level **+5.88%**(2.6×), 그리고 정본 술어로
재채점하면 phase B가 전 arm 0/192라 **오라클 자체가 미정의**.

이것은 **PS게이트 #4(집계 단위를 먼저 정하고 추정 대상과 맞는지 논증하라)** 의
정면 위반이다. 게다가 §3(f)가 LO/HI를 모델마다 **다른 절대 rate**로 잡으므로
pooled를 고르면 duration 몫(=phase 가중)이 arm마다 달라져 두 arm의 estimand가
서로 다른 양이 된다.

**⇒ 저자는 데이터를 본 뒤 세 값 중 고를 수 있다. 이것 하나만으로 단일 판정
질문의 답은 "아니오"다.**

재현: 위 sbatch 줄 · `CONSENSUS.md` §1-19 · 사전등록 §4 전체.

---

### F4. §1.1-1의 방어("부호는 단조 재척도에 불변")가 **대수적으로 거짓**이고, 판별력 축이 규칙에 없다

§1.1-1은 C2b 반론을 피하는 근거로 이렇게 적는다:

> 부호 `sign(dynamic − 자기 best static)`는 **모델별 단조 재척도에 불변**이므로
> 비(ratio)가 견디지 못하는 종류의 모델 차이를 견딘다.

estimand는 **임계 지시함수**다(`t ≤ 3.0 and m ≤ 0.06`). 절대 임계는 재척도
**뒤에** 적용되므로 부호는 단조 재척도에 불변이 **아니다**:

- 모델의 지연 분포가 통째로 임계 아래면 → `goodput ≡ throughput` → 모든 arm 동률
  → `Δ ≡ 0`. 정본이 이 상태를 이미 명명했다: *"위반 0건 술어의 goodput은
  throughput의 다른 이름 — 판별력 0"*(`CONSENSUS.md` §1-1, PS게이트 #6의 사례).
- 통째로 임계 위면 → `goodput ≡ 0` → 역시 `Δ ≡ 0`.
- 부호가 의미를 갖는 것은 **중간 판별 구간뿐**이고, 두 모델이 그 구간에 동시에
  있다는 보장이 설계 어디에도 없다.

★§3(f)의 λ/C 정규화가 이 문제를 **악화**시킨다. 같은 이용률 ρ에서 서비스 시간이
긴 모델의 대기시간이 길므로, 고정 3s 임계 대비 위치가 두 모델에서 체계적으로
다르다. 즉 rate 정규화는 부하 분율을 맞추면서 **임계 대비 위치를 어긋나게 한다.**

그리고 `tc1_rule.py`의 `cliff` 축은 `("clean","straddle")` **2상태뿐**이다.
"전 요청 통과"(판별력 0)와 "전 요청 실패"(판별력 0)에 해당하는 세계가 **축에 없다.**
⇒ 합격기준 1이 말한 "덮지 못하는 축" = 死因. `BOTHLOSE`가 뜰 때 그것이
"두 모델 다 동적이 못 이겼다"인지 "술어가 아무것도 재지 않았다"인지 **구별 불가**.

재현: `sharegpt_vary_bench.sbatch:96` · `tc1_rule.py:72` · `CONSENSUS.md` §1-1 ·
사전등록 §1.1-1 · §3(e) · §3(f).

---

## 2. 차단 (수리 가능 — 번호는 수리 순서 아님)

**B1 · sticky/realized 축 부재.** §8-2가 *"realized 체류 <0.90 → `MEASUREMENT_ABSENT`"*
를 **산문으로만** 등록했고 `AXES`에 `sticky_H`/`sticky_T`가 없다. 이것은 선행 실험
M3(872077)를 죽인 바로 그 confound다. (F1을 고쳐도 이 축은 별도로 필요.)
재현: `tc1_rule.py:66-79` vs 사전등록 §8-2.

**B2 · `boot` 축이 전역 단일.** arm/모델별 부팅 실패(정적은 살고 동적만 죽는 세계)를
표현할 수 없다. F1이 실제로 만드는 세계가 정확히 그것이다.
재현: `tc1_rule.py:67`.

**B3 · `grid` 축이 §3(a)/§5의 "한 칸 확장 후 재실행"을 모델링하지 않는다.**
코드는 edge면 즉시 `GRIDEDGE`(`:101-102`)인데 산문은 1회 확장을 허용한다. 확장
횟수·방향·확장 후 `δ_M`/`power_M` 재계산이 미등록 ⇒ 사후 자유. 추가로 **하향 확장
자산이 없다**: 정본 ladder(`results/slo_sched/`)에는 d16–d74만 있고 d08은
`results/cudagraph_probe/`에만 존재한다 ⇒ `edge_lo`는 구조적으로 해소 불가.
재현: `find . -name "pdmux_d*.yml"` · 사전등록 §3(a)·§5 두 번째 bullet.

**B4 · §3(a)의 전제가 정본과 충돌.** *"정본 Zamba2의 argmax d44는 정본 격자의
최댓값"* 은 **4-arm 격자 한정**이다. G16(`CONSENSUS.md` §1-33, jobs 884336/884410/
884411/884412)이 **7-arm 확장 격자**에서 이미 측정했다: HI `M_ttft` argmin =
**내부점 d44**(4/4 블록·부트스트랩 1.000), goodput argmax도 d44 **3.7973** vs d64
3.7752. ⇒ H arm의 "격자 끝점 강제" 걱정은 정본이 이미 답했는데 사전등록이 인용하지
않고 Stage 1로 재구매한다. 부수: `GOODPUT_BASE["H"]`를 §1-7의 3.220으로 잡느냐
G16의 3.7973으로 잡느냐에 따라 δ가 0.0966↔0.1139로 움직이고, §7.2 양성대조가 바로
거기 걸린다(B11).

**B5 · `UNDERPOWERED_NULL`이 실험에서 도달 불가 — power 게이트 전체가 장식이고,
저자에게 라벨 선택권을 준다.**
§4.2가 `lose(M) ⟺ ¬win(M) ∧ power_M = adequate`, `tie(M) ⟺ 그 외`로 정의하므로
`(sign="lose", power="inadequate")`는 **정의상 공집합**이다. 그런데
`tc1_rule.py:113-122`는 정확히 그 조합에서만 `UNDERPWR`를 낸다.
⇒ (i) `UNDERPOWERED_NULL`은 실제 실험에서 절대 안 뜬다. (ii) `T6_power_gates_nulls_only`
와 `M4_power_not_hollow`(7 worlds)는 **도달 불가 세계에서만 참인 성질**을 검증한다
= 게이트 #9(항등식) 계열. (iii) 같은 실제 조건(검정력 부족)이 `INCONCLUSIVE_TIE`로도
갈 수 있으므로 **저자가 TIE와 UNDERPOWERED_NULL 중 고를 수 있다.**
재현: 사전등록 §4.2 3줄 ↔ `tc1_rule.py:107-122`.

**B6 · `lose`가 "유의하게 나쁨"과 "증거 없음"을 합친다 ⇒ §2의 branch-B 해석을
지지하지 않는다.** §2는 부호가 같으면 *"negative는 메트릭+배포 primitive 탓으로
재프레이밍"* 한다 — 그건 **T의 동적도 실제로 진다**를 요구한다. 규칙의
`lose = ¬win`은 *"3% 이상 이긴 증거가 없다"* 일 뿐이다. 등가성 검정(TOST)도, 음의
방향 유의성도 등록되지 않았다. ⇒ **결정량이 등록된 해석을 떠받치지 않는다.**

**B7 · `MDE_SLACK = 2.0`이 무근거 상수이고 branch-B를 게이트하는 유일한 문턱이다.**
실측(감사 재실행): `power_adequate`는 CV ≤ 2%까지 통과하고 CV 3%에서 실패한다
(δ=3% 기준 MDE 4.02% vs 6.04%). 즉 *"3% 문턱으로 귀무를 채택"* 하면서 **4% 실효과를
놓칠 수 있는 설계**가 "adequate"다. 유도가 §4.2에 없다.
재현: `python3 -c "import tc1_rule as R; [print(cv, R.power_adequate(cv*3.220, 3.220)) for cv in (0.02,0.03)]"`.

**B8 · `power_T`의 SD가 잘못된 층에서 온다 + 페어링 미등록.** `SD_PRIOR["H"]=0.023`
= √(0.013²+0.019²) = **두 독립 arm SD의 합성(=unpaired 차이 SD)** 인데
`T_CRIT_N4=3.182`(df=3)는 **paired n=4**를 함의한다. 페어링 구조(같은 부팅? 같은
seed? 같은 trace?)가 §5 어디에도 없다. 또한 Stage 1은 **static 셀 SD를 n=3으로**
재는데 결정량은 **Stage 2 차이의 SD**를 요구한다(다른 양). n=3 SD로 검정력을
선언하는 것도 취약(χ²₂: σ의 95% 구간 대략 [0.52σ̂, 3.2σ̂]).
재현: `tc1_rule.py:56-63` 주석 ↔ 사전등록 §4.2 · §5 표.

**B9 · `CONTROLLER_DEGENERATE`의 세 번째 절이 계측으로 산출 불가(게이트 #25 재발).**
등록: *"결정 호출 중 두 urgency 분기가 한 번도 발화하지 않은 비율 ≥ 0.99"*.
코드: `SLO-BIND` 로그는 **`_new != _idx`일 때만** 찍힌다(`:833-837`), 그리고 `_dwell>0`
또는 `_hold`면 urgency 분기는 **평가조차 되지 않는다**(elif 단락). ⇒ 분모(총 결정
호출)도 분기별 카운터도 존재하지 않으며, 하네스가 세는 `SW=grep -c "SLO-BIND|SLO-SCHED"`
(`sharegpt_vary_bench.sbatch:76`)는 **이동 횟수**일 뿐이다. §8의 하네스 요건 목록에
이 필드 추가가 **없다**. ⇒ §3(c)가 도입한 유일한 관측 가능 술어가 산출 불가 ⇒
§3(c)의 항등식이 닫히지 않는다.

**B10 · §7.1 음성대조 — (a) T5를 증거로 오용(게이트 #9) (b) n=2라 빈 서명 통과
(게이트 #44) (c) 데이터 출처가 §8-7과 모순.**
(a) `T5`는 `tc1_rule.py:113-116`의 `if sT=="win" and sH=="lose"`를 다시 읽은
**항등식**이다. FLIP/REVFLIP은 정의상 `sH≠sT`를 요구하므로 *"같은 부호 6,912 세계,
귀속 0건"* 은 데이터와 무관하게 참이다. §7.1이 이를 *"이미 증명했다"* 고 인용하는 것은
범주 오류 — 실제 위험은 **부호 추정량**에 있지 라벨 함수에 있지 않다.
(b) 정본 n≥4를 두 유사-arm으로 쪼개면 **arm당 n=2**다. t₀.₉₇₅,₁ = 12.71이라
`t-CI가 0을 배제`가 사실상 불가능 ⇒ 두 유사-arm 모두 `win=false` ⇒ FLIP이 **구조적으로**
못 뜬다 ⇒ 대조가 **공허하게 통과**한다. 이것이 게이트 #44의 빈 서명 구멍 그 자체다.
"모든 분할에 대해 발화율 보고"(게이트 #24 대응)는 공허성을 고치지 못한다.
(c) §8-7이 게이트 #41로 *"2026-07 Zamba2 수치 재사용 금지"* 를 등록했는데 §7.1은 그
데이터를 쓴다.

**B11 · §7.2 양성대조는 항등식은 아니나 칼날 위에 있고, 등록된 변이 테스트가 무연산.**
감사 실계산(정본 §1-7 수치):

```
Δ̂_H = 3.132 − 3.220 = −0.0880        δ_H = 0.03 × 3.220 = 0.0966
주입 Δ̂_T = Δ̂_H + 2δ = +0.1052        win 문턱 = max(δ, t·se) = 0.0966
여유 = 0.0086 = 0.75 se               P(win | 참 Δ = 0.1052) = 0.773
```

⇒ 발화하긴 하지만 **여유가 0.75 se**다. 재측정된 `Δ̂_H`가 −0.0966보다 조금만 더
음이면(=동적이 3%보다 더 지면) **양성대조가 실패**하고, 실패 시 처리가 미등록이다.
그리고 *"★변이 테스트: `δ`를 `2δ`로 되돌린 변이본에서 이 대조는 반드시 실패해야
한다"* — 주입값이 **이미 2δ**이므로 이 문장은 **무연산 변이**이거나 오타다. 어느
쪽이든 변이 테스트가 등록되지 않은 것과 같다(CONSENSUS §3 **항목70**).
어느 H 데이터셋을 쓰는지(§1-7 계열 3.220 vs G16 계열 3.7973)도 미등록.

**B12 · `scoring` 축이 전역 단일 이진.** 모델별·부호별 불일치를 구별 못 한다. 또한
`sign_H`/`sign_T`가 **어느 채점의 부호인지** 코드에 없다(§4.1은 1차=p95라 하므로
p95여야 한다). 규칙 파일이 스스로 그 사실을 적어야 한다.

**B13 · `NO_COMMON_BAND`가 §3(f) 하에서 사실상 도달 불가.** rate를 λ/C 고정 분율로
**강제**하면 "공통 밴드"는 정의상 존재한다. 밴드 실패 경로는 전부 cliff 축으로
흡수된다 ⇒ `band` 축은 장식일 가능성이 높다.
★그리고 `T2_reachable`·`T11_no_inert_axis`는 **격자 내부 성질**이라 이런 **설계층
도달불가**를 원리적으로 못 잡는다. §6이 marginals에 대해 스스로 단 경고
(*"세계 개수는 확률이 아니다 … 격자의 산물"*)를 **자기 검사에는 적용하지 않았다.**

**B14 · M1–M4가 거의 자동 통과 구조.** 각 가드가 독립 축 하나에 1:1 대응하므로
*"가드를 빼면 라벨이 움직인다"* 는 `T2_reachable`이 참인 한 자동이다. 실제 반증력을
주려면 A0 rev4가 도달한 수준(지정 mutant × 판별검사 짝짓기 + `uncovered_mutants:[]`
메타검사, `CONSENSUS.md` 레지스트리 stage0ppp 행)이 필요하다. 현재 mutant 커버리지
검사가 **0건**이다.

**B15 · M5의 "1,728 라벨이 움직인다"는 판정과 무관.** `boot·band·cliff·grid·degen·
scoring`은 **전부 종결형 차단 라벨**이므로 그들끼리의 순서는 *"실질 판정에
도달하는가"* 를 바꾸지 않고 **차단의 이름만** 바꾼다. §6의 *"이 순서가 1,728개 라벨을
좌우한다"* 는 과대 서술이다(합격기준 3 자체는 통과 — §4 참조).

**B16 · §8-5 `input_lens`가 결정 역할 없는 "필수 공변량"** — 문턱도 축도 없으므로
데이터를 본 뒤 *"flip은 tokenizer 탓"* 이라 말할 수도, 안 말할 수도 있다. 통제가
아니라 사후 설명 채널이다.

**B17 · `PDMUX_SLO_ANCHOR_IDX`의 인덱스 오프셋 미등록.** `lff_bench.sbatch:47` 주석은
`idx2=d24`인데 `pdmux_slo.yml`의 division 리스트는 d16부터 5개다(오프셋 1). *"모델별
Stage 1 argmax"* 를 정수로 옮기는 사상이 미등록이다.

**B18 · 자체검사 개수 불일치.** 문서 머리(줄 6) **18건**, §6(줄 265) **17건**. 실행값은
**18**(감사 재실행 확인, `ALL PASS`).

---

## 3. §0 자수 항목 검정 (합격기준 5 전반) — **보고 (3)은 정확하다. 그러나 그 정정
문장 자체가 새 출처 허위를 만들었다.**

### 3.1 (3)의 사실관계 = **정확** ✓

```
results/kernel_mech/stage0ppp/stage0ppp_a0_rule.py:2
  """Stage 0''' A0 -- THE REGISTERED DECISION RULE, fixed in code (gate #66).
```

`PROJECT_STATUS.md` "방법론 게이트" **#66**(줄 7145–7164) = *"포크는 원본의
하드와이어를 상속한다 — 매 회차 새 거부 지점이 나오면 아키텍처 신호다"*
(S-6 rev1–rev4, 2026-08-23, `CONSENSUS.md` §3 항목86 대응). **rules-as-code 조항
없음.** ⇒ 사전등록의 지적은 옳다. 별건 등재 값어치가 있고, 동결 파일이므로 정오표
배너 권고도 옳다.

### 3.2 ★그런데 정정본이 **또 다른 틀린 출처**를 넣었다

```
tc1_rule.py:6-8
  ... The rules-as-code discipline is 교훈 항목66 in CONSENSUS §3, which is
  not gate #66.  This file therefore cites the discipline BY NAME ...
(사전등록 §0 줄 37-38도 같은 문장)
```

**거짓이다.**

- `CONSENSUS.md` §3 **항목66**(줄 3558) = *"사다리 해상도가 사전등록된 payoff
  구간을 은폐할 수 있다"*(G16 §1.3, 2026-08-17).
- rules-as-code 규율의 정본 위치는 **`CONSENSUS.md` §3 항목81**(줄 3871) =
  *"규칙을 산문으로 고정하면 구멍이 난다 — 결정 규칙은 코드로 고정하고 세계를
  전수 열거하라"*(P0-A rev1–rev6 + S-6 rev1–rev4, 2026-08-23), 대응 게이트는
  **`PROJECT_STATUS.md` 게이트 #61**(줄 7055).

즉 이 파일은 **틀린 게이트 번호를 틀린 교훈 번호로 교체**했고, 그 문장은
`VERIFIED 2026-08-27` 딱지를 달고 있다. 검증된 것은 *"#66은 rules-as-code가
아니다"* 라는 **부정 절반**뿐이고, 대체 인용(**항목66**)은 검증되지 않았다.
⇒ PS게이트 **#67/#70**(*"검증 칸에는 이미 실행된 검증만 적어라"*)의 즉시 재발이며,
사전등록이 §0에서 인용한 *"출처 허위 전파"* 교훈이 **한 세대가 아니라 같은 문장
안에서** 재발했다.

### 3.3 근본 원인 — 번호 체계는 **둘이 아니라 넷**이다

| 체계 | 위치 | 범위 |
|---|---|---|
| A | `CLAUDE.md` "방법론 게이트 (필수)" | #1–#8 |
| B | `PROJECT_STATUS.md` "방법론 게이트" | #1–#80 |
| **C** | **`CONSENSUS.md` §3 교훈 항목** | **#1–#100** |
| **D** | **메모리 topic `deconfound-measurement-lessons` 항목** | **#1–#80** |

사전등록이 쓴 *"교훈 항목53(변이 테스트)"* · *"교훈 항목66(rules-as-code)"* ·
*"교훈 항목80(출처 허위)"* 은 **전부 체계 D의 번호**인데 체계 C(`CONSENSUS §3`)로
표기됐다. 체계 C에서 변이 테스트는 **항목70**(줄 3657–3706), rules-as-code는
**항목81**이다. §0이 체계를 둘로만 셌기 때문에 D→C 오사상이 그대로 통과했다.

### 3.4 "실제 오인용 3건" = **한 자릿수 과소 집계**

`PREREG*.md` + `*_rule.py`로만 한정해도(tc1 제외) 한정어 없는 체계-A 인용이
**15개 파일 ~50건**이다. 놓친 것 중 심각한 것:

1. `results/s8_scaleup/PREREG_C2R_RULES_2026-08-15.md:82` — `게이트 #3 (n≥4)`.
   **완주 캠페인의 규칙 사전등록**인데 §0에 없다.
2. `results/nsl_lever/nsl_eb/nsl_eb_rule.py:302` — `# prereg sec 3; gate #3 (n>=4)`.
   ★**(3)과 완전히 같은 부류(규칙-as-코드 파일 안의 오인용)** 인데 §0은 (3)을 유일
   사례로 ★★★ 표시했다.
3. `results/p1_gates/gate2/PREREG_GATE2S_2026-08-09.md` — ★★**같은 파일이 `게이트 #4`를
   두 체계로 쓴다**: `:720` *"집계 단위(방법론 게이트 #4)"* = 체계 B,
   `:727`·`:735` *"정본 방법론 게이트 #4 정합"* / *"정본 게이트 #4(요청-내부
   token-ITL p95)"* = 체계 A. 저장소 전체에서 가장 나쁜 형태인데 §0이 못 잡았다.

재현: `grep -rnE "(게이트|gate) #[1-8]([^0-9]|$)" $(find workspace/engine-port/results -name "PREREG*.md" -o -name "*_rule.py")`

**⇒ 합격기준 5 전반 판정: §0의 방향과 (3)의 사실관계는 정확하지만, 집계가 한 자릿수
틀렸고, 스스로 등록한 규약을 같은 문서·같은 규칙 파일 안에서 즉시 위반했다. 과제
지시문의 표현대로, 이것 자체가 이 사전등록의 결함이다.**

---

## 4. 나머지 합격 기준별 판정

| # | 합격 기준 | 판정 |
|---|---|---|
| 1 | 12축이 만들 수 있는 세계를 전부 덮는가 | ✗ **아니오** — 死因 F4(판별력 축) · 차단 B1(sticky/realized) · B2(arm별 boot) · B3(격자 확장 상태) · B12(모델별 scoring) |
| 2 | §3(a)–(g)에 빠진 항등식·강제 분기 | ✗ **둘 놓쳤다** — F1(`MEASUREMENT_ABSENT` 강제) · F2(`NO_FLIP_BOTH_LOSE` 강제). 추가로 (b)의 처방이 (a)의 처방과 상호작용해 새 강제를 만든 것을 (a)–(g) 어디도 재점검하지 않음 |
| 3 | 가드 순서가 답을 미리 정하는가 | ✓ **아니오 — 통과(살아남음)**. 6개 가드가 전부 종결형 차단이라 상대 순서는 **차단의 이름**만 바꾸고 "실질 판정 도달 여부"는 안 바꾼다. 단 §6의 M5 서술은 과대(B15) |
| 4 | §7 두 대조가 항등식/빈 서명 통과 가능한가 | ✗ **둘 다 걸린다** — §7.1은 T5 인용이 항등식(a) + n=2로 빈 서명 통과(b)(B10). §7.2는 항등식은 아니나 여유 0.75 se이고 변이 테스트가 무연산(B11) |
| 5 | §0 보고 정확성 / §1.1 정직성 | ✗ §3 참조. §1.1-1은 **과소 서술 맞다**(F4 — 제시한 탈출구가 거짓). §1.1-4는 코드와 모순(§1.1-4 산문 "TC1의 측정 실패" vs `tc1_rule.py:43` `BOTHWIN`="중심 negative가 흔들린다" = 실질 라벨) ⇒ 사후 선택권 |

### §1.1 개별

- **§1.1-1** — C2b/E4 인용 자체는 **사실 정확**(`PROJECT_STATUS.md:4209-4211` +
  `handoff-report/experiment_plan_2026-08-04.md:171`에 축자 근거). 그러나 제시한
  탈출구(부호의 재척도 불변성)가 거짓이므로 **과소 서술**이다. 정직해지려면
  *"부호도 견디지 못한다 — 두 모델이 임계 대비 같은 판별 구간에 있음을 먼저
  실측해야 한다"* 로 고치고 그 축을 규칙에 넣어야 한다.
- **§1.1-2** — 견고. *"층 타입 때문이다 금지"* + NSL E-A(mamba pool) 후보 병기는
  정확하고 값어치 있다(`CONSENSUS.md` §1-23 대조 확인).
- **§1.1-3** — 견고.
- **§1.1-4** — **코드와 모순**(위). 수리 없이는 사후 선택권.
- **§1.1-5** — 견고.

---

## 5. 반증 실패 — 감사가 못 깬 것 (다음 판본이 지켜야 할 것)

1. **라벨 함수의 전역성·배타성**은 진짜다. 20,736 세계 전부가 정확히 한 등록 라벨을
   받고(`T1`), 서로 다른 실질 라벨이 세계를 공유하지 않는다(`T7`). 감사 재실행 `ALL PASS`.
2. **`T7b`·`T7c`는 항등식이 아니다.** `d(FLIP,REVFLIP)=2`, `d(BOTHWIN,BOTHLOSE)=2`,
   `d(FLIP,BOTHWIN)=1`은 규칙 의도와 문자 그대로 일치한다.
3. **가드 순서 등록(§6)은 정당하다** — 합격기준 3 통과(위 표).
4. **§3(a)(b)(c)(e)(f)의 문제 제기 자체**는 전부 실재하는 위험이고, (b)의 표본 분할
   (Stage 1 동결·SHA-256 해시, Stage 2는 신규 rep)은 max-over-K 승자의 저주에 대한
   **올바른 처방**이다. 유지 권고.
5. **§4.1의 p95/mean 이중 채점 + `SCORING_DEPENDENT` 차단**은 이 저장소가 세 번 낸
   진단을 GPU 0으로 닫는 **정당하고 값싼 설계**다. 강력 유지 권고.
6. **`T_CRIT_N4` 사용(percentile bootstrap 금지)** 은 PS게이트 #14와 정확히 일치.
   `T9`의 `P(win|Δ=0) ≈ 0` 산술도 감사 재계산에서 재현됨.
7. **§6의 marginals 경고**(*"세계 개수는 확률이 아니다"*)는 옳고 필요하다. 다만
   자기 검사에도 적용해야 한다(B13).
8. **§11 금지 문장 7건**은 전부 정확하고 스코프를 제대로 좁힌다.
9. **§0(3)의 사실관계는 정확하다** — 진짜 발견이고 별건 등재 값어치가 있다(§3.1).
10. **§8-6(`--max-mamba-cache-size = cap` 규칙, 절대상수 금지)** 은 `CONSENSUS.md`
    §1-23 따름정리와 축자 일치. 정확하다.

---

## 6. 이를 확정할 실험 (전부 GPU 0 또는 ≤0.3 GPU-hr)

1. **F1 확정 (GPU 0)** — `PDMUX_STICKY_PARTITION=1 PDMUX_SLO_SCHED=1`로 스케줄러
   init을 CPU에서 태워 `RuntimeError`를 아티팩트화한다. 코드로 이미 확정이므로
   실행은 형식이지만, §3(g)의 *"해소 가능"* 주장을 뒤집는 근거로 남긴다.
2. **F2 확정 (1 job ≈0.3 GPU-hr, ★본 캠페인 50 job보다 먼저)** — Zamba2 `bind+GATE`를
   **`PDMUX_SLO_ANCHOR_IDX = d44 인덱스`** 로 1 rep 돌려 `switch_count`·split 체류
   분포·goodput을 본다. `switch_count ≈ 0` ∧ goodput ≈ d44-static − (작은 정착 비용)
   이면 F2 확정 = 설계 재작성. **이 프로브를 안 사고 Stage 0–2를 사면 안 된다**
   (교훈 *"가장 비싼 지출은 GPU가 아니라 안 산 프로브"*).
3. **F3 해소** — `ESTIMAND ∈ {gpL, gpH, gpC}` 중 **하나**를 `tc1_rule.py` 상수로
   못 박고 나머지 둘은 **보고 전용**으로 명시. 세 값을 라벨 함수에 넣으면 그 자체가
   3배 자유다.
4. **F4 해소** — 축 `discrim_M ∈ (informative, all_pass, all_fail)` 신설,
   `cliff` 축을 `(below, straddle, above)` 3상태로 확장. Stage 0에서 **arm별 TTFT
   통과율·ITL 통과율**을 재고 두 모델 모두 (예) `[0.15, 0.85]`에 들어감을 **사전 문턱**
   으로 등록. 벗어나면 `NULL_DISCRIMINATION`으로 차단.
5. **B1/B2 해소** — `boot_M_arm`(2모델×2arm) + `sticky_M_arm` 축 신설, 0.90 문턱을
   코드 상수로.
6. **B10 해소(§7.1)** — 유사-arm 분할 **폐기**(n=2는 공허 통과). 대체: 측정된 SD로
   `Δ=0` 파라메트릭 부트스트랩을 돌려 **부호 추정량의 귀무 발화율**을 산출하고
   `≤ α`를 사전 등록. T5는 "항등식이므로 증거 아님"이라 규칙 파일에 명시.
7. **B11 해소(§7.2)** — 주입량을 `max(2δ, |Δ̂_H| + 2δ + t·se)` 처럼 **문턱 초과가
   보장되는 식**으로 등록. 변이 테스트를 두 개로 재작성: *"주입 0 → FLIP 미발화 필수"*
   + *"주입 δ/2 → 미발화 필수"*. 사용할 H 데이터셋을 파일 경로로 고정.
8. **B5 해소** — §4.2의 `lose` 정의에서 power 절을 빼고 power를 라벨 층에서만
   게이트하거나, 반대로 `UNDERPOWERED_NULL`을 삭제하고 `TIE`로 통합. **둘 중 하나를
   코드로 못 박아야** 선택권이 사라진다.
9. **B9 해소** — `_slo_decide_idx`에 결정 호출 카운터 + 분기별 카운터를 추가하고
   §8 하네스 요건에 명시(engine-porter 소관). 없으면 `CONTROLLER_DEGENERATE` 술어를
   `switch_count == 0 ∨ 체류 100% 단일 index` 두 절로 축소 등록.
10. **B14 해소** — 지정 mutant 목록 + 판별검사 짝짓기 + `uncovered_mutants: []`
    메타검사를 A0 rev4에서 이식.
11. **B4/§0 해소 (GPU 0)** — G16(`CONSENSUS.md` §1-33) 인용으로 H arm의 격자-내부성을
    선반영하고 Stage 1 H 셀을 줄인다. 번호 체계 4개를 §0 표에 등재하고
    `tc1_rule.py:6-8`을 **CONSENSUS §3 항목81 / PS게이트 #61**로 정정.

---

## 7. 금지 문장 (이 판정서가 신설)

- *"TC1이 규칙층을 통과했다"* · *"TC1 설계가 확정됐다"* (사전등록 §11 승계, 불변)
- ★*"`PDMUX_STICKY_PARTITION`이 M3의 estimand 미식별을 해소한다"* — 컨트롤러 arm에
  대해서는 **정의되지 않는다**(F1).
- ★*"anchor를 argmax에 두는 것은 SLO 쇼핑이 아니므로 중립적이다"* — 규칙은
  중립이지만 **효과는 중립이 아니다**(F2).
- ★*"부호는 모델별 단조 재척도에 불변이다"* — 임계 지시함수 estimand에서 거짓(F4).
- ★*"T5가 음성대조 성질을 이미 증명했다"* — T5는 항등식(B10a).
- ★*"규칙-as-코드 규율은 CONSENSUS §3 교훈 항목66이다"* — **항목81**이다(§3.2).
- ★*"저장소의 게이트 번호 오인용은 3건이다"* — 규칙·사전등록 파일만 세도 ~50건(§3.4).
- *"TC1의 게이트 번호 감사가 전수였다"*

---

## 8. 회계

GPU 지출 **0** · job 제출 **0** · 새 성능 판정 **0건** · 등급 변경 **0건** ·
정책 순위 변경 **0건** · HE0 **불변** · 대상 파일 **무수정**.
감사가 실행한 것: `tc1_rule.py` 스크래치 사본 재실행(18/18 PASS 확인) ·
검정력/양성대조 산술 재계산 · 저장소 grep 8회 · 코드 정독 3파일.
