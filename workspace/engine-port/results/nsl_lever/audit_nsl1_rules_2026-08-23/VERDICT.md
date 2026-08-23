# NSL-1 규칙층 적대 감사 판정서 (게이트 #34 1단계)

대상: `../DESIGN_NSL1_ADMISSION_LEVER_2026-08-23.md` (+ `../nsl1_power.py`, `../NSL1_POWER_2026-08-23.json`)
감사자: claims-auditor · 2026-08-23 · **읽기 전용**(대상 트리 수정 0건) · **GPU 지출 0** ·
새 성능 판정 **0건** · 등급 변경 **0건** · 정책 순위 변경 **0건** · HE0 불변

---

## 0. 단일 판정

# `NO-GO` (규칙층)

死因 3건 = **F1**(결정량이 데이터 관측 전에 부호가 정해짐 — G17 死因의 문자 그대로의 형태) ·
**F2**(등록된 arm×워크로드×SLO×메트릭 조합이 저장소에 **존재한 적이 없고**, 모든 교정상수가
다른 모델에서 온 수입값) · **F3**(설계가 깬 앨리어스보다 **큰 미등록 앨리어스** = cap ↔ 실현 D).
그 밖에 차단 F4–F11, 경미 F12–F15.

**★트랙 자체는 죽이지 말 것.** 질문(§5-8(b) admission 갈래)은 정본이 스스로 두 번 지목한
진짜 구멍이고, `cap=48`이 미등록 상수라는 사실 주장은 (스코프 정정 후) **맞다**.
죽은 것은 **이 형태**다. §6의 재설계로 살릴 수 있고, 그 재설계는 규칙층에서 GPU 0으로 끝난다.

**합격 기준 대조**(설계 §8이 지정): "결과가 사후 재해석·자유 문턱·항등식·측정 실패의 결과
라벨링으로 만들어질 경로가 없을 것" → **네 경로 모두 열려 있다**(F1=부호 강제, F10=자유 문턱 4건,
F5·F1=항등식/부호 강제, F3·F9=측정 실패가 `ADMISSION_AXIS_INERT`로 라벨링됨).

---

## 1. 차단 목록

### ★F1 — 결정량이 자기 판정을 제조한다 (G17 死因, 문자 그대로) — **치명**

`DESIGN §4.1`:
```
결정량:  argmax_{(D,cap)} goodput   vs   argmax_{D} goodput | cap = 48
판정:    (i) 결합 argmax의 cap ≠ 48  →  ADMISSION_AXIS_MATTERS
        (ii) 결합 argmax의 cap = 48  →  ADMISSION_AXIS_INERT
        (iii) 페어드 CI가 0을 포함    →  UNDETERMINED
```
`cap=48` 슬라이스는 결합 격자의 **부분집합**이다. 따라서
`Δ := max_joint − max_{cap=48} ≥ 0`이 **데이터와 무관하게 항상** 성립한다.
`audit_g17_rules_2026-08-18/a1_restricted_grid.py`가 G17을 죽인 판정문
(*"D_ttft = S_min ⇒ Δ ≥ 0 FORCED (positive inflation)"*)과 **같은 구조**이며, 여기서는
집합 포함으로 강제되므로 더 강하다.

재현(`b2_argmax.py`, 200,000 반복, CPU 0.1분):

| 진실 | n | 부팅 SD | `P(ADMISSION_AXIS_MATTERS)` | `E[Δ]` | `P(Δ<0)` |
|---|---|---|---|---|---|
| cap 완전 무력(9셀 동일 70pp) | 4 | 0.50 | **0.666** | +0.160 pp | **0.000** |
| 〃 | 4 | 0.88 | **0.667** | +0.282 pp | **0.000** |
| 〃 | 4 | 2.00 | **0.667** | +0.638 pp | **0.000** |
| 〃 | 4 | 4.80 | **0.667** | **+1.539 pp** | **0.000** |

⇒ (a) **귀무 하에서 위양성률 = 2/3**(cap 격자점 3개 중 48이 아닌 것을 뽑을 확률),
노이즈 크기와 **무관**. (b) `Δ`는 확률 1로 비음수. (c) `E[Δ|H0]`가 **노이즈와 함께 증가**
(0.16 → 1.54 pp) — 즉 **노이즈가 양성을 제조한다**(게이트 #20의 거울상: 잡음이 귀무를 채택시키는
게 아니라 대립을 채택시킨다). SD 4.8pp에서 `E[Δ|H0]`=1.54pp = 2.2% 상대차로,
설계가 방어선으로 인용한 **3% 게이트에 육박**한다 — 게이트 #3은 이 규칙을 보호하지 못한다.

추가 결함:
* (ii) `ADMISSION_AXIS_INERT`는 **등가검정 여백 없는 귀무 채택**인데, 설계 §0은 그것으로
  **정본 열린 항목(§5-8(b))을 닫겠다**고 적는다. 등가 여백(TOST) 없이 정본 항목을 닫을 수 없다.
* (iii) "페어드 CI"의 **피연산자가 정의돼 있지 않고**, §4.4는 페어링이 **구조적으로 불가능**하다고
  선언한다 — 같은 문서 안의 모순(F5 참조). (i)/(ii)/(iii) 사이의 **우선순위도 미정**.

**수리**: 결합 argmax를 버리고, **부호가 사전에 정해지지 않는 대조**로 재정의하라.
예: 각 `D`에 대해 `Δ_D := goodput(D, cap_alt) − goodput(D, 48)`를 **사전등록된 방향 없이**
양측으로 검정하고, `INERT` 판정에는 **사전등록 등가여백**(예: ±3% 상대)을 요구하며,
다중성(F6)을 명시적으로 통제한다. `argmax` 언어는 판정어에서 제거하라.

---

### ★F2 — 등록된 (arm × 워크로드 × SLO × 메트릭) 조합이 저장소에 존재한 적이 없다 — **치명**

* `Ha8` = **`Zyphra/Zamba2-7B-Instruct`** (`../../s8_frontier/DESIGN.md:30`).
* 변화 trace·tight-SLO·G16 하네스는 **전부 `MODEL=Zyphra/Zamba2-2.7B` 하드와이어**:
  `../../slo_sched/sharegpt_vary_bench.sbatch:32` · `interactive_bench.sbatch:34` ·
  `g16_grid.sbatch:142` · `he2_bench.sbatch:32` · `slo_bench.sbatch:25` · `mixed_bench.sbatch:25`.

⇒ 설계가 격자 교정에 쓴 정본 상수가 **전부 2.7B 기판의 것**인데 7B arm에 그대로 쓰였다:
`rate 3↔12`가 절벽 밖이라는 §1-14의 근거, 용량 ≈6.3/s, chat(300/50) 도달 가능성,
그리고 §4.4가 가정한 goodput 수준 70pp. **교훈 #31**(basis 미검증 수입)의 정면 재발이며,
sim→serving 전이 오류의 모델-스케일 판본이다.

**(a) §4.2의 arm 선택 근거가 거짓이다.**
> "Ha8은 gate #13 job-축 캠페인이 부팅 분산을 실측한 arm이라 **분산 사전정보가 있다**"

gate #13이 잰 것은 `r`(고정 SM에서의 decode ITL 비, ctx1024, 정상 부하)의 `σ_job`이다.
**변화-trace goodput의 부팅 분산에 대해 아무 정보도 주지 않는다.** 이 문장은 삭제돼야 한다.

**(b) 등록된 ITL SLO가 이 arm에서 물리적으로 도달 불가일 수 있다.**
`../../s8_scaleup/FINDINGS_8B_2026-07-28.md:100-101` — Ha8 decode ITL **p50 = 89.80 ms @ SM16**
/ 31.49 ms @ SM92 (ctx1024, batch 9, cudagraph는 piecewise만 OFF).
NSL-1의 술어는 **요청-내부 token-ITL p95 ≤ 50 ms**(p50보다 나쁜 통계)이고 격자는 **d16을 포함**한다.
⇒ `d16` 열 전체가 goodput ≈ 0으로 퇴화할 개연성이 크고, 그러면 argmax는 남은 두 열의
잡음 순위가 된다. 저장소는 **이 실패 모드를 이미 문서화했다**:
`../../s8_frontier/DESIGN.md:351-360` — *"If the ITL SLO term never binds ... the experiment
stopped being able to ask the question it exists to ask. That is a **design failure, not a null
result**"*. ⇒ **게이트 #18 재발**(저장소가 낸 진단을 설계가 안 봄).

**수리**: 둘 중 하나. (i) arm을 **Zamba2-2.7B로 되돌린다** — 모든 앵커가 존재하고 하네스가 포크가
아니게 된다(권고, §6 참조). (ii) Ha8을 고수하려면 **용량·SLO 도달성 재교정 캠페인이 선행**해야
하고 그 비용을 §7에 등재해야 한다.

---

### ★F3 — 설계가 깬 앨리어스보다 큰 미등록 앨리어스: **cap ↔ 실현 D** — **치명**

`CONSENSUS.md §1-25`(rev, 2026-08-03, 독립 재현 77,688 스냅샷):
> decode-active 시간 중 `decode_sms == D` 비율 — **Ha8 d16 0.104 / d24 0.110 / d44 0.148 / d54 0.187**
> ⇒ 셀 라벨의 decode 분할은 decode 작업시간의 **4–19%만 실현**되고 81–96%는 무분할 108 SM.

`CONSENSUS.md §1-26`(코드 사실, `srt/multiplex/multiplexing_mixin.py:773,792-794`):
> 이 기판에서 **"decode가 D SM에서 돌았다" ⟺ "prefill이 동시에 in-flight였다"는 같은 사건**이다.

⇒ **admission cap은 in-flight prefill 수를 직접 조절하므로 실현 D를 직접 움직인다.**
`(D, cap)` 격자는 **요인설계가 아니다** — 결합 argmax는 "cap 효과"와 "cap이 유발한 D 실현률 변화"를
분리할 수 없다. 설계가 §3에서 깬 앨리어스(cap↔mamba pool)는 이것보다 작다.
§1-26은 같은 결함으로 D축 추정량 `g`를 **"이 격자 한정 은퇴 · 인용 금지"** 처리한 전례다.

**P4가 유일한 방어인데 문턱이 등록돼 있지 않다**(F10). 정본 수치(10–19%)를 기준으로 어떤
정직한 문턱을 걸어도 **9셀이 전부 폐기**되고, 문턱을 안 걸면 자유 문턱이다.

**수리**: (i) D축을 **정본 권고 한 점(d44)으로 고정**해 순수 cap 스윕으로 만들거나
(§5-8(b)가 실제로 묻는 것도 그것이다), (ii) `PDMUX_STICKY_PARTITION`
(구현·correctness gate 통과 — `CONSENSUS.md §1-27`, `s8_frontier/DESIGN.md §4.3.11-12`)을
선행 조건으로 등록하라. (i)는 격자를 9→3으로 줄여 F6·비용 문제도 동시에 해소한다.

---

### F4 — 선행 연구 인용이 정본 **인용금지**를 위반한다 (게이트 #47 / 교훈 #41의 재발)

설계 §8-4는 *"게이트 #49(ILL-POSED)에 걸리는가"* 를 물었고 **틀린 질문이라 무사 통과**했다.
적용되는 금지는 `ILL-POSED`가 아니라 **인용금지**다:

* `reports/CONSENSUS.md:1407` — *"2026-08-01 실행된 E1 전제 실험 4건(jobs 870295/870296/870297
  용량 스캔, **870301 batch-cap**)은 **전부 claims-auditor 미통과 = 이 문서에 결론으로 올리지
  않는다** — 상태 기록은 PROJECT_STATUS에만 있고 **인용 금지**다."*
* `PROJECT_STATUS.md:1096` — *"⚠️**이 4건은 전부 claims-auditor 미통과 = 정본·논문 인용 금지**
  이며 등급어는 **미검증**"*

NSL-1은 §2에서 870301의 수치를 인용한다(`kv_occupancy ≈ 0.018–0.024`,
`frac_itl_le_50` 0.98→0.3, "cap 4배에도 TTFT·ITL 불변", d16 체류 18%), 그것을
**"★이미 확립된 것"** 이라는 제목 아래 두며(= 정본이 "미검증"이라 부른 것의 **등급 상향 언어**),
그 18%를 **P4의 참조값**으로, `kv_occupancy ≈ 0.02`를 **§6-3의 KV 갈래 배제 근거**로 재사용한다.
"설계 동기로만"이라는 단서는 정본 규칙(평면적 인용 금지)을 면제하지 않는다.

`check_citation_stops.py --file`은 `0 violation`을 냈지만 **레지스트리에 이 수치의 규칙이 없어서**다
(도구 자신이 등재한 한계). **도구 통과는 준수의 증거가 아니다.**

**수리**: 870301의 **모든 수치를 삭제**하고 *"cap 스윕이 한 번 돌았으나 인용금지·미검증이라
확립된 것이 없다"* 만 남겨라. P4 문턱은 정본 §1-25(인용 가능)에서 다시 유도하라.

---

### F5 — "페어링은 구조적으로 불가능"이 사실이 아니고, 그 위에 검정력 게이트 전체가 서 있다

`../nsl1_power.py:24-27` + §4.4:
> "Cells are compared ACROSS boots ... the comparison is UNPAIRED -- **pairing is unavailable by
> construction**"

반증 2건:
1. `../../s8_frontier/batchcap.sbatch:104-136` — **한 job 안에서 6개 (cell, cap) 조합을 순회 부팅**한다.
   그 헤더 `:96-99`는 스스로 이렇게 적는다: *"Same seeds across every (cell, cap) so the cap
   comparison **stays paired**."*
2. 같은 헤더 `:96-98` — `--seed`가 **프롬프트 집합 자체를 바꾼다**(`bench_serving.py:1705` →
   `datasets/sharegpt.py:98`의 `random.shuffle`). ⇒ **seed는 정당한 블록 인자**이고, 200-프롬프트
   변화 trace에서 워크로드 실현 분산은 큰 성분이다.

즉 페어링/블로킹은 **설계 선택**이지 구조적 불가능이 아니다. 그리고 job을 블록으로 쓰면
이 프로젝트가 반복해서 데인 **node·date·boot 성분이 제거**된다.

정량(`b3_power.py`): n=4에서 unpaired는 **σ_total ≤ 0.882 pp**를 요구하고, job-블록 설계는
**σ_residual ≤ 0.698 pp**를 요구한다(df 6→3 손실 반영). ⇒ **job 성분이 총분산의 약 40% 이상이면
블로킹이 이긴다** — 이것은 **측정할 문제이지 가정할 문제가 아니다**. P6은 스칼라 SD 하나가 아니라
**분산 성분 분해(job / seed / 잔차)** 를 재야 한다.

---

### F6 — 검정력은 **1회 쌍대 검정**용인데 결정량은 **9셀 argmax**다 (다중성 미통제)

`../nsl1_power.py`의 산술 자체는 **정확하다**. 독립 재계산(`b3_power.py`)이 4자리까지 일치:
goodput 70pp → `{4:0.882, 6:1.170, 8:1.394, 12:1.755}`, 50pp → `{4:0.630, 6:0.835, 8:0.996, 12:1.253}`.
**틀린 것은 산술이 아니라 그 산술이 계산된 대상**이다.

n=4, goodput 70pp, 3% 상대차에서 Bonferroni 보정 후 허용 SD:

| 비교족 | 허용 SD (pp) | 설계 등록 게이트 |
|---|---|---|
| 6개 cap 대조 | **0.288** | 0.882 |
| 8개 vs-best | **0.568** | 0.882 |

⇒ **P6 게이트 문턱이 자기 결정량에 대해 1.6–3.1× 관대하다.** P6을 통과해도 argmax는 underpowered.

부수: 비율형 goodput에 2표본 t를 쓰는 것 자체는 (셀 평균이 부팅 단위 평균이고 n≥4라)
치명적이지 않으나, **경계효과**(F2(b)처럼 goodput이 0 또는 1에 붙는 셀)에서는 정규근사·등분산이
동시에 깨진다. 등록된 격자가 그 경계에 걸릴 위험이 실재하므로(F2·F11), 셀별 goodput이
`[0.05, 0.95]` 밖이면 t를 쓰지 않는다는 **사전 규칙**을 함께 등록하라.

---

### F7 — 저장소가 이미 이 SD를 쟀다. 그리고 §4.4의 "예시 눈금 4.8 pp"가 바로 그 값이다

`reports/interactive_slo_retune_plan.md:153-158` (n=4, chat **300/50**, goodput attainment %,
변화 trace 아님·정상 r8):

| 정책 | mean ± **std** |
|---|---|
| **d44** | **73.2 ± 4.8** |
| d34 | 49.6 ± 3.9 |
| bind+GATE | 44.3 ± 2.7 |
| bind | 40.6 ± 0.4 |

`b3_power.py` (C): 이 SD들에서 필요한 n/셀과 부팅만의 9셀 비용 —
**4.8 → n=83 / 30.25 GPU-hr** · 3.9 → 56 / 20.41 · 2.7 → 27 / 9.84 · 0.4 → 3 / 1.09.

설계 §4.4(b)의 마지막 행 *"4.8 pp → n=83 → 30.25 GPU-hr"* 는 **정확히 d44의 정본 σ**인데,
설계 §4.4는 그 표에 대해 *"이 표는 어떤 관측도 아니다 … `4.8 pp` 행은 예시 눈금이지 이 기판의
측정값이 아니다"* 라고 적는다. 앵커였다면 그 부인문이 거짓이고, 우연이라면 독자가 반드시
오독한다. **둘 다 수리 대상**이다.

참고로 §1-7의 **변화 trace** 상대 SD는 arm마다 갈린다: d44 0.013/3.220 = **0.40%**,
d24 0.130/3.039 = **4.3%**, bind 0.306/2.934 = **10.4%** — 설계 요구치는 0.882/70 = **1.26%**.
⇒ 저장소 자신의 사전정보는 *"d44급 arm이면 통과, 그 밖은 3–8× 초과"* 를 가리킨다.
이것을 basis 단서와 함께 **등재하고 사전 행동을 약정**하는 것이 정직한 형태다.
"수입하면 #31에 걸린다"는 이유로 **아무 사전정보도 적지 않은 것**은 게이트 #18 재발이다.
(P6으로 게이트화한 방향 자체는 옳다 — 사전정보 은폐만이 결함이다.)

---

### F8 — 정본 하네스로부터 estimand·술어·SLO·계측이 전부 조용히 바뀐다 (그리고 포크는 S-6의 死因)

`../../slo_sched/sharegpt_vary_bench.sbatch:83-99`(정본 변화 trace):
* goodput = `g/dur` = **req/s**(비율 아님)
* 술어 = `m = mean(itls[i])`, `t<=3.0 and m<=0.06` ⇒ **mean-ITL**(정본 게이트 #4의 p95 아님)
* SLO **3 s / 60 ms 하드와이어**, `--max-running-requests 48` 하드와이어(`:53`), **telemetry 없음**

NSL-1은 (1) 비율형 goodput, (2) **p95** 술어, (3) chat 300/50, (4) 모델 2.7B→7B,
(5) cap 손잡이, (6) mamba pool 핀, (7) P3/P4/P5용 telemetry-ON — **7개를 동시에** 바꾼다.

* **교훈 #71**(포크는 원본의 하드와이어를 상속한다; 매 회차 새 거부 지점이면 아키텍처 신호)이
  **바로 오늘(2026-08-23) S-6를 HOLD로 보낸** 교훈이다. S-6는 7-arm 하네스를 **1-arm으로만**
  포크하려다 4회차 7개 독립 거부 경로로 죽었다. NSL-1은 그보다 큰 델타를 계획하면서 §7·§9에
  **하네스 위험을 한 줄도 등재하지 않았다**.
* **게이트 #41**(telemetry는 공짜 관찰자가 아니다) + **S-6 HOLD** ⇒ telemetry-ON 오버헤드는
  **미정량**이다. NSL-1 수치는 telemetry-OFF인 §1-7·§1-17 정본 수치와 **절대값 직접 비교 불가**.
* §7의 *"변화 trace 1라운드 벽시계를 기존 slo_sched 캠페인 로그에서 실측 상수로 채운다
  (수입값 금지, 교훈 #31)"* 는 **그 로그가 2.7B**라서 그 자체가 교훈 #31 위반이다.

---

### F9 — 대조 셀이 정본 운영점이 아니다 (pool 핀이 격자 전체를 이동시킨다)

§4.2는 `cap=48`을 *"저장소 전역 기본값(= 대조)"* 이라 부른다. **아니다.** 선행 캠페인은 전부
pool = 48(암묵 또는 `= cap`)이고, NSL-1은 **9셀 전부 pool = 96**이다. Ha8의 slot당 SSM state는
0.141 GB(§3-23)이므로 +48 슬롯 ≈ **+6.8 GB가 attention KV pool에서 빠진다** — §3-23이
*"Ha8의 attention KV pool은 이미 48×ctx의 52%만 잡혀 있어 mamba pool 증가분이 곧장 거기서 나온다"*
고 적은 바로 그 예산이다.

⇒ (a) NSL-1의 "대조"는 저장소 기준선이 **아니다** — `cap=48 / pool=48` **10번째 앵커 셀**이 필요하다.
(b) `cap=96` 셀은 cap이 아니라 **KV/token 예산에 묶일** 수 있다
(`srt/managers/scheduler.py:2469-2476`, `AddReqResult.NO_TOKEN → batch_is_full=True`) —
그러면 argmax가 48에 앉아 **`ADMISSION_AXIS_INERT`가 측정 실패의 라벨링으로 산출된다**(게이트 #21).
(c) 따라서 §6-3의 *"KV 갈래는 이 설계로 열리지 않는다 / 열린 채로 둔다"* 는 **부정확하다** —
이 설계는 KV 예산을 **능동적으로 이동시킨다**. 게다가 그 배제 근거 두 개(정본 관측치와 870301
`kv_occupancy≈0.02`)가 **둘 다 인용금지 수치**다(F4).

★**단, §3의 앨리어스 해소 논리 자체는 옳다.** pool을 전 셀 상수로 두면 cap만 움직이고
`kv_mamba_occupancy`의 `≡1.0` 항등식이 깨진다. §3-23의 `= cap` 규칙은 **그 항등식을 깨지 못한다**
(§3-23 자신이 *"pool 크기 = cap이므로 batch가 cap에 닿으면 정의상 1.0"* 이라고 적어 스스로
모순된다). NSL-1은 여기서 **정본을 교정하고 있다** — 그런데 그것을 "arm 간 vs arm 내 스코프 차이"로만
설명해 **교정 사실을 등재하지 않았다**. doc-steward 회부 대상.
부수 코드 사실(설계가 등록하지 않은 load-bearing 전제): `pool ≥ cap`이 cap을 잘라먹지 않는 것은
`--disable-radix-cache` 때문에 `_calculate_mamba_ratio()`가 1을 반환하기 때문이다
(`srt/model_executor/model_runner_kv_cache_mixin.py:390-392`, `:857-876`). 이 전제를 명시 등록하라.

---

### F10 — Stage 0 프로브가 자유 문턱이거나 공허참이다

| 프로브 | 결함 | 근거 |
|---|---|---|
| **P1** | *"부팅되는가 / KV pool이 L·batch를 감당하는가"* — **숫자가 없다**. 자유 문턱 | 부팅 로그의 `max_total_num_tokens`·`max_mamba_cache_size`·해소된 `max_running_requests`에 사전 문턱을 걸어라 |
| **P2** | **TTFT p90만** 본다. 결합 술어의 **ITL 쪽을 전혀 확인하지 않는데**, cap이 가장 크게 움직이는 항이 바로 ITL이다 | §4.3 본문. 설계 §2가 인용한 선행 관측도 "cap↑ → TTFT↓·**ITL↑**"이다 |
| **P3** | ★**fail-open**. `/get_server_info`는 `dataclasses.asdict(server_args)`에 `scheduler_info`를 덮어쓰는데(`srt/entrypoints/http_server.py:613-616`), `scheduler_info`(`srt/managers/scheduler.py:1290-1300`)에 `max_running_requests`가 **없다** ⇒ **요청값이 그대로 반사**되고 해소값 `min(cap, estimated, pool//ratio)`(`model_runner_kv_cache_mixin.py:857-876`)을 검증하지 못한다 | 부팅 로그 `srt/managers/scheduler.py:706`의 `max_running_requests=` 라인을 읽어라 |
| **P4** | **문턱 없음**(자유 문턱). 정본 Ha8 실현률 10.4–18.7%(§1-25) 기준 어떤 정직한 문턱도 전 셀을 폐기시킨다 | F3 |
| **P5** | ★**공허참 PASS**(게이트 #42/#57 형태). `kv_mamba_occupancy = 1 − avail/size`(`srt/multiplex/multiplexing_mixin.py:368-387`)이므로 pool=96 고정 시 batch가 96 미만인 **모든** 셀에서 `<1.0`이고, batch가 조금만 달라도 "셀 간 다른 값"이 성립한다. **핀이 걸렸는지 자체를 검증하지 못한다** | 부팅 로그의 `max_mamba_cache_size == 96` 결정론적 단언으로 교체 |
| **P6** | 스칼라 SD 하나만 잰다. **분산 성분 분해**(F5)와 **다중성 보정 문턱**(F6)이 필요하다 | F5·F6 |

---

### F11 — cap이 물려면 과부하여야 하고, 과부하면 metric이 정본 판정상 ill-posed다 (P2 ⊥ P3)

* cap이 레버이려면 running batch가 cap에 **닿아야** 한다(= P3). 선행 관측이 cap 48에 닿은 것은
  **정상 rate 16 · NP 128**에서였다(`../../s8_frontier/batchcap.sbatch:88-90`).
* 절벽을 피하려면(= P2) 용량 **아래**여야 하는데, 거기서는 batch가 작아 cap이 **안 문다**.
* 정본이 이 긴장을 이미 두 번 적었다:
  - `CONSENSUS.md §5-4` — *"현 goodput은 **과부하서 런 길이 의존 = ill-posed**"*
  - `CONSENSUS.md §1-19` — *"용량 이하=전부 통과 · 이상=전부 실패 … 유용하게 이기는 operating
    regime 없음"*
* NSL-1의 HI phase(rate 12)는 Zamba2-2.7B 용량(≈6.3/s) 대비 **≈2× 과부하**이고 7B arm이면 더 깊다.
  ⇒ **cap이 무는 구간과 metric이 ill-posed인 구간이 같다.**

설계 §4.3의 *"확인 실패 시 rate를 바꾸는 것이 아니라 설계를 되돌린다"* 는 규율은 옳다. 그러나
그 규율대로면 **P2가 통과할 개연성이 낮고**, 통과하면 이번엔 P3가 떨어진다. 설계는 이 상충을
인지하지 않았고 **연속형 대체 결정량**(예: TTFT/ITL 여유 마진의 분포, 다중 SLO 수준에서의
goodput 곡선, 용량 자체)을 예비로 등록하지 않았다. 감사 의뢰 §8의 "여기서 멈추지 말 것"
항목 7에 대한 답: **긴장은 실재하고 문서화돼 있다. 다만 "9셀 공통 off-cliff가 원리적으로 불가능"
까지는 확립되지 않는다** — 확립되는 것은 *"현 결정량으로는 두 요구를 동시에 만족시킬 사전
근거가 없고 설계가 그 상충을 다루지 않았다"* 이다.

---

### F12 (경미) — §0의 수 세기와 §1-3의 사실 주장

| 주장 | 실측 | 판정 |
|---|---|---|
| "`.sbatch` **66개**가 전부 48로 하드와이어" | `workspace/engine-port/results/**/*.sbatch`에서 **66** ✓ / **저장소 전체**로는 107개 중 96개가 이 플래그를 쓰고 **74개**가 리터럴 48, **16개**가 다른 값(8/16/24/32/40, 전부 `triage/`), 6개가 `$CAP`/`$TARGET_CAP` | **스코프 미기재**. "저장소 전체에서 한 번도 변수가 아니었다"는 부정확 |
| "`--max-mamba-cache-size` 지정 파일 **6개**, 전부 s8_frontier/s2_sticky/sticky_smoke 계열" | 정확히 6개, 계열도 일치 | ✓ |
| §1-3 "정본 결론을 만든 캠페인 계열(slo_sched·g2_0·**s8**)에서 cap은 변수였던 적이 없다" | 870301(`batchcap.sbatch`, **s8_frontier**)이 cap을 유일 변수로 돌았다 — **§2가 스스로 그렇게 적는다** | **문서 내 모순** |

재현: `find . -name '*.sbatch' -not -path './.git/*' -exec grep -l -- "--max-running-requests 48" {} + | wc -l`

---

### F13 (경미) — *"§1-4 kill chain의 마지막 마디가 **정확히 이 한 줄**이다"* 는 과잉 해석

인용된 file:line은 대체로 정확하다: `scheduler.py:2296-2299` ✓ ·
`pp_max_micro_batch_size` 실제 `:668-670`(인용 `:667-671`, ±1) ·
`:2432-2434` ✓ · `model_runner_kv_cache_mixin.py:218` ✓ · elif 실제 `:223-229`(인용 `:224-229`, ±1).

그러나 "한 줄"은 성립하지 않는다:
* `batch_is_full` 생산자는 `scheduler.py`만 해도 **`:2369` · `:2434` · `:2439` · `:2472/2476`**
  4곳이고, `disaggregation/prefill.py` · `scheduler_pp_mixin.py` · `dllm/mixin/scheduler.py`에도 있다.
* ★인용된 **`:2364-2370` 분기는 이 저장소 설정에서 발화할 수 없다** — 조건에
  `self.chunked_req is not None`이 있는데(`:2366`) 전 캠페인이 `--chunked-prefill-size -1`이라
  `chunked_req`가 설정되지 않는다. 실제로 작동하는 관문은 `:2353`(조기 반환)과 `:2433-2434`다.
* `:2472-2476`은 **KV/token 예산**(`AddReqResult.NO_TOKEN`)으로 같은 플래그를 세운다 —
  즉 §1-4의 **KV 갈래**도 같은 마디를 공유한다. "死因이 running-batch capacity이지 KV가 아니다"를
  이 코드로 분리할 수 없다.

⇒ 문구를 *"admission 관문이 `pp_max_micro_batch_size`를 경유해 `max_running_requests`에
매개된다는 코드 사실"* 로 낮추고, 다른 생산자들을 병기하라.

---

### F14 (차단) — §0의 payoff 문장이 §6-2와 모순되고, 정적 스칼라 스윕이 닫을 수 없는 것을 닫겠다고 한다

`CONSENSUS.md §5-8(b)`는 *"admission-control 또는 KV-aware한 **lever**로 이 死因을 직접 겨냥하는
시도"* 가 미구현이라고 적는다. NSL-1은 **레버가 아니라 기존 스칼라의 정적 설정 스윕**이고,
§6-2가 스스로 *"역할 인지 admission은 이 설계에 없다"* 고 적는다. 그런데 §0은
*"admission 갈래를 **닫을**"* 이라 쓴다. **같은 문서 안의 모순**이며, 후자는 등재 불가다.

등재 가능한 최대 문장은 §6-2 스코프다:
> *"Ha8·chat(300/50)·ctx4096·단일 노드·cudagraph-ON에서, 정적 스칼라 `--max-running-requests`를
> {24,48,96}로 바꾼 것이 정본 goodput을 [움직였다/등가여백 안에서 움직이지 않았다]."*
§5-8(b)의 갈래는 **열린 채로 남는다.**

---

### F15 (차단) — 한계로 등재되지 않은 것

§6-5가 "단일 노드·단일 SLO 쌍·ctx 4096"을 나열하나 **그 비용을 말하지 않는다**.
추가로 **미등재**:
1. 하네스 포크 위험(F8, 교훈 #71 — 오늘 S-6를 죽인 교훈).
2. telemetry-ON 스코프와 정본 수치와의 비교 불가(F8, 게이트 #41, S-6 HOLD).
3. D축이 10–19%만 실현된다는 사실(F3, §1-25) — 결합 격자 해석의 전제.
4. ★**cap > 48은 이 저장소에서 어떤 hybrid arm에서도 부팅된 적이 없다**
   (`batchcap.sbatch:80` `ARM="T8"`; hybrid 계열 스크립트는 전부 `CAP=48`). ⇒ P1은 필수인데
   그 미지의 크기가 §7 비용표에 반영돼 있지 않다.
5. 노드/날짜 축(gate #13이 ≥3노드를 정본 요구로 세웠고 job-축 캠페인도 2노드로 **부분 미충족**).
6. §3-23을 실질적으로 교정한다는 사실(F9 말미) — doc-steward 회부 필요.

---

## 2. 감사 의뢰 §8의 5개 질문에 대한 직접 답

| # | 질문 | 답 |
|---|---|---|
| 1 | §3의 pool 고정이 앨리어스를 깨는가 / 새 교락을 만드는가 | **깬다**(전 셀 상수 ⇒ cap만 이동). §3-23의 `= cap` 규칙이 항등식을 못 깬다는 지적도 **옳고 정본 교정**이다. 그러나 **격자 전체를 정본 운영점에서 이동**시키고(≈+6.8 GB를 KV에서 인출) `cap=96` 셀이 KV-bound가 될 수 있다 = **F9**. P1·P5로는 **불충분**(P1 자유 문턱, P5 공허참) = **F10**. 그리고 더 큰 앨리어스 **cap↔실현 D**를 놓쳤다 = **F3** |
| 2 | 결정량이 항등식·격자 산물인가 | ★**그렇다.** `Δ ≥ 0`이 집합 포함으로 강제되고, H0에서 위양성률 **2/3**, `E[Δ|H0]`가 노이즈와 함께 증가. **G17과 같은 死因, 더 강한 형태** = **F1** |
| 3 | §4.4 계산이 옳은가 | (a) **산술은 정확**(4자리 재현). (b) unpaired 가정은 **거짓** — `batchcap.sbatch`가 반례이고 seed는 블록 인자 = **F5**. (c) 비율형 t는 경계셀에서 위험(사전 규칙 필요) = **F6**. (d) "SD를 모른 채 등록한 것"은 **그 자체로는 차단이 아니다**(P6 게이트화는 오히려 옳은 방향). 차단인 것은 **저장소가 이미 가진 SD를 은폐한 것**(4.8/3.9/2.7/0.4 pp)과 **게이트 문턱이 자기 결정량에 대해 1.6–3.1× 관대한 것** = **F7·F6** |
| 4 | 870301 취급 | 열거는 **전수에 가깝고 정직**하다. 게이트 #49(ILL-POSED)에는 **안 걸린다** — 그러나 **더 강한 인용금지**에 걸린다(`CONSENSUS.md:1407`, `PROJECT_STATUS.md:1096`) = **F4**. 설계가 **틀린 질문을 던져 무사통과**했다. 놓친 선행 작업: `s2_sticky`·`sticky_smoke`·`m3r_realizability`·`e1_m3_control`·`m3r2_pinforce`·`ltsm_p1_probe`가 cap/pool을 변수로 배선해 뒀고(전부 `CAP=48`로 실행), `s8_frontier/DESIGN.md:770`이 *"a 96 was drafted"* 를 기록한다. 결정적으로 **§1-25/§1-26(D 실현률)·§5-4(과부하 ill-posed)·`DESIGN.md:351-360`(SLO 비구속=설계 실패)** 를 안 봤다 = **게이트 #18 재발 3건** |
| 5 | §6-2 순서 논증("스칼라 먼저") | **논리는 옳다**(스칼라가 움직이면 role-aware의 상한이 생기고, 안 움직여도 role-aware가 유효할 수는 있으나 그건 *예약*이 필요하다는 별개 명제다). ★그러나 **현 형태의 스칼라 실험이 신뢰할 답을 못 내므로 순서 논증이 무의미**하다 — F1·F3 하에서는 "스칼라가 움직였다/안 움직였다" 어느 쪽도 role-aware 결정의 입력이 될 수 없다 |

## 3. 그 밖의 지시 항목

* **HE0 오염**: **없음**. §0의 부인은 충분하고 §6-4가 동적 0개를 못 박는다. §4.1의 `argmax`는
  정적 설정 선택이지 컨트롤러가 아니다. ★다만 판정어 `ADMISSION_AXIS_MATTERS`는 제어 주장으로
  오독될 수 있으니, 등재 문장에 *"정적 설정값이지 컨트롤러가 아니다"* 를 명시할 것.
* **인용정지 승계**: C2 (a)(b)(arm별 ε/순위, 깨끗한 셀 CI·"n=4") **위반 없음** ·
  "3.06 vs 2.91" **미인용** · §1-4 "7.24s" **미인용** · `grand_mean_r` **미인용**.
  ⇒ 등재된 4건은 지켜졌고, **미등재 정지(870301)만 위반**(F4).
* **KV 갈래를 열어 둔다는 선언(§6-3)**: **부정확**. 이 설계는 KV 예산을 능동적으로 이동시키고
  (F9), 배제 근거 2건이 모두 인용금지 수치다(F4). "열어 둔다"가 아니라 **"KV 예산을 바꾼 채
  admission을 잰다"** 가 사실이며 그대로 등재해야 한다.
* **등재 가능한 문장이 나오는가**: 현 형태에서는 **아니다**. 양성은 F1이 제조하고, 음성은
  F3·F9·F2가 측정 실패로 만들 수 있으며, 어느 쪽도 §5-8(b)를 닫지 못한다(F14).

---

## 4. 값어치 판정

**질문은 산다. 이 형태는 사지 마라.**

* **왜 사는가**: `cap=48`이 미등록 상수라는 지적은 **저장소가 스스로 두 번 낸 진단**이다 —
  `../../s8_frontier/batchcap.sbatch:20-23`(*"an unregistered constant that on the evidence
  single-handedly determines the ITL axis"*)와 `CONSENSUS.md §3-23`. 정본 권고가 SM 축에서만
  최적화됐다는 것도 사실이다. §5-8(b)는 진짜 열린 항목이고 NSL-1이 그것을 겨눈 **첫 설계**다.
* **왜 지금 형태로는 안 되는가**: F1 하나만으로 **어떤 결과든 제조 가능**하고, F2·F3는 등록된
  격자에서 **측정 자체가 성립하지 않게** 만든다. 지금 집행하면 예산은 (i) `d16` 열 퇴화(F2b),
  (ii) P4 전 셀 폐기(F3), (iii) P1 부팅 실패(F15-4), (iv) P2/P3 상충(F11) 중 하나로 소진된다.
  §7이 스스로 *"총비용은 지금 쓸 수 없다"* 고 적은 것은 정직하나, **비용을 모르는 것보다
  결정 규칙이 부호를 강제하는 것이 더 큰 문제**다.
* **다음 회차 권고(전부 GPU 0, 규칙층에서 끝난다)** — 우선순위 순:
  1. **F1 수리**: 결합 argmax 폐기 → 방향 사전등록 없는 대조 + `INERT`용 등가여백 + 다중성 통제.
  2. **F2 수리 = arm을 `Zamba2-2.7B`로 되돌린다.** 모든 앵커(용량·절벽 rate·chat SLO 도달성·
     σ 사전정보)가 존재하고 하네스가 **포크가 아니게 된다**. Ha8은 후속으로 미룬다.
     — 이 한 수가 F2 전체와 F8 대부분을 동시에 죽인다.
  3. **F3 수리 = D축을 `d44` 한 점으로 고정**(또는 `PDMUX_STICKY_PARTITION` 선행 등록).
     격자 9→**3(+앵커 1)**, 다중성 소멸, 부팅 비용 ~1/3, §5-8(b)의 질문에 더 정확히 대응.
  4. **F4 수리**: 870301 수치 전면 삭제.
  5. **F10 수리**: P1 숫자 문턱 · P2에 ITL 항 추가 · P3를 부팅 로그로 · P4 문턱을 §1-25에서
     유도 · P5를 결정론적 핀 단언으로 교체 · P6를 **분산 성분 분해 + 보정 문턱**으로.
  6. **F9 수리**: `cap 48 / pool 48` 앵커 셀 추가, "= 대조" 표현 삭제, §3-23 교정 사실 등재.
* 이렇게 좁히면 남는 것은 **`sharegpt_vary_bench.sbatch`에 cap/pool 손잡이 2개를 더하는 작은
  델타**이고, 정본 §1-7 수치와의 비교 가능성도 (telemetry 축만 빼면) 보존된다.
  그 형태라면 **하네스층 감사(게이트 #34 2단계)로 넘어갈 값어치가 있다.**

---

## 5. 재현

```bash
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
python b2_argmax.py     # F1: H0에서 위양성 2/3, P(Delta<0)=0.000, E[Delta]가 노이즈와 함께 증가
python b3_power.py      # F5/F6/F7: sec4.4 재현(4자리 일치) + 페어드 대안 + 저장소 실측 SD + 다중성
```
사실 확인용 명령(모두 읽기 전용):
```bash
R=/scratch/ehmoon/whlee/prefill-layer-alloc
find $R -name '*.sbatch' -not -path '*/.git/*' -exec grep -l -- "--max-running-requests 48" {} + | wc -l   # 74 (repo) / 66 (results/)
grep -n "MODEL=" $R/workspace/engine-port/results/slo_sched/*.sbatch                                        # 전부 Zamba2-2.7B
sed -n '30p'   $R/workspace/engine-port/results/s8_frontier/DESIGN.md                                       # Ha8 = Zamba2-7B-Instruct
sed -n '100,101p' $R/workspace/engine-port/results/s8_scaleup/FINDINGS_8B_2026-07-28.md                     # Ha8 ITL p50 89.80/31.49 ms
sed -n '153,158p' $R/reports/interactive_slo_retune_plan.md                                                 # d44 73.2 +- 4.8
sed -n '1407p' $R/reports/CONSENSUS.md ; sed -n '1096p' $R/PROJECT_STATUS.md                                # 870301 인용금지
sed -n '83,99p' $R/workspace/engine-port/results/slo_sched/sharegpt_vary_bench.sbatch                       # g/dur, mean-ITL, 3s/60ms
sed -n '2364,2371p;2432,2434p;2469,2477p' /scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/managers/scheduler.py
sed -n '857,876p;390,392p' /scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py
sed -n '613,616p' /scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/entrypoints/http_server.py
```

---

## 6. 이 판정서가 **하지 않은** 것

* GPU를 쓰지 않았다. 새 성능 판정 0건. 어떤 등급도 바꾸지 않았다.
* HE0 · gate #13/#16 "닫았다" 금지 · switch-cost "닫았다" 금지 · C2 인용정지 (a)(b) — **전부 불변**.
* §5-8(b)는 **열린 채로 남는다**. 이 판정서는 *"NSL-1이 그것을 닫는다"* 도 *"닫을 수 없다"* 도
  주장하지 않는다 — 주장하는 것은 **현 규칙 집합으로는 어느 결과도 그 항목에 대해 정보를 갖지
  못한다**는 것뿐이다.
* ★**금지 문장 신설 제안**: *"NSL-1이 admission 축을 쟀다"* · *"cap 축이 무력함이 확인됐다"* ·
  *"NSL-1이 §5-8(b)의 admission 갈래를 닫았다"* — 위 F1–F3이 수리되기 전에는 쓸 수 없다.
