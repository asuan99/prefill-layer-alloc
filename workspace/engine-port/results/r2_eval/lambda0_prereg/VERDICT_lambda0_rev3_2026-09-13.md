# 규칙층 3차 감사 판정서 — 캠페인 0단계 λ* 사전등록 **rev3**

**대상**: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_eval/lambda0_prereg/PREREG_LAMBDA0_REV3_2026-09-13.md`
+ 동봉 `lambda0_plan.py` · `lambda0_lambda_inf.py` · `lambda0_cells.py` · `lambda0_cellprint.py` · `lambda0_analyze.py` · `lambda0_label.py` · `lambda0_mutation_check.py` · `lambda0.sbatch` · `workspace/engine-port/tests/test_lambda0_prereg.py`
**계보**: rev1 `NO-GO`(D1–D15) → rev2 `NO-GO`(D16–D23) → **이 rev3**
**감사일** 2026-09-13 · **GPU 지출 0** · 트리: HEAD **`b2b60e7`**(rev3가 인용 고정한 `bddff6a`에서 1커밋 이동), 작업트리 미커밋 10 M / 14 ??

## 0. 등급

# `NO-GO`

**死因 1건(도달불가 2행)**: **N3** — 등록 예보 **F2("shape B = `KNEE_BRACKETED`")의 참 분지가 정의역 ∅**이다. 부수적으로 **N1** — 그 결과 `R2 LAMBDA_STAR[B]`(이 단계의 두 산출물 중 하나)는 **어떤 데이터로도 산출될 수 없다**. 두 번째 도달불가 행: **`LADDER_TOO_HIGH`(÷4 재설계)가 A·B 양쪽에서 정의역 ∅** — §3.6이 "λ_inf가 λ*를 아래에서 묶는다"는 가정을 대체한다고 등록한 **데이터 검사 분지의 절반이 죽어 있다**.

**기전(한 문장)**: 계획이 `κ_pred ≥ 0.97`인 rung을 **전부** "저측 후보"로 인증하는데(`lambda0_plan.py:233-243`), R0의 배수 가드(`drain ≤ 2·L̂`, `lambda0_label.py:94-97`)가 **인증된 셀에 무조건 걸린다**. 그런데 R1의 **고측**은 `ratio ≤ 0.90`을 요구하고, 등록 자신의 항등식에 의해 그것은 `drain ≥ ((N/(N−1))/0.90−1)·span`을 뜻한다. shape B는 4/4 rung이 인증되므로(사전등록 §3.5가 6개 시나리오 전부에서 `low-side candidates: ['b_r0','b_r1','b_r2','b_r3']`라고 **스스로 인쇄**한다), **브래킷이 요구하는 바로 그 증거가 R0에서 측정 실패로 폐기된다**. 수치: 2·L̂ = **8.34–8.50 s** vs 필요 배수 **19.24–34.46 s**(24/24 셀, 비 **2.3–4.1×**).

**이 死因은 rev3의 잘못이 아니라 상당 부분 감사자(전임) 처방 D16/D17의 granularity 결함이다.** §3에 자기 철회로 명시했다. 수리는 **한 곳**이며 GPU 0으로 검증 가능하다(§6 E1, 실행해 확인함).

---

## 1. 반전·도달가능성 시험 표 (필수)

**원자료** = `results/longctx_conflict/probes/c_905835/{bench_*.jsonl, cell_*.json}`(보관 12셀) + 이 등록의 상수.
**추정량** = 등록 자신의 `lambda0_analyze.analyze` → `lambda0_label.rule`(**쉬핑 코드를 그대로 호출**했다. 손으로 만든 dict가 아니다).
**물리 모형**(반전 계산의 유일한 외부 입력, 새 자유 표면 아님 — 보관 원자료로 검증됨): 셀 지속 = `max(span + 배수, N/λ*)`, 배수 = `TTFT_floor + (out−1)·ITL(B̄)`, `ITL(B) = a′ + b′B`, **`a′ = 13.06 ms`는 측정값**(보관 3개 r0 셀의 마지막[무경합] 요청 ITL = **13.48 / 13.27 / 13.07 ms @ D16/D44/D92** — arm 무관), `b′`는 λ*로 결정. 보관 12/12 셀에서 `배수 ≈ 마지막 도착 요청의 e2e`가 **최대 오차 0.15 s**로 성립함을 확인했다.

| # | 자유 표면 / 축 | 민 범위 | 판정 변화 | 근거 수치 |
|---|---|---|---|---|
| **U1** ★★ | **인증(저측 후보) 집합 × R0 배수 가드의 결합** — 등록이 고정한 값(`KAPPA_MIN=0.97`, `DRAIN_MODEL_TOL=2.0`) 그대로 | 밀 필요 없음(등록값 자체) | ★**도달불가** shape B `KNEE_BRACKETED` **정의역 ∅**, `LAMBDA_STAR[B]` 산출 불가 | 6/6 시나리오 shape B: 인증 4/4, 자유 0. 셀별 `2·L̂` = 8.34/8.36/8.39/8.42(point-lower) … 8.37–8.50(abs-upper) vs `L90` = 21.56/20.95/19.90/20.19 … 19.24–19.55. **쉬핑 코드 실행 결과**(참 λ*(B) 스윕): 0.10–1.00 → 전부 `UNRESOLVED`(**등록 앵커 0.675 포함**), 1.10 이상 → `LADDER_TOO_LOW`. **`KNEE_BRACKETED` 0회** |
| **U2** ★ | 같은 결합 — **사다리 최하단** rung(A·B 공통) | 등록값 | ★**도달불가** `LADDER_TOO_HIGH`(÷4 재설계) **정의역 ∅**(양 shape) | A a_r0: `2L̂` 22.0–38.6 s vs `L90` 67.6–78.3 s(6/6). B b_r0: 8.34–8.37 vs 20.37–34.46(6/6). 스윕: λ*(A) ≤ 1.2 → `LADDER_TOO_HIGH`가 아니라 `UNRESOLVED`; λ*(B) ≤ 0.2 도 동일 |
| **U3** ★ | `KAPPA_MIN` | 0.97 → **0.99** | ★**반전(도달가능성)** shape B가 `KNEE_BRACKETED` **도달가능으로 바뀜** | b_r3 κ=0.9786–0.9812 < 0.99 ⇒ 미인증 ⇒ 고측 자유. **즉 등록값 0.97이 바로 F2를 죽이는 값이다** |
| **U4** ★ | `DRAIN_MODEL_TOL` | 2.0 → 5.0 | ★**반전(도달가능성)** 동일 | 5·L̂ = 20.9 s > `L90` 19.9 s(b_r2) ⇒ 가드 통과 |
| **U5** ★ | `MULT["B"]` 최상단 | 1.70 → 5.0·λ_inf | ★**반전(도달가능성)** 동일 | x=3.375에서 N=400 클램프 ⇒ span 118 s ⇒ κ=0.968 < 0.97 ⇒ 미인증 |
| **U6** | 측정된 `λ_inf(B)`(D18 술어의 출력) | 0.2 → 5.0 | **경계 확정**: λ_inf(B) **≤ 1.70이면 도달불가**, ≥ 2.0에서만 도달가능 | 등록 격자 0.35–1.20 · FALLBACK 0.675 · 보관 앵커 0.675 ⇒ **현실 범위 전체가 도달불가 영역**. 탈출하려면 I3가 앵커의 **3×**를 내야 한다 |
| **U7** | `PDMUX_LAMBDA0_INSTR`(어디를 볼지) | 존재하는 다른 디렉터리 지정 | ★**반전** shape A: 참 λ*(A)=5.0에서 앵커(λ_inf=2.1) → `KNEE_NOT_BRACKETED`(λ* 미보고) / FALLBACK → `KNEE_BRACKETED(λ*=5.0)` | 쉬핑 코드 스윕 2종. **감사 시점 현 트리에서 기본 INSTR 디렉터리는 존재하지 않았다** ⇒ 그 시점 제출이면 FALLBACK이 실행 경로 |
| **U8** | 측정창 rung간 이질성(600–700 s vs 170 s) | 등록값 | **반전 없음 — caveat** | shape A 고측(T=170)의 "≤0.90"은 실효적으로 "요청 e2e ≥ 18.9 s ⇔ ITL ≥ 37 ms ⇔ B̄ ≳ 34(=0.72×48)". 같은 셀이 T=600이면 0.968을 읽는다. 자기정합 모형 스윕에서 보고 λ*(A)는 커버리지 밴드 안에서 **오차 0.0%**였고, 밴드 바로 위에서만 −10.6%(FALLBACK, 참 λ*=8.0 → 7.155 + `KNEE_BRACKETED`) ⇒ 死因 아님, **등록 caveat**(§7 R3C-3) |
| **U9** | 예산 모형(포화 셀을 `N/x`로 가격) | λ*/λ_inf = 1.0 → 0.25(등록 커버리지 하한) | **반전 없음 — caveat**(수치 큼) | 등록 0.959–1.236 GPU-h vs 실제 1.05–1.41(λ*=λ_inf) / 1.34–1.98(0.5) / **2.03–3.51(0.25)**. `--time=04:30`은 최악 3.51을 덮지만 **요청 1.60 GPU-h는 2.2× 초과** |
| **U10** | 변이 하네스 도달범위 | 감사자 독립 변이 8종 | **반전 없음 — 그러나 5/8 escape** | Y1(achieved 출처)·Y2(`TTFT_LOAD_FACTOR` 3.2→1.0)·Y4(`load()` shape 폴백)·**Y5(`lambda0_cells.py`가 모든 rung을 인증)**·Y6(kappa/lowside 열 교환). **Y5는 이 판정서의 死因을 shape A에까지 복제하는 변이인데 36/36 하네스를 그대로 통과한다** |

---

## 2. 死因 상세

### N3 (1) — F2("shape B = `KNEE_BRACKETED`")의 참 분지가 정의역 ∅

**사실 1 (항등식, 재확인).** `achieved/realized_offered = (N/(N−1))·span/duration`. 보관 12/12 셀에서 양변 차이 **2.2e−16**(등록 주장 1e−12보다 강함). `duration = maxᵢ(arrivalᵢ + ttftᵢ + Σitlᵢ)`는 **+0.072 … +0.265 s**로 재현(12/12, 부호 일관 — `/server_info` GET이 `benchmark_duration` 안에 있다, `bench_serving.py:1403-1426`). **rev3의 §3.1은 참이다.**

**사실 2 (등록이 스스로 인쇄한 것).** §3.5는 6개 시나리오 **전부**에서 shape B에 대해 `low-side candidates: ['b_r0','b_r1','b_r2','b_r3']`을 인쇄한다(prereg `:248,:277,:306,:335,:364,:393`). 즉 **B의 모든 rung이 인증된다.**

**사실 3 (코드).** `lambda0_label.py:94-97`의 배수 가드는 `low_side_candidate`가 참인 **모든** 셀에 걸리고, 위반 1건이면 `:126-136`에서 **그 shape 전체가 `UNRESOLVED`**가 된다. 인증은 `lambda0_plan.py:233-243`에서 "저측이 될 **수 있다**"는 자격이지 "저측으로 **쓴다**"는 역할이 아니다.

**사실 4 (산술, 모형 무관).** R1 고측 = `ratio ≤ 0.90` ⇔ `drain ≥ ((N/(N−1))/0.90 − 1)·span`. shape B 24/24 셀에서 그 값은 **19.24–34.46 s**, 가드는 **8.34–8.50 s**. ⇒ **고측을 만족한 셀은 반드시 R0을 위반한다.**

**사실 5 (실행).** 쉬핑 `analyze()` → `rule()`에 물리 모형으로 만든 셀을 넣었다:

| 참 λ*(B) | 0.10 | 0.30 | 0.675 (앵커) | 1.00 | 1.10 | 2.00 |
|---|---|---|---|---|---|---|
| **쉬핑 rev3** | UNRESOLVED | UNRESOLVED | **UNRESOLVED** | UNRESOLVED | LADDER_TOO_LOW | LADDER_TOO_LOW |
| 처방 수리 후(§6 E1) | LADDER_TOO_HIGH | **KNEE_BRACKETED(0.300, err +0.0%)** | **KNEE_BRACKETED(0.675, err +0.0%)** | KNEE_BRACKETED(1.000) | LADDER_TOO_LOW | LADDER_TOO_LOW |

⇒ **쉬핑 rev3는 λ*(B) ∈ [0.3, 1.0] 전 구간(등록 점추정·보관 앵커 포함)을 `UNRESOLVED`로 버린다.** F2는 참이 될 수 없고, R3/F4[B]도 `lam is None`이라 D21에 의해 자동 `UNRESOLVED`다. 1차 범위(Claim D+P3)의 P2는 W4 prefill phase = shape B를 요구하므로 **이 단계는 약 1.2–3.5 GPU-h를 쓰고 목적의 절반을 구조적으로 달성하지 못한다.**

### N3 (2) — `LADDER_TOO_HIGH`(÷4)가 양 shape에서 정의역 ∅

`ladder[0]`(최하단)은 **정의상 항상 인증된 저측 후보**다(계획이 하위 2 rung을 κ ≥ 0.97까지 연장하고 selftest 7이 `window_unsatisfiable` 부재를 assert). 그 셀이 포화를 보이려면 배수 ≥ `L90`(A 67.6–78.3 s, B 20.4–34.5 s)인데 가드는 `2·L̂`(A 22.0–38.6 s, B 8.3–8.4 s)에서 먼저 발화한다 ⇒ **"사다리가 너무 높다"는 관측은 언제나 `UNRESOLVED`로 기록된다.** 스윕 확인: λ*(A) ≤ 1.2 → `UNRESOLVED`, λ*(B) ≤ 0.2 → `UNRESOLVED`. 따라서 §3.6-2가 "가정을 데이터로 검사되는 분지로 바꾼다"며 등록한 두 분지 중 **÷4는 발화 불가**다.

### 왜 rev3의 36/36 하네스와 50 테스트가 이것을 못 봤는가 (진단)

- **end-to-end leg의 고측이 인증되지 않은 셀이다** — `lambda0_label.py:508-509`가 `low_side_candidate=False`로 a_r4를 만든다. 즉 e2e 배선 시험은 **가드가 물 수 없는 유일한 구성**(shape A)만 재현하고, **모든 rung이 인증되는 shape B 구성은 한 번도 e2e로 돌지 않는다.**
- **`LADDER_TOO_HIGH`를 증명하는 테스트가 손으로 만든 셀을 쓴다** — `tests/test_lambda0_prereg.py:188-190`은 `LABEL._cell(r, 0.70)`을 쓰는데 이 헬퍼는 `drain_ok=True`가 기본값이다. **실제 인증 rung에서는 존재할 수 없는 셀**이다(교훈 9 계열: 게이트가 자기 전제를 입력으로 받았다).
- **§3.3과 §3.4가 서로를 부정한다** — §3.3은 "고측 rung은 읽는 값이 0.5–0.7"이라 적고(=배수가 span의 0.4–1.0배, b_r3에서 **120 s**), §3.4는 같은 셀에 `drain ≤ 2·L̂ = 8.4 s`를 요구한다. 두 문장은 한 페이지 안에서 상호배타적이다.

---

## 3. ★감사자 자기 점검 (게이트 #113 · 자기 철회 6회차)

1. **이 死因의 근인은 감사자의 rev2 처방이다.** D16은 "저측 후보는 κ ≥ ACH_HI+0.02"를, D16 후반은 "**저측 후보 셀에 대해** `drain_s ≤ 2·L̂`를 요구하라"를 규정했다. **"저측 후보"가 계획 시점의 *자격*이지 실행 후의 *역할*이 아니라는 것**을 쓰지 않았고, 그 자격이 shape B에서 사다리 전체를 덮는다는 것을 계산하지 않았다. rev3는 그 문언을 충실히 구현했다. **처방은 국소, 파급은 전역**(교훈 85)의 재발이며, 처방자가 함께 죽은 두 번째 사례다.
2. **중간 주장 1건을 철회한다.** 감사 중간에 "shape A의 `LADDER_TOO_LOW`가 6/6에서 정의역 ∅"라고 계산했다(상단 rung의 물리 천장 0.73–0.94 < 0.95). 그 계산은 **고측 rung이 초임계일 때 읽는 값이 λ*/x라는 것을 무시**했으므로 **틀렸다** — 자기정합 모형 스윕에서 λ*(A)=4.0은 실제로 `LADDER_TOO_LOW`를 낸다. 판정서 본문에서 제외했다. (남은 참인 부분은 U8의 caveat뿐이다.)
3. **아래 E1 처방은 직접 실행해 도달가능성을 확인했다**(§2 표의 "처방 수리 후" 행). 새 상수·새 estimand를 도입하지 않는다.

---

## 4. D16–D23 집행 결과 (문안이 아니라 코드·데이터로)

| D | 상태 | 확인 |
|---|---|---|
| **D16** (영점 등록) | **닫힘(저측) / 死因의 절반** | κ 산식은 보관 12셀에서 **항등식**(2.2e−16). rung별 인쇄·`low_side_candidate` fail-closed ✓. **그러나 인증이 "자격"이라 고측 rung까지 덮고, 그것이 死因** |
| **D17** (측정창) | **닫힘** | 하위 2 rung만 600–1800 s 중 최단, `N_MAX_LONG=2000`, 고측 미변경 ✓. 음성대조(selftest 5): rev2의 170 s 저측 rung이 **전부 거부**됨을 재현 ✓ |
| **D18** (fallback 객관 술어) | **닫힘(재량 제거) / 입력 표면 잔존** | 운영자 스위치 삭제 ✓. selftest가 **각 조건 단독으로 FALLBACK 강제**를 실행으로 보임 ✓(재현함). 잔여: `PDMUX_LAMBDA0_INSTR`가 **어느 디렉터리든** 가리킬 수 있고 I3 산출물의 sha가 기록되지 않는다(U7). `_last_int_per_line`은 "줄의 마지막 정수"를 읽으므로 `#running-req: N` 형식(newpair `:342` 등록)엔 맞지만 형식 변경에 취약 |
| **D19** (사각지대 정리 철회) | **닫힘** | §5.1에서 철회 + `lambda0_plan.py` selftest 11이 반례를 상시 보관 ✓ |
| **D20** (rr=1.0 = 재현 필요조건) | **닫힘 + 독립 확인** | 경로를 직접 확인: `np.random.seed`는 `bench_serving.py:1705-1706`, `get_dataset`는 `:1848`(**seed 이후**), np.random 소비처는 `datasets/common.py:60`(rr=1.0에서 퇴화)와 `random.py:147`(random_sample=False 분기만)·`:1354`(LoRA). `random.shuffle`은 파이썬 스트림이라 무관 ⇒ **rr=1.0은 실제로 필요조건**이고 analyzer가 되읽어 `UNRESOLVED` ✓. §4.4의 옛 서술 철회도 반영 ✓ |
| **D21** (미평가 ≠ 반증) | **닫힘** | `rule()`이 `lam is None`에서 `UNRESOLVED (not evaluated)` ✓, selftest (10) ✓ |
| **D22** (하네스 확장) | **부분** | 3모듈 36종 + CONTROL 2종 **재현**(escapes none, 81 s). **그러나** ①`lambda0_cells.py`가 MODULES 밖이고 sbatch도 그 selftest를 돌리지 않는다 ⇒ Y5/Y6 escape ②감사자 8종 중 **5종 escape** ③`lambda0_mutation_check.py:156`의 `r = _run(None,None,env)`는 **결과가 즉시 덮여 버려지는 죽은 CONTROL**이다 — 주입 확인: temp-dir 사본 경로가 깨지면(모듈 1개 누락) 36종 전부가 "FAILS (good)"을 찍고 **in-place CONTROL은 PASSES를 찍는다**(공허한 36/36 가능). 오늘은 공허하지 않음을 감사자 escape 5건이 증명하지만, **CONTROL이 그것을 인증하지는 못한다** |
| **D23** (미시도 shape도 EXPECT) | **닫힘** | `lambda0.sbatch:141-155`가 루프 **이전에** plan.json에서 EXPECT/REPEAT 구성 ✓, 테스트로 고정 ✓ |

---

## 5. 반증 실패 항목 (공정 기록 — 아래 E조건 전에는 `CONFIRMED` 아님)

- **배수 항등식·duration 재구성**: 12/12 비트 수준 재현(2.2e−16 / +0.072–0.265 s). **rev3 §3.1은 참이다.**
- **배수 ≈ 마지막 도착 요청의 e2e**: 12/12에서 최대 오차 0.15 s — 등록의 물리 모형은 옳은 모양이다.
- **`ITL_SOLO_S = 13.06 ms`**: rev2 감사가 "무관한 두 상수"라 지적한 것 중 이 값은 **옳다**(D16/D44/D92 마지막 요청 13.48/13.27/13.07 ms, B=1에서 arm 무관). 계획의 `L̂`는 그보다 큰 step 모형을 쓰므로 κ가 **보수적**이다 — 안전한 방향.
- **κ assert의 항등식 탈출 수리**: selftest 7이 `KAPPA_MIN`이 아니라 리터럴 `ACH_HI + 0.02`를 assert ✓. **다른 변이**(Y3: `+0.02 → +0.001`)로 독립 확인 — 차단됨.
- **R1/R5 실질 승계**: 보관 cell JSON 직접 투입으로 λ* 0.9331188685063582 / 0.675302644015995 / 0.18668462822130108 재현(selftest 0) ✓.
- **CPU 회귀 50/50 OK** · **변이 36/36 + CONTROL 2/2** 재현 ✓.
- **D4 음성대조**(rev1 glob 재현이 틀린 답을 냄) ✓ · **ONE BOOT PER CELL 기제 승계** ✓ · **`CONSENSUS §3 항목120` 회피**(판정량 전부 client-side) ✓.
- **인용 고정(게이트 #110)**: HEAD가 `bddff6a`→`b2b60e7`로 이동했으나 인용 파일 중 바뀐 것은 `sync_engine_tree.sh` 하나(24→**25항목**, append-only)뿐이고 D12 게이트는 `≥24`이므로 **여전히 통과**. 나머지 6개 인용 파일은 `same`.
- **arm 비교 금지·성능 판정 0건** 유지 — 이 단계는 어떤 결과가 나와도 HE0·정책 순위·Claim D/E 등급·stake #1을 움직이지 않는다.

---

## 6. 제출 전 필수 조건 (E1–E7) — rev4

**死因 해소(필수)**

- **E1 (N3, 핵심 · 한 곳)** **배수 가드는 셀을 저측에서 실격시켜야지 shape를 무효화해서는 안 된다.** `lambda0_label.rule`에서 `drain_model_ok`를 R0(shape 유효성)에서 **빼고** 저측 자격 술어로 옮겨라: `usable_low = low_side_candidate AND drain_model_ok`. 고측(`ratio ≤ 0.90`)에는 배수 조건을 걸지 않는다 — 배수가 예측을 넘었다는 것은 **바로 그 셀이 포화했다는 증거**이지 측정 실패가 아니다. R0에는 `ebar_guard_ok`·`range_ratio_ok`만 남긴다.
  *(실현가능성 — 감사자가 실행해 확인: shape B가 λ*(B) ∈ [0.3, 1.0]에서 `KNEE_BRACKETED` + **오차 0.0%**, ≤0.2에서 `LADDER_TOO_HIGH`, ≥1.1에서 `LADDER_TOO_LOW` ⇒ 4개 verdict 전부 정의역 비어있지 않음. shape A는 브래킷 밴드가 [1.5,3.5] → **[0.8,3.5]**로 넓어지고 `LADDER_TOO_HIGH`도 살아난다. 새 상수 0개, 새 estimand 0개, GPU 0.)*
- **E2 (도달가능성 사전등록의 일반화)** κ 표에서 멈추지 말고, **참 λ* 격자 → 방출 verdict** 표를 6개 시나리오 × 2 shape 전부에 대해 **실행 전에 인쇄**하고, 등록된 4개 verdict가 **각각 비어있지 않은 정의역**을 가짐을 assert하라(이 판정서 §2의 표가 그 형식이다). 등록에 쓸 물리 모형은 이미 등록돼 있다(step 모형 + `ITL_SOLO_S` + Little). GPU 0.
- **E3 (하네스 도달범위)** `lambda0_cells.py`를 `MODULES`에 넣고(그러면 `tests/test_lambda0_prereg.py:369-376`의 "정확히 3모듈" assert도 함께 고쳐야 한다 — 그 테스트는 현재 **하네스의 사각지대를 상수로 못박고 있다**), sbatch에 `lambda0_cells.py --selftest`를 추가하라. 감사자 변이 **Y1·Y2·Y4·Y5·Y6**를 등록 변이에 추가하라. `lambda0_mutation_check.py:156`의 죽은 `_run(None,None,env)`를 **실제 CONTROL로 채점**하게 고쳐라(temp-dir 경로 공허성 차단).
- **E4 (e2e leg의 사각지대)** `_end_to_end`에 **shape B 구성**(모든 rung 인증 + 포화한 상단)을 추가하라. 현재 leg는 고측이 `low_side_candidate=False`인 유일한 구성만 돈다(`lambda0_label.py:508-509`).

**등록 완결성(필수)**

- **E5 (예산)** 포화 rung을 `N/x`가 아니라 `N/λ*`로 가격하고, 등록 커버리지 하한(λ*=0.25·λ_inf)까지 표로 등록하라: **2.03–3.51 GPU-h**. 요청 예산 1.60을 그 값으로 올리거나, 커버리지 하한을 좁혀 등록하라. `--time=04:30:00`은 최악 3.51 GPU-h를 덮으므로 유지 가능. 또한 §3.3·§7의 "**0.96–1.47 GPU-h**"는 자기 §3.5 인쇄(**0.959–1.236**)와 불일치 — 인쇄가 정본이므로 문안을 고쳐라.
- **E6 (앵커 provenance)** `PDMUX_LAMBDA0_INSTR`로 해석된 **절대경로 + 3개 I3 산출물의 sha256**을 `LAMBDA_INF_DECISION.txt`와 `LAMBDA0_LABEL.json`에 기록하라(U7: 디렉터리 선택이 shape A의 라벨을 뒤집는 실례가 있다).
- **E7 (승계)** ①newpair 판정서 §5 **NPC-I**를 §10 필수 병기에 문자 추가: "**λ_inf는 legacy warm-up boot · D44 · cudagraph ON에서만 측정되며, B4(true-dual)의 포화 상한은 측정되지 않았다.**" ②Q5에서 빠진 절 "(이 단계 자신이 두 shape에서 다른 값을 낸다)"와 L3에서 빠진 "3b"를 복원하라(인용 금지 문안은 **문자 그대로** 승계한다).

---

## 7. 인용 금지 (이번 판정서가 추가 — 결과 문서·정본이 그대로 승계할 문안)

> **R3C-1** — **rev3(그리고 그 이전 어떤 판본)의 규칙으로는 λ*(shape B = 8192-in)를 측정할 수 없다.** 계획이 B의 네 rung을 모두 저측 후보로 인증하고 R0이 인증 셀에 `drain ≤ 2·L̂`를 걸기 때문에, R1의 고측(`ratio ≤ 0.90` ⇔ `drain ≥ 19.2–34.5 s`)을 만족하는 셀은 **반드시** R0(8.34–8.50 s)을 위반한다. **`KNEE_BRACKETED[B]`는 λ_inf(B) ≤ 1.70 req/s인 모든 경우에 정의역이 ∅이다** — 등록 격자(0.35–1.20)·FALLBACK(0.675)·보관 앵커(0.675) 전부가 여기 들어간다. **"0단계가 W4 prefill phase의 부하 라벨을 정했다"는 문장은 쓸 수 없다.**

> **R3C-2** — **`LADDER_TOO_HIGH`(÷4 재설계)는 rev3에서 한 번도 발화할 수 없다**(A·B 공통: 최하단 rung은 항상 인증돼 있고, 그 셀이 포화를 보이면 가드가 먼저 물어 `UNRESOLVED`가 된다). 따라서 **"rev3는 'λ_inf가 λ*를 아래에서 묶는다'는 가정을 데이터 검사 분지로 대체했다"(§3.6-3)는 문장을 인용하는 것을 금지한다** — 그 분지는 절반이 죽어 있다.

> **R3C-3** — **rev3의 브래킷 양변은 측정창이 서로 다르다**(저측 600–700 s, 고측 170 s). shape A 고측의 "≤0.90"은 실효적으로 "요청 e2e ≥ 18.9 s ⇔ decode ITL ≥ 37 ms ⇔ 평균 배치 ≳ 34(= max_running 48의 0.72배)"이며, **같은 셀을 600 s 창에서 재면 0.968을 읽는다**. 따라서 **보고되는 λ*(A)를 "용량"이라고 인용할 때는 (창 길이, N) 튜플을 함께 적어야 하고**, 사다리 상단 바로 위 구간에서는 과소평가가 발생한다(자기정합 모형: 참 λ*(A)=8.0 → 보고 7.155, **−10.6%**, 라벨은 `KNEE_BRACKETED`).

> **R3C-4** — **"변이 36/36 차단, escape 0"은 `lambda0_label.py`·`lambda0_analyze.py`·`lambda0_plan.py` 세 파일 한정 사실이다.** 결정 경로에는 `lambda0_cells.py`(인증 플래그를 analyzer로 나르는 렌더러)가 포함되는데 하네스가 변이시키지 않고 sbatch가 그 selftest를 돌리지도 않는다. 감사자 독립 변이 8종 중 **5종이 escape**했으며, 그중 **Y5(`int(bool(low_side_candidate))` → `1`)는 이 판정서의 死因을 shape A까지 복제하면서 36/36을 그대로 통과한다**. **"결정 경로 전체가 변이 인증됐다"고 인용하는 것을 금지한다.**

**기존 인용 금지는 전부 유효하며 승계 확인됨**: Q1–Q4 문자 동일 ✓, Q5·L1·L3 경미 드리프트(E7-②로 복원), L2 문자 동일 ✓. ★**게이트 #6 불충족 문장은 살아 있다**(§0 ★★ 문단 · §9 Q1 · `test_states_that_gate_6_is_not_closed`) — **다음 판본에서도 반드시 유지할 것.**

---

## 8. 신규 게이트 제안 (doc-steward용)

- **G-λ3-1** — *자격(eligibility)과 역할(role)을 구분하라.* 계획이 "이 셀은 X가 **될 수 있다**"고 인증하면, X 전용 가드가 **X가 아닌 역할로 쓰인 같은 셀에도** 걸린다. rev3에서 저측 자격이 사다리 전체를 덮어 고측 증거를 측정 실패로 폐기했다.
- **G-λ3-2** — *결정 규칙을 등록할 때는 문턱의 도달가능성을 **모든 verdict에 대해** 사전 계산하라.* rev3는 저측(κ)만 계산했고, 그 계산이 만든 새 도달불가를 보지 못했다(감사자 처방 D16의 granularity 결함).
- **G-λ3-3** — *같은 문서의 두 문장이 같은 셀에 상호배타적 요구를 하는지 대조하라.* §3.3("고측은 0.5–0.7을 읽는다" = 배수 120 s)과 §3.4("배수 ≤ 2·L̂ = 8.4 s")가 그 예다.
- **G-λ3-4** — *분지 존재 증명에 손으로 만든 레코드를 쓰지 마라.* `LADDER_TOO_HIGH` 테스트가 `drain_ok=True` 기본값 셀을 썼는데, 실제 인증 rung에서는 그런 셀이 존재할 수 없다(교훈 9 계열, 21번째 재발).
- **G-λ3-5** — *변이 하네스의 CONTROL은 변이가 실제로 도는 경로(temp-dir 사본)에서 채점하라.* rev3는 그 호출을 하고도 결과를 버린다(`lambda0_mutation_check.py:156`) — 주입 확인: 사본 경로가 깨지면 36/36이 공허하게 통과한다.

---

## 9. 판정 요약

`NO-GO` — **N3**(도달불가 2행: ①`KNEE_BRACKETED[B]`·`LAMBDA_STAR[B]` 정의역 ∅ [2·L̂ 8.34–8.50 s vs 필요 배수 19.24–34.46 s, 24/24 셀, λ_inf(B) ≤ 1.70 전 구간; 쉬핑 코드 스윕서 λ*(B)=0.675 → `UNRESOLVED`] ②`LADDER_TOO_HIGH` 정의역 ∅ [A·B 공통, 6/6]), 부수 **N1**(이 단계 두 산출물 중 하나가 원리적으로 산출 불가).

**D16–D23 중 6건은 코드·데이터로 닫힌 것을 확인**했고 배수 항등식·rr=1.0 필요조건·D21·D23·D4 음성대조·인용 고정은 건강하다. **死因은 단일 지점(E1)이며 GPU 0으로 수리·검증된다** — 감사자가 처방안을 실행해 4개 verdict 전부의 정의역이 비지 않음을 확인했다. E1–E7 적용 후 재감사에서 `GO-with-caveats`가 가능하다고 본다(잔여 caveat: U8 측정창 이질성, U9 예산, U7 앵커 입력 표면 — 전부 R3C-1…4로 문안화 가능).

**이 판정서의 死因은 상당 부분 감사자 자신의 rev2 처방(D16/D17)이 만든 것이다**(§3). 다음 판본은 **E1을 등록 상수로 받아들이지 말고 §2의 도달가능성 표를 스스로 재생성해 검증**해야 한다(게이트 #110).

---

### 관련 파일 (전부 절대경로)

- 감사 대상: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_eval/lambda0_prereg/PREREG_LAMBDA0_REV3_2026-09-13.md`(§3.2 `:158-186` · §3.3 `:187-201` · §3.4 `:203-218` · §3.5 인쇄 `:226-400` · §3.6 `:402-419` · §3.7 `:421-441` · §5 `:503-561` · §7 `:581-603` · §9 `:641-693` · §10 `:697-710` · §11.1 `:716-742`)
- 실행체: 같은 디렉터리 `lambda0_plan.py`(`:101-120`, `:225-246`) · `lambda0_label.py`(`:82-98`, `:101-204`, `:473-539`[특히 `:508-509`]) · `lambda0_analyze.py`(`:107-182`) · `lambda0_cells.py`(`:35-52`) · `lambda0_lambda_inf.py`(`:55-96`) · `lambda0_mutation_check.py`(`:36`, `:132-134`, `:148-182`[특히 `:156`]) · `lambda0.sbatch`(`:94-139`, `:141-155`, `:167-265`, `:272-274`)
- 테스트: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/tests/test_lambda0_prereg.py`(`:180-190`, `:356-380`) — 50/50 OK
- 직전 판정서: `.../lambda0_prereg/VERDICT_lambda0_rev2_2026-09-13.md` · `.../VERDICT_lambda0_rules_2026-09-13.md`
- 반전 계산 원자료: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/c_905835/`(`bench_*.jsonl` 12 · `cell_*.json` 12) · `.../probes/c_capacity.sbatch`(`:76`, `:111`, `:123-178`, `:142-151`)
- 엔진 사실(독립 확인): `/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/bench_serving.py`(`:938-950`, `:1314`, `:1403-1426`, `:1705-1706`, `:1848`) · `.../sglang/benchmark/datasets/common.py:53-64` · `.../sglang/benchmark/datasets/random.py:57-147`
- 인접 사전등록: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/newpair_prereg/PREREG_NEWPAIR_2026-09-13.md`(`:310-311`, `:342`) · `.../VERDICT_newpair_rev2_2026-09-13.md:201`(NPC-I)
- provenance: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh`(HEAD `b2b60e7`에서 **25항목**, `models/nemotron_h.py` 포함 ⇒ D12 `≥24` 통과)
