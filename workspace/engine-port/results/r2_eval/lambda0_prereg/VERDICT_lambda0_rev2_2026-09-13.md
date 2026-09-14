# 규칙층 재감사 판정서 — 캠페인 0단계 λ* 사전등록 **rev2**

**대상**: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_eval/lambda0_prereg/PREREG_LAMBDA0_REV2_2026-09-13.md`
+ 동봉 `lambda0_plan.py` · `lambda0_analyze.py` · `lambda0_label.py` · `lambda0_mutation_check.py` · `lambda0.sbatch` · `workspace/engine-port/tests/test_lambda0_prereg.py`
**출발점**: `.../lambda0_prereg/VERDICT_lambda0_rules_2026-09-13.md` (rev1 `NO-GO`, D1–D15)
**감사일** 2026-09-13 · **GPU 지출 0** · 트리 상태: HEAD `bddff6a`, 작업트리 미커밋 12 M / 9 ??

## 0. 등급

# `NO-GO`

**死因 1건(반전 2행)**: **N2** — 등록된 **판정량 자신**이 남긴 자유 표면(비포화 셀의 **영점(null) 기준선**)이 shape A의 라벨을 실제로 뒤집는다. 수치 있음. 두 번째 반전행은 **fallback 분기 선택이 재량**이라는 것(수치 있음).

이 `NO-GO`는 rev1과 성격이 다르다. **D1–D15 중 13건은 실제로 닫혔고(코드·데이터로 확인)**, 측정 격자·사다리 결정성·변이 하네스·sbatch·provenance 게이트는 건강하다. 死因은 **rev2가 D2를 고치면서 새로 도입한 추정량**에 있다: `achieved/realized_offered`는 포화도가 아니라 **`span/duration`(배수 소요시간의 함수)** 이며, 그 **비포화 기준선이 shape에 따라 0.93–1.02로 움직인다.** rev2는 이 추정량을 **배수가 무시할 만한 shape(8192-in, out 96)에서만 검증**하고, **배수가 지배하는 shape A(out 512)** 에 그대로 적용했다. shape A는 앵커가 없고 F3가 "구성상 성립"에만 기대는 바로 그 shape다.

---

## 1. 반전 시험 표 (필수)

반전 계산에 쓴 원자료·추정량을 먼저 고정한다. **원자료** = `results/longctx_conflict/probes/c_905835/{bench_*.jsonl, cell_*.json}` (보관 12셀) + `lambda0_plan.py`의 등록 상수. **추정량** = 이 등록 자신의 `lambda0_analyze.py:67-74, 89, 114-117`.

**먼저 판정량의 닫힌 형태를 원자료로 확립했다(새 자유 표면 아님 — 항등식이고 12/12에서 검증됨).**
보관 12셀 전부에서 `duration = max_i(arrival_i + ttft_i + Σitl_i)`가 기록된 `duration`을 **≤0.27 s** 오차로 재현한다(최대 오차 d92_r0 0.27 s, 나머지 ≤0.15 s). 따라서

> **`achieved_over_realized` = (N/(N−1)) · span / (span + L)** ,  L = 배수(drain) = `duration − span`

즉 **비포화 셀에서 이 값은 1이 아니라 `1/(1+L/span)`이다.** 저측 문턱 0.95 도달 조건은
`L/span ≤ N/(0.95(N−1)) − 1` (N=140에서 **6.02%**, N=400에서 5.53%).

| # | 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 |
|---|---|---|---|---|
| **T1** ★ | **판정량의 영점(비포화 기준선)** — 등록은 이것을 1로 가정한다(§3.2 닫힌 형태, `lambda0_plan.py:349 covers_registered_band`, F3 "구성상 성립"). 실제 값은 `1/(1+L/span)`이고 L은 **decode ITL**이 정한다. 등록 문서가 **두 개의 ITL 상수를 동시에** 들고 있다 | `ITL_SOLO_S = 13.06 ms`(`lambda0_plan.py:92`, warmup·예산 모형) ↔ 등록이 λ*(A) 사전추정에 쓴 회귀 `step(B)=19.82+0.5174·B ms`(§3.3) | ★**반전** shape A `KNEE_BRACKETED → KNEE_NOT_BRACKETED`, λ*(A) `값 → None`, **F3 REFUTED** | 시나리오 "point lower"(λ_inf=2.10) 사다리의 실현 비율: ITL 13.06 ms → **[0.9685, 0.9659, 0.8743, 0.696, 0.504] = BRACKETED**; ITL 19.90 ms(probe C가 **이 arm(D44)** 에서 실측한 ITL p50) → **[0.9494, 0.9472, …] = NOT_BRACKETED**; ITL 24.06 ms(B=6 실측 step) → [0.9382, 0.9362, …] = NOT_BRACKETED. **5개 등록 시나리오 중 4개 + fallback이 ITL 19.9 ms에서 전부 NOT_BRACKETED** (예외: "absolute lower"만 0.9533로 생존) |
| **T1′** ★ | 같은 표면 — **등록 자신의 모형으로만** 밀기 | 등록의 λ*(A) 유도식을 그대로 사용 | ★**반전(자기모순)** F3의 참 분지가 **등록 자신의 모형 아래 도달 불가** | 등록은 λ*(A)=48/(512·step(48)), step(48)∈[37.4, 44.65] ms로 λ*(A)=2.10–2.51을 얻는다. **같은 회귀를 저측 rung의 배치(B≈10, Little: 0.83 req/s × L≈12 s)에 넣으면 step = 23.3–25.0 ms** ⇒ L = 11.9–12.8 s ⇒ L/span = 7.1–7.6% **> 6.02% 허용치** ⇒ `r(a_r0) = 0.940 / 0.938`. 저측이 0.95를 넘으려면 B≈10에서 step ≤ **19.69 ms**가 필요한데 등록된 절편만 19.82 ms다 ⇒ **음의 기울기를 요구**한다 |
| **T2** ★ | **fallback 분기 선택** (§3.4 "워크스트림 A가 λ_inf를 주지 못하면"; 실행체에서는 운영자 env `PDMUX_LAMBDA0_FALLBACK=1`, `lambda0.sbatch:104`) | 등록이 객관 판정 기준을 주지 않으므로 두 분지 모두 허용 | ★**반전** shape B `BRACKETED ↔ NOT_BRACKETED` | λ_inf(B)=1.20(등록 "absolute upper")에서 앵커 창 `[0.5157, 1.835]`, fallback 창 `[0.4302, 1.026]`. λ*(B)=1.10이면 **앵커=BRACKETED / fallback=NOT_BRACKETED**. §3.4가 이 약점("창 상단 1.035가 절대 상한 1.20을 못 덮는다")을 **자백**하고 있으므로, 분기 선택이 재량인 한 자백은 방어가 아니다 |
| **T3** | shape B의 같은 영점 문제 | ITL 20 ms·TTFT 2.84 s(probe C 실측) | **반전 없음** | shape B rung0의 임계 ITL은 **153–302 ms**(5 시나리오), 실제 L = 4.1 s vs 허용 L_crit = 10.6–19.9 s ⇒ **2.6–4.9× 여유**. 그래서 probe C 원자료에서는 이 결함이 보이지 않았고, rev2가 "청정 3/3이 1.00±0.02로 수렴"을 근거로 전이한 것이 바로 이 비대칭을 놓친 지점이다 → **caveat** |
| **T4** | 재설계 1회(§3.5)가 T1을 구제하는가 | λ_inf ← 관측 최대 achieved | **구제 불가(반전 아님, T1 보강)** | `N = round10(min(x,λ_inf)·170)` ⇒ `span = (N−1)/x ≈ 170 − 1/x` — **사다리 배율과 무관하게 항상 ≈167–170 s**. L은 shape가 정한다 ⇒ **L/span은 재설계에 불변**. λ_inf가 크면 N_MAX=400 클램프가 창을 **더 짧게** 만든다("absolute upper": span 138.8 s, 임계 ITL **14.95 ms**) ⇒ 재설계는 더 나빠질 수 있다 |
| **T5** | R1 문턱 0.95/0.90 | 0.95→1.05, 0.90→0.80 | **반전 없음(probe C류 데이터)** | rev1 S4 재확인. 단 shape A 예측 사다리에서는 저측 문턱이 **데이터 위에 정확히 앉는다**(0.938–0.949 vs 0.95) — 이것이 T1의 다른 표현이며, 게이트 #6/#12가 말하는 **metric cliff(confound #5)** 형태다 |
| **T6** | `max(saturated)` vs `min` | M1 | **반전 없음** | 등록이 합성 쌍둥이(selftest 3b)로 고정 ✓. 단 **보관 데이터 쪽 근거(§5.2(i) "d44가 실제로 포화 셀 2개")는 배수 인공물 위에 서 있다**: `d44_r1_o96`의 실현 비율 **0.8988**(포화)은 이상값 `λ*/x_real = 0.9295`(사각지대)에서 **−3.07 pp** 밀린 결과다. 배수가 3.4%만 작았으면 `n_saturated_cells == 1`이 되어 M1이 다시 escape한다 → **caveat** |
| **T7** | seed 선택 규칙(1..5000, 동률 시 작은 seed) | 규칙 전 범위 | **반전 없음** | 독립 재현: `lambda0_plan.py --selftest` 통과, §3.4 인쇄가 생성기 출력과 **173/173행 바이트 동일**. Ē는 (seed,N)만의 함수임을 재확인 |
| **T8** | 예상 셀 목록 vs glob | 상단 셀 삭제 주입 | **반전 없음(수리 확인)** | 주입 실험 재현: 현행 → `UNRESOLVED`, rev1 glob 재현 → `KNEE_NOT_BRACKETED`(음성대조 존재) ✓ |

---

## 2. 死因 상세

### N2 (1) — 판정량의 영점이 미등록이고, shape A에서 라벨을 뒤집는다

**사실 1 (항등식, 원자료 12/12).** `achieved_over_realized = (N/(N−1))·span/(span+L)`. 비포화 셀에서도 1이 아니다. 보관 셀의 `L/span`: 0.72% / 1.52% / 1.73% / 7.32% / 9.29% / 12.19% / 51–60%.

**사실 2 (rev2의 검증이 닿지 않은 영역).** rev2가 D2 수리의 증거로 쓴 셀은 전부 `(8192 in, 96 out)`이고 배수가 0.7–1.7%다. 내가 독립 재계산해 rev2의 표를 정확히 재현했다 — d16_r0 **0.9940**, d44_r0 **0.9955**, d92_r0 **1.0175**. 그러나 같은 추정량을 `(256 in, 512 out)`에 적용하면 배수는 **511 토큰 × ITL**이 되고, 등록된 측정창은 `T_MEASURE_S = 170 s` 고정이다.

**사실 3 (엔진 코드 — 어느 ITL이 지배하는가).** `lambda0.sbatch`는 `PDMUX_R2_POLICY=fixed`·`PDMUX_R2_FIXED_DSM=44`를 export하고 `PDMUX_STICKY_PARTITION`을 unset한다(probe C와 동일). `src/multiplex/multiplexing_mixin.py:1009`에서 `_slo_on = bool(PDMUX_SLO_SCHED) or self.r2_policy is not None` ⇒ **참**. `:1066-1107`이 prefill 스팬/admission recheck마다 `_r2_decide_idx`로 D44 인덱스를 꽂고, `adjust_stream_groups()`는 **`:1111-1124`의 `elif adjust_stream_group:`에서만** 호출된다(= split-prefill 재생성 또는 **decode 배치가 빈** 경우). shape A는 decode 배치가 비지 않으므로 **decode는 44 SM에서 연속으로 돈다**(같은 파일 `:263` 주석 "decode runs at D SM continuously"와 일치). ⇒ 지배 ITL은 13.06 ms(비분할 B=1)가 아니라 **19.8–25 ms(D44)** 다. probe C의 D44 셀 4개 실측 ITL p50 = **19.92 / 20.29 / 20.38 / 23.53 ms**.

**사실 4 (반전).** 표 T1·T1′. 5개 등록 시나리오 + fallback 전부에 대해 계산한 저측 임계 ITL:

| 시나리오 | shape A rung0 (x, N, span) | L_crit | **ITL_crit** | shape B rung0 ITL_crit |
|---|---|---|---|---|
| point lower | 0.829, 140, 167.6 s | 10.09 s | **19.69 ms** | 178.5 ms |
| point upper | 1.000, 170, 169.0 s | 9.95 s | **19.41 ms** | 168.6 ms |
| absolute lower | 0.434, 70, 158.9 s | 10.79 s | **21.06 ms** | 301.7 ms |
| absolute upper | 2.875, 400, 138.8 s | 7.67 s | **14.95 ms** | 153.3 ms |
| B=6 plateau | 1.553, 270, 173.2 s | 9.79 s | **19.11 ms** | 169.6 ms |
| fallback | 1.099, 190, 172.0 s | 10.01 s | **19.60 ms** | — |

⇒ **shape A의 저측 시험은 사실상 "이 arm의 decode ITL이 20 ms보다 낮은가"이고, 그 답은 이 arm의 유일한 실측에서 "아니오"다.** 이는 용량 시험이 아니라 분할(SM 배분)의 함수이며, 등록된 R1의 estimand와 다르다.

**결과 예보(내 계산, 등록 모형만 사용)**: shape A 사다리 = `[0.940, 0.929, 0.874, 0.696, 0.504]` ⇒ `KNEE_NOT_BRACKETED`, λ*(A) 미보고, F3 `REFUTED`, 재설계 1회도 T4에 의해 동일 실패 ⇒ **약 2 GPU-h를 쓰고 λ*(B)만 얻는다.** 1차 범위(Claim D+P3)는 W3 전부와 W4 decode phase에 λ*(A)를 요구하므로 **단계의 목적이 달성되지 않는다.**

### N2 (2) — fallback 분기가 재량이고, 라벨을 뒤집는다

§3.4·§10은 fallback을 "워크스트림 A가 λ_inf를 주지 못하면"으로만 규정하고, 실행체는 운영자 env 하나(`lambda0.sbatch:104`)로 분기한다. "주지 못했다"의 객관 판정(산출 파일 존재·I3의 자기 유효성 조건)은 등록돼 있지 않다. 표 T2가 그 재량이 shape B의 라벨을 뒤집는 실례를 수치로 준다.

---

## 3. D1–D15 집행 결과 (문안이 아니라 코드·데이터로 확인)

| D | 상태 | 확인 방법 / 잔여 |
|---|---|---|
| **D1** (N3) | **닫힘** | `workloads.py:159-160`을 HEAD `bddff6a`와 **작업트리 양쪽에서** 확인: `8192 if is_prefill else 256` / `64 if is_prefill else 512`. "근사"·5% 예보 삭제됨, 테스트로 고정 |
| **D2** (N2-S1) | **부분 — 그리고 새 死因의 출처** | seed 결정성 ✓(§3.4 인쇄가 생성기와 173/173행 동일), 명목-분모 인공물 제거 ✓(0.9940/0.9955/1.0175 독립 재현). **그러나** ①"`bench_serving.py`에서 `np.random` 소비처가 정확히 2곳"은 **경로 기준으로 거짓**이다 — `sglang/benchmark/datasets/common.py:60` `compute_random_lens`가 `np.random.randint`를 **run당 2회** 소비한다(`datasets/random.py:67-76`). 소비량이 0인 것은 **오직 `--random-range-ratio 1.0`이 구간을 퇴화**시키기 때문이며, 내가 직접 실행해 확인했다(rr=1.0: 스트림 불변 `[0.28891…]`, rr=0.9: `[0.40076…]`로 이동). ②`ebar_guard`는 **재현 스트림 자신에서** 계산되므로 **재현이 깨진 경우를 원리적으로 탐지할 수 없다** — §4.4가 가드에 부여한 의미("오프라인 재현이 깨짐 → UNRESOLVED")는 성립하지 않는다. ③영점 문제 = 死因 |
| **D3** (N2-S2) | **닫힘(형식)** | 사다리·N·seed·창이 λ_inf의 순함수, `--selftest`·pure-function 테스트 통과. 단 T1′이 F3의 도달가능성 논증을 무효화 |
| **D4** (S9) | **닫힘** | 주입 실험 재현(현행 `UNRESOLVED`, rev1 glob 재현 `KNEE_NOT_BRACKETED`) + `lambda0.sbatch:152`가 **boot 이전에** EXPECT에 이름을 넣는다 ✓. **잔여 구멍**: `:242-245`의 `FAIL_STREAK≥2 → break`(및 벽시간 소진) 시 **아직 시도되지 않은 shape는 EXPECT에 아예 없고** 라벨 JSON에서 **그 shape가 통째로 사라진다**(§5.1 출력공간에 없는 칸) |
| **D5** (S10) | **닫힘** | analyzer·sbatch 존재, `bash -n` 통과, `shape` 필드 기록 ✓ |
| **D6** | **닫힘** | probe C 배수 `{0.60,0.90,1.30}`가 `c_capacity.sbatch:76`(근거는 `:78-79`)임을 **내가 직접 열어** 확인 — rev2의 판정서 인용 정정 2건은 **옳다**(게이트 #110 준수 사례). fallback 창 `[1.045,7.20]`/`[0.4275,1.035]` 테스트로 고정 ✓ |
| **D7** | **닫힘(형식)** | F2 처분·F4 shape별 독립 등록 ✓. **잔여**: ①재설계 입력 "λ_inf ← 관측 최대 achieved"가 **shape별인지 합동인지 미지정** ②T4에 의해 재설계는 T1을 구제 불가 |
| **D8** | **닫힘(라벨) / 정리는 거짓** | 사각지대 칸이 §5.1과 코드(`blind_spot_cells`)에 있다 ✓. **그러나 §3.2 "동시에 두 rung이 사각지대에 들어갈 수 없다"는 실제 추정량에서 거짓** — 예보 사다리의 **최저 두 rung이 동시에** (0.90,0.95)에 떨어진다(0.940, 0.929). 이 정리는 이상화 비율 `min(1, λ*/x)`에 대해서만 참이고, selftest 8번은 MULT 상수만의 산술 항등식이다 |
| **D9** | **닫힘(등록 17종) / 경계 있음** | 재실행: **CONTROL PASSES, M1–M17 17/17 FAILS, escapes none** ✓. **CONTROL의 비공허성을 주입으로 직접 확인**: 아카이브 경로를 깨면 CONTROL이 `FAILS`, RC=2 ✓; 아카이브 부재 시 selftest가 `AssertionError`로 **하드 실패**(조용한 skip 아님) ✓ — 자기 보고한 escape 경로는 실제로 닫혔다. **잔여**: 하네스가 **`lambda0_label.py` 파일 경계 안에서만** 변이한다. 내가 독립으로 만든 6종 중 **5종이 escape**: X1(`_bracketed` 고측 문턱 ACH_LO→ACH_HI), X2(포화 집합 ACH_HI), **X3(실현 분모 (N−1)→N, 0.7–2.5% 이동 — 핵심 추정량 자신)**, X4(`EBAR_GUARD` 밴드 0.5/1.5), X5(analyzer의 `shape` 필드 무시). X6(재현 표본 수 n−1→n)만 차단. 특히 X1/X2는 **shape A가 떨어질 것으로 예보되는 사각지대 영역**에서만 살아나는 구별이라 게이트 #181·교훈 53 계열이 재발 |
| **D10** | **닫힘** | sbatch에 4 플래그 전부, analyzer가 `input_len_exact`/`output_len_exact` 기록 ✓ |
| **D11** | **닫힘 + 1건 해소** | 5행 표 등록 ✓. 미확인으로 남긴 pdmux config 차이를 **내가 확인**: `pdmux_homog5.yml`(sm_group_num 5) vs `pdmux_r2.yml`(6) — `(64,44)` 분할은 **양쪽에 존재**하고 `_r2_decide_idx`가 `decode_sms` 정확 일치로 고르므로 fixed 정책 아래서는 동등. 단 두 파일은 `manual_divisions` 길이가 달라 **fixed가 아닌 경로에서는 다른 분할로 간다**(homog5의 마지막 division = (16,92)) |
| **D12** | **닫힘 · 게이트 실발화 확인** | HEAD `bddff6a`의 `sync_engine_tree.sh` 매니페스트 = **24항목, `models/nemotron_h.py` 포함**(작업트리에서는 25항목 — `flashinfer_backend.py` 추가, **미커밋**). `exit 3` 게이트가 첫 boot보다 앞(테스트가 인덱스 비교로 고정) ✓. 사소: `N_MANIFEST=$(grep -c . f || echo 0)`는 빈 매니페스트에서 `"0\n0"`을 만들어 `[ -lt ]`가 오류가 되나, 앞선 nemotron grep이 먼저 `exit 3`한다 |
| **D13** | **닫힘** | 코드 3건 직접 확인: `bench_serving.py:70 BENCH_AIOHTTP_TIMEOUT_SECONDS = 6*60*60` ✓, `scheduler.py:2012-2014` `max_queued_requests is None → 제한 미적용` ✓, `server_args.py:1959` triton assert ✓. `--time=03:30:00`은 최악(λ*/λ_inf=0.45로 전 셀 지속 2.2배)에서도 ≈1.6 h로 충분 |
| **D14** | **닫힘** | §1이 정본 정의 교체를 정정으로 등재 ✓ |
| **D15** | **닫힘** | 이중 서술은 **사후 여지를 만들지 않는다** — 요청(1.15)과 예측(0.98)을 명시 분리했고, 예산에 걸린 등록 예보가 없다. 분해의 잔차(보관 12셀 `duration_s` 합 **2394.1 s**)를 독립 재현 ✓ |

---

## 4. 승계 확인

- **Q1–Q5**: Q1·Q2·Q4 **문자 그대로 승계**(공백 정규화 후 동일). Q3만 `workloads.py:150`에 `(@HEAD bddff6a)` 주석이 추가됐고, §8이 "인용문 자체는 불변"이라고 명시 — 승계 성립. Q5도 승계 성립(차이는 절 분할 아티팩트).
- **필수 병기 6항**: 승계 성립(항목 5에 rev2의 분모 교체 주기가 괄호로 추가된 것 외 동일).
- ★**게이트 #6 불충족 문장은 살아 있다**: §0(★★ 문단)·§8 Q1·`test_states_that_gate_6_is_not_closed`. **이 문장은 다음 판본에서도 반드시 유지할 것.**
- rev1은 `SUPERSEDED` 배너와 함께 보존 ✓.

---

## 5. 제출 전 필수 조건 (D16–D23) — rev3

**死因 해소(필수)**

- **D16 (N2-1, 핵심)** **비포화 기준선을 등록하고 도달가능성을 사전계산하라.** 각 rung에 대해 `κ = (N/(N−1))·span/(span+L̂)`를 **등록 step 모형으로 예측**해 표로 인쇄하고, **저측 후보 rung은 `κ ≥ ACH_HI + 0.02`를 만족해야 한다**는 조건을 등록하라. 이는 probe C prereg §3.2가 했고 rev1 판정서가 "빠진 유일한 구조적 요소"라 지목한 그 도달가능성 계산을, **이상화 비율이 아니라 실제 추정량 위에서** 하는 것이다.
- **D17 (N2-1, 실행)** 위 조건을 만족시키는 유일하게 값싼 길은 **shape A의 저측 rung에서만 측정창을 늘리는 것**이다(고측은 배수에 강건하므로 손댈 필요 없다). 권고: `T_MEASURE_A_low = 600 s`, 하위 2 rung만 적용, `N_MAX`를 그 rung에 한해 1200으로. 그러면 `L/span ≈ 12/480 = 2.5%` vs 허용 5.4% ⇒ **2× 여유**. **비용 = +2 셀 × ≈430 s ≈ +14분 ≈ +0.24 GPU-h**(총 ≈1.2 GPU-h, 이미 요청한 `--time=03:30:00` 안). *(대안 — 추정량 교체: 저측을 "대기 지연이 증가하지 않는다"로 바꾸는 길. probe C 원자료에서 분리력 10×[TTFT p50 2.84 s vs 29.65 s]가 확인되지만 **새 문턱·구간 정의가 새 자유 표면**이므로 그 자신이 도달가능성 사전계산과 함께 등록돼야 한다. 값싼 길은 D17이다.)*
- **D18 (N2-2)** **fallback 분기를 객관 술어로 등록하라.** 예: "`I3a_shapeA.jsonl`·`I3b_shapeB.jsonl`이 모두 존재하고 `I3_max_running_req`가 48에 도달했으면 앵커, 아니면 fallback" — 운영자 env 단독 분기(`PDMUX_LAMBDA0_FALLBACK=1`)를 판정 경로에서 제거하고 sbatch가 파일 존재로 스스로 판정하게 하라.

**등록 완결성(필수)**

- **D19** §3.2의 "사각지대 rung은 최대 1개" 정리를 **철회하거나**, 실제 추정량(κ 포함) 위에서 다시 증명하라. 현 정리는 MULT 상수의 산술 항등식이다.
- **D20** `--random-range-ratio 1.0`이 **길이 정확성만이 아니라 RNG 재현의 필요조건**임을 등록하고(`datasets/common.py:60`), analyzer가 bench 레코드의 `random_range_ratio`(레코드에 존재함)를 읽어 **1.0이 아니면 그 셀을 `UNRESOLVED`**로 만들라. 동시에 §4.4에서 "가드 위반 = 오프라인 재현 파손"이라는 서술을 **철회**하라(가드는 재현 스트림 자신에서 계산되므로 그 사건을 볼 수 없다).
- **D21** R3/F4가 `lambda_star is None`일 때 `SEED_REPEAT_REFUTED`(rel_diff `NaN`)를 내는 것을 고쳐 `UNRESOLVED (not evaluated: no bracket)`로 하라 — **평가되지 않은 예보가 반증으로 기록된다**(교훈 21·게이트 #21 재발, 실행으로 재현됨).
- **D22** 변이 하네스를 **`lambda0_analyze.py`까지** 확장하라(최소: 실현 분모의 `N−1`, `EBAR_GUARD` 밴드, 그리고 **analyzer 실출력 dict를 `rule()`에 그대로 투입하는 end-to-end 배선 시험** — 게이트 #181). 감사자 X1·X2도 추가하라(사각지대 경계가 shape A에서 load-bearing이다).
- **D23** `lambda0.sbatch`의 `FAIL_STREAK≥2 → break`(및 벽시간 절단) 시 **시도되지 않은 shape도 EXPECT에 들어가게** 하라(셀 목록은 plan.json에 이미 있다) — 그렇지 않으면 shape가 라벨 JSON에서 통째로 사라져 §5.1 출력공간 밖으로 나간다.

**⚠️ 자기 처방 실현가능성 검사(게이트 #113)**
(a) D17의 긴 창은 **고측 rung의 TTFT p99를 키우지 않는다**(고측은 손대지 않는다). 저측 rung은 비포화이므로 TTFT는 바닥 근처에 머문다 — D13의 "무검증 영역"을 넓히지 않는다. (b) D17은 `N_MAX` 클램프를 rung별 상수로 만들므로 `lambda0_plan.py`의 순함수성·인쇄 가능성이 유지된다(전 시나리오 재인쇄 필요). (c) D16의 κ 예측은 **GPU 0**으로 가능하다(등록 step 모형 + Little 법칙). (d) **처방의 한계**: κ 예측 자신이 step 모형에 의존하므로, 모형이 틀리면 D16은 틀린 안전마진을 준다 — 그래서 D16은 "κ를 예측하라"가 아니라 **"κ를 셀마다 기록하고(=`duration − span`, 측정값) 저측 문턱을 κ 대비로 보고하라"**를 함께 요구한다. 기록은 GPU 0이며 사후 재량이 아니다(셀별 상수, 사전 등록된 산식).

---

## 6. 반증 실패 항목 (공정 기록 — 위 D조건 전에는 `CONFIRMED` 아님)

- **R1/R5 실질 승계**: 보관 cell JSON 9개 직접 투입으로 `C_LABEL.json`을 **비트 단위 재현**(0.9331188685063582 / 0.675302644015995 / 0.18668462822130108). 항등식 아님, 날조 아님 ✓
- **§3.4 전 시나리오 인쇄**: 생성기 출력과 **173/173행 바이트 동일** ✓ (전사 드리프트 0)
- **변이 하네스 CONTROL의 비공허성**: 주입으로 확인(아카이브 파손 → CONTROL FAILS·RC 2) ✓ — 보고된 자기 발견(temp dir 조용한 skip)은 **실제로 닫혔다**
- **인용 규율(게이트 #110)**: §10.1이 캠페인 측 인용을 HEAD `bddff6a`에 고정했고, 7개 인용을 HEAD와 작업트리 양쪽에서 재확인 — **전부 정확**. rev2가 판정서의 off-by-one 2건(`:18→:17`, `:78-79→:76`)을 **스스로 잡아낸 것도 독립 확인**했다(직전 판정서를 등록 상수로 승격하지 않은 모범 사례)
- **`CONSENSUS §3 항목120` 회피**: 성립(판정량이 telemetry를 경유하지 않음) ✓
- **D4 음성대조**: rev1 glob 재현이 실제로 틀린 답을 낸다 ✓
- **ONE BOOT PER CELL 기제 승계**: `c_capacity.sbatch:111`·`:123-178`을 직접 열어 대조, sbatch가 기제를 재현 ✓
- **arm 비교 금지·성능 판정 0건**: 유지 ✓ (이 단계는 어떤 결과가 나와도 HE0·정책 순위·Claim D/E 등급·stake #1을 움직이지 않는다)

---

## 7. 인용 금지(이번 판정서가 추가하는 것 — 결과 문서·정본이 그대로 승계할 문안)

> **L1** — **`achieved/realized_offered`는 포화도가 아니다. 그것은 `(N/(N−1))·span/(span+L)`이며, 비포화 셀에서의 값은 1이 아니라 배수(drain) 시간이 정한다.** 보관 12셀에서 `duration = max_i(arrival_i + e2e_i)`가 ≤0.27 s로 재현되므로 이는 항등식이다. **이 추정량으로 얻은 "비포화" 라벨을 out ≳ 256 토큰 shape에 인용하는 것을 금지한다** — (256,512)에서 저측 문턱 0.95는 "decode ITL < 20 ms"와 같은 뜻이고, 그것은 용량이 아니라 SM 분할의 성질이다.

> **L2** — **rev2가 보인 "청정 비포화 셀 3/3이 1.00±0.02로 수렴"은 `(8192 in, 96 out)` 한정 사실이다.** 같은 보정을 (256,512)에 적용하면 기준선이 0.93–0.97로 내려간다. **"실현-분모 보정이 인공물을 제거했다"를 shape를 명시하지 않고 인용하는 것을 금지한다.**

> **L3** — **`d44_r1_o96`이 "포화 셀"인 것은 배수 인공물이다**(측정 0.8988 vs 이상값 0.9295). rev2 §5.2(i)의 "실현 분모에서 d44가 실제로 포화 셀 2개"를 **M1 차단의 데이터 근거로 인용하는 것을 금지한다**(합성 쌍둥이 케이스 3b만 인용 가능).

---

**판정 요약**: `NO-GO` — **N2**(반전 2행: ①판정량의 미등록 영점이 shape A의 `KNEE_BRACKETED↔KNEE_NOT_BRACKETED`를 뒤집고, 등록 자신의 step 모형 아래에서는 F3의 참 분지가 도달 불가 [r(a_r0) 0.9685 @ITL 13.06 ms ↔ 0.9382 @ITL 24.06 ms, 임계 19.69 ms] ②fallback 분기 재량이 shape B의 라벨을 뒤집는다 [창 [0.5157,1.835] ↔ [0.4302,1.026], λ*(B)=1.10]). **D1–D15 중 13건은 코드·데이터로 닫힌 것을 확인**했고 하네스·sbatch·provenance 게이트·인용 규율은 건강하므로, **D16–D23 적용 후 재감사에서 `GO-with-caveats`가 가능하다고 본다.** 최소 추가 비용은 **+0.24 GPU-h**(shape A 하위 2 rung의 측정창 연장)이며, 그 한 가지 변경이 死因 ①을 구조적으로 제거한다.

---

### 관련 파일 (전부 절대경로)

- 감사 대상: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_eval/lambda0_prereg/PREREG_LAMBDA0_REV2_2026-09-13.md` · 같은 디렉터리의 `lambda0_plan.py`(`:63-104`, `:310-387`) · `lambda0_analyze.py`(`:67-74`, `:89`, `:114-121`) · `lambda0_label.py`(`:68-85`, `:107-173`, `:192-200`) · `lambda0_mutation_check.py`(`:77-98`) · `lambda0.sbatch`(`:92-101`, `:104`, `:150-152`, `:242-245`)
- 직전 판정서: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_eval/lambda0_prereg/VERDICT_lambda0_rules_2026-09-13.md`
- 반전 계산 원자료: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/c_905835/` (`bench_*.jsonl` 12, `cell_*.json` 12, `C_LABEL.json`) · `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/c_capacity.sbatch` (`:76`, `:78-79`, `:111`, `:123-128`, `:139-145`)
- 엔진 사실: `/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/bench_serving.py` (`:70`, `:925-950`, `:1705-1706`) · `.../sglang/benchmark/datasets/common.py:55-64` · `.../sglang/benchmark/datasets/random.py:57-76` · `.../sglang/srt/managers/scheduler.py:2010-2016` · `.../sglang/srt/server_args.py:1959`
- 분할 지속 근거: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/src/multiplex/multiplexing_mixin.py` (`:263`, `:904-963`, `:1009`, `:1066-1124`)
- provenance: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh` (`:8`, `:127-185`; HEAD 24항목 / 작업트리 25항목 **미커밋**)
- 인접 사전등록(λ_inf 공급원): `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/newpair_prereg/PREREG_NEWPAIR_2026-09-13.md` §I3 — ★그 문서의 **(D7)** 조항("I3를 사다리 **하한**에 쓰면 브래킷을 놓친다")과 rev2의 `MULT[A][0]=0.40·λ_inf`(하한을 λ_inf에 거는 설계)가 같은 날 서로 반대 방향을 등록하고 있다. rev3는 둘 중 하나를 정정해야 한다.
