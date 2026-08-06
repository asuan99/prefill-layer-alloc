# 실 trace를 **long-context로 전환**하는 문제 — 논의와 계획

발의: 사용자, 2026-07-17. 작성: 같은 날. ★**개정: 2026-07-25**(§0.5 신설 + §4.3 SLO 교정 + §6 L−2/L3s + §2 H_L5 — 벡터1 short-ctx 종결·r0c decode-floor grounding·시간/공간 분리·운영점 게이트 반영). ★★**개정: 2026-07-26**(§0.6 신설 — **L−2(Stage 0) 게이트 실행 완료·결과 기록**: non-binding, ctx≤16k. L−1 이상은 여전히 계획 미실행). ★★★**개정: 2026-07-28**(§0.6/H_L4/H_L5/L−2 행 **철회** — claims-auditor 감사(C1 CONFIRMED)로 Stage 0의 D108 무경합 앵커가 실은 decode 16 SM이었음이 확인되어, L−2가 "실행 완료·non-binding"이 아니라 **"게이트 미실행"**으로 정정됨. L−1 이상은 "게이트 실패로 보류"가 아니라 다시 계획 단계).
계기: *"motivation으로 attn과 mamba가 차이가 난다는 지점은 **long-sequence에서 심화**되는데, 실 trace에도 그런 데이터셋들이 존재할 것이고, 그에 따라 **실행의 범위가 달라질 수 있는** 가능성이 있다."*

관련: [CONSENSUS.md](CONSENSUS.md)(정본) · [research_arc.md §S-M](research_arc.md)(반증 지점·측정 환경·유효 경계) · [realtrace_findings_and_open_branches.md](realtrace_findings_and_open_branches.md)(얽힘 기전) · [stage0_verdict_2026-07-26.md](stage0_verdict_2026-07-26.md)(L−2 원 판정, ★★★2026-07-28 claims-auditor 감사로 철회 — 아래 §0.6 참조).
**이 문서는 계획이다.** §0.6·§6의 L−2 행은 2026-07-26엔 "예외 — 실행·확정됐다"고 기록했으나, ★★★**2026-07-28 claims-auditor 감사(C1 CONFIRMED)로 그 근거(D108 무경합 앵커)가 무효 확인**되어 L−2도 **게이트 미실행**으로 되돌아갔다 — 이 문서 전체가 다시 계획 단계다. L−1 이상 단계의 수치는 여전히 측정된 것이 아니다 — 확정 결론은 CONSENSUS/PROJECT_STATUS에만 쓴다.

---

## 0.6 ★갱신 (2026-07-26) — L−2(Stage 0) 게이트 실행: non-binding — ★★★2026-07-28 철회(C1 CONFIRMED)

`stage0_verdict_2026-07-26.md`(3-arm coupled-운영점 스윕, jobs 864230+864601,
`stage0_transformer_control_design_2026-07-25.md`의 사전 등록 설계)가 §6의 L−2 게이트를
닫았다고 기록했다. **결과 = 게이트 실패(§6 표의 "게이트 실패이자 강한 결과" 분기 실현)**:
운영점(cudagraph-ON)에서 decode ITL은 decode-SM(16→108)에 **무감각**이다 —
pure-Transformer(T)·pure-Mamba(M, 음성 대조)·hybrid(H) 전부, ctx {4k,8k,16k} 전부에서
(de-confounded 대조 D16 vs D108 = 1.00±0.01). raw coupled 곡선 자체는
confounded(prefill 경합/entanglement 아티팩트)였으나, 음성 대조 M + 무경합 앵커 D108의
삼각검증으로 이 confound를 우회해 clean null을 얻었다 — **고 봤다.**

★★★**철회(2026-07-28, claims-auditor 사전등록 게이트 집행, C1 CONFIRMED)**: 위
"무경합 앵커 D108"은 **실제로는 decode 16 SM이었다**(3중 독립 증거 — 코드 기전
`src/multiplex/multiplexing_mixin.py:725-742`의 threshold=0 오독, realized
telemetry 79–96% (92,16) 동거, `D108/D16` 클라이언트 서명 0.992–1.001인데 D92가
3.4–3.6× 빠름). "D16 vs D108 = 1.00±0.01"은 **동일 조건의 반복측정**이었다. "음성
대조 M + 무경합 앵커의 삼각검증"이라는 위 서술도 **거짓** — 무경합 앵커는 고장,
음성 대조 M의 전제("decode O(1) recurrent라 SM-bound 불가")도 틀렸다(그 O(1)은
context 길이에 대한 것이지 SM 수에 대한 것이 아니었다 — `../PROJECT_STATUS.md`
"8B decode-SM 민감도" C2 참조). raw coupled 곡선이 CONFOUNDED라는 진단(위 문단의
"raw coupled 곡선 자체는 confounded")만 생존한다.

⇒ §0.5(D)가 예고한 "(B)~(C)의 6.5× 微측정 민감도가 운영점서 사라진다 → long-ctx
충돌 가설(H_L4 시간축·H_L5 공간축, 아래 §2) 이 regime서 붕괴, 벡터1/HE0가
ctx-무관으로 강화된다"는 판정도 **철회**한다 — 근거(D108 앵커)가 무효이므로.
**§6 게이트 규칙이 예정한 "L−1 이상은 여기서 멈춘다"는 "게이트 실패로 멈춘 것"이
아니라 "게이트가 아무것도 측정하지 못해 미실행 상태로 남은 것"**이다(재개 권고
아님, 판정 부재라는 뜻).

★**scope 한정(필수, 원 판정 참고용)**: 위 판정은 {M/H/T 2.7–3B, triton attn+mamba,
cudagraph-ON green-context pdmux, ctx≤16k, coupled 하네스, one-shot 32-conc
burst}에 한정하려 했다. 상세는
[stage0_verdict_2026-07-26.md](stage0_verdict_2026-07-26.md)(원 판정, 위 항목들로
철회됨) · `../workspace/engine-port/results/s0_deconfound/
PARTITION_RESIDENCY_STAGE0.md`(C1 근거) · `../PROJECT_STATUS.md` "Stage 0" 절.

---

## 0. 한 줄

**동기는 정당하다**: 이 아크의 창립 동기(attn↔mamba 격차)는 **L 의존**인데, **서빙 반증은 전부 L<2k에서 났다**(ShareGPT mean 352·98%가 L<2000).
**그러나 이것은 layer-aware의 부활 경로가 아니다** — 재배분 lever(**Diff B**)는 **long-L에서 오히려 ≈1.0으로 닫힌다**(실측).
long-context가 실제로 흔들 수 있는 것은 **(1) 최적 static split의 위치, (2) 얽힘 기전의 병목(running-batch → KV), (3) HE0의 구조적 근거("두 regime의 최적이 충돌하지 않는다")** 이며,
**(3)이 동적 제어에 남은 유일한 upside**다.

---

## 0.5 ★갱신 (2026-07-25) — 벡터1 종결이 long-ctx 동기를 *날카롭게* 만든다

이 계획(2026-07-17) 이후 두 가지가 확정됐다. 이 절이 §2·§4·§6를 **대체하지 않고 정밀화**한다.

**(A) 단-ctx 충돌-regime(H_L4의 short-ctx 판)은 이제 CONFIRMED 종결.** G2.0 캠페인(pre→full→hard→de-cliff→rasweep 120job→raconf 24job, 감사 3회)이 Zamba2-2.7B·ctx4096·drained 2-phase에서 **견고한 off-cliff disjoint 부재**를 확정([CONSENSUS](CONSENSUS.md) §5-8(c)). 기전이 결정적이다: **short-ctx는 decode floor가 낮아(feasible-B = d54, 54 SM으로 충분) prefill 요구와 안 부딪힌다** → 단일 static(d54)이 양 phase 커버 → 충돌 없음. ⇒ **long-ctx의 진짜 질문 = "decode floor를 충돌이 생길 때까지 밀어올릴 수 있는가"**. 임의의 새 방향이 아니라 벡터1이 닫힌 *근본 원인의 정조준*이다.

**(B) decode floor가 ctx로 상승한다는 것은 이미 微측정으로 grounded**([results/r0c/](../workspace/engine-port/results/r0c/), job 858811, ctx16384 decode step 분해):

| decode SM | attn 9층 합 | mamba 54층 합 | attn 비중 |
|---|---|---|---|
| full(108) | 60.8ms | 16.0ms | **79%** |
| 44 | 148.6ms | 11.3ms | — |
| 16 | 394.6ms | 20.1ms | — |

- **mamba 54층 = O(1)**(ctx·SM 무관 ~0.2–0.3ms/층) — hybrid 구조 특성.
- **attn 9층이 long-ctx decode를 지배**(16k서 79%)·SM 급민감(108→16 SM에서 **6.5× 팽창**). short-ctx(ctx256 attn 5%)엔 decode가 SM-무관(=벡터1이 닫힌 이유)인데 **long-ctx에선 decode가 near-108 SM 요구**. 동시에 prefill은 attn O(L²)로 더 SM-hungry ⇒ **두 phase 동시 high-SM 요구 = short-ctx엔 없던 충돌.**

**(C) ★시간축 vs 공간축을 분리하라** — 초안 §6이 H_L4(temporal)만 봤으나 long-ctx는 두 갈래를 낳는다:

| 축 | 조건 | 이기는 것 | long-ctx가 하는 일 |
|---|---|---|---|
| **시간(temporal)** = 초안 H_L4 | prefill/decode 시간적 분리(혼합 trace) | SM을 시간에 이동 → **단일-GPU 동적** | decode floor↑로 per-phase 최적이 *발산* → §1-13 "충돌 없음" 반증 후보 |
| **공간(spatial)** = §1-20 증폭 | 동시 진행·둘 다 binding | 90+90>108 = **어떤 단일-GPU 할당도 불가** → **2-device decoupling만** | short-ctx §1-20은 116>108(≈8 초과)이었으나 **long-ctx는 ~180>108(≈72 초과)** → **decoupling headroom 대폭 확대** |

★**hybrid decoupling 훅**: disaggregation state transfer = **mamba state O(1)(저렴) + attn KV O(L)(증가)**. long-ctx서 KV가 전이비용 지배(mamba 이점 희석)하나 mamba 54층분은 공짜 → **hybrid disaggregation이 pure-attention보다 mamba 비율만큼 싸다**(논문 훅). **§1-20 방화벽 유지**: 공간축은 시간축(벡터1) 종결과 별개로 열려있다.

**(D) ★NEW 게이트 = Stage 0 (운영점 전이) — L−1보다 먼저.** (B)의 6.5×는 **triton/no-cudagraph 微측정**값이고, HE2는 **cudagraph 운영점서 decode non-binding**을 관측했다. 이 아크를 두 번 뒤집은 게 "micro ≠ serving"이다. ⇒ **long-ctx decode가 cudagraph 운영점에서 실제로 SM-binding하는가**가 make-or-break. 이게 안 되면 (B)~(C) 전체가 short-ctx처럼 붕괴하고 벡터1/HE0가 ctx-무관으로 강화된다. **가장 싸게 먼저 친다**(§6 L−2 신설).

**(E) 모델/rope 주의**: r0c는 Zamba2-2.7B를 ctx16384로 돌렸으나 이는 **decode-step 타이밍 微벤치(rope 외삽, 출력 품질 무관)**다. Stage 0(타이밍/SM-민감도)엔 Zamba2 재사용 가능(compute는 실측)하나, **L0+ 서빙 feasibility(품질·goodput)엔 진짜 long-ctx 학습 모델 필요**(§4.1 `granite-4.0-h-micro` 131072). config `max_position_embeddings=4096`(§4.1)과 r0c ctx16384 실행의 불일치는 이 rope-외삽으로 설명 — **품질 실험엔 외삽 금지.**

---

## 1. 왜 이 질문이 정당한가 (동기의 정확한 형태)

| | |
|---|---|
| **창립 동기** | hybrid의 attn 층과 mamba 층은 연산 특성이 다르다 ⇒ 다르게 다뤄야 한다 |
| **그 차이가 켜지는 곳** | **Diff A(비용비) = attn/mamba**: `attn ~ L^1.68` vs `mamba ~ L^0.61`, **교차점 ≈3k tok**. L=2k서 **0.47×**(mamba가 오히려 비쌈) → L=8k **2.4×** → L=32k **10.1×** (`results/prefill_knee/diffA_vs_diffB_table.md`, **no-cudagraph micro**). ★**지수 정정(2026-08-04, X2′, `AGGREGATE_COMPOSITION_2026-08-04.md`)**: steady OLS(ctx≥2050) **a=1.916±0.021 / b=0.954±0.008 (R²≥0.9996)** — 1.68/0.61은 **오염 끝점 2점 추정**. "L=2k서 mamba가 오히려 비쌈"은 **단위 의존이라 판정 불가**(집계 단위=커널/per-layer/정책 단위마다 역전 지점이 2.75k–17.9k로 달라짐), 인용 시 집계 단위를 명시할 것 |
| **그런데 서빙 판정이 난 곳** | ShareGPT **mean 352 · p50 204 · p95 1042** (98%가 L<2000) / synthetic **in2000·in3600** ⇒ **전부 교차점 아래 = Diff A가 닫혀 있거나 역전된 구간** |

⇒ ★**"attn과 mamba가 다르다"는 전제가 실제로 성립하는 구간을 이 아크는 *서빙으로 한 번도 밟지 않았다*.**
S0/S2/S9/S10의 서빙 반증은 각자의 환경에서 유효하지만(→ [research_arc.md §S-M.1](research_arc.md)), 그 환경은 **short-context**다.

---

## 2. ★먼저 적는 정직한 반론 (기대치를 미리 못박는다)

이 아크는 **결과를 본 뒤 서사를 맞추다가 4번 뒤집혔다.** 그래서 **측정 전에 예측을 등록**한다.

**(a) lever는 Diff A가 아니라 Diff B다 — 그리고 Diff B는 long-L에서 닫힌다.**
실측 Diff B(=SM 민감도비): **L≥8000서 0.96–1.04**, L=32000서도 **1.01–1.04**. 반면 **L=2000서 ≈1.35**(★**REFUTED, 2026-08-04, X2′**: steady **1.0081** [1.0078, 1.0084] — 보고 1.35는 계측 결함(버킷 비대칭+누산기 러닝평균) 아티팩트, `CONSENSUS.md` §1-3).
⇒ **재배분할 여지는 L이 *짧을수록* 열린다.** long-context는 layer-aware 가설에 **불리한 방향**이다. **long-context 실험은 layer-aware의 재판이 아니다.** ★결론(long-context가 layer-aware에 불리)은 **방향 불변**(L≥8000의 ≈1.0도 steady 재계산 시 1.01–1.04 근방으로 유지) — 이 문단이 정정하는 건 L=2000 지점의 magnitude뿐이다.

**(b) (D) granularity 비용은 L과 무관하게 남는다.** S2에서 **TPOT 42→124ms**로 정량화된 sub-step 재분할 비용은 구조적이며, L이 길어져도 사라지지 않는다.

**(c) ★long-context 데이터셋은 대개 *출력이 짧다*.** LongBench-v2 로더 기본 `output_len = 10`(`sglang/benchmark/datasets/longbench_v2.py`).
우리 기전상 **최적 D_sm = max(모델 floor, 부하항[∝ λ×output_len])** 이므로, **출력이 짧으면 부하항이 죽고 최적은 prefill-heavy 극단에 고정**된다.
⇒ **long-doc QA 단독 trace는 HE0를 뒤집기는커녕 *강화*할 공산이 크다**(움직일 최적이 없음 = LO phase의 극단판).

### 사전 등록 예측 (측정 전)

| ID | 예측 | 근거 | 반증되면 |
|---|---|---|---|
| **H_L1** | long-context서 **최적 static이 prefill-heavy(d16 쪽)로 이동** | 부하항 ↓(출력 짧음) → 모델 floor 지배 | "최적 = 부하 함수" 모델(CONSENSUS §1-5)이 틀렸거나 KV 병목이 새 항을 추가 |
| **H_L2** | **PD-mux(agnostic) 이득 자체는 유지 또는 확대** | prefill이 길어져 overlap 기회↑ | PD 분리(§1-1)마저 L 의존 |
| **H_L3** | **layer-type 정책은 여전히 死**(더 확실히) | Diff B가 long-L서 ≈1.0 | (거의 불가) Diff B≈1.0인데 이득이 나면 = 내 lever 모델 자체가 틀림 |
| **H_L4** (시간축) | ★**혼합(long-doc ↔ chat) trace에서만 두 regime의 최적이 *충돌*** | HE0의 구조적 근거가 "충돌 없음"이므로 | 충돌시켜도 동적이 지면 ⇒ **시간축 동적 트랙 완전 종결**. ★**반증(전제, 2026-07-26, Stage 0 L−2, jobs 864230+864601) — ★★★2026-07-28 이 반증도 철회(C1 CONFIRMED)**: 이 가설이 기대는 ctx-의존 decode floor 상승이 "ctx≤16k 운영점서 관측되지 않음(D16≡D108)"이라고 봤으나, 근거인 D108 앵커가 실은 decode 16 SM이었음이 확인돼 무효. H_L4는 **다시 미검증**(전제도 반증도 안 됨) |
| ★**H_L5** (공간축, 2026-07-25 신설) | **long-ctx서 decode floor↑ → feasible-A ∩ feasible-B = ∅ (공간 충돌 90+90>108) → decoupled-oracle headroom이 short-ctx +16%를 크게 상회** | §0.5-B/C: decode floor가 ctx로 near-108까지 상승(r0c 微측정) + §1-20 증폭 | ★**전제 Stage 0(L−2)**: 운영점서 decode가 non-binding이면 floor 안 오름 → H_L5 붕괴. 또는 binding이어도 headroom이 안 열리면 §1-20이 long-ctx서도 닫힘. ★★**반증 실현(2026-07-26, Stage 0 L−2, jobs 864230+864601, [stage0_verdict_2026-07-26.md](stage0_verdict_2026-07-26.md)) — ★★★2026-07-28 철회(C1 CONFIRMED)**: "운영점서 decode가 ctx≤16k까지 non-binding(D16≡D108)"이라는 관측이 D108 앵커 무효로 근거를 잃어, "정확히 이 전제 분기가 발생했다"는 판정을 철회한다. H_L5는 **다시 미검증**(붕괴도 실증도 안 됨) |

★**진짜 목표는 이제 둘**: **H_L4(시간축 동적)** 과 **★H_L5(공간축 decoupling)**. 벡터1 종결(2026-07-25)로 short-ctx 시간축이 닫혔으니 무게중심은 **H_L5**로 이동. 나머지는 그 전제(특히 **L−2 운영점 binding**)를 까는 작업이다.

---

## 3. 무엇이 걸려 있나 (payoff)

| 질문 | long-context가 바꾸나 | 왜 |
|---|---|---|
| layer-type 정책 부활 | ❌ 거의 없음 | Diff B가 닫히는 방향(§2-a) |
| **최적 static split의 위치** | ✅ 예 | 부하항이 바뀜 ⇒ 실전 권고("peak decode 기준 decode-heavy")의 **범위 한정**이 필요해질 수 있음 |
| ★**얽힘 기전의 병목 재편** | ✅ **가능성 높음** | 현 기전은 `max_running_requests`(48) 포화 → admission 차단. **long-context서는 KV pool이 먼저 cap을 때린다** ⇒ admission이 **KV-bound**로 바뀌고 "decode 굶김 → TTFT 폭발" 경로의 상수가 달라짐. **기전 일반성의 시험** |
| ★**HE0 반전(동적의 upside)** | ⚠️ **조건부** | **혼합 trace에서만**. 단일 long-doc trace로는 오히려 강화(§2-c) |
| PD-mux 이득 크기 | ✅ 아마 확대 | prefill이 길수록 분할의 여지 |

---

## 4. 하드 블로커 (덮고 갈 수 없는 것)

### 4.1 ★정본 벤치 모델이 long-context를 못 한다

**정책 캠페인(E3)은 전부 `Zyphra/Zamba2-2.7B` 단독**인데 **`max_position_embeddings = 4096`**(로컬 config 확인). Zamba2-7B도 **4096**.
⇒ **모델 교체가 필수**이고, 이는 **정본 벤치의 모델 변경** = 기존 수치(d44 3.220 등)와 **직접 비교 불가** ⇒ **전 baseline 재측정** 필요.

| 후보(로컬 캐시 보유) | max ctx | 층 | 적합성 |
|---|---|---|---|
| ★**`ibm-granite/granite-4.0-h-micro-base`** | **131072** | 40 | **권장**. 진짜 hybrid(attn/mamba 층 분리) + long ctx. E1서 서빙 이력 있음 |
| `tiiuae/Falcon-H1-3B-Base` | **131072** | 32 | long ctx는 되나 **층마다 attn+mamba 병렬 = 단일 타입** ⇒ layer-type 논의엔 무의미(≡agnostic). **PD-mux 전용 대조군**으로는 유효 |
| `tiiuae/Falcon-H1-7B-Instruct` | 262144 | 44 | 위와 동일 성격, 더 큼 |
| `nvidia/Nemotron-H-8B-Base-8K` | **8192** | 52 | **8k까지만** — 중간 지점(교차점 3k는 넘음) 실험엔 사용 가능 |
| `Zyphra/Zamba2-2.7B` (현 정본) | **4096** | 54 | ❌ long-context 불가 |

### 4.2 메모리 · 얽힘 상수가 **동시에** 변한다 (confound 경보)

A100 80GB · `mem-fraction-static 0.82`에서 **ctx 32k × running 48**은 KV가 안 들어간다.
⇒ `--max-running-requests`를 낮추거나 ctx를 낮춰야 하는데, **`max_running_requests`는 얽힘 기전의 축**이다(CONSENSUS §1-4).
★**"변수는 하나씩"(§3-8) 규율상, running 값을 바꾼 채 정책을 비교하면 해석 불가.** ⇒ **running을 명시적 sweep 축으로 승격**하고, 정책 비교는 running 고정 하에서만.

### 4.3 ★goodput 정의가 long-context서 무너진다

현 SLO는 **TTFT ≤ 3s ∧ TPOT ≤ 60ms**. 그런데 **32k prefill 단독으로 3s를 넘길 수 있다**(E4 micro: L=32k attn 108ms/층 × 9 + mamba 15ms × 54 ≈ 1.8s **@full 108 SM**, 분할하면 그 이상).
⇒ **모든 정책이 goodput 0** = 신호 소멸. 게다가 우리는 **"임계 지시함수는 절벽을 피해 측정하라"**(§3-6)는 교훈을 이미 비싸게 샀다.
**해결 없이는 실험 자체가 무의미하므로 L0 전에 정한다.**

★**교정(2026-07-25, venue-strategist prior-work 조사)**: 초안의 **(i) `TTFT_SLO(L)=a+b·L` 연속 길이-정규화 임계는 학회 통용 방법론이 아니다 = ad-hoc**. 표준은 아래 두 가지이며, 우리 계획을 그것으로 대체한다.

| 방법 | 내용 | prior-work 근거 |
|---|---|---|
| **★Primary = 절대 class SLO + SLO-scale sweep** | application-class별 절대 TTFT/TPOT(chat 300 / voice 150 / code 100 / RAG 400ms / batch 3s; [serving_slo_survey.md](serving_slo_survey.md) §2) 를 **단일점이 아니라 배수로 sweep**(DistServe "SLO scale": rate 고정·절대 SLO를 곱셈 스윕, 작을수록 tight) | DistServe OSDI'24 (Table 1: summarization TTFT **15s** = 그들의 long-context 답; SLO는 class별 절대·경험적). §1-14 단일-3s 실수 회피 |
| **★Secondary 진단 = prefill-normalized slowdown** | `slowdown(stretch) = 관측_TTFT / prefill_floor(L)` (그리고 excess = 관측 − floor). "정책의 손상 = 환원불가 prefill floor 위의 excess"를 분리 → physics(길이) vs policy(split) 분리 | queueing 계보 **slowdown/stretch**(Bansal & Harchol-Balter SIGMETRICS'01) + 표준 `TTFT = W_queue + T_prefill` 분해. ⚠️**"normalized latency"라 부르지 말 것**(Orca/vLLM선 지연/*output*길이=decode 축, 우리와 혼동) |
| (폐기) (i) `a+b·L` | 연속 길이-정규화 임계 | ad-hoc·계보 없음. **길이-불변 slowdown 임계(관측/floor ≤ k)** 또는 **per-length-bucket SLO-scale**로 대체 |
| (참고) (iii) 분포 지표 | TTFT p95 / 용량(req/s) 직접 비교 | 절벽 회피엔 유효하나 goodput 계보와 단절 — slowdown이 상위호환 |

★**필수 규율**: **prefill_floor는 반드시 *측정*(모델링 금지)** — 단일요청·목표 SM·무경합으로 실측(gate #1이 floor에도 적용; 모델링 floor면 아티팩트). Primary(절대)로 **제품 goodput**을 주장하고, Secondary(slowdown)로 **split 인과 효과**를 분리 — 둘 다 병기(mean-ITL secondary·per-req p95 primary 패턴과 동일).
※ **long-ctx 인터랙티브 TTFT는 본질적으로 완화/배치성 regime**일 수 있음(사용자 인내는 절대적, 16k prefill 2s는 아무리 scale해도 2s) — slowdown으로 그 사실을 *숨기지 말고* 절대 SLO로 함께 드러낸다.

---

## 5. 데이터셋 옵션 (엔진 내장 확인 완료)

`sglang/benchmark/datasets/__init__.py`의 `DATASET_MAPPING`: `sharegpt` · `random` · `random-ids` · `generated-shared-prefix` · `longbench_v2` · `mooncake` · `mmmu` · `image` · `custom` · `openai` · `autobench` · `speed-bench`.

| 옵션 | 실 trace? | 장점 | 단점 / 주의 |
|---|---|---|---|
| **`random` (in 8k/16k/32k)** | ❌ | **즉시 가능**·완전 통제·E1 계보와 연속 | 실 trace 아님. **L1의 도구**(최적 이동 확인용) |
| ★**`longbench_v2`** (THUDM/LongBench-v2) | ✅ | 진짜 long-doc, **로더 내장**, 길이 분포 넓음 | **출력 기본 10 tok**(`--sharegpt-output-len`으로 강제 가능하나 그러면 "실 trace"가 반쪽) · **데이터 미보유**(`hf_cache/raw/longbench_cache/`는 **비어 있음**, 8K) · **`HF_HUB_OFFLINE=1`이라 사전 다운로드 필수** |
| **`mooncake`** | ✅✅ | 실 서빙 **arrival + 길이 trace** = 가장 현실적 | prefix 재사용 전제 — 우리는 **`--disable-radix-cache`** 중이라 해석 주의(또는 이 실험만 radix 허용 = 또 다른 변수) |
| **`generated-shared-prefix`** | ❌ | long prefix 통제 | 위와 동일한 radix 문제 |
| ★**혼합 (ShareGPT ↔ long-doc 교대)** | ✅ | ★**H_L4 전용** — regime 충돌을 *만드는* 유일한 벤치 | 하네스 신규 필요(E3-vary의 rate 교대를 **dataset 교대**로 확장) |

**권고**: `random`(통제) → `longbench_v2`(실 trace) → **혼합**(판정). `mooncake`는 radix 결정을 별도로 내린 뒤.

---

## 6. 단계 계획 (게이트 포함)

> 각 단계는 **다음 단계의 전제를 검증**한다. 게이트를 통과 못 하면 **거기서 멈춘다** — 이 아크가 비싸게 배운 것은 *전제를 안 밟고 나아간 대가*다.

| 단계 | 내용 | 게이트 (통과 못 하면 중단) |
|---|---|---|
| ★**L−2 (Stage 0)** — ★★★**게이트 미실행(2026-07-28 정정)** — 원 기록 "완료(2026-07-26), 게이트 실패(non-binding)"는 철회 | **운영점 decode-SM 민감도 vs ctx**: cudagraph-ON, **decode-only** 부하, ctx ∈ {4k, 8k, 16k} × decode-SM ∈ {16, 44, 92, 108-ref}, **ITL 측정**, 3-arm(M 음성대조/H 타깃/T 양성대조), jobs 864230+864601 | ★**make-or-break**: long-ctx decode가 운영점서 **SM-binding하나**(ITL이 decode-SM에 단조 급민감)? **평탄 → 微측정 6.5×가 운영점서 사라짐 → (B)~(C) 붕괴 = long-ctx 충돌 가설 死, 벡터1/HE0가 ctx-무관 강화. 여기서 멈춘다.** binding 확인 → L−1. 원래 "실현된 결과 = 평탄(D16≡D108, 3 arm×3 ctx 전부)"이라 기록했으나 ★★★**철회(2026-07-28, claims-auditor 감사, C1 CONFIRMED)**: D108 앵커가 실은 decode 16 SM이었고("음성대조+무경합앵커 삼각검증"도 두 기구 모두 고장), raw 곡선의 CONFOUNDED 진단만 생존한다. **이 게이트는 SM-binding 여부를 측정한 적이 없다** — make-or-break 질문은 다시 열려 있다. 상세 [stage0_verdict_2026-07-26.md](stage0_verdict_2026-07-26.md)(원 판정, 철회됨) |
| **L−1** — ★**게이트 미충족, 보류(2026-07-26)** | **SLO 정의 확정**(§4.3: 절대 class SLO + slowdown 진단, **prefill_floor 실측**) + **용량 먼저 측정**(각 config의 최대 처리율) | 절벽 밖 rate 대역이 존재하는가. **L−2가 중단 분기로 실현돼 이 단계는 ctx≤16k regime에서 진행 근거 없음** — >16k·큰 모델로 L−2를 재시도하기 전까지 보류 |
| **L0** | ★**모델 교체 baseline**: `granite-4.0-h-micro`로 **E3-vary(ShareGPT rate 3↔12)를 그대로 복제**, static sweep d16–d54, **n≥4** | **HE0가 모델을 넘어 재현되나**(decode-heavy static이 최선?). 재현 안 되면 → 그 자체가 큰 발견이고, 이후 long-context 해석의 기준이 통째로 바뀜 |
| **L1** | **`random` long**: in ∈ {2k, 8k, 32k} × out 96, **static sweep** (동적 없음) | **H_L1 검증**: 최적 static이 prefill-heavy로 이동하나? **Diff A가 열리는데 최적이 안 움직이면** = 비용비는 split 결정에 무관하다는 강한 증거 |
| **L2** | **`longbench_v2` 실 trace**: static sweep + **PD-mux vs fused** | **H_L2**: PD 분리 이득이 long-context서도 사나 |
| **L3 (시간축)** | ★**혼합 trace**(long-doc phase ↔ chat phase 교대): **best-static vs 동적(bind+GATE)**, **n≥4** | ★**H_L4 = HE0 반전 시나리오**(temporal). 여기서도 static이 이기면 ⇒ **시간축 동적 트랙 최종 종결**(짧은·긴·혼합 전부 소진) |
| ★**L3s (공간축, NEW)** | **long-ctx feasibility 구조 + decoupled-oracle 분해**(§1-20 long-ctx판): ctx별 feasible-A ∩ feasible-B가 **disjoint로 벌어지나** (G2.0을 long-ctx로) + **decoupled oracle headroom**(prefill-best ⊗ decode-best) 측정, **drained 2-phase**(공간-우호). n≥4·slowdown 채점 | ★**H_L5(신설) = 공간 decoupling**: long-ctx서 feasible-A ∩ feasible-B = ∅ (short-ctx는 ≠∅로 종결) **이고** decoupled-oracle headroom이 short-ctx +16%를 크게 상회(→ 90+90>108 충돌 실재)하나? 그러면 **hybrid disaggregation 트랙 정당화**(state-transfer O(1) mamba 훅). 아니면 §1-20이 long-ctx서도 안 열림 |
| ~~L4~~ | ~~layer-aware 재판~~ | **하지 않는다** — §2-a(Diff B가 long-L서 닫힘)가 사전에 배제. **하려면 짧은 L(≤2k)에서** 해야 하며 그건 이 문서가 아니라 **S3 정정의 열린 질문** |

★**"이길 수도 있는" 실험은 이제 둘**: **L3(시간축, 동적)** 과 **L3s(공간축, decoupling/§1-20)**. 벡터1 종결로 short-ctx 시간축은 닫혔으므로, long-ctx의 무게중심은 **L3s(공간)** 로 기운다(사용자 관심사). L−2·L0–L2는 전부 *전제 검증*이며 결과가 어느 쪽이든 **CONSENSUS 권고 범위를 한정**한다. **전 단계는 L−2(운영점 binding) 게이트에 조건부** — 그게 죽으면 L−1 이하 전부 무의미.

★★**(2026-07-26) 실현, ★★★2026-07-28 철회(C1 CONFIRMED)**: 2026-07-26엔 L−2가
정확히 이 죽는 분기로 실현됐다(non-binding,
[stage0_verdict_2026-07-26.md](stage0_verdict_2026-07-26.md))고 기록했으나, 근거인
D108 앵커가 실은 decode 16 SM이었음이 claims-auditor 감사로 확인돼 이 판정을
철회한다. **L−1·L0·L1·L2·L3·L3s의 상태는 "게이트 설계대로 조기 정지(보류)"가
아니라 "게이트가 아무것도 측정하지 못한 게이트 미실행"**이다 — 판정 이전
상태이며, 이후 단계 재개를 권고하지도 금지하지도 않는다(단지 판정이 없다는
뜻). L−2 자체를 다시 통과시키는 것이 여전히 다음 단계다.

---

## 7. 측정 규율 (CONSENSUS §3 상속 + long-context 특유)

1. **n≥4 없이 정책 결론 금지.** long-context는 런당 시간이 길어 n을 깎고 싶어지는데, **그게 정확히 §2-4(underpowered)를 재발**시킨다. ⇒ n을 지키고 **prompts 수를 줄인다.**
2. **용량을 먼저 재고 절벽을 피한다**(§3-6). long-context는 rate 대역이 훨씬 좁다.
3. **`switch_count` + split 체류분포 + TTFT/ITL p50/95/99 병기.**
4. **라운드 duration은 합산**(`f921ae8`의 3× 버그 재발 방지).
5. **변수는 하나씩** — 특히 §4.2의 `max_running_requests`. **모델·ctx·running을 동시에 바꾸지 말 것**(L0가 존재하는 이유).
6. **길이 fingerprint를 매 런 기록**(워크로드 변동을 *주장*하지 말고 *검증*한다 — 이미 한 번 당했다).

---

## 8. 현재 상태 · 인접 작업

- **`knee2d_wide` (job 857371, array 0–7, 진행 중)**: L ∈ {256…32768} × B ∈ {1…16}, **L 256/512/1024만 완료**.
  ⚠️ 이건 **반대 방향(짧은 L) 조사**로, [research_arc.md §S3 정정](research_arc.md)의 열린 질문(*"L≈200–2000서 Diff B가 열리나"*)용이다. **이 문서의 계획과 혼동하지 말 것.**
  (그 sweep의 부산물로 드러난 것: 구 `knee2d` 격자는 **B축이 가짜**였다 — `chunked-prefill-size`가 토큰을 잘라 요청한 B=48이 실제로는 bs=1–10으로 관측됨.)
- **데이터**: `hf_cache/raw/longbench_cache/`는 **비어 있음**. `HF_HUB_OFFLINE=1`이므로 **L2 전에 다운로드 필요**(로그인 노드에서 별도 수행).
- **미결정**: radix-cache 정책(`mooncake`/shared-prefix를 쓸 것인가), SLO 정의(§4.3 (i) vs (iii)).
