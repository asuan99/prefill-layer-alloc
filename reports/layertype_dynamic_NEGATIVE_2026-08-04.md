# NEGATIVE 정본 — layer-type 동작 차이 · dynamic multiplexing

작성 2026-08-04.

> ⚠️ **이 문서의 지위**: 정본에 이미 있는 결과를 **negative 축으로만 모아 재배열한
> 종합 문서**다. **새 판정을 만들지 않는다.** 충돌 시 정본 우선.
>
> ⚠️ **감사 2건 진행 중(2026-08-04)** — §1.3의 강등 6건은 확정 전이다.
>
> 짝 문서: [`layertype_dynamic_POSITIVE_2026-08-04.md`](layertype_dynamic_POSITIVE_2026-08-04.md) ·
> [`layertype_dynamic_JUNCTION_2026-08-04.md`](layertype_dynamic_JUNCTION_2026-08-04.md)

---

## 0. 한 줄

**layer-type을 런타임 스케줄링 경계로 쓰는 정책은 全형태 死이고, 단일 GPU 위의
동적 SM 제어는 best-static을 못 넘는다 — SLO 엄격도와 무관하게.** 두 negative는
**서로 독립인 증거**로 서 있으며, 어느 하나가 흔들려도 다른 하나는 유지된다.

---

# 1. layer-type 축 — 全형태 死

## 1.1 서빙 등급 반증 (흔들리지 않는 것)

| # | 반증된 주장 | 결정적 수치 | 출처 |
|---|---|---|---|
| N1 | layer-aware(decode-side)가 agnostic을 이긴다 | **agnostic 4/4 승**. Zamba2는 rate 3부터 **goodput 0**(사전 예측은 정반대 "45/54 환원 대박") | `CONSENSUS.md` §1-3, E1 캠페인 |
| N2 | `agnostic_v2`가 (Granite서) 최적이다 | decode에 실작업 있으면 **최악**. 앞선 "최적" 판정은 tiny-batch micro 아티팩트 | 동일 |
| N3 | 조율만 하면 per-type layer-aware가 산다 | **TPOT 42 → 124ms**. `PDMUX_LA_COORD_OPT`로 절반(→85ms) 회수해도 **부호 robust** | `CONSENSUS.md` §1-3·§1-15, E2 |

★ **이 셋이 layer-type negative의 뼈대다.** 작동하는 구현으로 end-to-end goodput을
직접 비교한 결과이며, **Diff A/Diff B 기전 서사에 의존하지 않는다.** 기전 설명이
전부 틀려도 이 판정은 안 바뀐다.

## 1.2 기전 — (D) granularity

| | |
|---|---|
| **내용** | decode 한 스텝이 **19 윈도우로 파편화** → sync 직렬화 + partition 핀 + overlap 감소 |
| **핵심 정정** | ★ **(D)는 *실행시* 비용이지 *결정시* 비용이 아니다** — offline predictor로 고정해도 실행 시 attn↔mamba 경계마다 green-ctx 재분할이 필요하다. offline은 **결정 오버헤드만** 제거한다 |
| **"싼 전환"으로 우회 불가** | green-ctx는 **이미 pre-created**(스위치=인덱싱). 진짜 비용은 경계마다 `stream.synchronize()` **드레인**. GPU측 wait_stream으로 교체해도 **124→85ms(47%)만 회수**, 잔차는 구조적 오버랩 손실 + **cudagraph 비양립**(step 중간 전환 캡처 불가 → 운영점 진입 불가) |
| **출처** | `CONSENSUS.md` §1-15, `results/a_substrate/` |

## 1.3 구제 시도 kill-chain — 성립 조건 (C1)∧(C2)의 교집합이 공집합

[`per_layer_type_postmortem.md`](per_layer_type_postmortem.md) 프레임: 정책이 서려면
**(C1) lever 존재**(Diff B ≠ 1) **∧ (C2) 착취 가능**((D) 비용 < lever 가치).

| 시도 | 공략 | (C1) | (C2) | 결과 |
|---|---|---|---|---|
| **A** 창립형 decode-side | C1 | ✅ **참** (Diff B ≈ 4.0×) | ❌ 서빙 4/4 패 | 死 |
| **B** graduated per-type map | C2 | — | — | **재개** (미조율 아티팩트: coordinated 42ms vs 미조율 121ms) |
| **C** coordinated 실제 구현 | C2 정면 | — | ❌ **TPOT 42→124ms** | 死 |
| **D** prefill-side | C1 재공략 | ❌ Diff B ≈ 1.0 (L≥1024) | — | 死 |
| **E** WIDE 스윕 (짧은 L) | C1 재공략 | ⚠️ L256 1.42 / L512 1.22 **`[강등 제기]`** | ❌ stakes sub-ms ≪ (D) 42→124ms | 死 |
| **F** offline predictor로 고정 | C2 우회 | — | ❌ (D)는 실행시 비용 | 死 |
| **G** "decode-attn은 memory-bound" | C1 재해석 | ✅ lever 오히려 강화 | ❌ 불변 | 판정 불변 |
| **H** pre-created green-ctx | C2 정면 | — | ❌ 절반만 회수 | 死 |

★ **(C1)이 참인 regime마다 (C2)가 깨진다.**

## 1.4 `[감사 대기]` 2026-08-04 강등 제기 6건

claims-auditor가 **계측 결함 2종**을 확인하고 정본 문장 6건의 강등을 제안했다. **미확정.**

**계측 결함**
1. **비대칭 버킷** — `zamba2.py:163` `_zt("attn")`은 RadixAttention **코어만**(qkv_proj·o_proj·MLP 제외), `:259` `_zt("mamba")`는 mixer 전체. 해석적 FLOP 모형상 미계측 GEMM이 ctx2050서 전체 prefill FLOP의 **41.5%**(attn core 합계 2.46%의 17배). ⇒ **이 원자료로 "attn이 prefill의 X%"를 만들 수 없다.** [CONFIRMED]
2. **누산기 러닝평균** — `:391-394` `_zt_acc`가 emit 시 리셋 안 됨 + `knee2d_wide.sbatch:120`이 `tail -1` ⇒ 보고값이 cold-start를 영구 포함. `SM_LIST=(full 44 24 16 8)`이라 **편향이 SM=108 셀에만 체계적으로** 걸림 ⇒ **Diff B 구조적 상향 편향**. ⚠️ **mamba 수치만 제시됐고 attn 측 시계열 미제시** — 확인 대기

★★ **2026-08-04 감사 확장 — 같은 결함이 job 858811(§1-5 근거)에도 적용된다.**
`results/r0c/decode_knee_vs_ctx.sbatch:45`도 `SM_LIST=(full 44 24 16 8)`이고 `:90`이
`tail -1`을 취한다 ⇒ **`full`이 항상 부팅 직후 첫 arm**이라 편향이 **모든 비의 분모**에
걸린다. 추가로 **물리 불변량 위반**이 확인됐다: mamba SSD decode는 ctx에 O(1)이어야
하는데 sm44에서 **1.98× 산포**하고, `full`(108 SM)의 mamba가 sm44보다 **2.34× 느리다**
(ctx256 65.655 vs 28.057). ⇒ **§1-5의 "attn 비중 5%→79%"·"ctx256 = 1.1× SM-free"·
"knee 16→44→108→108"은 인용 금지**, 보정 시 ctx256 민감도 **1.140 → 2.513×**.
상세는 [POSITIVE §2.4](layertype_dynamic_POSITIVE_2026-08-04.md).
⚠️ **이것은 negative가 아니라 positive(P5)를 흔든다** — §1-5는 positive 축 항목이다.

**강등 6건 — ★★ X2′ 검증 완료(2026-08-04): 무효 0건, 서술 수정 2건**

X2′가 `_zt_acc`의 비리셋을 이용해 연속 emit에서 **블록평균을 역산**(4,104 ZBPT 줄 / 285 셀)해
attn·mamba를 **나란히** 재구성했다. 결과: **attn도 부풀려지나 mamba의 1/20~1/2**이다.

| 첫 블록 초과배수 (clean WIDE B=1, SM=108) | L256 | L512 | L4096 | L8192 |
|---|---:|---:|---:|---:|
| **attn** | 0.980 | 1.201 | 1.030 | 1.013 |
| **mamba** | **5.100** | **2.316** | **1.356** | **1.381** |

보고값 수준 중앙값: **attn 1.0009** vs **mamba 1.0215**. OLD `L=2000 B=1`은 **attn 1.0162 vs
mamba 1.3282**(코퍼스 최대 비대칭). 보정의 산술적 출처가 `f_mamba(108)/f_attn(108) =
1.3282/1.0162 = 1.307`이고 실측 비가 **1.3069** ⇒ "attn도 같은 비율로 부풀려졌다면 1.000이어야"
했다. **우려 반증.**

★★ **비순환 확증 — 두 독립 job이 같은 셀을 쟀다**: 보고 mamba 111.578 vs 116.648(**1.0454× 불일치**),
**steady 84.012 vs 83.941(1.0008× 일치)**. ⇒ **보고 Diff B의 1.32 vs 1.38 차이는 물리가 아니라
"몇 번 돌았나"다.**

| # | 대상 | 판정 | 정정값 |
|---|---|---|---|
| 1 | "WIDE로 **확증**: lever는 L≤512서 열림(1.42/1.22)" | **수치 REFUTED, "확증" 철회 정당** | 커널 단위 steady **1.24–1.34(밴드) / 1.198**. ★**정책 단위(U-M)에선 1.032/1.031로 소멸** |
| 2 | "L=2000선 Diff B≈1.35 ⇒ lever는 L↓서 열린다" | **REFUTED 확인** | **1.0081** [1.0078, 1.0084] |
| 3 | "실측 **확증**" | 강등 지지 | — |
| 4 | mamba "포화·**역행**"(9.05%) | ★**부분 정정 — 부호 생존, 크기 3.5× 축소** | "역행 철회"가 아니라 **"9% → 2.59%"** [2.44, 2.77] |
| 5 | `L^1.68/L^0.61` 지수 | 정정 확인. ★**"항등식 의심"은 근거 없음** | steady OLS ctx≥2050 **a=1.916±0.021, b=0.954±0.008** |
| 6 | `diffA_vs_diffB_table.md` L=2000 행 | 지지(사유 3중) | warm-up + clean 실패 + shape 혼합 |

**게이트 대조 (X2′ 자체 검증)**
- **여집합(#10)**: 감사자의 강한 형태 **"SM=108에만"은 반례 있음** — 편향은 *shape 첫 노출*에
  걸린다(`WIDE 16384 B=16`은 full 1.014인데 SM=44/16/8이 1.09).
- **끝점(#8)**: **감사자가 인용한 L=256의 1.240은 가장 노이즈 큰 단일 블록 추정량**이다
  (drop_first 1.336 / second_half 1.325). **L=256은 밴드 [1.24, 1.34]로만 인용 가능.**
- **감쇠**: blk2 이후 ±0.4% 평평 ⇒ **τ ≲ 1 forward, 지수 감쇠가 아니라 1-shot.**
- ★ **기전 — triton JIT 가설은 반증**: 이 캠페인은 attention도 triton이고 `TRITON_CACHE_DIR`이
  양쪽 sbatch에 설정돼 있으며(캐시 323 엔트리) `mamba/ops/*`에 `@triton.autotune`이 없고 로그에
  컴파일 흔적 0건. 지지되는 설명은 **버킷 경계 비대칭**: `layers_block_type[0..5]="mamba"`라
  forward 첫 6개 event 구간이 전부 mamba이고 `_zt("mamba")`는 mixer GEMM을 포함하는데
  `_zt("attn")`은 qkv/o_proj를 밖에 둔다 ⇒ **forward 시작부 일회성 비용이 구조상 mamba에만
  귀속 가능**. 결함 1과 같은 뿌리다.

⚠️ **남은 제약**: `n_indep = 1`(WIDE는 셀당 런 1개). 위 CI는 **런 내 블록 정밀도이지 재현성이
아니다**. 정책적 함의를 붙이려면 셀당 최소 4런(게이트 #3).

**파급 없음(확인됨)**: `CLAIM_EVIDENCE_MATRIX.md` Claim B(서빙 실증만 딛음), `PROJECT_STATUS.md`.

⇒ ★ **§1.1의 서빙 판정은 영향받지 않는다.** 강등이 확정되면 negative는 **더 깨끗해진다** —
서사가 *"lever는 있었는데 (D)가 삼켰다"* → *"정책 단위에서 lever가 애초에 없었다"* 로 단순화.

---

# 2. dynamic multiplexing 축 — best-static을 못 넘음

## 2.1 HE0 — SLO 엄격도 무관

| SLO | 결과 | 유의도 |
|---|---|---|
| **관대(TTFT 3s)** | d44 **3.220±0.013** > d34 3.171 > **bind+GATE 3.132±0.019** > d24 3.081 > slo 2.974 > d16 2.846 (n≥4, 변화 trace) | **5.4σ** |
| **tight(chat 300/50ms, 컨트롤러 재튜닝)** | d44 **73.2%** ≫ d34 49.6 > bind+GATE 44.3 > bind 40.6 | **10σ** |

★ **§1-16의 "tight SLO선 동적 우위"는 §1-17이 반증했다** — 재스코어는 3s-튜닝 컨트롤러의
정착 위치를 **사후 채점**한 아티팩트였다. ⇒ **방법론 게이트 신설**: SLO를 바꿔 평가할 땐
재스코어 금지, 그 SLO로 **재튜닝해 직접 측정**.

## 2.2 死因 — switch overhead가 아니다

| 후보 | 판정 |
|---|---|
| switch overhead | **반증** — switch 2회로 static 매칭한 rep 존재, `slo(5sw) < bind(21sw)` |
| 컨트롤러 CPU 비용 | **반증(직접 계측)** — mean 32–36µs, 누적 wall의 **0.014%** |
| **positioning** | ★ **실제 死因** — 최적 d44인데 컨트롤러는 dec_sm 16–24(평균 22)서 진동, 44에 **절대 도달 못 함**. risk/reward **18:1** |
| **얽힘** | ★ 그 진동 구간이 정확히 파국 구간(decode 굶김 → §1-4) |

## 2.3 구조적 이유 — 두 regime의 최적이 충돌하지 않는다

| 정책 | LO(rate 3) | HI(rate 12) |
|---|---|---|
| d44 (decode-heavy) | 2.858 ± 0.005 | **3.924 ± 0.039** |
| d16 (prefill-heavy) | **2.861** | 2.737 |
| **spread** | **0.067 (2.3%)** | **1.187 (43%)** |

★ **차별의 ~95%가 과부하 phase에서 나오고, 저부하 phase는 split에 완전히 무관심**하다.
⇒ LO엔 쫓아갈 최적점이 없고, HI의 최적은 LO에서도 공짜 ⇒ **"항상 HI 최적"=decode-heavy
static이 정의상 최선**이고 동적은 과도만 지불한다. **동적이 이기려면 regime 간 최적이
*충돌*해야 하는데 이 워크로드엔 그 구간이 없다.**

## 2.4 탈출구 봉쇄 — 4단계

| 단계 | 결과 |
|---|---|
| **mix-스윙 trace**(§1-18) | 최적이 d24↔d34로 **좁게만** 스윙. conjunctive SLO(TTFT∧ITL)가 양끝을 배제 ⇒ 중간 static d24가 양 phase 근최적, **d24 3.930 ≫ bind 3.419** |
| **극단 disjoint**(§1-19) | disjoint region은 **실재**(feasible-A ∩ feasible-B = ∅). **그런데도 동적 패배** — ORACLE 동적조차 **+2.1%**, reactive bind는 **−20.6%**. 게다가 이 region은 **overload에서만 존재** |
| **oracle 재구성**(§1-20) | 진짜 headroom +16%는 **116 SM > 108** 요구 ⇒ **단일 GPU 불가**. 단일 GPU 천장 = coupled ceiling **+2%** |
| **벡터1**(short-ctx disjoint escape hatch) | **CONFIRMED closure (scoped)** — 5단계 스윕 후 companion collapse. 단일 static(d54)이 양 phase 커버 ⇒ **disjoint 없음** |

## 2.5 게이트의 정체

★ feasibility 게이트는 지능적 제어가 아니라 **one-way ratchet auto-tuner**다:
`d24→d34` 1회 이동 후 prefill-ward 복귀를 **113회 전부 거부** ⇒ d34 영구 고정.
게다가 **틀린 static에 조기 수렴**(최적은 d44). **가치는 성능이 아니라 견고성**
(붕괴 1/4 → 0/9, 분산 16× 타이트).

## 2.6 아키텍처로도 못 깬다

`PDMUX_TRUE_DUAL_WORKER=1`은 **control-plane dual-worker**일 뿐이다 — 두 host issue
thread / role별 queue / thread-local role만 분리하고, **running batch·KV/mamba pool·SM
파티션은 전면 공유**된다(`SharedGpuArbiter` 단일 `stream_index`, ≤108). ⇒ 얽힘이 사는
substrate를 **구성상 깰 수 없다.** Claim D는 "control-plane coupling 감소"로 스코프
축소, 등급 **미검증**(GPU correctness gate 이력 없음).

---

# 3. 철회된 것 (되살리지 말 것)

| 철회 항목 | 사유 |
|---|---|
| sim/no-cudagraph의 **1.37–2.02× layer-aware 이득**을 실엔진 결과로 제시 | 등급 오분류 — 서빙에선 agnostic 4/4 승 |
| "최적 split = d16 · 부하 무관 불변" | 실 trace가 반증(d16 1.056 vs d24 6.240) |
| §B "+18%" | no-cudagraph + 비최적 static 이중 confound |
| "stationary 분산 = GPU 클럭 throttling" | 불필요 — 정체는 **메트릭 절벽** |
| stationary ShareGPT r8 벤치 | static d24조차 **5.282±1.302** |
| **Stage 0 "운영점 decode SM-무감각"** | D108 무경합 앵커가 실은 **decode 16 SM**(코드/telemetry/클라이언트 서명 3중) |
| R1 `PDMUX_DUAL_WORKER=1`의 dual-worker 인과 귀속 | observer-path A/B였음 |
| `g = A_free(d16)/A_free(d54)` · `A_free` · pin 게이트 | estimand 미식별 / 항등식 |
| C2 → `G_LEVER` 앵커 도출 | 주장 1만 생존, 2–5 REFUTED |
| "희석 attenuation" 보정 | control-arm reductio로 REFUTED |
| cross-substrate 이식 프레이밍 | MPS=정적, libsmctrl=비-vendor |

---

# 4. negative가 **덮지 않는** 범위 (커버리지 공백)

| 축 | 측정된 것 | 안 덮은 것 |
|---|---|---|
| 컨텍스트 길이 | ShareGPT mean 352(98% L<2k), synthetic in2000/3600 | **long-context 실 trace(8k–128k)**. Zamba2 ctx 4096 상한 |
| **입력 길이 *분포*** | 전 synthetic 캠페인 `range-ratio 1.0` **고정 길이**(52 sbatch 전수 확인) | **혼합비를 축으로 한 스윕 — 전무** |
| 모델 | 정책 캠페인은 Zamba2-2.7B 단독 | 운영점 다-모델 재확인 |
| 동시성 | `max_running_requests` 48 고정 | **얽힘 기전의 축인데 sweep 없음** |
| 예산 제약 프론티어 | — | ★**`ITL(D) vs TTFT(108−D)` 미측정 (E1 미제출)** |
| 하드웨어 | A100 PCIe 1장 | 멀티-GPU, H100급 |

★ **negative의 유효 범위는 "짧은 컨텍스트 · 고정 길이 · 단일 모델 · 고정 동시성 ·
예산-제약 프론티어 미측정" 위에 서 있다.** 이 다섯이 정확히 [JUNCTION 문서](layertype_dynamic_JUNCTION_2026-08-04.md)가
가리키는 미해결 좌표다.

---

# 5. 실전 권고 (negative로부터)

- **peak decode 부하 기준 decode-heavy static 고정.** 검증된 profile이 없으면 agnostic.
- 동적 제어 불요 — **관대·tight SLO 양쪽에서 확정.**
- 게이트를 쓴다면 성능이 아니라 **트랩 억제(견고성)** 목적으로만.
- layer composition은 런타임 스케줄링 경계로 쓰지 않는다. **offline decode-floor
  profile 입력으로만** 쓴다(그 형태도 Claim E로 **미검증**).
