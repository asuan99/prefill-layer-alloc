# 완료 실험 종합 보고 — Hybrid-LLM PD-mux 서빙 (2026-07 ~ 2026-09)

> **정본이 아니다.** 이 문서는 정본(`PROJECT_STATUS.md` > `reports/paper/` >
> `reports/CONSENSUS.md`)의 **요약·연결 서술**이다. 충돌하면 **정본이 이긴다.**
> 새 성능 판정 **0건** · GPU 지출 **0** · 정본 무수정. 기존 결론을 재도출하지 않고
> 수치와 경향성만 정리한다.
>
> 표기: `[READ]` = 로그·설정·코드·정본에서 직접 읽은 값 · `[DERIVED · 규약]` = 이번
> 집계 산출값 · **(BAN)** = 정본 인용 금지 항목(값을 옮겨 적지 않는다).
> 짝 문서: [`EXPERIMENT_DESIGN_2026-09-22.md`](EXPERIMENT_DESIGN_2026-09-22.md).

---

## 0. 한 문단 요약

**PD 분리(agnostic PD-mux) 자체는 운영점에서도 이득이 난다** — 단 술어·모델·워크로드
한정이고 "SM 분할이 원인"이라는 좁은 형태는 아직 지지되지 않는다. 그 위에 얹으려던
**두 개의 정책 레버는 모두 실패했다**: layer-type 런타임 정책은 全형태 死(서빙 실증
반증), 단일-GPU 동적 제어는 best static을 못 넘는다(HE0, n≥4·5.4σ, 관대·tight SLO
양쪽). 실패의 기전은 **전환 비용이 아니라 positioning + entanglement**이고, 전환
비용에는 이제 device-level 상한이 있다. 2026-08 이후 트랙의 무게중심은
"어떤 정책이 이기는가"에서 **"우리가 측정한다고 믿은 것을 정말 측정했는가"**로
옮겨갔고(Stage 0 D108 오류 → 실현 분할 4–19% → sticky → λ0 → residency census),
그 축에서 **라벨이 target이지 allocation이 아니었다**는 발견이 반복됐다.
2026-09-22 감사로 마지막 축이 하나 더 열렸다: **upstream pdmux는 최근접 선행
(MuxWise)의 engine 층 그 자체**이고, 우리 "agnostic"은 거기서 estimator·dispatcher를
뺀 것이다.

---

## 1. 트랙 지도 — 무엇이 무엇을 낳았는가

```
                 ┌──────────────────────────────────────────────┐
     2026-07     │ T1 이식 (P0–P1.7)  4 hybrid × SGLang v0.5.10 │
                 │   forward_split_prefill 4모델 패치           │
                 └───────────────┬──────────────────────────────┘
                                 │  "정책 레버가 있는가?"
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
   ┌──────────────────────────┐   ┌──────────────────────────────┐
   │ T2 layer-type 공간분할   │   │ T3 단일-GPU 동적 제어        │
   │  (R0a–R0d, coord per-type│   │  (SLO-aware Step C–G)        │
   │  ⇒ ★死 (TPOT 42→124ms)   │   │  ⇒ ★死 (HE0, n≥4·5.4σ)      │
   └───────────┬──────────────┘   └──────────┬───────────────────┘
               │ "왜 死인가?"                 │ "왜 지는가?"
               │                              ▼
               │                  ┌───────────────────────────────┐
               │                  │ §1-17 positioning + §1-4 얽힘 │
               │                  │ §1-8/§1-34 전환비용은 병목 아님│
               │                  └──────────┬────────────────────┘
               ▼                             │
   ┌──────────────────────────┐              │
   │ T5 SM 민감도 축          │◄─────────────┘  "레버가 정말 있나?"
   │  C2 decode축 2.36–2.91×  │
   │  prefill축 기울기비 4.7–5.2×│
   └───────────┬──────────────┘
               │ "그 라벨이 실현됐나?"        ★2026-07~08 축 전환
               ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ T6 실현 분할 · 계측 규율                                      │
   │  Stage0 D108=실은 16SM → §1-25 4–19% → sticky(S2) → λ0 →      │
   │  residency census(2026-09-21) → ★코드근거 감사(2026-09-22)    │
   └───────────┬───────────────────────────────────────────────────┘
               │
               ▼
   ┌──────────────────────────┐    ┌──────────────────────────────┐
   │ T4 PD-mux 자체 vs fused  │    │ T7 전환 비용 기전            │
   │  P1-opint 운영점 대조    │    │  kernel_mech / HOLB          │
   │  ⇒ 부분 지지(2모델 한정) │    │  ⇒ s≤0.04ms, 진짜는 생애주기 │
   └───────────┬──────────────┘    └──────────────────────────────┘
               │
               ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ T8 R2 architecture (true dual-worker) — correctness 단계에서   │
   │    막힘. P2 미착수, 블로커 3개.                                │
   │ T9 long-context / stake #1 — 이 기판에서 "구매 불가"로 종결.   │
   │ T10 계보·선행 대조 (2026-09-22) — upstream = MuxWise engine.   │
   └───────────────────────────────────────────────────────────────┘
```

**연결성의 핵심 3줄**

1. **T2·T3의 두 死가 T5·T6을 낳았다.** 정책이 안 먹히자 질문이 "레버가 있는가"
   (T5)로, 다시 "우리가 건 라벨이 실현됐는가"(T6)로 내려갔다.
2. **T6이 T4·T5의 해석을 되돌아가 바꿨다.** Stage 0 D108 오류 → 그 헤드라인 철회,
   §1-25(4–19%) → decode 축 라벨이 target임을 확정, λ0 → "fixed D44 셀이 D44
   측정이 아니다".
3. **T7이 T3의 死因을 닫지 않았다 — 오히려 좁혔다.** 전환이 싸다는 것이
   "그러니 동적을 되살릴 수 있다"가 아니라 "死因이 전환이 아니다"로만 쓰인다
   (정본 금지문장 #12).

---

## 2. 트랙별 결과 — 수치와 경향성

### T1. 이식·기능 (2026-07, P0–P1.7)

| 항목 | 값 | 출처 |
|---|---|---|
| 대상 모델 | NemotronH · Zamba2 · Falcon-H1 · Granite-4 (+ dense 대조 Qwen2.5-7B, pure-Mamba2 음성대조) | [READ] `dev_tree_edits.md` |
| 엔진 | SGLang **v0.5.10** + engine-port 패치 | [READ] |
| hybrid PD-mux를 가능케 한 것 | `forward_split_prefill` **4모델 전부 패치**(zamba2:752 · nemotron_h:859 · falcon_h1:503 · granitemoehybrid:622) | [READ] `dev_tree_edits.md` 항목 2/6/8/9, 24 |
| upstream 기본 제공 범위 | dense 9종만(PR #7634) — hybrid는 **우리가 더했다** | [READ] upstream git |

**경향성**: 이식은 성공했고, "hybrid에서 layer 단위 prefill 발사가 성립하는가"는
**닫힌 질문**이다. 남은 것은 **입도**(§3.3에서 다시 다룬다).

### T2. layer-type 공간분할 정책 — ★全형태 死

| 축 | 값 | 비고 |
|---|---|---|
| 시뮬레이터 예측 이득 | la/agnostic **1.2B 1.37× · 2.7B 2.02× · 7B 1.82×** | [READ] 정본. SLM 한정 아님 |
| 실엔진 서빙 결과 | coordinated per-type에서 **TPOT 42 → 124 ms** | [READ] 정본 — **반증** |
| (B,L) knee 전 격자 | 재배분 lever ≈ **1.0** | [READ] |

**경향성**: 크기가 커져도 시뮬레이터 이득은 유지되는데 **실엔진으로 전이되지
않는다.** ⇒ 정본 방법론 게이트 1("정책 주장은 반드시 서빙 실증")의 출처.
★2026-08-04 계측 결함 2종 감사로 **마그니튜드만 강등**, 死 판정은 서빙 실증
근거로 불변.

### T3. 단일-GPU 동적 제어 — ★HE0

| 항목 | 값 | 출처 |
|---|---|---|
| HE0 | 동적 제어(SLO-aware / binding-first / gate)는 **best static을 못 넘는다** — n≥4, **5.4σ** | [READ] §1-7 |
| SLO 엄격도 의존 | **없음** — 관대(TTFT 3 s)·tight(chat 300/50 ms) 양쪽 확정 | [READ] §1-7·§1-17 |
| 死因 | 전환 오버헤드 **아님**. **positioning**(§1-17) + **entanglement**(§1-4): decode 굶김 → ITL↑ → batch 정체 → admission 차단 → TTFT 폭발 | [READ] |
| 구조적 이유 | 두 regime의 최적이 **충돌하지 않는다**(§1-13) ⇒ 하나로 고정해도 잃는 게 적다 | [READ] |
| 게이트의 정체 | 지능적 제어가 아니라 **undershooting auto-tuner**(§1-10) | [READ] |
| 컨트롤러 CPU 비용 | 직접 계측으로 **死**(§1-12) | [READ] |
| Oracle 상한 재구성 | headroom은 +2%가 아니라 **+16%** — 단 그건 **disaggregation 몫**(§1-20) | [READ] |
| §1-20 재현 시도 | `sgptv{Lo,Hi}` rate-swing 격자(n=4, 정본술어)에서 "+16%" 크기 **재현 안 됨** — 음성 결과, 등급 불변(§1-32) | [READ] |

**경향성**: SLO를 조일수록 동적이 유리해질 것이라는 직관은 **반대로 나왔다**
(§1-16 → §1-17 반증: tight SLO로 **재튜닝**하면 동적이 크게 열위).
⇒ 정본 게이트 8("SLO 바꿔 평가할 땐 재스코어 금지, 재튜닝해 직접 측정")의 출처.

**실전 권고(정본)**: **peak decode 부하 기준 decode-heavy static 고정**(이 워크로드선
d44급). 검증된 profile이 없으면 agnostic.

### T4. PD-mux 자체 vs fused (P1 트랙) — 부분 지지

| 셀 | agnostic v1 vs fused(plain), 정본 goodput 술어 | 상태 |
|---|---|---|
| Zamba2-2.7B rate 2 | **+40.5%** (CI +0.431 ~ +0.620 req/s) | 정상상태, 인용 가능 |
| Zamba2-2.7B rate 3 | **+185.8%** | 정상상태, 인용 가능 |
| Granite-4.0-h-micro rate 3 | **+11.7%** | 정상상태, 인용 가능 |
| Granite-4.0-h-micro rate 4 | **+27.0%** | 정상상태, 인용 가능 |
| Granite rate 2 | +3.4% — **경계**(임계 48 ms로 내리면 소멸) | 경계 |
| Zamba2 rate 4·6, Granite rate 6 | 한쪽 이상이 **용량 위** ⇒ **부호만** 인용 | 부호만 |

[READ] jobs 873944/873945, n=5 paired, 사전등록 `results/p1_opint/PREREG.md`.

**경향성**: rate가 올라갈수록 이득이 커지다가 **용량 천장에서 해석 불가**가 된다.
rate 1에서는 도착률 천장 때문에 **동률**.

**★술어 의존(중요)**: 사전등록이 채택한 mean-ITL(TPOT) 술어로는 Granite에서
**부호가 뒤집힌다**(−0.3 ~ −1.6%, CI 0 배제). 단 그 술어는 Granite rate3·4에서
위반 요청이 0건이라 **goodput ≡ throughput**(항등) — 정본 게이트 #6의 사례.

**★남아 있는 것**: "PD-mux를 켜면 이득"은 **2모델**에서 지지. **"SM 분할 자체"·
"PD 분리 자체"가 원인**이라는 좁은 형태는 **NOT-YET-SUPPORTED**이고, gate #13/#16을
"닫았다"고 쓰는 것은 금지다(batch⊗regime 앨리어스). `%smid` R0·P0-A·F2가 도착했지만
전부 **L0 유사물**이라 Gate 2 귀속을 열지 못했다.

### T5. SM 민감도 축 — 레버 존재만 확립

| 축 | 설계 | 결과 | 등급 |
|---|---|---|---|
| **decode 축 (C2)** | prefill 16 SM 고정, decode SM 16→92 | decode ITL **2.36–2.91×**, **4 arm 모델-무관** | `CONFIRMED(scoped)`. **레버 존재만**, 정책 이득 아님. **인용정지 2건 유효**, B/L 확장 이식 금지 |
| **prefill 축** | decode 16 SM 고정, prefill 16→92 | TTFT~L 기울기비 4 arm 사실상 동일(**4.74–5.16×**) | **REAL scoped, 미감사 ⇒ 정본 인용 금지** |
| 반증된 추론(재사용가치) | "prefill 기울기 > decode 기울기 ⇒ 프론티어서 prefill이 더 급함" | **지지 안 됨** | — |
| **고-SM decode 평탄화 기전**(kernel_mech) | — | 규칙층 감사 rev3→rev8 **연속 `NO-GO`**(rev6–rev8은 死因 없음·국소/명세층) | 미해결 |

### T6. 실현 분할 · 계측 규율 — ★이 프로젝트의 중심축

| 사건 | 발견 | 여파 |
|---|---|---|
| **Stage 0**(2026-07-26 → 07-28 철회) | 헤드라인 "D16≡D108=1.00±0.01"의 **D108 앵커가 실은 16 SM** | 동일조건 반복측정이었음 ⇒ 헤드라인 무효, Claim A 서빙 증거 철회. 교훈 = **pin은 target 아닌 realized로 검증** |
| **§1-25**(2026-08-03) | decode 축 라벨이 **4–19%만 실현**(job 872077: T8 d16 0.038 → d54 0.093, Ha8 0.104 → 0.187; 나머지 81–96%는 무분할 108 SM) | 셀 라벨은 **allocation이 아니라 TARGET**. 희석률이 셀마다 달라 **estimand 내부 교락** |
| **§1-26**(같은 날) | `g` 은퇴, 희석-보정 가설 **REFUTED**, `A_free` 자체가 결함 | 추정량 이관 |
| **sticky**(`PDMUX_STICKY_PARTITION`, 2026-08-03 구현) | 분할을 decode-busy 동안 **유지**하면 실현률이 고쳐진다 | smoke: OFF `E1_DECODE_REALIZED=0.0805` → ON **1.0000** [READ] |
| **S2**(job 873015/873921, 20 boot) | sticky ON에서 T8 d16/d54/d92 전 셀 `ENABLED … fixed target index=1` 21/21 | §1-28/§1-30 최상위 열린 항목 **behavioural 종결**, E1은 **열리지 않음** |
| **λ0**(job 908623, 2026-09-14) | λ\*(A) ≈ **3.05 req/s**(±0.6%, n=2) · λ\*(B) ≈ **0.696 req/s**(±0.3%, n=2), 양쪽 `KNEE_BRACKETED`+`SEED_REPEAT_HOLDS` | ★**λ\*(A)는 D44 측정이 아니다** — 등록 규약 D44 점유 `a_r4` **13.64%** · `b_r3` **97.63%**. (시간가중 열 값은 **(BAN)** E2C-8′) |
| **residency census**(2026-09-21) | 아카이브 전수 재집계 — 1,077 파일 · 14 캠페인 · **33,115,587 스냅샷** · **235,387 s (65.4 h)**, 파싱 오류 0 | 5개 규약(R-cnt/time-all, R-cnt/time-busy, R-iter-busy)을 한 축에 올림 |
| **코드근거 감사**(2026-09-22) | sticky ON PART의 대부분이 **prefill-idle** | 아래 §2.T6-a |

#### T6-a. 2026-09-22 감사가 더한 수치 [DERIVED · R-time-all]

| 캠페인/arm | PART% | ∧prefill-busy | ∧prefill-idle | PART 중 idle |
|---|---|---|---|---|
| `s2_sticky` (20 boot, 전부 ON) | 78.53 | **7.57** | **70.96** | **90.4%** |
| `sticky_smoke` **ON** arm | 86.22 | 3.19 | 83.03 | 96.3% |
| `sticky_smoke` **OFF** arm (같은 job) | 5.57 | **3.07** | 2.50 | 44.9% |

**경향성 1 — sticky가 바꾸는 것은 거의 전부 prefill-idle이다.** 같은 job의 OFF→ON에서
**prefill-busy PART는 3.07 → 3.19%로 사실상 불변**이고, 더해진 80.5%p가 전부
prefill-idle이다. ⇒ "PART 백분율 = 동거 비율"로 읽으면 틀린다.

**경향성 2 — prefill SM이 적을수록 동거 비중이 커진다** (s2_sticky 셀별):

| cell | prefill SM (target) | PART∧prefill-busy (R-time-all) |
|---|---|---|
| d16 | 92 | 0.91 – 2.85% |
| d54 | 54 | 4.07 – 6.79% |
| d92 | **16** | **21.13 – 27.23%** |

⚠️ d92는 **다른 job(873921)·다른 날짜·4블록**이라 cell ≡ job 앨리어스가 있다.

**경향성 3 — 03절 비용 수치는 오염되지 않았다.** PART 체류 decode forward
(granite **1.53–1.66×** · zamba2 **1.73–1.94×**, 매칭 후 1.53–1.54 / 1.79–1.90;
`other_stream_fw_ms` PART **13.7–23.2 ms** vs FULL **0.2–0.6**)의 출처는
`p1_gates/gate2` HOLB 캠페인(jobs 874602/874633/874635)이고 그 캠페인은
**sticky OFF**다 ⇒ PART ⟺ prefill in-flight, 동거가 **구조적**. (풀링값은 **(BAN)**,
C2 값과 대조 금지.)

### T7. 전환 비용 기전 (kernel_mech, 2026-08-22)

| 항목 | 값 | 단서 |
|---|---|---|
| 전환 device-level 상한 | **s ≤ 0.04 ms/전환**, **d ≤ 0.07 ms/경계** | 가법성 가정 · agnostic `adjust_stream_groups` 경로 한정 · **2자리 이상 표기 금지** |
| 진짜 기전 | **prefill 생애주기 경계** — admission **+0.91** · merge **+0.45** · 둘 다+인덱스불변 **+1.29** · 요청 은퇴만 **+0.34 ms** | 전부 매칭 증분, 정상상태 대비. 이 넷이 전부라고 주장하지 않음 |
| 인덱스 불변 경계 | 간극 중앙값 **1.81 ms**(n=2,039) > 전환 경계 두 부류(**1.41 · 0.88 ms**) | ⇒ 순서관계로는 **전환 귀속이 비식별** |
| residency 초과 vs 전환 상한 | **1.7–3.5 ms/step** vs **≈0.001 ms/step** ⇒ **≈10³배**(파일별 1,616–2,984) | 공통 분모로만 비교 |

**경향성**: 전환은 싸다. **그러나 "그러니 layer-aware를 되살릴 수 있다"로 쓰는 것이
가장 위험한 오독 경로**(정본 금지문장 #12), 그리고 "switch-cost 트랙을 닫았다"는
서술도 금지다(컨트롤러 구동 전환·green→green 전환[관측 0건]·포화 운영점 미측정).

### T8. R2 architecture (true dual-worker) — correctness 단계에서 정지

| 항목 | 상태 |
|---|---|
| H-Architecture / H-Policy | **미검증 라이브 가설**. 구현 완료 ≠ 성능 주장 성립 |
| grad-guard 결함 | 커밋 `a9cd8dd`로 수리, job 908534로 인과가 `CONFIRMED(scoped)` 승급 — 단 "지배 원인" 서술은 감사가 **명시 거부** |
| P2 착수 블로커 3개 | ① λ0 `NO-GO` 해소 ② W4 λ\* 실측 부재(부분 이동·미해소) ③ **게이트 #6 불변** |
| GPU 장부 | R2 correctness 트랙 **1.322778 GPU-h** · λ0 트랙 **1.391389 GPU-h** |

### T9. long-context / stake #1 — 이 기판에서 구매 불가로 종결

- *"최적 static split 위치가 워크로드 모양에 따라 움직이는가"*는 규칙층 사전등록
  계열 **누적 17연속 `NO-GO`** 끝에 **구매 불가로 종결**(반증 아님, **스코프 선언**).
- 기전: `prefill SM + decode SM = 108`이 **엔진 강제**라 `μ_p(D)`와 `itl(·,D)`를
  분리할 자유도가 없고, 두 단조성이 상쇄가 아니라 **보강**하므로 순서 역전이
  구조적으로 관측 불가.
- ⚠️ **"장문에서 최적 split이 안 움직인다"로 쓰면 즉시 게이트 위반.**
- 후계: **Q-A(고정 split의 regret 프론티어)** — rev6에서 이 트랙 최초
  `GO-with-caveats`, jobs 906504–906507 제출(≈**10.1 GPU-h** 승인), **완료 0/4 ·
  결과 0건**. **Q-B′** — `GO-with-caveats`, Stage 1 미실행.

### T10. 계보·선행 대조 (2026-09-22 감사) — ★새 축

| 발견 | 내용 |
|---|---|
| **upstream pdmux의 정체** | PR #11592/#12275가 `multiplexing_mixin.py`(209줄)+`pdmux_context.py`(163줄)를 신규 생성. layer-wise prefill은 PR #7634(`jason-fxz` = **Xiaoze Fan**, MuxWise 5저자). 추적 이슈 #10813(`Raphael-Hao` = **Weihao Cui**, MuxWise 2저자)이 **MuxWise arXiv를 "Related resources"로 명시** |
| **우리 agnostic의 정체** | **MuxWise engine − estimator − dispatcher + hybrid `forward_split_prefill`**. upstream `adjust_stream_groups` 위에는 `TODO(jason-fxz): This is a temporary demo` 가 그대로 있다 |
| **선행이 이미 보고한 것** | "분할을 고정/격리해도 동거 때문에 decode가 느려진다" — MuxWise §3.3.1(dense Llama, A100·H100, **≈0–30%**) **와** Bullet §3.2.3(`isolated SM`에서도 memory/network 경합 잔존, decode가 prefill보다 경합에 민감) **둘 다** |
| **선행이 보고하지 않은 것** | **정의·분모·규약을 갖춘 집계 추정량으로서의 residency**. (Bullet Fig. 20a는 SM 구성별 지속시간 막대 **타임라인**을 보여준다 — 각주로 인정해야 한다) |
| **전환/재구성 비용** | Bullet camera-ready Table 5 `Resource Re-config` **Mean 4.1 μs / P99 5.9 μs** (CPU-side). 우리 `s ≤ 0.04 ms`는 device-level 상한 ⇒ **같은 자릿수·다른 양** |
| **평가 조건** | 우리만 **hybrid**. Bullet은 **A100 1장 평가도 갖고 있다**(§4.2.1). SLO regime이 셋 다 다름(우리 ITL p95 60 ms / MuxWise TBT 50·100 ms / Bullet TPOT 150–200 ms + 정규화 TTFT) |
| **★회부 항목** | **Bullet §4.4** *"there is no optimal fixed SM allocation"* — HE0와 문면상 정면 반대. 5개 축이 달라 **판정하지 않았다**, claims-auditor 회부 |

상세: [`../audit/2026-09-22_scope_lineage/REPORT.md`](../audit/2026-09-22_scope_lineage/REPORT.md).

---

## 3. 이번 종합에서 새로 드러난 구조적 사실 (GPU 0, 기술 집계)

### 3.1 아카이브 전체의 PART 체류 지형 [READ · census_groups_by_campaign.csv]

| 캠페인 | files | R-time-all PART% | cohab P-act% | 평균 PART 체류(격자제한) |
|---|---|---|---|---|
| `stage0_xctrl` | 42 | 91.09 | 99.69 | 14.46 s |
| `s8p_prefill` | 18 | 80.65 | 91.08 | 7.51 s |
| `s2_sticky` | 20 | 78.53 | **11.80** | 34.86 s |
| `s8_scaleup` | 408 | 69.15 | 97.13 | 22.72 s |
| `sticky_smoke` | 2 | 57.57 | **5.44** | 13.67 s |
| `s0_deconfound` | 24 | 51.52 | 99.76 | 88.19 s |
| `kernel_mech` | 1 | 44.55 | 23.08 | 8.88 s |
| `longctx_conflict` | 56 | 33.21 | 99.14 | 6.53 s |
| `p1_gates` | 272 | 32.32 | 71.28 | 1.19 s |
| `r2_eval` | 11 | 29.76 | 89.98 | 2.48 s |
| `s8_frontier` | 152 | 15.24 | 91.82 | 0.82 s |
| `e1_traceforce` | 8 | 15.80 | 99.28 | 0.36 s |
| `r2_correctness` | 29 | 14.11 | 99.84 | 0.82 s |
| `slo_sched` | 34 | 11.60 | 64.78 | 0.49 s |

**경향성**: **sticky를 쓴 두 캠페인만 `cohab P-act`가 한 자릿수~10%대**이고
나머지는 전부 **90%대 이상**이다. 즉 **sticky 캠페인을 제외하면 PART ≈ 동거**이고,
sticky 캠페인에서만 PART가 "분할 유지"를 뜻한다. 이것이 T6-a 경향성 1의 일반화다.
(`kernel_mech` 23.08%는 파일 1개·PART-busy n=26으로 표본이 극히 작다.)

### 3.2 ★층 단위 prefill 발사는 **짧은 컨텍스트에서 발화하지 않는다** [DERIVED]

코드 사실: `forward_count = max(1, split_forward_token_budget // extend_num_tokens)`
가 **한 번의 `forward_split_prefill` 호출이 도는 층 수**이고
(`model_runner.py:2717-2727`), `split_index`는 그만큼씩 전진한다. 따라서

> `extend_num_tokens ≤ split_forward_token_budget / num_hidden_layers` 인 요청은
> **한 번의 호출로 전 층을 돈다** = 층 단위 분할이 **발화하지 않는다.**

우리 config는 **59/59 전부 `split_forward_token_budget: 65536`** 이고 **한 번도
스윕된 적이 없다** [DERIVED · 전수 yaml 파싱]. 임계 토큰 수 = 65536 / L:

| 모델 | L (= terminal `split_index`) | 분할 발화 임계 |
|---|---|---|
| Qwen2.5-7B (T8) | 28 | ext ≥ **2,341** 토큰 |
| Nemotron-Nano-9B-v2 계열 | 56 | ext ≥ **1,171** 토큰 |
| Zamba2-7B (Ha8) | 81 | ext ≥ **810** 토큰 |

아카이브 실측 (`scratch/split_index_census.py`, prefill-active 스냅샷의
`split_index` 히스토그램):

| 캠페인 | prefill-active 스냅샷 | terminal index 비율 | 중간 사다리 |
|---|---|---|---|
| `s2_sticky` (ShareGPT, T8) | 1,228 | **100.00%** (28) | 없음 |
| `sticky_smoke` (ShareGPT, Ha8) | 10 | **100.00%** (81) | 없음 |
| **`e1_traceforce`** (ShareGPT, T8, **`FORCE_PREFILL=1`** 15,453건) | 16,512 | **99.84%** (28) | 20/21/23/25 합 **27건(0.16%)** |
| **`longctx_conflict`** (ctx 16384) | 73,723 | 94.06% (56) | **4,8,12,…,52 각 ~0.43–0.56%** — 4층 간격 사다리 |
| `r2_eval` | 30,042 | 99.01% (56) | 8/12/16/24/32/36/40/48 희소 |

**★독립 교차검증** — 코드 공식이 예측한 값과 관측이 맞는다. `s2_sticky`의 클라이언트
`input_lens`(n=1,600, ShareGPT `--sharegpt-context-len 3600`) 분포는
min 2 · p25 21 · **median 135** · p75 407 · p90 744 · p99 2,087 · max 3,357 (mean 285)
[DERIVED · bench_serving `input_lens`]. 여기에 위 임계를 적용하면:

| 모델 | 임계 ext | **예측** monolithic 요청 비율 | **관측** terminal-index 스냅샷 비율 |
|---|---|---|---|
| Qwen2.5-7B (L=28) | 2,340 | **99.2%** | **99.84%** (`e1_traceforce`, force-trace) |
| L=56 | 1,170 | 96.8% | 94.06% / 99.01% (longctx / r2_eval, 다른 워크로드) |
| Zamba2-7B (L=81) | 809 | 94.2% (`s2_sticky` 분포) · 90.0% (`sticky_smoke` 분포) | 100% (n=10, 표본 과소) |

예측 99.2% ↔ 관측 99.84%가 자릿수·크기 양쪽에서 맞고, 남는 ~0.8%가 바로 관측된
중간 사다리 27건(0.16%)에 대응한다. ⇒ **두 독립 경로(코드 공식 + 클라이언트 입력
길이 분포 / 엔진 telemetry 히스토그램)가 같은 결론을 낸다.**

**★양성 대조가 내장돼 있다**(교훈 232): `longctx_conflict`의 **4층 간격 사다리**가
계측기가 중간 상태를 **해상할 수 있음**을 같은 집계 안에서 보여준다
(`65536 // ext = 4` ⇒ ext ∈ [13,108, 16,384] — ctx 16384와 정합). 따라서 짧은
컨텍스트 캠페인의 terminal-only는 **계측 실패가 아니다.**

**의미 3줄**
1. **ShareGPT 계열 캠페인 대부분에서 prefill은 monolithic으로 발사된다** —
   MuxWise가 "bubble-less multiplex engine"의 핵심으로 드는 layer-wise 발사가
   우리 운영 설정에서는 **사실상 꺼져 있다**.
2. 이것은 정본 **§1-24**("이 기판의 ITL 꼬리는 decode step time이 아니라
   *monolithic prefill이 decode를 멈춘 시간*이 지배하며, 그건 **구조적**이다")에
   **코드·아카이브 양쪽에서 대응하는 기전**을 준다.
3. MuxWise는 같은 양을 **동적으로** 정한다 — `N_PL = ⌈(T_d × N_T)/T_P⌉`, 즉
   *"발사한 prefill 층의 지연이 대응 decode iteration을 덮도록"*. 우리는 **정적
   토큰 예산**을 쓰고, 그 기본값에서 그것이 "전 층 한 번에"로 붕괴한다.

⚠️ **판정이 아니다.** 이 절은 코드 사실 + 아카이브 히스토그램이다. "그래서 성능이
어떻다"는 어떤 문장도 여기서 따라 나오지 않는다 — 측정 설계는 짝 문서 **D-6**.

---

## 4. 현재 claim 등급 (정본 `CLAIM_EVIDENCE_MATRIX.md`)

| Claim | 현재 판정 |
|---|---|
| **A** hybrid composition/context/load에 따라 decode demand가 변한다 | **부분 지지** |
| **B** layer-level reconfiguration이 ITL critical path와 CUDA Graph를 훼손한다 | **강한 지지**, 현 구현 범위 한정 |
| **C** decode starvation이 TTFT도 악화시킨다 | running-batch 경로 **강함**; KV 경로 **부분** |
| **D** execution-state separation이 single-worker coupling을 줄인다 | **미검증** (2026-07-24 코드 리뷰로 scope 축소) |
| **E** hybrid-informed decode floor가 generic/global static보다 높은 SLO goodput | **미검증 핵심 가설** |
| **F** short-ctx conflict-regime에 동적이 이길 escape hatch가 있다 | **강한 지지(범위 한정)** — escape hatch **없음**을 확정 (scoped negative) |
| **G** PD-mux 활성화 **자체**가 goodput 이득의 원인이다 | **부분 지지(모델·워크로드 한정)**; "SM 분할 자체"의 좁은 형태는 **NOT-YET-SUPPORTED** |

---

## 5. 측정 규율 — 이 프로젝트가 값을 치르고 배운 것

정본 `CONSENSUS.md` §3은 현재 **항목 279**까지, `PROJECT_STATUS.md` 방법론 게이트는
**#259**까지 있다. 이 종합에 직접 걸리는 것만:

| 교훈 | 사건 |
|---|---|
| **pin은 target이 아니라 realized로 검증하라** | Stage 0 D108 → §1-25 4–19% → λ0 "fixed D44가 D44가 아니다" (3회 재발) |
| **게이트·확증서술·양성대조·감사 도구 자신이 항등식일 수 있다** | 20회 이상 재발. `phase=="measure"` 필터, `kv_mamba_occupancy=1.0`, `E-pact` |
| **정책 주장은 반드시 서빙 실증** | T2: 시뮬 1.37–2.02× → 실엔진 반증 |
| **재스코어 금지, 재튜닝해 직접 측정** | §1-16 → §1-17 반증 |
| **duration은 합산, `max()`는 3× 부풀림** | 실제 하네스 버그 |
| **추정량 열은 규약 없이 인용 불가** | E2C-8′ (같은 셀이 규약에 따라 두 값) |
| **변이 검증에는 무변이 대조군이 필수 — 단 거짓 생존은 못 잡는다** | census M0–M4 (M4가 잡은 것은 index 기반 정의뿐) |
| **"바이트 동일"을 보고하려면 그 계측기가 arm 차이를 해상함을 먼저 실증하라** | job 908179 (긍정 사례). ★이번 §3.2의 longctx 사다리도 같은 형태의 양성 대조 |
| **코드 분기 우회 확인이 상위 제약 소멸을 함의하지 않는다** | major 10+ `ValueError` 스코프 정정 |
| **존재 검사는 grep이 아니라 파서로** | ★2026-09-22 신규: `*pdmux*.yml` **55/59**(59/59 아님) |

---

## 6. 자원 장부 (정본 기준)

| 항목 | 값 |
|---|---|
| 누적 제출 (SLURM 계정) | **1,153 jobs / 3,698.3 CPU-h** (마지막 접수 job **908623**, 2026-09-14T21:30 정상 완주) |
| R2 correctness 트랙 | **1.322778 GPU-h** |
| λ0 트랙 | **1.391389 GPU-h** |
| gate #13 job-축 캠페인 | 등록 11.52 → 실측 **11.66 GPU-hr** (+취소 프로브 0.06) |
| G16 캠페인 | ≈**3.9 GPU-hr** |
| Q-A regret (제출·미완) | 승인 ≈**10.1 GPU-h**, 완료 **0/4** |
| E2 (등록·미실행) | **1.803 GPU-h** (최악 2.338, 하드캡 3.0), **지출 0** |
| **2026-09-15 이후 전체 GPU 지출** | **0** — SLURM 계정 블로커 |

---

## 7. 지금 막혀 있는 것 (한 화면)

1. **실행 자체가 막혀 있다** — SLURM 계정이 2026-09-15부터 전 파티션 제출 거부
   (`--test-only`도 거부; association·QOS 한도 비어 있음, 파일시스템 쿼터 정상 ⇒
   KISTI 외부 계정 관리 시스템). 원인 **미확정**(만료 vs 한도초과, 정황은 만료).
   해소는 **사용자 영역**(`account@ksc.re.kr`, 042-869-0597).
2. **기판이 바뀐다** — 2026-09-17 이양 이후 GPU 실행은 **대여 서버**(업체 미정).
   기존 격자·λ\*는 기판이 바뀌면 **그대로 성립하지 않는다**(게이트 1).
   E2 기판 이식성은 **`REFUTED`**(예산 집행기·상수·과금 모형 3경로), 새 OVERRIDE는
   **사용자 결정 사항**(미승인).
3. **P2(architecture)는 블로커 3개로 미착수** — λ0 `NO-GO` 해소 · W4 λ\* 실측 부재 ·
   게이트 #6.
4. **Q-A는 제출됐으나 결과 0건.**

---

## 8. 하지 않은 것

- **GPU 0.** 제출·기동·벤치 0건. 모든 수치는 디스크의 telemetry·로그·설정·코드
  재집계이거나 정본 인용이다.
- **정본 무수정.** `PROJECT_STATUS.md`·`reports/paper/`·`reports/CONSENSUS.md`·
  `RESUME.md` 편집 0건. 이 문서는 정본이 아니다.
- **새 성능 판정 0건.** 순위·우열·goodput·SLO·운영점·인과 귀속 진술 없음.
  HE0·layer-type 死·정책 순위·Claim A–G 등급 **전부 불변**.
- **결론 재도출 없음.** 확정된 것은 요약만 했고 다시 계산하지 않았다.
- **인용 금지 준수**: 풀링 residency 절대값·λ0 시간가중 열·REFUTED된 STEP0 규모
  수치 등 `citation_stops.tsv`에 등재된 값은 **하나도 옮겨 적지 않았다**
  (`check_citation_stops.py --file` 통과 — 목록을 여기 나열하는 것 자체가
  위반이라 등재 파일을 참조로만 가리킨다).
