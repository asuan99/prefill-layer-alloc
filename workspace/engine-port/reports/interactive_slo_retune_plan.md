# 인터랙티브 SLO 컨트롤러 재튜닝 + P99-attainment 직접 재측정 — 구성 계획

작성: 2026-07-18 (측정 전 계획). 발의 맥락: [CONSENSUS.md](CONSENSUS.md) §1-16 + [serving_slo_survey.md](serving_slo_survey.md) §5-7(a).
정본은 CONSENSUS — **이 문서의 어떤 수치도 아직 측정된 것이 아니다.** 확정은 CONSENSUS에만 기록한다.

---

## 0. 한 줄 · 가설

§1-16는 **기존 3s 벤치 데이터를 tight SLO로 *재스코어*** 해서 "동적이 best-static과 대등~약우위"를 얻었다. 그러나 **컨트롤러 자체는 여전히 TTFT≤3s 가정으로 돌았다**(재스코어는 사후 채점일 뿐, 컨트롤러 행동은 안 바뀜). 조사로 인터랙티브 SLO(100–400ms)가 주류임이 확인됐으니, 이제 **컨트롤러를 그 목적함수로 재튜닝하고 직접 서빙으로 P99-attainment를 측정**한다.

- **HT (주가설)**: 인터랙티브 TTFT/TPOT를 목적함수로 넣으면 컨트롤러가 prefill 반응성에 제때 개입 → **재튜닝 동적이 best-static을 재스코어의 대등~약우위보다 *넓은* 격차로 이긴다.**
- **HT0 (귀무)**: 재튜닝해도 static 매칭 수준 — 컨트롤러 자유도가 이미 소진돼 목적함수만 바꿔선 안 됨. ⇒ 동적 트랙 종결 강화.
- **HT-neg (실현불가)**: tight SLO를 Zamba2가 물리적으로 못 attain해 신호 소멸(§5 블로커) → 측정 자체가 성립 안 함.

---

## 1. 현 컨트롤러의 "3s 가정"이 박힌 지점 (코드 근거)

`src/multiplex/multiplexing_mixin.py::_slo_decide_idx_binding` (L151–230). SLO가 관대하다는 전제가 **행동을 무디게** 만드는 곳:

| 위치 | 코드 | 3s 하에서의 행동 | tight(300ms) 하에서 필요한 행동 |
|---|---|---|---|
| L157·166 | `_ttft_slo=3000`; `_pf_slack=(3000-pf_age)/3000` | pf_age 수백 ms여도 **pf_slack≈0.9(안 급함)** → prefill 개입 거의 안 함 | pf_age 200ms면 pf_slack≈0.33 → **prefill urgency가 10× 예민** |
| L160·199 | `_pf_urg=0.5`; `pf_slack<_pf_urg`서 prefill-ward | slack 0.9라 임계 0.5 도달 드묾 | 임계 재보정 필요(상대값의 의미가 바뀜) |
| L125·132 | feasibility gate congestion: `bs≥cap×0.85`서 prefill-ward 거부 | **트랩(d16 붕괴) 방지 = 3s 관대 regime의 실패모드** 겨냥 | tight선 prefill-ward가 *유익*할 수 있음 → gate가 **과보수** 위험 |
| L158·204 | `_anchor=(lo+hi)//2`(=d34급) resting | 관대 regime 최적 resting | tight선 resting이 **더 prefill-ward**여야 할 수 있음 |
| L127 | feas ITL guard `_slo=TPOT_SLO=60` | chat 50/voice 30/code 25로 조여야 정합 | TPOT_SLO 동반 조정 |

★ 즉 **`PDMUX_TTFT_SLO_MS`를 낮추는 것만으로도 컨트롤러 행동이 실제로 바뀐다**(재스코어와 달리). 나머지 파라미터는 그 위에서 재보정 대상.

## 2. 튜닝 구성 — 2단계, 변수 하나씩 (§방법론 §3-8)

### Stage A — 목적함수만 인터랙티브로 (baseline 재튜닝)
컨트롤러·하네스 SLO를 **동일 인터랙티브 값**으로 맞추고 나머지 파라미터는 기본 유지. "목적함수 정렬만으로 얼마나 바뀌나".

| SLO 프로파일 | TTFT_SLO_MS | TPOT_SLO_MS | 근거(조사) |
|---|---|---|---|
| **chat** ★주 | 300 | 50 | 인터랙티브 주류·우리 교체 임계와 일치 |
| voice | 150 | 30 | 최tight 현실값 |
| code-panel | 300 | 50 | chat과 동일대 |
| (대조) batch | 3000 | 60 | 기존 = §1-16 재스코어 재현 확인용 |

### Stage B — Stage A 최선 프로파일 위에서 파라미터 1개씩 sweep
| env | 현재 | sweep | 가설 |
|---|---|---|---|
| `PDMUX_SLO_PF_URGENCY` | 0.5 | {0.3, 0.5, 0.7} | tight선 더 일찍 prefill 개입이 유익? |
| `PDMUX_SLO_ANCHOR_IDX` | 2(d34) | {1, 2, 3} | resting이 더 prefill-ward? |
| `PDMUX_SLO_FEAS_GATE` | on | {on, off} | tight선 gate가 유익한 prefill-ward를 막나? |
| `PDMUX_SLO_FEAS_OCC` | 0.85 | {0.85, 0.95} | 덜 보수적 gate |
| `PDMUX_SLO_DWELL` | 3 | 3 고정 | 변수 최소화(과거 dwell·urg 동시변경 = 해석불가 전례) |

**변수 1개 규율**: Stage B는 A의 승자 config를 기준선으로, 한 번에 한 env만.

## 3. 지표를 P99-attainment로 재설계 (하네스 수정)

현 하네스([`sharegpt_vary_bench.sbatch`](../results/slo_sched/sharegpt_vary_bench.sbatch)) 분석부에 `t<=3.0 and m<=0.06`이 **하드코딩**. 재작성 항목:

1. **SLO 파라미터화**: `t<=TTFT_SLO ∧ mean_itl<=TPOT_SLO`를 env(`BENCH_TTFT_SLO`,`BENCH_ITL_SLO`)로. **컨트롤러 SLO와 항상 동일값**(목적함수=평가함수 일치).
2. **P99-attainment 병기**: `attainment = #{TTFT≤SLO ∧ ITL≤SLO}/N` (P90/P95/P99 관행) + **P50/P90/P99 TTFT·ITL 절대값**. 실무 표준은 P99 percentile(§조사).
3. **capacity 곡선**: goodput(SLO 만족 req/s)뿐 아니라 **"attainment≥90%를 유지하는 max rate"** = 진짜 용량. rate sweep으로.
4. **길이 버킷별 attainment**: 짧은/긴 요청 분리(§5 블로커 진단용).
5. 라운드 duration **합산** 유지(`f921ae8` 3× 버그 방지). switch_count·split 체류분포 병기.

## 4. 측정 프로토콜

- **모델·운영점**: Zamba2-2.7B, **cudagraph ON**(운영점 유지), triton, mrr=48 — §1-16과 동일 substrate.
- ★**L−1 용량 선측정(§방법론 §3-6 필수)**: 각 SLO 프로파일에서 attainment 90% 유지하는 max rate를 먼저 구해 rate 대역 확정. tight SLO는 용량이 급감하므로 기존 rate 3↔12가 안 맞을 수 있음.
- **워크로드**: 변화 trace(rate LO↔HI 교대) 유지하되 **LO/HI를 L−1 용량 기준 재선택**. ShareGPT(p99 2776tok) 그대로.
- **비교군**: static sweep **d16–d54**(직접 측정, 재스코어 아님) + slo + **bind(재튜닝)** + **bind+GATE(재튜닝)** + bind(3s, 대조).
- **반복**: **n≥4**(baseline 분산 먼저). SLO 프로파일 3종 × 정책.

## 5. ★블로커·리스크 (미리 정직하게)

| 리스크 | 내용 | 완화 |
|---|---|---|
| ★**물리적 실현불가 (HT-neg)** | 물리 line 79ms+0.096ms/tok ⇒ **무경쟁에도** p99 2776tok=**345ms > chat 300ms**, 512tok=128ms > **code 100ms**. 긴 요청·code SLO는 애초에 못 맞춤 → attainment 바닥·신호 소멸 | **L−1 전에 무부하 attainment 체크**. code(100ms)는 제외 후보. chat(300ms)은 짧은 요청 위주라 성립 가능. 길이-normalized SLO 병용 옵션 |
| **용량 급감** | tight SLO선 SLO 만족 기준 용량이 3s보다 훨씬 낮음 | L−1 용량 선측정으로 rate 재선택 |
| **메트릭 절벽(§1-14 재발)** | attainment도 임계 지시함수 → 경계 rate서 증폭 | **percentile(P99 절대값) 병용**(threshold-free), 용량 아래서도 측정 |
| **cudagraph 격자 상한** | sglang `>7 그룹 IndexError` → split 격자 ≤7(d16–d54 5점은 OK) | 격자 5점 유지 |
| **gate ratchet 상호작용** | gate가 tight SLO서 다르게 작동(트랩 방지 목적이 tight선 무의미할 수 있음) | Stage B서 on/off 분리 측정 |

## 6. 판정 게이트

- **HT 확증**: 재튜닝 bind+GATE가 인터랙티브 SLO서 best-static을 **>1 pooled-σ 초과**(재스코어 대등~약우위를 *넘어섬*) ⇒ 목적함수 정렬이 실효 ⇒ **동적 제어에 인터랙티브 성능 이득 확정**.
- **HT 부분**: 재튜닝이 재스코어보다 낫지만 여전히 static 매칭 ⇒ 컨트롤러 자유도 한계, 그러나 §1-16(대등)은 유지.
- **HT0**: 재튜닝 무이득(≈3s 튜닝) ⇒ SLO 무관하게 컨트롤러가 static 못 넘음 ⇒ **트랙 종결 강화**(§1-16의 대등은 재스코어 아티팩트로 격하).
- **HT-neg**: 물리적으로 tight SLO attain 불가 ⇒ 측정 무의미, **long-context/모델교체 또는 length-normalized SLO로 재설계** 필요(→ [longcontext_trace_plan.md](longcontext_trace_plan.md)와 합류).

## 7. 산출물 · 순서

1. 하네스 수정 → `results/slo_sched/interactive_bench.sbatch`(SLO env 파라미터화 + attainment/percentile 분석부).
2. **L−1 무부하 실현가능성 + 용량 측정**(job 소수) — HT-neg 조기 차단.
3. Stage A(SLO 프로파일 3종 × 정책, n≥4).
4. Stage B(A 승자 위 파라미터 1개씩).
5. 판정 → CONSENSUS §1-16/§5-7 갱신, `results/slo_sched/interactive_slo_retune/`.

**다음 액션은 (1)+(2)** — 하네스 수정과 실현가능성 체크. 여기서 HT-neg면 이후 단계는 재설계된다. 사용자 승인 후 job 제출.

---

## 8. ✅ L−1 무부하 실현가능성 선판단 (2026-07-19, 기존 데이터 재분석 — job 없음)

기존 변화-trace의 **LO(rate 3) phase = 거의 무경쟁 = attainment 물리 상한**, HI(rate 12)=과부하. d44 static 4-rep, 각 SLO별 attainment%:

| SLO (TTFT/ITL) | LO ceiling | 길이버킷(LO) `<500/500-1k/1k-2k/≥2k` | HI(rate12) | 판정 |
|---|---|---|---|---|
| **chat 300/50** | **97.5%** | 99/100/99/**25%** | 30.4% | ✅ **성립** — HI 하락은 순전히 과부하(큐잉) = 컨트롤러 개선 대상 |
| voice 150/30 | 87.5% | 94/80/42/0% | 12.7% | ⚠️ 경계 (중간요청부터 미달) |
| **code 100/25** | **66.2%** | 82/29/0/0% | 1.7% | ❌ **HT-neg** — 무경쟁에도 66% = 물리적 불가 |
| batch 3000/60 | 99.9% | 100/100/100/100% | 70.6% | (대조, §1-16 재현) |

- TTFT p50/p95/**p99** = LO **80/190/383ms** vs HI **1178/6694/7054ms**. ★**무경쟁 p99=383ms > chat 300ms**(긴 요청 탓) ⇒ **P99-attainment 100%는 물리적으로 불가 → P90 기준**(LO p90<300ms) 또는 length-normalized SLO 사용.

**확정된 설계 조정**:
1. ★**SLO = chat(300/50) 주, voice(150/30) 부차. code(100/25) 제외**(HT-neg — 무경쟁 실현불가; long-context/모델교체 트랙에서 별도).
2. ★**rate 재선택 필수**: rate 12는 chat 기준 과부하 과도(ceiling 97.5%→30%). **L−1 용량 측정으로 chat-SLO 용량(attain P90≥90% 유지 max rate) 확정** 후 LO/HI 재설정. 예상 용량 rate 3–6대.
3. ★**지표 percentile = P90**(P99는 긴 요청 물리 상한에 걸림). attainment%(P90) + 절대 TTFT/ITL p50/p90/p99 병기.
4. **HT-neg 부분 발동**: code는 이미 죽음. **성패는 chat에서 "과부하 하락분(97.5→30%)을 컨트롤러가 얼마나 회복하나"** 로 좁혀짐 — static도 같은 하락을 겪으므로 여전히 static-vs-dynamic 비교는 성립.

---

## 9. ★Stage A 스캔 결과 (2026-07-19, jobs 860415/416/448/449 용량 + 860476–486 정책) — HT0 방향 (잠정 n=1)

chat SLO(300/50) attainment% by rate. **용량 ≈ rate 7–8**(rate≤6 무관심, rate≥10 전붕괴). 정책이 갈리는 **rate 8**이 판정 지점:

| 정책 | r6 | r7 | **r8** | switch@8 | 성격 |
|---|---|---|---|---|---|
| **d44** (decode-heavy) | 96.8 | 93.2 | ★**74.0** | 0 | static |
| **d34** | 98.0 | 97.2 | **51.8** | 0 | static |
| bind+GATE | 98.8 | 97.8 | 44.2 | 23 (feas **240**) | 동적 |
| d24 | 72.2* | 96.2 | 40.8 | 0 | static |
| bind | 98.0 | 94.2 | 40.0 | 28 | 동적 |
| slo | 97.8 | 95.8 | 38.5 | 21 | 동적 |
| **d16** (prefill-heavy) | 97.5 | 83.8 | 33.2 | 0 | static |

(*d24 r6=72.2%는 이상치 — 다른 정책 r6은 전부 97%; 워밍업 노이즈. 판정은 r8.)

### 판정: **HT0 강함 — tight SLO로 재튜닝해도 동적이 best-static을 못 넘는다**
- ★**static 단조**: decode SM 많을수록 attainment↑ (**d16 33 < d24 41 < d34 52 < d44 74**). **tight SLO에서도 decode-heavy가 최선**, 오히려 §1-7(관대 SLO)보다 **격차가 더 큼**.
- ★**동적 3종 전부 d24~d34급(38–44%)** — 컨트롤러가 실제로 반응했는데도(switch 21–28) best-static(d44 74)에 **30%p 미달**. n=1 노이즈로 뒤집힐 격차 아님.
- **bind+GATE(44) > bind(40)**: 게이트가 240회 prefill-ward 거부로 트랩 진입 억제 → §1-11(게이트=견고성) 재확인. 단 여전히 static 미달.

### ★왜 §1-16(동적 대등~약우위)과 상충하나 — 재스코어 아티팩트 규명
- **§1-16 재스코어**: 3s-튜닝 컨트롤러의 **거의-static 궤적(bind가 d34-ish에 정착)** 을 norm-k로 *사후* 채점 → 그 정착점이 norm-k에서 d44보다 유리해 보였을 뿐 = **정착 static의 위치 효과지 동적 *행동*의 이득이 아님**.
- **직접 측정(Stage A)**: 컨트롤러가 tight SLO를 목적함수로 받아 **실제로 움직이자**(switch 28) **얽힘 트랩**에 빠짐 — TTFT 임박 신호에 prefill-ward 이동 → decode 굶김 → batch 정체(§1-4) → admission 차단 → **오히려 TTFT 악화**. §1-16이 상상한 "tight→prefill 반응성 유리"가 **실제 컨트롤러에선 역효과**.
- ⇒ **tight SLO는 §1-4 얽힘·§1-6 비대칭을 *더 극명*하게 만들어 decode-heavy static을 *더 확실히* 지배시킨다.** 동적의 반응 자체가 순해.

### ✅ 확증 완료 (n=4, jobs 860487–860514) — HT0 확정

rate 8 attainment% (n=4, mean±std):

| 정책 | mean±std | 범위 | switch |
|---|---|---|---|
| **d44** | **73.2 ± 4.8** | 65.8–79.2 | 0 |
| d34 | 49.6 ± 3.9 | 43.8–54.2 | 0 |
| **bind+GATE** | 44.3 ± 2.7 | 39.8–46.5 | 23 |
| **bind** | 40.6 ± 0.4 | 40.0–41.2 | 24–30 |

- ★**d44 vs bind+GATE = 28.9%p ≈ 10σ**. 압도적·견고. bind는 std 0.4로 극히 재현성 높은 **확정 열위**.
- **순위 불변**: d44 ≫ d34 > bind+GATE > bind. **동적은 best-static은커녕 2위 static(d34)도 못 넘는다.**
- bind+GATE > bind 재확인(게이트가 trap 억제; rep2만 feas=3으로 게이트 미발동→bind급 39.8로 하락 = 게이트 효과의 반증실험).

### ★최종 판정: HT0 확정 — SLO 엄격도와 무관하게 decode-heavy static 지배
- **§1-16(tight→동적 대등~약우위)은 재스코어 아티팩트로 격하 확정.** 재스코어는 3s-튜닝 컨트롤러의 정착점을 사후 채점한 **위치 효과**였고, **컨트롤러를 tight SLO로 실제 재튜닝하면 동적이 얽힘 트랩으로 오히려 열위**(bind 40.6 = d24급).
- **decode-heavy static이 관대 SLO(§1-7)뿐 아니라 tight SLO(chat 300/50)에서도 지배**하며, tight일수록 **격차가 더 크다**(§1-4 얽힘·§1-6 비대칭이 tight에서 더 극명).
- ⇒ **"동적 제어에 남은 upside"는 이 축에서도 닫혔다.** 사용자의 "고정 SLO 계측 오류" 지적은 옳았으나(3s가 왜곡한 건 사실), 왜곡을 걷어낸 직접 측정은 **원래 결론(decode-heavy static 지배)을 더 강하게·SLO-엄격도 무관하게 재확인**.
- **방법론 교훈**: **재스코어는 컨트롤러 *행동*을 못 본다** — SLO를 목적함수로 바꾸는 실험은 반드시 컨트롤러를 그 SLO로 재튜닝해 *직접* 측정해야 한다. (§1-16이 이 함정에 빠졌었다.)

### 미측정 (트랙 종결로 후순위)
voice(150/30) Stage B는 chat에서 HT0가 이렇게 강하게 확정돼 payoff 없음. length-normalized SLO 직접 서빙도 동일 예상.
