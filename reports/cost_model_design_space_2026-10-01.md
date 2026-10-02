# Cost model 설계 공간 — 설계 가능 영역 · 선행 cost model이 hybrid에서 깨질 것으로 예상되는 지점 · 알고리즘 (2026-10-01)

> **지위**: 설계 기준선(`reports/system_design_anchor_2026-09-30.md`)의 부속 문서. **측정 아님** — GPU 0 · 새 성능 판정 0 ·
> Claim 등급 변경 0. "예상"으로 표시한 항목은 전부 미검증 가설이며, 각 항목에 그것을 검증할 카드를 붙였다. 기존 KISTI 수치 중
> 인용 금지 열(r0c knee 절대값·attn 비중·HE2 수치)은 **방향만** 차용한다[CS-OK: 금지 사실을 적기 위한 언급].
> 시각화: artifact "Hybrid PD-mux Anchor" §9–§12. 정의의 사각지대: `reports/definition_blind_spots_2026-10-01.md`.

## 0. 왜 cost model이 이 설계의 중심인가

설계 기준선의 주장 1("generic 예측기가 hybrid에서 다른 결정을 낸다")은 **cost model의 형식 차이**에서만 나올 수 있다. 규칙은 하나로
고정했고 estimator는 프로파일 `points`만 읽으므로, hybrid성이 들어올 자리는 (i) `points`를 만드는 cost model과 (ii) 그 model이 요구하는
입력 변수(ctx 조성, 층 조성, SM, batch)뿐이다. 따라서 "어디에 cost model을 둘 수 있고, 선행 model이 어디서 틀리는가"가 곧 novelty의
좌표다.

## 1. cost model을 설계할 수 있는 영역 (CM-1 … CM-7)

| # | 영역 | 출력 | 입력 변수 | 누가 정하는가 (M 모델 · U 요청 · E 엔진) | hybrid 항(선행과 다른 곳) | 데이터 출처 | 식별 조건 |
|---|---|---|---|---|---|---|---|
| CM-1 | **decode step 시간** | `T_dec(SM, bs, ctx 조성)` | SM, batch, ctx 분포(p50/p95/max), 층 조성 | M: 층 조성·KV head·state 크기 · U: 각 요청 ctx · E: SM, bs, 배치 구성 ⇒ 배치 ctx 분포 = U×E | attention 층 ∝ ctx·bs, SSM 층 ≈ bs(ctx 무관), **SM 스케일링이 커널 계열마다 다를 수 있음**(가설 — ★2026-10-01 정정: "SSM은 SM 둔감"은 정본 §1-21·C2가 반증, r0c decode Diff B는 수정 전 계측이라 근거 불가. 3-a에서 새로 측정, `reports/prefill_attn_ssm_length_review_2026-10-01.md` §2.3) | 3-a 프로파일(SM × batch × ctx, cudagraph-ON) | 조성이 다른 모델 ≥2–3개 hold-out(단일 모델은 재매개화) |
| CM-2 | **prefill span 시간** | `T_span(layers, L_miss, SM)` | span의 층 집합, 미스 토큰 수, SM | M: 층 조성 · U: 입력 길이 L(·prefix 구조) · E: span budget, SM, radix | attention 층은 L에 초선형, SSM 층은 L에 선형 → span별 비용이 층 조성에 따라 불균등 | A1 span-SM 곡선(긴 입력 한정, D-6) | 짧은 입력은 span 1개라 도메인 공집합(§5 메모) |
| CM-3 | **경계·전환 비용** | `C_switch(from, to)`, `C_span_boundary` | 분할 인덱스 쌍, 스트림 드레인, PAUSE 진입/해제 | E 전용(분할 인덱스, 드레인, PAUSE) · M: SSM 경계 state 여부[미확인] | SSM 층 경계에서 state 실체화가 붙는지 여부(미확인) · ★(2026-10-01) 예산 초과 영역에서 토큰 축 분할 시 **chunk 경계 SSM state 이월** 비용(attention 전용 모델에 없는 항) | P12 전환 비용·bubble 측정 | 직접(드레인)과 간접(bubble) 분리 |
| CM-4 | **동거 간섭** | decode slowdown `δ(p, d, prefill 종류)` | 분할, 동시 prefill 층 종류, 메모리 대역 | E: 분할 · U×E: 동시 prefill의 길이 · M: 커널 종류 | SSM scan과 attention prefill의 대역 점유가 다름 → 같은 SM 분할에서 간섭이 층 종류에 의존 | 4-a(동거 vs SM 수 분리) | prefill 있음/없음 대조를 같은 job 안에 |
| CM-5 | **메모리·admission 예산** | 수용 가능 요청 수 `N_adm(ctx 분포)` | KV 토큰(attention 층만), mamba 슬롯(고정), radix 여부 | M: KV/슬롯 크기 · U: ctx 길이 분포(KV) · E: 요청 수(슬롯), max_running, radix | KV가 ctx에 비례하는 것은 attention 층뿐 → 긴 ctx에서도 슬롯이 먼저 묶임 | 엔진 상수(`max_mamba_cache_size`) + 실측 점유 | radix OFF에서는 슬롯 = running 상한(항등, M0) |
| CM-6 | **대기열·TTFT** | `TTFT = W_queue + S_prefill + 잔차` | 도착 과정, admission 상한, CM-2 | U: 도착 과정·길이 · E: admission 상한·순서 · CM-2 경유 M | S_prefill이 CM-2를 통해 층 조성 의존 | TTFT 분해 규약(서버 stamp) | 정상성 검사 3종 통과 셀만 |
| CM-7 | **prefix cache**(radix ON, 이월) | 히트 확률·재계산 토큰 | 분기 구조, SSM state 보유 노드 | U: prefix 공유 구조 · E: radix·축출 · M: SSM state 보유 조건 | 노드에 SSM state가 있어야 히트 → 분기 패턴에서 히트율 저하 예상 | trace plan §8.1 | radix ON 확장 뒤에만 |

CM-1이 주장 1의 직접 입력이고, CM-5·CM-6이 정상성·admission 규약을, CM-3·CM-4가 pause manager와 5-a 격차 해석을 받친다.

### 1.0 "조성"의 세 층 (2026-10-01 추가, 사용자 지시)

"hybrid 조성"이라 뭉뚱그려 부르던 것은 결정 주체가 다른 세 층이다.

| 층 | 요소 | 런타임에 바뀌나 | 정책 비교에서의 지위 |
|---|---|---|---|
| **M 모델(배포 시 고정)** | attention/SSM 층 수, KV head·head 차원, SSM state 크기, 은닉·MLP 차원 | 아니오(상수) | 단일 모델 안에서는 변수가 될 수 없음 — 귀속은 A11 hold-out만(K2) |
| **U 요청(사용자 쪽)** | 입력 길이 L, 출력 길이, 도착 시각·도착률, prefix 공유 구조, SLO 등급 | 예(요청마다) | 워크로드 축 — 엔진이 고를 수 없음, trace가 정함 |
| **E 엔진(런타임 결정)** | admission·순서(→ bs와 배치 구성), `max_running_requests`·mamba 슬롯·KV 풀, SM 분할·PAUSE, span budget, radix·backend·cudagraph | 예(정책) | 설계 대상 |

**배치 ctx 분포 = U × E.** 요청 길이는 사용자가, 어느 요청이 한 배치에 함께 있는지는 엔진이 정한다. M은 비용을 이렇게 나눈다:
attention 층 비용 ∝ Σctx(U·E 의존), SSM 층 비용 ∝ bs(E만 의존), KV ∝ ctx(attention 층만, U 쪽), mamba 슬롯 ∝ 요청 수(E 쪽).
⇒ hybrid가 바꾸는 것은 **decode 비용에서 U 성분의 비중**이다(SSM 비중↑ → 요청 길이가 decode 비용을 덜 움직이고 bs가 지배).
transformer 기준 예측기가 hybrid에서 틀릴 수 있는 순간은 **U만 변하고 E(bs)는 그대로인 순간**이다 — 정상 shape 하나로는 만들 수 없고
W5(1K↔8K context swing) 같은 trace가 필요하다.

**기존 실험과의 대응(방향만)**: M을 바꿨을 때 결정은 움직이지 않았다(P1.7 4모델 전부 agnostic 최적, 8B decode SM 민감도 4 arm
모델-무관[레버 존재만], layer-type 런타임 정책 全형태 死). U를 바꿨을 때 체제가 바뀌었다(λ0 shape A 입력 256 vs B 8192에서 prefill당
span 수·동거 비율·결정 지점 유무가 완전히 다름 — 미등록 재집계라 수치 인용 금지). ⇒ 이 기판에서 서빙 결정을 움직인 것은 U였고 M은 계수만
바꿨다. "hybrid 전용 정책"이 하중을 받지 못하는 이유이자, 타겟을 긴 문서 요약(U 축)으로 잡은 근거.

**함의**: P_gen vs P_hyb 차이는 U 축(ctx 분포 변화)에서 찾는다 · "조성 때문" 귀속은 M 축이라 A11 hold-out만 · E 레버 중 hybrid가 의미를
바꾸는 것은 메모리 단위(X-6 mamba 슬롯)와 PAUSE이며 둘 다 U의 길이 분포에 따라 이득 크기가 달라진다.

### 1.1 CM-1의 형식(제안, 3-a 적합 대상)

```
T_dec(SM, bs, ctx) = s_attn(SM) · Σ_{l∈attn} [ a0 + a1·bs + a2·bs·ctx_eff ]      # KV 읽기 ∝ bs·ctx
                   + s_ssm(SM)  · Σ_{l∈ssm}  [ m0 + m1·bs ]                     # state 갱신 ∝ bs, ctx 무관
                   + s_mlp(SM)  · [ f0 + f1·bs ]
                   + c_launch(SM, cudagraph)
ctx_eff = 배치 ctx의 합/평균 중 등록된 통계량 (p95 사용 시 보수적)
s_*(SM): 커널 계열별 SM 스케일링 곡선 — 등식이 아니라 실측 점(16/24/34/44/108)에서 보간
```
P_gen은 위에서 `s_ssm ≡ s_attn`, `m1·bs`를 `a2·bs·ctx`로 흡수(전 층 attention 가정)한 형식이고, P_hyb는 두 계열을 분리한다.
차이는 `ctx`가 길고 SSM 비중이 클수록, 그리고 SM이 작을수록 커진다(예상).

## 2. 선행 cost model이 hybrid에서 적용되지 않을 것으로 예상되는 지점 (전부 미검증)

| # | 선행 model · 가정 | hybrid에서 깨지는 이유(예상) | 예상 오류 방향 | 검증 카드 |
|---|---|---|---|---|
| X-1 | **Bullet 지연 예측기** — 상태 `(sl, pbs, pm, cl, dbs, dm)`에서 decode 지연이 decode ctx `cl`에 단조 증가(전 층 attention) | attention 층만 `cl`에 비례, SSM 층은 상수 → 긴 ctx에서 decode 지연을 **과대 예측** | `ReduceDecodeSM`이 decode SM을 필요 이상 남김 → prefill이 덜 받음 → TTFT ↑ (주장 1의 기제) | 3-a → S0′ (P_gen vs P_hyb 발산) → S1 |
| X-2 | **Bullet 두 점 보정 + online 재보정** — 오프라인 두 점으로 기울기, 온라인으로 절편 보정 | 재보정은 절편·스케일을 고치지만 **ctx 기울기의 형식 오류**는 못 고친다(ctx 분포가 바뀔 때마다 다시 틀림) | 정상 ctx에서는 맞고 ctx 분포 전환 직후 틀림 → 변화 trace에서만 드러남 | S0′ P_gen+recal arm, 5-b 변화 trace(W5 context swing) |
| X-3 | **Bullet 산술 강도 배치 규칙** — `ArithInten(batch) < peak`까지 prefill 배치 | SSM scan의 FLOP/byte 프로파일이 attention·MLP와 다름 → 같은 토큰 수에서 강도가 다름 | prefill 배치 크기를 잘못 잡음(방향 미정) | A1 span-SM 곡선에서 포화점 실측 |
| X-4 | **MuxWise `N_PL = ⌈T_d·N_T / T_P⌉`** — 층당 prefill 시간 `T_P` 균일 가정 | 층 비용이 층 종류에 따라 불균등(attention ≫ SSM, 긴 L에서 더) → "몇 층"이 시간의 단위가 아님 | 층 수로 맞춘 prefill 발사가 decode 1 iteration을 덮지 못하거나 초과 | CM-2 적합 후 비용 가중 층 수로 재정의 |
| X-5 | **MuxWise contention-tolerant estimator** — transformer 커널 간섭으로 보정 | SSM scan의 대역 점유가 달라 같은 분할에서 간섭 계수가 다름(예상) | 동거 시 decode slowdown 오예측 | 4-a(CM-4) |
| X-6 | **KV 토큰 예산 admission**(vLLM/Sarathi/DistServe 계열) — 메모리 ∝ Σ ctx | attention 층만 ctx 비례, mamba state는 요청당 고정 → 긴 ctx에서 KV보다 **슬롯이 먼저** 묶임 | "긴 ctx에서 KV pool이 먼저 cap"이라는 기존 예측(trace plan)이 hybrid에서 약화될 수 있음 | CM-5: 실측 점유 vs 슬롯 상한 병기 |
| X-7 | **"decode는 memory-bound라 SM에 둔감"** 전제(ReduceDecodeSM의 근거) | hybrid decode의 SM 민감도가 관측됨(8B 4 arm, **pure Mamba2 포함 모델-무관**, 레버 존재만·정책 이득 아님·범위 한정) — ★2026-10-01 정정: "attention만 SM 민감"이 아니라 SSM 포함 전부 민감(§1-21·C2); hybrid 고유 차이는 계열별 SM 스케일링 **곡선의 모양**이 다른지이며 미측정 | decode SM을 너무 줄여 ITL SLO 위반, 또는 반대로 SSM 지배 배치에서 너무 남김 | 3-a s_attn/s_ssm 분리 곡선 |
| X-8 | **chunked-prefill 선형 비용**(Sarathi) — chunk 비용 ∝ 토큰 | PD-mux는 chunked 대신 층 경계 span; SSM 층 경계에서 state 실체화·재진입 비용 가능 | span을 늘릴수록 경계 비용 누적 → TTFT ↑(방향만) | D-6 span 입도 스윕 + P12 |
| X-9 | **radix prefix 히트 모델** — 어떤 prefix 노드든 재사용 | MambaRadixCache는 노드에 SSM state가 있어야 히트 → 분기 패턴 히트율 저하 | 캐시 ON 기대 이득 과대 | radix ON 확장(CM-7) |
| X-10 | **선점/재계산 비용**(KV 축출 후 부분 재사용) | SSM state는 부분 재사용 불가 → retract 시 전량 re-prefill; `wait_queue_entry` 덮어써 W_queue 오염 | retract 비용 과소평가; PAUSE(축출 없음)가 대안인 이유 | pause manager, TTFT 분해의 retract 제외 규칙 |
| X-11 | **HE0 계열 반응형 제어의 cost 무관 가정** | 우리 자신의 선행 — 정책이 cost model 없이 slack에만 반응 → positioning 실패 | 정본 확정(HE0), 재논쟁 아님 | 규칙을 예측형으로 고정한 이유 |

X-1·X-4·X-6·X-7이 구조적(형식) 차이, X-2·X-3·X-5·X-8이 계수·포화점 차이, X-9·X-10이 캐시·선점 의미론 차이다. 논문에서 "자명하지
않다"를 뒷받침하는 것은 **구조적 차이 4건**이며, 나머지는 있으면 강화, 없으면 스코프 축소다.

★**정정(2026-10-01, 원문 정독 `reports/prior_cost_models_bullet_muxwise_2026-10-01.md` §4)**: 위 X-1의 "선행 model이 decode 지연을 ctx에
단조 증가로 가정"은 거칠다. MuxWise 식 2 `T_dec = θ1·Σr_i + θ2·bs + θ3`는 ctx 비례 항과 ctx 무관 항이 이미 분리돼 있어 **단일 모델·고정 SM에서
재적합하면 형식이 깨지지 않고**, Bullet SRM은 커널 무관 roofline이라 SSM 커널의 flop/mem을 열거하면 그대로 쓸 수 있다. 실제로 깨지는 것은
형식이 아니라 (a) θ 비율의 **SM 의존성**(커널 계열별 스케일링, X-7의 정확한 형태) (b) 층 조성이 다른 모델로의 **이식**(prefill `n²` 항, 커널 열거)
(c) `N_PL`의 층 균일 가정(X-4) (d) 간섭 대리변수(reused 토큰·memcpy+UG GEMM)의 SSM 부적합(X-5) (e) mamba 슬롯이라는 **변수 부재**(X-6)다.
⇒ P_gen의 정의를 **"MuxWise 식 2 형식을 hybrid 모델에 재적합 + Bullet식 온라인 α 보정"**으로 강화하고, P_hyb와의 차이는 SM 축·배치 ctx 조성
축에서만 기대한다. 단일 SM·단일 조성에서 차이가 0에 가까우면 정직한 결론은 "예측기 수준의 hybrid 확장은 자명하다"이며 novelty는 (c)(d)(e)와
pause manager 쪽에서 찾는다. 구조적 차이 목록도 X-1(형식) → **X-7(SM 의존)·X-4·X-6·X-5(대리변수)**로 바꾼다.

## 3. 알고리즘 (의사코드 — 사전등록 전 초안, 상수는 전부 등록 대상)

### A1. P_hyb: 프로파일 `points` 생성 (3-a 실측 → CM-1 적합 → 격자 합성)
```
input : measured points {(SM, bs, ctx) -> itl p50/p95/p99}, layer composition (n_attn, n_ssm)
fit   : (a0,a1,a2), (m0,m1), (f0,f1), s_attn(SM), s_ssm(SM), s_mlp(SM)  by least squares on p50
        residual_p95 = p95 of |pred - measured| on HELD-OUT points (not fitted ones)
emit  : for SM in {16,24,34,44,108}, bs in grid, ctx in grid:
            DecodeLatencyPoint(SM, bs, ctx, p50=T_dec, p95=T_dec+residual_p95, p99=..., repeats, measured_steps)
        heldout_residual_p95_ms = residual_p95      # margin = max(0.1*SLO, residual)
check : every allowed SM level has a full bs×ctx grid (missing -> estimator silently skips: forbidden)
```

### A2. P_gen: 두 점 보정(전 층 attention 가정)
```
input : two measured points per SM at (bs_ref, ctx_lo), (bs_ref, ctx_hi)
slope : k(SM) = (T_hi - T_lo) / (ctx_hi - ctx_lo)          # applied to ALL layers
model : T_gen(SM, bs, ctx) = T_lo(SM) * (bs/bs_ref) + k(SM) * (ctx - ctx_lo) * (bs/bs_ref)
emit  : same grid as A1 from T_gen; residual from the same held-out set
```

### A3. P_gen+recal: online 재보정 (Bullet 방식 모사 — 충실한 baseline)
```
state : scale e = 1.0 (EMA)
each evaluation epoch:
    observed = measured_itl_p95_ms (from snapshot);  predicted = T_gen(current SM, bs, ctx_p95)
    e <- (1-α)·e + α·(observed / predicted)           # α registered (e.g. 0.2)
    T_used(SM, bs, ctx) = e · T_gen(SM, bs, ctx)      # corrects level, NOT the ctx-slope form
```

### A4. 압력 검출과 규칙 (Bullet Algorithm-1 decode 측, 분기 순서 고정)
```
pressure := prefill_queue_depth > 0 and oldest_prefill_age_ms > τ_age
if not pressure or decode_running_batch_size == 0: return NOOP           # 2F9: decode-only is not a decision point
d_min := argmin{ d in grid : predict(d, bs, ctx_p95).p95 + margin <= SLO }  # via estimator
if d_min is None:                     return NOOP                          # no feasible SM: do not pause blindly
if d_min < current:                   return SQUEEZE(d_min)
if d_min == grid.min and pressure persists for ≥ k epochs
   and predicted_tpot_after_pause(n_max) <= SLO:                          # TPOT bound incl. pause gap
                                      return PAUSE(n = n_max)
return NOOP
```
분기 순서(SQUEEZE 먼저)·τ_age·k·n_max·"predicted_tpot_after_pause"의 정의는 사전등록 상수다(S0 rev1 死因 K3).

### A5. Pause manager (1급 상태, 불변식 4)
```
enter(n):
    require split_prefill_batch is not None and not wait_prefill_kernel_done       # (i)
    require not r2_admission_limited                                              # (iii) exclusivity
    return_idx <- current stream idx; drain(prefill, decode); select idx0 (108,0)
    paused_left <- min(n, n_max)                                                  # (ii)
each loop iteration while paused_left > 0:
    SKIP update_running_batch AND decode submit together (KV slots unchanged)
    exclude this iteration from ITL samples; record per-request gap separately
    paused_left -= 1
    if prefill_final_submitted or emergency_occupancy(): paused_left <- 0         # (iv), safety
exit():
    drain(prefill); select return_idx; resume ITL sampling; emit telemetry {paused_steps, reason}
assert never (paused_left > 0 and r2_admission_limited)
```

### A6. F1 재정렬 (Bullet `SortByLeastEstimLatency` 유사, 기본 OFF)
```
before get_new_batch_prefill():
    for r in waiting_queue: est[r] = T_span_total(L_miss(r), layers=all, SM=prefill_sm)   # CM-2
    order = sort(waiting_queue, key=est)                                                   # shortest first
    enforce TTFT bound: any r with age(r) + est_wait(r) > SLO_TTFT is moved to front (FCFS among them)
    waiting_queue <- order          # calc_priority is a no-op under fcfs; priority scheduling must be OFF
hybrid variant (radix ON only): L_miss(r) counts tokens to recompute from the nearest node WITH SSM state
```

### A7. admission 예산 (CM-5)
```
can_admit(r):
    kv_ok   = kv_tokens_free >= Σ_{l∈attn} ctx(r)               # attention layers only
    slot_ok = mamba_slots_free >= 1                              # fixed-size state per request
    run_ok  = running_bs < max_running_requests
    return kv_ok and slot_ok and run_ok
note: with disable_radix_cache and explicit max_running_requests, slot cap == run cap (identity, M0)
```

### A8. span 크기 (CM-2 기반, D-6)
```
given prefill batch tokens N and budget B: spans = ceil(n_layers / max(1, B // N))       # engine formula
cost-aware variant (proposal): choose B so that Σ_span [T_span + C_boundary] is minimized subject to
    max_span T_span <= T_dec_step(current d)     # MuxWise coverage condition, cost-weighted layers (X-4)
```

### A9. TTFT 분해 + 정상성 검사 (분석기)
```
per request (server stamps): W_queue = forward_entry - wait_queue_entry; S_prefill = prefill_finished - forward_entry
exclude requests with retract_time > 0 (count them)                                      # X-10
per cell×arm×rep: S1 = F-E ratio (thirds, rep-mean, 1.7); S2 = gate#12 first/second-half violation gap;
                  S3 = trend of prefill_queue_depth over benchmark phase (rank-based)
label = UNSTABLE if any fires else STATIONARY;  magnitudes reported only for STATIONARY cells
```

### A10. S0′ 결정 발산 검사 (CPU, 예측기만 변경)
```
fix rule A4 with registered constants; build predictors P_gen, P_gen+recal, P_hyb from the SAME 3-a points
stream = replay of runtime_snapshots (open loop, 3rd-party fixed policy trace) filtered by decision-point predicate
for each snapshot: label_P = A4(P)  for P in {gen, gen+recal, hyb};  d_min_P likewise
D(hyb vs gen+recal) = fraction of decision points with label differing;  M = fraction with same label, different d_min
mutations: M3 identical predictors -> D=0;  M5 comparator flipped -> must fail;
           M1' regenerate P_hyb points with n_attn = n_layers (all attention) -> must collapse to P_gen (identity check)
labels: DIVERGES / RANK_INVARIANT / NO_DIVERGENCE / MODEL_DEPENDENT with thresholds fixed BEFORE running
```

### A11. 조성 효과의 식별 (hold-out)
```
fit CM-1 on models M1..Mk (different n_attn/n_ssm); predict a held-out model M*'s 3-a points
report prediction error of P_hyb vs P_gen on M*;  claim "composition matters" only if P_hyb error < P_gen error on M*
single-model fits are reparameterizations (not evidence)
```

## 4. 식별 가능성과 함정

- **단일 모델에서는 조성 효과를 식별할 수 없다**(조성은 모델 상수). 설계 메모 §3 A1/B1의 감사 지적 그대로 — 모델 hold-out(A11)이
  주장 1의 "hybrid 때문"을 지탱하는 유일한 경로다.
- **estimator는 `points`만 읽는다.** P_hyb/P_gen의 차이는 반드시 `points` 재생성으로 만들고, 메타필드 조작은 항등식이다.
- **격자 결손은 조용히 실패한다**(허용 SM 수준에 점이 없으면 estimator가 건너뜀; 108 점이 없으면 긴급 추정이 `upper=inf`). A1의
  격자 완전성 검사가 필수다.
- **기판**: 모든 계수는 VESSL A100 값이며 KISTI 값과 섞지 않는다. 형식(어떤 항이 있는가)만 기판 간에 이식 가능하다고 가정하며, 그것도
  3-a 재적합으로 확인한다.
- **정상성**: CM-6의 TTFT 크기는 정상성 통과 셀에서만 인용한다.

## 5. 알고리즘 배치 지도 — 실행 위치 · 설명 대상 · 차별점

실행 흐름은 다섯 층이다: **L0 오프라인(CPU, 배포 전)** · **L1 컨트롤러 epoch(케이던스 평가)** · **L2 루프 iteration(매 decode step ∥ prefill span)**
· **L3 admission 지점(`get_new_batch_prefill` 직전)** · **L4 사후 분석(CPU, 런 후)**. 차별점 열의 ★는 선행(Bullet/MuxWise/SGLang upstream/
vLLM·Sarathi)에 없는 요소, ☆는 선행에 있으나 정의를 바꾼 요소, —는 선행과 같은 요소다.

| 알고리즘 | 실행 위치 | 무엇을 설명·결정하나 | 선행과의 관계 | 차별점 |
|---|---|---|---|---|
| A1 P_hyb points 생성 | L0 (3-a 후) | decode step 시간의 층 조성·SM 의존 분해 → estimator 입력 | Bullet SRM은 커널 roofline, MuxWise 식 2는 회귀. 둘 다 층 계열별 SM 스케일링 `s_attn≠s_ssm`을 두지 않음 | ★ 커널 계열별 SM 스케일링 곡선 + hold-out 잔차를 margin으로 |
| A2 P_gen 두 점 보정 | L0 | 전 층 attention 가정의 ctx 기울기 (generic baseline) | Bullet Fig. 9 2표본 보정과 MuxWise 식 2 형식 | — (선행 재현, baseline) |
| A3 P_gen+recal | L1 | 수준 오차의 온라인 보정 (충실한 baseline) | Bullet 식 2 α 온라인 보정 | — (선행 재현; 없으면 허수아비) |
| A4 압력 검출·규칙 | L1 | SQUEEZE/PAUSE/NOOP 결정, 분기 순서 고정 | Bullet Alg. 1의 decode 측(최소 SM, 극단 부하 일시중단) | ☆ 분기 순서·τ_age·"1 step"을 등록 상수로 고정(S0 rev1 K3), 2F9 결정점 술어 |
| A5 pause manager | L2 | PAUSE의 집행: 쌍 건너뜀·idx0 전환·자체 시계·latch 배타 | Bullet은 별도 프로세스라 decode 정지가 프로세스 경계에서 자유로움; MuxWise 선점은 prefill 측 | ★ 단일 프로세스·green-context 하드 분할·shared running batch 아래에서의 decode 정지 계약(불변식 4). 선행에 대응물 없음 |
| A6 F1 재정렬 | L3 | admission 순서 (예상 지연 오름차순, TTFT 한도) | Bullet `SortByLeastEstimLatency`; SGLang upstream은 fcfs/lof, radix OFF에서 lpm→fcfs | — (Bullet 재현) · radix ON 변형은 ★ SSM state 보유 노드 기준 L_miss |
| A7 admission 예산 | L3 | 수용 가능 요청 수 (KV ∧ mamba 슬롯 ∧ running) | vLLM/Sarathi/DistServe는 KV 토큰 예산; Bullet/MuxWise는 KV pool 공유만 | ★ mamba 슬롯 예산(단, radix OFF에서는 running 상한과 항등 — M0) |
| A8 span 크기 | L2 (prefill 배치 시작 시) | span 수·경계 비용의 균형, decode 1 iteration 덮기 조건 | MuxWise `N_PL = ⌈T_d·N_T/T_P⌉`(층 균일), 우리 엔진 `ceil(n_layers / (B // N))` | ☆ 비용 가중 층 수(층 종류별 T_span)로 덮기 조건 재정의(X-4) |
| A9 TTFT 분해·정상성 | L4 | W_queue/S_prefill 분해, UNSTABLE 라벨 | 선행은 TTFT/TPOT P90만 보고; 정상성·분해 규약 없음 | ★ retract 제외 규칙, 3종 정상성 검사, 크기 인용 조건 |
| A10 S0′ 발산 검사 | L0/L4 (CPU) | 예측기 형식만으로 결정이 갈리는가 | 선행에 없음(자기 예측기의 ablation만) | ★ 규칙 고정 + 예측기만 변경 + 변이 검사(M3/M5/M1′) |
| A11 hold-out 식별 | L0 | "조성 때문"의 식별 | 선행은 단일 모델 계열(Llama)에서만 검증 | ★ 조성이 다른 모델 hold-out |

읽는 법: novelty 후보는 ★ 행(A1·A5·A7·A9·A10·A11)이며, 그중 **주장 1**을 직접 받치는 것은 A1(형식)과 A10(검사)이고, **엔진 기여**는
A5, **방법론 기여**는 A9·A10·A11이다. A2·A3·A6은 선행 재현이라 baseline 자격이고 논문에서 새 것으로 쓰지 않는다. A4·A8은 선행 규칙의
재정의(☆)라 "다르다"가 아니라 "고정했다/가중했다"로 서술한다.
