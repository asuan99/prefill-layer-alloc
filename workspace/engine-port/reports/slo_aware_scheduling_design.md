# SLO-aware dynamic PD-mux scheduling — 실험 설계 + 정적 베이스라인 (step C)

작성: 2026-07-09. 사용자 통찰(2026-07-09): 현 pdmux는 SM split을 **decode 배치크기(SLO-blind)**로만 조정하고,
decode SLO(TPOT) 긴급성에 반응하는 스케줄링이 **없다**. 정적 split은 TTFT↔TPOT 트레이드의 **한 점**만 잡는다.
→ **SLO-aware 동적 스케줄러**가 binding SLO 쪽으로 SM을 밀면 정적을 이길 수 있는가? (layer-aware track과 직교; 닫힌 track 이후 첫 유망 방향.)

정본 track 문서. 빌드 순서 = **C(이 문서·베이스라인) → A(메커니즘) → B(mixed 워크로드·최종 비교)**.

---

## 1. 가설

- **H1 (stationary)**: 단일 SLO-aware 정책이, regime을 *모른 채로*, 각 regime의 **최적 정적 split에 수렴**한다 (in3600→d24, in2000→d44). 즉 tuned-uniform을 **regime별 수동 튜닝 없이** 재현.
- **H2 (mixed)**: 가변 in-len·burst 도착에선 정적 split이 평균에 튜닝돼 suboptimal → **SLO-aware 동적이 정적을 이긴다**(주 이득 지점).
- **H0 (귀무)**: 동적이 정적을 못 이긴다(정상서도 mixed서도) → 이 track도 종료.

## 2. 메커니즘 (step A에서 구현할 것)

현 `adjust_stream_groups`(decode_bs 기반)를 **SLO-slack 기반**으로 교체/증강:
- **신호**: (i) decode min TPOT slack = TPOT_SLO − max(decode 배치의 마지막토큰後 경과); (ii) prefill TTFT slack = TTFT_SLO − 최장 대기 prefill의 큐 시간.
- **제어(양방향 dynamic split)**:
  - decode slack ↓ (TPOT 위험) → stream_idx를 **decode-heavy**로 밀기(+ 극단시 그 스텝 prefill skip = decode 우선).
  - prefill slack ↓ (TTFT 위험) → stream_idx를 **prefill-heavy**로.
  - 둘 다 여유 → throughput 우선(prefill-heavy 기본).
- **knobs**: SLO(TPOT 60ms·TTFT 3s), slack 임계(예: TPOT의 1.5×step), stream_idx step, hysteresis(진동 방지).
- env-gated(`PDMUX_SLO_SCHED=1`), 기존 경로 불변. `req.time_stats`(토큰 타이밍 인프라 존재) 활용.

## 3. 정적 베이스라인 = 이겨야 할 목표 (R0c, clean async, m-config)

**goodput@SLO (req/s), TTFT≤3s ∧ TPOT≤60ms:**

### in3600/o32 (prefill-bound) — 최적 정적 = **d24**(decode24/prefill84)
| rate | agnostic(auto) | **d24** | d16 | fused |
|---|---|---|---|---|
| 1 | 1.19 | 1.20 | 1.09 | 0.79 |
| 2 | 2.06 | **2.24** | 1.87 | 0.27 |
| 3 | 0.55 | **1.18** | 1.03 | 0.00 |
| 4 | 0.31 | **0.47** | 0.41 | 0.00 |

### in2000/o96 (decode-heavy) — 최적 정적 = **d44**(decode44/prefill64)
| rate | agnostic | **d44** |
|---|---|---|
| 2 | 2.25 | 2.24 |
| 3 | 3.24 | 3.22 |
| 4 | 1.59 | **2.27** |

★**핵심 관찰**: 최적 정적 split이 **regime마다 다르다**(in3600=d24 / in2000=d44). 정적은 하나를 골라야 하므로 다른 regime선 suboptimal. **동적 SLO-aware가 둘 다 자동으로 잡으면 H1 성립.**

## 4. 워크로드

- **Stationary (기존 재사용)**: in3600/o32(prefill-bound), in2000/o96(decode-heavy). rates 1–6. → H1 검증(동적이 d24·d44 각각 재현하나).
- **Mixed (step B 신규)**: in-len을 분포(예: 512–4096 혼합)·출력 32–96 혼합·**burst 도착**(Poisson + burst). 재현가능 seed. → H2 검증(동적이 정적 이기나).

## 5. 성공 판정

- **H1**: 동적 goodput ≥ max(정적들) − ε, in3600·in2000 **양쪽 동시**(단일 정책·무튜닝). → "regime-adaptive 재현" 확인.
- **H2**: mixed서 동적 goodput > 최선 정적(단일 고정 split, 최적 튜닝된 것). → track 성공.
- 실패(H0): 어느 쪽도 못 이김 → 종료(정직).

## 6. A/B 스코프

- **A**: `event_loop_pdmux`(또는 agnostic 경로)에 SLO-slack 신호 + dynamic stream_idx 제어 추가(env-gated). 정확성 게이트(coherent) + stationary에서 H1 측정.
- **B**: mixed 워크로드 하네스(bench_serving 확장 또는 커스텀 arrival) + 정적(d24/d44/agn) vs 동적 최종 비교(H2).

## 7. caveat (정직)

- 정상 regime선 정적이 이미 near-optimal → 동적의 이득은 **주로 mixed/burst**. stationary는 H1(수렴·무튜닝 재현)이 목표지 "큰 이득"이 아님.
- 진동/오버헤드 리스크: dynamic 전환은 partition switch(sync) 비용 → step보다 잘게 바꾸면 (D)류 비용. **step 단위 전환**으로 제한(층 아님)해 (D) 회피.
- 이건 layer-aware가 아님: 층타입 신호가 아니라 **SLO 피드백 신호**로 **step-level split**을 동적 조정 = (D)와 무관.

---

## Step D — context-length **feedforward** split (2026-07-14)

동기(사용자 통찰 2026-07-13/14): prefill-side layer-aware(층타입으로 SM 재배분)는 (B,L) knee로 死(Diff B≈1,
`results/prefill_knee/next_steps.md`). 그러나 attn-prefill 비용은 **O(L²)**라 per-layer 시간이 L에 극단 의존
(L2k 1.0ms → L32k 108.6ms, 실측). 현 컨트롤러(A~C)와 agnostic은 split 신호가 **전부 feedback**(TPOT-EMA·큐깊이)
이고 **L을 직접 안 본다** → 긴 요청이 와도 latency가 망가진 *뒤에야* 반응(lag). agnostic은 이 때문에
prefill-starve로 TTFT 3025ms(실측). **L은 admission 시점에 알려진 feedforward 신호** → 긴 context 요청에
prefill SM을 **선제적으로** 키워 TTFT tail을 줄이면 SLO를 더 잘 맞출 수 있는가?

**이건 layer-aware 부활이 아님**: 층 단위 sub-step 재분할(=PF, 死)이 아니라, **L은 요청당 상수**라 **step-level
cross-phase split을 L의 함수로** 정하는 것(cudagraph-safe, (D) 무관). SLO-aware controller에 **feedforward 항 추가**.

### D.1 가설
- **HD1 (heterogeneous-L)**: 길이 이질 워크로드(짧은 요청 다수 + 긴 요청 소수 혼재)서 **SLO+LFF > SLO-v7b**(feedback-only)
  및 agnostic — goodput@SLO 그리고 특히 **긴-요청 TTFT tail**(p50/p99).
- **HD-iso (homogeneous-L)**: 단일 고정 L에선 LFF가 plain SLO로 **degenerate**(무해, no harm). isolation 게이트.
- **HD0 (귀무)**: heterogeneous서도 SLO-v7b 대비 무이득 → 큐 feedback이 이미 L을 잡음 → **L-feedforward 신호는 잉여, sub-track 종료.**

### D.2 메커니즘 (`_slo_decide_idx`에 env-gated 추가, `PDMUX_SLO_LFF`)
- **신호 `_Lsig`**: imminent prefill의 대표 context length(tok) = max(현 `split_prefill_batch.seq_lens_cpu_cache`, `waiting_queue` 프롬프트 길이).
- **feedforward center**: `ff_center = clamp( (lo+hi)/2 − gain·(_Lsig/L_ref − 1), lo, hi )`. 긴 L(>L_ref) → center를 **prefill-heavy(낮은 idx)**로, 짧으면 decode-heavy(높은 idx)로. `L_ref`=`PDMUX_L_REF_TOK`(기본 3600).
- **feedback와 합성**: TPOT-EMA 긴급(+1 decode)·큐backlog+TPOT여유(−1 prefill)는 그대로; **deadband(hold)에서 ff_center로 1-step drift**. 즉 긴급은 feedback, 정상 resting point는 feedforward가 결정.
- off(`PDMUX_SLO_LFF` 미설정)면 ff_center=(lo+hi)//2·deadband=hold = **기존 v7b와 byte-identical**.
- **knobs**: `PDMUX_SLO_LFF`(gain, 예 1~2), `PDMUX_L_REF_TOK`(기본 3600).

### D.3 워크로드
- **Heterogeneous-L (주, 신규)**: 한 서버에 **동시 2스트림** — short-flood(in~512, 고rate) + long-trickle(in~16000, 저rate). bimodal context → 매 순간 prefill 큐에 짧은·긴 요청 혼재 = LFF가 긴 요청 prefill을 선제 provision하되 짧은 요청 decode를 안 굶기는지 시험.
- **Wide-range (부)**: 단일 스트림, `--random-input-len 8000 --random-range-ratio 0.1`(≈800–8000 spread).
- **Homogeneous (isolation)**: 단일 고정 L(in3600) → LFF ≈ plain SLO 확인(HD-iso).

### D.4 비교군 · 지표
- 비교: **agnostic** / **d24(tuned static)** / **SLO-v7b(feedback)** / **SLO+LFF(feedforward)**.
- 지표: (i) overall **goodput@SLO**; (ii) **길이-버킷별 TTFT**(짧은 vs 긴 요청 p50/p99 — 핵심은 긴-요청 tail); (iii) TPOT p50/p99; (iv) **예측가능성**(TTFT std·p99/p50) — 사용자의 "균일성" 논거.

### D.5 성공 판정 (게이트)
- **HD1 성공**: heterogeneous서 SLO+LFF goodput > SLO-v7b **AND** 긴-요청 TTFT tail 감소, 짧은-요청 무손상 → 새 신호축 확인, track 지속.
- **HD-iso**: homogeneous서 SLO+LFF ≈ SLO-v7b(±ε) → 무해 확인(통과 필수).
- **HD0(실패)**: heterogeneous서 SLO-v7b 대비 무이득 → 큐 feedback이 L을 이미 흡수 → **정직하게 sub-track 종료.**

### D.7 ★결과 (2026-07-14, jobs 849105–849629) — Step D net win 아님(HD0-leaning) + 컨트롤러 오설계 발견

워크로드 = 이질-L 동시 2스트림(short-flood + long-trickle). 부하 3회 조정(과부하→경량→중간). **cudagraph-ON에서 부하-보정 후 중간 부하(short4@80·long8k0.6@14), 3-rep 집계:**

| mode | goodput mean[min-max] | SHORT p99[rng] | LONG p99[rng] |
|---|---|---|---|
| **d24 (static)** | **3.160[3.16-3.16]** | **2.31[2.2-2.4]** | **2.01[1.9-2.2]** |
| slo (feedback) | 2.933[2.75-3.06] | 3.79[3.6-4.1] | 3.26[3.0-3.5] |
| lff (feedforward) | 2.854[2.62-3.02] | 2.92[2.3-4.0] | 2.50[2.2-2.9] |

- **HD1 반증(goodput)**: lff goodput ≤ slo, 둘 다 < d24. feedforward가 goodput 개선 못 함.
- **부분 양성(tail)**: **lff tail < slo tail**(short 2.92<3.79·long 2.50<3.26, 3-rep robust) — feedforward가 **예측가능성(TTFT tail) 개선**은 실제로 함(당신 직관의 kernel). 단 goodput 전환 안 되고 소폭 goodput 비용.
- ★**static d24가 두 동적을 결정적·재현성(분산0) 있게 이김.** cudagraph-ON near-boundary regime서 동적이 static에 패.
- **regime 요약**: 경량(rep3)=전원 통과·구별 안 됨; 중간(rep5)=판별점; 과부하(rep2 no-cg·rep4)=전원 포화. dynamic은 stress마다 붕괴, static은 견고.

★★**진짜 발견 = SLO 컨트롤러가 TTFT-bound(cudagraph) regime에 오설계** (`results/slo_sched/` 진단, 849258 slo 포화 로그): 주 신호가 **TPOT(decode)**인데 cudagraph regime binding은 **TTFT(prefill backlog)**. TPOT median 38.7ms(정상)로 컨트롤러는 "OK" 판단하나 실제 TTFT 6-10s로 붕괴. prefill-우선 경로(`_qd>target`)가 "`_tpot<lo`일 때만"으로 gated → 경합으로 TPOT이 lo밴드 밖이면 prefill 위기에 반응 실패, 오히려 TPOT 스파이크에 decode-heavy 오이동. agnostic은 decode_bs가 직접 decode-heavy로 몰아 최악(rep5 goodput 0.092). **∴ "static이 stress서 동적을 이김"은 컨트롤러 오설계의 반사효과.**

**Step D 판정**: context-length feedforward는 **현 컨트롤러 위에서 net win 아님**(tail만 개선, goodput 무이득, static이 지배). 단 이는 **base 컨트롤러가 TTFT-bound에 오설계된 것과 confound** — 공정한 재시험은 **TTFT/queue를 직접 우선하는 컨트롤러(Step E 후보)** 위에서. → 다음 레버 = feedforward 신호가 아니라 **컨트롤러를 TTFT-binding에 맞게 재설계**(prefill-queue를 gated 아닌 1차 신호로).

### D.6 caveat
- 이득은 **길이 이질 workload에서만**; 균질이면 고정 split로 degenerate(HD-iso가 그걸 확인).
- SLO-v7b 큐-feedback이 긴-요청 압력을 *부분* 흡수 → **명시적 L-feedforward의 *증분*이 진짜 미지수**(lag 제거분).
- layer-aware와 직교·부활 아님: cross-phase **스케줄링 신호**(SLO-aware 계열)지 층타입 정책 아님.
- 파일: 하네스 `results/slo_sched/lff_bench.sbatch`, 컨트롤러 `src/multiplex/multiplexing_mixin.py`(+dev 미러).

---

## Step E — binding-signal-first 컨트롤러 (TTFT-aware 재설계) (설계, 2026-07-14)

### E.0 한 줄
현 컨트롤러는 **TPOT(decode) 주도**인데 cudagraph-ON 운영점의 binding은 **TTFT(prefill backlog)**다. 그래서 static d24에 진다.
Step E = **binding SLO를 직접 신호로** — prefill-queue slack을 gated 2차가 아니라 **decode-slack과 대등한 1차 신호**로 올려,
매 순간 *더 급한 쪽*으로 split을 민다.

### E.1 설계 동기 (전부 이 세션 실측에 근거)

1. **binding 이동 (측정)**: cudagraph-ON이 decode wall 제거(TPOT 41→12ms, [cudagraph_probe]) → decode ITL이 60ms SLO에 큰 여유 →
   **binding이 decode(TPOT)에서 prefill(TTFT)로 이동**. Step D 전 라운드서 goodput ≈ TTFT-attainment(ITL은 통과).
2. **컨트롤러가 그 이동을 못 따라감 (진단, job 849258 slo 포화)**: 포화 시 **TPOT median 38.7ms=정상**으로 컨트롤러가 "OK" 오판,
   그동안 **실제 TTFT 6–10s로 붕괴**. `_slo_decide_idx`의 prefill-우선 경로가 `_tpot < _lo_frac` **gated** → 경합으로 TPOT이 lo밴드
   밖이면 prefill 위기에 **반응 못 함**; 오히려 TPOT 스파이크에 `min(_hi, _idx+1)`로 **decode-heavy 오이동**(prefill 더 굶김).
3. **그 결과 static이 이김 (3-rep, §D.7)**: d24(고정 prefill 84)는 신호를 안 보고 prefill에 커밋 → goodput 3.160[분산0]·tail 2.0–2.3.
   동적(slo 2.933/agn 0.092/lff 2.854)은 신호 오판으로 열위·고분산. **∴ static 우위 = 컨트롤러 결함의 반사효과지 static이 본질 우월이 아님.**
4. **오이동의 2차 피해 (측정)**: lff는 feedforward(prefill)와 feedback(TPOT 스파이크→decode)이 **충돌해 45 switch**(slo 18의 2.5×) →
   green-ctx drain 폭증. agnostic은 decode_bs가 직접 decode-heavy로 몰아 최악(0.092). → **잘못된/과잉 switch가 throughput까지 깎음.**

⇒ 문제는 **신호 선택**이다(feedforward L도, layer-type도 아님). binding(TTFT)을 1차로 보게 고치면 동적이 static을 이길 수 있는가? = Step E.

### E.2 지표 × 메커니즘 지배관계 (설계의 중심)

각 지표를 *무엇이 가장 많이 움직이나* + *이 세션 어디서 관측됐나*:

| 지표 | 지배 메커니즘 | 방향/근거 (실측) | Step E가 거는 레버 |
|---|---|---|---|
| **TTFT** | **prefill SM(P_sm) + prefill 우선순위** (prefill=compute-bound·SM-민감, (B,L) knee) | P_sm↓ → TTFT 폭발: agn rep5 decode-heavy로 P 굶겨 TTFT 6.8s; d24(P84) 2.3s. no-cg 과부하선 17–57s | **prefill-queue slack을 1차 신호로** → backlog 쌓이면 즉시 P_sm↑ |
| **TPOT** | **decode SM(D_sm) floor + cudagraph** (decode=memory-bound, 완만) | cudagraph서 TPOT ~12ms로 광범위 여유(D_sm 무관); no-cg서 ~40ms라 D 굶기면 즉시 ITL 위반(rep2 lff) | **decode-slack이 실제 위협일 때만** D 방어(cudagraph선 드묾) = 과잉 decode-이동 제거 |
| **Throughput** | **PD overlap − green-ctx switch(drain) 비용** | 과잉 switch가 깎음: lff 45 switch로 drain 폭증→꼴찌; coordinated 42 vs 미조율 121ms(drain 크기) | **deadband+dwell 유지**, 신호가 진짜 바뀔 때만 switch(수렴 후 희소) |
| **goodput@SLO** | **binding 제약의 attainment** = cudagraph선 **≈ TTFT-attainment**(ITL 여유) | Step D cg 라운드 전부 goodput이 TTFT로 결정(long good=TTFT<3s 여부) | binding=TTFT를 직접 최적화 → goodput = TTFT-first 제어의 직접 목표 |

**핵심 통찰**: cudagraph 운영점에서 **TPOT은 대부분 non-binding**(여유 큼)인데 현 컨트롤러는 그걸 주신호로 씀 = **비-binding 지표를 좇다 binding(TTFT)을 놓침**. Step E는 신호 우선순위를 binding에 맞춘다.

### E.3 설계 원리 — binding-signal-first (dual-slack)

두 slack을 **대등하게** 계산하고 *더 급한 쪽*으로 split을 민다:
- **prefill slack** = TTFT_SLO − (가장 오래 대기 중인 prefill의 큐 경과 또는 backlog-기반 예상 TTFT). 신호원: `waiting_queue` 대기시간/깊이 + split_prefill 잔여.
- **decode slack** = TPOT_SLO − TPOT-EMA (기존 v7b 신호 재사용).
- **제어**: `argmin(slack)`이 prefill이면 idx↓(P_sm↑), decode면 idx↑(D_sm↑), 둘 다 여유면 throughput 우선(prefill-heavy 기본, static d24 쪽). **gating 제거** — prefill-우선이 `_tpot<lo`에 종속되지 않음.
- **안정화 유지**: EMA·deadband·dwell(§v7b) 그대로 → 과잉 switch 방지(throughput 보호).

구현 지점: `_slo_decide_idx`(env `PDMUX_SLO_MODE=binding` 등으로 gate, off=v7b byte-identical). prefill-slack 신호는 `waiting_queue`의 요청별 `recv_time`/대기시간(스케줄러 인프라 존재) 활용.

### E.4 각 지표에 대한 예측 (설계가 어떻게 바꾸나)

- **TTFT**: backlog에 gating 없이 즉시 P_sm↑ → 붕괴(6–10s) 방지, static d24 수준(2.0–2.3)으로 수렴 기대. **주 개선 지표.**
- **TPOT**: cudagraph 여유 덕에 D 방어 드묾 → 현재와 동등 유지(위반 없음). no-cg면 decode-slack이 자주 binding → 그때만 방어.
- **Throughput**: 오이동(TPOT 스파이크→decode) 제거 + dwell로 switch↓ → lff의 45-switch drain 손실 회수.
- **goodput**: binding(TTFT) 직접 최적화 → **목표: static d24를 매칭 또는 상회**(d24는 고정이라 mixed/burst·regime 변화에 약함 → 동적이 이길 여지는 그 변동에서).

### E.5 비교군 · 가설 · 게이트

- 비교: **d24(이겨야 할 static)** / agnostic / **slo-v7b(현 TPOT-주도)** / **binding-first(Step E)** / (선택) binding-first + L-feedforward.
- **HE1 (핵심)**: cudagraph-ON near-boundary 이질-L에서 **binding-first가 slo-v7b를 이기고 d24를 매칭/상회**(goodput) + stress서 붕괴 안 함. 3-rep.
- **HE2 (동적의 진짜 가치)**: **regime이 시간에 따라 바뀌는 mixed/burst**(§B)에서 binding-first가 d24를 goodput으로 이김(static은 한 점 고정이라 불가).
- **HE0 (귀무)**: binding-first도 d24를 못 이김 → PD-split 동적 제어는 이 기판/워크로드서 static 대비 이득 無 → SLO track 종료(정직).
- **isolation**: homogeneous·stationary에서 binding-first ≈ static(무해) 확인.
- **재실험 재개점**: L-feedforward(Step D)를 **고친 컨트롤러 위에서** 재평가(공정 시험) — TTFT-first 위에선 feedforward가 순보탬일 수 있음.

### E.6 caveat
- Step D 학습 반영: 부하는 **near-boundary로 사전 보정**(과부하=전원 붕괴·경량=무차별). 부하 스윕 필수.
- prefill-slack 예측 신호는 dirty(큐 경과는 admission 이후만 봄, 도착 burst 예측 불가) → feedforward L이 여기 보완재로 재등판 가능(E.5 선택군).
- static d24가 이 regime서 강한 건 실측 사실 → **동적의 정당성은 "매칭 + 변동 regime서 초과"**지 stationary 압승이 아님(SLO track 원래 논지 유지).
- 이건 layer-aware와 무관·직교(신호/스케줄링 축). [[slo-aware-scheduling-track]].

> **각주 (surplus/anchor 검토, 2026-07-14)**: "prefill이 굶는(binding) 반대편 — prefill surplus 케이스의 anchor는 무엇인가"는
> 타당한 지적(순수 error-driven 컨트롤러는 anchor 없으면 표류; 경량부하 rep3서 slo/lff 배회로 실측됨). **제안된 "layer-type 반응
> anchor"는 부적합**: (i) prefill layer-type → Diff B(prefill)≈1((B,L) knee) = 한계 SM 가치가 layer-type 무관이라 anchor 근거
> 자체가 없음; (ii) decode layer-type(§14 예약) → surplus라도 (D) drain·cudagraph 몰수 구조적, Probe 4서 step-fixed 상수(d16)로
> degenerate. **올바른 anchor = tuned-static resting point(cudagraph≈d16, no-cg≈d24)** — §E.3의 anchor가 이것. 단 그 anchor의
> *값* d16은 decode layer-type 민감도(attn-decode floor≈16·mamba-decode free, decode knee)에서 **오프라인 유도**됨 → layer-type은
> "anchor 레벨 세팅"에서만 살고 "런타임 반응"에서는 죽음. 결론: anchor는 static, 구현은 §E대로.

### E.7 ★구현·실측 (2026-07-14, jobs 849844–849968)

구현: `_slo_decide_idx_binding` + `_slo_prefill_age_ms`(둘 다 `multiplexing_mixin.py`+dev 미러), env `PDMUX_SLO_MODE=binding`
(off=v7b byte-identical), 하네스 `lff_bench.sbatch` `bind` 모드(anchor idx2=d24). 부하=이질-L 동시 2스트림.

**(1) 신호 버그→수정**: 첫 bind 실행 SLO-BIND 로그 `pf_age=0ms·pfslack=1.00 항상` → prefill-slack 死 → bind가 decode-only=v7b로
degenerate(bind≈slo). 원인=TTFT 병목인 긴 요청이 결정 시점엔 `waiting_queue`를 떠나 **진행 중 `split_prefill_batch`**에서 청킹.
수정=`_slo_prefill_age_ms`가 waiting_queue **+ split_prefill_batch.reqs** 최고령 age를 봄. **확인(무거운 부하)**: pf_age가
**2093→5631ms climb**, 컨트롤러가 prefill 위기에 dec_sm=16으로 반응. 신호 live.

**(2) stationary 중간부하 (cudagraph, 3-rep)**: **bind(fixed) 2.948[2.52-3.16] ≈ d24 3.160** (tail 2.31/1.91 ≈ d24 2.31/2.01),
**slo(2.933·tail 3.79/3.26) 대비 결정적 개선** — 동적이 static에 **지던 것을 대등으로** 되돌림. (이 부하선 bind는 대부분 anchor 유지=switch 0 — 아무것도 urgent 아님 → anchor 덕에 d24 복제.)

**(3) 무거운 부하 (saturation, cudagraph)**: **d24 1.224 > bind 0.483/0.441 > slo 0.279.** bind>slo(flaw1 수정 확인)이나 **여전히 d24에 패.**
로그: **pfslack·decslack 동시 음수**(-0.88,-0.54)=**양 SLO 동시 위반=포화** → binding-first가 "더 급한 쪽"을 좇다 **2↔1↔3 flip-flop 11회**
→ green-ctx drain 폭증 → 진동 안 하는 static에 패.

**두 flaw 판정**:
- **flaw 1 (TPOT-centric, TTFT 못 봄) = 수정됨** (binding-first가 slo 이김).
- **flaw 2 (포화 시 진동) = 신규 노출** → §F saturation-hold 필요.

**★삼-regime 통합 (anchor-predictor 프레이밍 확증)**:

| regime | slack 상태 | 최적 | 담당 |
|---|---|---|---|
| **surplus** | 둘 다 여유(+) | 정지 | **anchor** |
| **single-binding** | 하나만 위반(−) | feasible trade | **동적(SLO-aware)** ← 유일 가치 구간 |
| **saturation** | 둘 다 위반(−) | trade 불가 | **anchor** (thrash 금지) |

⇒ **동적 제어는 "정확히 하나만 binding"인 좁은 구간에서만 static을 이기고, 양 극단(surplus·saturation)은 anchor로 fallback.**
layer-aware=그 anchor(decode-floor)를 예측하는 오프라인 predictor([[prefill-layer-alloc-status]] Probe4 §14→d16). **anchor-predictor 역할이 오히려 더 중심적.**

---

## Step F — saturation-hold 규칙 (설계, 2026-07-14; predictive-hybrid로 개정)

### F.0 한 줄
포화(양 SLO 동시 위반)에서 binding-first는 "더 급한 쪽"을 좇다 **진동**한다(§E.7-(3), 11 switch → drain → static에 패).
Step F = **포화를 *예측*해 chase 대신 anchor에 hold** — 동적이 static 밑으로 떨어지는 마지막 결함 제거.
★**개정(사용자 지적)**: 포화는 *예측 가능한* regime이므로, 트리거를 반응적 `both_neg`(진동 시작 후 감지=lagging)가 아니라
**예측적 backlog-추세**(진동 진입 *전* 차단)로 하고, `both_neg`는 안전망으로만 둔다(hybrid).

### F.1 설계 동기 (§E.7 실측 + 예측가능성)
- 무거운 부하 SLO-BIND 로그: `pfslack=-0.88 decslack=-0.54`(둘 다 음수) 상태서 `2↔1↔3` flip-flop 11회.
- **포화에선 feasible trade가 없다**: prefill 주면 decode가 더 급해지고(→되돌림) 그 반대도 성립 → 무한 왕복 → 매 왕복이 green-ctx drain.
- static d24는 **진동을 안 해서** 이김(한 split로 backlog 최대 소진). ⇒ 포화 구간 최적 = **anchor 고수**(§E.7 삼-regime 표).
- ★**포화는 예측 가능**: 포화 = "어떤 split로도 양 SLO 불가" = offered load > SLO-feasible 용량. `both_neg`는 이미 터진 뒤의 **lagging 지표**.
  더 이른 신호 = **backlog 추세**(가장 prefill-heavy split를 줬는데도 pf_age가 자람 = 최선의 수로도 backlog 증가 = 포화)이고,
  이는 transient double-violation(일시 스파이크)과 sustained 포화를 **구분**한다.

### F.2 규칙 (predictive-hybrid; 설계)
`_slo_decide_idx_binding`의 결정 트리에 **최우선 포화 가드**(2-tier):
- **(예측, 1차) `sat_predict`**: 현재 idx가 이미 **가장 prefill-heavy(=`_lo`)** ∧ 최근 W 윈도우에서 **pf_age 단조 증가**(backlog가 최선의 split로도 자람) → 포화 예측 → **anchor로 drift, chase 억제, dwell 최대**. (idx=lo 조건이 "최선의 수를 이미 썼다"를 보장 → prefill 더 못 줌 = 포화 신호.)
- **(반응, 2차 fallback) `both_neg`**: `pf_slack<0 ∧ dec_slack<0`(또는 둘 다 `< PDMUX_SLO_SAT_MARGIN`) → 예측이 놓친 경우 안전망으로 동일하게 anchor hold.
- 둘 중 하나라도 참이면 **saturation-hold**; 아니면 기존 로직(single-binding=chase / surplus=anchor drift).
- 결정 순서: **saturation(예측∨반응)→anchor** ▷ single-binding→chase ▷ surplus→anchor.
- env: `PDMUX_SLO_SAT_WIN`(추세 윈도우 W, 기본 4), `PDMUX_SLO_SAT_MARGIN`(fallback 여유, 기본 0). anchor는 §E와 동일 `PDMUX_SLO_ANCHOR_IDX`.
- 상태: pf_age 히스토리(최근 W개) 링버퍼를 컨트롤러 state에 유지.

### F.3 지표 예측
- **Throughput**: 진동 제거(11→~0 switch) → drain 회수 → 포화서 static 수준 회복. 예측 트리거라 **진동 진입 자체를 조기 차단**(반응형보다 switch 더 적음).
- **goodput**: 포화서 **bind → anchor(d24/d16)로 우아하게 degenerate** → **d24 매칭**. single-binding 구간의 동적 이득은 유지.
- **TTFT/TPOT**: 포화선 anchor(prefill-heavy)가 backlog 최대 소진 → 둘 다 static과 동등(포화라 SLO 초과는 불가피, static과 동률).

### F.4 가설·게이트
- **HF1**: saturation-hold(predictive) 후 **무거운 부하서 bind ≥ d24(≈매칭)**, 진동 switch 급감. (현재 bind 0.48 → d24 1.22 근접 목표.)
- **HF-iso**: surplus·중간 부하선 §E.7 거동 불변(무해).
- **HF0**: 여전히 d24에 지면 → 포화서 동적은 원리적으로 static 못 넘음(동적 가치는 single-binding·시변 regime에만, HE2로 이동).

### F.5 caveat·후속
- **burst**: 단기 도착률 예측은 burst서 부정확 → **순수 예측 위험 → hybrid 필수**(예측 놓치면 both_neg fallback). action은 어차피 "anchor 고수"뿐(admission 제어=스케줄러 몫, 우리 레버 밖)이라 예측이 바꾸는 건 **트리거(언제)**지 행동 아님.
- **원리적 예측(옵션)**: backlog-추세 proxy 대신 λ(관측)×요청 work(L) vs service surface(knee)로 feasibility 직접 계산 가능 — 공수 큼, proxy 우선.
- **후속1 anchor=d16 검증**: cudagraph 최적 anchor=layer-type 예측값 d16(Probe4). d24 대신 d16 anchor 재측정 = anchor-predictor 실험확인.
- **후속2 HE2(시변 regime)**: binding이 *교대*하는 mixed/burst(§B, 포화 아님)에서만 동적이 static 초과 가능 → 최종 payoff.
- 파일: 컨트롤러 `multiplexing_mixin.py`(`_slo_decide_idx_binding`에 2-tier 가드 + pf_age 링버퍼), 하네스 `lff_bench.sbatch` `bind` 재사용.
