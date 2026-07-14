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

### D.6 caveat
- 이득은 **길이 이질 workload에서만**; 균질이면 고정 split로 degenerate(HD-iso가 그걸 확인).
- SLO-v7b 큐-feedback이 긴-요청 압력을 *부분* 흡수 → **명시적 L-feedforward의 *증분*이 진짜 미지수**(lag 제거분).
- layer-aware와 직교·부활 아님: cross-phase **스케줄링 신호**(SLO-aware 계열)지 층타입 정책 아님.
- 파일: 하네스 `results/slo_sched/lff_bench.sbatch`, 컨트롤러 `src/multiplex/multiplexing_mixin.py`(+dev 미러).
