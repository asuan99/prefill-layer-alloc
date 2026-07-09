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
