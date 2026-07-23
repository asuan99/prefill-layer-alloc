# Prefill vs Decode — 실행 관점의 차이, 그리고 layer-aware 함의

> **HISTORICAL/SUPERSEDED:** characterization 출발점으로만 사용한다. prefill은
> 항상 compute-bound, decode는 항상 memory-bound라는 식의 일반화와 미완 branch
> 상태는 현재 주장에 사용하지 않는다. 정본:
> [`../../../PROJECT_STATUS.md`](../../../PROJECT_STATUS.md).

작성: 2026-07-08. 목적: layer-aware(per-layer-type SM 배분)가 **decode에선 반증**됐는데 **prefill 쪽은 왜 다른 질문**인지를,
두 phase의 *실행 특성* 차이로 정초한다. prefill-knee 실험과 (있다면) prefill-coord 실험의 기반 문서.

관련: [sm_policy_report.html §07](sm_policy_report.html) · decode knee `results/r0c/knee_result_835571.txt` · roofline `results/motivation/fig_c_roofline_zamba2.txt`.

---

## 0. 한 문장

**Prefill = compute-bound · 크고 긴 커널(프롬프트 전체 토큰). Decode = memory-bound · 작고 짧은 커널(스텝당 1토큰/시퀀스).**
나머지 차이(SM 민감도·layer-aware 성립 여부·(D) granularity 결과)는 전부 여기서 파생된다.

## 1. 근본 차이 — 한 forward가 처리하는 토큰 수

| | forward당 토큰 | 커널 성격 | 층당 지속시간 |
|---|---|---|---|
| **Prefill** | B × L_prompt (예: 1×3600) | 큰 GEMM (가중치를 수천 토큰에 재사용) | **김** (수 ms/층, 전체 프롬프트) |
| **Decode** | B × 1 (예: 48×1) | 얇은 GEMM/GEMV (토큰당 가중치 1회 로드) | **짧음** (0.4–3 ms/층) |

이 하나가 아래 모든 걸 결정한다.

## 2. Roofline — compute-bound vs memory-bound → SM 민감도

- **Prefill(고 arithmetic intensity)**: 큰 matmul, 토큰 간 가중치 재사용 → **compute roof에 근접 → SM-민감**(SM↑ = FLOP/s↑).
- **Decode(저 AI)**: 토큰 적어 가중치·KV 로드가 지배 → **memory bandwidth-bound → SM에 둔감**(대역폭 한계라 SM 더 줘도 포화).

roofline 실측(fig C): "SSM은 memory-bound 영역(low AI)에 뭉치고, Attn은 장문서 compute roof로 이동(high AI)". 즉 **동일 층타입도 phase에 따라 roofline 위치가 이동**한다.

## 3. Layer-type × phase 비용 구조 (핵심 표)

| 층타입 | **Prefill** | **Decode** |
|---|---|---|
| **Attn** | O(L²) — 프롬프트 전체 self-attn. 장문서 prefill의 **지배 비용**. compute-bound·SM-민감. | O(L) KV-read — 캐시 전체를 1토큰 위해 읽음. memory-bound. **decode의 비싼 층**. |
| **Mamba(SSD)** | O(L) chunked scan — 선형, chunk matmul. compute-bound, attn보다 쌈. | O(1) — 단일토큰 재귀, 미세 상태갱신. memory-bound·미세. **decode의 싼·둔감 층**. |
| **MLP** | 큰 GEMM. compute-bound. | 얇은 GEMM. weight-load-bound. |

**decode knee 실측(Zamba2-2.7B, ctx3600, serving batch; `results/r0c`):**

| decode SM | per-attn 층 | per-mamba 층 |
|---|---|---|
| 108 | 3.14 ms | 0.36 ms |
| 44 | 6.27 | 0.27 |
| 24 | 11.77 | 0.40 |
| 16 | 17.51 | 0.53 |
| 8 | 34.0 | 0.97 |

→ decode: **attn 5.6× 민감·비쌈 / mamba ~1.5× 둔감·쌈**. 이 큰 비대칭이 decode-side layer-aware의 근거였다.

**prefill knee 실측(같은 조건, batched prefill; job 837931):**

| prefill SM | per-attn 층 | per-mamba 층 | attn 민감도 | mamba 민감도 | attn/mamba 비용비 |
|---|---|---|---|---|---|
| 108 | 10.97 ms | 9.03 ms | 1.0× | 1.0× | 1.21× |
| 44 | 25.18 | 17.51 | 2.30× | 1.94× | 1.44× |
| 24 | 45.78 | 31.98 | 4.17× | 3.54× | 1.43× |
| 16 | 68.52 | 47.62 | 6.25× | 5.27× | 1.44× |
| 8 | 135.4 | 94.8 | 12.3× | 10.5× | 1.43× |

→ **prefill: 두 타입 다 강하게 SM-민감**(108→8서 attn 12.3× · mamba 10.5×), attn이 mamba보다 **겨우 1.2–1.4× 비쌈**(near-symmetric). decode(attn이 mamba보다 8.7–35× 비싸고 mamba 평탄)와 **정반대**. **prefill엔 "공짜로 뺄 수 있는 둔감 층"이 없다.**

## 4. 스케줄링 — 어떻게 실행되나 (sglang pdmux, 코드 근거)

- **Decode = steady heartbeat**: 매 스텝 running batch 전체를 54층 1-forward(`forward`). 생성 토큰당 1회 → 요청당 **多**(96토큰=96스텝). **latency-critical(TPOT SLO)**.
- **Prefill = chunked burst**: `forward_split_prefill`(SPLIT_PREFILL)로 스텝당 `split_forward_token_budget//L` 층씩 전진, hidden state를 스텝 간 스레딩. 요청 admission시 **한 번**, 여러 decode 스텝에 걸쳐 쪼개짐. **throughput-critical(TTFT SLO)**.
- **PD-mux**: event loop가 매 스텝 decode(전체) ∥ prefill(한 chunk)를 **disjoint green-ctx 파티션**서 동시 실행. split (P_sm, D_sm), P+D=108.

⇒ 서빙선 prefill이 **full-prompt 한 방이 아니라 스텝당 ~18층 chunk**로 인터리브된다. 이게 prefill-side layer-aware의 *유효 창 granularity*를 정한다(§6).

## 5. SM의 한계효용 — PD-split이 성립하는 이유

- **Prefill(compute-bound)**: SM↑ → 거의 비례 가속(roof 전까지). **SM 한계효용 높음 → prefill이 SM을 갈망**.
- **Decode(memory-bound)**: SM↑ → 체감(대역폭 한계). **한계효용 낮음 → 적당한 SM서 포화**.

→ **PD-mux가 이기는 이유**: decode엔 대역폭 포화할 만큼만(적당), 나머지 전부 prefill(연산 갈망)에. agnostic/tuned-uniform이 이 균형을 step 단위로 잡는다. (fused는 이 분리를 안 해 decode가 prefill에 결합→TPOT 폭발.)

## 6. Layer-aware 함의 — 왜 decode-side는 죽고 prefill-side는 다른 질문인가

layer-aware(per-type SM 재배분)의 성립조건 = **① 타입 간 SM 한계효용 differential** + **② 그 differential이 사는 granularity서 SM을 재배분 가능**.

**Decode-side (반증됨):**
- ① differential **큼**(mamba ≈ SM-free). 
- ② **불가** — decode 층 창(~0.4ms mamba)이 concurrent prefill 커널보다 **짧아**, 풀린 SM을 prefill이 못 잡음. = **(D) granularity**. 4개 구현 전부 패배(inefficient_v1 698 / R0d 124 / (a) 85 / v4 95 ms), step-level(agnostic 42)에 짐.
- ⇒ **differential이 커도 짧은 창이 죽인다.**

**Prefill-side (미검증·비대칭):**
- ① differential은 **degree 차이**(attn O(L²) ≫ mamba O(L), 둘 다 compute-bound·SM-민감). decode보다 **작을** 수 있음.
- ② 그러나 prefill 커널이 **크고 길다** → **창이 김** → switch-cost 대 window 비율이 decode보다 **유리**할 수 있음. **(D) 논증이 대칭으로 적용 안 됨.**
- 방향도 다름: 싼 ssm-prefill서 SM을 **decode에 환원**(→ TPOT↓). decode-side와 반대.
- ⚠️ 단서: (i) 서빙 chunk로 유효 창은 full-prompt가 아니라 ~2.2ms/층(§4) — decode(0.4ms)보단 길지만 극적 아님. (ii) 노리는 metric(decode TPOT)이 고부하 binding(대개 TTFT)과 어긋날 수 있음 — decode-heavy regime서만 유리. (iii) 경쟁자는 step 단위 공짜 재배분하는 tuned-uniform.

⇒ **prefill-side layer-aware는 "argued moot"가 아니라 "genuinely untested with non-symmetric (D)".**

## 7. Prefill knee가 재는 것 (진행중, job 837931)

`SGLANG_ZAMBA_PREFILL_KNEE=1`이 decode-knee의 pin+timing을 **extend(prefill) forward**로 재타겟 → per-attn-prefill·per-mamba-prefill 시간을 SM(108/44/24/16/8)별 측정. **재는 것 = 조건 ①(differential 크기)**:
- attn-prefill·mamba-prefill **기울기 차이가 크면** → prefill-side lever 유의미 → §6-② timescale과 함께 prefill-coord 실험 정당.
- 기울기 차이가 작으면 → lever 미미 → 종결.

**판정(실측, §3):** 두 타입 다 강하게 SM-민감(108→24서 attn 4.17× · mamba 3.54×), differential은 **작다**(비용비 1.2–1.4× 내내). decode의 lever(mamba 평탄·8.7× 저렴)와 달리 **prefill엔 다치지 않고 뺄 수 있는 SM이 없다** — mamba-prefill을 빼면 거의 비례해 느려짐(1:1 트레이드). ⇒ per-type 세분의 이득은 **2차(작은 기울기 차)**, (D) 비용은 **1차** → **net 음수**. prefill-side layer-aware는 step-level prefill/decode split(tuned-uniform이 이미 공짜로 함)으로 degenerate. **prefill-coord 실험 불필요 — 종결.**

**대칭 완성:** decode = lever 큼(mamba free) → 짧은 창((D))이 죽임; prefill = 창은 더 길지만 → lever가 애초에 무시할 수준(둘 다 compute-bound). **이유는 반대, 결론 동일: per-layer-type 세분은 step-level을 못 이긴다.** 원자료 `results/prefill_knee/knee_result_837931.txt`.
