# Queue 시뮬레이터 설계 — hybrid serving의 SLO 하 분할 vs co-schedule

작성일: 2026-06-18 · 브랜치 `exp/queue-sim` (base `exp/e5-real-prefill`) · 코드 `experiments/e6_queue_sim/`
관련: [real_prefill_results](real_prefill_results.md) · [검수 A5](review_checklist.md) · [real_prefill_experiment_design §4.5](real_prefill_experiment_design.md)

---

## 1. 왜 필요한가 — microbench가 못 보는 것

지금까지의 E5(단일 셀: prefill chunk 1개 ∥ decode step 1개)는 **SLO 격차를 구조적으로 못 본다.** [SLO 분석](real_prefill_results.md)이 보였듯 microbench에선 two_stream이 decode 지연·throughput **양 축 모두 우세**(decode_inflation 3.6–5.3% vs green_ctx 66–101%) — 격리할 *지속 경합*이 없기 때문이다. MuxWise/Bullet식 분할 이득의 전제는 **다수 요청이 동시에 흐르며 prefill burst가 decode tail을 누르는 *지속 부하***이고, 이는 큐/스케줄링 현상이라 단일-커널 microbench로 재현 불가하다.

→ 이 시뮬레이터는 **측정된 per-step latency(E5/E3 LUT)를 *큐 동역학* 위에 합성**해, "지속 부하 + SLO(decode tail) 목적함수"에서 *동적 decode-보호 분할*이 *동적 co-schedule*을 이기는 regime이 있는지를 GPU 없이 탐색한다.

## 2. RQ & 가설

- **RQ:** 도착률 λ를 올릴 때, decode tail-latency(ITL p99)/goodput@SLO 기준으로 `dynamic_protect`(decode에 floor 예약) 가 `co_schedule`(예약 없는 동적 공유)을 이기는 부하 영역이 존재하는가? 존재한다면 어느 (모델, kernel-pair, λ)에서인가?
- **가설:** 저λ(한가): co_schedule 우세(work-conserving, 예약 비용만 손해) — microbench와 일치. 고λ(포화 근처): prefill 주입이 매 step decode를 밀어 co_schedule의 ITL p99가 폭발 → 예약(protect)이 tail을 bound해 SLO-goodput 역전. **이 역전점의 존재/위치가 결과.**

## 3. 아키텍처 — iteration-level continuous batching (+ chunked prefill)

```
 arrivals (Poisson λ)──▶ [waiting] ──admit──▶ [prefilling] ──prefill 완료(TTFT)──▶ [decoding] ──출력 끝──▶ [done]
                                        │                                          │
                                        └────────── 매 iteration ───────────────────┘
   각 iteration:
     1) 현재 시각 이전 도착 요청을 waiting에 편입
     2) 스케줄러: chunked-prefill 예산(기본 1 chunk)만큼 prefill + decode중 전부를 batch
        → 한 step의 작업 = (prefill chunk 0~1개) ∥ (decode batch B = |decoding|)
     3) step_ms = latency_model(policy, pf_layer, dec_layer, B, ctx, prefill_active)   ← E5 LUT
     4) clock += step_ms;  prefill 잔여 chunk--, decode 잔여 token-- (step당 1 token)
     5) prefill 완료→TTFT 기록·decoding 이동;  decode 완료→완료시각 기록
```

핵심: **chunked prefill이 step당 prefill 예산을 bound**하므로 한 step의 구성이 정확히 *E5가 측정한 "1 chunk ∥ decode batch"*다 → **E5 LUT이 step-time 모델로 적합**. 시뮬레이터가 더하는 것은 *큐잉*(prefill 요청 대기, 매 step chunk 주입이 누적적으로 decode tail에 미치는 영향)이다.

## 4. Latency 모델 (`latency_model.py`) — 측정 LUT 구동, GPU-free

E5 full CSV(`serving_coexec_full_{model}_*.csv`, decode-protect 포함시 `green_ctx_protect` 행)에서 LUT 구성:

| policy | 사용하는 backend 행 | 의미 |
|---|---|---|
| `co_schedule` | `two_stream` | 예약 없는 동적 공유 |
| `static@f` | `green_ctx`(해당 f) | 고정 분할 |
| `dynamic_protect` | `green_ctx_protect` | decode=E3 floor 예약 |

`step_ms(policy, pf_layer, dec_layer, decode_batch, ctx, prefill_active)`:
- `prefill_active=True` → LUT[backend, pf, dec, snap(decode_batch), ctx].`concurrent_ms`
- `prefill_active=False`(decode-only step) → 같은 cell의 `solo_decode_ms`(backend 불변)
- `decode_batch`는 LUT 격자{1,2,4,…512}로 **최근접 snap**(또는 로그선형 보간). 결측/실패(OOM·no_room) cell은 이웃으로 폴백 + 경고.

**모델의 한계(정직):** LUT은 *1 prefill chunk* 경합만 담는다. chunked-prefill이 step당 prefill을 1 chunk로 bound하면 정합하나, prefill 예산>1(여러 chunk 동시)이면 과소추정 → 그 경우 `--prefill-batch`로 측정한 LUT(이미 코드 지원)를 써야 한다(v1 확장). 절대 latency 신뢰도 경고(BW% 등)는 상속하나, *정책 간 상대 비교*가 목적이라 영향 작음.

## 5. 정책 (`scheduler` policy)

- `co_schedule` (baseline): 예약 없음. 모든 step이 two_stream LUT.
- `static_split@f`: 고정 f. green_ctx LUT.
- `dynamic_protect`: 매 step decode batch B에 대해 decode에 E3 floor(B) SM 예약 → green_ctx_protect LUT. (decode floor가 108이면 no_room → co_schedule로 폴백 + 카운트.)
- (확장) `adaptive`: 부하/큐 길이로 protect↔co_schedule 전환하는 hysteresis 정책.

## 6. 메트릭 (`run_queue_sim.py` 출력)

요청별: TTFT(=prefill 완료 시각−도착), 각 decode step의 ITL, 완료시각. 집계:
- **TTFT** p50/p99, **ITL(decode tail)** p50/**p99** ← 핵심 SLO 지표
- **throughput**(tok/s), **goodput**(ITL p99 ≤ SLO_target 인 요청 비율·또는 SLO 만족 req/s)
- 큐 길이·SM 점유 시계열(선택)

출력: `results_v2/e6/queue_{model}_{policy}.csv` + λ×policy 요약 표. **역전점**(protect가 co_schedule의 goodput을 추월하는 λ)을 보고.

## 7. 검증 계획

1. **저λ 극한 = microbench 재현:** λ→0이면 동시 요청 1개 수준 → co_schedule이 protect를 이김(real_prefill SLO 결과와 일치)해야. 안 그러면 LUT/루프 버그.
2. **LUT 일관성:** step_ms가 E5 CSV 값과 셀 단위 일치(snap 제외).
3. **보존 법칙:** Σ(per-req 시간) ≈ Σ(iteration 시간), 요청 수·토큰 수 보존.
4. **단조성 sanity:** λ↑ → 큐·tail 증가.

## 8. 범위 & 한계 (정직)

- **v0 = 단일 kernel-pair**(예: pf=ssm×dec=ssm 또는 2.7b의 pf=attn×dec=ssm). 전체 모델(ssm·attn 레이어 합성, embedding/LM head)은 *layer-fraction 가중합*으로 확장(v1).
- **LUT 기반 = 측정 충실하나 1-chunk 경합 한정**(§4). multi-prefill pile-up은 prefill_batch LUT 필요.
- **도착 모델**: Poisson + (prompt,output) 분포는 합성. 실 트레이스(예: ShareGPT, hf_cache에 sharegpt 있음)로 교체 가능.
- **discrete-event = 스케줄링 동역학 검증용**, 실 GPU serving 엔진 대체 아님. 양성 역전이 나오면 *실 엔진/MPS로 확증*하는 게 다음 단계.
- 이건 *분할이 이긴다*는 증명 시도가 아니라 **"microbench가 못 본 SLO regime에 역전점이 있나"를 GPU 없이 빠르게 탐색**하는 도구다. 음성(역전 없음)이면 "분할은 크기·objective·부하 어느 축에서도 무용" 결론이 큐까지 닫힌다.

## 9. 파일 (`experiments/e6_queue_sim/`)

- `latency_model.py` — E5/E3 CSV → step-time LUT + snap/폴백.
- `workload.py` — Poisson 도착 + (prompt_len, output_len) 분포 → 요청 스트림.
- `simulator.py` — iteration-level continuous-batching 루프(§3) + 정책.
- `run_queue_sim.py` — λ×policy 스윕 드라이버, SLO 메트릭 CSV/요약.
- `run_sim_pipeline.sh` — **사용자 실행용** 파이프라인(아래 §10).

## 11. 결과 (v0, job 776826 single-chunk LUT, 2026-06-19) — 역전 없음

single-chunk LUT(tokens=256+decode-protect)로 λ×policy 스윕(SLO: ITL p99 목표 ≤1.0ms, 400 req). 대표값:

| model · pair | λ(req/ms) | co_schedule SLO/ITLp99/tok·s | dynamic_protect SLO/ITLp99/tok·s |
|---|--|--|--|
| zamba2_2.7b attn×ssm | 0.2 | **1.00 / 0.62 / 28.3k** | 0.94 / 1.01 / 26.0k |
| zamba2_2.7b attn×ssm | 5.0(포화) | **1.00 / 0.75 / 29.9k** | 0.93 / 1.01 / 26.6k |
| zamba2_2.7b ssm×ssm | 5.0 | **1.00 / 0.74 / 29.7k** | 1.00 / 0.86 / 27.0k |
| zamba2_7b attn×ssm | 5.0 | **1.00 / 0.83 / 27.9k** | = co_schedule (protect=no_room→폴백) |

**결론: 세 구성·전 λ에서 `dynamic_protect`가 `co_schedule`을 SLO·throughput 어디서도 못 이긴다(역전점 0).** 고λ에선 오히려 protect가 *나쁨*(decode에 floor만 주고 prefill을 굶겨 ITL p99가 SLO를 넘고 throughput↓). 즉 **지속 부하·decode tail SLO 목적함수에서도 동적 co-schedule이 최적**이다.

**왜(중요·정직):** 시뮬레이터는 *chunked-prefill*(step당 prefill 1 chunk)을 가정 → 한 step의 비용이 E5 LUT의 "1 chunk ∥ decode" 그대로다. 이 regime에선 co_schedule의 decode tail이 **이미 LUT로 bound**되므로(1 chunk만 경합) 예약이 회수할 slack이 없다. MuxWise/Bullet식 이득이 나오려면 *step당 여러 prefill chunk가 쌓여 decode tail을 누르는* 상황이 필요한데, **chunked prefill이 바로 그걸 막는 표준 기법**이다 — 즉 그 regime이면 답은 *공간 분할이 아니라 시간적 chunking*이다. 큐는 TTFT(입장 지연)를 λ↑서 폭발시키지만(전 정책 공통), decode ITL은 step-LUT로 묶이고 거기선 co_schedule이 최선.

**한계:** v0 LUT=1 chunk/step. prefill 예산>1(burst)은 `--prefill-batch>1` LUT로 모델해야(미측정) — 단 그건 chunked-prefill이 회피하는 비표준 운영. zamba2_7b는 protect=no_room이라 co_schedule과 동일(폴백). 양성 역전이 나왔다면 실 MPS 엔진 확증이 다음이었으나, **음성이라 "분할은 크기·objective·부하·큐 어느 축에서도 co-schedule을 못 이긴다"가 닫힌다**(chunked-prefill 가정 하).

→ 산출: `results_v2/e6/queue_sim_{model}_{pf}x{dec}.csv`.

## 10. 사용자 실행 (`run_sim_pipeline.sh`)

시뮬레이터는 **single-chunk LUT**(1 chunk ∥ 1 decode step)가 필요하다 → E5 full + `--prefill-tokens 256 --decode-protect`로 생성. 이걸 제출+분석으로 묶은 사용자 실행 스크립트:

```bash
cd workspace/characterization
# (a) LUT 제출만
env -u BASH_ENV bash experiments/e6_queue_sim/run_sim_pipeline.sh submit
# (b) 잡 완료 후 분석만
env -u BASH_ENV bash experiments/e6_queue_sim/run_sim_pipeline.sh analyze
# (c) 제출→완료 폴링→분석 한 번에
env -u BASH_ENV bash experiments/e6_queue_sim/run_sim_pipeline.sh auto
```
튜닝(env): `MODELS="zamba2_2.7b zamba2_7b"`(≤4, QOS), `PF=attn DEC=ssm`(2.7b 예외 셀), `SLO=1.0`(ms), `LAMBDAS="0.02 … 2.0"`.
LUT → `results_v2/e5_sim/`(기존 e5/e5_dp 불간섭), sim 결과 → `results_v2/e6/queue_sim_*.csv`.
> QOS 한도상 다른 array가 큐에 있으면 submit 실패 — 그 잡이 끝난 뒤 실행. `analyze`는 GPU 불필요.
