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

**결론(scope 주의): step당 prefill budget = 1 chunk(256 tok)인 regime에서, 세 구성·전 λ에서 `dynamic_protect`가 `co_schedule`을 SLO·throughput 못 이긴다(역전점 0).** 고λ선 오히려 protect가 *나쁨*(decode에 floor만 주고 prefill을 굶겨 ITL p99>SLO, throughput↓). 즉 **작은-budget chunked-prefill에선 동적 co-schedule이 최적**이다.

**왜:** 이 LUT(1 chunk ∥ decode)에선 step당 prefill이 작아 co_schedule의 decode tail이 **이미 bound**돼 예약이 회수할 slack이 없다. 큐는 TTFT를 λ↑서 폭발시키지만(전 정책 공통), decode ITL은 step-LUT로 묶이고 거기선 co_schedule이 최선.

**⚠ 정정 — "비표준" 과대종결 철회:** 이전 초안은 "더 센 경합 regime은 chunked-prefill이 막는 비표준"이라 했으나 **틀렸다.** chunked prefill이 막는 건 *chunking을 안 하는* 케이스(통째 prefill이 decode를 통째로 막음)뿐이다. 그러나 **step당 prefill *budget*(Sarathi/vLLM `max_num_batched_tokens`, 보통 512–2048)이 큰 것은 완전히 표준**이고, budget이 크면 한 step에 prefill GEMM이 여러 chunk만큼 들어가 decode와 더 세게 경합한다. 본 sim은 **budget = 1 chunk(최소)** 만 썼으므로, **큰-budget(표준) regime은 미검증이며 분할이 이길 *유일하게 남은 합당한* 영역**이다(= MuxWise/Bullet의 실제 운영점일 공산). 이건 TTFT↔TBT 트레이드오프이고 budget과 분할이 같은 트레이드오프의 두 손잡이라, SLO 가중치에 따라 분할이 이길 수 있다.

**다음(§12에서 측정):** step당 prefill budget을 키워(`--prefill-batch>1` LUT = "k chunk ∥ decode") sim에 `--prefill-budget` 파라미터로 주입 → 큰-budget에서 역전점 탐색. zamba2_7b는 protect=no_room이라 co_schedule과 동일(폴백).

→ 산출: `results_v2/e6/queue_sim_{model}_{pf}x{dec}_b{budget}.csv`.

## 12. 수정된 결과 (decoupled 스케줄러 + goodput@SLO) — 분할 이득은 *실재하나 조건부*

§11/이전 §12의 음성 결론은 **두 결함의 산물**이었다(둘 다 사용자 지적으로 발견): **(A)** sim이 decode ITL = `concurrent_ms`로 둬 decode를 *굶긴 prefill*에 묶음(= partition의 decoupling을 죽임). **(B)** raw throughput/latency로만 봄(올바른 metric은 SLO 만족 throughput = **goodput**). 수정: partition 정책에 **decoupled 스케줄러**(decode ITL = `decode_stream_ms`, prefill과 독립 — MuxWise/Bullet 설계) + **goodput = throughput × SLO_attain**. (budget=8 LUT = job 778147, e5_sim_b8.)

핵심 표 (pf=attn×dec=ssm, SLO ITL≤1.0ms; **goodput**=SLO 만족 tok/s):

| 구성 | λ | co_schedule(sync) ITLp99/attain/raw/**good** | dynamic_protect(decoupled) ITLp99/attain/raw/**good** |
|---|--|--|--|
| 2.7b · b1 | 5.0 | 0.75 / 1.00 / 29.8k / **29.8k** | 0.79 / 1.00 / 30.9k / **30.9k** (근소 승) |
| **2.7b · b8** | 0.5 | 1.90 / 0.20 / 55.9k / **11.0k** | 0.71 / **1.00** / 33.5k / **33.5k** (압승) |
| **2.7b · b8** | 1.0 | 1.90 / 0.15 / 58.3k / **8.9k** | 0.71 / **1.00** / 33.6k / **33.6k** (3.8×) |
| fh1_3b · b8 | 0.5 | 0.95 / 1.00 / 63.2k / **63.2k** | 1.11 / 0.18 / 61.2k / **10.9k** (패) |

**(1) 사용자 직관 확인 — 이득은 raw throughput이 아니라 goodput에 있다.** co_schedule이 raw throughput은 항상 큼(58k vs 33k). **그러나 큰 budget·고부하에서 co_schedule의 decode ITL이 SLO를 위반(1.9ms>1.0)** → 그 throughput은 헛것(goodput 8.9k). protect(decoupled)는 decode를 reserved SM에서 돌려 **ITL을 0.71ms로 bound** → SLO 1.0 유지 → **goodput 33.6k (~3.8× 승)**. **이게 MuxWise/Bullet 이득의 재현이다 — 이전 "분할 모든 축 음성"은 모델 결함 산물이었다(철회).**

**(2) 단 *조건부*다(정직):** falcon_h1_3b는 protect의 *reserved* decode ITL이 1.1ms로 SLO(1.0)를 못 맞춰 → goodput 패. 즉 분할 이득은 **"reserved decode가 SLO를 만족하는가"**에 달렸다(모델의 decode floor·비용 의존). 그리고 protect는 **TTFT를 크게 희생**(2.7b b8 λ1.0: TTFT p99 1220ms vs co_schedule 432ms — prefill 굶김). → **TTFT↔TBT↔goodput 3-way 트레이드오프**이고, 분할은 *decode-SLO가 binding이고 reserved decode가 SLO를 맞추는 큰-budget·고부하 코너*에서 이긴다.

**(3) ★ 그러나 baseline이 틀렸다 — vLLM 기본은 two_stream이 아니라 *fused mixed-batch*다.** two_stream(별도 decode 커널)은 *약한* baseline이고, vLLM/SGLang 기본은 prefill+decode를 **한 batch로 융합**(decode가 prefill GEMM에 거의 공짜 편승). `fused`를 넣으니 (2.7b b8):

| λ | **fused(vLLM)** ITLp99/attain/**good** | co_schedule(two_stream) **good** | dynamic_protect **good** |
|--|--|--|--|
| 0.5 | 0.96 / 1.00 / **63.2k** | 11.0k | 33.5k |
| 1.0 | 1.33 / 0.94 / **83.2k** | 8.9k | 33.6k |

**fused가 goodput을 압도**(83k ≫ protect 34k ≫ two_stream 9k). decode가 융합 GEMM에 편승해 throughput↑(89k) + ITL도 mostly-SLO(0.94). → **올바른 baseline(fused) 대비 partition은 budget=8에서 *못 이긴다*. 제가 "partition 3.8× 승"이라 한 건 *약한 two_stream* 대비였을 뿐**(사용자 지적이 정확). 단 fused 모델은 *낙관적*(decode 완전 free 가정; 실제 decode-attn의 KV 읽기는 비용 있음)이라 fused goodput은 상한.

**(4) ★ fused 모델은 measured decode 비용에 극도로 민감 — fused-vs-partition은 *미결*.** fused step을 `solo_prefill`(decode free, frac=0)로 둔 게 §12-(3)의 "fused 압도"였다. **decode 비용은 실측돼 있다**(`solo_decode_ms`; dec=attn은 full-KV SDPA 읽기 포함). 이를 도로 더하면(frac=1, 보수) fused ITL이 SLO를 위반:

| 셀 (b8, λ1) | fused frac=0(낙관) good | **fused frac=1(보수)** good | dynamic_protect good |
|---|--|--|--|
| dec=ssm | 83.2k | **7.4k (ITL 2.3>SLO 붕괴)** | 33.6k |
| dec=attn | 37.6k | **0.16k (ITL 8.9 붕괴)** | 45.0k |

→ **낙관 fused면 fused 승, 보수 fused면 partition 승.** 진짜 fused 비용은 그 사이(가산은 weight-load 공유를 과대계상, free는 KV 읽기를 누락)이고, **E5는 *진짜 fused 커널*(한 batch에 P+B 토큰)을 측정한 적이 없다 — prefill·decode를 별도 커널로만 쟀다.** ∴ **fused-vs-partition은 현 데이터로 미결**(bounded: 낙관 fused 승 ~ 보수 partition 승). 결정하려면 **fused mixed-batch 커널 직접 측정**(신규 실험)이 필요.

**(5) ★ 진짜 fused 커널 실측(job 778472, `run_fused_step`) — 미결 닫힘: fusion saving ≈ 0.** P prefill + B decode를 한 forward로 융합한 layer-step을 직접 측정(proj GEMM은 (P+B) 한 번). **결과: `fusion_saving = (prefill_only+decode_only) − fused ≈ 0`**(전 셀 0.01–0.23ms):

| 셀 | fused 실측 | prefill_only+decode_only(보수) | prefill_only(낙관) |
|---|--|--|--|
| fh1_3b P2048 ssm B512 | 6.93 | 6.93 (=) | 0.80 |
| 2.7b P256 attn B512 | 31.1 | 31.2 (=) | 0.26 |

즉 **fused ≈ prefill+decode (보수가 맞다); decode는 fused에서 *공짜가 아니다*.** 공유되는 건 proj GEMM weight-load뿐인데 그건 총비용의 작은 조각이고 **mixer(특히 decode-attn KV 읽기)가 지배**해서 융합 절약이 거의 없다. → 내 §12-(3) 낙관 모델은 *틀렸고*, §12-(4) 보수가 실측과 일치.

**실측 fused로 sim 재실행(2.7b b8, `--fused-lut`):**

| layer | λ | fused(실측) ITLp99/attain/**good** | dynamic_protect **good** |
|---|--|--|--|
| ssm | 1.0 | 2.43 / 0.15 / **7.2k** | **34.4k (4.8×)** |
| attn | 1.0 | 9.22 / 0.01 / **0.15k** | **45.0k (≫)** |

→ **실측 fused 대비 decoupled partition이 goodput@SLO에서 이긴다**(큰 budget·고부하). fused는 decode를 iteration에 직렬화 → ITL 폭발; partition은 decode를 reserved SM서 동시 실행해 ITL bound. **fused-vs-partition 미결 = partition 승으로 닫힘**(MuxWise/Bullet 이득 실측 재현).

> **⚠ magnitude caveat:** 위는 **비최적화 vanilla SDPA** decode라 decode-attn 절대비용 부풀려짐 → §12-(6)에서 production decode로 닫음.

**(6) ★★ production decode 실측(job 783157·783158, `--opt-decode` = `flash_attn_with_kvcache`) — caveat 닫힘, 답은 *모델 의존*.** vLLM/SGLang가 쓰는 flash-decode로 fused·partition 둘 다 재측정. **decode 가속이 모델 attention 구조에 좌우됨:**

| 모델 | attn 구조 | decode(B64) vanilla(HBM%)→opt(HBM%) | sim goodput (attn b8 λ1): **fused** / protect |
|---|---|--|--|
| **falcon_h1_3b** | **GQA**(kv2), hd128 | 0.97ms(14%)→**0.23ms(56%) 4.1×** | **146k(SLO 1.0)** / 145k — **≈ 동률** |
| **zamba2_2.7b** | **無GQA**(kv32), hd160 | 4.01ms(**66%**)→3.98ms(66%) **1.0×** | **45(붕괴)** / 39k — **partition 압승** |

**★ 통합 framing (SLO 스윕으로 확정): partition 이득 구간 = { SLO_target < baseline(two_stream/fused)의 decode ITL }.** 그 구간에선 baseline이 decode tail로 SLO 위반→붕괴하나 partition은 ITL을 bound해 충족→goodput 승. baseline decode ITL = decode 비용 = **KV 크기 = GQA비**가 결정 → **두 모델 다 이득 구간이 *존재*하고 차이는 *문턱*뿐:**

| 모델 | baseline decode ITL | partition ITL(bound) | 이득 구간(SLO) | falcon λ2 b8 검증 |
|---|--|--|--|--|
| **falcon_h1_3b** (GQA) | ~0.54ms | 0.157ms | **SLO ≲ 0.5ms** (좁음, 공격적 per-token SLO) | SLO0.5→partition 182k vs two_stream 85k(2.1×); SLO1.0→two_stream 승 |
| **zamba2_2.7b** (無GQA) | ~8.8ms | 0.735ms | **SLO ≲ 8.8ms** (넓음, 사실상 모든 현실 SLO) | SLO1.0에서 이미 partition 39k vs two_stream 47(≫) |

즉 **GQA 모델은 빡빡한 SLO에서만, no-GQA 모델은 느슨한 SLO에서도 partition이 이긴다.** (이전에 "falcon은 이득 없음"이라 한 건 SLO=1.0만 봐서 틀렸음 — falcon도 SLO≲0.5ms면 partition 승.)

→ **차이의 진짜 원인 = GQA(=KV 크기), head_dim 아님(BW 계산으로 검증).** falcon은 GQA(kv2)라 KV가 작아 vanilla decode가 **오버헤드-바운드(HBM 14%)** → flash가 오버헤드 걷어 5× 가속 → decode 싸짐 → **fused가 SLO 맞춤 → partition 이득 소멸(fused≈partition)**. zamba2는 **GQA 없음(kv32)이라 KV가 ~20× 커서 vanilla가 이미 메모리-바운드(HBM 66%)** → flash도 *같은 KV를 같은 HBM에서 읽어야* 해 **메모리 벽 못 넘음(1.0×)** → decode 본질적으로 비쌈 → **fused SLO 위반 → partition 압승**. 즉 **PD-mux 이득은 "decode가 SLO 대비 비싼가"에 달렸고, 그건 *모델의 KV 크기(GQA비)*가 결정한다**(no-GQA = 메모리-바운드 decode = 어떤 커널로도 못 줄임). "partition 승"은 *no-GQA 모델의 비싼 decode*에 대한 rescue다.

**결론(production decode 실측 후 — 최종·정직):**
1. **fusion_saving ≈ 0 (실측, 견고)** — vLLM 융합은 decode를 공짜로 못 만든다(mixer 지배).
2. **PD-mux(decoupled partition) 이득은 *조건부·모델 의존*:** decode가 SLO 대비 비싼 regime(느린 커널 OR full-MHA·flash-비친화 구조 + 큰 budget·고부하)에서만 fused/co_schedule을 이긴다. **production decode + GQA 모델(falcon)에선 fused가 충분히 좋아 PD-mux 불필요(≈동률).** full-MHA·odd-head_dim(zamba2)에선 여전히 유효.
3. **two_stream은 부적절 baseline; attn↔ssm *layer-type* 분할은 무관하게 §3.1/§3.2/7B로 여전히 음성(헤드라인 불변).**
4. **메타:** 결론이 useless→wins(two_stream)→loses(낙관fused)→미결→partition승(vanilla fused)→**모델 의존(prod decode)**으로 거듭 뒤집힘 = **PD-mux 판정은 baseline·metric·커널·모델구조 가정에 병적 민감.** 단단한 것: *layer-type 음성* · *fusion-saving≈0* · *"PD-mux는 decode가 SLO 대비 비싼 모델/regime의 rescue지 보편 이득 아님"*.


## 13. Baseline 분류 & layer-aware (방법론 — "무엇과 비교하나")

| baseline/정책 | 무엇 | 답하는 질문 |
|---|---|---|
| **fused** | vLLM/SGLang 기본: prefill+decode를 한 mixed-batch로 융합(decode가 prefill GEMM 편승) | **실제 운영 대비** 이득이 있나 (← *올바른* 기준) |
| two_stream | MPS식 두 동시 스트림, 무분할 동적 공유 | 분할 *벽*이 자유공유보다 나은가 (floor; *약함* — 별도 decode 커널) |
| static-f | 고정 SM 분할 | 정적 분할이 값어치 있나 |
| dynamic-agnostic | 부하-적응 분할(decode floor), layer-type 무관 | 동적 분할이 정적보다 나은가 |
| dynamic-**layer-aware** | 분할 비율을 *prefill layer-type*으로도 조정 | **layer-type이 분할을 개선하나** (= 원래 thesis) |

**핵심(사용자 지적):** vLLM/SGLang은 *fused*를 쓰지 (two_stream 아님). 분할/layer-type을 *제안*하려면 경쟁 baseline은 fused(또는 기존 PD-mux의 agnostic 분할)이지 two_stream(무분할 floor)이 아니다. §12-(3)이 fused를 넣자 분할의 외견상 이득이 사라졌다 — two_stream baseline이 부적절했음을 확인.

**PD-disagg vs PD-mux:** disagg는 prefill/decode를 *다른 GPU*에(inter-GPU, 다른 문제). 우리는 intra-GPU라 비교축은 PD-mux이고, 그 안의 설계 축은 **정적 분할 vs 동적 분할**(§12의 static vs dynamic_protect가 이를 측정 — 동적이 정적을 이김).

**layer-aware vs agnostic (원래 thesis) — hybrid에선 operationally ill-posed:** 하이브리드는 *모든 요청이 attn·ssm 레이어를 다 거친다* → "attn-heavy 요청 vs ssm-heavy 요청"이 없다. layer-type-aware 분할을 하려면 **한 forward 안에서 레이어마다 재분할**해야 하고(per-layer swap), 그 오버헤드는 지배적이다: 38레이어 × ~7.8µs/swap ≈ **300µs** vs prefill ~0.5ms → **~60% 오버헤드**, 그 대가로 얻는 per-phase 최적 분할의 이득은 §3.1상 ≈0(비대칭은 분할의 유효 손잡이가 아님). **∴ layer-aware는 agnostic에 *구조적으로 진다*(오버헤드>이득).** "layer/model-aware 분할"이 말이 되는 건 *서로 다른 모델*이 다른 자원을 쓰는 **multi-model 서빙**(MuxServe류)이지 intra-hybrid layer-type이 아니다. → §3.1을 *분할 위에서* 재확인: **layer-type은 PD-mux를 개선하지 못한다.**

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

## 14. layer-type-AWARE PD-mux — *positive* (원래 thesis의 살아있는 형태)

`run_layer_aware.py`: 하이브리드 forward = n_a attn-layer + n_s ssm-layer로 보고, 전체-모델 decode/prefill을 레이어-타입별 stream 합산으로 계산(per-type LUT=e5_sim_b8_opt). 세 정책:
- `co_schedule`: 전 레이어 two_stream(예약 없음)
- `agnostic_protect`: 전 레이어 decode floor 예약(싼 ssm-decode 레이어에도 SM 낭비)
- **`layer_aware_protect`: 비싼 attn-decode 레이어만 예약, 싼 ssm 레이어는 공유(prefill에 SM 환원)**

**결과 (zamba2_2.7b = 54L = 9 attn + 45 ssm, budget 8, full-model per-token SLO):**

| SLO(TBT) | co_schedule | agnostic | **layer_aware** | 승 |
|---|--|--|--|--|
| 40ms | 7 | **629** | 242 | agnostic (좁은 band: 37.9<SLO<43.5) |
| 50ms | 24 | 629 | **1268** | **layer_aware (2.0×)** |
| 60ms | 126 | 629 | **1268** | **layer_aware (2.0×)** |
| 100ms | 752 | 629 | **1268** | **layer_aware (2.0×)** |

→ **SLO≥50ms에서 layer_aware가 agnostic을 2× 이긴다.** 45개 cheap ssm-decode 레이어에 SM을 낭비 예약하지 않고 prefill에 환원 → **throughput 2×·TTFT 절반**, 비싼 9개 attn-decode만 보호해 decode ITL은 여전히 SLO 충족(43 vs 38ms). **즉 "layer-type awareness가 PD-mux를 개선하는가 = YES"** — 단 조건:
- **temporal 하이브리드 한정**(zamba2: attn/ssm *분리* 레이어). **falcon_h1은 SPATIAL**(매 레이어 attn+ssm *병렬*)이라 attn-layer/ssm-layer를 분리 예약할 수 없어 적용 불가(`attn_frac=ssm_frac=1.0`).
- §13의 "layer-aware = per-layer 재분할 ~60% 오버헤드" 우려는 *microbench(prefill~0.5ms)* 스케일이었음. **full-model+budget8에선 레이어당 비용이 수 ms라 swap 오버헤드(~7.8µs×전이수 ≈ 0.1–0.2ms)는 ~0.4%로 무시 가능** → 이득(2×)이 압도. ∴ §13 정정: layer-aware는 *static SM 분할*로는 죽었지만 *per-layer-type 예약*으로는, *temporal 하이브리드·현실 SLO·full-model* 에서 **살아있다(positive).**

**구분 정리:** layer-type 공간 *분할*(static, coupled forward) = 死(§3.1/§3.2/§13) · layer-type-aware *예약*(temporal 하이브리드, PD-mux 위에서) = **生(2× goodput, 본 §14)**.

## 15. ⚠ 핵심 한계 — 실프레임워크(vLLM/SGLang) 미검증

**이 §11–§14의 모든 결론은 *단일-레이어 microbench LUT로 구동되는 discrete-event 시뮬레이션*이며, 어떤 실제 서빙 프레임워크와도 대조 검증되지 않았다.**
- `fused`("vLLM 기본") baseline은 *vLLM의 모델*이지 실측 vLLM이 아니다. fused *커널*은 측정했으나(`run_fused_step`) 그걸 sim이 *합성*한 것이고, **실제 vLLM/SGLang end-to-end serving을 돌린 적이 없다.** (vLLM·SGLang 둘 다 본 환경에 *미설치*.)
- 게다가 `dynamic_protect`/`layer_aware_protect`(green-ctx 예약)는 **어떤 프레임워크에도 구현돼 있지 않은 연구 아이디어**다 → 직접 비교하려면 *실 엔진 프로토타입을 만들어야* 한다.
- ∴ **§11–§14의 "PD-mux 이득 구간", "layer_aware 2× 승"은 전부 *sim 예측*이며, serving 주장으로 쓰려면 실프레임워크 검증이 선결.** 정직한 다음 단계: **(1)** 이 하이브리드들을 실제 vLLM/SGLang으로 서빙해 TTFT/TBT/throughput 측정 → sim의 fused/co_schedule 예측과 대조(일치하면 sim 신뢰, 불일치면 재보정). **(2)** 일치 시에도 partition/layer_aware는 엔진 프로토타입(green-ctx/MPS 통합)으로 확증 필요. 이건 *시스템 구현* 규모의 별도 작업.
- 본 sim의 가치 = "어디를 측정·구현할지"의 *가설 생성*(이득이 *나타날 수 있는* 영역·조건 식별)이지, *실증*이 아니다. 이번 thread가 보인 가정 민감도(결론 6회 반전)가 그 이유를 정확히 말해준다 — **실측 없이는 PD-mux 판정 불가.**
