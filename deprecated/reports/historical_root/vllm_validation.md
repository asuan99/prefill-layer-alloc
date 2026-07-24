# 실프레임워크(vLLM) 검증 — sim의 fused baseline 대조

작성일: 2026-06-21 · vLLM 0.22.1 (격리 venv `/scratch/ehmoon/whlee/vllm_venv`, torch 2.11+cu130) · A100-SXM4-80GB(gpu40)
관련: [queue_simulator_design §15](queue_simulator_design.md) · [closure note 5](project_closure_report.md)

> 사용자 지적("실 vLLM/SGLang 비교가 안 됐다")에 따른 검증. **결론을 먼저: sim의 *full-model* decode 모델은 (방향·비율·포화 magnitude) 검증됐으나, 문서가 헤드라인으로 쓴 "GQA가 PD-mux 이득 구간 폭을 결정 → zamba2 넓음/falcon 좁음(16×)" 주장은 *실측이 반증*한다. 그 16×는 per-attn-layer KV 커널 비율이고, full-model decode는 ssm 레이어가 지배해 두 모델이 ~1.3× 차이밖에 안 난다.**

## 1. 셋업 (전부 격리, 프로젝트 venv 불간섭)

- vLLM 0.22.1 (`cp38-abi3` wheel, Python 3.14). `Zamba2ForCausalLM`/`FalconH1ForCausalLM` 둘 다 레지스트리 지원.
- weight: `Zyphra/Zamba2-2.7B`, `tiiuae/Falcon-H1-3B-Base` (HF).
- 서버: `vllm serve --enable-chunked-prefill --max-num-batched-tokens 2048 --max-model-len 4096 --gpu-memory-utilization 0.85`. 클라이언트: `vllm bench serve --dataset-name random --random-input-len 2048 --random-output-len 128 --num-prompts 120 --request-rate {2,8,inf}`.
- 트러블슈팅: (a) offline `.metrics`가 v1엔진서 None → online bench로 전환. (b) `prometheus_fastapi_instrumentator`가 starlette `_IncludedRouter`에서 `.path` AttributeError로 모든 HTTP 요청 크래시 → `routing.py`를 `getattr` 가드로 패치(venv 내, 원래도 None 반환하던 함수라 무해).

## 2. 실측 (vLLM, in=2048 out=128, budget=2048)

| model | RR | req/s | out tok/s | TTFT med/p99 (ms) | **TPOT med/p99 (ms)** | ITL med/p99 (ms) |
|---|--|--|--|--|--|--|
| zamba2_2.7b | 2 | 1.96 | 251 | 182 / 2020 | **11.9 / 21.9** | 8.7 / 85 |
| zamba2_2.7b | 8 | 6.25 | 800 | 606 / 1455 | **66.1 / 93.9** | 41 / 125 |
| zamba2_2.7b | inf | 6.80 | 870 | 6375 / 13838 | **79.9 / 108.6** | 50 / 131 |
| falcon_h1_3b | 2 | 1.97 | 252 | 141 / 485 | **8.8 / 18.6** | 7.2 / 68 |
| falcon_h1_3b | 8 | 7.15 | 915 | 315 / 770 | **41.1 / 58.8** | 18 / 98 |
| falcon_h1_3b | inf | 8.39 | 1074 | 5350 / 11828 | **64.2 / 92.3** | 32 / 116 |

## 3. sim 예측 vs 실측

**(a) full-model decode ITL = Σ_layers(per-layer solo_decode)** [sim e5_sim_b8_opt]:

| | layers | B=1 | B=8 | B=64 | B=256 |
|---|--|--|--|--|--|
| zamba2_2.7b | 9a+45s | 24.2 | 28.6 | 73.9 | 253 ms |
| falcon_h1_3b | 32a+32s | 18.4 | 18.2 | 36.1 | 124 ms |
| **ratio z/f** | | **1.32×** | **1.57×** | **2.05×** | 2.04× |

**실측 TPOT ratio z/f: 1.35× (RR2), 1.61× (RR8), 1.24× (inf).** → **sim full-model ratio(1.3–2.0×) ≈ 실측(1.2–1.6×). 방향·비율 일치 ✓.** 포화 magnitude도: sim B64 zamba2 74ms / falcon 36ms vs 실측 inf 80ms / 64ms — 같은 자릿수(저부하선 sim이 ~2× 과대, unfused per-layer 커널 합산 오버헤드 때문, 고부하서 수렴).

**(b) per-LAYER attn-decode (GQA 효과 — 문서가 헤드라인으로 쓴 양):**

| B | zamba2(kv32) | falcon(kv2) | ratio |
|--|--|--|--|
| 8 | 0.591 | 0.056 | **10.6×** |
| 64 | 4.156 | 0.197 | **21.1×** |
| 256 | 15.72 | 0.725 | **21.7×** |

**(c) TTFT·throughput 동역학:** TTFT가 RR↑에 폭증(182→606→6375ms zamba2; 141→315→5350 falcon) = sim 큐 동역학과 정성 일치 ✓. 포화 throughput zamba2(870) < falcon(1074) = no-GQA decode가 더 비싸다는 sim 방향과 일치 ✓.

## 4. 검증 판정

| sim 주장 | 실측 결과 |
|---|---|
| decode 비용 zamba2 > falcon (no-GQA가 비쌈) | **✓ 검증** (TPOT/ITL/throughput 전부) |
| full-model decode ITL magnitude(포화 ~수십 ms) | **✓ 검증** (sim ~74/36ms ≈ 실측 80/64ms) |
| full-model decode ITL ratio z/f ≈ 1.3× | **✓ 검증** (실측 1.2–1.6×) |
| 저부하 decode ITL 절대값 | **△ sim이 ~2× 과대** (unfused 커널 합산; SLO 문턱은 상한으로 취급) |
| **"GQA가 이득 구간 폭 결정 → zamba2 넓음(8.8ms)/falcon 좁음(0.54ms), 16×"** (§12(6)/closure 5) | **✗ 반증.** 16×는 *per-attn-layer KV* 비율. **full-model decode는 ssm 레이어가 지배 → 두 모델 ~1.3× 차이뿐** → 이득 구간 폭은 *유사*하지, 16× 다르지 않다. |

## 5. 핵심 교훈 (실측이 아니면 못 잡았을 것 — 사용자 지적의 정당성)

- **decode ITL을 결정하는 건 KV-read(GQA)가 아니라 *모델 전체 weight·ssm-state memory traffic*다.** GQA는 attn-KV 커널을 10–21× 줄이지만, 그 커널은 full-model decode의 작은 일부(zamba2는 45/54가 ssm, falcon은 매 레이어가 ssm 포함). 그래서 GQA 효과가 ~1.3×로 희석된다.
- 따라서 **"PD-mux 이득 구간이 GQA로 wide/narrow 갈린다"는 모델-의존 결론(§12(6))은 single-layer 프레이밍의 artifact**였고, 실측은 두 모델 decode ITL이 *비등*함을 보인다 → 이득 구간 존재 여부의 모델 차이는 거의 없다(있다면 ~1.3× 수준).
- 반대로 **full-model sim(run_layer_aware §14)은 magnitude·ratio가 실측과 맞아 신뢰**된다 → layer_aware §14 결과(temporal 하이브리드서 prefill 환원으로 throughput↑)의 토대는 유효.

## 6. 남는 한계 (이 검증이 *못* 한 것)

- **partition/layer_aware_protect는 vLLM에 구현이 없어 직접 비교 불가.** 이번 검증은 *fused(=vLLM 기본) baseline의 현실성*만 확인했다 — 그게 핵심 토대지만, green-ctx/MPS 예약을 실엔진에 통합한 프로토타입 없이는 PD-mux/layer_aware의 *실증*은 여전히 미완(§15).
- 단일 (in,out)=(2048,128), 120 req, 3 rate, fp16, default Mamba2 커널("sub-optimal" 경고). 일차 검증이지 exhaustive sweep 아님.
- SGLang은 미수행(vLLM과 동급 결론 기대; 둘 다 fused mixed-batch 기본).

## 7. 전 크기 vLLM 실측 추가 (z1.2b·z7b, job 799168) — size-scaling 검증

z2.7b·falcon_3b(§2)에 더해 **zamba2 1.2b·7b 실측**으로 전 크기 커버:

| model | TPOT med/p99 (RR2/8/inf) | throughput(sat) |
|---|---|---|
| zamba2_1.2b | 5.6/22 · 26/36 · 47/67 | 1413 |
| zamba2_2.7b | 12/22 · 66/94 · 80/109 | 870 |
| zamba2_7b(Instruct) | 64/95 · 155/173 · 154/172 | 346 |
| falcon_h1_3b | 8.8/19 · 41/59 · 64/92 | 1074 |

**검증된 것:**
- **decode 비용의 크기-스케일링 일치:** 포화 p99 TPOT 정규화비 z1.2b 1.00 / z2.7b 1.63 / z7b 2.57 ≈ sim decode_total 비 1.00/1.52/2.62 → **sim이 *크기에 따른 decode 증가*를 정확히 포착**(절대값은 아래대로 과대지만).
- **z7b는 fused로 매우 느림**(TPOT 172ms·throughput 346 tok/s) → layer_aware의 decode-ITL 단축 여지가 *가장 큰* 모델.

**calibration(sim fused ITL ÷ vLLM 포화 p99 TPOT):** z1.2b **1.66×** · z2.7b **1.64×** · z7b **2.34×** · falcon **1.09×**. → 과대분이 *no-GQA attn-decode*에서 크고 **모델 크기와 함께 증가**(falcon은 GQA로 양호). throughput은 0.79~0.90× 일관 과소. ∴ **sim 절대값은 모델·크기 의존적으로 어긋남 → 정량 framework 비교는 실엔진 프로토타입 필요**([framework_comparison](framework_comparison.md)). 그림: `figures/vllm_calibration.png`.
