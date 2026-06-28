# 최종 결과 (논문용) — Layer-type-aware resource allocation in hybrid SSM+Attention serving

> 본 문서는 **확정된 최종 결론만** 담는다(중간 변화·수정 이력 제외). 상세·재현은 하위 보고서 참조.
> 환경: NVIDIA A100-SXM4-80GB(108 SM). 모델: Zamba2 1.2/2.7/7B(temporal 하이브리드), Falcon-H1 3/7B(spatial 하이브리드). 측정: 단일-커널~full-model 마이크로벤치(E1–E7) + LUT-구동 큐 시뮬레이터(E6) + 실 vLLM 0.22.1 서빙.

---

## 0. 핵심 주장 (3줄)

1. **정적 layer-type 공간 분할(한 forward 내 attn/ssm SM 쪼개기)은 서빙을 개선하지 않는다.**
2. **layer-type을 인지한 *decode SM 예약*(PD-multiplexing 위)은 temporal 하이브리드의 goodput을 layer-agnostic 예약 대비 1.37–2.02× 개선한다(모델 1.2B–7B 전 구간).**
3. **이 이득은 temporal 하이브리드에 고유하다 — spatial 하이브리드(매 레이어 attn+ssm 병렬, GQA로 attn-decode 저렴)에는 적용되지 않는다.**

## 1. 정적 layer-type 공간 분할 = 무효

attn 레이어와 ssm 레이어에 SM을 정적으로 분할하는 Green-Context 분할은 동적 co-scheduling(two-stream)을 **전 sweep(1576 셀)에서 한 번도 이기지 못한다**(최대 SLO-attain 0.987/0.996). 원인: attn(compute-bound)·ssm(memory-bound) 자원 비대칭은 존재하나 그것이 *분할의 lever가 아니다* — 한 자원을 나눠도 다른 축 경합이 풀리지 않으며(compute⊥memory), 분할은 양쪽을 함께 굶긴다.

## 2. layer-type-aware decode 예약 = 유효 (temporal 하이브리드)

PD-multiplexing(prefill·decode 동시 실행) 하에서 **decode가 비싼 attn 레이어에만 SM을 예약하고, decode가 저렴한 ssm 레이어에서는 SM을 prefill에 환원**하는 정책(layer-aware reservation)이 layer-agnostic reservation(전 레이어 예약)을 이긴다.

**goodput@SLO (현실적 per-token TBT, layer-agnostic 예약 대비):**

| 모델 | 구성(attn+ssm) | **layer_aware / agnostic** |
|---|---|---|
| Zamba2-1.2B | 6 + 32 | **1.37×** |
| Zamba2-2.7B | 9 + 45 | **2.02×** |
| Zamba2-7B | 13 + 68 | **1.82×** |

**메커니즘:** agnostic는 모든 레이어(저렴한 ssm 포함)에 decode SM floor를 예약해 prefill을 상시 굶긴다(goodput agnostic < co-schedule). layer-aware는 비싼 attn 레이어만 보호하고 ssm 레이어 SM을 prefill에 환원 → throughput↑·TTFT↓, decode ITL은 SLO 내 유지. 강도는 ssm-decode floor가 GPU에서 차지하는 비중에 비례(2.7–7B에서 큼).

## 3. 적용 범위: temporal-only — 두 전제

layer-aware lever는 두 조건을 함께 요구하며, 이는 하이브리드 아키텍처 부류로 갈린다:

| 전제 | TEMPORAL (Zamba2) | SPATIAL (Falcon-H1) |
|---|---|---|
| ① 레이어-타입 분리 (시간축 스케줄 가능) | ✅ attn/ssm 분리 레이어 | ❌ 매 레이어 attn+ssm 병렬 |
| ② 비싼 attn-decode 소수 (보호 가치) | ✅ no-GQA, 0.59–0.95 ms/layer | ❌ GQA 5×, 0.056 ms/layer (decode의 ~10%) |
| **적용성** | ✅ **1.37–2.02×** | ❌ **N/A** |

**아키텍처 공-설계 통찰:** temporal 하이브리드는 attention을 *드물게·비싸게*(no-GQA), spatial 하이브리드는 *어디서나·저렴하게*(GQA) 배치한다. 이 설계 선택이 layer-type 스케줄링의 가치를 결정한다.

## 4. 프로덕션 fused(vLLM) 대비

- **fusion_saving ≈ 0** (실측): prefill+decode를 한 forward로 fuse해도 분리 실행 대비 작업량 절감이 없다(mixer/KV traffic이 지배).
- **fused는 decode를 prefill forward에 결합** → 부하 하 decode ITL이 팽창. 실 vLLM TPOT가 부하↑에 따라 z2.7B 12→80 ms로 증가하여 이를 확인.
- layer-aware는 decode를 예약 SM에 분리 → decode ITL을 **fused 대비 4–6× 낮게** bound(시뮬레이션; 비율은 calibration 불변).

## 5. 실 vLLM 검증 (decode-cost 모델)

vLLM 실서빙(Zamba2 1.2/2.7/7B, Falcon-H1 3B) 대비:

| | 검증 결과 |
|---|---|
| 모델 간 decode ITL 비율 | 시뮬레이션 ≈ 실측 (~1.3× 수준) |
| 크기-스케일링 | sim decode 비 1.00/1.52/2.62 ≈ 실측 포화 TPOT 비 1.00/1.63/2.57 |
| full-model decode 구성 | ssm 지배(attn-decode는 temporal 19–25%, spatial ~10%) |

**실측 포화 p99 TPOT / throughput:** Zamba2-1.2B 67 ms/1413 · 2.7B 109 ms/870 · 7B 172 ms/346 · Falcon-H1-3B 92 ms/1074.

## 6. 부하 의존성 (확정)

- **시퀀스 길이:** attn-decode는 KV 전체 읽기로 O(L)(context↑서 비용↑); ssm-decode는 고정 state로 **O(1)**(context 무관, HBM BW util ~0.01%). 하이브리드의 근본 비대칭.
- **배치:** decode를 solo 속도로 유지하는 floor SM이 batch↑서 증가(고배치엔 전체 GPU 필요).
- **모델 크기:** decode 비용·보호 floor 증가; layer-aware 강도는 1.2B<2.7B>7B(2.7B 최대, ssm floor 비중 차이).
- **prefill SM-민감도:** 크기-불변(ssm ~2×, attn ~7× @108→14 SM).
- 전 모델 **DECODE/PREFILL < 1**(prefill-heavy): z1.2/2.7/7B 0.62/0.59/0.39, falcon 0.54/0.56.

## 7. 한계 (확정 스코프)

- **정량 framework 비교는 시뮬레이션만으론 불충분.** sim의 절대 fused decode ITL은 no-GQA 모델에서 vLLM 대비 1.6–2.3× 과대(크기↑서↑), throughput 0.8–0.9× 과소. *상대 추세·정책 간 ITL 비율·방향*은 실측과 정합하나, **"layer-aware vs vLLM N× goodput"의 정량 수치는 실엔진 프로토타입이 선결**(green-context 커널로 4 정책 실측).
- 측정 범위: A100-SXM4 단일 GPU, ≤7B, 합성 워크로드(prompt 2048/output 128). >7B·다른 GPU·실 트레이스 미측정.
- 비교군 중 **deployed 시스템은 fused(vLLM)뿐**; layer-agnostic/layer-aware reservation은 연구 설계(어떤 프레임워크에도 미구현).

## 8. 헤드라인 표 (한 장)

| 항목 | 결과 |
|---|---|
| 정적 layer-type 공간 분할 | 무효 (co-schedule 못 이김, 전 sweep) |
| layer-type-aware decode 예약 (vs agnostic) | **1.37× (1.2B) · 2.02× (2.7B) · 1.82× (7B)** |
| 적용 범위 | temporal 하이브리드 한정 (분리 + 비싼 no-GQA attn 필요) |
| fused 대비 decode ITL | 4–6× 낮음 (decode 분리·예약) |
| fusion_saving | ≈ 0 (실측) |
| decode 비대칭 | attn O(L)·memory-bound vs ssm O(1)·BW~0% |
| vLLM 검증 | decode-cost 모델의 *모델 간 비율·크기 스케일링* 정합 |

---
*상세: [layer_aware_benefit_report](layer_aware_benefit_report.md) · [temporal_vs_spatial_hybrid](temporal_vs_spatial_hybrid.md) · [framework_comparison](framework_comparison.md) · [vllm_validation](vllm_validation.md) · [metrics_and_load_dependence](metrics_and_load_dependence.md) · [partition_engine_design](partition_engine_design.md)*
