아래 전체를 복사해 Claude 웹(claude.ai)에 붙여넣으면, 파일·클러스터 접근 없이도 논문 작성을 이어갈 수 있습니다.

---

# [붙여넣기용 프롬프트]

나는 시스템 논문을 쓰고 있어. 아래가 프로젝트의 **확정된 최종 결과**와 작성 맥락이야. (이전에 여러 번 수정·반전이 있었지만, 아래 수치/결론만이 최종이고 이것만 사용해줘. 여정·반전은 논문에 넣지 않아.)

## 주제
하이브리드 SSM+Attention LLM 서빙에서 **layer-type-aware decode SM 예약(reservation)** 시스템. 제안 시스템 자체가 기여이고, "실패→재정식화" 서사로 쓰지 않아 — 재정식화된 예약 시스템을 직접 제시·평가하는 구조야.

## 환경·모델
- GPU: NVIDIA A100-SXM4-80GB (108 SM).
- **Temporal(sequential) 하이브리드 = Zamba2 1.2/2.7/7B**: attn 레이어와 ssm 레이어가 *깊이축에서 분리*. attention은 GQA 없음(q=kv heads) → KV 큼 → attn-decode 비쌈.
- **Spatial(parallel) 하이브리드 = Falcon-H1 3/7B**: 매 레이어가 attn+ssm *병렬*. attention은 GQA 5×(q10/kv2) → attn-decode 쌈.
- 방법: 단일~full-model 마이크로벤치 + LUT 구동 큐 시뮬레이터 + 실 vLLM 0.22.1 서빙.

## 3대 핵심 주장
1. **정적 layer-type 공간 분할(한 forward 내 attn/ssm SM 쪼개기)은 서빙을 개선하지 않는다.** (설계 근거 1–2문장으로만 언급; results 비트 아님.)
2. **layer-type을 인지한 decode SM *예약*(PD-multiplexing 위)은 temporal 하이브리드의 goodput을 layer-agnostic 예약 대비 1.37–2.02× 개선한다(모델 1.2B–7B 전 구간).**
3. **이 이득은 temporal 하이브리드에 고유하다 — spatial 하이브리드(매 레이어 병렬, GQA로 attn-decode 저렴)에는 적용되지 않는다.**

## 시스템(메커니즘)
PD-multiplexing(prefill·decode 동시 실행) 하에서, decode가 비싼 **attn 레이어에만 SM floor를 예약**하고 decode가 저렴한 **ssm 레이어에서는 SM을 prefill에 환원**한다. agnostic(전 레이어 예약)은 싼 ssm 레이어에서도 prefill을 굶겨 throughput을 떨어뜨리지만(agnostic < co-schedule), layer-aware는 prefill을 환원해 throughput↑·TTFT↓하면서 decode ITL은 SLO 내로 유지한다.

## 핵심 수치 (확정)
**(1) layer_aware / agnostic goodput@SLO (temporal):** Zamba2 1.2B=1.37×, 2.7B=2.02×(peak), 7B=1.82×.

**(2) 적용범위 판별 — per-layer attn-decode 비용(@batch8):** Zamba2(no-GQA) 0.59–0.95 ms/layer (decode의 19–25%), Falcon-H1(GQA) 0.056 ms/layer (decode의 ~10%). 두 전제: ①레이어-타입 분리(temporal ✓ / spatial ✗) ②비싼 no-GQA attn-decode(temporal ✓ / spatial ✗). 둘 다 있어야 lever 작동.

**(3) vs fused(vLLM 기본):** fused는 decode를 prefill forward에 결합 → 부하 하 decode ITL 팽창. layer-aware는 decode를 예약 SM에 분리 → decode ITL을 fused 대비 **4–6× 낮게** bound. `fusion_saving≈0`(실측: prefill+decode를 한 forward로 fuse해도 작업량 절감 없음).

**(4) 실 vLLM 측정(saturated p99 TPOT / output throughput):** Zamba2-1.2B 67ms/1413 · 2.7B 109ms/870 · 7B 172ms/346 · Falcon-H1-3B 92ms/1074.

**(5) decode-cost 모델의 vLLM 검증:** 모델 간 decode ITL 비율(sim≈실측 ~1.3×), 크기-스케일링(sim decode 비 1.00/1.52/2.62 ≈ 실측 포화 TPOT 비 1.00/1.63/2.57). full-model decode는 ssm 지배(attn 비중 temporal 19–25%, spatial ~10%).

**(6) 부하 의존성:** attn-decode는 KV 전체 읽기로 **O(L)**(context↑서 비용↑)·메모리바운드; ssm-decode는 고정 state로 **O(1)**(context 무관, HBM BW util ~0.01%). prefill SM-민감도는 **크기 불변**(ssm ~2×, attn ~7× @108→14 SM). 전 모델 **DECODE/PREFILL < 1**(prefill-heavy): Zamba2 0.62/0.59/0.39, Falcon 0.54/0.56.

## 한계 (반드시 유지 — reviewer가 물을 부분)
- **정량 framework 비교는 시뮬레이션만으론 불충분.** sim의 절대 fused decode ITL은 no-GQA 모델에서 vLLM 대비 **1.66/1.64/2.34×(1.2/2.7/7B) 과대**(크기↑서↑), throughput 0.8–0.9× 과소; Falcon은 GQA로 1.09× 양호. *상대 추세·정책 간 ITL 비율(4–6×)·크기 스케일링·방향*은 실측과 정합하나, **"layer-aware vs vLLM N× goodput"의 정량 수치는 실엔진 프로토타입(green-context 커널로 4 정책 실측)이 선결.**
- 측정 범위: A100-SXM4 단일 GPU, ≤7B, 합성 워크로드(prompt 2048/output 128, request-rate 2/8/inf). >7B·다른 GPU·실 트레이스 미측정.
- 비교군 4개(fused / co_schedule=two-stream / agnostic-reservation / layer-aware-reservation) 중 **deployed 시스템은 fused(vLLM)뿐**; 나머지는 연구 설계.

## Results 흐름 (시스템 중심, S1→S7)
- **S1** 동기: per-layer decode 비대칭(레이어별 보호 가치 다름) — 그림 `temporal_vs_spatial`(per-layer 구조)
- **S2** 시스템: layer-type-aware 예약 + 메커니즘 — 그림 mechanism 도식
- **S3** 주결과: goodput@SLO(연속 SLO 스윕) — 그림 goodput-vs-SLO 선그래프
- **S4** 일반성: 크기 1.2B–7B(1.37/2.02/1.82×) — 그림 size-trend 막대
- **S5** 적용범위: temporal-only(두 전제) — 그림 `temporal_vs_spatial` + prefill SM-민감도
- **S6** 프로덕션 대비: vs fused(vLLM), decode ITL 4–6× — 그림 4-way ITL 막대
- **S7** 검증·한계: vLLM이 decode 모델 검증(상대), 절대는 directional — 그림 calibration(sim vs 실측 + 크기스케일링 중첩선)

## 그림 자산 (이미 제작됨, 형태 비판 포함)
1. goodput-vs-SLO(선+우위구간 음영) — 주결과, 형태 최적
2. mechanism(54-레이어 SM 배분 도식) — 시스템 설명
3. size-trend(la/agnostic 막대 + 정책별 goodput) — 크기 일반화
4. temporal_vs_spatial(레이어별 decode 막대: temporal=비싼 attn 스파이크 분리 / spatial=병렬·GQA로 작음) — 적용범위, 형태 우수
5. prefill_sm_sensitivity(연속 SM 스윕, 3모델 중첩선=크기불변) — 전제② 근거, 형태 최적
6. framework_comparison(4정책 decode ITL 막대 + vLLM 앵커) — 단 fused가 y축 지배 → **log축 권고**
7. vllm_calibration(sim vs 실측 막대 + 크기스케일링 2선중첩) — 검증
- 형태 권고: throughput/TTFT/ITL을 한 패널에 막대+쌍축으로 욱여넣은 tradeoff 그림은 **분리/정규화**; 개요 dashboard·applicability 만화는 본문 제외(teaser/부록).
- 권장 추가 그림: **1.2/2.7/7B goodput-vs-SLO 중첩 선그래프**(S3+S4 한 장 통합).

## 내가 원하는 작업
[여기에 구체적으로 적어줘. 예: "위 흐름(S1–S7)대로 Results 섹션 초안을 써줘 / Abstract+Intro를 써줘 / 위 한계를 Limitations 절로 정리해줘 / S3+S4 통합 그림의 캡션과 본문 서술을 써줘."]

---

(끝. 위 [붙여넣기용 프롬프트] 블록 전체를 복사해 사용. 클러스터/파일 접근이 필요한 작업—새 측정, 프로토타입 구현—은 웹에서 불가하니 이 환경에서 계속할 것.)
