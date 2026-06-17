# 프로젝트 종결 보고서 — layer-type 기반 공간 SM 분할 (중단 권고)

작성일: 2026-06-17 · 대상: prefill-layer-alloc (v1 7B saturation → v2 SLM hybrid serving)
상태: E0–E5 완료, G0/G1 게이트 판정 완료, widened sweep 확인 중. **핵심 가설 기각 → 본 라인 종결 권고.**
관련 문서: [v2_report](../workspace/characterization/reports/v2_report.md) · [검수 체크리스트](review_checklist.md) · [추가가치 경로](additional_value_paths.md)

---

## 1. 한 줄 결론

> **"hybrid SSM+Attention 모델에서 layer-type(attn vs ssm) 자원 비대칭을 공간적 SM 분할(Green Context)로 활용한다"는 v2의 핵심 가설은 SLM/A100에서 포괄적으로 기각되었다. 분할은 어떤 prefill×decode 셀·batch·context에서도 단순 동시실행을 못 이긴다. 본 연구 라인은 추가 자원 투입의 한계 이득이 음(陰)이므로 중단을 권고한다.**

---

## 2. 무엇을 물었고, 무엇이 나왔나

| 단계 | 질문 | 결과 | 의미 |
|---|---|---|---|
| E1/G0 | scan은 유의미한 component인가 | OK (peak 62–74%) — **단 고batch에선 GEMM 지배, scan 20–40%로 붕괴** | 분할 동기는 저batch에 한정 |
| E2/G1 | attn-ssm sat_sm 비대칭이 있나 | ASYMMETRY_PRESENT (gap 13–54 SM, CI-분리) — **단 granularity 의존(§5)** | 비대칭은 "서술적 사실"이지 손잡이 아님 |
| E4 | 비대칭을 분할로 회수하나 | prefill-prefill split +1–7%뿐; aware split은 decode를 굶겨 **+60–144% 악화** | 분할은 손해 |
| E5 | serving 전 구간에서 분할이 동시실행을 이기나 | **단 한 셀도 못 이김** (`two_stream/green_ctx` max 0.987/0.996, 1576셀) | **가설 기각** |
| E5 | 그럼 실이득은 어디서 오나 | prefill+decode **overlap ~2.04×**, 분할 없이 co-schedule로 공짜 | 양성 산출(§4) |

---

## 3. 왜 중단하는가 — 메커니즘 수준의 근거

분할/disaggregation 이득의 전제는 **(1) 독립 스케줄 단위 + (2) 병목 상보성**이다. attn↔ssm은 **둘 다 없다**:

- **독립 단위 아님:** attn과 ssm은 *같은 forward pass의 데이터-의존 부분단계*다. "attention만 모은 배치"를 만들 수 없고 레이어마다 activation이 결합되어 disaggregation이 원천적으로 불가.
- **상보성 없음:** prefill 단계에서 둘은 *같은 SM/HBM 파이를 같은 시점에 경쟁*한다. 한쪽이 남긴 자원을 다른 쪽이 채우는 구조가 아니다.

따라서 비대칭(solo 커널의 포화점 차이)은 **활용 도구가 존재하지 않는 축**이다. 게다가 작은 SLM 커널은 full SM 풀에서 이미 잘 겹치므로(co-schedule), 하드 파티션의 경직성은 순손해다. 이건 튜닝으로 메울 수 있는 음수가 아니라 **구조적 음수**다.

> 대조군으로 확인된 양성 축 — prefill↔decode는 (1) 별개 요청의 작업(독립 단위, KV로 깨끗한 handoff) + (2) decode가 BW-bound로 SM을 놀려 prefill이 채움(상보성) → ~2× overlap. **그러나 이 이득조차 공간 분할이 아니라 HW 공동스케줄링이 더 잘 회수한다.** 즉 "분할"이라는 메커니즘 자체가 이 문제군에서 설 자리가 없다.

---

## 4. 회수 가능한 산출물 (중단 ≠ 손실)

음성 결과지만 다음은 그대로 가치가 있고, 종결과 무관하게 보존/활용 가능하다:

1. **발표 가능한 음성 특성화 결과.** "SLM hybrid serving에서 layer-type SM 분할은 무용, co-schedule이 정답"은 characterization 논문/리포트로 성립. 분할을 시도하려는 후속 연구의 시간을 아껴줌. (SSM-Scope 류 ISPASS characterization 트랙과 정렬.)
2. **양성 권고:** prefill+decode를 공유 SM에 **그냥 co-schedule**(MPS/multi-stream) → 저~중 decode batch에서 1.2–2.0×, 분할/aware 할당 불필요. 즉시 적용 가능한 운영 지침.
3. **비대칭의 비-할당 용도:** kernel fusion/튜닝 신호(ssm가 작아 일찍 포화), 모델↔하드웨어 매칭, layer별 quant/offload 우선순위. (할당이 아닌 곳에서 쓰임.)
4. **재사용 가능한 측정 프레임워크:** measured/derived/metadata 라벨 강제, bootstrap CI 포화점, subprocess 격리, BandwidthEstimator, E0 해석적 wave 예측, 게이트 자동판정(`adjudicate.py`). 다음 프로젝트의 인프라.
5. **음성을 강하게 만든 방법론적 발견:** granularity confound(SSM chunked vs Attn full-seq)가 비대칭을 부풀린다는 점, BW 절대%가 양방향으로 비신뢰라는 점 — 후속자에게 직접적 경고.

---

## 5. 종결 시 명시해야 할 한계 (정직성)

중단 보고서로 닫더라도 아래는 **검증되지 않은 채 남는다.** "분할 무용"을 무조건적 결론으로 과대주장하면 안 된다:

- **7B 미검증 (가장 큰 구멍):** `models.yaml:88` "7B entries are kept but **unused**". v2_report가 "결정적"이라 부른 7B 회귀는 **한 번도 실행되지 않았다.** 큰 커널이 SM을 포화시키면 분할이 의미를 가질 *여지*는 닫지 못함. → 결론 스코프 = **"SLM/A100 한정"**.
- **진짜 prefill 미검증:** E5는 microbench(prefill chunk=1, scan-only, decode step=1). GEMM·다중 chunk 포함 실제 prefill은 `--prefill-layer ssm_full`로만 점검 가능하며 미수행([검수 A2](review_checklist.md)).
- **비대칭의 robustness:** matched-granularity에서 zamba 20셀 중 9셀(REDUCED 7+ELIMINATED 2)에서 비대칭 약화/소멸([검수 A3](review_checklist.md)). "비대칭 실재"는 격자 의존.
- **HW 한정:** A100-SXM4 단일 디바이스. MPS 별도 프로세스가 아닌 multi-stream 근사(§6).

---

## 6. 권고

1. **본 라인(layer-type 공간 SM 분할) 종결.** 추가 분할-실험에 SXM4 자원 투입 금지. 한계 이득 음수, 메커니즘 수준에서 닫힘(§3).
2. **종결 전 [검수 체크리스트](review_checklist.md)의 P0 5건 통과 확인** — 특히 A2(ssm_full 진짜-prefill 1회)와 C1(7B 미실행 스코프 명기). 이것만 통과하면 "SLM에서 음성 확정"으로 깨끗이 닫힌다.
3. **단, 닫기 전 [추가가치 경로](additional_value_paths.md)를 1회 검토.** 7B 회귀(Path 1)는 *닫는 데도 필요한* 결정적 데이터점이며(음성이면 결론을 크기-무관으로 격상, 양성이면 라인 부활), 비용이 크지 않다. prefill/decode overlap 재프레이밍(Path 2)은 음성 라인을 양성 기여로 전환하는 별개 출구다.
4. **v1/v2 동결:** `archive/v1_7b_saturation/` 유지. v2 산출물(results_v2, figures, verdicts) 커밋 후 동결.

> **결정 트리:** 검수 P0 전부 통과 + 7B 미실행 수용 → **여기서 종결(음성 특성화로 발표).** 7B 1회라도 돌릴 여력 → `additional_value_paths.md` Path 1 먼저 → 그 결과로 최종 종결/전환.
