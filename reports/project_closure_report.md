# 프로젝트 종결 보고서 — layer-type 기반 공간 SM 분할 (중단 권고)

작성일: 2026-06-17 · 대상: prefill-layer-alloc (v1 7B saturation → v2 SLM hybrid serving)
상태: E0–E5 완료, G0/G1 게이트 판정 완료, widened sweep 확인 중. **핵심 가설 기각 → 본 라인 종결 권고.**
관련 문서: [v2_report](../workspace/characterization/reports/v2_report.md) · [검수 체크리스트](review_checklist.md) · [추가가치 경로](additional_value_paths.md)

---

## 1. 한 줄 결론

> **"hybrid SSM+Attention 모델에서 layer-type(attn vs ssm) 자원 비대칭을 공간적 SM 분할(Green Context)로 활용한다"는 v2의 핵심 가설은 SLM/A100에서 포괄적으로 기각되었다. 분할은 어떤 prefill×decode 셀·batch·context에서도 단순 동시실행을 못 이긴다. 본 연구 라인은 추가 자원 투입의 한계 이득이 음(陰)이므로 중단을 권고한다.**

> **(2026-06-17 추가 검토)** G1이 sat_sm 비대칭을 gate 술어로 쓴 것 자체가 잘못이라는 비판(§3.1)을 반영해도 — **종결 권고는 유지, 오히려 강화**된다. 올바른 술어(공유 하 회수 가능 slack)는 E4/E5가 이미 음성으로 측정했기 때문이다.

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

### 3.1 G1 술어가 잘못됐다는 비판을 반영해도 — 중단 권고 불변, 오히려 강화

E2/G1이 "attn-ssm sat_sm 비대칭이 있는가"를 gate 술어로 쓴 것은 **category error**다: sat_sm 비대칭은 *격리 실행한 solo 커널*의 occupancy 속성**(a)**이고, partition 이득을 결정하는 것은 *공유 실행 하에 동적 co-schedule이 회수하지 못하는 slack(파괴적 간섭)이 있는가***(b)**이다. 둘은 독립이다 — solo 포화점은 공유 하 상호작용에 대해 아무 정보도 주지 않는다. (PD-mux 이득도 (a)가 아니라 (b)에서 나오며, PD의 sat_sm 차이는 병목-종류 차이의 *증상*이지 원인이 아니다.) `g1_verdict.json`의 *"SSM frees SMs for attention → proceed to E4"*가 바로 (a)→(b) 비약을 담은 문장이다. v2_report §1이 RQ 재설정을 "부분적 과오"라 인정한 것의 정확한 정체가 이 술어 mis-specification이다.

**이 비판은 중단 권고를 약화시키지 않는다. 강화한다:**
- **올바른 술어(b)는 이미 측정됐다.** E4/E5의 `two_stream`(동적 공유) vs `green_ctx`(정적 분할) 비교가 정확히 "공유가 남기는 회수 가능 slack이 있는가"이며, SLM 전 1576셀에서 No(`two_stream/green_ctx` < 1.0). 잘못된 술어로 진입했으나 **올바른 술어의 답까지 음성으로 확보**돼 있다.
- 따라서 음성 결과엔 *"엉뚱한 걸 쟀으니 다시 재면 이득이 보일지 모른다"*는 escape hatch가 없다. 비대칭이 격자에 따라 약화/소멸하든 유지되든([검수 A3](review_checklist.md)) 결론과 무관 — 비대칭은 애초에 lever가 아니었고, lever였을 (b)는 이미 닫혔다.
- 남는 유일한 미검증 영역은 **(b)가 다른 regime(7B 큰 커널 / GEMM-heavy 진짜 prefill)에서 양수가 되는가**이다. 이는 "비대칭이 scale에서 살아나는가"가 아니라 **"공유 하 파괴적 간섭이 커널이 커지면 생겨 격리가 회수하는가"**라는 올바르게 재서술된 질문이며, 이미 [추가가치 Path 1·3](additional_value_paths.md)에 잡혀 있다. 술어 비판은 이 잔여 경로를 *확장*하지 않고 *정확히 재명명*할 뿐이다.

**판정:** 술어 비판 반영 후에도 — layer-type 공간 SM 분할(asymmetry 기반 thesis)은 **종결 권고 유지**. 비판은 오히려 음성 결과를 "예상된 비약의 귀결"로 격상시켜 종결 근거를 강화한다.

### 3.2 더 깊은 root-cause — execution 분할은 memory 분리의 proxy가 될 수 없다

"왜 우회로가 없었나"의 메커니즘. 원래 베팅은 *"execution path를 분할하면 memory abstraction을 새로 짓지 않고도 memory-execution 분리를 얻는다"*(= compute 분할이 memory 분리의 공짜 proxy)였으나, 세 층위에서 동시에 무너진다:

1. **하드웨어: compute partition ⊥ memory partition.** 본 연구가 쓴 도구(Green Context, MPS active-thread%)는 **SM(연산 유닛)만 분할**한다. A100에서 **L2 캐시·HBM 대역폭은 분할되지 않고 전부 공유** — SSM/Attn에 SM을 갈라줘도 둘은 같은 HBM 파이·같은 L2에서 메모리를 빨아들인다. 실행을 갈라도 memory subsystem은 한 덩어리다. (compute+memory를 같이 자르는 유일한 기제는 **MIG**이나, GPU를 분리 인스턴스로 쪼개는 것이라 단일 forward pass가 두 슬라이스에 걸쳐 activation을 주고받을 수 없음 → layer-type intra-model 분리엔 granularity가 틀림.)
2. **분리할 동시성 부재.** single forward pass 안에서 attn/ssm 메모리 트래픽은 시간 순차(데이터-의존 부분단계)라 동시 경쟁하는 두 스트림이 아니다 — 공간 분할로 격리할 대상이 애초에 없다. (serving에서 실제 동시 흐름은 prefill↔decode이고, 그건 generic 축이다 → §4.2.)
3. **스택 레이어 불일치.** 원한 memory 이득은 *state lifecycle*(footprint·evict·recompute·share = 데이터 구조)인데, SM 할당은 *런타임 스케줄링* 손잡이다. 스케줄링 손잡이는 state 저장/축출/공유를 건드리지 않으므로 memory 이득을 *내려보낼* 수도, *대체*할 수도 없다.

**귀결:** 비대칭은 memory-rooted symptom인데 손에 쥔 도구는 compute에만 닿고 memory는 공유였다 → memory에 뿌리박은 비대칭을 memory-level 분리로 *변환할 경로가 물리적으로 없었다.* E4/E5 음성은 측정 실패가 아니라 이 **orthogonality의 직접 지문**이다. **메모리-실행 분리에 도달하려면 회피하려던 state 추상화를 짓는 것이 유일한 경로였다** → [추가가치 Path 5](additional_value_paths.md).

---

## 4. 회수 가능한 산출물 (중단 ≠ 손실)

음성 결과지만 다음은 그대로 가치가 있고, 종결과 무관하게 보존/활용 가능하다:

1. **발표 가능한 음성 특성화 결과.** "SLM hybrid serving에서 layer-type SM 분할은 무용, co-schedule이 정답"은 characterization 논문/리포트로 성립. 분할을 시도하려는 후속 연구의 시간을 아껴줌. (SSM-Scope 류 ISPASS characterization 트랙과 정렬.)
2. **양성 권고 + 그 구조:** prefill+decode를 공유 SM에 **그냥 co-schedule**(MPS/multi-stream) → 저~중 decode batch에서 1.2–2.0×, 분할/aware 할당 불필요. 즉시 적용 가능한 운영 지침. 겹침의 정도는 두 축이 *따로* 지배:
   - **prefill 타입 = 천장(ceiling).** prefill이 긴 stream이라 "decode를 숨길 방"을 정함 — `pf=ssm` 1.8–2.04× vs `pf=attn` 1.06–1.43× (@db8,ctx4096). 단 메커니즘은 roofline 상보성이 아니라 **duration matching**(비슷한 길이 두 커널이 SM 풀에 함께 들어가 포개짐, occupancy 현상).
   - **decode 타입 = context 스케일링.** short-context에선 타입 무관히 다 숨지만, `dec=ssm`은 O(1) state라 1k→16k **평탄 ~2.0×**, `dec=attn`은 O(L) KV라 long-context에서 decode가 벽시간을 지배하며 **overlap 붕괴**.
   - window는 decode batch로 닫힘(2.04@db8 → 1.15@db256).
   - **함의(중요):** 이 양성 결과의 *흥미로운* 부분 — long-context serving에서 `dec=ssm`이 특별히 강하다는 점 — 조차 execution이 아니라 **memory-state 성질(O(1) state vs O(L) KV)**이 설명한다. 즉 hybrid가 serving에서 *고유하게* 좋은 지점도 뿌리는 메모리다(→ §3.2, [Path 5](additional_value_paths.md)).
3. **비대칭의 비-할당 용도:** kernel fusion/튜닝 신호(ssm가 작아 일찍 포화), 모델↔하드웨어 매칭, layer별 quant/offload 우선순위. (할당이 아닌 곳에서 쓰임.)
4. **재사용 가능한 측정 프레임워크:** measured/derived/metadata 라벨 강제, bootstrap CI 포화점, subprocess 격리, BandwidthEstimator, E0 해석적 wave 예측, 게이트 자동판정(`adjudicate.py`). 다음 프로젝트의 인프라.
5. **음성을 강하게 만든 방법론적 발견:** granularity confound(SSM chunked vs Attn full-seq)가 비대칭을 부풀린다는 점, BW 절대%가 양방향으로 비신뢰라는 점 — 후속자에게 직접적 경고.

---

## 5. 종결 시 명시해야 할 한계 (정직성)

중단 보고서로 닫더라도 아래는 **검증되지 않은 채 남는다.** "분할 무용"을 무조건적 결론으로 과대주장하면 안 된다:

- **7B 미검증 (가장 큰 구멍):** `models.yaml:88` "7B entries are kept but **unused**". v2_report가 "결정적"이라 부른 7B 회귀는 **한 번도 실행되지 않았다.** 단 정확한 질문은 "비대칭이 scale에서 살아나는가"가 아니라 **"큰 커널이 공유 하 파괴적 간섭을 일으켜 격리(분할)가 그 slack을 회수하는가"(§3.1의 술어 b)**이며, 이게 SLM에서만 닫혔다. → 결론 스코프 = **"SLM/A100 한정"**.
- **진짜 prefill 미검증:** E5는 microbench(prefill chunk=1, scan-only, decode step=1). GEMM·다중 chunk 포함 실제 prefill은 `--prefill-layer ssm_full`로만 점검 가능하며 미수행([검수 A2](review_checklist.md)).
- **비대칭의 robustness (이제 부차적):** matched-granularity에서 zamba 20셀 중 9셀(REDUCED 7+ELIMINATED 2)에서 비대칭 약화/소멸([검수 A3](review_checklist.md)). 단 §3.1에 따라 **비대칭은 애초에 lever가 아니므로 이 robustness는 결론을 좌우하지 않고, "비대칭은 서술적 사실"이라는 §4.3 sub-claim의 강도에만 영향**한다. "비대칭 실재"로 단정 금지.
- **HW 한정:** A100-SXM4 단일 디바이스. MPS 별도 프로세스가 아닌 multi-stream 근사(§6).

---

## 6. 권고

1. **본 라인(layer-type 공간 SM 분할) 종결.** 추가 분할-실험에 SXM4 자원 투입 금지. 한계 이득 음수, 메커니즘 수준에서 닫힘(§3).
2. **종결 전 [검수 체크리스트](review_checklist.md)의 P0 5건 통과 확인** — 특히 A2(ssm_full 진짜-prefill 1회)와 C1(7B 미실행 스코프 명기). 이것만 통과하면 "SLM에서 음성 확정"으로 깨끗이 닫힌다.
3. **단, 닫기 전 [추가가치 경로](additional_value_paths.md)를 1회 검토.** 7B 회귀(Path 1)는 *닫는 데도 필요한* 결정적 데이터점이며 — **올바른 술어(b)로 재서술하면 "큰 커널에서 공유 하 회수 가능 slack이 생기는가"를 묻는다**(음성이면 결론을 크기-무관으로 격상, 양성이면 라인 부활). 비용이 크지 않다. prefill/decode overlap 재프레이밍(Path 2)은 음성 라인을 양성 기여로 전환하는 별개 출구다.
4. **v1/v2 동결:** `archive/v1_7b_saturation/` 유지. v2 산출물(results_v2, figures, verdicts) 커밋 후 동결.
5. **종결의 범위 한정 (중요).** 본 종결은 *execution-path 라인*(SM 분할/스케줄링)에 대한 것이다. §3.2에 따라 **hybrid serving의 memory-state 추상화**(이질적 footprint admission, layer-type별 evict-vs-recompute 비대칭, SSM state의 prefix/radix 공유 붕괴, offload 배치)는 **미탐색의 별개 축이며 유일하게 abstraction-worthy한 잔여 방향**이다 → [추가가치 Path 5](additional_value_paths.md). 이건 죽은 라인의 부활이 아니라 *다른 프로젝트*다. 단 novelty는 SSM state의 *lossy-fold* 성질(sliding-window KV 선례와 구분되는 지점)에 걸리며, "두 state를 결합적·비대칭적으로 추론해야 하는가"가 베팅의 핵심.

> **결정 트리:** 검수 P0 전부 통과 + 7B 미실행 수용 → **execution-path 라인 종결(음성 특성화로 발표).** 7B 1회라도 돌릴 여력 → `additional_value_paths.md` Path 1 먼저 → 그 결과로 최종 종결/전환. **단 어느 쪽이든 memory-state 추상화(Path 5)는 닫히지 않은 별개 출구로 남는다.**
