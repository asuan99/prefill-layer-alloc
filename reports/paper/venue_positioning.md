# 투고 positioning 정리 (venue-strategist 산출물, doc-steward 기록)

최종 갱신: 2026-07-24. 작성 주체: `venue-strategist` subagent, 기록 주체:
`doc-steward` subagent.

> **이 문서는 투고 전략(positioning) 정리이며 claim/evidence 문서가 아니다.**
> 여기 나오는 "확정/부분 지지/미검증/철회" 등급은 전부
> [`CLAIM_EVIDENCE_MATRIX.md`](CLAIM_EVIDENCE_MATRIX.md)와
> [`../CONSENSUS.md`](../CONSENSUS.md)의 기존 판정을 **인용·요약**한 것이며, 이 문서
> 자체가 새 증거의 근거가 될 수 없다. 성능 판정(result-analyst)·반증(claims-auditor)은
> 이 문서가 대신하지 않는다. 정본 위계는 항상
> [`PROJECT_STATUS.md`](../../PROJECT_STATUS.md) → `reports/paper/` →
> `CONSENSUS.md` 순서다.
>
> **CFP 날짜·마감·수치는 전부 검증 필요.** venue-strategist의 지식 컷오프(2026-01) 이후
> 실제 CFP가 바뀌었을 수 있다. 아래 날짜는 전부 "추정·검증 필요"로 읽는다. 오늘 날짜
> 기준(2026-07-24)으로도 아직 어느 것도 재확인되지 않았다.

## 대조 결과 (doc-steward, 2026-07-24)

이 문서를 기록하기 전에 `PROJECT_STATUS.md`(2026-07-23 갱신본), `CLAIM_EVIDENCE_MATRIX.md`
(2026-07-23 갱신본), `CONSENSUS.md` §1/§4와 아래 내용을 대조했다. **모순 없음.**
인용된 모든 수치(agnostic 4모델 fused 승리, TPOT 42→124ms/85ms, HE0 5.4σ,
§1-17 10σ, d16 TTFT 7.24s vs d24 1.21s, disaggregation headroom +16%/116 SM)와
등급어(부분 지지/강한 지지(범위 한정)/미검증)는 CONSENSUS/matrix 표기와 정확히 일치한다.
증거 수준을 낮추거나 높이는 재해석은 없다.

**용어 대응(이 문서 한정, 새 claim ID 아님)**: 이 문서는 가독성을 위해 C1–C4 레이블을
쓰지만 이는 `CLAIM_EVIDENCE_MATRIX.md`의 공식 Claim A–E와 다른 문서 전용 축약이다.
혼동 방지용 대응표:

| 이 문서 레이블 | 내용 | 공식 대응 |
|---|---|---|
| C1 | PD separation이 fused를 이김 | `PROJECT_STATUS.md` "확정된 결과" #1 |
| C2 | layer-granular 정책 死 (기전) | Claim B — 강한 지지, 현 substrate 한정 |
| C3 | 단일-GPU reactive dynamic이 static 못 넘음 (entanglement) | `PROJECT_STATUS.md` "확정된 결과" #3/#5, CONSENSUS §1-4/1-6/1-7/1-17 |
| C4 | decode floor의 composition/context/load 의존 | Claim A — 부분 지지 |
| H-Arch | true dual-worker가 coupling 감소 | Claim D — 미검증 |
| H-Policy | Hybrid-informed policy가 generic/static보다 우수 | Claim E — 미검증 |

---

## 0. 한 줄 진단

지금 자산은 characterization과 기전 규명된 negative result가 중심이고, 검증된
constructive win이 아직 없다. 최근접 선행(Bullet ASPLOS'26, MuxWise, Drift —
**출처 검증 필요**, §5 참고)이 일반 transformer에서 dynamic PD-mux가 이긴다고
이미 주장했으므로, 우리의 "dynamic이 static을 못 넘음"(C3)은 좁게 scoping하지
않으면 substrate 아티팩트로 반박당할 위험이 있다. 정직한 결론: 지금 당장
top-tier systems(OSDI/NSDI/SOSP)는 불가, architecture(ASPLOS/ISCA/MICRO/HPCA)는
상당한 추가작업 후 가능, MLSys가 유일한 근시일 현실 경로.

## 1. 후보 기여 — 증거 수준별

### A. 확정 (서빙 직접 측정 지지)

- **C1**: Hybrid 모델서 PD resource separation이 fused보다 나은 operating point.
  근거 = agnostic이 4모델 전부서 fused 이김(cudagraph-ON) — `PROJECT_STATUS.md`
  "확정된 결과" #1. 한계 = 낮은 novelty(MuxWise/Bullet/Drift가 transformer서
  이미 확립; delta는 "hybrid에서도 성립"뿐).
- **C2** (Claim B, 강한 지지·범위 한정): layer-granular runtime 정책 死 — 기전
  규명. 근거 = coordinated per-type TPOT 42→124ms(최적화 후 약 85ms), sub-step
  drain, cudagraph 비양립. 한계 = substrate 한정(SGLang green-context/A100),
  Nsight timeline 없음, libsmctrl 미비교.
- **C3**: 단일-GPU reactive dynamic이 decode-heavy static 못 넘음 — 기전
  (entanglement·asymmetry·positioning). 근거 = HE0 5.4σ(관대 3s SLO, CONSENSUS
  §1-7), §1-17 10σ(tight chat 300/50ms); d16 TTFT 7.24s vs d24 1.21s(CONSENSUS
  §1-4). 한계 = 선행과 정면 긴장(Bullet/MuxWise는 dynamic 승리를 주장) → **"shared
  running-batch coupling 하의 reactive single-worker"로 scoping을 좁혀야
  substrate-artifact 반박을 막을 수 있다**.
- **C4** (Claim A, 부분 지지): decode floor는 composition·context·load 의존
  (offline predictor 입력). 근거 = Zamba2 knee, best-split 이동, decode-attn
  SM 민감도 ctx256→16k = 1.1×→10.5×. 한계 = "부분 지지" 등급 그대로 유지;
  cudagraph 운영점 joint surface·held-out 일반화는 P3 미완.

### B. 미검증 positive 가설 (구현 완료 ≠ 성능 주장)

- **H-Arch** (Claim D, 미검증): true dual-worker가 single-worker coupling을
  줄인다. R1은 observer로 판명·철회됐으므로 이 가설의 증거가 아니다(`R1_REANALYSIS.md`).
  novelty 리스크 = Bullet이 intra-device disaggregation(별도 프로세스+MPS)으로
  이 각도를 선점; delta는 hybrid 한정 + coupling 정량 기전.
- **H-Policy** (Claim E, 미검증): offline hybrid-profile decode-floor predictor가
  generic dynamic·global static보다 높은 SLO goodput을 낸다. 증거 없음. 가장
  novel한 잠재 positive지만, `EXPERIMENT_ROADMAP.md` P4 acceptance대로 B6가
  B1·B5를 paired CI로 유의하게 ≥3% 이겨야만 성립한다.
- **Disaggregation headroom** (CONSENSUS §1-20): decoupled oracle 기준 진짜
  headroom +16%는 disaggregation 몫이며, 116 SM > 108 = 단일-GPU 불가로 확인됨.
  미구현·방향 제시일 뿐 아직 어떤 claim에도 편입되지 않은 관측이다.

## 2. 학회별 fit (venue-strategist 평가, 검증 필요)

| Venue | 평가 |
|---|---|
| MLSys | 강(characterization)/중(win 요구 시) |
| ASPLOS | 중(P5+일반화 후)/현재 약 |
| ISCA/MICRO | 약~중 |
| HPCA | 약~중 (마감 타이밍 특히 검증 필요, §4) |
| OSDI/SOSP | 약 |
| NSDI | 약 |
| EuroSys/ATC | 약~중 |
| SC | 약 |

세부 근거는 venue-strategist 원 산출물 텍스트(아래 §2-detail)를 그대로 보존한다.

### 2-detail. 학회별 근거 (원문 보존)

- **MLSys**: 서빙 정책 + 모델 특성화, focused 기여에 관대, measurement 트랙 존재.
- **ASPLOS**: Bullet ASPLOS'26(검증 필요)이 PD-mux SM 공유를 in-scope로 이미
  다뤘고 constructive win을 보유 → characterization+guideline 프레이밍 +
  Nsight + libsmctrl 비교 + 다중 GPU 필요.
- **ISCA/MICRO**: SM-partition 기전·green-context는 architectural이나 순수
  SW-level 정책 negative는 약함; 정량 모델 + occupancy/DRAM/Tensor-Core Nsight +
  아키텍처 일반화 없으면 desk-reject 위험.
- **HPCA**: HPCA'27 abstract 마감이 오늘(2026-07-24, **검증 필요** — 매우
  구체적인 일치라 특히 재확인 필수)이라 사실상 불가로 추정; HPCA'28 사이클로
  밀림.
- **OSDI/SOSP**: 동작 scheduler + 실 워크로드 + 강한 baseline + constructive
  win 필수; 순수 negative 불가.
- **NSDI**: 단일-GPU PD-mux fit 약함; disaggregation을 cluster 수준으로
  확장하면 중간 정도.
- **EuroSys/ATC**: OSDI/SOSP보다 문턱 낮음; 동작 artifact + 견고 guideline이면
  ATC가 상대적으로 관대하나 여전히 constructive 요소 필요.
- **SC**: 규모·throughput·cluster를 보상; 단일 108-SM 미시 기전은 약함;
  disaggregation 다중 GPU throughput 실증이 필요.

## 3. 학회별 "받으려면 추가로 필요한 것" (ROADMAP 언어)

- **MLSys**(characterization+guideline): P3 full-model decode-floor profile을
  cudagraph 운영점에서(D16–D108 × batch1–48 × ctx256–16K) 돌려 C4를 지지로
  강화; P6 W1–W9 다중 워크로드로 entanglement(C3)를 일반 현상화 +
  applicability map; B0–B2, B7 최소 baseline; (선택) P5-lite Nsight로 (D)
  granularity sub-step drain timeline attribution.
- **ASPLOS/ISCA/MICRO/HPCA**(기전): P5 전체(Nsight Systems/Compute — kernel
  timeline, GPU idle gap, stream/event overlap, graph replay, SM active,
  occupancy, Tensor Core, DRAM/L2); libsmctrl vs green-context 비교(Bullet
  기판 재현, "green-context 아티팩트 아님" 반박 봉쇄용 — roadmap 밖 신규 작업
  필수); 다중 GPU/아키텍처 일반화(H200 등) + decode-floor 정량 모델; B7
  layer-granular negative 정식 baseline.
- **OSDI/SOSP/NSDI/EuroSys/ATC**(constructive system): P2 architecture gate
  (true dual fixed가 legacy fixed 대비 decode progress/ITL/queue-age 개선 +
  throughput regression ≤3% → Claim D 채택); P3 acceptance(estimator
  upper-bound coverage ≥95%, under-reservation ≤1%); P4 B0–B8 전체 +
  profile ladder(B6가 B1·B5를 paired CI 유의·≥3% → Claim E 채택); 강한 실측
  baseline(MuxWise/Bullet/Drift/DistServe 직접 비교); P6 실 워크로드 +
  artifact evaluation.

## 4. 현실적 1순위 경로 + 리스크

- **경로 A(지금 자산)**: MLSys "Hybrid PD-mux characterization + mechanistic
  negative + guideline". 내용 = (1) hybrid decode-floor의 composition/
  context/load 의존[C4] (2) layer-granular 정책이 (D) granularity·cudagraph
  비양립으로 死[C2] (3) 단일-GPU reactive dynamic이 entanglement로
  decode-heavy static 못 넘음[C3] (4) guideline: peak-decode 기준
  decode-heavy static + offline decode-floor. 추가 작업 = P3 + P6 일부 +
  B0–B2/B7(Nsight/libsmctrl 없이도 MLSys 제출 가능하나 있으면 강화). 타이밍 =
  MLSys 2027 마감 ~2026년 10–11월 추정(MLSys 2026판 마감이 2025-10-30이었다는
  전례 기반 추정, **검증 필요**).
  리스크: ① novelty vs MuxWise/Bullet/Drift(incremental 공격 가능) ②
  "dynamic이 진다"가 substrate 아티팩트로 반박당할 위험 → C3를 **"shared
  running-batch coupling 하의 reactive single-worker"**로 명시 scoping하고
  "Bullet의 승리는 coupling을 프로세스 분리로 깬 것이라 우리 기전과 일관"으로
  프레이밍해야 함 ③ 순수 negative 수용성(characterization 정량 깊이가 얕으면
  약함).
- **경로 B(H-Policy 검증 후)**: 전제 = P2(H-Arch gate) + P4(B6 > B1·B5 ≥3%
  paired CI). 성공 시 MLSys full 또는 EuroSys/ATC, 강하면 OSDI/NSDI 시도.
  리스크: ① CONSENSUS §1-13/§1-17이 "동적이 이길 regime 없음"을 강하게
  시사하므로 → near-saturation 상보 워크로드에서의 좁은 승리만 정직하게
  노려야 함(overclaim 시 claims-auditor에 걸림) ② H-Arch가 Bullet 재발명
  위험 ③ B6가 3% 못 넘으면 P4 실패 → 경로 A로 회귀, D/E는 negative
  architecture result로 그대로 보존.
- **권고**: 지금 당장 정직하게 제출 가능한 경로는 A뿐이다. novelty·
  substrate-artifact 반박을 막으려면 최소 P3 + entanglement 다중 워크로드
  일반화 + (가능하면) libsmctrl 스팟 비교가 사실상 필수다. 경로 B는 P2/P4
  게이트 선결이 필요하고 정본상(§1-13/§1-17) 성공 확률이 낮으므로 A를
  기본선으로 삼고 B는 upside로 병행하는 것이 정직하다. Architecture 학회
  (ASPLOS'27 fall ~2026년 9월 / ISCA'27 abstract ~2026년 11월, **검증
  필요**)는 P5 Nsight + libsmctrl + 다중 GPU 일반화 전엔 무리다.

## 5. Related work 차별점 (출처 전부 검증 필요)

DistServe/Splitwise = prefill·decode를 별도 GPU/instance로 분리(inter-device
disaggregation); Sarathi-Serve = chunked prefill로 시분할 간섭 완화; vLLM/
SGLang = continuous batching 엔진. 최근접 선행 Bullet(ASPLOS'26, libsmctrl+
MPS, intra-device disaggregation + dynamic provisioning), MuxWise
(green-context PD-mux, decode_bs 임계 SM-group 테이블 + per-partition
cudagraph, 2.2× goodput 주장), Drift(in-place phase-decoupled compute
partition)는 모두 일반 transformer에서 단일-GPU spatial PD-mux 승리를
constructive로 보인다(전부 **출처·수치 검증 필요**, §6 참고).

우리 차별점: (1) workload가 hybrid(attention+SSM) — decode floor가 layer
composition·context·load에 따라 이동[C4] (2) layer 구조를 runtime scheduling
boundary로 쓰려는 가설을 기전과 함께 반증[C2](green-context (D) granularity로
TPOT 42→124ms + cudagraph 비양립) — 선행이 하지 않은 negative (3) 단일-GPU
reactive 정책이 왜 static을 못 넘는지의 기전[C3](entanglement).

MuxWise의 "dynamic"이 사실상 decode_bs 임계 테이블(= load-dependent static
schedule)인 점은 HE0("static 지배", CONSENSUS §1-7)와 일관된다. Bullet의
승리가 프로세스 분리로 coupling을 깬 것이라는 점은 §1-20 disaggregation
headroom과 정합적이다. 즉 우리 negative는 선행 positive와 모순이 아니라
"coupling이 원인"이라는 동일 기전의 다른 면이며, **이 프레이밍이 유일한
substrate-artifact 반박 방어선**이다.

## 6. 참고 (검증 필요 소스)

- Bullet: arXiv 2504.19516 (ASPLOS'26) — **검증 필요**
- MuxWise / SLO-oriented PD-Multiplexing: arXiv 2504.14489 — **검증 필요**
- PD-Multiplexing with GreenContext: LMSYS blog, 2025-09-28 — **검증 필요**
- CFP: MLSys 2026/2027, HPCA 2027, ASPLOS 2027, OSDI 2027, NSDI 2027,
  EuroSys 2027 — **날짜 전부 검증 필요** (venue-strategist 지식 컷오프 2026-01
  이후 변동 가능; 이 문서 기록일 2026-07-24 기준으로도 아직 재확인 안 됨)

---

## 이 문서의 상태

이 문서는 살아있는 전략 문서다(`CONSENSUS.md` §4 "살아있는 문서" 표에 전략
문서로 등재). 새로운 서빙 측정이나 claim 등급 변경이 있으면 이 문서가 아니라
`CLAIM_EVIDENCE_MATRIX.md`/`PROJECT_STATUS.md`를 먼저 갱신하고, 이 문서는
그 변경을 반영해 갱신한다(반대 방향 금지).
