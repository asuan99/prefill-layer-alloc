# 투고 positioning 정리 (venue-strategist 산출물, doc-steward 기록)

최종 갱신: 2026-07-25(신규성 축 정정: disaggregation→multiplexing 재조준 +
negative 2갈래 분해 + ★cross-substrate serving 이식 "불필요·부적합"으로
재프레이밍[초판의 make-or-break 프레이밍 철회] + green-context vendor-primitive
방어 + Risk 2를 닫는 vendor-substrate 3수, §0.1). 작성 주체: `venue-strategist`
subagent, 기록 주체: `doc-steward` subagent.

> ⚠️**2026-08-16 doc-steward 인용금지 전파(내용 재해석 없음, 새 서빙 측정
> 아님)**: 이 문서가 인용하는 "d16 TTFT 7.24s vs d24 1.21s"(아래 §0.1
> 대조표 직후·"C3" 항목)는 2026-08-04 `layertype_dynamic_
> POSITIVE_2026-08-04.md:159`가 **폐기 벤치 n=1이라 magnitude 인용
> 금지**로 지정한 수치다(방향만 변화 trace n≥4가 지지). 2026-07-24
> 대조 당시엔 이 인용금지가 아직 존재하지 않아 "모순 없음" 판정이
> 정확했다 — 지금 이 문서를 재인용할 때는 아래 두 지점에 추가된
> 인용금지 각주를 함께 읽을 것. 상세 `../CONSENSUS.md` §1-4·rev33,
> `../PRIZE_SIZE_ARGUMENT_2026-08-16.md` §7.

> ⚠️**2026-09-10 doc-steward 갱신(신규 1차 증거 승급, 새 서빙 측정 아님)**:
> §5(related work)와 §6(참고 소스)에 `workspace/engine-port/external/
> muxwise/{sharegpt.yml, loogle.yml}` 설정 파일을 **직접 읽어 확인**한
> 결과를 반영했다 — 이 파일들이 같은 코드베이스(엔진)에서 워크로드별로
> 다른 SM 분할표를 이미 출하한다는 **구조적 사실**을 **저장소 내 1차
> 증거**로 승급했다(★2026-09-10 doc-steward 자기정정: 초판이 여기
> "같은 하드웨어"라 잘못 적었다 — 두 yml 전 행의 SM 합은 132로
> H100/H200급 다이이며 우리 A100 108-SM이 아니다, 아래 §5 표 참조).
> arXiv 2504.14489 논문 자체의 주장·수치·게재처는 여전히
> (ii) "검증 필요"로 분리했다(§5 말미·§6 참고). 근거 `../CONSENSUS.md`
> §5-6(2026-09-09 2차 세션 (D) 항목), `../impl_vs_external_pdmux_
> 2026-08-28.md` §2.5–2.6. **claim 등급 변경 0건 — stake #1("최적
> static split 위치가 워크로드 모양에 따라 움직이는가")은 이 프로젝트의
> 기판(A100 108-SM green-context)에서 여전히 구매 불가로 종결된
> 상태**(`EXPERIMENT_ROADMAP.md` longctx_conflict 절, `audit_p6_rules_
> 2026-09-09/VERDICT.md` Y1-f) — 아래 §5의 정정은 그 스코프 판정을
> 바꾸지 않으며, "선행이 이미 그 답을 전제하고 있다"는 positioning
> 사실만 추가한다.

> **2026-07-25 갱신 성격**: 이 갱신은 **새 서빙 측정이 아니라 prior-art 조사 +
> positioning reasoning**이다. 확정 서빙 결과(`PROJECT_STATUS.md` "확정된 결과"
> §1–5, Claim A–F 등급)는 하나도 바뀌지 않는다. §0.1은 (i) `venue_positioning.md`
> 자신의 §1/§5(2026-07-24, disaggregation 축 신규성 평가)와 (ii)
> `../spatial_decoupling_design_review_2026-07-25.md`의 신규성 절을
> **supersede**한다(축이 disaggregation→multiplexing으로 잘못 조준됐던 것을
> 정정). 아래 §0.1을 §1/§5보다 먼저 읽는다.

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

> ⚠️**dated 정정(2026-08-16, doc-steward)**: 위 "모순 없음" 판정은
> 2026-07-24 시점 기준으로 정확하다 — "d16 TTFT 7.24s vs d24 1.21s"의
> 인용금지는 그 열흘 뒤인 2026-08-04에야 생겼다(폐기 벤치 n=1,
> `../CONSENSUS.md` §1-4). **지금** 이 수치를 재인용할 때는 그 인용금지
> (방향만 유효, magnitude 인용 금지)를 함께 적용할 것 — 이 문단 자체는
> 이력 그대로 보존한다.

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

## 0.1 ★2026-07-25 정정 — 신규성 축(disaggregation→multiplexing) + negative 2갈래 + green-context vendor-primitive 방어 + 벡터2 게이트(cross-substrate 이식 철회→Transformer-control)

**계기(사용자 지적)**: `../spatial_decoupling_design_review_2026-07-25.md`와 이 문서의
2026-07-24판 §1/§5는 신규성을 **disaggregation 축**(DistServe/vLLM-SSM-disagg/DUET
대조)으로 평가했다. 그러나 본 논문의 실제 기여 축은 **co-located multiplexing**
(단일 GPU에서 prefill+decode 동시 실행, green-context로 SM 시공간 분할)이지
disaggregation(별도 엔진/GPU + KV 전이)이 아니다. 아래는 축을 바로잡은 재조사
결과다.

### (1) 신규성 재평가 (multiplexing 축) — 부분적으로 새롭다, 실증

multiplexing 선행 전수 확인 결과 **전부 Transformer-centric**이고, hybrid/Mamba
서빙 선행은 전부 **disaggregation 또는 전용 HW**다:

| 선행 | 유형 | 대상 모델 | 신규성 겹침 |
|---|---|---|---|
| DuetServe (arXiv 2511.04791) | co-located mux, libsmctrl, adaptive SM | 표준 Transformer | multiplexing 축 정면 겹침(throughput 메트릭) |
| MuxWise / SLO-oriented PD-Multiplexing (arXiv 2504.14489, ASPLOS'26 추정) | co-located mux, green-context, decode_bs 임계 SM-group + per-partition cudagraph | 표준 Transformer | multiplexing 축 정면 겹침(같은 기판 green-context) |
| SGLang PD-multiplexing with GreenContext (LMSYS blog, 2025-09) | co-located mux, green-context | 표준 Transformer | 우리 substrate의 upstream 원류 |
| Nexus (arXiv 2507.06608) | co-located mux | 표준 Transformer | multiplexing 축 겹침 |
| Bullet (ASPLOS'26 추정, libsmctrl+MPS) | co-located mux, intra-device disaggregation | 표준 Transformer | multiplexing 축 겹침 |
| SSM characterization (arXiv 2507.12442) | 특성화, mux 아님 | SSM/hybrid | SM-민감도는 다루나 ×mux 정책 각은 없음 |
| vLLM v0.20.0 blog (2026-04) | **disaggregation**, mamba state 전이 | hybrid(SSM) | mux 아님 — disagg 축 |
| DUET (DAC'26, arXiv 2603.15530) | **disaggregation** | hybrid(NemotronH/Zamba2) | mux 아님 — disagg 축 |

⇒ **"hybrid(attn+SSM)에 co-located PD-mux를 적용 + mamba decode의 SM-비민감성이
최적 정책 구조를 바꾼다(동적 lever가 구조적으로 약화되어 static이 지배한다)"는
특성화는 mux 문헌에 공백이다. 신규성 = 부분적이나 실재한다.**

방어 가능한 자산은 자명한 "mamba=memory-bound"(이미 문헌에 있음)가 아니라
**DuetServe(Transformer·libsmctrl·throughput 메트릭에서 동적이 static을 1.3×
이김)와의 정량적 상반**이다: 같은 아이디어(co-located adaptive SM 분할)가
hybrid·green-context·conjunctive-SLO goodput 조건에서는 진다. "왜 같은
아이디어가 이 substrate·이 메트릭·이 모델군에서는 실패하는가"의 기전 규명이
이 논문이 crowded positive 문헌에 보탤 수 있는 knowledge다.

⚠️ **모든 arXiv ID·학회명(ASPLOS'26/ICML'26 등)·acceptance 여부·날짜는
venue-strategist 지식 컷오프(2026-01) 이후 다수** = **인용 전 원문 재확인
필수**(§6 목록도 동일 caveat).

### (2) ★negative의 두 갈래 — 반드시 분리해서 팔 것

현재 정본(§1의 C2/C3, `CLAIM_EVIDENCE_MATRIX.md` Claim B/A/C)의 negative
result를 substrate-robustness 축으로 분해한다:

- **(A) green-context 종속 — 헤드라인으로 팔지 말 것.** layer-aware 死
  (coordinated per-type drain, TPOT 42→124ms, 최적화 후 85ms)와 cudagraph
  비양립(Claim B). 이것은 정확히 **DuetServe가 libsmctrl + interruption-free
  engine으로 일부러 우회한 바로 그 비용**이다. 이 축을 substrate-invariant
  negative로 헤드라인화하면 리뷰어의 "libsmctrl 쓰면 되잖아"로 데스크리젝트
  위험이 크다. `CLAIM_EVIDENCE_MATRIX.md`가 이미 Claim B를 "현 substrate
  한정"으로 스코프하고 있으며, 이 정정은 그 스코프를 이 positioning과
  정합시킬 뿐 새 등급 변경이 아니다.
- **(B) mechanism-independent — 방어 가능.** lever-weakness(mamba decode의
  SM-둔감성, roofline 성질 — Claim A), entanglement(공유 running-batch+KV로
  decode 굶김→ITL↑→admission 차단→TTFT 폭발 — Claim C), 비대칭(decode
  과소공급=파국). 이 셋은 **파티셔닝 primitive(green-context vs libsmctrl vs
  MPS)와 무관한 성질**이라고 주장할 수 있는 후보다. 단, 두 번째 substrate로
  serving을 이식해 재현할 **필요는 없다**(그 이식은 불필요·부적합 — 아래 (3)) —
  대신 아래 (3)의 vendor-substrate 3수(green-context 위 Transformer 대조 +
  roofline + 기측정 entanglement 귀속)로 primitive 독립성을 식별한다.

### (3) ★cross-substrate serving 이식은 불필요·부적합 — green-context = 배포-관련성 방어(약점 아님)

⚠️**2026-07-25 하향·재프레이밍(사용자 지적)**: 이 §0.1의 초판은
cross-substrate serving 이식(XS-series)을 "publishability make-or-break 필수"로
과잉 프레이밍했다. **철회한다.** 두 번째 substrate로 serving을 이식하는 것은
**불필요하며 부적합**하다:

- **MPS 부적합**: SM 파티션이 **프로세스별·정적**이라 런타임 동적 PD-mux 자체가
  불가하고, 단일-프로세스 `event_loop_pdmux`를 멀티-프로세스로 전면 재구조화하는
  비용이 실익을 초과한다.
- **libsmctrl 부적합**: NVIDIA가 제공하지 않는 **리버스-엔지니어링 per-arch SM
  마스킹**이라 하드웨어 세대·드라이버에 귀속된다(driver-580 BLOCKED가 바로 그
  증거). 논문의 배포 가이드라인 근거에 **비-vendor·비-이식 의존성**을 들이는
  셈이다.

★**green-context는 약점이 아니라 배포-관련성 방어다.** NVIDIA 공식
fine-grained SM primitive는 **green-context 하나**(CUDA Green Contexts,
12.4+)뿐이다. libsmctrl(비-vendor·arch-bound)·MPS(정적)는 배포 primitive가 아니다.
⇒ 리뷰어의 "libsmctrl 쓰면 되잖아"에 대한 답 =

> **green-context가 배포 가능한 유일한 vendor primitive다. DuetServe/Bullet의
> 동적-승은 libsmctrl(세대 귀속·비-vendor) 위에서만 성립한다 → 설령
> libsmctrl에서 hybrid 동적이 이겨도 그건 이식 불가한 research curiosity이지,
> 배포 가이드라인의 반례가 아니다.**

즉 (A) layer-aware 死는 **green-context-bound로 정직히 스코프**하되(헤드라인
아님), libsmctrl 비-이식성이 "그럼에도 배포에는 green-context 결과가 맞다"로
그 스코프를 오히려 받쳐준다.

**그럼 Risk 2(모델 vs substrate 귀속)는 무엇으로 닫나 — cross-substrate 이식
대신 vendor-substrate 3수**(전부 기존 green-context 위, 값쌈):

1. ★**Transformer-vs-hybrid 대조를 *같은* green-context + *같은*
   conjunctive-SLO에서**: 순수 Transformer(예: Qwen/Llama)를 기존 pdmux에
   통과시킨다. drain 비용은 두 모델에 **동일하게 작용**하므로 **상쇄**된다.
   동적-승/패가 모델에서 갈리면 그 flip은 substrate·메트릭 고정 하에 **모델
   (hybrid) 귀속**으로 식별된다. 이것이 XS-series를 대체하는 새 게이트이며,
   진짜 식별 실험이다. 정직한 리스크: Transformer도 지면 negative는 hybrid가
   아니라 **메트릭+배포-primitive 탓**이 되지만, 그래도 다른(여전히 유효한)
   기여다.
2. **lever-weakness = roofline microbenchmark**(r0c의 SM-민감도 데이터 보유):
   mamba decode의 SM-둔감성은 **연산강도(roofline) 성질**이라 partitioning
   primitive에 robust하다. "libsmctrl이 고친다"는 반론은 drain(=(A))에만 닿고
   **lever(=Claim A)엔 닿지 않는다**.
3. **헤드라인 HE0는 이미 entanglement 귀속으로 측정 완료**: 정본이
   컨트롤러 CPU 0.014%(§1-12)로 동적-패가 컨트롤러 오버헤드 탓이 **아님**을
   이미 직접 계측했다(CONSENSUS, `slo-aware-scheduling-track.md`). ★**2026-08-22
   정정**: `switch_count`는 **5/8/18/21**이고(그 자체가 32-샘플 서브샘플·arm
   의존 bias로 `g2s_analyze.py:1374-1377`에서 **단독 인용 금지**), 직접
   계측된 것은 컨트롤러 CPU(§1-12)와 **agnostic `adjust_stream_groups` 경로의
   전환/드레인 상한**(`s ≤ 0.04 ms/전환`·`d ≤ 0.07 ms/경계`, 2026-08-22,
   CONSENSUS §1 신규 행 34)이다 — **컨트롤러 구동 전환의 드레인은 여전히
   미측정**이다. → drain-아티팩트 반론은 layer-aware 死(=(A))에만 닿고
   **헤드라인(HE0)과는 무관**하다는 결론 자체는 불변.

⇒ 세 수 모두 **기존 green-context 기판 위에서** Risk 2를 닫으므로, 두 번째
substrate로의 serving 이식은 필요하지 않다. `EXPERIMENT_ROADMAP.md` "벡터2"는
이 정정에 맞춰 XS0/XS1/XS2를 "이식 불필요·부적합" 노트로 재프레이밍하고, 새
게이트 = **"green-context 위 Transformer-control 대조"**(위 1수)로 교체했다.

### (4) long-context의 역할 = Risk 1/3 + 모델-composition 다리(substrate 다리는 (3)이 담당)

`../longcontext_trace_plan.md`의 long-ctx 트랙(Stage 0/L−2 → L3/L3s)은 이번
세션의 반대의견 검토에서 나온 세 리스크 중 둘을 닫는다: **Risk 1**(negative가
실행가능한 가이드라인으로 이어지는가)과 **Risk 3**(negative가 언제 유효한지
경계를 그을 수 있는가 — limiting-case 취급 방지). 또한 **Risk 2의
모델-composition 다리**(granularity 비용이 hybrid 모델 구성에 따라 달라지는 게
아니라 ctx-불변 구조적 성질이라는 것)를 분리해 보여준다. **Risk 2의
substrate/모델-귀속 다리는 long-ctx가 아니라 위 (3)의 vendor-substrate 3수**
(green-context 위 Transformer-control 대조 + roofline lever-weakness + 기측정
entanglement 귀속)**가 닫는다** — cross-substrate serving 이식이 철회됐으므로
long-ctx가 그 "이식의 대체재"일 필요도 없다. long-ctx와 (3)은 서로 다른
질문(ctx-regime 경계 vs primitive/모델 귀속)을 담당한다.

### (5) venue positioning 갱신

- vendor-substrate 3수(위 (3): green-context 위 Transformer-control 대조 +
  roofline lever-weakness + 기측정 entanglement 귀속) + C1(lever-weakness)
  robust + 실행가능 가이드라인 → **MLSys가 1순위 현실권**(§4 경로 A 유지).
  ⚠️초판이 선결 조건으로 걸었던 cross-substrate serving 이식은 **철회**한다 —
  선결 조건이 아니며, green-context vendor-primitive 방어가 그 자리를 대신한다.
- 위에 더해 C3/H-Policy(offline decode-floor predictor가 실제로 승리)가
  성립하면 → ATC/EuroSys로 상향 가능(§4 경로 B).
- OSDI/NSDI는 순수 negative로는 여전히 불가(§2 표 불변). SC는 fit이 아님(§2
  표 불변).
- 근거 URL(재확인 필요): DuetServe https://arxiv.org/abs/2511.04791 · MuxWise
  https://arxiv.org/abs/2504.14489 · SGLang-pdmux LMSYS blog(2025-09) ·
  Nexus https://arxiv.org/abs/2507.06608 · SSM-char
  https://arxiv.org/abs/2507.12442 · vLLM hybrid-SSM-disagg blog(2026-04) ·
  DUET https://arxiv.org/abs/2603.15530.

이 갱신은 §1–§6(2026-07-24판)의 claim/evidence 인용을 바꾸지 않는다 — §1의
"C2/C3" novelty risk 서술과 §5의 related-work 문단은 disaggregation이 아니라
multiplexing 선행(DuetServe/MuxWise/SGLang-pdmux/Nexus/Bullet)을 1순위
비교군으로 읽어야 한다는 점만 위로 정정한다.

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
  §1-4, ⚠️**인용금지 2026-08-04 — 폐기 벤치 n=1, magnitude 인용 금지·방향만
  유효, doc-steward 전파 2026-08-16**). 한계 = 선행과 정면 긴장(Bullet/MuxWise는 dynamic 승리를 주장) → **"shared
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

★**2026-09-10 추가(doc-steward, 저장소 내 1차 증거로 승급)** — 위 문단의
"decode_bs 임계 테이블" 프레이밍은 지금까지 요약·추론이었으나, 그 설정
파일 자체가 저장소 안에 있어 **직접 읽어 확인**했다
(`workspace/engine-port/external/muxwise/{sharegpt.yml, loogle.yml}`):

| | `sharegpt.yml` | `loogle.yml`(long-context) |
|---|---|---|
| `sm_group_num` | 8 | 5 |
| `manual_divisions`(prefill SM, decode SM, `decode_bs_threshold`) | `[112,20,1] [104,28,5] [96,36,10] [80,52,15] [64,68,20] [56,76,25]` | `[80,52,1] [64,68,5] [56,76,10]` |
| `decode_bs=1`에서 decode SM | **20** | **52** |
| prefill+decode SM 합 | 132 (전 행 일정) | 132 (전 행 일정) |

이것은 **저장소 내에서 직접 검증된 사실**이다: 같은 코드베이스·같은 green-context
primitive 위에서, MuxWise는 워크로드(ShareGPT vs LooGLE)마다 **다른 SM 분할표를
출하**하고, 세 번째 필드 `decode_bs_threshold`가 "현재 decode batch가 이 문턱을
넘으면 이 분할로 전환"이라는 규칙으로 "분할점이 이동한다"를 자료구조로 인코딩한다
(선택 루프는 break 없이 마지막 만족 행을 취함 — 상세 대조는
`../impl_vs_external_pdmux_2026-08-28.md` §2.5–2.6, 특히 `manual_divisions` 3번째
값의 엔진 의미). ★합계 132 SM은 H100/H200급 다이(A100 108 SM 아님)이므로 이
수치를 우리 108-SM 격자로 그대로 이식하지 않는다 — 여기서 검증되는 것은 "분할표가
워크로드에 따라 달라진다"는 **구조적 사실**이지 특정 SM 수의 이식성이 아니다.

**증거 등급 구분(overclaim 방지)**: (i) **저장소 내에서 직접 검증됨** = 위 표의
설정 파일 내용 자체(파일 경로·필드·수치는 이 세션이 직접 읽어 확인, 인용 시
재확인 불필요). (ii) **여전히 검증 필요** = arXiv 2504.14489 논문 본문의 주장
(예: 2.2× goodput), 수치, 게재처(ASPLOS'26 추정) — venue-strategist 지식
컷오프(2026-01) 이후 문헌이라 원문 재확인 전 인용 금지(§6).

**이 사실이 포지셔닝에 갖는 의미(이미 정본에 등재된 해석, 새 주장 아님)**: stake
#1의 긍정 답("최적 static split 위치가 워크로드 모양에 따라 움직인다")은 이미
최근접 선행(MuxWise)의 **제품 전제**로 배포돼 있다 — 우리가 이것을 "발견"으로
팔 수 없다. 이는 HE0와도 모순이 아니라 **정합**이다: 부하-색인(load-indexed)
static 테이블은 여전히 static이지 reactive closed-loop 제어가 아니다(HE0가
반증한 것은 후자). ⇒ 남는 우리 몫은 *"고정 분할표(그리고 그 분할표가 이미
워크로드-조건부라는 사실)를 넘어서는 closed-loop reactive 판본이 왜 더 얻지
못하는가"*이며, 이것이 정확히 C3(entanglement 기전)의 자리다. `stake #1` 자체에
대한 우리 실험적 답은 이 기판(A100 108-SM, `prefill SM+decode SM=108` 엔진
강제)에서 **구매 불가로 종결**됐다(`../../workspace/engine-port/results/
longctx_conflict/audit_p6_rules_2026-09-09/VERDICT.md` Y1-f, `EXPERIMENT_
ROADMAP.md` longctx_conflict 절) — 이 문단은 그 판정을 재도출하지 않는다.

## 6. 참고 (검증 필요 소스)

- Bullet: arXiv 2504.19516 (ASPLOS'26) — **검증 필요**
- MuxWise / SLO-oriented PD-Multiplexing — **증거 등급 분리(2026-09-10)**:
  (i) 워크로드별 SM 분할표(`sharegpt.yml` decode 20 SM vs `loogle.yml` decode
  52 SM at `decode_bs_threshold=1`, `manual_divisions` 필드 구조)는 **저장소 내
  `workspace/engine-port/external/muxwise/{sharegpt.yml, loogle.yml}`를 직접
  읽어 검증 완료** — 재확인 불필요, 위 §5 표 참조. (ii) 논문 본문 주장·수치
  (예: "2.2× goodput")·게재처(arXiv 2504.14489, ASPLOS'26 추정)는 **여전히
  검증 필요**(venue-strategist 지식 컷오프 2026-01 이후 문헌, 원문 재확인 전
  인용 금지) — (i)과 (ii)를 혼동해 인용하지 말 것.
- PD-Multiplexing with GreenContext: LMSYS blog, 2025-09-28 — **검증 필요**
- CFP: MLSys 2026/2027, HPCA 2027, ASPLOS 2027, OSDI 2027, NSDI 2027,
  EuroSys 2027 — **날짜 전부 검증 필요** (venue-strategist 지식 컷오프 2026-01
  이후 변동 가능; 이 문서 기록일 2026-07-24 기준으로도 아직 재확인 안 됨)
- ★**2026-07-25 추가(§0.1, multiplexing 축 재조사)**: DuetServe arXiv
  2511.04791 · Nexus arXiv 2507.06608 · SSM characterization arXiv
  2507.12442 · vLLM v0.20.0 hybrid-SSM-disaggregation blog(2026-04) · DUET
  (DAC'26) arXiv 2603.15530 — **전부 검증 필요**(§0.1과 동일 caveat).

---

## 이 문서의 상태

이 문서는 살아있는 전략 문서다(`CONSENSUS.md` §4 "살아있는 문서" 표에 전략
문서로 등재). 새로운 서빙 측정이나 claim 등급 변경이 있으면 이 문서가 아니라
`CLAIM_EVIDENCE_MATRIX.md`/`PROJECT_STATUS.md`를 먼저 갱신하고, 이 문서는
그 변경을 반영해 갱신한다(반대 방향 금지).
