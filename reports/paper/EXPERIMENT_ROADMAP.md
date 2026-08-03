# R2 experiment roadmap

최종 갱신: 2026-08-03(같은 세션 4차 속행 — doc-steward 기록, **상태 기록
· 성능 판정 0건 · GPU 런(S2)은 별도 제출 중·결과 없음**. 3차 속행이 연
§0의 이분법((i)/(ii))이 **유지 불가**로 판정됐다 — `split_frac≥0.90`이
D 파티션 실행 토큰을 순수하지도 완전하지도 않게 잡는다는 **세 번째 후보
(iii)**가 실측으로 문서화됨(claims-auditor 자기감사 재프레이밍 +
result-analyst 독립 재현 `S0R_REPLICATION_2026-08-03.md`: 행 1·3 재현,
행 2 순서만, 행 5(클럭) 미발화, ★행 4(음성대조)가 UNSPLIT도 같은 슬로우
모드를 가짐을 보여 강한 형태를 죽임 — 살아남는 건 농축 2.33–3.24×뿐).
남은 두 읽기는 **오프라인 분리 불가**, S2가 인과 시험. **철회 3건**
(메인 세션이 같은 날 앞서 씀): "§0 stands as written"·"aggregation-
invariant"·"11.09는 집계 단위 미기록"(**틀림** — 산출자는
`m3_conditional.report_conditional` [3] `sp_p50=11.0905`,
`m3_conditional.py:158-161,251-262,316-329`에 문서화). **재사용 계측
결함 2건**: `c2_anchor.py` 표 [5]가 M8 전체·Ha8 d16을 조용히 누락
(측정 부재 아니라 텔레메트리 앵커 부재) · mode estimator 60ms 상한이
arm-이식 불가. **게이트 정의(3.4.4 결정규칙) 변경 없음.**
`G_LEVER`/`G_FLAT`는 여전히 UNDETERMINED. 상세
`../../PROJECT_STATUS.md` "8B decode-SM 프론티어" "2026-08-03(4차)"
소절, `../CONSENSUS.md` §1-30·§3-18·§3-19,
`../../workspace/engine-port/results/s8_frontier/DESIGN.md` §4.3.15,
사전등록 3건(`PREREG_S0_AXIS_2026-08-03.md`·`PREREG_S0R_MODE_2026-08-03.md`·
`PREREG_S2_STICKY_ITL_2026-08-03.md`). 이전(같은 세션 3차 속행 —
doc-steward 기록. C2 →
`G_LEVER`/`G_FLAT` 앵커 도출 시도(`c2_anchor.py`)를 claims-auditor가
감사해 **경로 폐기**(주장 1만 CONFIRMED, 2–5 REFUTED/NOT-YET-SUPPORTED).
**`G_LEVER`/`G_FLAT`는 여전히 UNDETERMINED**, 다음 시도는 감사자 발안
(α)(β)에 대한 독립 사전등록이 선행돼야 함. ★**신규 최상위 열린 항목**:
C2(865493)와 872077(E1 격자)의 "decode 16 SM" per-token ITL이 같은 arm·
서버 플래그·매칭 batch에서 **2.6× 다름**(28.79ms vs 11.09ms) — 872077의
`decode_sms==16`이 실제 16-SM 하드웨어 실행인지 미검증(`DESIGN.md`
§4.3.11의 잔여층), 또는 C2 값이 셀 배치 성질인지 미해소. **sticky 격자
제출보다 이 모순 해소가 선행돼야 한다.** D=54 앵커 측정(jobs 872920/
872921)은 keepalive 재현성 결함으로 취소, 독립 수렴으로 C2 high-residency
=워크로드 장치 산물임을 확인. **게이트 정의(3.4.4 결정규칙) 변경 없음.**
상세 `../../PROJECT_STATUS.md` "8B decode-SM 프론티어" "2026-08-03(3차)"
소절, `../CONSENSUS.md` §1-28·§1-29,
`../../workspace/engine-port/results/s8_frontier/DESIGN.md`
§4.3.13–4.3.14. 이전(같은 날 2차 속행 — doc-steward 기록. (I)
claims-auditor가 §1-26/여기 아래 기록된 `g` 은퇴의 근거였던 `A_free`
결함을 대체하는 **조건부 per-token 추정량**(`m3_conditional.py`)으로
estimand를 이관[AUDITED, blocking-threshold 스윕만 UNAUDITED — 감사자
자기산출 자기감사]. (II) engine-porter가 `PDMUX_STICKY_PARTITION`
구현·correctness gate 통과[구현 사실, **구현 완료 ≠ 성능 주장 성립**] —
CPU 회귀 40 tests + sticky 단위 테스트 12 + GPU smoke(job 872800)
byte-identical 출력. sticky 격자 런은 **여전히 미제출**, 제출 전
`G_LEVER`/`G_FLAT` 사전등록이 미결 열린 항목. **게이트 정의(3.4.4 결정규칙)
변경 없음.** 상세 `../../PROJECT_STATUS.md` "8B decode-SM 프론티어"
"2026-08-03(2차)" 소절, `../CONSENSUS.md` §1-27,
`../../workspace/engine-port/results/s8_frontier/DESIGN.md`
§4.3.10–4.3.12. 이전(같은 날 1차 속행) — job **872077**(M3)의 NO VERDICT
사유가 "CI 폭 부족"에서 **"estimand 미식별"**로 확장됨을 P6에 기록. 기판이
prefill 비-in-flight 시 항상 무분할로 되돌아가므로 이 격자에서는
"decode가 D SM에서 돌았다"⟺"prefill이 동시 in-flight였다"가 같은 사건이라
`g=A_free(d16)/A_free(d54)`는 **이 격자 한정 은퇴**(sticky-partition
기판 수정 전 인용 금지, 블록 증설 재실행 선행 금지). 메인 세션이 세운
"decode 실현 4–19%가 `g`를 attenuate했다"는 보정 가설도 claims-auditor에
**REFUTED**. 다음 gate = `PDMUX_STICKY_PARTITION` 구현 → sticky 격자
1회(872077 대조) → 사전등록 판별 예측(T8≈1.85·Ha8≈0.92 vs Ha8≈1.6).
게이트 정의(3.4.4 결정규칙) 변경 없음. 상세 `../../PROJECT_STATUS.md`
"8B decode-SM 프론티어" "2026-08-03" 소절, `../CONSENSUS.md` §1-26. 이전
rev2: ★P6의 E1 "설계 위험 3중" 중 (b)(c)를 claims-auditor 회부 결과로 **정정**하고, E1 대신 제출된 M3 Transformer-control 대조[job 872077]를 기록. 게이트 정의 변경 없음. 이전 rev1: 진행 상태 갱신만 — P6에
"E1 상태(2026-08-02)" 추가: 전제 실험 4건 완료(2026-08-01, **claims-auditor
미통과 = 인용 금지**), **본 스윕 미제출**, 설계 위험 3중으로 E1이 사전등록
분기 "설계상 이 질문에 도달할 수 없다"로 갈 위험, 선행 사전등록
`--max-mamba-cache-size` 공통 상수 고정). 이전:
2026-07-28(★★★claims-auditor 감사 — P6의 Stage 0/L−2 게이트 판정
[non-binding]을 **철회**(C1 CONFIRMED: D108 앵커가 실은 decode 16 SM). L−2는
"실행 완료"가 아니라 "게이트 미실행"으로 정정. 대신 8B decode-SM 민감도 측정
노트[C2, scoped]와 그 프론티어 게이트 E1–E4를 추가 — 아래 P6 절 갱신). 이전:
2026-07-26(P6 long-context Stage 0/L−2 게이트 실행 완료 — non-binding,
아래 P6 절 갱신 — ★2026-07-28 철회, 위 참조). 2026-07-25(벡터2 재프레이밍 — cross-substrate serving 이식
[XS-series]을 "불필요·부적합"으로 하향 후 새 게이트 "Transformer-control on
green-context"[TC-series]로 교체, positioning 판정)

## 벡터1 (disjoint conflict-regime escape hatch) — 별도 트랙, CONFIRMED closure (scoped, 종결)

이 항목은 P0–P6 Claim D/E gate 배선 밖의 별도 트랙이었다(`reports/CONSENSUS.md`
§5-8(c) 추적, `PROJECT_STATUS.md` "벡터1" 절, `reports/paper/CLAIM_EVIDENCE_MATRIX.md`
Claim F 참조). **2026-07-25 CONFIRMED closure(scoped)로 종결** — 아래 5단계
sweep 계열의 최종 판정.

1. **g2_0_full**(2026-07-24, 1차 스윕) — razor-thin disjoint(feasible-A ∩
   feasible-B = ∅) 발견.
2. **g2_0_hard**(2026-07-24/25, hardening 재스윕) — **ILL-POSED at rA5**: 1차
   disjoint가 재현되지 않음(TTFT 3s-cliff bimodality), "disjoint 소멸" 관측은
   별도 ITL-p95 percentile-window 아티팩트로 판명.
3. **g2_0_decliff stage-1**(jobs 863880–863948) — `rA{2,3,3.5,4}×{d16,d44,d54}`
   스캔, `rA=2`만 clean off-cliff(n=6 확증). 유일한 clean 지점에서 static `d54`가
   양 phase 동시 커버(`feasible-A={d16,d44,d54} ∩ feasible-B={d54} = {d54} ≠ ∅`)
   하나 **PLAUSIBLE closure, CONFIRMED 아님**(claims-auditor 반증 3항목: off-cliff
   에서도 살아있는 split→TTFT gradient·d54 배제 onset 미측정 전이대·
   "binding-A⟺on-cliff" 미증명).
4. **g2_0_rasweep**(120 job, `rA{2.25,2.5,2.75,3.0,3.25}×{d16,d34,d44,d54}×n6`) —
   off-cliff sub-band(rate≤2.75)에서 disjoint 재확인 없음, 전이대를 rate
   3.0–3.5로 좁힘.
5. **g2_0_raconf**(claims-auditor pre-registered 24-job 확증 열, `rate{3.5,3.75}×
   {d44,d54}×n6`) — 결정 규칙(어떤 rate서든 d54 견고히 <0.7(p90>3s, unimodal) ∧
   d44/d16 동시에 견고히 ≥0.95·off-cliff(p90<2s)면 disjoint 실재→REOPEN, 아니면
   d54가 양 phase 동시 커버하는 companion collapse면 CONFIRMED)를 **companion
   collapse로 판정**: rate3.5 d44 0.953±0.035≈d54 0.948±0.035(d54 failTTFT=0/6);
   rate3.75 d44 0.932±0.042 < **d54 0.948±0.062**(d54가 더 높음). REOPEN 전제
   양쪽 붕괴 — d54는 어느 rate서도 <0.7이 아니고 d44도 어느 rate서도 견고히
   ≥0.95가 아님(웜업성·split-대칭적 TTFT-blowup). d54는 Phase B 유일 feasible
   split이면서 Phase A도 d44와 대등하게 커버 → 단일 split이 양 phase를 시간축
   에서 커버 → **disjoint 없음, 최종 확정**.

★**필수 caveat**: magnitude는 ill-posed(metric cliff, run-length 의존)이나
**순위(d54≈d44, d54 미선-배제)는 견고**. Phase-B d44 0.188은 50ms 경계 바로 위라
magnitude fragile·방향 견고. scope는 {Zamba2-2.7B, ctx4096, Phase A in2048/o32,
Phase B in2048/o512@rB4, triton attn+mamba, disable-radix-cache, cudagraph-ON,
A100 108-SM green-context pdmux, SLO=TTFT 3s ∧ per-req ITL-p95 50ms, inter-phase
drain된 순차 2-phase, rate_A≤3.75}에 한정 — **"hybrid엔 disjoint 없음"으로
일반화 금지**. closure는 얽힘 억제(drain) 조건 관측 = 필요조건 bound이지 hot
varying-trace(Claim C 얽힘) 실증 아님.

이 트랙의 결과는 Claim D/E나 §1-20(spatial coupling-tax)에 영향을 주지 않는다 —
시간적 disjoint와 공간적 coupling-tax는 별개 축. **남은 방향(후속, 미실행)**:
(i) long-context(decode floor가 ctx 상승에 따라 올라가는 영역 — CONSENSUS
§1-5 — 충돌이 발생할 수 있음, 모델/ctx 교체 필요), (ii) §1-20 spatial
decoupling(별도 device pool disaggregation, +16% headroom). 상세 verdict:
`workspace/engine-port/results/g2_0_full/disjoint_verdict_2026-07-24.md`,
`workspace/engine-port/results/g2_0_hard/hardened_disjoint_verdict_2026-07-25.md`,
`workspace/engine-port/results/g2_0_decliff/decliff_verdict_2026-07-25.md`,
`workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md`.

## 벡터2 (substrate-robustness 식별) — Transformer-control on green-context (2026-07-25)

`reports/paper/venue_positioning.md` §0.1(2026-07-25, venue-strategist
prior-art 조사)의 판정: 이 논문의 central negative는 substrate-robustness 축으로
두 갈래다 — **(A) green-context 종속**(layer-aware 死·cudagraph 비양립, Claim
B) vs **(B) mechanism-independent 후보**(lever-weakness=mamba decode
SM-둔감·entanglement·decode 비대칭, Claim A/C).

⚠️★**2026-07-25 하향·재프레이밍(사용자 지적) — cross-substrate serving 이식
(XS0/XS1/XS2)은 불필요·부적합.** 이 절의 초판은 두 번째 substrate(libsmctrl/MPS)
serving 이식을 "make-or-break 필수 게이트"로 걸었으나 **철회**한다. 이식은
불필요할 뿐 아니라 부적합하다:

- **MPS**: SM 파티션이 프로세스별·정적 → 런타임 동적 PD-mux 불가 +
  단일-프로세스 `event_loop_pdmux`의 멀티-프로세스 전면 재구조화 비용이 실익
  초과.
- **libsmctrl**: NVIDIA 비제공 리버스-엔지니어링(per-arch SM 마스킹) →
  하드웨어 세대·드라이버 귀속(driver-580 BLOCKED가 증거). 배포 근거에 비-vendor·
  비-이식 의존성을 들이는 셈.
- ★**green-context = 배포 primitive 방어**: NVIDIA 공식 fine-grained SM
  primitive는 green-context 하나(CUDA Green Contexts 12.4+)뿐. ⇒ "libsmctrl
  쓰면 되잖아"의 답 = "green-context가 배포 가능한 유일 vendor primitive다.
  DuetServe/Bullet의 동적-승은 libsmctrl(세대 귀속·비-vendor) 위에서만 성립 →
  libsmctrl에서 hybrid 동적이 이겨도 이식 불가한 research curiosity이지 배포
  가이드라인의 반례가 아니다." 따라서 (A) layer-aware 死는 green-context-bound로
  정직히 스코프하고, 그 스코프를 libsmctrl 비-이식성이 오히려 받쳐준다.

따라서 XS0/XS1/XS2(별도 substrate serving 이식)는 **실행하지 않는다.** 대신
Risk 2(모델 vs substrate 귀속)를 **기존 green-context 위에서** 닫는 값싼
식별 실험 3수로 교체한다:

- ★**신규 게이트 = Transformer-control on green-context**: 순수 Transformer
  (예: Qwen/Llama)를 기존 pdmux(green-context)에 통과시켜 hybrid와 **같은
  green-context + 같은 conjunctive-SLO**에서 대조한다. drain 비용은 두 모델에
  동일하게 작용 → **상쇄**. 이 벡터의 ID prefix `TC`(Transformer-Control)는
  P4 baseline ID `B0`–`B8`와 별개다.
  - **TC0**: 순수 Transformer 모델(Qwen/Llama류)을 기존 green-context pdmux에
    배선(모델 로딩·correctness gate).
  - **TC1**: hybrid와 동일 워크로드/SLO/split-grid에서 reactive dynamic vs
    decode-heavy static을 측정.
  - **결정 규칙(사전 등록)**: **Transformer 동적-승 ∧ hybrid 동적-패 → flip은
    substrate·메트릭 고정 하에 모델(hybrid) 귀속 확정**(Risk 2 닫힘, 진짜 식별).
    **둘 다 동적-패 → negative는 hybrid가 아니라 메트릭(conjunctive-SLO
    goodput)+배포-primitive(green-context) 탓으로 재프레이밍**(여전히 유효하나
    다른 기여).
- **보강 (실행 불필요·기존 데이터)**: (2) **lever-weakness = roofline
  microbenchmark**(r0c SM-민감도 데이터 보유) — mamba decode SM-둔감은
  연산강도 성질이라 primitive-robust; "libsmctrl이 고친다"는 반론은
  drain(=(A))에만 닿고 lever(Claim A)엔 안 닿음. (3) **헤드라인 HE0는 이미
  entanglement 귀속으로 측정 완료**(`switch_count`≈0·컨트롤러 0.014% 직접 계측
  → 동적-패가 overhead/drain 탓 아님) → drain-아티팩트 반론은 (A)에만 닿고
  헤드라인 무관.

벡터1(disjoint conflict-regime, 시간축 disjoint-feasibility)과는 무관한 별개
트랙. 벡터2는 이제 파티셔닝 primitive 불변성을 **cross-substrate 이식이 아니라
green-context 위 모델-대조(+기존 microbench/telemetry)**로 식별한다.

## 공통 방법

- 먼저 B1 sustainable SLO rate `lambda*`를 모델별로 측정한다.
- configuration당 최소 5회, paired CI가 0을 교차하거나 variance가 크면 10회
  이상 수행한다.
- 모든 pair는 동일 immutable trace/hash, workload seed, server seed를 사용하고
  node 내 실행 순서를 randomize한다.
- warm-up/correctness/benchmark phase를 분리한다.
- CUDA Graph, backend, GPU clock/power, KV capacity, max-running을 고정한다.
- mean, median, SD, paired bootstrap 95% CI와 percent effect를 보고한다.
- request의 TTFT와 request 내부 token-level ITL p95가 모두 SLO 이하여야 primary
  goodput에 포함한다. 기존 request-mean-ITL score는 secondary로만 보존한다.
- 3% 미만 차이는 headline improvement로 사용하지 않는다.

## 단계와 stop/go gate

### P0 — 상태/R1 정정

`PROJECT_STATUS.md`, claim matrix, R1 재분석, stale banner를 정본으로 반영한다.

### P1 — 계측 observer effect

동일 fixed split에서 다음 네 arm을 paired AB/BA로 비교한다.

1. legacy, telemetry off
2. legacy, symmetric telemetry on
3. R1 observer state on, trace off
4. R1 observer state와 trace on

각 overhead effect가 3% 미만이고 CI가 0을 포함해야 진행한다. 실패 시 buffer,
sampling, writer를 수정하고 architecture 비교를 보류한다.

### P2 — Architecture

- legacy fixed D24/D44
- true dual fixed D24/D44

Decode-heavy/alternating에서 decode progress, ITL 또는 oldest queue age가
유의하게 개선되고 throughput regression이 3% 이하일 때만 Claim D를 채택한다.
그렇지 않으면 true dual은 negative architecture result로 남긴다.

### P3 — Offline profile/estimator

- CUDA Graph on, 실제 full-model decode
- D16/D24/D34/D44/D108
- batch 1/4/8/16/32/48
- context 256/1K/4K/8K, 지원 시 16K
- point당 warm-up 후 30 step 이상, 5회 이상

Profile은 model/revision/config hash, engine commit, GPU/driver/backend/graph,
attention/SSM count와 ratio, GQA metadata, latency percentile/residual을 기록한다.

Acceptance:

- upper-bound empirical coverage ≥95%
- under-reservation epoch ≤1%
- median over-reservation ≤한 state
- unseen/stale profile은 D44 fallback, live violation은 D108/admission limit

### P4 — Baseline와 profile ablation

| ID | Policy |
|---|---|
| B0 | vanilla continuous batching |
| B1 | training workload에서 선택한 one global static |
| B2 | per-workload offline best static oracle |
| B3 | 기존 single-worker dynamic |
| B4 | true dual fixed |
| B5 | true dual generic dynamic |
| B6 | true dual Hybrid-informed dynamic |
| B7 | layer-granular negative baseline |
| B8 | future trace와 transition/dwell cost를 아는 offline oracle |

Profile ladder는 no profile, model-size only, attention-ratio aware, full Hybrid
profile 순으로 동일 held-out point/trace에서 평가한다.

Claim E acceptance:

- target 영역에서 B6가 B1/B5보다 paired CI 기준 유의하고 ≥3% 개선
- non-target 영역에서 >3% regression 없음
- B2/B8은 upper bound로만 보고하며 이를 이긴다고 주장하지 않음

### P5 — Mechanism

대표 target/non-target point에서 Nsight Systems/Compute로 kernel timeline,
GPU idle gap, stream/event overlap, graph replay, SM active, occupancy, Tensor Core,
DRAM/L2를 수집한다.

### P6 — Workload

★**long-context(W5 등)의 논문적 역할(2026-07-25 positioning 판정,
`venue_positioning.md` §0.1(4))**: `longcontext_trace_plan.md`의 long-ctx
트랙(Stage 0/L−2 → L3/L3s)은 negative→가이드라인 전환과 "언제 유효한가" 경계
획정, 그리고 granularity 비용의 모델-composition 독립성(ctx-불변 구조적
성질)을 보이는 데 유효하다. **substrate 귀속(Risk 2)은 long-ctx가 아니라 위
벡터2(green-context 위 Transformer-control 대조 + roofline microbench + 기측정
entanglement 귀속)가 닫는다** — 별도 substrate serving 이식은 불필요·부적합으로
철회됐으므로, long-ctx가 "이식의 대체재"일 필요도 없다. 두 트랙은 서로 다른
질문(long-ctx=ctx-regime 경계, 벡터2=primitive/모델 귀속)을 담당한다.

★★**Stage 0(L−2) 게이트 실행(2026-07-26) — non-binding, ★★★2026-07-28 철회
(claims-auditor 감사, C1 CONFIRMED)**
(`../stage0_verdict_2026-07-26.md`, jobs 864230+864601): 2026-07-26엔 운영점서
decode SM-무감각이 hybrid·pure-Transformer·pure-Mamba 전부, ctx≤16k 전부로
확인됐다고 기록했으나, 근거였던 "D108 무경합 앵커"가 실은 decode 16 SM이었음이
3중 독립 증거(코드 기전·telemetry 재집계·클라이언트 서명)로 확인돼 **판정을
철회**한다. **L−2는 SM-binding 여부를 측정한 적이 없다** — "negative→가이드라인
전환" 역할은 실현되지 않았고, `longcontext_trace_plan.md` §6의 L−1 이상은
"게이트 실패로 보류"가 아니라 **"게이트 미실행"**이다.

★★**대신(2026-07-28) 8B decode-SM 민감도 측정 노트 — C2 CONFIRMED(scoped)**
(`../../workspace/engine-port/results/s8_scaleup/FINDINGS_8B_2026-07-28.md`,
jobs 865289–865533): prefill을 16 SM에 고정한 채 decode-SM만 16→92로 올리면
decode ITL이 **2.36–2.91×**(4 arm, 모델-무관) 개선된다 — Stage 0가 주장하던
"SM-무감각"과 정반대 방향. 단 이는 **decode 측 등량곡선**(예산 제약
`prefill+decode≤108` 없음)이라 **레버 존재만 확립**하며 정책 이득 근거가
아니다. **claims-auditor가 지정한 프론티어 게이트 E1**(`[108−D,D]` 스윕,
D∈{16,24,44,54,92}+best-static 대조, 4 arm, offered-rate 고정, n≥4, 사전등록
파티션 점유율≥0.80·활성률≥0.60 게이트, 결정규칙: best static 대비 conjunctive
goodput ≥3% 개선 & paired CI가 0 배제)가 이 레버가 예산 제약 하 net-positive인지
판정한다 — 병행 게이트 E2(ctx 확장)·E3(duty-cycle, 설계상 종결)·E4(C2b는 통제
불가하므로 주장 폐기). 상세 `../PROJECT_STATUS.md` "8B decode-SM 민감도 측정
노트"·"열린 긴장"·"다음 실험 gate" #8.

★★**E1 상태(2026-08-02) — 본 스윕 미제출, 설계 위험 3중**. 2026-08-01에 전제
실험 4건이 완료됐다(jobs **870295**=M8/**870296**=Ha8/**870297**=Hs8 용량
스캔 각 100 probe, **870301**=T8 batch-cap 24 probe, 전부 오류 0; 원자료
`../../workspace/engine-port/results/s8_frontier/`). ⚠️**이 4건은 전부
claims-auditor 미통과 = 미검증, 인용 금지**이며 수치는 `../PROJECT_STATUS.md`
"열린 긴장"의 "2026-08-01 실험 4건" 소절에만 둔다. 로드맵 차원에서 기록할
것은 **판정이 아니라 설계 위험**이다: (a) 사전등록 사다리 {50,60,80}ms가
as-run 설정에서 **네 arm 전부 헤드라인 룽 없음**, (b) 그 as-run 설정
(`--max-running-requests 48`)이 ITL·TTFT 두 축을 반대 방향으로 왜곡함이
직접 실험으로 드러남, (c) on-cliff 제외 규칙(d92 knee 2.80, 전 arm)이 **어떤
동작점에서도 decode-rich 끝을 제거** — C2 레버가 사는 끝.

> ★★★**정정(2026-08-02, claims-auditor 회부 + M1/M2/M4 후속) — 위 세 다리 중
> 둘이 무너졌고, 그래서 "설계상 도달 불가" 종결은 정당화되지 않는다.**
> - **(c) 기각.** C2 측정(`s8_scaleup/FINDINGS_8B_2026-07-28.md` §2)에서
>   SM16→44 구간이 log-range의 **75–81%**를 차지한다 ⇒ d92 하나 제외는 레버가
>   사는 끝이 아니라 **마지막 15–25%**만 자른다. 게다가 "공통 knee 2.80"
>   자체가 철회됐다(knee의 치역이 probe 격자뿐이라 일치가 부분 강제 —
>   살아남는 건 **순서**뿐, `results/s8_frontier/DESIGN.md` §4.3.7).
> - **(b) 운영구간 밖.** cap 왜곡은 rate 16에서만 관측됐고, 운영대역(≤2.80)
>   실측 동시성은 12–44 < cap 48이라 **cap이 구속할 수 없다** ⇒ E1에 적용 안 됨.
> - **(a)만 생존**하되 arm×룽 표는 **rate-confound로 폐기**(T8만 rate 12).
>   공통 rate로 재계산하면 `HEADLINE-ELIGIBLE RUNGS = NONE`은 **유지**된다
>   (두 estimand × 두 seed 전부).
>
> ⇒ **(A) 제출도 (B) 종결도 시기상조**로 판정하고, 대신 **M3
> Transformer-control 대조**를 사전등록·제출했다(job **872077**, §4.3.8(c)).
> M3는 goodput 이득이 아니라 **"ITL 축이 D에 반응하기는 하는가"**를 묻는다 —
> 그 질문이 긍정이어야 프론티어 질문이 성립하기 때문이다. 오프라인 예비값:
> blocking 제거 후 d16→d54 기울기가 T8 2.03× 대 Ha8 1.03×/0.89×.
> **선행 필수 정정**: `--max-mamba-cache-size`는 **전 arm 공통 절대상수가
> 아니라 `= cap` 규칙**으로 고정해야 한다 — slot당 비용이 arm마다 달라
> (M8 0.255 / Ha8 0.141 / Hs8 0.096 GB) 절대상수는 arm마다 다른 메모리 분할을
> 강제하는 **새 cross-arm 교락**이 된다(`../CONSENSUS.md` §1-23 따름정리 정정).
>
> ★★★**정정(2026-08-03, claims-auditor, 같은 세션 속행) — M3(872077)의
> NO VERDICT 사유가 "CI 폭 부족"에서 "estimand 미식별"로 확장됐다.** 코드
> 사실: `pdmux_context.py:initialize_stream_groups`가 마지막에 무조건
> `(0,108)` 무분할 그룹을 덧붙이고 prefill이 비-in-flight면 그리로 되돌아간다
> (`multiplexing_mixin.py:773,792-794`) ⇒ 이 격자에서는 **"decode가 D SM에서
> 돌았다"와 "prefill이 동시 in-flight였다"가 같은 사건**이다. §1-24(ITL 꼬리
> =monolithic prefill, 크기가 108−D에 단조)와 결합하면 `g`는 사전에
> **"decode-SM 탄력도 라벨을 단 prefill-SM 탄력도"**일 것이 예상되고, 실측
> (UNSPLIT-only 부분집합만으로 T8 헤드라인 재현)이 그와 일치한다. **이는 n을
> 늘려도 해결되지 않는 설계 결함**이라 `g = A_free(d16)/A_free(d54)`는 **이
> 격자 한정 은퇴**(sticky-partition 기판 수정 전 인용 금지), 블록 8→12–16
> 증설 재실행은 **선행 금지**. 동시에 메인 세션이 세운 "decode 실현 4–19%가
> `g`를 attenuate했다"는 보정 가설도 **REFUTED**(control-arm reductio: T8에
> 같은 보정 적용 시 corrected g 21–29×로 C2를 10배 위반; de-engagement 직접
> 실험에서 w=0에도 g 1–11%만 이동; "A(108) 셀 무관" 가정이 UNSPLIT-only
> 부분집합 분해로 반증). ⚠️"Ha8에 레버가 없다"는 CONFIRMED 아님 — **긴장
> A(HE2 vs C2)는 전혀 닫히지 않았다.** 다음 gate: `PDMUX_STICKY_PARTITION`
> 구현(engine-porter, correctness gate) → sticky 격자 1회(872077 동일 설계
> 8 block, non-sticky 대조로 872077 사용) → 사전등록 판별 예측(prefill
> 주도라면 T8≈1.85·Ha8≈0.92로 하강, 희석 가설이 옳았다면 Ha8≈1.6로 상승 —
> CI 비중첩이라 8 block으로 구분 가능) → `E1_DECODE_REALIZED≥0.90`이
> sticky에서는 **진짜 게이트**(더 이상 항등식 아님). 상세 `../CONSENSUS.md`
> §1-26, `../../PROJECT_STATUS.md` "8B decode-SM 프론티어" "2026-08-03"
> 소절, `../../workspace/engine-port/results/s8_frontier/DESIGN.md` §4.3.9.
>
> ★★**(2026-08-03, 같은 세션 2차 속행) 위 두 다음-gate가 모두 진행됐다 —
> estimand 이관 완료, 구현 완료, 런은 아직 미제출.** claims-auditor가
> `A_free`를 대체하는 **조건부 per-token 추정량**(`m3_conditional.py`,
> `DESIGN.md` §4.3.10)을 만들었다[AUDITED, blocking-threshold 스윕만
> UNAUDITED]. engine-porter가 `PDMUX_STICKY_PARTITION`을 구현했고(`DESIGN.md`
> §4.3.11) correctness gate 전부 PASS(CPU 회귀 40 tests + sticky 단위
> 테스트 12 + GPU smoke job 872800, 6개 고정 프롬프트 OFF/ON greedy 출력
> byte-identical) — **구현 완료 ≠ 성능 주장 성립**이며, realized 관측(n=1)
> `E1_DECODE_REALIZED` OFF 0.0839 → **ON 1.0000**(사전등록 게이트 ≥0.90
> 초과)만 기록됐다. **sticky 격자 런은 여전히 미제출** — `DESIGN.md`
> §4.3.12가 사전등록한 4개 게이트·primary 통계량은 확정했으나
> **`G_LEVER`/`G_FLAT`는 미결정으로 남겨 두었다**(기존 1.5/1.15는 `A_free`
> 스케일이라 새 추정량에 그대로 이전 불가 — C2 측정범위[2.36–2.91×]에
> 묶는 안이 논거는 있으나 E1의 D 범위[16→54]가 좁고 상보적[P+D=108]이라
> 확정 전 별도 사전등록 필요). 판별 예측(위 문단)은 불변. 상세
> `../CONSENSUS.md` §1-27, `../../PROJECT_STATUS.md` "8B decode-SM
> 프론티어" "2026-08-03(2차)" 소절, `DESIGN.md` §4.3.10–4.3.12.
>
> ★★★**(2026-08-03, 같은 세션 3차 속행) `G_LEVER`/`G_FLAT`의 미결정을
> C2 데이터로 닫으려던 시도 — 경로 폐기, UNDETERMINED 그대로.**
> `c2_anchor.py`로 시도한 5개 주장을 claims-auditor가 감사: **주장
> 1(realized 검증)만 CONFIRMED**(서술 2건 정정 필요 — 활성률은 count
> 가중, 108 SM 시간은 drain 전용), **주장 2(primary p95→p50)는 관측
> CONFIRMED·처방 REFUTED**(p50 전환 시 872077 T8 양성대조조차 1.00으로
> 무너져 캠페인을 구조적 NO VERDICT로 만드는 처방이었다 — **primary는
> `p95(SPLIT)` 유지**), **주장 3–5는 NOT-YET-SUPPORTED/REFUTED/REFUTED**
> (`G_LEVER=1.41`은 끝점 선택만으로 [1.41,2.40] 전 구간 도달 가능해
> REFUTED, `G_FLAT=1.25`는 LOO 실측 반폭이 1.30인데 Ha8 1.340으로
> 자기 데이터서 뒤집혀 REFUTED). ★**§0 신규 최상위 열린 항목**: 같은
> arm·서버 플래그·매칭 batch에서 C2(865493)와 872077의 "decode 16 SM"
> per-token ITL이 **2.6× 다르다**(28.79ms vs 11.09ms) — 872077의
> `decode_sms==16`이 실제 하드웨어 16-SM 실행인지(`DESIGN.md` §4.3.11의
> 미검증 잔여층) 또는 C2 값이 셀 배치 성질인지 미해소, **872077 전체와
> sticky 결과가 딛고 선 바닥**. **sticky 격자 제출은 이 모순 해소 이후로
> 미룬다.** 별도로, D=54 앵커 측정(jobs 872920/872921)이 keepalive 토큰
> 초과(재현성 결함)로 취소됐고, 독립 수렴으로 **C2의 높은 residency는
> decode 파티션 제어가 아니라 keepalive 워크로드 장치의 산물**임이
> 확인됐다(C2 앵커가 죽는 세 번째 이유). `G_LEVER`/`G_FLAT`는
> **UNDETERMINED로 유지**, 다음 시도는 감사자 발안 (α) sticky 파일럿
> 양성대조 효과크기 또는 (β) arm 간 대비 `g_T8/g_Ha8`(batch-매칭
> rate)에 대한 **독립 사전등록**이 선행돼야 한다(감사자가 자기 발안의
> 승인 주체일 수 없다). 상세 `../CONSENSUS.md` §1-28·§1-29,
> `../../PROJECT_STATUS.md` "8B decode-SM 프론티어" "2026-08-03(3차)"
> 소절, `DESIGN.md` §4.3.13–4.3.14.
>
> ★★★**(2026-08-03, 같은 세션 4차 속행) §0의 이분법이 유지 불가로 판정
> — 세 번째 후보 (iii) 실측 문서화, 오프라인 분리 불가, GPU(S2) 별도
> 제출 중·결과 없음. 성능 판정 0건.** §0은 (i) 872077의 `decode_sms==16`
> 이 실제 16-SM 실행이 아니다 / (ii) C2의 28–31ms가 셀 배치 성질이다 중
> 하나가 거짓이라는 이분법이었다. claims-auditor가 `FINDINGS_S0_AXIS_
> 2026-08-03.md`를 감사하며(자기감사, 방법론 교훈 12) 이 이분법이
> "`split_frac≥0.90`이 D 파티션 실행 토큰을 올바로 분리한다"는 전제 위에
> 서 있고 그 전제가 T8 세 공유 셀 전부에서 깨진다는 재프레이밍을 냈다 —
> E1 SPLIT 모집단이 이봉이고 윗봉이 C2 셀별 p50과 1–2% 일치, 아랫봉은
> 같은 job UNSPLIT과 통계적으로 동일. **독립 재현**(result-analyst,
> `S0R_REPLICATION_2026-08-03.md`, claims-auditor도 사전등록 세션도
> 아님, 감사된 `m3_conditional.py` 프리미티브만 프리미티브별 재사용
> 선언·생산자 자체에 gate 대조): 행 1·3 재현, 행 2는 순서만(d24−d16
> 미해결), **행 5(클럭 lag) 미발화**(최적 δ=+0.10s서도 slow share
> 11.4%). ★**행 4(음성대조)가 강한 형태를 죽였다**: 같은 estimator를
> UNSPLIT(108 SM)에 적용하면 T8 전 셀에서 동일한 슬로우 모드가 나타나고
> 그 위치가 셀을 따라간다(33.88→22.12→15.62→14.12ms, D 16→24→44→54;
> d16 슬로우 토큰 8,893개 중 7,903개=88.9%가 UNSPLIT 라벨) ⇒ SPLIT은
> 배타가 아니라 **농축**(2.33–3.24×). ⇒ **세 번째 후보 (iii)**: 두
> job은 같은 축이나 라벨이 순수하지도 완전하지도 않다 — §0은 이제
> 3지선다이며 오프라인으로 분리 불가. 남은 두 읽기(셀 수준 현상 vs 클럭
> 오프셋 누출)는 **S2**(GPU, `PREREG_S2_STICKY_ITL_2026-08-03.md`, 별도
> 제출 중, 결과 없음)만이 인과적으로 분리 가능. **철회 3건**(메인
> 세션이 같은 날 앞서 씀, `FINDINGS_S0_AXIS_2026-08-03.md` 배너와
> 동일): "§0 stands as written"·"aggregation-invariant"·"11.09는 집계
> 단위 미기록"(**틀림** — 산출자는 `m3_conditional.report_conditional`
> [3] `sp_p50=11.0905`, n=11,124, `a_free_only=True`,
> `m3_conditional.py:158-161,251-262,316-329`에 문서화, 없는 건 stdout
> 저장분뿐). **재사용 계측 결함 2건**: `c2_anchor.py` 표 [5]가 M8 전체·
> Ha8 d16을 조용히 누락(`meta`가 `t0_monotonic_s` 분기 안에서만 채워짐,
> `c2_anchor.py:181-187` — Ha8 d16은 돌았다, `itl_ms_p50=112.84`
> n=5200, 빠진 건 텔레메트리 앵커뿐) · mode estimator 60ms 상한은
> arm-이식 불가(Ha8은 토큰의 0.16%만 창 안). **증거 수준**: 강한
> 형태(레버=D-SM 실행)는 **채택 불가**(음성대조 반증), 약한 형태(이봉·
> 농축)는 **재현됨(독립성 부분적** — 추정량은 감사자 제안, 사전등록은
> 메인 세션, 실행만 독립**)**. 게이트 S1(§4.3.13)은 "부분 실현" 분기가
> 없어 **현 상태로 실행 불가**(4번째 분기 필요). `G_LEVER`/`G_FLAT`는
> §4.3.12(d) 그대로 **UNDETERMINED 유지**(이번 회차로도 미해소).
> **게이트 정의(3.4.4 결정규칙) 변경 없음.** 상세 `../CONSENSUS.md`
> §1-30·§3-18·§3-19, `../../PROJECT_STATUS.md` "8B decode-SM 프론티어"
> "2026-08-03(4차)" 소절, `DESIGN.md` §4.3.15.

| ID | 고정 workload |
|---|---|
| W1 | input 2K/output 128, Poisson 0.60 lambda* |
| W2 | input 8K/output 64, 10초 4× burst+30초 drain, 3 cycles |
| W3 | input 256/output 512, 0.80 lambda* |
| W4 | W2형/W3형 30초 phase를 3 cycles |
| W5 | 1K/8K context 교대, 지원 시 16K |
| W6 | input 2K, output 32/512 교대 |
| W7 | 0.20 lambda* |
| W8 | 0.90/0.95 lambda* |
| W9 | 1.10/1.25/1.50 lambda* |

ShareGPT와 현재 cache된 LongBench를 우선 사용한다. coding/agentic trace는
출처·license·전처리 규칙이 확정된 뒤 추가한다.

Applicability map은 average combined demand, peak-minus-average demand,
prefill/decode pressure temporal correlation으로 만들고 B6−B1 goodput effect를
색으로 표시한다. `peak sum>1`, `average≤1`, pressure 교대 영역을 사전 정의한
target으로 사용한다.

## Controller ablation

| Ablation | 검증 claim/mechanism |
|---|---|
| dual worker 제거 | D |
| dynamic 제거/fixed | architecture 대 policy |
| profile 및 feature ladder | A, E |
| runtime context 제거 | A |
| ITL slack 제거 | E의 SLO protection |
| feasibility gate 제거 | unsafe downshift |
| hysteresis/dwell 제거 | oscillation |
| safety margin 제거 | under-reservation |
| static floor | runtime load term |
| layer-level switching | B |
| CUDA Graph off | B와 operating-point sensitivity |
| chunked prefill on/off | orthogonal composability |

## Controller defaults

- steady: D16/D24/D34/D44; emergency D108
- evaluate every `max(4 decode iterations, 100 ms)` or bucket change
- immediate safe-boundary upshift on ITL violation or 85% KV/batch occupancy
- downshift only below 0.75×SLO for 3 epochs
- dwell `max(8 decode steps, 200 ms)`; upshift exempt
- D108 risk 또는 occupancy 90%가 2 epochs 지속되면 admission 제한
