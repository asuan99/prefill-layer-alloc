# 추가 이점 경로 — 진행 상황 보고서

작성일: 2026-06-17 · 대상: prefill-layer-alloc (v2 핵심 가설 기각 이후)
용도: "이대로 종결"([중단 보고서](project_closure_report.md)) 대신 **추가 이점이 실재하는 경로**가 있다면 그 진행도/다음수/예상이득/킬-기준을 정리. 각 경로는 독립 출구다.
관련: [v2_report](../workspace/characterization/reports/v2_report.md) · [검수 체크리스트](review_checklist.md)

상태 범례: **○ 미착수** · **◐ 부분진행** · **● 코드완료/측정만 필요** · **✦ 결정적**

---

## 요약 (우선순위순)

| # | 경로 | 상태 | 예상이득 | 비용 | 한 줄 |
|---|------|:--:|---|---|---|
| **5** | **memory-state 추상화 (별개 프로젝트)** | ○ ✦ | **유일하게 abstraction-worthy한 신규 연구축** | 중~고(새 프로젝트) | execution이 닫힌 곳의 다음 층 |
| 1 | **7B 회귀** | ◐ ✦ | 결론을 크기-무관으로 격상 **또는** 라인 부활 | 중(SXM4 E2+E5 ×2모델) | config 있음, 미실행 |
| 2 | **prefill/decode overlap 재프레이밍** | ◐ | 양성 기여 — *단 hybrid decode-지연 각도로 차별화해야*(MuxWise/Bullet) | 저(분석+ssm_full+선행연구 대조) | ~2× 측정됨, 포지셔닝 재조정 필요 |
| 3 | **진짜-prefill(ssm_full) E5** | ● | 헤드라인 신뢰도 확정 | 저(셀 일부 재측정) | 검수 A2와 동일 |
| 4 | **비대칭의 비-할당 용도** | ○ | 부수 기여(작은) | 저~중 | 새 RQ 필요 |

> Path 1–4는 *execution-path 라인* 안/주변의 출구다. **Path 5는 그 라인이 닫힌 자리(§closure §3.2)의 한 층 아래 — memory-state 축의 신규 연구**로, 우선순위상 가장 높지만 별개 프로젝트 규모다.

---

## Path 1 — 7B 회귀 ✦ (결정적)

**왜 이게 추가 이점인가.** v2 결론은 전부 SLM(1.2~3B)이고, "분할 무용"이 *작은 커널이 full 풀에서 잘 겹치기 때문*일 가능성이 명시적으로 열려 있다(v2_report §6 한계, §7.4). 7B는 이 단일 변수를 닫는다.

> **올바른 질문으로 재서술 (중단 보고서 §3.1 술어 비판 반영):** Path 1은 "비대칭이 scale에서 살아나는가"(틀린 술어 a)를 묻는 게 **아니다**. **"큰 커널이 공유 SM 풀에서 파괴적 간섭을 일으켜, 정적 분할(green_ctx)이 동적 co-schedule(two_stream)이 못 회수하는 slack을 회수하는가"(올바른 술어 b)**를 묻는다. 즉 판정 지표는 7B의 sat_sm gap이 아니라 **7B에서 `two_stream/green_ctx`가 1.0을 넘는 셀이 생기는가**이다.

- **음성이면(7B도 green_ctx ratio<1.0):** "큰 커널이어도 분할 무용" → 결론이 **모델 크기 무관**으로 격상. 음성 특성화 논문의 결정타.
- **양성이면(7B에서 green_ctx 우위 셀):** **본 연구 라인 부활** — "SLM에선 무용, scale에서 의미"라는 훨씬 강한 thesis.

→ 어느 쪽이든 가치가 높고, **닫는 데도·계속하는 데도 동일하게 필요한** 유일한 데이터점이다.

**현재 진행도(◐ → 코드 준비 완료, 2026-06-18).** 실행 인프라 완비, 측정만 남음:
- `models.yaml`의 7B/8B 엔트리(`zamba2`/`falcon_h1`/`nemotron_h`)를 **캐시 HF config.json 대비 전 필드 검증**(config_status: verified) + 별칭 `zamba2_7b`/`falcon_h1_7b`/`nemotron_h_8b`(`loaders._MODEL_ALIASES`) 추가.
- `submit_size_sweep.sh`가 `MODELS` env override + array 자동 크기 → `MODELS="zamba2_7b falcon_h1_7b" … submit_size_sweep.sh e5 -- --prefill-mode full --prefill-tokens 4096`. (브랜치 `exp/e5-real-prefill`, 설계 [real_prefill_experiment_design §3](real_prefill_experiment_design.md). 측정은 SXM4 필요.)
- **2.7b에서 이미 분할 예외(+12.5%, real prefill)가 났으므로** 7B는 "그 예외가 크기와 함께 커지는가"를 직접 측정 — 더 결정적.
- E0(해석적)는 7B를 즉시 예측 가능(GPU 불필요): `grid_sat_sm = batch·⌈tokens/chunk⌉·n_heads`. 7B는 n_heads↑(112)라 grid가 SM을 더 일찍·강하게 채움 → "큰 커널이면 비대칭이 더 빨리 죽는다"가 사전 예측.

**다음 수 (구체).**
1. **E0 먼저(GPU 불필요, 즉시):** 7B로 `e0_analytical` 재실행 → grid가 batch=1에서도 108 SM 초과하는지 확인. 초과하면 "분할 여지 없음"이 해석적으로 이미 예측됨 → 측정 우선순위 판단.
2. **E2 7B (SXM4) — 서술적 보조만:** `submit_size_sweep.sh e2`에 7B 2모델 추가. sat_sm gap은 *판정 지표가 아니라* 맥락(§3.1 술어 a). 굳이 빼도 결론 불변이나 E0 예측 검증용으로 1회.
3. **E5 7B (SXM4) — 판정(술어 b):** `submit_size_sweep.sh e5 -- --context-lens 1024 2048 4096 8192 16384`. **단 [검수 A2]대로 `--prefill-layer ssm_full`로** (7B는 GEMM이 더 크므로 microbench 왜곡이 SLM보다 심함). **이 `two_stream/green_ctx` ratio가 Path 1의 유일한 결정 지표.**
4. 7B는 메모리·시간이 크므로 batch 격자를 {1,8,32,128}로 축소해 잡 한도 내에서.

**예상 결과(사전).** E0 예측상 7B는 batch≥1에서 grid가 SM 초과 → 비대칭 headroom이 SLM보다 *작을* 가능성 → 분할 무용이 **강화**될 공산이 큼. 하지만 큰 커널의 launch/occupancy 거동이 다를 수 있어 측정 가치 있음.

**킬-기준.** E5 7B에서 `two_stream/green_ctx`가 **어떤 셀에서 ≥1.05**(green이 5%+ 우위)면 → 라인 부활, Path 1을 본 연구로 승격. 전 셀 <1.0이면 → 음성 결론 크기-무관 격상 후 **종결**.

---

## Path 2 — prefill/decode overlap 재프레이밍 (음성 → 양성 출구) ●

**왜.** 프로젝트의 유일한 양성 측정값을 *기여*로 전환. "layer-type 비대칭으로 분할하는 건 무용(음성)"과 "prefill+decode co-schedule이 분할 없이 ~2× (양성)"를 묶는다.

> **⚠ 메시지 수정 (선행연구 충돌, closure §5.1):** "자원을 나누지 말고 겹쳐라"는 단순 슬로건은 **과일반화**다 — **MuxWise·Bullet (ASPLOS'26)**이 단일 LLM에서 prefill/decode를 *분할*해 SLO 이득을 본다. 우리 ~2×는 *throughput·microbench·SLM* 한정이고, 그들의 분할 이득은 *SLO·production*이라 층위가 다르다. 따라서 Path 2의 *정직한* 기여는 "분할하지 말라"가 아니라 ↓ 의 **hybrid-고유 각도**다.
>
> **hybrid-고유 미답 질문 (진짜 novelty 후보):** ssm-decode는 O(1)·context-평탄이라 decode 지연 특성이 Transformer와 *근본적으로 다르다*. MuxWise/Bullet의 prefill/decode 분할 정책(prefill이 decode를 starve하는 걸 격리)은 (추정컨대) Transformer decode의 O(L) 지연 증가를 전제한다. **hybrid에서는 dec=ssm 요청의 지연이 context에 안 자라므로, 분할/multiplexing 정책의 최적점이 달라진다** — 이게 그들이 비운 틈이고, Path 5(memory)와 serving-scheduling에서 만나는 지점.

**현재 진행도(●, 측정 거의 끝).**
- E5에 측정 완료: pf=ssm 행 1.8–2.04×(@db8,ctx4096), window 2.04(db8)→1.15(db256), dec=ssm은 context 1k→16k 평탄 ~2.0×. `results_v2/e5/serving_coexec_*.csv` 4모델.
- overlap이 **roofline 상보성이 아니라 duration matching**(비슷한 길이 두 커널이 SM 풀에 함께 들어감)이라는 메커니즘 해석도 §4.6-A에 있음.

**다음 수.**
1. window가 닫히는 batch를 모델별로 정량화(§G1 recommended_action의 "collapse batch"). widened decode sweep(→512)로 db256 이후 곡선 확정 — **현재 PENDING 잡(770988)이 바로 이 데이터**.
2. [검수 A2] ssm_full 진짜-prefill에서도 2× overlap이 유지되는지(GEMM 포함 시 overlap이 더/덜 되는지).
3. dec=attn은 long context로 decode가 벽시간 지배 시 overlap↓ — balance 조건을 명문화.
4. **선행연구 대조 (필수, 저비용):** MuxWise·Bullet (ASPLOS'26) 원문을 읽어 (a) 그들의 분할 메커니즘·목적함수(SLO? goodput?)·대상 모델(Transformer 전제인가), (b) hybrid/ssm-decode를 다루는가 확인 → 우리 기여를 "그들이 안 다룬 hybrid decode-지연 특성"으로 정확히 포지셔닝. (closer §5.1의 reconciliation 가설을 사실로 확정/수정.)
5. **(승격 시) hybrid-aware multiplexing 정책 측정:** dec=ssm vs dec=attn 요청 혼합 부하에서 SLO 지표(decode tail latency)로 분할 정책 sweep → "hybrid에선 분할 정책이 달라야 한다"를 정량화.

**예상이득.** "분할 음성"만으로는 약함(§5.1). **MuxWise/Bullet이 안 본 hybrid decode-지연 비대칭으로 차별화**하면 serving 기여로 성립. 운영 권고(MPS/multi-stream)는 즉시 실용.

**킬-기준.** (a) ssm_full에서 overlap 1.2× 미만 붕괴 → 양성 주장 약화, Path 3에 종속. (b) 선행연구 대조 결과 MuxWise/Bullet이 *이미 hybrid/state-type을 다룸* → 차별화 소멸, Path 2는 운영 권고로만 축소.

---

## Path 3 — 진짜-prefill(ssm_full) E5 검증 ●

**왜.** Path 1·2·헤드라인 음성 결론 셋 다 **동일 미검증 가정(microbench=진짜 prefill)** 위에 서 있다. 이 한 측정이 세 경로 전부의 신뢰도를 동시에 올린다. [검수 A2]와 동일 작업이므로 **닫기 경로에서도 어차피 해야 함** → 가장 비용대비 효율 높은 추가 작업.

**현재 진행도(●).** 코드 경로 존재(`--prefill-layer ssm_full`, v2_report §6). 측정만 필요.

**다음 수.** E5 최고 셀(pf=ssm×dec=ssm) + db8/db256 양 끝 + ctx{1k,16k}만 ssm_full로 재측정(전수 불필요). green_ctx vs two_stream 정성 3결론 유지 확인.

**예상이득.** 헤드라인을 microbench에서 **진짜 prefill로 승격** → 발표/종결 모두 방어 가능.

**킬-기준.** ssm_full에서 green_ctx가 two_stream을 이기는 셀 발생 → 헤드라인 재검토(사실상 Path 1로 합류).

---

## Path 4 — 비대칭의 비-할당 용도 ○

**왜.** 비대칭(attn이 더 많은 SM에서 포화)은 *할당 손잡이*는 아니지만 다른 곳에 신호로 쓰인다(v2_report §7.3): (a) kernel fusion/튜닝 — ssm가 작아 일찍 포화 → fusion 후보, (b) 모델↔하드웨어 매칭 — 작은 GPU에 ssm-heavy 배치, (c) layer별 quant/offload 우선순위.

**현재 진행도(○).** 아이디어만. 새 RQ·새 측정 필요.

**다음 수(착수 시).** (a)가 가장 구체적 — ssm in/out_proj + scan fusion의 sat_sm/latency 개선을 1모델에서 측정. 단 **별도 프로젝트 규모**.

**킬-기준.** 본 라인 종결 후 별건으로만 검토. 종결 결정에 영향 주지 않음.

---

## Path 5 — hybrid serving의 memory-state 추상화 ○ ✦ (별개 프로젝트)

**왜 이게 유일하게 abstraction-worthy한가.** execution-path 라인이 닫힌 이유는 [closure §3.2](project_closure_report.md): compute 분할 손잡이(Green Context/MPS)는 SM만 가르고 HBM/L2는 공유라, memory 분리의 proxy가 못 된다. attn↔ssm의 *실행* 비동일성은 **증상**이고, 그 **원인은 state 구조의 이질성**이다 — Attention KV는 **O(L)**(위치-인덱싱·append·content-addressable), SSM recurrent state는 **O(1)**(길이 무관 고정 footprint·folded). pure-Transformer serving이 한 번도 마주한 적 없는 *"한 request·한 forward pass 안에 O(L)와 O(1) state가 공존"*이 hybrid-고유의 abstraction 표면이며, **serving의 정립된 추상화는 전부 memory 쪽**이다(PagedAttention=KV 가상메모리, RadixAttention=prefix 공유, KV evict/quant). 결정적으로, [closure §4.2]가 보였듯 *유일한 양성 결과의 흥미로운 구조(dec=ssm의 long-context 평탄성)조차 이 memory 성질이 설명*한다 → 메모리가 뿌리.

**hybrid-고유 추상화 후보 (generic으로 안 무너지는 것).**
1. **이질적 footprint 기반 admission/batching.** hybrid long-context request는 KV를 *attn 레이어 몇 개만* 들고 다님 → 동일 길이 Transformer 대비 메모리 압력이 근본적으로 다름. 이를 모델링한 batch admission은 hybrid에서만 유효.
2. **evict-vs-recompute 경제학이 layer-type마다 *반대*.** KV: 보유 비싸고(O(L)) 재생성 공짜(append). SSM state: 보유 공짜(O(1))지만 재생성 **full rescan(O(L))**. → KV에 없는 **checkpoint-frequency(state 스냅샷 주기)** 라는 새 손잡이.
3. **prefix/radix 공유 의미론이 SSM에서 붕괴.** KV는 위치-additive라 radix-tree 분기·부분 재사용 가능. SSM state는 *lossy fold*라 prefix-끝 상태만 통째 공유 가능, 중간 evict·부분 분기 불가 → 공유 추상화 재설계.
4. **offload/placement 계층.** small·hot·recurrent(SSM) vs large·append(KV)의 배치 프로파일 차이.

**현재 진행도(○).** 아이디어 단계. 본 v2(execution-path) 데이터/코드는 직접 재사용 불가 — *다른 프로젝트*다. 단 측정 프레임워크([closure §4.4])는 재사용.

**novelty calibration (과대주장 방지).** "이질적 KV footprint" 자체는 **부분 선례 있음** — sliding-window/local-global attention(Gemma2, Mistral 류)이 이미 "O(W) vs O(L) KV 혼재"를 serving에서 다룸. 단순 "두 종류 메모리 관리"는 generic으로 미끄러진다. **진짜 새로움은 SSM state의 *folded(lossy)* 성질** — windowed KV는 여전히 per-token·append·evictable이지만 SSM state는 per-token 복원·중간 evict·부분 prefix 공유가 *원리적으로 불가*. 따라서 (2)의 checkpoint/recompute 비대칭과 (3)의 공유 붕괴가 windowed attention엔 없는 hybrid-고유 표면.

**다음 수(착수 시).** (a) 한 hybrid 모델에서 per-request 메모리 footprint를 layer-type별로 분해 측정(KV growth vs state 상수) → admission 모델 스케치. (b) evict-vs-recompute 비용을 layer-type별로 측정(state 재-scan vs KV 재계산) → checkpoint-frequency 손잡이의 이득 곡선.

**킬-기준.** 핵심 베팅 = **"두 state를 *결합적·비대칭적으로* 추론해야만 이득이 나는가, 아니면 SSM state를 *상수 크기 여분 메모리*로 환원해도 generic 추상화가 다 잡는가."** 후자면(환원 가능) → 별건도 연구 의미 약함, 종결. 전자면(결합 필요) → 신규 프로젝트로 승격.

---

## 종합 권고

1. **즉시(GPU 불필요):** ① **MuxWise·Bullet (ASPLOS'26) 원문 대조** — 모든 포지셔닝의 전제(closure §5.1·검수 A5·Path 2의 reconciliation 가설 확정). ② Path 1의 E0 7B 예측 ③ Path 2의 widened window 분석(PENDING 잡 결과) ④ [검수 A5] green_ctx의 `decode_inflation_pct` 재평가.
2. **1회 SXM4 측정:** [검수 A2] = Path 3(ssm_full) — 닫든 계속하든 필수. 여력 되면 Path 1의 E2/E5 7B를 같은 잡 배치에 동봉.
3. **분기:**
   - 7B·ssm_full 모두 음성 → Path 2로 **재프레이밍 후 (execution-path) 종결**(양성 기여 + 크기-무관 음성).
   - 7B에서 green_ctx 우위 셀 발견 → **Path 1을 본 연구로 승격**(라인 부활).
4. **Path 5(memory-state 추상화)는 위 분기와 독립** — execution-path가 어떻게 닫히든 닫히지 않는 신규 연구축. 종결 후 별개 프로젝트로 착수 검토(킬-기준: "두 state를 결합적으로 추론해야 하는가"). Path 4는 종결 후 부수 별건.

> 한 문장: **"추가 이점은 분할(execution)을 더 파는 데 있지 않고 — (1) 7B로 음성을 크기-무관으로 못박거나 (2) co-schedule 양성을 기여로 전환하거나 (3) 한 층 아래 memory-state 추상화로 내려가는 데 있다. 앞 둘은 이 라인의 마무리, 셋째는 다음 프로젝트다."**
