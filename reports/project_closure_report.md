# 프로젝트 종결 보고서 — layer-type 기반 공간 SM 분할 (중단 권고)

작성일: 2026-06-17 · 대상: prefill-layer-alloc (v1 7B saturation → v2 SLM hybrid serving)
상태: E0–E5 완료, G0/G1 게이트 판정 완료, widened sweep 확인 중. **핵심 가설 기각 → 본 라인 종결 권고.**
관련 문서: [v2_report](../workspace/characterization/reports/v2_report.md) · [검수 체크리스트](review_checklist.md) · [추가가치 경로](additional_value_paths.md)

---

## 1. 한 줄 결론

> **"hybrid SSM+Attention 모델에서 layer-type(attn vs ssm) 자원 비대칭을 공간적 SM 분할(Green Context)로 활용한다"는 v2의 핵심 가설은 SLM/A100에서 포괄적으로 기각되었다. 분할은 prefill×decode 전 구간(widened b512 포함)에서 단순 동시실행을 *의미있게* 이기지 못한다(노이즈 수준 동률 외 승리 0; [widened 검증](widened_sweep_validation.md)). 본 연구 라인은 추가 자원 투입의 한계 이득이 음(陰)이므로 중단을 권고한다.**

> **(2026-06-17 추가 검토 1)** G1이 sat_sm 비대칭을 gate 술어로 쓴 것 자체가 잘못이라는 비판(§3.1)을 반영해도 — **종결 권고는 유지, 오히려 강화**된다. 올바른 술어(공유 하 회수 가능 slack)는 E4/E5가 이미 음성으로 측정했기 때문이다.

> **(2026-06-17 추가 검토 2 — scope 정정)** "분할이 동시실행을 못 이긴다"는 **우리 regime(SLM·microbench·*throughput* 지표) 한정**이다. 동일 메커니즘(prefill/decode SM 분할)이 **단일 LLM에서 이득**을 본다는 선행연구가 있다 — **MuxWise·Bullet (ASPLOS'26)**. 그들은 *SLO/latency 목적함수 + production 규모*에서 동작하며, **우리 음성을 반증하지 않지만 우리 결론의 일반화를 막는다**(§5.1). 따라서 *살아남는 강한 주장은 "분할 무용"이 아니라 메커니즘 논증(§3.1 비대칭은 lever 아님 / §3.2 compute 분할은 memory에 못 닿음)*이다.

> **(2026-06-18 추가 검토 3 — real prefill로 실측 정정)** A2를 닫는 real-prefill 측정(job 775529, GEMM-inclusive·16chunk)에서 **두 결론이 정정됐다**([real-prefill 결과](real_prefill_results.md)): (1) overlap ~2×는 microbench 산물 — real prefill에선 저배치 1.05×로 붕괴, 봉우리가 고배치로 이동(window가 *닫히는* 게 아니라 *열린다*); (2) **zamba2_2.7b에서 green_ctx가 +12.5% throughput 승**(검토 2의 MuxWise/Bullet regime 실측 재현) — 단 decode 80% 희생이라 SLO론 패. → 음성 헤드라인은 **"SLO 기준 분할 불리"**로 재정의하면 real prefill에서도 유지, throughput-only로는 대형모델 예외.

> **(2026-06-18 추가 검토 4 — 7B 회귀 완료, 음성)** Path 1(=v2 §7.4의 "결정적" 7B 회귀)을 real-prefill로 실측(job 776326): **zamba2_7b·falcon_h1_7b 모두 green_ctx가 two_stream을 못 이김**(max ts/gc 0.989·0.993). **2.7b의 +12.5% 예외는 7B에서 *사라진다* — 분할 이득은 모델 크기와 함께 커지지 않는다(비단조).** → "큰 커널이면 분할이 의미를 갖나"라는 가장 큰 미검증 구멍이 **NO로 닫힘**. 음성 헤드라인이 결정적 스케일점에서 *강화*된다. 남은 분할-이득 가능성은 크기 축이 아니라 *SLO 목적함수+decode-보호 분할*(검수 A5)뿐. [real_prefill_results §7](real_prefill_results.md).

> **(2026-06-19 추가 검토 5 — SLO·큐 축까지 음성, 마지막 구멍 닫힘)** 검토 4가 남긴 *SLO+decode-보호 분할* 가능성을 둘로 검증: **(a) microbench SLO**(job 776678): decode-보호 분할은 starvation 감소(66–100%→16–18%)하나 two_stream을 양 축 모두 못 이김(wins BOTH=0); 7B는 protect 자체 no_room. **(b) queue 시뮬레이터**(지속 부하): **초기 음성은 모델 결함이었고, 수정 후 *역전*됐다**(사용자 2개 지적으로 발견). 결함: ① sim이 decode ITL=concurrent로 둬 partition의 decode/prefill *decoupling*을 죽임, ② raw throughput/latency로 봄(올바른 metric=goodput@SLO). **수정(decoupled 스케줄러 + goodput) 결과: prefill↔decode 공간 분할(decoupled=MuxWise/Bullet 설계)이 큰 budget·고부하·decode-SLO-binding 코너에서 co_schedule을 goodput에서 이긴다**(예: 2.7b b8 λ1.0 goodput 33.6k vs 8.9k ~3.8×; co_schedule은 raw throughput 크나 ITL 1.9>SLO라 헛것). **단 조건부**(reserved decode가 SLO 만족하는 모델만; falcon_h1_3b는 패) + **TTFT 큰 희생**. [queue_simulator_design §12](queue_simulator_design.md).

> **⚠ (2026-06-19 추가 검토 5 자체-정정 ①)** "분할이 모든 축에서 co-schedule을 못 이긴다"는 **철회.** 그건 (a)*attn↔ssm layer-type* 분할(헤드라인 — §3.1/§3.2/7B로 여전히 음성, 불변)과 (b)*prefill↔decode* 분할(별개 thesis)을 혼동한 것. 본 프로젝트가 닫는 건 (a)뿐.

> **⚠ (2026-06-19 추가 검토 5 자체-정정 ②, baseline)** 검토5-(b)의 "분할이 goodput으로 이긴다"는 **부적절한 baseline(two_stream) 대비였다**(사용자 지적). vLLM/SGLang 기본은 two_stream이 아니라 **fused mixed-batch**(decode가 prefill GEMM 편승)다. `fused`를 넣으니 (2.7b b8 λ1.0) **fused goodput 83k ≫ dynamic_protect 34k ≫ two_stream 9k** — **올바른 baseline 대비 prefill/decode 분할은 budget=8에서 이득 없음.** 단정 가능: *two_stream은 부적절 baseline · fused 대비 분할은 budget=8 음성 · attn↔ssm layer-type 분할은 §3.1/§3.2/7B로 여전히 음성(불변)*. (fused ITL도 budget↑서 SLO 위반 시작 → 더 큰 budget crossover 가능성은 미측정·fused 낙관 보정 필요. **결론은 baseline·budget·모델 가정에 극도로 민감**.) [queue_simulator_design §12–§13](queue_simulator_design.md).

> **layer-type vs prefill/decode 분할 구분(검토 5 핵심):** *원래 thesis*(layer-type-aware 분할)는 hybrid에서 operationally ill-posed(요청마다 attn·ssm 다 거침 → per-layer 재분할 = ~60% swap 오버헤드, 이득 §3.1상 ≈0) → **layer-aware는 agnostic 분할에 구조적으로 진다**(§13). 즉 *layer-type이 PD-mux를 개선하는가 = No*. prefill/decode 분할(MuxWise/Bullet) 자체는 fused 대비 budget=8 음성이나 더 큰 budget·multi-model에선 미결 — 단 그건 layer-type thesis가 아니다.

---

## 2. 무엇을 물었고, 무엇이 나왔나

| 단계 | 질문 | 결과 | 의미 |
|---|---|---|---|
| E1/G0 | scan은 유의미한 component인가 | OK (peak 62–74%) — **단 고batch에선 GEMM 지배, scan 20–40%로 붕괴** | 분할 동기는 저batch에 한정 |
| E2/G1 | attn-ssm sat_sm 비대칭이 있나 | ASYMMETRY_PRESENT (gap 13–54 SM, CI-분리) — **단 granularity 의존(§5)** | 비대칭은 "서술적 사실"이지 손잡이 아님 |
| E4 | 비대칭을 분할로 회수하나 | prefill-prefill split +1–7%뿐; aware split은 decode를 굶겨 **+60–144% 악화** | 분할은 손해 |
| E5 | serving 전 구간에서 (prefill/decode) 분할이 동시실행을 이기나 | microbench: **의미있는 승리 0**. **단 real prefill(job 775529)에선 zamba2_2.7b·attn-prefill·고배치서 green_ctx +12.5%**(pf=attn×dec=ssm×db256) — 그러나 decode 80% 희생(SLO로는 패). [real-prefill 결과](real_prefill_results.md) | 가설 기각을 **"SLO/decode-latency 기준"으로 재정의** — throughput-only로는 대형모델 예외(§5.1 regime 실측 재현) |
| E5 | 그럼 실이득은 어디서 오나 | prefill+decode **overlap**: microbench ~2.04×(저배치). **단 real prefill에선 저배치 1.05×로 붕괴, 봉우리가 고배치로 이동(db512 1.6–1.76×)** — duration matching | 양성 산출(§4) — *regime이 prefill 크기에 종속* |

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

> **용어 — 보고서 전체가 이 구분 위에 선다.** 현대 GPU는 여러 커널을 동시에 돌린다(Fermi+); 동시성의 단위는 커널이 아니라 **thread block**이고, 하드웨어 블록 스케줄러가 블록을 비어있는 SM(A100=108개)에 뿌린다.
> - **`two_stream`(동적 공유):** 두 커널을 서로 다른 CUDA stream에 올리면, 스케줄러가 *빈 SM 아무 데나* 양쪽 블록을 채운다 → **work-conserving**(한쪽이 SM을 놀리면 다른 쪽이 즉시 침범). 벽 없음.
> - **`green_ctx`(정적 분할):** 각 커널을 *고정 SM 부분집합*에 가둠(예: 54/54). A가 놀려도 B가 못 씀 → **non-work-conserving**, 경직.
> - decode가 GPU를 못 채워(BW-bound·소수 블록) 남긴 빈 SM을 prefill이 채우는 게 overlap의 방. decode batch↑로 그 방이 차면 닫힌다.
>
> 그래서 정적 분할은 스케줄러의 자유를 *빼앗기만* 한다 — 이득은 동적 공유가 **파괴적 간섭**(L2 thrash, decode starvation)을 일으켜 격리가 회수할 때만 양수(§3.1 술어 b, §5.1).

---

## 4. 회수 가능한 산출물 (중단 ≠ 손실)

음성 결과지만 다음은 그대로 가치가 있고, 종결과 무관하게 보존/활용 가능하다:

1. **메커니즘 논증 (음성 *결과*보다 이게 본체).** "분할 무용"이라는 *결과* 자체는 §5.1(MuxWise·Bullet) 때문에 standalone 기여로는 약하다 — prefill/decode 분할이 이득이라는 선행연구가 이미 있어, 우리 음성은 "SLM·microbench·throughput에선 안 됨"이라는 *경계조건*에 가깝다. **발표 가능한 본체는 결과가 아니라 메커니즘 논증이다:** (a) §3.1 — sat_sm 비대칭은 partition lever가 아니다(category error), (b) §3.2 — compute 분할(Green Context)은 memory 분리에 *원리적으로* 못 닿는다. 이 둘은 regime 무관하게 성립하며, "layer-type 비대칭으로 hybrid serving을 최적화하라"는 방향을 *왜 접어야 하는지*를 설명한다. (characterization 트랙, SSM-Scope류와 정렬.)
2. **양성 권고 + 그 구조:** prefill+decode를 공유 SM에 **그냥 co-schedule**(MPS/multi-stream) → 저~중 decode batch에서 1.2–2.0×, 분할/aware 할당 불필요. 즉시 적용 가능한 운영 지침. 겹침의 정도는 두 축이 *따로* 지배:
   - **prefill 타입 = 천장(ceiling).** prefill이 긴 stream이라 "decode를 숨길 방"을 정함 — `pf=ssm` 1.8–2.04× vs `pf=attn` 1.06–1.43× (@db8,ctx4096). 단 메커니즘은 roofline 상보성이 아니라 **duration matching**(비슷한 길이 두 커널이 SM 풀에 함께 들어가 포개짐, occupancy 현상).
   - **decode 타입 = context 스케일링.** short-context에선 타입 무관히 다 숨지만, `dec=ssm`은 O(1) state라 1k→16k **평탄 ~2.0×**, `dec=attn`은 O(L) KV라 long-context에서 decode가 벽시간을 지배하며 **overlap 붕괴**.
   - window는 decode batch로 닫힘(2.04@db8 → 1.15@db256).
   - **함의(중요):** 이 양성 결과의 *흥미로운* 부분 — long-context serving에서 `dec=ssm`이 특별히 강하다는 점 — 조차 execution이 아니라 **memory-state 성질(O(1) state vs O(L) KV)**이 설명한다. 즉 hybrid가 serving에서 *고유하게* 좋은 지점도 뿌리는 메모리다(→ §3.2, [Path 5](additional_value_paths.md)).
   - **⚠ "분할 불필요"의 scope 한정 (선행연구 충돌 처리, §5.1):** 위 "co-schedule이 분할을 이긴다"는 **microbench·SLM·*throughput* 기준 한정**이다. **prefill/decode 공간 multiplexing 자체는 단일 LLM에서 이득이 입증돼 있다 — MuxWise, Bullet (둘 다 ASPLOS'26).** 우리 음성은 그들을 반증하지 않는다(§5.1: 목적함수·regime이 다름). 따라서 v2_report §7.2의 "분할도 aware 할당도 불필요"는 **"layer-type 분할은 불필요"로 좁혀 적어야** 하고, prefill/decode 분할은 *generic serving에서 이미 유효한 별개 사안*으로 분리한다.
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
- **metric 한정 (중요):** E5의 green_ctx-vs-two_stream 판정 지표는 **throughput overlap**(`two_stream/green_ctx`)뿐이다. **decode tail latency / SLO-goodput은 평가하지 않았다** — 그런데 *그게 바로 공간 분할이 이기는 metric*(MuxWise·Bullet의 목적함수, §5.1)이다. E5엔 `decode_inflation_pct` 컬럼이 있으므로 two_stream vs green_ctx의 decode 지연을 사후 비교 가능([검수 A5](review_checklist.md)).

### 5.1 선행연구 대비 위치 — MuxWise·Bullet (ASPLOS'26)

**사실(확인됨):** MuxWise, Bullet (둘 다 ASPLOS'26)은 **단일 LLM**에서 prefill과 decode를 **multiplexing으로 분리**해 이득을 본다. 즉 "prefill/decode 공간 multiplexing은 무용"이 *아니다*.

**우리 결과와 충돌하는가? — 직접 맞닿지만 반증은 아니다. scope(목적함수·regime)가 다르다:**
1. **같은 메커니즘이다 (정직히 인정).** E5의 `green_ctx@f`는 동시 실행되는 **prefill 스트림과 decode 스트림 사이에 SM을 분할**한다(CSV `prefill_sm_frac`/`prefill_sm`/`decode_sm`) — 즉 *그 자체가 prefill/decode SM 분할*이고, MuxWise·Bullet이 쓰는 것과 **같은 종류의 손잡이**다. 그래서 "다른 축이라 안 부딪힌다"는 변명은 성립하지 않는다. layer-type(attn/ssm)은 *분할 비율 f를 정하는 신호*로 들어갔을 뿐이고(원래 thesis), E5는 "layer-type으로 고른 f든 어떤 f든 동적 공유를 못 이긴다"를 보였다. **따라서 MuxWise·Bullet이 충돌하는 것은 E5 결과의 *일반화 버전*("prefill/decode 분할은 무용")이며, 이건 아래 2·3 때문에 *우리 regime 한정*으로 좁혀야 한다.** (단 *layer-type 비대칭이 유용한 분할 신호*라는 sub-claim은 MuxWise/Bullet과 무관하게 §3.1로 이미 죽었다 — 그 부분 헤드라인은 무사.)
2. **다른 목적함수.** E5는 *throughput overlap*만 본다. 동적 공유(two_stream)는 throughput엔 work-conserving이나 **decode 지연을 보호하지 않는다**(prefill grid가 SM을 덮쳐 decode stall). 정적 분할은 decode에 보장 슬라이스를 줘 **지연/ SLO를 회수**한다 — throughput을 조금 내주고 latency를 산다. prefill/decode 분할의 이득은 이 **SLO/goodput 축**에 있고, 우리는 그 축을 안 쟀다.
3. **다른 regime.** prefill/decode 분할이 의미를 갖는 곳은 **대형 모델·real prefill(GEMM-heavy, multi-chunk)·지속 부하**다 — 거기서 prefill이 decode를 *파괴적으로* starve하므로 격리가 회수한다(§3.1 술어 b의 *양성* 사례). E5는 **SLM·microbench(prefill chunk=1, scan-only)**라 그 간섭이 없어 못 봤다. (E4의 76/32 split은 오히려 *prefill을 편들어* decode를 굶긴 잘못된 방향 — Bullet류는 decode를 보호한다.)

> ⚠ **확인 필요:** 위 2·3(목적함수·regime·격리 메커니즘)은 *왜 그들이 우리와 모순 없이 이득을 보는가*에 대한 **우리 측 reconciliation 가설**이다. MuxWise·Bullet은 ASPLOS'26(본 작성자 지식 컷오프 이후)이라 **두 논문의 실제 메커니즘은 원문 대조로 확정해야 한다.** 본 절은 "그들이 SM 분할/MPS를 쓴다"를 사실로 단정하지 않는다 — 확정된 것은 "단일 LLM prefill/decode multiplexing으로 이득"뿐.

**함의(종결과 무관, 오히려 보강):** 선행연구가 분할 이득을 보인 regime(대형·real-prefill·SLO)이 *정확히 우리가 안 테스트한 영역*(7B/ssm_full, Path 1·3)임을 확인해준다 → 우리 음성의 **microbench/SLM/throughput scope를 더 분명히** 한다. 그리고 **hybrid-고유 미답 질문**을 연다: ssm-decode가 O(1)·context-평탄이라 decode 지연 특성이 Transformer와 달라, **hybrid에서 prefill/decode multiplexing 정책의 최적점이 다를 수 있다** — MuxWise/Bullet이 비운 틈([Path 2 확장](additional_value_paths.md)).

---

## 6. 권고

1. **본 라인(layer-type *비대칭을 분할 신호로* 쓰는 thesis) 종결.** 추가 layer-type-aware 분할 실험에 SXM4 자원 투입 금지 — §3.1·§3.2로 메커니즘 수준에서 닫힘(regime 무관). **단 "prefill/decode 분할 일반"을 종결하는 것이 아니다** — 그건 MuxWise·Bullet(ASPLOS'26)이 SLO·production서 이득을 보인 *살아있는* 영역이며(§5.1), 우리가 종결하는 건 *그것을 layer-type 비대칭으로 모는* 특정 가설이다.
2. **종결 전 [검수 체크리스트](review_checklist.md)의 P0 6건 통과 확인** — 특히 A2(ssm_full 진짜-prefill 1회), A5(green_ctx의 decode 지연/SLO 재평가), C1(7B 미실행 스코프 명기). 이것들이 통과해야 "SLM·throughput 한정 음성 확정"으로 깨끗이 닫힌다.
3. **단, 닫기 전 [추가가치 경로](additional_value_paths.md)를 1회 검토.** 7B 회귀(Path 1)는 *닫는 데도 필요한* 결정적 데이터점이며 — **올바른 술어(b)로 재서술하면 "큰 커널에서 공유 하 회수 가능 slack이 생기는가"를 묻는다**(음성이면 결론을 크기-무관으로 격상, 양성이면 라인 부활). 비용이 크지 않다. prefill/decode overlap 재프레이밍(Path 2)은 음성 라인을 양성 기여로 전환하는 별개 출구다.
4. **v1/v2 동결:** `archive/v1_7b_saturation/` 유지. v2 산출물(results_v2, figures, verdicts) 커밋 후 동결.
5. **종결의 범위 한정 (중요).** 본 종결은 *execution-path 라인*(SM 분할/스케줄링)에 대한 것이다. §3.2에 따라 **hybrid serving의 memory-state 추상화**(이질적 footprint admission, layer-type별 evict-vs-recompute 비대칭, SSM state의 prefix/radix 공유 붕괴, offload 배치)는 **미탐색의 별개 축이며 유일하게 abstraction-worthy한 잔여 방향**이다 → [추가가치 Path 5](additional_value_paths.md). 이건 죽은 라인의 부활이 아니라 *다른 프로젝트*다. 단 novelty는 SSM state의 *lossy-fold* 성질(sliding-window KV 선례와 구분되는 지점)에 걸리며, "두 state를 결합적·비대칭적으로 추론해야 하는가"가 베팅의 핵심.

> **결정 트리:** 검수 P0 전부 통과 + 7B 미실행 수용 → **execution-path 라인 종결(음성 특성화로 발표).** 7B 1회라도 돌릴 여력 → `additional_value_paths.md` Path 1 먼저 → 그 결과로 최종 종결/전환. **단 어느 쪽이든 memory-state 추상화(Path 5)는 닫히지 않은 별개 출구로 남는다.**
