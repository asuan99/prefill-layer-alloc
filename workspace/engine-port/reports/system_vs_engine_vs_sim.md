# Full-system vs. engine-policy vs. queue-sim — 어디서 결과가 갈리나 (fidelity ladder)

작성: 2026-07-12. 목적: 본 프로젝트의 모든 정책 결론(특히 **layer-aware 반증**, **SLO-aware 우위**)은
**"sglang 위에 얹은 SM-split 정책 레이어"** 라는 특정 위치에서 나왔다. 같은 정책이라도
(a) **queue-simulation**, (b) **본 engine 실측**, (c) **Bullet/MuxWise 같은 full serving system**
세 레이어에서 서로 다른 결과를 낼 수 있다. **어느 결론이 정책의 진실이고, 어디부터가 "정책이
잘못된 레이어에 얹혀서 생긴 substrate 아티팩트"인가**를 영역별로 분리한다.

관련: [policy_comparison.md](policy_comparison.md)(정책 goodput·메커니즘), [prefill_vs_decode_execution.md](prefill_vs_decode_execution.md)(실행 knee),
[p1_4_layer_aware_평가_kr.md](p1_4_layer_aware_평가_kr.md)(transfer 조건), `results/a_substrate/`(substrate isolation).

---

## 0. TL;DR — bracket 논제

> **queue-sim은 진실을 위에서(optimistic), 본 engine은 아래에서(pessimistic) 끼운다. Bullet 같은
> full-system은 그 사이에 있다.** 나는 시스템 전체가 아니라 split *결정 로직* 하나만 설계했고,
> 그 로직을 **sglang이 이미 만들어 둔 무거운 기판(green-ctx drain·no-cudagraph·single-process)**
> 위에 얹었다. 그래서:
>
> - **sim이 "layer-aware 이득"을 과대평가**한 건 switch 비용·불완전 isolation·decode wall을 전부 0으로 뒀기 때문.
> - **본 engine이 "layer-aware 반증"을 harsh하게 낸** 건 (D) drain과 no-cudagraph가 기판 탓으로 과대계상됐기 때문(sign은 견고, magnitude는 confounded).
> - **진짜 답(full-system)은 둘 사이**에 있으며, 각 결론을 **substrate-artifact vs fundamental**로 태깅해야 이식 가능성을 안다.

---

## 1. 세 레이어의 정의 (무엇을 이상화하나)

| | queue-simulation | **본 engine (내 작업)** | full serving system (Bullet/MuxWise) |
|---|---|---|---|
| SM 파티션 | service-rate 벡터 (추상) | green-ctx 공간분할 (실물) | libsmctrl TPC 마스크 / green-ctx (실물, 정책과 공동설계) |
| 파티션 전환 비용 | **0 (즉시)** | **비쌈** (stream drain/synchronize) | **쌈** (Bullet: bitmask write) |
| decode 실행 | 고정 service time | **eager (no-cudagraph)** | **cudagraph** (별도 프로세스) |
| 프로세스 구조 | 독립 서버 2개 (간섭 0) | **단일 프로세스·단일 event loop (GIL)** | **별도 prefill/decode 프로세스** |
| 스케줄러/배칭 | 이상화(M/M/1 류) | sglang 상속(chunked-prefill·radix·continuous) | 정책과 공동설계 |
| 메모리/KV | 무모델 | 공유 pool(핸드오프 無) | 별도 프로세스 → **KV IPC 핸드오프 비용** |
| 신호(제어 입력) | 완전·즉시 관측 | **측정 TPOT-EMA(잡음·drain 오염·지연)** | 전용 dispatcher의 clean `longest_queue_ms` |
| SM isolation | **완전 선형 speedup 가정** | L2/HBM-BW 공유(불완전) | 동일 물리한계(단 cudagraph가 BW 효율↑) |
| 내가 설계한 것 | (전부 모델러) | **split-index 결정 로직만** | **전 레이어 공동설계** |

핵심: 사다리를 올라갈수록(sim→engine→system) **이상화가 벗겨지지만**, 본 engine은 "정책만 바꾸고
나머지는 sglang 고정"이라 **기판이 정책에 맞춰 최적화돼 있지 않다**. Bullet은 기판을 정책에 맞춰 짰다.
이 "정책-기판 mismatch"가 아래 모든 차이의 근원이다.

---

## 2. 차이 발생 영역 (overhead / idealization 축)

사용자 예시("정책이 다른 레이어에 있어 추가 overhead")를 영역별로 전개. **편향 방향** = 그 영역이
내 engine 결과를 full-system 대비 어느 쪽으로 왜곡하는가.

### (A) 파티션 전환 비용 — **가장 큰 영역**
- **sim**: 0. layer-aware가 매 step ~19번 전환해도 공짜 → sim이 layer-aware를 이기게 만든 1차 원인.
- **engine(mine)**: green-ctx는 SM을 깨끗이 넘기려 **나가는 스트림을 quiesce(drain/synchronize)** 해야 함.
  adjust-block에서 이 sync가 **~수백 ms 스파이크**(초기 SLO 컨트롤러 신호를 오염시켜 v1→v4 버그 유발).
  → sub-step 강제전환(layer-aware)이 **(D) granularity**로 죽는 직접 원인. R0d coord-la TPOT
  **42ms(agnostic 평탄) → 117–124ms(부하시 폭발)**.
- **full-system(Bullet)**: `set_stream_mask`은 **bitmask write 1회**, drain 없음. **그래서 Bullet은 layer-span마다
  자주 전환하는 걸 정책으로 채택**할 수 있다.
- **판정**: 내 "(D)로 layer-*span* 잦은 전환이 손해" 결론은 **green-ctx 기판 아티팩트**. 싼 마스크였으면
  전환 자유. (단 layer-*type*-aware는 별개 축 — §4.)

### (B) decode 실행: cudagraph 유무 — **layer-aware 이식성의 열쇠**
- **sim**: decode = 고정 latency, launch overhead 무모델.
- **engine(mine)**: pdmux+mamba+triton 환경서 **cudagraph 불가** → decode가 eager라 per-op launch overhead +
  SM 감소에 민감. Zamba2 decode **TPOT 170–205ms ≫ SLO(60ms)** → **decode가 벽**.
- **full-system(Bullet)**: 별도 프로세스 + **cudagraph decode** → TPOT floor가 훨씬 낮고 SM 감소에 강건.
  decode에 SM을 덜 줘도 안 무너짐.
- **판정**: ★**[p1_4 실측 증거]** Zamba2서 **layer-aware의 prefill TTFT 이득(la<agn, −7~23%)은 확증**됐으나
  **goodput으로 미전환** — 이유가 정확히 **no-cudagraph decode wall**(TPOT≫SLO라 TTFT가 뭘 해도 무의미).
  즉 **engine 기판이 정책의 이득을 가렸다**. Full-system(cudagraph)이면 벽이 사라져 **이 이득이 전환될
  여지가 있다** — 미검증·open. → 내 절대 TPOT는 부풀려짐("green-ctx no-cudagraph 절대값 하한" caveat의 실체).

### (C) 프로세스 구조 / 제어흐름 간섭
- **sim**: 독립 서버, 간섭 0.
- **engine(mine)**: 단일 event loop → prefill 스케줄링과 decode 디스패치가 **같은 파이썬 스레드·GIL·CUDA
  컨텍스트** 공유, 제어흐름 **직렬화**. (a)substrate 실험서 남은 잔차 = **monolithic prefill 단일-윈도우
  오버랩**(구조적·model-independent)이 여기서 옴.
- **full-system(Bullet)**: 별도 프로세스 → 진짜 병렬 제어, prefill이 decode 디스패치를 못 막음.
- **판정**: 내 "구조적 잔차(단일-윈도우 오버랩)"의 일부는 **single-process 아티팩트**. 별도 프로세스면 더
  fine한 오버랩 가능. (a)substrate가 R0d 124ms의 **절반이 sync+pinning 탓**임을 이미 보임(sign은 유지).

### (D) 스케줄러/배칭 결합
- **sim**: 배칭 이상화(고정 batch), chunked-prefill·radix 無.
- **engine(mine)**: sglang 스케줄러 **상속** → 내 정책은 스케줄러가 넘겨주는 **prefill 경계에서만** 평가/전환
  가능. layer-span 평가의 granularity가 **host 스케줄러의 yield 지점에 상한**됨.
- **full-system**: 스케줄러를 SM 정책과 공동설계 → switch 지점을 최적 배치, admission을 SM 상태와 조율.
- **판정**: 내 정책 granularity는 "최적"이 아니라 "sglang이 허용하는 만큼". 이건 아티팩트라기보단 **설계
  자유도의 제약**.

### (E) 메모리 / KV 조율
- **sim**: 무모델(무한/고정 용량).
- **engine(mine)**: prefill·decode가 **KV pool·토큰 budget 공유**, 별도 핸드오프 비용 無.
- **full-system(Bullet)**: 별도 프로세스라 KV를 **IPC/shared-mem로 핸드오프** → 내 engine엔 없는 **추가 조율
  비용**. 단 이 비용을 내고 (B)cudagraph 이득을 산다.
- **판정**: 여기선 오히려 **내 engine이 유리**(핸드오프 0). Bullet은 이 비용 ↔ cudagraph 이득의 트레이드.
  → full-system이 무조건 우월한 게 아니라 **다른 비용 구조**임을 보여주는 영역.

### (F) 제어 신호의 품질/지연
- **sim**: 완전·즉시 관측.
- **engine(mine)**: TPOT를 **iteration wall-time로 자가측정** → 잡음, **자기가 유발한 drain 스파이크가 신호를
  오염**(제어 대상과 측정이 같은 루프 = v1→v4 버그의 근원), EMA 지연.
- **full-system(Bullet)**: 전용 dispatcher가 cross-process로 `longest_queue_ms`를 **깨끗이** 관측(자기 drain에
  오염 안 됨).
- **판정**: 내 closed-loop 신호는 **구조적으로 더 dirty**. Bullet의 분리된 dispatcher가 더 clean한 신호.
  내 outlier-rejection/EMA/deadband는 이 dirty 신호를 다루려는 방어책.

### (G) SM isolation 이상화
- **sim**: **완전 선형 speedup**(파티션당 service rate). 실물 아님.
- **engine/full-system(둘 다)**: green-ctx/libsmctrl 모두 **L2·HBM-BW 공유** → prefill이 decode의 BW를 훔침.
  물리한계는 sim과 무관하게 실재.
- **판정**: sim의 "파티션당 rate 할당"이 **가장 크게 과이상화**. sim-vs-real 갭이지 정책 레이어와 무관 →
  sim이 layer-aware를 낙관한 2차 원인.

---

## 3. 각 핵심 결론의 재분류 — substrate-artifact vs fundamental

| 결론 | 근거 기전 | 분류 | full-system서 뒤집힐까? |
|---|---|---|---|
| **layer-TYPE-aware(decode-window) 死** — (D) granularity | green-ctx drain per ~19 windows/step | **substrate-tinged** + **fundamental 반반** | switch 비용은 싼 마스크서 사라짐(substrate). 단 "prefill에 free SM 없음"(attn·mamba prefill 둘 다 SM-민감, differential ~1.44×)은 **fundamental** → 싼 기판이어도 이득 얇을 것 |
| **layer-aware prefill TTFT 이득** (la<agn −7~23%) | 둔감 mamba prefill SM 환원 | **fundamental (기전 확증)** | ★**측정됨(2026-07-13)**: prefill TTFT 우위는 유지(lacoord 697<<3025)나 **goodput 전환 실패** — cudagraph가 decode wall을 걷어도 **layer-aware는 sub-step 재분할로 cudagraph 비양립**이라 decode가 붕괴. **전환 open→CLOSED negative** |
| **SLO-aware = 유일 upside** (static 매칭·dual-stress +18%) | latency closed-loop 동적 split | **fundamental (정책 결과)** | 유지 예상. Bullet이 같은 계열(더 좋은 기판)이라 **더 강해질** 것 |
| **SLO 절대 magnitude(TPOT)** | no-cudagraph eager decode | **substrate 아티팩트** | full-system서 **절대값 개선**(하한일 뿐) |
| **type-aware span sizing 死** (TTFT 2–4×↑) | span 3× 짧아 prefill 반복 오버헤드 | **substrate-tinged** | 짧은 span의 재진입 비용은 기판 의존. 단 Zamba2 type-run(~6층)이 짧다는 건 model fact |
| **(a) 단일-윈도우 오버랩 잔차** | monolithic prefill·single-process | **substrate 아티팩트** | 별도 프로세스서 완화 가능 |

**요지**: layer-aware가 원하던 **prefill TTFT 기전 자체는 fundamental이고 확증**됐다. 죽은 건 (1) **sub-step
decode-type 전환**(green-ctx drain 아티팩트 + prefill free-SM 없음 fundamental)과 (2) **goodput 전환**(no-cudagraph
decode wall = engine 기판 아티팩트). 따라서 **"layer-aware 완전 반증"은 이 기판 한정 진술**이고, full-system
(cudagraph + 싼 마스크)서 **prefill-side layer-aware가 goodput 이득으로 전환되는지는 열린 문제**다.

---

## 4. queue-sim은 왜 정확히 틀렸나 (sim의 3대 낙관)

sim이 layer-aware 이득을 예측했으나 engine이 반증한 간극은 **§2의 세 영역이 sim에서 0이었기** 때문:

1. **(A) 전환 비용 = 0**: sim은 매 step 층타입 경계마다 재분할이 공짜 → (D) granularity 자체가 sim엔 부재.
2. **(G) isolation = 완전선형**: 파티션당 독립 service rate 가정 → 실제 L2/HBM-BW 경합(prefill이 decode BW 강탈)을 놓침.
3. **(B) decode wall = 무모델**: sim은 TPOT를 고정 service time으로 → no-cudagraph decode가 SLO 벽이 되는 현실 부재.

→ sim은 **정책의 upside만 모델하고 mechanism의 downside를 전부 이상화**했다. 그래서 **upper bound**.
반대로 내 engine은 그 downside를 **무거운 sglang 기판으로 과대계상** → **lower bound**. **진실은 사이.**

---

## 5. 함의 · 후속 검증

1. **보고서 스코프 명시**: policy_comparison 등은 "engine-hosted policy layer" 스코프임을 §1에 박스로 명시 필요
   (현재 암묵적). 본 문서가 그 스코프 정의.
2. **layer-aware 최종 판정의 정직한 재진술**: "green-ctx·no-cudagraph·single-process 기판 위에서 sub-step
   decode-type 분할과 goodput 전환이 죽었다"가 정확. **prefill-side 기전은 살아있음**. Bullet-급 기판
   (libsmctrl + cudagraph + 별도 프로세스) 재현 없이 "layer-aware 자체가 죽었다"고 단정하면 과대주장.
3. ~~**가장 결정적 후속 실험**: cudagraph 환경서 prefill-side layer-aware가 goodput 전환되나~~ →
   ★**완료·판정 negative (2026-07-13, [results/cudagraph_probe/cudagraph_results.md](../results/cudagraph_probe/cudagraph_results.md))**:
   cudagraph는 py3.14/torch2.9+hybrid+triton+**pdmux green-ctx**서 **정상 작동**했다("불가"는 오해·수동 flag).
   core 정책(agn/tuned/SLO)은 cudagraph로 **decode wall 넘음**(TPOT ~40→13ms, goodput ~1.5–2×↑, 랭킹 불변).
   그러나 **layer-aware는 못 넘는다** — ★**cudagraph ⊥ sub-step layer-aware**: sub-step green-ctx 재분할은 고정
   그래프로 캡처 불가라 coord decode가 영구 eager. ⇒ lacoord prefill TTFT는 우위(697<<3025 agn)나 decode SLO
   붕괴(gp r4 **0.027**). **"cudagraph가 layer-aware 구제" 가설은 반증** — 오히려 격차 확대. **(B)는 한계가
   아니라 기존 측정의 flag 선택**으로 격하(engine을 사다리 위로 이동 성공).
4. **차선 후속**: libsmctrl(driver 문제로 BLOCKED였음)로 (A)전환 비용을 실제로 낮춰 layer-span 잦은 전환의
   (D) 아티팩트 부분을 분리 측정.

---

## 부록 — 한 줄 요약 매핑

| 영역 | sim | engine(mine) | full-system | 내 결과 편향 |
|---|---|---|---|---|
| (A) 전환비용 | 0 | 비쌈(drain) | 쌈(mask) | layer-span 전환을 과소평가 |
| (B) decode | 무모델 | no-cudagraph(벽) | cudagraph | TPOT 과대·goodput 이득 가림 |
| (C) 프로세스 | 독립 | single-proc/GIL | 별도 proc | 오버랩 잔차 과대 |
| (D) 스케줄러 | 이상화 | sglang 상속 | 공동설계 | granularity 상한(제약) |
| (E) KV | 무모델 | 공유(핸드오프0) | IPC 핸드오프 | **내가 유리**(트레이드) |
| (F) 신호 | 완전 | dirty(자가오염) | clean dispatcher | 컨트롤러 복잡도↑ |
| (G) isolation | 완전선형 | BW 공유 | BW 공유 | (sim만 낙관) |
