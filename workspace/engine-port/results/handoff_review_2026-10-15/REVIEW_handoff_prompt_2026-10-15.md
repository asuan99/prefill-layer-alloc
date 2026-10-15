# 인계 프롬프트(`handoff/CLAUDE_CODE_PROMPT.md`, 2026-10-14) 순차 검토 — 2026-10-15

> 지위: main 트랙 검토 메모. 측정 0, 새 성능 판정 0. 정본 반영은 `promote` 몫.
> 사용자 결정(2026-10-15): (i) 프롬프트의 옛 환경 전제(SLURM·KISTI)는 무시하고 실행은 VESSL Job, (ii) 주 모델 = Nemotron-Nano-9B-v2.

## 1. "PD-mux가 fused 대비 ~3×" — 출처 검토 (G0 0단계)

**판정: 정본은 이 문장을 철회했다.** 프롬프트 §1 "살아남은 기여 (2)"는 그대로 쓸 수 없다.

| 계열 | 체크포인트 | 클라이언트 | 조건 | 정본이 허용하는 것 |
|---|---|---|---|---|
| NemotronH | `Nemotron-H-8B-Base-8K` | async `bench_serving` (random) | in2000/out96, **no-cudagraph, n=1** | 운영점 미측정 → 인용 불가 |
| Falcon-H1 | `Falcon-H1-3B-Base` | async | 같음, no-cudagraph, n=1 | 운영점 미측정 → 인용 불가 |
| Granite | `granite-4.0-h-micro-base` | async | **P1-opint: cudagraph-ON, n=5 paired** | r4 **+27.0%**만 (r3 +11.7%는 t-CI 강등, mean-ITL 술어로는 부호 반전) |
| Zamba2 | `Zamba2-2.7B` | 운영점은 async. 옛 fused 수치는 60스레드 GIL 클라이언트(폐기) | P1-opint, cudagraph-ON, n=5 | r2 **+40.5%**, r3 **+185.8%** |

- 근거: `PROJECT_STATUS.md:6664-6750`(2026-08-05 정정 "4모델 전부는 더 이상 쓸 수 없다", "항상 이득 철회"), `CONSENSUS.md` §1-1 행(3680), `CLAIM_EVIDENCE_MATRIX.md:642`.
- "≈3×"의 실제 출처는 두 곳이다. ① 비정본 `reports/sm_policy_report.html:378`의 표 칸. ② P1.5 NemotronH rate10의 1.16 vs 0.36. ②는 GIL 클라이언트로 포화(~4.3 req/s)를 넘겨 측정했고, 같은 메모리(`engine-port-p0-triage.md:37`)에서 아티팩트로 선언됐다. 정본 수치 중 3×에 가까운 것은 Zamba2 r3(2.86×) 한 셀뿐이다.
- fused는 **미조율**이다(`--disable-radix-cache --mem-fraction-static 0.82 --max-running-requests 48`, chunked prefill은 엔진 기본값). PD-mux arm은 `--enable-pdmux`가 `--chunked-prefill-size -1 --disable-overlap-schedule`을 assert로 강제하므로 **3-플래그 묶음 처치**다. "PD 분리 자체가 원인"은 NOT-YET-SUPPORTED이고, fused 조율 공간 소진(T2)은 주장 금지다(미시험 노브 6개 이상).
- goodput 술어도 캠페인마다 다르다. P1.6/1.7은 TTFT≤3s ∧ **mean** TPOT≤60ms이고, P1-opint는 TTFT≤3s ∧ 요청 내부 **ITL p95**≤60ms다.
- 정본에 맞는 문장: "A100·cudagraph-ON에서, 미조율 기본 fused 대비 PD-mux 활성화의 꼬리 SLO goodput 우위는 Zamba2-2.7B +40.5%(r2)·+185.8%(r3), Granite-4.0-h-micro +27.0%(r4)다. 기전 귀속은 미확립."
- **프롬프트 열린 질문 1 답**: 기존 NemotronH = 8K Base(16K 필터와 충돌). 주 모델은 Nano-9B-v2로 결정됐다. **Nano-9B-v2에서는 PD-mux vs fused가 한 번도 측정되지 않았다.** 따라서 G0는 "재현"이 아니라 **이 모델·이 워크로드에서의 첫 운영점 측정**이다.
- **문서 드리프트**: 작업 루트 `CLAUDE.md` "현재 결론" 첫 줄 "PD 분리 자체는 4모델 전부서 fused를 이김"은 PROJECT_STATUS(위계 1위)와 모순된다. 메모리 `engine-port-p0-triage.md:42`도 낡았다. → 다음 `promote` 때 정정(이 세션에서는 쓰지 않음).

## 2. 워크로드 — arXiv-Sum 16k 필터의 의의와 논문 포지션

### 2.1 실측 통계 (Nano-9B-v2 토크나이저, `arxsum/ARXSUM_STATS_2026-10-15.txt`)

`ccdv/arxiv-summarization` document/test(n=6,440). 입력 = article, 출력 = abstract.

| 토크나이저 | ≤16,384 유지율 | 유지분 입력 mean / p50 / p90 / p99 | 출력(abstract) mean / p50 / p99 | 유지분 중 >8.76k |
|---|---|---|---|---|
| Nano-9B-v2 | **91.7%** | 7,116 / 6,484 / 12,369 / 15,722 | 203 / 195 / 399 | **30.0%** |
| Qwen3.5-9B | 91.4% | 7,130 / 6,520 / 12,400 / 15,689 | 199 / 191 / 390 | 30.3% |
| Llama-3.1-8B | 93.2% | 6,922 / 6,321 / 12,107 / 15,752 | 198 / 191 / 373 | 28.0% |

- 프롬프트의 "~90% 유지, 평균 ~7.3K(Llama sentencepiece)"와 정합한다(현 토크나이저 기준 7.1K). Sarathi-Serve 보고치(중앙값 7,059, 출력 208)와도 맞는다. 모델 간 길이 차이는 3% 이내라 **모델 간 비교에서 토크나이저 교락은 작다**.
- 그림: `arxsum/arxsum_input_len_nano9b.png`. 봉우리는 4–6k에 있고, 유지분의 30%가 Nano-9B-v2의 **KV=state 경계(≈8.76k = 136.9 MiB / 16 KiB)**를 넘는다.
- ⚠️ 이름 충돌: Mooncake의 `arxiv.jsonl` trace는 arXiv-Sum 데이터셋이 아니다(통계가 toolagent와 사실상 동일, `MOONCAKE_STATS_2026-10-14.txt`). 문서에서 섞지 말 것.

### 2.2 arXiv-Sum 16k가 갖는 의의

1. **비교 가능성.** Sarathi-Serve(≤16k 그대로), Bullet(arXiv-Summary), Nexus(ArXiv-Sum)가 같은 데이터셋을 쓴다. PD-mux 선행 연구와 수치 축을 맞출 수 있는 유일한 공용 장문 세트다.
2. **통제된 장문 prefill.** Poisson 도착, prefix 재사용 없음(radix 문제 비발생), 128k 꼬리 없음, 입력:출력 ≈ 35:1. 모델·arm만 바꾸고 나머지를 고정하는 **교차 모델 비교(G2)**에 맞는 깨끗한 기판이다.
3. **CM-5 경계를 정확히 가로지른다.** 요청의 70%는 state가 KV보다 크고 30%는 반대다. hybrid의 "메모리 구속 자원이 무엇인가"가 한 워크로드 안에서 바뀐다(CM-5 X-6 실측 기회).
4. **PD-mux의 '홈그라운드'다.** 정본 기전(CM-4 §1-24)은 "un-chunked prefill이 decode를 멈추는 시간"이 ITL 꼬리를 지배한다는 것이다. 7k 토큰 prefill이 계속 들어오는 워크로드는 그 기전이 가장 크게 드러나는 곳이다. → **이점이자 약점**: 심사자는 "PD-mux에 유리한 워크로드를 골랐다"고 볼 수 있다. 이 세트 하나로 일반성을 말할 수 없다.
5. **decode가 가볍다(유도값).** rate 1.5 req/s × 출력 200 ≈ decode 300 tok/s, 동시 decode ≈ 1.5×200×0.03 s ≈ 9개. roofline B_max(195)와 비교하면 아주 작다. ⇒ anchor 주장 1의 핵심(hybrid decode floor를 generic 예측기가 틀리게 잡는가)은 **이 워크로드에서 거의 자극되지 않는다**. arXiv-Sum은 G0/G2용이고 anchor 사다리 5-a의 주 워크로드로는 약하다.
6. **용량 위치(유도값, 측정 아님).** prefill FLOP ≈ 2 × 8.9e9 × 7.1k ≈ 1.3e14/요청이다. A100 312 TFLOPS에서 MFU 50–60%를 가정하면 전 GPU prefill 포화는 ~1.2–1.5 req/s다. 프롬프트의 rate 0.2–1.5 스윕은 **포화 knee를 포함**한다. PD-mux에서는 prefill SM이 108보다 작으니 knee가 더 낮다. → 게이트 6(용량 먼저)이 그대로 걸린다.

### 2.3 각 워크로드가 논문 포지션에 주는 의미

어떤 세트를 주로 두느냐가 곧 "이 논문이 무엇에 관한 논문인가"를 정한다.

| 워크로드 | 성격 | 자극하는 비용 항 | 이것을 주로 두면 논문은… | 비교 대상 | 위험 |
|---|---|---|---|---|---|
| **arXiv-Sum 16k** | Poisson, 장문 입력·짧은 출력, 재사용 없음 | CM-2·CM-4(prefill stall)·CM-5 경계 | **"PD-mux 이득이 hybrid 계열·mixer·캐시 구성에 따라 어떻게 변하는가"**(특성화/일반화 논문, 프롬프트 G2) | Sarathi·Bullet·Nexus | PD-mux 유리 선택 의심, decode 축 약함 |
| **Mooncake Conversation** | 실 timestamp, p50 6.9k / p99 85k / 최대 126k, 재사용 36.6% | CM-4·CM-5(장문 꼬리)·CM-6(burst→entanglement)·CM-7 | **"실 서비스 trace에서 hybrid PD-mux 운영"**(외적 타당성 논문) | MuxWise·DuetServe | radix OFF면 재-prefill 편차, 비정상성 때문에 rate 축 정의가 어려움 |
| **ShareGPT** | 짧은 대화 | CM-1·CM-6 | 기존 결과와의 **연속성 앵커**(단독 주력 불가) | 거의 모든 선행 연구 | stationary r8은 정책 비교 벤치에서 폐기됨(게이트 2) |
| **OpenThoughts(추론)** | 출력 장문(평균 8.4k) | **CM-1(decode SM floor)**·CM-5(state 상수 vs KV 증가) | **"hybrid 고유 decode 비용이 결정을 바꾼다"**(anchor 주장 1 = 시스템 설계 논문) | MuxWise | decode-heavy라 PD-mux 이득 자체가 작을 수 있음 |
| **LooGLE** | 입력 장문, 출력 ~15 | CM-2 | MuxWise 장문 설정과의 직접 대조 | MuxWise | decode가 거의 없어 PD-mux 상호작용 빈약 |

**포지션 선택지(사용자 결정)**:

- **(P-A) 특성화/일반화**: arXiv-Sum을 주로 두고 ShareGPT를 앵커로 쓴다. 모델 축(Nano-9B-v2, Qwen3.5-9B/4B, Llama-3.1-8B)을 넓힌다. 이것은 프롬프트 G2이자 기여 프레이밍의 기여 1(비용 구조 특성화)이다.
- **(P-B) 시스템 설계(anchor 주장 1)**: Mooncake Conversation + OpenThoughts를 주로 둔다. arXiv-Sum은 비교 가능성 보조로 쓴다. decode floor가 자극돼야 하므로 출력 장문 축이 필수다.
- **(P-C) 혼합(권고)**: G0는 arXiv-Sum으로 한다(Nano-9B-v2 첫 운영점, 선행 연구와 정렬). 5-a는 Mooncake Conversation(주) + OpenThoughts(decode 축) + ShareGPT(앵커)로 한다. arXiv-Sum은 G2 교차 모델 표준 셀로 남긴다. 두 질문(PD-mux 일반화 / hybrid-aware 결정)이 서로 다른 워크로드를 요구한다는 점을 논문 구조에 드러낸다.

## 3. 비용 모델 A100 재계산 + G1 1차 검사

### 3.1 roofline 재계산 (`costmodel_a100/`, 유도값)

상수만 교체했다: `PEAK_FLOPS=312e12, PEAK_BW=2.039e12` (나머지는 원본과 동일, docstring 표기만 변경).

| 모델 (state fp32) | mixer:attn | KV/tok | state/seq | 8K B_max · tok/s | 8K 지배 층 | 128K B_max · tok/s | 128K 지배 층 |
|---|---|---|---|---|---|---|---|
| Nano-9B-v2 | 27:4 | 16 KiB | 136.9 MiB | 195 · 3,936 | Mamba 63% | 23 · 646 | Attn 69% |
| Qwen3.5-9B | 24:8 | 32 KiB | 49.1 MiB | 169 · 4,314 | Attn 58% | 12 · 356 | Attn 76% |
| Olmo-Hybrid-7B | 24:8 (MHA) | 120 KiB | 52.2 MiB | 53 · 1,475 | Attn 74% | (ctx 64k) | — |
| Llama-3.1-8B | 0:32 | 128 KiB | 0 | 52 · 1,497 | Attn 83% | 3 · 92 | Attn 81% |
| Qwen3.5-4B | 24:8 | 32 KiB | 49.1 MiB | 198 · 4,825 | Attn 64% | 14 · 408 | Attn 87% |

- state bf16: Nano-9B-v2 B_max=261 · 5,578 tok/s, Qwen3.5-9B 183 · 4,850. **순위 역전이 A100에서도 유지**된다(state dtype 고정 필수 근거 유지).
- A100/H100 tok/s 비는 전 모델 0.57–0.61로 대역폭 비 0.609와 같다. 즉 이 모델은 전 구간 memory-bound이고, 기판을 바꿔도 **비율만 바뀌고 순위·지배 층은 불변**이다.
- **이 모델이 말할 수 없는 것**: SM 축이 없다. 따라서 PD-mux 이득이나 SM 분할 효과를 예측하지 못한다. B_max는 KV/state 용량만 계산하므로 SGLang 실제 상한(`mem_fraction_static`, cudagraph bs 상한, `max_running_requests ≡ mamba 슬롯` CM-5)과 기계적으로 어긋난다. 측정 후 "예측 대비 편차"는 **어느 B_max와 비교하는지를 사전에 정의**해야 의미가 있다.

### 3.2 G1 이식성 (정적 검사 1차, dev tree `sglang_engine_dev`)

| 모델 | SGLang 등록 | ctx ≥16K | 비고 |
|---|---|---|---|
| Nano-9B-v2 | `NemotronHForCausalLM` ✓(기존 포트) | 131,072 | state dtype을 서버 인자로 명시 고정 필요 |
| Qwen3.5-9B / 4B | `Qwen3_5ForConditionalGeneration` ✓(`models/qwen3_5.py`) | 262,144 | VL 래퍼 클래스 — 텍스트 전용 경로·PD-mux split-prefill 배선은 미포트 |
| Olmo-Hybrid-7B | **✗ `OlmoHybridForCausalLM` 미등록**(olmo/olmo2/olmo3/olmoe만) | 65,536 | 모델 포트가 선행돼야 함 → G2 제외 후보 |
| Llama-3.1-8B | ✓ | 131,072 | 가중치 gated 접근 확인 필요 |

- GreenContext 안전성 grep(triton GDN `fla/`·`mamba/`): `grid.sync`/smid 인덱싱은 없었다. `layernorm_gated.py:173`의 `multi_processor_count`는 rows-per-block 성능 휴리스틱일 뿐이다(분할 시 108을 보게 되지만 정확성 문제 아님, piecewise cudagraph 켜면 상수). `causal_conv1d_triton.py:547` cooperative launch는 주석 처리돼 있다. **범위 한정**: flashinfer GDN 백엔드와 `sgl_kernel` CUDA 커널은 미검사다. 정확성 게이트(engine-porter)를 대체하지 않는다.

## 4. 방법론 게이트 — 프롬프트·새 방향과의 긴장 (논의용)

> ★**대체(2026-10-15 같은 날)**: 이 절은 `AGENDA_methodology_2026-10-15.md`가 대체한다. 정정 사항이 있다.
> TTFT SLO 형태는 "새로 정할 것"이 아니다. 이미 **A′(k·floor(L), k∈{3,5,10}, 2026-10-04 사용자 승인)**로 고정돼 있다.
> 열린 것은 혼합 길이에 A′를 적용하는 방식뿐이다. 아래 본문은 이력으로 남긴다.

프롬프트의 측정 규율과 CLAUDE.md 게이트 1–8(+ PROJECT_STATUS 게이트 #12·#13)을 대조했다. "그대로 적용", "적용 범위 논의 필요", "새로 정할 것"으로 나눈다.

### 4.1 게이트별 대조

**게이트 1 (정책 주장은 서빙 실증)** — 그대로 적용.
- 프롬프트는 "실험 전에 비용 모델 예측값 기록 → 측정 후 편차"를 요구한다. 이것은 게이트 1과 충돌하지 않고 오히려 강화한다. roofline은 **사전 예측**으로만 쓰고 결론 근거로 쓰지 않는다.
- 충돌점: 프롬프트 §3의 "headline 지표 = CUDA event latency + bandwidth estimator". 커널 수준 지표는 게이트 1상 마이크로 증거라 서빙 결론의 headline이 될 수 없다. **headline = 서빙 goodput(게이트 4)**, CUDA event는 기전 보조로 위치를 바꿔야 한다.

**게이트 2 (stationary ShareGPT r8 폐기, 정책 비교는 변화 trace)** — **적용 범위 논의 필요.**
- 이 게이트가 생긴 이유는 *동적 정책 비교*에서 static조차 ±1.3의 분산을 보였기 때문이다. G0(PD-mux vs fused)는 정적 시스템 두 개의 비교이고, 정본 P1-opint도 정상상태 단일 rate로 판정했다.
- 제안: 게이트 2를 "**동적 정책 비교**에는 변화 trace 필수, **정적 시스템/정적 split 비교**에는 고정-rate Poisson 허용(단 게이트 3·6 충족)"으로 명문화한다.
- 남는 문제: Mooncake처럼 timestamp가 있는 실 trace는 본질적으로 비정상이다. "rate 스윕"을 **시간축 압축 배율**로 정의할지, 이 경우 게이트 6의 "용량"을 무엇으로 정의할지 정해야 한다.

**게이트 3 (분산 먼저, n≥4, 3% 미만 headline 금지)** — **강화 필요.**
- 정본 통계 정정(2026-08-06): n=5 percentile bootstrap의 실제 coverage는 0.84다. n=5 paired 부호 순열검정의 최소 양측 p는 2/32 = 0.0625라서 **원리적으로 p<0.05에 못 닿는다**.
- ⇒ "n≥4"는 판정 가능성을 보장하지 않는다. 선택지:
  - (a) n≥6 (최소 p = 2/64 ≈ 0.031)
  - (b) 판정 검정을 t-CI로 사전 고정
  - (c) 둘 다
- 비용: arXiv-Sum 요청은 길어서 run당 시간이 ShareGPT보다 길다. n 증가는 Job 크기에 직결된다.

**게이트 4 (goodput = TTFT ≤ SLO ∧ 요청 내부 ITL p95 ≤ SLO)** — **프롬프트와 정면 충돌.**
- 프롬프트는 "TPOT SLO"(평균)를 쓴다. 정본에서 Granite는 mean-ITL 술어와 p95-ITL 술어 사이에서 **부호가 뒤집혔다**. 술어 선택이 결론을 바꾼다는 것이 이미 실증됐다.
- 권고: 주 술어는 ITL p95로 사전등록하고, TPOT 평균은 secondary로 둔다.
- TTFT SLO 형태(새로 정할 것):
  - 프롬프트: 고정 4–15 s (DistServe/TaiChi 선례)
  - 트랙 초안: 입력 길이 구간별 SLO (Vertumnus)
  - Bullet: 정규화 TTFT (ms/token)
  - 입력 길이가 3k–16k로 5배 퍼지는 워크로드에서 고정 SLO는 긴 요청에 체계적으로 불리하다. "goodput 손실이 어느 길이 구간에서 오는지"가 고정 SLO에서는 지시함수에 묻힌다. 고정 SLO를 쓰려면 길이 구간별 분해를 병기하는 것이 최소 조건이다.

**게이트 5 (동적 결과엔 switch_count·체류분포·p50/p95/p99)** — G0·G2가 정적이면 해당하지 않는다. 5-a 이후 동적 arm에는 그대로 적용한다.

**게이트 6 (metric cliff 회피, 용량 먼저)** — **그대로, 단 구체화.**
- 2.2-6의 유도값으로 보면 rate 0.2–1.5 스윕은 knee를 포함한다. 순서는 이렇게 고정한다: 용량 Job(fused·PD-mux 각각의 지속 가능 rate) → 그 뒤 판정 rate를 cliff에서 떨어진 곳에 사전등록.
- PROJECT_STATUS 게이트 #12(임계 지시함수 판정은 **임계 사다리·큐-성장 검정**으로 견고성 제시)도 함께 적용한다. TTFT SLO가 4–15 s 범위로 주어진 것은 사실상 임계 사다리이므로, 단일 SLO가 아니라 사다리 전체에서 부호가 유지되는지를 판정 규칙으로 쓸 수 있다.

**게이트 7 (라운드 합칠 때 duration 합산)** — 그대로. 하네스 코드 수준 확인 항목이다.

**게이트 8 (SLO 바꾸면 재스코어 금지, 재튜닝)** — **G0의 공정성 문제로 확장해야 한다.**
- chat SLO에서 고른 split·fused 설정을 summarization SLO에 그대로 쓰면 게이트 8 위반이다.
- 더 큰 문제: 정본에서 fused는 **미조율**이었다. 새 워크로드·새 SLO에서 "PD-mux vs fused"를 말하려면 양쪽 다 그 SLO로 튜닝해야 한다. PD-mux 쪽은 static split 선택, fused 쪽은 E-D 노브 스윕(`chunked-prefill-size` 2048/4096, `max-running-requests`, `num-continuous-decode-steps`, `prefill-max-requests`, schedule-conservativeness).
- 그러지 않으면 1절의 "미조율 fused" confound가 그대로 재생산된다. 이것이 G0 설계에서 가장 큰 결정이다.

**PROJECT_STATUS #13 (엔진 제약으로 인한 묶음 처치)** — `--enable-pdmux`의 3-플래그 강제는 그대로다. G0 결과는 "PD 분리"가 아니라 "PD-mux 모드(묶음)" 효과로 서술한다. 분리하려면 fused + `--chunked-prefill-size -1` + `--disable-overlap-schedule` arm(E-A 계열)을 같은 Job에 넣는다.

**기판 주의(게이트 1 연장)** — VESSL A100 SXM은 KISTI A100과 SM 수(108)는 같다. 하지만 N2에서 **SW power cap 노출**이 확인됐다(CM-4 크기 교락). 옛 P1-opint 수치와 새 수치를 섞지 않는다. 한 비교의 arm은 한 Job 안에 둔다. 클럭·전력 수집(N2 F-2)을 G0 Job 계측에 기본 포함할지 정해야 한다.

### 4.2 새로 정해야 할 규칙 (사용자 논의 안건)

1. **게이트 2의 범위**: 정적 비교에 고정-rate Poisson 허용을 명문화할지. 실 trace에서 "rate"의 정의(시간축 배율?)는 무엇으로 할지.
2. **판정 통계**: n과 검정(t-CI / 순열 / bootstrap) 조합을 사전 고정. n=5 순열검정 불능 문제 해소.
3. **TTFT SLO 형태**: 고정 사다리(4–15 s) vs 길이 구간별 vs 정규화(ms/token). 주 술어 하나 + 나머지 secondary.
4. **ITL 술어**: p95(정본) 고정 확인. TPOT 평균은 secondary.
5. **fused 베이스라인 정의**: "기본 설정 fused"(선행 연구 관행·재현성)와 "SLO-튜닝 fused"(공정성) 중 무엇을 주 대조로 할지. 둘 다 둔다면 어느 쪽 격차를 headline으로 할지.
6. **headline 지표 위계**: 서빙 goodput > TTFT/ITL 분포 > CUDA event 기전 지표. 프롬프트 §3 문구 교체.
7. **예측-측정 대조 규약**: roofline B_max·층 비중 예측을 사전 기록할 때 비교 대상(엔진 실측 `max_running_requests`? 실측 점유?)을 정의.
8. **전력·클럭 계측 기본 포함 여부**: CM-4가 전력 채널과 교락돼 있으므로, 새 Job 전부에 클럭 수집(F-2)을 기본 탑재할지.

## 5. 산출물

- `costmodel_a100/` — A100 상수 roofline 스크립트·cfgs·`dense_summary_s{32,16}.json`·층 비중 그림 2장
- `arxsum/` — `arxsum_stats.py`, `plot_arxsum_len.py`, `ARXSUM_STATS_2026-10-15.txt`(데이터·토크나이저 sha256 포함), `arxsum_input_len_nano9b.png`
- 원 데이터(parquet 104 MB, 토크나이저 3개)는 git 밖. 재생성 경로는 stats 파일 하단에 있다.
