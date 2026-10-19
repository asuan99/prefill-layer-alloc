# DESIGN v1 — cap 블록 + long-context 블록 (R16-3 · CM-5 · CM-1 ctx 항 · Claim E′) — 초안, PREREG 아님 (2026-10-19)

> 지위: main 트랙 설계 초안이다. 측정 0, 감사 0. 수치는 [유도]·[제안]이며 판정 규칙은 PREREG에서 확정한다. v0(`DESIGN_cap_characterization_v0_2026-10-19.md`)을 대체한다.
> 근거:
> - 결정 `../anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md` §1–7·8′
> - 명세 `../handoff_review_2026-10-15/SPEC_common_measurement_v0_2026-10-15.md` §9·§12
> - 비용 메모 `../handoff_review_2026-10-15/COSTMODEL_running_cap_2026-10-15.md`(§3-2 문장은 8′로 대체)
> - 감사 `../anchor_audit/VERDICT_claimE_first_arrow_2026-10-19.md`
> - J1 분석 `../multigpu_verdicts/ANALYSIS_J1_N1_2026-10-19.md`
> - I-5 구현 `src/multiplex/admission_probe.py`(CPU 검증만, 엔진 트리 미동기화)

## 0. 질문

| ID | 질문 | 블록 | 연결 |
|---|---|---|---|
| Q-A | 짧은 ctx(< s/k)에서 고정 state s가 메모리를 지배할 때, cap을 C\_mem까지 풀면 무엇이 admission을 막고 goodput은 어떻게 변하는가 | cap | CM-5, E′ 짧은 구간 |
| Q-B | 그 변화가 hybrid와 attention 대조에서 다른가(이중차분) | cap | 결정 4 "왜 hybrid인가" |
| Q-C | G0 cap 값 | cap | R16-1·D-17 |
| Q-D | 긴 ctx(32k–96k)에서 KV가 지배할 때 hybrid의 동시 수용·decode 비용·PD-mux 이득이 ctx에 따라 어떻게 변하는가, 그리고 대조 모델과 어떻게 다른가 | long-ctx | CM-1 ctx 항, CM-5, G(PD-mux 이득), E′ 긴 구간 |

Claim E 원안(decode floor 예측)은 이 설계에서 시험하지 않는다(결정 7).

## 1. 비용 구조 [코드 사실 + 유도]

- 요청당 메모리 = s + k·L이다. Nano-9B-v2는 s ≈ 0.1435 GB(fp32 conv+ssm, 길이와 무관), k ≈ 16 KiB/토큰이다. 교차점 L\* = s/k ≈ 9k [유도, GB/GiB 단위 확인 필요]이다. 순수 attention 모델은 s = 0이고 k가 훨씬 크다(attention 층이 많다).
- SGLang 정적 풀 [코드]:
  - mamba pool은 부팅 때 `torch.zeros`로 잡힌다(`memory_pool.py:245-262`).
  - radix OFF에서는 슬롯 수 = cap이다(`model_runner_kv_cache_mixin.py:223-230`).
  - 슬롯이 고갈되면 assert로 프로세스가 죽는다(`memory_pool.py:544`). 그래서 cap ≡ 슬롯이다.
  - I-5 조사 결과 거절 사유는 **KV / cap(≡슬롯) 2분류**다. 이 결합은 할당기에 종속되므로 기여 문장이 아니라 실험 설정 공시로 쓴다(결정 8′).
- C\_mem(L) = Pool / (s + k·L) [유도]: 2k ~270, 7.1k ~183, 16k ~116, 64k ~39, 128k ~20. 64k 이상에서는 C\_mem < 48이다. 그래서 long-ctx 블록에서는 cap 스윕이 무의미하고 cap = C\_mem(L) 하나만 둔다.

## 2. 요인

### cap 블록
| 요인 | 수준 |
|---|---|
| 모델 | Nemotron-Nano-9B-v2 · 순수 attention 대조(§5 열린 질문 2) |
| cap | {48, C\_mem}. C\_mem은 (모델, 셀)마다 부팅 로그로 실측한다. C\_mem/2는 선택 |
| 셀 | arXiv-Sum 16k(입력 p50 6.5k, 34%가 > 8k라 L\*의 양쪽에 걸친다 [측정]) · shape C(데이터셋 미정) |
| arm | fused\_sym · agnostic |
| seed | 4 |
| 분석 | 요청을 L < L\*와 L ≥ L\*로 층화하고, 층마다 거절 사유 몫(iteration 가중)과 TTFT·ITL을 본다 |

### long-ctx 블록
| 요인 | 수준 |
|---|---|
| 모델 | 같은 두 모델(대조 모델의 장문 지원이 선행 조건) |
| ctx | {32k, 64k, 96k} 고정 길이 또는 장문 데이터(§5 열린 질문 3). 128k는 프로브 결과에 따라 넣는다 |
| cap | C\_mem(L), 부팅 실측 |
| arm | fused\_sym · agnostic |
| 출력 길이 | 등록값(짧음/긴 것 중 하나 이상을 shape C와 맞춘다) |
| seed | 4 |

공통 고정값: radix OFF(F8), cudagraph ON, mem-fraction 0.82, `--max-mamba-cache-size = cap` 명시, 한 비교의 arm은 한 Job 안. long-ctx 블록은 `context_length` ≥ 100k다(현재 20480은 표준 셀 전용).

## 3. 계측
- **I-5**: `pdmux.admission/v1`. 사유 KV / cap(≡슬롯), sticky 상속 분리, decode bs 히스토그램, cap 이상 iteration 수, retract, `cap_slot_relation`. 몫은 시간 또는 iteration 가중으로 낸다(PD-mux가 결정 기회를 더 많이 갖기 때문이다). 엔진 트리 동기화와 GPU correctness gate가 먼저다.
- **I-6** cudagraph 패딩(cap에 따라 캡처 목록이 바뀐다). **I-1** 클럭·전력: 비는 400 W 한도에서의 총효과로만 보고한다(J1 F-1 문안).
- **부팅 로그**: `max_total_num_tokens`, 슬롯 수, KV 풀 바이트, 가중치 바이트(대조 모델 교락 공시용).
- **I-4** decode 커널 GPU 시간: long-ctx에서 decode가 느려진 것인지 span 정지 때문인지 가르려면 필수다(Q-D).

## 4. estimand와 판정 초안 [제안]
- **Q-A**: L 층별 사유 몫, λ\*\_SLO(cap = C\_mem) / λ\*\_SLO(cap = 48), Job 안 짝 bootstrap. |효과| < 3%면 방향을 인용하지 않는다.
- **Q-B**: 이중차분 Δ = [log λ\*(C\_mem) − log λ\*(48)]\_hybrid − [같은 값]\_attn. 부호까지만 "hybrid 고유" 문장에 쓴다(모델 크기·가중치 교락).
- **Q-C**: 두 셀 × 두 arm 모두에서 `CAP_BINDING`(10%, 짝 규칙)이 아닌 최소 cap. 없으면 C\_mem. 두 arm 모두를 규칙에 넣어 대칭을 지킨다.
- **Q-D**:
  - (i) ctx별 G\_SLO(L) = λ\*\_SLO,agnostic / λ\*\_SLO,fused\_sym와 그 ctx 추세(log G ~ log L 기울기, seed 공동 재표집)
  - (ii) 실현 동시 수와 C\_mem(L)의 대비(모델별)
  - (iii) decode step 시간의 ctx 의존(I-4)
  - 대조 모델과의 비교는 (ii)·(iii)의 형태 차이까지만 쓰고 G 크기 비교 문장은 금지한다(모델 교락).
- **SLO**: 짧은 셀은 G0의 A′, long-ctx는 **ctx 비례 TTFT SLO**(예: α·TTFT\_floor(L), TTFT\_floor ≈ 0.111 ms/토큰 [J1 측정, ≤16k 선형] — 장문에서는 선형이 깨질 수 있어 프로브로 다시 잰다)와 고정 ITL p95 SLO. metric cliff를 피하도록 용량을 먼저 측정한다(게이트 6).
- 공통: goodput 정의는 게이트 4, duration 합산(게이트 7), n ≥ 4.

## 5. 열린 질문
1. **shape C 데이터셋**: cap 블록 두 번째 셀이자 long-ctx 출력 길이 기준. 후보는 OpenThoughts/Mooncake(10-15 P-C 논의)다.
2. **attention 대조 모델**: Qwen2.5-7B는 기본 32k라 64k 이상은 YaRN이 필요하고 처치가 하나 늘어난다. 대안은 128k 네이티브 attention 모델(후보 조사 필요)이다. 같은 백엔드여야 한다(gate #83).
3. **long-ctx 워크로드**: 고정 길이 합성(통제가 쉬움) 또는 실제 장문(Mooncake conversation p99 85k, LooGLE — 분포가 넓어 ctx 축이 흐려짐). 권고는 합성 고정 길이를 1차로, 실제 장문을 보조로 두는 것이다.
4. **128k 프로브**: 기존 결정 대기 항목이다. long-ctx 격자 상한과 `context_length`를 정한다. 프로브는 Workspace에서 bring-up 전용으로만 한다(측정은 Job).
5. **블록 배치**: cap 블록과 long-ctx 블록을 한 Job에 넣을지 나눌지. long-ctx는 λ\*가 낮아 셀이 길다. 장문 prefill은 J1의 선형 계수 9.0k tok/s보다 느릴 수 있다. GPU-h는 PREREG에서 J1 셀 시간 모형(확장 발화 포함)으로 계산한다. 블록 안에서 두 모델과 두 arm은 같은 Job에 둔다.
6. ~~radix ON 비율 모드 arm~~: 제외한다(F8). 범위 문장으로 처리한다.

## 6. 금지 문안
"generic 스케줄러는 mamba state를 무시한다" · "Bullet/MuxWise가 C\_mem을 오예측한다" · "~52k가 hybrid 경계" · "9k 경계를 실측했다"(config 산술) · "슬롯 상한이 hybrid 고유" · "Claim E 반증" · J1 비의 크기 문장 · 대조 모델 대비 G 크기 문장.

## 7. 다음 단계
§5의 1–4 결정 → 128k 프로브(Workspace) → I-5 엔진 트리 동기화 + GPU correctness gate → PREREG(블록별) → 규칙층·하네스층 감사 → 봉인 → CPU preflight → `launch.sh` dry-run → 승인 → 제출. G0 PREREG는 Q-C 결과를 받는다.
