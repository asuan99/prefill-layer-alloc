# 프로젝트 설계 요소 정리 · span→ITL 인과 사슬 정정 · 단일/multi-GPU 통합안 (2026-10-15)

> 지위: main 트랙 검토 메모. 측정 0, 새 판정 0. [C]=코드·정본에 적힌 사실, [I]=추론(측정으로 확인 필요).
> `AGENDA_methodology_2026-10-15.md` §4.2–4.3의 "span이 많아지면 정지가 길어진다"는 서술을 이 문서 §2가 **정정**한다.

## 1. 이 프로젝트가 설계한 것 vs upstream PD-mux

### 1.1 upstream SGLang v0.5.10 PD-mux가 이미 가진 것 [C]

| 요소 | 위치 | 동작 |
|---|---|---|
| 이벤트 루프 | `multiplex/multiplexing_mixin.py:1240` `event_loop_pdmux` | 한 반복 = decode 1 step 발행 + prefill span 1개 발행 + decode 동기화 |
| 층 단위 split prefill | `:1415-1428`, `pdmux_context.py:20` | span 하나 = `split_forward_token_budget(65536) // 배치 토큰` 개 층. 진행 위치는 `split_index` |
| prefill 자리 1개 | `:1218-1220` | 진행 중인 배치가 끝나야 새 배치 형성 |
| green context stream 쌍 | `pdmux_context.py:124-138` | 분할 비율마다 (prefill, decode) stream 쌍을 미리 생성. 양 끝은 일반 stream |
| 분할 선택 | `:1151-1216` `adjust_stream_groups` | **prefill 시작·종료 때만**. 자동 격자(decode_bs 비례) 또는 manual 임계표 |
| decode-empty fallback | 같은 곳 | decode가 비면 무분할로 복귀 (§1-22: 라벨 ≠ 실현) |
| chunked prefill 금지 | `server_args.py:6125-6137` | `chunk == -1`, overlap off를 assert로 강제 |

### 1.2 이 프로젝트가 더한 것과 각각의 현재 판정

| 설계 요소 | 해결하려던 문제 | 구현 | 현재 판정 |
|---|---|---|---|
| **hybrid용 `forward_split_prefill`** (NemotronH·Zamba2·Granite·Falcon-H1) | hybrid 모델에서 PD-mux 자체가 돌지 않음. span 경계를 넘어 hidden state와 Mamba state를 이어야 함 | `src/patches/nemotron_h_forward_split_prefill.patch`, `src/models/zamba2.py` | 동작함. 4모델 정확성 게이트 통과. **이 프로젝트의 기반 기여** |
| **layer-type 인지 decode SM** (layer_aware, 조율 per-type 창) | attn·SSM 층의 SM 민감도 차이를 decode 중 층마다 이용 | `models/*` 의 `la_coord_windows`, `PDMUX_SLO_SPAN_TYPE` | **全형태 死**(TPOT 42→124 ms). 층 단위 창(0.4–0.8 ms)이 prefill 커널 지속(~2.2 ms)보다 짧음 |
| **span마다 분할 재결정** (SLO 컨트롤러 v7, R2 정책 fixed/generic/hybrid) | upstream은 prefill 시작·끝에서만 분할을 고름. span 경계를 개입 지점으로 써서 decode 굶김을 막으려 함 | `multiplexing_mixin.py:1313-1355` ("decide the target split EVERY prefill span") | 결정 비용 32–36 µs로 무시 가능(E), 전환 ≤0.04 ms(CM-3). 그러나 **동적은 best static을 못 넘음**(HE0). 원인은 positioning + entanglement |
| **admission latch** (R2) | 대기열이 decode를 굶기는 사슬(§1-4) 차단 | `:1221`, `_r2_admission_holds` | R2 하위, 미검증 가설 H-Policy의 일부 |
| **sticky partition** | decode-empty fallback 때문에 "D SM에서 decode"라는 라벨이 실현과 어긋남 | `PDMUX_STICKY_PARTITION` | 실현 ≥0.999로 라벨 문제 해결(S2, CONFIRMED scoped). 대신 estimand가 단일 테넌트 decode에 가까워짐 |
| **true dual worker** | 단일 스레드 루프의 결합(coupling) 감소 | `PDMUX_TRUE_DUAL_WORKER`, `dual_worker.py` | 구현 완료, **성능 미검증**(H-Architecture) |
| **계측** (telemetry, phase events, holb_probe, chunk_probe) | 정지·HOL·chunk 관측 | `src/multiplex/*.py` | holb_probe: "decode 일감이 있는데 decode forward가 없는 GPU 시간"을 4 arm 공통 정의로 잼. G5 관측자 효과는 미결정 |

**span 설계 의도에 대한 정리**:
- span은 prefill 안에 **개입 지점**을 만드는 장치다. 두 가지를 노렸다.
  - 경계마다 decode 반복이 끼어들 수 있게 한다.
  - 경계에서 SM 분할을 다시 고를 수 있게 한다.
- upstream은 앞의 것(decode 끼워 넣기)만 쓰고, 분할은 prefill 시작·끝에서만 바꾼다. 이 프로젝트는 뒤의 것(span 단위 재분할)을 더했다.
- 측정 결과는 이렇다. **개입 지점의 비용은 무시할 만했지만, 거기서 내린 동적 결정이 static을 이기지 못했다.** 즉 span이라는 구조가 아니라 그 위의 정책이 죽었다.

## 2. span → ITL 꼬리: 인과 사슬 (정정)

### 2.1 정본이 확정한 현상 [C] (CONSENSUS §1-24, 2026-08-02)

- 절대 토큰 방출 시각을 재구성했다. rate 2에서 stall probe 17개 중 16개는 그 구간 내내 어떤 요청이 prefill 중이었다. 그 요청은 거의 항상 probe 안의 최장 프롬프트(2.5–2.8k 토큰)였다.
- 정지 크기는 고정 프롬프트에서 **D에 단조 증가**한다: d24 168 → d44 226 → d54 264 → d92 866 ms. prefill SM(108−D)이 줄기 때문이다.
- ⇒ ITL 꼬리는 decode step 시간이 아니라 **prefill이 decode를 멈춘 시간**이 지배한다.

### 2.2 그 현상이 루프 구조에서 어떻게 나오는가 [I — 코드 읽기 + §1-24 크기 정합, 계측으로 분리 안 됨]

legacy 루프 한 반복(`:1380-1500`):
```
decode step 발행 (decode stream, cudagraph)
→ prefill span 발행 (prefill stream, eager, 층 k개)
→ decode_stream.synchronize()
→ 토큰 방출
```
- GPU 위에서 두 stream은 겹쳐 돈다.
- 그런데 다음 decode step은 **이번 반복의 span 발행이 끝난 뒤에야** 발행된다.
- §1-24의 정지 크기가 prefill SM 수에 반비례하는 것은 "span 발행 경로 어딘가에 앞선 prefill GPU 작업을 기다리는 host 동기화가 있다"는 가설과 맞는다. 정지가 CPU launch 시간만이라면 SM 수와 무관해야 하기 때문이다.
- 크기 검산: Nemotron-H-8B 52층, 2.5k 프롬프트 → span = 26층 × 2.5k ≈ 65k token-layer. 전 GPU에서 이 prefill 절반은 ~0.1 s이고, 16 SM이면 ~6.75배 → ~0.7 s. d92 실측 866 ms와 같은 자릿수다.
- ⇒ **prefill이 진행 중일 때 decode 토큰 하나의 ITL ≈ max(T_dec(D), T_span(108−D))**.

### 2.3 그래서 span 수와 길이는 ITL과 어떻게 연결되나 (앞 서술 정정)

`T_span ≈ (span 하나의 일 = 층 수 × 배치 토큰) / (108−D SM의 prefill 처리율)`

- 층 수 = 65536 // 배치 토큰이므로 **span 하나의 일은 배치 길이와 무관하게 ≈ 65,536 token-layer로 거의 일정**하다. 단 두 경우는 예외다.
  - 배치 토큰이 65,536을 넘으면 1층/span으로 바닥에 닿고, 그때부터 span 일이 길이에 비례해 커진다.
  - attention의 L² 항은 길이에 따라 커진다(Nano-9B-v2는 attn 4층이라 16k 이하에서는 작음).
- 따라서 입력이 길어져도 **정지 한 번의 크기는 대체로 그대로**다. 늘어나는 것은 **정지 횟수**(span 수)와 그 정지에 걸리는 decode 토큰 수다.
- ITL 술어는 **요청별 p95**다. 한 요청의 토큰 가운데 5% 넘게가 prefill 창에 걸리면 그 요청의 p95가 정지 크기(≈ T_span)로 올라간다.
  - 긴 입력이 계속 들어오는 워크로드는 prefill 창이 시간축 대부분을 덮는다.
  - 그러면 대부분의 요청에서 p95 ≈ T_span이 된다.
- 정리하면 **"span이 많아질수록 정지가 길어진다"는 틀렸고**, 올바른 사슬은 이렇다.
  1. 긴 입력 → prefill 창이 시간축을 더 많이 덮음 → 정지에 걸리는 토큰 비율이 5%를 넘음 → 요청별 p95가 T_span 수준으로 올라감.
  2. T_span 자체는 span 예산(65536)과 prefill SM(108−D)이 정한다. D를 키우면 커진다(§1-24).

### 2.4 이 사슬이 함의하는 것 [I]

1. **span은 문제의 원인이 아니라 완화책이다.**
   - span이 없으면(prefill 한 덩어리) 정지는 prefill 전체 길이가 된다.
   - fused와 비교해 보면, fused 기본(chunk 8192, overlap ON, decode와 섞지 않음)에서 decode는 chunk forward가 끝나기를 기다린다. chunk 하나 = 8192 토큰 × 56층 ≈ 459k token-layer를 108 SM에서 처리한다.
   - PD-mux span은 ≈65k token-layer를 64 SM(D44)에서 처리하므로, 시간으로 대략 65/459 × 108/64 ≈ **1/4**이다.
   - 이것은 정본의 "PD-mux 이득은 꼬리, 비용은 중앙"(decode를 적은 SM에서 돌려 중앙값 5–45% 손해, 꼬리 정지는 짧음)과 같은 방향의 기전 후보다. **기전 귀속은 정본상 미확립**이므로 가설로만 둔다.
2. **`split_forward_token_budget`가 ITL 꼬리의 1차 손잡이다.**
   - 예산을 줄이면 T_span이 줄지만, span 수와 반복 수가 늘어 S_prefill과 오버헤드가 커진다.
   - 82개 config 전부 65536 기본값이고 **스윕 0회**다. G0에 넣을지 결정이 필요하다(AGENDA 안건 9 확장).
3. **ITL 50 ms는 prefill 창 안에서 달성 불가일 수 있다.**
   - Nano-9B-v2 arXiv-Sum 7k 배치 → 9층/span, D44에서 T_span ≈ 수백 ms(유도).
   - 따라서 attainment는 "요청 토큰의 몇 %가 prefill 창에 걸리는가"로 정해지고, 이것은 rate와 입력 길이의 함수다.
   - **floor 측정 블록에서 무경합 decode ITL과 함께 '단일 span 동거 ITL'도 재야** 판정 rate를 cliff 밖에 둘 수 있다(게이트 6).
4. 계측 공백(AGENDA §4.4의 1·2번)이 바로 이 사슬을 검증하는 데 필요하다: span별 GPU 시간, decode 커널만의 GPU 시간, 반복 벽시계.

## 3. 단일 GPU와 multi-GPU 통합

### 3.1 현황 대조

| 항목 | main (단일 GPU, G0 / 5-a) | multigpu (M3, 미제출) |
|---|---|---|
| 모델 | Nano-9B-v2 | **Nano-9B-v2-Base**, revision 고정 (같음) |
| 워크로드 | 미정: arXiv-Sum / Mooncake / OpenThoughts / ShareGPT | random shape A(256/512), B(8192/64). 감사 의견: shape C 부재 |
| arm | 미정 (AGENDA 안건 5) | **fused**(기본, chunk 8192·overlap ON) · **fused_sym**(chunk −1·overlap off, PD-mux와 같은 스케줄 노브, 2026-10-12 사용자 결정) · **agnostic**(manual 등가표) · **d44**(R2 fixed) |
| 판정량 | 미정. N3 방식 = A′ attainment 90% 용량 | **G_sym = λ\*_agnostic / λ\*_fused_sym** (처리율 knee 용량비). 2차 G_def, G_d44 |
| 용량 정의 | A′ SLO attainment (TTFT k·floor ∧ ITL p95 50 ms) | 처리율 포화(KNEE_BRACKETED, 큐 성장 g). **SLO는 기술값** |
| 반복 | 미정 (안건 2) | 같은 boot 안 seed 4개, boot 효과 미분리 공시 |
| 기판 | VESSL A100 ×1 | x8 노드 하나에서 N 부분집합, 블록마다 N=1 앵커, GPU 편차 블록 |
| 코드 | main | `multigpu-scaling` 브랜치. `src/multiplex` 변경이 N2/N3/E2 봉인 digest를 깨므로 **그 캠페인 종료 전 merge 금지** |

**결론: 단일 GPU G0는 M3 설계의 N=1 단면과 거의 같은 실험이다.** 다른 점은 워크로드와 용량 정의뿐이다.
- 안건 5(fused 정의)는 multigpu가 이미 **fused / fused_sym 2단 대조**로 답했다. 사용자도 2026-10-12에 이를 결정했다.
- 이 2단 대조는 단일 GPU의 "3-플래그 묶음 처치" 문제를 그대로 푼다. fused → fused_sym은 스케줄 노브 효과이고, fused_sym → agnostic은 PD-mux 기능 효과다.

### 3.2 통합 원칙 (제안)

1. **측정 명세 하나, 실험 축 둘.**
   - goodput·용량·SLO·arm·n·계측을 정의하는 **공통 측정 명세** 1개를 둔다(AGENDA 안건 결정 결과).
   - 단일 GPU와 multi-GPU PREREG는 둘 다 이 명세를 인용한다.
   - 단일 GPU = N=1에서 **깊이**(Job 반복, 기전 계측, span 예산·D 스윕, 워크로드 다양성)
   - multi-GPU = 같은 arm·워크로드에서 **N 축 폭**
2. **arm 집합 통일**: {fused, fused_sym, agnostic, static D}. 여기에 단일 GPU 전용으로 유한 fused 튜닝 스윕(선택)을 더한다.
3. **용량을 두 가지로 병기**: λ\*_thr(처리율 knee, multigpu 방식)와 λ\*_SLO(k)(A′ attainment, N3 방식).
   - 처리율 knee는 SLO와 무관해 N 간 비교가 쉽다.
   - SLO 용량은 사용자 관점이고 논문 headline에 가깝다.
   - 두 값이 갈리면 그 자체가 결과다(예: PD-mux가 처리율은 같고 꼬리만 개선).
4. **N 간 SLO 문제 (결정 필요)**: A′ floor는 N·mode에 따라 바뀐다. TP는 floor를 낮춘다.
   - (a) N마다 자기 floor로 k를 적용 → "같은 상대 SLO"
   - (b) N=1 floor로 고정한 절대 T를 모든 N에 적용 → "같은 사용자 SLO"
   - 연구 감사(D)는 N 간 비교에 절대 SLO가 필요하다고 봤다. 권고는 **판정 = (b), 기술 = (a) 병기**.
5. **워크로드 통일**: shape A/B(합성 앵커) + arXiv-Sum 16k(공용 장문) + 5-a 실 trace. multigpu M3 PREREG 개정 대기 항목("shape C 부재", "`max_prefill_tokens` 미등록")을 이 통일로 함께 닫는다.
6. **코드 통합 순서**: N2 후속(F-1…F-4) 결정 → N2/N3/E2 봉인 캠페인 종료 선언 → `multigpu-scaling` merge → 공통 엔진 트리 하나. 그 전까지 G0는 main 트리에서 돈다. 이 경우 G0와 M3의 N=1 앵커는 **바이너리가 달라 직접 비교 금지**다.
7. **문서 위치**: 공통 측정 명세는 정본 성격이므로 main 세션 `promote`로 반영한다. 각 트랙은 링크만 건다(트랙 모델 규칙).

### 3.3 실행 순서 제안

| 단계 | 내용 | 결정 주체 |
|---|---|---|
| 1 | AGENDA 안건 1–9 결정 → 공통 측정 명세 초안 | 사용자 |
| 2 | N2 종료 여부(F-1…F-4 중 무엇을 할지) → merge 가능 시점 확정 | 사용자 |
| 3 | 명세 claims-auditor 규칙층 감사 → promote | main |
| 4 | M3 PREREG 개정(명세 인용, arXiv-Sum shape 추가, 절대 SLO) — multigpu 트랙의 대기 결정 "개정 여부"에 대한 답 | multigpu 세션 |
| 5 | G0 PREREG = 명세 + N=1 깊이 항목(Job ≥3, span 예산 2–3점, 계측 보강) | main |
| 6 | 계측 보강 패치(engine-porter, correctness gate) → 두 트랙 공용 | engine-porter |

### 3.4 논문 구조와의 연결 [I]

- 단일 GPU: "hybrid PD-mux의 비용 구조와 이득의 기전"(span 정지, D 선택, hybrid 메모리 구속).
- multi-GPU: "그 이득이 N·TP/DP에서 어떻게 변하는가"(일반화 축).
- 둘을 같은 명세와 arm으로 묶으면 포지션 (P-A 특성화) 또는 (P-C 혼합)의 일반화 절이 하나의 그림이 된다. 예: G_sym(N), 워크로드별 선.
