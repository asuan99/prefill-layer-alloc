# PLAN — hybrid 서빙 비용 모델: 항별 최소 측정 (2026-10-20, main, 결정 14) — 계획, PREREG 아님

> 지위: 결정 14(프레이밍)에 따라 설계를 다시 짠 것이다. 측정 0, 감사 0. GPU-h는 거친 [유도]이고 PREREG에서 다시 계산한다.
> 이 문서가 대체하는 것: `PREREG_CAP_LONGCTX_rev0_2026-10-20.md`(질문마다 전 요인을 곱해 Lean 약 123 GPU-h; Q-B 이중차분은 처치 방향이 모델마다 반대라 성립하지 않음).
> 근거: `COSTMODEL.md` CM-1…7 · 결정 `../anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md` 1–14 · 128k 프로브 `probe_128k/RESULT_…`·`AUDIT_…` · J1 `../multigpu_verdicts/ANALYSIS_J1_N1_2026-10-19.md`·`AUDIT_J1_N1_analysis_2026-10-19.md`.

## 0. 프레이밍 (결정 14)
- **주 목적 (C)**: hybrid 서빙 비용 모델을 만든다. 대상은 Nemotron-Nano-9B-v2-Base이고, 기준 기판은 A100 단일 GPU이며, 여기에 N·TP 항을 확장한다.
- **(G) PD-mux 이득**은 그 비용 모델이 설명하고 예측해야 하는 **검증 대상**이다. G0와 M3 셀이 검증 데이터가 된다.
- **(H) "왜 hybrid인가"**는 비용 모델 안의 **hybrid 고유 항을 transformer와 대비**해서 답한다. Llama-3.1-8B는 goodput 비교 대상이 아니라 그 항들의 **최소 대비 측정**에만 쓴다.
- 측정 방식도 바뀐다. 항은 **고정 배치·고정 길이의 엔진 내 측정**(마이크로)으로 채운다. rate 사다리(λ\*)는 마지막 검증 단계에서만 쓴다.

## 1. 비용 모델 골격 (lockstep PD-mux, 기존 정본 구조)
- 반복 하나 = decode step 1개 + prefill span 1개다. prefill 창 안의 ITL은 max(T\_dec·ρ, T\_span)이다(DESIGN\_span; CM-4).
- TTFT = W\_queue + Σ T\_span + 통신(CM-6).
- 동시 수 상한 = min(cap, C\_mem(L)), 수요 = λ·E[out]·E[ITL](Little 법칙; CM-5·CM-6).
- 여기에 N·TP 항이 붙는다: 샤딩에 따른 s/N·k/N(KV head 수 조건), collective 시간, rank lockstep.

## 2. 항별 최소 측정

| 항 | 무엇을 잰다 | Nano (주) | Llama (H 대비, 최소) | TP2 (N 항) | GPU-h [유도] |
|---|---|---|---|---|---|
| **T1** CM-1 `T_dec(D, bs, ctx)` | 고정 배치 decode step 시간(I-4 decode 커널 시간 포함) | D ∈ {16, 24, 34, 44, 64, 108} × bs ∈ {1, 8, 32, 48, 96} × ctx ∈ {2k, 8k, 32k, 64k}, 2회 | D = 108 × bs {8, 48} × ctx 4점 (**ctx 기울기 대비 = hybrid 고유 항**) | D {44, 108} × bs {8, 48} × ctx 3점 | 약 2.5 + 0.3 + 1.5×2 |
| **T2** CM-2 `T_span(N_tok, 108−D)` | prefill(단일·무경합) 시간의 L·SM 의존 | L ∈ {1k, 4k, 8.8k, 16k, 32k, 64k} × prefill SM {44, 64, 74, 108} | L 6점, SM 108 (**선형 대 초선형 대비**) | L 4점 × SM 2점 | 약 1 + 0.3 + 0.5×2 |
| **T3** CM-4 `ρ` 동거 간섭 | 고정 decode 배치와 prefill 1개를 동거시킨 step 연장(클럭 공변 기록, 총효과로만) | D {24, 44} × bs {8, 32} × L {8k, 32k} | 없음 | 없음(후속) | 약 1 |
| **T4** CM-5 메모리·admission | 부팅 로그 풀 크기, cap {48, C\_mem}에서 동시 수 채우기, 거절 사유(I-5: KV / cap≡슬롯), L ∈ {4k, 8.8k, 16k, 64k} | 위 전부 | 부팅 로그 + 채우기 L 2점 (**s=0, k 8배 대비**) | 부팅 로그 + 채우기 L 2점 | 약 1 + 0.3 + 0.5×2 |
| **T5** CM-6 대기·TTFT | 측정 없음. T1–T4로 조립하고 검증 단계에서 대조 | — | — | — | 0 |
| **V** 검증 (G) | 비용 모델 예측 대 실측 goodput: λ\*\_SLO | arXiv-Sum 자연 + shape C(arXiv-Sum × 출력 4k) × arm {fused\_tuned, agnostic(최선 static D)} × rung 5 × seed 4, 한 Job | 없음 | 선택: 같은 셀 TP2 일부(M3 재개의 축소판) | 약 10–14 (+TP2 선택 약 8) |

- **합계 [유도, 거침]**: 단일 GPU 항 측정 약 7 GPU-h + TP2 항 약 6 GPU-h(GPU 2장 × 약 3 h) + 검증 약 10–14 GPU-h, 즉 **약 25 GPU-h**다. TP2 검증을 넣으면 약 33 GPU-h다. rev0 Lean의 약 123 GPU-h 대비 약 1/4–1/5이다.
- Job 크기(사용자 메모 "GPU Job 적정 크기"): 항 측정은 단일 GPU Job 1개(T1–T4, 약 7 h), TP2는 2-GPU Job 1개(약 3 h, 두 GPU 동시 사용), 검증은 단일 GPU Job 1개(10–14 h, 16 h 캡 안)로 나눈다. arm은 각 Job 안에 함께 둔다.
- mem-fraction은 0.6이다(결정 13). long-ctx 상한은 64k다(커널 한계 약 94.8k 아래에서 고른 격자 선택). TP2에서는 in\_proj 폭이 rank마다 절반이므로 이 경계가 오를 것으로 본다 [유도, 미확인]. T2의 TP2 점에서 96k 1점을 시도해 확인할 수 있다(서버가 죽을 위험이 있으므로 Job의 마지막에 둔다).

## 3. (H) 답의 형태 — hybrid 고유 항 대비 (goodput 비교 아님)
| 항 | Nano에서 기대하는 형태 [가설·유도] | Llama 대비 |
|---|---|---|
| 요청당 메모리 s + k·L | s = 136.9 MiB, k = 16 KiB/토큰 [측정, 부팅], L\* ≈ 8.8k | s = 0, k = 128 KiB |
| decode ctx 기울기 ∂T\_dec/∂ctx | 작다(attention 4/56층) | 크다(32/32층) |
| prefill L 의존 | 약 선형(chunked 기준 124k까지 [기술값]) | 64k를 넘으면 초선형 [기술값] |
| cap 결합 | cap ≡ 슬롯(SGLang 정적 풀, 설정으로 공시) | cap은 admission만 정한다 |
| decode SM 민감도 | 존재(C2 CONFIRMED scoped) | T1의 D 축에서 대비하지 않음(Llama는 PD-mux 없이 D=108만) |

- 위 표의 Llama 칸만 측정하면 (H)의 근거가 된다. 모델 간 goodput 이중차분은 하지 않는다(rev0 Q-B 폐기).
- 금지 문안(승계): "hybrid는 KV가 작아 배치 여유가 크다"(L≈1–1.5k에서 REFUTED; 긴 L에서의 방향은 T4 측정 전 가설) · "~52k 경계" · "PD-mux는 64k를 넘을 수 없다" · "OOM 때문".

## 4. 선행 조건
1. I-5 엔진 동기화 + int64 conv 패치(b안, 선택)를 **한 번의 digest 변경으로** 묶고, GPU correctness gate를 1회 돈다.
2. 측정 하네스: 고정 배치 decode·prefill 마이크로 측정. N2 PROFILE(3-a) 하네스를 재사용할 수 있는지 확인한다. 그 데이터는 인용 금지이고 경량 확인만 거쳤으므로 **재측정**한다.
3. 합성 생성기 CPU 검증(감사 4번), Llama에서 PD-mux 부팅 확인(검증 단계에는 불필요, 대비 측정은 fused만).
4. fused\_tuned config: G0 튜닝을 cap = C\_mem(Nano, 0.6)에서 V보다 먼저 한다(명세 R8-2 개정 필요, 사용자 결정 O-5 승계).

## 5. 감사로 넘길 쟁점
- 마이크로 측정 항을 조립한 예측이 서빙 goodput을 검증할 수 있는가. 검증 실패 시 판정 규칙을 사전 등록해야 한다(예: 예측 λ\* 대비 실측의 허용 오차).
- Llama 대비 항이 교락(모델 크기, 층 수, RoPE, 토크나이저)을 넘어 "hybrid 고유"라고 말할 수 있는 범위.
- T3 ρ는 전력 한도와 교락되므로(CM-4) 총효과로만 쓴다.
