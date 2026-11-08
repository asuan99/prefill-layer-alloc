# 검토 — analytic-v1 해석적 비용 수식의 선행연구 대조와 엄밀성 (2026-11-02)

> ★결과 감사 r1(2026-11-08) GO-with-edits — 판정서 `reports/audit/2026-11-08_unaudited_r1/AUDIT_cap_mem_memo_and_formula_review_r1_2026-11-08.md`, 수정 전 인용 금지 목록 적용. 본문은 고치지 않았다.

> **지위**: 검토 문서(venue-strategist). 정본이 아니다. `ANALYTIC_COST_FORMULAS.md`(analytic-v1)와 `analytic_terms.py`는 **고치지 않았다**. 수정이 필요하면 정본 변경 규칙(코드·문서 동시 수정, v2, §9 이력, `render_formula_tables.py`)을 따른다.
> **GPU 0 · 새 측정 0.** 수치는 전부 `analytic_terms.py`로 다시 계산했다(손계산 인용 없음).
> **출처 규율**: 아래 1차 출처는 전부 PDF를 직접 내려받아 `pdftotext`로 본문을 읽었다(검색 스니펫은 쓰지 않음). 절·표·식 번호는 그 판본 기준이다. 판본이 바뀌면 번호가 달라질 수 있다.
> 질문(사용자): "Attention·Mamba의 FLOPs·메모리는 기존 연구에 이미 정리돼 있고, BW로 포화 정도를 두는 것만 우리 것 아니냐."
> **짧은 답**: 대체로 맞다. 그리고 **BW 하한의 분해(a₀ + b₀·b + c₀·b·c)나 c\*조차 새것이 아니다.** 층별 FLOPs·상주 메모리는 Marconi·SSD·FA2·Kaplan/Chinchilla에, decode 바이트 분해와 state↔KV 교차 문맥(c\* 논리)은 DistServe·Pope·ReplaySSM(2026-10-09, analytic-v1보다 13일 앞섬)·Priming에 이미 있다. 우리 몫은 (i) Nemotron-H/SGLang 구현 사실에 맞춘 **정확한 일반화와 실측 대조**, (ii) **PD-mux SM 분할 층**(u(D)·κ·ρ·두 벽·고정점)을 이 사전 위에 얹은 것이다. (ii)의 구성요소도 Bullet SRM(N_d = 30)·MuxWise contention guard와 겹친다.

## 0. 열어 본 1차 출처

| 약칭 | 문헌 | 본 판본 · 위치 |
|---|---|---|
| Marconi | Pan et al., "Marconi: Prefix Caching for the Era of Hybrid LLMs", MLSys 2025 | arXiv 2411.19379**v3**(2025-04-10), 부록 A.1, **Table 1**, Table 2. https://arxiv.org/pdf/2411.19379 (MLSys camera-ready와 같은지는 따로 대조하지 않음) |
| ReplaySSM | Liou, Nover, Naik, Skliar, Shovkun, Shahaf, Dao, "ReplaySSM: Cache SSM Inputs, Not State" | arXiv 2610.13778v1(**2026-10-09**), §3.1, **부록 C.1(식 12), C.2, C.3(식 13)**. https://arxiv.org/pdf/2610.13778 — **이 이름의 논문은 실재한다**(모델 지식 컷오프 이후 논문이라 이 검토에서 처음 확인했다) |
| SSD | Dao & Gu, "Transformers are SSMs" (ICML 2024) | arXiv 2405.21060, §6 "Hardware-efficient algorithm" — Center Blocks / Low-Rank Right·Center·Left Factors / Total Cost, 복잡도 표(Attention·SSM·SSD). https://arxiv.org/pdf/2405.21060 |
| Nemotron-H | NVIDIA, "Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models" | arXiv 2504.03624, §2.1, Table 1. https://arxiv.org/pdf/2504.03624 |
| Kaplan | Kaplan et al., "Scaling Laws for Neural Language Models" | arXiv 2001.08361, 식 (2.1)(2.2), **Table 1**. https://arxiv.org/pdf/2001.08361 |
| Chinchilla | Hoffmann et al., "Training Compute-Optimal LLMs" | arXiv 2203.15556, **부록 F**(FLOPs 계산). https://arxiv.org/pdf/2203.15556 |
| FA2 | Dao, "FlashAttention-2" | arXiv 2307.08691, §3.1 "Causal masking", §4.1 벤치 설정(FLOPs 식), §4.2. https://arxiv.org/pdf/2307.08691 |
| Pope | Pope et al., "Efficiently Scaling Transformer Inference" (MLSys 2023) | arXiv 2211.05102, §2 "Memory costs / Compute costs". https://arxiv.org/pdf/2211.05102 |
| Roofline survey | Yuan et al., "LLM Inference Unveiled: Survey and Roofline Model Insights" | arXiv 2402.16363, §2.2, **Table 1**(Llama-2-7b, A6000). https://arxiv.org/pdf/2402.16363 |
| DistServe | Zhong et al., OSDI 2024 | arXiv 2401.09670, §3.1(M/D/1), **부록 A.2–A.3**(T_Prefill, T_Decoding). https://arxiv.org/pdf/2401.09670 |
| Jenga | Zhang et al., "Jenga: Effective Memory Management for Serving LLM with Heterogeneity" (vLLM hybrid allocator 계열) | arXiv 2503.18292, §1·§2(Mamba 고정 크기 state vs 토큰별 KV, LCM 페이지). https://arxiv.org/pdf/2503.18292 |
| Priming | Chattopadhyay et al., "Priming: Hybrid State Space Models From Pre-trained Transformers" | arXiv 2605.08301, 추론 절 "A model of decode throughput", **식 (21)–(25)**. https://arxiv.org/pdf/2605.08301 |
| Bullet | ASPLOS '26 camera-ready | 로컬 `~/Experiments/KISTI/Papers/Bullet.pdf`, §3.2.2 SRM 식 1, Fig. 8(N_d = 30, A100) |
| MuxWise | ASPLOS '26 camera-ready | 로컬 `~/Experiments/KISTI/Papers/Muxwise.pdf`, §3.3.2 Table 2·식 1–2, contention guard, Fig. 11 |
| SGLang 커널 | 로컬 dev tree `~/Experiments/KISTI/sglang_engine_dev/python/sglang/srt/` | `configs/mamba_utils.py`(state dtype·shape), `layers/attention/mamba/ops/mamba_ssm.py`(selective_state_update), `ssd_chunk_scan.py`, `ssd_bmm.py`, `ssd_combined.py` |

**보지 못한 것**: 검색에는 떴지만 본문을 열지 않은 것(SketchSSM 2609.33051, DUET 2603.15530, HYDRA 2608.19395, Asymmetric paging 2605.22416, vLLM ReplaySSM RFC/PR, Nemotron Nano 2 기술보고서)은 근거로 쓰지 않았다. 2507.12442(SSM·hybrid long-context 특성화)는 내려받았지만 검토하지 않았다. **related work 절을 쓰기 전에 이것들을 열어야 한다**. 특히 SketchSSM은 같은 "state traffic" 축이다.

## 1. 식별 대조표

표기: 우리 기호를 쓴다. Marconi의 D는 d_model, N은 state 차원이다. "일치"는 같은 가정을 두면 같은 식이라는 뜻이다.

| # | 우리 식 (analytic-v1) | 기존 문헌의 동치 식 (출처) | 판정 | 어느 쪽이 맞나 · 근거 |
|---|---|---|---|---|
| 1 | W_A = d·d_h(n_q+2n_kv) + n_q d_h d | Kaplan Table 1: QKV n_layer·d_model·3d_attn + Project d_attn·d_model(MHA). Marconi Table 1의 8LD²(= 2·4D²·L, MHA, n_q d_h = D) | **일반화**(GQA·n_q d_h ≠ d) | 우리 식이 맞다. Nano는 n_q d_h = 5,120 ≠ d = 4,480이고 n_kv = 8이라 MHA 식은 틀린다. 부팅 배너 파라미터 수와 일치(§6) |
| 2 | 투영·MLP FLOPs = 2·W (dense matmul = 2×파라미터×토큰) | Kaplan 식 (2.2) "2N … multiply-accumulate", Pope §2 "2N matmul FLOPs per token", 서베이 각주 "MAC = 2 ops" | **일치**(교과서 규약) | — |
| 3 | **F_A,pre = 2·n_q d_h L²**(causal) | FA2 §4.1: "4·seqlen²·head_dim·heads, **causal이면 2로 나눔**" → 2 n_q d_h L². Kaplan Table 1 "Attention: Mask 2 n_layer n_ctx d_attn"(토큰당)을 L개 토큰에 합하면 같은 값. **Marconi Table 1: 4L²D**(causal 절약 없음). 서베이 Table 1: prefill qk matmul 34G = 2·2048²·4096(전체 행렬)이라 qk+sv = 4L²d. Chinchilla 부록 F: logits 2·L²·d_attn + reduction 2·L²·d_attn = 4(비causal). FA2 §4.2: Megatron 관례(12·n_layer·d·L²)는 causal을 반으로 나누지 않는다고 명시 | **계수 차이: 2 vs 4** | **둘 다 "규약"이고 어느 쪽도 오류는 아니다.** 우리 2는 "causal 커널이 실제로 계산하는 원소 수"(FA2 §3.1: causal 블록 건너뛰기, 블록 약 절반)에 맞고, Marconi·서베이·Chinchilla·Megatron의 4는 "밀집 행렬 기준"이다. FlashInfer/FA 계열 causal 커널로 서빙하는 우리 기판에는 2가 물리적으로 맞다. 정확히는 2 n_q d_h L(L+1)이고, 대각 블록은 다 계산하니 블록 크기만큼 약간 더 든다(L ≫ 블록이면 무시 가능). **단 이 선택이 L_q와 이차항 비중을 2배 바꾼다**: 계수 4를 쓰면 L_q 382k → 191k, 64k 이차항 비중 0.17 → 0.34. 논문에는 규약을 명시하고 FA2 §4.1을 인용해야 한다. 정본 §9의 "Marconi 대비 차이" 목록에 이 항목이 **빠져 있다**(형식은 Marconi를 따랐다고 써 놓고 계수를 바꿨다) |
| 4 | F_A,dec(c) = 4·n_q d_h c | 서베이 Table 1 decode qk matmul 17M = 2·2048·4096, sv 동일 → 합 4cd. Chinchilla 부록 F의 토큰 1개 버전. DistServe A.3 "2sl FLOPs"(head당, **QKᵀ만 센 것으로 보인다**) | **일치**(서베이·Chinchilla) | 우리 4가 맞다(QKᵀ와 AV 두 matmul). DistServe의 2sl은 AV를 빼고 셌거나 MAC 단위다. 그러나 DistServe는 이 항을 메모리 바운드로 모델링하므로 결과에는 영향이 없다 |
| 5 | k_A = 2·n_kv d_h β | Marconi 부록 A.1: "2(K,V)·L·D·2 bytes" = 4LD(MHA, fp16). ReplaySSM C.2: "2 KV heads × 2 × 128 bytes(FP8)". Pope §2: KV cache | **일반화**(GQA) | 우리 식이 맞다(GQA는 n_kv로 공유) |
| 6 | W_M = d(2d_in + 2GN + n_h) + d_in d | Marconi Table 1 SSM 12LD²(= in_proj D→4D와 out_proj 2D→D, expand 2를 가정한 것으로 보인다. B·C·dt 투영은 없음). Nemotron-H §2.1은 구조 사실만 준다(Mamba-2, 8 groups, expand 2, conv 4, head_dim 64 — 8B/56B 기준. Nano-9B-v2는 config상 p = 80) | **일반화** | 우리 식이 맞다(Mamba2 in_proj = z, x, B, C, dt). Nano에서 Marconi 12d²는 2.408e8, 우리 2W_M은 2.947e8(B·C·dt가 in_proj의 9.6%, d_in = 2.29d) |
| 7 | conv FLOPs 2K·d_conv / 토큰 | Marconi Table 1에는 conv 항이 없다(state 크기만 부록 A.1에서 언급) | **우리가 추가** | 맞다(depthwise conv, 채널마다 K MAC). bias·SiLU는 빠졌지만 무시 가능하다(α의 0.02%) |
| 8 | **F_S,pre = 2GQN + 2n_hQp + 4n_hpN**(chunk 간 scan 생략) | **SSD §6**: Center Blocks CᵀB = BMM(T/Q, Q, Q, N), X 곱 = BMM(T/Q, Q, P, ·), Right Factors BMM(T/Q, N, P, Q), Left Factors BMM(T/Q, Q, P, N), Center Factors scan "TNP/Q, negligible". BMM(B,M,N,K) = O(BMNK). Marconi Table 1 SSM "16LDN + 10L"(chunk 항 없음, Mamba-1식 순환 계수로 보인다) | **SSD와 일치**(head·group 일반화), **Marconi와 불일치** | 우리 식이 SSD 알고리즘을 정확히 옮긴 것이다. 토큰당 MAC = G·Q·N + n_h·Q·p + n_h·p·N + n_h·p·N, ×2. CᵀB는 group마다 한 번이다(ssd_bmm이 group 단위, `ssd_combined.py:136`). 참고로 SSD §6 본문은 X 곱을 "BMM(T/Q, Q, P, N)"로 적었는데, 차원상 마지막 축은 Q여야 한다(본문은 N = P = Q를 두고 총합을 내므로 결론에는 영향이 없다). **Marconi 16LDN은 Mamba2 SSD prefill에 맞지 않는다**: chunk 내 이차 항(2n_hQp가 F_S,pre의 32%)이 없다. Nano에서 Marconi식 16dN = 9.18e6, 우리 F_S,pre = 8.13e6 |
| 9 | F_S,dec = 5·d_in·N | **ReplaySSM 부록 C.1**: head당 decay·outer product·합·readout = d_v d_k + d_v d_k + d_v d_k + (2d_v d_k − d_v) = **5 d_v d_k − d_v**(d_v = p, d_k = N). SSD 복잡도 표: inference O(N²)/head | **일치**(ReplaySSM은 −d_v까지 정확) | 같은 식이다. 우리는 −d_in과 D skip·dt softplus·gating을 뺐다(합쳐 F_S,dec의 <1%). **ReplaySSM이 analytic-v1(2026-10-22)보다 13일 먼저 공개됐으므로 이 식은 인용 대상이지 우리 유도가 아니다** |
| 10 | s_M = n_h p N β_s + d_conv(K−1)β, β_s = 4 | Marconi 부록 A.1: SSM state = "**D·N·2 bytes**"(fp16), conv state = in_ch·kernel·2 bytes(7B 모델 state의 6.1%, Table 1에서는 생략). ReplaySSM C.2: "128 × 64 × 128 × 4 bytes = 4 MiB"(n_h·p·N·fp32), conv 없음. Jenga §1–2: Mamba = 요청당 고정 크기 state, KV = 토큰별 | **일반화 + Marconi와 불일치** | **우리 식이 맞다**(SGLang `mamba_utils.py`: temporal shape (n_h, p, N), dtype 기본 fp32 / conv shape (d_conv, K−1), dtype 기본 bf16. 부팅 배너 143.5 MB와 일치). Marconi Table 1의 2DN은 우리 기판에 그대로 쓰면 **층당 4.6배 과소**다(Nano: 1.15 MB vs 5.32 MB). 원인은 (i) d 대신 d_in(= n_h p)이어야 하고 (ii) fp32다. Marconi 표 안에서도 투영 FLOPs(12LD²)는 expand 2를 가정하면서 state는 D·N으로 적어 **내부적으로 어긋나 보인다**. 이 판단은 Marconi가 쓴 7B 모델 config를 확인하지 않은 추정이므로, 인용할 때는 "Marconi 표의 D·N은 expand를 흡수한 표기일 수 있음"을 병기한다. conv state(K−1, bf16)는 Nano에서 s_M의 1.4%다 |
| 11 | W_F = 2d·d_ff(비게이트 ReLU²), FLOPs 2W_F | Kaplan Table 1 Feedforward 2·d_model·d_ff(비게이트). Marconi 16LD²(d_ff = 4D, 비게이트). Nemotron-H §2.1: "squared ReLU" | **일치** | 비게이트라는 사실은 Nemotron-H 보고서가 아니라 config(gate_proj 없음)와 파라미터 총수 일치로 확인된다 |
| 12 | LM head 2dV / 샘플 토큰, prefill에서 1회 | Kaplan Table 1 De-embed 2·d_model·n_vocab(토큰당). Chinchilla 부록 F Final Logits 2·L·d·V(전 위치) | **일치 + 구현 사실** | "prefill은 마지막 위치만"은 SGLang 기본 동작에 대한 서술이다(input logprob 요청 시 전 위치로 바뀐다). 맞지만 조건을 적어야 한다 |
| 13 | F_pre(L) = αL + γL² + 2dV | MuxWise Table 2: Prefill w/o cache O(Ld² + L²d), 식 1 θ1Σn² + θ3Σn + θ4. DistServe A.2: C1(4th² + 2thm) + C2·3ht2/b + C3. Kaplan 식 (2.2) | **형식은 기존과 같다**(1차+2차 다항) | 우리 몫은 계수를 층 조성으로 닫힌 식으로 쓴 것뿐이다(γ ∝ n_A, α에 SSD 항). L_q = α/γ는 Kaplan의 "d_model > n_ctx/12면 문맥항은 작다"와 같은 종류의 교차 길이다. γ의 계수 2/4 규약에 따라 L_q가 2배 달라진다(#3) |
| 14 | B_dec = β(가중치) + b·2s + b·c·k | **Pope §2** "Memory costs": weights와 KV cache를 매 forward마다 HBM에서 한 번씩 읽는 "memory time". **DistServe A.3**: T_Decoding = C4(4h² + 2hm) + C5·3h·Σl(가중치 + KV). **ReplaySSM C.1–C.2**: SSM state "loads and stores" = 2b·d_v d_k/head, KV는 load만. **Priming 식 (22)–(23)**: T = n_A t_attn(b, l) + n t_mlp(b) + n_M t_ssm(b), "SSM state 비용은 b에 비례하고 l과 무관" | **기존 항들의 합** | 세 항이 각각 문헌에 있다. 우리는 이것을 한 식에 바이트 단위로 묶었을 뿐이다. 빠진 항은 활성값(b·d·β/층), KV 쓰기 b·k, logits b·V·4, 정규화 가중치로, b = 32·c = 8k에서 합쳐 <0.2%다 |
| 15 | **T_dec ≥ a₀ + b₀b + c₀bc**, a₀ = W·β/BW, b₀ = 2s/BW, c₀ = k/BW | DistServe A.3(C4, C5·3h·t — 측정 보정 상수), MuxWise 식 2: T_Decode = θ1Σr_i + θ2·bs + θ3. Bullet SRM 식 1: T' = flop·[min(flop/mem·D_p, C_p)]⁻¹ | **형식은 MuxWise 식 2와 같다** | **새 것은 계수를 BW에서 닫힌 식으로 준 것뿐**이고, 그것조차 Pope의 "memory time"에 hybrid state 항을 더한 것이다. 엄밀성 문제는 §2-B를 보라(하한이라는 주장의 조건) |
| 16 | L\* = s/k | Marconi Table 1 캡션: "SSM state = **N/2 = 64배** 단일 토큰 KV"(층당). ReplaySSM C.2: "one SSM state equals the KV cache for 4 MiB/0.5 KiB = **8192 tokens**". Priming: "state = at most 2K tokens of KV". Jenga: 고정 크기 vs 토큰별 | **기존 개념** | 같은 비율이다. 우리 것은 모델 전체(n_M·s_M vs n_A·k_A)로 쓴 것일 뿐이다. **새 개념으로 쓰면 안 된다** |
| 17 | c\* = 2s/k | **ReplaySSM C.2 "Crossover"**: "SSM layers move 40×8 = 320 MiB … attention 8×0.5N KiB … equal at N = 81,920, the layer ratio of five times 8K tokens **times two for the load and the store**. Batching does not move the crossover" | **동일 논리** | ReplaySSM이 먼저 냈다(2026-10-09). 우리 c\*는 그 식에 Nano 상수를 넣은 것이다. **"b와 무관"이라는 관찰까지 같다**. 새것으로 주장 불가 |
| 18 | §8 u(D) 무릎 D ≈ 30, κ ≈ 1 | **Bullet §3.2.2**: C_p = C_peak·N_p/N(연산 SM 선형), D_p = D_peak·min(1, N_p/N_d), "**A100 N_d = 30**"(memcpy 커널, Fig. 8a) | **기존 결과의 재관측** | PROPOSAL_dstar §2의 "무릎 D ≈ 30, 무릎 위 기울기 0.21배"와 "prefill은 P에 반비례"(κ 1.011–1.021)는 Bullet SRM의 두 가정을 hybrid decode/prefill에서 **독립 재관측**한 것이다. 새 발견으로 쓰면 안 된다. "Bullet SRM의 N_d = 30이 hybrid decode step 전체에도 성립한다"는 정도가 정직한 서술이다(그것도 a(D)에 고정비가 섞여 있다는 가정 A2 아래에서만) |
| 19 | §8 ρ(동거 slowdown) 1.13–1.19 | **MuxWise contention guard**: GreenContext는 SM만 나누고 대역폭은 못 나눈다. decode slowdown 0–30%(Fig. 11), 최대 배수로 가드. Bullet §3.2.3: memcpy+GEMM 최악 간섭 | **기존 개념** | ρ를 곱하는 형식은 MuxWise 가드와 같다. 우리 값은 측정 범위다 |
| 20 | D\* 서빙 모형(Little 고정점, M/G/1 W, 두 벽) | DistServe §3.1: prefill 인스턴스를 **M/D/1**로 닫힌 식으로 둔 TTFT 모형. MuxWise §3.4: decode SLO 우선, 최악 추정으로 best-fit decode SM, 나머지는 prefill | **부분적으로 새것** | 대기열로 SLO 용량을 푸는 형식은 DistServe에 있고, "decode에 SLO를 지키는 최소 SM"(아래 벽)은 MuxWise 디스패처·Bullet Alg. 1(ReduceDecodeSM)의 정적 버전이다. 새로운 조합은 **비sticky lockstep 루프에서 아래 벽(ITL·cap)과 위 벽(TTFT)이 닫는 창과 그 위의 cap 고원**이라는 서술이다. 다만 이것은 사후 재현(PROPOSAL §0 공시)이고 예측 검증(EVAL)은 미제출이다 |

## 2. 엄밀성 문제 목록

### A. 계수·규약

1. **F_A,pre 계수 2(causal)와 Marconi 4의 차이가 문서에 공시돼 있지 않다**(대조표 #3). 정본 머리말은 "형식은 Marconi Table 1을 따르고"라고 쓰고, §9 차이 목록에는 GQA·head·chunk·fp32·conv·비게이트·decode 트래픽만 있다. 2는 맞는 선택이지만 **규약 선언 + FA2 §4.1 인용 + "Megatron/Marconi 관례로는 4, 이 경우 L_q = 191k"**를 v2에서 적어야 한다. §7의 64k 이차항 비중(0.17/0.63/1.21/1.99)과 P4 예측 서술은 계수에 비례하므로 같은 각주가 필요하다.
2. **F_S,pre의 chunk 내 항은 "밀집" 기준이다.** SGLang `ssd_chunk_scan.py`는 IS_CAUSAL = True로 실행돼(launch 인자) K 블록을 대각까지만 돌므로, 2n_hQp 항은 실제로 대략 (Q + BLOCK_K)/(2Q) 배만 계산된다. 반대로 CᵀB(`_bmm_chunk_fwd`, `ssd_combined.py:136`)는 causal = False로 전부 계산된다. Nano에서 차이는 토큰·층당 ≈ 1.31e6 FLOP, α의 0.23%다. **무시할 수 있지만**, #1과 같은 규약(causal 절약을 센다)을 attention에는 쓰고 SSD에는 안 써서 **규약이 일관되지 않다**. 각주 한 줄로 충분하다.
3. **chunk 간 scan 생략**: 2n_h pN/Q = 20,480 FLOP/토큰·층(F_S,pre의 0.25%). SSD §6이 "negligible"로 명시한 것과 같다. 정당하다.
4. **F_A,dec는 "4·n_q d_h c"이고 n_kv는 들어가지 않는다**: GQA는 FLOPs를 줄이지 않고 KV 바이트만 줄인다. 맞다.
5. **빠진 비matmul 항**: softmax(Chinchilla 부록 F: 3·heads·L², causal이면 약 절반이라 γ의 약 0.6%), RMSNorm·gated RMSNorm, dt softplus, exp(dt·A), D skip, ReLU². 전부 1% 미만이다. **RoPE는 Nemotron-H에 없다**(§2.1 "We do not use any position embeddings"). 따라서 생략이 맞다(다른 hybrid 모델로 이식할 때는 확인 필요). 비matmul 항은 FLOP 비중은 작아도 **시간 비중은 클 수 있다**(FA2 §3.1: A100 비matmul fp32 19.5 TFLOP/s vs matmul 312). §8 η가 흡수할 항목임을 적어 두는 게 정직하다.

### B. decode 메모리 바운드 하한(T_dec ≥ …)의 조건

6. **"≥"는 max가 아니라 합의 한쪽만 본다.** roofline 하한은 Σ_k max(F_k/π, B_k/BW)다(서베이 §2.2, Bullet SRM 식 1). 문서의 식은 메모리 항만 쓴다. Nano·b = 32·c = 8k에서는 전체 강도 18.6 FLOP/B로 바이트가 지배하니 결론은 서 있다. 하지만 (i) 가중치 GEMM의 강도는 약 b FLOP/B(bf16)라서 **b ≳ 150(A100 ridge)이면 연산 바운드**로 넘어가고, 이때 식은 하한이긴 해도 느슨해진다. (ii) SSM state 갱신은 강도 5/8 ≈ 0.63 FLOP/B(fp32, ReplaySSM 식 12)라서 항상 메모리 바운드다. 이 조건(b ≤ 약 150, cap 48이면 충족)을 식 옆에 써야 한다.
7. **b₀ = 2s/BW는 물리 하한이 아니라 "conventional kernel" 하한이다.** SGLang `selective_state_update`는 head 블록마다 state를 한 번 로드(`mamba_ssm.py:179`, fp32로 올려 계산)하고 한 번 저장(`:297`, DISABLE_STATE_UPDATE가 아닐 때)한다. 따라서 **현재 기판에서 2s는 맞다**(conv는 `causal_conv1d_update`가 창 K−1을 읽고 다시 쓴다. s의 1.4%). 그러나 ReplaySSM은 쓰기를 주기적으로 미뤄 state 트래픽을 (L+1)/(2L) ≈ 0.53배로 줄인다(C.3 식 13, L = 16에서 0.58배). 즉 **"b·2s는 하한"이라는 표현은 "eager read+write 커널 기준"으로 한정해야 한다.** 정보량 하한은 s(읽기 전용, ReplaySSM 결론 절 "read the state once and write nothing" 계열)이고, 저정밀 state면 더 내려간다. 이 한정은 **c\* = 2s/k에도 그대로 전파된다**: ReplaySSM 커널이면 c\* ≈ 1.06·L\* ≈ 9.3k이지 17.5k가 아니다.
8. **가중치를 step당 1회 읽는다는 가정**: A100 L2는 40 MB이고 층당 가중치(Mamba 295 MB, MLP 281 MB, attention 110 MB, bf16)가 이보다 훨씬 커서 step 간 재사용은 없다. b ≤ GEMM M 타일(대개 64–128)이면 한 번에 흘러간다. **b가 M 타일을 넘으면 타일마다 가중치를 다시 읽을 수 있다**(L2가 일부 흡수). cap 48에서는 1회 가정이 그럴듯하지만 커널을 확인한 것은 아니다. **PD-mux 동거 중에는 prefill forward도 같은 가중치를 따로 읽는다.** 그러니 동거 iteration의 HBM 바이트는 a₀ 하나로 끝나지 않는다. 이것은 ρ(§8)에 들어가야 하는 항이고, 기전 가설 목록에 "가중치 이중 읽기"를 명시할 가치가 있다.
9. **BW = 2.039 TB/s는 사양 피크다.** 달성 가능한 스트리밍 대역폭은 그보다 낮다(이 검토에서 측정하지 않았다). 사양값을 쓰면 하한은 유효하지만 느슨하고, 실측 a(108) = 13.06 ms 대비 u = 0.62에는 고정비(launch·sampling)와 달성 BW 손실이 섞여 있다(PROPOSAL 가정 A2와 같은 말). u를 "대역폭 효율"로 부르지 말 것.
10. **cudagraph 패딩**: b가 캡처 bs로 올림되면 패딩 행이 state를 읽고 쓰는지가 b₀ 항을 바꾼다. 커널은 `state_batch_idx == pad_slot_id(−1)`이면 load·store를 마스크한다(`mamba_ssm.py:178`). 그러나 **SGLang cudagraph replay가 패딩 행에 −1을 넣는지 더미 슬롯을 넣는지는 이 검토에서 확인하지 않았다**(§8 표의 I-6 항목). 가중치 항은 패딩과 무관하다.
11. **c는 "모든 요청이 같은 문맥"이라는 가정이다.** 실제로는 Σ_i c_i이고 식은 b·c → Σc_i로 정확히 일반화된다(DistServe A.3의 t = Σl, MuxWise θ1Σr_i). 문서에 한 줄 적으면 된다.

### C. 메모리·구현 사실

12. **s_M의 정밀도는 맞다**(SGLang 기본: temporal fp32, conv bf16 — `mamba_utils.py` `mamba2_state_dtype`, `SGLANG_MAMBA_CONV_DTYPE`/`SGLANG_MAMBA_SSM_DTYPE`·config `mamba_ssm_dtype`로 바뀐다). 식의 β_s를 "기본값 fp32, 설정에 따라 바뀜"으로 한정하면 된다. conv state shape는 (d_conv, K−1)로 맞다.
13. **GQA KV 공유는 k_A에 반영돼 있다**(n_kv). TP에서는 n_kv < TP일 때 KV head 복제가 생겨 k/N이 성립하지 않는다(§8 TP 행 "s/N, k/N"은 n_kv ≥ N일 때만 맞다. Nano n_kv = 8). 또 TP에서 Mamba group 복제(`extra_groups_for_head_shards`)가 생기면 conv state가 정확히 1/N이 되지 않는다. 지금은 단일 GPU라 영향이 없고, J-C를 확장할 때 각주가 필요하다.
14. **prefix 재사용·radix의 state 복사 트래픽은 모형 밖이다**(Marconi가 다루는 축). 문서가 "접두 재사용 없음"으로 이미 한정했다. 정당하다.

### D. 오류는 없음

- §6·§7의 대입값(W_A·W_M·W_F, s, k, L\*, F_S,pre, F_S,dec, α, γ, L_q, a₀, b₀, c₀, c\*)은 `analytic_terms.py`로 다시 계산해 문서와 전부 일치했다. 산술 오류는 없다.
- §6 실측 대조 2건(배너 일치, 0.111 ms/토큰 → 141 TFLOP/s ≈ 45%)도 재계산값이 일치한다.

## 3. 기여 구분

### (a) 문헌에 이미 있다(그대로 또는 일반화) — 반드시 인용

- 밀집 matmul 2N 규약, 투영·MLP·LM head FLOPs(Kaplan Table 1, Pope §2, Chinchilla 부록 F).
- causal attention FLOPs 2·n_q d_h L²(FA2 §4.1), decode 4·n_q d_h c(서베이 Table 1, Chinchilla 부록 F).
- KV 바이트 2·n_kv·d_h·β(Marconi A.1의 GQA 일반화).
- Mamba2 SSD prefill의 chunk 항(SSD §6). head/group으로 정확히 옮긴 것은 우리 작업이지만 기계적인 일반화다.
- **decode state 갱신 5·d_in·N FLOPs와 read+write 2s 트래픽, 강도 5/(2β_s)**(ReplaySSM 부록 C.1, 2026-10-09).
- 층별 FLOPs·state 크기 표라는 형식 자체(Marconi 부록 A Table 1).
- state 대 KV 등가 토큰 수 = L\*(Marconi Table 1 캡션 N/2, ReplaySSM C.2 8192 토큰, Priming "2K tokens").
- **state↔KV decode 트래픽 교차 문맥 = c\*, 그것이 b와 무관하다는 점**(ReplaySSM C.2).
- decode 시간 = 가중치 + per-request + per-token-context 분해(Pope §2 memory time, DistServe A.3, MuxWise 식 2, Priming 식 (22)–(23)).
- SM 분할 시 연산 ∝ SM, 대역폭은 N_d ≈ 30에서 포화(Bullet SRM §3.2.2, A100).
- 동거 간섭을 최악 slowdown 배수로 두는 것(MuxWise contention guard, Bullet §3.2.3).
- 대기열로 SLO 용량을 푸는 형식(DistServe §3.1 M/D/1).

### (b) 우리가 새로 쓴 것(정직한 범위)

- **Nemotron-H·SGLang 구현 사실에 맞춘 정확한 합본**: GQA, n_q d_h ≠ d, Mamba2 in_proj의 B·C·dt, group 공유 CᵀB, conv state(K−1, bf16), fp32 temporal state, 비게이트 ReLU², 마지막 위치 LM head를 **한 표에 넣고, 부팅 배너(파라미터·s·k)로 확인한 것**. 선행 표(Marconi)를 이 기판에 그대로 쓰면 state가 4.6배 과소, SSD chunk 항(F_S,pre의 32%)이 빠진다는 **정정 자체가 작은 기여**다. 단 "Marconi가 틀렸다"고 쓰지 말고 "다른 모델·정밀도 가정"으로 쓴다(#10의 [미확인] 참조).
- **세 항(가중치 / per-request state / per-token KV)을 바이트 단위 닫힌 식으로 묶어 한 모델의 측정 floor 계수(a(D), b, c)와 대조한 것**. 각 항은 기존에 있지만, **같은 모델에서 해석적 a₀ 대비 측정 a(D)의 비 u(D)를 SM 수 축으로 낸 것**은 문헌에서 확인하지 못했다. Bullet은 memcpy 커널로 N_d를 재고, MuxWise 식 2는 SM 변수가 없다(우리 메모 §2.2 [미확인]). 다만 이것은 **새 법칙이 아니라 Bullet SRM의 hybrid decode-step 재관측**이다.
- **조성 변형(n_A/n_M 축) 위에서 k·c₀가 n_A에 정확히 비례하고 γ/α가 단조라는 산술 예측(P3·P4)을 사전 등록해 측정과 맞춰 보는 구조**. 식은 자명하지만 "사전 등록한 예측 → 서빙 측정"이라는 검증 절차가 기여다(측정 결과가 나와야 성립한다).
- **PD-mux D\* 모형의 조합**: 비sticky lockstep 루프에서 ρ·T_dec(D) ≤ S_ITL(아래 벽)과 κ·T2·108/P ≤ S_TTFT(위 벽)가 만드는 창, 그 위의 slot-cap 고원으로 W1의 D44를 설명한 것. **현재 지위는 사후 재현**이다(PROPOSAL §0). EVAL 사전 예측이 맞기 전에는 "모형이 D\*를 예측한다"고 쓸 수 없다.

### (c) 새롭다고 쓰면 안 되는 것

- "hybrid 층별 FLOPs·메모리 해석식"(Marconi·SSD·ReplaySSM에 있다).
- "SSM state는 길이와 무관한 고정 비용, KV는 선형"(Marconi·Jenga·Priming·ReplaySSM 공통).
- **L\*와 c\*의 개념**(Marconi 캡션, ReplaySSM C.2). c\*의 "×2(load+store)" 논리까지 ReplaySSM에 있다.
- **"T_dec = a + b·bs + c·bs·ctx" 함수형**(MuxWise 식 2, DistServe A.3의 형식). 우리는 계수에 물리적 하한값을 붙였을 뿐이다.
- **"decode 대역폭은 SM 약 30개에서 포화, prefill은 SM에 선형"**(Bullet SRM, A100 N_d = 30).
- "GreenContext는 대역폭을 나누지 못해 동거 slowdown이 생긴다"(MuxWise §3.3).
- "BW 포화로 둔 것이 우리 것"이라는 사용자 가설도 **절반만 맞다**. BW로 decode 하한을 두는 것은 Pope·DistServe·Bullet·ReplaySSM의 표준 도구다. 우리 것은 BW 하한과 **측정 floor의 SM 축 비(u(D))와 그것이 D\* 선택에 들어가는 경로**다.

## 4. v2에서 고칠 것(정본 변경 규칙으로 넘김 — 이 검토는 고치지 않음)

| 우선 | 위치 | 내용 |
|---|---|---|
| 필수 | 머리말 "출처", §9 | Marconi 대비 차이에 (i) **attention 계수 2(causal, FA2 §4.1) vs Marconi 4** (ii) **SSM state d_in·N·fp32 vs Marconi D·N·fp16** (iii) SSM FLOPs: SSD chunk 식 vs Marconi 16LDN 추가. 출처에 **ReplaySSM(C.1·C.2)·SSD §6·FA2·DistServe A.3·Pope §2** 추가 |
| 필수 | §3 F_S,dec, §5 B_dec·c\* | "5 d_in N, 2s, c\* = 2s/k는 ReplaySSM 부록 C.1–C.2와 동일"을 명시하고, 2s를 **"eager read+write 커널(SGLang selective_state_update) 기준"**으로 한정. ReplaySSM식 커널이면 b₀·c\*가 약 절반이 된다는 각주 |
| 필수 | §5 T_dec | "≥"의 조건: 메모리 항만 쓴 roofline 하한, b ≲ ridge(약 150), 사양 BW 사용, 가중치 step당 1회(b ≤ GEMM M 타일), 동거 시 prefill의 가중치 재읽기는 ρ로 넘김 |
| 권장 | §5·§7 | L_q·이차항 비중이 계수 규약에 비례한다는 각주(4를 쓰면 L_q 191k, 64k 비중 0.34) |
| 권장 | §3 F_S,pre | chunk_scan의 causal 블록 절약(α의 0.2%)을 세지 않았다는 각주 → attention과 규약 불일치 공시 |
| 권장 | §5 decode | b·c → Σc_i 일반화 한 줄 |
| 권장 | §8 TP 행 | n_kv < N이면 KV head 복제, Mamba extra group으로 s/N이 정확하지 않음 |
| 확인 | §8 cudagraph 행 | 패딩 행의 state_batch_idx가 PAD_SLOT_ID(−1)인지 SGLang replay 코드로 확인(b₀ 항이 캡처 bs로 올라가는지 결정) |

## 5. 논문 positioning 함의(짧게)

- 수식 절은 **"Background/Model" 절로 내리고 각 식에 출처를 단다**. 기여 목록에 "해석적 비용 모델"을 독립 항목으로 올리면 리뷰어가 Marconi·ReplaySSM·DistServe를 들어 바로 꺾는다. 특히 ReplaySSM은 같은 Mamba2·같은 Nemotron 계열·같은 "state traffic이 KV보다 크다"는 주장이고 Tri Dao가 공저자라 리뷰어가 알 가능성이 높다.
- 정직하게 쓸 수 있는 것: "선행 식들을 hybrid serving 기판(SGLang·fp32 state·GQA·SSD)에 맞게 정확히 합본하고, 해석적 하한 대비 측정 floor의 SM 축 거동을 PD-mux 분할 결정에 연결했다." 이 연결의 예측력은 **EVAL 사전 예측이 맞는지에 달려 있다**. 지금은 사후 재현뿐이다.
- related work에 반드시 넣을 것: Marconi, ReplaySSM, (열어서 확인 후) SketchSSM, Priming 추론 절, Jenga, DistServe 부록 A, Bullet SRM, MuxWise.
