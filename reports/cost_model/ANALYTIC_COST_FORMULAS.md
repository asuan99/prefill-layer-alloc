# Hybrid LLM 해석적 비용 수식 (고정본)

> **지위**: 사용자 결정 27(2026-10-22)로 고정한 **비용 모델의 수식 기반**이며 논문에 그대로 들어갈 수식이다. 측정이 아니라 닫힌 식이고, 모든 수치는 [유도]다.
> **버전**: `analytic-v1 (2026-10-22)`. 계산의 단일 출처는 `workspace/engine-port/scripts/costmodel/analytic_terms.py`이고, 이 문서는 그 식을 서술한다. 시험 `workspace/engine-port/tests/test_analytic_terms.py`가 이 문서의 수치를 재현한다.
> **변경 규칙**: 식을 고칠 때는 코드와 이 문서를 같은 커밋에서 함께 고치고 버전을 올린다(v1 → v2). 그 다음 `python3 workspace/engine-port/scripts/costmodel/render_formula_tables.py`로 축 지도 artifact의 §8 수식 표를 다시 쓰고 같은 URL로 재발행한다(표는 코드가 생성, 손으로 고치지 않음). 이미 쓴 식을 지우지 않고, 바뀐 이유와 날짜를 §9 이력에 남긴다. 비용 모델의 확장(PD-mux 항, 효율 η, 간섭 등)은 §8의 형태로 이 식 위에 덧붙인다.
> **출처**: 형식은 Marconi(Pan et al., MLSys 2025) 부록 A Table 1을 따르고, Nemotron-H 구조(GQA, Mamba2 SSD의 head·group, 비게이트 ReLU² MLP, SGLang의 fp32 SSM state)에 맞게 일반화했다. 모델 상수는 Nemotron-Nano-9B-v2-Base config(rev `dc0661c829b14e5b9246c05cfa89094a0875e052`).

## 1. 기호와 규약

| 기호 | 의미 | Nano-9B-v2 |
|---|---|---|
| d | hidden | 4,480 |
| L | prefill 길이(요청 하나) | — |
| c | decode 시점의 문맥 길이 | — |
| b | decode 배치 크기 | — |
| n_q, n_kv, d_h | query head, KV head, head 차원 | 40, 8, 128 |
| n_h, p, d_in = n_h·p | Mamba head, head 차원, 내부 차원 | 128, 80, 10,240 |
| G, N, K, Q | SSM group, state 차원, conv 커널, SSD chunk | 8, 128, 4, 128 |
| d_conv = d_in + 2GN | conv 채널 | 12,288 |
| d_ff | MLP 중간 차원 | 15,680 |
| V | 어휘 | 131,072 |
| n_A, n_M, n_F | attention / Mamba / MLP 층 수 | 4 / 27 / 25 |
| β, β_s | 원소 바이트: bf16 = 2 / SSM state = 4(fp32, SGLang `mamba2_state_dtype`) | 2 / 4 |

- FLOPs는 곱셈과 덧셈을 각각 1로 센다. 밀집 행렬곱은 2 × 파라미터 × 토큰이다.
- 출력층(LM head)은 샘플링하는 토큰마다 2·d·V이다. SGLang은 prefill의 마지막 위치만 샘플링하므로 prefill에서는 한 번만 든다. 임베딩 조회는 FLOPs 0으로 둔다.

## 2. Attention 층 (causal GQA)

- 파라미터: **W_A = d·d_h·(n_q + 2n_kv) + n_q·d_h·d**
- 투영 FLOPs / 토큰: **2·W_A**
- prefill 점수·값(요청 하나): **F_A,pre(L) = 2·n_q·d_h·L²** (causal 반삼각, QKᵀ와 AV 두 행렬곱)
- decode 점수·값 / 토큰: **F_A,dec(c) = 4·n_q·d_h·c**
- 상주 메모리(KV) / 토큰: **k_A = 2·n_kv·d_h·β**
- decode 트래픽 / step: 가중치 **2·W_A**(배치 공유) + KV 읽기 **b·c·k_A**

## 3. Mamba2 층 (SSD)

- 행렬 파라미터: **W_M = d·(2d_in + 2GN + n_h) + d_in·d** (in_proj의 z, x, B, C, dt와 out_proj)
- 투영 FLOPs / 토큰: **2·W_M**
- conv FLOPs / 토큰: **2·K·d_conv**
- prefill SSD / 토큰: **F_S,pre = 2·G·Q·N + 2·n_h·Q·p + 4·n_h·p·N**
  - chunk 안의 CBᵀ(2GQN)와 그것을 X에 적용(2n_hQp), chunk state BᵀX(2n_hpN), chunk 간 출력 Ch(2n_hpN). chunk 간 scan(O(n_h·p·N/Q))은 생략.
- decode 상태 갱신 / 토큰: **F_S,dec = 5·d_in·N** (h ← exp(dt·A)·h + dt·B·x 에 3pN, y = C·h 에 2pN, head마다). 문맥 c와 무관.
- 상주 메모리(요청당, 길이 무관): **s_M = n_h·p·N·β_s + d_conv·(K−1)·β**
- decode 트래픽 / step: 가중치 **2·W_M** + state 읽기·쓰기 **b·2·s_M**

## 4. MLP 층

- 파라미터: **W_F = 2·d·d_ff** (비게이트 ReLU²; 게이트형 SwiGLU이면 3·d·d_ff)
- FLOPs / 토큰: **2·W_F**
- 상주 메모리: 없음. decode 트래픽 / step: 가중치 **2·W_F**

## 5. 모델 전체

**요청당 상주 메모리**

- 고정 state: **s = n_M · s_M**
- KV / 토큰: **k = n_A · k_A**
- 두 항이 같아지는 길이: **L\* = s / k**

**Prefill FLOPs (요청 하나, 접두 재사용 없음)**

**F_pre(L) = α·L + γ·L² + 2dV**, 여기서

- α = n_A·2W_A + n_M·(2W_M + 2K·d_conv + F_S,pre) + n_F·2W_F
- γ = n_A·2·n_q·d_h
- 이차항이 선형항과 같아지는 길이: **L_q = α / γ**

**Decode 한 step (배치 b, 각 요청 문맥 c)**

- FLOPs: **b · [n_A·(2W_A + F_A,dec(c)) + n_M·(2W_M + 2K·d_conv + F_S,dec) + n_F·2W_F + 2dV]**
- HBM 바이트: **B_dec(b, c) = β·(n_A·W_A + n_M·W_M + n_F·W_F + d·V) + b·2·s + b·c·k**
- 메모리 바운드 하한: B_dec / 대역폭. 따라서

**T_dec ≥ a₀ + b₀·b + c₀·b·c**, 여기서 a₀ = 가중치 바이트 / BW, b₀ = 2s / BW, c₀ = k / BW

이것이 PREREG의 T1 함수형 `T_dec = a + b·bs + c·bs·ctx`의 물리적 하한이다. state 항과 KV 항이 같아지는 문맥은 **c\* = 2s / k**.

## 6. Nano-9B-v2 대입값 [유도]

| 항 | 값 |
|---|---|
| W_A / W_M / W_F | 55,050,240 / 147,374,080 / 140,492,800 |
| 전체 파라미터(정규화·임베딩 포함, 비공유) | 8,888,227,328 (부팅 배너 16.58 GiB와 일치) |
| s / k / L\* | 143,548,416 B / 16,384 B/토큰 / ≈ 8,762 토큰 (배너 실측 143.5 MB · 16,382 B와 일치) |
| F_S,pre / F_S,dec | 8,126,464 / 6,553,600 FLOP/토큰 |
| α / γ / L_q | 15.645 GFLOP/토큰 / 40,960 / ≈ 382k |
| 64k에서 이차항 비중 γL/α | 17% |
| a₀ / b₀ / c₀ (A100 SXM 2.039 TB/s) | 8.14 ms / 0.141 ms/요청 / 8.0 ns/(요청·토큰) |
| c\* | ≈ 17.5k |
| decode b=32, c=8k: 바이트 / 연산 강도 | 30.1 GB (가중치 16.6 + state 9.2 + KV 4.3) / 18.6 FLOP/B — 메모리 바운드(A100 ridge 약 150) |

**실측 대조(2건)**
1. 부팅 배너의 가중치·s·k와 일치(위 표).
2. CM-2 무경합 floor 기울기 0.111 ms/토큰(J1, 감사 CONFIRMED, L 1k–16k)에 α를 넣으면 실효 약 141 TFLOP/s, A100 bf16 피크의 약 45%다. floor가 이 수식과 정합한다는 기술값이며, 효율 η의 첫 실측점이다.

## 7. 조성 변형 대입값 (결정 24, 같은 골격) [유도]

| 변형 (n_A / n_M) | s | k (B/토큰) | L\* | L_q | 64k 이차항 비중 | c\* |
|---|---|---|---|---|---|---|
| a4 (4 / 27) | 143.5 MB | 16,384 | 8,762 | 382k | 0.17 | 17.5k |
| a13 (13 / 18) | 95.7 MB | 53,248 | 1,797 | 104.5k | 0.63 | 3.6k |
| a22 (22 / 9) | 47.8 MB | 90,112 | 531 | 54.0k | 1.21 | 1.1k |
| a31 (31 / 0) | 0 | 126,976 | 0 | 32.9k | 1.99 | 0 |

- k와 decode 문맥 계수 c₀는 **n_A에 정확히 비례**한다(PREREG rev3.3 예측 P3의 산술 근거).
- γ는 n_A에 비례하고 α는 n_A가 늘수록 줄어, prefill 비선형성이 조성에 따라 단조 증가한다(예측 P4의 근거).

## 8. 이 수식이 담지 않는 것과 확장 방향 (비용 모델 정립의 틀)

비용 모델은 이 수식을 **사전(prior)**으로 두고, 측정이 아래 항을 덧붙이는 형태로 정립한다. 각 확장은 이 문서에 새 절로 추가하고 버전을 올린다.

| 확장 항 | 형태(제안) | 채울 측정 |
|---|---|---|
| 층 종류별 효율 η | T = F / (η_X(SM)·피크) 또는 B / (η_X(SM)·BW), X ∈ {A, M, F} | T1·T2 (rev3.3), opchar(격리 커널, 특성화) |
| PD-mux SM 분할 | decode에 D SM, prefill에 108−D SM. η가 SM 수의 함수 | T1 D 축, T2 P 축 |
| lockstep·동기화 | iteration = max(T_dec·ρ, T_span) + 동기화 비용 | CM-3(무시 가능 확정), T3 |
| 동거 간섭 ρ | decode(대역폭 바운드)와 prefill(연산 바운드)의 HBM·SM 경합. 기전은 가설 | T3 ρ·σ (합성값 규칙 승계) |
| cudagraph 패딩 | b → 캡처 bs로 올림 | I-6 |
| 메모리 할당 구조 | SGLang 정적 두 풀: 동시 수 = min(cap, ⌊(풀 − cap·s)/(k·(L+R))⌋) | T4 (산식 B, 구현 사실 확인) |
| TP 분할 | s/N, k/N, 가중치/N + collective | J-C |

## 9. 이력

- 2026-10-22 analytic-v1: 최초 고정(결정 27). Marconi Table 1 대비 차이: GQA(n_kv), Mamba2 head·group 구조와 chunk 항, SSM state fp32(SGLang), conv state 포함, 비게이트 MLP, decode 트래픽식(가중치·state 읽기쓰기·KV) 추가.
