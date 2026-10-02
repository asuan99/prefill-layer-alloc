# 선행 cost model 정리 — Bullet(ASPLOS '26)과 MuxWise(ASPLOS '26)가 실제로 쓰는 지연·간섭 모델 (2026-10-01)

> **지위**: `reports/cost_model_design_space_2026-10-01.md`의 부속. 출처는 로컬 원문 `Papers/Bullet.pdf`(camera-ready, 본문 §3.2–§3.3)와
> `Papers/Muxwise.pdf`(camera-ready, §3.3–§3.4)를 `pdftotext`로 추출해 직접 읽은 것이다(2026-10-01). 수치는 **논문 주장**이며 우리
> 기판에서 재현한 값이 아니다. 공개 코드 상태는 `reports/impl_vs_external_pdmux_2026-08-28.md`·`reports/audit/2026-09-28_contributions/
> AUDIT_1_prior_art_and_arm_2026-09-28.md`의 판정을 그대로 승계한다. **GPU 0 · 측정 0.**
> ★이 정리는 §4에서 우리 설계 문서의 X-1·X-4·X-7 서술을 **정정**한다(선행 모델의 "형식"은 hybrid에서도 대체로 살아남는다 — 깨지는 것은
> 형식이 아니라 항의 SM 의존성·층 단위·메모리 단위·간섭 대리변수다).

## 1. Bullet — SM-scaling Roofline Model(SRM) + 보정 계수 + 간섭 보정

### 1.1 문제 정식화 (§3.2.1)
동시 실행되는 prefill/decode의 지연을 결정하는 **실행 상태** `ES = (sl_i, pbs, pm, cl_i, dbs, dm)`: prefill 시퀀스 길이·배치 크기·SM 수,
decode 컨텍스트 길이·배치 크기·SM 수. 조합이 수백만이라 전수 프로파일은 불가 → 분석 모델 + 소량 샘플.

### 1.2 SRM (§3.2.2, 식 1)
커널 `k`가 `flop_k` 연산과 `mem_k` 바이트를 쓸 때, `N_p`개 SM에서의 이론 지연:
```
T'_{k,p} = flop_k · [ min( (flop_k/mem_k) · D_p , C_p ) ]^{-1}
C_p = C_peak · N_p/N                          # 연산 성능은 SM 수에 선형
D_p = D_peak · min(1, N_p/N_d)                # 대역폭은 변곡점 N_d까지 선형, 이후 포화 (A100 예시 N_d = 30)
```
각 `ES`에 대해 **llm-viewer로 모든 LLM 커널의 산술 강도를 계산**해 SRM으로 커널별 기준 지연을 얻고 합산 → `T'_{p,ES}`.
⇒ 형식 자체는 **커널 무관**(flop/mem만 있으면 어떤 커널이든 넣을 수 있다). transformer 가정은 SRM이 아니라 **커널 열거(llm-viewer)와
`ES`의 변수 집합**에 들어 있다.

### 1.3 보정 (§3.2.2, 식 2)
```
α_{p,ES} = T^{measured}_{p,ES} / T'_{p,ES}
```
실측 몇 점에서 α를 구해 **미측정 구성으로 선형 외삽**("유사 커널 입력 사이에서 이용률 패턴이 거의 선형"). Fig. 9: **표본 2개**로 SM을
바꿔 가며 decode 지연을 보정. 평균 α = 1.61(논문 값).

### 1.4 간섭 모델 (§3.2.3)
- SM을 분리해도 **메모리 서브시스템·네트워크 간섭**은 남는다. 커널 수준 온라인 식별은 어렵지만 end-to-end 지연은 안정적(7360 ES ×
  30회, 95%가 ±6.8% 이내 — 논문 값) → 그 안정성을 간섭 모델링의 근거로 씀.
- 최악 메모리 간섭은 `N_p` SM의 memcpy 커널 + 나머지 SM의 up-gated GEMM(UG) 동시 실행으로 정량화. UG는 60% 이상 SM에서 성능 저하 <8%
  → prefill 커널은 식 2로 소량 샘플 보정 가능. decode는 `sl` 길이 prefill과 동시 실행할 때의 **달성 대역폭 `D_{p,sl}`을 측정해 SRM에
  반영**한 뒤 식 2로 정련.
- 네트워크 간섭은 낮다고 봄(`sl` 또는 `dbs`에 비례, `dbs`가 작음).

### 1.5 프로파일링·온라인 보정 (§3.2.4)
오프라인: SM 수를 바꿔 가며 compute/memory/network 성능을 **한 번 스윕**해 SRM 구성 + 동시 실행 구성 소수를 샘플해 α. Nsight Compute
불필요. 온라인: SRM 예측을 사전 측정 α의 보간으로 보정, 연속 재보정. 예측·갱신 비용 무시 가능(성능 예측 10.2 µs, §4). Fig. 11:
"contention ignored" 예측기와 `ES` 변수의 선형 회귀(`pl` 2차항 포함)는 부정확, SRM은 표본 범위 밖에서도 정확(논문 주장). 프로파일
1시간 미만.

### 1.6 소비자 — Algorithm 1 (§3.3.2)
상태 `S = (ES, PS, RS)`. 매 step: `ttft ← EstimateLatency(S)`, `tpot` 읽기, `SortByLeastEstimLatency(Q)`(TTFT SLO를 깨지 않는 한도),
새 prefill step은 `ArithInten(next_tasks, ES) < peak`까지 배치. `P90(tpot) ≤ Γ_d`면 `ReduceDecodeSM`, `> Γ_d`면 `ReducePrefillSM`,
둘 다 못 지키면 `SetBalancedSM`. 반환값에 `L_exe + L_step`(이번 step에 돌릴 층 수). prefill 우선, decode는 SLO를 만족하는 **최소 SM**,
극단 부하에서 TPOT SLO가 유지되는 한 decode 일시 중단(Fig. 12-➁). decode는 CUDA graph 1개로 발행, prefill은 층 단위.

### 1.7 공개 코드 상태 (승계)
예측기는 `.so`로 외부 주입(`predictor_param_file`)이라 **공개 저장소에 없다**. `enable_sm_partition` 없이는 고정 분할로 동작. libsmctrl은
CUDA ≤ 12.6. ⇒ Bullet의 적응형 정책은 공개 코드로 재현 불가.

## 2. MuxWise — 복잡도 기반 solo-run 회귀 + contention guard(최악 배수) + N_PL 디스패처

### 2.1 설계 목표 (§3.1, §3.3.2)
정확한 예측이 아니라 **SLO 보장**: 배정된 자원에서 phase 지연이 목표를 넘지 않게 하는 것. 그래서 solo-run 예측 × **최대 slowdown 배수**의
최악 추정을 쓴다. 다섯 변수: reused 길이·input 길이·output 길이·decode 배치 크기·분할 구성. LLM–머신 쌍당 1회 오프라인 프로파일.

### 2.2 solo-run 예측기 (§3.3.2, Table 2, 식 1·2)
복잡도 분석(bs=1, `d` 은닉 차원, `L` 총 토큰, `r` reused, `n = L − r` 신규):

| | Attention | FFN |
|---|---|---|
| Prefill w/o cache | O(Ld² + L²d) | O(Ld²) |
| Prefill w/ cache | O(nd² + Lnd) | O(nd²) |
| Decode | O(d² + (r+1)d) | O(d²) |

```
T_Prefill = θ1·Σ_i n_i²  + θ2·Σ_i n_i·r_i + θ3·Σ_i n_i + θ4        (1)
T_Decode  = θ1·Σ_i r_i   + θ2·bs          + θ3                     (2)
```
θ는 오프라인 프로파일로 적합. 최대 편차 prefill 8.16% / decode 8.84%(논문 값), 프로파일 수 시간. ★식 1·2에는 **SM 변수가 없다** —
solo-run 예측이 분할 구성별로 따로 적합되는지, 전 GPU 기준인지는 본문에서 확정되지 않는다[미확인]. 분할 구성은 contention guard의
변수로만 명시된다.

### 2.3 contention guard (§3.3.1–§3.3.2)
GreenContext는 SM만 나누고 대역폭은 못 나눔 → decode slowdown이 분할 구성에 따라 0–30%(Fig. 11, Llama-8B/70B, A100/H100)로 **예측
불가**하다고 보고 최대 배수로 가드. 격자 표본: prefill 신규/reused 토큰, decode 배치 크기, decode 총 reused 토큰, 분할 구성의 5변수;
토큰은 4의 거듭제곱(2K–128K), 배치 ≈20종, **분할 입도 16 SM**(A100 6구성, H100 7구성; 이유: H100 thread block cluster가 16 SM 요구 +
16 SM이면 충분한 개선). 표본 ≈7K, 12시간. 최대 slowdown A100 20% / H100 30%. 온라인 실행 데이터로 가드를 계속 갱신. **가드는 decode에만**
(prefill SLO는 직접 보장하지 않음 — prefill SLO 위반은 용량 초과의 신호로 해석).

### 2.4 디스패처 (§3.4)
decode SLO 우선, **최악 추정으로 best-fit SM을 decode에**, 나머지를 prefill에. prefill 예측은 정확할 필요 없이 "발사한 prefill 층의 예측
지연이 해당 decode iteration 지연을 넘는가"만 필요 →
```
N_PL = ⌈ (T_d × N_T) / T_P ⌉     # T_d: 추정 decode 지연, T_P: 추정 prefill(전체) 지연, N_T: transformer 층 수
```
결정 시점: **prefill 배치 완료 후**와 **decode iteration 끝**. 선점: P2가 P1을 선점(P1의 TTFT SLO를 깨지 않을 때만), 비재귀, optional.
query-based sync로 prefill 완료 즉시 decode 배치에 merge.

### 2.5 공개 코드 상태 (승계)
공개 엔진(우리 base의 upstream)에는 solo-run 예측기·contention guard·`N_PL` 심볼이 **없다**. 있는 것은 `manual_divisions`의
`decode_bs_threshold` 표(upstream 주석 `temporary demo`)와 `split_forward_token_budget` 기반 span 수 공식뿐이다. 논문의 디스패처는
공개 코드로 재현 불가.

## 3. 나란히 비교

| 축 | Bullet | MuxWise |
|---|---|---|
| 모델 형식 | 커널별 roofline(SRM) 합산 + 실측/이론 배수 α | 복잡도 기반 다항 회귀(prefill 2차·decode 1차) + 최악 배수 |
| 입력 변수 | `ES` 6개(sl, pbs, pm, cl, dbs, dm) — **SM이 1차 변수** | reused/new 토큰·bs(·분할 구성은 가드에서만) — 식에 SM 없음[미확인] |
| ctx 처리 | 커널 flop/mem을 통해(attention 커널이 `cl`에 비례) | decode: `θ1·Σ r_i`(선형), prefill: `n²`, `n·r` 항 |
| SM 스케일링 | `C_p ∝ N_p`, `D_p` 변곡점 포화 — 커널별 강도로 자동 결정 | 명시 없음(가드 격자의 16-SM 구성) |
| 간섭 | memcpy+UG 프록시로 최악 대역폭, `D_{p,sl}` 측정, α 정련 | 5변수 격자의 최대 slowdown(decode만), 온라인 갱신 |
| 보정 | α 선형 외삽, 2표본으로 SM 축 | θ 적합(수 시간), 가드 갱신 |
| 결정 소비자 | Alg. 1: 정렬·강도 배치·SM 3분기·층 수 반환·decode 일시중단 | best-fit decode SM·`N_PL`·선점 |
| 결정 시점 | 매 step(층 단위 prefill, graph 단위 decode) | prefill 배치 완료·decode iteration 끝 |
| 정확도 주장 | run-to-run ±6.8%(95%), 회귀 대비 우위(Fig. 11) | 최대 편차 8–9%, slowdown ≤ 30% |
| 프로파일 비용 | < 1 h | 수 시간(예측기) + 12 h(가드) |
| 공개 코드 | 예측기 비공개 | 예측기·가드·N_PL 없음 |

## 4. hybrid에서의 함의 — 우리 X 항목의 정정

원문을 읽고 나면 "선행 model이 decode 지연을 ctx에 단조 증가로 가정한다"는 우리 X-1 서술은 **너무 거칠다**. 정확히는:

- **MuxWise 식 2는 이미 hybrid decode의 형식이다.** `θ1·Σ r_i`(ctx 비례 항)와 `θ2·bs`(ctx 무관 항)가 분리돼 있어, hybrid 모델에서
  다시 적합하면 θ1/θ2 비율만 작아진다. **단일 모델·고정 SM에서는 형식이 깨지지 않는다.** 깨지는 곳은 (a) **θ 비율이 SM에 따라
  달라지는 것**(attention 커널과 SSM 커널의 SM 스케일링이 다르면 SM마다 θ1/θ2가 달라 SM 축을 명시하지 않은 회귀는 분할 구성 간
  이식이 안 됨 — X-7의 정확한 형태) (b) prefill 식 1의 `n²` 항이 SSM 층에는 없어 층 조성이 다른 모델로 **이식**할 때 틀림(단일 모델
  재적합은 여전히 가능) (c) `N_PL`의 **층 균일 가정**(X-4, 구조적) (d) 가드 격자의 변수 "reused 토큰"이 KV 대역폭의 대리인데 SSM 층은
  reused 토큰과 무관 → 가드가 과대(X-5의 정확한 형태).
- **Bullet SRM은 커널 무관이다.** SSM scan 커널의 flop/mem을 열거하면 형식은 그대로 쓸 수 있다. 깨지는 곳은 (a) llm-viewer 계열
  **커널 열거가 transformer 커널만** 다룬다는 점(SSM 커널의 강도를 넣어야 함 — 이것이 Bullet 자신이 "straightforward, future work"라
  적은 확장의 실체) (b) `ES`에서 decode ctx `cl`이 attention 커널 flop/mem에만 들어가야 하는데 열거가 그렇게 돼 있는지 (c) **2표본 SM
  외삽**이 커널 계열마다 다른 변곡점 `N_d`(attention KV 읽기 vs SSM state 갱신)를 한 α로 뭉갤 위험(X-7) (d) 간섭 프록시(memcpy+UG
  GEMM)가 SSM scan의 대역 패턴을 대표하는지(X-5).
- **메모리 예산(X-6)**은 두 논문 모두 KV pool 공유를 전제하며 mamba state 슬롯을 다루지 않는다 — 이것은 형식이 아니라 **변수 부재**다.

⇒ 우리 주장 1을 지탱할 P_gen의 정의를 다시 쓴다. **가장 강한 generic baseline은 "MuxWise 식 2 형식을 hybrid 모델에 재적합한 것 +
Bullet식 온라인 α 보정"**이며, 이것과 P_hyb의 차이는 **SM 축과 배치 ctx 조성 축**에서만 나온다. 단일 SM·단일 조성에서는 차이가
0에 가까울 것으로 예상하며, 그 경우 정직한 결론은 "예측기 수준에서는 hybrid 확장이 자명하다"이고 novelty는 (c) 층 단위 디스패치·(d)
간섭 가드·메모리 슬롯·pause manager 쪽에서 찾아야 한다. 이 정정은 `cost_model_design_space_2026-10-01.md` §2 표에 반영한다.

## 5. 인용 시 주의
- 위 수치(±6.8%, 8.16%, 20/30%, 12 h 등)는 논문 주장이며 인용 시 "논문 보고 값"을 병기한다. 우리 기판 재현 없음.
- MuxWise 예측기의 SM 의존 여부는 [미확인] — 원문 §3.3.2가 명시하지 않는다. 인용 시 단정하지 말 것.
- Bullet camera-ready와 arXiv 판본이 다를 수 있다(CONSENSUS §3 항목283 "preprint/camera-ready 판본 함정"). 이 정리는 로컬 camera-ready
  PDF 기준이다.
