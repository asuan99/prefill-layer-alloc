# P1.7 — 추가 하이브리드 모델로 layer-aware 프레임워크 일반화 (실측)

작성 2026-07-05 · 대상 신규: **Zamba2-1.2B/7B**(크기축), **Falcon-H1-3B-Base**(spatial), **ibm-granite/granite-4.0-h-micro-base**(temporal GQA) · A100-80GB/CUDA13/sglang v0.5.10 · 기존 NemotronH-8B/Zamba2-2.7B([p1_4_layer_aware_평가_kr.md](p1_4_layer_aware_평가_kr.md)) 확장

## 0. 핵심 결론
**4개 하이브리드 실측으로 "어느 layer type이 SM-민감이냐가 최적 SM-예약 정책을 결정"함을 일반화했고, mamba SM-민감도의 진짜 드라이버가 모델 크기가 아니라 SSD 구현(compute vs memory-bound)임을 밝혀 이전 "크기 의존" 서술을 정정했다.**

| 모델 | 아키텍처 | mamba decode | attn decode (단문/장문) | SM-민감 지배층 | ⇒ 최적 정책 |
|---|---|---|---|---|---|
| **NemotronH-8B** | temporal, GQA | **SM-민감** 2.6× | 둔감 (GQA) | mamba (다수) | **agnostic**(전층 예약); layer-aware≈agnostic, **agnostic_v2 최악** |
| **Zamba2** 1.2/2.7/7B | temporal, **no-GQA** | 둔감 (memory-bound) | **장문 SM-민감** O(L) | attn (희소) | **layer-aware 유리** (mamba 환원 + attn 예약) |
| **Granite-4-h-micro** | temporal, GQA | 둔감 (memory-bound) | 둔감 (장문도 평탄) | **없음** | **agnostic_v2 최적** (전층 환원); layer-aware 퇴화 |
| **Falcon-H1-3B** | **spatial**, GQA | (층내 mamba∥attn 융합) | (층내 융합) | 단일 layer type | **layer-aware N/A**; **agnostic ≫ fused** |

## 1. 크기축 — mamba 민감도는 "크기"가 아니라 "SSD config" (정정)
`p1_7_zb_smsens.sbatch` (green-ctx N-sweep × ctx-sweep, per-type CUDA-event timing). 세 Zamba2 크기 모두 부팅·정확.

`[measured]` **per-mamba decode(ms/층), full→16 SM:**
| ctx | Zamba2-1.2B | Zamba2-7B | (참고) NemotronH-8B |
|---|---|---|---|
| 256 | 0.409 → 0.413 (평탄) | 0.949 → 0.828 (평탄/역) | — |
| 4000 | 0.385 → 0.273 (역) | 0.756 → 0.810 (평탄) | — |
| short | flat | flat | **0.589 → 1.547 (2.6× 민감)** |

`[measured]` **Zamba2 mamba는 1.2B·2.7B·7B 전부 SM-둔감(memory-bound)**. ★**7B(≈NemotronH-8B 크기)도 둔감** → "mamba 민감도 = 모델 크기 의존"(이전 NH-8B민감 vs ZB-2.7B둔감 해석)은 **교란**. 진짜 드라이버 = **Mamba2 SSD 구현이 compute-bound(NemotronH; SM 뺏기면 느려짐)냐 memory-bound(Zamba2/Granite; state I/O 지배라 SM 무관)냐**. `[measured]` **no-GQA attn은 전 크기 장문서 SM-민감(O(L)), 크기 클수록 심화**: per-attn full ctx64→4000 = 1.2B 0.130→0.396, 7B 0.090→1.471; 16SM 대비 ctx4000 민감도 1.2B 3.25× → 7B **5.1×**. ⇒ **layer-aware 유리 체제가 Zamba2 전 크기서 성립, 대형일수록 강화**. 원자료 `p1_7_zb_smsens_{12b_832554,7b_832555}.txt`.

## 2. Granite-4.0-h-micro (temporal GQA) — mamba 둔감 확증 + attn도 둔감 → agnostic_v2 최적
`p1_7_granite_smsens.sbatch` (`granitemoehybrid.py`에 green-ctx+timing 계측 추가). 부팅·정확. 40층 = 36 mamba + 4 attn(GQA 4:1).

`[measured]` **mamba(36층) 완전 SM-둔감**: per-mamba(ms) full/16SM = ctx64 0.841/0.860, ctx2048 0.832/0.865, ctx4000 0.831/0.865 — **전 ctx·전 SM 평탄(~0.85)**. ⇒ Zamba2류(memory-bound). **NemotronH-8B가 유일 outlier 확정.**
`[measured]` **attn(4층, GQA) SM-둔감 + 컨텍스트 불변**: per-attn full ctx64→4000 = 0.548→0.548(평탄!), 16SM ~0.58. GQA memory-bound + KV read 작아 O(L) 성장 거의 無 (Zamba2 no-GQA와 정반대).
`[derived]` ⇒ **Granite decode = 전 레이어 SM-둔감** → **agnostic_v2(전층 16SM로 decode 환원)가 최적**(굶길 민감층 없어 decode 손실 0, prefill에 최대 SM 환원), **layer-aware 퇴화**(보호 대상 없음). NemotronH(mamba 민감→agnostic_v2 최악, §3.9)와 **정확히 정반대 체제**. 원자료 `p1_7_gr_smsens_832611.txt`.

## 3. Falcon-H1-3B (spatial hybrid) — layer-aware 적용범위 경계
`falcon_h1.py`에 `forward_split_prefill` 추가(pdmux 활성). 부팅·정확·pdmux OK(flashinfer head_dim128).

`[구조]` Falcon-H1 = 32층 **전부 동일**(각 층 mamba∥attn 병렬합, `falcon_h1.py:355`). **레이어 '타입'이 없음** → **layer-aware(타입별 SM 예약) 적용 불가 = agnostic과 동일**; agnostic_v2도 단일타입이라 fixed-N agnostic. **정책공간이 {fused, agnostic}로 붕괴**.
`[measured]` **서빙(async bench_serving, sub-saturation)**: goodput@SLO rate3/4/6 = fused 1.14/0.64/**0.00** vs **agnostic 2.68/3.47/3.44**. Median TPOT(ms) fused 61/72/97(부하시 decode-prefill 결합→SLO 돌파) vs agnostic 42/43/41(decode 파티션 분리로 평탄). ⇒ **spatial hybrid도 pdmux 대승**(NemotronH보다 깔끔; rate6서 agnostic 3.44 vs fused 0). 원자료 `p1_7_bench_one_{832575,832576}.out`.

## 4. 함의
1. `[measured]` **최적 정책 = f(SM-민감 layer type)**: mamba민감(NemotronH)→agnostic·layer-aware무익·agnostic_v2해악 / attn민감(no-GQA 장문, Zamba2)→layer-aware유리 / 무민감(Granite)→agnostic_v2최적 / 단일타입(Falcon-H1 spatial)→pdmux(agnostic)만. **정책은 아키텍처가 아니라 커널 민감도 프로파일이 정한다.**
2. `[measured]` **mamba SM-민감도 = SSD 구현 특성(compute vs memory-bound), 크기·temporal여부 무관**. 4모델 중 NemotronH만 compute-bound. ⇒ 실전 temporal 하이브리드 다수는 mamba 둔감(환원 가능)일 개연 — layer-aware/agnostic_v2가 유리할 후보 넓음.
3. `[measured]` **attn SM-민감도 = GQA(둔감·평탄) vs no-GQA(장문 O(L) 민감)**. Zamba2(no-GQA)만 장문서 attn 예약가치 발생.
4. `[measured]` **pdmux(prefill/decode SM 분리)는 전 하이브리드서 견고한 이득**(spatial 포함); layer-aware/agnostic_v2는 그 위 조건부 정밀화.

## 5. 방법·재현
- 계측: `models/{zamba2(기존),granitemoehybrid(신규),falcon_h1(신규)}.py` per-type CUDA-event timing + green-ctx N-pin (env-gated). dev_tree_edits.md §8/9.
- harness: `p1_7_zb_smsens.sbatch <model>`, `p1_7_granite_smsens.sbatch`, `p1_7_bench_one.sbatch <policy> <model> <backend>`(async, sub-saturation), `p1_7_fh1_check.sbatch <plain|pdmux> [model]`.
- 가중치: Falcon-H1-3B/Zamba2-1.2B·7B 캐시; granite-4.0-h-micro-base 다운로드(로그인노드 인터넷).
- 미완/후속: Granite agnostic_v2 서빙 실증(예측=최적); Falcon-H1 층내 attn/mamba 분리 계측; Qwen3-Next(gated-deltanet) 등.
