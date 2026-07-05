# P1.7 — 추가 하이브리드 모델로 layer-aware 프레임워크 일반화 (실측)

작성 2026-07-05 · 대상 신규: **Zamba2-1.2B/7B**(크기축), **Falcon-H1-3B-Base**(spatial), **ibm-granite/granite-4.0-h-micro-base**(temporal GQA) · A100-80GB/CUDA13/sglang v0.5.10 · 기존 NemotronH-8B/Zamba2-2.7B([p1_4_layer_aware_평가_kr.md](p1_4_layer_aware_평가_kr.md)) 확장

## 0. 핵심 결론
**서빙 실증된 견고 결론: pdmux(agnostic v1)는 4개 하이브리드 전부서 fused 대비 견고한 이득; agnostic_v2는 decode에 실작업 있는 모델서 최악(NemotronH·Granite 서빙 확증). layer-aware의 서빙 이득은 어느 모델도 미확증.** 또한 mamba SM-민감도 드라이버는 크기가 아니라 SSD 구현(§1).

> ⚠️ **정정(2026-07-05, 사용자 지적+서빙 실증)**: 초판은 Granite를 "전층 SM-둔감→agnostic_v2 최적"으로 판정했으나 **서빙 실증서 반증**(§2). **decode-side micro-timing(작은 batch)이 서빙 batch의 SM-민감도를 과소평가**하는 게 근본원인 — decode에 실작업 있으면 서빙 batch서 compute-bound가 되어 SM 굶기면 TPOT 폭발. ⇒ **"둔감→환원가능" 라벨은 서빙 미검증이면 신뢰 불가.**

| 모델 | 아키텍처 | mamba decode(micro) | **서빙 실증 최적 정책** |
|---|---|---|---|
| **NemotronH-8B** | temporal, GQA | 민감 2.6× | **agnostic**; agnostic_v2 최악 (§3.9 async) |
| **Granite-4-h-micro** | temporal, GQA | micro-둔감이나 **서빙 batch서 민감** | **agnostic**(TPOT 34-37 평탄, goodput→5.81); **agnostic_v2 최악**(TPOT→106, goodput→0) |
| **Zamba2** 1.2/2.7/7B | temporal, **no-GQA** | micro-둔감 | layer-aware는 decode-side/TTFT 정황뿐, **서빙 goodput 미실증**(triton decode-bound) → 유보 |
| **Falcon-H1-3B** | **spatial**, GQA | (층내 mamba∥attn 융합) | **agnostic ≫ fused**(단일 타입→layer-aware N/A) |

## 1. 크기축 — mamba 민감도는 "크기"가 아니라 "SSD config" (정정)
`p1_7_zb_smsens.sbatch` (green-ctx N-sweep × ctx-sweep, per-type CUDA-event timing). 세 Zamba2 크기 모두 부팅·정확.

`[measured]` **per-mamba decode(ms/층), full→16 SM:**
| ctx | Zamba2-1.2B | Zamba2-7B | (참고) NemotronH-8B |
|---|---|---|---|
| 256 | 0.409 → 0.413 (평탄) | 0.949 → 0.828 (평탄/역) | — |
| 4000 | 0.385 → 0.273 (역) | 0.756 → 0.810 (평탄) | — |
| short | flat | flat | **0.589 → 1.547 (2.6× 민감)** |

`[measured]` **Zamba2 mamba는 1.2B·2.7B·7B 전부 SM-둔감(memory-bound)**. ★**7B(≈NemotronH-8B 크기)도 둔감** → "mamba 민감도 = 모델 크기 의존"(이전 NH-8B민감 vs ZB-2.7B둔감 해석)은 **교란**. 진짜 드라이버 = **Mamba2 SSD 구현이 compute-bound(NemotronH; SM 뺏기면 느려짐)냐 memory-bound(Zamba2/Granite; state I/O 지배라 SM 무관)냐**. `[measured]` **no-GQA attn은 전 크기 장문서 SM-민감(O(L)), 크기 클수록 심화**: per-attn full ctx64→4000 = 1.2B 0.130→0.396, 7B 0.090→1.471; 16SM 대비 ctx4000 민감도 1.2B 3.25× → 7B **5.1×**. ⇒ **layer-aware 유리 체제가 Zamba2 전 크기서 성립, 대형일수록 강화**. 원자료 `p1_7_zb_smsens_{12b_832554,7b_832555}.txt`.

## 2. Granite-4.0-h-micro (temporal GQA) — decode-side "둔감" 판정 → 서빙서 반증 (agnostic이 최적)
`p1_7_granite_smsens.sbatch` (`granitemoehybrid.py`에 green-ctx+timing 계측). 부팅·정확. 40층 = 36 mamba + 4 attn(GQA 4:1).

`[measured]` **decode-side micro-timing(작은 batch)**: per-mamba(ms) full/16SM = ctx64 0.841/0.860, ctx4000 0.831/0.865(평탄); per-attn 0.55(평탄). → 초판은 이걸로 "전층 SM-둔감→agnostic_v2 최적"이라 판정.

`[measured]` **★서빙 실증(async, fused/agnostic/agnostic_v2; jobs 832638/832639/832640)이 이를 반증:**
| rate | Median TPOT (ms) | | | goodput@SLO | | |
|---|---|---|---|---|---|---|
| | fused | agnostic | **agnostic_v2** | fused | agnostic | **agnostic_v2** |
| 3 | 50 | **36** | 56 | 2.76 | **3.35** | 2.23 |
| 4 | 61 | **37** | **106** | 1.95 | **4.33** | **0.00** |
| 6 | 78 | **34** | **107** | 0.26 | **5.81** | **0.00** |

⇒ **agnostic_v2 = 최악**(부하시 TPOT 106ms로 SLO 폭파, goodput 0), **agnostic(v1) = 최고**(TPOT 34-37 평탄, goodput 5.81까지 스케일). **Granite = NemotronH형**(§3.9와 동일), "전층 둔감" 아님.

`[근본원인]` **decode-side micro-timing(decode batch~8)이 서빙 batch(≤48)의 SM-민감도를 과소평가**: 작은 batch선 mamba/MLP GEMM이 latency/bw-bound라 둔감으로 보이나, 서빙 batch선 GEMM 커져 compute-bound → 16 SM으로 굶기면 TPOT 폭발. (+per-type timing이 attn층의 co-located MLP를 포함해 "attn ctx-평탄"도 granularity 아티팩트.) `[교훈]` **정책 주장은 반드시 서빙 실증**(P1.6c GIL-client 오측에 이은 2번째 micro-measurement 오도). ⇒ **decode-side "둔감→환원가능" 라벨은 서빙 미검증이면 신뢰불가; Zamba2 layer-aware 이득도 동일하게 유보(서빙 미실증).** 원자료 `p1_7_gr_smsens_832611.txt`, `p1_7_bench_one_{832638,832639,832640}.out`.

## 3. Falcon-H1-3B (spatial hybrid) — layer-aware 적용범위 경계
`falcon_h1.py`에 `forward_split_prefill` 추가(pdmux 활성). 부팅·정확·pdmux OK(flashinfer head_dim128).

`[구조]` Falcon-H1 = 32층 **전부 동일**(각 층 mamba∥attn 병렬합, `falcon_h1.py:355`). **레이어 '타입'이 없음** → **layer-aware(타입별 SM 예약) 적용 불가 = agnostic과 동일**; agnostic_v2도 단일타입이라 fixed-N agnostic. **정책공간이 {fused, agnostic}로 붕괴**.
`[measured]` **서빙(async bench_serving, sub-saturation)**: goodput@SLO rate3/4/6 = fused 1.14/0.64/**0.00** vs **agnostic 2.68/3.47/3.44**. Median TPOT(ms) fused 61/72/97(부하시 decode-prefill 결합→SLO 돌파) vs agnostic 42/43/41(decode 파티션 분리로 평탄). ⇒ **spatial hybrid도 pdmux 대승**(NemotronH보다 깔끔; rate6서 agnostic 3.44 vs fused 0). 원자료 `p1_7_bench_one_{832575,832576}.out`.

## 4. 함의 (서빙 실증 기준)
1. `[measured]` **pdmux(agnostic v1)는 4모델 전부서 fused 대비 견고 승** — 서빙 실증된 유일하게 확실한 결론(NemotronH·Granite·Falcon-H1 goodput; decode/prefill SM 분리로 TPOT 평탄 유지).
2. `[measured]` **agnostic_v2(전층 저 floor 환원)는 decode에 실작업 있으면 최악** — NemotronH·Granite 서빙 확증(TPOT 부하시 106ms↑ SLO 폭파). "전층 둔감→환원가능"은 서빙 batch서 무너짐.
3. `[measured→반증]` **decode-side micro-timing(작은 batch)은 서빙 정책을 예측 못 함**: SM-민감도가 batch 의존(작은 batch=latency-bound=둔감, 서빙 batch=compute-bound=민감)이라 과소평가. ⇒ **"둔감→layer-aware/agnostic_v2 유리" 류 주장은 서빙 미검증이면 신뢰불가**(Granite서 반증). **Zamba2 layer-aware 이득도 서빙 미실증→유보.**
4. `[measured]` **mamba micro-민감도 = SSD 구현(compute vs memory-bound), 크기·temporal 무관**(§1; ZB-7B≈NH크기지만 micro-둔감) — 이 관찰 자체는 유효하나, **micro-둔감이 서빙 정책상 "환원가능"을 뜻하진 않음**(Granite 교훈).
5. `[measured]` **attn: GQA(둔감) vs no-GQA(장문 O(L) 민감)** — micro 수준 관찰. 정책 함의(no-GQA 장문서 attn 예약가치)는 서빙 미실증(Zamba2 triton decode-bound).
6. **layer-aware의 서빙 goodput 이득은 4모델 중 어느 것도 미확증** — decode-side 정황만으로 주장했던 이득이 서빙 실증을 못 통과(NemotronH la≈agn, Granite/Zamba2 미실증). **layer-aware는 현재 "가설"; pdmux(agnostic)가 실전 권고.**

## 5. 방법·재현
- 계측: `models/{zamba2(기존),granitemoehybrid(신규),falcon_h1(신규)}.py` per-type CUDA-event timing + green-ctx N-pin (env-gated). dev_tree_edits.md §8/9.
- harness: `p1_7_zb_smsens.sbatch <model>`, `p1_7_granite_smsens.sbatch`, `p1_7_bench_one.sbatch <policy> <model> <backend>`(async, sub-saturation), `p1_7_fh1_check.sbatch <plain|pdmux> [model]`.
- 가중치: Falcon-H1-3B/Zamba2-1.2B·7B 캐시; granite-4.0-h-micro-base 다운로드(로그인노드 인터넷).
- 미완/후속: Granite agnostic_v2 서빙 실증(예측=최적); Falcon-H1 층내 attn/mamba 분리 계측; Qwen3-Next(gated-deltanet) 등.
