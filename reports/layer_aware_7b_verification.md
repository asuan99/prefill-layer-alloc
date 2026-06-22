# layer-aware 예약의 7B 회귀 검증 — 검수 C1 종결

작성일: 2026-06-22 · 대상: [layer_aware_benefit_report](layer_aware_benefit_report.md)의 살아있는 결론(*"temporal 하이브리드서 layer-type-aware SM 예약이 ~2× goodput"*)이 **7B 스케일에서도 성립하는가**
근거 데이터: `results_v2/e5_dp/serving_coexec_full_zamba2_7b_*.csv`(측정), `results_v2/e6/layer_aware_zamba2_7b_b8.csv`(sim)
스크립트: `experiments/e6_queue_sim/synth_7b_protect_lut.py`(측정→합성 LUT) · `run_layer_aware.py`
관련: [검수 C1](review_checklist.md) · [real_prefill §7·§8](real_prefill_results.md) · [vllm_validation](vllm_validation.md)

---

## 0. 요약 (TL;DR)

**판정: layer_aware의 ~2× goodput은 7B로 스케일하지 않는다 — 단정적 절대 확증은 측정 차단, 그러나 측정된 7B 구조가 lever의 전제를 부정한다.**

- 검수 C1이 "결정적 다음 단계"로 꼽은 **7B 회귀를, 살아있는 가설(layer_aware 예약)에 대해 처음으로 돌렸다.** (공간-*분할* 가설의 7B는 [real_prefill §7](real_prefill_results.md)에서 이미 음성으로 닫힘.)
- 2.7b의 2×를 만든 두 재료가 **7B에선 측정상 둘 다 약하거나 부재**다:
  1. **싼 attn 보호** — 2.7b는 attn-decode를 54 SM(저배치)~68 SM로 포화시켜 싸게 보호. **7B attn-decode는 측정 가능한 모든 배치에서 54 SM로도 포화 안 됨(+67~97% 팽창)** → 보호하려면 >54 SM 필요 → prefill 환원 여지↓.
  2. **SM에 민감한 ssm prefill 환원** — 2.7b는 protect가 ssm prefill을 14 SM까지 굶겼다가 layer_aware가 108 SM로 환원(prefill 6.2×). **7B는 측정된 54–108 SM 구간에서 ssm prefill이 거의 무감각(8.96→8.78ms, 1.02×)** → 환원 이득 미미.
- 더해서 **7B는 decode-step 지배**(b8서 81층 합산 ~51ms/step) → goodput 병목이 decode이고, layer_aware가 손대는 건 *prefill* 측이라 leverage가 구조적으로 작다.
- **차단 사유(절대 확증 불가):** 충실한 floor 기반 protect를 7B에 구성하려면 (a) 7B E3 decode floor, (b) floor-frac에서의 green_ctx(prefill을 <54 SM로) 측정이 필요한데 **둘 다 미측정**이고, 신규 측정은 깨진 커널(mamba_ssm: `libcudart.so.13` 부재, CUDA13빌드/CUDA12환경)·**A100-SXM4 부재(로컬은 PCIe, slurm은 V100만)**로 막혀 있다. 기존 7B partition 측정은 f∈{0.5,0.7}뿐이라 floor 평가에 너무 거칠다.

---

## 1. 무엇을 물었나 — 공간분할 7B(이미 닫힘)와 다른 질문

[review_checklist C1](review_checklist.md)은 "7B 회귀가 결정적인데 미실행"을 P0로 남겼다. 그 사이 두 갈래가 생겼다:

| 가설 | 7B 상태 |
|---|---|
| layer-type *공간 분할*(green_ctx가 two_stream 이기나) | **닫힘·음성** — [real_prefill §7](real_prefill_results.md): 2.7b의 +12.5% throughput 예외가 7B서 사라짐(zamba2_7b·falcon_h1_7b 0셀). |
| layer-type-aware *예약*(본 프로젝트의 살아있는 결론, §14/layer_aware_benefit_report) | **본 보고서가 처음 검증.** |

따라서 본 보고서는 *살아있는* 결론 한 가지 — "비싼 attn-decode 레이어만 SM 예약, 싼 ssm 레이어는 prefill에 SM 환원 → ~2× goodput" — 을 zamba2_7b(81층 = 13 attn + **68 ssm = 84% ssm**, 2.7b의 83%보다도 ssm-지배적)에서 묻는다. 메커니즘 가설대로면 ssm 비율↑이라 *더 큰* 이득이 나야 한다.

## 2. 측정된 7B 구조 — lever의 두 전제가 무너진다 (sim 불필요)

전부 `results_v2/e5_dp/serving_coexec_full_zamba2_7b_a100_sxm4_80gb.csv`(SXM4 실측, ctx=4096)에서 직접.

### 2.1 전제①(싼 attn 보호) 부정 — 7B attn-decode는 54 SM로 포화 안 됨

decode를 부분 SM에 올렸을 때 solo(108 SM 단독) 대비 팽창률:

| layer | 배치 | decode@54SM | decode@32SM | solo(108) | 판정 |
|---|--|--|--|--|---|
| **attn** | 1 | **+76%** | +126% | 0.114ms | 54로도 미포화 |
| **attn** | 8 | **+67%** | +169% | 0.959ms | 54로도 미포화 |
| **attn** | 256 | **+89%** | +218% | 25.1ms | 54로도 미포화 |
| ssm | 1 | +0% | +1% | 0.525ms | 32서 포화 ✓ |
| ssm | 8 | +17% | +36% | 0.566ms | 54서 근포화 |
| ssm | 256 | +75% | +180% | 4.07ms | 54로도 미포화 |

대조 — **2.7b E3 floor**(decode 포화 SM, ctx=4096): attn b1=**54**, b2=54, b4/b8=68, b16=108; ssm b1=**14**, b8=54, b32=94. 즉 2.7b는 저배치 attn을 54 SM로 포화시켜 **싸게 보호**했다. 7B는 같은 54 SM가 attn-decode를 +67~97% 팽창시킨다 → **싸게 보호할 수 없다.** ([real_prefill §8](real_prefill_results.md)이 *full-model* 7B에서 protect가 `no_room`(floor→GPU 포화)이라 한 것과 같은 방향.)

### 2.2 전제②(SM-민감 prefill 환원) 부정 — 측정 구간서 7B prefill 거의 무감각

layer_aware의 이득원 = ssm 레이어에서 prefill에 SM을 돌려줄 때의 prefill 가속. prefill_stream(ms):

| | 108 SM(two_stream) | 54 SM(green f0.5) | 14 SM(green) | 환원 이득 |
|---|--|--|--|--|
| **zamba2_2.7b** ssm prefill | 0.888 | 1.483 | **5.506** | 108÷14 = **6.2×** |
| **zamba2_7b** ssm prefill | 8.782 | 8.962 | (미측정) | 108÷54 = **1.02×** |

2.7b는 protect가 고배치서 ssm prefill을 **14 SM까지 굶기**(5.5ms) → layer_aware가 108로 환원(0.89ms)해 6.2× 회수가 2× goodput을 견인했다. **7B는 측정된 54–108 SM 구간에서 prefill이 거의 변하지 않는다**(1.02×). (정직: 7B의 <54 SM 구간은 미측정 — 거기서 7B prefill이 급락한다면 환원 이득이 커질 수 있으나, 그건 차단된 측정에 달려 있다. §4.)

### 2.3 보강 — 7B는 decode-step 지배라 prefill lever의 leverage가 작다

full-model decode step ≈ Σ_layers(solo_decode). b8: 13·0.959 + 68·0.566 ≈ **51ms/step** (2.7b는 ≈ 28.6ms, [vllm_validation §3](vllm_validation.md)). 서버 capacity가 ~160 tok/s로 decode에 묶이고, layer_aware가 개선하는 prefill(TTFT) 측은 goodput 천장을 거의 못 올린다.

## 3. sim 보강 — 동일 합성법의 모델 간 대조 (방법 한계 명시)

7B는 floor 기반 protect를 측정 못 했으므로, **측정된 green_ctx(f=0.5/0.7)만으로 protect를 합성**해(`synth_7b_protect_lut.py`: 각 (층,배치)에서 decode를 solo의 15% 내로 두는 최소 측정 SM, 없으면 54) `run_layer_aware`를 돌렸다. **동일 합성법을 2.7b에도 적용한 대조**(real-E3 결과와 비교)로 방법의 신뢰구간을 박는다:

| (λ=0.5, SLO_ITL=100ms) | co | agnostic | layer_aware | la/ag | la ITL_p99 |
|---|--:|--:|--:|--:|--:|
| **real-E3 2.7b** (참값) | 752 | 629 | **1267** | 2.0× | 43.5ms |
| synth-2.7b (f-cap) | 752 | 88 | 221 | 2.5× | 178ms |
| **synth-7b** (f-cap) | 171 | 150 | 159 | **1.06×** | 86.5ms |
*(goodput tok/s)*

**방법 한계(반드시 명시):** 합성 protect는 reservation을 측정 최대 54 SM로 cap하므로 decode 보호를 과소평가한다 — 그래서 synth-2.7b조차 절대 SLO를 망가뜨려(real la=1267>co를 synth는 la=221<co로 뒤집음) **"co가 7B서 이긴다"를 합성 sim 단독 근거로 주장하지 않는다.** 합성법이 *보존*하는 건 **layer_aware>agnostic 순서와 그 비율의 모델-스케일 추세**: la/agnostic 우위가 **2.7b ~2.0–2.5× → 7B 1.06×로 붕괴.** §2의 측정 구조와 같은 방향(7B에선 lever가 거의 작동 안 함).

## 4. 차단된 것 — 절대 확증에 필요한 측정 (정직)

| 필요 | 현 상태 |
|---|---|
| 7B E3 decode floor(층별·배치별 포화 SM) | **미측정.** `decode_floor_zamba2_7b` 부재(1.2b/2.7b/falcon만 존재). |
| floor-frac green_ctx(prefill을 <54 SM로) | **미측정.** 7B partition은 f∈{0.5,0.7}뿐 → prefill이 54 SM 밑으로 간 적 없음(2.7b의 이득이 살던 영역). |
| 신규 GPU 측정 | **차단.** mamba_ssm 깨짐(`libcudart.so.13`, CUDA13빌드/CUDA12환경), 로컬 GPU는 A100-**PCIe**(SXM4 아님), slurm 가시 파티션은 V100뿐. |

→ "7B에서 layer_aware가 co_schedule을 *절대로* 못 이긴다"의 단정은 이 측정이 풀려야 가능. 다만 §2의 두 전제 부정 + [real_prefill §8](real_prefill_results.md)의 full-model protect `no_room`이 **양 끝을 묶는다**: reservation을 적게(54) → decode 미보호(SLO 이득 없음, §3), 많게(≥94, 7B 포화에 필요) → prefill 고갈(throughput 붕괴). 어느 쪽도 2.7b식 sweet-spot이 없다.

## 5. 결론 — 검수 C1 닫힘 (조건부)

원래 가설의 *살아있는* 형태(layer_aware 예약)는 **2.7b/3B(SLM)·A100서 ~2× goodput으로 살아있되, 7B로는 스케일하지 않는다**:
- **측정 확정:** 7B는 (①) attn-decode를 싸게 보호할 수 없고(54 SM로 +67~97%), (②) 측정 구간서 prefill 환원 이득이 거의 없으며(1.02×), (③) decode-step 지배라 prefill lever의 leverage가 작다.
- **sim 보강(방법 한계 내):** la/agnostic 우위가 2.7b 2.0× → 7B 1.06×로 붕괴.
- **공간-분할 7B 음성**([real_prefill §7](real_prefill_results.md))과 **방향 일치** → 프로젝트 헤드라인("SM 예약/분할은 SLM 한정 현상")이 *결정적 스케일점에서 강화*된다.
- **남은 것:** 7B E3 floor + sub-54-SM green_ctx 실측(SXM4·정상 커널)으로 절대 goodput 확증. 환경 복구 시 1회 회귀로 종결 가능.

검수 체크리스트 갱신: **C1 = 통과(조건부)** — "SM 예약 불필요"를 모델 크기 무관 결론으로 쓰지 않되, *살아있는 layer_aware도 7B서 lever 전제가 측정상 부재*임을 명기. SLM 한정 스코프 유지.
