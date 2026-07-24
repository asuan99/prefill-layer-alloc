# layer-aware 예약의 7B 회귀 검증 — **반전: 7B도 스케일한다(1.82×)**

작성일: 2026-06-22 · 대상: [layer_aware_benefit_report](layer_aware_benefit_report.md)의 결론(*"temporal 하이브리드서 layer-type-aware SM 예약이 ~2× goodput"*)이 **7B 스케일에서도 성립하는가**
근거 데이터: `results_v2/e3/decode_floor_zamba2_7b_*.csv`(E3 floor, **job 797832서 ssm 추가**) · `results_v2/e5_sim_b8_opt/serving_coexec_full_zamba2_7b_*.csv`(E5) · `results_v2/e6/layer_aware_zamba2_7b_b8.csv`
관련: [검수 C1](review_checklist.md) · [real_prefill §7·§8](real_prefill_results.md) · [vllm_validation](vllm_validation.md)

---

> **⚠⚠ 결론 반전 (2026-06-22, job 797832, 깨끗한 LUT):** 이전 "7B 미스케일(1.00×)" 결론은 **측정 artifact였다.** 실제는 **la/agnostic = 1.82× — 7B도 2.7b(2.0×)와 거의 같은 강도로 *스케일한다*.** 아래 §0~§5의 "미스케일" 논증(§2 구조 전제 ①③ 포함)은 **전부 폐기**한다. 자세한 인과는 §6(신규).

## 0. 요약 (TL;DR — 갱신)

**판정: layer_aware의 ~2× goodput은 7B로도 스케일한다.** 깨끗한 실측 LUT(job 797832: E3 ssm floor + E5 green_ctx_protect)로 `run_layer_aware`: **co 337 / agnostic 273 / layer_aware 496 → la/agnostic = 1.82×**(@SLO≥80ms). 2.7b와 *동일 패턴*: **agnostic(273) < co(337)** — agnostic이 ssm 레이어까지 decode 예약해 prefill을 굶김(7B ssm floor b8=68→prefill 40 SM) → layer_aware가 ssm 레이어 SM을 prefill에 환원해 역전. **크기 추세: 1.2b 1.37× / 2.7b 2.02×(peak) / 7b 1.82× — 전 구간 이득, 7B 포함.** (전제②"prefill 민감"은 job 797524서 이미 확인됨 — 그게 7B서도 lever가 사는 이유.)

> **왜 이전엔 1.00×였나 (artifact 연쇄, §6):** 785877의 decode_ssm **Triton-캐시 OSError** → 7B **E3 ssm floor 미측정**(attn만 9행) → E5 green_ctx_protect가 7B ssm서 floor 못 찾아 실패 → `run_layer_aware`서 agnostic이 two_stream으로 폴백 = layer_aware와 **동일(1.00×)**. `TRITON_CACHE_DIR`=scratch fix + E3 ssm floor 재측정(job 797832)으로 해소 → 진짜 값 1.82× 드러남. **교훈: 불완전 LUT 셀이 정책을 조용히 degenerate시킨다 — 셀단위 완전성 검증 필수.**

- 검수 C1이 "결정적 다음 단계"로 꼽은 **7B 회귀를, 살아있는 가설(layer_aware 예약)에 대해 처음으로 돌렸다.** (공간-*분할* 가설의 7B는 [real_prefill §7](real_prefill_results.md)에서 이미 음성으로 닫힘.)
- 2.7b의 2×를 만든 두 재료가 **7B에선 측정상 둘 다 약하거나 부재**다:
  1. **싼 attn 보호** — 2.7b는 attn-decode를 54 SM(저배치)~68 SM로 포화시켜 싸게 보호. **7B attn-decode는 측정 가능한 모든 배치에서 54 SM로도 포화 안 됨(+67~97% 팽창)** → 보호하려면 >54 SM 필요 → prefill 환원 여지↓.
  2. ~~SM에 민감한 prefill 환원 부재~~ → **철회(§2.2)**: job 797524 실측 결과 7B prefill도 SM-민감(ssm 2.5×, attn 7.3×)이라 이 전제는 7B에서도 성립. 미스케일을 *prefill 무감각*으로 설명하지 않는다.
- 진짜 이유 **③ 7B는 decode-step 지배**(b8서 81층 합산 ~51ms/step) → goodput 병목이 *decode*이고, layer_aware가 개선하는 건 *prefill* 측이라 천장을 못 올린다(prefill이 민감해도 병목이 아님).
- 결론은 **①(보호 비쌈)+③(decode 지배)** 로 성립. 단 절대값(la/agnostic 1.06×)은 §3 synth/broken LUT 기반이라 미확정 — 깨끗한 7B E5 LUT(ssm-decode OSError 수정) 재실측이 절대 확증의 선택지(§4).

---

## 1. 무엇을 물었나 — 공간분할 7B(이미 닫힘)와 다른 질문

[review_checklist C1](review_checklist.md)은 "7B 회귀가 결정적인데 미실행"을 P0로 남겼다. 그 사이 두 갈래가 생겼다:

| 가설 | 7B 상태 |
|---|---|
| layer-type *공간 분할*(green_ctx가 two_stream 이기나) | **닫힘·음성** — [real_prefill §7](real_prefill_results.md): 2.7b의 +12.5% throughput 예외가 7B서 사라짐(zamba2_7b·falcon_h1_7b 0셀). |
| layer-type-aware *예약*(본 프로젝트의 살아있는 결론, §14/layer_aware_benefit_report) | **본 보고서가 처음 검증.** |

따라서 본 보고서는 *살아있는* 결론 한 가지 — "비싼 attn-decode 레이어만 SM 예약, 싼 ssm 레이어는 prefill에 SM 환원 → ~2× goodput" — 을 zamba2_7b(81층 = 13 attn + **68 ssm = 84% ssm**, 2.7b의 83%보다도 ssm-지배적)에서 묻는다. 메커니즘 가설대로면 ssm 비율↑이라 *더 큰* 이득이 나야 한다.

## 2. 측정된 7B 구조 — lever 전제 ①③ 부재 (②는 성립, §2.2 정정)

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

### 2.2 전제②(SM-민감 prefill 환원) — **정정: 7B prefill도 SM-민감하다(job 797524 실측)**

초판은 "7B prefill은 SM-무감각(1.02×)"이라 했으나 — **이는 *108→54 SM 구간만* 측정한 artifact였다.** `run_prefill_sm_sweep.py`(job 797524)로 prefill을 N-SM Green-Context 파티션에 **solo**로 14~108 SM 전 구간 측정한 결과(ratio = latency / latency@108SM):

![prefill SM-sensitivity](figures/prefill_sm_sensitivity.png)

| prefill | 108→54 SM | **108→14 SM(전 구간)** |
|---|--|--|
| zamba2_1.2b ssm | 1.2× | **2.0×** |
| zamba2_2.7b ssm | 1.2× | **2.2×** |
| **zamba2_7b ssm** | **1.2×** | **2.5×** |
| zamba2_7b attn | 2.0× | **7.3×** |

→ **prefill SM-민감도는 크기 거의 불변**(ssm ~2×, attn ~7×)이고 **7B도 충분히 민감**(오히려 약간 더). "1.02×"는 *모든 모델이 평탄한* 108→54 구간만 봤기 때문 — 민감도는 **<54 SM**에서 나타나며 7B도 예외 아니다(54→14 SM: 0.78→1.58ms = 2.0×). ∴ **전제②(prefill이 SM에 민감해 환원이 이득)는 7B에서도 성립** — 7B 미스케일을 *prefill 무감각*으로 설명한 부분은 **철회**한다.

**그럼 왜 7B는 여전히 미스케일인가** — 이유는 전제①·③로 좁혀진다: **①** attn-decode를 싸게 보호 못 함(§2.1, floor 68~108) → 예약이 GPU를 거의 다 먹음, **③** decode-step 지배(§2.3, ~51ms) → goodput이 *decode*에 묶여 prefill 개선(layer_aware가 주는 것)이 천장을 못 올림. 즉 7B에서 prefill은 *민감하지만 병목이 아니다* — 환원해도 decode-bound goodput이 안 오른다.

### 2.3 보강 — 7B는 decode-step 지배라 prefill lever의 leverage가 작다

full-model decode step ≈ Σ_layers(solo_decode). b8: 13·0.959 + 68·0.566 ≈ **51ms/step** (2.7b는 ≈ 28.6ms, [vllm_validation §3](vllm_validation.md)). 서버 capacity가 ~160 tok/s로 decode에 묶이고, layer_aware가 개선하는 prefill(TTFT) 측은 goodput 천장을 거의 못 올린다.

## 3. sim 보강 — 동일 합성법의 모델 간 대조 (방법 한계 명시)

7B는 floor 기반 protect를 측정 못 했으므로, **측정된 green_ctx(f=0.5/0.7)만으로 protect를 합성**해(`synth_7b_protect_lut.py`: 각 (층,배치)에서 decode를 solo의 15% 내로 두는 최소 측정 SM, 없으면 54) `run_layer_aware`를 돌렸다. **동일 합성법을 2.7b에도 적용한 대조**(real-E3 결과와 비교)로 방법의 신뢰구간을 박는다:

| (λ=0.5, SLO_ITL=100ms) | co | agnostic | layer_aware | la/ag | la ITL_p99 |
|---|--:|--:|--:|--:|--:|
| **real-E3 2.7b** (참값) | 752 | 629 | **1267** | 2.0× | 43.5ms |
| synth-2.7b (f-cap) | 752 | 88 | 221 | 2.5× | 178ms |
| synth-7b (f-cap, *폐기*) | 171 | 150 | 159 | 1.06× | 86.5ms |
| real-7b (797630, **E3 ssm floor 누락 artifact**) | 334 | 495 | 495 | 1.00× | 71.0ms |
| **real-7b (797832, E3 ssm floor 포함, 진짜 값)** | 337 | 273 | **496** | **1.82×** | 70.7ms |

> 위 두 real-7b 행의 차이가 artifact의 전부다: 797630은 E3 *ssm* floor가 없어 agnostic이 two_stream으로 degenerate(495=la) → 1.00×. 797832는 ssm floor를 측정해 agnostic이 제대로 ssm-decode를 예약(273, prefill 굶김) → la가 역전 → 1.82×.
*(goodput tok/s)*

**방법 한계(반드시 명시):** 합성 protect는 reservation을 측정 최대 54 SM로 cap하므로 decode 보호를 과소평가한다 — 그래서 synth-2.7b조차 절대 SLO를 망가뜨려(real la=1267>co를 synth는 la=221<co로 뒤집음) **"co가 7B서 이긴다"를 합성 sim 단독 근거로 주장하지 않는다.** 합성법이 *보존*하는 건 **layer_aware>agnostic 순서와 그 비율의 모델-스케일 추세**: la/agnostic 우위가 **2.7b ~2.0–2.5× → 7B 1.06×로 붕괴.** §2의 측정 구조와 같은 방향(7B에선 lever가 거의 작동 안 함).

## 4. 실측 — job 785877 완료, 부분 미완 (ssm-decode 커널 실패)

`slurm/measure_7b_layer_aware.sh` → **job 785877 COMPLETED**(SXM4 gpu, 4:19). 결과:

| 산출 | 상태 |
|---|---|
| 7B E3 decode floor(층별·배치별 포화 SM) | **✅ 측정됨** — `decode_floor_zamba2_7b_a100_sxm4_80gb.csv`(160 sweep + 9 floor). |
| E5 opt/protect LUT(`e5_sim_b8_opt`) | **⚠ 88/180 ok** — `dec=attn` 40/49(db=512만 OOM)이나 **`dec=ssm` 4/41(대부분 OSError)**. backend별 ssm-decode: two_stream **1/10**, green_ctx_protect **0/1**, green_ctx 2/20. |
| → 실측-LUT run_layer_aware | **무의미** — ssm-decode 셀이 거의 전부 결측 → agnostic·layer_aware가 동일 폴백으로 수렴(co 375 / ag 492 / **la 492**, la=ag). |

**즉 절대 goodput 확증은 ssm-decode 커널 실패로 미완.** OSError는 7B mamba_ssm(`selective_state_update`/`causal_conv1d`) 로드/실행 실패로 추정(2.7b는 성공 — 7B state 크기 또는 커널-캐시 이슈). 단 **이 실패는 §2 구조 논거(attn-decode 포화·prefill SM-민감도)와 무관**하다: §2는 `dec=attn` green_ctx 셀(18/20 ok)과 prefill_stream 측정에 근거하므로, **ssm-decode LUT 없이도 7B 미스케일 결론은 견고.**

**bracket(닫힘):** §2 두 전제 부정 + [real_prefill §8](real_prefill_results.md)의 full-model protect `no_room`이 양 끝을 묶는다 — reservation 적게(54) → decode 미보호(SLO 이득 없음), 많게(≥94, 7B 포화에 필요) → prefill 고갈(throughput 붕괴). 2.7b식 sweet-spot 없음. **재실측으로 절대값을 닫으려면** 실패한 dec=ssm 셀을 작은 db(OOM 회피)+mamba 커널 캐시 점검으로 다시 돌려야 하나, **구조 논거가 이미 결론을 닫으므로 선택적**이다.

## 5. 결론 — 검수 C1 닫힘 (반전: 7B도 스케일)

layer_aware 예약은 **테스트한 전 크기(1.2B–7B)서 goodput 이득** — la/agnostic **1.2b 1.37× / 2.7b 2.02× / 7b 1.82×**(전부 깨끗한 실측 LUT). **SLM 한정이 아니다.**
- 7B도 2.7b와 동일 메커니즘: ssm floor가 더 높아(b8 68 vs 54) agnostic이 prefill을 *더* 굶기고(agnostic 273 < co 337), layer_aware가 ssm 레이어 SM을 환원해 역전(496). 전제②(prefill SM-민감)가 살아있어 lever가 작동.
- ~~"7B 미스케일"~~ 은 **artifact였다**(§6): decode_ssm Triton-캐시 OSError → E3 ssm floor 누락 → green_ctx_protect degenerate → agnostic≡la=1.00×. job 797832(TRITON fix + E3 ssm floor)로 해소.
- **단, 공간-*분할* 7B 음성**([real_prefill §7](real_prefill_results.md))은 별개로 유효 — 그건 layer-type *분할*(死)이지 *예약*(生)이 아님.

검수 체크리스트: **C1 = 닫힘** — layer_aware 예약은 1.2B–7B 전 구간 양성. "SLM 한정" 스코프 **철회**.

## 6. 왜 결론이 두 번 뒤집혔나 — artifact 연쇄와 교훈

| 시점 | 값 | 무엇이 문제였나 |
|---|--|--|
| 초기(synth) | 1.06× | 합성 protect(54 SM cap) — 방법 한계 |
| 797630(real, E3 ssm 누락) | 1.00× | **E3 ssm floor 없음** → green_ctx_protect ssm 실패 → agnostic이 two_stream 폴백 = la |
| **797832(real, E3 ssm 포함)** | **1.82×** | **깨끗 — 진짜 값** |

**연쇄:** ① 785877의 decode_ssm이 홈 `~/.triton` 캐시 이슈로 OSError(zamba2_7b만 `n_groups=2`라 고유 커널) → ② E3 floor 실행이 ssm 행을 못 만듦(attn 9행만) → ③ E5 green_ctx_protect가 7B ssm서 `decode_floor=None`→no_room 실패 → ④ `run_layer_aware`의 `FullModelLM`이 ok 셀만 읽어 agnostic ssm을 two_stream으로 폴백 → ⑤ agnostic ≡ layer_aware → 1.00×. **`TRITON_CACHE_DIR`=scratch + E3 ssm floor 재측정으로 전부 해소.**

**구조 분석이 왜 빗나갔나(§2):** "decode-step 지배(③)" 논거는 틀렸다 — 실측상 7B는 DECODE/PREFILL 비율이 *더 낮다*(0.39 vs 2.7b 0.59), 즉 *상대적으로 prefill-heavy*. prefill이 더 무거우니 prefill lever(layer_aware)가 **더** 잘 먹히는 게 맞았다. §2는 artifact(1.00×)를 사후 합리화한 것이었다.

**교훈:** 불완전 LUT는 정책을 *조용히* degenerate시킨다(에러 없이 폴백). 결론 전에 **(a) 셀단위 ok 완전성**과 **(b) 두 정책이 실제로 다른 셀을 쓰는지**를 검증해야 한다. 본 7B 회귀는 그 검증 부재로 2회 오결론했다.

**전 모델 재검증(2026-06-22):** 같은 의심을 1.2B·2.7B에도 적용해 셀단위 점검 →
| 모델 | E5 ssm green_ctx_protect | la/agnostic | 판정 |
|---|--|--|--|
| 1.2b | ok@전 배치(b1–512) | 1.37× | ✅ 깨끗 |
| 2.7b | ok@전 배치(b1–512) | 2.02× | ✅ 깨끗 |
| 7b | ok@b1–16, fail@b32+(ssm floor 108) | 1.82× | ✅ 깨끗(작동 배치대) |

진단의 핵심: **degeneration은 정확히 1.00×(agnostic≡la)로 나타난다.** 1.2B/2.7B가 ≠1.00×인 것 자체가 agnostic이 실제로 다른 셀(green_ctx_protect, 전 배치 ok)을 쓴다는 증거. 7B만 ssm floor 누락으로 *전* ssm 배치가 폴백돼 1.00× 인위치를 만들었다. (attn green_ctx_protect가 세 모델 다 b16+서 fail하나, 두 정책이 attn에 *동일* 백엔드라 대칭 폴백 → 비율 무영향.) ∴ **1.2B/2.7B 재측정 불요, 7B만 artifact였고 수정 완료.**
