# Prefill-side multiplexing — 연구 아크 전사 (출발점 → 현재)

작성 2026-07-17. 정본 [CONSENSUS.md](CONSENSUS.md)의 **서사(narrative) 짝**: CONSENSUS가 *무엇이 참인가*라면 이 문서는 ***어떻게 거기 도달했나*** 다.
각 단계를 **① 시작 논의 → ② 촉발 지표 → ③ 해소 지표 → ④ 판정과 부정 사유(수치)** 로 기록한다.

**전체 요약 (한 문단)**: 출발 가설은 *"hybrid 모델의 층 타입(attention vs mamba)마다 SM 민감도가 다르니 층 타입별로 SM을 나누면 agnostic을 이긴다"* 였다.
이 가설은 **decode-side → coordinated → prefill-side** 순으로 세 번 후퇴하며 **전부 반증**됐고, 마지막 피난처였던 **시간축 동적 제어(SLO-aware)** 마저 **best-static을 못 넘었다**.
최종 상태: **PD 분리 자체는 이득(agnostic이 fused를 4모델 전부서 이김), 그 위의 모든 "똑똑한 정책"은 死.** 실전 = **peak decode 부하 기준 decode-heavy static 고정.**

**아크를 관통하는 두 축**:
- **핵심 개념 분리**: **Diff A(층 타입 간 *비용비*)** ≠ **Diff B(층 타입 간 *SM 민감도비* = 재배분의 진짜 lever)**. 가설이 요구한 건 Diff B인데 관찰되던 건 Diff A였다.
- **반복된 실패 유형**: **micro-measurement가 서빙을 예측하지 못함**(4회). 그때마다 결론이 뒤집혔다.

---

## S0. 출발점 — 창립 가설(layer-aware)과 4-모델 서빙 실증

| | |
|---|---|
| **① 시작 논의** | hybrid 모델은 attention 층과 mamba/SSM 층의 연산 특성이 다르다 ⇒ **층 타입별로 SM을 다르게 배분**하면 타입에 무지한 agnostic보다 낫다. **sim에서 生**이었다 |
| **② 촉발 지표** | sim의 **층 타입별 비용비(Diff A)** + sim goodput |
| **③ 해소 지표** | **실엔진 4-모델 서빙 goodput**(NemotronH / Zamba2 / Falcon-H1 / Granite-4, sglang v0.5.10) |
| **④ 판정** | ★**반증** — **pdmux agnostic v1이 4모델 전부에서 최적**. layer-aware는 **저부하=동등, 부하 시 열위→붕괴**. **Zamba2가 최악**: rate 3부터 **goodput 0**(사전 예측은 정반대인 "45/54 환원 대박"이었다). Falcon-H1은 단일 타입이라 정의상 ≡agnostic. `agnostic_v2`는 **decode에 실작업이 있으면 최악**(NemotronH·Granite) |

**부정 사유**: 예측이 sim/micro 기반이었고, 서빙 배치에서 재현되지 않았다.
**이 단계의 교훈(이후 반복 적중)**: **micro-measurement 2회 오도** — (1) GIL-client 과부하, (2) tiny-batch 민감도. Granite의 "agnostic_v2 최적"도 **decode-side micro-timing(작은 batch)이 서빙 batch의 SM 민감도를 과소평가한 아티팩트**였다.
⇒ **정책 주장은 반드시 서빙으로 실증한다**는 규율이 여기서 성립.

---

## S1. 1차 후퇴 — "layer-aware는 이진 strawman이다" (R0b)

| | |
|---|---|
| **① 시작 논의** | (사용자 지적) 지금까지의 la는 **이진 strawman**이고, agnostic은 **over-provisioning**이다. 층 타입별 **점진적(graduated)** SM map을 주면 결론이 달라질 수 있다 |
| **② 촉발 지표** | **agnostic이 prefill에 최대 74 SM을 남기는 잉여** |
| **③ 해소 지표** | `PDMUX_LA_SM_MAP`(per-type graduated) 하의 **decode TPOT** |
| **④ 판정** | **1차 결론 "knee≈84 · 잉여 없음" → 철회**(사용자 재지적). 구현이 decode를 **독립 green-ctx**에 두어 pdmux prefill과 **미조율(uncoordinated)** → 경합. **결정적 수치: agnostic *coordinated* decode 54 SM = 42ms vs *미조율* 54 SM = 121ms** |

**부정 사유**: 부정된 건 가설이 아니라 **내 측정**이었다 — 관찰된 "SM이 더 필요하다"는 **조율 결함의 아티팩트**였고, **잉여는 실재**했다. ⇒ 가설은 아직 살아있고, **조율된 구현으로 재판**해야 한다.

---

## S2. 2차 — coordinated per-type layer-aware를 **실제로 구현**해 재판 (R0c/R0d)

| | |
|---|---|
| **① 시작 논의** | 미조율이 문제라면, **조율된 per-type la를 진짜로 구현**해서 정면으로 판정하자 |
| **② 촉발 지표** | S1의 **42ms vs 121ms**(조율 격차) |
| **③ 해소 지표** | green-ctx event-loop 수술(`event_loop_pdmux_coord`)로 구현 후 **TPOT / goodput**. 정확성 게이트 통과 |
| **④ 판정** | ★**결정적 패배** — **TPOT 42 → 124ms**. tuned uniform split도 agnostic을 이김(d24 in3600 **+114%**, d44 in2000 **+43%**, regime 의존) |

**부정 사유 = (D) granularity의 실증**: decode 한 스텝이 **19개 윈도우로 파편화** → **sync 직렬화 + partition 핀 + overlap 감소**. 즉 **sub-step 단위 green-ctx 재분할의 비용이 층 타입 이득을 삼킨다.**
⇒ ★**창립 layer-aware 가설이 *작동하는 구현*으로 실엔진에서 최종 반증**됨. (3a Bullet/libsmctrl 경로는 driver 580에서 BLOCKED → 3b 수술로 우회.)

**추가 검증(사용자의 sim-vs-serving 회의에 답함)**: `PDMUX_LA_COORD_OPT`(GPU wait_stream + decode 핀 제거)로 **124ms의 약 절반이 substrate 기여**임을 확인 — **magnitude에 대한 회의는 옳았다**. 그러나 **부호(sign)는 robust**: 여전히 agnostic/tuned-uniform에 패배하며, 잔차는 **monolithic prefill의 단일-윈도우 오버랩** = **구조적·모델 독립**.

---

## S3. 3차 후퇴 — **prefill-side**로: "decode에선 죽었지만 prefill은?"

| | |
|---|---|
| **① 시작 논의** | decode-side la는 sub-step (D)drain + **cudagraph 비양립**으로 死. 그러나 **prefill은 원래 sub-step(≈18층 청크, `forward_split_prefill`)으로 돈다** ⇒ la의 자연스러운 무대. sim §14의 **layer-type 예약**도 生이었다 |
| **② 촉발 지표** | prefill의 청크 실행 구조 + sim §14 |
| **③ 촉발 지표(사용자 지적 2건)** | (1) **"L(시퀀스 길이)에 따라 민감도비가 달라질 것"** — attention은 O(L²)이니 긴 L에서 격차가 열린다. (2) **"B(배치)는 아예 재지도 않았다"** |
| **④ 해소 지표** | ★**Diff A와 Diff B의 분리**, 그리고 **(B,L) 2D knee**: **L 2k–32k × B 1–48** 전 격자 |

**판정 = 반증**:
- **Diff A(비용비)는 L에 따라 실제로 열린다** ⇒ 사용자의 직관은 *비용*에 대해선 옳았다.
- ★그러나 **Diff B(SM 민감도비) ≈ 1.0이 전 격자**(2k–32k × B1–48).
- **부정 사유**: **mamba SSD-prefill도 compute-bound**다. 두 층 타입이 **같은 자원(SM)에 같은 방식으로 민감**하면, 비용이 달라도 **재배분해서 얻을 것이 없다**. 가설이 필요로 한 건 Diff A가 아니라 **Diff B**였다.
- sim §14 예약도 Probe4에서 **fixed d16으로 degenerate** → 死.

⇒ ★**layer-type-aware, 全형태 死**(decode-side·coordinated·prefill-side·예약).

> ### ⚠️ S3 정정 (2026-07-17) — "Diff B ≈ 1.0 전 격자"는 과장이었다
> 원자료 재검토 + 시각화([`../results/prefill_knee/diffA_vs_diffB.png`](../results/prefill_knee/diffA_vs_diffB.png), 표 [`diffA_vs_diffB_table.md`](../results/prefill_knee/diffA_vs_diffB_table.md)):
> - **Diff A는 21× 진폭**(0.47× @L2k → 10.1× @L32k). 기전: **attn ~ L^1.68 vs mamba ~ L^0.61**, 교차점 ≈3k tok. **사용자 기억("짧으면 0.5×, 길면 2×")은 정확** — 실제로는 L=8k서 2.4×, L=32k서 10×까지 간다.
> - ★**Diff B는 L≥8000에서만 ≈1.0**(0.96–1.04). **L=2000에선 ≈1.35**(B=1 1.38 / B=48 1.34; B=4만 0.97) — mamba가 짧은 L에서 floor에 근접해 SM을 덜 먹는다. **lever는 L↓에서 열린다.**
> - ★**격자(L 2k–32k)가 실제 서빙 regime을 안 덮는다**: ShareGPT는 **mean 352 · p50 204 · p95 1042 tok, 98%가 L<2000**.
>
> **영향**: **서빙 수준 반증(S0·S2)은 무관하게 유효**(실 워크로드 직접 측정). 무너지는 건 **기전 서사**다 — **"lever가 없어서 죽었다"는 long-context 한정**이고 실 서빙 구간엔 **외삽**이다. 짧은 L에서 죽은 진짜 이유는 **(D) granularity**(S2에서 **TPOT 42→124ms**로 정량화)일 것이다.
> **부활 가능성 낮음**: Diff B>1이어도 (D) 비용을 넘어야 하는데, 짧은 L의 절대 stakes(per-layer 1–8ms)가 그 비용보다 작다. **열린 질문**으로 남긴다(L≈200–2000 Diff B 실측).

**같은 시기의 깨끗한 부수 결과(크기 추세)**: la/agnostic = **1.2B 1.37× / 2.7B 2.02× / 7B 1.82×** ⇒ **SLM 한정이 아니라 1.2B–7B 전 구간 이득**(commit `bdeca45`).
⚠️ 7B는 **"1.00× 미스케일"로 2회 오결론**했다가 반전 — **E3 ssm floor 누락으로 agnostic이 조용히 degenerate**한 아티팩트였다(`TRITON_CACHE_DIR` fix + 재측정 job 797832). **micro/설정 아티팩트가 결론을 뒤집은 3번째 사례.**

---

## S4–S8. 마지막 피난처 — **시간축**(SLO-aware 동적 PD-split)

**① 시작 논의**: 층 타입이 lever가 아니라면 남은 축은 **시간**이다. 현 pdmux는 split을 **decode batch size(SLO-blind)** 로만 조정한다 ⇒ **latency 피드백으로 split을 동적 제어**하면 static을 넘을 수 있다.
**② 촉발 지표**: pdmux가 SLO를 보지 않는다는 **구조적 관찰**. (v1–v6에서 isolation으로 **steady-state = static** 임을 먼저 증명, regime 자동수렴 확인.)

| 단계 | 시작 논의 (대부분 사용자 견인) | 해소 지표 | 판정 · 수치 |
|---|---|---|---|
| **Step D**<br>context-length feedforward | (사용자) **"prefill에서 L이 길면 attention에 SM을 더 주고 짧으면 덜 줘도 되지 않나? SLO를 더 잘 맞출 수 있다"** | `PDMUX_SLO_LFF`, cudagraph 중간부하 3-rep의 **tail + goodput** | **HD0** — lff tail < slo tail(**예측가능성만** 개선)이나 **goodput lff ≤ slo < d24**(static 지배). ★**진짜 발견(부산물)**: **SLO 컨트롤러가 TTFT-bound(cudagraph) 운영점에 오설계** — 주신호가 TPOT라 **prefill backlog 위기를 못 본다** |
| **Step E**<br>binding-first dual-slack | Step D의 부산물 정면 대응: prefill-slack(`split_prefill_batch` 최고령 age)과 decode-slack을 **대등화** | 중간부하/포화 goodput | 중간부하 **bind ≈ d24 ≫ slo**(flaw1 수정). 포화 **d24 > bind > slo** (**flaw2 = 진동** 신규 발생) |
| **Step F**<br>saturation-hold | (사용자) **"prefill이 남는 경우엔 anchor가 필요한 것 아닌가?"** / **"진동은 예측 가능한 것 아닌가?"** | 4-param(margin/deep/latch) sweep, predictive-hybrid | **전부 static 미달**. **부정 사유**: 포화에서 slack이 **split-coupled**이고 **boundary-hover** → **reactive 제어가 원리적으로 무력**. ⇒ **삼-regime 정리**: surplus·saturation → anchor / **single-binding만 동적 가치**. **layer-aware를 offline anchor(decode-floor) predictor로 재정의**(런타임 死) |
| **HE2**<br>"동적이 static을 *이기나*" | 매칭이 아니라 **우위**를 직접 판정 | 진짜 decode-bound phase(in256/o512) + static 전 sweep | ★**HE0** — 최적 split이 **static·불변**, 동적은 **best-static(d16 6.098)에 패**(bind 5.18–5.24, anchor 무관). **부정 사유**: 최적이 **prefill-heavy edge에 고정**(cudagraph decode가 robust해 binding 안 됨) ⇒ **좇을 이동이 없고 이동 자체가 순비용** |
| **granularity 반론** | (사용자) **"SM을 8개씩 움직이니 최적점을 쉽게 이탈하는 것 아닌가?"** | **±2SM 미세 그리드** + std/percentile | **step size 무관** — **d16 5.81±0.36 > bind-fine 5.35±0.73 ≈ bind-coarse 5.29±0.10**. 미세해도 격차를 못 닫음. (green-ctx는 미세 지원; sglang cudagraph **>7그룹 IndexError** 버그로 격자 상한 7) |

**이 시기의 자기 철회**: **§B의 "+18%"** 는 **no-cudagraph(비운영점) + vs d44(최적 아닌 static)** 이중 confound → **철회**.
**운영점 확정**: cudagraph-ON(decode wall 제거: TPOT 41→12ms, goodput ~1.5–2×↑) ⇒ 이전 no-cudagraph 수치는 **전부 하한**.

---

## S9. ★반전 — 실 trace가 "최적=d16·불변"을 뒤집다

| | |
|---|---|
| **① 시작 논의** | (사용자) **"tuned-fine으로 나온 최적이 d16인데, workload에 따라 최적점이 *움직이지 않는다*는 상황이 이해가 안 된다. 실제 trace나 흔한 서빙 벤치에서도 그런가?"** |
| **② 촉발 지표** | HE2의 "최적 = d16, split-flat" 결론이 **전부 synthetic 고정-길이**(range-ratio 1.0)였다는 점 |
| **③ 해소 지표** | **ShareGPT 실 trace**, 이어서 **시간 변화 trace**(rate 3↔12) |
| **④ 판정** | ★**내 결론이 반증됨** — **"최적=d16·불변"은 저-decode-부하 synthetic 아티팩트**. ShareGPT에서 **d16 붕괴: 1.056 vs d24 6.240 (5.9×)**. 변화 trace에선 **최적이 실제로 이동**(HI=d44) |

**그러나 상위 결론은 살아남음**: **static(d44) > dynamic(bind)** — 최적이 *움직이는데도* 동적이 못 이겼다.

★**기전 규명 = 얽힘(entanglement)**: prefill·decode가 **running batch(`max_running_requests`=48)와 KV를 공유** ⇒
**decode 굶김 → ITL↑ → running batch 정체 → prefill admission 차단 → TTFT 폭발.**
**결정적 수치**: **d16은 prefill에 92 SM(최대)를 주고도 TTFT 7.24s**, d24(prefill 84 SM)는 **1.21s**. ⇒ *"prefill을 굶기지 않았는데 prefill이 죽는다"* — prefill 성능이 **prefill의 SM이 아니라 decode의 SM**으로 결정된다.

⇒ **`최적 D_sm = max(모델 floor[attn-decode knee], 부하항[∝ λ×output_len])`**, 그리고 ★**비대칭**: decode **과다공급 = 저부하서 거의 무해** / **과소공급 = 고부하서 파국** ⇒ **최악 phase 기준 decode-heavy static이 두 phase 모두 안전 → 지배.**

---

## S10. 현재 세션 — 측정 자체를 의심하다

| 단계 | 시작 논의 | 해소 지표 | 판정 · 수치 |
|---|---|---|---|
| **분산 발견** | (사용자) "지표엔 p50/p95/p99나 std를 포함하라 — 없으면 견고성을 못 본다" | n=4 반복 | ★**stationary ShareGPT r8 벤치 폐기** — **switch=0인 static d24조차 5.282±1.302 (1/4 붕괴, min 3.102)**. ⇒ **"bimodal은 트랩 때문"·"d24는 ±0.039"·"게이트가 성능 회복" 전부 철회** |
| **n≥4 캠페인** | 유효 벤치(변화 trace)로 **HE0를 견고하게 재판** | 변화 trace n≥4 + switch/percentile | ★**HE0 견고 확정**: **d44 3.220±0.013 (n=4) > d34 3.171±0.025 > bind+GATE 3.132±0.019 (n=9) > d24 3.081 > slo 2.974 > bind no-gate 2.934±0.306 > d16 2.817**. 격차 **5.4σ** |
| **게이트의 정체** | 게이트가 정말 "지능적 제어"인가? | 서버 로그의 이동/거부 구조 | ★**one-way ratchet auto-tuner**: `2→3`(d24→d34) **1회 이동 후 prefill-ward 복귀를 113회 전부 거부**(`bs=47 ≥ 0.85×48` 상시 참) → **d34 영구 고정**. 수치 일치(**3.132 ≈ d34 3.171 − 0.039 정착비용**). ★**틀린 static에 조기 수렴** — 최적은 d44, 정지 규칙(tpot<51ms)이 **최적점 못 미쳐 발동** |
| **게이트의 가치** | 그럼 게이트는 무용한가? | 붕괴 빈도·분산 | ★**성능이 아니라 견고성**: no-gate **2.934±0.306, 1/4 붕괴(2.405)** → gate **3.132±0.019, 0/9, 분산 16× 타이트**. **d44가 ±0.013인 = 노이즈 없음이 증명된 벤치**에서의 붕괴이므로 **트랩은 실재하고 게이트가 막는다**(단 stationary bimodal은 그 증거가 못 됨) |
| **컨트롤러 CPU 비용** | switch=0인데 static 미달인 rep들 → "도는 것만으로 비용?" | ★**추론 말고 직접 계측**(`SLO-CTLCOST`) | ★**死** — **mean 32–36µs, max 267µs, 누적 ~34ms/≥1000call** = decode 한 step의 **0.9%**, wall clock의 **0.014%**. ⚠️gotcha: `_slo_decide_idx` **호출부가 둘** — `adjust_stream_groups`는 dead, **활성은 v7 이벤트루프 prefill-span** |
| ★**노이즈 추적** | "동일 config·프롬프트인데 goodput 2× 변동"의 정체 | §아래 | ★**노이즈가 아니라 메트릭 절벽** |

### S10-★ 노이즈의 정체 = **메트릭 절벽** (상세 [bench_noise_root_cause.md](bench_noise_root_cause.md))

1. **워크로드 변동 소거**: d24 4런의 `input_lens` **fingerprint 완전 일치**(`e0b77fd12329`). ⚠️ 나는 "seed 고정"이라 **주장만 하고 검증한 적이 없었다**(하네스는 `--seed`를 안 넘긴다 — bench_serving 기본 seed 덕에 우연히 맞았다).
2. ★**설명 대상은 3.5×가 아니라 3%였다**: throughput **6.36→6.14 (3%)**, ITL 28.0→30.2 (8%) — 그런데 **goodput 6.317→3.102 (2×)**.
3. ★**증폭기 = r8이 하필 TTFT ≈ SLO(3s) 경계에 앉은 것**. 과부하 큐(offered 8/s vs 용량 6.3/s)의 TTFT **평탄역이 3% 결손에 1.5s → 3.7s로 이동해 임계선을 넘는다** ⇒ **400/400 → 206/400**. goodput이 **CDF의 가장 가파른 지점**에서 평가되고 있었다.
4. **rate로 재현 — 불안정한 건 오직 경계 regime**: **rate 3 = 견고**(200/200 ×3) / **rate 8 = 불안정**(400, 400, 357, 206) / **rate 12 = 견고**(142, 141, 142).
5. ⇒ ★**GPU 클럭 throttling 가설 철회**(불필요) · ★**자원 격리 불필요**.

### S10-★★ 부산물 — **HE0의 구조적 이유**

| 정책 | **LO (rate 3)** | **HI (rate 12)** |
|---|---|---|
| d44 (decode-heavy) | 2.858 ± 0.005 | **3.924 ± 0.039** |
| d16 (prefill-heavy) | **2.861** | 2.737 |
| **정책 간 spread** | **0.067 (2.3%)** | **1.187 (43%)** |

★★ **차별의 ~95%가 과부하 phase에서 나오고, 저부하 phase는 split에 완전히 무관심**(양극단 d16 2.861 ≈ d44 2.858 = **구분 불가**).
⇒ **HE0는 컨트롤러의 실패가 아니라 워크로드의 구조적 귀결**이다:
- LO에는 **쫓아갈 최적점이 없다**(split이 무의미).
- HI의 최적(decode-heavy)은 **LO에서도 공짜**(과다공급 페널티 ≈ 0 = **비대칭의 정량 확인**).
- ⇒ **두 regime의 최적이 *충돌하지 않는다*** ⇒ **"항상 HI 최적을 쓰기" = decode-heavy static이 정의상 최선**이고, 동적은 거기 도달하는 **과도만 지불**한다.
- ★**동적 제어가 이기려면 regime 간 최적이 *충돌*해야 한다. 이 워크로드엔 그 구간이 없다.**
- d16만 **HI(2.737) < LO(2.861)** — 부하 하에서 유일하게 **붕괴**(얽힘 기전).

**부산물(하네스 버그)**: 변화-trace 분석기가 good은 3라운드 합산·분모는 `dur=max(dur,d)` ⇒ **goodput 3× 부풀림**(수정 `f921ae8`). **순위 완전 보존**이라 정책 결론은 무사하나, 구 보고값(9.649 등) 인용 시 **÷3**.

---

## 아크 요약 — 가설은 어떻게 죽었나

```
창립 가설: "층 타입별 SM 배분이 agnostic을 이긴다"   [sim에서 生]
  │
  ├─ S0  decode-side, 4모델 서빙          → 死  (agnostic이 4/4 승; Zamba2는 rate3부터 goodput 0)
  │        └ 부정 사유: sim/micro가 서빙 배치를 예측 못함
  ├─ S1  "이진 strawman이다"(사용자)      → 재개  (내 측정이 미조율 아티팩트: 42ms vs 121ms)
  ├─ S2  coordinated per-type, 실제 구현   → 死  (TPOT 42→124ms)
  │        └ 부정 사유: (D) granularity — decode 19윈도우 파편화 → sync 직렬화·핀·overlap 감소
  ├─ S3  prefill-side + (B,L) knee        → 死  (Diff B ≈ 1.0 전 격자)
  │        └ 부정 사유: mamba SSD-prefill도 compute-bound ⇒ 비용(Diff A)은 달라도 lever(Diff B)가 없음
  │
  └─ 마지막 피난처: 시간축 동적 제어
       ├─ S5 Step D (L feedforward)       → HD0 (goodput lff ≤ slo < d24)
       ├─ S6/S7 Step E/F (binding·anchor) → static 매칭이 천장 (포화서 reactive 무력)
       ├─ S8 HE2 + granularity(±2SM)      → HE0 (d16 5.81±0.36 > bind 5.35±0.73)
       ├─ S9 실 trace(사용자)             → 내 "최적=d16·불변" 반증. 그러나 여전히 static > dynamic
       │      └ 기전: 얽힘 (d16은 prefill에 92SM 주고도 TTFT 7.24s)
       └─ S10 n≥4 + 노이즈 추적           → HE0 견고(5.4σ) + ★구조적 이유(LO spread 0.067 vs HI 1.187)
                                             ⇒ 두 regime의 최적이 충돌하지 않음 = 적응할 대상이 없음
```

**살아남은 것**: **PD 분리 자체**(agnostic이 fused를 4모델 전부서 승), **cudagraph 운영점**, **얽힘 기전**, **비대칭**, **부하 의존 최적**.
**실전 권고**: **peak decode 부하 기준 decode-heavy static split 고정.** 동적 제어 불요.
**layer-aware의 유일한 잔존 형태**: **offline anchor(decode-floor) predictor** — 런타임 정책으로는 死.

## 방법론 교훈 (이 아크가 실제로 가르친 것)

1. ★**micro-measurement는 서빙을 예측하지 못한다** — **4회** 결론을 뒤집었다(GIL-client 과부하 / tiny-batch 민감도 / E3 ssm floor 누락 / decode-side micro-timing). **정책 주장은 서빙으로만 확정.**
2. ★**"비용이 다르다"(Diff A) ≠ "재배분할 수 있다"(Diff B)** — 아크 전체가 이 혼동 위에 세워졌었다.
3. ★**베이스라인의 분산을 먼저 재라** — switch=0인 static이 ±1.302를 내는 벤치에서 정책을 비교하고 있었다.
4. ★**임계 지시함수 지표(goodput=TTFT≤SLO)는 절벽을 피해 측정하라** — 경계 regime에서 **3% 섭동이 2× 신호로 증폭**된다. 과부하 구간의 threshold-goodput은 **런 길이 의존 = ill-posed**.
5. ★**추론하지 말고 계측하라** — 컨트롤러 CPU 비용 가설은 **한 줄 계측(32µs)** 으로 즉사했다. 그 전까지는 goodput 차이에서 *추론*하려다 노이즈에 막혔다.
6. **사용자의 반증 압력이 거의 매 단계 방향을 바꿨다** — S1(strawman), S3(L·B 미측정), S5(L feedforward), S7(anchor·예측가능성), S8(SM step), **S9(실 trace = 가장 결정적)**, S10(std·switch 분리).
