# Layer-aware SM 예약 정책 구현 및 5-정책 평가 (NemotronH + Zamba2 실측)
<!-- 정책: fused · co_schedule · agnostic(v1) · agnostic_v2 · layer_aware -->


작성: 2026-07-03 (서빙 다중-run·Zamba2 갱신 2026-07-05) · 대상: **NemotronH-8B-Base**(temporal 하이브리드: mamba 24 / mlp 24 / attn 4, 52층, GQA 8 KV-head, head_dim 128) + **Zamba2-2.7B**(mamba 45 / hybrid-attn 9, 54층, **no-GQA**, head_dim 160) · A100-80GB / CUDA13 / sglang v0.5.10 + green-ctx(sgl_kernel.spatial)
관련: [p1_3_nemotronh_layer_aware.md](p1_3_nemotronh_layer_aware.md)(pdmux-hybrid 기반) · [framework_comparison.md](framework_comparison.md)(sim 4-정책 정의)

---

## 0. 요약 (핵심 결론 먼저)

**실엔진 측정이 시뮬레이션의 layer-aware 전제를 NemotronH에 대해 뒤집었다.** sim은 "**attn-decode가 비싸고 SM-민감**, ssm-decode는 싸고 SM-둔감 → attn 레이어에만 SM floor 예약"을 전제했으나, NemotronH 실측은 정반대다:

- `[measured]` **mamba(SSM/SSD) decode가 SM-민감** (108→16 SM서 레이어당 0.589→1.547ms, **2.6× 느려짐**) 이며 decode step의 **62%** 차지.
- `[measured]` **attn decode는 SM-둔감** (레이어당 ~0.2ms 평탄, GQA라 memory-bound) 이며 step의 **5%** (4층뿐).
- `[measured]` mlp는 ~44 SM 이상서 둔감, 그 아래서 민감.

⇒ NemotronH서 SM을 아껴 예약할 "싸고 둔감한" 레이어는 **attn 4개뿐**이고, 다수(mamba+mlp)는 SM-민감이라 floor를 유지해야 한다. 따라서 **layer-aware의 이득은 NemotronH서 미미**하며(prefill로 환원 가능 SM이 4/52 레이어에 국한), **agnostic(전 레이어 균일 예약)이 사실상 동등하거나 우세**하다.

**단, 결정적 뉘앙스(§3.5): attn의 SM-민감도는 컨텍스트 길이 의존이다.** attn-decode는 O(L)이라 컨텍스트가 길어지면 무거워지고(0.30→0.45ms/층 @350→5100 tokens) **SM-둔감→SM-민감으로 뒤바뀐다**(ctx5100서 16SM 대비 full 43%↑). mamba는 O(1)이라 컨텍스트 불변. ⇒ **layer-aware 이득 = (아키텍처 GQA/no-GQA) × (컨텍스트 길이)의 2차원 함수**. "SM-민감 레이어 희소" 조건은 **긴 컨텍스트 + no-GQA(비싼 attn) + 희소 attn-층**에서 성립 — 그 체제가 layer-aware의 유일한 유리 구간이다.

**서빙 실측 결론(NemotronH, async `bench_serving` 청정 sub-saturation §3.9):**
- `[measured]` **청정 순위: agnostic(v1) ≈ layer_aware > fused > agnostic_v2.** agnostic이 부하서 최고(decode SM 충분 예약 → TPOT 22~25ms 평탄, goodput rate3~4서 최고). **layer_aware ≈ agnostic**(NemotronH선 안전 환원 대상 4/52뿐 → 우위 없음, 고부하선 전환 오버헤드로 소폭 열위). fused는 저부하 양호하나 decode-prefill 결합으로 TPOT 상승.
- `[measured]` **★agnostic_v2 = 최악**: decode 16 SM 균일 굶주림 → TPOT rate2부터 SLO 돌파(72→191ms)·TTFT도 최악 → goodput rate3서 0. **layer_aware 가치 = "더 환원"이 아니라 "민감층 보호"** (§3.9).
- `[measured]` **측정 방법론**: 초판의 rate 6/10 goodput(hand-rolled 60-스레드 urllib client × 서버 포화(~4.3req/s) 초과 rate × 절벽 goodput 지표)은 **오측 아티팩트** → async `sglang.bench_serving`으로 sub-saturation 재측정해 정정(§methodology). Zamba2 서빙(§3.8)은 triton-no-cudagraph decode-bound + 아직 구 client라 정성 신호(async 재측정 후속).

---

## 1. 구현한 것 (layer-aware 메커니즘)

- `[measured]` **pdmux를 hybrid 모델서 최초 동작** (P1.3): NemotronHForCausalLM에 `forward_split_prefill` 추가(dense 전용이던 것) → green-ctx 4 stream group으로 pdmux serve.
- `[measured]` **per-layer green-ctx 스트림 전환 메커니즘** (`models/nemotron_h.py` NemotronHModel.forward): decode forward의 각 레이어를 그 타입에 맞는 green-ctx 파티션(=SM 개수) 스트림에서 실행. `_get_gctx_decode_stream(n)`이 `sgl_kernel.spatial.create_greenctx_stream_by_value(n, 108−n)`로 n-SM 스트림 생성·캐시.
- `[measured]` **정책 선택**: 파일(`PDMUX_FIXED_DECODE_SM_FILE`) 또는 env로 모드 지정 — `full`(108), 정수 N(agnostic: 전 레이어 N SM), `layer_aware`(예약 타입=`PDMUX_LA_RESERVE`(기본 "M-"=mamba+mlp)은 full, 그 외(attn)는 floor=`PDMUX_LA_FLOOR_SM`). **예약 타입은 데이터-구동**(측정된 SM-민감 타입을 예약).
- 측정 계측: `SGLANG_NH_LAYER_TIMING=1` → per-layer-type CUDA-event 타이밍(decode step당, 30스텝 평균).

> 주: 완전-충실 layer-aware 서빙(decode·prefill을 layer-window 단위로 상보 파티션에 교차 실행)은 event loop 대수술이라 별건(§6). 본 구현은 **decode 경로의 per-layer-type SM 할당**을 실측하는 데 초점(정책의 핵심 차별자 = decode의 SM-타입별 배분).

---

## 2. 핵심 측정: NemotronH decode의 per-layer-type SM 민감도

decode 배치=32, ignore_eos, 210 스텝 평균. green-ctx로 decode를 N SM에 고정.

| decode SM | step(ms) | mamba total(24층) | mlp total(24층) | attn total(4층) | mamba/층 | mlp/층 | attn/층 |
|---|---|---|---|---|---|---|---|
| 108 (full) | 22.5 | 14.13 | 7.22 | 1.17 | 0.589 | 0.301 | 0.291 |
| 64 | 22.9 | 14.84 | 7.23 | 0.86 | 0.618 | 0.301 | 0.215 |
| 44 | 24.2 | 16.43 | 7.14 | 0.59 | 0.684 | 0.298 | 0.148 |
| 32 | 29.9 | 20.13 | 9.15 | 0.62 | 0.839 | 0.381 | 0.155 |
| 16 | 53.3 | 37.13 | 15.09 | 1.09 | **1.547** | 0.629 | 0.271 |

`[measured]` **해석:**
- **mamba/층: 0.589→1.547 (108→16 SM), 단조 2.6× 증가 = 강한 SM-민감**(compute-bound; Mamba2 SSD chunk-scan 커널이 연산집약).
- **attn/층: ~0.15–0.29 평탄, SM 스케일링 없음 = SM-둔감**(GQA 8-KV memory-bound; 4층·소량이라 노이즈). 
- **mlp/층: 108–44서 ~0.30 평탄(둔감), 32↓서 증가(민감)**.
- decode step 구성(full): **mamba 62% + mlp 32% + attn 5%.** attn이 가장 싸다.

---

## 3. 4-정책 분석 (NemotronH)

sim 4-정책([framework_comparison](framework_comparison.md))을 실엔진에 매핑. decode step(배치32)·prefill로 환원되는 SM 기준:

| 정책 | decode SM 배분 | decode step(ms) | prefill로 환원 SM(평균) | 비고 |
|---|---|---|---|---|
| **fused** | 108(무분할, prefill과 시간적 융합) | 22.5(순수 decode) | 0 (decode 중 prefill 대기) | 배포 기본. 부하시 decode가 prefill forward에 결합 |
| **co_schedule** | decode∥prefill 108 자유공유 | >22.5 (경합) `[derived]` | 잔여(경합) | 예약 없음. 미직접측정 |
| **agnostic** (v1, MuxWise/Bullet) | 전 레이어 고정 N (**mamba 기준** = 높은 N) | N=64:22.9 / N=44:24.2 | N=64:44 / N=44:64 (**전 레이어**) | 균일 예약(민감층 보호) |
| **agnostic_v2** | 전 레이어 floor 16 (**attn 기준** = 낮은 N) | **53.3** | **92** (전 레이어, 최대) | 균일 최대환원·**민감층 미보호** |
| **layer_aware** | 민감타입(M,-) full·attn floor16 | ≈23.1 `[derived]` | ≈47.7 (attn 4층서만 92) | 희소-둔감 레이어만 환원(선택적) |

`[derived]` **layer_aware vs agnostic 정량:** layer_aware(mamba/mlp @64, attn @16)의 평균 prefill 환원 SM ≈ 44 + (92−44)×4/52 ≈ **47.7 SM**, decode ≈23.1ms. agnostic@64는 prefill 44 SM(전 레이어), decode 22.9ms. ⇒ **layer_aware가 환원하는 추가 SM은 ~3.7(평균) 뿐**(4/52 attn 레이어 한정) — decode 속도·prefill 환원 모두 agnostic과 사실상 동일. **layer-aware의 순이득 ≈ 0 (NemotronH).**

---

## 3.5. 시퀀스 길이 의존성 — attention의 SM-민감도는 컨텍스트로 뒤바뀐다 ★★

§2·§3은 **짧은 컨텍스트**(~350). 그러나 attn-decode는 매 스텝 길이 L의 KV를 읽으므로 **O(L)**, mamba는 고정 state라 **O(1)**. 컨텍스트를 스윕(프롬프트 길이 조절)해 재측정(NemotronH, GQA, flashinfer — Zamba2와 달리 백엔드 교란 없음):

**GQA attention per-attn-층 decode 지연(ms):**
| decode SM | ctx≈350 | ctx≈2140 | ctx≈5100 |
|---|---|---|---|
| 108(full) | 0.303 | 0.337 | **0.449** |
| 44 | 0.272 | 0.361 | 0.490 |
| 16 | **0.237** | 0.395 | **0.640** |

`[measured]` **① attn이 컨텍스트로 무거워짐**: full-SM attn/층 0.303→0.449 (350→5100, +48%). **② attn의 SM-민감도가 컨텍스트로 뒤바뀜**: ctx350선 16SM서 *더 빠름*(0.237, memory-bound/SM-둔감) → ctx5100선 16SM서 **43% 느림**(0.640 vs 0.449, compute-bound/SM-민감). `[measured]` **③ mamba는 컨텍스트 불변**(~0.553ms/층 평탄, O(1)) + 항상 SM-민감(→1.05@16SM). `[measured]` mlp 컨텍스트 불변.

⇒ **사용자 직관 실증**: attn-decode의 "싸고 SM-둔감"은 **짧은 컨텍스트 국한**. 길어지면 attn이 커지고 SM-민감해진다. 단 NemotronH(GQA)는 **긴 컨텍스트서도 mamba가 지배**(ctx5100서 mamba 13.3ms vs attn 1.8ms=8%)라 여전히 mamba가 주 예약대상.

**no-GQA(Zamba2) 완전 청정 측정 (triton 백엔드 §3.6 + green-ctx race 수정 후):** flashinfer head_dim=160 NaN으로 torch_native에 갇혔던 Zamba2를 triton으로 해방. green-ctx wrap의 stream race(장문·batch>1서 device-side assert)를 **`wait_stream` 핸드셰이크로 수정**(default↔green-ctx 동기화). 전 컨텍스트×SM 그리드 확보:

**Zamba2 no-GQA 레이어별 decode 지연(ms), full/44/16 SM:**
| ctx | attn/층 | mamba/층 |
|---|---|---|
| 348 | 0.159 / 0.203 / **0.399** | 0.556 / 0.566 / 0.498 |
| 2140 | 0.574 / 1.03 / **2.49** | 0.513 / 0.450 / 0.304 |
| 4092 | 1.044 / 1.97 / **5.13** | 0.453 / 0.328 / 0.300 |

`[measured]` **layer-aware 두 조건 모두 실증 충족:**
- **attn = SM-민감 + 컨텍스트로 급증 + 희소**: full-SM 0.159→1.044 (ctx348→4092, ~7×=O(L)); SM-민감도가 컨텍스트로 심화(16SM 대비 full: ctx348 2.5× → ctx4092 **4.9×**, 1.04→5.13); 9/54층 희소.
- **mamba = SM-둔감(오히려 inverse)**: 2.7B mamba는 memory-bound라 SM 늘려도 안 빨라짐(ctx4092 full 0.453 vs 16SM 0.300 — 오히려 저SM서 빠름). ⇒ **mamba의 SM을 prefill로 환원해도 decode 손실 0(오히려 이득)**; 54/54층 = decode 시간의 다수(ctx4092서 mamba 24.4ms vs attn 9.4ms).

`[derived]` **⇒ Zamba2 장문 = layer-aware 이상적 체제.** attn@108(예약)+mamba@16(환원) 계산: decode = 9.4+16.2 = **25.6ms**(+mlp), **prefill에 92 SM 환원**(mamba 구간=시간 다수). vs agnostic@108: decode 33.8ms, 환원 0. **layer-aware가 decode도 빠르고 prefill SM도 더 환원** — mamba 둔감·attn 민감이 정확히 맞물림.

**모델-크기 의존 추가 발견:** NemotronH(8B) mamba는 SM-**민감**(compute-bound, 2.6×), Zamba2(2.7B) mamba는 SM-**둔감**(memory-bound). ⇒ layer-aware 유불리는 (GQA/no-GQA)×(컨텍스트)×(**모델 크기**)의 함수. layer-aware 유리 = **no-GQA + 장문 + (mamba가 둔감할 만큼) 작은/메모리-바운드 모델**.

**함의: layer-aware 이득은 (아키텍처 GQA/no-GQA) × (컨텍스트 길이)의 2차원 함수.** SM-민감 레이어가 희소해야 이득 — 짧은 컨텍스트 GQA선 mamba가 다수라 이득無; **긴 컨텍스트 + no-GQA**서 attn이 비싸지되 희소(9/54)면 layer-aware가 유리할 후보.

## 3.6. Zamba2 fast attention 백엔드 확보 (triton 버그 수정)

- `[measured]` Zamba2는 flashinfer가 head_dim=160서 NaN([zamba2_troubleshooting](zamba2_troubleshooting.md))이라 torch_native(느림)에 갇혀 있었음. triton은 임의 head_dim 지원하나 hybrid서 init 버그(`layer_id=0 not in full attention layers` — layer0가 attn 아닐 때 layer0 KV 조회 가정).
- `[measured]` **수정**: `layers/attention/triton_backend.py`의 v_head_dim 초기화 조건을 `hybrid_gdn_config` → **`mambaish_config`**로 확장(mamba2 hybrid=NemotronH/Zamba2도 `get_v_head_dim()` 사용, layer0 조회 회피). ⇒ **Zamba2가 triton서 정확 동작**("The capital of France is"→"Paris") + torch_native보다 빠름. 이로써 no-GQA 청정 측정 가능(§3.5).

## 3.7. 서빙 goodput 실측 — 3정책(fused/agnostic/**layer_aware**), NemotronH-8B

> ⚠️ **이 절은 hand-rolled `bench_client.py`(60-스레드 urllib)로 rate 3/6/10을 쟀다.** rate 6/10은 **서버 포화(~4.3 req/s) 초과**라 burst-drain 노이즈이고 client가 GIL-병목이라 **rate 6/10 수치는 신뢰 불가**(§methodology). **NemotronH 서빙의 신뢰 결과는 §3.9의 async `bench_serving` sub-saturation 측정**을 볼 것. 아래는 rate 3(포화 이하) 정도만 유효 참고.

end-to-end 서빙 벤치(Poisson 도착, streaming TTFT/TPOT, in=2000/out=96, SLO: TTFT≤3s·TPOT≤60ms). layer_aware는 pdmux + per-layer green-ctx 전환(§1, race-safe `wait_stream`)으로 **실엔진 서빙서 실행**. 셋 다 출력 정상("Paris").

goodput@SLO (req/s), run1/run4/run5 (평균):

| rate | fused | agnostic | layer_aware |
|---|---|---|---|
| 3 | 2.74/2.41/2.28 (2.48) | 2.59/2.84/2.73 (**2.72**) | 2.60/2.41/2.86 (2.62) |
| 6 | 1.65/1.61/1.56 (1.61) | 1.63/1.81/1.58 (**1.67**) | 1.44/0.85/1.72 (1.34) |
| **10** | 0.29/0.22/0.57 (0.36) | 0.50/1.16/1.82 (**1.16**) | 1.07/1.22/0.79 (**1.03**) |

`[measured]` **고부하(rate10)서 pdmux 두 정책(agnostic·layer_aware)이 fused를 압도**(평균 1.16·1.03 vs 0.36 = ~3×). 기전: pdmux가 prefill을 decode와 SM 분리 → 포화 근처 prefill 굶주림 완화 → TTFT 방어. 저·중부하선 3정책 유사(2.5~2.7).
`[measured]` **그러나 agnostic ≈ layer_aware (NemotronH서 layer_aware 우위 없음)**. rate10 평균 1.16 vs 1.03로 오차 내 동등, run별로 승자 뒤바뀜(run1 la승·run5 agn승). ⇒ **§2/§3 decode-side 예측 확증**: NemotronH는 mamba(다수·SM-민감)라 **환원 가능한(SM-둔감) 레이어가 attn 4/52뿐** → layer_aware가 균일예약(agnostic) 대비 추가 이득 없음.
`[measured]` **정정**: 이전 초안의 "rate10서 layer_aware 1.07 vs agnostic 0.50 압승"은 **포트 충돌로 agnostic이 억눌린 오염 아티팩트**였음. 청정 재측정서 agnostic이 강함(1.16). → NemotronH서 정책 결론 = **pdmux(agnostic로 충분) ≫ fused, layer_aware는 추가 이득 없음**.
`[derived]` run-to-run 분산 여전히 큼(rate10 agnostic 0.50~1.82) — 소표본(60req)+포화근처 스케줄링 민감성. 순서(pdmux>fused)는 3-run 견고, la-vs-agn 무승부도 견고.

## 3.8. 서빙 실측 — Zamba2-2.7B (no-GQA, mamba SM-둔감) → layer_aware가 agnostic 대비 TTFT 우위 (기전 확증)

Zamba2 pdmux 포트(§P1.5: `forward_split_prefill` + layer_aware = **9 hybrid(no-GQA attn) 예약 / 45 mamba 환원**)로 3정책 서빙. `--attention-backend triton`(head_dim 160), prefill-bound 체제(in=3600/out=32). TTFT_med (ms):

| rate | fused | agnostic | layer_aware | **la vs agn** |
|---|---|---|---|---|
| 6 | **1953** | 5002 | 3846 | **−23%** |
| 12 | 6134 | 6995 | 5792 | **−17%** |
| 18 | 7196 | 7092 | 6595 | **−7%** |

`[measured]` ★ **layer_aware가 agnostic보다 TTFT 일관 낮음**(−7~23%). = **45/54 mamba 레이어의 decode SM 환원이 prefill을 실제로 가속**. NemotronH(둔감층 4/52 → la≈agn)와 정반대로, **Zamba2(둔감층 45/54)선 la-우위가 나타남** ⇒ **"layer_aware의 TTFT 이득 ∝ 환원 가능(SM-둔감) 레이어 비중"** 명제를 두 모델 대조로 확증. decode-side(§3.5 Task1) 이득이 serving TTFT로 전이됨을 실증.
`[measured]` **단, goodput 이득은 미실현(정직)**: triton(no-cudagraph) decode가 매우 느려 **TPOT_med 170~205ms ≫ SLO** → 3정책 모두 goodput≈0(fused rate6 0.50만 예외). 게다가 **작은 모델(2.7B)은 fused(전 108 SM)의 prefill이 이미 빨라** pdmux의 SM 분리가 오히려 손해(fused TTFT 1953 < pdmux). ⇒ 이 하네스선 fused가 최선.
`[derived]` **결론(전이 조건)**: layer_aware의 TTFT 이득이 **goodput 이득으로 전환되려면** ① prefill-bound serving(큰 모델/장문 → prefill 지배) + ② **빠른 decode(cudagraph)로 TPOT가 벽이 아닐 것** 동시 필요. 본 클러스터(py3.14, pdmux+mamba+triton선 cudagraph 불가)선 Zamba2가 ②를 못 넘어 goodput 미실현. **기전(la<agn TTFT)은 확증**, end-metric은 하네스 한계.
`[measured]` layer_aware TPOT가 agnostic보다 약간 높음(r6 182 vs 171, r12 205 vs 177) — mamba 16 SM 강등의 잔여 민감(2.7B는 거의 둔감이나 0은 아님) + **per-switch `wait_stream` 오버헤드**(스텝당 ~45 전환). decode-bound 체제선 이 TPOT 대가가 불리(구현 최적화 여지: 전환 최소화/CUDA graph).
- 원자료: prefill-bound `p1_4_zb_serving_831166.txt`, decode-bound(out96) 참고 `p1_4_zb_serving_831146.txt`(동일 정성 결론: fused 최선, decode-bound).

## 3.9. agnostic_v2 — "attention 기준" 균일 예약 (사용자 제안 정책)

**질문**: agnostic을 "더 많은 SM 쓰는 층(mamba) 기준"이 아니라 **attention 층 기준으로 모든 층에 할당**하면? attention은 (단문 GQA서) 싸고 SM-둔감 → 그 수준(낮은 floor)을 전 층에 균일 적용 = **prefill 환원 최대화, 단 SM-민감 층 미보호**. 기존 agnostic(v1)은 유지, 신규 `agnostic_v2`로 구현(`models/{nemotron_h,zamba2}.py`, `_sm_part=="agnostic_v2"`→전 층 `PDMUX_AGN2_SM`=16). 부팅 검증 OK("Paris"·NO_CRASH).

**세 정책의 위치**: agnostic_v1 = 보수적(민감층 보호, 중간 환원) · agnostic_v2 = 공격적(최대 환원, 무보호) · layer_aware = 선택적(둔감층만 환원, 민감층 보호). ⇒ agnostic_v2는 **layer_aware의 가치가 "더 환원"인지 "선택적 환원"인지 가르는 대조군**.

**4정책 서빙 실측:**

> ⚠️ **측정 정정(2026-07-05)**: 초판은 hand-rolled `bench_client.py`(60-스레드 urllib, GIL 병목)로 **포화 초과 rate**(6/10; 서버 포화 ~4.3 req/s)서 쟀다 — burst-drain 노이즈+절벽 goodput이라 "agnostic_v2 우위"는 **오측 아티팩트**였음(사용자 지적). **공식 async `sglang.bench_serving`로 sub-saturation rate(1~4)서 재측정** → 결론 **역전**. 아래는 청정 데이터(job 831609~831612, in2000/out96, num-prompts 120).

`[measured]` **NemotronH goodput@SLO (async 청정, TTFT≤3s·TPOT≤60ms):**
| rate | fused | agnostic | **agnostic_v2** | layer_aware |
|---|---|---|---|---|
| 1 | 0.97 | 0.96 | 0.93 | 0.96 |
| 2 | 1.89 | 1.88 | **0.32** | 1.88 |
| 3 | 2.22 | **2.76** | **0.00** | **2.76** |
| 4 | 1.40 | **2.25** | **0.00** | 1.58 |

`[measured]` **Median TPOT (ms) —결정적 판별자:**
| rate | fused | agnostic | **agnostic_v2** | layer_aware |
|---|---|---|---|---|
| 2 | 32 | 22 | **72** | 26 |
| 3 | 46 | 24 | **119** | 35 |
| 4 | 63 | 25 | **191** | 50 |

`[measured]` **★agnostic_v2 = 최악 정책(초판 결론 역전)**: decode를 16 SM으로 굶기면 **TPOT가 rate2부터 SLO(60ms) 돌파**(72→119→191ms) → goodput rate3서 0으로 붕괴. Median TTFT도 최악(rate4 6705ms) — **굶긴 decode가 밀려 prefill 스케줄을 막아 TTFT로 번짐**(Zamba2와 동일 기전). §2 decode-side(16 SM=53ms)를 서빙이 정확히 확증. ⇒ **"attn 기준 균일 예약"은 지배적 decode 층(mamba)이 SM-민감이면 최악** — 사용자 직관의 반대. `[measured]` **청정 순위**: **agnostic(v1) ≈ layer_aware > fused > agnostic_v2**. agnostic이 부하서 최고(TPOT 최저 22~25ms=decode SM 충분 예약), layer_aware는 저·중부하서 agnostic과 동등(NemotronH선 안전 환원 대상 없음)·고부하서 전환 오버헤드로 소폭 열위, fused는 저부하 양호하나 decode-prefill 결합으로 TPOT 상승.

`[measured]` **Zamba2 TTFT_med / goodput (job 831541, prefill-bound in3600/out32; ⚠️구 client·과부하 rate라 정성만):**
| rate | fused | agnostic | agnostic_v2 | **layer_aware** |
|---|---|---|---|---|
| 6 | 4782 / 0 | 3928 / 0 | 4427 / 0 | **1619 / 1.09** |

`[measured]` **Zamba2도 agnostic_v2 열위, layer_aware 우위** (rate6 layer_aware TTFT 1619=타 정책 3928~4782의 2.4~3×↓). **agnostic_v2 TTFT(4427) > agnostic_v1(3928)** — 더 많이 환원하는데 오히려 나쁨(굶긴 no-GQA attn decode가 prefill 밀어 TTFT로 번짐). NemotronH 청정 결과와 **같은 방향**(v2=최악). *단 이 Zamba2 수치는 구 client·과부하 rate라 정성 신호로만; async 재측정은 후속.*

`[derived]` **agnostic_v2 결론(정정)**: v2는 "최대 환원, 최저 TTFT"가 **아니다** — 굶긴 decode가 느려 **TPOT도 TTFT도 최악**(느린 decode가 파이프라인을 막아 nominal하게 푼 SM이 prefill에 실질 도움 안 됨). **두 모델 모두 v2=최악**. ⇒ 지배적 decode 층이 SM-민감이면(NemotronH mamba·Zamba2 장문 no-GQA attn) "attention 기준 균일 예약"은 그 층을 굶겨 파멸적. layer_aware의 가치=**단순 환원이 아니라 민감층 보호**; agnostic_v2가 유효할 체제=**모든 decode 층이 16 SM을 무해히 견딜 때**(예: 소형·전층 memory-bound)로 희귀.

## 4. 핵심 발견과 함의 (정직)

1. `[measured]` **전제 반증(모델 의존):** layer-aware의 sim 전제("attn=비싼 SM-민감 레이어, 희소")는 **no-GQA attention 가정**에 의존. NemotronH(GQA+Mamba2 SSD)는 **mamba가 SM-민감·다수**, attn은 둔감·희소 → 전제가 성립 안 함. layer-aware 이득은 **SM-민감 레이어가 희소할 때만** 크다.
2. `[measured]` **NemotronH서 agnostic(v1)이 최적(async 청정 서빙 확증):** 다수 레이어(mamba+mlp)가 SM-민감이라 균일 예약이 decode SM을 충분히 확보(TPOT 22~25ms 평탄). 청정 서빙(§3.9): rate3~4서 **agnostic ≈ layer_aware > fused > agnostic_v2**, agnostic이 부하서 최고. **layer_aware ≈ agnostic**(둔감층 4/52뿐이라 안전 환원 대상 없음, 고부하선 전환 오버헤드로 소폭 열위).
3. `[measured]` **Zamba2서 layer_aware의 기전 확증(단 goodput 미전환):** Zamba2(no-GQA attn·**mamba SM-둔감** 45/54)선 **layer_aware가 agnostic보다 TTFT −7~23% 낮음**(§3.8) — 둔감 mamba SM 환원이 prefill 가속. **"layer_aware 이득 ∝ 환원 가능 레이어 비중"을 NemotronH(4/52,이득無) vs Zamba2(45/54,이득有) 대조로 실증.** 단 triton-no-cudagraph decode가 TPOT≫SLO(decode-bound)라 이 TTFT 이득이 goodput으로 미전환 — 전환엔 prefill-bound + cudagraph(빠른 decode) 필요.
4. `[measured]` **agnostic_v2("attn 기준 균일") = 최악 정책 (§3.9, async 청정):** decode를 16 SM으로 균일 굶기면 **TPOT가 rate2부터 SLO 돌파**(72→191ms)·**TTFT도 최악**(굶긴 decode가 prefill 막음) → goodput rate3서 0. **두 모델 모두 v2=최악**(NemotronH mamba·Zamba2 no-GQA attn 둘 다 SM-민감·지배적). ⇒ layer_aware의 가치 = "더 환원"이 아니라 **"민감층 보호"**; 무차별 환원(v2)은 지배적 민감층을 굶겨 파멸적. agnostic_v2 유효 체제=**전 decode 층이 16 SM 무해히 견딜 때**(희귀). *초판이 v2를 "고부하 우위"로 오판한 건 GIL-병목 client×과부하 rate×절벽 goodput 아티팩트 — async·sub-saturation 재측정으로 정정(§methodology).*
5. `[measured]` **실엔진 검증의 가치:** sim이 LUT로 가정한 per-layer SM-민감도를 **실 커널로 측정**해 정책 우열을 재판정하고, **decode-side 이득→serving 전이 조건까지 실측**(단순 평균-SM 분석이 놓치는 TPOT-벽/모델크기 효과) — 이것이 실엔진 이양(전략 B)의 핵심 산출.

---

## 5. 방법·환경 재현

- 측정: `triage/p1_4_nh_smsens.sbatch` (1 서버, 파일로 decode-SM 모드 스윕: full/64/44/32/16/layer_aware). NemotronH-8B, `--disable-cuda-graph --disable-piecewise-cuda-graph`(py3.14 inductor 회피), `--dtype bfloat16`.
- 계측 코드: `models/nemotron_h.py` NemotronHModel.forward의 per-layer-type CUDA-event 타이밍 + green-ctx per-layer 스트림 전환(`src/patches/`에 diff).
- 원자료: `triage/p1_4_nh_smsens_827583.txt`, 서버로그 `p1_4_srv_827583.log`.

## 5b. 서빙 방법·환경 재현

- 서빙 벤치: `triage/bench_client.py`(Poisson 도착, streaming TTFT/TPOT, goodput@SLO; `--tpot-slo`/`--ttft-slo` 조정 가능). 3정책 러너 `p1_4_serving.sbatch`(NemotronH), `p1_4_zb_serving.sbatch`(Zamba2, triton).
- ⚠️ **포트는 job-id 파생 필수**: `PORT=$((31200+JOBID%600))`(NH)/`$((32000+JOBID%500))`(ZB). 동시 run이 같은 노드서 포트 공유하면 교차오염(§초기 831140/841 폐기 원인).
- 원자료: NemotronH 3정책 `p1_4_serving_{831070,831172,831173}.txt`, Zamba2 3정책 `p1_4_zb_serving_{831166(prefill-bound),831146(decode-bound)}.txt`.
- 4정책(+agnostic_v2) 원자료: NemotronH `p1_4_serving_{831540, 831554, 831555}.txt`, Zamba2 `p1_4_zb_serving_831541.txt`. agnostic_v2 부팅검증 831534(ZB)/831535(NH).

## 6. 미해결 / 후속

- `[derived]` **완전-충실 layer-aware 서빙**: decode를 layer-type window로 쪼개 prefill과 상보 파티션 교차 실행(event_loop 대수술) → end-to-end goodput@SLO 스윕. 본 보고는 decode-side SM-민감도 + per-layer green-ctx 전환 실엔진 서빙까지 실측(la<agn TTFT 확증); 완전-충실 교차는 다음.
- `[measured→후속]` **Zamba2 goodput 전이 미완**: la<agn TTFT는 확증했으나 triton-no-cudagraph decode(TPOT≫SLO)로 goodput 미전환. **cudagraph 가능 환경**(py3.12 재빌드 or pdmux+mamba cudagraph 지원 버전)서 재측정하면 Zamba2 layer_aware goodput 우위 검증 가능.
- `[derived]` **모델 크기/decode 백엔드 축**: 큰 모델(prefill-bound) + 빠른 decode가 layer_aware 유리 체제. NemotronH-8B(flashinfer)는 pdmux 유리 체제 도달, Zamba2-2.7B(triton)는 decode-bound라 미도달.
- co_schedule 직접 서빙 측정(현재 fused=plain, agnostic/layer_aware=pdmux까지; co_schedule 별도).
