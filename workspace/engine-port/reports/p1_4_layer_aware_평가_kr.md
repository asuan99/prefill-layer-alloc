# Layer-aware SM 예약 정책 구현 및 4-정책 평가 (NemotronH 실측)

작성: 2026-07-03 · 대상: NemotronH-8B-Base (temporal 하이브리드: mamba 24 / mlp 24 / attn 4, 52층, GQA 8 KV-head, head_dim 128) · A100-80GB / CUDA13 / sglang v0.5.10 + green-ctx(sgl_kernel.spatial)
관련: [p1_3_nemotronh_layer_aware.md](p1_3_nemotronh_layer_aware.md)(pdmux-hybrid 기반) · [framework_comparison.md](framework_comparison.md)(sim 4-정책 정의)

---

## 0. 요약 (핵심 결론 먼저)

**실엔진 측정이 시뮬레이션의 layer-aware 전제를 NemotronH에 대해 뒤집었다.** sim은 "**attn-decode가 비싸고 SM-민감**, ssm-decode는 싸고 SM-둔감 → attn 레이어에만 SM floor 예약"을 전제했으나, NemotronH 실측은 정반대다:

- `[measured]` **mamba(SSM/SSD) decode가 SM-민감** (108→16 SM서 레이어당 0.589→1.547ms, **2.6× 느려짐**) 이며 decode step의 **62%** 차지.
- `[measured]` **attn decode는 SM-둔감** (레이어당 ~0.2ms 평탄, GQA라 memory-bound) 이며 step의 **5%** (4층뿐).
- `[measured]` mlp는 ~44 SM 이상서 둔감, 그 아래서 민감.

⇒ NemotronH서 SM을 아껴 예약할 "싸고 둔감한" 레이어는 **attn 4개뿐**이고, 다수(mamba+mlp)는 SM-민감이라 floor를 유지해야 한다. 따라서 **layer-aware의 이득은 NemotronH서 미미**하며(prefill로 환원 가능 SM이 4/52 레이어에 국한), **agnostic(전 레이어 균일 예약)이 사실상 동등하거나 우세**하다.

**단, 결정적 뉘앙스(§3.5): attn의 SM-민감도는 컨텍스트 길이 의존이다.** attn-decode는 O(L)이라 컨텍스트가 길어지면 무거워지고(0.30→0.45ms/층 @350→5100 tokens) **SM-둔감→SM-민감으로 뒤바뀐다**(ctx5100서 16SM 대비 full 43%↑). mamba는 O(1)이라 컨텍스트 불변. ⇒ **layer-aware 이득 = (아키텍처 GQA/no-GQA) × (컨텍스트 길이)의 2차원 함수**. "SM-민감 레이어 희소" 조건은 **긴 컨텍스트 + no-GQA(비싼 attn) + 희소 attn-층**에서 성립 — 그 체제가 layer-aware의 유일한 유리 구간이다.

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
| **agnostic** (MuxWise/Bullet) | 전 레이어 고정 N | N=64:22.9 / N=44:24.2 | N=64:44 / N=44:64 (**전 레이어**) | 균일 예약 |
| **layer_aware** | 민감타입(M,-) full·attn floor16 | ≈23.1 `[derived]` | ≈47.7 (attn 4층서만 92) | 희소-둔감 레이어만 환원 |

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

⇒ **사용자 직관 실증**: attn-decode의 "싸고 SM-둔감"은 **짧은 컨텍스트 국한**. 길어지면 attn이 커지고 SM-민감해진다. 단 NemotronH(GQA)는 **긴 컨텍스트서도 mamba가 지배**(ctx5100서 mamba 13.3ms vs attn 1.8ms=8%)라 여전히 mamba가 주 예약대상. **no-GQA(Zamba2)면 attn KV 4× → 긴 컨텍스트서 attn 비중·민감도가 훨씬 커져 layer-aware(attn 예약) 이득이 커질 체제**(Zamba2 실측은 torch_native 교란·장문 CUDA오류로 불완전; mamba/층 ~0.56은 NemotronH와 일치 확인, attn 비교는 백엔드 교란).

**함의: layer-aware 이득은 (아키텍처 GQA/no-GQA) × (컨텍스트 길이)의 2차원 함수.** SM-민감 레이어가 희소해야 이득 — 짧은 컨텍스트 GQA선 mamba가 다수라 이득無; **긴 컨텍스트 + no-GQA**서 attn이 비싸지되 희소(9/54)면 layer-aware가 유리할 후보.

## 4. 핵심 발견과 함의 (정직)

1. `[measured]` **전제 반증(모델 의존):** layer-aware의 sim 전제("attn=비싼 SM-민감 레이어, 희소")는 **no-GQA attention 가정**에 의존. NemotronH(GQA+Mamba2 SSD)는 **mamba가 SM-민감·다수**, attn은 둔감·희소 → 전제가 성립 안 함. layer-aware 이득은 **SM-민감 레이어가 희소할 때만** 크다.
2. `[derived]` **NemotronH서 agnostic이 실용적 우위:** 다수 레이어(mamba+mlp)가 SM-민감이라, decode를 중간 SM(44–64)에 균일 예약하면 decode 손실 소(22.5→22.9~24.2, +2~8%)로 prefill에 44–64 SM을 **전 레이어** 환원 — layer-aware보다 총 환원 SM이 크다.
3. `[derived]` **Zamba2 전망:** Zamba2는 **no-GQA attention**(비싼 attn-decode 가능성)이라 attn이 SM-민감이면 layer-aware가 유리할 수 있음. 단 Zamba2의 mamba도 Mamba2 SSD(SM-민감)라 동시 민감이면 이득 축소. **Zamba2 per-layer-type 민감도 측정이 후속 필수**(현재 torch_native라 느리나 측정 가능).
4. `[measured]` **실엔진 검증의 가치:** sim이 LUT로 가정한 per-layer SM-민감도를 **실 커널로 측정**해 정책 우열을 재판정 — 이것이 실엔진 이양(전략 B)의 핵심 산출.

---

## 5. 방법·환경 재현

- 측정: `triage/p1_4_nh_smsens.sbatch` (1 서버, 파일로 decode-SM 모드 스윕: full/64/44/32/16/layer_aware). NemotronH-8B, `--disable-cuda-graph --disable-piecewise-cuda-graph`(py3.14 inductor 회피), `--dtype bfloat16`.
- 계측 코드: `models/nemotron_h.py` NemotronHModel.forward의 per-layer-type CUDA-event 타이밍 + green-ctx per-layer 스트림 전환(`src/patches/`에 diff).
- 원자료: `triage/p1_4_nh_smsens_827583.txt`, 서버로그 `p1_4_srv_827583.log`.

## 6. 미해결 / 후속

- `[derived]` **완전-충실 layer-aware 서빙**: decode를 layer-type window로 쪼개 prefill과 상보 파티션 교차 실행(event_loop 대수술) → end-to-end goodput@SLO 스윕. 본 보고는 decode-side SM-민감도(정책 핵심 차별자)까지 실측; 서빙 goodput 스윕은 다음 단계.
- **fused/co_schedule 직접 서빙 측정**(현재 decode-side만; 동시 prefill 경합 포함한 goodput은 미측).
- **Zamba2 per-layer-type 민감도**(no-GQA attn 전제 검증 — layer-aware가 유리할 후보 모델).
- co_schedule/fused는 sglang plain(fused)·pdmux(agnostic)로 서빙 벤치 가능; layer_aware는 위 §6 대수술 후.
