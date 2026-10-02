# 핵심 동기 검토 — attention vs SSM의 시퀀스 길이별 실행 시간: 무엇이 측정됐고 무엇이 성립하는가 (2026-10-01, rev2)

> **지위**: 기존 측정의 재정리·검토. **새 측정 0 · GPU 0 · 새 판정 0.** 확정 결론(layer-type 런타임 정책 全형태 死)은 재도출하지 않는다.
> **rev2**: rev1을 claims-auditor가 `INACCURATE`(정정 16건)로 판정했고 이 판은 그 정정을 전부 반영했다. 메인 세션이 F3 반증을 원자료로
> 독립 재확인했다(§8). 판정서 요지는 §8.
> 출처 위계: `reports/CONSENSUS.md` §1-3(줄 ~3487)·§1-21(~3498)·§3 항목22(~3813) > `deprecated_v2/README.md`(스코프 명시 기록) >
> `workspace/engine-port/results/prefill_knee/AGGREGATE_COMPOSITION_2026-08-04.md` REV.2(UNAUDITED) > `diffA_vs_diffB_table.md`(계측 경고 배너).
> `reports/prefill_vs_decode_execution.md`는 HISTORICAL/SUPERSEDED라 근거로 쓰지 않는다. **job 896776 재측정 수치는 인용 금지**
> (`deprecated_v2/README.md` §7 6항) — 이 문서는 그 job을 정의 확인에만 쓰고 수치를 근거로 쓰지 않는다.

## 0. 결론 먼저

1. **비용 비대칭(Diff A)은 실재하고 L과 함께 커진다 — 단 "attention 코어 vs SSM mixer" 단위에서, L ≳ 2k에서.** 코어 시간은 L^1.916±0.021,
   mixer 시간은 L^0.954±0.008로 증가한다(steady OLS, ctx ≥ 2050). 코어/mixer 시간비는 L 256 → 32k에서 약 0.11 → 11로 변한다.
2. **교차점(코어 1회 = mixer 1회)은 ≈2.75k 토큰(steady)이며, 조건을 붙이면 인용 가능하다.** rev1의 "교차점 불일치"는 내가 attention
   **모듈**(qkv/o_proj 포함) 필드를 코어 필드와 비교해서 만든 가짜였다. 이 교차점은 커널 효율비(코어 대 mixer)가 정하므로 백엔드에 의존한다.
   층 집계 교차점은 가정 하나로 3k↔30k를 오가 판정 불가다.
3. **SM 민감도 비대칭(Diff B)은 prefill 정책 단위에서 없다.** 커널 단위로는 짧은 L에서 타입 차이가 있다(mamba가 44 SM에서 일찍 포화).
   그러나 정책 단위로 환산하면 attention 몫이 작아 1로 끌린다. 정본의 현행 서술은 "(D)가 삼켰다"가 아니라 **"정책 단위에서 lever가 애초에 없었다"**이다.
4. **decode에서 "SSM은 SM에 평탄하다"는 쓸 수 없다.** 정본 §1-21과 C2(pure Mamba2 포함 4 arm에서 decode ITL이 SM에 민감)가 반증하며,
   근거였던 decode Diff B ≈4.0×는 비율 자체가 결함 계열(수정 전 계측)이다.
5. **전부 한 모델·한 백엔드·한 기판의 micro 측정이다**(Zamba2-2.7B · triton · no-cudagraph · KISTI A100 · decode와 비동거 · 서빙 아님).
   타겟(Nemotron-H Nano-9B · flashinfer · VESSL)에서는 아무것도 재지 않았고, 이 기판에서 백엔드 효과와 모델 효과는 분리할 수 없다.
6. **방어 가능한 동기 문장은 하나다**: "attention 코어와 SSM mixer의 시간은 L ≳ 2k에서 각각 L^1.92, L^0.95로 증가한다(스코프 병기)."
   "prefill 시간의 구성이 L에 따라 바뀌고 span 비용은 조성 × 길이의 함수다"는 **아직 지지되지 않는 가설**이다(§4).

## 1. 무엇을 쟀나

| 측정 | job | 격자 | 조건 | 재는 양 |
|---|---|---|---|---|
| (B,L) 2D knee, OLD | 847690/847711/847897 | L 2k–32k × B 1–48 × SM {8…108} | Zamba2-2.7B, triton, `--disable-cuda-graph`, pinned single-stream prefill micro, chunk 16384, radix OFF | 2026-08-04 이전 ZBPT 버킷(`attn` = RadixAttention 코어만) |
| WIDE 스윕 | 857371/857477 | L 256–32768 × B 1–16 | 동일 | 동일 |
| 대칭 버킷 재측정 | 896776 | L {2000, 8000} × B {1, 4} × SM {full, 24} | triton만 성공, 2026-08 빌드 | ZBPT2 4버킷(attn 모듈·mlp·other·mamba) + 진단 `attn_core` + steady 블록 필드 — **수치 인용 금지, 정의 확인용** |
| decode knee | 835571 계열 | ctx 3600, SM {8…108} | Zamba2, triton, no-cudagraph | 수정 전 계측(per-attn/per-mamba) |

**두 비율의 정의**
- **Diff A (비용비)** = attention **코어** 1회 시간 / SSM **mixer** 1회 시간(per-op, 같은 L·B·SM). 층 단위 비율이 아니다(§3 F1).
- **Diff B (민감도비)** = 코어의 SM 8→108 속도 향상 / mixer의 SM 8→108 속도 향상. 층 종류별 SM 재배분 lever를 잰다.

## 2. 측정이 말하는 것 (스코프: Zamba2-2.7B · triton · no-cudagraph · micro · KISTI A100 · n_indep = 1)

### 2.1 Diff A — 비용 비대칭은 L ≳ 2k에서 열린다
- 지수(steady OLS, WIDE B=1, ctx ≥ 2050): **코어 ∝ L^1.916±0.021, mixer ∝ L^0.954±0.008**(R² ≥ 0.9996). 이전 1.68/0.61은 OLD의 오염 끝점
  (ctx 1778 warm-up 오염, 28442 청크 평균) 두 점에 의존한 구간 추정이라 정정됐다.
- **짧은 L은 이 법칙 밖이다**: mixer는 ctx 258 → 1026에서 거의 평탄(고정비 지배), 코어의 국소 기울기는 0.8–1.7(REV.2 §15.1).
- per-op 교차점 **≈2.75k**(steady 2,753). 코어/mixer 비는 L=32768 steady에서 약 11(clean). 이 교차점은 커널 효율비(코어 대 mixer 처리율)가
  정하므로 **백엔드가 바뀌면 이동한다**(REV.2 §17(d)) — Nemotron·flashinfer에서 다를 것이라는 예상의 정량적 이유.
- B 축은 판정 불가: B > 1 셀은 shape 혼합(`dirty`) 셀이다.

### 2.2 Diff B — prefill 정책 단위에서 lever 없음
- steady: L=256은 단일값 인용 금지, **밴드 [1.24, 1.34]**; L=512 **1.198**; L=1024 **0.904**; L=2000 **1.0081** [1.0078, 1.0084](보고 1.32–1.38은 REFUTED);
  L ≥ 8000 **0.96–1.04**. 정리하면 L ≥ 1024에서 0.90–1.04(L ≥ 2048에서 0.99–1.04).
- 기전: **커널 단위로는 타입 차이가 있다** — mamba mixer가 44 SM에서 일찍 포화하고 L=256에서 44→108 SM이 오히려 약간 느리다(역행, steady 2.59%).
  그러나 정책 단위(모듈 전체)로 환산하면 `R_policy ≈ 1 + w_attn·(DiffB − 1)`, w_attn ≈ 9.6%라 ≈1.03으로 소멸한다 — 이는 U-K 결과와 조성으로 정해지는
  함수이지 **독립 증거가 아니다**.
- 정본 현행 서술: "(D)가 삼켰다" → **"정책 단위에서 lever가 애초에 없었다"**(CONSENSUS §1-3).

### 2.3 decode — "SSM은 SM에 평탄"은 철회된 서술이다
- 정본 §1-21: "decode O(1) recurrent라 SM-bound 불가" 전제는 틀렸다고 명시("SM 수의 O(1)이 아니었다").
- C2 CONFIRMED(scoped): pure Mamba2 arm을 포함한 4 arm 모두 decode ITL이 SM에 민감(모델-무관, 레버 존재만, 정책 이득 아님).
- "ctx256 1.1× SM-free"는 인용 금지(§1-5)[CS-OK: 금지 사실 서술].
- decode Diff B ≈4.0×(835571)는 수정 전 계측이다. full-SM 셀이 항상 첫 arm이라 warm-up이 full-SM mamba를 부풀려 SM 이득을 작게 보이게 하는
  방향 — **비율 자체가 결함 계열**이다. 기록으로만 남기고 "SM-free" 표현은 금지.

## 3. 측정이 말하지 못하는 것

| # | 한계 | 근거 | 영향 |
|---|---|---|---|
| F1 | **버킷 비대칭(OLD/WIDE)**: 2026-08-04 이전 `attn` 버킷은 RadixAttention 코어만(qkv/o_proj·MLP·layernorm 제외), `mamba` 버킷은 mixer 전체 | REV.2 §2(d); 2026-08-04 이후 코드는 `attn` = `Zamba2Attention` 모듈 전체, 코어는 진단 버킷 `attn_core`(`workspace/engine-port/src/models/zamba2.py:269-275`) | Diff A는 "코어 vs mixer"이지 "attention 층 vs SSM 층"이 아니다. OLD/WIDE에서 계측된 op은 prefill FLOP의 58–70%, 빠진 30–42%는 공유 transformer 블록의 GEMM(L에 선형). **보정 방향(CONFIRMED)**: 짧은 L에서 층 단위 비는 측정값보다 크고, 32k에서도 빠진 FLOP이 30%라 격자 안에서 수렴하지 않는다. "attention 층"은 공유 transformer 블록(attention 모듈 + MLP + LN)으로 정의해야 한다(`Zamba2HybridLayer`로 정의하면 mixer를 포함해 항등식). SSM 쪽 LN·residual 누락은 무시 가능(fwd의 수 %) |
| F2 | **누산기 러닝평균**: `_zt_acc` 미리셋 + 하네스 `tail -1` | CONSENSUS §1-3 강등 | OLD/WIDE의 steady 값은 **연속 줄 역산 블록평균**(REV.2 §12)이고, 896776 이후 ZBPT2는 **steady 블록 필드**(`b_per_*`)를 직접 낸다 — 출처가 다르다 |
| F3 | **집계(9 hybrid vs 45 pure, 성분 귀속) 교차점은 판정 불가**: 미계측 GEMM 효율 가정 하나로 3k↔30k | CONSENSUS §3 항목22 | 층 집계 기준 "어디부터 attention이 지배하나"는 현재 자료로 답할 수 없다. 층 단위(hybrid 모듈 vs pure)는 hybrid가 mixer를 포함하므로 항등식으로 >1 |
| F4 | **n_indep = 1**(WIDE 셀당 런 1개) | §1-3 | CI는 런 내 블록 정밀도이지 재현성이 아니다 |
| F5 | **micro는 서빙이 아니다**. PD-mux prefill 경로는 원래 eager라(`SPLIT_PREFILL`은 cudagraph 대상 밖, `reports/bcg_applicability_2026-08-21.md:41`) no-cudagraph 자체는 prefill 쪽 큰 스코프 문제가 아니다. **그러나 다른 스코프 차이가 크다**: decode와 비동거(green-context 분할 아래 L2/HBM 경합 미측정) · prefill SM이 full 또는 고정(서빙은 108 − D) · micro는 chunk 16384, 서빙은 `chunked_prefill_size == -1` 강제 · 서빙은 층 단위 span으로 나뉘고 span당 오버헤드가 있음 · radix OFF | REV.2 §7(c)3, 설계 메모 R1·R3 | 서빙 결론으로 승격 불가(게이트 1) |
| F6 | **한 모델·한 백엔드**: Zamba2(코어 9 / mixer 54) · triton | `deprecated_v2/README.md` §1, CONSENSUS §3 항목103 | Nemotron-H는 triton 부팅 거부, Zamba2는 flashinfer에서 스케줄러 사망 → **이 기판에서 모델 효과와 백엔드 효과 분리 불가**. 교차점이 커널 효율비에 의존하므로 백엔드 교락이 직접 수치를 움직인다 |
| F7 | **타겟 조건 미측정**: Nemotron-H Nano-9B · flashinfer · VESSL | — | 계수 전부 재측정 필요(게이트 1 연장) |

## 4. 핵심 동기 문장 — 쓸 수 있는 것과 없는 것

| 문장 | 판정 | 조건·이유 |
|---|---|---|
| "attention 코어와 SSM mixer의 시간은 L ≳ 2k에서 각각 L^1.92, L^0.95로 증가한다" | **쓸 수 있음** | 스코프 필수: triton · Zamba2-2.7B · no-cudagraph micro · n_indep = 1 · 서빙 아님 · 코어 vs mixer(층 아님). 896776을 근거로 쓰지 않는다 |
| "코어 1회와 mixer 1회의 시간이 같아지는 길이는 ≈2.75k 토큰" | **조건부 쓸 수 있음** | 위 스코프 + "층 단위 아님" + "커널 효율비가 정하므로 백엔드 의존" |
| "코어/mixer 시간비는 L=32k에서 약 11" | 조건부 | clean steady(L=32768). F1 단서 필수 |
| "Diff A 진폭 21×" | **쓸 수 없음** | 끝점 둘 다 오염(L2000 warm-up, L32000 청크 평균). 정본 §1-3 배너는 아직 "스코프 내 참"으로 둠 → **doc-steward 회부**(이 문서의 금지가 정본보다 엄격) |
| "층 집계 기준 교차점 ≈3k" 또는 "≈12–18k" | **쓸 수 없음** | F3 |
| "prefill 시간의 구성이 L에 따라 바뀌고, 시간 예측·span 비용은 조성 × 길이의 함수다" | **NOT-YET-SUPPORTED(가설)** | "조성 × 길이의 함수"는 per-op 비용이 타입·L에 따라 다르다는 전제의 회계 항등식이고, 경험적 내용은 Diff A의 기울기뿐. M 축 n_model = 1. "span 비용"은 서빙 양이라 micro로 못 삼(게이트 1). GEMM 보정은 짧은 L에서 조성 이동을 **완화**한다(코어 지배는 16–32k 이상에서만, 사실상 끝점 L=32768 하나에 의존 — 게이트 #8). 현실 밴드(2–4k)에서 총 prefill의 유효 지수는 ≈1.09이고 코어 몫은 11→19%로 작게 움직인다(REV.2 §7(c)1) |
| "따라서 SM을 층 종류별로 나눠야 한다" | **쓸 수 없음** | 정책 단위 lever 없음 + 서빙 직접 측정으로 全형태 死 |
| "decode에서 SSM은 SM에 둔감하다" | **쓸 수 없음** | §2.3 |
| "Nemotron-H에서도 같다" | **쓸 수 없음** | F6·F7 |

방어 가능한 가설 형태(사전등록용): *"단일 모델 micro에서 코어/mixer 시간비가 L 256 → 32k에서 약 0.11 → 11로 변한다. 따라서 조성이 다른
모델의 prefill 시간을 예측하려면 타입별·길이별 계수가 필요하다는 것이 가설이다(3-a·A1 서빙 측정 전 미지지)."*

## 5. 이 동기의 역사적 위치 (재도출 아님, 맥락)

- 원래 동기는 sim의 layer-aware 이득(크기 추세 1.2B 1.37× / 2.7B 2.02× / 7B 1.82×)이었고, sim→서빙 전이 실패(confound #1)로 무너졌다.
- 그 뒤 prefill로 후퇴하며 "Diff A가 L에 따라 열린다"가 동기의 정량 형태가 됐지만, 필요했던 것은 Diff A가 아니라 Diff B였고 그것은 정책 단위에서 없었다.
- `reports/longcontext_trace_plan.md` §1은 같은 동기를 "교차점 ≈3k"로 서술한다 → 이 문서의 조건부 형태(per-op ≈2.75k, 층 단위 아님, 백엔드 의존)와
  정합시켜야 한다(doc-steward 회부).
- 8B prefill 축 서빙 캠페인(`results/s8p_prefill`, 미감사·인용 금지)에서 hybrid arm의 TTFT~L 곡률은 판정 불가였고 pure Mamba2는 평탄, transformer만
  곡률이 있었다 — **서빙 수준에서 hybrid의 L 의존 조성 이동은 아직 해상되지 않았다**는 반대 방향 신호(방향만, 인용 금지 자료).

## 6. 설계와의 연결

- **살아 있는 축은 비용(Diff A)이지 민감도(Diff B)가 아니다** — 설계 메모 §3 A1 "layer type의 역할은 SM 개수가 아니라 남은 prefill 시간 예측".
  cost model CM-2(span 시간의 층 불균일), X-4(MuxWise `N_PL`의 층 균일 가정), A8(비용 가중 span)이 이 위에 있다. 다만 §4대로 "span 비용"
  문장은 가설이고, 현실 밴드(2–4k)에서는 조성 이동이 작다는 점을 설계 기대치에 반영해야 한다 — X-4가 의미를 갖는 것은 긴 L(≳16k) 쪽이다.
- **M/U/E**: Diff A는 M(조성) × U(L)의 상호작용. U 축(L)은 단일 모델에서 움직일 수 있으나 M 축은 hold-out 필요.
- **decode 쪽**: CM-1의 `s_attn ≠ s_ssm`과 X-7은 "SSM은 SM에 둔감"에 기대면 안 된다(§2.3). 커널 계열별 SM 스케일링이 **다르다**는 것은 3-a에서
  새로 재야 할 가설이며, 지금 근거는 없다 → cost model 문서 CM-1·X-7 서술 정정 필요(아래 §7).

## 7. 재측정 제안과 후속

- **하네스는 이미 대칭 버킷이다**(2026-08-04 재작업: attn 모듈·mlp·other·mamba + 진단 `attn_core` + closure 게이트 + steady 블록 필드; 896776·873783에서
  사용). 남은 일은 `knee2d.sbatch:105` 파서(ZBPT → ZBPT2) 수리와 재측정이다.
- 재측정 격자(3-a/D-6에 흡수): 타겟 Nemotron-H Nano-9B + flashinfer · L {1k, 1.5k, 2k, 3k, 4k, 8k, 16k, 24k, 32k} + 65536 초과 1점 · B=1 ·
  SM {full, 운영점 108 − D 두 점} · cudagraph·청킹은 서빙 운영점과 동일 · 독립 부팅 n ≥ 2를 같은 job 안에서 짝으로 · 4버킷 + closure ≥ 0.98 보고.
  이어서 PD-mux 서빙에서 L 밴드별 TTFT 분해(`W_queue`/`S_prefill`, n ≥ 4, paired CI)로 "span 비용" 문장을 검증.
- GPU 0 후속: result-analyst가 896776 ZBPT2 전 줄을 `dirty=0` + 첫 블록 제외로 재파싱해 `b_per_attn_core`를 REV.2 steady와 같은 표에 정의를 병기해
  등재(인용 가능 형태로 만들기 위함).
- doc-steward 회부: (a) 정본 §1-3 배너의 "21×" 서술 재검토 (b) `longcontext_trace_plan.md` §1 교차점 서술 정합 (c) cost model 문서 CM-1·X-7의 "SSM은
  SM 둔감" 근거 정정.

## 8. 감사 기록 (claims-auditor, 2026-10-01, rev1 대상)

판정 `INACCURATE`, 정정 16건(판정을 바꾸는 것 4건: F3 가짜 불일치 · 896776 인용 정지 위반 · decode SSM 평탄 서술 · 폐기된 기전 서술). 확인된 것:
지수(0.954±0.008이 steady, 0.957±0.006은 정정 전 표) · Diff B steady 수치 · F1 수치와 보정 방향 · 집계 교차점 3k↔30k · n_indep = 1 · 백엔드 교락 ·
"PD-mux prefill은 eager". 누락으로 지적된 것(F5의 비동거·SM·청킹·span 오버헤드, 유효 지수 ≈1.09, 교차점의 커널 효율 의존, 8B 서빙 캠페인의 반대 신호,
sim 크기 추세, longcontext_trace_plan 정합)은 이 판에 반영했다.

**메인 세션 독립 재검증(GPU 0)**: (i) `workspace/engine-port/src/models/zamba2.py:269-275` — 주석이 "`attn_core`는 2026-08-04 이전 `attn`이 잰 것과 정확히 같은 진단 버킷, 회계용
`attn`은 qkv_proj + adapters + core + o_proj 전체"라고 명시 — 확인. (ii) 896776 원로그에서 `dirty=0`·첫 블록 제외로 직접 재계산: ctx 1778에서
코어/mixer 비가 교차점 2,753 아래(< 1), 모듈/mixer 비는 > 1 — rev1이 인용한 값은 모듈 필드였음을 확인. 수치는 인용 정지 대상이라 적지 않는다.
(iii) `deprecated_v2/README.md` §7 6항 "job 896776 재측정 수치 인용 금지" — 확인.
