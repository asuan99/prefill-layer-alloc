# P1 운영점(cudagraph-ON) 대조 — 채점 결과

> **AUDITED (2026-08-05, claims-auditor) — 원자료·통계는 정본 반영 완료,
> §0의 두 판정 문구는 감사자가 채택하지 않았다(아래 참조).** result-analyst
> 산출물. 셀 요약·paired bootstrap·위반 분해(§1–§6)는 claims-auditor 감사를
> 통과해 `reports/CONSENSUS.md` §1-1(rev12)·`PROJECT_STATUS.md` "확정된
> 결과" 1번의 근거로 인용됐다. **단 §0 "한 줄 판정" 표의 "Zamba2 P1 운영점
> 확인"·"Granite P1 전면 강등" 두 문구는 감사자가 명시적으로 등재 금지했다**
> — 전자는 유의한 셀이 전부 절벽 위(§1.3–1.5)라 "확인"이 성립하지 않고,
> 후자는 그 술어가 위반 요청 0건인 항등식(goodput≡throughput)이라 강등
> 근거가 되지 못한다(§3.3 참조). 정본 문구는 `reports/CONSENSUS.md` §1-1
> 전문을 참조 — 이 문서를 직접 인용할 때도 §0 표가 아니라 §1-1의 표현을
> 쓴다. 원문은 삭제하지 않고 보존한다(doc-steward, 2026-08-05).

- 작성: 2026-08-05, result-analyst
- 원자료: `workspace/engine-port/results/p1_opint/` (jobs **873944** Zamba2-2.7B,
  **873945** Granite-4.0-h-micro-base, 둘 다 COMPLETED)
- 사전등록: 같은 디렉터리 `PREREG.md` (제출 전 고정). **규칙 사후 수정 없음.**
- 분석 도구: 정본 `workspace/engine-port/benchmarks/pdmux_eval/analyze.py`
  (`summarize_requests`, `paired_bootstrap_ci` seed=1/10000 samples, `percentile`).
  per-request TPOT 백분위는 이번에 정본 모듈에 `request_tpot_percentiles`로 **추가**
  (기존 percentile 규약 준수, 순수 함수·가산적).

---

## 0. 한 줄 판정

| 모델 | 사전등록 지표(mean-ITL SLO) 판정 | 정본 지표(p95 token-ITL, 게이트 #4) 판정 |
|---|---|---|
| **Zamba2-2.7B** | **P1 운영점 확인** — 단 rate ≥ 4 한정(rate 2·3은 <3%·CI가 0 포함) | 확인 (rate 2부터 전부) |
| **Granite-4.0-h-micro-base** | **P1 전면 강등**(비운영점 아티팩트) — 전 rate <3%, 부호는 오히려 음(−) | 확인 (rate 2부터 전부) |

★ **Granite의 판정은 rate가 아니라 goodput 지표 정의가 가른다.** 두 정의 모두 데이터를
보기 전에 고정돼 있었다(PREREG는 원 캠페인 비교가능성을 이유로 mean-ITL을 명시 선택,
`CLAUDE.md` 게이트 #4는 p95-ITL을 정본으로 규정). 어느 쪽이 정본 claim을 지배하는지는
정의 문제이며 claims-auditor 몫이다. 아래 §3에 두 지표 전부 병기한다.

---

## 1. J1 — cliff flag (어느 셀이 절벽 위인가)

### 1.1 사전등록 플래그의 기계적 적용 (Phase 0, n=1, 60 prompts, seed 7)

플래그 A(completed/total<1.0) · B(TTFT p50 > 2× 직전 rate) · C(goodput/throughput<0.95).

| 모델 | 정책 | r2 | r3 | r4 | **r6** | (참고) r5 | r8 | r10 | r12 |
|---|---|---|---|---|---|---|---|---|---|
| Zamba2 | plain | – | – | – | **CLIFF (C=0.200)** | CLIFF (C=0.133) | CLIFF (B,C) | CLIFF (C) | CLIFF (C) |
| Zamba2 | agnostic | – | – | – | **CLIFF (C=0.850)** | – | CLIFF (C) | CLIFF (C) | CLIFF (C) |
| Granite | plain | – | – | – | – | – | CLIFF (B,C=0.350) | – | – |
| Granite | agnostic | – | – | – | – | – | CLIFF (B) | – | CLIFF (C=0.650) |

**채점 rate(2/3/4/6) 중 절벽 위로 플래그된 셀 = Zamba2 rate 6, 양 정책 모두.** 그 외
채점 셀은 사전등록 플래그상 절벽 아래.

플래그 A는 전 격자에서 미발화(모든 셀 120/120·60/60 completed, error 0).

### 1.2 플래그 C는 대수적으로 "위반율 ≥ 5%"와 동일하다

goodput과 request_throughput은 분모(duration)가 같으므로
`goodput/throughput = good/total`. 즉 C는 TTFT가 SLO에 근접했는지를 재는 게 아니라
**SLO 위반 요청 비율이 5% 이상인지**를 재는 검정이다. 판정에는 문제없지만, 이름이
암시하는 "SLO 근접"과 실제 추정 대상이 다르므로 명시한다.

### 1.3 사전등록 플래그가 놓친 셀 — Zamba2 plain rate 4 (중요)

- Phase 0에서 Zamba2 plain r4의 C 비율은 **정확히 0.950000**(57/60). 검정이 `<0.95`
  이므로 **위반 요청 1개 차이로 미발화**했다.
- 같은 셀을 Phase 1(120 prompts, n=5)에서 재면 비율 **0.687 ± 0.209** — 위반율 31%로
  결정적으로 절벽 위다. Phase 0(60 prompts, n=1)이 **런 길이 부족으로 과소탐지**했다.
- 즉 **실질적 절벽 위 셀 = Zamba2 {plain r4, plain r6, agnostic r6}**
  (사전등록 플래그 기준으로는 r6 두 셀만). agnostic r4는 0.970 ± 0.049로 경계 근처.

### 1.4 절벽이 정책마다 다른 rate에 온다 ⇒ rate-교락

Phase 1 위반율 기준(= 1 − goodput/throughput, 5 rep 평균):

| 모델 | 정책 | r2 | r3 | r4 | r6 |
|---|---|---|---|---|---|
| Zamba2 | plain | 0.990±0.007 | 0.982±0.037 | **0.687±0.209** | **0.117±0.117** |
| Zamba2 | agnostic | 0.990±0.007 | 0.998±0.004 | 0.970±0.049 | **0.567±0.214** |
| Granite | plain | 0.998±0.004 | 1.000±0.000 | 1.000±0.000 | 0.992±0.012 |
| Granite | agnostic | 0.998±0.004 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 |

Zamba2는 **plain이 r4에서 이미 절벽 위, agnostic은 아니다**. 두 정책 공통으로 절벽
아래인 rate는 **{2, 3}뿐**이며, 그 구간의 사전등록 goodput 차이는 −0.09%/+1.92%
(CI 모두 0 포함) — 즉 **공통 off-cliff 구간에서는 효과 없음**.

단, "plain이 먼저 절벽에 간다"는 사실 자체가 용량 차이라는 소견(§4)과 분리되지 않는다.
절벽 셀을 버릴지(측정 아티팩트로 보고) 살릴지(용량 차이의 발현으로 보고)는 해석이며,
여기서는 **양쪽 수치를 모두 제시**하고 절벽 표시를 병기한다.

### 1.5 threshold-goodput의 런 길이 의존(ill-posed) 실측

요청 launch 순서 전반부/후반부 위반율 (5 rep pooled, 600 req/셀):

| 모델·정책 | r4 전반 → 후반 | r6 전반 → 후반 |
|---|---|---|
| Zamba2 plain | 0.203 → **0.423** | 0.917 → 0.850 |
| Zamba2 agnostic | 0.010 → 0.050 | 0.127 → **0.740** |
| Granite plain | 0.000 → 0.000 | 0.000 → 0.017 |
| Granite agnostic | 0.000 → 0.000 | 0.000 → 0.000 |

Zamba2 과부하 셀은 런이 길수록 goodput이 계속 떨어진다 ⇒ **Zamba2 r4·r6의 효과
"크기"는 런 길이에 의존하는 ill-posed 수치**(게이트 #6). 부호는 5 rep 중 4~5 rep에서
일관되지만, +41.8%/+398%라는 숫자를 절대값으로 인용해선 안 된다. Granite은 정상 상태
(전·후반 위반율 0) — 이 문제 없음.

### 1.6 experiment-runner 중간보고 재현 여부

재현됨. Zamba2 plain capscan r4→r5: TPOT p50 **40.98 → 88.55 ms**(보고 40.18→87.19,
bench_serving의 `(latency−ttft)/(outlen−1)` 정의와 mean-ITL 정의 차이 수준),
TTFT p99 **2.452 → 6.524 s**(보고 2451→6524 ms), r5 request throughput **3.82**
(보고 3.82 포화). 일치.

---

## 2. J2 집계 단위 선언 + 데이터 무결성

- **독립 반복 단위 = repetition(rep 1..5).** `n_indep = 5` per (모델, 정책, rate).
  rep마다 fresh server boot + 독립 seed(`9000+100·rep+rate`) ⇒ 독립 draw.
  **셀 내부 120개 요청은 반복 단위가 아니다**(요청 간 산포 ≠ rep 간 산포).
- **pairing = 동일 rep index.** 동일 seed ⇒ prompt 집합·Poisson 도착열 동일.
  검증: 모든 (모델, rate, rep)에서 `input_lens` 배열이 arm 간 완전 일치, 셀 duration도
  rate 2–4에서 ≤0.1 s 이내 일치(예: Zamba2 r2 rep1 74.4 s / 74.4 s).
- **duration**: JSONL 한 줄 = 한 rate 셀(bench_serving 1회 호출). 여러 라운드를 한
  파일에 append한 구조가 **아니므로** `sum` vs `max` 문제가 발생하지 않는다(게이트 #5
  점검 완료). 각 셀 goodput 분모 = 그 셀 자신의 duration.
- **완결성**: 2 모델 × 2 정책 × 4 rate × 5 rep = 80 셀 전부 존재, 각 120/120 completed,
  error 0. Phase 0은 2×2×9 = 36 셀 전부 존재, 각 60/60.
- **ITL 미기록 요청 처리(선언)**: 일부 요청(전체의 ≤1%, 예: Zamba2 5 rep 합산 r2 6건)은
  `output_len=96`인데 `itls`가 빈 배열이다. 해당 요청의 TPOT는 **정의 불가 ⇒ 위반으로
  채점**(사전등록 스크립트 `p1op_analyze.py`의 규약과 동일). 민감도: 반대 규약
  (미기록=통과, 정본 `passes_legacy_mean`의 기본 동작)로 재채점해도 paired 효과는
  최대 0.17%p 변동(Zamba2 r4 +41.76% → +41.90%), **어떤 판정도 뒤집히지 않는다.**
  해당 요청은 seed가 같아 **양 arm에서 동일 index로 동일 개수 발생**(paired 균형).

---

## 3. J2/J3/J4 — 셀별 표와 paired 효과

### 3.1 셀 요약 (n=5 rep, mean ± SD across reps; TTFT는 초, TPOT는 ms)

**Zamba2-2.7B (triton, ctx 4096)**

| 정책 | rate | goodput(PREREG) | goodput(정본 p95) | throughput req/s | TTFT p50 | p95 | p99 | TPOT p50 | p95 |
|---|---|---|---|---|---|---|---|---|---|
| plain | 2 | 1.848 ± 0.160 | 1.314 ± 0.056 | 1.866 ± 0.152 | 0.164 | 0.365 | 0.752 | 12.78 | 23.78 |
| agnostic | 2 | 1.847 ± 0.159 | 1.847 ± 0.159 | 1.865 ± 0.151 | 0.230 | 0.576 | 0.970 | 12.53 | 19.30 |
| plain | 3 | 2.894 ± 0.243 | 1.032 ± 0.268 | 2.958 ± 0.360 | 0.211 | 0.543 | 0.878 | 22.10 | 43.17 |
| agnostic | 3 | 2.949 ± 0.354 | 2.949 ± 0.354 | 2.954 ± 0.353 | 0.385 | 1.121 | 1.421 | 17.71 | 29.26 |
| plain | 4 ⚠ | 2.705 ± 0.755 | 0.562 ± 0.373 | 3.970 ± 0.189 | 0.306 | 1.366 | 2.657 | **48.51** | **81.18** |
| agnostic | 4 | 3.834 ± 0.074 | 3.376 ± 0.729 | 3.961 ± 0.200 | 0.743 | 2.050 | 2.842 | 30.95 | 49.11 |
| plain | 6 ⚠ | 0.513 ± 0.519 | 0.221 ± 0.339 | 4.283 ± 0.226 | 2.127 | 5.580 | 6.084 | **91.32** | **112.26** |
| agnostic | 6 ⚠ | 2.554 ± 0.959 | 1.621 ± 0.665 | 4.515 ± 0.084 | 2.335 | 4.488 | 4.944 | 48.13 | **66.66** |

**Granite-4.0-h-micro-base (flashinfer, ctx 8192)**

| 정책 | rate | goodput(PREREG) | goodput(정본 p95) | throughput req/s | TTFT p50 | p95 | p99 | TPOT p50 | p95 |
|---|---|---|---|---|---|---|---|---|---|
| plain | 2 | 1.965 ± 0.359 | 1.904 ± 0.256 | 1.968 ± 0.359 | 0.123 | 0.222 | 0.298 | 8.44 | 13.21 |
| agnostic | 2 | 1.968 ± 0.369 | 1.968 ± 0.369 | 1.972 ± 0.370 | 0.168 | 0.375 | 0.503 | 8.73 | 11.19 |
| plain | 3 | 2.933 ± 0.254 | 2.619 ± 0.258 | 2.933 ± 0.254 | 0.126 | 0.284 | 0.350 | 9.90 | 17.86 |
| agnostic | 3 | 2.924 ± 0.253 | 2.924 ± 0.253 | 2.924 ± 0.253 | 0.177 | 0.503 | 0.634 | 9.73 | 12.89 |
| plain | 4 | 3.975 ± 0.317 | 3.103 ± 0.304 | 3.975 ± 0.317 | 0.153 | 0.320 | 0.374 | 12.54 | 20.80 |
| agnostic | 4 | 3.943 ± 0.292 | 3.943 ± 0.292 | 3.943 ± 0.292 | 0.243 | 0.686 | 1.047 | 11.25 | 13.80 |
| plain | 6 | 5.649 ± 0.683 | 2.021 ± 0.692 | 5.702 ± 0.743 | 0.218 | 0.587 | 0.729 | 22.94 | 40.81 |
| agnostic | 6 | 5.557 ± 0.597 | 5.557 ± 0.597 | 5.557 ± 0.597 | 0.691 | 1.465 | 1.754 | 13.27 | 14.97 |

⚠ = 절벽 위(§1). TPOT p99는 ITL 미기록 요청이 2건 이상인 셀에서 ∞가 되므로 생략했다
(p50·p95는 유한 요청만으로도 동일 순위에서 결정됨).

**baseline 분산 먼저**: plain의 rep 간 SD는 Zamba2 0.160/0.243/**0.755**/0.519,
Granite 0.359/0.254/0.317/0.683 (req/s). Zamba2 r4의 SD 0.755는 평균의 **28%** —
절벽 셀의 전형적 증폭이다.

### 3.2 paired bootstrap (agnostic − plain), n=5 pairs, seed=1, 10000 samples

**사전등록 지표 — goodput @ (TTFT ≤ 3 s ∧ per-request mean ITL ≤ 60 ms)**

| 모델 | rate | plain mean ± SD | agnostic mean | mean_effect | effect% | CI95 (req/s) | SD(effect) | 0 배제 |
|---|---|---|---|---|---|---|---|---|
| Zamba2 | 2 | 1.848 ± 0.160 | 1.847 | −0.0017 | **−0.09%** | [−0.0035, +0.0006] | 0.0026 | no |
| Zamba2 | 3 | 2.894 ± 0.243 | 2.949 | +0.0555 | **+1.92%** | [−0.0082, +0.1680] | 0.1230 | no |
| Zamba2 | 4 ⚠ | 2.705 ± 0.755 | 3.834 | +1.1295 | **+41.76%** | [+0.5105, +1.6503] | 0.7455 | **YES** |
| Zamba2 | 6 ⚠ | 0.513 ± 0.519 | 2.554 | +2.0409 | **+397.56%** | [+1.6133, +2.5548] | 0.6178 | **YES** |
| Granite | 2 | 1.965 ± 0.359 | 1.968 | +0.0034 | **+0.17%** | [−0.0026, +0.0134] | 0.0111 | no |
| Granite | 3 | 2.933 ± 0.254 | 2.924 | −0.0091 | **−0.31%** | [−0.0136, −0.0044] | 0.0058 | YES(음) |
| Granite | 4 | 3.975 ± 0.317 | 3.943 | −0.0323 | **−0.81%** | [−0.0549, −0.0171] | 0.0257 | YES(음) |
| Granite | 6 | 5.649 ± 0.683 | 5.557 | −0.0922 | **−1.63%** | [−0.1730, −0.0315] | 0.0911 | YES(음) |

**정본 지표 — goodput @ (TTFT ≤ 3 s ∧ request 내부 p95 token-ITL ≤ 60 ms)**

| 모델 | rate | plain mean ± SD | agnostic mean | effect% | CI95 (req/s) | 0 배제 |
|---|---|---|---|---|---|---|
| Zamba2 | 2 | 1.314 ± 0.056 | 1.847 | **+40.50%** | [+0.4310, +0.6198] | YES |
| Zamba2 | 3 | 1.032 ± 0.268 | 2.949 | **+185.84%** | [+1.4844, +2.4065] | YES |
| Zamba2 | 4 ⚠ | 0.562 ± 0.373 | 3.376 | **+501.03%** | [+2.1767, +3.2970] | YES |
| Zamba2 | 6 ⚠ | 0.221 ± 0.339 | 1.621 | **+634.06%** | [+0.9685, +1.8802] | YES |
| Granite | 2 | 1.904 ± 0.256 | 1.968 | **+3.37%** | [+0.0008, +0.1732] | YES(경계) |
| Granite | 3 | 2.619 ± 0.258 | 2.924 | **+11.65%** | [+0.1409, +0.5149] | YES |
| Granite | 4 | 3.103 ± 0.304 | 3.943 | **+27.04%** | [+0.4972, +1.2649] | YES |
| Granite | 6 | 2.021 ± 0.692 | 5.557 | **+174.92%** | [+2.6814, +4.5919] | YES |

### 3.3 여집합 음성대조 — SLO-무관 throughput (게이트 #10)

| 모델 | rate | request throughput effect% [CI95 req/s] | output tok/s effect% |
|---|---|---|---|
| Zamba2 | 2 | −0.09% [−0.0035, +0.0006] | −0.09% |
| Zamba2 | 3 | −0.12% [−0.0167, +0.0136] | −0.12% |
| Zamba2 | 4 | −0.24% [−0.0238, +0.0050] | −0.24% |
| Zamba2 | 6 | **+5.42%** [+0.0766, +0.3982] | **+5.42%** |
| Granite | 2 | +0.17% [−0.0026, +0.0134] | +0.17% |
| Granite | 3 | −0.31% [−0.0136, −0.0044] | −0.31% |
| Granite | 4 | −0.81% [−0.0549, −0.0171] | −0.81% |
| Granite | 6 | **−2.55%** [−0.2732, −0.0315] | −2.55% |

읽는 법:
- **Zamba2 r4**: goodput +41.8%인데 throughput은 −0.24%(무효과). 즉 r4의 이득은
  "더 많이 처리"가 아니라 **"같은 양을 SLO 안에서 처리"** — 순수 지연분포 효과이며,
  임계 지시함수 증폭을 정면으로 맞는 구간이다.
- **Zamba2 r6**: throughput도 **+5.4%(CI 0 배제)** — 여기서만 SLO와 무관한 실제 용량
  차이가 존재한다.
- **Granite**: goodput(정본 지표)에서는 +3.4~+175%인데 throughput은 −0.3~−2.6%.
  즉 Granite의 정본-지표 이득은 **전적으로 지연 분포(꼬리) 효과**이고 처리량은 오히려
  pdmux가 근소하게 손해다. (§1-16 전례와 같은 형태의 정보.)

### 3.4 J3 — 死因(TPOT 벽)이 운영점에서 어디에 다시 나타나는가

plain arm의 per-request TPOT (ms, 5 rep 평균):

| 모델 | 지표 | r2 | r3 | r4 | r6 | (Phase 0) r8 |
|---|---|---|---|---|---|---|
| Zamba2 plain | p50 | 12.78 | 22.10 | **48.51** | **91.32** | 60.58 |
| Zamba2 plain | p95 | 23.78 | 43.17 | **81.18** | **112.26** | 121.84 |
| Granite plain | p50 | 8.44 | 9.90 | 12.54 | 22.94 | 52.50 |
| Granite plain | p95 | 13.21 | 17.86 | 20.80 | 40.81 | 86.38 |

**요청 단위 위반 분해 (5 rep pooled, 600 req/셀):**

| 모델 | 정책 | rate | TTFT>3s | mean-ITL>60ms | p95-ITL>60ms | (p95 위반 ∧ mean 정상) |
|---|---|---|---|---|---|---|
| Zamba2 | plain | 2 | 0 | 6 | 176 | 170 |
| Zamba2 | plain | 3 | 0 | 11 | 384 | 373 |
| Zamba2 | plain | 4 | 7 | **184** | 510 | 330 |
| Zamba2 | plain | 6 | **169** | **477** | 541 | 64 |
| Zamba2 | agnostic | 4 | 16 | 4 | 87 | 83 |
| Zamba2 | agnostic | 6 | 187 | 166 | 361 | 195 |
| Granite | plain | 2 | 0 | 1 | 16 | 15 |
| Granite | plain | 3 | 0 | 0 | 63 | 63 |
| Granite | plain | 4 | 0 | 0 | 128 | 128 |
| Granite | plain | 6 | 0 | 5 | **378** | 373 |
| Granite | agnostic | 2/3/4/6 | 0 | ≤1 | ≤1 | 0 |

**답:**
1. **Zamba2에서는 TPOT 벽이 운영점에서도 살아 있다.** cudagraph-ON에서도 plain의
   TPOT p95는 rate 3→4 사이에서, p50은 rate 4→6 사이에서 60 ms를 넘는다. rate 4에서
   184/600(31%), rate 6에서 477/600(80%)이 mean-ITL SLO를 위반한다. **P1이 Zamba2에서
   살아남는 이유는 정확히 "TPOT 벽"이다.**
2. **Granite에서는 채점 구간의 TPOT 벽이 사라졌다.** plain TPOT p95가 rate 6에서도
   40.8 ms로 SLO 아래이고 mean-ITL 위반은 600건 중 5건. 벽은 rate ≈ 8로 밀렸다
   (Phase 0 n=1, r10·r12가 다시 정상으로 돌아오는 **비단조** 구간이라 이 위치 자체는
   신뢰도 낮음). **그래서 사전등록 지표에서 Granite의 P1이 강등된다.**
3. **다만 pdmux가 실제로 없애는 것은 "느린 decode"가 아니라 "멈춘 decode"다.**
   요청당 60 ms 초과 토큰 간격(=prefill이 decode를 막은 사건) 횟수와 요청 내 최대 ITL:

   | 모델·정책 | r2 | r3 | r4 | r6 | max-ITL p95 (r2/r3/r4/r6, ms) |
   |---|---|---|---|---|---|
   | Zamba2 plain | 3.97 | 7.97 | 14.87 | 17.70 | 381 / 994 / 2096 / 2960 |
   | Zamba2 agnostic | 0.04 | 0.02 | 3.57 | 24.71 | 37.5 / 54.1 / 75.3 / 238 |
   | Granite plain | 1.65 | 2.61 | 3.67 | 7.12 | 275 / 323 / 474 / 1848 |
   | Granite agnostic | 0.01 | 0.01 | 0.02 | 0.01 | 41.0 / 40.6 / 39.4 / 40.9 |

   Granite agnostic은 rate 6에서도 요청당 stall 0.01회, 최대 ITL p95 41 ms로 완전히
   평탄하다. plain은 최대 ITL이 최대 1.8 s까지 튄다. **mean-ITL 지표는 95개 토큰 중
   7번의 stall을 평균으로 희석해 이 차이를 0으로 만들고, p95-ITL 지표는 그대로
   드러낸다.** Granite에서 두 지표가 정반대 판정을 내는 기전이 이것이다.

### 3.5 J4 — TTFT (§3.1 표에 병기, 요약)

- agnostic은 **저부하에서 TTFT p50이 체계적으로 나쁘다**(prefill이 SM을 나눠 갖기
  때문): Zamba2 r2 0.230 s vs plain 0.164 s, Granite r6 0.691 s vs plain 0.218 s.
- 고부하에서는 역전: Zamba2 plain r6 TTFT p95/p99 = 5.580/6.084 s vs agnostic
  4.488/4.944 s.
- TTFT SLO(3 s) 위반은 Zamba2 r6에서만 대량 발생(plain 169, agnostic 187/600) —
  **rate 6에서는 TTFT 위반 수가 agnostic이 오히려 많다.** Zamba2 r6의 goodput 이득은
  TTFT가 아니라 ITL 축에서 온다.

---

## 4. 용량(sustainable rate) — 게이트 #6 "용량 먼저"

Phase 0 달성 throughput(req/s, offered rate 대비):

| 모델·정책 | r1 | r2 | r3 | r4 | r5 | r6 | r8 | r10 | r12 |
|---|---|---|---|---|---|---|---|---|---|
| Zamba2 plain | 1.07 | 2.14 | 3.14 | 3.97 | 3.82 | 4.44 | 3.87 | 4.16 | 4.26 |
| Zamba2 agnostic | 1.07 | 2.13 | 3.12 | 3.93 | 4.39 | 4.48 | 4.44 | 4.48 | 4.44 |
| Granite plain | 1.02 | 2.04 | 3.01 | 3.94 | 4.83 | 5.67 | 5.42 | 7.50 | 7.47 |
| Granite agnostic | 1.02 | 2.03 | 2.98 | 3.89 | 4.75 | 5.56 | 6.36 | 6.47 | 6.47 |

위반율 ≤ 5% 기준 최대 offered rate(**n=1·60 prompts 스캔이므로 지시적 값**):
- Zamba2: plain ≈ **4**(정확히 5.0%), agnostic ≈ **5**(1.7%) — pdmux가 SLO 용량을
  약 1 rate 단위 올린다.
- Granite: plain은 r6까지 0% 위반이나 r8에서 65% 위반 후 r10/r12에서 1.7%/3.3%로
  복귀하는 **비단조** — r8 셀은 n=1 일시적 이상으로 의심되며 용량 추정 불가.
  agnostic은 r8까지 0%, r10 5%, r12 35%.
- 원 throughput 포화점: Zamba2 plain ≈ 4.3, agnostic ≈ 4.5(pdmux 우위);
  Granite plain ≈ 7.5, agnostic ≈ 6.5(**plain 우위** — pdmux가 raw 처리량은 손해).

---

## 5. J6 — agnostic + cudagraph 조합의 건전성 (최초 측정)

**결론: J2를 무효화할 이상 없음. 단 미계측 2건.**

| 점검 | 결과 |
|---|---|
| cudagraph 실제 ON | Phase-1 80개 결과행 전부 `server_info.disable_cuda_graph = False`, `disable_piecewise_cuda_graph = False` |
| decode graph 캡처 성공 | plain = 1 pass × bs[1..48] 10개 / agnostic = **4 pass × 10개**(stream group 4개 각각). 캡처 메모리 Zamba2 0.13→0.52 GB, Granite 0.23→**0.85 GB**(≈4×) — 그룹별 캡처가 실제로 일어난 물증 |
| eager fallback | decode 경로 없음. **piecewise(prefill) graph는 양 arm 모두 OFF**이나 사유가 다름 — plain: "some layers do not apply Standard GQA"(hybrid SSM 구조상 자동 비활성), agnostic: "the capture size is not set". **순효과는 대칭**(양쪽 다 decode-only graph) ⇒ piecewise 비대칭 교락 없음 |
| pdmux 부팅 | `PD-Multiplexing enabled with 4 stream groups, sm_counts (prefill_sm, decode_sm): [(108,0), (74,34), (54,54), (0,108)]` — 양 모델 |
| 정확성 게이트 | 24 boot(2 모델 × 2 정책 × (capscan+5 rep)) 전부 `boot_ok=1`, `CORRECTNESS=PASS`. capscan sanity의 `output_ids`가 plain·agnostic **완전 동일**(Zamba2 `[5465,28723,13,1014,5565,302,272,2969]`, Granite `[12366,13,578,6864,315,279,3723,4273]`) |
| KV 풀 동일성 | Zamba2 327,637 vs 327,536 tokens / Granite 7,230,374 vs 7,228,106 (차이 0.03%) — 교락 아님 |
| 순서 무작위화 | Zamba2 coin: agn-first 2회·plain-first 3회 / Granite: plain-first 3회·agn-first 2회 |
| backend·플래그 동일성 | attention_backend, dtype, mem_fraction_static 0.82, max_running_requests 48, disable_radix_cache, context_length, cuda_graph_bs 전부 arm 간 동일. 두 job은 같은 커밋 트리(`runtime_source_manifest_*.sha256` 존재) |

**미계측 / 남는 교락 2건 (판정에 병기 필수):**

1. **실현 파티션 미측정.** `PDMUX_TELEMETRY_PATH`가 설정되지 않아 telemetry.jsonl이
   없다 ⇒ **`controller_summary`(split_transitions / residency / dwell) 계산 불가**
   (게이트: dynamic arm 병기 요건 부분 미충족). 파티션 실현의 근거는 (a) 부팅 설정
   로그, (b) 그룹별 4회 캡처, (c) **행동 증거** — decode stall이 요청당 1.65~17.7회
   → 0.01~0.04회로 붕괴하고 최대 ITL이 1.8 s → 41 ms로 잘리는 것은 prefill이 decode를
   막지 않고 동시 실행됐다는 뜻이다. 강한 간접 증거이나, "target≠realized" 전례
   (Stage 0의 D108 앵커가 실제로 16 SM이었던 건) 때문에 **직접 측정으로 격상 필요**.
2. **plain↔agnostic은 1-플래그 대조가 아니라 3-플래그 묶음이다.**
   `--enable-pdmux` + `--chunked-prefill-size -1` + `--disable-overlap-schedule`이
   함께 바뀐다(pdmux가 요구). PREREG의 "바뀌는 것은 cudagraph 하나뿐"은 **원 캠페인
   대비 축**에서는 참이지만, **이 캠페인 내부의 정책 대조**에는 해당하지 않는다.
   두 부수 플래그의 부호는 대체로 agnostic에 불리(overlap schedule 비활성은 핸디캡)
   할 것으로 보이나 **측정되지 않았다**. Zamba2 효과를 "PD 분리 때문"이라고 귀속하려면
   `plain + chunked-prefill -1 + disable-overlap`(pdmux 없이) 제3 arm이 필요하다.

부수 관찰(이상 아님): paired arm 간 생성 텍스트는 요청의 85~92%만 일치한다. 두 arm의
batch 구성이 다르므로 `--enable-deterministic-inference` 없이는 정상 범위의 배치 의존
비결정성이며, greedy 정확성 게이트는 양쪽 모두 동일 토큰으로 통과했다.

---

## 6. J5 — 사전등록 규칙 대조 (기계적)

PREREG §"사전등록 판정 규칙" 원문 3조를 **사전등록 지표(mean-ITL SLO)** 에 적용.

### Zamba2-2.7B → **P1 운영점 확인 (CONFIRMED), rate ≥ 4 한정**

- 조 1(축소) 전제 "rate ≤ 4에서 차이 < 3%": **거짓**(rate 4 = +41.76%) ⇒ 미발화.
- 조 2(전면 강등) 전제: 조 1이 미발화이므로 미발화.
- 조 3(확인) "차이 ≥ 3% ∧ paired CI가 0 배제": **rate 4에서 발화**
  (+41.76%, CI [+0.5105, +1.6503] req/s), **rate 6에서도 발화**
  (+397.56%, CI [+1.6133, +2.5548]).
- 부하 의존성 명시: rate 2 −0.09%(CI 0 포함), rate 3 +1.92%(CI 0 포함) ⇒ **저부하
  구간에는 효과 없음.** "rate ≤ 4"를 블록으로 다루는 규칙 문언과 실제 데이터가
  블록 내부에서 이질적이다 — **규칙 문언의 모호점으로 보고**하며, 임의 재해석하지
  않는다(조 1의 전제가 문자 그대로 거짓이므로 조 3 적용이 유일한 기계적 귀결).
- ⚠ **효과 크기는 신뢰 불가**: rate 4·6은 절벽 위(§1.3–1.5)이고 위반율이 런 길이에
  의존한다. 살아남는 것은 **부호와 존재**(rate 4에서 5 rep 중 4 rep이 양, CI 0 배제)
  이지 +41.8%/+398%라는 숫자가 아니다.

### Granite-4.0-h-micro-base → **P1 전면 강등 (DEMOTED)**

- 조 1 전제 "rate ≤ 4에서 차이 < 3%": **참**(max |diff| = 0.81%) ⇒ 축소 발화.
- 조 2 전제 "rate 6에서도 차이 < 3%": **참**(1.63%) ⇒ **전면 강등 발화**.
- 조 3: 전 rate에서 3% 미달이므로 미발화. rate 3/4/6은 CI가 0을 배제하지만 **부호가
  음(agnostic이 근소하게 열위)** 이며 크기가 헤드라인 하한(3%) 미만 — "통계적으로는
  분해되나 실무적으로 무의미하고 방향은 P1의 반대".
- 이 셀들은 절벽 위가 **아니다**(위반율 0.000~0.008, 전·후반 정상 상태) ⇒ 이 강등
  판정은 절벽 아티팩트가 아니라 깨끗한 측정이다.

### 사전등록 밖(그러나 사전 고정) 정본 지표에서의 대조

정본 게이트 #4 정의(p95 token-ITL)로 같은 규칙을 적용하면 **두 모델 모두 전 rate에서
조 3 발화**(Granite rate 2 +3.37% CI [+0.0008, +0.1732]는 경계선, rate 3 이상은 여유
있게 발화) ⇒ **P1 운영점 확인**. 즉 **Granite의 판정은 지표 정의가 100% 결정한다.**
- 이는 사후 지표 교체가 아니다: PREREG가 mean-ITL을 명시 선택하며 그 이유(원 캠페인
  과의 직접 대조)를 사전 기록했고, p95-ITL은 `CLAUDE.md` 게이트 #4로 사전 고정돼 있다.
- 재스코어 금지(게이트 #7)에도 저촉되지 않는다 — SLO **임계값**을 바꾼 게 아니라 같은
  60 ms 임계에 대해 **요청 내부 집계자**(mean vs p95)가 다른 두 사전 정의를 병기한
 것이며, 컨트롤러 재튜닝이 필요한 종류의 변경이 아니다. 다만 어느 정의가 정본
  claim을 지배하는지는 **정의 판단**이므로 claims-auditor로 넘긴다.

---

## 7. 게이트 대조표

| 게이트 | 상태 | 근거 |
|---|---|---|
| n ≥ 4, baseline 분산 우선 보고 | **통과** | n_indep = 5 rep/셀, plain SD 전부 §3.1에 병기 |
| 3% 미만은 헤드라인 아님 | **통과** | Granite 전 rate < 3% ⇒ 강등. Zamba2 r2/r3도 헤드라인 아님으로 처리 |
| stationary 단독 비교 금지 | **해당 없음/주의** | 이 캠페인은 정책 A/B(정적 구성 대조)이지 동적 컨트롤러 비교가 아니다. 다만 **정상상태 단일 rate 격자**이므로 변화 trace 결론으로 확장 금지 |
| metric cliff | **부분 위반, 명시함** | Zamba2 r4·r6가 절벽 위. 용량·TTFT 분포·런 길이 의존성 §1.5/§4에 병기 |
| duration 합산 버그 | **해당 없음(검증)** | 한 줄 = 한 rate 셀, 라운드 append 구조 아님 |
| dynamic arm의 split telemetry 병기 | **미충족** | telemetry 미수집 ⇒ controller_summary 불가(§5-1) |
| SLO 변경 시 재튜닝 | **저촉 없음** | 임계값 불변, 사전 고정된 두 집계자 병기 |
| arm 간 조건 동일성 | **부분** | cudagraph/backend/seed/메모리/커밋 동일. **정책 축은 3-플래그 묶음**(§5-2) |
| 격자 간 수치 이전 금지 | **준수** | 원 4-모델(no-cudagraph) 캠페인 수치를 이 표에 넣지 않았다. 비교는 서술로만 |

---

## 8. 다음에 필요한 것

1. **실현 파티션 직접 계측 재현런**: `PDMUX_TELEMETRY_PATH` 설정 후 agnostic arm
   1 rep 재실행 → `controller_summary`(split_transitions/residency/dwell) 산출.
   현재 §5-1의 행동 증거를 realized 측정으로 격상. (GPU 약 10분.)
2. **3-플래그 묶음 분해**: `plain + --chunked-prefill-size -1 --disable-overlap-schedule`
   (pdmux 미사용) 제3 arm을 Zamba2 rate {3,4,6}에 n=5로 추가. 이것 없이는 Zamba2의
   +41.8%를 "PD 분리 덕분"으로 귀속할 수 없다. (약 1 job.)
3. **Zamba2 크기 추정은 goodput 대신 용량으로**: 절벽 위 threshold-goodput 대신
   "위반율 ≤ 5%를 만족하는 최대 sustainable rate"를 n≥4로 직접 측정
   (현재 plain ≈ 4 / agnostic ≈ 5는 n=1·60 prompts 지시값).
4. **Granite rate 8 비단조 구간 재측정**(n≥4): 현재 r8만 65% 위반이고 r10/r12는 정상
   — 용량 곡선을 신뢰하려면 필요.
5. **지표 정의 판단**: Granite의 P1 생사가 mean-ITL vs p95-ITL로 갈린다. 추가 데이터로
   해결되지 않는 정의 문제이므로 claims-auditor 판단 사항.

---

## 부록: 재현

```bash
# 정본 라이브러리 경유 채점 (본 문서의 모든 수치)
PYTHONPATH=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/benchmarks \
  python3 <scratch>/p1op_verdict.py      # 셀 요약 + paired bootstrap + 규칙 적용
PYTHONPATH=... python3 <scratch>/p1op_failmode.py   # 위반 분해 + 용량 + 절벽 비율
# 사전등록 스크립트(동일 데이터, mean-ITL 지표만)
python3 workspace/engine-port/results/p1_opint/p1op_analyze.py \
  --tag zamba2-27b --tag granite-40-h-micro-base
```

정본 모듈 변경: `pdmux_eval/analyze.py`에 `request_tpot_percentiles()` 추가
(가산적 순수 함수, 기존 `percentile` 규약 사용, 미기록 ITL은 기본 `inf`=위반 처리).
