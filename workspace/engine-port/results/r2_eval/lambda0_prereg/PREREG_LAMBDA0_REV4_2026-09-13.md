# 사전등록 rev4 — 캠페인 **0단계**: λ\*(용량) 측정, shape별 × 기준 arm B1

**작성** 2026-09-13 (engine-porter) · **상태: 미실행, 이 판본의 GPU 지출 0** · **제출 전**

**계보**: rev1 `NO-GO`(D1–D15) → rev2 `NO-GO`(D16–D23) → rev3 `NO-GO`(E1–E7) → **이 rev4**.
세 판본 모두 SUPERSEDED 배너와 함께 보존한다(삭제 금지).

**rev3 3차 감사의 요지**: D16–D23 중 6건이 닫힌 것으로 확인됐고 배수 항등식·rr=1.0
필요조건·D21·D23·D4 음성대조·인용 고정은 건강했다. 死因은 **등록 예보 F2("shape B =
`KNEE_BRACKETED`")의 참 분지가 정의역 ∅**이라는 것이었고, 판정서 §3이 **그 死因의 근인이
감사자 자신의 rev2 처방(D16/D17)의 granularity 결함**임을 자기 철회로 명시했다(자기 철회
6회차). rev4는 그 한 곳(E1)을 고치고, **같은 형태의 결함이 남아 있지 않은지 스스로
재생성해 확인**했다(§3, 게이트 #110).

★**λ_inf가 실측됐다** — job 907959(2026-09-13, 0.356 GPU-h). 이 판본의 사다리는 추정이
아니라 그 측정 위에 선다(§4).

사용자 결정 3건이 선행했다: 모델 = **`nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base`**,
백엔드 = **flashinfer**, 1차 범위 = **Claim D + P3**.

---

## 0. 왜 0단계인가 — 그리고 ★이 단계가 닫지 못하는 것

캠페인 워크로드는 전부 λ\*의 분수로 정의된다. `scripts/r2_eval/generate_campaign.sh:17`
(HEAD `bddff6a`)의 `sustainable_rate`는 **측정값이 아니라 기본값 `4`**였다.

★★**그러나 이 단계는 어떤 결과가 나와도 방법론 게이트 #6("용량 먼저 측정")을 닫지
못한다.** 게이트 #6이 요구하는 용량은 **지표 절벽 대비** 용량이고, 이 단계가 재는 것은
throughput 포화다. rev1 판정서가 job 905835 d44 원자료를 정본 goodput 술어(TTFT ≤ 3000 ms
∧ 요청-내부 token-ITL p95 ≤ 60 ms, `benchmarks/pdmux_eval/campaign.py:139-140` @HEAD
`bddff6a`)로 직접 재채점한 결과: **0.59·λ\*_thr에서 goodput 53.8% · 0.89·λ\*_thr에서
5.8% · 1.27·λ\*_thr에서 0.8%** ⇒ **λ\*_SLO(B) < 0.59 × λ\*_throughput**(비 추정 2.3–3.4×).
SLO 절벽은 이 단계가 재는 수보다 **아래**에 있다. §9 Q1으로 문자 고정한다.

---

## 1. λ\*의 정의 · 정본 정의 교체의 등재 (D14, 유지)

> **λ\*(shape) = 기준 arm B1(legacy 루프, PD-mux fixed D44)에서 그 shape의
> open-loop(Poisson) throughput 포화율.**

**D14**: `reports/paper/EXPERIMENT_ROADMAP.md:669`는 λ\*를 "sustainable **SLO** rate"로
정의한다. **이 등록은 그것을 의도적으로 throughput 포화로 교체한다**(실측 비 shape B에서
≥1.7×). **λ\*_SLO는 이 단계 이후에도 측정되지 않는다.**

**왜 arm을 고정하는가**: λ\*는 arm마다 5× 다르다(job 905835: D16 0.933 / D44 0.675 /
D92 0.187 req/s). ★워크로드 이름은 arm을 넘어 거짓이 된다(§9 Q4).

---

## 2. 측정 격자 · 통제 요인 (rev3에서 불변 — 감사가 닫힌 것으로 확인한 부분)

### 2.1 ONE BOOT PER CELL
셀마다 별 boot·별 telemetry. 기제 승계 = `c_capacity.sbatch:111`의 셀 루프 + `:123-178`의
셀별 launch/kill/`pkill`(`:30-35`는 주석이지 기제가 아니다). ★**더 강한 차단**: 판정량은
**전부 client-side**이고 telemetry를 경유하지 않는다(`CONSENSUS §3 항목120` 회피).

### 2.2 shape (D1, 유지)

| shape | 정체 | 앵커 |
|---|---|---|
| **A = (256 in, 512 out)** | **W3 전부**이자 **W4 decode phase 그 자체** | **λ_inf(A) = 3.0939 req/s**(측정, §4) |
| **B = (8192 in, 64 out)** | **W4 prefill phase** | **λ_inf(B) = 0.6956 req/s**(측정, §4) |

**D1**: `benchmarks/pdmux_eval/workloads.py:159-160` 실행 확인 — W4 decode phase =
(in 256, out 512) = shape A 그 자체. **근사는 존재하지 않는다.** (in 64, out 512)는
저장소에 없다.

### 2.3 통제 요인 (캠페인 러너와 일치)
`--attention-backend flashinfer` · ctx **16384** · `--mem-fraction-static 0.82` ·
`--max-running-requests 48` · `--disable-radix-cache` · `--chunked-prefill-size -1` ·
`--disable-overlap-schedule` · cudagraph **ON** · `--random-seed 1` · PD-mux `fixed D44`
(legacy 루프) · TP=1 · config `pdmux_homog5.yml`.

### 2.4 D11 — probe C(905835)와의 차이

| 항목 | probe C | 이 등록 | 캠페인 러너 | 처리 |
|---|---|---|---|---|
| mem-fraction / ctx | 0.80 / 8704 | **0.82 / 16384** | 0.82 / 16384 | 캠페인에 맞춤 |
| **`--disable-piecewise-cuda-graph`** | **ON**(`c_capacity.sbatch:126`) | ★**붙이지 않는다** | cudagraph-OFF 분기에서만 | **캠페인에 맞춤**(8192-token prefill 처리율을 바꿀 수 있음 — 미검증으로 등록) |
| **`--random-seed`** | 미지정 | **1** | run record | 등록 |
| **부하 생성기** | `bench_serving` Poisson | 동일 | **`pdmux_eval.trace_loadgen`** | ★**해소되지 않는 불일치로 등록**(§10-6) |
| pdmux config | `pdmux_homog5.yml`(그룹 5) | 동일 | `pdmux_r2.yml`(6) | ★**감사가 확인·해소**: `(64,44)` 분할이 양쪽에 있고 `_r2_decide_idx`가 `decode_sms` 정확 일치로 고르므로 **fixed 정책 아래 동등** |

**ctx 16384 / mem 0.82가 용량을 바꾸는가 — 아니다**(구속은 `max_mamba_cache_size = 48`).
★**이제 부분 실측됨**: job 907959 I1 배너가 5 boot 전부에서
`max_total_num_tokens = 2,721,290` · `available_gpu_mem = 11.59 GB`를 보였다. sbatch는
셀마다 같은 배너를 기록해 **확인 항목**으로 유지한다.

### 2.5 D10 — 하네스 줄
`--dataset-name random --dataset-path $SGPT_RAW --tokenize-prompt --random-range-ratio 1.0
--output-details --output-file --model --host --port`. 입력·출력 길이 정확성이 셀마다
`input_len_exact`/`output_len_exact`로 기록된다. ★`--random-range-ratio 1.0`의 두 번째
역할은 §5.3(D20)에 있다.

### 2.6 D12 — provenance를 GPU 전에 확인
첫 boot 전에 manifest 항목 수 ≥ **24**와 **`models/nemotron_h.py` 포함**을 확인하지
못하면 `exit 3`. (감사 확인: HEAD `b2b60e7`에서 **25항목**, nemotron_h 포함 ⇒ 통과.)

---

## 3. ★E1 — 死因 해소: 자격(eligibility)과 역할(role)의 분리

### 3.1 무엇이 죽어 있었나 (내가 독립 재계산해 확인)

**판정량의 정체(rev2에서 확립, 여기서 유지)**: `achieved/realized_offered`는
`(N/(N−1)) · span / duration`이고 `duration = span + L`(L = 배수)이므로 곧
**`(N/(N−1)) · span / (span + L)`**이다 — 모델링 선택이 아니라 **항등식**이며 보관 12셀에서
`duration = maxᵢ(arrivalᵢ + e2eᵢ)`가 ≤0.27 s로 재현된다. 비포화 셀의 영점은 1이 아니라
**κ**(코드·표에서는 `kappa`/`kappa_pred`)다. shape A의 배수는 **511 토큰 × ITL**이고, probe C가 **이 arm(D44)**에서 실측한
ITL p50은 **19.92 / 20.29 / 20.38 / 23.53 ms**다(job 907959 I2의 ITL(B=1) 12.96 ms는
무경합 값이라 다른 양이다).

rev3의 R0은 **계획이 저측 후보로 인증한 모든 셀**에 `drain ≤ 2·L̂`를 걸고, 위반 1건이면
**그 shape 전체를 `UNRESOLVED`**로 만들었다. 그런데 R1의 **고측**(`ratio ≤ 0.90`)은 등록
자신의 항등식에 의해 **`drain ≥ ((N/(N−1))/0.90 − 1)·span`**을 뜻한다. 두 조건은 같은
셀에 상호배타적이다.

내 재계산(감사 수치를 상수로 받지 않고 `lambda0_plan`을 직접 돌려 산출):

| 시나리오 | shape | 인증 | 셀 | 가드 `2·L̂` | 고측이 요구하는 배수 | 비 |
|---|---|---|---|---|---|---|
| point lower | B | **4/4** | b_r0…b_r3 | 8.35 / 8.36 / 8.39 / 8.42 s | 21.56 / 20.95 / 19.90 / 20.19 s | **2.4–2.6×** |
| point upper | B | **4/4** | b_r0…b_r3 | 8.35–8.44 s | 19.53–21.63 s | 2.3–2.6× |
| abs upper | B | **4/4** | b_r0…b_r3 | 8.37–8.50 s | 19.24–20.37 s | 2.3–2.4× |
| point lower | A | 2/5 | a_r0, a_r1 | 23.70 / 28.22 s | 69.63 / 67.62 s | 2.4–2.9× |

⇒ **모든 인증 rung에서 예외 없이 충돌한다. 이것은 모형이 아니라 산술이다.**
shape B는 4/4가 인증되므로 **`KNEE_BRACKETED[B]`의 정의역이 ∅**이었다.

### 3.2 수리 (한 곳, 새 상수 0개, 새 estimand 0개)

> **배수 가드는 셀을 저측에서 실격시키고, shape를 무효화하지 않는다.**
> `usable_low = low_side_candidate AND drain_model_ok`. **고측에는 배수 조건이 없다** —
> 배수가 예측을 넘었다는 것은 **그 셀이 포화했다는 증거**이지 측정 실패가 아니다.
> R0에는 `ebar_guard_ok`·`range_ratio_ok`만 남는다.

근거: κ는 **예측 배수로부터** 계산된 천장이다. 배수가 예측을 넘으면 무효가 되는 것은
**그 셀의 천장 인증**뿐이고, 그 셀이 포화를 보였다는 사실은 온전하다.

### 3.3 ★E2 — 그리고 **같은 결함이 하나 더 있었다**(내가 찾음)

감사의 E1만 적용하면 충분한가? 아니었다. §5.4의 도달가능성 지도를 **쉬핑 코드로
재생성**하자 **`LADDER_TOO_LOW[shape A]`의 정의역이 여전히 ∅**임이 드러났다:

- rev3의 `LADDER_TOO_LOW`는 "**최상단 rung이 ≥ 0.95를 읽는다**"였다. 그런데 최상단 rung은
  `T = 170 s`에서 돌고 그 **천장 κ는 0.90–0.94**다 ⇒ 0.95를 **구조적으로 읽을 수 없다.**
  이것은 rev3 死因과 **정확히 같은 형태**(천장 아래에 문턱을 두는 것)의 재발이다.
- ★**감사 §3-2의 자기 철회도 재현되지 않는다.** 판정서는 "λ\*(A)=4.0은 실제로
  `LADDER_TOO_LOW`를 낸다"고 적었으나, 쉬핑 코드는 그 점에서
  **`KNEE_BRACKETED`(보고 3.742, −6.4%)**를 낸다. `LADDER_TOO_LOW[A]`는 앵커 사다리에서
  어떤 λ\*에서도 발화하지 않았고, fallback 사다리에서 λ\* ≈ 30(= 14×λ_inf)에서야 나왔다.

**수리**: `LADDER_TOO_LOW ⟺ 어떤 rung도 포화하지 않았다`(`not saturated`). 포화는 천장에
제약되지 않으므로 이 형태는 도달 가능하다. `LADDER_TOO_HIGH`(최하단 rung이 포화)와
대칭이다. 검증은 §5.4의 census.

### 3.4 왜 rev3의 36/36 하네스와 50 테스트가 이것을 못 봤는가 — 그리고 rev4의 조치

| 진단(감사) | rev4의 조치 |
|---|---|
| e2e leg의 고측이 `low_side_candidate=False`인 **유일한 구성**만 돌았다 | **E4**: `_end_to_end`에 **shape B 구성**(4/4 인증 + 포화 상단)을 추가. 이 leg는 rev3 규칙에서 실패한다 |
| `LADDER_TOO_HIGH` 증명이 `drain_ok=True` 기본값의 **손으로 만든 셀**을 썼다(교훈 9) | **E2**: 분지 존재 증명을 **쉬핑 analyze→rule 위의 λ\* 스윕**으로 교체 |
| 하네스가 `lambda0_cells.py`를 변이시키지 않았다(Y5 escape) | **E3**: MODULES를 **결정 경로 6파일 전체**로 확대, 렌더러 selftest가 인증 열이 계획과 일치함을 assert |
| `_run(None,None,env)`의 결과를 버리는 **죽은 CONTROL** | **E3**: CONTROL을 **변이가 도는 바로 그 temp-dir 경로**에서 채점. ★주입으로 확인: analyze 모듈을 사본에서 빼면 CONTROL이 `FAILS`하고 하네스가 **중단**한다(rev3라면 36/36이 공허 통과) |
| §3.3과 §3.4가 같은 셀에 상호배타적 요구 | E1이 그 모순을 제거(고측에 배수 조건 없음). §5.1 표가 셀별 역할을 분리해 적는다 |

---

## 4. ★λ_inf 실측 (job 907959) 과 D18 술어

### 4.1 측정값

| 산출물 | 값 |
|---|---|
| `I3a_shapeA.jsonl` (256,512) `--request-rate inf --max-concurrency 64` | **λ_inf(A) = 3.0939 req/s** (300 req / 96.97 s) |
| `I3b_shapeB.jsonl` (8192,64) 동일 프로토콜 | **λ_inf(B) = 0.6956 req/s** (300 req / 431.29 s) |
| `I3_max_running_req.txt` | **48** — 엔진 상한 도달(newpair **F5** 충족) |
| I1 배너(5 boot) | `max_total_num_tokens = 2,721,290` · `available_gpu_mem = 11.59 GB` · warm-up 분할 실현 측정(idx4 = (64,44)) |
| I2 (256,512) 동시성 1 | TTFT 바닥 **43.0 ms** · **ITL(B=1) median 12.96 ms** |

★**ITL 상수의 3중 확인**: 등록 상수 `ITL_SOLO_S = 13.06 ms`(P2 job 905712) · rev3 감사의
보관 셀 마지막(무경합) 요청 **13.48 / 13.27 / 13.07 ms**(D16/D44/D92, arm 무관) ·
job 907959 I2 **12.96 ms**(이 모델·이 shape). 셋이 일치한다.

★★**그럼에도 등록된 한계**: κ 모형이 필요로 하는 것은 **부하 배치(B̄ ≈ 6–27)의 ITL**이고
**그것은 여전히 미측정**이다(I2는 B=1만 쟀다). 그래서 §5.4의 지도는 "**등록 모형 아래의
도달가능성**"이지 실행 결과의 예측이 아니다. 모형이 틀렸을 때의 방어는 `drain_model_ok`
이며, E1 이후 그것은 shape를 무효화하지 않고 **그 셀을 저측에서 실격**시킨다.

### 4.2 ★NPC-I — 스코프 (§10에 필수 병기로 문자 승계)

> **λ_inf는 legacy warm-up boot · D44 · cudagraph ON에서만 측정되며, B4(true-dual)의
> 포화 상한은 측정되지 않았다.**

### 4.3 ★같은 job의 correctness 게이트는 `FAIL`이다 — 그리고 이 등록은 그 판정을 앞지르지 않는다

job 907959의 R2 correctness 게이트는 **`FAIL`**이다(true-dual boot 2개가 C층에서 CUDA
OOM). S층·O층 동등성은 불일치 0이었다. **원인 진단과 결과 감사가 병행 중이며, 이 FAIL이
0단계(legacy arm)에 무엇을 차단하는지는 아직 판정 전이다.** 이 사전등록은 그 판정을
가정하지 않는다 — 차단된다면 0단계는 실행되지 않고, 차단되지 않는다면 §5의 규칙 그대로
실행된다. 어느 쪽이든 **이 문서의 라벨 정의는 바뀌지 않는다.**

### 4.4 D18 — 앵커/fallback은 **파일이 정한다**(운영자 스위치 없음)

> **ANCHORED ⟺** (a) `I3a_shapeA.jsonl`·`I3b_shapeB.jsonl`이 둘 다 존재·파싱되고 각각
> `request_rate = inf`·`request_throughput > 0`, **그리고** (b) `I3_max_running_req.txt`의
> **모든 값이 ≥ 48**(newpair F5). 그 밖에는 **FALLBACK**.

실측 아티팩트에 대해 실행 확인: **`LAMBDA0_MODE=ANCHORED`**, A=3.093892399983411,
B=0.6955869013158362, 실격 사유 0건.

★**E6 — 앵커 provenance**: 술어는 해석된 **절대경로**와 **입력 3개의 sha256**을
`LAMBDA_INF_DECISION.txt`/`.json`에 기록하고, 그 JSON이 `LAMBDA0_LABEL.json`에 **그대로
임베드**된다. 근거: rev3 감사 U7이 디렉터리 선택만으로 shape A 라벨이 뒤집히는 실례를
보였다(앵커 vs fallback).

실측 아티팩트 digest(실행 시점 고정):
`I3a_shapeA.jsonl` `8539b43a…b6b12c8` · `I3b_shapeB.jsonl` `7fbc81e7…7764b55` ·
`I3_max_running_req.txt` `654ee9da…a27cc3ef`.

---

## 5. 판정 규칙 (데이터 생성 전 고정 — `lambda0_label.py`)

| # | 이름 | 규칙 | 성격 |
|---|---|---|---|
| **R0** | `CELL_VALIDITY[shape]` | 모든 셀: `1/Ē ∈ [0.95,1.05]` **및** `random_range_ratio = 1.0`. ★**배수 조건은 여기 없다**(E1) | 측정 유효성(위반 → `UNRESOLVED`) |
| **R1** | `KNEE_BRACKETED[shape]` | **저측으로 쓸 수 있는** 셀(`κ_pred ≥ 0.97` **AND** `drain_model_ok`)이 `ach/realized ≥ 0.95`이고, **offered가 더 높은** 셀에 `≤ 0.90`이 있다. **고측에 배수 조건 없음** | 존재, shape별 |
| **R2** | `LAMBDA_STAR[shape]` | 문턱 없는 **측정값** — 포화 셀(≤0.90) 중 **최대 achieved rate**. R1이 BRACKETED일 때만 | 측정 |
| **R3** | `SEED_REPEAT[shape]` | 제2 seed로 반복한 최상단 rung이 포화를 유지하고 λ\*를 **5% 이내** 재현. 브래킷 없으면 `UNRESOLVED (not evaluated)` | ★등록 예보 |
| — | 비브래킷의 **방향** | 최하단 rung 포화 → `LADDER_TOO_HIGH`(÷4) · **어떤 rung도 포화 안 함** → `LADDER_TOO_LOW`(×4) · 그 밖 → `KNEE_NOT_BRACKETED` | 재설계 결정성 |
| — | 기대 셀 결손 | 그 shape는 **`UNRESOLVED`**(측정 실패 ≠ 규칙 실패, 교훈 21) | |

**D4**: 기대 셀 이름 목록을 **인자로 받아 순회**(glob 아님). **D23**: EXPECT/REPEAT는 루프
**이전에** plan.json에서 구성 ⇒ 미시도 shape도 `UNRESOLVED`로 등재된다.
**D7**: F4/R3는 shape별 독립, 반복 셀은 사전에 최상단 rung으로 등록, R1 사다리에 넣지 않음.
**D21**: 평가되지 않은 예보를 반증으로 기록하지 않는다.

### 5.1 ★셀의 역할 표 (G-λ3-1: 자격 ≠ 역할)

| 셀의 상태 | 저측으로 쓸 수 있나 | 고측으로 쓸 수 있나 | R2 후보인가 |
|---|---|---|---|
| 인증(κ≥0.97) + 배수 예측 이내 + `r ≥ 0.95` | **예** | 아니오 | 아니오 |
| 인증 + **배수가 예측 초과** | **아니오**(천장 인증 무효) | **예**(그 초과가 포화의 증거) | `r ≤ 0.90`이면 **예** |
| 미인증(천장이 문턱 아래) | 아니오 | **예** | `r ≤ 0.90`이면 **예** |
| `0.90 < r < 0.95` | 아니오 | 아니오 | 아니오 |

### 5.2 출력공간 전수 (D8/D19)

shape별 verdict: `UNRESOLVED`(기대 셀 결손 또는 R0 위반) · `KNEE_BRACKETED`(λ\* 보고,
R3 평가) · `LADDER_TOO_HIGH`(÷4) · `LADDER_TOO_LOW`(×4) · `KNEE_NOT_BRACKETED`
(λ\* 미보고, R3 `UNRESOLVED`). ★어떤 조합도 정책·arm 순위·Claim 등급을 움직이지 않는다.
**D19**: rev2의 "사각지대 rung 최대 1개" 정리는 **철회**됐다(rev3에서 등재, 유지).

### 5.3 D2/D20 — 도착 실현과 그 필요조건 (유지)

판정량은 `achieved / realized_offered`, `realized_offered = (N−1)/span`. 명목 분모는
포화도가 아니라 **도착 실현 계수 `1/Ē`**다(905835 12/12 셀 동일 방향, `N ≥ 1537` 요구).
seed는 `1..5000` 중 `max|Ē−1|` 최소 2개로 **결정적 선택**.
★**D20**: `--random-range-ratio 1.0`은 길이 정확성뿐 아니라 **오프라인 도착 재현의
필요조건**이다(`benchmark/datasets/common.py:56-64`가 run당 2회 `randint`를 소비하고
rr=1.0에서만 퇴화 — 실행 확인). analyzer가 되읽어 ≠1.0이면 `UNRESOLVED`.
★rev2의 "Ē 가드 = 재현 파손 탐지" 서술은 **철회됨**(Ē는 재현 스트림 자신에서 계산된다).

### 5.4 ★E2 — 도달가능성 지도 (κ 표에서 멈추지 않는다)

아래는 `python3 lambda0_reachability.py`의 **출력 전문**이다. 7개 시나리오(측정값 포함)
× 2 shape × **참 λ\* 12점**에 대해 **쉬핑 `analyze()` → `rule()`**을 돌려 방출 verdict와
보고 오차를 인쇄한다. 손으로 만든 레코드는 하나도 없다(G-λ3-4).

```text
### MEASURED job 907959 (I3a 3.0939 / I3b 0.6956 req/s)   lambda_inf=(3.0939, 0.6956)
    lam*/lam_inf |            shape A            |            shape B
            0.10 | LADDER_TOO_HIGH    lam*= 0.309 err=    --   low=0/2 | LADDER_TOO_HIGH    lam*= 0.070 err=    --   low=0/4
            0.20 | LADDER_TOO_HIGH    lam*= 0.619 err=    --   low=0/2 | LADDER_TOO_HIGH    lam*= 0.139 err=    --   low=0/4
            0.25 | KNEE_NOT_BRACKETED lam*= 0.773 err=    --   low=0/2 | LADDER_TOO_HIGH    lam*= 0.174 err=    --   low=0/4
            0.35 | KNEE_BRACKETED     lam*= 1.083 err=  +0.0% low=1/2 | LADDER_TOO_HIGH    lam*= 0.243 err=    --   low=0/4
            0.50 | KNEE_BRACKETED     lam*= 1.547 err=  +0.0% low=2/2 | KNEE_NOT_BRACKETED lam*= 0.348 err=    --   low=0/4
            0.70 | KNEE_BRACKETED     lam*= 2.166 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.487 err=  +0.0% low=1/4
            0.85 | KNEE_BRACKETED     lam*= 2.630 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.591 err=  +0.0% low=2/4
            1.00 | KNEE_BRACKETED     lam*= 3.094 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.696 err=  +0.0% low=2/4
            1.20 | KNEE_BRACKETED     lam*= 3.713 err=  -0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.835 err=  +0.0% low=2/4
            1.45 | KNEE_BRACKETED     lam*= 4.486 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 1.009 err=  -9.1% low=3/4
            1.80 | KNEE_BRACKETED     lam*= 5.569 err=  -2.2% low=2/2 | LADDER_TOO_LOW     lam*= 1.252 err=    --   low=3/4
            2.50 | LADDER_TOO_LOW     lam*= 7.735 err=    --   low=2/2 | LADDER_TOO_LOW     lam*= 1.739 err=    --   low=4/4

### point estimate lower  (lambda*(A)=2.1 regression extrapolation)   lambda_inf=(2.1, 0.675)
    lam*/lam_inf |            shape A            |            shape B
            0.10 | LADDER_TOO_HIGH    lam*= 0.210 err=    --   low=0/2 | LADDER_TOO_HIGH    lam*= 0.068 err=    --   low=0/4
            0.20 | LADDER_TOO_HIGH    lam*= 0.420 err=    --   low=0/2 | LADDER_TOO_HIGH    lam*= 0.135 err=    --   low=0/4
            0.25 | LADDER_TOO_HIGH    lam*= 0.525 err=    --   low=0/2 | LADDER_TOO_HIGH    lam*= 0.169 err=    --   low=0/4
            0.35 | KNEE_BRACKETED     lam*= 0.735 err=  +0.0% low=1/2 | LADDER_TOO_HIGH    lam*= 0.236 err=    --   low=0/4
            0.50 | KNEE_BRACKETED     lam*= 1.050 err=  +0.0% low=1/2 | KNEE_NOT_BRACKETED lam*= 0.338 err=    --   low=0/4
            0.70 | KNEE_BRACKETED     lam*= 1.470 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.472 err=  +0.0% low=1/4
            0.85 | KNEE_BRACKETED     lam*= 1.785 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.574 err=  +0.0% low=2/4
            1.00 | KNEE_BRACKETED     lam*= 2.100 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.675 err=  -0.0% low=2/4
            1.20 | KNEE_BRACKETED     lam*= 2.520 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.810 err=  +0.0% low=2/4
            1.45 | KNEE_BRACKETED     lam*= 3.045 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.979 err= -10.3% low=3/4
            1.80 | KNEE_BRACKETED     lam*= 3.780 err=  -1.6% low=2/2 | LADDER_TOO_LOW     lam*= 1.215 err=    --   low=3/4
            2.50 | LADDER_TOO_LOW     lam*= 5.250 err=    --   low=2/2 | LADDER_TOO_LOW     lam*= 1.688 err=    --   low=4/4

### point estimate upper  (lambda*(A)=2.5 ctx-768 KV-term variant)   lambda_inf=(2.51, 0.8)
    lam*/lam_inf |            shape A            |            shape B
            0.10 | LADDER_TOO_HIGH    lam*= 0.251 err=    --   low=0/2 | LADDER_TOO_HIGH    lam*= 0.080 err=    --   low=0/4
            0.20 | LADDER_TOO_HIGH    lam*= 0.502 err=    --   low=0/2 | LADDER_TOO_HIGH    lam*= 0.160 err=    --   low=0/4
            0.25 | LADDER_TOO_HIGH    lam*= 0.627 err=    --   low=0/2 | LADDER_TOO_HIGH    lam*= 0.200 err=    --   low=0/4
            0.35 | KNEE_BRACKETED     lam*= 0.878 err=  +0.0% low=1/2 | LADDER_TOO_HIGH    lam*= 0.280 err=    --   low=0/4
            0.50 | KNEE_BRACKETED     lam*= 1.255 err=  +0.0% low=1/2 | KNEE_NOT_BRACKETED lam*= 0.400 err=    --   low=0/4
            0.70 | KNEE_BRACKETED     lam*= 1.757 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.560 err=  +0.0% low=1/4
            0.85 | KNEE_BRACKETED     lam*= 2.133 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.680 err=  +0.0% low=2/4
            1.00 | KNEE_BRACKETED     lam*= 2.510 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.800 err=  +0.0% low=2/4
            1.20 | KNEE_BRACKETED     lam*= 3.012 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.960 err=  +0.0% low=2/4
            1.45 | KNEE_BRACKETED     lam*= 3.639 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 1.160 err=  -7.0% low=3/4
            1.80 | KNEE_BRACKETED     lam*= 4.518 err=  -1.6% low=2/2 | LADDER_TOO_LOW     lam*= 1.440 err=    --   low=3/4
            2.50 | LADDER_TOO_LOW     lam*= 6.275 err=    --   low=2/2 | LADDER_TOO_LOW     lam*= 2.000 err=    --   low=4/4

### absolute lower bound  (A=1.08 worst observed step, B=0.35)   lambda_inf=(1.08, 0.35)
    lam*/lam_inf |            shape A            |            shape B
            0.10 | LADDER_TOO_HIGH    lam*= 0.108 err=    --   low=0/2 | LADDER_TOO_HIGH    lam*= 0.035 err=    --   low=0/4
            0.20 | LADDER_TOO_HIGH    lam*= 0.216 err=    --   low=0/2 | LADDER_TOO_HIGH    lam*= 0.070 err=    --   low=0/4
            0.25 | LADDER_TOO_HIGH    lam*= 0.270 err=    --   low=0/2 | LADDER_TOO_HIGH    lam*= 0.087 err=    --   low=0/4
            0.35 | KNEE_BRACKETED     lam*= 0.378 err=  +0.0% low=1/2 | LADDER_TOO_HIGH    lam*= 0.122 err=    --   low=0/4
            0.50 | KNEE_BRACKETED     lam*= 0.540 err=  +0.0% low=1/2 | KNEE_NOT_BRACKETED lam*= 0.175 err=    --   low=0/4
            0.70 | KNEE_BRACKETED     lam*= 0.756 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.245 err=  +0.0% low=1/4
            0.85 | KNEE_BRACKETED     lam*= 0.918 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.297 err=  +0.0% low=2/4
            1.00 | KNEE_BRACKETED     lam*= 1.080 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.350 err=  -6.1% low=2/4
            1.20 | KNEE_BRACKETED     lam*= 1.296 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.420 err= -15.4% low=2/4
            1.45 | KNEE_BRACKETED     lam*= 1.566 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.507 err= -24.7% low=3/4
            1.80 | KNEE_BRACKETED     lam*= 1.944 err=  -3.1% low=2/2 | LADDER_TOO_LOW     lam*= 0.630 err=    --   low=3/4
            2.50 | LADDER_TOO_LOW     lam*= 2.700 err=    --   low=2/2 | LADDER_TOO_LOW     lam*= 0.875 err=    --   low=4/4

### absolute upper bound  (A=7.15 step(B=1) floor, B=1.20)   lambda_inf=(7.15, 1.2)
    lam*/lam_inf |            shape A            |            shape B
            0.10 | LADDER_TOO_HIGH    lam*= 0.715 err=    --   low=0/1 | LADDER_TOO_HIGH    lam*= 0.120 err=    --   low=0/4
            0.20 | LADDER_TOO_HIGH    lam*= 1.430 err=    --   low=0/1 | LADDER_TOO_HIGH    lam*= 0.240 err=    --   low=0/4
            0.25 | KNEE_BRACKETED     lam*= 1.788 err=  +0.0% low=1/1 | LADDER_TOO_HIGH    lam*= 0.300 err=    --   low=0/4
            0.35 | KNEE_BRACKETED     lam*= 2.502 err=  +0.0% low=1/1 | LADDER_TOO_HIGH    lam*= 0.420 err=    --   low=0/4
            0.50 | KNEE_BRACKETED     lam*= 3.575 err=  +0.0% low=1/1 | KNEE_NOT_BRACKETED lam*= 0.600 err=    --   low=0/4
            0.70 | KNEE_BRACKETED     lam*= 5.005 err=  +0.0% low=1/1 | KNEE_BRACKETED     lam*= 0.840 err=  +0.0% low=1/4
            0.85 | KNEE_BRACKETED     lam*= 6.078 err=  +0.0% low=1/1 | KNEE_BRACKETED     lam*= 1.020 err=  +0.0% low=2/4
            1.00 | KNEE_BRACKETED     lam*= 7.150 err=  +0.0% low=1/1 | KNEE_BRACKETED     lam*= 1.200 err=  +0.0% low=2/4
            1.20 | KNEE_BRACKETED     lam*= 8.580 err=  +0.0% low=1/1 | KNEE_BRACKETED     lam*= 1.440 err=  +0.0% low=2/4
            1.45 | KNEE_BRACKETED     lam*=10.367 err=  +0.0% low=1/1 | KNEE_BRACKETED     lam*= 1.740 err=  -0.9% low=3/4
            1.80 | KNEE_BRACKETED     lam*=12.870 err=  -2.5% low=1/1 | LADDER_TOO_LOW     lam*= 2.160 err=    --   low=3/4
            2.50 | LADDER_TOO_LOW     lam*=17.875 err=    --   low=1/1 | LADDER_TOO_LOW     lam*= 3.000 err=    --   low=4/4

### B=6 plateau variant   (A=3.91, B=1.00)   lambda_inf=(3.91, 1.0)
    lam*/lam_inf |            shape A            |            shape B
            0.10 | LADDER_TOO_HIGH    lam*= 0.391 err=    --   low=0/2 | LADDER_TOO_HIGH    lam*= 0.100 err=    --   low=0/4
            0.20 | LADDER_TOO_HIGH    lam*= 0.782 err=    --   low=0/2 | LADDER_TOO_HIGH    lam*= 0.200 err=    --   low=0/4
            0.25 | KNEE_NOT_BRACKETED lam*= 0.978 err=    --   low=0/2 | LADDER_TOO_HIGH    lam*= 0.250 err=    --   low=0/4
            0.35 | KNEE_BRACKETED     lam*= 1.369 err=  +0.0% low=1/2 | LADDER_TOO_HIGH    lam*= 0.350 err=    --   low=0/4
            0.50 | KNEE_BRACKETED     lam*= 1.955 err=  +0.0% low=2/2 | KNEE_NOT_BRACKETED lam*= 0.500 err=    --   low=0/4
            0.70 | KNEE_BRACKETED     lam*= 2.737 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.700 err=  +0.0% low=1/4
            0.85 | KNEE_BRACKETED     lam*= 3.324 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.850 err=  +0.0% low=2/4
            1.00 | KNEE_BRACKETED     lam*= 3.910 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 1.000 err=  +0.0% low=2/4
            1.20 | KNEE_BRACKETED     lam*= 4.692 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 1.200 err=  +0.0% low=2/4
            1.45 | KNEE_BRACKETED     lam*= 5.670 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 1.450 err=  -3.5% low=3/4
            1.80 | KNEE_BRACKETED     lam*= 7.038 err=  -1.0% low=2/2 | LADDER_TOO_LOW     lam*= 1.800 err=    --   low=3/4
            2.50 | LADDER_TOO_LOW     lam*= 9.775 err=    --   low=2/2 | LADDER_TOO_LOW     lam*= 2.500 err=    --   low=4/4

### FALLBACK literal ladders   lambda_inf=(2.1, 0.675)  [FALLBACK ladders]
    lam*/lam_inf |            shape A            |            shape B
            0.10 | LADDER_TOO_HIGH    lam*= 0.210 err=    --   low=0/2 | LADDER_TOO_HIGH    lam*= 0.068 err=    --   low=0/4
            0.20 | LADDER_TOO_HIGH    lam*= 0.420 err=    --   low=0/2 | LADDER_TOO_HIGH    lam*= 0.135 err=    --   low=0/4
            0.25 | LADDER_TOO_HIGH    lam*= 0.525 err=    --   low=0/2 | LADDER_TOO_HIGH    lam*= 0.169 err=    --   low=0/4
            0.35 | LADDER_TOO_HIGH    lam*= 0.735 err=    --   low=0/2 | LADDER_TOO_HIGH    lam*= 0.236 err=    --   low=0/4
            0.50 | KNEE_NOT_BRACKETED lam*= 1.050 err=    --   low=0/2 | LADDER_TOO_HIGH    lam*= 0.338 err=    --   low=0/4
            0.70 | KNEE_BRACKETED     lam*= 1.470 err=  +0.0% low=1/2 | KNEE_NOT_BRACKETED lam*= 0.472 err=    --   low=0/4
            0.85 | KNEE_BRACKETED     lam*= 1.785 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.574 err=  +0.0% low=1/4
            1.00 | KNEE_BRACKETED     lam*= 2.100 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.675 err=  -0.0% low=1/4
            1.20 | KNEE_BRACKETED     lam*= 2.520 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.810 err=  +0.0% low=2/4
            1.45 | KNEE_BRACKETED     lam*= 3.045 err=  +0.0% low=2/2 | KNEE_BRACKETED     lam*= 0.979 err=  -9.2% low=2/4
            1.80 | KNEE_BRACKETED     lam*= 3.780 err=  +0.0% low=2/2 | LADDER_TOO_LOW     lam*= 1.215 err=    --   low=3/4
            2.50 | KNEE_BRACKETED     lam*= 5.250 err=  +0.0% low=2/2 | LADDER_TOO_LOW     lam*= 1.688 err=    --   low=4/4

### verdict domain census (must be non-empty for each registered verdict)
    shape A  KNEE_BRACKETED        56 grid points  e.g. ('MEASURED job 907959 (I3a 3.0939 / I3b 0.6956 req/s)', 0.35)
    shape A  KNEE_NOT_BRACKETED     3 grid points  e.g. ('MEASURED job 907959 (I3a 3.0939 / I3b 0.6956 req/s)', 0.25)
    shape A  LADDER_TOO_HIGH       19 grid points  e.g. ('MEASURED job 907959 (I3a 3.0939 / I3b 0.6956 req/s)', 0.1)
    shape A  LADDER_TOO_LOW         6 grid points  e.g. ('MEASURED job 907959 (I3a 3.0939 / I3b 0.6956 req/s)', 2.5)
    shape B  KNEE_BRACKETED        34 grid points  e.g. ('MEASURED job 907959 (I3a 3.0939 / I3b 0.6956 req/s)', 0.7)
    shape B  KNEE_NOT_BRACKETED     7 grid points  e.g. ('MEASURED job 907959 (I3a 3.0939 / I3b 0.6956 req/s)', 0.5)
    shape B  LADDER_TOO_HIGH       29 grid points  e.g. ('MEASURED job 907959 (I3a 3.0939 / I3b 0.6956 req/s)', 0.1)
    shape B  LADDER_TOO_LOW        14 grid points  e.g. ('MEASURED job 907959 (I3a 3.0939 / I3b 0.6956 req/s)', 1.8)
```

★**등록 assert**: 4개 verdict(`KNEE_BRACKETED`·`KNEE_NOT_BRACKETED`·`LADDER_TOO_HIGH`·
`LADDER_TOO_LOW`) 각각이 **두 shape 모두에서 비어있지 않은 정의역**을 가진다
(`lambda0_reachability.py --selftest`, sbatch가 첫 boot 전에 실행).
★**음성대조**: 같은 지도를 **rev3 규칙**으로 돌리면 shape B가 측정 앵커에서 그대로
`UNRESOLVED`를 낸다 — E1이 실제로 무언가를 바꾼다는 증거(교훈 53).

### 5.5 검증 (D9/D22/E3) — 변이 **47종 × 6모듈**, CONTROL **5종**, escapes 0

rev1 감사: 11종 중 7 escape. rev2: 17종 전부 차단했으나 **한 파일만** 변이시켜 감사자 6종
중 5 escape. rev3: 36종·3모듈, 그러나 `lambda0_cells.py`가 밖에 있어 감사자 8종 중 5 escape
(그중 **Y5는 rev3 死因을 shape A로 복제하면서 36/36을 통과**).

rev4: **MODULES = 결정 경로 6파일**(`label`·`analyze`·`plan`·`cells`·`lambda_inf`·
`reachability`), 변이 **47종**(M1–M17 · X1–X6 · R3a–R3m · **Y1–Y6** · **R4a–R4e**),
CONTROL 5종을 **변이가 도는 temp-dir 경로에서 채점**. **escapes: none (47/47).**
★**CONTROL 비공허성 주입 확인**: 사본에서 `lambda0_analyze.py`를 빼면 CONTROL이 `FAILS`
하고 하네스가 중단한다(rc 2) — rev3의 죽은 CONTROL이었다면 47종 전부가 "FAILS (good)"을
찍었을 자리다.

---

## 6. 예보와 출력공간 라벨 (F1–F4)

- **F1 (부팅)**: 전 셀 boot 성공. 근거 = job 905835 12/12 + job 907959 5/5(legacy warm-up).
- **F2 (shape B)**: `KNEE_BRACKETED`. ★rev3에서는 이 분지의 정의역이 **∅**이었다. rev4에서는
  측정 앵커 λ_inf(B)=0.6956에서 지도가 `KNEE_BRACKETED`(오차 −0.0%)를 내고, 그 사실이
  실행 전 assert된다.
- **F3 (shape A)**: `KNEE_BRACKETED`. 커버리지 밴드 `λ*/λ_inf ∈ [0.24, 1.79]`(§5.4 인쇄).
- **F4 = R3**: 각 shape의 최상단 rung을 제2 seed로 1회 반복 → 포화 유지 + λ\* 5% 이내 재현.
  반증되면 λ\*를 두 seed의 구간으로 보고하고 5% 주장을 `REFUTED`로 등재한다.

---

## 7. 이 단계가 **주지 않는 것** · D13 · E5 예산

- **성능·정책 판정 0건**. Claim D/E 등급 불변 · HE0 · 정책 순위 · stake #1 불변.
- **새 모델의 true-dual correctness를 건드리지 않는다**(§9 Q2, §4.3).
- **단일 스칼라 λ\* 설계 문제는 이 단계가 해소하지 않는다**(§9 Q3).

**D13** — 상단 셀 TTFT p99는 probe C 최댓값 101.43 s 위의 무검증 영역이다. 사전 확인:
`bench_serving.py:70` 타임아웃 6시간 · `scheduler.py:2013-2014` `max_queued_requests is
None` ⇒ 503 없음. ★D17의 긴 창은 **고측 rung을 건드리지 않으므로** 이 영역을 넓히지 않는다.

**★E5 — 예산 정정(rev3의 이중 오류 2건 수리)**
1. rev3 §3.3·§7의 "0.96–1.47 GPU-h"는 **자기 §3.5 인쇄(0.959–1.236)와 불일치**했다.
   **인쇄가 정본**이며 아래 표로 대체한다.
2. rev3은 포화 rung을 `N/x`로 가격했다. **용량을 넘은 rung은 `N/x`가 아니라 `N/λ*`에
   끝난다** ⇒ 등록 커버리지 하한(λ\* = 0.25·λ_inf)에서 비용이 크게 오른다.

| 시나리오 | λ\*=λ_inf | 0.50× | **0.25×(등록 하한)** |
|---|---|---|---|
| **MEASURED job 907959** | **1.241** | 1.611 | **2.592** |
| point lower / upper | 1.309 / 1.278 | 1.732 / 1.679 | 2.830 / 2.722 |
| absolute lower / upper | 1.412 / 1.049 | 1.927 / 1.341 | 3.224 / **2.029** |
| B=6 plateau | 1.249 | 1.593 | 2.549 |
| FALLBACK | 1.348 | 1.977 | **3.509** |

> **요청 예산 = 3.60 GPU-h**(등록 최악 3.509를 덮는다). **실제로 돌 시나리오(측정값)의
> 예상은 1.24 GPU-h**이고 0.25× 코너에서도 2.59다. `--time=04:30:00` 유지(최악 3.51 h < 4.5 h).
> 등록된 재설계 1회는 **두 번째 제출**이다.

이 트랙 GPU 장부: R2 correctness 0.43 + job 907959 0.356 GPU-h — **0단계 자신은 아직 0**.

---

## 8. 동봉 실행체와 재현 절차

| 파일 | 역할 |
|---|---|
| `lambda0_plan.py` | 사다리·측정창·N·seed·κ 표·창·**예산 표(E5)**의 결정적 생성기 |
| `lambda0_reachability.py` | ★**E2** 참 λ\* 격자 → 방출 verdict 지도 + 4 verdict 정의역 assert + rev3 음성대조 |
| `lambda0_lambda_inf.py` | **D18** 앵커/fallback 객관 술어 + **E6** 절대경로·sha256 |
| `lambda0_cells.py` | plan.json → sbatch 셀 스펙 렌더러(`-` placeholder, 인증 열 = 계획) |
| `lambda0_analyze.py` | per-cell analyzer(실현 도착률·span/drain·κ_pred·가드·길이 검증) |
| `lambda0_cellprint.py` | 셀 1줄 요약 |
| `lambda0_label.py` | 판정 규칙 R0–R3 + end-to-end leg(**E4**: shape A·B 양 구성) |
| `lambda0_mutation_check.py` | 변이 47종 × 6모듈 + CONTROL 5종(temp-dir 경로에서 채점) |
| `lambda0.sbatch` | 제출 실행체(ONE BOOT PER CELL · 절대경로 · `--comment` · D12 · D18 · D23 · 전 selftest + 변이 게이트) |

```bash
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc
P=workspace/engine-port/results/r2_eval/lambda0_prereg
for m in plan label cells lambda_inf reachability; do python3 $P/lambda0_$m.py --selftest; done
python3 $P/lambda0_mutation_check.py        # 0 escapes required
python3 $P/lambda0_plan.py --registered-scenarios
python3 $P/lambda0_reachability.py

sbatch $P/lambda0.sbatch      # 앵커/fallback은 파일이 정한다
```

---

## 9. ★인용 금지 — Q1–Q5 · L1–L3 · **R3C-1…4** (전부 문자 승계)

> **Q1** — **이 캠페인 0단계는 어떤 결과가 나오더라도 방법론 게이트 #6("용량 먼저 측정")을
> 닫지 못한다.** 게이트 #6이 요구하는 용량은 **지표 절벽 대비** 용량이고, 이 단계가 재는
> 것은 throughput 포화다. 905835 원자료를 정본 술어(TTFT≤3000 ms ∧ 요청내부 token-ITL
> p95≤60 ms)로 재채점하면 shape B(8192-in)의 SLO 절벽은 **0.59·λ\*_throughput 아래**에
> 있다(0.59×에서 goodput **53.8%**, 0.89×에서 **5.8%**, 1.27×에서 **0.8%**). 따라서
> "λ\*를 측정했으므로 W3/W4의 부하 라벨이 참이 되었다"는 문장은 **쓸 수 없다**.

> **Q2** — **λ\*(B1)의 측정은 (Nano-9B-v2, flashinfer) 쌍의 R2 correctness를 어느 정도도
> 확인하지 않는다.** 이 단계와 job 905835는 **legacy 루프 + fixed split**만 발화시켰다.
> `PDMUX_TRUE_DUAL_WORKER` 경로는 이 모델에서 **한 번도 실행된 적이 없다**.
> 907100·907456·X1의 결론은 **(Zamba2-2.7B, triton) 한정으로 동결**이고, X1의 민감도
> 시연은 `triton_attention_num_kv_splits`가 flashinfer에 없으므로 이식되지 않는다.

> **Q3** — **이 단계가 내놓는 두 λ\*는 W4를 파라미터화하지 못한다 — 상호 배타적이다.**
> `generate_campaign.sh:18`은 9 워크로드에 단일 스칼라를 쓰고 `workloads.py:150`의 W4는 두
> phase에 **같은** `per_phase`를 쓴다. 실측/추정 λ\*로 계산하면: λ\*=0.675 주입 →
> prefill phase 0.79×λ\*(B) ✓ / decode phase **0.21–0.25×λ\*(A)** ✗; λ\*=2.1 주입 →
> decode 0.79×λ\*(A) ✓ / prefill **2.47×λ\*(B)** ✗; 기본값 4 → prefill **4.7×λ\*(B)**.
> **어떤 단일 스칼라도 두 phase를 동시에 0.80으로 만들 수 없다.** W4를 Claim D의 P2에서
> 쓰려면 워크로드 정의 자체(phase별 독립 λ\*)를 고쳐야 하며, 그것은 이 단계가 주지 않는다.

★**Q3 편집 주석(인용문 불변)**: Q3의 `generate_campaign.sh:18`은 rev1 판정서 표기이며
HEAD 기준 정확한 줄은 **`:17`**이다. 병행 워크스트림이 워크로드 정의를 phase별 독립 λ\*로
고쳤으므로 Q3의 마지막 문장은 여전히 참이다 — 이 단계는 **두 수**를 줄 뿐이다.
**이 단계가 그 코드 변경을 자기 공로로 인용하는 것을 금지한다.**

> **Q4** — **"B1의 용량이 시스템의 용량"이 아니다.** 워크로드 이름(W8 "near saturation",
> W9 "overload")은 **B1에서만** 참인 서술이고, B4에서 같은 trace는 자기 용량의 0.2×일
> 수도 5×일 수도 있다. 워크로드 이름을 regime 서술로 인용하는 것을 금지한다.

> **Q5** — **shape A·B 두 shape를 쟀다는 것이 W1·W5·W6·W7·W8·W9의 부하 라벨을 고치지
> 않는다.** 그 워크로드들의 shape는 미측정으로 남으며, λ\*는 shape 의존적이다
> (이 단계 자신이 두 shape에서 다른 값을 낸다).

> **L1** — **`achieved/realized_offered`는 포화도가 아니다. 그것은 `(N/(N−1))·span/(span+L)`
> 이며, 비포화 셀에서의 값은 1이 아니라 배수(drain) 시간이 정한다.** 보관 12셀에서
> `duration = maxᵢ(arrivalᵢ + e2eᵢ)`가 ≤0.27 s로 재현되므로 이는 항등식이다. **이
> 추정량으로 얻은 "비포화" 라벨을 out ≳ 256 토큰 shape에 인용하는 것을 금지한다** —
> (256,512)에서 저측 문턱 0.95는 "decode ITL < 20 ms"와 같은 뜻이고, 그것은 용량이 아니라
> SM 분할의 성질이다.

> **L2** — **rev2가 보인 "청정 비포화 셀 3/3이 1.00±0.02로 수렴"은 `(8192 in, 96 out)`
> 한정 사실이다.** 같은 보정을 (256,512)에 적용하면 기준선이 0.93–0.97로 내려간다.
> **"실현-분모 보정이 인공물을 제거했다"를 shape를 명시하지 않고 인용하는 것을 금지한다.**

> **L3** — **`d44_r1_o96`이 "포화 셀"인 것은 배수 인공물이다**(측정 0.8988 vs 이상값
> 0.9295). rev2 §5.2(i)의 "실현 분모에서 d44가 실제로 포화 셀 2개"를 **M1 차단의 데이터
> 근거로 인용하는 것을 금지한다**(합성 쌍둥이 케이스 **3b**만 인용 가능).

> **R3C-1** — **rev3(그리고 그 이전 어떤 판본)의 규칙으로는 λ\*(shape B = 8192-in)를 측정할
> 수 없다.** 계획이 B의 네 rung을 모두 저측 후보로 인증하고 R0이 인증 셀에 `drain ≤ 2·L̂`를
> 걸기 때문에, R1의 고측(`ratio ≤ 0.90` ⇔ `drain ≥ 19.2–34.5 s`)을 만족하는 셀은 **반드시**
> R0(8.34–8.50 s)을 위반한다. **`KNEE_BRACKETED[B]`는 λ_inf(B) ≤ 1.70 req/s인 모든 경우에
> 정의역이 ∅이다** — 등록 격자(0.35–1.20)·FALLBACK(0.675)·보관 앵커(0.675) 전부가 여기
> 들어간다. **"0단계가 W4 prefill phase의 부하 라벨을 정했다"는 문장은 쓸 수 없다.**

> **R3C-2** — **`LADDER_TOO_HIGH`(÷4 재설계)는 rev3에서 한 번도 발화할 수 없다**(A·B 공통).
> 따라서 **"rev3는 'λ_inf가 λ\*를 아래에서 묶는다'는 가정을 데이터 검사 분지로 대체했다"
> (§3.6-3)는 문장을 인용하는 것을 금지한다** — 그 분지는 절반이 죽어 있다.

> **R3C-3** — **rev3의 브래킷 양변은 측정창이 서로 다르다**(저측 600–700 s, 고측 170 s).
> shape A 고측의 "≤0.90"은 실효적으로 "요청 e2e ≥ 18.9 s ⇔ decode ITL ≥ 37 ms ⇔ 평균 배치
> ≳ 34(= max_running 48의 0.72배)"이며, **같은 셀을 600 s 창에서 재면 0.968을 읽는다**.
> 따라서 **보고되는 λ\*(A)를 "용량"이라고 인용할 때는 (창 길이, N) 튜플을 함께 적어야
> 하고**, 사다리 상단 바로 위 구간에서는 과소평가가 발생한다(자기정합 모형: 참 λ\*(A)=8.0
> → 보고 7.155, **−10.6%**, 라벨은 `KNEE_BRACKETED`).

> **R3C-4** — **"변이 36/36 차단, escape 0"은 `lambda0_label.py`·`lambda0_analyze.py`·
> `lambda0_plan.py` 세 파일 한정 사실이다.** 결정 경로에는 `lambda0_cells.py`가 포함되는데
> 하네스가 변이시키지 않고 sbatch가 그 selftest를 돌리지도 않는다. 감사자 독립 변이 8종 중
> **5종이 escape**했으며, 그중 **Y5는 이 판정서의 死因을 shape A까지 복제하면서 36/36을
> 그대로 통과한다**. **"결정 경로 전체가 변이 인증됐다"고 인용하는 것을 금지한다.**

★**R3C-1…4의 rev4에서의 지위**: R3C-1·R3C-2는 **rev3 이하 판본에 대한 사실**이며 rev4가
E1/E2로 해소했다(§3, §5.4 지도). R3C-3은 **rev4에도 유효한 caveat**이다 — 측정창
이질성은 남아 있고, §5.4 지도가 그 과소평가를 수치로 등재한다(사다리 상단 바로 위 구간에서
−7 … −25%). R3C-4는 **rev4에서 해소**됐다(6모듈·47변이·CONTROL 5, §5.5) — 단
"결정 경로 전체가 변이 인증됐다"는 문장은 **그 6파일 한정**으로만 쓴다.

---

## 10. ★실행 후 필수 병기 (rev1 판정서 §5 + **NPC-I**)

1. **λ\*는 B1(legacy, fixed D44) 한정 throughput 포화율이다.** split을 바꾸면 5× 변한다.
2. **정본 술어 goodput은 이 단계에서 측정되지 않았다.**
3. **상단 셀의 TTFT/ITL 백분위는 인용 불가다**(의도적 과포화 셀).
4. `switch_count`·split 체류분포는 판정에 쓰이지 않았다(fixed split).
5. **포화 셀의 achieved는 도착 실현에 무관하나, 비포화 셀의 `ach/off`는 도착 실현
   계수다**(905835 1.03–1.19, 12/12 동일 방향). ★그리고 실현 분모로 바꾼 뒤에도 비포화
   셀의 영점은 1이 아니라 **κ**다(L1).
6. 이 단계의 부하 생성기는 `sglang.bench_serving`(Poisson)이고 캠페인은
   `pdmux_eval.trace_loadgen`(고정 도착시각, `input_ids=[1]*n`)이다. **λ\*는 다른
   하네스에서 측정된 상수로 캠페인을 파라미터화한다.**
7. ★**NPC-I** — **λ_inf는 legacy warm-up boot · D44 · cudagraph ON에서만 측정되며,
   B4(true-dual)의 포화 상한은 측정되지 않았다.**

---

## 11. 인용 고정 · 하류 인터페이스

### 11.1 인용 고정표 (교훈 80 · 게이트 #110)

캠페인 쪽 인용은 **커밋 `bddff6a`에 고정**한다(rev3 감사 시점 HEAD는 `b2b60e7`로 이동했고,
인용 파일 중 바뀐 것은 `sync_engine_tree.sh` 하나[24→25항목, append-only]뿐이며 D12 게이트가
`≥24`라 **여전히 통과**함을 감사가 확인했다).

| 인용 | 확인된 내용 | 고정 트리 | worktree도 유효 |
|---|---|---|---|
| `benchmarks/pdmux_eval/workloads.py:159-160` | `8192 if is_prefill else 256` / `64 if is_prefill else 512` | HEAD `bddff6a` | **예** |
| `benchmarks/pdmux_eval/workloads.py:150` · `campaign.py:139-140` · `scripts/r2_eval/generate_campaign.sh:17` | W4 per_phase · SLO 상수 · 미측정 스칼라 기본값 | HEAD `bddff6a` | 아니오(병행 워크스트림이 교체) |
| `scripts/r2_eval/engine_bench_runner.sh:99` · `r2_eval.sbatch:38` | piecewise 분기 · 절대 project_root | HEAD `bddff6a` | 예 |
| `c_capacity.sbatch:76`/`:111`/`:123-178`/`:126`/`:139-145` | ρ 사다리 · 셀 루프 · 셀별 launch·kill · piecewise · warmup | HEAD | 예(보관 아티팩트) |
| `sglang/bench_serving.py:70` · `:948` · `:1354` · `:1705-1706` · `benchmark/datasets/common.py:56-64` · `srt/managers/scheduler.py:2013-2014` · `srt/server_args.py:1959` | 타임아웃 · 간격 · LoRA · seed · randint 소비 · 무제한 큐 · triton assert | dev tree | — |

★**판정서 인용의 한 줄 어긋남 2건 정정 유지**: `generate_campaign.sh:18` → **`:17`**,
probe C ρ 사다리는 `:78-79`가 아니라 **`:76`**.

★**도구 한계**: `check_line_citations.py`의 `SEARCH_ROOTS`는 `benchmarks/`와 `srt` 밖
`sglang/bench_serving.py`를 해석하지 못한다. 이 표가 그 구멍을 수작업으로 메운다.

### 11.2 하류 인터페이스

병행 워크스트림의 **shape별 λ\* 표**(`benchmarks/pdmux_eval/lambda_star.py`, schema
`pdmux.lambda_star/v1`, 키 `"256x512"`/`"8192x64"`)가 미측정이면 fail-closed다. 이 단계의
산출 `LAMBDA0_LABEL.json`은 **`definition = throughput_saturation`** 쪽에만 들어간다
(`source`를 `unmeasured` → `measured`, `evidence`에 job id + 이 사전등록 경로).
★`slo_sustainable`은 이 단계 이후에도 **비어 있다**(§9 Q1).
**번역기 자체는 통합 시점에 코디네이터가 배선한다** — rev4는 그 표로 번역될 것을 전제할
뿐 번역을 수행하지 않는다.

---

## 12. 반증 실패 항목 (공정 기록)

- **배수 항등식·duration 재구성**: 보관 12/12에서 재현(감사 독립 확인 2.2e−16 / +0.072–0.265 s).
- **`ITL_SOLO_S = 13.06 ms`**: 3중 확인(등록 상수 · 보관 셀 13.07–13.48 ms · I2 12.96 ms).
- **R1/R5 실질 승계**: 보관 cell JSON 직접 투입으로 λ\* 0.9331188685063582 /
  0.675302644015995 / 0.18668462822130108 재현.
- **rr=1.0 필요조건**(D20) · **D21** · **D23** · **D4 음성대조** · **ONE BOOT PER CELL 기제
  승계** · **`CONSENSUS §3 항목120` 회피**: 전부 감사가 코드·데이터로 재확인.
- **D17 음성대조**: rev2의 170 s 저측 rung이 전부 거부됨을 selftest가 재현.
- **κ assert의 항등식 탈출 수리**: 리터럴 `ACH_HI + 0.02` assert, 감사 변이 Y3로 독립 확인.
- **arm 비교 금지·성능 판정 0건**: 유지 — 이 단계는 어떤 결과가 나와도 HE0·정책 순위·
  Claim D/E 등급·stake #1을 움직이지 않는다.
