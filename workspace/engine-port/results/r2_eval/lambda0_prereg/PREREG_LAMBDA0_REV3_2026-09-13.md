> ## ★ SUPERSEDED (2026-09-13) — 인용 금지, 삭제 금지
>
> 이 rev3는 claims-auditor 3차 감사에서 **`NO-GO`**를 받았다
> (`VERDICT_lambda0_rev3_2026-09-13.md`). D16–D23 중 6건은 닫힌 것으로 확인됐으나,
> 死因은 **등록 예보 F2("shape B = `KNEE_BRACKETED`")의 참 분지가 정의역 ∅**이라는 것이다:
> 계획이 shape B의 4/4 rung을 저측 후보로 인증하고 R0의 배수 가드(`drain ≤ 2·L̂` =
> 8.34–8.50 s)가 인증 셀 전부에 걸리는데, R1의 **고측**은 `drain ≥ 19.2–34.5 s`를
> 요구한다 ⇒ **브래킷이 요구하는 증거가 측정 실패로 폐기된다**. 부수로
> `LADDER_TOO_HIGH`(÷4)도 A·B 양쪽에서 발화 불가. **미실행**(GPU 0).
> ★판정서 §3이 **이 死因의 근인이 감사자 자신의 rev2 처방(D16/D17)**임을 자기 철회로 명시한다.
> **현행 정본은 `PREREG_LAMBDA0_REV4_2026-09-13.md`.**
>
> ★인용 금지(판정서 §7): **R3C-1**(이 판본 규칙으로는 λ\*(shape B)를 측정할 수 없다) ·
> **R3C-2**(`LADDER_TOO_HIGH`는 한 번도 발화 불가 ⇒ §3.6-3의 "데이터 검사 분지로 대체했다"
> 인용 금지) · **R3C-3**(브래킷 양변의 측정창이 다르다) · **R3C-4**("변이 36/36 차단"은
> 3파일 한정 사실 — `lambda0_cells.py`는 하네스 밖이었다).
> ★**rev4가 독립 재생성으로 추가 발견**: `LADDER_TOO_LOW[shape A]`도 이 판본에서 정의역 ∅
> 이며(최상단 rung의 천장 κ ≈ 0.90–0.94 < 0.95), 판정서 §3-2의 자기 철회("λ\*(A)=4.0은
> `LADDER_TOO_LOW`를 낸다")도 **재현되지 않는다**(쉬핑 코드는 `KNEE_BRACKETED`, −6.4%).

# 사전등록 rev3 — 캠페인 **0단계**: λ\*(용량) 측정, shape별 × 기준 arm B1

**작성** 2026-09-13 (engine-porter) · **상태: 미실행, GPU 지출 0** · **제출 전**

**이력**: rev1 `NO-GO`(`VERDICT_lambda0_rules_2026-09-13.md`, D1–D15) → rev2 `NO-GO`
(`VERDICT_lambda0_rev2_2026-09-13.md`, D16–D23) → **이 rev3**. 두 판본 모두 SUPERSEDED
배너와 함께 보존한다(삭제 금지 — 사전등록 이력의 감사 가능성).

**rev2 재감사의 요지**: D1–D15 중 **13건은 코드·데이터로 닫힌 것이 확인**됐고 측정 격자·
사다리 결정성·변이 하네스·sbatch·provenance 게이트·인용 규율은 건강하다. 死因은
**rev2가 D2를 고치면서 새로 도입한 추정량**에 있었다(반전 2행). rev3는 그 두 곳만
바꾸고 나머지는 유지한다.

사용자 결정 3건이 선행했다: 모델 = **`nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base`**,
백엔드 = **flashinfer**(NemotronH에서 triton은 엔진이 거부, `server_args.py:1959`),
1차 범위 = **Claim D + P3**.

---

## 0. 왜 0단계인가 — 그리고 ★이 단계가 닫지 못하는 것

캠페인 워크로드는 전부 λ\*의 분수로 정의된다(W1 0.60 · W3 0.80 · W8 0.90 · W9 1.10 ·
W2/W4는 phase당 요청 수). 그런데 `scripts/r2_eval/generate_campaign.sh:17`(HEAD
`bddff6a`)의 `sustainable_rate`는 **측정값이 아니라 기본값 `4`**다.

★★**그러나 이 단계는 어떤 결과가 나와도 방법론 게이트 #6("용량 먼저 측정")을 닫지
못한다.** 게이트 #6이 요구하는 용량은 **지표 절벽 대비** 용량이고, 이 단계가 재는 것은
throughput 포화다. rev1 판정서가 job 905835 d44 원자료를 정본 goodput 술어(TTFT ≤ 3000 ms
∧ 요청-내부 token-ITL p95 ≤ 60 ms, `benchmarks/pdmux_eval/campaign.py:139-140` @HEAD
`bddff6a`)로 직접 재채점한 결과: **0.59·λ\*_thr에서 goodput 53.8% · 0.89·λ\*_thr에서
5.8% · 1.27·λ\*_thr에서 0.8%** ⇒ **λ\*_SLO(B) < 0.59 × λ\*_throughput**(비 추정 2.3–3.4×).
SLO 절벽은 이 단계가 재는 수보다 **아래**에 있다. §9 Q1으로 문자 고정한다.

> ★**인용 시점 고정(게이트 #110)**: 캠페인 쪽 인용은 **커밋 `bddff6a`(HEAD) 기준**이다.
> 같은 날 병행 워크스트림이 작업트리에서 캠페인 생성기·워크로드 정의를 고쳤다(미커밋):
> 스칼라 기본값은 제거되고 **shape별 λ\* 표**(`benchmarks/pdmux_eval/lambda_star.py`,
> schema `pdmux.lambda_star/v1`, 키 `"256x512"`/`"8192x64"`)가 들어갔으며 미측정이면
> fail-closed다. 전수 대조는 §11.1.

---

## 1. λ\*의 정의 · ★정본 정의 교체의 등재 (D14, 유지)

> **λ\*(shape) = 기준 arm B1(legacy 루프, PD-mux fixed D44)에서 그 shape의
> open-loop(Poisson) throughput 포화율.** 워크로드의 분수는 "**B1 용량의 n%**"를 뜻한다.

**D14 — 정본 정의 교체를 정정으로 등재한다.** `reports/paper/EXPERIMENT_ROADMAP.md:669`
(감사 시점 `:652`)는 λ\*를 "sustainable **SLO** rate"로 정의한다. **이 등록은 그것을
의도적으로 throughput 포화로 교체한다.** 실측 비는 shape B에서 **≥1.7×**(위 재채점, 하한).
**λ\*_SLO는 이 단계 이후에도 여전히 측정되지 않는다.**

**왜 arm을 고정하는가**: λ\*는 arm마다 다르다 — 같은 모델·같은 shape에서 **D16 0.933 /
D44 0.675 / D92 0.187 req/s(5×)**(job 905835). ★그러나 **워크로드 이름은 arm을 넘어
거짓이 된다**(§9 Q4).

---

## 2. 측정 격자 · 통제 요인 (rev2에서 불변 — 판정서가 닫힌 것으로 확인한 부분)

### 2.1 ONE BOOT PER CELL

★셀마다 별 boot·별 telemetry 파일이다. 한 boot 안에서 rate ladder를 돌리지 않는다
(`CONSENSUS §3 항목120`). **기제 승계**: `c_capacity.sbatch:111`의
`for SPEC in "${CELLS[@]}"` 루프 + `:123-178`의 셀별 launch/kill/`pkill`(`:30-35`는 그
규율을 서술한 **주석**이지 기제가 아니다 — rev1 판정서 인용 정정). ★**더 강한 차단**:
판정량(`achieved = completed/duration`, `span` = 재현된 도착 간격 합)은 **전부
client-side**이고 telemetry를 한 번도 경유하지 않는다.

### 2.2 shape (D1, 유지)

| shape | 정체 | 앵커 |
|---|---|---|
| **A = (256 in, 512 out)** | **W3 전부**이자 **W4 decode phase 그 자체** | 없음(§3.3) |
| **B = (8192 in, 64 out)** | **W4 prefill phase** | job 905835 D44 @ 8192-in/96-out = **0.675 req/s**(provisional) |

**D1**: `benchmarks/pdmux_eval/workloads.py:159-160`을 실행해 확인 — W4 prefill phase =
(in 8192, out 64), **W4 decode phase = (in 256, out 512)** = shape A 그 자체이며 동시에
W3 전부다. **근사는 존재하지 않는다.** 저장소 전체에 (in 64, out 512)인 워크로드는 없다.
미측정으로 남는 shape는 W1/W5/W6/W7/W8/W9의 것들이다.

### 2.3 통제 요인 (캠페인 러너와 일치)

`--attention-backend flashinfer` · `--context-length 16384` · `--mem-fraction-static 0.82` ·
`--max-running-requests 48` · `--disable-radix-cache` · `--chunked-prefill-size -1` ·
`--disable-overlap-schedule` · cudagraph **ON** · `--random-seed 1` · PD-mux `fixed D44`
(legacy 루프) · TP=1 · config `pdmux_homog5.yml`.

### 2.4 D11 — probe C(905835)와의 차이 (판정서가 1건을 해소)

| 항목 | probe C | 이 등록 | 캠페인 러너 | 처리 |
|---|---|---|---|---|
| mem-fraction | 0.80 | **0.82** | 0.82 | 캠페인에 맞춤 |
| ctx | 8704 | **16384** | 16384 | 캠페인에 맞춤 |
| **`--disable-piecewise-cuda-graph`** | **ON**(`c_capacity.sbatch:126`) | ★**붙이지 않는다** | cudagraph-OFF 분기에서만(`engine_bench_runner.sh:99`) | **캠페인에 맞춤.** 이 차이가 8192-token prefill 처리율을 바꿀 수 있음을 등록한다(미검증) |
| **`--random-seed`** | 미지정 | **1** | run record의 `server_seed` | 등록 |
| **부하 생성기** | `bench_serving` Poisson | 동일 | **`pdmux_eval.trace_loadgen`**(고정 도착시각, `input_ids=[1]*n`) | ★**해소되지 않는 불일치로 등록**(§10-6) |
| pdmux config | `pdmux_homog5.yml`(sm_group_num 5) | 동일 | `pdmux_r2.yml`(6) | ★**판정서가 확인·해소**: `(64,44)` 분할은 양쪽에 존재하고 `_r2_decide_idx`가 `decode_sms` 정확 일치로 고르므로 **fixed 정책 아래서는 동등**. 단 `manual_divisions` 길이가 달라 fixed가 아닌 경로에서는 다른 분할로 간다 |

**ctx 16384 / mem 0.82가 용량을 바꾸는가 — 아니다**: 구속은
`max_mamba_cache_size = max_running_requests = 48`이고 KV는 6.6× 과공급. 그 유일한 미측정
전제는 `lambda0.sbatch`가 셀마다 배너를 기록해 **확인 항목**으로 만든다.

### 2.5 D10 — 하네스 줄

```
python -m sglang.bench_serving --backend sglang --model "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
  --random-input-len <256|8192> --random-output-len <512|64> --random-range-ratio 1.0 \
  --num-prompts "$NP" --request-rate "$RATE" --seed "$CSEED" \
  --output-details --output-file "$OUT/bench_${NAME}.jsonl"
```

`--dataset-path` + `--tokenize-prompt`로 입력 길이가 정확해지고(probe C 오차 0),
`input_len_exact`/`output_len_exact`가 셀마다 기록된다. `--output-details --output-file`
없이는 후속 재채점이 불가능하다. warmup = 8 요청 순차(`--seed 7`), 측정 제외.
★`--random-range-ratio 1.0`의 **두 번째 역할**은 §4.3(D20)에 있다.

### 2.6 D12 — provenance를 GPU 전에 확인

`lambda0.sbatch`는 첫 boot 전에 manifest를 만들고 (a) 항목 수 ≥ **24**, (b)
**`models/nemotron_h.py` 포함**을 확인하지 못하면 `exit 3`. 근거: job 905835는 manifest
17항목인 채로 NemotronH를 12 boot 서빙했다(모델 구현 provenance 0).
판정서 확인: HEAD `bddff6a`의 매니페스트는 **24항목·nemotron_h 포함**(작업트리는 25항목,
`flashinfer_backend.py` 추가, 미커밋).

---

## 3. ★D16–D17 — 판정량의 **영점(κ)** 등록과 측정창 (死因 ① 해소)

### 3.1 판정량이 실제로 무엇인가 (항등식, 원자료 12/12로 재확인)

rev2가 등록한 판정량은

```
achieved_over_realized = achieved / realized_offered
                       = (N/(N−1)) · span / duration
                       = (N/(N−1)) · span / (span + L)      L = duration − span
```

이며 이는 **모델링 선택이 아니라 항등식**이다. 나는 보관 12셀에서 **양변이 1e−12 이내로
일치**함과 `duration = maxᵢ(arrivalᵢ + ttftᵢ + Σitlᵢ)`가 기록된 `duration`을 **≤0.265 s**로
재현함을 독립 재계산으로 확인했다(테스트 `test_drain_identity_on_archived_cells`).

> ★**따라서 비포화 셀은 1을 읽지 않는다. `κ = (N/(N−1))/(1 + L/span)`을 읽는다.**

rev2는 이 추정량을 **배수(drain)가 무시할 만한 shape((8192, 96), L/span 0.7–1.7%)에서만**
검증하고 **배수가 지배하는 shape A(out 512)**에 그대로 적용했다. shape A의 배수는
**511 토큰 × ITL**이고 rev2의 측정창은 170 s 고정이었으므로 κ = 0.91–0.95 —
**저측 문턱 0.95 아래**다. 즉 저측 시험이 조용히 "이 arm의 decode ITL이 20 ms 미만인가"가
됐고, probe C가 **바로 이 arm(D44)에서 실측한 ITL p50은 19.92 / 20.29 / 20.38 / 23.53 ms**다.

★**자기모순(판정서 T1′)의 독립 재현**: 등록 자신의 step 모형으로 rev2 사다리의 저측
rung을 계산하면 κ = **0.9127–0.9459**(내 계산; 판정서 0.938–0.949보다 약간 더 낮다 —
나는 TTFT 부하계수를 함께 넣었다). **한 개도 0.95를 넘지 못한다** ⇒ F3의 참 분지가
등록 자신의 모형 아래 도달 불가였다.

### 3.2 D16 — κ는 **천장**이다. 저측 자격을 κ로 사전 등록한다

> **κ(rung) = 그 rung에서 `achieved_over_realized`가 가질 수 있는 최댓값**이다 —
> 대기 지연이 전혀 없을 때의 값. **κ < 0.95인 rung은 용량보다 아무리 아래에 있어도
> 저측 시험을 통과할 수 없다.**

따라서 등록한다:

> **R1의 저측은 `κ_pred ≥ KAPPA_MIN = ACH_HI + 0.02 = 0.97`인 rung만 공급할 수 있다.**
> (`lambda0_plan.py`가 rung마다 κ를 계산해 실행 전 인쇄하고, `low_side_candidate`
> 플래그를 셀 JSON에 싣는다. `lambda0_label.py`는 그 플래그가 없거나 거짓인 셀을 저측으로
> 쓰지 않으며, **플래그 부재는 fail-closed**다.)

**κ의 배수 모형(등록, 실행 전 고정)** — 이것은 rev2의 결함 하나를 더 고친다: rev2는
**두 개의 무관한 ITL 상수**를 들고 있었다(warmup·예산용 13.06 ms와 λ\*(A) 유도용 회귀).
rev3는 **하나의 모형**만 쓴다.

```
ITL = a + b·min(B̄, 48),   B̄ = x·(out−1)·ITL        (Little)
   ⇒ ITL = a / (1 − b·x·(out−1)),  단 B̄ ≤ 48
   a = 19.82 ms, b = 0.5174 ms/req   (rev1 판정서 §3-3의 srv_d44 회귀)
L̂  = TTFT_floor(shape)·3.2 + (out−1)·ITL
κ  = (N/(N−1))·span/(span + L̂),   span = (N−1)/x
B̄ > 48  ⇒ 그 rung은 **모형이 포화라고 말하는 것**이다(= 고측 rung, κ 미적용)
```

**자기 정합성(코드로 assert)**: `48/((512−1)·step(48)) = 2.10 req/s`로 §3.3의 등록
점추정 λ\*(A)를 **재현한다.** 배수 모형과 용량 추정이 같은 모형이다.

### 3.3 D17 — 움직일 수 있는 유일한 손잡이는 **측정창**이다

κ는 `L̂/span`이 정하고 L̂은 shape가 정하므로, 남는 손잡이는 span = 측정창뿐이다.

> **하위 2 rung에 한해**, `T ∈ (170, 600, 700, …, 1800) s` 중 **κ ≥ 0.97을 만족하는 가장
> 짧은 창**을 고른다(`N_MAX`는 그 rung에 한해 2000). 고측 rung은 배수에 강건하므로
> (읽는 값이 0.5–0.7) **손대지 않는다.** 모형이 포화라고 부른 rung은 연장 대상이 아니다.

비용: 등록 5개 시나리오 + fallback 전부에서 **0.96–1.47 GPU-h**(§3.5 인쇄) — rev2 예측
0.80–0.98 대비 **+0.2~0.5 GPU-h**. 판정서 추산(+0.24)과 같은 크기다.

★**판정서가 제시한 대안(저측 추정량을 "대기 지연 미증가"로 교체)을 채택하지 않은 이유**:
그것은 **새 문턱·새 구간 정의 = 새 자유 표면**이고 그 자신이 도달가능성 사전계산과 함께
등록돼야 한다(판정서 §5-D17 자신의 단서). 값싼 길(D17)은 **승계된 R1을 그대로 두고**
창 하나만 늘린다 — 새 estimand가 없으므로 probe C에서 감사된 규칙의 승계가 유지된다.

### 3.4 ★D16 후반 — κ는 **예측**이므로, 예측이 틀리면 판정을 내리지 않는다

판정서 §5의 자기 처방 실현가능성 검사 (d)를 그대로 승계한다: κ 예측은 step 모형에
의존하므로 **모형이 틀리면 틀린 안전마진을 준다.** 따라서 등록한다:

- 셀마다 **측정된 배수** `drain_s = duration − span`과 `drain_over_span`을 기록한다
  (추가 계측 0 — 둘 다 이미 있는 값의 차).
- 저측 후보 셀에 대해 **`drain_s ≤ 2.0 × L̂`**를 요구한다. 위반하면 그 shape는
  **`UNRESOLVED`**(천장 인증 무효) — 틀린 안전마진 위에서 자신 있는 라벨을 내는 대신.
- 셀마다 **κ_pred · 실측 비율 · `slack = 비율 − 0.95` · `headroom = κ_pred − 0.95`**를
  함께 보고한다(= "저측 문턱을 κ 대비로 보고하라").

★**항등식 함정 회피의 명시**: `(N/(N−1))·span/(span + drain_s)`는
`achieved_over_realized`와 **대수적으로 같다**. 따라서 **측정된 배수로 판정량을 정규화하지
않는다**(그 비는 항상 1이다). 측정 배수는 (a) 부족분의 귀속과 (b) **등록된 예측과의 대조**
에만 쓴다. 이 문장은 코드 주석에도 동일하게 있다.

### 3.5 전 시나리오 사전 인쇄 (게이트 #197 — 자유 표면 0)

아래는 `python3 lambda0_plan.py --registered-scenarios` **출력 전문**이다. 5개 λ_inf
시나리오 + fallback 전부에 대해 사다리·측정창·N·Ē·실현 rate·**ITL̂/B̄/L̂/L/span/κ/저측
자격**·브래킷 창·F4 반복 셀·예산이 실행 전에 고정돼 있다.

```text
### lambda_inf-anchored: point estimate lower  (lambda*(A)=2.1 regression extrapolation)
seed1=4056 (max|Ebar-1|=0.0196)  seed2=2130 (max|Ebar-1|=0.0224)  boots=11  KAPPA_MIN=0.97
  shape A (256, 512)  lambda_inf=2.1
    cell     offered_nom     N    T_s  ITLhat  Bhat  Lhat   L/span   kappa  lowside  est_dur_s
    a_r0          0.525   320   600  23.0    6  11.9  1.95%  0.9839    YES      619.5
    a_r1           1.05   630   600  27.4   15  14.1  2.36%  0.9785    YES      613.2
    a_r2           1.78   300   170  37.4   34  19.2 11.44%  0.9003     -       187.2
    a_r3           2.73   400   170    --    --     --     --      SAT      -       146.5
    a_r4            4.2   400   170    --    --     --     --      SAT      -        95.2
    low-side candidates: ['a_r0', 'a_r1'] (need >= 1; kappa >= 0.97)
    window(realized)     [0.4995, 3.788]
    window(nominal)      [0.4987, 3.78]
    window(conservative) [0.4995, 3.78] = lambda*/lambda_inf in [0.2378, 1.8]
    prior point (2.1, 2.5) inside: True   prior absolute (1.08, 7.15) inside: False   (reported, NOT asserted)
    F4 repeat cell: a_r4_s2 (Ebar@seed2=1.0097)
  shape B (8192, 64)  lambda_inf=0.675
    cell     offered_nom     N    T_s  ITLhat  Bhat  Lhat   L/span   kappa  lowside  est_dur_s
    b_r0          0.304    50   170  20.0    0   4.2  2.59%  0.9946    YES      165.4
    b_r1          0.472    80   170  20.1    1   4.2  2.50%  0.9880    YES      171.6
    b_r2          0.776   130   170  20.3    1   4.2  2.52%  0.9830    YES      170.4
    b_r3           1.15   200   170  20.6    1   4.2  2.43%  0.9812    YES      177.3
    low-side candidates: ['b_r0', 'b_r1', 'b_r2', 'b_r3'] (need >= 1; kappa >= 0.97)
    window(realized)     [0.2937, 1.019]
    window(nominal)      [0.2888, 1.035]
    window(conservative) [0.2937, 1.019] = lambda*/lambda_inf in [0.435, 1.509]
    prior point (0.675, 0.675) inside: True   prior absolute (0.35, 1.2) inside: False   (reported, NOT asserted)
    F4 repeat cell: b_r3_s2 (Ebar@seed2=0.9886)
  BUDGET bench 43.6 min + warmup 6.5 min + boot/teardown 20.2 min = 1.172 GPU-h  (with the one registered redesign round: 2.344)

### lambda_inf-anchored: point estimate upper  (lambda*(A)=2.5 ctx-768 KV-term variant)
seed1=3201 (max|Ebar-1|=0.0171)  seed2=4859 (max|Ebar-1|=0.0203)  boots=11  KAPPA_MIN=0.97
  shape A (256, 512)  lambda_inf=2.51
    cell     offered_nom     N    T_s  ITLhat  Bhat  Lhat   L/span   kappa  lowside  est_dur_s
    a_r0          0.627   380   600  23.8    8  12.2  2.02%  0.9828    YES      616.7
    a_r1           1.25   750   600  29.6   19  15.2  2.54%  0.9765    YES      614.4
    a_r2           2.13   360   170    --    --     --     --      SAT      -       169.0
    a_r3           3.26   400   170    --    --     --     --      SAT      -       122.7
    a_r4           5.02   400   170    --    --     --     --      SAT      -        79.7
    low-side candidates: ['a_r0', 'a_r1'] (need >= 1; kappa >= 0.97)
    window(realized)     [0.5989, 4.532]
    window(nominal)      [0.5957, 4.518]
    window(conservative) [0.5989, 4.518] = lambda*/lambda_inf in [0.2386, 1.8]
    prior point (2.1, 2.5) inside: True   prior absolute (1.08, 7.15) inside: False   (reported, NOT asserted)
    F4 repeat cell: a_r4_s2 (Ebar@seed2=0.9844)
  shape B (8192, 64)  lambda_inf=0.8
    cell     offered_nom     N    T_s  ITLhat  Bhat  Lhat   L/span   kappa  lowside  est_dur_s
    b_r0           0.36    60   170  20.1    0   4.2  2.55%  0.9917    YES      168.1
    b_r1           0.56   100   170  20.2    1   4.2  2.37%  0.9867    YES      181.0
    b_r2           0.92   160   170  20.4    1   4.2  2.43%  0.9824    YES      177.0
    b_r3           1.36   230   170  20.7    2   4.2  2.51%  0.9798    YES      172.6
    low-side candidates: ['b_r0', 'b_r1', 'b_r2', 'b_r3'] (need >= 1; kappa >= 0.97)
    window(realized)     [0.3424, 1.219]
    window(nominal)      [0.342, 1.224]
    window(conservative) [0.3424, 1.219] = lambda*/lambda_inf in [0.428, 1.524]
    prior point (0.675, 0.675) inside: True   prior absolute (0.35, 1.2) inside: True   (reported, NOT asserted)
    F4 repeat cell: b_r3_s2 (Ebar@seed2=1.0031)
  BUDGET bench 42.6 min + warmup 6.5 min + boot/teardown 20.2 min = 1.154 GPU-h  (with the one registered redesign round: 2.308)

### lambda_inf-anchored: absolute lower bound  (A=1.08 worst observed step, B=0.35)
seed1=385 (max|Ebar-1|=0.0204)  seed2=1295 (max|Ebar-1|=0.0221)  boots=11  KAPPA_MIN=0.97
  shape A (256, 512)  lambda_inf=1.08
    cell     offered_nom     N    T_s  ITLhat  Bhat  Lhat   L/span   kappa  lowside  est_dur_s
    a_r0           0.27   160   600  21.3    3  11.0  1.87%  0.9878    YES      599.9
    a_r1           0.54   320   600  23.1    6  11.9  2.02%  0.9833    YES      602.6
    a_r2          0.918   160   170  26.2   12  13.5  7.77%  0.9337     -       186.7
    a_r3            1.4   240   170  31.5   23  16.2  9.47%  0.9173     -       186.9
    a_r4           2.16   370   170    --    --     --     --      SAT      -       171.3
    low-side candidates: ['a_r0', 'a_r1'] (need >= 1; kappa >= 0.97)
    window(realized)     [0.2602, 1.936]
    window(nominal)      [0.2565, 1.944]
    window(conservative) [0.2602, 1.936] = lambda*/lambda_inf in [0.2409, 1.792]
    prior point (2.1, 2.5) inside: False   prior absolute (1.08, 7.15) inside: False   (reported, NOT asserted)
    F4 repeat cell: a_r4_s2 (Ebar@seed2=1.0195)
  shape B (8192, 64)  lambda_inf=0.35
    cell     offered_nom     N    T_s  ITLhat  Bhat  Lhat   L/span   kappa  lowside  est_dur_s
    b_r0          0.158    40   170  19.9    0   4.2  1.69%  1.0086    YES      251.0
    b_r1          0.245    40   170  20.0    0   4.2  2.62%  0.9994    YES      163.4
    b_r2          0.402    70   170  20.1    1   4.2  2.43%  0.9904    YES      175.8
    b_r3          0.595   100   170  20.2    1   4.2  2.52%  0.9853    YES      170.6
    low-side candidates: ['b_r0', 'b_r1', 'b_r2', 'b_r3'] (need >= 1; kappa >= 0.97)
    window(realized)     [0.1485, 0.543]
    window(nominal)      [0.1501, 0.5355]
    window(conservative) [0.1501, 0.5355] = lambda*/lambda_inf in [0.4289, 1.53]
    prior point (0.675, 0.675) inside: False   prior absolute (0.35, 1.2) inside: False   (reported, NOT asserted)
    F4 repeat cell: b_r3_s2 (Ebar@seed2=1.0088)
  BUDGET bench 47.5 min + warmup 6.5 min + boot/teardown 20.2 min = 1.236 GPU-h  (with the one registered redesign round: 2.472)

### lambda_inf-anchored: absolute upper bound  (A=7.15 step(B=1) floor, B=1.20)
seed1=2823 (max|Ebar-1|=0.0105)  seed2=973 (max|Ebar-1|=0.0131)  boots=11  KAPPA_MIN=0.97
  shape A (256, 512)  lambda_inf=7.15
    cell     offered_nom     N    T_s  ITLhat  Bhat  Lhat   L/span   kappa  lowside  est_dur_s
    a_r0           1.79  1250   700  37.6   34  19.3  2.77%  0.9738    YES      717.1
    a_r1           3.58   400   170    --    --     --     --      SAT      -       111.7
    a_r2           6.08   400   170    --    --     --     --      SAT      -        65.8
    a_r3           9.29   400   170    --    --     --     --      SAT      -        43.1
    a_r4           14.3   400   170    --    --     --     --      SAT      -        28.0
    low-side candidates: ['a_r0'] (need >= 1; kappa >= 0.97)
    window(realized)     [1.719, 12.8]
    window(nominal)      [1.7, 12.87]
    window(conservative) [1.719, 12.8] = lambda*/lambda_inf in [0.2404, 1.791]
    prior point (2.1, 2.5) inside: True   prior absolute (1.08, 7.15) inside: False   (reported, NOT asserted)
    F4 repeat cell: a_r4_s2 (Ebar@seed2=1.0066)
  shape B (8192, 64)  lambda_inf=1.2
    cell     offered_nom     N    T_s  ITLhat  Bhat  Lhat   L/span   kappa  lowside  est_dur_s
    b_r0           0.54    90   170  20.2    1   4.2  2.54%  0.9862    YES      169.0
    b_r1           0.84   140   170  20.4    1   4.2  2.54%  0.9823    YES      169.7
    b_r2           1.38   230   170  20.8    2   4.2  2.54%  0.9795    YES      170.2
    b_r3           2.04   350   170  21.2    3   4.3  2.48%  0.9786    YES      175.3
    low-side candidates: ['b_r0', 'b_r1', 'b_r2', 'b_r3'] (need >= 1; kappa >= 0.97)
    window(realized)     [0.5124, 1.821]
    window(nominal)      [0.513, 1.836]
    window(conservative) [0.513, 1.821] = lambda*/lambda_inf in [0.4275, 1.517]
    prior point (0.675, 0.675) inside: True   prior absolute (0.35, 1.2) inside: False   (reported, NOT asserted)
    F4 repeat cell: b_r3_s2 (Ebar@seed2=0.9985)
  BUDGET bench 30.9 min + warmup 6.5 min + boot/teardown 20.2 min = 0.959 GPU-h  (with the one registered redesign round: 1.919)

### lambda_inf-anchored: B=6 plateau variant   (A=3.91, B=1.00)
seed1=38 (max|Ebar-1|=0.0155)  seed2=1662 (max|Ebar-1|=0.0191)  boots=11  KAPPA_MIN=0.97
  shape A (256, 512)  lambda_inf=3.91
    cell     offered_nom     N    T_s  ITLhat  Bhat  Lhat   L/span   kappa  lowside  est_dur_s
    a_r0          0.978   590   600  26.7   13  13.8  2.28%  0.9793    YES      616.0
    a_r1           1.96  1370   700  41.1   41  21.1  3.02%  0.9714    YES      719.6
    a_r2           3.32   400   170    --    --     --     --      SAT      -       120.5
    a_r3           5.08   400   170    --    --     --     --      SAT      -        78.7
    a_r4           7.82   400   170    --    --     --     --      SAT      -        51.2
    low-side candidates: ['a_r0', 'a_r1'] (need >= 1; kappa >= 0.97)
    window(realized)     [0.9436, 7.107]
    window(nominal)      [0.9291, 7.038]
    window(conservative) [0.9436, 7.038] = lambda*/lambda_inf in [0.2413, 1.8]
    prior point (2.1, 2.5) inside: True   prior absolute (1.08, 7.15) inside: False   (reported, NOT asserted)
    F4 repeat cell: a_r4_s2 (Ebar@seed2=0.9985)
  shape B (8192, 64)  lambda_inf=1.0
    cell     offered_nom     N    T_s  ITLhat  Bhat  Lhat   L/span   kappa  lowside  est_dur_s
    b_r0           0.45    80   170  20.1    1   4.2  2.38%  0.9891    YES      179.7
    b_r1            0.7   120   170  20.3    1   4.2  2.47%  0.9841    YES      174.2
    b_r2           1.15   200   170  20.6    1   4.2  2.43%  0.9812    YES      177.3
    b_r3            1.7   290   170  21.0    2   4.2  2.49%  0.9791    YES      174.2
    low-side candidates: ['b_r0', 'b_r1', 'b_r2', 'b_r3'] (need >= 1; kappa >= 0.97)
    window(realized)     [0.4211, 1.519]
    window(nominal)      [0.4275, 1.53]
    window(conservative) [0.4275, 1.519] = lambda*/lambda_inf in [0.4275, 1.519]
    prior point (0.675, 0.675) inside: True   prior absolute (0.35, 1.2) inside: False   (reported, NOT asserted)
    F4 repeat cell: b_r3_s2 (Ebar@seed2=1.0129)
  BUDGET bench 41.9 min + warmup 6.5 min + boot/teardown 20.2 min = 1.144 GPU-h  (with the one registered redesign round: 2.287)

### FALLBACK (audit-recommended literal ladders; selected only by the objective predicate in prereg sec 3.7)
seed1=4386 (max|Ebar-1|=0.0146)  seed2=4162 (max|Ebar-1|=0.0151)  boots=11  KAPPA_MIN=0.97
  shape A (256, 512)  lambda_inf=2.1
    cell     offered_nom     N    T_s  ITLhat  Bhat  Lhat   L/span   kappa  lowside  est_dur_s
    a_r0            1.1   660   600  27.9   16  14.4  2.40%  0.9781    YES      613.5
    a_r1            1.8  1260   700  37.8   35  19.4  2.78%  0.9738    YES      718.9
    a_r2              3   400   170    --    --     --     --      SAT      -       133.3
    a_r3            4.9   400   170    --    --     --     --      SAT      -        81.6
    a_r4              8   400   170    --    --     --     --      SAT      -        50.0
    low-side candidates: ['a_r0', 'a_r1'] (need >= 1; kappa >= 0.97)
    window(realized)     [1.03, 7.191]
    window(nominal)      [1.045, 7.2]
    window(conservative) [1.045, 7.191] = lambda*/lambda_inf in [0.4976, 3.424]
    prior point (2.1, 2.5) inside: True   prior absolute (1.08, 7.15) inside: True   (reported, NOT asserted)
    F4 repeat cell: a_r4_s2 (Ebar@seed2=1.0151)
  shape B (8192, 64)  lambda_inf=0.675
    cell     offered_nom     N    T_s  ITLhat  Bhat  Lhat   L/span   kappa  lowside  est_dur_s
    b_r0           0.45    80   170  20.1    1   4.2  2.38%  0.9891    YES      179.7
    b_r1           0.62   110   170  20.2    1   4.2  2.38%  0.9857    YES      180.0
    b_r2           0.85   140   170  20.4    1   4.2  2.57%  0.9820    YES      167.7
    b_r3           1.15   200   170  20.6    1   4.2  2.43%  0.9812    YES      177.3
    low-side candidates: ['b_r0', 'b_r1', 'b_r2', 'b_r3'] (need >= 1; kappa >= 0.97)
    window(realized)     [0.4252, 1.034]
    window(nominal)      [0.4275, 1.035]
    window(conservative) [0.4275, 1.034] = lambda*/lambda_inf in [0.6333, 1.532]
    prior point (0.675, 0.675) inside: True   prior absolute (0.35, 1.2) inside: False   (reported, NOT asserted)
    F4 repeat cell: b_r3_s2 (Ebar@seed2=0.9948)
  BUDGET bench 42.2 min + warmup 6.5 min + boot/teardown 20.2 min = 1.147 GPU-h  (with the one registered redesign round: 2.294)
```

### 3.6 ★사다리 하한의 출처 — 인접 사전등록과의 모순 해소

`results/r2_correctness/newpair_prereg/PREREG_NEWPAIR_2026-09-13.md` §5-I3 **(D7)**·
**NP-3′**가 "**I3는 사다리의 위쪽 끝만 고정한다 … I3를 하한에 쓰면 브래킷을 놓친다**"를
등록했는데, rev2의 `MULT[A][0] = 0.40·λ_inf`는 **하한을 λ_inf에 걸고** 있었다. λ_inf는
λ\*의 **상한**이므로 λ\*/λ_inf가 작으면 `0.40·λ_inf`가 λ\* 위에 남아 비포화 셀이 하나도
없게 된다. **rev3는 newpair 쪽이 옳다고 보고 이쪽을 고친다**:

1. **`MULT[A][0]`을 0.40 → 0.25로 내린다.** 창의 하단이 `0.95×0.25 = 0.2375·λ_inf`가 되어
   λ\*/λ_inf가 **0.24까지 떨어져도** 브래킷이 구성상 성립한다(rev2는 0.45가 한계였다).
2. **가정을 데이터로 검사되는 분지로 바꾼다.** 최하위 rung이 포화로 돌아오면 그 shape는
   `LADDER_TOO_HIGH`이고, 등록된 **결정적** 재설계는 사다리를 **4로 나눈다**. 반대로
   최상위 rung이 비포화면 `LADDER_TOO_LOW`이고 **4를 곱한다**. 재설계는 전 단계 통틀어
   **1회**, 사람 판단 개입 0. 2회차 이상은 사용자 승인 사항이다.
3. ★**이 단계의 어떤 주장도 "λ_inf가 λ\*를 아래에서 묶는다"에 의존하지 않는다.**

*(코디네이터 메모: newpair 쪽 재감사가 (D7)을 고치는 결론을 내면 조정은 코디네이터가
한다. rev3는 그 결론과 무관하게 안전한 쪽 — 하한을 내리고 분지를 등록하는 쪽 — 을 잡았다.)*

### 3.7 ★D18 — fallback 분지를 객관 술어로 (死因 ② 해소)

rev2는 운영자 env 하나(`PDMUX_LAMBDA0_FALLBACK=1`)로 분기했고, 판정서는 그 재량이 라벨을
뒤집는 실례를 줬다(λ\*(B)=1.10에서 앵커 창 `[0.5157, 1.835]` = BRACKETED / fallback 창
`[0.4302, 1.026]` = NOT_BRACKETED). **rev3에는 운영자 스위치가 없다.**
`lambda0_lambda_inf.py`가 **파일로 판정한다**:

> **ANCHORED ⟺** (a) `<instr>/I3a_shapeA.jsonl`·`I3b_shapeB.jsonl`이 **둘 다 존재·파싱**
> 되고 각각 **`request_rate`가 inf이며 `request_throughput` > 0**, **그리고**
> (b) `<instr>/I3_max_running_req.txt`가 존재하고 **거기 적힌 모든 값이 ≥ 48**
> (newpair **F5**: 48에 미달한 셀의 achieved는 포화 처리율로 인용 금지).
> 그 밖에는 **FALLBACK**.
> λ_inf(A) = I3a의 `request_throughput`, λ_inf(B) = I3b의 것.

`PDMUX_LAMBDA0_INSTR`는 **어디를 볼지**만 말하고 **무엇을 결론지을지**는 말하지 않는다.
selftest가 **각 조건이 단독으로 FALLBACK을 강제함**을 보인다(술어가 장식이 아님).

★**λ_inf의 지위(유지)**: 상한 프로브이며 λ\*의 대체가 아니다. 판정서 §8의 자기
실현가능성 검사를 승계한다 — closed-loop(concurrency 64)는 open-loop보다 admitted가 더
찰 수 있어 **λ_inf ≥ λ\*_Poisson이 기대되나 보장되지 않는다.** λ_inf는 **사다리를 놓는
데만** 쓰고, 보고되는 λ\*는 언제나 이 단계의 Poisson 셀에서 읽는다.

---

## 4. D2 (rev1 死因 N2-S1) — 도착 실현 · ★D20 정정

### 4.1 등록해야 할 사실 (유지)

`bench_serving.py:1705-1706`이 프로세스당 한 번 seed를 걸고 `:943-950`이 간격을
`np.random.exponential(1.0/request_rate)`로 뽑는다(yield 후 sleep ⇒ N개 요청에 N−1개 간격).

> ★**비포화 셀의 `achieved/offered`(명목 분모)는 포화도가 아니라 도착 실현 계수 `1/Ē`다.**
> 905835 실측: **12/12 셀 전부** 도착 구간이 명목보다 짧았고(Ē = 0.826–0.872, seed 41),
> 11/12 셀이 `ach/off > 1`(최대 1.1886)을 보고했다. seed는 프로세스당 한 번 고정되므로
> 사다리 전체에 **공통모드**로 들어가고 셀을 늘려도 평균되지 않는다.
> `1.96/√(N−1) ≤ 0.05`는 **N ≥ 1537**을 요구한다. seed 40개 모의 저측 실패율 **15–28%**.

### 4.2 선택: (b) 실현 도착률 분모 — 그리고 그 **한계**(rev3에서 새로 등록)

판정량은 `achieved / realized_offered`, `realized_offered = (N−1)/span`.
오프라인·GPU 0으로 재현 가능하다. 보관 원자료에서 청정 비포화 셀 3/3이 명목 1.170/1.142/
1.189 → **0.994/0.996/1.018**로 수렴하고, **감사된 probe C 판정을 바꾸지 않는다**
(세 arm 전부 `KNEE_BRACKETED`, λ\* = 0.9331188685063582 / 0.675302644015995 /
0.18668462822130108 — `C_LABEL.json`과 비트 단위 일치, selftest가 매번 재확인).

★★**그러나 rev2가 이 수렴에서 끌어낸 일반화는 틀렸다** — §3.1의 이유로 그것은
**(8192, 96) 한정 사실**이다(§9 **L2**). rev3는 그 한계를 κ로 명시 등록한다.

### 4.3 ★D20 — `--random-range-ratio 1.0`은 **RNG 재현의 필요조건**이다

rev2는 "`bench_serving.py`에서 `np.random` 소비처가 정확히 2곳"이라고 적었다. 그것은
**그 파일에 대해서는 참이고 경로에 대해서는 거짓**이다. 판정서 지적을 내가 직접 실행해
확인했다:

- `sglang/benchmark/datasets/common.py:56-64` `compute_random_lens`가
  `np.random.randint(max(int(full·rr),1), full+1, size=num)`를 **run당 2회**(입력/출력 길이)
  소비한다(`datasets/random.py:66-76`).
- **rr = 1.0이면 구간이 퇴화**(`randint(L, L+1)`)해 numpy가 스트림을 **전혀 소비하지
  않는다.** 실행 확인: rr=1.0 → 이후 지수 스트림 `[0.59970503, …]`(무소비와 동일),
  rr=0.9 → `[0.49437720, …]`로 **이동**.

⇒ 등록: **`--random-range-ratio 1.0`은 길이 정확성뿐 아니라 오프라인 도착 재현의 필요조건**
이며, `lambda0_analyze.py`가 bench 레코드의 `random_range_ratio`를 **되읽어** 1.0이 아니면
그 셀을 **`UNRESOLVED`**로 만든다.

★**동시에 rev2 §4.4의 서술을 철회한다**: "Ē 가드 위반 = 오프라인 재현 파손"은 **성립하지
않는다.** Ē는 **재현 스트림 자신에서** 계산되므로 재현이 깨진 사건을 원리적으로 볼 수
없다. Ē 가드가 실제로 잡는 것은 **등록되지 않은 seed·num-prompts로 돈 셀**뿐이다.
재현 파손을 잡는 것은 `range_ratio_ok`다.

### 4.4 seed는 정수로 등록된다 (유지)

`Ē`는 **(seed, N)에만** 의존한다(간격 = X/rate, X ~ Exp(1)). 따라서:

> **seed1, seed2 = `1..5000` 중 `max_cells |Ē(seed, N_cell) − 1|`를 최소화하는 두 seed
> (동률이면 작은 seed).**

전 시나리오에서 두 seed 모두 `|Ē−1| ≤ 0.025`(판정서 option (c) 밴드 `1/Ē ∈ [0.95,1.05]`의
절반 폭). 결과 seed는 §3.5에 전부 인쇄돼 있다.

---

## 5. 판정 규칙 (데이터 생성 전 고정 — `lambda0_label.py`)

| # | 이름 | 규칙 | 성격 |
|---|---|---|---|
| **R0** | `CELL_VALIDITY[shape]` | 모든 셀: `1/Ē ∈ [0.95,1.05]` **및** `random_range_ratio = 1.0`. **저측 후보**는 추가로 `drain_s ≤ 2.0·L̂` | 측정 유효성 가드(위반 → `UNRESOLVED`) |
| **R1** | `KNEE_BRACKETED[shape]` | **등록된 저측 후보(κ_pred ≥ 0.97)** 중 `ach/realized ≥ 0.95`인 셀이 있고, **offered가 더 높은** 셀에 `≤ 0.90`이 있다 | 존재, shape별 |
| **R2** | `LAMBDA_STAR[shape]` | 문턱 없는 **측정값** — 포화 셀(≤0.90) 중 **최대 achieved rate**. **R1이 BRACKETED일 때만** | 측정 |
| **R3** | `SEED_REPEAT[shape]` | 제2 seed로 반복한 **최상단 rung**이 포화를 유지하고 λ\*를 **5% 이내** 재현. ★**브래킷이 없으면 `UNRESOLVED (not evaluated)`** | ★등록 예보 |
| — | 비브래킷의 **방향** | 최하위 rung이 포화 → `LADDER_TOO_HIGH`(÷4) · 최상위가 비포화 → `LADDER_TOO_LOW`(×4) · 그 밖 → `KNEE_NOT_BRACKETED` | 재설계 결정성 |
| — | 기대 셀 결손 | 그 shape는 **`UNRESOLVED`** — 측정 실패이며 규칙 실패가 아니다(교훈 21) | |

**D4**: 규칙 코드는 **기대 셀 이름 목록을 인자로 받아 순회**하고 glob하지 않는다.
`--mutation-missing-cell`이 rev1 결함을 주문형으로 재현한다.

**D7**: F4/R3는 **shape별 독립**이고 반복 셀은 **사전에 최상단 rung으로 등록**돼 있다.
반복 셀은 R1 사다리에 넣지 않는다(같은 offered rate 2셀이 자기 자신을 브래킷할 수 있어
`hi[0] > lo[0]` 동률 가드가 load-bearing이고, selftest가 고정한다).

**★D21**: R3가 `lambda_star is None`일 때 `SEED_REPEAT_REFUTED`(rel_diff NaN)를 내던
rev2 동작을 고쳐 **`UNRESOLVED (not evaluated: no bracket)`**로 한다 — **평가되지 않은
예보를 반증으로 기록하지 않는다**(게이트 #21 재발 차단).

### 5.1 ★D8/D19 — 출력공간 전수, 그리고 **철회된 정리**

셀 하나의 비율 `r`:

| `r` | 라벨 | R1/R2에서의 역할 |
|---|---|---|
| `r ≥ 0.95` **且 저측 후보** | 비포화(저측 자격) | 브래킷의 **아래**가 될 수 있다 |
| `r ≥ 0.95` **但 저측 후보 아님** | 비포화(자격 없음) | **어느 쪽도 아니다**(천장이 문턱 아래인 rung) |
| `0.90 < r < 0.95` | ★**사각지대** | **어느 쪽도 아니다.** R2의 포화 집합에도 들어가지 않는다. `blind_spot_cells`에 열거 |
| `r ≤ 0.90` | 포화 | 브래킷의 **위**가 될 수 있고 R2의 후보 |

shape별 verdict 전수: `UNRESOLVED`(기대 셀 결손 **또는** R0 위반) · `KNEE_BRACKETED`
(λ\* 보고 + R3 평가) · `LADDER_TOO_HIGH` / `LADDER_TOO_LOW` / `KNEE_NOT_BRACKETED`
(λ\* 미보고, R3는 `UNRESOLVED`). ★어떤 조합도 정책·arm 순위·Claim 등급을 움직이지 않는다.

★★**D19 — rev2 §3.2의 "사각지대 rung은 최대 1개" 정리를 철회한다.** 그 정리는 이상화
비율 `min(1, λ*/x)` 위에서만 참인 **MULT 상수의 산술 항등식**이었다. 실제 추정량에서는
rev2 예보 사다리의 **최저 두 rung이 동시에** (0.90, 0.95)에 떨어진다(0.940, 0.929 — 내
모형으로는 0.934, 0.915). 대체물은 정리가 아니라 **§3.2의 κ 자격 조건**이다: 사각지대에
떨어질 rung은 **애초에 저측 자격을 받지 못한다.** `lambda0_plan.py --selftest`가 그
반례를 상시 보관한다.

### 5.2 검증 (D9/D22) — 변이 **36종**, 전부 차단

rev1 판정서는 rev1 selftest에서 11개 중 **7개 ESCAPE**를 보고했다. rev2는 등록한 17종을
전부 막았으나 **`lambda0_label.py` 한 파일만** 변이시켰고, 재감사자의 독립 6종 중
**5종(X1·X2·X3·X4·X5)이 escape**했다 — 그중 X3는 **핵심 추정량 자신**(실현 분모 N−1→N)
이었다.

rev3의 하네스는 **decision path 전체**(`lambda0_label.py`·`lambda0_analyze.py`·
`lambda0_plan.py`)를 변이시키고, selftest에 **end-to-end leg**를 넣었다(게이트 #181):
**analyzer의 실제 출력 dict를 `rule()`에 그대로 투입**하고, 판정량을 **정의로부터 독립
재유도한 값**과 1e−12로 대조한다. 그래서 analyzer 변이가 보인다.

**결과: CONTROL 2/2 PASSES · M1–M17 · X1–X6 · R3a–R3m 합 36/36 FAILS · escapes none.**
(`lambda0.sbatch`는 첫 boot 전에 이 하네스를 돌리고, escape가 하나라도 있으면
`ABORT_MUTATION_ESCAPE`로 중단한다.)

---

## 6. 예보와 출력공간 라벨 (F1–F4)

- **F1 (부팅)**: 전 셀 boot 성공(근거 = job 905835 12/12). 실패 시 **측정 실패**로 기록하고
  engine-porter 트리아지.
- **F2 (shape B)**: `KNEE_BRACKETED`. λ\*(B)는 앵커 0.675와 같을 필요 없다. 수치 예보 없음.
- **F3 (shape A)**: `KNEE_BRACKETED`. ★rev2에서는 이 분지가 **자기 모형 아래 도달 불가**
  였다. rev3에서는 **저측 후보가 시나리오마다 ≥1개 존재하고 그 천장이 0.97 이상**임이
  실행 전에 assert된다(§3.5 인쇄 + `--selftest`). 남는 불확실성은 **모형 오차**이고,
  그것은 F3를 거짓 BRACKETED로 만들지 않고 `drain_model_ok` 위반 → `UNRESOLVED`로 간다.
- **F4 = R3**: 각 shape의 최상단 rung을 제2 seed로 1회 반복했을 때 (a) 포화를 유지하고
  (b) λ\*를 **5% 이내** 재현. 근거: 이미 용량에 도달한 인접 두 셀의 achieved 차이가
  d92 0.16%, d16 1.67%. 반증되면 λ\*를 두 seed의 구간으로 보고하고 5% 주장을 `REFUTED`로
  등재한다. **브래킷이 없으면 평가하지 않는다**(D21).

---

## 7. 이 단계가 **주지 않는 것** · D13 · D15

- **성능·정책 판정 0건** — arm 비교 없음, goodput 평가 없음. Claim D/E 등급 불변 ·
  선결 #1–#5 불변 · HE0 · 정책 순위 · stake #1 불변.
- **새 모델의 correctness를 건드리지 않는다** — (Nano-9B-v2, flashinfer) 쌍의 R2
  correctness 게이트는 아직 한 번도 돌지 않았다. 이 단계는 **legacy 루프 + fixed split**만
  발화시킨다(§9 Q2).
- **단일 스칼라 λ\*를 W1–W9가 공유하는 설계 문제는 이 단계가 해소하지 않는다**(§9 Q3).

**D13 (유지·재확인)** — 상단 셀의 TTFT p99는 ≈90–130 s로 probe C 최댓값 101.43 s 위
**무검증 영역**이다. 사전 확인(코드 직접): `bench_serving.py:70`
`BENCH_AIOHTTP_TIMEOUT_SECONDS = 6*60*60` ⇒ 클라이언트 타임아웃 무구속;
`scheduler.py:2013-2014` `max_queued_requests is None` ⇒ admission 제한 미적용(503 없음).
남는 상한은 SLURM 벽시간이며 rev3는 **`--time=04:30:00`**(최악 시나리오 1.47 GPU-h의 3배)
로 올렸다. ★**D17의 긴 창은 고측 rung을 건드리지 않으므로 이 무검증 영역을 넓히지 않는다**
(저측 rung은 비포화라 TTFT가 바닥 근처다).

**D15 (재정정)** — **요청 예산 1.15 → 1.60 GPU-h(11 boot)**, 등록된 재설계 회차 포함
최악 **≈2.9 GPU-h**(두 번째 제출). 분해 모형의 시나리오별 예측은 **0.96–1.47 GPU-h**
(§3.5의 BUDGET 줄). 증가분은 전부 D17의 저측 창 연장이다. 분해 = 셀당 boot/teardown
**108.8 s**(905835 잔차: 1.11 GPU-h − 보관 12셀 `duration_s` 합 2394.1 s − warmup 모형
296.4 s, ÷12) + warmup(shape A 53.5 s / B 13.9 s) + 예상 bench 시간.
이 트랙 GPU 장부: R2 correctness 0.43 GPU-h(별개) — 이 단계는 아직 **0**.

---

## 8. 동봉 실행체와 재현 절차

| 파일 | 역할 |
|---|---|
| `lambda0_plan.py` | 사다리·측정창·N·seed·**κ 표**·창·예산의 **결정적** 생성기. `--registered-scenarios` / `--selftest` |
| `lambda0_lambda_inf.py` | ★**D18** 앵커/fallback 객관 술어(파일 기반). `--selftest` |
| `lambda0_cells.py` | plan.json → sbatch 셀 스펙 렌더러(`-` placeholder로 bash `read` 정렬 보존). `--selftest` |
| `lambda0_analyze.py` | per-cell analyzer. `shape`·실현 도착률·**span/drain**·κ_pred·가드·길이 검증·백분위 |
| `lambda0_cellprint.py` | 셀 1줄 콘솔 요약(비율·천장·배수·가드를 함께) |
| `lambda0_label.py` | 판정 규칙 R0–R3 + end-to-end leg. `--selftest`, `--mutation-missing-cell` |
| `lambda0_mutation_check.py` | 변이 36종 × 3모듈 + 무변이 CONTROL 2종 |
| `lambda0.sbatch` | **제출 실행체**. ONE BOOT PER CELL, 절대경로, `--comment`, D12 게이트, D18 술어, D23 EXPECT 선구성, 변이 게이트 |

```bash
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc
P=workspace/engine-port/results/r2_eval/lambda0_prereg
python3 $P/lambda0_plan.py --selftest
python3 $P/lambda0_label.py --selftest
python3 $P/lambda0_lambda_inf.py --selftest
python3 $P/lambda0_cells.py --selftest
python3 $P/lambda0_mutation_check.py            # 0 escapes required
python3 $P/lambda0_plan.py --registered-scenarios

sbatch $P/lambda0.sbatch     # 앵커/fallback은 파일이 정한다 (운영자 스위치 없음)
```

CPU 회귀는 `workspace/engine-port/tests/test_lambda0_prereg.py`가 위 전부(+ sbatch 규율
+ **음성대조 3종**: rev1 glob · rev2의 170 s 저측 rung · 배수 항등식)를
`python -m unittest discover -s workspace/engine-port/tests` 안에서 고정한다.

---

## 9. ★인용 금지 — Q1–Q5(rev1 판정서 §6, 문자 승계) + **L1–L3**(rev2 판정서 §7, 문자 승계)

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

★**Q3 편집 주석(인용문 불변)**: Q3의 `generate_campaign.sh:18`은 rev1 판정서의 표기이며
HEAD `bddff6a`의 정확한 줄은 **`:17`**이다(§11.1). 같은 날 병행 워크스트림이 **워크로드
정의 쪽을** phase별 독립 λ\*로 고쳤으므로 Q3의 마지막 문장은 **여전히 참**이다 — 이 단계는
**두 수**를 줄 뿐이고, 그 수를 phase별로 쓰게 만드는 코드 변경은 다른 워크스트림의
산출이다. **이 단계가 그 변경을 자기 공로로 인용하는 것을 금지한다.**

> **Q4** — **"B1의 용량이 시스템의 용량"이 아니다.** 워크로드 이름(W8 "near saturation",
> W9 "overload")은 **B1에서만** 참인 서술이고, B4에서 같은 trace는 자기 용량의 0.2×일
> 수도 5×일 수도 있다. 워크로드 이름을 regime 서술로 인용하는 것을 금지한다.

> **Q5** — **shape A·B 두 shape를 쟀다는 것이 W1·W5·W6·W7·W8·W9의 부하 라벨을 고치지
> 않는다.** 그 워크로드들의 shape는 미측정으로 남으며, λ\*는 shape 의존적이다.

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
> 근거로 인용하는 것을 금지한다**(합성 쌍둥이 케이스만 인용 가능).

*(L3 준수 확인: `lambda0_label.py` selftest는 M1 차단을 **합성 쌍둥이 (3b)**로 고정하고,
보관 데이터의 `n_saturated_cells == 2`는 "기록"이라고 코드 주석에 명시했다.)*

---

## 10. ★실행 후 필수 병기 6항 (rev1 판정서 §5 문안 승계 — 어떤 라벨이 나오든 무조건)

1. **λ\*는 B1(legacy, fixed D44) 한정 throughput 포화율이다.** split을 바꾸면 같은 모델·
   같은 shape에서 **5× 변한다**(D16 0.933 / D44 0.675 / D92 0.187 req/s). B4의 λ\*는
   측정되지 않았다.
2. **정본 술어 goodput은 이 단계에서 측정되지 않았다.**
3. **상단 셀의 TTFT/ITL 백분위는 인용 불가다** — 의도적 과포화 셀이다.
4. `switch_count`·split 체류분포는 이 단계의 판정에 쓰이지 않았다(fixed split).
5. **포화 셀의 achieved는 도착 실현에 무관하나(λ\*는 robust), 비포화 셀의 `ach/off`는 도착
   실현 계수다**(905835에서 1.03–1.19, 12/12 동일 방향). ★**그리고 실현 분모로 바꾼 뒤에도
   비포화 셀의 영점은 1이 아니라 κ다**(L1).
6. 이 단계가 쓴 부하 생성기는 `sglang.bench_serving`(Poisson)이고 캠페인은
   `pdmux_eval.trace_loadgen`(고정 도착시각, `input_ids=[1]*n`)이다. **λ\*는 다른
   하네스에서 측정된 상수로 캠페인을 파라미터화한다.**

---

## 11. 인용 고정 · 하류 인터페이스

### 11.1 인용 고정표 (교훈 80 · 게이트 #110)

병행 워크스트림이 캠페인 생성기·워크로드 정의·`r2_correctness.sbatch`를 작업트리에서
고치는 중이므로(미커밋), 캠페인 쪽 인용은 **커밋 `bddff6a`(HEAD)에 고정**한다.

| 인용 | 확인된 내용 | 고정 트리 | worktree도 유효 |
|---|---|---|---|
| `benchmarks/pdmux_eval/workloads.py:159-160` | `8192 if is_prefill else 256` / `64 if is_prefill else 512` | HEAD `bddff6a` | **예**(병행 워크스트림이 의도적으로 보존) |
| `benchmarks/pdmux_eval/workloads.py:150` | `per_phase = max(1, round(0.80 * sustainable_rate * intensity * 30.0))` | HEAD | 아니오(`_phase_count(...)`로 교체) |
| `benchmarks/pdmux_eval/campaign.py:139-140` | `ttft_slo_ms=3000.0` / `itl_slo_ms=60.0` | HEAD | 아니오 |
| `scripts/r2_eval/generate_campaign.sh:17` | `sustainable_rate="${PDMUX_SUSTAINABLE_RATE:-4}"` | HEAD | 아니오(shape별 표로 교체) |
| `scripts/r2_eval/engine_bench_runner.sh:99` | `server_args+=(--disable-cuda-graph --disable-piecewise-cuda-graph)` | HEAD | 예 |
| `scripts/r2_eval/r2_eval.sbatch:38` | `project_root="${PDMUX_PROJECT_ROOT:-…}"` | HEAD | 예 |
| `c_capacity.sbatch:76`/`:78-79`/`:111`/`:123-178`/`:126`/`:139-145` | ρ 사다리 `{0.60,0.90,1.30}` / 1.30 근거 / 셀 루프 / 셀별 launch·kill / piecewise / warmup | HEAD | 예(보관 아티팩트) |
| `sglang/bench_serving.py:70` · `:948` · `:1354` · `:1705-1706` | 6시간 타임아웃 · 간격 추출 · LoRA 전용 · seed | dev tree | — |
| `sglang/benchmark/datasets/common.py:56-64` · `datasets/random.py:66-76` | `compute_random_lens`의 `np.random.randint` **run당 2회** | dev tree | — |
| `sglang/srt/managers/scheduler.py:2013-2014` | `max_queued_requests is None → 제한 미적용` | dev tree | — |
| `sglang/srt/server_args.py:1959` | `assert self.attention_backend != "triton"` | dev tree | — |

★**판정서 인용의 한 줄 어긋남 2건을 정정한다**(rev2에서 등재, rev2 재감사가 독립 확인):
`generate_campaign.sh:18` → **`:17`**, probe C ρ 사다리는 `:78-79`가 아니라 **`:76`**.
어느 쪽도 판정서의 **주장**을 바꾸지 않는다.

★**도구 한계 기록**: `scripts/discipline/check_line_citations.py`의 `SEARCH_ROOTS`는
`benchmarks/`와 `srt` 밖 `sglang/bench_serving.py`를 해석하지 못한다(bare basename이
`UNRESOLVED`). 이 표는 그 구멍을 수작업으로 메운 것이며, 공유 도구는 이 세션에서 고치지
않았다(같은 시각 다른 에이전트가 인접 파일을 편집 중).

### 11.2 하류 인터페이스

병행 워크스트림이 캠페인 생성기에 **shape별 λ\* 표**(`benchmarks/pdmux_eval/lambda_star.py`,
schema `pdmux.lambda_star/v1`)를 도입했고 `definition` 필드가 **`throughput_saturation`** /
`slo_sustainable`을 구분한다. 이 단계의 산출 `LAMBDA0_LABEL.json`은
**`throughput_saturation`** 쪽에만 들어간다(shape key `256x512` / `8192x64`, `source`를
`unmeasured` → `measured`, `evidence`에 job id + 이 사전등록 경로). ★`slo_sustainable`은
이 단계 이후에도 **비어 있다**(§9 Q1). **번역기 자체는 통합 시점에 코디네이터가 배선한다 —
rev3는 그 표로 번역될 것을 전제할 뿐 번역을 수행하지 않는다.**

---

## 12. 반증 실패 항목 (공정 기록 — rev2 판정서 §6 승계)

- **R1/R5의 실질 승계**: 보관 cell JSON 직접 투입으로 `C_LABEL.json`을 비트 단위 재현
  (0.9331188685063582 / 0.675302644015995 / 0.18668462822130108). 항등식 아님, 날조 아님.
- **§3.5 전 시나리오 인쇄**: 생성기 출력과 바이트 동일(전사 드리프트 0).
- **변이 하네스 CONTROL의 비공허성**: 아카이브 파손 시 CONTROL이 FAILS·RC 2.
- **인용 규율(게이트 #110)**: §11.1이 HEAD에 고정, 재감사가 7개 인용을 양쪽에서 재확인.
- **`CONSENSUS §3 항목120` 회피**: 성립(판정량이 telemetry를 경유하지 않음).
- **D4 음성대조**: rev1 glob 재현이 실제로 틀린 답을 낸다.
- **ONE BOOT PER CELL 기제 승계**: `c_capacity.sbatch:111`·`:123-178` 직접 대조.
- **R4(MODEL_HOLDS)를 승계하지 않은 결정**: 옳다(그 닫힌 형태는 항등식). 되살리지 말 것.
- **ctx 16384 / mem 0.82가 용량을 바꾸는가**: 바꾸지 않는다.
- **arm 비교 금지·성능 판정 0건**: 유지 — 이 단계는 어떤 결과가 나와도 HE0·정책 순위·
  Claim D/E 등급·stake #1을 움직이지 않는다.
