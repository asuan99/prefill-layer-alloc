> ## ★ SUPERSEDED (2026-09-13) — 인용 금지, 삭제 금지
>
> 이 rev2는 claims-auditor 재감사에서 **`NO-GO`**를 받았다
> (`VERDICT_lambda0_rev2_2026-09-13.md`). D1–D15 중 **13건은 코드·데이터로 닫힌 것이
> 확인**됐으나, 死因은 **rev2가 D2를 고치면서 새로 도입한 추정량**에 있다:
> `achieved/realized_offered = (N/(N−1))·span/(span+L)`는 **항등식**이라 비포화 셀이
> 1이 아니라 **κ**를 읽고, shape A(out 512)의 고정 170 s 창에서 κ = 0.91–0.95로
> **저측 문턱 0.95 아래**에 앉는다(반전 2행: ①판정량 영점 미등록 ②fallback 분기 재량).
> **미실행**(GPU 0)이며 어떤 셀도 돌지 않았다.
> **현행 정본은 `PREREG_LAMBDA0_REV3_2026-09-13.md`.** 이 파일은 감사 대상 원본 보존을
> 위해 남겨 둔다.
>
> ★인용 금지(판정서 §7): **L2** — 이 문서 §4.2의 "청정 비포화 셀 3/3이 1.00±0.02로
> 수렴"은 **(8192 in, 96 out) 한정 사실**이다. **L3** — §5.2(i)의 "실현 분모에서 d44가
> 포화 셀 2개"는 **배수 인공물** 위에 서 있다(0.8988 측정 vs 0.9295 이상값). §3.2의
> 사각지대 정리는 **철회됐다**(rev3 D19).

# 사전등록 rev2 — 캠페인 **0단계**: λ\*(용량) 측정, shape별 × 기준 arm B1

**작성** 2026-09-13 (engine-porter) · **상태: 미실행, GPU 지출 0** · **제출 전**
**rev1**(`PREREG_LAMBDA0_2026-09-13.md`)은 claims-auditor 규칙층 감사에서 **`NO-GO`**
(`VERDICT_lambda0_rules_2026-09-13.md`) — 死因 **N2**(미등록 client `--seed` + 미등록
shape A 사다리) · **N3**(W4 "근사" 예보의 정의역 `∅`) + 제출 차단 운영결함 3건
(`UNRESOLVED` 도달 불가 · `shape` KeyError · sbatch/analyzer 부존재).
**이 rev2는 판정서 §4의 제출 전 필수 조건 D1–D15를 전부 반영한다.** rev1은
SUPERSEDED 표시만 하고 삭제하지 않는다(사전등록 이력의 감사 가능성).

사용자 결정 3건이 선행했다: 모델 = **`nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base`**,
백엔드 = **flashinfer**(NemotronH에서 triton은 엔진이 거부, `server_args.py:1959`),
1차 범위 = **Claim D + P3**.

---

## 0. 왜 0단계인가 — 그리고 ★이 단계가 닫지 못하는 것

캠페인 워크로드는 전부 λ\*의 분수로 정의된다(W1 0.60 · W3 0.80 · W8 0.90 · W9 1.10 ·
W2/W4는 phase당 요청 수 = `round(0.80·λ*·intensity·30)`). 그런데
`scripts/r2_eval/generate_campaign.sh:17`(HEAD `bddff6a`)의 `sustainable_rate`는 **측정값이 아니라 기본값 `4`**다.

> ★**인용 시점 고정(2026-09-13, 이 rev2 작성 중 확인)**: 위 `generate_campaign.sh:18`과
> §8 Q3의 같은 인용은 **커밋 `bddff6a`(HEAD) 기준**이다. 같은 날 **병행 워크스트림**이
> 작업트리에서 이 파일을 이미 고쳐 놓았다(미커밋): 스칼라 기본값은 제거되고 **shape별
> λ\* 표**(`benchmarks/pdmux_eval/lambda_star.py`, 필수 env `PDMUX_LAMBDA_STAR_TABLE`)가
> 들어갔으며, W4는 phase별로 **자기 shape의** λ\*를 쓴다. 따라서 `:18`은 **작업트리에서는
> 이미 이동했다** — 인용은 HEAD 사실로 읽되 현재 코드 상태로 읽지 말 것(교훈 80·게이트 #110).
> ★반면 D1이 근거로 드는 `workloads.py:159-160`은 **작업트리에서도 여전히 정확**하다
> (`8192 if is_prefill else 256` / `64 if is_prefill else 512`) — 재확인함.

★★**그러나 이 단계는 어떤 결과가 나와도 방법론 게이트 #6("용량 먼저 측정")을 닫지
못한다.** 게이트 #6이 요구하는 용량은 **지표 절벽 대비** 용량이고, 이 단계가 재는 것은
throughput 포화다. 판정서가 job 905835 d44 원자료를 정본 goodput 술어(TTFT ≤ 3000 ms ∧
요청-내부 token-ITL p95 ≤ 60 ms, `benchmarks/pdmux_eval/campaign.py:139-140` @HEAD `bddff6a`)로 직접 재채점한 결과:
**0.59·λ\*_thr에서 goodput 53.8% · 0.89·λ\*_thr에서 5.8% · 1.27·λ\*_thr에서 0.8%**
⇒ **λ\*_SLO(B) < 0.59 × λ\*_throughput**(비 추정 2.3–3.4×). 즉 SLO 절벽은 이 단계가
재는 수보다 **아래**에 있다. 이 문장은 §8 Q1으로 문자 고정한다.

---

## 1. λ\*의 정의 · ★D14 정본 정의 교체의 등재

> **λ\*(shape) = 기준 arm B1(legacy 루프, PD-mux fixed D44)에서 그 shape의
> open-loop(Poisson) throughput 포화율.** 워크로드의 분수는 "**B1 용량의 n%**"를 뜻한다.

**★D14 — 정본 정의 교체를 정정으로 등재한다.** `reports/paper/EXPERIMENT_ROADMAP.md:669`
(감사 시점 `:652`, 이후 배너 추가로 줄 이동)는 λ\*를 "먼저 B1 sustainable **SLO** rate
`lambda*`를 모델별로 측정한다"로 정의한다. **이 등록은 그것을 의도적으로 throughput
포화로 교체한다.** 두 양의 실측 비는 shape B에서 **≥1.7×**(위 goodput 재채점, 하한),
추정 2.3–3.4×. **λ\*_SLO는 이 단계 이후에도 여전히 측정되지 않는다.** 교체를 숨기면
정본 충돌이므로 여기 명시하고, 결과 문서는 §9-2를 병기한다.

**왜 arm을 고정하는가**: λ\*는 arm마다 다르다 — 같은 모델·같은 shape에서 **D16 0.933 /
D44 0.675 / D92 0.187 req/s(5×)**가 이미 측정돼 있다(job 905835). B1은 B4(true_dual
fixed)가 비교당하는 그 베이스라인이다. ★그러나 **워크로드 이름은 arm을 넘어 거짓이
된다** — "W9 = overload"는 B1 한정 서술이고 B4에서 같은 trace는 `0.2·λ*(B4)`일 수 있다
(§8 Q4).

---

## 2. 측정 격자 · 통제 요인

### 2.1 ONE BOOT PER CELL

★셀마다 별 boot·별 telemetry 파일이다. 한 boot 안에서 rate ladder를 돌리지 않는다 —
하나의 telemetry를 시간 창으로 잘라 귀속하는 기계를 다시 들이면 이 트랙이 반복해 죽은
지점(`CONSENSUS §3 항목120`)으로 돌아간다.

★**인용 정정(판정서 §3-2)**: rev1이 "probe C `c_capacity.sbatch:30-35`에서 **문자
승계**"라 적은 그 줄은 **주석**이다. **기제**는 `c_capacity.sbatch:111`의
`for SPEC in "${CELLS[@]}"` 루프 + `:123-178`의 셀별 launch/kill/`sleep 10`/`pkill`이며,
`lambda0.sbatch`가 승계하는 것은 **그 기제**다(주석이 아니다).

★**더 강한 차단이 이미 성립한다**: 판정량 `achieved = completed/duration`과 realized
offered는 **전부 client-side**이고 telemetry를 한 번도 경유하지 않는다. 항목120의 死因
(스냅샷 상태를 표본 간격 전체로 부과)은 구조적으로 발생할 수 없다.

### 2.2 shape

| shape | 정체 | 앵커 |
|---|---|---|
| **A = (256 in, 512 out)** | **W3 전부**이자 **W4 decode phase 그 자체** | 없음(§3) |
| **B = (8192 in, 64 out)** | **W4 prefill phase** | job 905835 D44 @ 8192-in/96-out = **0.675 req/s**(provisional) |

★**D1 (N3 해소)** — rev1의 "W4 근사"와 "5% 이내" 예보는 **전부 삭제**한다.
`benchmarks/pdmux_eval/workloads.py:159-160`을 실행해 확인한 사실: W4 prefill phase = (in 8192, out 64),
**W4 decode phase = (in 256, out 512)** = shape A **그 자체이며 동시에 W3 전부**다.
**근사는 존재하지 않는다.** 저장소 전체에 (in 64, out 512)인 워크로드는 없다 — rev1의
오독 출처는 `WorkloadSpec.output_distribution = "64/512 by phase"`(phase별 **출력**
길이)를 입력 shape로 읽은 것이고, 같은 오독이 `EXPERIMENT_ROADMAP.md`·과제 지시문에도
승계돼 있었다(2026-09-13(3)에 정본 정정 등재 완료). **미측정으로 남는 shape는 W1/W5/
W6/W7/W8/W9의 shape들이다.**

### 2.3 통제 요인 (캠페인 러너와 일치)

`--attention-backend flashinfer` · `--context-length 16384` · `--mem-fraction-static 0.82` ·
`--max-running-requests 48` · `--disable-radix-cache` · `--chunked-prefill-size -1` ·
`--disable-overlap-schedule` · cudagraph **ON** · `--random-seed 1` · PD-mux `fixed D44`
(legacy 루프, `PDMUX_R2_POLICY=fixed`) · TP=1 · config `pdmux_homog5.yml`.

### 2.4 ★D11 — probe C(905835)와의 차이를 **전부** 등록

| 항목 | probe C (905835) | 이 등록 | 캠페인 러너 | 처리 |
|---|---|---|---|---|
| mem-fraction | 0.80 | **0.82** | `${PDMUX_MEM_FRACTION:-0.82}` | 캠페인에 맞춤 |
| ctx | 8704 | **16384** | `${PDMUX_CONTEXT_LENGTH:-16384}` | 캠페인에 맞춤 |
| **`--disable-piecewise-cuda-graph`** | **ON**(`c_capacity.sbatch:126`) | ★**붙이지 않는다** | cudagraph-OFF 분기에서만 붙는다(`engine_bench_runner.sh:99`) | **캠페인에 맞춤** — 이 단계의 소비자는 캠페인이므로 운영점(cudagraph-ON) 구성과 같아야 한다. probe C와의 이 차이가 8192-token prefill 처리율을 바꿀 수 있음을 등록한다(`--chunked-prefill-size -1`이라 영향이 작을 가능성이 높으나 검증되지 않았다). |
| **`--random-seed`** | 미지정(랜덤) | **1** | run record의 `server_seed` | 등록 |
| **부하 생성기** | `bench_serving` Poisson | `bench_serving` Poisson | **`pdmux_eval.trace_loadgen`** — `/generate`에 `input_ids=[1]*n`, 고정 도착시각 재생 | ★**해소되지 않는 불일치로 등록** — λ\*는 **다른 하네스에서 측정된 상수로 다른 하네스를 파라미터화한다**(§9-6). 프롬프트 내용·토크나이즈·도착 분포가 모두 다르다. |
| pdmux config | `pdmux_homog5.yml` | `pdmux_homog5.yml` | `pdmux_r2.yml` | ★미확인 차이로 등록(실행 전 확인 항목) |
| max-running | 48 | 48 | 48 | ✓ |

**ctx 16384 / mem 0.82가 용량을 바꾸는가 — 아니다**(판정서 §3-5): 구속 자원은
`max_mamba_cache_size = max_running_requests = 48`이고 KV는 **6.6× 과공급**
(`max_total_num_tokens = 2,618,868` vs 필요 48×8256 = 396K). 앵커 0.675의 provisional
강등은 충분하다. ★단 이 추론의 유일한 미측정 전제(ctx 16384/mem 0.82에서도 48 유지)는
`lambda0.sbatch`가 셀마다 배너를 기록해 **확인 항목**으로 만든다.

### 2.5 ★D10 — 하네스 줄 (누락 4개 복원)

```
python -m sglang.bench_serving --backend sglang --model "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --dataset-name random --dataset-path "$SGPT_RAW" --tokenize-prompt \
  --random-input-len <256|8192> --random-output-len <512|64> --random-range-ratio 1.0 \
  --num-prompts "$NP" --request-rate "$RATE" --seed "$CSEED" \
  --output-details --output-file "$OUT/bench_${NAME}.jsonl"
```

- `--dataset-path` + `--tokenize-prompt` 없이는 입력 길이가 텍스트 재토크나이즈에
  흔들린다. probe C는 이 둘을 썼고 그래서 오차 0이었다(`bench_d44_r2_o96.log`:
  Total input tokens **1064960** = 130 × 8192).
- ★**검증 항목은 출력 길이만이 아니라 입력 길이도**다. `lambda0_analyze.py`가 셀마다
  `input_len_exact` / `output_len_exact`를 기록한다(보관 아티팩트 6셀로 확인: 6/6 True).
- `--output-details --output-file` 없이는 per-request `ttfts`/`itls`가 없어 후속
  재채점이 불가능하다.
- warmup: 8 요청 순차(`--max-concurrency 1 --seed 7`), 측정 제외(probe C `:139-145`
  승계). 비용은 shape A에서 **53.5 s/셀**(probe C out=96은 17.2 s) — D15 예산에 반영.

### 2.6 ★D12 — provenance를 GPU 전에 확인

`lambda0.sbatch`는 **첫 boot 전에** `sync_engine_tree.sh`로 manifest를 만들고,
(a) 항목 수 ≥ **24**, (b) **`models/nemotron_h.py` 포함**을 확인하지 못하면 `exit 3`으로
중단한다. 근거: job 905835는 manifest **17항목**(models/에 zamba2·mamba2뿐)인 채로
NemotronH를 12 boot 서빙했다 — **모델 구현 provenance 0**으로 돌았다(커밋 `87213a9`가
17→24항목으로 해소).

---

## 3. ★D3/D6 — 사다리는 λ_inf의 **결정적 함수**다 (자유 표면 0)

### 3.1 입력: 워크스트림 A가 사는 λ_inf

워크스트림 A(새 모델 correctness 게이트)가 같은 시점에 `--request-rate inf
--max-concurrency 64` **2셀**로 shape A·B의 **closed-loop 포화 처리율 λ_inf(A), λ_inf(B)**
를 측정한다. `bench_serving.py:943-945`가 `request_rate == inf`에서 sleep을 전부
건너뛰므로 **Poisson 실현이 존재하지 않는다** ⇒ 이 앵커 자신은 N2-S1의 영향을 받지 않는다.

★**λ_inf는 λ\*의 대체가 아니라 상한 프로브다.** 판정서 §8의 자기 실현가능성 검사가
등록한 한계를 그대로 승계한다: closed-loop(concurrency 64)는 open-loop Poisson보다
admitted가 더 찰 수 있어 **λ_inf ≥ λ\*_Poisson이 기대되나 보장되지는 않는다.**
λ_inf는 **사다리를 놓는 데만** 쓰고, 보고되는 λ\*는 언제나 이 단계의 Poisson 셀에서 읽는다.
(부수 확인 항목: A의 `Concurrency:` 보고값이 엔진 상한 48 근처에 붙는지 — 클라이언트
GIL 병목[confound #7]이 아님을 보이기 위해.)

### 3.2 등록 공식 (`lambda0_plan.py`, 실행 전 고정)

```
ladder(shape)   x_k      = round3( MULT[shape][k] · λ_inf(shape) )
                MULT[A]  = (0.40, 0.70, 1.15, 1.45, 2.00)     # 5 rung, 최소 단차 1.261×
                MULT[B]  = (0.45, 0.70, 1.15, 1.70)           # 4 rung, 최소 단차 1.478×
N(shape,k)               = clamp( round10( min(x_k, λ_inf) · 170 s ), 40, 400 )
seed1, seed2             = SEED_CANDIDATES(1..5000) 중 max_cells |Ē(seed,N) − 1| 최소 2개
                           (동률 시 작은 seed)
x_real,k                 = x_k / Ē(seed, N_k)                 # 실현 도착률
브래킷 창(shape)          = [ 0.950 · min(x_real) , 0.900 · max(x_real) ]
```

**브래킷 창의 근거**(판정서 §1이 probe C 3 arm 전부에서 원자료로 검증한 닫힌 형태):
포화에서 `ach/off = λ*/offered`이므로
> **브래킷 성립 ⟺ λ\* ∈ [0.950 · x_min , 0.900 · x_max]**

역검증(판정서): d16 `[0.532,1.098]` ∋ 0.933 ✓ · d44 `[0.380,0.774]` ∋ 0.675 ✓ ·
d92 `[0.105,0.217]` ∋ 0.1867 ✓.

**등록 도달가능성 주장(scenario-free)**: 위 곱수 집합에서 창은
**A: λ\*/λ_inf ∈ [0.380, 1.800]** · **B: [0.428, 1.530]** 이다 ⇒ **λ\*/λ_inf가
[0.45, 1.45] 안에 있는 한 브래킷은 구성상 성립한다.** 5개 등록 시나리오 전부에서
`lambda0_plan.py --selftest`가 이것을 assert한다.

**사각지대 정리(D8 보강)**: 한 rung이 `(0.90, 0.95)`에 떨어지는 것은
`x/λ* ∈ (1.0526, 1.1111)` — 곱셈 폭 **5.6%**의 창이다. 등록된 최소 rung 단차는
A 1.261× · B 1.478×로 그보다 훨씬 넓으므로 **동시에 두 rung이 사각지대에 들어갈 수
없고**, 한 rung이 들어가도 그 아래 rung은 `≤ x/1.261 < 0.881·λ*` ⇒ 비율 ≈ 1 (≥0.95),
위 rung은 `≥ 1.261·1.0526·λ* = 1.327·λ*` ⇒ 비율 ≤ 0.754 (≤0.90) ⇒ **브래킷은 그대로
성립한다.** (실제로 등록 시나리오 중 "B=6 plateau variant"에서 `a_r1`이 λ\*=2.5 가정
아래 0.9117로 사각지대에 떨어지는데, 판정은 바뀌지 않는다.)

### 3.3 ★λ\*(A)의 사전 추정 (실행 전 공시, 예보 아님)

앵커가 없는 shape A에 대해 판정서 §3-3이 엔진 로그에서 유도한 값을 그대로 승계한다.
`srv_d44_*.log`의 decode step 곡선 `step = 1000·B/G`(B=2..7 회귀 `19.82 + 0.5174·B` ms),
구속 자원 `max_mamba_cache_size = 48`, `λ*(A) = 48 / (512 · step(48))`:

| step(48) 가정 | 근거 | λ\*(A) |
|---|---|---|
| 13.06 ms | 측정된 절대 최소 step(B=1) | **7.15**(절대 상한) |
| 24 ms | B=6 실측이 48까지 평탄 | 3.91 |
| **37.4 ms** | 회귀 기울기에서 KV 항을 ctx 768로 축소 | **2.51** |
| **44.65 ms** | 회귀를 B=48로 직선 연장(ctx 8192) | **2.10** |
| 87 ms | 관측된 최악 step(D16 @ B=30) | **1.08**(절대 하한) |

★**점추정 λ\*(A) ≈ 2.1–2.5 req/s · 절대 경계 [1.08, 7.15]**. 이는 **B=7→48의 6.9배
외삽**이며 예보가 아니다 — 그래서 사다리를 이 추정이 아니라 **측정된 λ_inf**에 건다.

### 3.4 ★전 시나리오 사전 인쇄 (게이트 #197 — "중간점은 예보가 아니라 자유 표면")

아래는 `python3 lambda0_plan.py --registered-scenarios`의 **출력 전문**이다. 5개
λ_inf 시나리오(점추정 하/상, 절대 하한/상한, B=6 plateau)와 **감사 권고 literal
사다리(fallback)** 전부에 대해 사다리·N·Ē·실현 rate·브래킷 창·사각지대 rung·F4 반복
셀·예산이 **실행 전에** 고정돼 있다. λ_inf가 이 범위 어디에 떨어져도 돌아갈 계획은 이미
종이 위에 있다.

```text
### lambda_inf-anchored: point estimate lower  (lambda*(A)=2.1 regression extrapolation)
seed1=1376 (max|Ebar-1|=0.0176)  seed2=2083 (max|Ebar-1|=0.0203)  boots=11
  shape A (256, 512)  lambda_inf=2.1
    cell     offered_nom   N   Ebar    offered_real  est_dur_s
    a_r0           0.84   140  1.0129       0.8293      168.8
    a_r1           1.47   250  1.0037        1.465      170.7
    a_r2           2.42   360  1.0075        2.402      171.4
    a_r3           3.04   360  1.0075        3.017      171.4
    a_r4            4.2   360  1.0075        4.169      171.4
    window(realized)     [0.7878, 3.752]
    window(nominal)      [0.798, 3.78]
    window(conservative) [0.798, 3.752]  = lambda*/lambda_inf in [0.38, 1.787]  covers registered band (0.45, 1.45): True
    prior point (2.1, 2.5) inside: True   prior absolute (1.08, 7.15) inside: False   (reported, NOT asserted)
    expected blind-spot rungs (0.90<r<0.95): none
    F4 repeat cell: a_r4_s2 (Ebar@seed2=0.9944)
  shape B (8192, 64)  lambda_inf=0.675
    cell     offered_nom   N   Ebar    offered_real  est_dur_s
    b_r0          0.304    50  1.0175       0.2988      167.4
    b_r1          0.472    80  1.0136       0.4656      171.8
    b_r2          0.776   110  1.0176       0.7626      163.0
    b_r3           1.15   110  1.0176         1.13      163.0
    window(realized)     [0.2838, 1.017]
    window(nominal)      [0.2888, 1.035]
    window(conservative) [0.2888, 1.017]  = lambda*/lambda_inf in [0.4279, 1.507]  covers registered band (0.45, 1.45): True
    prior point (0.675, 0.675) inside: True   prior absolute (0.35, 1.2) inside: False   (reported, NOT asserted)
    expected blind-spot rungs (0.90<r<0.95): none
    F4 repeat cell: b_r3_s2 (Ebar@seed2=1.0048)
  BUDGET bench 30.9 min + warmup 6.5 min + boot/teardown 20.2 min = 0.959 GPU-h  (worst case with the one registered redesign round: 1.919)

### lambda_inf-anchored: point estimate upper  (lambda*(A)=2.5 ctx-768 KV-term variant)
seed1=3539 (max|Ebar-1|=0.0112)  seed2=1671 (max|Ebar-1|=0.0133)  boots=11
  shape A (256, 512)  lambda_inf=2.51
    cell     offered_nom   N   Ebar    offered_real  est_dur_s
    a_r0              1   170  1.0000            1      170.0
    a_r1           1.76   300  1.0112         1.74      172.4
    a_r2           2.89   400  0.9955        2.903      159.4
    a_r3           3.64   400  0.9955        3.656      159.4
    a_r4           5.02   400  0.9955        5.043      159.4
    window(realized)     [0.95, 4.538]
    window(nominal)      [0.95, 4.518]
    window(conservative) [0.95, 4.518]  = lambda*/lambda_inf in [0.3785, 1.8]  covers registered band (0.45, 1.45): True
    prior point (2.1, 2.5) inside: True   prior absolute (1.08, 7.15) inside: False   (reported, NOT asserted)
    expected blind-spot rungs (0.90<r<0.95): none
    F4 repeat cell: a_r4_s2 (Ebar@seed2=1.0054)
  shape B (8192, 64)  lambda_inf=0.8
    cell     offered_nom   N   Ebar    offered_real  est_dur_s
    b_r0           0.36    60  0.9986       0.3605      166.4
    b_r1           0.56   100  0.9934       0.5637      177.4
    b_r2           0.92   140  0.9902       0.9291      175.0
    b_r3           1.36   140  0.9902        1.374      175.0
    window(realized)     [0.3425, 1.236]
    window(nominal)      [0.342, 1.224]
    window(conservative) [0.3425, 1.224]  = lambda*/lambda_inf in [0.4281, 1.53]  covers registered band (0.45, 1.45): True
    prior point (0.675, 0.675) inside: True   prior absolute (0.35, 1.2) inside: True   (reported, NOT asserted)
    expected blind-spot rungs (0.90<r<0.95): none
    F4 repeat cell: b_r3_s2 (Ebar@seed2=1.0107)
  BUDGET bench 30.8 min + warmup 6.5 min + boot/teardown 20.2 min = 0.958 GPU-h  (worst case with the one registered redesign round: 1.916)

### lambda_inf-anchored: absolute lower bound  (A=1.08 worst observed step, B=0.35)
seed1=1671 (max|Ebar-1|=0.0134)  seed2=4175 (max|Ebar-1|=0.0152)  boots=11
  shape A (256, 512)  lambda_inf=1.08
    cell     offered_nom   N   Ebar    offered_real  est_dur_s
    a_r0          0.432    70  0.9950       0.4342      161.2
    a_r1          0.756   130  0.9904       0.7634      170.3
    a_r2           1.24   180  0.9896        1.253      166.7
    a_r3           1.57   180  0.9896        1.586      166.7
    a_r4           2.16   180  0.9896        2.183      166.7
    window(realized)     [0.4125, 1.964]
    window(nominal)      [0.4104, 1.944]
    window(conservative) [0.4125, 1.944]  = lambda*/lambda_inf in [0.3819, 1.8]  covers registered band (0.45, 1.45): True
    prior point (2.1, 2.5) inside: False   prior absolute (1.08, 7.15) inside: False   (reported, NOT asserted)
    expected blind-spot rungs (0.90<r<0.95): none
    F4 repeat cell: a_r4_s2 (Ebar@seed2=1.0152)
  shape B (8192, 64)  lambda_inf=0.35
    cell     offered_nom   N   Ebar    offered_real  est_dur_s
    b_r0          0.158    40  1.0134       0.1559      256.6
    b_r1          0.245    40  1.0134       0.2418      165.5
    b_r2          0.402    60  1.0061       0.3996      171.4
    b_r3          0.595    60  1.0061       0.5914      171.4
    window(realized)     [0.1481, 0.5323]
    window(nominal)      [0.1501, 0.5355]
    window(conservative) [0.1501, 0.5323]  = lambda*/lambda_inf in [0.4289, 1.521]  covers registered band (0.45, 1.45): True
    prior point (0.675, 0.675) inside: False   prior absolute (0.35, 1.2) inside: False   (reported, NOT asserted)
    expected blind-spot rungs (0.90<r<0.95): none
    F4 repeat cell: b_r3_s2 (Ebar@seed2=0.9973)
  BUDGET bench 32.2 min + warmup 6.5 min + boot/teardown 20.2 min = 0.982 GPU-h  (worst case with the one registered redesign round: 1.964)

### lambda_inf-anchored: absolute upper bound  (A=7.15 step(B=1) floor, B=1.20)
seed1=4696 (max|Ebar-1|=0.0053)  seed2=2823 (max|Ebar-1|=0.0066)  boots=11
  shape A (256, 512)  lambda_inf=7.15
    cell     offered_nom   N   Ebar    offered_real  est_dur_s
    a_r0           2.86   400  0.9947        2.875      139.1
    a_r1              5   400  0.9947        5.027       79.6
    a_r2           8.22   400  0.9947        8.264       55.9
    a_r3           10.4   400  0.9947        10.46       55.9
    a_r4           14.3   400  0.9947        14.38       55.9
    window(realized)     [2.732, 12.94]
    window(nominal)      [2.717, 12.87]
    window(conservative) [2.732, 12.87]  = lambda*/lambda_inf in [0.382, 1.8]  covers registered band (0.45, 1.45): True
    prior point (2.1, 2.5) inside: False   prior absolute (1.08, 7.15) inside: False   (reported, NOT asserted)
    expected blind-spot rungs (0.90<r<0.95): none
    F4 repeat cell: a_r4_s2 (Ebar@seed2=1.0051)
  shape B (8192, 64)  lambda_inf=1.2
    cell     offered_nom   N   Ebar    offered_real  est_dur_s
    b_r0           0.54    90  0.9947       0.5429      165.8
    b_r1           0.84   140  1.0022       0.8382      167.0
    b_r2           1.38   200  1.0004        1.379      166.7
    b_r3           2.04   200  1.0004        2.039      166.7
    window(realized)     [0.5157, 1.835]
    window(nominal)      [0.513, 1.836]
    window(conservative) [0.5157, 1.835]  = lambda*/lambda_inf in [0.4298, 1.529]  covers registered band (0.45, 1.45): True
    prior point (0.675, 0.675) inside: True   prior absolute (0.35, 1.2) inside: False   (reported, NOT asserted)
    expected blind-spot rungs (0.90<r<0.95): none
    F4 repeat cell: b_r3_s2 (Ebar@seed2=0.9934)
  BUDGET bench 21.3 min + warmup 6.5 min + boot/teardown 20.2 min = 0.799 GPU-h  (worst case with the one registered redesign round: 1.598)

### lambda_inf-anchored: B=6 plateau variant   (A=3.91, B=1.00)
seed1=3669 (max|Ebar-1|=0.0091)  seed2=2823 (max|Ebar-1|=0.0123)  boots=11
  shape A (256, 512)  lambda_inf=3.91
    cell     offered_nom   N   Ebar    offered_real  est_dur_s
    a_r0           1.56   270  1.0044        1.553      173.8
    a_r1           2.74   400  0.9992        2.742      145.9
    a_r2            4.5   400  0.9992        4.504      102.3
    a_r3           5.67   400  0.9992        5.674      102.3
    a_r4           7.82   400  0.9992        7.826      102.3
    window(realized)     [1.476, 7.044]
    window(nominal)      [1.482, 7.038]
    window(conservative) [1.482, 7.038]  = lambda*/lambda_inf in [0.379, 1.8]  covers registered band (0.45, 1.45): True
    prior point (2.1, 2.5) inside: True   prior absolute (1.08, 7.15) inside: False   (reported, NOT asserted)
    expected blind-spot rungs (0.90<r<0.95): [('a_r1', 2.5, 0.9117)]
    F4 repeat cell: a_r4_s2 (Ebar@seed2=1.0051)
  shape B (8192, 64)  lambda_inf=1.0
    cell     offered_nom   N   Ebar    offered_real  est_dur_s
    b_r0           0.45    80  1.0012       0.4495      178.0
    b_r1            0.7   120  0.9909       0.7064      169.9
    b_r2           1.15   170  0.9994        1.151      170.0
    b_r3            1.7   170  0.9994        1.701      170.0
    window(realized)     [0.427, 1.531]
    window(nominal)      [0.4275, 1.53]
    window(conservative) [0.4275, 1.53]  = lambda*/lambda_inf in [0.4275, 1.53]  covers registered band (0.45, 1.45): True
    prior point (0.675, 0.675) inside: True   prior absolute (0.35, 1.2) inside: False   (reported, NOT asserted)
    expected blind-spot rungs (0.90<r<0.95): none
    F4 repeat cell: b_r3_s2 (Ebar@seed2=0.9994)
  BUDGET bench 26.4 min + warmup 6.5 min + boot/teardown 20.2 min = 0.885 GPU-h  (worst case with the one registered redesign round: 1.771)

### FALLBACK (audit-recommended literal ladders, used iff workstream A does not deliver lambda_inf)
seed1=1056 (max|Ebar-1|=0.0085)  seed2=2357 (max|Ebar-1|=0.0114)  boots=11
  shape A (256, 512)  lambda_inf=2.1
    cell     offered_nom   N   Ebar    offered_real  est_dur_s
    a_r0            1.1   190  1.0007        1.099      172.8
    a_r1            1.8   310  1.0011        1.798      172.4
    a_r2              3   360  0.9995        3.001      171.4
    a_r3            4.9   360  0.9995        4.902      171.4
    a_r4              8   360  0.9995        8.004      171.4
    window(realized)     [1.044, 7.203]
    window(nominal)      [1.045, 7.2]
    window(conservative) [1.045, 7.2]  = lambda*/lambda_inf in [0.4976, 3.429]  covers registered band (0.45, 1.45): False
    prior point (2.1, 2.5) inside: True   prior absolute (1.08, 7.15) inside: True   (reported, NOT asserted)
    expected blind-spot rungs (0.90<r<0.95): none
    F4 repeat cell: a_r4_s2 (Ebar@seed2=1.0075)
  shape B (8192, 64)  lambda_inf=0.675
    cell     offered_nom   N   Ebar    offered_real  est_dur_s
    b_r0           0.45    80  0.9938       0.4528      176.7
    b_r1           0.62   110  1.0085       0.6148      178.9
    b_r2           0.85   110  1.0085       0.8428      163.0
    b_r3           1.15   110  1.0085         1.14      163.0
    window(realized)     [0.4302, 1.026]
    window(nominal)      [0.4275, 1.035]
    window(conservative) [0.4302, 1.026]  = lambda*/lambda_inf in [0.6373, 1.52]  covers registered band (0.45, 1.45): False
    prior point (0.675, 0.675) inside: True   prior absolute (0.35, 1.2) inside: False   (reported, NOT asserted)
    expected blind-spot rungs (0.90<r<0.95): none
    F4 repeat cell: b_r3_s2 (Ebar@seed2=0.9886)
  BUDGET bench 31.3 min + warmup 6.5 min + boot/teardown 20.2 min = 0.966 GPU-h  (worst case with the one registered redesign round: 1.931)
```

★**fallback 등록(D6)**: 워크스트림 A가 λ_inf를 주지 못하면 **감사 권고 literal 사다리**
(A `{1.1, 1.8, 3.0, 4.9, 8.0}` · B `{0.45, 0.62, 0.85, 1.15}`)를 쓴다. 위 인쇄가 확인하듯
그 창은 **A `[1.045, 7.20]` · B `[0.4275, 1.035]`**로 판정서 수치와 정확히 일치한다.
fallback의 알려진 약점도 함께 등록한다: B의 창 상단 1.035는 **λ\*(B) 절대 상한 1.20을
덮지 못한다**(λ_inf 앵커 사다리는 덮는다). ⇒ fallback은 차선이며, λ_inf가 오면 쓰지 않는다.

★**D6 정정**: rev1은 shape B 상단을 `×1.2`로 쓰며 "probe C와 같은 배수 구조"라 했으나
probe C 실제 배수는 **`{0.60, 0.90, 1.30}`**이다(`c_capacity.sbatch:76`; 사다리 상단을 1.30으로 올린 근거는 `:78-79`. ★판정서는 이 배수 집합을 `:78-79`로 적었다 — 한 줄 어긋난 인용이다). probe C의
prereg §3.2는 `ρ=1.10`이 `1/1.10 = 0.909 > 0.90`이라 **산술적으로 도달 불가**임을 보이고
상단을 의도적으로 1.30으로 올렸다. rev1의 `×1.2`는 그 여유를 되돌린 것이었고, 등록 자신의
예보 방향("out 96→64이므로 λ\*가 약간 높을 것")이 곧 비브래킷 방향이었다. rev2의
`MULT[B]` 상단 1.70(창 상단 `0.90×1.70 = 1.53·λ_inf`)은 그 실패 형태를 구조적으로 없앤다.

### 3.5 재설계 회차 (D7의 절반)

`KNEE_NOT_BRACKETED`가 나오면 **전 단계 통틀어 1회**의 재설계 회차를 허용하며, 그 재설계
역시 **결정적**이다: `λ_inf ← 관측된 최대 achieved rate`로 갈아끼우고 같은 공식을 다시
돌린다(사람 판단 개입 0). 2회차 이상은 사용자 승인 사항이다.

---

## 4. ★D2 (N2-S1 해소) — client seed 등록 + 도착 실현 독립 판정량

### 4.1 등록해야 할 사실

`bench_serving.py:1705-1706`이 프로세스당 한 번 `random.seed(args.seed);
np.random.seed(args.seed)`를 걸고, `:943-950`이 비-inf rate에서 간격을
`np.random.exponential(1.0/request_rate)`로 뽑는다(yield 후 sleep ⇒ N개 요청에 N−1개 간격).

> ★**비포화 셀의 `achieved/offered`는 포화도가 아니라 도착 실현 계수 `1/Ē`다.**
> (Ē = N−1개 Exp(1) 표본평균.) 905835 실측: **12/12 셀 전부** 도착 구간이 명목보다 짧았고
> (Ē = 0.826–0.872, seed 41), 그래서 11/12 셀이 `ach/off > 1`(최대 1.1886)을 보고했다.
> seed는 프로세스당 한 번 고정되므로 **사다리 전체에 공통모드로 들어가고 셀을 늘려도
> 평균되지 않는다.** `1.96/√(N−1) ≤ 0.05`는 **N ≥ 1537**을 요구하므로, 명목 분모 위에서
> 0.95 문턱은 실현 가능한 N으로 신뢰 도달이 불가능하다. seed 40개 모의 저측 실패율
> **15–28%**(N=90·150·300).

### 4.2 선택: **(b) 분모를 실현 도착률로 교체** — 그리고 왜 (a)·(c)가 아닌가

판정서 §4-D2가 제시한 3안 중 **(b)**를 택한다.

```
realized_offered_rate = (num_prompts − 1) / Σ intervals
achieved_over_realized = achieved_rate / realized_offered_rate      ← ★판정량
```

**왜 (b)인가 — 판정서가 "구현 필요(송신 시각 기록)"라 본 비용이 실제로는 0이기 때문이다.**
`bench_serving.py` 전체에서 `np.random` 소비처는 **정확히 2곳**뿐이다(`:948` 간격,
`:1354` LoRA — LoRA 미사용). 따라서 간격 열은 `(seed, num_prompts, rate)`만으로
**오프라인·GPU 0으로 정확히 재현된다.** 하네스 수정도, 엔진 수정도 필요 없다.

**보관 원자료로 검증했다(GPU 0).** 905835 12셀에 이 보정을 적용:

| 셀 | 명목 `ach/off` | Ē(seed 41) | **실현 `ach/real`** |
|---|---|---|---|
| d16_r0 (저rate, 청정) | 1.1700 | 0.8496 | **0.9940** |
| d44_r0 (저rate, 청정) | 1.1422 | 0.8715 | **0.9955** |
| d92_r0 (저rate, 청정) | 1.1886 | 0.8560 | **1.0175** |
| d16_r2 (포화) | 0.7649 | 0.8524 | 0.6520 |
| d44_r2 (포화) | 0.7852 | 0.8487 | 0.6664 |
| d92_r2 (포화) | 0.7746 | 0.8270 | 0.6406 |

⇒ **인공물이 제거된다**(청정 비포화 셀 3/3이 1.00±0.02로 수렴). 그리고 ★**이 보정은
감사된 probe C 판정을 바꾸지 않는다**: 세 arm 전부 `KNEE_BRACKETED`, λ\* = **0.9331188685063582 /
0.675302644015995 / 0.18668462822130108** — `C_LABEL.json`과 **비트 단위 일치**
(`lambda0_label.py --selftest`가 보관 cell JSON을 직접 읽어 매번 재확인한다).

- **(a)를 택하지 않은 이유**: `--request-rate inf`는 **closed-loop** 포화율이라 이 단계가
  정의한 open-loop 양이 아니다. 판정서 자신이 §8에서 "λ_inf는 λ\*의 대체가 아니라 사다리
  양 끝을 고정하는 상한 프로브로만 등록하라"고 못 박았다. 우리는 (a)를 **버리지 않고**
  §3.1의 **앵커**로 쓴다 — 즉 (a)의 이득(死因 제거)은 사다리 배치에서 취하고, 판정량은
  (b)로 독립시킨다. 두 경로를 겹쳐 쓰는 것이 어느 한쪽보다 강하다.
- **(c)를 단독으로 택하지 않은 이유**: (c)는 명목 분모를 유지한 채 문턱을 실현 계수 분포로
  **교정**하는 안이라 문턱 자체가 확률적 객체가 된다. 대신 (c)의 **가드**는 그대로
  채택한다(§4.4).

### 4.3 seed는 정수로 등록된다 — 선택 규칙도 결정적이다

`Ē`는 **(seed, N)에만** 의존하고 rate와 무관하다(간격 = X/rate, X ~ Exp(1) ⇒ Ē = mean(X)).
따라서 seed는 사다리가 정해지는 즉시 결정된다:

> **seed1, seed2 = `SEED_CANDIDATES = 1..5000` 중 `max_cells |Ē(seed, N_cell) − 1|`를
> 최소화하는 두 seed(동률이면 작은 seed).**

이것은 엔진과 무관한 순수 RNG 성질 위의 선택이고, 규칙·후보집합·점수가 전부 사전등록되며
결과가 §3.4에 전부 인쇄돼 있다(예: λ_inf=(2.10, 0.675)이면 **seed1 = 1376, seed2 = 2083**,
최대 편차 0.0176 / 0.0203). **어떤 시나리오에서도 두 seed 모두 |Ē−1| ≤ 0.025를 만족한다**
(`lambda0_plan.py --selftest`가 assert). 이는 (c)가 요구한 밴드 `1/Ē ∈ [0.95, 1.05]`의
**절반 폭**이다.

### 4.4 ★R0 가드 (c안 승계, 살아 있는 분기)

`lambda0_analyze.py`가 셀마다 `ebar`·`inv_ebar`·`ebar_guard_ok`를 기록하고,
**`1/Ē ∉ [0.95, 1.05]`인 셀이 하나라도 있으면 그 shape는 `UNRESOLVED`**다. 선택된 seed는
이 밴드 안에 여유를 두고 들어가므로, 위반은 "그 셀이 등록된 계획으로 돌지 않았다"(잘못된
seed·num-prompts, 또는 `bench_serving`의 RNG 소비 구조 변경으로 오프라인 재현이 깨짐)는
뜻이며 **측정 실패**다(규칙 실패가 아니다, 교훈 21). 이 가드는 죽은 장식이 아니다 —
probe C 자신의 seed 41 셀들을 shipped CLI에 넣으면 실제로 `UNRESOLVED`가 나온다(확인함).

---

## 5. 판정 규칙 (데이터 생성 전 고정 — `lambda0_label.py`)

probe C의 이미 감사된 규칙(`c_capacity_label.py` R1/R5)을 실질 승계하고 키를 arm → shape로
바꾸었으며, 분모만 §4의 실현 도착률로 교체했다.

| # | 이름 | 규칙 | 성격 |
|---|---|---|---|
| **R0** | `ARRIVAL_REALIZATION[shape]` | 모든 셀이 `1/Ē ∈ [0.95, 1.05]` | 측정 유효성 가드 |
| **R1** | `KNEE_BRACKETED[shape]` | `ach/realized ≥ 0.95`인 셀이 있고, **그보다 offered가 높은** 셀에 `ach/realized ≤ 0.90`이 있다 | 존재, shape별 |
| **R2** | `LAMBDA_STAR[shape]` | 문턱 없는 **측정값** — 포화 셀(≤0.90) 중 **최대 achieved rate**. **R1이 BRACKETED일 때만 보고** | 측정 |
| **R3** | `SEED_REPEAT[shape]` | 제2 seed로 반복한 **최상단 rung**이 포화를 유지하고 λ\*를 **5% 이내**로 재현 | ★등록 예보 |
| — | 기대 셀 결손 | 그 shape는 **`UNRESOLVED`** — 측정 실패이며 규칙 실패가 아니다(교훈 21) | |

★**D4 (S9 해소)**: 규칙 코드는 **기대 셀 이름 목록을 인자로 받아 순회**한다 —
디렉터리를 glob하지 **않는다**. rev1은 glob이라 boot 실패 셀이 리스트에서 조용히 사라져
`any(c is None)`이 영원히 거짓이었고, `UNRESOLVED`가 **구조적으로 도달 불가**했다(boot
실패가 `KNEE_NOT_BRACKETED`로 인쇄됨). `lambda0_label.py --mutation-missing-cell`이 이
결함을 주문형으로 재현한다.

★**D7 (나머지 절반)**: **F4/R3는 shape별로 독립 적용**한다(각 shape가 자기 최상단 rung을
자기 제2 seed 셀로 확인한다). 합동이 아니다 — S1의 seed 공통모드가 두 shape를 동시에
뒤집을 수 있으므로 shape별 독립이 더 보수적이다. ★**반복 셀은 사전에 최상단 rung으로
등록**돼 있다(사후에 "잘 나온 점"을 고르는 것이 아니다). ★**반복 셀은 R1 사다리에 넣지
않는다** — 같은 offered rate의 셀 2개가 사다리에 들어가면 동률 쌍이 자기 자신을
브래킷할 수 있다(그래서 `hi[0] > lo[0]` 동률 가드가 load-bearing이고, selftest가 고정한다).

### 5.1 ★D8 — 출력공간 전수 (사각지대 칸 포함)

셀 하나의 비율 `r = ach/realized`는 아래 3구간 중 하나에 떨어진다. **`(0.90, 0.95)`는
저측도 고측도 아니다** — 이 칸이 rev1 §5에서 누락돼 있었다.

| 셀 비율 | 라벨 | R1에서의 역할 |
|---|---|---|
| `r ≥ 0.95` | 비포화(저측 자격) | 브래킷의 **아래**쪽이 될 수 있다 |
| `0.90 < r < 0.95` | ★**사각지대** | **어느 쪽도 아니다.** 셀은 유효하나 R1에 기여하지 않고, R2의 포화 집합에도 들어가지 않는다. `blind_spot_cells`에 열거해 보고한다 |
| `r ≤ 0.90` | 포화(고측 자격) | 브래킷의 **위**쪽이 될 수 있고 R2의 후보다 |

shape별 verdict 전수:

| verdict | 조건 | λ\* 보고 | R3 |
|---|---|---|---|
| `UNRESOLVED` | 기대 셀 결손 **또는** R0 가드 위반 | 없음 | 평가 안 함 |
| `KNEE_BRACKETED` | R1 성립 | **있음** = max(포화 셀 achieved) | `SEED_REPEAT_HOLDS` / `SEED_REPEAT_REFUTED` / `UNRESOLVED`(반복 셀 결손) |
| `KNEE_NOT_BRACKETED` | R1 불성립(전 셀 비포화 / 전 셀 포화 / 사각지대만) | **없음**(포화 셀이 있어도 보고하지 않는다) | 위와 같음 |

★**어떤 조합도 정책·arm 순위·Claim 등급을 움직이지 않는다.**

### 5.2 검증 (D9) — 변이 17종이 전부 막힌다

판정서 §3-6b는 rev1 selftest에서 변이 **11개 중 7개가 ESCAPE**했다고 보고했다.
rev2의 selftest는 보관 cell JSON을 **직접 읽어**(전사본 아님) 대조하고, 아래를 고정한다:

(i) 포화 셀 2개 이상 사다리에서 `max(saturated)` — 실현 분모에서 **d44가 실제로 포화 셀
2개**가 되므로 구별이 살아난다(+ 합성 쌍둥이 케이스로 아카이브 이동에도 견디게) ·
(ii) 비포화 셀 achieved가 포화 plateau보다 큰 붕괴형 사다리로 "포화 셀 한정" 고정 ·
(iii) 비단조 사다리(저rate 0.85 / 고rate 0.98) **및** 동률 rate 쌍이 브래킷 **아님** ·
(iv) `ach/off`가 정확히 0.95·0.90인 경계는 브래킷이고 0.94·0.91은 사각지대 ·
(v) R0 가드(명시 위반 + 필드 부재) · (vi) R2는 브래킷 없이는 보고되지 않음 ·
(vii) 셀 파일 삭제 → `UNRESOLVED`.

`lambda0_mutation_check.py`(동봉)가 M1–M17을 기계적으로 돌리며, **무변이 CONTROL이
PASSES인지 먼저 확인한다**(이게 없으면 "파일이 아예 안 돌아서 FAILS"를 성공으로 오독한다 —
이 하네스 초안이 실제로 그 함정에 빠졌고, M1·M12가 그 때문에 escape했다).

---

## 6. 예보와 출력공간 라벨 (F1–F4)

- **F1 (부팅)**: 전 셀 boot 성공. 근거 = job 905835가 같은 모델·백엔드로 12/12 boot
  (`BOOT_FAILED` 0). 실패 시 **측정 실패**로 기록하고 engine-porter 트리아지(ctx 16384·
  mem 0.82·`--disable-piecewise-cuda-graph` 부재는 905835와 다르므로 그쪽을 먼저 본다).
- **F2 (shape B)**: `KNEE_BRACKETED`. λ\*(B)는 앵커 0.675와 같을 필요 없다(out 96→64·
  ctx·mem·piecewise가 다르다). **수치 예보 없음, 브래킷 존재만** 예보한다.
  ★`KNEE_NOT_BRACKETED`일 때의 처분 = §3.5의 결정적 재설계 1회(shape A와 공유).
- **F3 (shape A)**: `KNEE_BRACKETED`. 앵커는 없지만 사다리가 λ_inf에 걸려 있으므로
  `λ*/λ_inf ∈ [0.45, 1.45]`인 한 구성상 성립한다(§3.2).
- **F4 = R3 (제2 seed)**: 각 shape의 **최상단 rung**을 제2 seed로 1회 반복했을 때
  (a) 포화를 유지하고 (b) λ\*를 **5% 이내**로 재현한다. 근거: 이미 용량에 도달한 인접 두
  셀의 achieved 차이가 d92 **0.16%**, d16 **1.67%**(판정서 §1-S5) ⇒ 5%는 도달 가능하며
  자기기각이 아니다. **반증되면** λ\*를 두 seed의 구간 `[min, max]`로 보고하고 5% 주장을
  `REFUTED`로 등재한다(측정 실패가 아니다).

---

## 7. 이 단계가 **주지 않는 것** · D13 · D15

- **성능·정책 판정 0건** — arm 비교 없음, goodput 평가 없음(λ\*는 throughput 포화다).
  Claim D/E 등급 불변 · 선결 #1–#5 불변 · HE0 · 정책 순위 · stake #1 불변.
- **새 모델의 correctness를 건드리지 않는다** — (Nano-9B-v2, flashinfer) 쌍의 R2
  correctness 게이트는 아직 한 번도 돌지 않았다. 이 단계와 905835는 **legacy 루프 + fixed
  split**만 발화시킨다. 907100·907456·X1의 결론은 **(Zamba2-2.7B, triton) 한정 동결**이고
  X1의 민감도 시연은 `triton_attention_num_kv_splits`가 flashinfer에 없으므로 이식되지 않는다.
- **단일 스칼라 λ\*를 W1–W9가 공유하는 설계 문제는 해소되지 않는다.** 오히려 이 단계는
  **그것이 불가능함을 증명한다**(§8 Q3): `generate_campaign.sh:18`은 9 워크로드에 하나의
  스칼라를 쓰고 `workloads.py:150`(@HEAD `bddff6a`)의 W4는 두 phase에 **같은** `per_phase`를 쓴다.

**★D13 — 상단 셀의 지연 규모와 하네스 상한(실행 전 확인 완료)**

- 등록 사다리의 상단 셀은 의도적 과포화이므로 TTFT p99가 **≈90–130 s**까지 간다
  (감사 권고 사다리에서 ≈126 s). probe C 최댓값은 **101.43 s**(`d92_r2_o96`)였으므로 그
  위는 **무검증 영역**이다.
- 사전 확인(코드 직접): `bench_serving.py:70` **`BENCH_AIOHTTP_TIMEOUT_SECONDS = 6 시간`**
  ⇒ 클라이언트 타임아웃은 구속하지 않는다. `max_queued_requests`는 기본 `None`이고
  `scheduler.py:2013-2014`가 `None`일 때 admission 제한을 **적용하지 않는다** ⇒ 503 거부도
  없다. 남는 상한은 **SLURM 벽시간**이며 `--time=03:30:00`으로 등록 예산의 3배 이상을 잡았다.
- ★그럼에도 **상단 셀의 TTFT/ITL 백분위는 인용 불가**다(§9-3).

**★D15 — 예산 정정**

- **요청 예산: 1.05–1.15 GPU-h(11 boot), 등록된 재설계 회차 포함 최악 ≈ 2.0 GPU-h.**
  (판정서 D15의 수치를 그대로 등록한다: 905835 실측 5.6분/boot × 11 + shape A warmup 증분.)
- **분해 모형의 예측치는 그보다 낮다: 0.80–0.98 GPU-h**(§3.4 인쇄의 BUDGET 줄, 5 시나리오
  + fallback). 분해 = 셀당 boot/teardown **108.8 s**(905835 잔차: 1.11 GPU-h − 보관 12셀
  `duration_s` 합 2394.1 s − warmup 모형 296.4 s, ÷12) + warmup(shape A 53.5 s / B 13.9 s)
  + 등록 사다리의 예상 bench 시간. 차이의 출처는 판정서의 평탄한 "5.6분/boot"가 probe C의
  **더 긴 bench 창**(평균 200 s vs 이 단계 168 s)을 함께 품고 있다는 것이다.
  ⇒ **요청은 큰 쪽(1.15), 예측은 작은 쪽(0.98)으로 등재**한다. rev1 헤드라인
  0.75–0.95 GPU-h는 **warmup 누락**(shape A에서 3배 비쌈)으로 낮았다.
- 이 트랙 GPU 장부: R2 correctness 0.43 GPU-h(별개 장부) — 이 단계는 아직 **0**.

---

## 8. ★인용 금지 Q1–Q5 (판정서 §6 문안 승계, 문자 그대로)

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
> `generate_campaign.sh:18`은 9 워크로드에 단일 스칼라를 쓰고 `workloads.py:150`(@HEAD `bddff6a`)의 W4는 두
> phase에 **같은** `per_phase`를 쓴다. 실측/추정 λ\*로 계산하면: λ\*=0.675 주입 →
> prefill phase 0.79×λ\*(B) ✓ / decode phase **0.21–0.25×λ\*(A)** ✗; λ\*=2.1 주입 →
> decode 0.79×λ\*(A) ✓ / prefill **2.47×λ\*(B)** ✗; 기본값 4 → prefill **4.7×λ\*(B)**.
> **어떤 단일 스칼라도 두 phase를 동시에 0.80으로 만들 수 없다.** W4를 Claim D의 P2에서
> 쓰려면 워크로드 정의 자체(phase별 독립 λ\*)를 고쳐야 하며, 그것은 이 단계가 주지 않는다.

★**Q3 편집 주석(인용문 자체는 불변)**: Q3의 `generate_campaign.sh:18` 인용은 HEAD
`bddff6a` 기준이다(§0 인용 시점 고정 참조). 같은 날 병행 워크스트림이 **워크로드 정의
쪽을** phase별 독립 λ\*로 고쳤으므로 Q3의 마지막 문장("그것은 이 단계가 주지 않는다")은
**여전히 참**이다 — 이 단계는 **두 수**를 줄 뿐이고, 그 수를 phase별로 쓰게 만드는 코드
변경은 다른 워크스트림의 산출이다. **이 단계가 그 변경을 자기 공로로 인용하는 것을 금지한다.**

> **Q4** — **"B1의 용량이 시스템의 용량"이 아니다.** 워크로드 이름(W8 "near saturation",
> W9 "overload")은 **B1에서만** 참인 서술이고, B4에서 같은 trace는 자기 용량의 0.2×일
> 수도 5×일 수도 있다. 워크로드 이름을 regime 서술로 인용하는 것을 금지한다.

> **Q5** — **shape A·B 두 shape를 쟀다는 것이 W1·W5·W6·W7·W8·W9의 부하 라벨을 고치지
> 않는다.** 그 워크로드들의 shape는 미측정으로 남으며, λ\*는 shape 의존적이다(이 단계
> 자신이 두 shape에서 다른 값을 낸다).

---

## 9. ★실행 후 필수 병기 6항 (판정서 §5 문안 승계 — 어떤 라벨이 나오든 무조건)

1. **λ\*는 B1(legacy, fixed D44) 한정 throughput 포화율이다.** 905835 실측으로 split을
   바꾸면 같은 모델·같은 shape에서 **5× 변한다**(D16 0.933 / D44 0.675 / D92 0.187 req/s).
   B4의 λ\*는 측정되지 않았다.
2. **정본 술어 goodput은 이 단계에서 측정되지 않았다.** 이 단계는 TTFT·ITL SLO를 쓰지 않는다.
3. **상단 셀의 TTFT/ITL 백분위는 인용 불가다** — 의도적 과포화 셀이므로 지연 수치가 아니라
   처리율 plateau를 읽기 위한 것이다.
4. `switch_count`·split 체류분포는 이 단계의 판정에 쓰이지 않았다(fixed split, 동적 제어 없음).
5. **포화 셀의 achieved는 도착 실현에 무관하나(λ\*는 robust), 비포화 셀의 `ach/off`는 도착
   실현 계수다.** 905835에서 1.03–1.19, 12/12 셀 동일 방향. (rev2는 이 때문에 분모를
   실현 도착률로 교체했다 — §4.)
6. 이 단계가 쓴 부하 생성기는 `sglang.bench_serving`(Poisson)이고 캠페인은
   `pdmux_eval.trace_loadgen`(고정 도착시각, `input_ids=[1]*n`)이다. **λ\*는 다른
   하네스에서 측정된 상수로 캠페인을 파라미터화한다.**

---

## 10. 동봉 실행체 (D5) 와 재현 절차

| 파일 | 역할 |
|---|---|
| `lambda0_plan.py` | 사다리·N·seed·창·예산의 **결정적** 생성기. `--registered-scenarios`가 §3.4 전문을 인쇄, `--selftest` |
| `lambda0_analyze.py` | **per-cell analyzer**(신규). `shape` 필드 기록(rev1 `KeyError` 해소), 실현 도착률·Ē·가드·입출력 길이 검증·백분위 |
| `lambda0_label.py` | 판정 규칙 R0–R3. **기대 셀 목록을 인자로** 순회(D4), `--selftest`, `--mutation-missing-cell` |
| `lambda0_mutation_check.py` | 변이 M1–M17 + 무변이 CONTROL |
| `lambda0.sbatch` | **제출 실행체**(신규). ONE BOOT PER CELL 기제 승계, 절대경로 고정, `--comment` directive, D12 provenance 게이트, 포트 충돌 회피 |

**하류 인터페이스(2026-09-13 확인)**: 병행 워크스트림이 캠페인 생성기에 **shape별 λ\* 표**
(`benchmarks/pdmux_eval/lambda_star.py`, schema `pdmux.lambda_star/v1`)를 도입했고 그
`definition` 필드가 **`throughput_saturation`** / `slo_sustainable` 두 값을 구분한다.
이 단계의 산출 `LAMBDA0_LABEL.json`은 **`throughput_saturation`** 쪽에만 들어간다
(shape key `256x512` / `8192x64`, `source`를 `unmeasured` → `measured`로, `evidence`에
job id + 이 사전등록 경로). ★`slo_sustainable`은 이 단계 이후에도 **비어 있다**(§8 Q1).

```bash
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc
P=workspace/engine-port/results/r2_eval/lambda0_prereg
python3 $P/lambda0_plan.py  --selftest
python3 $P/lambda0_label.py --selftest
python3 $P/lambda0_mutation_check.py            # 0 escapes required
python3 $P/lambda0_plan.py --registered-scenarios

# 제출(워크스트림 A의 λ_inf가 나온 뒤):
PDMUX_LAMBDA_INF_A=<req/s> PDMUX_LAMBDA_INF_B=<req/s> sbatch $P/lambda0.sbatch
# 또는 등록된 fallback:
PDMUX_LAMBDA0_FALLBACK=1 sbatch $P/lambda0.sbatch
```

CPU 회귀는 `workspace/engine-port/tests/test_lambda0_prereg.py`가 위 전부(+ sbatch 규율,
+ rev1 glob 결함의 **음성대조**)를 `python -m unittest discover -s workspace/engine-port/tests`
안에서 고정한다.

---

## 10.1 ★인용 고정표 (교훈 80 · 게이트 #110)

이 문서의 모든 코드 인용을 **작성 시점에 한 줄씩 열어** 확인했다. ★**같은 날 병행
워크스트림이 캠페인 생성기·워크로드 정의·`r2_correctness.sbatch`를 작업트리에서 고치고
있으므로**(미커밋), 캠페인 쪽 인용은 **커밋 `bddff6a`(HEAD)에 고정**한다. "worktree도 유효"
열이 `아니오`인 인용을 현재 코드로 읽으면 틀린다.

| 인용 | 확인된 내용 | 고정 트리 | worktree도 유효 |
|---|---|---|---|
| `benchmarks/pdmux_eval/workloads.py:159-160` | `8192 if is_prefill else 256` / `64 if is_prefill else 512` | HEAD `bddff6a` | **예**(병행 워크스트림이 의도적으로 보존) |
| `benchmarks/pdmux_eval/workloads.py:150` | `per_phase = max(1, round(0.80 * sustainable_rate * intensity * 30.0))` | HEAD `bddff6a` | 아니오(`_phase_count(...)`로 교체) |
| `benchmarks/pdmux_eval/workloads.py:156` | `start + 30.0 * rng.random()` | HEAD `bddff6a` | 아니오(:158로 이동) |
| `benchmarks/pdmux_eval/campaign.py:139-140` | `ttft_slo_ms=3000.0` / `itl_slo_ms=60.0` | HEAD `bddff6a` | 아니오 |
| `scripts/r2_eval/generate_campaign.sh:17` | `sustainable_rate="${PDMUX_SUSTAINABLE_RATE:-4}"` | HEAD `bddff6a` | 아니오(스칼라 제거, shape별 표로 교체) |
| `scripts/r2_eval/engine_bench_runner.sh:99` | `server_args+=(--disable-cuda-graph --disable-piecewise-cuda-graph)` | HEAD `bddff6a` | 예 |
| `scripts/r2_eval/r2_eval.sbatch:38` | `project_root="${PDMUX_PROJECT_ROOT:-/scratch/ehmoon/whlee/prefill-layer-alloc}"` | HEAD `bddff6a` | 예 |
| `results/longctx_conflict/probes/c_capacity.sbatch:76` / `:78-79` / `:111` / `:123-178` / `:126` / `:139-145` | ρ 사다리 `{0.60,0.90,1.30}` / 1.30 근거 / 셀 루프 / 셀별 launch·kill / `--disable-piecewise-cuda-graph` / warmup | HEAD `bddff6a` | 예(보관 아티팩트) |
| `sglang/bench_serving.py:70` | `BENCH_AIOHTTP_TIMEOUT_SECONDS = 6 * 60 * 60` | dev tree(sync 후) | — |
| `sglang/bench_serving.py:948` | `interval = np.random.exponential(1.0 / request_rate)` | dev tree | — |
| `sglang/bench_serving.py:1354` | `lora_name = np.random.choice(...)` (LoRA 전용, 미사용) | dev tree | — |
| `sglang/bench_serving.py:1705-1706` | `random.seed(args.seed)` / `np.random.seed(args.seed)` | dev tree | — |
| `sglang/srt/managers/scheduler.py:2013-2014` | `self.max_queued_requests is None or len(...)+1 <= ...` | dev tree | — |
| `sglang/srt/server_args.py:1959` | `assert self.attention_backend != "triton"` (NemotronH) | dev tree | — |

★**판정서 인용의 한 줄 어긋남 2건을 정정한다**(판정서를 무비판 승계하지 않는다 —
게이트 #110): (a) `generate_campaign.sh:18` → **`:17`**(:18은 `request_count`),
(b) probe C의 ρ 사다리 `{0.60,0.90,1.30}`는 `c_capacity.sbatch:78-79`가 아니라 **`:76`**.
어느 쪽도 판정서의 **주장**을 바꾸지 않는다(내용은 그 파일 안에 실제로 있다).

★**도구 한계 기록**: `scripts/discipline/check_line_citations.py`의 `SEARCH_ROOTS`는
`src` · `results` · `scripts` · `tests` · 엔진 `srt`뿐이라 **`benchmarks/`와 `srt` 밖의
`sglang/bench_serving.py`를 해석하지 못한다**(bare basename이 `UNRESOLVED`로 뜬다).
이 표는 그 구멍을 수작업으로 메운 것이며, 도구 자체는 이 세션에서 고치지 않았다
(공유 도구이고 같은 시각 다른 에이전트가 인접 파일을 편집 중이다).

---

## 11. 반증 실패 항목 (공정 기록 — 판정서 §7 승계, 이 rev2에서도 유지)

- **R1/R5의 실질 승계**: 보관 cell JSON 9개 직접 투입으로 `C_LABEL.json`의 R1·R5를 비트 단위
  재현. 항등식 아님, 날조 아님. **rev2의 실현-분모 교체 후에도 유지된다**(§4.2).
- **R4(MODEL_HOLDS)를 승계하지 않은 결정**: 옳다. 그 닫힌 형태는 `μ_ach·out·itl_load`로
  환원되는 **항등식**이다. 되살리지 말 것.
- **`CONSENSUS §3 항목120` 회피**: 성립(판정량이 telemetry를 경유하지 않음).
- **ctx 16384 / mem 0.82가 용량을 바꾸는가**: 바꾸지 않는다. 앵커 0.675의 provisional
  강등으로 충분하다.
- **F4의 제2 seed 통제**: 옳은 통제다 — 같은 seed로는 도착 실현이 재표본되지 않는다.
- **W4가 rate가 아니라 phase당 요청 수 승수라는 서술**: 코드와 일치, 정확하다
  (`benchmarks/pdmux_eval/workloads.py:150`, `:156` 30 s 구간 sorted-uniform, 둘 다 @HEAD `bddff6a`).
- **arm 비교 금지 규율**: 사다리가 arm-상대이므로 일관되게 유지한다.
