# Gate 2 — 4-arm 분해 사전등록 **rev3** (P1 운영점 기전 귀속)

작성: 2026-08-06 (메인 세션). **rev1 NO-GO**(설계 감사 1차) → **rev2 NO-GO**(설계 감사 2차).
rev3은 2차 감사의 F1–F8을 반영한 것이다. 죽은 규칙은 지우지 않고 **§12에 보존**한다.

**제출 전 고정. 결과를 본 뒤 바꾸지 않는다**(방법론 게이트 #8).
**현재 상태 = UNAUDITED (rev3 기준).** §11의 제출 선행조건 3건이 남아 있다.

상위 정본: [`PROJECT_STATUS.md`](../../../../../PROJECT_STATUS.md) "다음 실험 gate" #10 Gate 2 ·
[`CONSENSUS.md`](../../../../../reports/CONSENSUS.md) §1-1(rev12).
선행 캠페인: [`../../p1_opint/`](../../p1_opint/) — jobs 873944/873945.

---

## 0. 무엇이 걸려 있나

정본 §1-1은 2026-08-05 감사로 **"PD 분리 자체"라는 기전 귀속이 NOT-YET-SUPPORTED**로 강등됐다.
`--enable-pdmux`가 다른 두 플래그를 **assert로 강제**하기 때문이다:

```
server_args.py:6124-6137   (직접 확인, 2026-08-06)
  if self.enable_pdmux:
      assert self.pp_size == 1
      assert self.chunked_prefill_size == -1
      assert self.disaggregation_mode == "null"
      assert self.disable_overlap_schedule
```

⇒ 873944/873945가 잰 것은 **묶음 처치**(방법론 게이트 #13)다. 게다가 fused 측 死因(prefill
배치가 decode를 막음)의 fused 측 레버(`--chunked-prefill-size` 축소)가 미조율 기본값이었다.

**진짜 스테이크는 논문 신규성 축이다.** `plain+chunk512`가 `agnostic`을 대체하면 §1-1은
"PD-mux는 head-of-line blocking을 없애는 **여러 수단 중 하나**"가 된다.

### 0.1 두 번의 NO-GO가 가르쳐 준 것 — 이 사전등록의 핵심 문단

| rev | primary 결정 변수 | 죽은 이유 |
|---|---|---|
| rev1 | goodput | **A4가 바닥 절단.** A4 위반 0–6/600 ⇒ "A3가 따라잡았다"는 사실 상태가 곧 `UNSCOREABLE` 조건 ⇒ **"대체 가능"을 원리적으로 출력 못 함.** |
| rev2 | `frac_stalled(60)` 위의 격차-충족률 φ | **A1이 천장 절단**(Zamba2 plain r4 0.9849 → r6 0.9800으로 **감소**, 같은 구간 TTFT p50 306→2127ms) + **φ가 단조 재표현에 불변이 아님**(9 시나리오 중 4개서 판정 반전; φ=0.80 발화 지점에서 A3의 토큰당 stall 확률이 A4보다 **13.5× 나쁨**) + **§5.4가 goodput 절단을 판정 문구로 재수입.** |

**두 번 다 같은 병이다: 세 arm(A1, A3, A4)을 하나의 임계 지시함수 위에 올려놓으니, 어느 쪽
끝에서든 반드시 누군가가 포화한다.** A1은 위에서, A4는 아래에서 포화하고, 그 사이를 다 담는
τ는 없다(τ=100은 A1을 잘 펴지만 A4를 5/5 rep에서 정확히 0으로 만들고, τ=40은 그 반대다).

**rev3의 교정 = 추정 대상에서 A1을 뺀다.** 답해야 할 질문은 "A3가 A1→A4 격차의 몇 %를
메웠나"(A1이 들어가는 3-arm 비율)가 아니라 **"A3와 A4가 같은가"**(2-arm 등가성)다. A1이
빠지면 A1의 천장이 estimand에서 사라지고, **A4가 가장 잘 분해되는 τ를 고를 자유**가 생긴다.

부수로 죽는 것: 비율 추정량 φ(눈금 문제), Fieller(비율이 없으니 불필요 — 단 §12에 보존),
밴드 상수 4개(1개로 축소).

---

## 1. Arm 정의 (4-arm, 다른 모든 것 고정)

| arm | 이름 | 서버 플래그 (공통 플래그에 **추가**되는 것만) |
|---|---|---|
| **A1** | `plain` | (없음 — fused 기본값) |
| **A2** | `plain+aux` | `--chunked-prefill-size -1 --disable-overlap-schedule` |
| **A3** | `plain+chunk512` | `--chunked-prefill-size 512` |
| **A4** | `agnostic` | `--enable-pdmux --pdmux-config-path <cfg> --chunked-prefill-size -1 --disable-overlap-schedule` |

A1은 estimand에서 빠져도 **arm으로는 남는다** — 문맥(격차가 실재하는가)과 §3의 정상상태
판정, §5.3의 서술량에 필요하다.

공통 고정 플래그(873944/873945와 동일): `--trust-remote-code --dtype bfloat16
--disable-radix-cache --mem-fraction-static 0.82 --max-running-requests 48
--context-length <CTX> --attention-backend <BACKEND>`, **cudagraph ON**.
다른 `PDMUX_*` env 전부 미설정.

### 1.1 각 arm이 실제로 바꾸는 것 (코드 확인) [F5 — rev2 사실오류 정정]

**A2 = A4 − pdmux** (2차 감사도 이 등식을 깨지 못했다). A2/A4가 함께 켜는 것:

1. **이벤트 루프**: `event_loop_overlap()` → `event_loop_normal()` (`scheduler.py:3492-3503`)
2. **prefill 배치 상한**: chunking 없음, `max_prefill_tokens=16384`가 상한
3. **`piecewise_cuda_graph_max_tokens`: 8192 → −1** (`server_args.py:1253-1259`에서 파생)

⚠️ **rev2 §1.1-4("A2/A4는 `MambaRadixCache`로 바뀐다")는 사실오류였다 — 삭제한다.**
`scheduler.py:737-739, 762`의 분기는 `effective_chunked_prefill_size is not None`이고,
`None`이 되는 조건은 **multimodal + transformers backend 뿐**이다. Zamba2/Granite는 아니다.
realized 값 실측이 `chunked_prefill_size = -1`이고 `-1 is not None` → True ⇒
**네 arm 전부 `ChunkCache`다.** 따라서 Δ_flags는 ≥4가 아니라 **≥3 기전 묶음**이고,
"mamba pool 경로가 다르다"는 항목은 존재하지 않는다.

★ **A3에도 대칭 캐비어트를 건다(rev2가 빠뜨린 비대칭):** `piecewise_cuda_graph_max_tokens`는
A3에서도 **8192 → 512**로 함께 바뀐다(같은 파생). ⇒ **`Δ_chunk`는 "chunked prefill 조율"이
아니라 `chunked_prefill_size` **AND** `piecewise_cuda_graph_max_tokens`의 **≥2-기전 묶음**이다.**
R3′/R4′의 결론 문구를 "chunked-prefill 조율이 …"로 쓰는 것을 **금지**한다.

### 1.2 realized-vs-target 플래그 게이트 (게이트 #1의 플래그 축)

arm마다 서버가 최종 해석한 값을 `/get_server_info`에서 추출해
`g2_<tag>_flags_<arm>_<jobid>.json`에 저장한다: `chunked_prefill_size` ·
`disable_overlap_schedule` · `enable_mixed_chunk` · `piecewise_cuda_graph_max_tokens` ·
`max_prefill_tokens` · **tree_cache 클래스명**(§1.1의 정정을 매 job에서 재확인) ·
**`max_mamba_cache_size`**(§2.1). 추출 실패 시 그 arm은 `UNVERIFIED-FLAGS`로 판정 제외.

A1의 값은 선행 로그 실측으로 이미 알려져 있다: `chunked_prefill_size=8192`,
`disable_overlap_schedule=False`, `enable_mixed_chunk=False`, `piecewise=8192`.

### 1.3 `--enable-mixed-chunk` 가용성 프로브 (판정 arm 아님)

실제 게이팅은 `scheduler.py:881-884`의
`is_mixed_chunk = (chunked_prefill_size is not None) and enable_mixed_chunk` **하나뿐**이다.
1회 boot → 런타임 활성 여부 확인 → `AVAILABLE`/`UNAVAILABLE` 기록. 승격하지 않는다.

---

## 2. 격자 · 반복 · 페어링

- `Zyphra/Zamba2-2.7B` (ctx 4096, triton): rate ∈ **{2, 3}**
- `ibm-granite/granite-4.0-h-micro-base` (ctx 8192, flashinfer): rate ∈ **{3, 4, 6}**
- 선택 근거: 앞 넷은 **정본 §1-1(rev12)이 정량 인용 가능으로 열거한 셀**(정본 종속).
  Granite r6은 A1 판별력이 가장 큰 셀이라 추가(rev3의 estimand는 A1 천장에 걸리지 않는다).
- **n = 5 paired**, 모델당 1 job, 같은 노드·같은 job 안에서 4 arm 전부.
- **seed = 9000 + 100·rep + rate** — 873944/873945와 같은 공식. 같은 (rep, rate)에서 4 arm이
  동일 trace 공유.
- **arm 순서는 rep마다 무작위 순열**(4! 중 하나), 로그에 기록.
- 워크로드: `random-ids`, `--random-input-len 2000 --random-output-len 96
  --random-range-ratio 1.0`, 채점 120 prompts, `--warmup-requests 8`.

### 2.1 정확성·가용성 게이트 [F7]

A3는 이 프로젝트에서 실행된 적 없는 경로를 켠다: **단일 요청의 prefill이 여러 forward로
쪼개지고 그 사이에 mamba SSM state가 이월된다**(A1은 8192 > 2000이라 절대 안 쪼개짐).

⚠️ **rev2가 든 근거("`ChunkCache`에 mamba 처리가 없다")는 §1.1의 정정으로 무효다** — 그건
A1·A4에도 똑같이 참이고 둘 다 수백 런을 돌았다. 진짜 위험은 **다중-chunk prefill + 동시성**이며,
2차 감사가 지적한 대로 **은닉 오답보다 가용성 쪽이 크다**:
`mem_cache/memory_pool.py:528-548`이 chunk 간 state를 요청별 pool slot으로 보존하도록
**설계**돼 있고(주석에 명시), 고갈은 `assert mid is not None`으로 **hard crash**다.
그런데 realized `max_mamba_cache_size = 48 = max_running_requests`이고, A3에서는 각 요청이
**4 forward 동안** slot을 점유한다(A1은 1) ⇒ A1이 겪은 적 없는 pool 압력 영역.

**게이트(성능 측정 전):**
- **G-1 단일**: 벤치와 **동일한 생성기·동일한 2000 토큰** 프롬프트(≥2048이 아니라 정확히
  2000 — chunk 경계 512·512·512·464를 그대로 밟게), greedy, A1 vs A3 **출력 토큰열
  byte-identical**. A2도 동일 대조. A4는 배치 비결정성 때문에 **자기 2회 재현성**으로.
- **G-2 동시**(rev2에 없던 것, 이게 진짜 게이트): 같은 2000-토큰 greedy 요청 **≥16개를 동시에**
  A3에 던지고, 각 출력을 A1의 batch=1 greedy 출력과 byte-identical 대조. 혼합 배치 +
  mamba pool 압력을 실제로 밟는 유일한 방법이다.
- 산출물 `g2_<tag>_correctness_<jobid>.json`(출력 sha256, 동시 요청 수, pool 통계).
- **불일치 ⇒ A3 폐기**(또는 `--enable-deterministic-inference` 재확인 후 판단).

**우발 조항(사전등록)**: mamba pool assert crash가 나면 그 (arm, rep)을 **1회 재시도**하고,
재차 실패하면 그 rep을 **모든 arm에서 제거**한다. 제거 후 **n < 4면 그 모델은 판정 불가**로
보고한다(값을 낮춰 채점하지 않는다). crash 자체는 **결과로 보고**한다 — "A3는 이 동작점에서
운용 불가"는 유효한 발견이다.

---

## 3. Phase 0 — 정상상태 판정 [F6]

Phase 0: **4 arm 전부** 스캔. rate 그리드 `{1,2,3,4,5,6,8,10,12}`, `num-prompts=60`,
**`--seed` 3종(7, 17, 27)으로 n=3**(rev2의 n=1은 채점 셀의 정상상태를 n=1 다른-길이 런으로
증명하려던 격자 이전이었다 — 같은 rate에서 capscan과 채점 런의 TTFT p50이 최대 65% 어긋난다).

채점 rate마다 arm별로:

- **F-A(포화)**: `completed/total < 1.0`
- **F-B(TTFT 폭주) — 전역 검정으로 교체**: rev2의 "직전 rate 대비 2×"는 **연속 계단비가
  1.34/1.98/1.95/1.97로 전부 2.0 바로 아래**인 실측 폭주(Zamba2 agnostic rate 2→6에서
  TTFT p50 230→2352ms, **10.2×**)를 한 번도 잡지 못한다. 대신 **rate 1 대비 배수 > 4.0**
  ∨ **런 내부 도착 인덱스에 대한 TTFT 회귀 기울기의 95% CI가 0을 배제**.
- **F-E(큐 성장) — 임계 재도출**: 런을 도착 순서로 3등분한 `TTFT_p50(3분위)/TTFT_p50(1분위)`의
  **rep 평균**에 임계 **1.7**. rev2의 2.4는 **per-run 귀무분포**(40런, 0.514–2.388)에서 뽑아
  **rep-mean 통계량**에 걸었던 것이라 rep-mean 귀무(8 arm-cell, **0.964–1.240**) 대비 약
  13 SD 위 = 검정력 0이었다(집계 단위 교훈의 7번째 재발). 1.7은 rep-mean 귀무 상단
  1.240과 실측 양성례(Granite r6 agnostic rep-mean **1.940**, 같은 구간 TTFT p50 243→691ms)
  **사이**에서 잡았다 — 끝점이 아니라 간극에서 선택.
- ⚠️ **F-D(drain tail)는 삭제한다.** `bench_serving.py:1619-1625`의 `result_details`는
  `input_lens/output_lens/ttfts/itls/generated_texts/errors`뿐이고 **도착 타임스탬프·
  arrival_span이 어디에도 기록되지 않는다.** 기대값 `(N−1)/rate`로 대용하면 논리적으로
  불가능한 값(<1, 실측 최소 0.788)이 나온다. **계산 불가능한 게이트를 사전등록에 두지 않는다.**
  (도착 span을 덤프에 추가하는 것은 하네스 후속 항목으로 등재 — 이번 캠페인 범위 밖.)
- **모든 F-검정은 어느 데이터셋에서 계산하는지 명시한다** — F-A/F-B는 capscan(n=3),
  F-E는 **채점 런**(capscan과 같은 rate에서 최대 30.3% 어긋나므로 혼용 금지).
- **flag 시 조치**: 그 arm이 참여하는 비교는 **크기 인용 금지, 부호만 보고**. 셀·rep은
  버리지 않는다(페어링·n 보존).

---

## 4. 지표

### 4.1 ★ primary — A3 vs A4의 2-arm 등가성 [F1]

$$X_{\tau}(\text{arm}, \text{rep}) \;=\; \frac{\#\{\text{요청} : \max_i(\text{itl}_i) > \tau\}}{\#\{\text{요청}\}}
\qquad \textbf{primary } \tau = \mathbf{40\,ms}$$

**τ=40인 이유(rev2에서 바뀐 지점):** estimand에서 A1이 빠졌으므로 A1의 천장은 무관하고,
**A4가 바닥에서 가장 멀리 떨어지는 τ를 고를 자유**가 생긴다. 실측 A4: τ=40에서
Zamba2 r3 **0.1703**(rep별 0 없음) vs τ=60에서 0.0151(3/5 rep이 정확히 0) vs τ=100에서
0.0118(4/5 rep이 0). **τ=40이 유일하게 A4를 바닥에서 띄운다.**

- 집계 단위 **명시**: 요청 단위 지시함수를 **rep 안에서** 평균 → rep당 스칼라 1개 → n=5.
- **미기록-ITL 요청**(`itls` 빈 배열)은 분자·분모 모두에서 제외, 개수를 `n_missing_itl`로
  별도 보고. (선행 캠페인의 Zamba2 r2 "6건"이 지연이 아니라 계측 누락이었고, **plain·
  agnostic 양쪽에 같은 개수**로 존재해 paired 차분에서 상쇄된다 — 2차 감사 확인.)

### 4.2 보조 지표

| 지표 | 역할 |
|---|---|
| **TTFT p95** | **§5.2 R3′의 연언 조건**(서술 아님) [F2] |
| 요청별 token-ITL p95의 p90 | 서술 |
| **raw request_throughput** | **여집합 음성대조**(게이트 #10) — 모든 비교에 병기 |
| 토큰당 stall 확률 `p = #(gap>τ)/#(gaps)` | **서술 전용**(눈금 해석용). ⚠️ A4에서 run당 이벤트 ~2건으로 Poisson 잡음이 커 결정에 못 쓴다. |
| φ (rev2의 격차-충족률) | **서술 전용.** 점추정과 Fieller 구간을 그대로 싣되 **밴드 판정을 하지 않는다**(§0.1 눈금 문제). |
| goodput (정본 술어 · mean-ITL 술어) | **보고만. 어떤 판정 문구에도 쓰지 않는다** — §5.4를 포함해서(rev2의 위반 지점). |
| TTFT p50/p99, ITL p50, `n_err`, completed/total, `n_missing_itl` | 병기 |

### 4.3 τ 사다리 — 축소된 역할 [F8]

τ ∈ {40, 60, 100}에서 **Δ_chunk의 부호에만** 적용한다. 이유: Δ_pdmux 방향은 A1≫A4(≈50×)라
어떤 τ에서도 부호가 불변 = 그 절반은 **항등식에 가깝다**.

- **분모 변화표를 반드시 병기**한다 — τ를 바꾸면 A1−A4가 최대 20%, CV가 6×, φ 도달 상한이
  20% 변한다. **사다리는 고정 estimand의 견고성 검정이 아니다.**
- **양방향 공허 스크린**: 비교 두 arm 모두 `n_viol_latency == 0`(바닥) **또는** 둘 다
  `X ≥ 0.99`(천장)이면 그 rung은 **`VACUOUS`**, 부호 불변 판정에서 제외. rev2는 바닥만
  막았고 천장 방향에 구멍이 있었다(실측: Zamba2 r6 plain τ=40에 정확히 1.0000인 rep 1개).
- `n_viol`은 `n_viol_latency` / `n_viol_missing`으로 분리 집계, 스크린은 전자만.
- ⚠️ rev1·rev2의 **"부호 뒤집힘 = 절벽"** 해석은 삭제 상태를 유지한다. 뒤집힘은 절벽이 아니라
  **술어 포화**다.

### 4.4 구간 추정

- **차분의 구간 = paired t(n=5)를 primary**, bootstrap은 보조로 병기.
  근거(**result-analyst 독립 재현 완료 2026-08-06, `p1_gates/verify/`**):
  `analyze.py:115-142`의 `paired_bootstrap_ci`는 n=5 percentile bootstrap of the mean
  (BCa·studentization 없음)이고 실제 coverage = **0.8395 ± 0.0012**(100k trial, 정규),
  이 캠페인의 실제 rep-diff 경험분포에서는 **0.8254**, 왜도가 있으면 **0.723**까지 떨어진다.
  ⇒ 명목 95% 구간의 한쪽 오류율이 **≈8.0%**(명목 2.5%의 3.2배).
  **원인은 고정 seed도 정규 가정도 아니라 n=5 그 자체**(매 trial 새 설계로 리샘플해도
  0.8397). 폭 비는 감사자가 말한 0.58이 아니라 **0.624**(이론값 0.6314; 0.58은 n=4와 n=5
  사이 값) — **감사자 자기산출 3개 중 이것만 틀렸고 결론 방향은 불변.**
- ★ **n=5 paired의 구조적 한계(재현 검증에서 새로 나온 것)**: 정확 부호뒤집기 순열검정의
  두측 p 하한은 **2/32 = 0.0625**다. ⇒ **n=5 paired 셀은 분포무가정 검정으로 p<0.05에
  원리적으로 도달할 수 없다.** 이 캠페인의 "CI가 0을 배제"는 전부 모수 가정에 의존한다.
  n≥6이면 하한이 0.031로 내려간다 — §5.1의 검정력 재검증(§11-2)에서 이를 감안하라.
- **비율(φ)의 구간 = Fieller.** 2차 감사가 독립 MC(20,000회 × 정규/절단 DGP)로 검정해
  coverage **0.948–0.958**, 무한·역구간 **0/20000**, 5셀 전부 `g = 0.0008–0.0035 ≪ 1`.
  **반증 실패 = 채택.** 단 φ는 서술 전용이므로(§4.2) 판정에 쓰지 않는다.
  ⚠️ 이 결론은 **분모가 0에서 먼 이 격자의 성질에 종속**이다(게이트 #11: 새 셀로 이전 금지).

---

## 5. 사전등록 판정 규칙 (고정)

### 5.1 primary 검정 — TOST 등가성 (A3 vs A4)

$$H_0: \; |\mu(X_{40}(A3)) - \mu(X_{40}(A4))| \;\ge\; \delta
\qquad\text{vs}\qquad H_1: \; |\Delta| < \delta$$

paired, n=5, 양측 90% 구간(= 두 개의 단측 5% 검정). **상수는 δ 하나뿐이다**(rev2는 4개).

- **δ = 0.05** (frac_stalled 단위 = 요청의 5%p). 2차 감사 MC: 참 격차 0 → P(등가)=**0.975**,
  격차 0.05 → 0.070, 격차 0.10 → **0.002**. rev2의 φ 규칙(φ_true ≥ 0.85 필요)보다 압도적으로
  검정력이 좋다.
- ⚠️ **그 MC는 τ=60 분산 구조로 돌았다.** rev3은 τ=40으로 옮겼으므로 **δ와 검정력을
  873944/873945의 τ=40 A4 rep 분산으로 재검증하는 것이 제출 선행조건**이다(§11-2).
  재검증 결과 검정력이 부족하면 δ를 올리는 것이 아니라 **n을 올리거나 셀을 좁힌다**
  (임계를 결과에 맞춰 움직이지 않는다 — 게이트 #8).

### 5.2 판정

| 규칙 | 조건 | 결론 |
|---|---|---|
| **R3′ — 신규성 축 전환** | **지정 셀**에서 TOST 등가 성립(90% CI ⊂ (−δ, +δ)) **∧** **TTFT p95 비열등**: A3 − A4의 paired t 95% 상한 ≤ **+10%** | **chunk512는 충분한 대안.** §1-1을 "PD-mux는 head-of-line blocking을 없애는 **여러 수단 중 하나**"로 재작성. **논문 신규성 축이 바뀐다.** |
| **R4′ — 대체 불가(scoped)** | 지정 셀에서 A4가 A3보다 유의하게 우수(paired t 95% CI가 0 배제, 방향 A4 우수) | 이 동작점·이 모델 한정으로 **≥2-기전 chunk 묶음**(§1.1)이 pdmux를 대체하지 못한다. |
| **INCONCLUSIVE** | 둘 다 아님(등가도 우열도 미확립) | 검정력 부족을 "차이 없음"으로 쓰지 않는다. 점추정·구간을 그대로 싣는다. |
| **R1′ / R2′ — 묶음 분해** | A2 vs A4에 같은 TOST/우열 검정을 적용 | 등가 ⇒ **§1-1은 플래그 아티팩트로 붕괴**(pdmux 고유 기여 없음) / A4 우세 ⇒ **"PD 분리" 문구 부분 해금**, 해금 크기는 A4−A2. |

★ **F2가 연언인 이유(rev2가 봉인했던 것):** primary는 **admission-side 큐잉에 완전히
맹목**이다. 실존 증명 — Granite capscan **agnostic r10/r12에서 `frac_stalled(60)=0.0000`
(err=0, empty_itl=0)인데 TTFT p50 = 1827/2238ms**. A3의 기전(prefill당 forward 4배)이
정확히 "지연을 ITL에서 TTFT로 옮기는" 방향이다. TTFT 조건 없이 등가를 선언하면 **비용을
옮긴 것을 없앤 것으로 읽는다.**

### 5.3 다중비교 — 지정 셀 1개 + Holm [F4]

rev2의 "어느 한 셀에서라도 발화하면 채택"은 **실효 임계를 0.80에서 ≈0.72로 낮췄다**
(5셀 MC: φ_true=0.70에서 P(ANY 발화)=**0.319**).

- **R3′(충분성 확립)의 지정 셀 = Granite rate 4** (사전 고정). 근거: 5셀 중 A1 천장 여유
  0.050·CV(D) 2.60%·Fieller g 0.0010으로 최선.
- 나머지 4셀은 **Holm 보정된 사전등록 replication**. 지정 셀이 R3′이고 replication이
  뒷받침하지 않으면 `R3′(단일 셀, 미복제)`로 표기한다.
- ★ **두 주장을 분리한다**: **(i) pdmux 필요성 반증**(A4가 A3보다 낫다는 근거 없음 =
  어느 셀에서든 등가 방향 증거)은 검정력이 충분하고 다중비교 문제가 작다.
  **(ii) chunk512 충분성 확립**(R3′)은 **긍정적 정량 주장**이라 "한 셀이면 족하다"가
  적용되지 않는다. 보고서에서 이 둘을 섞지 않는다.

### 5.4 배치 이득 — goodput 사용 금지 [F3]

⚠️ **rev2 §5.4를 삭제한다.** rev2는 "배치 이득 = `G(A4) − max(G(A1), G(A3))`, 이 값이 0
이하면 '기전은 있으나 배치 이득 없음'"이라 썼는데, **이건 판정이고 그 값은 구조적으로 ≤0**
이다 — Granite A4 goodput은 r3/r4/r6 전부 **5/5 rep에서 정확히 1.0000**이라 A3가 1.000에
닿는 순간 자동으로 0.000이 나온다. **§0.1이 명명한 바로 그 절단을 재수입한 것.**

대신 **A2의 핸디캡 경고는 유지한다**: A2는 의도적으로 미조율된 arm이므로 A4−A2는
**기전 분해량이지 배치 이득이 아니다**(confound #2의 거울상 — 미조율 대조군이 처치의 공로를
부풀린다). 배치 관점 비교가 필요하면 **절단 없는 지표**(요청별 ITL p95의 p90, TTFT p95)의
쌍대 비교로만 보고한다.

### 5.5 인용 가능성

- 정량치 인용은 비교 arm이 전부 **steady-state(§3)** 일 때만. 아니면 부호만.
- `VACUOUS` rung(§4.3)은 부호 불변 판정에서 제외.
- goodput 수치는 어떤 판정 근거로도 쓰지 않는다.
- **φ는 점추정·구간을 싣되 밴드 판정 없이** — 독자가 자기 눈금을 적용할 수 있게 한다
  (rev2 §9-2의 "상수를 줄였으니 방어된다"는 논변은 **삭제**한다. 규칙별 체리피킹만 막고
  수준 자체의 임의성은 전혀 막지 못했다: 밴드를 바꾸면 φ_true=0.70에서 P(R3′)가
  **3%–93%** 사이 어디로든 간다).

---

## 6. 재현 진단 (판정 아님)

A1/A4는 873944/873945와 같은 seed 공식을 쓴다. 두 캠페인을 대조해 기록한다.

- ⚠️ "byte-identical 워크로드"는 **프롬프트·도착열에만** 참이다. 선행은 한 부팅 안에서
  rate를 `2→3→4→6` 순으로 돌렸다(`p1op_run.sbatch:145`). rev3의 Granite는 `{3,4,6}`이라
  **r3가 첫 번째**가 된다 ⇒ **Granite r3는 재현 대조군이 아니다.**
- **적신호 임계**: A1 또는 A4의 `X_40`이 선행 대비 **±10%**를 넘으면 **분석 전 중단**하고
  원인 조사(노드·클럭·manifest SHA).
- ⚠️ **교차-job 수치를 §5의 어떤 규칙에도 넣지 않는다**(방법론 게이트 #11; 최근 세 번
  위반, 그중 한 번은 금지 규칙을 쓴 문서의 바로 다음 문단).
- manifest SHA를 873944/873945와 대조 기록.

---

## 7. 아티팩트 레이아웃

```
workspace/engine-port/results/p1_gates/gate2/
  PREREG_GATE2_2026-08-06.md              (본 문서 rev3)
  AUDIT_GATE2_rev3_<date>.md              (rev3 감사 — §11-3)
  g2_run.sbatch · g2_analyze.py
  pdmux_a100_smoke.yml
  runtime_source_manifest_<tag>_<jobid>.sha256
  g2_<tag>_flags_<arm>_<jobid>.json       (§1.2)
  g2_<tag>_correctness_<jobid>.json       (§2.1 G-1 + G-2)
  g2_<tag>_capscan_<arm>_seed<s>_<jobid>.jsonl   (n=3)
  g2_<tag>_<arm>_rep<r>_<jobid>.jsonl
  g2_<tag>_srv_<arm>_*_<jobid>.log
```

---

## 8. 예산 · 운영

선행 job 역산: 벤치 33.5분(Zamba2)/31.8분(Granite), boot ≈1분. rev3은 capscan이 n=3이 되어
Phase 0이 3× 늘어난다 ⇒ 추정 **≈105분(Zamba2) / ≈115분(Granite)**. **`--time=04:00:00`.**
partition `amd_a100nv_8`, `--gres=gpu:1`, `--comment=pytorch`. 두 job 독립, 동시 제출 가능.

---

## 9. 이 사전등록이 스스로 인정하는 약점 (rev3)

1. **δ=0.05는 외부 검증이 없다.** 상수를 4개에서 1개로 줄였지만 그 1개는 여전히 내가 골랐다.
   §11-2가 완화책이나 제거는 아니다.
2. **§4.1의 τ=40 선택은 "A4가 바닥에서 가장 먼 τ"라는 기준으로 골랐고, 그 기준 자체는
   선행 데이터를 보고 정했다.** 사전등록으로 고정하는 것 외의 완화책이 없다.
3. **`X_τ`는 여전히 임계 지시함수다.** A1을 estimand에서 빼서 천장은 피했지만 눈금 문제
   (§0.1의 13.5× 논증)는 **완전히 사라지지 않았다** — δ=0.05가 토큰당 확률 단위로 무엇을
   뜻하는지는 셀마다 다르다. §4.2가 토큰당 `p`를 서술량으로 병기하게 한 이유다.
4. **A2/A3가 부팅되는지 미확인.** Phase 0 첫 boot가 게이트다.
5. **§5.3의 지정 셀(Granite r4)이 선행 데이터로 골라졌다.** replication이 완화책.
6. **실현 파티션은 이 게이트가 측정하지 않는다** — Gate 1 소관.
7. **G-2 동시 게이트는 batch 조성을 완전히 통제하지 못한다** — 16개 동시 요청이 실제
   서빙의 prefill/decode 혼합을 재현한다는 보장은 없다.
8. **이번이 세 번째 설계 반복이다.** rev3도 감사에서 죽으면, 다음 수는 **또 다른 통계량이
   아니라 다른 실험**을 설계하는 것이어야 한다(예: head-of-line blocking 지속시간을 엔진
   계측으로 직접 재는 것). 통계량 교체 루프를 무한히 돌지 않는다.

---

## 10. 인프라 판정 (2차 감사가 GO로 남긴 것)

§1(arm 정의·A2=A4−pdmux 등식) · §1.2(realized 플래그) · §2(페어링·seed·순열) ·
§6(교차-job 금지) · §7 · §8은 2차 감사에서 **GO 수준**으로 남았다. rev3은 이 부분을
§1.1의 사실 정정과 §2.1의 게이트 보강 외에는 건드리지 않았다.

---

## 11. 제출 선행조건 (셋 다 충족 전 제출 금지)

1. ~~result-analyst의 C-1/C-2/C-3 독립 재현~~ → ✅ **충족(2026-08-06, `p1_gates/verify/`)**.
   셋 다 CONFIRMED(C-1은 폭 비만 0.58→0.624 정정). §4.4 반영 완료, §13 갱신 완료.
   ⚠️ **파생 결과 하나가 rev3의 격자를 바꾼다**: **Granite rate3는 paired-t에서 0을 포함**
   (`[−0.0032, +0.6132]`, p=0.0515) ⇒ §2의 근거였던 "정본 인용 가능 셀 4개"가 **3개**로
   줄었다(Zamba2 r2·r3, Granite r4). **Granite r3는 격자에 유지하되**(A3-vs-A4 등가성은
   A1의 유의성과 무관하므로 estimand가 죽지 않는다) **"정본 인용 가능 셀이라서 골랐다"는
   §2의 선택 근거는 그 셀에 한해 성립하지 않음**을 명시한다. §5.3의 지정 셀
   **Granite r4는 어떤 방법으로도 살아남으므로 불변**.
2. **δ=0.05와 검정력을 τ=40 분산 구조로 재검증**(§5.1). 2차 감사 MC는 τ=60에서 돌았다.
3. **rev3 감사.** ⚠️ **1차 감사자의 처방(φ/frac_stalled)은 2차 감사가 죽였고, 2차 감사의
   처방(TOST/τ 이동/지정 셀)을 rev3이 채택했다 — 즉 rev3의 핵심은 다시 자기감사 상태다.**
   3차 감사는 **§5의 TOST 설계와 §4.1의 τ=40 선택을 1차 표적**으로 삼아야 한다.

---

## 12. 죽은 규칙 보존 (rev1 · rev2 — 재발 방지용)

**rev1 (goodput + 3% 임계)** — 전부 무효:
- R1/R3: `|Δ|/G(A4) < 3% ∧ paired CI 0 포함` ⇒ 붕괴/대체 가능. **귀무 채택형인데 등가성
  검정이 아니었고**, 실측 MDE가 3%의 3.6–11배:

  | 셀 | paired diff SD | CV | MDE(n=5,.80) | 3% 등가 필요 n(TOST) |
  |---|---|---|---|---|
  | Zamba2 r2 | 0.1211 | 6.56% | 10.90% | ≈22 |
  | Zamba2 r3 | 0.5995 | 20.33% | 33.79% | — |
  | Granite r3 | 0.2482 | 8.49% | 14.11% | — |
  | Granite r4 | 0.5126 | 13.00% | 21.61% | ≈85 |

- §3 F-C ∧ §4.1의 결합 불가능성: `G ≈ thr × pass`이므로 "모든 arm `pass ≥ 0.95`"와
  "한 arm `pass < 1.0`"을 동시에 만족하는 셀의 **최대 가능 효과 = 5.26%**, 결정 임계 3%,
  실측 MDE 10.9–33.8% ⇒ **통과한 셀은 정의상 판정 불가**였다.

**rev2 (frac_stalled(60) 위의 φ)** — 무효:
- φ 밴드 0.80/0.60/0.30/0.50: **단조 재표현에 불변이 아님.** 같은 물리적 A3를 frac_stalled /
  토큰당 p / log p로 채점하면 9행 중 4행에서 판정 반전. φ=0.80(R3′ 발화) 지점에서 A3의
  토큰당 stall 확률이 A4의 **13.5×**.
- 밴드 민감도: φ_true=0.70에서 P(R3′) = 0.030 / 0.321 / 0.717 / 0.902 / 0.929 (밴드 선택별).
- "어느 한 셀에서라도 발화" ⇒ 실효 임계 0.80 → **≈0.72**.
- §5.4 배치 이득: goodput 절단 재수입(Granite A4 = 1.0000 ×5/5, 세 셀 전부).
- §1.1-4 `MambaRadixCache`: **사실오류**(§1.1).
- F-E 임계 2.4: per-run 귀무 → rep-mean 적용(**13 SD 위**, 검정력 0).
- F-B 국소 2×: 실측 10.2× 폭주를 계단비 1.98/1.95/1.97로 통과시킴.
- F-D: **계산 불가**(arrival span 미기록).
- §9-2 밴드 방어 문장: 수준의 임의성을 막지 못함.

---

## 13. 이 게이트 밖의 정본 정정 별건 (doc-steward 이관 대상, **검증 중**)

**§11-1 완료 전에는 정본에 반영하지 않는다.**

1. `CONSENSUS.md` §1-1(rev12)의 **"임계 사다리 40–300ms 전 구간 부호 불변"** — Granite r3/r4는
   T=150·300에서 **양 arm 위반 0건**이고 부호가 음(throughput 효과)으로 뒤집힌다.
2. 같은 절의 **"Granite rate3 +11.7%, CI 0 배제"** — paired-t 구간 **[−0.0032, +0.6131]**로
   0을 포함한다는 주장.
3. `paired_bootstrap_ci`(`analyze.py:115-142`)의 n=5 undercoverage는 **이 도구를 쓴 다른
   정본 수치 전반**에 걸린다 — 영향 범위 조사 필요.
4. **(신규, 2차 감사)** `bench_serving`이 **도착 타임스탬프를 기록하지 않는다**
   (`bench_serving.py:1619-1625`) ⇒ 이 프로젝트의 어떤 캠페인도 **drain-tail/개회로 가정을
   사후 검증할 수 없다.** 하네스 후속 항목.
