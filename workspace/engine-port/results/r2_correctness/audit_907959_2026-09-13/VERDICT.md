# 결과 감사 판정서 — job 907959 (새 (Nano-9B-v2, flashinfer) 쌍 R2 correctness 게이트 첫 실행)

**감사일** 2026-09-13 · **감사자** claims-auditor (read-only) · **GPU 신규 지출 0** (기존 아티팩트 + CPU만)
**대상** `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/job_907959/` (+ `instrument/`)
**사전등록** `.../r2_correctness/newpair_prereg/PREREG_NEWPAIR_2026-09-13.md` (rev3, sha `e832fc49…`, 커밋 `b2b60e7`)
**규칙층 판정서 2건** 같은 디렉터리 `VERDICT_newpair_rules_2026-09-13.md`(sha `45ee250d…`) · `VERDICT_newpair_rev2_2026-09-13.md`(sha `cd1ffe8b…`)

---

## 0. 스코프 튜플 (이 판정서의 모든 문장에 붙는다)

> **(model `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base` = `NemotronHForCausalLM`, 56층 · backend `flashinfer` 휠 **0.6.10 이라고 보고한 설치본** · ctx **16384** · `mem-fraction-static 0.82` · `max-running-requests 48` · `--disable-radix-cache --chunked-prefill-size -1 --disable-overlap-schedule --random-seed 1` · split **fixed D44** = green stream index **4** (prefill 64 SM / decode 44 SM, 드라이버 read-out 실현 확인) · **cudagraph ON**(decode graph; piecewise는 엔진이 자동 비활성) · **verdict rule v2 (2026-09-11)** · 채점기 sha `ec355e17…` · commit `b2b60e79…` · A100-SXM4-80GB 108 SM 1장 · job 907959, gpu38)**

이 튜플의 **어떤 축이라도 다르면 다른 실험이다.** (Zamba2-2.7B/triton 결과 907100·907456·X1은 이 job이 되살리지도 확장하지도 않는다 — NP-4.)

## 0-b. 판정 요약

| # | 항목 | 판정 |
|---|---|---|
| 1 | `VERDICT FAIL`이 사전등록 규칙의 기계적 산물인가 | **CONFIRMED** (비트 단위 재현) |
| 1b | C층 중 크래시가 `B2_NO_CRASH`로 FAIL을 내는 것이 규칙 v2와 정합하는가 | **CONFIRMED** — 그리고 **과결정**(C층 request error를 제외해도 FAIL) |
| 2 | "true-dual이 OOM으로 죽는다"를 이 job이 지지하는가 | 서술적 사실 **CONFIRMED(scoped)** / 기전 귀속 **NOT-YET-SUPPORTED**(사전등록 §8이 금지) |
| 3 | S/O 불일치 0을 "동등성 인증"으로 쓸 수 있는가 | **REFUTED** — §2 인증문은 PASS 전용. 사용 가능 문안은 §4에 문자로 고정 |
| 4 | F2′가 정보량 0이었는가(NPC-B) | **CONFIRMED** — 투영치와 실측이 3 유효숫자까지 일치 |
| 5a | I1–I3 수치의 "D44 한정" 태그(NP-9/NPC-I) | **I2에 대해 REFUTED**(구성상 D44에서 돌 수 없다) · **I3에 대해 NOT-YET-SUPPORTED**(분할 혼합비 미측정) |
| 5b | `I3_max_running_req=48`이 클라이언트 병목 아님을 보이는가 | **셀 A만 CONFIRMED(scoped)** |
| 5c | F5(두 셀 각각 48 도달) | ★**REFUTED for cell B** — D4 레시피 재계산 결과 **셀 B 최댓값 = 2**. 등록된 처분(인용 금지)이 발화한다 |
| 5d | λ_inf(B) vs 905835의 0.675 (+3.00%)가 "closed-loop 상한"의 증거인가 | **PLAUSIBLE(무증거)** — out 96→64 교락이 같은 크기 |
| 6 | κ 모형의 핵심 입력(저측 rung 부하 B≈10의 ITL)이 측정됐는가 | **REFUTED** — I2는 B=1만 쟀고, 그 B=1조차 **비분할(108 SM)** 값이다 |
| 7 | NPC-A(D2′ 오버라이드) 발화 | **미발화**(`failures ≠ ['S_TIER_MISMATCH']`) |
| 8 | provenance 무결성 | **CONFIRMED** (5항목 전부 파일에서 직접 확인) |
| 9 | 이 FAIL의 차단 범위 | **P2/Claim D 차단** · λ0 0단계·P1은 이 FAIL로는 차단되지 않음(단 §6의 앵커 결함은 별건으로 λ0를 막는다) |
| ★신규 | λ0 rev3 `lambda0_lambda_inf.py`의 ANCHORED 술어가 F5 셀 B 실패를 **탐지하지 못한다** | ★**결함 확인 — 실행으로 재현** (§6-4) |

---

## 1. 재현 (감사자 독립 실행, GPU 0)

무수정 채점기를 이 job 디렉터리에 다시 돌렸다 — 출력이 `verdict.txt`와 **비트 단위 동일**:

```
failures=['TD1:B2_NO_CRASH', 'TD2:B2_NO_CRASH'] infra=[] unrealized=[]
VERDICT FAIL
```

`verdict_rule.txt`(job에 핀된 규칙 원문)는 채점기 `r2_correctness_check.py` 자신의 docstring과 **선행 구분선 1줄을 제외하고 문자 단위 동일**하다. 규칙은 2026-09-11에 고정됐고 채점기 sha는 사전등록이 예고한 `ec355e17…` 그대로다(§9).

---

## 2. ★질문 1 — 라벨이 규칙대로인가 · "C층은 진단 전용"의 정확한 사정거리

### 2-1. FAIL은 규칙의 기계적 산물이다 — **CONFIRMED**

규칙 v2 판정선 1: *"any B2–B6 failure in any boot"* → FAIL. 채점기 `:430-432`가 `B2_NO_CRASH`를 `failures`에 넣고 `:490-491`이 `if failures: verdict = "FAIL"`. 재량이 개입할 자리가 없다.

### 2-2. C층 데모션의 사정거리 — **규칙과 정합한다(CONFIRMED). 문안으로 확정한다.**

핀된 규칙 원문(`verdict_rule.txt:10-11`):

```
  C  32 staggered concurrent requests.                     DIAGNOSTIC ONLY
     (mismatch counts are reported and never enter the verdict)
```

괄호가 데모션의 정의역을 **`mismatch counts`로 명시적으로 한정**한다. 반면 `B2 NO_CRASH`(`:16-17`)는 **퍼-boot 술어**이고 티어 수식어가 없다:

```
  B2 NO_CRASH      no non-benign traceback, no "Scheduler hit an exception",
                   no "unsafe ... split transition", no request error.
```

⇒ **"C층은 진단 전용"은 "C층의 토큰 불일치 수가 판정에 들어가지 않는다"는 뜻이지, "C층 동안 일어난 일이 판정에 들어가지 않는다"는 뜻이 아니다.** 사후 재해석이 아니라 2026-09-11 고정 문안의 직독이다.

### 2-3. ★그리고 이 FAIL은 **과결정(over-determined)** 이다 — 데모션 재해석으로 뒤집을 수 없다

`B2`의 네 접속항을 boot별로 분해했다(리포트 JSON 원값):

| boot | `tracebacks` | `scheduler_exceptions` | `unsafe_transition`(로그) | `request_errors` | B2 |
|---|---|---|---|---|---|
| L1 | 0 | 0 | 0 | 0 | True |
| **TD1** | **1** | **1** | 0 | **31** | **False** |
| L2 | 0 | 0 | 0 | 0 | True |
| **TD2** | **1** | **1** | 0 | **31** | **False** |

**반전 시험 2건(감사자 실행, 채점기 사본을 스크래치패드에 두고 원본 무수정):**

| 반사실 | 변경 | 결과 |
|---|---|---|
| **CF1** — C층 request error를 `req_errors` 합산에서 제외(가장 관대한 "C 진단 전용" 독법) | `for tier in ("S","O","C")` → `("S","O")` | **여전히 `VERDICT FAIL`** (`TD*:B2_NO_CRASH`) — 로그의 traceback 1 + scheduler exception 1이 단독으로 B2를 죽인다 |
| **CF2** — B2를 통째로 `True`로 고정 | `chk["B2_NO_CRASH"] = True` | `VERDICT PASS` |

⇒ **CF1이 결정적이다.** "C가 진단 전용이므로 C의 request error는 세면 안 된다"는 (규칙 문안에 반하는) 독법을 최대한 양보해도 라벨은 FAIL로 남는다. 서버 프로세스가 **스스로 예외를 던지고 SIGQUIT로 죽은 것**은 티어와 무관한 boot 수준 사실이기 때문이다. CF2는 "B2가 유일한 판별자"임을 보이는 분해일 뿐이며 **대안 판정이 아니다**(사후 규칙 수정 = 금지).

---

## 3. ★★질문 2 — 귀속: "true-dual이 OOM으로 죽는다"를 이 job이 지지하는가

### 3-1. 관측 사실 (전부 아티팩트 직인용)

두 TD boot에서 **동일·결정적** 크래시:

```
File ".../multiplex/dual_worker.py", line 336, in _loop
  result = task.callback(task.context)
File ".../multiplex/multiplexing_mixin.py", line 1211, in _run_prefill
  _result = self.run_batch(_batch)
  ... tp_worker.py:545 forward_batch_split_prefill
  ... model_runner.py:2721 forward_split_prefill
  ... models/nemotron_h.py:879 forward_split_prefill
  ... models/nemotron_h.py:437 _forward_mamba
  ... mamba/mamba.py:690  hidden_states = self.norm(preallocated_ssm_out, gate[:num_actual_tokens])
  ... mamba/mixer2_rms_norm_gated.py:110 forward_cuda -> :97 forward_native
        return self.weight * x.to(input_dtype)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 198.00 MiB.
  GPU 0 ... 79.25 GiB of which 55.19 MiB is free ... 77.25 GiB is allocated by PyTorch,
  156.00 MiB allocated in private pools (CUDA Graphs), 1014.25 MiB reserved but unallocated
```
서버 상태 동일: 직전 `Decode batch, #running-req: 14, #full token: 21285`. TD1 22:08:56 / TD2 22:12:33. 이어 `SIGQUIT received`(엔진 자기 종료) → 하네스 `SERVER_DIED_DURING_CLIENT`.

### 3-2. ★대조가 무엇인가 — **부하 궤적이 크래시 직전까지 완전히 동일하다**

C층 서버측 prefill batch 시퀀스를 L1 / TD1에서 전수 대조했다(`Prefill batch` 줄은 `process_batch_result_prefill` → `report_prefill_stats`에서 **완료 후** 찍히므로, 크래시한 배치는 줄이 없다):

| 순서 | L1 (22:07:00→05) | TD1 (22:08:52→54) |
|---|---|---|
| 1 | `#new-seq 1, #new-token 55, #running 0, #queue 0` | **동일** |
| 2 | `1 / 368 / run 1 / q 1` | **동일** |
| 3 | `1 / 1468 / run 2 / q 3` | **동일** |
| 4 | `4 / 3241 / run 3 / q 8` | **동일** |
| 5 | `8 / 6245 / run 7 / q 14` | **동일** |
| 6 | **`14 / 10125 / run 14 / q 3` — 완주** | **동일 배치에서 OOM(줄 없음)** |
| 7 | `3 / 3491 / run 22 / q 0` — 완주 | (도달 못 함) |

**산술 재구성(독립):** C층 32요청 프롬프트 합 = **24,993 토큰**(`gen_L1.json`/`gen_L2.json`에서 전수; 두 L boot 동일). L1의 C층 prefill 토큰 합 = 55+368+1468+3241+6245+10125+3491 = **24,993**(32 seq, 정확 일치). TD가 완주한 5배치 = 15 seq / **11,377** 토큰 ⇒ 잔여 17 seq / **13,616** 토큰 = **10,125 + 3,491**. 크래시 직전 `#queue-req`가 14→0으로 비었으므로 **크래시한 배치 = 14 seq**, 즉 **L1이 완주한 바로 그 10,125 토큰 배치**다.

**커널측 교차검증:** mamba intermediate = `mamba_num_heads × mamba_head_dim` = 128 × 80 = **10,240**(체크포인트 `config.json` 직독), dtype bf16. `10125 × 10240 × 2 = 207,360,000 B = 197.75 MiB`. PyTorch 캐싱 할당자의 대형 세그먼트 2 MiB 올림 ⇒ **198.00 MiB**. 역으로 "198.00 MiB"가 허용하는 토큰 구간은 **[10036, 10137]** — 재구성값 10125가 그 안에 있다. (유일 결정은 아니고 **정합 확인**이다.)

### 3-3. 대안 설명 전수와 처분

| # | 대안 | 처분 | 근거(아티팩트) |
|---|---|---|---|
| A1 | **부하 비대칭(TD가 더 많이/더 큰 배치를 받았다)** | ★**기각** | 크래시 배치까지 prefill 궤적이 seq 수·토큰 수·`#running`·`#queue`까지 **완전 동일**. 게다가 L1/L2는 **더 큰** 배치를 완주했다(L1 14 seq/10,125, **L2 16 seq/12,516**), TD가 완주한 최대는 8 seq/6,245 |
| A2 | **동시 decode 인구가 TD에서 더 컸다**(크래시 순간 TD decode 14 @21,285 vs L1의 14-seq prefill 순간 decode 8 @16,780) | **기각(방향이 반대)** | L1은 같은 10,125 배치를 돌린 **직후** decode 22 @20,293으로 올라갔다 ⇒ L1의 동시 상태(22 decode + 10,125 prefill)가 TD의 것(14 decode + 10,125 prefill)보다 **크다**. 또한 mamba ssm state(6.46 GB)·KV(41.5 GB)는 **사전 할당 풀**이라 decode 인구는 활성화 메모리에 O(MB)만 기여 |
| A3 | **boot 순서 효과 / 이전 boot 잔류 메모리** | ★**기각** | 5 boot(warm-up 포함) 전부 배너가 **동일**: `max_total_num_tokens=2721290`, `available_gpu_mem=11.59 GB`, `Mamba ... 6.46GB`, `KV ... K 20.76/V 20.76 GB`. 순서 `L TD L TD`에서 L2(#3)가 TD1(#2)과 TD2(#4) 사이에 있고 clean. warm-up 서버는 22:05:17 SIGTERM, L1은 22:06:04 부팅 |
| A4 | **계측(`R2C_INSTRUMENT=1`)의 50분 부하 잔류** | **기각** | A3와 동일 근거. 계측은 warm-up boot 전용이고 scored boot은 별도 프로세스·별도 포트. 사전등록 §9(b)-D12의 분기 조건("**첫 scored boot**이 OOM")도 미충족 — 첫 scored boot L1은 정상 |
| A5 | **`request_errors=31`이 크래시의 원인** | ★**기각(인과 방향 반대)** | 31건 전부 시그니처 `RemoteDisconnected: Remote end closed connection without response` = **서버 사망의 결과**. C층 32건 중 성공 1건(C01, 서버 로그의 22:08:53 `POST /generate 200 OK`와 일치). S층 0 error, O층 probe 0 error, O층 bg 0 error |
| A6 | **`unsafe_decisions=16`이 관여** | **기각(무관, 그리고 B2에 안 들어감)** | 텔레메트리 전수: 16건 전부 `policy=fixed, reason='unsafe_transition', target_decode_sms=44, current_decode_sms=44, admission_limited=False`. `FixedPolicy.decide`(`controller.py:95-101`)가 `safe_boundary=False`면 `target=current`를 돌려주는 **설계상 경로**이고 분할은 바뀌지 않는다(B8·O3가 346/346 idx 4로 확인). 시각 분포: O층 8건(probe 1개당 1건, 2.8 s 간격) + C층 8건 ⇒ **prefill∧decode 동시성의 표지**이지 결함이 아니다. **B2가 보는 것은 로그 정규식** `unsafe (automatic )?split transition`이고 그 값은 4 boot 전부 **0** |
| A7 | **L boot의 C층 7 불일치가 다른 실행 경로를 뜻한다** | **기각** | `C_pairs['L1-L2']` 7건은 전부 `len [n,n]` 동일 길이의 **중간 토큰 분기**(first_divergence 4/4/28/29/45/73/94) = legacy 자체의 동시부하 비결정성. 실행 경로 차이가 아니다. (NP-5: 907032의 7/32와 **비교 금지** — 우연히 같은 7이다) |
| A8 | **true-dual이 prefill 활성화를 worker 스레드에 붙들고 있어 피크가 높다**(스레드/스트림별 할당자 풀 분리, `prefill_future.result()` 지연 수확) | ★**보류 — 유력하나 미측정** | `multiplexing_mixin.py:1206-1216`(submit) ↔ `:1255-1262`(수확)의 코드 구조와 정합하지만 **이 job은 메모리 계측을 하지 않았다**(`torch.cuda.max_memory_allocated` 텔레메트리 부재). 기전 주장 불가 |
| A9 | **flashinfer 워크스페이스/JIT 차이** | **기각** | 4 boot 모두 동일 배너·동일 `available_gpu_mem`. NPC-E의 휠 드리프트 위험은 arm 간 대조에는 영향 없음(같은 프로세스 이미지) |

### 3-4. 판정

> **CONFIRMED(scoped)** — 위 스코프 튜플에서, **동일한 14-seq / 10,125-token split-prefill 배치**에 대해 **true-dual boot 2/2가 `torch.OutOfMemoryError`로 스케줄러 예외를 내고 죽었고, legacy boot 2/2는 같은(그리고 더 큰) 배치를 완주했다.** 부하·배치·배너·commit·seed·backend·cudagraph·분할은 전부 동일하며, 열거한 대안 A1–A7·A9는 아티팩트로 기각된다.

> **NOT-YET-SUPPORTED** — "**true-dual이 (일반적으로) OOM으로 죽는다**" · "**true-dual의 활성화 피크가 legacy보다 X GB 크다**" · "**true-dual이 이 쌍에서 틀렸다**". 사전등록 §8 FAIL 행이 **이 job 단독의 귀속을 명시적으로 금지**하고 원인 분해 회차를 요구한다(그 회차는 등록되지 않았다). 기전 후보 A8은 미측정이고, 마진(11.59 GB 헤드룸 중 legacy가 얼마를 썼는지)은 측정되지 않았다.

---

## 4. ★질문 3 — S층·O층 불일치 0을 어떻게 쓸 수 있는가 (문자로 확정)

**측정 사실(전사):**
- `S_pairs` 6쌍 전부 `n=16, mismatches=0`: `L1-TD1 0 · L1-L2 0 · L1-TD2 0 · TD1-L2 0 · TD1-TD2 0 · L2-TD2 0`
- `O_pairs` 6쌍 전부 `n=8, mismatches=0`; `O_within_arm_mismatches {L1-L2:0, TD1-TD2:0}`, `O_cross_arm_mismatches` 4쌍 전부 0; `O_null_control_complete=True`
- O1–O4가 **4 boot × 8 probe = 32/32 전부 confirmed**(TD의 `prefill_task_delta=2`, `decode_task_delta=45–63`, `probe_in_decode_batch=True`, `window_stream_indices=[4]`)
- `B6_reference_degenerate = 0/56`, `prompt_tokens_consistent {S: True, O: True}`
- F3 검증: O00의 probe `prompt_tokens=1280` vs 문턱 `7×(17+160)=1239` ⇒ **최악 여유 +41 토큰**, 사전등록 preflight 예측과 **정확히 일치**(양성대조 성립)

**쓸 수 있는 문장(결과 문서가 그대로 승계):**

> 위 스코프 튜플의 job 907959에서 `PDMUX_TRUE_DUAL_WORKER=1` 경로와 legacy 루프는 **S층(16요청 순차)과 O층(8 probe 중첩 프로토콜)의 전 boot 쌍에서 토큰 id가 동일했고(6/6·6/6 쌍, 불일치 0), O1–O4가 32/32 probe-boot에서 확인됐다.** 이것은 **측정 사실의 기록**이다.

**쓸 수 없는 문장(금지):**

> ❌ "동등성이 인증됐다 / 게이트를 통과했다 / true-dual이 같은 토큰을 낸다" — 사전등록 §2는 인증 문장을 **"PASS일 때, 그리고 PASS일 때만"** 허용한다. 이 job의 라벨은 **FAIL**이고, 규칙 v2 판정선 5(PASS)는 "every boot passes B1–B8"을 요구하는데 **B2가 두 boot에서 실패**했다. 따라서 §2의 인증문은 **사용 불가**다.
> ❌ "S/O는 깨끗했으므로 사실상 PASS다 / C만 아니었으면 PASS였다" — 감사자의 CF2(§2-3)가 그 반사실을 계산했지만 그것은 **규칙의 사후 수정**이며 대안 판정이 아니다.
> ❌ "동시 부하에서도 같은 토큰을 낸다" — O 프로토콜 밖의 동시성은 §2가 인증 범위에서 제외했고, 이 job에서 C층은 **TD arm이 31/32 요청을 완료조차 못 했다**.

---

## 5. ★질문 4 — NPC-B 검산: F2′는 정보량 0이었는가

| 양 | 재감사 투영치(실행 전, `VERDICT_newpair_rev2:187` · 사전등록 §7) | 실측(5 boot 전부 동일) | 등록 문턱 | 오차 |
|---|---|---|---|---|
| `max_total_num_tokens` | **≈ 2.72 × 10⁶** | **2,721,290** | ≥ 2.0 × 10⁶ | **−0.05%** |
| `available_gpu_mem` | **≈ 11.6 GB** | **11.59 GB** | > 5 GB | **−0.09%** |

**판정: CONFIRMED — F2′는 형식 요건(출력공간에 거짓 존재)만 충족했고 실질 정보량은 0이었음이 실증됐다.** 실행 전 투영치가 두 양 모두 **3 유효숫자까지** 맞았고, 등록 문턱까지의 거리(≈11.0 GB / ≈6.6 GB)는 식별된 신규 소비자 총합(≈0.21 GB)의 **30–50배**였다.

**다음 회차가 인용해야 할 문안(문자 승계):**

> **F2′(그리고 그 계열의 "배너 확인" 예보)는 사후 확률이 1에 가까운 기록 항목이다.** job 907959에서 실행 전 투영치 `2.72×10⁶ / 11.6 GB`가 실측 `2,721,290 / 11.59 GB`를 0.1% 이내로 맞혔다. **"KV 여유를 측정으로 확인했다"는 쓸 수 없고**, 이런 예보는 **사전등록의 반증 가능성 요건을 채우는 용도로만** 유효하다. 새 회차가 배너 예보를 넣을 때는 **투영치와 문턱의 거리를 함께 등록**해 정보량을 스스로 공시하라.

또한 **NP-7′는 이 job에서 재확인**된다: `max_mamba_cache_size: 48` / `max_running_requests=48`은 5 boot 전부에서 48이며, 이는 `--disable-radix-cache --max-running-requests 48`의 **연역**이지 ctx 16384 / mem 0.82에서의 측정이 아니다.

---

## 6. ★★질문 5 — I3(λ_inf)의 사용 가능성

### 6-1. (a) NPC-I 스코프 적합성 — **부분 성립**

| 축 | 판정 | 근거 |
|---|---|---|
| legacy 루프 | ✅ | `srv_warmup.log`에 `true dual-worker enabled` 배너 **0건** |
| warm-up boot(비채점) | ✅ | 계측은 `boots.txt`에 없는 boot에서만; 채점기는 5개 경로 템플릿만 연다 |
| cudagraph ON | ✅ | warm-up: `Capture cuda graph end` 1, Decode `cuda graph: True` **17,487 / False 0**, piecewise 자동 비활성 1건 |
| **D44** | ❌ **I2는 거짓 · I3는 미검증** | §6-2 |

### 6-2. ★★"D44 한정"(NP-9 / NPC-I / 사전등록 §5-I2 D6-ii)은 **I2에 대해 거짓이다** — REFUTED

드라이버 read-out(`instrument/green_warmup.json`, `instrument/I1_green_warmup.txt`)이 말하는 것:

```
idx=0 target=(108,0)  realized=(None,None) null=(True,True)     <- 비분할(green ctx 없음)
idx=4 target=(64,44)  realized=(64,44)     null=(False,False)   <- D44
idx=5 target=(0,108)  realized=(None,None) null=(True,True)     <- 비분할(green ctx 없음)
```

**scored boot 텔레메트리 전수(감사자 재집계, L1):** `(state, stream_index, prefill_sms, decode_sms)`

| 상태 | stream_index | 스냅샷 수 |
|---|---|---|
| idle / prefill-only | **0** (108, 0) — 비분할 | 11,725 |
| **prefill ∧ decode** | **4** (64, 44) — **D44** | **592** |
| decode-only | **5** (0, 108) — 비분할 | 141 |

즉 **D44(idx 4)는 prefill과 decode가 *동시에* 활성인 구간에서만 사용된다.** 엔진 코드가 같은 말을 한다(`multiplexing_mixin.py:922-971`: decode 비어 있으면 idx 0, decode만 있고 split-prefill 없으면 `real_sm_group_num-1`=5) — 그리고 규칙 v2 원문 자신도 `verdict_rule.txt:84-86`에 "S alone never runs on the D division"이라 적고 있다.

⇒ **I2는 `--max-concurrency 1`이다. 구성상 prefill∧decode 중첩이 발생할 수 없다**(I2 구간 `#running-req` 최댓값 = **1**, prefill 줄은 전부 `#running-req: 0`). 따라서 **I2의 TTFT 43.03 ms와 ITL 13.0 ms는 108 SM 비분할 값이고 D44 값이 아니다.**

이것은 **정본과의 모순 해소**이기도 하다: λ0 rev1 판정서 `:151`이 이미 *"(B=1은 13.06 ms — pdmux 분할 미적용 구간.)"*라고 적었다. 새 사전등록의 (D6-ii)가 그 정본 문장과 충돌했고, **정본 쪽이 옳았다.**

I3a/I3b는 중첩이 자주 발생하므로 **일부는 D44에서 돌았을 것**이나, warm-up boot은 `PDMUX_TELEMETRY_PATH` 없이 뜨므로 **한 줄도 기록이 없다** ⇒ 혼합비 **미측정**. "λ_inf는 D44에서 쟀다"는 **NOT-YET-SUPPORTED**.

### 6-3. ★(b)(c) F5 — D4 셀별 재계산: **셀 B는 48에 도달하지 못했다 (F5 REFUTED for cell B)**

사전등록 §5-I3 (D4)가 **실행 전에 문자로 고정한** 레시피를 그대로 실행했다(`I_log_offsets.txt`: start 배타 / end 포함):

| 셀 | 구간(줄) | **셀별 max `#running-req`** | 분포 |
|---|---|---|---|
| I2 (256,512) conc 1 | 85–4226 | **1** | {0:9, 1:4119} |
| **I3a (256,512) inf/64** | 4227–8434 | **48** ✅ | decode 줄 3,880 중 **2,814(72.5%)가 정확히 48**, `#queue-req` 최대 47 |
| **I3b (8192,64) inf/64** | 8435–18380 | ★**2** ❌ | {0:2, 1:31, 2:9,599}. prefill 줄 151개가 전부 `#new-seq: 2, #new-token: 16384`(= `max_prefill_tokens` 상한), `#queue-req`가 대부분 60, 최대 **62** |
| 전역(하네스가 기록한 값) | 전체 | 48 | ← **I3a에서만 온 값** |

**⇒ F5("두 셀 **각각** 48 도달")는 셀 B에서 반증됐다.** 1차 판정서 §1 행 7이 예고한 결함("전역 max 1개로는 '두 셀 모두'를 판정할 수 없다")이 **실제로 발생**했고, D4가 그 판정을 문서에 등록했기 때문에 **재량 없이** 처분이 결정된다:

> 사전등록 F5 "어긋나면" 행: **"미달이면 그 셀의 achieved를 포화 처리율로 인용 금지."** + (D4) **"미달의 원인이 상류 병목이라고 단정하지 않는다."**

**따라서 `λ_inf(B) = 0.6956 req/s`는 이 사전등록 하에서 포화 처리율/상한 앵커로 인용할 수 없다.**

**(감사자 주 — 구제 시도를 명시적으로 차단한다.)** 셀 B에서 `#queue-req`가 런 대부분 60 이상으로 유지됐다는 사실은 "클라이언트가 아니라 서버가 병목"이라는 **다른 계기(instrument)** 의 증거다. 그러나 그 계기는 **등록되지 않았다**. 지금 그것으로 F5 실패를 무효화하는 것은 **사후 계기 교체**이며, 이 프로젝트가 두 번 데인 재스코어 계열(confound #6)의 한 형태다. → **새 사전등록에서만** 사용 가능. 권장 등록 문안은 §13-D2.

또한 셀 B의 진짜 구속은 **`max_prefill_tokens = 16384`** (배너 실측; 905835도 ctx 8704에서 **동일하게 16384**)이지 mamba 슬롯 48이 아니다 — 즉 F5의 `#running-req≥48` 기준은 **prefill 지배 shape에 대해 애초에 틀린 계기**였다. 이것은 사전등록의 설계 결함이지 측정 실패가 아니다.

### 6-4. ★★신규 결함 — λ0 rev3의 ANCHORED 술어가 이 실패를 **탐지하지 못한다** (실행으로 확인)

`results/r2_eval/lambda0_prereg/lambda0_lambda_inf.py`(미커밋, λ0 rev3 동봉)는 조건 (b)를 이렇게 구현한다:

```python
MIN_RUNNING_REQ = 48        # newpair F5
RUNNING_FILE = "I3_max_running_req.txt"
...
vals = _last_int_per_line(rp.read_text())
if min(vals) < MIN_RUNNING_REQ:
    reasons.append(... "newpair F5 ...")
```

그런데 newpair 하네스(`r2_correctness.sbatch:429-430`)는 그 파일에 **전역 최댓값 한 줄**만 쓴다:

```bash
grep -oE "#running-req: [0-9]+" "$OUT/srv_warmup.log" | awk '{print $2}' \
  | sort -n | tail -1 > "$IOUT/I3_max_running_req.txt"
```

감사자 실행 결과(무수정, job 907959의 실제 `instrument/`):

```
LAMBDA0_MODE=ANCHORED
LAMBDA0_LAMBDA_INF_A=3.093892399983411
LAMBDA0_LAMBDA_INF_B=0.6955869013158362
LAMBDA0_DISQUALIFICATIONS=[]
```

> ★**술어는 `ANCHORED`·실격 사유 0건을 내놓는다 — 셀 B의 실제 셀별 최댓값이 2인데도.** 술어의 selftest는 두 줄 포맷(`"I3a_shapeA 48\nI3b_shapeB 31"` → FALLBACK)을 시험하지만, **산출측 하네스가 그 포맷을 절대 만들지 않으므로 그 분기는 live에서 도달 불가**다. 즉 F5 실격 경로는 **실효 항등식**이다(교훈 9 계열 / 게이트 #200 계열 재발: "대체 지표가 live 산출물에 실제로 존재하는 형태인지 확인하라"를 *소비자 쪽에서* 다시 어겼다).

**이것은 λ0 rev3 제출 전 반드시 고쳐야 하는 결함이다.** (§13-D1)

### 6-5. (d) λ_inf(B) vs 905835의 0.675 — **PLAUSIBLE(무증거)**

| 양 | job 907959 I3b (closed-loop, `rate inf`, conc 64) | job 905835 `d44_r2_o96` (open-loop Poisson) |
|---|---|---|
| shape | 8192 in / **64** out, ctx **16384** | 8192 in / **96** out, ctx **8704** |
| req/s | **0.69559** | **0.67530** (achieved, offered 0.86, ach/off 0.785) |
| input tok/s | **5,698.25** | **5,532.08** |
| ITL p50 | 20.30 ms | 20.38 ms |
| 차이 | **+3.00%** (req/s와 input tok/s가 동일 비율) | — |

**해석 판정: closed-loop ≥ open-loop라는 해석과 *모순되지 않으나 그 해석의 증거는 아니다*.** +3.00%는 **out 96→64(디코드 토큰 33% 감소)** 라는 동시변경 교락과 같은 크기이며, 두 변수를 함께 바꿨다(confound #10). 또한 셀 B는 §6-3에 의해 애초에 포화로 인용 불가다. 부수적으로 **I3b가 shape B에 대해 905835 이상의 정보를 주지 못했다**는 것도 기록해 둔다(3% 차이 = 교락 폭).

### 6-6. λ_inf(A)=3.0939 — 등록된 λ*(A) 사전추정보다 **23–47% 크다**

λ0 rev1 판정서 §3-3의 λ*(A) 점추정은 2.10–2.51 req/s(`step(48)∈[37.4, 44.65] ms`)였다. I3a 실측 λ_inf(A)=3.094 (⇒ 등가 step ≈ 48/(3.094×512) = **30.3 ms**). λ_inf ≥ λ*이므로 **형식적 모순은 없다**. 그러나 이 간극은 §6-2의 **분할 혼합비 미측정**과 직접 연결된다(비분할 idx 5 구간에서 decode가 108 SM으로 돌면 step이 D44 회귀보다 작아진다). ⇒ **λ_inf(A)를 "D44에서의 포화 상한"으로 읽는 것은 지지되지 않는다.**

---

## 7. ★질문 6 — I2의 ITL(B=1)과 λ0 rev2 死因의 접점

λ0 rev2의 死因 T1/T1′는 **저측 rung 부하(B≈10)에서의 ITL**이 19.8–25 ms이면 `κ = (N/(N−1))·span/(span+L)`이 0.95를 못 넘어 shape A의 `KNEE_BRACKETED`가 뒤집힌다는 것이었다. 저측이 0.95를 넘으려면 B≈10에서 step ≤ 19.69 ms가 필요한데 등록된 회귀의 **절편만 19.82 ms**다.

**I2가 잰 것:** 8요청 × (256 in, 512 out), 동시성 1, duration 54.52 s.
TTFT mean **43.03** / median 42.57 / p99 44.48 ms · TPOT mean 13.22 / median 12.97 / p99 14.64 ms · ITL mean 13.22 / median **12.96** / p95 13.05 / p99 13.25 / **max 929.91** ms.

**판정: κ 모형의 핵심 입력은 여전히 미측정 — 그리고 두 겹으로 미측정이다. (REFUTED: "I2가 死因을 건드렸다")**

1. **B 축**: I2는 **B=1만** 쟀다. 死因이 요구하는 것은 **B≈10에서의 step**이다. I2는 그 자리를 건드리지 않는다.
2. **분할 축**(§6-2 신규): B=1 값 자체가 **비분할 108 SM** 값이다. 死因 T1이 대비시킨 두 상수가 바로 **13.06 ms(비분할)** ↔ **19.8–25 ms(D44)** 인데, I2는 **비분할 쪽을 다시 쟀을 뿐**이다.
3. **정보 증분 ≈ 0**: I2의 median ITL **12.96 ms**는 λ0가 이미 들고 있던 `ITL_SOLO_S = 13.06 ms`와 **0.8% 차**다. (D6-i의 "클라이언트 ITL ≠ 엔진 step" 단위 경고는 형식적으로 옳으나 **수치적으로 무의미**했다.)

**`max ITL = 929.91 ms` 이상치의 정체 — 첫 토큰도 컴파일도 아니다:**
per-request ITL 배열(`I2_shapeA_conc1.jsonl`, `--output-details`) 전수 재집계:

| req | n(ITL) | TTFT | median | max | >50 ms |
|---|---|---|---|---|---|
| 0 | 511 | 44.12 | 12.97 | 26.24 | — |
| **1** | 511 | 43.77 | 12.96 | **929.91 @ 토큰 인덱스 311** | 1건 |
| 2–5,7 | 511 | 42–45 | 12.96–12.97 | 13.3–16.2 | — |
| 6 | 511 | 42.13 | 12.96 | 56.57 @ 433 | 1건 |

- **첫 토큰 아님**(인덱스 311/511), **컴파일/캡처 아님**(warm-up 1요청 + 요청 0이 이미 완주), **배칭 아님**(동시성 1).
- **서버측 독립 확인**: 해당 요청 구간(21:55:05→13)의 초당 decode 줄 수 = `05:4, 06:78, 07:77, 08:77, 09:75, **10:8**, 11:77, 12:78, 13:37` ⇒ **21:55:10 한 초에 8 step만** 진행. 엔진 이벤트 루프가 ~0.9 s 멈춘 것이 맞다(클라이언트 아티팩트 아님).
- 그 초에 로그 이벤트 없음(prefill 없음, capture 없음, 분할 전환 로그 없음, 텔레메트리 미기록 boot). **원인 미확정**으로 등재한다.
- 결과: **"ITL 바닥은 13.0 ms"라는 요약은 max를 70× 밑돈다.** 8요청 4,088 샘플에서 >50 ms가 2건. SLO 술어(토큰-ITL p95)에는 안 걸리지만 **꼬리 사실로 병기**해야 한다.

---

## 8. 질문 7 — NPC-A(D2′ 오버라이드) 발화 여부

**미발화.** D2′의 3조건 중 1번(`report["failures"] == ["S_TIER_MISMATCH"]`)이 거짓이다 — 실제 `failures = ['TD1:B2_NO_CRASH','TD2:B2_NO_CRASH']`. 부수적으로 2번도 거짓(booted L boot = 2). ⇒ **NPC-A의 `S_pairs` 전수 전사 의무와 "재제출 1회 금지" 조항은 이 job에 발화하지 않는다.** (그럼에도 §4에 S_pairs 6쌍을 전수 전사해 두었다 — 무해한 초과 이행.)

---

## 9. 질문 8 — provenance 무결성 (전부 파일에서 직접 확인)

| 항목 | 요구 | 실측 | 판정 |
|---|---|---|---|
| commit | `b2b60e79…` | `provenance.txt:2` `commit=b2b60e791dcaff80f73f1c490a3f612721a02116`, `src_dirty:`(빈 값) | ✅ |
| D9(제출 전 커밋) | 4경로/7파일 | 커밋 `b2b60e7` (2026-09-13 **21:49:58**) = prereg 3 md + preflight 2 + sbatch + sync_engine_tree.sh + test 8파일. job 시작 **21:51:36** ⇒ **제출 전 커밋 충족** | ✅ |
| manifest | 25줄, `flashinfer_backend.py`가 끝 | `runtime_source_manifest.sha256` **25줄**, 마지막 줄 = `9181648bb265…  .../layers/attention/flashinfer_backend.py`; `models/nemotron_h.py 713333e87b4a…`, `configs/nemotron_h.py bcdface1473d…` 포함 | ✅ |
| 채점기 sha | `ec355e17…` | `provenance.txt:9` 및 현재 트리 `sha256sum` **양쪽 `ec355e171a66d68eab1300edcc7616a693cd55aa2c3fb5c251ee4b42eac50d30`** | ✅ |
| flashinfer | 0.6.10 | `provenance.txt:17` `flashinfer_version 0.6.10 /home01/ehmoon/.local/.../flashinfer/__init__.py` — **NPC-E 승계 필수**(버전 문자열일 뿐 내용 해시 아님, repo·venv 밖) | ✅(제한) |
| ctx/backend | 16384 / flashinfer | `provenance.txt:15` + `checks.server_args_across_boots`: `attention_backend ['flashinfer']`, `context_length ['16384']`, `random_seed ['1']`, `max_running_requests ['48']`, `disable_cuda_graph ['False']`, `triton_attention_num_kv_splits ['8']`, `pdmux_config_path` 단일 | ✅ |
| 사전등록 sha | `e832fc49…` 등 | `provenance.txt:32-37`의 5개 sha가 **현재 트리 파일 sha와 전부 일치**(사후 편집 없음) | ✅ |
| 규칙 핀 | rule v2 | `verdict_rule.txt` ≡ 채점기 docstring | ✅ |

**판정: CONFIRMED.**

---

## 10. ★질문 9 — 이 FAIL이 무엇을 차단하는가 (정확한 범위)

| 대상 | 차단? | 근거 |
|---|---|---|
| **P2 (Architecture: legacy fixed D24/D44 vs true dual fixed D24/D44)** | ★**차단** | 사전등록 §8 FAIL 행: "P2 캠페인 착수 금지". P2의 두 arm 중 하나가 **이 튜플에서 32동시 부하를 견디지 못한다**(2/2 boot 결정적 OOM) |
| **Claim D** | **등급 불변 = 미검증** · 추가로 **선결 1건이 FAIL로 확정** | Claim D는 P2의 결과다. 이 게이트는 P2의 **필요조건 하나**일 뿐이었고(NP-8), 그 하나가 이제 닫힌 게 아니라 **실패했다** |
| **λ0 캠페인 0단계(λ\* 측정)** | **이 FAIL로는 차단되지 않음** | λ\*는 **B1 = legacy + fixed**에서 잰다. warm-up·L1·L2가 legacy로 정상 동작했고 job 905835가 같은 모델·백엔드·arm을 12/12 boot시켰다(사전등록 §0). **단 별건으로 막힌다**: §6-3 F5 셀 B 반증 + §6-4 술어 결함 ⇒ **λ0 rev3는 현 상태로 제출 불가** |
| **P1 (계측 observer effect)** | **차단되지 않음** | P1의 4 arm은 legacy±telemetry와 **R1 observer**(`PDMUX_DUAL_WORKER=1`)이며 `PDMUX_TRUE_DUAL_WORKER`를 쓰지 않는다(`EXPERIMENT_ROADMAP.md` "P1 — 계측 observer effect") |
| **Claim E / hybrid 정책 트랙** | 이 FAIL과 **직교** | fixed 정책만 돌았다. 단 Claim E 캠페인이 true-dual arm을 포함한다면 같은 차단을 상속한다 |
| **재제출로 덮기** | ★**금지** | 사전등록 §9-D5: "**`FAIL`·`INCONCLUSIVE`·`NO_VERDICT_UNREALIZED`는 재제출로 덮어쓰지 않는다.** 원인 분해가 필요하면 그것은 **새 사전등록**의 대상이다." 재제출 1회 허용권은 `NO_VERDICT_INFRA` 전용이며 이 job은 그 라벨이 아니다 |
| **기존 정본** | **전부 불변** | HE0·정책 순위·stake #1·gate #13/#16·C2 인용정지·Zamba2/triton 동결(NP-4) — 이 job은 어느 것도 건드리지 않는다 |

---

## 11. ★"측정 실패로 강등하려는 통로" 점검 (게이트 #21과 그 역)

게이트 #21은 "**측정 실패를 게이트 실패로 라벨하지 마라**"이다. 그 **역**도 규율이다 — 진짜 게이트 실패를 측정 실패로 강등하지 마라. 강등 통로를 전수 조사했고 **전부 막혀 있다**:

| 통로 | 열리나? | 차단 근거 |
|---|---|---|
| 규칙선 2 `NO_VERDICT_INFRA`("boot 누락인데 크래시 없음") | ❌ | boot 누락 0(`B1_BOOTED` 4/4 True, `gen_*.json` 4개 존재), 그리고 로그에 **크래시가 있다** |
| §8 D2′ 오버라이드 | ❌ | `failures ≠ ['S_TIER_MISMATCH']` (§8) |
| §9 모드 (a) 부팅 실패 | ❌ | 4 boot 전부 health 통과·클라이언트 산출물 존재 |
| §9 모드 (b) OOM | ❌ | 그 행의 증상은 "**부팅 중** CUDA OOM"이다. 이 OOM은 서빙 중이다. D12 분기("**첫 scored boot**이 OOM")도 미충족(첫 scored boot = L1, 정상) |
| §9 모드 (c) warm-up 미가동 / (d) read-out timeout / (f) 벽시계 | ❌ | warm-up 정상, `I2_rc=0 I3a_rc=0 I3b_rc=0`, 21분 완주(한도 150분) |
| §9 모드 (g) 서버가 client 도중 사망 | ✅ **해당** — 그리고 처분이 "**rule 1/2 경로**"다. 로그에 traceback이 있으므로 **rule 1 = FAIL** | `BOOT_FAILURES.txt`의 두 줄이 정확히 이 모드 |
| "C는 진단 전용이니 무효" | ❌ | §2-2(규칙 문안이 mismatch counts로 한정) + §2-3 CF1(과결정) |
| "재제출해서 다시 본다" | ❌ | D5 (§10) |

**⇒ 이 `FAIL`은 게이트 실패이며, 측정 실패가 아니다.** 사전등록이 실행 전에 이 경로를 명시적으로 등록해 두었기 때문에 재량이 없다.

동시에 **역방향 과잉도 차단한다**: 이 FAIL은 "true-dual 아키텍처가 틀렸다"가 아니라 **"이 튜플·이 부하에서 true-dual 경로가 split-prefill 중 OOM으로 죽는다"** 이다(§3-4).

---

## 12. ★필수 병기 문안 (결과 문서·정본이 **문자 그대로** 승계)

> **NPC-A … NPC-J 중 이 결과에 실제로 걸리는 것**: **NPC-B**(§5, 발화) · **NPC-D**(기본 경로 명령 단위 동일성 주장 금지) · **NPC-E**(flashinfer 휠) · **NPC-F**(manifest 파급) · **NPC-G**(테스트 `BANNER` 합성 픽스처 인용 금지) · **NPC-I**(λ_inf의 arm 한정 — 단 §6-2에 의해 "D44" 부분은 **정정 필요**) · **NPC-J**(재제출 계수, 이 job에서는 재제출 0). **NPC-A 미발화**(§8) · **NPC-C**는 rev3에서 주석 교체로 해소(단 이전 판본 job 디렉터리에는 유효) · **NPC-H**는 제출 전 조건이라 결과 문서에는 기록만.

**N-1 (라벨)** — job 907959의 판정은 **`FAIL`**이며, 이는 verdict rule v2(2026-09-11, 데이터 관측 전 고정)의 기계적 산물이다. 감사자가 무수정 채점기로 **비트 단위 재현**했다. 실패 항목은 `TD1:B2_NO_CRASH`·`TD2:B2_NO_CRASH` 단 둘이고, 다른 모든 체크(B1·B3·B4·B5·B6·B7·B8·O_TIER)는 4 boot 전부 통과했다.

**N-2 (C층 데모션의 사정거리)** — "C층은 진단 전용"은 **"C층의 토큰 불일치 수가 판정에 들어가지 않는다"** 는 뜻이며, "C층 동안 일어난 서버 크래시가 판정에 들어가지 않는다"는 뜻이 **아니다**. `B2 NO_CRASH`는 티어 수식이 없는 퍼-boot 술어다. C층 request error를 제외해도 **로그의 traceback 1건 + `Scheduler hit an exception` 1건이 단독으로 FAIL을 낸다**(감사자 반사실 CF1).

**N-3 (귀속 한계)** — 지지되는 문장은 **"위 스코프 튜플에서, 동일한 14-seq / 10,125-token split-prefill 배치에 대해 true-dual boot 2/2가 `torch.OutOfMemoryError`로 죽었고 legacy boot 2/2는 같은(그리고 더 큰) 배치를 완주했다"** 뿐이다. **"true-dual이 (일반적으로) OOM으로 죽는다" · "true-dual의 메모리 피크가 legacy보다 크다" · "true-dual이 이 쌍에서 틀렸다"는 이 job 단독으로 쓸 수 없다**(사전등록 §8 FAIL 행). 기전은 미측정이다(메모리 계측 없음).

**N-4 (S/O)** — S16·O8에서 두 arm의 토큰 출력이 6/6·6/6 쌍 모두 불일치 0이었고 O1–O4가 32/32 probe-boot에서 확인됐다. **이것은 측정 기록이며 동등성 인증이 아니다** — 사전등록 §2의 인증문은 `PASS`에서만 허용되고 이 job은 `FAIL`이다.

**N-5 (C층 수치의 정체)** — `verdict.txt`의 `C cross-arm: 31`은 **토큰 분기 31건이 아니라 "TD 쪽 출력이 비어 있는 요청 31건"** 이다(31건 전부 `len [n, 0]`, `first_divergence=0`, 에러 시그니처 `RemoteDisconnected`). 마찬가지로 `C within-arm TD1-TD2: 0`은 **두 TD가 같은 31건에서 똑같이 빈 출력**이어서 나온 0이며 결정성의 증거가 **아니다**. L1-L2의 7은 legacy 자체 비결정성이고 **907032의 7/32와 비교 금지**(NP-5).

**N-6 (`unsafe_decisions=16`)** — 이는 텔레메트리의 `controller_decision.safe=False` 계수이며 **B2가 보는 로그 문자열 `unsafe split transition`(값 0)과 다른 양이다.** 16건 전부 `target=current=44`라 분할은 바뀌지 않았고(B8/O3가 346/346 idx 4 확인), 시각 분포상 **prefill∧decode 동시성의 표지**다(O층 8 + C층 8). 결함이 아니다.

**N-7 (I2의 분할 — ★NP-9/D6-ii 정정)** — **I2(동시성 1)의 TTFT 43.03 ms·ITL median 12.96 ms는 D44 값이 아니라 비분할(108 SM) 값이다.** 엔진은 prefill∧decode가 **동시에** 활성일 때만 D44(green idx 4)를 쓰고, 그 외에는 green context가 `null`인 idx 0/5에서 돈다(드라이버 read-out + 같은 job scored boot 텔레메트리 12,458 스냅샷 전수: idx0 11,725 / **idx4 592** / idx5 141). 동시성 1에서는 중첩이 구성상 불가능하다. **사전등록 §5-I2 (D6-ii)와 NP-9의 "D44 한정" 태그는 I2에 대해 거짓이며, I3에 대해서는 혼합비가 미측정이라 검증되지 않았다.** (λ0 rev1 판정서 `:151` "B=1은 pdmux 분할 미적용 구간"이 옳았다.)

**N-8 (★F5 / λ_inf(B) 인용 금지)** — 사전등록 §5-I3 (D4)의 셀별 재계산 레시피로 계산한 결과 **I3b(8192,64)의 셀별 `#running-req` 최댓값은 2**(I3a는 48)다. **F5는 셀 B에서 반증됐고, 등록된 처분에 따라 `λ_inf(B)=0.6956 req/s`를 포화 처리율/상한 앵커로 인용할 수 없다.** 하네스가 기록한 `I3_max_running_req.txt = 48`은 **전역 최댓값**이며 셀 A에서만 온 값이다. 미달의 원인은 **미확정으로 기록**한다(사전등록 D4가 상류 병목 단정을 금지).

**N-9 (λ_inf(A))** — `λ_inf(A)=3.0939 req/s`는 셀 A가 엔진 상한 48에 붙은 상태(decode 줄 3,880 중 2,814 = 72.5%가 48, `#queue-req` 최대 47)에서 얻은 **closed-loop 포화 처리율의 상한 프로브**이며 **λ\*가 아니다**(NP-3′). 이 값은 **legacy · warm-up boot · cudagraph ON**에서 측정됐고, **분할 혼합비는 미측정**이다(warm-up boot은 텔레메트리를 쓰지 않는다). λ0 rev1이 등록한 λ\*(A) 점추정 2.10–2.51보다 **23–47% 크다**(λ_inf ≥ λ\*이므로 모순은 아니다).

**N-10 (I3b vs 905835)** — +3.00%(0.69559 vs 0.67530; input tok/s 5,698 vs 5,532)는 **out 96→64 · ctx 8704→16384 동시 변경**과 같은 크기의 차이이므로 "closed-loop가 open-loop를 상회한다"의 증거로 쓸 수 없다.

**N-11 (I2의 꼬리)** — I2의 `max ITL = 929.91 ms`는 요청 1의 **토큰 인덱스 311**에서의 단발 정지이며(첫 토큰 아님·컴파일 아님·동시성 1), 서버 로그에서 해당 1초 구간의 decode step이 77→**8**로 떨어진 것으로 독립 확인된다. **원인 미확정.** "ITL 바닥 13.0 ms"만 인용하지 말고 이 꼬리를 병기하라.

**N-12 (F2′)** — §5의 문안을 그대로 승계한다(NPC-B).

**N-13 (스코프 배선, D13-i)** — `verdict.txt` 단독 인용 금지. `checks.server_args_across_boots`(§9 표)와 `provenance.txt`의 `model=`·`context_length=… attention_backend=…`·`instrument=`·`flashinfer_version` 줄을 **함께 전사**한다.

---

## 13. 확정에 필요한 정확한 실험 (게이트 #113 자기 적용: 실현가능성 함께 적는다)

### D1 — ★λ0 rev3 제출 전 필수 (GPU 0, 즉시 가능)
`lambda0_lambda_inf.py`의 조건 (b)를 **셀별 값**으로 바꾼다. 두 갈래 중 하나:
- (i) newpair 하네스가 `I3_max_running_req.txt`에 **`<cell> <max>` 두 줄**을 쓰도록 고친다(D4 레시피를 sbatch에 그대로 이식). 술어의 selftest가 이미 이 포맷을 시험하므로 **소비자 변경 0**. **단 이는 newpair 하네스 변경이므로 새 회차의 사전등록 대상이다**(현 job을 소급 수정할 수 없다).
- (ii) 술어가 `I_log_offsets.txt` + `srv_warmup.log`에서 **스스로 셀별 재계산**하게 한다. 산출물 두 개가 이미 job 디렉터리에 있으므로 **GPU 0·소급 적용 가능**. 권장.
**실현가능성**: 두 안 모두 파일 2개만 읽는다. 감사자가 이미 (ii)를 손으로 실행해 48/2를 얻었다. 회귀는 `test_lambda0_prereg.py` 범위.
**주의**: 이 수정 자체가 판정을 바꾼다(ANCHORED → FALLBACK). λ0 rev3 사전등록은 **그 결과를 실행 전에 등록**해야 한다(어느 분지로 갈지 이미 알고 고르면 T2 死因 재발).

### D2 — ★shape B의 포화 계기 재등록 (새 사전등록, GPU 0으로 설계)
F5의 `#running-req ≥ 48`은 **prefill 지배 shape에 대해 틀린 계기**다(구속은 `max_prefill_tokens=16384`). 새 사전등록이 등록할 것:
> "prefill 지배 셀의 서버-병목 판정은 `#running-req`가 아니라 **런 지속시간의 ≥90% 구간에서 `#queue-req ≥ K`**(K를 실행 전에 수치로 등록)로 한다. 두 계기를 **둘 다** 기록하고, 어느 쪽으로 판정할지는 shape별로 **실행 전에** 배정한다."
**이번 job의 원자료로 사전계산 가능**: I3b는 151개 prefill 줄 전부에서 `#queue-req`가 60 부근, 최대 62. K=10에서도 만족. **이 사실은 새 등록의 *예보*로만 쓰고, 현 job의 F5 실패를 무효화하는 데 쓰지 않는다.**

### D3 — ★★true-dual OOM 원인 분해 회차 (사전등록 필수, 추정 **0.3–0.5 GPU-h**)
사전등록 §8이 요구하는 "원인 분해 회차". 최소 설계:
- **arm**: `L`(legacy) × 2 boot, `TD`(true-dual) × 2 boot — 순서 `L TD L TD` 유지, 이 job과 **동일 스코프 튜플**, 동일 클라이언트(seed 20260911).
- **처치 축(하나만 변경, confound #10 회피)**: `mem-fraction-static` 사다리 `{0.82, 0.78, 0.74}` 중 **한 값씩 별도 회차**. 0.82에서의 결과는 이미 있다(이 job).
- **필수 신규 계측(engine-porter 선행 작업)**: 매 `runtime_snapshot`에 `torch.cuda.max_memory_allocated()` / `memory_reserved()`를 실어 **arm별 활성화 피크를 직접 측정**. 이것 없이는 A8이 영원히 보류다. ★**실현가능성 경고**: 이 필드는 현재 **존재하지 않는다** ⇒ D3는 **엔진 패치 + CPU 회귀 + 사전등록**이 선행하며 "다음 job"이 아니다.
- **예보(실행 전 등록)**: (F-a) TD 2/2가 10,036–10,137 토큰 split-prefill 배치에서 OOM한다 / (F-b) L의 같은 배치 피크 allocated가 TD보다 작다 / (F-c) mem 0.74에서는 두 arm 모두 완주한다.
- **판정 규칙**: (F-b)가 부호를 반대로 내면 "true-dual 메모리 피크" 가설은 **REFUTED**이고 원인은 다른 곳이다(예: 배치 형성 타이밍).
- **금지**: 이 회차를 "성능 비교"로 읽지 않는다(NP-2 계열).

### D4 — I2/I3의 분할 태그 확정 (추가 GPU **불필요하게 만들 수 있음**)
warm-up boot에도 `PDMUX_TELEMETRY_PATH`를 붙이면 `stream_index` 분포가 기록된다. **단 이는 사전등록 §6이 "채점 경로 비접촉"을 보증하기 위해 일부러 빼 둔 것**이므로, 별도 파일 경로(`instrument/tel_warmup.jsonl`)로만 쓰고 채점기가 그 이름을 열 수 없음을 테스트로 고정해야 한다(기존 35 케이스 패턴 그대로). **실현가능성: 높음, GPU는 다음 계측 회차에 무료로 편승.** 그 전까지 **"λ_inf는 D44에서 쟀다"는 쓸 수 없다.**

### D5 — I2의 929.91 ms 정지 (선택, 낮은 우선순위)
동시성 1·512 토큰 × 요청 수를 8→32로 늘려 재현율을 재고, `PYTORCH_CUDA_ALLOC_CONF`·노드 동거 여부를 기록. **현 상태로는 인용 시 "원인 미확정 단발 이상치"로만 병기.**

---

## 14. 신규 게이트 후보 (doc-steward 등재용, 번호는 정본이 부여)

1. **"진단 전용" 라벨의 정의역을 문안으로 적어라** — "C층은 진단 전용"이 *불일치 수*만 가리키는지 *그 층에서 일어난 모든 일*을 가리키는지 규칙이 구분하지 않으면, 결과 해석 시점에 **미등록 재량**이 생긴다. job 907959는 규칙이 괄호로 정의역을 적어 둔 덕에 재량이 0이었다. **긍정 사례**.
2. **퍼-셀 판정을 요구하는 예보는 퍼-셀 산출물을 만들게 하라** — F5는 "두 셀 각각"을 요구했는데 하네스는 **전역 max 1개**를 썼고, 다운스트림 소비자(`lambda0_lambda_inf.py`)가 그 1개를 `min(vals) ≥ 48`으로 읽어 **실패를 통과로 바꿨다**. 1차 감사가 지적했고 D4가 **문서에만** 반영된 결과다. **수리는 국소, 주장은 전역**(교훈 85)의 소비자 버전.
3. **포화 판정 계기는 shape의 병목 축과 맞춰라** — `#running-req ≥ 48`(decode 슬롯)은 prefill 지배 shape에서 구조적으로 도달 불가하다. 실제 구속은 `max_prefill_tokens`였다. "계기가 그 shape에서 발화 가능한가"를 등록 전에 확인하라(게이트 #200 계열).
4. **분할 실험의 수치에 D를 붙이기 전에 그 구간이 정말 분할 위에 있었는지 확인하라** — 이 엔진에서 green 분할(idx 1–4)은 **prefill∧decode 동시성에서만** 활성이고 그 외에는 `green_ctx_is_null=true`인 비분할 그룹에서 돈다. 동시성 1 프로브는 **구성상 D를 밟을 수 없다.** X1 결과 감사의 "양성대조가 운영점 위에 떨어졌는지 확인하라"의 자매 항목이며, 이번에는 **사전등록이 정본 문장(λ0 `:151`)과 충돌하는 방향으로 틀렸다**.
5. **`verdict.txt`가 인쇄하는 이름이 게이트가 보는 양과 같은지 확인하라** — `unsafe_decisions`(텔레메트리 `safe=False` 계수, TD 16) ≠ B2의 `unsafe split transition`(로그 정규식, 0). 이름 충돌이 "16건 unsafe인데 왜 통과?"라는 오독을 유도한다.
6. **에러로 비어 버린 출력의 "불일치 0"은 결정성이 아니다** — `TD1-TD2 C = 0`은 31건의 빈 출력이 서로 같아서 나온 값이다. 비교 함수가 `error` 레코드를 등가류에 넣는 한, 크래시한 arm은 자기 자신과 항상 일치한다(항등식, 교훈 9 계열).

---

## 15. 자기 적용 (게이트 #113) · 반증 실패 공시

- **감사자 처방의 실현가능성**을 §13에 전부 적었다. 특히 **D3는 현재 존재하지 않는 텔레메트리 필드를 요구**하므로 "다음 job"이 아니라 엔진 패치 선행이며, 그 사실을 처방 안에 박았다. **D1(ii)만이 GPU 0·즉시 실행 가능**하다.
- **직전 판정서의 지적을 상수로 승격하지 않았다**(게이트 #110): 1차 판정서가 지적한 F5 결함(전역 vs 셀별)을 인용만 하지 않고 **원자료로 독립 재계산**해 48/2를 얻었으며, NP-7′의 "48은 연역"도 이 job의 5 boot 배너로 재확인했다. NPC-B의 투영치는 **재감사 문서와 사전등록 §7 양쪽에서 문자로 확인한 뒤** 실측과 대조했다.
- **반증 실패 공시**: 라벨 `FAIL`의 규칙 적합성(§2)·provenance(§9)·S/O 측정 사실(§4)·O1–O4 32/32(§4)·F2′ 충족(§5)·F3/F4 충족(§7 주변)에 대해서는 **반증을 만들지 못했다**. 다만 §4의 S/O 사실은 **§13-D3의 원인 분해 회차 전까지 "동등성이 인증됐다"로 승격되지 않는다** — 규칙이 PASS를 내지 않았기 때문이다.
- **새 성능 판정 0건.** HE0·정책 순위·stake #1·gate #13/#16·Claim D 등급(미검증)·Zamba2/triton 동결 — 전부 불변.
- **GPU 신규 지출 0.** 이 트랙(R2 correctness) 누적은 0.43 → **0.786 GPU-h**(job 907959의 21분 21초 = 0.356 GPU-h 포함). longctx_conflict 트랙 장부(15.42 GPU-h)와는 별개.

---

### 관련 파일 (절대경로)

- 아티팩트: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/job_907959/` (+ `instrument/`)
- 사전등록: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/newpair_prereg/PREREG_NEWPAIR_2026-09-13.md`
- 규칙층 판정서: 같은 디렉터리 `VERDICT_newpair_rules_2026-09-13.md` · `VERDICT_newpair_rev2_2026-09-13.md`
- 채점기·하네스: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/r2_correctness_check.py` · `r2_correctness.sbatch`
- ★수리 대상: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_eval/lambda0_prereg/lambda0_lambda_inf.py` (조건 (b) — §6-4)
- 대조 원자료: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/c_905835/cell_d44_r2_o96.json` · `bench_d44_r2_o96.log` · `srv_d44_r2_o96.log`
- 엔진 근거: `/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/multiplex/multiplexing_mixin.py:904-971`(분할 선택) · `:1206-1262`(true-dual submit/수확) · `.../multiplex/controller.py:87-101`(`FixedPolicy`, `safe=safe_boundary`) · `.../layers/attention/mamba/mixer2_rms_norm_gated.py:97`(OOM 지점)
