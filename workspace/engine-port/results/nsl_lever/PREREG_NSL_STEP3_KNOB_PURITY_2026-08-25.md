# 사전등록 — NSL **③**(손잡이 순화): cap 축 대조에서 **KV 예산을 분리한다**

2026-08-25 · **새 GPU 지출 0**(규칙 + 하네스 규약. 검증은 다음 캠페인 부팅에 편승) ·
**새 성능 판정 0건** · HE0 불변.
근거: `audit_nsl_p0_capbind_2026-08-24/VERDICT.md` §⑥ 권고 **3-(a)**(死因 **B3** 수리) +
`RESULT_NSL_EA_892561_2026-08-25.md`(그 처방이 **작동함을 실증**).

> ★**순서**: ③이 ②보다 **먼저**다. `MEMO_NSL_STEP2_DECISION_2026-08-24.md` §4 —
> *"**②만 사는 것은 값이 없다**"*(B3가 남아 arm 비교가 서지 않으므로). E-A가 ③의 처방이
> 실제로 KV 예산을 cap에서 떼어냄을 보였으므로 **③을 먼저 확정하고 그 위에 ②를 얹는다.**

> ★**인용 재확인(2026-09-01)**: CP-0 P1 `chunk_probe_scheduler_hook.patch`(2026-08-28)가
> `managers/scheduler.py`에 순 +13줄을 삽입해 §4의 `managers/scheduler.py` 인용(`:2369`·`:2434`·
> `:2476`, 전부 ③이 아닌 §4 "다음은 ②다" 참조)이 `:2382`·`:2447`·`:2489`로 밀렸다. §2.2가 이미
> 기록한 감사→이 문서 드리프트(`:155→:228` 등)와 **같은 종류의 재확인**이며, ③ 자체의 규칙
> R1–R6·판정은 불변.

---

## 0. 이 문서는 무엇인가 / 아닌가

- **이다**: cap 축을 쓰는 **모든** 앞으로의 NSL 셀이 지켜야 할 **부팅 규칙 + 검증 게이트**의 사전등록.
- **아니다**: 판정이 아니다. ★**`"cap이 문다 / 안 문다"` 는 감사 B1로 여전히 무효**이고 이 문서는
  거기에 답하지 않는다. 지연·처리량·goodput 문장도 **전면 금지**(§6).
- **닫지 않는다**: B3를 **좁히지만 닫지 않는다**(§2). E-A가 이미 그렇게 적었다.

## 1. 감사가 지시한 것 (원문)

> **3. ★★B3 수리 = 설계 형태 변경.** 두 선택지뿐이다:
> **(a)** 두 cap arm 모두 `--max-mamba-cache-size`를 **동일 값으로 고정**하고 실현
> `max_total_num_tokens`·`max_mamba_cache_size`·`max_running_requests`를 **배너에서 읽어 셀마다 기록**
> (= rev3 P1 부활 + B7 수리), 또는 **(b)** cap 축 대조를 포기하고 단일 cap에서 rate 축만 돌린다.

★**(a)를 채택한다.** 근거는 E-A: `c48m96`(cap 48 + mamba 96)의 KV 예산이 `c96`과 **오차 0으로 동일**
(308,727)하고 `c48`(327,601)과 다르다 ⇒ **KV 예산은 cap이 아니라 mamba pool을 따른다.**

## 2. ★ 무엇이 순수해지고 **무엇이 안 되는가** (코드 근거)

`--max-running-requests`가 이 빌드에서 움직이는 것들:

| # | 경로 | mamba 고정 후 | 근거 |
|---|---|---|---|
| 1 | **admission cap** (원하는 축) | 그대로 | — |
| 2 | `max_mamba_cache_size` (미지정 시 `cap // dp`) | ★**끊긴다**(명시 지정) | `model_runner_kv_cache_mixin.py:228` ★**직접 확인**(감사 B3는 `:155` [HIST] 로 적었다 — §2.2) |
| 3 | **attention KV 토큰 예산** (`total_rest − mamba_state_memory`) | ★**끊긴다** — mamba pool만 따름 | `:250-255` ★**직접 확인**(감사는 `:249-254` [HIST]) + **E-A C1 대조(오차 0)** |
| 4 | `req_to_token_pool` 크기(`max_num_reqs`) | ✗ **남는다** | `:404` `_init_pools` ★직접 확인 |
| 5 | ★**클램프** `max_num_reqs = min(max_num_reqs, max_mamba_cache_size // ratio)` | ✗ **남는다 — 그리고 방향이 뒤집힌다**(§3 R2) | `model_runner_kv_cache_mixin.py:870-874` |

⇒ ★**B3는 닫히지 않는다.** 얻는 것은 **가장 큰 항(3)의 제거**이고, 그 크기는 E-A가 쟀다 —
mamba를 고정하지 않으면 cap 48→24 대조가 KV 예산을 **+9,438 토큰(+2.9%)** 함께 끌고 간다.

### 2.2 ★ 인용 줄번호 정정 (승계 검증)

감사 B3가 적은 줄번호 두 개는 **현재 트리와 맞지 않는다**. 이 문서는 **직접 확인한 줄만** 쓴다:

| 감사가 적은 것 | 현재 트리 실측 | 내용 |
|---|---|---|
| `:155` [HIST] | ★`model_runner_kv_cache_mixin.py:228` | `server_args.max_mamba_cache_size = server_args.max_running_requests // (…)` |
| `:249-254` [HIST] | ★`:250-255` | `mamba_state_memory = max_mamba_cache_size * mamba_cache_per_req … return total_rest_memory − mamba_state_memory` |

★**내용은 감사가 서술한 그대로이고, 위치만 다르다** — 판정에 영향 없음. `sync_engine_tree.sh`가
설치하는 `mamba2_pure_ssm_arch.patch`가 이 파일을 건드리므로 줄이 밀릴 수 있다.
★**교훈 #80 형태**(규칙 문서 안의 출처를 다음 판본이 재검증 없이 승계한다)를 여기서 끊는다.

### 2.1 ★ 새로 확인한 코드 사실 — 클램프가 **fix를 배신할 수 있다**

`model_runner_kv_cache_mixin.py:862-875`:

```
max_num_reqs = min(max_running_requests // dp, estimated)      # estimated ∈ [2048, 4096]
if mambaish:  max_num_reqs = min(max_num_reqs, max_mamba_cache_size // ratio)
```

`ratio = _calculate_mamba_ratio()`(`:390-402`)는 **`--disable-radix-cache`면 1**, 아니면
`MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 3`(+ extra buffer 0/1/2).

⇒ ★★**mamba pool을 낮게 고정한 채 cap을 올리면 실현 cap이 조용히 잘린다.**
radix cache가 **켜져 있으면** `M = 96` 고정은 실현 cap을 **32로** 자른다(96 // 3).
E-A가 실현 cap = 요청 cap을 관측한 것은 **`--disable-radix-cache`(ratio 1)** 부팅에서였다.
**이 사실은 이 사전등록에서 처음 등재된다** — 규칙 R2가 그것을 막는다.

## 3. ★ 등록 규칙 (앞으로의 모든 cap-축 셀에 구속)

| | 규칙 |
|---|---|
| **R1** | **모든 셀**이 `--max-mamba-cache-size M`을 **명시**한다. 미지정 셀은 cap 축 대조에 **쓸 수 없다** |
| **R2** | ★`M ≥ max(cap arms) × ratio`. `ratio = 1` **iff** `--disable-radix-cache`, 아니면 `3 + extra`. 이 부등식은 **부팅 전에 검사**한다 |
| **R3** | 셀마다 배너에서 `max_mamba_cache_size` · `max_total_num_tokens` · **실현** `max_running_requests` 3개를 읽어 기록한다. **하네스는 이미 있다** — `nsl_ea_analyze.py` + `nsl_ea_capbanner.sbatch`(job 892561에서 완주) |
| **R4** | ★**실현 cap ≠ 요청 cap** ⇒ 그 셀은 `CAP_CLAMPED` = **MEASUREMENT 조건**, 판정에서 제외 |
| **R5** | ★**셀 간 `max_total_num_tokens`가 다르면** `KV_BUDGET_UNEQUAL` ⇒ 그 셀 집합으로 **cap 축 대조 금지**(R1·R2를 지켰다면 발화하지 않아야 한다 — 발화하면 §2 표에 없는 경로가 있다는 뜻이므로 **조사 대상**) |
| **R6** | `M` 값 자체를 **자유 모수로 등재**하고 캠페인 문서에 적는다. ★`M`은 KV 예산을 정하므로 **arm이 아니라 상수**여야 한다 |

★**R5가 이 규칙 집합의 자기검사다** — R1·R2를 지키고도 KV 예산이 갈리면 §2 표가 불완전한 것이고,
그때 규칙을 고치는 것이지 결과를 고치는 것이 아니다.

## 4. 이 문서가 **사지 않는 것**

- ✗ **B1(점 술어 무효)** — `running_bs ≥ cap`은 참 술어가 아니다(`can_run_list`가 telemetry에 없음).
  ①이 구간 `T ∈ [0.080, 0.119]`로 묶었을 뿐이다. **점 술어는 ②(사이트별 카운터)로만 산다.**
- ✗ **귀속**(cap 경로 vs KV 경로) — reason 문자열이 두 경로를 뭉갠다(감사 B10). ②의 몫.
- ✗ 어떤 **성능·SLO·정책** 문장도.

⇒ ★**③ 다음은 ②다**: `managers/scheduler.py:2382` / `:2447` / `:2489` 세 사이트에 기본 OFF·arm 무관 카운터
(+ 발화 시점의 `running_bs`·`len(can_run_list)`). `PDMUX_HOLB_PATH` 선례를 따른다.

## 5. 가격

**새 GPU 0.** 규칙(R1–R6)과 하네스 규약뿐이고, 하네스는 E-A에서 이미 작성·완주됐다.
검증은 **다음 cap-축 캠페인의 부팅에 편승**하며, 그 캠페인의 사전등록이 R1–R6을 인용해야 한다.
★인용하는 기존 결과의 지출: **0.103 GPU-hr**(job 892561, 이미 집행).

## 6. ★ 쓰면 안 되는 문장

- ✗ ★★★ *"③이 B3를 닫았다"* — **좁혔을 뿐**(§2 표의 4·5가 남는다).
- ✗ ★★★ *"cap이 문다 / 안 문다"* — B1로 **점 술어 무효**. ③은 여기에 답하지 않는다.
- ✗ ★★ *"mamba를 고정하면 실현 cap은 요청 cap과 같다"* — **`--disable-radix-cache`(ratio 1)에서만**.
  radix cache가 켜지면 `M // 3`으로 **잘린다**(§2.1).
- ✗ ★★ *"슬롯당 393 토큰"을 다른 모델·ctx·mem-frac으로 이식* — **Zamba2-2.7B · ctx 4096 ·
  mem-frac 0.82 한정**(E-A §4).
- ✗ ★ *"③을 샀으니 NSL-1이 선다"* — **②가 남는다**(MEMO §4의 (b) 계열 가설).
- ✗ ★ *"E-A가 cap 축 대조를 정당화했다"* — E-A는 **요청 0건**이라 어떤 arm 비교도 하지 않았다.
- **승계**: *"NSL-1이 admission 축을 쟀다"* · *"cap 축이 무력함이 확인됐다"* · *"rev3이 규칙층을
  통과했다"* · HE0 · gate #13/#16 *"닫았다"* 금지 · switch-cost *"닫았다"* 금지 · C2 인용정지 (a)(b).
