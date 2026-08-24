# 감사 판정서 — **NSL P0 cap-binding 프로브** 규칙층 감사 (2026-08-24, claims-auditor, 게이트 #34 1단)

대상: `PREREG_NSL_P0_CAPBIND_2026-08-24.md`
**GPU 지출 0 · 제출 0건 · 새 성능 판정 0건 · 등급 변경 0건 · 정책 순위 변경 0건 · HE0 불변 · 대상 트리 수정 0건.**
선행 판정서 3건(rev1·rev2·rev3)은 **보존, 무효화 아님.**

---

## ① 단일 판정

## **`NO-GO` — 死因 3건(B1·B2·B3), 차단 B4–B13.**

**死因이 있다.** 세 개 모두 *설계 형태*를 바꿔야 하는 것이고, 국소 수정으로 닫히지 않는다.

| 판정 질문 | 답 |
|---|---|
| `CAPBLOCK`이 정말 cap 경로를 재는가 | ★★**아니다.** 등록된 항등식은 이 부팅 설정에서 **한 번도 실행되지 않는 분기**의 가드다(B1) |
| `CAP_NEVER_BINDS`가 등재 가능한 음성인가 | ★★**아니다.** 등록되지 않은 추정기 선택 하나가 **저장소에 이미 있는 데이터에서 판정을 X 문턱 양쪽으로 뒤집는다**(B2) |
| 감사가 지시한 것을 승계했는가 | ★**아니다.** 감사는 **코드 사이트별 발화 계수**를 지시했는데 프로브는 **상태 대리변수**로 치환했다(B1·B5) |
| 이 프로브가 새 죽음을 만들었는가 | ★**그렇다.** cap 24 arm은 cap·req pool·mamba pool·attention KV 예산을 **동시에** 바꾼다(B3 = confound #10) |

---

## ② 세 코드 사실 검증 — ★어느 트리를 봤는가

★**전부 실제로 도는 엔진 트리 `/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/` 에서 직접 확인했다.** 오버레이(`workspace/engine-port/src/`)는 **교차 검증용으로만** 썼다:

```
diff -q .../src/multiplex/dual_worker.py        .../srt/multiplex/dual_worker.py        → IDENTICAL
diff -q .../src/multiplex/multiplexing_mixin.py .../srt/multiplex/multiplexing_mixin.py → IDENTICAL
```
`managers/scheduler.py`는 **엔진 트리에만** 존재(오버레이는 holb 패치만 배포) ⇒ 어제의 "틀린 트리 grep" 결함(`kernel_mech/NVTX_EVIDENCE_CORRECTION_2026-08-24.md`) 형태는 이 판정서에 없다.

### 사실 1 — `get_num_allocatable_reqs` 항등식 ⇒ ★**REFUTED (死因)**

인용 줄번호는 **전부 정확**하다:
```
scheduler.py:2296-2300  get_num_allocatable_reqs(running_bs) = pp_max_micro_batch_size − running_bs   ✓
scheduler.py:668-671    pp_max_micro_batch_size := max(max_running_requests // pp_size, 1)            ✓
scheduler.py:2364-2369  get_num_allocatable_reqs(...) <= 0 ∧ chunked_req is not None ⇒ batch_is_full  ✓
scheduler.py:2432-2434  len(adder.can_run_list) >= get_num_allocatable_reqs(running_bs) ⇒ True        ✓
scheduler.py:2468-2476  res == AddReqResult.NO_TOKEN ⇒ True                                           ✓
```
★**그러나 §1의 항등식은 두 cap 경로 중 하나에만 성립하고, 그 하나가 이 설정에서 죽은 코드다.**

* **`:2364` 경로**(항등식이 성립하는 유일한 곳)는 가드가 `and self.chunked_req is not None`이다.
  등록 하네스는 `--chunked-prefill-size -1`이고 `scheduler.py:890-891`이
  `chunked_prefill_size <= 0 ⇒ self.chunked_prefill_size = None`,
  그러면 `schedule_policy.py:399,703` `rem_chunk_tokens is None  # chunked prefill is disabled`
  로 `new_chunked_req`가 **절대 세팅되지 않는다** ⇒ `self.chunked_req ≡ None`
  ⇒ ★**`:2369`는 이 부팅에서 실행 불가능하다.**
* **살아 있는 cap 경로는 `:2433`뿐**이고 그 조건은
  ```
  len(adder.can_run_list) >= cap − running_bs   ⟺   running_bs + |can_run_list| ≥ cap
  ```
  ★`|can_run_list|`가 **누락**됐다. `running_bs ≥ cap`은 이 조건의 **진부분집합**(`|can_run_list| = 0`인 특수해)이다.
* 추가로 `batch_is_full`은 **래치**다. `:2353`이 `batch_is_full or 큐 빔`이면 즉시 `return None`하고
  (chunked_req가 항상 None이므로 그 다리는 무력), 리셋은 `:2239`(last_batch 축소)·`:2565`·`:2630`·`:3340`
  뿐이다. ⇒ 스냅샷이 읽는 reason 문자열은 **몇 스텝 전에 다른 `running_bs`에서 세워진 래치**일 수 있는데,
  `cap_bound(s)`는 **그 순간의** 카운터다. **두 양의 시각이 다르다.**

⇒ ★★**H2와 동형이다**: *"측정된 적 없는 사건"* 을 *"관측 가능한 인접량"* 으로 치환하고 동치라고 선언했다. 편향 방향도 한쪽이다 — 진짜 cap 차단이 `running_bs < cap`에서 일어나면 그것은 `KVBLOCK`으로 오분류되어 **`CAP_NEVER_BINDS`를 쉽게 만든다**(게이트 #9의 귀무편향형).

### 사실 2 — telemetry만으로 관측자가 켜지는가 / 아키텍처를 안 바꾸는가 ⇒ **CONFIRMED (단, 게이트가 공허)**

```
multiplexing_mixin.py:85-88  trace_path = os.environ.get("PDMUX_TELEMETRY_PATH", os.environ.get("PDMUX_DUAL_WORKER_TRACE",""))
multiplexing_mixin.py:93     self.dual_worker_trace_path = trace_path
multiplexing_mixin.py:468-476 _dual_worker_sync: if not (dual_worker_enabled or dual_worker_trace_path): return
multiplexing_mixin.py:546-550 architecture = "true_dual" if true_dual else ("r1_observer" if dual_worker_enabled else "legacy")
```
✓ 켜진다. ✓ `architecture == "legacy"`로 남는다. ✓ **스케줄러 상태를 안 바꾼다** — `observe_scheduler`
(`dual_worker.py:566-602`)는 scheduler 속성을 **읽기만** 하고 자기 `coordinator` dict과
`arbiter.stream_index`(현재값 재확인 = no-op)만 만진다. `Req.finished()`는 순수 읽기
(`schedule_batch.py:936-938`). **관측자 효과 반증 시도 실패 — 인정.**

★단 **§2.1 게이트 0의 `architecture=="legacy"` 검사는 거의 공허**하다: 자기가 설정한 env를 되읽는 것뿐이다.
그리고 §1.1의 괄호 주석 *"(단일 워커)"* 는 오해를 부른다 — 이벤트 루프는 여전히 `event_loop_pdmux`
(`scheduler.py:3525-3529`, `_dual_worker_sync` 호출부는 `multiplexing_mixin.py:965,997,1242`로 **전부 pdmux 루프 안**)이다. "legacy"는 *관측자 모드* 라벨이지 아키텍처가 아니다(B13).

### 사실 3 — 필드가 스냅샷마다 나오는가 ⇒ **CONFIRMED**

```
dual_worker.py:591-601  blocked := queue_depth>0 ∧ prefill.active_batch_size==0
                        reason  := "shared_batch_capacity" if running_batch.batch_is_full else "scheduler_admission_pending"
dual_worker.py:615,616,618  prefill_admission_blocked / _block_reason / decode_running_batch_size
schedule_batch.py:1476-1477 ScheduleBatch.batch_size() == len(self.reqs)     ⇒ decode_running_batch_size == len(running_batch.reqs) ✓
```
★**그러나 부수 사실 2개가 등록되지 않았다**: (i) `reason`은 `blocked`가 참일 때만 채워진다 ⇒
**`batch_is_full`을 독립적으로 관측할 수 없다**; (ii) `prefill.active_batch`는
`getattr(scheduler,"split_prefill_batch",None)`(`dual_worker.py:573`)인데, 두 샘플 지점 중
`:997`은 `update_split_prefill_batch`(`:1004`) **이전**이라 그 지점의 `active_batch`는 구조적으로 비어 있기 쉽다 ⇒ §4-7이 주장한 *"보수적(과소계수)"* 부호가 지점마다 다르다(B10).

---

## ③ 합격 기준 6개 판정

| # | 기준 | 판정 | 근거 |
|---|---|---|---|
| 1 | 결정량이 제조되지 않는가 (#9) | ★**FAIL** | 결정량이 **죽은 분기의 가드**(B1). 오분류 편향이 한쪽(`CAP_NEVER_BINDS`)으로 고정. 추가로 실데이터에서 `CAPBLOCK ≡ CAPSAT ≡ blocked∧shared`가 4부팅 중 3부팅서 **집합으로 완전 일치**(B10) ⇒ 분리 절이 지지집합 위에서 항진 |
| 2 | 구속성을 옳은 양으로 재는가 (#68) | ★**FAIL** | 명목은 *막힌 사건*, 실제는 **`decode_running_batch_size ≥ cap`이라는 점유 지시함수**로 붕괴(④-B10 실측). H2가 죽인 *인구형 대리변수*가 **모집단만 바꿔 재발** |
| 3 | cap 경로와 KV 경로가 분리되는가 | ★★**FAIL(死因)** | (a) 배치 크기로 안 갈린다(B1). (b) 이 빌드에서 `--max-running-requests`가 **req pool·mamba pool 크기 그 자체**이므로 두 경로의 경계가 같은 자리에 놓인다(B3) |
| 4 | 측정 실패 vs 판정을 가르는 대조 (#21) | ★**FAIL** | §2.2 대조는 **깨질 수 있는 것을 시험하지 않는다**. 항등식이 틀렸을 때 두 arm이 **같은 방향으로** 오측정되므로 대조는 통과한다(B5). 반대로 `N_min=200`은 cap이 무는 셀에서 **간신히** 넘어 측정실패 라벨이 물리 신호에 붙을 위험(B9) |
| 5 | 가격·자유 모수·중단 규칙 | ★**FAIL** | *"자유 모수 전수 7개"* 가 **거짓**(최소 6개 누락, 그 중 하나가 판정을 뒤집음, B6). 부팅 상수 `145.8 s`는 rev3 감사 **H10이 이미 차단한 값**이고 실측치가 같은 저장소에 있다(**32.8 s**, n=10, B8) |
| 6 | 인용정지 준수 | ✓ **PASS** | `check_citation_stops.py --file <prereg>` → `180 added line(s), 0 violation(s), 16 rules OK`. 수동 확인: C2 (a)(b)·job 870295–870301 계열·gate #13/#16 인용 **0건**. rev1의 F4 위반은 **재발 안 함 — 인정** |

---

## ④ 차단 상세

### ★★★ B1 (死因) — §1의 항등식은 **죽은 분기의 가드**다
②-사실1 참조. **수리 방향**: 항등식을 버리고 감사가 실제로 지시한 것으로 돌아가라 —
`scheduler.py:2369` / `:2434` / `:2476` **세 사이트에 각각 카운터를 심는 엔진 패치**
(감사 원문: *"`:2433-2434` vs `:2472-2476` — 후자는 KV 경로이므로 **분리 계수**"*).
`§1.1 엔진 패치가 필요 없다`는 절 제목 자체가 死因이다.

### ★★★ B2 (死因) — 추정기가 미등록이고, **이미 있는 데이터에서 판정이 뒤집힌다**

§2는 `CAPBLOCK := 시간비{ s : … }`라고 쓰는데 대상은 **스냅샷 집합**이다. §4-5가 문제를 인지하고도
(*"스냅샷 간격 분포를 병기한다"*) **추정기를 등록하지 않았다.**

★저장소에 이미 있는 **동일 arm·동일 플래그·동일 cap·telemetry-ON** 로그
(`results/slo_sched/g16_blk{1..4}_d44_boot1_*_telemetry.jsonl`, Zamba2-2.7B / ctx4096 / `--max-running-requests 48` /
`--chunked-prefill-size -1` / `--enable-pdmux` / `architecture=legacy`)로 프로브의 결정량을 **그대로 계산**했다:

| 부팅 | `CAPBLOCK` (스냅샷 **개수** 가중) | `CAPBLOCK` (**시간** 가중, 다음 스냅샷까지 간격) |
|---|---|---|
| blk1 | 0.00147 | **0.08784** |
| blk2 | 0.00148 | **0.08675** |
| blk3 | 0.00147 | **0.08576** |
| blk4 | 0.00139 | **0.07938** |

★★**등록 문턱 `X = 0.05`를 사이에 두고 정확히 반대편이다.** 개수 가중이면 4/4 부팅 전부
`CAP_NEVER_BINDS`(문턱의 1/34), 시간 가중이면 4/4 전부 `CAP_BINDS_SOMEWHERE`. **58배 차이.**

기전(같은 파일에서 실측):
* 스냅샷 간격은 **균일하지 않다** — p50 = 1.95 ms, p99 = **213.8 ms**, max = 658.5 ms.
* 20 s 구간별 스냅샷 밀도가 **118배** 변한다 — 유휴 구간 9,181개/20 s(459/s) vs
  **배치가 cap에 앉은 구간 78–96개/20 s(3.9–4.8/s)**.
* ⇒ 개수 가중 "시간비"는 **관심 사건과 반상관**이다. `decode_running_batch_size == 0`인 스냅샷이
  **97.5 %**(시간 가중으로는 25.9 %)를 차지한다.

★이 문제는 **엔진 소스 자신이 이미 경고**하고 있다(`multiplexing_mixin.py:508-520`):
*"any 'fraction of snapshots' statistic … is biased … **TIME-WEIGHTED** statistics (weight each snapshot by the gap to the next one) … are unaffected."*
⇒ 교훈 #61/#63/#72(**보고 필드의 의미를 코드와 대조하지 않고 추정하지 마라**)의 재발이다.

> ★★**위 수치는 감사 진단용이며 성능·물리 판정이 아니다.** B1 때문에 술어 자체가 무효이므로
> *"cap이 문다/안 문다"* 어느 쪽으로도 인용 금지.

### ★★★ B3 (死因) — `--max-running-requests`는 이 빌드에서 **순수 admission 손잡이가 아니다** (confound #10)

`model_runner_kv_cache_mixin.py`:
```
:223-229  elif server_args.disable_radix_cache and server_args.max_running_requests is not None:
              server_args.max_mamba_cache_size = server_args.max_running_requests // dp
:249-254  mamba_state_memory = max_mamba_cache_size * mamba_cache_per_req ;  return total_rest_memory − mamba_state_memory
:404-447  _init_pools: max_num_reqs = self.max_running_requests ; req_to_token_pool(size=max_num_reqs, mamba_size=max_mamba_cache_size)
```
등록 하네스는 `--disable-radix-cache`를 쓰고 `--max-mamba-cache-size`를 **주지 않는다**.
⇒ `cap`을 48 → 24로 내리면 **동시에**:
1. admission cap 24, 2. `req_to_token_pool` 슬롯 24, 3. **mamba state pool 24**,
4. mamba 메모리가 반환되어 **attention KV 토큰 예산(`max_total_num_tokens`) 증가**.

⇒ ★§2.2 양성대조의 두 arm은 **네 개가 동시에 다르다.** 그리고 하필 그 넷이
*"cap 경로 vs KV 경로"* — 프로브가 분리하겠다는 바로 그 두 축이다. 대조가 발화해도
**cap 때문인지 pool 때문인지 원리적으로 귀속 불가**.
★부수: mamba pool 고갈은 graceful `NO_TOKEN`이 아니라 **assert(크래시)** 다
(`mem_cache/memory_pool.py:542-546`) ⇒ cap이 pool의 **보호막**이다. cap과 pool 경계가 같은 자리에 있다.
★rev3 §5의 **P1**(`--max-mamba-cache-size 96` 고정 + 남은 attention KV pool 값 기록)이 정확히 이 문제를
겨냥해 이미 등록돼 있었는데 **P0가 그것을 버렸다.**

### ★★ B4 — **격자의 절반이 이미 측정돼 있다**(선행 등록 승계 실패 + 예산 정직성)

`prefill_admission_block_reason`을 담은 telemetry 파일이 저장소에 **989개**. 그 중
`g16_*_d44_boot1_*`은 arm·모델·ctx·cap·전 서버 플래그·telemetry 설정(`trace_every` 기본 32)이
프로브 등록값과 **일치**하고 rate 3↔12를 3라운드 돈다. ⇒ 프로브가 사려는
`cap 48 × {rate 3, rate 12}` 셀은 **GPU 0원에 이미 있다**(위 표가 그 증거).
사전등록은 이것을 **한 줄도 인용하지 않는다.** ⇒ 0.25 GPU-hr 중 상당분이 **재구매**다.

### ★★ B5 — 양성대조가 **깨질 수 있는 것을 시험하지 않는다** (게이트 #21)

§2.2는 *"cap 24의 CAPBLOCK > cap 48의 것"*. 이 대조가 잡는 실패는 *"둘 다 0"* 뿐이고
그건 §2.1 게이트 0이 이미 잡는다. **실제 위험(B1의 항등식 오류, B2의 추정기)** 은 두 arm에
**같은 방향으로** 작용하므로 대조는 그대로 통과한다. 게다가 대조 자체가 준항진에 가깝다 —
배치는 자기 cap에서 잘리므로 더 작은 cap이 더 오래 자기 cap에 앉는 것은 거의 구조적이다
(실측 근거: d44 cap48에서 `decode_running_batch_size` 분포는 **0(97.5 %) / 6–12 덩어리 / 48 스파이크**의
삼봉이고, 48 스파이크는 **HI(rate 12) 창에서만** 나온다).
★**진짜 양성대조**: (i) 자연 배치보다 명백히 작은 `cap = 4` 또는 `8` 셀 하나(그 셀에서
`CAPBLOCK`이 크지 않으면 측정이 깨진 것), 그리고 (ii) **사이트별 엔진 카운터 vs 상태 대리변수의 교차 검증**.

### ★★ B6 — *"자유 모수 전수 (7개)"* 가 **거짓** (최소 6개 누락)

| 누락 모수 | 왜 치명적인가 |
|---|---|
| ★**추정기**(개수 가중 vs 시간 가중) | **판정을 뒤집는다**(B2) |
| ★`PDMUX_DUAL_WORKER_TRACE_EVERY` (기본 32) | 분모를 정의한다. ★**같은 저장소가 6일 전 `PREREG_S6`에서 감사 지적(`audit_s6_prereg_rev2/VERDICT.md:283`)을 받고 스코프로 등재한 항목**이다 — 재발 |
| ★`PDMUX_TRACE_FORCE_PREFILL` | 엔진 주석이 *"fraction-of-snapshots 통계는 이 플래그를 건너 비교 불가"* 라고 명시. `CAPBLOCK`이 정확히 그 형태 |
| "셀당 60 s"의 조작적 정의 | `bench_serving`은 `--num-prompts`로 돈다. 60 s를 어떻게 만드는지 미등록 |
| 셀당/부팅당 반복 `n` | 사실상 n=1. 명시 없음 |
| 셀 순서·랜덤화, 서버 플래그 전집합, pdmux cfg | *"rev3 승계"* 로 뭉갬 — rev3도 열거하지 않는다. ★그런데 **B1과 B3의 성립 여부가 `--chunked-prefill-size -1`·`--disable-radix-cache`에 직접 걸려 있다** |

### ★★ B7 — `cap`을 **CLI 인자**로 읽는다 (G9 fail-open 회귀)

§2는 *"`cap` = 그 부팅의 `--max-running-requests`"*. 그런데 엔진은 클램프한다:
```
model_runner_kv_cache_mixin.py:864-874
  max_num_reqs = min(max_running_requests // dp, estimated)
  if mambaish: max_num_reqs = min(max_num_reqs, max_mamba_cache_size // _calculate_mamba_ratio())
```
rev3의 **P3가 이미 수리한 결함**(*"`/get_server_info`는 요청한 server_args를 반사할 뿐 fail-open ⇒
엔진이 실제로 확정한 값(`scheduler.py:702-708` 배너)을 읽는다"*)을 P0가 **되돌렸다.**
★메모리 교훈과 정면 충돌: *"pin은 target 아닌 realized로 검증"*(Stage 0 D108≡D16 헤드라인이 이걸로 죽었다).

### ★ B8 — 부팅 상수 `145.8 s`: **H10 재발** (실측치가 같은 저장소에 있다)

rev3이 스스로 *"basis 오류"* 라 적었고(rev3 §3 표) rev3 감사가 **H10**으로 차단한 값(8B arm의 boot+60 s 창).
★같은 하네스 계열이 **Zamba2-2.7B 부팅을 이미 실측**했다
(`results/slo_sched/g16grid_8843*.out`, `G16_BOOT_WALLTIME … boot_s=`):
**n = 10, 평균 33.3 s, 중앙값 32.8 s, 범위 32.7–37.9 s.** ⇒ 인용값이 **4.4배 과대**.
방향은 보수적(과예산)이라 사고는 안 나지만, **차단이 제기된 다음 날 같은 상수를 다시 쓴 것**이 문제다(교훈 #41/#50).

### ★ B9 — `N_min = 200`이 미교정이고 **하필 결정적 셀에서 아슬아슬**하다

실측 스냅샷 밀도: cap에 앉은 20 s 구간에서 **78–96개/20 s → 60 s 셀 ≈ 234–288개**.
문턱 200 대비 여유 **17–44 %뿐**. 더 심한 과부하면 밀도가 더 떨어진다.
⇒ ★`UNDETERMINED (TELEMETRY DEAD)` 라벨이 **물리 신호가 가장 센 셀에** 붙을 수 있다 —
**게이트 #21의 정확한 형태**(*"측정 실패를 게이트 실패로 라벨링 마라"*의 거울상).

### ★ B10 — `prefill_admission_blocked`의 **편향 부호가 근거 없이 단정**됐고, 술어 스택이 **퇴화**한다

§4-7은 *"보수적(과소계수)이며 `CAP_NEVER_BINDS`를 더 쉽게 만든다"* 고 등재하고 감사에게는
*"얼마나 과소계수하는가"* 만 묻는다 — **과대계수 가능성을 프레임에서 지웠다.** 실측:
* `blocked`는 `queue_depth > 0`인 스냅샷의 **78 %** 에서 발화 ⇒ 사실상 *"대기 요청이 있다"* 에 가깝다.
* 샘플 지점이 두 곳(`mixin:997` = admission **이전**, `:1242` = 루프 말미)이고 부호가 **지점마다 다르다**.
* ★★그리고 실데이터에서 세 집합이 **동일**하다:
  `{blocked ∧ shared ∧ drbs≥48}` = `{drbs≥48}` = `{blocked ∧ shared}` — blk1/2/3 **완전 일치**,
  blk4만 1개 차이. `KVBLOCK = 0` (4/4 부팅).
  ⇒ ★**`blocked`와 reason 문자열이 아무 정보도 더하지 않는다.** 결정량 전체가
  *"배치가 cap에 앉아 있는가"* 라는 **점유 지시함수**로 붕괴한다 ⇒ 합격기준 2 위반, H2의 재발.

### ★ B11 — 워크로드를 **변화 trace → 정상(stationary) rate**로 바꿨는데 논증이 없다

rev3 §4는 *"변화 trace rate 3↔12 유지(절벽 회피)"* 를 등록했다. P0 §3은 *"ShareGPT 정상 rate, 셀당 60 s"*.
§3의 각주는 *"3·12는 변화 trace의 두 점"* 이라고 근거를 대지만 **정상 rate로 도는 것은 변화 trace가 아니다.**
결과: (a) 기존 g16 데이터와 직접 비교 불가(B4의 절약분이 날아감), (b) 정본 게이트 #2(정상 ShareGPT 폐기)와의
관계가 미논증. ★부수: rate 5개를 **한 부팅 안에서 순차** 실행하는데 순서·이월(mamba pool 단편화) 통제 0.

### ★ B12 — 게이트 #34 **2단(하네스층 감사)을 건너뛴다**

§6 다음 단계가 *"(1) 규칙층 감사 → (2) 제출"*. 정본은 2단계다(`CONSENSUS.md:250-255`
*"다음은 §10 하네스층 감사[게이트 #34 2단계]"*), rev3 §7의 로드맵도 *"(4) 하네스 → (5) 하네스 감사 → (6)…"* 였다.
★`nsl_lever/`에 **`.sbatch`가 0개** — 하네스가 존재하지 않는데 "제출 ≈0.25 GPU-hr"이라 적혀 있다.
배관 스모크(교훈 #25)도 미등록.

### B13 — 게이트 0의 `architecture=="legacy"`가 공허 + 괄호 주석이 오도 (②-사실2)

---

## ⑤ ★ 반증 실패 (깨려고 시도했으나 못 깬 것 — 인정)

1. ★**세 코드 사실의 줄번호가 전부 정확하다.** 5개 `scheduler.py` 인용, 3개 `multiplexing_mixin.py` 인용,
   2개 `dual_worker.py` 인용을 **실행 트리에서** 한 줄씩 대조 — 오차 0. 어제의 "틀린 트리" 형태 결함 **없음**.
2. ★**관측자 효과(§5-5)를 깨지 못했다.** `observe_scheduler`는 스케줄러 상태를 **변경하지 않는다**
   (읽기 + 자기 dict + `select_partition` no-op). docstring 주장은 **상태에 대해 참**이다.
   게다가 g16 로그가 **이미 telemetry-ON**이라 게이트 #41 비교 가능성도 유리하다.
3. ★**실현 cap이 요청 cap과 다를 가능성을 깨려 했으나 실패했다.** `_calculate_mamba_ratio`는
   `disable_radix_cache`면 **1을 반환**(`:391-392`) ⇒ `min(48, 48//1) = 48`. 실데이터도
   `max(decode_running_batch_size) = 48` **정확히** — 실현 cap = 요청 cap. (단 B7은 여전히 유효: 그 보장이
   **사전등록에 열거되지도 않은 플래그**에 걸려 있다.)
4. ★**"n=1이 치명적"이라는 공격에 실패했다.** g16 4부팅에서 이 결정량은 매우 안정적이다
   (개수 가중 0.00139–0.00148, 시간 가중 0.0794–0.0878, CV ≈ 4 %). **복제 수는 이 프로브의 병목이 아니다.**
5. ★**인용정지 위반을 찾지 못했다.** 기계 검사 0건 + 수동 확인 0건. rev1의 F4는 재발하지 않았다.
6. ★**§4의 "닫지 않는 것" 7개 항목은 정직하다.** 특히 §4-2(*"`CAP_BINDS_SOMEWHERE`가 나와도 레버가
   유용하다는 뜻이 아니다"*)와 §4-4(telemetry 축 대가)는 **충분한 수위**다. §5의 자기 지목 6개도
   정직하고, 그 중 1·3·5는 **감사가 실제로 깨야 할 곳을 정확히 짚었다**(그리고 1은 깨졌다).

⇒ **B1·B2·B3은 반증 실패가 아니라 실제 반증이다.** 다만 위 6개는 이 사전등록이 rev3보다
**형식적으로는 진보**했음을 보여준다(범위 축소·비용 축소·주장 축소가 전부 옳은 방향).

---

## ⑥ 값어치 판정

## **질문(H2)은 여전히 산다. 그러나 이 프로브는 사지 마라 — 그리고 GPU를 쓰기 전에 할 일이 있다.**

* **전진시킨 것**(되돌리지 말 것): 주장 범위를 *"cap이 admission을 막기는 하는가"* 로 좁힌 것 ·
  goodput·정책 판정을 완전히 뺀 것 · telemetry-ON 대가를 등재한 것 · `X`를 *"0과 구별"* 로만
  쓴다고 명시한 것 · 인용정지 준수.
* **왜 사면 안 되나**: 결정량이 **죽은 분기의 가드**이고(B1), 추정기가 미등록이라
  **이미 있는 데이터에서 판정이 뒤집히며**(B2), 대조 arm이 **손잡이 하나가 아니라 넷을 동시에 움직인다**(B3).
  이 셋 중 어느 하나만으로도 `CAP_NEVER_BINDS`는 **등재 불가능한 음성**이다.
* ★★**이번 감사가 새로 연 것**: 프로브가 사려는 셀의 **절반이 이미 저장소에 있다**(B4).
  그리고 `--max-running-requests`가 이 빌드에서 **순수 스케줄링 손잡이가 아니라 메모리 풀 크기 그 자체**라는 것(B3) —
  이것은 P0뿐 아니라 **NSL-1 트랙 전체의 전제**(*"cap = admission lever"*)에 대한 지적이다.

### 다음 회차 권고 (우선순위 · GPU 0에서 시작하라)

1. ★★★**GPU 0 재분석을 먼저 하라.** `results/slo_sched/g16_*_d44_boot1_*_telemetry.jsonl`(n=4 부팅)에
   **시간 가중 추정기를 사전등록하고** 결정량을 계산하라. `cap 48 × {rate 3, 12}` 셀이 여기서 나온다.
   ★단 **B1을 고치기 전에는 그 수치도 판정이 아니다** — 추정기 등록·재현 스크립트 확보용으로만 써라.
2. ★★★**B1 수리 = 엔진 패치.** `scheduler.py:2369` / `:2434` / `:2476` 세 사이트에 각각 카운터
   (+ 발화 시점의 `running_bs`·`len(can_run_list)`)를 심어라. `PDMUX_HOLB_PATH` 선례처럼
   **기본 OFF·arm 무관**으로. 그것이 감사 §8-2가 원래 지시한 *"분리 계수"* 다.
   상태 대리변수는 **그 카운터에 대한 교차검증용**으로만 병기하라.
3. ★★**B3 수리 = 설계 형태 변경.** 두 선택지뿐이다:
   (a) 두 cap arm 모두 `--max-mamba-cache-size`를 **동일 값으로 고정**하고 실현
   `max_total_num_tokens`·`max_mamba_cache_size`·`max_running_requests`를 **배너에서 읽어 셀마다 기록**
   (= rev3 P1 부활 + B7 수리), 또는 (b) **cap 축 대조를 포기**하고 단일 cap에서 rate 축만 돌린다.
4. ★**B2 수리**: `CAPBLOCK`을 **시간 가중으로 정의**하고, 개수 가중 값을 **병기**하되 판정에는 쓰지 마라.
   `trace_every`·`TRACE_FORCE_PREFILL`을 스코프에 못박아라(`PREREG_S6` §D6 형식 승계).
   `X = 0.05`는 **두 추정기 각각에 대해** 교정하라(현재 값은 시간 가중에서만 의미가 있다).
5. ★**B5 수리**: 양성대조를 `cap ∈ {4 또는 8}` 셀 + **사이트 카운터 vs 대리변수 교차검증**으로 교체.
6. **B6·B8·B9·B11·B12**: 자유 모수 재열거(전수 주장 시 근거 명시) · 부팅 상수 **32.8 s**로 교체 ·
   `N_min`을 실측 밀도(3.9–4.8 스냅샷/s @ cap-bound)로 교정 · 워크로드 변경 논증 또는 변화 trace 복귀 ·
   하네스 작성 → **게이트 #34 2단 감사** → 스모크 → 제출.

---

## ⑦ ★ 쓰면 안 되는 문장

**승계(불변)**: *"cap이 문다"*(측정 전) · *"NSL-1이 admission 축을 쟀다"* · *"cap 축이 무력함이 확인됐다"* ·
*"§5-8(b)를 닫았다/좁혔다"* · *"rev3이 규칙층을 통과했다"* · *"rev3이 G1–G15를 전부 수리했다"* ·
*"cap은 TTFT 다리에만 작용한다"* · *"HI에서 ITL 다리는 안 문다"* · *"HI 동시성 44.19/48이므로 cap은 문다"* ·
*"rate 6–8은 정본이 절벽으로 판정한 대역"* · *"정본 goodput은 TTFT-only 지표였으므로 HE0가 흔들린다"* ·
HE0 · gate #13/#16 "닫았다" · switch-cost "닫았다" · C2 인용정지 (a)(b) · 2026-08-01 E1 4건.

★**신설 6건**:
1. *"`running_bs ≥ cap`이 cap 경로의 발화 조건이다"* — **B1로 반증**(`:2369`는 이 부팅서 실행 불가,
   살아 있는 `:2434`의 조건은 `running_bs + |can_run_list| ≥ cap`).
2. *"`batch_is_full`만 보면 배치 크기로 cap과 KV를 가를 수 있다"* — **B1·B3로 반증**.
3. *"이 프로브는 엔진 패치가 필요 없다"*(P0 §1.1 제목) — 사이트별 분리 계수는 패치를 요구한다.
4. *"`CAPBLOCK` = 0.0015이므로 cap은 이 워크로드에서 안 문다"* / *"= 0.087이므로 문다"* —
   ★**이 판정서의 두 수치 중 어느 쪽도 물리 판정이 아니다**(술어가 B1로 무효). **추정기 민감도의 증거로만** 인용하라.
5. *"cap 24 vs 48 대조가 cap 효과를 분리한다"* — **B3**(cap·req pool·mamba pool·attention KV 예산이 동시에 변함).
6. *"프로브가 `CAP_NEVER_BINDS`를 등재했다"* — 현 설계로는 **등재 불가능한 라벨**이다(합격기준 1·2·4 동시 실패).

---

### 부록 — 신규 방법론 교훈 후보 3건 (doc-steward 판단용)

* **(A)** ★★***"시간비"라고 쓰기 전에 분모가 시간인지 확인하라.*** 스냅샷 집합에 `시간비{...}`를 쓰면
  **간격이 비균일할 때 관심 사건과 반상관**이 될 수 있다(여기선 118배 밀도 차 → 58배 추정치 차 → **판정 반전**).
  ★엔진 소스가 이미 경고문을 달아 놓은 사안이라 **교훈 #61/#63/#72의 4번째 재발**이다.
* **(B)** ★***손잡이가 순수한지 코드로 확인하라.*** `--max-running-requests`는 이 빌드에서 admission cap이자
  `req_to_token_pool` 크기이자 `max_mamba_cache_size`이자 (간접적으로) attention KV 예산이다.
  *"A 하나만 바꾼다"* 는 CLI 인자 이름이 아니라 **초기화 경로 전체**로 증명하라.
* **(C)** ★***프로브를 사기 전에 저장소를 grep하라.*** 사려던 격자의 절반(989개 telemetry 파일)이
  동일 arm·동일 플래그로 이미 존재했다. **"이 셀이 이미 측정됐는가"를 사전등록의 필수 절로 만들어라**
  (선행 등록 승계 항목의 데이터층 변종).
