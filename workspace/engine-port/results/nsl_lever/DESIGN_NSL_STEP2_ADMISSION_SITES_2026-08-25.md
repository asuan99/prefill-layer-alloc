# 설계 — NSL **②**(사이트별 admission 카운터): B1을 엔진 패치로 수리한다

2026-08-25 · **GPU 지출 0**(코드 대조만) · **새 성능 판정 0건** · HE0 불변 · ★**패치 미작성**.
선행: `audit_nsl_p0_capbind_2026-08-24/VERDICT.md` 권고 **2**(死因 B1 수리) ·
`MEMO_NSL_STEP2_DECISION_2026-08-24.md` · **`PREREG_NSL_STEP3_KNOB_PURITY_2026-08-25.md`(③, 선행)**.

> ★**②만 사는 것은 값이 없다**(MEMO §4) — B3가 남으면 cap arm 비교가 서지 않는다.
> ③이 먼저 등록됐으므로(2026-08-25) 이제 ②가 값을 가진다. **둘은 같은 캠페인에서 함께 산다.**

---

## 0. ② 가 사는 것 / 못 사는 것 (MEMO §3 승계)

| 질문 | ①+서술이 답하는가 | **②가 답하는가** |
|---|---|---|
| 배치가 cap에 앉는 시간 비중 | 예 (`T ∈ [0.080, 0.119]`) | 불필요 |
| 그 동안 prefill이 안 나가는가 | 예 (시간비 1.000) | 불필요 |
| ★**막은 것이 cap 경로인가 KV 경로인가** | ★**아니오** | ★**예 — 이것이 ②의 전부다** |
| cap 경로가 **몇 번** 발화했는가 | 아니오(상태만) | 예 |

## 1. ★ 감사가 지목한 세 사이트 — **직접 확인했고, 두 가지가 다르다**

감사 권고 2는 `managers/scheduler.py:2369` / `:2434` / `:2476`을 지목했다. **줄번호는 현재 트리와 일치한다**
(교훈 #80에 따라 직접 확인; 같은 세션에서 다른 감사의 줄번호 2건은 어긋나 있었다).
그러나 **두 가지가 감사 서술과 다르다**.

### 1.1 ★ `batch_is_full`을 세우는 자리는 **셋이 아니라 다섯**이다

| 사이트 | 술어 | **pdmux arm**에서 | 닫는 것 |
|---|---|---|---|
| `:2369` | `chunked_req is not None ∧ ¬enable_priority_preemption` | ★**미도달** | **엔진 assert** — `--enable-pdmux`는 `chunked_prefill_size == -1`을 강제하고(`server_args.py:6129-6131`), `<= 0`이면 `chunked_prefill_size = None`이 되어(`managers/scheduler.py:890-891`) `chunked_req`가 **항상 None**이다 |
| **`:2434`** | `len(adder.can_run_list) ≥ get_num_allocatable_reqs(running_bs)` | ★**도달 — cap 경로** | — |
| `:2439` | `len(can_run_list) ≥ req_to_token_pool.available_size()` | **미도달** | **엔진 assert** — pdmux는 `disaggregation_mode == "null"`을 강제(`server_args.py:6132-6134`) |
| `:2472` | `NO_TOKEN ∧ enable_hierarchical_cache` (True **또는** False로 배정) | **미도달** | 플래그 미사용(**assert 아님** — 이 셋 중 유일하게 구성으로만 닫힌다) |
| **`:2476`** | `NO_TOKEN ∧ ¬enable_hierarchical_cache` | ★**도달 — KV/토큰 경로** | — |

⇒ ★★**pdmux arm에서 도달 가능한 사이트는 다섯 중 둘뿐이고, 그 둘이 정확히 감사가 가르려던
cap 경로와 KV 경로다.** 감사가 센 셋 중 `:2369`는 pdmux에서 **엔진이 assert로 닫는다**.
★**정정(2026-08-25, 이 문서 커밋 직후 자기 검출)**: 초판은 `:2369`를 *"도달 — chunked-prefill 가드"* 로
적었다. **거짓이다** — 근거는 위 assert이며, 이 문서가 스스로 그 assert를 인용하면서 반대로 적었다.

★**설계 결정은 그대로**: 다섯 사이트를 **전부** 계측하고 닫힌 셋이 **0을 유지하는지 검사**한다.
이유가 하나 늘었다 — ★**사이트 도달성은 arm마다 다르다.** fused 비교 arm `chunk512`(pdmux 없음,
`--chunked-prefill-size 512`, `g2_run.sbatch:104,115`)에서는 `:2369`가 **도달 가능**하다.
arm 무관 카운터가 arm마다 다른 도달성을 갖는다는 사실 자체가 **기록 대상**이다.

### 1.2 ★★ cap 술어의 상수는 `--max-running-requests`가 **아니다**

`get_num_allocatable_reqs`(`managers/scheduler.py:2296-2300`):

```
res = get_global_server_args().pp_max_micro_batch_size - running_bs
if self.pp_size > 1: res = min(res, self.req_to_token_pool.available_size())
```

⇒ `:2434`의 술어는 실제로 **`running_bs + |can_run_list| ≥ pp_max_micro_batch_size`** 다.

`pp_max_micro_batch_size`는 **미지정이면** `max(max_running_requests // pp_size, 1)`로 채워진다
(`managers/scheduler.py:668-671`). ⇒ ★**`pp_size == 1` ∧ 플래그 미지정일 때만** 그 상수가 cap과 같다.

- ★**NSL ①의 `cap = 48` 사용은 이 구성에서 정당하다** — `pp_size == 1`은 **엔진이 assert로 강제**하고
  (`server_args.py:6127-6128`, `--enable-pdmux` 하에서), `pp_max_micro_batch_size`는 이 프로젝트가
  주지 않는다. **확인했으므로 이제 가정이 아니다.** ★단 **둘의 성격이 다르다** — 앞은 강제,
  뒤는 **관례**다. 강제되지 않는 쪽만 부팅 전제로 검사하면 된다.
- ★**그러나 항등식이 아니라 조건부다.** ②의 사전등록은 **부팅 전제**로 못박는다:
  `pp_size == 1` ∧ `pp_max_micro_batch_size` 미지정, 그리고 **실현값을 배너/`get_server_info`에서
  읽어 셀마다 기록**(③의 R3와 같은 형태). 어긋나면 `CAP_CONSTANT_MISMATCH` = MEASUREMENT 조건.
- ★이것은 ①을 **반증하지 않는다** — 근거를 **가정에서 확인으로** 옮긴다(교훈 #79).

## 2. 카운터 설계

**선례 = `PDMUX_HOLB_PATH`**(`src/multiplex/holb_probe.py` + `src/patches/holb_probe_scheduler_hooks.patch`):
**기본 OFF · arm 무관 · 훅 자리는 `is not None` 검사로 붕괴 · 별도 모듈에 로직**.

| 항목 | 값 |
|---|---|
| 스위치 | `PDMUX_ADMISSION_SITES_PATH`(미설정 ⇒ 팩토리가 `None` 반환, **패치 전과 동일 동작**) |
| 위치 | `src/multiplex/admission_sites.py`(신규) + `src/patches/admission_sites_scheduler_hooks.patch` |
| arm | ★**arm 무관** — `--enable-pdmux` 여부와 무관하게 같은 코드가 계측한다(Gate-1 결함 재발 방지: pdmux arm에만 telemetry가 있던 것) |
| 사이트당 기록 | `site_id` · `running_bs` · `len(adder.can_run_list)` · `get_num_allocatable_reqs(running_bs)` · `waiting_queue` 길이 · 단조 `seq` |
| 집계 | 사이트별 **발화 횟수**(사건 수) + 발화 시점 분포. ★**상태 대리변수는 교차검증용으로만 병기**(감사 §8-2) |
| 쓰기 | 스케줄러 스레드에서 파일 I/O 금지 — telemetry와 같은 큐 경유 |

### 2.1 ★ `|can_run_list|`가 여기서 처음 관측된다

NSL ①이 참 술어를 계산 못 한 이유가 **`can_run_list`가 telemetry 44필드에 없다**는 것이었다
(하계 `running_bs ≥ cap` ⊆ T ⊆ 상계 `running_bs + queue_depth ≥ cap`, `T ∈ [0.080, 0.119]`).
②는 그 항을 **발화 시점에** 기록하므로 **구간이 점으로 좁혀진다**.
★단 그것은 **②가 도는 새 런에서만** 그렇다 — **기존 아티팩트는 소급되지 않는다.**

## 3. 등록 규칙

| | 규칙 |
|---|---|
| **S1** | 다섯 사이트 **전부** 계측한다(도달 불가 둘 포함) |
| **S2** | ★`:2439`·`:2472`의 발화 수가 **0이 아니면** `CONFIG_DRIFT` = **MEASUREMENT 조건**, 그 런의 귀속을 채점하지 않는다 |
| **S3** | 부팅 전제 `pp_size == 1` ∧ `pp_max_micro_batch_size` 미지정을 검사하고 **실현값을 기록**. 어긋나면 `CAP_CONSTANT_MISMATCH` |
| **S4** | ③의 **R1–R6을 동시에 적용**한다(mamba pool 고정 + 배너 3값 기록 + 실현 cap 검사). ②만 도는 캠페인은 **없다** |
| **S5** | `reason` 문자열은 **기록하되 귀속에 쓰지 않는다** — 감사 B10: 그 문자열이 두 경로를 뭉갠다 |
| **S6** | 기본 OFF 동치성: 플래그 미설정 런이 패치 전과 **byte-identical telemetry**를 쓰는지 단위 테스트로 고정(sticky·holb 선례) |

## 4. 이 설계가 **아직 안 산 것**

- ✗ **규칙층 감사 미실행** — 게이트 #34 1단. 이 문서가 그 대상이다.
- ✗ **패치 0줄** · 하네스 0줄 · 캠페인 미설계(arm·rate·n·워크로드 미등록).
- ✗ **B3는 ③이 좁힐 뿐 닫지 않는다**(cap은 여전히 `req_to_token_pool` 크기와 클램프를 움직인다).
- ✗ ★**NSL-1의 가설은 여전히 미검증** — ②+③은 *"무엇이 막았는가"* 를 사는 것이고,
  *"cap을 조절해 SLO goodput을 얻는다"* 는 **별도의 서빙 실증**을 요구한다.

## 5. ★ 쓰면 안 되는 문장

- ✗ ★★★ *"②를 설계했으니 cap이 admission lever임이 곧 선다"* — **귀속을 사는 것**이지 레버 주장이 아니다.
- ✗ ★★★ *"cap이 문다 / 안 문다"* — B1로 점 술어 무효, ②가 **돌기 전까지** 그대로다.
- ✗ ★★ *"감사가 세 사이트를 잘못 셌다"* — 감사의 셋은 **`batch_is_full`을 세우는 자리 중 가장
  중요한 셋**이며 줄번호도 정확하다. 이 문서가 더한 것은 (a) 자리가 **다섯**이라는 것, (b) pdmux
  arm에서 **도달 가능한 것은 둘**이라는 것, (c) 닫힌 셋을 계측해 구성 변화를 잡는다는 규율이다.
- ✗ ★★ *"`:2369`는 pdmux arm에서 발화한다"* — **엔진 assert가 닫는다**(§1.1). 이 문서 **초판이
  그렇게 적었고 정정했다.**
- ✗ ★★ *"pdmux와 chunked prefill을 함께 쓴다"* — `server_args.py:6129-6131`이 **assert로 금지**한다.
  이 저장소에서 `--chunked-prefill-size 512`는 **fused 비교 arm(`chunk512` = A3)** 이지 운영점이 아니다.
- ✗ ★★ *"cap 술어의 상수는 `max_running_requests`다"* — **`pp_max_micro_batch_size`** 이고,
  둘이 같은 것은 `pp_size == 1` ∧ 플래그 미지정일 때뿐이다(§1.2).
- ✗ ★★ *"①의 구간이 ②로 소급해 좁혀진다"* — **새 런에서만**(§2.1). 기존 아티팩트에 `can_run_list`는 없다.
- ✗ ★ *"`KVBLOCK = 0`이므로 KV는 안 문다"* — reason 문자열이 두 경로를 뭉갠다(B10).
- **승계**: *"NSL-1이 admission 축을 쟀다"* · *"cap 축이 무력함이 확인됐다"* · *"rev3이 규칙층을
  통과했다"* · HE0 · gate #13/#16 *"닫았다"* 금지 · switch-cost *"닫았다"* 금지 · C2 인용정지 (a)(b).
