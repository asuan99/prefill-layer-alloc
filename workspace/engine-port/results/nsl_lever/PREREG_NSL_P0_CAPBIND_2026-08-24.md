# 사전등록 — **NSL P0**: admission cap이 **이 운영점에서 실제로 막는가**

2026-08-24 · 메인 세션 · **규칙층 초안**(게이트 #34 1단, ★**미감사**) · GPU 지출 **0**(미제출) ·
새 성능 판정 **0건**.

> **출처**: [`audit_nsl1_rules_rev3_2026-08-23/VERDICT.md`](audit_nsl1_rules_rev3_2026-08-23/VERDICT.md)
> §8 권고 2. 감사 원문: *"**H2를 프로브로 승격하라 — 이것이 이제 이 설계의 P0다.** …
> **결정 규칙**: *"cap이 문다"* = `admission이 batch_is_full로 막힌 스텝 시간비 ≥ X %`
> (X를 사전등록). 이것이 `NO_WINDOW`의 **유일한 정직한 판정**이다.
> ★**Little 법칙 동시성은 이 목적에 쓰지 마라**(H2에서 반증됨)."*
>
> ★**이 문서는 NSL-1 설계(rev1–rev3)가 아니다.** rev3은 `NO-GO`(死因 H1·H2)이고, 이 프로브는
> 그 차단들을 **고치지 않는다** — **rev4를 쓸 자격이 있는지**를 먼저 묻는다.

---

## 0. 무엇이 문제였나 (H2)

NSL-1 rev1–rev3의 **유일한 생존 전제**는 *"HI에서 동시성 44.19/48 = 92 % ⇒ cap이 문다"* 였다.
감사가 그것을 무너뜨렸다:

* 44.19는 Little 법칙으로 복원한 ★**in-system 인구**(대기 + 실행)이고,
  `--max-running-requests`는 ★**running batch만** 제한한다. **다른 양이다.**
* 같은 추정량이 **rate 10에서 63.96, rate 12에서 83.06** — ★**cap 48을 크게 초과**한다.
  ⇒ 그 양이 cap의 상한이 아님을 저장소 데이터가 직접 보인다.

⇒ ★**"cap이 문다"는 이 저장소에서 한 번도 측정된 적이 없다.**

---

## 1. ★ 코드 사실 — 무엇이 "cap이 막았다"인가 (2026-08-24 직접 확인)

`managers/scheduler.py`:
```
:2296-2299  get_num_allocatable_reqs(running_bs) = pp_max_micro_batch_size − running_bs
:667-671    pp_max_micro_batch_size := max(max_running_requests // pp_size, 1)
:2364-2371  get_num_allocatable_reqs(...) <= 0 ∧ chunked_req ⇒ batch_is_full = True   ← cap 경로
:2432-2434  len(can_run_list) >= get_num_allocatable_reqs(running_bs) ⇒ True           ← cap 경로
:2468-2476  res == AddReqResult.NO_TOKEN            ⇒ True                             ← ★KV 경로
```
★**두 경로가 같은 불리언 하나를 세운다.** 그래서 `batch_is_full`만으로는 **cap과 KV를 못 가른다**
(감사 지시: *"후자는 KV 경로이므로 **분리 계수**"*).

★★**분리를 가능하게 하는 항등식**: cap 경로의 조건은 `pp_max_micro_batch_size − running_bs ≤ 0`
⟺ ★**`running_bs ≥ cap`**. ⇒ **관측 가능한 배치 크기가 cap 경로의 발화 조건 그 자체**다.
KV 경로는 배치가 cap 미만이어도 발화할 수 있다. ⇒ **둘은 배치 크기로 갈린다.**

### 1.1 ★ 엔진 패치가 필요 없다

`multiplex/multiplexing_mixin.py:86,93` — `PDMUX_TELEMETRY_PATH` → `dual_worker_trace_path`.
`:468-476` `_dual_worker_sync`는 `dual_worker_enabled` **또는** `dual_worker_trace_path`가 있으면
`observe_scheduler`를 돈다 ⇒ ★**telemetry만 켜면 관측자가 `architecture="legacy"`(단일 워커)
에서 동작한다.** `PDMUX_TRUE_DUAL_WORKER`는 **켜지 않는다**(그것은 아키텍처를 바꿔 교락이 된다).

`multiplex/dual_worker.py:591-601,615-618`이 매 스냅샷에 방출:
`decode_running_batch_size`(=`active_batch_size`) · `prefill_admission_blocked` ·
`prefill_admission_block_reason` ∈ {`shared_batch_capacity`(=`batch_is_full` 참), `scheduler_admission_pending`}.

---

## 2. 결정량

스냅샷 `s`에 대해 (`cap` = 그 부팅의 `--max-running-requests`):

```
blocked(s)   := s.prefill_admission_blocked ∧ s.prefill_admission_block_reason == "shared_batch_capacity"
cap_bound(s) := s.decode_running_batch_size >= cap          # §1의 항등식
CAPBLOCK     := 시간비{ s : blocked(s) ∧ cap_bound(s) }      # ★cap 귀속 차단
KVBLOCK      := 시간비{ s : blocked(s) ∧ ¬cap_bound(s) }     # ★KV 귀속 차단
CAPSAT       := 시간비{ s : cap_bound(s) }                   # 배치가 cap에 앉은 시간비(서술)
```
★**1차 산출물은 `CAPBLOCK`의 `rate × cap` 곡선**이다. 나머지는 병기.

### 2.1 등록 판정

```
0. 계측 게이트: architecture=="legacy" ∧ 스냅샷 수 ≥ N_min ∧ reason 필드가 비어있지 않은 스냅샷 존재
                                                        아니면 UNDETERMINED (TELEMETRY DEAD)
1. 어떤 (rate, cap) 셀에서 CAPBLOCK ≥ X   →  CAP_BINDS_SOMEWHERE   → rev4를 쓸 자격 있음
2. 전 셀에서 CAPBLOCK < X                 →  ★CAP_NEVER_BINDS      → ★이 워크로드로는 물을 수 없다
3. 양성대조 실패(§2.2)                     →  UNDETERMINED (CONTROL FAILED)
```
★**등록값 `X = 0.05`**(스케줄러 관측 시간의 5 %) — ★**[임의]**. 이 값의 역할은 *"0과 구별"* 이지
*"충분히 크다"* 가 아니다. ⇒ **1차 보고는 항상 곡선 전체**이고 `X`는 2번 분기에만 쓰인다.

### 2.2 ★ 양성대조 (발화 가능해야 한다)

> 같은 rate에서 **`cap = 24`의 `CAPBLOCK`이 `cap = 48`의 것보다 크다.**

작은 cap이 더 자주 막지 않으면 **측정이 깨진 것**이다(라벨 `CONTROL FAILED`). ★이 대조는
**데이터가 정한다** — 항등식이 아니다(cap이 전혀 안 물면 둘 다 0이 되어 대조가 발화하지 않고,
그때는 2번이 아니라 3번으로 간다. ★**"둘 다 0"과 "cap 무력"을 구별하기 위해 `CAPSAT`을 병기**한다).

---

## 3. 격자·비용

| 항목 | 값 |
|---|---|
| arm | **Zamba2-2.7B**(rev3 승계 — 변화 trace 하네스가 실제로 도는 모델) |
| `D` | **d44** 한 점(정본 권고 운영점) |
| cap | **24 · 48** ⇒ 부팅 **2회**(cap은 서버 인자라 재부팅 필요) |
| rate | **3 · 6 · 7 · 8 · 12** — 한 부팅 안에서 순차 실행 |
| 워크로드 | ShareGPT 정상 rate, 셀당 60 s |
| telemetry | ★`PDMUX_TELEMETRY_PATH` **ON**, `PDMUX_TRUE_DUAL_WORKER` **OFF** |
| 비용 | 2 부팅 × 145.8 s + 10 셀 × 60 s ≈ **892 s ≈ 0.25 GPU-hr** |

★**rate 목록의 근거**: 3·12는 변화 trace의 두 점, 6·7·8은 감사가 chat SLO 부팅 간 SD를 실측한
점(각 0.82 / 1.44 / **6.09** pp)이며 **8만 절벽**이다(rev3 §0의 *"6–8 전체가 절벽"* 은 감사 H3가
반증했다). ⇒ 이 프로브는 **절벽 위 셀도 포함**한다 — `CAPBLOCK`은 goodput이 아니므로 SLO 임계
지시함수의 절벽에 걸리지 않는다(★이것이 이 프로브를 **먼저** 도는 이유다).

### 3.1 자유 모수 전수 (7개)

| # | 모수 | 값 | 성격 |
|---|---|---|---|
| 1 | `X`(판정 문턱) | `0.05` | ★**[임의]** — §2.1 |
| 2 | rate 격자 | `3,6,7,8,12` | 근거는 위, ★**[임의]** |
| 3 | cap 격자 | `24,48` | 양성대조를 만들기 위한 최소 2점 · ★**[임의]** |
| 4 | 셀당 지속 | `60 s` | ★**[임의]** |
| 5 | `N_min`(계측 게이트) | `200` 스냅샷/셀 | ★**[임의]** |
| 6 | `D` | `d44` | 정본 권고 운영점 |
| 7 | arm·모델·ctx | rev3 승계 | — |

★**신규 통계 게이트 0 · 검정 0 · CI 0.** 전부 시간비와 순서 비교다.

---

## 4. 닫지 **않는** 것

1. ★**goodput을 재지 않는다.** 성능·정책 주장 **0건**. HE0·정책 순위 전진 0.
2. ★**"cap을 바꾸면 goodput이 바뀐다"를 묻지 않는다** — *"cap이 admission을 막기는 하는가"* 만.
   `CAP_BINDS_SOMEWHERE`가 나와도 **레버가 유용하다는 뜻이 아니다.**
3. ★**NSL-1 rev3의 차단 H1·H3–H11을 닫지 않는다.** 특히 **H1**(다리 분리 오독)은 이미 철회
   대상이고, **H7**(p95 추정기 미등록)·**H11**(δ 기준 불일치)은 rev4가 닫아야 한다.
4. ★**telemetry 축의 대가**(게이트 #41): 이 프로브는 telemetry-ON이므로 **telemetry-OFF 시대
   로그와 절대값 직접 비교 금지**. ⇒ 이 프로브의 산출물은 **자기 안에서만** 비교된다
   (같은 telemetry 설정의 셀끼리).
5. `CAPBLOCK`은 **스케줄러 관측 시간비**이지 벽시계 비율이 아니다(스냅샷 간격이 균일하다는
   보장이 없으므로 ★**스냅샷 간격 분포를 병기**한다).
6. **단일 arm · 단일 `D` · 단일 기판 · ctx 4096.**
7. ★**`prefill_admission_blocked`의 정의 한정**(`dual_worker.py:591`): *"prefill 큐에 대기가
   있는데 prefill 활성 배치가 0"* 일 때만 참이다. ⇒ **prefill이 조금이라도 돌고 있으면 막힘으로
   세지 않는다** — 이것은 **보수적**(과소계수) 방향이며, `CAP_NEVER_BINDS`를 **더 쉽게** 만든다.
   ★그 비대칭을 판정서에 병기한다.

---

## 5. 감사 의뢰 (게이트 #34 1단 — ★**GPU 전 필수**)

> **단일 질문**: *"`CAPBLOCK`이 정말 cap 경로를 재는가, 그리고 `CAP_NEVER_BINDS`가
> 이 워크로드에 대한 **등재 가능한 음성**인가?"*

**먼저 깨뜨릴 곳**:
1. ★★**§1의 항등식이 옳은가** — `running_bs ≥ cap`이 정말 cap 경로의 발화 조건과 동치인가?
   `pp_size > 1`·`chunked_req`·`enable_priority_preemption` 분기에서 깨지는가?
   그리고 `decode_running_batch_size`(=`active_batch_size`)가 정말 `len(running_batch.reqs)`인가?
   ★**이 항등식이 깨지면 이 프로브는 H2와 같은 실수를 반복하는 것이다.**
2. ★**`prefill_admission_blocked`의 전제**(§4-7)가 `CAPBLOCK`을 얼마나 과소계수하는가 —
   그 편향이 `CAP_NEVER_BINDS` 쪽이라면 **2번 판정이 편향된 라벨**이 아닌가?
3. ★**양성대조(§2.2)가 발화 가능한가** — cap 24와 48이 실제로 다른 `CAPBLOCK`을 낼 수 있는
   rate가 이 격자에 있는가? 없으면 대조가 공허하다.
4. **`X = 0.05`가 "거의 항상" 또는 "거의 안" 발화하는 값인가.**
5. **telemetry-ON이 admission 거동 자체를 바꾸는가**(관측자 효과) — `_dual_worker_sync`가
   스케줄러 상태를 바꾸지 않는다고 docstring이 주장한다(*"without changing legacy scheduler
   state"*). **코드로 확인하라.**
6. §4-2의 스코프(*"레버가 유용하다는 뜻이 아니다"*)가 충분한가.

---

## 6. 상태

**미제출 · 미감사 · GPU 지출 0.**
다음: (1) **규칙층 감사** → (2) 제출(≈0.25 GPU-hr) → (3) `CAP_BINDS_SOMEWHERE`면 **rev4**
(H1 철회 + H3–H11), 아니면 ★**`CAP_NEVER_BINDS` 등재 — 이 워크로드로 admission 축을 물을 수
없다는 정보 있는 음성**.

★**쓰면 안 되는 문장**: *"cap이 문다"*(측정 전) · *"NSL-1이 admission 축을 쟀다"* ·
*"§5-8(b)를 닫았다/좁혔다"* · *"cap 축이 무력함이 확인됐다"*(`CAP_NEVER_BINDS`는 **이 워크로드
한정**이며 lever 갈래에 대한 진술이 아니다) · HE0·gate #13/#16·switch-cost 관련 전부.
