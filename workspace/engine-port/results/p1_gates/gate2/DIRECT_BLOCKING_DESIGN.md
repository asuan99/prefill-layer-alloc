# Gate 2 — head-of-line blocking **직접 계측** 설계 (HOLB probe)

작성: 2026-08-06 · engine-porter · **상태: 설계 + 구현 + CPU correctness gate 통과, GPU gate 미실행**

이 문서는 `PREREG_GATE2_2026-08-06.md` **§9-8**이 사전에 못 박은 분기를 집행한다:

> "rev3도 감사에서 죽으면 다음 수는 또 다른 통계량이 아니라 **다른 실험**이어야 한다 — 예:
> head-of-line blocking 지속시간을 엔진 계측으로 직접 재는 것."

**이 문서는 계측의 설계·정확성 증명까지만 다룬다. 성능 주장은 하나도 포함하지 않는다.**
"이 계측이 Gate 2의 primary가 될 수 있는가"에 대한 정직한 판정은 §10에 있다.

---

## 0. 무엇을 재려는가

정본 `CONSENSUS.md` §1-1이 서술한 fused 측 死因:

> "prefill 배치가 decode를 막음, 클러스터 지속 166ms@r2 → 474ms@r4."

지금까지 세 번의 rev는 이것을 **요청 단위 SLO 지시함수**로 간접 측정했고, 세 번 다
**절단**(A4 바닥 / A1 천장)으로 죽었다. 여기서는 같은 현상을 **엔진 타임라인 위의
밀리초 연속량**으로 직접 잰다.

추정 대상(estimand), 한 문장:

> **관측 창 동안, "진행시킬 decode 작업이 존재하는데 decode forward가 실행되고 있지 않은"
> GPU 타임라인 시간(ms).**

---

## 1. 정의 (수식)

### 1.1 원자 단위 = forward span

엔진의 모든 GPU forward는 `Scheduler.run_batch()`를 통과한다(**네 arm 전부**).
각 forward `f`에 대해 다음을 기록한다.

| 기호 | 뜻 | 출처 |
|---|---|---|
| `kind(f)` | `decode` / `other` | `f.forward_mode` (§1.3) |
| `stream(f)` | 실행 스트림 식별자 | `torch.cuda.current_stream().cuda_stream` |
| `pend(f)` | forward 발행 시점의 **decode-대기 요청 수** | §1.2 |
| `[s_f, e_f]` | forward의 **GPU 타임라인** 구간 | `torch.cuda.Event(enable_timing=True)` 쌍 |
| `dur(f) = e_f − s_f` | GPU 실행 시간(ms) | `start.elapsed_time(end)` |

호스트 시계가 아니라 **GPU 이벤트**를 쓰는 것이 이 설계의 핵심이다. 근거는 §4.1.

### 1.2 `pend` — decode-대기 요청 수

```
pend = Σ_{r ∈ self.running_batch.reqs} 1[¬ r.finished()]      (running_batch가 prefill-only가 아닐 때)
     = 0                                                        (그 외)
```

`self.running_batch`는 **네 arm 전부에서 같은 필드**다 — prefill을 끝내고 다음 토큰을
기다리는 요청 집합. 아직 prefill되지 않은 요청(`waiting_queue`, `split_prefill_batch`,
`chunked_req`)은 **decode-대기가 아니므로 포함하지 않는다**(이걸 섞으면 admission 큐잉과
head-of-line blocking이 한 숫자로 뭉개진다).

### 1.3 `kind` — decode를 진행시키는가

```
advances_decode(mode) = mode.is_decode() ∨ (mode == MIXED)
```

| ForwardMode | kind | 비고 |
|---|---|---|
| `DECODE` | decode | |
| `MIXED` | decode | chunked prefill + decode 혼합 배치. **§3에서 별도 처리** |
| `EXTEND` | other | A1/A2/A3의 prefill (A3는 chunk마다 1회) |
| `SPLIT_PREFILL` | other | A4의 prefill |
| `IDLE` | other | dp_size=1이라 발생하지 않아야 함 |
| `TARGET_VERIFY`/`DRAFT_EXTEND`/`PREBUILT`/`DLLM_EXTEND` | other | 이 캠페인에서 발생하지 않아야 함 |

**모든 mode의 관측 횟수를 그대로 기록**한다 — 발생하지 않아야 할 mode가 발생하면
분석 단계에서 즉시 드러나야 한다(가정을 주장하지 말고 계측한다).

### 1.4 decode 타임라인과 gap

decode forward들을 발행 순서로 `D_0, D_1, …, D_{N−1}` 이라 하자.

```
gap_k  = s_{D_{k+1}} − e_{D_k}          (ms, GPU 타임라인)
```

`gap_k`는 **decode가 한 스텝도 진행하지 못한 연속 구간**이다. 이것이 정본이 말한
"클러스터 지속"의 직접 대응물이다.

### 1.5 primary 량

```
stall_ms       = Σ_k  gap_k · 1[gap_k가 strict]
stall_req_ms   = Σ_k  gap_k · bs(D_{k+1}) · 1[gap_k가 strict]      (요청·ms)
```

- `bs(D_{k+1})` = 그 gap을 끝낸 decode forward의 미완료 요청 수 = **그 시간 동안 실제로
  기다리고 있던 요청 수**. 지연 부담(latency burden)의 단위.
- **strict 판정**: `gap_k` 안에 있던 non-decode forward 중 `pend = 0`인 것이 **하나도
  없으면** strict. 하나라도 있으면 `ambiguous`로 **따로** 집계한다.
  (`pend = 0`인 관측이 있었다는 건 그 구간의 일부에서 decode 대기열이 실제로 비어 있었다는
  뜻 — 예: 저부하에서 큐가 빈 뒤 새 요청이 도착해 prefill 되는 구간. 이걸 blocking으로
  세면 도착 간격을 blocking으로 세는 셈이다.)
  ⚠️ strict/ambiguous 경계는 **임계값이 아니라 관측된 상태**다. `pend`는 정수 카운트이고
  0 여부만 본다. 자유 상수 없음.

### 1.6 창(window) 분해 항등식 — arm 대칭성의 근거

관측 창을 `T = e_{D_{N−1}} − s_{D_0}` 로 잡으면 **네 arm 전부에서** 다음이 성립한다.

```
T  =  Σ_k dur(D_k)   +   Σ_k gap_k
   =  decode_fw_ms   +   stall_ms + stall_ambiguous_ms
```

즉 **같은 벽시계 창을 같은 세 바구니로 쪼갠다.** arm이 다른 이벤트 루프를 도는 것은
바구니의 *구성*만 바꾸고 *분할 자체*는 바꾸지 않는다. 이것이 C1(arm 대칭성) 논증의 핵심이며,
Gate 1의 실패(파티션 텔레메트리가 pdmux arm에만 존재)와 결정적으로 다른 점이다.

부수 효과: `decode_fw_ms / N` = 평균 decode forward 시간, `stall_ms / N` = decode 스텝당
추가 지연 — **둘 다 ITL과 같은 단위**다. 그래서 δ를 "A4의 decode 스텝 주기의 몇 %"로
**측정량에 앵커**할 수 있다(자유 상수를 하나 줄인다 — 다만 없애지는 못한다, §10).

### 1.7 보조 분해 (진단 전용, primary 아님)

각 `gap_k`를 다음으로 쪼갠다.

```
same_stream_fw_ms(k)  = Σ { dur(f) : f ∈ gap_k, kind(f)=other, stream(f) = stream(D_k) }
other_stream_fw_ms(k) = Σ { dur(f) : f ∈ gap_k, kind(f)=other, stream(f) ≠ stream(D_k) }
bubble_ms(k)          = gap_k − same_stream_fw_ms(k)
```

- **같은 스트림에서 실행된 non-decode forward만 gap을 실제로 채운다** (스트림 내부는 직렬
  실행이 보장되므로). 다른 스트림의 forward는 *동시* 실행이므로 gap을 채우지 않는다.
- 이 규칙은 **arm별 분기가 아니라 데이터 기반**이다: 기록된 `stream(f)`만 본다.
  A1/A2/A3는 결과적으로 전부 same-stream, A4는 전부 other-stream이 되지만, **코드는
  arm을 모른다.**
- `bubble_ms(k) < 0`이면 내부 모순(직렬성 가정 위배) → `n_inconsistent` 증가 + 그 gap을
  진단에서 제외. **정합성 자가검사**다.

또한 사전 지시(§0의 권고 정의)를 그대로 보존한 양:

```
blocking_fw_ms = Σ { dur(f) : kind(f) = other ∧ pend(f) > 0 }
```

⚠️ **`blocking_fw_ms`는 primary가 될 수 없다 — A4에서 구조적으로 0이다**(A4의 prefill은
decode와 다른 스트림에서 동시 실행되므로 decode 타임라인을 절대 채우지 않는다).
이건 항등식이고, 항등식을 증거로 쓰지 말라는 방법론 게이트의 정확한 재발이다(MEMORY
[[deconfound-measurement-lessons]] 항목 6·9). **그래서 primary는 §1.5의 `stall_ms`(합집합)이다.**
`blocking_fw_ms`는 "fused arm에서 gap이 정말 prefill로 채워졌는가"를 보이는 진단으로만 쓴다.

### 1.8 episode 분포

**episode = gap 하나** (두 decode forward 사이의 극대 정지 구간). 병합하지 않는다 —
사이에 decode forward가 있었다면 decode는 진행한 것이다.

산출: `n_episodes`, `p50/p95/p99/max(gap_k)`, 그리고 창 대비 비율 `stall_ms / T`.
정본의 "166ms@r2 → 474ms@r4"와 **같은 축**의 수치가 여기서 직접 나온다.

---

## 2. 훅 위치 (file:line) — 네 arm 전부

기준 트리: `/scratch/ehmoon/whlee/sglang_engine_dev/python` (SGLang v0.5.10), 패치 **적용 전**
줄 번호.

### 2.1 arm → 이벤트 루프 → forward 진입점

| arm | 서버 플래그 | 이벤트 루프 | `run_batch` 내부 분기 | 훅 |
|---|---|---|---|---|
| **A1** `plain` | (없음) | `event_loop_overlap` (`scheduler.py:1331`) | overlap 분기 `scheduler.py:2683-2687` | H-A |
| **A2** `plain+aux` | `--chunked-prefill-size -1 --disable-overlap-schedule` | `event_loop_normal` (`scheduler.py:1303`) | else 분기 `scheduler.py:2723-2726` | H-C |
| **A3** `plain+chunk512` | `--chunked-prefill-size 512` | `event_loop_overlap` | overlap 분기 `scheduler.py:2683-2687` | H-A |
| **A4** `agnostic` | `--enable-pdmux …` | `event_loop_pdmux` (`multiplex/multiplexing_mixin.py:952`) | **decode**: else 분기 `2723-2726` / **prefill**: split 분기 `2714` | H-C, H-B |

디스패치: `scheduler.py:3487-3503` (`dispatch_event_loop`).

### 2.2 실제 편집 지점 (5곳, 전부 `scheduler.py`)

| # | 위치(패치 전) | 내용 |
|---|---|---|
| P0 | `scheduler.py` import 블록 | `from sglang.srt.multiplex.holb_probe import maybe_create_holb_probe` |
| P1 | `scheduler.py:424` 뒤 (`init_request_dispatcher()` 직후) | `self.holb_probe = maybe_create_holb_probe(self)` |
| **H-A** | `scheduler.py:2683` (`with self.record_forward_metrics(batch):` — overlap 생성 분기) | forward 호출 앞뒤에 `begin/end` |
| **H-B** | `scheduler.py:2714` (`forward_batch_split_prefill` — pdmux split prefill) | 〃 |
| **H-C** | `scheduler.py:2723` (`with self.record_forward_metrics(batch):` — non-overlap 생성 분기) | 〃 |

**세 훅 전부 `torch.cuda.current_stream()`을 쓴다.** 그 지점에서 ambient stream이 곧 실제
실행 스트림이다:

- H-A: `with self.forward_stream_ctx:` (`scheduler.py:2680`) 안쪽 → `self.forward_stream`.
- H-B / H-C: pdmux는 `with torch.cuda.stream(prefill_stream/decode_stream)` 안에서
  `run_batch`를 부른다(`multiplexing_mixin.py:1013-1091`, `1092-1160`) → 각각 해당 스트림.
  A2는 `event_loop_normal`이 `self.schedule_stream` 컨텍스트에서 돈다(`scheduler.py:1299`).

⇒ **arm별 분기 코드 없음.** 세 훅은 문법적으로 동일하고, 어느 스트림인지는 런타임이 정한다.

### 2.3 `multiplexing_mixin.py`는 건드리지 않는다

A4의 두 forward도 전부 `run_batch`를 통과하므로 pdmux 이벤트 루프 수정이 필요 없다.
(A4에만 추가 훅을 넣으면 그 순간 arm 대칭성이 깨진다.)

### 2.4 신규 파일

`sglang/srt/multiplex/holb_probe.py` (신규, tracked).
`sglang/srt/multiplex/`에 두는 것은 **디렉터리 관례일 뿐 스코프가 아니다** — 이 모듈은
pdmux와 무관하게 네 arm 전부에서 로드된다(§7 명명 규칙 참조).

---

## 3. mixed forward 처리 (가장 미묘한 지점 — 명시적 결정)

### 3.1 이 격자에서 MIXED는 발생하지 않아야 한다

실제 게이팅은 한 줄뿐이다 (`scheduler.py:881-884`):

```
is_mixed_chunk = (chunked_prefill_size is not None) and enable_mixed_chunk
```

`--enable-mixed-chunk`는 **네 arm 어디에서도 켜지지 않는다**(A1 realized 실측
`enable_mixed_chunk=False`, PREREG §1.2). 따라서 A3의 chunked prefill도 순수 `EXTEND`
forward로 쪼개질 뿐 decode와 섞이지 않는다.

### 3.2 그래도 규칙을 사전에 고정한다

발생 시:

1. `MIXED`는 **decode-advancing으로 센다** — 그 배치 안의 decode 요청은 실제로 토큰을
   받으므로, gap을 끝내는 사건이 맞다.
2. `dur`은 `decode_fw_ms`로 들어간다. **prefill 부분의 비용이 decode forward 시간에
   섞이는 편향**이 생긴다(과대). 이건 피할 수 없으므로 `n_mixed`와 `mixed_fw_ms`를
   **항상 별도 보고**한다.
3. **`n_mixed > 0`인 arm이 하나라도 있으면 그 셀의 arm 간 비교는 크기 인용 금지**,
   부호만. (사전등록 규칙으로 고정.)

### 3.3 chunked prefill이 만드는 비대칭 — 이건 편향이 아니라 신호다

A3는 2000토큰 prompt를 512·512·512·464로 쪼갠다 ⇒ prefill forward가 **4배 많고 각각 짧다**.
`stall_ms` 정의에는 forward 개수도 길이 임계도 들어가지 않으므로, A3의 효과는

- `n_episodes` **증가**,
- `p95(gap)` **감소**,
- `stall_ms` **총합은 대체로 보존**(같은 prefill 일을 하니까; overlap 손실만큼 오히려 증가 가능)

으로 나타난다. **이건 정의의 편향이 아니라 chunking의 실제 물리다.** 세 통계(총합·episode 수·
분위수)를 함께 내는 이유가 이것이다 — 총합만 보면 chunking의 효과가 안 보이고, 분위수만 보면
총량 보존이 안 보인다.

⚠️ 반대로 **A2/A4는 chunking이 없어 prefill forward가 요청당 1개**다. 그래서
"episode 수"는 arm 간에 **직접 비교 불가**(prefill forward 개수에 종속). episode 수는
**분포 서술의 정규화 인자**로만 쓰고 결정량으로 쓰지 않는다.

### 3.4 A4의 split prefill도 "쪼개진 prefill"이다

A4의 `SPLIT_PREFILL`은 레이어 축으로 쪼개져 forward가 여러 번 발행된다
(`multiplexing_mixin.py:1119-1145`, `split_forward_token_budget` 기준). 이들은 **prefill
스트림**에서 실행되므로 decode 타임라인의 gap을 채우지 않고, §1.7의 `other_stream_fw_ms`로만
잡힌다. 정의상 일관되다.

---

## 4. arm 대칭성 논증 (C1)

### 4.1 왜 호스트 시계가 아니라 GPU 이벤트인가 — **이게 대칭성의 급소**

A1/A3는 `event_loop_overlap`, A2/A4는 non-overlap이다. overlap은 **호스트 시간과 GPU 시간의
관계 자체를 바꾼다**: iteration `i`의 호스트 벽시계는 batch `i`의 실행이 아니라 batch `i−1`의
완료 대기에 대응한다(`scheduler.py:1352-1370`의 `pop_and_process`).

⇒ 호스트 타임라인으로 "decode가 in-flight인 구간"을 그리면, overlap arm에서는 decode의
in-flight 구간이 prefill 발행 구간과 **겹쳐 보인다**(실제 GPU에서는 직렬인데도).
그러면 A1/A3의 blocking이 **체계적으로 과소평가**된다 — 즉 **"A3 ≈ A4"라는 결론 방향으로
편향**된다. 이 캠페인이 판정하려는 바로 그 방향으로.

**따라서 호스트 타임라인 기반 정의는 채택 불가.** GPU 이벤트는 이 문제가 없다:
`cudaEventRecord`는 스트림 순서로 enqueue되고, 타임스탬프는 GPU가 그 지점에 실제로 도달한
시각이다. overlap이든 아니든 같은 뜻이다.

### 4.2 세 바구니 분할이 arm과 무관함

§1.6의 항등식 `T = decode_fw_ms + stall_ms + stall_ambiguous_ms`는 정의상 항상 참이고,
좌변은 네 arm 모두 **같은 벽시계 창**이다. 계산에 arm 정보가 들어가지 않는다.

**자가 검증**: 프로브는 `T`(GPU 타임라인 합)와 `host_span_ms`(첫/마지막 decode forward의 호스트
타임스탬프 차)를 둘 다 기록한다. 비율이 1에서 크게 벗어나면 계측이 깨진 것이다.
사전 고정 기준: `|T / host_span_ms − 1| > 0.02`인 런은 **계측 무효**로 보고(판정 제외).

### 4.3 A4의 구조적 이점은 항등식인가? — 아니다 (단, 하위 항목 하나는 항등식이다)

| 양 | A4에서 구조적으로 0인가 | primary 자격 |
|---|---|---|
| `blocking_fw_ms` (§1.7) | **예** (prefill은 다른 스트림) | ❌ **금지** |
| `same_stream_fw_ms` | **예** (같은 이유) | ❌ 진단만 |
| `stall_ms` (§1.5) | **아니오** — A4의 gap = 루프 호스트 오버헤드 + green-ctx **드레인**(`multiplexing_mixin.py:1052-1080`의 `prefill_stream.synchronize(); decode_stream.synchronize()`) + `process_batch_result` | ✅ |

A4는 "prefill이 decode를 막지 않는" 대가로 (i) decode forward 자체가 느려지고(SM 축소),
(ii) 파티션 전환마다 드레인한다. **(i)은 `decode_fw_ms`에, (ii)는 `stall_ms`에 잡힌다.**
즉 이 계측은 A4에 유리한 쪽만 재지 않는다 — **A4의 알려진 두 비용이 모두 계측 안에 들어온다.**
이것이 §1.5를 primary로 고르는 실질적 이유다.

### 4.4 남는 비대칭 (숨기지 않는다)

| # | 비대칭 | 방향 | 완화 |
|---|---|---|---|
| B1 | **overlap 버블**: A2/A4는 `disable_overlap_schedule`이라 호스트 작업이 GPU 버블로 노출된다. `stall_ms`에 들어간다. | A2/A4에 불리 | **A2가 바로 이 항의 대조군**이다(A2 = A4 − pdmux). A4−A2가 pdmux 고유분, A2−A1이 플래그 묶음분. 4-arm 설계가 이걸 위해 존재한다. |
| B2 | `pend`는 overlap에서 **한 iteration 낡은** 상태(방금 발행한 decode의 결과가 아직 처리 전) | A1/A3에서 `pend`를 최대 (완료 요청 수)만큼 과대 | 부울(>0) 판정에는 거의 영향 없음. **가중량 `stall_req_ms`에만 영향** → 가중량은 보조로만 사용. `n_decode_bs` 원자료를 그대로 실어 재계산 가능하게 한다. |
| B3 | `process_batch_result`의 D2H 복사는 span 밖 → 버블로 계상 | 전 arm 공통(단 overlap arm은 숨겨짐) → B1과 같은 항 | B1과 동일 대조 |
| B4 | H-A(overlap)의 span은 `forward_stream.wait_stream(schedule_stream)` **뒤**에서 시작 / H-C는 wait 자체가 없음 | 무시 가능(schedule_stream에는 forward 작업 없음) | span 시작을 `record_forward_metrics`와 같은 지점에 맞춰 최소화함 |
| B5 | A4의 두 스트림 간 `elapsed_time`은 cross-stream | 이론상 문제없음(같은 device, 같은 GPU 글로벌 타이머). 파티션 전환 시 decode 스트림 객체가 바뀜(`stream_groups[idx]`) | cross-stream `elapsed_time`이 실패하거나 음수면 그 gap을 `n_invalid`로 빼고 **개수를 보고**. 사전 고정: `n_invalid / N > 0.01`이면 계측 무효 |

**B1은 진짜 confound다.** `stall_ms` 총합의 arm 간 차이를 "prefill이 decode를 막은 정도"로
읽으면 안 된다. 읽을 수 있는 것은 "decode가 진행하지 못한 시간"이고, 그 안의 기전 귀속은
`bubble_ms` vs `same_stream_fw_ms` 분해 + A2 대조로만 한다. 보고서에서 이 둘을 섞지 않는다.

---

## 5. 구현

### 5.1 파일

| 경로 | 상태 |
|---|---|
| `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/src/multiplex/holb_probe.py` | **신규**(정본 소스) |
| `.../src/patches/holb_probe_scheduler_hooks.patch` | **신규**(scheduler.py 미러 패치) |
| `.../scripts/bootstrap/sync_engine_tree.sh` | 설치 + manifest 등재 |
| `/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/multiplex/holb_probe.py` | 설치본 |
| `/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/managers/scheduler.py` | 패치 적용 |

### 5.2 구조

- `HolbAccountant` — **순수 파이썬 상태 기계**. CUDA를 모른다. 입력은 이미 해소된
  `(kind, pending, stream_key, dur_ms, gap_ms)`. §1.5–1.8의 회계를 전부 여기서 한다.
  ⇒ **CPU-only 단위 테스트의 대상**.
- `HolbProbe` — CUDA 이벤트 풀 + 지연 드레인 + `AsyncJsonlTelemetry` 출력.
  이벤트 팩토리를 주입 가능하게 해서 fake event로도 테스트한다.
- `maybe_create_holb_probe(scheduler)` — env 미설정이면 **`None` 반환**.

### 5.3 지연 드레인 (블로킹 없음)

`torch.cuda.Event(enable_timing=True)` 쌍을 forward마다 기록하고, **`query()`가 True인
것만** FIFO로 꺼내 `elapsed_time`을 읽는다. **스케줄러 스레드에서 동기화하지 않는다.**
이건 upstream `sglang/srt/utils/device_timer.py:23-30`(`DeviceTimer._report`)와 같은 패턴이다.

- 이벤트는 **풀에서 재사용**(free-list). 재사용은 그 이벤트의 타임스탬프를 다 읽은 뒤에만.
- 미해소 span이 `PDMUX_HOLB_MAX_INFLIGHT`(기본 4096)를 넘으면 **새 span을 버리고**
  `dropped_spans`를 센다. **절대 블로킹하지 않는다.**
- 드레인은 호출당 최대 `PDMUX_HOLB_DRAIN_BUDGET`(기본 64)개까지만.
- `close()`(atexit)**도 동기화하지 않는다** — CUDA 컨텍스트 teardown 중 블로킹 대기는
  복구 불가다. 미해소분은 `n_unresolved_at_close`로 **보고**하고 버린다(주기 요약이
  손실 상한을 잡아 준다).
- **`PDMUX_TRUE_DUAL_WORKER`와 동시 사용 금지 → 프로브 생성 시 `RuntimeError`로 fail-fast.**
  두 host issue thread는 프로브가 전제하는 단일 FIFO 순서를 깨뜨린다. Gate 2 arm은
  true-dual을 쓰지 않으므로 실사용 제약이 아니다. 그럴듯한 틀린 수를 내느니 죽는다.

### 5.4 출력

`AsyncJsonlTelemetry`(`src/multiplex/telemetry.py`) 재사용 — 별도 writer 스레드, 스케줄러
스레드에서 파일 I/O 없음. **네 arm 전부 같은 writer**(관측자 효과 대칭성).

레코드 2종:

1. `holb_gap` — decode forward마다 1개.
   `gap_ms, gap_class(strict|ambiguous|first|invalid), decode_dur_ms, decode_bs,
   n_other_fw, same_stream_fw_ms, other_stream_fw_ms, bubble_ms, pend_min_in_gap,
   host_ts, seq, stream_key, mode_counts_delta`
2. `holb_summary` — `PDMUX_HOLB_SUMMARY_EVERY_S`(기본 5s)마다 1개, 누적 카운터 전체.
   (파일이 잘려도 살아남는 요약.)

볼륨 추정: ITL 40ms → 25 rec/s → 35분 런에 ≈ 50k 레코드(≈15MB). 허용.

### 5.5 창 분할

서버 1부팅 = 여러 rate 벤치. 프로브는 `host_ts`(`time.time()`)를 실으므로, sbatch가 각
`bench_serving` 호출의 시작/종료 epoch을 사이드카(`*_windows.jsonl`)에 기록하면 분석기가
셀 단위로 자른다. **엔진 쪽에 임계값 기반 자동 분할을 넣지 않는다**(임계 도입 금지).

---

## 6. C4 — 기본 OFF, OFF일 때 경로 불변

- `PDMUX_HOLB_PATH` 미설정 ⇒ `maybe_create_holb_probe()`가 `None` ⇒ `self.holb_probe is None`.
- 세 훅은 전부 다음 형태다:

```python
_hspan = self.holb_probe.begin(batch, self.running_batch) if self.holb_probe is not None else None
<원래 forward 호출 — 한 글자도 바뀌지 않음>
if _hspan is not None:
    self.holb_probe.end(_hspan)
```

- OFF일 때 추가 비용 = forward당 `LOAD_ATTR` + `is not None` 비교 **2회**. CUDA 호출 0,
  할당 0, 시계 읽기 0, I/O 0.
- ⚠️ **정직한 단서**: "바이트 수준 동일"은 문자 그대로는 불가능하다(분기 명령 2개가 는다).
  주장할 수 있는 것은 **의미론적 불변 + 관측 불가 수준의 비용**이며, 이를 GPU
  correctness gate(§8)에서 greedy 출력 동치로, 관측자 효과 런(§9)에서 성능 동치로 각각
  실증한다.
- manifest: 이번 변경으로 `scheduler.py`와 `holb_probe.py` **2줄이 새로 추가**된다.
  873944/873945의 manifest에는 이 두 줄이 아예 없으므로 **기존 줄들의 대조는 깨지지 않는다**
  (신규 줄은 diff에서 "추가"로만 뜬다). Gate 1의 `manifest_diff_*.txt` 방식 그대로 확인 가능.

---

## 7. env 플래그 (기존 `PDMUX_*` 네임스페이스, **pdmux 전용 아님**)

| 이름 | 기본 | 뜻 |
|---|---|---|
| `PDMUX_HOLB_PATH` | `""` (**OFF**) | JSONL 출력 경로. 설정 시에만 프로브 생성. **네 arm 전부에서 동작**(pdmux 여부 무관) |
| `PDMUX_HOLB_RUN_ID` | `PDMUX_RUN_ID` → `"unlabeled"` | 태그 |
| `PDMUX_HOLB_WORKLOAD_ID` | `PDMUX_WORKLOAD_ID` → `"unlabeled"` | 태그 |
| `PDMUX_HOLB_SUMMARY_EVERY_S` | `5.0` | 요약 레코드 주기 |
| `PDMUX_HOLB_MAX_INFLIGHT` | `4096` | 미해소 span 상한(초과 시 drop, 블로킹 안 함) |
| `PDMUX_HOLB_DRAIN_BUDGET` | `64` | 드레인 호출당 최대 처리 수 |
| `PDMUX_HOLB_EMIT_GAPS` | `1` | 0이면 요약만(볼륨 축소용) |

접두사 `PDMUX_`는 **이 프로젝트의 env 네임스페이스**이지 스코프 제한이 아니다.
`HOLB` = head-of-line blocking. 모듈 docstring과 로그 배너에 이 사실을 명시한다
(서버 로그에 `HOLB probe ENABLED (all arms, pdmux=<bool>) -> <path>` 한 줄).

---

## 8. correctness gate (성능 측정 전 필수)

| # | 게이트 | 상태 (2026-08-06) |
|---|---|---|
| G0 | CPU 회귀 전체 (`python -m unittest discover -s workspace/engine-port/tests`) | ✅ **140 tests OK** (기존 102 + 신규 38) |
| G1 | HOLB 단위 테스트 — 정의 경계조건 핀 고정 | ✅ **38 tests OK** (`tests/test_holb_probe.py`) |
| G2 | `sync_engine_tree.sh` idempotent + manifest | ✅ 2회 연속 실행 manifest 동일 |
| G2b | **미적용 트리에서 from-scratch 재현** (`scheduler.py` 원본 복원 + `holb_probe.py` 삭제 후 sync) | ✅ 두 파일 SHA-256 완전 일치 |
| G2c | 미러 패치가 설치본을 **정확히** 재생성(reverse→forward 왕복) | ✅ 단위 테스트로 상시 검증 |
| G2d | 873944 reference manifest 대비 **추가 2줄, 변경 0줄** | ✅ `holb_probe.py`, `scheduler.py`만 `>`로 추가 |
| G3 | **GPU**: 네 arm × {probe ON, OFF} greedy 출력 byte-identical | ⛔ **미실행**(이번 작업 GPU 0) |
| G4 | **GPU**: 계측 무효 판정식 3종 (`\|T/host−1\|≤0.02`, `n_invalid/N≤0.01`, `dropped_spans=0`) | ⛔ 미실행 |
| G5 | **GPU**: 관측자 효과 paired < 3%, CI가 0 포함 | ⛔ 미실행 (§9) |

G0–G2d 재현 명령:
```bash
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh
python -m unittest discover -s workspace/engine-port/tests
```

G3 세부(사전 고정):
- A1/A2/A3: 벤치와 동일 생성기의 2000-토큰 프롬프트, `temperature=0`, 순차 1요청.
  ON/OFF 출력 토큰열 sha256 일치.
- A4: 배치 비결정성 때문에 ON/OFF 각각 **자기 2회 재현성** + ON vs OFF 일치(bs=1 순차).
- 추가: 동시 16요청에서도 ON/OFF의 **완료 요청 수·에러 수** 일치.

---

## 9. 관측자 효과 측정 계획 (C3) — sbatch 초안까지만, 실행은 experiment-runner

- 설계: **paired, arm × {ON, OFF}**, 같은 seed 공식(`9000 + 100·rep + rate`), rep마다 ON/OFF
  순서 무작위, **n = 4** 최소(정본 방법론 게이트 #3), 모델 = Zamba2-2.7B(r3) + Granite(r4).
- 응답변수(전부 병기): `request_throughput`, `TTFT p50/p95`, `ITL p50/p95`, `mean E2E`.
- 판정: arm별로 `(ON − OFF)/OFF`의 paired 95% CI. **|점추정| < 3% AND CI가 0 포함**이면 통과.
  ⚠️ n=4 paired의 분포무가정 검정력 한계(PREREG §4.4의 n=5 논의와 같은 구조)를 그대로 상속한다
  — 이건 **비열등 확인이지 등가 증명이 아니다.** 부족하면 n을 올린다.
- 부가: 프로브 자체의 자기보고 비용(`drain_ms_total`, `emit_ms_total`, `dropped_spans`)을
  요약 레코드로 낸다 — 외부 측정과 내부 측정을 대조.
- 초안: `workspace/engine-port/results/p1_gates/gate2/g2_holb_observer.sbatch` (**미제출**).
  Phase A(정확성) 실패 시 Phase B를 **실행하지 않고 종료**한다 — 출력이 바뀌는 계측의
  관측자 효과 수치는 의미가 없다. Phase B는 각 벤치 호출의 epoch을
  `g2holb_<tag>_windows_<jobid>.jsonl`에 적어 HOLB 스트림을 셀 단위로 자를 수 있게 한다
  (`bench_serving`이 도착 타임스탬프를 기록하지 않으므로 — PREREG §3 F-D).
  실행 예:
  ```
  sbatch g2_holb_observer.sbatch Zyphra/Zamba2-2.7B 4096 triton zamba2 3
  sbatch g2_holb_observer.sbatch ibm-granite/granite-4.0-h-micro-base 8192 flashinfer granite 4
  ```

---

## 10. ★ 이 계측이 Gate 2의 primary가 될 수 있는가 — 정직한 판정

**판정: "지금 당장은 아니다. 조건부로는 그렇다."** 근거를 분리해서 쓴다.

### 10.1 rev1–rev3를 죽인 병은 실제로 고쳐진다

- **절단 없음**: `stall_ms`는 양 끝에서 포화하지 않는다. A1은 위로 열려 있고(무한대까지),
  A4는 0이 아니다(드레인 + 루프 오버헤드). rev1의 A4 바닥 절단, rev2의 A1 천장 절단이
  **둘 다 구조적으로 발생 불가**.
- **단조 재표현 불변성**: rev2의 φ를 죽인 건 "임계 위의 비율"이라는 눈금 종속이었다.
  `stall_ms`는 물리 단위(ms)이고 단조 재표현 자체가 정의되지 않는다.
- **arm 대칭**: §4.2의 창 분해 항등식이 arm과 무관하다. Gate 1의 실패(한 arm에만 존재하는
  계측)를 반복하지 않는다.
- **기전 직접성**: 정본 §1-1이 서술한 死因을 그 서술과 같은 축(ms, 클러스터 지속)에서 잰다.

### 10.2 그런데도 오늘 primary로 제출하면 안 되는 이유 4가지

1. **δ가 여전히 필요하고, 아직 앵커할 데이터가 없다.**
   연속량이 되었다고 등가성 임계가 사라지지 않는다. §1.6 덕분에 δ를 "A4의 decode 스텝
   주기의 x%"로 **측정량에 앵커**할 수는 있지만, **x는 여전히 내가 고르는 수**다.
   더 나쁜 건 **rep 간 분산을 전혀 모른다**는 점 — δ와 검정력을 사전등록할 근거가 없다.
   rev3의 δ=0.05는 최소한 873944/873945 실측 분산 위에서 잡혔다. **이 지표는 그게 없다.**
   ⇒ **파일럿(n≥4, 두 모델, 채점 셀) 없이는 사전등록 불가.**
2. **primary가 답하는 질문이 Gate 2의 질문과 완전히 같지 않다.**
   Gate 2는 "A3가 A4를 **대체**하는가"를 묻는다. 대체는 (i) 기전 등가 **AND** (ii) 결과
   비열등이다. `stall_ms`는 (i)만 잰다. A3의 알려진 위험은 정확히 (ii) 쪽이다 — 지연을
   ITL에서 **TTFT로 옮기는 것**(PREREG §5.2 F2, Granite agnostic r10/r12에서 `frac_stalled=0`
   인데 TTFT p50 1827/2238ms). ⇒ **단독 primary가 아니라 연언의 한 항**이어야 한다.
   (rev3의 R3′와 같은 구조. 통계량만 바뀌고 연언 구조는 남는다.)
3. **B1(overlap 버블) confound가 총합에 섞여 있다.**
   `stall_ms` 총합의 arm 간 차이는 "prefill blocking"이 아니라 "decode 미진행"이다.
   기전 귀속은 `same_stream_fw_ms` vs `bubble_ms` 분해 + A2 대조로만 가능하고, **그 분해는
   arm 간에 구조가 다르다**(A4는 same_stream이 항등적으로 0). 즉 **"총합은 대칭, 분해는
   비대칭"** — 보고서가 이 둘을 섞으면 즉시 감사에서 죽는다.
4. **아직 한 번도 돌지 않았다.** 관측자 효과·계측 무효 판정식·cross-stream `elapsed_time`
   건전성(B5) 전부 미실증. §8의 G3–G5가 전부 ⛔이다.

### 10.3 그래서 권고

- **rev3 제출 경로를 죽이지 마라.** rev3은 사전등록 가능한 δ를 가진 유일한 설계다.
  이 계측을 rev3 캠페인에 **`PDMUX_HOLB_PATH`만 켜서 동승**시키면(추가 GPU 시간 0),
  같은 런에서 (a) rev3의 primary와 (b) 기전 직접 계측을 **동시에** 얻는다.
  관측자 효과 게이트만 먼저 통과하면 된다.
- **HOLB를 primary로 승격하는 조건**(사전 고정 제안):
  1. §8 G3–G5 전부 통과,
  2. 파일럿에서 rep 간 CV와 MDE를 측정해 δ를 **측정량 앵커 방식**으로 고정,
  3. **연언**: `stall_ms` 등가 **AND** TTFT p95 비열등 **AND** ITL p95 비열등,
  4. 보고서에서 `blocking_fw_ms`/`same_stream_fw_ms`를 **절대 결정에 쓰지 않음**을 명문화
     (A4 항등식).
- 이 계측 자체는 **primary 여부와 무관하게 값이 있다**: §1-1의 "166→474ms" 서술을
  간접 지표가 아니라 **엔진에서 직접 잰 수치**로 교체할 수 있다. 그건 통계량 논쟁과
  독립인 순수 이득이다.

---

## 11. 알려진 위험 요약

| 위험 | 심각도 | 상태 |
|---|---|---|
| B1 overlap 버블이 `stall_ms`에 섞임 | **높음** | A2 대조로 분리, 보고 규칙 명문화(§4.4) |
| `blocking_fw_ms`가 A4에서 항등식 | **높음** | primary 금지 명시(§1.7) |
| δ 사전등록 근거 부재 | **높음** | 파일럿 필요(§10.2-1) |
| B5 cross-stream `elapsed_time` | 중간 | `n_invalid` 계측 + 무효 판정식(§4.4) |
| B2 overlap `pend` 지연 | 낮음 | 가중량만 영향, 원자료 병기 |
| MIXED 발생 시 편향 | 낮음(발생 안 해야 함) | `n_mixed` 계측 + 발생 시 크기 인용 금지(§3.2) |
| 프로브 자체 비용 | 미측정 | §9 관측자 효과 런 |
| true-dual-worker와 동시 사용 시 순서 붕괴 | 낮음 | 프로브 생성 시 `RuntimeError` fail-fast(§5.3) |

---

## 12. 산출물 색인

| 경로 | 내용 |
|---|---|
| `workspace/engine-port/src/multiplex/holb_probe.py` | 정본 소스(`HolbAccountant` = 순수 파이썬 회계, `HolbProbe` = CUDA 이벤트 층) |
| `workspace/engine-port/src/patches/holb_probe_scheduler_hooks.patch` | `scheduler.py` 미러 패치(5 hunk) |
| `workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh` | 설치 + grep-guard 패치 + manifest 2줄 추가 |
| `workspace/engine-port/tests/test_holb_probe.py` | CPU 회귀 38건 |
| `workspace/engine-port/env/dev_tree_edits.md` 항목 18–19 | dev-tree 편집 기록 |
| `results/p1_gates/gate2/g2_holb_observer.sbatch` | GPU 게이트 초안(미제출) |
| `results/p1_gates/gate2/pdmux_a100_smoke.yml` | A4 pdmux 설정(873944/873945와 동일 파일) |
