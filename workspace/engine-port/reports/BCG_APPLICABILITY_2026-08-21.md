# Breakable CUDA Graph (BCG) — prefill-layer-alloc 적용 타당성 평가와 P0 설계 (2026-08-21)

> **이 문서는 정본이 아니다.** 새 성능 판정 0건 · 등급 변경 0건 · 정책 순위 변경 0건 ·
> GPU 지출 0. 전부 **코드 사실 + 이미 등재된 측정**의 재조합이며, 실행 전 결정을 위한
> 설계·타당성 문서다. 정본 위계(`PROJECT_STATUS.md` > `reports/paper/` >
> `reports/CONSENSUS.md`)는 이 문서로 갱신되지 않는다.

대상 기능: <https://docs.sglang.io/docs/advanced_features/breakable_cuda_graph>
(`@eager_on_graph` / `break_graph()` / `SGLANG_USE_BREAKABLE_CUDA_GRAPH=1` / `--debug-cuda-graph`).

---

## 0. 읽는 순서

1. **§1 왜 이 기능이 이 프로젝트와 만나는가** — 정본에 이미 등재된 두 개의 열린 상처.
2. **§2 검증표** — 이 조사에서 *직접 확인한 것* / *추정한 것*(방법론 게이트 #28).
3. **§3 BCG가 실제로 무엇을 주고 무엇을 안 주는가** — 기능의 두 다리 분해.
4. **§4 이점 시나리오 S1–S4** — 조건과 함께, 순위대로.
5. **§5 이점이 없는 지점** — 명시적으로.
6. **§6 손실 L1–L6.**
7. **§7 P0 프로브 설계** — 이거 통과 전엔 §4 어느 것도 시작하지 않는다.
8. **§8 이것으로도 닫히지 않는 것** · **§9 하지 말 것** · **§10 비용** · **§11 다음 액션**.

**한 문장 요약**: BCG의 가장 큰 가치는 **layer-aware 부활이 아니라(그건 여전히 死)**,
① **운영점(cudagraph-ON)에서 처음으로 per-layer 계측을 가능하게 하는 것**과
② **이 프로젝트 운영점에서 유일하게 남은 완전-eager 구성요소인 prefill을 그래프화할
가능성**이다. 그리고 이 둘 다 **아직 아무도 안 잰 상위 게이트 하나(§7 P0-A)** 뒤에 있다.

---

## 1. 왜 하필 이 기능인가 — 정본에 등재된 두 개의 열린 상처

### 1.1 상처 A — "커스텀 sub-graph 캡처가 大공수라 미실행" (`cudagraph_results.md:129-136`)

정본 부속 문서가 layer-aware에 남은 마지막 fair-shot으로 **per-window cudagraph**를
지목하고, 3개 사유로 **미실행 권고**했다:

> (i) 창 사이 green-ctx 재분할 drain은 그대로 …, (ii) **sglang cudagraph는
> full-decode-forward를 캡처하지 실행 중 임의 층-범위 sub-graph를 캡처 안 함 →
> 커스텀 캡처 필요(大공수)**, (iii) 예상 payoff는 낮음(위 34× 격차의 대부분이 drain).
> **권고: 미실행**(공수 대비 futile 예상).

**BCG는 (ii)를 정확히 겨냥한 업스트림 인프라다.** (i)·(iii)은 건드리지 않는다 — §4-S2에서
이게 왜 결론을 바꾸지 못하는지 정량으로 다룬다.

### 1.2 상처 B — "per-layer 계측은 운영점에서 실행되지 않는다" (구조적 제약)

`handoff-report/session_handoff_2026-08-13.md:150-153` 및
`results/ltsm_probe/PREREG_LTSM_P1_PROBE_2026-08-14.md:49`:

> `cuda_graph_runner.py:547` `capture_forward_mode=DECODE`, `:1161` `replay()`가
> **Python forward를 호출하지 않는다** ⇒ **per-layer 계측은 cudagraph-ON에서 실행되지
> 않는다.** 운영점이 cudagraph-ON이므로 이 캠페인은 **정의상 off-operating-point**.

즉 이 저장소의 **모든** per-layer 계측은 구조적으로 비운영점이다. BCG의 break point는
그래프 replay 도중 호스트로 제어를 돌려주므로, 이 제약의 **성격 자체**를 바꾼다.

이 상처는 2026-08-20 이후 더 아프다. `CONSENSUS.md` §3 항목52 追記(6)이
`UNAVAILABLE (CUPTI×GREEN-CONTEXT)`를 확정하면서 `kernel_mech`의 Stage B(ncu 커널 내부
카운터)가 폐기됐고, 고-SM decode 평탄화의 4개 후보 기전 중 하나가 문자 그대로
**"cudagraph 직렬화"** 인데 그걸 잴 수단이 Stage A(nsys) 하나로 줄었다.

---

## 2. ★ 검증표 — 확인한 것 / 추정한 것

### 2.A 이 조사에서 **직접 확인**한 것 (전부 read-only, GPU 0)

| # | 사실 | 근거 |
|---|---|---|
| **F1** | **BCG는 이 트리에 없다.** `grep -rn "breakable\|eager_on_graph\|BREAKABLE\|debug-cuda-graph\|debug_cuda_graph" --include=*.py sglang_engine_dev/python/sglang/` → **0 hit**. `model_executor/runner_backend_utils/` 디렉터리 자체가 부재 | dev tree (v0.5.10) |
| **F3** | **graph를 타는 건 decode뿐.** `cuda_graph_runner.py:547` `capture_forward_mode = ForwardMode.DECODE`; `forward_batch_info.py:167-173` `is_cuda_graph()` = `{DECODE, TARGET_VERIFY, IDLE, DLLM_EXTEND}` — **`SPLIT_PREFILL` 불포함**(`is_extend()` `:112-121`의 `:119`에만 들어감); `model_runner.py:2866` `elif forward_batch.forward_mode.is_split_prefill(): ret = self.forward_split_prefill(...)`. ⇒ **운영점에서 prefill은 100% eager** | 코드 |
| **F4** | 그래프는 **(stream_group × bs)** 로 부팅 시 전량 사전 캡처: `cuda_graph_runner.py:806-817` `for i, sg in enumerate(self.stream_groups): with graph_capture(stream=sg[1]) …: _capture_one_stream(i)`, key `f"{stream_idx}_{bs}"`(`:798`, `:681`, `:1158`). **스위치 시 재캡처 없음** | 코드 + `PROJECT_STATUS.md:2727-2729` |
| **F5** | 캡처 실측(같은 bs 리스트 `[1,2,4,8,12,16,24,32,40,48]`): **pdmux 11.20 s / 0.83 GB** vs **plain 3.97 s / 0.13 GB** (≈2.8× 시간, ≈6.4× 메모리) | `srv_pdmux_cgON_847391.log:59-70`, `srv_cgON_847296.log:46-51` |
| **F6** | 스위치는 **iteration 경계에서 2× `synchronize()` drain 후에만** 발생(`multiplexing_mixin.py:1053-1068`, `:1070-1083`) ⇒ **현재 설계엔 캡처/재생 ↔ 스위치 충돌이 없다.** replay는 항상 그 replay가 issue되는 바로 그 스트림에서 캡처된 그래프를 쓴다 | 코드 |
| **F7** | **switch 비용은 死因이 아니다**: switch 2회 rep가 static 매칭, **`slo(5sw) < bind(21sw)`**; 컨트롤러 CPU = mean 32–36 µs / **wall clock의 0.014%** | `CONSENSUS.md:1358`(§1-8), `:1362`(§1-12) |
| **F8** | **싼 전환은 이미 사봤다**: `PDMUX_LA_COORD_OPT`가 경계 `stream.synchronize()` drain을 GPU측 wait_stream 순서화로 교체 → TPOT **124→85 ms(갭 ~47% 회수)**, **그래도 agnostic 42 ms 평탄에 패배**. 잔차 = 구조적 오버랩 손실 + cudagraph 비양립 | `CONSENSUS.md:1391`(§1-15) |
| **F9** | **상위 게이트가 미측정**: green-ctx 스트림에서 **캡처된** CUDA graph를 replay할 때 그 green context의 SM 제한이 전달되는지 **아무 아티팩트도 잰 적이 없다**(구멍 C / 가정 E5). 문서 자신이 *"여기서 방향을 단정하지 않는다"* 라고 명시 | `SMID_DIRECT_INSTRUMENTATION_DESIGN_2026-08-11.md:77-96`, `:145` |
| **F12** | PD-mux는 **chunked prefill 금지**(`server_args.py:6130-6131` `assert self.chunked_prefill_size == -1`), pp>1·disaggregation·overlap schedule도 금지(`:6126-6137`). 대신 **층축** split(`split_forward_token_budget`, 기본 65536, `pdmux_context.py:20`). 업스트림은 **torch ≥ 2.7에서 green-ctx × cudagraph 성능 저하**를 경고(`server_args.py:6139-6146`) — **현 환경 torch 2.9** | 코드 |
| **F13** | `CURRENT_STREAM_IDX`는 **모듈 전역**(`pdmux_context.py:10`, `:144-149`) — ContextVar가 아니다. thread-local로 바뀐 건 TP role 플래그뿐(`src/patches/pdmux_thread_local_role.patch`) | 코드 |
| **F14** | Zamba2는 ABAB 교대라 `la_coord_windows()`(=maximal same-type run)가 **창 ~19개**로 쪼개진다 | `models/zamba2.py:791-804` + `cudagraph_results.md:78` |
| **F15** | `%smid` L0 census 기계(`results/smid_census/smid_l0_census.py`, 829줄)는 **이미 작성돼 있고 사전등록**(`PREREG_SMID_R0_2026-08-14.md`)까지 됐으나 **GPU에서 실행된 적이 없다** — 디렉터리에 `smid_l0_cpu_selftest_*.json`만 있고 `smid_l0_raw_*.json`이 없다. `_census_once(kernel, stream, …)`(`:329-340`)이 임의 스트림을 받는 형태라 **P0-A의 기반으로 그대로 쓸 수 있다** | 코드 + 디렉터리 |
| **F16** | ★**(이 조사에서 새로 발견) SGLang의 1급 훅 확장점 `--forward-hooks`로는 캡처된 그래프 안에 break를 넣을 수 없다.** `model_runner.py:656` `init_device_graphs()`(=캡처; npu/cpu 경로에 `:659`로 하나 더)가 `:666` `register_forward_hooks(self.model, server_args.forward_hooks)`(가드 `:665`)**보다 먼저** 실행된다. ⇒ `--forward-hooks`로 붙인 훅은 캡처된 decode 그래프에 부재하고 eager 경로에서만 발화한다. 업스트림 측 수정(훅 등록을 캡처 위로 이동)은 한 줄이지만 **dev tree 변경**이다. 대신 `ModelRunner.load_model`(정의 `:1072`, 호출 `:508` — 둘 다 캡처 이전)을 감싸면 디스크에 쓰지 않고 주입할 수 있다 | 코드 |

### 2.A′ **업스트림 인용 — 이 저장소에 스냅샷이 없다** (교훈 #31)

아래 3건은 §2.A와 달리 **로컬에서 대조할 아티팩트가 없다**(`find … -name runner_backend_utils`
→ 0건). 특히 **F10이 틀리면 §3 다리①과 §4-S2의 논증 구조가 바뀐다.** 백포트(P0-B) 시
커밋 SHA와 스냅샷 경로를 여기에 등재하고, 그 전까지는 §2.A와 같은 강도로 인용하지 않는다.

| # | 인용 | 출처 |
|---|---|---|
| **F2** | 업스트림 모듈 = `model_executor/runner_backend_utils/breakable_cuda_graph/{__init__,breakable_cuda_graph,context,cuda_utils}.py`, 본체 ≈450줄. sglang 내부 의존은 `srt.utils` + `srt.compilation.weak_ref_tensor` 뿐이고 **후자는 이 트리에 이미 존재**(`srt/compilation/weak_ref_tensor.py`), `cuda.bindings`도 venv에서 import 성공 ⇒ **백포트 난이도 낮음** | 업스트림 + 로컬 확인 |
| **F10** | stock BCG replay 루프는 `for i, seg in enumerate(self._segments): seg.replay(); if i < len(self._break_fns): self._break_fns[i]()` — 세그먼트를 **replay 시점의 현재 스트림**에 launch. **세그먼트별 green-ctx 재배치는 stock에 없다** | 업스트림 소스 |
| **F11** | 업스트림 BCG의 헤드라인 용처는 **prefill**: *"For prefill, Breakable CUDA Graph is now SGLang's default"*, eager 대비 **1.70×**(full capture 1.93×, TC piecewise 1.45×; gpt-oss-120b TP4 4×GB300 prefill-only), 컴파일 기반 대비 그래프 빌드 **3.8–5.2× 빠름**. 메모리 사례: GLM-5.2에서 42 shapes = **2.4 GB** | LMSYS 블로그 2026-08-17 |

### 2.B 이 문서가 **추정**한 것 (검증 안 됨 — 프로브 필요)

> ★**번호 체계 주의**: B1–B5는 **이 문서 고유 번호**다.
> `SMID_DIRECT_INSTRUMENTATION_DESIGN_2026-08-11.md`의 E1–E9와 **다른 체계**이며,
> 이 문서의 **B4 = 그 문서의 E5**(구멍 C)다. 혼동을 막으려 접두어를 B로 바꿨다.

| # | 추정 | 왜 미검증인가 | 해소 |
|---|---|---|---|
| B1 | BCG 백포트가 v0.5.10 트리에서 import**되고 엔진이 실제로 그 경로를 탄다**. ★F1이 확인한 대로 이 트리엔 `debug-cuda-graph`/`debug_cuda_graph` 문자열이 **0 hit**이므로 서버 플래그 배선(`server_args.py`)과 캡처 라우팅(`cuda_graph_runner.py`)은 F2의 "4파일"에 **없다** ⇒ 백포트는 **3층**(모듈 4파일 + 플래그 + 호출부)이고 "난이도 낮음"은 모듈 층에만 해당 | F2는 *의존 이름*만 확인했다. 시그니처 드리프트·배선 층은 안 봤다 | §7 P0-B stage 0 (세그먼트 수 양성대조) |
| B2 | BCG의 `torch.cuda.Stream.wait_stream` 후킹이 `graph_capture`(`parallel_state.py:483-506`)·`ExternalStream`·true-dual-worker의 event 부기와 충돌하지 않는다 | 세 경로 모두 wait_stream/record_event를 무겁게 쓴다. 상호작용 미측정 | §7 P0-B |
| B3 | per-break 비용이 이 기판·이 모델에서 ITL의 3% 미만이다 | 업스트림 문서가 per-break 비용을 **정량화하지 않았다**("overhead remains negligible"은 서술) | §7 P0-B(iii) |
| B4 | green-ctx 스트림에서 캡처한 그래프가 replay 시 SM 한정을 전달한다 | **F9. 아무도 안 쟀다** | §7 P0-A |
| B5 | prefill 캡처 키가 (span 길이 × token 버킷 × stream_group)로 유한하게 닫힌다 | span 길이가 `split_forward_token_budget // extend_num_tokens`로 **입력 길이 의존 가변**(`multiplexing_mixin.py:1128-1141`) | §4-S3 설계 필요 |

---

## 3. BCG가 실제로 주는 것과 안 주는 것 — 기능의 두 다리 분해

`cudagraph_results.md:78`가 불가능성을 두 다리로 서술했다:

> cuda graph 캡처는 **① 고정 스트림** 위 **② 고정 op 시퀀스**를 요구한다.

- **다리 ② (고정 op 시퀀스) — BCG가 자른다.** 세그먼트 사이에 eager 함수를 넣을 수 있다.
  이게 §1.1 상처 A의 (ii)를 해소하는 정확한 지점이다.
- **다리 ① (고정 스트림) — BCG는 못 자른다.** F10: stock replay는 세그먼트를 *현재 스트림*에
  올릴 뿐이고, 세그먼트마다 **다른 green context**로 옮기는 기능은 없다. 그러려면
  세그먼트 × stream_group 으로 각각 캡처하고 세그먼트 경계마다 cross-stream event
  순서화를 넣는 **추가 커스텀**이 필요하다.

⇒ **BCG는 "cudagraph ⊥ sub-step layer-aware"의 절반만 해소한다.** 이 문장이 §4 전체의 뼈대다.

부수적으로, `--debug-cuda-graph`는 **전체 forward를 eager로 돌리되 graph runner 경로
(입력 버퍼·패딩·`replay_prepare`·per-group attn backend)는 그대로 지난다.** 이건 경로 자체가
갈리는 현행 `--disable-cuda-graph`와 **다른 종류의 대조군**이다(→ S4).

---

## 4. 이점이 생기는 지점 — 시나리오와 성립 조건

### S1 — 운영점 계측 복원 ★최고 가치, 성능 주장 아님

**무엇**: decode forward의 층(또는 층타입 창) 경계에 `@eager_on_graph` break를 넣고, 그
eager 갭에서 CUDA event를 record한다. 그래프는 **여전히 캡처된 채**로 남고, break 지점만
호스트로 돌아온다.

**왜 가치 있나**: §1.2의 구조적 제약 — 이 저장소의 per-layer 계측은 전부 정의상
off-operating-point다. BCG break는 **운영점에서 실행되는 첫 per-layer 계측 경로**다.
2026-08-20에 ncu×green-ctx가 `UNAVAILABLE`로 죽어 Stage A(nsys)만 남은 `kernel_mech`
트랙에 세 번째 경로가 생긴다. 특히 "cudagraph 직렬화"는 고-SM decode 평탄화 4개 후보 중
하나인데, **BCG break로 세그먼트별 체류시간을 재면 직렬 사슬 가설을 직접 겨눈다**.

**성립 조건 (전부 만족해야 함)**:
1. P0-B가 백포트·부팅·wait_stream 충돌을 통과할 것(B1·B2).
2. **null-break 대조**(브레이크 0개, BCG 경로 ON)로 per-break 비용을 측정해 ITL의
   **3% 미만**임을 먼저 보일 것 — 방법론 게이트 #3이 3% 미만을 headline으로 안 치므로,
   perturbation이 그 이상이면 이 계측은 자기가 재려는 양을 오염시킨다(B3).
3. 브레이크 개수를 층 수가 아니라 **소수의 macro 경계**로 제한할 것. F14 때문에 Zamba2에서
   층타입 창을 다 쓰면 step당 19 브레이크다.

**구현상 함정 (F16)**: 가장 자연스러워 보이는 주입 경로인 `--forward-hooks`는 **못 쓴다**.
훅 등록(`model_runner.py:666`)이 캡처(`:656`)보다 **뒤**라 훅이 캡처된 그래프에 안 들어간다.
주입은 캡처 이전 지점(`ModelRunner.load_model`, 정의 `:1072`)에서 해야 한다 —
P0-B의 `p0b_break_shim.py`가 dev tree를 건드리지 않고 그렇게 한다.

**한계 (반드시 병기)**: 이건 **타임라인 해상도**이지 커널 내부 카운터가 아니다.
wave quantization·occupancy는 여전히 못 잰다. Stage B의 대체재가 아니라 Stage A의
in-process 대안이다.

### S2 — per-window cudagraph 트랙: 비용은 내려가나 payoff는 그대로 ★권고: 여전히 미실행

**BCG가 해소하는 것**: `cudagraph_results.md:129-136`의 사유 **(ii)만**. "大공수 커스텀 캡처"가
부분적으로 업스트림 인프라가 된다.

**BCG가 해소하지 못하는 것 — 그리고 이게 결론을 지배한다**:
- (i) 창 사이 green-ctx 재분할 **drain 그대로**. 그리고 정본이 이 잔차를
  **윈도우수 무관·모델 독립**으로 등재했다 — `CONSENSUS.md:1391`: 싼 전환으로도 못 지우는
  잔차 = **구조적 오버랩 손실**(monolithic prefill이 window 0만 오버랩). 세그먼트를 아무리
  잘 쪼개도 이 성분은 그대로다.
  ⚠️ **F8의 수치(124→85 ms, 47%)는 이 bullet의 근거가 아니다.** 그 캠페인은
  `a_substrate/a_coord_opt_bench.sbatch:41` `--disable-cuda-graph`이고 워크로드당 **n=1**이며,
  `A_substrate_isolation_results.md:35`가 *"R0d의 '124ms'는 CPU-sync+pinning으로 부풀려진 값"*
  이라고 스스로 표시했다. `CONSENSUS.md` §1-2가 "기존 no-cudagraph 수치는 전부 하한"이라
  못 박은 이상, **cgOFF n=1 값을 cudagraph 상태를 바꾸는 것이 존재 이유인 트랙의 상한으로
  쓸 수 없다.** F8은 "싼 전환 축이 존재하고 그것만으로는 부족했다"는 *방향*까지만 지지한다.
- (iii) 낮은 payoff 그대로. 대조가 `cudagraph_results.md:122-125`에 있다:
  sub-step 실현(lacoord) gp 0.027 vs 자기 step-fixed degenerate(d16) gp 0.913.
  ⚠️ **이 34× 비를 payoff 상한으로 쓰지 않는다.** 같은 파일 `:51-52`가 lacoord arm을
  *"`forward_split_decode`=eager, cudagraph replay 불가 → CAVEAT: decode 미가속"* 으로
  표시했으므로 이 비는 **cudagraph 상태를 가로지르는 비교**(eager decode vs graph decode)이고,
  n=1이며, rate4 prefill-bound 붕괴 구간(metric cliff)이다. 게다가 그 eager-decode 페널티는
  per-window cudagraph가 **없애려는 대상**이라 분자에 들어가 있으면 안 된다. 원 파일 자신이
  `:84`("=cudagraph 격차 그대로")와 `:135`("34× 격차의 대부분이 drain")로 서로 다른 귀속을
  적어 두었다.
- **다리 ①(§3)**: F10 때문에 세그먼트별 SM 재배치는 stock BCG 밖이다. 세그먼트 ×
  stream_group 캡처 + cross-stream event 순서화라는 커스텀이 여전히 남는다.
- **F14**: Zamba2 ABAB → 창 ~19개 = step당 브레이크 19개. 브레이크가 적어야 유리한데
  이 모델은 최악 조건이다. ★**그렇다고 "창이 적은 모델이면 살아난다"로 빠져나가지 말 것** —
  `CONSENSUS.md:1391`이 잔차를 **윈도우수 무관·모델 독립**으로 등재했으므로 그 탈출구는
  정본이 이미 닫았다. F14는 死因을 하나 더 얹을 뿐, 조건부 부활 경로가 아니다.

⇒ **BCG는 이 실험의 비용을 낮출 뿐, 기대 payoff를 올리지 않는다.**
`cudagraph_results.md`의 미실행 권고는 **유지**한다. ★**이 NO-GO를 떠받치는 것은 F8도
34× 비도 아니고**, `CONSENSUS.md:1391`(구조적 오버랩 손실 — 윈도우수 무관·모델 독립)과
§1-3(layer-type 런타임 정책 4모델 서빙 직접 측정 반증)이다. 그 둘은 cgOFF n=1 근거에
의존하지 않으므로, 위 두 ⚠️ 를 걷어내도 판정은 그대로 선다.

**설계상 부수 발견(기록용, 주장 아님)**: 세그먼트 × stream_group 캡처의 *메모리* 총량은
직관보다 작다. 세그먼트 합 = 전체 forward 1개분이므로, 2개 그룹(protect/release)만 쓰면
2 그룹 × |bs| × (전체 1개분) 이고 이는 현행 8 그룹 × |bs| × (전체 1개분)보다 **작다**.
늘어나는 건 메모리가 아니라 **캡처 begin/end 횟수**(≈ 창 수 배)다. — 이건 F5의 실측을
외삽한 산술이지 측정이 아니다.

### S3 — prefill 그래프화 ★최대 잠재 이득, 완전 미탐색

**논거 세 줄**:
1. F11: 업스트림 BCG의 헤드라인 용처가 **정확히 prefill**이고, eager 대비 **1.70×**다.
2. F3: 이 프로젝트 운영점에서 **prefill은 유일하게 남은 완전-eager 구성요소**다.
   decode는 2026-07-13에 이미 벽을 넘었다(TPOT 41→12 ms).
3. `cudagraph_results.md`가 cudagraph-ON 하 decode-heavy에서 *"전 TPOT<60ms라 decode SLO는
   고rate까지 여유, **goodput은 TTFT가 지배**"* 라고 적었다 — 남은 레버가 TTFT 쪽에 있다.

**장애 (B5, 해결해야 실행 가능)**: F12 때문에 PD-mux는 chunked prefill을 못 쓰고 층축
split을 쓴다. 캡처 키가 **(span 층수 × token 버킷 × stream_group)** 이라 조합이 열린다.
`forward_count = split_forward_token_budget // extend_num_tokens`(`multiplexing_mixin.py:1128-1141`)가
입력 길이 의존이라 span 층수가 가변이다. ⇒ **span 층수를 고정 버킷으로 강제**해야
캡처가 닫히고, 그러면 **컨트롤러의 스위치 해상도가 그 버킷에 묶인다**(순수 이득이 아니라 교환).

**검증이 필요한 예측 (주장 아님)**: decode 그래프화는 tuned 최적을 **d24 → d16으로 작게**
밀었다(`cudagraph_results.md` Probe 4: attn-decode가 싸져 decode를 16 SM까지 굶겨도
TPOT<SLO → prefill이 SM을 더 받음).
⚠️ **이 전제부터 약하다**: Probe 4는 셀당 **n=1**이고, "밀었다"는 **r3·r4에서만** 성립하며
(r2는 d24가 이긴다), r4의 차이는 **2.7%**로 게이트 #3 문턱 미만이고, **동일 격자의 cgOFF
arm이 없어** rate 축과 cudagraph 축이 분리되지 않는다. 전제도 단언하지 않는다.
그럼에도 대칭으로, **prefill 그래프화는 최적을 반대로
(decode SM ↑) 되밀 것**으로 예측된다. **이건 예측일 뿐이고 방법론 게이트 #1(정책 주장은
반드시 서빙 실증) 대상이다. 측정 전에 순위 진술로 쓰지 않는다.**

**부수 참고**: 기존 `piecewise_cuda_graph_runner.py`는 prefill 그래프화의 *기존* 레버지만
**pdmux 인식이 0**이고(stream_idx 키잉 없음) `SPLIT_PREFILL`은 애초에 `forward_extend`에
도달하지 않아 그 경로를 안 탄다. 게다가 전 PD-mux sbatch가
`--disable-piecewise-cuda-graph`다. F11의 "빌드 3.8–5.2× 빠름"이 BCG를 piecewise보다
현실적 후보로 만든다.

### S4 — correctness / debug 오라클 ★저비용, 단 공짜는 아니다

⚠️ **선결**: `--debug-cuda-graph`는 **이 트리에 없다**(F1). 따라서 S4도 백포트를 요구하고,
따라서 **L1(기판 변경 = 비교가능성 손실)을 지불한 뒤에만** 쓸 수 있다.

`--debug-cuda-graph`는 graph runner 경로(버퍼·패딩·`replay_prepare`·per-group attn backend)를
유지한 채 전부 eager로 돌린다. 현행 `--disable-cuda-graph`는 **경로 자체**가 갈린다.
⇒ 두 용도:
- **engine-porter의 correctness gate**: PD-mux 패치가 green ctx 하에서 그래프와 같은 수를
  내는지 훨씬 타이트하게 미분(differential) 검사.
- **"cgOFF vs cgON 격차" 귀속 정밀화**: 현재 62→13.5 ms 격차(⚠️ Zamba2 단일모델·n=1 관측 —
  `PROJECT_STATUS.md:980-982` "직접 이식은 아니다")에 "graph replay 효과"와
  "graph runner 버퍼/패딩 효과"가 섞여 있다. `--debug-cuda-graph` arm이 그 둘을 가른다.

---

## 5. 이점이 **없는** 지점 (명시)

1. **HE0(단일 GPU 동적 제어)는 안 되살아난다.** 死因은 switch 비용이 아니라
   **positioning + 공유 running-batch/KV 얽힘**이고(F7: switch~0 rep가 static 매칭,
   `slo(5sw) < bind(21sw)`, 컨트롤러 CPU 0.014%), BCG는 둘 중 어느 것도 건드리지 않는다.
   `CONSENSUS.md:3277`이 열린 lever로 지목한 건 **admission-control / KV-aware**지 그래프가 아니다.
2. **prefill 측 mux 인터리빙**: 이미 층 단위로 손수 구현돼 있다(`split_forward_count` —
   사실상 hand-rolled breakable execution). BCG가 새로 주는 인터리빙은 없다.
3. **캡처 조합 폭발·그래프 메모리**: BCG는 완화하지 않는다. 세그먼트 수만 늘린다(F5 위에 가산).
4. **decode step의 선점(preemption)**: break는 CPU 제어를 돌려줄 뿐, in-flight decode step을
   중단해 prefill에 SM을 넘기지 못한다. 얽힘 기전(decode 굶김→ITL↑→admission 차단→TTFT 폭발)에
   개입할 수단이 아니다.

---

## 6. 손실

| # | 손실 | 크기·근거 |
|---|---|---|
| **L1** | **기판 변경 = 비교가능성 손실** (최대 비용) | 엔진 트리 업그레이드면 저장소 내 **97개 `.sbatch` 캠페인의 basis가 무효**. 백포트여도 `sync_engine_tree.sh` manifest에 새 항목이 들어가고, 방법론 교훈 #31(수입한 보조 수치는 basis 검증)이 그대로 걸린다 |
| **L2** | **캡처 시간·메모리** | F5의 pdmux 8× (11.20 s / 0.83 GB) 위에 세그먼트당 begin/end capture가 가산. prefill까지 캡처하면 F11의 "42 shapes = 2.4 GB" 급 → `--mem-fraction-static 0.82` 하에서 KV/mamba state 압박 → **decode batch 도달성을 더 좁히는 방향**. ⚠️ 참고 인용한 "ctx4096 SM92 B≥9 도달률 0.0–3.0%"는 **7–8B arm·ctx4096 격자**이고 `CONSENSUS.md:2365-2370`이 *"메인 세션이 독립 확인한 것은 T8 행 하나뿐 … 도달률 %는 재현하지 않았다"* 로 provenance를 제한했다. Zamba2-2.7B 구성으로 **이식하지 않는다**(교훈 #31) — 방향 진술로만 쓴다 |
| **L3** | **운영점 오염** | 브레이크당 `cudaGraphLaunch` 1회 + eager Python 호출 1회. **request-내부 token-ITL p95가 goodput 정의에 직접 들어가고**(방법론 게이트 #4) 게이트 #3이 3% 문턱이라, perturbation이 주장 가능한 효과와 **같은 오더**일 수 있다. 업스트림 문서는 per-break 비용을 정량화하지 않았다(B3) |
| **L4** | **호환 리스크** | `SGLANG_MEMORY_SAVER_CUDA_GRAPH`와 비양립(업스트림 명시); `cuda-python` 필요(로컬 `cuda.bindings` 확인됨); **`torch.cuda.Stream.wait_stream` 후킹** — 이 저장소는 `graph_capture`(`parallel_state.py:483-506`)·`ExternalStream`(green ctx)·true-dual-worker의 `record_event`/`wait_event` 부기를 무겁게 쓴다(B2) |
| **L5** | **스레드 안전** | F13: `CURRENT_STREAM_IDX`가 모듈 전역. break 콜백은 replay를 수행하는 스레드에서 돌므로, 그 안에서 파티션 상태를 읽으면 true-dual-worker 하에서 scheduler 스레드의 변경과 경쟁한다. 현 설계가 안전한 건 **오직** 스위치 전 2× drain + arbiter의 `safe_to_switch()` 덕분이다 |
| **L6** | **상위 게이트 미해결** | F9(B4): green-ctx 캡처 그래프가 replay 시 SM 한정을 전달하는지 미측정. + F12: 업스트림이 torch ≥ 2.7에서 green-ctx × cudagraph 저하를 경고하고 현 환경은 **2.9** |

---

## 7. P0 프로브 설계 — 이거 통과 전엔 §4 어느 것도 시작하지 않는다

> 두 프로브 모두 **도구 타당성 판정이지 성능 판정이 아니다**(2026-08-20 job 886718 선례
> 문구 준수). 서버를 띄우지 않거나(P0-A) 띄우더라도 지연/처리량 수치를 판정에 쓰지 않는다.
> `.sbatch`는 작성하되 **제출하지 않는다** — 제출은 사용자 승인 후 experiment-runner 소관.

### P0-A — graph replay가 SM 한정을 전달하는가 (구멍 C / 가정 B4 = SMID 문서의 E5)

**질문**: green-ctx 스트림 위에서 **캡처된** CUDA graph를 replay할 때, 그 커널들이 그
green context의 SM 부분집합에서만 도는가?

**왜 이게 BCG보다 먼저인가**: 만약 전달되지 **않는다면**, decode 측 세그먼트별 SM 이야기
(S2)는 물론이고 **운영점의 decode SM 분할 서사 전체가 재검토 대상**이 된다. 이 게이트는
BCG와 무관하게 이미 열려 있었고(2026-08-11 등재), BCG 논의는 그걸 다시 최전선으로 끌어올릴 뿐이다.

**설계**: `results/smid_census/smid_l0_census.py`(F15)의 L0 census 기계를 **재사용**하되
**수정하지 않는다**(사전등록된 아티팩트 — 2026-08-20 `p1_greenctx_target.py` 선례대로
파생 스크립트를 새로 만든다). 추가되는 건 대조 하나뿐:

| leg | 실행 | 역할 |
|---|---|---|
| `eager_green` | `_census_once(kernel, green_stream, …)` 그대로 | 기준선(= 기존 R0) |
| `graph_green` | 같은 커널을 `torch.cuda.CUDAGraph`로 **green_stream 위에서 캡처** → replay → census 버퍼 판독 | **조건 under test** |
| `eager_plain` | green ctx 밖 평범 스트림 | 양성대조 (census가 살아있음을 증명) |
| `graph_plain` | 평범 스트림 위 캡처 → replay | 두 겹 대조 (그래프 경유 자체가 census를 안 깨뜨림을 증명) |

**판정 규칙 (사전 고정)**: `U(leg)` = 관측된 distinct `%smid` 합집합 크기.
- `U(graph_green) ≈ U(eager_green)` **그리고** `U(graph_green) ≪ U(graph_plain)` ⇒ **전달됨**.
- `U(graph_green) ≈ U(graph_plain) ≈ 108` ⇒ **전달 안 됨** — 정본 재검토 트리거.
- `U(graph_plain)`이 포화 안 됨 / census 계측 자체가 깨짐 ⇒ **`UNDETERMINED
  (MEASUREMENT ABSENT)`** — 기존 스크립트의 verdict 어휘를 그대로 쓴다. **측정 실패를
  게이트 실패로 라벨링하지 않는다**(방법론 교훈 #21, 8회+ 재발).

**선행 조건**: `smid_l0_census.py`의 **R0(green ctx에서 `%smid`가 일관된 물리 identity를
주는가)가 아직 GPU에서 실행된 적이 없다**(F15). R0가 `GLOBALLY_CONSISTENT_LABEL`이 아니면 P0-A는
해석 불가다. ⇒ **`smid_l0_run.sbatch`를 먼저 돌린다(≈10분).** P0-A는 그 다음이다.

**비용**: R0 ≈10분 + P0-A ≈10분 = **≲0.35 GPU-hr**. 서버 부팅 없음, 모델 로드 없음,
요청 발행 없음 ⇒ 출력에 인용 가능한 성능 수치가 **구조적으로** 존재하지 않는다.

### P0-B — BCG 백포트 스모크 (가정 B1·B2·B3)

**설계**: 업스트림 모듈을 dev tree에 백포트한 뒤 pdmux + Zamba2-2.7B를 부팅한다. 서버
플래그는 운영점 그대로(`--enable-pdmux --pdmux-config-path … --chunked-prefill-size -1
--disable-overlap-schedule --attention-backend triton --disable-radix-cache
--mem-fraction-static 0.82 --max-running-requests 48`), 여기에 BCG만 얹는다.

**★ importability는 wiring이 아니다 (B1).** F1이 이 트리에 `debug-cuda-graph` 문자열이
0 hit임을 확인했으므로 서버 플래그 배선(`server_args.py`)과 캡처 라우팅
(`cuda_graph_runner.py`)은 F2의 "4파일"에 **없다**. 모듈만 떨어지면 전 arm이 멀쩡히
부팅하고 전 delta가 ≈0으로 나오는데, 그건 "브레이크가 공짜"가 아니라 "브레이크가 한 번도
실행 안 됨"이다. ⇒ stage 1에 **양성대조**(shim이 `wrapped>0`을 보고)를 넣고 실패 시
`UNAVAILABLE (BCG PATH NOT WIRED)`로 조기 종료한다.

**★ 브레이크는 decode에만 걸린다.** Zamba2는 같은 layer 객체를 **두 곳**에서 부른다 —
그래프화된 decode forward(`zamba2.py:600-608`)와 **완전 eager**인
`forward_split_prefill`(`:766-772`, PD-mux 층축 prefill split). 게이트 없이 감싸면
`delta_k`가 "그래프 break + prefill 층당 eager Python 호출"의 합이 된다(변수 2개 동시 변경).
shim은 `forward_mode.is_decode()`로 막는다.

**★ arm 순서가 브레이크 수와 앨리어스되면 안 된다.** 결정 규칙이 `delta_3 > delta_6`인데
arm이 항상 null→k6→k3 순이면 **단조 drift만으로 규칙이 공짜로 성립**한다. 이 저장소는
같은 실패를 이미 등재했다(C2 감사: `results/r0c/decode_knee_vs_ctx.sbatch:45`가 `full`을 항상 첫 arm에
둬 warm-up 편향이 모든 비의 분모에 걸림 — `CONSENSUS.md:1355`). ⇒ **3 블록 인터리브 +
중간 블록 순서 역전**, delta는 **블록 내에서** 취한다. null arm 3회 부팅이 곧 σ_boot 추정치다.

| 측정 | arm | 판정 |
|---|---|---|
| (i) 가용성 | `A_base_pre` · `B_debugcg`(=S4 오라클 arm) · `P_wiring`(양성대조) | 부팅 실패 또는 `wrapped=0` ⇒ **UNAVAILABLE**, 이하 전부 미채점 |
| (ii) 충돌 | 위 + `F_dualwk_null` · **`G_dualwk_k6`** | BCG arm에만 나오는 illegal-memory/capture-fail/hang/traceback ⇒ L4·L5 실현. **`G`가 L5가 경고한 유일한 셀**(dual-worker × break>0)이라 반드시 있어야 한다 |
| (iii) per-break 비용 | `b{1,2,3}_{null,k6,k3}` (3 블록, 순서 counterbalance) | 아래 게이트 |
| (iv) drift 앵커 | `A_base_post`(마지막) | `A_post` vs `A_pre` 차이가 `sd_b(delta_6)`를 넘으면 블록 내 delta만 인용 가능 |

**게이트 (사전 고정, 두 지표 모두에서 성립해야 함)**:
- **estimand**: goodput 정의는 request-내부 token-ITL **p95**(게이트 #4)인데
  `bench_serving`은 그걸 안 준다. ⇒ 결정량을 **`Median ITL`과 `P99 ITL` 둘 다**로 선언하고
  둘 다에서 게이트가 성립해야 한다. **median으로 재고 p95를 논하는 estimand 바꿔치기 금지**
  (브레이크는 CPU 왕복이라 median보다 tail을 움직일 개연성이 높다).
- **S1 사망 조건**: `mean_b delta_6 − 1.96·sd_b(delta_6)/√3 > 0.03·mean ITL(null)`
  — 즉 **신뢰하한**이 3%를 넘을 때. 점추정만 3%를 넘는 것으로는 S1을 죽이지 않는다(n=1로
  3% 문턱을 결정 규칙으로 쓰는 건 게이트 #3의 오용 — #3은 "이하는 headline 아님"이라는
  *하한*이지 n=1 판정 문턱이 아니다). `sd_b(delta_6)` 자체가 3%를 넘으면
  **`UNDETERMINED (BOOT VARIANCE EXCEEDS THE GATE)`** 로 재파워링한다.
- **변이 검사 (문턱도 사전 고정)**: `mean_b delta_3 / mean_b delta_6 ≥ 1.5`.
  k=3은 k=6의 약 2배 브레이크이므로 진짜 per-break 비용이면 스케일해야 한다.
  **1.5 미만이면 delta는 per-break 비용이 아니며 stage 2는 아무것도 보고하지 않는다**
  — 어느 방향으로도 게이트 판정을 내지 않는다(교훈 #53).

**비용**: 15 부팅(`A_pre`·`B`·`P`·3블록×3·`F`·`G`·`A_post`) × ≈33 s 부팅 + 짧은 부하.
`--time=02:30:00` 예약.

### 작성된 파일 (전부 `results/bcg_probe/`, **제출 안 됨**)

| 파일 | 역할 | 검증 상태 |
|---|---|---|
| `p0a_graph_sm_confinement.py` | P0-A 본체. `smid_l0_census.py`를 **import**(수정 없음)해 4-leg 대조 + 순수함수 scorer | `--selftest-analyzer` 13/13 PASS **+ `--selftest-mutants` 5/5 PASS**. ★변이는 주석 속 사고실험이 아니라 **실행되는 하네스**다: 소스에서 P2/P3/P4/NOCAP 가드를 삭제하고 TOL을 74로 넓힌 변이본 5개를 컴파일·실행해 **각각이 실제로 판정을 뒤집는지** assert한다(교훈 #53) |
| `p0a_graph_sm_confinement.sbatch` | P0-A 실행(3단계: 분석기+변이 자기검사 → GPU legs → 채점) | `bash -n` OK · `--comment` directive가 첫 실행 라인 위 |
| `p0b_break_shim.py` | break 주입 shim. F16 때문에 `ModelRunner.load_model`을 감싼다. **dev tree 미변경** · `PDMUX_BCG_BREAK_EVERY=0`이면 아무것도 감싸지 않음(null-break arm) · **`forward_mode.is_decode()` 게이트**로 eager prefill 오염 차단 · break primitive 없이 k>0이면 **조용히 0-break로 퇴화하지 않고 SystemExit**(교훈 #9) | `py_compile` OK |
| `p0b_bcg_backport_smoke.sbatch` | P0-B 실행. stage 0 모듈 존재 확인 → stage 1 가용성 + **wiring 양성대조** → stage 2 **3블록 인터리브·순서 counterbalance** per-break 비용 → stage 3 dual-worker **양 셀** → drift 앵커. stage 4에 사전 고정 판독 규칙 인쇄 | `bash -n` OK · `--comment` OK · 15 부팅 / `--time=02:30` |

*계획 대비 위치 변경*: `scripts/bcg_probe/` 대신 **`results/bcg_probe/`** 에 뒀다 —
선례(`results/kernel_mech/p1_probe/`, `results/smid_census/`)가 프로브 스크립트를
아티팩트와 같은 디렉터리에 두고, 산출물도 같은 곳에 떨어진다.

### 게이팅

```
smid R0 (기존, 미실행)  ──▶ P0-A ──┐
                                   ├──▶ S1 설계  ──▶ (서빙 실증 필요) S3 설계
P0-B ──────────────────────────────┘
S2 는 P0-A/P0-B 를 통과해도 미실행 권고 유지 (§4-S2)
S4 는 P0-B (i) 만 통과하면 사용 가능 (단 L1 선지불)
```

---

## 8. 이것으로도 닫히지 않는 것 (필수 절)

- **layer-aware 死 판정은 안 바뀐다.** BCG는 §3의 다리 ②만 자르고, F8이 다리 ① 축의
  상한을 이미 측정했다. P0가 다 통과해도 S2는 미실행 권고다.
- **HE0는 안 열린다.** §5-1.
- **wave quantization / occupancy는 여전히 못 잰다.** S1은 타임라인 해상도지 커널 내부
  카운터가 아니다. `CONSENSUS.md` §3 항목52의 "진짜 원인은 커널 단위 측정 0건이라 미식별"은
  **계속 유효**하다.
- **gate #13 · gate #16은 이 문서와 무관하고 여전히 안 닫힌다.** 관련 금지 배너 유효.
- **P0-A가 "전달됨"으로 나와도** 그건 SM identity 수준의 확인이지, green ctx 하 커널
  효율이나 고-SM 평탄화 기전에 대한 진술이 아니다.

---

## 9. 하지 말 것 (이 트랙에 고정)

1. **"BCG가 layer-aware를 되살린다" / "per-window 트랙을 GO로 바꾼다"** — F8이 상한을
   이미 측정했다. BCG는 실험 비용을 낮출 뿐이다.
2. **"BCG가 HE0를 뒤집을 수 있다"** — 死因 불일치(§5-1).
3. **S3의 최적점 이동을 주장으로 승격** — 예측이다. 게이트 #1(서빙 실증) 전엔 순위 진술 금지.
4. **업스트림 벤치(1.70× 등)를 이 기판의 기대값으로 인용** — gpt-oss-120b / TP4 / 4×GB300 /
   prefill-only다. 모델·병렬도·하드웨어·워크로드가 전부 다르다(방법론 교훈 #31).
5. **P0 결과를 성능 판정으로 쓰기** — 도구 타당성 판정이다.
6. **`smid_l0_census.py`를 P0-A 용도로 개조** — 사전등록 아티팩트다. **기능 추가·재목적화는
   파생 스크립트로** 한다(P0-A는 import만 한다). ★**2026-08-21 정정**: 이 금지는 *개조*에
   대한 것이지 *결함 수리*에 대한 것이 아니다 — 2026-08-14 감사가 R0를 `CONDITIONAL-GO`로
   묶은 **fail-open 결함**(`:410` 죽은 `JITFunction.cache` 읽기 + `:523`/`:529` 가드가 `None`을
   통과)은 **제자리에서 수리**했다(2026-08-21, engine-porter, 변이 테스트 7/7). 판정 어휘·
   4-outcome 매트릭스·정지 규칙은 불변이다. 상세 `../results/smid_census/
   PREREG_SMID_R0_2026-08-14.md` §15.
7. **`.sbatch` 자동 제출** — 승인 후 experiment-runner 소관.

---

## 10. 비용 산정

| 항목 | 인력 | GPU | 리스크 |
|---|---|---|---|
| 이 문서 (분석) | 완료 | **0** | 0 |
| smid R0 (기존 사전등록, 미실행) | 0(작성됨) | ≈0.17 GPU-hr | 낮음 |
| P0-A (파생 census + 그래프 leg) | 반나절 | ≈0.17 GPU-hr | 낮음 |
| P0-B (백포트 **3층** + 스모크 15 부팅) | 1–2일 | **2.5 GPU-hr 예약** | **중**(B1·B2) |
| S1 계측 설계·실행 | P0 통과 후 별도 산정 | — | 중 |
| S3 prefill 캡처 설계 | B5 해소 필요 | — | **높음** |
| S2 | — | — | **미실행 권고** |

★**예약 기준으로 다시 쓴다**: `.sbatch`의 `--time` 예약 합은
`smid_l0_run 00:15` + `p0a 00:20` + `p0b 02:30` = **3.08 GPU-hr**이고, 위 "≈" 값들은
과거 캠페인 부팅 시간에서의 **외삽**이다(SMID 설계문서 E9와 같은 성격 — 스모크로 검증할 것).
**예약 기준 ≈3.1 GPU-hr**로 보고한다 — P0-B가 σ_boot을 재려고 15 부팅으로 커진 결과다.
어느 쪽이든 등록 단위(gate #13 rev3 = 11.52 GPU-hr)에 비하면 작다.

---

## 11. 다음 액션 (제안 — 승인 필요)

1. `smid_l0_run.sbatch` 제출 (기존 사전등록, 미실행). R0 verdict 확보.
2. R0가 `GLOBALLY_CONSISTENT_LABEL`이면 P0-A 제출.
3. 병렬로 P0-B 백포트(engine-porter 규율: dev tree 변경 + manifest 갱신 + CPU 회귀
   `python -m unittest discover -s workspace/engine-port/tests`).
4. P0 두 건의 verdict가 나온 뒤에만 S1 설계 착수. S3는 B5 설계가 먼저.
5. 이 문서의 S1–S4·손실·판정 규칙을 **claims-auditor**에 넘겨 적대 감사
   (특히: S2가 슬며시 GO로 미끄러지지 않았는지, S3 예측이 주장으로 승격되지 않았는지,
   P0-B (iii)의 검사가 변이 테스트를 통과하는지).

---

### 부록 A — 이 문서를 만들며 실행한 명령 (전부 CPU / 읽기, GPU 0)

```
grep -rn "breakable|eager_on_graph|BREAKABLE|debug-cuda-graph" --include=*.py sglang_engine_dev/  # 0 hit
sed -n '545,549p'   .../model_executor/cuda_graph_runner.py           # F3a
sed -n '160,180p'   .../model_executor/forward_batch_info.py          # F3b
sed -n '2862,2872p' .../model_executor/model_runner.py                # F3c
sed -n '755,830p'   .../model_executor/cuda_graph_runner.py           # F4
sed -n '1036,1105p' .../multiplex/multiplexing_mixin.py               # F6
sed -n '6124,6146p' .../srt/server_args.py                            # F12
cat                 .../srt/multiplex/pdmux_context.py                # F13
sed -n '770,830p'   .../srt/models/zamba2.py                          # F14
sed -n '329,420p'   .../results/smid_census/smid_l0_census.py         # F15
sed -n '1358p;1362p;1391p' reports/CONSENSUS.md                       # F7, F8
sed -n '77,96p'     .../reports/SMID_DIRECT_INSTRUMENTATION_DESIGN_2026-08-11.md  # F9
python -c "import cuda.bindings"                                      # F2
```

업스트림 확인: SGLang 문서 `advanced_features/breakable_cuda_graph`,
LMSYS 블로그 *Advanced CUDA Graph Techniques in SGLang* (2026-08-17),
`sgl-project/sglang` main 트리의 `runner_backend_utils/breakable_cuda_graph/`.
