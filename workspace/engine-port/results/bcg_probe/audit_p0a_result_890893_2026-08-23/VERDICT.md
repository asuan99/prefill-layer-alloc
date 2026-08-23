# 감사 판정서 — P0-A **결과** 적대 감사 (job 890893, 2026-08-23, claims-auditor, read-only, GPU 0)

**대상**: `p0a_verdict_890893.json`(정본 인용 대상) · `p0a_raw_890893.json` ·
`PREREG_P0A_2026-08-22.md` rev8 · `audit_p0a_harness_2026-08-23/VERDICT.md` ·
참조 `../smid_census/smid_l0_verdict_889631.json`(R0).
**선례**: R0 결과 감사(2026-08-22, `CONFIRMED(scoped)` + 조건 4건).
**목적**: 결과가 정본에 등재되기 **전에** 무엇을 말할 수 있는지 확정한다.

---

## ★ 단일 판정

> ### `CONFIRMED with conditions` — 조건 **C1–C8**
>
> * **결정 레그 판정**(`CONFINEMENT_PRESERVED_THROUGH_GRAPH_REPLAY`) = **CONFIRMED(scoped)**.
>   사전등록이 실제로 집행됐고, 판정은 원자료에서 **바이트 동일하게 재현**되며, `Δ=∅`은
>   **공허참이 아니다**(아래 §2에 양적 근거).
> * **부수 도구 타당성 사실**(green 스트림 위 그래프 캡처·replay가 이 기판에서 **가능**) =
>   **CONFIRMED(scoped)** — 이 프로브 전까지 저장소 전례 0건이었고 `NOCAP`은 사전등록된
>   **살아 있는 결과**였다.
> * ★**교차 레그 2개의 *기전* 해석**(*"한정이 캡처 시점에 박힌다"*) = **NOT-YET-SUPPORTED
>   (비식별)**. 관측 두 수(34 / 108)는 **경쟁 가설 2개가 똑같이 예측**한다(§4). 관측값만
>   기록하고 기전 문장은 쓰지 않는다.
> * **`REFUTED`는 없다** — 판정 자체를 뒤집으려는 반증 시도 **10건**(§1–§2·§6: 사전등록
>   미집행 · 사후 편집 · 빈/축퇴 census · 커버리지 결손 · 좁아진 graph 레그 · warmup 오염 ·
>   부착 판독 고착 · 레그 오배선 · 금지 문장 누출 · 규칙 재구현)이 **전부 실패**했다.
>   ★**성공한 공격은 판정이 아니라 *문장*에 대한 것 4건**: §6 스코프 목록 불완전(4항목) ·
>   *"기판이 R0와 일치"* 과잉 · 인용 아티팩트 결함 2건(드라이버 버전 부재 · 극성 읽기 함정) ·
>   교차 레그 기전 비식별. 단 **반증 실패 ≠ 무조건 등재**: 아래 C1–C8을 지키는 문구로만
>   등재하고, §11 F1·F2 전에는 §2 잔여 전제와 §4 기전 문장을 단정하지 않는다.

**한 줄 요약**: 이 트랙이 5회에 걸쳐 죽인 "공허참" 결함은 이번엔 **살아나지 않았다**.
등재를 좁히는 것은 판정의 신뢰도가 아니라 **문장의 상한**이다 — 특히 (i) 교차 레그 기전과
(ii) §6이 열거하지 않은 스코프 4건.

---

## 1. 사전등록이 실제로 집행됐는가 — **집행됐다**(재현 완료)

| 검사 | 결과 |
|---|---|
| `--analyze` 재현(프로젝트 venv, GPU 불요) | ★**전 키 바이트 동일**(only-in-orig 0 · only-in-repro 0 · differing 0). 채점이 원자료의 **순수 함수**임을 실증 |
| `raw_sha256` | `e5392fba…0d550e` = 디스크 `p0a_raw_890893.json` **일치** |
| `run_harness_sha256` | `f4ba1d46…d64c6` = 작업트리 및 커밋 `366841c` blob **3중 일치**(런 이후 하네스 미수정) |
| `run_rule_module_sha256` | `40709b24…3e43b` = 작업트리 및 `366841c` blob **일치** |
| 사전등록 동결(§9-5) | `PREREG_P0A_2026-08-22.md` sha256 `9c10f16b…48bb2` = `366841c` blob 일치. 마지막 수정 커밋 **0aedecd(17:12:59)**, 잡 stage 0이 찍은 HEAD도 **0aedecd** ⇒ **결과를 본 뒤 §4·§5·§8을 바꾼 흔적 0건** |
| 규칙 재구현 여부 | 없음. 하네스는 `RULE.score(w)` **한 줄**만 호출(`grep -n "TOL"` = 0). §4 순서(P0→P1→P2→P3→Δ_null→P4a→P4b→CAP→REP→P0g→P5→P4c→결정)가 `p0a_rule_totality.score()`에 **그대로** 구현돼 있음을 줄 단위 대조 |
| §5 진입 조건 | *"25개 (grid, repeat) 쌍 전부 캡처 성공"* — `graph_plain` 25/25 · `graph_green` 25+25 · 실패 0. §12(H3)의 **75/75** 요구 충족 |
| §8-32(`--sample` 금지) | sbatch는 `--selftest`만 호출, `--sample` 미사용 ✅ |

**결론**: §4 선행조건·§5 매트릭스를 **그대로** 따랐다. 하네스가 규칙과 다른 것을 계산한
경로를 찾지 못했다.

★**단, 두 가지 provenance 흠**(판정 입력 아님, 등재 시 문구에만 영향):
* **(a)** 판정서의 `run_provenance.git_head = 366841c`는 **잡이 시작한 HEAD가 아니다**
  (stage 0이 찍은 것은 `0aedecd`). 그 사이 17:14:09에 **다른 트랙(kernel_mech) 문서 커밋**이
  들어왔고, `run()`이 stage 2 시점에 HEAD를 다시 읽었다. `git_dirty=false`이고 두 임계 파일
  해시가 커밋 blob과 일치하므로 **실질 위험 0**이지만, *"이 커밋에서 실행됐다"* 로 쓰면 부정확.
* **(b)** `runtime_source_manifest`: `entry_count=13` vs `expected_entry_count=15`,
  `entry_count_matches_sync_script=false`. **provenance 전용**(엔진 모델 코드를 이 프로브는
  로드하지 않으며, 실제로 쓰인 `pdmux_context.py`·`sgl_kernel/spatial.py` 해시는 R0와 **정확히
  동일**하다). 그러나 아티팩트가 **자기 불일치를 스스로 보고**하고 있으므로 등재 문장이
  *"매니페스트 검증 통과"* 라고 말해서는 안 된다.

---

## 2. ★★`PRESERVED`가 공허하게 참인가 — **아니다** (이 감사에서 가장 중요한 음성)

이 트랙이 5회에 걸쳐 죽인 결함(빈 census가 모든 집합 술어를 공허 만족)이 재발하지 않았음을
**양적으로** 확인했다.

| 근거 | 실측 |
|---|---|
| graph 레그가 **실제로 관측했다** | `graph_green` 2 sweep 각각 **union 34** · 라벨별 최소 히트 **476** · sweep당 총 히트 **16,740** |
| ★**총 히트 = 발사된 블록 수와 정확히 일치** | `Σ(GRID_SWEEP)×REPEATS = 3,348×5 = 16,740`. ⇒ **모든 블록이 유효 라벨을 기록**했고 `-1`(미기록) 칸이 **0개**다. 커버리지 결손으로 `E`가 비었다는 설명이 성립하지 않는다 |
| ★**탈출 탐지력** | graph 레그의 **33,480 block 관측**(2 sweep) 중 34-라벨 집합 밖에 떨어진 것이 **0건**. 양성 하한 문턱은 1이므로 **블록 1개만 새도 잡힌다** |
| 대칭차 양방향 | `E = ∅` **그리고** `E_rev = ∅` ⇒ `exact_set_match=true`(C1 수리가 막으려던 "좁아진 graph 레그" 세계 아님) |
| 잡음 눈금 | `Δ_split` = 세 green 레그 **전부 `∅`**(같은 레그 2회 census가 라벨 집합에서 완전 일치) |
| 포화 | 5 레그 전부 **첫 grid 점(108 blocks)에서 이미 평평**(사다리 `[34,34,34,34,34]` / `[74×5]` / `[108×5]`) |
| `P4c` (F1 수리) | `S(eg) ∩ S(egp) = ∅` ∧ `S(eg) ∪ S(egp) = D`(=108) — R0 disjointness를 **이 프로세스에서 독립 복제** |
| 귀무 채널 | `S(graph_plain) △ S(eager_plain) = ∅` **양방향** |
| 부착 양측 대조 | green 2개 부착 ∧ 평범 대조 **2개 모두** 미부착 보고(음성 대조 발화) |
| 계측 생존(P1) | `runtime_ptx_smid_sites=1`, `runtime_spin_back_edge=true`, **변종 1개만 컴파일됨** ⇒ 모든 레그가 같은 계측 커널을 돌렸다 |

★**주의(문구 정밀도)**: `min_hits` **155–476은 문턱이 아니다**. 사전등록 문턱은
`MIN_HITS_REPORTED = 1`이고 155–476은 **관측값**이다(§8-4·§5.1). 등재 문장에서
*"양성 하한 155를 통과"* 라고 쓰면 문턱을 날조하는 것이다.

★**남는 전제 1건(등재 필수, §5-C4)**: 이 프로브에는 *"census 텐서를 쓴 것이 **replay**이지
**캡처 시점 launch**가 아니다"* 를 보이는 **양성 대조가 없다**. 두 하네스층 감사(B5)와
이 감사 모두 그것을 **CUDA stream-capture API 보증**(캡처 중 launch는 실행되지 않는다)에
의존해 논증한다 — 아티팩트 안에는 그 구분을 짓는 필드가 없다. 그 보증이 깨지면 판정은
*틀린* 것이 아니라 **공허**해진다(= eager green을 두 번 잰 것). **0.02 GPU-hr짜리 대조**로
닫을 수 있다: *캡처만 하고 replay를 하지 않는 레그의 census는 반드시 빈 집합이어야 한다.*

---

## 3. §6 상한이 옳은가 — **핵심 문장은 옳다(오히려 보수적). 단 "이 문장에 없는 것" 목록이 불완전하다**

**§6 핵심 문장 평가**
* *"…라벨 집합 밖으로 나가지 않았다"* = `E=∅`만 주장 ⇒ 실측(`E=∅` ∧ `E_rev=∅`, 정확 일치)
  보다 **좁다**. 과소 주장이므로 안전하고, 등재 문장은 **정확 일치**까지 말해도 §6을 넘지 않는다
  (`exact_set_match`는 §2.3의 등록된 서술량이다).
* *"물리 SM"이 아니라 전역 일관 라벨* · *카디널리티→계산량 환산 금지* — 판정서 `scope`
  문자열에 그대로 실려 있고, 아티팩트 전체에서 금지 형용사의 출현은 **부정문 안에서만**이다
  (A8a/A8b 실측 + 이 감사 재확인).

**★§6의 "이 문장에 없는 것" 목록에서 빠진 스코프 4건**(등재 시 **반드시 추가**):
1. ★**동시부하 0** — 모든 레그가 **단독**으로 돌았다. 상보 절반(prefill 74)은 graph 레그가
   도는 동안 **유휴**였다. R0에는 동시성 블록(`concurrent_idx1`/`R5`)이 있었으나 **P0-A에는
   없다**. 운영점에서 문제되는 조건(양쪽 절반 동시 가동)은 **측정되지 않았다**.
2. ★**캡처당 replay 1회** — 125개 그래프를 각각 **한 번씩** replay했다. 엔진은 하나의 그래프를
   **수천 번** replay하고 그 사이에 다른 스트림 그룹의 그래프가 낀다. *"반복 replay 후에도
   한정이 유지되는가"* · *"stream-group 전환 후에도 유지되는가"* 는 **미측정**.
3. ★**그래프 노드 1개** — 단일 Triton 커널 노드. memory op·NCCL·fork/join·CPU 노드 없음.
4. ★**prefill 절반은 graph 레그가 없다** — 74-SM 절반은 **eager로만** 관측됐다. 판정은
   **decode 절반(34)에 한정**된다.
   (부수: compute mode `Default`·단일 프로세스·MPS 아님도 명시 권장.)

**★아티팩트 결함 2건**(R0의 N1/N2 계열, 등재 문구에 직접 영향):
* **P1(신규)** — ★**드라이버 버전이 인용 가능 아티팩트에 없다.** `580.105.08`은 `.out`
  콘솔 전사에만 있고 `.json`에는 없다(`grep driver` → `driver_readout`뿐). §6이 licence하는
  문장은 *"이 드라이버 위에서"* 라고 말하는데, **그 드라이버를 `.json`에서 인용할 수 없다**
  (게이트 #56과 정면 충돌). 경과시간·GPU-hr도 동일(SLURM 기록/커밋 메시지에만 존재).
* **P2(신규)** — ★**극성 읽기 함정이 판정서 층에 살아 있다.** `adapter.attachment_positive
  .observed = {"green_prefill": false, "green_decode": false}` 는 **생산자 원값
  `green_ctx_is_null`**(false = **부착됨**)이며, 통과 세계에서는 `why`가 **빈 문자열**이라
  파일 안에 설명이 없다. 긍정형 `run_green_ctx_attached`(true/true)가 **함께** 있으므로
  아티팩트 결함은 아니지만, **인용자가 정반대로 읽을 수 있는 자리**다(R0 N2 = 게이트 #61의
  한 층 위 재발). ⇒ **부착을 인용할 땐 `run_green_ctx_attached`만 인용한다.**

---

## 4. ★★★ 교차 레그 2개 — **비식별**. 기전 문장 금지

**관측(원자료)**: `capture_green_replay_plain` → union **34**, `capture_plain_replay_green`
→ union **108**. ★이 감사가 추가로 계산한 것: 두 집합은 카디널리티가 아니라 **원소까지
정확히 일치**한다 — `S(cross_green_capture) == S(eager_green)`(34개 전부),
`S(cross_plain_capture) == D`(108개 전부). 두 레그 모두 사다리 첫 점부터 포화,
`min_hits` 476 / 155, 캡처·replay 25/25, `descriptive_legs_incomplete=false`.

### (a) 이 관측이 *"한정은 캡처 시점에 결정된다"* 를 지지하는가 — **지지하지 못한다(비식별)**

경쟁 가설 **두 개가 정확히 같은 두 수를 예측**한다:

| | 가설 | 예측 |
|---|---|---|
| **H1** | 그래프가 **캡처(=instantiate) 측**의 green-context SM 한정을 실어 나른다 | cross-green-capture=34, cross-plain-capture=108 |
| **H2** | `torch.cuda.CUDAGraph.replay()`가 ambient 스트림이 아니라 **캡처 스트림**(또는 어쨌든 replay 스트림이 아닌 곳)에 launch한다 ⇒ 레그 이름의 *"replay 스트림"* 이 **오칭** | 동일: 34 / 108 |

★**H2를 배제하는 증거가 아티팩트·저장소 어디에도 없다.** 설치된 torch(2.9.1+cu130)는
`ATen/cuda/CUDAGraph.h`만 헤더로 배포하고 `replay()`의 **launch 스트림을 명시하지 않으며**,
`torch/cuda/graphs.py`의 docstring도 *"Replay the CUDA work captured by this graph."* 뿐이다
(이 감사가 직접 확인). 하네스 docstring의 *"replay on the replay stream"* 은 **가정이지
측정이 아니다**.

★**배제된 대안 1건**(적어 둘 가치가 있다): *"`with torch.cuda.stream(green)` 자체가 이
하네스에서 무력하다"* 는 **거짓**이다 — 같은 래퍼를 쓴 eager 레그가 34 라벨로 정확히
갇힌다. 즉 관측은 *"eager launch를 가두는 바로 그 스트림 컨텍스트가 graph launch는 가두지
못했다"* 까지는 확립한다. 그 이상(어디서 결정되는가)은 H1/H2가 갈리지 않는다.

★**또 하나의 대안 폐기 시도**: *"캡처가 실제로 실행돼 캡처 스트림 라벨이 찍힌 것"* 은
`capture_green_replay_plain`(34)을 설명하지만 **`capture_plain_replay_green`(108)과 함께
보면 replay가 한 번도 쓰지 않았다는 뜻**이 되어 §2의 전제와 같은 문제로 환원된다 —
즉 **§2 잔여 전제와 동일한 대조 하나(캡처만/replay 없음 레그)가 이 축도 함께 좁힌다.**

### (b) 레그 이름 ↔ 스트림 대응 — **오늘은 맞다. 단 기계 검증은 여전히 없다**

`run():484-485`의 `cross_spec` dict를 직접 읽어 확인:
`CROSS_LEGS[0]="capture_green_replay_plain" → (g_decode, plain)`,
`CROSS_LEGS[1]="capture_plain_replay_green" → (plain, g_decode)`,
`_leg_graph(kernel, capture_stream, replay_stream, …)` 위치 인자 순서와 일치. `g_decode`가
정말 decode 절반이라는 것은 **데이터가 확인**한다(같은 스트림의 eager 레그 = 34 = `d_sm`).
★그러나 하네스층 감사 **H5의 잔여**는 남아 있다 — `A7e`는 *"dict 키가 `CROSS_LEGS`의
subscript인가"* 만 검사하고 **(capture, replay) 순서는 검사하지 않는다**. 즉 두 서술 레그가
뒤바뀌어도 스위트는 통과한다. 오늘 옳음은 **사람이 읽어서** 확인된 것이다.

### (c) 서술량을 정본에 올릴 수 있는 강도 — **"관측값만, 기전 문장 없이"**

* 올릴 수 있는 것: **두 수(34 / 108)와 집합 동일 사실**, 그리고 *"판정 규칙이 붙어 있지
  않은 서술 레그이며 n=1 sweep(split-half 없음), 규칙이 이 레그에 어떤 게이트도 적용하지
  않았다"* 는 한정.
* 올릴 수 **없는** 것: *"한정은 캡처 시점에 박힌다"* · *"replay 스트림은 무관하다"* ·
  *"엔진이 캡처 스트림만 맞추면 된다"* — 전부 H1/H2 비식별 구간이다.
* ★**이것을 값싸게 닫는 법(GPU 0)**: pytorch v2.9.1 태그의 `aten/src/ATen/cuda/CUDAGraph.cpp`
  `CUDAGraph::replay()`가 `cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream())`
  인지 **한 줄 확인**하면 H2가 즉시 죽는다. 그 확인 **전에는** 기전 문장을 쓰지 않는다.

---

## 5. `NOCAP`은 살아 있는 결과였는가 — **그렇다. 등재 가능(스코프 한정)**

* 사전등록 rev7 헤더·rev8 §12·하네스층 감사 판정서가 **모두** *"green 스트림 위 그래프 캡처
  자체가 가능한지 미지수"* 라고 **실행 전에** 적었다(blind 선언 §0.4와 함께 검증됨).
  ⇒ 사후 합리화가 아니다.
* 실측: green(decode 절반) 스트림 캡처 **50/50**(결정 레그) + **25/25**(서술 레그) = **75/75**,
  같은 스트림 replay **50/50**, warmup 실패 **0**, 예외 필드 전부 `null`.
* ★**한정**: 단일 Triton 커널 노드 · `pool=None`(엔진은 `_capture_graph`에서 **공유 pool**) ·
  `capture_error_mode="global"` · `CUDAGraph()` 생성/`del` 125회 · torch 2.9.1+cu130 ·
  triton 3.5.1 · cc(8,0) · compute mode `Default`. ⇒ *"cudagraph capture works under
  green context"* 를 **엔진 캡처 경로(공유 pool·다중 노드·KV 텐서)** 로 옮겨 쓰면 위반이다.

---

## 6. 금지 문장 점검 — **누출 0건**

| 검사 | 결과 |
|---|---|
| `.json`에 *"물리 SM"* | `physical` 1회 — **`scope` 문자열의 부정문 안**("not an established physical SM index") ✅ |
| 성능 어휘(latency/throughput/goodput/TTFT/ITL/speedup) | `.json`에 **0건**(`occupancy`는 `scope`의 부정문 안 1회) ✅ |
| *"구멍 C가 닫혔다"* · *"R4에 답했다"* | 아티팩트·`.out`·커밋 메시지 전부 **부정형으로만** 등장 ✅ |
| *"cudagraph-ON 운영점에서 한정이 유지된다"* | 0건. `scope`가 명시적으로 금지 ✅ |
| 커밋 메시지(`d121926`) | 교차 레그를 *"NOT interpreted here … go to claims-auditor"* 로 유보 ✅ · *"Nothing is registered in canon by this commit"* ✅ |
| ★**커밋 메시지의 과잉 1건** | *"Substrate matches R0 (job 889631): no substrate_mismatch banner."* — **배너는 두 축만 본다**(compute mode + `smid_l0_census.py` sha256). 노드는 다르고(gpu40 → gpu43), R0 원자료에는 `torch_version`·드라이버 필드가 **아예 없어** 그 축은 비교 자체가 불가능하다. ⇒ 정본에는 *"기판 일치"* 가 아니라 *"검사한 두 축에서 R0와 불일치 없음"* 으로 쓴다 |
| ★**커밋 메시지의 과잉 1건(경미)** | *"Green-stream graph capture WORKS on this substrate"* — 위 §5 한정(단일 커널·`pool=None`) 없이 단독 인용 금지 |

---

## 7. R0 의존성(§9-2) — **성립한다. 단 의존은 생각보다 얕다**

* `substrate_mismatch` 배너 부재 = **두 축**(compute mode `Default`, census 파일 sha256
  `1cee2191…a325`)만의 일치이며 **"같은 기판"의 증명이 아니다**(노드·드라이버·torch 축 미검).
  ★게다가 참조값 `Default`는 **R0의 `.out`(비인용 아티팩트)에서 전사**된 값이다 — 배너
  입력으로는 무해하나, 정본이 *"R0와 같은 compute mode"* 를 인용할 땐 출처를 밝혀야 한다.
* ★**반대로, 이 프로브는 R0의 핵심을 이 프로세스에서 다시 잰다**: `P4b`(|S(eg)|=34=d_sm) ·
  `P4c`(서로소 ∧ `D` tile) · 사다리 포화가 전부 **자체 관측**이다. R0에서 **수입한 것은
  라벨 동일성 해석("전역 일관 라벨")과 배너 참조값뿐**이다. ⇒ §9-2는 형식적으로 유효하나,
  R0가 뒤집혀도 무너지는 것은 **해석 어휘**이지 이 런의 집합 관측이 아니다.
* ★**등재 금지(미등록 사후 분석)**: 이 감사가 계산해 보니 R0(gpu40) `idx1_decode` 라벨 집합과
  P0-A(gpu43) `eager_green` 라벨 집합이 **원소까지 동일**하고 prefill 쪽도 동일하다. 흥미롭지만
  **어느 사전등록에도 없는 사후 계산**이며 n=2 노드다. 정본 등재 금지 — 쓰려면 별도 등록.

---

## 8. ★ 정본에 등재할 문장 (그대로 복사해 쓸 것 — §6 상한 준수)

> **P0-A(cudagraph replay × green-context SM 한정) — job 890893, 2026-08-23, gpu43,
> A100-SXM4-80GB cc (8,0), compute mode `Default`, exit `0:0`.
> 판정 = `CONFINEMENT_PRESERVED_THROUGH_GRAPH_REPLAY`(claims-auditor
> `CONFIRMED with conditions`, 조건 C1–C8).**
>
> **① 결과.** 엔진·모델·요청·서버가 없는 별도 프로세스에서, `divide_sm(108,(8,0),2)[0] =
> (74,34)`(운영점 `slo_sched/pdmux_d34.yml`의 `manual_divisions`와 같은 수) 분할의
> **decode 절반(34) 스트림**에 캡처한 **단일 토이 커널 그래프**를 **같은 스트림에서
> replay**했을 때, 관측된 `%smid` 라벨 집합은 같은 스트림 eager census의 34-라벨 집합과
> **정확히 일치했다**(`Δ = E ∪ E_rev = ∅`, `exact_set_match=true`, `S(graph_green) ⊉ D`).
>
> **② 이 `∅`은 관측 부재가 아니다.** graph 레그는 2 sweep × 16,740 = **33,480 block
> 관측**을 냈고(총 히트 = 발사 블록 수와 정확히 일치 ⇒ 미기록 칸 0), 그중 34-라벨 집합
> 밖에 떨어진 것은 **0건**이다(라벨별 최소 히트 476, 양성 하한 **문턱은 1**). 같은 레그를
> 두 번 census한 `Δ_split`은 세 green 레그 전부 `∅`.
>
> **③ 전 사전등록 게이트 통과.** 5 레그 전부 포화(첫 grid 점부터 평평) · green pair
> **서로소 ∧ `D`(=108) tile**(`P4c` — R0 disjointness의 이 프로세스 복제) · 귀무 채널
> `S(graph_plain) △ S(eager_plain) = ∅`(양방향) · green 2 스트림 부착 ∧ 평범 대조 2개
> 미부착(양측 대조 발화) · 계측(P1) 생존, 컴파일 변종 1개.
>
> **④ 부수 도구 타당성 사실(신규).** **green-context 스트림 위 CUDA 그래프 캡처·replay가
> 이 기판에서 가능하다** — 저장소 전례 0건이었고 `NOCAP`은 사전등록된 살아 있는 결과였다.
> 실측 캡처 **75/75**(결정 50 + 서술 25), replay 실패·warmup 실패 0건. ★단
> **단일 커널 노드 · `pool=None`(엔진은 공유 memory pool) · `capture_error_mode="global"` ·
> torch 2.9.1+cu130 · triton 3.5.1** 한정이며, 엔진의 캡처 경로에 대한 진술이 아니다.
>
> **⑤ 서술 레그 2개(판정 규칙 없음, n=1 sweep, 게이트 미적용).** `capture_green_replay_plain`
> union **34**(eager green 집합과 원소까지 동일), `capture_plain_replay_green` union
> **108**(`D`와 원소까지 동일). ★**기전 해석 금지** — 이 패턴은 (i) *"한정이 캡처 시점에
> 박힌다"* 와 (ii) *"`CUDAGraph.replay()`가 ambient 스트림이 아닌 곳에 launch한다"* 를
> **구분하지 못한다**(두 가설이 같은 두 수를 예측). 관측만 기록한다.
>
> **⑥ 스코프(전수).** 모델·요청·서버·엔진 **없음** · 공유 graph memory pool **아님** ·
> 분할 **1개**(74,34)의 **decode 절반만**(prefill 74 절반은 eager로만 관측) ·
> ★**동시부하 0**(상보 절반 유휴, R0의 동시성 블록에 해당하는 측정 없음) ·
> ★**캡처당 replay 1회**(반복 replay·stream-group 전환 후 유지 여부 미측정) ·
> 그래프 노드 1개(memory op·NCCL 없음) · compute mode `Default`·단일 프로세스 ·
> cc (8,0)·이 클러스터의 이 드라이버·이 torch 빌드 한정. **`%smid`는 전역 일관 라벨이지
> 물리 SM 인덱스가 아니며, 집합 카디널리티를 계산량으로 환산하지 않는다.**
> ★**`PRESERVED`는 아무 게이트도 닫지 않는다**(사전등록 §0.2 — 값어치는 반증 쪽에 있었다).
>
> **⑦ 인용 규율.** 인용 대상은 **`p0a_verdict_890893.json`뿐**(게이트 #56 — `.txt`는 콘솔
> 전사). 부착을 인용할 땐 **`run_green_ctx_attached`**(긍정형)만 쓴다 —
> `adapter.attachment_positive.observed`의 값은 생산자 원값 `green_ctx_is_null`이라
> **의미가 반대**다. 드라이버 버전·경과시간·GPU-hr는 `.json`에 **없다**(잡 전사/SLURM 기록).

---

## 9. ★ 등재 **금지** 문장 목록

1. *"구멍 C가 닫혔다"* · *"P0-A가 R4에 답했다"*(사전등록 §0.3-8, R0 §9가 R4를 **L1 전용**으로
   등록).
2. *"cudagraph-ON 운영점에서 SM 분할이 전달된다"* · *"엔진의 decode 그래프도 갇힌다"*.
3. *"한정은 캡처 시점에 박힌다"* · *"replay 스트림은 무관하다"*(§4 비식별).
4. *"물리 SM"* · 라벨 카디널리티의 **계산량/점유율 환산**.
5. 성능·정책 문장 **전부**(latency·throughput·goodput·occupancy·wave quantization·
   고-SM 평탄화 기전). **HE0 불변 · 정책 순위 변경 0건.**
6. *"Gate 2('PD 분리 자체' 귀속)가 전진했다"* · *"S3/G1-d가 닫혔다"*.
7. *"기판이 R0와 일치한다"*(두 축만 검사) · *"매니페스트 검증을 통과했다"*
   (`entry_count_matches_sync_script=false`).
8. *"양성 하한 155(또는 476)를 통과했다"*(문턱은 **1**, 155·476은 관측값).
9. *"하네스가 옳다고 판정됐다"* · *"규칙층이 닫혔다"* · *"9/9(13/13) 변이 통과이므로 계약이
   옳다"*(하네스층 감사 지정 금지 문장 승계).
10. *"green context 하에서 cudagraph capture는 문제없다"*(§5 한정 없이 단독 인용 금지).
11. d16/d44·다른 분할·다른 커널·다중 노드 그래프로의 **일반화**.

---

## 10. 남는 한계(등재 조건 C1–C8)

| # | 조건 |
|---|---|
| **C1** | `.json`만 인용. 부착은 `run_green_ctx_attached`만 인용(극성 함정 P2). |
| **C2** | 교차 레그는 **관측값만**. 기전 문장은 **H2 배제 후에만**(pytorch v2.9.1 `CUDAGraph::replay()` 한 줄 확인, GPU 0). |
| **C3** | 드라이버 버전·경과시간·GPU-hr는 `.json`에 없다 — 출처를 밝히거나 생략. |
| **C4** | ★*"census 텐서를 쓴 것은 replay다"* 는 **CUDA API 보증에 의존**하며 이 런에 양성 대조가 없다. 깨지면 판정은 **공허**해진다(틀린 게 아니라). |
| **C5** | 스코프 4건 추가 필수: 동시부하 0 · 캡처당 replay 1회 · 그래프 노드 1개 · prefill 절반 미시험. |
| **C6** | *"R0와 같은 기판"* 금지 — 검사된 축은 compute mode·census sha256 **둘뿐**, 노드는 다름. |
| **C7** | 교차 레그의 이름↔스트림 순서는 **기계 검사가 없다**(H5 잔여). 오늘 옳음은 사람이 읽어 확인. |
| **C8** | R0 판정이 뒤집히면 **해석 어휘**가 무너진다(§9-2). 단 집합 관측 자체는 이 런에서 재측정됐다. |

**미해결로 남는 질문(이 프로브가 답하지 않은 것)**: 엔진 프로세스 안에서(공유 pool·실제
decode 그래프·요청 있는 상태·양쪽 절반 동시 가동·수천 회 replay·stream-group 전환)에서도
같은가 = **R4/구멍 C 그대로 열림**.

---

## 11. 이를 확정할 정확한 후속 실험 (권고, 값싼 순)

| # | 목적 | 설계 | 비용 |
|---|---|---|---|
| **F1** | §4 비식별(H2) 제거 | pytorch **v2.9.1** 소스에서 `CUDAGraph::replay()`의 launch 스트림 한 줄 확인 | **GPU 0** |
| **F2** | §2 잔여 전제(C4) 제거 | 레그 1개 추가: **캡처만 하고 replay 없음** → census가 **빈 집합**이어야 한다(양성 대조). 기존 하네스에 shim 분기 1개 | ≈0.02 GPU-hr |
| **F3** | 반복 replay 축 | 같은 그래프를 **N회 replay**(N=1/8/64) 후 census. 라벨 집합 불변 여부. 사전등록 필요(결정량은 여전히 `Δ`) | ≈0.05 GPU-hr |
| **F4** | 동시부하 축 | 상보 절반(prefill 74)에 spin 부하를 **동시에** 걸고 graph_green 레그 재측정(R0의 `concurrent_idx1` 형태) | ≈0.05 GPU-hr |
| **F5** | 엔진 층(구멍 C 본체) | **L1 in-server boot census** — 캡처된 그래프가 **엔진의 것**이어야 한다(R0 §9). 이 프로브로는 도달 불가 | 별도 설계 |

★F1·F2는 **이 판정의 두 잔여를 정확히** 닫는다. F1 전에는 §8-⑤의 기전 문장을,
F2 전에는 *"replay가 썼다"* 를 단정하지 않는다.

---

## 12. 재현 방법 (프로젝트 venv, GPU 불요)

```bash
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/bcg_probe

# (1) 채점 재현 — 판정서와 전 키 바이트 동일해야 한다
python3 p0a_graph_sm_confinement.py --analyze p0a_raw_890893.json \
        --tag REPRO --outdir /tmp/<scratch>
python3 - <<'PY'
import json
a=json.load(open('p0a_verdict_890893.json')); b=json.load(open('/tmp/<scratch>/p0a_verdict_REPRO.json'))
print(sorted(set(a)^set(b)), [k for k in a if a[k]!=b.get(k)])   # -> [] []
PY

# (2) 무결성
sha256sum p0a_graph_sm_confinement.py p0a_rule_totality.py p0a_raw_890893.json
git show 366841c:workspace/engine-port/results/bcg_probe/PREREG_P0A_2026-08-22.md | sha256sum

# (3) 비공허성 — 총 히트 = 발사 블록 수, 집합 관계
python3 - <<'PY'
import json; raw=json.load(open('p0a_raw_890893.json'))
S=lambda L: set().union(*[set(s["union"]) for s in L["sweeps"]])
L,C=raw["legs"],raw["cross_legs"]
for n,l in list(L.items())+list(C.items()):
    for s in l["sweeps"]:
        assert sum(map(int,s["hits"].values()))==16740, n
print("eg==gg", S(L["eager_green"])==S(L["graph_green"]))
print("cross_green==eg", S(C["capture_green_replay_plain"])==S(L["eager_green"]))
print("cross_plain==D", S(C["capture_plain_replay_green"])==S(L["eager_plain"]))
PY
```

---

**GPU 지출 0 · 새 성능 판정 0건 · 등급 변경 0건 · 정책 순위 변경 0건 · HE0 불변 ·
gate #13/#16 "닫았다" 금지 · switch-cost "닫았다" 금지 · 전부 유지.**
