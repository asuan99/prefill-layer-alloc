# 사전등록 — **Stage 0‴ A0**: nsys가 green-context **graph-node 커널**을 귀속 가능한 형태로 내는가

**rev3** (2026-08-24). rev1 규칙층 감사 **`NO-GO`**(X1–X13), rev2 **재감사도 `NO-GO`**(N1–N13, ★死因 없음,
`../audit_stage0ppp_a0_rev2_2026-08-24/VERDICT.md`). 이 판본은 재감사가 **구매 전제**로 지목한 **N1·N2**와
N3–N13을 닫는다. 이전 근거: rev1 판정(
`../audit_stage0ppp_a0_2026-08-24/VERDICT.md`). **GPU 미집행 · 새 성능 판정 0건.**
규칙 정본: **`stage0ppp_a0_rule.py`** (`RULE_REV=3`, sha256 `ca7f0fb615cdcd65…`) — 문서와 코드가 다르면 **코드가 이긴다**.

---

## 0. 무엇이고 무엇이 아닌가

**측정 대상**: nsys가 green context 스트림에서 **replay된 CUDA graph의 노드 커널**을
(i) 행으로 내는가 (ii) green context에 **귀속**시킬 수 있는가 (iii) **어느 replay가 냈는지** 이을 수 있는가.

★**도구 타당성 판정이다**(2026-08-20 job 886718 P1 선례). 서버·모델·요청이 **없다** ⇒ 지연·처리량·
goodput을 **구조적으로 생산할 수 없다**. HE0·정책 순위·gate #13/#16 어느 것도 건드리지 않는다.

### ★★ 0-1. 전이 간극 — **A0는 필요조건 스크린이다** (감사 X8)

A0가 재는 것은 **엔진 없이 손으로 만든 spin 커널 그래프**다. 선행 등록이 재려던 것은
**sglang decode 그래프**(`CUDAGraphRunner` + 메모리 풀 + `capture_forward_mode`, 노드 수백 개,
green ctx/스트림 **2개 동시**)다. 따라서:

- **A0-음성**은 (거의) 결정적이다 — 합성 그래프에서도 안 되면 엔진 그래프에서 될 이유가 없다.
- ★**A0-양성은 엔진 그래프로 전이되지 않는다.** 특히 `stream_only` basis는 엔진에서 **훨씬 약해진다**
  (동시 스트림이 2개다). 결과에는 반드시 `substrate="synthetic_probe_graph"`를 병기한다.
- ⇒ **A0+A1을 다 사도 2026-08-15 §2 Stage 0 항목 1은 미구매**로 남는다. **A1에 엔진 기판 Q1/Q2
  재측정을 등록한다**(§12 — rev2의 `(§11)`은 참조 오류였다, 재감사 N12).

### ★★ 0-2. 선행 등록 승계표 (감사 X11 — 승계 상실을 차단으로 채점했다)

rev1은 아래 두 문서를 **인용하지 않았고**, 그 결과 이미 저장소에 있던 사실을 "오늘 확인했다"고 다시 썼다.

| 선행 등록 | 내용 | A0가 사는가 |
|---|---|---|
| `../../../reports/PER_LAYER_TYPE_SM_RESPONSE_DESIGN_2026-08-11.md` **V12**(:188) | `cuda_graph_runner.py:1161` `replay()` — *"replay는 Python forward를 호출하지 않는다"* | **아니오** — 이미 확립된 엔진 사실. 재구매 아님 |
| 같은 문서 **V14**(:190) | stock layerwise NVTX 훅 **존재**(file:line 3개) + *"Python 훅이라 eager 전용"* + ★**올바른 트리를 겨눈 grep 명령**(:643-645) | **아니오** — `../NVTX_EVIDENCE_CORRECTION_2026-08-24.md`가 승계 |
| 같은 문서 **V15**(:191) / **가정 E5**(:210) / **프로브 P4**(:541,563) | nsys granularity 지원 · *"replay된 그래프에서 커널 행을 내는가 — **미검증**"* · **≈0.2 GPU-hr**(엔진 포함 s8 1셀) | ★**A0가 그 질문의 도구층 절반을 산다**(엔진 절반은 A1) |
| `../DESIGN_KERNEL_MECH_2026-08-15.md` **§2 Stage 0 항목 1**(:58-71) | *"`nsys profile`이 pdmux green-context 스트림의 cudagraph 재생 커널을 타임라인에 기록하는가"*, **≈15분, exclusive 불필요**, ★*"이것이 통과하지 못하면 나머지 전부 무의미하다. **사전등록 전 필수**"* | ★**A0의 Q1이 이것이다**(합성 기판 한정, §0-1) |
| 같은 문서 §2 정지 규칙(:70) **뒤 절반** | *"'측정 실패'로 기록하지 '게이트 실패'로 라벨하지 않는다(#21)"* | ★**승계** — A0의 측정조건 라벨 5종이 이것의 구현 |
| 같은 문서 §2 정지 규칙(:70) **앞 절반** | *"1이 실패하면 **전 설계 폐기**"* | ★**승계하지 않고 대체한다**(재감사 N12 — rev2는 이 절반을 조용히 무효화했다). 사유: 단일 음성에 트랙 폐기를 붙이는 규칙은 부모 감사 **E1이 반증**했다(생존 경로 4개). 대체물 = `PRIMARY_ESTIMAND_UNCONSTRUCTIBLE` |
| 같은 문서 §2 항목 2–4(ncu·카운터 오염·서버 생존) | — | **폐기** — Stage B(ncu)는 P1 프로브가 이미 폐기 확정 |

★**감사 판정 인용**: *"이 트랙의 가장 비싼 지출은 GPU가 아니라 이 프로브를 안 산 것"* — A0는
**재도출이 아니라 지연된 이행**이고, 초과분(Q2a/Q2b/대조)도 **과잉 구매가 아니다**.

## 1. 등록 질문

| | 질문 | 형태 |
|---|---|---|
| **Q1** | green ctx replay의 노드 커널 행이 **기대 개수 대비 ≥ `Q1_FRAC`(0.90)** 나오는가 | 비율, 기대치는 **L2에서 측정** |
| **Q2a** | 그 행의 `greenContextId`가 **`NULL`/0이 아닌 값으로 채워지고** `streamId`가 L3의 green 스트림과 **일치**하는가 | 3분기 |
| **Q2b** | 그 행의 `correlationId`가 **그 replay의 `cudaGraphLaunch` RUNTIME 행과 조인**되는가 (성공률 ≥ `JOIN_HIGH`=0.95) | 성공률 병기 |

## 2. 다리 **5개** — ★대조 4겹 (부모 E2 · X5 · 재감사 N7)

★**재감사 N7 정정**: rev2는 §2·§9에서 5 다리, §4에서 4 다리, companion 코드에서 3 다리라고 적었다.
**정본은 5 다리**(L1·L2·L3·L4·L2′)이고 **두 입도 각각** 실행하며 **companion도 5 다리**로 채점한다.

★★**분기 순서도 규칙의 일부다**(재감사 N1). 등록 순서:
**① 측정(run/export) → ② 전면 캡처 실패 → ③ 무결성(절단) → ④ green realized → ⑤ 존재 대조(L1·L2·L3)
→ ⑥ 비대칭 캡처 → ⑦ 조용한 eager fallback → ⑧ Q1 → ⑨ 귀속 → ⑩ 조인**.
rev2는 순서를 자유 모수로 등록하지 않았고, 순서를 바꾸면 라벨 3,840–18,816개가 바뀌는데도 자기검사가
전부 통과했다. rev3은 순서를 **자유 모수 #26**으로 등재하고 **순서 mutant 3종**(`g_order_capgreen_early`·
`g_order_ctl_before_trunc`·`g_order_ctl_before_green`)으로 **배타성 검사 `T20a–e`가 실제로 깨지는지** 실증한다.

한 프로세스에서 **아래 순서로** 실행, granularity마다 `.nsys-rep` 하나:

| # | 다리 | 역할 | 실패 시 |
|---|---|---|---|
| 1 | **L1** full-GPU **eager**(green ctx 생성 **이전**) | 양성대조 — nsys가 커널을 하나라도 보는가 | `PROBE_INVALID` |
| 2 | **L2** full-GPU **graph** 캡처+replay ×20 | 대조(**green 무관**) + ★**기대 노드 수 정의** | `NODE_TRACE_UNAVAILABLE` |
| 3 | **L3** green ctx 스트림 **eager** | 대조 — green 위 커널이 보이는가 | `GREENCTX_INVISIBLE` |
| 4 | **L4** ★**green ctx 스트림 graph 캡처+replay ×20** | **본 조건** | 규칙으로 채점 |
| 5 | **L2′** full-GPU graph 재실행 (**L4 뒤**) | ★**후행 대조**(감사 X5) — 트레이스가 끝까지 살아남았는가 | `TRACE_TRUNCATED` |

★**L2′가 왜 필요한가**: L4가 마지막이면 버퍼 소진·후반 절단 편향이 **정확히 본 조건을 겨눈다**.
rev1은 이 실패를 `PRIMARY_ESTIMAND_UNCONSTRUCTIBLE`(실질 라벨)로 채점했다 — 게이트 #21 위반.

★**green 파티션은 R0와 동일**: `divide_sm(108,(8,0),2)` → `sizes=[74,34]`.
★★**`realized_sm`은 보고 항목이 아니라 판정 분기다**(감사 X3): `green ∈ {matched, mismatched, absent}`.
`absent`(생성 실패·조용한 fallback) → `PROBE_INVALID`, `mismatched` → `GREEN_PARTITION_MISMATCH`.
근거: 이 저장소는 **target을 믿고 realized를 안 봐서** 헤드라인 하나를 잃었다(Stage 0 de-confound,
"D108 앵커가 실은 16 SM"). rev1은 `ctx=null` + `stream=match` 세계에 **최상위 긍정 라벨**을 줬다 —
`stream_only` 멤버십은 **green ctx가 아예 없어도 언제나 성립**하기 때문이다.

## 3. 판정 규칙 = **코드** (게이트 #66)

`stage0ppp_a0_rule.py` `RULE_REV=3`. 자기검사 아티팩트: **`selftest_rev3_2026-08-24.{txt,json}`**
(★재감사 N10 수리 — 이제 **커밋된 스크립트 자신이 `.json`을 쓴다**).

| | rev1 | rev2 | **rev3** |
|---|---|---|---|
| 세계 | 8,192 | 414,720 | ★**663,552** |
| 라벨 | 11 | 15 | **17** |
| mutant | 12 | 18 | ★**24**(★**분기 순서 mutant 3종 포함**) |
| 판별 검사 | 3 | 12 | **16** |
| mutant 커버리지 | 8/12 | 18/18 | **24/24** |
| ★**함의된(공허한) 검사** | 미검사 | ★**2건**(`T14`·`T15`가 `T4`에 함의) | ★**0건**(`T19`로 전수 확인) |

★**추가된 메타 검사 `T10`**(감사 X2의 실질): *"모든 mutant가 판별검사 ≥1개를 깬다"*.
**반증 가능성 실증** — 12개 검사 중 **9개**는 제거하면 커버리지가 **실제로 깨진다**(예: `T9a` 제거 →
`g_q1`·`g_q1_floor`·`g_q1_value` 미커버).

★**X1 수리 — 문턱이 이제 코드에 산다**: rev1에서 `JOIN_HIGH=0.95`는 `score()`가 **한 번도 참조하지
않는 死코드**였고 `Q1_FRAC`은 **0.60/0.70/0.99 어느 값으로 바꿔도 자기검사가 전부 통과**했다.
rev2는 (a) `join_rate`를 실수 축으로 만들어 `JOIN_HIGH`로 비교하고 **0.94/0.95가 문턱을 걸치며**,
(b) **문턱값을 리터럴로 재선언하는 판별검사** `T9a`(`UNCONSTR ⟺ frac < 0.90`)·`T9b`(`OK ⇒ join_rate ≥ 0.95`)를
등록하고, (c) **값 변이 mutant** `g_q1_value`·`g_join_value`가 그 검사를 실제로 깨뜨림을 `T8`로 실증한다.

★**rev1의 "항등식 자기 검출"은 철회한다**(감사 X2): 초판 `T6b`는 **항등식이 아니었고**(`g_contra`가 깬다)
`T8`이 잡은 것은 **검사↔지정 mutant 짝짓기 실패**였다. *"게이트 #9의 17번째 재발"* 표기는 **무효**.

### 라벨 전수와 ★각 라벨이 **허가하지 않는 것** (감사 X7 — rev1 표는 15개 중 5개를 빠뜨렸다)

**측정 조건 (판정 아님, 게이트 #21)**

| 라벨 | 뜻 | ★금지 |
|---|---|---|
| `MEASUREMENT_ABSENT` | 실행/export 실패 | 어떤 판정도 금지 |
| `PROBE_INVALID` | L1 실패 · 전면 캡처 실패 · **green `absent`** | *"Q1=아니오"* 로 읽기 금지 |
| `GREEN_PARTITION_MISMATCH` | realized ≠ target | 성능·도구 판정 **둘 다** 금지 |
| `TRACE_TRUNCATED` | export 부분 성공 · **L2′ 사망** · 비균일 손실 | ★*"노드 행이 부족하다"* 로 읽기 금지 |
| `CAPTURE_FAILS_ONLY_UNDER_GREEN` | L2는 캡처되는데 green만 실패 | ★**실질 사실**이나 **Q1의 답이 아니다**(감사 X13) |

**도구 한계 (green 무관)**

| `NODE_TRACE_UNAVAILABLE` | L2 노드 행 0 | **green과 무관** — green 귀속 서술 금지 |
| `GREENCTX_INVISIBLE` | L3 행 0 | P1과 같은 형태. 성능 함의 없음 |

**실질 판정**

| 라벨 | 뜻 | ★금지 |
|---|---|---|
| `PRIMARY_ESTIMAND_UNCONSTRUCTIBLE` | Q1=아니오 | ★★**"트랙 종결" 아님**(부모 E1: (b)(c)(d) 잔존) |
| `NODE_ROWS_EXCEED_EXPECTATION` | 행이 기대 **초과** | 통과 아님. 기대치 산출식 재검토 신호 |
| `KSET_NOT_ATTRIBUTABLE` | ctx·stream 둘 다 불가 | 도구 한계로 서술 금지(귀속 실패일 뿐) |
| `CONTRADICTORY_ATTRIBUTION` | ctx는 green, stream은 아님 | **하네스/분석기 버그** ⇒ 판정 안 함 |
| `KSET_JOINS_CAPTURE_NOT_REPLAY` | 부모 E5 | 필드 존재를 *"Q2=예"* 로 쓰기 금지 |
| `KSET_NEEDS_TIME_ATTRIBUTION` | 조인 대상 없음 | E1-(d)로 이동. 실패 아님 |
| `KSET_CONSTRUCTIBLE_PARTIAL` / `KSET_CONSTRUCTIBLE` | 조인율 < / ≥ 0.95 | ★★**basis 없이 인용 금지** — `stream_only`면 `greenContextId`가 **아니라** `streamId`로 선 것이다 |

**companion (입도 `graph`)**

| `E1B_UNMEASURED` / `E1B_PROBE_INVALID` | 측정 조건 | 판정 금지 |
| ★`E1B_GRAPH_TRACE_UNAVAILABLE` | **L2g(full-GPU) whole-graph 행 0** | ★**green 무관 도구 한계** — *"(b) 경로가 죽었다"* 로 쓰기 금지 |
| `E1B_DEAD` | L2g는 있는데 **green에서만** 없음 | (b) 경로 부정. **L2g 대조 없이 인용 금지** |
| `E1B_NEEDS_STREAM` / `E1B_ALIVE` | ctx 불가 / 가능 | basis 병기 |

## 4. E1-(b)를 **같은 job에서** 산다 + companion 대조 (감사 X4)

`nsys profile --help`(2025.3.2, 실측): *"If 'node' is selected, node activities will be collected,
**but CUDA graphs will not be traced as a whole**."* ⇒ 입도 **상호 배타** ⇒ 4 다리를 **두 입도로 각각**.

★**rev1 결함**: companion에 **대조가 0겹**이라 *"`graph` 입도가 **어디서도** whole-graph 행을 못 낸다"*
는 **green 무관 도구 한계**가 `E1B_DEAD`(= green에서 (b)가 죽었다)로 등재됐다 — **부모 E2를 그 수리
안에서 재발**시킨 것이다. rev2는 **L2g**(full-GPU graph 대조)·capture·export·green 축을 넣고
`E1B_GRAPH_TRACE_UNAVAILABLE`로 분기한다. **이 다리는 이미 사고 있으므로 추가 비용 0.**

★**답하지 않는 것**: `[:<launch origin>]` = `host-only\|host-and-device`(호스트/디바이스 **코드** 기원)이며
**캡처/replay 시점을 가르지 않는다**(★rev2 정정, 재감사 N11: *"launch origin은 granularity=graph에서만 지원"* 은
**`host-and-device` 값에만** 걸린다 — `host-only`엔 제한이 없다) ⇒ **부모 E5 미해결**. ★감사 추가 확인: launch origin은
**granularity=graph에서만 지원**되고 **기본 granularity는 driver ≥11.7이면 `graph`** ⇒ 1차 실행에
`=node`를 **명시하는 것이 필수**다.

## 5. 환경 등록값 · ★**권한 축** (감사 X10)

| 항목 | 등록값 | 확인 |
|---|---|---|
| nsys | **2025.3.2.474-253236389321v0** | 2026-08-24 실측 |
| CUDA / driver | **13.0.2** / **580.105.08** | ★**컴퓨트 노드 재확인 필수**(게이트 #46) |
| ★`perf_event_paranoid` | 미등록 → **아티팩트에 기록** | 이 클러스터는 `2` |
| ★`NVreg_RestrictProfilingToAdminUsers` | 미등록 → **아티팩트에 기록** | — |
| ★persistence mode | 미등록 → **아티팩트에 기록** | — |

★**선행 E5(2026-08-11)가 "컴퓨트 노드 권한"을 미검증 가정으로 명시**했는데 rev1은 그 축을 자유
모수에서 빠뜨렸다. **총 권한 실패는 `MEASUREMENT_ABSENT`가 흡수하지만 부분 실패는 `TRACE_TRUNCATED`가
받는다**(rev1에선 실질 라벨로 샜다).

★★**관측 채널 등록(재감사 N6)** — rev2는 축만 만들고 *채우는 방법*을 하네스에 남겼다:
- **`green ∈ {matched, mismatched, absent}`**: ★**프로브 자신의 CUDA API 값**으로 판정한다
  (green ctx 생성 반환값 + `numMultiprocessors`를 target `[74,34]`과 대조). ★**nsys 쪽 값
  (`TARGET_INFO_CUDA_CONTEXT_INFO.isGreenContext`)을 쓰면 Q2a와 순환**하므로 **금지**한다.
- **`export ∈ {ok, partial, fail}`**: `nsys export` 종료코드 + ★**`nsys stats`의 손실/드롭 경고**로 판정한다
  (§6에 `--stats` 등록). 경고 원문을 아티팩트에 기록한다.

## 6. 프로파일러 호출 — 전 스위치 등록

```
nsys profile --trace=cuda --sample=none --cpuctxsw=none --gpu-metrics-devices=none \\
             --cuda-graph-trace=node  -o <out>/a0_node_<jobid>  --force-overwrite true \\
             python3 stage0ppp_a0_probe.py --granularity node
# 2차: --cuda-graph-trace=graph, -o a0_graph_<jobid>, --granularity graph
nsys export --type sqlite <rep> -o <sqlite>
nsys stats --report cuda_gpu_kern_sum <rep>   # ★재감사 N6: 손실/드롭 경고를 export 판정에 쓴다
```

- `--sample`·`--cpuctxsw`는 target을 launch하면 **기본이 `process-tree`** 이므로 `none` 명시가 필요하다(감사 ⑤-5).
- ★**`--exclusive`·`--constrain=hwperf` 불필요** — 감사가 `--help`로 확인: ★**rev2 정정(재감사 N11)**: rev2가 권한 요구 목록에 넣은 `--cuda-event-trace`는
  2025.3.2 `--help`에 **권한 문구가 없다**(메인 세션 직접 재확인). 목록에서 **삭제한다**.
  ★**결론은 불변**: `--trace=cuda`에 root/paranoid 요구가 **없다**는 것은 독립 확인됐다. ⇒ 비exclusive `--gres=gpu:1`.
  (★단 이 클러스터엔 반대 선례가 있다 — `../p1_probe/p1_greenctx_ncu_nonexcl.sbatch:26-30`
  *"`--constrain=hwperf` requires `--exclusive`"*, 비exclusive ncu는 `ERR_NVGPUCTRPERM`. 모순은
  **아니다**(카운터 권한 vs CUPTI activity 층) — 그래서 §5에 권한 축을 등록한다.)

## 7. 결정량의 정의 — ★**행 선택 술어와 기대치 산출식** (감사 X6)

rev1은 **비율 문턱만** 고정하고 *어떤 행이 분자에 드는가*와 *기대치를 어떻게 만드는가*를 하네스에
남겼다 ⇒ **결정량의 정의가 하네스 작성자 손에** 있었다. 등록:

- **분자(다리별)**: `CUPTI_ACTIVITY_KIND_KERNEL` 중 `graphNodeId != 0` **그리고** 그 다리의
  `streamId`에 속하는 행. ★`graphNodeId == 0`인 eager 행은 **분자에서 제외하고
  `eager_contamination` 카운터로 별도 보고**(부모 E10이 명시 지목한 혼입).
- **기대치**: L2의 20 replay 각각에 대해 `correlationId`→`cudaGraphLaunch` 조인으로 replay별 노드 수를
  세고 그 **중앙값**을 `expected_nodes_per_replay`로 쓴다. 조인이 L2에서 불가하면
  `total/20`으로 대체하고 `expectation_basis="mean"`을 **아티팩트에 기록**(fallback 사실을 숨기지 않는다).
- **`frac`** = L4 노드 행 수 / (`expected_nodes_per_replay` × 20).
- ★**replay별 벡터를 함께 낸다**(감사 X5): `profile ∈ {full, uniform_short, tail_missing, excess, empty}`.
  `tail_missing`(접미 replay 결손) → `TRACE_TRUNCATED`이지 Q1의 답이 **아니다**. rev1은 스칼라
  하나여서 **균일 손실**(L2 기대치가 함께 줄어 비율이 상쇄 ⇒ 정상으로 보임, 교훈 #20)과 후반 손실을
  구분하지 못했다.

## 8. 자유 모수 — ★**재열거 27개**("전수" 표기는 이 목록에 한한다)

1 SLURM 할당(비exclusive) · 2 `--trace` · 3 `--sample` · 4 `--cpuctxsw` · 5 `--gpu-metrics-devices` ·
6 입도(node/graph **2회**) · 7 launch origin(기본) · 8 `-o`/`--force-overwrite` · 9 종료 방식 ·
10 `nsys export --type sqlite` · 11 다리당 replay 수(**20**) · 12 green 파티션(`(8,0),2`→`[74,34]`) ·
13 spin 커널 grid/지속 · 14 캡처 전 warmup 수 · 15 스트림 배치 · 16 torch/CUDA 모듈 버전 ·
17 아티팩트 형식(`.json` 단일 정본) · 18 결정성(무작위 없음) ·
★19 **행 선택 술어**(§7) · ★20 **기대치 산출식**(§7, 중앙값 vs 평균 fallback) ·
★21 **다리 실행 순서**(L1→L2→L3→L4→**L2′**) · ★22 **두 입도 실행의 순서·독립성**(별 프로세스, node 먼저) ·
★23 **권한 상태 3종**(§5) · ★24 `Q1_FRAC=0.90` · ★25 `JOIN_HIGH=0.95` ·
★26 **`score()`의 분기 순서**(§2, 재감사 N1) · ★27 **`green`·`export` 축의 관측 채널**(§5, 재감사 N6)

## 9. 예산 — ★**등록 가격은 벽시계 상한** (감사 X9)

| 항목 | 값 |
|---|---|
| 다리 | 5 × 2 입도 |
| ★**등록 가격** | **≤ 0.33 GPU-hr** (= `--time=00:20:00`, 비exclusive `--gres=gpu:1`) |
| 참고 예상치 | ≈0.03–0.05 GPU-hr — ★**앵커 미검증**이므로 참고치로만 |
| 엔진 부팅 | **0** |

★rev1은 `--time=00:20:00`(0.33)과 `≈0.03–0.05`를 **같은 표에** 올려 7× 차이 두 가격을 띄웠고,
인용한 선례 둘 다 **nsys를 돌린 적이 없다**(R0=프로파일러 무, P1=ncu·**exclusive**). ⇒ 등록은 상한으로,
실측은 사후 기록으로 분리한다.
★**선행 P4의 0.2 GPU-hr는 A0가 대체하지 않는다** — 그것은 **엔진 포함 s8 1셀**이고 §0-1의 전이 간극
때문에 **A1에서 여전히 사야 한다**.

## 10. 아티팩트 규율

★**`.json` 단일 정본**(게이트 #56 — R0 결함 N1: `.txt` 절단으로 3필드 소실) · ★**필드명↔값 극성 일치**
(게이트 #57) · 병기 필수: nsys/CUDA/driver(**컴퓨트 노드**) · **권한 3종** · `realized_sm` · 규칙 sha256 ·
`profile` 벡터 · `eager_contamination` · `expectation_basis` · 조인 성공률 · **basis** · `substrate`.
## 11. ★ 쓰면 안 되는 문장

**승계**: *"kernel_mech 트랙을 열었다/닫았다"* · *"rev7·rev8 차단이 해소됐다"*(B1–B5·B7 / C1–C10 **불변**) ·
*"`TOOL_CANNOT_DEFINE_K_SET`는 트랙 종결 조건이다"* · *"엔진에 NVTX가 없다"* · *"NVTX 선행조건이 사라졌다"* ·
*"nsys는 green-context 커널을 낸다"*(A0 실행 전) · HE0 · gate #13/#16 · switch-cost · C2 인용정지 (a)(b) ·
**rev1 감사(X1–X13) 신설분 전부**.

**rev2 재감사(N1–N13)가 신설 — 전부 승계**:
- ✗ *"rev2/rev3이 규칙층 감사를 통과했다"* — ★**rev3은 미감사**다.
- ✗ ★★★ *"`T14`/`T15`가 X3/X5를 검증한다"* / *"`T14`가 276,480 세계를, `T15`가 304,128 세계를 구속한다"* —
  ★**둘 다 `T4`에 함의됐다**(반례 0/18,662,400). **한계 구속 0세계.** rev3에서 배타성 형태로 교체했고
  `T19`가 함의 쌍을 전수 검사한다.
- ✗ ★★★ *"규칙이 코드로 고정됐으므로 채점이 보호된다"* — rev2에 대해 **거짓이었다**(분기 순서가 무방비).
  rev3에 대해서만, **순서 mutant 3종과 `T20a–e`가 확인한 범위에서** 참.
- ✗ ★★ *"문턱 0.90/0.95가 코드에 산다"* — rev2에선 **밴드 `(0.80,1.00]` / `[0.95,1.00]`에만** 살았다.
  rev3은 `frac`에 0.89/0.91을 넣고 `T9b`를 iff로 올렸으나, **밴드가 완전히 한 점으로 좁혀졌다는 주장은 하지 않는다.**
- ✗ ★★ *"`GREENCTX_INVISIBLE`은 green-무관 도구 한계다"* — rev2에선 부분 export·후행 대조 사망·파티션
  불일치·green 부재가 **전부 이 라벨로 왔다**(clean 6.3%). rev3은 `T20a`로 막았으나 **이 문장 자체는 금지 유지**.
- ✗ ★★ *"`CAPTURE_FAILS_ONLY_UNDER_GREEN`은 실질 사실이다"* — rev2에선 정합 24세계 중 **clean 1개**.
- ✗ ★★ *"green ctx가 realized로 검증되므로 `stream_only`가 정당화된다"* — `matched`는 **컨텍스트 생성**의
  증거이지 **커널 소속**의 증거가 아니다. rev3은 아예 **`OK`가 `ctx+stream` basis를 요구**하도록 바꿨다.
- ✗ ★ *"companion이 고쳐졌다"*(rev2 기준) · *"조용한 eager fallback이 흡수된다"*(rev2 기준).
- ✗ ★ *"자기검사가 아티팩트로 재현된다"*(rev2 기준 — `.json`을 커밋된 스크립트가 만들 수 없었다).
- ✗ ★ *"`--cuda-event-trace`는 root/paranoid를 요구한다"*(**거짓**) · *"launch origin은 graph 입도에서만
  지원된다"*(**`host-and-device` 값만**).
- ✗ *"세계 공간이 커졌으므로 커버리지가 그만큼 커졌다"* — rev2의 414,720 중 **정합은 7,072개**(98.29% 부정합).
  ★rev3의 663,552에도 같은 주의가 적용된다.

## 12. 다음 단계

**① 규칙층 감사(rev3 재감사)** → ② 하네스(`stage0ppp_a0_probe.py` + `.sbatch`) → ③ **하네스층 감사(2단)** → ④ 제출.
★**②③ 없이 제출 금지.** ★**A1에 등록할 것**(§0-1): **엔진 기판 Q1/Q2 재측정** · K1(양 다리 모두 엔진
telemetry ITL 중앙값) · Q3(축소). ★감사 권고 — node 입도의 *"significant runtime overhead"*(도구 문서
자신의 경고)에 대해 **프로파일러 없이 같은 프로브를 1회 더 도는 다리**를 붙이면 K1의 **프로브층 하한**을
거의 공짜로 산다(★엔진 K1로 전이 금지).

## 13. 변경 이력 — ★각 행에 **실행된** 검증만 적는다 (게이트 #62/#67/#70)

| 판본 | 변경 | ★검증 방법(예측 아님, **실행**) |
|---|---|---|
| Stage 0″ | (전신) | 규칙층 감사 `NO-GO`, 차단 E1–E11 |
| A0 rev1 | 규칙 코드화·대조 3겹·E5 분기·companion | 규칙층 감사 `NO-GO`, 차단 **X1–X13**, 死因 없음 |
| A0 rev2 | X1–X13 반영 시도 | ★**재감사 `NO-GO`, 차단 N1–N13, 死因 없음** — 닫힘 4 · **부분 9** |
| ~~rev2 X3 검증 = `T14`, n=276,480 세계 구속~~ | ★**철회(N2)** | `T4 ⟹ T14` 반례 **0/18,662,400** ⇒ **한계 구속 0세계**(메인 세션 재현) |
| ~~rev2 X5 검증 = `T15`, n=304,128 세계 구속~~ | ★**철회(N2)** | `T4 ⟹ T15` 반례 **0** ⇒ 동일 |
| **rev3** N1 분기 순서 | 순서 재배치(§2 ①–⑩) + **순서 mutant 3종** + **배타성 검사 `T20a–e`** | `T8`: `T20c`가 `g_order_capgreen_early`에서, `T20a/b`가 `g_order_ctl_before_trunc`에서 **실제 실패 확인**(`selftest_rev3_2026-08-24.txt`) |
| **rev3** N2 함의 배제 | `T14`/`T15`를 배타성 형태로 교체 + ★**메타검사 `T19`** 신설 | `T19` 전 쌍(24×23) 전수 → **함의 0쌍**(`entailed_pairs: []`). ★2단 탐색은 **건전**하다(부분집합의 반례는 전 공간의 반례) |
| **rev3** N3 문턱 밴드 | `PROFILE_FRAC`에 0.89/0.91 + `T9b` iff 승격 | `T9a`/`T9b`가 `g_q1_value`(0.85)·`g_join_value`에서 실제 실패 |
| **rev3** N4 적극적 반대 증거 | `ctx="parent"` → 신규 **`ATTRIBUTION_DISCONFIRMED`** + `T11b` | `T11b`가 `g_disconf`·`g_ctx`에서 실제 실패 |
| **rev3** N5·N7 companion | **5 다리**로 확장 + `mismatched`·`partial`·`fail_green_only`·`l2g_post` 축 + `E1B_TRACE_TRUNCATED`·`E1B_GREEN_PARTITION_MISMATCH` | companion **6,912 세계**, `C1–C6` PASS, `C6`(=`E1B_DEAD`가 다른 실패로 오염되지 않음)이 `h_trunc`에서 실제 실패 |
| **rev3** N8 stream 구조성 | **`OK`는 `ctx+stream` basis를 요구**, `stream_only`는 신규 **`KSET_STREAM_ONLY_ATTRIBUTION`**(1차 추정량 아님) | `T21`이 `g_streamonly`에서 실제 실패 |
| **rev3** N9 eager fallback | `profile="eager_only"` 축 + **`PROBE_INVALID`** 분기 | `T22`가 `g_eager`에서 실제 실패 |
| **rev3** N6·N10·N11·N12·N13 | 관측 채널 등록(§5) · **커밋된 스크립트가 `.json`을 쓴다** · nsys 문장 2건 정정 · 승계표 "대체" 명시 + `(§11)`→`(§12)` · 정본 금지문 정정 | `selftest_rev3_2026-08-24.json`(`all_pass: true`, `uncovered_mutants: []`, `entailed_pairs: []`) · `nsys profile --help` 재확인 · `PROJECT_STATUS.md:4587` 정정 |
| **rev3** 자기 검출 2건 | `_reaches_join`이 `join_target=="replay"`를 요구해 `T12`/`T9b` iff가 성립 불가 · `T20a/b`가 `capture=="ok"`를 과잉 요구 | ★**검사 스위트가 첫 실행에서 잡았다**(5 FAIL → 수리 → 1 FAIL → 수리 → `ALL PASS`). ★그 1건도 **검사↔mutant 짝짓기 실패**(라벨 삭제 mutant로는 "clean" 검사를 못 깬다)였고 **순서 mutant 도입으로 해소**했다 |
