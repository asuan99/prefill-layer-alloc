# 사전등록 — **Stage 0‴ A0**: nsys가 green-context **graph-node 커널**을 귀속 가능한 형태로 내는가

**rev2** (2026-08-24). rev1은 규칙층 감사 **`NO-GO`**(차단 X1–X13, ★死因 없음,
`../audit_stage0ppp_a0_2026-08-24/VERDICT.md`). 이 판본은 감사가 **구매 전제**로 지목한
**X1·X3·X4·X5·X8**과 나머지 8건을 닫는다. **GPU 미집행 · 새 성능 판정 0건.**
규칙 정본: **`stage0ppp_a0_rule.py`** (`RULE_REV=2`, sha256 `7d61d6bc821f4c83…`) — 문서와 코드가 다르면 **코드가 이긴다**.

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
  재측정을 등록한다**(§11).

### ★★ 0-2. 선행 등록 승계표 (감사 X11 — 승계 상실을 차단으로 채점했다)

rev1은 아래 두 문서를 **인용하지 않았고**, 그 결과 이미 저장소에 있던 사실을 "오늘 확인했다"고 다시 썼다.

| 선행 등록 | 내용 | A0가 사는가 |
|---|---|---|
| `../../../reports/PER_LAYER_TYPE_SM_RESPONSE_DESIGN_2026-08-11.md` **V12**(:188) | `cuda_graph_runner.py:1161` `replay()` — *"replay는 Python forward를 호출하지 않는다"* | **아니오** — 이미 확립된 엔진 사실. 재구매 아님 |
| 같은 문서 **V14**(:190) | stock layerwise NVTX 훅 **존재**(file:line 3개) + *"Python 훅이라 eager 전용"* + ★**올바른 트리를 겨눈 grep 명령**(:643-645) | **아니오** — `../NVTX_EVIDENCE_CORRECTION_2026-08-24.md`가 승계 |
| 같은 문서 **V15**(:191) / **가정 E5**(:210) / **프로브 P4**(:541,563) | nsys granularity 지원 · *"replay된 그래프에서 커널 행을 내는가 — **미검증**"* · **≈0.2 GPU-hr**(엔진 포함 s8 1셀) | ★**A0가 그 질문의 도구층 절반을 산다**(엔진 절반은 A1) |
| `../DESIGN_KERNEL_MECH_2026-08-15.md` **§2 Stage 0 항목 1**(:58-71) | *"`nsys profile`이 pdmux green-context 스트림의 cudagraph 재생 커널을 타임라인에 기록하는가"*, **≈15분, exclusive 불필요**, ★*"이것이 통과하지 못하면 나머지 전부 무의미하다. **사전등록 전 필수**"* | ★**A0의 Q1이 이것이다**(합성 기판 한정, §0-1) |
| 같은 문서 §2 정지 규칙(:70) | *"'측정 실패'로 기록하지 '게이트 실패'로 라벨하지 않는다(#21)"* | ★**승계** — A0의 측정조건 라벨 5종이 이것의 구현 |
| 같은 문서 §2 항목 2–4(ncu·카운터 오염·서버 생존) | — | **폐기** — Stage B(ncu)는 P1 프로브가 이미 폐기 확정 |

★**감사 판정 인용**: *"이 트랙의 가장 비싼 지출은 GPU가 아니라 이 프로브를 안 산 것"* — A0는
**재도출이 아니라 지연된 이행**이고, 초과분(Q2a/Q2b/대조)도 **과잉 구매가 아니다**.

## 1. 등록 질문

| | 질문 | 형태 |
|---|---|---|
| **Q1** | green ctx replay의 노드 커널 행이 **기대 개수 대비 ≥ `Q1_FRAC`(0.90)** 나오는가 | 비율, 기대치는 **L2에서 측정** |
| **Q2a** | 그 행의 `greenContextId`가 **`NULL`/0이 아닌 값으로 채워지고** `streamId`가 L3의 green 스트림과 **일치**하는가 | 3분기 |
| **Q2b** | 그 행의 `correlationId`가 **그 replay의 `cudaGraphLaunch` RUNTIME 행과 조인**되는가 (성공률 ≥ `JOIN_HIGH`=0.95) | 성공률 병기 |

## 2. 다리 — ★대조 **4겹** (부모 E2 + 감사 X5)

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

`stage0ppp_a0_rule.py` `RULE_REV=2`. 자기검사 아티팩트: **`selftest_rev2_2026-08-24.{txt,json}`**.

| | rev1 | **rev2** |
|---|---|---|
| 세계 | 8,192 | ★**414,720** |
| 라벨 | 11 | **15**(전부 도달 가능 실증) |
| mutant | 12 | **18**(전부 load-bearing) |
| 판별 검사 | 3 | ★**12** |
| **mutant 커버리지** | ★**8/12** | ★**18/18** |
| 공허한 검사 | 미검사 | ★**0건**(최소 구속 8세계, 전 검사 실측) |

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
**캡처/replay 시점을 가르지 않는다** ⇒ **부모 E5 미해결**. ★감사 추가 확인: launch origin은
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

## 6. 프로파일러 호출 — 전 스위치 등록

```
nsys profile --trace=cuda --sample=none --cpuctxsw=none --gpu-metrics-devices=none \\
             --cuda-graph-trace=node  -o <out>/a0_node_<jobid>  --force-overwrite true \\
             python3 stage0ppp_a0_probe.py --granularity node
# 2차: --cuda-graph-trace=graph, -o a0_graph_<jobid>, --granularity graph
nsys export --type sqlite <rep> -o <sqlite>
```

- `--sample`·`--cpuctxsw`는 target을 launch하면 **기본이 `process-tree`** 이므로 `none` 명시가 필요하다(감사 ⑤-5).
- ★**`--exclusive`·`--constrain=hwperf` 불필요** — 감사가 `--help`로 확인: root/paranoid 요구는
  `--cpu-core-events`·`--cpuctxsw=system-wide`·`--cuda-event-trace`·`--ftrace`·`--run-as`·
  `--sample=system-wide`에만 붙고 **`--trace=cuda`에는 없다**. ⇒ 비exclusive `--gres=gpu:1`.
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

## 8. 자유 모수 — ★**재열거 25개**("전수" 표기는 이 목록에 한한다)

1 SLURM 할당(비exclusive) · 2 `--trace` · 3 `--sample` · 4 `--cpuctxsw` · 5 `--gpu-metrics-devices` ·
6 입도(node/graph **2회**) · 7 launch origin(기본) · 8 `-o`/`--force-overwrite` · 9 종료 방식 ·
10 `nsys export --type sqlite` · 11 다리당 replay 수(**20**) · 12 green 파티션(`(8,0),2`→`[74,34]`) ·
13 spin 커널 grid/지속 · 14 캡처 전 warmup 수 · 15 스트림 배치 · 16 torch/CUDA 모듈 버전 ·
17 아티팩트 형식(`.json` 단일 정본) · 18 결정성(무작위 없음) ·
★19 **행 선택 술어**(§7) · ★20 **기대치 산출식**(§7, 중앙값 vs 평균 fallback) ·
★21 **다리 실행 순서**(L1→L2→L3→L4→**L2′**) · ★22 **두 입도 실행의 순서·독립성**(별 프로세스, node 먼저) ·
★23 **권한 상태 3종**(§5) · ★24 `Q1_FRAC=0.90` · ★25 `JOIN_HIGH=0.95`

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
*"nsys는 green-context 커널을 낸다"*(A0 실행 전) · HE0 · gate #13/#16 · switch-cost · C2 인용정지 (a)(b).

**감사 X1–X13이 신설한 것**:
- ✗ *"A0가 통과하면 엔진 decode 그래프에서도 node row가 난다"*(X8 — 기판이 다르다)
- ✗ *"Q1·Q2b가 NVTX 선행조건 질문에 실측으로 답한다"* — ★**rev1 §10에 있던 문장이며 삭제했다**(A0는 엔진을 돌리지 않는다)
- ✗ *"A0가 2026-08-15 §2 Stage 0 항목 1을 이행했다"*(필요조건 스크린일 뿐)
- ✗ *"규칙이 코드로 고정됐으므로 문턱이 보호된다"* — rev1에 대해 **거짓이었다**. rev2에 대해서만, `T9a`·`T9b`·`T10` 범위에서 참
- ✗ *"`T8`이 항등식을 잡는다"* / *"초판 `T6b`는 항등식이었다"*(X2 — 철회됨)
- ✗ *"mutant가 전부 판별검사로 덮인다"* — rev1은 **8/12**였다. rev2는 18/18이나 **`T10`을 실행해 확인한 범위에서만**
- ✗ *"`E1B_DEAD` = (b) 경로가 죽었다"* — **L2g 대조 없이 금지**(X4)
- ✗ *"`KSET_CONSTRUCTIBLE`이 떴으므로 green context 귀속이 선다"* — **basis 없이 금지**(X3)
- ✗ *"A0는 0.03–0.05 GPU-hr다"*(X9 — 등록가는 상한 0.33)
- ✗ *"`PROBE_INVALID`이므로 아무것도 알 수 없다"*(X13 — `CAPTURE_FAILS_ONLY_UNDER_GREEN`은 실질 사실)

## 12. 다음 단계

**① 규칙층 감사(rev2 재감사)** → ② 하네스(`stage0ppp_a0_probe.py` + `.sbatch`) → ③ **하네스층 감사(2단)** → ④ 제출.
★**②③ 없이 제출 금지.** ★**A1에 등록할 것**(§0-1): 엔진 기판 Q1/Q2 재측정 · K1(양 다리 모두 엔진
telemetry ITL 중앙값) · Q3(축소). ★감사 권고 — node 입도의 *"significant runtime overhead"*(도구 문서
자신의 경고)에 대해 **프로파일러 없이 같은 프로브를 1회 더 도는 다리**를 붙이면 K1의 **프로브층 하한**을
거의 공짜로 산다(★엔진 K1로 전이 금지).

## 13. 변경 이력 — ★각 행에 **실행된** 검증만 적는다 (게이트 #62/#67/#70)

| 판본 | 변경 | ★검증 방법(예측 아님, **실행**) |
|---|---|---|
| Stage 0″ | (전신) | 규칙층 감사 `NO-GO`, 차단 E1–E11 |
| A0 rev1 | 규칙 코드화·대조 3겹·E5 분기·companion | 규칙층 감사 `NO-GO`, 차단 **X1–X13**, 死因 없음 |
| **rev2** X1 문턱을 코드에 살림 | `join_rate` 실수 축 + `T9a`/`T9b` 리터럴 재선언 + 값 변이 mutant | `T8`: `g_q1_value`·`g_q1_floor`·`g_q1`·`g_join_value`에서 **실제 실패 확인**(`selftest_rev2_2026-08-24.txt`) |
| **rev2** X2 커버리지 메타검사 | `T10` 신설 + 판별검사 3→**12** | `T10` **반증 가능성 실측**: 12개 중 **9개**를 제거하면 커버리지가 실제로 깨짐. 공허 검사 **0건**(최소 구속 8세계) |
| **rev2** X3 green-ness 가드 | 축 `green ∈ {matched,mismatched,absent}` + `GREEN_PARTITION_MISMATCH` | `T14`(+`T8` `g_green`), `n=276,480` 세계 구속 |
| **rev2** X4 companion 대조 | `l2g`·capture·export·green 축 + `E1B_GRAPH_TRACE_UNAVAILABLE` | `C5`(+메타 `h_l2g`), companion 512 세계 `C1–C5` PASS |
| **rev2** X5 절단 가드 | `profile` 벡터 + **L2′ 후행 대조** + export 3분기 | `T15`(+`T8` `g_trunc`), `n=304,128` 세계 구속 |
| **rev2** X13 비대칭 캡처 | `CAPTURE_FAILS_ONLY_UNDER_GREEN` | `T17`(+`T8` `g_capgreen`·`g_capture`), 도달 `n=23,040` |
| **rev2** X6·X7·X8·X9·X10·X11·X12 | 행 술어·기대치 산출식 등록 · 라벨 표 **15+6 전수** · 전이 간극 §0-1 · 등록가 상한 · 권한 축 · **선행 등록 승계표 §0-2** · 자기검사 아티팩트 저장 | `selftest_rev2_2026-08-24.json`(`uncovered_mutants: []`) · 라벨 표 ↔ 코드 자동 대조 |
