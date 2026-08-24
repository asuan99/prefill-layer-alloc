# 감사 판정서 — `Stage 0″` 사전등록 (2026-08-24, claims-auditor, 게이트 #34 1단)

**판정: `NO-GO` — 차단 E1–E11 · ★死因 없음**(전부 GPU 0으로 닫힘).
GPU 지출 **0** · 새 성능 판정 **0건** · 등급 변경 **0건** · 정책 순위 변경 **0건**.

> ★**작성 경위**: 감사자가 규율상 대상 트리에 쓰지 않고 전문을 반환했고 메인 세션이 보존했다.
> 대상: `PREREG_STAGE0PP_2026-08-23.md`(146줄).

## 0. 단일 질문 — **두 절반 모두 아니오**

**(a) "Q1·Q2 어느 답이든 등재 가능한가"** → ★**아니오, 한쪽만.** `Q1=예 ∧ Q2=예`는 등재
가능하나 ★**`Q1=아니오`는 이 설계로 등재 불가** — **양성 대조가 0건**이라 *"green-ctx 커널이
안 보인다"* 와 *"이 런이 아무것도 트레이스 못 했다"*(부착 오지정·버퍼 손실·서버/클라이언트
분리)를 **구분할 수 없다**. 그 상태로 `TOOL_CANNOT_DEFINE_K_SET`를 등재하면 **측정 실패를
게이트 실패로 라벨링**(게이트 #21, 최다 재발 8회+)이고 하필 그 결과가 **트랙 종결**이라
되돌릴 기회가 없다. P1이 자기 귀속이 깨끗하다고 한 근거인 ★**"두 겹의 대조"** 를 **하나도
승계하지 않았다**.
**(b) "종결 조건인가"** → ★★**아니다. 재설계 조건이다**(E1).

---

## 1. ★★★ 표적 2의 답 — **절반은 이미 알려져 있었다**(게이트 #36 부분 위반)

로그인 노드 **명령 5줄·GPU 0**으로 확인:

1. **도구 확정**: `/apps/cuda/13.0.2/bin/nsys` → **nsys 2025.3.2.474-253236389321v0**,
   `--cuda-graph-trace=<granularity>[:<launch origin>]`에 `node` 값 존재.
2. ★**Q2의 문서층 답 = "예"** — 동봉 export 스키마:
   `CUPTI_ACTIVITY_KIND_KERNEL(contextId, ★greenContextId, streamId, correlationId, ★graphNodeId)` ·
   `TARGET_INFO_CUDA_CONTEXT_INFO(★parentContextId, ★isGreenContext)`.
   그리고 **stock 리포트가 그 컬럼을 실제로 SELECT** 한다(`reports/cuda_gpu_trace.py:88,115,145,184`
   — `greenContextId AS "greenContext"`, 헤더에 *"GreenCtx / Strm / Ctx"*).
3. **CUPTI 층 확인**: `cupti_activity.h` — `CUpti_ActivityContext3`에 `isGreenContext`·
   `parentContextId`·`numMultiprocessors`, `CUpti_ActivityKernel10`에 `graphNodeId`·`graphId`.
4. ★**표적 4(캡처 vs replay)의 답도 문서에 있다** — `graphNodeId` 주석:
   *"The unique ID of the graph node that **launched this kernel through graph launch APIs**.
   This field will be 0 if the kernel is not launched through graph launch APIs."*
   ⇒ node-granularity는 **실행(replay) 시점 활동에 노드 id를 태깅**하는 것. ★**사전등록은 이
   구분을 혼동하지 않았다** — 다만 **근거를 한 줄도 인용하지 않았다**.
5. ★**`K_set(k)` 구성 방식이 nsys의 shipped feature다** — `reports/nvtx_gpu_proj_trace.py`:
   *"A GPU operation is considered 'contained' by an NVTX range **if the CUDA API call used to
   launch the operation is within the NVTX range**."* ⇒ rev8 §2의 정의와 **문자 그대로 같다**
   (조인 키 = `correlationId`).
6. **저장소엔 답이 없다** — `*.nsys-rep`/`*.sqlite` **0건**, `p1_probe/`는 **ncu 전용**.

| | 이미 답이 있는가 |
|---|---|
| **Q2**(필드 존재) | ★**예 — 문서·스키마·헤더에서 확정, GPU 0** |
| **Q1**(green-ctx 커널이 node row로) | **아니오 — 이 드라이버(580.105.08) 실측 없음** |
| **표적 4** | ★**예 — 문서로 해소** |

⇒ ★**게이트 #36 부분 위반**: 사전등록이 §5-2에서 *"이 확인이 가장 값싼 표적"* 이라고 **스스로
지정해 놓고 한 줄도 수행하지 않은 채** 감사에 넘겼다.

---

## 2. 반증 실패 — 사전등록이 **옳게 한 것**

1. ★★**§4-4(P1 구분)는 옳고 이제 근거가 생겼다** — ncu 실패는 CUPTI **프로파일링/PM 카운터**
   층, nsys는 CUPTI **activity** 층이고 activity 층은 green context를 **1급 시민으로 모델링**한다.
   ⇒ **P1의 `UNAVAILABLE`은 nsys를 함의하지 않는다**(표적 5 반증 실패). ★단 §4의 실험 전에는
   `CONFIRMED` 아님(문서가 지원을 선언 ≠ 이 드라이버가 채운다).
2. **표적 4 반증 실패** · 3. **게이트 #21 자각이 본문에 명시**(단 E2로 실효 없음) ·
4. **§4-1·2·3·5·6 정확**(단 Q2 쪽 거울상 누락 = E5) · 5. **rev8 우회 선언 정직** ·
6. **NVTX 없이 물을 수 있다는 판단은 옳고 §4에서 더 강하게 옳다** · 7. **정본 오염 0**.

---

## 3. 차단 E1–E11 (요약)

### ★★★E1 — `TOOL_CANNOT_DEFINE_K_SET`는 **재설계 조건**이다(표적 1)
대안이 전수되지 않았다. 생존 경로 최소 4개:

| # | 경로 | 살아남는 것 |
|---|---|---|
| **(a)** | ★**graph-launch API row 귀속** — cudagraph-ON에서 decode 스텝 = `replay()` **1회**(`cuda_graph_runner.py:1155-1161`). `CUPTI_ACTIVITY_KIND_RUNTIME`의 graph-launch row가 **host 시각 경계 + correlation id를 동시에** 준다 | `boundary(k)`·`K_set(k)` **둘 다**. ★**NVTX 불요** |
| (b) | `--cuda-graph-trace=graph` + `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`(스키마에 `greenContextId`·`correlationId`·`graphExecId` 전부) | `T_span`·`G_inter`·`frac_inter`. 죽는 것은 `Σkernel_dur` ⇒ **`G_intra`(1차)만** |
| (c) | 스트림별 분리(`streamId` 단독 — decode는 단일 스트림) | green ctx 식별자 없이도 멤버십 |
| (d) | device-시각 포함 귀속 — rev8이 **설계 선택으로** 버린 경로이지 불가능한 경로가 아니다 | `K_set(k)` |

⇒ ★`Q1=아니오`가 죽이는 것은 **1차 결정량**이지 트랙 전체가 아니다. `Q2=아니오`는 **아무것도
죽이지 않는다.** **수리**: `PRIMARY_ESTIMAND_UNCONSTRUCTIBLE`로 강등 + 4경로를 본문에 열거.

### ★★★E2 — **양성 대조 0건.** `Q1=아니오` 가지는 등재 불가
P1이 깨끗했던 이유는 **두 겹의 대조**, `%smid` R0는 **3중 대조**. Stage 0″는 **0건**.
실패 모드가 전부 `Q1=아니오`로 오독된다(부착 오지정·CUDA trace 미활성·버퍼 손실·조기 종료·
export 실패). **수리**: 같은 프로세스에 (i) green ctx 밖 eager (ii) green ctx 밖 graph replay
(iii) green ctx 위 eager 를 대조로 넣고 **(i)(ii) 실패 시 `PROBE_INVALID`로 중단**.

### ★★★E3 — 게이트 #36(§1). **수리**: 근거 인용 + Q2를 *"필드 존재"* → ★*"이 드라이버·이
replay 경로에서 `NULL`/0이 아닌 값으로 **채워지는가**"* 로 재작성(3분기).

### ★★★E4 — **하네스가 틀렸다: Q1·Q2는 엔진이 필요 없다**
저장소에 **엔진 없이 green ctx를 만드는 프로브가 둘** 있고 둘 다 완주했다
(`smid_l0_census.py:810`이 ★**엔진이 실제로 호출하는 그 함수**를 쓴다, 0.0275 GPU-hr ·
`p1_probe/p1_greenctx_target.py`, 0.032). 반대로 엔진이 정말 필요한 **Q3·K1은 결정 규칙이 0개**
(E8·E9). ⇒ ★**비싸고 깨지기 쉬운 절반을 규칙 없는 항목 때문에 사고, 공짜로 살 수 있는 절반을
같이 위험에 노출**시킨다.

### ★★E5 — `TOOL_CAN_DEFINE_K_SET` 라벨도 **과잉**(§4-3의 거울상 누락)
graph node 커널에서 "launch한 API"가 **replay 시점 `cudaGraphLaunch`** 인지 **캡처 시점 개별
launch** 인지 **문서가 가르지 않는다**. 후자면 필드는 있는데 멤버십에 **못 쓴다**.
**수리**: Q2를 ★*"node row의 correlation id가 그 replay의 graph-launch API row와 **조인
성공하는가**"*(성공률 보고)로.

### ★★E6 — **`P0`가 실행 불가능**(등록값 부재). 감사가 확인한 값:
nsys **2025.3.2.474-253236389321v0** · CUDA **13.0.2** · driver **580.105.08**(★**컴퓨트 노드에서
재확인 필수** — 게이트 #46 로그인 노드 귀속 오류 선례).

### ★★E7 — **프로파일러 호출·부착 대상 미등록**
`--trace`·`--sample`/`--cpuctxsw`(이 클러스터 `perf_event_paranoid=2`, K1 오염)·
★`--gpu-metrics-devices`(rev8 P5가 *"System scope 샘플러가 `G_intra` 오염"*이라 했는데 **OFF
등록 없음**)·`-o`·종료 방식 전부 미등록.
★**가장 위험**: B6 워크로드는 **서버 + 클라이언트 두 프로세스**인데 **nsys가 어느 쪽을 감싸는지
미등록**이다. 클라이언트를 감싸면 GPU row **0개** → `Q1=아니오` → **트랙 종결**.
★**가격 부수**: CUDA API/커널 트레이스는 **`--exclusive --constrain=hwperf`가 불필요**하다
(그 권한은 `--gpuctxsw`·system-wide sampling·GPU metrics 전용) ⇒ P1처럼 exclusive를 쓰면 가격이
**최대 8× 틀어진다**.

### ★★E8 — **K1: 시계 채널 미등록 · 순환 · 사용 규칙 0**(표적 3)
ON 다리를 타임라인으로 재려면 **스텝 경계가 필요**(= Q3) ⇒ ★**Q3=아니오면 K1이 정의되지
않는다**(순환). *"자릿수로만 쓴다"* 는 ★**현재 형태로 정직하지 않다** — D9가 지적했듯 Stage
0′/A 가격이 **바로 이 배수의 함수**다. **수리**: 양 다리 모두 **엔진 telemetry ITL 중앙값**으로
통일(Q3 의존 절단) · **구간 라벨**(`≈1/1.x/≥2/≥10`) 등록 · 3자리 표기 금지 · **중단 규칙 등록**.

### ★★E9 — **Q3: 판정 규칙 없음 + 대안 미전수 + ★근거 grep의 스코프 오류**
* 등록된 귀결 *"안 되면 NVTX 선행조건이 확정된다"* 는 **거짓** — E1-(a)가 남는다.
* ★★**"NVTX 패치 미작성" 근거가 틀린 트리를 봤다**: `grep -rn nvtx workspace/engine-port/src/`
  = 0건은 **PD-mux 오버레이 디렉터리**이고, **실제로 도는 엔진**(`sglang_engine_dev/python/
  sglang/srt/`)에는 **NVTX가 이미 있다**(`server_args.py:616,5384`
  `--enable-layerwise-nvtx-marker` · `utils/nvtx_pytorch_hooks.py` ·
  `model_executor/model_runner.py:1196`). ⇒ **차단의 근거 문장이 무효**(교훈 #57 계열).
  ★공정하게: 그 훅은 **모듈 forward hook**이라 cudagraph replay에서 **발화하지 않을 개연성이
  크다** — 그러나 그것은 **문서에 없는 추론**이고, 지금 상태에선 **결론이 우연히 살아있을 뿐
  근거는 무효**다.
* **문턱 없는 서술** = rev8 D1-(d)가 지목한 형태가 **그 차단을 우회하려 만든 문서에서 재발**.

### ★E10 — **규칙이 산문뿐**(교훈 #81/게이트 #66의 **즉시 재발**)
**바로 전날** 등재한 *"결정 규칙은 코드로 고정하고 세계를 전수 열거하라"* 로 되돌아갔다.
미정의: `Q1=예`가 **row 몇 개**인가 · 어느 스트림/컨텍스트여야 하는가 · `greenContextId=NULL`인데
`contextId`가 부모와 다르면 · `graphNodeId=0`인 eager row 혼입 · export 실패 라벨.

### ★E11 — §6이 **하네스층 감사(2단)를 건너뛴다**. `find … kernel_mech` = **`.sbatch`·runner 0건**.
**부수**: §3.1 *"자유 모수 전수 6개"* 는 거짓 — SLURM 할당(가격 8×)·nsys 스위치 6종·부팅 수가
범위(`1–2`)·`R`·종료 방식·아티팩트 형식 ⇒ **최소 6개 더**(rev8 D6가 4연속 지적한 결함의 5회차).

---

## 4. ★★ 재설계 — `Stage 0‴`(2단 분리)

### Stage A0 — **Q1·Q2 전용, 엔진 없음** · ≈**0.02–0.03 GPU-hr** · ★**비exclusive 단일 GPU**
프로세스 1개에서 **4 다리**를 순서 실행, **하나의 `.nsys-rep`**:

| 다리 | 내용 | 역할 |
|---|---|---|
| **L1** | full-GPU eager (green ctx **이전**) | ★양성대조 — row 0이면 `PROBE_INVALID` |
| **L2** | full-GPU **graph** 캡처+replay ×20 | ★대조 — node row가 나오는가(green **무관**) |
| **L3** | green ctx 스트림 **eager** | ★대조 — green 위 커널이 보이는가(graph **무관**) |
| **L4** | **green ctx 스트림 graph 캡처+replay ×20** | ★**본 조건** |

수집: `nsys profile --trace=cuda --sample=none --cpuctxsw=none --gpu-metrics-devices=none
--cuda-graph-trace=node -o <path> python probe.py` → `nsys export --type sqlite`.

**판정(코드로 고정 + 전수 열거 자기검사)**:
```
PROBE_INVALID          := L1 커널 row = 0                        # 게이트 #21, 판정 안 함
NODE_TRACE_UNAVAILABLE := L1 ok ∧ L2 node row(graphNodeId≠0) = 0  # green과 무관한 도구 한계
GREENCTX_INVISIBLE     := L1,L2 ok ∧ L3 row = 0                   # ★P1과 같은 형태의 종결형 사실
Q1  := L4의 graphNodeId≠0 커널 row ≥ (replay당 기대 노드 수 × 20 × 0.9)
Q2a := L4 row의 greenContextId ∉ {NULL,0} ∧ streamId가 L3 스트림과 일치
Q2b := L4 row의 correlationId가 그 replay의 cudaGraphLaunch RUNTIME row와 조인 성공(성공률 보고)
```
라벨: `Q1∧Q2a∧Q2b` → `KSET_CONSTRUCTIBLE(scoped)` / `Q1∧Q2a∧¬Q2b` →
`KSET_NEEDS_TIME_ATTRIBUTION`(E1-(d)) / `¬Q1` → **`PRIMARY_ESTIMAND_UNCONSTRUCTIBLE`**
(★**"트랙 종결" 아님** — E1-(b) 잔존).
병기 필수: nsys/CUDA/driver 버전(★컴퓨트 노드 재확인) · `realized_sm` · ★**`.json` 단일 정본**
(게이트 #56·#57) · 필드명↔값 극성 일치.

### Stage A1 — **K1 + Q3(축소), 엔진 필요** · A0가 `KSET_CONSTRUCTIBLE`일 때만 · ≈**0.05–0.08**
nsys가 감싸는 프로세스 **명시(서버)** + 종료 규칙 · **K1은 양 다리 모두 엔진 telemetry ITL
중앙값**(타임라인 금지 ⇒ Q3 의존 절단) · 구간 라벨 · 중단 규칙(`K1 ≥ 2`면 node-granularity로
안 산다) · **Q3(축소)** = *"graph-launch RUNTIME row가 스텝 경계를 구분하는가"*(★NVTX 없이
경계가 서는지에 대한 **직접 답**).

**공통 규율**: 규칙은 **코드 1파일 + 전수 열거 자기검사** · 하네스층 감사를 **제출 전** 별도로 ·
이력표 각 행에 **검증 방법 병기** · 자유 모수 **재열거(≥12)**.

---

## 5. 부수 발견 (범위 밖 · **미검증**)
nsys 2025.3.2 스키마에 **`CUPTI_ACTIVITY_KIND_BLOCK_TRACE(… correlationId, nodeId, SMId …)`**
가 있다. `%smid`/S3와 표면적으로 관련되나 ★`nsys profile --help`에 **활성화 스위치가 없고**
아키텍처 요구·green ctx 거동 **전부 미확인** ⇒ **후속 조사 항목으로만 기록**, S3 근거로 인용 금지.

---

## 6. 쓰면 안 되는 문장
**승계**: *"Stage 0″가 kernel_mech를 열었다"* · ★*"kernel_mech 트랙을 닫았다"*(Q1·Q2 결과 전) ·
*"rev8 차단이 해소됐다"* · HE0 · gate #13/#16 · switch-cost · C2 인용정지 (a)(b).
★**신설**: *"`TOOL_CANNOT_DEFINE_K_SET`는 트랙 종결 조건이다"*(E1 반증) ·
*"P1이 nsys에도 함의된다"*(§2-1, 단 실측 전 `CONFIRMED` 아님) ·
*"Q2는 이미 답이 나왔으므로 프로브가 불필요하다"*(문서층만 답했다) ·
★*"엔진에 NVTX가 없다"*(E9 — 근거 grep이 틀린 트리) ·
*"nsys는 green-context 커널을 낸다"*(★Stage A0 **전에는 금지**).

## 7. 한 줄
**`NO-GO` — 死因 없음.** 물음은 옳고, ★**Q2의 절반은 이미 도구 문서에 답이 있었으며**,
남은 절반은 ★**엔진 없이 ≈0.02–0.03 GPU-hr에 대조 3겹과 함께** 살 수 있다. 현재 형태는
**틀린 하네스로 산 뒤**(E4) **대조 없이**(E2) **과잉 종결 규칙**(E1)을 발화시키는 구조라,
최악의 경우 **배관 실패가 트랙 종결로 등재된다.** E1–E11은 **전부 GPU 0으로 닫힌다.**
