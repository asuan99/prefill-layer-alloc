# 설계 rev2 — 고-SM 평탄화의 기전 판별 (nsys + ncu, sticky 2×2)

**작성 2026-08-16, experiment-runner. GPU 지출 0(전부 로그인 노드 `ncu`/`nsys`
명령 + 코드 읽기). 미제출 · 사전등록 아님(설계 문서). 성능 판정 0건 · 정책
주장 0건.** 3세션 연속 이월 항목(2026-08-15 rev1 설계 → 감사 "설계 재작성"
판정 → 2026-08-16(1차·2차) 미착수 → 본 rev2). rev1은
`DESIGN_KERNEL_MECH_2026-08-15.md`(커밋 `6b3d6d5`, 미제출).

> **정본 위계**: `PROJECT_STATUS.md` > `reports/paper/` > `reports/CONSENSUS.md`
> (rev34, 2026-08-16) 순. 이 문서는 CONSENSUS §3 항목50–53(rev28, 2026-08-15
> 등재)과 방법론 게이트 #35/#36/#37/#40(PROJECT_STATUS.md "방법론 게이트")을
> 전제한다. `deprecated/`는 근거로 쓰지 않았다.

> ⚠️ **재현 경로 결손 고지**: rev1을 차단한 claims-auditor 감사의 **판정
> 자체를 담은 별도 아티팩트 파일이 저장소에 없다**(`results/kernel_mech/`에는
> rev1 설계 문서 하나뿐). 감사 판정의 유일한 서면 기록은
> `handoff-report/session_handoff_2026-08-15.md` §2.9(도구 사실 4건 + "최대
> 결함")와 §4 실험 트랙("GPU 0에서 지금 닫히는 것 6건 + Stage 0에 반드시 넣을
> 것 6건")뿐이다. 아래 §1의 12건 표는 그 두 절과 CONSENSUS §3 항목50–53의
> 교차 대조로 **재구성**한 것이며, claims-auditor가 직접 "12건, 항목별로
> 이렇게 나눈다"라고 번호를 매긴 원문은 존재하지 않는다 — 감사 판정을 낸
> 코드/체크리스트 자체가 저장소에 보존되지 않은 채 서술로만 남았다는 점에서
> `deconfound-measurement-lessons.md` 항목39의 "공백은 claims-auditor의
> 저장소 밖 독립 재구현이 메웠으나 그 코드는 보존되지 않았다"(2026-08-16,
> C2-R 캠페인)는 패턴과 **동형**이다 — 항목39 자체의 제목은 아니고 그 안의
> 재현-경로-결손 서술과 구조가 같다는 뜻이다.

---

## 0. 왜 rev1이 죽었는가 (1문단 요약)

rev1의 유일한 1차 결정량 `gap_frac`(decode step 안 커널시간 대 간극시간의
비)은 SM 스윕(16→24 vs 44→92)에서 그 비를 비교해 기전을 가르려 했다. 그런데
기본(sticky OFF) 스케줄러에서는 `initialize_stream_groups`가 항상 마지막에
무분할 (0,108) 그룹을 덧붙이고, `adjust_stream_groups`가 "decode busy인데
prefill in-flight 없음"일 때마다 그리로 폴백한다 — 즉 **"decode가 partition
D에서 돌았다"와 "prefill이 동시에 in-flight였다"가 이 기판에서 같은 사건**
이다(CONSENSUS §1-26). rev1은 이 사실을 §0에서 언급조차 안 했고 "prefill
동거 불필요"라고 (거짓으로) 적었다. 결과: `gap_frac(D)`의 SM에 따른 변화가
SM-탄력도 때문인지 D별로 다른 prefill-간섭 빈도 때문인지 원리상 분리
불가능했다 — 감사가 "설계의 유일 결정량이 §1-26이 이미 은퇴시킨 축의 커널
층 판본"이라 판정한 이유다. 동시에 rev1 §3은 **존재하지 않는** 오염(green
context 하 `launch__waves_per_multiprocessor`의 분모 고정)을 피하려다
수제 카운터로 `wave_eff≡1`이라는 항등식을 만들었다(게이트 #9 여덟 번째
재발, 게이트 #36 신설의 근거).

**본 rev2의 두 축**: (A) `wave_eff` 수제 유도를 폐기하고 ncu 자체 메트릭을
1차로 승격 (B) `PDMUX_STICKY_PARTITION`을 편입해 SM-탄력도와 prefill-간섭을
**실험적으로** 분리하는 2×2를 짠다.

---

## 1. 감사 지적 12건 → 해소 표

출처: `session_handoff_2026-08-15.md` §2.9(★1–★4, "최대 결함", "탈출구") +
§4 실험 트랙("(a)–(e)") + CONSENSUS §3 항목50·51·52·53(rev28). "GPU
0에서 지금 닫히는 것"을 G, "Stage 0/설계에 반드시 넣을 것"을 S로 구분했다
(원문의 "6건+6건" 분류를 그대로 따름).

| # | 지적(원문 근거) | 해소 방법 | 상태 |
|---|---|---|---|
| **G1** | `launch__waves_per_multiprocessor`는 green context 하에서 **이미** SM 수로 스케일링됨(ncu 문서, ncu≥2024.3/드라이버≥560 요건 충족) — rev1이 "가장 중요한 함정"이라 부른 오염은 **존재하지 않는다**(CONSENSUS §3 항목52(1)) | §4.1에서 **로그인 노드 재확인**(본 세션, 2026-08-16, ncu 2025.3.1.0) — 원문 그대로 재현. 수제 `wave_eff` 유도(rev1 §3) **전면 폐기**, ncu가 보고하는 값을 그대로 1차 결정량으로 승격(§6 W1) | **해소** |
| **G2** | `nsys --cuda-graph-trace` 기본값은 **`graph`**(CUDA 드라이버≥11.7일 때) → "node activities will not be collected"(CONSENSUS §3 항목52(2)) | §4.2에서 재확인(`nsys profile --help`, 2025.3.2.474): "If 'graph' is selected... node activities will not be collected. If CUDA driver version is 11.7 or higher, default is 'graph'". Stage 0 체크리스트 §7-0 항목3에 `--cuda-graph-trace=node`(전체 옵션은 `node:host-and-device`, CUDA≥12.3) 명시 캡처 추가 | **해소**(설계에 명시 조항으로 편입) |
| **G3** | `--exclusive`는 ncu 요구사항이 아님 — 직렬화 락은 per-device, GPU는 `--gres=gpu:1`로 이미 전유, `hwperf` 카운터 게이트는 A100 노드 전체 기본 feature, 선행 로그에 `ERR_NVGPUCTRPERM` 0건(CONSENSUS §3 항목52(3)) | §4.3에서 재확인(`sinfo`: `gpu[30-33,36-43] A100-80GB_8,hwperf`, 2026-08-16). ★**추가 확인**(본 rev2 신규): `ncu --help`의 `--graph-profiling`은 **기본값이 이미 `node`**("node (default) — Profile individual kernel nodes") — ncu 쪽은 노드 단위가 기본이라 nsys처럼 명시 플래그조차 필요 없다. `--exclusive` 전제를 sbatch 스켈레톤(§7-부록)에서 철회 | **해소** |
| **G4** | 선행 ncu 시도 이미 있음 — 아카이브 8 job, `error code 9` 1,986건 + 메트릭 정규식 실패 1,920건, **전부 full GPU**(green context 하 측정 0건), `_ncu_target.py:73-74` "ncu profiling always runs at full GPU"(CONSENSUS §3 항목52(4)) | §9에서 **원인 진단 완료**(본 rev2 신규 — 로그 원문 대조): `run_ncu_profile.sh`의 주석 자체가 근본 원인을 적어 두고 있었다 — ncu(cuda/13.0.2 모듈, 2025.x)가 **cuda12 venv 심볼릭 링크로 뜬 대상 프로세스**를 프로파일하려다 "Failed to prepare kernel for profiling"/exit 9로 죽음. 이미 `NCU_PYTHON` 오버라이드로 트리에 수정돼 있으나, 그 스크립트는 `workspace/characterization/`(옛 시뮬/마이크로벤치 트랙)이지 이 설계가 쓸 `workspace/engine-port/` 경로가 아니다 — **재사용 금지, 원인만 인용**. "커널 단위 측정 0건"에 운영점·green-ctx 스코프 주석을 유지 | **해소**(원인 특정 + 재발방지책 §9) |
| **G5** | ★**최대 결함**: 유일 1차 결정량 `gap_frac`이 §1-26이 은퇴시킨 축의 커널 층 재현 — "decode가 D SM에서 돌았다"⟺"prefill 동시 in-flight"인 기판에서 어떤 통계도 SM-탄력도와 prefill 간섭을 못 가름. rev1 §0의 "prefill 동거 불필요"는 거짓 | §5에서 **sticky ON/OFF 2×2**로 두 축을 실험적으로 직교화(§0 요약 참조) — 재작성의 핵심. `gap_frac`은 폐기하지 않되 **단독 1차 결정량 지위를 박탈**, 2×2의 한 조건(cell C)에서만 의미를 갖는 것으로 재정의 | **해소**(구조적 재설계) |
| **G6** | `PDMUX_STICKY_PARTITION` 탈출구가 **이미 구현·correctness-gate 통과**돼 있는데(realized 0.084→1.000, CPU 40+12 tests, GPU smoke byte-identical) rev1이 안 씀 | §5에서 편입. 코드 근거 `multiplexing_mixin.py:206-305`(본 rev2가 직접 재확인, 아래 §5.1 인용) | **해소** |
| **S1** | §3(수제 wave 유도) 폐기하고 ncu 메트릭을 primary로 | G1과 동일 조치, §6 W1 | **해소** |
| **S2** | 후보 집합에 (v) 메모리 동거 간섭 · (vi) 클럭/DVFS · (vii) 커널 고정 오버헤드 추가(4→7개) | §6에 7개 후보표로 확장, 각 후보에 결정량·메트릭·서명 매핑 | **해소** |
| **S3** | sticky ON/OFF 2×2 편입으로 §1-26 탈출 | §5 전체 | **해소** |
| **S4** | exclusive 전제 철회 | G3과 동일, sbatch 스켈레톤(§7-부록)에 `--exclusive` 없음 | **해소** |
| **S5** | 비용 재추정(당시 audit 어림 ≈7–12 GPU-hr) | §8에서 Stage별 하향식 재추정: **≈7.4–9.6 GPU-hr**(부팅 25–40초 실측 기반, C2-R 883574/883575 로그 인용) — 감사 어림 범위 안쪽, 상향 편향 없음을 확인 | **해소**(재추정 완료, 실행으로 검증은 미완 — 아래 §10 참조) |
| **S6** | Stage 0에 반드시 넣을 항목들(★본 rev2가 종합·명문화) — (a) green-ctx 하 wave 메트릭 비오염이 **컴퓨트 노드**에서도 재현되는지(로그인 노드 문서 확인만으론 부족, 게이트 #36 문언 그대로) (b) `--cuda-graph-trace=node`가 **sticky-ON 운영점**에서 실제로 커널 노드를 뱉는지 (c) 프로파일러 부착이 sticky ON/OFF의 correctness-gate(byte-identical)를 깨지 않는지 (d) **★신규 발견**(§3): 도달 가능한 배치 격자에서 `waves>1`이 되는 (D,B) 셀이 **하나라도** 있는지 — 없으면 wave-quantization 후보 자체가 이 배치 체제에서 **구성상 검정 불가**임을 미리 선언 | §7 Stage 0에 4개 항목으로 명문화(§7-0). (d)는 §3에서 대수적으로 먼저 점검(게이트 #40) | **해소**(설계에 반영, GPU 실행은 미완 — §10) |

**집계**: 12건 중 **12건 해소**(설계 층에서). 단 "해소"의 의미를 정확히
해야 한다 — G1–G4, S1–S4는 **문서/코드 확인만으로 닫히는 조건**이라 이
rev2로 완전히 닫힌다. G5·G6·S3·S5·S6은 **설계를 바꿔서 조건을 충족**시킨
것이지, **Stage 0 GPU 실행으로 그 설계가 실제로 작동함을 아직 확인하지
않았다** — 이 구분을 흐리지 않는다(§10 "닫지 않는 것" 참조).

---

## 2. 감사 지적 12건과 별개로 — 이 재작성이 새로 발견한 것 1건

이 문서를 쓰는 과정에서 **감사가 지적하지 않은 새 리스크**를 하나 찾았다
(아래 §3). 이는 게이트 #40("결정량 자체가 항등식일 수 있다")을 이 설계의
**신규 1차 결정량**(ncu wave 메트릭)에 선제 적용한 결과다 — G1의 "해소"가
새로운 결정량으로 이어졌으니, 그 새 결정량도 같은 검사를 통과해야 한다.

---

## 3. 결정량의 항등식 점검 (게이트 #40 — 이 문서의 존재 이유)

### 3.1 원 사고(wave_eff≡1)를 대수로 복기

rev1 §3의 수제 유도:
```
waves      = ceil( grid_size / (SM_realized × CTAs_per_SM) )
wave_eff   = grid_size / (waves × SM_realized × CTAs_per_SM)
```
버그는 `SM_realized` 자리에 **디바이스 전체 SM 수(108, 고정)**를 넣은
것이었다(green context가 34/44/92만 주는데도). SGLang이 grid_size를 흔히
108의 배수로 고르므로(풀-GPU 커널 런치 관행), `grid_size mod (108 ×
CTAs_per_SM) ≡ 0`가 거의 항상 성립해 `wave_eff`가 **데이터와 무관하게
1**이 됐다 — 분모 자체가 데이터와 상관없이 그 값을 강제하는 고전적
항등식이다.

### 3.2 ncu 자체 메트릭으로 바꾸면 이 항등식이 사라지는가 — 대체로 예, 단 새 축퇴 조건 하나

G1의 해소로 우리는 이제 `launch__waves_per_multiprocessor`를 ncu가
직접(green-context-aware) 계산한 값으로 읽는다. 분모 버그는 사라진다.
그런데 **다른 극단 사례**가 있는지 대수적으로 점검해야 한다(게이트 #40의
요구 그대로).

`waves = ceil(grid_size / (SM_realized × CTAs_per_SM))`이고 decode step
커널의 `grid_size`는 대체로 **배치 크기에 비례**(시퀀스당 CTA 1개 또는
attention head 수 배수)한다. **극단 사례**: 만약 이 프로젝트의 도달 가능
배치 격자 전체에서

```
grid_size ≤ SM_realized_min × CTAs_per_SM   (SM_realized_min = 16, 이 설계의 최저 스윕점)
```

가 항상 성립하면, `waves ≡ 1`이 SM 스윕 **전 구간**(16→92)에서 성립하고
`wave_eff = grid_size / (SM_realized × CTAs_per_SM)`은 `SM_realized`에
대해 **순수 산술적으로 단조 감소**하는 함수가 된다 — 즉 "wave_eff가
44→92에서 떨어지고 16→24에서는 아니다"라는 rev1 §4의 판정 규칙이 **어떤
GPU 거동과도 무관하게, grid_size와 CTAs_per_SM이 SM에 안 붙어 있다는
사실만으로 강제로 참**이 될 수 있다. 이번엔 계산 버그가 아니라 **이
프로젝트의 배치 체제 자체가 만드는 축퇴**다.

**이 축퇴가 실제로 발생하는지 사전에 값을 대입해 본다**(CONSENSUS §3
항목51, ctx4096 SM92 max decode_bs: T8·M8·Hs8=4, Ha8=12; SM44:
T8·M8=8, Hs8=6, Ha8=16). `grid_size`를 배치 크기의 배수로 근사하면(정확한
상수는 커널마다 다르므로 Stage 0에서 실측 필요), `CTAs_per_SM`이 흔한
값(1–4, 레지스터/공유메모리 점유율에 따라)일 때 `SM_realized_min ×
CTAs_per_SM = 16×1..4 = 16..64`가 되어 T8/M8/Hs8의 `grid_size≈4×k`
(k=커널당 CTA/배치 배수)가 이 범위 **안**에 들어갈 개연성이 낮지 않다 —
즉 **이 위험은 가상이 아니라 현재 배치 격자에서 실제로 일어날 수 있다.**

**대응**(Stage 0 필수 항목, §7-0-4): 프로파일 시작 전에 **적어도 하나의
(D, B) 셀에서 `grid_size > SM_realized × CTAs_per_SM`(즉 `waves ≥ 2`)를
실측으로 확인**한다. 확인 안 되면 — **"wave-quantization 후보 (i)는 이
배치 체제에서 데이터와 무관하게 검정 불가(UNTESTABLE-BY-CONSTRUCTION)"라고
Stage A/B 전에 미리 선언**하고, 후보 (i)를 판정 로직에서 제외한 채
(iii)/(v)/(vi)/(vii)만으로 진행한다. 이 선언 없이 §6의 결정 규칙을
돌리면 R2의 결정량② 재발(게이트 #9 열 번째 재발과 동형)이 된다.

★**부수 관측**(긍정적 상호작용, §5-4에서 재론): sticky+prefill-부재 셀은
CONSENSUS §3 항목51의 "prefill SM 고정 하 decode batch 도달성 폐쇄"
자체를 우회할 가능성이 있다(그 폐쇄는 prefill 서비스율 λ가 상한이 되는
기전인데, sticky+prefill-부재 구간엔 prefill 요청이 아예 없으므로 λ
제약이 발동하지 않는다) — 이는 §3.2의 `waves≥2` 확보 가능성을 **높이는
쪽**으로 작용할 수 있으나, **검정되지 않은 가설**이며 Stage 0가 확인할
항목이지 이 문서가 가정할 항목이 아니다.

### 3.3 다른 신규 결정량들의 항등식 점검 (요약)

| 결정량 | 극단 사례 점검 | 결론 |
|---|---|---|
| `gap_frac`(cell C, legacy 지위로 격하) | `Σkernel_dur`이 nsys의 **device-side 타임스탬프에서 독립 합산**되지 않고 `T_step`(서버 층 ITL)에서 **역산·차감**되면, gap_frac은 이미 알려진 ITL-SM 탄력도의 재진술(항등식)이 된다 | 위험 있음 — **필수 조건**: `Σkernel_dur`은 반드시 nsys 커널 시작/종료 타임스탬프의 직접 합이어야 하고, `T_step`에서 빼서 구하면 안 된다(§7-A에 하네스 assert로 등재) |
| `dram__bytes.sum`(대역폭 후보) | ITL에서 역산한 값이면 CONSENSUS 게이트 #9 일곱 번째 재발(roofline `achieved_BW=bytes/ITL` 항등식)과 동형이 된다 | 이 메트릭은 **하드웨어 카운터 실측**(§4.4에서 존재 확인: `dram__bytes`, `dram__bytes_read`, `dram__bytes_write`는 `Counter` 타입) — ITL과 독립이라 항등식 아님. 단 ncu는 커널을 직렬화하므로 **속도**(대역폭 "포화 여부")는 여전히 못 잰다(§9-해석제한 유지) |
| stall-reason 점유율(`smsp__warp_issue_stalled_long_scoreboard_per_warp_active`) | 이 메트릭이 SM 수에 대해 항상 같은 방향으로 움직이도록 정의상 강제되는 극단이 있는가 — 없음(스톨 사유 분해는 워크로드·점유율에 의존하는 실측치이지 재배열로 나온 항등식이 아님) | 위험 낮음, Stage B에서 그대로 사용 |
| `E1_DECODE_REALIZED`/`prefill_active`(에폭 게이팅용, §5) | 이 값이 sticky ON에서 **항상** 1에 가깝게 강제되는가 — **아니오**, sticky ON의 정의(§5.1)는 "decode busy일 때만" D를 유지하고 prefill in-flight 여부는 건드리지 않으므로, prefill_active는 워크로드가 실제로 주입하는 요청 유무에 **좌우된다**(엔진이 강제하는 값이 아니다) | 항등식 아님. 단 워크로드 설계(§5.2)가 실제로 idle/active 양쪽 구간을 만드는지는 Stage 0에서 실측 확인 필요 |

---

## 4. 도구 타당성 — 로그인 노드 실측 재확인 (게이트 #36)

전부 `module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0` 후
`glogin01`에서 **2026-08-16 본 세션이 직접 실행**(GPU 미사용). 버전은
CONSENSUS §3 항목52가 인용한 것과 **동일**(ncu 2025.3.1.0 / nsys
2025.3.2.474) — 드리프트 없음.

### 4.1 green-context wave 메트릭 (G1)

```
$ ncu --query-metrics-collection launch --chip ga100 | grep -A3 waves_per_multiprocessor
launch__waves_per_multiprocessor   Counter   Number of waves per SM. Partial waves can lead to
                                              tail effects where some SMs become idle while
                                              others still have pending work to complete. When
                                              using green contexts, this metric is scaled with
                                              the number of SMs used by the green context.
```
CONSENSUS §3 항목52(1)이 인용한 문구와 **문자 그대로 일치**. ★이 확인은
**로그인 노드 문서 확인**일 뿐이다 — §3.2에서 지적했듯 실제 green-context
스트림 하에서 이 메트릭이 실측으로도 스케일링되는지는 **컴퓨트 노드에서
아직 확인되지 않았다**(Stage 0 §7-0-1의 존재 이유).

### 4.2 nsys cuda-graph-trace 기본값 (G2)

```
$ nsys profile --help | grep -A10 "cuda-graph-trace"
--cuda-graph-trace=<granularity>[:<launch origin>]
   Possible values for <granularity> are 'graph' or 'node'.
   If 'graph' is selected, CUDA graphs will be traced as a whole and node
   activities will not be collected. ... requires CUDA driver version 11.7
   or higher.
   If 'node' is selected, node activities will be collected, ...
   If CUDA driver version is 11.7 or higher, default is 'graph', otherwise
   default is 'node'.
```
컴퓨트 노드 드라이버는 580.105.08(≥11.7 요건 충족, 실제로는 CUDA 13
드라이버) → **기본값은 `graph`이고 우리는 `node`가 필요**하다. §7-A
하네스는 `--cuda-graph-trace=node`를 **명시**한다(생략 시 Stage 0 항목2가
FAIL해야 정상 — 이게 발화 안 하면 그 자체가 이상 신호).

### 4.3 --exclusive 불필요 + ncu 자체 노드 기본값 (G3)

```
$ sinfo -o "%N %f" | grep hwperf
gpu[30-33,36-43] A100-80GB_8,hwperf
```
A100 파티션 전 노드가 `hwperf`(카운터 권한 게이트) 기본 보유 —
`--exclusive`나 별도 constraint 불필요. ★신규 확인:
```
$ ncu --help | grep -A3 "graph-profiling"
--graph-profiling arg (=node)   CUDA graph profiling mode:
                                  node (default)  (Profile individual kernel nodes)
                                  graph           (Profile entire graphs)
```
ncu는 **기본값이 이미 `node`** — nsys와 달리 플래그 명시조차 불필요(단
§7-A 하네스는 명시적으로 `--graph-profiling node`를 적어 향후 ncu 버전이
기본값을 바꿔도 안전하게 한다, 방어적 관례).

### 4.4 메트릭 존재 확인 — Stage B가 쓰려는 원시 카운터 전부 실재

```
$ ncu --query-metrics-collection launch --chip ga100 | grep -E "^launch__(grid_size|occupancy_limit)"
launch__grid_size                     Counter          Maximum total number of blocks for the kernel launch.
launch__occupancy_limit_barriers      Counter   block   Occupancy limit due to the number of used barriers.
launch__occupancy_limit_blocks        Counter   block   Occupancy limit due to maximum number of blocks managable per SM.
launch__occupancy_limit_registers     Counter   block   Occupancy limit due to register usage.
launch__occupancy_limit_shared_mem    Counter   block   Occupancy limit due to shared memory usage.
launch__occupancy_limit_warps         Counter   block   Occupancy limit due to block size.

$ ncu --query-metrics --chip ga100 | grep -E "^(sm__ctas_launched|smsp__cycles_active|smsp__warp_issue_stalled_long_scoreboard_per_warp_active)\b"
sm__ctas_launched                                     Counter   block   # of CTAs launched
smsp__cycles_active                                   Counter   cycle   # of cycles with at least one warp in flight
smsp__warp_issue_stalled_long_scoreboard_per_warp_active   Ratio           proportion of warps per cycle, waiting for a scoreboard dependency on L1TEX

$ ncu --query-metrics --chip ga100 | grep -E "^dram__bytes\b|^dram__bytes_(read|write)\b"
dram__bytes          Counter   byte   # of bytes accessed in DRAM
dram__bytes_read     Counter   byte   # of bytes read from DRAM
dram__bytes_write    Counter   byte   # of bytes written to DRAM
```
`CTAs_per_SM`은 `sm__ctas_launched`(실측) 또는
`launch__occupancy_limit_*`(이론 상한) 어느 쪽으로도 도출 가능 — Stage 0가
둘 중 어느 쪽이 green-context 하에서 더 안정적인지 판정한다(로그인
노드로는 실측 vs 이론의 차이를 검증 불가, GPU 필요).

### 4.5 클럭/DVFS 후보(vi)용 메트릭 — 미확정, Stage 0로 이관

로그인 노드에서 `gpc__cycles_elapsed`(Counter, cycle)는 확인했으나
"clock frequency"라는 이름의 단일 메트릭은 안 잡힌다 — ncu 관례상
`<counter>.avg.per_second` 롤업으로 주파수를 구하는데, 이 롤업이
green-context 스트림 profiling 세션에서 안정적으로 나오는지는 **로그인
노드 문서만으로 확정 못 한다**(런타임 세션 필요). Stage 0 항목1에
`gpc__cycles_elapsed.avg.per_second`(GPC 클럭 근사)가 포함되는지 함께
확인하도록 편입.

---

## 5. sticky ON/OFF 2×2 — §1-26 탈출 설계

### 5.1 sticky의 실제 동작 (코드 근거, 본 rev2 직접 재확인)

`workspace/engine-port/src/multiplex/multiplexing_mixin.py:206-305`
(주석 인용):

> "With the flag ON, 'decode is busy' alone keeps the target division, so
> decode runs at D SM continuously and prefill SM may sit idle. That is the
> budget-constrained quantity C2 measured directly (prefill pinned, decode
> continuously at D)."
>
> "DEFAULT-OFF GUARANTEE. Every predicate added below short-circuits on
> `self.sticky_partition_enabled` being False, leaving the pre-patch branch
> and the pre-patch value in every case ... CUDA graphs are captured per
> stream-group index ... so holding a division index replays a captured
> graph exactly as the fallback index did -- no eager fallback."

즉:
- **sticky ON**: decode가 busy이기만 하면 prefill in-flight 여부와
  무관하게 D를 유지한다 — prefill이 사라진 구간에도 D SM에서 decode가
  계속 돈다. cudagraph는 그대로 유지(스트림 그룹별 캡처, eager 폴백
  없음 — 운영점 보존).
- **sticky OFF**(기본): "decode busy ∧ prefill 없음"이면 무분할(108)로
  폴백 — 이게 §1-26 confound의 원천.
- 미정의 조합은 `RuntimeError`로 거부(`PDMUX_LA_COORD`,
  `PDMUX_SLO_SCHED`, `PDMUX_FIXED_DECODE_SM_FILE`, non-`fixed`
  `PDMUX_R2_POLICY`) — 이 설계는 `PDMUX_R2_POLICY=fixed` +
  `PDMUX_R2_FIXED_DSM=<D>`만 쓰므로 전부 허용 조합.
- correctness gate: CPU 회귀 40 + sticky 단위 테스트 12
  (`tests/test_sticky_partition.py`) + GPU smoke(job 872800, Ha8 d16)
  고정 프롬프트 6개 greedy 출력 OFF/ON byte-identical. **이 문서가
  새로 도는 것은 아니다** — 기존 gate를 그대로 전제한다.

### 5.2 워크로드 재설계 — C2의 keepalive 홍수 방식은 버린다

C2/E-1이 썼던 keepalive 방식(고빈도 짧은 prefill 폭탄으로 co-residency를
강제)은 이 설계엔 **부적합하다** — 목적이 정반대다(co-residency를
강제하는 게 아니라 **있음/없음을 직접 통제**하는 것). 게다가 keepalive
설계는 이 프로젝트에서 **두 번**(865533의 1794>1792 토큰 초과,
C2-R 스모크 1의 1794>1536) 사고를 냈다(CONSENSUS §3 항목50 addendum,
항목56) — 세 번째로 재사용하지 않는다.

대신:
1. **warm-fill**: 서버 부팅 후 decode 배치를 원하는 크기까지 채운다(긴
   output 요청 다수 발사, 이후 신규 요청 없이 decode만 진행하게 둠).
2. **prefill-ABSENT 구간**: warm-fill 이후 N초 동안 **신규 요청을 아예
   보내지 않는다**(sticky ON이면 D가 유지된 채 decode-only로 돎).
3. **prefill-PRESENT 구간**: 그 뒤 M초 동안 짧은 prefill 요청을 낮은
   빈도로 주기 주입(co-residency를 만들되 keepalive처럼 포화시키지
   않음 — 토큰 길이는 여유 있게, 예: ctx cap의 1/4 이하로 assert).
4. 한 트레이스 안에서 (2)→(3) 전이를 캡처하면 nsys/ncu 세션 하나로
   cell A와 cell B를 **동시에** 얻는다(별도 부팅 불필요, 아래 §5.3
   표의 A/B는 같은 부팅의 다른 시간 구간).

이 워크로드는 keepalive보다 **단순**하고(토큰 예산 계산이 필요 없다 —
prefill 요청 자체가 sparse), 이전 keepalive 실패 두 건이 원리적으로
재발 못 한다(overflow할 고빈도 루프가 없다).

### 5.3 2×2 셀 정의

| 셀 | sticky | prefill 주입 | 실제 실현되는 것 | 무엇을 분리하는가 |
|---|---|---|---|---|
| **A** | ON | ABSENT | D를 연속 유지, prefill 완전 부재 | **순수 decode-SM 탄력도** — 이 프로젝트 최초로 prefill 간섭 없이 잰다 |
| **B** | ON | PRESENT(저빈도) | D를 연속 유지, prefill이 간헐 co-resident | decode-SM 탄력도 **+** 희석 없는 실제 prefill 간섭 |
| **C** | OFF | PRESENT | D는 prefill in-flight 구간에서만 실현(폴백 기본) | **레거시/자연 상태** — 기존 C2·gap_frac 이력과의 외부 타당성 앵커 |
| **D** | OFF | ABSENT | D가 **한 번도** 실현 안 됨(decode busy ∧ prefill 없음 → 항상 무분할 108로 폴백) | SM 스윕에 넣지 않는 **단일 참조점**(SM=108, decode-only 상한) — sanity check용 |

**비교 경로**(이게 §1-26을 실제로 여는 지점):
- **A vs B (같은 D 고정)**: SM 수를 통제한 채 prefill 간섭 유무만 바뀜 →
  **prefill 간섭이 커널 서명에 미치는 순수 기여**를 처음으로 분리한다.
- **A를 D=16,24,44,92에 걸쳐**: prefill 간섭이 전혀 없는 상태에서 SM만
  바뀜 → **순수 SM-탄력도**(후보 (i)/(iii)의 진짜 영역). rev1의
  `gap_frac`이 원리상 못 하던 일.
- **C vs B (같은 D, 같은 prefill 주입 강도로 맞춤)**: sticky **메커니즘
  자체**가 서명을 바꾸는지 확인하는 **타당성 대조**. 다르면 sticky가
  측정 인공물을 넣는다는 뜻이고(후보 (iv) 계열 위험), 그 경우 A/B의
  결과 해석에 그 아티팩트를 병기해야 한다 — **falsifiable 게이트**로
  §7-0-3에 등재.
- **D**: 스윕이 아니라 "간섭도 SM 제약도 없는" 바깥쪽 기준점 1개.

### 5.4 §3.2의 대응 관계

§5.2 워크로드에서 셀 A는 prefill이 아예 없으므로 CONSENSUS §3 항목51의
"prefill SM 고정 → λ 상한 → decode batch 도달 불가" 폐쇄 기전이 **발동
조건 자체가 없다**(그 폐쇄는 prefill 서비스율이 상한을 만드는 것인데,
prefill이 이 구간엔 없다). 따라서 셀 A는 warm-fill 단계에서 결정되는
배치 크기까지 자유롭게 채울 수 있어(§3.2가 요구하는 `waves≥2` 확보에
유리), item 51의 폐쇄를 **부수적으로 우회할 가능성**이 있다 — 단,
**가설**이며 Stage 0에서 실측 확인이 먼저다(§7-0-4).

---

## 6. 후보 집합 확장 (4 → 7개)

| # | 후보 | 결정량 | ncu/nsys 메트릭 | 서명 |
|---|---|---|---|---|
| (i) | wave quantization | `waves`, `wave_eff`(§3.2 축퇴조건 통과 시만 유효) | `launch__waves_per_multiprocessor`(직접, 수제 유도 금지) | 계단형, SM↑ 시 꼬리 wave 점유율 악화 |
| (ii) | 층 직렬 사슬(간극, 커널 밖) | `gap_frac`(cell C만) | nsys 타임라인(node) | 커널은 줄지만 간극 불변 |
| (iii) | 점유율/latency | `occupancy`, stall 분해 | `sm__ctas_launched`/`launch__occupancy_limit_*`, `smsp__warp_issue_stalled_long_scoreboard_per_warp_active` | sub-linear, long_scoreboard 지배 |
| (iv) | cudagraph 직렬화(간극, 커널 밖) | 위 (ii)와 ncu로는 미분리(§0 원 설계 계승) | cudagraph OFF 대조(간극 기전 한정, 성능 비교 금지) | 간극이 SM에 불변 |
| **(v)** ★신규 | 메모리 동거 간섭(A vs B가 직접 조준) | ITL/커널시간 delta(A−B, 같은 D) | 시간 비교만으로 충분(전용 ncu 메트릭 불요), 보조로 `dram__bytes.sum` delta | B가 A보다 유의하게 느림·바이트 큼 |
| **(vi)** ★신규 | 클럭/DVFS | GPC 클럭 근사 | `gpc__cycles_elapsed.avg.per_second`(§4.5, Stage 0 확인 대상) | SM↑ 시 클럭 하락(전력/열 상한) — cell A(순수 decode) vs cell C(자연 상태) 비교로 격리 |
| **(vii)** ★신규 | 커널 고정 launch 오버헤드 | 커널당 평균 duration의 SM-무관 성분 | nsys node 타임라인의 커널 dispatch~exec 간격 | 오버헤드가 SM/워크로드 양쪽에 불변 — (iv)와 구분 필요(iv는 프리필과의 직렬화, vii는 단일 스트림 자체의 launch 비용) |

(i)/(iii)/(v)/(vi)/(vii)는 **셀 A**(순수 decode)에서 판별력이 가장
크다 — prefill 간섭이 없으므로 커널 내부 신호가 오염되지 않는다.
(ii)/(iv)는 여전히 **간극** 문제라 ncu로는 못 가르며, cell B/C의 nsys
비교(§5.3 A vs B)가 그 역할을 대신한다.

---

## 7. 3단 설계 (재작성)

### Stage 0 — 타당성 프로브 (GPU 목표 ≈30–40분, exclusive 불필요)

1. **green-ctx wave 비오염 컴퓨트 노드 재확인**(§3.2, §4.1): 임의의 작은
   green-context 스트림(예: (92,16))에 대해 `ncu --graph-profiling node`
   로 커널 1개를 잡아 `launch__waves_per_multiprocessor`와
   `launch__grid_size`를 동시에 읽고, `waves × SM_realized(16) ×
   CTAs_per_SM`이 `grid_size`에 근접하는지 **손으로 대조**(로그인 노드
   문서 확인을 런타임으로 승격).
2. **nsys `--cuda-graph-trace=node`가 sticky-ON 운영점에서 커널 노드를
   뱉는지**(§4.2): 30초 캡처, 커널 이름·시작/종료 타임스탬프 유무 확인.
3. **sticky ON/OFF가 프로파일러 부착 상태에서도 correctness를 유지하는지**
   (§5.3 falsifiable 게이트): `--nightly`가 아니라 §5.1의 기존
   correctness gate(고정 프롬프트 greedy 출력)를 **nsys/ncu 부착 하에**
   재실행, OFF/ON byte-identical 재확인. 프로파일러가 스케줄러 타이밍을
   흔들어 correctness를 깨면 그 자체가 (iv)/(vii) 후보의 강한 증거이자
   설계 정지 신호.
4. ★**`waves≥2` 실현 가능성 확인**(§3.2, 게이트 #40 대응): 워크로드
   §5.2 워밍필을 최대 배치까지 채운 뒤, D=16(가장 SM 작은 셀, `waves≥2`
   가 가장 나오기 쉬운 지점)에서 decode step 커널의 `launch__grid_size`
   대 `SM_realized×CTAs_per_SM`을 비교. **`waves≥2`가 D=16에서도 안
   나오면 후보 (i)는 UNTESTABLE-BY-CONSTRUCTION으로 선언하고 Stage
   A/B의 결정 규칙에서 제외**(§3.2 대응 조치 그대로 집행).

**정지 규칙**: 1·2가 실패하면 §0.9(원 설계 계승, 아래 §8-정지규칙)와
동일하게 전 설계 재고. 3이 실패하면(프로파일러가 correctness를 깬다)
**Stage A/B 전면 중단** — 측정 자체가 운영점을 벗어나므로 어떤 결과도
"cudagraph-ON 운영점" 주장을 할 수 없다(CLAUDE.md 운영점 원칙 위반).
4는 실패해도 설계 중단 아님 — 후보 (i)만 제외하고 진행(측정 실패를
게이트 실패로 라벨링 금지, 게이트 #21).

### Stage A — nsys 타임라인, 2×2 (GPU ≈2–3hr, exclusive 불필요)

- arm: Ha8(주) + T8(대조, hybrid 고유성 판별)
- SM(D): 16·24·44·92(레버 생존 구간 + 사멸 구간 둘 다)
- sticky × 워크로드: **ON 부팅 1개**로 cell A(prefill-absent 구간)+cell
  B(prefill-present 구간) 동시 캡처, **OFF 부팅 1개**로 cell C.
  → (arm × D × sticky-보팅) = 2 × 4 × 2 = **16 부팅**.
- 결정량: `T_step`, `gap_frac`(cell C 한정 유효, §3.3), `wave_eff`(ncu
  아님 — Stage A는 nsys만이므로 wave는 Stage B), A−B/A−C 시간차(후보
  (v) 조준).

### Stage B — ncu 커널 내부 (GPU ≈2.5–4hr, exclusive 불필요, 큐 대기 김)

- Stage A가 `waves≥2`를 확인한 SM 점(§7-0-4 조건부) + `KERNEL_DOMINATED`
  로 판정된 구간만 대상 — Stage A `GAP_DOMINATED`면 그 (arm,D)는 Stage
  B 우선순위 낮춤(원 설계 §1 판별표 로직 계승).
- **cell A와 cell B만**(sticky ON 부팅 2×4=8개 재사용, OFF는 재프로파일
  안 함 — Stage A의 C vs B 비교로 이미 sticky 아티팩트 여부는 답함).
- decode step 5–10개, 메트릭: `launch__waves_per_multiprocessor`,
  `launch__grid_size`, `sm__ctas_launched`,
  `smsp__warp_issue_stalled_long_scoreboard_per_warp_active`,
  `dram__bytes.sum`, `gpc__cycles_elapsed.avg.per_second`(§4.5,
  Stage 0에서 유효성 확인된 경우만).

---

## 8. 비용 · 큐 · 게이트 #26

| 단계 | 부팅 수 | GPU 어림 | exclusive |
|---|---|---|---|
| Stage 0 | 1–2(재사용) | ≈0.5–0.7hr | 불필요 |
| Stage A | 16 | ≈2.0–2.7hr(부팅 25–40초×16 + 30–60초 캡처×16 + 분석 준비) | 불필요 |
| Stage B | 8(Stage A 부팅 재사용 가능성 있으나 보수적으로 별도 계상) | ≈2.5–4.0hr(ncu 직렬화·다중 패스 오버헤드는 순수 nsys보다 큼) | 불필요(G3) |
| **합계** | — | **≈5.0–7.4hr**(부팅 재사용 최대 가정) **/ ≈7.4–9.6hr**(전부 별도 부팅 가정) | — |

부팅 시간 기준: C2-R 캠페인(jobs 883574/883575, 2026-08-16) 실측
**25–36초/부팅**(`session_handoff_2026-08-16.md` §8.1, ≈1 GPU-hr/24
부팅). 감사가 어림한 "≈7–12 GPU-hr"(S5) 범위 **안쪽**이며 상향 편향은
없다 — 단 이 재추정 자체는 **Stage 0 실측으로 아직 검증되지 않았다**
(캡처·분석 준비 시간이 부팅보다 클 수 있어 하방보다는 상방 리스크가 큼).

**게이트 #26(대형 캠페인 제출 전 배관 스모크) 발동 여부**: 이 캠페인은
(a) 신규 코드(§5.2 워크로드 클라이언트, nsys/ncu 래퍼 하네스)를 쓰고
(b) 총 GPU가 ≈5–10hr로 게이트 #26이 예방한 6.40 GPU-hr 오판(job
877107/877109) 규모와 같은 자릿수다 → **발동. Stage A/B 본 캠페인
제출 전 파이프라인 스모크(≈0.1–0.2 GPU-hr: 부팅 1회 + 워크로드 30초 +
nsys/ncu 각 1회 최소 캡처 + 파서 실행까지 end-to-end)가 필수**다. 이
문서는 그 스모크조차 아직 설계만 하고 실행하지 않았다(§10).

---

## 9. error code 9 재발 방지책 — 원인 진단

아카이브 로그(`logs/archived/ncu_720116.log`, `ncu_726120.err` 등,
2026-05, `workspace/characterization/` 트랙) 원문:

```
==ERROR== Failed to prepare kernel for profiling
==ERROR== Unknown Error on device 0.
==ERROR== Failed to profile "ampere_bf16_s16816gemm_bf16_2..." in process ...
==ERROR== The application returned an error code (9).
```

**근본 원인**(같은 저장소 `workspace/characterization/slurm/
run_ncu_profile.sh`가 나중에 **이미 스스로 진단해 코멘트로 남겨 뒀다**,
본 rev2가 재확인):

> "ncu 2025.x (from cuda/13.0.2) cannot profile a CUDA 12 target process" —
> venv의 `bin/activate`가 `pytorch_2.9.1_cuda12`로 심볼릭 링크된 python을
> 가리키는데, ncu는 `cuda/13.0.2` 모듈에서 로드돼 **툴킷 버전이 안 맞는
> 프로세스**를 프로파일하려다 죽는다. `ERR_NVGPUCTRPERM`(권한)이 **아니다**
> — CONSENSUS §3 항목52(3)의 "0건" 확인과 모순되지 않는다(다른 실패
> 종류).

`error code 9` 1,986건은 **권한 문제가 아니라 툴킷 버전 불일치**였고,
그 스크립트는 나중에 `NCU_PYTHON` 환경변수로 프로파일 대상 python
경로를 venv 활성화 **이전에** 캡처하는 방식으로 고쳐졌다(같은 파일,
`workspace/characterization/` 트랙 — 이 설계가 재사용할 경로는 아니다,
§1 G4 참조).

**이 설계에서의 재발 방지책**:
1. **모듈 로드를 단일화**한다 — CLAUDE.md/이 프로젝트 표준
   (`conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0`)만 쓰고,
   `workspace/engine-port`의 venv(`sglang_engine_venv`)가 어떤 CUDA
   빌드에 링크됐는지 **Stage 0 항목0**(§7-0에 추가 필요)으로
   `python -c "import torch; print(torch.version.cuda)"`와
   `ncu --version`을 **같은 로그에 나란히 찍어** 사전 확인한다.
2. ncu가 별도 서브프로세스를 스폰하는 구조가 아니라(원 사고는
   `ncu_runner.py`가 **자신이 spawn한** 짧은 파이썬 타겟 프로세스를
   프로파일하는 구조 — `_ncu_target.py`) 이 설계는 **이미 떠 있는
   장수 SGLang 서버 프로세스에 ncu를 attach**하는 구조를 쓴다(`ncu
   --target-processes application-only` 류가 아니라, 서버가 이미
   구동 중인 PID에 `--pid` 부착 또는 서버 자체를 `ncu <server-launch-cmd>`
   로 감싸 부팅). **서버와 ncu가 같은 모듈 로드 시퀀스(같은 셸)에서
   나온다는 것을 보장**하면 이 클래스의 실패는 구조적으로 재발하지
   않는다 — 원 사고는 "관측 스크립트가 자기가 통제 못 하는 venv를
   활성화"였는데, 이 설계엔 그런 2차 활성화 지점이 없다.
3. Stage 0 정지 규칙에 **"버전 문자열 불일치 시 즉시 중단"**을 명시
   조항으로 추가(§7-0에 항목0으로 편입 권고, 아래 §10에 미반영 사항으로
   정직하게 기록).

---

## 10. ★닫지 않는 것 (정직하게)

1. **Stage 0 GPU 실행 자체가 미완**이다 — 이 문서는 설계이지 결과가
   아니다. §7-0-4(`waves≥2` 확인)와 §9의 버전-매치 확인은 **아직 한
   번도 컴퓨트 노드에서 실행되지 않았다.**
2. **§8의 비용 재추정은 검증되지 않았다** — 부팅 시간은 C2-R에서
   가져왔지만 캡처·분석 준비 시간(nsys/ncu 특유의 오버헤드)은 이
   프로젝트에서 green-context 하 실측 전례가 없다(선행 8 job은 전부
   full GPU, §1 G4).
3. **게이트 #26 파이프라인 스모크가 아직 설계만 되고 실행되지 않았다**
   (§8) — 이것이 실행되기 전에는 Stage A/B 본 캠페인을 제출할 수 없다.
4. **§9-3의 "버전 문자열 불일치 시 중단" 조항은 Stage 0 체크리스트
   본문(§7-0)에 아직 번호가 매겨져 편입되지 않았다** — §9에서만 언급.
   다음 세션이 실제로 사전등록을 쓸 때 §7-0에 항목0으로 넣어야 한다.
5. **§5.3의 "C vs B 타당성 대조"는 설계상 존재할 뿐 판정 기준(임계값)이
   없다** — "다르면 아티팩트"라고만 적었지 얼마나 달라야 유의한지
   정하지 않았다. 사전등록 단계에서 반드시 채워야 할 구멍이다.
6. **§6 후보 (v)/(vi)/(vii)의 판정 규칙이 (i)/(ii)/(iii)/(iv)만큼
   정밀하지 않다** — 신규 후보라 메트릭은 확인했지만 "이 값이 나오면
   (v)로 판정한다"는 정량 문턱을 아직 정하지 않았다. Stage A 원자료를
   본 뒤 사전등록에서 확정해야 한다(문턱을 먼저 정하고 데이터를 보면
   게이트 #35 재발 — 값이 나온 뒤 문턱을 소급 조정하지 않도록 조심).
7. **Gate 2 본 질문("PD 분리 자체" 귀속)·비-SM-split lever·HE0/정책
   주장에는 이 설계가 여전히 전진 0**이다(원 설계 §6 "사지 않는다" 계승).
8. **claims-auditor 재감사를 아직 받지 않았다.** 다음 단계(§11)에서
   요청 형식을 명시한다.

---

## 11. 다음 단계 — 감사 의뢰 형식 (게이트 #37 적용)

이 설계 트랙은 이미 **1회 차단**됐다(rev1). 게이트 #37("적대 감사에는
합격 기준과 단일 판정 질문을 함께 줘라 — 안 그러면 범위 없는 검토는
항상 NO-GO를 낸다")을 이번엔 **처음부터** 적용해 재차단을 피한다.

**단일 판정 질문(제안)**: *"이 설계(sticky 2×2 + 확장 후보 7개 +
Stage 0 게이트 4개)가 §1-26 confound(decode-SM 탄력도 vs prefill 간섭
비식별)를 실험적으로 분리하는가, 그리고 그 분리가 §3의 항등식 위험
없이 성립하는가?"**

**합격 기준(제안)**:
1. §5.3의 A/B/C/D 셀 정의가 §1-26이 묘사한 confound를 실제로
   직교화하는지(코드 근거 §5.1과 논리적으로 정합한지) — 예/아니오.
2. §3의 축퇴 점검(waves≥2 사전 확인 요구)이 게이트 #40 요구사항을
   충족하는지 — 예/아니오.
3. §9의 error-code-9 원인 진단이 근거(로그 원문 인용)를 갖는지 —
   예/아니오.

**caveat 라우팅**: 위 3항목 범위 밖의 발견(예: §10에 이미 자백한 6개
결손, 비용 추정의 정밀도, 후보 (v)/(vi)/(vii) 문턱 부재)은 **NO-GO
사유가 아니라 caveat로 접수**하도록 명시 요청한다 — C2-R rev1→rev2
전례(게이트 #37) 그대로.

이 감사가 GO를 내면 다음 세션은 **Stage 0 사전등록**(GPU ≈30–40분)을
쓰고, 그 전에 **게이트 #26 파이프라인 스모크**(≈0.1–0.2 GPU-hr)를
먼저 통과시킨다(§8).

---

## 부록 — Stage 0 sbatch 스켈레톤 (제출 안 함, `--test-only`만)

아래는 **골격**이다 — 실제 워크로드 클라이언트(§5.2)와 ncu/nsys 래퍼는
아직 코드로 존재하지 않는다(설계 문서 산출물 요구사항이 여기까지다).
`--comment` 배치와 파티션/자원 규약만 미리 검증한다.

```bash
#!/bin/bash
#SBATCH --job-name=kmech-stage0
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech/%x_%j.out
#SBATCH --error=/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/kernel_mech/%x_%j.err
# STUB — Stage 0 타당성 프로브 골격. §7-0의 4개 항목을 채워 넣기 전까지
# 실행 대상이 아니다(서버 boot·워크로드 클라이언트·ncu/nsys 래퍼 미구현).
# 이 파일은 sbatch 헤더 규약(§CLAUDE.md --comment 게이트) 검증 전용.
set -euo pipefail
echo "STUB — not runnable yet. See DESIGN_KERNEL_MECH_REV2_2026-08-16.md §7-0."
exit 1
```

**검증 완료(2026-08-16, experiment-runner, 이 문서와 별도로 스크래치
사본에서 실행 — `results/kernel_mech/`에는 두지 않았다, GPU 미사용)**:
`check_sbatch_comment.py` **1/1 conformant**, `sbatch --test-only` **최초
시도(`--cpus-per-task=16`)는 거부**됨(`requested CPU cores per node (16)
exceed the allowed limit (8) for 1 GPU(s)` — `amd_a100nv_8`는 GPU 1개당
CPU 8개 상한, 이 프로젝트의 일반 sbatch 관례 문구 "`--cpus-per-task=16`"는
**단일-GPU job에는 안 맞는다**, 기존 `s8_c2r.sbatch`가 8을 쓰는 이유와
일치). `--cpus-per-task=8`로 정정 후 재시도: `sbatch: Job 884186 to start
at 2026-08-19T04:39:09 using 8 processors on nodes gpu39 in partition
amd_a100nv_8`(exit 0, 스케줄러가 실제로 수락함을 확인, **제출은 안
함**). 위 코드 블록은 이 정정을 반영한 최종본이다. 이 파일은
`results/kernel_mech/`에 실제로 두지 않는다(스텁을 실행 가능한 것처럼
저장소에 남기면 다음 세션이 오인 제출할 위험 — 본문에만 인용).
