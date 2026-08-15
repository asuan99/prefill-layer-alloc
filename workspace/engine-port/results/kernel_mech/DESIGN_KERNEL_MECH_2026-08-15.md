# 설계 — 고-SM 평탄화의 기전 판별 (nsys + ncu 2단)

**작성 2026-08-15. 미제출 · 사전등록 아님(설계 문서).** 감사 전이며, **이 문서를 사전등록으로
인용하지 않는다**(2026-08-14 설계문서 2건이 같은 조항을 스스로 걸었던 선례).
성격: **기전 측정. 정책 주장 0건, 성능 판정 0건.**

---

## 0. 무엇을 왜 재는가

정본 확정: decode ITL의 SM 탄력도가 **저-SM 0.77–0.90(16→24) vs 고-SM 0.09–0.35(44→92)**로
4–9× 붕괴한다. 정책이 머물러야 하는 decode-heavy 영역이 하필 그 평탄 구간이다.
**평탄화의 원인은 미식별**이고(후보 4개, **커널 단위 측정 0건**), 그래서 정본은
"memory-bound"·"HBM 포화" 서술을 금지하고 있다.

E-1(rev1–rev3)은 이 질문을 **서빙 층 탄력도**로 치려다 세 번 다 막혔다 — 마지막엔
**목표 동작점(양 arm 공통 B=9 @ SM92)이 이 기판에서 구조적으로 도달 불가**임이 실증됐다
(prefill 16 SM 고정 ⇒ λ 상한 ⇒ Little's law로 B가 안 오름; T8·M8·Hs8이 d92에서 정확히 4).

**커널 층은 그 제약을 받지 않는다.** 공통 B도, arm 매칭도, prefill 동거도 필요 없다.
필요한 것은 "정상상태 decode가 돌고 있는 서버"뿐이다.

### ★ 도구 층 확인 (2026-08-15, GPU 0)

| | 결과 |
|---|---|
| `ncu` | **2025.3.1.0**, `--graph-profiling {node,graph}` **지원** ⇒ cudagraph 재생 커널 프로파일 가능 |
| `nsys` | **2025.3.2.474** 사용 가능 |

⇒ **운영점(cudagraph-ON)에서 잰다.** `PER_LAYER_TYPE_SM_RESPONSE_DESIGN`이 물린
"`replay()`가 Python forward를 호출하지 않아 계측이 안 돈다"는 제약은 **프로파일러 층에는
적용되지 않는다**(드라이버/하드웨어 카운터이지 Python 훅이 아니다). 단 §2.0에서 **실측 확인**한다.

---

## 1. 판별표 — 후보 4개가 서로 다른 서명을 낸다

| 후보 | 시간이 커널 **안**인가 | SM↑ 시 커널 시간 | 서명 |
|---|---|---|---|
| **(i) wave quantization** | 예 | **계단형**(부드럽지 않음) | `waves/SM`이 정수 경계를 넘나듦; 꼬리 wave 점유율이 SM↑에 따라 **악화** |
| **(iii) 점유율/latency** | 예 | 부드럽지만 **sub-linear** | achieved warps가 이론치 아래 포화; stall이 `long_scoreboard`(메모리 지연) 지배 |
| **(ii) 층 직렬 사슬** | **아니오(간극)** | — | 커널은 줄지만 **간극이 안 줄어** 간극 비중이 SM↑에 따라 상승 |
| **(iv) cudagraph 직렬화** | **아니오(간극)** | — | 간극이 그래프 노드 launch 오버헤드 ⇒ **SM에 불변** |

★ **핵심 구조**: (ii)와 (iv)는 둘 다 "간극"이라 **ncu로는 갈리지 않는다** — ncu는 커널을
직렬화하므로 간극을 못 본다. 반대로 (i)과 (iii)은 **nsys로는 갈리지 않는다** — 둘 다 커널
안이다. **두 도구가 서로 다른 분기를 담당한다.**

⇒ **1차 분기는 nsys가 낸다**: 시간이 커널 안인가 밖인가.
그 답이 "안"이면 ncu가 (i) vs (iii)을 가르고, "밖"이면 ncu는 그 분기에 불필요하다.

---

## 2. 3단 설계 — 대기시간 제약에 맞춤

`ncu`는 노드 전유(`--exclusive`)가 필요해 큐 대기가 길다. 그래서 **대기 중에 돌 수 있는
단계를 앞에 배치**한다.

### Stage 0 — 타당성 프로브 (GPU ≈15분, **exclusive 불필요**)

**이것이 통과하지 못하면 나머지 전부 무의미하다.** 사전등록 전 필수.

1. `nsys profile`이 pdmux green-context 스트림의 **cudagraph 재생 커널을 타임라인에
   기록하는가** — 커널 이름·시작/종료 타임스탬프가 실제로 나오는지.
2. `ncu --graph-profiling node`가 같은 커널을 **resolve 하는가** — 커널 하나라도
   메트릭이 나오는지(전체 스윕 아님, 존재 확인만).
3. green context 하에서 카운터가 **오염되는 범위** 확인(§3).
4. 두 도구가 서버를 죽이지 않는지(부팅·정상상태 유지).

**정지 규칙**: 1이 실패하면 전 설계 폐기(다른 접근 필요). 2만 실패하면 **Stage A만** 진행하고
(i)/(iii) 분기는 열어 둔다 — **"측정 실패"로 기록하지 "게이트 실패"로 라벨하지 않는다**(#21).

### Stage A — nsys 타임라인 (GPU ≈1hr, **exclusive 불필요, 대기 짧음**)

- arm: **Ha8**(주, 프로젝트 대상) + **T8**(대조: 이게 hybrid 고유인가?)
- SM: **16 · 24 · 44 · 92** — ★**레버가 살아 있는 구간(16→24)과 죽은 구간(44→92)을 둘 다**
  포함한다. 평탄화를 설명하는 기전은 **그 두 구간에서 스스로 달라야** 한다. 한 점만 재면
  아무것도 판별하지 못한다.
- 워크로드: C2 셋업 그대로(prefill 16 SM 고정, 정상상태 decode), 30초 캡처.
- **결정량**: decode step 하나 안에서
  `T_step = Σ kernel_dur + Σ gap`, 그리고 **`gap_frac = Σ gap / T_step`**.

### Stage B — ncu 커널 내부 (GPU ≈1–2hr, **exclusive 필요**)

Stage A가 "시간이 커널 안"이라고 답할 때만 결정적이다(아니어도 절대 대역폭 확인 목적으로 값어치는 있다).

- 같은 (arm, SM) 좌표. decode step **5–10개**만 프로파일한다(처리량이 목적이 아니다).
- 메트릭(§3): waves · occupancy · stall reason · **DRAM 실측 트래픽**.

---

## 3. 메트릭과 green context 오염 — ★가장 중요한 기술적 함정

★★ **`launch__waves_per_multiprocessor`를 그대로 쓰면 안 된다.** 이 메트릭의 분모는
**디바이스 SM 수(108)**인데 green context는 34/44/92만 준다. ⇒ **우리가 가장 알고 싶은
wave quantization 지표가 정확히 가장 오염되는 지표다.**

**대응**: 원시 카운터로 **직접 계산**한다.
```
waves      = ceil( grid_size / (SM_realized × CTAs_per_SM) )
tail_frac  = (grid_size mod (SM_realized × CTAs_per_SM)) / (SM_realized × CTAs_per_SM)
wave_eff   = grid_size / (waves × SM_realized × CTAs_per_SM)
```
- `grid_size` ← `launch__grid_size` (오염 없음)
- `CTAs_per_SM` ← `launch__occupancy_limit_*` 또는 `sm__ctas_launched` / 활성 SM 수
- `SM_realized` ← ★**E-3이 확립**: 드라이버 보고 realized SM 수가 요청값과 7/7 정확 일치
  (`results/smsplit_realized/`). 이 실험이 E-3에 의존하는 지점이며, **드라이버 보고 층**임을
  명시한다(하드웨어 실행 층 아님).

**% of peak 계열 전부 동일 오염** ⇒ `sm__throughput.avg.pct_of_peak_*`,
`sm__warps_active.avg.pct_of_peak_*`는 **보고만 하고 판정에 쓰지 않는다.**
판정은 **원시 카운터**(cycles, warps active, ctas launched, dram bytes)로 한다.

### ★ ncu의 DRAM 카운터가 항등식을 깬다

정본에 등재된 게이트 #9 일곱 번째 재발: roofline의 `achieved_BW = bytes_step/ITL`에서
`bytes_step`이 config 계산량이라 **`|ε_BW| ≡ |ε_ITL|`이 정의상 성립**했다 — roofline이
독립 확증을 준 적이 없다.

**ncu의 `dram__bytes.sum`은 실측이다.** ITL에서 역산한 값이 아니라 하드웨어 카운터다.
⇒ **"decode가 대역폭 포화인가"를 처음으로 독립 측정**한다. 이것만으로도 정본의
"achieved BW 45.4–57.7%(계산치)" 서술이 **측정치로 대체**된다.

---

## 4. 결정 규칙 초안 (사전등록 시 확정)

**Stage A (1차 분기)**: `gap_frac`을 SM 16→24와 44→92에서 비교.

| 관측 | 판정 |
|---|---|
| `gap_frac`이 44→92에서 유의하게 상승 ∧ 16→24에서는 아님 | **`GAP_DOMINATED`** ⇒ (ii)/(iv) 쪽. Stage B는 (i)/(iii) 판별엔 불필요 |
| `gap_frac`이 두 구간 모두 낮고 불변 | **`KERNEL_DOMINATED`** ⇒ (i)/(iii). **Stage B가 결정적** |
| 그 외 | `UNDETERMINED` — 결과이며 통과 아님 |

**(ii) vs (iv) 분리**(Stage A에서 `GAP_DOMINATED`일 때): cudagraph **OFF** 대조를 1셀 추가.
간극이 eager에서도 남으면 (ii), 사라지면 (iv). ⚠️ 이 대조는 **간극 기전에 한정**되며
**성능 비교로 쓰지 않는다**(cudagraph-OFF는 비운영점 — confound #8).

**Stage B (2차 분기)**: SM 16→24 vs 44→92에서
- `wave_eff`가 44→92에서 떨어지고 16→24에서는 아니면 → **(i) wave quantization**
- achieved warps/SM이 포화하고 stall이 `long_scoreboard` 지배면 → **(iii)**
- `dram__throughput`이 사양에 접근하면 → **대역폭 포화**(정본이 현재 금지하는 서술이 측정으로 해금됨)
- 둘 이상이 동시에 → `MULTI_CAUSE`로 보고하고 **하나를 고르지 않는다**

---

## 5. 자기무력화 (게이트 #21)

1. Stage 0의 nsys 항목 실패 → 전 설계 폐기(다른 접근).
2. Stage 0의 ncu 항목만 실패 → Stage A만, (i)/(iii) 분기 **미해결로 명시**.
3. green context 하에서 §3의 원시 카운터가 나오지 않으면 → `UNSCOREABLE (COUNTERS
   UNAVAILABLE UNDER GREEN CONTEXT)`, 대체 지표를 사후에 만들지 않는다.
4. 프로파일러가 정상상태를 깨면(큐 성장·런길이 표류) 그 셀 폐기.
5. `MULTI_CAUSE`는 결과다. 하나를 고르라고 강제하지 않는다.
6. **판정은 exit code가 아니다.**

---

## 6. 해석 제한

1. **ncu는 커널을 직렬화한다 ⇒ end-to-end 타이밍·ITL·처리량 주장 일체 금지.** 커널 내부
   구조만.
2. **nsys는 수 % 오버헤드가 있다 ⇒ 절대 ITL을 비프로파일 런과 비교하지 않는다.** 타임라인
   **구성비**(gap_frac)만 쓴다.
3. **정책 주장 0건.** 기전이 밝혀져도 "그러니 이런 컨트롤러가 답"으로 가지 않는다(게이트 #1,
   HE0 불변). 기전 → 정책 추론은 이 프로젝트에서 두 번 실패했다.
4. 측정 상자(arm·SM·B·L·backend) 밖 이식 금지(#11).
5. SM 수는 **드라이버 보고 realized**(E-3). 물리 SM id는 미측정(`%smid` 소관).
6. cudagraph-OFF 대조는 **간극 기전 한정**, 성능 비교 금지.
7. Gate 2 본 질문("PD 분리 자체" 귀속) 전진 0.

---

## 7. payoff (게이트 #28 — 좁게)

**산다**
1. **평탄화 후보 4개 중 최소 2개를 배제**한다(1차 분기만으로도). 현재 정본은 넷 다 열려 있고
   커널 측정이 0건이다.
2. ★**DRAM 트래픽을 처음으로 실측**한다 ⇒ `achieved_BW`의 항등식(#9 일곱 번째 재발)이 깨지고,
   "대역폭 포화 여부"가 계산치가 아니라 측정치가 된다.
3. **운영점(cudagraph-ON)에서 잰다** — LTSM P1 사다리(3–4 GPU-hr, 정의상 off-operating-point)를
   대체한다.

**사지 않는다**
- 정책 주장·HE0·layer-type 부활.
- Gate 2 귀속.
- **탄력도 자체를 설명하지 않는다** — 기전 서명을 주지 시간 배분의 인과를 주지 않는다.
- E-1이 쫓던 "조성 vs B-단조" 질문과는 **다른 질문이다**(그건 여전히 미해결).

---

## 8. 비용 · 큐

| 단계 | GPU | exclusive | 비고 |
|---|---|---|---|
| Stage 0 | ≈15분 | **불필요** | 사전등록 **전** 실행 권고(타당성) |
| Stage A (nsys) | ≈1hr | **불필요** | ncu 큐 대기 중 병행 가능 |
| Stage B (ncu) | 1–2hr | **필요** | 대기 길다 |

★ **대기 전략**: Stage 0 → Stage A를 먼저 돌리고, **Stage B를 큐에 넣은 채 Stage A를 분석**한다.
Stage A가 `GAP_DOMINATED`를 내면 Stage B의 우선순위가 내려간다(그 분기엔 불필요).

---

## 9. 다음 단계

1. 이 설계를 claims-auditor에 감사(설계 층).
2. **Stage 0 먼저 실행**(≈15분, exclusive 불필요) — 타당성이 확인되기 전에는 사전등록을 쓰지 않는다.
   rev1–rev3가 죽은 패턴이 "타당성 확인 전에 설계를 확정"이었다.
3. Stage 0 통과 후 Stage A+B 사전등록 작성 → 감사 → 제출.
