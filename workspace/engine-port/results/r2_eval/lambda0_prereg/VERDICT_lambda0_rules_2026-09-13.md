<!-- claims-auditor 규칙층 감사 반환문 원문 전사 (2026-09-13, 메인 세션이 스크립트로
     추출; 재타이핑·요약 아님). 대상 = PREREG_LAMBDA0_2026-09-13.md rev1 (sha fa4bb517).
     판정 NO-GO: 死因 N2(미등록 client seed + 미등록 shape A 사다리가 판정을 뒤집음)
     + N3(W4 근사 예보의 정의역이 공집합 - W4 decode phase는 (256,512) = shape A 자체).
     제출 차단 운영 결함 3건: UNRESOLVED 도달 불가 / shape KeyError / sbatch·analyzer 부존재. -->

판정서를 작성한다. 모든 수치는 원자료에서 독립 재계산했다.

---

# 규칙층 감사 판정서 — 캠페인 0단계 λ\* 사전등록

**대상**: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_eval/lambda0_prereg/PREREG_LAMBDA0_2026-09-13.md` (sha `fa4bb517…`) + `.../lambda0_label.py` (sha `101a19b9…`)
**감사일**: 2026-09-13 · **GPU 지출 0** (모든 재계산은 기존 원자료 + CPU)
**원자료**: `.../longctx_conflict/probes/c_905835/{cell_*.json, bench_*.jsonl, srv_*.log, C_LABEL.json}` · `ccap_905835.out` · `c_capacity.sbatch` · `c_capacity_label.py` · `c_capacity_analyze.py` · `benchmarks/pdmux_eval/{workloads.py,campaign.py,trace_loadgen.py}` · `scripts/r2_eval/{r2_eval.sbatch,engine_bench_runner.sh,generate_campaign.sh}` · `sglang/bench_serving.py` (editable tree)

## 0. 등급

# `NO-GO`

**死因 2건**: **N2**(반전 확인, 수치 있음 — 미등록 자유 표면 `--seed`가 F2/F3 판정을 뒤집는다) · **N3**(등록 예보의 정의역이 `∅` — W4 근사 예보가 존재하지 않는 shape에 대한 것임을 워크로드 생성기 실행으로 확인).

이 `NO-GO`는 "설계가 틀렸다"가 아니다. **측정 격자와 승계된 규칙(R1/R5)은 건강하다** — 아래 §8에서 나는 규칙이 probe C의 감사된 판정을 원자료 직접 대조로 정확히 재현함을 확인했다. `NO-GO`의 이유는 (a) 등록이 판정을 좌우하는 자유도 2개(client seed, shape A 사다리 전체)를 열어 뒀고 (b) 등록 예보 1건의 정의역이 비었으며 (c) **제출할 실행체(sbatch·per-cell analyzer)가 존재하지 않고, 존재하는 유일한 판정 코드는 이 저장소에 있는 어떤 cell JSON 형식도 읽지 못한다**는 것이다. D1–D14를 적용하면 다음 회차에 `GO` 또는 `GO-with-caveats`가 가능하다고 본다.

---

## 1. 반전 시험 표 (필수)

반전 계산에 쓴 원자료·추정량을 먼저 고정한다. **추정량**: `ach/off = achieved_rate/offered_rate`, 포화에서 `ach/off = λ*/offered`. 이 관계는 905835의 포화 셀 **3/3에서 소수 4자리까지 일치**한다 — d16 `0.9331/1.22 = 0.7649` vs 관측 `0.7649`; d44 `0.6753/0.86 = 0.7852` vs `0.7852`; d92 `0.1867/0.241 = 0.7747` vs `0.7746`. 따라서 R1의 브래킷 조건은 닫힌 형태로 환원된다:

> **브래킷 성립 ⟺ λ\* ∈ [0.950 · x_min , 0.900 · x_max]** (x = 등록된 **명목** rate 사다리)

역검증: d16 `[0.532, 1.098]` ∋ 0.933 ✓ · d44 `[0.380, 0.774]` ∋ 0.675 ✓ · d92 `[0.105, 0.217]` ∋ 0.1867 ✓. **감사된 3 arm 전부에서 이 공식이 실제 판정을 재현한다.** 이 공식 자신은 새 자유 표면이 아니다 — 등록된 문턱(0.95/0.90)과 등록된 사다리만으로 결정되고, 원자료로 검증됐다.

| # | 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 |
|---|---|---|---|---|
| **S1** | **client `--seed S`** (§2가 `S`를 기호로만 남김; probe C는 41로 고정) | 임의의 정수 | ★**반전** `KNEE_BRACKETED → KNEE_NOT_BRACKETED`, λ\* `값 → None` | `bench_serving.py:1706 np.random.seed(args.seed)` + `:948 np.random.exponential(1/rate)`. 비포화 셀의 `ach/off ≈ 1/Ē`(Ē = (N−1)개 Exp(1)의 표본평균). **905835 실측: 12/12 셀 전부 도착 구간이 명목보다 짧았다**(clean 셀 d44_r0 −15.5%, d16_r0 −18.3%; 평균 −2.0σ) ⇒ Ē ≈ 0.82–0.86, `ach/off` 1.03–**1.19**. 다른 seed에서는 반대로 간다: seed 24(N=90) → `ach/off = 0.757` ⇒ **용량의 53%에서 "포화" 오라벨**; seed 18·34(N=150) → 0.941·0.942 ⇒ **0.90 < x < 0.95 사각지대**로 저측 조건이 어느 셀에서도 성립 불가. seed 40개 모의: 저측 실패율 **N=90 27.5% · N=150 15.0% · N=300 20.0%** |
| **S2** | **shape A 사다리 4점의 값 전체** (§2 "rate 4점(앵커가 없어 넓게)" — 네 수가 **적혀 있지 않다**) | 등록이 제약하지 않음 | ★**반전** F3 `BRACKETED ↔ NOT_BRACKETED`, λ\* `값 ↔ None` | λ\*(A) 추정 2.1 req/s(§6)에서: `{1.0,2.0,4.0,8.0}` → 창 `[0.95,7.20]` ⇒ **BRACKETED**, λ\*=2.1. `{2.0,3.0,4.0,6.0}`(기본값 4를 믿은 설계) → 창 `[1.90,5.40]`, 최저점 2.0이 이미 λ\* 위 ⇒ 4셀 전부 `ach/off ≤ 0.90`, `≥0.95` 셀 0개 ⇒ **NOT_BRACKETED**. `{0.8,1.1,1.4,1.8}` → 창 `[0.76,1.62]` ⇒ **NOT_BRACKETED**(1.8 셀 `ach/off = 2.1/1.8 = 1.17`). ★R1은 **x_min·x_max에만** 의존하므로 중간 2점은 판정에 전혀 기여하지 않는다 |
| **S3** | **shape B 상단 배수** (§2가 `×1.2`로 등록하고 "probe C와 같은 배수 구조"라 서술; probe C 실제는 `{0.60,0.90,**1.30**}`, `c_capacity.sbatch:78-79`) | 등록이 주장한 승계값 `×1.30`까지 | ★**반전** `NOT_BRACKETED → BRACKETED` (구간 λ\*(B) ∈ (0.729, 0.790]) | 등록 `0.675×{0.55,0.85,1.20}={0.371,0.574,0.810}` → 창 `[0.3527, **0.7290**]` = 앵커 **+8.0%**까지. 주장된 승계 `×1.30` → `x_max=0.8775` → 창 상단 **0.790** = 앵커 **+17.0%**. ★등록 자신이 §2·F2에서 "out 96→64이므로 **약간 높을 것**"이라 예보 ⇒ **예보 방향이 곧 비브래킷 방향**. probe C prereg §3.2는 도달가능성 때문에 상단을 1.30으로 **올린** 것인데 이 등록은 그 여유를 되돌렸다 |
| **S4** | R1 문턱 `ACH_HI=0.95` / `ACH_LO=0.90` | 0.95→1.05, 0.90→0.80 | **반전 없음** | 905835 데이터는 문턱에서 멀다: 포화측 최대 0.7852(0.90보다 **0.115 아래**), 비포화측 최소 1.0266(0.95보다 **0.077 위**). 0.7852–1.0266 사이에 **데이터 0개**. 변이 M5(0.90→0.80)·M6(0.95→1.05) 모두 판정 불변 ⇒ 이 자유 표면은 probe C류 데이터에서 판정을 못 뒤집는다 → **등록 caveat로 전환**(selftest가 못 잡는다는 별건은 D9) |
| **S5** | `lambda_star = max(saturated)` vs `min`/`median` | max→min | **반전 없음 (905835에서), 단 정의역 밖** | 905835는 arm마다 포화 셀이 **정확히 1개**라 max≡min. 변이 M1이 selftest를 **통과**(ESCAPE)한다. 4점 사다리에서는 포화 셀 2–3개가 되어 구별이 살아난다. 크기: 이미 용량에 도달한 인접 두 셀의 achieved 차이는 d92 `0.1864→0.1867` = **0.16%**, d16 `0.9178→0.9331` = **1.67%** ⇒ max-편향 **≤1.7%**, 프로젝트 3% 문턱 아래 → **死因 아님, caveat** |
| **S6** | 동률·순서 규약 (`hi[0] > lo[0]` 단조 요구) | 단조 요구 제거 | **반전 없음 (코드는 옳다)** | 변이 M2가 selftest를 **통과**(ESCAPE)하나, 실제 코드는 단조를 강제하므로 비단조 사다리의 거짓 브래킷을 막는다. S1이 만드는 저rate 거짓포화(seed 24류)와 결합했을 때 단조 요구가 바로 그 방어선이다 → 코드 정당, **시험이 비어 있음**(D9) |
| **S7** | λ\* 정의 (throughput 포화 vs 정본의 "sustainable **SLO** rate") | 정본 정의로 교체 | **이 단계의 F1–F4는 불변**, 그러나 **생산되는 스칼라가 1.7–3.4× 달라짐** | `EXPERIMENT_ROADMAP.md:652` "먼저 B1 sustainable **SLO** rate `lambda*`를 모델별로 측정한다". 정본 goodput 술어(TTFT≤3000ms ∧ 요청내부 token-ITL p95≤60ms, `campaign.py:139-140`)로 905835 d44를 **내가 직접 재채점**: `0.59·λ*`(off 0.400) → **53.8%** · `0.89·λ*`(0.600) → **5.8%** · `1.27·λ*`(0.860) → **0.8%**. ⇒ λ\*_SLO(B) **< 0.40 = < 0.59·λ\*_throughput**. 두 정의 비 **≥1.7×(실측 하한), 추정 2.3–3.4×** → **死因 아님**(F1–F4 불변), **최강 caveat**(§4-1b, §7 인용금지 Q1) |
| **S8** | W4 근사 예보 "(64,512) ≈ (256,512), 5% 이내" | 예보의 정의역 확인 | ★**정의역 `∅`** | `workloads.py:159-160`을 실행: W4 **prefill phase = (in 8192, out 64)**, **decode phase = (in **256**, out 512)**. 저장소 전체 워크로드에 **(in 64, out 512)는 존재하지 않는다**. "64/512"는 `WorkloadSpec.output_distribution` 문자열(phase별 **출력** 길이)이며 입력 길이가 아니다 ⇒ **N3** |
| **S9** | 셀 결손 처리 (`UNRESOLVED` vs 규칙 라벨) | boot 실패 주입 | ★**반전** `UNRESOLVED → KNEE_NOT_BRACKETED`(측정 실패를 규칙 판정으로) | `lambda0_label.py:104` `for f in sorted(d.glob("cell_*.json"))` — 없는 셀은 리스트에 **안 들어가므로** `any(c is None)`이 거짓이고 `UNRESOLVED`가 **구조적으로 도달 불가**. 내가 주입 실험: 4점 사다리(λ\*=2.5)에서 상단 셀 1개 결손 → `KNEE_BRACKETED`(라더 3점으로 판정) / 하단 2점만 부팅 → **`KNEE_NOT_BRACKETED`**. 등록 §4는 "셀 JSON 결손 → **UNRESOLVED**, 측정 실패이며 규칙 실패가 아니다(교훈 21)"라고 적었다 ⇒ **코드가 등록 문안과 불일치**. probe C는 기대 셀 이름 목록을 순회해 이걸 옳게 했다(`c_capacity_label.py:48-59, 76-77`) |
| **S10** | cell JSON의 `shape` 키 | 기존 형식 투입 | **실행 불가** | `lambda0_label.py:106 rec["shape"]` — `c_capacity_analyze.py`는 `shape`를 **출력하지 않는다**(키: `label, random_input_len, random_output_len, offered_rate, achieved_rate, achieved_over_offered, …`). 실제 probe C 셀 JSON 투입 시 `KeyError: 'shape'`. 그리고 이 디렉터리에 **per-cell analyzer도 sbatch도 없다**(파일 2개뿐) ⇒ 제출할 실행체가 부존재 |

---

## 2. 死因 상세

### N2 — 반전 확인 (S1 + S2)

**S1(client seed)이 핵심이다.** 이 트랙이 한 번도 명시하지 않은 엔진 사실을 내가 확인했다:

- `bench_serving.py:1705-1706` — `random.seed(args.seed); np.random.seed(args.seed)`
- `bench_serving.py:943-950` — 비-inf rate에서 간격은 `np.random.exponential(1.0/request_rate)`, **yield 후 sleep**(⇒ N개 요청에 N−1개 간격)

따라서 비포화 셀의 `ach/off`는 **서비스 포화도가 아니라 도착 실현 계수 `1/Ē`**다. 905835의 "ach/off > 1" 11/12건(최대 1.1886)은 노이즈가 아니라 **이 계수**이고, 내가 `(N−1)/rate` 대비 도착 구간을 재구성해 **12/12 전부 같은 방향(−15~−18%, 평균 −2.0σ)**임을 확인했다 — seed가 프로세스당 한 번 고정되므로 **사다리 전체에 공통모드로 들어가고, 셀을 늘려도 평균되지 않는다**.

이것이 만드는 반전은 두 가지다.

1. **저측 소멸**: Ē ≥ 1.0526이면 비포화 셀이 `ach/off < 0.95`가 되어 R1의 저측이 어느 셀에서도 성립하지 않는다 → **두 shape가 동시에** `KNEE_NOT_BRACKETED`. seed 40개 모의 실패율 **15–28%**(N=90·150·300). 구체값: seed 18/34 @N=150 → 0.941/0.942(사각지대), seed 24 @N=90 → 0.757.
2. **거짓 포화**: 용량의 53%인 셀이 `ach/off = 0.757 ≤ 0.90`으로 포화 라벨을 받는다(seed 24). 이때 `max(saturated)`가 보호막이 되지만(낮은 rate → 낮은 achieved), S6의 단조 요구가 제거되면 거짓 브래킷까지 간다.

★**N≥400로도 못 고친다**: `1.96/√(N−1) ≤ 0.05`는 N ≥ 1537을 요구한다. N=400에서도 `P(ach/off < 0.95) ≈ 16%`. **등록된 0.95 문턱은 명목-분모 `ach/off` 위에서는 실현 가능한 N으로 신뢰 도달이 불가능하다.**

**S2(shape A 사다리 미등록)**는 독립적으로 같은 판정을 뒤집는다. 네 수가 적혀 있지 않은데 R1은 x_min·x_max만 쓴다 ⇒ **F3는 현재 예보가 아니다**. 게다가 §3이 "추가 1회 재설계 허용"을 미리 등록했으므로, 자유 격자 + 데이터 본 뒤 재설계가 겹친다.

### N3 — 예측 도달 불가 (S8)

§3이 "**반증 가능한 예보로 등록한다: 두 shape의 λ\*는 5% 이내로 같다**"고 쓴 대상 `(64, 512)`가 **저장소에 없다**. 생성기를 실행해 확인:

```
W4 n=96
   (in=256, out=512) phase=decode   n=48
   (in=8192, out=64) phase=prefill  n=48
```

**shape A = (256,512)는 W4 decode phase의 근사가 아니라 정확히 그것이고, 동시에 W3 전부다.** 즉 이 단계에 **근사는 존재하지 않으며**, §3 두 번째 bullet 전체와 그 5% 예보는 정의역 `∅` 위에 서 있다. (부수: 이 오독은 `WorkloadSpec.output_distribution = "64/512 by phase"`를 입력 shape로 읽은 데서 왔다 — 과제 지시문에도 같은 오독이 승계돼 있었다. 정본이 아니라 **코드 실행**이 이겼다.)

---

## 3. 항목별 판정 (질문 1–8)

### 1(a) 워크로드 분수가 이 정의 위에서 의미를 갖는가 — **PLAUSIBLE(조건부). 의도는 정합, 이름은 거짓.**

"B1 용량의 n%"는 **paired arm 비교 설계로서 옳다** — P2는 B1과 B4를 **같은 trace**로 비교하므로 부하를 한 arm 기준으로 고정하는 것이 정확히 필요한 통제다. B4가 그 부하에서 과부하인지 여유인지는 측정 대상이고, 그게 이 설계의 요점이다. 이 부분은 반증 실패.

단 세 가지를 등록해야 한다.

- **이름이 arm을 넘어 거짓이다.** 905835 실측 λ\*는 split에 따라 **5× 차이**(D16 0.933 / D44 0.675 / D92 0.187). "W9 = overload"는 B1 한정 서술이고, B4에서 같은 trace가 `0.2·λ*(B4)`일 수 있다. §6이 "λ\*는 B1 한정"은 적었으나 "**따라서 워크로드 이름이 다른 arm에서 거짓이 된다**"는 적지 않았다.
- **§0의 동기가 1차 범위와 불일치**: §0은 W8·W9 라벨을 근거로 들지만 1차 캠페인 범위(Claim D)는 **W3+W4만** 쓴다(`PROJECT_STATUS.md` C-2, `EXPERIMENT_ROADMAP.md` P2). W8/W9는 이 캠페인에서 **돌지 않는다**. 실제 in-scope 귀결은 "W3의 0.80 분수"와 "W4의 phase당 요청 수"뿐이다.
- ★**단일 스칼라가 W4를 자기모순으로 만든다 — 이 단계가 그 증명을 생산한다.** `generate_campaign.sh`는 9 워크로드에 **하나의** `PDMUX_SUSTAINABLE_RATE`를 쓴다(`:18`, `:26-32`). 내가 수치로 고정했다:

| 주입 λ\* | W4 per_phase | prefill phase (8192,64) | decode phase (256,512) |
|---|---|---|---|
| 기본 **4** | 96 | 3.20 req/s = **4.7 × λ\*(B)** | 3.20 = **1.3–1.5 × λ\*(A)** |
| λ\*(B) = **0.675** | 16 | 0.533 = **0.79 × λ\*(B)** ✓ | 0.533 = **0.21–0.25 × λ\*(A)** ✗ |
| λ\*(A) = **2.1** | 50 | 1.667 = **2.47 × λ\*(B)** ✗ | 1.667 = **0.79 × λ\*(A)** ✓ |

⇒ **어떤 스칼라도 W4의 두 phase를 동시에 0.80으로 만들 수 없다.** §6은 "설계 문제는 해소되지 않는다"고만 적었는데, 실제 결과는 더 강하다: **이 단계가 내놓는 두 λ\*는 W4에서 상호 배타적으로만 쓸 수 있다.** 이건 결함이 아니라 이 단계 최고 가치의 산출이지만, **반드시 등록돼야** 하고, 등록되지 않으면 결과 문서가 "λ\*를 측정했으므로 W4가 타당해졌다"로 새어 나간다(§7 Q3).

### 1(b) ★throughput 포화 vs SLO-지속가능 rate의 구분이 유지 가능한가 — **구분 자체는 유지 가능. 그러나 이 단계는 게이트 #6을 닫지 못한다. 라벨은 무의미하지 않고, "다른 것"이다.**

반증 시도 결과:

- **정본과 충돌한다.** `EXPERIMENT_ROADMAP.md:652`의 정본 문구는 "B1 **sustainable SLO rate** `lambda*`"다. 이 등록은 §1에서 "**throughput 포화이며 SLO-지속가능 rate가 아니다**"라고 **명시적으로 다른 양으로 교체**했으면서, 그것이 정본 정의의 교체임을 적지 않았다. 지시받은 규율대로 **문서가 이긴다** → 이 교체는 정정으로 등재되거나 철회돼야 한다.
- **두 정의의 차이를 내가 실측했다.** 905835 d44, 정본 goodput 술어(게이트 #4: TTFT ≤ 3000 ms **∧** 요청-내부 token-ITL p95 ≤ 60 ms, `campaign.py:139-140`)로 원자료 직접 재채점:

| offered | λ\*_thr 대비 | TTFT p50 | TTFT p95 | ITL p95 | **goodput(정본 술어)** |
|---|---|---|---|---|---|
| 0.400 | 0.59× | 2.84 s | 5.62 s | 22.6 ms | **53.8%** |
| 0.600 | 0.89× | 8.90 s | 16.53 s | 23.0 ms | **5.8%** |
| 0.860 | 1.27× | 29.65 s | 63.49 s | 23.1 ms | **0.8%** |

  ⇒ **"0.60·λ\*"는 이미 SLO 바깥이다**(0.59×에서 goodput 53.8%, 즉 절반이 이미 위반). λ\*_SLO(B) **< 0.59 × λ\*_thr**, 추정 0.20–0.30 req/s(비 2.3–3.4×). **W3 분수 0.80·λ\*(B) = 0.54 req/s에서 goodput은 대략 10–15%**, W8(0.90)≈5.8%, W9(1.10)≈1–2%.
- **그런데 이것이 라벨을 무의미하게 만들지는 않는다.** λ\*_thr는 잘 정의된 측정량이고 R2는 그것을 편향 없이 잰다(포화 셀의 achieved는 도착 실현과 무관 — S1 분석 참조). 그리고 **shape별로 갈린다**: shape A(256-in)의 TTFT 바닥은 256-token prefill ≈ 0.05–0.15 s(8192-token prefill이 64 SM에서 2.13 s인 것으로부터 선형 환산)이고 ITL은 13–26 ms ≪ 60 ms ⇒ **W3는 0.80·λ\*(A)에서 정본 술어로 측정 가능하다.** 절벽은 **8192-token prefill phase(shape B)에 국소화**된다.
- **게이트 #6·#12와의 대조 결론**: 게이트 #6("용량 먼저 측정")은 **지표의 절벽 대비 용량**을 요구한다. λ\*_thr은 그 절벽을 **위치시키지 않는다**(절벽은 0.59·λ\*_thr 아래). ⇒ **이 단계는 게이트 #6을 닫지 못한다.** 이것이 이 감사의 단일 최중요 결론이고 §7 Q1으로 문자 고정한다.
- **metric cliff 판정**: W4 prefill phase는 0.80·λ\*_thr에서 goodput 0.8–15% 구간 = **지시함수의 바닥 근처**. 두 arm(B1/B4)이 모두 바닥에 붙으면 P2는 **바닥 대조**가 되고, 53.8% 근처면 **절벽 최급구간**이 된다. 어느 쪽이든 P2의 측정가능성이 λ\* 정의 선택에 달려 있다 → P2 사전등록의 선결 사항으로 회부.

판정: **이 단계의 F1–F4는 이 쟁점으로 뒤집히지 않는다(死因 아님)**. 대신 **최강 등록 caveat** + **정본 정의 교체의 명시적 등재** 요구.

### 2 ONE BOOT PER CELL 승계 충실성 — **CONFIRMED (단, 더 강한 이유로), 인용 정정 1건 + 누락 2건.**

- **항목120의 死因은 실제로 회피된다.** 항목120은 "스냅샷 상태를 표본 간격 전체로 부과해 상태-지속 추정량을 만들고 그것을 문턱 게이트로 쓰는 것"이다(`CONSENSUS.md` §3 항목120, 실측 bin0 64×0.406 = 25.70 s 부풀림). 이 단계의 판정량 `achieved = completed/duration`은 **telemetry를 전혀 쓰지 않는다**(client-side count). ⇒ 한 boot 안 rate ladder 금지보다 **더 강한 차단**이 이미 성립. 등록은 약한 이유만 적었다.
- ★**인용 정정**: §2가 "probe C(`c_capacity.sbatch:30-35`)에서 **문자 승계**"라 했는데, 그 줄은 **주석**이다(근거 서술). 기제는 `:111` `for SPEC in "${CELLS[@]}"` 루프 + `:123-178`의 셀별 launch/kill/`sleep 10`/`pkill`이다. **주석의 문자 승계는 기제의 승계가 아니다.** 게이트 #110(직전 판정서 지적을 상수로 승격 금지)에 따라 나는 이 줄을 직접 열어 확인했다.
- **warmup 폐기**: probe C `:139-145`(8요청 `--max-concurrency 1 --seed 7`, 측정 제외) — 등록 §2가 승계 명시 ✓. 단 **비용이 shape A에서 3배**: 8 × (0.07 + 511×13.06 ms) ≈ **54 s/셀**(probe C out=96에서는 17 s). 예산에 반영돼 있지 않다.
- `--random-range-ratio 1.0` ✓ 등록됨(probe C `:142,:150`).
- ★**누락 2건**: §2의 하네스 줄에 **`--tokenize-prompt`와 `--dataset-path`가 없다**. probe C가 이걸 썼고, 그래서 입력 길이가 **정확히** 지켜졌다 — `bench_d44_r2_o96.log`: "Total input tokens: **1064960**" = 130 × 8192 (오차 0), "Total generated tokens: **12480**" = 130 × 96 (오차 0). 이 플래그 없이는 입력 길이가 텍스트 재토큰화에 흔들린다. 또 §2는 **출력** 길이만 검증 항목으로 적었다 — **입력 길이도** 검증 항목이어야 한다. `--output-details --output-file`도 누락(per-request `ttfts`/`itls` 없이는 후속 분석·정본 술어 재채점이 불가).

### 3 shape A 앵커 부재 — **예보 없는 4점 ladder는 타당하지 않다. 기존 데이터로 범위를 수치로 추정할 수 있고, 아래에 제시한다.**

먼저 등록이 제안한 경로를 **반증**한다: `c_capacity_analyze.py`의 닫힌 형태
`L_decode = [(1−R)/R]·(itl_load/itl_solo)/s(D)`
는 λ\* 예측에 **쓸 수 없다 — 항등식이다**. 대입하면

```
[(1−R)/R] = out·itl_solo/floor_ref ,  s_D = 1/(floor_ref·μ_ach)
⇒ pred = out·itl_load/(floor_ref·s_D) = μ_ach · out · itl_load
```

즉 `pred`는 **μ_ach(=λ\*)를 입력으로 요구**한다. 수치 확인: d44_r2 `0.6753 × 96 × 0.0204 = 1.3229` vs 보고된 `L_decode_obs/pred = 1.0068`, 그리고 `obs = μ_ach·out·mean_itl`이므로 `obs/pred = mean_itl/itl_p50 × (out−1)/out` — **R4 MODEL_HOLDS는 대체로 ITL 분포의 평균/중앙값 비를 검정한 것**이었다. (λ0 등록이 R4를 승계하지 **않은 것은 옳다**. 되살리지 말 것.)

**쓸 수 있는 경로: 엔진 로그의 decode step 곡선.** `srv_d44_*.log`의 `Decode batch, #running-req: B … gen throughput (token/s): G`에서 `step = 1000·B/G` (median, n≥3):

| B | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| step (ms) | 20.83 | 21.69 | 20.96 | 22.66 | 24.06 | 22.69 |

(B=1은 13.06 ms — pdmux 분할 미적용 구간.) **B=2..7 회귀: step(B) = 19.82 + 0.5174·B ms** (ms/req 기울기 = SSM 상태 트래픽; `srv` 로그 실측 `ssm_state size: 6.46GB / 48 slot = 134.6 MiB/req`, 읽기+쓰기 269 MiB, + ctx 8192 KV 128 MiB = 397 MiB/req).

구속 자원은 **`max_mamba_cache_size: 48` = `max_running_requests 48`**(둘 다 `srv_d44_r2_o96.log` 실측). KV는 전혀 구속하지 않는다 — `max_total_num_tokens = 2,618,868` vs 필요 48×8256 = 396K (**6.6× 과공급**).

```
λ*(A) = 48 / (512 · step(48))
```

| step(48) 가정 | 근거 | λ\*(A) |
|---|---|---|
| 13.06 ms | 측정된 **절대 최소** step(B=1) | **7.15** (절대 상한) |
| 24 ms | B=6 실측 step이 48까지 평탄 | 3.91 |
| **37.4 ms** | 회귀 기울기에서 **KV 항을 ctx 768로 축소**(397→281 MiB) | **2.51** |
| **44.65 ms** | 회귀를 B=48로 직선 연장(ctx 8192 기울기) | **2.10** |
| 70 ms | 기울기가 대형 B에서 악화 | 1.34 |
| 87 ms | 관측된 **최악** step(D16 @ B=30) | **1.08** (절대 하한) |

★**점추정 λ\*(A) ≈ 2.1–2.5 req/s** (두 독립 경로 일치) · **설계 브래킷 [1.3, 4.5]** · **절대 경계 [1.08, 7.15]**.
**참조**: 캠페인 기본값 λ\*=4는 shape A에서 1.6–1.9× 과대, **shape B(0.675)에서 5.9× 과대**. λ\*를 재는 것은 정당하다.

**권고 사다리(shape A, 5점)**: **{1.1, 1.8, 3.0, 4.9, 8.0} req/s**
→ 브래킷 창 `[0.950×1.1, 0.900×8.0] = [1.045, 7.20]` ⊇ 절대 경계 전체. `KNEE_NOT_BRACKETED`로 재설계 회차를 날릴 위험을 구조적으로 제거한다.
N/셀 = achieved 추정 × 170 s ⇒ **{190, 300, 360, 360, 360}** (상단 셀 duration ≈ 360/2.1 = 171 s, TTFT p99 ≈ 126 s — probe C가 101 s로 완주했으므로 하네스 내에 있다. 단 D13 참조).

**권고 사다리(shape B, 4점)**: **{0.45, 0.62, 0.85, 1.15} req/s**
→ 창 `[0.4275, 1.035]` = 앵커 **−37% ~ +53%**. 등록안 `[0.3527, 0.7290]`(+8.0%)은 **등록 자신의 예보 방향에서 깨진다**. N/셀 = **{90, 124, 140, 140}**.

### 4 W4 근사의 정당성 — **REFUTED (근사 대상이 존재하지 않음). 5%라는 수는 어디서도 오지 않았고, 올 필요도 없다.**

- §3의 전제가 틀렸다(§2 N3): W4 decode phase = **(256, 512)** = shape A **그 자체**. 근사 없음, 예보 불필요, 5% 불필요.
- **가정적으로** (64,512)가 존재했다면 5%는 변호 가능했을 것이다 — 단 등록이 적은 정성 논증("512 step 공유, prefill만 64 vs 256")으로는 부족하고, 실제 하중 지지 단계는 "**prefill이 어느 shape에서도 구속 자원이 아니다**"다: 256-token prefill ≈ 0.046 s(64 SM 환산) ⇒ prefill 용량 ≈ **22 req/s** ≫ λ\*(A) ≈ 2.1; 64-token이면 ≈ 86 req/s. 둘 다 무관. KV/mamba도 동일(mamba 136.90 MiB/req는 길이 무관, cache 48 고정). 고정 오버헤드 지배 영역에서도 같은 방향 ⇒ 두 극한 모두 예보를 지지.
- **기존 데이터로 검증도 반박도 불가**(이 모델에 (64,512)도 (256,512)도 측정이 없다) — 그래서 등록이 "반증 가능한 예보"라 라벨한 것은 **오라벨**이다. 이 단계가 검정하지 않는 명제는 예보가 아니라 **가정**이다. (§3 후단이 "가정이며 측정이 아니다"라고 자기 교정하고 있어 내부 모순이기도 하다.)
- **W4의 λ\*가 rate가 아니라 phase당 요청 수 승수라는 점**: §3 세 번째 bullet은 **정확하다** — `workloads.py:150` `per_phase = max(1, round(0.80·sustainable_rate·intensity·30.0))`, `:156` 30 s 구간 sorted-uniform. 반증 실패. 단 §3은 "**그 값[shape B의 λ\*]이 W4의 intensity 스케일링에 쓰인다**"고 적었는데, §2 표는 shape A를 "W4 decode phase"에 배정한다 — **둘 다 같은 한 스칼라에 들어갈 수 없다**(§3-1a 표). 이 모순이 등록 안에 미해결로 남아 있다.

### 5 통제 요인 불일치 — **앵커 0.675의 provisional 강등은 불충분하지 않다(앵커는 쓸 수 있다). 그러나 "러너와 일치시켰다"는 주장은 거짓이고, 차이는 3건이 아니라 최소 5건이다. ctx 16384는 용량을 바꾸지 않는다.**

**ctx 16384 / mem 0.82가 용량을 바꾸는가 — 답: 아니다, 구속 자원이 바뀌지 않는다.** `srv_d44_r2_o96.log` 실측(ctx 8704, mem 0.80):
```
Mamba Cache is allocated. max_mamba_cache_size: 48, conv_state 0.09GB, ssm_state 6.46GB
KV Cache is allocated. #tokens: 2618868, K 19.98 GB, V 19.98 GB
max_total_num_tokens=2618868, max_prefill_tokens=16384, max_running_requests=48,
context_len=8704, available_gpu_mem=13.38 GB
```
- 구속은 **`max_mamba_cache_size = 48` (= `max_running_requests`)**이고, ctx·mem과 무관하게 48이다(`max_mamba_cache_size=None` → 자동 유도, 134.6 MiB/req × 48 = 6.46 GB가 mem 0.82에서도 여유 13.38 GB 안에 든다).
- KV 토큰은 **6.6× 과공급**(2.62 M vs 필요 396 K). ctx 8192→16384는 요청당 필요 KV를 늘리지 않는다(프롬프트는 여전히 8192+64 / 256+512). mem 0.80→0.82는 static pool을 **늘린다**(KV 더 많이) ⇒ 어느 쪽도 용량 방향으로 구속되지 않는다.
- ⇒ **§2의 "차이 2건" 처리는 방향이 옳다**(앵커는 provisional로 쓸 수 있다). 반증 실패.

**그러나 "캠페인이 돌릴 구성과 일치시킨다"는 주장은 거짓이다.** `engine_bench_runner.sh`의 실제 server_args(cudagraph ON 경로)와 대조하면:

| 항목 | probe C (905835) | λ0 등록 §2 | 실제 캠페인 러너 | 상태 |
|---|---|---|---|---|
| mem-fraction | 0.80 | **0.82** | `${PDMUX_MEM_FRACTION:-0.82}` | ✓ 일치, 등록됨 |
| ctx | 8704 | **16384** | `${PDMUX_CONTEXT_LENGTH:-16384}` ← **run record가 이김** + `pdmux_eval.context_limit` preflight | ✓ 일치, 등록됨 |
| **`--disable-piecewise-cuda-graph`** | **ON** (`c_capacity.sbatch:126`) | **미언급** | **붙지 않는다**(cudagraph-OFF 경로에서만) | ★**미등록 차이** |
| **`--random-seed`** | 미지정(랜덤) | 미언급 | **`--random-seed ${server_seed}`** (run record) | ★**미등록 차이** |
| **부하 생성기** | `sglang.bench_serving` Poisson | `bench_serving` Poisson | **`pdmux_eval.trace_loadgen`** — `/generate`에 `input_ids=[1]*n`, `ignore_eos`, `temperature 0`, **고정 도착시각 재생** | ★**미등록 차이(가장 큰 것)** |
| pdmux config | `pdmux_homog5.yml` | "fixed D44(legacy)" | `benchmarks/configs/pdmux_r2.yml` | ★미확인 |
| max-running | 48 | 48 | run record (48) | ✓ |

- **`--disable-piecewise-cuda-graph`**: 등록이 "cudagraph ON"만 적었다. probe C는 piecewise를 **껐고** 캠페인은 **켠다**. `--chunked-prefill-size -1`이므로 영향이 작을 가능성이 높으나, 이 단계가 재는 것이 **8192-token prefill 처리율**이라는 점에서 비워 둘 자리가 아니다. 등록하거나(끄고 probe C와 맞추거나 켜고 캠페인과 맞추거나) 두 값 모두에서 한 셀 재는 것 중 하나를 **명시**해야 한다.
- **부하 생성기 불일치**가 본질적이다. λ\*는 `bench_serving` Poisson(난수 도착, 텍스트/토큰 데이터셋)으로 재고, 캠페인은 `trace_loadgen`(**결정적 도착시각**, `input_ids=[1]*n` 합성 토큰)으로 쓴다. 두 경로는 프롬프트 내용·토크나이즈·도착 분포가 모두 다르다. λ\*가 **다른 하네스에서 측정된 상수로 다른 하네스를 파라미터화**한다는 사실은 등록돼야 한다. (`input_ids=[1]*n`은 radix-cache OFF라 캐시 오염은 없지만, 단일 반복 토큰은 SSM 상태 궤적이 실제 텍스트와 다르다 — 이건 캠페인 쪽 기존 선택이고 이 단계의 결함은 아니다.)
- `runtime_source_manifest.sha256` "24항목, 모델 구현 포함"은 `PROJECT_STATUS.md`가 기록한 provenance 구멍(nemotron_h 수동 복사, manifest 17항목엔 zamba2·mamba2뿐)의 수리를 **전제**한다. 905835의 manifest는 24항목이 아니다. 이 단계 실행 전에 24항목이 **실제로** 나오는지 확인(D12).

### 6 판정 규칙 코드 — **규칙 승계는 CONFIRMED. selftest는 항등식이 아니다. 그러나 코드가 등록 문안과 불일치하고, 실행 불가하며, 변이 7/11이 빠져나간다.**

**(a) 항등식인가 — 아니다, 그리고 나는 독립 대조로 확인했다.**
selftest의 하드코딩 숫자 12개 전부가 원자료의 충실한 반올림이다(d16 ach/off `1.1700/1.0798/0.7649` → `1.17/1.08/0.765`; achieved `0.6552/0.9178/0.9331` → `0.655/0.918/0.933`; d44 `1.1422/1.0883/0.7852` → `1.14/1.09/0.785`, achieved `0.4569/0.6530/0.6753` → `0.456/0.654/0.675`; offered는 `c_capacity.sbatch:81-86`과 일치). 날조 없음.
더 강하게: **`shape` 키만 주입해 실제 보관 cell JSON 9개를 이 규칙에 직접 통과시켰다.**
```
d16  KNEE_BRACKETED  λ*=0.9331188685063582
d44  KNEE_BRACKETED  λ*=0.675302644015995
d92  KNEE_BRACKETED  λ*=0.18668462822130108
```
`C_LABEL.json`의 R1(3 arm BRACKETED)·R5(`mu_p_d16 = 0.9331188685063582`)와 **비트 단위 일치**. ⇒ **R1/R5의 실질 승계는 CONFIRMED.**

**(b) 변이 시험 — 11개 중 7개 ESCAPE.**

| 변이 | selftest |
|---|---|
| M3 `ACH_HI 0.95→1.20` | FAILS ✓ |
| M4 `ACH_LO 0.90→0.70` | FAILS ✓ |
| M9 포화 판정 부호 반전 | FAILS ✓ |
| M11 `ach/off`로 정렬(offered 대신) | FAILS ✓ |
| **M1 `max(saturated)→min`** | **PASSES ★** |
| **M2 단조 요구 `hi[0]>lo[0]` 제거** | **PASSES ★** |
| **M5 `ACH_LO 0.90→0.80`** | PASSES |
| **M6 `ACH_HI 0.95→1.05`** | PASSES |
| **M7 `>=`→`>` (ACH_HI 경계)** | PASSES |
| **M8 `<=`→`<` (ACH_LO 경계)** | PASSES |
| **M10 λ\* = 전 셀 max(포화 무관)** | **PASSES ★** |

M1·M10은 **포화 셀이 2개 이상일 때만** 살아나는 구별이고, 905835는 arm마다 정확히 1개였다 — **shape A의 4–5점 사다리는 2–3개를 만든다**. 즉 selftest는 이 단계가 실제로 들어갈 영역을 **한 번도 밟지 않는다**(교훈 53의 형태: 되돌린 변이본에서 실패해야 하는데 실패하지 않는다). M5–M8의 통과는 S4에서 본 "문턱이 데이터에서 멀다"의 이면으로 **반전은 아니지만 향후 재튜닝 무방비**다.

**(c) 등록 문안과 코드 불일치 — S9, 死因급 운영 결함.** `UNRESOLVED`가 live 경로에서 **도달 불가**하고, boot 실패가 **규칙 판정**(`KNEE_NOT_BRACKETED`)으로 나온다. 내가 주입 실험으로 재현했다. probe C는 이걸 옳게 했다. 등록 §4와 교훈 21·게이트 #21 직접 위반.

**(d) 실행 불가 — S10.** `rec["shape"]`가 필수인데 `c_capacity_analyze.py`는 `shape`를 내보내지 않는다(`KeyError: 'shape'` 재현). 그리고 **`lambda0_prereg/`에는 파일이 2개뿐** — sbatch도 per-cell analyzer도 없다. PREREG_CAPACITY는 sbatch + analyzer + label을 함께 출하하고 §4에서 "규칙은 그 파일과 이 절이 **동일**해야 한다"를 요구했다. 현재 상태로는 **제출할 것이 없다**.

### 7 출력공간·스코프 누출 — **F1–F4에 라벨 없는 칸 2개. 누출 경로 4개 확인.**

**라벨 없는 칸:**
1. **shape B의 `KNEE_NOT_BRACKETED`**. §3의 "추가 1회 재설계 허용"은 **앵커 부재 bullet(= shape A)에 달려 있다**. F2가 NOT_BRACKETED로 나왔을 때의 처분이 등록돼 있지 않다 — 그리고 S3가 보인 대로 그 확률이 실질적이다.
2. **F4 반전 시의 shape별 처분**. "브래킷은 미확정으로 등재하고 λ\*를 보고하지 않는다"는 적혀 있으나, **F4는 두 shape에 각각 적용되는지 합동인지** 불명. S1(seed 공통모드)에서는 제2 seed가 **양 shape를 동시에** 뒤집을 수 있어 이 구분이 실제로 갈린다.
3. (경계) `ach/off ∈ (0.90, 0.95)` 사각지대 셀의 라벨. 규칙상 양쪽 다 아님이 올바른 처리이나 §5 "출력공간 전수"가 이 칸을 열거하지 않았다. S1의 seed 18/34(0.941/0.942)가 정확히 여기로 떨어진다.

**누출 경로(차단 문안은 §7 Q1–Q4):**
- **(i) "λ\*를 측정했으므로 캠페인이 타당해졌다"** — §0이 이 단계를 게이트 #6 위반의 해소로 제시하므로 가장 강한 경로. §3-1b로 차단 필요.
- **(ii) "B1 용량이 곧 시스템 용량"** — §6이 부분 차단(5× 문장)했으나 워크로드 **이름**으로 되새어 나온다(§3-1a).
- **(iii) X1/907100의 Zamba2 결론 이식** — §6이 명시적으로 차단 ✓ (triton/flashinfer, `triton_attention_num_kv_splits` 부재까지 적었다). **반증 실패, 잘 쓰였다.** 다만 역방향 누출이 남는다: **905835(Nano-9B-v2 + flashinfer)의 12/12 boot 성공이 "새 모델 correctness가 부분적으로 확인됐다"로 읽히는 경로**. 905835는 **legacy 루프 + fixed split**만 발화했고 `PDMUX_TRUE_DUAL_WORKER` 경로는 **한 번도 돌지 않았다**.
- **(iv) λ\*(A)·λ\*(B)가 "W4를 파라미터화했다"** — §3-1a 표가 불가능을 증명. 차단 필요.

### 8 우선순위 — **권고: (1) 새 모델 correctness 게이트를 먼저 사라. 단 통상 생각보다 의존은 약하고, 진짜 이유는 다른 데 있다.**

**의존성의 정직한 해부:**
- λ\*는 **B1 = legacy 루프 + fixed D44**에서 잰다. 905835가 **이 모델·이 백엔드·이 arm을 12/12 boot·완주**시켰다. ⇒ **λ\*(B1)은 true-dual correctness 게이트를 선결로 요구하지 않는다.** "correctness 먼저"의 통상적 논거(실패하면 1 GPU-h 낭비)는 **성립하지 않는다** — correctness 게이트는 **B4**에 대한 것이다.
- 따라서 순서 권고는 다른 근거에 서야 한다. 그 근거는 **correctness 게이트가 λ\* 사다리 설계를 추측에서 측정으로 바꾼다**는 것이다. 내 λ\*(A) 점추정 2.1–2.5는 **B=7→48의 6.9배 외삽**에 서 있다 — 이 프로젝트가 반복해 처벌한 형태다(d16 앵커 0.94가 2점 외삽이었고, 맞았지만 그건 운이었다).

**correctness 게이트에 추가 boot 0으로 붙일 계측 3건(≈0.2–0.3 GPU-h 안에서):**
1. ctx **16384** / mem **0.82**에서 `max_mamba_cache_size`·`max_total_num_tokens`·`available_gpu_mem` 배너 기록 → 48 고정이 유지되는지 **확인**(내 §3-5 추론의 유일한 미측정 전제).
2. (256, 512) 8요청 **동시성 1** 프로브 → 256-token prefill 바닥과 ITL(B=1) 실측. shape A의 TTFT 바닥·warmup 비용·`step` 절편이 한 번에 나온다.
3. ★**(256,512)·(8192,64) 각각 `--request-rate inf --max-concurrency 64 --num-prompts 300`, 셀 1개씩 (합 ≈ 0.1 GPU-h)**. `bench_serving.py:943-945`가 `request_rate == inf`에서 **sleep을 전부 건너뛴다** ⇒ **Poisson 실현이 존재하지 않는다 ⇒ S1(N2 死因)이 구조적으로 소멸한다.** achieved = N/duration = 포화 처리율이 직접 나온다. 이것이 λ\* 사다리 양 끝을 **추측 없이** 고정한다.

**결론적 권고**: **(1) correctness 게이트 + 위 계측 3건을 먼저 산다(≈0.3–0.4 GPU-h).** 그 뒤 0단계 사다리를 **측정된 양 끝**으로 재등록한다. 이 순서가 더 좋아지는 이유는 "부팅·ITL 기초 수치를 얻으므로"가 맞지만, **더 결정적으로는 항목 3의 `--request-rate inf` 셀이 N2 死因 자체를 제거**하기 때문이다.

⚠️**자기 처방의 실현가능성 검사(게이트 #113)**: `--request-rate inf` 경로의 위험 2건을 내가 먼저 적어 둔다. (a) **confound #7(GIL/client 과부하)** — N개 연결을 동시에 열면 클라이언트가 병목이 될 수 있다. `--max-concurrency 64`로 묶고(엔진 상한 48보다 위), 클라이언트가 병목이 아님을 `Concurrency:` 보고값이 48 근처에 붙는지로 **확인 항목**으로 등록해야 한다. (b) **closed-loop(concurrency 64) 포화율 = open-loop Poisson 포화율인가.** 같다고 볼 근거: 양쪽 모두 엔진의 `max_running_requests 48`/mamba 48에서 포화한다. 다를 수 있는 근거: 905835의 Poisson 포화 셀 d44_r2는 `Concurrency: 24.08`(그중 `L_pre = 22.75`가 **대기**, `L_decode = 1.33`만 실행)이었다 — 즉 8192-in에서는 admitted 배치가 2–3에 머물렀다. closed-loop 64에서는 admitted가 더 찰 수 있어 **λ\*_inf ≥ λ\*_Poisson**일 수 있다. ⇒ **`--request-rate inf`는 λ\*의 대체가 아니라 사다리 양 끝을 고정하는 상한 프로브로만 등록하라.** 이 한계를 숨기면 내 처방이 다음 회차의 死因이 된다.

---

## 4. 제출 전 필수 조건 (D1–D14)

**死因 해소 (필수, 이거 없이는 `NO-GO` 유지)**

- **D1 (N3)** §3의 "W4 근사" bullet과 "5% 이내" 예보를 **삭제**하고 다음으로 교체: "shape A = (256,512)는 **W3 전부 + W4 decode phase 그 자체**다(`workloads.py:159-160` 실행 확인). 근사는 없다. 미측정으로 남는 shape는 W1/W5/W6/W7/W8/W9의 shape들이다." §2 표의 "W4 decode phase의 **근사**"에서 "근사"를 삭제.
- **D2 (N2-S1)** client `--seed`를 **정수 값으로 등록**하고, 동시에 **R1의 저측을 도착 실현 독립 형태로 교체**한다. 셋 중 하나를 택해 등록:
  (a) **권장** — 상단 2점을 `--request-rate inf --max-concurrency 64`로 재고 λ\*를 그 achieved에서 직접 읽는다(Poisson 실현 없음). Poisson 사다리는 "이 값이 open-loop 포화율의 상한임"을 확인하는 보조로만.
  (b) `ach/off`의 분모를 **명목이 아닌 실현 도착률**로 바꾼다(구현 필요: 송신 시각 기록).
  (c) Poisson·명목 분모를 유지하되 **저측 문턱을 실현 계수의 분포로 교정**하고, **모든 셀의 실현 계수를 필수 보고 항목**으로 등록 + `1/Ē ∉ [0.95, 1.05]`인 런은 `UNRESOLVED`로 라벨.
  ★어느 경로든 **§2에 "비포화 셀의 `ach/off`는 포화도가 아니라 도착 실현 계수 `1/Ē`다(905835 실측 1.03–1.19, 12/12 동일 방향)"를 명시**할 것.
- **D3 (N2-S2)** shape A의 **네(또는 다섯) rate를 수치로 등록**한다. 권고: **{1.1, 1.8, 3.0, 4.9, 8.0}**. 함께 **브래킷 창 `[0.950·x_min, 0.900·x_max]`를 계산해 적고**(권고안 = `[1.045, 7.20]`), §3의 λ\*(A) 사전 추정(점 2.1–2.5 / 절대 경계 [1.08, 7.15])과 그 유도(엔진 decode step 회귀 + mamba cache 48)를 **실행 전 공시**한다. probe C prereg §3.2·§3.3이 한 도달가능성 사전계산을 승계하라 — 그게 이 등록에 **빠진 유일한 구조적 요소**다.
- **D4 (S9)** `lambda0_label.py`를 **기대 셀 이름/사다리 목록을 인자로 받아** 순회하게 고치고, 결손 셀이 있으면 그 shape를 **`UNRESOLVED`**로 라벨. 변이 시험 추가: "상단 셀 파일 삭제 → `UNRESOLVED`"가 **반드시 실패→통과로 전환**되는지 확인.
- **D5 (S10)** per-cell analyzer를 **이 디렉터리에 출하**하고 `shape` 필드를 쓰게 하라(또는 `label`에서 파생). 그리고 **sbatch를 출하**하라 — 현재 제출할 실행체가 없다.

**등록 완결성 (필수)**

- **D6 (S3)** shape B 사다리를 **{0.45, 0.62, 0.85, 1.15}**(창 `[0.4275, 1.035]`)로 넓히고, §2의 "probe C와 같은 배수 구조" 서술을 **정정**하라 — probe C는 `{0.60, 0.90, **1.30**}`이고 그 prereg §3.2는 도달가능성 때문에 상단을 의도적으로 1.30으로 올렸다.
- **D7** F2가 `KNEE_NOT_BRACKETED`일 때의 처분을 등록(재설계 허용이 shape A에만 달려 있다). F4를 **shape별 독립**으로 적용할지 합동인지 명시.
- **D8** `ach/off ∈ (0.90, 0.95)` 사각지대 칸을 §5 출력공간 전수에 **명시적으로 열거**하라.
- **D9** selftest에 다음 4 케이스 추가(ESCAPE 봉쇄): (i) **포화 셀 2개 이상**인 사다리에서 `max(saturated)`를 고정(M1), (ii) 비포화 셀의 achieved가 포화 plateau보다 큰 경우로 "포화 셀 한정"을 고정(M10), (iii) **비단조** 사다리(저rate 0.85 / 고rate 0.98)가 브래킷 **아님**을 고정(M2), (iv) `ach/off`가 정확히 0.95·0.90인 경계 케이스(M7/M8). 그리고 **하드코딩 대신 `results/longctx_conflict/probes/c_905835/cell_d*_o96.json`을 직접 읽어** 대조하라(파일 존재, GPU 0) — 전사본이 아니라 감사된 아티팩트와 맞아야 한다.
- **D10 (§3-2)** §2의 하네스 줄에 **`--dataset-path $SGPT_RAW --tokenize-prompt --output-details --output-file`**을 추가하고, **달성 입력 길이**도 검증 항목으로 등록(probe C는 `Total input tokens = 130×8192` 오차 0). `--model --host --port`도 명시.
- **D11 (§3-5)** 미등록 통제 차이 **3건을 등록**: `--disable-piecewise-cuda-graph`(probe C ON / 캠페인 OFF — 어느 쪽으로 맞출지 택하고 근거 기재) · `--random-seed`(캠페인은 run record의 `server_seed`를 쓴다) · **부하 생성기 불일치**(λ\* = `bench_serving` Poisson / 캠페인 = `trace_loadgen` 고정 도착시각 + `input_ids=[1]*n`). pdmux config가 `pdmux_r2.yml`인지 확인.
- **D12** `runtime_source_manifest.sha256`가 실제로 **24항목 + nemotron_h 모델 구현**을 담는지 **실행 전 확인**(현 정본은 17항목 provenance 구멍을 기록 중). 안 담기면 그것부터 수리.
- **D13** 상단 셀의 TTFT p99 추정치(권고 사다리에서 ≈126 s)와 `bench_serving` 타임아웃·`max_queued_requests=None` 상호작용을 사전 확인. probe C 최대값은 101 s였다(`d92_r2_o96` p99 101.43 s) — 그 위로 올라가므로 무검증 영역이다.
- **D14 (§3-1b)** §1에 **정본 정의 교체를 정정으로 등재**하라: "`EXPERIMENT_ROADMAP.md:652`는 λ\*를 **sustainable SLO rate**로 정의한다. 이 등록은 의도적으로 **throughput 포화**로 교체한다. 두 양의 실측 비는 shape B에서 **≥1.7×**(정본 술어 goodput: 0.59·λ\*_thr에서 53.8%, 0.89·λ\*_thr에서 5.8%)." 교체를 숨기면 정본 충돌이다.

**예산 정정**

- **D15** 예산 헤드라인을 **0.75–0.95 GPU-h → 1.05–1.15 GPU-h(11 boot), 등록된 재설계 회차 포함 최악 ≈2.0 GPU-h**로 고쳐라. 근거: shape A warmup이 셀당 **54 s**(out 512×8요청, probe C는 17 s), 셀 5+4+2 = 11, 905835 실측 5.6분/boot + warmup 증분.

---

## 5. 실행 후 필수 병기 (결과 문서·정본이 그대로 승계할 문안)

결과가 나오면 아래를 **문자 그대로** 붙인다(어떤 라벨이 나오든 무조건).

1. **λ\*는 B1(legacy, fixed D44) 한정 throughput 포화율이다.** 905835 실측으로 split을 바꾸면 같은 모델·같은 shape에서 **5× 변한다**(D16 0.933 / D44 0.675 / D92 0.187 req/s). B4의 λ\*는 측정되지 않았다.
2. **정본 술어 goodput은 이 단계에서 측정되지 않았다.** 이 단계는 TTFT·ITL SLO를 쓰지 않는다.
3. **상단 셀의 TTFT/ITL 백분위는 인용 불가다** — 의도적 과포화 셀이므로 지연 수치가 아니라 처리율 plateau를 읽기 위한 것이다.
4. `switch_count`·split 체류분포는 이 단계의 판정에 쓰이지 않았다(fixed split, 동적 제어 없음).
5. **포화 셀의 achieved는 도착 실현에 무관하나(λ\*는 robust), 비포화 셀의 `ach/off`는 도착 실현 계수다.** 905835에서 1.03–1.19, 12/12 셀 동일 방향.
6. 이 단계가 쓴 부하 생성기는 `sglang.bench_serving`(Poisson)이고 캠페인은 `pdmux_eval.trace_loadgen`(고정 도착시각, `input_ids=[1]*n`)이다. **λ\*는 다른 하네스에서 측정된 상수로 캠페인을 파라미터화한다.**

## 6. 인용 금지 초안 (Q1–Q5)

> **Q1** — **이 캠페인 0단계는 어떤 결과가 나오더라도 방법론 게이트 #6("용량 먼저 측정")을 닫지 못한다.** 게이트 #6이 요구하는 용량은 **지표 절벽 대비** 용량이고, 이 단계가 재는 것은 throughput 포화다. 905835 원자료를 정본 술어(TTFT≤3000 ms ∧ 요청내부 token-ITL p95≤60 ms)로 재채점하면 shape B(8192-in)의 SLO 절벽은 **0.59·λ\*_throughput 아래**에 있다(0.59×에서 goodput **53.8%**, 0.89×에서 **5.8%**, 1.27×에서 **0.8%**). 따라서 "λ\*를 측정했으므로 W3/W4의 부하 라벨이 참이 되었다"는 문장은 **쓸 수 없다**.

> **Q2** — **λ\*(B1)의 측정은 (Nano-9B-v2, flashinfer) 쌍의 R2 correctness를 어느 정도도 확인하지 않는다.** 이 단계와 job 905835는 **legacy 루프 + fixed split**만 발화시켰다. `PDMUX_TRUE_DUAL_WORKER` 경로는 이 모델에서 **한 번도 실행된 적이 없다**. 907100·907456·X1의 결론은 **(Zamba2-2.7B, triton) 한정으로 동결**이고, X1의 민감도 시연은 `triton_attention_num_kv_splits`가 flashinfer에 없으므로 이식되지 않는다.

> **Q3** — **이 단계가 내놓는 두 λ\*는 W4를 파라미터화하지 못한다 — 상호 배타적이다.** `generate_campaign.sh:18`은 9 워크로드에 단일 스칼라를 쓰고 `workloads.py:150`의 W4는 두 phase에 **같은** `per_phase`를 쓴다. 실측/추정 λ\*로 계산하면: λ\*=0.675 주입 → prefill phase 0.79×λ\*(B) ✓ / decode phase **0.21–0.25×λ\*(A)** ✗; λ\*=2.1 주입 → decode 0.79×λ\*(A) ✓ / prefill **2.47×λ\*(B)** ✗; 기본값 4 → prefill **4.7×λ\*(B)**. **어떤 단일 스칼라도 두 phase를 동시에 0.80으로 만들 수 없다.** W4를 Claim D의 P2에서 쓰려면 워크로드 정의 자체(phase별 독립 λ\*)를 고쳐야 하며, 그것은 이 단계가 주지 않는다.

> **Q4** — **"B1의 용량이 시스템의 용량"이 아니다.** 워크로드 이름(W8 "near saturation", W9 "overload")은 **B1에서만** 참인 서술이고, B4에서 같은 trace는 자기 용량의 0.2×일 수도 5×일 수도 있다. 워크로드 이름을 regime 서술로 인용하는 것을 금지한다.

> **Q5** — **shape A·B 두 shape를 쟀다는 것이 W1·W5·W6·W7·W8·W9의 부하 라벨을 고치지 않는다.** 그 워크로드들의 shape는 미측정으로 남으며, λ\*는 shape 의존적이다(이 단계 자신이 두 shape에서 다른 값을 낸다).

## 7. 반증 실패 항목 (공정 기록)

다음은 깨뜨리려 했고 **깨지지 않았다** — 단 위 D조건 전에는 `CONFIRMED`가 아니다.

- **R1/R5의 실질 승계**: 실제 보관 cell JSON 9개 직접 투입으로 `C_LABEL.json`의 R1·R5를 **비트 단위 재현**. 항등식 아님, 날조 아님. ✓
- **R4(MODEL_HOLDS)를 승계하지 않은 결정**: 옳다. 그 닫힌 형태는 `μ_ach·out·itl_load`로 환원되는 **항등식**이고 `obs/pred`는 사실상 `mean_itl/itl_p50`를 검정했다. 되살리지 말 것.
- **`CONSENSUS §3 항목120` 회피**: 성립. 판정량이 telemetry를 전혀 경유하지 않는다(ONE BOOT PER CELL보다 강한 차단).
- **ctx 16384 / mem 0.82가 용량을 바꾸는가**: 바꾸지 않는다. 구속은 `max_mamba_cache_size = max_running_requests = 48`이고 KV는 6.6× 과공급(`max_total_num_tokens = 2,618,868`). 앵커 0.675의 provisional 강등은 **충분하다**.
- **F4의 제2 seed 통제**: 옳은 통제다 — `np.random.seed(args.seed)` 때문에 **같은 seed로는 도착 실현이 재표본되지 않는다**. 이 설계 요소만이 S1을 부분적으로 방어한다.
- **§6의 Zamba2 결론 차단 문안**: 잘 쓰였다(triton/flashinfer 비교, `triton_attention_num_kv_splits` 부재까지). 유지하고 Q2로 역방향 누출까지 막으면 된다.
- **W4가 rate가 아니라 phase당 요청 수 승수라는 §3 세 번째 bullet**: 코드와 일치, 정확하다.
- **arm 비교 금지 규율**(§4·§6, 사다리가 arm-상대): 일관되게 유지됐다.

---

**판정 요약**: `NO-GO` — **N2**(미등록 `--seed`와 미등록 shape A 사다리가 F2/F3와 λ\*를 뒤집는다; seed 실패율 15–28%, 사다리 반전 예시 3건) + **N3**(W4 근사 예보의 정의역 `∅`, 생성기 실행으로 확인). 여기에 제출 차단급 운영 결함 3건(`UNRESOLVED` 도달 불가 · `shape` KeyError · sbatch·analyzer 부존재). **설계의 골격과 승계된 규칙은 건강하므로**, D1–D15 적용 후 재감사에서 `GO-with-caveats`가 가능하다고 본다. **우선순위 권고: correctness 게이트 + `--request-rate inf` 2셀(≈0.3–0.4 GPU-h)을 먼저 사서 사다리 양 끝을 측정으로 고정하라 — 그것이 N2 死因을 구조적으로 제거한다.**

---

관련 파일 경로(전부 절대경로):
- 감사 대상: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_eval/lambda0_prereg/PREREG_LAMBDA0_2026-09-13.md` · `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_eval/lambda0_prereg/lambda0_label.py`
- 승계 원본: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/PREREG_CAPACITY_2026-09-09.md` · `.../probes/c_capacity.sbatch` · `.../probes/c_capacity_label.py` · `.../probes/c_capacity_analyze.py`
- 반전 계산에 쓴 원자료: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/c_905835/` (`cell_*.json` 12, `bench_*.jsonl` 12, `srv_d44_r1_o384.log`, `srv_d44_r2_o96.log`, `srv_d16_r1_o384.log`, `C_LABEL.json`)
- 캠페인 정의: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/benchmarks/pdmux_eval/workloads.py` (`:150`, `:159-160`) · `.../campaign.py` (`:138-140`) · `.../trace_loadgen.py` · `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/scripts/r2_eval/{generate_campaign.sh,r2_eval.sbatch,engine_bench_runner.sh}`
- 엔진: `/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/bench_serving.py` (`:913-950`, `:1426`, `:1705-1706`)
- 정본: `/scratch/ehmoon/whlee/prefill-layer-alloc/PROJECT_STATUS.md` (2026-09-13 배너 C-2·D) · `/scratch/ehmoon/whlee/prefill-layer-alloc/reports/paper/EXPERIMENT_ROADMAP.md` (`:652` 공통 방법, P2 절) · `/scratch/ehmoon/whlee/prefill-layer-alloc/reports/CONSENSUS.md` (§3 항목120)
