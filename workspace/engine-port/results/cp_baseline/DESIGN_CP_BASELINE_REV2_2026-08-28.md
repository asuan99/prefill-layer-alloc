# DESIGN rev2 — 정책 비교군 정리 + **chunked prefill 축 신설**(CP 캠페인)


> ⚠️ **날짜 정정**: 파일명/본문 날짜 2026-08-28은 오기 — 실제 작성/실행일 **2026-09-01**. 상세: `DATE_CORRECTION_NOTE.md`.

2026-08-28 · 메인 세션 · **사전등록 아님 — 규칙층 초안 rev2**(방법론 게이트 #34 1단 재제출 대상) ·
GPU 지출 **0**(제출 0건) · 새 성능 판정 **0건** · 등급 변경 **0건** · 정책 순위 변경 **0건** ·
**정본 아님**

> **rev1은 `NO-GO`였다.** `audit_cp_rules_2026-08-28/VERDICT.md`(claims-auditor, 死因 9 · 국소 12).
> 이 문서는 그 9건에 대한 응답이며, rev1(`DESIGN_CP_BASELINE_2026-08-28.md`)을 **대체**한다.
> rev1 문서와 `cp_rule_draft.py`에는 SUPERSEDED 배너를 달았다 — **인용 금지, 이력용.**
>
> ★**불변 승계**: HE0 · 정책 순위 · gate #13/#16 "닫았다" 금지 · switch-cost "닫았다" 금지 ·
> C2 인용정지 (a)(b). 이 문서는 그중 어느 것도 건드리지 않는다.
>
> **게이트 인용 규약(감사 L1)**: 저장소에 번호 체계가 셋(PROJECT_STATUS "방법론 게이트" #N /
> `CONSENSUS.md` §3 항목 N / 메모리 토픽 파일 항목 N)이라 rev1이 규칙 파일 안에서 둘을 섞었다.
> 이 문서와 `cp_rule.py`는 **번호 + 문서명 + 문구 첫머리**를 함께 적는다.

---

## 0. 한 줄 — rev1의 전제가 틀렸고, 정정하면 동기가 더 강해진다

rev1은 *"정본의 모든 정책 비교가 `--chunked-prefill-size -1`에서 측정됐다"*고 적었다. **거짓이다.**
- **pdmux arm**: `server_args.py:6129-6131`의 assert가 `-1`을 **강제**한다.
- **fused arm**: 정본 하네스가 cps 플래그를 **아예 주지 않는다**(`g2_run.sbatch:114-116`의 `plain`,
  `p1op_run.sbatch:51-58`) ⇒ `server_args.py:1196-1200`(A100-80GB 분기)이 **기본값 8192**를 유도한다.

⇒ 축 E는 **상수가 아니라 arm 종류에 따라 갈리는 미등록 교락**이었다. 정본의 모든 fused-vs-pdmux
대조는 *"chunked(8192) fused vs monolithic pdmux"*였고 **아무도 그걸 등록한 적이 없다.**
이건 rev1의 프레이밍보다 **더 강한 동기**다 — 빠진 baseline을 추가하는 문제가 아니라,
**이미 있던 교락을 등록하고 분리하는 문제**다.

---

## 1. 정리 — 지금 비교 검토 중인 정책 전수

### 1.1 축 구조

| 축 | 값 | 상태 |
|---|---|---|
| A. SM 공간분할 유무 | fused / pdmux(green-context) | 비교 중 |
| B. split 값을 무엇이 정하나 | 고정 / 배치크기 proxy / 측정 latency(closed-loop) / offline profile / context-length feedforward | 비교 중 (본 트랙) |
| C. 재분할 granularity | 없음 / step / layer-span 평가 / sub-step layer-type | 완료 — layer-type 全형태 死 |
| D. 프로세스 아키텍처 | single event loop / true dual worker | 구현 완료, 성능 미검증(R2) |
| **E. prefill 시간분할** | monolithic(`-1`) / chunked(`N`) | ★**pdmux=`-1` 강제, fused=미등록 기본 8192 — 등록된 적 없는 arm-종속 교락** |

### 1.2 정책 전수표 (축 B·C 중심)

판정·수치는 전부 정본(`CONSENSUS.md` §1-7/§1-10/§1-11/§1-16/§1-32/§1-33) 인용이며 재도출하지 않는다.

| # | 정책 (MODE / env) | 결정 신호 | granularity | 현재 판정 | 증거 등급 |
|---|---|---|---|---|---|
| 1 | **fused** (pdmux OFF) | 없음 | — | 대조군. 운영점 꼬리 SLO에서 pdmux에 열세 — 술어·모델·워크로드 한정, 기전 귀속 미확립 | 서빙, n=5 paired |
| 2 | **agnostic** (`agn`) | decode 배치크기 | step | 견고한 기본값 | 서빙 |
| 3 | **agnostic_v2** | 없음(전층 floor 16) | 고정 | decode 실작업 있으면 최악 | 서빙 |
| 4 | **tuned-uniform static** (`d16`…`d54`) | 없음(사전 튜닝) | 고정 | ★**최적 = d44** — 변화 trace goodput **3.220±0.013 (n=4)** | 서빙, n≥4 |
| 5 | **SLO-aware** (`slo`) | 측정 TPOT-EMA + prefill 큐 | layer-span 평가 | best-static 미달 (2.964±0.025) | 서빙, n≥4 |
| 6 | **binding-first** (`bind`) | 구속 다리(TTFT/TPOT) | 〃 | 2.934±0.306 (n=4), 1/4 붕괴 | 서빙, n≥4 |
| 7 | **feasibility gate** (`bind+GATE`) | 〃 + 안전 게이트 | 〃 | 3.132±0.019 (n=9). 견고성 확정, **성능 회복 반증**(d34 one-way ratchet) | 서빙, n=9 |
| 8 | **L-feedforward** (`PDMUX_SLO_LFF`) | admission시 context length | step | **HD0 — net win 아님** | 서빙, 3-rep |
| 9 | **R2 정책** (`fixed\|generic\|hybrid`) | offline hybrid-profile decode-floor | step | **구현 완료 ≠ 성능 주장 성립** | 미측정 |
| 10 | **layer-aware decode-type** | decode 층타입 | **sub-step** | ★**반증**(4개 실현 전부) | 서빙 |
| 11 | **layer-aware prefill-type** (PF) | prefill 층타입 | **sub-step** | ⚠️**인용 제한**(변수 동시변경 confound) | confound |
| 12 | **type-aware span sizing** | span 경계=type 경계 | span | net-negative | 서빙 |
| 13 | **true dual worker** | (축 D) | — | 미검증(H-Architecture) | 미측정 |

**★HE0(정본, 불변)** — 변화 trace(rate 3↔12, 3라운드):
**d44 3.220 > d34 3.171 > bind+GATE 3.132 > d24 3.039 > slo 2.964 > d16 2.846**,
d44↔bind+GATE 격차 0.088 = **5.4 pooled-σ**. tight SLO 재튜닝에서도 동일(rate8 attainment
d44 73.2% ≫ bind+GATE 44.3%). ⇒ **동적 제어는 best-static을 못 넘는다.**

### 1.3 비교군에 없는 것

| 없는 것 | 왜 | 지위 |
|---|---|---|
| **chunked prefill (등록된 축으로서)** | pdmux는 assert로 `-1`, fused는 미등록 기본 8192 | ★**이 문서가 신설** |
| `--enable-mixed-chunk` | E-A에서 시험 — decode를 `ForwardMode.MIXED` extend 경로로 재라우팅, ITL이 병합 decode 수에 선형 | 제외(§5.2, 한정 병기) |
| `--prefill-max-requests`, `--num-continuous-decode-steps`, `--max-running-requests`, `--schedule-conservativeness`, `--mamba-scheduler-strategy` | 미시험 | 정본 gate **E-D**(T2 종결) |

★**T2(불변)**: *"fused 조율 공간을 소진했다"*는 **정본 등재 금지.**

---

## 2. chunked prefill의 현재 지위 — 이미 측정된 것 (재도출 금지)

| 캠페인 | arm | 술어 | 남긴 판정 |
|---|---|---|---|
| **Gate 2 rev4** (875344/875346) | `A3 chunk512` vs `A4 agnostic` | `X_60`(요청-내부 max-ITL 임계 지시함수), n=10 paired | 5셀 전부 `Rprime4`(A4 우세, 등가 미발화). 크기 인용 가능 셀 2개 |
| **E-A** (875654/875657/875661) | `plainmix`·`chunk512`·`chunk512mix`·`agnostic` | 〃 | mixed-chunk 고유 기여 **+0.010=1.2%**(★**4셀 범위 1.2–10.6%**, ★**천장근접 하향편향 + HOLB 잔차와 같은 자릿수라 기전 해석 금지**, ★그 셀은 정본이 **사후 지정 셀 이동**으로 표시) |
| **g2ctrl / HOLB** | 4 arm에 `chunk512` 포함 | 관측자 효과·HOL blocking | 본 설계와 직교 |

### 2.1 아직 안 된 것 (이 캠페인이 사는 것)

1. **등록된 축이 아니었다** — §0. fused baseline은 8192에서 돌았고 아무도 그것을 처치로 세지 않았다.
2. **정본 벤치에서 잰 적이 없다** — Gate 2/E-A의 술어는 `X_60`, 정책 비교의 유효 벤치는 변화 trace +
   conjunctive goodput이다(PROJECT_STATUS "방법론 게이트" #2·#4). HE0 표(§1.2)에 CP 행이 없다.
3. **비교 상대가 `agnostic`뿐이었다** — 정본 최적은 **d44**다.
4. **SLO 임계 사다리가 없다** — §3.2.

---

## 3. 왜 추가해야 하는가

### 3.1 외부 대조군 parity
`impl_vs_external_pdmux_2026-08-28.md` §1: **MuxWise·BulletServe 둘 다 baseline이 chunked prefill**,
본 프로젝트만 fused. MuxWise는 **우리 base의 upstream**(`pdmux_context.py`와 2줄 차이).

### 3.2 ★기판 검증 — 우리 goodput 임계가 앉은 절벽
`CONSENSUS.md` §3 항목89: **monolithic prefill이 만드는 고정폭 스톨 때문에 요청-내부 ITL
히스토그램이 58–60ms 이봉 모드를 갖고, 정본 60ms 임계가 그 모드 위에 앉는다**(d24: 55ms 10.2% →
65ms 99.3%). 등급 PLAUSIBLE(서빙 직접 개입으로 미분리).
⚠️**중요(감사 D4 반영)**: 그 이봉은 **`d24`= pdmux arm에서 측정됐고**, CP arm은 assert 때문에
그 기판에 못 올라간다. ⇒ 이 payoff를 사는 것은 **cross-substrate 대조가 아니라 fused 기판 안의
chunk-size 대조**(`cp512 ↔ fused_mono`)다. 그래서 §6의 **1차 결정량이 C1**이다. rev1은
1차 결정량을 `bestCP − d44`로 두어 이 payoff를 **식별하지 못했다**.

### 3.3 등재된 gate E-D — ⚠️**이 워크로드로는 절반만 산다**
정본 gate #10이 E-D(미시험 fused 레버 스윕, T2 종결)를 등재했고 첫 줄이
`--chunked-prefill-size 2048/4096`이다. ★**그러나 §4.2가 보이듯 이 trace에서 cps4096은 요청을
0개 쪼개고 cps2048은 4개 쪼갠다** ⇒ **E-D의 그 항목은 이 워크로드에서 구매 불가**이고,
사려면 **다른 워크로드(더 긴 프롬프트)** 가 필요하다. 이 캠페인이 사는 것은 cps 512/1024뿐이다.

### 3.4 분류학 — CP는 "언제 재분할해도 되나"의 시간분할 판본
축 C(공간 재조정 시점)에서 layer-type은 全형태 死로 끝났다. CP의 경계는 **chunk 경계**이며
upstream이 이미 지원하는 자연 선점점이다.

---

## 4. 설계를 지배하는 사실 (전부 이 세션 직접 확인)

### 4.1 코드 — pdmux ⊥ chunked prefill
`sglang_engine_dev/python/sglang/srt/server_args.py:6125`(`if self.enable_pdmux:`) ~ `:6131`:
```python
assert (
    self.chunked_prefill_size == -1
), "PD-Multiplexing is not compatible with chunked prefill."
```
⇒ `CP × pdmux` 합성 arm은 **부팅되지 않는다**. CP는 fused-측으로만 들어온다.
⚠️ **"양립 불가능하다"로 쓰지 말 것** — 확인된 것은 **assert가 그렇게 막는다**까지다.

### 4.2 데이터 — arm 집합은 **벤치가 실제로 서빙하는 200개**에서 고른다
`measure_served_population.py` → `served_population.json`. ★rev1의 `sharegpt_prompt_lens.json`은
**다른 모집단**(파일 앞 4000개, 프롬프트 길이 필터만)이었고 네 값 전부 관대한 방향으로 과대였다.
정본 샘플러(`sample_sharegpt_requests`, `random.seed(1)` → shuffle → `prompt+output ≤ 4000` 드롭 →
앞 200개)를 **그대로 호출**해 재측정:

| p10 | p25 | **p50** | p75 | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|
| 13 | 27 | **217** | 522 | 845 | 1029 | **2776** | 3089 | 341.4 |

★**교차 검증**: p99 = **2776**은 정본 §1-16이 기록한 *"ShareGPT p99 2776tok"*과 **정확히 일치**한다
⇒ 이것이 정본이 실제로 돌린 모집단이 맞다.

**요청 단독으로 multi-chunk 되는 개수**(비율 아님 — seed 고정이라 200개는 **결정론적 상수**다):

| cps | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| **분할 요청 수 / 200** | **51** | **10** | 4 | **0** |
| rev1이 적었던 값 | 27.88% | 7.55% | 2.05% | 0.48% |

**등록 최소치 = 6개** = `ceil(δ·N)` = `ceil(0.03 × 200)` — **유도된 값이지 고른 값이 아니다**:
요청-채널 활동이 헤드라인 문턱(δ=3%, PROJECT_STATUS "방법론 게이트" #3 *"3% 미만 차이는 headline
아님"*)보다 적은 수의 요청에 닿는 arm은 **그 채널로는 헤드라인을 움직일 수 없다**.
⇒ **arm 집합 = {cps 512, cps 1024}**. cps 2048(4개)·4096(0개)은 **요청 채널에서 탈락**.

⚠️ 이건 요청 축뿐이다 — `chunked_prefill_size`는 **prefill 배치 전체의 토큰 예산**도 자른다.
그 채널은 런타임에만 관측되므로 CP-0 (c)가 잰다.

### 4.3 코드 산술 — CP가 layer-wise 분할을 퇴화시키는 **조건** (예측, 미측정)
`multiplexing_mixin.py:1169-1195`: `forward_count = max(1, 65536 // extend_num_tokens)`
(budget 65536은 `pdmux_context.py:20` 및 전 config에서 확인, Zamba2-2.7B `num_hidden_layers=54`).
`65536 // N ≥ 54` ⟺ **N ≤ 1213**.
⇒ cps 512·1024는 span 1개로 퇴화하지만 ★**cps 2048은 퇴화하지 않는다**(span ≥2).
rev1은 이를 일반 명제로 적었다 — **감사 L5로 정정**.
★**이것은 예측이다 — 측정된 바 없다**(PROJECT_STATUS "방법론 게이트" #70 *"검증 칸에는 이미 실행된
검증만 적어라"*). §5.4의 별도 트랙에서만 검증 대상이다.

### 4.4 플래그 묶음 — rev2는 **단일 레버 대조로 설계해 회피**한다
rev1은 A1/A2 선례를 근거로 "핸디캡 arm 불요"라고 적었다. 감사 L3이 3겹 부실을 지적했다:
(i) 정본이 그 비교에 **필수 병기 배너**를 달았다 — *"이 A2-vs-A4 비교는 사전등록 분석기
`g2_analyze.py`가 계산하지 않는다 … 저장된 primary 산출물 위의 **사후 계산**"*
(`CONSENSUS.md:1308-1313`, §3 항목34) (ii) A1↔A2는 **두 플래그가 함께 다르다**
(cps 8192→−1 **그리고** overlap ON→OFF) (iii) 그 측정은 `X_60`·다른 벤치다.

⇒ **rev2는 그 선례에 기대지 않는다.** 대신 fused 대조군을 **둘** 등록해 단일 레버로 만든다:

| arm | 플래그 | 역할 |
|---|---|---|
| `fused_mono` | `--chunked-prefill-size -1` | chunking OFF 기준. `cp512`와 **cps 하나만** 다르다 |
| `fused_default` | (cps 플래그 없음 → 8192) | **정본 fused arm이 실제로 돈 설정**(§0) |
| `cp512` / `cp1024` | `--chunked-prefill-size 512 / 1024` | 처치 |

전 arm 공통: overlap schedule **ON**(pdmux가 아니므로 강제 없음), 나머지 부팅 플래그 동일.
⇒ **E1 가족(C1·C2)은 완전 단일 레버**다. A1/A2 선례는 **인용하지 않는다.**

---

## 5. 캠페인 설계

### 5.0 공통 규약
- **모델**: Zamba2-2.7B(정본 변화-trace 기판). 2번째 모델은 CP-1 결과 후 조건부.
- **운영점**: cudagraph **ON**. `--disable-radix-cache`·`--mem-fraction-static 0.82`·
  `--max-running-requests 48`·`--context-length 4096` 전 arm 동일.
- **벤치**: 변화 trace(LO=3, HI=12, ROUNDS=3, NP=200).
- **채점**: 내장 스코어러를 쓰지 않는다 — 그것은 `mean`-ITL로 채점한다
  (`sharegpt_vary_bench.sbatch:94` `m=(sum(I[i])/len(I[i]))` → `:96` `if t<=3.0 and m<=0.06`,
  이 세션 직접 확인). 정본 술어(PROJECT_STATUS "방법론 게이트" #4)는 **요청-내부 token-ITL p95**다.
  별도 `score_cp.py`가 **두 다리 모두 산출 + 어느 다리가 구속하는지 필수 병기** + 사다리 6점 +
  ITL 히스토그램(임계 근방 해상도) + duration 라운드 **합산**(게이트 #7).
  ⚠️**부수 발견(결론 무변, doc-steward 이관)**: 정본이 이 사실을 인용할 때 쓰는 좌표
  `sharegpt_vary_bench.sbatch:88,92`는 현재 트리에서 그 두 줄을 안 가리킨다(현재·`31b3e96`·
  `f921ae8` 모두 94/96; `92`가 goodput 술어와 맞는 것은 `0b4fa4d` 판본뿐). **주장 자체는 참.**
  `check_line_citations.py`의 등재된 사각(쉼표 목록 미포착)의 실례.
- **통계**: **paired t-CI(df = n−1)**. percentile bootstrap을 쓰지 않는다 — PROJECT_STATUS
  "다음 실험 gate" #8이 그 undercoverage(`CONSENSUS.md:1472-1473`, n=4 0.798 / **n=6 0.859** /
  n=8 0.888)를 **제출 선행조건**으로 등재했고, 정본 처방은 n 증가가 아니라 **추정량 교체**였다.
  **n = 10** paired rep. FWER는 4개 등록 대조에 **Holm**.
- **보고 필수**(게이트 #5): TTFT/ITL p50/p95/p99 · ITL 히스토그램 · pdmux arm은 `switch_count`+
  체류분포. ★**CP/fused arm의 `switch_count`는 `null`**(0은 "전환이 없었다"는 관측처럼 읽혀 거짓 대칭).
- **배치 규율**: rep 순차 제출 · `SLURM_NODELIST` rep마다 기록 · ★**한 rep의 5 arm은 같은 노드에서**
  (감사 L12 — 정본 §1-32 필수 한정 (6)이 이 하네스 계열의 arm×node/날짜 교락·블록 배치·
  co-tenancy를 실제 문제로 기록). 불가능하면 `n_indep(placement)`를 별도 보고.

### 5.1 Stage CP-0 — 배관 스모크 + 양성대조 + 용량 (**채점 판정 0건**)
PROJECT_STATUS "방법론 게이트" #26(*"대형 캠페인 제출 전 배관 스모크를 규율로 등재하라"*).

| 항목 | 측정 | 사전등록 기준 |
|---|---|---|
| (a) **config echo** | `/server_info` + `internal_states[0]` + HOLB `arm_chunked_prefill_size` | 요청값과 일치. ⚠️**"3중 증인"이 아니다** — 셋 다 같은 `server_args` 필드를 읽는 **에코**다(`holb_probe.py:574`, 감사 L6). 실현 검증은 (b)/(c)가 한다 |
| (b) **양성대조 채널1 — 요청 분할** | ≥2 chunk로 쪼개진 **요청 수** | **≥6** (§4.2) |
| (c) **양성대조 채널2 — 배치 예산** | prefill forward당 평균 extend 토큰이 `fused_mono` 대비 | **≥3% 상대차 + paired CI가 0 배제**(프로젝트 표준 등가 마진 재사용) |
| (d) cudagraph 상태 | decode cudagraph ON · piecewise 상태 | arm별 기록 |
| (e) **용량** | arm별 achieved throughput 포화점(3 rate, n=2) | CP-1b의 HI2 rate + `capacity` 축 값을 여기서 확정 |
| (f) correctness | G-1(batch=1 결정성 vs `fused_default`) · G-2(16-동시) | ★Granite `chunk512`의 16-동시 비트단위 재현성 상실은 **기지 관측**(correctness 결함 아님) |

★**(b)와 (c)가 둘 다 음성인 arm은 `fused_mono`와 같은 arm**이다(`DEGENERATE_ARM`).
§6.2 시나리오 D′가 이 경우 **`NOTHING_PURCHASABLE`**임을 수치로 보인다 — 그래서 이 게이트가 있다.

### 5.2 Stage CP-1a — 정본 변화 trace (HI=12)
- **arm(5)**: `cp512` · `cp1024` · `fused_mono` · `fused_default` · `d44`.
- **n = 10**, arm 순서 무작위, rep당 노드 고정.
- **등록 대조 4개**(§6.1). 전부 보고, Holm 보정. **argmax 없음.**
- **SLO 사다리**: TTFT {1.0, 3.0}s × ITL-p95 {50, 60, 80}ms = 6점, **CI 기반**(§6.1).
- **제외 arm과 사유**: `--enable-mixed-chunk` — E-A가 기전을 서빙 직접 측정으로 규명.
  ⚠️**필수 병기**: 그 근거의 크기는 **1.2%가 아니라 4셀 범위 1.2–10.6%**이고, 정본이
  *"천장근접 하향편향 + HOLB 잔차와 같은 자릿수라 기전 해석 금지"* + *"사후 지정 셀 이동"*을 달았다.
  ⇒ **"mixed-chunk가 이 벤치에서 죽었다"고 쓰지 않는다.** 제외는 **예산 결정**이지 판정이 아니다.
- **cps 2048/4096 제외 사유**: §4.2 (4개 / 0개 < 6). ⇒ **E-D의 그 항목은 이 워크로드로 못 산다**(§3.3).

### 5.3 Stage CP-1b — 용량 아래 HI2 (**무조건**, 감사 D9)
CP-0 (e)가 정한 HI2로 같은 5 arm × n=10. rev1은 이를 라벨 조건부로 두었는데, 도달 가능한 패배
라벨이 정확히 하나여서 **"결과가 부정적이지 않을 때만 크기를 산다"는 비대칭 선택**이 됐다.
**rev2에는 라벨이 지출을 게이트하는 곳이 없다.**

### 5.4 Stage CP-3 — `CP × pdmux` 합성 (별도 트랙, 이 캠페인 밖)
§4.1 assert 제거 필요 + §4.3 예측 검증 필요. 순서: engine-porter 실현가능성 프로브(correctness 우선)
→ layer-span 퇴화 여부 **측정** → 그때 정책 비교. **CP-1 결과가 정당화할 때만.**

### 5.5 예산 — `sacct` 실측 기반 (rev1은 `--time` 상한 역산으로 ~3× 과대)
실측: `sacct -X --name=sgptv`, **n=34 완료 런, mean 8.0분(7.2–11.9)**.

| 단계 | 런 수 | GPU-hr | 근거 |
|---|---|---|---|
| CP-0 | — | ≈1.5 | ★**추정**(새 job 형태라 sacct 근거 없음 — 이 표에서 유일한 미실측 항목) |
| CP-1a | 5 arm × 10 rep = 50 | **6.7** | 50 × 8.0분 |
| CP-1b | 50 | **6.7** | 〃 |
| **합계** | | **≈15** | |

---

## 6. 결정 규칙 (rev2) + 도달가능성 인증

규칙은 코드로 고정: **`cp_rule.py`(`RULE_REV=2`)**, 5축 **192세계**, **14라벨**(실질 11).
자기검사: **`cp_rule_selftest.py`** → `cp_rule_selftest_output.txt`.

### 6.1 무엇이 바뀌었나 (死因 → 응답)

| 死因 | rev2의 응답 |
|---|---|
| **D1** `ci_shape`가 전수도 배타도 아니었고 평가 순서가 라벨을 정함 | `ci_shape(lo,hi,δ)`를 **전역 함수**로 고정(sign × magnitude의 곱, 8값). 자기검사가 **전수성·배타성·전사성**을 격자 반례탐색 + **독립 2차 구현과의 일치**로 증명하고, **변이 5종 전부 검출**. ★rev1이 값조차 없던 셀 `pos_indet`가 바로 정본 최다 인용 효과의 모양이다(`[+0.068,+0.108]` vs δ=0.0966) |
| **D2** 축이 아닌 3-arm 사후 argmax | **argmax 삭제.** 대조 4개를 이름으로 등록(C1–C4), 전부 보고, **Holm** 보정 |
| **D3** ladder가 점추정 부호 OR-스크린이고 검정력보다 먼저 | **CI 기반**(`concordant`/`conflicting`/`uninformative`) + **검정력 검사를 먼저**. ★**귀무 발화율 등록 = 1.696%**(6점, α=0.05, 독립 가정 = **보수적 방향**; 6점은 같은 rep 재채점이라 양의 상관 ⇒ 실제로는 더 낮다) |
| **D4** estimand가 §3.2 payoff를 식별 못 함 | **1차 결정량 = C1**(`cp512 − fused_mono`, **단일 레버**). d44 대조는 C4로 분리 + 스코프 배너 강제 |
| **D5** arm 정체 미등록·게이트 자기모순 | fused 대조군 **둘**(`fused_mono`/`fused_default`) 명시 등록(§4.4) · 정체 규칙 **하나**(양 채널 음성 ⟺ `DEGENERATE_ARM`)로 통일 · §0 전제 정정 |
| **D6** 모집단 오류 | 벤치 샘플러 직접 호출로 재측정, **개수로 등록**(§4.2) |
| **D7** `§1-33`/`2.36×` 오귀속 · `capacity`가 arm-쌍 위에서 미정의 | **§1-32 / 2.15–2.35×**로 정정하고 rule·spec 전 파일에 전파 · `capacity`를 **대조 위에서 정의**(both arms 규칙, **자유 모수 없음**) |
| **D8** 임계 3종 | `REQ_SPLIT_MIN_N=6`(=⌈δ·N⌉, **유도**) · `BATCH_MARGIN_REL=0.03`(프로젝트 표준 마진 재사용) · **CI 추정량 = paired t-CI** 명시 + gate #8 선행조건 문서화 · n=10 · 예산 sacct 재산정 |
| **D9** 조건부 지출 | CP-1b **무조건**. 규칙 파일에 지출 조건 **없음** |
| **L10** sub-δ 승리가 그냥 WIN | `CP_WINS_SUBDELTA` **별도 라벨** |

★**신규**: `IMPOSSIBLE_WORLD` — 축 조합이 공존 불가한 세계를 **이름 붙여 배제**한다(primary 운영점은
사다리 6점 중 하나이므로 `ladder=uninformative`인데 primary CI가 0을 배제할 수는 없다).
rev1에는 이 개념이 없어 격자 개수가 "가능한 결과"처럼 읽혔다(3회차 A1 감사 R1의 형태).

### 6.2 도달가능성 인증 (원자료 `reach_*.json`, 전부 재실행 가능)

| 시나리오 | 제약(출처 강제) | 실질 라벨 | 판정 |
|---|---|---|---|
| **C1** primary(`cp512−fused_mono`), HI=12 | `req_split=yes`(51 ≥ 6) | **11개 전부** | `DISCRIMINATING` |
| **C4** parity(`cp512−d44`), HI=12 | + `capacity=above`(§1-32 2.15–2.35×, d44측 확정 + both-arms 규칙) | 4개(RANK_ONLY 3 + LADDER_CONFLICT) | `DISCRIMINATING`, **크기 7개 도달불가** |
| **D** 탈락 arm cps4096 | `req_split=no`(0 < 6) | 11개 — **단 배치 채널이 물 때만** | `DISCRIMINATING`(조건부) |
| **D′** cps4096 + 배치도 안 뭄 | + `batch_budget=slack` | **없음** | ★`NOTHING_PURCHASABLE` |
| **R** 기각 판본(capacity가 전부를 게이트) | C4와 동일 | **없음** | ★`NOTHING_PURCHASABLE` |

**읽기**
1. ★**rev1 대비 최대 개선**: rev1은 정본 trace에서 **순위만** 살 수 있었다(`*_SIZED` 전부 도달불가).
   rev2의 **C1은 크기까지 산다** — d44가 없는 fused-내부 대조라 `capacity`가 §1-32에 묶이지 않고,
   두 arm의 용량이 CP-0 (e)에서 실측되기 때문이다. **크기를 사려고 rev1이 만든 조건부 단계가
   애초에 필요 없었다.**
2. **C4(외부 parity)는 여전히 순위만**. d44측이 HI=12에서 용량 위인 것이 정본이다. CP-1b가 그것을 산다.
3. **CP-0 양성대조는 장식이 아니다** — D′가 실제로 `NOTHING_PURCHASABLE`이다.
4. **기각 판본도 여전히 검사한다**(R) — "capacity가 전부를 게이트"는 데이터 전에 캠페인을 결정한다.

### 6.3 자기검사가 실제로 무엇을 잡았나 (정직 고지 2건)

**(1) 규칙 버그** — `cp_rule_selftest.py` **첫 실행에서 `cp_rule.py`의 실제 버그를 잡았다** — magnitude 라벨을 `"ge"`로
만들어 `"pos_ge"`를 반환했고, 이는 축 값 `"pos_ge_delta"`가 아니다. 전수성·전사성·독립구현 일치
검사 셋이 동시에 FAIL했고, 수정 후 전부 PASS. ⇒ **이 검사는 통과하도록 쓰인 것이 아니라
실패할 수 있는 검사다**(CONSENSUS §3 항목53 *"자기 수리를 검증하는 검사는 그 수리를 되돌린
변이본에서 반드시 실패해야 한다"*).

**(2) 아티팩트 자체가 파싱 불가였다** — `served_population.json`을 처음 만들었을 때
`sample_sharegpt_requests`가 stdout에 찍는 `#Input tokens:` / `#Output tokens:` 두 줄이
JSON 앞에 붙어 **유효한 JSON이 아니었다**. 문서·표의 수치는 옳았으므로 눈으로는 통과했고,
**나중에 `json.load()`로 자기 일관성을 검사할 때 비로소 드러났다**. 수정: 샘플러 stdout을
`contextlib.redirect_stdout`으로 포획해 `sampler_stdout` 필드에 보존한다(버림이 아니라 기록).
★**교훈 후보**: *"사전등록 산출물은 사람이 읽어 검사하지 말고 그것을 소비할 도구로 열어봐라"* —
규율 도구(`presubmit.py`·`design_reachability.py`)는 등록된 파일만 보므로, 등록되지 않은
아티팩트의 형식 오류는 아무도 안 잡는다.

---

## 7. 금지 문장 (사전등록 후보)

**rev1 승계(7)**
- *"chunked prefill이 pdmux를 대체한다 / 못 한다"* — Gate 2 rev4가 **다른 술어·다른 상대**로 낸 판정.
- *"fused 조율 공간을 소진했다"* — **T2 불변**. ★rev2는 E-D의 그 항목조차 **절반만** 산다(§3.3).
- *"CP와 pdmux는 양립 불가능하다"* — 확인된 것은 `server_args.py:6129-6131`의 assert뿐.
- *"CP에서는 layer-wise 분할이 무효화된다"* — §4.3은 **예측**이고 미측정이며 **cps 2048에서는 거짓**.
- *"CP-1이 HE0를 흔든다/되살린다"* — CP-1은 arm 추가이지 재스코어가 아니다.
- *"도달가능성 검사를 통과했으므로 제출할 수 있다"* — 다른 死因이 독립으로 남는다.
- *"양성대조가 통과했으므로 CP arm이 유효하다"* — (b)/(c)는 **arm 정체**를 검사하지 성능이 아니다.

**신규(감사 (e) FAIL 대응, 7)**
- ★*"CP가 d44를 이겼다/졌다"*를 **C4 스코프 배너 없이** — C4는 pdmux·overlap·chunking **3중 변경**이다.
- ★*"이득/손실의 원인은 chunking이다"*를 **C4로부터** — 단일 레버는 C1·C2뿐이다.
- ★*"`LADDER_CONFLICT`이므로 정본 60ms 임계가 기판 상수의 산물임이 확인됐다"* —
  `LADDER_CONFLICT`는 **임계에 대한 관측**이지 §3.2 기전의 확증이 아니다(CP arm은 §3 항목89가
  측정된 pdmux 기판에 못 올라간다).
- ★*"n=10이 undercoverage를 해결했다"* — 해결한 것은 **추정량 교체(t-CI)**이지 n이 아니다.
- ★*"`fused`는 chunked prefill이 없는 arm이다"* — `fused_default`는 **8192**다.
- ★*"CP-1이 E-D를 닫았다"* — cps 2048/4096은 이 워크로드에서 구매 불가(§3.3).
- ★*"mixed-chunk는 이 벤치에서 죽었다"* — E-A는 다른 술어·다른 벤치이고, 제외는 **예산 결정**이다.

---

## 8. rev2가 **고치지 않은 것** (감사 부채, 다음 감사에 그대로 제시)

"수리는 국소, 주장은 전역"을 피하기 위해 **남은 것을 먼저 적는다.**

1. **L2(하네스 하드와이어)** — 문서로만 기록했고 코드는 아직 안 건드렸다. 전수 grep 결과(이 세션):
   `sharegpt_vary_bench.sbatch:35-41`(MODE마다 pdmux CFG yml 필수, 미지 MODE는 `exit 1`) ·
   `:54`(`--enable-pdmux --pdmux-config-path --chunked-prefill-size -1 --disable-overlap-schedule`
   한 줄 묶음) · `:76`(`SW`를 pdmux 전용 로그 grep으로 유도) · `:77-104`(mean-ITL 스코어러) ·
   `:43`(`PDMUX_NO_CG` 분기) · `:46`(TAG 규약). **비-pdmux MODE 추가는 이 6곳을 전부 건드린다** —
   PROJECT_STATUS "방법론 게이트" #66(*"포크 착수 전 하드코딩 지점을 전수 grep하라"*)의 1단만 이행했다.
2. **CP-0 예산 ≈1.5 GPU-hr은 유일한 미실측 추정치**(§5.5).
3. **`capacity` 축의 both-arms 규칙**은 정의는 무-모수지만, **CP arm 용량이 아직 미측정**이라
   C1/C2/C3의 `capacity` 값은 CP-0 (e) 전에는 **미정**이다. 그 전에 크기 라벨을 인용하면 안 된다.
4. **2번째 모델(Granite) 미포함** — 외적 타당성은 CP-1 후 조건부.
5. **`score_cp.py` 미작성** — 규칙층 감사 통과 전에는 하네스를 만들지 않는다(게이트 #34).
6. **`check_line_citations.py` 스냅샷 미등록** — 이 문서의 인용을 검증한 **직후에만** 등록해야 하며
   (`--snapshot`이 기준선을 조용히 갱신할 수 있다), 아직 하지 않았다.

---

## 9. 다음 단계

1. ★**claims-auditor 규칙층 재감사(2회차)** — 대상: `cp_rule.py`(RULE_REV=2) + 본 문서 §4–§8 +
   `cp_rule_selftest.py` + `served_population.json` + `reach_*.json`. 단일 판정 질문과 합격 기준을
   함께 준다(PROJECT_STATUS "방법론 게이트" #37).
   ★**1회차 감사가 명명한 경고**: *"수리는 국소, 주장은 전역"* — §8이 남긴 6건과, rev2가 "정정했다"고
   적은 것들의 **파급**이 재감사의 1순위 표적이다.
2. 통과 시 → `PREREG_CP_2026-XX-XX.md` + `cp_rule.py` RULE_REV 승격 + `reachability_spec` 동봉 +
   `presubmit_registry.json` 등재.
3. 하네스(§8-1의 6곳) + `score_cp.py` → CP-0 → CP-1a → CP-1b.
4. 결과: result-analyst → claims-auditor → doc-steward.

---

## 부록 — 아티팩트

| 파일 | 내용 | GPU |
|---|---|---|
| `cp_rule.py` (`RULE_REV=2`) | 결정 규칙 — 192세계, 실질 11라벨, `ci_shape` 전역 함수 | 0 |
| `cp_rule_selftest.py` · `cp_rule_selftest_output.txt` | 전수성·배타성·전사성 증명 + 변이 8종 + 귀무 발화율 | 0 |
| `cp_rule_rejected_capacity_gates_all.py` | 기각 판본(증거 보존, rev2 추종) | 0 |
| `measure_served_population.py` · `served_population.json` | 벤치가 실제로 서빙하는 200개의 길이 분포·분할 **개수** | 0 (CPU) |
| `spec_{C1,C4,D,D_slack,R}_*.json` · `reach_*.json` | 도달가능성 사양·인증 | 0 |
| `audit_cp_rules_2026-08-28/VERDICT.md` | 1회차 감사 판정서(`NO-GO`) | 0 |
| ~~`DESIGN_CP_BASELINE_2026-08-28.md`~~ · ~~`cp_rule_draft.py`~~ · ~~`measure_sharegpt_prompt_lens.py`~~ | **SUPERSEDED — 인용 금지, 이력용** | 0 |
