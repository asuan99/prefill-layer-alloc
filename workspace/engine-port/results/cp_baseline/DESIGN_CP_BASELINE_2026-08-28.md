> ⛔ **SUPERSEDED (2026-08-28) — 인용 금지, 이력용.** claims-auditor 규칙층 감사가 이 판본에
> `NO-GO`(死因 9 · 국소 12)를 냈다: `audit_cp_rules_2026-08-28/VERDICT.md`.
> 현행 판본은 [`DESIGN_CP_BASELINE_REV2_2026-08-28.md`](DESIGN_CP_BASELINE_REV2_2026-08-28.md)
> + `cp_rule.py`(`RULE_REV=2`)다. ★이 문서의 다음 서술은 **거짓으로 확인**됐다 —
> (1) "정본의 모든 정책 비교는 `--chunked-prefill-size -1`에서 측정됐다"
>     (fused arm은 미등록 기본값 8192에서 돌았다)
> (2) ShareGPT 분할 비율 27.88/7.55/2.05/0.48% — 다른 모집단.
>     실제 서빙 집합은 **51/10/4/0 개**(`served_population.json`)
> (3) `CONSENSUS §1-33` · `2.15–2.36×` — 정본은 **§1-32** · **2.15–2.35×**.

> ⚠️ **날짜 정정**: 파일명/본문 날짜 2026-08-28은 오기 — 실제 작성/실행일 **2026-09-01**. 상세: `DATE_CORRECTION_NOTE.md`.

# DESIGN — 정책 비교군 정리 + **chunked prefill 축 신설**(CP 캠페인)

2026-08-28 · 메인 세션 · **사전등록 아님 — 규칙층 초안**(방법론 게이트 #34 1단 대상) ·
GPU 지출 **0**(제출 0건) · 새 성능 판정 **0건** · 등급 변경 **0건** · 정책 순위 변경 **0건** ·
**정본 아님**

> **지위**: 이 문서는 (1) 현재 비교 검토 중인 정책 전수를 정리하고, (2) 지금까지 **모든 arm에서
> 상수로 고정돼 있던 축**(`--chunked-prefill-size`)을 비교 대상으로 승격하는 캠페인을 설계한다.
> 성능 주장은 하나도 하지 않는다. 인용된 기존 수치는 전부 정본(`PROJECT_STATUS.md`,
> `reports/CONSENSUS.md`)에서 온 것이고, 이 문서가 새로 **측정**한 것은
> `sharegpt_prompt_lens.json`(CPU, 토크나이저 통계)뿐이다.
>
> ★**불변 승계**: HE0 · 정책 순위 · gate #13/#16 "닫았다" 금지 · switch-cost "닫았다" 금지 ·
> C2 인용정지 (a)(b) — 이 문서는 그중 어느 것도 건드리지 않는다.

---

## 0. 한 줄

**chunked prefill은 "빠진 baseline"이 아니라 "상수로 박혀 있던 축"이다.** 정본의 모든 정책 비교는
`--chunked-prefill-size -1`(monolithic prefill)에서 측정됐고, 정본 자신이 그 monolithic prefill이
**요청-내부 ITL p95의 58–60ms 이봉 모드를 만든다**고 기록했다(`CONSENSUS.md` §3 항목89) — 즉 우리
goodput 술어의 임계(60ms)가 앉아 있는 절벽을 **그 상수가 만든다**. 따라서 CP는 baseline 추가이자
**정본 운영점의 기판 검증**이다. 그리고 SGLang은 `enable_pdmux`와 chunked prefill을 **코드에서 상호
배제**하므로(아래 §4.1), CP는 pdmux arm에 얹을 수 없고 **fused-측 대안 기전**으로만 들어온다.

---

## 1. 정리 — 지금 비교 검토 중인 정책 전수

### 1.1 축 구조 (무엇을 변화시키고 있나)

| 축 | 값 | 상태 |
|---|---|---|
| **A. SM 공간분할 유무** | fused / pdmux(green-context) | 비교 중 |
| **B. split 값을 무엇이 정하나** | 고정 / 배치크기 proxy / 측정 latency(closed-loop) / offline profile / context-length feedforward | 비교 중 (본 트랙) |
| **C. 재분할 granularity** | 없음 / step / layer-span 평가 / sub-step layer-type | 비교 완료 — layer-type 全형태 死 |
| **D. 프로세스 아키텍처** | single event loop / true dual worker | 구현 완료, 성능 미검증(R2) |
| **E. prefill 시간분할** | monolithic(`-1`) / **chunked(`N`)** | ★**전 arm 상수 `-1` — 비교 안 됨** |

`E`가 이 문서가 신설하려는 축이다.

### 1.2 정책 전수표 (축 B·C 중심)

라벨 근거: `sharegpt_vary_bench.sbatch`의 MODE 분기 + `src/multiplex/`의 `PDMUX_*` 런타임 모드.
판정·수치는 전부 정본(`CONSENSUS.md` §1-7/§1-10/§1-11/§1-16/§1-17/§1-33) 인용이며 재도출하지 않는다.

| # | 정책 (MODE / env) | 결정 신호 | granularity | 현재 판정 | 증거 등급 |
|---|---|---|---|---|---|
| 1 | **fused** (pdmux OFF) | 없음 | — | 대조군. 운영점(cudagraph-ON) 꼬리 SLO에서 pdmux에 열세 — **술어·모델·워크로드 한정, 기전 귀속 미확립** | 서빙, n=5 paired (P1-opint) |
| 2 | **agnostic** (`agn`) | decode 배치크기(부하 proxy) | step | 견고한 기본값. 4모델 전부서 fused 이김(no-cudagraph 캠페인) | 서빙 |
| 3 | **agnostic_v2** (`PDMUX_AGN2_SM`) | 없음(전층 floor 16) | 고정 | decode 실작업 있으면 최악 | 서빙 |
| 4 | **tuned-uniform static** (`d16/d24/d34/d44/d54`…) | 없음(사전 튜닝) | 고정 | ★**최적 = d44** — 변화 trace goodput **3.220±0.013 (n=4)** | 서빙, n≥4 |
| 5 | **SLO-aware** (`slo`, `PDMUX_SLO_SCHED`) | 측정 TPOT-EMA + prefill 큐 | layer-span 평가, idx 변경시만 drain | best-static 미달 (2.964±0.025) | 서빙, n≥4 |
| 6 | **binding-first** (`bind`, `PDMUX_SLO_MODE=binding`) | 구속 다리(TTFT/TPOT) | 〃 | 2.934±0.306 (n=4) — 1/4 붕괴 | 서빙, n≥4 |
| 7 | **feasibility gate** (`bind+GATE`) | 〃 + 안전 게이트 | 〃 | 3.132±0.019 (n=9). ★**견고성은 확정**(0/9 붕괴), **성능 회복은 반증** — 실체는 d34에 고정되는 **one-way ratchet** | 서빙, n=9 |
| 8 | **L-feedforward** (`PDMUX_SLO_LFF`) | admission시 알려진 context length | step | **HD0 — net win 아님**(TTFT tail만 조임, goodput 무전환) | 서빙, 3-rep |
| 9 | **R2 정책** (`PDMUX_R2_POLICY=fixed\|generic\|hybrid`) | offline hybrid-profile decode-floor 예측 | step | **구현 완료 ≠ 성능 주장 성립** — 미검증 라이브 가설(H-Policy) | 미측정 |
| 10 | **layer-aware decode-type** (`PDMUX_LA_COORD*`) | decode 층타입(구조 proxy) | **sub-step** | ★**반증**(4개 실현 전부) | 서빙 |
| 11 | **layer-aware prefill-type** (`PDMUX_LA_COORD_PF`) | prefill 층타입 | **sub-step** | 표면상 최악이나 ⚠️**인용 제한**(변수 동시변경 confound) | confound |
| 12 | **type-aware span sizing** (`PDMUX_SLO_SPAN_TYPE`) | span 경계=type 경계 | span | net-negative | 서빙 |
| 13 | **true dual worker** (`PDMUX_TRUE_DUAL_WORKER`) | (축 D) | — | 미검증(H-Architecture) | 미측정 |

**★HE0(정본, 불변)** — 변화 trace(rate 3↔12, 3라운드) 위에서
**d44 3.220 > d34 3.171 > bind+GATE 3.132 > d24 3.039 > slo 2.964 > d16 2.846**,
d44↔bind+GATE 격차 0.088 = **5.4 pooled-σ**. tight SLO(chat 300/50ms)로 **재튜닝**해도 동일
(rate8 attainment d44 73.2% ≫ bind+GATE 44.3%, ≈10σ). ⇒ **동적 제어는 best-static을 못 넘는다.**

### 1.3 비교군에 **없는** 것 (이 문서의 표적)

| 없는 것 | 왜 없나 | 지위 |
|---|---|---|
| **chunked prefill** | 전 arm이 `--chunked-prefill-size -1` 고정(pdmux가 코드에서 요구, §4.1) | ★**이 문서가 신설** |
| `--enable-mixed-chunk` | E-A에서 시험됨 — decode를 `ForwardMode.MIXED` extend 경로로 재라우팅해 ITL이 병합 decode 수에 선형 | 기전 규명됨, 재구매 불필요(§5.2 제외 사유) |
| `--prefill-max-requests`, `--num-continuous-decode-steps`, `--max-running-requests`, `--schedule-conservativeness`, `--mamba-scheduler-strategy` | 미시험 | 정본 gate **E-D**(T2 종결)로 등재됨 |

★**T2(불변)**: *"fused 조율 공간을 소진했다"*는 **정본 등재 금지**. 위 미시험 노브가 남아 있다.

---

## 2. chunked prefill의 현재 지위 — 이미 측정된 것 (재도출 금지)

CP는 **완전한 미측정이 아니다.** 이미 arm으로 존재한 적이 있다.

| 캠페인 | arm 정의 | 측정된 것 | 남긴 판정 |
|---|---|---|---|
| **Gate 2 rev4** (jobs 875344/875346, Zamba2-2.7B·Granite-4.0-h-micro-base) | `A3 chunk512 = --chunked-prefill-size 512` vs `A4 agnostic` | primary = `X_60`(요청-내부 max-ITL 임계 지시함수) TOST 등가검정, n=10 paired, 5셀 | 5셀 전부 `Rprime4`(A4 유의 우세, 등가 미발화) ⇒ **"chunk512는 pdmux를 대체하지 못한다"** — 크기 인용 가능 셀은 F-E 통과분 2개뿐 |
| **E-A** (875654/875657/875661) | `plainmix`·`chunk512`·`chunk512mix`·`agnostic`, cudagraph-ON | mixed-chunk 레버 격리 | mixed-chunk 고유 기여 **+0.010 (=1.2%)**뿐. 기전 규명: MIXED extend 경로 재라우팅 → ITL이 병합 decode 수에 선형 |
| **g2ctrl / HOLB observer** (875610/875611 등) | 4 arm에 `chunk512` 포함 | 관측자 효과·HOL blocking | G5 UNDETERMINED 등 (본 설계와 직교) |

### 2.1 그래서 무엇이 **아직** 안 됐나 (이 캠페인이 사는 것)

1. **chunk 크기가 512 하나뿐.** 정본 자신이 `--chunked-prefill-size` **2048/4096을 미시험 노브로
   명시 등재**했다(T2 목록). 512는 §4.2가 보이듯 ShareGPT에서 **가장 공격적인** 설정이다.
2. **정본 벤치에서 잰 적이 없다.** Gate 2/E-A의 술어는 `X_60`이고, 정책 비교의 유효 벤치는
   **변화 trace(rate 3↔12, 3라운드)** + conjunctive goodput이다(방법론 게이트 #2·#4).
   ⇒ **HE0 표(§1.2)에 CP 행이 없다.** 지금 "CP vs d44"를 말할 근거가 없다.
3. **비교 상대가 `agnostic` 하나였다.** 정본 최적은 `agnostic`이 아니라 **d44**다.
   가장 강한 pdmux 정책과 CP가 붙은 적이 없다.
4. **SLO 임계 사다리가 없다.** §3.2의 이유로 CP는 임계 자체를 움직일 수 있는 arm이다.

---

## 3. 왜 추가해야 하는가 (네 가지, 독립)

### 3.1 외부 대조군 parity — 리뷰어가 반드시 묻는다
`reports/impl_vs_external_pdmux_2026-08-28.md` §1 표: **MuxWise와 BulletServe 둘 다 비교
baseline이 chunked prefill**이고, 본 프로젝트만 `fused(P1)`이다. MuxWise는 **우리 base의 upstream**
(우리 `pdmux_context.py`와 2줄 차이)이므로, "upstream이 쓰는 baseline을 우리는 안 썼다"가 된다.

### 3.2 ★기판 검증 — 우리 goodput 임계가 앉은 절벽을 그 상수가 만든다
`CONSENSUS.md` §3 항목89(및 §1-13 인근): **monolithic prefill(`--chunked-prefill-size -1`)이 만드는
고정폭 스톨 때문에 요청-내부 ITL 히스토그램이 58–60ms 이봉 모드를 갖고, 정본 60ms 임계가 그 모드
위에, chat 50ms 임계가 그 아래 바닥에 앉는다**(d24: 55ms 10.2% → 65ms 99.3%). 등급은 PLAUSIBLE
(서빙 직접 개입으로 분리 안 됨) — ★**CP arm이 바로 그 "직접 개입"이다.** chunk 크기를 바꾸면 스톨
폭이 바뀌고, 그 이봉 모드가 임계를 넘어 이동하는지 **직접** 볼 수 있다.

⇒ 이건 baseline 추가를 넘어선다: **정본 운영점(60ms)의 판별력이 기판 상수의 산물인지 검증**한다.
동시에 이것이 §6의 설계에 `LADDER_UNSTABLE`을 **실질 라벨**로 넣은 이유다 — 사다리에서 부호가
뒤집히면 그건 정책에 대한 진술이 아니라 **임계에 대한 진술**이고, 그 자체가 사는 것이다.

### 3.3 등재된 gate E-D의 최대 항목 — T2 종결의 절반
정본 "다음 실험 gate" #10이 **E-D(미시험 fused 레버 스윕, ≈8 GPU-hr, T2 종결)**를 등재했고
그 목록 첫 줄이 `--chunked-prefill-size 2048/4096`이다. 이 캠페인은 E-D의 그 항목을 산다.
⚠️ **T2를 닫지는 않는다** — 나머지 5개 노브가 남는다(§7 금지 문장).

### 3.4 분류학 — CP는 "언제 재분할해도 되나"의 **세 번째 granularity**
본 프로젝트의 축 C는 *공간* 분할의 재조정 시점(step / layer-span / sub-step)이었고
**layer-type은 全형태 死**로 끝났다. CP는 같은 질문의 *시간* 분할 판본이며, 경계가
**chunk 경계**다 — upstream이 이미 지원하고, 스케줄러가 이미 아는 자연스러운 선점점이다.
"layer 경계는 죽었다. 그럼 chunk 경계는?"이 논문 서사의 빈 칸이다.

---

## 4. 설계를 지배하는 사실 (코드·데이터)

### 4.1 ★코드 사실 — pdmux와 chunked prefill은 **상호 배제**
`/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/server_args.py:6125`(`if self.enable_pdmux:`) ~ `:6131`(assert 문구), **이 세션 직접 확인**:

```python
if self.enable_pdmux:
    assert self.pp_size == 1, ...
    assert (self.chunked_prefill_size == -1
    ), "PD-Multiplexing is not compatible with chunked prefill."
    assert self.disaggregation_mode == "null", ...
    assert self.disable_overlap_schedule, ...
```

⇒ **`CP × pdmux` 조합 arm은 현재 부팅되지 않는다.** 따라서 CP는 **fused-측 arm으로만** 들어오고,
"두 정책을 같은 기판에서 비교"가 아니라 **"상호 배제되는 두 기판의 대조"**다. 설계·서술 전부
이 사실 위에 서야 한다.

⚠️ **"CP와 pdmux는 양립 불가능하다"로 쓰지 말 것** — 확인된 것은 **assert가 그렇게 막는다**까지다.
물리적/원리적 불가능은 **미측정**이다(§5.4의 CP-3가 그것을 묻는 별도 트랙).

### 4.2 데이터 사실 — arm 집합은 **측정된 프롬프트 길이 분포**에서 고른다 (이 세션 측정)
`measure_sharegpt_prompt_lens.py` → `sharegpt_prompt_lens.json`
(Zamba2-2.7B 토크나이저, ShareGPT 첫 turn, n=3,949):

| p10 | p25 | **p50** | p75 | p90 | p95 | p99 | mean |
|---|---|---|---|---|---|---|---|
| 12 | 26 | **202** | 571 | 940 | 1181 | 2923 | 409.3 |

**요청 단독으로 multi-chunk 되는 비율**: `512 → 27.88%` · `1024 → 7.55%` · `2048 → 2.05%` ·
`4096 → 0.48%`.

⇒ ★**`--chunked-prefill-size 4096`은 요청 축에서 사실상 무효**다(0.48%). 정본 T2 목록이 지목한
2048/4096을 **그대로 arm으로 쓰면 4096 arm은 fused와 구분되지 않을 위험**이 있다. 이것이
§6의 도달가능성 시나리오 C/C′가 잡아낸 실패다.

⚠️ **이건 요청 축뿐이다.** `chunked_prefill_size`는 **prefill 배치 전체의 토큰 예산**도 자른다 —
평균 409 토큰 × 대기 요청 10개면 4096도 배치 축에서는 문다. 그래서 §5.3의 양성대조는
**채널 2개**(요청 분할 / 배치 예산 구속)를 따로 잰다.

### 4.3 코드 산술 — CP는 layer-wise 분할을 **퇴화**시킨다 (실행 관측 아님, 산술)
`multiplexing_mixin.py:1169-1195` (= MuxWise와 문자 그대로 동일):
`forward_count = max(1, 65536 // extend_num_tokens)`.
`extend_num_tokens ≤ 512`면 `forward_count ≥ 128 ≥ num_hidden_layers(~54)` ⇒ **prefill 전체가 span
1개**. ★즉 assert를 걷어내더라도 `CP × pdmux`에서 layer-wise 재분할은 **자동으로 사라지고**,
남는 것은 **chunk 경계 단위의 재분할**이다.

★**이것은 예측이다 — 측정된 바 없다.** 검증 칸에 적지 말 것(방법론 게이트 #70).

### 4.4 플래그 묶음 confound와 그 방향
`agnostic` arm은 `--enable-pdmux --chunked-prefill-size -1 --disable-overlap-schedule` 3개를 함께
켠다(assert가 뒤 둘을 강제). CP arm은 pdmux가 없으므로 **overlap schedule을 켜 둘 수 있다**.
정본이 이미 그 aux 쌍을 격리했다: `A1 plain`과 `A2 plainaux`는 5셀에서 `|A1−A2| ≤ 0.023`으로
사실상 동일하고, **그 부호는 pdmux arm에 불리한 핸디캡 방향**이다.
⇒ **처방**: CP arm은 overlap schedule **ON**(fused의 배포 최적 구성)으로 두고, 그것이
**우리 쪽에 불리한 방향**임을 명시 병기한다. 별도 핸디캡 통제 arm은 만들지 않는다(근거 = 위 격리).

---

## 5. 설계 — CP 캠페인 (4단계, 단계마다 게이트)

### 5.0 공통 규약
- **모델**: Zamba2-2.7B(정본 변화-trace와 동일 기판). 외적 타당성용 2번째 모델
  (Granite-4.0-h-micro-base)은 **CP-1 결과가 나온 뒤 조건부**.
- **운영점**: cudagraph **ON**(정본 운영점). `--disable-radix-cache`·`--mem-fraction-static 0.82`·
  `--max-running-requests 48`·`--context-length 4096` 전 arm 동일.
- **벤치**: `sharegpt_vary_bench.sbatch`의 변화 trace(LO=3, HI=12, ROUNDS=3, NP=200) —
  MODE 분기에 **비-pdmux 부팅 경로**를 추가하는 것이 유일한 하네스 변경.
- **채점**: ★스크립트 내장 스코어러를 쓰지 않는다. 내장 스코어러는 `mean`-ITL로 채점하며
  (`sharegpt_vary_bench.sbatch:94` `m=(sum(I[i])/len(I[i]))` → `:96` `if t<=3.0 and m<=0.06`,
  **이 세션 직접 확인**), 정본 술어(게이트 #4)는 **요청-내부 token-ITL p95**다.
  ⚠️**부수 발견(결론 무변, doc-steward 이관)**: 정본(`CONSENSUS.md` §1-33 · §1-7 행)이 이 사실을
  인용할 때 쓰는 좌표 `sharegpt_vary_bench.sbatch:88,92`는 **현재 트리에서 해당 두 줄을 가리키지
  않는다** — 현재·`31b3e96`·`f921ae8` 모두 94/96이고, `92`가 goodput 술어와 맞는 것은
  `f921ae8` **이전**(`0b4fa4d`) 판본뿐이다(BUGFIX 주석 4줄이 그 뒤에 삽입됨). **주장 자체는 참**
  (직접 읽어 확인). 이건 `check_line_citations.py`의 **등재된 사각(bare 인용·쉼표 목록 미포착)**에
  정확히 걸리는 실례이므로, 그 도구의 케이스로 등록할 가치가 있다.
  별도 스코어러 `score_cp.py`를 두고 **두 다리를 모두 산출 + 어느 다리가 구속하는지 필수 병기**
  (§1-33 재감사 권고). duration은 라운드 **합산**(게이트 #7).
- **보고 필수**(게이트 #5): TTFT/ITL p50/p95/p99 · ITL 히스토그램(임계 근방 해상도) ·
  pdmux arm은 `switch_count`+체류분포. ★CP arm의 `switch_count`는 **`null`로 보고**한다 —
  `0`으로 적으면 "전환이 없었다"는 관측처럼 읽혀 거짓 대칭이 된다.
- **배치 규율**: rep은 **순차 제출**하고 `SLURM_NODELIST`를 rep마다 기록한다. G16에서 블록
  동시 배치로 `n_indep(placement)=3 ≠ 4`가 된 선례가 있다.

### 5.1 Stage CP-0 — 배관 스모크 + 양성대조 + 용량 (≈2 GPU-hr, **채점 판정 0건**)
게이트 #25(대형 캠페인 전 배관 스모크)를 따른다. **성능 수치를 인용하지 않는다.**

| 항목 | 측정 | 통과 기준 (사전등록) |
|---|---|---|
| (a) realized 플래그 | `/server_info` + `internal_states[0]` + HOLB `arm_chunked_prefill_size` **3중 증인**(E-A 선례) | 요청한 `N`과 일치 |
| (b) **양성대조 채널1 — 요청 분할** | `≥2` chunk로 쪼개진 요청 비율 | **≥1%** 미달이면 그 arm은 `DEGENERATE_ARM`, CP-1에서 **제외** |
| (c) **양성대조 채널2 — 배치 예산 구속** | prefill forward당 평균 extend 토큰이 fused arm과 차이 | 등록 마진 이상이면 `binds` |
| (d) cudagraph 상태 | decode cudagraph ON · piecewise 상태 | arm별 기록(E-A의 런타임 반증은 2모델 한정) |
| (e) **용량** | arm별 achieved throughput 포화점 (3 rate 짧은 스캔, n=2) | CP-1b의 sub-capacity HI rate를 **여기서** 정한다 |
| (f) correctness G-1/G-2 | batch=1 결정성 vs plain · 16-동시 | ★Granite `chunk512`의 16-동시 비트단위 재현성 상실은 **기지 관측**(correctness 결함 아님) — 이 arm에서 재현되면 실패가 아니라 기록 |

★**(b)와 (c)가 둘 다 음성인 arm은 fused와 같은 arm이다** — 그 arm을 채점하면 항등식을 채점하는 것이다.

### 5.2 Stage CP-1a — 정본 변화 trace에서 순위 (≈12 GPU-hr)
- **arm**(CP-0 통과분): `cp512` · `cp1024` · `cp2048` · `fused`(CP 없는 대조) · **`d44`**(정본 최적,
  **반드시 in-campaign**). `agn`은 예산 허용시 추가.
  ⚠️ 기존 캠페인의 d44 수치를 끌어다 쓰지 않는다(job/node 축 교란 — gate #13 교훈).
- **n = 6**(n=4가 아님). 사유: 프로젝트 자체 paired percentile bootstrap이 **n=4에서 coverage
  0.798**(방법론 게이트 #14)이고, 이 캠페인의 결정 규칙은 단방향이 아니라 양방향이라 그
  undercoverage가 양쪽으로 샌다.
- **arm 순서 무작위화**, rep마다 노드 기록.
- **1차 결정량**: `D = goodput(best CP arm) − goodput(d44)`, rep으로 페어드.
- **★SLO 사다리 필수**: TTFT {1.0, 3.0}s × ITL-p95 {50, 60, 80}ms = 6점. 부호가 6점에서
  유지되지 않으면 라벨은 `LADDER_UNSTABLE`이고 **정책 판정을 내지 않는다**(§3.2).
- **제외 arm과 사유(사전등록)**: `--enable-mixed-chunk` 계열 — E-A가 기전을 서빙 직접 측정으로
  규명(MIXED extend 경로 재라우팅, ITL이 병합 decode 수에 선형)했고 고유 기여가 1.2%였다.
  ⚠️ 단 그 측정은 `X_60` 술어·다른 벤치였으므로 **"mixed-chunk가 이 벤치에서 죽었다"고 쓰지 않는다.**

### 5.3 Stage CP-1b — 크기 인용 (조건부, ≈9 GPU-hr)
**발화 조건**: CP-1a 라벨이 `CP_LOSES_RANK_ONLY`가 **아닐** 때만. (CP가 순위에서 명확히 지면
크기를 살 이유가 없다.)
- CP-0 (e)가 정한 **arm별 용량 아래** HI rate로 변화 trace 재실행. arm = {best CP, d44, fused}, n=6.
- 이유: 정본 HI(rate 12)는 **offered/achieved 2.15–2.36×**로 기록돼 있어 threshold-goodput의
  **크기**가 ill-posed다(게이트 #6). 순위는 살아도 크기는 못 산다 — §6이 이를 수치로 확인한다.

### 5.4 Stage CP-3 — `CP × pdmux` 합성 (별도 트랙, **이 캠페인에 포함 안 함**)
§4.1의 assert를 걷어내야 하고, 걷어내면 §4.3의 산술상 layer-wise가 퇴화한다. 순서:
(1) engine-porter 실현가능성 프로브(부팅·correctness 우선, 성능 아님) →
(2) layer-span 퇴화 여부 **측정**(§4.3의 예측 검증) → (3) 그때 비로소 정책 비교.
**CP-1 결과가 이 투자를 정당화할 때만 진행한다.**

### 5.5 예산 요약

| 단계 | GPU-hr | 조건 |
|---|---|---|
| CP-0 | ≈2 | 무조건 (게이트 #25) |
| CP-1a | ≈12 | CP-0에서 arm ≥2개 생존 |
| CP-1b | ≈9 | CP-1a 라벨 ≠ `CP_LOSES_RANK_ONLY` |
| CP-3 | 미산정 | 별도 트랙 |
| **첫 판정까지** | **≈14** | |

---

## 6. 결정 규칙 초안 + ★도달가능성 인증 (GPU 0, **이미 실행함**)

규칙은 산문이 아니라 **코드**로 고정했다(게이트 #66): `cp_rule_draft.py`(`RULE_REV=1`,
5축 96세계, 9라벨 전부 격자에 출현). 그리고 `scripts/discipline/design_reachability.py`를
**설계 단계에서 미리** 돌렸다 — `REACHABILITY_FINDING_2026-08-28.md` 처방 #1·#2의 첫 이행이다.

### 6.1 라벨
실질(substantive): `CP_WINS_{SIZED,RANK_ONLY}` · `CP_LOSES_{SIZED,RANK_ONLY}` ·
`CP_EQUIV_{SIZED,RANK_ONLY}` · `LADDER_UNSTABLE`.
비실질: `DEGENERATE_ARM`(양성대조 양채널 음성) · `UNDERPOWERED`(CI가 ±δ보다 넓음).

- `CP_EQUIV_*`는 **CI ⊂ (−δ,+δ)** 일 때만 발화한다(TOST형). 노이즈로 "차이 없음"을 사는
  귀무-채택형 게이트를 피하기 위함(게이트 #20).
- `capacity=above`는 **크기 라벨만** 막고 순위 라벨은 살린다.

### 6.2 도달가능성 결과 (원자료 `reach_*.json`)

| 시나리오 | 제약(출처 강제) | 설계가 낼 수 있는 실질 라벨 | 판정 |
|---|---|---|---|
| **A** 정본 trace만(HI=12) | `capacity=above`(§1-33: 2.15–2.36×) · `req_split=yes`(cps512 → 27.88%) | WINS/LOSES/EQUIV **_RANK_ONLY** + LADDER_UNSTABLE (4) | `DISCRIMINATING` — ★**`*_SIZED` 3개 도달불가** |
| **B** CP-1b(sub-capacity HI) 추가 | `req_split=yes`만 | **7개 전부** | `DISCRIMINATING` |
| **C** arm을 cps 4096 하나로 | `req_split=no`(0.48% < 1%) · `capacity=above` | 4개 — 단 **배치 채널이 물 때만** | `DISCRIMINATING`(조건부) |
| **C′** cps 4096 + 배치 예산도 안 뭄 | 위 + `batch_budget=slack` | **없음** | ★`NOTHING_PURCHASABLE` |

**읽기**
1. **정본 변화 trace만으로는 "CP가 d44보다 몇 % 낫다/못하다"를 살 수 없다** — 순위만 산다.
   크기를 원하면 CP-1b가 **필수**다. (설계 안에서 미리 알아낸 것이고, 데이터를 본 뒤가 아니다.)
2. **CP-0의 양성대조는 장식이 아니다.** C′는 실제로 `NOTHING_PURCHASABLE`이다 —
   양성대조 없이 cps 4096 arm에 GPU를 쓰면 **판정이 도착 전에 정해져 있는 지출**이 된다.
3. **arm 집합은 측정된 길이 분포에서 나와야 한다**(§4.2). T2 목록의 `2048/4096`을 그대로
   베끼면 4096 arm이 C/C′ 영역에 앉는다.

### 6.3 ★기각한 규칙 판본도 함께 검사했다
"HI가 용량 위니까 threshold goodput은 ill-posed → `NO_VERDICT`"라는 **더 보수적으로 보이는**
판본(`cp_rule_rejected_capacity_gates_all.py`)을 시나리오 A에 돌리면 **`NOTHING_PURCHASABLE`**이다
(`reach_R_rejected_variant.json`). 캠페인 전체가 데이터 도착 전에 결정된다.
이는 **NSL D4**("cap 축 제거가 답을 미리 정함")와 같은 형태이자 게이트 #40의 설계층 판본이다.
⇒ 채택 판본은 capacity를 **크기 축에만** 걸었다. 근거: 정본 자신이 **바로 그 운영점에서 HE0
순위를 인용**한다(§1-7).

---

## 7. 금지 문장 (사전등록 후보)

- *"chunked prefill이 pdmux를 대체한다 / 못 한다"* — Gate 2 rev4가 **다른 술어(`X_60`)·다른
  비교상대(`agnostic`)**로 낸 판정이다. CP-1 결과와 합치거나 갈아끼우지 않는다.
- *"fused 조율 공간을 소진했다"* — **T2 불변**. CP-1은 E-D의 한 항목만 산다.
- *"CP와 pdmux는 양립 불가능하다"* — 확인된 것은 `server_args.py:6129-6131`의 assert뿐(§4.1).
- *"CP에서는 layer-wise 분할이 무효화된다"* — §4.3은 **코드 산술에서 나온 예측**이고 미측정이다.
- *"CP-1이 HE0를 흔든다 / 되살린다"* — CP-1은 **arm 추가**이지 기존 arm 재스코어가 아니다.
  HE0는 pdmux 정책들 사이의 순위이고 CP는 그 집합 밖이다.
- *"도달가능성 검사를 통과했으므로 제출할 수 있다"* — 다른 死因이 독립으로 남는다
  (`REACHABILITY_FINDING_2026-08-28.md`가 이미 등재한 금지문).
- *"양성대조가 통과했으므로 CP arm이 유효하다"* — (b)/(c)는 **arm 정체성**을 검사하지 성능을
  검사하지 않는다.

---

## 8. 다음 단계 (순서 고정)

1. ★**claims-auditor 규칙층 감사** — 방법론 게이트 #34(규칙 → 하네스 2단). `cp_rule_draft.py` +
   본 문서 §5·§6·§7이 대상. 합격 기준과 단일 판정 질문을 함께 준다(게이트 #38).
   **감사 통과 전에는 하네스도, 사전등록도, 제출도 없다.**
2. 감사 반영 → `PREREG_CP_2026-XX-XX.md` + `cp_rule.py`(`RULE_REV` 승격) +
   `reachability_spec.json`을 **사전등록 산출물로 동봉**.
3. 하네스: `sharegpt_vary_bench.sbatch`에 비-pdmux MODE 분기 추가 + `score_cp.py`(p95 다리·
   구속 다리 병기·사다리·히스토그램). `--comment="field=efficientai;appl=pytorch"` 필수.
4. CP-0 제출(스모크, 채점 0건) → 양성대조로 arm 집합 확정 → CP-1a → 조건부 CP-1b.
5. 결과는 result-analyst(통계) → claims-auditor(주장) → doc-steward(정본 반영) 순.

---

## 부록 — 이 문서가 만든 아티팩트

| 파일 | 내용 | GPU |
|---|---|---|
| `measure_sharegpt_prompt_lens.py` · `sharegpt_prompt_lens.json` | ShareGPT 프롬프트 토큰 길이 분포(재현 스크립트 동봉) | 0 (CPU) |
| `cp_rule_draft.py` | 결정 규칙 초안, `RULE_REV=1`, 96세계 | 0 |
| `cp_rule_rejected_capacity_gates_all.py` | 기각한 판본(증거로 보존) | 0 |
| `spec_{A,B,C,Cp,R}_*.json` · `reach_*.json` | 도달가능성 사양·판정 | 0 |
