# 세션 핸드오프 — 2026-09-07

## 이번 세션 요약

`cp_baseline` 트랙의 **AF-1 무경쟁 바닥 프로브를 완주**했다. 규칙층 rev6이 이 트랙 최초로
`GO`를 받은 상태에서 하네스를 쓰고, 배관 스모크 2회로 계측 결함 2건을 잡고, 6 boot를
돌려 등록된 분석기가 **`STRATUM_DEPENDENT → 분기 b`** 를 냈다(**이 트랙의 첫 실질 라벨**).
이어 사용자 지시로 **"층을 고정할 근거"** 를 제안했으나 규칙층 적대 감사에서 `NO-GO`(死因
7건)를 받고 수용했고, 1차 출처 재조회·관련연구 조사로 **경로 (a)("외부 앵커 좌표 하나를
운영점으로 등록")가 닫힌 이유는 우리 측정이 아니라 그런 좌표가 존재하지 않기 때문**임을
확정했다. 마지막으로 이 트랙이 감사 7회 연속 NO-GO를 받은 **구조적 원인**을 명명하고
방향을 정리했다. **GPU 0.88 GPU-hr / 11 job / 커밋 11건 · 새 성능 판정 0건.**

## 결정·측정

### 1. AF-1 완주 — 첫 실질 라벨 (`results/cp_baseline/RESULT_AF1_2026-09-07.md`)

- 6 boot(seed 11·23·37·53·67·71), 전 boot `gpu42`, Nemotron-Nano-9B-v2-Base, ctx 32768,
  동시성 1, arm = `plain`/`cp2048`/`d44`.
- `world = sufficient|neutral|disagree|unique|intact` → **`STRATUM_DEPENDENT` → `b`**.
  중단 규칙 S5. 셀을 먼저 들여다보지 않고 분석기를 **1회** 실행했다.
- **어긋난 것은 사다리 하나뿐**: neutrality·anchor(=`batch_async`)는 p90·p99에서 같고,
  ladder만 `intact`↔`pruned`(T=500 **행 전체 6칸**). 사전등록 §4.4가 미리 등재한
  *"탈락은 항상 행/열 단위 6점"* 이 실측에서 그대로 나왔다.
- **이 축의 무처치 SD 첫 측정**: TTFT SD는 p90 이하 ≤0.53%, p99에서 1.9–7.9%. ITL ≤0.6%.
- ★이월 항목 해소: **`d44`의 q=0.99 부팅 간 SD 52.69 ms = `plain`의 4.35배**. 귀속 안 함
  (arm이 묶음).
- **등록 예측 5건 중 3건이 틀렸다**(1·2·4 ❌ / 3·5 ✅). §5.1이 *"틀리는 것이 결과다"* 로
  등록했으므로 산출물.
- 분기 `b`의 **등록된** 처방(`PREREG:428` *"방향 없는 단일 라벨 + 사다리 전수 공표"*) 이행
  = `FOLLOWUP_STRATUM_2026-09-07.md` §나. 36칸×2층×2모드 전수. **ITL 축은 어느 조합에서도
  한 칸도 못 떨어뜨린다** ⇒ `STRATUM_DEPENDENT`의 실체는 **TTFT 축 단 한 행**.

### 2. 스모크가 잡은 계측 결함 2건 (`AF1_HARNESS_ADDENDA_2026-09-07.md` C–F절)

- **층별 첫 요청 오염**(job 904819_0): `agg_within_boot`이 준-최대라 첫 요청이 그대로 boot
  값이 됐다(`cp2048` q=0.50에서 **16.51×**). 비용이 부팅이 아니라 프롬프트 *모양*을 따라
  **층마다** 워밍업이 필요. `WARMUP_REQUESTS_PER_STRATUM = 2` **DESIGN 상수**로 등록.
  ★추정량 불변 — 측정된 반복은 하나도 안 버리고 워밍업 TTFT는 `warmups_ms`로 공표.
- ★★**§9-1 반증 + 파티션 무발화 기전 확정**: `srv_d44.log:28`이 `sm_counts
  [(108,0),(64,44),(0,108)]`를 직접 찍고 `args_d44`가 무장을 확인한다. **동시성 1에서는
  분할 그룹 index 1 `(64,44)`가 선택 불가** ⇒ `d44`는 SM 분할을 한 번도 쓰지 않았다.
  판독기가 per-step 필드를 못 찾은 것은 계측 실패가 아니라 **참인 관측**(게이트 #21의
  거울 방향). 판독기는 **덧붙여** 수리(`CONFIGURED`/`REALISED`), 판정 술어 불변.
  ⇒ **신규 한계 L-10**을 캠페인 **전에** 등록.
- **호스트 동거 — 의심 → 측정 → 기각**: 첫 제출이 같은 노드 2-way 동시 실행이었고 스모크는
  단독이었다. 직렬화 대신 **어차피 버릴 boot로 페어 측정** ⇒ 12셀 중 10셀 −1.2%~+0.2%
  (부호도 음수) ⇒ 기각. ★부수 소득: 두 이상치가 서로 반대 방향이라 **스파이크의 동거
  원인이 배제**됐다. 지출 0.27 GPU-hr로 미등록 nuisance 크기 + 원인 배제를 샀다.

### 3. 층-근거 제안 rev1 → 규칙층 감사 `NO-GO` (死因 7건, 수용)

- 판정서 `results/cp_baseline/audit_stratum_ground_2026-09-07/VERDICT.md`(317줄).
  단일 판정 질문 답: **"이름만 바꿨다"** — 자유 표면이 `q`에서 **모집단**으로 옮겨갔다.
- **내가 원본을 열어 독립 확인한 死因 4건**: U1(같은 줄 후반절이 *"P90–P99"* 범위이고
  `af1_predicates.py:196-197`이 이미 그것을 p99 **반대**로 인용) · U2(풀 =
  `ShareGPT_long2048_cap8192`, 이름 자체가 꼬리 선별) · U3(q=0.99에서 **등록 예측 4가
  False→True로 뒤집힘** = 게이트 #8 정면 위반) · U7(분기 `b`의 등록 처방은 *"사다리 전수
  공표"* 이지 *"T=500을 피하라"* 가 아니다).
- ★**내 오류(U4)**: 제안 §7 금지문 #2가 *"바뀌는 것은 어느 바닥을 비교하느냐뿐"* 이라
  적었는데 U3가 반증한다. **오독을 막으려 쓴 금지문 자신이 거짓 진술을 담았고**, 금지문은
  산문이라 어떤 도구도 못 잡았다.

### 4. 1차 출처 재조회 — 앵커는 "미명세"가 아니라 **독자 입력** (`FOLLOWUP` §가)

- Spheron 표는 실재하고 6행이 `SURVEY_POINTS`와 전부 일치 ⇒ `C14`·서베이 §1 인용 건전.
- 두 조회가 겉보기에 어긋났고 **그 어긋남이 답**: 표는 길이를 고정하지 않지만(1차) 본문이
  *"**P99 prompt length**"* 로 평가하라고 직접 지시한다(2차, worked example **512 tok**).
- ⇒ **분위수는 0.99로 정해지고 남는 자유 표면은 오직 모집단.** 死因 U2는 반증이 아니라
  날카로워졌다. 직전 수용 기록에 *"외부 앵커의 미명세"* 로 등재할 뻔한 진단은 틀렸을 것
  (주장으로 안 올려서 다행) — 정정 배너 부착.
- 부수: 서베이가 1차 출처 **스냅샷을 안 남겨** 재조회가 필요했고 재조회가 더 찾아냈다.

### 5. 제거 판정 4건 해부 + 워크로드 탐색 (`WORKLOAD_REVIEW_2026-09-07.md`)

- 제거 4건은 **전부 순수 TTFT 탈락**(ITL 바닥이 최엄격 예산의 절반 이하). 최악 arm은 전 층
  `cp2048`.
- 역산: 각 좌표를 되살리는 p99 길이 = code 604 / voice 1,092 / **chat 2,463** / **rag 3,332**
  tok. 앞 둘은 외삽, 뒤 둘은 내삽.
- ★★**현 풀은 코퍼스 상위 2.12%**(meta 자기기록 92,886→1,968). AF-1의 네 층은 코퍼스
  **p98.09 / p98.94 / p99.79 / p99.98** ⇒ **전형 요청을 한 번도 안 봤다.** 출처가 지시한
  *"당신의 P99"* 는 **코퍼스 p99 = 풀 q≈0.53 ≈ 캠페인 2,514 tok = `STRATUM_TOKENS[0.50]`**.
- ★★그 층은 **측정된** 층이라 실측을 쓴다: 최악 309.5 ms ⇒ `chat` 탈락(초과 **3.2%**,
  metric cliff) · `rag`·`batch` 통과 ⇒ **`anchor = multiple` = "이 절차는 고를 수 없다"**.
  **`unique`는 오직 우리 풀의 p99에서만 나온다** ⇒ §4.5의 사슬이 아니라 **풀 선택이 답을
  정했다**(게이트 #86 설계층).
- ShareGPT를 빼면 **로컬 후보 0개**. 외부 후보(Azure 80%가 2K 이내 · BurstGPT
  short-dominated · LMSYS 평균 69.5 tok · ServeGen 2K 이하 집중)가 **독립적으로 같은 방향**
  ⇒ 일반 워크로드를 등록하면 `anchor = multiple`.
- ★진짜 관문은 데이터셋이 아니다: **좌표와 모집단은 같은 배치에서 와야 한다.**

### 6. 관련연구 조사 (`RELATEDWORK_LONGCTX_SLO_2026-09-07.md`, 392줄)

- SLO 임계 설정 유형 (a)–(f) 분류. ★**long-context 전략은 넷뿐**: S1 class별 완화 절대값
  (DistServe 요약 **15s**/LongBench · Mooncake 실배포 30s · MLPerf 405B 6s) · S2 무경쟁
  지연 배수 · **S3 TTFT를 SLO에서 제거**(Sarathi-Serve·LoongServe·Llumnix·커널 논문 전부) ·
  S4 프로파일 길이 곡선(**Etalon 단독**).
- **인터랙티브 100–400ms를 long-ctx에 적용한 논문 0건** ⇒ *"long-ctx 절대 SLO 표 부재"*
  확인. 감사 死因은 **문헌과 정합적**.
- ★**선행연구 위협(내가 1차 확인)**: Etalon(arXiv 2407.07000v2)이 *"TTFT is oblivious of
  prompt length"* · *"static SLO on TTFT … is not practical"* 를 적고, **격리 프로파일 →
  길이 곡선 fit → `D_p(L)`** 를 이미 처방한다. 남는 차이는 **용도**(Etalon은 fluidity
  메트릭의 deadline, 불가능 영역 선언 안 함).
- 논문 fit: OSDI/SOSP/NSDI **약** · EuroSys/ATC 약–중 · **MLSys 중(단 현 크기는 섹션)** ·
  워크숍 **강**.

## 코드·문서 변경 (전부 커밋됨)

| 경로 | 무엇이 왜 |
|---|---|
| `results/cp_baseline/af1_{prompts,client,probe.sbatch,analyze}.py` | AF-1 하네스 신규(게이트 #34에 따라 규칙층 `GO` 이후 작성) |
| `results/cp_baseline/af1_predicates.py` | `WARMUP_REQUESTS_PER_STRATUM = 2` DESIGN 상수 추가 |
| `results/cp_baseline/af1_selftest.py` | 위 상수의 양방향 무력 증명(`DESIGN_SUBST`) |
| `results/cp_baseline/af1_probe.sbatch` | 파티션 판독 **덧붙여** 수리(`CONFIGURED`/`REALISED`) |
| `results/cp_baseline/PREREG_AF1_2026-09-04.md` | 머리글이 rev5에 stale(코드·이력은 rev6) → 표기 정정 + 배너 |
| `results/cp_baseline/AF1_HARNESS_ADDENDA_2026-09-07.md` | A–F절(3반복·프롬프트 변이·워밍업·§9-1 반증·스모크2·동거) |
| `results/cp_baseline/RESULT_AF1_2026-09-07.{md,json}` | 판정 + 등록 산출 5–8 |
| `results/cp_baseline/PROPOSAL_STRATUM_GROUND_2026-09-07.md` + `_VERDICT_ACCEPTED.md` | 제안 rev1과 그 `NO-GO` 수용 기록 |
| `results/cp_baseline/audit_stratum_ground_2026-09-07/VERDICT.md` | 감사 판정서 317줄 |
| `results/cp_baseline/FOLLOWUP_STRATUM_2026-09-07.md` | (가) 1차 출처 판정 + (나) 사다리 전수 공표 |
| `results/cp_baseline/WORKLOAD_REVIEW_2026-09-07.md` | 제거 4건 해부 + 워크로드 후보 |
| `results/cp_baseline/DESIGNNOTE_LONGCTX_SAFETY_2026-09-07.md` | long-ctx 안전기준 재구성 검토(+ §1 정정 배너) |
| `results/cp_baseline/RELATEDWORK_LONGCTX_SLO_2026-09-07.md` | 관련연구 392줄 |
| `results/cp_baseline/DIRECTION_2026-09-07.md` | 방향 D1–D5 |
| `results/cp_baseline/af1_discarded/` + `README.md` | 채점 제외 5 round 격리(삭제 아님 — 게이트 #88) |
| **`reports/serving_slo_survey.md`** | ★**정본 오인용 정정 배너**(DistServe SLO scale 귀속) |

## 열린 항목 / 다음 세션 시작점

1. ★**D4 즉시**: `reports/longcontext_trace_plan.md` §4.3 *"연속 길이-정규화 임계는 계보
   없음 = ad-hoc"* 이 **부분 정정 대상**(Etalon `D_p(L)`·LoongServe normalized input
   latency 선례). 선형 `a+b·L` 기각과 처방은 유효. **doc-steward 이관.**
2. ★**D1의 규칙층 설계를 GPU 0으로 먼저 감사**(게이트 #34). sweep **격자 자체가 새 자유
   표면**이므로(게이트 #86) 이것도 감사 대상. 여기서 `CLAUDE.md` 게이트 2·3·6·7이 전부 걸린다.
3. 통과하면 D1 실행. **통과 못 하면 D5(트랙 축소)를 진지 검토** — **8회 연속 NO-GO는 설계가
   아니라 질문의 문제**라는 신호.
4. D2(`floor(L,c)`)·D3(16K/32K)은 **따로 사지 않는다** — sweep이 `c`와 `L`을 어차피 훑는다.
5. rev2를 쓴다면 미해소 死因 **U5**(격자가 답을 미리 정함)·**U6**(p99에서 밴드가 타이밍
   분산이 아니라 길이 폭에 지배 — addenda B)·**U1**(P90–P99는 범위)을 따로 답해야 한다.

## 미완·주의

- ⚠️**8,192 토큰 초과는 미측정**이다. 32K 바닥 ≈3,785 ms는 **외삽 ×5.5**이고
  *"32K에서 `batch_async`조차 불가"* 는 **주장으로 쓰면 안 된다**.
- ⚠️**L-10**: 동시성 1에서 `(64,44)` 미발화 ⇒ `∀arm`은 *"∀ 코드경로"* 지 *"∀ 자원 배분"*
  이 아니다. **경쟁 하 이식 금지.**
- ⚠️`WORKLOAD_REVIEW` §A.4의 4행 표는 **1행만 실측**이고 나머지는 적합 기반. **판정 아님.**
- ⚠️**arXiv 2607.05876**("floor-first" 어휘, 해석적 decode-step floor) **본문 미확인** —
  선점 위험이 남아 있다.
- ⚠️`RELATEDWORK`·`DESIGNNOTE`·`DIRECTION` **셋 다 claims-auditor를 안 거쳤다**(제안·검토
  문서라 판정이 아니지만, 정본으로 승격하려면 감사 필요).
- 방치된 job 없음(큐 비어 있음). 미커밋 없음(다른 세션 트랙 파일은 손대지 않았다).
