# 세션 핸드오프 — 2026-07-31 ~ 2026-08-02

## 이번 세션 요약

`catch-up`으로 시작해 E1(8B 프론티어) 캠페인의 남은 전제를 닫으려 했으나, **claims-auditor
감사가 이 세션에서 내가 세운 주장 대부분을 반증**했고 그 과정에서 하네스 결함 5건과
사전등록 공백 4건이 드러났다. 전부 수정·사전등록한 뒤 4개 job(3 arm 용량 스캔 +
batch-cap 실험)을 돌렸고, 그 결과가 **E1 설계 자체에 대한 판정**으로 이어졌다: 사전등록
사다리 {50,60,80}ms가 **네 arm 전부에서 헤드라인 룽을 못 만들고**, 그 측정이 이뤄진
설정(`--max-running-requests 48`)이 **미등록 상수이며 두 축을 반대로 왜곡**하고 있었음이
확인됐다. 본 스윕은 **미제출**이며, E1이 사전등록한 "설계상 이 질문에 도달할 수 없다"
분기로 갈 위험이 높다는 것이 이 세션의 실질 성과다. 오늘 결과는 **전부
claims-auditor 미통과 = 정본 인용 금지**.

---

## 결정·측정

### 1. claims-auditor 감사 (C-A ~ C-E) — 내 주장 다수 반증

| 주장 | 판정 |
|---|---|
| C-A seed divergence = 도착 draw의 성질 | **REFUTED** |
| C-B 하네스 버그 #7/#8 수정 | **PLAUSIBLE(조건부)** — 방향은 옳으나 커밋 메시지의 "일치" 주장이 거짓, 버그 #9 잔존 |
| C-C escalation 발동 | **PLAUSIBLE(조건부)** — 정성 견고, 숫자·표현 틀림 |
| C-D ITL 구속 rate ⟂ d92 off-cliff | **REFUTED** |
| C-E T8 = ITL-NONBINDING, C2 판정 불가 | **CONFIRMED**(근거 교체), 후반부는 NOT-YET-SUPPORTED |

- **C-A가 틀린 이유(재사용 가치 높음)**: 내가 근거로 든 `arrival_rps`는 측정값이 아니라
  seed로부터 **RNG replay로 재생성된 값**(`e1_capacity_scan.sbatch:200-207`)이라
  `(seed, n)`만의 결정론적 함수다. "5셀 전부 동일"은 **항등식**이고 서버에 대한 정보가 0.
  예측 2건도 실패 — rate 8은 도착률이 2.3%만 달라도 d92 TTFT가 3269 vs 6699로 갈리고,
  rate 12/16은 **부호가 반대**(도착률 높은 seed가 TTFT 더 낮음). 진짜 기전 =
  `bench_serving.py:1705`가 `random.seed()`도 걸고 `datasets/sharegpt.py:98`의
  `random.shuffle`이 그 스트림을 소비 ⇒ **seed가 프롬프트 집합 자체를 바꾼다**
  (offered input-tok/s 비 seed2/seed1: rate1 1.86 → rate12 0.77, 부호 반전이 여기서 발생).
  seed 비-pooling 결정은 유지, **이유만 교체**(두 seed = 서로 다른 워크로드).
- **C-D가 틀린 이유**: `ITL-NONBINDING: false`는 "15% 이내 근접"일 뿐 구속이 아니다.
  요청 단위로 세면 T8은 rate 1–32 전 구간·전 룽에서 **≥93%가 ITL 통과** ⇒ "ITL 구속 rate"가
  공집합이므로 배타성 명제는 **공허참**.
- **knee 정정**(첫 교차 기준): d16 12.6 / d24 16.0 / d44 16.0 / d54 **8.45** / d92 **2.80**.
  초판의 4.10은 "임계 아래 마지막 점"을 썼고 곡선이 단조가 아니라 두 정의가 크게 갈린다.
  견고한 것은 **d92 knee가 나머지보다 3배 이상 낮다는 순서**(세 임계 × 두 x축 불변).
- **"5셀 동시 off-cliff rate 부재"는 부정확** — rate≲2에선 d92 포함 전 셀이 plateau 위라
  §4.2의 escalation 조건은 엄밀히 발동하지 않았다. 성립하는 명제는 **"공통 off-cliff
  band(≲2–3 req/s)가 ITL 항이 움직이는 영역과 완전히 분리돼 있다"**.

### 2. 관측자 효과 게이트 (job 867298, 이미 완료돼 있던 것을 판정) — 조건부 통과

ABBA n=4 paired. d16은 전 지표 t95 CI가 0 포함. **d92만** `itl_p95` **+2.00%**
[+0.78, +3.21] · `itl_p99` −1.28% · `itl_mean` +0.56%가 0을 배제. 크기(≤2%)보다 **위치**가
문제 — 하필 결정 규칙이 임계하는 ITL-p95이고 D-격자 한쪽 끝에서만 난다.
⇒ **본 스윕은 `PDMUX_TRACE_FORCE_PREFILL=0`, pin 검증만 별도 짧은 ON 런으로 분리**
(사전등록 `DESIGN.md` §4.7.1). 그 job의 `PIN_CHECK`는 인자순서 버그로 전부 크래시해
pin 데이터 없음(paired summary는 별도 분석기 산출이라 무영향).

### 3. ★ batch-cap 실험 (job 870301, T8 × {d16,d44} × cap{48,96,192} × 4 seed)
    — 사전등록 branch 1 **CONFIRMED**

| cell | cap | ITL-p95 mean±sd | TTFT-p50 | frac ITL≤50 | frac ITL≤60 |
|---|---|---|---|---|---|
| d16 | 48 | **49.7 ± 0.4** | 612 ± 273 | 0.937 | 1.000 |
| d16 | 96 | **59.0 ± 2.9** | 114 ± 11 | 0.318 | 0.915 |
| d16 | 192 | **62.0 ± 0.8** | 121 ± 11 | 0.271 | 0.825 |
| d44 | 48 | 25.4 ± 0.6 | 193 ± 95 | 1.000 | 1.000 |
| d44 | 192 | 26.7 ± 1.2 | 111 ± 34 | 1.000 | 1.000 |

- seed-paired(n=4): d16 cap48→192 **ΔITL = +12.2 ± 0.7 ms**(4 seed 전부 +11.7~+13.3),
  d44 **+1.3 ± 0.7**. cap192에서 d16이 60ms를 **4/4 seed 초과**.
- batch가 실제로 자람: d16 48 → **60–75**, d44 48 → 50–55. cap192에서도 60–75에서 정지
  ⇒ **cap 48만 실제로 구속**, 96 이상은 도착·서비스율이 정하는 자연 평형.
- **판정: 셀별 ITL "천장"은 모델 성질이 아니라 설정 성질.**
- ★ **부수 발견이 더 중요할 수 있음**: cap 48은 d16의 TTFT를 **612 → 114 ms (5.4×)** 로
  악화시키고 있었다. cap이 admission을 조여 큐를 밀리게 하는 대신 batch를 작게 유지해
  ITL을 좋아 보이게 만든 것 ⇒ **`--max-running-requests`는 D split과 나란한 숨은 세 번째
  정책 손잡이**로 작동했고 **두 축을 반대 방향으로 동시에 왜곡**했다.
- 원자료: `results/s8_frontier/e1bcap_T8_870301_result.txt`. **T8(순수 Transformer) 2셀만
  측정** — 이 scope를 뗀 인용 금지.

### 4. 3 arm 용량 스캔 (jobs 870295=M8 / 870296=Ha8 / 870297=Hs8) — 100/100 probe, 오류 0

knee(첫 교차, plateau 2×, pooled realized `arrival_rps`):

| arm | d16 | d24 | d44 | d54 | **d92** | 구속 셀 |
|---|---|---|---|---|---|---|
| T8 | 12.6 | 16.0 | 16.0 | 8.45 | **2.80** | d92 |
| M8 | 5.60 | 5.60 | 5.60 | 4.20 | **2.80** | d92 |
| Ha8 | 5.60 | 5.60 | 4.20 | 3.08 | **2.80** | d92 |
| Hs8 | 9.09* | 12.6 | 9.09 | 8.45 | **2.80** | d92 |

- **d92가 네 arm 전부에서 구속 셀**이고 knee 2.80으로 동일 ⇒ 공통 off-cliff 상한 2.80 req/s.
- ⚠️ 내가 §4.3.5(b)에 추측으로 적은 **"느린 decode arm은 반대로 d16을 잃을 것"은 지지되지
  않음** — d92의 prefill 16 SM이 모델과 무관하게 먼저 무너진다. (사전등록에 추측으로
  표시해둔 덕에 손해 없음.)
- `*` **Hs8 d16 knee 9.09는 견고하지 않음**: 임계 168.6에 대해 172(2% 초과)인 한 점이
  정하고 다음 점은 139로 회귀. 같은 셀 `arr=12.61 seed=1`에 **이상치 1건**(TTFT p50
  6904ms · ITL-p95 6659.7ms인데 같은 probe의 ITL p50은 76.2ms; 옆 seed2는 **더 높은**
  부하 14.07에서 TTFT p50 254ms — 27× 불일치). 기전 추정 =
  `--chunked-prefill-size -1`이라 seed가 뽑은 단일 장문이 prefill 윈도우를 점유.

### 5. ★ 룽 분류 — 네 arm **전부** `HEADLINE-ELIGIBLE RUNGS = NONE`

| arm | 50ms | 60ms | 80ms |
|---|---|---|---|
| T8 | CLIFF-HAZARD (0.9%) | NONBINDING | NONBINDING |
| Hs8 | CLIFF-HAZARD (14%) | NONBINDING | NONBINDING |
| M8 | **ALWAYS-BINDING** | CLIFF-HAZARD (3.9%) | CLIFF-HAZARD (11%) |
| **Ha8** | **ALWAYS-BINDING** | **ALWAYS-BINDING** | **ALWAYS-BINDING** |

**이 세션에 신설한 `ITL-ALWAYS-BINDING` 플래그가 등록 몇 시간 뒤 Ha8에서 정확히
발동했다.** 없었다면 Ha8의 conjunctive goodput 0이 "어떤 D도 못 이김" → 레버
net-negative로 읽혔을 것이다. `degenerate_goodput_guard()`가 결정 규칙을 차단한다.

### 6. ★★ KV/메모리 — 항등식을 증거로 착각한 사례 (기록 가치 높음)

- batch-cap에서 `kv_occupancy` 0.018–0.031을 보고 **"메모리는 한 번도 구속하지 않음"**이라
  썼는데, batch-cap은 **T8 = 순수 Transformer 하나만** 측정했다. **scope 오류.**
- hybrid arm telemetry를 보면 **`kv_mamba_occupancy` = 1.0000 = 포화**(M8/Ha8/Hs8 전부),
  attention KV(`kv_full_occupancy`)는 0.011–0.279.
- **그러나 이건 메모리 구속의 증거가 아니라 항등식**:
  `model_runner_kv_cache_mixin.py:223-229`가 `disable_radix_cache ∧ max_running_requests`
  일 때 **`max_mamba_cache_size = max_running_requests`** 로 설정한다. 이 캠페인이 정확히
  그 조건 ⇒ pool 크기 = 48 ⇒ `kv_mamba_occupancy = batch/48` ⇒ batch가 cap에 닿으면
  **정의상 1.0**. 가용 메모리에 맞추는 `else` 분기(ratio 기반)는 한 번도 타지 않았다.
- ⇒ **이 데이터로는 hybrid에서 "메모리 vs 스케줄링" 중 무엇이 구속적인지 판정 불가.**
  지지되는 것은 좁다: *ctx ≤4k ShareGPT에서 attention KV pool은 어느 arm에서도 구속
  근처에 없었고(0.011–0.279), mamba pool은 독립 측정이 아니었으며, 실제로 구속한 것은
  admission cap이었다.*
- ★★ **새 교락 발견**: `--max-running-requests`는 **arm 계열마다 다른 손잡이**다 —
  T8에서는 admission 하나, hybrid에서는 **admission + mamba state pool 크기**를 동시에
  움직인다. ⇒ **batch-cap의 T8 결과는 hybrid arm으로 이전 불가**이고, 내가 세션 중
  제시한 "cap↑이 T8·Hs8을 살리고 M8·Ha8을 악화시킨다"는 **철회**한다.
- 해소책: `--max-mamba-cache-size`를 전 arm 공통 상수로 **명시 고정**(위 코드의 첫 분기)
  ⇒ pool이 cap에서 분리되고 cap이 다시 순수 admission 손잡이가 된다. **E1 correctness에
  필요 = 다음 세션 사전등록 대상.**

### 7. 하네스 결함 5건 (전부 수정)

| # | 결함 | 영향 |
|---|---|---|
| traceforce | `PIN_CHECK` 인자순서가 bug#6 이전 순서 → `float(argv[2])`가 경로 문자열 | 867298의 pin 전부 크래시 |
| #7 | capscan regex에 `_s<seed>` 없음 → 867231의 100 probe 전부 놓치고 **옛 스모크로 조용히 대체** | 완성된 형태의 **틀린 판정** 출력 |
| #7b | regex가 `job`을 캡처만 하고 미사용 → job 간 병합 | 3.5분 스모크가 arm-level 판정을 찍음 |
| #8 | capscan 경로에 warmup 폐기 미적용 | TTFT p95 상향 편향(§4.3.3 마진이 평가하는 값) |
| #9 | sbatch는 nearest-rank, analyzer는 선형보간 | 같은 33 표본에서 **33.6%** 차이, d92 판정이 실제로 뒤집힘 |

추가로 **`PROBE_TARGET_S`가 문서화된 20이 아니라 기록되지 않은 8로** 867231이 돌았음이
드러났다. 이 때문에 처음 제출한 3 job(870287–289)은 기본값 20을 쓰게 돼 있어 **arm 간
비교 불가 + 시간 초과 위험**이었고, **취소 후 전 파라미터를 명시해 재제출**(870295–297).

---

## 코드·문서 변경 (전부 커밋됨, **push 없음**)

| 커밋 | 내용 |
|---|---|
| `d4fa672` | traceforce `PIN_CHECK` 인자순서 수정 + `DESIGN.md` §4.7.1(force-trace를 pin 검증 전용으로 분리) 사전등록 |
| `3fc74b5` | 정본의 "분석 산출물 0건" 오기 정정 + escalation 기록 + 미추적 핸드오프 추가 |
| `dfbe84e` | capscan 버그 #7/#8 |
| `ed3504b` | 반증된 seed 근거 교체, 버그 #9(추정기 병기+불일치 플래그)·#7b(job 필터) 노출, p95 bootstrap CI 병기 |
| `d1f157a` | claims-auditor 판정 정본 반영(knee 정정, batch cap 미등록, decode duty cycle) |
| `5d1094a` | `decode_duty_check.py`(decode 측 duty-cycle 게이트) + `batchcap.sbatch` |
| `c9d86a2` | `DESIGN.md` §4.3.5/§4.3.6 사전등록 + `e1_analyze.py` 구현(ITL-ALWAYS-BINDING, 룽 4분류, degenerate guard, on-cliff 제외) |

**신규 파일**: `results/s8_frontier/decode_duty_check.py` · `results/s8_frontier/batchcap.sbatch`.
`decode_duty_check.py`는 prefill 게이트와 **구조적으로 동일한 추정량**(시간가중,
episode-cluster bootstrap, `n_episodes`=유효 표본수, `n_episodes==1`은 UNRELIABLE)으로
만들어 감사자 수치를 독립 재현: duty_frac d16 **0.110** / d24 0.112 / d44 **0.166** /
d54 0.201 / d92 **0.518**. **의도적으로 cite-blocking 아님** — §5가 게이트 2개를
사전등록했는데 데이터를 본 뒤 세 번째 임계를 더하면 게이트를 데이터에서 고르는 것이 된다.

**미커밋**: 없음(작업 트리 clean). **4개 job의 원자료·result.txt는 커밋 대상 아님**
(`results/` 관례대로 디스크에만).

---

## 열린 항목 / 다음 세션 시작점

1. **claims-auditor 재회부 (최우선)** — 오늘 §3·§6의 주장은 강하고 **전부 미감사**다.
   반증 대상: (i) batch-cap이 **1 arm·2셀**만 측정했는데 "천장 = 설정 성질"로 일반화한 것,
   (ii) cap 96과 192가 사실상 동일하다는 해석, (iii) mamba pool 항등식 논증
   (`disable_radix_cache` 경로가 실제로 탔는지 서버 로그로 확인 필요), (iv) 네 arm
   `HEADLINE NONE`이 cap 아티팩트라는 추론, (v) Hs8 이상치가 knee 추정을 오염시키는 정도.
2. **`--max-mamba-cache-size` 사전등록** — E1 correctness에 필요(§6). 전 arm 공통 상수로
   고정해 cap을 순수 admission 손잡이로 되돌린다. `DESIGN.md` §4.3.6에 추가.
3. **E1 본 스윕 제출 여부 판단** — 아래 세 가지가 동시에 성립한다:
   (a) 사전등록 사다리가 as-run 설정에서 **네 arm 전부 판정 불가**,
   (b) 그 as-run 설정 자체가 **왜곡으로 증명됨**(cap 48),
   (c) 제외 규칙(d92 knee 2.80, 전 arm)이 어떤 동작점에서도 **decode-rich 끝을 제거** —
   C2의 레버가 사는 바로 그 끝.
   ⇒ **E1은 §4.3.5(b)가 사전등록한 "설계상 이 질문에 도달할 수 없다" 판정으로 갈 위험이
   높다.** 이건 실패가 아니라 미리 적어둔 분기이며, **본 스윕에 GPU를 쓰기 전에 알아낸
   것**이 이 세션의 실질 성과다.
4. **cap을 사전등록 파라미터로 승격할지** — "batch가 cap-bound가 아닌 최소 cap"처럼
   데이터 독립적 규칙으로(현 측정상 96). **단 §6의 교락 때문에 2번을 먼저 해야 한다.**
   arm별 cap 튜닝은 손잡이만 바꾼 SLO 쇼핑이므로 명시적으로 금지할 것.
5. **신규 방향 후보 — 메모리 분할 축** (E1 전망이 어두운 만큼 검토 가치 있음).
   `mamba_full_memory_ratio`는 **SSM state pool과 attention KV pool 사이의 메모리 분할**로,
   Transformer엔 없는 **hybrid 고유의 강제된 자유도**다. 이 프로젝트가 다룬 건 compute
   분할(green-context SM split)뿐이고 메모리 분할은 미착수. long-ctx(16k–32k)에서
   attention KV는 자라고 mamba state는 안 자라므로 거기서 두 계열이 갈린다 —
   **"hybrid에서 메모리 vs 스케줄링"이 비로소 측정 가능한 질문이 되는 지점.**
   `venue-strategist`와 함께 SM 분할 negative 결과들과 어떻게 엮이는지 판단할 것.
6. **`s8p_prefill`을 claims-auditor에** (이전 세션 이월, 미해소).
7. **`s8p_analyze.py` 버그 2건 수정** (이전 세션 이월, 미해소).

---

## 미완·주의

- **오늘 결과는 전부 claims-auditor 미통과 = 정본·논문 인용 금지.** `PROJECT_STATUS.md`에
  들어간 것은 이전 감사(C1/C2 계열)를 통과한 항목과, 오늘 감사가 **직접 판정한** 항목뿐이다.
- **batch-cap은 T8(순수 Transformer) 2셀만.** "천장 = 설정 성질"을 hybrid로 옮기지 마라 —
  §6의 cap–mamba 교락 때문에 특히 위험하다.
- **`--max-running-requests`는 arm 계열마다 다른 손잡이.** 이 사실을 빼고 cap 관련 수치를
  인용하면 안 된다.
- **decode duty-cycle 수치(0.110–0.518)는 1차 근사** — pin 게이트와 같은
  `runtime_snapshot` 샘플링을 쓰므로 값 자체가 재측정 대상이다. sampling에 둔감한 것은
  **순서와 D에 대한 단조 공변**이고, 그게 이걸 반올림 오차가 아니라 교락으로 만든다.
- **Hs8 `arr=12.61 seed=1` 이상치** 미조사. knee 9.09의 견고성에 영향.
- 실행 중 job **없음**(4개 전부 COMPLETED, squeue 비어 있음).
- `PROBE_TARGET_S=8`은 문서화된 20이 아니다. 867231·870295–297·870301이 전부 8로 돌았고
  이제 submit 라인에 명시돼 있다. **비교 대상 런을 추가할 때 반드시 8로 맞출 것.**
- **이 세션의 메타 교훈(가장 재사용 가치 높음)**: 철회된 8건 중 **3건이 항등식·정의를
  증거로 착각**한 것이고(`arrival_rps`, `kv_mamba_occupancy`, 배타성 명제), **3건이 한
  arm/셀에서 잰 것을 일반화**한 것이다. 방법론 게이트 #4(집계 단위 선확정)와 같은 뿌리 —
  **"이 양이 내가 재려는 것과 논리적으로 독립인가"를 먼저 물어야 한다.**
