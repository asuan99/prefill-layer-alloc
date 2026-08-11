# PREREG — Gate 2-S **E1 addendum**: 전제 결정량 `max(decode_running_batch_size)`를
# **이 캠페인 자신의 셀에서 n=10 직접 산출**

작성: 2026-08-11 · 작성자: result-analyst · 대상 job: **877756**(Zamba2-2.7B r2·r3) ·
**877757**(Granite-4.0-h-micro-base r3·r4) · GPU 증분 **0**(기존 telemetry 재집계)

> 이 문서는 `PREREG_GATE2S_2026-08-09.md`의 **addendum**이다. 원문을 덮어쓰지 않는다.
> 원 사전등록의 판정 규칙(§4 primary, §5 앵커, §5.6.1 명명, §6.4 F-계열,
> §8.9.1 단방향 반증기, Holm, 9-셀 좌표)은 **한 글자도 바꾸지 않는다.**

---

## ★ 0. 최상단 경고 — **이 등록은 blind가 아니다 (post-hoc, not blind)**

**2026-08-11 claims-auditor가 반증 시도 과정에서 이 문서가 등록하려는 바로 그 값을
이미 산출했다.** 감사자가 보고한 값(이 문서 작성 시점에 작성자가 알고 있는 값):

| 셀 | 감사자가 이미 산출한 값 |
|---|---|
| Granite r3 | pooled `max(decode_bs)` = **10** (n=10) |
| Granite r4 | pooled `max(decode_bs)` = **13** (n=10) |
| Granite A4 rep별 max | **[9, 13]** 범위 |
| Zamba2 r2·r3 | (감사자 보고에 없음 — 작성자 미인지) |

감사자 자신이 **"이 수치를 승격의 licence로 쓰면 방법론 게이트 #8/#20 위반"**이라고
명시했다. 이 addendum은 **그 위반을 피하는 유일한 길이 사후 인지를 공개 선언하는
것**이라는 전제 위에 있다. 따라서:

1. **이 문서는 "결과를 보기 전에 고정된 규칙"이 아니다.** Granite 2셀에 대해 규칙은
   **이미 알려진 값과 정합하도록 선택되었을 수 있다**(작성자는 그러지 않았다고
   주장하지만, 그 주장은 검증 불가능하다 — 그것이 blind의 존재 이유다).
2. ⇒ **이 addendum이 산출하는 Granite r3·r4의 상태는 "사후 인지 상태에서 등록된
   규칙에 의한 상태"로만 인용된다.** 등급어를 낮추는 것은 선택이 아니라 의무다.
3. **Zamba2 r2·r3에 대해서는 작성자가 값을 모른다** — 그 2셀에 한해 이 규칙은
   사전(prospective)이다. 이 비대칭을 결과 보고에 반드시 병기한다.
4. **문턱(36)만은 사후 선택이 아니다** — `decode_bs_divisor`는 config 파일 상수이며
   Gate 1(2026-08-06)·G1-b·G1-c가 모두 같은 값을 썼다(§2 참조). 새 자유모수 0.

⚠️ **사후 인지의 자백은 면죄가 아니다**(방법론 게이트 #9 "자수는 면죄 아님").
이 문서가 하는 일은 위반을 없애는 것이 아니라 **위반의 크기를 관측 가능하게 만드는
것**뿐이다. 재감사자는 §7의 "이 addendum이 스스로 무력화되는 조건"부터 읽어라.

---

## ★ 1. 왜 새 통계량인가 — §8.9.1의 단방향성은 **자동 이전되지 않는다**

`PREREG_GATE2S §8.9.1`은 **"통과가 구조적으로 불가능하고 실패만 가능한"** 장치다.
그 안전 불변식(`g2s_analyze.py:626-628`, `FALSIFIER_STATES = ("refuted","unchanged")`)은
**`frac(idx2)`(pop A 시간가중 stream_index 히스토그램)에 대한 규정**이다.

이 addendum이 산출하는 `max(decode_running_batch_size)`는 **다른 통계량**이며,
**VERIFIED를 낼 수 있다.** 즉:

> ★ **§8.9.1의 단방향성(저자에게 유리해질 경로가 구조적으로 없음)은 이 통계량에
> 이전되지 않는다. 이 addendum은 저자에게 유리해질 수 있는 장치이며, 따라서
> `PREREG_GATE2S` 본문의 4차 감사를 통과한 안전성을 상속하지 않는다 —
> ★ **재감사 대상이다.**

이 문서는 그 재감사를 요청하며, 감사 전에는 아래 §6의 어떤 문장도 정본에 반영하지
않는다.

**§8.9.1을 대체하지도, 수정하지도 않는다.** §8.9.1은 그대로 돌아가고 그대로
`unchanged`를 냈다(4셀 전부, `g2s_report_*.json:premise_falsifier`). 이 addendum은
**그 옆에 다른 통계량을 나란히 놓을 뿐**이다.

---

## 2. 결정량과 그 출처 (새 자유모수 **0**)

### 2.1 통계량 — 생산자 코드에서 **동사(verbatim) 수입**

```
gate1b_analyze.py:135-137
    def max_decode_bs(rows):
        vals = [e.get("decode_running_batch_size", 0) or 0 for e in rows]
        return max(vals) if vals else None
```
- 집계 대상 행: **pop A가 아니라 그 window의 benchmark-phase 전 행**
  (`gate1b_analyze.py:8-9` 모듈 docstring: *"over ALL benchmark-phase rows in the
  window, not just population A"*).
- 행 필터: `gate1b_analyze.py:55` `load_rows(..., require_phase="benchmark")`
  ⇒ `event == "runtime_snapshot"` ∧ `phase == "benchmark"`.
- `gate1c_analyze.py:147`이 같은 함수를 그대로 복제해 쓰고 있고, rev22가 인용한
  Granite 값(10/10)이 그 함수의 출력이다 ⇒ **비교 가능성 보장.**

### 2.2 문턱 — config 상수, 사후 선택 아님

- `pdmux_a100_smoke.yml`: `sm_group_num: 4`, `decode_bs_divisor: 36`,
  `manual_divisions` **없음**.
- 선택자(`src/multiplex/multiplexing_mixin.py:893-909`, decode 비어있지 않고
  prefill 동거하는 분기):
  `stream_idx = max(1, min(real_sm_group_num-2, decode_bs*(real_sm_group_num-2) // decode_bs_divisor))`
  `real_sm_group_num-2 = 2`, `decode_bs_divisor = 36`
  ⇒ `stream_idx = max(1, min(2, decode_bs // 18))`
  ⇒ **`stream_idx == 2` ⟺ `decode_bs ≥ 36`.**
- 문턱 상수: `gate1c_analyze.py:60` `DECODE_BS_DIVISOR = 36`(같은 값, 수입).
- A4(`agnostic`)는 `g2s_run.sbatch:111` — `PDMUX_R2_POLICY` 미설정 ⇒ sticky 분기
  (`:889-891`) 미진입, 위 자동 선택자 분기가 실제로 실행되는 arm이다.

**⇒ 이 addendum이 도입하는 새 자유모수 = 0개.** (문턱·통계량·행 필터 전부 수입)

### 2.3 ★ 항등식 자백 — **경험적 내용은 하나뿐이다**

(CONSENSUS rev22 §1-1 G1-c 블록 해제 조건 ③ / §3 항목41 / `PROJECT_STATUS` 게이트 #27)

`idx == 2 ⟺ decode_bs ≥ 36`은 **코드 항등식**이다. 따라서

> `max(decode_bs) < 36` ⟹ (샘플된 모든 순간에) `frac(idx2) = 0`

이며, 그 역방향의 정보는 없다. ⇒ **`max_decode_bs < 36`과 `frac_5454 = 0`은
독립적인 두 증거가 아니다. 하나의 경험적 사실을 두 번 본 것이다.**

- 이 addendum의 보고문·요약·표에서 두 수치를 **"각각 전제를 지지한다"**는 형태로
  병기하는 것을 **금지**한다.
- `frac(idx2)`는 §8.9.1이 이미 산출했으므로 이 addendum은 **재계산하지 않는다**
  (재계산은 항등식을 두 번 세는 것이다). 참조로만 인용한다.
- 경험적 내용의 **정본 표현은 하나**: *"동거 구간의 realized decode_bs가 샘플된
  어떤 순간에도 문턱 36에 도달하지 않았다."*

### 2.4 ★ 결정량은 **pooled max**이며, rep별 max는 **독립 검정이 아니다**

`max`는 결합적이므로 `M_cell ≡ max_r M_rep`이다. 즉

> **"pooled max < 36"과 "모든 rep에서 rep-max < 36"은 같은 명제다.**

⇒ **결정량 = pooled `M_cell`** 로 사전 지정한다. rep별 max는 **여유(headroom)의
분포를 보이는 서술량**이며, **두 번째 검정으로 세지 않는다**(§2.3과 같은 종류의
이중 계상 금지). 두 값이 다르게 나오면 그것은 구현 버그이며 즉시 UNDETERMINED다
(§4 무결성 규칙 I5).

### 2.5 ★ 결정량의 밀도 의존성 — 편향의 방향이 **저자에게 유리하다**

(CONSENSUS §3 항목41 수입)

telemetry는 `PDMUX_DUAL_WORKER_TRACE_EVERY = 32`로 **32 스케줄러 반복마다 1 샘플**이다.
샘플된 max는 참 max의 **하한**이다:

> `M_observed ≤ M_true` (항상)

⇒ `M_observed < 36`은 **"문턱을 넘은 적이 없다"의 증명이 아니다.**
**"샘플된 어떤 순간에도 넘지 않았다"**일 뿐이다. 그리고 이 편향의 방향은
**VERIFIED 쪽(저자에게 유리한 쪽)** 이다. 따라서:

- **VERIFIED 상태의 명칭 자체에 해상도를 박아 넣는다**:
  `VERIFIED_AT_SAMPLED_INSTANTS`. 축약형("VERIFIED") 단독 인용 금지.
- **여유(36 − M_cell)를 항상 병기**한다. 여유가 작을수록 하한 편향이 결론을
  뒤집을 확률이 크다.
- ⚠️ **G1-c(877974)와의 밀도 차이에 대한 정직한 진술**: G1-c는
  `PDMUX_TRACE_FORCE_PREFILL=1`(동거 구간 강제 emit)이라 **pop A 밀도가 32배**다.
  그러나 **이 addendum의 통계량은 pop A로 조건화하지 않는다**(§2.1) — 전 행 대상이고
  두 캠페인의 기저 케이던스(`TRACE_EVERY=32`)는 **같다**. ⇒ `frac(idx2)`에 대해
  §8.9.1이 겪었던 "1/32 밀도" 페널티는 **이 통계량에는 같은 크기로 적용되지 않는다.**
  이것이 이 addendum이 존재할 수 있는 이유다. **그러나 페널티가 0이라는 뜻은
  아니다** — decode_bs 첨두가 어느 population에서 발생하든 두 캠페인 모두 1/32로만
  본다. G1-c 대비 이 addendum의 우위는 **밀도가 아니라 반복수(n=1 → n=10)와
  셀 일치(복제 격자 → 그 캠페인 그 셀)** 에서 온다. 이것을 밀도 우위로 오독하지 마라.

---

## 3. rep 경계를 어떻게 가르는가 (사전 명시)

셀 파일 `g2s_<tag>_telemetry_agnostic_r<rate>_<jobid>.jsonl`은 boot 단위 파일 10개를
**`sorted(glob(...rep*...))` 순서**(⇒ rep10, rep1, rep2, …, rep9)로 이어붙인 것이며
시간순이 아니다(`g2s_run.sbatch:274-289`). 레코드 자체에 rep 필드는 **없다**
(스키마 확인: `run_id`·`workload_id`는 있으나 rep 식별자 없음).

**⇒ gap 기반 휴리스틱을 쓰지 않는다.** 경계는 하네스가 **측정 시점에 기록한**
오프셋으로 정확히 복원한다(§9.1 rev6 기록 — 사후 재구성이 아니다):

```
for tel in sorted(glob(g2s_{tag}_telemetry_agnostic_rep*_{job}.jsonl)):
    s0, s1 = json.load(open(tel -> _teloffset_ .json))["r{rate}"]
    rep_rows = open(tel).readlines()[s0:s1]
```

- **1차 산출(primary)**: 위 per-boot 슬라이스에서 rep별로 직접 계산. 셀 파일의
  이어붙인 순서에 **의존하지 않는다.**
- **2차 대조(cross-check)**: 셀 파일 전체에서 pooled max를 따로 계산.
  §2.4의 항등식에 의해 두 값은 **반드시 같아야 한다.** 다르면 UNDETERMINED.
- **길이 대조**: Σ(슬라이스 길이) == 셀 파일 줄 수. 다르면 UNDETERMINED.
- rep이 10개 미만이면 그 셀은 UNDETERMINED.

---

## 4. 무결성 규칙 (판정 전 통과 필수)

| ID | 검사 | 실패 시 |
|---|---|---|
| I1 | JSON 파손 라인 수 == 0 (셀 파일·per-boot 슬라이스 양쪽) | UNDETERMINED |
| I2 | `dropped_events` 최대값 == 0 (셀 내 전 행) | UNDETERMINED |
| I3 | Σ(rep 슬라이스 줄 수) == 셀 파일 줄 수 | UNDETERMINED |
| I4 | rep 수 == 10 | UNDETERMINED |
| I5 | per-boot 경로 pooled max == 셀 파일 경로 pooled max | UNDETERMINED |
| I6 | rep 내부 `sample_index` **역전 0건** · 중복 0건 | UNDETERMINED |
| I7 | 전 행 `event=="runtime_snapshot"` ∧ `phase=="benchmark"` 비율 == 1.000 | 기록 후 계속(서술) |

- **I2를 "임의 문턱"이 아니라 구조적 경계로 둔다**: 드롭된 행은 **관측되지 않은
  decode_bs**이고, 그 값을 위로 bound할 방법이 없다. 드롭이 1건이라도 있으면
  하한 편향(§2.5)이 통제 불능이 된다. 문턱 0은 유일한 비임의 선택이다.
- **셀 파일 전체(rep 이어붙인 것)의 `sample_index` 역전은 I6에서 제외**한다 —
  glob 순서 때문에 경계에서 역전이 **정확히 9회** 나타나는 것이 정상이다.
  실제 관측 역전 수가 9가 아니면 기록해 보고한다(판정 아님, 서술).
- **서술 전용 진단(판정에 미사용)**: 셀별 스냅샷 수 · rep별 줄 수 ·
  rep 내부 `sample_index` gap의 중앙/최대 · rep **경계** 시간 gap ·
  rep 경계에 인접한 행이 pop A인 건수(§8.9.1의 `t_total` 오염 규모 참고용,
  **`frac`을 재계산하지 않는다**).

---

## 5. 판정 규칙 (**값을 보기 전에 고정** — 단, §0의 사후 인지 하에서)

셀 c에 대해, `M_c` = pooled `max(decode_running_batch_size)`:

| 조건 | 상태 |
|---|---|
| 무결성 I1–I6 전부 통과 **∧** `M_c < 36` | ★ **`VERIFIED_AT_SAMPLED_INSTANTS`** — 그 셀에서 §8.9 전제가 **이 캠페인 자신의 데이터로 직접** 성립. 여유 `36 − M_c` 병기 의무. |
| §8.9.1이 이미 낸 `frac(idx2) ≥ 0.01` | **REFUTED** — 단 이 경로는 **§8.9.1 소관이며 이 addendum이 만들지 않는다**(이 캠페인 4셀 전부 `frac(idx2)=0.0000`, 발화 없음) |
| 무결성 I1–I6 중 하나라도 실패 | **UNDETERMINED**(사유 명시) |
| `M_c ≥ 36` **∧** `frac(idx2) < 0.01` | **UNDETERMINED** — 문턱이 샘플된 순간에 넘겼으나 §8.9.1 반증 문턱은 미발화. **VERIFIED 아님, REFUTED 아님.** |

★ **이 addendum은 새 REFUTED 경로를 만들지 않는다.** 반증 권한은 §8.9.1이 계속
독점한다(그 단방향 장치를 약화시키지 않기 위해서다).

★ **셀 단위로만 판정한다.** 4셀을 묶은 요약 상태(예: "4셀 전부 검증")를 만들지
않는다 — 셀별 상태를 나란히 쓴다. (게이트 #16 거울상: 격자 전체 무제한 주장 금지)

---

## 6. ★ 이 addendum이 **바꾸는 것**과 **안 바꾸는 것** (사전 선언)

### 6.1 바꿀 수 있는 것 — **증거 등급 하나뿐**

전제 라벨의 **근거 문구**가 바뀔 수 있다:

| | 현재(rev22) 근거 | E1 후 근거(VERIFIED 시) |
|---|---|---|
| Zamba2 r2·r3 | Gate 1 rev15 / **job 875293**(G1-b), **n=1**, 873944 **복제 격자** | + 이 캠페인 **877756**, **그 셀**, **A4**, **n=10 rep** 직접 관측 |
| Granite r3·r4 | G1-c / **job 877974**, **n=1**, 873945 **복제 격자** (rev22 해제조건 ④가 "Gate 2-S 셀에서의 직접 관측 아님"을 명시) | + 이 캠페인 **877757**, **그 셀**, **A4**, **n=10 rep** 직접 관측 |

⇒ 사라지는 것은 **"복제 격자에서 이전(transfer)"이라는 논증 단계**와
**rev22 해제조건 ④의 "n=1 · Gate 2-S 셀에서의 직접 관측 아님"** 이라는 등급 제한,
그리고 **모델 간 기준 비대칭**(Zamba2·Granite 양쪽 모두 같은 방식으로 격상되므로).

### 6.2 ★ 바꾸지 **않는** 것 (전부 불변 — 하나라도 어기면 이 addendum 무효)

1. **성능 판정 전부 불변.** Δ = C−T′ = **+13.95 / +24.88 / +16.04 / +19.11 ms**,
   4셀 Holm 후 최대 보정 p = 1.88e-06, 9-셀 좌표 16블록 전부 `S1-C`,
   TTFT p95 악화 −183~−536 ms — **한 글자도 바뀌지 않는다.**
   (코드 근거: `premise` 라벨은 `g2s_analyze.py:1157-1161`에서 `nine_cell`·
   `gate_label` 산출에 **미입력**이다.)
2. ★★ **크기 인용 자격에 대해 이 문서는 아무 말도 하지 않는다.** 그것은 **F-계열
   gate 소관**이며 이 addendum은 그 층을 건드리지 않는다. "크기 인용 가능 셀"의
   수·정체에 대한 어떤 진술도 이 문서에서 도출되지 않는다.
   (2026-08-11 세션이 정확히 이 혼동으로 오판했다 — 방법론 게이트 #28.)
3. **§8.9.1 단방향 반증기 불변** — 규칙·출력·`unchanged` 결과 그대로.
   `g2s_analyze.py`는 **수정하지 않는다**(rev20 결과를 낸 감사 통과 스코어러).
4. **§5.6.1 명명 규칙 불변**, rev22의 조건부 해제 6조건·금지 8건 **불변**.
   특히 금지 ④(**"(54,54)는 도달 불가"** 금지 — 부하가 커지면 도달하며 Zamba2 r6에서
   실제 발생)와 ⑧(무제한 서술 금지)은 이 addendum에도 그대로 적용된다.
5. **selector-level 서술 한정** — 하드웨어 SM 프로브(S3) 미실행. "물리적으로
   prefill 74 SM / decode 34 SM"으로 쓰지 않는다.
6. **§1-1 귀속 불변** — A4-vs-fused 격차의 성분·기여분·분해로 서술 금지,
   "PD 분리 자체가 원인" 승격 금지.

### 6.3 ★ 인용 금지 (이 addendum의 산출물에 대해)

- 이 telemetry의 **TTFT / TPOT / ITL / goodput 어떤 수치도 인용 금지.**
- **pop C 절대·상대 분수 인용 금지.**
- **pop A `t_total_s` 인용 금지**(경계 dt 5–29% 상향 편향, rev22 등재).
- **`M_cell`을 "부하 수준"·"용량"·"동시성"의 대리 지표로 쓰지 마라** — 이것은
  선택자 문턱 도달 여부만을 답하는 양이다.
- **모델 간·rate 간 `M` 비교로 성능·거동 주장 금지**(예: "Granite r4가 r3보다
  batch가 크다") — 이 addendum은 그 비교를 위해 설계되지 않았다.

---

## 7. ★ 이 addendum이 스스로 무력화되는 조건 (재감사자가 먼저 읽을 것)

1. **§0의 사후 인지가 치명적이라고 판단되면** — Granite 2셀에 대해서는 이
   addendum을 **파기하고**, 그 2셀의 근거를 rev22 상태(G1-c, n=1)로 되돌려라.
   Zamba2 2셀은 값 미인지 상태에서 등록되었으므로 분리해 판단할 수 있다.
2. **§2.5의 하한 편향이 여유 대비 크다고 판단되면**(예: `36 − M_cell`이 작으면)
   — VERIFIED를 UNDETERMINED로 강등하라. 이 addendum은 그 강등을 위한 사전
   등록된 정량 문턱을 **가지고 있지 않다**(새 자유모수를 만들지 않기 위해).
   **이것은 이 설계의 알려진 구멍이며, 숨기지 않는다.**
3. **§1의 재감사 요건** — VERIFIED를 낼 수 있는 장치이므로 §8.9.1의 안전성을
   상속하지 않는다. 감사 전 정본 반영 금지.
4. **payoff가 코드로 검증되지 않았다면**(CONSENSUS §3 항목42) — 이 addendum의
   payoff는 §6.1의 "증거 등급 한 칸"뿐이라고 주장한다. 감사자는 그 payoff가
   실제로 어떤 정본 문장을 바꾸는지 **코드/문서 대조로** 확인하라. 바꾸는 문장이
   없다면 이 addendum은 **비용만 발생시킨 것**이며 그렇게 기록되어야 한다.

---

## 8. 산출물

- 구현: `g2s_e1_premise.py`(신규 파일. `g2s_analyze.py` **수정 없음**,
  import도 하지 않는다 — 상수·규칙은 **생산자**인 `gate1/gate1b_analyze.py`·
  `gate1/gate1c_analyze.py`·`pdmux_a100_smoke.yml`·`multiplex/multiplexing_mixin.py`
  에서 수입한다. 방법론 게이트 #14: 자기가 검증할 코드를 복사한 게이트는 항등식에
  가깝다 — 대조는 생산자에 건다.)
- 결과: `g2s_e1_premise_<tag>_<jobid>.json`(셀별 pooled/rep별 max, 여유, 무결성
  진단, 상태).
- **정본 문서 갱신은 이 문서가 하지 않는다** — doc-steward 소관이며,
  §1의 재감사 통과가 선행 조건이다.
