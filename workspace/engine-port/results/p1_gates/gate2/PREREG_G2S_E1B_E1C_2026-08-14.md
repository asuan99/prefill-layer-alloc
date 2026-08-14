# PREREG — **E1-b / E1-c**: Gate 2-S §8.9 전제를 **동거 구간 조밀 샘플링**으로 재측정 + 그 **양성대조**

작성: **2026-08-14** · 작성자: **engine-porter** · 상태: **미제출 · GPU 지출 0** ·
대상 하네스: `g2s_e1b_run.sbatch` · 스코어러: `g2s_e1b_premise.py`

> 이 문서는 `PREREG_GATE2S_2026-08-09.md`(rev6 + 2026-08-11 addendum)와
> `PREREG_G2S_E1_ADDENDUM_2026-08-11.md`의 **후속 사전등록**이다.
> 두 원문을 **한 글자도 덮어쓰지 않는다.** 원 사전등록의 판정 규칙(§4 primary,
> §5 앵커, §5.6.1 명명, §6.4 F-계열, §8.9.1 단방향 반증기, Holm, 9-셀 좌표)은
> **전부 불변**이며 이 캠페인은 그 어느 것도 재계산하지 않는다.

---

## ★ 0. 최상단 — **blind 선언 (이 등록은 완전한 blind가 아니다)**

E1 addendum이 `post-hoc, not blind`를 최상단에 선언한 선례를 따른다.
**작성자(engine-porter)가 이 문서를 쓰기 전에 이미 본 값의 전수 목록**:

| # | 이미 본 값 | 출처 | 이 등록에서의 역할 |
|---|---|---|---|
| B1 | E1의 4셀 pooled `max(decode_bs)` = **14 / 23 / 10 / 13**(여유 22/13/26/23) | `g2s_e1_premise_877756_877757.json` (오늘 직접 읽음) | **결정 규칙 입력 아님**(§5는 E1-b 자신의 데이터만 씀). 서술 대조로만. |
| B2 | G1-b(875293) Zamba2 **rate 6 → max 40, frac(54,54)=8.37%**, force=1 | `PROJECT_STATUS.md`·핸드오프 | **E1-c 대조 셀 선택의 근거**(= 이 선택은 blind가 아니다, §3.3) |
| B3 | Gate 2-S 자신의 capscan Zamba2 r6 **max 31**(force=0, 60 prompts) | 핸드오프 §2.2 | 같은 위(문턱 양쪽을 감싼다는 근거) |
| B4 | E1의 적대적 bound 25/41/24/27 (q=1e-9), Zamba2 r3는 q=1e-6에서 37 | 핸드오프 §2.2 | 인용만, 규칙 입력 아님 |
| B5 | T′ max = 13/23/10/12 | 핸드오프 §2.2 | 인용만 |
| B6 | ★**작성 중 내가 직접 새로 산출한 값 4건**(아래) | 이 문서 작성 세션, GPU 0 | §5의 **결정량 정의를 바꿨다** — §11.2에 전량 기재 |

**B6 전수 (숨기지 않는다)** — 기존 아티팩트 재집계, 새 GPU 0:

| 값 | 수치 | 어디에 영향 |
|---|---|---|
| G1-c(877974, force=1, Granite 복제격자) 전 행 `max(decode_bs)` | sched 15 / sched+forced **15** ⇒ **D = 0** | §5 P3가 **0을 낼 가능성이 높다**는 사전 인지. 규칙은 그대로 둔다(§7-2). |
| 같은 파일, 동거(pop A) 행 수 | scheduled 821 → 전체 26,340 ⇒ **densification 32.1×** | §5 P3의 모집단을 **동거 구간으로 좁힌** 근거(§11.2 자유모수 2) |
| 같은 파일 `trace_forced` 히스토그램 | False 41,551 / True 25,519 | I9(계측 engage) 임계의 비임의성 |
| Gate 2-S 4셀 A4(force=0) 동거 행 수·동거 max | 5,007/2,363/2,917/1,788 행, 동거 max **12/23/10/12**(전 행 max 14/23/10/13) | ★**"전 행 max"로 D를 정의하면 D≡0에 가까운 항등식**임을 실행 전에 발견 → §5 P3 정의 변경(게이트 #9 **수정**, 자수 아님) |

⇒ **Zamba2 r2·r3·Granite r3·r4에서 force=1로 측정한 값은 이 세상에 존재하지 않는다**
(존재했다면 E1이 그걸 썼을 것이다). 그 값들에 대해 **이 등록은 사전(prospective)**이다.
**대조 셀(E1-c, Zamba2 r6)의 예상 결과는 blind가 아니다**(B2). 양성대조는 정의상
"이미 발화한 것으로 아는 셀"을 고르는 장치이므로 정당하지만, **독립 확증이 아니다** —
E1-c가 발화해도 그것은 "계측이 살아 있다"만 말한다(§5 C1).

⚠️ **사후 인지의 자백은 면죄가 아니다**(방법론 게이트 #9). 이 절이 하는 일은 위반을
없애는 것이 아니라 **관측 가능하게 만드는 것**뿐이다. 재감사자는 **§7(자기무력화)**부터 읽어라.

---

## 1. 왜 새 캠페인인가 — E1이 스스로 못 닫은 구멍 하나

E1 addendum §2.5는 결정량이 **하한**임을 자백했다: telemetry는
`PDMUX_DUAL_WORKER_TRACE_EVERY=32`의 카운트 서브샘플이고 `M_observed ≤ M_true`이며
**편향의 방향이 저자에게 유리**하다. 그래서 상태 이름 자체에 해상도를 박았다
(`VERIFIED_AT_SAMPLED_INSTANTS`). 그리고 §7-2에서:

> "§2.5의 하한 편향이 여유 대비 크다고 판단되면 VERIFIED를 UNDETERMINED로 강등하라.
> 이 addendum은 그 강등을 위한 **사전 등록된 정량 문턱을 가지고 있지 않다**.
> **이것은 이 설계의 알려진 구멍이며, 숨기지 않는다.**"

E1-b는 그 구멍을 **문턱을 새로 발명해서**가 아니라 **편향을 직접 측정해서** 닫는다.
`PDMUX_TRACE_FORCE_PREFILL=1`이면 **prefill이 in-flight인 모든 sync가 emit**하므로
(`multiplexing_mixin.py:527-538`), 전제가 관심 갖는 모집단(=동거 구간)의 샘플링이
1/32에서 **sync 단위 사실상 전수**로 바뀐다. 실측 densification = **32.1×**(§0 B6).

### 1.1 ★ 게이트 #30 — 저장소가 "경로 없음"이라 못박은 지점 옆이다 (자백만으로 면제 안 됨)

negated-invariant를 grep해 원문 대조했다:

```
g2s_analyze.py:631  FALSIFIER_STATES = ("refuted", "unchanged")
g2s_analyze.py:682  "...PDMUX_TRACE_FORCE_PREFILL=0 gives ~1/32 of G1B pop-A density.
                     NO UPGRADE PATH EXISTS -- state stays 'unchanged'."
```

★ **그 금지문이 스스로 밝힌 사유가 바로 "force=0의 1/32 밀도"다.** E1-b는 그 사유를
제거한다. 그러므로 이것은 **금지의 근거를 없애는 실험**이고, 자백만으로 넘어가면
다섯 번째 재발의 재판이다. **처방(자백 아님)**:

1. **E1-b는 `PREMISE_LABEL`(g2s_analyze.py:84-89)을 쓰지 않는다.** 스코어러
   `g2s_e1b_premise.py`는 `g2s_analyze.py`를 **판정 목적으로 import하지 않는다**
   (`paired_t`/`metrics` **통계 함수만** 수입 — 코드로 확인: §9 표).
2. **"금지 사유가 사라졌으니 금지도 사라진다"는 추론을 이 문서는 명시적으로 금지한다.**
   `:682`의 밀도 서술은 **충분조건**이지 유일 사유가 아니다(교차 캠페인 이전, pop-A
   `t_total` 인용 금지, 셀 불일치 등 다른 사유가 병존). **라벨 변경은 이 캠페인의
   산출물이 아니며**, 원한다면 별도 등록·별도 감사가 필요하다.
3. E1-b가 만들 수 있는 것은 **(i) REFUTED**(저자에게 불리, 생산자 규칙 그대로),
   **(ii) 강등**(§5 P3), **(iii) caveat 문구의 해상도 교체**뿐이다.
4. ⇒ **E1-b는 단방향 장치가 아니다.** §8.9.1의 안전성을 **상속하지 않으며
   재감사 대상**이다(E1 §1과 같은 구조).

---

## 2. 이 캠페인이 서 있는 코드 사실 (전부 2026-08-14 직접 확인, file:line)

| # | 사실 | 근거 |
|---|---|---|
| C1 | force 플래그 파싱은 **올바르다**(`in ("1","true","True")`) — `"0"`은 OFF | `multiplexing_mixin.py:103-105` |
| C2 | `dual_worker_trace_count`는 **모든 sync에서 증가**하고, `scheduled`는 순수 카운트 규칙, `forced`는 `not scheduled ∧ force ∧ split_prefill_batch is not None` | `:501`, `:527-538` |
| C3 | ⇒ **legacy 격자는 플래그와 무관하게 동일**하고 forced 레코드는 **순증**이다. 생산자 주석이 소비자에게 `trace_forced != true` 필터로 legacy 비교가능 통계를 복원하라고 명시 | `:502-525` |
| C4 | `trace_forced` 필드는 **force 모드에서만** payload에 실린다 ⇒ force=0 런에는 **필드 자체가 없다** | `:565-571` (데이터 확인: 877756 r3 64,000행 전부 필드 없음) |
| C5 | `forced ⟹ 동거`는 **코드 항등식**(forced 조건이 `split_prefill_batch is not None`) — 데이터 확인: G1-c forced 25,519행 중 동거 25,519행(100%) | `:531-538` |
| C6 | 선택자 분기: `not running_batch.is_empty() and (split_prefill_batch or sticky)`; A4는 sticky OFF ⇒ **동거 순간에만** 인덱스 공식이 돈다. 공식은 `max(1, min(2, decode_bs // 18))` ⇒ **`idx==2 ⟺ decode_bs ≥ 36`** | `:881-909`, `pdmux_a100_smoke.yml`(`sm_group_num 4`, `decode_bs_divisor 36`, `manual_divisions` 없음) |
| C7 | A4는 `PDMUX_R2_POLICY` 미설정 ⇒ sticky 분기 미진입, C6의 자동 선택자가 실제로 도는 arm | `g2s_run.sbatch:111` |
| C8 | 스냅샷 payload는 `prefill_active_batch_size`·`decode_running_batch_size`·`stream_index`·`prefill_sms`·`decode_sms`를 싣는다 | `dual_worker.py:604-625` |
| C9 | 그 필드들은 `dual_worker_enabled`와 **무관하게** 채워진다(`observe_scheduler`가 `scheduler.split_prefill_batch`를 직접 읽음) ⇒ A4에서도 유효 | `dual_worker.py:566-570`; 데이터 확인: Gate 2-S A4 4셀 동거행 1,788–5,007 |
| C10 | `_dual_worker_sync`는 `PDMUX_TELEMETRY_PATH`만 있으면 돈다(dual_worker OFF여도) | `multiplexing_mixin.py:468-474` |
| C11 | ★ **event loop 한 반복에 sync는 두 번뿐**: `:997`(요청 수신 직후, prefill 어드미션 **이전**)과 `:1242`(반복 말미, adjust **이후**). 어드미션(`:1003-1004`)과 `adjust_stream_groups`(`:1070-1080`) **사이에는 sync가 없다** | `:997`, `:1003-1004`, `:1070-1080`, `:1242` |
| C12 | ⇒ **선택자가 자기 입력(`running_batch.batch_size()`)을 읽는 순간은 force=1로도 관측되지 않는다.** 그 구간이 정확히 **G1-a가 사려는 것**이다 | C11 |
| C13 | `premise` 라벨은 `paired_t`·`nine_cell`·`gate_label`·`mde_pct` 어디에도 **입력되지 않는다**(출력 dict의 라벨 필드일 뿐) | `g2s_analyze.py:1128-1172` |
| C14 | `paired_t`는 SD=0에서 **`degenerate=True`, CI=[mean,mean]**을 이미 반환하고 "등가 선언 금지"를 명시 | `g2s_analyze.py:408-423` |
| C15 | ★ **C2·C3·C4는 이미 CPU 단위 테스트로 고정돼 있다** — `tests/test_trace_force_prefill.py`의 5개 테스트가 (i) 기본 OFF, (ii) **OFF는 pre-patch emitter와 레코드 집합·필드까지 동일**, (iii) **ON은 레코드를 추가만 하고 scheduled 격자를 흔들지 않으므로 `trace_forced != True` 필터로 legacy와 동일하게 재채점 가능**, (iv) prefill 창이 관측 가능해짐, (v) 스케줄러 상태 미변경을 검정한다. 이 테스트는 `g2s_e1b_run.sbatch`가 GPU 이전에 돌리는 **차단 게이트**의 일부다 | `tests/test_trace_force_prefill.py:1-20, 111-170` (2026-08-14 venv에서 전체 140 tests OK / 10.6s 확인) |

---

## 3. Arm · 격자 · n (제출 전 고정)

### 3.1 Arm — **2개, 둘 다 Gate 2-S의 A4(agnostic)**

| arm | 서버 플래그 | env |
|---|---|---|
| `a4_force1` | `--enable-pdmux --pdmux-config-path <yml> --chunked-prefill-size -1 --disable-overlap-schedule` | `PDMUX_TELEMETRY_PATH`, `PDMUX_DUAL_WORKER_TRACE_EVERY=32`, **`PDMUX_TRACE_FORCE_PREFILL=1`** |
| `a4_force0` | 동일 | 동일 − force 플래그(**unset**) |

- **다른 arm은 존재하지 않는다.** C/T/T′/A2는 이 캠페인에 없다 ⇒ 이 캠페인에서
  **어떤 성능 대비도 계산되지 않는다**(§8).
- 두 arm은 **rep마다 무작위 순열**로 실행(순서 교락 제거), **(arm,rep)마다 부팅**.

### 3.2 셀 · 반복

| 모델 | rate | 역할 |
|---|---|---|
| Zamba2-2.7B | 2, 3 | **E1-b 셀**(Gate 2-S 지정 격자) |
| Zamba2-2.7B | **6** | **E1-c 양성대조 — Gate 2-S 셀이 아니다** |
| Granite-4.0-h-micro-base | 3, 4 | **E1-b 셀** |

- **n = 6**(제출 전 고정, 중간 확인 없음). n≥4를 요구한 E1-b·E1-c와 n≥6을 요구한
  **E1-d(Zamba2 r3 여유 13 정면 검정)를 하나의 설계로 동시 충족**한다.
  ⇒ **E1-d는 별도 캠페인으로 제출하지 않는다**; E1-d의 질문은 Zamba2 r3에 대한
  §5 P3와 동일하다.
- seed = `9000 + 100*rep + rate` — 873944/875344/875346/877756과 **같은 공식**.
- 한 부팅 안에서 rate는 **오름차순**(Zamba2 2→3→6). G1-b(875293)가 한 부팅에서
  2→3→4→6을 돌려 r6=40을 얻었으므로, **대조가 같은 within-boot carryover 아래**
  측정된다. (carryover 자체는 §7-6의 알려진 한계.)
- bench 파라미터는 `g2s_run.sbatch:160-164`와 **동일**(`random-ids`,
  input 2000 / output 96 / range-ratio 1.0, **120 prompts**, warmup 8,
  `--output-details` 필수).

### 3.3 ⚠️ 대조 셀 선택은 blind가 아니다 (사전에 밝힌다)

Zamba2 r6은 **발화한 것으로 이미 아는 셀**(B2: G1-b 40)이라서 골랐다. 동시에 이
캠페인 자신의 capscan은 같은 rate에서 **31**을 봤다(B3) ⇒ 문턱 36이 두 관측 **사이에
있다** ⇒ 대조는 **진짜로 실패할 수 있다**(항등식이 아님, §11.1 검사 5).

---

## 4. 무결성 규칙 (판정 전 통과 필수)

| ID | 검사 | 실패 시 |
|---|---|---|
| I1 | JSON 파손 라인 수 == 0 (rep 슬라이스 전부) | UNDETERMINED |
| I2 | `dropped_events` 최대값 == 0 | UNDETERMINED |
| I4 | rep 수 == **6** | UNDETERMINED |
| I6 | rep 내부 `sample_index` 역전 0건 · 중복 0건 | UNDETERMINED |
| **I8** | 생산자 `gate1b_analyze.grid_completeness()`를 **scheduled 부분집합**에 적용해 `violations == 0 ∧ missing == 0` | UNDETERMINED |
| **I9** | 셀의 `n_forced ≥ 1` — 계측이 실제로 engage했는가 | UNDETERMINED |

- **I2의 문턱 0은 비임의**다: 드롭된 행은 **관측되지 않은 `decode_bs`**이고 위로
  bound할 방법이 없다(E1 §4와 같은 논증).
- **I9가 없으면 §5 P3는 공허하다**: `n_forced == 0`이면 `D ≡ 0`이 자동이고
  "강등 안 됨"이 계측이 죽어 있었다는 뜻과 구별되지 않는다(게이트 #15의 거울상).
- ★ **E1의 I3·I5(셀 파일 vs per-boot 대조)는 상속하지 않는다.** 그 두 검사는
  **연결된 셀 파일을 읽는 경로**를 검증하는 것인데, 이 스코어러는 per-boot 슬라이스만
  읽는다(`g2s_e1b_premise.py:rep_slices`) ⇒ **대조할 대상이 없다.** 셀 파일은
  감사용으로 하네스가 계속 생성한다(스코어러 미소비).
- **서술 전용(판정 미사용)**: rep별 줄 수·densification 배수·`sample_index` 간격·
  rep 경계 시간 gap.

---

## 5. 판정 규칙 (**제출 전 고정 — 결과를 본 뒤 바꾸지 않는다**)

셀 `c`, arm `a4_force1`. 모든 통계는 **paired t 95% CI**로만 한다 —
**percentile bootstrap 금지**(게이트 #14/#27), Holm p 출처는 아래 명시.

### P1 — 전제 스크린 (E1-b 4셀)

`M_c` = pooled `max(decode_running_batch_size)`, 생산자 `gate1b_analyze.max_decode_bs()`를
`load_rows()`(event=`runtime_snapshot` ∧ phase=`benchmark`) 뒤에 그대로 적용, **전 행**
(scheduled ∪ forced). `max`는 결합적이므로 **pooled == max over reps**이며 rep별 max는
**두 번째 검정이 아니다**(E1 §2.4 상속).

| 조건 | 상태 |
|---|---|
| I1–I9 통과 ∧ `M_c < 36` | ★ **`PREMISE_HOLDS_AT_ALL_OBSERVED_COHABITING_SYNCS`** |
| I1–I9 통과 ∧ `M_c ≥ 36` | → **P2로** |
| 무결성 실패 | **UNDETERMINED**(사유 명시) |

### P2 — 반증 (생산자 규칙, **P1이 통과 못 할 때만 조회**)

pop A(생산자 `classify()=="A"`) 시간가중 `frac((54,54))`, 생산자 sparse guard
(`MIN_TOTAL_S=0.5`, `MIN_EPISODES=2`) 포함, 문턱 `WITHDRAWAL_THRESHOLD=0.01`
(`gate1b_analyze.py:52`, G1-b·G1-c가 쓴 바로 그 규칙).

| 조건 | 상태 |
|---|---|
| `frac ≥ 0.01` | ★ **REFUTED** — 그 셀에서 Gate 2-S §8.9 전제가 **깨진다**. ⇒ CONSENSUS rev15/rev22가 그 셀에 준 §5.6.1 **명명 층 해제는 즉시 철회**되어야 한다. |
| `frac < 0.01` | **UNDETERMINED** — 문턱은 닿았으나 체류가 철회 문턱 미만. **VERIFIED 아님, REFUTED 아님.** |
| sparse guard 발화 | **UNDETERMINED** |

★ **게이트 #9 적용(자수 아니라 수정)**: `idx==2 ⟺ decode_bs ≥ 36`은 코드 항등식(C6)
이므로 `M<36 ⟹ frac=0`이다. 두 수치를 "각각 전제를 지지한다"로 병기하면 **하나의
경험적 사실을 두 번 세는 것**이다. 그래서 **`frac`은 항등식이 정보를 주지 않는 분기
(`M ≥ 36`)에서만 조회**한다. `M<36` 분기에서는 스코어러가 `frac`을 **계산하지도
출력하지도 않는다**(코드: `g2s_e1b_premise.py`의 P1/P2 분기).

### P3 — **하한 편향의 크기**(E1 §7-2가 없다고 자백한 그 문턱)

- 모집단: **동거 구간**(생산자 `classify()=="A"`). 32× densification이 실제로 작용하는
  유일한 모집단이다(§0 B6, §11.2 자유모수 2).
- rep별 `D_rep = max(decode_bs | 동거, dense) − max(decode_bs | 동거, scheduled-only)`.
  `scheduled-only` = `trace_forced is not True`(C3·C4가 근거인 **단일** 술어).
- `D_rep ≥ 0`은 **부분집합 관계로 결정적**이다 ⇒ 관심은 부호가 아니라 **크기**.
- 통계: 생산자 `g2s_analyze.paired_t()` (n=6, `t_{0.975,5}=2.571`). **SD=0이면
  `degenerate=True`, CI=[0,0]**으로 보고하고 **"편향이 0으로 입증됐다"로 쓰지 않는다**
  (C14가 이미 "등가 선언 금지"를 규정).
- **강등 규칙(저자에게 불리한 방향으로만 발화)**:

  > `Mcoh_c^sched(pooled) + U_c ≥ 36` 이면 (U_c = paired t 95% CI 상한)
  > → 그 셀에 대해 **E1의 `VERIFIED_AT_SAMPLED_INSTANTS`는 향후 인용에서
  > UNDETERMINED로 재등급**된다.

- **다중비교 보정을 하지 않는다(의도적)**: 이 규칙의 "발화"는 **전부 저자에게 불리한
  방향**이다. Holm은 발화를 억제하므로, 여기에 적용하면 저자에게 유리한 쪽으로
  보수성을 잃는다. **그 판단을 사전에 명시하고 고정한다.**
- ★ `any()` over n reps 스크린은 **쓰지 않는다.** P1은 pooled max 하나이고
  `max`는 결합적이라 "어떤 rep에서든" = pooled(E1 §2.4) ⇒ 검정 1개, 귀무 발화율
  팽창 `1−(1−α)ⁿ` **해당 없음**(게이트 #24를 회피가 아니라 **비해당**으로 처리).
  P3만이 통계 검정이며 셀당 1개다.

### C1 — E1-c 양성대조 (Zamba2 r6)

| 조건 | 상태 |
|---|---|
| `M_r6 ≥ 36` | **`CONTROL_FIRED`** — 계측이 발화 가능함이 이 캠페인 안에서 시연됨 |
| `M_r6 < 36` | **`CONTROL_DID_NOT_FIRE`** |

★ **대조 셀은 Gate 2-S 셀이 아니므로 `PREMISE_*`/`REFUTED`로 라벨하지 않는다**
(그렇게 쓰면 "전제가 반증됐다"로 오독된다). 대조는 **계측의 생사만** 말한다.

★ **자기무력화 결선**: `CONTROL_DID_NOT_FIRE`이면 **이 캠페인의 모든
`PREMISE_HOLDS_*` 상태는 `UNINFORMATIVE_SCREEN`으로 자동 재라벨**되고
(스코어러가 강제 수행), **E1을 강화하는 근거로 인용할 수 없다**(게이트 #15:
"한 번도 발화한 적 없는 스크린의 통과"). Granite job에는 대조 셀이 없으므로
그 job의 상태는 **Zamba2 job의 대조 결과 없이는 읽을 수 없다**(스코어러가
`control_dependency`를 셀마다 각인).

### O1 — 관측자 효과 (자기무력화 전용, **비율만**)

`α = alpha_ITLp95_mean`(Gate 2-S §4.1.2 정의, 생산자 `g2s_analyze.metrics()` 수입).
rep로 paired, `a4_force1 − a4_force0`, paired t 95% CI, 상대 % 병기.

| 조건 | 상태 |
|---|---|
| `\|Δα\|/α > 3%` **∧** CI가 0을 포함하지 않음 | **`OBSERVER_PERTURBED`** — P1/P2/P3 상태에 이 태그를 **강제 병기**하고, E1 재등급 근거로 쓸 수 없다 |
| 그 외 | `OBSERVER_WITHIN_3PCT` |

3%는 이 프로젝트의 상시 관측자-효과 게이트 값 수입(신규 자유모수 아님).
★ **α의 절대값·다른 arm과의 비교·Gate 2-S 수치와의 대조는 전면 금지**(§8).

### 정지 규칙

1. **중간 확인 없음.** 6 rep을 전부 돌리고 그 뒤 채점한다. 결과를 보고 rep을 더
   붙이지 않는다(optional stopping 금지).
2. 셀의 사용 가능한 rep이 6개 미만이면 그 셀은 **UNDETERMINED**이고, "채우기 위한"
   부분 재실행을 하지 않는다. 재실행하려면 **전 셀 재실행 + 이 사전등록 재발행**.
3. 매니페스트 트립와이어 실패(exit 4) 시 **부분 채점 금지**.
4. 스모크(§10) 실패 시 **본 캠페인 제출 금지**.
5. **판정은 절대 exit code가 아니다**(게이트 #21). `REFUTED`/`UNDETERMINED`/
   `UNINFORMATIVE_SCREEN`은 전부 정상 종료(exit 0)다. exit 21은 **측정 부재**에만.

---

## 6. ★ 후속 게이트 — **G1-a가 필요한가**를 이 원자료로 판정한다 (게이트 #28)

### 6.1 코드 사실 확인 결과 — **E1-b는 G1-a를 대체하지 않는다** (전달받은 사실의 정정)

들어온 지시는 "`:1242` 스냅샷이 이미 필요한 필드를 싣고 force=1이면 동거 반복마다
강제 emit되므로 E1-b가 G1-a를 공짜로 대체한다"였다. 코드로 재확인한 결과:

- **맞는 부분**: `:1242`가 `prefill_active_batch_size`·`stream_index`·`prefill_sms`/
  `decode_sms`·`timestamp_s`를 싣는 것(C8), force=1이면 동거 sync마다 emit되는 것(C2),
  A4에서도 필드가 채워지는 것(C9) — **전부 참**.
- ★ **틀린 부분**: G1-a의 결정량은 "**어드미션-후(`:1003-1004`)/adjust-전(`:1070-1080`)
  구간의 시간 비율과 절대 ms**"인데, **그 구간 안에는 sync 호출이 하나도 없다**(C11).
  두 sync(`:997`, `:1242`)는 그 구간을 **바깥에서 감쌀 뿐**이다. 게다가 선택자가 자기
  입력을 읽는 순간(`:1080`) 자체가 어떤 샘플과도 일치하지 않는다(C12).
- ⇒ **E1-b가 추가로 사는 것은 "반복 내부 순서와 sub-iteration ms"가 아니라,
  G1-a 구간에 대한 상한(bound) 하나뿐이다.** G1-a의 점추정은 E1-b 원자료에서
  **도출 불가능**하다.

### 6.2 그래서 무엇을 판정하는가 — **상한만**, 그리고 한 방향으로만

스코어러는 `stale_window_upper_bound_frac`을 산출한다:
**인접 샘플쌍(Δ`sample_index` == 1)이면서 앞 행이 pop A인 구간의 dt 합 / 벤치 span.**

- 인접성 요구는 **비인접 dt 오염**(게이트 #15 두 번째 함정, pop-A `t_total_s` 인용
  금지 사유)을 구조적으로 배제하기 위한 것이다.
- G1-a의 구간은 반드시 **어떤 인접쌍 하나 안에** 들어 있으므로 이 합은 **상한**이다.
- **알려진 보수성 ≈2×**: 합에는 `:1242(k) → :997(k+1)` 쌍도 들어가는데 거기엔 stale
  구간이 없다. 그래도 상한이라 그대로 보고한다(게이트 #31: **bound와 점추정 구별**).

| 조건 | 판정 |
|---|---|
| 모든 E1-b 셀에서 상한 < `0.01` | **G1-a는 0.2 GPU-hr의 값이 없다** — 결정량의 상한이 이 프로젝트가 이미 "의미 있음"의 문턱으로 쓰는 값 미만 |
| 어느 셀이든 상한 ≥ `0.01` | **G1-a는 열린 채로 둔다** — 단 이것은 "G1-a가 무엇을 보일지"에 대해 **아무 말도 하지 않는다** |

문턱 `0.01`은 `gate1b_analyze.WITHDRAWAL_THRESHOLD` 수입(신규 자유모수 아님).
★ 이 게이트는 **"G1-a 불필요"만 결론낼 수 있고 "필요하며 X를 보일 것"은 결코 결론낼
수 없다.** 그 비대칭을 사전에 못박는다.

### 6.3 E1-b/E1-c가 실제로 **풀어주는 것**(코드 대조 완료) / **안 풀어주는 것**

**푼다**
1. E1 §7-2의 자백된 구멍 — 하한 편향에 **측정된 크기와 사전등록된 강등 규칙**이 생긴다.
2. 게이트 #15 — Gate 2-S 캠페인 안에서 **스크린이 발화 가능함이 시연**된다
   (현재 이 캠페인의 capscan r6는 31로 **한 번도 발화한 적이 없다**).
3. **REFUTED 경로가 살아 있다** — 4셀 중 어디서든 동거 조밀 관측이 36을 넘고 체류가
   1%를 넘으면 CONSENSUS의 명명 층 해제가 그 셀에서 철회된다(저자에게 불리).
4. G1-a 트리아지(§6.2, **상한 한 방향**).

**안 푼다 (코드로 확인)**
- 성능 수치 어느 하나도 안 바뀐다 — `premise`는 Δ/CI/Holm/F-계열 산출에 **미입력**(C13).
- **크기 인용 자격은 F-계열 gate 소관**이고 이 캠페인은 그 층을 건드리지 않는다.
  ⇒ 크기 인용 가능 셀은 **Zamba2 r2 하나 그대로**.
- §1-1의 "PD 분리 자체" 귀속은 **한 눈금도 안 움직인다**.
- `PREMISE_LABEL`의 하드코딩(`g2s_analyze.py:84-89`, Granite를 여전히 `unverified`로
  고정)은 **이 배치에서 고치지 않는다**(engine-porter 별건, 트리·매니페스트 불변 유지).

---

## 7. ★ 이 캠페인이 스스로 무력화되는 조건 (재감사자가 먼저 읽을 것)

1. **E1-c 대조 미발화** → 모든 `PREMISE_HOLDS_*`는 `UNINFORMATIVE_SCREEN`
   (스코어러가 강제). E1을 강화하지 못한다. §5 C1.
2. **P3가 퇴화(D≡0)** → CI=[0,0]이지만 **"편향 0 입증"이 아니다.** 동거 dense max도
   여전히 sync 해상도의 하한이며(C12), 선택자 자신의 읽기 순간은 미관측이다.
   ★ **사전 인지**: 유일한 기존 force=1 데이터(G1-c)에서 D=0이었다(§0 B6) ⇒
   **퇴화가 가장 그럴듯한 결과**다. 그래도 규칙을 바꾸지 않는다.
3. **O1이 `OBSERVER_PERTURBED`** → force 모드가 관측 대상 regime 자체를 흔든 것이므로,
   P1/P2/P3는 **섭동된 regime의 서술**이고 E1 재등급 근거가 되지 못한다.
4. **G-3(force 플래그 중립성) MISMATCH** → 두 arm의 batch=1 greedy sha가 다르면
   "관측 전용"이라는 코드 주석(`:502-525`)이 반증된 것이다. 그 경우 이 캠페인의
   **모든 상태는 UNDETERMINED**이고 별도 조사가 선행한다.
   (게이트 #21: 전송 실패는 MISMATCH가 아니라 UNDETERMINED.)
5. **I9 미충족(`n_forced == 0`)** → 계측이 engage하지 않았다 ⇒ E1이 이미 본 것 외에
   아무것도 측정하지 않은 것이다. 해당 셀 UNDETERMINED.
6. **within-boot carryover** — Zamba2 대조가 r2·r3 뒤에 돌기 때문에, 대조 미발화가
   "계측 불능"이 아니라 "carryover로 부하가 낮았다"일 수 있다. 이 캠페인은 그 둘을
   **구별하지 못한다**. 구별하려면 r6 단독 부팅이 필요한데, 그러면 G1-b와의 비교
   가능성(같은 within-boot 구조)을 잃는다 — **의도적 트레이드오프, 해소 안 됨.**
7. **§0의 사후 인지가 치명적이라 판단되면** — 특히 §5 P3의 모집단 정의가 §0 B6의
   실측을 보고 정해졌다는 사실(§11.2 자유모수 2)이 치명적이라 판단되면, **P3를 파기**
   하고 P1/P2/C1만 남겨라. P1/P2/C1의 규칙은 전부 생산자 수입이므로 그 파기가
   나머지를 오염시키지 않는다.

---

## 8. 해석 제한 (인용 시 항상 병기 — CONSENSUS §1-1과 **모순 없이**)

이 캠페인의 산출물에 대해 다음을 **금지**한다. (Gate 2-S rev20·G1-c rev22·
E1 rev23이 이미 건 금지와 **동일 방향**이며, 어느 것도 완화하지 않는다.)

1. **성능 판정 인용 금지.** 이 telemetry·bench의 TTFT/TPOT/ITL/goodput/throughput
   어떤 절대 수치도 인용 금지. O1의 α는 **비율만**, 자기무력화 목적만.
2. **크기 인용 가능 셀은 여전히 Zamba2 r2 하나**다. 이 캠페인은 F-계열 gate를 건드리지
   않으므로 그 수·정체에 대해 **아무 말도 하지 않는다**(2026-08-11 세션이 정확히 이
   혼동으로 오판 — 게이트 #28).
3. **§1-1 귀속 불변** — A4-vs-fused 격차의 성분·기여분·분해로 서술 금지,
   "PD 분리 자체가 원인" 승격 금지.
4. **selector-level 한정** — 하드웨어 SM 프로브(S3/`%smid`) 미실행. "물리적으로
   prefill 74 SM / decode 34 SM"으로 쓰지 않는다.
5. **`(54,54)`는 도달 불가**로 쓰지 않는다(rev22 금지 ④). 부하가 커지면 도달하며
   Zamba2 r6에서 실제 발생했다 — 이 캠페인의 E1-c가 바로 그 확인이다.
6. **pop A `t_total_s` 인용 금지**(경계 dt 5–29% 상향 편향, rev22 등재). §6.2의 상한은
   `t_total_s`가 아니라 **인접쌍 dt 합**이며 **상한**으로만 인용한다.
7. **pop C 절대·상대 분수 인용 금지**(Gate 1 제한 유지).
8. **`M`을 부하·용량·동시성의 대리 지표로 쓰지 마라.** 선택자 문턱 도달 여부만 답한다.
9. **모델 간·rate 간 `M` 비교로 거동 주장 금지.**
10. ★ **877756/877757의 `M`(force=0)과 이 캠페인의 `M`(force=1, 전 행)을 나란히 놓는
    비교를 결정 규칙에 넣지 않는다** — §11.3 참조(밀도 교락 처리 결정).
11. **이 캠페인 결과로 `PREMISE_LABEL`을 승격하지 않는다**(§1.1).
12. **"이 결과가 §1-1을 승격한다"고 쓰지 않는다.** 이 캠페인은 §1-1을 승격시킬 수 있는
    어떤 층도 건드리지 않는다.

---

## 9. 산출물 · **진단 필드 ↔ 실제 산출 경로** (게이트 #25, 여덟 번째 재발까지 간 항목)

사전등록이 "산출한다"고 쓴 필드는 **전부 코드로 산출 경로를 확인**했다. 확인 방법은
주장이 아니라 **실행**이다: `g2s_e1b_premise.py --selftest`가 합성 fixture로 전체
스코어러를 돌리고 **아래 필드가 하나라도 없으면 assert 실패**한다.
하네스는 GPU를 잡기 전에 이 selftest를 **차단 게이트**(exit 6)로 돌린다.

| 필드 | 산출 함수 | 검증 |
|---|---|---|
| `pooled_max_all_dense` / `_scheduled` | `score_cell` ← `g1b.max_decode_bs` | selftest |
| `pooled_max_cohab_dense` / `_scheduled` | 동, pop A 필터 `g1b.classify` | selftest |
| `headroom_all_dense` | `score_cell` | selftest |
| `n_forced_total`, `densification_x_median` | `score_cell` per-rep 집계 | selftest(+fixture가 forced 경로를 실제로 밟는지 assert) |
| `deficit{values,mean,ci_lo,ci_hi,degenerate,projected,downgrade_E1}` | `score_cell` ← `g2s.paired_t` | selftest |
| `frac{measurable,frac_5454,n_episodes}` | `frac_split` ← `g1b.analyze_window`+`g1b.frac_in` | dry-run(대조 발화 fixture) |
| `stale_window_upper_bound_frac_max`, `g1a_needed` | `stale_window_bound` | selftest |
| `alpha{state,rel_pct,ci_lo,ci_hi,n}` | `alpha_pair` ← `g2s.metrics` | selftest |
| `integrity_failures`(I1–I9), `per_rep[].grid` | `score_cell` ← `g1b.grid_completeness` | selftest |
| `control_fired`, `control_dependency` | `main` | dry-run(발화/미발화 양쪽) |

**아티팩트**
- `g2se1b_report_<tag>_<job>.json` — 셀별 전 필드
- `g2se1b_scoreout_<tag>_<job>.txt` — 스코어러 stdout
- `g2se1b_preflight_<tag>_<job>.json` — 측정 인벤토리(게이트 #21 분류용)
- `g2se1b_<tag>_g3_force_neutrality_<job>.json` — force 1/0 greedy sha 대조
- `manifest_diff_g2se1b_<tag>_<job>.txt`, `runtime_source_manifest_e1b_<tag>_<job>.sha256`
- 원자료: `g2se1b_<tag>_telemetry_<arm>_rep<N>_<job>.jsonl` +
  `..._teloffset_...json`(**rep 경계 복원의 유일한 근거**; gap 휴리스틱 금지) +
  `g2se1b_<tag>_<arm>_rep<N>_<job>.jsonl`(bench)
- **정본 문서 갱신은 이 문서가 하지 않는다** — doc-steward 소관이고,
  **claims-auditor 재감사 통과가 선행 조건**이다.

---

## 10. 배관 스모크 (방법론 게이트 #26, 개정판 — **배치 합** 기준)

이 배치(E1-b/E1-c)는 **신규 하네스 1개 + 신규 스코어러 1개**를 쓰고 예상 GPU가
1 GPU-hr을 넘는다 ⇒ **스모크 필수**.

| 항목 | 값 |
|---|---|
| 제출 | `G2S_TAG=zamba2-27b E1B_MODE=smoke sbatch --time=00:25:00 g2s_e1b_run.sbatch` |
| 규모 | 2 arm × **2 rep** × rate{2,3,6} × **20 prompts** ≈ **10–15분(≈0.2 GPU-hr)** |
| 확인 1 | rep jsonl에 **비어 있지 않은 `itls`/`ttfts`**가 rate 3개 전부에 존재(877107/877109를 죽인 그 필드) |
| 확인 2 | `_teloffset_`가 **rate 3개 슬라이스**를 기록했고 Σ슬라이스가 telemetry 줄 수와 정합 |
| 확인 3 | force=1 arm telemetry에 `trace_forced` **True/False가 모두** 존재(계측 engage) · force=0 arm에는 **필드 부재** |
| 확인 4 | 스코어러 rc=0이고 §9의 전 필드가 report json에 존재 |
| 확인 5 | G-3(force 중립성) 라인이 출력되고 PASS/MISMATCH/UNDETERMINED 중 하나로 확정 |
| **예상 출력** | 스코어러가 **`I4 rep count 2 != 6` → UNDETERMINED**를 찍는 것이 **정상**이다(무결성 게이트가 살아 있다는 증거) |

확인 1·2·4는 **코드가 산출**한다(각각 preflight json의 `have/want`+`defects`,
`per_rep[].n_lines`/`offset`, report json의 §9 필드). 확인 3의 앞 절반(force=1에
True/False 공존)은 `per_rep[].n_forced`/`n_scheduled`로 코드 산출이고, **뒤 절반
(force=0 telemetry에 필드 부재)과 확인 5는 사람이 stdout·원자료를 눈으로 보는
검사**다 — §9의 "필드"가 아니므로 게이트 #25의 산출 경로 요구 대상이 아니다.

★ **스모크 PASS는 P1/P2/P3/C1/O1 및 어떤 Gate 2-S 주장에 대해서도 정보가 0이다.**
아티팩트는 `g2se1bsmoke_manifest_<job>.txt`에 전량 열거되고 **DO-NOT-CITE**다.

---

## 11. 자기 반박 · 자유 모수 전수 · **게이트 #9 항등식 감사**

### 11.1 ★ 게이트 #9 — 내가 세운 게이트가 항등식이 아닌지 **실행 전에 코드로 확인한 결과**

이 프로젝트에서 게이트-항등식은 **일곱 번 재발**했고 "자수만 하고 원안대로 실행"이
다섯 번째 재발의 死因이었다. 그래서 **발견한 항등식은 고쳤다.**

| # | 검사한 명제 | 결과 | 조치 |
|---|---|---|---|
| 1 | `M<36`과 `frac(idx2)=0`이 독립 증거인가 | ★**항등식**(C6) | **고침**: `frac`은 `M≥36` 분기에서만 조회·출력. 두 수치의 병기 자체를 코드에서 제거. |
| 2 | `forced ⟹ 동거`인가 | ★**항등식**(C5, 데이터 25,519/25,519) | 결정에 무해(동거 제한이 **scheduled 쪽만** 거른다). **명시 기록.** |
| 3 | **`D`를 전 행에서 정의하면 D≡0인가** | ★**거의 항등식** — 전역 peak가 동거 밖에서 나면 forced 추가가 max를 못 올린다. 실측: G1-c 전 행 sched 15 = dense 15; Gate 2-S 4셀 전 행 max 14/23/10/13 vs 동거 max **12/23/10/12** | ★**고침**: P3의 모집단을 **동거 구간으로 변경**(32× densification이 실제로 작용하는 곳). **자수로 넘어가지 않았다.** |
| 4 | `D ≥ 0`이 항등식인가 | **참**(부분집합) | 결정량을 **부호가 아니라 크기**로 정의. 양측 검정을 "D≠0 발견"으로 읽지 않는다. |
| 5 | E1-c 대조가 반드시 발화하는가(=항등식) | **아님** — G1-b 40(발화) vs 이 캠페인 capscan 31(미발화)로 문턱이 **끼여 있다** | 그대로 사용. **미발화가 실제 가능**하므로 판별력 있음. |
| 6 | I9(`n_forced ≥ 1`)가 항상 참인가 | **아님** — 동거가 없는 셀에서는 0 가능 | 그대로 사용(게이트 #15 negative control 역할) |
| 7 | O1의 3% 게이트가 항등식인가 | **아님**(귀무 채택형이지만 **자기무력화 방향으로만** 쓰므로 게이트 #20의 "노이즈 보상" 함정과 반대 방향) | 그대로 사용, 단 "무섭동 입증"으로 쓰지 않음을 §8에 명시 |

### 11.2 자유 모수 전수 목록 (**신규 3개**, 나머지는 전부 수입)

| # | 모수 | 값 | 성격 |
|---|---|---|---|
| 1 | `n` | **6** | 신규. E1-b(≥4)·E1-c(≥4)·E1-d(≥6)를 하나로 덮기 위한 최소값. |
| 2 | ★ P3의 모집단 | **동거 구간(pop A)** | 신규. **§0 B6의 실측을 보고 정했다**(§11.1 검사 3). 이 인지는 §7-7의 파기 조건이 된다. |
| 3 | P3 강등 규칙의 형태 | `Mcoh^sched + U ≥ 36` | 신규(형태만). 상수 36은 수입. |
| — | 문턱 36 | `gate1b_analyze.py:51` = `pdmux_a100_smoke.yml` | 수입 |
| — | 철회 문턱 0.01 | `gate1b_analyze.py:52` | 수입(§6.2의 G1-a 트리아지에도 재사용) |
| — | `TRACE_EVERY=32` | `gate1b_analyze.py:50` / G1-b·G1-c·Gate 2-S 공통 | 수입 |
| — | 관측자 3% | 프로젝트 상시 게이트 | 수입 |
| — | 통계량·행 필터·pop 분류·t-CI·α 정의 | 생산자 import(§9) | 수입, 재구현 0줄 |
| — | 워크로드(2000/96/120/warmup 8/seed 공식) | `g2s_run.sbatch` | 수입 |

### 11.3 ★ **밀도 교락 처리 — (a)를 택한다 (애매하게 남기지 않는다)**

문제: `g2s_e1_premise.py`는 `trace_forced`를 참조하지 않는다(코드 확인: 그 파일에
문자열이 0회 등장). 따라서 877756/877757의 `M_c`(force=0)와 E1-b의 `M_c`(force=1,
전 행)를 나란히 놓으면 **밀도 교락 비교**다.

**택한 것: (a) `trace_forced` 필터를 구현한다.** 이유:

1. 생산자 코드가 **직접 그렇게 하라고 쓴다** — `multiplexing_mixin.py:502-525`:
   *"Recompute a legacy-comparable count statistic by filtering to `trace_forced != true`."*
   그리고 forced는 순증이고 scheduled 격자는 플래그와 무관하게 같다(C2·C3).
2. 필터가 있어야 **캠페인 내부 비교** `Mcoh^dense` vs `Mcoh^sched`가 정의된다 —
   이것이 §5 P3의 전부이고, **교락이 구조적으로 0**이다(같은 런·같은 부팅·같은 행 집합의
   중첩된 두 부분집합).
3. 구현: `g2s_e1b_premise.py:is_scheduled()` = `e.get("trace_forced") is not True`.
   force=0 런은 필드 자체가 없으므로(C4) **같은 술어 하나로 양쪽이 정확히 처리**된다.
4. ★ **그 술어의 정당성은 이 문서의 주장이 아니라 이미 있는 단위 테스트다**(C15):
   `test_trace_force_prefill.py`가 "ON은 추가만 하고 scheduled 격자를 흔들지 않으므로
   `trace_forced != True` 필터로 legacy처럼 재채점 가능"을 **명시적으로 검정**하고,
   그 테스트는 하네스가 GPU 이전에 돌리는 차단 게이트에 들어 있다. ⇒ 필터가 깨지면
   **캠페인이 시작되기 전에** `exit 5`로 죽는다.

**동시에 (b)의 금지도 명문화한다 — 다른 축이기 때문이다.** 필터는 **밀도**를 고치지만
**job/일자/섭동**은 못 고친다. 그래서:

- **결정 규칙에 교차 캠페인 비교를 넣지 않는다.** §5의 P1·P2·P3·C1·O1 어느 것도
  877756/877757의 값을 입력으로 받지 않는다(스코어러는 그 파일들을 **열지도 않는다**).
- 교차 캠페인 대조는 **보고서 서술에서만**, 그것도 **`M^sched` vs E1의 `M`**
  (= 같은 샘플링 규칙끼리)만 허용하고 "교차 캠페인"이라 명시해야 한다.
- **`M^dense`와 E1의 `M`을 나란히 놓는 표·문장은 전면 금지**(§8-10).

---

## 12. 환경 · **제출 순서** · 예산

### 12.1 ★ 제출 순서 — **E1-b/E1-c는 G1-a보다 먼저 제출돼야 한다**

`g2s_e1b_run.sbatch`는 `g2s_run.sbatch:81-89`의 매니페스트 트립와이어를 **그대로
상속**한다: 873944 기준 매니페스트 대비 `added==2 ∧ removed==0`, 위반 시 **exit 4**.
이 실험은 **엔진 트리를 한 바이트도 바꾸지 않으므로** 그 검사가 그대로 통과해야 한다.

> ★ **G1-a(관측 전용 sync 1회 추가)를 먼저 트리에 넣으면 E1-b는 `exit 4`로 죽는다.**
> 순서: **스모크 → E1-b/E1-c → (원하면) G1-a + 새 기준 매니페스트 동결.**
> `%smid` P1·P2와 P1 캠페인은 매니페스트 무변경이라 **동시 제출 안전**.

### 12.2 부팅·게이트

- `sync_engine_tree.sh` → 매니페스트 → 트립와이어 → **CPU 회귀 전체**(`exit 5`) →
  **스코어러 selftest**(`exit 6`) → correctness gate(G-1/G-2 per arm + **G-3 force 중립성**)
  → Phase 1.
- ★ **게이트 #33 병기**: 매니페스트 sha 일치는 **추적 15파일**에 대한 것이고 런타임
  바이트 동일성을 함의하지 않는다(`sgl_kernel/spatial.py`,
  `sglang/srt/multiplex/pdmux_context.py`가 매니페스트 밖). 하네스가 이 문장을 stdout에 찍는다.
- ★ **`SGLANG_*_TIMING` 회피(코드 미수정)**: `zamba2.py:544`·`granitemoehybrid.py:474`·
  `nemotron_h.py:646`이 `bool(os.environ.get(...))` 형태라 **`"0"`이 truthy**다.
  하네스는 이 이름들을 **`unset`**한다(`0`으로 세팅하지 않는다). **코드는 이 배치에서
  고치지 않는다** — 고치면 트리가 바뀌어 §12.1의 트립와이어가 발화한다. engine-porter 별건 이관.
- `--comment="field=efficientai;appl=pytorch"`가 `#SBATCH` 블록 안(첫 실행 라인 위)에
  있음을 `check_sbatch_comment.py`로 확인 완료(1/1 conformant).

### 12.3 예산 (★ PROJECT_STATUS 추정치를 초과한다 — 숨기지 않는다)

| job | 부팅 수 | 부팅당 bench | 예상 |
|---|---|---|---|
| Zamba2 (E1-b 2셀 + E1-c 대조) | 2 arm × 6 rep = **12** | r2·r3·r6 | ≈ 0.8 GPU-hr |
| Granite (E1-b 2셀) | 2 arm × 6 rep = **12** | r3·r4 | ≈ 0.6 GPU-hr |
| 스모크 | 4 | — | ≈ 0.2 GPU-hr |
| **합** | | | **≈ 1.6 GPU-hr** |

PROJECT_STATUS의 "E1-b 0.3–0.5 + E1-c 0.2"는 **n=4 · α 페어링 없음**을 가정한 값이다.
이 설계는 **n=6 + force0 페어링 arm**을 넣어 그 두 가정을 모두 깼다.
α 페어링을 빼면 ≈0.8 GPU-hr로 내려가지만 **§7-3의 자기무력화 검사가 사라진다**
(교차 캠페인 α 대조는 job·일자 교락이라 대체 불가). **페어링을 유지한다.**

`--time=02:30:00`(job당). SLURM: `partition=amd_a100nv_8`, `--gres=gpu:1`.
제출: `G2S_TAG=zamba2-27b sbatch .../g2s_e1b_run.sbatch` /
`G2S_TAG=granite-40-h-micro-base sbatch .../g2s_e1b_run.sbatch`.

---

## 13. 하지 말 것 (제출 전 고정)

1. **결과를 본 뒤 §5의 어떤 규칙도 바꾸지 마라.** 바꾸려면 이 문서를 파기하고
   재등록·재감사한다.
2. **`g2s_run.sbatch`·`g2s_analyze.py`·`g2s_e1_premise.py`·`PREREG_GATE2S`·
   `PREREG_G2S_E1_ADDENDUM`을 수정하지 마라.** 전부 감사 통과한 생산물이다.
3. **엔진 트리·`src/`를 바꾸지 마라.** 바꾸면 트립와이어가 발화하고, 발화하지 않게
   기준을 다시 동결하면 877756/877757과의 비교가능성이 사라진다.
4. **판정을 exit code로 승격하지 마라.** exit 21은 측정 부재에만.
5. **claims-auditor 재감사 전에 정본(`CONSENSUS.md`/`PROJECT_STATUS.md`)에 반영하지 마라.**
6. **rep을 나중에 더 붙이지 마라**(§5 정지 규칙 1·2).
7. **이 캠페인의 α·TTFT를 Gate 2-S의 어떤 arm과도 비교하지 마라.**
8. **`M^dense`(force=1 전 행)와 E1의 `M`(force=0)을 같은 표에 넣지 마라**(§11.3).
