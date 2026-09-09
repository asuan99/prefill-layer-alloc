# 사전등록 — **계측 프로브 P1–P4** (측정 먼저, 설계는 그 다음)

2026-09-08 · 총 **≈0.75 GPU-h** · **정책 판정 없음 · arm 순위 없음** ·
발단 = 사용자 지적(*"추정으로 재현 가능성이 부족하다고 실행 안 하는 것처럼 보인다"*)

> ## 왜 이 문서가 짧은가
> 이 세션은 규칙층 판본 3개(rev3·rev4·`RATIO`)를 종이로 쓰고 **14연속 `NO-GO`**를 받았다.
> 그 死因 다수가 **기판의 경험적 사실**(듀티사이클·바닥·부팅 가능성)이었고 값싼 프로브로
> 답이 나오는 것들이었다. 이 문서는 **그것들을 먼저 사기 위한 최소 등록**이다.
> 저장소 관행과도 일치한다 — TC1은 규칙층 `NO-GO` 3회 중 job 896565(0.20 GPU-hr)를 샀고,
> cp_baseline은 `NO-GO` 4회 중 1.36 GPU-hr을 계측 검증에 썼다.

## 0. 공통 등록
- 모델 **`nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base`** · 백엔드 **`flashinfer`**(게이트 #83 강제)
- `--disable-piecewise-cuda-graph --disable-overlap-schedule --chunked-prefill-size -1
  --disable-radix-cache --mem-fraction-static 0.80 --max-running-requests 48` · cudagraph ON
- `PDMUX_R2_POLICY=fixed` · `PDMUX_R2_FIXED_DSM=<D>` · telemetry 3종 env **필수**
- 데이터셋 `random` + `--tokenize-prompt` + `--random-range-ratio 1.0`
- 분석은 **정본 함수 `e1_pin_check::compute_decode_realized`** 사용(재구현 금지).
  ★**두 추정량을 항상 병기**: `f_time`(시간가중) · `f_count`(스냅샷 개수가중).
  이번 세션 감사가 둘의 비가 곧 **구간-부과 길이편향**임을 보였다.
- ★**`split ∧ prefill_active_batch_size == 0` 건수를 전 프로브에서 병기**한다
  (record-skew, 감사 死因 R3).

## 1. P1 `SPLIT_FIRES` — ★가장 중요 (2 부팅, ≈0.10 GPU-h)
**질문**: Nemotron + flashinfer + **동시성 > 1**에서 SM 분할이 **발화하는가.**
저장소에 증거가 **0건**이다(AF-1은 동시성 1이라 L-10으로 원천 미발화, W1은 파티션 텔레메트리 미기록).

**설정**: ctx 8192 · out 96 · `--request-rate 1.0` · 요청 160 · `d44` 1부팅 + `d92` 1부팅
(기존 `stage0_xctrl/pdmux_d44.yml`·`pdmux_d92.yml`).

| 라벨 | 조건 | 귀결 |
|---|---|---|
| `SPLIT_ABSENT` | 어느 arm이든 **split 라벨 decode-active 스냅샷이 0개** | ★**`RATIO` 계열 전부 부존재** — 이 기판·이 모델에서 처치가 안 걸린다는 **결과**로 기록 |
| `SPLIT_FIRES` | 두 arm 모두 split 스냅샷 > 0 **이고** `d44`·`d92`의 `decode_hist` 최빈 split 값이 각각 44·92 | 진행 |
| `SPLIT_UNRESOLVED` | 부팅 실패·telemetry 결손 | 1회 재실행 |

★**크기 문턱을 두지 않는다.** 이번 세션이 `f > 0.20`을 등록했다가 **자기 설계를 51배 차이로
기각**당했고, 감사가 그 추정량이 편향돼 있음을 보였다 ⇒ **존재만 묻는다.**
`f_time`·`f_count`는 **기술 값으로 공표**하되 판정에 쓰지 않는다.

## 2. P2 `FLOOR` — 외삽 제거 (4 부팅, ≈0.30 GPU-h)
**질문**: `floor_ref(ctx)`·`itl_solo`의 **실측값**. 지금까지 `RATIO`의 셀 좌표·도착률·SLO 문턱·
예산이 **전부 다른 모델에서 외삽한 바닥** 위에 있었다.

**설정**: **동시성 1**(`--request-rate` 미지정·순차) · ctx ∈ {1024, 4096, 8192, 16384} ·
out 96 · 요청 22(**앞 2개 워밍업 제외**) · `d44` 1부팅/ctx.
**산출**: `floor_ref = TTFT p50` · `itl_solo` = **고정 토큰창 [9, 32]의 ITL p50** ·
실현 입력 토큰 수 · TTFT p50/p95.

**판정 없음**(캘리브레이션). ★단 **등록 예측**: 이전 외삽값(0.13/0.49/0.96/1.90 s)의
**±30% 밖이면 `RATIO`의 셀 좌표·`k`·예산을 전부 재계산**한다(그 자체가 산출물).
★**L-10 승계**: 동시성 1에서는 분할이 발화하지 않으므로 이 값은 **arm 무관 물리량**이다.

## 3. P3 `BOOT_SANITY` — 균질화 config (3 부팅, ≈0.15 GPU-h)
**질문**: 3F3 수리로 등록한 **`sm_group_num: 5` + division 3행**이 **실제로 부팅되는가.**
지금까지 **코드 독해로만** 확인했다(`decode_states={16,44}` 가드 통과 · `_r2_decide_idx` 유일 매칭).

**설정**: 균질화 yml 1개를 `d16`/`d44`/`d92` 세 target으로 각 1부팅, ctx 16384.
**산출**: 부팅 성공 여부 · 배너의 `sm_counts`·`max_total_num_tokens`·mamba pool ·
cudagraph 캡처 목록 · 짧은 스모크 8요청.

| 라벨 | 조건 |
|---|---|
| `BOOT_OK` | 3/3 부팅 + 캡처 완료 + `sm_counts` 5행 |
| `BOOT_FAIL` | 어느 하나라도 실패 ⇒ **3F3 수리 무효**, 재설계 사유 |

★**부수 등록**: division 3개의 비용(감사 차단 B12)을 **여기서 실측**한다 —
`max_total_num_tokens`를 기존 1-division config와 대조해 병기.

## 3-C. P3-C `DIVISION_COST` — ★추가 등록 (2026-09-09, 1 부팅, ≈0.05 GPU-h)

**왜 추가하나 — 사전등록 결함의 수리**: §3은 division 3개의 비용을 *"기존 1-division config와
대조해 병기"*하겠다고 적었으나, **대조 가능한 아티팩트가 있는지 먼저 확인하지 않았다.**
P3 완료 후 확인하니 저장소의 Nemotron 1-division 선례는 `mem-fraction-static`·
`max-running-requests`가 달라(w1probe R=48이나 mem-frac 미일치, cp2048은 R=8)
**Δ를 division 수에 귀속할 수 없다** ⇒ 감사 차단 **B12는 P3로 닫히지 않았다.**

**설정 — 한 변수만 다르다**: P3의 `d44` 부팅과 **모든 플래그·env·ctx·target을 동일**하게 두고
**config만** `probes/pdmux_homog5.yml`(5 groups) → **`stage0_xctrl/pdmux_d44.yml`(3 groups,
division 1행)** 로 바꾼 1 부팅. `PDMUX_R2_FIXED_DSM=44` 고정.

**산출(전부 배너·로그에서)**: `max_total_num_tokens` · cudagraph 캡처 목록과 개수 ·
`PD-Multiplexing enabled with N stream groups` · 부팅 소요 시간 · 8요청 스모크 rc.

| 라벨 | 조건 | 귀결 |
|---|---|---|
| `HOMOG_FREE` | `Δmax_total_num_tokens` **< 5%** **이고** 캡처 목록 크기 동일 | 균질화는 사실상 무상 — B12 닫힘 |
| `HOMOG_COSTED` | 위 둘 중 하나라도 어긋남 | 균질화에 **선언된 비용**이 있다 ⇒ 이를 쓰는 캠페인은 **운영점이 정본 1-division 캠페인과 다르다**는 것을 병기해야 한다(B12는 "닫힘"이 아니라 "정량화됨") |
| `CONTROL_FAILED` | 부팅 실패 | 재실행 1회 |

★**5%의 출처**(게이트 #93): 이 저장소의 `PRACTICAL_FLOOR`(3%)보다 관대한 값을 **의도적으로**
골랐다 — 이것은 성능 문턱이 아니라 *"메모리 예산이 실질적으로 같은가"*의 공학적 눈금이며,
**성능 판정에 쓰지 않는다.**
★**주장하지 않는 것**: 이 Δ가 **성능에 미치는 영향** — 재지 않는다. KV 예산 크기와
goodput 사이에 어떤 관계도 주장하지 않는다.

## 4. P4 `SKEW` — 계측기 자신을 잰다 (2 부팅, ≈0.20 GPU-h)
**질문 두 개**: (a) `PDMUX_DUAL_WORKER_TRACE_EVERY=32`(기본) 부표집이 `f`에 주는 편향 ·
(b) **sticky ON이 `f`를 실제로 올리는가**(코드 독해로만 주장했다).

**설정**: P1과 동일 셀(`d44`, ctx 8192, rate 1.0) × **`TRACE_EVERY=1`** 2부팅 —
하나는 **sticky OFF**, 하나는 **`PDMUX_STICKY_PARTITION=1`**.

★**등록 예측(틀리면 그게 결과다)**:
1. `TRACE_EVERY=1`의 `f_count`가 P1(`=32`)의 `f_count`와 **±20% 안**이면 부표집은 무해.
   밖이면 **`FINDING_BIN0`의 수리 범위를 전 bin으로 확장**해야 한다(감사 제안 5).
2. **sticky ON의 `f_count`가 OFF의 3배 이상**이면 `FINDING_DUTYCYCLE §2`의 코드 독해가
   지지된다. **3배 미만이면 그 독해는 철회**한다.
3. sticky ON에서 `split ∧ pab==0` 비율이 **OFF보다 커야** 한다(분할을 prefill 없이도 유지하므로).
   반대면 record-skew 기전 설명이 틀린 것이다.

## 4-C. ★P1×P2 결합 예측 — **P2 결과를 보기 전에 등록**(2026-09-09, GPU 추가 0)

P1 실측(메인 세션 재계산): `d44` **`f_time` 0.9921** · `d92` **0.9982** ·
**`split ∧ pab==0` = 0/474, 0/480**. g16(Zamba2)의 `f` 0.10–0.28 · `pab==0` 35.8%와 **정반대 체제**다.

**가설**: `f ≈ ρ_pf = rate × floor_ref(ctx)`. P1은 ctx 8192·rate 1.0이므로
`floor_ref(8192) ≈ 1.0 s`면 `ρ_pf ≈ 1.0`이고 관측 `f ≈ 0.99`와 맞는다.
**P2가 `floor_ref(8192)`를 실측하므로 이 예측은 추가 GPU 없이 검정된다.**

| 결과 | 사상 |
|---|---|
| `ρ_pf`(=1.0 × `floor_ref(8192)`)가 관측 `f`의 **±15% 안** | `f ≈ ρ_pf`가 **이 체제에서** 지지됨. ⇒ 감사 死因 R1의 기전(듀티사이클 = prefill 부하)은 **반증이 아니라 확증**되고, `RATIO` 셀 A(`ρ_pf`가 셀 B의 0.89%)에 대한 R1의 지적은 **그대로 살아 있다** |
| ±15% 밖 | `f ≈ ρ_pf`는 **이 체제에서도** 성립 안 함 ⇒ `f`의 기전이 미상이며 `FINDING_DUTYCYCLE`의 철회는 유지, 새 설명이 필요 |

★**주의(등록)**: `rate 1.0`은 §1이 *"임의 선택이며 운영점 주장 아님"*으로 등록한 값이다.
⇒ **`f ≈ 0.99`를 기판의 성질로 읽지 않는다** — **이 부하에서의 값**이다. 낮은 prefill 부하
셀에서 `f`가 어떻게 되는지는 **P1이 답하지 않았다**.
★g16(Zamba2·sticky OFF)과의 절대값 비교는 **모델·백엔드·부하가 전부 달라 금지**(게이트 #32).

## 5. 이 프로브들이 **주장하지 않는 것**
- 성능·goodput·정책 순위 — **재지 않는다**(P1의 rate 1.0은 임의 선택이며 운영점 주장 아님).
- `f`의 절대값을 물리량으로 승격하지 않는다(감사가 이미 강등).
- HE0·정본 결론에 대한 어떤 함의도 없다.
- P1이 `SPLIT_FIRES`라도 그것은 *"처치가 존재한다"*까지이고 **실현 정도가 아니다**.

## 6. 쓰면 안 되는 문장
- *"P1이 분할 실현을 검증했다"* — **발화 여부**만.
- *"`f`가 처치 걸린 시간 비율이다"* — 32-sync 부표집 격자의 지속시간 몫이다(감사 개선안 3).
- *"P2 바닥이 다른 모델·부하에 이식된다"* — 게이트 #32.
- *"프로브가 규칙층을 대체한다"* — 규칙층은 여전히 `NO-GO`이고 캠페인은 미제출이다.
