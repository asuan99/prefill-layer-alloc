# 실 trace 검증 결과 + 동적 컨트롤러의 남은 두 갈래

> ⚠️ **정본은 [CONSENSUS.md](CONSENSUS.md)**. 이 문서 **§2.6/§2.7의 "bimodal은 트랩 때문" 귀속은 철회**됨 —
> static d24(switch=0)도 6.32↔3.10으로 붕괴(±1.302)하므로 분산의 상당 부분은 **시스템 노이즈**다.
> 트랩 *기전*(pf_age 단조증가)은 유효하나 **크기 귀속 불가**. §3.4 게이트 설계·§1 실trace 정정은 유효.

작성: 2026-07-16. 목적: **(1)** 실 trace(ShareGPT)·변화 trace 검증이 기존 synthetic 결론을 어떻게 정정했는지,
**(2)** 그 결과 드러난 컨트롤러의 두 결함 — **트리거(언제 움직이나)**와 **행동 모델(어느 쪽으로 움직이나)** — 을
각각 *무슨 문제를 서술하려는 것인지*와 *어떤 실측값이 그렇게 지목하는지*로 정리.

관련: [session_handoff_2026-07-15.md](session_handoff_2026-07-15.md)(전체 흐름), [slo_aware_scheduling_design.md](slo_aware_scheduling_design.md)(§D~HE2-3), [policy_comparison.md](policy_comparison.md).

---

## 0. 한 줄

**최적 split은 *부하 의존*이라 이동한다**(내 synthetic "d16 불변"은 저부하 아티팩트 — 실 trace서 d16이 6.10→**1.06** 붕괴).
**그럼에도 dynamic은 best-static을 못 넘는다**(실 trace·변화 trace 모두). 이유 = **비대칭**(decode 과소공급=파국, 과다공급=저렴)
→ **최악 phase 기준 decode-over-provision static이 robust하게 지배**. 컨트롤러의 두 결함(트리거·행동모델)은 **미분리**로 남음.

---

## 1. 실 trace 검증 — 기존 결론의 정정

### 1.1 stationary ShareGPT (rate 8, cudagraph, n=400)

| mode | goodput | good/400 | TTFT p50/p95/p99 | ITL p50/p95/p99 (ms) | switch |
|---|---|---|---|---|---|
| **d16** (prefill-heavy) | **1.056** | 76 | 7.24/10.1/**10.7s** | 33.5/**61.9**/129.9 | 0 |
| **d24** (best) | **6.240** | 393 | 1.21/1.95/**2.13s** | 29.5/39.9/64.6 | 0 |
| d44 | 5.973 | 389 | 0.12/0.99/5.59s | 24.9/37.5/56.1 | 0 |
| bind | 4.895 | 317 | 2.44/3.54/3.85s | 30.6/44.5/78.7 | **21** |
| slo | 4.085 | 271 | 1.73/4.38/4.95s | 30.8/42.5/59.0 | **5** |

★**synthetic서 "최적"이던 d16이 실 trace서 꼴찌로 붕괴**(6.098→1.056). ⇒ **"최적=d16·불변"은 저부하(o32/o96, rate 3–4) 아티팩트.**

### 1.2 변화 trace (ShareGPT, rate 3↔12 교대)

| mode | switch | LO(rate3) | HI(rate12) | **COMBINED** |
|---|---|---|---|---|
| **d44** | 0 | 8.583 | **11.892** | ★**9.706** |
| bind | 18 | 8.566 | 10.828 | 9.348 |
| d24 | 0 | 8.380 | 10.843 | 9.232 |
| slo | 18 | 8.580 | 9.483 | 8.901 |
| d16 | 0 | 8.582 | 8.183 | 8.438 |

★**최적이 실제로 이동**(LO=split 무관 ~8.5 / HI=d44 최적 11.9·d16 최악 8.2). **그래도 dynamic(9.35) < best-static d44(9.71).**

### 1.3 정립된 메커니즘

**얽힘(entanglement)**: prefill·decode가 `max_running_requests`(48)·KV pool을 **공유**하는 running batch로 결합.

> decode 굶김(낮은 D_sm) → ITL↑ → 요청이 batch에 오래 체류 → batch 포화 → **새 prefill admit 불가** → 큐 대기 → **TTFT 폭발**

**실측 확증**: d16은 prefill에 **92 SM(최대)**를 주는데도 TTFT p50=**7.24s**. 원인은 ITL p95=**61.9ms(SLO 초과)** → 정체 → admission 차단.
d24(prefill 84, ITL p95 39.9=여유) → TTFT 1.21s. **prefill SM을 *더* 줬더니 TTFT가 6× 악화** = 얽힘의 직접 증거.

**최적 D_sm = max( 모델-구조 floor [attn-decode knee ≈d16], 부하-의존 decode 예약 [∝ 도착률 × output_len] )**
- 저-decode-부하(synthetic o32/o96, rate 3–4) → floor 지배 → d16처럼 보임
- 고-decode-부하(ShareGPT rate 8–12) → **부하항 지배 → d24/d44**

**비대칭**: decode **과다공급**은 저부하서 거의 무해(LO phase 전 split ≈8.5), **과소공급**은 고부하서 파국(d16 HI 8.18, stationary 1.06).
⇒ **최악 phase 기준으로 decode를 넉넉히 준 static이 두 phase 모두 안전 → 지배.** 이것이 "최적이 이동해도 static이 이기는" 이유.

### 1.4 Switch decomposition (분석 프레임)

> **Net dynamic 이득 = Σ(B: positioning 이득) − (A: switch_count × per-switch drain)**
> - **(A) overhead 지점** = green-ctx drain × 전환 횟수. 순수 비용.
> - **(B) positioning 지점** = 각 구간에서 split이 *그 순간 최적*에 얼마나 붙어있나. 최적 접근=+, 이탈=−.

| | switch | (B) positioning | (A) overhead |
|---|---|---|---|
| bind(varying) | 18 | HI서 d24 배회(d44 미달) = **주손실** | 부차 |
| slo(varying) | 18 | 더 나쁜 위치(HI 9.48) = **더 큰 손실** | 동일 |
| slo(stationary) | **5** | 4.085 | — |
| bind(stationary) | **21** | 4.895 | — |

★**slo(5 switch)가 bind(21 switch)보다 *나쁨*** ⇒ **손실은 (A)overhead가 아니라 (B)positioning이 지배.**
"switch를 줄이면 좋아진다"가 아니라 **"split이 순간 최적에 붙어있나"가 결정.**

---

## 2. 남은 갈래 (a) — 순수 트리거 효과 (언제 움직이나)

### 2.1 무슨 문제를 서술하려는 것인가
**"static이 이미 SLO를 충분히 충족(headroom 존재)하는데, 컨트롤러는 왜 움직이는가?"**
컨트롤러는 **static이 부족할 때만** 개입해야 한다. 충족 regime에서 움직이면 (A)overhead + (B)이탈 = **순손실**.
즉 "올바른 컨트롤러는 headroom이 있으면 **inert(=static)**여야 한다"는 원리의 검증.

### 2.2 어떤 결과값이 이를 지목하는가
**bind switch 로그(stationary ShareGPT r8, d24 대비 healthy regime):**

| switch | pf_age | **pfslack** | decslack | 그 순간 실제 상태 |
|---|---|---|---|---|
| 2→1 (d16으로) | 1566ms | **0.48** | 0.57 | TTFT 1.5s ≪ SLO 3s (통과) |
| 2→1 | 2089ms | 0.30 | 0.48 | TTFT 2.1s < 3s (통과) |
| 2→1 | 2114ms | 0.30 | 0.44 | 통과 중 |

- **d24 static은 393/400 통과·TTFT p99 2.13s** = headroom 명확. 그런데 bind은 **21회** 발동.
- **decslack은 항상 0.4–0.65** = decode 전혀 위험 없음. 그래도 prefill-chase 발동.
- **원인 지목**: `_pf_urg=0.5` → "TTFT 예산의 **절반**만 써도 위급" 판정. **위반 근접(slack→0)이 아니라 *수준(level)*에 반응** → 정상 부하서 상시 발동.

### 2.3 왜 아직 미확인인가 (confound)
시험을 돌렸으나 **두 변수를 동시에 바꿈**: `pf_urg 0.5→0.1` **AND** `dwell 3→10`.
결과: **goodput 2.160 / switch 8** (bind-tight 4.895/21sw보다 **악화**).
- switch는 21→8로 줄었으나 **회복 실패**. 혐의: **dwell=10이 d16(파국) 체류를 10스텝 연장** → excursion당 피해 증대.
- ⇒ **트리거 완화의 순수 효과는 미측정.**

### 2.4 시험 설계 (해야 할 것)
- **pf_urg = 0.1** (위반 근접에서만 발동: TTFT > 2.7s), **dwell = 3 고정**(기존값). 변수 1개만.
- 비교군: bind-tight(pf_urg 0.5) / bind-puretrigger(0.1) / **d24 static(6.240)**.
- 측정: **switch_count + split 체류분포 + goodput + TTFT/ITL p50/p95/p99** (decomposition 프레임 적용).

### 2.5 판정 게이트
- **switch↓ ∧ goodput → d24(6.24) 근접** ⇒ **트리거가 주원인** — "headroom 있으면 inert" 원리 확증(사용자 가설 성립).
- **switch↓인데 goodput 여전히 미달** ⇒ **트리거는 부차** — 문제는 *언제*가 아니라 *어느 쪽으로* 움직이나 ⇒ 갈래 (b)가 주범.

### 2.6 ★결과 (2026-07-16, jobs 854640/854641/854642) — 둘 다 맞았고, **양극성(bistable)** 발견

`pf_urg 0.5→0.1`, **dwell=3 고정**(변수 1개). stationary ShareGPT r8, 2-rep:

| rep | goodput | switch | split 체류 | TTFT p50 |
|---|---|---|---|---|
| **rep4** | **6.237** ✅ | **2** | d24(1)·d34(1) — **d16 미방문** | 1.16s |
| **rep3** | **2.219** ❌ | **16** | **d16(7)**·d24(8)·d34(1) | 3.91s |
| *d24 static (rep1/rep2)* | *6.240 / 6.317* | *0* | — | *0.84s* |

**같은 설정이 실행마다 6.24↔2.22로 갈림 = bistable.**

**(a) 확증**: 완화하면 rep4처럼 **switch 2회로 static 매칭**(6.237≈6.28) 가능 ⇒ tight 트리거가 불필요 churn의 원인 맞음.
또한 **switch 2회 = overhead 사실상 0** ⇒ **(A)overhead는 문제가 아님**을 직접 증명(§1.4 프레임 확증).

★★**(b)가 진짜 killer — 양성 피드백 트랩(positive-feedback trap) 발견.** rep3 로그(SLO-BIND 궤적):

```
2->1 dec_sm=16 pf_age=2800ms tpot=31.5ms pfslack=0.07   ← 트랩 진입: transient spike가 0.1 임계 넘음
1->2 dec_sm=24 pf_age=3200ms             pfslack=-0.07  ← d16 갔는데 pf_age가 *늘어남*
2->1 dec_sm=16 pf_age=3530ms             pfslack=-0.18  ← 또 증가
2->1 dec_sm=16 pf_age=4146ms             pfslack=-0.38
1->2 dec_sm=24 pf_age=4743ms             pfslack=-0.58  ← 2800→4743 단조 증가, 탈출 불가
```
vs **rep4**: pf_age가 ~1400ms 유지, **0.1 임계를 한 번도 안 넘음** → d16 미방문 → 6.237.

**기전**: 컨트롤러는 "prefill로 이동 → pf_age↓"를 가정하나, 얽힘 때문에 **pf_age↑**(d16 → ITL>SLO → batch 정체 → admission 차단).
⇒ **음성 피드백 설계가 양성 피드백으로 작동 → 불안정 → 양극.** 한 번 들어가면 **행동이 오류 신호를 스스로 증폭**해 못 나옴.
**트랩 진입 순간 tpot=31.5ms**(d24서 decode 멀쩡) → 상대비교(`pf_slack<dec_slack`) 통과 → 이동 승인 → 파국. **단 하나의 transient spike가 6.24 vs 2.22를 가름.**

### 2.7 두 갈래 관계 (확정)
- **(a) 트리거 tightness** = **트랩 *진입 확률*** 결정. 완화 → 진입 드묾 → 성공할 *수도*.
- **(b) 잘못된 행동 방향** = 트랩을 **self-reinforcing(탈출 불가)**로 만듦. **완화만으론 제거 불가**(rep3가 증명).
⇒ **(a)는 확률만 낮춤; robust하려면 (b) 필수.** (b) 없이는 **같은 설정도 실행마다 6.24↔2.22 도박.**
⇒ 이 트랩이 지금까지 **모든** dynamic 실패(bind-tight 21sw 4.895 / loose+dwell10 2.160 / slo 4.085 / varying bind 18sw)의 **통일적 설명**.

---

## 3. 남은 갈래 (b) — 얽힘-aware 행동 모델 (어느 쪽으로 움직이나)

### 3.1 무슨 문제를 서술하려는 것인가
**"컨트롤러가 'prefill 위급'이라 판단했을 때 하는 행동(=decode SM을 뺏어 prefill에 줌)이, 얽힘 때문에 오히려 prefill을 죽인다."**
즉 **행동의 *방향*이 틀렸다.** 컨트롤러 내부 모델은 "prefill SM↑ → TTFT↓"인데, 부하 하에서 이 관계가 **반전**한다
(decode 굶김 → batch 정체 → admission 차단 → TTFT↑). 컨트롤러는 이 **decode→prefill backpressure 경로를 모른다.**

### 3.2 어떤 결과값이 이를 지목하는가
**직접 증명 (stationary ShareGPT r8, static 비교 — 컨트롤러 무관하게 성립):**

| static | prefill SM | ITL p95 | **TTFT p50** | goodput |
|---|---|---|---|---|
| **d16** | **92 (최대)** | **61.9ms (SLO 초과)** | **7.24s (붕괴)** | 1.056 |
| d24 | 84 | 39.9ms (여유) | **1.21s** | 6.240 |

⇒ **prefill SM을 8 더 줬더니(84→92) TTFT가 6× 악화.** "prefill SM↑→TTFT↓"는 **거짓**. 원인은 decode가 SLO를 넘겨 생긴 정체.

**컨트롤러가 정확히 이 방향으로 움직임:** bind switch 로그의 **모든 "prefill 위급" 발동이 `2→1`(d24→d16)** — 즉 **파국 방향**.
그리고 그때 **decslack=0.4–0.65로 "decode 여유"로 보였음** — 상대 비교(`pf_slack < dec_slack`)가 통과했기 때문.

### 3.3 현 설계의 정확한 결함
컨트롤러는 **`pf_slack < dec_slack` (상대 비교)** 만 확인한다. 그러나:
- **"현재 d24에서 decslack=0.5(여유)"** ≠ **"d16으로 내려가도 decode가 버틴다"**.
- **decode knee는 비선형**: 24→16 SM에서 ITL이 급증(실측 39.9→61.9ms, SLO 돌파). 현재 여유는 *현재 split에서의* 여유일 뿐.
- ⇒ 컨트롤러는 **후보 split에서의 decode 실현가능성(feasibility)을 예측하지 않는다.** 현재값만 보고 이동한다.

### 3.4 ★설계 — knee 기반 feasibility 게이트 (설계만; 미구현)

**목적**: 트랩 **진입 자체를 원천 차단**. "prefill이 위급해도 decode가 그 SM을 내줄 여력이 없으면 뺏지 않는다."
(prefill의 TTFT 문제는 decode를 굶겨서 풀 수 없다 — §1.3 얽힘.)

#### 3.4.1 게이트 규칙 (핵심 로직)
```
# _slo_decide_idx_binding에서 이동 결정 후, *적용 전* 검사
if idx' < idx:                                   # prefill-ward (D_sm 감소) 이동만 게이트
    D_cand   = sm_counts[idx'][1]
    itl_pred = predict_ITL(D_cand, decode_bs, ctx)
    if itl_pred > TPOT_SLO * FEAS_MARGIN:        # 예: 60ms * 0.9 = 54ms
        REFUSE -> idx' = idx                     # 현재 split 유지
```
- **prefill-ward 이동만 게이트**: D_sm을 *늘리는*(decode-ward) 이동은 이 실패 모드를 못 만듦(§3.4.5 비대칭).
- **safety margin**(0.9): 예측 오차 흡수. 보수적일수록 static에 수렴.

#### 3.4.2 ★구체화 중 발견 — **ITL 게이트만으로는 트랩을 못 잡는다 (설계 정정)**

실제 수치를 넣어보니 §3.4.1의 ITL 가드가 **실패**한다:

**(가) knee 절대값은 전혀 transfer 안 됨** (프로파일 조건이 다름: ctx3600·no-cudagraph vs ShareGPT ctx~352·cudagraph):

| D_sm | knee(오프라인, 9×per-attn+54×per-mamba) | ShareGPT r8 실측 ITL p50 |
|---|---|---|
| 44 | 70.7ms | **24.9ms** |
| 24 | 127.4ms | **29.5ms** (4.3× 차이) |
| 16 | 186.3ms | **33.5ms** |

⇒ 절대값 사용 불가. **SHAPE(비)만** 쓰고 현재 split의 실측 TPOT로 앵커링해야 함.

**(나) 그렇게 해도 트랩 진입을 통과시킴**: rep3 진입 시점(현재 d24, tpot=31.5, 후보 d16)
`predict_ITL(d16) = 31.5 × (186.3/127.4) = 46.1ms` **< 54ms(=60×0.9) → 이동 허용 = 트랩 미차단.**

**(다) 근본 이유**: **d16의 실패는 ITL 위반이 아니다.** d16의 **ITL p50=33.5ms는 60ms SLO를 통과**한다.
d16이 죽는 건 **TTFT=7.24s**, 즉 **혼잡(congestion)**이다. ⇒ **ITL-SLO를 예측하는 게이트는 애초에 틀린 대상을 본다.**

#### 3.4.2b 올바른 1차 가드 — Little's law 혼잡

`필요 동시성 N = 도착률 λ × (output_len × ITL)` vs `max_running_requests(=48)`:

| split | ITL p50 | 체류 = 213×ITL | **N = 8×체류** | vs cap 48 | 실측 TTFT p50 |
|---|---|---|---|---|---|
| d44 | 24.9ms | 5.30s | **42** | 여유 ✓ | **0.12s** |
| d24 | 29.5ms | 6.28s | **50** | 경계 | **1.21s** |
| d16 | 33.5ms | 7.14s | **57** | **초과 → 포화** | **7.24s** |

**TTFT 순서를 정확히 재현**(0.12 < 1.21 ≪ 7.24) ⇒ **혼잡이 진짜 기전.**
런타임 관측 대체물: **`decode_bs` vs `max_running_requests`** (λ·output_len 불요 — output_len은 런타임에 미지이므로 결정적 장점).

**게이트 적용 결과**: 진입 시점 batch ≈ 48(cap) ≥ 48×0.85=40.8 → **거부** → d24 유지 → **트랩 미진입 → 결정론적 6.24**(rep4 경로 강제).
**양극성 제거가 게이트의 1차 효과.**

#### 3.4.3 predict_ITL 후보 (택1 또는 조합)
| 방식 | 입력 | 장점 | 단점 |
|---|---|---|---|
| **(i) 오프라인 knee 테이블** ★권장 | `results/r0c/knee_result_*.txt`(SM 108/44/24/16/8별 per-attn/per-mamba ms) → ITL(D_sm) | 런타임 학습 불요·결정론적 | 모델/config별 프로파일 필요, batch·ctx 의존 보정 필요 |
| (ii) 온라인 관측맵 | 런 중 방문한 D_sm→TPOT-EMA 히스토리 | 자동 적응 | **cold start + 닭-달걀**(d16을 *가보지 않고* 알아야 하는데 가는 게 위험) |
| (iii) 해석적 외삽 | 관측 2점 → `ITL ≈ a + b/D_sm` 적합 | 값쌈·프로파일 최소 | 외삽 오차(knee 비선형 구간서 위험) |

**권고: (i) 오프라인 knee 1차 + (iii) 온라인 보정**(현재 D_sm의 실측 TPOT로 테이블을 스케일 → batch/ctx 드리프트 흡수).
★**의의**: 이것이 **layer-aware/decode-knee 데이터가 실제로 쓰이는 자리** — anchor *값* 예측이 아니라 **행동 feasibility 예측**.
([[prefill-layer-alloc-status]]의 decode knee가 입력. 죽은 줄 알았던 knee가 여기서 부활.)

#### 3.4.4 knobs · 구현 지점
- env: `PDMUX_SLO_FEAS_GATE=1`(게이트 on, off면 기존과 byte-identical), `PDMUX_SLO_FEAS_MARGIN`(기본 0.9), `PDMUX_SLO_KNEE_PATH`(테이블).
- 코드: `src/multiplex/multiplexing_mixin.py`의 `_slo_decide_idx_binding` — `_new` 확정 직후 게이트 삽입 + `_predict_itl()` 헬퍼 신규.
- 로그: `SLO-BIND`에 `feas=refused/ok itl_pred=..` 추가 → 거부 횟수를 측정 가능하게(진입 차단 실증용).

#### 3.4.5 왜 prefill-ward만 게이트하나 (비대칭)
- **decode 굶김 → prefill 죽음**: decode 지연이 running batch를 점유 → **admission 차단** → 새 prefill이 아예 못 들어옴. **전파됨.**
- **prefill 굶김 → decode 영향 미미**: 이미 admit된 decode 요청은 계속 진행. prefill이 느려도 decode를 막지 않음. **전파 안 됨.**
⇒ 위험한 방향은 **prefill-ward(D_sm↓)** 하나뿐. (대칭 게이트는 불필요·과보수 위험.)

#### 3.4.6 검증 계획
- **HG1 (주)**: stationary ShareGPT r8, **≥3 rep**. 목표 = **분산 붕괴** — 현재 6.24↔2.22 양극이 **일관되게 ~6.2**로. 지표: goodput mean±std, **d16 체류=0**, `feas=refused` 횟수>0(게이트 실제 발동 증명), switch 수, TTFT/ITL p50/p95/p99.
- **HG2**: 변화 trace(3↔12) — HI phase서 prefill-ward 드리프트 거부 → d44 쪽 유지 → **bind→d44(9.71) 근접**?
- **HG-iso**: 저부하(rate 4)서 게이트가 과발동해 정상 이동까지 막지 않는지(무해 확인).
- **HG0**: 여전히 미달 → 예측기 부정확 or 제3의 실패모드 → 재진단.

#### 3.4.7 기대 효과와 **천장** (정직하게)
- **얻는 것**: 트랩 제거 → **결정론적 static 매칭**(6.24), 양극성/도박 소멸. **robustness**가 산출물.
- **못 얻는 것**: **static 초과 아님.** §1.3 비대칭(decode over-provision이 저부하서 무해) 때문에 어떤 static도 못 따라가는 구간이 실질적으로 없음 → **"goodput 이득 無" 트랙 결론 불변.**
- ⇒ 게이트의 가치 = **성능 반전이 아니라 (1) 기전 확증(트랩이 원인이었다) + (2) 동적을 *안전하게* 만들기**(최악 2.22 → 안정 6.2).

### 3.5 판정 게이트
- **feasibility 게이트 후 d16 excursion 소멸 → bind ≈ d24(stationary)·≈d44(varying)** ⇒ **행동모델이 주범이었음** 확정. 단 **천장은 "static 매칭"**.
- 여전히 미달 ⇒ dynamic은 매칭조차 못 함 → 트랙 결론(무이득) 강화.

### 3.6 한계 (미리 정직하게)
(b)가 성공해도 **기대 천장은 "static 매칭"**이다. dynamic이 static을 *초과*하려면 "최적이 이동하는데 어떤 static도 못 따라가는" 구간이 필요한데,
**§1.3 비대칭** 때문에 그 구간이 실질적으로 없다(decode-over-provision static이 저부하서도 안전 → 두 phase 모두 커버).
⇒ **(a)·(b) 모두 고쳐도 "goodput 이득 無"라는 트랙 결론은 뒤집히지 않을 공산이 크다.** 의의는 *왜* 동적이 지는지의 기전 규명.

---

## 4. 두 갈래의 관계 · 우선순위

| | 갈래 (a) 트리거 | 갈래 (b) 행동모델 |
|---|---|---|
| 질문 | **언제** 움직이나 | **어느 쪽으로** 움직이나 |
| 결함 | headroom 있는데 발동(pf_urg=0.5) | 파국 방향(d16)으로 이동(얽힘 무시) |
| 증거 | switch 21회 @ pfslack 0.3–0.5, SLO 통과 중 | d16 = prefill SM 최대인데 TTFT 6× 악화 |
| 고치면 | 정상 regime서 inert → static 매칭 | 잘못된 이동 차단 → static 매칭 |
| 상태 | **미확인**(dwell과 confound) | **미구현** |

**우선순위 권고**: **(a) 먼저** — 변수 1개(dwell 고정)만 바꾸는 값싼 실험이고, (a)의 결과가 (b)의 필요성을 판정한다.
- (a)로 goodput이 d24에 붙으면 → 문제는 트리거였고 (b)는 부차.
- (a)로도 미달이면 → **(b)가 주범** 확정 → feasibility 게이트 구현.

**어느 쪽이든 §3.6의 천장(static 매칭)은 유효** — 두 갈래는 *성능 반전*이 아니라 **기전 규명**을 위한 작업.

## 5. 파일
- 하네스: `results/slo_sched/sharegpt_bench.sbatch`(stationary), `sharegpt_vary_bench.sbatch`(rate 3↔12, switch count 병기).
- 컨트롤러: `src/multiplex/multiplexing_mixin.py` — `_slo_decide_idx_binding`(env `PDMUX_SLO_PF_URGENCY`·`_DWELL`·`_ANCHOR_IDX`).
- 데이터: `sgpt_*.out`(SGPT_RESULT/SGPT_PCT), `sgptv_*.out`(SGPTV_RESULT/SGPTV_PCT + `SWITCHES`).
