# 실 trace 검증 결과 + 동적 컨트롤러의 남은 두 갈래

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

### 3.4 시험 설계 (해야 할 것) — feasibility 게이트
행동 **전에** 후보 split의 결과를 예측해 거부:
- **decode knee(오프라인 계측)**로 후보 `D_sm`에서의 **ITL 예측** → `predicted_ITL(D_sm_candidate) > TPOT_SLO`면 **그 이동을 거부**.
- 즉 규칙: *"prefill이 위급해도, decode가 그 SM을 내줄 여력이 없으면 뺏지 않는다"*. (prefill의 TTFT 문제는 decode를 굶겨서 못 푼다.)
- ShareGPT에 적용시: d24→d16 후보의 예측 ITL(≈62ms) > 60ms → **이동 거부** → d24 유지 → **static 매칭 기대**.
- ★**의의**: 이것이 **layer-aware/knee 데이터가 실제로 쓰이는 자리** — anchor *값* 예측이 아니라 **행동 feasibility 예측**으로. ([[prefill-layer-alloc-status]]의 decode knee가 입력.)

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
