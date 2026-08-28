# 사전등록 — **P3 프로브: Qwen 컨트롤러 상태 + 수준차** (실행 전 등록)

2026-08-28 · rev3 감사(`../audit_tc1_rules_rev3_2026-08-28/VERDICT.md`) §확정실험 5 이행 ·
**≈0.4 GPU-hr, 2 job** · n=1 · **정책 판정 아님**

## 0. ★감사 표현 하나를 정정하고 시작한다
감사는 이 프로브가 *"하나가 넷을 산다"*며 `sd_int`(B27)를 포함시켰다. **1 rep은 SD를 사지 못한다.**
이 프로브가 사는 것은 **셋**이고, `sd_int`는 사지 못한다 — 그 사실을 지금 등록한다.

## 1. B34 — `ctrl_*` 경계를 **실행 전에** 고정한다 (감사 요구)
서버 로그에서 `BIND` = `SLO-BIND a->b` 줄 수, `FEAS` = `SLO-FEAS refused` 줄 수,
`visited` = `SLO-BIND` 목적지 인덱스 집합 ∪ {anchor}. **우선순위 순서로** 판정한다:

| 순위 | 라벨 | 조건 |
|---|---|---|
| 1 | `DEAD` | `BIND == 0 ∧ FEAS == 0` — 컨트롤러가 아무것도 제안하지 않았다 |
| 2 | `BLOCKED` | `FEAS > 0 ∧ argmax ∉ visited` — 제안했고 거부당했으며 argmax에 못 갔다 |
| 3 | `MISPOSITIONED` | `FEAS == 0 ∧ argmax ∉ visited` — 자유롭게 움직였으나 다른 데 정착했다 |
| 4 | `REACHES` | `argmax ∈ visited` |

상호배타·전수이며 전부 서버 로그에서 산출된다. **이 표가 rev4의 `ctrl_*` 정의다.**

## 2. 셀 (2 job, 전부 정본 하네스·정본 anchor)
`MODEL=Qwen/Qwen2.5-3B` · `CTX=4096` · cudagraph ON · telemetry 미설정 · 변화 trace 3↔12 ·
**anchor = 정본 상수 idx 2**(모델 무관) — job 896565의 idx 4가 아니다.
1. `d44` static, rep 91 — θ_T의 분모
2. `bind` + `PDMUX_SLO_FEAS_GATE=1`, rep 91 — θ_T의 분자 + `ctrl_T`

## 3. 결정 규칙 (실행 전 고정)

### P1 — `ctrl_T` (감사 F12). ★**두 분기 다 TC1에 나쁘다**
정본 `ctrl_H`는 `BLOCKED`이다(§1-10: `BIND=1`·`FEAS=113`, argmax d44 ∉ {d24,d34}).

| `ctrl_T` | 라벨 | 뜻 |
|---|---|---|
| `BLOCKED` | **`BOTH_BLOCKED`** | 두 모델의 "동적"이 모두 **틀린 static에 고정**된 것 ⇒ `θ_T−θ_H`는 **anchor 위치 페널티의 차이**이고 모델 귀속이 아니다 |
| `REACHES` · `MISPOSITIONED` | **`ASYMMETRIC_TREATMENT`** | 두 모델의 "동적"이 **서로 다른 처치**가 된다 ⇒ 2×2 식별 가정 위배(감사 F12 둘째 뿔) |
| `DEAD` | `MEASUREMENT_ABSENT` | 배선·설정 실패 |

★**이 프로브는 TC1을 죽일 수 있다.** 그것을 실행 전에 등록하는 것이 이 문서의 목적이다.

### P2 — 수준차 (감사 F10, **공변량**·판정 아님)
`level_gap = |gp_T(d44) − 3.220| / 3.220`. 감사 시뮬레이션상 치환 대조의 통과창은 **≈6–9%**.
⚠️정본 3.220은 2026-07 telemetry-OFF 런이라 **게이트 #41 드리프트**가 섞인다 ⇒ **지시적 값**으로만 쓴다.

### P3 — 토큰비 (B37, **공변량**)
두 런의 `input_lens` median 비. 감사 오프라인 실측 1.136–1.163을 **in-run으로 확인**한다.
rev3 `TOKEN_MAX_GAP=0.20` 대비 어디인지 보고한다.

## 4. 금지 문장
*"Qwen이 더 낫다/못하다"* · *"n=1로 정책을 비교했다"* · *"이 프로브가 `sd_int`를 샀다"*(§0) ·
*"`level_gap`이 게이트 #41을 넘어 비교 가능하다"* · *"P1이 `ASYMMETRIC`이면 TC1이 산다"*(둘 다 나쁘다).
