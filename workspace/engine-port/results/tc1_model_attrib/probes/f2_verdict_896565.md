# F2 확정 프로브 결과 — job **896565** (`F2_CONFIRMED`)

2026-08-28 · gpu39 · `COMPLETED` exit `0:0` · 11분 42초 · **≈0.20 GPU-hr**
사전등록: [`PREREG_F2_ANCHOR_PROBE.md`](PREREG_F2_ANCHOR_PROBE.md)(실행 **전** 등록) · n=1

## 채점 (사전등록 규칙 축자 적용)

| 항목 | 사전등록 조건 | 실측 | |
|---|---|---|---|
| 부팅 | `boot_ok=1` | **1** | ✓ |
| `SW` | `≤ 2` → CONFIRMED | **0** | ✓ |
| `gpC` | `≥ 3.10` → CONFIRMED | **3.232** | ✓ |

⇒ ★**`F2_CONFIRMED`**

## 원자료 (인용 시 이 줄만)
```
SWITCHES tag=bind_rep1_L3H12_896565 count=0
SGPTV_RESULT tag=bind_rep1_L3H12_896565 rate=3/12 sw=0 | LO gp=2.849 | HI gp=3.981 | COMBINED=3.232
SGPTV_PCT ... LO | TTFT p50/p95/p99=0.08/0.27/1.02 ITL 13/17/28
SGPTV_PCT ... HI | TTFT p50/p95/p99=1.16/6.51/6.84 ITL 27/35/41
```
설정: `bind` + `PDMUX_SLO_FEAS_GATE=1` + **`PDMUX_SLO_ANCHOR_IDX=4`**(= d44,
근거 [`B17_ANCHOR_INDEX_MAP.md`](B17_ANCHOR_INDEX_MAP.md)) · Zamba2-2.7B · cudagraph ON ·
telemetry 미설정 · 변화 trace 3↔12 3라운드.

## 읽기

컨트롤러를 **자기 argmax(d44)에 anchor**시키면 **런 전체에서 이동이 0회**이고 static을 재현한다.
⇒ `Δ = dynamic − static ≈ 0`이 되어 `δ = 3%`에 한참 못 미치므로, **모델과 무관하게 두 arm이
`lose`로 떨어지고 `NO_FLIP_BOTH_LOSE`가 기전적으로 강제된다** — 감사 F2가 실측으로 확정됐다.
⇒ **TC1 rev1의 "anchor = Stage 1 argmax" 처방은 폐기가 맞고, rev2의 모델-무관 상수 고정이 필요하다.**

## 금지 (사전등록 §금지 문장)
*"d44 anchor가 더 낫다"* · *"n=1로 정책을 비교했다"* · *"F2가 닫혔으므로 rev2를 제출할 수 있다"*.
`gpC=3.232`를 정본 d44-static(3.220±0.013)과 **크기 비교하지 않는다** — n=1이고, 이 프로브는
정책이 아니라 **컨트롤러의 도달가능성**만 묻는다. stderr의 `Killed`는 하네스 자신의
종료 루틴(`kill $SRV`)이며 측정 실패가 아니다(`DONE_` 마커·exit `0:0` 확인).

## ★rev2에 대한 함의 — 다음 감사가 반드시 봐야 할 것

rev2가 F2 수리로 도입한 **관측 축 `visits_argmax`가 Zamba2 arm을 막을 가능성이 높다.**
정본 §1-10: 정본 anchor(idx 2 = d24)에서 `bind+GATE`는 **1회 이동(d24→d34) 후 113회 거부**로
d34에 고정됐고 **argmax인 d44에 도달한 적이 없다**. ⇒ `visits_argmax = no` ⇒ rev2 규칙은
그 arm을 **`CONTROLLER_DEGENERATE`로 차단**한다.

**이것은 결함이 아니라 축이 제 일을 하는 것**이다 — *"자기 최적에 도달조차 못 하는 컨트롤러로
모델 차이를 재는 것은 모델을 재는 게 아니다"*. 그러나 결과적으로 **TC1 rev2가 Zamba2에 대해
실질 판정을 내지 못할 구조적 가능성**이 생겼고, 이는 본 캠페인 50 job을 사기 **전에**
재감사가 판정해야 할 사항이다. rev2 §6(설계층 도달가능성) 표에 이 사례를 반영해야 한다.
