# 사전등록 — TC1 **F2 확정 프로브** (실행 전 등록)

2026-08-28 · 감사 `audit_tc1_rules_2026-08-27/VERDICT.md` §6이 지정 · **≈0.3 GPU-hr** · n=1

## 무엇을 묻나
감사 F2: *"동적 arm의 anchor를 static argmax에 못 박으면 컨트롤러가 LO·HI 양쪽에서 anchor를
떠나지 않아 `NO_FLIP_BOTH_LOSE`가 기전적으로 강제된다."*

> **컨트롤러를 d44(=자기 argmax)에 anchor시키면, 실제로 거기 앉아 버리는가?**

## 이것은 정책 판정이 **아니다** — n=1이다
`CLAUDE.md 게이트 #3`(n≥4)은 **정책 결론**에 걸린다. 이 프로브는 정책을 비교하지 않고
**컨트롤러의 도달가능성이라는 설계 사실** 하나를 묻는다. 어떤 정책 순위도 산출하지 않으며,
`goodput` 값을 **정본과 비교하는 크기 주장으로 쓰지 않는다**.

## 셀
`sharegpt_vary_bench.sbatch bind 1 3 12` + `PDMUX_SLO_FEAS_GATE=1` +
**`PDMUX_SLO_ANCHOR_IDX=4`**(= d44, 근거 `B17_ANCHOR_INDEX_MAP.md`).
Zamba2-2.7B · cudagraph ON · telemetry 미설정(정본 HE0 캠페인과 같은 조건, 게이트 #41).

## 결정 규칙 (실행 전 고정)
`SW` = 서버 로그의 `SLO-BIND|SLO-SCHED` 라인 수(하네스 `:76`), `gpC` = COMBINED goodput.

| 라벨 | 조건 |
|---|---|
| **`F2_CONFIRMED`** | `SW ≤ 2` **∧** `gpC ≥ 3.10` — 컨트롤러가 anchor에 앉고 static d44(정본 3.220) 바로 아래를 낸다 ⇒ anchor=argmax가 대조를 붕괴시킨다는 F2가 실재 |
| **`F2_REFUTED`** | `SW ≥ 8` — 컨트롤러가 anchor를 실제로 떠난다 ⇒ F2의 기전 전제가 깨진다 |
| **`F2_INCONCLUSIVE`** | 그 외 (`3 ≤ SW ≤ 7`, 또는 `SW ≤ 2`인데 `gpC < 3.10`) |
| **`MEASUREMENT_ABSENT`** | 부팅 실패 · `boot_ok=0` · bench 미완주 |

★`gpC ≥ 3.10` 문턱의 근거: 정본 d34-static 3.171 − 정착비용 0.039 = 3.132(§1-10)가
"anchor에 앉은 컨트롤러"의 알려진 서명이고, d24-static은 3.081이다. 3.10은 **그 둘 사이**로,
"d34 이상 급에 앉았다"를 "d24 급에 머물렀다"와 가른다. 데이터를 보고 고른 값이 아니다.

## 금지 문장
*"이 프로브가 d44 anchor가 더 낫다는 것을 보였다"* · *"n=1로 정책을 비교했다"* ·
*"F2가 닫혔으므로 rev2를 제출할 수 있다"*(재감사 필요).
