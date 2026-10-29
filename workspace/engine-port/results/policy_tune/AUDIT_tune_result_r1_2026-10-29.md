# 결과 감사 r1 — TUNE (2026-10-29)

- 감사자: claims-auditor(읽기 전용, CPU 재계산) · 1/2회 · **판정: EVAL을 HE0 정책 판정으로 돌리는 것은 NO-GO(N1·N3) → 사용자 결정 필요**

## (1) 선택 — CONFIRMED
`analyze_tune.py` raw 재실행: configs 표 `tune_analysis.json`과 바이트 동일, 선택 `dyn_a4_w3`. trace 동일성 seed 101·102 True. 이름 순 깨기는 PREREG §4-3 본문에 없고 봉인 코드 `ph_verdict.select_tune`(165·170행, key `(sw, name)`)에만 있음. 반전: 이름 역순 → a4_w6, 동률 폭 0% → a4_w6 — 둘 다 anchor 4·switch 0, dwell은 이동 뒤에만 작동해 EVAL 행동 동일 → 판정 변화 없음.
caveat: "선택 `dyn_a4_w3`는 최고 goodput 구성이 아니며, 이름 순 깨기는 PREREG 본문이 아니라 봉인 코드에 있다."

## (2) EVAL은 사실상 D44 대 D44
구조적 전환 불가는 아님(`_slo_decide_idx_binding`에서 idx 4 → 3·5 가능). 그러나 a4 run 4개 모두 서버 로그 `SLO-FEAS refused idx=4->4 dec_sm=44->34` 72–78회 — feasibility gate의 혼잡 가드(대표 `bs=47` ≥ 0.85×`max_running_requests=48`)가 매번 거부. D64 쪽 이동은 TPOT-EMA > 51 ms 필요, 실측 최대 35.6 ms로 미발동. → "이 부하에서 조건 미충족"이나 그 조건은 h=1.5 과부하에서 batch가 상한 48에 붙는 것으로 사실상 결정. EVAL은 rate·trace 구조 동일·seed만 다름 → 같은 고정 예보(TUNE 4/4 + 엔진 기전).
퇴화: (a) dyn vs s44 = 같은 분할에서 코드 경로만 다름(SLO_SCHED vs `R2_POLICY=fixed`) + 부팅 구조 차(동적은 seed별 재부팅, static은 한 boot 2 seed) → 기대값 0의 영 비교. (b) dyn vs 사후 best static = 사실상 static끼리. → `DYN_ABOVE_STATIC` 기대상 도달 불가(N3), 라벨이 컨트롤러가 아닌 static 격자를 잼(N1, 교훈 9 계열). 남는 정보: 같은 분할 두 코드 경로의 영 비교(오버헤드 상한, 부팅 교락 포함), n=4 static D 5개 순위(기술값).

## (3) TUNE 기술 사실 한계(n=2, CI 없음)
순위 a4 > a3 > a2는 두 seed 일치, 3% 초과는 a2 vs a4뿐(a4 vs a3 0.6–2.5%, headline 아님). **"전환이 많을수록 goodput이 낮다"는 NOT-YET-SUPPORTED(confound 9, positioning)**: a3 seed102는 전환 0회인데 2.189/2.179로 a4보다 낮음, a2는 대부분 D24 체류(≈260/380 s) → 차이는 전환 횟수가 아니라 앵커 위치와 공변. dwell 3 vs 6 차이 없음(a3 seed101 둘 다 4회, D44 체류 84 s).
금지: "전환이 goodput을 낮춘다" · "최적 동적 = 비전환"(PREREG §8: 격자 6점·seed 2) · "동적은 static과 같다" · "gate가 함정을 막아 이득을 냈다".
허용: "이 셀에서 앵커 4 구성은 FEAS 혼잡 가드에 의해 D44에 머물렀다."

## (4) HI 과부하와 해상도
HI goodput ≈1.7 req/s = 도착률 10.94의 16% — "구간 초반 대기열이 3 s를 넘기 전에 통과한 요청 수"를 셈, batch 상한 48과 decode 속도가 지배. ±5% cliff 검사(2.7%)는 시간축 절벽(구간마다 한 번 경계 통과)을 못 잡음(confound 5 변형). LO 전 구성 2.42 동일 → HI 차 25%가 전체 7%로 희석. 같은 과부하가 동적 arm을 고정 → h=1.5가 컨트롤러 여지를 닫음.

## (5) 기준 셀 0.983 → 0.908
각 120요청 1회, λ*는 정의상 cliff 위(TTFT p95 2.82 → 3.12 s, SLO 3 s 횡단, confound 5). Job 내 반복 구성 비교로 표류 증거 없음, seed·순서 혼재로 판정 불가. caveat 4 승계(Job 간 해석 금지).

## (6) 사용자 결정
- (A, 권고) EVAL을 HE0 라벨 없이 범위 축소 실행: "static D 5개 n=4 기술 순위 + dyn_a4_w3 vs s44 같은 분할 영 비교", TUNE은 기술 결과로 정리.
- (B) EVAL 생략, TUNE 기술 결과로만.
- (C) 동적 arm을 전환이 일어나는 구성으로 변경 — 사후 변경이라 addendum 불가, 새 PREREG rev2·새 튜닝 seed·규칙층 감사 필요, "움직이는 구성을 고른다" 자체가 새 자유 표면.
(A) 시 승계 caveat: "TUNE에서 선택된 동적 arm(anchor D44)은 4/4 run에서 FEAS 혼잡 가드에 막혀 전환 0회였으며, EVAL의 dyn 대 static 비교는 같은 분할의 코드 경로 비교이고 HE0(동적 대 best static) 판정이 아니다." · "이 캠페인은 HE0 셀 라벨을 산출하지 않는다."
