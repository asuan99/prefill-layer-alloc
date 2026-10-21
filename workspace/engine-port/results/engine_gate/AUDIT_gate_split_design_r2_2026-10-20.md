# AUDIT — gate 하네스 재감사 2회차(상한), diff `9561f867..5ce338d8` (claims-auditor, 2026-10-20)

**G-α 재실행: GO-with-caveats · gamma: NO-GO(N2) → 사용자 결정**

1. **사후 수정의 정당성 — MAJOR(차단 아님).** 1회차 B2 원자료를 확인하니 bs 40 미도달은 맞지만, 작성자가 설명한 기전은 **반증**됐다.
   - 실제 경과: stair\_span +0.068 s부터 요청 0(출력 16)이 bs=1로 decode 15 step을 돌고 끝났다. +1.63 s에 나머지 39개가 **단일 extend로 일괄 합류**했다(n\_extend=2, bs 1→39). 그 뒤 39→1로 내려갔다.
   - 따라서 "한 개씩 합류"가 아니라 요청 0 종료와 일괄 prefill 사이의 **경합**이다. 재실행 M은 39 또는 40일 수 있다. 수리: REG 주석과 §9-1 문장을 이 관측으로 바꾼다(상수·digest 불변).
   - 새 기준의 판정력: 연속성은 출력 길이가 모두 달라 거의 구조적으로 성립한다(probe 누락만 잡는다). M==I-5는 같은 batch를 두 hook에서 센 값의 대조라 항등식은 아니지만 독립 증거도 아니다. 실질적인 판정력은 M≥32 하나다(직렬 합류를 잡는다).
   - 반전 시험: 하한 1–39 / 40이면 PASS / FAIL(구 기준). window 선택은 중점이면 PASS, 겹침이면 FAIL인데, 겹침 규칙은 cap 단계의 48을 끌어오므로 정의상 무효다 → caveat.
   - "재실행 데이터로만 판정"한다는 공시는 충분하다.
2. **OFF/OFF 진단 — BLOCKER(gamma).** score\_gamma 안에서는 pair ≥100이 TOST에서 빠진다. 그러나 `run_gate`(engine\_gate\_driver.py:1521)는 stage 하나라도 SKIPPED/UNRESOLVED면 PASS를 UNRESOLVED로 낮춘다. 등록 추정 EST=650 s에서 OO 102가 시작하려면 t ≤ 650 − s0/22여야 하므로 **확정적으로 SKIPPED**된다. 결국 진단이 headline 판정을 뒤집는다. `test_diagnostics_never_change_verdict`는 rescore 경로만 시험하므로 이 경로를 잡지 못한다. seed 공유는 문제없다. n=2라 SD를 추론 근거로 쓰는 것은 금지한다.
3. **STALL\_BOUNDARY — MINOR.** 5%±1%p는 1−0.95에서 연역한 값이다. 30 ms는 J1 예보에서 gamma 이전에 고정됐다. 다만 agnostic stall 봉우리가 30–45 ms라 문턱에 민감하다(라벨 전용이므로 caveat).
4. **gamma 시간 예산 — MAJOR.** 드라이버 `timeout`(270분)이 하드캡을 자체 집행하고 설치 시간은 예산에 포함되지 않으므로, 이건 비용 문제다. ON/OFF가 먼저 돌고 짝 단위 `fits()`가 있어 미완 경로는 없다. 실제 위험은 2번의 판정 뒤집힘이다.
5. **교훈 276에 해당한다**(진단을 넣자 판정 경로가 오염됐다. 설계 문제가 아니라 배선 결함).

## 인용 금지 문장 (G-α 재실행 결과 문서가 승계)
- 「B2 stair의 PD-mux 기준(하한 32)은 1회차 M=39를 본 뒤 정했다.」
- 「bs 40 미도달은 요청 0의 종료와 39개 일괄 합류 사이의 경합에서 나온 결과이며, PD-mux 스케줄의 불변량으로 인용하지 않는다.」
- 「M==I-5는 두 hook 사이의 계수 대조이지 독립 증거가 아니다.」

## 사용자 결정 (gamma)
- (a) **권고**: OO 짝을 gamma에서 빼고 r1 gamma(GO-with-caveats)로 제출한다. OO는 확장 Job으로 옮긴다.
- (b) 1521행에서 pair ≥100 stage를 제외하고 run\_gate 경로 시험과 변이 시험을 더한 뒤, 재감사 없이 사용자 승인으로 처리한다.
- (c) OO를 1짝으로 줄인다(1521 결함은 남는다).
