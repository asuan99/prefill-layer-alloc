# 결과 감사 r1 — CAP1-R 용량 결과 (2026-10-29)

- 감사자: claims-auditor(읽기 전용, CPU 재계산) · 1/2회 · 대상 `RESULT_cap1r_2026-10-29.md`, raw `cap_char/policy_runs/pdmux-policy-cap1r-4bef3cf-20261029003539/raw`
- 재계산: RAW_SHA256SUMS OK(run 100), `ph_analyze.cap_addendum` 재실행 → λ*=7.29491, D44, `CAPACITY_BRACKETED`, rate별 attainment 표 일치.
- **진행: CAP2 GO-with-caveats · HE0 GO-with-caveats · H-D GO-with-caveats**(H-D 결과에 W1 phase별 TTFT 추세 기술값 병기)

## 판정
1. λ* 산정 — **CONFIRMED**. attainment = Σgood/Σn(두 seed 합) ≥ 0.90, 첫 실패 직전 rung(§3-3·§13-1 D4). 동률 처리는 미등록: dict 삽입 순서(run 파일명 사전순)로 정해지며 뒤집으면 best_D=64, λ* 불변.
2. UNSTABLE 셀 λ*를 rate 입력으로 — **PLAUSIBLE(조건부)**. HE0 라벨은 같은 Job 내 짝 비교라 부호·라벨 타당성을 직접 해치지 않음. 공시: S1이 이 셀 크기에서 판별력 없음(attainment 1.0인 5.516 셀도 UNSTABLE, D44 s201 S1 2.15; 100 중 99 UNSTABLE). λ*는 120요청(≈16 s) 버스트 용량 — D44 7.29 s201 TTFT 중앙값 quintile 81→108→261→1628→2859 ms. HE0 HI(200요청 ≈18 s)는 척도 일치, H-D q=1.0 W1 phase(875/4375요청)는 셀의 7–36배로 지속 과부하 위험 미측정.
3. throttle 교락 — **PLAUSIBLE(순위 비교락, 크기는 기판 상태에 묶임)**. D16·D24 5.5에서 attainment 0.09/0.29 vs 상위 D 1.0(9pp throttle로 설명 불가), 순위 경계 D34 vs D44 throttle 13.8% vs 13.5%, D64 최다 throttle인데 D44와 동률, 실행 순서 무상관. Job 단위 throttle CAP1 3.7% → ≈13%. boot 창 복원은 server.log 첫/끝 시각과 ≤6 s(창의 1% 미만) 일치, 셀별 throttle은 없음.
4. 동률 D44 vs 사전 지정 D44 — **REFUTED(문제 아님)**: HE0 (a) D44는 KISTI 승자로 데이터 전 지정, (b)는 평가 seed 사후 선택, CAP argmax는 어디에도 안 들어감.
5. HI=10.94 — **PLAUSIBLE**. 과부하 의도와 맞으나 상위 D가 전부 실패는 아님(11.09에서 0.74–0.78 고원). HI 인접 rung(9.65·11.09) TTFT ±5% 비율 최대 7.5% < 10%. 이탈: CAP1 기준 D16 용량 1.66 < LO 2.74 → LO는 KISTI LO처럼 분할 무관 구간 아님(Job 간 기술값).
6. CAP2 사다리 유지 — **PLAUSIBLE**. 바꾸면 미등록 변경. 위험: W1 λ*가 §7 가정(3)의 2.4배 → W3도 같으면 ≈0.7로 최상단 0.76 근접 → bracket 실패 가능(등록 1회 재실행이 처리). caveat 6의 9%/33%는 λ 비 10:1 가정, 실측 비는 CAP2 뒤 재계산(예: 36:1이면 W3 비중 ≈3%).

## 반전 시험
동률 규칙 D44→D64 · λ*의 bracket 내 위치 [7.29, 8.39) · attainment 문턱 0.85–0.95 · seed 합산 vs seed별 — 모두 판정 변화 없음.

## caveat (문자 그대로)
1. "λ*_W1=7.29491은 120요청(약 16 s) 셀의 버스트 용량이며 지속 용량이 아니다. 용량 셀은 두 seed 모두 S1 UNSTABLE(15.9, 30.7)이고, 이 크기의 셀에서 S1은 attainment 1.0인 셀에도 발화한다."
2. "H-D의 q=1.0 W1 phase(120/600 s)는 용량을 측정한 셀 길이의 7–36배이며, 그 길이에서 λ*가 지속 가능한지는 측정되지 않았다."
3. "CAP1-R의 throttle(SW power cap)은 boot별 8.6–17.7%로 CAP1(2.4–4.4%)보다 높다. λ*의 크기는 이 Job의 기판 상태에 묶이며, 후속 Job에서 HI/λ*의 실효 배율은 같다고 보장되지 않는다."
4. "D44·D64 동률의 argmax=44는 구현 순서의 결과이며 등록된 동률 규칙이 아니다(λ* 불변)."
5. "LO=2.74는 CAP1 기준 D16 용량(1.66)보다 높아 KISTI LO처럼 분할 무관 구간이 아니다(Job 간 기술값)."

## 금지 문안
"Nano W1 용량은 7.3 req/s다"·"D44≈D64가 용량 포화점이다" 류 크기·관절 인용 · "throttle이 높은 D의 용량을 깎았다/깎지 않았다" 인과 서술 · CAP1과 CAP1-R 용량 차를 기판 변화·사다리 효과로 해석 · caveat 6의 9%/33%를 CAP2 전에 인용.

## 확정 실험(선택, 사용자 결정)
λ* 지속성: 한 Job 안에서 W1, D44·D64 × 7.29 req/s × 600 s × seed {201, 202}, 같은 노드 throttle 기록, phase 구간별 S1.
