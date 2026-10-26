# 규칙층 감사 r1 — PLAN_DP_routing rev0 (커밋 `00f5b763`), 2026-10-26

> claims-auditor(읽기 전용) 판정을 main 세션이 옮겨 적었다. 같은 대상 최대 2회 중 1회차.

## 전체 판정
rev0 그대로 PREREG화 = **NO-GO**(死因 N1·N3 + 결정 14·24 충돌). 수리 반영 시 `GO-with-caveats` 후보 — 수리 1·2는 사용자 결정 필요. **9.5 h 정지 판독은 REFUTED, 즉시 정정**.

## 死因·MAJOR 요약
- 1a 死因(N1): Llama는 s = 0 → c\* = 0 → τ ∈ {c\*, 2c\*} = {0}이라 전 요청이 긴 쪽, R3/R2 ≤ 1이 구성상 강제 → 등록 예측이 측정 전에 참(항등식).
- 1b 死因(정본 충돌): 결정 14("모델 간 goodput 이중차분 폐기")·결정 24("Llama 대비 귀속 불가, 참조점 1–2셀")·비용 항 PREREG §4-2 금지 문안과 충돌. Transformer에서도 길이 분리는 HoL·KV 단편화·batch 균질화로 이득 가능.
- 3 死因(N3): τ = 2c\*는 자연 출력 arXiv-Sum에서 긴 쪽 0%(ARXSUM_STATS 유지분 5,906건 최대 입력+출력 16,744 < 17,520; 입력 > 8,760 = 30.0%), shape C에서는 τ 선택으로 긴 쪽 비율이 7배 이상 움직임. 봉인 2단계(구조 fetch 전 / 수치 fetch 뒤) 미명시, "decode floor 최소 D"는 T1만으로 정의 안 됨(T3 ρ 필요).
- MAJOR: R4가 D·cap 동시 변경, R3 공통 D가 LEN 분할 아래 미재조율 · fused·LEN arm 부재(라우팅 × 엔진 상호작용 식별 불가, 분모 미조율) · 클라이언트 라우터가 `ignore_eos`로 실제 출력 길이를 앎(출력 길이 oracle, `trace_loadgen.py:64-66`) · 단일 클라이언트 GIL이 ITL p95에 직접 들어감 · 분모 R1 정당성 · **집계 attainment가 긴 클래스를 (1−a)/p\_long까지 버려도 통과**(R3·R4 쪽 고정 편향) · 게이트 2(O-D4) 미결 · cliff 규칙 없음 · Job을 config별로 쪼개면 한 비교의 arm이 다른 Job으로 갈림.
- MINOR: LL "엔진 `TOTAL_TOKENS` 대응"은 사실 오류(엔진 값 = 점유 토큰 + 대기 seqlen, `scheduler_metrics_mixin.py:866-883`) · N 서버 자원(GPU당 11 CPU·119 GiB, ×8 NUMA 2) · 정적/동적 경계(라우팅은 요청별 상태 의존 정책 → 게이트 2·3 적용) · duration 합산 규칙 · 5 arm 순서 설계 · 결정 30 형식 · "T1–T4 그대로 적용"은 가설 · 셀당 175 s 출처 추적 불가, 과소 예보 1.46× 적용 시 모델당 약 85 GPU-h.

## 9.5 h 정지(§7) — REFUTED
J2 11:43:08Z→21:14:02.986Z(9.5153 h) · J4 11:43:18Z→21:14:03.027Z(9.5125 h) · J8 11:43:23Z→21:14:03.065Z(9.5111 h), 전부 `terminated`; J1 8.613 h `succeeded`. 동시 종료로 배제되는 것은 **Job별 정밀 타이머뿐**이고, **주기적 sweep으로 집행되는 9.5 h 경과 상한**(세 Job 모두 9.5 h를 40–55 s 넘긴 직후, 종료 순서 = 생성 순서)이나 계정 단위 일괄 종료와 구분되지 않는다. 계정 이력에 9.515 h를 넘긴 Job 없음. "경과 시간 상한 아님"은 새 증거 없는 격상이었다. Job ≤ 6 h 유지.

## 수리 문안(원문 요지, PLAN rev1에 반영)
1. [사용자 결정] 귀속 포기(a, 권고): Llama는 같은 Job 실재 Transformer 참조점 R2·R3 각 1셀, 모델 간 비·이중차분 미산출, Llama 분할은 Nano 기준 같은 요청 ID 집합 / (b) 결정 14 번복 + 상호작용 I = ln(R3/R2)\_Nano − ln(R3/R2)\_Llama.
2. [사용자 결정] O-D4: 정상 Poisson 사다리 λ\*\_SLO 1차 + 변화 trace(긴 요청 30%↔60% 교대 3라운드) 채택 / 미채택 공시.
3. τ = c\* 단일 봉인(부팅 로그 s·k), 셀별 f\_long(τ) CPU 사전 계산, 0.1 ≤ f\_long ≤ 0.9 밖은 `ROUTING_DEGENERATE`, 2c\* 미사용.
4. 봉인 2단계(구조: τ·집합 크기·R3 공통 D·R4 집합별 D·cap 규칙(목적함수·floor 수치 정의·쓰는 항)·LL 함수 = fetch 전 / 수치 = fetch 뒤 `DP_REG`).
5. R1′ `fused_tuned·LEN` 추가, 분모 = max(R0, R1, R1′), R3 D = LEN 분할 아래 공통 D, R4-D 1차·R4-DC 기술값, 상호작용 ln(R3/R2) − ln(R1′/R1) seed 짝 t(3).
6. LL 함수 = 미완료 요청의 (입력 + max\_new − 수신) 합 최소, 동률 작은 index, 전 arm 공통.
7. 공시: 오버헤드 0 + 출력 길이 oracle.
8. `FRONTEND_SATURATED`(송신 지연 p99 > 5 ms 또는 클라이언트 CPU > 85%), 클라이언트 대 엔진 간격 p95 차 > 3%면 `CLIENT_ITL_UNVERIFIED`([제안], 계측 OFF와의 충돌은 하네스층 확인).
9. λ\*\_SLO = 짧은·긴 두 클래스 각각 attainment ≥ a 동시 만족(a는 G0 승계), 전체 기준은 기술값.
10. `CLIFF_ADJACENT`·`G_UNRESOLVED`, 라운드 안 N 서버 goodput = good 수 합 ÷ 공통 벽시계, 라운드 간 합산, arm 순서 정순·역순 교대.
11. Job 분할 단위 = seed 블록(각 Job이 그 seed의 모든 arm·혼합 포함), Job = 짝 CI 블록 인자, 비용은 시간 모형 코드 + 1.46× 상한 공시, N = 8·축소는 결정 30 형식.
12. 9.5 h 서술 정정(위 §7).
