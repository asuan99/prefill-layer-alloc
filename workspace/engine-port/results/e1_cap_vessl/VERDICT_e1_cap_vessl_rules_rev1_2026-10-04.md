# 판정서 — E-1 cap 대조 VESSL 사전등록 rev1 규칙층 감사 (claims-auditor, 2026-10-04)

> 읽기 전용, GPU 0, 파일 수정 0. 감사자에게 쓰기 도구가 없어 메인 세션이 보고 본문을 옮겨 기록했다(감사자 Monte Carlo·재계산은 세션 scratchpad).
> 대상: `PREREG_E1_CAP_VESSL_2026-10-04.md`(rev1), `e1_cap_vessl.sh`, `e1_plan.py`, `tests/test_e1_cap_vessl.py`, 미커밋 엔진 diff `src/multiplex/multiplexing_mixin.py`(PDMUX_PHASE_EVENTS) + `tests/test_phase_events.py`.

## 등급: `GO-with-caveats`

死因 0(N1–N4 수치로 미성립), 등록 caveat 9건(E1C-1…9). 실행 승인이지 "후보 (7)을 닫았다"가 아니다. ★어느 라벨이 나와도 **KISTI HE0의 후보 (7) 판정은 바뀌지 않는다**(E1C-1).

## 독립 검증

- 엔진(dev tree `model_runner_kv_cache_mixin.py`): `handle_max_mamba_cache` elif(radix off ∧ cap 지정 → pool = cap), `_resolve_max_num_reqs`의 `min(cap, pool // ratio)`(radix off ratio 1) 확인 — PREREG §3-4 1–3 정확, "cap 192 + pool 48 구성 불가" 정확 ⇒ M→C192 = pool 고정 cap 대조 성립. static 경로(`adjust_stream_groups`)는 cap 비의존; cap 의존은 FEAS 가드(D arm)·snapshot capacity뿐.
- HE0 로그(856889): boot ≈60 s, 캡처 bs 10개 × 3 group 16.8 s·0.42 GB, `max_total_num_tokens=327601`, `full token usage` 최대 0.09 ⇒ BOOT_S 300 s 보수적, M3 여유 예보(측정 전).
- 로컬: `e1_plan.py --selftest` OK; `--budget` 3.05 GPU-h/Job · 18.3/6 Job · 천장 33.33 (PREREG §8 일치); `test_e1_cap_vessl` 29 통과; `test_phase_events`는 로컬에 sglang/sgl_kernel이 없어 **재현 못 함**(작성자 컨테이너 결과 의존).

## 질문별 판단

1. **주 지표 vs HE0 부호 충돌**: 실체 충돌 (a)는 과부하 버스트 prefill-ward 이동의 TTFT 부호(AUDIT_LOGIC A2) ⇒ TTFT 타당. KISTI 원자료(`results/slo_sched/sgptvHi_d{24,44}_*_L3H12_*.jsonl`, 896689 제외) C48 대조: TTFT p50 +0.157, p95 +0.108, p99 +0.117, mean +0.111(859006 제외); goodput legacy·ttft-only·ITL65 ln(d44/d24) +0.076~+0.078 — **전 지표 같은 부호** ⇒ C48에서 지표 선택 반전 없음. 단 HE0 headline은 goodput 순위 ⇒ E-1 라벨은 HE0 goodput 부호 증거 아님(E1C-4). C192에서 ITL이 반대 부호를 낼 수 있음(예보) ⇒ E1C-4. ★§4 근거 "p50은 정지 꼬리 비민감"은 **원자료로 반증**: 859006 포함 시 Δ48(p50) 0.157→0.231, 859025(d16) HI p50 683 ms vs 같은 arm 2095–2225 ms. 라벨은 못 뒤집음(이상치 1블록은 CI를 넓혀 AMBIGUOUS로) ⇒ E1C-6.
2. **한 노브 분해(#10)**: M→C192는 cap과 함께 추가 cudagraph 캡처(bs 56–192), decode 배치 크기, prefill 배치 구성(new-seq/입장, `max_prefill_tokens=16384`, `split_forward_token_budget 65536 // extend_num_tokens` split 횟수 — PREREG 미공시)을 움직임. cap의 매개변수라 불가피, 死因 아님 ⇒ 라벨은 "cap 노브 효과"까지, 슬롯 회전 기전 아님(E1C-5). KV 축소는 C48→M에 격리, FEAS는 D arm만. 빈틈 1: M2가 arm 합동 중앙값 — S24_C192만 구속돼도 통과 가능, 편향은 NOT_CAP_BOUND 쪽(수치 없음). 빈틈 2: M2·M3는 PrefillAdder 토큰 예약 구속을 못 봄.
3. **검출력·N3**: 등록 `decide()` 그대로 MC 4000회(seed 1), 런 σ KISTI d34 0.045·d44 0.075, 참값 ΔM=Δ48=0.17.

   | 참값 | σ 0.045 n6 | σ 0.075 n6 | σ 0.075 n4 |
   |---|---|---|---|
   | 완전 구속(Δ192=0) | ATTENUATED 0.92 · VANISHED 0.01 | ATTENUATED 0.49 | UNRESOLVED(baseline/sign-lost) 0.67 |
   | 부분 구속(Δ192=0.10, 41% 감쇠) | NOT_CAP_BOUND 0.57 | NOT_CAP_BOUND 0.30 | — |
   | 비구속(Δ192=0.17) | NOT_CAP_BOUND 0.97 | 0.64 | 0.18 |
   | 반전(Δ192=−0.10) | REVERSED 0.86 | 0.35(+ATTENUATED 0.38) | — |

   **N3 아님**(양쪽 결론 도달 가능). VANISHED는 사실상 죽은 라벨(≤1%). 결함은 허용 문장: NOT_CAP_BOUND "3% 넘게 줄지 않았다"는 부분 구속 참값에서 30–57% 거짓, ATTENUATED 영문 "a penalty … remained"는 완전 구속 참값(92%)에서 거짓 — 고정 규칙의 문언 결함 ⇒ E1C-2/3.
4. **축소안**: 4블록 — σ 0.075에서 UNRESOLVED ≈67%, 완전 구속 ATTENUATED 0.92→0.62(σ 0.045), 검출력 손실 큼. D arm 제거 — 라벨 영향 0, 블록당 ≈0.45 GPU-h(6블록 ≈2.7 GPU-h) 절감, C-1 VESSL 입력·게이트 5 동적 기술 상실, `ABORT_ARM_COUNT 12` 수정 → 델타 감사. 어느 쪽이든 블록 1 제출 전 동결(E1C-9).
5. **엔진 변경 correctness**: 기본 OFF 경로·줄 위치 불변·관측 전용을 코드 독해로 확인. CPU fake 변이 시험 구조 적절(로컬 재현 못 함). CG2는 동시성 1이라 동거 경로 미경유 — 출력 동치는 비동거 경로만. 실질 검사는 `decode_iteration` 0건 시 중단뿐. mamba triton 커널 autotune 없음 ⇒ CG2 결정론으로 Job이 막힐 위험 낮음(E1C-8).
6. **기판 혼합(#11)**: PREREG §2(a)–(d) 적절. 누락: E-1 동거율 (a)로 KISTI CORESIDENCY 31/39/57% 폭을 좁히는 통로, 정본 §1-4 KISTI 귀속 판정 불변성 미명시(E1C-1).

## 반전 시험 표

원자료: KISTI HE0 HI jsonl(d16/d24/d34/d44 각 4런, 896689 제외), `analyze.load_bench_serving_rounds`·`percentile`; 예보는 등록 `decide()`, σ·n은 위 값만.

| # | 자유 표면 | 범위 | 판정 변화 | 근거 |
|---|---|---|---|---|
| 1 | 주 지표 p50 | p95·p99·mean·goodput 4술어 | 없음(C48) | +0.108…+0.318 / gp +0.076…+0.094, canonical +1.06(cliff). C192 데이터 없음 → E1C-4; p50 정지 민감 → E1C-6 |
| 2 | d24 vs d16 | d16 | 없음(C48) | ln(d16/d44) = +0.379 |
| 3 | cap 192 값 | 등록 점값 | 시험 불가 | M2 합동 중앙값이 arm 비대칭 구속을 가림 → E1C-7 |
| 4 | S34_M 없음 | — | 없음 | S34는 라벨 입력 아님 |
| 5 | M1 0.50/M2 0.10/M3 0.90/pin 0.99 | M1 0.50→0.67 | 없음 | KISTI 입장 기준 0.70(0.67–0.75), KV 최대 0.09 |
| 6 | G=ln1.03, t-CI | t↔bootstrap, n 4–6 | 라벨 클래스 방향 불변, 확률만 이동 | MC 표 → E1C-2/3 |
| 7 | 블록=Job | — | 없음 | 블록 내 시점 효과는 무작위 순서로만; clk 병기 |
| 8 | 시드=블록 | — | 없음 | 기계적 |
| 9 | 예비 블록 | 0–2 | 수치 없음 | 무효 사유 전수 보고 |
| 10 | CG2 결정론 | — | 라벨 무관 | autotune 없음 → E1C-8 |
| 11 | 이벤트 시각 정의 | ±1 iteration | 라벨 무관 | pin·동거율 기술만 |
| 12 | 관측자 효과 d34만 | — | 라벨 무관 | 이벤트 μs vs iteration ms → E1C-8 |
| 13 | 캡처를 cap 처치에 포함 | — | 없음 | E1C-5 |
| 14 | FEAS 0.85×cap | — | 라벨 무관 | D arm만 |
| 15 | seed 1, green read-out | — | 없음 | 대칭 |
| 16 | 엔진 트리 | — | 없음 | VESSL 내부 비교 |
| 17 | preflight 부분집합 | — | 없음 | 러너 digest 기록 |
| 18 | 로그 파서 | 0건 | M3 fail-closed; M1/M2 부분 NaN은 `med`가 조용히 제외 | → rev2 R6 |
| 19 | GPU 경로 미검증 + 블록 1 후 rev2 | 블록 1 결과로 설계 변경 | 열린 갈래 | → E1C-9 |
| 신규 | 축소안(4블록/D 제거) | 6→4 | 확률만 이동 | UNRESOLVED 0.24→0.67(σ 0.075) → E1C-9 |
| 신규 | 클라이언트 tenancy(#7) | — | 수치 없음 | 859025형 송신 정지 → E1C-6 |

## 등록 caveat (결과 문서·정본이 문자 그대로 승계)

- **E1C-1 (기판·범위)** — "E-1의 어떤 라벨·수치도 KISTI HE0(정본 §1-4·§1-7)에서의 후보 (7) 판정을 바꾸지 않는다. KISTI의 'backlog의 cap 48 귀속 NOT-YET-SUPPORTED'는 그대로 남고, E-1은 VESSL 노드 집합에서 같은 서빙 arm으로 시험한 별도 결과로만 등재한다. E-1의 동거율 (a)·cap-bound 값으로 KISTI CORESIDENCY 추정 폭(31/39/57%)이나 C-4 값을 좁히거나 보정하지 않는다."
- **E1C-2 (NOT_CAP_BOUND)** — "NOT_CAP_BOUND는 'cap 192(pool 192)에서도 d24의 HI TTFT p50 불리가 3% 넘게 남았다(부호 유지)'까지만 뜻한다. 'cap을 올려도 불리가 줄지 않았다 / 3% 넘게 줄지 않았다'로 쓰지 않는다. DiD_cap 비유의는 감쇠 부재의 증거가 아니며, DiD_cap 평균과 CI를 반드시 병기한다(사전 예보: 41% 감쇠 참값에서 NOT_CAP_BOUND 30–57%)."
- **E1C-3 (ATTENUATED / VANISHED)** — "CAP_ATTENUATED는 'cap 48→192가 d24 불리를 3% 넘게 줄였다(DiD_cap CI > 0)'까지만 뜻한다. '불리가 남았다'는 Δ192가 sig_pos일 때만 쓰고, 아니면 'cap 192에서의 잔여 불리는 판정되지 않았다(Δ192 CI 병기)'로 쓴다. CAP_BOUND_VANISHED는 n ≤ 6에서 사실상 도달 불가(예보 ≤ 1%)이므로, 그것이 나오지 않았음을 '불리가 사라지지 않았다'로 읽지 않는다."
- **E1C-4 (TTFT 한정 / 정책 금지)** — "모든 E-1 라벨은 HI TTFT p50 부호에 관한 것이다. goodput·ITL·HE0 순위·실무 권고(decode-heavy static)·'버스트에서 prefill-ward가 낫다'로 확장하지 않는다. 이 금지(PREREG §6-4 금지 8)는 `e1_plan.FORBIDDEN_TEXT`에 빠져 있으므로, 이 문장이 코드 출력보다 우선한다."
- **E1C-5 (기전)** — "cap 48→192 처치는 running 상한과 함께 decode 배치 크기, prefill 배치 구성(입장당 new-seq, split forward 횟수), 추가 cudagraph 캡처(bs 56–192)를 움직인다. CAP_BOUND_* 라벨은 cap 노브의 효과이지 '슬롯 회전 기전'을 확인한 것이 아니다."
- **E1C-6 (p50 정지 민감)** — "TTFT p50은 정지에 견고한 지표가 아니다. KISTI에서 859006을 포함하면 Δ48이 0.157에서 0.231로 바뀌고, 859025 d16의 HI p50은 683 ms로 같은 arm의 2095–2225 ms와 다르다. 런별 서버 정지(로그 무음 ≥ 1 s)와 클라이언트 정지 사건은 라벨과 함께 기술 집계로만 병기하며, 사후 제외는 금지한다."
- **E1C-7 (M2 합동 / 비-cap 구속)** — "M2 통과는 S24_C192·S44_C192 합동 중앙값 기준이다. 라벨을 인용할 때 arm별 중앙값을 병기한다. S24_C192 단독 중앙값이 0.10을 넘으면 NOT_CAP_BOUND·ATTENUATED를 'cap 192가 d24에서 여전히 일부 구속' 단서 없이 인용하지 않는다. M2·M3는 토큰 예약(PrefillAdder) 구속을 보지 못한다."
- **E1C-8 (관측자 / CG2)** — "OBSERVER_OK는 관측자 효과가 없다는 증거가 아니다(d34만 측정, CI가 0을 포함할 뿐). CG2 통과는 동시성 1(동거 없음) 경로의 greedy 출력 동치만 보인다. 계측이 동거 경로의 스케줄링을 바꾸지 않는다는 근거는 CPU fake 시험뿐이다."
- **E1C-9 (절차 동결)** — "본 블록 수(6 또는 4), D arm 포함 여부, BOOT_S·RUN_NEED_S 追記는 블록 1 제출 전에 코드 상수로 고정해 REGISTRATION_SHA256에 남긴다. 블록 1 회수 후의 rev2는 엔지니어링 실패(boot·CG2·events 0·banner·pin 측정 불가)로만 발동할 수 있고, 발동하면 그 이전 블록은 라벨에서 제외한다. M1–M3·Δ 값을 보고 설계를 바꾸는 것은 등록 위반이다. UNRESOLVED_*는 후보 (7)의 음성 결과가 아니다."

## rev2 수정 요구 (권고)

GO-with-caveats는 rev1 그대로 실행을 승인한다. 채택하면 그 diff만 델타 감사. R1–R3는 문언만(판정 규칙 불변) — 제출 전 반영 강력 권고.

- **R1.** `ALLOWED_TEXT` NOT_CAP_BOUND / CAP_ATTENUATED를 E1C-2/3 부호 수준 문장으로 교체(코드 상수 출처 허위 = 교훈 80 계열).
- **R2.** `FORBIDDEN_TEXT`에 PREREG 금지 8 추가(현재 PREREG 8개 vs 코드 7개).
- **R3.** PREREG §4 "p50은 정지 꼬리 비민감" 근거 문장 정정.
- **R4.** M2를 arm별 중앙값의 최댓값 ≤ 0.10으로(규칙 변경 → 델타 감사).
- **R5.** 기술 지표 M2b 등록: HI `decode_iteration` 중 `queue_len > 0 ∧ ¬prefill_in_flight ∧ bs < cap` 비율(비-cap 입장 차단, 라벨 입력 아님).
- **R6.** M1/M2 부분 NaN의 중앙값 조용한 제외 방지 — NaN 수 병기, 과반 NaN이면 unmeasured.
- **R7.** 블록 1 이후 rev2 규칙(E1C-9)을 PREREG §10-6·§12-19 본문에.
- **R8 (선택).** NOT_CAP_BOUND에 동등성 조건(DiD_cap CI 상한 < G) 추가 — 규칙 변경·검출력 비용.
- **R9.** 정지 사건 판정기(서버 무음 ≥ 1 s, HI 창)를 기술 필드로 사전 등록(제외 규칙 아님).
