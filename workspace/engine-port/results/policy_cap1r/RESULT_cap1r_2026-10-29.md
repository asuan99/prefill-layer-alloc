# CAP1-R 결과 (2026-10-29) — 기술값, 미감사

Job `pdmux-policy-cap1r-4bef3cf-20261029003539` (job-lrd99x21e2yi), PREREG rev1.4 §19(사다리 5.516×1.15^k, k=0..9, 5 D 전부, W1, seed {201,202}, 셀당 120 req). λ\*는 이 Job 하나의 데이터로만 산정(원 CAP1 데이터 미사용; `cap_addendum`은 다른 CAP Job run이 섞이면 `ABORT_MIXED_CAP_JOBS`로 거부하며 이번 입력은 CAP1R만). **아래 수치는 측정 기술값이며 성능 판정이 아니다.**

## 1. 무결성
`RAW_SHA256SUMS` 229줄 전부 OK, `RAW_EXPORT_OK` 있음, run 100개. trace 동일성((rate, seed)별 D 간 trace_hash) 위반 0. correctness 5 boot 모두 n_bad=0(n 105–122). `switch_count` 전부 0(static). 엔진 digest 일치(`ENGINE_DIGEST_CHECK`).

## 2. 용량 (attainment ≥ 0.90, 2 seed 합산)
| D | 마지막 통과 | 첫 실패 |
|---|---|---|
| 16 | 없음 | 5.516 (att 0.087) |
| 24 | 없음 | 5.516 (0.292) |
| 34 | 6.3434 | 7.29491 (0.896) |
| 44 | 7.29491 (0.971) | 8.389146 (0.700) |
| 64 | 7.29491 (0.975) | 8.389146 (0.708) |

- 비단조 없음. 상위 D(34·44·64)는 8.39 이후 0.52–0.78대에서 수렴(고원).
- **`CAPACITY_BRACKETED`.** argmax D = 44(D44·D64 동률 7.29491, 코드는 먼저 나온 44), **λ\*_best-static(W1) = 7.29491 req/s**. `rerun_exhausted_user_decision = false`, 재실행 불필요.
- 원 CAP1과 겹치는 영역의 기술 비교(데이터 합산 아님): D34 용량 CAP1 5.516(첫 실패 7.447) vs CAP1-R 6.343(첫 실패 7.295), D44·D64 CAP1은 7.447에서 0.93/0.95 통과였으나 CAP1-R 사다리의 7.295에서 0.97/0.98, 8.389에서 0.70. 사다리 격자가 달라 직접 대응 점은 없으며 용량 인접 구간에서 attainment가 가파르게 떨어진다는 점만 일치한다.

## 3. 검사
- **정상성: 용량 셀(D44, 7.29491) S1·S2 = seed 201 (30.7, 0.18), seed 202 (15.9, 0.00) → 두 seed 모두 `UNSTABLE`**(S1 > 1.7; 마지막 1/3 TTFT 중앙값이 처음 1/3의 16–31배: 120 req 셀 안에서 큐가 쌓임). D34·D64도 동일. 100 run 중 99 UNSTABLE(사다리 5.516 이상 전 구간이 용량 근처 이상). → PREREG caveat 3: 이 λ\*로 정한 rate의 결과는 크기를 인용하지 않는다. S3는 미검사.
- CLIFF_ADJACENT(TTFT ±5% 요청 비율 > 10%): 50셀 중 0(최대 7.5%; 용량 셀 D44 5.4%, D64 2.5%, D34 5.8%).
- 용량 셀 TTFT p95: D44 2922 ms, D64 2791 ms(SLO 3000에 근접).

## 4. 기판·throttle
단일 Job·노드. 전부 SW power cap(reason 0x4)이며 SM clk 중앙값 1410 MHz. boot(D)별 창 내 비율: D16 8.6%, D24 11.4%, D34 13.8%, D44 13.5%, D64 17.7%. 창은 clk 추적의 유휴 구간(보트 사이 ≈ 10 s)에서 복원한 근사(run mtime은 복사 시각이라 쓰지 못함; `throttle_by_D.py`). 높은 D일수록 비율이 높고 CAP1(2.4–4.4%)보다 큼 — D 간 비율 차(최대 9 pp)는 arm 간 기판 동일성이 완전하지 않음을 뜻하므로 D 간 용량 차를 해석할 때 공변량으로 병기. 실행 순서(boot 순): D34, D64, D24, D16, D44.

## 5. addendum 초안 (미봉인·미제출) — PREREG 규칙 그대로
- HE0 rate: `he0_rates(λ*)`, h = 1.5(O-P3), HI/LO = 4 → **HI = 10.9424, LO = 2.7356 req/s**. 주의: HI는 모든 D의 첫 실패 rate(8.39)보다 높아 HI 구간에서 static도 attainment 약 0.7대. 이는 등록된 h(초안값)의 귀결이다.
- CAP2(W3): 사다리 §15-2 `W3 rmin = 0.1 ×1.5, 6점`. W1 λ\*에서 W3 사다리를 도출하는 등록 규칙은 없어 그대로 둔다(변경은 사용자 결정).
- 파일: `addendum_cap1r_result_DRAFT.json`.

## 6. 산출물
`cap1r_analysis.json`, `throttle_by_D.json`, `cap1r_capacity.png`(λ\*_D 막대+bracket, rate별 attainment), 위 addendum 초안. 원자료 `cap_char/policy_runs/<job>/raw/`(gitignore 규칙, 매니페스트만 추적). 스크립트: `policy_cap1/analyze_cap1.py`·`plot_cap1.py` 재사용, `throttle_by_D.py`(CAP1-R용).

## cost model 기여
W1(Nemotron Nano 9B, 이 기판)의 static D별 용량: D16·D24는 ≤ 5.5 미만(CAP1: 1.66·4.09), D34 6.34, D44·D64 7.29로 포화(D44≈D64). D ≥ 44에서 용량 이득이 사라지는 지점 후보 — 분할 D 대비 용량 곡선의 관절이 약 44 근처임을 기술. 용량 셀이 S1 UNSTABLE이므로 이 값은 λ\*(rate 산정 입력)로 쓰되 크기 인용은 금지.
