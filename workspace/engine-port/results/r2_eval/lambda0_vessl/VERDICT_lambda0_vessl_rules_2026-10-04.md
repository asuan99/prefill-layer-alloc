# 판정서 — λ0 VESSL 기판 이식 追記 규칙층 감사 (2026-10-04, claims-auditor)

> 읽기 전용 감사, GPU 0, 저장소 수정 0. 감사자에게 쓰기 도구가 없어 메인 세션이 보고 본문을 옮겨 기록했다.
> 대상: `PREREG_LAMBDA0_VESSL_2026-10-04.md`, `lambda0_vessl.sh`, `lambda0_vessl_plan.py`, `stage_lambda0_inputs.sh`, `tests/test_lambda0_vessl.py`.
> 대조: rev5 사전등록·追記, `VERDICT_result_lambda0_908623_2026-09-14.md` §3·§4-A·§6·§7, `e2_sticky_prereg/VERDICT_e2_substrate_portability_2026-09-18.md`,
> `handoff-report/vessl_operating_model_2026-10-02.md` §1·§9–§11, `scripts/vessl/{job_entry,launch,install_runtime}.sh`.

## 0. 등급

**`GO-with-caveats`** — 死因 0, 등록 caveat 14건(LV-1…LV-14, §4).
N1–N4 중 어느 것도 만들지 못했다. 판정량(셀별 achieved, rev5 라벨)은 측정이고, 정책 주장이 아니며, 모든 결과가 거짓 도달 가능하고, 반전 시험에서 판정을 뒤집는 수치를 만들지 못했다.
**실행 승인이다. 게이트 #6을 닫지 않는다. λ0R-8(iii) 정규화 교락을 닫지 않는다. 기판 동등성을 판정하지 않는다.**

## 1. 독립 재검증 (게이트 #110)

| 주장 | 재실행 | 결과 |
|---|---|---|
| rev5 11파일 + 판정 입력 5 + config 908623 바이트 동일 | `lambda0_vessl_plan.py --verify-digests` | `DIGESTS OK (11, 5, 1)` |
| 엔진 트리 25/25 동일 | B5 Job manifest vs 908623 `--engine-diff` | `IDENTICAL (0 differ)` |
| green read-out 코드가 908623 트리에 이미 존재 | `green_readout.py` ∈ 908623 manifest 25항목 | arm 차이는 env 플래그 1개(코드 차이 아님) |
| 테스트 | `test_lambda0_vessl.py` | 23/23 OK |
| kill 층 중복 | watchdog만 남긴 변이본 실행 | `fake_rc=143`, 20.5 s 절단, 이후 `NOT_ATTEMPTED_BUDGET` — 각 층 단독 절단 실측(테스트엔 timeout 단독 경우만 있었음) |
| 예산·입장 | `--budget`·`cell_cost_s` | 1.00× 2.57 GPU-h · 0.25× rev5 부분 14,961 s ≤ 15,600 s(여유 639 s); need(0.25×) a_r4 1,055 s · b_r3 1,439 s |
| Job 안 실현 분할 추정량 | 908623 `tel_a_r4`·`tel_b_r3`에 `e2_realized_mix.py` | 셀당 0.2 s; a_r4 E-time 13.64%, b_r3 97.63% — E2C-8′ 교체값 재현 |

## 2. 반전 시험 (요약)

원자료 `lam0_908623/cell_*.json`·`plan.json`, 쉬핑 함수·상수, 참조 비율 = 908623 λ\*(A 1.454×, B 1.033×; 비용 규모용, VESSL 사전값 아님).

| 자유 표면 | 범위 | 판정 변화 | 근거 |
|---|---|---|---|
| boot 여유 240 s | 110→1,000 s/boot | 없음 | 참조 시나리오에서 boot ≈1,017 s/셀 초과해야 11번째 rev5 셀 거부, 분산 마지막 셀 ≈657 s(B4 실측 151 s). 0.25× 코너는 λ\*\_VESSL이 KISTI의 17%(A)·24%(B)일 때라 미도달; 바뀌는 건 측정 실패 라벨뿐 |
| 입장 비율 0.25× | 0.25→1.0 | 없음 | 참조 시나리오에서 15셀 전부 입장; 1.0× 입장 후 마감 절단은 `UNRESOLVED` = `NOT_ATTEMPTED`와 같은 라벨 |
| 캡 시계 기준점 | job_entry↔script start | 없음 | 차이 ≈30 s, 여유 ≥5,000 s; `timing.txt` 부재 시 script start 폴백(기록) |
| 분산 시드 251·2630 | 다른 Ēbar 통과 시드 | 없음(기술) | 포화 top rung은 도착 실현 무관; 908623 seed 차 A 0.60%·B 0.31%; §4-4 λ\*는 s1만 사용 |
| 분산 셀 마지막 | 앞 배치 | 없음, 편향 보수 | ★정정: "s3·s4 1–1.5 h 뒤"는 A ≈40–56 min, B ≈7–23 min(LV-7) |
| 분산 포화·R0 | ratio 0.90 | 없음 | top rung ratio a_r4 0.382/0.390, b_r3 0.607/0.601; 비포화엔 VESSL 용량 ×2.36(A)·×1.48(B) 필요 |
| §4-4 Δ 추정량 | 라벨 λ\*↔n=4 평균↔최대 | 수치로 못 만듦 | KISTI 측 폭 A 0.60%, B 0.31% ⇒ LV-3로 라벨 λ\* 고정 |
| §4-4 n<4 | 문언↔n=3 | 거짓 양성 반전 없음 | 측정 부족을 발견처럼 표기 위험 ⇒ LV-4 |
| 집행기 절단 셀 analyze 생략 | rev5처럼 analyze | 없음 | bench_serving은 완료 시에만 output 작성 ⇒ 어차피 `UNRESOLVED` |
| green read-out 관측자 효과 | ON↔OFF | 못 만듦(미측정) | `init_pdmux` 1회·try/except; 배너 `max_total_num_tokens` 2,722,025가 유일 단서(LV-11) |
| 엔진 트리 drift=abort / staging / unset 2건 / timeout·setsid | — | 없음 | fail-closed; digest 16개 강제; 허용목록이 unset보다 먼저; setsid 실패 시 `UNRESOLVED` |
| watchdog pkill 패턴 | — | 없음(비용 축) | ★실제 경로(`timeout -k 30` WARM/BENCH, `sglang.*`)는 테스트 미경유 — 지워도 23/23 통과(LV-13) |
| resume 노드 혼합 | 이전 Job 셀 복사 | 수치로 못 만듦 | λ\*·SEED_REPEAT가 노드를 섞을 수 있음(LV-9) |
| 이중 kill 층 | 한 층 제거 | 없음 | 단독 절단 실측 |
| ★S1: KISTI 보정 `drain_pred_s`·`kappa_pred`(B 저측 자격) | VESSL 서비스 속도 이동 | 의미 변화(고정 상수라 N2 아님) | b_r0 drain/pred 1.570, b_r1 1.191, TOL 2.0 ⇒ B 브래킷 영역 [0.574, 0.979] req/s; λ\*\_VESSL(B)이 KISTI 대비 −17.7% 이하 또는 +40% 이상이면 `KNEE_BRACKETED[B]` 소멸 가능. A는 견고(하한 ≈1.47 req/s = −52%) ⇒ LV-2 |
| ★S2: Job 2 집계 | Job1·Job2 불일치 | 수치로 못 만듦 | 집계 규칙 부재 ⇒ LV-8 |

**반전 0건.**

## 3. 질문별 답

1. **KISTI 상수 의존**: λ_inf 2.10/0.675·FALLBACK 사다리는 job 907959 입력의 결정론적 재생(항등식, provenance 검사) ⇒ LV-1. B 저측 자격의 절대 보정값(S1) — 등록 문서 VC-2 "라벨은 수준 이동에 둔감"은 shape B에 대해 부분적으로 거짓 ⇒ LV-2. PREFLIGHT 900 s·0.25× 코너는 실시간 잔여시간 기준이라 자가 보정.
2. **908623 교락(λ\*(A)≠D44)**: 해소가 아니라 기록(`mix_<cell>.json` 4추정량·green read-out·§7-4 병기). `multiplexing_mixin.py` 바이트 동일 ⇒ 기전 동일, 크기는 이 Job이 잰다. E2(sticky 대조) 미실행 ⇒ λ0R-8(iii) 열림. 신규: telemetry 큐 포화 시 이벤트 폐기(`dropped_events`) ⇒ LV-12.
3. **n=4와 게이트 3**: "한 VESSL Job·한 노드, 포화 top rung throughput n≥4"까지만. 정책 지표 기준선 분산·노드 간 산포 아님, boot·seed·시점 미분리, df=3(SD 95% CI ≈ [0.57, 3.7]×s) ⇒ LV-6.
4. **E2 이식성 3경로**: 집행기 — 셀 구간만 막힘(preflight·tail·반출 무 kill 층; 크레딧 차단은 생성 차단) ⇒ "4.5 h 하드캡"은 과금 하드캡 아님. 상수 — 대부분 막힘(boot 여유 VESSL B4로 재산출, 엔진 manifest 검증 승격), 잔여 S1. 과금 모형 — 부분(launch dry-run은 단가만 표시) ⇒ LV-13. 셋 다 승인 문언 문제 ⇒ caveat.
5. **기판 혼합(#11) 문구**: 잘 된 것 — 비교만/합치지 않음, KISTI 값으로 사다리 미구성, 원인 귀속 금지. 빠진 것 — (i) §3이 λ0R-3·λ0R-8을 "이 Job에서도 그대로 참"이라 실행 전 단언 (ii) λ0R-8 문자 승계 시 E2C-8′ 인용 금지 정밀값 유입 ⇒ LV-5 (iii) "구별 불가 ⇒ KISTI 결과 이식 가능" 역방향 통로 ⇒ LV-3 (iv) Job 2·resume ⇒ LV-8·LV-9.

## 4. 등록 caveat (결과 문서·정본이 문자 그대로 승계)

> **LV-1 (인용 금지)** — "VESSL에서 λ_inf 술어가 FALLBACK을 냈다 / VESSL이 λ_inf를 재확인했다" 금지. 이 Job의 `LAMBDA_INF_DECISION`은 KISTI job 907959 입력(digest 고정)의 결정론적 재생이며 VESSL 측정이 아니다. λ_inf(A,B)=2.10/0.675는 KISTI 사전값(λ0R-5 승계).

> **LV-2 (필수 병기 — 라벨의 기판 의미)** — shape B 저측 자격(`low_side_candidate` ∧ `drain_model_ok`)은 KISTI 보정 절대 예측 `drain_pred_s`(4.18–4.21 s)·`kappa_pred`에 묶여 있다(908623 b_r0·b_r1 drain/pred 1.570·1.191, 허용 2.0). λ\*\_VESSL(B) ≲ 0.574 req/s(KISTI 대비 −17.7%)이면 B의 `KNEE_BRACKETED`는 용량 차이가 아니라 KISTI 보정 상수의 부적합으로 사라질 수 있다. VESSL B 라벨이 `KNEE_BRACKETED`가 아니면 "사다리가 knee를 놓쳤다"·"VESSL B 용량이 다르다"로 쓰지 않는다. `drain_disqualified_low_side`와 셀별 drain/pred를 병기한다. VC-2 "라벨은 수준 이동에 둔감"은 shape A에만 해당한다.

> **LV-3 (인용 금지 — §4-4 양방향)** — (a) Δ는 `LAMBDA0_LABEL.json`의 λ\* 차(VESSL − KISTI 908623 = A 3.053130 / B 0.697240)로만 계산한다. (b) "다르다"는 "이 VESSL Job(GPU UUID·hostname·시점 병기)의 λ\*가 KISTI job 908623(n=2)의 λ\*와 다르다"까지. "VESSL 기판의 용량이 KISTI와 다르다"는 금지(노드 1개). (c) "구별되지 않는다"로부터 "KISTI의 λ\*·격자·HE0·정책 순위·caveat 수치를 VESSL에 이식할 수 있다"는 금지(게이트 1 연장, #11).

> **LV-4 (필수 병기)** — VESSL 라벨이 λ\*를 내지 않거나 분산 블록이 `UNRESOLVED_N_BELOW_4`이면 §4-4는 "비교 불가(측정 부족)"로 쓴다. "구별되지 않는다"로 쓰지 않는다(게이트 #21). 분산 블록 top-rung 평균을 빠진 λ\*\_VESSL의 대용으로 쓰지 않는다.

> **LV-5 (승계 형식)** — λ0R-1…10과 rev5 caveat는 금지 문장의 형식으로만 승계한다. 그 안의 KISTI 수치(48/217, ±0.6%·±0.3%, 131.01 s·286.85 s, D44 점유 범위)는 VESSL 사실로 승계하지 않는다. λ0R-3(48 구속)·λ0R-8(비분할 (0,108) 지배)은 이 Job의 서버 로그 `#running-req`/`#queue-req`와 `mix_*.json`으로 재확인한 뒤에만 쓴다. 등록 문서 §3 "이 Job에서도 그대로 참이다"는 기전의 예상으로 읽는다. λ0R-8 KISTI 정밀값을 인용하려면 E2C-8′ 교체값(a_r4 E-time 13.64%, b_r3 97.63%)과 추정량 규약을 함께 쓴다.

> **LV-6 (인용 금지 — 게이트 3 범위)** — "VESSL 기준선 분산을 측정했다 / 게이트 3을 충족했다" 금지. 허용형은 "한 VESSL Job·한 노드에서 포화 top rung achieved의 n=4(boot·seed·시점 미분리, df=3)"뿐. 이 CV를 정책 비교 지표의 기준선 분산이나 노드 간 산포로 쓰지 않는다.

> **LV-7 (필수 병기 — 시간 교락 비대칭)** — 등록 문서 §5-3 "s3·s4는 1–1.5 h 뒤"는 부정확하다. rev5 순서(A 전부 → B 전부)상 A의 s3·s4는 약 40–56분 뒤, B는 약 7–23분 뒤(참조 비용 예보, 실제는 `cell_*.complete`로 확인). shape 간 CV 비교 금지.

> **LV-8 (인용 금지 — Job 2)** — 선택 Job 2를 돌리면 두 Job을 따로 보고한다. 하나만 §4-4를 만족하면 "다르다"를 쓰지 않는다. 이 등록 아래 제출된 Job은 `vessl_jobs.tsv` 원장의 전부를 보고한다. Job 2가 Job 1과 같은 GPU UUID면 노드 간 산포 측정이 아니다.

> **LV-9 (인용 금지 — resume 혼합)** — `RESUMED_CELLS.tsv`가 비어 있지 않은 결과로는 §4-4 "다르다"를 쓰지 않는다. 라벨 인용에 "Job 혼합(출처 Job·UUID)"을 병기한다.

> **LV-10 (필수 병기 — 실현 분할과 같은 arm 여부)** — λ\*\_VESSL과 λ\*\_KISTI 차이를 같은 arm의 기판 차이로 쓰려면 a_r4/b_r3의 D44 점유 4추정량을 나란히 적는다. 점유가 다르면 arm 차이와 기판 차이를 분리할 수 없다. λ\*\_VESSL(A)로 B4/true-dual을 정규화하는 설계는 λ0R-8(iii) 교락을 그대로 진다. 908623 판정서 §7 E2는 미실행이다.

> **LV-11 (필수 병기 — arm 차이)** — 이 Job의 arm은 908623과 env 플래그 `PDMUX_GREEN_READOUT=1` 하나가 다르다(코드 바이트 동일, boot당 1회). 관측자 효과 0을 주장하지 않는다. 배너 `max_total_num_tokens`·`max_mamba_cache_size`가 908623의 2,722,025 / 48과 다르면 비교표에 병기한다. 빠진 pip 3종(VC-4)·manifest 밖 파일(VC-7)은 측정된 null이 아니다.

> **LV-12 (필수 병기 — telemetry 손실)** — telemetry writer는 큐가 차면 이벤트를 버린다(`dropped_events`). 쓰기 대상이 CephFS(KISTI는 Lustre)이므로 `mix_*.json` 인용 시 그 셀의 최종 `dropped_events`를 병기한다. 0이 아니면 4추정량은 "표본 손실 하 기술값".

> **LV-13 (예산 승인 문언)** — 4.5 h(16,200 s, $6.66)는 job_entry 시계 기준 셀 구간 집행이며 과금 하드캡이 아니다. preflight·tail·반출에는 kill 층이 없다. VESSL 크레딧 차단은 Job 생성 차단이지 실행 중 종료가 아니다. 실제 경로(`timeout -k 30` WARM/BENCH, watchdog `sglang.*`)는 테스트 미경유·코드 독해로만 확인. 사용자 승인에는 단가 외에 예상 ≈2.5–2.6 GPU-h(≈$3.7–3.8), 셀 구간 캡 4.5 h, 꼬리 비한정, Job 2 선택 시 두 배를 명시한다.

> **LV-14 (불변 배너 재확인)** — 이 Job은 어떤 결과로도 게이트 #6을 닫지 않는다(throughput 포화 ≠ SLO 용량, λ5C-8/λ0R-9). 성능 판정 0건. HE0·layer-type 死·정책 순위·Claim D/E 등급·stake #1·게이트 #13/#16 불변. "λ\*\_VESSL을 측정했으니 VESSL 캠페인의 부하 라벨이 참이 됐다"는 금지.

## 5. 비차단 권고 (하네스층, engine-porter 회부)

1. 실제 경로 kill 구조 테스트: `run_cell`의 WARM·BENCH가 `timeout -k 30 "$CAP"`로 감싸였는지 정적 검사 — 래퍼 제거 변이본에서 실패해야 함(교훈 53).
2. tail·preflight 상한: `vesslctl job create`에 최대 실행시간 옵션이 있는지 확인(미확인). 없으면 job_entry target 호출 전체 `timeout` 검토(job_entry 변경이라 B5 재확인 범위).
3. KILLED 판정 조건: `budget_fired`만으로 정상 완료 셀을 버리는 수 초 경합 창 — `BRC ∈ {124,137,143}`로 좁히는 것 검토(판정 반전 없음, 우선순위 낮음).

## 6. 이를 확정할 실험

- "VESSL 기판의 λ\*가 KISTI와 다르다"(기판 귀속): 서로 다른 GPU UUID의 VESSL Job ≥2(각 n=4) + KISTI 측 n≥4 — **KISTI n≥4는 기판 이양으로 영구 불가** ⇒ 이 주장은 "n=2 KISTI Job 대비 관측 차이"로 영구 제한. 원인 귀속은 한 노브씩 대조 없이는 불가(#10).
- λ\*\_VESSL(A)의 D44 의존: 908623 판정서 §7 E2와 같음 — `PDMUX_STICKY_PARTITION=1` 한 노브 대조 arm을 같은 Job에(새 사전등록).
- 게이트 #6: 908623 판정서 §7 E1(정본 술어 goodput, 셀당 n≥4, cliff 회피 격자)을 λ\*\_VESSL 기준으로 새로 등록.

## 7. 자기 적용

게이트 #110: 직전 판정서 수치(13.64%·97.63%, digest, 25/25)를 이번에 직접 재산출, 작성자 "1–1.5 h"를 독립 계산으로 정정(LV-7). 게이트 #113: 처방 1 실현 가능, 처방 2는 플랫폼 옵션 미확인, §6 기판 귀속 실험은 수행 불가 명시. 반전 계산의 자유 표면은 원자료·쉬핑 함수·908623 실측 비율로 고정. 표면 18개 중 반전 0 ⇒ `NO-GO` 아님, caveat 14건 없이는 `GO` 아님.
