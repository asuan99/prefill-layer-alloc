# 감사 r1 — smoke v2 판별 기준(정책 PREREG §16, rev1.1, 24e49964) (2026-10-28)

- 감사자: claims-auditor(읽기 전용) · 1회차 · **판정: 재채점 진행 GO-with-caveats**(死因 0) · CPU 70/70 OK
- 독립 재집계(회수 tel.jsonl): 동거 반복 s16 {16:113}, s64 {64:107}, dyn {34:112}; kernel-done-wait 반복 s16 {16:33}, s64 {64:304}, dyn {34:148}.

## (1) 항등식 아님
`multiplexing_mixin.py:2252-2262` `decode_sms` = `sm_counts[stream_idx]`, `stream_idx`는 `:1405` decode를 실제로 띄운 stream index이며 `:1352-1356` 전환 뒤 갱신 — 목표 D 복사 필드 아님(전환 실패 시 108이 찍혀 VOID/NOT_APPLIED 가능). 필터 `prefill_in_flight`(`:2240`)도 전환 뒤 평가. 한계: config 라벨이지 하드웨어 SM 실측 아님(v1도 같음).

## 반전 시험
N_CO_MIN 1–107 · APPLY_FRAC 0.5–1.0 · kdw 포함/제외(146/146, 411/411) · 최빈값 검사 유무 — 모두 판정 변화 없음. 사후 변경 허용 근거: v2 관측량은 smoke 전 D9 정의(`engine_gate_diag.py:260-287`), 기전 코드 추적, 정확성 판정.

## 문제
1. MINOR `ph_smoke.py:5-8` docstring이 v1 기준 설명
2. MINOR `ph_smoke.py:95-96` ④ 아래 최빈값 검사 잉여(M60은 라벨로만 잡힘) — 공시
3. `PREREG:26` Q-3은 dyn arm switch_count 기록을 요구하나 §16-2(`:309`)가 제외 → v2 PASS로 Q-3 dyn 부분은 닫히지 않음
4. `PREREG:317-319` 의미 결함: stage1 `on_slo`에 slo_switch drain 1건(stream 3→1, 108→34)인데 같은 boot `SLO-` 로그 0건. 로그(`SLO-BIND`/`SLO-SCHED`, `:1082`·`:1143`)는 컨트롤러 결정 변경만 셈; 비sticky에서 decode-only→anchor 물리 전환은 span마다(static도 ≈88회). 게이트 5 "static은 0"과 같은 이름이 arm마다 다른 양 → TUNE 동률 규칙(`:92`) 사용 전 정의 수정 필요.

## 재채점 결과 문서에 붙일 문장
"v2 관측량은 smoke 전부터 있던 gate D9 정의이지만, 기준 채택 시점에 회수 원자료(s16 113/113, s64 107/107)는 이미 알려져 있었다. 따라서 이 PASS는 사전 고정 판정이 아니다. N_CO_MIN(1–107)·APPLY_FRAC(0.5–1.0)·kernel-done-wait 포함 여부는 판정을 바꾸지 않았다. decode_sms는 decode를 띄운 stream의 config 라벨이며 하드웨어 SM 실측이 아니다. 이 smoke는 dyn arm의 전환 기록(Q-3 후반)을 확인하지 않았다."
caveat 1 개정 문구 승인, 끝에 추가: "판정은 stream 라벨 기준이며 SM 실현량 측정이 아니다."

## §16-4와 CAP1
CAP1은 막지 않음(정적 5 D, 봉인 조건 caveat 1뿐). **Job-TUNE·HE0 봉인은 막음** — switch_count 정의와 양성 대조 방법 결정 전까지 게이트 5 dyn 항목 `UNOBSERVABLE_UNCONFIRMED`.
