# 재채점 — smoke v2 판별 기준을 회수 자료에 적용 (2026-10-28)

성격: **정확성 smoke 재채점**(성능 판정 아님). 대상 Job `pdmux-policy-gate-smoke-71f5006-20261028064810`(커밋 `71f5006a`), 기준 = 정책 PREREG §16(rev1.1) `ph_smoke.smoke_verdict_v2`, 감사 `AUDIT_smoke_v2_r1_2026-10-28.md`(GO-with-caveats). GPU 재실행 0, 회수 원자료(`results/cap_char/policy_runs/<job>/smoke/{s16,s64,dyn}/tel.jsonl`)에 CPU 재채점만 했다.

## 결과: `SMOKE_PASS` (v2)

| arm | 동거 반복 수 (prefill_in_flight ∧ ¬kernel_done_wait) | 동거 decode_sms 분포 | 목표 D 비율 |
|---|---|---|---|
| s16 (목표 16) | 113 | {16: 113} | 1.00 |
| s64 (목표 64) | 107 | {64: 107} | 1.00 |
| dyn (anchor idx 3) | 112 | {34: 112} | — (요구 없음) |

판정 순서: boot 3/3 성공 → 정적 arm 동거 반복 ≥ 30(113, 107) → 최빈값 상이(16 ≠ 64) → 목표 D 비율 ≥ 0.95(1.00, 1.00) → dyn 동거 반복 ≥ 30(112) → PASS. 독립 재집계(감사)와 일치. 구 v1(`prefill_span_start` 최빈값)의 `SMOKE_VOID_SAME_SPLIT`은 판별량 결함이었다(전환 전 분할을 기록) — 이력으로 남긴다.

## 공시 (문자 그대로)

"이 기준은 smoke 결과(VOID)를 본 뒤 판별량 결함 진단(DIAG_gate_smoke)에 따라 바꾼 것이며, 성능 판정이 아닌 정확성 smoke다. 새 기준은 회수된 원자료에 재채점으로만 적용했다."

## 감사 추가 문장 (문자 그대로)

"v2 관측량은 smoke 전부터 있던 gate D9 정의이지만, 기준 채택 시점에 회수 원자료(s16 113/113, s64 107/107)는 이미 알려져 있었다. 따라서 이 PASS는 사전 고정 판정이 아니다. N_CO_MIN(1–107)·APPLY_FRAC(0.5–1.0)·kernel-done-wait 포함 여부는 판정을 바꾸지 않았다. decode_sms는 decode를 띄운 stream의 config 라벨이며 하드웨어 SM 실측이 아니다. 이 smoke는 dyn arm의 전환 기록(Q-3 후반)을 확인하지 않았다."

## 범위
caveat 1(개정, 끝 문장 포함)에 대한 답이다: 비sticky `R2_POLICY=fixed`는 동거 중 D별로 다른 stream 라벨을 적용했다(16 대 64). dyn arm의 전환 기록은 확인되지 않았고 양성 대조 smoke(PREREG §17)로 넘긴다. 최빈값 상이 검사는 목표 D 비율 ≥ 0.95와 D16 ≠ D64가 함께 성립하면 잉여다(판정 순서상 먼저 라벨 `SMOKE_VOID_SAME_SPLIT`을 붙이는 역할만 한다).
