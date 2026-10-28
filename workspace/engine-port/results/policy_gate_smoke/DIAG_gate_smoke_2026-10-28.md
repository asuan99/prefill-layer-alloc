# gate+smoke Job 진단 — pdmux-policy-gate-smoke-71f5006-20261028064810 (2026-10-28)

성격: **정확성 gate·smoke 판별량 진단**. 측정 해석이 아니고, 이 문서의 어떤 수치도 성능 근거가 아니다.
작성: engine-porter(읽기·분석만, 엔진·하네스·PREREG 무수정). 판정 변경(봉인 유효/무효)은 이 문서가 하지 않는다 — §4 사용자 결정.

## 0. 회수 상태

- `fetch.sh`는 Job의 `results/` 하위를 저장소 `workspace/engine-port/results/`에 **병합**(cp -an)하고, `meta/ logs/ SHA256SUMS DONE`만
  `results/policy_gate_smoke/vessl/<job>/`에 둔다(`scripts/vessl/fetch.sh:31-44`). 그래서 vessl 디렉터리 안에서 `sha256sum -c`를 돌리면
  `./results/...` 항목이 전부 "No such file"로 뜬다 — 결손이 아니라 경로 기준점 차이다.
- `results/` 기준으로 재검증: 상세 산출물 **전부 존재·일치**. 불일치 4건은 전부 `__pycache__/*.pyc`(cp -an이 기존 로컬 pyc를 보존,
  fetch가 의도적으로 충돌 검사에서 제외). 즉 산출물 미회수 아님.
  - stage1: `results/engine_gate/<job>/` (GATE_RESULT.json, plan.json, off/on/phase_only/static_ns/on_slo 각 tel.jsonl·server.log 등)
  - stage2: `results/cap_char/policy_runs/<job>/` (stages.txt, w1_lengths.json, smoke/smoke.json, smoke/{s16,s64,dyn}/tel.jsonl·seg/)
- `DONE exit_code=1` = `ph_gate_smoke.sh`의 `max(rc1=0, rc2=1)`.

## 1. stage1 — 요약 줄은 모순이 아니다

`engine_gate_diag: rc=0 (0 PASS, 1 FAIL, 4 UNRESOLVED, 13 engine digest)`는 `scripts/vessl/engine_gate_diag.sh:19`의 **고정 범례**
(`echo "engine_gate_diag: rc=${rc} (0 PASS, 1 FAIL, 4 UNRESOLVED, 13 engine digest)"`)다. 괄호 안은 개수가 아니라 rc 코드표
(`engine_gate_diag.py:421` `{PASS:0, FAIL:1}.get(verdict, 4)`, `:368` digest 불일치 시 13). rc=0 = PASS. 출력 버그는 아니나 읽는 사람을
오도하는 형식이다(개선 제안: `rc=0 [codes: 0=PASS ...]` 식으로 — 하네스 변경이므로 이번엔 하지 않음).

`GATE_RESULT.json` 실제 판정(전부 PASS, verdict/verdict_for_D1/verdict_for_D3 = PASS):

| 검사 | 판정 | 사실 |
|---|---|---|
| D0 digest | PASS(암묵) | 불일치면 `GATE_RESULT.json`에 ABORT_ENGINE_DIGEST + rc 13(`:365-368`); checks까지 진행했으므로 일치 |
| D1 greedy identity | PASS | 순차 6요청, n_diff 0 |
| D2 levels / D3 records(ON·slo·static_ns) / D4 decode_probe / D5 clocks | PASS | — |
| D6 slo_switch | PASS | n_slo_switch 1 |
| D7 sync audit | PASS | diag_* 0, total_stream_synchronize 49016(래퍼 live) |
| D8 phase_only | PASS | — |
| D9 static non-sticky(D34, `pdmux_agnostic_manual.yml`) | PASS | n_co 115 전부 D34, decode-only 153 전부 108, prefill-only chunk 4 전부 108, n_spans 11 |

D9는 `decode_iteration`(co-resident = `prefill_in_flight and not prefill_kernel_done_wait`) 기준이다(`engine_gate_diag.py:260-287`).
범위 주의: D9는 smoke와 **다른 config**(agnostic_manual, D34)다.

## 2. stage2 — `SMOKE_VOID_SAME_SPLIT`은 판별량 결함에서 나왔다

### 2.1 smoke 판별량
`ph_smoke.py:27-35` `realized_pairs`는 `prefill_span_start` 이벤트의 (prefill_sms, decode_sms)를 세고, `:54`가 arm별 decode_sms
**최빈값**이 같으면 VOID를 낸다.

### 2.2 이 이벤트는 전환 **전** 분할을 기록한다 (정적 근거, `src/multiplex/multiplexing_mixin.py`)
event_loop_pdmux 한 반복의 순서:
1. `:1292` `update_split_prefill_batch` → `:1235` `_dual_worker_start_prefill` → `:870` `_phase_on_prefill_start`
   → `:2186-2187` `idx = get_current_stream_idx(); prefill_sms, decode_sms = self.sm_counts[idx]` — **이 시점의 현재 idx**를 기록.
   이 함수는 배치만 만들고 커널을 내지 않는다.
2. `:1313-1355` R2 컨트롤러 `_r2_decide_idx`(controller_decision/split_transition emit) → 목표≠현재면 drain 후 `set_current_stream_idx(_tgt)`.
3. `:1405` decode launch, 그 뒤 prefill forward — **전환 후** 분할에서 실행.

비sticky에서는 prefill이 없을 때 decode-only 그룹(`adjust_stream_groups` `:1199-1200`, `real_sm_group_num-1` = (0,108))에
있으므로, decode가 도는 중 새 prefill이 들어오면 span_start는 (0,108)을 찍고 같은 반복에서 D로 전환된다.
decode가 비어 있으면 idx 0 = (108,0)(`:1211`). span 종료 직후 다음 span이 곧바로 들어온 경우에만 목표 D가 찍힌다.

### 2.3 회수 자료로 확인 (`results/cap_char/policy_runs/<job>/smoke/*/tel.jsonl`)

| arm | span_start (p,d) | split_transition (cur→tgt) | co-resident decode_iteration decode_sms | decode-only |
|---|---|---|---|---|
| s16 | (0,108)×88, (92,16)×18, (108,0)×5 | 108→16 ×88, 0→16 ×1 | **16: 113/113** (+kernel_done_wait 33, 전부 16) | 108: 3538/3538 |
| s64 | (0,108)×76, (44,64)×24, (108,0)×6 | 108→64 ×76, 0→64 ×1 | **64: 107/107** (+kdw 304, 전부 64) | 108: 3398/3398 |
| dyn | (0,108)×88, (74,34)×17, (108,0)×6 | (이벤트 없음, 아래) | **34: 112/112** (+kdw 148, 전부 34) | 108 |

span_start 뒤 첫 decode_iteration은 s16 107/109가 (92,16, in_flight), s64 101/104가 (44,64) — 나머지는 decode 없는 span.
(0,108) 88건은 전부 running_bs>0인 span_start이고 그 직후 split_transition 108→16이 같은 타임스탬프 근방에 찍힌다.
**s16·s64 모두 동거 decode는 100% 목표 D에서 돌았고 두 arm의 분할은 서로 다르다.** 최빈값 108은 "decode-only 상태에서 span이
시작된다"는 사실의 반영이며, 단일 prefill만 순차 진행되는 smoke 부하(he0_trace LO 1.2 / HI 4.8 req/s, 각 60 s; span이 1–2 decode
반복으로 짧아 decode-only 체류가 지배)에서는 arm과 무관하게 구조적으로 108이 최빈값이 된다. 최빈값을 고쳐도 `frac_at_target`
(s16 0.16, s64 0.23)이 0.95 미만이라 같은 판별량으로는 `SMOKE_FAIL_NOT_APPLIED`가 났을 것이다 — 이 역시 같은 결함이다.

dyn 부기: dyn arm은 R2 정책이 아니라 `PDMUX_SLO_SCHED` 경로라 controller_decision/split_transition을 내지 않고, PHASE_DIAG도
꺼져 drain 이벤트도 없다. 동거 decode는 전부 D34(anchor idx 3)였으나 **switch_count는 이 텔레메트리로 관측 불가**(0으로 읽으면 안 됨).

## 3. 결론: (b) smoke 판별량 결함

- (a) 아님: 비sticky fixed는 D를 적용한다 — smoke(D16/D64, 동거 decode 100%)와 stage1 D9(D34, 115/115) 양쪽 같은 노드에서.
- (c) 아님: 산출물 전부 회수·sha 일치.
- 따라서 `SMOKE_VOID_SAME_SPLIT`은 caveat 1이 겨냥한 현상("D 간 분할이 같음")의 관측이 아니라, 전환 전 분할을 기록하는
  이벤트를 판별량으로 쓴 하네스 결함의 산물이다. ★단 이것은 engine-porter의 진단이며, 봉인 caveat 1의 기계적 판정 규칙이
  `ph_smoke.py`로 고정돼 있었다면 "봉인 무효" 판정을 뒤집는 것은 **사용자 결정**이다(아래).

## 4. 최소 다음 조치 (사용자 결정 필요)

1. 판별량 교체: `ph_smoke.py`의 `realized_pairs`를 D9와 같은 정의 — `decode_iteration`, `prefill_in_flight and not
   prefill_kernel_done_wait` 의 decode_sms — 로 바꾸고, 판정 = 각 arm 동거 decode의 ≥95%가 목표 D AND 두 arm 최빈값 상이.
   이는 봉인 caveat 1의 **운용 정의를 바꾸는 것**이므로 사용자 결정 + claims-auditor 1회 확인 후 적용.
2. 재판정 경로 두 갈래(사용자 선택):
   (i) 이미 회수한 tel.jsonl을 새 판별량으로 CPU 재스코어(GPU 재실행 없음 — 위 표대로라면 PASS 조건 충족) — 단 "판정 규칙을
   결과를 본 뒤 바꿨다"는 사후성이 남으므로 그 사실을 봉인 기록에 공시;
   (ii) 판별량을 고친 뒤 smoke만 재실행(stage1은 PASS라 재실행 불요)해 사전 고정 규칙으로 판정.
3. CAP1 제출은 위 결정 전까지 보류 유지(현 봉인 71f5006a 문면상 무효 판정 상태).
4. (부수) `engine_gate_diag.sh:19` 범례 문구와 dyn arm switch 관측 부재는 별건 개선 항목.

## 근거 파일
- 회수 메타: `results/policy_gate_smoke/vessl/pdmux-policy-gate-smoke-71f5006-20261028064810/{logs/target.log,SHA256SUMS,DONE}`
- stage1: `results/engine_gate/pdmux-policy-gate-smoke-71f5006-20261028064810/GATE_RESULT.json`
- stage2: `results/cap_char/policy_runs/pdmux-policy-gate-smoke-71f5006-20261028064810/smoke/{smoke.json,s16,s64,dyn}/tel.jsonl`
- 코드: `scripts/vessl/engine_gate_diag.{sh:19,py:260-287,365-368,421}`, `results/cap_char/policy_harness/ph_smoke.py:27-61`,
  `results/cap_char/policy_harness/ph_gate_smoke.sh`, `src/multiplex/multiplexing_mixin.py:870,1199-1211,1235,1292,1313-1355,1405,2182-2204`
