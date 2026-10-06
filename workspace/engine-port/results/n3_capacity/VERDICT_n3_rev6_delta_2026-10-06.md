# 판정서: N3 사전등록 rev5 → rev6 델타 감사 (규칙층 + 하네스층 합본, claims-auditor, 2026-10-06)

> 읽기 전용, GPU 0, 성능 판정 0건, Claim 등급 변경 0건, 게이트 #6 불변. 저장소에서 바꾼 것은 이 판정서뿐이다. 변이 시험은 스크래치 사본(`engine-port` 심볼릭 링크 트리 + `results/n3_capacity`·`tests` 실복사)에서만 했다.
> 대상: `git diff 460fda2 -- results/n3_capacity tests/test_n3_capacity.py`(5파일, +264/−46). 기준 = rev5 봉인 `460fda2`(HEAD).
> 원자료: 2회차 Job `pdmux-n3-capacity-79999aa-20261006081638` 로컬 사본 `n3b_…/{bench,tel,warmup,marks}_floor_head.*`(git-ignored). 엔진 근거: `src/multiplex/multiplexing_mixin.py`, `src/multiplex/telemetry.py`(독해).

## 0. 판정: **`GO-with-caveats`**

- **死因 0건.** N1(항등식)·N3(도달 불가)·N4(제1원칙) 해당 없음. N2: 반전 시험 7개 자유 표면 중 등록 판정(floor 유효/무효 또는 floor 값)을 바꾼 것 0건(§7).
- **필수 수리 4건(R-1…R-4)**: fail-open 경로 2개(R-1 PHASE_EVENTS 중도 꺼짐, R-2 decode 기록 부재), 기술값 1건(R-3 floor 창 경쟁 지문), doc_facts 1건(R-4). 앞의 셋은 floor 블록 안에서만 동작하고 다른 판정 경로와 결합하지 않는다(§6 Q7).
- **새 caveat N3C-28…N3C-33**(§9).
- 핵심 판정: rev6 (iii)의 두 입력은 **둘이 함께일 때만** 실현 증거다. `prefill_span_end.stream_idx` 하나만으로는 실현 분할이 증명되지 않는다(§1). 그리고 rev5가 무효로 본 입장 라벨 경쟁은 그냥 기록의 경쟁이 아니라 **서버 쪽 admission 정지의 지문**이다. 그 정지는 그 요청의 TTFT를 실측으로 +106 ms 부풀렸다. rev6는 분할 판정에서는 옳지만, 정지에 오염된 floor 요청을 새로 받아들인다(§3, §8 Q8 → N3C-30, R-3).

## 실행 기록 (요청된 명령, 로컬 CPU)

| 명령 | 결과 |
|---|---|
| `python3 n3_plan.py --selftest` | rc 0, `N3 PLAN SELFTEST OK (… floor formula (rev6 realised evidence) + 6 invalidating controls + entry-label race admitted; P3 (ii) …)` |
| `python3 n3_plan.py --floor B n3b_…/bench_floor_head.jsonl n3b_…/tel_floor_head.jsonl <scratch>/audit_tk.json` | rc 0, `valid true`, floor_s **0.9067605545278639**, T_3/5/10 = 2.7203/4.5338/9.0676 s, n_spans 28, n_span_ends 28, n_decode_iterations 1581, `end_stream_idx_set [0]`, `entry_label_off_count 1`, `entry_running_bs_nonzero_count 1`, `dropped_events_max 0`, `dispersion_max_over_min 1.00999`. **진단 재계산이며 채점이 아니다.** |
| `python3 -m unittest tests.test_n3_capacity.TestFloorRev6RealJob tests.test_n3_capacity.TestBootBranchSnippets.test_t_h3_floor -v` | 3/3 OK (5.6 s) |
| `check_citation_stops.py` | 0 violation |
| `check_doc_facts.py` | **FAIL 1건**: `regression tests: document says 957, artefact says 959 (DESIGN_A1_REV2_STICKY_2026-08-25.md)` → R-4 |

## 1. Q1 — 두 입력이 정말 "실현" 상태인가 — **CONFIRMED(조건부: 두 입력의 논리곱으로만)**

`event_loop_pdmux`의 한 반복 안 순서는 다음과 같다(행 번호는 `multiplexing_mixin.py`).
1. `recv_requests`(1283)
2. `update_split_prefill_batch` → `_dual_worker_start_prefill`(1235) → **`_phase_on_prefill_start`(870, 2182-2204)**: 이 시점의 `get_current_stream_idx()`·`running_batch.batch_size()`를 적는다. 이것이 입장 라벨이다(조정 전).
3. `update_running_batch`가 끝난 요청을 걸러 낸다(1297). 이어서 `stream_idx>0 ∧ running 비어 있음`이면 조정을 트리거한다(1304-1306).
4. `_slo_on` 분기(1313-1357)가 있다. `PDMUX_R2_POLICY=fixed`라 `_slo_on`은 참이지만 이 분기는 `running` 비어 있지 않음을 요구한다. 그렇지 않으면 `elif adjust_stream_group`(1358-1371) → `adjust_stream_groups`(1151-1216)로 간다. running이 비면 1211에서 `set_current_stream_idx(0)`.
5. decode 발사(1379-1405, running이 걸러진 뒤 비어 있지 않으면 **매 반복 무조건**)
6. prefill chunk 발사(1410-1480, 현재 지역 `prefill_stream`)
7. decode sync → **`_phase_on_decode_sync`(1494, 2245-2270)**
8. prefill 완료 관측(`query` + allreduce, 1516-1527) → `_dual_worker_prefill_ready` → **`_phase_on_prefill_end`(883, 2206-2224)**: 이 시점의 `get_current_stream_idx()`를 적는다. 이후 merge하고 `adjust_stream_group=True`(1541).

- `prefill_span_end.stream_idx`는 커널 완료 관측 뒤, 그 span 동안 적용된 마지막 조정 **후** 값이다. 2회차 span 6(입장 stream 5·running_bs 1, end stream 0)의 기전과 일치한다.
- 시각: `RuntimeEvent.timestamp_monotonic_s`는 `time.perf_counter`이고 **생성 시점에 스케줄러 스레드에서** 찍힌다(`telemetry.py:23`, `emit` 102-109). 비동기 writer가 찍는 값이 아니다. `decode_iteration`은 1494의 `decode_stream.synchronize()` **뒤**에 생성되므로 그 시각은 sync 이후 값이다(필드 `t_sync`와 같은 시계). 2회차 자료에서 decode 시각 동률은 0건이다.
- **stale 경로 탐색(死因 후보)**: end 라벨이 실제 chunk의 stream과 다를 수 있는 길은 span 도중의 stream 전환이다. 그러면 end에는 마지막 chunk의 stream만 남는다. 전환 지점은 1160·1198·1200·1211·1351뿐이다(1745 이후는 `event_loop_pdmux_coord` 전용이고, PHASE_EVENTS와 함께 쓰면 거부된다). span 도중에 비-0에서 0으로 가는 전환은 "running 비어 있지 않음 → 비어 있음"을 요구한다. running이 걸러진 뒤에도 비어 있지 않은 반복에서는 5단계 decode가 무조건 발사되고 7단계에서 그 sync 시각으로 기록된다. 그 시각은 2단계 start보다 뒤이고 8단계 end보다 앞이므로 **닫힌 구간 안**에 들어간다. 따라서 "end 0인데 chunk가 비-0에서 돌았다"면 반드시 span 안 decode가 있다. **end 라벨 단독으로는 stale일 수 있고, (iii) 논리곱이 그것을 닫는다.** 독해로는 이 경로가 死因이 아니다. 단 사전등록 §4-1 "코드 근거" 문장은 두 입력을 각각 "실현된 상태"라고 적었다. 이것은 과대 서술이다 → N3C-29.
- 한계: 발사 SM 자체를 계측하지는 않았다(N3C-2 유지). stream 0과 마지막 stream은 둘 다 plain 그룹(green context 없음, `check_green` 495-496)이라서 "(108,0)"은 SM 제한이 실현됐다는 증거가 아니라 그룹 라벨이다.

## 2. Q2 — fail-open 경로

| 하위 | 판정 | 근거(실제 2회차 자료에 가공을 가해 rev6 `floor_from`에 적용) |
|---|---|---|
| (a) `prefill_span_end` 전무 | **fail-closed 확인** | end 전부 삭제 → `28 prefill spans without prefill_span_end` 무효 |
| (a′) R7 PHASE_EVENTS 중도 꺼짐 | **fail-open — REFUTED(“실현 증거 부재는 통과가 아니다” 주장)** | span 16 end 뒤 전부 절단 → **valid**, n_spans 16, n_dec 762. "마지막 16"이 span 1–16(사전 3·warm-up 8·bench 테스트 1·floor 4)이 되어 floor 요청 16개 중 **12개가 관측되지 않았는데 통과**한다. span 20에서 절단해도 valid. 러너는 R7을 floor 블록 **앞**(`n3_capacity.sh:472-475`)에서 감지하지만, 그 결과를 셀 기술값 무효화에만 쓰고 floor에는 쓰지 않는다. rev5에도 있던 결함이지만, rev6가 부재 논증을 판정 입력으로 삼으면서 하중을 받게 됐다 → **R-1** |
| (b) `decode_iteration` 0건 | **fail-open** | decode 전부 삭제 → **valid**(n_dec 0, problems 없음). 독해상 span은 기록되고 decode만 빠지는 엔진 경로는 없다(둘 다 같은 플래그, 예외 → 전체 disable → (a′)로 귀결, 큐 포화 → `dropped_events`). 그러나 부재 논증이 판정 입력이면 같은 boot에서 계측기가 살아 있었다는 실증이 필요하다(교훈 232). `n_decode_iterations` 기술값만으로는 판정에 들어가지 않는다 → **R-2**. D1 계열("근거 부재는 통과가 아니다")과 일관된다 |
| (c) stream 0 ↔ (108,0) | **CONFIRMED** | `check_green`(`n3_plan.py:465-497`)이 `stream_groups()[0] = FLOOR_PAIR`와 green read-out의 group 0 `(target_prefill_sm, target_decode_sm)`을 대조하고 plain 여부도 검사한다. 러너는 같은 boot에서 warm-up 전에 실행한다(`n3_capacity.sh:409`, 실패 시 `ABORT_GREEN`). 텔레메트리 파일은 unit마다 비운다(388). 따라서 boot 간 혼입은 없다 |
| (d) 닫힌 구간·bisect | **CONFIRMED(동치 변이 기록)** | `bisect_right(t1) − bisect_left(t0)` = 닫힌 구간. 반-열림 변이 M6·M6b는 생존했다(경계 시험 없음). 실자료에서는 decode 시각 동률 0, floor span 앞 간격 최소 15.8 ms, 뒤 간격 최소 14.1 ms라서 동치 변이다(교훈 254: 기록, 숨기지 않음) → 권고 W-2 |
| (e) 같은 `span_seq`의 end가 둘 이상 | **PLAUSIBLE(엔진상 불가, 규칙은 침묵)** | 실자료에 비-0 중복 end를 **뒤에** 넣으면 valid(첫 것 채택), **앞에** 넣으면 무효. 엔진은 seq를 start에서만 증가시키고 end는 batch당 한 번이라 중복 경로가 없다. 변이 M7(마지막 end 채택)은 생존(동치) → 권고 W-1 |

또 하나: `_read_jsonl`은 디코드 실패 줄을 조용히 건너뛴다. 그래도 큐는 FIFO 단일 writer이고 이벤트는 시간순으로 기록되므로, span 안 decode는 언제나 그 span의 end **앞** 줄에 있다. 꼬리 절단(SIGKILL)은 end를 남기고 그보다 앞의 decode만 지울 수 없다. teardown은 SIGTERM 뒤 5 s를 기다리고, 2회차 마지막 이벤트는 bench_end + 4.98 s였다. 이 경로는 fail-open이 아니다.

## 3. Q3 — 입장 `running_bs` 제외가 D4를 약화시키는가 — **분할 측면 CONFIRMED(약화 없음) / 지연 측면 REFUTED(rev6 근거 (d)가 불완전)**

- **분할 측면**: event_loop_pdmux에서 "running에 decode가 있는데 그 반복에 decode iteration이 없다"는 상태는 불가능하다. 1297 필터 뒤 running이 비어 있지 않으면 1379에서 무조건 발사되고, 1494에서 기록된다. retraction은 floor(동시성 1)에서 발생하지 않는다. "span 안 decode 0건"은 decode 유휴의 충분한 실현 증거다. 동시성 1 bench에서 "decode 정지 상태로 running에 남아 있음"은 이 루프 구조상 생기지 않는다.
- **지연 측면(새 발견)**: 입장 라벨 경쟁의 필요조건은 "마지막 decode 직후 반복이 클라이언트 왕복(정상 16–20 ms)보다 오래 시작되지 않음", 즉 **스케줄러 정지**다. 정지가 없으면 바로 다음 반복이 끝난 요청을 걸러 내고 라벨은 (108,0)/0이 된다. 2회차 실측:
  - span 6(경쟁) 앞 간격 124.8 ms. warm-up 8개 TTFT는 [1748.1, 907.3, **1011.6**, 908.0, 903.6, 902.4, 903.7, 906.7] ms이고, 3번째(= span 6)는 다른 정상 warm-up의 중앙값(≈905 ms)보다 **+106 ms(+11.7%)** 높다. 앞 간격 초과분(124.8 − ~18 ≈ 107 ms)과 일치한다.
  - 즉 rev5 (ii)/(iii)가 우연히 하던 일은 정지에 오염된 요청이 있는 boot를 버리는 것이었다. rev6는 그런 요청을 floor 표본으로 받아들인다. floor 창(span 13–28)의 앞 간격은 16.1–20.0 ms이고 경쟁 0건이라 2회차 floor 값은 영향받지 않았다.
  - 편향 방향: floor ↑ → T_k ↑ → SLO가 느슨해짐 → 용량 과대. 오염 요청이 8개 미만이면 중앙값은 순서통계량 한 칸씩만 움직인다(이 boot 분산 기준 ≲1 ms). 8개 이상이면 관측 정지 크기로 ≈ +11.7%. 정지 크기에는 상한이 없다.
  - 사전등록 §12-23(d)는 "running_bs가 같은 경쟁으로 1을 찍었다 → 대신 decode 0건이 decode 유휴를 말한다"고만 적었다. **경쟁 = 지연 오염의 지문**이라는 사실이 빠졌다 → **R-3**(기술값) + **N3C-30**.

## 4. Q4 — 테스트가 항등식이 아닌가 — **CONFIRMED(rev6 핵심 수리) / 부분 공백 기록**

스크래치 사본 변이 13개, 각각 selftest + `TestFloorRev6RealJob` + `test_t_h3_floor`:

| 변이 | 결과 | 잡은 것 |
|---|---|---|
| M1 **rev6 수리 되돌림**(end stream → start 라벨) | KILLED | selftest·RealJob·T-H3 셋 다 |
| M2 end stream 검사 제거 | KILLED | 셋 다 |
| M3 span 안 decode 검사 제거 | KILLED | selftest·T-H3 |
| M4 짝 없는 span 검사 제거 | KILLED | selftest |
| M5 dropped_events 검사 제거 | KILLED | selftest |
| M8 (iv)에 입장 running_bs 재삽입 | KILLED | selftest·T-H3 |
| M9 (iv) 제거 | KILLED | selftest |
| M11 FLOOR_MIN_SPANS 검사 제거 | KILLED | selftest |
| M12 stream 5 decode 무시 | KILLED | selftest(기술값 단언) |
| M6/M6b 반-열림 구간 | SURVIVED | 동치(실자료 동률 0) |
| M7 마지막 end 채택 | SURVIVED | 동치(엔진상 중복 없음) |
| M10 bad_ts 검사 제거 | SURVIVED | 대조 없음(엔진 시계로는 도달 불가) |

- 교훈 53이 요구하는 조건, 즉 "수리를 되돌린 변이본에서 반드시 실패"는 M1이 세 경로 모두에서 충족한다.
- `TestFloorRev6RealJob`의 "rev5 술어 발화"는 rev5 술어를 인라인으로 재평가한다. 동시에 **rev5 코드가 Job 안에서 만든** 봉인 `T_K.json`의 problems 문자열(`["1 prefill spans not at (108, 0)"]`)을 대조하므로 항등식이 아니다.
- 클린 클론(Job preflight)에서는 원시 jsonl이 git-ignored라 `TestFloorRev6RealJob`이 skip된다. 이때 rev6 보호는 selftest(대조 6+양성 2)와 T-H3(러너 조각 실행)가 맡는다. 둘 다 M1을 죽이므로 공백은 아니다. 다만 "실제 Job 자료에서 rev6 검증"이 preflight 기록에 남는다고 쓰면 거짓이다 → N3C-32.
- 공백: R-1·R-2 경로(PHASE_EVENTS 중도 꺼짐·decode 기록 부재)에는 대조가 없다. 수리와 함께 변이 대조를 넣어야 한다(§8).

## 5. Q5 — 규칙 외 변경 0 — **CONFIRMED**

- 변경 파일은 5개뿐이다(`PREREG`·`SUBMISSIONS`·`n3_capacity.sh`·`n3_plan.py`·`test_n3_capacity.py`). `n3_analyze.py`·`n3_realized_mix.py`·config·정본 `analyze.py`는 diff 0이다.
- `n3_plan.py`의 변경은 docstring, 새 리터럴 `FLOOR_STREAM_IDX = 0`, `floor_from`, selftest floor 블록, selftest 출력 문자열뿐이다. 격자·seed·예산·순서·P3 (ii)·R1–R7·CG2·O1–O4 상수에는 diff가 없다.
- `floor_from`/`FLOOR_STREAM_IDX`/`FLOOR_PAIR`는 `n3_plan.py`와 테스트 밖에서 참조되지 않는다(grep). 채점·R3·mix 경로와 결합이 없다.
- 러너: `ABORT_FLOOR` 메시지 1줄. exit 7, 위치, `TKOUT` 분기는 불변이다.
- 문서 측 부수 효과: `def test_` 63→65로 doc_facts가 재발했다(**11차**, 957→959) → R-4. 사전등록 §13의 "52/52" 표기도 실제 수(65)와 다르다. rev5 이전부터 이어진 낡은 표기다 → 권고 W-4.

## 6. Q6 / Q7

- **Q6(2회차 Job 처리) — CONFIRMED(문구 보강 필요)**: 머리말·§12-23(e)·SUBMISSIONS 追記는 모두 "진단 재계산, 재채점·재개 금지, 새 Job"이라고 적었다. 남는 해석 여지는 둘이다. (i) 0.90676 s를 새 Job floor의 "기대값·검증 기준"으로 쓰는 것. 이는 새 자유 표면이 된다. (ii) "rev6가 2회차 floor를 유효화했다"를 그 Job의 T_K가 성립한다는 뜻으로 읽는 것. 규칙은 그 자료를 본 **뒤에** 쓰였다. 다만 rev6는 유효성 분기만 바꾸고, 유효할 때의 floor 값(클라이언트 TTFT 중앙값)은 바꾸지 않는다. 그래서 사후 규칙 선택이 값 쪽으로 새는 경로는 §3의 오염 요청 수용뿐이다 → N3C-28 금지 문안.
- **Q7(처방의 노브 결합, 교훈 250)**: R-1은 floor 블록 안, `TKOUT = T_K.json`일 때만 적용한다. 셀의 R7 처리(기술값 무효화)와 tail/resume 진단은 그대로다. R-2·R-3은 `floor_from` 안에서만 동작한다. `floor_from`은 채점·R3 arm identity·mix 어디에서도 호출되지 않는다(§5 grep). R-2가 거짓 abort를 낼 위험도 따져 봤다. 등록 output_len은 B 64·A 512이고, 2회차 bench `output_lens` 16×64(ignore-eos)였으며, floor 각 span 뒤 첫 decode는 +14.1–14.7 ms다. 그래서 정상 boot에서는 발화하지 않는다. 처방 실현가능성: 세 수리 모두 기존 자료로 판정 불변을 확인했다(§7).

## 7. 반전 시험 표

원자료 = 2회차 `tel/bench_floor_head`(28 span, 1581 decode). 추정량 = rev6 `floor_from` 그대로. 판정량 = floor 유효성과 floor_s. arm 자료가 없으므로 라벨 반전은 시험할 수 없고, 아래 결과는 floor 단계에 한정된다.

| 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 |
|---|---|---|---|
| floor 요청의 입장 `running_bs`/라벨 포함 여부(rev5↔rev6) | 포함 / 제외 | floor 창: 없음(경쟁 0/16). boot 전체: rev5 무효 ↔ rev6 유효 = **사용자 승인된 의도 변경**. floor_s는 유효할 때 동일 | floor 창 앞 간격 16.1–20.0 ms. 오염 시 예측 +106 ms/요청(warm-up #3 1011.6 vs ≈905) → N3C-30 |
| `dropped_events` 허용 | 0 / >0 허용 | 없음 | max 0 |
| 구간 경계 | 닫힘 / 반-열림(양끝) | 없음 | 동률 0, 경계 여유 ≥14.1 ms |
| 중복 end 선택 | 첫 것 / 마지막 것 | 없음 | 28 start, 28 end, seq 유일 |
| `FLOOR_STREAM_IDX` 대응 | 0 / read-out 대조 | 없음 | `end_stream_idx_set [0]`, check_green PASS |
| span 하한 | 16 / 24(warm-up+floor) / 28 | 실자료 없음. **절단 자료에서 fail-open**(n=16에서 valid) | §2 (a′) → R-1 |
| decode 계측기 생존 요구 | 없음 / floor span마다 뒤에 ≥1 | 실자료 없음(16/16 충족). **decode 삭제 자료에서 fail-open** | §2 (b) → R-2 |

반전 계산 자체가 새 자유 표면이 되지 않도록, 절단 지점(span 16/20)과 삭제 가공은 "fail-open이 존재한다"는 존재 증명에만 쓴다. 라벨이나 값을 이 가공으로 만들지 않았다.

## 8. 필수 수리 (제출 전)

- **R-1 (fail-open, 러너)**: floor 블록 안에서 `TKOUT = T_K.json`이고 `$OUT/phase_events_disabled_${NAME}`이 존재하면, `floor_from` 결과와 무관하게 `ABORT_FLOOR`(exit 7)로 끝낸다. 메시지에는 "PHASE_EVENTS disabled mid-boot — floor spans unobserved"를 포함한다. R7 감지(472-475)는 이미 floor 블록 앞에 있으므로 순서 변경이 없다. 테스트: T-H3 조각 실행에 플래그 파일을 둔 대조(→ exit 7, `unit_X.complete` 없음)와 그 줄을 중화한 변이(→ 통과, 즉 검사가 공허하지 않음). 사전등록 §4-1 (ii)에 1문장을 추가한다.
- **R-2 (fail-open, `floor_from`)**: 마지막 `FLOOR_N` span 각각에 대해, 그 span의 end 시각 초과부터 다음 `prefill_span_start` 시각 이하(마지막 span이면 파일 끝)까지 사이에 `decode_iteration`이 ≥1건 있어야 한다. 없으면 무효로 하고 사유는 "decode recorder not shown live after floor span"으로 적는다. 근거: 부재 논증의 계측기 생존 실증(교훈 232). 등록 output_len은 64/512다. selftest 대조: 마지막 span 뒤 decode 삭제 → 무효, decode 전부 삭제 → 무효. 실자료 회귀: valid 유지. 사전등록 §4-1 (ii)와 §12-23(a)에 반영한다.
- **R-3 (기술값, `floor_from`)**: `T_K.json`에 `floor_entry_race_count`(마지막 FLOOR_N span 중 입장 라벨 ≠ (108,0) 또는 입장 running_bs ≠ 0인 수)와 `floor_admission_gap_ms`(각 floor span start와 직전 `decode_iteration` 시각 사이 간격, 16개 목록)를 기록한다. **판정 입력이 아니다.** §12-23(d)/(g)에 "입장 라벨 경쟁은 admission 정지의 지문이며 그 요청 TTFT를 부풀린다(2회차 실측 +106 ms)"를 적는다. N3C-30 승계 근거다.
- **R-4 (규율)**: `check_doc_facts.py` FAIL(957 → 959, 11차 재발)을 정정한다. 진리원은 불변이다.

권고(비차단):
- W-1: `span_seq` 유일성(start·end 각각) 위반을 무효로 한다(엔진상 도달 불가, M7 동치).
- W-2: 닫힌 구간 경계 대조(decode 시각 = start, = end → 무효)를 둔다. M6/M6b를 죽인다.
- W-3: `bad_ts` 대조(end < start)를 둔다. M10을 죽인다.
- W-4: §13 "52/52"를 실제 수로 갱신한다.
- W-5(선택, 입력 추가라 사용자 결정): 마지막 16 span start ∈ [bench_start, bench_end] 대조. 같은 CLOCK_MONOTONIC이며 2회차에서 span 12–28이 창 안에 있다. span 12는 bench_serving 테스트 요청이다. R-1이 있으면 불필요하다.

## 9. 새 caveat (결과 문서·정본이 문자 그대로 승계, 인용 금지 문장)

- **N3C-28**: "rev6로 2회차 Job(`pdmux-n3-capacity-79999aa-20261006081638`)의 floor가 유효해졌다", "N3 floor는 0.907 s다", "새 Job floor가 0.907 s와 일치하므로 타당하다"는 **인용 금지**. 0.9067605545278639 s는 그 자료를 본 뒤 쓴 규칙으로 계산한 진단 재계산이다. 그 Job은 arm 셀 0·라벨 0이고 재채점·재개 대상이 아니며, 새 Job floor의 기대값이나 검증 기준으로 쓰지 않는다.
- **N3C-29**: "`prefill_span_end.stream_idx == 0`이므로 그 prefill은 (108,0)에서 실현됐다"는 **인용 금지**. end 라벨은 span 도중 전환이 있으면 마지막 chunk의 stream만 말한다. 실현 판정은 "span 안 decode_iteration 0건"과의 논리곱으로만 성립한다(event_loop_pdmux 독해, 등록 엔진 manifest 한정). 또한 stream 0은 green context가 없는 plain 그룹이라, "(108,0)"은 SM 제한이 실현됐다는 증거가 아니라 그룹 라벨이다(N3C-2).
- **N3C-30**: "rev6는 floor 값을 바꾸지 않는 순수 라벨 수리다"는 **인용 금지**. 입장 라벨 경쟁은 스케줄러 admission 정지의 지문이다(2회차 warm-up #3: 앞 간격 124.8 ms, TTFT 1011.6 ms vs 정상 ≈905 ms, +106 ms). rev5에서는 그런 요청이 있는 boot가 무효였지만 rev6에서는 floor 표본에 들어간다. 편향 방향은 floor ↑ → T_k 완화 → 용량 과대이고, 크기는 `floor_entry_race_count`·`floor_admission_gap_ms`(R-3)로만 공시된다. 정지 원인(cgroup 쓰로틀 후보)은 미확정이며 rev6는 그것을 고치지 않는다.
- **N3C-31**: "2회차 Job에서 28/28 span이 rev6 기준을 만족했으므로 decode 유휴 admission은 (108,0)이다", "경쟁 발생률은 1/28이다"는 **인용 금지**. n=1 boot의 사전 근거이고 발생률 추정이 아니다.
- **N3C-32**: "rev6 floor 규칙은 실제 Job 자료로 preflight에서 검증된다"는 **인용 금지**. `TestFloorRev6RealJob`은 git-ignored 원자료에 의존해 클린 클론(Job preflight)에서 skip된다. preflight의 rev6 보호는 합성 selftest와 러너 조각 시험(T-H3)뿐이다(둘 다 수리 되돌림 변이 M1을 죽인다).
- **N3C-33**: "rev6 floor 판정의 모든 분기가 변이 시험으로 검증됐다"는 **인용 금지**. 반-열림 구간(M6/M6b)·마지막 end 채택(M7)은 동치 변이로 생존했고, `bad_ts` 분기(M10)는 대조가 없다(W-1…W-3 미반영 시).

## 10. 게이트 #113 · #110

- #113(처방 실현가능성): R-1…R-3은 2회차 실자료에서 판정을 바꾸지 않는다(valid 유지, R-2 조건 16/16 충족, R-3은 기술값). 정상 boot에서 거짓 abort가 날 경로는 output_len ≥ 2에서 없다.
- #110: 직전 rev5 판정서의 N3C-23…27과 진단 문서의 "기전" 서술은 등록 상수로 승격하지 않았다. 원인(입장 라벨 = 조정 전)은 소스 독해(§1)와 원자료(span 6 앞 간격·warm-up TTFT)로 독립 재검증했다.

## 메인 세션 반영 필요 항목

- 필수: R-1(러너 floor 블록 R7 → ABORT_FLOOR + T-H3 대조·중화 변이), R-2(floor span 뒤 decode 생존 조건 + selftest 대조 2 + 실자료 회귀), R-3(`floor_entry_race_count`·`floor_admission_gap_ms` 기술값 + §12-23 문장), R-4(doc_facts 957→959). 반영 뒤 rev6 봉인 커밋과 재봉인 표를 이 판정서 끝에 追記한다(메인 세션).
- 권고: W-1…W-5.
- caveat: N3C-28…N3C-33을 사전등록 §12/결과 문서에 문자 그대로 승계한다.

## 메인 세션 追記 (2026-10-06) — R-1…R-4·W-1…W-4 반영 후 재봉인 표

R-1(러너 floor 블록 R7 표식 → ABORT_FLOOR + T-H3 대조·중화 변이·재개 경로), R-2(floor span 뒤 decode_iteration ≥ 1 + selftest 대조 2 + RealJob 회귀 +1), R-3(`floor_entry_race_count`·`floor_admission_gap_ms` 기술값 + §12-23(d)(g)), R-4(doc_facts 957→960, 11차), W-1(span_seq 유일성 무효), W-2·W-3(경계·역순 시각 대조), W-4(§13 테스트 수 66) 반영. W-5 미반영(사용자 결정). N3C-28…33은 사전등록 §11에 문자 그대로 승계. 2회차 실자료 재확인: `valid`, floor_s 0.9067605545278639, `floor_entry_race_count` 0, R-2 16/16(진단). 반영 후 작업 트리 값:

| 파일 | sha256 |
|---|---|
| `results/n3_capacity/n3_plan.py` | `2e91ee17e97cd26e13c15fc86b5a3d102eb304ff30154dfe082ad7399b4e15a3` |
| `results/n3_capacity/n3_capacity.sh` | `cccccb9edc33762e08c64768eafd43bb0f371a383bdd357933a464840d84ac33` |
| `results/n3_capacity/n3_analyze.py` | `eb89c0a2c48faba77615c5b3d7de3cccd8b43be58cf6e1faf17c4a1be9d5b974` |
| `results/n3_capacity/n3_realized_mix.py` | `3b6f8934c69ad09b23fd4490a882dafbdecef05e7980f51b2679fe062d2e65c3` |
| `results/n3_capacity/configs/pdmux_n3_A_agn.yml` | `01675b2d7fa8ff23bb1cb142ff66fa4dfd0c021722ae9b8e7acedc2f4d38fca1` |
| `benchmarks/pdmux_eval/analyze.py` | `1be495110ea67c95f275b6c8ab7e7af25d0778c2910bf6a830170dbc078aeaa2` |
| `tests/test_n3_capacity.py` | `7024dbc83bcff6b580c5720e9de55f1c2b95bd2188e2584b78ab259cd9ebecf7` |
| `results/n3_capacity/PREREG_N3_CAPACITY_2026-10-04.md` | `7024ea7abd8f71e8ca0ae672bee781730007f6ea957c87f233adf93bdff8b044` |

불변 확인: `n3_analyze.py`·`n3_realized_mix.py`·config·정본 `analyze.py`는 rev5 재봉인 표와 동일 sha(위 4줄).

사용자 승인(§6 게이트 5): 2026-10-06 사용자 결정 "a" — rev6 권고안 승인 및 재제출 진행. 제출 기록은 `SUBMISSIONS_2026-10-06.md`.

(사전등록 sha는 §13 끝 줄[66/66] 追記 후 값으로 갱신, 2026-10-06.)
