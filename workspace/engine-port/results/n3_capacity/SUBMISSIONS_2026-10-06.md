# N3(게이트 #6) 제출 기록 (메인 세션)

| shape | job name | slug | 제출(UTC) | 커밋 | 이미지 | spec | 승인 |
|---|---|---|---|---|---|---|---|
| B | `pdmux-n3-capacity-a4b0c2e-20261006064114` | `job-sxdnvi86hxsy` | 2026-10-06T06:41Z | `a4b0c2e` | `sglang-runtime@1e0d335b…` | `resourcespec-a100x1` | 사용자 2026-10-05 "N3 dry-run 진행" + 2026-10-06 "실험 job들에 대해서 다시 진행"(E2 r2 종료·회수 뒤 제출, 동시 제출 없음) |

- 제출 시 상태: `git status --porcelain --untracked-files=no` 비어 있음, presubmit 차단 = M4R·TC1 2건(등록된 범위), 사전등록 rev4(`23c4175`)·하네스층 H-1 수리(`1a90b08`) 커밋됨, 봉인 sha = `VERDICT_n3_harness_2026-10-05.md` 追記 표.
- env: `PDMUX_N3_SHAPE=B`만 추가(허용 목록). 예산 문언(§8-3): 보통 3.37 GPU-h / 최악 3.83 / 하드캡 4.0(셀 구간).
- shape A는 B 결과 감사 뒤 별도 승인.

## shape B 1회차 실패 진단 (2026-10-06, 메인 세션, GPU 소비 ≈1 min — 셀 0 측정)

- 상태 `failed`, 컨테이너 06:41:37Z→06:42:34Z(57 s), `exit_code=4`, 마지막 줄 `ABORT_SELFTEST n3_plan`. 회수 메타 `vessl/pdmux-n3-capacity-a4b0c2e-20261006064114/`(26 files, sha 통과), OUT 디렉터리 `n3b_…/`(preflight 산출물만).
- **통과한 preflight**: CONFIG OK · SUBSTRATE OK(A100 80GB cc8.0 108 SM) · 모델 revision · ShareGPT sha · clock · `install_runtime` + manual_edits patch · `--verify-digests "$ENG"` → `DIGESTS OK (3 files + 9 mirror entries)` · `ENGINE_TREE == registered HEAD manifest (0 entries differ)` · thread-local role patch.
- **실패 지점**: `n3_plan.py --selftest` → `selftest()` 기본 인자 `eng=ENG` → `verify_digests(eng)` →
  `benchmarks/pdmux_eval/analyze.py: MISSING`.
- **원인(로컬 재현으로 확정)**: `n3_plan.py:36–37`이 `ENG = Path(__file__).resolve().parents[1]`로 엔진 루트를 **자기 파일의 실경로**에서 유도한다. `job_entry.sh:100–103`은 Job 안에서 `workspace/engine-port/results/`를 `/data/runs/<job>/results`로 **옮기고 심볼릭 링크**로 바꾸므로 `resolve()`가 `/data/runs/<job>/results/n3_capacity/n3_plan.py`로 풀리고 ENG = `/data/runs/<job>`가 된다. 그 루트 밑에는 `results/…`(config·`lambda0_plan.py`)만 있고 `benchmarks/`·`src/`는 없다 → 등록 3파일 중 analyze.py만 MISSING(그래서 정확히 1건). 같은 helper의 `--verify-digests "$ENG"`는 러너가 루트를 **인자로** 넘겨서 통과했다(두 경로의 차이가 증거). 로컬에서는 `results/`가 실제 디렉터리라 selftest가 통과했다. ~~Job 레이아웃(results 심볼릭 링크)은 사전등록·하네스 감사 어디에도 전제로 적혀 있지 않았다~~ **(정정, rev5 델타 감사 R-2: 거짓)** — 하네스 판정서 `VERDICT_n3_harness_2026-10-05.md` §7은 이 레이아웃을 적었으나 OUT 경로에만 적용했고 helper의 루트 유도에는 적용하지 않았다. 같은 함정은 저장소 안에 이미 진단·수리된 선례가 있다(`results/e1_cap_vessl/e1_plan.py:45-47`, 2026-10-04, GPU 없는 컨테이너 dry run에서 발견) — N3 하네스가 그 선례를 승계하지 않았다(교훈 18 계열).
- **동일 결함 파급**(로컬 심볼릭 링크 모사 레이아웃에서 재현): `n3_analyze.py:37–40`도 같은 유도(`ENG/"benchmarks"/"pdmux_eval"`를 sys.path에 추가) → `ModuleNotFoundError: No module named 'analyze'` — selftest를 넘겼더라도 **채점 단계에서 실패**했을 결함. `n3_realized_mix.py:66–69`는 `results/` 밑 경로만 쓰므로 Job 레이아웃에서도 성립(단 λ0 archive는 로컬 전용).
- **E2/λ0가 같은 함정을 피한 이유**: `e2_vessl_plan_r2.py`는 HERE를 자기 디렉터리 sys.path에만 쓰고 엔진 루트는 러너가 `--verify-digests-r2 HERE_DIR ENGINE_ROOT` 인자로 넘김; `lambda0_vessl_plan.py`는 `results/` 밑 상대경로만 유도.
- **수리 방향(봉인 파일 변경 ⇒ rev5 + 규칙층 델타 감사 + 하네스 재감사 필요, 승인 전 미수정)**: n3_plan.py·n3_analyze.py의 엔진 루트를 `PDMUX_ENGINE_ROOT`(러너가 `$ENG`로 export) 또는 `Path(__file__).parent`(비해석, `resolve()` 제거)로 유도하고, 테스트에 "results가 심볼릭 링크인 레이아웃에서 selftest/analyze import가 성립" 케이스(수리 되돌림 변이에서 실패해야 함, 교훈 53)를 추가. 러너 `n3_capacity.sh`는 `ENG`를 `$ROOT`에서 유도하므로 무변경 가능.
- 등록 라벨·셀 측정 0건 ⇒ D23(부분 집합 재라벨 금지) 해당 없음. 재제출은 새 job name(커밋 sha 포함)로.

## shape B 2회차 제출 (rev5, 2026-10-06)

| shape | job name | slug | 제출(UTC) | 커밋 | 이미지 | spec | 승인 |
|---|---|---|---|---|---|---|---|
| B | `pdmux-n3-capacity-79999aa-20261006081638` | `job-er6a72360ha1` | 2026-10-06T08:16Z | `79999aa`(rev5 `5c58dbc` + doc_facts 정정) | `sglang-runtime@1e0d335b…` | `resourcespec-a100x1` | 사용자 2026-10-06 "수리 진행 후 실행" |

- 선행: rev5 델타 감사 `VERDICT_n3_rev5_delta_2026-10-06.md` = `GO-with-caveats`(R-1…R-3 반영, 재봉인 표 追記), `test_n3_capacity` 63/63, citation 0위반, presubmit 차단 = M4R·TC1 2건(등록된 범위, doc_facts 0), tracked tree clean. `RESUME_FROM` 미설정(1회차는 재개 불가, N3C-27).

## shape B 2회차 실패 진단 (2026-10-06, 메인 세션) — rev5 경로 수리는 성립, 새 지점에서 정지

- 상태 `failed`, 컨테이너 08:17:37Z→08:25:58Z, `exit_code=7`, 마지막 줄 `ABORT_FLOOR floor invalid … (P-2)`. 회수 `vessl/pdmux-n3-capacity-79999aa-20261006081638/`, OUT `n3b_pdmux-n3-capacity-79999aa-20261006081638/`. **arm 셀 측정 0, 라벨 0.**
- **통과**: 등록 digest·config·substrate·엔진 트리·selftest 3종(plan/mix 합성/analyze)·preflight unittest — 1회차 결함(rev5 수리 대상)은 해소됨. floor_head boot OK, P3 PASS(6그룹 토큰 2,721,290), warmup·bench rc 0, bench 16/16(동시성 1, median TTFT 906.76 ms — 기술값, 인용 금지).
- **실패 지점**: `n3_plan.floor_from` 조건 (iii) "boot 안 **모든** prefill span이 (108,0)" 위반 1건: 28 span 중 span_seq 6(warmup 구간, `marks` bench_start 이전)이 `(prefill,decode)=(0,108)`, `stream_idx=5`, `running_bs=1`. floor bench 창 안의 span(마지막 16 + bench warmup 1)은 전부 (108,0)·running_bs 0.
- **기전(텔레메트리 + 소스 독해)**: 직전 요청의 decode가 1827721.495에 끝나고 125 ms 뒤 다음 요청 prefill이 들어왔을 때 `running_batch`가 아직 끝난 요청을 담고 있었다(batch_size 1). `multiplexing_mixin.adjust_stream_groups`(`src/multiplex/multiplexing_mixin.py:1169-1200`)는 "running 비어 있지 않음 ∧ split prefill 없음"이면 plain decode 그룹 `real_sm_group_num-1` = (0,108)을 고르고, 그 스트림에서 prefill이 실행됐다(span_end 시점 running_bs 0). 즉 동시성 1에서도 **끝난 요청의 필터링 전 상태가 다음 prefill의 그룹 선택에 새는 경쟁**이며, 이 boot에서 28 span 중 1회 발생. (0,108)도 plain(비green) 그룹이라 물리적 SM은 전량일 가능성이 높으나 미검증 — 등록 규칙은 쌍 라벨로 판정하므로 규칙대로 abort가 맞다(fail-closed 정상 작동, 하네스 결함 아님).
- **함의**: 같은 경쟁이 floor bench 창의 16 요청에 걸리면 (ii) `running_bs != 0`으로도 floor가 무효가 된다. 관측 빈도(1/28)를 그대로 쓰면 boot 전체 조건 (iii)의 통과 확률은 대략 40%, floor 창만으로 좁혀도 약 55% 수준(외삽, n=1 boot — 계획용 어림, 판정 아님). 그대로 재제출하는 것은 비효율.
- **수리 방향(규칙층 변경 ⇒ rev6 + 규칙층 델타 감사, 사용자 결정 필요 — 미수정)**: 후보는 대화 보고 참조.
- **(追記 2026-10-06, 메인 세션) 사용자 결정 "a" = rev6 권고안 승인 → rev6 작성**: floor 판정 입력을 입장 라벨에서 실현 증거로(각 span `prefill_span_end.stream_idx == 0` ∧ span 안 `decode_iteration` 0건; 짝 없는 span·`dropped_events > 0` 무효; 마지막 16 span `n_new/queue_len` 1/0 유지, 입장 `running_bs` 제외). 사전등록 머리말 rev6 줄·§4-1·§12-23·§13·§16·§17-5, `n3_plan.floor_from` 재작성 + selftest 대조 6+2, `n3_capacity.sh` ABORT_FLOOR 메시지 1줄, 테스트 `test_t_h3_floor` 변이 교체 + `TestFloorRev6RealJob` 2개. **진단 재계산(채점 아님)**: 2회차 실제 floor 자료에 rev6 CLI `--floor B` → `valid`, floor_s 0.9067605545278639 s(bench 중앙값과 동일), `entry_label_off_count` 1, `end_stream_idx_set` [0], `dropped_events_max` 0, decode_iteration 1581건 전부 span 밖. 2회차 Job은 여전히 셀 0·라벨 0·`RESUME_FROM` 불가. 델타 감사(rev5 봉인 `460fda2` 대비)는 아래 3회차 제출 절에 기록.

## shape B 2회차 실패 진단 (2026-10-06, 메인 세션, 로컬 CPU 재집계 — 판정 아님)

- 상태 `failed`, 컨테이너 08:17:37Z→08:25:58Z, `exit_code=7`, 마지막 줄 `ABORT_FLOOR floor invalid … (P-2)`. 회수 `vessl/pdmux-n3-capacity-79999aa-20261006081638/`(fetch sha 통과), OUT `n3b_pdmux-n3-capacity-79999aa-20261006081638/`.
- **rev5 수리는 작동했다**: preflight 전부 통과(plan·mix·analyze selftest OK, `ENGINE_TREE` 0 차이, preflight unittest 통과), units=48, 포트 창 정상, floor_head boot OK, P3 PASS(6그룹 2,721,290), warmup·floor bench rc 0(16/16, TTFT p50 906.76 ms·p99 909.99 ms).
- **실패 지점**: `n3_plan.floor_from` 조건 (iii) "boot의 모든 `prefill_span_start`가 (108,0)". 28 span 중 **1개**(span_seq 6, warmup 8건 중 4번째, bench 창 이전)가 `prefill_sms=0, decode_sms=108, stream_idx=5, running_bs=1`로 기록됨 → `T_K.json` `valid:false` → 셀 측정 0·라벨 0.
- **원인(코드 순서 + 텔레메트리로 확인, 기전 1개는 추정)**:
  1. `event_loop_pdmux`는 한 반복 안에서 `update_split_prefill_batch`(새 prefill 입장 + `_phase_on_prefill_start` 기록) → `update_running_batch`(끝난 요청 제거) → `adjust_stream_groups`(decode 비면 stream 0 = (108,0)) → prefill 커널 발사(`multiplexing_mixin.py:1408/1468`) 순서다. 즉 span_start의 분할·`running_bs`는 **입장 시점(조정 전) 상태**를 적는다.
  2. 직전 요청의 마지막 decode 결과와 다음 요청 도착이 **같은 반복**에 들어오면, 끝난 요청이 아직 `running_batch`에 남아 있어 `running_bs=1`·이전 decode stream(5 = (0,108))이 찍힌다. 그다음 같은 반복에서 끝난 요청이 제거되고 stream 0으로 조정된 뒤 prefill이 발사된다.
  3. 실현 분할은 (108,0)이었다: 같은 span의 `prefill_span_end.stream_idx=0`, span 시작 +15 ms·+47 ms `runtime_snapshot`이 `stream_index=0, prefill_sms=108, decode_running_batch_size=0`, span 안 decode_iteration 0건, prefill 시간 886.8 ms(다른 span 882–892 ms). — **기록 라벨의 경쟁(race)이지 분할 위반이 아니다.**
  4. 이 span 직전 decode→prefill 간격이 124.8 ms(다른 span 15.8–20.0 ms)로, 스케줄러 루프가 ~0.1 s 멈춘 동안 클라이언트가 마지막 토큰을 받고 다음 요청을 보낸 것으로 보인다. 후보는 cgroup CPU quota 쓰로틀(11 CPU, period 100 ms; bench 전 누적 `nr_throttled` 99)이나, 구간별 해상도가 없어 **미확정**.
- **채점 경로 영향 범위**: R3 arm identity는 `decode_iteration`(decode sync 시점 라벨)과 snapshot만 읽으므로 이 경쟁과 무관. span_start 분할 라벨을 판정에 쓰는 곳은 floor 조건 (iii)과 마지막 16 span의 `running_bs==0` 검사뿐(`n3_plan.py` `floor_from`). mix의 W-idx·cap-bound는 기술량.
- **6그룹 프로브(2026-10-05) 3 boot·33 span**에서는 `running_bs>0` span 0건 — 드문 경쟁(이번 boot 1/28).
- 수리 방향은 규칙(P-2 floor 판정 입력) 변경이라 사용자 결정 + rev6 델타 감사 필요. 이 문단은 진단 기록일 뿐 규칙·봉인 바이트를 바꾸지 않는다.

## shape B 3회차 제출 (rev6, 2026-10-06)

| shape | job name | slug | 제출(UTC) | 커밋 | 이미지 | spec | 승인 |
|---|---|---|---|---|---|---|---|
| B | `pdmux-n3-capacity-38ed3f4-20261006092813` | `job-n60kw8bsejuz` | 2026-10-06T09:28Z | `38ed3f4`(rev6 봉인) | `sglang-runtime@1e0d335b…` | `resourcespec-a100x1` | 사용자 2026-10-06 "a" = rev6 권고안 승인 + "사전등록 개정·델타 감사·재제출까지 진행"(catch-up 브리핑 선택지 A) |

- 선행: rev6 델타 감사 `VERDICT_n3_rev6_delta_2026-10-06.md` = `GO-with-caveats`(死因 0, R-1…R-4 반영, W-1…W-4 반영, W-5 미반영[사용자 결정], N3C-28…33 §11 승계, 재봉인 표 追記), `test_n3_capacity` 66/66, citation 0위반, doc_facts 0위반(11차 정정 957→960), presubmit 차단 = M4R·TC1 2건(등록된 범위), tracked tree clean. `RESUME_FROM` 미설정(1·2회차 모두 셀 0, 재개 불가, N3C-27). dry-run(커밋 전 `460fda2` 기준 + 커밋 후 `38ed3f4` 기준 2회)으로 명령 형태가 2회차와 동일함을 확인(`PDMUX_N3_SHAPE=B`, 같은 이미지·spec·볼륨).
- N3C-27 주의: 1·2회차가 `/data/substrate_history.tsv`에 측정 0 행을 남겼으므로 이번 Job의 `SUBSTRATE_WARNING`은 그 행 기준으로 발화할 수 있다(측정 셀 간 기판 혼합 아님).
- 제출 결과: 업로드 3 file(76.7 MB), Job 생성(slug `job-n60kw8bsejuz`), tag `n3-capacity`, `vessl_jobs.tsv` 행 추가(fetched=no). 회수: `scripts/vessl/fetch.sh n3_capacity pdmux-n3-capacity-38ed3f4-20261006092813`(DONE+sha256 통과가 완료).
