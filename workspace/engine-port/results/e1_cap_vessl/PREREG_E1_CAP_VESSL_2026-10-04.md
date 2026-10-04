# 사전등록 — E-1: running cap 48 vs 192 대조 (mamba pool 통제 포함) · VESSL A100 (rev1.1, 2026-10-04)

> **지위**: **rev1.1** = rev1 판정 규칙 그대로 + 문언·기술 필드 수정. rev1 규칙층 감사 `VERDICT_e1_cap_vessl_rules_rev1_2026-10-04.md` = **`GO-with-caveats`**(死因 0, E1C-1…9).
> rev1.1이 반영한 것: 감사 권고 R1·R2·R3·R5·R7·R9(전부 문언 또는 라벨 입력이 아닌 기술 필드, `e1_plan.decide()`는 rev1과 바이트 동일)과 E1C-1…9 등록(§15).
> 반영하지 않은 것: R4·R6·R8·블록 수·D arm 제거 — 규칙 변경이라 **사용자 결정 대기 옵션**(§16), 채택 시 델타 감사.
> 제출은 여전히 §10 게이트를 따른다(사용자 승인 전 `launch.sh --submit` 금지).
> - 사용자 승인(2026-10-04): E-1 사전등록 초안 + VESSL 하네스 작성(로컬 준비만, GPU 0).
> - 작성 시점 GPU 지출 0 · 성능 판정 0건 · Claim 등급 변경 0건 · 정본 미편집. 작성: engine-porter. 커밋은 메인 세션.
> - 같은 VESSL 계정에서 λ0 재앵커 Job이 돌고 있다. 이 문서·하네스는 그 Job에 아무것도 하지 않는다(§11-3).
>
> **파일**(모두 `workspace/engine-port/` 기준)
> - `results/e1_cap_vessl/e1_cap_vessl.sh` — 러너(VESSL Job target, **Job 1개 = 블록 1개**).
> - `results/e1_cap_vessl/e1_plan.py` — 등록 상수·블록 순서·예산·런별 분석·블록 유효성·라벨. 판정 술어는
>   `benchmarks/pdmux_eval/analyze.py`를 **호출**한다(복사 없음).
> - `tests/test_e1_cap_vessl.py` — 러너·헬퍼 구조/동작/변이 시험.
> - **엔진 변경 1건**: `src/multiplex/multiplexing_mixin.py`에 `PDMUX_PHASE_EVENTS`(기본 OFF) — §5. 시험 `tests/test_phase_events.py`.
>
> **줄 번호 인용 규약**: 커밋된 기존 파일은 `파일:줄`, 이 회차의 새·변경 파일은 아직 미커밋이라 **코드 텍스트**로 인용한다(λ5A-11·λ0 VESSL 追記와 같은 이유).

---

## 0. 한 줄 요약

HE0 캠페인과 같은 서빙 arm·같은 변화 trace(ShareGPT 200, rate 3↔12, 3라운드)를 VESSL A100에서, 한 Job(=한 노드) 안 무작위 순서 블록으로
static {d24, d34, d44} × {cap 48, cap 192}와 **pool 통제 arm**(cap 48 + pool 192)을 돌려, 후보 (7) — "버스트 prefill-ward 부호 충돌
(prefill-heavy static이 HI TTFT를 악화)은 running cap 48 = mamba pool 48 포화가 만든 것일 수 있다" — 를 **pool을 192로 고정한 채 cap만
48→192로 바꾸는 한 노브 대조**로 가른다. 동거율 정의 (a)를 직접 재는 per-iteration 이벤트(기본 OFF 신규 계측)와 그 관측자 효과 arm을 같이 돈다.
**VESSL 수치는 KISTI HE0 수치와 합치지 않는다.**

---

## 1. 질문 · 동기 · 가르지 않는 것

### 1-1. 출처
- `results/he0_contamination_2026-10-01/AUDIT_LOGIC_VERDICT_2026-10-01.md` B절 후보 **(7)**: 하네스 `--max-running-requests 48`,
  서버 로그 `max_mamba_cache_size: 48`, HI bs 포화 ⇒ "decode-heavy 유리·prefill-ward 불리(기계적), 부호 반전 가능성 불명", 해소 = E-1.
  같은 판정서 D-3: "cap(= mamba pool) 48이 포화된 과부하에서" 한정 문장, cap 비구속 일반화는 NOT-YET-SUPPORTED. E절 E-1 초안.
- `AUDIT_FOLLOWUP_VERDICT_2026-10-02.md` §2 **C-4**: 입장 가중 0.70(로그만)·대기열 존재 중 cap 만석 0.74(재구성), **arm 공통**,
  "cap이 부호를 만들었는지는 E-1만 가린다". §1·§6: 같은 job·같은 노드·무작위 블록 n ≥ 4, telemetry(`decode_sms`, `timestamp_monotonic_s`) ON.
- `CORESIDENCY_2026-10-02.md` B-1: 동거율 정의 (a) = prefill in-flight ∩ decode-active 벽시간(사용자 결정). 좁히려면 prefill 시작/종료·decode
  iteration 타임스탬프 telemetry가 필요하다.

### 1-2. 이 캠페인이 가르는 것 (단 하나)
"pool을 192로 고정했을 때, running cap을 48에서 192로 올리면 HI phase에서 d24(prefill-ward) 대 d44의 TTFT 부호·크기가 바뀌는가."

### 1-3. 가르지 않는 것 (등록)
- **HE0 자체의 재판정이 아니다.** HE0는 KISTI 기판의 결과다. VESSL에서 무엇이 나와도 HE0의 KISTI 수치·부호를 재스코어하지 않는다(게이트 1 연장).
- **동적 제어의 우열이 아니다.** bind+GATE arm(D)은 기술용이다(§6-6). 컨트롤러는 cap 48에서 튜닝된 것이고 cap 192에 맞게 재튜닝하지 않았다(게이트 8).
- **Bullet 재현이 아니다.** 재정렬·decode 지연 arm이 없다(E-2). p+d=108 엔진 강제(stake #1)도 그대로다.
- **다른 모델·trace로의 일반화가 아니다**(Zamba2-2.7B, HE0 trace 한정).
- **게이트 #6(SLO 용량)을 닫지 않는다.**

---

## 2. 기판

| 축 | 값 | 근거 |
|---|---|---|
| GPU | VESSL A100-SXM4-80GB ×1, 108 SM, cc 8.0, MIG off | `results/b5_smoke/` · `vessl_operating_model_2026-10-02.md` §10–§11 |
| 격자 실현 | (84,24)(74,34)(64,44)(92,16) 요청 = 실현 %smid 집합 크기, 서로소, graph replay 중 격리 유지 — **빈 GPU 미시 조건**. (54,54)는 미프로브 | B3a·B3c (2026-10-04) |
| 노드 | 공유 노드, Job마다 다를 수 있음 → **블록 = Job = 노드**, 노드는 블록 효과로 흡수 | 운영모델 §1-2 |
| 클럭 | 고정 불가. job_entry가 `meta/clk.csv`(100 ms)·`clk_summary.json` 기록 | `job_entry.sh` |
| 이미지 | `ghcr.io/asuan99/sglang-runtime@sha256:1e0d335b…`(pristine v0.5.10) + 시작 시 `install_runtime.sh` | `scripts/vessl/vessl.env` |
| 엔진 트리 | **이 커밋**의 `src/` + manual edits. HE0 실행 시점 코드(2026-07)와 같지 않다(R2·sticky·green read-out·이번 phase events 등 기본 OFF 추가물) | §3-2 D6 |

**KISTI와의 관계(LV-3 형식 승계)**: (a) E-1 수치를 KISTI HE0 수치와 평균·풀링·비율화하지 않는다. (b) "VESSL이 HE0를 재현/반박했다"고 쓰지 않는다.
(c) 부호가 같게 나와도 "KISTI 결과를 VESSL로 이식할 수 있다"로 쓰지 않는다. (d) 원인 귀속(드라이버·코드·노드)은 금지 — 여러 축이 동시에 다르다.

---

## 3. 설계

### 3-1. 서빙 arm = HE0 (같은 것)

| 항목 | HE0 출처 (`results/slo_sched/sharegpt_vary_bench.sbatch`) | E-1 |
|---|---|---|
| 모델 | `:35` `Zyphra/Zamba2-2.7B` | 같음. 리비전 `31afeeac…`(`_migration_2026-09-17/D_misc/hf_hub_revisions.tsv`) — VESSL `/data/hf` refs/main이 다르면 `ABORT_MODEL_REVISION` |
| 서버 플래그 | `:54-57` triton, `--disable-radix-cache`, mem 0.82, ctx 4096, `--enable-pdmux`, `--chunked-prefill-size -1`, `--disable-overlap-schedule`, cudagraph ON(CGFLAGS 빈 값) | 같음 + §3-2의 추가분. 구조 시험이 두 스크립트의 플래그 집합을 비교한다 |
| PD-mux config | `results/slo_sched/pdmux_d{24,34,44}.yml`(static, `sm_group_num 3`), `pdmux_slo.yml`(bind, 5 division) | **같은 파일**. sha256 등록(`e1_plan.CONFIG_DIGESTS`), 다르면 `ABORT_CONFIG_DRIFT`. 마지막 변경 2026-07-09/07-15(HE0 이전) |
| trace | `:68-74` ShareGPT `--sharegpt-context-len 4000`, `--num-prompts 200`, LO rate 3 → HI rate 12, 3라운드, **phase마다 별도 bench_serving 프로세스**(대기열 이월 없음) | 같음. trace 파일 sha256 `35f0e213…` 등록, 다르면 `ABORT_TRACE` |
| bind+GATE env | `:42-43` `PDMUX_SLO_SCHED=1 PDMUX_SLO_MODE=binding PDMUX_TPOT_SLO_MS=60 PDMUX_TTFT_SLO_MS=3000 PDMUX_SLO_ANCHOR_IDX=2` + `PDMUX_SLO_FEAS_GATE`(HE0는 미기록, FEAS 로그 줄로 추정 — 감사 후보 (9)) | 같음 + `PDMUX_SLO_FEAS_GATE=1` 명시 |
| duration | `:91-95` 라운드 duration **합산**(게이트 7) | `analyze.load_bench_serving_rounds` 호출(합산) |

### 3-2. HE0와 다른 것 — 전수

| ID | 차이 | 판정에 닿는 경로 | 처리 |
|---|---|---|---|
| D1 | `--random-seed 1`(HE0는 미지정 = 매 부팅 무작위) | 샘플링이 greedy(`temperature 0`)라 출력 경로에는 무관할 것으로 봄(코드 독해) | 전 arm 동일 |
| D2 | bench `--seed <블록 시드>`(HE0는 기본값 1 고정 → 전 런이 같은 200 prompt) | prompt 집합·도착 시각이 블록마다 바뀜(감사 E-1 권고 "블록별 seed 변동·블록 내 공유"). 블록 1 = 시드 1 = HE0와 같은 prompt·도착 | 블록 안 전 arm 공유 → 짝 비교 |
| D3 | 기판(§2), 엔진 트리(§2), Python/numpy(이미지) | 수준값 이동 | 비교 금지(§2) |
| D4 | telemetry ON(`PDMUX_TELEMETRY_PATH` + `PDMUX_PHASE_EVENTS=1`)이 primary arm 전부에 켜짐(HE0는 OFF) | 관측자 효과 | 전 primary arm **대칭**. 크기는 O arm으로 측정(§5-4) |
| D5 | `PDMUX_GREEN_READOUT=1`(boot당 1회 드라이버 조회) | 1회성 | 전 arm 대칭, 관측자 효과 0을 주장하지 않음 |
| D6 | 엔진 코드가 HE0 시점과 다름(기본 OFF 추가물들) | `adjust_stream_groups`·legacy loop 경로는 이번 회차에서 **바이트 불변**(phase events는 줄 위치를 옮기지 않는 같은 줄 편집 + 끝 추가, §5-2) | HE0 대비 귀속 금지(§2) |
| D7 | VESSL 배선(경로·캐시·결과 위치·예산 집행기·CG2 사전 boot 2회) | 집행기 발화 시 런 누락 | §8 |

### 3-3. 노브 허용 목록 (자유 표면 닫기)
러너는 허용 목록 밖의 `PDMUX_*`가 보이면 GPU 작업 전에 `ABORT_ENV`(exit 2). 허용: job_entry/launch 식별자(`PDMUX_COMMIT`·`_JOB_NAME`·`_CAMPAIGN`·`_TARGET`·`_IMAGE`·`_SPEC`·`_ROOT`·`_PROJECT_ROOT`·`_IO`·`_DATA`·`_OPT`),
`PDMUX_E1_BLOCK`, `PDMUX_E1_DRYRUN`. 테스트 전용(`PDMUX_E1_TEST_*`, `PDMUX_TEST_SKIP_SYNC`)은 DRYRUN이 아니면 `ABORT_TEST_KNOB_IN_MEASUREMENT`.
arm별 엔진 env(telemetry·phase events·green read-out·bind+GATE)는 **러너가 arm 표에서만** 만든다 — `--env`로 주입할 수 없다(구조 시험).

### 3-4. cap과 pool은 엔진에서 결합돼 있다 — 한 노브 대조의 구성

**엔진 사실(SGLang v0.5.10, 설치 트리 `srt/model_executor/model_runner_kv_cache_mixin.py`)**
1. `handle_max_mamba_cache`: `--max-mamba-cache-size`가 없고 `disable_radix_cache ∧ max_running_requests is not None`이면
   `server_args.max_mamba_cache_size = server_args.max_running_requests // (dp …)`(코드 텍스트 `# Use explicitly set max_running_requests when radix cache is disabled`).
   ⇒ HE0 명령줄에서 **pool은 cap에서 나온다**(서버 로그 `max_mamba_cache_size: 48`과 정합).
2. `_resolve_max_num_reqs`: `max_num_reqs = min(max_num_reqs, self.server_args.max_mamba_cache_size // ratio)`, radix off면 `ratio = 1`(`_calculate_mamba_ratio`).
   ⇒ **running cap은 pool에 묶인다.** "cap 192 + pool 48"은 running 48로 되돌아가 구성 불가.
3. pool 메모리는 `mem_fraction_static` 안에서 KV 토큰 풀과 나눠 쓴다(`rest_memory = self.handle_max_mamba_cache(rest_memory)` 뒤 `rest_memory // cell_size`).
   ⇒ pool을 키우면 **KV 토큰 수가 같이 줄어든다**(mem fraction은 0.82 고정, 바꾸지 않는다).
4. cudagraph 캡처 배치 크기는 `req_to_token_pool.size`(= running cap) 이하로 걸러진다(`cuda_graph_runner.py` `get_batch_sizes_to_capture`).
   ⇒ cap 192는 bs 56…192 그래프를 **추가로** 캡처한다(stream group마다). bs ≤ 48 그래프 집합은 cap 48과 같다. `cuda_graph_max_bs`는 기본 256(HE0 로그 `cuda_graph_max_bs=256`)으로 전 arm 같다.

**그래서 등록하는 세 cap 구성**

| 구성 | `--max-running-requests` | `--max-mamba-cache-size` | 실현 pool | 바뀌는 것 |
|---|---|---|---|---|
| **C48** | 48 | (없음) | 48 | HE0 명령줄 그대로 |
| **M** | 48 | **192** | 192 | C48 대비 **pool만**(+ KV 토큰 감소). running cap 48·캡처 집합 동일 |
| **C192** | 192 | (없음) | 192 | M 대비 **cap만**(+ cap에 딸린 캡처 bs 56…192). pool 192·KV 토큰 동일 |

- **cap 효과(한 노브) = M → C192.** **pool/KV 재배분 효과(한 노브) = C48 → M.** C48 → C192는 두 노브가 함께 움직이므로 판정 대조로 쓰지 않는다.
- "cap에 딸린 캡처 bs 56…192"는 cap을 올리는 한 피할 수 없다(캡처하지 않으면 bs > 48이 eager로 떨어져 운영점이 깨진다). 캡처 추가분은 "cap 192 처치의 일부"로 정의한다.
- **메모리 산술(KISTI HE0 로그 기준, VESSL에서 배너로 재확인)**: pool 48 ⇒ ssm 3.23 GB + conv 0.08 GB, KV 327,359 토큰(K+V 56.2 GB).
  pool 192 ⇒ ssm·conv 약 4배(≈13.2 GB) ⇒ KV ≈ 46 GB ≈ 27만 토큰으로 줄어든다(예상). HE0 HI 200 요청 전부가 동시에 최대 길이로 상주해도
  ≈ 200 × (평균 입력 341 + 평균 출력 237) ≈ 11.6만 토큰(HE0 HI jsonl `input_lens`·`output_lens`)이므로 KV는 구속되지 않을 것으로 예상한다 — **예상일 뿐이고 M3로 측정·게이트한다**(§6-1).
- 배너 게이트: 매 런 서버 로그의 `max_mamba_cache_size`·`max_running_requests`가 등록값과 다르면 그 런은 무효(§6-1 M4).

### 3-5. arm 표 (12개, `e1_plan.ARMS`)

| arm | 제어 | config | cap | pool | telemetry | 역할 |
|---|---|---|---|---|---|---|
| S24_C48 · S34_C48 · S44_C48 | static | d24/d34/d44 | 48 | 48(auto) | ON | P |
| S24_C192 · S34_C192 · S44_C192 | static | d24/d34/d44 | 192 | 192(auto) | ON | P |
| S24_M · S44_M | static | d24/d44 | 48 | 192(explicit) | ON | M (pool 통제) |
| S34_C48_OFF · S34_C192_OFF | static | d34 | 48 / 192 | auto | **OFF** | O (관측자 효과) |
| B_C48 · B_C192 | bind+GATE | slo | 48 / 192 | auto | ON | D (기술용) |

- telemetry ON = `PDMUX_TELEMETRY_PATH` + `PDMUX_PHASE_EVENTS=1` + `PDMUX_RUN_ID`/`PDMUX_WORKLOAD_ID`(기존 32-샘플 `runtime_snapshot` 격자 포함, `PDMUX_DUAL_WORKER_TRACE_EVERY` 기본값).
  OFF = 그 넷 모두 없음(HE0와 같은 상태). green read-out은 전 arm ON.
- **d16·d54 static은 넣지 않는다**: 사용자 지정 최소 집합이 {d24,d34,d44}이고, 판정 대조는 d24 vs d44다. d34는 경향 기술용.
- **no-gate(bind, FEAS 게이트 없음) n ≥ 8은 이번 등록에 넣지 않는다**: 그 질문(856964 서버 정지·게이트 견고성)은 cap과 직교하고, 드문 사건을 n = 8로 가를 수도 없다.
  필요하면 별도 등록(§13).

### 3-6. 블록 · 순서 · 시드
- **Job 1개 = 블록 1개**: 12 arm을 한 번씩, 블록별 고정 무작위 순서로(`random.Random("e1cap-2026-10-04:<block>").shuffle(sorted(arms))`). 순서는 실행 전에 `REGISTRATION_SHA256.txt`에 기록된다.
- bench 시드 = 블록 번호(1…8). 블록 안 전 arm·전 라운드·LO/HI 모두 같은 시드(HE0 구조: 6개 하위 런이 같은 prompt 집합).
- 블록 1…6 = 본 블록, 7·8 = **예비**(본 블록이 무효일 때만 순서대로 대체, §7). 블록은 Job을 넘지 않는다 — **재개 없음**. 끊긴 블록은 무효이고 예비로 대체한다.
- 블록 = 노드 = 시점이므로 노드·시점 효과는 블록 효과로 흡수되고, 판정량은 **블록 안 짝 차이**다.

---

## 4. 측정량

모든 percentile은 `analyze.percentile`(선형 보간), 요청 판정 술어는 `analyze.RequestResult.passes`/`passes_legacy_mean`(호출).
phase(LO/HI)별로 3라운드 600요청을 합치고 duration은 합산.

- **Primary**: HI phase **TTFT p50**(ms). 블록 b, cap 구성 c에서 `Δc,b = ln(TTFT_p50[S24_c] / TTFT_p50[S44_c])` — 양수 = prefill-ward(d24)가 HI TTFT를 악화(HE0 방향).
  - 선택 근거: 후보 (7)과 실체 충돌 (a)는 **TTFT 부호**에 관한 주장이다. TTFT p50은 지시함수가 아니라 cliff가 없다(게이트 6).
  - ★rev1.1 정정(감사 R3·E1C-6): rev1은 "p50은 정지 꼬리에 비민감"이라 적었으나 **KISTI 원자료로 반증됐다** — 859006을 포함하면 C48 대조 Δ48(p50)이 0.157에서 0.231로 바뀌고, 859025(d16)의 HI p50은 683 ms로 같은 arm의 2095–2225 ms와 다르다(판정서 질문 1). p50도 정지에 견고하지 않다. 이상치 블록은 CI를 넓혀 라벨을 `UNRESOLVED_AMBIGUOUS` 쪽으로 미는 방향이며, 런별 정지 사건은 기술 필드로만 병기하고 **사후 제외는 금지**한다(§5-5).
- **Secondary(기술, 라벨 없음)**: HI TTFT p95/p99/mean의 같은 Δ; HI goodput 4술어의 `ln(gp[S44]/gp[S24])`(양수 = d24 열위) —
  정본 술어(TTFT ≤ 3 s ∧ 요청내 ITL p95 ≤ 60 ms, 게이트 4) · legacy(mean ITL ≤ 60) · ITL p95 ≤ 65 · TTFT ≤ 3 s만.
  ★legacy 술어 정의 차이: HE0 하네스(`:97`)는 ITL이 없는 요청을 실패(9.0 s)로 셌고, `analyze.passes_legacy_mean`은 통과로 센다. ShareGPT 표본은 `benchmark/datasets/sharegpt.py`가 출력 길이 < 2를 버려 요청마다 ITL이 최소 1개라 영향이 없을 것으로 보지만(ignore_eos 가정) 런별 `no_itl_requests`를 기록한다.
  정본 술어는 **metric cliff 진단**(요청내 ITL p95 ∈ [55, 65] ms 비율 `cliff_frac`)과 함께만 보고한다(게이트 6, KISTI에서 cliff 증폭 `CONFIRMED`).
- 게이트 5 병기: 전 arm의 TTFT p50/p95/p99, 요청내 ITL p95 분포의 p50/p95/p99, 토큰 ITL p50/p95/p99. 동적 arm은 switch_count(`SLO-BIND` 줄) + 분할 체류분포(§5-5).

---

## 5. 계측 — `PDMUX_PHASE_EVENTS` (엔진 변경, 기본 OFF)

### 5-1. 기존 계측으로 되는가 — 안 된다
- 기존 `runtime_snapshot`은 `_dual_worker_sync`의 **32번째 sync마다** 표본이다(`decode_sms`·`stream_index`·`timestamp_monotonic_s` 포함). 짧은 prefill 구간을 놓친다는 것은 이미 기록돼 있다(같은 함수의 주석: E1/T8/d44에서 prefill 활성 4.9% 대 표본 0.07%).
- `PDMUX_DUAL_WORKER_TRACE_EVERY=1`로 올리면 **유휴 루프 회전마다** 표본이 나간다(이벤트 루프는 idle에서도 돈다). 큐(65,536)가 차서 `dropped_events`가 생기고, 다음 phase 시작 직후 이벤트가 버려질 수 있다.
- 그래서 **일이 있을 때만** 나가는 이벤트 3종을 최소 추가했다.

### 5-2. 설계 (`src/multiplex/multiplexing_mixin.py`)
- 이벤트(같은 `AsyncJsonlTelemetry` writer, 스케줄러 스레드에 파일 I/O 없음):
  - `prefill_span_start` — split-prefill 배치 입장 시: `span_seq, n_new, extend_tokens, running_bs, queue_len, cap, stream_idx, prefill_sms, decode_sms`.
  - `prefill_span_end` — 그 배치의 커널 완료를 루프가 관측한 순간(merge 전): `span_seq, n_done, running_bs, queue_len, stream_idx`.
  - `decode_iteration` — decode 한 step: `iter_seq, t_launch`(decode forward 발행 직후, 이 iteration의 prefill chunk 발행 전), `t_sync`(decode stream drain 직후), `bs, stream_idx, prefill_sms, decode_sms, prefill_in_flight, prefill_kernel_done_wait, queue_len`.
  - split-prefill 배치는 한 번에 하나뿐이므로 start/end가 엄격히 교대한다(`span_seq`로 짝).
- **기본 OFF 보장**: `resolve_phase_events`가 unset/`"0"`이면 OFF, `"1"`만 ON, 그 밖의 값은 거부(`ValueError`). ON인데 `PDMUX_TELEMETRY_PATH`가 없거나 `PDMUX_LA_COORD`(死 트랙)면 boot 거부(`RuntimeError`).
  OFF에서 각 hook은 속성 1회 검사 후 반환하고 이벤트·필드·상태를 만들지 않는다.
- **줄 위치 불변**: 다섯 호출 지점은 같은 줄 편집(`; hook(...)`)이고 나머지는 클래스 끝·파일 끝에 추가했다. `scripts/discipline/line_citations.json`의 스냅숏 인용(201개)이 동기화된 트리 기준으로 위반 0이다.
  호출 지점: `init_pdmux`의 `dual_worker_trace_error_logged` 줄, `_dual_worker_start_prefill`, `_dual_worker_prefill_ready`, legacy loop의 `decode_done = True` 줄, decode `synchronize()` 줄.
- **관측 전용**: 스케줄링·입장·분할 선택·배치 구성에 쓰이지 않는다. 레코드 생성/emit 실패 시 그 boot 동안 이벤트를 끄고 1회 경고(부팅을 죽이지 않음).
- cudagraph: 호스트 측 Python만 추가된다. stream group·캡처 shape·eager 경로 변경 없음(운영점 cudagraph ON 유지).
- 범위 밖: coordinated loop(거부), true-dual 경로(호출 지점은 공유 코드라 이벤트가 나가지만 **시험하지 않았다** — E-1은 legacy만 쓴다).

### 5-3. correctness gate (성능 측정 전)
1. **CPU 회귀**(`tests/test_phase_events.py`, 실제 `event_loop_pdmux`를 CPU fake로 구동): OFF에서 이벤트 0·스케줄링 추적(입장·prefill chunk·기타 telemetry)이 ON과 **동일**,
   decode step당 이벤트 1개(루프 자체 카운터와 일치), start/end 교대, `prefill_in_flight`가 실제 동거 iteration과 일치, emit 실패 시 비활성화·스케줄링 불변, 설치 트리 반영 확인.
   변이 6종(각 hook 제거 5지점 중 단독 제거 가능한 4 + 기본값 ON 강제 + in-flight 반전) 전부 KILLED, 무변이 사본 PASS(교훈 53·254). Job preflight에서 다시 돈다.
2. **GPU 출력 동치(CG2, Job마다)**: 측정 전에 S44_C48 서버를 ON/OFF로 각각 띄워 ShareGPT 16 prompt를 동시성 1·greedy로 생성, **생성 텍스트 16/16 동일**이어야 한다.
   불일치·부팅 실패·ON에서 `decode_iteration` 0건이면 `ABORT_CG2_*`(exit 7), 그 Job의 블록은 돌지 않는다.
   ★자기 공시: 동시성 1에서 greedy가 boot 간 결정론적이라는 것은 가정이다. 불일치가 나면 그것은 "계측이 출력을 바꿨다"의 증거가 아니라 **조사 대상**이다(fail-closed).
3. **thread-local role patch 확인**: 설치 트리 `parallel_state.py`에 `pdmux_role_is_thread_local`이 없으면 `ABORT_ENGINE`. (E-1은 true-dual을 쓰지 않으므로 기능상 필요는 없고, CLAUDE.md 게이트 3의 설치 확인으로만 둔다.)
4. 설치된 `multiplexing_mixin.py`가 이 커밋의 `src/` 미러와 바이트 동일하지 않으면 `ABORT_ENGINE`.

### 5-4. 관측자 효과 (게이트 4)
- O arm 2개(S34_C48_OFF, S34_C192_OFF)와 짝 ON arm(S34_C48, S34_C192). 블록 짝 차이 `ln(m[ON]/m[OFF])`, m ∈ {HI TTFT p50, HI legacy goodput}, cap 2 × 지표 2 = 4개.
- **OBSERVER_OK** ⇔ 4개 모두 n ≥ 4, |평균| < ln 1.03, 95% t-CI가 0 포함. 아니면 **OBSERVER_FAIL**(어느 것인지 기록).
- 효과: OBSERVER_FAIL이어도 primary 라벨은 계산한다(전 primary arm이 같은 계측 하에 있으므로 대칭) — 단 라벨 문장에 "telemetry ON 조건에서"를 필수 병기하고, 동거율·cap-bound 수치는 "관측자 효과 미통과 하 기술값"으로만 쓴다.
- ★한계(자기 공시): "CI가 0 포함"은 부재의 증거가 아니다. n = 6에서 이 검정의 검출력은 블록 간 산포에 달렸고, 산포는 이 캠페인이 처음 잰다. d34에서만 재므로 d24(동거 최대)의 관측자 효과는 외삽이다.

### 5-5. 동거율 (a)와 실현 분할 — 정의 (기술용, 판정 입력 아님)
- phase 창: 러너가 bench 프로세스 앞뒤에 `time.perf_counter()`(CLOCK_MONOTONIC — 서버 이벤트와 같은 시계, Job마다 확인·아니면 `ABORT_CLOCK`)와 벽시계를 `PHASE_MARKS_<arm>.tsv`에 남긴다.
  창 = 그 구간 안 첫 작업 이벤트 ~ 마지막 작업 이벤트(bench의 데이터셋 적재 유휴 시간 제외).
- PF = ∪[prefill_span_start, prefill_span_end], DEC = ∪[t_launch, t_sync]. **동거율 (a)** = |PF ∩ DEC| / |창|(`coresidency_a_wall`), 보조 |PF ∩ DEC| / |DEC|(`coresidency_a_decode`).
  둘 다 **호스트 관측** 구간이고 해상도는 루프 iteration 1회 수준이다. GPU 실행 구간과 같다고 주장하지 않는다.
- 실현 분할: DEC를 `decode_sms`로 나눈 시간 비율(phase별). static arm의 **pin 검사**: HI의 동거 decode iteration 중 `decode_sms == D` 비율 ≥ 0.99, 아니면 런 무효(Stage 0 D108 유형 — target이 아니라 realized로 검증).
- cap-bound 입장 비율(C-4의 VESSL 판): HI `prefill_span_start` 중 `running_bs + n_new ≥ cap ∧ queue_len > 0` 비율. 서버 로그 `Prefill batch` 줄로 같은 정의를 따로 계산해 대조한다(로그는 1초 해상도로 phase에 배정).
- **정지 사건(감사 R9, 기술 필드 — 제외 규칙 아님)**: HI 창 안에서 (i) telemetry 작업 이벤트 사이 간격 ≥ 1 s(`stalls`: 횟수·합·최대), (ii) 서버 로그 타임스탬프(1 s 해상도) 연속 간격 ≥ 2 s = 무음 ≥ 1 s(`log_silences_HI`). 라벨 출력에 PRIMARY 6 arm × 블록 표(`stalls_HI_descriptive_not_exclusion`)로 병기한다. 어떤 런도 이 값으로 제외하지 않는다(E1C-6).
- **M2b(감사 R5, 기술 지표 — 라벨 입력 아님)**: HI `decode_iteration` 중 `queue_len > 0 ∧ ¬prefill_in_flight ∧ bs < cap` 비율(개수·시간 가중) = cap 외 원인(토큰 예약 등)에 의한 입장 정체의 지표. M2·M3가 PrefillAdder 토큰 예약 구속을 못 본다는 E1C-7의 빈칸을 기술로만 채운다.
- 손실: 런별 `dropped_events` 최댓값을 기록한다. 0이 아니면 그 런의 telemetry 기반 값은 "표본 손실 하 기술값"이고 cap-bound는 로그 값을 쓴다(LV-12 형식).

---

## 6. 판정 규칙 (사전 고정, `e1_plan.decide`·`label_blocks`)

통계: 사용 블록 n개의 짝 차이에 대한 평균과 **95% t-CI**(df = n−1). 효과 크기 문턱 **G = ln 1.03**(게이트 3).
`sig_pos(x)` ⇔ 평균 > G ∧ CI 하한 > 0 · `sig_neg(x)` ⇔ 평균 < −G ∧ CI 상한 < 0 · `within(x)` ⇔ CI ⊂ (−G, +G).

### 6-1. 조작 확인 (라벨 전에 평가)
- **M1 (cap 48이 이 기판에서도 구속하는가)**: C48·M arm(S24/S44)의 HI cap-bound 입장 비율 **중앙값 ≥ 0.50**. (KISTI는 로그만 추정 0.70 — 후보 (7)의 전제. 문턱 0.50은 "과반 입장이 cap에 묶임"으로 정했다.)
- **M2 (cap 192는 구속하지 않는가)**: C192 arm(S24/S44)의 같은 비율 **중앙값 ≤ 0.10**.
- **M3 (KV가 새 구속이 되지 않았는가)**: C192·M arm 서버 로그 `full token usage` **최댓값 < 0.90**.
- 기술 병기(라벨 입력 아님, E1C-7): `M1_per_arm_median`·`M2_per_arm_median`(arm별 중앙값), M2b(§5-5). 규칙의 M2는 rev1 그대로 S24_C192·S44_C192 **합동** 중앙값이다(arm별 최댓값 규칙 = R4, §16).
- **M4 (배너)**: 런별 `max_running_requests`·`max_mamba_cache_size`가 등록값과 같아야 그 런이 유효.
- **pin**: §5-5. 런 유효 조건.

### 6-2. 판정 대조
- `Δ48` = Δ(C48), `ΔM` = Δ(M), `Δ192` = Δ(C192) (§4). **cap 효과 `DiD_cap = ΔM − Δ192`**(pool 192 고정, 한 노브). pool 효과 `DiD_pool = Δ48 − ΔM`(기술).

### 6-3. 라벨 (위에서부터 처음 맞는 것)

| 순서 | 조건 | 라벨 |
|---|---|---|
| 1 | 사용 블록 < 4 | `UNRESOLVED_N_BELOW_4` |
| 1b | M1·M2·M3 중 하나라도 측정 불가(입장 0건·로그 사용률 줄 0건 → NaN) | `UNRESOLVED_MANIPULATION_UNMEASURED` |
| 2 | M1 거짓 | `UNRESOLVED_CAP48_NOT_BINDING` |
| 3 | M2 거짓 | `UNRESOLVED_CAP192_BINDS` |
| 4 | M3 거짓 | `UNRESOLVED_KV_BINDS` |
| 5 | ¬sig_pos(Δ48) | sig_neg(Δ48)면 `UNRESOLVED_BASELINE_REVERSED`, 아니면 `UNRESOLVED_BASELINE_NOT_REPRODUCED` |
| 6 | ¬sig_pos(ΔM) | `UNRESOLVED_SIGN_LOST_AT_POOL192` |
| 7 | sig_neg(Δ192) | **`CAP_BOUND_REVERSED`** |
| 8 | sig_pos(DiD_cap) | within(Δ192)면 **`CAP_BOUND_VANISHED`**, 아니면 **`CAP_ATTENUATED`** |
| 9 | sig_pos(Δ192) | **`NOT_CAP_BOUND`** |
| 10 | 그 밖 | `UNRESOLVED_AMBIGUOUS` |
| — | 블록 원장 중복·미등록 블록 | `UNRESOLVED_LEDGER_CONFLICT` |

- 라벨은 상호배타·망라다(순서 평가). 각 분기는 selftest와 합성 캠페인 9종으로 도달 가능함을 시험한다(§12).
- **`UNRESOLVED_*`는 후보 (7)에 대한 음성 결과가 아니다**(게이트 #21: 측정 실패 ≠ 발견). 특히 `BASELINE_*`는 "이 기판·이 노드들에서 cap 48 기준 부호가 재현되지 않아 (7)을 시험할 수 없었다"로만 쓴다.
- ★검출력 공시: 문턱 3%와 n = 4–6 t-CI의 조합에서 `CAP_BOUND_VANISHED`(CI 전체가 ±3% 안)는 블록 간 산포가 작아야만 도달 가능하다. 산포가 크면 같은 실체가 `CAP_ATTENUATED`나 `UNRESOLVED_AMBIGUOUS`로 나온다. 이것은 등록된 보수성이다.

### 6-4. 라벨별 허용·금지 문구
허용 문장(`e1_plan.ALLOWED_TEXT`, 전부 "이 VESSL 노드 집합·Zamba2-2.7B·HE0 trace·cudagraph ON·chunked prefill off" 범위 병기 필수):
- `CAP_BOUND_REVERSED`: "pool을 192로 고정하고 cap을 48→192로 올리자 HI TTFT p50에서 d24가 d44보다 낮아졌다(부호 반전)."
- `CAP_BOUND_VANISHED`: "… d24의 HI TTFT 불리가 사라졌다(짝 CI가 ±3% 안)."
- `CAP_ATTENUATED` (rev1.1, E1C-3): "pool을 192로 고정하고 cap을 48→192로 올리자 d24의 HI TTFT p50 불리가 3% 넘게 줄었다(DiD_cap 짝 CI > 0)." 잔여 불리는 Δ192가 sig_pos일 때만 "남았다"고 쓰고, 아니면 "cap 192에서의 잔여 불리는 판정되지 않았다(Δ192 CI 병기)"로 쓴다.
- `NOT_CAP_BOUND` (rev1.1, E1C-2): "cap 192(pool 192)에서도 d24의 HI TTFT p50 불리가 3% 넘게 남았다(부호 유지, Δ192 짝 CI > 0)." DiD_cap 평균·CI를 반드시 병기하고, "cap을 올려도 불리가 줄지 않았다 / 3% 넘게 줄지 않았다"로 쓰지 않는다(DiD_cap 비유의는 감쇠 부재의 증거가 아니다).
- (rev1 문장은 감사에서 고정 규칙의 문언 결함으로 판정됐다 — 부분 구속 참값에서 rev1 NOT_CAP_BOUND 문장은 30–57%, 완전 구속 참값에서 rev1 ATTENUATED 영문 문장은 92% 거짓 예보. 코드 상수 `ALLOWED_TEXT`도 같이 고쳤다.)

금지(`e1_plan.FORBIDDEN_TEXT`, 전 라벨 공통):
1. VESSL 수치와 KISTI HE0 수치를 섞는 문장(풀링·비율·"HE0 크기를 재현").
2. "HE0가 반박/확증됐다" — E-1은 다른 기판에서 후보 (7)을 시험한다.
3. D arm으로 "동적 제어가 static을 이긴다/진다".
4. C48 대 C192만으로 "admission cap이 HE0를 설명한다"(두 노브가 움직임; cap 대조는 M→C192).
5. "Bullet 결과를 재현했다".
6. `UNRESOLVED_*`를 후보 (7)의 음성 결과로 읽는 문장.
7. 동거율·cap-bound 수치를 그 런의 `dropped_events`와 관측자 효과 결과 없이 인용.
8. (추가) `CAP_BOUND_*`·`NOT_CAP_BOUND`를 HI TTFT p50 부호 밖으로 확장: goodput·ITL·HE0 순위·실무 권고(decode-heavy static)·"SM 분할 자체의 효과가 없다"·"버스트에서 prefill-ward가 낫다"(정책 주장 = 게이트 1). ★rev1.1: rev1 코드 `FORBIDDEN_TEXT`에는 이 8번이 빠져 있었다(감사 R2·E1C-4) — 이제 8개로 일치한다.

### 6-5. secondary
TTFT p95/p99/mean과 goodput 4술어의 같은 대조를 C48/M/C192별로 평균·CI만 보고한다. 라벨을 만들지 않는다. primary와 부호가 다르면 그 사실을 병기한다.

### 6-6. 동적 arm (기술용)
- B_C48·B_C192: switch_count, HI/LO 실현 분할 체류분포(`decode_sms` 시간 비율, 동거 iteration 한정 포함), TTFT/ITL p50/p95/p99, goodput.
- ★결합 공시: FEAS 게이트의 혼잡 가드는 `bs ≥ 0.85 × max_running_requests`(`_slo_feasible`)라 cap 192에서는 문턱이 163으로 움직인다 — B_C192는 "같은 코드, 다른 유효 동작"이다. B_C48 대 B_C192 차이를 cap 효과로 쓰지 않는다.
- C-1(자리 분해) VESSL 판: 결과 분석(result-analyst) 몫으로 넘긴다. 등록하는 것은 입력뿐 — 같은 블록의 static S24/S34/S44(같은 cap)과 B arm의 동거 iteration 분할 체류 가중치.
  가중치의 d24+d34+d44 합이 0.95 미만이면(d16/d54 체류) "C-1 계산 불가"로 기록.

---

## 7. 블록 유효성 · 선택

- 런 유효(`analyze_run`): 배너(M4) · LO/HI 각 3라운드 × 200 완료 · 요청 오류 0 · PHASE_MARKS 6줄 rc 0 · 예산 집행기 미발화 · telemetry ON arm은 `decode_iteration` > 0 · static ON arm은 pin ≥ 0.99.
- **블록 유효(primary)** ⇔ PRIMARY 6 arm(S24/S44 × C48/M/C192)이 모두 유효. O·D arm 실패는 그 분석에서만 빠진다.
- 선택: 본 블록 1…6 중 유효한 것 + (무효이거나 미제출인 본 블록 수만큼) 예비 7·8 중 유효한 것, 블록 번호 순, 최대 6. 같은 블록 번호가 두 Job에 있으면 `UNRESOLVED_LEDGER_CONFLICT`.
- **선택적 중단 없음**: 본 블록 6개는 중간 결과와 무관하게 전부 제출한다. 라벨은 본 블록(과 필요한 예비)이 모두 회수된 뒤 한 번만 계산한다.
  예비는 본 블록이 **무효**일 때만 쓴다(유효 판정은 `--block-summary`로 기계적, 결과값을 보지 않는다).

---

## 8. 예산 · 하드캡 · Job 분할 (`e1_plan.budget()`, 단가 $1.48/h)

| 항목 | 값 |
|---|---|
| 런 1개 예상 | boot 300 s(★자리표시자, §11) + bench 480 s + teardown 20 s + 분석 15 s ≈ 815 s |
| Job 1개 예상 | preflight 1,200 s(빠른 단위시험 + CG2 boot 2회) + 12 × 815 s ≈ **10,980 s ≈ 3.05 GPU-h ≈ $4.51** |
| 캠페인 예상(본 6 Job) | **≈ 18.3 GPU-h ≈ $27.1** |
| Job 셀 구간 캡 | job_entry 시작부터 **14,400 s(4.0 h)**, 꼬리 예비 600 s. 상수는 `e1_plan.HARD_CAP_S`, 측정 모드에서 env로 못 바꿈 |
| 입장 규칙 | 남은 시간 ≥ **1,500 s**일 때만 런 시작(아니면 `NOT_ATTEMPTED_BUDGET`, 이후 런 전부 생략 → 블록 무효 → 예비) |
| 실행 중 상한 | bench마다 `timeout -k 30 <남은 시간>` + watchdog(마감에 `sglang.bench_serving`·`sglang.launch_server` 종료). 이중 kill 층 — 한 층 제거는 등가 변이(시험에 기록) |
| 캠페인 천장(본 6 + 예비 2, 각 캡 + 꼬리) | ≈ 33.3 GPU-h ≈ **$49.3** — 셀 구간 집행 기준이며 **과금 하드캡이 아니다**(LV-13 형식: preflight 일부·꼬리·반출에는 kill 층이 없고, 크레딧 차단은 생성 차단이다) |

- 운영모델 §1-3 "Job당 ≲3–4 GPU-h"에 예상치가 들어간다. 모든 런이 입장 상한(1,500 s)만큼 걸리는 최악이면 캡 안에 8런만 들어가 블록이 무효가 된다 — 그 경우 예비로 넘어가고, 예비까지 무효면 `UNRESOLVED_N_BELOW_4`.
- 감사 E-1 초안의 "≈ 10–20 GPU-h"보다 크다: n = 6 블록 + 관측자 arm + pool 통제 arm 때문이다. 줄이는 선택지(사용자 결정): 본 블록 4개(≈ 12.2 GPU-h ≈ $18.1, 검출력 하락), D arm 제외(블록당 −2런).
- 재개: 없음(블록 = Job 원자). 같은 Job 안 재실행 시 `.complete`가 있는 런은 건너뛴다(Workspace 디버깅용).

---

## 9. 등록 기록 · 중단 코드

- Job마다 `REGISTRATION_SHA256.txt`: 이 문서, 러너, 헬퍼, `analyze.py`, 두 시험 파일, `multiplexing_mixin.py`(src), config 4개의 sha256 + 커밋 + 블록·시드·순서.
  `SUBSTRATE_E1.txt`: CPU·cgroup·TZ·`perf_counter` 시계 구현·torch·GPU UUID·드라이버·모델 refs/main·trace sha256. job_entry의 `substrate.json`·`clk.csv`·manifest를 함께 반출.

| 코드 | exit | 조건 (전부 GPU 작업 전, CG2 제외) |
|---|---|---|
| `ABORT_ENV` / `ABORT_TEST_KNOB_IN_MEASUREMENT` | 2 | §3-3 |
| `ABORT_BLOCK` | 2 | `PDMUX_E1_BLOCK` ∉ 1…8 |
| `ABORT_PREREG` / `ABORT_CONFIG_DRIFT` | 3 | 이 문서 부재 / config sha256 불일치 |
| `ABORT_MODEL_REVISION` / `ABORT_TRACE` / `ABORT_CLOCK` | 3 | Zamba2 refs/main ≠ `31afeeac…` / ShareGPT sha ≠ `35f0e213…` / perf_counter ≠ CLOCK_MONOTONIC |
| `ABORT_ENGINE` | 3 | 설치 트리에 phase events·thread-local patch 없음, 또는 설치 mixin ≠ src 미러 |
| `ABORT_PREFLIGHT` | 5 | 헬퍼 selftest·단위시험 실패 |
| `ABORT_CG2_BOOT` / `_BENCH` / `_MISMATCH` / `_NO_EVENTS` | 7 | §5-3 |
| `ABORT_2_CONSECUTIVE_BOOT_FAILURES` | — | 이후 런 미시도(블록 무효) |

---

## 10. 제출 게이트 · 선결 조건

1. 이 문서가 규칙층 `GO`/`GO-with-caveats`를 받는다(`NO-GO`면 rev2).
2. **Zamba2-2.7B를 `/data/hf`에 적재**(CPU Workspace `pdmux-build`, `stage_data.sh hf_hub_revisions.tsv Zyphra/Zamba2-2.7B` — 스토리지만, GPU 0). 현재 VESSL에는 Nemotron-Nano-9B-v2만 있다(운영모델 §9).
3. §11의 λ0 의존 자리표시자를 채우는 追記(rev2 또는 追記) — **결과 비의존 항목만**(boot 시간·스로틀 빈도). λ0의 처리율 값은 E-1 부하를 바꾸지 않는다(§11-1).
4. 메인 세션 커밋. 커밋 후 Job의 `REGISTRATION_SHA256.txt`가 이 커밋 바이트를 기록한다.
5. CPU 회귀(공개 이미지 컨테이너 + 로컬) · `check_citation_stops.py` 0위반 · `check_line_citations.py --check --all` 0위반(동기화된 트리 기준).
6. 사용자가 `launch.sh` dry-run 출력(명령·단가·job name, `--env PDMUX_E1_BLOCK=k`)을 보고 블록 1의 제출을 승인한다. 이후 블록 2…6은 같은 승인 범위 안에서 순차 제출(같은 커밋).
   ★**절차 동결(감사 R7·E1C-9)**: 본 블록 수(6 또는 4), D arm 포함 여부, `BOOT_S`·`RUN_NEED_S` 追記는 **블록 1 제출 전**에 `e1_plan.py` 코드 상수로 고정해 `REGISTRATION_SHA256.txt`에 남긴다. 블록 1 회수 후의 rev2는 **엔지니어링 실패**(boot·CG2·events 0건·배너·pin 측정 불가)로만 발동할 수 있고, 발동하면 그 이전 블록은 라벨에서 제외한다. M1–M3·Δ 값을 보고 설계를 바꾸는 것은 등록 위반이다.

---

## 11. λ0 결과 의존 · 동시 실행

### 11-1. λ0이 E-1에 주는 것과 주지 않는 것
- **주지 않는 것(등록)**: E-1 부하(rate 3↔12, NP 200)는 HE0 값을 그대로 쓴다. λ0은 다른 모델(Nemotron-Nano-9B-v2)·다른 workload(random shape A/B)의 처리율이므로 **E-1 부하 설정의 입력이 아니다**.
  "λ\*\_VESSL 비례 rate" 보조 arm(감사 E-1 초안의 보조안)은 이번 등록에 넣지 않는다 — Zamba2 ShareGPT의 VESSL 용량은 측정된 적이 없다.
- **자리표시자(결과 비의존, λ0 Job 회수 후 追記로 확정)**:
  - `BOOT_S`(현재 300 s): λ0 Job 서버 로그의 boot 시간 분포. 입장 규칙(1,500 s)과 예산표만 바꾼다. 라벨은 바꾸지 않는다.
  - VESSL 노드 스로틀 빈도(`clk_summary.json`): 높으면 블록 = 노드 설계의 노드 효과가 커진다는 공시를 강화한다(설계 변경 없음).
- **E-1이 대신 직접 재는 것**: HI 포화 여부는 λ0 대신 **M1**(cap 48 구속)으로 이 Job 안에서 잰다. VESSL에서 HI가 포화되지 않으면 `UNRESOLVED_CAP48_NOT_BINDING`이다.

### 11-2. 엔진 트리 결합 (★다른 트랙에 미치는 영향)
- λ0 VESSL 러너는 엔진 트리 25항목이 908623과 내용 동일하지 않으면 `ABORT_ENGINE_TREE_DRIFT`다(그 追記 §1-3, §8 표면 10). 이번 엔진 변경(`multiplexing_mixin.py`)이 커밋되면 **그 커밋 이후로 제출하는 λ0 Job(예비 Job 2·재개)은 중단된다**.
  지금 돌고 있는 λ0 Job은 자기 커밋 bundle로 돌기 때문에 영향이 없다. λ0 추가 Job이 필요하면 이번 변경 **이전 커밋**으로 제출하거나 λ0 追記가 필요하다 — 메인 세션·사용자 결정.

### 11-3. 동시 실행
E-1 Job은 λ0 Job과 GPU를 공유하지 않는다(Job마다 전용 A100). 같은 계정의 크레딧만 공유한다. 이 문서 작성 중 λ0 Job에 대한 조회·조작은 하지 않았다.

---

## 12. 감사자가 봐야 할 자유 표면 (자기 공시)

1. **primary 지표 = HI TTFT p50**: p95·mean·goodput을 primary로 했으면 라벨이 달라질 수 있다. 선택 근거는 §4(cliff 없음; 정지에 견고하다는 rev1 근거는 반증됨 — E1C-6). secondary 부호 불일치는 병기 의무.
2. **대조 arm d24 vs d44**: d16(가장 prefill-heavy)을 쓰면 효과가 커질 수 있다. d24는 bind+GATE가 실제로 앉은 자리이고 사용자 지정 최소 집합 안이다.
3. **cap 192**: running 상한이 HI 200요청을 거의 다 받는 값. 다른 값(96 등)은 시험하지 않는다. M2가 비구속을 측정으로 확인한다.
4. **pool 통제 arm을 S24/S44에만**: S34_M 없음(경향 기술 불가).
5. **M1 문턱 0.50 · M2 0.10 · M3 0.90 · pin 0.99**: 결과 보기 전에 고정. M1의 KISTI 근거(0.70)는 로그만 추정치다.
6. **G = ln 1.03, t-CI(df = n−1)**: n = 4–6에서 반보수적일 수 있는 bootstrap 대신 t를 썼다. 정규성 가정은 검증하지 않는다.
7. **블록 = Job**: 노드 효과는 흡수되지만 블록 안 시점 효과(12런 ≈ 3 h)는 무작위 순서로만 다룬다.
8. **bench 시드 = 블록 번호**: 블록 1이 HE0 prompt·도착과 같다. 시드 선택은 결과와 무관하게 기계적.
9. **예비 블록 규칙**: 무효 판정이 기계적이어도, 예비를 쓰는 것 자체가 노드 선택을 바꾼다(무효가 특정 노드 유형에 몰리면 편향). 무효 사유를 전부 보고한다.
10. **CG2 결정론 가정**: §5-3-2.
11. **phase events 시각 정의**: `t_launch`는 decode forward 발행 **직후**(CPU 준비 시간 제외), `prefill_span_end`는 완료를 **관측한** 시각(커널 실제 완료보다 최대 1 iteration 늦음). 동거율 (a)는 호스트 관측값.
12. **관측자 효과 arm이 d34뿐**: §5-4.
13. **추가 캡처 bs를 cap 처치에 포함**: §3-4. 분리하려면 `--cuda-graph-bs`를 고정하는 별도 arm이 필요하다(등록 안 함).
14. **D arm의 FEAS 문턱 cap 상대값**: §6-6.
15. **`--random-seed 1`·green read-out 추가**: 전 arm 대칭, HE0와의 차이(§3-2).
16. **엔진 트리가 HE0 시점과 다름**: legacy loop 경로 바이트는 이번 변경에서 줄 위치 포함 불변이지만, HE0(2026-07) 이후 다른 기본 OFF 추가물이 있다. VESSL 내부 비교에는 무관, HE0 대비 귀속 금지.
17. **Job preflight는 시험 부분집합**: 설치 트리·규칙 코드 시험(`test_phase_events` 전부 + `test_e1_cap_vessl`의 명령줄·헬퍼·분석 기하·라벨 변이·fail-closed 배선)만 Job에서 돈다. 러너 동작 시험(허용 목록·dry run·예산 집행기)은 러너 바이트가 digest로 기록되므로 로컬/CI 회귀에서만 돈다.
18. **서버 로그 파서**: 정규식은 합성 로그 시험 + KISTI HE0 실제 서버 로그 1개(`sgptvsrv_bind_rep41_L3H12_856930.log`, git 밖)에 대한 로컬 대조로만 확인했다. VESSL 로그 형식이 다르면 M3(`full token usage`)·로그 cap-bound가 측정 불가가 된다 — 사용률 줄이 0건이면 `max_full_token_usage = NaN`으로 두고 라벨은 `UNRESOLVED_MANIPULATION_UNMEASURED`(fail-closed). (초안 작성 중 "0건 → 0.0 → KV 비구속으로 통과"하는 구멍을 발견해 이 회차에 막았다.)
19. **실제 GPU 경로 미검증**: 서버 boot·bench·phase events의 실제 방출·CG2 결정론은 GPU에서만 확인된다. 첫 Job(블록 1)이 사실상 그 확인이다. ★rev1.1(R7·E1C-9): 블록 1 이후 rev2는 엔지니어링 실패(boot·CG2·events 0건·배너·pin 측정 불가)로만 발동하고, 발동하면 그 이전 블록은 라벨에서 제외한다. M1–M3·Δ 값을 본 뒤의 설계 변경은 등록 위반이다(§10-6).

---

## 13. 이 회차가 하지 못하는 것

- 게이트 #6, HE0 재판정, 동적 제어 우열, Bullet 재현(E-2), 모델 계열 일반화(E-3)는 범위 밖.
- no-gate 견고성(n ≥ 8)과 856964 유형 서버 정지 원인: 이번 등록 밖(별도 등록 필요). 단 전 런의 서버 로그는 남으므로 정지 사건의 기술 집계는 가능하다.
- 노드 간 산포는 블록 효과로 흡수할 뿐 분리 추정하지 않는다.
- true-dual 경로에서 phase events: 시험 안 됨.

## 14. 로컬 검증 (GPU 0)

- 실행 결과(단위시험·변이·컨테이너 dry-run·회귀)는 이 문서가 아니라 engine-porter 보고에 적는다 — 이 문서의 digest가 실행 기록에 들어가므로(λ0 VESSL 追記 §9와 같은 이유).

---

## 15. 등록 caveat E1C-1…9 (rev1 규칙층 감사 승계, 문자 그대로)

출처: `VERDICT_e1_cap_vessl_rules_rev1_2026-10-04.md`(claims-auditor, `GO-with-caveats`). 결과 문서·정본은 아래를 문자 그대로 승계한다. 이 절과 코드 출력(`ALLOWED_TEXT`/`FORBIDDEN_TEXT`)이 다르면 **이 절이 우선**한다.

- **E1C-1 (기판·범위)** — "E-1의 어떤 라벨·수치도 KISTI HE0(정본 §1-4·§1-7)에서의 후보 (7) 판정을 바꾸지 않는다. KISTI의 'backlog의 cap 48 귀속 NOT-YET-SUPPORTED'는 그대로 남고, E-1은 VESSL 노드 집합에서 같은 서빙 arm으로 시험한 별도 결과로만 등재한다. E-1의 동거율 (a)·cap-bound 값으로 KISTI CORESIDENCY 추정 폭(31/39/57%)이나 C-4 값을 좁히거나 보정하지 않는다."
- **E1C-2 (NOT_CAP_BOUND)** — "NOT_CAP_BOUND는 'cap 192(pool 192)에서도 d24의 HI TTFT p50 불리가 3% 넘게 남았다(부호 유지)'까지만 뜻한다. 'cap을 올려도 불리가 줄지 않았다 / 3% 넘게 줄지 않았다'로 쓰지 않는다. DiD_cap 비유의는 감쇠 부재의 증거가 아니며, DiD_cap 평균과 CI를 반드시 병기한다(사전 예보: 41% 감쇠 참값에서 NOT_CAP_BOUND 30–57%)."
- **E1C-3 (ATTENUATED / VANISHED)** — "CAP_ATTENUATED는 'cap 48→192가 d24 불리를 3% 넘게 줄였다(DiD_cap CI > 0)'까지만 뜻한다. '불리가 남았다'는 Δ192가 sig_pos일 때만 쓰고, 아니면 'cap 192에서의 잔여 불리는 판정되지 않았다(Δ192 CI 병기)'로 쓴다. CAP_BOUND_VANISHED는 n ≤ 6에서 사실상 도달 불가(예보 ≤ 1%)이므로, 그것이 나오지 않았음을 '불리가 사라지지 않았다'로 읽지 않는다."
- **E1C-4 (TTFT 한정 / 정책 금지)** — "모든 E-1 라벨은 HI TTFT p50 부호에 관한 것이다. goodput·ITL·HE0 순위·실무 권고(decode-heavy static)·'버스트에서 prefill-ward가 낫다'로 확장하지 않는다. 이 금지(PREREG §6-4 금지 8)는 `e1_plan.FORBIDDEN_TEXT`에 빠져 있으므로, 이 문장이 코드 출력보다 우선한다."
- **E1C-5 (기전)** — "cap 48→192 처치는 running 상한과 함께 decode 배치 크기, prefill 배치 구성(입장당 new-seq, split forward 횟수), 추가 cudagraph 캡처(bs 56–192)를 움직인다. CAP_BOUND_* 라벨은 cap 노브의 효과이지 '슬롯 회전 기전'을 확인한 것이 아니다."
- **E1C-6 (p50 정지 민감)** — "TTFT p50은 정지에 견고한 지표가 아니다. KISTI에서 859006을 포함하면 Δ48이 0.157에서 0.231로 바뀌고, 859025 d16의 HI p50은 683 ms로 같은 arm의 2095–2225 ms와 다르다. 런별 서버 정지(로그 무음 ≥ 1 s)와 클라이언트 정지 사건은 라벨과 함께 기술 집계로만 병기하며, 사후 제외는 금지한다."
- **E1C-7 (M2 합동 / 비-cap 구속)** — "M2 통과는 S24_C192·S44_C192 합동 중앙값 기준이다. 라벨을 인용할 때 arm별 중앙값을 병기한다. S24_C192 단독 중앙값이 0.10을 넘으면 NOT_CAP_BOUND·ATTENUATED를 'cap 192가 d24에서 여전히 일부 구속' 단서 없이 인용하지 않는다. M2·M3는 토큰 예약(PrefillAdder) 구속을 보지 못한다."
- **E1C-8 (관측자 / CG2)** — "OBSERVER_OK는 관측자 효과가 없다는 증거가 아니다(d34만 측정, CI가 0을 포함할 뿐). CG2 통과는 동시성 1(동거 없음) 경로의 greedy 출력 동치만 보인다. 계측이 동거 경로의 스케줄링을 바꾸지 않는다는 근거는 CPU fake 시험뿐이다."
- **E1C-9 (절차 동결)** — "본 블록 수(6 또는 4), D arm 포함 여부, BOOT_S·RUN_NEED_S 追記는 블록 1 제출 전에 코드 상수로 고정해 REGISTRATION_SHA256에 남긴다. 블록 1 회수 후의 rev2는 엔지니어링 실패(boot·CG2·events 0·banner·pin 측정 불가)로만 발동할 수 있고, 발동하면 그 이전 블록은 라벨에서 제외한다. M1–M3·Δ 값을 보고 설계를 바꾸는 것은 등록 위반이다. UNRESOLVED_*는 후보 (7)의 음성 결과가 아니다."

## 16. 사용자 결정 대기 옵션 (rev1.1에 반영하지 않음 — 채택 시 그 diff만 델타 감사)

| 옵션 | 내용 | 판정 영향(감사 판정서 근거) |
|---|---|---|
| R4 | M2를 S24_C192·S44_C192 **arm별 중앙값의 최댓값** ≤ 0.10으로 | 한 arm(특히 S24_C192)만 cap 192에 구속돼도 합동 중앙값으로 통과해 NOT_CAP_BOUND 쪽으로 기우는 빈틈(E1C-7)을 닫는다 — 그 경우 라벨이 `UNRESOLVED_CAP192_BINDS`로 바뀐다. 편향 크기는 감사도 수치로 못 냄 |
| R6 | M1/M2 입력에 NaN이 섞이면 NaN 수를 병기하고 **과반 NaN이면 unmeasured** | 현재는 부분 NaN을 중앙값이 조용히 제외해 적은 런으로 M1/M2가 통과할 수 있다 — 채택 시 그런 경우가 `UNRESOLVED_MANIPULATION_UNMEASURED`로 바뀐다(측정 실패를 판정으로 읽지 않는 쪽) |
| R8 | NOT_CAP_BOUND에 **동등성 조건**(DiD_cap CI 상한 < ln 1.03) 추가 | NOT_CAP_BOUND가 "감쇠가 3% 미만"까지 뜻하게 되지만 검출력 비용이 크다 — 부분 구속 참값의 상당 부분과 비구속 참값 일부가 `UNRESOLVED_AMBIGUOUS`로 이동(감사 MC: 비구속 참값 NOT_CAP_BOUND σ 0.045 n6 0.97, σ 0.075 n6 0.64에서 추가로 줄어듦) |
| 블록 수 | 본 블록 6 → 4 | ≈ 6.1 GPU-h 절감. 감사 MC: σ 0.075에서 UNRESOLVED ≈ 0.24 → 0.67, 완전 구속 참값 ATTENUATED 0.92 → 0.62(σ 0.045) |
| D arm 제거 | B_C48·B_C192 제외 | 라벨 영향 0, 블록당 ≈ 0.45 GPU-h(6블록 ≈ 2.7 GPU-h) 절감. C-1 VESSL 입력·게이트 5 동적 기술 상실, 러너 `ABORT_ARM_COUNT 12` 수정 필요 |

어느 것을 채택하든 **블록 1 제출 전** 코드 상수로 고정한다(E1C-9, §10-6).
