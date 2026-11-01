# STOCK-BASE Phase B 측정 하네스 — 설계 보고 (2026-11-01, engine-porter 작성 요지를 main 세션이 전사; CPU 전용, 미봉인)

> 지위: 하네스 작성 완료·CPU 테스트 통과. **실제 서버 실행 0, 성능 수치 0, 판정 0.** 봉인 전 사용자 결정 2건(§3 vLLM 배치, §4 O-6 J3 시간) + 하네스 감사 1회 필요.
> PREREG `PREREG_STOCK_BASE_rev1_DRAFT_2026-11-02.md`. 엔진 digest `2836112b…`(`ENGINE_DIGEST.txt`, HEAD 트리 scratch 설치로 재계산 일치).

## 1. 파일 (`results/stock_baseline/`, 접두사 `sb_`)
| 파일 | 역할 |
|---|---|
| `sb_spec.py` | arm 7개(엔진·트리 pristine/patched/vllm·이미지 키·argv·env), 셀(사다리 5.516×1.15^k 10단 + 확장 k=10..13, 지속 {1.0, 0.8}×λ\*·600 s·seed 201(+0.5 조건부), R₁/R₂ 0.5/0.8·λ\*_ref·200요청·seed 1–4, trace LO 0.2/HI 0.8·200/구간·3라운드, W3 사다리 0.1×1.5^k 6단, R₃ 0.8·λ\*_W3·96요청, refcell, smoke 20, 계측 60·seed 301), Job plan J1(smoke+사다리)·J1b(지속+refcell)·J2(ABBA R₁/R₂+계측)·J3(ABBA trace+W3)·J1-ext, 시간 모형·하드캡 ×1.25·`plan_problems`. plan sha는 기계 무관(PD-mux config 경로 `{ENG}`) |
| `sb_runner.py` | SGLang boot = `engine_tree.py boot --stamp boots/<id>/tree_<arm>.json -- python -m sglang.launch_server … --port P`; env는 Job 배관 `PDMUX_*`만 남기고 제거, arm별 `PDMUX_ENGINE_TREE`. 순서: warm-up 60 s → smoke → 사다리 → 지속(두 점 실패 시 0.5점) → refcell → ABBA 측정 → (J3) floor → W3 사다리 → R₃(BRACKETED 아니면 건너뜀 rc 17) → 계측 boot(원시 gzip). run 스탬프 `job_name`·`gpu_uuid`·`engine_tree`·트리 스탬프·trace hash·`t_load_start/end`(epoch·monotonic); 요청별 `t_sent_mono`·`token_t_mono`·`t_finished_mono`; SG-mix `mixed_forward_proxy`; `server_facts.json`(backend·mem fraction) |
| `sb_loadgen.py` | SGLang = 봉인 `ph_loadgen.replay_natural(abs_mono=True)` 바이트 그대로; vLLM = `/v1/completions` 쌍둥이 클라이언트(같은 스탬프 지점·레코드 키·오류 분류, 토큰 수 `usage.completion_tokens`); 정의 일치 smoke = 순차 요청 + `/metrics` TTFT 히스토그램 Δsum(Δcount=1) |
| `sb_analyze.py` | addendum(λ\*_burst·refcell_rate·λ\*_sustained·λ\*_ref=min + scope/missing), §4 라벨·K3, 유효성 사슬(trace hash·cliff·Job/GPU·§1 트리 스탬프·플라시보·smoke·요청 실패 0), 태그 CELL_OUT_OF_BAND·SUBSTRATE_SHIFT/UNCHECKED·MIX_NOT_FIRED·seed-8 재실행, r_ser/r_mix(부하 창을 프로브 boot `(t_realtime, t_host)`로 사상 → `pacing_ratios`), PM r_span `pc_reduce_windowed`(창 필터 새 함수), P-1…P-5 채점, "측정하지 않은 것" `not_measured_md` |
| `sb_mutants.py` | 변이 42개 |
| `sb_digest.py` · `sb_job.sh` | HARNESS_DIGEST(자기 파일 + 재사용 봉인 파일 21개 + plan sha), `ENGINE_DIGEST` ≠ `2836112b…`면 make 거부; job.sh는 미봉인·Job 이름·엔진 digest(rc 13)·PRESERVE_STATUS≠OK(rc 15)·plan/addendum(rc 16) 거부, raw export 후 sha 검증 |
| `tests/test_stock_baseline_harness.py` | 56 tests + 변이 래퍼(`SB_RUN_MUTANTS=1`) |

## 2. 테스트 (CPU)
- `test_stock_baseline_harness` 56 OK(1 skip). 변이 42/42 검출(1차에 M03·M04·M38·M39·M41 생존 → 테스트 보강 후 전부 검출). 과제 지정 6종: duration max() M01 · 창 밖 혼입 M02–M04 · 다른 Job 쌍 M05 · engine_tree 불일치 M06(+M07–M09) · 동률 높은 rate M10 · 플라시보 완화 M11–M14.
- 전체 회귀 1938 tests, F41/E21 — 전부 기존 환경 실패(`sglang` import·`PDMUX_ROOT`), 새 테스트 실패 0. 기준 실행과 직접 비교는 안 함.
- `check_citation_stops` OK. **GPU 검증 0.**

## 3. vLLM 배치 — 사용자 결정 (미구현, `VLLM_PLACEMENT="PENDING_USER_DECISION"`, runner `ABORT_VLLM_PLACEMENT`)
제약: Job 하나 = 이미지 하나(`launch.sh -i`). SGLang 이미지 py3.14·torch 2.9.1+cu130, vLLM 이미지 py3.12.3·torch 2.13.0·vllm 0.31.0·flashinfer 0.7.0.post1. PREREG §4 ③ "쌍 = 같은 Job·같은 GPU".

| 선택지 | 장점 | 단점 |
|---|---|---|
| (a) VL-stock 별도 Job | 두 이미지 그대로 | Q3·Q4 쌍이 Job·GPU를 가로지름 → 현재 분석기 INVALID(fail-closed). Job 간 규칙 새 등록 = 설계 변경. λ\*_ref(min)도 Job 혼합. vLLM용 job_entry 경로 신설 |
| (b) SGLang Job 안 격리 venv | 쌍이 같은 Job | 도는 vLLM이 push한 이미지가 아님 → freeze sha boot마다 검사. 인터프리터·site-packages·sitecustomize·PYTHONPATH·JIT 캐시 분리. flashinfer 0.7 JIT용 CUDA toolkit이 SGLang 베이스에 있는지 미확인 |
| **(b′) (b)를 새 공개 이미지에 굽기** (`sglang-runtime` + `/opt/vllm-venv`) | (b) + 출처 digest 고정. 엔진 코드 없음(규약 OK) | SG arm 이미지 digest 변경 → 기존 수치와 같은 기판 주장은 SGLang site-packages freeze sha 동일성 검사 추가 시만 |
| (c) vLLM 이미지 안에 SGLang | — | SG arm 베이스·CUDA·파이썬 전부 변경, PM-ref·SG-off 다리 역할 깨짐 |

**권고 (b′), 차선 (b).** 어느 쪽이든 bring-up (a)–(e)는 그 환경에서 다시. smoke 통과 전까지 엔진 간 쌍은 INVALID. **J1b 전에 결정해야 한다**(VL이 나중에 들어오면 λ\*_ref가 바뀌어 J2·J3 rate 무효).

## 4. PREREG와 다르게 한 것 (편차 6) / 내부 불일치 해소 (13)
편차: **D-1** SG-stock 계측 boot는 pristine에 프로브가 없어 patched 트리·OFF argv로(Q6은 "수정 트리·OFF" 범위) · **D-2** 서버 first-token 시각이 없어 순차 요청 + `/metrics` Δ(SGLang smoke에만 `--enable-metrics`) · **D-3** 계측 boot는 R₂가 J1b 뒤 확정되므로 J2로 이동 · **D-4** refcell rate = addendum `refcell_rate`(기본 λ\*_SG-stock 버스트 rung·seed 201·120요청, Job 1에 같은 셀 없으면 `SUBSTRATE_UNCHECKED`; O-1) · **D-5** 지속 판정의 분모 = 창별 관측 도착 수(명목 rate는 Poisson 잡음으로 건강한 run도 탈락 확률 ≈0.3; 명목값 병기) · **D-6** floor·W3 사다리는 J3 안(시작 refcell boot에서 floor → SG-stock W3 사다리 → R₃).
불일치 해소: O-2 사다리 arm = §7의 5개 · O-3 W3 arm = SG-stock·SG-rep·PM-ref·VL-stock(확인 필요) · O-4 Job 4개 · **O-6(★) J3 ≈ 670 min(하드캡 840)** — PREREG가 trace run을 HE0 기준 7.5 min으로 셌으나 HI 0.8·LO 0.2·λ\*_ref에서는 run당 ≈16 min, W3 사다리 ≈64 min·ABBA 이중 boot도 미산정. J1 160·J1b 166·J2 181 min은 상한 안. **J3 분할·축소는 사용자 결정**(VL 포함 시 Job당 +30–95 min) · O-7 P-4 셀 = 측정 bs 중앙값 최근접 격자·ctx 2048·L ±25%·단일 요청 · O-8 Q6 "512/2k/8k 각 ≥10"은 W1(문맥 ≤4000)로 불가, 8k는 0 → `Q6_FEW_SPANS` 예상 · O-9 MIX 발화 수는 서버 로그 proxy(GPU 미확인) · O-10 stock은 SGLang 기본값(로그 채취), PM-ref는 봉인 argv(mem 0.6·flashinfer·seed 1·mamba 48·chunked −1·overlap OFF) → Q2 = "구성 묶음 전체 대 기본값" · O-11 vLLM `--host --trust-remote-code --dtype bfloat16 --max-model-len 20480`, `--max-prefill-tokens` 대응 없음 · O-12 클라이언트는 PREREG §5-0(SGLang `/generate`, vLLM `/v1/completions`) · O-13 동률 = (rate, seed) 점 접두(M10 고정) · O-14 ABBA round 2 = round 1 역순, arm당 boot 2회.

## 5. 남은 위험
GPU 미확인: pristine 보존(`PDMUX_REQUIRE_PRISTINE=1` 없으면 rc 15 거부)·EXTEND 이벤트·overlap 단일 stream·계측 ON 출력 불변·관측자 비용·SG-mix proxy 정규식·vLLM 로그 패턴·두 서버 TTFT 메트릭 이름 · J3 시간(O-6, 지금 상태로 제출 불가) · λ\*_ref 범위(VL 미정 동안 SGLang만) · Q6 범위(D-1) · Q2 범위(O-10) · 하네스 감사 미실시(`HARNESS_DIGEST.txt`는 봉인 때 1회).
correctness gate 단언 범위 = CPU 단위 테스트·변이 검출까지. GPU correctness·출력 동치·observer effect 측정 0.
