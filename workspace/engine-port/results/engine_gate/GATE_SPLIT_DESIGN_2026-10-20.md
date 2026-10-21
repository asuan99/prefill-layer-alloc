# 엔진 묶음 분할 GPU gate 설계 (G-α → G-β → G-γA ∥ G-γF) — 2026-10-20, engine-porter

> 지위: **설계·등록 문서**다. GPU 실행 0회, 데이터 0. 여기 적힌 문턱·사다리·순서는 **Job 데이터를 보기 전에** 등록한다.
> 근거: 사용자 결정 16(main `workspace/engine-port/results/anchor_audit/DECISION_headline_G_shapeC_2026-10-19.md` O-2·gate 분할),
> SPEC v0 §9(main `results/handoff_review_2026-10-15/SPEC_common_measurement_v0_2026-10-15.md`) gate (a)–(d),
> PREREG cost-model rev2 §8-1(gate 요구 목록)·§1-7(vanilla 경로 엔진 측 decode 시간), rev1 감사 MAJOR 7.
> 이 문서는 "gate가 통과했다"고 말하지 않는다. 통과 판정은 Job 회수 후 result-analyst, 반증은 claims-auditor 몫이다.

## 0. 대상 엔진 묶음과 digest

| 항목 | 값 |
|---|---|
| 묶음 | I-5 동기화 + Triton causal\_conv1d int64(두 곳) + **I-4·I-6**(`src/multiplex/decode_probe.py` + `src/patches/decode_probe_scheduler_hook.patch`) |
| manifest | 30항목(28–29 = I-5·conv, 30 = decode\_probe). 29항목 digest `2b1ab054…`는 **배포·봉인된 적 없음**, 폐기 |
| 예상 ENGINE\_DIGEST (pathless-v1, 30항목) | `env/dev_tree_edits.md` 항목 35에 기록(커밋 바이트 기준). gate Job은 install 직후 이 값과 다르면 `ABORT_ENGINE_DIGEST` |
| 반증 arm digest(int64 패치만 역적용) | 같은 항목 35에 기록. 역적용 후 conv 파일 바이트 = pristine upstream |
| 스코어러 | `scripts/vessl/engine_gate_checks.py`, 등록 상수 `REG` sha256 = 항목 35에 기록(`--print-reg`) |

## 1. 공통 조건

- 기판: VESSL A100 SXM 80GB, **Job마다 GPU 1장**(동시 사용 1 = 할당 1). 이미지 digest는 `scripts/vessl/vessl.env` 값. 엔진은 `install_runtime.sh`로 설치.
- 모델: Nemotron-Nano-9B-v2-Base(rev `dc0661c…`), 대조 Llama-3.1-8B base(rev `d04e592b…`, G-α만).
- 공통 노브: `--mem-fraction-static 0.6`(결정 13), radix OFF, cudagraph ON(G-α eager 대조 boot 하나만 예외), `--disable-piecewise-cuda-graph`, server seed 1, cap 48(`--max-running-requests 48 --max-mamba-cache-size 48`, Nano), greedy, `ignore_eos`는 단계별 명시.
- arm: `agnostic` = PD-mux `benchmarks/configs/pdmux_agnostic_manual.yml`. `fused_sym` = vanilla + `--chunked-prefill-size -1 --disable-overlap-schedule`(M3 PREREG §2). Llama = fused\_sym 노브 + `--attention-backend flashinfer`.
- **계측 ON 수준**(판정 수준, 이후 비용 모델 Job과 같게 둔다): `PDMUX_ADMISSION_PATH`(I-5), `PDMUX_DECODE_PROBE_PATH`(I-4·I-6, `PDMUX_DECODE_PROBE_GPU_EVENTS` 미설정 = 켬), I-1 외부 수집. PD-mux arm은 추가로 `PDMUX_TELEMETRY_PATH` + `PDMUX_PHASE_EVENTS=1`(이 둘은 mixin 전용이라 fused arm에는 없다 — 구조적 비대칭, 공시).
- **계측 OFF 수준**(최소): 클라이언트 기록 + I-1만. 위 `PDMUX_*` 경로 env 전부 미설정, holb/chunk probe 미설정.
- 모든 Job: install → digest 검사 → boot마다 `alive_after_<step>.json`(MG F2 형식) → teardown 후 GPU memory.used < 1 GiB 확인 → 다음 boot. 끝에 SHA256SUMS. 회수는 `fetch.sh`(DONE + sha256)가 완료.
- **Job 사이에서 수준을 비교하지 않는다.** 모든 판정은 같은 Job 안의 짝·대조로 끝난다.
- **SPEC §9 승계 (감사 r1 MAJOR 2, 2026-10-20)**: "판정 Job마다 ON/OFF 짝"을 따른다 — 모든 판정 Job(G-α, G-β, G-γA, G-γF)은 자기 Job 안에서 ON 대 OFF(또는 패치 대 역적용) 짝으로 끝난다. G-γ의 "λ 짝의 양"은 **단일 seed HI achieved**이며 SPEC/MG의 λ\*_thr(4 seed 중앙값)와 **다른 양**이다. 이 gate의 TOST 값을 MG λ\*와 같은 이름으로 인용하지 않는다.
- **하네스 수리 (감사 r1, 2026-10-20)**: ① OFF boot은 probe 파일이 없고 로그에 "decode probe ENABLED"/"admission probe ENABLED"가 없어야 하며(ON은 둘 다 있어야) 서버 인터프리터의 `sglang.__file__`이 설치 트리 아래여야 한다(`check_instrumentation`). ② fused I-4 창 = 고립 decode 단계의 첫 decode forward부터 끝 decode forward까지(`check_i4_fused(..., window)`, 창 없으면 UNRESOLVED). ③ TOST 3분기: CI⊂±δ → PASS, CI가 0을 배제 → OBSERVER_EFFECT, CI가 0을 포함하면서 δ 초과 → UNRESOLVED_POWER. α 분할: 초기 look(짝 ≤10)과 사전 등록 1회 확장 look(≤16)이 각각 단측 α=0.025(95% 양측 CI) — **기존 90% CI 단일 look을 대체**한다. 확장은 별도 Job(`PDMUX_GATE_GAMMA_PHASE=ext`, 짝 11–16, seed 7007·8008·9009·1111·2222·3333 [제안])이며 `pool-gamma`로 합산한다. 사후 짝 추가 금지. ④ 무효 짝이 ON 쪽에서 OFF 쪽보다 2개 이상 많으면 OBSERVER_EFFECT. ⑤ β-0에 서버 없는 green-ctx event 보정 단위(`green_event_calib_unit.py`: [sleep→matmul] 대 [matmul→sleep] × green 대 기본 stream, 사전 예측 P1–P4는 파일 머리에 고정; **데이터를 본 뒤 허용 범위를 고치지 않으며 FAIL이면 사용자 결정으로 넘긴다**). ⑥ (4g) 계단 bs 1..40 전부 통과, cap+1(49) 동시 요청에서 I-6 최대 bs=48·I-5 `decode_bs_hist` 최대=48·I-5가 cap 사유 stop을 기록(`check_i5_i6`). ⑦ boot마다 Triton 캐시 key(dir·항목 수·항목명 hash·conv 소스 sha)를 `boot.json`에, Job마다 `substrate.json`을 결과 디렉터리에 둔다. ⑧ 재부팅 실패·메모리 미해제는 `HARNESS_UNRESOLVED`(FAIL로 세지 않음). β-2의 "pristine이 깨졌다" = 서버 사망 OR 패치 트리 출력과 불일치. ⑨ ≤64k 출력 일치는 "sanity"(wrap 지점 ≈94.8k 아래)로만 표기한다.
- **등록 상수 개정**: 위 ③④⑥으로 `engine_gate_checks.REG`가 바뀌어 sha256이 `f3310545…`에서 `1c20b953d57313d9451bd75351f410a2f22d90fcb765be88372f90d091edef26`로 바뀐다(`--print-reg`). `engine_gate_checks.py`는 sync manifest 항목이 아니므로 ENGINE_DIGEST는 불변.

## 2. 단계 표

| Job | GPU | 예상 시간 [유도] | 하드캡 | 사용률(할당 GPU 중 사용 / 부하 구간 비율) | PASS 조건(요약) | 다음 단계 조건 |
|---|---|---|---|---|---|---|
| **G-α 정확성** (의도적 crash 없음) | A100 1 | 약 1.2 h | 2.0 h | 1/1, 약 40% | (a) 순차 greedy 출력 ON=OFF 완전 일치(3 arm) + 동시 패스 MG G1(ctl 대비) 비FAIL · (b) int64 전후 ≤64k 출력 완전 일치(agnostic·fused\_sym) · 모든 ON boot에서 I-4·I-5·I-6 방출(vanilla·Llama 포함) · I-6 패딩 감사 PASS · eager 대조 PASS · (c)-I4 PD-mux·fused 각 PASS(host 변이 FAIL 확인) · digest 3회 검사 일치 | 전부 PASS → G-β 제출 |
| **G-β 반증·메모리** (의도적 crash 포함) | A100 1 | 약 1.0 h | 2.0 h | 1/1, 약 50% | ③ 단위 시험 표(§4) 전부 예측대로 · ① 패치 후 fused\_sym·agnostic 98,304 통과 · ② 역적용 tree에서 95,100 서버 사망 2/2(causal\_conv1d 서명) + 재적용 후 95,100 통과 · digest 3회 일치. (e) 126,976 OOM은 FAIL이 아니라 `T2EXT_OOM_L126976` 라벨 | ①②③ PASS → G-γA·G-γF **동시** 제출 |
| **G-γA 관측자 TOST, agnostic 계열** | A100 1 | 약 3.4 h | 4.5 h | 1/1, 약 60% | ON/OFF 짝 ≥ 8개 유효, λ\*\_thr log 비와 ITL p95 log 비 **둘 다** 90% CI ⊂ (−ln 1.015, +ln 1.015) | 두 계열 모두 PASS → 묶음 gate PASS(digest 봉인 가능) |
| **G-γF 관측자 TOST, fused\_sym 계열** | A100 1 | 약 3.4 h | 4.5 h | 1/1, 약 60% | 같은 규칙, 계열 독립 판정 | 위와 같음 |

- 총 예상 **약 9.0 GPU-h**, 하드캡 합 13.0 GPU-h. 모든 Job ≤ 4.5 h 하드캡(목표 6 h 이하, 2026-10-18 약 9.5 h 외부 종료 선례).
- 단일 Job 초안(약 8.5 h)보다 기대 GPU-h가 조금 크다: Job마다 설치·preflight(약 10–15분)가 반복되고, Llama boot 2개(MAJOR 7 대응)와 eager 대조 boot가 추가되었다.
- 실패 처리: 어느 Job이든 FAIL이면 다음 Job을 제출하지 않고 engine-porter로 돌아간다. 수정으로 digest가 바뀌면 **G-α부터 다시** 한다. `UNRESOLVED`(비판별·표본 부족)는 PASS가 아니다. 같은 Job 재제출은 사용자 결정.
- 시간 근거: boot 약 120 s(J1 실측 71–97 s에 여유), 셀 170 s + drain 25 s + 셀 간격 24 s(`results/multigpu/mg_const.py` EST 값), 장문 TTFT(128k 프로브 기술값: 64k 약 8 s, 98k 약 14 s, 124k 약 18 s — 예산 입력으로만 사용).

## 3. G-α 정확성 (boot 9개, crash 없음)

| boot | arm·계측 | 단계 |
|---|---|---|
| B1 | Nano agnostic OFF | s-pass(64 prompt, max\_new 128, 동시 1) · c-pass 2회(동시 32, c와 c′ = ctl) · (b) 장문 단일 L ∈ {4,096, 16,384, 32,768, 65,536} × 2 합성 입력(longctx 헬퍼 `synth_ids`, 같은 seed), max\_new 64 |
| B2 | Nano agnostic ON | s-pass · c-pass · **고립 decode**(48 요청 입력 256·출력 1,024 `ignore_eos`, 동시 투입) · **계단**(40 요청, 요청 k의 출력 = 16·k 토큰, `ignore_eos` → 실행 bs가 40→1을 모두 지난다) · green readout(실현 SM 로그, rev2 §1-8) |
| B3 / B4 | Nano fused\_sym OFF / ON | B1 / B2와 같은 단계(fused\_sym은 overlap OFF라 `event_loop_normal`) |
| B5 / B6 | Llama fused\_sym OFF / ON | s-pass·c-pass 2회 / s-pass·c-pass·고립 decode·계단 |
| B7 | Nano fused\_sym ON + `--disable-cuda-graph` | 계단 · 고립 decode(I-6 eager 대조) |
| — | int64 패치만 역적용 → digest = 반증 arm 값 확인 | |
| B8 / B9 | Nano agnostic OFF / fused\_sym OFF, **역적용 tree** | (b) 장문 단일(B1과 같은 입력) |
| — | 재적용 → digest = 기준 값 확인 | |

판정(스코어러 함수 이름):
1. **(a) 출력 일치**: arm마다 s-pass ON 대 OFF `outputs_identical` = PASS(64/64 완전 일치). c-pass는 `m2g_correctness_gate.score_g1`(m·q, ctl = 같은 OFF boot의 c 대 c′) — FAIL이면 G-α FAIL, `UNINFORMATIVE`는 기록만.
2. **(b) int64 전후**: B1↔B8, B3↔B9의 장문 출력 토큰 `outputs_identical` PASS. 출력 logprob를 받을 수 있으면 bit 일치도 기록(하네스 확인 필요, §8).
3. **방출**(rev2 §1-7, MAJOR 7): B2·B4·B6·B7 각각 `check_emission` PASS — decode forward 기록 ≥ 50, I-4 해소 비율 ≥ 0.99, I-5 admission window 존재. **B4·B6(vanilla, Llama 포함)에서 미방출이면 `NOT_COLLECTED` = G-α FAIL.**
4. **I-6**: B2·B4·B6 계단 구간 `check_i6` PASS(패딩된 hit ≥ 1, hit은 부팅 capture 목록 bisect 값과 100% 일치, eager는 무패딩). B7은 `check_i6(..., expect_all_eager_reason="graph_runner_absent")` PASS.
5. **(c)-I4**(변이 시험): B2 고립 decode 창 `check_i4_pdmux`(`decode_iteration`의 t\_sync와 대조, 비 ∈ [0.90, 1.00], 기록별 포함 위반 0, n ≥ 200), B4·B6 `check_i4_fused`(다음 forward 시작까지 구간, 비 ∈ [0.75, 1.00]). **host 시간 변이**(= `decode_fwd_host_ms`로 바꾼 기록, 변이 probe가 낼 값과 같다)가 같은 데이터에서 FAIL해야 PASS이고, 변이도 통과하면 `UNRESOLVED`(비판별) → G-α PASS 아님.
6. **(d) 스키마**: 모든 ON 기록 `validate_record` 오류 0, `record_schema` = `pdmux.decode_probe/v1`·`pdmux.admission/v1`.
7. 생존: 모든 boot 모든 단계 직후 alive. 비정상 종료 1회라도 → FAIL.

## 4. G-β 반증·메모리 (의도적 crash 포함)

**β-0 단위 시험**(★판정 기준은 §10으로 개정됨: 출력 sha vs 원본 contig, TP2 폭 offset 5,120·dim 6,144; 아래 표의 'MATCH/MISMATCH'는 그 기준으로 읽는다)(서버 없음, 사례마다 별도 프로세스 — illegal memory access가 컨텍스트를 오염시키므로): `scripts/vessl/conv1d_int64_unit.py`. 커널 사본 3개 = 패치본(설치 tree), pristine(역적용 사본), `chunk_offset`만 고친 변이본.

| 커널 | layout(stride) | T | 반복 | 예측(등록) |
|---|---|---|---|---|
| 패치본 | view 22,656 | 12,288 · 94,784 · 94,800 · 95,100 · 126,976 | 경계 2회씩 | 전부 MATCH |
| 패치본 | view 11,328(TP2 폭, rev2 §5-3 (가)) | 95,100 · 126,976 · 189,000 · 190,500 | 1 | 전부 MATCH |
| 패치본 | contig | 95,100 | 1 | MATCH |
| pristine | view 22,656 | 12,288 · 94,784(×2) | | MATCH |
| pristine | view 22,656 | 94,800(×2) · 95,100(×2) | | crash 또는 MISMATCH |
| pristine | view 11,328 | 95,100 · 126,976 · 189,000 → MATCH / 190,500 → crash 또는 MISMATCH | 1 | 경계 189,572 [산술] |
| pristine | contig | 95,100 | 1 | MATCH, 출력 digest = 패치본 view와 같음 |
| co-only 변이 | view 22,656 | 94,790 · 95,100 | 1 | crash 또는 state MISMATCH |

**β-1 ① 패치 후 장문**(기준 tree): Nano fused\_sym boot → 단일 요청 98,304 ×2, 126,976 ×2, 95,100 ×1. Nano agnostic boot → 98,304 ×2, 126,976 ×2. 출력 비공백 + illegal access 서명 없음. **(e) 메모리**: NVML 100 ms 샘플러의 최대 memory.used, 부팅 배너의 활성화 여유. 126,976에서 OOM이면 `T2EXT_OOM_L126976`(그 arm의 T2-ext 126,976 제외, 패치 판정과 무관). 98,304 OOM도 같은 방식으로 라벨.

**β-2 ② 역적용 crash 루프**: int64 패치만 역적용 → digest 확인 → fused\_sym boot #1: 94,500 (OK 예측) → 95,100 (서버 사망 예측) → 사망 감지(poll returncode 또는 health 실패) → 로그 서명 `illegal memory access` + `causal_conv1d` 확인 → process tree kill → GPU memory.used < 1 GiB 대기(≤120 s) → boot #2 `CUDA_LAUNCH_BLOCKING=1`: 95,100 → 사망 + traceback이 `causal_conv1d_triton.py`를 가리킴 → teardown → 재적용 → digest = 기준 확인 → fused\_sym boot #3: 95,100 OK.

PASS: β-0 표 전부 예측대로 · ① 98,304 전부 통과(두 arm) · ② 사망 2/2 + 서명 2/2 + 재적용 후 통과 · digest 3회 일치. 사망 대기 하드 타임아웃(boot당 15분) 초과는 `UNRESOLVED`.

## 5. G-γ 관측자 TOST (계열별 Job 2개, 병렬 가능)

- 계열: G-γA = Nano agnostic, G-γF = Nano fused\_sym. TOST는 계열마다 판정한다. 두 Job의 값은 서로 비교하지 않는다.
- **짝 10개**(n ≥ 8 필요, 2개 여유). 짝 i = ON instance + OFF instance(각각 별도 boot, 계측 env가 boot 시점이라서). 짝 안 순서는 i 홀수 = ON 먼저, 짝수 = OFF 먼저(ABBA). 짝 i의 loadgen seed(HI·LO 공통, ON·OFF 공통) = (4386, 4162, 251, 2630, 1001, 2002, 3003, 4004, 5005, 6006)[i] [제안].
- **사다리 등록**(데이터 전): shape A(입력 256·출력 512 고정, `ignore_eos`), cap 48, mem 0.6. **HI = 1.6 × 2.99 = 4.784 req/s**(요청 수 810), **LO = 0.60 × 2.99 = 1.794 req/s**(요청 수 300), 셀 170 s. 두 계열 같은 값이며 데이터에 의존하지 않는다(M3의 "LO = 0.6 × fused HI achieved" 규칙은 쓰지 않는다). 근거: J1(N=1, mem 0.82) shape A knee fused\_sym 3.100·agnostic 3.117 → HI/λ\* 약 1.54, LO/λ\* 약 0.58 [유도]. cap 48·shape A에서 KV 수요(48 × 768 토큰)는 mem 0.6 풀보다 훨씬 작아 knee가 mem에 거의 무관하다는 가정 [유도, 미확인].
- instance당: boot → warm-up 8요청 → HI 셀 → LO 셀 → teardown. λ\*\_thr = HI achieved(seed 1개, `KNEE_BRACKETED` = HI g ≥ 0.10 ∧ LO g ≤ 0.05, MG §4-2 g 정의). ITL p95 = LO 셀 요청 전체의 token 간격 풀 p95(클라이언트).
- 짝 무효: 어느 instance든 `KNEE_BRACKETED` 아님, 클라이언트 오류 > 1%, 서버 측 오류 > 1%, 비정상 종료. 무효 짝은 건너뛰지 않고 제외만 한다(10개 중 유효 ≥ 8).
- 판정: `tost_log_ratio` — log(ON/OFF) 짝 평균의 90% t-CI(df = n−1)가 (−ln 1.015, +ln 1.015) 안이면 PASS, 아니면 `OBSERVER_EFFECT`(SPEC 문언), 유효 짝 < 8이면 `UNRESOLVED`. λ\*\_thr와 ITL p95 **둘 다** PASS여야 계열 PASS.
- 검정력 공시: 짝 log 비 SD 1%면 n=8 CI 반폭 0.67%(통과 가능), SD 2%면 1.34%로 평균이 거의 0이어야 통과 [유도]. J1 boot 내 seed 간 λ\* CV 0.02–1.2%, boot 간 SD는 shape A fused\_sym에서 CV 1.05%(Job 간) [측정, `ANALYSIS_J1_N1_2026-10-19.md`]. ITL p95의 짝 SD는 미측정 — 통과 실패가 "효과"가 아니라 "분산"일 수 있다(§8).
- K4(SPEC §9) 승계: 관측자 TOST 통과는 G에 arm 비대칭 편향을 최대 ±3%(log)까지 허용한다.

## 6. 기존 산출물 재사용

| 산출물 | 위치 | 재사용 | 쓰는 곳 | 필요한 수정·추가 |
|---|---|---|---|---|
| 128k 프로브 스크립트 | `scripts/vessl/longctx_probe.sh` + `longctx_probe_helper.py` | **부분** | G-α (b) 장문, G-β ①② | `synth_ids`(정확 길이 합성)·`stream_generate`·`log_signatures`/`sig_delta`(crash 서명)·`server_alive`·teardown memory 검사·boot 배너 파서(`parse_banner`, (e))는 그대로. 추가: digest 검사·int64 역적용/재적용 단계, "사망이 예측"인 단계의 판정 반전, 재부팅 루프, NVML 100 ms 최대치 집계 |
| MG G1 비교기 | `scripts/multigpu/m2g_correctness_gate.py` (`g1_metrics`, `score_g1`, `make_alive_record`) | **그대로**(c-pass) | G-α (a) | ctl을 "N=1 비교" 대신 같은 OFF boot의 c 대 c′로 넣는다(함수 인자만 다름). 순차 s-pass 완전 일치는 `engine_gate_checks.outputs_identical` |
| conv1d 단위 하네스 | `scripts/vessl/conv1d_int64_unit.py` | **그대로**(사례 1개) | G-β β-0 | `--width-inproj 11328` 지원 확인됨(인자). **사례 오케스트레이터(사례별 subprocess, crash = 비0 종료/신호 기록, 커널 사본 3개 생성)는 없다 → 신규 작성 필요** |
| MG loadgen·λ\* 라벨 | `scripts/multigpu/mg_loadgen.py`, `mg_const.py`(g 정의, HI/LO 요청 수 규칙) | **부분** | G-γ | 고정 rate(HI 4.784·LO 1.794) 주입, instance 짝 순서·seed 목록, OFF/ON env 전환 |
| TOST | `results/p1_gates/gate2/g2_analyze.py::tost_equivalence` | 참조만 | G-γ | 이 gate는 `engine_gate_checks.tost_log_ratio`(같은 90% CI 규칙, 자기완결)를 쓴다 |
| gate 스코어러 | `scripts/vessl/engine_gate_checks.py` | 신규(CPU 시험 21개) | 전 단계 | Job 하네스(runner·launcher)는 미작성 → experiment-runner |

## 7. 비판별·변이 원칙 요약 (교훈 53)

- I-4 host 변이: CPU 표에서 FAIL(gpu\_ms 1.5 vs 20.0), GPU에서는 같은 데이터의 `decode_fwd_host_ms` 치환으로 평가. 변이가 통과하는 데이터면 `UNRESOLVED`.
- I-6 `bs_padded := bs_real` 변이: CPU 표 FAIL, GPU 계단 구간에서 패딩된 hit이 0이면 `UNRESOLVED`.
- 그 밖의 CPU 변이(시작 이벤트를 forward 뒤에 기록, 해소 시 blocking, hit을 bs ≤ max\_bs로 판단, mixed 누락, eager 패딩)는 `tests/test_decode_probe.py`에서 전부 FAIL.
- int64: 역적용 tree(pristine 바이트)에서 95,100이 실제로 죽어야 패치 PASS가 의미를 가진다(β-2).

## 8. 미해결 (설계 범위 밖·감사 필요)

1. Job 하네스(G-α/β/γ runner, β-0 사례 오케스트레이터) 미작성 — experiment-runner. 스코어러만 있다.
2. (c)-I4 band [0.90, 1.00]·[0.75, 1.00]은 SPEC의 "±5%"(I-3 prefill span 문언)에서 벗어난 [제안]이다. decode 괄호에는 sampling·host 처리가 들어 있어서다. claims-auditor 확인 필요.
3. λ\*\_thr를 seed 1개 HI achieved로 잰다(MG는 seed 4개 중앙값). TOST 짝이 반복 단위라 seed 수를 줄였다 — 감사 필요.
4. ITL p95 짝 분산 미측정 → G-γ가 `OBSERVER_EFFECT`로 끝나도 효과인지 분산인지 못 가를 수 있다. 짝 10개로도 부족하면 짝 추가는 사용자 결정.
5. PHASE\_EVENTS가 PD-mux 전용이라 ON 수준이 arm 사이에 구조적으로 다르다(SPEC 정의대로 공시만).
6. (b) logprob bit 일치는 하네스가 `return_logprob`를 받는지 확인 후 결정. 토큰 일치만이면 "토큰 동일"까지만 쓴다.
7. green-context 스트림 위의 timing event 정확성은 CPU로 보일 수 없다 — (c)-I4 PD-mux 셀이 유일한 확인이다.
8. TP2(stride 11,328) 서빙 96k 점은 이 gate에 없다(단위 시험만). PREREG rev2 §5-3 (가).

## 9. 개정 r2 (2026-10-20, G-α 1회차 데이터를 본 뒤의 수정 — 재감사 대상)

**공시: 이 절의 수정은 G-α 1회차 데이터(`pdmux-engine-gate-alpha-0990828-20261020081434`, `i5i6_B2` FAIL 1건)를 본 뒤에 이루어졌다. 판정은 재실행 데이터로만 한다. 1회차 데이터에 새 기준을 적용한 결과는 기술용 확인이며 판정에 쓰지 않는다.** REG 새 sha256 = `b726ea3f83f93ff8115d51dd49852b3e348cc732511f71cb243bb73cede913b3`(이전 `1c20b953…`).

1. **(4g) 계단, PD-mux arm(B2)만**: 방문한 decode `bs_real` 집합이 1부터 M까지 **연속** ∧ M = 계단 구간과 중점이 놓이는 I-5 `admission_window`의 `decode_bs_hist` 최댓값 ∧ M ≥ 32 [제안]. ★재감사 2회차 정정: 1회차 B2 원자료에서 요청 0(출력 16)이 bs=1로 15 step 단독 decode한 뒤 나머지 39개가 **단일 extend로 일괄 합류**(bs 1→39)했다. bs 40 미도달은 요청 0 종료와 일괄 prefill 사이의 **경합** 결과이며 "한 개씩 합류"가 아니다. 재실행 M은 39 또는 40일 수 있고, PD-mux 스케줄 불변량으로 인용하지 않는다. 하한이 공허하지 않은 이유: 직렬 합류 실패라면 M이 한 자릿수에 머문다. 32 = 제시한 40의 80%다. 하한 근거는 설계상 판단이지 측정이 아니다(1회차 B2는 39였고, 이 값을 본 뒤 정했다는 점이 위 공시의 핵심이다). fused·Llama arm은 기존 기준(1..40 전부)을 유지한다. 한계: I-5 등식은 엔진 자체 계수와 묶어 주지만, window 선택을 "중점이 계단 창 안"으로 했다(첫 CPU 시험에서 겹침 기준이 cap 단계 window를 끌어와 M=40 대 48로 어긋남 — 그 뒤 수정). 1회차 B2 windows는 중점 기준에서도 최댓값 39 불변.
2. **G-γ 진단 추가(판정 규칙 불변)**: (a) 계측 OFF/OFF 짝 2개(짝 id 101·102, level `offa`/`offb`, seed 4386·4162 = ON/OFF 짝 1·2와 같은 trace)를 **main phase의 ON/OFF 짝 10개 뒤에** 같은 Job 안에서 돌려, 같은 rate 사다리의 log(offb/offa)(λ\*\_thr, LO ITL p95)를 기술값으로 `diag_off_off`에 남긴다. ext phase에는 없다. 사전 등록된 용도: `UNRESOLVED_POWER`·`OBSERVER_EFFECT` 해석 보조만. TOST 표본에 들어가지 않고 어떤 verdict도 바꾸지 않는다. (b) LO 셀 token ITL 중 > 30 ms 비율을 셀마다 계산하고 5% ± 1%p(4–6%) 안이면 `STALL_BOUNDARY` 라벨(`diag_stall_boundary`). 라벨만이며 판정 불변.
3. **G-γ 시간 재계산**: 하드캡 270분, 드라이버 예산 = 270×60 − 600 = 15,600 s, instance 추정 650 s(`EST["inst"]`). ON/OFF 10짝 = 20 instance = 13,000 s(216.7분). OFF/OFF 2짝 = 4 instance = 2,600 s → 합 15,600 s = 예산과 정확히 같다(여유 0). 설치·preflight 10–15분은 드라이버 예산 밖이므로 EST 650 s를 그대로 믿으면 총 약 275분으로 270분을 넘을 수 있다. 실측 가능성이 큰 600 s/instance면 200 + 40 + 15 = 약 255분으로 들어간다. 안전장치: OFF/OFF 짝을 뒤에 두어 예산이 모자라면 `fits()`가 OFF/OFF 짝을 SKIPPED로 건너뛴다(반쪽 짝 없음, ON/OFF 짝은 보호). 짝 3개로 늘리면 EST 기준 6 instance 추가로 268분 초과 → 불채택. 더 필요하면 확장 Job으로 옮긴다(사용자 결정).

## 10. 개정 r3 (2026-10-20, G-β 1회차 데이터를 본 뒤의 수정 — 결정 20)

**공시: 이 절의 수정은 G-β 1회차 데이터(`pdmux-engine-gate-beta-b91dbde-20261020095124`, FAIL; 하네스 기대값 결함 3건)를 본 뒤에 이루어졌다(결정 20, 하네스 감사 상한 도달 후 사용자 승인). 판정은 재실행 데이터로만 한다. 1회차 데이터에 새 기준을 적용한 결과는 기술용 확인이며 판정에 쓰지 않는다.** REG 새 sha256 = `75236e5cb01e702f32f003580d29e169f17ec10287385651b0847c2ca3511ea8`(이전 `b726ea3f…`). `engine_gate_checks.py`는 sync manifest 항목이 아니므로 ENGINE_DIGEST 불변.

1. **β-0 판정 기준(§4 표의 "MATCH/MISMATCH"를 대체)**: torch 참조 대비 atol은 **기술값**(`max_abs_diff`)으로만 남긴다(1회차: 유효한 모든 T에서 0.04–0.057 = fp32 참조 대비 bf16 반올림, atol 0.02로는 판정 불가). 판정은 같은 (dim, T)의 **원본(pristine) 커널 contig 실행 출력 sha**와의 비교다.
   - 참조(REF) 사례: pristine contig, T = {12,288 · 94,784 · 94,800 · 95,100 · 126,976}(stride 22,656 계열), {95,100 · 126,976 · 189,000 · 190,500}(TP2 계열); 변이 사례용 patched contig T=94,790.
   - MATCH = 완료 ∧ conv state 정확 ∧ 출력 sha = 참조 sha. 패치본 view는 모든 T에서, 원본 view는 경계 아래(≤94,784; TP2 ≤189,000)에서 MATCH여야 한다.
   - BROKEN = crash ∨ state 불일치 ∨ 출력 sha ≠ 참조 sha. 원본 view 94,800·95,100(TP2 190,500)이 BROKEN이어야 하고, **붕괴 판정 = "원본 view ≠ 원본 contig 또는 crash"**다. chunk_offset-only 변이(`co_only`)의 참조는 **패치본 contig**(94,790 · 95,100)이며 "다름 ∨ crash ∨ state 불일치"를 BROKEN으로 판정한다. 변이가 참조와 같으면 FAIL(변이 시험이 공허).
   - rc 1은 단위가 `STATE_MISMATCH` verdict를 썼을 때만 state 불일치이고, 출력 파일 없는 rc 1(파이썬 예외)은 CRASH다. 입력 오류(rc 2)는 FAIL이 아니라 UNRESOLVED(하네스 결함). 참조가 없거나 완료하지 못하면 어떤 사례도 PASS가 아니다.
   - score_beta는 저장된 `case_ok`를 믿지 않고 사례 파일에서 다시 판정한다.
   - 사례 수 27 → 36(참조 contig 9개 추가: stride 22,656 계열 4 + TP2 4 + patched 94,790 1). 시간 영향: 1회차 Job 전체가 884 s였고 단위 사례는 사례당 수십 초 이하라 약 +5분 이내 [추정]; 하드캡 120분·기대 1.0 h 불변.
2. **TP2 사례 폭(w11328) 수정**: 1회차는 `--offset 10240 --dim 12288`(TP1 값)을 그대로 줘 offset+dim > 11,328로 전부 rc=2였다. Nano-9B-v2 config(d_inner 10,240 = 128 heads × 80, 8 groups × state 128)를 TP=2로 나누면 rank당 d_inner 5,120, groups 4, heads 64 → in_proj 폭 2·5,120 + 2·512 + 64 = **11,328**, xBC 구간 offset **5,120**, dim **6,144**(= 5,120 + 2·512; 끝 11,264 ≤ 11,328). 사례마다 `--dim/--offset`를 넘긴다. 예측(int32 경계, 뷰 기저 기준 최대 원소 인덱스 (T−1)·stride + dim − 1 ≤ 2³¹−1): stride 22,656 → T ≤ 94,785까지 안전(94,786부터 wrap; 94,784 안전·94,800 붕괴와 일치), stride 11,328 → T ≤ **189,572**까지 안전(189,573부터 wrap). 따라서 pristine view 95,100 · 126,976 · 189,000 = MATCH, 190,500 = BROKEN; 패치본 view는 190,500까지 MATCH 예측. (경계 값은 산술이며 189,572/189,573 한 칸의 차이는 사례로 검증하지 않는다.) [유도, 미측정]
3. **β-2 비차단 boot 서명**: 비차단(boot #1)은 "사망 ∧ `illegal memory access` 서명"만 요구한다(1회차: 사망 rc −9 + illegal 서명이 있었으나 비동기 traceback에 conv 파일이 없어 FAIL). traceback의 `causal_conv1d_triton.py`는 차단(`CUDA_LAUNCH_BLOCKING=1`) boot에서만 요구한다(사망 시 illegal ∧ conv 언급 ∧ traceback 파일). 서버가 살아남은 경우의 "출력 ≠ 패치본 출력" 판정은 불변.
4. **G-γ**: 결정 20 (a)에 따라 **OFF/OFF 진단 짝(§9-2 (a))을 기본 계획에서 뺀다**. `PDMUX_GATE_GAMMA_OO_PAIRS=1|2`(실제 env, 기본 0)로만 켜는 확장 Job 옵션이다(id 101.. main / 201.. ext). OO 짝이 예산 부족으로 못 시작하면 `DIAG_SKIPPED`로 기록하며 `run_gate`의 SKIPPED/UNRESOLVED 집계(이 상태는 제외됨)에 들어가지 않는다. ON/OFF 짝의 SKIPPED는 그대로 verdict를 낮춘다. STALL_BOUNDARY 라벨은 셀 기술값이라 유지한다(stage 기록 없음, `checks.diag_stall_boundary`는 `DIAGNOSTIC`로 rollup 제외) — `run_gate` 전체 경로 시험(모든 LO 셀이 ~5% stall이어도 verdict PASS, SKIPPED/UNRESOLVED stage 0)으로 확인했다.
   - **G-γ 시간 재계산**: 하드캡 270분, 드라이버 예산 15,600 s, instance 추정 650 s. ON/OFF 10짝 = 20 instance = 13,000 s = **약 217분**(예산 대비 여유 2,600 s ≈ 43분; 설치·preflight 10–15분 포함해도 약 232분 [EST 기준], 실측 가능성이 큰 600 s/instance면 약 215분). §9-3의 "여유 0"는 OO 제거로 해소된다.
5. **시험**: judge_case 표 시험(양방향), 사례 기하·참조 순서 시험(offset+dim ≤ width), 변이 시험(patched 출력 오염 → FAIL, co_only가 참조와 같음 → FAIL, 파이썬 예외 rc 1 → CRASH, 사례 파일 sha 오염 재채점 → FAIL, 참조 사례 소실 → PASS 불가, 비차단 서명 기준 양방향), gamma OO 기본 제외·옵션·DIAG_SKIPPED·STALL run_gate 경로 시험. 고친 가짜 서버는 실제 관찰대로 비동기 crash에 conv 파일 traceback을 쓰지 않는다.
