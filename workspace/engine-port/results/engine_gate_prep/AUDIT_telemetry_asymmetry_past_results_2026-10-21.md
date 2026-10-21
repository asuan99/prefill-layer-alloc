# AUDIT: telemetry ON/OFF arm 비대칭이 과거 결론에 미친 영향 (claims-auditor 1회차, 2026-10-21, 읽기 전용)

> 감사자 최종 보고를 main 세션이 그대로 옮겨 적었다(감사자는 읽기 전용). 재감사는 이 판정 이후 diff만, 최대 1회.

## 0. 요약

- **비대칭은 구조적으로 실재한다.** telemetry writer(`AsyncJsonlTelemetry`)는 `init_pdmux` 안에서만 만들어진다(`sglang_engine_dev/python/sglang/srt/multiplex/multiplexing_mixin.py:224`, 호출은 `scheduler.py:378-379` `if self.enable_pdmux`). emit/sync 13곳은 전부 `multiplexing_mixin.py`에 있다. 그래서 fused(vanilla) arm은 env와 무관하게 telemetry를 켤 수 없다 — `PDMUX_TELEMETRY_PATH`를 준 PD-mux 대 fused 비교는 설계상 PD-mux 쪽만 관측자 비용을 진다.
- **그러나 확정 결론 대부분은 노출되지 않았다.** P1-opint(+40.5/+185.8/+27.0%)·HE0는 **전 arm telemetry OFF**, N2 Job-EA는 **전 boot ON(대칭)**. 실제 비대칭은 **M3 J1의 λ\*·G 비**뿐이다.
- I-4 CUDA event(ITL p95 +1.0%)는 2026-10-20 커밋 `34f2acc8`, PHASE_EVENTS는 2026-10-04 신설 — 그 이전 캠페인에는 없었으므로 과거 결론과 무관하다.
- 분해 Job의 "TEL"은 `PDMUX_TELEMETRY_PATH` + `PDMUX_PHASE_EVENTS=1` **묶음**이다(`engine_gate_decomp.py:57`). "기존 telemetry(경로만)가 −1.7%"는 **NOT-YET-SUPPORTED**(confound #10).

## 1. 트리거 주장(분해 Job) 판정

| 주장 | 판정 | 근거 / confound |
|---|---|---|
| TEL이 PD-mux λ_thr을 −1.7% 낮춘다 | **CONFIRMED(scoped)** — 묶음 처치, 이 Job·shape A·agnostic·Nano-9B-v2 한정 | `engine_gate/pdmux-engine-gate-decomp-d1f2ddb-20261021022225/DECOMP_RESULT.json`: n=4 블록, ln −0.0172, t-CI [−2.46, −0.95]%. 6검정 Bonferroni 통과(p=0.0056 < 0.0083). gamma A 전체 ON −2.6% [−2.9, −2.3](n=10)과 정합 |
| 위 효과가 "기존 telemetry(TELEMETRY_PATH만)"의 비용이다 | **NOT-YET-SUPPORTED** | #10 변수 동시 변경: TEL 수준은 PHASE_EVENTS(decode iteration마다 emit, 이 boot 18,995건)를 함께 켠다. 과거 캠페인·J1은 경로만. 몫 미분리 |
| I-4가 ITL p95를 +1.0% 올린다 | **CONFIRMED(scoped)** — I46 vs I6 증분, p=0.0073(Bonferroni 통과) | 과거 캠페인에는 없던 계측 |
| fused arm에서는 telemetry가 꺼져 있었다 | **CONFIRMED**(코드 구조) | §0 |
| (부수) gamma A/F의 "ON"은 같은 처치가 아니다 | 사실 확인 | `engine_gate_driver.py:439-445`: agnostic ON = I5+I46+TEL, fused ON = I5+I46. A(−2.6%)와 F(−0.65%, UNRESOLVED_POWER)를 "같은 관측자 비용"으로 대비 금지 — 차이는 대략 TEL 묶음 몫 |

## 2. 과거 결론별 판정표

| 결론 | arm 간 telemetry | 근거 | 편향 방향 | 뒤집힐 가능성 | 판정 |
|---|---|---|---|---|---|
| **P1-opint** Zamba2 r2 +40.5%·r3 +185.8%, Granite r4 +27.0% (CONSENSUS §1-1, PROJECT_STATUS:6664-6750) | 양쪽 OFF | `results/p1_opint/p1op_run.sbatch`에 `PDMUX_*` export 0, `P1_OPINT_RESULT_2026-08-05.md:330`, telemetry.jsonl 없음. PHASE_EVENTS·I-4/5/6 당시 부재 | 없음 | 없음(이 축) | **CONFIRMED-불변**(이 축 한정; 기존 제한 — 3-플래그 #13, fused 미조율, t-CI, T<113 ms — 그대로) |
| **HE0** §1-7·§1-17 | 전 arm OFF(전부 PD-mux arm) | `slo_sched/sharegpt_vary_bench.sbatch`·`interactive_bench.sbatch`에 경로 없음, g16 외 telemetry 파일 0(`DESIGN_S6_TELEMETRY_CONTRAST_2026-08-17.md` §0). 컨트롤러 CPU 비용은 처치의 일부, 0.014%(§1-12) | 없음 | 없음 | **CONFIRMED-불변** |
| **N2 Job-EA** §1-38 | 전 boot ON(경로 + PHASE_EVENTS, 대칭) | `n2_static_grid/n2_plan.py:683-694` | 플래그 대칭, 단 `observe_scheduler`가 sync마다 O(queue) → backlog 있는 C 창에서 host 비용 더 클 수 있음 | 낮음: S-per는 GPU 쪽 구간, host 간격 증가 중앙 +0.29 ms, prefill 발사·대기 iteration S-per 차 −0.0004(허용 문안 C) | **CONFIRMED-불변**(라벨, scoped). 크기는 이미 NOT-YET-SUPPORTED; 합성 성분 목록에 "부하 의존 관측자 비용(미분리)" 공시 추가 권고 |
| **J1 CM-2 floor** | 비대칭: fused_sym(OFF)·agnostic(ON) | `mg_const.py:157 FLOOR_ARMS`; arm별 floor 1024 tok 0.12272 대 0.12249 s(+0.18%), 16k 1.8234 대 1.8245 s(−0.06%) | 무경합 단일 요청이라 사실상 0 | 없음(≤0.2% < 공시 산포 ±0.6%) | **CONFIRMED-불변** |
| **M3 J1** λ\*·G 비(B G_sym 0.720, A G_sym 1.0056, A G_d44,sym 0.947) | 비대칭: PD-mux arm ON, fused·fused_sym OFF, PHASE_EVENTS 양쪽 unset | `manual_retrieval/pdmux-multigpu-2106f42-20261018074950/.../boots/m3_N1_*/env.json` 12개 전수 | PD-mux 불리 | B: 뒤집히려면 약 39% 필요, −1.7% 보정해도 0.733 → 부호 불변. A G_sym: 편향이 효과(0.56%)의 약 3배라 "≈1" 크기 해석 불성립(보정 시 약 1.023). A d44: 5.3% 열세 중 최대 약 1/3이 관측자 몫 가능 | **부호 견고·크기 공시 필요.** A G_sym 인용 시 재측정 필요(현재 PARTITION_NOT_EXERCISED로 미인용) |
| J1 CM-6 B 허용 문안(요청 ITL p95 SLO 통과율 동일, p95 중앙 16 대 24–29 ms) | 비대칭(위와 같음) | 같음 | PD-mux ITL 상승 가능 | 낮음: TEL ITL p95 효과 +0.38%, CI∋0; 격차 50–80%; PD-mux 통과율 1.00은 불리한 편향 하 | **CONFIRMED-불변**(허용 문안 그대로) |

### 외삽 가능성
**판단 불가** — 위 판정은 외삽 없이 성립한다. 걸리는 confound: #11 기판 혼합(KISTI 캠페인으로 이식 불가, 같은 VESSL이라도 J1은 다른 Job·GPU UUID) · #10 묶음(분해 TEL엔 PHASE_EVENTS 포함, J1엔 없음) · 모델·부하(분해는 shape A 256-token prefill로 sync 빈도 높음; B/X는 sync 드물고 queue 깊음, 비용 O(queue)라 방향 연역 불가) · #3 small-n(n=4). P1-opint·HE0는 OFF라 외삽 불필요; 하중을 받는 곳은 J1과 향후 G0뿐.

### 추가 노출(범위 밖, 미검사, 등급 영향 미판정)
1. telemetry ON으로 PD-mux arm끼리만 비교한 계열(C2/S2/E1/G16 §1-33, N3 4회차 §1-37[이미 "PHASE_EVENTS ON 조건에서"로 한정]): 플래그 대칭이나 비용이 부하에 단조일 수 있어 arm과 상관 가능. S-6가 지목했으나 미실행(rev1–rev4 NO-GO).
2. fused/chunked 기준선을 쓰는 다른 캠페인의 arm별 env 미전수: `p1_gates/gate2`(G2S telemetry 사용, HOLB probe 양 arm export), `cp_baseline`, `longctx_conflict`.

## 3. 인용 금지 문안 후보

1. 금지: "기존 telemetry는 PD-mux 처리율을 1.7% 낮춘다." 대체: "Job `pdmux-engine-gate-decomp-d1f2ddb-20261021022225`(VESSL A100 `GPU-7b1940f0…`, Nano-9B-v2, agnostic, shape A, n=4 블록, Latin 순서)에서 `PDMUX_TELEMETRY_PATH`+`PDMUX_PHASE_EVENTS=1` 묶음은 OFF 대비 HI 셀 achieved를 ln −0.0172(−1.7%, t-CI [−2.5, −1.0]%) 낮췄다. 두 knob의 몫은 분리되지 않았다."
2. 금지: "과거 PD-mux 대 fused 결과는 telemetry 비대칭 편향을 받았다"(일반형). P1-opint·HE0는 전 arm OFF; 비대칭 확정 계열은 M3 J1뿐.
3. 금지: "PD-mux의 관측자 비용은 −1.7%(또는 −2.6%)로 상한된다." 단일 Job·shape A·묶음의 점추정이지 상한이 아니다.
4. 금지: "I-4 CUDA event가 과거 ITL 결과를 +1% 부풀렸다." 2026-10-20 이전엔 없던 계측.
5. J1 A shape의 "G_sym ≈ 1"·"d44가 fused_sym보다 5.3% 낮다"는 "PD-mux arm에만 telemetry ON(분모 fused는 OFF, PD-mux 불리 방향, 크기 미측정)" 병기 없이 인용 금지.
6. 금지: "telemetry가 대칭이었으니 N2 ρ에는 관측자 성분이 없다." 부하 의존 성분 미분리.

## 4. 해소 실험 (최소 비용)

- **P1-opint·HE0·N2·CM-2**: GPU 불필요. CPU 확인 1건 권고 — 873944/873945의 `runtime_source_manifest_*.sha256`에 대응하는 `multiplexing_mixin.py`에 `_dual_worker_sync`의 "trace path 없으면 즉시 return" 분기가 있었는지 아카이브 바이트로 확인(없었다면 OFF arm에도 관측 비용 존재; 그래도 fused는 0이라 편향은 PD-mux 불리, 결론은 보수적 쪽).
- **J1 / G0**:
  - (a) **G0 사전등록 규칙(가장 쌈)**: 헤드라인 셀은 전 arm에서 `PDMUX_TELEMETRY_PATH`·`PHASE_EVENTS`·I-4/5/6 OFF, 실현 파티션(E-time) 등 telemetry 파생량은 별도 boot·셀에서 잰다. 추가 GPU 0(E-time 필요 시 arm마다 telemetry 전용 boot 1개).
  - (b) J1 수치 보정을 원하면 1 Job: agnostic, shape {A, B}, 수준 {OFF, TELPATH-only(PHASE_EVENTS unset)}, 4 블록 paired·Latin, J1과 같은 셀 정의(HI·LO rung). 판정량 ln(TELPATH/OFF)의 λ\*_thr·LO 셀 요청 ITL p95, t-CI. `clk.csv`·0x4 노출 기록(B는 J1에서 SW cap 노출 47–93%). 약 4–5 h, 제출 전 `mg_plan`식 예보로 확정(J1 예보 46% 빗나간 전례). #10도 이것이 해소. 기존 `engine_gate_decomp.py` `level_env` 기구(driver `:438-439`)로 엔진 바이트 변경 없이 구현 가능.

## 5. 감사 메타

- 이 대상 1회차. 정본(CONSENSUS §1/§6)과 모순되는 새 판정 0, 등급 변경 제안 0. Claim G 근거 P1-opint는 이 축에서 불변.
- 참조: engsync `results/engine_gate/pdmux-engine-gate-decomp-d1f2ddb-20261021022225/DECOMP_RESULT.json`, `…/pdmux-engine-gate-gamma-{a,f}-ad11bcc-*/GATE_RESULT.json`, `scripts/vessl/engine_gate_driver.py`(:439-445), `engine_gate_decomp.py`(:6-11, :57); `sglang_engine_dev/python/sglang/srt/multiplex/multiplexing_mixin.py`(:170-181, :224, :743-805, :2129-2300); main `results/p1_opint/p1op_run.sbatch`, `P1_OPINT_RESULT_2026-08-05.md:330`, `results/slo_sched/sharegpt_vary_bench.sbatch`, `DESIGN_S6_TELEMETRY_CONTRAST_2026-08-17.md`, `results/n2_static_grid/n2_plan.py`(:683-694), `results/multigpu/mg_const.py:157`, J1 `manual_retrieval/…/boots/*/env.json`·`FLOOR.json`, `multigpu_verdicts/ANALYSIS_J1_N1_2026-10-19.md`·`AUDIT_J1_N1_analysis_2026-10-19.md`.

## 6. 후속 CPU 확인 결과 (2026-10-21, 결정 23; 읽기 전용 확인, 감사 회차 아님)

- P1-opint manifest `results/p1_opint/runtime_source_manifest_{zamba2-27b_873944,granite-40-h-micro-base_873945}.sha256`(두 파일 동일)의 `multiplexing_mixin.py` sha256 `59eaafb4…`는 커밋 `f60a1286`(2026-08-03)의 `workspace/engine-port/src/multiplex/multiplexing_mixin.py`와 **바이트 일치**(그 경로 27개 리비전 중 유일). 같은 커밋의 telemetry.py·dual_worker.py·controller.py·profile.py도 manifest와 일치.
- 분기(f60a1286 줄 번호): `_dual_worker_sync` :470-474 — dual worker도 trace path도 없으면 즉시 return(observe_scheduler O(queue) 순회·mark_phase·json emit 전부 건너뜀) · :526 trace path 있을 때만 기록 · telemetry.py :62-69 path None이면 writer 스레드 없음, :78 emit 즉시 반환 · :968 `_slo_on` False라 컨트롤러·EMA 블록 미실행.
- OFF 근거: `p1op_run.sbatch`에 PDMUX_* export 0, agnostic 서버 로그에 observer/true-dual 활성 로그 없음, `P1_OPINT_RESULT_2026-08-05.md:330`. 셸 상속 env는 직접 확인 불가.
- 남는 매 iteration 비용: :1092-1094 카운터 +1, :613-617 decode 배치 O(batch) `finished()` 순회(결과 폐기) — GPU 동기화 없음.
- **결론: P1-opint agnostic arm의 telemetry 관측 비용 0 확인**(위 사소한 CPU bookkeeping 제외). plain arm은 `enable_pdmux=False`라 mixin 자체를 실행하지 않았다. §2의 P1-opint `CONFIRMED-불변` 판정을 바이트 수준에서 뒷받침한다. 남은 `synchronize`·이벤트 기록(:1073-1074, :1184, :1199, :1217)은 PD-mux 실행 기구이며 telemetry와 무관(업스트림 대조는 안 함).
