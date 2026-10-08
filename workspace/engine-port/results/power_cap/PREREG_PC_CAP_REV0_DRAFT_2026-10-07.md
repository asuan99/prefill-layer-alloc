# 사전등록 초안 — **PC-CAP (GPU power cap 한 노브 대조)** **rev0 DRAFT** (2026-10-07)

> **상태: 초안. 실행 불가.** `-pl`(power limit 설정)이 허용되는 기판이 아직 없다(VESSL 컨테이너는 권한 거부 —
> `VESSL_A100_CONTEXT.md` §4, `handoff-report/vessl_cloud_setup_plan_2026-09-24.md:268-269`; VESSL 안의 유일한 root 경로는 **VM Cluster**(H100 이상·8 GPU/노드·선불 계약, `reports/power_cap_viability_2026-10-07.md` §2-1) —
> 그 경우 기판이 H100으로 바뀌어 §2·§3의 A100 anchor(λ\*·격자·N3 참조)는 전부 재앵커 대상이다).
> 기판이 정해지면 rev1로 올려 claims-auditor 규칙층 감사 → 하네스층 감사 → 사용자 승인 → 제출 순서를 밟는다.
> 구조는 `results/r2_eval/e2_sticky_prereg/PREREG_E2_STICKY_REV3_2026-09-15.md`(한 Job 안 one-knob A/B)와
> `results/n3_capacity/PREREG_N3_CAPACITY_2026-10-04.md` §2-4/§8(기판 기록·runner 규율)을 따른다.
> 근거 자료: `RESULT_power_cap_passive_2026-10-07.md`(Rung A, 기술적) · `reports/power_cap_viability_2026-10-07.md`.
> GPU 지출 0. 라벨 0건.

## 0. 한 줄

**같은 Job·같은 GPU에서 Nemotron-H D44 sticky-OFF(B1) arm을 고정하고 chip-wide power cap만 {400, 300, 250, 200} W로 바꿀 때,
고정 부하에서 TTFT p95·요청 ITL p95·J/req가 cap을 따라 어떻게 움직이는지**를 shape A/B 각각 seed 4개 짝으로 잰다.
정책 비교가 아니다. split×cap 격자(rung D)의 선행 조건(cap이 실제로 물리는가, 지표가 cliff 밖인가)을 확인하는 회차다.

## 1. 질문과 판정량

| 기호 | 판정량 | 등급 |
|---|---|---|
| **Q1** | cap이 **물리는가**: cap 수준별 bench window의 0x4 노출 %·평균 SM MHz·p95 W (`clk.csv`, `n3_analyze.clk_windows` 확장) | 1차(처치 검증) |
| **Q2** | 같은 seed 짝에서 **TTFT p95 Δ(cap − 400 W)** | 1차 |
| **Q3** | 같은 seed 짝에서 **요청 내 token-ITL p95의 p50 Δ** | 1차 |
| **Q4** | **J/req, J/out_tok**(E = Σ P·Δt, `pc_energy.py` 정의) Δ | 2차(기술) |
| **Q5** | 정본 술어 goodput 율(TTFT ≤ SLO ∧ 요청 ITL p95 ≤ SLO) | 3차(병기, cliff 밖에서만) |
| **Q6** | 실현 분할(`n3_realized_mix.py` E-time/W-dec) — **arm 동일성 검증** | 한 노브 검증 |

판정 추론기: seed-paired 차이의 **paired bootstrap 95 % CI**(`benchmarks/pdmux_eval/analyze.py:115 paired_bootstrap_ci`, B=10,000, seed 1).
"변했다" = |Δ| ≥ 3 % **그리고** CI가 0을 제외(게이트 3). n ≥ 4 미만이면 `UNRESOLVED_N_BELOW_4`.

## 2. arm · config · binding · 기판 전제

- **모델**: Nemotron-H(`nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base`) 단일 — Nemotron-first 사다리(2026-10-04 사용자 결정). Zamba2 등 확장 금지.
- **arm 고정**: `PDMUX_R2_POLICY=fixed`, `PDMUX_R2_FIXED_DSM=44`, sticky OFF, cudagraph ON, config digest = N3 `pdmux_n3_{A_agn,B_d92}.yml`
  (`n3_plan.CONFIG_DIGEST`). **노브는 power cap 하나뿐.**
- **cap 격자**: {400(기본·대조), 300, 250, 200} W. 200 W는 A100-SXM4 최소 100 W의 2×로, 클럭이 확실히 떨어지는 점(예보 §9). 순서는 seed 블록마다 교차(ABBA 확장: 400-300-250-200 / 200-250-300-400 …).
- **기판 전제(B0 프로브 체크리스트, 대여 직후·제출 전)**:
  1. `nvidia-smi -q -d POWER`에서 `Current/Default/Min/Max Power Limit` 읽기 — A100-SXM4-80GB 기대 400/400/100/400 W(VESSL A100에서 2026-10-07 실측 일치 — `probe_job-x3lzellrkm8z`; 새 기판에서는 다시 읽는다).
  2. `nvidia-smi -pm 1` rc=0, `nvidia-smi -pl 300` rc=0 **그리고** 읽기값 `power.limit == 300.00 W`.
  3. 되돌림 `nvidia-smi -pl <default>` rc=0, 읽기 재확인. 실패 시 이 사전등록은 그 기판에서 **무효**.
  4. (선택) `nvidia-smi -lgc` rc — clock-locking arm(§4 L) 가능 여부.
  5. 드라이버/CUDA/`sgl_kernel` 빌드·green context smCount(`check_green`)·cgroup quota(교훈 268) 기록 → `substrate.json` 확장 필드
     `power_limit_{current,default,min,max}`, `persistence_mode`.
- **같은 기판 안 비교만**: λ\*·격자·수치를 VESSL/KISTI 값과 **혼합 금지**(CLAUDE.md 기판 주의). 부하 anchor는 §3.

## 3. 부하 (고정, 변화 trace 아님)

- shape B: 8192→64, **0.30 req/s**(N3 `obs_rate`, ≈0.43 λ\*\_VESSL(B)); shape A: 256→512, **1.80 req/s**(≈0.60 λ\*\_VESSL(A)). N3와 같은 `n_prompts = rate × span_s`, seed 4개(`n3_plan.SHAPES[*]["seeds"]`).
- 이 점은 VESSL에서 cliff 밖(N3 r3: 0.30에서 good_frac 0.96)이었다 — **새 기판에서는 다시 확인**(floor 유닛 + 400 W arm이 N3 참조 ±5 % 안이어야 함, §7).
- 변화 trace(rate 3↔12)는 이 회차에 없다 — 정책 비교가 아니므로 게이트 2는 적용 대상이 아니다. 단 결과를 "정책"으로 읽는 것은 그 이유로 금지(§10).

## 4. Job 구조 · 유닛

한 shape = 한 Job(ARRAY 금지), 같은 GPU UUID.

| 순서 | 유닛 | 수 | 목적 |
|---|---|---|---|
| 1 | `floor_head`(400 W) | 1 | T_k 눈금(N3 P-2) + 기판 재현 게이트 |
| 2 | `cap{400,300,250,200}_s{seed}` | 16 | 본 측정(4 cap × 4 seed, 블록별 교차) |
| 3 | (선택) `lock{f}_s{seed}` — `-lgc f`(f = 250 W arm의 실측 평균 SM MHz) | 4 | clock-locking 베이스라인(arXiv 2605.11999의 "locking이 capping을 이긴다" 대조) |
| 4 | `floor_tail`(400 W) | 1 | 드리프트 |

유닛마다: `settle` → **`nvidia-smi -pl $CAP` → 읽기 검증(§5)** → 서버 기동 → health → warm-up → CG2 → bench → teardown. PHASE_EVENTS=1(decode_iteration 필요: §6 Q3′).

## 5. 한 노브 검증 (fail-closed)

- 노브 등록: `PDMUX_POWER_CAP_W`(정수 W)를 **runner 허용목록에만** 추가(`n3_capacity.sh:53-59` 패턴). 엔진(`src/`)은 해석하지 않는다. `launch.sh --env`는 `PDMUX_*`만 통과(`launch.sh:42-44`)하므로 추가 배관 불필요.
- 설정 직후: `nvidia-smi --query-gpu=power.limit,power.min_limit,power.max_limit,persistence_mode --format=csv,noheader` → `SETTLE.txt`에 기록; `|limit − want| > 1 W` 또는 rc≠0이면 **`ABORT_POWER_CAP`**(새 코드; `fail()` 번호 체계에서 미사용 번호).
- cleanup trap: 기본값 복원 + 읽기 재확인. 복원 실패도 Job 실패로 기록(다음 유닛 오염 방지).
- 사후(Q1): bench window 0x4 % + 평균 SM MHz. **binding 정의**: 0x4 ≥ 50 % **또는** 평균 SM MHz ≤ 0.90×max. 400 W arm은 기본 환경(VESSL에서 25–50 %)과 같은 범주이므로 binding 여부와 무관하게 대조 arm.
  어떤 cap 수준이 binding이 아니면 그 수준의 Q2–Q4는 **`CAP_NOT_BINDING`으로 라벨만** 남기고 판정 제외.
- arm 동일성(Q6): 모든 유닛의 실현 분할 E-time D44 점유가 유닛 간 **±5 %p** 안이어야 한다. 벗어나면 그 유닛은 `ARM_DRIFT`(cap 때문에 분할이 바뀌었다면 그것 자체가 보고 대상이지만 Q2–Q4 판정에서는 제외).

## 6. 추정량 · 결정 규칙 · 라벨

- cap 수준 c ∈ {300, 250, 200}, seed s: Δ_c,s(X) = X(c, s) − X(400, s), X ∈ {TTFT p95, 요청 ITL p95의 p50, J/req, J/out_tok}.
- 라벨(수준별, 사전 고정):
  - `CAP_NOT_BINDING` (§5)
  - `TTFT_SHIFT` : Q2 변함 ∧ Q3 안 변함
  - `ITL_SHIFT` : Q3 변함(Q2 무관)
  - `NO_SHIFT` : 둘 다 안 변함(|Δ| < 3 % 또는 CI가 0 포함)
  - `UNRESOLVED_N_BELOW_4` : 유효 seed < 4
- Q3′(보고 전용, 판정 아님): `pc_itl_throttle.py` 방식으로 decode step 시간을 bs×동거 층에서 0x4 유무로 비교 — cap 수준별 "클럭 채널" 크기.
- Q4는 기술 전용: J/req가 cap을 따라 줄어도 "에너지 절감"이라 쓰지 않는다(Q5가 cliff 밖에서 유지될 때만 "SLO 유지 하 J/req 감소"로 병기).

## 7. 게이트 · 킬

- G1 기판 재현: 400 W arm 4 seed의 achieved rate·TTFT p95·요청 ITL p95 p50이 N3 r3 `g_r030_*`(B) / 해당 A 참조와 **±5 %** 안. 아니면 Job 중단(`ABORT_SUBSTRATE_REPRO`) — 수치는 새 기판 anchor로만 보관.
- G2 cap 설정 검증 실패 → `ABORT_POWER_CAP`.
- G3 예산: `n3_plan.unit_cost_s` 모형 + 유닛별 admission(§8). 하드캡 B 17,100 s / A 14,400 s(rev7 per-shape cap 승계).
- G4 3회 연속 boot 실패 → 중단(N3 규율 승계).
- G5 thermal: `temperature.gpu` 유닛 간 최대 편차 > 10 °C면 `THERMAL_DRIFT` 플래그(판정 제외 아님, 보고).

## 8. 예산 (GPU-h만; 과금 서술 금지)

`n3_plan.py` rev7 상수(2026-10-07 로컬 계산: `unit_cost_s`):

| shape | floor | obs 유닛 | 1 floor + 16 obs | + preflight 392 s | (선택) +4 lock |
|---|---|---|---|---|---|
| B | 220 s (worst 262) | 293 s (335) | **1.36 h** (worst 1.56) | 1.47 h (1.67) | +0.33 h |
| A | 270 s (310) | 275 s (315) | **1.30 h** (worst 1.48) | 1.40 h (1.59) | +0.31 h |

A 상수는 rev7 주석대로 **미측정**(A 제출 전 재추정). 새 기판에서는 boot 시간이 달라지므로 첫 floor 유닛의 TIMING으로 상수를 재추정하고 admission을 다시 돌린다(교훈 273).

## 9. 반증 가능 예보 (등록 시 고정)

- F1 (shape B): cap을 낮출수록 TTFT p95가 **단조 증가**(prefill compute-bound; KISTI prefill 축에서 전력이 prefill SM 몫을 따랐고, N3 shape B에서 0x4가 prefill 동거((64,44))와 거의 겹침 — RESULT §2-1·§3, 캠페인·arm 한정).
- F2 (shape B): 300 W에서 요청 ITL p95 **무변화**(|Δ| < 3 %); 250/200 W에서 평균 SM MHz가 1200 미만으로 떨어질 때만 변화. (RESULT §3: 400 W 환경의 −2.8 % 클럭에서 bs1 (0,108) step +0.2 % — 전환 인접 표본, 운영 분할 미측정.)
- F3: J/req는 cap을 따라 감소하되 TTFT가 SLO를 넘는 수준(Q5 cliff)에서 goodput이 먼저 무너진다 — "SLO 유지 하 에너지 절감"은 300 W까지만 성립(또는 전혀 성립 안 함).
- F4 (shape A): decode 비중이 커도(bs ≤ 48) ITL 변화는 F2와 같은 임계(SM MHz < 1200)에서만 나타난다.
- F5: 400 W arm의 0x4 노출은 VESSL 값(B 24 %, A 25 %)과 ±10 %p 안 — 아니면 기판 차이로 기록(G1과 별개).
- 어느 예보가 깨져도 **cap 정책 주장은 생기지 않는다**; rung D 설계 입력만 바뀐다.

## 10. 인용 규칙 · 금지 문안

- 이 회차는 **단일 split·단일 rate**다: "최적 split이 cap 따라 이동"(rung D), "cap 하 λ\*", "cap-aware 정책/H-Policy", "에너지 절감 X %" 전부 금지.
- 기판 간 수치 혼합 금지(VESSL/KISTI 값은 G1·F5의 참조로만).
- <3 % Δ는 headline 아님. Q5(goodput)는 cliff 밖에서만 병기(게이트 6).
- PEACE(SoCC'26)와의 관계: "일반 기제는 선행이 보였고 PD-mux판은 미검증"으로만 서술. "최초" 금지(교훈 260).

## 11. 이 회차가 못 하는 것 (rung D로 넘김)

split×cap 격자({D16,D24,D44,D64}×{400,300,200}×{A,B}×4 seed ≈ 96 유닛, 점격자 하한 ≈ 5.5 h/shape; capacity-first면 cap별 λ\* 스윕이 필요해 ~3× N3)는 이 회차 결과(어느 cap이 binding이고 어느 지표가 움직이는가)로 격자를 줄인 뒤에만 사전등록한다.
per-partition 전력 귀속(NVML은 전체 GPU 값)은 어느 회차도 못 한다 — 서술에서 "power partition" 금지.

## 12. 제출 전제

1. `-pl` 가능한 기판 확보 + §2 B0 체크리스트 전부 PASS(사용자 결정).
2. rev1 작성 → claims-auditor **규칙층** 감사 `GO`/`GO-with-caveats` → runner 구현(engine-porter/experiment-runner, `n3_capacity.sh` 포크) → **하네스층** 감사.
3. launch dry-run → 사용자 승인 → `--submit`. 회수 = `fetch.sh` DONE + sha256.
4. 결과 감사 후에만 RESULT 작성. 정본 반영은 doc-steward.

## 13. 로컬 검증 계획 (GPU 0)

- plan 모듈(`pc_cap_plan.py`): 유닛 목록·예산·`--selftest`(교란 변이: cap 순서 섞임 검출, 예산 상수 변경 시 admission 변화).
- 분석기: `pc_energy.py`/`pc_itl_throttle.py` 재사용 + paired bootstrap(기존 `analyze.py`). 합성 데이터에서 `TTFT_SHIFT`/`NO_SHIFT`/`CAP_NOT_BINDING` 세 라벨을 각각 산출하는 selftest(무효과 대조군 포함, 교훈 53/254).
- runner 변경은 mock `nvidia-smi`로 dry-run(VESSL 운영 모델의 mock 전례 `vessl_operating_model_2026-10-02.md:116`).
