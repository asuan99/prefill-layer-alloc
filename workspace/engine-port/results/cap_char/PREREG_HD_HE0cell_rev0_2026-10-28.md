# 사전등록 rev0.2 (POLICY-1) — H-D(phase 경계 여지) + Nano HE0 핵심 셀, 정책 Job 1차, 2026-10-28

> **봉인 상태: 미봉인 초안 rev0.** 문서 작업(CPU)만 했다. 측정 0, GPU 0, `launch.sh` 실행 0(dry-run 포함), 판정 0. 하네스 코드는 쓰지 않았다(감사 뒤).
> 지위: 정책 Job 1차(결정 40·43). 두 실험을 한 PREREG에 두되 **판정·라벨·CI는 서로 섞지 않는다**: (A) H-D = static arm만의 phase-oracle 여지 · (B) Nano HE0 핵심 셀 = static 격자 대 reactive bind+GATE.
> 근거: 설계 `PLAN_V_oracle_gap_dynamic_headroom_2026-10-26.md` rev4 §2(결정 32·33 반영본; 이 문서는 §2를 **그대로 승계**하고 아래 명시한 곳만 더한다) · 결정 43–45(`tracks/main.md`) · 변화 trace 감사 `../anchor_audit/VERDICT_change_trace_validity_r1_2026-10-27.md` · 진단 짝 PREREG `../lockstep_diag/PREREG_DIAG_stall_drain_rev0_2026-10-28.md` · 형식 `PREREG_COSTMODEL_TERMS_MEAS_rev3_2026-10-20.md`(봉인 S1′ 형식).
> **rev0.2(2026-10-28, 미봉인): 하네스층 감사 r1(`AUDIT_harness_HD_HE0cell_r1_2026-10-28.md`, 死因 D1–D4·MAJOR M1–M7·MINOR m1–m9) 수리 + 결정 49(O-P6: 전 Job static arm 비sticky). 감사 항목 → 수리 위치 대조표는 §13. rev0.1 = 결정 47(W3 = arXiv-Sum 16k 자연 출력, G0·O-V2 의존 제거, shape C는 V-O 2차)·결정 48(phase 사이클 1).**
> 수치 라벨: [측정](출처 명시) · [유도](식 대입·시간 모형, 측정 아님) · [코드] · [결정] · [제안](이 문서가 고른 값, 봉인 시 사용자 확인) · [미정] · **[봉인 시 기입]**. `[미정→초안 채택]` = 사용자 결정이 아직 없어 초안이 채택한 값.
> **V-O(oracle 격차)는 이 PREREG에 없다**(결정 43: V\_REG·G0 PREREG 뒤 2차).

## 0. 무엇을 하고 하지 않는가

**(A) H-D**: Nano에서 phase 경계에서만 전환하고 전환 비용이 0인 정책의 여지를 static arm만으로 잰다(설계 rev4 §0). `ABSENT`여도 phase 내부 적응 정책의 여지는 닫지 않는다. 정본 §1-19의 per-static oracle +2.1%(KISTI, n = 1–2, overload-only, 강등)와 결합·비교하지 않는다.

**(B) Nano HE0 핵심 셀**: KISTI 변화 trace 구조를 Nano로 옮겨, 동적 제어기(reactive bind+GATE)를 Nano에서 **재튜닝**(게이트 8)한 뒤 static D 격자와 n ≥ 4 짝 seed 구간으로 비교한다.

**허용 문안 틀(결정 44, 문자 그대로 틀)**: "결과는 Nano/VESSL 별도 claim이며 KISTI HE0를 확인·반박하지 않는다."

**하지 않는 것**: KISTI HE0(Zamba2-2.7B·triton·cudagraph ON·변화 trace)의 확인·반박, lockstep 정지의 기여 귀속(진단 PREREG 몫), fused 대비 이득, V-O, 실 trace 재생(R-1, 결정 42), 단일 rate 근사 일반화.

## 1. 선행 조건

| # | 선행 조건 | 상태 | 막는 것 |
|---|---|---|---|
| Q-1 | **W3 = arXiv-Sum 입력(≤16,384 필터, Nano 토크나이저 p50 6,484 [측정, `COMPARE_shapeC_workloads_2026-10-20.md`]) × 자연 출력**(`ignore_eos` 강제 없음; 자연 출력 평균 203 [측정, 같은 문서]) [결정 47]. shape C(강제 긴 출력)는 V-O 2차. G0·O-V2 의존 없음. | 확정 | — |
| Q-2 | VESSL 볼륨에 ShareGPT(`/data/hf/raw/ShareGPT_V3_unfiltered_cleaned_split.json`, 비용 항·N3 등에서 사용 [측정]) 존재 확인(다른 캠페인에서 사용된 이력으로 존재 추정) · **arXiv-Sum 입력 적재 확인만 남은 선행 조건**(확인 안 됨). 없으면 `stage_data.sh` 적재(CPU Workspace, 사용자 승인). | W3 데이터 미확인 | Job-CAP2·H-D |
| Q-3 | **동적 arm의 Nano 실현 가능성(O-H3와 같은 엔진 질문)**: reactive bind+GATE(`PDMUX_SLO_SCHED=1`·`PDMUX_SLO_MODE=binding`·`PDMUX_SLO_FEAS_GATE`)와 다중 division 부팅이 Nano(nemotron\_h)·flashinfer·cudagraph ON·VESSL에서 부팅하고 전환(`switch_count`)이 기록되는지 engine-porter가 확인한다. 이 컨트롤러는 Zamba2-2.7B·triton 위에서만 돌았다 [측정 이력]. 실패 시 (B)의 동적 arm은 `INFEASIBLE_BY_RULE`, 비교는 static 격자만 남고 §6 표 R-4에 공시한다. | 미확인 | Job-TUNE |
| Q-4 | 하네스(§9) 구현·CPU 시험·변이 → **하네스층 감사 1회** → 봉인. | 대기 | 봉인 |
| Q-5 | **PHASE 계측 OFF 확인**: 이 PREREG의 모든 측정 run은 PHASE\_EVENTS·chunk/drain 계측 OFF(정책 지표를 관측자 비용 없이 얻는다. 진단 PREREG D-3이 ON/OFF 짝으로 관측자 비용과 drain 몫을 따로 잰다). decode probe(I-4)도 OFF. | 규칙 | 봉인 |
| Q-6 | 용량 측정 결과 addendum: **Job-CAP1·CAP2 회수 → λ\*\_best-static(W1)·λ\*\_best-static(W3) 확정 → rate addendum 봉인**(§4-3) → 그 뒤에만 Job-TUNE·HE0·H-D 제출. V\_REG addendum과 같은 방식(구조는 데이터 전에 봉인, 용량에서 나온 수치만 addendum). | 대기 | 후속 Job |
| Q-7 | 기판: VESSL A100 SXM 80GB 1장, Nano-9B-v2-Base(rev `dc0661c…`, 봉인 시 전체 해시), cudagraph ON, radix OFF, `--mem-fraction-static` 0.6(결정 13), `--max-running-requests 48`, `--context-length` 20,480. **엔진 digest = 최종 트리 하나, 진단 PREREG와 같은 값** — `policy_harness/ENGINE_DIGEST.txt`(자리표시 `PENDING_FINAL_TREE`; `ph_job.sh`는 이 값이면 `ABORT_ENGINE_DIGEST_PENDING` rc 13으로 거부, `ph_digest make`도 거부). 비용 항 값 `99bbef91…`는 이 Job의 엔진이 아니다(계측 커밋 96ae3afa 이후 트리). caveat(인용 금지 형태): "PHASE·diag 계측 OFF 경로가 비용 항 트리와 동일하게 동작한다는 것은 정적 확인이며 측정되지 않았다. 비용 항(99bbef91) 수치와 이 Job의 수치를 결합하지 않는다." | [봉인 시 기입: 최종 트리 digest] | 봉인 |

## 2. 공통 규약

- **판정 술어(게이트 4)**: goodput = 요청 TTFT ≤ SLO ∧ **요청 내부 토큰 ITL p95** ≤ SLO. mean-ITL은 secondary만. **원 KISTI 하네스(`sharegpt_vary_bench.sbatch:88,92`)는 mean-ITL로 채점했다**(CONSENSUS §1-7 追記 2026-08-23) — 이 PREREG는 p95 술어이므로 KISTI HE0와 **채점 술어도 다르다**(별도 claim의 이유 하나).
- SLO(W1 ShareGPT): TTFT 3 s, ITL(요청 내부 p95) 60 ms — KISTI HE0의 정본 SLO 점 승계 [제안]. SLO(W3): TTFT = 3 × floor(같은 Job 첫 블록의 단일 요청 TTFT 중앙값, N3 규칙 승계), ITL 60 ms [제안]. SLO를 바꿔 재평가하지 않는다(게이트 8: 다른 SLO는 재튜닝해 직접 측정).
- goodput = Σgood / Σduration(**duration은 합산**, 게이트 7: 라운드·phase 합산에 `max()` 금지).
- 동일 trace hash·workload seed·server seed(Job 내 모든 arm), 실행 순서 randomize(boot 단위 블록 무작위, 계획 해시에서 파생, 봉인 시 기입).
- cudagraph(운영점 ON), backend(flashinfer), KV capacity, max-running 고정. GPU clock/power는 VESSL에서 고정 불가 — `clk.csv`·`substrate.json`·`clk_summary.json` 기록, arm 간 동일성 검사. **대조 arm은 같은 Job 안.**
- warm-up(60 s) / correctness(첫 요청 출력 길이 검사) / benchmark phase 분리.
- 게이트 5: 모든 arm에 `switch_count`(static은 0)·split 체류 분포·TTFT/ITL p50/p95/p99 병기(§5의 표).
- 게이트 6: 어느 arm·phase든 TTFT가 SLO ± 5% 안에 드는 요청 비율 > 10%면 `CLIFF_ADJACENT`. 그 arm·phase는 해당 라벨에서 cliff 규칙(§3-3, §5-3)을 따른다.
- 비교의 짝 = **같은 seed**. CI는 **seed 수준 짝 비의 t(n−1) 95% 구간**이며 **percentile bootstrap은 쓰지 않는다**(정본 §3 항목56: n = 4 재표집 공간 35개 퇴화). "rep SD" 문구 금지.
- seed 수준 반복은 **여러 Job**에 나눈다. hostname·GPU UUID는 공변량으로 남긴다.

## 3. (A) H-D — 설계 rev4 §2 승계

### 3-1. 승계하는 것 (rev4 §2-1…§2-3 그대로)
- trace: **W1**(짧은 대화, ShareGPT) / **W3**(긴 입력) phase 교대.
- rate: `rate_W = q × λ*_best-static(W)`, **q = 1.0 (1차)**(결정 33 O-H4).
- 두 변형: **깨끗한 시작 변형**(phase 블록 앞 drain: 도착 0, running + queue = 0까지; 120 s·600 s 둘 다) · **연속 변형**(이월 있음; **600 s만**, 결정 33). 연속 변형은 기술값이다.
- arm: **static D {16, 24, 34, 44, 64}**(결정 32). 5 arm.
- 선택 단위 = phase 종류(W1, W3), 인스턴스별 argmax 금지.
- **seed 8 교차적합**(결정 32 (a)): seed 1–4 선택/5–8 평가 후 역할 교환, CI는 평가 쪽 seed 수준 짝차 `h_s = ln(G_phase-oracle,s / G_best,s)`의 t(n−1) 95%, 두 역할 CI 중 **더 넓은 것**을 라벨에 쓴다.
- `H = ln(G_phase-oracle / G_best)`. 판정표(`DYNAMIC_HEADROOM_PRESENT` / `_ABSENT` / `HEADROOM_UNRESOLVED`)·cliff 규칙(r2 N-1)·다중성 비대칭 공시(r2 N-5)·함정(phase 경계 이월 warm 창 phase 길이의 10%, duration 합산, LO 절단)을 rev4 §2-2·§2-3 그대로 적용한다. 판정표 조건: PRESENT = 깨끗한 시작 변형에서 교차적합 H ≥ ln 1.03 ∧ 하한 > 0, **두 phase 길이 모두**(q = 1.0에서); ABSENT = 두 길이·**모든 q** 상한 < ln 1.03; 그 외 UNRESOLVED.
- **q = 1.0만이므로 이 Job에서는 ABSENT를 선언할 수 없다**(rev4 r2 N-2: ABSENT는 모든 q 조건). 결과가 `HEADROOM_UNRESOLVED`면 q = 0.8 추가 Job을 사용자 결정으로 요청한다(§7-3의 조건부 Job).
- 쓰임(결정 31): PRESENT는 HyMux PREREG 착수 허가일 뿐 성능 주장이 아니다.

### 3-2. 이 문서가 더하는 것
- **phase 사이클 수 = 1 (W1 한 블록 + W3 한 블록)** [결정 48]. **사이클 내 반복 없음을 공시한다**(run 안에서 phase 종류당 인스턴스 1개). 반복은 seed 8 교차적합이 담당한다. 연속 변형의 경계는 한 번뿐이며 시작 phase는 **seed 홀수 = W1 먼저, 짝수 = W3 먼저**로 교차한다(seed 1–4와 5–8 각각 2:2). (참고) 사이클 2는 H-D 비용이 약 34.5 → 약 66 GPU-h가 된다 [유도, §7]. 잃는 범위는 §6 표 R-5.
- **phase 길이 O-H1 = {120 s, 600 s} [미정→초안 채택]**. 120 s 변형은 깨끗한 시작에서만, 600 s는 깨끗한 시작과 연속 둘 다.
- **시간 척도 공시(변화 trace 감사 허용 요구, 모든 H-D·HE0 셀 결과 문장에 붙임)**: "HE0 phase는 약 17 s/67 s(도착 기준), H-D phase는 120 s/600 s이며 둘 다 실 trace 시간 단위 변화보다 짧다." 실 trace 재생은 이 Job에 없다(결정 42).
- HE0 범위 문장(감사 허용 문장, 문자 그대로): "HE0는 KISTI A100·Zamba2-2.7B·cudagraph-ON에서, ShareGPT rate 3↔12 교대(구간당 200요청, 도착 약 67 s/17 s, 구간 사이 drain, 3라운드)와 같은 계열의 mix-스윙 trace로 관측한 *달성된* reactive 동적 제어 계열의 결과다. 분–시간 척도의 실 trace, 중간 rate·혼합, 다른 기판·모델로의 일반화는 측정되지 않았고, HE0 run에는 phase 계측이 없어 lockstep 정지의 기여는 판단할 수 없다."
- **fused\_tuned 제외(결정 43)**, **전환 가능 구성 대조 arm(O-H3)은 "선택 블록, engine-porter 확인 전 미포함"**: Q-3의 다중 division 부팅 확인이 나오기 전에는 이 PREREG에 포함하지 않는다. 확인 후 포함하려면 별도 addendum(새 Job 블록, 같은 Job 안에 arm으로 추가)이고, 이 초안의 비용에는 없다. 잃는 범위: §6 표 R-1·R-2.

### 3-3. 용량 측정 블록 (Q-6, Job 앞에 명시 — rev4 §2-4가 "비용 미포함"으로 둔 부분)
- 목적: `λ*_best-static(W1)`, `λ*_best-static(W3)` = 각 workload에서 static D {16, 24, 34, 44, 64} 중 가장 큰 SLO 용량.
- 용량 정의: N3(`../n3_capacity/PREREG_N3_CAPACITY_2026-10-04.md`) 승계 — 격자 rate 중 attainment ≥ 90%인 최대 rate, 구간(`CAPACITY_BRACKETED`)과 절벽 구간 병기. 정상성 검사 S1–S3 승계.
- 격자: 기하 열 (공비 ×1.35 [W1], ×1.5 [W3]), W1 10점·W3 6점, 범위는 비용 항 모델의 T1/T2 예측에서 잡는다 [미정 — 봉인 전 기입; 근거 없이 임의로 두지 않는다]. 셀당 요청 수 W1 120, W3 48. seed 2개(용량은 입력이고 판정이 아니므로 n = 2 [제안]; 용량 구간의 폭이 후속 rate 선택의 해상도다).
- **q = 1.0의 rate = 격자 점(보간 금지)** — 용량 구간 하단의 격자 점을 쓴다 [제안]. 용량의 절벽 구간에 걸리면 `CLIFF_ADJACENT` 라벨이 후속 Job에서 규칙대로 작동한다.
- Job: Job-CAP1(W1), Job-CAP2(W3). 각 Job 안에서 5 D가 모두 있으며(같은 Job 같은 노드) 용량 비교는 Job 안에서만 한다.
- 이 블록의 비용은 §7 합계에 **포함**한다.

## 4. (B) Nano HE0 핵심 셀 (결정 44)

### 4-1. KISTI 구조 이식 (`../slo_sched/sharegpt_vary_bench.sbatch` [코드])
- **트레이스**: 구간당 200 요청, LO 구간 → HI 구간 교대, **3 라운드**, 구간 사이 drain(구간마다 `bench_serving`을 따로 실행해 구간 사이가 비워짐), ShareGPT(`--sharegpt-context-len 4000`). NP = 200, ROUNDS = 3 [코드, `:35`].
- **달라지는 것(별도 claim의 이유, 공시 의무)**: 모델 Zamba2-2.7B → Nano-9B-v2-Base · backend triton → flashinfer · 기판 KISTI → VESSL · mem 0.82 → 0.6 · 채점 mean-ITL → 요청 내부 p95 · rate 3/12 → 용량 환산 · 컨트롤러 재튜닝 · `--chunked-prefill-size -1`·`--disable-overlap-schedule` 등 서버 플래그는 Nano 비용 항 PREREG의 부팅 규약을 따른다(KISTI 플래그 그대로가 아님, 봉인 시 목록 기입).
- 동적 arm의 SLO env: `PDMUX_TTFT_SLO_MS=3000`, `PDMUX_TPOT_SLO_MS=60` — 채점 SLO와 같다(SLO 일치).

### 4-2. rate 환산 (결정 44)
- **LO/HI 비 4× 유지**: KISTI rate 3/12.
- 정본에서 찾은 원 위치(근거):
  - 원 HI(rate 12)는 **과부하 phase**다. CONSENSUS §1-13: 정책 간 spread가 LO 0.067(2.3%) 대 HI 1.187(43%)이라 차별의 약 95%가 과부하 phase에서 발생. §1-19: "이 region은 OVERLOAD서만 존재 … 용량 이하 = 전부 통과·이상 = 전부 실패".
  - 원 LO(rate 3)는 **용량 이하**다. §1-13: LO는 split에 무관심(d16 2.861 ≈ d44 2.858 req/s ≈ 도착률 3의 약 95%).
  - **원 HI/λ\*의 정량적 배율은 정본에서 찾지 못했다**(KISTI Zamba2-2.7B의 SLO 용량 λ\*가 N3류 정규 측정으로 등록된 적이 없다 — 게이트 #6 "λ\*\_SLO 정규화 격자 미등록"). [미정]
- 환산 규칙: `rate_HI = h × λ*_best-static(W1)`, `rate_LO = rate_HI / 4`. **h = 1.5 [결정: 감사 r1 권고 수용, O-P3 닫힘; caveat: 원 KISTI HI/λ* 배율은 미상이라 한 점이다]** (HI가 용량 위 = 과부하이면서 cliff 경계 TTFT≈SLO에 서지 않도록 용량의 절반 위; LO = 0.375 λ\* 는 용량 이하). h를 바꾸는 sweep은 하지 않는다(잃는 범위 R-6). cliff는 §2 규칙으로 검사한다.
- 구간당 요청 수가 고정(200)이라 phase 길이는 rate에 반비례한다: LO = 200 / rate\_LO s, HI = 200 / rate\_HI s. 이 길이는 용량 측정 뒤 addendum에 기입한다.

### 4-3. 튜닝 절차 (게이트 8 — 데이터 전에 등록)
- **튜닝 대상**: reactive bind+GATE(`PDMUX_SLO_MODE=binding` + `PDMUX_SLO_FEAS_GATE`). 튜닝 노브 = [코드: `multiplexing_mixin.py:988-1142`] `PDMUX_SLO_ANCHOR_IDX`(앵커 division 인덱스), `PDMUX_SLO_DWELL`(최소 체류, 기본 3), `PDMUX_SLO_FEAS_OCC`(0.85)·`PDMUX_SLO_FEAS_MARGIN`(0.9) 중 선택. 초안 격자 = ANCHOR\_IDX ∈ {2, 3, 4} × DWELL ∈ {3, 6} = **6 구성** [제안; 인덱스↔D 매핑(Nano의 {16, 24, 34, 44, 64}에서 어느 D인가)과 노브 목록은 engine-porter가 Q-3에서 확인해 봉인 전 기입, 미정].
- **튜닝 트레이스 = 평가와 같은 HE0 셀**(같은 rate·구조) 이지만 **튜닝 seed 101·102는 평가 seed 1–8과 겹치지 않는다**. 튜닝 run은 Job-TUNE에서만 돈다.
- **선택 규칙(데이터 전)**: 구성별 두 튜닝 seed의 TRUE goodput(duration 합산) 평균이 최대인 구성. 1% 이내 동률이면 `switch_count`가 적은 쪽. 결과(구성·튜닝 goodput·switch\_count)는 addendum에 기입해 봉인하고 평가 전에 바꾸지 않는다.
- **SLO는 채점 SLO(3 s/60 ms)와 일치**하게 둔다(튜닝 노브가 아님).
- **비대칭 공시**: 동적 arm은 독립 seed로 튜닝되지만 static D는 평가 seed 위에서 사후 best를 고른다(5개 중 최댓값의 낙관 편향이 static 쪽에 있다). 따라서 "동적 > best static" 결론에는 보수적이고 "동적 ≤ static" 결론에는 관대하다. 이를 막기 위해 비교는 두 가지로 한다: (a) **사전 지정 static D44**(KISTI HE0 승자 D, 데이터 전 지정) 대 동적, (b) **사후 best static** 대 동적. 두 비교 모두 보고하고 어느 쪽이 라벨에 쓰이는지는 §5-2에 정했다.
- 튜닝 Job의 static arm은 두지 않는다(비용). 잃는 범위: 튜닝 시점의 동적 대 static 비교 없음.

### 4-4. arm과 반복
- arm 6개: static D {16, 24, 34, 44, 64} + bind+GATE(튜닝 구성). fused\_tuned 없음(§6 R-1).
- n = 4 seed(seed 1–4, 2 Job × 2 seed) — 판정 하한 n ≥ 4(게이트 3). 운영 목표 n ≥ 5이고 CI가 0을 가로지르거나 분산이 크면 10+(§7-3 조건부 확장 Job).
- 한 Job = 6 arm × 2 seed. static arm은 boot 1개에서 두 seed를 무작위 순서로, 동적 arm은 컨트롤러 상태가 boot에 남으므로 seed마다 재부팅(boot 7개/Job).

## 5. (B)의 판정

### 5-1. 지표
phase별(LO/HI)·라운드 합산 TRUE goodput(게이트 7), 전체 goodput. TTFT/ITL p50/p95/p99, `switch_count`, split 체류 분포(동적), 요청 내부 ITL p95 > SLO인 요청 비율, `CLIFF_ADJACENT` 플래그.

### 5-2. 짝 비교와 라벨
- 기준 쌍: 동적 대 (a) D44 · (b) 사후 best static(평가 seed 평균 최대 D). 각 쌍에서 `d_s = ln(G_dyn,s / G_ref,s)`, seed 4개, t(3) 95% 구간(t = 3.182).
- 라벨(여유 ln 1.03 — 3% 미만은 headline 아님, 게이트 3):

| 라벨 | 조건 |
|---|---|
| `DYN_ABOVE_STATIC` | (a)와 (b) **둘 다** 구간 하한 > ln 1.03 |
| `DYN_BELOW_STATIC` | (b) 구간 상한 < −ln 1.03 |
| `DYN_EQUIV_STATIC` | (a)와 (b) 구간이 모두 (−ln 1.03, ln 1.03) 안 |
| `HE0CELL_UNRESOLVED` | 그 외, 또는 동적 또는 기준 arm이 `CLIFF_ADJACENT`(cliff 규칙: 라벨을 `UNRESOLVED`로 내리고 cliff 밖 rate에서의 재측정은 사용자 결정) |

- `DYN_ABOVE_STATIC`에 (b) 사후 best를 요구하는 것은 사후 선택 편향이 static에 유리하기 때문(보수적). `DYN_BELOW_STATIC`은 (b)만 요구하는 것이 관대하다는 점을 §4-3 공시와 함께 쓴다 — 결과 문장에 (a)(b) 둘 다 표기한다.
- 동적 arm이 `INFEASIBLE_BY_RULE`(Q-3 실패)이면 라벨 없음, static 격자 순위만 기술값.
- 허용되는 결과 문장은 이 PREREG §8의 틀뿐이다.

### 5-3. cliff와 용량
rate가 용량 환산이라 HI에서 TTFT가 SLO 근방에 놓일 수 있다. `CLIFF_ADJACENT`를 phase별·arm별로 보고하고 위 라벨 규칙을 적용한다. 임계 사다리(두 SLO 각각에서 ± 한 단계)는 이 PREREG에서 재측정하지 않는다 — 필요하면 사용자 결정으로 별도 Job(SLO를 바꿔 평가할 땐 재튜닝해 직접 측정, 게이트 8).

## 6. 측정점 삭제 원칙 (결정 30 형식 — 데이터 전 전수표)

원칙(고정): (P) 물리적 불가 또는 (R) 더 강한 증거로 대체될 때만 삭제, (C) 설정·예산·순서 제약이면 별도 boot·보정으로 살리거나 잃는 범위를 공시. 사유가 예산이면 허용 사유가 아님을 같이 적는다.

| # | 측정점 | 이유 | 분류 | 상태 | 잃는 범위·대안 [유도] |
|---|---|---|---|---|---|
| R-1 | fused\_tuned arm(H-D·HE0 셀) | 결정 43. fused\_tuned는 G0 튜닝 argmax를 요구하고 G0 PREREG가 미작성(순서) | **(C) 순서·설정** — (P)·(R) 아님. 결정 43이 V-O 2차로 미룬다 | 삭제(2차) | 잃는 범위: (i) HE0 셀에서 PD-mux 대 fused 비교 없음(분모 없음) → "PD-mux 이득" 서술 불가. (ii) H-D의 phase-oracle 후보에는 원래 fused가 기술값이었다(rev4 §2-1) — 판정 H에는 영향 없음, 기술 참고값만 잃는다. 대안: V-O 2차(G0 뒤)에 포함, 비용은 7 arm 대비 약 1/5 증가 [유도] |
| R-2 | 전환 가능 구성 대조 arm(O-H3, 다중 division 부팅 D\* 고정) | 실현 가능성 미확인 | (C) 설정 — 확인 전 미포함 | **선택 블록, engine-porter 확인 전 미포함** | 잃는 범위: "전환 가능 구성 자체의 비용"을 못 잼 → 동적 정책이 static보다 지는 경우 비용을 구성 때문인지 정책 때문인지 분리 못 함. 확인되면 Job 안 arm 1개 추가(H-D: +1/5 비용, 약 +6.9 GPU-h; HE0 셀: +1 arm ≈ +1.2 GPU-h) [유도] |
| R-3 | q = 0.8 (H-D) | 결정 33 O-H4 | (C) 예산(허용 사유 아님) | 1차 미포함; `UNRESOLVED`일 때만 추가 | 잃는 범위: 1차 결과로 `HEADROOM_ABSENT` 선언 불가. 조건부 Job H-D-q08 약 34.5 GPU-h [유도] |
| R-4 | 동적 arm(Q-3 실패 시) | Nano에서 컨트롤러 실현 불가 | (P)가 될 수 있음(엔진 구조) — 실패 시 분류 확정 | 조건부 | 잃는 범위: HE0 셀의 동적 대 static 비교 자체 |
| R-5 | 연속 변형 120 s, phase 사이클 > 1 | 결정 33(연속은 600 s만) / 결정 48(사이클 1) | (C) 예산(허용 사유 아님) | 미포함 | 잃는 범위: 연속 변형 phase 120 s 기술값; 연속 trace의 W1→W3 경계가 run당 1회뿐이라 이월 편향 추정이 seed 간 교차(W1→W3와 W3→W1)로만 가능. 사이클 2: H-D 약 66 GPU-h [유도] |
| R-6 | HE0 셀 h 스윕(HI = 1.5 λ\* 한 점), LO/HI 비 4× 한 점, 다른 SLO(chat 300/50) | 결정 44가 4× 유지 한 점 | (C) 예산·결정 | 미포함 | 잃는 범위: 동적이 이기는 operating regime 탐색(정본 §1-19는 용량 이하 = static 자명). 한 점에서의 결과를 "Nano에서 동적은 못 이긴다"로 일반화하지 못한다 |
| R-7 | 튜닝 Job의 static 대조, 튜닝 seed 수 > 2 | 예산 | (C) 예산 | 미포함 | 잃는 범위: 튜닝 신뢰 구간. 튜닝 구성 선택이 2 seed에 의존 |
| R-8 | KISTI 설정(Zamba2-2.7B) 재현 | 결정 44의 별도 claim 원칙 | (R) 기판·모델 교락 | 삭제 | 잃는 범위: KISTI HE0와의 직접 비교. 이 PREREG의 결과는 KISTI HE0를 확인·반박하지 못한다 |
| R-9 | 실 trace 재생 R-1, V-O | 결정 42·43 | 순서 | 2차 | 별도 PREREG |

## 7. Job 구성, 시간, GPU-h [유도 — 측정 아님, Job 크기 결정용]

### 7-1. 시간 모형 (비용 항 PREREG rev3 §7-1 승계)
boot 200 s + warm-up 60 s = **4.3 min/boot**, 셀 오버헤드 15 s. 용량 셀: W1 (120 요청/rate + 30 s 꼬리 + 15 s) ≈ **85 s**(rate 평균 약 3 req/s 가정), W3 (48 요청/rate + 45 s) ≈ **3.4 min**(rate 평균 약 0.3 req/s 가정). HE0 셀 1 run(3 라운드) ≈ **15 min**(λ\*\_W1을 임의 3 req/s, h = 1.5로 둔 모형: LO 178 s + HI 44 s + backlog drain, 라운드 약 260 s, 오버헤드 포함). H-D 1 (arm, seed) = 깨끗한 120 s(2 phase × (120 + drain 20 s) ≈ 4.7 min) + 깨끗한 600 s(2 × 620 s ≈ 20.7 min) + 연속 600 s(20 min) + 변형 사이 2 min ≈ **47.5 min** + boot 4.3 min. **λ\* 의존 항은 용량 측정 뒤 갱신한다. 모든 값은 측정 전.**

### 7-2. Job 표
| Job | 내용 | 스펙 | boot | run/셀 | 예상 h | 하드캡(×1.2) | 사용률 |
|---|---|---|---|---|---|---|---|
| **Job-CAP1** W1 용량 | 5 D × (10 rate × 2 seed), D마다 1 boot | `resourcespec-a100x1` | 5 | 100 셀 | 2.72 | 3.3 | 142 min / 163 min = 87% |
| **Job-CAP2** W3 용량 | 5 D × (6 rate × 2 seed), D마다 1 boot | 〃 | 5 | 60 셀 | 3.78 | 4.5 | 204 min / 227 min = 90% |
| **Job-TUNE** 동적 튜닝 | 6 구성 × 2 튜닝 seed, **(구성, seed)마다 1 boot**(컨트롤러 상태가 boot에 남음, 감사 M3) | 〃 | 12 | 12 run | 3.86 | 4.6 | 180 min / 232 min = 78% |
| **Job-HE0-a** HE0 평가 seed 1·2 | 6 arm × 2 seed (static 5 boot + 동적 2 boot) | 〃 | 7 | 12 run | 3.50 | 4.2 | 180 min / 210 min = 86% |
| **Job-HE0-b** HE0 평가 seed 3·4 | 〃 | 〃 | 7 | 12 run | 3.50 | 4.2 | 86% |
| **Job-HD-1…8** H-D, seed 1…8 (Job당 1 seed) | 5 static arm × (깨끗 120 s + 깨끗 600 s + 연속 600 s), arm마다 1 boot | 〃 | 5 | 5 × 3 변형 | 4.32 | 5.2 | 237.5 min / 259 min = 92% |
| 합계 | 13 Job | | | | **약 51.8 GPU-h** | 약 62.2 | |
| 내역 | 용량 6.50 · 튜닝 3.86 · HE0 평가 7.00 · H-D 34.5 | | | | | | |

- 합계 산식 [유도]: 2.72 + 3.78 + 3.86 + 2 × 3.50 + 8 × 4.32 = 51.8 GPU-h (감사 M3: 재부팅 반영 +0.43). rev4 §2-4의 H-D 24–36 GPU-h(용량 측정 미포함)와 비교하면 H-D 본체 34.5는 그 범위 안이고, 용량 6.5가 더해진 비용이 "비용 미포함 문제"의 해소분이다.
- **같은 비교의 arm은 한 Job 안**: H-D(5 arm × 한 seed)·HE0(6 arm × 두 seed)·CAP(5 D)·TUNE(6 구성) 모두 비교 단위가 한 Job에 완결된다. seed는 Job에 따라 달라지므로 seed 효과와 Job 효과가 교락한다 — CI는 같은 Job 안 arm 쌍의 짝차로만 계산하고 Job을 공변량으로 병기한다.
- **Job당 ≤ 6 h**: 모든 Job 하드캡 ≤ 5.2 h(최대 H-D 5.2 h). 정지 선례(약 9.5 h 외부 정지)보다 충분히 아래.
- **사용률(GPU Job 적정 크기)**: 모두 `resourcespec-a100x1`(GPU 1장, 동시 사용 1장). boot 구간 외에는 서빙 중이라 사용률은 셀 시간 비율(86–92%). TP2·다중 GPU 항목 없음.
- 의존: CAP1 → (rate addendum) → TUNE → HE0-a/b. CAP1 + CAP2 → (rate addendum) → HD-1…8. TUNE·HE0는 W1만 쓰므로 CAP2를 기다리지 않는다.
- 이 Job들이 채우는 COSTMODEL 칸: **`미지정`**(정책 검증 칸 — 비용 구조 CM-1…CM-7 중 어느 칸도 채우지 않는다. 용량 측정 블록만 CM 칸 연결 가능 여부를 봉인 시 정한다).

### 7-3. 조건부 Job (이 초안의 합계에 포함 안 함)
- H-D-q08(q = 0.8): H-D가 `HEADROOM_UNRESOLVED`일 때 사용자 결정으로. 약 34.5 GPU-h [유도].
- 확장 Job HE0-c(seed 5·6): `DYN_*` 구간이 ±3% 경계에 걸리거나 CI가 0을 가로지를 때. 3.5 GPU-h [유도]. n을 데이터를 본 뒤 늘리므로 순차 검정이며 결과에 "n = 6 확장" 사실을 적는다(확장 규칙은 데이터 전에 여기서 등록).
- 전환 가능 구성 대조 arm(R-2) addendum.

## 8. 라벨과 금지 문안

라벨: `DYNAMIC_HEADROOM_PRESENT`·`_ABSENT`(이 Job에서는 선언 불가)·`HEADROOM_UNRESOLVED`·`CLIFF_ADJACENT`·`DYN_ABOVE_STATIC`·`DYN_BELOW_STATIC`·`DYN_EQUIV_STATIC`·`HE0CELL_UNRESOLVED`·`CAPACITY_BRACKETED`(N3 승계)·`INFEASIBLE_BY_RULE`·`DRAIN_TIMEOUT`.

**허용 문안 틀**: "VESSL A100-SXM4 · Nemotron-Nano-9B-v2-Base · flashinfer · cudagraph ON · &lt;SLO&gt; · &lt;trace&gt; 에서, &lt;라벨&gt;(짝 seed n = &lt;n&gt;, t(n−1) 95% 구간 …). 이 결과는 Nano/VESSL 별도 claim이며 KISTI HE0를 확인·반박하지 않는다. 시간 척도: HE0 phase 17/67 s, H-D 120/600 s, 둘 다 실 trace 시간 단위 변화보다 짧다."

**금지 문안**
- "Nano/VESSL 결과가 HE0를 확인했다(또는 반박했다)"·"HE0이 뒤집힌다/유지된다"·"HE0 범위가 이번에 새로 좁아졌다"·"HE0는 실 워크로드·일주기 변동에서도 성립한다"·"변화 trace는 실 trace를 모사한다"·"느린 변화에서는 동적이 이긴다/HE0는 느린 변화에서 무효다"·"실 trace는 수 시간 척도로만 변한다"·"drain 때문에 변화 trace는 동적(또는 static)에 유리하다"·"HE0는 lockstep의 산물이다"/"lockstep과 무관하다".
- H를 "동적 정책이 얻을 이득"으로 인용(상한 성질은 깨끗한 시작 변형·단조성 가정 아래에서만), `ABSENT`를 "동적 제어는 원리적으로 불가"·"HE0 천장이 닫혔다"로 확장(정본 §3 항목57 미해소), q = 1.0만의 결과를 ABSENT로 쓰는 것.
- 정본 §1-19의 per-static oracle +2.1%(KISTI)와 이 결과의 결합·비교.
- "PD-mux가 fused보다 낫다/못하다"(fused 없음), "동적 제어 N 기여"의 기전 귀속(정지·drain·positioning·entanglement 중 무엇인지 — 진단 PREREG 몫), "정지가 없어서/있어서 동적이 졌다".
- 동적 대 static 비교를 SLO를 바꿔 재스코어해 서술(게이트 8). mean-ITL을 주 지표로 쓰는 것.
- 3% 미만 차이를 headline으로 쓰는 것, n < 4 정책 결론, cliff 위의 arm에 라벨 부여.
- "Nano에서 동적이 졌으므로 PD-mux 정책 연구는 무의미"류 확장, 한 rate·한 h·한 SLO 점의 일반화.
- 튜닝 seed(101·102)의 결과를 평가 풀에 섞는 것. 튜닝 구성이 "최적 동적"이라는 서술(격자 6점·seed 2).
- Job 간(예: HE0-a와 HD-3) goodput 비교, 용량(Job-CAP)의 절대값을 KISTI 용량과 비교하는 것.

## 9. 신규 하네스 구현 항목 (감사 뒤 작성, 이 문서에서 코드 0)

① 용량 측정 러너(rate 격자·정상성 S1–S3·`CAPACITY_BRACKETED`; N3 `n3_plan.py`/`n3_analyze.py` 재사용 가능성은 감사 때 판정) ② H-D phase 교대 트레이스 생성기(W1/W3 블록, 깨끗한 시작 drain 규칙, seed 홀짝 시작 교차, `DRAIN_TIMEOUT` 처리) ③ HE0 셀 러너(KISTI 구조 이식: 구간당 200, 3 라운드, 구간 사이 drain; bind+GATE env 주입, `switch_count`·체류 수집) ④ 튜닝 러너·선택 규칙 구현 ⑤ 교차적합 분석(seed 분할·평가 pass·두 역할 CI 중 넓은 쪽) ⑥ 짝 seed 구간 분석(percentile bootstrap 경로가 없음을 시험) ⑦ cliff 검사(±5%, > 10%) ⑧ duration 합산 검사 ⑨ 계측 OFF 확인(Q-5). **변이 시험**: 평가 seed에 튜닝 seed를 섞음 · `max()` duration · bootstrap 구간으로 바꿈 · cliff 임계 변경 · 선택 단위를 인스턴스별 argmax로 변경 · LO/HI 비를 4× 아닌 값으로 — 전부 검출돼야 한다.

## 10. 봉인 절차

1. Q-2…Q-5 충족, 하네스 구현·CPU 시험·변이.
2. **하네스층 감사 1회**(규칙층 재감사 없음 — 이 PREREG는 rev4 §2 승계 + 위 추가분이라 새 규칙은 §4(B)와 §3-3에 한정).
3. 봉인(규칙·하네스·엔진·config digest, [봉인 시 기입] 칸, 장부류는 봉인 바이트 확정 뒤 1회).
4. Job-CAP1·CAP2 → `launch.sh` dry-run(명령·job name·`$/h` 표시) → **사용자 승인** → `--submit` → `fetch.sh`(DONE + `sha256sum -c`) → rate/구성 addendum 봉인 → 후속 Job 같은 절차. **이 문서 작성 시점에는 dry-run도 하지 않았다.**

## 11. 열린 결정 [미정] (rev0.2: 감사 r1 권고값 채택으로 대부분 닫힘)

**닫힘(결정 49 + 감사 권고 채택)**: O-P6 = 전 Job static arm 비sticky(`ph_spec.STATIC_STICKY = False`, 봉인 상수; caveat "static D는 동거 중 목표값이며 실현 할당이 아니다") · O-P3 h = 1.5(한 점) · O-P4 튜닝 격자 ANCHOR\_IDX {2, 3, 4} × DWELL {3, 6}, 매핑 1=D16…5=D64 · O-H1 phase 길이 {120 s, 600 s} · connector 한도 0(무제한) · warm-up 60 s · W3 `max_new_tokens` 1024 · `--context-length` 20,480 · 판정 여유 ln 1.03 · n = 4 · CAP seed 2 · W3 floor = **CAP2 상수**(HD Job의 자체 floor는 공변량 기록만) · 용량 bracket 실패 시 사다리 재배치·CAP 재실행 규칙(§3-3, 등록).
**열린 채 허용**: O-H3(전환 가능 구성 대조 arm, 미포함) · q = 0.8(조건부 Job) · O-P5 용량 사다리 하한 `cap_ladder.rmin`(비용 항 T1/T2 예측으로 CAP 전에 기입, 봉인 addendum).
**남은 선행**: Q-7 최종 트리 digest(`PENDING_FINAL_TREE`) · Q-3 GPU smoke(비sticky `R2_POLICY=fixed` 의미 확인 포함, engine-porter) · Q-2 arXiv-Sum 볼륨 존재 확인.

## 12. 하네스 구현 기록 (2026-10-28, 정적 확인 — 측정 0, 미봉인, 하네스층 감사 대상)

하네스 위치 `policy_harness/`(`ph_spec/ph_trace/ph_verdict/ph_analyze/ph_runner/ph_digest/ph_job.sh/ph_mutants`), 시험 `tests/test_policy_harness.py`. 이 절은 본문 규칙을 바꾸지 않고 [미정] 해소 근거와 새 열린 항목만 적는다.

- **Q-2 arXiv-Sum 적재(해소)**: 16k 사전 토큰화 파일이 VESSL 클러스터 볼륨 `/data/datasets/arxsum_x/arxsum_x_tokens_v1.bin`에 이미 적재돼 있다(M3 rev12 R-H7, 2026-10-16, sha256 `4c32f1af…888f` = `mg_const.X_TOKENS_SHA256`, 168,330,617 B, 로컬 sha와 원격 sha 일치 기록 `results/multigpu/RH7_tokenizer_upload_2026-10-16.md`). `vesslctl`로 현재 볼륨을 직접 조회하지는 않았다(기록 기준). `ph_job.sh`가 Job 시작 때 sha256을 재검사해 불일치면 `ABORT_DATASET`(rc 13). 파일이 없으면 적재 절차: 로컬 `mg_x_pretokenize.py build` 산출물을 `vesslctl volume upload`로 object 볼륨 `staging/`에 올리고 `pdmux-build` Workspace에서 `/data/datasets/arxsum_x/`로 복사 후 `sha256sum -c`(RH7 문서 §2 명령 그대로). 실행하지 않았다.
- **Q-3 / O-P4 bind+GATE 실현 가능성(정적 확인)**: (i) 배선: `PDMUX_SLO_SCHED=1`+`PDMUX_SLO_MODE=binding`+`PDMUX_SLO_FEAS_GATE`는 `multiplexing_mixin.py`(`adjust_stream_groups` → `_slo_decide_idx` → `_slo_decide_idx_binding` → `_slo_feasible`)에서 모델·backend 분기 없이 `update_decode_attn_backend(idx)`로 stream index별 decode attn backend 그룹을 바꾼다(`model_runner.py:2624`, 모델 무관). CUDA graph는 stream-group index별로 캡처된다(`f"{stream_idx}_{bs}"`, 같은 파일 주석). (ii) 분할 index: 비용 항 D64 후보 config(5 division)의 stream group = [(108,0), (92,16), (84,24), (74,34), (64,44), (44,64), (0,108)], `real_sm_group_num` = 7, 컨트롤러 범위 `_lo`=1, `_hi`=5. **ANCHOR_IDX↔D 매핑 확정: 1 = D16, 2 = D24, 3 = D34, 4 = D44, 5 = D64**(기존 KISTI `pdmux_slo.yml`은 5번째가 (54,54)=D54라 D64 없음 — 이 Job은 D64 후보 config를 쓴다). 튜닝 격자 {2, 3, 4}는 D24/D34/D44, 기본 앵커 `(1+5)//2` = 3(D34). (iii) 제약: `PDMUX_STICKY_PARTITION`+`PDMUX_SLO_SCHED`는 엔진이 `RuntimeError`로 거부한다 → 동적 arm은 sticky가 아니다. (iv) **미확인(정적으로 알 수 없음)**: Nano(nemotron\_h·mamba 상태)·flashinfer·현 엔진 트리에서 실제로 부팅·전환·`switch_count`가 기록되는지, 5개 index × bs 캡처 목록으로 부팅 시간이 늘어나는지(boot 4.3 min 모형은 단일 index 기준). 결론: **`INFEASIBLE_BY_RULE` 근거는 정적으로 발견되지 않았다. 실현 가능성은 GPU smoke(Workspace `pdmux-probe`, 비측정)로만 닫힌다** — 선행 조건 Q-3은 "정적 확인 완료·실행 확인 대기"로 갱신한다.
- **O-P6 (rev0.2에서 닫힘, 결정 49)**: ~~static arm의 sticky 여부~~ → 전 Job 비sticky. 이하는 rev0.1 당시 서술이다: static arm의 sticky 여부. 동적 arm은 sticky일 수 없으므로(위 iii) 동적과 짝을 이루는 static이 sticky이면 "static 대 동적" 비교에 sticky 유무가 섞인다(KISTI 원 static d44는 단일 division·비sticky였다). 하네스 기본값은 비용 항 PREREG에서 검증된 경로(`PDMUX_R2_POLICY=fixed` + sticky)이고 addendum의 `static_sticky`로 바꾼다. CAP 용량·TUNE·EVAL·HD 모두 같은 값을 써야 한다(용량 환산 rate가 그 static 모드에서 나온다). 선택은 사용자 결정 + engine-porter의 비sticky fixed 의미 확인.
- **구현 중 정한 [제안] 값(감사에서 확인)**: W3 natural-output 상한 `max_new_tokens` = 1024; W1 요청은 출력 길이 고정(`ignore_eos`, ShareGPT 응답 길이; **KISTI와 같다** — 감사 m3: `sharegpt_vary_bench.sbatch`에 `--disable-ignore-eos`가 없고 bench_serving 기본이 ignore_eos 사용이다. 편차 아님); W1 길이 표는 Job 시작 때 서빙 모델 토크나이저로 ShareGPT 첫 turn 쌍에서 생성(`input+output ≤ 4000`); 서버 `--context-length` 20,480; H-D 단계 goodput = 그 phase에 도착한 양호 요청 수 / 명목 phase 길이, HE0 goodput = Σ양호 / Σsegment 실측 duration(게이트 7); H-D `G_best`는 평가 seed에서 고른다(보수적); `DRAIN_TIMEOUT`은 표지만(replay가 timeout을 받지 않음); 용량 격자 하한 `cap_ladder.rmin`은 addendum에서 받는다(O-P5).
- **공유 코드 변경 없음**(rev0.2 정정, 감사 m2): rev0.1의 `trace_loadgen.py` 수정은 `4aff6b30`에서 원복했다(봉인 multigpu rev12, sha `8da91574…`). W3 자연 출력은 `policy_harness/ph_loadgen.py` 래퍼(원 `replay` 호출, `ignore_eos` payload만 조정)가 맡고 `abs_mono_loadgen.py`와 조합된다.


## 13. rev0.2 수리 내용과 감사 r1 → 수리 위치 대조표

### 13-1. 새로 등록한 규칙 (감사 수리가 만든 데이터 전 등록)
- **용량 인정 조건(D4)**: argmax D의 용량이 `CAPACITY_BRACKETED`(실패 rate가 실제로 존재)일 때만 λ\*로 받는다. 사다리 최상단까지 통과(`CAPACITY_NOT_BRACKETED`)면 λ\* = None, 사다리를 위로 이동(`rmin' = rmin × factor^(n−1)`)해 CAP를 재실행한다(재실행은 새 job name, 사실을 결과에 적는다). 통과 rate 없음(`NO_PASSING_RATE`)이면 아래로 이동(`rmin' = rmin / factor^(n−1)`). λ\* = None에서 `he0_rates`는 `ABORT_NO_LAMBDA`.
- **정상성(M6)**: S1 = 도착 순 마지막 1/3 대 처음 1/3의 TTFT 중앙값 비 > 1.7 → `UNSTABLE`(N3 S1 임계 승계) · S2 = 마지막 1/3 attainment가 처음 1/3보다 0.20 넘게 낮음 → `UNSTABLE` [제안] · **S3(runtime\_snapshot 기반)는 구현하지 않는다**: PHASE 계측이 필요한데 Q-5가 금지한다. 결정 30 표 행: 잃는 범위 = 실현 분할 혼합 기반 정상성 검사; 분류 (R/C) = 계측 OFF 규약이 우선. `UNSTABLE` 용량 셀은 크기 인용 불가로 표시한다.
- **trace 동일성(D2·M7)**: 요청 내용(도착·길이·문서)은 (workload, seed, rate/variant)만의 함수이며 arm·run\_id와 무관하다. `trace_hash`는 내용 해시(request\_id·segment\_id 제외)이고 분석은 같은 seed의 arm 간 hash 일치를 대조해 불일치면 `ABORT_TRACE_IDENTITY`.
- **correctness·warm-up(M5)**: boot마다 W1 2 req/s 60 s warm-up(판정 제외); 고정 길이(`ignore_eos`) warm-up 요청은 요청한 토큰 수와 정확히 일치해야 하며 하나라도 어긋나거나 오류면 boot `INVALID_CORRECTNESS`(run 미실행, Job rc 17).
- **로그 자르기(M2)**: 동적 arm의 `switch_count`·체류는 그 run이 실행되는 동안의 서버 로그 구간만으로, 체류는 run 시작 시각의 index에서 시작해 계산한다. 비sticky static의 체류는 "목표값이며 실현 할당이 아니다"(m7).
- **CI 정의(m1)**: H-D 교차적합의 "더 넓은 CI"는 두 역할 CI의 **포락(envelope: 가장 낮은 하한, 가장 높은 상한)** 으로 등록한다 — UNRESOLVED 쪽으로만 기운다.
- **HE0 라벨 지표(m4)**: 라벨은 LO + HI 합산 전체 goodput(Σ양호/Σduration)으로 매긴다. LO·HI 단계별 짝 구간(`phase_ci`)은 기술값으로 출력하고 라벨에 쓰지 않는다. static arm 5개 + 동적이 모두 없으면 `he0_compare`가 거부한다(m9).
- **`DRAIN_TIMEOUT`(m5)**: H-D 깨끗한 시작 변형에서 segment의 실측 duration − 명목 phase 길이 > 120 s이면 표지(`DRAIN_TIMEOUT`)만 붙이고 값은 쓴다(replay에 timeout이 없다). 표지가 있는 phase는 결과 표에 병기한다.
- **H-D 게이트 5·연속 변형(m6)**: arm별 TTFT/ITL p50/p95/p99와 가상 phase-oracle의 전환 수(사이클당 W1·W3 선택 arm이 다르면 1)·체류를 출력한다. 연속 변형은 phase 첫 10%를 제외한 H를 제외 전 H와 함께 보고한다(기술값).
- **W3 floor(M4)**: SLO의 floor는 CAP2 Job의 봉인 상수다(`§2`의 "같은 Job" 문구는 floor에는 적용하지 않고 공시한다). HD Job은 자기 floor를 공변량으로 기록(`floor_covariate.json`)만 한다 — Job 간 W3 SLO 변동을 일으키지 않는다.
- **데이터 고정(m8)**: ShareGPT json sha256을 Job마다 기록하고, addendum의 `w1_lengths_sha256`이 있으면 Job 시작 시 대조(불일치 `ABORT_DATASET` rc 13).
- **H-D 요청률 가중 caveat(감사)**: "H-D의 H는 요청률 가중이라 W1 phase가 지배한다. λ 비 약 10:1이면 W3 비중 약 9%, W3 phase만의 여지가 PRESENT를 만들려면 약 33% 이상 [유도, §7 가정 λ 3/0.3]." — 모든 H-D 결과 문장에 병기한다.
- **BELOW 비대칭·귀무 caveat(감사 반전 시험)**: `DYN_BELOW_STATIC`이 (b)만 요구해도 귀무 하에서 승자 편향 ≈ 0.58σ < 구간 반폭 ≈ 2.25σ [유도]라 뒤집히지 않는다. h·동률 1%·ln 1.03은 원자료가 없어 봉인 전 상수로 둔다.

### 13-2. 감사 항목 → 수리 위치

| 감사 항목 | 수리 | 위치 |
|---|---|---|
| D1 엔진 digest 불일치 | `policy_harness/ENGINE_DIGEST.txt` 신설(`PENDING_FINAL_TREE`), `ph_job.sh`가 placeholder면 거부, `ph_digest`가 파일 포함·placeholder면 make 거부, Q-7 갱신 | `ph_job.sh`, `ph_digest.py`, `ENGINE_DIGEST.txt`, 이 문서 Q-7; 시험 `test_pending_engine_digest_refuses`, 변이 M50 |
| D2 CAP trace가 arm마다 다름 | 요청 샘플 키 = (workload, seed, rate) | `ph_trace.cap_trace`; 시험 `test_cap_trace_identical_for_every_arm`, 변이 M32 |
| D3 O-P6·배선 불일치 | `STATIC_STICKY = False` 봉인 상수, 전 plan·`arm_env`가 사용, `static_sticky` addendum 삭제 | `ph_spec.py`; 시험 `test_every_job_has_nonsticky_static_arms`, 변이 M34; 문서 §11 |
| D4 절단 용량 | bracket 요구, 재배치 규칙, `he0_rates` 거부 | `ph_verdict.lambda_star_best_static/next_ladder_rmin`, `ph_spec.he0_rates`, `ph_analyze.cap_addendum`; 시험 `test_truncated_capacity_is_not_accepted`, 변이 M36; 문서 §13-1 |
| M1 connector 상한 | `connector_limit=0` | `ph_loadgen.py`, `ph_runner.default_replay`; 시험 `test_connector_limit_is_unlimited`, 변이 M44 |
| M2 로그 전체로 switch 계산 | run별 byte offset 자르기·시작 index | `ph_runner.run_boot/parse_switches/idx_after`; 시험 `test_switch_count_is_sliced_per_run`, 변이 M45·M26 |
| M3 TUNE 재부팅 없음 | (구성, seed)마다 boot 12, 비용 3.86 h·하드캡 278 min | `ph_spec.plan_tune/HARD_CAP_MIN`, §7; 시험 `test_tune_grid_reboots_per_config_and_seed`, 변이 M35 |
| M4 W3 floor 다른 Job | CAP2 봉인 상수, HD는 공변량 | `ph_spec.plan_hd`, `ph_runner.measure_floor`, §13-1; 시험 `test_hd_first_boot_records_floor_covariate_only` |
| M5 correctness·warm-up | 60 s warm-up + 토큰 수 검사, `INVALID_CORRECTNESS` | `ph_runner.warmup_and_check`; 시험 `test_warmup_is_60s_and_correctness_gate`, 변이 M46·M47 |
| M6 정상성 | S1·S2 구현, S3 결정 30 행 | `ph_verdict.stationarity`, `ph_analyze.cap_addendum`; 시험 `test_stationarity_s1_s2`, 변이 M40·M41; §13-1 |
| M7 hash arm 의존 | 내용 hash + 분석 대조 | `ph_trace.trace_hash`, `ph_analyze.*_trace_problems`; 시험 `test_run_job_end_to_end_and_resume`, 변이 M33 |
| m1 더 넓은 CI | 포락으로 등록 | §13-1, `ph_verdict.crossfit_H` 주석 |
| m2 §12 trace\_loadgen 문구 | 삭제·정정 | §12 |
| m3 KISTI 자연 EOS | 정정(KISTI도 고정 출력) | §12 |
| m4 HE0 라벨 지표 | 전체 goodput 명시 + `phase_ci` | §13-1, `ph_verdict.he0_compare`; 시험 `test_he0_reports_phase_level_pairs_descriptively` |
| m5 DRAIN\_TIMEOUT | 의미 정의 | §13-1 |
| m6 H-D 게이트 5·warm 10% | `hd_gate5`, `exclude_warm` | `ph_verdict.py`, `ph_analyze.hd_labels`; 시험 `test_hd_gate5_has_oracle_switch_count`, 변이 M42·M43 |
| m7 static 체류 | 목표값 표지 | `ph_runner.run_boot`(`split_dwell_note`), §13-1 |
| m8 ShareGPT sha | Job별 sha 기록 + addendum 대조 | `ph_job.sh`, `ph_runner.main`; 시험 `test_w1_sha_mismatch_aborts_before_any_server`, 변이 M49 |
| m9 best-of-4·boot 실패 rc 0 | 5 static 요구, `exit_code` 17 | `ph_verdict.he0_compare`, `ph_runner.exit_code`; 시험 `test_he0_requires_all_five_static_arms`·`test_exit_code_flags_dead_boot`, 변이 M39·M48 |
| 빠진 변이: phase 완료 귀속·빈 ITL | 시험·변이 추가 | `test_hd_phase_attribution_by_arrival`(M37), `test_empty_itl_is_good_single_token_output`(M38) |
| 반전 시험 caveat·미정 닫기 | §11, §13-1 | 이 문서 |
