# Nemotron-H 사다리 N0–N6 — 선수관계(DAG)와 실행 정리 (2026-10-04)

> **지위**: 계획 문서다. GPU 0 · 새 성능 판정 0 · Claim 등급 변경 0.
> - 입력: `experiment_rationale_and_git_history_2026-10-04.md` §2(N0–N6 표), `reports/system_design_anchor_2026-09-30.md` §3·§8,
>   `reports/cost_model_design_space_2026-10-01.md` A10·A11, E2 rev3 + 이식성 판정서(2026-09-18), λ0 VESSL 결과·결과 판정서, 908623 판정서 §7.
> - 이 문서가 새로 확인한 코드·데이터 사실은 §0에 모았다. 각 사실에 근거 위치를 붙였다.
> - 판정 아님: 아래 "의존"은 설계 해석이다. N4 종결과 static arm 의미론(§1-2)은 claims-auditor 규칙층 감사를 거쳐야 정본에 오른다.

> ★**감사 반영(같은 날, claims-auditor `handoff-report/VERDICT_n4_dsem_rules_2026-10-04.md`)** — 아래 본문 중 다음은 **틀렸거나 좁혀졌다**. 충돌하면 이 배너가 이긴다.
> 1. **F1 `REFUTED`**: decode를 고정할 수 있는 SM은 config의 `manual_divisions`가 정한다. {16,24,34,44} 필터는 generic·hybrid의 결정 격자에만 걸린다. 넓은 SM 곡선은 엔진이 아니라 **config 변경**으로 된다. (0,108)은 sticky로 고정할 수 없다(긴급 108은 격리 없는 공유 상태).
> 2. **F7**: A11은 **최소 3모델**(적합 ≥2 + hold-out 1)이 필요하다. Zamba2 보존은 필요조건일 뿐이다.
> 3. **N1은 λ0R-8(iii)을 해소하지 않는다**(E2C-5). DAG의 N1 산출 "λ0R-8(iii) 판정"과 Q-b의 "N1 전까지 금지"는 철회한다. 부하점은 **절대 offered rate**로 등록하고 λ\*는 참조 라벨로만 쓴다(DS-4).
> 4. **3-a(sticky ON, decode 단독) ≠ 동거 구간 step 시간**(정본 2F9, DS-3). 예측기 입력으로 쓰려면 동거 대조가 필요하다. → N2에 **E-α(4-a 복원: 단독 vs 동거 paired)**를 포함한다.
> 5. **N4 문안 `REFUTED`** → 판정서 §1-3 허용 문안으로만 종결한다. "실질 D44"와 "3-a로 흡수"는 금지다. 16k–64k 입력은 **미측정 열린 항목**이다.
> 6. "Bullet 의미론" 문장은 Bullet 동적 시스템에만 해당한다. 우리 static ≠ Bullet §4.4 static(DS-5).
> 7. D-SEM(모든 arm sticky OFF)은 엔진 강제이므로 `CONFIRMED`다. N3를 N1과 독립으로 등록하는 것도 `CONFIRMED`다. N3 필수 조항 11개는 판정서에 있다. 특히 **ladder 공통 config**: `pdmux_homog5.yml`은 decode {16,44,92}라 R2 격자가 {16,44}뿐이므로, N3·5-a·5-b가 같은 `sm_counts`를 쓰도록 config를 먼저 정해야 한다.

## 0. 이번에 확인한 사실 (DAG를 바꾸는 것만)

| # | 사실 | 근거 | 함의 |
|---|---|---|---|
| F1 | R2 정책(fixed·generic·hybrid)은 decode 상태로 **{16, 24, 34, 44}만** 받는다. 긴급 상태는 108로 하드코딩돼 있다. | `src/multiplex/multiplexing_mixin.py:250-283` `_build_r2_policy` | sticky는 fixed 정책 전용이다(`:419-423`). 따라서 서빙 루프 안에서 decode를 고정할 수 있는 SM은 이 4개 + (0,108)뿐이다. 이것이 곧 예측기의 결정 격자(`allowed_decode_sms`)이므로 3-a 격자는 이 5점이면 S0′에 충분하다. 더 넓은 SM 곡선은 엔진 변경이 필요하다. |
| F2 | sticky OFF에서는 prefill이 없는 동안 decode가 (0,108)로 돌아간다. 셀 라벨은 "할당"이 아니라 "목표"다. | 같은 파일 `:358-372` 주석, λ0 VESSL §6 | **3-a는 sticky ON 없이 측정하면 Stage 0 D108 오류를 decode 축에서 되풀이한다.** N2는 sticky(또는 그에 준하는 고정)에 의존한다. |
| F3 | shape B도 저부하에서는 희석된다. D44 E-time이 b_r0(0.45 req/s)에서 VESSL 70.07%(KISTI 66.18%), b_r1에서 86.05%, b_r3에서 97.38%다. b_r0의 TTFT p95는 이미 6.9 s다. | λ0 VESSL `RESULT…` §6, `analysis_2026-10-04/lam0v_descriptive.json` | 게이트 #6(SLO 용량)이 보는 부하는 b_r0보다 낮다(908623 §7 E1a: 0.10–0.45). 그 영역은 희석이 더 크다. "shape B는 희석이 없어 E2와 무관하다"는 **포화 셀에서만 참**이다. |
| F4 | 엔진 소스는 `4d1a0ec`(E2 rev3 기준 트리)와 `1d47d3e`(λ0 VESSL)가 **바이트 동일**하다. 그 뒤 변경은 `0922512`의 `PDMUX_PHASE_EVENTS`(기본 OFF, 같은 줄 편집, `tests/test_phase_events.py`) 하나뿐이다. | `git diff --stat 4d1a0ec 1d47d3e -- workspace/engine-port/src/` = 공백 | E2-α는 두 arm이 같은 Job·같은 트리라 판정이 트리에 면역이다. 트리 일치가 필요한 것은 R6(908623 교차 비교)뿐인데, R6은 이미 "판정 아님"으로 강등됐다. |
| F5′ | ★**E2 R1의 OFF 절은 VESSL에서 이미 깨져 있다.** R1은 `a_r4`·`a_r2` 전 seed에서 OFF arm의 네 추정량이 **모두 < 20.0%**이기를 요구한다. 그런데 λ0 VESSL `a_r2`(seed 4386, sticky OFF = E2 OFF arm과 같은 조건)의 E-qcond가 **20.57%**다(KISTI 17.65%). `a_r4`는 15.94–18.33%로 통과한다. | `e2_label.py:34,182`, prereg rev3 §6 R1, `lam0v_descriptive.json` vessl/a_r2/mix | ★**정정(같은 날)**: ADDENDUM A5가 R1을 4분지로 이미 쪼개 두었다. 그래서 OFF 절 실패는 `OFF_BASELINE_ABOVE_BAND`(대조 arm에 대한 진술)로 나오고, ON arm에 대한 거짓 귀속은 생기지 않는다(`e2_label.py:172-196`). 실제 영향은 1차 라벨이 깨끗한 `STICKY_REALIZES_A`가 아니라 `OFF_BASELINE_ABOVE_BAND`(ON 절 통과 시)로 나올 가능성이 크다는 것이다. 이식성 판정서 항목 13 "미결"은 VESSL 파일럿 데이터로 **불리하게 해소**됐다. 처리 방안(문턱 유지 + 예보 공시 vs 재등록)은 A13에 공시하고 claims-auditor가 정한다. 문턱을 느슨하게 하는 쪽만 보지 말고 조이는 방향도 반전 시험해야 한다(교훈 251). |
| F5 | E2 이식 追記(A13)에 필요한 기판 상수가 λ0 VESSL 아티팩트에 이미 있다. | `lam0v_descriptive.json`, `RESULT…` §3·§6 | 들어 있는 값: a_r4/a_r2/b_r3 duration(133.62/142.90/289.67 s), OFF arm 4추정량(a_r4 4 seed, a_r2 1 seed — F5′ 참조), boot 시각. A13은 **GPU 0으로 작성 가능**하다. |
| F6 | VESSL 실측 소요는 KISTI 예산식의 약 1.36배다. λ0 등록 1.391 GPU-h에 대해 컨테이너 1.894 h가 걸렸다. | λ0 VESSL 결과 | 아래 예산은 모두 ×1.36으로 환산했다. 단가는 `resourcespec-a100x1` $1.48/h다. |
| F7 | A11(조성 효과 식별)은 **다른 조성의 모델 2개 이상에서 3-a 점**을 요구한다. 단일 모델 적합은 재매개화다. | `cost_model_design_space` A11, §4 | "Nemotron 우선"은 순서이지 범위가 아니다. 주장 조건 1의 "hybrid 때문"은 다른 모델의 3-a 없이는 서지 않는다. **Zamba2 가중치 삭제 결정과 직접 연결된다.** |
| F8 | S0′(A10)의 입력 스트림은 고정 정책 서빙 런의 `runtime_snapshot` 재생이다. 스냅샷은 32 sync마다 1개라 짧은 prefill span을 못 본다. | A10, mixin `PDMUX_PHASE_EVENTS` 주석 | N5는 N2(points)와 함께 **목표 워크로드의 고정 정책 telemetry**도 입력으로 요구한다. N3 런이 그것을 낼 수 있다. N3에서 PHASE_EVENTS를 켤지가 설계 결정이 된다(관측 효과 대조 필요). |

## 1. 선수관계

### 1-1. DAG

```
N0 기판·λ0 (완료)
 │
 ├──► N1 E2 sticky 대조 ──────────────┐ (sticky 노브의 서빙 검증: Q1 고정 유지·P13 correctness; λ0R-8(iii)은 해소 못 함 — 배너 3)
 │        │                           │
 │        │ [D-SEM: static arm 의미론 결정]◄── 사용자 + 규칙층 감사
 │        │        │                  │
 │        │        ▼                  ▼
 │        │   N3 게이트 #6 SLO 용량   N2 3-a decode 프로파일 (sticky ON, d∈{16,24,34,44}+108)
 │        │        │                  │     + L>65536 계측 셀 (별도 boot 구성)
 │        │        │                  │
 │        │        │                  ├──► N2′ 다른 조성 모델 3-a (A11용, ≥2모델)
 │        │        │                  │
 │        │        └──── telemetry ───┤
 │        │                           ▼
 │        │                    N5 S0′ 결정 발산(CPU) ──► A11 hold-out(CPU, N2′ 필요)
 │        │                           │
 │        ▼                           ▼
 │   5-a 최적 static 격차 ◄── N3 부하점        S1 pause manager + F1 훅
 │        │                                    (구현·CPU gate는 지금 가능, GPU ablation은 N5 뒤)
 │        └────────────► 5-b 변화 trace, n≥4 ◄──┘
 │
 ├──► N4 3-b λ*(B): λ0 shape B로 (8192,64) 점 충족 → 문서 종결만(GPU 0)
 └──► (선택) λ0 Job 2 노드 간 산포 ──► 5-a/5-b가 N3와 다른 노드일 때의 부하점 이식성
```

### 1-2. 실험별 입력·산출·외부 의존 상수

| # | 입력(선수) | 산출 | 설계 단계에서 다른 실험 결과에 기대는 상수 |
|---|---|---|---|
| N1 E2 | λ0 VESSL 상수(F5)<br>새 OVERRIDE | sticky ON의 D44 유지율(Q1)<br>같은 offered에서 achieved 변화(Q2)<br>P13 출력 동치(양 arm)<br>~~λ0R-8(iii) 판정~~(배너 3: 해소 못 함) | offered 8.0/3.0/1.15: λ0 rung이라 VESSL에서도 유효(a_r2는 1.002×λ\*로 이동, LVR-3)<br>T_cap·per-boot·WALL_BUDGET: **VESSL 값으로 교체 필요**<br>**R1 OFF<20: VESSL a_r2에서 이미 20.57%로 초과(F5′) → 실행 전 재등록 필수** |
| N2 3-a | sticky 노브 검증(N1의 Q1·P13 — 또는 N2 자체 셀별 실현 분할 검사)<br>배치 통제 하네스 | `points(d, bs, ctx)` → P_gen·P_gen+recal·P_hyb 공통 입력<br>1층 span 시간 vs decode step(L>65536) | d 격자 = 엔진 결정 격자(F1)<br>bs 상한 48(max_running = mamba 캐시, 908623 E3 교락 주의)<br>ctx ≤16384(λ0 구성), L>65536 셀은 `--context-length`가 달라 KV/mamba 예산이 바뀌므로 **별도 boot·별도 라벨** |
| N3 게이트 #6 | λ\*\_VESSL(있음)<br>**D-SEM**<br>SLO 값 등록 | 정본 술어 goodput의 SLO 용량, 셀당 n≥4 기준선 분산 → 5-a/5-b 부하점 | SLO 임계: **HE0 감사의 ITL p95 55–62 ms 절벽 회피 필수.** shape A a_r2 ITL p95가 이미 48 ms다.<br>부하 격자: λ\*\_VESSL 정규화 |
| N4 3-b | λ0 VESSL shape B | λ\*(B)≈0.690 (8192/64, D44 E-time 97%로 실질 D44) | — (긴 L 확장은 N2의 L>65536 셀과 prefill 길이 격자로 흡수) |
| N5 S0′ | N2 points<br>A4 규칙 상수(τ_age·분기 순서·"1 step")<br>고정 정책 snapshot 스트림(N3 또는 λ0 telemetry) | 예측기 3판 결정 발산 라벨 | 라벨 문턱은 실행 전 고정(A10)<br>`MODEL_DEPENDENT` 이상은 실측 points 위에서만 |
| A11 | N2′(다른 모델 3-a) | "조성이 중요하다" 성립 여부 | 모델 집합 = **사용자 결정**(업로드 범위) |
| S1 | anchor §4 계약(있음) | pause manager·F1 훅 + correctness gate | 없음 — 구현은 독립. ablation은 N5 예측기가 필요하다. |
| 5-a | N3 부하점 + D-SEM | 워크로드별 최적 static 격차(선택기 상한) | arm 정의가 D-SEM에 묶인다 |
| 5-b | S1 + N5 + N3 + 변화 trace(W2/W4) | 주장 조건 2·3 | n≥4, 짝지은 CI, 3% 문턱 |

### 1-3. 지난 세션이 남긴 세 의문에 대한 답

**Q-a. N1과 N2는 독립인가?**
- 판정량으로는 독립이다. 메커니즘으로는 아니다.
- N2는 F1·F2 때문에 decode를 고정하는 수단으로 sticky ON 말고는 서빙 안에서 쓸 방법이 없다. N1은 그 노브를 서빙 부하에서 처음 검증하는 등록 실험이다(Q1 유지율, P13 양 arm 출력 동치, cudagraph 재생).
- 단 N1은 d=44만 검증한다. N2는 d=16/24/34에서도 셀별 실현 분할 검사가 어차피 필요하다.
- 권고: 사전등록은 병행한다. 제출은 N1 결과 뒤에 한다(비용 약 $4, 리스크 대비 싸다).
- 별도로, 3-a를 decode 단독 부하에서 잴지 prefill 동시 부하에서 잴지가 N2 설계 질문으로 남는다. green-context는 SM만 나누고 메모리 대역폭은 공유한다. 결정은 prefill 압력 하에서 일어난다.

**Q-b. N3는 N1 결과를 기다려야 하나?**
- **D-SEM 결정에 달려 있다.**
- 5-a/5-b의 static·정책 arm을 모두 sticky OFF로 정의하는 경우(권고):
  - 근거: pause manager는 sticky와 조합 금지다(anchor §4). Bullet 의미론도 "prefill이 없으면 decode가 전 GPU"다.
  - 이 경우 N3는 현 운영점(sticky OFF)에서 바로 등록할 수 있다. 처치는 동거 구간의 분할이고 희석은 실세계 효과다.
  - N1은 "λ\*를 D44의 용량이라고 쓰지 말라"는 해석 제약만 준다.
- arm을 sticky ON으로 정의하는 경우: N3는 N1을 기다려야 한다. F3 때문에 shape B 저부하도 같은 문제다.
- 어느 쪽이든 "λ\*\_VESSL(A)로 arm 용량을 정규화"하는 서술은 N1 전까지 금지다(λ0R-8(iii)).

**Q-c. 3-b는 λ0 shape B로 대체되는가?**
- (8192,64) 한 점에서는 **대체된다**. λ\*(B)=0.690이고, 포화 셀의 D44 E-time이 97%라 사실상 D44 측정이다.
- 대체되지 않는 부분:
  - SLO 용량 — N3의 몫이다.
  - 8192보다 긴 입력 — ctx 16384 구성 한계 때문이다. 16k–32k와 65536 초과는 N2의 L>65536 셀과 prefill 길이 격자(`prefill_attn_ssm_length_review` §116)로 흡수한다.
- 문서 종결에 필요한 절차: doc-steward 반영 + claims-auditor 짧은 규칙층 확인.

## 2. 실행 정리

### 2-1. 실험별 준비 상태

| # | 사전등록 | 하네스 | 모델·데이터 | 엔진 트리 | 감사 순서 |
|---|---|---|---|---|---|
| N1 | KISTI rev3 + ADDENDUM 있음. **VESSL 이식 追記(A13) 필요**: P7 T_cap·per-boot·P14 WALL_BUDGET(SLURM `--time` 참조 → VESSL wall-clock 집행기)·R1 OFF 여유 재공시·E2C-21/E2C-6 수치 VESSL 재산출. 결정 규칙 R1–R6·seed·arm은 불변(이식성 판정서) | `e2_sticky.sbatch` → VESSL 러너로 이식. λ0 VESSL 패턴(digest 강제·wall-clock·resume·cgroup·clk) 재사용 | Nano-9B-v2·λ0 trace 적재 완료 | **HEAD 사용 권고**(PHASE_EVENTS OFF, F4). P4 manifest에 1파일 차이 공시, R6은 비판정 유지. 908623 동일 트리를 고집하면 `1d47d3e` 제출이지만 그 커밋에는 launch/job_entry 수정(밑줄·cgroup)이 없다 | A13 → claims-auditor 규칙층(이식) → 하네스층 → **새 OVERRIDE(사용자)** → dry-run → 승인 → submit |
| N2 | **없음(미설계)** | 신규: 배치 통제(동시성 = bs 고정, 출력 길이 고정), sticky ON 셀별 실현 분할 검사, 셀 = (d, bs, ctx) 1 boot 다수 측정 가능 여부는 engine-porter 확인 | Nano-9B-v2 있음. ctx 격자용 합성 프롬프트 필요. L>65536용 긴 입력 미확인 | HEAD | engine-porter(하네스·노브 범위) → 사전등록 → claims-auditor 규칙층 → 승인 |
| N2′ | 없음 | N2와 동일 | Zamba2 적재됨. Falcon-H1·Granite-4·대조군(순수 attention/Mamba)은 미업로드 | HEAD | N2 뒤, 모델 집합은 사용자 결정 |
| N3 | **없음**. 908623 §7 E1이 처방 원형(셀·n·분할 E1a/E1b) | λ0 VESSL 러너 재사용 + goodput 정본 술어 산출 | 있음 | HEAD (PHASE_EVENTS 켤지 결정 — 켜면 관측 효과 대조 필요) | D-SEM·SLO 결정 → 사전등록 → 규칙층 → 승인 |
| N4 | 해당 없음 | — | — | — | doc-steward 반영 + claims-auditor 확인 |
| N5 | S0 rev1 NO-GO 판정서가 S0′ 조건을 줌 | CPU 스크립트(A10 변이 M1′/M3/M5 포함) | N2 points + snapshot 스트림 | — | 사전등록 → 규칙층 → 실행(CPU) → 결과 감사 |
| S1 | anchor §4 계약 + engine-porter 검토 있음 | 엔진 구현 + CPU fake 루프 gate | — | 신규 커밋 | engine-porter 구현 → correctness(CPU→GPU) |

### 2-2. 예산 (VESSL 환산 ×1.36, $1.48/h)

| # | KISTI 기준 GPU-h | VESSL 예상 h | 비용 | Job 분할 |
|---|---|---|---|---|
| N1 | 1.803 (최악 2.338) | ≈2.45 (최악 3.18) | ≈$3.6 (최악 $4.7) | **1 Job**(E2-α: 두 arm 같은 Job) |
| N2 | 0.8–1.5 (추정) | ≈1.1–2.0 + L>65536 셀 | ≈$2–4 | 1 Job(같은 노드에서 d 전 격자) + L>65536 별도 boot |
| N2′ | 모델당 N2와 비슷 | 모델당 ≈1.1–2.0 | 모델당 ≈$2–4 | 모델당 1 Job |
| N3 (sticky OFF) | E1a 1.55 + E1b 2.74 | ≈2.1 + 3.7 | ≈$3.1 + $5.5 | 2 Job(shape별). 한 비교의 arm은 한 Job |
| N3 (sticky ON 병기 시) | ×2 | ≈11.7 | ≈$17 | 2 Job |
| N4·N5 | 0 | 0 | 0 | — |
| S1 GPU correctness | ≈0.5 | ≈0.7 | ≈$1 | 1 Job |
| (선택) λ0 Job 2 | 1.39 | ≈1.9 | ≈$3 | 1 Job |
| (선택) CPU quota 대조 | ≈0.4 | ≈0.5 | ≈$1 | 1 Job |

N1–N3 합계는 약 $13–30이다. 잔액 $391 대비 **돈은 구속 조건이 아니다.** 구속 조건은 감사 회차와 사용자 승인이다.

### 2-3. 권고 실행 순서

| 파 | 무엇을 | GPU | 차단 요인 |
|---|---|---|---|
| 1 (지금) | N1 A13 追記 작성(F5 값 사용)<br>N4 종결 문안<br>**D-SEM·SLO 값·N2 격자 범위 결정 요청**<br>(선택) S1 구현 착수 | 0 | 사용자 결정 |
| 2 | N1 규칙층 감사(**R1 OFF 절 재등록 포함, F5′**)·하네스 감사 → 새 OVERRIDE → **N1 제출**<br>병행: N2·N3 사전등록과 규칙층 감사 | N1 ≈$4 | OVERRIDE |
| 3 | N1 결과 감사 → N2 제출, N3 제출(E1a 먼저) | ≈$10 | N1 결과(N2 노브 검증) |
| 4 | N5 S0′(CPU)<br>모델 집합 결정 후 N2′ → A11 | N2′만 | 모델 업로드 범위 |
| 5 | S1 GPU correctness → 5-a → 5-b | 미산정 | N5 라벨, N3 부하점 |

### 2-4. 블로커와 필요한 사용자 결정

| 결정 | 무엇을 막나 | 권고 |
|---|---|---|
| **E2 새 OVERRIDE(VESSL)** | N1 제출 | 승인. 범위: 20 boot, 하드캡 ≈4 h 컨테이너 |
| **D-SEM: 5-a/5-b arm의 sticky 의미론** | N3 설계, 5-a arm 정의 | 정책·static arm 모두 sticky OFF. sticky ON은 프로파일(N2) 전용 |
| **N3 SLO 값** | N3 사전등록 | 목표 워크로드(긴 문서 요약)용 TTFT SLO + ITL p95 SLO를 절벽(55–62 ms) 밖에 등록. 값은 사용자가 정하고 감사가 절벽 여유를 확인 |
| **N2 격자 범위** | N2 설계 | 엔진 결정 격자 {16,24,34,44,108}만 사용(엔진 변경 0). 더 넓은 SM 곡선은 후속 |
| **A11 모델 집합 / Zamba2 가중치** | N2′, 주장 조건 1의 귀속 | **Zamba2 삭제 보류**(A11 후보). 모델 집합은 N2 결과 뒤에 결정 |
| E-1 보류 확정 | 없음(사다리 밖) | 보류 |
| push 방식 | 없음 | 사용자 몫 |

## 3. 주의

- **N1의 a_r2는 VESSL에서 1.002×λ\*에 놓인다**(KISTI 0.983×). 같은 offered rate가 다른 ρ에 있다. E2C-6(Q5 절벽 위)의 VESSL 재산출이 A13에 들어가야 한다.
- **R1 OFF 밴드는 여유가 줄어든 정도가 아니라 이미 초과다**(F5′). 단 A5 4분지 덕분에 결과는 거짓 귀속이 아니라 `OFF_BASELINE_ABOVE_BAND`다. VESSL a_r2 E-qcond는 20.57%, a_r4는 18.33%다. KISTI 기준 여유는 2.35pp였다. 아직 실행 전이므로 재등록은 사후 조정이 아니다. 단 근거는 λ0 VESSL 단일 seed 한 셀이다. 새 문턱이 이 한 값에 과적합되지 않도록 감사가 확인해야 한다.
- **N2의 bs 축**: max_running 48 = mamba 캐시 48이 결합돼 있다(908623 E3). bs 상한을 올리려면 `--max-mamba-cache-size`를 분리 고정해야 하고, 그 correctness가 미확인이다. 48 이하로 두는 것을 권고한다.
- vesslctl 토큰은 2026-10-05 14:44 KST에 만료된다. Workspace 2개는 `standby` 상태이며, 과금 여부를 콘솔에서 확인해야 한다.
