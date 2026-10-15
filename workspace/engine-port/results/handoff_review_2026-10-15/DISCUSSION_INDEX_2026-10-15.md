# 논의 재개 목록 — 2026-10-15 main 세션 (인계 프롬프트 검토 후속)

> 지위: main 트랙 논의 색인. 측정 0, 새 판정 0. 각 항목의 근거는 같은 폴더의 메모에 있다.
> - R = `REVIEW_handoff_prompt_2026-10-15.md`
> - A = `AGENDA_methodology_2026-10-15.md`
> - D = `DESIGN_span_and_integration_2026-10-15.md`
> - C = `COMPARE_span_sync_muxwise_bullet_2026-10-15.md`
> - K = `COSTMODEL_running_cap_2026-10-15.md`

## 0. 이미 결정된 것 (다시 논의하지 않음)

| # | 결정 | 날짜 | 출처 |
|---|---|---|---|
| F1 | 옛 환경 전제(SLURM·KISTI)는 무시, GPU 실행은 VESSL Job | 2026-10-15 | 사용자 |
| F2 | 주 모델 = Nemotron-Nano-9B-v2 | 2026-10-14, 10-15 재확인 | 사용자 |
| F3 | **true dual worker(독립 박자) = AMD 트랙 전용** | 2026-10-15 | 사용자 |
| F4 | **main의 lockstep PD-mux 구조가 multi-GPU PD-mux로 확장·이양**. 단일 GPU 결과는 "lockstep 기판" 범위, lockstep 정지는 특성화 대상 | 2026-10-15 | 사용자 |
| F5 | TTFT SLO = A′ (k·floor(L), k∈{3,5,10}), ITL = 요청 내부 p95 50 ms, 판정 = attainment 90% 용량 | 2026-10-04 | 사용자 승인 (`slo_definition_rationale_2026-10-04.md`) |
| F6 | multi-GPU arm = fused / fused_sym / agnostic / d44, 1차 estimand G_sym | 2026-10-12 | 사용자 (`PREREG_MULTIGPU`) |
| F7 | TP cap = 48·N | 2026-10-12 | 사용자 |
| F8 | radix OFF 우선 | 2026-09-24 | 사용자 |

F3·F4에 맞춰 이미 정합한 것: M3 PREREG는 TP에서 true-dual을 거부하고(`multigpu_guard`), AMD 설계는 자신을 "H-Architecture의 디바이스 확장판"으로 적는다.

## 1. 논의 안건 (의존 순서)

### 1군 — 먼저 풀어야 다른 것이 풀림

| # | 안건 | 선택지 | 권고 | 근거 |
|---|---|---|---|---|
| Q1 | **N2 종료 여부** (후속 F-1…F-4 중 무엇을 할지) | 전부 / F-1·F-2만 / 종료 | F-2(클럭 수집)는 측정 명세에 흡수하고, F-1은 G0 Job 계측으로 대체한 뒤 **N2 종료** | N2/N3/E2 봉인이 끝나야 `multigpu-scaling` merge → 엔진 트리 하나 (D §3.2-6) |
| Q2 | **논문 포지션** | P-A 특성화·일반화 / P-B 시스템 설계(anchor 주장 1) / P-C 혼합 | P-C | 두 질문이 서로 다른 워크로드를 요구 (R §2.3) |
| Q3 | **워크로드 세트** | Mooncake Conv(주) + ShareGPT + OpenThoughts + LooGLE(선택) + **arXiv-Sum 16k 추가 여부** | G0와 G2 교차 모델 표준 셀 = arXiv-Sum 16k, 5-a = Mooncake Conv + OpenThoughts + ShareGPT | R §2.2: arXiv-Sum은 PD-mux 유리 영역이고 decode 축이 약함 |

### 2군 — 공통 측정 명세 (단일·multi GPU 공용)

| # | 안건 | 권고 | 근거 |
|---|---|---|---|
| Q4 | 게이트 2 적용 범위 | 정적 비교는 고정-rate 허용(현 관행 명문화). 실 trace rate = 시간축 압축 배율 | A §5-1 |
| Q5 | **n의 층위와 검정** | seed ≥4 × **Job ≥3**, 판정은 Job 수준 (부호 일치 + t-CI) | A §2. σ_Job ≈ 3% 문턱 |
| Q6 | A′의 혼합 길이 적용 | 요청별 k·floor(L_i), floor 6점 실측, 곱셈형 유지 | A §3.3 |
| Q7 | ITL 술어 세부 | 50 ms 유지 + 사다리 병기. 짧은 출력 워크로드는 요청당 ITL 표본 수 공시. 실패 요청 처리 통일 | A §3.3 |
| Q8 | **단일 GPU arm 집합** | multi-GPU와 같은 {fused, fused_sym, agnostic, static D} 채택 (+선택: 유한 fused 튜닝 스윕) | F6, D §3.1 |
| Q9 | 용량 정의 | λ\*_thr(처리율 knee) + λ\*_SLO(k)(A′) **병기** | D §3.2-3 |
| Q10 | **N 간 SLO** | 판정 = N=1 floor 기준 절대 T, 기술 = N별 상대 SLO | D §3.2-4 |
| Q11 | 전력·클럭 | 판정 전 arm × 창 cap 노출 공변 검사를 사전등록 단계로 (#268) | A §5-8 |
| Q12 | headline 위계 | 용량 → 분포 → CUDA event 기전 | A §5-6 |

### 3군 — lockstep 기판 특성화 (F4에 따른 main의 몫, multi-GPU로 이양될 지식)

| # | 안건 | 권고 | 근거 |
|---|---|---|---|
| Q13 | **span 예산 스윕** `split_forward_token_budget` {8k, 16k, 65k} | G0 또는 직후 Job에 포함. 예측: prefill 창 안 ITL ∝ 예산, TTFT는 반대 방향 | D §2.4, C §4-2 |
| Q14 | 계측 보강 (span GPU 시간, decode 커널만의 GPU 시간, admission 거절 사유, cudagraph 패딩, clk 정렬) | G0 전에 1·2·5번. engine-porter correctness gate 선행. multi-GPU도 공용 | A §4.4 |
| Q15 | **§1-24 "고칠 수 없다" 범위 재서술** | "upstream lockstep 구현의 성질"로 좁히는 문안을 claims-auditor에 회부. MuxWise·Bullet은 동기화 구조로 회피 | C §4-1 |
| Q16 | **cap 처리** | G0 = 48 유지 + 구속 공시. 비용 모델 검증 = {48, C_mem/2, C_mem} 스윕(`max-mamba-cache-size = cap`) | K §4 |

### 4군 — 비용 모델 (COSTMODEL 항 연결)

| # | 안건 | 연결 항 | 근거 |
|---|---|---|---|
| Q17 | (cap, D) 결합 결정을 비용 모델 출력으로: `λ·E[out]·E[ITL] ≤ C ≤ min(C_mem, C_SLO)` | CM-5·CM-6·CM-1 | K §2–3 |
| Q18 | hybrid state 예약(cap × 0.14 GB) → generic 예측기의 C_mem 과대평가 = anchor 주장 1 후보 | CM-5 | K §3-2 |
| Q19 | KV ↔ 슬롯 구속 경계(cap 48일 때 ~52k) 실측 = X-6 | CM-5 | K §3-3 |
| Q20 | lockstep 정지 = f(span 예산, 108−D)를 CM-4에 별도 항으로 둘지 (동거 간섭 ρ와 분리) | CM-4 | D §2.3 |
| Q21 | roofline 예측 대 실측 대조 규약 (roofline B_max vs 엔진 실효 상한 별개 열) | — | A §5-7 |

### 5군 — 통합 실행 순서

| # | 안건 | 권고 |
|---|---|---|
| Q22 | 공통 측정 명세의 위치 | 2군 결정 → claims-auditor 규칙층 감사 → main `promote`로 정본화, 트랙은 링크만 |
| Q23 | M3 PREREG 개정 (multigpu 대기 결정에 대한 답) | **개정**: 명세 인용, arXiv-Sum shape 추가, `max_prefill_tokens`·span 예산 등록, 절대 SLO |
| Q24 | G0 PREREG | 명세 + N=1 깊이 항목 (Job ≥3, span 예산, 계측). merge 전이면 M3 N=1 앵커와 바이너리가 달라 **직접 비교 금지** |

## 2. promote 때 반영할 문서 정정 (이 세션에서는 쓰지 않음)

1. 작업 루트 `CLAUDE.md` "현재 결론" 첫 줄 "4모델 전부서 fused를 이김" → 정본(2모델·운영점·꼬리 SLO goodput·미조율 fused)으로 정정.
2. 메모리 `engine-port-p0-triage.md:42` 낡은 문장 ("4모델 전부 fused 대비 승") 정정.
3. F3·F4를 MEMORY와 `tracks/amd.md`·`tracks/multigpu.md`에 반영.
4. 인계 프롬프트(`handoff/CLAUDE_CODE_PROMPT.md`)의 "~3×", "TTFT 4–15 s · mean TPOT", "SLURM"은 정본과 불일치. 프롬프트 자체는 수정하지 않고, 이 색인을 대응표로 쓴다.

## 3. inbox에 이미 있던 main 결정 (변동 없음)

- radix 처리 (F8과 정합, OFF 유지 + 편차 공시)
- 128k 단일 요청 프로브 (Q6 floor 측정과 합칠 수 있음)
- 장문 hold-out 모델
- CONSENSUS §1 행 38 유지 여부

## 4. 다음 세션 시작점 — Q1 "N2 종료 여부" 답변 준비 자료

**무엇을 정하나**:
- N2(3-a: 동거 간섭 CM-4의 static 격자 측정)를 어디까지 하고 닫을지 정한다.
- 닫아야 N2/N3/E2 봉인 digest 제약이 풀린다. 그래야 `multigpu-scaling` 브랜치를 main에 merge해 엔진 트리를 하나로 만들 수 있다(D §3.2-6).

**현재 사실** [C]:
- Job-PROFILE(`a5de2d6`): 경량 확인만 거쳤다.
  - 동거 4추정량은 산출 전에 claims-auditor 확인이 필요하다(블라인드 노출 N2C-87·98).
- Job-EA(`7e09cc3`): 결과 감사 死因 0, 라벨 8/8 `COLOC_SLOWER_GE3` CONFIRMED(scoped).
  - ρ ≈ 1.16은 동거와 전력 한도 클럭 하락의 합성값이라 크기로 인용할 수 없다(N2C-102).
  - "N2는 완결되지 않았다"(N2C-10).
- 후속 F-1…F-4 (`n2_static_grid/VERDICT_result_n2_ea_7e09cc3_2026-10-13.md` §10, 처방 아님):

| 후속 | 내용 | 실현 조건 / 비용 | 닫는 것 |
|---|---|---|---|
| F-1 전력 정합 대조 | 같은 boot 안에 C_burn arm 추가: decode 단독 + 보완 분할에 FMA burner로 클럭 ±1%·cap 노출 정합 → ln ρ를 전력·클럭 성분과 prefill 고유 성분으로 분해 | 서버 프로세스 안 green-ctx burner 주입 경로 **미확인**, burner의 ≈376 W 도달 가능성 미확인 → 엔진 작업 선행 | CM-4 크기 (전력 분리) |
| F-2 클럭 수집 | `clocks.mem`·NVML 버퍼 SM 클럭 수집 확인 (`sm_clk_samples.csv` 0행 문제) | 하네스 수정, 다음 아무 Job에 편승 가능 | 계측 공백 |
| F-3 운영 trace | 운영 rate trace 부하에서 동거/단독 비와 창별 cap 노출 | 별도 사전등록. N2C-107(운영 대표성)을 닫는 유일 경로 | 운영 대표성 |
| F-4 확장 격자 | b ∈ {1,48}, c 512, D24/34 + 등록 동등성 라벨 | 격자 확장 Job | "D·b·c 무관" 주장 |

**선택지와 고려점** [I]:
- (a) **N2 종료 선언 + 흡수**: F-2는 측정 명세의 기본 계측(Q11·Q14)으로 흡수한다. F-1은 G0/명세 Job의 계측으로 흡수한다. F-3은 5-a(실 trace)가 같은 질문을 다룬다. F-4는 포기한다.
  - PROFILE 동거 추정량은 감사 1회로 마감한다.
  - → 장점: merge가 빨라지고 엔진 트리가 하나가 된다. 단점: CM-4 크기 분리가 G0 이후로 밀린다.
- (b) **F-1·F-2만 수행 후 종료**: CM-4 크기 분리를 먼저 얻는다. 단 F-1은 burner 주입 경로가 미확인이라 엔진 작업 + 정확성 게이트가 선행된다.
- (c) **전부 수행**: 가장 느리고, merge가 가장 늦다.
- 권고: (a). F4(lockstep의 multi-GPU 이양) 결정으로 엔진 트리 통일의 가치가 커졌다. 또 lockstep 정지(Q13·Q20)가 CM-4와 같은 ITL 경로에 있으므로, CM-4 분해를 G0 계측(span·decode GPU 시간)과 같이 하는 편이 교락을 덜 만든다.

**답하려면 확인할 것**:
1. PROFILE 동거 추정량 감사 범위 (claims-auditor 1회로 충분한가)
2. 봉인 digest 제약 해제 조건이 "캠페인 종료 선언"만으로 충족되는가 (`tracks/multigpu.md` "main merge 금지" 문구 근거 확인)
3. N3 shape B(게이트 #6 미닫힘)와 E2도 함께 종료로 볼지

## 5. 안건별 답변 출발점 (각 Q를 열 때 먼저 읽을 것)

| Q | 먼저 읽을 곳 |
|---|---|
| Q1 | 위 §4, `tracks/multigpu.md`, N2 VERDICT §10–11 |
| Q2–Q3 | R §2 (워크로드 의의·포지션 표), `workload_survey/`, `arxsum/ARXSUM_STATS_2026-10-15.txt` |
| Q4–Q12 | A §1 (VESSL 이후 해석 표), A §2 (n 층위), A §3 (goodput 정의), D §3.1 (단일 vs multi 대조) |
| Q13–Q15 | D §2 (span→ITL 사슬), C §1–4 (MuxWise·Bullet 대조), CONSENSUS §1-24 |
| Q16–Q21 | K §1–4 (cap 출처·세 제약·창), `costmodel_a100/` |
| Q22–Q24 | D §3.2–3.3, `PREREG_MULTIGPU_2026-10-12.md`(multigpu worktree) |
