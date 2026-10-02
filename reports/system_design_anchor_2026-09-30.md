# 시스템 설계 기준선(anchor) — hybrid LLM PD-mux 서빙: 배분·재정렬·일시중단 통합 설계 (2026-09-30)

> **지위**: 사용자 지시(2026-09-30)로 이 문서를 **프로젝트 전체의 설계 기준선(main anchor)**으로 고정한다. 이후의 실험 카드·
> 사전등록·엔진 수정·논문 서술은 이 좌표계 위에서 정의한다. **측정 결과가 아니다** — GPU 0 · 새 성능 판정 0 · Claim 등급 변경 0.
> 결과 정본은 여전히 `PROJECT_STATUS.md` > `reports/paper/` > `CONSENSUS.md`이며, 이 문서는 그 위계와 충돌하지 않고 "앞으로 무엇을
> 어떤 구조로 검증하는가"만 고정한다. 확정 결론(PD 분리 이득·layer-type 死·HE0·decode-heavy static 권고)은 재도출하지 않는다.
> 시각화: artifact "Hybrid PD-mux Anchor"(https://claude.ai/artifact/WEgWi5nvCLJCM7Niu8XsTZ, 2026-09-30 v1, 스냅샷 `handoff-report/artifacts_2026-09-30/hybrid_pdmux_anchor.html`).
> 관련: `workspace/engine-port/results/hybrid_sched_s0/{DESIGN_S0_REV1_2026-09-30.md, audit_s0_rules_2026-09-30/VERDICT.md,
> engine_porter_review_2026-09-30.md}`, `reports/ttft_measurement_definition_review_2026-09-28.md`.

## 1. 중심 주장과 반증 조건

**주장(검증 대상, 미확인)**: SM 배분형 단일-GPU PD-mux 스케줄러(Bullet/MuxWise 계열: 예측기 기반 최소-SM 배분 + 요청 재정렬 +
극단 부하 시 decode 일시중단)를 hybrid(attention+SSM) 모델로 확장하는 일은 **자명하지 않다**. Bullet 원문은 이 확장을
"straightforward, future work"로 남겼고 아무도 수행하지 않았다(인용 의무, novelty 감점 아님).

**자명하지 않음의 조작적 정의(셋 다 필요)**:
1. generic 예측기(전 층 attention 가정, 두 점 보정, online 재보정 포함)가 hybrid에서 **다른 decode 최소 SM·다른 행동**을 낸다.
2. 그 차이가 **goodput**(TTFT ≤ SLO ∧ 요청내 token-ITL p95 ≤ SLO)에 나타난다 — n≥4, 짝지은 CI, 3% 이상, 변화 trace.
3. hybrid 보정판(attn/SSM 분리 프로파일)이 그 손실을 **고친다**.

**반증**: 1이 없으면 확장은 자명(부정 결과로 기록) · 1은 있되 2가 없으면 "결정은 다르나 성능은 같다"(상수 튜닝 수준) ·
1·2는 있되 3이 없으면 "hybrid에서 이 계열 스케줄러가 깨지지만 프로파일로는 못 고친다"(기전 결과). 네 결과 전부 논문 결과다.
**금지 서술**: "이미 검증됐다" · 충실한 baseline(§3의 P_gen+recal) 없이 얻은 우세.

## 2. 시스템 구조 (기존 / 신규)

| 층 | 구성 요소 | 상태 | 근거 |
|---|---|---|---|
| 부하 생성 | open-loop Poisson 클라이언트(`bench_serving` / `trace_loadgen`), 거부 없음, TTFT = 송신→첫 토큰 | 기존 | TTFT 메모 §1 |
| API/토크나이저 | upstream, `SchedulerReqTimeStats` stamp(`wait_queue_entry`, `forward_entry`, `prefill_finished`) | 기존 | TTFT 메모 §1.3·engine-porter §5 |
| 대기열·admission | `waiting_queue` → **F1 재정렬 훅(신규, 기본 OFF)** → `get_new_batch_prefill` → split-prefill span | 훅 신규 | engine-porter §2 방식 A |
| 실행 | prefill 스트림(span 단위 layer 진행) ∥ decode 스트림(step 단위), green-context stream group {idx0 (108,0) · 분할 (p,d) · (0,108)} | 기존 | `pdmux_context.py`, mixin |
| 관측 | `runtime_snapshot`(decode bs·ctx p95·queue depth·oldest age·ITL p95·occupancy) | 기존 | `controller.py` RuntimeSnapshot |
| 예측기 | `ConservativeDecodeFloorEstimator.estimate(batch, ctx, SLO)` → `d_min` — 입력은 프로파일 `points` | 기존, **3판으로 확장** | §3 |
| 규칙 | Bullet Algorithm-1형: 예측기로 SLO 만족 최소 SM(SQUEEZE), 극단 부하에서만 PAUSE | **고정(신규)** | 감사 S0′ |
| 실행기 | `SplitDecision` → **pause manager(신규)** → stream group 선택·드레인 | 신규 | engine-porter §1·§3 |
| 분석 | `analyze.py` goodput + 정상성 검사 3종 + TTFT 분해(`W_queue`/`S_prefill`) | 규약 신규 | TTFT 메모 §3 |

## 3. 정책 스택 — 규칙은 하나, 예측기는 셋

규칙(고정): 압력(`prefill_queue_depth>0 ∧ oldest_prefill_age > τ_age`)일 때 `d_min = estimate(batch, ctx_p95, SLO)`; `d_min < current`면
`SQUEEZE(d_min)`; `d_min == 격자 최소`이고 압력 지속이며 예측 TPOT가 SLO 안이면 `PAUSE`; 그 외 `NOOP`. 분기 순서·τ_age·"1 step"
정의는 사전등록에서 상수로 고정한다(S0 rev1 死因 K3 교훈).

| 예측기 | 형식 | 역할 |
|---|---|---|
| P_gen | 같은 모델 두 점 보정, 전 층 attention 가정의 ctx 기울기(Bullet Fig. 9 방식) | generic baseline |
| P_gen+recal | P_gen + online 재보정(Bullet 방식 모사) | **충실한 baseline** — 없으면 허수아비 |
| P_hyb | attn/SSM 분리 곡선(attention ∝ ctx, SSM ≈ 상수)에서 합성한 프로파일 `points` | 제안 |

hybrid성은 `points` **값**으로만 들어간다(estimator는 `attention_ratio`를 읽지 않음 — 메타필드 조작은 항등식). 따라서 P_hyb와 P_gen의
차이는 프로파일 재생성으로만 만들 수 있고, 3-a 실측 프로파일(SM × batch × ctx)이 선행한다.

## 4. Pause manager 계약

PAUSE는 1급 스케줄러 상태다. 관리자가 소유하는 것:
- **쌍 건너뜀**: `update_running_batch`(→`prepare_for_decode`, KV 슬롯 전진)와 decode 제출을 **함께** 건너뛴다. decode만 건너뛰면
  step마다 KV 할당이 누적된다.
- **스트림 전환**: PAUSE 진입 시 idx0 `(108,0)`으로 전환(드레인 1회), 해제 시 등록된 복귀 인덱스로(드레인 1회). green-context는 하드
  분할이라 전환 없는 PAUSE는 prefill SM 이득 0.
- **자체 시계**: 컨트롤러 케이던스는 `decode_iterations`로 세므로 PAUSE 중 얼어붙는다. 종료는 관리자의 카운터(n_max) 또는 prefill
  final 제출 이벤트로만.
- **불변식**: (i) `split_prefill_batch is not None ∧ ¬wait_prefill_kernel_done`에서만 PAUSE (ii) 하드 상한 n_max (iii) R2 admission
  latch와 상호 배타(assert) (iv) prefill final 제출 시 자동 해제. 긴급 occupancy 조건은 PAUSE를 취소한다.
- **계측**: 정지 iteration을 ITL 표본에서 제외하되, 정지된 요청의 실제 토큰 간격은 요청 단위로 따로 기록(전 arm 대칭).
- **금지 조합**: sticky partition · SLO_SCHED · LA_COORD · TP>1(broadcast 미구현) — fail-fast.
- **correctness gate**: CPU fake 루프(legacy·true-dual, 정지 step에 `prepare_for_decode` 0회, 재개 후 결과열 OFF와 동일, 기본값 OFF
  바이트 동일, 변이본[decode만 건너뜀]·[불변식 i/ii 제거] 실패) → GPU(PAUSE OFF/ON 출력 동치, observer effect).

## 5. F1 재정렬 훅

`get_new_batch_prefill()` 직전에 `waiting_queue`를 정렬(기본 OFF env, fcfs에서 `calc_priority` no-op이라 순서 보존). radix OFF에서는
upstream `lpm`/`dfs-weight`가 조용히 FCFS로 바뀌므로 "예상 지연 오름차순"(Bullet `SortByLeastEstimLatency`)은 새 정렬 코드다.
`--enable-priority-scheduling`과 조합 금지. hybrid 슬롯 예산 정렬은 radix OFF에서 running 상한과 **항등**(M0 확인) — radix ON 확장에서만
의미(SSM state 재계산 비용).

## 6. 시나리오(개념, 측정 아님)

| # | 상황 | 규칙의 행동 | 관리자·엔진 | 무엇을 검증하나 |
|---|---|---|---|---|
| S1 | 분할 정상 동거(prefill span 진행, decode 여유) | `NOOP` | (p,d) 유지 | 기준 상태 |
| S2 | 압력 발생, `d_min < current` | `SQUEEZE(d_min)` | 분할 인덱스 하향, 드레인 | 예측기 3판의 `d_min` 차이(S0′) |
| S3 | 압력 지속, `d_min` = 격자 최소, 예측 TPOT 여유 | `PAUSE` | idx0 전환, 쌍 건너뜀, n_max 시계 | pause manager 계약·gate 2 |
| S4 | R2 admission latch 발효 중 압력 | `PAUSE` 금지(불변식 iii) | prefill 보류만, decode drain | 데드락 부재 |
| S5 | decode 빈 상태 | 규칙 미발화(2F9) | `adjust_stream_groups` (108,0) | 결정점 아님 |
| S6 | shape A(짧은 입력·긴 출력) vs shape B(긴 입력·짧은 출력) | A: span 1개, 결정점 희소 / B: span 다수, 동거 지배 | — | 타겟 워크로드 선택 근거 |

## 7. 워크로드 좌표계

타겟 = **긴 문서 요약**(one-shot, radix OFF 정합). RAG는 radix ON 확장 뒤에만 이름 붙인다. chat(shape A)은 기존 확정 결론의
본거지이며 재측정 대상, reasoning(긴 출력)은 분할 무관, agentic(TraceLab)은 radix ON 이후. 정책 비교는 변화 trace(W2 8K/64 버스트,
W4 (8192,64)↔(256,512) 교대)로만, 정상 shape는 5-a 스윕에만.

**축의 주체(2026-10-01 추가)**: 워크로드 좌표계는 **U(요청) 축**이다 — 입력·출력 길이, 도착 과정, prefix 구조는 사용자가 정하고 엔진은
고를 수 없다. 설계 대상은 **E(엔진)** — admission·배치 구성·분할·PAUSE·span·메모리 예산. **M(모델)** — 층 조성·KV/state 크기는 배포 상수라
단일 모델 안에서는 축이 아니며 귀속은 A11 hold-out으로만. 배치 ctx 분포는 U×E의 곱이고, hybrid는 decode 비용에서 U 성분의 비중을 낮춘다
(SSM 층 비용은 bs, 즉 E에만 의존). 기존 결과의 방향: M을 바꿔도 결정은 불변(P1.7·8B·layer-type 死), U를 바꾸면 체제가 바뀜(λ0 shape A/B).
상세 `reports/cost_model_design_space_2026-10-01.md` §1.0.

**입력 길이 영역(2026-10-01 추가)**: 타겟 "긴 문서 요약"은 배치 토큰 수로 두 영역으로 나눈다. (a) 배치 토큰 ≤ span 예산(65536):
층 분할 단독, span 시간이 예산으로 묶임 — **이 설계의 주 영역**. (b) 예산 초과: span이 "시퀀스 전체 × 1층"에서 하한에 걸려 span 시간이
L에 비례(attention은 L²)해 늘어나고, 입도를 되찾으려면 토큰 축 분할(chunk)이 필요해 chunk의 문제(KV 반복 읽기, SSM state 이월, 노브 2개,
mixed chunk 충돌)가 돌아온다 — **별도 영역, 이 설계의 결론을 이식하지 않는다.** 정의들의 사각지대 전체는
`reports/definition_blind_spots_2026-10-01.md`.

## 8. 증거 사다리

V-0/D-0 기판 점검 → **3-a 실측 decode 프로파일(SM × batch × ctx; + L > 65536 계측 셀 1개: 1층 span 시간 vs decode step 시간, 결정 규칙 0개)** → 3-b λ\*(B) → **S0′(CPU, 예측기 3판 결정 발산; `MODEL_DEPENDENT`
이상은 실측 프로파일 위에서만)** → S1(pause manager + F1 훅 구현, correctness gate, Bullet §4.5형 ablation: 분할만 / +재정렬 / +PAUSE,
각 예측기별) → 5-a(워크로드별 최적 static 격차 = 선택기 상한) → 5-b(변화 trace, n≥4, 짝지은 CI). 각 단계의 NO-GO는 다음 단계를
막고, 그 자체가 기록되는 결과다.

## 9. 고정 / 열린 / 하지 않음

- **고정**: 규칙 형식(Bullet Algorithm-1형) · 예측기 3판 · pause manager 계약 · TTFT 정의(송신 기준, 거부 없음) · 정상성 검사 3종 ·
  타겟 워크로드 · 사다리 순서 · radix OFF 1순위.
- **열린(사용자 결정)**: VESSL 예산·크레딧 · E2 새 OVERRIDE · 5-b arm 채택 · 모델 업로드 범위 · radix ON 확장 시점.
- **하지 않음**: sim 이득을 실엔진 결론으로 쓰기(게이트 1) · 기판 간 수치 혼합 · layer-type 런타임 정책 재개 · 단일 GPU reactive 동적
  제어의 HE0 재논쟁(정책 정의를 바꾼 새 주장으로만 다룬다).

## 10. 부속 문서 (2026-10-01 추가)

- `reports/cost_model_design_space_2026-10-01.md` — cost model 설계 가능 영역 CM-1…CM-7, 선행 cost model이 hybrid에서 깨질 것으로
  예상되는 지점 X-1…X-11(구조적 차이 4건: X-1·X-4·X-6·X-7), 알고리즘 A1–A11(의사코드, 상수는 전부 사전등록 대상). artifact v2 §9–§11.
- `reports/definition_blind_spots_2026-10-01.md` — 정의 D-1…D-12의 사각지대(간과하는 것·방향·보완). 최상위 5: p95가 PAUSE 간격을 가림 ·
  1층 하한 · 변화 trace의 정상성 미정의 · 결정 반영 지연 · 기판(H) 층 누락.
