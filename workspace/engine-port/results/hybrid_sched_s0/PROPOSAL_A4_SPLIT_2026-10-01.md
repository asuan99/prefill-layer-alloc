# 제안 — A4 규칙의 분리: decode 측 SM은 정적 표, step 단위 레버는 PAUSE·prefill 발사량만 (2026-10-01, 감사 전 · 미반영)

> **지위**: 사용자 지적(2026-10-01, "decode는 step-wise라 이런 규칙은 큰 의미가 없다")에 대한 메인 세션의 재구성 제안. **아직 anchor·
> cost model 문서·artifact에 반영하지 않았다.** 반영 전 claims-auditor가 기존 실험·정본과의 충돌을 감사한다. GPU 0 · 측정 0.
> 대상 문서: `reports/system_design_anchor_2026-09-30.md` §3·§4·§8, `reports/cost_model_design_space_2026-10-01.md` §3(A4·A8·A10)·§5,
> artifact "Hybrid PD-mux Anchor" v3 §3·§11·§12.

> ★★**판정(2026-10-01, 같은 날): 규칙층 감사 `NO-GO`**(`audit_a4_split_2026-10-01/VERDICT.md`, 死因 2·차단 10·반증 실패 8). 死因: (K1) 오프라인 표에서는
> P_gen+recal의 recal 입력이 없어 표_{gen+recal} ≡ 표_gen — 충실한 baseline이 사라지거나(오프라인), 사건 시점 재보정을 넣으면 A4a가 반응형이 됨 ·
> (K2) 단일 모델에서 P_hyb 표와 P_gen(식 2 SM별 재적합) 표는 같은 3-a 점 위의 재매개화라 **D ≡ 0**, M1′는 hybrid 검사가 아님 — 귀속은 A11 모델
> hold-out 또는 "ctx 기울기를 대상 모델에서 재지 않는 P_gen"으로만 가능. 그 밖에 F1 arm 삭제(anchor 사다리 위반), 발사량 레버의 엔진 대응물
> 부재(비동기 span 발행), `manual_divisions` 형식 불일치(bs 단일 문턱·break 없음), Claim E는 승계가 아니라 재정의, A4b-2는 layer-type 정책
> 계열이라 anchor §9와 충돌(사용자 결정). **미반영 유지.** HE0·E2·λ0·게이트 #13/#16/#6과는 충돌하지 않음(반증 실패 8건).

## 1. 지적의 요지와 수용

- decode는 loop 1회 = decode 1 step(CUDA graph 1개)이라 step 안에서 바꿀 것이 없다. d_min의 입력(bs, ctx 조성)은 prefill 완료 merge·
  요청 완료·admission 사건에서만 바뀌고, 재집계(설계 메모 §7–§8, 미등록·방향만)에 따르면 shape B에서 prefill 한 구간 동안 decode bs
  변화는 드물다. ⇒ 매 step/epoch "규칙"을 돌려도 결과는 구간 내내 같고, 이는 **(bs, ctx-class) → d_min 정적 표의 사건 시점 조회**와 같다.
- 이는 정본 HE0(반응형 동적 제어는 best static을 못 넘음)·MuxWise 공개 엔진의 `decode_bs_threshold` 표(load-indexed static)와 정합한다.
  ⇒ 현행 A4("Bullet Alg.1 decode 측 규칙, 케이던스 평가")는 **제어가 아닌 표**로 재서술한다.

## 2. 재구성안

### 2.1 A4 → A4a(정적 표) + A4b(step 레버)
- **A4a 분할 표**: `d_min(bs_class, ctx_class) = min{ d ∈ 격자 : P_*(d, bs, ctx).p95 + margin ≤ SLO }`를 **오프라인**(L0)에서 P_gen / P_gen+recal /
  P_hyb 각각으로 생성. 런타임에는 merge·완료·admission 사건에서만 조회(L2 사건 훅), **케이던스 컨트롤러 없음**. 형식은 MuxWise
  `manual_divisions` 표와 동일(엔진 기존 경로 재사용 가능성 — engine-porter 확인 대상).
- **A4b step 레버 (둘뿐)**:
  1. **PAUSE(k)**: 이산 선택. 비용 = decode 요청들의 ITL에 `k·T_d` 가산(CM-1), 이득 = prefill이 `k·T_d` 동안 108 SM(CM-2로 단축량 계산).
     술어: `gain_TTFT(k) > 0 ∧ ∀ 요청 ITL_pred + k·T_d ≤ SLO ∧ 불변식 4(pause manager)`. k는 등록 격자.
  2. **prefill 발사량**: 매 decode step에 발사할 prefill 층/span 수 `L_step`을 CM-2의 층 종류별 비용으로 정해 그 step을 덮는다(MuxWise
     `N_PL`·Bullet `L_step`의 비용 가중판, X-4). 현 엔진은 budget 기반 고정 span이라 **새 레버**(엔진 수정).
- SQUEEZE-as-control(케이던스 downshift/upshift, dwell, streak)은 **제거**. `CoarseGrainedController`(Claim E 컨트롤러)는 이 설계에서
  쓰지 않는다 — 단 Claim E 등록 범위(hybrid 프로파일 예측기가 결정을 바꾸는가)는 A4a 표의 값 차이로 그대로 승계.

### 2.2 S0′ → 표 발산 검사
- 상태 격자 `(bs_class × ctx_class)` 위에서 `d_min_hyb ≠ d_min_gen(+recal)`인 셀의 비율을, **경험 상태 분포**(λ0 telemetry의 (bs, ctx_p95)
  히스토그램, 규약 등록)로 가중해 D를 계산. telemetry 재생·규칙 시뮬레이션 불필요. 변이 검사 M3/M5/M1′ 유지.
- PAUSE 술어의 발산(P_* 별로 "PAUSE 이득" 셀이 다른가)은 별도 지표 D_pause.

### 2.3 S1 ablation 재정의
- arm: 분할 표만(P_gen / P_hyb) → +발사량 → +PAUSE. SQUEEZE 제어 arm은 **삭제**. 각 arm은 같은 job 내 교차배치, 변화 trace, n≥4.

### 2.4 주장 1 문장
"generic 예측기가 hybrid에서 다른 **분할 표**를 내고, 그 표가 goodput을 바꾼다" + 동적 레버 주장은 PAUSE·발사량에만.

## 3. 감사에 묻는 것 (충돌 후보 — 메인 세션이 아는 범위)

1. **P1 agnostic arm과의 동일성**(AUDIT_1 C5/C6, CONSENSUS §5-14): A4a 표는 MuxWise 공개 엔진의 `decode_bs_threshold` 표와 같은 기제다.
   이미 측정된 P1 auto-grid agnostic arm(2단계 문턱 표)과 **무엇이 다른가** — 값(P_*로 생성) vs 형식. "새 방법"으로 명명 금지 조건(교훈 261).
2. **HE0·게이트 #13/#16·stake #1**: A4a는 static이라 HE0와 충돌하지 않는가, 아니면 "load-indexed static도 HE0 범위"라 A4b(PAUSE·발사량)만
   동적 주장이 되는가. stake #1("최적 static 위치가 워크로드에 따라 움직인다"는 MuxWise 제품 전제)과 A4a의 표 발산 주장이 겹치는가.
3. **Claim E 등록 범위·R2 컨트롤러**: `CoarseGrainedController`·`PDMUX_R2_POLICY=hybrid` 계열(E2·λ0 등록 arm, job 908534/908623)이 이
   재구성으로 **폐기**되는가, 승계되는가. E2 사전등록(sticky 분할 대조, OVERRIDE 대기)과 충돌하는가.
4. **layer-type 死**와 A4b-2(발사량을 층 종류별 비용으로): "prefill span을 층 종류 경계에서 자른다"(`PDMUX_SLO_SPAN_TYPE`, R4·R7 계열,
   type-aware span sizing의 TTFT 악화 선례)와 어디까지 같은 것인가. 死 판정을 되살리는 서술이 되지 않는가.
5. **2F9·E2C-21**: A4a 사건 훅이 decode-only 구간에서 표를 조회하는 것은 결정점이 아닌 곳의 동작 — 사건 시점 정의가 2F9와 정합하는가.
6. **설계 메모 §7–§8 재집계 의존**: "bs 변화 드묾"은 미등록 스크립트 수치(인용 금지). 재구성의 전제로 써도 되는 범위.
7. **PAUSE 술어의 항등식 위험**(교훈 9): `gain_TTFT(k) > 0`가 CM-2에서 항상 참(prefill이 108 SM을 받으면 항상 빨라짐)이면 술어는 ITL 제약
   하나로 축소되고, 그 제약은 P_*에 따라 갈릴 수 있는가.
8. 재구성이 anchor의 "고정" 목록(규칙 형식 Bullet Alg.1형)을 바꾸므로 anchor 자체의 개정 절차(사용자 승인·doc-steward)가 필요한가.
