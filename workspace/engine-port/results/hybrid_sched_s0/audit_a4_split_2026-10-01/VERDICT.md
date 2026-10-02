# 판정서 — `PROPOSAL_A4_SPLIT_2026-10-01.md` 규칙층 적대 감사 (기존 실험·정본과의 충돌)

2026-10-01 · claims-auditor · **`NO-GO`** · 死因 **2** · 차단 **10** · 반증 실패 **8** · **GPU 0 · 파일 수정 0**

> ★감사 에이전트는 read-only라 **메인 세션이 반환문을 기록**한다. 재계산 스크립트 없음. 死因 2건은 코드·문서 대수의 구조 논증이며
> 수치에 의존하지 않는다. 인용정지 검사 0건 위반. 말미 "메인 세션 독립 재검증"은 기록 시점에 메인 세션이 코드로 확인한 것이다.

## 단일 판정 질문에 대한 답

충돌한다. 대부분은 문구·절차 층이라 개정으로 고칠 수 있으나, 제안 전체를 막는 것이 둘이다.

- **(1) P_gen+recal이 사라진다.** 표 3판을 "오프라인에서 생성, 케이던스 컨트롤러 없음"이라 했는데 recal(A3)은 서빙 중 관측 ITL로 도는
  EMA다. 오프라인에는 관측값이 없으므로 표_{gen+recal} ≡ 표_gen. S1 arm 목록에도 P_gen+recal이 없다. anchor가 강제한 "충실한 baseline"이
  빠지고 S0 rev1 死因 K2(허수아비)가 재발한다.
- **(2) S0′ 표 발산은 단일 모델에서 측정이 아니다.** 판정량 D는 등록자가 고른 P_gen 형식에서 연역된다(아래 K2).

A4a를 "정적 표"로 재서술하는 것 자체는 HE0·E2·λ0 등록과 충돌하지 않는다. A4a는 기제상 upstream decode-bs 문턱 선택기와 같다(교훈 261 적용 대상).

## 死因 2건

**K1 ★★★ (충실한 baseline이 항등식으로 붕괴) P_gen+recal의 오프라인 실현.** (a) 오프라인 표를 문자 그대로 읽으면 recal 입력이 없어
표_{gen+recal} ≡ 표_gen — 예측기 3판이 2판으로 줄고 "P_hyb vs 충실한 baseline" 비교에 구조적으로 도달 불가(anchor 금지 서술 위반).
(b) recal을 사건 시점에 돌리면 표 값이 관측 ITL에 반응해 시간에 따라 바뀌어 A4a가 반응형 성분을 갖는다 — "A4a는 static, 동적 주장은
PAUSE·발사량에만"이 거짓이 되고 HE0 인접 도메인으로 들어간다. 어느 쪽을 택해도 자기모순.

**K2 ★★★ (단일 모델 표 발산은 재매개화 항등식).**
- 엔진 estimator는 SM별 측정 `points`를 bilinear 보간할 뿐이며 외삽하지 않고 clamp한다(`src/multiplex/profile.py:224-238`, `:239-290`).
  표의 SM 격자 {16,24,34,44,108}은 곧 3-a 실측 SM이다.
- 각 SM에서 CM-1은 {1, bs, bs·ctx}의 선형식이고 `s_*(SM)`·`c_launch`가 측정점마다 자유이므로, SM당 자유도가 "MuxWise 식 2 SM별 재적합"
  (§2 정정이 정한 P_gen)의 자유도 이상이다. 같은 3-a 점 위의 최소자승이면 두 예측이 같고 **D ≡ 0**.
- 형식이 가장 없는 기준은 실측점 직접 보간(= 현 Claim E 프로파일, 설계 메모 R6). 이 기준에서 P_hyb는 적합 잔차만큼만 다르다.
- A2(두 점, 절편 없음)를 쓰면 D>0이나 그 차이는 bs 비례 형식과 보정 점 2개라는 **비-hybrid 핸디캡**에서 온다. A2의 "전 층 attention 가정"은
  기울기를 같은 hybrid 모델에서 재므로 작동하지 않는 가정이다.
- **M1′는 hybrid 검사가 아니다**: CM-1에서 `n_attn`은 적합 계수에 흡수되는 곱수라 P_hyb(n_attn=n_layers)는 예측상 P_hyb와 같다.
- ⇒ "DIVERGES ∧ M1′ 통과"에 도달 불가. anchor ③("보정판이 손실을 고친다")도 측정점 위에서는 기댓값상 도달 불가. 이는
  `prior_cost_models §4`의 "단일 SM·단일 조성에서 차이 ≈0" 예보를 표 전체로 넓힌 것이며, 이전 S0 판정서의 S0′ 승인("3-a 뒤에는
  MODEL_DEPENDENT 이상 가능")은 이 논증으로 좁아진다(게이트 #110: 이전 판정을 상수로 쓰지 않음).
- 벗어나는 경로는 둘뿐: (i) **A11 모델 hold-out** (ii) ctx 기울기를 대상 모델에서 재지 않는 P_gen(Transformer 대조에서 이식, 또는
  transformer-only 커널 열거 roofline + 수준만 α 보정). 제안에는 둘 다 없다.

### 반전 시험 표

| 자유 표면 | 민 범위 | 판정 변화 |
|---|---|---|
| P_gen 형식 | A2 ↔ 식 2 SM별 재적합 ↔ 실측점 보간 | D>0∧M1′ 실패 ↔ D≡0∧M1′ 통과 ↔ D=적합 잔차 (해석적, 死因) |
| P_gen+recal 실현 | 오프라인 삭제 ↔ 사건 시점 재보정 | 3판→2판 ↔ A4a 반응형화 (해석적, 死因) |
| margin·bs/ctx class 경계·상태분포 가중·조회 사건 집합 | 등록 범위 | 3-a 없음 → 미생성, caveat |
| sticky | OFF ↔ ON | 물리 의미 변화(2F9), 차단 B8 |
| PAUSE gain의 C_switch | 제외 ↔ 포함 | 제외 시 gain>0 공허, 차단 B9 |

## 차단 10건

- **B1** "Claim E 그대로 승계"는 거짓. Claim E = "B6 true-dual hybrid-informed dynamic이 B1/B5 대비 ≥3%"(CEM Claim E 행, ROADMAP P2·Controller 절).
  제안은 컨트롤러를 표로 바꾸고 B5 generic dynamic 대조를 없애고 라이브 안전장치(D108 긴급·admission 제한)를 제거하며 아키텍처(single/true-dual)
  미선언 ⇒ **새 claim E′**. 구현된 `HybridInformedPolicy`는 "미측정·미검증 유지"로 기록.
- **B2** S1에서 F1 재정렬 arm 삭제 — anchor §8·PROJECT_STATUS (d)의 고정 사다리("분할만 / +재정렬 / +PAUSE") 위반, Bullet 귀속 성분 대조
  (CONSENSUS §5-11) 상실.
- **B3** anchor "고정" 목록 개정 필요(규칙 형식·예측기 3판) — (F).
- **B4** A4b-2 "발사량"의 기전이 엔진에 없다: legacy 루프는 루프 1회에 span 1개를 **비동기** 발행하고 prefill 스트림은 최종 이벤트 `query`
  (`multiplexing_mixin.py:1518`) 전까지 동기화하지 않는다(메모 R1 "CPU 루프 경계 ≠ GPU 완료 경계"). "L_step으로 decode step을 덮는다"의
  실행 대응물이 없고, 만들려면 span당 동기화/throttle이라는 두 번째 노브가 필요(confound #10). shape A는 span 1개라 정의역 공집합.
- **B5** "형식은 `manual_divisions` 표와 동일"은 거짓: 엔진 표는 bs 단일 문턱 열이고 루프에 `break`가 없어 마지막 만족 행을 고른다
  (`multiplexing_mixin.py:1184-1188`). ctx_class 축은 엔진 수정+correctness gate 필요. 만족 행 없으면 `stream_idx` 미정의 위험(CONSENSUS §5-12).
- **B6** 비교 arm 누락: P1 auto-grid agnostic(`:1190-1199`, 같은 기제·profile ladder "no profile" 단계), decode-heavy static B1, best static B2.
  같은 코드 경로(threshold 0인 1행 표)로 구현해야 하며 `PDMUX_R2_POLICY=fixed`와 섞으면 `_slo_on`·ITL 샘플링·admission recheck가 함께 바뀜(AUDIT_1 C6).
- **B7** 상태분포 규약 결함: λ0 908623은 KISTI, 3-a는 VESSL(기판 혼합, 게이트 1) · 필드는 `decode_running_batch_size`(S0 B1) · λ0 실현 분할은
  대부분 `(0,108)`(§1-25/26, 교훈 246) · 가중은 E2C-8′ 규약.
- **B8** sticky OFF 등록 필수. decode-only 구간 표 조회 금지(2F9·E2 노브 교락·E2C-21).
- **B9** PAUSE 술어 수정: gain에 C_switch(드레인 2회) 포함 또는 gain 조건 삭제 명시 · ITL 제약은 goodput 다리(요청내 token-ITL p95)와 정합
  (현 `∀ ITL_pred + k·T_d`는 max 의미) · P10 얽힘 지표 병기 · goodput은 정지 간격 포함 클라이언트 토큰 시각으로.
- **B10** §1 문구 정정: "MuxWise 공개 엔진의 표"→ upstream 선택기(`temporary demo`, AUDIT_1 C1) · "HE0와 정합/load-indexed static"은 VP 서술이지
  정본 아님(AUDIT_1 C5) · "규칙 = 사건 시점 정적 표" 동치는 거짓(원 A4는 압력·τ_age·k-epoch 지속 조건이 매 step 변하고 하향 전용; 표는
  양방향; ctx는 매 step 1토큰씩 증가) · "bs 변화 드묾"은 미등록 스크립트 수치라 인용 금지(CONSENSUS §5-13).

## 반증 실패 8건

1. decode 1 step 안에 레버가 없다는 것은 참(graph가 `(stream_idx, bs)` 키로 캡처, 메모 R3).
2. A4a는 비반응형이라 HE0 범위 밖(§1-17 각주).
3. E2·λ0는 `CoarseGrainedController`에 의존하지 않는다(`FixedPolicy(44)`·`_r2_decide_idx`·sticky 경로).
4. SQUEEZE arm 삭제는 기존 등록과 충돌하지 않는다(SQUEEZE를 쓰는 등록 없음).
5. 인용정지 위반 0건.
6. 같은 셀에서의 예측기 차이로만 서술하면 stake #1과 구별된다.
7. 게이트 #13·#16·#6과 충돌하지 않는다(#6은 3-b가 담당).
8. upstream 의미론(prefill 진행 중일 때만 조회)을 따르면 A4a 사건 훅은 2F9와 정합.

## 제안 §3 질문 1–8 답 (요지)

1. **같은 기제.** A4a 조회 = upstream 경로(`adjust_stream_groups`는 prefill admission 시와 decode 빈 때만 호출, prefill 없으면 `(0,108)`).
   차이는 값(P_*)·ctx 축(엔진 수정)·추가 사건. 값 생성기 교체는 정당한 estimand("profile 유래 문턱 vs 휴리스틱")이나 K2로 단일 모델에서는
   hybrid estimand가 아니다. 명명: "upstream decode-bs 선택기의 profile 유래 문턱판"(교훈 261).
2. HE0 무충돌. 단 A4a는 static이 아니라 부하가 이끄는 시간가변 분할 궤적(feedforward). d_min 규칙은 decode를 최소로 잡으므로 §1-17
   positioning·§1-4 얽힘의 실패 방향 → decode-heavy static 대조 필수. PAUSE는 미시험 레버(§5-11 (v)).
3. E2·λ0 존속. 단 등록 결정 경로 파일을 편집하면 sha가 바뀌어 E2는 manifest 실행 또는 재등록 필요(새 OVERRIDE는 사용자 결정). Claim E는
   승계가 아니라 재정의(B1). 컨트롤러 코드는 삭제 금지(E2·λ0 재현·테스트가 사용).
4. (D) 참조.
5. "prefill 진행 중만 조회, decode-only는 (0,108)"이면 2F9 정합. 효과 정의역은 동거 구간뿐 → shape A 검정력 위험(E2C-21 방향).
6. 인용 금지 수치는 전제로 쓸 수 없다. "사건 사이 입력 불변"은 bs에 한해 구성상 참, ctx·압력에는 거짓(B10).
7. C_switch 제외 시 `gain_TTFT(k)>0` 공허(교훈 9). 남는 ITL 제약은 P_*의 T_d 의존이나 K2로 단일 모델에서 D_pause ≡ 0.
8. 필요(F).

## (A)–(F) 판정

- **(A)** 기제 동일성 확인(코드). "P_*로 생성"은 새 estimand("profile 유래 문턱")이나 "hybrid 조성 효과"가 아님(K2). "새 방법"·"MuxWise 표" 명명은 교훈 261·AUDIT_1 C1 위반.
- **(B)** HE0 무충돌(비반응형). A4a가 이겨도 offline 선택 주장(메모 B2). stake #1과는 같은 셀 값 차이로만 구별; 표의 ctx 행 변화를 "최적이 움직인다"로 쓰면 겹침(N4).
- **(C)** E2·λ0 존속(결정 경로 파일 미편집 또는 manifest 실행 조건). Claim E는 재정의, 등급 미검증 불변.
- **(D)** A4b-2는 §1-3 사망 대상(층 종류별 SM)을 그대로 되살리진 않으나, 결정이 다음 층들의 종류에 의존하므로 메모 R6 분류상 PF 계열(layer-type 정책 쪽).
  최근접 선례 R4/R7(SPAN_TYPE type-aware span sizing, TTFT 악화 — no-cudagraph·Zamba2·SUPERSEDED, 방향만). anchor §9 "layer-type 런타임 정책 재개 안 함"과 충돌 → **사용자 결정 필요**. 기전은 B4에서 부정.
- **(E)** PAUSE 술어는 C_switch 없으면 부분 항등식. S0′ 표 발산에는 구조적 항등 이유(K2).
- **(F)** anchor는 사용자 지시로 고정됨. 규칙 형식·예측기 3판·사다리 변경은 **사용자 승인 + doc-steward**, 날짜 단 개정(덮어쓰기 금지). 전파: anchor §2·§3·§6 S2·§8·§9 · cost_model §3 A4/A8/A10·§5 · PROJECT_STATUS 배너 (a)(c)(d) · ROADMAP 배너 · CEM Claim E · CONSENSUS §5-14 · artifact §3·§11·§12.

### 개정판이 승계할 문장(인용 조건)
1. "A4a는 upstream `adjust_stream_groups` decode-bs 문턱 선택기(P1 auto-grid agnostic arm과 같은 기제)의 값 생성기 교체다. 새 스케줄링 기제가 아니다."
2. "A4a가 이겨도 offline 부하-인덱스 분할 선택 주장이지 step 경계 동적 제어 주장이 아니며, HE0를 반박하지 않는다."
3. "P_hyb 표와 P_gen 표의 차이는 같은 (bs, ctx) 셀에서의 예측기 값 차이이며, stake #1의 증거가 아니다."
4. "단일 모델에서의 표 발산은 hybrid 조성 효과로 귀속할 수 없다. 귀속은 모델 hold-out(A11)에서만 가능하다."
5. "A4b-2는 비용 가중 span 크기 조정이며 layer-type 런타임 정책의 재개가 아니다. 결정이 현재 span의 층 종류의 함수이면 이 문장은 거짓이 된다."
6. "shape A(입력 256)에서 A4b-2의 정의역은 엔진 공식상 공집합이다."
7. "A4a는 원 A4(압력 게이트·하향 전용)와 다른 정책이다."

## 개선 제안(판정과 분리)

- **K2 처방**: hybrid 귀속의 1차 운반체를 S0′에서 **A11(조성 다른 모델 ≥2–3개 hold-out)**로 옮긴다. S0′를 유지하려면 P_gen을 "ctx 기울기를 대상
  모델에서 재지 않는 예측기 + 수준만 α 보정"으로 정의(Transformer 대조 모델에서 이식 또는 transformer-only 커널 열거 roofline) — 단
  Transformer 대조는 3-a 모델 1개 추가(GPU), roofline은 "Bullet이 아니다" 반박 위험. 허수아비 여부는 사용자 positioning 결정.
- **K1 처방**: recal을 "held-out 점에서 구한 오프라인 α"로 할지 "사건 시점 α"로 할지 등록. 후자면 A4a를 "feedforward + 수준 재보정"으로 스코프하고 HE0 비해당 논증을 따로.
- **B4 처방**: 결정 규칙 0개의 계측부터 — VESSL에서 span GPU 시간 vs decode step 시간을 비교해 비동기 발행 아래 A4b-2의 정의역이 있는지 확인(engine-porter 기본 OFF 이벤트 계측). span이 이미 step을 덮으면 A4b-2는 무의미.
- **B6 처방**: static 대조를 threshold 0인 1행 `manual_divisions`로 같은 코드 경로에 구현. SLO마다 표를 새로(게이트 8).

## ★메인 세션 독립 재검증 (기록 시점, GPU 0)
- K2 전제 "estimator는 clamp·보간, 외삽 없음": `profile.py:224-238` `_bounds`가 target 밖이면 끝점으로 clamp, `_lerp`가 fraction을 [0,1]로 자름 — 확인.
- B5 "break 없이 마지막 만족 행": `multiplexing_mixin.py:1184-1188` 루프에 `break` 없음 — 확인.
- 조회 지점: `adjust_stream_groups` 호출 `:1368`(prefill admission 후)·`:1628` 2곳 — 확인(감사 인용 `:1236→1358`·`:1304`와 같은 경로).
- 사용자 지시("반영 전 감사")에 따라 **재구성은 반영하지 않았다.** 제안 파일에 NO-GO 배너만 부착.
