# 판정서 — `DESIGN_S0_REV1_2026-09-30.md` 규칙층 적대 감사

2026-09-30 · claims-auditor · **`NO-GO`** · 死因 **5** · 차단 **7** · 반증 실패 **6** · **GPU 0 · 파일 수정 0**

> ★감사 에이전트는 read-only라 **메인 세션이 반환문을 기록**한다. 말미 "메인 세션 독립 재검증"은 이 기록
> 시점에 메인 세션이 직접 코드로 확인한 것이다. 감사자가 쓴 재계산 스크립트(`s0probe.py`, `s0probe2.py`)는
> scratchpad에만 있으며 **미등록·자체검사 없음** — 그 수치(D 0.295 / 0.993–1.000 등)는 판정 논거로만 쓰고
> **인용 금지**(항목252 계열). 핵심 死因 K1·K2는 수치와 무관한 구조 논증이다.

재계산 원자료: λ0 job 908623 로컬 telemetry `results/r2_eval/lambda0_prereg/lam0_908623/tel_*.jsonl` 11셀.
결정점 = `prefill_queue_depth>0 ∧ decode_running_batch_size>0`, 스냅숏 개수 기준(시간가중 아님). 규칙은 설계 §1 F2
문면 그대로. `d_min`은 정본 `ConservativeDecodeFloorEstimator.estimate`를 import해 계산. 합성 프로파일은 설계 §2.2
식, knee 3종(mild/mid/steep)은 **감사자 선택**(새 자유 표면 — 그래서 K1·K2는 knee 무관 논증, knee 의존 수치는 K3에서만).

## 단일 판정 질문에 대한 답

S0는 자기 질문("hybrid 고유 입력이 Bullet식 결정을 바꾸는가")을 **측정하지 않는다.**

- **(a) 재생 경로의 D는 계산 전에 정해진다.** 개방 루프라 `current`가 기록값(0/44/108)에 고정되고, 결정점 중 격자
  최소 16에 있는 것이 0개라 H-SQUEEZE는 PAUSE를 한 번도 낼 수 없다. G-PAUSE는 slack ≥ 1 step이면 무조건 PAUSE.
  ⇒ D는 비용 격자와도 `attn_ratio`(0~1)와도 무관한 상수다.
- **G-PAUSE는 Bullet이 아니다.** Bullet Algorithm 1의 decode 규칙(`ReduceDecodeSM`, "SLO를 만족하는 최소 SM")은 decode
  ctx `cl_i`를 입력으로 받는 **예측기**로 최소 SM까지 줄이는 규칙이고, 일시중단은 극단 부하의 끝단이다. 이 구조는
  H-SQUEEZE와 같다. 즉 H 대 G는 "Bullet식 예측형 vs 허수아비 반응형"이다.
- **hybrid 모델로 프로파일한 Bullet(= 충실한 G)은 정의상 H와 같다.** `estimate(batch, ctx, slo)` 시그니처에 hybrid 항이
  없고 hybrid성은 프로파일 **값**으로만 들어온다. ⇒ D ≡ 0.
- 결론: `DIVERGES`/`NO_DIVERGENCE` 어느 쪽이 나와도 "Bullet on hybrid" 반박에 정보가 없다.

## 死因 5건

**K1 ★★★ (항등식) (a)의 발산률은 규칙의 분기 순서와 개방 루프에서 연역된다.** H의 PAUSE 가지는 "`d_min` < 현재면
SQUEEZE"를 먼저 검사하므로 `current ≤ 16`에서만 도달. 재생 결정점의 `decode_sms` ∈ {0, 44, 108}. ⇒ label_H ∈
{SQUEEZE, NOOP}이고 G=PAUSE인 점은 전부 자동 발산. M1(`attn_ratio` ∈ {0, 1/7, 1/4, 1}) 전부에서 D가 바이트 동일 —
"격자 전체에서 견고"는 견고성이 아니라 격자가 D에 들어가지 않는다는 뜻. prefill `k` 축은 F2에 아예 안 들어간다.
longctx 3F2(죽은 가지)와 같은 형태, 교훈 9 재발.

**K2 ★★★ (estimand 부재) G-PAUSE는 Bullet의 허수아비이고, 충실한 Bullet-on-hybrid는 H와 항등이다.** Bullet 원문
(`Papers/Bullet.pdf`): ES = (`sl`,`pbs`,`pm`,`cl`,`dbs`,`dm`), decode ctx `cl`이 예측기 입력 · `ReduceDecodeSM`으로
SLO 만족 최소 SM, 일시중단은 극단 부하만 · "Extending the latency model to cover new attention variants … is
straightforward and left to future work". 설계의 G-PAUSE는 예측기 없는 실측 p95 반응형이고 PAUSE를 먼저 검사 —
Bullet과 **정반대 우선순위**. 설계가 말하는 hybrid 고유 입력(ctx 기반 decode floor)은 `profile.py:303` `estimate()`이며
attn/SSM 항이 없다. ⇒ 판정량은 "hybrid 입력 효과"가 아니라 "예측형 vs 반응형 + 분기 순서" 효과. 단일 질문의 estimand가 비어 있다.

**K3 ★★ (반전) 등록 문면이 허용하는 해석 폭만으로 라벨이 뒤집힌다.** H 분기 순서 모호("d_min이 격자 최소이고 압력
지속이면 PAUSE"가 먼저인지 나중인지)만으로 shape B에서 D ≥ 0.10 셀 비율 100% → 11%. G의 "1 step" 정의만으로 shape A
D 0.295 → 1.000. PAUSE-먼저 해석에서 c0 격자 끝점 한 칸 이동으로 shape B 비율 0.11 → 0.33 → 0.67(20% 경계 통과,
`NO_DIVERGENCE` → `MODEL_DEPENDENT`). 교훈 8.

**K4 ★★ (라벨 비전사) 도달 가능한 결과가 어떤 라벨에도 안 들어간다.** 문면 그대로면 결과는 "D ≥ 0.10 격자 100% ∧
M1 실패"인데 `DIVERGES`는 변이검사 통과 요구, 나머지 셋은 D < 0.10 또는 20–80% 요구 ⇒ 라벨이 사후 재량. M2도 같은
결함: G+의 우선순위 미등록 — PAUSE-먼저 유지면 G+ ≠ H 구조적, "d_min으로 SQUEEZE" 채택이면 G+ ≡ H 구조적 ⇒ M2는
측정이 아니다. M1 통과 조건 "방향이 바뀜"은 비율 D에 대해 정의되지 않는다.

**K5 ★★ (b) 폐쇄 루프의 "같은 결정점"이 정의되지 않는다.** G와 H가 다른 상태 궤적을 만드는데 D의 분자·분모가 어느
궤적 위인지 미등록. 설계 :81이 가리키는 "최소 큐 모형(§2.2)"은 §2.2에 없다(비용식만 있음). PAUSE/SQUEEZE가 다음
상태에 주는 효과를 정하는 것은 등록되지 않은 시뮬레이터. 처치 이후 궤적으로 결정점을 조건화 = longctx per-protocol
collider와 같은 자리.

### 반전 시험 표

| 자유 표면 | 민 범위 | 판정 변화 |
|---|---|---|
| H 분기 순서(:58 문면 모호) | SQUEEZE-먼저 ↔ d_min=16이면 PAUSE-먼저 | shape B 셀 비율 1.00 → 0.11, `DIVERGES`권 → `NO_DIVERGENCE`권 |
| G "1 step" 정의(:57 미등록) | ewma / p95 / 1 ms | shape A D 0.295 / 0.270 / 1.000 |
| c0 격자 끝점(교훈 8) | {8,13,20} → {13,20,24} → {20,24,30} | PAUSE-먼저에서 B 비율 0.11 → 0.33 → 0.67 |
| attn_ratio(M1) | 0 → 1 | 변화 없음(D 바이트 동일) = hybrid 입력 무관(K1) |
| 2F9 술어 필드 | `decode_batch_size` ↔ `decode_running_batch_size` | 결정점 모집단 자체가 바뀜(B1) |
| 정본 `decide()` 재사용 vs 신규 규칙 | :60-62 vs :57-58 | shape A 결정점 다수에서 정본은 둘 다 긴급 108(UPSHIFT, 라벨 공간 밖) |

## 차단 7건

- **B1** 2F9 술어 `decode_batch_size > 0`(:52)은 무효 필터 — telemetry는 running batch가 0이어도 이 값을 1로
  채운다(`multiplexing_mixin.py:491` `decode_batch_size=max(1, batch_size)`). `decode_running_batch_size`를 써야 한다.
- **B2** "정본 함수 재사용, 재구현 금지"(:60-62)는 실현 불가 — 순수 함수는 `estimate()`뿐. 두 `decide()`는 상태를
  가진 `CoarseGrainedController.stabilize`(케이던스 HOLD·`downshift_epochs=3`·dwell·`_live_underprediction`)를 거친다.
  `GenericDynamicPolicy`의 slack 규칙은 인라인이고 PAUSE도 τ_age도 없다 ⇒ G-PAUSE는 새 코드.
- **B3** 미등록 상수: τ_age, "1 step", knee s(24)/s(34)(estimator는 16/24/34/44에 점이 있어야 동작), heldout
  residual(margin), SLO(telemetry 60 ms), 셀 간 풀링 가중.
- **B4** 격자 축 결함: `c_attn·attn_ratio`는 곱으로만 들어가 축 중복 · `k`는 F2에 안 쓰이는 죽은 축 ⇒ 162셀 중 유효
  54 · 등록 범위에서 ctx 항은 8K에서도 c0의 ≈13% 이하 ⇒ 격자 구성 자체가 M1 결과를 미리 정한다.
- **B5** "fixed D44" 전제와 달리 결정점의 `current`는 대부분 108(realized ≠ target, 교훈 246). 개수/시간 가중 규약
  미등록(교훈 252). 스냅숏 필드 간 read-skew 가능성(§3 항목120).
- **B6** 결정점이 과부하 셀에 몰림(shape A 다수가 점유율 ≥ 0.90, shape B r2–r3 oldest age 중앙값 용량 밖). 용량 대비
  rate 층화 없이 풀링하면 metric-cliff 계열 왜곡.
- **B7** W4 표기 오류: 설계 :81 "8K/256"이나 실제 W4는 (8192, 64) ↔ (256, 512)(`workloads.py:191-193`).

## 반증 실패 6건

1. M0 항등 자기공시는 사실(λ0 `max_running_requests=48`·`disable_radix_cache`·`max_mamba_cache_size=None` ⇒
   `model_runner_kv_cache_mixin.py:222-230`에서 슬롯 수 = running 상한). F1을 판정에서 뺀 것은 정당.
2. 질문 5의 전제("shape A 결정점 거의 0")는 거짓 — running 술어 기준 결정점이 다수 있다.
3. 게이트 1(성능 판정 없음)·GPU 0은 문면상 준수.
4. (a)/(b) 불일치 시 보수적 선택 규칙은 결함 아님.
5. M3·M5의 형태는 교훈 53·254에 부합.
6. "라벨 발산 vs 크기 발산"(정책인가 상수인가) 구분 개념은 옳다 — 단 K1 때문에 이 설계에서는 작동하지 않는다.

## 개선 제안(판정과 분리)

- **S0′**: 결정 규칙을 **Bullet Algorithm 1 decode 쪽 하나로 고정**(예측기로 최소 SM까지, 예측 TPOT가 SLO 안일 때만
  일시중단)하고 **예측기만** 바꾼다: P_hyb(attn/SSM 분리 프로파일) · P_gen(같은 모델 두 점 보정, Bullet Fig. 9 방식, 전
  층 attention 가정 ctx 기울기) · P_gen+recal(Bullet online 재보정 모사 — 없으면 다시 허수아비). estimand = 예측기
  형식만으로 생기는 d_min·라벨 발산. `estimate()`가 순수라 CPU 실현 가능.
- **(b) 폐쇄 루프는 삭제**하거나, 제3의 고정 정책이 만든 공통 상태 스트림 위의 개방 루프로 한정(동역학 서술 금지).
- **그래도 합성 격자에서는 `MODEL_DEPENDENT`에 머물 수밖에 없다.** 순서는 대여 기판에서 hybrid decode 지연을
  (SM × batch × ctx)로 **실측 프로파일**부터(= 3-a). 결정 규칙 0개의 계측. S0′는 그 다음.

## 설계 §6 질문 1–7 답

1. **허수아비.** Bullet 주 행동은 예측기 기반 최소 SM, 일시중단은 끝단. G-PAUSE는 우선순위 역전 + 예측기 부재(K2). 발산은
   부풀려지는 수준이 아니라 **구조적으로 생성**된다.
2. **구조적 이유 있음.** (a)에서 H는 PAUSE 도달 불가, G는 PAUSE 먼저 ⇒ label_H ≠ label_G ⇔ G=PAUSE(K1). 분기 순서를
   뒤집으면 d_min=16에서 H=G=PAUSE로 발산 0(K3).
3. 문면 그대로면 격자가 D에 전혀 안 들어간다(끝점 의존보다 나쁜 항등식). PAUSE-먼저 해석에서는 격자 밖 1점이 라벨을
   20% 경계 너머로 옮긴다.
4. 자기 선택 편향은 **DIVERGES 쪽**(`current`가 16에 못 가 H의 PAUSE 억제). 폐쇄 루프에서는 부호 미정이고 estimand
   자체가 미정의(K5).
5. 전제 거짓(shape A 결정점 다수). 단 대부분 점유율 ≥ 0.90·`current`=108인 과부하 상태. shape B 5셀만으로는 단일
   체제(ctx≈8K·batch≈2·용량 밖)라 `MODEL_DEPENDENT` 이상 불가.
6. `INPUT_ONLY`가 나오면: 무의미해지는 S1 arm = "H vs G-PAUSE", "hybrid 전용 정책 vs generic". 남는 arm = 같은 규칙에서
   hybrid 프로파일 예측기 vs generic 예측기(= 기존 H-Policy/Claim E 등록 범위). **논문 novelty 상한** = "Bullet/MuxWise식
   최소-SM 규칙에 hybrid 보정 프로파일을 **입력**으로 넣으면 결정이 바뀐다"까지이며, 이것도 Bullet이 "straightforward,
   future work"로 예고한 확장이라고 적어야 한다. **금지 서술**: "hybrid 전용 스케줄링 정책", "새 기제". (현 설계의 M2는
   측정이 아니므로 이 축소는 S0′ 결과에만 적용.)
7. longctx 死因 계보 셋을 **모두** 다시 밟는다: 처치 미발화 f≈0(H의 PAUSE 0건) · per-protocol 조건화((b) 궤적 의존
   결정점) · 공선(shape A vs B는 ctx·batch·과부하 여부가 동시에 다름, W5는 고정 rate에서 ctx가 압력과 함께 이동). 죽은
   가지(3F2형)도 재발: M1 "방향", M2.

**(D) S1이 GO여도 "Bullet on hybrid" 반박이 성립하는 경로**: (i) 충실한 Bullet-on-hybrid baseline이 공개 코드로 재현
불가(예측기 `.so` 비공개, `impl_vs_external_pdmux_2026-08-28.md` §3.5) (ii) libsmctrl CUDA ≤ 12.6, TPC(2 SM) 입도·부분
공유·프로세스 분리가 green-context 4-SM 격자와 달라 우열을 기판·기구 탓으로 돌리는 반박이 항상 가능 (iii) Bullet은
layer 단위 진행도(`L_step`)와 재정렬을 함께 쓰므로 일부만 이식한 arm은 "Bullet이 아니다" (iv) Bullet의 online 재보정을
넣지 않은 generic arm은 허수아비.

## ★메인 세션 독립 재검증 (이 기록 시점, GPU 0)

- **B1 확인**: `multiplexing_mixin.py:491` `decode_batch_size=max(1, batch_size)` — 메인 세션이 감사 전 스냅숏 생성 코드를
  읽을 때 이미 본 줄. 설계 §1 F2 술어는 무효 필터였다.
- **K2의 "estimate()에 hybrid 항 없음" 확인**: `profile.py:303-` 시그니처 `estimate(batch_size, context_tokens, itl_slo_ms,
  runtime)`, `_point_value`(`:239-247`)는 `profile.points`만 조회. `attention_ratio`의 소비처는 저장소 전체에서
  `benchmarks/pdmux_eval/profile_cli.py:106`(메타 기록) 1곳뿐 — engine-porter 독립 검토도 동일 결론("M1을 profile
  메타필드로 하면 항등식").
- **B7 확인**: `workloads.py:191-193` `W4_PREFILL_SHAPE=(8192,64)`, `W4_DECODE_SHAPE=(256,512)`. 설계 :81 오기.
- **M0 확인**: `model_runner_kv_cache_mixin.py:222-230` 분기 메인 세션 직접 독해(감사 전).
- 감사자의 D 수치·knee 3종·반전 표 수치는 **재현하지 않았다**(scratchpad 스크립트 미등록). 판정은 K1·K2 구조 논증만으로
  성립하므로 수치 재현 없이 `NO-GO`를 기록한다.
- ★**해석 정정(같은 날, 사용자 지적)**: §6-6의 "Bullet future work 예고"는 **인용 의무**로만 승계한다. 선행 논문이 미착수로
  남긴 확장은 아직 아무도 하지 않았으므로 그 사실이 novelty를 낮추지 않는다. 이 판정서에서 유효하게 남는 것은 K1·K2·K3·K4·K5(rev1
  **설계** 결함)와 "충실한 baseline(P_gen+recal) 없이는 허수아비"라는 조건뿐이다. novelty 조건은 "확장이 자명하지 않음"을 보이는
  것으로 재정의한다(`DESIGN_S0_REV1` 배너). 엔진 수정 비용은 사용자 결정으로 중단 사유가 아니다.
