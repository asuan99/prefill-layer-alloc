# Stage 0 de-confound — pre-registered design

작성 2026-07-26. 대상 = `reports/stage0_verdict_2026-07-26.md`가 **미실행으로 남긴**
"완전 de-confound 재측정". Stage 0 결론(운영점 decode SM-무감각)의 **방향을 뒤집으려는
실험이 아니라 airtight화**가 목적이다. 방향이 뒤집히면 그것대로 강한 결과지만,
사전 확률은 낮다고 본다(§5).

## 1. 무엇이 confounded였나

Stage 0의 coupled 스윕은 division `[108-D, D]`를 썼다. decode-SM을 D로 올리면
prefill-SM이 `108-D`로 **같이 내려간다**. 따라서 관측된 ITL(D) 곡선
(H 16k: 46.8→19.8→13.1 ms)은 두 원인이 섞여 있다.

- (i) decode가 SM을 더 받아서 빨라짐 — 측정하고 싶은 것
- (ii) prefill이 SM을 잃어 느려지고 경합이 줄어서 decode가 빨라짐 — 잡음

Stage 0는 이 곡선을 **CONFOUNDED로 판정**하고, 대신 공변이 0으로 붕괴하는 특수점
(D16 vs D108)만 인용했다. 그 판정 자체는 음성대조 M(원리상 SM-bound 불가한데도
동형 곡선)과 D108 앵커의 비단조성으로 견고하다. **본 실험은 (i)을 직접 분리해
곡선 전체를 해석 가능하게 만든다.**

### 1.1 D16 vs D108 앵커의 정확한 성격 (재인용 시 필수)

> ★★★**무효(2026-07-28, claims-auditor C1 CONFIRMED).** 아래 표와 그 결론
> ("등가는 SM 축소가 비용을 안 냈다는 상한 논증으로 유효")은 **틀렸다** — D108은
> "경합 있음"조차 아니라 **실제로는 decode 16 SM 자체였다**(legacy auto-path가
> `manual_divisions=[92,16,0]`의 threshold=0을 오독해 항상 stream_idx
> 1=(92,16)을 선택; realized telemetry 재집계로 decode-active 샘플의 79–96%가
> (92,16); `D108/D16` 클라이언트 서명 0.992–1.001인데 D92가 D16보다 3.4–3.6×
> 빠름 — 108 SM이 92 SM보다 느릴 수 없으므로 D108이 D16과 동일 조건이었음이
> telemetry 없이도 확인됨). 상세는
> [`PARTITION_RESIDENCY_STAGE0.md`](PARTITION_RESIDENCY_STAGE0.md),
> [`../../../PROJECT_STATUS.md`](../../../PROJECT_STATUS.md) "Stage 0" 절,
> [`../../../reports/CONSENSUS.md`](../../../reports/CONSENSUS.md) §1-21. 이
> 표는 이력 보존용으로만 남긴다 — **재인용 금지**.

하네스(`stage0_pdmux_capture.sbatch:106-108`) 확인 결과 두 점은
**"prefill 경합이 0인 두 점"이 아니다**:

| arm | decode SM | keepalive prefill | prefill 경합 |
|---|---|---|---|
| D16 | 16 (green-ctx) | 2 workers | **있음** |
| D108 | 108 (full stream) | 0 | 없음 |

즉 D16은 SM이 6.75× 적고 **동시에** prefill 경합까지 받는데 ITL이 D108과 같다.
두 요인 모두 D16을 느리게 만드는 방향이므로, 등가는 **"SM 축소가 비용을 안 냈다"**
는 상한 논증으로 유효하다(방향 견고). 다만 "두 점 모두 무경합"이라는 서술은
부정확하므로 정본 재인용 시 이 표를 쓴다.

## 2. 설계 — prefill-SM을 고정하고 decode-SM만 스윕

`manual_divisions: [prefill_sm, decode_sm, _]`에서 **세 번째 값은 엔진이 버린다**
(`pdmux_context.py:115`). green-ctx 생성은 2단 split이라
(`csrc/spatial/greenctx_stream.cu`, 심볼 확인) `a+b < 108`이면 **남은 SM은 부모
green-ctx 밖 = idle**이 된다. 따라서 아래가 **config만으로** 표현된다.

| arm | division | prefill SM | decode SM | idle SM |
|---|---|---|---|---|
| P16D16 | `[16,16,76]` | 16 | 16 | 76 |
| P16D24 | `[16,24,68]` | 16 | 24 | 68 |
| P16D44 | `[16,44,48]` | 16 | 44 | 48 |
| P16D92 | `[16,92,0]` (+guard) | 16 | 92 | 0 |

**prefill은 전 arm에서 16 SM 고정** ⇒ 경합 항이 상수, 변하는 것은 decode-SM뿐.
이것이 (i)의 순수 측정이다. 사전 게이트 = `greenctx_alloc_probe.py`
(§4)로 "decode_sm이 실제로 반영되는가"를 먼저 확증한다 — 지금까지 쓰인 모든
division이 정확히 108로 합해져서, 기존 데이터로는 `decode_sm`이 존중된 것인지
`108-prefill` 나머지였을 뿐인지 **구분 불가**하기 때문.

### 2.1 엔진 guard (우회 필수)

`_build_r2_policy`(`multiplexing_mixin.py:130-142`)는 `decode_sm ∈ {16,24,34,44}`
인 sm_counts 항목에서만 decode_states를 만들고, 비면 RuntimeError로 부팅 실패.
- P16D16/24/44 = 그 자체로 충족.
- **P16D92만** guard-satisfier division `[16,44,48]`을 하나 더 둔다(캡처만 되고
  선택되지 않음 — `PDMUX_R2_FIXED_DSM=92`가 idx를 정확히 매칭). Stage 0
  `pdmux_d92.yml`이 쓴 것과 같은 트릭이되, guard 쪽 prefill도 16으로 맞춰
  "혹시 선택돼도 prefill이 안 흔들리게" 한다.

## 3. 측정 규약 (프로젝트 방법론 게이트 준수)

- **steady-state, 용량 이하**: one-shot 32-conc burst(Stage 0) 대신 고정 arrival
  rate. metric cliff 회피(gate #6) — 여기서 보는 값은 goodput 지시함수가 아니라
  ITL 분포이므로 절벽 민감도는 낮지만, 용량 초과 시 큐잉이 ITL에 섞이므로 금지.
- **batch occupancy 고정**: decode batch 크기를 arm 간 동일하게(고정 동시성 +
  warmup 폐기 후 정상구간만). occupancy가 흔들리면 ITL(D)에 batch-size 효과가 섞인다.
- **step별 로깅**: decode step wall-time을 step 단위로 남겨 p50/p95/p99와
  분포(다봉성)를 본다. mean만 보지 않는다.
- **n≥4 reps** per cell, paired 비교. 3% 미만 차이는 headline 아님.
- **음성 대조 유지**: M(pure Mamba2) arm을 반드시 포함 — 원리상 decode가 SM-bound
  일 수 없으므로, M에서 유의한 기울기가 나오면 그것은 남은 confound의 지표다.
  (Stage 0의 핵심 방법론 교훈: coupled 스윕엔 음성 대조 + 무경합 앵커 필수.)
- **clock 로깅(신규 confound)**: arm마다 활성 SM 총량이 다르다(32 vs 108). 유휴
  SM이 많은 arm이 boost clock을 더 받으면 저-D arm이 부당하게 빨라져 **민감도를
  과소평가**할 수 있다. `nvidia-smi --query-gpu=clocks.sm` 샘플링을 arm마다 남기고,
  arm 간 clock 차가 유의하면 magnitude 해석에서 제외한다.

## 4. 사전 게이트 (먼저 통과해야 본 스윕 의미 있음)

`greenctx_alloc_probe.{py,sbatch}` (job 864669) — 모델·서빙 없이 green-ctx만 만들고
compute-bound matmul 처리량으로 **실효 SM 수**를 역산.

- `B(16,44) ≈ B(64,44)` ⇒ **decode_sm 존중** → 본 설계 config-only로 진행.
- `B(16,44) ≈ B(16,92)` ⇒ **나머지일 뿐** → config-only 불가, 엔진 변경 필요.
  이 경우 Stage 0의 D16/D44/D92는 구조적으로 prefill을 같이 움직인 것이며,
  de-confound는 별도 패치 트랙이 된다.
- 생성 자체가 거부 ⇒ 동일하게 엔진 변경 필요.

**결과 (job 864669, gpu36, A100-SXM4-80GB, 2026-07-27) = DECODE_SM_HONORED.**
baseline(108 SM, default stream) = 220.85 TFLOP/s 기준 실효 SM 역산:

| requested (a,b) | sum | A eff | B eff |
|---|---|---|---|
| (92,16) | 108 | 90.6 | **17.7** |
| (64,44) | 108 | 82.6 | **55.6** |
| (16,92) | 108 | 21.0 | **110.7** |
| **(16,44)** | **60** | 21.0 | **55.7** |
| **(16,16)** | **32** | 20.9 | **20.9** |
| **(44,16)** | **60** | 55.7 | **21.0** |

`B(16,44)=55.7 ≈ B(64,44)=55.6`(0.2% 차) 이고 `B(16,92)=110.7`과는 2× 차 ⇒
**decode_sm은 존중되고 sum<108의 잔여 SM은 idle**. §2 설계는 config-only로 성립.
`a+b>108`만 "Not enough SMs available"로 거부된다.

★**부수 관측 2건 (본 스윕 해석에 필요)**

1. **eff는 보정된 SM 수가 아니다** — 선형 환산이라 `B req 92 → eff 110.7`처럼
   물리 108을 넘는다. 작은 파티션이 SM당 처리량을 20–30% 더 받는다(클럭 헤드룸·
   SM당 대역폭). 판정은 **같은 b끼리의 like-for-like 비교**라 견고하지만, eff
   절대값을 SM 수로 인용하면 안 된다. 이 SM당 비선형성은 §3의 clock confound가
   **실재함을 정량 확인**해 준 것이기도 하다 — 저-D arm이 부당하게 빨라져
   **민감도를 과소평가**하는 방향으로 작용하므로, 평탄한 결과가 나와도 magnitude를
   "민감도 0"으로 읽지 말고 **상한**으로 읽는다.
2. **B는 (a+b) − A_actual** — A는 요청값에 가깝지만(A req 92 → 90.6) 상향 라운딩되면
   B가 요청보다 작아진다: `(92,16)`의 `B eff 17.7`은 다른 모든 16-요청(eff ~21.0)보다
   낮다. **본 설계는 전 arm이 a=16**이고 16은 라운딩이 없어(eff 20.9–21.0 일관)
   `B = (16+b) − 16 = b`가 arm 간 일관되게 성립한다 — 설계에 유리한 방향.

## 5. 사전 등록 예측과 해석 규칙

- **예측(H0, 사전 확률 높음)**: prefill 고정 시 ITL(D)는 **평탄**(16→92 SM에서
  arm 간 차 <5%), 3 arm 공통. ⇒ Stage 0 방향 확증, 곡선의 기울기는 전부 (ii)에
  귀속되어 CONFOUNDED 판정이 정량적으로 닫힌다.
- **반전(H1)**: prefill 고정에서도 유의한 단조 기울기(예: D16이 D92보다 >20% 느림).
  ⇒ Stage 0의 "non-binding"은 **특수점 등가에 기댄 과잉 일반화**였다는 뜻이고,
  decode floor가 실재. HE0/벡터1의 ctx-무관 강화 논거를 되돌려야 한다.
  이 경우 곧바로 정본을 고치지 말고 claims-auditor 반증을 먼저 건다.
- 어느 쪽이든 **magnitude는 이 하네스 한정**, scope {2.7–3B, triton, cudagraph-ON,
  ctx≤16k}는 Stage 0와 동일하게 유지한다.
