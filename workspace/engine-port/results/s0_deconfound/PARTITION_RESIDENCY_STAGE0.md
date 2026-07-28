# Stage 0 파티션 활성시간 재집계 — FINDINGS_865098 §4의 (a)/(b) 판별 (정본 아님)

2026-07-27. 입력 = 기존 `results/stage0_xctrl/*_telemetry.jsonl` (jobs 864230 / 864601)
+ 같은 캠페인의 client `STAGE0_SUMMARY`. **신규 GPU 실행 없음.**
스크립트 = `stage0_partition_residency.py`, `stage0_realized_sm.py` (이 디렉터리).

DESIGN.md §5의 사전등록 규칙(정본 수정 전 claims-auditor)에 따라 **여기까지만 기록**한다.
PROJECT_STATUS / CONSENSUS / CLAIM_EVIDENCE_MATRIX 는 의도적으로 손대지 않았다.

## 0. 한 줄

Stage 0의 de-confound 앵커였던 **D108 arm은 108 SM에서 돌지 않았다** — decode 작업이
있던 샘플의 82–97%에서 **realized decode 파티션 = 16 SM**(D16 arm과 동일 stream group).
따라서 `D16 vs D108 = 1.00±0.01`은 **같은 파티션을 두 번 측정한 값**이며, Stage 0의
헤드라인("decode는 운영점서 SM-무감각")을 지지하지 않는다.

## 1. 계측 방법 (PIN_CHECK가 못 본 것)

- `stage0_pdmux_capture.sbatch:165-180`의 PIN_CHECK는 `controller_decision.target_decode_sms`
  = **정책의 목표값**만 히스토그램했다.
- `runtime_snapshot` 행에는 **실현값**이 있다: `dual_worker.py:608`이
  `sm_counts[arbiter.stream_index]`를 그대로 `prefill_sms`/`decode_sms`로 찍는다.
  stream_index 0 = `(108,0)` = 분할 없음, 1 = 설정된 분할, 마지막 = `(0,108)` = decode 전용.
- 가중치 3종을 모두 보고: 샘플 수 / 샘플 간 wall-time / decode step 수.
  **`decode_step_count`는 전 파일에서 항상 0**(dead field, `decode_last_tpot_ms`와 동일한
  증상) → step 가중은 불가. 아래는 **decode 작업 in-flight 샘플(`decode_running_batch_size>0`)**
  로 조건화한 값이다. 샘플링은 `PDMUX_DUAL_WORKER_TRACE_EVERY` 기본 32(≈16 iteration당 1).

## 2. 실현 파티션 (decode-active 샘플 기준)

| arm | ctx | D16 | D44 | D92 | **D108** |
|---|---|---|---|---|---|
| H | 4096 | 16SM 76% | 44SM 84% | 92SM 90% | **16SM 82%** |
| H | 8192 | 16SM 84% | 44SM 91% | 92SM 79% | **16SM 93%** |
| H | 16384 | 16SM 94% | 44SM 80% | 92SM 97% | **16SM 96%** |
| M | 4096 | 16SM 80% | 44SM 90% | 92SM 90% | **16SM 87%** |
| M | 8192 | 16SM 87% | 44SM 94% | 92SM 78% | **16SM 94%** |
| M | 16384 | 16SM 93% | 44SM 96% | 92SM 97% | **16SM 97%** |
| T | 4096 | 16SM 69% | 44SM 74% | 92SM 90% | **16SM 86%** |
| T | 8192 | 16SM 68% | 44SM 85% | 92SM 94% | **16SM 80%** |
| T | 16384 | 16SM 60% | 44SM 78% | 92SM 97% | **16SM 96%** |

나머지는 전부 `(0,108)` decode-전용 구간(prefill 미동거)이다. 즉 **모든 arm이
"동거 중엔 nominal 분할, decode-only일 땐 108"** 이고, D16/D44/D92는 nominal을 실현했으나
**D108만 nominal을 실현하지 못했다**(동거 중 16 SM).

## 3. 왜 그렇게 됐나 (코드 근거)

- 하네스는 D108 셀에 `CFG[108]="pdmux_d16.yml"`(sm_counts `[(108,0),(92,16),(0,108)]`)를
  주고 `PDMUX_R2_POLICY`를 **unset**했다(`stage0_pdmux_capture.sbatch:105-139`).
- 정책이 없으면 event loop는 legacy 경로 `adjust_stream_groups`로 간다
  (`multiplexing_mixin.py:726-733`): `manual_divisions`가 있으면
  `_, _, threshold = manual_divisions[i]` → `[92,16,**0**]`의 세 번째 원소 = threshold = 0
  → `decode_bs >= 0`이 **항상 참** → `stream_idx = 1 = (92,16)`.
  즉 **running batch가 있고 split prefill이 있으면 무조건 16 SM 분할로 들어간다.**
- decode만 남으면 `set_current_stream_idx(real_sm_group_num-1)` = `(0,108)`,
  둘 다 없으면 0. → §2의 관측과 정확히 일치.
- 부가로 "keepalive=0이면 무경합"이라는 전제도 성립하지 않았다: client 자체가 conc=32로
  4k–16k 프롬프트를 계속 밀어 넣으므로 **D108 arm의 decode-active 샘플 중 82–96%가
  prefill과 동거**했다.

## 4. client 수치가 독립 확인 (같은 캠페인, 3-rep 평균 ITL p50)

| arm | ctx | D16 | D44 | D92 | D108 | D16/D92 | **D108/D16** |
|---|---|---|---|---|---|---|---|
| M | 4096 | 19.27 | 9.88 | 7.83 | 19.22 | 2.46× | **0.997×** |
| M | 8192 | 16.96 | 8.75 | 7.02 | 16.83 | 2.42× | **0.992×** |
| M | 16384 | 16.33 | 8.15 | 6.62 | 16.26 | 2.47× | **0.996×** |
| H | 4096 | 45.62 | 20.38 | 13.61 | 45.49 | 3.35× | **0.997×** |
| H | 8192 | 45.38 | 20.17 | 13.28 | 45.20 | 3.42× | **0.996×** |
| H | 16384 | 46.77 | 19.82 | 13.06 | 46.66 | 3.58× | **0.998×** |
| T | 4096 | 17.81 | 10.14 | 8.67 | 17.82 | 2.05× | **1.001×** |
| T | 8192 | 18.37 | 10.96 | 9.59 | 18.27 | 1.92× | **0.995×** |
| T | 16384 | 25.31 | 13.83 | 11.59 | 25.27 | 2.18× | **0.998×** |

D108이 진짜 108 SM이었다면 **D92(13.06 ms)보다 빨라야** 한다. 실제로는 D16(46.77 ms)과
**0.2% 이내로 일치**한다 — telemetry의 파티션 판독과 완전히 정합적이며, 9/9 셀에서 재현된다.
즉 `1.00±0.01`의 그 유명한 tightness는 SM-무감각의 증거가 아니라 **동일 조건 반복측정**의
증거였다.

## 5. FINDINGS_865098 §4의 (a)/(b)에 대한 판정

- §4가 세운 가설("Stage 0의 sub-108 파티션이 상당 시간 **비활성**이라 D16이 D108처럼 보였다")은
  **결론은 맞고 방향은 반대**다. D16이 108로 새어나간 게 아니라 **D108이 16으로 고정**됐다.
- 결과적으로 **Stage 0의 유일한 de-confounded 대조가 무효**다. Stage 0은 decode-SM 축의
  108 끝점을 **측정한 적이 없다** ⇒ 865098의 기울기를 반박하는 근거로 쓸 수 없다.
- **(b)(음성대조 전제 오류) 쪽으로 증거가 기운다**: 음성대조 M의 기울기가 **서로 다른 두
  캠페인에서 재현**된다.
  - Stage 0(prefill 92/64/16 가변, keepalive 2, conc 32): M D16/D92 = **2.42–2.47×**
  - 865098(prefill **16 고정**, closed-loop in-flight 16, matched-batch 층화): M = **2.32–2.39×**
  두 캠페인은 prefill-SM 공변 confound의 유무가 정반대인데 M 기울기가 ~2.4×로 일치한다.
  T도 마찬가지(1.92–2.18× vs 1.89–2.04×). "pure Mamba2 decode는 O(1)이라 SM-bound 불가"는
  **context-length에 대한 O(1)일 뿐 SM 수에 대한 진술이 아니었다**는 §4(b)와 부합한다.
- 865098 자체의 파티션 실현도 재확인했다(같은 계측): P16D{16,24,44,92} 전부 nominal
  `(16,D)`를 68–94% 실현, 나머지는 `(0,108)` decode-only. **865098의 pin은 실현값 기준으로도 유효**.

## 6. 남는 한계 (여전히 미해소)

1. **(a)를 완전히 배제하진 못한다.** 두 캠페인 모두 decode batch가 D와 공변한다
   (865098 §2). 865098의 matched-batch는 사후 층화이고, Stage 0은 층화조차 안 했다.
   설계상 batch를 고정한 재측정(865098 §6-3)은 여전히 미실행.
2. 두 캠페인 공통으로 **decode-only 구간(0,108)이 4–40% 섞여** 있어 client ITL은
   혼합값이다. 위 표의 기울기는 그만큼 **과소평가**(하한) 성격.
3. `decode_step_count`가 dead라 step 가중 귀속 불가 — 샘플/시간 가중만 가능.
4. 이 재집계는 **Stage 0 헤드라인을 무효화**할 뿐, "hybrid decode가 SM-bound"임을
   **양의 방향으로 확정**하진 않는다. 그건 §6-1의 batch-통제 재측정 몫.

## 7. 정본에 미치는 영향 (제안, 미반영 — claims-auditor 선행)

- `PROJECT_STATUS.md` Stage 0 절 / `reports/CONSENSUS.md` §1-21·§3-9·§5-6 /
  `reports/stage0_verdict_2026-07-26.md` / `reports/longcontext_trace_plan.md` §0.6·§2·§6 /
  `CLAIM_EVIDENCE_MATRIX.md`(Claim A)의 **"D16 vs D108(무경합) = 1.00±0.01"** 근거는
  철회 대상이다. (memory `stage0-deconfound-contested.md`가 지적한 "D108은 keepalive 0이라
  무경합" 표현 오류도 같은 뿌리 — 실제로는 무경합도 아니고 108도 아니었다.)
- Stage 0에서 **살아남는 것**: D16/D44/D92 arm의 pin은 실현값 기준으로 유효하므로,
  그 세 점의 기울기 자체는 데이터로서 유효하다(단 §6의 batch·prefill-SM 공변 미통제).
- **재실행이 필요하다면** 최소 수정: D108 셀에 threshold를 만족하지 않는 config를 주거나
  `PDMUX_R2_POLICY=fixed`+`PDMUX_R2_FIXED_DSM=108`로 `(0,108)`를 명시 타깃팅하고,
  PIN_CHECK를 `target_decode_sms`가 아니라 **`runtime_snapshot`의 realized (prefill_sms,
  decode_sms)** 로 바꾼다(이 문서의 스크립트가 그 게이트다).

## 8. 교훈 (방법론 게이트 후보)

> **pin은 목표값이 아니라 실현값으로 검증한다.** 정책 target 히스토그램은 legacy
> auto-path·guard·fallback이 실제로 어느 파티션을 선택했는지 말해주지 않는다.
> 이번 건은 "policy OFF = 기준선"이라는 가정이 legacy 경로의 `threshold=0` 때문에
> 조용히 깨진 사례다. `results/*/telemetry.jsonl`이 이미 실현값을 담고 있으므로
> 비용은 0이다.
