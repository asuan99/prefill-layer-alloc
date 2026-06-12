# Archive — v1 (7B saturation-point study)

이 디렉토리는 연구 방향 v1 → v2 전환에 따라 **동결(frozen)** 된 v1 실험 스냅샷이다.

## v1이 무엇이었나

- **main model:** 7B 하이브리드 (Zamba2-7B-Instruct, Falcon-H1-7B-Instruct)
- **RQ:** "prefill에서 layer-type별 SM 할당의 **포화점(saturation point)** 은 몇 SM인가?"
- **구조:** Stage1 (SM scaling sweep) → Stage2 (전환 overhead) → Stage3 (HM 정책 평가)

## 왜 archive했나 (v2 전환 사유)

1. **main model 이전:** 7B → SLM (Zamba2-1.2B / Falcon-H1-1.5B)
2. **RQ 전환:** "포화점은 몇 SM?" → "포화 **메커니즘**이 bandwidth인가 grid인가?"
   (이것이 v2 전체 framing의 단일 실패점 = G1 gate)
3. **계측 승격:** `BandwidthEstimator`를 부차 → **주 계측기**로
4. **축 추가:** sweep에 **batch 축** 추가 (grid 메커니즘 판별에 필수)
5. **decode 정책 기각:** decode-내 SM reconfig 정책은 폐기. decode는 donor/fallback 역할만.

## 구성요소별 유효성 상태

| 구성요소 | 상태 | 사유 |
|---|---|---|
| Stage1 SSM/Attn sweep | **측정 무효** | full-layer scope 버그(GEMM 포함) + n_blocks 28× 오류 + batch 축 부재 |
| Stage1 chunked 결과 | 부분 유효 | chunk granularity는 v2도 채택, 단 batch 축 없어 재측정 필요 |
| Stage2 overhead | **유효** | 전환 비용 측정 로직은 v2 temporal 정책에 그대로 재사용 |
| Stage3 Policy A (fixed) | 참조용 | baseline 개념은 유지, decode 포함분만 폐기 |
| Stage3 Policy B (step) | **폐기** | decode-step 적응 정책 자체가 기각됨 |
| Stage3 Policy C (layer) | 한정 재사용 | decode 제거 후 Zamba2 temporal 전용으로 v2에서 부활 |
| B-1 버그 (dispatch 누락) | **미해결** | v2에서 최우선 수정 — archive 코드엔 버그 그대로 남김(기록용) |

### 보충 메모

- **n_blocks 28× 오류:** v1 일부 경로가 model-agnostic `batch × seq // 4` 공식을 썼다.
  올바른 공식은 `batch × ceil(seq/chunk) × n_heads`. Zamba2(n_heads=112)에서 약 28× 과소
  추정이 발생했다. v2는 `experiments/common/wave_model.py` 단일 출처로 통일한다.
- **Stage2 overhead 값 (재참조용):** `results/stage2/ctx_switch_overhead_a100_sxm4_80gb.json`
  의 layer-boundary swap 비용은 **약 7.8 μs/swap** (sync 포함, ssm↔attn).
  v2 E0의 `temporal_overhead_budget` 표와 E4 temporal 정책이 이 archive 결과를 그대로
  재참조한다 (= "복사본 유지": v1 stage2는 여기 archive에 보존되며 v2가 중복 생성하지 않는다).
- **B-1 버그 위치 (기록용):** prefill kernel dispatch가 layer_type을 분기하지 않고 항상
  SSM kernel을 실행하던 문제. archive의 `stage3_hm_eval/coexec_microbench.py::_build_prefill_fn`
  은 항상 chunked-SSM prefill을 만든다 (attn config여도 attn kernel을 실행하지 않음).
  v2 `experiments/e4_concurrent/run_concurrent_ab.py`에서 layer_type 분기 + 단위테스트로 수정.

## 이동 내역 (git mv, 히스토리 보존)

```
stage1_sm_scaling/   ← workspace/characterization/stage1_sm_scaling/
stage2_overhead/     ← workspace/characterization/stage2_overhead/
stage3_hm_eval/      ← workspace/characterization/stage3_hm_eval/   (Policy A/B/C 전부)
results/             ← workspace/characterization/results/          (stage1, stage1_v2, stage2, stage3, decode_scaling 전부)
```

- `src/` 는 **이동하지 않았다** (v2가 그대로 import: `src.models.layer_runner`,
  `src.smctrl`, `src.profiling.metrics` 등).
- `configs/` (→ `shared/configs/` 심볼릭 링크) 도 이동하지 않았다. v2는 동일 configs를
  재사용하되 1.2B/1.5B 항목만 참조한다.

---

v2 실험은 `experiments/`에서 진행. 이 디렉토리는 재현·감사 목적의 동결 스냅샷이며 수정하지 않는다.
