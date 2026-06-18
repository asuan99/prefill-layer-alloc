# 실험 설계 — E5 optional "real prefill" 모드 (A2 close-out)

작성일: 2026-06-18 · 브랜치: **`exp/e5-real-prefill`** (base: `v2-prefill-layer-alloc`)
목적: 검수 [A2](review_checklist.md)가 지적한 microbench 한계 — E5의 prefill이 *scan/attn token-mixer 1 chunk·batch=1*뿐이라, GEMM-heavy·다중 chunk인 **진짜 prefill에서 overlap·분할 결론이 유지되는지 미검증** — 를 닫기 위한 optional 측정 경로.
관련: [widened 검증](widened_sweep_validation.md) · [closure §5.1](project_closure_report.md) · [추가가치 Path 3](additional_value_paths.md)

---

## 1. 설계 원칙 — optional, 기존 불간섭

- **기본값 = 현재 microbench 그대로.** 새 플래그 없이 실행하면 byte-for-byte 동일(layer=ssm/attn, batch=1, 단일 chunk). 기존 `results_v2/e5/serving_coexec_*.csv`·그림·게이트 **회귀 0**.
- **full 모드는 별도 파일·별도 스키마.** `serving_coexec_full_*.csv`로 써서 microbench CSV를 절대 덮지 않음. full 스키마는 micro 스키마 + `prefill_mode`/`prefill_batch`/`n_chunks` 3개 metadata 컬럼.
- **세 토글이 독립**이라 fidelity를 점진적으로 올릴 수 있다(아래 §2).

## 2. 추가된 토글 (전부 default off)

| 플래그 | 기본 | 효과 |
|---|---|---|
| `--prefill-mode {micro,full}` | `micro` | `full` → prefill을 **GEMM-inclusive** 커널로: `ssm`→`ssm_full`(in_proj+scan+out_proj), `attn`→`attn_full`(**신규**: qkv_proj+SDPA+o_proj). compute-bound·SM-filling → 진짜 prefill 근사. |
| `--prefill-tokens N` | `0`(=chunk) | `N > chunk`면 **multi-chunk**: `ceil(N/chunk)`회 chunk 커널 반복(예: 4096 = 16×256). prefill duration을 현실화(2× artifact 점검). |
| `--prefill-batch B` | `1` | in-flight prefill 시퀀스 수. 진짜 serving의 prefill 배칭. |

구현: 커널은 `experiments/common/kernels.py`(`build_attn_full_fn` 등 + `attn_full` dispatch), 러너는 `experiments/e5_serving/run_serving_coexec.py`(모드 해석·multi-chunk wrapper·스키마/파일 분기·`--dry-run`).

## 3. 실행

```bash
# (GPU 불필요) 스윕 플랜만 검증 — 제어 로직 sanity
python -m experiments.e5_serving.run_serving_coexec --dry-run --prefill-mode full \
    --prefill-tokens 4096 --prefill-batch 4

# (SXM4) A2 최소 검증 — 최고 셀 + window 양 끝, GEMM-inclusive·4k multi-chunk
env -u BASH_ENV bash experiments/slurm/submit_size_sweep.sh e5 -- \
    --prefill-mode full --prefill-tokens 4096 --prefill-batch 4 \
    --decode-batches 8 256 --context-lens 4096
# 산출: results_v2/e5/serving_coexec_full_{model}_a100_sxm4_80gb.csv
```
(`submit_size_sweep.sh`는 `--` 뒤 인자를 그대로 forward → 제출 스크립트 수정 불필요.)

## 4. 통과 기준 (A2가 닫히는 조건)

`micro` vs `full` 두 CSV를 같은 셀에서 비교해 **세 정성 결론이 유지되는지**:
1. **(분할)** `green_ctx`가 `two_stream`을 못 이김(`two_stream/green_ctx`<1.0)이 full에서도 유지.
2. **(overlap 스케일)** speedup이 prefill 크기에 따라 변하는 양상이 일관 — 단 full은 prefill이 훨씬 길어 **2×가 줄어들 것으로 예상**(GEMM이 SM을 채워 slack↓). 그 *감소 폭*이 핵심 측정치.
3. **(window)** decode batch↑로 window가 닫히는 경향 유지.

→ 1·3 유지 + 2의 감소가 "치명적이지 않음"이면 microbench 결론을 진짜-prefill로 승격. green_ctx가 full에서 이기는 셀이 생기면 → [Path 1/3](additional_value_paths.md)로 합류(헤드라인 재검토).

## 5. 알려진 단순화 (정직성 — 다음 fidelity 단계)

- **multi-chunk = timing proxy.** chunk 커널을 `n_chunks`회 반복해 *연산량·duration*을 근사하나, SSM **state passing**(chunk 간 상태 전달)·attn **KV 누적 증가**(chunk i가 i-1까지 attend)는 미반영. → prefill 절대 latency는 충실, chunk 간 의존 효과는 미포착.
- **prefill context_len=0.** prefill은 자기 프롬프트를 causal 처리(맞음). 셀의 `context_len`은 decode-side KV에만 적용(기존과 동일).
- **여전히 full *model* forward는 아님.** 단일 layer-type(ssm_full/attn_full)의 반복이지 전체 모델(모든 레이어 interleave + embedding/LM head)은 아님. A2의 핵심(GEMM 유무·prefill 길이)은 닫되, 진짜 엔진 통합(chunked-prefill 배치 융합)은 별도 작업.
- **green_ctx `f`는 여전히 prefill 우대(0.5/0.7).** widened §1.3이 보인 decode starvation은 full에서도 재현될 것 — decode-보호형 `f` 스윕은 [검수 A5](review_checklist.md)의 별도 항목.

## 6. 변경 파일 (브랜치 `exp/e5-real-prefill`)

- `experiments/common/kernels.py` — `build_attn_qkv_proj_fn`·`build_attn_o_proj_fn`·`attn_full` dispatch(+`VALID_PREFILL_TYPES`).
- `experiments/e5_serving/run_serving_coexec.py` — `--prefill-mode/-tokens/-batch`·`--dry-run`·`_SCHEMA_FULL`·multi-chunk·`_full` 출력 분기.
- 기존 micro 경로·스키마·파일명 불변(검증: `--dry-run` micro = 기존과 동일, `py_compile` OK).
