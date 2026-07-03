# Zamba2 SGLang Port — Troubleshooting Report (degenerate output → root cause: flashinfer head_dim=160)

작성: 2026-07-03 · 상태: **해결(정확성 확인)**. 남은 것 = 빠른 attention 백엔드 선정(perf).
관련: [zamba2_port_plan.md](zamba2_port_plan.md) · [triage/notes.md](../triage/notes.md) · 코드 `src/models/zamba2.py`

## 1. 증상
- Zamba2-2.7B 포트가 구조적으로 부팅·serve OK(dummy=gibberish 정상). **실가중치 531개 전부 로드**(skipped/missing=0). 그러나 실가중치 출력이 **토큰 0 반복**(빈 문자열)로 degenerate. NaN 경고는 로그에 없었음.

## 2. 디버깅 방법 (activation 국소화)
`Zamba2Model.forward`에 per-layer 통계(mean/std/amax/nan) 로깅 삽입 → 실가중치 1 forward:
| 지점 | 통계 |
|---|---|
| embed | std 0.012, amax 0.05, finite |
| L00–L04 (mamba) | finite, std 0.03→0.08 완만 증가 |
| L05 (mamba) | std 0.22, **amax 24.75** (증가, 그러나 finite) |
| **L06 (첫 hybrid)** | **NaN** ← 최초 NaN, 이후 전 레이어 전파 |

hybrid 서브스텝 로깅 → `post_in_ln` finite(amax 57.5) → **`post_attn` = NaN**. 즉 **shared attention**에서 발생.
attention 내부 로깅 → **q/k/v 전부 finite**(q std 0.92, k std 3.36/amax 35, v std 0.95) → **`post_dpa`(RadixAttention 커널 출력) = NaN**. heads=32, **head_dim=160**, scale=0.1118.

## 3. 배제한 가설 (측정으로 확인)
- **mamba A 부호 버그(A_log vs −exp)**: sglang `MambaMixer2`의 `A`는 custom weight_loader `composed(sharded_weight_loader(0), λx:−exp(x))`(mamba.py:369-372). 내 loader는 `param.weight_loader` 호출 → −exp 적용됨. ⇒ A 정확. (L05 amax 24.75는 Zamba2 실제 activation 스케일, 버그 아님 — torch_native서 L06 finite로 확인.)
- **attention 구조/스케일/GQA**: HF `modeling_zamba2.py` 확인 — `scaling=(head_dim/2)**-0.5`(내 값 일치), **qk-norm 없음**, k_proj=`num_key_value_heads*head_dim`이나 로드 clean ⇒ num_kv=32(GQA 없음). q/k/v 계산 정확.
- **가중치 매핑**: 531→527(q/k/v→qkv fusion) 전부 로드, shared block이 `layers.6/12.shared_transformer.*`에 정상 등록.

## 4. 근본 원인
- `[measured]` **flashinfer attention 커널이 head_dim=160에서 실가중치 크기(k std 3.36)에 NaN 생성.** dummy(작은 k)→finite, 실가중치(큰 k)→NaN. 수치안정 softmax라면 finite 입력에 NaN 불가 ⇒ **flashinfer의 head_dim=160 경로가 수치적으로 불안정/미지원**. head_dim 160은 비표준(=32×5, 2의 거듭제곱 아님)이라 flashinfer 미지원 가능성 높음.

## 5. 수정 & 검증 ✅
- `[measured]` **`--attention-backend torch_native`(수치안정 SDPA, 임의 head_dim 지원)로 정확한 출력**:
  `"The capital of France is"` → **`" Paris.\nThe capital of the United"`** (job 827019). post_dpa finite(std 0.68), 전 레이어 finite. **⇒ Zamba2 포트는 기능적으로 정확.**

## 6. 남은 것 (perf 백엔드 — P1.4 전 선정)
- torch_native는 정확하나 느림(pure SDPA). 빠른 head_dim-160 지원 백엔드 필요:
  - `triton`: 별도 버그 — `layer_id=0 not in full attention layers`(hybrid서 layer0가 attn 아님인데 백엔드 init이 layer0 attn 가정). 패치 필요.
  - `fa3`/`fa4`: Hopper 전용(A100 불가 가능성).
  - `flex_attention`: 테스트 중(job 827022) — 임의 head_dim, torch.compile 기반(py3.14서 no-op shim이라 fallback 확인 필요).
  - 대안: q/k/v head_dim 160→192/256 패딩 후 flashinfer; 또는 flashinfer head_dim 지원 확장.
- `[derived]` **layer-aware 연구(P1.3)는 head_dim 128인 NemotronH서 먼저 진행**(flashinfer 정상) → Zamba2 attention perf가 블로킹 아님. Zamba2는 torch_native로 정확성 확보 상태, perf 백엔드는 P1.4서 확정.

## 7. 교훈 (다른 temporal 하이브리드 포트 시)
- 비표준 head_dim(160 등)은 flashinfer NaN 위험 → 포트 시 `--attention-backend torch_native`로 먼저 정확성 검증 후 빠른 백엔드 탐색.
- "전 가중치 로드"≠"정확". per-layer activation 로깅이 forward 버그 국소화에 결정적.
