# Prefill knee — per-layer-type prefill SM-sensitivity (does a prefill-side layer-aware lever exist?)

작성: 2026-07-08. 동기: 사용자 지적 — "attn·ssm 둘 다 SM-민감해도 **민감도가 다르면** layer-aware가
성립할 수 있지 않나?" (layer-aware는 flatness가 아니라 **differential**만 필요). 옳은 지적 → prefill 판 측정.

## 방법
`SGLANG_ZAMBA_PREFILL_KNEE=1`이 decode-knee의 pin+timing 기계(zamba2.py)를 **extend(prefill) forward**로
재타겟(플래그 off면 byte-identical). prefill 전 층을 N SM에 green-ctx pin하고 per-type 시간 로깅(ZBPT).
Zamba2-2.7B, ctx3600, 60 concurrent prompts/mode, no pdmux(clean). job 837931.

## 결과 (per-layer ms, batched prefill)
| prefill SM | attn-prefill | mamba-prefill | attn 민감 | mamba 민감 | attn/mamba 비용비 |
|---|---|---|---|---|---|
| 108 | 10.97 | 9.03 | 1.0× | 1.0× | 1.21× |
| 44 | 25.18 | 17.51 | 2.30× | 1.94× | 1.44× |
| 24 | 45.78 | 31.98 | 4.17× | 3.54× | 1.43× |
| 16 | 68.52 | 47.62 | 6.25× | 5.27× | 1.44× |
| 8 | 135.4 | 94.8 | 12.3× | 10.5× | 1.43× |

**decode 참고(R0c knee):** attn 3.14→17.5(5.6× @108→16)·mamba 0.36→0.53(둔감); attn/mamba 비용비 **8.7–35×**.

## ★ 후속 (B,L) 2D 일반화 — 2026-07-13, [next_steps.md](next_steps.md) 참조
이 문서는 **ctx3600·단일 batch** 단일점. 사용자 지적("Diff B가 L·B에 따라 열릴 수 있다")에 따라 (B,L) 격자
스윕(L 2k–32k × B 1–48, jobs 847690/711/897): **비용비 Diff A는 L 따라 열림(attn/mamba 0.5×→9× per-layer)**
하나 **lever인 Diff B(민감도비)는 전 격자 ≈1.0**, L↑서 1.0으로 수렴(L32k B1 = **1.01**: attn 13.09× vs mamba
12.91× 민감). mamba SSD-prefill도 compute-bound라 SM을 attn만큼 계속 먹음 → decode-side 비대칭(mamba≈SM-free)
prefill 미전이. **prefill-side layer-aware = 2k–32k×B1–48 전 격자 실측 死**(단일점 死를 격상). 상세·히트맵 next_steps.md.

## 판정 (ctx3600 단일점 기준, 위 2D가 확장·확증)
- **사용자 개념 맞음**: prefill 두 타입 다 SM-민감하되 **기울기 다름**(attn>mamba). flatness 불요, differential이면 됨 — 옳다.
- **그러나 differential이 empirically 너무 작다**: attn/mamba 비용비 1.2–1.4× 내내(near-symmetric), 둘 다 compute-bound라 10×+ 민감. decode(8.7–35× + mamba 평탄)와 정반대. **"공짜로 뺄 둔감 층"이 prefill엔 없다.**
- ⇒ mamba-prefill SM을 빼면 거의 비례해 느려짐 = **1:1 트레이드, free lunch 없음**. per-type 세분 이득=2차(작은 기울기차), (D) granularity 비용=1차 → **net 음수**. prefill-side LA는 tuned-uniform의 step-level split으로 degenerate.
- **대칭 완성**: decode=lever 큼→짧은 창((D))이 죽임; prefill=창 더 길지만→lever 무시할 수준. 이유 반대, 결론 동일. **prefill-coord 실험 불필요 — 종결.**

## 코드
zamba2.py forward: `_pk`/`_phase`로 pin+timing을 prefill(extend)에 재타겟(env `SGLANG_ZAMBA_PREFILL_KNEE`,
`PDMUX_FIXED_PREFILL_SM_FILE`), log tag ZBPT, pk모드 log every 8. src 미러됨. 전체 서술 `reports/prefill_vs_decode_execution.md`.
