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
하나 **lever인 Diff B(민감도비)는 L≥8000 전 격자 ≈1.0**(0.96–1.04), L↑서 1.0으로 수렴(L32k B1 = **1.01**: attn 13.09× vs mamba
12.91× 민감). mamba SSD-prefill도 compute-bound라 SM을 attn만큼 계속 먹음 → decode-side 비대칭(mamba≈SM-free)
prefill 미전이. **prefill-side layer-aware = 2k–32k×B1–48 전 격자 실측 死**(단일점 死를 격상). 상세·히트맵 next_steps.md.

> ### ⚠️ 정정 (2026-07-17) — 위 "Diff B ≈ 전 격자 1.0"은 **부정확**했다
> 원자료(`knee2d_table.csv`) 재검토 결과 두 가지를 바로잡는다. 시각화: **[`diffA_vs_diffB.png`](diffA_vs_diffB.png)**, 표: [`diffA_vs_diffB_table.md`](diffA_vs_diffB_table.md), 스크립트: `plot_diffA_vs_diffB.py`.
>
> **(1) L=2000에서 Diff B는 1.0이 아니라 ≈1.35다.** B=1 **1.38**(attn 11.5× vs mamba 8.3×), B=48 **1.34**(12.7× vs 9.5×) — 3셀 중 2셀이 일치하고 B=4만 0.97로 이견.
> 절대시간이 작지 않으므로(B=48 SM108: attn 7.74ms / mamba 16.47ms) **micro 아티팩트로 치부하기 어렵다**. 기전 추정: **짧은 L에서 mamba가 floor에 근접해 SM을 더 못 먹는다**(B1→B4에서 mamba가 4× 일감에 1.37×만 증가 = floor 징후). ⇒ **lever는 L이 짧아질수록 *열린다*.**
>
> **(2) 격자가 실제 서빙 regime을 아예 안 덮는다.** 스윕은 L=2000에서 시작하는데 **ShareGPT 실 trace는 mean 352 · p50 204 · p95 1042 tok — 요청의 98%가 L<2000**이다. 즉 **Diff B는 실제로 서빙되는 구간에서 측정된 적이 없다.**
>
> **결론에 미치는 영향**: layer-aware의 **서빙 수준 반증은 유효**하다 — 그건 이 격자가 아니라 **실 워크로드에서의 직접 측정**(4-모델 스윕에서 agnostic 4/4 승; coordinated per-type 구현이 TPOT 42→124ms)에 근거하기 때문이다.
> 무너지는 것은 ***기전 설명***이다: **"lever(Diff B)가 없어서 죽었다"는 서사는 long-context(L≥8k)에만 성립**하고, 실제 서빙 구간(L<2k)에는 **외삽**이다. 그 구간에서 layer-aware가 죽은 진짜 이유는 lever 부재가 아니라 **(D) granularity**(sub-step green-ctx 재분할 비용 + cudagraph 비양립)일 가능성이 높다.
> ⇒ **열린 질문**: L≈200–2000 격자에서 Diff B를 실측할 것. 단 **Diff B>1이어도 정책이 되려면 (D) 비용을 넘어야** 하는데, S2에서 그 비용은 **TPOT 42→124ms**로 측정됐다 — 짧은 L의 절대 stakes(per-layer 1–8ms)가 그보다 작으므로 **부활 가능성은 낮다.**

## 판정 (ctx3600 단일점 기준, 위 2D가 확장·확증)
- **사용자 개념 맞음**: prefill 두 타입 다 SM-민감하되 **기울기 다름**(attn>mamba). flatness 불요, differential이면 됨 — 옳다.
- **그러나 differential이 empirically 너무 작다**: attn/mamba 비용비 1.2–1.4× 내내(near-symmetric), 둘 다 compute-bound라 10×+ 민감. decode(8.7–35× + mamba 평탄)와 정반대. **"공짜로 뺄 둔감 층"이 prefill엔 없다.**
- ⇒ mamba-prefill SM을 빼면 거의 비례해 느려짐 = **1:1 트레이드, free lunch 없음**. per-type 세분 이득=2차(작은 기울기차), (D) granularity 비용=1차 → **net 음수**. prefill-side LA는 tuned-uniform의 step-level split으로 degenerate.
- **대칭 완성**: decode=lever 큼→짧은 창((D))이 죽임; prefill=창 더 길지만→lever 무시할 수준. 이유 반대, 결론 동일. **prefill-coord 실험 불필요 — 종결.**

## 코드
zamba2.py forward: `_pk`/`_phase`로 pin+timing을 prefill(extend)에 재타겟(env `SGLANG_ZAMBA_PREFILL_KNEE`,
`PDMUX_FIXED_PREFILL_SM_FILE`), log tag ZBPT, pk모드 log every 8. src 미러됨. 전체 서술 `reports/prefill_vs_decode_execution.md`.
