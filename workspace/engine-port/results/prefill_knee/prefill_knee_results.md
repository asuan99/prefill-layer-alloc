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
> ⇒ **열린 질문**: L≈200–2000 격자에서 Diff B를 실측할 것.>
> ### ✅ 후속 완료 (2026-07-18) — WIDE 스윕(L 256–32768 × B 1–16)이 열린 질문에 답함
> 하네스 `knee2d_wide.sbatch`(200셀 전부 boot 성공, jobs 857371/857477), 분석 `analyze_knee2d_wide.py`, 표 [`knee2d_wide_table.md`](knee2d_wide_table.md)/`.csv`, 시각화 **[`knee2d_wide.png`](knee2d_wide.png)**.
>
> **답: lever는 L≤512에서 실제로 열린다** (Diff B, B=1): **L=256 → 1.42**, **L=512 → 1.22**, **L=1024 → 0.91**, 이후 L≥1024 전부 **≈1.0**(0.91–1.10). ⇒ 옛 L=2000 저변의 1.35는 이 열림의 꼬리였다. **실 워크로드(mean 352·p50 204 tok)가 정확히 lever 구간 안**이다.
>
> ★**기전 (결정적)**: 짧은 L에서 lever가 열리는 이유는 "타입 간 compute scaling 차이"가 아니라 **둘 다 SM을 다 못 쓰기 때문**이다 — L=256서 SM 8→108 speedup이 **attn 5.9× / mamba 4.2× (이상적 13×에 한참 못 미침)**. 특히 mamba는 **44 SM에서 이미 포화**(L256 B1: mamba 44SM=0.85ms인데 108SM=0.93ms로 *역행*)하고 attn만 계속 개선(0.13→0.09ms). ⇒ **짧은 L에선 prefill이 SM을 거의 필요로 하지 않는다**(얽힘 결론과 정합: SM은 decode로).
>
> ★**정책 함의는 여전히 死**: (a) lever가 열리는 절대 stakes가 **sub-ms/layer**(L256: attn 0.09ms, mamba 0.9ms)라, S2에서 측정된 **(D) granularity 비용 TPOT 42→124ms**에 압도된다. (b) **batch는 lever를 안 연다** — 관측 bs 버킷 전부 Diff B 0.98–1.09(히트맵의 B=8@16k=1.32는 bs=1로 붕괴한 아티팩트, 속빈 마커). ⇒ **"lever 부재" 서사는 L≥1024에 정확**하고, L≤512에선 **lever는 있으나 (D)가 삼킨다**로 정정. **결론(layer-aware 死) 불변, 기전만 정밀화.** 단 **Diff B>1이어도 정책이 되려면 (D) 비용을 넘어야** 하는데, S2에서 그 비용은 **TPOT 42→124ms**로 측정됐다 — 짧은 L의 절대 stakes(per-layer 1–8ms)가 그보다 작으므로 **부활 가능성은 낮다.**

## 판정 (ctx3600 단일점 기준, 위 2D가 확장·확증)
- **사용자 개념 맞음**: prefill 두 타입 다 SM-민감하되 **기울기 다름**(attn>mamba). flatness 불요, differential이면 됨 — 옳다.
- **그러나 differential이 empirically 너무 작다**: attn/mamba 비용비 1.2–1.4× 내내(near-symmetric), 둘 다 compute-bound라 10×+ 민감. decode(8.7–35× + mamba 평탄)와 정반대. **"공짜로 뺄 둔감 층"이 prefill엔 없다.**
- ⇒ mamba-prefill SM을 빼면 거의 비례해 느려짐 = **1:1 트레이드, free lunch 없음**. per-type 세분 이득=2차(작은 기울기차), (D) granularity 비용=1차 → **net 음수**. prefill-side LA는 tuned-uniform의 step-level split으로 degenerate.
- **대칭 완성**: decode=lever 큼→짧은 창((D))이 죽임; prefill=창 더 길지만→lever 무시할 수준. 이유 반대, 결론 동일. **prefill-coord 실험 불필요 — 종결.**

## 코드
zamba2.py forward: `_pk`/`_phase`로 pin+timing을 prefill(extend)에 재타겟(env `SGLANG_ZAMBA_PREFILL_KNEE`,
`PDMUX_FIXED_PREFILL_SM_FILE`), log tag ZBPT, pk모드 log every 8. src 미러됨. 전체 서술 `reports/prefill_vs_decode_execution.md`.
