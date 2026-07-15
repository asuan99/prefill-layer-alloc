# Session handoff — 2026-07-15 (prefill-phase multiplexing: layer-aware 死 확정 + SLO-aware 동적 트랙 종결(HE0) + 실trace 열림)

이전 핸드오프 [session_handoff_2026-07-13.md](session_handoff_2026-07-13.md)(cudagraph 재측정·layer-aware 종결)의 후속.
이 세션 = **(1) prefill-side layer-aware를 (B,L) knee로 최종 매장 → anchor-predictor로 재정의, (2) L-feedforward(Step D) HD0,
(3) SLO-aware 컨트롤러 재설계(Step E binding-first) + saturation-hold(Step F) 종결, (4) HE2로 "동적이 static을 이기나" 결정 —
HE0(못 이김), (5) granularity·std로 견고화, (6) 실trace(ShareGPT) 열림.**

정본 상세: [slo_aware_scheduling_design.md](slo_aware_scheduling_design.md)(§D~§HE2-3), [policy_comparison.md](policy_comparison.md), [prefill_knee/next_steps.md](../results/prefill_knee/next_steps.md).

---

## 0. 한 줄 상태

**layer-type을 런타임 정책으로 쓰는 모든 형태 死**((B,L) knee: Diff B≈1 전격자). **layer-aware는 offline anchor-predictor로만 생존**(cudagraph 최적 static=d16=attn-decode floor 예측). **동적 PD-split 제어(L-feedforward·SLO-aware·binding-first·saturation-hold)는 cudagraph 운영점서 최적 static(d16)을 못 넘음**(HE2 HE0). **granularity(±2SM 미세)·std로도 견고.** 실전 = **layer-predicted 최적 static split 고정.** ⚠️**전부 synthetic 고정-길이 — 실trace(ShareGPT) 미검증 = 유일 열린 결정 실험.** 커밋 `3fa5278`..`976c151`.

---

## 1. 흐름 (아크별 · 단계별 근거)

각 전환은 대부분 **사용자 지적**이 견인.

### 아크 1 — layer-aware 全형태 死 → anchor-predictor
- decode-side(mamba≈SM-free lever有)=sub-step (D)granularity+cudagraph 몰수로 死.
- prefill-side: **"L·B로 sensitivity 변할 것"(사용자)** → **(B,L) 2D knee**(L 2k–32k×B 1–48, jobs 847690–847897): 비용비(Diff A)는 L 따라 열리나 **재배분 lever(Diff B)는 전격자≈1.0** → prefill-side 死(mamba SSD도 compute-bound). §14 예약도 Probe4서 fixed d16으로 degenerate=死.
- **"cross-phase를 층타입으로"(사용자)** → 핀서((a)prefill-type lever無·(b)decode-type sub-step死). ⇒ **layer-aware=런타임 死, anchor(decode-floor d16) 예측 offline predictor로 재정의.**

### 아크 2 — L-feedforward (Step D) = HD0
- attn O(L²) 근거 `PDMUX_SLO_LFF`(L feedforward). 부하 3회 보정. cudagraph 중간부하 3-rep: **lff tail은 조이나 goodput lff≤slo<d24**(HD0). ★부산물: **컨트롤러가 TTFT-bound에 오설계**(주신호 TPOT) 발견.

### 아크 3 — SLO-aware 컨트롤러 (Step A–F) = static 매칭 천장
- **Step E binding-first**(dual-slack: prefill-slack[split_prefill_batch 최고령 age]·decode-slack 대등): v7b 크게 개선(phaseA 붕괴 수정), static **매칭**.
- **"anchor 필요"(사용자)** → **Step F saturation-hold**: 포화서 4-param(margin/deep/latch) 전부 static 미달. **"포화 예측 가능"(사용자)** → predictive-hybrid도 미달. 근본=포화서 slack이 split-coupled·boundary-hover → reactive 무력.

### 아크 4 — HE2 (동적이 static을 *이기나*) = HE0
- phase-alternating(prefill↔decode) cudagraph: d24 static 승.
- **"최적 불변은 workload 아티팩트?"(사용자)** → **진짜 decode-bound phase(in256/o512)+static 전sweep**: **d16-static 최적, 동적 anchor 무관 패**. decode-bound phase도 **split-flat**(cudagraph decode robust해 binding無)→최적 prefill-heavy edge 불변.
- **"SM 스텝 8-SM 커서 이탈?"(사용자)** → **미세 그리드(±2SM)+std/percentile**: green-ctx 미세 지원(HW한계 아님; sglang cudagraph >7그룹 IndexError 버그). **미세도 격차 못 닫음**(d16 5.81±0.36 > bind-fine 5.35±0.73 ≈ bind-coarse 5.29±0.10). step-size 무관.

---

## 2. 확정 결론
1. **layer-type 런타임 정책 全死** → offline anchor-predictor(d16 예측)로만 생존.
2. **동적 PD-split은 cudagraph 운영점서 최적 static(d16) 못 넘음** — 최적이 prefill-heavy edge에 고정(decode가 binding 안 됨)이라 좇을 이동 없고 이동 자체가 순비용. granularity·std로 견고.
3. **실전 = layer-predicted 최적 static split(d16) 고정.**
4. §B +18%(no-cudagraph·vs d44)는 비운영점·비최적static confound로 철회.

## 3. 데이터·코드 맵
- **코드**: dev `/scratch/ehmoon/whlee/sglang_engine_dev/.../multiplex/multiplexing_mixin.py`(+미러). env: `PDMUX_SLO_MODE=binding`(Step E), `PDMUX_SLO_ANCHOR_IDX`/`_PF_URGENCY`/`_SAT_MARGIN`/`_SAT_DEEP`/`_SAT_LATCH`(Step F), `PDMUX_SLO_LFF`/`_L_REF_TOK`(Step D). off=v7b byte-identical.
- **(B,L) knee**: `results/prefill_knee/`(knee2d.sbatch·analyze_knee2d.py·next_steps.md). src `models/zamba2.py`(bs/ntok 로그).
- **SLO/HE2**: `results/slo_sched/`(he2_bench.sbatch=phase-alternating+percentile; lff_bench.sbatch; pdmux_d{16,24,34,44,54}.yml·pdmux_fine.yml·pdmux_slo.yml). 지표: HE2_RESULT(goodput)·HE2_PCT(TTFT/ITL p50/p95/p99).
- **시각화**: 아티팩트 static_vs_dynamic_flow(시간흐름+granularity+std).
- **환경**: venv py3.14/torch2.9.1+cu130, module `conda/pytorch_2.9.1_cuda13`, 파티션 `amd_a100nv_8`(A100-80GB). cudagraph ON(disable flags 제거).

## 4. 열린 항목 (유일·결정적)
★**실trace 검증(진행중)**: 전 결론이 **synthetic 고정-길이**(range-ratio 1.0) 한정. 실trace(ShareGPT V3 로컬=`hf_cache/raw/ShareGPT_V3_unfiltered_cleaned_split.json`, bench_serving `--dataset-name sharegpt`)로 (a)최적 static이 시간에 따라 *움직이나* (b)움직이면 동적이 static 이기나 검증. 하네스 `results/slo_sched/sharegpt_bench.sbatch`. **이게 "최적 불변·동적 무이득" 결론을 실 workload로 확정 또는 반증하는 마지막 실험.**
- 부차: sglang cudagraph >7 그룹 IndexError 버그(미세 그리드 full-range 막음); SLO 컨트롤러 임계값 cudagraph 재튜닝.

## 5. 메모리 포인터
[[slo-aware-scheduling-track]](§D~HE2-3·granularity 전 흐름), [[prefill-layer-alloc-status]](layer-aware 死·anchor-predictor·(B,L) knee), [[engine-port-p0-triage]], [[fidelity-ladder-scope]].
