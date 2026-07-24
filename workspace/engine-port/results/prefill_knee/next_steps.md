# Prefill-side lever 재조사 — (B, L) 2D knee sweep 계획

작성: 2026-07-13. 동기: 사용자 지적 — "prefill Diff B ≈ 1.17× (lever 없음)" 결론은 **단일 운영점(ctx3600·단일 batch)**
에서 나왔다. lever(재배분 여지)를 결정하는 두 축이 **전부 미측정**이다:
- **L (context)**: knee는 ctx3600 하나뿐. attn O(L²) vs mamba O(L) → **L↑에서 Diff B가 열릴** physics.
- **B (batch)**: prefill knee=batched-prefill 한 세팅. B-sweep 전무. 게다가 knee는 micro-measurement인데
  메모리에 **tiny-batch가 서빙 SM-민감도를 과소평가**한 아티팩트 기록(Granite 오판) → "Diff B≈1"이
  **작은 B의 아티팩트**일 수 있음.

관련: [prefill_knee_results.md](prefill_knee_results.md)(ctx3600 단일점 원측정),
[../../reports/prefill_vs_decode_execution.md](../../../../reports/prefill_vs_decode_execution.md)(prefill vs decode 실행),
[../../reports/policy_comparison.md](../../../../reports/policy_comparison.md)(정책 종합), [../../reports/system_vs_engine_vs_sim.md](../../../../reports/system_vs_engine_vs_sim.md)(fidelity-ladder).

---

## 0. 재는 것 — Diff B(B, L)

lever는 **비용비(A)**가 아니라 **한계 SM-생산성 차이(B)**로 성립한다:

- **Diff A (비용비)** = attn-prefill time / mamba-prefill time (고정 SM). L↑에서 자명하게 커짐(O(L²)/O(L)). **lever 아님** — 둘 다 SM 똑같이 먹으면 재배분은 zero-sum.
- **Diff B (SM-민감도 비)** = slope_attn(SM) / slope_mamba(SM), slope = time(8SM)/time(108SM). **이게 재배분 lever의 실체.**
  - `Diff B ≈ 1` → lever 없음 (mamba SM 빼면 attn과 똑같이 손해 = 1:1 트레이드).
  - `Diff B ≫ 1` → cheap-type(mamba)에서 SM 빼 expensive-type(attn)/decode로 돌릴 여지 존재.

ctx3600 단일점 실측: prefill Diff B ≈ **1.17×** (attn 12.3×, mamba 10.5× @108→8). decode는 Diff B 거대(mamba 평탄).

### 가설 (falsifiable)

| 축 | prefill Diff B 예측 | 근거 |
|---|---|---|
| **L↑** | **열림** | mamba O(L)=연산 작아 낮은 SM서 조기포화 → 민감도↓; attn O(L²)=계속 SM-갈망 → 민감도 유지. roofline 분리. |
| **B↑** | **열림 가능** | 작은 B=occupancy-bound(둘 다 둔감→Diff B 눌림, tiny-batch 아티팩트); 큰 B=compute-bound(분리). |

**핵심 반증조건**: (B, L) 격자 전체에서 Diff B가 1 근처면 → prefill-side layer-aware **死 확정**(단일점 아닌 전 격자로).
어느 도달가능 셀에서 Diff B가 decode-급(≫2×)으로 열리면 → 그 regime서 재도전 정당.

---

## 실험 스텝

### Phase 0 — micro↔serving 정합성 (게이트, 선행)
knee는 micro-measurement라 두 번 오도된 전례(GIL-client, tiny-batch). corner 셀(작은/큰 (B,L))에서
knee의 SM-민감도가 **full serving** phase-time과 일치하는지 대조. 특히 **작은 B서 knee가 민감도를
과소평가하는지** 직접 확인(Diff B≈1이 batch 아티팩트인지 판정). 어긋나면 Phase 1을 serving-in-loop 계측으로 전환.

### Phase 1 — prefill knee 2D 스윕 (★critical path)
살아있는 아키텍처(lever=prefill 파티션 sizing, decode=whole+cudagraph 표준경로 — sub-step 아님, cudagraph 유지)의
lever가 여기 걸려 있음. 하네스 `knee2d.sbatch`(SLURM array over L) + 분석 `analyze_knee2d.py`.
- **SM** ∈ {full(108), 44, 24, 16, 8}
- **L** ∈ {2000, 8000, 16000, 32000}  (attn O(L²) 켜지는 구간까지)
- **B** ∈ {1, 4, 16, 48}  (occupancy→compute 전이 포착; 실제 batch는 `bs=`/`ntok=` 로그로 관측)
- 출력: per-attn / per-mamba prefill time → **Diff B(B, L) 히트맵** + mamba 층 시간비율(이득 상한용)

### Phase 2 — decode knee 2D 스윕 (diagnostic, 저우선)
decode-side는 cudagraph 비양립으로 sub-step 死 → policy-actionable 아님. 단 Diff B(B,L) 맵은 (i) Phase 0
아티팩트 교차검증, (ii) "prefill mamba도 decode mamba처럼 포화하나" 대조군으로 가치. `PDMUX_FIXED_DECODE_SM_FILE`
(knee flag off). 우선순위 낮음.

### Phase 3 — reachability overlay
lever 맵 위에 서빙이 실제로 밟는 (B, L) 분포를 겹침. Diff B 열리는 셀이 **도달 가능한 운영점**인지 판정.
장문맥·중대형 batch가 hybrid 모델 타겟 regime이라 겹칠 개연성 높으나 실제 workload로 확인.

### Phase 4 — 조건부 정책 실험 (게이트 통과시만)
Phase 1–3에서 **도달가능 셀 × Diff B 충분히 큼** 확인시만:
- prefill 파티션을 층타입-aware로 sizing + decode는 통짜 cudagraph 표준경로 구현.
- 그 (B,L) regime서 **type-blind 동적 split**(장문맥선 어차피 prefill에 SM 더 줌) 대비 **증분 이득** = 실 payoff quantum.
- 비교군: tuned-uniform / SLO-aware / agnostic 동일 (B,L).

---

## 결정 게이트

```
Phase 0 정합성 ──✗──> knee 폐기 → serving-in-loop 계측으로 전환
        │✓
Phase 1 prefill Diff B(B, L)
        ├─ 전 격자 ≈1 ─────> prefill-side layer-aware 死 확정 (L·B 무관, 종결)
        └─ 어느 셀 ≫1 ─────> Phase 3
                              ├─ unreachable ──> 실무 死 (이론 lever만)
                              └─ reachable ────> Phase 4 정책실험
```

어느 분기든 결론이 남는다: "≈1이면 전 격자로 死 확정", "≫1이나 unreachable이면 실무 死",
"≫1 reachable이면 유일 재도전 트랙 확보". 기존 "ctx3600 한정 死"의 애매함을 제거.

---

## 주의 (메모리 교훈 반영)

- **Diff B만으론 부족**: lever 이득 = Diff B × (옮길 SM량) × (cheap-type 층 시간비율). 히트맵에 mamba 시간비율 병기.
- **비용도 (B,L) 의존**: L↑서 window 길어져 (D) drain 상대비↓ — Phase 4는 net(gain−drain)을 봐야.
- **micro 신뢰 하한**: Phase 0 건너뛰면 세 번째 오도 위험. 반드시 선행.
- **byte-identical off**: `SGLANG_ZAMBA_PREFILL_KNEE` off면 코드경로 무변(bs/ntok 로그는 timing-branch 한정).

## 파일
- 하네스: `results/prefill_knee/knee2d.sbatch` (array over L)
- 분석: `results/prefill_knee/analyze_knee2d.py` (RESULT 파싱 → CSV + Diff B 히트맵)
- src: `src/models/zamba2.py` (+dev 미러) — ZBPT/ZBLT 로그에 `bs=`/`ntok=` 추가(관측된 batch), prefill 로그 게이트 `n%8→n%4`(대배치/짧은L 셀 emit).

---

## ★ 결과 · 판정 (2026-07-13, jobs 847690/847711/847897) — 가설 반증, 트랙 종결(negative)

원자료 `knee2d_result_L{2000,8000,16000,32000}_*.txt` · `knee2d_table.csv` · `knee2d_diffB.png`.

### Diff B (lever) @8SM — 전 격자

| L\B | 1 | 4 | 16 | 48 |
|---|---|---|---|---|
| 2000 | 1.38 | 0.97 | – | 1.34 |
| 8000 | 1.04 | 1.02 | 1.00 | 0.99 |
| 16000 | 1.02 | 1.02 | 1.01 | 0.96 |
| **32000** | **1.01** | 1.04 | 1.01 | – |

### 판정
- **비용비(Diff A)는 열림**: attn/mamba 층당 비용비 0.5×(L2k B1) → 2.4×(L8k) → 9×(L32k, per-attn 108.6 vs per-mamba 11.8ms). O(L²)/O(L) 확증. mamba 시간비율 0.93→0.37(attn이 장문서 지배). **사용자의 Diff A 직관은 옳음.**
- **그러나 lever(Diff B)는 전 격자 ≈1.0**, L↑에서 오히려 1.0으로 **수렴**(L2k noisy 1.38 → L32k 1.01). 결정적 셀 L=32000 B1: attn 108→8SM **13.09×** / mamba **12.91×** = **Diff B 1.01**. 28k 컨텍스트서 attn이 층당 9× 비싼데도 SM-민감도는 동일.
- **물리**: prefill mamba(SSD chunked-scan)=matmul-heavy=**compute-bound** → attn만큼 SM을 계속 먹음. **decode-side 비대칭(mamba≈SM-free, O(1) memory-bound 재귀)이 prefill로 전이 안 됨.** "장문서 mamba 조기포화" 가설 오류.
- **B=48도 ≈1.0** → Diff B≈1이 tiny-batch 아티팩트가 아님(메모리 우려 해소). 소규모 코너(L2k B1/B48=1.38/1.34)는 mamba_frac 0.93=attn 무시가능 영역의 노이즈, 사용 가능한 lever 아님.

**결정 게이트 → "전 격자 ≈1" 분기 → prefill-side layer-aware 死 확정.** 기존 "ctx3600 한정 死"를 **2k–32k × B1–48 전 격자 실측 死**로 격상. **Phase 3/4 미정당, 트랙 종결.** layer-type-aware의 모든 형태(decode sub-step·prefill sub-step/PF·span·§14 예약·prefill-only lever)가 이제 실엔진서 死.
