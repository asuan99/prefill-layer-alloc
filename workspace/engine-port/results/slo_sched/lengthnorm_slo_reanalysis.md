# 길이-정규화 SLO 재계측 — HE0가 SLO 정의에 의존한다

측정: 2026-07-18. 방법: **기존 varying-trace 벤치 데이터 재분석**(job 재제출 없음). 스크립트 [`reanalyze_lengthnorm_slo.py`](reanalyze_lengthnorm_slo.py).
동기: 사용자 지적 — 고정 `TTFT≤3s` SLO는 workload 길이와 무관한 자유 파라미터인데, 정책 우열이 여기에 걸려 있는 것 아닌가.

각 `sgptv{Lo,Hi}_*.jsonl`에 per-request `input_lens`/`ttfts`/`itls`가 저장돼 있어, goodput 지시함수의 TTFT 항만
`TTFT≤3s(고정)` → `TTFT≤SLO(L)=k·(a₀+b₀·L)`(길이-정규화)로 바꿔 재계산했다. TPOT 항(≤60ms)은 길이 무관이라 불변.
물리 line은 d44 LO(rate3, 큐잉 최소) phase 회귀: **TTFT ≈ 79ms + 0.096ms/tok·L** (217-tok→100ms, p99 2776-tok→345ms).

## 결과: 순위가 SLO 정의에 따라 뒤집힌다

| policy | n | **fixed-3s** | norm-k2 | norm-k3 | norm-k4 |
|---|---|---|---|---|---|
| **d44** (가장 decode-heavy) | 4 | **3.220±0.013 ①** | 2.334 | 2.508 ③ | 2.614 |
| d34 | 4 | 3.171±0.025 ② | 2.383 ② | 2.540 ② | 2.649 ② |
| d24 | 1 | 3.081 | 2.307 | 2.435 | 2.557 |
| d16 (prefill-heavy) | 1 | 2.817 ⑦ | 2.285 | 2.435 | 2.490 |
| slo | 1 | 2.974 | 2.337 | 2.488 | 2.596 |
| **bind+GATE** (dynamic) | 9 | 3.132±0.019 ③ | **2.441±0.015 ①** | **2.582±0.009 ①** | **2.671±0.016 ①** |
| bind+nogate | 4 | 2.934±0.306 | 2.307 | 2.440 | 2.518 |

- **fixed-3s**: `d44 > d34 > bind+GATE` — CONSENSUS §1-7과 정확히 일치(sanity 통과). **static 지배 = HE0.**
- **norm-k(2/3/4) 전부**: `bind+GATE ①` — **동적이 best-static을 이기거나 대등.** fixed에서 1등이던 **d44는 3~4등으로 강등**.
- 유의성(norm-k3): bind+GATE vs **d44 +0.073 (~3.0σ, 유의)**; vs **d34 +0.042 (~1.5σ, 겹침)**. `k3+0.5floor` 변형선 d34 2.730 ≈ bind 2.728(무승부).

⇒ 정직한 판정: **"동적이 static을 *압도*"가 아니라 "동적이 best-static과 *대등~약우위*"** — 그러나 **HE0(동적이 static에 확실히 진다)는 반증됐고, decode-heavy static(d44) 지배는 SLO 정의의 산물이었다.**

## 기전: phase 분해가 이유를 보여준다

| policy | fixed LO / HI | norm-k3 LO / HI |
|---|---|---|
| d44 | 2.858 / **3.924** | 2.817 / **1.908** |
| d34 | 2.852 / 3.785 | 2.802 / 2.037 |
| **bind+GATE** | 2.856 / 3.657 | 2.850 / **2.070** |
| d16 | 2.861 / 2.737 | 2.794 / 1.794 |

- **LO(rate3)**: 두 SLO 모두 전 정책 ~2.8로 **무관심**(split이 무의미) — CONSENSUS §1-13의 "LO엔 쫓을 최적점 없음"은 SLO 정의와 무관하게 유지.
- ★**HI(rate12)에서 반전이 일어난다**:
  - **fixed**: `d44 3.924 > … > bind 3.657` — TTFT 3s가 관대해 goodput ≈ *완료율* ⇒ **decode throughput 지배 ⇒ decode-heavy static 승**.
  - **norm-k3**: `bind 2.070 > d34 2.037 > d44 1.908` — 짧은 요청도 300ms 안에 first-token 요구 ⇒ **prefill 반응성 지배 ⇒ 부하 중 prefill/decode를 저글링하는 동적이 승**. decode-heavy static(d44)은 prefill을 굶겨 짧은-요청 TTFT를 못 맞춰 **꼴찌권**.

⇒ ★**CONSENSUS §1-13의 "HI 최적 = 고정된 decode-heavy 지점"은 관대한-SLO의 산물.** SLO를 빡빡하게 하면 **HI 최적이 "고정점"이 아니라 "부하 내 변동에 반응하는 동적"** 이 되고, static은 그걸 못 커버한다. "두 regime의 최적이 충돌하지 않는다"가 아니라 **"HI 최적 자체가 static으로 표현 불가능"** 해진다.

## ★반전의 진짜 축 = latency 엄격도(tightness), *길이-비례성 아님* (robustness)

사용자 직관은 "길이-비례 SLO"였으나, 원인을 분리해보니 다르다.

**(A) fixed-tight sweep (길이 *무관*, 순수 엄격도만):**

| SLO | 승자 |
|---|---|
| 3.0 / 2.0 / 1.5 / 1.0 / 0.75s | **d44 (decode-heavy static)** |
| 0.5s | d34 (bind 근소 2등) |
| **0.335s** | **bind+GATE (dynamic)** |

승자 교체 임계 ≈ **TTFT 0.5–0.75s**. 실 prefill 비용이 mean~112ms이니 그 **3–5배**가 경계.

**(B) matched-average — 같은 ~335ms 평균 예산, flat vs 길이-비례:**

| SLO | d44 | d34 | bind+GATE | 승자 |
|---|---|---|---|---|
| fixed-0.335s (flat) | 2.449 | 2.549 | **2.593** | bind+GATE |
| norm-k3 (length-prop) | 2.508 | 2.540 | **2.582** | bind+GATE |

⇒ ★**같은 평균 tightness면 길이-비례 여부는 순위에 거의 무영향**(bind 2.593≈2.582). **반전을 만든 것은 길이-비례성이 아니라 SLO가 3s→0.3s로 빡빡해진 것.**
**(C) fit-source**: d44/d34/bind 중 누구 LO line으로 fit해도 norm-k3 승자 = bind+GATE(robust).

**정정된 결론**: HE0(static 지배)는 **관대한 latency SLO(TTFT≳0.75s)의 산물**이다. 실 workload 지연 스케일(mean prefill ~112ms)에 3s TTFT는 **~27× 과대 예산** — goodput을 사실상 *완료율*로 만들어 decode throughput(=decode-heavy static)을 이기게 했다. SLO를 workload 스케일에 맞춰 빡빡하게(≲0.5s) 잡으면 **first-token 반응성이 지배 → 동적이 best-static과 대등~약우위**. 사용자 지적("고정 SLO가 계측을 왜곡")은 옳았고, 정확한 축은 **길이-무관 vs 비례가 아니라 latency 엄격도**였다.

## 고정 SLO가 실제로 차별한 대상 (예상과 반대)

fixed-3s pass율(d44, input-length 버킷): `<500 86% · 500-1k 81% · 1k-2k 93% · **≥2k 100%**`.
⇒ ★**고정 SLO는 긴 요청이 아니라 *짧은 요청*을 떨군다** — 긴 요청은 수가 적고(96개) 3s 예산이 넉넉해 100% 통과, 짧은 요청은 HI phase 큐잉으로 3s 근처까지 밀려 탈락. (내 사전 직관 "고정 SLO가 긴 요청 차별"은 틀렸다.)
길이-정규화의 효과는 "긴 요청 구제"가 아니라 **짧은 요청에 길이-비례 예산(217tok→300ms)을 매겨 first-token 지연을 실제로 평가**하는 것.

## 한계 (정직하게)

1. **여전히 Zamba2-2.7B · short-context(ShareGPT p99 2776tok).** long-context는 별개(→ [longcontext_trace_plan.md](../../../../reports/longcontext_trace_plan.md)).
2. `norm` (a,b)는 d44 LO fit — 단 이는 **d44에 유리한 방향**이므로 반전을 *만든* 편향이 아님(오히려 보수적).
3. d24/d16/slo는 **n=1**(참고값). robust 비교는 d44/d34/bind(n≥4)뿐.
4. bind+GATE vs **d34**는 1.5σ로 미결(대등). 확실한 건 **d44 강등(3σ)** 과 **HE0 반증**.
5. **"옳은 SLO"는 응용이 정한다** — 고정 3s도 norm-k3도 임의. 진짜 발견은 특정 SLO의 승자가 아니라 **정책 우열이 SLO 정의에 민감하다**는 것.
