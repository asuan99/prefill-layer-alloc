# CONSENSUS — engine-port PD-mux 연구의 합의점 (정본)

최종 갱신: 2026-07-17 (2차: 변화-trace n≥4 캠페인 반영 — HE0 견고 확정 + 게이트=auto-tuner 규명 + 트랩 부분 복권). **이 문서가 현재 확정/철회 상태의 단일 정본이다.** 개별 보고서가 이와 충돌하면 **이 문서가 우선**한다.
과거 보고서는 `deprecated_reports/`로 이관(이력 보존용, 내용은 당시 시점 기준이라 현재 결론과 충돌할 수 있음).

---

## 0. 한 줄

**PD-mux(prefill↔decode SM 분할)는 이득이나, 그 위의 "똑똑한 정책"은 전부 실패했다.**
layer-type 기반 정책은 全형태 死. 동적(SLO-aware/binding-first/feasibility-gate) 제어는 **best-static을 못 넘는다** — 유효 벤치 n≥4로 견고 확정: **d44 9.649±0.039 (n=4) > bind+GATE 9.343±0.083 (n=7)**, 4.7σ.
**최적 split은 모델 상수가 아니라 *decode 부하*의 함수**이며, 실전 권고는 **peak decode 부하 기준 decode-heavy static 고정**.
살아남은 동적의 유일한 값어치는 **성능이 아니라 견고성**(게이트가 트랩 붕괴를 막음: 1/4 → 0/5) — 그마저도 정체는 **틀린 static에 조기 수렴하는 auto-tuner**다.

---

## 1. 확정 결론 (robust — 노이즈·재현성 검증 통과)

| # | 결론 | 근거 |
|---|---|---|
| 1 | **PD 분리 자체는 항상 이득** | agnostic이 fused를 4모델 전부서 이김 |
| 2 | **운영점 = cudagraph-ON** | decode wall 제거(TPOT 41→12ms), goodput ~1.5–2×↑. 기존 no-cudagraph 수치는 전부 하한 |
| 3 | ★**layer-type 런타임 정책 全형태 死** | **(B,L) 2D knee**: 재배분 lever **Diff B ≈ 1.0이 전 격자**(L 2k–32k × B 1–48). decode-side는 lever 있으나 sub-step (D)drain + **cudagraph 비양립**. §14 예약도 fixed d16으로 degenerate |
| 4 | ★**얽힘(entanglement)** | prefill·decode가 running batch(`max_running_requests`)·KV 공유 → **decode 굶김 → ITL↑ → batch 정체 → prefill admission 차단 → TTFT 폭발**. 실측: **d16은 prefill에 92SM(최대)를 주고도 TTFT 7.24s**, d24(84SM)는 1.21s |
| 5 | ★**최적 split = 부하 의존 (이동함)** | `최적 D_sm = max(모델 floor[attn-decode knee], 부하항[∝ λ×output_len])`. 저-decode-부하(synthetic o32/o96)=d16 / 실 trace(ShareGPT r8)=**d24·d44**. **d16 1.056 vs d24 5.28 = 5× 격차로 노이즈(±1.3) 압도** |
| 6 | ★**비대칭** | decode **과다공급**=저부하서 거의 무해 / **과소공급**=고부하서 파국 ⇒ **최악 phase 기준 decode-heavy static이 두 phase 모두 안전 → 지배** |
| 7 | ★**동적이 best-static을 못 넘음 (HE0)** — **n≥4로 견고 확정 (2026-07-17)** | **변화 trace**(유효 벤치). **d44 9.649±0.039 (n=4)** > **d34 9.471±0.090 (n=4)** > **bind+GATE 9.343±0.083 (n=7)** > bind no-gate 8.733±1.023 (n=4). d44↔bind+GATE 격차 **0.307 = 4.7 pooled-σ**. (n=1 참고: d24 9.232 / slo 8.901 / d16 8.438) |
| 8 | **switch overhead는 병목이 아님** | switch 2회로 static 매칭한 rep 존재; **slo(5sw) < bind(21sw)** ⇒ 손실은 (A)overhead 아니라 **(B)positioning** |
| 9 | **§B의 +18%는 confound** | no-cudagraph(비운영점) + vs d44(최적 아닌 static) — best-static 대비가 아니었음 |
| 10 | ★**feasibility 게이트 = 동적 제어가 아니라 "undershooting auto-tuner"** (2026-07-17 규명) | **구조**: 로그상 `2→3`(d24→d34) **1회 decode-ward 이동 후 prefill-ward 복귀를 113회 전부 거부**(`bs=47 ≥ 0.85×48` 상시 참) ⇒ **d34에 영구 고정 = one-way ratchet**. **수치**: bind+GATE **9.322** ≈ **d34-static 9.471** − 0.150(정착 비용). ★**그런데 틀린 static으로 수렴** — 최적은 **d44(9.649)**. 정지 규칙(decode가 더는 급하지 않음: tpot<51ms)이 **최적점 못 미쳐 발동해 ratchet이 조기 정지** |
| 11 | ★**게이트의 가치 = 성능이 아니라 견고성 (트랩 방지)** | **유효 벤치(d44 ±0.039 = 노이즈 없음이 증명된 벤치)에서**: no-gate **8.733±1.023, 1/4 붕괴(6.961; HI phase gp 5.750, sw=10)** vs gate **9.322±0.087, 0/5 붕괴, 분산 12× 타이트**. ⇒ **그 붕괴는 시스템 노이즈가 아니라 컨트롤러 탓**(§2-1 부분 복권). 단 **게이트는 동적을 *안전*하게 만들 뿐 static은 여전히 못 이김** |
| 12 | ★**컨트롤러 CPU 오버헤드 = 死 (직접 계측)** | `SLO-CTLCOST`(v7 이벤트루프 활성 경로 계측): **mean 32–36µs, max 267µs, 누적 ~34ms / ≥1000 call**. 최악의 단일 호출조차 **decode 한 step(ITL p50 ~30ms)의 0.9%**, 누적은 **wall clock의 0.014%**. ⇒ "컨트롤러가 도는 것만으로 이벤트 루프를 지연시킨다"는 가설 **명시적 반증**. 과거 "bind가 switch=0인데 static 미달"은 CPU 비용이 아니라 **§5-4 시스템 노이즈** 탓 |

**실전 권고**: **peak decode 부하 기준 decode-heavy static split 고정**(이 워크로드선 d44급). 동적 제어 불요.
**게이트를 굳이 쓴다면**: 수동 튜닝 없이 안전한 static을 자동으로 찾아주는 **auto-tuner**로서만 값어치(단 최적에 −3.4% 미달).

---

## 2. ★철회·불확실 (2026-07-17 분산 측정으로 무너진 것)

| # | 이전 주장 | 현재 상태 |
|---|---|---|
| 1 | "stationary bimodal(6.24↔2.22)은 **양성피드백 트랩** 때문" | ★**stationary 벤치 한정 과잉 귀속 — 철회 유지**(static d24도 switch=0인데 6.32↔3.10 붕괴 ⇒ 거기선 노이즈와 분리 불가). ★**그러나 트랩 자체는 2026-07-17 부분 복권**: **유효 벤치(변화 trace)** 에서 **d44가 ±0.039 = 노이즈 없음이 증명된 조건**인데도 **no-gate만 1/4 붕괴(6.961), gate는 0/5** ⇒ 거기서의 붕괴는 **컨트롤러 탓이 맞다**(§1-11). **정정된 주장**: "트랩은 실재하고 게이트가 막는다 — 단 stationary 벤치의 bimodal은 그 증거가 못 된다" |
| 2 | "d24-static은 ±0.039로 안정" | **n=2의 운.** 실제 **5.282 ± 1.302 (n=4, min 3.102)** |
| 3 | "feasibility 게이트가 트랩을 없애 **성능 회복**" | ★**2026-07-17 유효 벤치서 분해 — 절반 확정·절반 반증.** **견고성은 확정**(no-gate 8.733±1.023·1/4 붕괴 → gate 9.322±0.087·0/5, 12× 타이트 = §1-11). **성능 회복은 반증**(gate 9.322 < d34 9.471 < **d44 9.649**; §1-10 = ratchet이 틀린 static에 조기 정지). ⇒ "**트랩은 없애나 성능은 여전히 static 미달**" (구 stationary 수치 5.928/5.282/5.269는 노이즈 교란이라 폐기) |
| 4 | 최근 n=1~3 정책 비교 다수 | **underpowered** — 베이스라인 ±1.3이 정책 차이를 삼킴. 재측정 없이 인용 금지 |
| 5 | 초기 SLO track "isolation 오버헤드 0"(2.318≡2.319) | **주의 플래그** — 당시도 n이 작았다면 같은 함정. 재확인 전까지 약한 근거로 취급 |

---

## 3. 방법론 (교훈 — 앞으로 필수)

1. ★**stationary ShareGPT r8 = 정책 비교 벤치로 부적합·폐기.** 동일 프롬프트(`--seed` 고정)·switch 0인 static조차 **±1.302** → 신호를 삼킴.
2. ★**변화 trace(rate 3↔12, 3라운드 평균) = 유효 벤치** (±0.02). **정책 비교는 이걸로.**
3. **베이스라인 분산을 먼저 측정**하고 시작. **n≥4** 없이 정책 결론 금지.
4. **dynamic 결과엔 항상 `switch_count` + split 체류분포 + TTFT/ITL p50/p95/p99 병기.**
5. **Switch decomposition**: `Net = Σ(B positioning) − (A switch × drain)`. (A)와 (B)를 분리 귀속.
6. **변수는 하나씩** (pf_urg와 dwell 동시 변경 → 해석 불가였던 전례).

---

## 4. 살아있는 문서 (이것만 참조)

| 문서 | 역할 |
|---|---|
| **CONSENSUS.md** (이 문서) | **정본** — 확정/철회 상태 |
| `realtrace_findings_and_open_branches.md` | 실 trace 검증 + 얽힘 기전 + 남은 갈래(트리거/행동모델) 상세 |
| `slo_aware_scheduling_design.md` | SLO-aware track 설계·실측 전사(§C–§HE2-3, Step D/E/F/G) |
| `policy_comparison.md` | 정책 taxonomy·메커니즘 (⚠️ §goodput 수치는 no-cudagraph·구벤치 — §1·§2 우선) |
| `system_vs_engine_vs_sim.md` | fidelity ladder (sim/engine/full-system 편향 분리) |
| `prefill_vs_decode_execution.md` | prefill/decode 실행 특성·knee 기초 |
| `session_handoff_2026-07-15.md` | 최근 세션 흐름 |
| `sm_policy_report.html` | layer-aware 원자료 기록(死 트랙, 이력용. §coordinated 개정 미완 = stale) |

`deprecated_reports/` = 초기 triage·포팅·모델별 평가·구 핸드오프·구 리포트. **이력 보존용, 현재 결론과 충돌 가능.**

---

## 5. 열린 항목

1. ~~**변화-trace 기반 재검증**~~ → ✅ **완료 (2026-07-17, jobs 856889–856975)**. n≥4 캠페인으로 **HE0 견고 확정**(§1-7, 4.9σ) + **게이트 정체 규명**(§1-10/11).
2. ~~**게이트 정교화 필요?**~~ → ✅ **성격이 바뀜**: 게이트는 지능적 제어가 아니라 **auto-tuner**(§1-10). 살릴 값어치가 있다면 **ratchet의 조기 정지 수정**(정지 규칙이 d34에서 멈춰 d44를 놓침) — 단 그래봐야 천장은 "best-static 매칭"이라 payoff는 *튜닝 자동화*뿐.
3. ~~**컨트롤러 CPU 오버헤드**~~ → ✅ **직접 계측으로 死 (2026-07-17, jobs 857111/2)** = §1-12.
4. ★**시스템 노이즈의 정체** (stationary 한정, 자원 격리 필요해 보류중) — **새 가설(2026-07-17)**: 나쁜 run(3.102)은 **ITL 정상(p50 30.9 vs 29.0)인데 TTFT만 3.5× 악화(2.92 vs 0.84s)** ⇒ **decode는 멀쩡하고 prefill만 느려짐** = **GPU 클럭/전력 throttling 시그니처**(prefill=compute-bound라 클럭 민감 / decode=memory-bound라 둔감 — 우리 roofline 결과와 정합). 검증엔 `nvidia-smi` 클럭·전력 샘플링 + co-tenant 기록 + 클럭 고정(`-lgc`)이 필요 ⇒ **자원 격리 대기시간 때문에 후순위**. **주의: 변화 trace는 이 노이즈에서 자유롭다(d44 ±0.039)** — 따라서 이 항목은 정책 결론을 막지 않고, *stationary 벤치 부활* 여부만 좌우.
5. (낮음) 얽힘-aware 행동모델의 정밀화 — §1-7(HE0)상 천장이 "static 매칭"이라 payoff 제한.
