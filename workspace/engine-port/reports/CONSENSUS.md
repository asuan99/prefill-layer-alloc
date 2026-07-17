# CONSENSUS — engine-port PD-mux 연구의 합의점 (정본)

최종 갱신: 2026-07-17. **이 문서가 현재 확정/철회 상태의 단일 정본이다.** 개별 보고서가 이와 충돌하면 **이 문서가 우선**한다.
과거 보고서는 `deprecated_reports/`로 이관(이력 보존용, 내용은 당시 시점 기준이라 현재 결론과 충돌할 수 있음).

---

## 0. 한 줄

**PD-mux(prefill↔decode SM 분할)는 이득이나, 그 위의 "똑똑한 정책"은 전부 실패했다.**
layer-type 기반 정책은 全형태 死. 동적(SLO-aware/binding-first/feasibility-gate) 제어는 **best-static을 못 넘는다**.
**최적 split은 모델 상수가 아니라 *decode 부하*의 함수**이며, 실전 권고는 **peak decode 부하 기준 decode-heavy static 고정**.

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
| 7 | ★**동적이 best-static을 못 넘음 (HE0)** | **변화 trace**(저분산 벤치, ±0.02): **d44 9.706 > bind+GATE 9.37 > bind 9.348 > d24 9.232 > slo 8.901 > d16 8.438**. 게이트가 switch 18→2로 줄여도 **goodput 불변** |
| 8 | **switch overhead는 병목이 아님** | switch 2회로 static 매칭한 rep 존재; **slo(5sw) < bind(21sw)** ⇒ 손실은 (A)overhead 아니라 **(B)positioning** |
| 9 | **§B의 +18%는 confound** | no-cudagraph(비운영점) + vs d44(최적 아닌 static) — best-static 대비가 아니었음 |

**실전 권고**: **peak decode 부하 기준 decode-heavy static split 고정**(이 워크로드선 d44급). 동적 제어 불요.

---

## 2. ★철회·불확실 (2026-07-17 분산 측정으로 무너진 것)

| # | 이전 주장 | 현재 상태 |
|---|---|---|
| 1 | "stationary bimodal(6.24↔2.22)은 **양성피드백 트랩** 때문" | ★**과잉 귀속 — 철회.** **static d24(switch=0)도 6.32↔3.10으로 붕괴**(±1.302, 1/4 발생). switching이 없으니 **분산의 상당 부분은 시스템 노이즈**. 트랩 *기전*(pf_age 2800→4743 단조증가)은 로그로 실재하나 **크기 귀속 불가** |
| 2 | "d24-static은 ±0.039로 안정" | **n=2의 운.** 실제 **5.282 ± 1.302 (n=4, min 3.102)** |
| 3 | "feasibility 게이트가 트랩을 없애 성능 회복" | **미입증.** bind+GATE **5.928±0.684 (n=6, 붕괴 0/6)** vs static **5.282±1.302 (1/4)** vs no-gate **5.269±1.761 (1/4)** — 방향 유리·분산 절반이나 **분산 겹쳐 유의하지 않음**. 말할 수 있는 건 **"해롭지 않고 아마 약간 유리"**까지 |
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

1. **변화-trace 기반 재검증** — 폐기된 stationary 대신 유효 벤치로 게이트/동적을 n≥4 재측정(현재 bind+GATE n=2로 9.37, d44 9.706 미달).
2. **컨트롤러 CPU 오버헤드** — SLO 경로의 per-span Python 작업(`_slo_prefill_age_ms` 큐 순회)이 single-process 이벤트 루프를 지연시키는지. PIN 격리(`slo_rep30` = 6.252)는 1 rep이라 미결.
3. **시스템 노이즈의 정체** — 동일 config·프롬프트·노드에서 goodput 2× 변동의 원인(GC/GIL/열/공유노드). 이걸 잡아야 stationary 벤치 부활 가능.
4. (낮음) 얽힘-aware 행동모델의 정밀화 — 다만 §1-7(HE0)상 천장은 "static 매칭"이라 payoff 제한.
