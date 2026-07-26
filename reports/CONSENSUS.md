# CONSENSUS — engine-port PD-mux 연구의 합의점 (2026-07-19 historical)

> **현재 전체 정본:** [`../PROJECT_STATUS.md`](../PROJECT_STATUS.md).
> 이 문서는 dual-worker/R2 이전까지 확정된 phase separation, layer-granular
> negative result, entanglement, single-worker dynamic 결과의 정본으로 유지한다.

최종 갱신: 2026-07-26 (★★**Stage 0(long-ctx L−2 게이트)**: 운영점 decode
SM-무감각을 hybrid·pure-Transformer·pure-Mamba·ctx≤16k 전부로 확장 확인 —
non-binding, long-ctx 충돌 가설 이 regime서 붕괴, HE0/벡터1 ctx-무관으로 강화.
상세 §1-21). 이전: 2026-07-19 (★★**§1-16 반증 — tight SLO로 컨트롤러를 실제 재튜닝하면 동적은 best-static에 크게 열위(§1-17). SLO 엄격도와 무관하게 decode-heavy static 지배 확정.** 2026-07-18 §1-16의 "tight→동적 우위"는 재스코어 아티팩트로 격하). 2026-07-17 (변화-trace n≥4 — HE0 견고 확정 + 게이트=auto-tuner 규명).
과거 보고서는 `deprecated_reports/`로 이관(이력 보존용, 내용은 당시 시점 기준이라 현재 결론과 충돌할 수 있음). ★**2026-07-24**: 저장소 전체 격리처를 단일 `deprecated/`로 통합하면서 이 디렉터리는 [`../deprecated/reports/quarantine_engine_port/`](../deprecated/reports/quarantine_engine_port)로 물리 이동했다(내용·판정 불변, 경로만 변경). ★**2026-07-24 스코프 정정(claims-auditor 감사, doc-steward)**: §5-7의 "⇒ 동적 제어 트랙 완전 종결" 표현이 [`../PROJECT_STATUS.md`](../PROJECT_STATUS.md)(Claim D/E = 미검증) 및 §1-20(decoupled substrate에 +16% headroom 실재)과 모순돼 **overclaim으로 범위 축소**. §1-7·§1-17(single-worker·SM-split·reactive 제어, n≥4)의 확정성은 불변. 상세는 §5-7·§5-8. ★★**2026-07-24 구조적 정정 추가(engine-porter 코드 리뷰, 읽기전용, [`r2_decoupling_review_2026-07-24.md`](r2_decoupling_review_2026-07-24.md))**: §5-8(a)가 §1-20의 decoupled substrate 후보로 언급했던 `PDMUX_TRUE_DUAL_WORKER=1`은 file:line 근거로 **control-plane dual-worker(호스트 스레드/큐/role만 분리)일 뿐, running batch/KV/SM은 전면 공유**로 확인됨 — decoupled substrate에 **해당하지 않는다**. 상세는 §5-8(a). ★**2026-07-25 §5-8(c) 업데이트**: HE0-reopen 벡터1(G2.0 disjoint 스윕, n≥4 재시도)이 (c)의 underpowered 상태를 부분적으로 검증 — **ILL-POSED at rA5 판정**(escape hatch 근거로 "지지 안 됨"이나 "종결"도 아님, §1-20과는 무관한 별도 축). 상세는 §5-8(c). ★**2026-07-25 §5-8(c) 추가 업데이트(de-cliff stage-1, 실험 2026-07-24~25·기록 2026-07-25)**: rA5 절벽을 벗어난 rA2에서 disjoint를 찾지 못함(`d54`가 양 phase 동시 커버) — **PLAUSIBLE closure이나 CONFIRMED 아님**(claims-auditor 반증 3항목 + narrow-rA 확증 sweep 선행 필수). 한 눈금 전진이지 종결 아님. 상세는 §5-8(c). ★★**2026-07-25 §5-8(c) 최종 업데이트(narrow-rA 확증, `g2_0_rasweep`+`g2_0_raconf`)**: g2_0_rasweep(120 job, off-cliff sub-band rate≤2.75서 disjoint 재확인 없음)이 전이대를 rate 3.0–3.5로 좁혔고, claims-auditor pre-registered 24-job 확증 열 `g2_0_raconf`(rate{3.5,3.75}×{d44,d54}×n6)가 결정 규칙을 충족 — **companion collapse**(rate3.5: d44 0.953±0.035≈d54 0.948±0.035; rate3.75: d44 0.932±0.042<**d54 0.948±0.062**, REOPEN 전제 양쪽 붕괴). **벡터1(short-ctx disjoint) = CONFIRMED closure(scoped)로 종결** — PROJECT_STATUS "벡터1" 절·`g2_0_raconf/raconf_final_verdict_2026-07-25.md` 참조. §5-8(c)의 미결 갈래 (c)는 이제 닫혔다(scope=short-ctx drained; long-ctx·§1-20 spatial decoupling은 별도 미결).

---

## 0. 한 줄

**PD-mux(prefill↔decode SM 분할)는 이득이나, 그 위의 "똑똑한 정책"은 전부 실패했다.**
layer-type 기반 정책은 全형태 死. 동적(SLO-aware/binding-first/feasibility-gate) 제어는 **best-static을 못 넘는다 — 관대·tight SLO 양쪽에서 확정**. 관대 SLO(TTFT 3s): d44 3.220±0.013 > bind+GATE 3.132±0.019, 5.4σ. ★**tight SLO(chat 300/50ms)로 컨트롤러를 실제 재튜닝해도 열위, 오히려 격차 확대**: rate8 attainment d44 **73.2%** ≫ d34 49.6 > bind+GATE 44.3 > bind 40.6 (28.9%p≈10σ, §1-17). 컨트롤러가 tight TTFT에 반응해 prefill-ward 이동→decode 굶김→§1-4 얽힘 트랩→TTFT 악화.
★**한때(2026-07-18, §1-16) tight SLO 재스코어에서 "동적 우위"로 보였으나, 컨트롤러를 실제 재튜닝한 직접 측정(§1-17)이 이를 반증** — 재스코어는 3s-튜닝 컨트롤러의 정착 static 위치를 사후 채점한 아티팩트였다. ⇒ **"decode-heavy static 지배"는 SLO 엄격도와 무관한 결론.**
**최적 split은 모델 상수가 아니라 *decode 부하*의 함수**이며, 실전 권고는 **peak decode 부하 기준 decode-heavy static 고정**.
살아남은 동적의 유일한 값어치는 **성능이 아니라 견고성**(게이트가 트랩 붕괴를 막음: 1/4 → 0/5; 정체는 **틀린 static에 조기 수렴하는 auto-tuner**).

---

## 1. 확정 결론 (robust — 노이즈·재현성 검증 통과)

| # | 결론 | 근거 |
|---|---|---|
| 1 | **PD 분리 자체는 항상 이득** | agnostic이 fused를 4모델 전부서 이김 |
| 2 | **운영점 = cudagraph-ON** | decode wall 제거(TPOT 41→12ms), goodput ~1.5–2×↑. 기존 no-cudagraph 수치는 전부 하한 |
| 3 | ★**layer-type 런타임 정책 全형태 死** | 근거는 **서빙 직접 측정**: 4-모델서 agnostic 4/4 승 + **coordinated per-type 구현이 TPOT 42→124ms**. **(B,L) 2D knee**: 재배분 lever **Diff B ≈ 1.0 (L≥8000, 0.96–1.04)**. ⚠️**정정(2026-07-17)**: **L=2000선 Diff B≈1.35**(B1 1.38/B48 1.34)이고 **격자가 실 서빙 regime(ShareGPT 98%가 L<2k)을 안 덮음** ⇒ **"lever 부재" 기전은 long-context 한정·짧은 L엔 외삽**. 결론은 서빙 측정이 지탱하며, 짧은 L의 死因은 **(D) granularity**로 추정. 시각화 `results/prefill_knee/diffA_vs_diffB.png`. **✅ WIDE 스윕(L 256–32768×B 1–16, 2026-07-18, jobs 857371/857477)으로 확증**: lever는 **L≤512서 실제로 열림**(Diff B 256→1.42/512→1.22), L≥1024 ≈1.0; 기전=짧은 L서 둘 다 SM 미활용(mamba 44SM 포화); **그래도 死**(stakes sub-ms/layer ≪ (D) 42→124ms, batch 무영향). `knee2d_wide.png`. decode-side는 lever 있으나 sub-step (D)drain + **cudagraph 비양립**. §14 예약도 fixed d16으로 degenerate |
| 4 | ★**얽힘(entanglement)** | prefill·decode가 running batch(`max_running_requests`)·KV 공유 → **decode 굶김 → ITL↑ → batch 정체 → prefill admission 차단 → TTFT 폭발**. 실측: **d16은 prefill에 92SM(최대)를 주고도 TTFT 7.24s**, d24(84SM)는 1.21s |
| 5 | ★**최적 split = 부하 의존 (이동함)** | `최적 D_sm = max(모델 floor[attn-decode knee], 부하항[∝ λ×output_len])`. ★**floor 자체가 ctx 의존 (2026-07-18, job 858811)**: decode step의 attn 비율이 ctx 따라 이동(ctx256=5%→ctx16k=79%)해 **whole-decode SM-민감도가 1.1×(ctx256, SM-free)→10.5×(ctx16k, SM-hungry)**, 최적 decode SM knee **16→44→108→108**. ⇒ 짧은 ctx=decode에 SM 조금·긴 ctx=많이. (per-type 분할 아님=offline predictor 입력; `results/r0c/decode_knee_vs_ctx.png`). ⚠️**단 '민감도'는 triton/no-cudagraph 마이크로벤치 값**: decode-attn은 원리상 memory-bound지만 이 커널은 **HBM 대역폭 미포화(MLP-limited)라 108 SM까지 ~선형 스케일**(효율 44→108서도 ≈1.0). **운영점(cudagraph)선 HE2가 decode non-binding으로 관측** ⇒ 운영점 magnitude는 열린 질문. `decode_attn_saturation.png`. 저-decode-부하(synthetic o32/o96)=d16 / 실 trace(ShareGPT r8)=**d24·d44**. **d16 1.056 vs d24 5.28 = 5× 격차로 노이즈(±1.3) 압도** |
| 6 | ★**비대칭** | decode **과다공급**=저부하서 거의 무해 / **과소공급**=고부하서 파국 ⇒ **최악 phase 기준 decode-heavy static이 두 phase 모두 안전 → 지배** |
| 7 | ★**동적이 best-static을 못 넘음 (HE0)** — **n≥4 견고, SLO 엄격도 무관 (§1-17로 tight까지 확정)** | **변화 trace**(유효 벤치). **d44 3.220±0.013 (n=4)** > **d34 3.171±0.025 (n=4)** > **bind+GATE 3.132±0.019 (n=9)** > bind no-gate 2.934±0.306 (n=4). d44↔bind+GATE 격차 **0.088 = 5.4 pooled-σ**. (n≥4: d24 3.039±0.130 / slo 2.964±0.025 / d16 2.846±0.055). ※ 전부 **TRUE goodput** — 구 보고값(9.649 등)은 하네스 3× 부풀림, `f921ae8`서 수정, **순위 불변**. ★**tight SLO(chat 300/50)로 재튜닝해도 동일**(§1-17: d44 73.2%≫bind+GATE 44.3%) — 관대 SLO 한정 아님 |
| 8 | **switch overhead는 병목이 아님** | switch 2회로 static 매칭한 rep 존재; **slo(5sw) < bind(21sw)** ⇒ 손실은 (A)overhead 아니라 **(B)positioning** |
| 9 | **§B의 +18%는 confound** | no-cudagraph(비운영점) + vs d44(최적 아닌 static) — best-static 대비가 아니었음 |
| 10 | ★**feasibility 게이트 = 동적 제어가 아니라 "undershooting auto-tuner"** (2026-07-17 규명) | **구조**: 로그상 `2→3`(d24→d34) **1회 decode-ward 이동 후 prefill-ward 복귀를 113회 전부 거부**(`bs=47 ≥ 0.85×48` 상시 참) ⇒ **d34에 영구 고정 = one-way ratchet**. **수치**: bind+GATE **3.132 (n=9)** ≈ **d34-static 3.171** − 0.039(정착 비용). ★**그런데 틀린 static으로 수렴** — 최적은 **d44(3.220)**. 정지 규칙(decode가 더는 급하지 않음: tpot<51ms)이 **최적점 못 미쳐 발동해 ratchet이 조기 정지** |
| 11 | ★**게이트의 가치 = 성능이 아니라 견고성 (트랩 방지)** | **유효 벤치(d44 ±0.013 = 노이즈 없음이 증명된 벤치)에서**: no-gate **2.934±0.306, 1/4 붕괴(2.405, sw=10)** vs gate **3.132±0.019 (n=9), 0/9 붕괴, 분산 16× 타이트**. ⇒ **그 붕괴는 시스템 노이즈가 아니라 컨트롤러 탓**(§2-1 부분 복권). 단 **게이트는 동적을 *안전*하게 만들 뿐 static은 여전히 못 이김** |
| 12 | ★**컨트롤러 CPU 오버헤드 = 死 (직접 계측)** | `SLO-CTLCOST`(v7 이벤트루프 활성 경로 계측): **mean 32–36µs, max 267µs, 누적 ~34ms / ≥1000 call**. 최악의 단일 호출조차 **decode 한 step(ITL p50 ~30ms)의 0.9%**, 누적은 **wall clock의 0.014%**. ⇒ "컨트롤러가 도는 것만으로 이벤트 루프를 지연시킨다"는 가설 **명시적 반증**. 과거 "bind가 switch=0인데 static 미달"은 CPU 비용이 아니라 **§5-4 시스템 노이즈** 탓 |

| 21 | ★★**Stage 0(long-ctx L−2 게이트, 2026-07-26) — 운영점 decode SM-무감각, hybrid·16k로 확장** | 3-arm coupled-운영점 스윕(M=pure Mamba2-2.7B 음성대조·H=Zamba2-2.7B hybrid·T=Qwen2.5-3B 양성대조, ctx{4k,8k,16k}, decode-SM{16,44,92}+108-ref, jobs 864230+864601, PIN_CHECK 전부 PASS). **판정1(CONFOUNDED, CONFIRMED)**: raw ITL(D16/D44/D92) 곡선은 decode-SM binding이 아니라 prefill 경합/entanglement 아티팩트 — D108(무경합 앵커)이 D16과 9셀 전부 ≤0.5% 동일(비단조 최속점 D92=prefill 16SM 굶김 지점), **음성 대조 M**(decode O(1) recurrent라 SM-bound 불가)이 H와 동형의 "민감도"(2.4×대)를 보이는 것 자체가 곡선=confound의 증거. **판정2(NULL, CONFIRMED)**: 유일 de-confounded 대조 **D16 vs D108 = 1.00±0.01, 3 arm×3 ctx 전부** ⇒ **운영점 decode는 16→108 SM에 무감각, pure-Transformer·pure-Mamba·hybrid 전부, 16k ctx까지**(r0c mamba SM-불변을 hybrid·서빙 in-situ·16k로 확장). **판정3**: `longcontext_trace_plan.md` §6 사전등록 게이트의 "게이트 실패이자 강한 결과" 분기 실현 — H_L4(시간축)·H_L5(공간축) 이 regime서 붕괴, **HE0/벡터1이 ctx-무관으로 강화**(반전 아니라 강화). ★scope 한정(필수): {M/H/T 2.7–3B, triton, cudagraph-ON green-context pdmux, ctx≤16k, coupled 하네스, one-shot 32-conc burst} — 더 큰 모델·>16k·완전 de-confound 재측정(prefill-SM 고정+steady-state)은 **미실행**(사용자가 현 증거로 결론 확정 결정), 결론은 magnitude 아니라 **방향(non-binding)만**. 상세 [`stage0_verdict_2026-07-26.md`](stage0_verdict_2026-07-26.md) |

| 13 | ★★**HE0의 구조적 이유 — 두 regime의 최적이 *충돌하지 않는다*** | **TRUE per-phase goodput**: 정책간 spread가 **LO(rate 3) 0.067 (2.3%) vs HI(rate 12) 1.187 (43%)** ⇒ **차별의 ~95%가 과부하 phase에서 발생**. LO는 split에 **무관심**(prefill-heavy 극단 d16 2.861 ≈ decode-heavy 극단 d44 2.858 = 구분 불가) ⇒ **LO엔 쫓아갈 최적점이 없고, HI의 최적은 LO에서도 공짜**(§1-6 비대칭의 정량 확인). ⇒ **"항상 HI 최적"=decode-heavy static이 정의상 최선**이고 동적은 과도만 지불. **동적이 이기려면 regime 간 최적이 *충돌*해야 하는데 이 워크로드엔 그 구간이 없다**. ⚠️**정정 이력**: 2026-07-18(§1-16)엔 "이 논증은 관대 SLO 한정, tight선 HI 최적이 동적"이라 봤으나, **§1-17(직접 재튜닝)이 반증** — tight SLO에서도 HI 최적은 **고정 decode-heavy(d44)**이고 동적은 얽힘 트랩으로 열위. ⇒ **이 논증은 tight SLO에서도 성립**(SLO 엄격도 무관) |
| 14 | ★**stationary r8의 "시스템 노이즈" = 메트릭 절벽 (외인성 아님)** | 워크로드 4런 전부 동일(fingerprint), 하부 섭동은 **thru 3%·ITL 8%**뿐인데 goodput 2× — **r8이 TTFT≈SLO(3s) 경계에 앉아** 3% 결손이 TTFT 평탄역을 1.5s→3.7s로 밀어 임계선을 넘김. **3=견고/8=불안정/12=견고** ⇒ 경계 regime만 불안정. 상세 [bench_noise_root_cause.md](bench_noise_root_cause.md) |

| 20 | ★★★**Oracle 재구성(TTFT⊗ITL 분해): headroom은 +2%가 아니라 +16% — 단 그건 disaggregation 몫 (사용자 지적, 2026-07-19)** | §1-19의 "+2.1%"는 per-static oracle이라 **coupled 절충점만** 봄. **goodput을 TTFT-pass ⊗ ITL-pass로 분해**(사용자 지적)하면 진짜 headroom이 보임: phase A(양 SLO 동시 binding)서 **d16 TTFT-pass 57.8%(ITL 실패) / d24+ ITL 100%(TTFT 하락)** — **DECOUPLED oracle**(d16의 TTFT ⊗ d24의 ITL)=**57.8% = best static 49.7% 대비 +16%**. ★**그러나 그 headroom은 92 prefill SM(d16-TTFT) + 24 decode SM(100% ITL) = 116 SM > 108 요구 = coupling TAX(8 SM 초과)라 단일-GPU 불가**(+ 얽힘이 batch로 추가 결합). ⇒ **두 개의 다른 천장**: 단일-GPU **동적**=coupled ceiling **+2%**(못 이김) / **decoupling=disaggregation ceiling +16%**(별도 디바이스 풀서만). **진짜 headroom은 디바이스 간에 존재, 단일-GPU split(동적이든)엔 없음.** `oracle_corrected.png` |
| 19 | ★★★**disjoint-feasibility region은 존재하나 동적은 거기서도 패배 — 이유는 conjunctive SLO의 구조 (사용자 극단-도전 검증, 2026-07-19)** | 사용자 논리(어떤 static도 양 phase 두 SLO 동시충족 못 하는 workload 필연 존재)를 극단 mix(A prefill-heavy in2048/o32@8, **B decode-heavy in2048/o512@5=긴ctx라 ITL-binding**)로 실증: **median-feasibility DISJOINT 확인**(feasible-A={d16} ∩ feasible-B={d24,d34}=∅; job 860497–518). ★**그런데 동적 여전히 패배**: graded goodput서 per-phase 최적이 **인접**(A→d24, B→d34)이라 **ORACLE 동적조차 best-static +2.1%뿐**(d24가 양 phase 근최적: A 2.73=최적, B 1.01 vs 1.09), **reactive bind는 −20.6%**(오배치, 양 phase 실패). ★**깊은 이유**: conjunctive SLO(TTFT∧ITL)가 동적을 **동기부여**(d16 최고TTFT·d44 최고ITL)하는 바로 그 힘이 **각 phase 최적을 중간 compromise로 당김**(d16은 A서 ITL벽·d44는 B서 TTFT벽) → 서로 다른 phase 최적이 인접 → 단일 중간 static이 양쪽 서빙. ★**게다가 이 region은 OVERLOAD서만 존재**(전 정책 gp 0.99–1.87, 대다수 SLO 실패): 용량 이하=전부 통과(static 자명)·이상=전부 실패(static 최소손실). **동적이 유용하게 이기는 operating regime 없음.** `extreme_disjoint.png` |
| 18 | ★★**mix-스윙 트레이스서도 동적 패배 — 최적은 좁은 중간대만 스윙 (사용자 도전 검증, 2026-07-19)** | 기존 결론은 rate만 변하는 fixed-mix 트레이스 한정이었음. **mix-스윙**(phaseA prefill-heavy in2048/o32 ⇄ phaseB decode-heavy in256/o512, static sweep+bind, job 860452–470) 실측: **최적이 d24(A)↔d34(B)로 *좁게만* 스윙**(d16↔d44 아님). ★**prefill-heavy phase를 d16이 안 이김**(d24 5.006 > d16 3.016 > d44 1.564). 기전=**goodput=TTFT-SLO ∧ ITL-SLO가 반대로 당김**: d16 최고 TTFT(1.47s)·최악 ITL(56ms, 60벽 근접); d44 반대(ITL 22ms·TTFT 3.64s로 3s 실패); **중간 d24가 둘 다 충족→승**. **단일 중간 static d24가 양 phase 근최적**(A 5.006=최적, B 2.854 vs 최적 2.870=0.6%차)이라 **combined d24 3.930 ≫ bind 3.419**. ⚠️caveat: phaseB 포화(thru 2.9<offered 5)·n=2; **어떤 static도 양 phase서 두 SLO 동시충족 불가한 극단 mix는 미검증(동적의 남은 문)**. `mixswing.png` |
| 17 | ★★**동적이 지는 이유 = 오버헤드 아니라 *positioning* (실패 지점 규명, 2026-07-19)** | "오버헤드>이득"은 이미 반박(switch~0 §1-8, CPU 0.014% §1-12). 로그가 실패 지점을 정확히 보임: **최적=dec_sm 44(d44, throughput·goodput 양쪽 1위)인데 컨트롤러는 dec_sm 16–24(평균 22)서 진동하며 44에 절대 도달 못 함 = decode-STARVED**. 기전 = **reactive**(TPOT 스파이크 후에야 decode에 SM)+**symmetric**(두 slack 대등화)이라 decode가 잠깐 괜찮아지면 즉시 prefill로 회수 → 구조적으로 decode-heavy 최적에 누적 불가. 손실은 **switch 비용이 아니라 앉은 위치**. ★**risk/reward 18:1**: prefill-ward 이동의 LO 이득 ≤2.3%(§1-13 LO split-무관) vs HI 오배치 손실 ≤41.5% ⇒ 매 스위치가 나쁜 베팅. ★**모든 수정(anchor·비대칭 penalty·이동 중단)이 "44에 앉기"=static으로 수렴** — gate(ratchet, 34서 정지)가 best-dynamic이나 undershoot. **동적은 안 움직여 static과 *tie*가 상한, 이길 regime 없음**(§1-13). `why_dynamic_loses.png` |
| 16 | ★**정책 차이는 throughput이 아니라 SLO-attainment 효과 (2026-07-19)** | 같은 변화-trace 런을 **throughput(SLO-무관 req/s)**으로 재정렬: **스프레드 3.4%** (d44 3.776 > d34 3.759 > bind+GATE 3.745 > bind 3.700 > slo 3.696 > d24 3.677 > d16 3.654) vs **goodput 스프레드 13.1% (4×)**. ⇒ **모든 split이 GPU를 거의 동일하게 포화**시키고, split이 정하는 건 "몇 개 완료"가 아니라 "어느 요청이 TTFT 벽에 부딪히나"(SLO attainment 77.9–85.3%). 순위는 안 뒤집힘(d44 양쪽 1위). ★단 **d16이 양쪽 최하** — 얽힘이 raw throughput도 소량(3.4%) 깎음(decode 굶김→batch 정체→admission 차단→완료↓); d16 goodput 결손의 **~1/4는 실 throughput 손실·~3/4는 SLO attainment**. `throughput_vs_goodput.png` |
| 15 | ★**(D) granularity = *실행시* 비용이지 *결정시* 비용 아님 (2026-07-18)** | per-layer-type SM 분할을 **offline predictor/floor로 고정해도 死**. (D)는 "누가 split을 정하나(런타임 vs offline)"가 아니라 "한 forward *안에서* 파티션이 layer 경계마다 바뀌나"의 문제 — offline 고정값이라도 실행 시 **attn↔mamba 경계마다 green-ctx 재분할 필요**(step 파편화·sync 직렬화·overlap 감소, S2서 **TPOT 42→124ms**). offline은 **결정 오버헤드만** 제거·**실행 파편화 비용은 그대로**. ⇒ **살아있는 offline 역할은 오직 whole-phase floor**(step 내내 단일 파티션, composition으로 크기만 결정 = §1-5). ★**비용 분해(2026-07-18 확인)**: green-ctx는 시작 시 `initialize_stream_groups`로 **전부 pre-created**(스위치=인덱싱; 생성비용 없음). 실제 스위치 비용 = 경계마다 `stream.synchronize()` **드레인**. 이를 GPU측 wait_stream 순서화로 교체(`PDMUX_LA_COORD_OPT`)하면 **124→85ms(갭 ~47% 회수)**나 **여전히 패배**(agnostic 42ms 평탄). **잔차 = 구조적 오버랩 손실**(monolithic prefill이 window 0만 오버랩, 윈도우수 무관·모델 독립) + **cudagraph 비양립**(step 중간 green-ctx 전환 캡처 불가→eager 강제→운영점 진입 불가). ⇒ **"싼 전환"으론 절반만 없앰; 나머지 절반은 pre-created로도 불가.** `results/a_substrate/` |

| 16 | ★**HE0는 goodput SLO 엄격도에 의존 — tight SLO에선 동적이 best-static과 대등~약우위 (2026-07-18)** | **기존 벤치 재분석**(job 재제출 없음, per-request `input_lens`/`ttfts`/`itls` 재스코어; `results/slo_sched/lengthnorm_slo_reanalysis.md`, `reanalyze_lengthnorm_slo.py`). **sanity**: fixed-3s 재현이 §1-7과 정확 일치(d44 3.220±0.013). ★**fixed-tight sweep**(길이 무관, 순수 엄격도): 승자 = **d44@{3.0,2.0,1.5,1.0,0.75s} → d34@0.5s → bind+GATE@0.335s**, 교체 임계 **TTFT 0.5–0.75s**(=실 prefill mean~112ms의 3–5×). ★**축은 길이-비례성 아니라 엄격도**: 같은 ~335ms 평균예산서 flat SLO(bind 2.593) ≈ 길이비례 SLO(bind 2.582) = 둘 다 동적 승(길이비례 여부 무영향). **n≥4 baseline**(d24/d16/slo, job 859005–859059)서도 norm-k(2/3/4) 전부 bind+GATE ①. **강도(보수적)**: bind+GATE vs **d44 +0.073(~3σ, 유의)** / vs best-static **d34 +0.042(~1.5σ, 대등)**, floor변형선 d34≈bind 무승부 ⇒ **"동적이 압도"가 아니라 "동적이 best-static과 대등~약우위, decode-heavy static 지배는 반증"**. **기전**(phase 분해): 반전은 HI(과부하) phase에서만 — 관대SLO=완료율 지배(decode throughput=decode-heavy 승) / tight SLO=first-token 반응성 지배(부하 중 prefill 저글링하는 동적 승, decode-heavy static은 prefill 굶겨 꼴찌권). **한계**: Zamba2 short-ctx·ShareGPT p99 2776tok 한정, 재분석은 3s 벤치 데이터 재스코어(인터랙티브 TTFT를 직접 attain 측정한 건 아님). ★**실무 관행 조사로 지위 강화 (2026-07-18, `serving_slo_survey.md`)**: 프로덕션 인터랙티브 TTFT P99 = **chat 300ms·voice 150ms·code 100ms·RAG 400ms** = **전부 tight regime(동적 승)**; 우리 정본 3s는 표에서 **"batch async" 행**에 정확 대응 ⇒ **"static 지배"는 배치 서빙 한정, 인터랙티브 주류는 동적 regime**. SLO를 배수로 sweep(DistServe "SLO scale")은 표준 방법론이고 "엄격할수록 구조/반응성 이점이 드러남"도 알려진 패턴(DistServe: strict→disaggregation). ⚠️★**이 "동적 우위"는 §1-17(직접 재튜닝 측정)에서 아티팩트로 반증됨 — 재스코어는 컨트롤러 *행동*을 못 봤다** |
| 17 | ★★**§1-16 반증 — tight SLO로 컨트롤러를 *실제 재튜닝*하면 동적은 best-static에 크게 열위 (2026-07-19)** | §1-16은 3s-튜닝 컨트롤러의 궤적을 tight SLO로 *사후 재스코어*(행동 불변)한 것. 이번엔 **컨트롤러 SLO를 chat(TTFT 300/ITL 50ms)로 실제 설정**해 직접 서빙(`interactive_bench.sbatch`, jobs 860415–860514). **용량 = rate 7–8**(rate≤6 무관심·≥10 전붕괴), 판정은 경계 **rate 8, n=4 attainment%**: **d44 73.2±4.8 ≫ d34 49.6±3.9 > bind+GATE 44.3±2.7 > bind 40.6±0.4**. ★**d44 vs bind+GATE = 28.9%p ≈ 10σ**. static 단조(decode SM↑=attain↑: d16 33<d24 41<d34 50<d44 73), **동적은 2위 static(d34)도 못 넘음**. **기전**: 컨트롤러가 tight TTFT에 반응해 prefill-ward 이동(switch 24–30)→decode 굶김→§1-4 얽힘 트랩→batch 정체→TTFT 악화(bind TTFT p90 1.6s vs d44 0.44s). §1-16이 상상한 "tight→prefill 반응성 유리"가 실제론 **역효과**. bind+GATE>bind는 게이트가 trap 억제(§1-11 재확인, rep2 feas=3서 게이트 미발동→bind급 하락=반증실험). ★**결론: decode-heavy static이 관대 SLO(§1-7)뿐 아니라 tight SLO에서도 지배, 오히려 격차 더 큼(§1-4·§1-6이 tight서 더 극명). §1-16의 조건부화는 취소 — SLO 엄격도와 무관하게 static 지배.** code(100/25)는 무경쟁 66%로 HT-neg(물리 불가). 상세 [interactive_slo_retune_plan.md](interactive_slo_retune_plan.md) §9 |

**실전 권고**: **peak decode 부하 기준 decode-heavy static split 고정**(이 워크로드선 d44급). 동적 불요 — **관대(3s)·tight(chat 300ms) SLO 양쪽에서 확정**(§1-7·§1-17).
**게이트를 굳이 쓴다면**: 수동 튜닝 없이 안전한 static을 자동으로 찾는 **auto-tuner**로서만 값어치(최적에 미달; tight SLO선 trap 억제로 bind보다 낫지만 여전히 static 미달).

---

## 2. ★철회·불확실 (2026-07-17 분산 측정으로 무너진 것)

| # | 이전 주장 | 현재 상태 |
|---|---|---|
| 1 | "stationary bimodal(6.24↔2.22)은 **양성피드백 트랩** 때문" | ★**stationary 벤치 한정 과잉 귀속 — 철회 유지**(static d24도 switch=0인데 6.32↔3.10 붕괴 ⇒ 거기선 노이즈와 분리 불가). ★**그러나 트랩 자체는 2026-07-17 부분 복권**: **유효 벤치(변화 trace)** 에서 **d44가 ±0.013 = 노이즈 없음이 증명된 조건**인데도 **no-gate만 1/4 붕괴(TRUE 2.405 vs 정상 3.10–3.12), gate는 0/9** ⇒ 거기서의 붕괴는 **컨트롤러 탓이 맞다**(§1-11). **정정된 주장**: "트랩은 실재하고 게이트가 막는다 — 단 stationary 벤치의 bimodal은 그 증거가 못 된다" |
| 2 | "d24-static은 ±0.039로 안정" | **n=2의 운.** 실제 **5.282 ± 1.302 (n=4, min 3.102)** |
| 3 | "feasibility 게이트가 트랩을 없애 **성능 회복**" | ★**2026-07-17 유효 벤치서 분해 — 절반 확정·절반 반증.** **견고성은 확정**(no-gate 2.934±0.306·1/4 붕괴 → gate 3.132±0.019 (n=9)·0/9, 16× 타이트 = §1-11). **성능 회복은 반증**(gate 3.132 < d34 3.171 < **d44 3.220**; §1-10 = ratchet이 틀린 static에 조기 정지). ⇒ "**트랩은 없애나 성능은 여전히 static 미달**" (구 stationary 수치 5.928/5.282/5.269는 노이즈 교란이라 폐기) |
| 4 | 최근 n=1~3 정책 비교 다수 | **underpowered** — 베이스라인 ±1.3이 정책 차이를 삼킴. 재측정 없이 인용 금지 |
| 6 | (내 가설) "stationary 분산 = **GPU 클럭/전력 throttling**" | ★**철회 (2026-07-17)** — 불필요. 설명 대상은 TTFT 3.5×가 아니라 **throughput 3%**였고, 증폭기는 **SLO 임계 절벽**이었다(§1-14). 잔여 3%(co-tenant/클럭/페이지캐시)는 상존·무해 |
| 5 | 초기 SLO track "isolation 오버헤드 0"(2.318≡2.319) | **주의 플래그** — 당시도 n이 작았다면 같은 함정. 재확인 전까지 약한 근거로 취급 |

---

## 3. 방법론 (교훈 — 앞으로 필수)

1. ★**stationary ShareGPT r8 = 정책 비교 벤치로 부적합·폐기.** 동일 프롬프트(`--seed` 고정)·switch 0인 static조차 **±1.302** → 신호를 삼킴.
2. ★**변화 trace(rate 3↔12, 3라운드 평균) = 유효 벤치** (±0.02). **정책 비교는 이걸로.**
3. **베이스라인 분산을 먼저 측정**하고 시작. **n≥4** 없이 정책 결론 금지.
4. **dynamic 결과엔 항상 `switch_count` + split 체류분포 + TTFT/ITL p50/p95/p99 병기.**
5. **Switch decomposition**: `Net = Σ(B positioning) − (A switch × drain)`. (A)와 (B)를 분리 귀속.
6. ★**메트릭이 임계 지시함수(goodput=TTFT≤SLO)면 절벽을 피해 측정할 것** — 용량을 먼저 재고, TTFT 평탄역이 SLO 임계에 걸치는 rate는 **3% 섭동을 2× 신호로 증폭**한다. 과부하 구간의 threshold-goodput은 **런 길이 의존 = ill-posed**.
7. ★**여러 라운드를 한 파일에 append하는 하네스는 분모를 반드시 합산**(`dur+=d`; `max()`는 라운드 수만큼 부풀림 — `f921ae8`서 3× 버그로 실현).
8. **변수는 하나씩** (pf_urg와 dwell 동시 변경 → 해석 불가였던 전례).
9. ★**coupled 스윕은 반대편 축을 공변시켜 confound되기 쉽다**(2026-07-26,
   Stage 0). decode-SM을 낮추며 동시에 prefill-SM을 높이는(또는 그 역) 하네스는
   관측 곡선이 스윕 축 자체의 효과인지 반대쪽에서 늘어난 경합/de-batch의 효과인지
   구분 못 한다. **음성 대조**(그 축에 원리상 binding 불가능한 arm)와 **무경합
   앵커**(공변이 0으로 붕괴하는 특수점)의 조합이 confound를 identify한다 —
   "정책/측정 주장에는 음성 대조가 필수"라는 교훈. 상세
   [`stage0_verdict_2026-07-26.md`](stage0_verdict_2026-07-26.md) §5.

---

## 4. 살아있는 문서 (이것만 참조)

| 문서 | 역할 |
|---|---|
| **CONSENSUS.md** (이 문서) | **정본** — 확정/철회 상태 |
| **`per_layer_type_postmortem.md`** | ★**per-layer-type 구제 시도→기각 kill-chain**(시도 A–H, 두 조건 C1 lever·C2 착취 관점) |
| **`research_arc.md`** | ★**연구 아크 전사** — 출발점→현재의 단계별 시작논의/촉발지표/해소지표/부정사유(수치 포함). CONSENSUS의 서사 짝. **§S-M = 서빙 수준 반증 지점(등급별) · 측정 환경(E1–E4) · 결론의 유효 경계** |
| **`longcontext_trace_plan.md`** | ★**계획 문서**(L−1 이상은 측정 전) — 실 trace를 long-context로 전환하는 문제. 동기(Diff A는 L≥3k서 열림) vs 정직한 반론(Diff B는 long-L서 닫힘) · 하드 블로커(Zamba2 ctx 4096 / goodput SLO 붕괴) · 단계 L0–L3. **L−2(Stage 0) 게이트만 실행·확정**(2026-07-26, non-binding — 아래 `stage0_verdict_2026-07-26.md` 참조), 그 이상은 여전히 계획 |
| **`stage0_verdict_2026-07-26.md`** | ★**Stage 0(long-ctx L−2) 최종 판정** — 운영점 decode SM-무감각을 hybrid·pure-Transformer·pure-Mamba·ctx≤16k로 확장 확인(§1-21). coupled 스윕 confound 진단 + de-confounded D16≡D108 null 결과 |
| `results/slo_sched/lengthnorm_slo_reanalysis.md` | ★**길이-정규화/tight SLO 재계측**(§1-16) — HE0가 SLO 엄격도 의존임을 기존 벤치 재분석으로 확정. 스크립트 `reanalyze_lengthnorm_slo.py` |
| **`serving_slo_survey.md`** | ★**실 서빙 SLO 관행 조사**(§1-16 후속) — 프로덕션 인터랙티브 TTFT(chat 300/voice 150/code 100/RAG 400ms)가 전부 tight regime; 우리 3s=batch async. DistServe SLO-scale sweep=표준. goodput 메트릭 비판 |
| **`interactive_slo_retune_plan.md`** | ★**tight-SLO 컨트롤러 재튜닝 + P90-attainment 직접 측정**(§1-17) — §9에 최종 결과(HT0 확정, d44≫동적). 하네스 `results/slo_sched/interactive_bench.sbatch` |
| `bench_noise_root_cause.md` | ★벤치 노이즈 근본원인(메트릭 절벽)·3× 하네스 버그·HE0 구조적 이유 |
| `realtrace_findings_and_open_branches.md` | 실 trace 검증 + 얽힘 기전 + 남은 갈래(트리거/행동모델) 상세 |
| `slo_aware_scheduling_design.md` | SLO-aware track 설계·실측 전사(§C–§HE2-3, Step D/E/F/G) |
| `policy_comparison.md` | 정책 taxonomy·메커니즘 (⚠️ §goodput 수치는 no-cudagraph·구벤치 — §1·§2 우선) |
| `system_vs_engine_vs_sim.md` | fidelity ladder (sim/engine/full-system 편향 분리) |
| `prefill_vs_decode_execution.md` | prefill/decode 실행 특성·knee 기초 |
| `sm_policy_report.html` | layer-aware 원자료 기록(死 트랙, 이력용. §coordinated 개정 미완 = stale) |
| `paper/venue_positioning.md` | ★**전략 문서(2026-07-24, 2026-07-25 §0.1 갱신)** — 투고 positioning. claim/evidence 아님, CONSENSUS/matrix 판정을 인용·요약만 함. 새 증거 근거로 쓰지 말 것. §0.1 = 신규성 축(disaggregation→multiplexing) 정정 + negative (A)/(B) 분해 + green-context vendor-primitive 방어 + 벡터2 게이트(cross-substrate 이식 철회 → Transformer-control on green-context) |
| `spatial_decoupling_design_review_2026-07-25.md` | §1-20 disaggregation 기판 실현가능성 설계 검토. **§1(신규성)은 위 `venue_positioning.md` §0.1로 부분 supersede**(문서 상단 HISTORICAL 노트), §2–§5(SGLang v0.5.10 기판 file:line, long-ctx 게이트 연결점)는 유효 |

`deprecated_reports/`(2026-07-24부터 [`../deprecated/reports/quarantine_engine_port/`](../deprecated/reports/quarantine_engine_port)) = 초기 triage·포팅·모델별 평가·구 핸드오프·구 리포트. **이력 보존용, 현재 결론과 충돌 가능.**

---

## 5. 열린 항목

1. ~~**변화-trace 기반 재검증**~~ → ✅ **완료 (2026-07-17, jobs 856889–856975)**. n≥4 캠페인으로 **HE0 견고 확정**(§1-7, 5.4σ) + **게이트 정체 규명**(§1-10/11).
2. ~~**게이트 정교화 필요?**~~ → ✅ **성격이 바뀜**: 게이트는 지능적 제어가 아니라 **auto-tuner**(§1-10). 살릴 값어치가 있다면 **ratchet의 조기 정지 수정**(정지 규칙이 d34에서 멈춰 d44를 놓침) — 단 그래봐야 천장은 "best-static 매칭"이라 payoff는 *튜닝 자동화*뿐.
3. ~~**컨트롤러 CPU 오버헤드**~~ → ✅ **직접 계측으로 死 (2026-07-17, jobs 857111/2)** = §1-12.
4. ~~**시스템 노이즈의 정체**~~ → ✅ **종결 (2026-07-17)** = **노이즈가 아니라 메트릭 절벽**. 상세 [bench_noise_root_cause.md](bench_noise_root_cause.md).
   워크로드는 4런 전부 **동일**(fingerprint 일치)이고, 하부 섭동은 **throughput 3%·ITL 8%뿐**. 증폭기는 **r8이 하필 TTFT≈SLO(3s) 경계에 앉은 것** — 과부하 큐의 TTFT 평탄역이 3% 결손에 1.5s→3.7s로 이동해 **임계선을 넘음 → goodput 반토막**. rate로 재현: **3=견고(200/200×3) / 8=불안정(400,400,357,206) / 12=견고(142,141,142)** ⇒ 불안정한 건 **경계 regime뿐**. ★**GPU 클럭 throttling 가설 철회**(3.5×가 아니라 3%만 설명하면 됨). ★**자원 격리 불필요했음**. **stationary 부활 조건**: 용량(d24≈6.3/s) 아래 rate에서 측정하거나, 임계 지시함수 대신 TTFT 분포/용량 지표 사용(현 goodput은 과부하서 런 길이 의존 = ill-posed).
5. ~~(낮음) 얽힘-aware 행동모델의 정밀화~~ → **천장이 "static 매칭"으로 확정**(관대·tight 양쪽, §1-7·§1-17). 동적 upside는 **모든 SLO 엄격도에서 닫힘**. payoff 없음.
7. ✅**완료·종결 (2026-07-19)**: **tight-SLO 서브트랙.** (b) SLO 관행 조사(`serving_slo_survey.md`): 인터랙티브 주류(chat/voice/code, TTFT 100–400ms)가 tight regime. (a) **컨트롤러 재튜닝 직접 측정**([interactive_slo_retune_plan.md](interactive_slo_retune_plan.md) §9, jobs 860415–860514): chat(300/50)으로 컨트롤러 실제 재튜닝 → **§1-17 = HT0 확정**(d44 73.2%≫bind+GATE 44.3%, 10σ). ★**§1-16의 "tight 동적 우위"는 재스코어 아티팩트로 반증** — 재스코어는 컨트롤러 *행동*을 못 봤다(방법론 교훈: SLO를 목적함수로 바꾸는 실험은 반드시 컨트롤러를 그 SLO로 재튜닝해 직접 측정). code(100ms)는 HT-neg(무경쟁 66%, 물리 불가). ⇒ **single-worker·SM-split·reactive 동적 제어(SLO-aware/binding-first/feasibility-gate)는 관대(3s, §1-7, n≥4)·tight(chat 300/50, §1-17, n=4) SLO 양쪽에서 best-static을 못 넘는다 — 이 범위는 확정, 반증 실패.**
   ★**정정(2026-07-24, claims-auditor 감사)**: 위 "⇒ 동적 제어 트랙 완전 종결"이라는 이전 표현은 **overclaim이라 철회**(취소선 아님, 이 정정으로 대체) — [`../PROJECT_STATUS.md`](../PROJECT_STATUS.md)는 "완전 종결"을 말하지 않으며 Claim D(true dual-worker coupling 감소)·Claim E(Hybrid-informed policy)를 **미검증**으로 명시적으로 열어둔다. §1-20(oracle 분해)도 canon 스스로 disaggregation ceiling +16%가 decoupled substrate엔 열려 있다고 정량화해 "완전 종결"과 정합하지 않았다. **종결된 것은 위 좁은 범위(single-worker·SM-split·reactive)뿐**이며, 미결 항목은 §5-8 참조.
6. ★**long-context 실 trace로의 전환** (사용자 발의 2026-07-17) — **계획 단계**. 근거: **모든 서빙 반증이 short-context**(ShareGPT 98%가 L<2k)이고 창립 동기의 Diff A는 **L≈3k에서 교차해 열린다** ⇒ 결론의 **유효 경계**가 컨텍스트 축에서 미확인. 단 **layer-aware 부활 경로 아님**(Diff B는 long-L서 ≈1.0으로 닫힘). 실제 stake = **최적 static 위치 · 얽힘 병목의 KV 재편 · HE0 반전(혼합 trace에서만 가능)**. 블로커: **Zamba2-2.7B ctx 4096**(모델 교체 필수 → 전 baseline 재측정) · **goodput SLO가 long prefill서 붕괴**(전 정책 0). 상세·단계 게이트 [longcontext_trace_plan.md](longcontext_trace_plan.md).
   ★★**Stage 0(L−2) 게이트 실행 완료 — non-binding, 2026-07-26**(§1-21,
   [`stage0_verdict_2026-07-26.md`](stage0_verdict_2026-07-26.md)): 이 항목이
   전제하는 "decode floor가 ctx로 상승해 운영점서 binding해진다"는 **ctx≤16k에서는
   반증**됐다 — D16≡D108(1.00±0.01)이 pure-Transformer·pure-Mamba·hybrid 전부에서
   성립. `longcontext_trace_plan.md` §6의 사전 등록 게이트 규칙대로 **L−1 이상
   (모델 교체 baseline·SLO 재정의·시간축/공간축 충돌 스윕)은 이 regime에서
   진행 근거가 없어 멈춘다**. 남은 미측정: **>16k ctx, 더 큰 모델** — 이 항목은
   그 범위로 좁혀져 여전히 열려 있다.
8. ★**미결(종결 아님, 2026-07-24 스코프 정정으로 신설)**: §5-7의 "완전 종결"은 아래 세 갈래를 배제하지 않는다.
   - **(a) dual-worker(decoupled) 동적** — [`../PROJECT_STATUS.md`](../PROJECT_STATUS.md) Claim D("역할별 queue/host issue loop/CUDA stream을 실제로 분리하면 single-worker control-plane coupling을 줄일 수 있다")는 **서빙 측정 0건**(증거 수준 "미검증"). §1-20이 정량화한 **+16% disaggregation ceiling**(92 prefill SM + 24 decode SM = 116 > 108 = 단일-GPU coupling tax로 불가능)은 **별도 디바이스 풀 + hybrid state transfer를 갖춘 decoupled substrate에만 열려 있다**. ★★**정정(2026-07-24, engine-porter 코드 리뷰, 읽기전용, [`r2_decoupling_review_2026-07-24.md`](r2_decoupling_review_2026-07-24.md))**: `PDMUX_TRUE_DUAL_WORKER=1`는 이 decoupled substrate에 **해당하지 않는다** — file:line 근거로 확인한바 두 host issue thread/role별 task queue/immutable `ExecutionContext`/thread-local role(ContextVar)만 분리하는 **control-plane dual-worker**이고, running batch(`max_running_requests`)·KV/mamba pool·SM 파티션(`SharedGpuArbiter`의 단일 `stream_index`)은 **전면 공유**된다(92+24=116의 별도 device pool이 아니라 ≤108 단일 coupled index). §1-4 死因 얽힘(공유 running-batch+KV)의 substrate가 불변이므로 이 구현은 **구성상 +16% headroom에 도달 불가**하며 coupled ceiling(+2%, §1-20) 위에 앉아 있다. state-transfer 경로·mamba conv/ssm state migration은 **코드에 전무**(스캐폴딩조차 없음). ⇒ **Claim D는 "control-plane coupling 감소"로만 유의미하게 측정 가능**, "얽힘 깨기"로 팔 수 없다. 부가: admission latch(`r2_admission_limited`)에 **known-latent stale-True 버그** 확인 — split batch가 None으로 배수되면 재평가 경로가 없어 latch가 True로 고착되어 prefill admission을 영구 차단할 수 있다(clear 경로 부재). **사용자 결정으로 현재 수정하지 않고 보류.** R2는 GPU correctness gate를 통과한 이력이 없다(`results/r2_eval/` 디렉터리 미생성, `architecture=true_dual` telemetry 전무). 이 headroom이 (다른 substrate에서) 실현되는지는 여전히 미검증. 게이트는 PROJECT_STATUS "다음 실험 gate" §2 참조.
   - **(b) 비-SM-split lever** — §1-4 얽힘의 死因은 **공유 running-batch capacity·KV**(SM 분할 자체가 아님). admission-control 또는 KV-aware한 lever로 이 死因을 직접 겨냥하는 시도는 **구현조차 되지 않았다**. 지금까지 종결된 것은 전부 *SM-split* 기반 컨트롤러(SLO-aware/binding-first/feasibility-gate)뿐.
   - **(c) 충돌 regime 워크로드** — "동적이 이길 disjoint-feasibility escape hatch가 없다"는 결론은 §1-18(mix-스윙, **n=2**)·§1-19(극단 disjoint, **n=1~2, overload-only**)의 **underpowered 탐침**에 근거하며, 이 자체가 §2-4 방법론("n≥4 없이 정책 결론 금지")에 못 미친다. long-context 혼합 trace(§5-6의 stake 중 "HE0 반전")도 아직 미측정.
     ★**HE0-reopen 벡터1(n≥4 재시도, 2026-07-24 실행·2026-07-25 판정)**: G2.0 short-ctx disjoint 스윕(Zamba2-2.7B, ctx4096, Phase A in2048/o32 vs Phase B in2048/o512–1024, rA5)으로 (c)를 n≥4로 재검증 시도. **g2_0_full**(n=4/mode, `../workspace/engine-port/results/g2_0_full/disjoint_verdict_2026-07-24.md`)는 razor-thin real disjoint를 보고(feasible-A={d16,d44} ∩ feasible-B={d54}=∅, d54 Phase-A 0.852 fragile). **g2_0_hard**(n=6–10/mode, hardening axis 2개 + claims-auditor 독립 재채점, `../workspace/engine-port/results/g2_0_hard/hardened_disjoint_verdict_2026-07-25.md`) 판정: **ILL-POSED at rA5, escape hatch 근거로 "지지 안 됨"이나 "종결"도 아님.** 근거: (i) 동일 byte-identical Phase-A 워크로드에서 d44/d54 견고성 순위가 sweep 간 완전 반전(d44 0.969→0.695, d54 0.852→0.938), pool(n=10) 시 둘 다 ~0.86–0.90로 통계적 구분 불가 — 이는 §1-14 "메트릭 절벽" 기전과 동일한 TTFT 3s-cliff bimodality. (ii) "축1서 feasible-B가 d34로 넓어져 disjoint 소멸"은 물리적 완화가 아니라 **per-request ITL-p95 percentile-window 아티팩트**(OB512→1024서 ITL 중앙값은 상승하는데 p95만 하락 — 초기 고정 스파이크가 더 긴 출력에 희석). ⇒ 견고한 disjoint도 견고한 공유 static도 입증 안 됨. **§1-20(spatial coupling-tax, 116>108, disaggregation +16%)과는 무관 — 별도 축(시간적 disjoint-feasibility vs 공간적 SM-budget), §1-20은 이 결과로부터 영향 없음.** **방법론 교훈 강화**: 또 절벽 위에서 측정(§3 gate #6 위반) — g2_0_hard가 gate #6을 지키려 워크로드를 바꿨으나(axis1) 그 자체가 새 percentile 절벽에 올라앉아 실패; feasibility 판정 전 **capacity de-cliff 선행** 필수, "결과가 바뀌었는가"가 아니라 "메트릭이 여전히 절벽/percentile 경계에 앉았는가"로 de-cliff 여부를 검증해야 함. **de-cliff 재스윗 pending**(rA 추가 인하로 Phase-A p90≪3s 확보 → 출력-길이 불변 ITL 지표 → ≥2-SM-step 간극 n≥6 paired). (c)는 여전히 **열려 있음** — 이번 라운드로도 확정도 반증도 안 됨.
   ⇒ **정확한 종결 범위**: single-worker·SM-split·reactive 동적 제어가 관대·tight SLO 양쪽에서 best-static을 못 넘는다는 것만 확정. **(c) 충돌 regime(short-ctx)은 2026-07-25 `g2_0_raconf` 확증으로 CONFIRMED closure(scoped) — 아래 참조**. 트랙의 나머지(dual-worker architecture (a), non-SM-split lever (b), **long-ctx 충돌 regime**)는 여전히 **열려 있음**.

   ★**de-cliff stage-1 완료(실험 2026-07-24~25, 기록 2026-07-25, jobs 863880–863948,
   `../workspace/engine-port/results/g2_0_decliff/decliff_verdict_2026-07-25.md`)**:
   위 `g2_0_hard`가 권고한 de-cliff 재스윗의 1단계(capacity de-cliff only, feasibility
   판정 아님). `rA{2,3,3.5,4}×{d16,d44,d54}`를 n=3(coarse)로 스캔한 결과 **`rA=2`만
   전 split에서 clean off-cliff**(mean p90<2.0s ∧ std p90<0.15s); `rA≥3`은 전부
   여전히 bimodal(`g2_0_full`/`g2_0_hard`와 같은 절벽 재현). `rA=2`를 n=6으로 확증.
   **판정 = "견고한 off-cliff disjoint를 찾지 못함 = PLAUSIBLE closure, CONFIRMED
   아님"**: 유일한 clean off-cliff 지점 `rA2`에서 static `d54`가 양 phase를 동시
   커버(Phase-A frac_good 0.974±, TTFT p99≤1028ms; Phase-B ITL-p95 42.3±0.10ms,
   frac_good 1.0, 출력-불변 지표로도 견고) — `feasible-A={d16,d44,d54} ∩
   feasible-B={d54} = {d54} ≠ ∅`. 후보 disjoint는 `rate≥3.5`(전부 bimodal 절벽,
   n=3)에서만 재등장 — 새로 해소된 regime이 아니라 기존 절벽의 재확인.
   ★**"종결" 불가(claims-auditor 반증)**: (a) split→TTFT gradient가 off-cliff에서도
   살아있음(`rA2` p99 `d16` 712±37ms vs `d54` 1028±101ms, `t≈7.2`, 단조) → rate↑
   시 `d54`가 절벽을 먼저 넘는 disjoint 발생 경로 미배제; (b) `d54`-배제 onset
   (~rate 3.0–3.5)이 정확히 **미측정·n=3·bimodal 전이대**라 논증만으론 못 닫음;
   (c) 핵심 등식 "`binding-A` ⟺ `on-cliff`"는 "`d54` 배제"와 "`d54` 절벽-flicker"를
   혼동한 **미증명 경험명제**. **선행 필수(pre-registered stage-2, 진행 중, 미실행)**:
   narrow `rA{2.25,2.5,2.75,3.0,3.25}×{d16,d44,d54(+d34)}` n≥6 확증 sweep — 반증
   표적은 어떤 rate에서 `d54` 견고히 <0.7(p90>3s, unimodal) ∧ `d44`/`d16` 동시에
   견고히 ≥0.95·off-cliff(p90<2s)이면 disjoint 실재 → **벡터1 REOPEN**. **scope
   한정**(필수): {Zamba2-2.7B, ctx4096, Phase A in2048/o32, Phase B in2048/o512@rB4,
   triton attn+mamba, `disable-radix-cache`, cudagraph-ON, A100 108-SM
   green-context pdmux, SLO=TTFT 3s ∧ per-req ITL-p95 50ms, inter-phase drain된
   순차 2-phase} — "hybrid엔 disjoint 없음"으로 일반화 불가. **drain caveat**:
   closure는 얽힘 억제(drain) 조건 관측 = 필요조건 bound이지 hot varying-trace
   (Claim C 얽힘) 실증 아님. ★**§1-20 방화벽 유지**: 시간적 disjoint(단일 static이
   양 phase를 시간축에서 커버)와 공간적 coupling-tax(92+24=116>108, disaggregation
   +16%)는 별개 축 — "단일 static으로 충분 ⟹ coupling tax 없음"으로 새지 않는다.
   §1-20은 이 결과로부터 영향 없음. de-cliff stage-1 시점엔 (c)가 **한 눈금
   전진**(rA5 ILL-POSED → rA2 PLAUSIBLE-not-CONFIRMED closure)이었으나 여전히
   **열려 있었다** — stage-2 확증 sweep 전까지 어떤 방향의 결론도 채택하지 않았다.

   ★★**(c) 최종 종결 — CONFIRMED closure(scoped), 2026-07-25**: 위 stage-2
   확증 sweep 두 단계가 완료됐다. **`g2_0_rasweep`**(120 job,
   `rA{2.25,2.5,2.75,3.0,3.25}×{d16,d34,d44,d54}×n6`)이 off-cliff sub-band
   (rate≤2.75)에서 disjoint를 재확인하지 못해 전이대를 rate 3.0–3.5로
   좁혔다(raw jsonl만 존재, 별도 verdict 미작성 — provenance
   `../workspace/engine-port/results/g2_0_rasweep/`). 그 좁혀진 창을 겨눈
   claims-auditor pre-registered **`g2_0_raconf`**(24 job, `rate{3.5,3.75}×
   {d44,d54}×n6`, 결정 규칙: 어떤 rate서든 `d54` 견고히 <0.7(p90>3s, unimodal)
   ∧ `d44`/`d16` 동시에 견고히 ≥0.95·off-cliff(p90<2s)면 disjoint 실재→REOPEN,
   아니면 `d54`가 양 phase를 동시 커버하는 companion collapse면 CONFIRMED)가
   **companion collapse로 판정**: rate3.5 `d44` 0.953±0.035 ≈ `d54`
   0.948±0.035(d54 failTTFT=0/6); rate3.75 `d44` 0.932±0.042 < **`d54`
   0.948±0.062**(d54가 오히려 높음). REOPEN 전제 둘 다 붕괴(`d54`는 어느
   rate서도 <0.7이 아니고, `d44`도 어느 rate서도 견고히 ≥0.95가 아님 — 웜업성
   TTFT-blowup이 rep 하나에서 0.844–0.875까지 끌어내리며, 이 blowup은
   **split-대칭적**이라 disjoint를 만들지 않는다). `d54`는 Phase B의 유일
   feasible split(`d44` ITL-p95 50.7ms로 50ms SLO 초과, `frac_good` 0.188;
   `d54`는 44.1ms, `frac_good` 1.000, SD=0)이면서 Phase A도 `d44`와 대등하게
   커버 → 단일 split이 양 phase를 시간축에서 커버 → **disjoint 없음, 최종
   확정**. **magnitude는 ill-posed(metric cliff)나 순위(d54≈d44, d54
   미선-배제)는 견고**; Phase-B `d44` 0.188은 50ms 경계 위라 magnitude
   fragile·방향 견고. scope는 위와 동일({Zamba2-2.7B, ctx4096, Phase A
   in2048/o32, Phase B in2048/o512@rB4, ..., drain된 순차 2-phase,
   **rate_A≤3.75**} 한정, "hybrid엔 disjoint 없음"으로 일반화 금지). §1-20
   방화벽 불변(시간적 disjoint와 공간적 coupling-tax는 별개 축). **남은
   방향**: long-context(decode floor 상승 영역, §1-5) 재검증, §1-20 spatial
   decoupling. ★**2026-07-26 갱신**: 전자는 Stage 0(L−2) 게이트로 부분 실행됨
   — ctx≤16k에서는 decode floor가 운영점서 상승하지 않아(§1-21) **이 항목도
   ctx-무관으로 강화되는 방향**(disjoint 없음이 long-ctx로도 재확인될 가능성이
   높아짐), 단 >16k·큰 모델은 미측정이라 최종 확정 아님. §1-20 spatial
   decoupling은 여전히 미실행. 상세
   [`../workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md`](../workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md),
   [`stage0_verdict_2026-07-26.md`](stage0_verdict_2026-07-26.md).
