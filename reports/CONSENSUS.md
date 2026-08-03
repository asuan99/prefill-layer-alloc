# CONSENSUS — engine-port PD-mux 연구의 합의점 (2026-07-19 historical)

> **현재 전체 정본:** [`../PROJECT_STATUS.md`](../PROJECT_STATUS.md).
> 이 문서는 dual-worker/R2 이전까지 확정된 phase separation, layer-granular
> negative result, entanglement, single-worker dynamic 결과의 정본으로 유지한다.

최종 갱신: 2026-08-03 rev5 (★★§1-27 신설[같은 날 2차 속행, doc-steward 기록]:
(I) claims-auditor가 §1-26의 `A_free` 결함을 대체하는 **조건부 per-token
추정량**(`m3_conditional.py`)으로 estimand를 이관 — SPLIT/UNSPLIT 라벨(≥0.90/
≤0.10, 사이는 배제), primary `p95(SPLIT)` 비, UNSPLIT control(대비 정의상 0)
[AUDITED, blocking-threshold 스윕(3)만 UNAUDITED — 감사자가 자기 산출을 자기가
감사한 형태]. (II) engine-porter가 `PDMUX_STICKY_PARTITION` 구현 완료 —
decode-busy 시 무분할 fallback 우회, decode-empty 시엔 의도적 release(hold
아님), OFF는 short-circuit으로 patch 전과 byte-identical, correctness gate
전부 PASS(CPU 회귀 40 tests + sticky 단위 테스트 12 + GPU smoke job 872800
byte-identical 출력), realized 관측(n=1) `E1_DECODE_REALIZED` OFF 0.0839→
ON 1.0000. **구현 완료 ≠ 성능 주장 성립.** (III) sticky 런 사전등록: 872077
소급 재분석은 DIAGNOSTIC 전용, primary 1개 선언, 게이트 4종, **`G_LEVER`/
`G_FLAT`는 미결정으로 기록**(스케일 불일치로 기존 1.5/1.15 이전 불가). 상세
`../PROJECT_STATUS.md` "8B decode-SM 프론티어" "2026-08-03(2차)" 소절,
`results/s8_frontier/DESIGN.md` §4.3.10–4.3.12]. 이전 rev4: ★★★§1-26
신설[같은 날 속행: 메인 세션이 세운
"decode 실현 4–19%가 g를 attenuate했다"는 보정 가설을 claims-auditor가 REFUTED —
control-arm reductio(T8에 같은 보정 적용 시 corrected g 21–29×로 C2를 10배
위반)·de-engagement 직접 실험(w=0에서도 g 거의 불변)·"A(108) 셀 무관" 가정의
실측 위반(UNSPLIT-only 부분집합만으로 T8 헤드라인 재현) 3중. 동시에 872077의
NO VERDICT 사유가 "CI 폭 부족"에서 **"estimand 미식별"**로 확장됨 —
`initialize_stream_groups`가 마지막 (0,108) 무분할 그룹을 항상 덧붙이는 기판에서는
"decode가 D SM에서 돌았다"⟺"prefill이 동시에 실행 중이었다"가 같은 사건이라 이
격자의 어떤 통계도 decode-SM 탄력도와 prefill 간섭을 분리 못 함(§1-24와 결합).
`g = A_free(d16)/A_free(d54)`는 **이 격자 한정 은퇴**(sticky partition 기판
수정 전까지 인용 금지), 블록 증설 재실행은 **선행 금지**. `A_free` 추정량
자체도 결함 확인(blocking 필터가 prefill 작업의 74–77%를 통과시켜 §1-24 stall이
d16–d54까지 오염 범위 확장 + 극단 percentile 퇴화) + arm 비교의 decode-batch-size
미제거 교락. §3에 항목15(항등식에서 파생된 양을 자유 모수처럼 나누지 마라) 신설.
상세는 `../PROJECT_STATUS.md` "8B decode-SM 프론티어"·`results/s8_frontier/
DESIGN.md` §4.3.9]. 이전 rev3: ★★§1-25 신설[pin 게이트가 항등식이었고, decode 축은 라벨이 4–19%만 실현 — claims-auditor 회부 + 77,688 스냅샷 독립 재현]. 이전 rev2: ★§1-24 신설[M4: ITL 꼬리 = monolithic prefill, 구조적] + §1-23 따름정리 정정[`--max-mamba-cache-size`는 공통 절대상수가 아니라 `= cap` 규칙]. 2026-08-01 실험 4건은 claims-auditor 회부 완료 — 2 REFUTED / 2 NOT-YET-SUPPORTED / 1 CONFIRMED, 상세·철회 목록은 `../PROJECT_STATUS.md` 해당 절의 supersession 박스. 이전 rev1: 진행 상태 갱신만, 결론 개정 아님 — §1에 항목23
"`--max-running-requests`는 arm 계열마다 다른 손잡이 · `kv_mamba_occupancy=1.0`
은 항등식" 신설(**소스 읽기로 검증되는 코드 사실**, 성능 판정 아님) + §3에
항목14 "항등식을 증거로 쓰지 마라 — 이 양이 재려는 것과 논리적으로 독립인가"
신설(13번과 같은 뿌리, 실패 모드는 다름). 2026-08-01 실행된 E1 전제 실험
4건(jobs 870295/870296/870297 용량 스캔, 870301 batch-cap)은 **전부
claims-auditor 미통과 = 이 문서에 결론으로 올리지 않는다** — 상태 기록은
`../PROJECT_STATUS.md` "열린 긴장"의 "2026-08-01 실험 4건" 소절에만 있고
**인용 금지**다. §1-4 얽힘의 KV 갈래는 등급 불변(occupancy 데이터는 생겼으나
de-confound 안 됨 — `../PROJECT_STATUS.md` "확정된 결과" KV 항목).
이전: 2026-07-29 (진행 상태 갱신만, 결론 개정 아님 — §3에 항목13
"집계 단위를 먼저 정하고 추정 대상과 맞는지 논증하라" 신설(9·10번을 특수
사례로 흡수) + §1에 항목22 "green-context 분할은 decode가 비면 무분할로
auto-revert — 셀 라벨은 목표이지 실현 배분 아님" 신설(관측 사실, 성능 판정
아님). `results/s8p_prefill/`(prefill 축 SM 민감도) 완료·claims-auditor 미통과
(정본 인용 금지 유지), `results/s8_frontier/`(E1) 하네스 구축 완료·본 스윕
미실행. 상세는 `../PROJECT_STATUS.md`). 이전: 2026-07-28 (★★★claims-auditor가
`workspace/engine-port/results/
s0_deconfound/DESIGN.md` §5 사전등록 게이트를 집행 — **부분 GO**. **C1
CONFIRMED**: §1-21이 인용한 Stage 0 D108 무경합 앵커는 코드 버그로 실제로는
decode 16 SM이었음을 3중 독립 증거로 확인 → §1-21 판정2(NULL)·판정3(게이트
non-binding)을 **철회**, 판정1(CONFOUNDED)만 생존, §5-6 long-ctx open item을
"게이트 실패로 보류"에서 **"게이트 미실행"**으로 복원. §3-9(방법론 교훈)는
정정이 아니라 **재작성**(무경합 앵커·음성대조 모두 고장났던 사실을 반영). **C2
CONFIRMED(scoped)**: prefill 16 SM 고정 시 decode ITL SM16→SM92 2.36–2.91×,
4 arm 모델-무관(8B 측정 노트, `../PROJECT_STATUS.md` "8B decode-SM 민감도"
절). **C2b("hybrid 급락=Zamba2 성질") NOT-YET-SUPPORTED로 강등.** §1·§1-5·
§1-7·HE0/HE2는 **철회하지 않는다** — 긴장 2건(HE2 vs C2, r0c 부분 복권)을
열린 항목으로 기록. 상세는 아래 §1-21·§3-9·§5-6, `../PROJECT_STATUS.md`).
이전: 2026-07-26 (★★**Stage 0(long-ctx L−2 게이트)**: 운영점 decode
SM-무감각을 hybrid·pure-Transformer·pure-Mamba·ctx≤16k 전부로 확장 확인 —
non-binding, long-ctx 충돌 가설 이 regime서 붕괴, HE0/벡터1 ctx-무관으로 강화 —
★★★2026-07-28 이 판정의 핵심 근거가 철회됨, 위 참조. 상세 §1-21). 2026-07-19
(★★**§1-16 반증 — tight SLO로 컨트롤러를 실제 재튜닝하면 동적은 best-static에 크게 열위(§1-17). SLO 엄격도와 무관하게 decode-heavy static 지배 확정.** 2026-07-18 §1-16의 "tight→동적 우위"는 재스코어 아티팩트로 격하). 2026-07-17 (변화-trace n≥4 — HE0 견고 확정 + 게이트=auto-tuner 규명).
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

| 21 | ★★**Stage 0(long-ctx L−2 게이트, 2026-07-26) — ★★★2026-07-28 판정2/판정3 철회(C1 CONFIRMED), 판정1만 생존** | 3-arm coupled-운영점 스윕(M=pure Mamba2-2.7B 음성대조·H=Zamba2-2.7B hybrid·T=Qwen2.5-3B 양성대조, ctx{4k,8k,16k}, decode-SM{16,44,92}+108-ref, jobs 864230+864601, PIN_CHECK 전부 PASS). **판정1(CONFOUNDED, CONFIRMED, 생존)**: raw ITL(D16/D44/D92) 곡선은 decode-SM binding이 아니라 prefill 경합/entanglement 아티팩트 — 이는 prefill=108−D가 항상 공변하는 설계상 사실이라 D108 앵커의 유효성과 무관하게 참이다. 원 **판정2(NULL, CONFIRMED)**: 유일 de-confounded 대조 D16 vs D108 = 1.00±0.01, 3 arm×3 ctx 전부 ⇒ 운영점 decode는 16→108 SM에 무감각. 원 **판정3**: long-ctx 충돌 가설 붕괴, HE0/벡터1이 ctx-무관으로 강화. ★★★**반증(2026-07-28, claims-auditor 사전등록 게이트 집행, C1 CONFIRMED)** — 판정2·판정3 철회: "D108(무경합 앵커)"은 **실제로는 decode 16 SM**이었다. 3중 독립 증거: (i) 코드 기전 — `manual_divisions=[92,16,0]`의 세 번째 값 0이 legacy auto-path threshold로 읽혀 `decode_bs>=0`이 항상 참 → 항상 stream_idx 1=(92,16) 선택(`src/multiplex/multiplexing_mixin.py:725-742`); (ii) realized telemetry 재집계 — decode-active 샘플의 79–96%가 (92,16)(9/9 셀); (iii) telemetry와 독립인 클라이언트 서명 — D108/D16=0.992–1.001(9/9 셀)인데 D92는 3.4–3.6× 빠름(108이 92보다 느릴 수 없음). ⇒ "D16 vs D108=1.00±0.01"은 **동일 조건 반복측정**. ★**"3중 삼각검증" 표현도 철회** — 무경합 앵커는 고장, 음성 대조 M의 전제("decode O(1) recurrent라 SM-bound 불가")도 틀렸음이 확인됨(context 길이의 O(1)이지 SM 수의 O(1)이 아니었다 — `../PROJECT_STATUS.md` "8B decode-SM 민감도" C2 참조), de-batch 논거는 미감사 — 1/3만 남는다. D16/D44/D92의 **pin 자체**는 realized 기준 유효함 유지. **HE0/HE2/§1-5/§1-7은 철회하지 않는다** — 대신 §5-6이 "게이트 미실행"으로 복원되고, 열린 긴장 2건(HE2 vs C2, r0c 부분 복권)이 `../PROJECT_STATUS.md`에 기록된다. ★scope 한정(필수, 판정1엔 여전히 적용): {M/H/T 2.7–3B, triton, cudagraph-ON green-context pdmux, ctx≤16k, coupled 하네스, one-shot 32-conc burst}. 상세 [`stage0_verdict_2026-07-26.md`](stage0_verdict_2026-07-26.md)(원 판정, 위 항목들로 철회됨), `../workspace/engine-port/results/s0_deconfound/PARTITION_RESIDENCY_STAGE0.md`(C1 근거) |

| 22 | **green-context 분할은 decode가 비면 무분할로 auto-revert한다 — 관측 사실, 성능 판정 아님(2026-07-29)** | 코드: `multiplexing_mixin.py:726,745-748`. 결과적으로 **셀 라벨 `[P,D]`는 목표(target)이지 실현(realized) 배분이 아니다**. 실측(`results/s8_frontier/` job 866066, T8, 시간가중 직접 집계): 목표 `[92,16]`(=d16) 셀은 prefill-active 시간의 **85%만** target `(92,16)`에서 돌고 **15%는 무분할 `(108,0)`**에서 돌았다(2.030s 중 0.311s); 목표 `[16,92]`(=d92) 셀은 **100%** target에서 돌았다(47.144s 중 47.091s, 무분할 0.053s=0%). 이 비대칭은 **prefill이 빠른 셀일수록 크다**(같은 뿌리에서 셀별 동시성도 갈린다: 시간가중 `concurrent_time_frac` d16 ~1.3% / d44 4.9% / d92 25–40%, prefill에 SM을 많이 줄수록 prefill이 빨리 끝나 decode와 덜 겹친다). ⇒ **파티션 스윕 결과는 목표 배분이 아니라 실현 배분의 시간가중 분포와 함께 보고해야 한다**(§3-11의 활성률 게이트와 결합). Stage 0(§1-21)의 D108 앵커 실패와 **같은 구조**(라벨 vs 실현)이나 **원인은 다르다** — 그건 legacy auto-path의 threshold 오독이라는 설정 버그, 이건 **정책이 설계대로 동작한 결과**(decode-empty 시 무분할 fallback은 의도된 경로) |

| 23 | ★★**(2026-08-02, 코드 사실) `--max-running-requests`는 arm 계열마다 다른 손잡이이고, `kv_mamba_occupancy=1.0`은 항등식이다 — 성능 판정 아님** | 코드: `sglang/srt/model_executor/model_runner_kv_cache_mixin.py:223-229`. `disable_radix_cache ∧ max_running_requests is not None`이면 **`max_mamba_cache_size = max_running_requests`**(앞 분기 `:218-222`는 `--max-mamba-cache-size` 명시 시, 뒤 `else`는 가용 메모리 ratio 기반 — 이 분기는 s8/E1 계열에서 한 번도 타지 않았다). E1/s8 캠페인이 정확히 그 조건이므로 **SSM 포함 arm(M8/Ha8/Hs8)에서 cap은 admission + mamba state pool 크기를 동시에** 움직이고, **T8(순수 Transformer)에서는 admission만** 움직인다. ⇒ (i) **T8에서 잰 cap 효과는 hybrid로 이전 불가**, (ii) cap을 실험 파라미터로 쓰려면 `--max-mamba-cache-size`를 명시 고정해 두 축을 분리해야 한다. ★**정정(2026-08-02, M5 재설계)**: 그 고정은 **전 arm 공통 절대상수가 아니라 `= cap` 규칙**이어야 한다 — slot당 SSM state 비용이 arm마다 달라(M8 0.255 / Ha8 0.141 / Hs8 0.096 GB) 공통 절대상수는 arm마다 다른 메모리 분할을 강제하는 **새 cross-arm 교락**이 된다(특히 Ha8의 attention KV pool은 이미 48×ctx의 52%만 잡혀 있어 mamba pool 증가분이 곧장 거기서 나온다). 명시 고정은 `:218` 분기를 타므로 항등식을 깨는 목적은 `= cap`으로도 완전히 달성된다. **따름정리**: pool 크기 = cap이므로 batch가 cap에 닿으면 `kv_mamba_occupancy`는 **정의상 1.0** — 이 값을 "hybrid는 메모리가 구속한다"의 근거로 쓸 수 없다(§3-14의 사례 2). §1-4 얽힘의 KV 갈래 등급은 **불변**: 이 regime(ctx 4k ShareGPT)에서 지지되는 것은 좁게 "attention KV(`kv_full_occupancy`)가 어느 arm에서도 구속 근처에 없었다"뿐이며 그 관측치 자체는 **claims-auditor 미통과(인용 금지)**다(`../PROJECT_STATUS.md` "확정된 결과" KV 항목·"방법론 게이트" #5) |
| 24 | ★★**(2026-08-02, M4 — GPU 0) 이 기판의 ITL 꼬리는 decode step time이 아니라 *monolithic prefill이 decode를 멈춘 시간*이 지배한다 — 그리고 그건 구조적이다** | 절대 토큰-방출 시각 재구성(arrival replay + TTFT + 누적 ITL) 결과, rate 2에서 stall probe **17개 중 16개**에서 어떤 요청이 **stall 전 구간 동안 prefill 중**이었고 그 요청은 거의 항상 그 probe의 **최장 프롬프트**(2469–2776 tok)였다. 크기는 고정 프롬프트에서 D에 **단조**(Ha8 seed1: d24 167.7 → d44 225.6 → d54 263.6 → **d92 865.6 ms**) — prefill SM = 108−D 가 줄기 때문. ⇒ **conjunctive goodput의 ITL 항은 decode-SM 레버와 반대 부호로 움직이는 항을 내장하고 있다.** ★**고칠 수 없다**: `server_args.py:6130`이 `enable_pdmux`일 때 `chunked_prefill_size == -1`을 **하드 assert**한다("PD-Multiplexing is not compatible with chunked prefill") ⇒ un-chunked prefill은 연구 대상 기판의 **전제**이지 설정 실수가 아니며, `venue_positioning.md` §0.1의 **(A) green-context 종속** 버킷에 속한다(DuetServe가 libsmctrl로 우회한 바로 그 비용 계열). **따름정리**: ITL 보고는 `A_all`(등록 SLO 항)과 `A_free`(≥1024 tok prefill 창과 겹치는 ITL 제외)를 **병기**한다. `A_free`는 d92에서는 여전히 오염(prefill 16 SM이라 짧은 프롬프트도 막음)이라 **d16–d54에서만 해석**한다. 사전등록 = `results/s8_frontier/DESIGN.md` §4.3.8(a). ⚠️2026-08-01 핸드오프의 기전 추측이 **옳았고** 2026-08-02 감사자의 REFUTED는 **틀린 검정**을 썼다 — prefill 중인 요청은 아직 decode를 안 하므로 자기 stall을 볼 수 없고, 그 요청의 max ITL이 작은 것은 가설과 모순이 아니다 |
| 25 | ★★★**(2026-08-03, claims-auditor + 독립 재현) decode 축은 라벨이 4–19%만 실현된다 — 그리고 기존 pin 게이트는 항등식이었다** | **(a) pin 게이트 = 항등식.** telemetry 120파일·prefill-active 스냅샷 **77,688개**에서 `prefill_sms ≠ target ⟺ decode_running_batch_size == 0`이 **양방향 위반 0건**(off-target∧decode-empty 50,328 / on-target∧decode-busy 27,360). `multiplex/multiplexing_mixin.py:773,792-794`가 decode batch가 비면 **설계대로** 무분할로 떨어뜨리므로(§1-22가 이미 '의도된 경로'로 기록), `e1_pin_check.py`의 시간가중 pin 게이트는 파티션 제어가 아니라 **'prefill in-flight 중 decode가 안 비어 있던 시간 몫'**을 잰다 ⇒ **방법론 게이트 #6 위반**(정확성을 강제하려고 만든 게이트가 저질렀다). 따름정리로 **조건부 pin = 27,360/27,360 = 1.000 정확** — **파티션은 질문이 성립하는 곳에서 완전 실현**(job 872077 재채점 64/64 PASS). **(b) 정작 게이트가 없던 축 = decode.** decode-active 시간 중 `decode_sms == D` 비율(872077, n=8/셀): T8 d16 **0.038**/d24 0.047/d44 0.082/d54 0.093, Ha8 d16 0.104/d24 0.110/d44 0.148/d54 0.187 ⇒ **셀 라벨의 decode 분할은 decode 작업시간의 4–19%만 실현되고 81–96%는 무분할 108 SM**. ⇒ (i) E1 격자는 **지속적 decode-SM 배분을 주지 않으므로** C2가 잰 물리량(prefill 16 고정·decode 연속 D)과 **다른 양**이다 — 긴장 A를 이 격자의 비(比)로 닫을 수 없다; (ii) 희석 계수가 **셀마다 다르다**(T8 0.038→0.093, Ha8 0.104→0.187) ⇒ `A_free(d16)/A_free(d54)`는 SM 수준과 **분할 engagement 비율을 동시에 움직인다 = 추정량 내부 교락**. **Stage 0 D108(라벨≠실현, §1-21)의 decode 축 판본**이며, §1-22가 요구한 실현분포 보고가 prefill 축에만 적용돼 있었다. 신규 게이트 `E1_COND_PIN`·`E1_DECODE_REALIZED`(`e1_pin_check.py`, 2026-08-03)로 코드화, 상세 `results/s8_frontier/DESIGN.md` §4.3.8(h) |

| 26 | ★★★**(2026-08-03, claims-auditor, 같은 날 §1-25 속행) `g`는 이 격자에서 은퇴한다 — 희석-보정 가설 REFUTED, NO VERDICT 사유가 "estimand 미식별"로 확장, `A_free` 자체가 결함, arm 비교에 미제거 교락** | **(A) 희석 attenuation 가설 REFUTED.** 메인 세션이 세운 모형("`E1_DECODE_REALIZED` 4–19% ⇒ `A_free(dD)=w_D·A(D)+(1−w_D)·A(108)` 혼합, 보정 시 Ha8 g≈1.62–1.70")을 3중으로 반증: (i) **control-arm reductio** — 같은 보정식을 T8에 적용하면 corrected g **21–29×**(C2의 2.36–2.91×를 10배 위반, 요구 ITL p95 352–360ms 대 실측 30.67ms); (ii) **de-engagement 직접 실험**(같은 셀 unsplit 분포에서 engagement를 낮춤) — `A_free` 변화 **1–11%뿐**, w=0에서도 g 거의 그대로(Ha8 1.173/T8 1.796); (iii) 핵심 가정 "`A(108)` 셀 무관"이 실측 위반 — Ha8 SPLIT-only 0.920[0.842,0.998](CI가 1 배제, 부호 반대), **T8 헤드라인은 UNSPLIT-only(decode SM 대비가 정의상 0인 부분집합)에서 그대로 재현**(1.795[1.589,2.001] ≈ ALL 1.837). 죽은 것은 보정이지 §1-25가 확립한 "engagement가 낮다"는 전제 자체가 아니다(세 계측기 교차확인으로 견고). **(B) 872077 NO VERDICT 사유 확장.** 코드 사실: `pdmux_context.py:initialize_stream_groups`가 `SM_COUNTS=[(108,0)]+divisions+[(0,108)]`를 하드코딩하고 `multiplexing_mixin.py:773,792-794`가 prefill 비-in-flight 시 무조건 `(0,108)`로 되돌린다 — 즉 이 기판에서 **"decode가 D SM에서 돌았다"⟺"prefill이 동시에 in-flight였다"는 같은 사건**이다. §1-24(ITL 꼬리=monolithic prefill, 크기가 108−D에 단조)와 결합하면 `g`는 사전에 **"decode-SM 탄력도 라벨을 단 prefill-SM 탄력도"**일 것이 예상되고, 실측이 그와 일치(UNSPLIT-only에서 T8 헤드라인 재현)한다 — **n으로 해결되지 않는 설계 결함**. `g = A_free(d16)/A_free(d54)`는 **이 격자 한정 은퇴**(sticky-partition 기판 수정 전 인용 금지), 블록 8→12–16 증설 재실행은 **선행 금지**. ⚠️"Ha8에 decode-SM 레버가 없다"는 CONFIRMED 아님 — 현 데이터는 그 질문에 답하지 못한다, 긴장 A(HE2 vs C2)는 전혀 닫히지 않았다. **(C) `A_free` 추정량 자체의 결함.** `e1_m3_control.sbatch:281-306`: `PREFILL_BLOCK_TOK=1024` 필터가 이 워크로드 요청의 4–5%만 걸러 prefill 작업의 **74–77%가 필터를 통과**(제거되는 ITL은 전체의 ~2%뿐) ⇒ §1-24가 확정한 monolithic-prefill stall이 **d16–d54까지 오염 범위 확장**(§1-24 원문의 "d92만 오염" 한정을 넓힘). 요청의 27.5–29.5%가 output≤25 토큰이라 내부 p95가 사실상 max ITL로 퇴화 — 평균의 선형 혼합 항등식이 극단 분위수에 성립하지 않음(비단조 응답으로 실증). `A_free`는 blocking-제거본이 아니라 대부분 monolithic-prefill stall로 이루어진 극단꼬리 통계다. **(D) arm 비교의 미제거 교락.** 공통 rate 2에서 T8 conc 12.8/decode batch 4.5/ITL p50 ~11ms 대 Ha8 conc 30.6/batch 15.8/~30ms — 양 arm 모두 off-cliff 평탄역(metric cliff 아님)이나 decode batch size가 memory/compute-bound 여부를 결정하는 공변량이라 arm과 완전 교락 ⇒ "attributable to the arm" 문구는 **현재 허용 안 됨**(통제는 realized concurrency/decode batch를 맞춘 rate여야 함); 이 교락은 관측 *방향*을 설명하지 않으므로(batch 큰 쪽이 오히려 무반응) 대안 설명이 아니라 **미제거 교락**으로만 기록. 부수: d16은 `sm_group_num:3`, d54는 4(guard row)로 셀마다 green context 수가 다르고 d54의 `decode_sms==44`는 telemetry에 미관측(guard row 미선택) — 행동 교락 아니나 셀 간 차이. `DESIGN.md` §4.3.8(h)의 "A re-run under RULE_BOUNDS would settle it"은 **이제 틀렸다** — sticky partition 구현 전 재실행은 같은 estimand-미식별 문제를 반복한다. 다음 gate: `PDMUX_STICKY_PARTITION` 구현(engine-porter, correctness gate) → sticky 격자 1회(872077 대조 8 block) → 사전등록 판별 예측(T8≈1.85·Ha8≈0.92 vs Ha8≈1.6, CI 비중첩) → `E1_DECODE_REALIZED≥0.90`이 sticky에서는 항등식이 아닌 **진짜 게이트**. 상세 `results/s8_frontier/DESIGN.md` §4.3.9. **부수(UNAUDITED, 인용 금지 — result-analyst 산출, claims-auditor 미통과)**: `m3_decode_empty.py` 분석이 §1-25의 희석이 decode-empty가 아니라 **prefill 부재**(decode-active 시간에 조건부라 decode-empty는 정의상 분자·분모에 안 들어감)로 나온다고 보고하며, 죽은 telemetry 필드 4개(`decode_ready_queue_depth`·`active_decode_sequences`·`decode_idle_ratio`/`prefill_idle_ratio`)를 코드 근거로 식별한다 — **claims-auditor 미통과이나 부하를 올려 engagement를 높이는 방향이 지지되지 않는다는 결론만은 감사자와 독립적으로 수렴해 그 좁은 항목만 AUDITED로 인용 가능**(rate를 올리면 prefill·decode가 비례해 늘고 split-eligible iteration 수는 셀 무관 194–214로 거의 일정) |
| 27 | ★★**(2026-08-03, 같은 날 2차 속행) `A_free` 대체 추정량으로 estimand 이관 완료[AUDITED, (3)만 예외] + `PDMUX_STICKY_PARTITION` 구현·correctness gate 통과[구현 사실, 성능 판정 아님]** | **(I) 조건부 per-token 추정량**(`m3_conditional.py`, §1-26의 `A_free` 결함을 대체). 단위 = 개별 ITL 구간 1개(요청별 집계 없음), 라벨 `split_frac(a,b)≥0.90`(SPLIT)/`≤0.10`(UNSPLIT)/사이는 양쪽에서 배제(AMBIGUOUS 0.3–1.4%), 통계량은 셀-블록별 **직접 분위수**(primary `p95(SPLIT)`, control `p95/p50(UNSPLIT)` — 두 셀 모두 108 SM이라 대비가 정의상 0). **[AUDITED]**: T8 d16 pooled per-token p95 **11.60**(`A_free`는 28.31, 2배 이상 바깥 꼬리 — 요청의 27.5–29.5%가 outlen≤25라 `A_free`가 사실상 max ITL로 퇴화, `A_free`는 요청 ~9.6개에 얹히는 반면 새 추정량은 셀-블록당 1,390–7,045 토큰). client↔telemetry 시계정렬은 `phase=="benchmark"` 필터 **금지**(그 마커는 warm-up 요청에 발화, 실제 probe 시작이 아님); `ALIGN_R_MIN=0.95` 미달은 flag-only(조용히 배제 금지) — 배제하면 오히려 대비가 커짐(LOO 실측, T8 1.688→1.822). **[UNAUDITED — 감사자가 이번 턴에 새로 생산해 자기 산출을 자기가 감사한 형태, 별도 확증 전 인용 금지]**: `PREFILL_BLOCK_TOK` 스윕(1024→512→256→0)에 **무릎이 없고** 임계 0에서 d16-vs-d54 대비가 두 arm 모두 소멸(0.986/1.005) — 임계는 자유 모수가 아니라 답을 정하는 손잡이. 권고: **`A_free` 은퇴**, `A_all` 유지 병기, **primary 라벨 = realized partition**(`decode_sms==D`, persisted state variable이라 견고); secondary(prefill overlap) 라벨은 872077이 `PDMUX_TRACE_FORCE_PREFILL=0`이라 **계측 결손**(SPLIT 라벨 토큰 중 "overlap-free"로 나오는 비율 Ha8 66.3/38.8%·T8 76.1/27.1% — 물리적으로 불가능해야 할 값, 곧 계측 과소표집의 증거). **(II) `PDMUX_STICKY_PARTITION`**(`multiplexing_mixin.py` +151/−17): decode-busy 시 무분할 fallback을 우회(`if not running_batch.is_empty() and (split_prefill_batch or sticky_partition_enabled)`), **decode-empty 시엔 의도적으로 index 0 release**(hold 아님 — 보호할 decode 작업이 없고 `E1_DECODE_REALIZED`가 decode-active 가중이라 가중치 0). OFF는 short-circuit으로 **패치 전과 byte-identical**(독립 재구현 pre-patch selector와 전 격자 동등성 테스트로 확인). cudagraph 보존(스트림별 캡처 유지, eager fallback 없음). `PDMUX_LA_COORD`/`PDMUX_SLO_SCHED`/`PDMUX_FIXED_DECODE_SM_FILE`/비-`fixed` `PDMUX_R2_POLICY`와의 조합은 init `RuntimeError`로 거부(반쪽 sticky 방지). **correctness gate 전부 PASS**: CPU 회귀(40 tests, sync manifest SHA-256 일치) + sticky 단위 테스트 12건 + **GPU smoke(job 872800, Ha8 d16)**: 고정 프롬프트 6개 greedy 출력이 OFF/ON **byte-identical**. **realized 관측(게이트 아님, n=1)**: sticky OFF `E1_DECODE_REALIZED=0.0839`(기존 동작 재현) vs **ON=1.0000**(사전등록 ≥0.90 초과, 튜닝한 것 없음) — decode-busy 스냅샷이 ON에서 `(idx1,92,16)` 139/139, OFF에서 무분할 진입 146회. **구현 완료 ≠ 성능 주장 성립** — throughput/latency/goodput/`g` 그 무엇도 주장되지 않는다. **(III) sticky 런 사전등록**: 872077 소급 재분석은 **DIAGNOSTIC 전용**(re-score 금지, 설계 판정의 근거로만), primary 통계량 1개(`p95(SPLIT)` 비) 선언, 게이트 4종(`ALIGN_R_MIN`·`E1_DECODE_REALIZED≥0.90`·`AMBIG_FRAC` 상한·`MIN_N_SPLIT` 하한), **★`G_LEVER`/`G_FLAT`는 미결정으로 기록**(기존 1.5/1.15는 `A_free` 스케일이라 그대로 이전 불가 — C2 측정범위(2.36–2.91×)에 묶는 안이 논거는 있으나 E1은 D 범위가 좁고 상보적(P+D=108)이라 그대로 못 씀, sticky 런 제출 전 별도 사전등록 필요), sticky 후 primary 모집단이 `SPLIT ∧ BLOCK-FREE`로 이동함을 미리 등록, trace-force ON 재허용 여부는 **열린 설계 쟁점**으로 미결정 기록. 판별 예측(§4.3.9 승계) 불변: prefill 주도면 T8≈1.85·Ha8≈0.92, 희석 가설이 옳았다면 Ha8≈1.6(CI 비중첩, 8 block으로 구분 가능). 상세 `results/s8_frontier/DESIGN.md` §4.3.10–4.3.12 |

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
   구분 못 한다. 이 결론 자체는 유효하다(§1-21 판정1, 생존). ★★★**재작성
   (2026-07-28, claims-auditor 감사 후) — 아래 문장은 정정이 아니라 대체다.**
   원래 여기 있던 "음성 대조 + 무경합 앵커의 조합이 confound를 identify했다"는
   서술은 **거짓이었다** — Stage 0에서 그 두 기구는 **둘 다 고장 나 있었다**:
   "무경합 앵커"(D108)는 실제로는 decode 16 SM이었고(코드 버그로 legacy
   auto-path가 항상 (92,16) 선택), "음성 대조"(pure-Mamba M)의 전제 자체가
   틀렸다("decode는 O(1) recurrent라 SM-bound 불가"의 O(1)은 context 길이에
   대한 것이지 SM 수에 대한 것이 아니었다 — 8B 재측정에서 M도 T·H와 동일 밴드로
   SM-민감했다). 이 확산에서 얻는 교훈 3개로 대체한다:
   - **pin은 policy target이 아니라 realized 파티션으로 검증한다**
     (`runtime_snapshot`의 `(prefill_sms, decode_sms)`, `dual_worker.py:608`).
     비용 0. Stage 0은 controller가 지정한 값(target)만 확인하고 실제로
     선택된 stream_index(realized)를 확인하지 않아 D108이 D16이었음을
     놓쳤다.
   - **음성대조는 그 축에 binding 불가능함이 독립 입증된 뒤에만 음성대조다.**
     "Mamba decode는 O(1)"이라는 직관을 검증 없이 음성 대조의 자격으로 썼다가,
     그 O(1)이 다른 축(context 길이)에 대한 것이었음이 드러나며 대조 자체가
     무효화됐다.
   - **"policy OFF = 중립 기준선"은 legacy fallback 경로 때문에 조용히
     깨진다.** `PDMUX_R2_POLICY`가 unset이면 legacy `adjust_stream_groups`가
     `manual_divisions`의 threshold 필드(0)를 조건문으로 오독해 의도한 값과
     다른 파티션을 고른다 — "정책을 껐다"가 "분할을 안 했다"를 뜻하지 않는다.
   상세 [`stage0_verdict_2026-07-26.md`](stage0_verdict_2026-07-26.md) §5(원
   교훈, 철회됨), `../workspace/engine-port/results/s0_deconfound/
   PARTITION_RESIDENCY_STAGE0.md`(재작성 근거).
10. ★★**(2026-07-28) pin은 realized로 검증한다.** 위 항목 9의 첫 소항목과 동일 —
    `PROJECT_STATUS.md` "방법론 게이트(신규)" 참조. ★**아래 13번의 특수
    사례**(target-vs-realized 집계 단위 불일치)로 재분류(2026-07-29).
11. ★★**(2026-07-28) 파티션 활성률을 사전등록 게이트로 삼는다.** green-context
    분할은 split-prefill 동거 중에만 유효하고, 비면 무분할로 되돌아간다 —
    최소 활성률(예: ≥0.60)을 실험 전에 정하고 미달 셀은 폐기한다. ★**정정
    필요(2026-07-29)**: 이 활성률은 **시간가중**으로 재정의해야 한다 — 아래
    13번 참조. 스냅샷 **개수** 기반 활성률은 실제값을 최대 16–26× 과소평가할
    수 있다(`results/s8_frontier/` 실측: 동시성 0.015 → 시간가중 0.25–0.40).
12. ★★**(2026-07-28) keepalive는 짧은 prefill을 자주.** 긴 keepalive는 활성률을
    오히려 떨어뜨린다(실측 0.66–0.93 → 0.32–0.63) — 긴 prefill 윈도우 동안
    decode가 진행되지 않아 시간적으로 분리된다.
13. ★★★**(2026-07-29) 집계 단위를 먼저 정하고, 그 단위가 추정 대상과 맞는지
    논증하라.** 위 9번(coupled 스윕 confound)·10번(pin은 realized로 검증)의
    상위 개념 — 그 둘을 이 원칙의 특수 사례로 흡수하되 문구는 지우지 않는다.
    `results/s8_frontier/`(E1 프론티어) 하네스 구축 중 **집계 단위가 답을
    5번 바꿨고 전부 반대 결론을 낼 뻔했다**:
    1. **target vs realized 파티션**(Stage 0 무효화, 위 §1-21).
    2. **게이트 모집단** `prefill_active OR decode_active` vs 조건부 — 설계상
       정상인 decode-only 무분할 윈도우를 pin 실패로 셈: pin **0.029 FAIL →
       0.976 PASS**.
    3. **스냅샷 개수 가중 vs 시간 가중** — `runtime_snapshot`이 이벤트루프
       iteration당 발화해 235ms prefill 스텝과 11ms decode 스텝이 같은 무게를
       가짐: 동시성 **0.015 → 0.25–0.40**(16–26×).
    4. **drain 꼬리 포함 duration vs 도착 구간만의 duration** —
       `achieved_rps` **3.40 → arrival_rps 8.96**.
    5. **스냅샷 vs 에피소드**, 그리고 그 안에서 다시 **개수 vs 시간** —
       d16 pin_frac **0.583 → 0.847**.

    **따름정리**: 하나의 추정량으로 두 질문에 답하지 마라. "그 파티션에서
    실행됐는가"(게이트)는 **시간 가중**으로 답하고, "이 요청의 지연은 어느
    파티션 것인가"(귀속)는 **요청별 bracket**으로 답한다. 후자는 전자의
    데이터를 대부분 버리므로(대부분의 스냅샷이 요청 경계 안쪽이 아니라
    사이에 놓임) **게이트에 쓰면 검정력이 무너진다**(예: d16 요청-bracket
    귀속 n_episodes=8, lower95=0.554 → FAIL — 사유는 검정력 부족이지 잘못된
    SM 아님). 상세 `../workspace/engine-port/results/s8_frontier/` 하네스
    설계 문서(결함 5종 수정 기록).
14. ★★★**(2026-08-02) "이 양이 내가 재려는 것과 논리적으로 독립인가"를 먼저
    물어라 — 항등식을 증거로 쓰지 마라.** 13번(집계 단위 선확정)과 같은
    뿌리이나 실패 모드가 다르므로 별도 항목으로 둔다: 13번은 *같은 양을 어떻게
    세는가*의 문제이고, 이것은 *그 양이 애초에 답을 담고 있는가*의 문제다.
    2026-07-31~08-01 세션에서 철회된 주장 8건 중 **3건이 정의·항등식을 증거로
    착각**한 것이었다:
    1. `arrival_rps`는 seed로부터 RNG replay로 **재생성된** 값
       (`e1_capacity_scan.sbatch:200-207`)이라 `(seed, n)`만의 결정론적
       함수 — "5셀 전부 동일"은 **항등식**이고 서버에 대한 정보량이 0이다.
    2. `kv_mamba_occupancy = 1.0`은 pool 크기가 `--max-running-requests`와
       같아서 생기는 **항등식**(§1-23).
    3. "ITL 구속 rate ⟂ d92 off-cliff"는 ITL 구속 rate가 **공집합**이라
       **공허참**(T8은 전 구간·전 룽에서 요청의 ≥93%가 ITL 통과).
    나머지 3건은 **한 arm/셀에서 잰 것을 일반화**한 것이었다(batch-cap은 T8
    2셀만 측정). ⇒ 새 지표를 증거로 올리기 전에 **그 지표가 실험 설정으로부터
    해석적으로 결정되는 값이 아닌지** 먼저 확인한다. `../PROJECT_STATUS.md`
    "방법론 게이트" #6과 동일 항목.
15. ★★★**(2026-08-03) 항등식에서 파생된 양을 자유 모수처럼 나누지 마라 —
    14번과 같은 뿌리, 이번엔 GPU를 쓰기 전에 잡혔다.** §1-26(A). `E1_DECODE_
    REALIZED`(§1-25)는 pin 게이트와 마찬가지로 항등식 경계(prefill in-flight
    ⟺ decode busy)에서 파생된 시간 몫이다. 이걸 "decode-SM 레버가 A(D)에
    engagement 비율 `w`만큼만 반영된다"는 **자유 모수**로 취급해 `A_free(dD)
    =w·A(D)+(1−w)·A(108)`을 역산하면, `w`가 실은 §1-24(prefill=108−D)가
    이미 결정한 duty cycle이라 **역산 대상과 역산 도구가 같은 양**이 되고
    보정치는 자기충족적이다. 검정 순서가 재사용 가치: (i) 같은 보정을
    **대조군(레버가 있다고 이미 알려진 arm)에 적용**해 알려진 값을 위반하는지
    본다(control-arm reductio) — 위반하면 모형 사망; (ii) 모형이 요구하는
    잠재 관측치(여기선 split-조건부 ITL)를 **직접 계산**해 실측과 대조한다;
    (iii) 모형의 핵심 가정을 **부분집합 분해**로 직접 검정한다(여기선
    UNSPLIT-only에서 헤드라인이 그대로 재현 = "A(108) 셀 무관" 가정의 반례).
    셋 다 GPU 데이터 없이 기존 텔레메트리 재사용만으로 수행됐다.

---

## 4. 살아있는 문서 (이것만 참조)

| 문서 | 역할 |
|---|---|
| **CONSENSUS.md** (이 문서) | **정본** — 확정/철회 상태 |
| **`per_layer_type_postmortem.md`** | ★**per-layer-type 구제 시도→기각 kill-chain**(시도 A–H, 두 조건 C1 lever·C2 착취 관점) |
| **`research_arc.md`** | ★**연구 아크 전사** — 출발점→현재의 단계별 시작논의/촉발지표/해소지표/부정사유(수치 포함). CONSENSUS의 서사 짝. **§S-M = 서빙 수준 반증 지점(등급별) · 측정 환경(E1–E4) · 결론의 유효 경계** |
| **`longcontext_trace_plan.md`** | ★**계획 문서**(L−1 이상은 측정 전) — 실 trace를 long-context로 전환하는 문제. 동기(Diff A는 L≥3k서 열림) vs 정직한 반론(Diff B는 long-L서 닫힘) · 하드 블로커(Zamba2 ctx 4096 / goodput SLO 붕괴) · 단계 L0–L3. ★★★**L−2(Stage 0) 게이트는 2026-07-26 non-binding으로 "실행·확정"됐다고 기록했으나 2026-07-28 claims-auditor 감사(C1 CONFIRMED)로 무효 — 게이트는 사실상 미실행이었다**(아래 `stage0_verdict_2026-07-26.md` 참조), L−1 이상은 "게이트 실패로 보류"가 아니라 "게이트 미실행" |
| **`stage0_verdict_2026-07-26.md`** | ★★★**Stage 0(long-ctx L−2) 원 판정 — 2026-07-28 판정2/판정3 철회(C1 CONFIRMED, §1-21)**. coupled 스윕 confound 진단(판정1)만 생존, "de-confounded D16≡D108 null" 결과는 D108 앵커가 실은 decode 16 SM이었음이 확인돼 무효. 이력 보존용, 새 분석 근거로 재인용 금지(단독으로는) — 재인용 시 §1-21 전문과 병기 |
| `results/slo_sched/lengthnorm_slo_reanalysis.md` | ★**길이-정규화/tight SLO 재계측**(§1-16) — HE0가 SLO 엄격도 의존임을 기존 벤치 재분석으로 확정. 스크립트 `reanalyze_lengthnorm_slo.py` |
| **`serving_slo_survey.md`** | ★**실 서빙 SLO 관행 조사**(§1-16 후속) — 프로덕션 인터랙티브 TTFT(chat 300/voice 150/code 100/RAG 400ms)가 전부 tight regime; 우리 3s=batch async. DistServe SLO-scale sweep=표준. goodput 메트릭 비판 |
| **`interactive_slo_retune_plan.md`** | ★**tight-SLO 컨트롤러 재튜닝 + P90-attainment 직접 측정**(§1-17) — §9에 최종 결과(HT0 확정, d44≫동적). 하네스 `results/slo_sched/interactive_bench.sbatch` |
| `../../workspace/engine-port/results/s8_frontier/DESIGN.md` | ★★**E1 프론티어 하네스 설계·전사**(§4.3.1–4.3.12) — 사전등록·버그 수정·게이트 이력의 정본. §4.3.9=`g` 격자 한정 은퇴, §4.3.10=`A_free` 대체 조건부 추정량[AUDITED, blocking 스윕만 UNAUDITED], §4.3.11=`PDMUX_STICKY_PARTITION` 구현 사실, §4.3.12=sticky 런 사전등록(`G_LEVER`/`G_FLAT` 미결정) |
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
   ★★**Stage 0(L−2) 게이트 실행 — non-binding, 2026-07-26 (★★★2026-07-28
   철회, C1 CONFIRMED)**(§1-21, [`stage0_verdict_2026-07-26.md`](stage0_verdict_2026-07-26.md)):
   2026-07-26엔 이 항목이 전제하는 "decode floor가 ctx로 상승해 운영점서
   binding해진다"가 **ctx≤16k에서는 반증**됐다고 봤으나(D16≡D108(1.00±0.01)),
   근거였던 D108 앵커가 실은 decode 16 SM이었음이 claims-auditor 감사(C1
   CONFIRMED, §1-21)로 확인돼 이 판정을 **철회**한다. **L−2 게이트는 사실상
   아무것도 측정하지 않았다** — 따라서 `longcontext_trace_plan.md` §6의 게이트
   규칙이 예정한 "L−1 이상은 진행 근거 없어 멈춘다"는 **"게이트 실패"가 아니라
   "게이트 미실행"**으로 정정한다(재개 권고 아님, 판정 부재라는 뜻). 이 항목
   (long-context 실 trace 전환)은 **판정 이전 상태로 되돌아가 여전히 열려
   있다** — 남은 미측정은 ctx≤16k을 포함한 전 구간(L−2 재시도부터).
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
   decoupling. ★**2026-07-26 갱신, ★★★2026-07-28 철회**: 2026-07-26엔 전자가
   Stage 0(L−2) 게이트로 부분 실행돼 "ctx≤16k에서는 decode floor가 운영점서
   상승하지 않아(§1-21) 이 항목도 ctx-무관으로 강화되는 방향"이라고 봤으나,
   그 근거(D108 무경합 앵커)가 claims-auditor 감사(C1 CONFIRMED, §1-21)로
   무효 확인돼 **철회**한다. **이 항목은 다시 미검증**(Stage 0이 "게이트
   실패"가 아니라 "게이트 미실행"이었으므로 강화도 약화도 아니고 원점).
   §1-20 spatial decoupling은 여전히 미실행. 상세
   [`../workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md`](../workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md),
   [`stage0_verdict_2026-07-26.md`](stage0_verdict_2026-07-26.md).
