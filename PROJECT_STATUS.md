# `prefill-layer-alloc` project status

최종 갱신: 2026-08-03 (★★같은 날 2차 속행, doc-steward 기록 — 두 갈래 완료.
**(I) claims-auditor의 추정량 이관**: 아래 1차 속행이 확인한 `A_free`
결함을 대체하는 **조건부 per-token 추정량**(`results/s8_frontier/
m3_conditional.py`) — 단위는 개별 ITL 구간 1개, SPLIT(`split_frac≥0.90`)/
UNSPLIT(`≤0.10`) 라벨(사이는 배제), primary `p95(SPLIT)` 비 + UNSPLIT
control(대비가 정의상 0). **[AUDITED]**: `A_free`가 요청 ~9.6개에 얹히는
극단꼬리 통계였음을 확인(T8 d16 pooled p95 11.60 vs `A_free` 28.31),
client↔telemetry 정렬은 `phase=="benchmark"` 필터 **금지**(그 마커는
warm-up 요청 발화, probe 경계 아님) 규율 확정, `ALIGN_R_MIN=0.95`
flag-only(배제하면 대비가 오히려 커짐, LOO 실측). **[UNAUDITED — 감사자가
자기 산출을 자기가 감사, 별도 확증 전 인용 금지]**: `PREFILL_BLOCK_TOK`
임계 스윕(1024→0)에 무릎 없음·임계 0에서 대비 소멸(임계는 자유 모수가
아니라 답을 정하는 손잡이). 권고 = `A_free` 은퇴, primary 라벨 = realized
partition(`decode_sms==D`). **(II) engine-porter의 `PDMUX_STICKY_PARTITION`
구현 완료 + correctness gate 통과**[구현 사실, 성능 판정 아님]:
decode-busy 시 무분할 fallback 우회, decode-empty 시엔 의도적 index-0
release(hold 아님), OFF는 short-circuit으로 patch 전과 byte-identical(독립
재구현 pre-patch selector 대비 전 격자 동등성 테스트), cudagraph 보존.
**correctness gate 전부 PASS**: CPU 회귀 40 tests + sticky 단위 테스트
12건 + **GPU smoke(job 872800, Ha8 d16)** 고정 프롬프트 6개 greedy 출력
OFF/ON byte-identical. **realized 관측(n=1, 성능 아님)**:
`E1_DECODE_REALIZED` sticky OFF 0.0839(기존 동작 재현) → **ON 1.0000**
(사전등록 ≥0.90 초과, 튜닝 없음). **구현 완료 ≠ 성능 주장 성립.** **(III)**
sticky 런 사전등록 기록 — 872077 소급 재분석은 **DIAGNOSTIC 전용**(재분석
사후 채택 금지), primary 1개(`p95(SPLIT)` 비) 선언, 게이트 4종, **`G_LEVER`/
`G_FLAT`는 미결정으로 기록**(스케일 불일치로 기존 1.5/1.15 그대로 이전
불가). 상세는 아래 "8B decode-SM 프론티어" "2026-08-03(2차)" 소절,
`reports/CONSENSUS.md` §1-27, `results/s8_frontier/DESIGN.md`
§4.3.10–4.3.12. 이전(같은 날 1차 속행 — 2026-08-03 이른 회차가 세운
"pin 게이트=항등식·decode 실현 4–19%"(§1-25급) 위에서, 메인 세션이 그
희석을 `g = A_free(d16)/A_free(d54)`의 **보정 모형**으로 확장했다가
claims-auditor가 **REFUTED**시켰다 — control-arm reductio(T8에 같은 보정
적용 시 corrected g 21–29×로 C2를 10배 위반)·de-engagement 직접 실험
(w=0에서도 g 거의 불변, 1–11%만 이동)·"A(108) 셀 무관" 가정의 실측 위반
(UNSPLIT-only 부분집합만으로 T8 헤드라인이 그대로 재현) 3중. 동시에
job **872077**의 NO VERDICT 사유가 "CI 폭 부족"에서 **"estimand
미식별"**로 확장됨 — `initialize_stream_groups`가 마지막 무분할 그룹을
항상 덧붙이는 이 기판에서는 "decode가 D SM에서 돌았다"와 "prefill이
동시에 in-flight였다"가 같은 사건이라, 어떤 통계도 decode-SM 탄력도와
prefill 간섭을 분리 못 한다(§1-24와 결합, n으로 해결 안 됨). **`g`는 이
격자 한정 은퇴**(sticky-partition 기판 수정 전 인용 금지), **블록 증설
재실행은 선행 금지**. `A_free` 추정량 자체도 결함(blocking 필터가 prefill
작업의 74–77%를 통과시켜 stall 오염이 d16–d54까지 확장 + 극단 percentile
퇴화) + arm 비교의 decode-batch-size 미제거 교락도 확인. 부수(UNAUDITED,
정본 인용 금지): result-analyst의 `m3_decode_empty.py` 진단이 희석 원인을
decode-empty가 아니라 **prefill 부재**로 재귀속하고 죽은 telemetry 필드
4개를 식별 — 단 "부하를 올려도 engagement가 안 는다"는 결론만은
claims-auditor와 독립 수렴해 그 좁은 항목만 인용 가능. 상세는 아래
"8B decode-SM 프론티어" 절·`CONSENSUS.md` §1-26·`results/s8_frontier/
DESIGN.md` §4.3.9. 이전: 2026-08-02 (진행 상태 갱신만, 결론 개정 아님 — 2026-08-01에 실행된
E1 전제 실험 4건(jobs **870295**=M8 / **870296**=Ha8 / **870297**=Hs8 용량
스캔, **870301**=T8 batch-cap)의 결과를 **상태로만** 기록. ⚠️**이 4건은 전부
claims-auditor 미통과 = 정본·논문 인용 금지**이며 등급어는 **미검증**이다
(아래 "열린 긴장"의 `results/s8_frontier/` 절). 새로 확정으로 올린 것은
**소스 읽기로 검증되는 코드 사실 2건뿐**(`--max-running-requests`가 arm
계열마다 다른 손잡이 · `kv_mamba_occupancy=1.0`은 항등식 — "방법론 게이트"
#5)이고, 여기서 **방법론 게이트 #6**("항등식을 증거로 쓰지 마라")을 신설했다.
이 세션에 세웠다가 claims-auditor에 반증돼 **철회한 6건**은 "철회된 가설"
절에 기록. E1 본 스윕은 **미제출**(설계 위험 3중, "다음 실험 gate" #8).
세션 전문은
[`handoff-report/session_handoff_2026-08-02.md`](handoff-report/session_handoff_2026-08-02.md)).
이전: 2026-07-31 (진행 상태 갱신만, 결론 개정 아님 — `results/s8p_prefill/`
**완료**[claims-auditor 미통과, 정본 인용 금지 유지]·`results/s8_frontier/`(E1)
하네스 구축 완료·본 스윕 미실행. 하네스 전제 job 867231(T8 용량 스캔)·867298
(관측자 효과 게이트)은 2026-07-29 세션 핸드오프에는 PENDING으로 기록됐으나
★**2026-07-31 `sacct` 재확인 결과 둘 다 2026-07-29 21:15–22:41에 이미
COMPLETED**(핸드오프의 예측이 backfill로 빗나감). ★**2026-07-31 정정**: 같은 날
앞선 갱신은 "분석/판정 파일 0건"이라고 적었으나 **오기** — 두 job 모두 **분석이
job 안에서 이미 실행**되어 결과가 `e1cap_T8_867231_result.txt`·
`tfgate_T8_867298_result.txt`에 있고, 판정은 `handoff-report/
session_handoff_2026-07-29.md` §10에 기록돼 있다. 없는 것은 별도 FINDINGS 문서뿐이다.
관측자 효과 게이트 = **조건부 통과**(d92에서만 `itl_p95 +2.00%` t95 [+0.78,+3.21] —
결정 지표 위 비대칭 교란) ⇒ **본 스윕은 `PDMUX_TRACE_FORCE_PREFILL=0`, pin 검증만
별도 ON 런으로 분리**(사전등록: `results/s8_frontier/DESIGN.md` §4.7.1).
⚠️**T8 용량 스캔 = ITL 축이 전 구간 non-binding**(claims-auditor 2026-07-31
확정, 아래 "열린 긴장"). ★**이 절의 2026-07-31 초판이 쓴 "5셀 동시 off-cliff rate
부재 = §4.2/§9.1 escalation 발동"은 부정확해 철회** — rate≲2에서는 d92 포함 전 셀이
plateau 위에 있어 §4.2가 정의한 "D16과 D92 안전대 비중첩"은 엄밀히는 발동하지 않았다.
실제 명제는 **"공통 off-cliff band(≲2–3 req/s)가 ITL 항이 조금이라도 움직이는 영역과
완전히 분리돼 있다"**이다. knee 수치도 정정(첫 교차 기준): d16 12.6 / d24 16.0 /
d44 16.0 / d54 **8.45** / d92 **2.80** req/s(초판의 4.10은 "임계 아래 마지막 점"을
쓴 값이며 곡선이 단조가 아니라 두 정의가 크게 갈린다). **견고한 것은 d92 knee가
나머지보다 3배 이상 낮다는 순서**(세 임계 × 두 x축에서 불변).
★신규 방법론 게이트(집계 단위 선확정 — "방법론 게이트"
절 #4) + green-context auto-revert 실현-배분 관측 사실(`reports/CONSENSUS.md`
§1-22) 등재. 상세는 아래 "열린 긴장"·"다음 실험 gate"·"방법론 게이트" 절).
이전: 2026-07-28 (★★claims-auditor가
사전등록 게이트(`workspace/engine-port/
results/s0_deconfound/DESIGN.md` §5)를 집행 — **부분 GO**. **C1 CONFIRMED**:
Stage 0(2026-07-26)이 인용한 "D108 무경합 앵커"는 코드 버그로 **실제로는 decode
16 SM**이었음이 3중 독립 증거(코드 기전·telemetry 재집계·클라이언트 시그니처)로
확인 ⇒ Stage 0 판정2(NULL)·판정3(게이트 non-binding)을 **철회**, 판정1(raw ITL(D)
스윕 = CONFOUNDED)만 생존, long-ctx L−1 이상은 "게이트 실패로 보류"가 아니라
**"게이트 미실행"**으로 복원. **C2 CONFIRMED(scoped)**: prefill을 16 SM에 고정한
채 decode-SM만 올리면 ITL이 **2.36–2.91× 개선**(4 arm, 모델-무관) — 측정 노트로
정본 진입하되 **정책 이득이 아님**(프론티어 `[108−D,D]` 미측정). **C2b("hybrid
급락=Zamba2 성질") NOT-YET-SUPPORTED**로 강등. **Claim A 등급 변경 없음**(부분
지지), **HE0/HE2/§1-5/§1-7 철회 안 함** — 대신 긴장 2건을 열린 항목으로 기록.
상세는 아래 "Stage 0" 절·"8B decode-SM 민감도 측정 노트" 절·"열린 긴장" 절·
"다음 실험 gate"). 이전: 2026-07-26 (Stage 0[long-ctx L−2 게이트]: 운영점 decode
SM-무감각을 hybrid·16k ctx까지 확장 확인 — non-binding, long-ctx 충돌 가설 이
regime서 붕괴, HE0/벡터1 ctx-무관으로 강화 — ★★2026-07-28 이 판정의 핵심 근거가
철회됨, 위 참조). 2026-07-25 (벡터1[G2.0 short-ctx disjoint
conflict-regime]: CONFIRMED closure, scoped — narrow-rA 확증 sweep g2_0_raconf
완료; 같은 날 논문 positioning 판정[multiplexing 신규성 축 + Transformer-control
게이트, cross-substrate 이식 프레이밍 철회] "다음 실험 gate" #6 추가). 이 문서가
프로젝트 전체의 유일한 현재 상태
정본이다. 이전 문서와 충돌하면 이 문서와
[`reports/paper/`](reports/paper)의
판정을 우선한다.

## 논문 방향

Layer composition을 runtime scheduling boundary로 사용하지 않는다. Hybrid
구조는 coarse-grained decode floor를 예측하는 offline model profile로만
사용한다.

- **H-Architecture:** 역할별 queue, host issue loop, CUDA stream을 실제로
  분리하면 single-worker control-plane coupling을 줄일 수 있다.
- **H-Policy:** model profile과 runtime load로 예측한 decode floor가 시간적으로
  상보적인 near-saturation workload에서 generic dynamic 및 global static보다
  높은 SLO goodput을 제공한다.

두 가설은 아직 검증되지 않았다. 구현 완료와 성능 주장 성립을 구분한다.

## 확정된 결과

1. 여러 Hybrid 모델에서 prefill/decode resource separation은 fused execution보다
   유리한 operating point를 제공한다.
2. 현재 A100/SGLang green-context substrate에서 layer-boundary resource
   switching은 sub-step drain과 synchronization을 일으켜 decode TPOT을 약
   `42→124 ms`로 악화시켰다. 최적화 후에도 약 `85 ms`였다.
3. decode starvation은 active sequence 체류시간과 shared running-batch capacity를
   증가시켜 prefill admission과 TTFT를 악화시킨다. 대표적으로 D16은 D24보다
   prefill SM이 많지만 TTFT가 `7.24 s` 대 `1.21 s`였다.
4. 적절한 static split은 workload와 context/load에 따라 이동한다.
5. 기존 single-worker SLO-aware/binding-first dynamic은 valid varying trace에서
   best static을 넘지 못했다.

KV congestion은 plausible mechanism이지만 기존 artifact에 구조화된 KV occupancy가
없어 아직 독립적인 causal claim이 아니다.

★**갱신(2026-08-02) — occupancy 데이터는 생겼으나 de-confound가 안 된 상태다.**
`results/s8_frontier/`(2026-08-01, 미감사)에서 처음으로 arm별 occupancy가
기록됐다. 그러나 hybrid arm에서 관측된 `kv_mamba_occupancy = 1.0000`은
**메모리 구속의 증거가 아니라 항등식**이다 — 이 캠페인의 설정
(`--disable-radix-cache` ∧ `--max-running-requests 48`)이
`sglang/srt/model_executor/model_runner_kv_cache_mixin.py:223-229`의 분기를
타서 `max_mamba_cache_size = max_running_requests`가 되므로 pool 크기 = cap
이고, batch가 cap에 닿으면 정의상 1.0이다(코드 사실, "방법론 게이트" #5).
⇒ **이 데이터로 hybrid에서 "메모리 vs 스케줄링" 중 무엇이 구속적인지 판정
불가.** 이 regime(ctx 4k ShareGPT)에서 지지되는 것은 좁다: **attention
KV(`kv_full_occupancy`)는 어느 arm에서도 구속 근처에 없었다** — T8 0.032 /
M8 0.145 / Ha8 0.279 / Hs8 0.011(2026-08-01 캠페인 관측치, **claims-auditor
미통과 = 인용 금지**). Claim C의 등급은 변경 없음(running-batch 경로 강함,
KV 경로 부분).

### 벡터1 (G2.0 short-ctx disjoint conflict-regime escape hatch) — CONFIRMED closure (scoped)

6. short-ctx band(아래 scope)에는 동적 제어가 이길 수 있는 disjoint-feasibility
   conflict regime(어떤 static도 두 phase를 동시에 못 커버하는 워크로드)이 **없다**.
   `reports/CONSENSUS.md` §5-8(c) 미결 갈래 (c)를 최종적으로 닫는다.

실험 계열(2026-07-24 실행 시작·2026-07-25 최종 판정): `reports/CONSENSUS.md`
§5-8(c)("충돌 regime 워크로드" — 동적이 이길 disjoint-feasibility escape hatch가
있는가)를 n≥4로 재검증하는 G2.0 short-ctx 스윕(Zamba2-2.7B). 1차 라운드
(g2_0_full/g2_0_hard)는 **ILL-POSED at rA5**로 판정됐다: g2_0_full이 찾은
razor-thin real disjoint(feasible-A={d16,d44} ∩ feasible-B={d54}=∅)는 g2_0_hard
hardening 스윕에서 재현되지 않았고(TTFT 3s-cliff bimodality, n=10 pool 시 d44/d54
둘 다 ~0.86–0.90로 통계적 구분 불가), "disjoint 소멸" 관측은 별도의 ITL-p95
percentile-window 아티팩트였다.

**de-cliff stage-1(jobs 863880–863948) 완료**: `rA{2,3,3.5,4}×{d16,d44,d54}`를
스캔해 `rA=2`만 clean off-cliff임을 확인(`rA≥3`은 전부 여전히 bimodal)하고
`rA=2`를 n=6으로 확증 — 유일한 clean off-cliff 지점에서 static `d54`가 양
phase를 동시 커버, 단 **PLAUSIBLE closure, CONFIRMED 아님**(claims-auditor 반증
3항목: off-cliff에서도 살아있는 split→TTFT gradient·d54 배제 onset이 미측정
전이대·"binding-A⟺on-cliff" 미증명).

**narrow-rA 확증 sweep 완료 — CONFIRMED로 승격**: `g2_0_rasweep`(120 jobs,
`rA{2.25,2.5,2.75,3.0,3.25}×{d16,d34,d44,d54}×n6`)이 off-cliff sub-band
(rate≤2.75)에서 disjoint 부재를 재확인해 전이대를 rate 3.0–3.5로 좁혔고,
그 창을 겨눈 **pre-registered 24-job 확증 열 `g2_0_raconf`**(rate{3.5,3.75}×
{d44,d54}×n=6, 결정규칙: 어떤 rate서든 d54 견고히 <0.7(p90>3s, unimodal) ∧
d44/d16 동시에 견고히 ≥0.95·off-cliff(p90<2s)면 disjoint 실재→REOPEN, 아니면
d54가 양 phase 동시 커버하는 companion collapse면 CONFIRMED)가 **companion
collapse로 판정**:

| rate | split | frac_good mean±SD (n=6) | 비고 |
|---|---|---|---|
| 3.5 | d44 | 0.953 ± 0.035 | |
| 3.5 | d54 | 0.948 ± 0.035 | failTTFT=0/6 (Phase-A도 d44와 통계적 동률) |
| 3.75 | d44 | 0.932 ± 0.042 | |
| 3.75 | d54 | **0.948 ± 0.062** | **d54가 d44보다 높음** |

REOPEN 전제 둘 다 붕괴(d54는 어느 rate서도 <0.7이 아니고, d44도 어느 rate서도
견고히 ≥0.95가 아님: rep 하나가 warm-up성 TTFT-blowup으로 0.844–0.875까지
떨어짐 — 이 blowup은 **split-대칭적**이라 disjoint를 만들지 않음). d54는 Phase B의
유일 feasible split(d44 ITL-p95 50.7ms로 50ms SLO 초과, `frac_good` 0.188;
d54는 44.1ms, `frac_good` 1.000, SD=0)이면서 Phase A도 d44와 대등하게 커버 →
단일 split(d54)이 양 phase를 시간축에서 커버 → **disjoint 없음, 최종 확정**.
상세 per-rep 표·기전·caveat:
[`workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md`](workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md).

★**필수 caveat(overclaim 방지)**: **magnitude는 ill-posed, 순위는 견고** —
Phase-A frac_good≈0.95는 웜업성 TTFT/ITL tail-event(metric cliff)가 결정해
run-length 의존이나, "d54≈d44·d54 미선-배제"라는 **순위**는 견고하다. Phase-B
d44 0.188은 50ms 경계 바로 위라 magnitude는 fragile하나 방향(d44는 decode 못
커버)은 견고하다.

**scope 한정(필수)**: {Zamba2-2.7B, ctx4096, Phase A in2048/o32, Phase B
in2048/o512@rB4, triton attn+mamba, disable-radix-cache, cudagraph-ON, A100
108-SM green-context pdmux, SLO=TTFT 3s ∧ per-req ITL-p95 50ms, inter-phase
drain된 순차 2-phase, rate_A≤3.75} — **"hybrid엔 disjoint 없음"으로 일반화
금지**. **drain caveat**: closure는 얽힘 억제(drain) 조건 관측 = 필요조건
bound이지 hot varying-trace(Claim C 얽힘) 실증 아님. Claim D/E와 §1-20(spatial
coupling-tax, 92+24=116>108, disaggregation +16% headroom)에는 영향 없음 —
시간적 disjoint(단일 static이 시간축에서 양 phase를 커버)와 공간적
coupling-tax는 별개 축이며 "단일 static으로 충분 ⟹ coupling tax 없음"으로
새지 않는다.

**남은 방향(벡터1 종결이 열어두는 것)**: (i) **long-context**(decode floor가
ctx 상승에 따라 올라가므로 — CONSENSUS §1-5 — 충돌이 발생할 수 있는 영역).
★**2026-07-26 갱신, ★★2026-07-28 철회**: 2026-07-26엔 이 방향의 전제(운영점서
decode floor가 ctx로 상승해 binding해지는가)를 Stage 0 게이트가 "ctx≤16k에서는
상승하지 않는다"로 닫았다고 봤으나, **그 근거(D108 무경합 앵커)가 2026-07-28
claims-auditor 감사(C1)로 무효 확인**됐다(아래 "Stage 0" 절). ⇒ **이 전제는 다시
미검증으로 되돌아간다** — "게이트 실패로 보류"가 아니라 "게이트 미실행". (ii)
**§1-20 spatial decoupling**(별도 device pool disaggregation, +16% headroom)은
아직 실행되지 않았다.

상세 verdict:
[`workspace/engine-port/results/g2_0_full/disjoint_verdict_2026-07-24.md`](workspace/engine-port/results/g2_0_full/disjoint_verdict_2026-07-24.md),
[`workspace/engine-port/results/g2_0_hard/hardened_disjoint_verdict_2026-07-25.md`](workspace/engine-port/results/g2_0_hard/hardened_disjoint_verdict_2026-07-25.md),
[`workspace/engine-port/results/g2_0_decliff/decliff_verdict_2026-07-25.md`](workspace/engine-port/results/g2_0_decliff/decliff_verdict_2026-07-25.md),
[`workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md`](workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md).

### Stage 0 (long-ctx L−2 게이트, 2026-07-26) — ★★반증(2026-07-28, claims-auditor C1 CONFIRMED)

7. 운영점(cudagraph-ON, green-context pdmux)에서 decode ITL은 decode-SM(16→108,
   6.75×)에 **무감각(non-binding)**하다 — pure-Transformer(Qwen2.5-3B, 양성
   대조)·pure-Mamba(Mamba2-2.7B, 음성 대조)·hybrid(Zamba2-2.7B, 타깃) **전부**,
   ctx {4k, 8k, **16k**} 전부에서. 유일한 de-confounded 대조 **D16 vs D108(prefill
   경합 0인 두 점) = 1.00 ± 0.01, 3 arm × 3 ctx 전부**.

실험(jobs 864230[H/T]·864601[M], `workspace/engine-port/results/stage0_xctrl/`,
PIN_CHECK 전부 PASS): raw coupled ITL(D) 스윕(D16/D44/D92)은 **CONFOUNDED로
판정**됐다 — D16이 D108과 9셀 전부 ≤0.5% 동일(6.75× SM 증가가 무이득), 최속점 D92는
비단조(prefill을 16 SM으로 굶기는 지점), 그리고 **음성 대조 M**(decode가 O(1)
recurrent라 원리상 SM-bound 불가)이 H와 동형의 "민감도"(2.4×대)를 보이는 것 자체가
그 곡선이 decode-SM이 아니라 prefill 경합/batch-entanglement를 재고 있다는 증거다.
D108 무경합 앵커만이 이 confound를 우회한다.

★★**반증(2026-07-28, claims-auditor 사전등록 게이트 집행, C1 CONFIRMED)**: 위
"D108 무경합 앵커"는 **실제로는 decode 16 SM이었다** — 3중 독립 증거: (i)
**코드 기전** — `manual_divisions=[92,16,0]`의 세 번째 값 0이 legacy auto-path의
threshold로 읽혀 `decode_bs>=0`이 항상 참이 되고 그 결과 **항상 stream_idx
1=(92,16)이 선택**된다(`src/multiplex/multiplexing_mixin.py:725-742`); (ii)
**realized telemetry 재집계** — decode-active 샘플의 **79–96%가 (92,16)**
파티션에서 돌았다(9/9 셀); (iii) **telemetry와 독립인 클라이언트 서명** —
`D108/D16 = 0.992–1.001`(9/9 셀)인데 D92는 D16보다 3.4–3.6× 빠르므로, 108 SM이
92 SM보다 느릴 수 없다는 물리로부터 telemetry 없이도 D108이 실은 D16과 동일
조건이었음이 확인된다. ⇒ **"D16 vs D108 = 1.00±0.01"은 동일 조건의 반복측정**이었다.

**판정2(NULL, "decode SM-무감각")·판정3(게이트 non-binding, "long-ctx 충돌
가설 붕괴·HE0/벡터1 ctx-무관 강화")를 철회한다. 판정1(raw ITL(D) 스윕 =
CONFOUNDED)만 생존**한다(prefill=108−D 공변은 설계상 사실이라 D108 앵커의
유효성과 무관하게 참). ★**"3중 삼각검증" 표현도 철회** — 무경합 앵커는
고장, 음성 대조 M의 전제("decode O(1) recurrent라 SM-bound 불가")도 **틀렸다**
(그 O(1)은 context 길이에 대한 것이지 SM 수에 대한 것이 아니었다 — 아래 "8B
decode-SM 민감도 측정 노트" C2 참조), de-batch 논거는 미감사 — 1/3만 남는다.
D16/D44/D92 각 division의 **pin 자체**(controller 지정값이 realized로도 그
값이었다는 것)는 유효함이 유지된다 — 무효화되는 것은 **D108 앵커 하나뿐**이다.

**연쇄 반영**: `reports/CONSENSUS.md` §1-21 판정2/판정3, `reports/
longcontext_trace_plan.md` §0.6·H_L4·H_L5·L−2 행, `reports/paper/
CLAIM_EVIDENCE_MATRIX.md` Claim A의 Stage 0 evidence 인용, `workspace/
engine-port/results/s0_deconfound/DESIGN.md` §1.1의 "D108: keepalive 0 →
prefill 경합 없음" 표(거짓 — 실제로는 82–96%가 (92,16) 동거; 이 문서가 "재인용
시 필수"로 지정했으므로 정본 재인용 시 반드시 무효 표시)도 함께 철회한다.
**long-ctx 트랙은 "게이트 실패로 보류"가 아니라 "게이트 미실행"으로 복원**한다
— L−2가 실은 아무것도 측정하지 않았으므로 L−1 이상이 멈출 근거가 사라졌다
(재개하라는 뜻은 아니다 — 판정이 없다는 뜻).

상세 [`reports/stage0_verdict_2026-07-26.md`](reports/stage0_verdict_2026-07-26.md)
(원 판정, 위 항목들로 철회됨), `workspace/engine-port/results/s0_deconfound/
PARTITION_RESIDENCY_STAGE0.md`(C1 근거), `workspace/engine-port/results/
s0_deconfound/DESIGN.md`(재측정 설계 — §1.1 표만 무효, 나머지 유효).

### 8B decode-SM 민감도 측정 노트 (scoped, 2026-07-28) — C2 CONFIRMED(scoped)/C2b NOT-YET-SUPPORTED

claims-auditor 판정(사전등록 게이트, `workspace/engine-port/results/
s0_deconfound/DESIGN.md` §5): 아래 측정은 **정책 결론이 아니라 레버 존재를
확립하는 측정 노트**로만 정본에 진입한다. scope 문구는 축약하지 않고 그대로
인용한다.

> {Mamba-Codestral-7.3B / Zamba2-7B / Nemotron-H-8B / Qwen2.5-7B, A100 80GB TP1,
> cudagraph-ON green-context pdmux, `--disable-overlap-schedule
> --chunked-prefill-size -1 --disable-radix-cache`, max-running-requests 48,
> **ctx1024**, conc16 closed-loop, out512, n=4 rep} 조건에서, **prefill을 16
> SM에 고정한 채** decode 파티션만 16→92 SM으로 올리면 decode ITL p50이
> **2.36–2.91×**(4 arm, rep 간 sd 0.01–0.07) 개선된다. 이는 **decode 측
> 등량곡선**이며 `[prefill,decode,idle]=[16,16,76]…[16,92,0]`로 저-D 셀이
> SM을 일부러 놀린다 — **정책 비교가 아니다.** 실제 정책은
> `prefill_SM+decode_SM ≤ 108`을 받으므로 판단 대상은 프론티어 **ITL(D) vs
> TTFT(108−D)**이고 **그것은 미측정**이다. 정본의 실패 기전(얽힘: decode
> 굶김→ITL↑→batch 정체→prefill admission 차단→TTFT 폭발)은 decode-ITL
> 지표에 원리상 보이지 않는다. ⇒ **레버의 존재만 확립하며, 레버를 움직여
> SLO goodput이 나아진다는 근거가 아니다(게이트 #1). HE0(동적 < best
> static)를 되살리지 않는다** — HE0의 死因은 레버 부재가 아니라 positioning
> + 얽힘이었으므로 바뀌는 것은 negative의 **설명**뿐이다. **SM16→SM108(=np)
> 비율은 인용 금지**(분할 자체가 없어 prefill 할당·동시상태·SM clock이
> 함께 바뀜: prefill_active_bs 0.6–0.8 vs 4.5–6.7, clock 1293–1396 vs
> 1396–1403 MHz). ctx는 1024만 귀속 측정됐다(ctx4096은 엔진측 ITL-EWMA
> 프록시로 Ha8 3.46→3.57×, M8 2.56×로 유지 관찰 — **보조 증거**; 8k/16k
> prefill-고정은 미측정). Nemotron-H는 flashinfer, 나머지는 triton(측정
> offset +2.3%, n=1 스모크).

**C2 인용 규율**: 구간(2.36–2.91×)으로 인용, 단일 소수점 금지(bin 선택으로
점추정이 ±0.1 이동: Hs8 2.58 vs 2.67). **C2b("hybrid 급락=Zamba2 additive
성질", Hs8/M8=0.86)는 NOT-YET-SUPPORTED** — 모델간 절대비교(파라미터·형상·
tokenizer 동시 상이) + backend 교차(offset 근거가 20초 스모크 n=1) + 제시
기전(weight-traffic 추정)에 Hs8 데이터가 아예 없고 방향도 반대라 강등한다.
인용 시 scope 문구: "「Hs8/M8=0.86, Ha8/M8=1.57–1.69」는 모델간 절대비용의
**통제되지 않은 관찰**이다(파라미터 수·형상·tokenizer·backend 동시 상이).
아키텍처 계열(additive vs substitutive)에 대한 **기전 주장으로 쓰지 않는다.**"

상세 [`workspace/engine-port/results/s8_scaleup/FINDINGS_8B_2026-07-28.md`](workspace/engine-port/results/s8_scaleup/FINDINGS_8B_2026-07-28.md)
(§6에 (a)"레버 존재≠정책 이득" (b)"HE0를 되살리지 않는다" 명시, §3 retraction을
"SM108이 근소하게 빠르다"에서 "비교 불가"로 강화 — 2026-07-28 doc-steward 반영).

## 열린 긴장 (2026-07-28, claims-auditor 지정 — HE2/§1-5/§1-7/HE0 철회 아님)

- **긴장 A (HE2 vs C2)**: C2가 맞는데 왜 HE2(운영점서 decode 최적 split=static·
  불변, 동적이 anchor 무관 패)는 평탄했나? 유력 가설 = HE2는
  `prefill+decode≤108` 예산 제약 하에서 D를 움직였으므로 decode 이득이 prefill
  손실+얽힘으로 상쇄된다 — **레버는 있으나 예산 제약 하 net-positive가 아닐 수
  있다**는 뜻. **이것은 가설이며 미측정**이고, 정확히 아래 E1(프론티어 실험)의
  대상이다.
- **긴장 B (r0c 부분 복권)**: `reports/CONSENSUS.md` §1-5가 r0c의 no-cudagraph
  decode-knee(ctx256 1.1×→ctx16k 10.5×)를 "운영점 magnitude는 열린 질문"으로
  강등했는데, C2(cudagraph-ON 서빙, ctx1024, 2.36–2.91×)가 그 곡선 위에 앉는다
  ⇒ "열린 질문"이 일부 닫히는 **방향**이다. **"정합"까지만 쓴다 — "확증"으로
  쓰지 않는다.**
- **`results/s8p_prefill/`(prefill 축 SM 민감도) — 완료(2026-07-29), 정본 인용
  금지 유지**(claims-auditor 미통과). 판정서
  [`FINDINGS_PREFILL_2026-07-29.md`](workspace/engine-port/results/s8p_prefill/FINDINGS_PREFILL_2026-07-29.md).
  한 줄 요약(등급어 없이 REAL scoped·미감사로만 인용): "prefill 축 SM 민감도 =
  기울기 비 **4.74–5.16×**, 탄력도 ε **0.89–0.94**, 4 arm 모델-무관". 아래
  "다음 실험 gate" #8에 claims-auditor 반증 축과 함께 기록.
- **`results/s8_frontier/`(E1 8B 프론티어) — 하네스 구축 완료, 본 스윕
  미실행**. ★**2026-08-02 갱신**: 2026-08-01에 나머지 3 arm 용량 스캔
  (870295/870296/870297)과 T8 batch-cap 실험(870301)이 완료됐고, 그 결과가
  **E1 설계 자체에 대한 판정**으로 이어졌다(아래 "2026-08-01 실험 4건" 소절).
  **본 스윕은 여전히 미제출**이다. 용량 스캔(job **867231**, T8,
  5셀)·관측자 효과 게이트(job
  **867298**, `results/e1_traceforce/`)는 세션 핸드오프(2026-07-29) 작성
  시점엔 PENDING으로 기록됐으나, ★**2026-07-31 doc-steward 갱신 시 `sacct`
  재확인 — 둘 다 COMPLETED**(867231: 2026-07-29 21:15:25–22:41:31 / 867298:
  21:16:56–21:50:21). ★**2026-07-31 정정** — 같은 날 앞선 갱신의 "분석/판정
  파일 0건"은 **오기**다. 두 job 모두 **분석 단계가 job 스크립트 안에서 이미
  실행**되어 결과가 `e1cap_T8_867231_result.txt`(100 probe)·
  `tfgate_T8_867298_result.txt`(`=== PAIRED SUMMARY ===`)에 있고, 판정은
  [`handoff-report/session_handoff_2026-07-29.md`](handoff-report/session_handoff_2026-07-29.md)
  §10에 기록돼 있다. 부재한 것은 별도 FINDINGS 문서뿐이며, **재실행은 불필요**하다.
  - **관측자 효과 게이트 = 조건부 통과.** d16은 전 지표 t95 CI가 0을 포함,
    d92만 `itl_p95` **+2.00%** [+0.78, +3.21] · `itl_p99` −1.28% · `itl_mean`
    +0.56%가 0을 배제. 교란의 크기(≤2%)보다 **위치**가 문제 — 하필 결정 규칙이
    임계하는 ITL-p95이고 D-격자의 한쪽 끝에서만 난다. ⇒ **본 스윕은
    `PDMUX_TRACE_FORCE_PREFILL=0`(기본값)으로 돌리고, pin 검증은 같은 arm/cell/
    rate/seed의 짧은 ON 런으로 분리**한다(사전등록 = `results/s8_frontier/
    DESIGN.md` §4.7.1). ⚠️ 그 job의 `PIN_CHECK`는 `traceforce_gate.sbatch`의
    옛 인자 순서 때문에 전부 크래시해 **pin 데이터가 없다**(paired summary는
    별도 분석기 산출이라 무영향). 인자 순서는 **2026-07-31 수정 완료**.
  - ⚠️★**T8 용량 스캔은 §4.2/§9.1이 사전등록한 escalation 분기를 발동시켰다.**
    ★**2026-07-31 claims-auditor가 이 항목의 초판을 정정** — "5셀 동시
    off-cliff rate 부재"는 부정확하다(rate≲2에서 d92 포함 전 셀이 plateau 위).
    knee(첫 교차, plateau 2×): d16 12.6 / d24 16.0 / d44 16.0 / d54 **8.45** /
    d92 **2.80** req/s. **견고한 것은 d92 knee가 나머지보다 3배 이상 낮다는
    순서**(세 임계 × 두 x축 불변, 기전도 확실: 저부하 TTFT plateau가 181 vs
    47–62ms이므로 같은 rate에서 ρ가 3–4배). 실제로 성립하는 명제는 **"공통
    off-cliff band(≲2–3 req/s)가 ITL 항이 움직이는 영역과 완전히 분리돼
    있다"**이며, 셀별로 다른 rate를 골라 우회하는 것은 §4.2가 금지한 rate-교락
    이므로 여전히 불가하다. 상세 = 아래 "열린 긴장".
  - ★★**T8의 ITL 항은 전 rate·전 사다리 룽에서 non-binding**(claims-auditor
    2026-07-31, C-E CONFIRMED). 요청 단위 직접 집계로 rate 1–32 전 구간에서
    `ITL-p95 ≤ 50ms`를 요청의 **≥93%**가 통과한다. ⇒ T8에서는 **어떤 결과가
    나와도 C2 판정 불가**(§4.3.4가 사전등록한 `ITL-NONBINDING` 취급). ⚠️단
    **"그러므로 E1의 판정력이 M8/Ha8/Hs8에 걸린다"는 NOT-YET-SUPPORTED** —
    같은 batch cap이 그 arm들에도 걸려 같은 계단 구조가 생기고, Ha8은 외삽상
    d16 ≈149ms로 **전 룽 초과**(모든 셀에서 구속 ⇒ conjunctive goodput 전멸
    ⇒ 역시 판정 불가) 가능성이 있다. §4.3.4에 대칭 플래그
    **`ITL-ALWAYS-BINDING`이 없다** = 미등록 실패 모드(신설 필요).
    ★**2026-08-02 후속**: 플래그는 2026-07-31에 `DESIGN.md` §4.3.5(a-1)로
    **신설·사전등록**됐고, 2026-08-01 스캔에서 **Ha8이 전 룽
    `ITL-ALWAYS-BINDING`으로 실제 발동**했다(위 "2026-08-01 실험 4건" (3),
    **미감사**) — 위 "판정력이 M8/Ha8/Hs8에 걸린다"는 그만큼 부분 반대
    증거를 얻었다(철회 기록은 "철회된 가설" 절).
  - ★★**미등록 하네스 상수가 ITL 축을 단독 결정한다 — `--max-running-requests
    48`**(`e1_capacity_scan.sbatch:143`, `e1_sweep.sbatch:178`). `DESIGN.md`에
    **단 한 번도 등장하지 않는다**(grep 0건). 증거: telemetry의
    `decode_running_batch_size`가 d16/d24/d44/d54 전부 정확히 48에서 절단
    (d92만 38 — prefill admission이 먼저 막혀 cap 미도달 ⇒ **d92의 ITL은 다른
    셀과 비교 불가**), 그 시점 `kv_occupancy = 0.024`(2.4%, 자원 강제 아님 —
    ⚠️**T8 한정 관측**이며 "메모리는 어느 arm에서도 구속하지 않는다"로
    일반화 금지: hybrid arm의 mamba pool은 cap과 항등이라 별개 문제다,
    "방법론 게이트" #5),
    d16 ITL-p95가 rate 12→32에서 50.2–50.6ms로 완전 평탄. ⇒ 셀별 ITL "천장"
    {d16 50.6 / d24 37.9 / d44 26.2 / d54 23.9 / d92 19.1}은 **모델 성질이
    아니라 설정 성질**이며(★2026-08-01 batch-cap 실험이 T8 2셀에서 이를 직접
    시험 — 위 "2026-08-01 실험 4건" (1), **미감사**), "60ms 도달 불가"의
    scope가 달라진다.
  - ★★★**decode 측 realized-partition duty cycle이 D와 공변한다**(Stage 0
    C1과 같은 종, 이번엔 decode 축). `decode_sms`를 decode-active 구간에서
    시간가중하면 **라벨 D SM에서 보낸 시간 비율 = d16 0.110 / d24 0.112 /
    d44 0.166 / d54 0.201 / d92 0.518** — 나머지는 무분할 108 SM이다. 즉
    **D 축이 "decode SM 양"과 "그 제한이 걸리는 시간 비율" 두 변수를 동시에
    움직인다.** 사전등록 `E1_PIN_GATE`는 **prefill 측만** 검사하므로 이걸 못
    잡는다(`D=16(P92) pin_frac=0.950`은 prefill이 92 SM인지의 지표). 배치·
    파티션을 동시에 맞추면(b=48, `decode_sms==target`) d16 28.8 / d24 22.2 /
    d44 19.4 / d54 18.4ms로 **d16이 27% 더 느리다**(혼합 집계의 22.6보다 큼).
    ⇒ **decode 측 duty-cycle 게이트 신설 필요**(수치 자체도 스냅샷 샘플링
    기반 1차 근사라 엔진측 누적 시간으로 재측정 대상).
  사전등록 SLO·결정 규칙은 아래 "다음 실험 gate" #8 참조.

  #### 2026-08-01 실험 4건 — ⚠️**전부 claims-auditor 미통과, 등급 = 미검증, 정본·논문 인용 금지**

  > ★★**2026-08-02 감사 완료 — 아래 절의 상당수가 이미 철회·대체됐다. 이 절을
  > 인용하기 전에 반드시 이 박스를 먼저 읽을 것.** claims-auditor 회부 결과
  > **2 REFUTED / 2 NOT-YET-SUPPORTED / 1 CONFIRMED**, 후속 M1/M2/M4로 다음이
  > 확정됐다(정본 = `results/s8_frontier/DESIGN.md` §4.3.7–§4.3.8 +
  > `FINDINGS_M1_M2_2026-08-02.md`):
  >
  > - **(1) "천장 = 설정 성질" → NOT-YET-SUPPORTED.** 1 arm·2셀에서 모델 변이
  >   0인 설계로 "모델 성질이 아니다"를 결론할 수 없고, d44의 +1.3ms도 실은
  >   CI가 0을 배제한다(=설정×셀 상호작용). 더구나 **rate 16 = 공통 off-cliff
  >   대역의 5.7배**라 E1 운영구간 밖이다. **재실험하지 않고 폐기.**
  > - **(2) "cap 96 ≈ 192" → REFUTED.** `frac(ITL≤60)` −0.091[−0.166,−0.016],
  >   TTFT p50 +6.75ms 모두 0 배제. d44 cap96 행은 누락이 아니라 미보고이고
  >   넣으면 비단조.
  > - **(3) mamba pool 항등식 → CONFIRMED**(코드 분기 + 서버 로그
  >   `max_mamba_cache_size: 48` + telemetry의 1/48 양자화, 3방향 독립).
  >   단 "1.0000 = 포화"는 **max 통계**였고 실제로는 스냅샷의 0.006–0.08%.
  > - **(4) "HEADLINE NONE = cap 아티팩트" → REFUTED.** 운영대역 실측 동시성
  >   12–44 < cap 48 ⇒ **cap이 구속할 수 없다**. 다만 결론 자체
  >   (`HEADLINE-ELIGIBLE RUNGS = NONE`)는 **살아남았다** — 공통 rate·두
  >   estimand·두 seed 전부에서 재확인. **arm×룽 표는 rate-confound로 폐기**
  >   (T8만 rate 12, 나머지 rate 2 — §4.3.5(b) 명시 위반), `--common-rate`
  >   출력으로 대체.
  > - **(5) knee — "네 arm 공통 2.80" 철회.** knee의 치역이 probe 격자뿐이라
  >   일치가 부분적으로 강제된다(게이트 #6). 9-변형 집합은 arm마다 다름.
  >   **살아남는 것은 순서**(d92가 먼저 무너짐: T8/M8/Hs8 9/9, Ha8 8/9).
  >   Hs8 d16 knee 9.09의 취약성은 실재하나 **원인 귀속이 틀렸다** — 그
  >   이상치를 지워도 knee 불변.
  > - **(6) stall 원인 규명(M4, GPU 0).** 17개 stall probe 중 **16개**에서
  >   최장 프롬프트의 **monolithic prefill**이 stall 전 구간을 덮는다. 크기는
  >   prefill SM = 108−D 이므로 **D에 단조**. `enable_pdmux`가
  >   `chunked_prefill_size == -1`을 하드 assert하므로 **기판 구조이지 버그가
  >   아니다**(venue positioning의 (A) green-context 종속 버킷).
  >
  > E1 본 스윕은 여전히 미제출이며, 대신 **M3 Transformer-control 대조**를
  > 사전등록·제출했다(job **872077**, §4.3.8(c)).

  아래는 **상태 기록**이며 "확정된 결과"가 아니다. 원자료 =
  `workspace/engine-port/results/s8_frontier/`. 4 job 전부 COMPLETED,
  probe 오류 0(870295/870296/870297 각 100 probe, 870301 24 probe).
  판정서는 아직 없다(`e1bcap_T8_870301_result.txt` 등 in-job 분석 산출물만
  존재). 반증 대상 목록은 `handoff-report/session_handoff_2026-08-02.md`
  "열린 항목" 1번.

  - **(1) batch-cap 실험(job 870301, T8 × {d16,d44} × cap{48,96,192} × 4
    seed) — 미검증**. 셀별 ITL "천장"은 **모델 성질이 아니라 설정 성질**로
    보인다: d16 ITL-p95가 cap 48→96→192에서 **49.7 → 59.0 → 62.0 ms**,
    seed-paired Δ(48→192) = **+12.2 ± 0.7 ms**(4/4 seed), d44는 **+1.3 ±
    0.7**. decode batch가 48 → 60–75로 자란 뒤 cap 192에서도 그 자리에서
    정지 ⇒ **cap 48만 실제로 구속**하고 96 이상은 도착·서비스율이 정하는
    자연 평형. ⚠️**scope: T8(순수 Transformer) 2셀만 측정 — 이 문구 없이
    인용 금지**(hybrid 이전 불가, 사유는 "방법론 게이트" #5). 부수 관측:
    같은 d16에서 TTFT-p50이 cap 48일 때 **612 ± 273 ms**인데 cap을 풀면
    **114 ± 11 ms**로 내려간다 ⇒ **cap 48이 TTFT를 5.4× 악화**시키고 있었고,
    같은 설정이 ITL은 좋아 보이게 만들었다. cap이 숨은 admission control로
    작동해 **두 축을 반대 방향으로 동시에 왜곡**한 것이다.
  - **(2) 4-arm 용량 knee(jobs 870295/870296/870297 + 기존 867231) —
    미검증**. 첫 교차 기준 knee(req/s):

    | arm | d16 | d24 | d44 | d54 | **d92** | 구속 셀 |
    |---|---|---|---|---|---|---|
    | T8 | 12.6 | 16.0 | 16.0 | 8.45 | **2.80** | d92 |
    | M8 | 5.60 | 5.60 | 5.60 | 4.20 | **2.80** | d92 |
    | Ha8 | 5.60 | 5.60 | 4.20 | 3.08 | **2.80** | d92 |
    | Hs8 | 9.09* | 12.6 | 9.09 | 8.45 | **2.80** | d92 |

    **d92가 네 arm 전부에서 구속 셀**이고 knee가 2.80으로 일치 ⇒ 공통
    off-cliff 상한 2.80 req/s. ★**기존 `results/s8_frontier/DESIGN.md`
    §4.3.5(b)에 *추측으로* 적어둔 "느린 decode arm(M8/Ha8)은 반대로 d16을
    먼저 잃을 것"은 지지되지 않는다** — 모델과 무관하게 d92의 prefill 16
    SM이 먼저 무너진다(사전등록에 추측으로 표시해둔 덕에 손해 없음).
    `*` Hs8 d16의 9.09는 견고하지 않다(한 점이 임계를 2% 초과하고 다음
    점이 회귀; 같은 셀 `arr=12.61 seed=1`에 미조사 이상치 1건).
  - **(3) 룽 분류 — 미검증**. 사전등록 사다리 {50, 60, 80} ms가 **네 arm
    전부에서 `HEADLINE-ELIGIBLE RUNGS = NONE`**이다(T8/Hs8 = 50ms
    CLIFF-HAZARD·60/80ms NONBINDING; M8 = 50ms ALWAYS-BINDING·60/80ms
    CLIFF-HAZARD; **Ha8 = 전 룽 `ITL-ALWAYS-BINDING`**). 2026-07-31에
    신설한 `ITL-ALWAYS-BINDING` 플래그가 **등록 몇 시간 뒤 Ha8에서 실제로
    발동**했다 — 없었다면 Ha8의 conjunctive goodput 0이 "어떤 D도 못 이김
    = 레버 net-negative"로 오독됐을 것이다(`degenerate_goodput_guard()`가
    결정 규칙을 차단).
  - **(4) E1 설계 위험(미검증, 판정 아님)**: (a) 사전등록 사다리가 as-run
    설정에서 **네 arm 전부 판정 불가**, (b) 그 as-run 설정 자체가
    **왜곡으로 증명됨**(cap 48, 위 (1)), (c) 제외 규칙(d92 knee 2.80, 전
    arm)이 **어떤 동작점에서도 decode-rich 끝을 제거** — C2의 레버가 사는
    바로 그 끝. ⇒ **E1이 `DESIGN.md` §4.3.5(b)가 사전등록해둔 분기
    "E1 as designed cannot reach this question"(설계상 이 질문에 도달할 수
    없다)으로 갈 위험이 높다.** 이는 **실패가 아니라 미리 적어둔 분기**이며,
    본 스윕에 GPU를 쓰기 전에 알아낸 것이다. 본 스윕 **미제출**.
  - ⚠️**as-run 상수 경고**: `PROBE_TARGET_S`는 문서화된 20이 아니라 **8**로
    867231·870295–297·870301이 전부 돌았다. **비교 런을 추가할 때 반드시
    8로 맞출 것**(현재는 submit 라인에 명시돼 있다).

  #### 2026-08-03 (같은 날 속행) — `g`가 이 격자에서 은퇴한다: 희석
  attenuation REFUTED, NO VERDICT 사유 확장, `A_free` 결함, arm 교락

  > ★★★**출처 구분(overclaim 금지).** A–D = **[AUDITED]**(claims-auditor가
  > 872077 원자료 telemetry 64 + bench 64를 독립 재분석해 판정, 정본 인용
  > 가능). E = **[UNAUDITED]**(result-analyst 산출, claims-auditor 미통과,
  > **정본 인용 금지** — 명시된 한 항목만 예외).

  - **(A) 메인 세션의 "희석 attenuation" 주장 — REFUTED.** 이날 앞선 회차의
    `E1_DECODE_REALIZED`(4–19%, 위 gate #7·`CONSENSUS.md` §1-25)를 근거로
    "`A_free(dD)=w_D·A(D)+(1−w_D)·A(108)` 혼합이고 그게 `g`를 1 쪽으로
    attenuate시킨다"는 보정 모형을 세워 Ha8 보정치 ≈1.62–1.70을 역산했다.
    **판정 = REFUTED**, 독립 증거 3줄: (i) **control-arm reductio** — 같은
    보정식을 T8에 적용하면 corrected g **21–29×**(A108∈{12,14,15}, b∈{0,2}) —
    정본 C2(SM16→92 2.36–2.91×)를 더 좁은 16→54 구간에서 **10배 위반**하고,
    모형이 요구하는 T8 d16 split-조건부 ITL p95(352–360ms)가 실측
    **30.67ms**와 10배 어긋난다. (ii) **de-engagement 직접 실험**(split
    라벨 토큰을 같은 셀 unsplit 분포에서 재추출해 engagement를 `f·w`로
    낮춤) — `A_free` 변화는 **1–11%뿐**(Ha8 d16 113.51→112.29 −1.1%, Ha8
    d54 107.45→95.74 −10.9%, T8 d16 28.31→26.97 −4.7%, T8 d54 15.38→15.02
    −2.3%). **w=0에서 g = Ha8 1.173 / T8 1.796** — 헤드라인이 거의 그대로
    남는다. (iii) 핵심 가정 "`A(108)` 셀 무관"이 실측에 반한다 — `A_free`
    형태를 하위 모집단에 적용 시 Ha8 ALL 1.068[0.947,1.190] / **SPLIT-only
    0.920[0.842,0.998]**(CI가 1 배제, **부호 반대**) / UNSPLIT-only
    1.146[0.997,1.296]; **T8 헤드라인 효과 전부가 decode SM 대비가 정의상
    0인 UNSPLIT-only 모집단에서 재현**(1.795[1.589,2.001] ≈ ALL 1.837).
    confound 유형 = #1(서빙 직접 측정을 오프라인 산술 모형으로 대체) +
    **#6**(항등식에서 파생된 `w`를 자유 모수처럼 나눔, "방법론 게이트" #6
    새 사례로 등재). **살아남은 것**: engagement가 낮다는 §1-25의 전제
    자체는 견고(스냅샷·event-driven·토큰 기준 세 계측기 교차확인) — 죽은
    것은 **보정**뿐. 집계 단위(게이트 #5) 부호는 확정(시간-몫 > 토큰-몫 ⇒
    시간가중 engagement는 과대평가)됐으나 2차항이 이미 모형을 죽인다.
  - **(B) 872077 — NO VERDICT 사유를 "규칙 모호"에서 "estimand 미식별"로
    확장.** 코드 사실: `pdmux_context.py:initialize_stream_groups`가
    `SM_COUNTS=[(108,0)]+divisions+[(0,108)]`를 하드코딩하고
    `multiplexing_mixin.py:773,792-794`가 prefill 비-in-flight 시 무조건
    `real_sm_group_num-1`=plain `(0,108)`로 되돌린다 ⇒ **"decode가 D SM에서
    돌았다"와 "prefill이 동시에 실행 중이었다"는 이 기판에서 같은 사건**이다.
    이 격자의 어떤 통계도 decode-SM 탄력도와 prefill 간섭을 분리 못 한다.
    §1-24(ITL 꼬리=monolithic prefill, 크기가 108−D에 단조)와 결합하면 `g`는
    사전에 **"decode-SM 탄력도 라벨을 단 prefill-SM 탄력도"**일 것이
    예상되고, 실측이 그와 일치(UNSPLIT-only서 헤드라인 재현)한다. **이는
    n으로 해결되지 않는 설계 결함이다.** 기록: 872077의 NO VERDICT 지위는
    **유지**하되 사유를 위와 같이 확장한다. **`g = A_free(d16)/A_free(d54)`는
    이 격자 한정 은퇴**(sticky partition 기판 수정 전까지 인용 금지). 블록
    8→12–16 증설 재실행은 **선행 금지**(참고: 현 mean/sd 유지 가정 시 상한
    ≤1.15 확률은 n=12 43%/16 55%/24 75%/32 87%/40 93%였으나, 기판 수정 후
    sd가 달라지므로 이 표는 사전에 무효). ⚠️**"Ha8에 decode-SM 레버가
    없다"는 CONFIRMED가 아니다** — 현 데이터는 그 질문에 답하지 못한다.
    **긴장 A(HE2 vs C2)는 전혀 닫히지 않았다.** `CLAIM_EVIDENCE_MATRIX.md`/
    `EXPERIMENT_ROADMAP.md`에도 이 상태(등급 임의 변경 없이, 닫히지
    않았다는 사실만)를 반영한다.
  - **(C) `A_free` 추정량 자체의 결함(E1 하네스 전반에 걸림).**
    `e1_m3_control.sbatch:281-306`을 읽은 결과: (i) blocking 필터가 작동하지
    않는다 — `PREFILL_BLOCK_TOK=1024`인데 이 워크로드는 요청의 4–5%만
    ≥1024 tok이고 그 prefill 토큰 몫은 23–26%뿐이라 **prefill 작업의
    74–77%가 필터를 통과**(실제 제거되는 ITL은 전체의 ~2%). §1-24가 확정한
    monolithic prefill stall(원문은 "d92만 오염"으로 한정)이 **d16–d54까지
    오염 범위가 확장됨**을 기록한다 — T8 d16 꼬리를 만드는 요청 input은
    221–804 tok로 전부 임계 아래. (ii) 요청의 27.5–29.5%가 output≤25
    토큰이라 내부 p95가 사실상 max ITL로 퇴화 — 평균의 선형 혼합 항등식이
    극단 분위수에 성립하지 않는다(비단조 응답 Ha8 d54
    107.45→101.89→102.84→97.75가 실증). ⇒ **`A_free`는 blocking 제거본이
    아니라 대부분이 monolithic-prefill stall로 이루어진 극단꼬리 통계다.**
    추정량 교체가 후속 로드맵 항목(아래).
  - **(D) arm 간 비교의 미제거 교락.** 공통 rate 2에서 T8 conc 12.8/decode
    batch 4.5/ITL p50 ~11ms 대 Ha8 conc 30.6/batch 15.8/~30ms. 양 arm 모두
    off-cliff 평탄역(0.88–1.07)이라 metric cliff는 아니나, decode batch
    size가 decode step의 memory-bound/compute-bound 여부를 결정하는
    공변량이라 arm과 완전 교락 ⇒ "attributable to the arm" 문구는 **현재
    허용되지 않는다**(통제는 arrival rate가 아니라 realized concurrency/
    decode batch를 맞춘 rate). ⚠️이 교락이 관측 *방향*을 설명하지는
    않으므로(batch 큰 쪽이 오히려 무반응) **대안 설명이 아니라 미제거
    교락**으로 기록한다. 부수: d16은 `sm_group_num:3`, d54는 4(guard
    row)로 셀마다 green context 수가 다르고, d54에서 `decode_sms==44`는
    전 telemetry에서 미관측(guard row 미선택) — 행동 교락은 아니나 셀 간
    차이로 기록.
  - **(E) [UNAUDITED — 정본 인용 금지] result-analyst의 decode-empty 진단.**
    산출 스크립트 `results/s8_frontier/m3_decode_empty.py`(재실행 가능).
    claims-auditor 미통과이나 (B)와 독립적으로 수렴하는 부분이 있어 기록
    가치가 있다. ★희석의 원인은 decode 공백이 아니라 **prefill 부재**다
    (`E1_DECODE_REALIZED`는 decode-active 시간에 조건부라 decode-empty는
    정의상 분자·분모 어디에도 안 들어감; 실측 기여 T8 −0.0006±0.0046/Ha8
    −0.0001±0.0028 = 0, block-paired 분해에서 gradient의 100%가 prefill
    점유율 gradient로 설명됨, 잔차 CI가 0 포함). 부하창 내 decode-empty는
    **0.6–1.3%뿐**(원자료의 15–19%는 클라이언트 warmup→dataset 준비 갭 +
    종료 후 꼬리로 인한 **측정창 아티팩트** — 클라이언트 `duration`으로
    앵커한 부하창에서 분석해야 함). ★게이트 #6 필드 감사 — 죽은 telemetry
    필드 4개(코드 근거 포함): `decode_ready_queue_depth` 항등 0(dual-worker
    가드 안에서만 채워지는데 872077은 `architecture=="legacy"`),
    `active_decode_sequences`≡`decode_running_batch_size`,
    `decode_idle_ratio`/`prefill_idle_ratio` 항등 0.0(`controller.py:55`에서
    선언만 되고 대입 없음), `running_batch_occupancy`≡min(1,drb/48),
    `prefill_admission_blocked`≡(pqd>0∧pab==0). 집계 단위(게이트 #5)
    재확인: decode-empty 시간몫이 시간가중 0.0056–0.0127 vs 개수
    0.333–0.424(30–60× 차이), 분해 비율(A 지배)은 세 단위 모두 강건.
    arm별 realized 천장이 워크로드 성질(ΣTTFT/decode-busy, T8
    0.120–0.154/Ha8 0.381–0.558)이라는 관측은 클라이언트 측 양이라
    텔레메트리 계측 문제에 면역 — `g`의 arm 간 비교에 D의 교락과 별개인
    추가 축. UNDETERMINED: realization gradient의 크기(스냅샷 dt p95
    195–805ms가 prefill span과 동 자릿수라 물리량으로 인용 금지) 및
    t_pa 기반 prefill span 증가와 클라이언트 TTFT 증가의 미해소 모순(해소법
    = prefill batch start/end 이벤트 직접 emit 후 재측정). ★**단 이 항목
    하나만은 claims-auditor와 독립적으로 수렴해 [AUDITED]로 인용 가능**:
    부하를 올려 engagement를 높이는 방향은 데이터가 지지하지 않는다
    (decode는 이미 부하 중 ~99% busy; rate를 올리면 prefill·decode가
    비례해 늘고, 셀 간 `w` 차이조차 arrival이 아니라 prefill이 108−D에서
    느려져 생긴 것; split-eligible iteration 수는 셀 무관 194–214로 거의
    일정).

  **다음 실험 gate(2026-08-03 1차 속행 시점 계획)**: 감사자 권고 순서 —
  1–3 = 이 절 자체를 정본에 기록(완료, GPU 0), 4 = `PDMUX_STICKY_PARTITION`
  구현 + correctness gate(engine-porter, ~0.25h — **구현 완료 ≠ 성능 주장
  성립**), 5 = sticky 격자 1회(872077 동일 설계 8 block, ~3.0h; 872077이
  non-sticky 대조군 ⇒ 총 ~3.3 GPU-hour), 6 = 블록 증설은 **4·5 이후에만**.
  ★**4는 아래 "2026-08-03(2차)" 소절에서 완료됐다 — 5는 여전히 미제출**.

  #### 2026-08-03 (같은 날 2차 속행) — estimand 이관 완료 + `PDMUX_STICKY_PARTITION`
      구현·correctness gate 통과

  > 출처 구분(overclaim 금지). **(I)**는 claims-auditor의 산출이며 (1)(2)는
  > **[AUDITED]**, **(3)만 [UNAUDITED]**(감사자가 이번 턴에 새로 생산한
  > 것을 감사자 자신이 감사한 형태 — 별도 확증 전 정본 인용 금지). **(II)**는
  > engine-porter의 **구현 사실**이며 성능 판정이 아니다. **(III)**은
  > 사전등록이며 결과가 아니다. 원자료·코드:
  > `results/s8_frontier/m3_conditional.py`(신규, 미추적),
  > `src/multiplex/multiplexing_mixin.py`(+151/−17), `tests/
  > test_sticky_partition.py`(신규), `results/sticky_smoke/`(신규). 전문
  > `results/s8_frontier/DESIGN.md` §4.3.10–4.3.12.

  **(I) `A_free`를 대체하는 조건부 per-token 추정량.**

  1. **[AUDITED] 정의.** 단위 = 개별 ITL 구간 1개(요청별 내부 집계 없음 —
     이것이 `A_free`의 두 병리, 요청별-p95 이중극단·outlen≤25 퇴화를 피하는
     전부). 라벨: 구간 `[a,b]`(client 시계, 스냅샷 계단함수 기준)에
     `split_frac(a,b) = (decode_sms==D였던 시간)/(b−a)`,
     **SPLIT := ≥0.90 / UNSPLIT := ≤0.10 / AMBIGUOUS := 사이(0.3–1.4%) →
     강제분류 없이 양쪽 배제**. 통계량은 셀-블록별 **직접 분위수**(primary
     `p95(SPLIT)`, secondary `p50(SPLIT)`, **control `p95/p50(UNSPLIT)`**,
     mean 병기). ★**UNSPLIT control이 핵심 안전장치**: 두 셀 모두 108 SM이라
     decode-SM 대비가 **정의상 0** — 여기서 비가 1이 아니면 그 차이는
     decode SM이 만든 게 아니다. 대비량은 block 내 paired, 8-block
     block-clustered t-구간(`TCRIT[8]=2.365`, percentile bootstrap 금지 —
     n=8 커버리지 79.8% 기지). 회피 근거: T8 d16 pooled per-token p95
     **11.60**인데 `A_free`=**28.31**(2배 이상 바깥 꼬리); 요청의
     27.5–29.5%가 outlen≤25; `A_free` outer-p95는 요청 ~9.6개에 얹히는 반면
     새 추정량은 셀-블록당 1,390–7,045 토큰. ⚠️**이 추정량은 `A_free`보다
     잘 정의됐을 뿐 사전등록된 SLO 항이 아니다** — `A_all`(등록 SLO 항)은
     계속 병기.
  2. **[AUDITED] client↔telemetry 시계 정렬.** `phase_marker(phase==
     "benchmark")` 앵커 → `replay_arrivals(seed,rate,n)`을
     `e1_m3_control.sbatch:284-287`과 bit-identical 복제 → client 계단함수
     구성 → telemetry `decode_running_batch_size`와 Pearson r 최대화.
     **lag이 13.9–35.9s인 이유**: `benchmark` 마커는 서버가 처음 본
     **warm-up 요청**에 발화하고(`multiplexing_mixin.py:386-398`) 이어
     데이터셋 토크나이즈가 실제 probe 시작을 미룬다 ⇒ **마커는 탐색
     앵커일 뿐 probe 경계가 아니므로 `load_telemetry`는 `phase==
     "benchmark"` 필터를 걸면 안 된다**. 감사자 자기정정: "정렬 약한
     probe 1개(T8 d16 b1, r=0.882)"는 24개 spot-check 결과였고 전수
     64 probe에서는 **2개**(T8 d16 b6, r=0.930 추가) — **보고 수치는 전부
     8 block 전수 계산이라 변경 없음**, 정정 대상은 서술뿐. **규칙**:
     `ALIGN_R_MIN=0.95`, 미달은 ALIGN-WEAK로 flag하고 포함본/제외본 둘 다
     보고(조용한 drop 금지) — 근거: (a) `r`이 block 내 paired라 probe
     하나 배제 = block 전체 소실 = `n_indep` 8→7, (b) 오정렬은 라벨을
     무작위화해 두 하위모집단을 pooled로 끌어당길 뿐(분리를 약화만 시킴,
     편향이 보수적), (c) 실측 LOO(`sp_p95` 비): T8 full 1.688 → b1 제외
     **1.822**, b6 제외 1.640 ⇒ **가장 약한 block을 빼면 대비가 오히려
     커진다**. Ha8 full 1.340, LOO 1.219–1.375(전 probe r≥0.993).
  3. **[UNAUDITED — 감사자 자신이 이번 턴에 새로 생산, 자기 산출을 자기가
     감사한 셈. 정본 인용 전 별도 확증 필요] Blocking 필터 임계 스윕.**
     `PREFILL_BLOCK_TOK` 1024→512→256→0(keep_frac/pooled-p95/`A_free`-form,
     ms): Ha8 d16 0.979/98.66/**113.51** → 0.827/33.50/**34.63**; Ha8 d54
     0.960/88.01/**107.45** → 0.756/33.01/**35.11**; T8 d16
     0.994/11.60/**28.31** → 0.928/11.44/**11.65**; T8 d54
     0.984/14.01/**15.38** → 0.877/11.38/**11.59**. **무릎(knee)이 없고**,
     임계 0에서 d16-vs-d54 대비가 두 arm 모두 **소멸**(Ha8 0.986, T8
     1.005) ⇒ 임계는 자유 모수가 아니라 **답을 정하는 손잡이**이고 사후
     선택 금지(임계 0은 estimand를 "prefill이 전혀 없던 조용한 순간의
     ITL"로 바꾸고 그 선택이 부하와 상관되므로 선택 편향). ★이 표는
     telemetry·시계정렬을 전혀 쓰지 않아 조건부 분석에 대한 계측 반론에는
     면역이다(그 점만은 강함).

  **권고**: `A_free` **은퇴**(임계 재조정 아님), `A_all` 유지·병기.
  **primary 라벨 = realized partition**(`decode_sms==D`, persisted state
  variable이라 조밀 idle-spin 스냅샷에서도 견고). secondary 라벨(prefill
  overlap, `prefill_overlap_frac(a,b)=구간 중 prefill_active_batch_size>0
  였던 시간 몫`, BLOCK-FREE := ≤0.0)은 872077이 `PDMUX_TRACE_FORCE_
  PREFILL=0`이라 **계측 결손**(prefill-active가 wall의 4.9%인데 스냅샷의
  0.07%): SPLIT 라벨 토큰 중 "overlap-free"로 나오는 비율이 Ha8 d16/d54
  66.3%/38.8%, T8 76.1%/27.1%(물리적으로 SPLIT면 prefill in-flight여야
  하므로 이 불일치분이 곧 계측 결손).

  **재구현 시 갈라지는 3곳**(코드 주석에도 명시): ① `replay_arrivals`는
  sbatch와 bit-identical, ② `load_telemetry`는 `phase=="benchmark"` 필터
  **미적용**, ③ `report_deengagement`는 rng를 전체 중첩 루프에 하나만
  생성해 **루프 순서 의존**(보고값 재현엔 순서 고정 필요; 프로덕션 흡수 시
  `(arm,cell,block,f)`별 seed로 바꾸면 값이 MC 잡음만큼 움직임).

  **(II) [구현 사실 — 성능 주장 아님] `PDMUX_STICKY_PARTITION` 구현 완료,
  correctness gate 통과.** 전부 `src/multiplex/multiplexing_mixin.py`
  (+151/−17, 4 hunks).

  - **되돌림 경로 5곳 전수 처리**: `adjust_stream_groups`:904의 핵심
    되돌림(decode busy·prefill 부재 → plain (0,108))을
    `if not running_batch.is_empty() and (split_prefill_batch or
    sticky_partition_enabled)`로 가드해 sticky ON에선 이 경로에 도달하지
    않음. :906의 decode-**empty** → plain (108,0)은 **의도적 미변경**.
    `event_loop_pdmux`:1010–1012 트리거도 미변경(#2와 일관성 유지 주석).
    `event_loop_pdmux_coord` + `PDMUX_LA_COORD` 조합은 init에서
    `RuntimeError`로 거부(반쪽 sticky 방지). SLO 분기·v7 `_tgt`는 prefill
    span 중에만 도달하므로 미변경.
  - **decode-empty 시 index 0으로 release(hold 아님)** — 이유 3: (a)
    `E1_DECODE_REALIZED`가 decode-active 시간 가중이라 이 구간은 가중치 0,
    (b) 보호할 decode 작업이 없어 D SM을 prefill로부터 놀리기만 함(upside
    0), (c) 경로 #3을 유효하게 유지해 기전의 두 반쪽을 일관 유지. smoke
    실측: decode-empty 스냅샷이 양 arm 모두 `(0,108,0)`(OFF 10812 / ON
    12046) ⇒ **두 arm은 decode-busy 모집단에서만 다르다**.
  - **OFF 바이트 동등**: 유일 변경이 `split_prefill_batch or
    sticky_partition_enabled`이고 flag OFF면 short-circuit으로 patch
    이전과 정확히 동일. `test_off_matches_pre_patch_selector`가 **독립
    재구현한 pre-patch selector**와 3 config × decode_bs{0,1,4,47,48,96} ×
    {prefill 무/유} 전 격자에서 동등 assert. 텔레메트리 코드 미변경.
  - **cudagraph 보존**: `cuda_graph_runner.py`가 `f"{stream_idx}_{bs}"`로
    키잉하고 `capture()`가 모든 stream-group 인덱스를 캡처 ⇒ division
    인덱스 유지 시 캡처된 그래프 재생, eager fallback 없음.
  - **모드 상호작용**: sticky ON은 init에서 `PDMUX_LA_COORD`,
    `PDMUX_SLO_SCHED`, `PDMUX_FIXED_DECODE_SM_FILE`, 비-`fixed`
    `PDMUX_R2_POLICY`, `real_sm_group_num<3`을 `RuntimeError`로 거부.
    허용: `PDMUX_R2_POLICY=fixed`(target 인덱스를 init에서 해석), 무정책,
    `PDMUX_DUAL_WORKER`, `PDMUX_TRUE_DUAL_WORKER`.
  - **텔레메트리 = realized 확인**: `_dual_worker_sync(stream_idx)` →
    `observe_scheduler` → `arbiter.select_partition(stream_index)` →
    `metrics()`가 `arbiter.sm_counts[arbiter.stream_index]` 반환 — CUDA
    스트림을 고르는 바로 그 인덱스이므로 `decode_sms`는 라벨이 아니라
    **decode가 실제로 돈 green context의 SM 수**(그래서 patch 이전 런이
    셀 라벨이 아닌 D108을 92–96% 보였던 것). ⚠️**미재검증 잔여 스코프**:
    "green context를 `create_greenctx_stream_by_value`로 만들면 하드웨어가
    그 SM 수를 실제로 부여한다" 단계는 이 패치가 바꾸지 않았고 재probe하지
    않았다.
  - **correctness gate 결과**: (1) CPU 회귀 sticky OFF **PASS** — 40
    tests(기존 28+신규 12), `sync_engine_tree.sh` 후 manifest SHA-256 일치
    (`multiplexing_mixin.py =
    59eaafb4ac61cc09ad8d28f663c495cf6a0e850435c15b873547b8a6e5a7d20a`,
    동기화 스크립트 변경 불필요). (2) sticky 단위 테스트 12건 **PASS**
    (설치된 런타임에서 로드하므로 sync 실행 여부까지 assert). (3)
    thread-local role patch 정상. (4) **GPU smoke PASS** — job **872800**,
    `amd_a100nv_8`/gpu38, ~9분. Ha8=Zamba2-7B-Instruct, d16
    (`PDMUX_R2_POLICY=fixed`, `PDMUX_R2_FIXED_DSM=16`), server args를
    `e1_m3_control.sbatch`에서 그대로 복사, 두 부팅이
    `PDMUX_STICKY_PARTITION`만 다름. 고정 프롬프트 6개 greedy(`temperature
    0`, `max_new_tokens 48`) → **OFF/ON 출력 6개 전부 byte-identical**.
  - **★realized 관측(게이트가 아니라 관측)**: probe = ShareGPT 100
    prompts, rate 2, seed 1, `PDMUX_TRACE_FORCE_PREFILL=0`,
    `e1_pin_check.py:compute_decode_realized`.

    | arm | `E1_DECODE_REALIZED` | decode-active 히스토그램 |
    |---|---|---|
    | sticky OFF | **0.0839** | D108 66.7s, D16 6.1s |
    | sticky ON | **1.0000** | D16 164.3s (D108 부재) |

    OFF는 기존 동작 재현(872077 Ha8 d16 = 동일 추정기로 0.104–0.127). ON은
    **1.0000, 사전등록 0.90 초과이며 이를 맞추려 튜닝한 것 없음**. 스냅샷
    교차확인: decode-busy 스냅샷이 ON에서 `(idx1,92,16)` 139/139, OFF에서
    `(idx2,0,108)` 146 / `(idx1,92,16)` 8 — ON에서 index 2 진입 없음.
    ⚠️**해석 없는 주의**: decode-active wall time이 두 arm에서 다르다
    (72.8s vs 164.3s) — arm이 서로 다른 파티션에서 decode를 돌리기
    때문이다. **n=1, 미반복, 성능 측정 아님.** ★**구현 완료 ≠ 성능 주장
    성립.** throughput/latency/goodput/`g`에 대한 어떤 진술도 허용되지
    않는다. 이 패치가 주장하는 것은 **`E1_DECODE_REALIZED`가 항등식이기를
    멈추고 진짜 게이트가 됐다**는 것뿐이다.
  - **신규 파일**: `tests/test_sticky_partition.py`,
    `results/sticky_smoke/sticky_smoke.sbatch`,
    `results/sticky_smoke/stksmoke_Ha8_d16_872800_result.txt` + smoke
    아티팩트(telemetry.jsonl 2개, 각 ~14MB, git 미추적 권고).
    `pdmux_context.py`는 미변경(`[(108,0)]+divisions+[(0,108)]` 유지 —
    sticky는 trailing 그룹을 제거하지 않고 회피).

  **(III) sticky 런 사전등록 — 기록하되 `G_LEVER`/`G_FLAT`는 비워 둠.**

  - **(a) 872077 소급 적용의 지위 = DIAGNOSTIC 전용, 판정 금지.** 이
    추정량은 872077을 본 뒤 선택됐으므로 그 데이터로 판정을 채택하면
    confound(re-score vs re-tune)다 — §4.3.8(c)의 RULE_BOUNDS
    forward-only와 같은 논리. 소급 적용은 "estimand 미식별"이라는 **설계
    판정의 근거 자료**로만 쓴다. `m3_conditional.py`가 실행 끝마다 이
    문장을 출력한다.
  - **(b) primary 1개 고정**: `p95(SPLIT tokens)` 비를 primary로 선언,
    나머지(p50/mean/UNSPLIT)는 secondary — 6개 통계량이 있어 선언 없이
    가면 사후 선택(multiplicity)이 열린다.
  - **(c) 게이트 4종**(등록 전 각각이 estimand와 논리적으로 독립인지 재확인
    — 방법론 게이트 #6/#7, `MIN_PA_SNAPSHOTS`가 estimand의 여집합을 셌던
    실수 반복 금지): `ALIGN_R_MIN` 0.95(flag-only) / `E1_DECODE_REALIZED`
    ≥0.90 per cell-block(sticky에서 항등식이 아니게 되므로 비로소 진짜
    게이트) / `AMBIG_FRAC` 상한(872077 관측 0.003–0.014) / `MIN_N_SPLIT`
    cell-block당 하한(872077 관측 1,390–7,045, sticky에서 늘어야 정상).
  - **(d) ★`G_LEVER`/`G_FLAT` = 미결정.** 기존 1.5/1.15는 `A_free`(이중
    극단) 스케일 값이고 새 추정량은 per-token 분위수라 스케일이 다르다 —
    옮기는 것 자체가 사후 재단. 살아있는 논거: 새 추정량이 C2와 같은
    축(per-token ITL 분위수)이므로 `G_LEVER`를 C2 측정범위(**2.36–2.91×**,
    prefill 16 SM 고정·SM16→SM92)에 묶는 것이 원리적으로 정당화 가능하나,
    E1은 D 범위가 16→54로 좁고 **complementary**(prefill=108−D가 함께
    움직임)라 C2 값을 그대로 쓸 수 없다. **결정 근거를 문서에 남긴 뒤
    sticky 런 제출 전에 사전등록해야 할 열린 항목**으로 기록한다.
  - **(e) sticky에서 달라지는 것 2가지 — 미리 등록**: (i) UNSPLIT
    하위모집단이 비거나 매우 작아진다(목적) ⇒ `un_*` 행을 필수 산출로
    요구하지 말고 정의 불가 시 NaN, 판정은 SPLIT 계열로(UNSPLIT이 여전히
    크면 sticky 미적용이므로 `E1_DECODE_REALIZED` 게이트가 먼저 잡음).
    (ii) **primary 모집단이 `SPLIT ∧ BLOCK-FREE`로 이동**한다(sticky
    이전엔 두 라벨이 사실상 중복이지만 이후엔 decode-SM을 prefill 간섭에서
    분리하는 유일한 모집단) — 이 이동을 미리 등록해야 사후 선택이 아니다.
  - **(f) 판별 예측(주장 A 최종 검정, §4.3.9 승계 불변)**: sticky에서
    꼬리가 prefill 주도면 Ha8≈0.92 / T8≈1.85 쪽으로, 주장 A가 옳았다면
    Ha8≈1.6 쪽으로. CI 비중첩 ⇒ **8 block으로 구분 가능**. 블록 증설은
    이 판별 **이후에만** 검토.
  - **(g) 열린 설계 쟁점(claims-auditor 판단으로 남김, 결론 내지 않음)**:
    primary 모집단이 `SPLIT ∧ BLOCK-FREE`로 가면 §4.7.1의 trace-force 금지가
    **구속 조건**이 되는데(prefill overlap 라벨의 계측 결손 때문),
    §4.3.8(h)의 SCHED-only 분해가 trace-force의 *시스템* 효과를 mean
    +0.001(sign 4+/2−)로 재귀속했으므로 **trace-force ON 재허용 여부는
    sticky 설계에서 다시 판단할 사안**이다.

  상세 전문 `results/s8_frontier/DESIGN.md` §4.3.10(estimator)–§4.3.11
  (implementation)–§4.3.12(pre-registration), `reports/CONSENSUS.md`
  §1-27.

## 철회된 가설

- attention/SSM layer별 static resource partition이 보편적으로 유리하다.
- layer-granular switching이 phase-granular allocation보다 유리하다.
- 기존 single-worker dynamic이 best static을 이긴다.
- 과거 simulation/no-CUDA-Graph 결과의 1.37–2.02× layer-aware goodput 향상이
  현재 real-engine 논문 결과다.
- R1 `PDMUX_DUAL_WORKER=1`이 독립 worker architecture를 구현했다.
- ★**(2026-07-28) Stage 0 "운영점 decode SM-무감각(non-binding), ctx≤16k·
  hybrid 전부"** — D108 무경합 앵커가 실은 decode 16 SM이었음이 확인되어
  (C1 CONFIRMED, 코드/telemetry/클라이언트 서명 3중 증거) 철회. 상세는 위
  "Stage 0" 절.
- ★**(2026-08-02) E1 하네스 세션에서 세웠다가 claims-auditor에 반증돼 철회한
  6건**(전부 2026-07-31~08-01 세션 내부 주장, 정본에 확정으로 오른 적은
  없다 — 되살아나지 않도록 여기 보이게 남긴다):
  - **"seed divergence = 도착 draw의 성질"**(C-A **REFUTED**) — 근거로 든
    `arrival_rps`가 측정값이 아니라 seed로부터 RNG replay로 재생성된 값
    (`e1_capacity_scan.sbatch:200-207`)이라 `(seed, n)`만의 결정론적 함수 ⇒
    "5셀 전부 동일"은 **항등식**이고 서버 정보량 0. 진짜 기전은
    `bench_serving.py:1705`의 `random.seed()`를 `datasets/sharegpt.py:98`의
    `random.shuffle`이 소비해 **seed가 프롬프트 집합 자체를 바꾸는 것**.
    seed 비-pooling 결정은 유지, **이유만 교체**(두 seed = 서로 다른 워크로드).
  - **"ITL 구속 rate ⟂ d92 off-cliff(배타성)"**(C-D **REFUTED**) — T8은
    rate 1–32 전 구간·전 룽에서 요청의 ≥93%가 ITL을 통과해 "ITL 구속 rate"가
    **공집합** ⇒ 배타성 명제가 **공허참**.
  - **"그러므로 E1의 판정력이 M8/Ha8/Hs8에 걸린다"**(**NOT-YET-SUPPORTED**) —
    2026-08-01 룽 분류에서 Ha8이 **전 룽 ITL-ALWAYS-BINDING**으로 나와 부분
    반대 증거가 생겼다.
  - **"cap↑이 T8·Hs8의 사다리를 살리고 M8·Ha8을 악화시킨다"** — **철회**.
    cap은 hybrid에서 admission이 아니라 **다른 손잡이**다("방법론 게이트" #5).
  - **"메모리는 한 번도 구속하지 않음"** — **scope 오류로 철회**. batch-cap은
    **T8 하나만** 측정했다.
  - **knee 수치·framing 정정**(이미 커밋 `d1f157a`로 정본 반영, 중복 확인만):
    d92 4.10 → **2.80**, d54 14 → **8.45**; "5셀 동시 off-cliff rate 부재"는
    부정확 → 성립 명제는 **"공통 off-cliff band(≲2–3 req/s)가 ITL 항이
    움직이는 영역과 분리돼 있다"**.
- ★★★**(2026-08-03, 같은 세션 속행) 메인 세션이 세운 "희석 attenuation"
  가설 — claims-auditor REFUTED**(전문은 위 "8B decode-SM 프론티어" 절
  "2026-08-03" 소절 (A)): "`E1_DECODE_REALIZED` 4–19%이므로 `A_free(dD)
  =w_D·A(D)+(1−w_D)·A(108)` 혼합이 872077의 `g`를 attenuate시켰고, Ha8
  보정치는 ≈1.62–1.70이다"는 control-arm reductio(T8 보정 시 21–29×로
  C2를 10배 위반)·de-engagement 직접 실험(w=0에서도 g 1–11%만 이동)·
  "A(108) 셀 무관" 가정의 실측 위반(UNSPLIT-only에서 T8 헤드라인 재현)
  3중으로 반증됐다. 확정으로 오른 적 없는 이 세션 내부 주장이나, 되살아나지
  않도록 여기 보이게 남긴다. **살아남은 것**: engagement가 낮다는 §1-25의
  전제 자체는 견고 — 죽은 것은 **보정**뿐.

## R1 판정

`job_862512`는 **R1 observer-path A/B**다. 기존 scheduler queue/batch의 alias와
logical arbiter를 추가했을 뿐 동일 `event_loop_pdmux`에서 실행됐다. dual arm에만
동기식 JSONL bookkeeping이 있었고 CUDA Graph가 꺼졌으며 server seed가 달랐고
각 arm은 한 번만 실행됐다.

따라서 “4 improved / 0 worse / 3 mixed” 판정과 decode-heavy 성능 차이의
dual-worker 인과 귀속을 철회한다. 모든 scenario는 방향성 관측, 통계·인과
미확정이다. 전체 수치는
[`R1_REANALYSIS.md`](reports/paper/R1_REANALYSIS.md)에
보존한다.

## 현재 구현

- `PDMUX_DUAL_WORKER=1`: R1 observer 재현 전용
- `PDMUX_TRUE_DUAL_WORKER=1`: 두 long-lived host issue thread, role task queue,
  immutable execution context, safe-boundary resource lease를 사용하는 R2 경로
- `PDMUX_TELEMETRY_PATH`: architecture와 무관하게 동일한 비동기 telemetry 사용
- versioned `HybridModelProfileV1`, conservative floor estimator, fixed/generic/
  Hybrid controller가 구현되어 `PDMUX_R2_POLICY`로 engine safe-boundary에 연결됨
- W1–W9 deterministic trace, paired campaign manifest, request token-ITL p95
  goodput와 paired bootstrap 분석 도구가 구현됨

True dual 경로는 module-global PD-mux role을 thread-local `ContextVar`로 바꾸는
tracked patch를 필수로 요구하며, 적용되지 않은 runtime에서는 fail-fast한다.
GPU correctness/performance 검증 전에는 production-ready로 분류하지 않는다.
현재 controller의 online ITL p95는 최근 decode-iteration wall-time window의
추정치이며 request token-level p95는 load generator에서 별도로 계산한다.

### 코드 리뷰 스코프 정정 (2026-07-24)

읽기 전용 코드 리뷰([`reports/r2_decoupling_review_2026-07-24.md`](reports/r2_decoupling_review_2026-07-24.md),
engine-porter, file:line 근거)가 확인한 구조: `PDMUX_TRUE_DUAL_WORKER=1`은
**control-plane dual-worker**다 — 두 host issue thread, role별 task queue,
immutable `ExecutionContext`, thread-local role(ContextVar)만 분리한다.
**data/resource plane은 전면 공유**된다: running batch(`max_running_requests`)는
단일 scheduler 속성이고 완료된 prefill을 같은 running batch로 in-place merge하며,
KV/mamba pool도 단일 객체, SM 파티션도 `SharedGpuArbiter`가 하나의
`stream_index`만 추적한다(92+24=116의 별도 device pool이 아니라 ≤108 단일
coupled index). 확정된 결과 §3의 死因 얽힘이 사는 substrate(공유
running-batch+KV)를 이 구현은 **구성상 깰 수 없다** — 관측될 win/loss는
host-thread overlap(control-plane)에 귀속되며, 별도 device pool disaggregation과
hybrid mamba/SSM state transfer가 필요한 headroom에는 도달 불가하다. 이 두
경로는 코드에 **미구현**이다(state-transfer 경로 전무, mamba conv/ssm state
migration 스캐폴딩조차 없음). 따라서 **Claim D는 "control-plane coupling
감소"로 범위를 축소**한다 — "얽힘을 깬다"는 프레이밍으로 쓰지 않는다. R2는 GPU
correctness gate를 통과한 이력이 없다(`results/r2_eval/` 디렉터리 미생성,
`architecture=true_dual` telemetry 전무). 부가로 admission
latch(`r2_admission_limited`)에 **known-latent 버그**가 코드 근거로 확인됐다:
split batch가 None으로 배수되면 재평가 경로가 없어 latch가 True로 고착되어
prefill admission을 영구 차단할 수 있다(clear 경로 부재) — **사용자 결정으로
현재 수정하지 않고 보류**한다.

## 증거 수준

| Claim | 상태 |
|---|---|
| A. composition/context/load-dependent decode demand (★2026-07-26 Stage 0의 "운영점 decode SM-무감각" rider는 ★★2026-07-28 claims-auditor 감사(C1 CONFIRMED, D108 앵커 무효)로 철회 — 대신 C2(scoped): prefill 16 SM 고정 시 decode ITL SM16→SM92 2.36–2.91×, 4 arm 모델-무관, "8B decode-SM 민감도 측정 노트" 참조. 등급 변경 없음 — 레버 존재만 확립, 정책 이득 근거 아님) | 부분 지지 |
| B. layer-level reconfiguration의 critical-path 손상 | 강한 지지, 현 substrate 한정 |
| C. decode starvation의 TTFT entanglement | running-batch 경로 강함, KV 경로 부분 |
| D. true dual-worker가 coupling 감소 (★2026-07-24 코드 리뷰로 control-plane 범위로 축소, 위 "코드 리뷰 스코프 정정" 참조) | 미검증 |
| E. Hybrid-informed policy가 generic/static보다 우수 | 미검증 |

## 다음 실험 gate

1. 대칭 telemetry의 observer effect가 3% 미만이고 paired CI가 0을 포함해야 한다.
2. 동일 fixed split에서 true dual이 legacy 대비 decode progress/ITL/queue age를
   개선하며 throughput regression이 3%를 넘지 않아야 Claim D를 채택한다.
3. estimator의 95% upper-bound coverage가 95% 이상, under-reservation epoch가
   1% 이하여야 한다.
4. target applicability 영역에서 proposed가 B1/B5보다 paired CI 기준 유의하고
   effect가 3% 이상이어야 Claim E를 채택한다.
5. 벡터1(disjoint conflict-regime escape hatch, CONSENSUS §5-8(c)): **CONFIRMED
   closure (scoped, 2026-07-25)** — g2_0_full → g2_0_hard → g2_0_decliff →
   g2_0_rasweep → g2_0_raconf(pre-registered 24-job 확증 열, 결정 규칙 충족)로
   short-ctx band(scope는 위 "벡터1" 절 참조)에 견고한 disjoint 없음을 최종
   확정. 더 이상의 게이트 없음(트랙 종결) — 남은 방향은 (i) long-context
   재검증(decode floor 상승 영역, 미실행), (ii) §1-20 spatial coupling-tax/
   decoupled substrate(별도 트랙, 아래 항목 2 참조). 어느 쪽도 아직 실험
   설계·게이트가 없다.
6. **논문 positioning(2026-07-25, venue-strategist prior-art 조사,
   [`reports/paper/venue_positioning.md`](reports/paper/venue_positioning.md)
   §0.1)**: 신규성 축은 disaggregation이 아니라 **co-located multiplexing**
   (DuetServe/MuxWise/SGLang-pdmux/Nexus/Bullet 대조) — 부분적 신규성 실증,
   방어 자산은 DuetServe(libsmctrl·Transformer서 동적 승)와의 정량적 상반.
   negative를 (A) green-context 종속(Claim B, 헤드라인 금지) / (B)
   mechanism-independent 후보(Claim A/C, lever-weakness·entanglement)로 분리.
   ★**green-context = 배포 가능한 유일 vendor primitive(CUDA 12.4+)** →
   "libsmctrl 쓰면 되잖아"는 배포 불가 research curiosity로 반박(약점 아님).
   ⚠️**초판의 "cross-substrate serving 이식 make-or-break" 프레이밍은 철회**
   (MPS=정적·프로세스별, libsmctrl=비-vendor·세대귀속 → 이식 불필요·부적합).
   대신 (B)를 **기존 green-context 위에서** 닫는 vendor-substrate 3수: 새 게이트
   = **Transformer-control 대조**(같은 green-context+SLO, drain 상쇄 → 동적
   flip이 모델서 갈리면 hybrid 귀속 식별) + roofline lever-weakness microbench
   (기존 r0c) + 기측정 entanglement 귀속(switch≈0). long-ctx(위 5번 (i))는
   ctx-regime 경계용(별도 질문). 상세는
   [`reports/paper/EXPERIMENT_ROADMAP.md`](reports/paper/EXPERIMENT_ROADMAP.md)
   "벡터2"(TC-series) 절.
7. **long-context Stage 0(L−2) 게이트: ★★철회(2026-07-28, C1 CONFIRMED)** —
   2026-07-26엔 "실행 완료, non-binding"으로 기록했으나, 근거였던 D108 무경합
   앵커가 실은 decode 16 SM이었음이 확인돼(위 "Stage 0" 절) 판정2/판정3이
   철회됐다. **L−2 게이트는 사실상 아무것도 측정하지 않았다** — 따라서
   L−1 이상(SLO 재정의·모델 교체 baseline·시간축/공간축 충돌 스윕)은 "게이트
   실패로 보류"가 아니라 **"게이트 미실행"**으로 되돌아간다(재개 권고 아님,
   판정 부재라는 뜻). 상세
   [`reports/stage0_verdict_2026-07-26.md`](reports/stage0_verdict_2026-07-26.md)
   (원 판정, 철회됨).
8. **8B decode-SM 프론티어 실험 E1 (2026-07-28 지정, 2026-07-29 하네스 구축
   완료·본 스윕 미실행)** — 위 "8B decode-SM 민감도 측정 노트"(C2)가 확립한
   레버가 예산 제약 하에서 net-positive인지 판정하는 gate. 설계: `[108−D,
   D]`(D∈{16,24,44,54,92}) + best-static 대조, 4 arm, **offered-rate 고정**
   (closed-loop 금지), 용량 선측정 후 off-cliff rate 선택, **n≥4**, paired
   bootstrap, TTFT p50/p95/p99 + request-내부 token-ITL p95 + conjunctive
   goodput 보고. **사전등록 게이트 2개**: realized 파티션 점유율 ≥0.80, 파티션
   활성률 ≥0.60(2026-07-29 시간가중으로 정정 — 아래 "방법론 게이트" #4 참조).
   **사전등록 결정규칙**: 어떤 D가 best static을 conjunctive goodput에서 ≥3%
   이기고 paired CI가 0을 배제하면 채택, 아니면 C2는 "ITL 레버는 있으나 예산
   제약 하 net-negative"로 확정한다. **사전등록 SLO(2026-07-29, 사용자 지적으로
   개정)**: 1차 **ITL-p95 = 60ms 고정**(근거는 데이터 적합이 아니라
   `serving_slo_survey.md` chat-class + §1-17 선례), 사다리 {50,60,80}ms
   민감도 병기. 초안의 150ms는 8B 측정 ITL p50이 전 arm·전 D에서 그 아래라
   ITL 항이 non-binding해져 conjunctive goodput이 TTFT-only로 붕괴시키므로
   폐기됨. TTFT SLO는 ≥1 셀에서 binding + 모든 셀 p95로부터 ≥15% 마진, 없으면
   "TTFT 축 ill-posed"로 보고한다. **게이트 #8(동적 컨트롤러 규율의 재스코어
   금지)은 전 셀 `FixedPolicy`인 E1의 사전등록 사다리 재스코어에는 적용되지
   않음**을 명시.
   - **E2**: ctx∈{1024,4096,16384}로 확장, t0 패치된
     `workspace/engine-port/results/s8_scaleup/s0dc_client.py` 사용.
   - **E3**: duty-cycle 2수준(짧은 keepalive vs 긴 keepalive) 설계상 종결 —
     추가 실행 불필요, "짧은 prefill을 자주" 규율만 준수.
   - **E4**: C2b("hybrid 급락=Zamba2 성질")는 현존 체크포인트로 통제된 비교가
     불가능하므로 — 파라미터·형상·tokenizer가 동시에 다름 — **주장 폐기가
     정직한 수순**이다(추가 실험으로 구제하지 않는다).
   **완료**: `results/s8p_prefill/`(prefill 축 SM 민감도) — 2026-07-29 판정,
   정본 인용 금지 유지(claims-auditor 미통과). 요약: 기울기 비 4.74–5.16×,
   탄력도 ε 0.89–0.94, 4 arm 모델-무관(REAL scoped, 미감사). claims-auditor
   반증 대상(위 "열린 긴장" 참조): 게이트 미달 5셀의 strict 재귀속 충분성·
   곡률 크기가 attention FLOP 예측의 5배(기전 미상)·두 게이트가 동일 사건이라
   사전등록 강도가 1개분인 것·`--probe-conc 2` 사전등록 미실행.
   **진행 중(2026-07-29 신규)**: `results/s8_frontier/` 하네스 구축 완료(양축
   판정기·분석기·config 5, seed-per-rep·rep-고정-per-cell 정책), 측정 방법론
   결함 5종(집계 단위) 발견·수정 완료. 하네스 전제 job **867231**(T8 5셀 용량
   스캔, `RATES="1 2 3 4 6 8 12 16 24 32"`×2 seed)·**867298**(`results/
   e1_traceforce/` 관측자 효과 게이트, ABBA n=4)은 세션 핸드오프(2026-07-29)
   시점 PENDING으로 기록됐으나 ★**2026-07-31 doc-steward 갱신 시 `sacct` 재확인
   — 둘 다 COMPLETED**(867231: 21:15:25–22:41:31 / 867298: 21:16:56–21:50:21,
   둘 다 2026-07-29). ★**2026-07-31 정정: "분석/판정 파일 0건"은 오기**이며 두
   job 모두 in-job 분석이 완료돼 있다(위 "8B decode-SM 민감도 측정 노트" 절의
   정정 참조) — 관측자 효과 게이트 **조건부 통과**(본 스윕 force-trace OFF,
   pin 검증만 분리: `DESIGN.md` §4.7.1), 용량 스캔은 **§4.2/§9.1 escalation
   분기 재정의**(knee 첫 교차: d16 12.6 / d24 16.0 / d44 16.0 / d54 8.45 /
   d92 2.80 req/s; 성립 명제 = "공통 off-cliff band ≲2–3 req/s가 ITL 항이
   움직이는 영역과 분리"). 남은 전제는 이 해소이지 두 job의 재실행이
   아니다 — 본 스윕은 (i) 867231 용량 스캔 분석으로 SLO 사다리 확정, (ii) 867298
   분석으로 관측자 효과 게이트 통과 여부(=`PDMUX_TRACE_FORCE_PREFILL` 점화
   여부) 확정 — **(i)(ii) 모두 2026-07-31 완료**. ⇒ **다음 액션은 (iii)
   escalation 해소**(공통 off-cliff band와 ITL 구속 영역의 분리를 어떻게
   처리할지 결정)이며,
   나머지 3 arm(M8/Ha8/Hs8) 용량 스캔은 그 결정이 스캔 설계를 바꿀 수 있으므로
   그 뒤에 제출한다.
   - ★**2026-08-02 상태 갱신 — 위 "그 뒤에 제출한다"는 이미 집행됐다.**
     2026-08-01에 3 arm 용량 스캔(jobs **870295**=M8 / **870296**=Ha8 /
     **870297**=Hs8, 각 100 probe, 오류 0)과 T8 batch-cap 실험(job
     **870301**, {d16,d44}×cap{48,96,192}×4 seed, 24 probe, 오류 0)이
     완료됐다. 결과 요약·수치는 위 "열린 긴장"의 **"2026-08-01 실험 4건"**
     소절에 있으며 ⚠️**전부 claims-auditor 미통과 = 미검증, 인용 금지**다.
     ⚠️단 위 (iii) **escalation 해소는 선행되지 않았다** — 3 arm 스캔은
     그 결정 전에 제출됐고, 결과는 escalation을 **더 넓혔다**(d92 knee 2.80이
     네 arm 전부에서 구속 ⇒ 공통 off-cliff 상한이 arm-무관하게 2.80).
     escalation 해소는 여전히 **열린 항목**이다.
   - ★**본 스윕은 미제출이며, 그 판단 근거는 세 가지가 동시에 성립하기
     때문이다**: (a) 사전등록 사다리 {50,60,80}ms가 as-run 설정에서 **네 arm
     전부 판정 불가**(HEADLINE-ELIGIBLE RUNGS = NONE), (b) 그 as-run 설정
     자체가 **왜곡으로 증명됨**(`--max-running-requests 48`이 ITL·TTFT 두
     축을 반대 방향으로 왜곡), (c) 제외 규칙(d92 knee 2.80, 전 arm)이 **어떤
     동작점에서도 decode-rich 끝을 제거** — C2 레버가 사는 끝. ⇒ **E1은
     `results/s8_frontier/DESIGN.md` §4.3.5(b)가 사전등록한 "설계상 이 질문에
     도달할 수 없다" 분기로 갈 위험이 높다.** 이는 실패가 아니라 **미리
     적어둔 분기**이다. 긴장 A(HE2 vs C2)는 이 경우 E1으로 닫히지 않는다.
   - **다음 액션 후보(미결정, 사전등록 필요)**: (i) `--max-mamba-cache-size`를
     전 arm 공통 상수로 명시 고정해 cap을 순수 admission 손잡이로 되돌리기
     (E1 correctness에 필요, "방법론 게이트" #5), (ii) cap을 데이터 독립적
     규칙("batch가 cap-bound가 아닌 최소 cap")으로 사전등록 파라미터 승격
     — 단 (i)이 선행해야 하고 **arm별 cap 튜닝은 손잡이만 바꾼 SLO 쇼핑이라
     명시적으로 금지**, (iii) 2026-08-01 결과 4건의 claims-auditor 회부.
   - **하네스 결함·사전등록(전부 커밋 완료)**: 버그 #7/#7b/#8/#9 +
     `traceforce` `PIN_CHECK` 인자순서 수정, `DESIGN.md`
     §4.3.5(`ITL-ALWAYS-BINDING`·룽 4분류·on-cliff 셀 제외)·§4.3.6(미등록
     cap 상수·decode duty cycle)·§4.7.1(force-trace를 pin 검증 전용으로
     분리) 사전등록, `results/s8_frontier/decode_duty_check.py` 신설
     (prefill 게이트와 구조적으로 동일한 시간가중 추정량; **의도적으로
     cite-blocking 게이트가 아님** — §5가 게이트 2개를 사전등록한 뒤 데이터를
     보고 세 번째 임계를 더하면 게이트를 데이터에서 고르는 것이 된다).
   - ★**상태 갱신(2026-08-03, 같은 날 2차 속행)**: 위 (iii)은 이미 집행됐고
     (§1-26/§1-27), 이어서 `A_free`가 은퇴하고 조건부 per-token 추정량으로
     estimand가 이관됐으며(위 "8B decode-SM 프론티어" "2026-08-03(2차)"
     소절 (I)), `PDMUX_STICKY_PARTITION`이 구현·correctness gate 통과했다
     (동 소절 (II)). **다음 액션 = sticky 격자 1회 제출**(872077 동일 설계
     8 block, 872077이 non-sticky 대조) — 단 제출 전 **`G_LEVER`/`G_FLAT`
     사전등록이 미결 열린 항목**이다(동 소절 (III)(d)).

실험·통계·fallback의 상세 정본은
[`EXPERIMENT_ROADMAP.md`](reports/paper/EXPERIMENT_ROADMAP.md)다.

## 방법론 게이트 (2026-07-28 신설, 2026-07-29 #4, 2026-08-02 #5·#6, 2026-08-03 #7 추가·#6 사례 추가)

Stage 0/8B de-confound 감사에서 확인된 실패 모드로부터 도출된 3개 항목(1–3),
E1 하네스 구축에서 도출된 상위 원칙(4), 그리고 2026-08-01 캠페인에서 나온
코드 사실 1건(5)과 메타 교훈 1건(6), 그리고 2026-08-03 M3 캠페인에서 나온 항목(7).
`CLAUDE.md`의 기존 8개 게이트에 추가로, 이 정본에 등재한다.

7. ★★★**게이트를 만들 때, 그 게이트가 재는 양이 "게이트가 통과시키려는 조건"과
   논리적으로 독립인지 먼저 증명하라 — 게이트 자신이 #6을 위반할 수 있다.**
   2026-08-03에 `e1_pin_check.py`의 시간가중 pin 게이트가 **항등식**임이 확인됐다
   (telemetry 120파일·77,688 스냅샷에서 `prefill_sms ≠ target ⟺
   decode_running_batch_size == 0`, 양방향 위반 0). 즉 "파티션 제어 실패"로 채점한
   것이 전부 **설계된 decode-empty auto-revert**였다(`multiplexing_mixin.py:773,
   792-794`; `CONSENSUS.md` §1-22가 이미 "의도된 경로"로 기록). 이 게이트는
   **정확성을 강제하려고 만든 것**이고, 그 자신이 게이트 #6을 위반했다.
   ★**역방향 사례도 같은 세션에서 나왔다**: 표본 부족을 막으려 만든
   `MIN_PA_SNAPSHOTS`가 하필 **추정 대상의 여집합**(off-target 스핀이 스냅샷의
   대부분)이라 **핀이 잘 걸린 probe일수록 미측정 판정**을 받았다(UNMEASURABLE
   20/20이 pin 최상위 두 seed에 집중, 최하위 두 seed는 0건).
   ⇒ 실무 규칙: (i) 게이트를 코드로 쓰기 전에 **통과 조건과 측정량의 결합분포를
   손으로 적어보고**, 한쪽이 다른 쪽을 함의하는지 확인한다. (ii) 새 게이트는
   **기존 게이트와 병기**한다(제거하면 이전 보고서가 재현 불가). (iii) 게이트가
   무엇을 **여집합으로** 세고 있지 않은지 확인한다.
   ★**따름정리(별개 발견)**: 같은 감사에서 **decode 축은 게이트가 아예 없었고**,
   셀 라벨의 decode 분할이 decode 작업시간의 **4–19%만 실현**됨이 드러났다
   (`CONSENSUS.md` §1-25). §1-22의 "라벨≠실현" 요구가 **prefill 축에만** 적용돼
   있었던 것 — 축이 둘이면 게이트도 둘이어야 한다.

1. **pin은 policy target이 아니라 realized 파티션으로 검증한다.** telemetry의
   `runtime_snapshot`이 보고하는 `(prefill_sms, decode_sms)`(코드 근거
   `dual_worker.py:608`)를 매 실험에서 재집계해 controller가 지정한 값과
   실제로 일치하는지 확인한다 — 검증 비용은 0이며, 이걸 생략해서 Stage 0의
   D108 앵커가 실은 D16임을 놓쳤다. ★**아래 4번의 특수 사례**(target-vs-realized
   집계 단위 불일치)로 재분류.
2. **파티션 활성률을 사전등록 게이트로 삼는다.** green-context 분할은
   split-prefill 동거 중에만 유효하고, 비면 legacy `adjust_stream_groups`가
   무분할로 되돌아간다 — 활성률이 낮으면 셀 평균이 목표 파티션과 무분할의
   혼합이 된다. 실험 전에 최소 활성률(예: ≥0.60)을 정해두고 미달 셀은 폐기한다.
   ★**2026-07-29 정정 필요**: 이 활성률은 반드시 **시간가중**으로 재정의한다
   (아래 4번 참조) — 스냅샷 **개수** 기반 활성률은 실제값을 16–26× 과소평가할
   수 있다(E1 하네스에서 실측).
3. **keepalive는 짧은 prefill을 자주 넣는 방향으로 설계한다.** 긴 keepalive는
   활성률을 오히려 떨어뜨린다(실측: 0.66–0.93 → 0.32–0.63) — 긴 prefill
   윈도우 동안 decode가 진행되지 않아 시간적으로 분리되기 때문이다.
4. ★★**(2026-07-29) 집계 단위를 먼저 정하고, 그 단위가 추정 대상과 맞는지
   논증하라.** 위 1–3번을 포괄하는 상위 원칙(1번을 특수 사례로 흡수, 지우지
   않음). `results/s8_frontier/` 하네스 구축 중 **집계 단위가 답을 5번 바꿨고
   전부 반대 결론을 낼 뻔했다**:
   1. target vs realized 파티션(위 1번, Stage 0 무효화의 원인).
   2. 게이트 모집단 `prefill_active OR decode_active` vs 조건부 —
      설계상 정상인 decode-only 무분할 윈도우를 pin 실패로 셈: pin **0.029
      FAIL → 0.976 PASS**.
   3. 스냅샷 **개수** 가중 vs **시간** 가중 — `runtime_snapshot`이 이벤트루프
      iteration당 발화해 235ms prefill 스텝과 11ms decode 스텝이 같은 무게를
      가짐: 동시성 **0.015 → 0.25–0.40**(16–26×).
   4. drain 꼬리를 포함한 duration vs 도착 구간만의 duration —
      `achieved_rps` **3.40 → arrival_rps 8.96**.
   5. 스냅샷 vs 에피소드, 그리고 그 안에서 다시 개수 vs 시간 —
      d16 pin_frac **0.583 → 0.847**.

   **따름정리**: 하나의 추정량으로 두 질문에 답하지 마라. "그 파티션에서
   실행됐는가"(게이트, 예: 파티션 점유율/활성률)는 **시간 가중**으로 답해야
   하고, "이 요청의 지연은 어느 파티션 것인가"(귀속)는 **요청별 bracket**으로
   답해야 한다. 후자는 전자의 데이터를 대부분 버리므로(대부분의 스냅샷이
   요청 경계 안쪽이 아니라 사이에 놓임) **게이트에 쓰면 검정력이 무너진다**
   (예: d16 요청-bracket 귀속 n_episodes=8, lower95=0.554 → FAIL — 이유는
   검정력 부족이지 잘못된 SM이 아니다). 상세
   [`workspace/engine-port/results/s8_frontier/`](workspace/engine-port/results/s8_frontier/)
   설계 문서(하네스 내 결함 5종 수정 기록).
5. ★★**(2026-08-02, 코드 사실 — 성능 주장 아님) `--max-running-requests`는
   arm 계열마다 다른 손잡이다.** 근거:
   `sglang/srt/model_executor/model_runner_kv_cache_mixin.py:223-229` —
   `disable_radix_cache ∧ max_running_requests is not None`이면
   `max_mamba_cache_size = max_running_requests`로 설정된다(그 앞 분기
   `:218-222`는 `--max-mamba-cache-size`가 명시된 경우, 뒤의 `else`는 가용
   메모리 ratio 기반). E1/s8 계열 캠페인이 정확히 그 조건이므로 **SSM을
   포함한 arm(M8/Ha8/Hs8)에서 cap은 admission과 mamba state pool 크기를
   동시에** 움직이고, **T8(순수 Transformer)에서는 admission만** 움직인다.
   ⇒ (i) **T8에서 잰 cap 효과는 hybrid로 이전 불가**, (ii) cap을 실험
   파라미터로 쓰려면 `--max-mamba-cache-size`를 전 arm 공통 상수로 **명시
   고정**해 두 축을 분리해야 한다.
   **따름정리 — `kv_mamba_occupancy = 1.0`은 메모리 구속의 증거가 아니라
   항등식이다**(pool 크기 = cap이므로 batch가 cap에 닿으면 정의상 1.0).
   hybrid arm에서 관측된 1.0을 "hybrid는 메모리가 구속한다"의 근거로 쓰지
   않는다. 위 "확정된 결과"의 KV congestion 항목과 함께 읽을 것.
6. ★★★**(2026-08-02) "이 양이 내가 재려는 것과 논리적으로 독립인가"를 먼저
   물어라 — 항등식을 증거로 쓰지 마라.** 위 4번(집계 단위 선확정)과 같은
   뿌리이나 실패 모드가 다르므로 별도 항목으로 둔다. 2026-07-31~08-01
   세션에서 철회된 주장 8건 중 **3건이 정의·항등식을 증거로 착각**한
   것이었다:
   1. `arrival_rps`는 seed로부터 RNG replay로 재생성된 값이라 "5셀 전부
      동일"이 **항등식**(서버에 대한 정보량 0).
   2. `kv_mamba_occupancy = 1.0`은 pool 크기가 cap과 같아서 생기는
      **항등식**(위 5번).
   3. "ITL 구속 rate ⟂ d92 off-cliff"는 ITL 구속 rate가 **공집합**이라
      **공허참**.
   나머지 3건은 **한 arm/셀에서 잰 것을 일반화**한 것이다(batch-cap =
   T8 2셀만). ⇒ 새 지표를 증거로 올리기 전에 **그 지표가 실험 설정으로부터
   해석적으로 결정되는 값이 아닌지** 먼저 확인한다.
   4. ★★★**(2026-08-03) 새 사례 — 항등식에서 파생된 양을 자유 모수처럼
      나누지 마라.** `E1_DECODE_REALIZED`(#7이 확인한 조건부 pin의 결과물,
      4–19%)를 "decode-SM 레버가 `A(D)`에 engagement 비율 `w`만큼만
      반영된다"는 **자유 모수**로 취급해 `A_free(dD)=w·A(D)+(1−w)·A(108)`을
      역산하는 보정 모형을 세웠다가 claims-auditor에 REFUTED됐다 — `w`는
      실은 §1-24(prefill=108−D)가 이미 결정한 duty cycle이라 역산 대상과
      역산 도구가 같은 양이었다. **잡은 순서(재사용 가치)**: (i) 같은
      보정을 레버가 있다고 이미 알려진 **대조군에 적용**해 알려진 값을
      위반하는지 본다(control-arm reductio, T8 corrected g 21–29× vs
      정본 C2 2.36–2.91×로 즉시 사망) — GPU 없이, 기존 872077 텔레메트리
      재사용만으로 수행. 상세 `CONSENSUS.md` §1-26·§3-15, `results/
      s8_frontier/DESIGN.md` §4.3.9.
