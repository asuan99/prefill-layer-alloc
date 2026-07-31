# `prefill-layer-alloc` project status

최종 갱신: 2026-07-31 (진행 상태 갱신만, 결론 개정 아님 — `results/s8p_prefill/`
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
  미실행**. 용량 스캔(job **867231**, T8, 5셀)·관측자 효과 게이트(job
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
  - ★★**미등록 하네스 상수가 ITL 축을 단독 결정한다 — `--max-running-requests
    48`**(`e1_capacity_scan.sbatch:143`, `e1_sweep.sbatch:178`). `DESIGN.md`에
    **단 한 번도 등장하지 않는다**(grep 0건). 증거: telemetry의
    `decode_running_batch_size`가 d16/d24/d44/d54 전부 정확히 48에서 절단
    (d92만 38 — prefill admission이 먼저 막혀 cap 미도달 ⇒ **d92의 ITL은 다른
    셀과 비교 불가**), 그 시점 `kv_occupancy = 0.024`(2.4%, 자원 강제 아님),
    d16 ITL-p95가 rate 12→32에서 50.2–50.6ms로 완전 평탄. ⇒ 셀별 ITL "천장"
    {d16 50.6 / d24 37.9 / d44 26.2 / d54 23.9 / d92 19.1}은 **모델 성질이
    아니라 설정 성질**이며, "60ms 도달 불가"의 scope가 달라진다.
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

실험·통계·fallback의 상세 정본은
[`EXPERIMENT_ROADMAP.md`](reports/paper/EXPERIMENT_ROADMAP.md)다.

## 방법론 게이트 (2026-07-28 신설, 2026-07-29 #4 추가)

Stage 0/8B de-confound 감사에서 확인된 실패 모드로부터 도출된 3개 항목(1–3).
`CLAUDE.md`의 기존 8개 게이트에 추가로, 이 정본에 등재한다.

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
