# 추가 이점 경로 — 진행 상황 보고서

작성일: 2026-06-17 · 대상: prefill-layer-alloc (v2 핵심 가설 기각 이후)
용도: "이대로 종결"([중단 보고서](project_closure_report.md)) 대신 **추가 이점이 실재하는 경로**가 있다면 그 진행도/다음수/예상이득/킬-기준을 정리. 각 경로는 독립 출구다.
관련: [v2_report](../workspace/characterization/reports/v2_report.md) · [검수 체크리스트](review_checklist.md)

상태 범례: **○ 미착수** · **◐ 부분진행** · **● 코드완료/측정만 필요** · **✦ 결정적**

---

## 요약 (우선순위순)

| # | 경로 | 상태 | 예상이득 | 비용 | 한 줄 |
|---|------|:--:|---|---|---|
| 1 | **7B 회귀** | ◐ ✦ | 결론을 크기-무관으로 격상 **또는** 라인 부활 | 중(SXM4 E2+E5 ×2모델) | config 있음, 미실행 |
| 2 | **prefill/decode overlap 재프레이밍** | ● | 음성 라인 → 양성 기여(논문) | 저(분석+ssm_full 검증) | 이미 ~2× 측정됨 |
| 3 | **진짜-prefill(ssm_full) E5** | ● | 헤드라인 신뢰도 확정 | 저(셀 일부 재측정) | 검수 A2와 동일 |
| 4 | **비대칭의 비-할당 용도** | ○ | 부수 기여(작은) | 저~중 | 새 RQ 필요 |

---

## Path 1 — 7B 회귀 ✦ (결정적)

**왜 이게 추가 이점인가.** v2 결론은 전부 SLM(1.2~3B)이고, "분할 무용"이 *작은 커널이 full 풀에서 잘 겹치기 때문*일 가능성이 명시적으로 열려 있다(v2_report §6 한계, §7.4). 7B는 이 단일 변수를 닫는다:
- **음성이면:** "큰 커널이 SM을 포화시켜도 분할 무용" → 결론이 **모델 크기 무관**으로 격상. 음성 특성화 논문의 결정타.
- **양성이면(분할이 7B에서 이득):** **본 연구 라인 부활** — "SLM에선 무용, scale에서 의미"라는 훨씬 강한 thesis.

→ 어느 쪽이든 가치가 높고, **닫는 데도·계속하는 데도 동일하게 필요한** 유일한 데이터점이다.

**현재 진행도(◐).** config는 존재하나 미사용:
- `shared/configs/models.yaml`: `zamba2`(Zamba2-7B, n_heads 112), `falcon_h1`(Falcon-H1-7B), `nemotron_h` 엔트리 존재. **line 88: "the 7B entries above are kept but unused."**
- E0(해석적)는 7B를 즉시 예측 가능(GPU 불필요): `grid_sat_sm = batch·⌈tokens/chunk⌉·n_heads`. 7B는 n_heads↑(112)라 grid가 SM을 더 일찍·강하게 채움 → "큰 커널이면 비대칭이 더 빨리 죽는다"가 사전 예측.

**다음 수 (구체).**
1. **E0 먼저(GPU 불필요, 즉시):** 7B로 `e0_analytical` 재실행 → grid가 batch=1에서도 108 SM 초과하는지 확인. 초과하면 "분할 여지 없음"이 해석적으로 이미 예측됨 → 측정 우선순위 판단.
2. **E2 7B (SXM4):** `submit_size_sweep.sh e2`에 7B 2모델 추가. attn-ssm sat_sm gap이 SLM 대비 커지는지/유지되는지.
3. **E5 7B (SXM4):** `submit_size_sweep.sh e5 -- --context-lens 1024 2048 4096 8192 16384`. **단 [검수 A2]대로 `--prefill-layer ssm_full`로** (7B는 GEMM이 더 크므로 microbench 왜곡이 SLM보다 심함).
4. 7B는 메모리·시간이 크므로 batch 격자를 {1,8,32,128}로 축소해 잡 한도 내에서.

**예상 결과(사전).** E0 예측상 7B는 batch≥1에서 grid가 SM 초과 → 비대칭 headroom이 SLM보다 *작을* 가능성 → 분할 무용이 **강화**될 공산이 큼. 하지만 큰 커널의 launch/occupancy 거동이 다를 수 있어 측정 가치 있음.

**킬-기준.** E5 7B에서 `two_stream/green_ctx`가 **어떤 셀에서 ≥1.05**(green이 5%+ 우위)면 → 라인 부활, Path 1을 본 연구로 승격. 전 셀 <1.0이면 → 음성 결론 크기-무관 격상 후 **종결**.

---

## Path 2 — prefill/decode overlap 재프레이밍 (음성 → 양성 출구) ●

**왜.** 프로젝트의 유일한 양성 측정값을 *기여*로 전환. "layer-type 분할은 무용(음성)"과 "prefill+decode co-schedule이 분할 없이 ~2× (양성)"를 묶으면 **완결된 메시지**가 된다: *"하이브리드 serving에서 자원을 나누지 말고 겹쳐라."* 별도 큰 측정 없이 발표 단위가 선다.

**현재 진행도(●, 측정 거의 끝).**
- E5에 측정 완료: pf=ssm 행 1.8–2.04×(@db8,ctx4096), window 2.04(db8)→1.15(db256), dec=ssm은 context 1k→16k 평탄 ~2.0×. `results_v2/e5/serving_coexec_*.csv` 4모델.
- overlap이 **roofline 상보성이 아니라 duration matching**(비슷한 길이 두 커널이 SM 풀에 함께 들어감)이라는 메커니즘 해석도 §4.6-A에 있음.

**다음 수.**
1. window가 닫히는 batch를 모델별로 정량화(§G1 recommended_action의 "collapse batch"). widened decode sweep(→512)로 db256 이후 곡선 확정 — **현재 PENDING 잡(770988)이 바로 이 데이터**.
2. [검수 A2] ssm_full 진짜-prefill에서도 2× overlap이 유지되는지(GEMM 포함 시 overlap이 더/덜 되는지).
3. dec=attn은 long context로 decode가 벽시간 지배 시 overlap↓ — balance 조건을 명문화.

**예상이득.** 분할 음성 + co-schedule 양성 = characterization 논문 1편 분량. 운영 권고(MPS/multi-stream)는 즉시 실용.

**킬-기준.** ssm_full에서 overlap이 1.2× 미만으로 무너지면 → 양성 주장 약화, Path 3 결과에 종속.

---

## Path 3 — 진짜-prefill(ssm_full) E5 검증 ●

**왜.** Path 1·2·헤드라인 음성 결론 셋 다 **동일 미검증 가정(microbench=진짜 prefill)** 위에 서 있다. 이 한 측정이 세 경로 전부의 신뢰도를 동시에 올린다. [검수 A2]와 동일 작업이므로 **닫기 경로에서도 어차피 해야 함** → 가장 비용대비 효율 높은 추가 작업.

**현재 진행도(●).** 코드 경로 존재(`--prefill-layer ssm_full`, v2_report §6). 측정만 필요.

**다음 수.** E5 최고 셀(pf=ssm×dec=ssm) + db8/db256 양 끝 + ctx{1k,16k}만 ssm_full로 재측정(전수 불필요). green_ctx vs two_stream 정성 3결론 유지 확인.

**예상이득.** 헤드라인을 microbench에서 **진짜 prefill로 승격** → 발표/종결 모두 방어 가능.

**킬-기준.** ssm_full에서 green_ctx가 two_stream을 이기는 셀 발생 → 헤드라인 재검토(사실상 Path 1로 합류).

---

## Path 4 — 비대칭의 비-할당 용도 ○

**왜.** 비대칭(attn이 더 많은 SM에서 포화)은 *할당 손잡이*는 아니지만 다른 곳에 신호로 쓰인다(v2_report §7.3): (a) kernel fusion/튜닝 — ssm가 작아 일찍 포화 → fusion 후보, (b) 모델↔하드웨어 매칭 — 작은 GPU에 ssm-heavy 배치, (c) layer별 quant/offload 우선순위.

**현재 진행도(○).** 아이디어만. 새 RQ·새 측정 필요.

**다음 수(착수 시).** (a)가 가장 구체적 — ssm in/out_proj + scan fusion의 sat_sm/latency 개선을 1모델에서 측정. 단 **별도 프로젝트 규모**.

**킬-기준.** 본 라인 종결 후 별건으로만 검토. 종결 결정에 영향 주지 않음.

---

## 종합 권고

1. **즉시(GPU 불필요):** Path 1의 E0 7B 예측 + Path 2의 widened window 분석(PENDING 잡 결과).
2. **1회 SXM4 측정:** [검수 A2] = Path 3(ssm_full) — 닫든 계속하든 필수. 여력 되면 Path 1의 E2/E5 7B를 같은 잡 배치에 동봉.
3. **분기:**
   - 7B·ssm_full 모두 음성 → Path 2로 **재프레이밍 후 종결**(양성 기여 + 크기-무관 음성).
   - 7B에서 green_ctx 우위 셀 발견 → **Path 1을 본 연구로 승격**(라인 부활).
4. Path 4는 종결 후 별건.

> 한 문장: **"추가 이점은 분할을 더 파는 데 있지 않고, (1) 7B로 음성을 크기-무관으로 못박거나 (2) co-schedule 양성을 기여로 전환하는 데 있다."**
