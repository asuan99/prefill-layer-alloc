# R-시리즈 재정립 — 실행 상태 보고

작성: 2026-07-06 · 실행자: Claude Code · 대상 프로젝트: `prefill-layer-alloc` (engine-port)
정본 보고서: [`sm_policy_report.html`](sm_policy_report.html) (v3) · 이전본 보존: `sm_policy_report_v2.html`

> **한 줄 결론:** T1(R0a) clean async 재측정은 v2 §07 은퇴 결론과 **정합**(분기 **(a)**). layer-aware는
> Zamba2-2.7B in3600/out32 clean async에서 **전 rate agnostic 이하**(rate2부터 goodput 0). 게이트 통과 →
> T2/T3/T4 실행 완료. 모순(분기 b)·불안정(분기 c) 없음.

라벨 규약: 아래 수치는 별도 표기 없으면 **measured**. goodput은 **derived** =
`good / duration`, 여기서 `good = #{req : TTFT ≤ 3.0s ∧ mean(ITL) ≤ 0.06s}` (프로젝트 표준 SLO).

---

## T0 — 사전 확인 (완료)

- **v2 확인**: `sm_policy_report.html` §07에 "Chasing the cause took four passes" 문단 존재(구 line 472). → v2 확정.
- **프로토콜 원본 스크립트**:
  - NemotronH clean async 템플릿 = `triage/p1_6_bench_one.sbatch` (jobs 831609–831612). `python -m sglang.bench_serving` async.
  - 4-모델 clean async 템플릿(진화형) = `triage/p1_7_bench_one.sbatch` (model/backend/ctx 인자화; Zamba2-27b 이미 832701/832702 측정).
  - Zamba2 구프로토콜 = `triage/p1_4_zb_serving.sbatch` (서버 플래그·모델경로 참조원). **클라이언트 = `bench_client.py`(GIL 60-thread urllib) = 교체 대상**.
- **정책 구동 플래그 (스크립트+소스 확인, 추측 없음)**:
  - `fused`: pdmux 미사용. `--attention-backend triton --disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache --mem-fraction-static 0.82 --max-running-requests 48 --context-length 4096`.
  - `agnostic`(v1): 위 + `--enable-pdmux --pdmux-config-path pdmux_a100_smoke.yml --chunked-prefill-size -1 --disable-overlap-schedule`. (`PDMUX_FIXED_DECODE_SM_FILE` 미설정 → 균일 예약)
  - `layer_aware`: agnostic 플래그 + env `PDMUX_FIXED_DECODE_SM_FILE=<"layer_aware" 파일>`, `PDMUX_LA_FLOOR_SM=16`. **Zamba2는 `PDMUX_LA_RESERVE` 불요**(레이어타입 내재 예약; `sglang_engine_dev/.../models/zamba2.py` L394–403 `_la and not isinstance(layer, Zamba2HybridLayer)` → mamba층만 floor로 환원, hybrid attn층은 base 유지). p1_7의 `M-`는 Zamba2엔 no-op.

### T0에서 발견한 중요한 사실 (task 전제 정정)

- **§07 헤드라인 표의 Zamba2 행(3.24→0.00 등)은 이미 clean async 데이터**다 — P1.7 jobs **832701(agn)/832702(la)**,
  in2000/out96 ([`triage/p1_7_layeraware_validation.txt`](../triage/p1_7_layeraware_validation.txt)와 정확히 일치). 구 GIL-client 아님.
  → task T2-4의 "§07 Zamba2 행 = 구클라이언트" 전제는 **사실과 다름**. §07은 이미 clean.
- **구 GIL-client가 실제로 남아있던 유일한 곳 = §04 Zamba2 패널** (in3600/out32, `p1_4_zb_serving_831541.txt` 등, rates 6/12/18 과포화, TPOT-SLO 0.12s hack).
  → 따라서 **T1은 §04를 위한 in3600/out32 clean 재측정**으로 스코핑. §07(in2000/out96, 4모델 동일 워크로드)은 값 변경 없이 provenance만 정정.

---

## T1 (R0a) — Zamba2-2.7B clean async 재측정 (완료)

**프로토콜**: `results/r0a/r0a_zamba2_bench.sbatch` (신규). 클라이언트 `python -m sglang.bench_serving`
(async, random-ids). Zamba2-2.7B, triton, ctx4096, **in3600/out32**(구측정과 동일), num-prompts 120,
rates **1/2/3/4/6**({1,2,3,4}+포화직상 6). 정책 fused/agnostic/layer_aware × **2 reps**. SLO TTFT≤3s, TPOT≤60ms.

**잡 ID**: rep1 — agnostic **834914** · fused **834915** · layer_aware **834916**. rep2 — agnostic **834927** ·
fused **834929** · layer_aware **834928**. (전부 COMPLETED, boot_ok, SANITY "Paris" 정확.)

**측정 결과 — goodput@SLO (req/s), rep1/rep2 (measured; goodput=derived):**

| rate | fused | agnostic | layer_aware |
|---|---|---|---|
| 1 | 0.789 / 0.809 | **1.109 / 1.195** | 0.815 / 0.805 |
| 2 | 0.274 / 0.274 | **2.065 / 2.066** | 0.000 / 0.000 |
| 3 | 0.000 / 0.000 | **0.669 / 0.624** | 0.000 / 0.000 |
| 4 | 0.000 / 0.000 | **0.314 / 0.314** | 0.000 / 0.000 |
| 6 | 0.000 / 0.000 | **0.225 / 0.225** | 0.000 / 0.000 |

보조 지표(measured, rep1): agnostic TPOT p50 ~40ms 전 rate 평탄(SLO 내), 고 rate서 TTFT가 binding(rate3 p50 7.0s).
layer_aware TPOT p50 rate1 55ms→rate2 129ms(SLO 돌파) + TTFT rate2 5.3s. fused TPOT rate2 88ms→rate3 303ms.

**산출물** (`workspace/engine-port/results/r0a/`):
`r0a_summary.csv`(30 rate-rows, 헤더 포함 31줄) · per-run `zamba2_async_{policy}_r{rate}_rep{n}.csv`(30개) ·
per-policy JSONL(raw ttfts/itls) · 서버/벤치 로그 · sbatch. 전 수치 `measured` 라벨, goodput 산식은 위 명기.

### T1 판정 → **분기 (a)** (은퇴 확정)

- layer_aware는 **전 rate에서 agnostic 이하**. 저부하(rate1) 동률 이하(la 0.81 vs agn 1.15 — la TPOT floor가 높아 60ms SLO 더 미스),
  rate2부터 **완전 붕괴(goodput 0)**. → v2 §07("la ≤ agnostic, 모든 하이브리드서 은퇴")과 **정합**.
- **분기 (b) 아님**: la가 어느 rate서도 agnostic을 이기지 못함. 구클라이언트 §04의 "la가 TTFT서 우위" 주장도 clean에서 **역전**
  (la TTFT p50 rate2 5.3s ≫ agnostic 1.2s). ⇒ 결론 문서 임의수정·중단 조건 미발동.
- **분기 (c) 아님**: rep1↔rep2 분산이 정책 간 격차보다 **압도적으로 작음**(agn rate2 2.065 vs 2.066; la rate2 0.000 vs 0.000).
  안정적.

---

## T2 (R0c) — 보고서 v2 내부 모순 수정 → v3 (완료, 분기 a 조건 충족)

원본 `sm_policy_report_v2.html`로 보존. `sm_policy_report.html`을 v3로 수정:

1. **서두 "The one relation" 블록**: "layer-aware … the only policy safe on both models" **제거** →
   §07 granularity 결론으로 교체("트레이드는 step 단위서만 성립 = agnostic이 이미 하는 것; per-layer는 그 아래라 포착 불가 → la never beats agnostic").
2. **§06 표 layer-aware 행**: High-load `≥ agnostic*`(win) → **`≤ agnostic — (D) granularity`(lose)**. "Wins when…" 셀 → 은퇴 사유(step granularity 하한, Zamba2 "이상적 케이스"서 최악).
3. **§04 Zamba2 패널**: 구클라이언트 TTFT 차트(rates 6/12/18) → **T1 clean goodput@SLO 차트**(`ttft_zb`→`good_zb`, rates 1–4, means+run dots, ymax 3.0로 좌 패널과 정합). "Disregard this panel" 삭제. 서술 = agnostic 압승·la 최악·구클라 TTFT우위 역전.
4. **§04 sec-sub / "honest caveat" 콜아웃**: "triton이 전 정책 TPOT 170–205ms floor" **정정** — 그건 GIL-client 아티팩트. clean에선 agnostic TPOT ~40ms·실 goodput; la/fused만 130ms↑. 순위 = agnostic > fused > layer-aware.
5. **§01 layer-aware 정책 카드**: 부제 "the synthesis" → "the synthesis — retired (§07)", psrc에 "refuted in serving" 추가(top-down 독자 오독 방지).
6. **Sources**: Zamba2 serving = clean(§07 832701/832702; §04 R0a 834914–916 +reps 834927–929). 구 831166/831146/831541은 "superseded old-client"로 명시. **Caveats**: "Zamba2 serving still from the old client / async re-measure pending" **삭제**. co-schedule 미측정 caveat 유지. py3.14 no-cudagraph caveat = 절대값만 이동·순위 불변으로 명확화.
7. **차트 데이터**: `good_zb` JS 블록에 실측 rep1+rep2 평균+run dots 임베드(placeholder 없음). JS brace/paren/bracket 균형 확인, svg-id ↔ 렌더콜 parity 확인, `ttft_zb` 잔재 0.

**§07 표는 값 미변경** (task T2-4 문자 그대로는 미이행): 이유 = §07 Zamba2 행은 **이미 clean async**(832701/832702, in2000/out96)이고,
4모델을 **동일 워크로드**로 비교하는 표에 §04의 in3600/out32를 주입하면 비교 설계가 깨짐. T1 in3600/out32는 §04에 반영, §07은 provenance/caveat만 정정. (자동 재작성 대신 이 판단을 여기 기록.)

---

## T3 (R3) — supersession 라벨링 (완료)

내용 삭제 없이 배너·주석만 삽입(반전 이력 보존):

| 문서 | 조치 |
|---|---|
| [`reports/layer_aware_benefit_report.md`](../../../reports/layer_aware_benefit_report.md) | 상단 SUPERSEDED 배너 |
| [`reports/FINAL_SUMMARY.md`](../../../reports/FINAL_SUMMARY.md) | 상단 배너 + §10 "한 줄 결론"에 정정 1블록(실엔진 실증 완료·결론 반전) |
| [`reports/framework_comparison.md`](../../../reports/framework_comparison.md) | 상단 배너(4-way sim 표 무효화 명시) |
| [`reports/p1_3_nemotronh_layer_aware.md`](p1_3_nemotronh_layer_aware.md) | 상단 배너 + §2 "이상적 케이스" 프레이밍 정정 주석(민감도 반전: mamba가 SM-민감, attn 아님) |
| [`reports/figure_index.md`](../../../reports/figure_index.md) | 상단 배너 + summary_dashboard·layer_aware_result·layer_aware_size_trend 행에 "sim prediction; superseded (§07)" 표기 |

배너 요지: sim 예측 la/agnostic 1.37–2.02×는 실엔진 serving으로 기각. 정본 = sm_policy_report.html §07. 문서는 가설·이력 계층으로만 유효.

---

## T4 (R2 일부) — 인덱스·마킹 (완료)

1. **실험 인덱스** [`reports/experiment_index.md`](../../../reports/experiment_index.md): 신규 §H(engine-port serving 재측정) 추가 — R0a 잡 ID·조건·산출 경로 등록 + sim 결론 supersede 포인터. (sim §C–G 이력은 보존.)
2. **DEPRECATED 마커** [`triage/DEPRECATED_gil_client.md`](../triage/DEPRECATED_gil_client.md) 신규 생성 — GIL-client(`bench_client.py`)와 그 산출(Zamba2 831146/831166/831541, NemotronH p1_4_serving 계열)을 사유(GIL 60-thread urllib·과포화·TPOT 아티팩트)와 함께 마킹, 대체본(async) 명시. **원본 데이터 삭제·이동 없음.**
   - 배치 판단: task T4가 "구클라이언트 산출 디렉토리에 마커 생성"을 명시 → 구파일이 있는 `triage/`에 비파괴적 문서파일로 생성(“신규 산출→results/r0a” 규약은 측정 아티팩트 대상; 마커는 문서). 파일이 어떤 잡을 deprecate하는지 정확히 스코핑(triage 전체 아님).

---

## 준수 사항 확인

- 구클라이언트 재사용 없음(전부 `sglang.bench_serving` async). 과포화 다점 측정 없음(rates 1–6, 결론은 sub-saturation 기준).
- ncu/nsys 미사용. CUDA graph 복구·py 환경 변경·sglang 버전 변경 없음.
- 기존 결과 CSV·로그 수정·삭제·이동 없음. 신규 측정 산출은 전부 `results/r0a/` 이하. v2 보고서 `_v2`로 보존.
- 한국어 논문 LaTeX(S3/S4) 미수정(범위 밖).

## R0b (후속, 2026-07-07) — 사용자 비판 대응: graduated per-type la 직접 측정

사용자 지적: 은퇴시킨 la가 실제 아이디어(per-type graduated)가 아닌 **2단계 이진 근사**(strawman 우려)이고
agnostic over-provisioning을 미서술. → graduated 구현(`PDMUX_LA_SM_MAP`, dev-tree+src 미러) + decode-SM knee 스윕.

- 측정: 미조율 graduated는 <84서 붕괴, ≥84 정상. ⚠️**초기 판정 "knee≈84/잉여無/agnostic over-provisioning 아님"은 오류 → 사용자 재지적으로 철회.** `PDMUX_LA_SM_MAP` 구현이 decode를 **독립 green-ctx**에 고정 → pdmux prefill 파티션과 **미조율→겹침→경합**(binary la와 동일 결함=four-pass C). ★**결정적 대조**: agnostic decode=coordinated pdmux 쌍(decode 34-54/prefill 상보 74-54)=**42ms** vs 같은 54SM 미조율 green-ctx=**121ms** ⇒ "84 붕괴"=조율 오버헤드지 decode SM 요구 아님. **잉여 실재**(agnostic prefill에 최대 74SM 동적 회수). ⇒ R0b는 미조율 구현이 짐을 재확인할 뿐 **개념(coordinated per-type) 반증 못함**. **coordinated per-type la=event_loop 수술 필요·미구현·미검증**((D) granularity는 논증). 실전 agnostic 권고 유지하나 **"la 근본 열위"는 미증명**. 상세 [`results/r0b/R0b_graduated_results.md`](../results/r0b/R0b_graduated_results.md)(정정 배너).
- 보고서: §07 콜아웃을 **정정**(strawman→coordination catch; "no surplus" 철회; "any implementation" 완화; coordinated per-type=미검증 명시). Sources에 R0b 잡(835044–835209) 추가.
- 근거: `pdmux_context.py:134`(각 그룹=상보 배타 쌍), `multiplexing_mixin.py:104`(decode=stream_group[1]), `zamba2.py:343`(내 override=독립 green-ctx).

## 미완/후속 (차단 아님)

- co-schedule 정책은 여전히 직접 미측정(v2·v3 공통 caveat 유지).
- 완전-조율 layer-aware(decode window ∥ 상보 prefill 파티션)는 미구현 — R0b가 더 근본 이유(knee~84라 잉여 자체가 없음) 제시; §07 (D) granularity는 백스톱 논증(별도 event_loop 수술 트랙).
- R0b knee는 Zamba2 in3600/out32 한정; 타 모델·regime knee는 별도.
- py3.14 no-cudagraph는 Zamba2 decode 절대 TPOT를 높임(순위엔 무영향; graph-capable env는 별도 트랙).
