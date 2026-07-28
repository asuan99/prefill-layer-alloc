# Stage 0 (long-ctx L−2) 게이트 — 최종 판정 (2026-07-26)

> ★★★**HISTORICAL / 2026-07-28 판정2·판정3 철회.** claims-auditor가
> `workspace/engine-port/results/s0_deconfound/DESIGN.md` §5 사전등록 게이트를
> 집행해 **C1 CONFIRMED**: 아래 §3의 "D108 무경합 앵커"는 실제로는 decode
> 16 SM이었다(legacy auto-path 코드 버그로 `manual_divisions=[92,16,0]`의
> threshold=0이 항상 stream_idx 1=(92,16) 선택 + realized telemetry 79–96%
> (92,16) 동거 + `D108/D16` 클라이언트 서명 0.992–1.001인데 D92가 3.4–3.6×
> 빠름 = 108이 92보다 느릴 수 없다는 물리적 모순). **판정 1(CONFOUNDED)만
> 생존**하고 판정 2(NULL)·판정 3(게이트 non-binding)은 철회됐다. §5의 "3중
> 삼각검증" 서술도 무효(무경합 앵커·음성대조 모두 실제로는 고장 상태였다).
> **현재 정본은 [`PROJECT_STATUS.md`](../PROJECT_STATUS.md) "Stage 0" 절과
> [`CONSENSUS.md`](CONSENSUS.md) §1-21이다** — 이 문서는 원 판정의 이력
> 보존용으로 남기며, 아래 본문은 철회 전 시점 그대로 유지한다.

작성: 2026-07-26 (doc-steward, result-analyst rigorous 판정 기록). 사용자 결정: 아래
증거로 결론 확정(완전 de-confound 재측정은 실행하지 않음 — §4에 명시).

관련: [`longcontext_trace_plan.md`](longcontext_trace_plan.md) §0.5(D)·§6 L−2(게이트
정의) · [`stage0_transformer_control_design_2026-07-25.md`](stage0_transformer_control_design_2026-07-25.md)
(실험 설계·사전 등록 예측 S0-M/S0-H/S0-T) · [`CONSENSUS.md`](CONSENSUS.md) §1(HE2 decode
non-binding, r0c decode-floor micro-측정) · 원자료
[`../workspace/engine-port/results/stage0_xctrl/`](../workspace/engine-port/results/stage0_xctrl/)
(jobs **864230**[H/T, ctx4k/8k/16k]·**864601**[M, ctx4k/8k/16k], PIN_CHECK 전부 PASS).

**정본 위계상 지위**: 이 문서는 `reports/` tier(CONSENSUS와 같은 층위 산하 서브 문서)이며,
결론은 [`PROJECT_STATUS.md`](../PROJECT_STATUS.md)와 [`CONSENSUS.md`](CONSENSUS.md)에
반영된다(아래 "반영" 참조). 이 문서 자체가 아니라 그 두 정본이 충돌 시 우선한다.

---

## 0. 한 줄

**Stage 0 게이트 = non-binding.** 운영점(cudagraph-ON, green-context pdmux)에서 decode
ITL은 decode-SM(16→108, 6.75×)에 **무감각**이다 — pure-Transformer(Qwen2.5-3B)·
pure-Mamba(Mamba2-2.7B)·hybrid(Zamba2-2.7B) **전부**, ctx 4k/8k/**16k** 전부에서.
`longcontext_trace_plan.md` §6이 사전 등록한 게이트 판정표의 **"게이트 실패이자 강한
결과"** 분기가 실현됐다: long-context 충돌 가설(H_L4 시간축·H_L5 공간축)은 이 regime에서
전제(decode floor의 운영점 상승)를 얻지 못해 붕괴하고, **HE0/벡터1(동적이 best-static을
못 넘음)은 ctx-무관으로 강화**된다.

---

## 1. 실험 설계 요약

3-arm coupled-운영점 스윕(설계 전문은 `stage0_transformer_control_design_2026-07-25.md`):

| 인자 | 값 |
|---|---|
| 모델 arm | **M** = pure Mamba2-2.7B(음성 대조, SSM-only) · **H** = Zamba2-2.7B(hybrid 타깃) · **T** = Qwen2.5-3B(양성 대조, pure-attn) |
| dtype | 전부 bf16 |
| ctx | {4k, 8k, 16k} |
| decode-SM | coupled 핀 {16, 44, 92} + **108(no-split reference, prefill 경합 0)** |
| 운영점 | cudagraph-ON, green-context pdmux capture, `--keepalive-prefill`(coupled: sub-108 파티션 활성화) |
| 부하 | conc 32, output 32 tok, REPS=3 |
| 검증 | 매 arm `PIN_CHECK`(telemetry `target_decode_sms`가 지정 SM을 실제로 지배) — 전부 PASS |

## 2. 통합 ITL 표 (p50 mean, ms; D16 / D44 / D92 / D108)

| arm | ctx4k | ctx8k | ctx16k |
|---|---|---|---|
| T (pure-attn) | 17.8 / 10.1 / 8.7 / 17.8 | 18.4 / 11.0 / 9.6 / 18.3 | 25.3 / 13.8 / 11.6 / 25.3 |
| M (pure-Mamba) | 19.3 / 9.9 / 7.8 / 19.2 | 17.0 / 8.8 / 7.0 / 16.8 | 16.3 / 8.2 / 6.6 / 16.3 |
| H (hybrid) | 45.6 / 20.4 / 13.6 / 45.5 | 45.4 / 20.2 / 13.3 / 45.2 | 46.8 / 19.8 / 13.1 / 46.7 |

절대 레벨 H > M ≈ T는 per-token decode 비용 차이(hybrid가 무거움)이며 SM 민감도가 아니다.

## 3. 판정

### 판정 1 — coupled ITL(D) 스윕은 CONFOUNDED (CONFIRMED)

D16→D44→D92 곡선이 decode-SM binding이 아니라 **prefill 경합/batch-entanglement
아티팩트**를 재고 있다는 것이 3중 삼각검증으로 확인된다.

1. **D108 앵커**: D108(108 SM decode, prefill 경합 0)이 D16(16 SM)과 9셀 전부에서
   **≤0.5% 동일**하다. decode SM을 6.75× 늘려도 ITL 개선이 없다. 최속점은 D92(최대
   108이 아님)로 **비단조**.
2. **음성 대조 M**: pure Mamba2는 decode가 O(1) recurrent라 원리상 SM-bound일 수
   없는데도 full 2.46×/2.42×/2.47×(ctx별) "민감도"를 H와 동형으로 보인다 — 곡선을
   만드는 것이 decode-SM이 아니라는 직접 증거.
3. **5× wall 시그니처**: 최속점 D92는 prefill을 16 SM으로 굶기는 지점이다 —
   워크로드가 직렬화(wall 5.0–5.4× 붕괴)하고, decode ITL 하락은 SM 가속이 아니라
   de-batch(작아진 running batch가 decode를 더 빨리 도는 것) 결과다.

⇒ 예비분석에서 본 "T < M < H 순서(2.05 < 2.46 < 3.35, ★반증 이전 관측)"는 binding의
후보 순서가 아니라 **confound가 그대로 재현**된 것이다(M이 T와 H 사이에 앉음 =
decode-SM binding으로는 설명 불가). 이건 **measurement-design confound = coupled
entanglement**(CLAUDE.md 방법론 게이트 #4/#7 부류: 변화 trace·baseline 분산 선측정
규율과 같은 계열의 함정 — 여기서는 "공변 축을 분리하지 않은 스윕").

### 판정 2 — 살아남는 clean 신호 = NULL (CONFIRMED)

유일하게 de-confounded인 대조는 **D16 vs D108**(prefill 경합 0인 두 점 비교, coupled
공변이 걸리지 않는 유일한 쌍)이다. 결과: **1.00 ± 0.01, 3 arm × 3 ctx 전부**.

⇒ **운영점서 decode는 16→108 SM에 무감각(non-binding)** — pure-Transformer·
pure-Mamba2·hybrid 전부, **16k ctx까지**. r0c(no-cudagraph micro에서 관측한 mamba
SM-불변)를 확증하는 동시에, **hybrid·서빙 in-situ·16k로 확장**한다.

### 판정 3 — Stage 0 게이트 = non-binding (설계 §4 "게이트 실패이자 강한 결과" 분기)

- long-ctx 충돌 가설(H_L4 시간축·H_L5 공간축, `longcontext_trace_plan.md` §2)은
  필요한 decode-floor 상승을 이 regime에서 얻지 못한다 → **붕괴(이 regime서)**.
- **HE0/벡터1이 ctx-무관으로 강화**된다(short→16k 전 구간에서 decode가 비-lever임을
  확인). CONSENSUS의 HE2(운영점 decode 비-binding)·r0c(no-cudagraph decode-floor
  micro-측정)와 정합·확장 관계다 — **아크의 반전이 아니라 강화**다.

---

## 4. ★필수 scope/caveat (overclaim 방지)

- **regime 한정**: {M/H/T 2.7–3B, triton attn+mamba, cudagraph-ON green-context
  pdmux, ctx ≤16k, 이 coupled 하네스, one-shot 32-conc burst}. **더 큰 모델·>16k·다른
  substrate는 미측정.**
- coupled 스윕의 **"민감도 magnitude"는 confound로 측정 불가**하다 — 결론은
  **방향(non-binding = SM-무감각)만**이며, clean 하위신호(D108-앵커·M-control·D92
  비단조)에서 도출됐다. **완전 de-confound**(prefill-SM 고정 + steady-state rate +
  occupancy 통제) 재측정은 **미실행**이다 — 사용자가 현 증거로 결론 확정을 결정했다.
  이 미측정 상태를 명시한다.
- **설계 문서와의 관계**: `stage0_transformer_control_design_2026-07-25.md` §4의
  사전 등록 결정 규칙은 원래 raw ITL(D) 곡선으로 S0-M(음성 sanity: M 평탄)/S0-T1(양성
  sanity: T 급민감)을 판정하도록 짜여 있었다. **coupled 원곡선만 보면 S0-M sanity는
  실패한 것처럼 보인다**(M이 평탄이 아니라 T·H와 동형의 곡선을 보임). 그러나 그 실패
  자체가 판정 1의 근거(M이 SM-bound 불가능한데도 곡선을 보인다 = 곡선이 confound)이며,
  **D108 무경합 앵커라는 사후 보정 지표로 재구성한 sanity test(D16≡D108)는 3 arm
  전부 통과**한다. 결과적으로 최종 판정은 설계 §4가 예측한 "S0-H2 반증" 분기
  ("평탄 유지 → long-ctx 트랙 전체 死, HE0 ctx-무관 강화")와 **일치**한다 — 설계의
  전제(sanity)는 원안이 아닌 보정된 형태로 충족됐다.

---

## 5. ★방법론 교훈 (CLAUDE.md 게이트에 추가할 항목)

**coupled ITL(D) 스윕은 decode-SM(D)과 prefill-SM(108−D)을 공변시켜 confounded되기
쉽다.** decode-SM을 낮추면서 동시에 prefill-SM을 높이는(또는 그 역인) 하네스는, 관측된
곡선이 decode-SM 자체의 효과인지 그 반대쪽에서 늘어난 prefill 경합/de-batch의 효과인지
구분하지 못한다. 이번엔 **음성 대조**(원리상 그 축에 binding될 수 없는 arm — pure-Mamba
decode)와 **무경합 앵커**(공변이 0으로 붕괴하는 특수점 — D108 no-split reference)의
조합이 confound를 identify했다.

⇒ **정책/측정 주장에는 음성 대조가 필수**다. 이 프로젝트의 아크는 이미 두 번 뒤집힌
전력이 있다(micro↔serving, sim↔engine). 이번엔 M-arm(음성 대조)이 세 번째 반전을 막았다
— T/H만 봤다면 "M이 중간에 낀 이상한 순서"를 못 알아채고 confound를 binding으로
오독했을 것이다.

---

## 6. 반영 (정본 전파)

- [`PROJECT_STATUS.md`](../PROJECT_STATUS.md) — 확정된 결과에 운영점 decode SM-무감각
  (16k·hybrid까지 확장) 추가, "다음 실험 gate"에 Stage 0 게이트 결과 갱신.
- [`CONSENSUS.md`](CONSENSUS.md) §1 — 새 확정 행 + §3 방법론 교훈 + §5 open items(long-context
  전환 항목) 갱신.
- [`longcontext_trace_plan.md`](longcontext_trace_plan.md) §6 — L−2 행을 "완료·게이트
  실패(non-binding)"로 갱신, L−1 이상은 이 regime에서 보류.
- [`../reports/paper/CLAIM_EVIDENCE_MATRIX.md`](paper/CLAIM_EVIDENCE_MATRIX.md) — Claim
  A(lever-weakness) 증거에 운영점 16k 확장 추가(판정 등급 불변, 부분 지지 유지).
