# 세션 핸드오프 — 2026-07-27 ~ 2026-07-28

## 이번 세션 요약

전 세션이 미해결로 남긴 **Stage 0 de-confound 도전((a) 잔여 confound vs (b) 음성대조
전제 오류)** 을 GPU 없이 판별하는 것으로 시작해, **Stage 0의 D108 앵커가 실제로는 16 SM에서
돌았다**는 것을 telemetry로 확정했다(= Stage 0 헤드라인 무효). 이어서 사용자 지시로
**7-8B 스케일업**(pure-Mamba2 / additive hybrid / substitutive hybrid / pure-Transformer
4-arm)을 새로 구축·실행했고, 파티션·batch·노출을 모두 통제한 결과 **decode SM 민감도는
실재하며 모델-무관(prefill 고정, SM16→SM92에서 2.4–2.9×)** 이라는, Claim A와 반대 방향의
증거를 얻었다. **정본은 의도적으로 손대지 않았다** — `s0_deconfound/DESIGN.md` §5
사전등록(H1이면 claims-auditor 선행)이 이 질문에 계속 적용된다.

---

## 결정·측정

### 1. Stage 0 파티션 활성시간 재집계 (GPU 불요, 기존 telemetry만)

`results/s0_deconfound/PARTITION_RESIDENCY_STAGE0.md` · 스크립트 3종 동 디렉터리.

- **계측 원리**: PIN_CHECK가 본 것은 `controller_decision.target_decode_sms`(정책 목표).
  `runtime_snapshot`에는 **실현값**(`sm_counts[arbiter.stream_index]`, `dual_worker.py:608`)이
  들어 있다.
- **결과**: D108 arm의 decode-active 샘플 중 **82–97%가 realized 16 SM**. 원인은
  `CFG[108]=pdmux_d16.yml` + 정책 OFF → legacy `adjust_stream_groups`
  (`multiplexing_mixin.py:726-733`)가 `manual_divisions [92,16,**0**]`의 threshold 0 때문에
  `decode_bs >= 0` 상시 참 → 무조건 stream_idx 1 = (92,16).
- **client 수치가 독립 확인**: `D108/D16 = 0.992–1.001` (9/9 셀)인데 D92는 3.6× 빠름.
  D108이 진짜 무분할이었다면 D92보다 빨라야 한다 ⇒ **같은 파티션을 두 번 측정한 값**이고,
  Stage 0은 108 끝점을 **측정한 적이 없다**.
- **(a)/(b) 판정**: Stage 0을 증거집합에서 빼면 865098과의 충돌이 사라지고, 음성대조 M의
  기울기가 confound 구조가 정반대인 두 캠페인에서 재현(2.42–2.47× vs matched-batch
  2.32–2.39×) ⇒ **(b) 쪽**. "SSM decode는 O(1)이라 SM-bound 불가"의 O(1)은 context 길이에
  대한 것이었다.

### 2. 7-8B 스케일업 (신규 캠페인 `results/s8_scaleup/`)

jobs 865289/865297/865305(스모크·backend 대조) · 865311/865312(v1) · 865493(v2) · 865533(v3).
결과·한계 전문 = `results/s8_scaleup/FINDINGS_8B_2026-07-28.md`.

- **arm**: M8 Mamba-Codestral-7B(pure Mamba2 7.3B) / Ha8 Zamba2-7B(**additive** hybrid) /
  Hs8 Nemotron-H-8B(**substitutive**, M24/−24/*4) / T8 Qwen2.5-7B.
  pure-Mamba2 7-8B는 Codestral이 **유일한 실용 경로**(state-spaces 2.7B 상한 ·
  nvidia/mamba2-8b=Megatron · falcon-mamba-7b=Mamba-1 sglang 미지원).
- **핵심 결과**(파티션×batch 동시 매칭, 구간 귀속, ctx1024, prefill 16 SM 고정,
  batch=12 고-n): **SM16/SM92 = M8 2.91× · Hs8 2.58× · T8 2.36× · Ha8 2.85×(b9)**
  ⇒ **decode SM 민감도 실재 + 모델-무관**. batch 1–16 전 구간 안정, 곡선 단조.
- **"hybrid 급락"은 hybrid 성질이 아니라 Zamba2(additive) 성질**: matched batch·SM에서
  **Hs8/M8 = 0.86**(substitutive가 pure-Mamba보다 **빠름**), **Ha8/M8 = 1.57–1.69**.
  Zamba2는 mamba 층을 대체하지 않고 그 위에 shared attn을 **더하는** 구조
  (weight-traffic 1차추정 M 5.40 / T 6.17 / H 7.66 GB, 비 1.42×와 같은 자릿수).
- **backend 대조(job 865305)**: NemotronH는 triton 금지(`server_args.py:1959`),
  Zamba2-7B는 flashinfer에서 cudagraph capture 사망 ⇒ 어떤 hybrid도 두 backend 불가.
  대조를 T8/M8로 이전 → **attention 경로 +2.3%**(Qwen 15.06→15.40ms),
  **pure-Mamba +0.3%**(음성대조 성립). Hs8이 낀 비교에 이 offset이 따라붙는다.

### 3. 이 세션에서 **철회한 것** (기록물에도 반영됨)

- **"공간 분할이 시간 공유를 3.2× 이긴다"** — batch confound였다(NP 셀 batch 12–16 vs
  d92 3.7). matched batch에선 SM108 무분할이 SM92 분할보다 근소하게 빠르다.
  ⇒ 이 데이터는 PD-mux 공간분할 이득에 대해 **아무 말도 하지 않는다**.
- **"Zamba2가 가장 SM-민감(4.44×)"** — 미통제 셀 평균 아티팩트(활성률 최저 + batch 공변).
  통제 후엔 오히려 **최저(2.54–2.85×)**.
- 헤드라인 비율을 SM16→**SM108**로 인용한 것 — SM108(=np)은 분할 자체가 없는 조건이라
  prefill 할당도 같이 바뀐다. **인용은 SM16→SM92**(prefill 고정 구간)로 한다.

### 4. 이 측정의 지위 (오독 방지 — 다음 세션이 반드시 알 것)

de-confound 설정은 `[prefill, decode, idle]` = `[16,16,76]/[16,24,68]/[16,44,48]/[16,92,0]`로
**저-D 셀이 SM 대부분을 일부러 놀린다**. 즉 **decode 쪽 등량곡선**이지 정책 비교가 아니다.
실제 정책은 `prefill_SM + decode_SM ≤ 108` 제약을 받으므로 판단 대상은 프론티어
**ITL(D) vs TTFT(108−D)** 이고 **그건 미측정**이다. 또한 정본의 실패 기전(얽힘: decode
굶김→ITL↑→batch 정체→prefill admission 차단→TTFT 폭발)은 decode-ITL 지표에 원리상 안 보인다.
⇒ **레버의 존재만 확립했고, 레버를 움직여 goodput이 나아진다는 근거는 아니다**(게이트 #1).
레버가 있다는 사실이 HE0(동적 < best static)를 되살리지도 않는다 — HE0의 死因은 레버 부재가
아니라 positioning + 얽힘이었으므로, 오히려 negative의 **설명**이 바뀔 뿐이다.

---

## 코드·문서 변경 (전부 커밋됨, push 없음)

| commit | 내용 |
|---|---|
| `300b286` | `src/models/mamba2.py`에 `backbone.embeddings.`(복수형) 매핑 추가 — Codestral이 **랜덤 임베딩으로 서빙될 뻔한 버그**. + `scripts/models/make_mamba2_hf_sglang.py`(safetensors 헤더에서 차원 도출하는 shim 생성기). `.gitignore`에 생성 프롬프트 제외 규칙. |
| `df0c0f8` | `results/s0_deconfound/PARTITION_RESIDENCY_STAGE0.md` + 재집계 스크립트 3종. |
| `5cb65f9` | `results/s8_scaleup/` 전체(하네스 4 sbatch, config 8, 분석기 3, FINDINGS·NOTES 2). |
| `55b5bd3` | 전 세션 handoff(2026-07-27) 커밋. |

메모리: `scale-8b-sm-sensitivity.md` 신규 + `stage0-deconfound-contested.md` 갱신(판별 완료)
+ `MEMORY.md` 인덱스 2줄 갱신. **정본(PROJECT_STATUS/CONSENSUS/CLAIM_EVIDENCE_MATRIX)은
의도적으로 미수정.**

---

## 열린 항목 / 다음 세션 시작점

1. **claims-auditor (블로커, 최우선)** — `PARTITION_RESIDENCY_STAGE0.md` +
   `FINDINGS_8B_2026-07-28.md`를 함께 건다. 사전등록(`DESIGN.md` §5)상 이걸 통과해야 정본
   수정에 착수할 수 있다. 특히 반증받아야 할 3가지:
   (i) ctx1024 한정을 4096으로 확장해도 2.4–2.9×가 유지되는가,
   (ii) **활성-구간 표집 편향**(파티션이 활성이던 구간만 표본에 들어감)이 기울기를 만들 수 있는가,
   (iii) 이 ITL 기울기가 실제 SLO-goodput 차이로 전이되는가.
2. **정본 개정(2 이후)** — doc-steward. 대상: `PROJECT_STATUS.md` Stage 0 절 ·
   `CONSENSUS.md` §1-21/§3-9/§5-6 · `stage0_verdict_2026-07-26.md` ·
   `longcontext_trace_plan.md` §0.6/§2/§6 · `CLAIM_EVIDENCE_MATRIX.md` Claim A —
   전부 "D16 vs D108(무경합) = 1.00±0.01"에 근거하고 있다.
3. **8B 프론티어 캠페인(다음 실험)** — idle SM 없이 `[108−D, D]` 스윕 + TTFT·ITL·
   conjunctive goodput 동시 측정. **설계 전제 2개**(이번에 실측으로 배움):
   활성률이 D와 상관되고(coupled d16 0.128 vs d92 0.774) 그때 decode batch도 같이 움직인다
   ⇒ **활성률을 사전등록 게이트로 걸고**, closed-loop가 아니라 **offered-rate 고정
   워크로드**(정본의 변화-trace 형태)로 설계할 것.
4. ctx4096 귀속 데이터 부재 — 필요하면 패치된 클라이언트로 재측정(1회 ~2.5h).

---

## 미완·주의

- **어떤 8B 결과도 아직 정본 아님.** claims-auditor 미통과 상태에서 논문·정본에 인용 금지.
- **클라이언트 복사 출처 주의**: `t0_monotonic_s` 패치는 `results/s8_scaleup/s0dc_client.py`
  에만 있다. `s0_deconfound/s0dc_client.py`는 865098이 실제로 돌린 **역사적 기록**이라
  일부러 그대로 뒀다 — **새 캠페인은 s8_scaleup 쪽을 복사할 것**.
- **표의 희박한 칸 인용 금지**: `n>=50` 필터는 drain 꼬리 과도 상태를 못 거른다
  (예: Ha8 batch=11의 10.49×는 아티팩트). 대표값은 batch 12–16 슬라이스에서 뽑는다.
- 실행 중 job 없음. 미커밋 변경 없음. push 안 함.
- 새 방법론 게이트 후보 3개(§ 다음 캠페인에 반영 필요):
  (1) **pin은 policy target이 아니라 realized 파티션으로 검증**,
  (2) **파티션 활성률을 사전등록 게이트로**(green-ctx 분할은 prefill 동거 중에만 유효),
  (3) **긴 keepalive는 역효과**(실측: ~2000토큰×4가 짧은 프롬프트×8보다 활성률 낮음 —
  긴 prefill 윈도우 동안 decode가 안 돌아 시간적으로 분리됨) ⇒ **짧은 prefill을 자주**.
