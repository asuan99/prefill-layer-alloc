# Stage 0 (L−2) + Transformer 대조 — 실험 설계 (2026-07-25, 초안)

작성: 이번 세션. 사용자 요청("A에 대한 Transformer 대조 실험 설계"). **초안** — §5 기술
리스크(cudagraph+green-context 핀 양립, Transformer 모델 통과, 대조 모델 선정)는
engine-porter 확인 후 확정. 확정 결론은 CONSENSUS/PROJECT_STATUS에만.

관련: [longcontext_trace_plan.md](longcontext_trace_plan.md) §6 L−2 · [venue_positioning.md](paper/venue_positioning.md)
§0.1(Risk 2 대체 게이트) · r0c 하네스 `workspace/engine-port/results/r0c/decode_knee_vs_ctx.sbatch`.

---

## 0. 한 줄

**하나의 decode-only microbenchmark로 두 질문을 동시에 닫는다**: (Stage 0) 운영점
(cudagraph-ON)에서 hybrid long-ctx decode가 SM-binding해지는가? + (Transformer 대조)
그 binding/non-binding이 **모델(mamba 비중) 귀속**인가 substrate 아티팩트인가?
Transformer arm이 **lever-weakness 귀속 실험이자 positive control**(계측기가 binding을
탐지할 수 있음을 증명 → hybrid 평탄 = 진짜 null이지 눈먼 계측 아님)을 겸한다.

## 1. 동기 (두 트랙이 공유하는 단일 게이트)

- **Stage 0 게이트**(longcontext §6): micro(no-cudagraph, r0c)의 decode SM-민감도 6.5×
  (108→16 SM)가 **운영점(cudagraph-ON)에서 사는가**. 이 아크를 두 번 뒤집은 게 "micro ≠
  serving"이라, long-ctx 전체(H_L4 시간축·H_L5 공간축)의 전제가 이 하나에 걸린다. 평탄이면
  long-ctx 충돌 가설 死, 벡터1/HE0가 ctx-무관으로 강화. binding이면 lever 부활 후보.
- **Transformer 대조의 역할**(venue_positioning §0.1 Risk 2 대체): "hybrid에서 동적 lever가
  약하다"를 substrate가 아니라 **모델**에 귀속시키는 값싼 식별. green-context drain은 두
  모델에 동일하게 걸려 **상쇄** → decode SM-민감도가 모델에서 갈리면 그 차이는 mamba 비중
  귀속. libsmctrl 이식(비-vendor·세대귀속) 없이 vendor-substrate에서 Risk 2를 닫는 축.

## 2. 설계 (2-arm decode-only 스윕)

| 인자 | 값 |
|---|---|
| **모델 arm** | **(H) hybrid** = Zamba2-2.7B (attn 소수 + mamba 54층) · **(T) Transformer** = 순수 attention ~3B (§5-3 확정) |
| ctx | {4k, 8k, 16k} (+ 옵션 256/1k = short-ctx null 확인, r0c와 연속) |
| decode-SM (green-ctx 핀) | {16, 44, 108(full)} |
| 운영점 | ★**cudagraph-ON**, **pdmux stream-group green-ctx capture 경로**(§5 정정 — r0c in-forward 핀 아님) |
| 부하 | **decode-only** (prefill 경합 0 → knee가 깨끗). conc 32, output 32 tok |
| 측정 | **wall ITL** (mean + p50/p95). ZBLT 내부 per-type 분해 아님(cudagraph가 step을 캡처하므로 wall만) |
| 고정(상쇄) | 두 arm 동일 substrate·cudagraph·conc·output·**capture 메커니즘**. **모델·ctx·decode-SM만 변동** |

★**하네스 정정(engine-porter, §5)**: r0c의 `PDMUX_FIXED_DECODE_SM_FILE` in-forward 핀은
cudagraph replay가 forward를 우회해 **런타임에 死**(eager 전용). 운영점 측정은 **pdmux
green-ctx capture**(`--enable-pdmux` + cudagraph-ON + `manual_divisions`에 decode-SM 열거 +
`PDMUX_R2_POLICY=fixed`)로 재설계 — 이 경로가 CLAUDE.md의 실제 운영점(4모델서 pdmux가 fused
이김)이라 **Stage 0가 오히려 더 충실**해진다(micro 도구가 아니라 서빙 운영점 자체를 잼).

- 모델 크기 정합: Zamba2-2.7B vs ~3B Transformer = 파라미터 대략 동급(절대 ITL 비교가 아니라
  **각 모델 내부의 SM-민감도 *형태*를 비교**하므로 크기 정확 매칭은 2차. 형태=ITL(SM) 곡선의
  기울기).
- ctx>4096(Zamba2 학습 ctx)은 `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`로 rope 외삽
  (타이밍-only, 생성 품질 무관 — r0c와 동일 규율, longcontext §0.5-E).

## 3. 예측 (사전 등록 — 측정 전)

| ID | arm | 예측 | 반증되면 |
|---|---|---|---|
| **S0-H1** | H | short-ctx(256–4k): decode ITL이 SM에 **둔감**(mamba O(1) 지배) | mamba가 실은 SM-민감 = lever 모델 자체가 틀림 |
| **S0-H2** | H | long-ctx(16k): decode ITL이 SM에 **민감해짐**(attn 비중↑) — *단 운영점서* | 평탄 유지 = 운영점서 decode 절대 non-binding → **long-ctx 트랙 전체 死, HE0 ctx-무관 강화** |
| **S0-T1** | T | **모든 ctx**에서 decode ITL이 SM에 **급민감**(attn KV-scan이 decode 항상 지배) | Transformer도 평탄 = **계측기가 binding을 못 잡음(positive control 실패)** → 하네스 무효, Stage 0 재설계 |
| **S0-C** | H vs T | short-ctx서 **T는 민감·H는 둔감**(대조 최대) → long-ctx로 갈수록 H가 T에 수렴 | H가 short-ctx서도 T만큼 민감 = mamba SM-비민감성 반증(정본 §1-3 위협) |

★**positive control 논리**: S0-T1이 성립해야(=Transformer는 민감) S0-H1의 "H 평탄"이
**진짜 model property**임이 선다. T도 평탄이면 계측이 눈먼 것 → null 해석 불가. 이게
Transformer arm이 대조이자 sanity인 이유.

## 4. 게이트 / 결정 규칙

- **S0-H2 = 운영점 binding (long-ctx서 H의 ITL(SM) 기울기가 유의미)** ∧ **S0-T1 성립(positive
  control OK)** → **게이트 통과**: long-ctx서 lever 부활 실재, L−1(SLO 정의·용량) 진행. Risk 2
  모델-귀속 다리 확보(H는 ctx로 T에 수렴 = mamba 비중이 lever 세기를 결정).
- **S0-H(전 ctx 평탄) ∧ S0-T1 성립** → **게이트 실패이자 강한 결과**: 운영점서 hybrid decode는
  ctx 무관 non-binding = **동적 lever 근본 부재**, long-ctx 공간/시간 트랙 死, **HE0/벡터1이
  ctx-무관으로 강화**(정본 반영감). positive control이 살아있으니 이건 아티팩트 아님.
- **S0-T도 평탄(positive control 실패)** → 계측/운영점 설정 문제(cudagraph 캡처가 SM 핀을
  무효화하는 등, §5-1) → 하네스 정정 후 재측정. **null 해석 금지.**

## 5. 기술 리스크 (engine-porter 확인 완료 — file:line 근거)

1. ★**r0c in-forward 핀 × cudagraph = 死 (설계 블로커, 해결됨).** `PDMUX_FIXED_DECODE_SM_FILE`은
   매 step forward 내부에서 파일 read + per-layer green-ctx 전환(`nemotron_h.py:653-656,691-712`)인데
   **capture 가드가 없다**. cudagraph replay는 forward를 재실행 않고 순수 graph replay
   (`cuda_graph_runner.py:1161`; forward는 capture 시 1회만 `:1011`) → boot 시 `full`로 캡처된 뒤
   파일 쓰기가 **무시**되어 스윕 knob이 런타임에 死. r0c가 `--disable-cuda-graph`였던 이유.
   ⇒ **"r0c 하네스 + cudagraph-ON"은 원리적으로 불가.**
   **해결 = pdmux stream-group green-ctx capture 경로**(`cuda_graph_runner.py:810-817`): capture가
   `enable_pdmux`면 whole-decode 그래프를 green-ctx decode stream `sg[1]` 위에 캡처, replay는
   stream_idx 인덱싱(`:1158`). = **whole-phase 단일 파티션**(step 내내 불변 = CLAUDE.md 유일 허용
   lever와 정확히 일치). Stage 0 재설계: `--enable-pdmux` + cudagraph-ON + `manual_divisions`에
   decode-SM {16,44,108} 열거 + decode-only + `PDMUX_R2_POLICY=fixed`로 stream_idx 고정.
   piecewise-cudagraph는 대안 아님(핀 루프에 piecewise 가드 없음).
2. **핀은 hybrid-only → Transformer arm도 재설계 필요 (같은 해결로 수렴).** decode-SM 핀 코드는
   `nemotron_h.py`/`zamba2.py`/`granitemoehybrid.py`에만 존재, `qwen2/qwen3/llama`엔 **부재** →
   Qwen/Llama는 in-forward 핀을 조용히 무시(항상 full-SM). **그러나 pdmux green-ctx capture 경로는
   model-agnostic**(green-ctx가 capture 래퍼 수준 `:813-816`, forward generic, occupancy도
   mamba-tolerant `multiplexing_mixin.py:274`) → **리스크 1 해결책이 Transformer arm까지 동시 커버.**
   in-forward 핀이 아니라 이 경로로 두 arm 모두 재설계.
3. **로컬 캐시 pure-Transformer 없음 → 사전 fetch 선행.** `hf_cache/hub/`엔 hybrid 연구모델만
   (Nemotron-H-8B/Falcon-H1/Zamba2/Granite-4). 유일 pure-attn 항목 Qwen2.5-0.5B는 **메타데이터만
   (safetensors 부재)**. `HF_HUB_OFFLINE=1`이라 런타임 fetch 불가 → **로그인 노드서 size-matched
   ~3B(Qwen2.5-3B ~3.1B/ctx32k, 또는 Llama-3.2-3B ~3.2B/ctx128k) 사전 다운로드 필요**(0.5B는
   2.7B hybrid와 크기 비대칭이라 부적합).

### ★선결 = GPU correctness 스모크 (성능 주장 전 필수)
pdmux green-ctx capture로 **decode-SM 스윕 config(manual_divisions 열거)를 cudagraph-ON에서
캡처 성공**하는지는 코드-경로-존재까지만 확인됨, **GPU 실행 미검증**. Stage 0 본실험 전
**부팅+캡처 스모크**(N개 decode-SM 그래프 캡처 성공 + decode ITL 정상 출력) 1회로 게이트.

## 6. 측정 규율 (CONSENSUS §3 상속)

- decode-only microbenchmark라 정책 결론이 아님 → n은 arm×ctx×SM 조합당 소수 rep로 충분하나
  ITL p95를 위해 conc×rep 샘플 확보. **길이 fingerprint 기록**(prompt 토큰 실측).
- **변수 하나씩**: cudagraph만 켜고 나머지는 r0c와 동일하게(대조군 무결성). 두 arm 간엔 모델만.
- 절벽 회피: decode-only라 goodput 절벽 무관(ITL 연속량 직접 측정) — Stage 0가 값싸고 깨끗한 이유.
- 결과는 `workspace/engine-port/results/stage0_xctrl/`에.

## 7. 산출물과 다음 단계

- **Stage 0 통과** → longcontext L−1(SLO 정의·용량) → L0(모델 교체 baseline) → L1/L3/L3s.
- **Transformer 대조 결과**는 Risk 2 모델-귀속(venue_positioning §0.1)의 1차 증거. 서빙-레벨
  static-vs-dynamic Transformer 대조(HE0 귀속의 full 버전)는 **후속 별도 실험**(이건 decode-SM
  민감도 대조까지만 — lever-weakness 축).
