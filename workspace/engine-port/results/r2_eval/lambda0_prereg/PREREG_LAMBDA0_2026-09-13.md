# 사전등록 — 캠페인 **0단계**: λ\*(용량) 측정, shape별 × 기준 arm

작성 2026-09-13 (메인 세션), **GPU 미지출 상태에서 고정**. 추정 **0.75–0.95 GPU-h**
(이 트랙 0.43 → 약 1.2–1.4). 사용자 결정 3건이 선행했다: **모델 = `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base`**,
**백엔드 = flashinfer**(NemotronH에서 triton은 엔진이 거부), **1차 범위 = Claim D + P3**.

## 0. 왜 0단계인가

캠페인 워크로드는 전부 **λ\*의 분수**로 정의된다(W1 0.60 · W3 0.80 · W8 0.90 · W9 1.10 ·
W2/W4는 phase당 요청 수 = `round(0.80·λ*·intensity·30)`). 그런데 `generate_campaign.sh:10`의
λ\*는 **측정값이 아니라 기본값 `4`**다 — 프로젝트 방법론 게이트 **#6("용량 먼저 측정")의 직접
위반**이고, λ\*가 틀리면 "near saturation"·"overload"라는 **라벨 자체가 틀린다**.

## 1. λ\*의 정의 (사용자 결정, 2026-09-13)

> **λ\*(shape) = 기준 arm B1(legacy 루프, PD-mux fixed D44)에서 그 shape의 throughput 포화율.**
> 워크로드의 분수는 "**B1 용량의 n%**"를 뜻한다.

- **왜 arm을 고정하는가**: λ\*는 arm마다 다르다 — 같은 모델·같은 shape에서 **D16 0.933 vs D44 0.675
  vs D92 0.187 req/s(5×)**가 이미 측정돼 있다(job 905835). 기준 arm을 정하지 않으면 "0.80·λ\*"가
  무엇의 80%인지 정의되지 않는다. B1은 B4(true_dual fixed)가 비교당하는 **그 베이스라인**이므로
  기준으로 자연스럽다. B4가 그보다 잘하거나 못하는 것은 **측정 대상**이다.
- ★★**throughput 포화이며 SLO-지속가능 rate가 아니다.** "0.80·λ\*"는 "**SLO를 만족하는 부하의
  80%**"를 뜻하지 **않는다**. SLO-goodput(정본 술어: TTFT ≤ SLO ∧ 요청-내부 token-ITL p95 ≤ SLO)
  평가는 P2에서 별도로 한다. 이 구분을 결과 문장에 반드시 병기한다.

## 2. 측정 격자 (ONE BOOT PER CELL)

★**셀마다 별 boot·별 telemetry 파일**이다. 한 boot 안에서 rate ladder를 돌리지 않는다 — 하나의
telemetry를 시간 창으로 잘라 귀속하는 기계를 다시 들이면 이 트랙이 반복해 죽은 지점
(`CONSENSUS §3 항목120`)으로 돌아간다. 이 규율은 probe C(`c_capacity.sbatch:30-35`)에서 **문자 승계**한다.

| shape | 용도 | 앵커 |
|---|---|---|
| **A = (256 in, 512 out)** | **W3 전부**(decode heavy) + W4 decode phase의 **근사** | **없음**(아래 §3) |
| **B = (8192 in, 64 out)** | **W4 prefill phase** | job 905835 D44 @ 8192-in/96-out = **0.675 req/s**(out 96→64이므로 약간 높을 것) |

- shape A: rate 4점(앵커가 없어 넓게), shape B: rate 3점(앵커 기준 ×0.55/×0.85/×1.2 — probe C와
  같은 배수 구조). **브래킷을 만든 두 점을 제2 seed로 1회 반복**해 브래킷을 확인한다.
  총 **8–10 boot** ≈ 0.75–0.95 GPU-h(905835 실측 5.6분/boot).
- 하네스: `sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1.0
  --random-input-len <A|B> --random-output-len <512|64> --num-prompts NP --request-rate RATE --seed S`,
  **8 요청 순차 warmup 폐기**(probe C `:136-145` 승계). 달성 출력 길이를 **검증 항목**으로 기록한다
  (고정 길이가 실제로 지켜졌는지).

### 통제 요인 — ★**캠페인이 돌릴 구성과 일치시킨다**
`--attention-backend flashinfer` · ctx **16384** · `--mem-fraction-static **0.82**` ·
`--max-running-requests 48` · `--disable-radix-cache` · `--chunked-prefill-size -1` ·
`--disable-overlap-schedule` · cudagraph **ON** · PD-mux `fixed D44`(legacy 루프) · TP=1.
★**probe C(905835)와의 차이 2건을 등록한다**: 그 job은 `--mem-fraction-static **0.80**`·ctx **8192**
였다. 따라서 **0.675는 앵커로만 쓰이는 provisional 값**이며 이 측정의 결과와 같을 필요가 없다
(probe C의 prereg §2.3이 config 차이를 같은 방식으로 처리한 선례를 따른다).
엔진 소스는 **측정 시점 HEAD**를 쓰고 `runtime_source_manifest.sha256`(**24항목**, 2026-09-13부터
모델 구현 포함)로 기록한다 — Zamba2 correctness job들과의 핀 비교는 여기서 적용되지 않는다
(모델·백엔드가 다르므로 애초에 비교 대상이 아니다).

## 3. shape A의 앵커 부재와 W4 근사 (둘 다 사전 공시)

- **앵커 부재**: (256, 512)는 이 모델에서 측정된 적이 없다. 256-in은 prefill이 싸고 512-out이
  지배하므로 λ\*가 8192-in 값(0.675)보다 **훨씬 높을 것**으로 예상되지만 **수치 예보는 하지 않는다**.
  ⇒ `KNEE_NOT_BRACKETED`가 나올 수 있고, 그것은 **측정 실패가 아니라 ladder 재설계 신호**다
  (추가 1회 재설계까지 허용, 그 이상은 사용자 승인).
- **W4 근사**: W4의 decode phase는 (64, 512)이고 이 단계는 (256, 512)로 근사한다. 근거 = 두 shape는
  요청당 512 decode step을 공유하고 prefill만 64 vs 256 토큰으로 다르다. **반증 가능한 예보로
  등록한다: 두 shape의 λ\*는 5% 이내로 같다.** 이 단계에서는 측정하지 않으므로 **가정이며 측정이
  아니다** — W4 결과를 인용할 때 반드시 병기한다.
- **W4의 λ\*는 rate가 아니다**: W4는 Poisson이 아니라 phase당 요청 수(`round(0.80·λ*·intensity·30)`)로
  부하가 정해진다. 이 단계는 shape B의 **rate 용량**을 재고, **그 값이 W4의 intensity 스케일링에
  쓰인다**는 것까지만 등록한다. "W4 전체의 용량"은 재지 않는다(그건 W4 패턴 자체를 스케일하는
  별도 측정이며 이 단계 범위 밖이다).

## 4. 판정 규칙 (데이터 생성 전 고정 — `lambda0_label.py`)

probe C의 이미 감사된 규칙(`c_capacity_label.py` R1/R5)을 **실질 그대로** 승계하고 키만 arm → shape로 바꿨다.
- **R1 `KNEE_BRACKETED[shape]`**: ach/off **≥ 0.95**인 tested rate가 존재하고, 그보다 **높은** rate에서
  ach/off **≤ 0.90**인 셀이 존재한다.
- **R2 `LAMBDA_STAR[shape]`**: 문턱 없는 **측정값** — 포화 셀(ach/off ≤ 0.90) 중 **최대 achieved rate**.
  **R1이 BRACKETED일 때만 보고한다.**
- 셀 JSON 결손(boot/bench 실패) → **`UNRESOLVED`**, 이는 **측정 실패이며 규칙 실패가 아니다**(교훈 21).
- ★**arm 비교 금지**: ladder는 기준 arm 자신의 예측 용량에 상대적이므로 다른 arm의 셀과 비교 불가.
- selftest가 **이미 감사된 probe C 결과를 이 코드로 재현**한다(D16 0.933·D44 0.675 = `KNEE_BRACKETED`),
  그리고 비포화 ladder·너무 높게 겨눈 ladder·셀 결손·입력 순서 무관성을 각각 고정한다. 통과 확인됨.

## 5. 예보와 출력공간 라벨 (F1–F4)

- **F1 (부팅)**: 전 셀 boot 성공. 근거 = job 905835가 같은 모델·백엔드로 12/12 boot. 실패 시
  **측정 실패**로 기록하고 engine-porter 트리아지(ctx 16384·mem 0.82는 905835와 다르므로 그쪽을 먼저 본다).
- **F2 (shape B)**: `KNEE_BRACKETED`. λ\*(B)는 0.675(8192-in/96-out 앵커)와 **같을 필요 없다** —
  out 96→64·ctx·mem-fraction이 다르다. 수치 예보 없음, **브래킷 존재만** 예보한다.
- **F3 (shape A)**: `KNEE_BRACKETED`. 단 앵커가 없어 **`KNEE_NOT_BRACKETED` 확률이 실질적**이다 ⇒
  그 경우 ladder를 재설계해 1회 재실행(위 §3).
- **F4 (제2 seed 확인)**: 브래킷 두 점의 ach/off 부호가 제2 seed에서도 유지된다(≥0.95 / ≤0.90).
  뒤집히면 **브래킷은 미확정**으로 등재하고 λ\*를 보고하지 않는다.
- **출력공간 전수**: {BRACKETED, NOT_BRACKETED, UNRESOLVED} × {shape A, B} × {seed 확인 유지/반전}
  전 조합에 위 라벨이 붙는다. **어떤 조합도 정책·arm 순위·Claim 등급을 움직이지 않는다.**

## 6. 이 단계가 **주지 않는 것**

- **성능·정책 판정 0건** — arm 비교 없음, goodput 평가 없음(λ\*는 throughput 포화다).
- Claim D/E 등급 불변 · 선결 #1–#5 불변 · HE0 · 정책 순위 · stake #1 불변.
- λ\*는 **B1(legacy fixed D44) 한정**이다. 다른 arm의 λ\*는 **5× 다를 수 있다**(905835 실측).
- **새 모델의 correctness는 이 단계가 건드리지 않는다** — (Nano-9B-v2, flashinfer) 쌍의 R2
  correctness 게이트는 **아직 한 번도 돌지 않았다**. 907100·907456·X1의 결론은 **(Zamba2, triton)
  한정으로 동결**이고, X1의 민감도 시연은 `triton_attention_num_kv_splits`가 flashinfer에 없으므로
  **이식되지 않는다**.
- 단일 스칼라 λ\*를 W1–W9가 공유하는 설계 문제는 **해소되지 않는다** — 이 단계는 Claim D에 필요한
  2 shape만 잰다. W1·W5·W6·W7·W8·W9의 shape는 미측정으로 남는다(그 워크로드를 쓰려면 같은 절차 반복).
