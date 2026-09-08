# 정정 기록 — **스모크 "3.59× 노드 대조"는 노드 대조가 아니었다**

2026-08-31 · GPU **0** · 새 성능 판정 **0건** · 정본 편집 **0건** ·
출처: [`audit_s0_rules_2026-08-31/VERDICT.md`](audit_s0_rules_2026-08-31/VERDICT.md) 死因 **S0-F1** ·
메인 세션 독립 재확인 완료

> ★**원문은 보존한다.** 아래 3개 문서의 해당 절은 지우지 않고 이 정정으로 상단 배너를 단다
> (정본 관례 — 삭제가 아니라 정정 표시).

## 1. 무엇이 틀렸나

3개 문서가 스모크 job **896764 ↔ 896765**를 *"설정 동일 · 노드와 런만 다름"* 대조로 인용하고,
`Median TPOT` **18.98 vs 68.06 ms(3.59×)**를 **플랫폼(노드/런) 분산**의 증거로 썼다.

**거짓이다.** 두 런의 `server_info.server_args`(같은 jsonl 안에 있다):

| job | port | `attention_backend` | `disable_cuda_graph` | `median_tpot_ms` | `median_itl_ms` | `input_throughput` |
|---|---|---|---|---|---|---|
| 896764 | 37164 | **`flashinfer`** | **False** | 18.98 | 18.834 | 4536.6 |
| 896765 | 37165 | ★**`torch_native`** | ★**True** | 68.06 | 26.289 | 4546.9 |
| 896767 | 37167 | `flashinfer` | False | 29.69 | 29.499 | 6481.8 |

⇒ **백엔드와 cudagraph가 함께 바뀐 대조**다(스모크 sbatch의 `--attention-backend ${ATTN:-flashinfer}`가
`ATTN=torch_native`로 덮여 실행됐고, **`.out`에는 그 사실이 찍히지 않는다**).
`server_args` 덤프는 `n9srv_896765.log:14`에도 있다.

## 2. 무엇이 따라 죽고, 무엇이 사는가 (파급 재도출 — `CONSENSUS §3` 항목100)

### 죽는 것
1. **`S0` 사전등록 §1 전체** — 동기 표와 *"노드와 런만 다르다"* 문장. §5 사전확률 문단
   (*"CV 0.797 ⇒ `VAR_WIDE` 유력"*)도 그 위에 서 있었으므로 함께 무효.
2. **`TRACK_DESIGN_LAYER_COST_CANON_2026-08-30.md` C9** — *"런/노드 분산 미측정"*의 증거로 쓴 3.59×.
3. **`../tc1_model_attrib/probes/p4_reach/REACH_FINDING_2026-08-29.md` §4** — *"같은 모델·같은 플래그·
   같은 워크로드"*라는 표 머리글과 *"런/노드 효과가 유력"*이라는 해석.

### 사는 것 (재확인)
- `REACH_FINDING` **§1–§3**(닫힌 형태 문턱 · `BOTH_DEAD` 도달불가 · FEAS congestion 사망)은
  **코드 상수에서만 나온 산술**이라 이 정정의 영향을 받지 않는다.
- `REACH_FINDING` §4가 말한 **"체제를 정하는 것은 TPOT"** 자체는 산다 — 다만 관측된 TPOT 이동의
  원인이 *플랫폼*이 아니라 **백엔드·cudagraph**였다는 것으로 귀속이 바뀐다. ★그 형태로는 오히려
  게이트 #83(엔진 강제 지원영역)과 정본의 *"cudagraph-ON = 운영점"* 규율에 **직접 걸리는 관측**이다.
- `TRACK_DESIGN` §0(컨트롤러가 Zamba2 조성의 knee를 먹는다) · C1–C8 · C10은 **무영향**.

## 3. ★그리고 사려던 것은 이미 디스크에 있었다 (PS게이트 #72 / `CONSENSUS §3` 항목92 위반)

감사가 지목한 `results/slo_sched/g16_blk{1..4}_d44_boot1_*` — 메인 세션 독립 재계산:

```
모델 Zyphra/Zamba2-2.7B · ctx 4096 · triton · cudagraph ON · pdmux_d44.yml · max_running_requests 48
부팅 4개(884336=gpu42, 884410/884411/884412=gpu41) · 각 부팅 3라운드 · 동일 플래그

LO(rate 3)  TPOT_med per-boot = [12.602, 12.577, 12.533, 12.578]   CV = 0.00228
            ITL_med  per-boot = [10.970, 10.950, 10.943, 10.943]   CV = 0.00115
HI(rate 12) TPOT_med per-boot = [27.667, 27.541, 27.451, 27.534]   CV = 0.00324
            ITL_med  per-boot = [20.158, 20.243, 20.180, 20.221]   CV = 0.00190
```

⇒ **부팅 간 CV ≈ 0.2–0.3%**(2개 노드 포함). S0가 가정한 0.797과 **250배 이상** 다르다.
`n_required(0.10)`는 6이 아니라 **2**다 ⇒ 등록 라벨로는 `VAR_TIGHT`.

★**스코프(엄수)**: 이 수치는 **Zamba2-2.7B · triton · ShareGPT · ctx 4096** 한정이다.
Nemotron-Nano-9B · flashinfer · 긴 프롬프트 arm으로 **크기 이전 금지**(게이트 #83 · §1-3 배너).
이 데이터가 확정하는 것은 *"이 플랫폼이 구조적으로 불안정하다"*는 **전제의 반증**이지,
새 arm의 분산값이 아니다.

## 4. S0 설계에 대한 귀결

1. **6 부팅 캠페인(≈0.8 GPU-hr)은 제출하지 않는다.** 동기가 거짓이고, 질문의 절반은 이미 답이 있다.
2. 남는 질문은 **훨씬 좁다**: *"Nemotron·flashinfer·긴 프롬프트 arm의 부팅 간 CV가
   g16이 보인 0.2–0.3%대인가, 아니면 자릿수가 다른가?"* — **2 부팅이면 방향을 가른다**(≈0.2 GPU-hr).
3. **추정량을 바꾼다**: 주 대상은 `median_tpot_ms`가 아니라 **`median_itl_ms`**로 간다
   (감사 S0-F2: 두 런의 `input_throughput`이 0.23% 차이인데 `TPOT_med`만 3.59× 벌어졌고,
   분해하면 level 1.396 × 꼬리 2.568 — 즉 `TPOT_med`는 꼬리 통계다. 실제로 위 표에서
   `median_itl_ms`는 18.834 vs 26.289로 **1.40×**에 그친다).
   ★`median_itl_ms`는 **같은 jsonl에 이미 있다** — 등록하지 않았던 것이 결함이었다.
4. **모든 반복의 `server_info.server_args`를 아티팩트로 저장하고 기대값과 assert**한다.
   이번 오독은 **`.out`만 보고 백엔드를 추정**해서 났다(메모리항목 72의 정확한 재발).

## 5. 이 문서가 하지 않는 것
성능 판정 · 정책 순위 변경 · 정본 편집 · 새 arm의 분산값 확정.
불변: HE0 · 정책 순위 · gate #13/#16 "닫았다" 금지 · switch-cost "닫았다" 금지 ·
C2 인용정지 (a)(b) · `CONSENSUS §1-24` 미반증.

## 6. 재현
```bash
cd workspace/engine-port/results/model_roster/smoke
python3 -c "import json;[print(json.loads(l)['server_info']['server_args']['attention_backend']) for l in open('sglang_0828_40_8000_96.jsonl') if l.strip()]"
cd ../../slo_sched   # 부팅 간 CV
python3 -c "..."     # 위 §3 블록 (glob 'g16_blk*_d44_boot1_*_[LH]*.jsonl', telemetry 제외)
```
