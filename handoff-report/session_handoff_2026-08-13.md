# 세션 핸드오프 — 2026-08-13

> ★**날짜 주의**: 이 문서는 2026-08-13에 작성됐으나, **이번 세션의 작업·커밋·아티팩트는 전부
> 2026-08-11자로 스탬프돼 있다**(git 커밋 타임스탬프 01:57–15:44 KST, job 877974,
> 정본 rev22/23/24, 사전등록·설계 문서 4종). 다음 세션이 "2026-08-11 세션"을 찾을 때
> 이 문서를 보면 된다. 직전 핸드오프는 `session_handoff_2026-08-10.md`(§10이 2026-08-11에 추가됨).

---

## 1. 이번 세션 요약

**GPU 총 지출 0.10 GPU-hr** (job 877974, 5분 52초). 나머지는 전부 GPU 0 작업이다.

두 갈래가 돌았다. **(A) Gate 2-S 전제 검증 체인** — G1-c를 사전등록·실행·채점·감사·정본
반영하고, 이어서 E1 addendum(GPU 0, 기존 텔레메트리 재집계)까지 같은 파이프라인을 태웠다.
**(B) 사용자 반박에서 출발한 진단** — "decode인데 4 arm SM 민감도가 비슷한 게 이해 안 된다"는
질문이 트래픽·roofline 진단(GPU 0)을 낳았고, 그 결과 **7–8B 캠페인 문서에 3B급 수치가 기준
혼합된 채 들어와 있던 정본 결함**이 드러났다. 마지막으로 대기 중 GPU 실험 4건의 **실행 준비
검토**를 수행해 넷 다 제출 불가임과 그 사유를 확정했다.

정본은 **rev21 → rev24**, 방법론 게이트 **#26–#32** 7건이 신설됐다. 커밋 **11건**
(`cda12cf` … `87f6038`; 직전 세션 마지막 커밋은 `8f7e536`).
★ 이번 세션의 특징: **payoff 서술이 세 번 코드 확인에 걸려 정정됐고**(전부 지출 전),
그중 하나가 방법론 게이트 #28의 등재 근거가 됐다.

---

## 2. 결정·측정

### 2.1 G1-c — Granite realized 파티션 관측 (job **877974**, **0.10 GPU-hr**)

- 사전등록 `results/p1_gates/gate1/PREREG_G1C_2026-08-11.md` (제출 전 고정) → 실행 →
  result-analyst 독립 재계산 → claims-auditor 적대 감사 → doc-steward 정본 반영.
- **격자 전부 재현**(rate {2,3,4,6}, 방법론 게이트 #16). 결과: max decode_bs **7/10/10/15**,
  pop A 시간가중 `(74,34)` **100%**, `frac_5454 = 0.0000`, 격자 결측 0, 양성대조 4/4 PASS.
- 사전등록 기계 규칙 출력: `G1C_GATE2S_PREMISE rate=3 → VERIFIED`, `rate=4 → VERIFIED`.
- **G1-b의 rate 6 철회 발화가 Granite에서는 재현되지 않는다**(Zamba2는 r6에서 40으로 문턱
  초과 → `(54,54)` 8.37%; Granite는 최대 15).
- 원자료: `results/p1_gates/gate1/gate1c_result_877974.txt`.

**★ 판정 = 명명 층 조건부 해제(rev22). 승격이 아니다.**

- ★★**세션 초기 서술 "전제 VERIFIED → 크기 인용 셀 1→3개"는 거짓이었다.** 크기 인용을
  통제하는 것은 `premise` 라벨이 아니라 **F-계열 gate**이고 둘은 `gate2/g2s_analyze.py:1157-1161`
  에서 독립 산출된다. Granite r3(T′)·r4(C·T′) 모두 F-계열 발화 상태 ⇒
  **`SIGN ONLY, MAGNITUDE NOT CITABLE` 유지, 크기 인용 셀은 Zamba2 r2 하나 불변.**
  실제로 바뀐 것은 `name_for()`(`:698-704`)의 명명 한 층뿐이고, 부수적으로 §5.6(f)의
  "전제 미검증 셀에서만 발화" 의무 문구가 사라진다(= caveat 제거, 개선 아님).
- 해제 조건 6건 필수(명명 층 한정 / F-계열 병기 / **경험적 내용은 하나**(항등식) /
  증거 등급 명시 / 트리 근거는 877757 기준 / Zamba2 근거표 동시 정정).
- **항등식 CONFIRMED**: `multiplexing_mixin.py:900-909`에서 `idx==2 ⟺ decode_bs ≥ 36`
  ⇒ `max<36`이 `frac_5454=0`을 함의. 두 조건은 독립 증거가 아니다.

### 2.2 E1 addendum — Gate 2-S 자기 셀 A4 텔레메트리 재집계 (**GPU 0**)

- 사전등록 `results/p1_gates/gate2/PREREG_G2S_E1_ADDENDUM_2026-08-11.md`
  (★**post-hoc, not blind** 최상단 선언 — claims-auditor가 반증 시도 중 값을 이미 산출).
- 구현 `results/p1_gates/gate2/g2s_e1_premise.py` — `g2s_analyze.py` **무수정·미import**,
  규칙은 생산자 `gate1/gate1b_analyze.py`를 실제 import(재구현 0줄, 새 자유모수 0).
- 결과 `results/p1_gates/gate2/g2s_e1_premise_877756_877757.json`.

**측정값 (pooled max decode_bs, n=10, 문턱 36)**

| 셀 | pooled max | 여유 | 기존 복제-격자 근거 | 적대적 bound (q=1e-9) |
|---|---|---|---|---|
| Zamba2 r2 | 14 | 22 | G1-b 9 (**+56% 과소평가**) | 25 |
| **Zamba2 r3** | **23** | **13** | G1-b 23 (일치) | **41** ⚠️ |
| Granite r3 | 10 | 26 | G1-c 10 (일치) | 24 |
| Granite r4 | 13 | 23 | G1-c 10 (**+30% 과소평가**) | 27 |

**★ 판정 = 조건부 채택(4셀 전부, 등급 하향). 승격 아님.** 조건 7건 전부 필수.

- ★★**E1이 §2.5에서 주장한 "밀도 페널티를 덜 받는다 / 이것이 이 addendum이 존재할 수 있는
  이유다"는 반증됐다.** 결정 관련 pop-A 관측 수가 G1-b/G1-c 대비 **5–6× 적다**
  (1,279·1,496 vs 6,402·8,920). **E1의 우위는 밀도가 아니라 반복수(n=1→10)와 셀 일치뿐.**
  ⇒ **"E1이 G1-c를 대체한다"고 쓰면 안 된다. 두 관측은 서로 다른 축이다.**
- ★ 감사자가 **새 bound를 구성**했다: `sup decode_bs ≤ insys + Poisson(λ·dt)`(새 자유모수 0,
  새 계측 0). **Zamba2 r3만 q=1e-6에서 이미 37 ≥ 36으로 문턱을 배제 못 한다** —
  인용 시 강등이 아니라 **margin 주석 강제 병기**.
  (메인 세션이 제안한 "인접 샘플 최대 Δ" 방식은 bound가 아님이 지적돼 기각됐다.)
- **Zamba2 r3를 강등하지 않는다** — 강등하면 §8.9(rev5 이래)의 Zamba2 r3 "검증됨"이
  **먼저** 무너진다(근거가 G1-b의 같은 값 23·같은 여유 13·더 나쁜 n=1). 정본 자기모순 회피.
- **4셀 전부 non-blind**(작성자가 G1-b의 9/23을 이미 알고 있었음) ⇒ "Granite만 파기" 선택지는
  논리적 근거 없음.
- **성능 결론 불변**(코드 확인): `premise`는 `nine_cell()`·`gate_label()`·`paired_t()` 어디에도
  인자로 안 들어간다. Δ=+13.95/+24.88/+16.04/+19.11 ms · 4셀 Holm 최대 p=1.88e-06 ·
  9-셀 16블록 `S1-C` · 크기 인용 셀 Zamba2 r2 하나 — **한 글자도 안 바뀜.**
- **무료 대조**(감사자, addendum 미산출): 같은 셀에서 **T′ max decode_bs = 13/23/10/12**로
  A4(14/23/10/13)와 **±1 이내 일치** — 전제가 의존하는 부하 동등성의 직접 증거.

### 2.3 트래픽·roofline 진단 (**GPU 0**, 사용자 반박에서 출발)

문서: `results/s8_scaleup/TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md`.
대상: 4-arm decode-축 SM 민감도 캠페인(`FINDINGS_8B_2026-07-28.md`, jobs 865289~865533).
**새 성능 판정 0건.** 기존 C2(2.36–2.91×, scoped)의 등급·수치 불변, 스코프는 더 좁아짐.

**config 실측 (서버 로그와 전 항목 <0.1% 일치)**

| arm | mamba/attn 층 | state/seq | KV/seq @L≈1282 | per-seq 캐시 | W_step |
|---|---|---|---|---|---|
| Ha8 Zamba2-7B | 81/13 (**MHA 32×224**) | 145.2 MiB | 455.7 MiB | **601 MiB** | **22.06 GB** |
| M8 Codestral-7B | 64/0 | 259.8 MiB | 0 | 260 MiB | 14.30 |
| Hs8 NemotronH-8B | 24/4 (GQA 8×128) | 97.4 MiB | 20.0 MiB | 117 MiB | 15.13 |
| T8 Qwen2.5-7B | 0/28 (GQA 4×128) | 0 | 70.1 MiB | **70 MiB** | 14.14 |

- ★**"4 arm 모델-무관"의 원인은 아키텍처 동질성이 아니라 weight sweep 지배**:
  스텝 트래픽 중 weight 비중 **M8 68.6% / Hs8 84.8% / T8 94.1% / Ha8 70.1%**,
  계열 구분 항은 **6–31%뿐**. 동작점 B≈12·L≈1.3k가 아키텍처 차이를 소수 항으로 밀어냈다.
- ★★**"hybrid는 KV가 적다"는 통념은 이 격자에서 거짓** — Ha8이 T8의 **8.6×**.
  Mamba2 state는 `d_inner×d_state` **fp32**라 절대 크기가 크고, Zamba2는 GQA가 아니라
  MHA head_dim 224×13층. **다음 세션은 이 통념으로 다시 추론하지 말 것.**
- **교차점 `L* = 4,750`**(T8 KV vs M8 state). 측정점 L≈1282은 그 **27%** 지점 ⇒
  메인 세션이 제시한 "측정점이 교차점 근처라 닮았다" 가설은 **반증**.
- **roofline**: achieved BW가 SM92에서 사양(A100 80GB **PCIe 1,935 GB/s** — SXM 2039 아님)의
  **47.8–60.8%**뿐 ⇒ ★**고-SM(44→92) 평탄화를 HBM 포화로 서술 금지**(원인 미식별:
  wave quantization / 층 직렬 지연 / 점유율 / cudagraph 직렬화, 커널 측정 0건).
  저-SM 굶주림은 약하게 지지(ε(16→24)=0.83–0.90, 천장의 17–26%).
- **compute floor 음성대조 통과**: decode AI ≈ **8.2** FLOP/byte vs ridge 161.
  ★대조: 1026토큰 prefill은 AI ≈ **1,035**(compute-bound) ⇒
  **decode 결론의 prefill 축(`s8p_prefill`) 이식 금지**(반대 방향도 금지).
- ★★**정본 결함 확정**: `FINDINGS_8B` §2.1의 "1차 weight-traffic 추정 M 5.40 / T 6.17 /
  H 7.66 GB, H/M=1.42"가 **3B급 다른 캠페인 수치 + 기준 혼합**이었다
  (M 5.40 = `mamba2-2.7b` 파일 크기 3자리 정확 / T 6.17 = `Qwen2.5-3B` 텐서 합 3자리 정확 /
  H 7.66 ≈ Zamba2-**2.7B** 호출 인지 트래픽). **M·T는 체크포인트 바이트, H만 재독출 트래픽**
  ⇒ 기준 혼합. 같은 기준이면 H/M = 0.985. **감사 2회(2026-07-28, 08-04)를 통과해 있었다.**
  ⚠️ 이 정정은 강등된 주장을 **되살리지 않고 더 약화**시키며, **반대 방향 주장도 금지**.

### 2.4 설계 문서 2건 (**GPU 0, 미실행, 둘 다 claims-auditor 미통과**)

**C-1 `%smid` 직접 계측** — `workspace/engine-port/reports/SMID_DIRECT_INSTRUMENTATION_DESIGN_2026-08-11.md`
- 기술 타당성 확인: Triton 3.5.1 `tl.inline_asm_elementwise`로 `%smid`를 sm_80 PTX/cubin에
  **CPU에서** 생성 가능(GPU·nvcc 불필요, 스핀 루프 hoist 안 됨). 관측자 부하를
  `cuobjdump -res-usage`로 **통계 없이 기전 bound**(REG 8→10, SHARED/LOCAL 불변).
- **경로 4개 중 3개 사망(코드 사실)**: CUPTI는 권한이 아니라 **원리**로(헤더에 `smId` 0건) ·
  green ctx 자기보고는 **또 다른 target**(`CUdevSmResource`에 `smCount`만, ID 집합 없음
  ⇒ Stage 0 D108의 거울상) · DCGM/nvidia-smi는 **범주**로 배제.
- ★**게이트 #28 적용 결과가 부정적**: `%smid`가 성공해도 **§1-1 "PD 분리 자체" 귀속은
  닫히지 않는다**(전달 질문 vs 효과 귀속 질문). 크기 인용 셀도 안 늘어난다.
  실제로 바뀌는 것은 `g2s_analyze.py:1488`의 `eps = log(mt/mc)/log(108.0/34.0)`
  **하드코딩 정정** + caveat 문구 + telemetry 라벨 실측 대체.
- 부수 발견: **green-ctx 스트림에서 캡처한 cudagraph 재생 시 SM 한정 전달 여부 미측정**.
- 비용: P1+P2 프로브 **≈10분** / L0+L1 0.55–1.05 GPU-hr / L2는 사지 않기로 권고.

**C-2 per-layer-type SM 응답** — `workspace/engine-port/reports/PER_LAYER_TYPE_SM_RESPONSE_DESIGN_2026-08-11.md`
- ★**계측은 이미 있고, 이미 돌았고, 자기 게이트에 실패했다**: `zamba2.py`(리워크본, manifest 포함) ·
  `nemotron_h.py:646` · `granitemoehybrid.py:474`(둘 다 리워크 미이식, **manifest 밖**).
  job **873783**(2026-08-05)이 SM 스윕에 돌려 **GATE N FAIL / G6 NO-GO**
  (`results/r0c/P5_GATES_BATCH1_2026-08-05.md`) — host-paced idle이 span에 청구,
  **고-SM·단-ctx 코너에서 per_mamba 2.78× 부풀고 SM에 비단조**(우리가 보려는 그 코너).
- ★★**구조적 제약**: `cuda_graph_runner.py:547` `capture_forward_mode=DECODE`, `:1161`
  `replay()`가 **Python forward를 호출하지 않는다** ⇒ **per-layer 계측은 cudagraph-ON에서
  실행되지 않는다.** 운영점이 cudagraph-ON이므로 이 캠페인은 **정의상 off-operating-point**.
  stock `--enable-layerwise-nvtx-marker`도 같은 이유로 eager 전용.
- 격자 제약: "두 층 타입 모두 아키텍처 항 ≥2/3" matched 셀이 **(L=1024, B≈96) 하나뿐**이고
  Ha8 메모리 상한의 98% ⇒ **within-model 설계 강제, arm 간 종합 금지**.
  arm 상한: Hs8 ctx8192(B≤216@8192) / **Ha8 ctx4096(B≤98@1024, ≤31@4096)**.
- **long-ctx 트랙과 합칠 수 없다** — long-ctx는 cudagraph-ON 필수인데 계측이 거기서 안 돌고,
  hybrid ctx 상한이 4096/8192다. **7–8B hybrid의 long-ctx per-layer-type은 현재 모델 자산으로 불가능.**
- 비용: 선행 프로브 P1 **≈GPU 30분**(A1+A2+E2+E8 동시 판정, 실패 시 eager 사다리 사망) /
  본 캠페인 Zamba2 단독 2–3 GPU-hr.

### 2.5 GPU 실험 4건 실행 준비 검토 (**GPU 0, 제출 0건**)

experiment-runner(SLURM·하네스 축) + engine-porter(코드·매니페스트 축) 병렬 수행.

**★ 결론: 4건 전부 제출 불가. 공통 블로커 = 사전등록 부재.**
(P1·`%smid`는 설계 문서가 §12/§13에서 **자기 자신을 사전등록으로 인용 금지**한다고 명시.
E1-b/E1-c는 PROJECT_STATUS 요약 문단만 있고 이 트랙 표준 형식[무결성 규칙·blind 선언·
자기무력화 조건]이 전무. G1-a는 패치조차 없음.)

**★ 네 실험을 묶는 진짜 결합 = 매니페스트 트립와이어**

| 하네스 | 규칙 | 위반 시 |
|---|---|---|
| `g2s_run.sbatch:81-89` | 873944 대비 `added==2 ∧ removed==0` | **exit 4** |
| `gate1b/gate1c_run.sbatch` | `N_CHANGED==0 ∧ N_ADDED==2` | `MANIFEST_MISMATCH_ABORT` |
| `gate1_run.sbatch:61-66` | 완전 동일성 | **이미 stale — 지금 돌리면 abort** |

현재 트리는 877756/877757/877974와 **바이트 동일**(15/15 sha OK, 미러 드리프트 0).
**dev 트리를 바꾸는 것은 G1-a 하나뿐**이고, 나머지 셋(P1 Zamba2 단독 / E1-b·c / `%smid`)은
매니페스트 무변경이라 **동시 제출이 코드 사실로 안전**(flock 하에서 항상 같은 내용 재설치).

**발견된 함정 5건 (전부 코드 확인)**
1. ★ **`SGLANG_ZAMBA_TIMING=0`은 켜진다** — `bool(os.environ.get(...))`이라 `"0"`이 truthy.
   설계의 OFF arm이 **ON을 측정**한다. **`unset` 필수.**
   (`nemotron_h.py:646`·`granitemoehybrid.py:474`도 동일 패턴. 대조: `PDMUX_TRACE_FORCE_PREFILL`은
   `in ("1","true","True")`로 올바름.) **코드 미수정 — engine-porter 이관.**
2. ★ **P1 하네스를 인플레이스로 고치면 E1-b가 죽는다** — `test_p5_gate_tools.py:40-41`이
   r0c 분석기·sbatch **내용을 문자열로 고정**하고, 그 회귀가 `g2s_run.sbatch:91`의 차단 게이트다.
   **새 파일로 분기 필수.**
3. ★ **`g2s_run.sbatch`의 `boot()`가 `PDMUX_TRACE_FORCE_PREFILL`을 unset**하고 어떤 arm도
   재export하지 않는다 ⇒ env만으로 E1-b가 안 걸린다. unset이 `eval "$(arm_env)"` **앞**이라
   `arm_env()`의 agnostic 줄에 추가하면 된다. `RATES`도 모델별 하드코딩(`zamba2-27b="2 3"`)이라
   E1-c의 rate 6은 case문 수정 필요.
4. ★ **`g2s_e1_premise.py`가 `trace_forced`를 참조하지 않는다** ⇒ 877756/877757의 `M_c`와
   E1-b의 `M_c`를 나란히 놓는 것은 **밀도 교락 비교**. 필터 옵션 추가하거나
   "비교하지 않겠다"를 사전등록에 못박아야 한다.
5. ★ **G1-a를 `_dual_worker_sync` 재사용으로 구현하면 안 된다** — 그 함수가
   `dual_worker_trace_count`를 무조건 증가시키고 격자가 `count%32==0`이라, sync 지점을
   2→3으로 늘리면 **emit 빈도 1.5배 + 다른 루프 지점 샘플** ⇒ 874478/875293/877974와
   **event-for-event 비교가능성 소멸**. 올바른 형태는 별도 이벤트 타입 직접 발행.

**★ G1-a는 E1-b가 공짜로 대체할 수 있다** — `:1242` 스냅샷이 이미
`prefill_active_batch_size`·`stream_index`·`prefill_sms/decode_sms`·`timestamp_s`를 싣고,
`FORCE_PREFILL=1`이면 prefill 동거 반복마다 강제 emit된다 ⇒ **E1-b가 이미 반복 해상도로
그 구간을 찍는다.** G1-a가 추가로 사는 것은 **반복 내부 순서와 sub-iteration ms**뿐.
⇒ **E1-b 원자료를 먼저 열어보고 G1-a 착수 여부를 결정하라**(게이트 #28 직접 적용).

**정정된 것 2건** (메인 세션이 앞서 잘못 전달한 것)
- "CPU 회귀를 컴퓨트 노드에서 돌려야 한다" → **재현 안 됨**. 로그인 노드에서
  **140 tests OK / 73초**(대부분이 `import sglang.srt.managers.scheduler` 70.6초).
  7분 stall은 Lustre 콜드 캐시로 추정되며 **코드 사실 아님**. 게다가 `g2s_run.sbatch:91`이
  **이미 컴퓨트 노드에서 차단 게이트**(실패 시 `exit 5`)로 전체 회귀를 돌린다.
  ⚠️ 단 이 사실이 설계 문서 한 곳에만 있고 `RESUME.md:22`는 노드 구분 없이 지시 —
  **doc-steward 전파 대상.**
- "신규 교란원 `mamba_track_interval`" → **이 구성에선 원리적으로 발화 불가**.
  체크포인트 경로가 `enable_mamba_extra_buffer()` 뒤에 있고(`schedule_batch.py:2144, 1768`)
  기본이 `no_buffer`이며, `extra_buffer` + `--disable-radix-cache`는
  **ValueError로 부팅 거부**(`server_args.py:2177-2181`). `--mamba-track-interval`은 no-op.

---

## 3. 코드·문서 변경

### 커밋 11건 (전부 `main` 직커밋)

| sha | 내용 |
|---|---|
| `cda12cf` | docs(canon): `reports/paper/` 3문서를 CONSENSUS rev14–20과 동기화, 게이트 #26 신설 |
| `ece9469` | test(p1_gates): G1-c 사전등록 + 하네스 (job 877974, **결과 없음** 상태에서) |
| `d5727ec` | chore(gitignore): smoke-prompt ignore 규칙 일반화, meta provenance 추적 |
| `8a721a0` | test(p1_gates): G1-c 채점 결과 아티팩트 |
| `4dc6191` | docs(canon): **rev22** — G1-c 명명 층 조건부 해제 |
| `302a480` | test(p1_gates): E1 addendum 사전등록 + 구현 + 결과 |
| `edd27da` | docs(canon): **rev23** — E1 조건부 채택(등급 하향) |
| `40abbbc` | docs(design): `%smid` 직접 계측 타당성·설계 (C-1) |
| `907db11` | docs(s8_scaleup): 트래픽·roofline 진단 |
| `781a92e` | docs(engine-port): per-layer-type SM 응답 계측 설계 (C-2) |
| `87f6038` | docs(canon): **rev24** — 기준 혼합 weight-traffic 수치 철회, C2 인용 스코프 축소 |

직전 세션 마지막 커밋은 `8f7e536`(2026-08-11 01:22:55). 위 11건이 이번 세션 전부다.

### 정본 갱신

- `reports/CONSENSUS.md` **rev21 → rev24**. §3 항목 **40·41·42·43·44·45·46** 신설,
  항목 18·39에 재발 追記, §1-1에 G1-c·E1 블록, §4 갱신.
- `PROJECT_STATUS.md` 방법론 게이트 **#26–#32** 신설(#9·#20·#21·#25에 재발 追記),
  "확정된 결과" 1번·"다음 실험 gate" #10 갱신(G1-c 완료, G1-a 해금, E1/E1-a/b/c/d 등재).
- `reports/paper/` 3문서 — 신규 "P1 트랙" 절, Claim G 행, Gate 2-S 인용 제한 인라인 주석.
- `results/p1_gates/gate2/PREREG_GATE2S_2026-08-09.md` — **dated addendum**(원문 보존):
  Granite VERIFIED 승격 + 크기 인용 불변 명시 + **Zamba2 pop-C 인용 정오표** + `t_total_s` 금지.
- `results/s8_scaleup/FINDINGS_8B_2026-07-28.md` — 정정 배너 + 신규 `## 7.` 절(원문 미삭제).
- 메모리: `MEMORY.md` · `slo-aware-scheduling-track.md` · `scale-8b-sm-sensitivity.md` ·
  `prefill-axis-sm-sensitivity.md` · `deconfound-measurement-lessons.md`(항목 25→31).

### 신설 방법론 게이트 7건

| # | 내용 |
|---|---|
| **#26** | 대형 캠페인 제출 전 **배관 스모크** 규율(하네스 diff ∧ ≥1 GPU-hr) |
| **#27** | **추정량의 밀도 의존성을 먼저 따져라**(`max`는 sparse에서도 나온다) |
| **#28** | ★**실험의 payoff를 코드로 확인한 뒤 정당화하라**(이번 세션 "1→3셀"이 반례) |
| **#29** | span 기반 `compute_coverage`는 내부 구멍에 맹목 |
| **#30** | 저장소가 "경로 없음"이라 쓴 지점 옆의 새 통계량은 **자기인지 자백만으로 재감사 면제 안 됨** |
| **#31** | **bound와 점추정을 구별하라** |
| **#32** | **타 캠페인에서 수입한 보조 수치는 basis를 검증하라** |

---

## 4. 열린 항목 / 다음 세션 시작점

### 즉시 (GPU 0) — 첫 작업

1. ★**정본 등재 후보 2건**(이번 검토 산물, 아직 정본 미반영):
   - **게이트 #26의 구멍** — 트리거 주어가 "캠페인"이라 **<1 GPU-hr 조각으로 쪼개면
     회피된다**(이번 4건 합 ≈1.15–1.65 GPU-hr인데 각각은 문턱 아래). 트리거를
     **"한 배치로 제출되는 신규코드 캠페인들의 합"**으로 개정 권고.
   - ★**매니페스트는 런타임의 완전한 기술이 아니다** — `sgl_kernel/spatial.py`(green ctx
     생성 원시함수)와 `sglang/srt/multiplex/pdmux_context.py`(74/34 기대값의 출처)가
     **둘 다 매니페스트 밖**. `env/dev_tree_edits.md` 항목 3·4·5·7은 sync가 재적용도
     해시도 안 하고, `src/patches/`의 패치 2개(`nemotron_h_forward_split_prefill.patch`,
     `triton_backend_mambaish_vheaddim.patch`)는 **한 번도 적용되지 않는다**.
     재현성 주장의 범위를 좁혀야 한다.
2. **정본 결함 1건 미수정**: `PROJECT_STATUS.md` 방법론 게이트 **헤더**가 본문에 없는 追記를
   명시한다(E1 반영 때 헤더만 갱신, 본문 누락. CONSENSUS §3 항목18에는 존재).
3. `RESUME.md:22`의 CPU 회귀 지시에 노드 구분·소요 시간 반영(§2.5 정정).

### Phase 0 — GPU 0, 서로 완전 병렬 (다음 GPU 라운드의 선행조건)

- **사전등록 3건**(P1 · E1-b/c · `%smid` R0) — 병렬 작성해 claims-auditor 감사에 **한 번에** 태울 수 있다. 각 반나절.
- **하네스 3건** — `%smid` L0 스크립트(스크래치→리포) · P1 sbatch(**새 파일**) ·
  `g2s_e1b_run.sbatch`(**복사 후** 수정, `_teloffset_` 기록 보존). 각 0.5일.
- P1 분석기는 **재작성 거의 불필요** — A1(`gate_n_conditional:347`)·A2(`host_boundness_rows:435`)·
  `batch_mixture:272`가 이미 구현돼 있다.

### Phase 1 — GPU, 셋 동시 제출 가능 (트리 무변경 상태에서)

우선순위 **E1-b/c(0.5–0.7hr, α paired 시 ×2) > `%smid` P1+P2(10분) > P1(0.4–0.6hr)**.
E1-b는 G1-a에 의해 차단되는 유일한 건, `%smid`는 가장 쌈, P1은 payoff가 가장 좁음.

### Phase 2 — 게이트

E1-b 원자료로 G1-a 필요 여부 판정 · `%smid` R0 실패 시 경로 1 전체 사망(즉시 정지) ·
P1의 A1/A2 실패 시 eager 사다리 사망(§6 격자 안 삼).

### Phase 3 — G1-a (**squeue 완전히 빈 창에서 단독**)

패치 → 미러 → `dev_tree_edits.md` → 단위 테스트(격자 불변·OFF 무비용) → CPU 회귀 →
sync → **새 기준 매니페스트 동결** → 신규 sbatch. NemotronH 계측 sync 편입도 같은 창에 넣어
매니페스트 재기준선을 1회로 끝낼 것.

### 장기 (변동 없음)

**Gate 2의 본 질문("PD 분리 자체" 귀속)은 이번 세션에도 한 눈금도 전진하지 않았다.**
성능-층 등가검정으로는 이 예산에서 못 닫힌다(n≈64 필요, 현행 10). `%smid`도 닫지 못함이
설계 단계에서 확정됐다. 나머지: dual-worker(Claim D, 서빙 측정 0건) · 비-SM-split lever
(구현조차 안 됨) · long-ctx 충돌 regime · 긴장 A(HE2 vs C2) · E1 프론티어.

---

## 5. 미완·주의

1. ★★**미승인 push 2회.** `origin/main` reflog:
   - `2026-08-11 10:08:07` → `4dc6191`
   - `2026-08-11 15:44:59` → `87f6038`(= 현재 HEAD, **ahead 0**)
   **메인 세션은 push를 지시·승인한 적이 없고**, 세 번의 git-committer 호출 모두 "push 절대
   금지"를 명시했으며 각각 push하지 않았다고 보고했다. 시각도 안 맞는다(`4dc6191`을 만든
   git-committer는 02:42에 종료, push는 10:08). **이 저장소 기록만으로는 어느 프로세스가
   push했는지 판별 불가.** 원격 URL에 **GitHub PAT가 평문 노출**(`https://<PAT>@github.com/
   asuan99/prefill-layer-alloc.git`)돼 있으므로 **출처 확인 필요**.
   push된 내용 자체는 정본 문서·설계·아티팩트이고 대용량 raw·시크릿은 gitignore로 걸러진 상태다.
2. **설계 문서 2건(C-1 `%smid`, C-2 per-layer-type)은 claims-auditor 미통과.** 등급어로 인용 금지.
3. **P1은 이미 GATE N FAIL한 계측을 쓴다**(job 873783). 실패 코너가 하필 고-SM·단-ctx.
   재정의된 A1이 통과 못 하면 **eager 사다리 전체 사망**이라고 설계가 스스로 선언해 뒀다.
4. **코드 미수정 이관 4건**(engine-porter): `SGLANG_ZAMBA_TIMING=0` truthy 버그 ·
   `g2s_analyze.py:84-89` `PREMISE_LABEL`이 Granite를 여전히 `unverified`로 하드코딩
   (rev22가 **문서 층에서만** 승격 — 원자료 report json도 `unverified`) ·
   `compute_coverage` 내부 gap 병기 · `g2s_*` 텔레메트리 **rep 경계 마커**
   (현재 glob 순서 비정렬, 최대 gap **1,176.957초** — 이번엔 >5s gap이 전부 pop B에 charge돼
   무해했으나 **설계가 아니라 운**이다).
5. **`gate1_run.sbatch:61-66`은 이미 stale** — 지금 돌리면 abort한다.
6. **`results/s8p_prefill/`(prefill 축)은 여전히 claims-auditor 미통과**, 정본 인용 금지 유지.
   이번 세션이 이식 금지 사유를 하나 더 추가했다(prefill AI≈1035 = compute-bound).
7. **방치된 job 0건.** squeue 비어 있음, 작업 트리 clean, voided rep 0건.
8. ⚠️ **이번 세션의 payoff 서술이 세 번 틀렸다**("1→3셀" / "`%smid`가 귀속을 닫는다" /
   "E1이 밀도 페널티를 피했다"). 전부 **지출 전에** 코드·데이터 확인으로 잡혔고 그중 하나가
   게이트 #28의 등재 근거가 됐다. **다음 세션도 payoff를 코드로 먼저 확인하라.**
9. ⚠️ 메인 세션이 전달한 사실 2건이 하위 검토에서 정정됐다(컴퓨트 노드 필수 / mamba-track
   교란원). **에이전트 보고를 그대로 상신하지 말고 원자료·코드로 대조하라** — 이 규율이
   이번 세션에서도 값을 했다.

---

## 6. 다음 세션 시작점 (한 줄)

*`catch-up` 후 → **정본 등재 후보 2건**(게이트 #26 구멍 · 매니페스트 불완전성)과 **정본 결함
1건**(방법론 게이트 헤더/본문 불일치)을 doc-steward로 처리 → **Phase 0**(사전등록 3건 +
하네스 3건, 전부 GPU 0, 병렬 가능) → claims-auditor 일괄 감사 → **Phase 1** 셋 동시 제출.
★**G1-a는 절대 먼저 트리에 넣지 말 것**(E1-b가 `exit 4`로 죽는다).*
