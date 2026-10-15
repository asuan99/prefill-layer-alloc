# `--max-running-requests`(cap) 48의 출처와 비용 모델이 설명해야 할 것 (2026-10-15)

> 지위: main 트랙 검토 메모. 측정 0. 수치는 A100 roofline 유도값(`costmodel_a100/`, 단일 GPU 전 SM, 경합 없음)이며 하한 추정이다.
> [C]=코드·문서 사실, [I]=추론.

## 0. 트랙 귀속 메모 (사용자 지시, 2026-10-15)

- **true dual worker**(독립 박자 실행)는 **AMD 트랙(attention–mamba disaggregation) 전용** 자원으로 귀속한다(2026-10-15 정정: multi-GPU 아님).
- **main의 lockstep PD-mux 구조가 그대로 multi-GPU PD-mux로 확장·이양**된다(M3 PREREG도 true-dual을 TP에서 거부, `multigpu_guard`).
- main(단일 GPU)은 현 lockstep 기판을 그대로 쓴다.
- ⇒ 단일 GPU 결과는 **"lockstep 기판에서"**로 범위를 명시한다. lockstep 정지는 고칠 대상이 아니라 **특성화 대상**이다. span 예산 스윕이 그 특성화다.
- 이 메모는 다음 handoff 때 `tracks/main.md`에 기록하고, AMD·multigpu 트랙 반영은 promote 몫이다.

## 1. cap 48은 어디서 왔나 [C]

- **최초 등장**: 2026-07-04 커밋 `40d6869`(P1.4 Task2, NemotronH-8B 서빙 sbatch)의 서버 기동 인자 `--mem-fraction-static 0.82 --max-running-requests 48`.
  - 커밋 메시지와 문서 어디에도 **선택 근거가 없다**. 이후 모든 캠페인이 이 값을 상속했다.
- **정본의 기존 지적**:
  - "얽힘 기전의 축인데 sweep 없음" (`research_arc.md:235`, `layertype_dynamic_NEGATIVE_2026-08-04.md:223`)
  - rate 10·12에서 시스템 인구가 cap 48을 초과(63.96·83.06)해 cap이 구속 (CONSENSUS §1-4 관련, `:2083`)
  - λ0·E2 achieved는 "48-구속"이라 시스템 용량이 아님 (λ0R-3, E2C-10)
  - multigpu는 TP에서 48·N으로 결정 (2026-10-12)
- **코드 사실 (§1-23)**: radix OFF이면 `max_mamba_cache_size = cap`.
  - hybrid에서 cap은 **admission 상한 + decode 배치 상한 + mamba state pool 크기**를 동시에 정한다. 순수 Transformer에서는 admission만 정한다.
  - cudagraph 캡처 bs 목록도 cap에서 잘린다 (`cuda_graph_runner.py:454-480`).

## 2. cap을 묶는 세 제약 (Nano-9B-v2, A100, `mem_fraction_static` 0.82)

풀 메모리 ≈ 0.82×80 GB − 가중치 17.8 GB ≈ **47.8 GB** (activation·cudagraph 예약 제외 → 실제는 더 작음).
요청당 메모리 = state 0.1435 GB(fp32, 길이 무관) + KV 16 KiB × L.

### (1) 메모리 상한 `C_mem = Pool / (s + k·L)`

| ctx L | 요청당 GB | C_mem |
|---|---|---|
| 2k | 0.177 | ~270 |
| 7.1k (arXiv-Sum 평균) | 0.260 | ~183 |
| 16k | 0.412 | ~116 |
| 64k | 1.217 | ~39 |
| 128k | 2.29 | ~20 |

- cap 48은 **ctx ≈ 52k 이하에서는 메모리보다 먼저 문다**. 그 위에서는 KV가 먼저 문다.
- 정본의 "KV 6.6× 과공급"과 같은 현상이다. arXiv-Sum 평균 길이에서 KV 풀은 48개 요청분의 약 4배다 [I].
- hybrid 고유 항: state pool은 **cap × 0.1435 GB가 사용 여부와 무관하게 예약**된다(48 → 6.9 GB). cap을 올리면 KV 풀이 그만큼 줄어든다.

### (2) ITL SLO 상한 `T_dec(C, L, D) ≤ T_ITL`

decode step roofline (전 GPU 108 SM, ms):

| L \ bs | 1 | 16 | 32 | **48** | 64 | 96 | 128 | 192 |
|---|---|---|---|---|---|---|---|---|
| 2k | 8.3 | 10.7 | 13.2 | **15.7** | 18.2 | 23.2 | 28.3 | 39.4 |
| 7.1k | 8.3 | 11.3 | 14.5 | **17.6** | 20.8 | 27.1 | 33.5 | 47.2 |
| 16k | 8.4 | 12.5 | 16.9 | **21.2** | 25.6 | 34.3 | 43.0 | 61.5 |
| 128k | 9.3 | 27.2 | 46.3 | **65.5** | 84.6 | 122.8 | 161.0 | 238.4 |

- 16k 이하에서는 decode step만 보면 50 ms 안에 bs ~190까지 들어간다. **cap 48은 ITL SLO에서 나온 값도 아니다.**
- 128k에서는 bs ~36부터 50 ms를 넘는다(attention KV 읽기가 지배).
- D SM으로 줄이면 더 느려진다. opchar 격리 실측: memory-bound 연산자는 44 SM에서 108 SM 속도의 ≥86%. 이 표는 하한이다.
- **lockstep 기판에서는 prefill 창 안의 ITL이 T_dec가 아니라 span 시간(~0.2 s, 7k 배치·64 SM 유도)이 정한다.** 그래서 이 상한은 prefill 창 밖에서만 의미가 있다.

### (3) 수요 하한 (Little) `C ≥ λ · E[출력] · E[ITL]`

| 워크로드 | λ | E[출력] | E[ITL] | 필요 동시 decode |
|---|---|---|---|---|
| arXiv-Sum | 1.5 | 200 | 20 ms (prefill 밖) | ~6 |
| arXiv-Sum | 1.5 | 200 | 200 ms (lockstep 정지가 지배) | **~60 > 48** |
| ShareGPT (HE0 rate 12) | 12 | ~270 | ~25 ms | 실측 인구 83 > 48 [C] |

- 수요가 cap을 넘으면 admission이 막힌다. 그러면 대기열이 생기고(W_queue↑) TTFT가 폭발한다. 이것이 **얽힘 사슬(§1-4)의 정량형**이다.
- 같은 arXiv-Sum이라도 ITL이 decode step 수준이면 cap이 비구속이고, lockstep 정지 수준이면 구속이다.
- ⇒ **cap의 구속 여부는 기판의 동기화 구조에 달려 있다** [I]. 이 점이 위 0절의 lockstep 범위 명시와 직결된다.

### 실현 가능 창

```
λ·E[out]·E[ITL]  ≤  C  ≤  min( C_mem(L), C_SLO(L, D) )
```

- 위 창이 비면(하한 > 상한) 그 rate는 용량 밖이다.
- cap 48은 이 창에서 도출된 값이 아니다. 고정된 상수다.

## 3. 비용 모델이 설명해야 하는 것 (CM-5·CM-6·CM-1 연결)

1. **cap은 단독 손잡이가 아니라 (cap, D)의 결합 결정이다.**
   - bs가 커지면 T_dec가 늘고 필요한 decode SM이 늘어, prefill SM이 줄고 span 정지가 길어진다(§1-24).
   - 정본 결론 "최적 split = max(모델 floor, 부하항 ∝ λ×출력길이)"(§1-5)의 부하항이 곧 위 수요 하한이다.
   - 비용 모델은 C와 D를 함께 내야 한다.
2. **hybrid에서 cap의 비용 구조가 다르다.**
   - Transformer: cap ↑ = 메모리 비용 0 (KV는 실제 사용분만 할당).
   - hybrid: cap ↑ = state 예약 s·C만큼 KV 풀 감소.
   - generic 예측기(KV만 보는 모델)는 hybrid의 C_mem을 과대평가한다. 이것은 **anchor 주장 1의 직접 사례 후보**다. 다만 측정으로 확인해야 한다.
   - 단 radix ON이면 mamba 슬롯 ratio가 3+α가 돼 running 상한이 1/3 이하로 준다(`kv_cache_mixin.py:871-874, 391-402`). CM-7 확장 때 함정이다.
3. **긴 ctx에서 무엇이 먼저 묶이는가 (X-6).**
   - 위 표대로라면 cap 48일 때 ~52k를 경계로 구속 자원이 cap(=슬롯)에서 KV로 바뀐다.
   - Mooncake Conversation(p99 85k, 최대 126k)은 이 경계를 가로지른다.
   - 측정할 것: admission 거절 사유(KV / cap / slot)를 계측해야 이 경계를 실측할 수 있다(AGENDA §4.4 5번).
4. **cudagraph 범위.** cap이 캡처 bs 목록의 상한이다. cap을 바꾸면 graph 커버리지와 패딩 낭비가 같이 바뀐다.

## 4. 결정 안건 (AGENDA §5에 추가: 안건 10)

**cap 처리 방식**:
- (a) 48 유지 + 구속 여부를 셀마다 공시 (현 관행)
- (b) 워크로드별로 위 창에서 도출한 값으로 등록
- (c) cap을 실험 축으로 {48, C_mem(L̄)의 1/2, C_mem(L̄)} 스윕

**권고**:
- G0: (a) + 구속 공시 + admission 거절 사유 계측.
- 비용 모델 검증 단계: (c)를 같은 Job 안에서 수행. 반드시 `--max-mamba-cache-size = cap` 명시 규칙(§1-23)을 지킨다.
- 이유: 기존 결과와의 연결을 유지하면서, cap이 얽힘의 축이라는 정본 지적을 처음으로 측정하기 위해서다.
