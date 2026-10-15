# prefill span–decode step 동기화: 현 구현 vs MuxWise vs Bullet (2026-10-15)

> 지위: main 트랙 검토 메모. 측정 0, 새 판정 0. [C]=코드·논문 원문, [I]=추론.
> 원문: MuxWise = arXiv 2504.14489v3 (ASPLOS'26, "Towards High-Goodput LLM Serving with Prefill-decode Multiplexing"),
> Bullet = arXiv 2504.19516v2. 페이지 번호는 각 PDF 기준.

## 1. 세 설계 비교

| 항목 | 현 구현 (SGLang v0.5.10 PD-mux + 이 프로젝트 패치) | MuxWise (ASPLOS'26) | Bullet |
|---|---|---|---|
| 프로세스 구조 | **단일 프로세스·단일 이벤트 루프**. true dual worker도 같은 루프에서 두 future를 매 반복 기다림(`multiplexing_mixin.py:1487-1505`) | 단일 프로세스 (green context, p.6) | **prefill·decode 엔진을 별도 프로세스로 분리**. 가중치·KV는 `cudaIpc`로 공유, 상태는 `/dev/shm` 공유 버퍼와 ZeroMQ로 교환 (p.8) |
| SM 분할 수단 | green context stream 쌍(사전 생성). 전환 시 양쪽 stream drain | green context. "재구성 비용 = stream 동기화 1회(µs)" (p.6) | **libsmctrl stream mask** + MPS. 다음 커널부터 즉시 적용, 재구성 중 SM 일부 중첩 허용 (p.7-8) |
| prefill 단위 | 층 단위 span. 층 수 = `65536 // 배치 토큰` → **일(token-layer)이 대략 일정한 정적 예산** | 층 단위(PLs). **N_PL = ⌈T_d · N_T / T_P⌉** — span 시간을 **decode 반복 시간에 맞춤**(추정기 기반, p.9) | 한 step에 고정 층 수 발행, step마다 CPU 동기화 후 다음 결정 (p.7) |
| decode와의 동기화 | **매 반복 lockstep**: decode 1 step 발행 → span 1개 발행 → decode sync | decode 우선 발행(launch <0.5 ms vs prefill 수십 ms). **query 기반 동기화**: CUDA event를 polling하고, 완료된 prefill만 decode 배치에 병합. 나머지는 비동기로 계속 발행 (p.6-7) | **독립 cadence**. decode는 CUDA graph step 단위로 자기 루프에서 돌고, prefill 완료는 메타데이터 메시지로만 전달 (p.8) |
| 동기화가 필요한 시점 | 매 반복 | prefill 완료 병합, 분할 재구성 | prefill 완료 병합(메타데이터), 메모리 할당(atomic lock), 재구성 |
| 분할 결정 | upstream: prefill 시작·끝. 프로젝트: span마다 (R2/SLO 컨트롤러) | SLO-aware dispatcher. 최악 경우 추정(solo 예측 × 경합 상한) | step마다. SM-scaling roofline 추정기 + 온라인 보정 |
| 장문 HOL | 없음. 진행 중 배치는 끝까지 감 | **층 경계 선점**: 짧은 요청이 긴 prefill을 선점(비재귀, 선택적, p.9) | 대기열을 예측 지연 오름차순으로 재정렬(TTFT SLO를 깨지 않는 범위) |
| prefill 배치 형성 | `max_prefill_tokens` 토큰 합 상한, 첫 요청 무조건 | dispatcher 결정 | **산술 강도가 peak에 닿을 때까지** 담음 (Alg. 1) |
| chunked prefill | 금지 (assert) | 쓰지 않음 (층 단위가 대체) | 쓰지 않음 |

**upstream SGLang PD-mux와 MuxWise의 관계** [C+I]:
- 공통점: green context stream 쌍, 층 단위 split prefill, decode 우선 발행 순서, 마지막 span 완료를 `event.query()`로 확인하는 병합(`:1515`).
- upstream에 없는 것: 추정기 기반 N_PL 정렬, SLO dispatcher, 선점.
- ⇒ **upstream은 MuxWise 엔진의 단순화 판**이다. span을 decode 시간에 맞추는 대신 고정 token-layer 예산을 쓴다 [I: 저장소 코드와 논문 대조].

## 2. 질문별 답

### 2.1 decode step과 prefill span을 매번 동기화해야 하나

필요 없다. 원리적으로 동기화가 필요한 곳은 세 곳뿐이다.
1. prefill 완료 → decode 배치 병합. KV·mamba state 슬롯의 소유권이 넘어가는 지점이다.
2. SM 재분할.
3. 공유 메모리 풀의 할당과 해제.

- MuxWise는 1을 event polling으로, 2를 stream 동기화 1회로 처리한다.
- Bullet은 두 엔진을 분리하고 공유 버퍼의 control bit(사용자가 말한 "플래그")로 상태를 주고받는다.
- 현 구현의 매 반복 lockstep은 **단일 스레드 루프라는 구현 구조에서 생긴 것**이지 PD-mux의 본질이 아니다 [I].
- hybrid라서 더 필요한 동기화도 없다. Mamba state는 마지막 prefill 층이 쓰고 병합 뒤 decode가 읽으므로 KV와 같은 완료 이벤트 하나로 충분하다 [I].

### 2.2 decode step이 span보다 짧은가

대체로 그렇다. 크기를 맞춰 보면 다음과 같다 [유도값 I].

| 양 | 값 | 근거 |
|---|---|---|
| decode step (8B급, 작은 배치) | ~11 ms (D92) / ~29 ms (D16) | CONSENSUS §1-31 α, 엔진 측 실측 [C] |
| decode step (Nano-9B-v2, bs 1, 전 GPU, roofline) | 8.35 ms | `costmodel_a100/` |
| prefill 층 1개 (Nano-9B-v2, 7k 토큰, 64 SM) | ≈ 2·9e9·7k / (312e12·0.55) / 56 × 108/64 ≈ **22 ms** | MFU 55% 가정 |
| span (7k 배치 → 9층, 64 SM) | ≈ **0.2 s** | 위 × 9 |

- lockstep에서는 prefill이 진행되는 동안 decode가 span 하나당 토큰 하나만 낸다. 그래서 **ITL ≈ span 시간(decode step의 10–20배)**이 된다. §1-24의 현상과 같은 방향이다.
- 중요한 점: 7k 토큰이면 **층 1개만으로도 이미 decode step보다 길다**. MuxWise식 "span 시간 = decode 시간" 정렬은 층 단위로는 장문에서 도달할 수 없다.
  - 장문에서는 **decode가 span 하나 동안 여러 step을 독립적으로 돌아야만** ITL이 decode step 수준으로 내려온다.
  - Bullet의 프로세스 분리, MuxWise의 비동기 발행이 바로 이 문제를 다룬다.
- multiplexing에서 decode 배치는 작다(이 프로젝트 cap 48, arXiv-Sum 동시 decode ~9). 배치가 작을수록 decode step은 짧고 span과의 격차는 커진다.

### 2.3 span 길이는 정적인가 동적인가

**층 수는 동적이고, 일의 양은 정적이다.**
- 층 수는 배치마다 `65536 // 배치 토큰`으로 바뀐다. 짧은 배치는 전 층을 한 span에, 7k 배치는 9층, 16k 배치는 4층.
- 결과적으로 span 하나의 일은 약 65,536 token-layer로 고정된다.
- decode 시간이나 SLO에는 반응하지 않는다.
- 프로젝트가 더한 span별 재결정(R2)은 **분할 비율**을 바꾸는 것이지 span 길이를 바꾸지 않는다. layer-type 창으로 span 경계를 바꾼 경로(`la_coord_windows`)는 死 판정이다.

## 3. 현 구조가 갖는 장점 (정직하게)

| 장점 | 설명 | 대가 |
|---|---|---|
| **단일 프로세스·공유 상태** | KV·mamba state 풀, radix, 스케줄러 상태가 SGLang 그대로. Bullet식 cudaIpc + 할당 lock + 프로세스 간 메타데이터가 필요 없음. hybrid의 mamba 슬롯 관리(CM-5 `max_mamba_cache_size = cap`)를 손대지 않고 씀 | 단일 루프 결합 → lockstep |
| **공식 CUDA API (green context)** | driver ≥12.4의 정식 API. Bullet의 libsmctrl은 비공식 stream mask라 드라이버 의존성이 있음(인계 프롬프트도 CUDA 13/driver 550+에서 libsmctrl 불가로 적음) | 재분할 때 drain 필요. 사전 생성 격자만 가능 |
| **upstream 유지 코드** | 누구나 SGLang에서 재현 가능한 기준선. hybrid 지원(`forward_split_prefill`)은 이 프로젝트 기여 | MuxWise 고유 기능(추정기, 선점, 정렬) 없음 |
| **예산 기반 정지 상한** | span 일이 입력 길이와 무관(≤65k 토큰)하므로 정지 한 번의 크기에 추정기 없이 상한이 있음 | 상한이 decode step의 10–20배 |
| **측정 가능성** | 반복 단위 결정론적 교차 → telemetry를 반복 단위로 귀속하기 쉬움. 결정 비용 32–36 µs, 전환 ≤0.04 ms 실측 | 기전 분리에 필요한 GPU 시간 계측은 아직 없음 |

## 4. 이 비교가 프로젝트에 주는 함의 [I — 감사 전, 정본 변경 아님]

1. **§1-24 "ITL 꼬리 = monolithic prefill, 고칠 수 없다"의 범위.**
   - 정본이 "고칠 수 없다"고 한 근거는 chunked prefill assert다.
   - 그런데 MuxWise·Bullet은 chunking 없이 **동기화 구조**로 같은 문제를 푼다.
   - 따라서 "ITL 꼬리는 PD-mux의 본질"이 아니라 **"upstream SGLang PD-mux의 lockstep 구현이 가진 성질"**일 가능성이 있다.
   - 정본 문구를 바꾸려면 claims-auditor를 거쳐야 하고, 아래 측정이 필요하다.
2. **싼 검증 실험 두 가지.**
   - (a) `split_forward_token_budget` 스윕 {8k, 16k, 65k}: lockstep 가설이 맞으면 prefill 창 안의 ITL이 예산에 비례해 줄고, TTFT는 반복 수 증가로 늘어난다.
   - (b) decode 커널 GPU 시간과 반복 벽시계를 분리 계측(AGENDA §4.4의 2번).
3. **H-Architecture(true dual worker)의 실제 위치** — ★2026-10-15 사용자 결정: true dual worker는 **AMD 트랙 전용**이다. main과 multi-GPU는 lockstep 기판을 유지하므로, 아래 '독립 박자 패치'는 AMD 트랙의 후보로만 남는다.
   - 현 구현은 host 발행 스레드를 나눴지만 **매 반복 두 future를 함께 기다리므로 cadence는 여전히 lockstep**이다.
   - Bullet 수준의 분리(decode가 span과 무관하게 step을 계속 도는 구조)는 아직 구현되지 않았다.
   - Claim D는 "서빙 측정 0건"이다.
   - 독립 cadence 패치(decode 루프가 prefill 완료 event만 polling)가 다음 엔진 후보다(engine-porter, correctness gate 선행).
4. **선행 연구와 수치를 비교할 때의 단서.**
   - 이 프로젝트의 PD-mux arm은 MuxWise 원형이 아니라 upstream 단순화 판이다.
   - "MuxWise 대비" 문장은 쓸 수 없다. "SGLang PD-mux 대비"로 써야 한다.
   - anchor 주장("Bullet/MuxWise를 hybrid로 확장하는 일이 자명하지 않은가")을 검증하려면 기판 자체에 독립 cadence와 span 정렬이 들어가야 한다. 이것이 venue 서술의 범위를 정한다.
