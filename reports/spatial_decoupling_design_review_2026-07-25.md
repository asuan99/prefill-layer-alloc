# §1-20 Spatial-decoupling 트랙 — 설계 검토 (2026-07-25)

> ★★**HISTORICAL 상단 노트(2026-07-25, 같은 날 후속 정정)**: 이 문서 **§1(신규성,
> venue-strategist)은 신규성 평가 축을 disaggregation으로 잘못 조준**했다(사용자
> 지적). 정정된 판정(축=co-located **multiplexing** — DuetServe/MuxWise/
> SGLang-pdmux/Nexus/Bullet 대조, negative의 (A)green-context종속/(B)
> mechanism-independent 2갈래 분해, cross-substrate publishability 게이트)은
> [`paper/venue_positioning.md`](paper/venue_positioning.md) §0.1이 **정본**이며
> 이 문서 §1을 **supersede**한다. **§2(engine-porter, SGLang v0.5.10 기판
> 실현가능성 file:line)·§3·§4·§5(long-ctx 게이트 연결점)는 disaggregation
> 기판 자체에 대한 사실 판단이라 영향 없이 유효하다** — supersede는 §1(신규성
> 결론)에만 적용된다. 아래 원문은 삭제하지 않고 보존한다(당시 시점의 조사
> 근거로서 이력 가치가 있다).

작성: 이번 세션 catch-up 후속 (사용자 요청 "C = §1-20 공간 decoupling 트랙 설계 검토").
근거: venue-strategist(신규성·학회 fit, web search prior-art) + engine-porter(SGLang v0.5.10
기판 실현가능성, 읽기전용 file:line) 두 병렬 조사 통합. **이 문서는 검토·권고다. 확정
결론은 CONSENSUS/PROJECT_STATUS에만 쓴다.**

트랙 정의: 벡터1(short-ctx 시간축 disjoint) 종결이 열어둔 **공간축** — §1-20이 정량화한
"단일-GPU coupling tax(최적 oracle 92 prefill SM + 24 decode SM = 116 > 108)이라 진짜
+16% headroom은 별도 device pool로 disaggregate해야만 열린다", long-ctx서 90+90>108로 증폭
(H_L5, 미측정). 핵심 연구 훅으로 상정됐던 것: **hybrid disaggregation state transfer =
mamba state O(1) + attention KV O(L)이라 pure-attention보다 mamba 비율만큼 싸다.**

---

## 한 줄 판정

**§1-20 spatial-decoupling을 독립 연구 기여로 추진하는 것은 비권장 — 협공(pincer)에 걸린다.**
(1) **신규성 사망**: disaggregation 훅은 이미 구현·발표됨(재확인일 뿐). (2) **정직한 검증은
비쌈**: 값싼 경로는 가설을 시험 못 하고(자원총량 confound), 가설을 깨끗이 시험하는 경로는
기판 근본 장애를 가진 미구현 신규 아키텍처. **§1-20의 논문적 가치는 기여가 아니라
방화벽(framing)** — "단일-GPU 실패의 mechanistic characterization"이 자산이고, 그 탈출구
(disaggregation)는 DistServe의 알려진 결과다.

---

## 1. 신규성 (venue-strategist) — 훅은 이미 알려짐·이미 구현됨

| 주장 | prior art | 판정 |
|---|---|---|
| PD disaggregation이 coupling을 깬다 (§1-20 상위 명제) | **DistServe (OSDI'24, 2401.09670)** 중심 명제. Splitwise가 heterogeneous HW로 확장 | **이미 확립** — §1-20의 116>108 SM 측정은 이 명제의 SM-단위 재확인 |
| hybrid/SSM 모델 disaggregation | **vLLM v0.20.0 blog (2026-04)**: Mamba+attention 이미 disaggregate, KV+mamba conv/ssm state를 NIXL/RDMA로 전송 | **이미 구현** |
| ★mamba state O(1)이 KV보다 싸다 (핵심 훅) | 위 vLLM 블로그가 **constant-size SSM state를 명시적 "benefit"으로** 취급, zero-overhead 최적화에 이용(~50MB/req). **DUET (DAC'26, 2603.15530)**가 Nemotron-H/Zamba2를 prefill/decode로 disaggregate | **이미 선점** — 훅도, "hybrid가 더 싸다" 정성 framing도 |

**"그래서 뭐?" 리스크 실재**: mamba 절약분 ~50MB/req, KV가 여전히 O(L)로 transfer 비용
지배 → mamba O(1) 이점은 2차 효과로 희석(문헌이 스스로 인정). "constant vs linear가 싸다"는
정의상 자명 = trivial observation. 비자명 문제로 전환할 여지가 좁음.

**유일하게 문헌과 상반되는 자산 = mechanistic negative (venue-strategist 앵글 C)**:
**DuetServe (2511.04791)**는 adaptive SM 분할이 static을 1.3× 이긴다 주장(단 throughput 메트릭,
libsmctrl, 표준 transformer, hybrid 아님). 이 프로젝트는 **conjunctive-SLO goodput /
green-context / hybrid / single-worker reactive** 범위에서 **동적이 static에 진다**를
mechanistic 규명(entanglement §1-4, granularity 42→124ms §1-15, cudagraph 비양립). 이 **상반**을
"왜 같은 아이디어가 이 substrate·이 메트릭에선 실패하나"로 대조하는 것이 crowded positive
문헌이 못 가진 knowledge. **단 이것은 disaggregation 훅이 아니라 negative characterization.**

⚠️ arXiv ID·acceptance·수치는 live search(지식 컷오프 2026-01 이후 다수) → 인용 전 원문 재확인.

## 2. 기판 실현가능성 (engine-porter, file:line) — "처음부터 만들기"는 틀린 전제

**이미 SGLang v0.5.10에 존재하는 것:**
- PD disaggregation 1급 기능(`sglang/srt/disaggregation/`, `disaggregation_mode` server_arg,
  프로세스당 role 하나 + bootstrap/KV transfer).
- ★**hybrid mamba state 전이 전체**: `_send_mamba_state`(mooncake/conn.py:1027)가 conv_state +
  temporal_state 둘 다 전이, 요청당 mamba index 매핑, cross-TP slice까지(mooncake 완전, nixl
  동일-TP 한정). `HybridLinearKVPool` isinstance 분기라 NemotronH/Zamba2/Falcon-H1/Granite-4가
  그대로 흐름. hybrid+disagg 차단 guard 없음.

⇒ **"신규 mamba-state 전이 코드 = 이 트랙의 구현 부담이자 신규성"이라는 longcontext_trace_plan.md
§0.5-C의 전제는 틀렸다.** 전이는 upstream이 이미 했다(신규성이 "어디로 두느냐"로 이동하는데,
그 "어디로"도 §1에서 이미 알려짐).

**"별도 device pool"의 두 해석과 각각의 함정:**
- **(a) 2 물리 GPU** — SGLang disagg가 오늘 그대로 지원(mamba 포함). 신규 코드 ≈0.
  **그러나 2×108 SM = 자원총량 2배 = §1-20의 "한 GPU 116>108" 가설을 깨끗이 시험 못 함**(coupling-tax
  반증에 부정직한 셋업). 런타임도 mooncake/nixl 미설치(venv 확인) → 설치(RDMA NIC) 또는 로컬
  device-copy 백엔드 신규 작성 필요.
- **(b) 1 GPU, green-context로 SM 분할 + 별도 KV pool 독립 스케줄러 2개 + intra-GPU state transfer**
  — **미지원, 신규 아키텍처 작업.** green-context가 **프로세스-내부 CUDA 구성물**이라 두 프로세스가
  한 GPU에서 disjoint SM+KV+intra-GPU transfer를 공유하는 건 근본 장애. SGLang에 스캐폴딩 전무.
  `enable_pdmux`(green-ctx, 단일 스케줄러)와 `disaggregation_mode`(별도 프로세스)는 현재 **직교·상호작용
  없음**.

⇒ **가설을 정직하게 시험하는 건 (b)뿐인데 (b)는 미구현 신규 아키텍처. (a)는 값싸지만 가설을 안 시험.**
true-dual-worker(`PDMUX_TRUE_DUAL_WORKER=1`)는 disagg와 완전 별개 코드 — 손봐도 disagg 안 켜짐.

## 3. 통합 판정 — 협공

| 축 | 결과 |
|---|---|
| 신규성 | **사망**. (b)를 만들어 성공해도 착지점 = "disaggregation이 hybrid에 유리·mamba state 싸다" = 문헌 재확인 |
| 값싼 검증 (a) | 자원총량 confound로 §1-20 가설을 시험 못 함 |
| 정직한 검증 (b) | green-context 프로세스-내부 장애 = 미구현 신규 아키텍처, 비쌈 |
| §1-20의 실제 가치 | **기여 아님, 방화벽/framing** — DistServe 기존 결과. "우리 기여는 단일-GPU 실패의 characterization" |

## 4. 권고

1. **§1-20 spatial-decoupling을 지금 독립 트랙으로 구현·추진하지 말 것.** disaggregation 기판을
   세우는 것(특히 (b))은 신규성 없는 결과에 비싼 엔지니어링을 쓰는 것.
2. **정본 정합성(claims-auditor 규율)**: CONSENSUS/PROJECT_STATUS의 §1-20을 인용할 때 **"disaggregation
   headroom(+16%)은 우리 기여가 아니라 DistServe(OSDI'24)의 기존 결과를 SM-단위로 재확인한 것"**이라고
   명시. hybrid mamba-O(1) 훅은 vLLM v0.20.0·DUET 선점 → "신규 훅"으로 프레이밍 금지. **→ doc-steward
   위임**(canon 문구 수정).
3. **논문 무게중심 재배치**: disaggregation 훅 대신 **DuetServe와 정면 대조되는 mechanistic negative**
   (green-context·conjunctive-SLO·hybrid에서 동적 SM 제어 실패). MLSys-class focused characterization으로는
   가능성, 단 constructive win(H-Policy 검증) 없이는 "아직 부족"이 정직한 상한.
4. **longcontext_trace_plan.md 정정**: §0.5-C "hybrid decoupling 훅(state-transfer O(1) mamba)"을
   신규 훅으로 서술한 부분과 L3s/H_L5의 "hybrid disaggregation 트랙 정당화" 문구에 위 §1-2 사실을 반영
   (전이는 upstream 존재·훅은 선점). **→ doc-steward 위임.**

## 5. ★연결점 — A(long-ctx Stage 0)와 공유하는 단일 게이트

spatial 트랙(H_L5)의 make-or-break 전제는 **"long-ctx decode floor가 실제로 상승해 90+90>108
공간 충돌이 생긴다"** 인데, 이 전제는 A(Stage 0, L−2)가 검증하는 **바로 그 게이트**와 동일하다:
**운영점(cudagraph-ON)에서 long-ctx decode가 SM-binding인가?**

- **Stage 0 = 평탄(non-binding)** → decode floor 안 오름 → 공간 충돌 없음 → **spatial 트랙(H_L5)도
  temporal 트랙(H_L4)도 붕괴**, 벡터1/HE0가 ctx-무관으로 강화. §1-20은 long-ctx서도 안 열림.
- **Stage 0 = binding** → 공간 충돌(90+90>108)이 실재 → H_L5가 측정 가치를 가짐. **그러나 §1의
  신규성 사망은 불변** — 측정해도 착지점은 문헌 재확인(linear-attn KV-bound → disagg 유리).

⇒ **결론: 신규 disaggregation 기판을 세우기 전에 A(Stage 0)를 먼저 친다.** 값싸고(Zamba2 타이밍
재사용) 결정적이며, spatial·temporal 두 long-ctx 트랙 전체를 게이트한다. Stage 0가 죽으면 §1-20
공간 트랙은 자동으로 닫힌다(별도 구현 투자 불필요). 살아도 §1-20의 가치는 characterization이지
disaggregation 훅이 아니다.

## 참고 (인용 전 재확인 필요)
- DistServe OSDI'24 (2401.09670) · DUET DAC'26 (2603.15530) · vLLM hybrid-SSM-disagg blog (2026-04) ·
  DuetServe (2511.04791) · "Aggregation or Disaggregation? Unifying" (2508.01989) · DOPD (2511.20982).
- 기판 file:line: `disaggregation/prefill.py`·`decode.py`·`mooncake/conn.py:1027,1048`·
  `mem_cache/memory_pool.py:395`·`model_runner_kv_cache_mixin.py:654,443`·`scheduler.py:372,979`.
