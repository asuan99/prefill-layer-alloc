# 판정서 — 선행과 엇갈리는 결과의 설계 오염 감사 (논리·코드 층)

2026-10-01 · claims-auditor · **`MIXED`** · 선행 대조 17항목 · 오염 후보 15 · **GPU 0 · 파일 수정 0 · 새 성능 판정 0 · 정본 등급 변경 제안 0(문구 범위 축소만)**

> ★감사 에이전트는 read-only라 **메인 세션이 반환문을 기록**한다. 정량 재집계(result-analyst, 같은 디렉터리 `he0_realized*`)는 병행 중이며
> 이 판정서의 C절 기준(C-1…C-5)을 검증 대상으로 넘긴다. 선행 원문은 로컬 camera-ready PDF(`Papers/Bullet.pdf`, `Papers/Muxwise.pdf`)를 직접 읽음.
> 이 판정서가 인용한 "예비 산술"(노드별 근사 평균, d24/d34 고정 런 평균)은 **인용 금지**(근사, 미등록).

## 단일 판정

메인 세션의 1차 정리("선행과 정면 충돌은 HE0 하나뿐")는 **REFUTED**. 문면상 정면 충돌 4건: HE0 · 얽힘 기전의 정책 귀결(Claim C) · §1-20 "headroom은
디바이스 간에만" · venue 문장 "Bullet 승리 = 프로세스 분리로 coupling을 깸". 이 중 HE0와 §1-20은 따져 보면 다른 질문(ii)이고, **실체 충돌은 2건**:
(a) 과부하 버스트에서 prefill-ward 이동의 부호(우리: TTFT 악화 / Bullet §4.3.1: prefill에 전 SM·decode 지연이 대기 해소) (b) Bullet ablation 오독(Bullet §4.5는 분리 엔진만으로는
불균형, 분할+스케줄러 둘 다 필요).

HE0 자체는 "이 반응형 단일 레버 컨트롤러 계열이 이 trace의 best static을 못 넘었다" 범위에서 **오염 가능성 낮음** — 노드 교락·3× 버그·cudagraph·술어·winner's
curse 모두 부호를 뒤집지 못함. 그러나 **정본 문구("동적 제어는 best-static을 못 넘는다, SLO 엄격도 무관")는 과장**:
- (a) **허수아비 컨트롤러 CONFIRMED(scoped)**: bind+GATE 9런은 d44(idx4)에 한 번도 가지 않았고 d24/d34 혼합이었다.
- (b) **trace 설계가 동적 이득을 거의 0으로 강제**: LO goodput이 도착률로 천장 절단.
- (c) **신규 최대 후보 — running-batch cap 48 = mamba pool 48**: 포화 시 admission이 슬롯 회전에 묶여 decode-heavy에 기계적으로 유리, prefill-ward에 기계적으로 불리.
  HE0·얽힘 부호와 같은 방향이고 Bullet 운영점에는 이 구속이 없다. **부호 반전 가능성: 불명(있음)**.

## A. 분류 (요지)

| # | 우리 결과 | 선행 | 분류 |
|---|---|---|---|
| A1 | HE0 (§1-7, §1-17) | Bullet §4.4 "no optimal fixed SM allocation" | 문면 (i) → 실체 (ii): Bullet static = prefill만 고정·decode 전 SM(겹침), 우리 엔진은 p+d=108 강제로 재현 불가; Bullet 동적엔 재정렬·decode 지연 포함; §4.5 `w/ Partition` 단독은 TTFT 악화(우리와 같은 방향). MuxWise §4.4.1 Fig.18 "rate 안정 시 relatively static"(정합) |
| A2 | 얽힘의 정책 귀결: TTFT 압력에 prefill-ward 반응 → decode 굶김 → TTFT 악화 (§1-4, §1-17, Claim C) | Bullet §4.3.1 버스트 시 prefill 전 SM·decode 일시 지연 | **(i) 실체** |
| A3 | §1-20 "진짜 headroom은 디바이스 간" | Bullet §4.2.2 intra-GPU가 multi-node disaggregation급 이득 | 문면 (i) → (ii): 우리 "disaggregation"은 116 SM 가상 합성 오라클 |
| A4 | venue_positioning "Bullet 승리 = 프로세스 분리" | Bullet §4.5 `Naive` 불균형 | **(i)** 선행 해석이 선행 ablation과 충돌 |
| A5 | §1-6/§1-13 LO는 split 무관심 | MuxWise/Bullet decode 최소 SM | (ii) goodput 함수 한정, TTFT로는 같은 방향 |
| A6 | §1-17 SLO 엄격도 무관 | MuxWise §5 tight에서 우위, loose면 기회 없음 | (ii) 기준선 다름 |
| A7 | §1-18/19 동적이 이기는 regime 없음 | Bullet Azure-Code 버스트 | (ii) n=1–2, 과부하 한정 |
| A8 | 권고 "decode-heavy static" | MuxWise sharegpt 표·Bullet decode 최소 | (iii) 라벨만 위험(d44도 prefill 64/108) |
| A9–A17 | layer-type 死, C2, PD-mux>fused, Gate 2-S, 전환 비용 μs급, 부하 의존 최적 split, ITL 꼬리=monolithic stall, 동거 decode 감속, Diff B 등 | — | (ii) 또는 (iii) 또는 비충돌 |

## B. 오염 후보 (요지)

| 후보 | 근거 | 편향 | 부호 반전 | 해소 |
|---|---|---|---|---|
| (1) 처치 노출 희석 | `src/multiplex/multiplexing_mixin.py` 컨트롤러 idx는 running≠∅ ∧ split_prefill_batch일 때만 적용, 그 밖은 두 arm 동일 auto-revert | null 쪽, 두 arm 공통 | HE0: **아니오**(희석은 부호를 못 만듦) · LO "무관심"(A5): 예/불명 | GPU 0 재집계(C-2/C-3) |
| (2) **허수아비 컨트롤러 CONFIRMED(scoped)** | `sgptvsrv_*.log` 9런 전수: idx4 방문 0, 3런은 d24에서 이동 없음; decode-ward 발화 조건(tpot>51 ms) 대비 로그 tpot 21–43 ms | 동적에 불리 | 이 컨트롤러: 아니오 / 동적 SM 공급 일반: **불명(가능)** | C-1 자리 분해. 정본 §1-10 "1회 이동 후 113회 거부 → d34 영구 고정"은 n=9 arm을 기술하지 않음 — 재확인 필요 |
| (3) best static 사후 선택 | §1-7 | static 유리 | 최고 static 대비 아니오(≲1 SD ≪ 격차), 비교 대상 선택엔 예: "동적 < static"은 상위 2개 static에만 성립 | 문구 축소 |
| (4) 워크로드 영역 | 하네스 NP=200·seed 없음(6개 하위 런이 같은 200 prompt)·phase마다 별도 bench(큐 이월 없음)·HI 과부하·LO 천장 절단 | 동적 이득 구조적 0 | trace 안에선 아니오, **이 trace는 선행 주장을 시험할 수 없음** | GPU(E) |
| (5) 술어·SLO | 하네스 mean-ITL, p95 재채점 순위 보존; §1-17은 rate 8 절벽 대역 | 크기 증폭 | 부호 아니오 / 크기 예 | 임계 사다리 재계산 |
| (6) 기판·기구 | 전 런 cudagraph ON(서버 인자) · 단일 모델 | cudagraph 교락 REFUTED, 모델 계열 미시험 | 불명 | GPU(E-3) |
| **(7) cap 48 = mamba pool 48** | 하네스 `--max-running-requests 48`, 서버 로그 `max_mamba_cache_size: 48`, KV는 구속과 거리 멂, HI bs 46–47 포화 | decode-heavy 유리·prefill-ward 불리(기계적) | **불명 — 가능성 있음** | GPU(E-1) |
| (8) 진행도 레버 부재 | PD-mux가 chunked prefill hard off | 동적 행동공간 협소 | 불명 | 엔진+GPU(E-2) |
| (9) arm 소속 미기록 | `PDMUX_SLO_FEAS_GATE` 미기록, 로그 FEAS 줄로만 추정 | provenance | 추정 맞으면 아니오 | 재집계 확인 |
| (10) 노드·시점 | gpu38/39 분포 | 약함 | 아니오(같은 노드만 남겨도 격차 유지, 근사) | 재계산 |
| (11) σ의 성격 | 5.4 = 격차/pooled SD, 같은 prompt 반복이라 워크로드 분산 누락 | 일반화 불확실성 과소 | 아니오(trace 내) | seed 변동 |
| (12)(13) 3× 버그·warm-up | — | 스케일·공통모드 | 아니오 | 완료 |
| (14) 순환 근거 | §1-17 positioning이 HE0 자기 런 로그에서 도출, "18:1"은 n=1 끝점 | 독립 확증 아님 | 아니오 | 문구 |
| (15) 판정 불가 | stake #1(p+d=108 엔진 강제) | Bullet static 계열 재현 불가 | — | A1은 이 기판에서 부분 판정 불가 |

## C. HE0 판정과 재집계 기준

논리·코드: (1) 동거 구간 밖에서 두 arm은 동일 경로 → 희석은 Δ를 0 쪽으로 줄일 뿐 5.4σ를 만들 수 없음("희석만으로 설명" REFUTED). (2) 동거 구간 안에서 동적 arm은 실제로 다른
분할(d24/d34)에 앉았다 → cap 포화 아래 decode 감속이 슬롯 회전을 늦춰 동거 밖 TTFT로 전파 — "동적이 나빴다"의 내용은 **제어가 아니라 앉은 자리의 열위**. (3) 고유한 동적 페널티 증거는 없음(예비 산술, 인용 금지).

재집계 판정 기준(result-analyst): **C-1 자리 분해**(런별·phase별 d24/d34 체류비 → 같은 노드 static 가중 혼합 예측 → 잔차 R; |R| ≤ 1 pooled SD면 positioning만, R < −2 SD면 고유 페널티, R > +2 SD면 HE0 설명 약화) ·
**C-2 노출**(HI f_co ≥ 0.5면 고노출, ≤ 0.1이면 전파 증거 요구) · **C-3 LO 노출**(작으면 §1-13 재서술) · **C-4 cap 구속**(HI에서 running ≥ 46 시간비 과반이면 후보 (7) 생존) · **C-5 전파**(도착 순번별 TTFT 괴리 누적).

## D. 범위 재서술 제안 (인용 가능 문장 — 정본 반영은 사용자 승인 + doc-steward)

- **D-1 (HE0)**: "A100·Zamba2-2.7B·ShareGPT 200-prompt rate 3↔12 trace(`--max-running-requests 48`, chunked prefill off, cudagraph ON)에서, 분할 인덱스만 움직이는 반응형 컨트롤러 3종(slo, bind, bind+GATE)은 이 trace의 best static(d44)을 넘지 못했다. 최선인 bind+GATE는 d24/d34에 머물렀고(d44 방문 0회) 그 static 대역 안의 값을 냈다. 이 trace는 LO goodput이 도착률로 절단돼 동적 이득이 구조적으로 거의 없으므로, 이 결과는 해당 컨트롤러의 위치 선정에 관한 증거이지 동적 SM 공급의 천장에 관한 증거가 아니다. 시험되지 않은 것: cap 비구속 운영점, 요청 재정렬·decode 지연, 진행도 제어, 예측형 컨트롤러, Transformer 모델, 다른 trace."
- **D-2**: "SLO 엄격도와 무관" → "시험한 두 SLO 점(3 s/60 ms mean-ITL; chat 300/50 ms @ rate 8, 절벽 대역)에서", 크기에 절벽 caveat 필수.
- **D-3 (§1-4, Claim C)**: "running batch cap(= mamba pool) 48이 포화된 과부하에서, prefill-ward 이동은 decode 슬롯 회전을 늦춰 TTFT를 악화시킨다." cap 비구속 운영점 일반화는 NOT-YET-SUPPORTED.
- **D-4 (§1-20)**: "단일-GPU SM 분할 오라클 상한은 작고(n=1–2, 과부하), 결합 오라클은 108 SM을 넘는 가상 구성이다." "진짜 headroom은 디바이스 간에만"은 인용 금지.
- **D-5 (§1-6/§1-13)**: "goodput 함수 기준으로 LO는 split에 둔감." "decode 과다공급은 무해"는 함수·노출 조건 없이 인용 금지.
- **D-6 (venue_positioning)**: "Bullet 승리 = 프로세스 분리로 coupling을 깸" 삭제 권고 → "Bullet ablation은 분할 단독으로는 TTFT가 악화된다고 보고하며, 이는 HE0와 같은 방향이다."
- **D-7**: "decode-heavy"는 격자(16–54) 상대 라벨, d44에서도 prefill 다수(64/108) 병기.
- **D-8 (§1-10)**: "d34 영구 고정" → "9런 중 3런 d24 고정, 나머지 d24↔d34, d44 미도달"(C-1 확인 후).

## E. 양성 대조(선행 재현) 최소 설계 — 사전등록 + 규칙층 감사 선행

- **E-1 (엔진 수정 불필요, 우선)**: static {d24, d34, d44, d54} + bind+GATE + bind no-gate × cap {48, 192}. `--max-mamba-cache-size = cap`, 전 arm `cuda_graph_max_bs=256`, triton·cudagraph ON, 한 job 안 무작위 순서(같은 노드), 블록별 seed 변동·블록 내 공유, `PDMUX_SLO_FEAS_GATE` 기록, telemetry ON. 워크로드 1차 3↔12, 보조로 cap별 λ\* 비례 rate. SLO 1차 정본 술어, 2차 Bullet식 P90. n ≥ 4 블록(no-gate ≥ 8). best static은 별도 보정 블록 argmax로 사전 등록. Δ 블록 짝 95% t-CI, ±3%. **분기**: cap48 패 ∧ cap192 승/동률 → HE0·얽힘 부호는 cap이 만든 것(설계 결함 확정) · 둘 다 패 → cap 무관, Bullet 스케줄러 성분/기판으로 귀속 → E-2 · 둘 다 승 → 컨트롤러 결함 · cap192에서 static argmax가 prefill 쪽 이동 → 실전 권고 재검토. 예산 ≈ 10–20 GPU-h. cap과 용량이 함께 움직이는 것은 등록 caveat.
- **E-2 (엔진 작업)**: Bullet식 버스트 arm — 큐 압력 시 idx0(prefill 전 SM, decode 지연) + 요청 재정렬. correctness 게이트 선행(= 설계 기준선의 pause manager).
- **E-3 (모델 계열)**: Transformer(Qwen2.5), 모델별 λ\*.

## 반증 실패 (오염 아님으로 확인)

cudagraph OFF(전 런 ON) · 3× 버그(스케일, 순위 불변) · 노드 교락(근사상 유지) · winner's curse(격차 대비 작음) · mean-ITL 술어(p95 재채점 순위 보존) · warm-up(공통모드) ·
전환 비용(μs급) · 동적 하네스 경로 자체 페널티(job 896565 anchor d44가 static 대역과 정합, n=1, 크기 비교 금지) · 희석이 부호를 만들었을 가능성(코드상 불가).

## ★메인 세션 독립 재검증 (기록 시점, GPU 0)
- `sgptvsrv_bind_rep41_L3H12_856930.log`: `max_running_requests=48`, `max_mamba_cache_size: 48`, `disable_cuda_graph=False` — 확인.
- bind 계열 서버 로그 idx 방문 집계: HE0 캠페인 런(8569xx/8570xx/855250)은 idx 2·3만 등장, idx 4는 job 896565(F2 anchor d44 런, HE0 캠페인 아님)에만 — "d44 방문 0회" 확인.
