# COSTMODEL — 지금까지 드러난 비용 구조 (진척판, 2026-10-13 신설 · 2026-11-10 갱신)

> **지위**: 측정 결과를 새로 만들지 않는다. 이미 정본(`reports/CONSENSUS.md` §1, `reports/cost_model_design_space_2026-10-01.md`,
> 감사 판정서)에 있는 사실을 **비용 항(CM-1…CM-7) 축으로 다시 배치**한 장부다. 연구 축은 `reports/system_design_anchor_2026-09-30.md`
> (스케줄러 확장)이며, 이 장부는 그 과정에서 **cost model이 얼마나 성립했는지**를 보여 주는 진척판이다.
> 규칙: 현재 형태만 적는다(이력은 git). 한 항 최대 6줄. 새 근거는 결과 분석의 "cost model 기여" 절에서만 들어온다.
> 상태 어휘: `가설` · `데이터 있음(미감사)` · `형태 확인` · `계수 확정(범위 한정)` · `서빙 검증` · `반증`

## 축 정의 (고정)

`비용 = f(E 엔진/시스템 부하, M 모델 구조, U 워크로드)` — **배치 ctx 분포 = U × E**. M은 배포 시 상수라 단일 모델 안에서는 변수가 아니다
(조성 귀속은 모델 hold-out에서만). 상세 `cost_model_design_space_2026-10-01.md` §1.0. ★**수식 기반(고정, 결정 27, 2026-10-22)**: 층별 FLOPs·상주 메모리·decode 트래픽의 닫힌 식은 `reports/cost_model/ANALYTIC_COST_FORMULAS.md`(analytic-v1, 계산 단일 출처 `workspace/engine-port/scripts/costmodel/analytic_terms.py`)이고, 각 CM 항은 이 식을 사전(prior)으로 두고 측정으로 효율·PD-mux 항을 덧붙인다 — 요약은 파일 끝 절.

## 진척 요약 (한눈에)

| 항 | 무엇 | 상태 | 다음 데이터 |
|---|---|---|---|
| CM-1 | decode step 시간 `T_dec(SM, bs, ctx)` | 형태 확인(Nano-9B-v2, 고정 배치·저부하; **T1 F1은 비분할 P=108만 채택**, sticky-D는 F2/F3) / **hybrid 고유 항 미입증**(연결 근거 부재, 반증 아님) / **D\* 모형(결정 61) 함수 형태: 고원(D34–D64)·D24 비율 확인(E-1…E-5 HIT), A6 경성 ITL 문턱은 D16 LO에서 깨짐(E-6·E-7 MISS) — 플래그 OFF 대체물·1 GPU·W1 EVAL 셀 한정** | hold-out 서빙(고비용)·opchar(특성화) · 전력 상태는 T1만 확인 |
| CM-2 | prefill span 시간 `T_span(L, SM)` | 형태 확인(마이크로벤치·격리 커널) / **서빙 floor 계수 확정(범위 한정: Nano-9B-v2 무경합)** / **T2(P=108) 전부 채택, q 1.088–1.097 `PREFILL_NEAR_LINEAR_SCOPED`(Nano, 라벨)** · 장문 98k·124k `LONG_SERVED`(서빙 사실, 함수 형태 연장 아님) / chunk -1 장문 상한 ≈94.8k = Triton conv int32 오프셋(성능 항 아님) / **PM5·P=64 chunk 중앙 실측/예측 ≈1.0, 첫 chunk +15%(사후·1 GPU); "γ 1.3–1.7배 과소예측" 주장 철회(감사 REFUTED)** / **py3.14 이미지에서 SGLang piecewise CUDA graph 부팅 불가 → SGLang 계열 prefill 전부 eager(Phase B A7)** | T2 전력 상태·T2′ 전력 상태 NOT-YET-SUPPORTED(원자료 필요) · A1 span-SM, opchar 후속 |
| CM-3 | 전환·경계 비용 | **계수 확정(범위 한정)** — 무시 가능 | 컨트롤러 구동 경로만 미측정 |
| CM-4 | 동거 간섭 `δ(p,d)` | **존재 확인(N2 Job-EA)** / **T3: 동거 창 SW power cap 노출 96–100% 대 단독 0–4%, ρ 1.13–1.19는 총효과로만(CONFIRMED)** / 크기·D 순서는 NOT-YET-SUPPORTED / J1: cap 노출 순서가 shape 의존이라 "PD-mux가 클럭을 낮춘다" 일반 가정 불가 / **lockstep 정지 진단(VESSL·Nano·flashinfer·65536·8k/32k·PHASE ON 범위): 동거 간격은 span 시간에 거의 반응 안 함(기울기 0.0015, CI [0.0005, 0.0027]), 별도 정지 항 ≤1 ms 해상도 안 미관측(감사 r1 PLAUSIBLE 조건부)** / **piecewise OFF SGLang 기본 구성의 도착당 간극 이봉 ITL(3 Job·GPU 재현)·짧은 프롬프트 상수항 관측, 원인 미측정(Phase B 감사 r1 A1–A3·A5·A6, 기술값)** (★2026-10-29 범위 정정, 감사 r1 #11: 이 run들은 적체 영역[span > 동거 decode step, prefill 미동기 발행]이었고 이 영역의 무반응은 루프 구조상 예상값이며 박자 영역은 측정되지 않았다) | N2 후속 F-1…F-4(사용자 결정) · 동적 arm 전환 drain 대기 미측정 · N3 Job 3 첫-chunk 차단 원인 미해결 · Phase B·stock_quick: piecewise OFF SGLang의 도착당 간극(≈64 ms)·상수항 관측(기술값), 원인 미측정 · upstream 기본(piecewise ON) 미측정 |
| CM-5 | 메모리·admission 예산 | **형태 확정(코드 사실)** · **s·k 부팅 로그 사실 CONFIRMED**(Nano: k 16,382 B/토큰, s 143.5 MB/요청; L\*≈8.8k [유도]) / **T4 산식 B 재현은 NOT-YET-SUPPORTED**(cap 비구속 점 2개: HELD 1·MISS 1) / 두 구간 특성화 미완 | T4 MISS 원인 = 클러스터 볼륨 원자료 필요 |
| CM-6 | 대기열·TTFT | 구조 확인(entanglement 사슬) / 분해 규약 확정 / J1 B에서 처리율↔SLO 역순은 **미확립**(G0 몫) / stock_quick: SG W1은 ITL 문턱 절벽, W3 등록 술어에서 SG ≥ 모든 static(0.506 이상에서만 뚜렷; n=2·다른 Job·기술값) / Phase B: 버스트 120요청 사다리·600 s 안정성 규칙의 **규칙 출력만**(용량 순서·크기 인용 금지, A8·A9) | N3 용량 · G0 LO·중간 rung |
| CM-7 | prefix cache (radix ON) | 가설 | 보류(radix OFF 우선, 2026-09-24 결정) |
| (E) | 컨트롤러·결정 비용 | **계수 확정** — 무시 가능 | — |
| (확장) | N-GPU·TP, 조성, 층 배치(AMD), 에너지 | **조성 P1·P2·P4 CONFIRMED(J-H1 한 Job), P3 조건부 · TP2 F1·F2·F5 CONFIRMED(Nano, J-C), F3·F4 NOT-YET-SUPPORTED** · N-GPU/AMD/에너지는 가설 — 정확성 게이트·캡처 경로·merge preflight(CPU)만 통과. M3는 J1(N=1)만 완주, J2·J4·J8은 2026-10-18 21:14:03Z ±40 ms 동시 종료(경과 9.511–9.515 h, 주기 집행 상한·일괄 종료와 구분 불가, 원인 미확인) → **G_sym 전 N 미정의**, N>1 성능 수치 0, G0 뒤로 연기(2026-10-19 결정) | 각 트랙 TRACK.md · G0 PREREG |

## 항별 장부

### CM-1 decode step 시간
- 제안 형태: attn 층 `∝ bs·ctx`, SSM 층 `∝ bs`(ctx 무관), MLP `∝ bs`, 계열별 SM 스케일링 `s_attn(SM)`, `s_ssm(SM)` (`cost_model_design_space` §1.1).
- **확정**: decode는 SM에 둔감하지 않다 — decode ITL SM16→92 **2.36–2.91×, 4 arm 모델-무관(pure Mamba2 포함)** (C2 `CONFIRMED(scoped)`, 레버 존재만). "decode는 memory-bound라 SM 둔감"(X-7)은 이 기판에서 반증. attention 비중·SM 민감도가 ctx에 따라 커지는 방향(§1-5)은 확정, 절대 knee 값은 마이크로벤치라 인용 금지. "hybrid는 KV가 작아 배치 여유가 크다"는 REFUTED(CM-5).
- **미입증(가장 중요한 빈칸)**: "layer-type별 decode SM 스케일링 차이(a)는 유효 측정 0건(결함 job 철회, per-type UNAUDITED)으로 미입증이다." ★정정(2026-10-19, claims-auditor `anchor_audit/VERDICT_claimE_first_arrow_2026-10-19.md`): 10-12 의견("증거 사슬이 첫 화살표에서 끊긴다")은 반증이 아니라 **연결 근거 부재**다 — Diff B·C2·opchar는 s_attn≈s_ssm의 근거가 아니다. a는 Claim E의 필요조건이 아니다(§6 Claim E는 "미검증 핵심 가설" 그대로).
- "generic 예측기로 충분" 가설은 기각되지 않았다(검증되지도 않았다). 단일 모델 내 예측표 발산은 재매개화 항등식이며, hybrid 귀속은 모델 hold-out(또는 충실한 이식 generic)에서만 시험 가능하다(A4 K2). N2 PROFILE은 인용 금지.
- **T1 서빙 측정(Nano-9B-v2, J-A 2 round·고정 배치·저부하, `cap_char/RESULT_costterms_2026-10-26.md` rev1 §17 · 감사 `AUDIT_costterms_result_r1_2026-10-27.md`)**: "F1 `a + b·bs + c·bs·ctx`는 비분할(108u·108v)만 채택되고(`FORM_INADEQUATE` 0), sticky-D 5개는 bs 96 한 셀(ctx 2k) 외삽 13.6–18.3%로 F1이 떨어져 F2(D16·34·44·64)·F3(D24)가 채택됐다. J-A T1에서 `CLOCK_SENSITIVE`는 없었다(노출 20% 초과 창 0)." 조성 변형 a22·a31은 `FORM_INADEQUATE_T1_108v`(격자 안 보간만). 두 Job T1 F1 계수 차 ≤ 2.7%는 **PLAUSIBLE**(인용 금지). AJ-2 J-H1 병기(b(a22)·b(a31)·기울기)는 `NA_FORM`이라 **인용 금지**.
- **D\* 모형 사전 예측 채점(implfix R Job, 결과 감사 r1; 범위: 봉인 EVAL이 아니라 엔진 digest `0a5d9c3c` 플래그 OFF 대체물·GPU 1장·W1 EVAL 셀)**: E-1…E-5 HIT(고원 max/min 1.015, G24/G44 0.733), E-6(G16/G44 ≤ 0.30 → 0.395)·E-7(LO D16 좋은 비율 ≤ 0.5 → 0.562) MISS, E-8 점추정 MISS(HI D64/D44 0.983, paired 95% CI [0.937, 1.028] n=4 미분해). 깨진 가정 = A6 "ITL 경성 문턱"("저부하에서는 연속적"은 [가정]). 예측 +7%였던 D64의 HI 이득은 나타나지 않았다. 식 수정은 채점 뒤 별도 절차. `implfix:workspace/engine-port/results/policy_implfix/RESULT_policy_implfix_2026-11-01.md` ★사후 재현 한정(감사 r1 2026-11-08, `reports/audit/2026-11-08_unaudited_r1/AUDIT_design_defects_policy_memos_dstar_r1_2026-11-08.md` §3-4): D\* 식과 CAP 결과의 대조(구속 제약·W1 순서·D\* 방향·W3 D64 불가)는 식을 본 뒤의 사후 재현이고 실질 시험은 ITL 벽 위치(D24–D34)뿐, 예측 시험은 위 R Job 채점뿐.
- 연산자 단위(opchar, 특성화 전용·격리 커널, `AUDIT_opchar_2026-10-13.md` Safe wording 범위): decode에서 SSM state update 비중 B 32→128에서 36%→64%(ctx 2048); 메모리 바운드 연산자는 44 SM에서 108 SM 속도의 ≥86%; GEMM 계단 = wave quantization(104→108 점프).

### CM-2 prefill span 시간
- **확정(마이크로벤치, 서빙 아님)**: attn/SSM SM 민감도 비 Diff B는 L≤512에서 열리고 L≥1024에서 ≈1.0(§1-3 WIDE 스윕). 런타임 per-type 분할은 granularity 비용이 압도(TPOT 42→124 ms, §1-3/§1-15) ⇒ 층 종류 차이는 **존재하나 층 경계 SM 재분할 레버가 아니다**(이 구현, Diff B≈1, Zamba2·triton·no-cudagraph micro; 층 종류별 비용 차 Diff A를 span 시간 예산에 쓰는 레버는 미시험 — 아래 열린 질문; 2026-10-29 범위 정정, 감사 r1 #14). 연산자 단위(opchar, 격리 연산자 시간 합): Mamba 층 ~66–71%(L=1k–32k), "모델 시간의 %"로 인용 금지.
- **서빙 floor 계수(J1 N=1 감사 CONFIRMED·범위 한정, 2026-10-19)**: "무경합 TTFT floor ≈ 9 ms + 0.111 ms/토큰 × L (L 1k–16k, 기울기 기준 ≈ 9.0k tok/s; Nano-9B-v2, 108 SM, chunked-prefill −1, VESSL A100, Job 4개·boot 8개에서 16k 산포 ±0.6%)." 이 기울기와 비용 항 T2의 대비는 "mem 0.82 결과, 기술값"이며 "mem 0.6에서 J1 재현"·"실효 47% 피크"는 금지.
- **T2(P=108만, `cap_char/RESULT_costterms_2026-10-26.md` rev1 §17 · 감사 `AUDIT_costterms_result_r1_2026-10-27.md`)**: Nano q 1.088–1.097 `PREFILL_NEAR_LINEAR_SCOPED`(**CONFIRMED 라벨**, 등록 규칙 적용 결과이며 계수 일반화 아님); Llama·attention 증가 변형은 `_SUPERLINEAR_SCOPED`. T2·T2′의 전력 상태(`CLOCK_SENSITIVE`)는 원자료(클러스터 볼륨) 없이 **NOT-YET-SUPPORTED**.
- **장문(사실, 성능 항 아님; scoped)**: Nano 98,304·126,976 토큰 장문 점이 서빙됐고(`LONG_SERVED`), TP2 PD-mux는 `108u`(D44 아님) 한정 124k다. "124k까지 함수 형태 연장"은 금지. chunk -1 상한은 Triton `causal_conv1d` int32 오프셋 약 94.8k(T ≥ 94,793)이고 PD-mux 고유 아님; int64 패치는 gate G-β로 정확성만 확인(감사 `cap_char/probe_128k/AUDIT_probe_128k_2026-10-20.md` 기전 PLAUSIBLE). "PD-mux는 64k를 넘을 수 없다"·"OOM 때문" 금지.
- **implfix chunk 단위(감사 r1 문안)**: "PM5·P=64에서 chunk 중앙 비는 약 1.0이고 첫 chunk가 약 +15%다(사후, 1 GPU, 계측 boot). 극소 extend에서는 FLOPs 식에 없는 층당 고정 바닥(약 1 ms/층)이 지배한다(warmup 트래픽 기술값)." "γ 1.3–1.7배 과소예측"은 REFUTED로 철회(잔차는 span 최대값 통계·첫 chunk·오염). implfix r_span·dose 수치(1.30·1.29·0.552·0.503 등)는 인용 금지 — 봉인 PC 축약이 warmup·프로브 prefill을 섞은 하네스 결함.
- **SGLang prefill 경로(Phase B 결과 감사 r1 A7, 2026-11-07)**: "이 이미지(py3.14)에서 SGLang v0.5.10은 NemotronH의 piecewise CUDA graph로 부팅되지 않았다(`typing.Union` AttributeError, 2회째 관찰). 따라서 이 캠페인의 SGLang 계열 prefill은 전부 eager다." upstream 기본(piecewise ON)의 prefill 비용은 미측정.
- 미감사·인용 금지: prefill 축 SM 스윕(TTFT~L 기울기비). 열린 형태 질문: MuxWise `N_PL` 층 균일 가정(X-4)이 비용 가중 층 수로 바뀌어야 하는가. 정본 수식 §8 "균일 η" 승격은 roofline r1로 불가(층·집계 미분리).

### CM-3 전환·경계 비용
- **계수 확정(agnostic 경로·2모델·이 config 한정)**: `s ≤ 0.04 ms/전환`, `d ≤ 0.07 ms/경계` (§1-8/§1-34). 진짜 비용은 전환 자체가 아니라 prefill 생애주기 경계.
- 미측정: 컨트롤러 구동 전환 경로, PAUSE 진입/해제, SSM chunk 경계 state 이월.

### CM-4 동거 간섭
- **KISTI·Zamba2-2.7B·triton의 이 루프에서 관측**(구 "구조 확인"; 2026-10-29 범위 정정, 감사 r1 #10, 정지 기전 미확정·VESSL/Nano 미관측): ITL 꼬리는 decode step time이 아니라 최장 프롬프트 prefill과 시간상 겹치는 정지가 지배하며 D(decode SM)에 단조 증가 — conjunctive goodput의 ITL 항이 decode-SM 레버와 반대 부호 항을 내장(§1-24). PD-mux는 chunked prefill과 비양립(엔진 assert).
- **존재 확인(N2 Job-EA `7e09cc3`, 라벨 8/8 `COLOC_SLOWER_GE3` CONFIRMED(scoped))**: "창별 실측 클럭 하락을 전부 빼도 64/64 짝의 ln ρ는 ln 1.03을 넘었다 … `COLOC_SLOWER_GE3`의 존재는 클럭 성분만으로 설명되지 않는다. 이 문장은 크기 진술이 아니다."
- **크기는 미분리**: ρ는 동거 + 전력 한도 클럭 하락의 합성값 — "간섭 크기" 인용 금지(N2C-102), "D·b·c 무관" NOT-YET-SUPPORTED. 분리 경로 = F-1·F-3(사용자 결정 대기). ρ는 적체 영역 측정(8k 프롬프트·예산 65536의 8층 혼합 span; 2026-10-29 범위 정정, 감사 r1 #12).
- **J1 허용 문안 3·5(감사 2026-10-19)**: "J1에서 SW power cap 노출률의 arm 간 순서는 shape에 따라 달랐고(B에서는 PD-mux가 낮고 X에서는 높음) 부하·전력이 매개한다. 'PD-mux가 클럭을 낮춘다'는 일반 가정으로 쓸 수 없다." / "J1 GPU0 SW power cap(0x4) 노출 60.6%, 셀별 노출은 A 0–6%·B 47–93%·X 80–99%(HI 셀), 400 W 한도. 등록 throttle 공변량은 이것을 잡지 못한다. B·X의 arm 비는 400 W 한도 아래 클럭 하락을 포함한 총효과로만 보고하며, 매개와 교란은 분리하지 않는다."
- **lockstep 정지 진단(감사 r1 PLAUSIBLE 조건부, 2026-10-27; `anchor_audit/VERDICT_lockstep_diag_claims_r1_2026-10-27.md` P1, 기술값·범위 한정)**: "VESSL A100-SXM4 · Nemotron-Nano-9B-v2-Base · flashinfer · cudagraph ON · split 예산 65536 · 프롬프트 8k(T3는 32k 포함) · PHASE_EVENTS ON 조건의 N3(Job 2개)·N2-EA·T3 run에서, 동거 decode 간격은 span 시간(W/n_chunks)에 거의 반응하지 않았고(기울기 0.0015, 95% CI [0.0005, 0.0027]) 풀링 토큰 ITL p99 꼬리는 동거 decode step 수준이었다. 이 범위에서 CM-4의 별도 정지 항은 해상도(≤1 ms) 안에서 관측되지 않았다. ρ_on 1.156–1.193은 동거·전력 한도 클럭 하락·관측자 비용의 합성값이다. 한 Job(N3 3회차)에서 0.66–1.34 s 첫-chunk host 차단이 드물게 관측됐고 원인은 미해결이다." ★범위 정정(2026-10-29 범위 정정, 감사 r1 #11): 위 문안 뒤에 "이 run들은 적체 영역(span > 동거 decode step, prefill 미동기 발행)이었고, 이 영역에서 무반응은 루프 구조상 예상되는 값이다. 박자 영역에서는 측정되지 않았다."를 붙여 쓴다. 위 "구조 확인"(§1-24)은 KISTI 기판 서술이며 이 진단과 기전 동일성은 미확인. 금지 문장 목록은 판정서 P1 링크.
- **stock_quick·Phase B 정지 관측(stock_quick 결과 감사 r1 A·B PLAUSIBLE 조건부 · Phase B 결과 감사 r1 GO-with-edits A1–A3·A5·A6, 2026-11-07; 봉인 밖/n=1–2·다른 Job/GPU·기술값; 판정서 `workspace/engine-port/results/stock_baseline/AUDIT_result_phaseB_claims_r1_2026-11-07.md`)**: (A1) "VESSL A100·Nano·W1 ShareGPT에서, piecewise CUDA graph를 끈 SGLang v0.5.10 기본 구성(이미지 py3.14 제약; upstream 기본은 ON)의 토큰 ITL은 이봉이었다 — 기본 step 약 21 ms(5.5 req/s)와 요청 도착 1건당 약 1회 생기는 간극 약 64 ms. 이 관찰은 서로 다른 세 Job·GPU에서 재현됐다(기술값, rung당 n=2)." (A2) "짧은 프롬프트(≤512토큰)에서 간극 길이는 '기본 step + 약 42 ms'로, 프롬프트 길이에 거의 무관한 상수항이 지배했다(1 run에서 유도 — ≤1024토큰 상수항 약 42 ms, 1024토큰 초과에서는 길이 의존 정지가 추가). 상수항의 원인(eager prefill 발행·동기화 대기·기타 host 작업)은 측정되지 않았다." (A3) "요청 내부 ITL p95 술어에서는 요청당 간극 비율이 5%를 넘으면 요청 ITL p95가 간극 길이와 같아지므로, 간극 길이 약 64 ms가 60 ms SLO 바로 위에 있다는 사실이 SG 계열 attainment를 결정한다(ITL 축 metric cliff)." (A5) "진단 boot(n=1, 5.516 req/s, seed 201)에서 PD-mux D44 구성의 argv·env·트리를 유지한 채 `--enable-pdmux`와 PD-mux config만 빼자 요청 ITL p95 중앙이 약 69 ms로 돌아오고 attainment가 0.19였다(첫 번째 Job의 PD-mux ON 45 ms·1.00은 다른 GPU). 같은 진단에서 overlap OFF·chunked prefill 끔·mem 0.6+mamba 48은 SLO 위 간극을 없애지 못했다(chunk·mem 두 변형은 이 trace·부하에서 처치가 발화하지 않는 조건)." (A6) "PD-mux 구성에서도 도착당 간극은 남아 있으며(약 45 ms), PD-mux가 바꾼 것은 간극의 유무가 아니라 길이(SLO 60 ms의 위→아래)다. `--enable-pdmux`는 이벤트 루프·SM 분할·split-prefill·stream별 graph를 함께 바꾸므로, 이 차이를 '동시 실행' 하나로 귀속하지 않는다." 범위: 이는 §1-24·위 lockstep 진단("별도 정지 항 미관측")과 **다른 모양의 정지**(짧은 프롬프트에도 도착마다 생기는 상수항, piecewise OFF 구성) 관측이며 그 미관측 서술을 덮어쓰지 않고 KISTI 8월 PD-mux 우위의 기전 귀속에도 쓰지 않는다. 위 stock_quick 수치(W1 5.5–9.6 req/s에서 ITL p95 약 63–66 ms, 60 ms 초과 토큰 9.2–11.4%)도 piecewise OFF 구성 한정으로 읽는다(A7). 인용 금지는 판정서 링크.
- **T3 비용 항 측정(Nano, `cap_char/RESULT_costterms_2026-10-26.md` rev1 §17 · 감사 `AUDIT_costterms_result_r1_2026-10-27.md`)**: "동거 창 SW power cap 노출 96–100% 대 단독 창 0–4%이며 ρ(1.13–1.19)는 총효과로만 읽는다." 합성 성분 = 동거 + 400 W 한도 클럭 하락 + 부하 의존 관측자 비용(TEL ON, 미분리). 간섭 크기·D 순서는 **NOT-YET-SUPPORTED**("ρ D24 > D44"는 rep 2·차 < 3%·D24 동거 클럭이 더 낮아 분리 불가 — 금지).

### CM-5 메모리·admission 예산
- **형태 확정(코드 사실)**: `disable_radix_cache ∧ max_running_requests`이면 `max_mamba_cache_size = max_running_requests`, 즉 cap≡mamba 슬롯 (§1-23). 이는 SGLang 정적 풀(부팅 시 할당, 슬롯 고갈 = assert)의 구현 결과로 할당기에 종속이며(통합 풀이면 소멸) **실험 설정으로 공시**한다. 순수 Transformer에선 admission만.
- **확정**: hybrid KV가 오히려 큼(8.6×), 구속 자원은 mamba 슬롯(2026-09-28 감사). slot당 state 비용은 모델마다 다름(§1-23).
- **E′ 문안(사용자 결정 8′, 2026-10-19 — 결정이며 판정 아님·측정 0)**: "hybrid의 요청당 메모리는 길이 무관 고정 state s를 가져, 짧은 ctx(< s/k, Nano-9B-v2 약 8.8k [유도: 실측 k·s에서, GB 단위; GiB로 9.4k])에서는 s가, 긴 ctx에서는 (작은) KV가 지배한다. 이 비용 구조와 goodput 영향을 양 구간에서 특성화한다. SGLang 정적 풀에서는 cap이 admission·decode bs·state 예약을 함께 정한다(실험 설정 공시)."
- **실측**: 부팅 로그 두 줄(128k 프로브·비용 항 Job)에서 Nano KV 16,382 B/토큰, state 143.5 MB/요청(±0.05%, KV 풀 합 2% 안, **CONFIRMED**) → s/k ≈ 8.8k(GB, 유도). 경계 자체의 실측이 아니다. **T4 산식 B 재현은 NOT-YET-SUPPORTED**: 판정 가능 10셀 중 8셀은 cap 구속(n_A = n_B = cap)이라 원식과 산식 B를 가르지 못하고, cap 비구속 점은 2개(fused cap 48 L 65,536 n_real 21 = n_B 21 HELD / PD-mux cap C_A 114 L 8,800 n_real 92, n_B 78·n_A 114 사이 MISS). 사전 봉인 ratio_A 예측 0.68 ± 0.05는 실측 0.81로 이탈했다. "산식 B 9/10" 인용 금지. cap 48은 2026-07-04 `40d6869`에서 근거 기록 없이 처음 쓰여 통제 상수로 승계됐고(cap 128·114 boot도 있음), mem 0.6에서 cap 256의 SSM·conv state 예약 36.7 GB는 측정 풀 33.1 GB를 넘어 boot 불가 [유도](감사 r1 2026-11-08, `…/AUDIT_cap_mem_memo_and_formula_review_r1_2026-11-08.md`; "평균 1.3k면 mem 0.75" 인용 금지, mem 0.75의 cap 256 수용은 문맥 분포 의존이라 부팅 측정 필요). `cap_char/RESULT_costterms_2026-10-26.md` rev1 §17 · 감사 `AUDIT_costterms_result_r1_2026-10-27.md`
- 금지: "~52k가 hybrid 경계"·"9k 경계 실측"(유도값). I-5 거절 사유 KV / cap(≡슬롯) 2분류 — `src/multiplex/admission_probe.py`; gate G-α 28/28 PASS(정확성). 설계 v1 `cap_char/DESIGN_cap_longctx_v1_2026-10-19.md`. 낡은 문장: `COSTMODEL_running_cap_2026-10-15.md` §3-2 "generic 예측기가 C_mem 과대평가"는 결정 8′로 폐기.

### CM-6 대기열·TTFT
- **구조 확인(방향, n≥4 변화 trace)**: entanglement 사슬 decode 굶김 → ITL↑ → batch 정체 → admission 차단 → TTFT 폭발(§1-4). TTFT 차이의 ≥95%가 공유 대기열(산술 CONFIRMED).
- **확정**: 비대칭 — decode 과소공급은 고부하에서 파국, 과다공급은 (조건 명시 시) 저부하에서 작은 손실(§1-6). ⇒ 최적 split = max(모델 floor, 부하항 ∝ λ×출력길이)(§1-5, 형태만).
- 분해 규약 확정: `TTFT = W_queue + S_prefill + 잔차`, 정상성 검사 3종(`system_design_anchor` A9).
- **J1(N=1) B shape 허용 문안(감사 2026-10-19)**: "J1(N=1)의 B shape에서, fused 계열은 prefill step 동안 decode가 멈추는 꼬리(ITL 상위 꼬리 ≈ 입력 하나의 prefill 시간)를 보이고 PD-mux arm은 보이지 않는다. 다만 achieved가 같은 rung에서 요청 ITL p95 SLO 통과율은 fused_sym과 PD-mux가 같으며, 이 꼬리는 미조율 fused(chunk 8192, mixed-chunk off)의 노브 효과일 수 있다. 처리율과 SLO의 역순은 미확립(G0 몫)이다." (원문 "처리율↔ITL 역순"은 REFUTED, `AUDIT_J1_N1_analysis_2026-10-19.md`)
- **용량 수치(CAP1-R·CAP2 결과 감사 r1, 기술값)**: λ*_W1=7.29491 req/s(D44·D64)는 120요청 버스트 셀 용량이며 지속 용량이 아니다(600 s에서 과부하 포화 완료율 5.06–5.34 req/s, goodput 아님 — ★정오 2026-11-02: 이전 "≈3.0"은 단위 오류). λ*_W3=0.3375(D16)는 통과 여유 1요청으로 취약하다. Job마다 GPU·throttle이 달라 Job 간 수치를 합치지 않는다.
- **stock_quick(결과 감사 r1; 구성 묶음 대 기본값 비교이며 PD-mux 단독 효과도 static 최적도 아님)**: SG 기본은 W1 5.516 req/s에서 ITL 문턱 60→0.21·64→0.54·66→0.93·68→0.99의 이중 metric cliff 위에 있고(80 ms는 사후), W3 등록 술어에서는 모든 static 이상이나 격차는 0.506·0.759 req/s에서만 잡음과 구분된다.
- **Phase B 규칙 출력(결과 감사 r1 A8·A9, 2026-11-07; 판정서 `workspace/engine-port/results/stock_baseline/AUDIT_result_phaseB_claims_r1_2026-11-07.md`)**: (A8) "같은 Job(GPU bc0be109)의 120요청 버스트 사다리에서 봉인 규칙 출력은 vLLM 0.31.0 8.389146, PD-mux D44 구성 묶음 7.29491 req/s였다(rung당 seed 2, 격자 간격 1.15배). PD-mux 값이 이전 VESSL 캠페인(CAP1-R)과 같은 것은 격자 승계 때문이며 기판 동일성 근거가 아니다." (A9) "600 s 안정성 규칙(E2E ≤ 60 s 완료 ≥ 0.95 ∧ TTFT 중앙 증가 ≤ 2배; SLO goodput 용량 아님)에서 두 arm 모두 격자 {1.0, 0.8, 0.5}×버스트 중 0.5× 점만 통과했다(seed 1·boot 1). 실제 경계는 PD-mux (3.65, 5.84), vLLM (4.19, 6.71) req/s 구간 안이며 격자로 정해지지 않는다. 0.8× 과부하 점의 포화 완료 처리율은 PD-mux 묶음 약 5.15, vLLM 약 6.5 req/s였다(n=1, throttle 15–24%, 기술값)." 용량 순서·크기·비는 인용하지 않는다.

### CM-7 prefix cache — 가설 (radix ON 확장 때만)

### (E) 결정 비용
- **계수 확정**: 컨트롤러 CPU 오버헤드 mean 32–36 µs/call — decode step 대비 무시 가능(§1-12). ⇒ 동적 제어 실패는 비용이 아니라 positioning(§1-17, HE0). ★2026-10-29 범위 정정, 감사 r1 #6: 무시 가능한 것은 컨트롤러 CPU·agnostic 경로 전환 비용이고, 컨트롤러 전환의 진행 중 drain은 미측정(bind+GATE 상한 ≤0.72%, slo arm 배제 불가).

### (확장) 조성 · TP2 · N-GPU
- **조성 스윕(J-H1 한 Job, dummy 가중치 a4·a13·a22·a31, `cap_char/RESULT_costterms_2026-10-26.md` rev1 §17 · 감사 `AUDIT_costterms_result_r1_2026-10-27.md`)**: "P1·P2·P4 HELD/MONOTONE(CONFIRMED, J-H1 한 Job 범위)", P3(ctx 기울기 단조)는 조건부(모형 없는 ctx 차분은 ±25% 안). J-H2 조성별 SM 민감도 단조는 promote 금지. 조성 효과는 층당 파라미터·가중치 바이트 교락 공시.
- **TP2(J-C, Nano)**: F1·F2·F5 HELD(**CONFIRMED**, Nano 한 Job 범위); F4는 cap 128 boot `F4_STRIDE_DERIVED`·cap 48 장문 boot `_MISS`, F3은 `FILL_INCOMPLETE` — F3·F4는 **NOT-YET-SUPPORTED**. J-C와 J-A bs 96 해석을 엮지 않는다.
- **N-GPU**: N>1 성능 수치 0; 정확성 게이트(Nano M2g, TP4·TP8은 `TP_NUMERIC_DIVERGENT` 병기 의무)만 통과. M3 J2·J4·J8도 PD-mux arm만 telemetry ON(rev12 자료를 메인에 섞지 않음, 결정 22). 근거 `reports/hymux_review/MULTIGPU_EVIDENCE_REVIEW_2026-10-26.md` §2·§3. DP 라우팅(결정 35–37)은 설계 단계.

## cost model이 이미 답한 것 (논문 서술로 이어지는 문장)

1. 단일 GPU PD-mux의 agnostic 경로 전환 연산과 결정 계산(CM-3·E)은 무시 가능하다 → 정책의 성패는 **예측 정확도(CM-1·CM-6)**와 실행 루프 구조(발행·입장·전환 drain)에 달려 있고, 후자는 공개 루프 하나에서만 측정됐다. 진행 중 전환 drain(조작 비용)은 미측정(2026-10-29 범위 정정, 감사 r1 #13).
2. hybrid의 층 종류 차이는 연산자 비용에는 보이지만(CM-1·CM-2, 격리 커널) 층 경계 SM 재분할 레버로는 쓸 수 없고(이 구현·CM-2 실행시 비용; 층 종류별 비용 차 Diff A를 span 시간 예산에 쓰는 레버는 미시험, CM-2 열린 질문; 2026-10-29 범위 정정, 감사 r1 #14),
   그 차이가 SM 스케일링 곡선의 차이(=예측기를 틀리게 만드는 항)로 이어진다는 증거는 아직 없다.
3. hybrid의 요청당 메모리는 길이 무관 고정 state s를 가져 짧은 ctx에서는 s가, 긴 ctx에서는 (작은) KV가 지배한다(CM-5, 구간 경계는 유도값). SGLang 정적 풀에서는 cap≡슬롯이 admission·decode bs·state 예약을 함께 정한다 — 할당기 종속이라 "선행 모델이 놓치는 변수"로는 쓰지 않는다(허수아비 점검, 결정 8′).
4. ITL 꼬리는 동거 간섭(CM-4)이 지배하며 decode-SM 레버와 반대 부호다(KISTI 정지 기전 미확정, VESSL·Nano lockstep 진단 범위의 별도 정지 항 미관측 — CM-4, 단 piecewise OFF SGLang 기본 구성의 다른 모양 정지는 관측·원인 미측정; 2026-10-29 범위 정정, 감사 r1 #10). 분할 후에도 동거 감속은 남고(Job-EA), 그 일부는 전력
   한도를 거친다 — 단일 GPU 비용 모형에 **전력 항**이 필요할 수 있다(크기 미분리).

## 아직 답하지 못한 것 (= 다음 실험이 채울 칸)

- **계측 관측자 비용(측정 방법 항, 2026-10-21)**: PD-mux 쪽 telemetry 묶음(경로+PHASE_EVENTS, 몫 미분리)이 λ를 −1.7%, I-4가 ITL p95를 +1.0% 움직였다(분해 Job, shape A·agnostic·n=4 한정; 보정 계수 아님). 정책: 판정 측정은 전 arm 계측 OFF + Job 내 ON/OFF 짝 공시(결정 22). 영향 범위는 M3 J1·J2·J4·J8(rev12도 PD-mux arm만 계측 ON, 2026-10-27 정정)이며 P1-opint·HE0 불변(CONSENSUS §8).
- 비용 항 T1–T4·조성·TP2는 6 Job 전부 회수·감사 r1 반영(2026-10-27, `cap_char/RESULT_costterms_2026-10-26.md` rev1 §17 · 감사 `AUDIT_costterms_result_r1_2026-10-27.md`). 아직 비어 있는 것: **T2·T2′·T4의 전력 상태**(`CLOCK_SENSITIVE`)와 **T4 MISS 원인**은 클러스터 볼륨 원자료가 있어야 답할 수 있고 **산식 B 재현·TP2 F3·F4는 NOT-YET-SUPPORTED**; T1은 J-A에서 전력 상태 확인(없음). 검증(V)은 별도 PREREG(결정 38로 설계 동결, 측정 우선).
- `s_attn(SM)` ≠ `s_ssm(SM)`인가 — 유효 측정 0건(Claim E 판정서 2026-10-19). 이 칸이 비어 있어도 Claim E는 반증되지 않으며, a는 E의 필요조건도 아니다(우회: 모델 hold-out 조성 귀속·memory/admission). 판정에는 hold-out 서빙이 필요하다(고비용).
- `δ(p,d)`의 **전력 무관 크기**와 prefill 종류 의존 (N2 후속 F-1…F-4, 사용자 결정 대기). 동적 arm 전환 drain 대기(`:1344`,`:1361`, 미측정)와 KISTI 정지 기전(phase 계측 0, 모델·backend 교락, 게이트 #83)은 결정 40 (가) 진단 Job이 채울 칸이다. VESSL 진단의 "별도 정지 항 미관측"은 KISTI로 옮겨 쓸 수 없다. J1 GPU0 SW power cap 노출 60.6%라 multi-GPU 쪽도 총효과로만 읽힌다.
- 비교 기준선·측정 순서(결정 2026-11-09/10, 판정 아님): SG chunk·SG mixed·vLLM 모두 기준선, 주 비교 SG mixed(부하점별 최선 대비, 단독 이득 표기 불가). 1단계 앞 동시 실행 상한·decode 메모리 트래픽 측정(cap·mem 고정값을 정함), 층별 decode nsys는 1단계 첫 걸음, Qwen3.5-9B는 motivation 뒤. 아직 측정 0. `PROJECT_STATUS.md` 끝 10차.
- 기본 서빙 구조 기준선(CM-6 비교 기준): Phase B 결과 감사 r1 반영(2026-11-07, A1–A11). 비어 있는 것 — 상수항·간극의 **원인**(host 발행 대 동기화 대기), **upstream 기본(piecewise ON)** 측정, 두 arm 자기 용량 근처의 용량 직접 측정(Q4), B5 같은-Job 재현, r_ser 프로브 Nano 값(세 번째 Job 미실행). D\* 식 수정(E-6·E-7 MISS 기반)과 박자 영역(r≈1) 측정은 새 PREREG·사용자 결정 몫(PC 축약 부하 창 결함 수정 선행).
- 워크로드 축: shape C(긴 입력 + 긴 출력)는 2026-10-19 사용자 결정으로 G0 두 번째 표준 셀로 편입됐으나(구체값은 G0 PREREG) **아직 측정 0**. hybrid ctx 항이 의미를 갖는지는 열린 질문이다.
- hybrid의 고정 state s 대 KV의 두 구간(짧은 ctx에서 s, 긴 ctx에서 KV)과 cap의 goodput 영향 — s·k 부팅 로그 사실만 확정, T4 cap 비구속 점은 2개뿐(산식 B 미입증)이고 goodput 영향은 측정 0.
- ρ의 D·b·c 무관성 (F-4) — 해제는 F-4 격자 등록 동등성 라벨 새 캠페인의 결과 감사로만(C-2). N2·N3·E2는 2026-10-15 종료, 게이트 #6 미닫힘. T3 비용 항(ρ 1.13–1.19)도 총효과이며 D 순서는 NOT-YET-SUPPORTED.
- N-GPU·TP(multi-GPU)와 층 배치(AMD)가 새로 만드는 항(통신·핸드오프). M3는 J2·J4·J8 외부 종료로 G_sym이 전 N에서 미정의이며 G0 뒤로 연기됨(2026-10-19 결정); 하네스 시간 모형이 확장 셀을 빠뜨려 rev12 예산(190 GPU-h)이 과소했다(`results/multigpu_verdicts/M3_SCOPE_OPTIONS_2026-10-19.md`).

## 해석적 수식 기반 (고정, 결정 27 — 2026-10-22; 기존 줄 번호 보존을 위해 파일 끝에 둔다)

정본 수식 문서 `reports/cost_model/ANALYTIC_COST_FORMULAS.md`(analytic-v1)의 요약. 수치는 모두 [유도]이며 변경은 그 문서의 변경 규칙을 따른다.
- **상주 메모리**: s = n_M·(n_h·p·N·β_s + d_conv·(K−1)·β), k = n_A·2·n_kv·d_h·β, L\* = s/k. Nano: 143.5 MB · 16,384 B/토큰 · ≈ 8.8k (배너 실측과 일치) → **CM-5**.
- **prefill**: F_pre(L) = α·L + γ·L² (+ 2dV), γ = n_A·2·n_q·d_h. Nano: α = 15.65 GFLOP/토큰, γ = 40,960, 이차항 = 선형항인 L ≈ 382k(64k에서 17%) → **CM-2**. J1 floor 기울기 0.111 ms/토큰은 실효 약 141 TFLOP/s(피크의 약 45%)에 해당.
- **decode**: 메모리 바운드 하한 T_dec ≥ a₀ + b₀·b + c₀·b·c, a₀ = 가중치 바이트/BW(Nano 8.14 ms), b₀ = 2s/BW(0.141 ms/요청, SSM state 읽기·쓰기), c₀ = k/BW(8.0 ns/요청·토큰). state 항 = KV 항인 문맥 c\* = 2s/k ≈ 17.5k → **CM-1**의 함수형 `a + b·bs + c·bs·ctx`의 물리적 하한.
- **조성 스케일링**: k와 c₀는 n_A에 정확히 비례, s는 n_M에 비례, prefill 비선형성은 n_A에 따라 단조 증가(조성 변형 표는 정본 수식 문서 §7) → J-H 예측 P3·P4의 근거.
- **선행연구 대조(감사 r1 2026-11-08)**: decode state 갱신 FLOPs·2s·c\*=2s/k(b 무관)는 ReplaySSM 2610.13778v1 부록 C.1–C.2와 같은 형태(우리 식은 conv state 포함, 우리가 처음 도출했다고 쓰지 않음); F_A,pre의 계수 2는 FA2 causal 관례로 Marconi·Chinchilla 밀집 관례와 2배 차; §6·§7 대입값은 손계산 일치.
- **확장 틀**: 측정이 덧붙이는 항 = 층 종류별 효율 η_X(SM), PD-mux SM 분할, lockstep·동기화, 동거 간섭 ρ, cudagraph 패딩, SGLang 정적 두 풀 할당, TP 분할(정본 수식 문서 §8).
