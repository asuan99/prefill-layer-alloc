# COSTMODEL — 지금까지 드러난 비용 구조 (진척판, 2026-10-13 신설)

> **지위**: 측정 결과를 새로 만들지 않는다. 이미 정본(`reports/CONSENSUS.md` §1, `reports/cost_model_design_space_2026-10-01.md`,
> 감사 판정서)에 있는 사실을 **비용 항(CM-1…CM-7) 축으로 다시 배치**한 장부다. 연구 축은 `reports/system_design_anchor_2026-09-30.md`
> (스케줄러 확장)이며, 이 장부는 그 과정에서 **cost model이 얼마나 성립했는지**를 보여 주는 진척판이다.
> 규칙: 현재 형태만 적는다(이력은 git). 한 항 최대 6줄. 새 근거는 결과 분석의 "cost model 기여" 절에서만 들어온다.
> 상태 어휘: `가설` · `데이터 있음(미감사)` · `형태 확인` · `계수 확정(범위 한정)` · `서빙 검증` · `반증`

## 축 정의 (고정)

`비용 = f(E 엔진/시스템 부하, M 모델 구조, U 워크로드)` — **배치 ctx 분포 = U × E**. M은 배포 시 상수라 단일 모델 안에서는 변수가 아니다
(조성 귀속은 모델 hold-out에서만). 상세 `cost_model_design_space_2026-10-01.md` §1.0.

## 진척 요약 (한눈에)

| 항 | 무엇 | 상태 | 다음 데이터 |
|---|---|---|---|
| CM-1 | decode step 시간 `T_dec(SM, bs, ctx)` | 형태 일부 확인 / **hybrid 고유 항 미입증** | N2 PROFILE(경량 확인만, 수치 인용 금지) · opchar(연산자 단위, 특성화) |
| CM-2 | prefill span 시간 `T_span(L, SM)` | 형태 확인(마이크로벤치·격리 커널) / 서빙 계수 미확정 | A1 span-SM, opchar 후속 |
| CM-3 | 전환·경계 비용 | **계수 확정(범위 한정)** — 무시 가능 | 컨트롤러 구동 경로만 미측정 |
| CM-4 | 동거 간섭 `δ(p,d)` | **존재 확인(N2 Job-EA)** / 크기는 전력 한도와 교락 | N2 후속 F-1…F-4(사용자 결정) |
| CM-5 | 메모리·admission 예산 | **형태 확정(코드 사실)** | 실측 점유 병기 |
| CM-6 | 대기열·TTFT | 구조 확인(entanglement 사슬) / 분해 규약 확정 | N3 용량 |
| CM-7 | prefix cache (radix ON) | 가설 | 보류(radix OFF 우선, 2026-09-24 결정) |
| (E) | 컨트롤러·결정 비용 | **계수 확정** — 무시 가능 | — |
| (확장) | N-GPU·TP, 층 배치(AMD), 에너지 | 가설 — 정확성 게이트·캡처 경로만 통과, 비용 측정 0 (multi-GPU 트랙 main 이양·merge `dc06c870` 완료 2026-10-15, 비용 측정 0) | 각 트랙 TRACK.md |

## 항별 장부

### CM-1 decode step 시간
- 제안 형태: attn 층 `∝ bs·ctx`, SSM 층 `∝ bs`(ctx 무관), MLP `∝ bs`, 계열별 SM 스케일링 `s_attn(SM)`, `s_ssm(SM)` (`cost_model_design_space` §1.1).
- **확정**: decode는 SM에 둔감하지 않다 — prefill 16 SM 고정, decode ITL SM16→92 **2.36–2.91×, 4 arm 모델-무관(pure Mamba2 포함)** (C2 `CONFIRMED(scoped)`, 레버 존재만, 정책 이득 아님). ⇒ "decode는 memory-bound라 SM 둔감"(X-7) 전제는 이 기판에서 반증.
- **확정(방향)**: decode에서 attention 비중이 ctx에 따라 커지고 SM 민감도도 같이 커진다(§1-5) — 단 절대 knee 값은 triton/no-cudagraph 마이크로벤치라 인용 금지.
- **미입증(가장 중요한 빈칸)**: `s_attn` vs `s_ssm` 곡선 모양이 다른가(= hybrid 고유 항). 2026-10-12 읽기 전용 연구 감사
  (`handoff-report/session_handoff_2026-10-13.md` (D)절, **claims-auditor 미경유 의견**)는 증거 사슬 "layer-type 스케일링 차이 →
  비용 곡면 → generic 예측 오류 → d_min → 행동 → goodput"이 **첫 화살표에서 끊긴다**고 보았고, "generic 예측기로 충분" 가설은
  기각되지 않았다(A4 판정 K2와 같은 방향). N2 PROFILE은 경량 확인만 거쳐 수치 인용 금지.
- 연산자 단위(opchar, 특성화 전용·격리 커널·서빙 함의 없음, `AUDIT_opchar_2026-10-13.md` Safe wording 범위): 격리 연산자 시간 합
  기준 decode에서 SSM state update 비중이 B 32→128에서 36%→64%(ctx 2048); 메모리 바운드 연산자는 격리 시 44 SM에서 108 SM 속도의
  ≥86%; GEMM 시간 계단은 wave 수 변화와 일치(105-CTA split-K decode GEMM의 104→108 점프 = wave quantization).
- 반증된 가정: "hybrid는 KV가 작아 배치 여유가 크다"(REFUTED — 아래 CM-5).

### CM-2 prefill span 시간
- **확정(마이크로벤치, 서빙 아님)**: attn/SSM SM 민감도 비 Diff B는 L≤512에서 열리고 L≥1024에서 ≈1.0(§1-3 WIDE 스윕) — 짧은 L에선 둘 다 SM 미활용.
- **확정(서빙)**: 그 차이를 런타임 per-type 분할로 쓰면 실행시 granularity 비용이 압도(TPOT 42→124 ms, §1-3/§1-15) ⇒ CM-2의 층 종류 차이는 **존재하나 레버가 아니다**.
- 연산자 단위(opchar, 같은 범위 제한): 격리 연산자 시간 합 기준 Mamba 층이 prefill의 ~66–71%(L=1k–32k). "모델 시간의 %"로 인용 금지.
- 미감사·인용 금지: prefill 축 SM 스윕(TTFT~L 기울기비).
- 열린 형태 질문: MuxWise `N_PL` 층 균일 가정(X-4)이 비용 가중 층 수로 바뀌어야 하는가.

### CM-3 전환·경계 비용
- **계수 확정(agnostic 경로·2모델·이 config 한정)**: `s ≤ 0.04 ms/전환`, `d ≤ 0.07 ms/경계` (§1-8/§1-34). 진짜 비용은 전환 자체가 아니라 prefill 생애주기 경계.
- 미측정: 컨트롤러 구동 전환 경로, PAUSE 진입/해제, SSM chunk 경계 state 이월.

### CM-4 동거 간섭
- **구조 확인**: ITL 꼬리는 decode step time이 아니라 **un-chunked prefill이 decode를 멈춘 시간**이 지배하며, 그 크기는 D(decode SM)에 단조 증가(prefill SM = 108−D 감소) — conjunctive goodput의 ITL 항이 decode-SM 레버와 반대 부호 항을 내장(§1-24). PD-mux는 chunked prefill과 비양립(엔진 assert) ⇒ 기판 전제.
- **존재 확인(N2 Job-EA `7e09cc3`, 결과 감사 `VERDICT_result_n2_ea_7e09cc3_2026-10-13.md`, 라벨 8/8 `COLOC_SLOWER_GE3` CONFIRMED(scoped))**:
  분할해도 지속 prefill과 동거한 decode step은 단독보다 길다. 허용 문안 (B) 그대로: "창별 실측 클럭 하락을 전부 빼도 64/64 짝의
  ln ρ는 ln 1.03을 넘었다 … `COLOC_SLOWER_GE3`의 존재는 클럭 성분만으로 설명되지 않는다. 이 문장은 크기 진술이 아니다."
- **크기는 미분리**: ρ는 동거 + 전력 한도 클럭 하락의 합성값(허용 문안 (A)) — "간섭 크기"로 인용 금지(N2C-102), "D·b·c 무관" NOT-YET-SUPPORTED.
  ⇒ CM-4에는 **전력 채널**이 실제로 붙어 있다(C 창 64/64 SW power cap 노출). 분리 경로 = 후속 F-1(전력 정합 대조)·F-3(운영 trace).

### CM-5 메모리·admission 예산
- **형태 확정(코드 사실)**: `disable_radix_cache ∧ max_running_requests`이면 `max_mamba_cache_size = max_running_requests` — 슬롯 상한 ≡ running 상한(§1-23). hybrid에서 cap은 admission과 state pool을 동시에 움직이고 순수 Transformer에선 admission만.
- **확정**: hybrid KV가 오히려 큼(8.6×), 구속 자원은 mamba 슬롯(2026-09-28 감사). slot당 state 비용은 모델마다 다름(§1-23).
- 남은 것: 긴 ctx에서 KV vs 슬롯 중 무엇이 먼저 묶이는지 실측 점유(X-6).

### CM-6 대기열·TTFT
- **구조 확인(방향, n≥4 변화 trace)**: entanglement 사슬 decode 굶김 → ITL↑ → batch 정체 → admission 차단 → TTFT 폭발(§1-4). TTFT 차이의 ≥95%가 공유 대기열(산술 CONFIRMED).
- **확정**: 비대칭 — decode 과소공급은 고부하에서 파국, 과다공급은 (조건 명시 시) 저부하에서 작은 손실(§1-6). ⇒ 최적 split = max(모델 floor, 부하항 ∝ λ×출력길이)(§1-5, 형태만).
- 분해 규약 확정: `TTFT = W_queue + S_prefill + 잔차`, 정상성 검사 3종(`system_design_anchor` A9).

### CM-7 prefix cache — 가설 (radix ON 확장 때만)

### (E) 결정 비용
- **계수 확정**: 컨트롤러 CPU 오버헤드 mean 32–36 µs/call — decode step 대비 무시 가능(§1-12). ⇒ 동적 제어 실패는 비용이 아니라 positioning(§1-17, HE0).

## cost model이 이미 답한 것 (논문 서술로 이어지는 문장)

1. 단일 GPU PD-mux의 결정 비용(CM-3·E)은 무시 가능하다 → 정책의 성패는 **예측 정확도(CM-1·CM-6)**에 달려 있다.
2. hybrid의 층 종류 차이는 연산자 비용에는 보이지만(CM-1·CM-2, 격리 커널) 런타임 분할 레버로는 쓸 수 없고(CM-2 실행시 비용),
   그 차이가 SM 스케일링 곡선의 차이(=예측기를 틀리게 만드는 항)로 이어진다는 증거는 아직 없다.
3. hybrid 고유의 구속 자원은 KV가 아니라 mamba 슬롯이다(CM-5) — 선행 admission 모델(X-6)이 놓치는 변수.
4. ITL 꼬리는 동거 간섭(CM-4)이 지배하며 decode-SM 레버와 반대 부호다. 분할 후에도 동거 감속은 남고(Job-EA), 그 일부는 전력
   한도를 거친다 — 단일 GPU 비용 모형에 **전력 항**이 필요할 수 있다(크기 미분리).

## 아직 답하지 못한 것 (= 다음 실험이 채울 칸)

- `s_attn(SM)` ≠ `s_ssm(SM)`인가 (N2/3-a) → 예측기 P_gen vs P_hyb 발산(S0′)의 전제.
- `δ(p,d)`의 **전력 무관 크기**와 prefill 종류 의존 (N2 후속 F-1…F-4, 사용자 결정 대기).
- 워크로드 축: hybrid ctx 항이 의미를 갖는 영역은 입력·출력이 모두 긴 구조(가칭 shape C)일 가능성 — 연구 감사 의견, 1-GPU·
  multi-GPU 어디서도 등록·측정 안 됨(사용자 결정 대기).
- ρ의 D·b·c 무관성 (F-4) — 해제는 F-4 격자 등록 동등성 라벨 새 캠페인의 결과 감사로만(C-2). N2·N3·E2는 2026-10-15 종료, 게이트 #6 미닫힘.
- 위 항들이 N-GPU·TP(multi-GPU)와 층 배치(AMD)에서 어떤 항을 새로 만드는가(통신·핸드오프 항).
