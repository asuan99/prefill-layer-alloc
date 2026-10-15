# TRACK: main — anchor 사다리 / N2(3-a) / 장문 워크로드 재정의   (worktree `prefill-layer-alloc`, branch `main`)

> 이 트랙의 현재 상태 유일본(60줄). 이력은 git, 세부는 링크. 정본 반영은 promote.

## 목표
anchor(`reports/system_design_anchor_2026-09-30.md`) 주장 1 — generic 예측기가 hybrid에서 다른 결정을 내는가 — 의 사다리.
다음 단계는 5-a(워크로드별 static 격차)이며, 그 워크로드를 실 trace 기반으로 다시 정하는 중.
## cost model 기여
- 채우는 칸: CM-1(`s_attn` vs `s_ssm`), CM-4(`δ(p,d)`), CM-5(장문에서 KV vs mamba 슬롯)
- 이번에 알게 된 것: 없음(측정 0). Nano-9B-v2의 KV=state 경계 ≈8.6k(config 계산)는 CM-5 해석 기준점 — 측정 전이라 COSTMODEL 미반영
- promote 필요: 아니오
## 현재 상태
- **주 모델 = Nemotron-Nano-9B-v2-Base**(2026-10-14 사용자 결정; 131,072 ctx, attn 4/Mamba 27/MLP 25, KV 16 KiB/token, state ≈0.14 GB/요청 fp32). Nemotron-H-8B-Base-8K는 기존 결과 연결 셀로만.
- 사용자 방향(2026-10-14): 고정 길이 shape가 아니라 **실제 워크로드 trace**에 맞춰 평가.
- 문헌 조사 완료(측정 아님): PD-mux 4편(MuxWise·Bullet·DuetServe·Nexus) 전부 일반+장문 세트로 평가, MuxWise·DuetServe는 Mooncake Conversation 사용. 페이지 https://claude.ai/artifact/2onik2dLZom4XTxgjzjmkd , 스냅샷·스크립트 `workspace/engine-port/results/workload_survey/`
- Mooncake 4 trace 통계 직접 계산(`MOONCAKE_STATS_2026-10-14.txt`, sha256 포함): conversation 입력 p50 6.9k/p99 85k/최대 126k(131k 초과 0%), 출력 평균 343, 블록 재사용 36.6% — Vertumnus Table 1과 일치.
- N2: Job-PROFILE 경량 확인만, Job-EA 결과 감사 死因 0(정본 rev95). 연구 감사(D) 의견: 증거 사슬 첫 화살표 미입증.
## 다음 행동
1. 아래 워크로드 세트 결정 → 5-a 사전등록 초안(Nano-9B-v2, 실 trace 시간축 재생, 입력 길이 구간별 TTFT SLO)
2. Job-PROFILE 동거 4추정량 산출 전 claims-auditor 확인(블라인드 노출 N2C-87·98)
## 사용자 결정 대기
- **워크로드 세트** — 문헌상 일반+장문 세트가 표준 / 권고: Mooncake Conversation(주) + ShareGPT(앵커) + OpenThoughts(출력 장문) + LooGLE(선택, MuxWise 비교용)
- **radix 처리** — conversation 블록 재사용 36.6%를 radix OFF면 재-prefill / 권고: OFF 유지 + 편차 사전등록 명시, ON은 확장 arm
- **128k 단일 요청 프로브** — TTFT 하한·decode stall 기전 측정(결정 규칙 0) / 5-a 첫 Job 포함 권고
- **N2 후속 F-1…F-4** — 간섭 크기를 전력과 분리 / F-1 전력 정합 대조 · F-2 클럭 수집 · F-3 운영 trace · F-4 확장 격자 (결과 감사 §10)
- **장문 hold-out 모델(보조)** — Nano-12B-v2(크기) · Granite-4.0-H-Micro(구성, NoPE) · Falcon-H1-7B(병렬 hybrid, AMD L-B 적용 불가 대조); Nemotron-3-Nano-30B-A3B는 TP2(multigpu)
- **CONSENSUS §1 행 38 유지 여부** — doc-steward 판단으로 신설 / 유지 · 되돌림
## 실행 중·미회수 Job
- 없음 (Workspace `pdmux-probe`·`pdmux-build` standby)
## 금지 서술 (링크만)
`workspace/engine-port/results/n2_static_grid/VERDICT_result_n2_ea_7e09cc3_2026-10-13.md` 금지 32–42·N2C-102…108 · 문헌 조사의 "hybrid×PD-mux×실 장문 trace 연구 없음"은 조사 범위 한정(최초 주장 금지, 교훈 260)
## 최근 로그
- 2026-10-14: 주 모델 Nano-9B-v2 결정; 장문 hybrid 모델·워크로드·선행 평가 설정 조사(아티팩트 2건: 트랙 지도 QJmmhxaBepgeLLMhSxmX2X, 워크로드 지형 2onik2dLZom4XTxgjzjmkd)
- 2026-10-13: Job-EA 회수·재집계·결과 감사·정본 rev95; 하네스 재구성(트랙 모델·COSTMODEL 신설)
- 2026-10-12: Job-PROFILE 완주·경량 확인, rev18 봉인
