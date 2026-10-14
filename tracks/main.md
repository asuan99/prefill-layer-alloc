# TRACK: main — anchor 사다리 / N2(3-a)   (worktree `prefill-layer-alloc`, branch `main`)

> 이 트랙의 현재 상태 유일본(60줄). 이력은 git, 세부는 링크. 정본 반영은 promote.

## 목표
anchor(`reports/system_design_anchor_2026-09-30.md`) 주장 1 — generic 예측기가 hybrid에서 다른 결정을 내는가 — 의 사다리.
현재 단계 N2(3-a): decode 프로파일(SM×bs×ctx)과 동거 간섭 측정.
## cost model 기여
- 채우는 칸: CM-1(`s_attn` vs `s_ssm`), CM-4(`δ(p,d)`)
- 이번에 알게 된 것: CM-4 동거 감속 존재 확인, 크기는 전력 한도와 교락(Job-EA 결과 감사) — COSTMODEL 반영 완료(2026-10-13)
- promote 필요: 아니오
## 현재 상태
- **주 모델 = Nemotron-Nano-9B-v2-Base**(2026-10-14 사용자 결정; 131,072 ctx, attn 4/Mamba 27/MLP 25, KV 16 KiB/token, state ≈0.14 GB/요청 fp32). Nemotron-H-8B-Base-8K는 8K 한계라 기존 결과 연결 셀로만.
- N2 Job-PROFILE(`a5de2d6`) 경량 확인만 · Job-EA(`7e09cc3`) 결과 감사 死因 0, 정본 rev95 반영. N2 미완결.
- 연구 감사 의견(미감사, 2026-10-12): 증거 사슬이 첫 화살표(layer-type 스케일링 차이)에서 끊김 → 사다리 방향 재검토 필요.
## 다음 행동
1. 아래 사용자 결정 → 결정에 따라 F-x 사전등록 또는 shape C 5-a 설계
2. Job-PROFILE 동거 4추정량 산출 전 claims-auditor 확인(블라인드 노출 N2C-87·98)
## 사용자 결정 대기
- **사다리 방향** — Job-EA로 "분할해도 동거 감속이 남는다"가 확인됐고 연구 감사는 hybrid 고유 항 미입증을 지적 / 계속(S0′→S1→5-b) · 정리 · 워크로드 재정의 중 택1
- **N2 후속 측정 F-1…F-4** — 간섭 크기를 전력과 분리하려면 필요 / F-1 전력 정합 대조 arm · F-2 메모리·SM 클럭 수집 · F-3 운영 trace · F-4 확장 격자 (처방 아님, 결과 감사 §10)
- **shape C 구체값** — 8k/1k는 Nano-9B-v2의 KV=state 경계(≈8.6k)에 걸쳐 hybrid 설계점을 시험하지 못함(2026-10-14 계산) / C-reason(32k–64k 입력·2k–8k 출력, radix OFF) 우선 권고 · C-agent(64k–256k, radix ON 재결정 필요); 128k 단일 요청 프로브(TTFT 하한·decode stall) 첫 Job 포함 여부; `max_prefill_tokens` 상수 vs 축
- **장문 hold-out 모델 세트(보조)** — 주 모델 결정 뒤 / Nano-12B-v2(크기) · Granite-4.0-H-Micro(구성, NoPE) · Falcon-H1-7B(병렬 hybrid 대조); Nemotron-3-Nano-30B-A3B는 TP2(multigpu)
- **CONSENSUS §1 행 38 유지 여부** — doc-steward 판단으로 신설됨 / 유지 · 되돌림
## 실행 중·미회수 Job
- 없음 (Workspace `pdmux-probe`·`pdmux-build` standby)
## 금지 서술 (링크만)
`workspace/engine-port/results/n2_static_grid/VERDICT_result_n2_ea_7e09cc3_2026-10-13.md` 금지 32–42·N2C-102…108
## 최근 로그
- 2026-10-14: 주 모델 Nano-9B-v2 결정; 장문 워크로드·hybrid 모델 탐색(shape C를 C-reason/C-agent로 분리 제안)
- 2026-10-13: Job-EA 회수·재집계·결과 감사·정본 rev95; 하네스 재구성(트랙 모델·COSTMODEL 신설)
- 2026-10-12: Job-PROFILE 완주·경량 확인, rev18 봉인
