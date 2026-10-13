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
- N2 Job-PROFILE(`a5de2d6`) 경량 확인만 · Job-EA(`7e09cc3`) 결과 감사 死因 0, 정본 rev95 반영. N2 미완결.
- 연구 감사 의견(미감사, 2026-10-12): 증거 사슬이 첫 화살표(layer-type 스케일링 차이)에서 끊김 → 사다리 방향 재검토 필요.
## 다음 행동
1. 아래 사용자 결정 → 결정에 따라 F-x 사전등록 또는 shape C 5-a 설계
2. Job-PROFILE 동거 4추정량 산출 전 claims-auditor 확인(블라인드 노출 N2C-87·98)
## 사용자 결정 대기
- **사다리 방향** — Job-EA로 "분할해도 동거 감속이 남는다"가 확인됐고 연구 감사는 hybrid 고유 항 미입증을 지적 / 계속(S0′→S1→5-b) · 정리 · 워크로드 재정의 중 택1
- **N2 후속 측정 F-1…F-4** — 간섭 크기를 전력과 분리하려면 필요 / F-1 전력 정합 대조 arm · F-2 메모리·SM 클럭 수집 · F-3 운영 trace · F-4 확장 격자 (처방 아님, 결과 감사 §10)
- **타겟 워크로드(shape C)** — hybrid ctx 항이 의미를 갖는 유일 영역 후보, 어디서도 미측정 / 8192/512 · 8192/1024 · 현행 유지; `max_prefill_tokens` 상수 vs 축
- **CONSENSUS §1 행 38 유지 여부** — doc-steward 판단으로 신설됨 / 유지 · 되돌림
## 실행 중·미회수 Job
- 없음 (Workspace `pdmux-probe`·`pdmux-build` standby)
## 금지 서술 (링크만)
`workspace/engine-port/results/n2_static_grid/VERDICT_result_n2_ea_7e09cc3_2026-10-13.md` 금지 32–42·N2C-102…108
## 최근 로그
- 2026-10-13: Job-EA 회수·재집계·결과 감사·정본 rev95; 하네스 재구성(트랙 모델·COSTMODEL 신설)
- 2026-10-12: Job-PROFILE 완주·경량 확인, rev18 봉인
