# TRACK: main — anchor 사다리 / N2(3-a) / 장문 워크로드 재정의   (worktree `prefill-layer-alloc`, branch `main`)

> 이 트랙의 현재 상태 유일본(60줄). 이력은 git, 세부는 링크. 정본 반영은 promote.

## 목표
anchor(`reports/system_design_anchor_2026-09-30.md`) 주장 1 — generic 예측기가 hybrid에서 다른 결정을 내는가 — 의 사다리.
단일 GPU(lockstep 기판)에서 측정 명세·기판 특성화를 세우고, 그 구조를 multi-GPU PD-mux로 이양한다.
## cost model 기여
- 채우는 칸: CM-1(`s_attn` vs `s_ssm`), CM-4(`δ(p,d)` + lockstep span 정지 분리), CM-5(장문 KV vs mamba 슬롯, cap 결합)
- 이번에 알게 된 것: 측정 0. 유도값으로 cap 실현 창 `λ·E[out]·E[ITL] ≤ C ≤ min(C_mem, C_SLO)`을 정리했다. Nano-9B-v2에서 cap 48은 ctx≈52k 이하에서 KV보다 먼저 문다. 근거: `results/handoff_review_2026-10-15/COSTMODEL_running_cap_2026-10-15.md`
- promote 필요: 아니오 (2026-10-15 promote 완료)
## 현재 상태
- **사용자 결정(2026-10-15)**:
  - (F1) 옛 환경 전제(SLURM·KISTI)는 무시하고 실행은 VESSL.
  - (F2) 주 모델 Nano-9B-v2.
  - (F3) **true dual worker = AMD 트랙 전용**.
  - (F4) **main의 lockstep PD-mux 구조가 그대로 multi-GPU PD-mux로 이양**. 단일 GPU 결과는 "lockstep 기판" 범위이고, lockstep 정지는 특성화 대상이다.
- 인계 프롬프트(`../handoff/CLAUDE_CODE_PROMPT.md`) 검토 완료. 전부 `workspace/engine-port/results/handoff_review_2026-10-15/`에 있다.
  - "PD-mux ~3× vs fused"는 정본이 철회한 문장이다. 허용 문장은 Zamba2 +40.5%/+185.8%, Granite +27.0%이고, 미조율 fused 대비다.
  - goodput은 A′(2026-10-04)로 이미 고정돼 있다. 프롬프트의 "TTFT 4–15 s·mean TPOT"은 환산 좌표로만 쓴다.
  - arXiv-Sum 16k를 Nano 토크나이저로 실측했다: 유지 91.7%, 평균 7.1k, 출력 203, 30%가 KV=state 경계(8.76k) 초과.
  - A100 roofline 재계산: H100 대비 0.6배, 순위 불변.
  - G1 1차: Olmo-Hybrid 미등록, Qwen3.5 등록.
- 구조 파악:
  - upstream PD-mux는 MuxWise의 단순화 판이다. span = 정적 65,536 token-layer 예산이고, decode와 매 반복 lockstep으로 돈다.
  - prefill 창 안 ITL ≈ span 시간이다. §1-24 "고칠 수 없다"는 lockstep 구현의 성질일 가능성이 있다(감사 전).
  - cap 48은 2026-07-04 `40d6869`에서 근거 없이 상속된 값이다.
- 논의 안건 24개(Q1–Q24)와 각 안건의 출발 문서: `results/handoff_review_2026-10-15/DISCUSSION_INDEX_2026-10-15.md`(§4 = Q1 준비 자료, §5 = 안건별 읽을 곳).
## 다음 행동
1. **Q1 N2 종료 여부 결정** — DISCUSSION_INDEX §4를 읽고, 확인 3건(PROFILE 감사 범위, digest 해제 조건, N3·E2 동시 종료)을 거친 뒤 사용자 결정
2. Q2–Q3(포지션·워크로드) → 2군 공통 측정 명세(Q4–Q12) 초안 → claims-auditor 규칙층 감사 → promote
3. 그 뒤 M3 PREREG 개정(Q23)·G0 PREREG(Q24)
## 사용자 결정 대기
- **Q1 N2 종료** — merge·엔진 트리 통일의 선행 조건 / (a) 종료 + F-2·F-1은 명세 계측, F-3은 5-a로 흡수 · (b) F-1·F-2 후 종료 · (c) 전부. 권고 (a)
- **Q2 포지션 · Q3 워크로드 세트** — 측정 명세의 워크로드 축을 정함 / P-A·P-B·P-C, arXiv-Sum 16k 추가 여부. 권고 P-C + arXiv-Sum을 G0·G2 표준 셀로
- **Q4–Q24** — DISCUSSION_INDEX 표의 권고 열 참조(공통 명세, lockstep 특성화, cap, 비용 모델, 통합 순서)
- 기존 유지: radix(OFF+편차 공시) · 128k 프로브(Q6 floor 측정과 결합 가능) · 장문 hold-out 모델 · CONSENSUS §1 행 38
## promote 때 반영 (main 전용)
완료(2026-10-15): CLAUDE.md fused 문장·메모리 `engine-port-p0-triage.md` 정정, F3·F4를 MEMORY·PROJECT_STATUS·amd·multigpu 트랙에 기록
## 실행 중·미회수 Job
- 없음 (Workspace `pdmux-probe`·`pdmux-build` standby)
## 금지 서술 (링크만)
`workspace/engine-port/results/n2_static_grid/VERDICT_result_n2_ea_7e09cc3_2026-10-13.md` 금지 32–42·N2C-102…108 · 문헌 조사 "hybrid×PD-mux×실 장문 trace 연구 없음"은 조사 범위 한정(교훈 260) · "MuxWise 대비"(현 arm은 SGLang PD-mux 단순화 판) · lockstep 기전·span 크기 수치는 유도값(측정 0)
## 최근 로그
- 2026-10-15: 인계 프롬프트 순차 검토(provenance·arXiv-Sum 실측·A100 roofline·G1 1차) · 방법론 안건·n 층위·goodput 정의·런타임 요인 지도 · span/lockstep·MuxWise/Bullet 대조 · cap 비용 모델 · 결정 F3·F4 · DISCUSSION_INDEX Q1–Q24
- 2026-10-14: 주 모델 Nano-9B-v2 결정; 장문 hybrid 모델·워크로드·선행 평가 설정 조사(아티팩트 2건)
- 2026-10-13: Job-EA 회수·재집계·결과 감사·정본 rev95; 하네스 재구성(트랙 모델·COSTMODEL 신설)
- 2026-10-12: Job-PROFILE 완주·경량 확인, rev18 봉인
