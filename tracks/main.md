# TRACK: main — anchor 사다리 / 공통 측정 명세 / G0 + multi-GPU M3(이양)   (worktree `prefill-layer-alloc`, branch `main`)

> 이 트랙의 현재 상태 유일본(60줄). 이력은 git, 세부는 링크. 정본 반영은 promote.

## 목표
anchor 주장 1(generic 예측기가 hybrid에서 다른 결정을 내리는가)의 사다리. 단일 GPU(lockstep 기판)에서 측정 명세·기판 특성화를 세우고 같은 구조를 multi-GPU PD-mux로 확장한다(F4). **2026-10-15부터 multigpu 트랙(M3, N-스케일링 G_sym)도 이 트랙이 맡는다.**
## cost model 기여
- 채우는 칸: CM-1(`s_attn` vs `s_ssm`), CM-4(`δ(p,d)` + lockstep span 정지 분리), CM-5(장문 KV vs mamba 슬롯, cap 결합), (확장) N·TP 통신 항
- 이번에 알게 된 것: 측정 0. 규칙층(명세 v0.2, 감사 2회)과 캠페인 종료·merge만.
- promote 필요: 아니오 (2026-10-15 2차 promote 완료)
## 현재 상태 (2026-10-15)
- 결정: F1–F4 · Q1 (a) N2·N3·E2 새 Job 없이 종료(`CLOSURE_*_2026-10-15.md`, 태그 `seal/*`) · Q2 P-C · Q3 arXiv-Sum 16k 표준 셀 · D-1…D-16(`results/handoff_review_2026-10-15/DECISIONS_Q1_consolidated_2026-10-15.md`)
  - D-3/D-15 fused 튜닝 11 config(비혼합 6 + SGLang mixed-chunk 5), 1차 = G_mixed, 분모 = 엄격 argmax λ\*_SLO(D-16)
  - D-6 span 예산 {8k,16k,32k,65k} · D-7 오류 1%는 클라이언트 측만
- 공통 측정 명세 **v0.2**(`SPEC_common_measurement_v0_2026-10-15.md`): 감사 1회차 NO-GO → 수리 → 2회차 D-C′ → 사용자 (a)로 해소. 감사 상한 도달, 이후 변경은 PREREG 단계 감사.
- mixed-chunk 코드 대조: "Sarathi식" 호칭 금지, hybrid mamba 경로 assert 없음·GPU 정확성 미검증(`CHECK_mixed_chunk_semantics_2026-10-15.md`)
- **merge** `multigpu-scaling` → main `dc06c870`(조건부 가, `VERDICT_spec_v0_1_and_closure_round2_2026-10-15.md` §6). ENGINE_DIGEST `82a5fc5b…`. 트랙 문서 후속 merge `351c904`.
- multigpu 승계: M2g 정확성 게이트 PASS 4표 봉인(99ed704). TP4·TP8 G2 `TP_NUMERIC_DIVERGENT` 병기 의무(MGH-3). M3 PREREG rev11·하네스 봉인 — merge로 `job_entry.sh` 로그 미러·heartbeat가 측정 경로에 들어와 "M3 경로 무변경" 확인은 무효.
## 다음 행동
2. §6-1 preflight(G0·M3 공용 1회): 설계 진행 중 `PREFLIGHT_merge_cl3_design_2026-10-15.md` → 사용자 승인 → VESSL 실행 → `PREFLIGHT_PASS`여야 GPU 측정 Job
3. M3 PREREG 개정(Q23: 명세 인용·arXiv-Sum shape·`max_prefill_tokens`·span 예산·절대 SLO·로그 knob `PDMUX_HEARTBEAT_S=0`) → 재감사 → M3 dry-run·승인
4. G0 계측 패치 I-1–I-5 + mixed-chunk 정확성 gate(engine-porter) → G0 PREREG(Q24)
## 사용자 결정 대기
- (해소 2026-10-15) main dev tree 동기화 완료 — ENGINE_DIGEST `82a5fc5b…` 일치. amd·opchar 로컬 영향 공지 `handoff-report/NOTICE_engine_tree_sync_2026-10-15.md`(각 트랙 세션이 `SGLANG_ENGINE_DEV`를 자기 사본으로 지정해야 함)
- **M3 PREREG 개정 범위** — 명세 v0.2 R-16대로 rev11 결과에 소급 금지 / 개정 후 재감사(권고) · 현행 rev11 제출(로그 knob·절대 SLO 미반영)
- **M3 제출 승인** — 모형 예보 예약 약 124 GPU-h, 최장 약 9 h(개정 뒤 재산정)
- **D-10** `CAP_BINDING` 문턱 10% 확정 — G0 PREREG 봉인 전
- 기존 유지: radix(OFF+편차 공시) · 128k 프로브(D11로 3-a 잔여와 묶음) · 장문 hold-out 모델 · CONSENSUS §1 행 38
## 실행 중·미회수 Job
- 없음 (Workspace `pdmux-probe`·`pdmux-build` standby)
## 금지 서술 (링크만)
N2 `VERDICT_result_n2_ea_7e09cc3_2026-10-13.md` 금지 32–42·N2C-102…108 · 감사 판정서 2개의 S-1–S-8·K1–K9·C-1–C-6·CL-1–CL-3 · MG `results/multigpu/PREREG_MULTIGPU_2026-10-12.md` §7·§15–§18 · "MuxWise 대비"·"Sarathi식"·"회귀 0건" · lockstep 기전·span 크기 수치는 유도값
## 최근 로그
- 2026-10-15(오후): promote(문서 정정·F3·F4) · Q1 준비·명세 v0 → 감사 2회 → v0.2 · N2·N3·E2 종료 · mixed-chunk 대조 · merge `dc06c870` · multigpu 트랙 이양
- 2026-10-15(오전): 인계 프롬프트 검토 · 방법론 안건 Q1–Q24 · span/lockstep·MuxWise/Bullet 대조 · cap 비용 모델 · F3·F4
- 2026-10-14: 주 모델 Nano-9B-v2 결정; 장문 hybrid 워크로드 조사
- 2026-10-13: Job-EA 회수·결과 감사·정본 rev95; 하네스 재구성
