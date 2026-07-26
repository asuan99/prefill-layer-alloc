# Session handoff — 2026-07-26

## 이번 세션 요약

catch-up 후 사용자 지시 **B→C→A 순** 진행: (B) 직전 handoff 커밋, (C) §1-20 spatial-decoupling
트랙 설계 검토 → 사용자 지적으로 **신규성 축을 disaggregation→multiplexing으로 정정**하고
정본화, (A) long-context **Stage 0(L−2)** 실험을 설계→구현→측정→판정까지 완주. Stage 0에서
**SGLang에 pure Mamba2 지원을 신규 구현**하고 3-arm(M/H/T) coupled 스윕을 돌린 결과,
**coupled 측정이 confounded임을 pure-Mamba 음성대조가 발화**시켜 잡아냈고(아크 3번째 반전 방지),
clean 신호(D16≡D108=1.00)로 **decode는 운영점서 SM-무감각(non-binding), 16k·hybrid까지** 확정.
long-ctx 충돌트랙(H_L4/H_L5) 붕괴·HE0/벡터1 ctx-무관 강화. 시각화(Artifact)+정본 반영 완료.
전부 로컬 커밋(7개), push 안 함.

## 결정·측정

### C — multiplexing positioning 정정 (정본화)
- **spatial 검토 초판(`reports/spatial_decoupling_design_review_2026-07-25.md`)의 신규성 절은
  disaggregation 축으로 오조준**(사용자 지적). venue-strategist 재조사(multiplexing 축):
  DuetServe/MuxWise(ASPLOS'26)/SGLang-pdmux/Nexus/Bullet **전부 Transformer-centric** → **hybrid
  co-located PD-mux는 공백(신규성 부분적·실재)**. 방어자산 = **DuetServe(동적>static, Transformer·
  libsmctrl·throughput)와 정면 상반**(우리=hybrid·green-ctx·conjunctive-SLO서 동적<static).
- **negative 2갈래 분해**: (A) green-context 종속(layer-aware死·drain 42→124ms·cudagraph 비양립 —
  헤드라인 금지) / (B) mechanism-independent(lever-weakness·entanglement·비대칭 — 방어가능).
- **cross-substrate 이식 불필요** 확정: MPS=재구조화 과대·정적, libsmctrl=비-vendor·세대귀속
  (driver-580 BLOCKED 증거). 대신 **green-context=배포 가능한 유일 vendor primitive**를 방어로 삼고,
  Risk 2(모델 vs substrate 귀속)는 **같은 green-context 위 Transformer 대조 + roofline lever-weakness
  + 기측정 entanglement**로 닫음. **XS-series 철회 → TC-series(Transformer-control) 게이트로 대체.**
- engine-porter 기판 리뷰(부수): SGLang v0.5.10엔 PD disaggregation + hybrid mamba state 전이가
  **이미 upstream 존재**(→ "처음부터 만들기" 전제 틀림). 단 "116>108 한 GPU" 가설의 정직한 검증은
  green-context 프로세스-내부 장애로 미구현 신규 아키텍처.

### A — Stage 0 (long-ctx L−2) — decode 운영점 SM-binding 게이트
- **설계 진화**: r0c in-forward 핀이 cudagraph replay서 死(forward 우회) 발견 → **pdmux green-ctx
  capture 경로**로 재설계(운영점 충실). 사용자 제안으로 **3-arm 구성 스펙트럼**:
  M=pure Mamba2-2.7B(음성, SSM-only) / H=Zamba2-2.7B(hybrid 타깃) / T=Qwen2.5-3B(양성, pure-attn),
  전부 bf16(dtype 일치). coupled 운영점(keepalive-prefill), decode-SM{16,44,92 핀 + 108 no-split ref},
  ctx{4k,8k,16k}, REPS=3.
- **측정(jobs 864230=H,T · 864601=M, 전 config PIN_CHECK PASS)**:

  | arm | ctx | D16/D44/D92/D108 p50 (ms) | D16/D92 (co-swept) | D16/D108 (clean) |
  |---|---|---|---|---|
  | T | 16k | 25.3/13.8/11.6/25.3 | 2.18× | **1.00** |
  | M | 16k | 16.3/8.2/6.6/16.3 | 2.47× | **1.00** |
  | H | 16k | 46.8/19.8/13.1/46.7 | 3.58× | **1.00** |
  (4k/8k 동일 패턴, ctx-불변. 원자료 `results/stage0_xctrl/s0sweep_*_result.txt`·`*_raw.jsonl`.)

- **판정 1 — coupled ITL(D) 스윕 CONFOUNDED (CONFIRMED)**: ITL(D) 곡선이 decode-SM binding이 아니라
  **prefill 경합/batch-entanglement**를 잼. 3중 삼각검증: (i) D108 앵커=D16과 ≤0.5% 동일이고 D92보다
  느림=비단조 → SM이 원인 아님, (ii) **M 음성대조**가 full 2.46× 곡선(mamba decode는 O(1)라 SM-bound
  불가), (iii) 최속점 D92=prefill 16SM 굶긴 곳, wall 5.0–5.4× 붕괴=직렬화. **예비분석의 "H 3.35×=
  binding 후보"는 confound가 그대로 재현**(M이 T<M<H 중간에 앉음이 반증).
- **판정 2 — clean 신호 = NULL**: 유일 de-confounded 대조 **D16 vs D108 = 1.00±0.01, 3 arm×3 ctx** →
  **decode는 16→108 SM에 무감각(non-binding), Transformer·Mamba2·hybrid 전부, 16k까지.** r0c 확증 +
  hybrid·서빙 in-situ·16k 확장.
- **판정 3 — Stage 0 게이트 = non-binding**(설계 §4 "게이트 실패이자 강한 결과" 분기): long-ctx
  충돌 가설(H_L5 공간·H_L4 시간)이 필요한 decode-floor 상승 못 얻음 → **붕괴(이 regime)**, HE0/벡터1
  ctx-무관 강화. HE2·r0c와 정합·확장(반전 아님).

## 코드·문서 변경 (전부 커밋됨, push 안 함)

- `b3878a3` — spatial-decoupling + Stage 0 xctrl 설계 검토 + canon 상태.
- `f8ad50e` — **pure Mamba2 아키텍처 SGLang 지원 신규 구현**: `src/{configs,models}/mamba2.py`(미러),
  `src/patches/mamba2_pure_ssm_arch.patch`, `scripts/models/convert_mamba2_native.py`,
  sync_engine_tree.sh manifest. 국소 수정 4건(cell_size==0 guard · v_head_dim `full_layer_nums==0`
  guard · dtype · native `.bin` 로더). CPU unittest 23/23 통과, dev-tree 편집분은 patch로 추적.
- `9b8246b`/`fdbb0f9`/`1bf4264` — Stage 0 하네스(smoke/sweep/client/pdmux configs) + coupled/(A)
  3-arm 개정 + 설계 문서.
- `c782280` — **Stage 0 verdict 정본화**: 신규 `reports/stage0_verdict_2026-07-26.md` +
  `reports/stage0_sm_scaling.html`(figure, 데이터 임베디드) + PROJECT_STATUS(§Stage0)·CONSENSUS
  (§1-21 결과·§3-9 방법론 교훈)·longcontext_trace_plan(§0.6, H_L4/H_L5 반증)·CLAIM_EVIDENCE_MATRIX
  (Claim A 16k 확장)·EXPERIMENT_ROADMAP.
- `4d5b296` — 스윕 QOS 재구조화(array 0-2 per-arm + ctx in-job loop) + `s0_prelim_itl_analysis.py`.
- **메모리**: `MEMORY.md` + `slo-aware-scheduling-track.md`에 C positioning·Stage 0 결과·방법론
  교훈 반영(doc-steward, 세션 중). ⚠️ home01 inode 쿼터 임박(99358/100000)으로 Edit가 EDQUOT →
  in-place Python 재작성으로 우회. **다음 메모리 편집 시 재발 가능**(정리는 사용자 동의 필요).
- **시각화 Artifact**(repo 밖): https://claude.ai/code/artifact/54fa7c37-6774-40cb-90c8-1f8d254960e8
  (repo엔 `reports/stage0_sm_scaling.html`로 커밋됨).

## 열린 항목 / 다음 세션 시작점

- **★de-confound 재측정(미실행, 사용자가 현 증거로 확정 택함)**: coupled ITL(D)의 confound를 없앤
  깨끗한 decode-SM 측정. 조건 = **{decode-SM만 변동 · prefill-SM 고정(예 `[16,D]`·나머지 idle) 또는
  decode-only · batch occupancy 고정·step별 로깅 · 용량 이하 steady-state arrival rate}**. non-binding
  방향은 견고하므로 airtight화용(방향 뒤집을 가능성 낮음). prefill-SM 고정은 config-only 가능성 有
  (d92 두-division 트릭처럼).
- **>16k·큰 모델 미측정**: Stage 0 결론은 {2.7–3B, ctx≤16k, 이 coupled 하네스} 한정. long-ctx
  frontier(decode-floor가 attention 비중으로 오르는 영역)는 더 긴 ctx·큰 모델서 재검증 여지.
- **TC-series(Transformer-control on green-context)** — venue positioning의 Risk 2 게이트. 이번
  Stage 0가 이미 green-context 위 Transformer(T arm) SM-민감도를 재긴 했으나, TC-series는 **서빙-레벨
  static-vs-dynamic Transformer 대조**(HE0 귀속의 full 버전) — 별도 실험. `EXPERIMENT_ROADMAP 벡터2`.
- **§1-20 spatial decoupling** — 논문 기여 아님(방화벽)으로 정리됐으나 트랙 자체는 열림(별도 device
  pool disaggregation). 우선순위 낮음.

## 미완·주의

- **하네스 버그(sync-race)**: 동시 array run(MaxJobsPU=2)이 **공유 dev-tree에 sync_engine_tree.sh를
  동시 install** → race(`install: File exists`). M arm 최초 실행(864230_0)이 이 때문에 1초 실패 →
  단독 재실행(864601)으로 회복. **`sync_engine_tree.sh`에 flock 필요** 또는 멀티-arm sbatch는 sync를
  1회 선행/직렬화. ★**해결됨(2026-07-26 후속 세션, `f99fd84`)**: cross-node flock(자기 re-exec)
  + manifest temp+rename. 재현 검증 = 락 없이 10동시 중 7–8 실패 / 락 적용 0 실패(3 trial),
  타임아웃 시 nonzero exit. 실기전은 parent-dir mkdir이 아니라 `install`의 unlink→create.
- **admission latch stale-True 버그**(multiplexing_mixin.py:758 근처) known-latent, 미수정·보류(직전
  세션부터). R2/admission 착수 시 선수정.
- **de-confound 미실행** = Stage 0 "민감도 magnitude"는 confound로 측정 불가, **방향(non-binding)만**
  clean 하위신호(D108 앵커·M-control·D92 비단조)에서. 재인용 시 이 caveat 유지. regime scope 명시.
- **미커밋**: runtime 결과(`s0sweep_*_result.txt`·`s0smoke_*`·`s0pin_*`·`*_prompt.txt`)·
  `runtime_source_manifest.sha256` — 의도적 제외(규율). 실행 중 job 없음. 로컬 커밋 push 안 됨(PAT 노출).
- **claims-auditor 미경유**: Stage 0 confound/non-binding 판정은 result-analyst rigorous 판정(3중
  삼각검증)까지. 정본에 "확정"으로 쓸 때 필요하면 claims-auditor 반증을 추가로 걸 것(현재는 negative·
  범위한정이라 overclaim 위험 낮음).
