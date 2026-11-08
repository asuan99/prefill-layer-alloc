# 메모 — PD-mux 구현 수정 계획(E1–E3)이 정책 캠페인 수치에 주는 편차 검토 (2026-10-30)

> ★결과 감사 r1(2026-11-08) GO-with-edits — 판정서 `reports/audit/2026-11-08_unaudited_r1/AUDIT_design_defects_policy_memos_dstar_r1_2026-11-08.md`, 수정 전 인용 금지 목록 적용. 본문은 고치지 않았다.

> 지위: 검토 메모(미감사). 새 측정 0. 대상 계획 = 작업 루트 `PDMUX_IMPL_FIX_PLAN_2026-10-28.md`, 진행 = worktree `prefill-layer-alloc-implfix`(branch `impl-defects-e123`, Phase 0 `results/impl_defects/PHASE0_CPU_CHECK_2026-10-28.md`, GPU gate c03d622 FAIL — 원인은 플래그 경로 밖: G0 결정론 boot 16k 할당 실패·G1 이중 모드 키·G4 `.item()` poll 계수).
> 정책 캠페인 수치(CAP1-R·CAP2·TUNE, EVAL 예정)는 전부 엔진 digest 65cbb49d(커밋 3858f129) = **수정 전 루프**에서 나왔다.

## 1. 우리 W1 셀의 실행 영역 [측정: server.log `#new-token`, 유도: chunks = ceil(56 / max(1, 65536 // tokens))]
CAP1-R 5 boot·TUNE 12 boot 모두 prefill 배치 토큰 p50 36–51, p90 724–1092, 최대 3.5k–9.2k → **multi-chunk span 비율 5.5–9.4%**, chunk 수 p90 1, 최대 4–8. 즉 W1(ShareGPT)은 대부분 "미발화"(span 1개) 영역이고, 소수의 큰 배치만 2–8 chunk.
(boot 전체 집계, warm-up·기준 셀 포함 — 근사)

## 2. 결함·수정별 예상 편차 (방향과 대상; 크기는 미측정)
| 수정 | 우리 캠페인에서 걸리는 곳 | 예상 |
|---|---|---|
| B1/E3 (전환 drain → handoff) | 동적 arm 전환 시에만 | **≈0**: TUNE 선택 arm은 FEAS 혼잡 가드로 전환 0회(4/4), EVAL도 같은 고정 예보. a2/a3 구성(전환 4–9회)만 해당 — span ≥2 chunk에서 전환한 경우뿐(§1상 드묾) |
| B2 (decode-empty 시 쌓인 span) | 모든 PD-mux arm, decode가 빈 순간 | TTFT 측만, 작음: HI 과부하에서 decode가 거의 비지 않고, W1 span 대부분이 1 chunk라 쌓인 span이 짧음. LO 구간에서 드물게 발생 가능 |
| E1 (시간 기준 span) | multi-chunk가 될 큰 배치(5.5–9.4%)와 지금 1 chunk로 도는 ~0.1 s급 배치 | **편차 발생 가능성 가장 큼**: 큰 배치를 decode 박자 단위로 쪼개면 TTFT(그 요청)↑ 또는 ↓, 동거 decode 간격 분포 변화 → ITL p95 다리·λ*·best D가 움직일 수 있음. 다만 VESSL·Nano 진단상 1 chunk span이 decode를 멈추지 않았으므로(DIAG_lockstep, 별도 정지 항 ≤1 ms) ITL 측 개선 여지는 작을 수 있음 |
| E2 (발행 깊이 제한) | 적체 영역 | W1에서는 작음(§1). **W3(arXiv-Sum 7k 평균)는 적체 영역**이라 CAP2 λ*_W3(이미 취약, 통과 여유 1요청)는 크게 움직일 수 있음 → H-D 보류 판단과 일치 |
| OFF 경로 동일성 | 수정 트리로 엔진을 바꿀 때 | 코드상 OFF 동일 주장, **GPU 미검증**(gate FAIL 원인이 비플래그라도 OFF 동일성 자체는 실측되지 않음). 수정 트리 digest(fad4f110/0a5d9c3c) ≠ 65cbb49d |

## 3. 결론(제안)
- 정책 캠페인 수치는 "SGLang 공개 PD-mux 구현(수정 전 루프)" 범위로 고정해 쓴다(계획 D2·D3와 같은 방향).
- EVAL(봉인 8d960036, digest 65cbb49d)은 수정 계획과 독립적으로 유효 — 선택 동적 arm이 전환하지 않아 B1/E3 영향 ≈0, W1 대부분 1 chunk라 E1/E2 영향도 제한적일 것으로 예상(미측정).
- 편차 크기는 추정이 아니라 측정으로 답해야 한다: implfix GPU gate 통과 뒤, **같은 Job·같은 GPU**에서 W1 HE0 셀(static D44 + dyn_a4_w3)과 W3 셀 하나를 OFF vs E1(+E2) ON 짝으로 돌리는 블록이 최소 실험(n=4 seed). 그 전에는 "수정이 결과를 바꾼다/안 바꾼다" 어느 쪽도 쓰지 않는다.
