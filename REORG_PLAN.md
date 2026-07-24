# 리포지토리 리팩토링 계획서 (실행 완료 — 아래는 원안 그대로 보존)

작성일: 2026-07-24. **집행 상태(같은 날 갱신)**: 사용자가 §6 결정 필요 항목 중
Q1/Q2/Q3에 대해 "적극안"을 승인해 실행했다(`deprecated/` 신설, 루트 `reports/`·
`deprecated_reports/`·`workspace/engine-port/reports/deprecated_reports/`·
`results/arxiv/` 이동, R1 로그 campaign 편입). 이어서 **후속 작업**으로
`workspace/engine-port/reports/` 전체가 저장소 최상위 `reports/`로 승격됐다
(`REPOSITORY_LAYOUT.md` "예외" 절 참조) — 이 계획서 본문의 `workspace/engine-port/reports/...`
경로 언급은 **작성 당시(이동 전) 기준**이며 사후에 다시 쓰지 않았다(계획서는
서술 문서라 실제 경로는 `deprecated/README.md`·`REPOSITORY_LAYOUT.md`·
`reports/paper/DOCUMENT_STATUS.md`를 따른다). 아래 본문은 원안 그대로다.

## 0. 핵심 결론 먼저

이 저장소는 이미 상당히 잘 정리돼 있다. 특히
`workspace/engine-port/reports/paper/DOCUMENT_STATUS.md`(2026-07-24 갱신, 오늘)가
"**파일은 link 보존을 위해 이동·삭제하지 않는다**"를 명시하고, Canonical/Historical
evidence/Superseded-stale 표로 사실상 전체 `reports/` 트리(루트 + engine-port)를
이미 **제자리 유지로 판정**해뒀다. 개별 오도 소지 파일에는 이동 대신 인라인
"SUPERSEDED" 배너를 이미 붙여뒀다(`figure_index.md`, `layer_aware_benefit_report.md`,
`framework_comparison.md`). 따라서 이번 재정리에서 실제로 옮길 만한 대상은
**`results/`의 미참조 중복 데이터**와 **루트에 떨어진 SLURM 로그 몇 개**로 좁다.

### 집계

| 판정 | 파일 수(대략) | 비고 |
|---|---:|---|
| KEEP-CANONICAL / KEEP-IMPORTANT (그대로) | ~1,500+ | 대부분 — 이미 정본 위계·자체 README·배너로 관리됨 |
| DEPRECATE (이번에 이동 제안) | 158 | `results/arxiv/`(155) + 루트 `r1_dual_worker_smi_862456.{out,err}`(2, tracked 중복) + `flash_attn_build.log`(1, untracked) |
| 결정 필요 | 7건 | 아래 §6 |

### 목표 구조 (제안)

```text
prefill-layer-alloc/
├── PROJECT_STATUS.md                      # 그대로 (정본 tier1)
├── REPOSITORY_LAYOUT.md                   # 그대로 (정본)
├── CLAUDE_CODE_SESSION_SUMMARY.md         # 그대로 (DOCUMENT_STATUS가 이동금지 명시)
├── reports/                               # 그대로 (전체 historical, README+개별 배너로 이미 관리)
├── deprecated_reports/                    # 그대로 두는 것을 권고 (§6-1 참조, 병합은 선택지)
├── results/
│   ├── stage1/ stage2/ stage3/ motivation/ serving-eval/   # 그대로 (참조되는 원자료)
│   ├── stage1_v2/                         # 빈 디렉터리 — 조치 불요 (§6-2)
│   └── arxiv/  → deprecated/results/arxiv/   [DEPRECATE]
├── slurm/                                 # 그대로 (고유 재현 스크립트)
├── deprecated/                            # ★신설 — 이번에 새로 격리하는 것만 담음
│   ├── results/arxiv/                     # results/arxiv/ 전체 이동
│   └── logs/
│       ├── r1_dual_worker_smi_862456.out
│       ├── r1_dual_worker_smi_862456.err
│       └── flash_attn_build.log
└── workspace/
    ├── engine-port/                       # 그대로 (paper/, CONSENSUS, results 13개 campaign 전부 KEEP)
    ├── characterization/                  # 그대로 — 이미 archive/v1_7b_saturation/ARCHIVE_NOTE.md로 자체 정리됨
    ├── serving-eval/, shared/             # 그대로 (활성 소스 트리)
```

기존 `deprecated_reports/`(루트) 와 `workspace/engine-port/reports/deprecated_reports/`는
**이번엔 건드리지 않는 것을 권고**(§6-1). 새 `deprecated/`는 이번 리팩토링에서
처음으로 격리 판정을 받은 항목만 담는 **별도 신규 트리**로 제안한다. 두 체계를
합칠지는 결정 필요 항목으로 남긴다.

---

## 1. 확인한 근거 (읽은 순서)

1. `CLAUDE_CODE_SESSION_SUMMARY.md` §4 — 현재 정본/역사적 참고자료/폐기 분류. (2026-07-22 작성, HISTORICAL 배너 있음, PROJECT_STATUS.md 하위로 이미 강등돼 있으나 파일 자체는 여전히 인용 대상)
2. `REPOSITORY_LAYOUT.md` — 트랙별 canonical 배치 규칙. `triage/`와 `results/<campaign>/*.sbatch`는 "이번 정리에서 일괄 이동하지 않았다"고 **명시적으로 스코프 제외**.
3. `PROJECT_STATUS.md` §"철회된 가설" — 무엇이 죽었는지(철회 배지 유지, 파일 이동과 무관).
4. `workspace/engine-port/reports/CONSENSUS.md` §4 "살아있는 문서" — 어떤 문서가 아직 인용 가능한지, `deprecated_reports/`는 이미 이력용으로 이관돼 있다고 자체 명시.
5. `workspace/engine-port/reports/paper/DOCUMENT_STATUS.md`(2026-07-24, **오늘 갱신**) — Canonical / Historical evidence 유지 / Superseded-stale 3단 표, **"파일은 link 보존을 위해 이동·삭제하지 않는다"**를 명문화. 이 문서가 사실상 이번 재정리의 상위 제약이다.
6. `reports/README.md` — 루트 `reports/`가 전체 characterization/simulator/초기 vLLM/철회 layer-aware 트랙의 historical artifact임을 자체 선언.
7. `workspace/characterization/archive/v1_7b_saturation/ARCHIVE_NOTE.md` — characterization 트랙은 **이미 자체적으로 v1→v2 아카이브 이관을 완료**하고 `git mv` 이력·이유·재사용 가능 컴포넌트를 문서화해뒀다. 모범 사례이므로 손대지 않는다.
8. 그 외 개별 파일 헤더(`head`)로 SUPERSEDED 배너 유무 확인, `grep`으로 상호 참조 여부 확인, `git ls-files`/`git log --diff-filter=R`로 tracked 여부와 과거 이동 이력 확인.

**중요 제약**: DOCUMENT_STATUS.md의 3개 표(Canonical / Historical evidence 유지 /
Superseded-stale)에 이름이 오른 문서는 **전부 이동 대상에서 제외**한다(문서
자체가 "이동하지 않는다"고 명문화했으므로). 여기엔 `reports/PAPER_RESULTS.md`,
`reports/FINAL_SUMMARY.md`, `CLAUDE_CODE_SESSION_SUMMARY.md`, 그리고
`workspace/engine-port/reports/`의 `research_arc.md`, `per_layer_type_postmortem.md`,
`dual_worker_design.md`, `realtrace_findings_and_open_branches.md`,
`slo_aware_scheduling_design.md`, `policy_comparison.md`, `system_vs_engine_vs_sim.md`,
`prefill_vs_decode_execution.md`, `r1_dual_worker_progress.md`,
`r1_dual_worker/job_862512.md`가 포함된다. **이 규율은 문서(리포트)에만 적용되고,
결과 원자료(`results/`)는 별개 판단**이다(rule 4: 위치 정리는 가능).

---

## 2. 루트 `reports/`, `deprecated_reports/` (67 + 14 tracked)

| 현재 경로 | 판정 | 대체한 정본 | 사유 | 제안 이동 |
|---|---|---|---|---|
| `reports/PAPER_RESULTS.md` | KEEP-CANONICAL | PROJECT_STATUS.md (claim만) | DOCUMENT_STATUS가 이동 금지 명시 | 없음 |
| `reports/FINAL_SUMMARY.md` | KEEP-CANONICAL | PROJECT_STATUS.md | 상동 | 없음 |
| `reports/layer_aware_benefit_report.md` | KEEP-IMPORTANT | — | 이미 인라인 SUPERSEDED 배너 있음(2026-07) | 없음 |
| `reports/framework_comparison.md` | KEEP-IMPORTANT | — | 이미 인라인 SUPERSEDED 배너 있음 | 없음 |
| `reports/figure_index.md` | KEEP-IMPORTANT | — | 이미 세트별 SUPERSEDED 배너·개별 행 표기 있음 | 없음 |
| `reports/layer_aware_7b_verification.md`, `matched_granularity_*.md`, `metrics_and_load_dependence.md`, `partition_engine_design.md`, `project_closure_report.md`, `queue_simulator_design.md`, `real_prefill_experiment_design.md`, `real_prefill_results.md`, `review_checklist.md`, `serving_eval_report.md`, `stage1_corrected_vs_hm_thesis.md`, `sweep_consistency_audit.md`, `temporal_vs_spatial_hybrid.md`, `vllm_validation.md`, `widened_sweep_validation.md`, `additional_value_paths.md`, `decode_saturation_and_crossover.md`, `experiment_index.md`, `experiment_schematic.md`, `figure_narrative_critique.md`, `handoff_prompt_for_web.md` | KEEP-IMPORTANT | — | 중복 아님 — characterization/simulator 트랙의 유일한 상세 기록. `results/stage1·2·3·motivation·serving-eval`을 참조 소스로 인용하고 있어(§3) 이동 시 링크 파손 | 없음 |
| `reports/figures/*` (8 PNG) | KEEP-IMPORTANT | — | `figure_index.md`가 참조하는 활성 그림 세트(일부는 세트별로 SUPERSEDED 표기됨) | 없음 |
| `reports/figures_v3/`, `reports/figures_v4/` | 결정 필요 | — | §6-3 | — |
| `deprecated_reports/` (14 files, 전체) | KEEP AS-IS | — | 이미 tier-5 격리 완료, 자체 README 있음, CONSENSUS.md가 "과거 보고서는 deprecated_reports/로 이관"이라 인용 | §6-1에서 병합 여부만 결정 |
| `CLAUDE_CODE_SESSION_SUMMARY.md` (루트) | KEEP-CANONICAL | PROJECT_STATUS.md | DOCUMENT_STATUS가 이동 금지 명시 (HISTORICAL 배너 이미 최상단에 있음) | 없음 |

**결론: 루트 `reports/`·`deprecated_reports/`는 이번 리팩토링에서 이동 대상이
없다.** 이미 dir-level README + 개별 인라인 배너로 관리되는 방식이 확립돼
있고, DOCUMENT_STATUS.md가 이를 오늘 날짜로 재확인했다.

---

## 3. 루트 `results/` (203 tracked)

| 현재 경로 | 판정 | 대체한 정본 | 사유 | 제안 이동 |
|---|---|---|---|---|
| `results/stage1/` (2 files, chunked SSM csv) | KEEP-IMPORTANT | — | `reports/sweep_consistency_audit.md`, `matched_granularity_nemotron_h.md`, `stage1_corrected_vs_hm_thesis.md`, `serving_eval_report.md`, `deprecated_reports/*` 등 다수가 상대경로로 인용. 워크스페이스의 `archive/v1_7b_saturation/results/stage1`은 **다른 세대**(파일명 규칙 다름, fig 번호 다름 — 후속 재측정)이지 동일본 아님 | 없음 |
| `results/stage2/` (2 files) | KEEP-IMPORTANT | — | 동일 사유, `serving_eval_report.md` 등이 인용 | 없음 |
| `results/stage3/` (7 files) | KEEP-IMPORTANT | — | `serving_eval_report.md`가 인용 | 없음 |
| `results/motivation/` (18 files) | KEEP-IMPORTANT | — | `workspace/engine-port/reports/prefill_vs_decode_execution.md`(DOCUMENT_STATUS "이동 금지" 목록)가 인용 — 데이터를 옮기면 그 문서의 링크가 깨짐 | 없음 |
| `results/serving-eval/` (19 files) | KEEP-IMPORTANT | — | `reports/serving_eval_report.md`, `vllm_validation.md`가 인용 | 없음 |
| `results/stage1_v2/` | 조치 불요 | — | tracked 파일 0개, 완전히 빈 디렉터리(git은 빈 디렉터리를 추적하지 않음) | §6-2 |
| `results/arxiv/{stage1,stage1_arxiv,stage1_archived}` (155 files) | **DEPRECATE** | — | 저장소 전체에서 `results/arxiv`를 인용하는 `.md`가 **0건**(grep 확인). 3개 하위 디렉터리가 서로 90%+ 겹치는 동일 stage1 그림/csv 번들(파일명만 약간 다름 — `a100_80gb` vs `a100-sxm4-80gb` 등, 아마 arxiv 제출용으로 만든 스냅샷 세대들). 병합 편집은 하지 않고 통째로 격리만 한다 | `git mv results/arxiv deprecated/results/arxiv` |

---

## 4. 루트 `slurm/`, 루즈 루트 파일

| 현재 경로 | 판정 | 사유 | 제안 이동 |
|---|---|---|---|
| `slurm/measure_7b_e5.sh`, `measure_7b_layer_aware.sh`, `measure_layer_aware_inputs.sh`, `measure_prefill_sm.sh`, `remeasure_7b_e3e5.sh`, `sweep_stage1_v2.sh` | KEEP-IMPORTANT | `workspace/characterization/slurm/*.sh`와 이름이 겹치지 않는 고유 1회성 측정 스크립트. `remeasure_7b_e3e5.sh`는 장기메모리에 인용된 **job 797832**(7B TRITON_CACHE_DIR 수정 재측정)의 재현 스크립트 — 삭제·이동 시 재현 경로 소실 | 없음 |
| `slurm/logs/*` (10 files, **untracked**) | KEEP AS-IS | 위 스크립트들의 원시 로그, `git ls-files`에 없음(gitignored) — git-mv 대상 아님 | 없음(로컬 스크래치) |
| `CLAUDE_CODE_SESSION_SUMMARY.md` | 위 §2 참조 | — | — |
| `r1_dual_worker_smi_862456.out`/`.err` (**tracked**) | **DEPRECATE** | `workspace/engine-port/results/r1_dual_worker/job_862456/slurm.{out,err}`에 이미 동일 job의 정리된 사본이 있음(파일명만 `slurm.out`/`slurm.err`로 정규화) — 루트 사본은 중복 leftover. CLAUDE.md 규칙("루트 SLURM .out/.err는 커밋 대상 아님")에도 위배되게 커밋돼 있었음 | `git mv r1_dual_worker_smi_862456.out deprecated/logs/` (`.err` 동일) |
| `r1_dual_worker_smi_862512.out`/`.err`, `r1_dual_worker_no_smi_862513.out`/`.err` (**untracked**) | 결정 필요 | §6-4 | — |
| `sglang_0723_8_128_16.jsonl` (**untracked**) | 결정 필요 | §6-5 | — |
| `flash_attn_build.log` (**untracked**) | **DEPRECATE**(파일시스템 이동, git 무관) | 특정 실험과 무관한 환경 빌드 로그, 5월 7일자 | `mv flash_attn_build.log deprecated/logs/` |
| 루트 `logs/`(323 files) + `logs/archived/`(53 files) | KEEP AS-IS(범위 외) | 전부 untracked(gitignored) — git 이력과 무관한 로컬 스크래치. 이미 `archived/` 하위분류가 있음 | 없음(원하면 파일시스템 정리만, git 조작 아님) |
| `.slurm-agent/`, `workspace/slurm-agent/` | 범위 외 | 연구 결과/보고서가 아니라 slurm-agent 도구의 내부 상태·자체 잡 리포트 캐시. 사용자가 지정한 스캔 범위(`reports/`, `results/`, `slurm/`, `deprecated_reports/`, `workspace/engine-port`)에 없음. `.slurm-agent/`는 gitignored | 이번 계획에서 제외, 필요시 별도 검토 |

---

## 5. `workspace/engine-port/`, `workspace/{characterization,serving-eval,shared}`

| 영역 | 판정 | 사유 |
|---|---|---|
| `workspace/engine-port/reports/paper/*` (7) | KEEP-CANONICAL | 정본 tier2, `.claude`급 보호 |
| `workspace/engine-port/reports/CONSENSUS.md`, `RESUME.md` | KEEP-CANONICAL | 정본 tier3/4 |
| `workspace/engine-port/reports/*.md`(그 외 14개), `*.png`(9), `plot_*.py`(7) | KEEP-CANONICAL | CONSENSUS §4 "살아있는 문서" 표에 전부 등재 |
| `workspace/engine-port/reports/deprecated_reports/` (12) | KEEP AS-IS | 이미 자체 README로 격리 완료(2026-07-17), 링크 보존 위해 미이동 |
| `workspace/engine-port/reports/r1_dual_worker/` (4 .md) | KEEP-CANONICAL(단 `job_862512.md`은 stale) | DOCUMENT_STATUS가 이동 금지 명시 |
| `workspace/engine-port/results/{a_substrate,cudagraph_probe,pf_boundary,pf_boundary_fix1,pf_boundary_fix2,prefill_knee,r0a,r0b,r0c,r0d_coord_la,r1_dual_worker,slo_sched,v4_multiwin}` (13개 campaign, 929 tracked files) | KEEP-CANONICAL | 서로 다른 실험 세대(연대기 순: r0a→r0b→r0c/r0d→a_substrate→v4_multiwin, 별도로 slo_sched·prefill_knee·pf_boundary·cudagraph_probe·r1_dual_worker) — 중복 없음. `REPOSITORY_LAYOUT.md`가 `results/<campaign>/`를 immutable raw artifact로 명시 보호 |
| `workspace/engine-port/triage/` (364 files) | KEEP AS-IS(범위 외) | `REPOSITORY_LAYOUT.md`가 "이번 정리에서 일괄 이동하지 않았다"고 이미 명시. `DEPRECATED_gil_client.md`도 이미 파일명으로 자체 표기됨 |
| `workspace/characterization/` (367 files: `experiments/`, `src/`, `results_v2/`, `archive/v1_7b_saturation/`, `slurm/`, `reports/v2_report.md`) | KEEP AS-IS | 활성 소스+시뮬레이터 트랙. `archive/v1_7b_saturation/ARCHIVE_NOTE.md`가 v1→v2 전환의 `git mv` 이력·재사용 가능 컴포넌트를 이미 자체 문서화(모범 사례로 손대지 않음) |
| `workspace/serving-eval/` (17 files + `.venv/` gitignored) | KEEP AS-IS | 활성 서빙 평가 하네스 소스 |
| `workspace/shared/` (7 files) | KEEP AS-IS | 트랙 공통 설정 |

이 영역엔 **이동 제안이 없다.** 13개 campaign 디렉터리를 개별 확인했으나
이름·시기·참조처가 전부 달라 중복이 없었다(즉 사용자가 우려한 "같은 내용을
여러 곳에 중복 보유"가 이 하위트리에서는 발견되지 않았다).

---

## 6. 결정 필요 항목

### 6-1. 두 `deprecated_reports/`를 단일 `deprecated/`로 흡수할지

- 현재: 루트 `deprecated_reports/`(14, 자체 README) + `workspace/engine-port/reports/deprecated_reports/`(12, 자체 README, 2026-07-17 격리) — 이미 각자 tier-5 규율을 만족한 상태.
- 병합 시 이득: 저장소 전체에서 "격리처가 하나"라는 사용자 요구를 문자 그대로 만족.
- 병합 시 비용: `CONSENSUS.md`("과거 보고서는 `deprecated_reports/`로 이관"), 두 README.md, `CLAUDE_CODE_SESSION_SUMMARY.md`의 상대경로 링크를 **함께 수정**해야 함 — 이는 "내용 편집"에 해당하지 않지만 여러 정본급 문서의 경로 참조를 건드리는 작업이라 신중해야 함.
- **질문**: (A) 두 곳 다 현행 유지(권고, 리스크 최소) vs (B) 루트 `deprecated_reports/` → `deprecated/reports/root/`, engine-port 쪽 → `deprecated/reports/engine_port/`로 통합 이동 + 참조 문서 4곳 링크 일괄 수정. 어느 쪽?

### 6-2. `results/stage1_v2/`

- tracked 파일 0개, 완전히 빈 디렉터리. 남겨진 이유(과거 실험이 결과 없이 끝났거나, `.gitkeep` 없이 빈 채 커밋된 흔적)를 알 수 없음.
- **질문**: 그대로 둘지(원 상태 보존), 아니면 삭제(디렉터리라 git엔 흔적이 없으므로 "삭제 금지" 규율과 무관 — 파일이 아니라 빈 폴더)할지?

### 6-3. `reports/figures_v3/`(7) · `reports/figures_v4/`(19) vs `reports/figures/`(8, `figure_index.md` 인용)

- `figures_v3`(7/14 09:32)·`figures_v4`(7/14 09:53)는 `figure_index.md`(7/6 작성)보다 **나중**에 만들어졌고, 각자 자체 README로 데이터 출처(job id 포함)를 잘 기록해뒀다. 그런데 **어떤 `.md` 리포트도 이 두 세트를 인용하지 않는다**(자기 자신의 README 제외) — `figure_index.md`가 갱신 안 된 채 방치된 것으로 보인다.
- `reports/figures/`의 일부(summary_dashboard, layer_aware_result, layer_aware_size_trend)는 이미 SUPERSEDED로 배너 표기됨, 반면 `figures_v3/v4`는 엔진 실측 기반이라 오히려 더 최신·더 신뢰도 높은 그림일 수 있음.
- **질문**: (A) `figure_index.md`를 갱신해 `figures_v3`/`figures_v4`를 현재 색인에 편입할지, (B) 이 두 세트를 "미채택 초안"으로 보고 그대로 둘지(현행), (C) `reports/figures/`의 SUPERSEDED 개별 PNG만 별도 격리할지 — 이는 콘텐츠·결과 판정(result-analyst/claims-auditor 영역)에 가까워 doc-steward가 임의로 정하지 않음.

### 6-4. 루트 `r1_dual_worker_smi_862512.{out,err}`, `r1_dual_worker_no_smi_862513.{out,err}` (untracked)

- `workspace/engine-port/results/r1_dual_worker/job_862512/`에는 상세 아티팩트(bench/trace/server 로그 다수)가 있지만 **`slurm.out`/`slurm.err` 자체는 없음** — 즉 이 두 로그가 job 862512의 SLURM 실행 표준출력을 담은 **유일한 사본**. job 862513은 아예 engine-port results 하위에 캠페인 폴더가 없고, `workspace/slurm-agent/reports/job_862513.md`(slurm-agent 도구 리포트)만 존재.
- **질문**: (A) 캠페인 기록을 완성하는 쪽으로 `job_862512/slurm.{out,err}`, 신규 `job_862513/slurm.{out,err}`로 옮길지(레코드 보완, 권고), 아니면 (B) CLAUDE.md의 "루트 SLURM out/err는 커밋 대상 아님" 문구를 문자 그대로 따라 `deprecated/logs/`로 격리할지?

### 6-5. `sglang_0723_8_128_16.jsonl` (untracked, 루트)

- 파일명 패턴(`sglang_<날짜>_<params>.jsonl`)이 서빙 trace/벤치 출력처럼 보이나 어느 campaign에도 파일링되지 않음. 날짜(0723)가 R1 dual-worker job들(862xxx, 7/23)과 겹침.
- **질문**: 내용을 확인해 관련 campaign(`workspace/engine-port/results/r1_dual_worker/` 등)에 편입할지, 아니면 `deprecated/logs/`로 격리만 할지? (내용 미확인 상태이므로 임의로 확정하지 않음.)

### 6-6. `workspace/engine-port/triage/` 내부 미결 잔재

- `triage/.triton_cache`, `triage/.sglang_cache`는 untracked(gitignored) 이라 조치 불요. `triage/t5_greenctx`(6 tracked), `triage/p1_0_SUCCESS_evidence`(0 tracked, untracked)는 `REPOSITORY_LAYOUT.md`가 명시적으로 스코프 제외한 `triage/` 안에 있어 이번 계획에서 그대로 둔다. 확인만 해두고 조치는 제안하지 않음.

### 6-7. `.slurm-agent/`, `workspace/slurm-agent/` 스코프 포함 여부

- 사용자가 지정한 스캔 범위 밖이라 이번 계획에 포함하지 않았다. 필요하면 별도 리팩토링 계획으로 다룰 것을 제안.

---

## 7. 실행 스크립트 (승인 후에만 실행 — 지금은 실행하지 않음)

```bash
#!/usr/bin/env bash
# REORG_PLAN.md 승인 후 실행. 이 스크립트는 아직 실행되지 않았다.
set -euo pipefail
cd /scratch/ehmoon/whlee/prefill-layer-alloc

# --- 신규 격리 트리 생성 ---
mkdir -p deprecated/results deprecated/logs

# --- (확정 DEPRECATE 1) results/arxiv/ 전체 — 어떤 문서도 참조하지 않는
#     3-way 중복 stage1 번들. 내용 병합/편집 없이 통째로 격리.
git mv results/arxiv deprecated/results/arxiv

# --- (확정 DEPRECATE 2) 루트에 tracked 상태로 남은 R1 job 862456 로그 —
#     workspace/engine-port/results/r1_dual_worker/job_862456/slurm.{out,err}에
#     이미 정리된 사본이 있는 중복 leftover.
git mv r1_dual_worker_smi_862456.out deprecated/logs/r1_dual_worker_smi_862456.out
git mv r1_dual_worker_smi_862456.err deprecated/logs/r1_dual_worker_smi_862456.err

# --- (확정 DEPRECATE 3, untracked라 git mv 아님) 특정 실험과 무관한 환경 빌드 로그.
mv flash_attn_build.log deprecated/logs/flash_attn_build.log

# --- 이하는 §6 "결정 필요" 항목이 해소된 뒤에만 추가한다 (지금은 실행 안 함) ---
# 예시 (6-1, 옵션 B를 고른 경우에만):
#   mkdir -p deprecated/reports
#   git mv deprecated_reports deprecated/reports/root
#   git mv workspace/engine-port/reports/deprecated_reports deprecated/reports/engine_port
#   # + CONSENSUS.md · 두 README.md · CLAUDE_CODE_SESSION_SUMMARY.md의
#   #   deprecated_reports/ 상대경로 링크 일괄 수정 (별도 커밋 권고)
#
# 예시 (6-4, 옵션 A를 고른 경우에만):
#   mv r1_dual_worker_smi_862512.out workspace/engine-port/results/r1_dual_worker/job_862512/slurm.out
#   mv r1_dual_worker_smi_862512.err workspace/engine-port/results/r1_dual_worker/job_862512/slurm.err
#   mkdir -p workspace/engine-port/results/r1_dual_worker/job_862513
#   mv r1_dual_worker_no_smi_862513.out workspace/engine-port/results/r1_dual_worker/job_862513/slurm.out
#   mv r1_dual_worker_no_smi_862513.err workspace/engine-port/results/r1_dual_worker/job_862513/slurm.err

echo "완료: DEPRECATE 확정 3건만 이동함. 결정 필요 7건은 사용자 응답 후 별도 처리."
```

---

## 8. 요약

- 계획서 경로: `/scratch/ehmoon/whlee/prefill-layer-alloc/REORG_PLAN.md`
- KEEP(현행 유지): ~1,500+ 파일 — 정본 위계·DOCUMENT_STATUS.md의 "이동 금지" 명시·
  기존 dir-level README/개별 배너 체계가 이미 충분히 격리 기능을 하고 있음을 확인.
- DEPRECATE(이번에 이동 제안, 실행 전): **158개 파일** — `results/arxiv/`(155, 무참조
  3-way 중복) + 루트 tracked `r1_dual_worker_smi_862456.{out,err}`(2, 이미 다른 곳에
  정리된 사본) + untracked `flash_attn_build.log`(1).
- 결정 필요: **7건** — (6-1) 두 `deprecated_reports/` 병합 여부, (6-2) 빈
  `results/stage1_v2/` 처리, (6-3) `figures_v3`/`figures_v4` 색인 편입 여부(콘텐츠
  판정이라 doc-steward 임의 결정 보류), (6-4) 루트 미조직 R1 862512/862513 로그
  귀속처, (6-5) 미분류 `sglang_0723_8_128_16.jsonl` 처리, (6-6) `triage/` 내부
  잔재(조치 불요, 확인만), (6-7) `.slurm-agent/`/`workspace/slurm-agent/` 스코프
  포함 여부.
