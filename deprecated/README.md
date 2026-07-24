# deprecated/ — 통합 격리처 (2026-07-24 신설)

이 트리는 저장소 여러 곳에 흩어져 있던 historical/superseded 산출물을 top-level
축소를 위해 하나로 모은 것이다. `REORG_PLAN.md`(2026-07-24 작성)의 결정 필요
항목 Q1/Q2/Q3에 대해 사용자가 "적극안"을 승인해 실행했다. 전부 `git mv`(tracked
파일) 또는 plain `mv`(untracked 파일)이며 **내용은 편집하지 않았다** — 경로만
바뀌었다. `git status`가 rename으로 기록하므로 필요하면 되돌릴 수 있다.

## 출처 3곳 + 2건

| 새 위치 | 원 위치 | 파일 수 | 왜 여기로 |
|---|---|---:|---|
| `reports/historical_root/` | 루트 `reports/`(전체, `figures/`·`figures_v3/`·`figures_v4/` 포함) | 67 | characterization/simulator/초기 vLLM validation/철회된 layer-aware 논문 트랙의 historical artifact. 기존엔 자체 README 배너로만 격리돼 있었으나(link 보존 방침), 오늘 top-level 축소를 위해 물리 이동으로 override 승인됨 |
| `reports/quarantine_root/` | 루트 `deprecated_reports/` | 14 | 이미 tier-5 격리 완료 상태였던 것을 통합 위치로 재배치 |
| `reports/quarantine_engine_port/` | `workspace/engine-port/reports/deprecated_reports/` | 12 | 상동 — engine-port 쪽 격리분을 통합 위치로 재배치 |
| `results/arxiv/` | 루트 `results/arxiv/`(`stage1/`, `stage1_arxiv/`, `stage1_archived/`) | 155 | 저장소 어떤 문서도 참조하지 않는 3-way 중복 stage1 그림/csv 번들(파일명 규칙만 다른 여러 세대 스냅샷). 병합·편집 없이 내부 구조 그대로 통째로 이동 |
| `logs/` | 루트 tracked `r1_dual_worker_smi_862456.{out,err}` | 2 | `workspace/engine-port/results/r1_dual_worker/job_862456/slurm.{out,err}`에 이미 정리된 사본이 있는 중복 leftover |
| `logs/` | 루트 untracked `flash_attn_build.log` | 1 | 특정 실험과 무관한 환경 빌드 로그(2026-05-07) |

## 경로 링크 갱신 범위

이동으로 깨지는 markdown 링크는 다음 원칙으로 고쳤다(내용·결론 문구는 건드리지
않고 경로만):

1. `reports/historical_root/` 내부의 `.md`가 트리 **바깥**(`../PROJECT_STATUS.md`,
   `../results/...`, `../workspace/...` 등)을 가리키던 상대링크는 depth가 2단계
   늘어난 만큼(`reports/` → `deprecated/reports/historical_root/`) 일괄 보정했다.
   트리 **내부** 상호참조(`figure_index.md`, `figures/*.png` 등 형제 파일 bare
   참조)는 전체가 통째로 이동했으므로 그대로 유효하다.
2. `reports/quarantine_root/`(구 루트 `deprecated_reports/`) 내부에는 다른 파일을
   가리키는 상대링크가 원래 하나도 없었다(전부 `#anchor` 내부 목차 링크) — 고칠
   것이 없었다.
3. `reports/quarantine_engine_port/`(구 `workspace/engine-port/reports/deprecated_reports/`)는
   폴더가 옮겨오면서 깊이가 1단계 **얕아졌고 브랜치도 바뀌어서**(예:
   `workspace/engine-port/reports/CONSENSUS.md`를 가리키던 `../CONSENSUS.md`가
   더는 유효하지 않음), 실제로 깨지는 링크만 개별 확인해 고쳤다
   (`README.md`의 `CONSENSUS.md`/`sm_policy_report.html`/`research_arc.md` 참조,
   `session_handoff_2026-07-15.md`의 자기 폴더 내 참조).
4. 이 두 quarantine 폴더를 가리키던 **바깥쪽** 정본 문서(`CONSENSUS.md`,
   `workspace/engine-port/reports/system_vs_engine_vs_sim.md`,
   `realtrace_findings_and_open_branches.md`, 루트 `CLAUDE_CODE_SESSION_SUMMARY.md`,
   `reports/paper/DOCUMENT_STATUS.md`)의 링크도 새 경로로 갱신했다.
5. `reports/quarantine_engine_port/` 안의 나머지 상대링크(`../triage/notes.md`,
   `../results/cudagraph_probe/...`, `../../../reports/layer_aware_benefit_report.md` 등,
   주로 `p0_triage.md`·`r_series_status.md`·`zamba2_troubleshooting.md`·
   `session_handoff_2026-07-13.md`·`session_handoff_2026-07-15.md`에 있음)은
   **오늘 이동 전부터 이미 깨져 있던 링크**(2026-07-17에 이 파일들을
   `reports/`에서 `reports/deprecated_reports/`로 옮겼을 때 상대경로가 갱신되지
   않았던 잔재)라서 이번 이동으로 새로 생긴 문제가 아니다. 오늘은 건드리지
   않았다 — 필요하면 별도 정리 대상.

> **내부 경로 언급 caveat**: 위 세 격리 트리 안의 문서들이 본문에서 언급하는
> 상대경로 중 일부(특히 백틱 코드-스팬으로만 적힌 것, 실제 markdown 링크가
> 아닌 것)는 **재배치 이전 위치 기준**으로 쓰여 있다. 이런 prose 언급까지는
> 전부 쫓아 고치지 않았다(historical artifact의 당시 시점 기록을 보존하는
> 것이 우선). 클릭 가능한 markdown 링크(`[text](path)`)만 위 원칙대로
> 고쳤다.

## 이번에 건드리지 않은 것

`REPOSITORY_LAYOUT.md`의 "격리처: 루트 `deprecated/`" 절 참조. 정본 문서
(`PROJECT_STATUS.md`, `reports/paper/*`, `reports/CONSENSUS.md`,
`workspace/engine-port/RESUME.md`), 13개 engine-port campaign 결과, `triage/`, 참조되는
원자료(`results/{stage1,stage2,stage3,motivation,serving-eval}`), `slurm/`은 이미
정본 위계나 자체 README로 관리되고 있어 이번에도 제자리를 유지했다 — `deprecated/`로는
옮기지 않았다.

> **후속(같은 날, 2026-07-24)**: 위 목록의 `workspace/engine-port/reports/paper/*`·
> `CONSENSUS.md`는 이 통합 이후 **또 한 번** 이동했다 — `workspace/engine-port/reports/`
> 전체가 저장소 최상위 `reports/`로 승격됐다(`REPOSITORY_LAYOUT.md` "예외: engine-port
> `reports/`의 최상위 승격" 절). `deprecated/` 트리 자체와 그 안의 내용은 이 후속
> 이동의 영향을 받지 않았다 — 단 `deprecated/**`의 markdown 링크 중
> `workspace/engine-port/reports/...`를 가리키던 것들은 새 `reports/...` 경로로
> 재보정했다.

> **추가 후속(같은 날, 2026-07-24)**: `reports/quarantine_engine_port/`에 있던
> 세션 핸드오프 3건(`session_handoff_2026-07-07.md`, `-07-13.md`, `-07-15.md`)은
> **한 번 더** 이동해 전용 아카이브 [`../handoff-report/`](../handoff-report/)로
> 나갔다(다른 quarantine 문서와 분리 관리). 위 §"경로 링크 갱신 범위" 3번·5번에서
> 언급한 이 세 파일 관련 서술(자기 폴더 내 참조, 07-15의 부분 반증 주석 등)은
> **당시(quarantine_engine_port 안에 있던 시점) 기준**이며, 07-15의 주석 본문은
> `handoff-report/README.md`로 옮겨 보존했다.
