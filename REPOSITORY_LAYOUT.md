# Repository layout

이 저장소는 실험 트랙을 유지하면서 산출물의 목적을 분리한다. 새 파일은
가능하면 해당 트랙 아래의 canonical 경로에 둔다.

| 목적 | canonical 위치 | 포함하는 것 |
|---|---|---|
| 진행·설계·판정 보고서 | `workspace/<track>/reports/` — **단, engine-port는 예외**(아래 참조) | 연구 진행, 설계, 해석, 실패 원인, 인수인계 |
| 실험 원자료·실행 결과 | `workspace/<track>/results/<campaign>/` | CSV/JSON, SLURM 로그, manifest, job별 산출물 |
| 재적용 가능한 소스 | `workspace/<track>/src/` | 엔진 패치 미러, 모델·연산·공통 라이브러리 |
| 단위 테스트 | `workspace/<track>/tests/` | CPU-only 및 회귀 테스트 |
| 실행·제출 스크립트 | `workspace/<track>/scripts/` | 환경 bootstrap, `sbatch`, 분석 실행 드라이버 |

현재 트랙은 다음과 같다.

- `workspace/characterization/`: 커널·SM 특성화와 시뮬레이터(자체 `reports/v2_report.md` 보유 — 아래 예외에 포함되지 않음)
- `workspace/engine-port/`: SGLang 포팅과 실제 서빙 검증(`reports/`는 최상위로 승격, 아래 참조)
- `workspace/serving-eval/`: serving 평가 하네스
- `workspace/shared/`: 트랙 공통 설정

## 예외: engine-port `reports/`의 최상위 승격 (2026-07-24)

프로젝트의 논문/연구 정본이 사실상 engine-port 트랙 하나에 집중돼 있어(다른
트랙은 characterization/simulator처럼 이미 종결됐거나 하네스일 뿐), "트랙별
`workspace/<track>/reports/`" 규약의 **명시적 예외**로 `workspace/engine-port/reports/`
전체를 `git mv`로 저장소 최상위 `reports/`로 승격했다(`paper/`·`r1_dual_worker/`·
`CONSENSUS.md`·모든 `*.md`/`*.png`/`plot_*.py`/`sm_policy_report.html` 포함, 내용
불변·경로만 이동). **다른 트랙(characterization/serving-eval/shared)의 `reports/`는
이 예외에 해당하지 않으며 계속 `workspace/<track>/reports/` 그대로다** — 이 문서를
읽고 "모든 reports/가 최상위에 있다"고 오인하지 말 것.

engine-port의 나머지 산출물(`results/<campaign>/`, `scripts/`, `src/`, `benchmarks/`,
`tests/`, `triage/`, `env/`, `external/`)은 **트랙 밑에 그대로 남아 있다** — 옮긴
것은 `reports/`뿐이다.

## engine-port canonical layout (2026-07-24 갱신)

논문 상태, 구현, benchmark와 raw artifact를 다음처럼 분리한다.

```text
reports/                            # 2026-07-24부터 최상위(예외, 위 참조)
├── paper/                          # current claim/evidence/roadmap
├── r1_dual_worker/                 # historical job reports
├── CONSENSUS.md, research_arc.md, per_layer_type_postmortem.md, ...
└── *.png, plot_*.py, sm_policy_report.html

workspace/engine-port/
├── src/multiplex/                 # dual runtime, profile, controller, telemetry
├── benchmarks/pdmux_eval/        # workload/campaign/statistics tools
├── tests/                         # concurrency/policy/analysis regression
├── scripts/
│   ├── bootstrap/sync_engine_tree.sh
│   ├── r1_dual_worker/            # historical observer launchers
│   └── r2_eval/                   # paired campaign wrappers
├── results/r1_dual_worker/        # historical raw artifacts
└── results/r2_eval/               # immutable R2 traces/manifests/results
```

프로젝트 전체 현재 상태는 루트 `PROJECT_STATUS.md`가 유일한 정본이다.

`triage/`와 기존 `results/<campaign>/*.sbatch`는 과거 P-series/R-series
재현 경로이므로 이번 정리에서 일괄 이동하지 않았다. 새 실험은 `scripts/`에
제출 스크립트를 두고, 결과는 반드시 해당 campaign의 `results/` 아래에
기록한다. 루트의 SLURM `.out`/`.err` 파일은 커밋하지 않는다.

## 격리처: 루트 `deprecated/` (2026-07-24 신설)

top-level 잡동사니를 줄이기 위해 historical/superseded 산출물의 격리처를
저장소 전체에서 하나로 통합했다.

```text
deprecated/
├── README.md                         # 무엇이 어디서 왔는지, 되돌리는 법
├── reports/
│   ├── historical_root/              # 구 루트 reports/ 전체(figures/·figures_v3/·figures_v4/ 포함)
│   ├── quarantine_root/              # 구 루트 deprecated_reports/ (14 files)
│   └── quarantine_engine_port/       # 구 workspace/engine-port/reports/deprecated_reports/ (12 files)
├── results/
│   └── arxiv/                        # 구 results/arxiv/ — 무참조 3-way 중복 stage1 번들
└── logs/                              # 루트에 떨어졌던 tracked/untracked SLURM 로그·빌드 로그
```

이동은 전부 `git mv`(tracked) 또는 plain `mv`(untracked)이며 내용은 편집하지
않았다 — 위치만 옮겼다. 반대로 아래는 **그대로 제자리**를 유지한다(각자 이미
정본 위계·자체 README·CONSENSUS §4/`reports/paper/DOCUMENT_STATUS.md`로 관리되고
있어 추가 이동이 필요 없다고 판단):

- `reports/paper/*`, `reports/CONSENSUS.md`, `workspace/engine-port/RESUME.md`, 그리고
  `CONSENSUS.md` §4 "살아있는 문서" 표에 등재된 문서 전체(2026-07-24 같은 날
  `workspace/engine-port/reports/` → 최상위 `reports/`로 한 번 더 승격됐다 — 위
  "예외: engine-port `reports/`의 최상위 승격" 절 참조. `deprecated/`로는 이동하지
  않았다).
- `workspace/engine-port/results/<campaign>/` 13종 (raw artifact, immutable).
- `workspace/engine-port/triage/`(과거 정리에서도 스코프 제외).
- `results/{stage1,stage2,stage3,motivation,serving-eval}`, `slurm/`(참조되는
  원자료·고유 재현 스크립트).
- `workspace/{characterization,serving-eval,shared}`(활성 소스; `characterization`은
  자체 `archive/v1_7b_saturation/ARCHIVE_NOTE.md`로 이미 v1→v2 이관을 문서화함).

`reports/paper/DOCUMENT_STATUS.md`가 이 이관의 문서별 판정·경로를 관리한다.

## `handoff-report/` — 세션 핸드오프 전용 아카이브 (2026-07-24 신설)

`reports/`(현재 연구 정본)와는 **분리**해서 세션 간 인수인계(handoff) 기록만
최상위 `handoff-report/`에 모은다. 핸드오프는 point-in-time 세션 기록이지
갱신되는 연구 결론이 아니므로, 별도 관리가 관리 편의상 더 낫다고 판단했다.

```text
handoff-report/
├── README.md                         # 목록 + 각 핸드오프 현재 상태(반증 여부 등)
├── session_handoff_2026-07-07.md
├── session_handoff_2026-07-13.md
├── session_handoff_2026-07-15.md     # 부분 반증됨(README에 주석 보존)
└── session_handoff_2026-07-24.md
```

기존 세션 핸드오프 3건은 `deprecated/reports/quarantine_engine_port/`에서,
1건(07-24)은 다른 세션이 옛(승격 전) 스킬 경로 `workspace/engine-port/reports/`에
잘못 저장해 되살아난 것을 여기로 옮겼다(옮긴 뒤 그 디렉터리는 다시 비워
"예외: engine-port `reports/`의 최상위 승격" 상태로 되돌아갔다). 2026-07-24부터
세션 핸드오프 스킬은 이 경로에 저장한다.
