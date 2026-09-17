<!-- generated 2026-09-17 by tools/migration/scan_machine_refs.py (re-run on the new machine with --live <new root>) -->

# Claude 도구 머신 의존 참조 체크리스트

카테고리별 조치:

| 카테고리 | 조치 |
|---|---|
| 경로 | 새 작업 루트·홈 경로로 치환(가능하면 환경변수/상대경로화) |
| 메모리 경로 | 새 작업 루트의 slug(`/`→`-`)로 치환, 메모리 디렉터리도 그 위치로 이동 |
| 스케줄러 | 새 클러스터 스케줄러·파티션·제출 정책(`--comment` 등)에 맞게 수정, 스케줄러가 없으면 실행 규약 자체 재작성 |
| 모듈·환경 | 새 환경의 Python/CUDA/torch 스택과 venv·dev tree 경로로 수정 |
| 하드웨어·기관 | 새 GPU(모델·SM 수·인터커넥트)와 기관 정보로 수정 - 실험 기판 동일성 판단과 연결 |
| git 원격·자격증명 | 새 머신 자격증명 방식(토큰 평문 URL 여부)에 맞게 push/원격 규칙 재검토 |

카테고리별 줄 수(한 줄이 여러 카테고리에 걸칠 수 있음): 경로 30, 메모리 경로 3, 스케줄러 36, 모듈·환경 15, 하드웨어·기관 8, git 원격·자격증명 9

## `CLAUDE.md` — 21줄

| 줄 | 카테고리 | 내용 |
|---:|---|---|
| 5 | 하드웨어·기관 | A100 green-context SM 분할**로 서빙하며, prefill↔decode multiplexing(PD-mux)의 |
| 40 | 경로, 모듈·환경 | - editable tree: `/scratch/ehmoon/whlee/sglang_engine_dev/python` · venv: |
| 41 | 경로, 모듈·환경, 하드웨어·기관 | `/scratch/ehmoon/whlee/sglang_engine_venv` · A100(108 SM), CUDA 13, torch 2.9.1. |
| 44 | 모듈·환경 | module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0 |
| 45 | 경로, 모듈·환경 | source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate |
| 46 | 경로 | cd /scratch/ehmoon/whlee/prefill-layer-alloc |
| 47 | 모듈·환경 | workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh   # PD-mux src + thread-local role patch 설치 + SHA-256 manifest |
| 50 | 스케줄러 | - SLURM: partition `amd_a100nv_8`, `--gres=gpu:1`. R2 캠페인은 |
| 51 | 스케줄러 | `PDMUX_CAMPAIGN=.../results/r2_eval/campaign.json sbatch --array=0-<N-1> .../scripts/r2_eval/r2_eval.sbatch`. |
| 52 | 스케줄러, 하드웨어·기관 | - **SLURM `--comment` 필수 (2026-08-12 18:00 KST 이후 뉴론 정책 — 없으면 제출 거부).** |
| 53 | 스케줄러 | 모든 job script에 아래 한 줄을 **`#SBATCH` 블록 안**(= 첫 실행 라인보다 위 — Slurm은 |
| 56 | 스케줄러 | #SBATCH --comment="field=efficientai;appl=pytorch" |
| 59 | 스케줄러, 하드웨어·기관 | SM 분할 서빙 효율 연구) · **`appl=pytorch`**(SGLang은 showappl 목록에 없음; PyTorch |
| 61 | 스케줄러 | `salloc --partition=amd_a100nv_8 --gres=gpu:1 --comment="field=efficientai;appl=pytorch"`. |
| 62 | 스케줄러 | 허용값은 로그인 노드 `showappl`로 확인. CLI에서 `--comment`를 덮어쓰지 말 것 |
| 63 | 스케줄러 | (script directive를 무효화해 거부된다). 2026-08-14 기준 저장소 내 `.sbatch` 97개 |
| 68 | 스케줄러 | - 루트의 SLURM `.out`/`.err`는 커밋하지 않는다. 결과는 `results/<campaign>/`에만 기록. |
| 87 | 스케줄러 | - **experiment-runner** — sbatch 제출·squeue 모니터·로그 triage·아티팩트 수집·완료 job on-demand 보고서 생성(옛 slurm-agent observer 대체). |
| 91 | git 원격·자격증명 | - **git-committer** — 스테이징 검증·관례적 커밋 메시지·로컬 커밋(명시 지시 없이 push 금지). |
| 98 | 경로 | > 워크스페이스(`/scratch/ehmoon/whlee`, **git 저장소 아님** — `.git`이 빈 디렉터리)에 살지만, |
| 115 | 경로, 메모리 경로 | 장기 메모리 인덱스: `/home01/ehmoon/.claude/projects/-scratch-ehmoon-whlee/memory/MEMORY.md`. |

## `.claude/agents/claims-auditor.md` — 참조 0


## `.claude/agents/doc-steward.md` — 1줄

| 줄 | 카테고리 | 내용 |
|---:|---|---|
| 48 | 경로, 메모리 경로 | `/home01/ehmoon/.claude/projects/-scratch-ehmoon-whlee/memory/MEMORY.md` = 세션 간 |

## `.claude/agents/engine-porter.md` — 4줄

| 줄 | 카테고리 | 내용 |
|---:|---|---|
| 3 | 경로, 모듈·환경 | description: SGLang v0.5.10 엔진 내부(PD-mux) 전문. event-loop/green-context/thread-local role/split-prefill 패치 구현·검증, PDMUX_* 런타임 모드 배선, 패치 SHA-256 manifest 유지, 성능 주… |
| 9 | 하드웨어·기관 | SM 분할과 PD-mux 런타임을 실제 엔진에 배선한다. **철칙: 구현 완료 ≠ 성능 주장 성립.** |
| 14 | 경로, 모듈·환경 | - editable tree: `/scratch/ehmoon/whlee/sglang_engine_dev/python` (SGLang v0.5.10 기반). |
| 19 | 모듈·환경 | - 설치: `workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh` — src를 dev-tree에 |

## `.claude/agents/experiment-runner.md` — 31줄

| 줄 | 카테고리 | 내용 |
|---:|---|---|
| 3 | 스케줄러, 모듈·환경 | description: engine-port 서빙 실험의 SLURM 운영 담당. sbatch 작성/제출, squeue·sacct 모니터, .out/.err/telemetry triage, boot/OOM/CUDA-graph/triton-cache 실패 시그니처 탐지, results/<c… |
| 8 | 스케줄러 | 너는 이 프로젝트(`prefill-layer-alloc`)의 **SLURM 실험 운영자**다. 제출·모니터·로그 |
| 15 | 모듈·환경 | module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0 |
| 16 | 경로, 모듈·환경 | source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate |
| 17 | 경로 | cd /scratch/ehmoon/whlee/prefill-layer-alloc |
| 19 | 경로, 모듈·환경, 하드웨어·기관 | - editable tree: `/scratch/ehmoon/whlee/sglang_engine_dev/python`. A100 108 SM, CUDA 13, torch 2.9.1. |
| 20 | 모듈·환경 | - **실행 전 항상** `workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh <manifest.sha256>` — |
| 27 | 스케줄러 | - 파티션 `amd_a100nv_8`, `--gres=gpu:1`, `--cpus-per-task=16`, `--mem=64G`. |
| 28 | 스케줄러 | - **`--comment` 필수 — 없으면 스케줄러가 제출 자체를 거부한다** (2026-08-12 18:00 KST |
| 29 | 하드웨어·기관 | 이후 뉴론 정책). 이 프로젝트 고정값: |
| 31 | 스케줄러 | #SBATCH --comment="field=efficientai;appl=pytorch" |
| 33 | 하드웨어·기관 | - `field=efficientai` = Efficient & Scalable AI Systems (PD-mux SM 분할 서빙 효율). |
| 34 | 스케줄러 | `appl=pytorch` = SGLang이 showappl 목록에 없어 PyTorch 기반으로 신고. **`vllm`으로 |
| 35 | 스케줄러 | 바꾸지 마라**(다른 엔진). 허용값 확인은 로그인 노드에서 `showappl`. |
| 36 | 스케줄러 | - **배치 위치가 중요하다**: Slurm은 첫 비주석·비공백 라인 이후의 `#SBATCH`를 무시한다. |
| 39 | 스케줄러 | - **CLI에서 `--comment`를 덮어쓰지 마라.** `sbatch --comment=pytorch ...` 같은 구형식 |
| 40 | 스케줄러 | 인자는 script directive를 무효화해 거부된다. driver `.sh`가 `sbatch`를 감싸는 경우도 |
| 41 | 스케줄러 | `--comment`를 넘기지 않는 게 정상(각 `.sbatch`가 이미 들고 있다). |
| 42 | 스케줄러 | - 인터랙티브: `salloc --partition=amd_a100nv_8 --gres=gpu:1 --comment="field=efficientai;appl=pytorch"`. |
| 43 | 스케줄러 | - **제출 전 게이트 (신규 `.sbatch`이거나 헤더를 손댔으면 반드시)** — grep 말고 위치까지 |
| 49 | 스케줄러 | sbatch --test-only <script>   # 스케줄러가 실제로 수락하는지 최종 확인 |
| 51 | 스케줄러 | 대형 캠페인이면 방법론 게이트 #26(배관 스모크)과 함께 수행한다 — `--comment` 누락은 |
| 58 | 스케줄러 | sbatch --array=0-<N-1> workspace/engine-port/scripts/r2_eval/r2_eval.sbatch |
| 76 | 스케줄러 | - `squeue -u $USER`, `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode`. |
| 82 | 스케줄러 | - **제출 거부(`--comment`)** → job id가 안 나오고 `.out`/`.err`도 안 생긴다. sacct에도 |
| 85 | 스케줄러 | - **OOM** → `CUDA out of memory` / sacct MaxRSS 근접 / State OUT_OF_MEMORY. mem·KV·batch 확인. |
| 92 | 모듈·환경 | - **thread-local role 미적용** → true dual이 fail-fast. `sync_engine_tree.sh`로 패치 재설치. |
| 96 | 스케줄러 | 옛 `workspace/slurm-agent/` observer(상시 systemd 데몬이 사용자 전체 job을 무필터로 |
| 104 | 스케줄러 | "프로젝트 실험 아님"으로 보고서 생성하지 않는다(무관 sacct job을 긁지 마라). |
| 105 | 모듈·환경 | - **내용**: job id·State·Elapsed·ExitCode, `sync_engine_tree.sh` manifest(source hash), |
| 110 | 스케줄러 | 상태 캐시(`.slurm-agent/` 같은 별도 registry) 만들지 말 것. |

## `.claude/agents/git-committer.md` — 10줄

| 줄 | 카테고리 | 내용 |
|---:|---|---|
| 3 | 스케줄러, git 원격·자격증명 | description: git 커밋 전담 에이전트. 스테이징 검토 → 관례적(conventional) 커밋 메시지 작성 → 로컬 커밋을 규율대로 수행. "커밋해", "commit", "변경사항 정리해서 커밋" 할 때 사용. **명시적 지시 없이는 절대 push하지 않는다**(이 프로젝트… |
| 14 | git 원격·자격증명 | 명시하지 않으면 **절대 `git push` 하지 마라.** (과거 한 subagent가 요청 없이 커밋+push해 |
| 16 | git 원격·자격증명 | 2. **원격 URL 토큰 주의.** `prefill-layer-alloc` origin URL에 GitHub PAT(`ghp_…`)이 평문으로 |
| 17 | git 원격·자격증명 | 박혀 있다. push를 지시받으면 **먼저 그 사실을 경고**하고 진행 여부를 확인하라. 토큰 값을 |
| 25 | 경로 | - inner: `/scratch/ehmoon/whlee/prefill-layer-alloc` — 연구/논문 repo(`.git` 있음, 원격 있음). |
| 27 | 경로 | - outer: `/scratch/ehmoon/whlee` — **git 저장소가 아니다.** `.git`이 **빈 디렉터리**라 |
| 28 | 경로 | `git -C /scratch/ehmoon/whlee …`는 `fatal: not a git repository`로 죽는다. 이전 판본이 |
| 52 | 스케줄러, git 원격·자격증명 | - **커밋하면 안 되는 것**: 루트 SLURM `.out`/`.err`(CLAUDE.md 규칙), 토큰/키/시크릿, |
| 77 | git 원격·자격증명 | - push/force-push/rebase/tag/원격 조작은 **명시적 지시 없이 금지**. |
| 84 | git 원격·자격증명 | 안 담긴 항목 명시), 브랜치, **push 여부(기본 "push 안 함")**. 정직하게. |

## `.claude/agents/result-analyst.md` — 참조 0


## `.claude/agents/venue-strategist.md` — 참조 0


## `.claude/skills/catch-up/SKILL.md` — 6줄

| 줄 | 카테고리 | 내용 |
|---:|---|---|
| 30 | 경로 | git -C /scratch/ehmoon/whlee log --oneline -12 |
| 31 | 경로 | git -C /scratch/ehmoon/whlee/prefill-layer-alloc log --oneline -12 |
| 32 | 경로 | git -C /scratch/ehmoon/whlee/prefill-layer-alloc status --short   # 미커밋 변경 |
| 34 | 경로 | ls -t /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/*/ \| head |
| 35 | 경로 | ls -t /scratch/ehmoon/whlee/prefill-layer-alloc/*.out 2>/dev/null \| head |
| 37 | 스케줄러 | squeue -u "$USER" 2>/dev/null |

## `.claude/skills/handoff/SKILL.md` — 6줄

| 줄 | 카테고리 | 내용 |
|---:|---|---|
| 14 | 경로 | git -C /scratch/ehmoon/whlee/prefill-layer-alloc log --oneline -15 |
| 15 | 경로 | git -C /scratch/ehmoon/whlee/prefill-layer-alloc status --short          # 미커밋 |
| 16 | 경로 | git -C /scratch/ehmoon/whlee/prefill-layer-alloc diff --stat             # 무엇이 바뀜 |
| 17 | 경로 | ls -t /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/*/ \| head |
| 43 | 경로, 메모리 경로 | `/home01/ehmoon/.claude/projects/-scratch-ehmoon-whlee/memory/`. |
| 59 | git 원격·자격증명 | - ★★**push는 하지 않는다.** 묻지도, 보고하지도 마라(원격 URL에 토큰이 평문 노출돼 있다). |

## `.claude/skills/review-lead/SKILL.md` — 참조 0


## `prefill-layer-alloc/tools/claude/sync_claude_tools.sh` — 2줄

| 줄 | 카테고리 | 내용 |
|---:|---|---|
| 9 | 경로 | #   /scratch/ehmoon/whlee, which is NOT a git repository (its .git is an empty |
| 38 | 경로 | live_root="${CLAUDE_TOOLS_LIVE:-/scratch/ehmoon/whlee}" |

## `prefill-layer-alloc/tools/claude/test_sync_claude_tools.sh` — 1줄

| 줄 | 카테고리 | 내용 |
|---:|---|---|
| 13 | 경로 | live_src="${CLAUDE_TOOLS_LIVE:-/scratch/ehmoon/whlee}" |

## `prefill-layer-alloc/tools/claude/README.md` — 2줄

| 줄 | 카테고리 | 내용 |
|---:|---|---|
| 8 | 경로 | `/scratch/ehmoon/whlee`는 **git 저장소가 아니다**(`.git`이 빈 디렉터리). 따라서 |
| 21 | 경로 | \| live 사본 \| `/scratch/ehmoon/whlee/{CLAUDE.md,.claude/{agents,skills}}` \| Claude Code가 실제로 읽는 것 \| |

## 참고: 저장소 전체의 절대경로 참조 (`git grep`, 줄 수)

| 디렉터리 | 줄 수 |
|---|---:|
| `workspace/engine-port/results` | 5473 |
| `deprecated` | 254 |
| `results` | 127 |
| `workspace/engine-port/triage` | 73 |
| `reports` | 32 |
| `tools` | 31 |
| `workspace/characterization/slurm` | 27 |
| `workspace/engine-port/tests` | 13 |
| `handoff-report` | 12 |
| `workspace/engine-port/scripts` | 11 |
| `PROJECT_STATUS.md` | 7 |
| `slurm` | 7 |
| `workspace/engine-port/reports` | 6 |
| `workspace/serving-eval/slurm` | 6 |
| `workspace/engine-port/RESUME.md` | 3 |

파일 확장자별 파일 수: `.log` 309, `.json` 222, `.sbatch` 131, `.sha256` 106, `.py` 102, `.md` 89, `.txt` 84, `.sh` 51
