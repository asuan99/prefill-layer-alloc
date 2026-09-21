# 이양 후 운영 계획 — 로컬 개발 + 대여 GPU 실행 (2026-09-18)

사용자 결정(2026-09-18): **코드 작업은 로컬 PC에서, 실행만 대여 GPU 서버에서.** GPU 서버에서는 Claude Code를
쓰지 못할 수 있다. 이 문서는 그 전제에서 무엇을 고쳐야 하는지만 적는다. 연구 결론·게이트·Claim 등급은
이 문서와 무관하게 불변이며, 여기서 새 성능 판정을 하지 않는다.

짝 문서: `session_handoff_2026-09-17.md`(이양 체크포인트) · `migration_inventory_2026-09-17.md`(저장 대상·전송) ·
`migration_claude_tools_retarget_2026-09-17.md`(Claude 도구 줄 단위 목록).

## 0. 전송은 끝났다 (2026-09-18 사용자 측 확인)

로컬 PC `~/Experiments/KISTI/migration_2026-09-17`에서: 파일 **57개**(목록 55 + `MANIFEST.tsv` + `SHA256SUMS`),
`sha256sum -c --quiet` **무출력 = 전부 일치**, `git clone <bundle>` 성공(11,254 objects, HEAD `7c19668`),
로컬 전용 브랜치 `fix/gate14-tci-analyze` 복구 확인. ⇒ **번들 대상은 전량 이전 완료.** HF 모델 가중치는
사용자 결정으로 제외(리비전만 `D_misc/hf_hub_revisions.tsv`).

이 서버(KISTI Neuron)는 SLURM 계정 차단이 유지돼 실행이 불가하므로 **읽기 전용 보관소**로만 취급한다.
앞으로 커밋은 로컬 클론에서 한다(두 곳에서 커밋하면 갈라진다).

## 1. 새 구조

| 역할 | 위치 |
|---|---|
| 정본 저장소·커밋·문서·분석 | 로컬 PC 클론 |
| Claude Code(에이전트·스킬·메모리) | 로컬 PC |
| 실행(서빙 벤치·telemetry 생성) | 대여 GPU 서버 (ssh 접속 대상) |
| 원시 결과 보관 | 로컬(+사본). git에는 지금처럼 요약·판정서만 |

GPU 서버에 Claude Code를 설치하지 못해도 된다 — Claude는 로컬에서 돌면서 `ssh`/`rsync`로 원격 실행·로그 회수를
지시하면 된다. 서버에는 저장소 사본과 실행 환경만 있으면 된다.

## 2. GPU 선택 기준 (연구 결론과 직결되므로 대여 전에 확인)

- **아키텍처 제약**(dev tree `sglang/srt/multiplex/pdmux_context.py`의 `get_arch_constraints`, green context 분할 입도):
  - major 7(V100): 최소 2 SM·2 단위
  - **major 8(A100·L40S·RTX 40 계열): 최소 4 SM·2 단위** ← 지금까지의 모든 측정이 여기(A100, 108 SM)
  - **major 9(H100/H200): 최소 8 SM·8 단위** — 총 SM도 132로 달라 격자가 통째로 바뀐다
  - **major 10 이상(Blackwell: B200·RTX 50 계열): `ValueError: Unsupported compute capability`** ⇒ 코드 확장 없이는 못 돌린다
- **드라이버**: green context 생성에 CUDA 드라이버 ≥ 12.4 필요(`sgl_kernel/spatial.py`의 실패 메시지 기준).
- **GPU 전체가 필요**: MIG/vGPU로 쪼개 주는 인스턴스는 안 된다.
- **비교 가능성**: **A100 80GB 1장**이면 기존 격자(D16/44/92/108)·λ\*와 같은 기판이라 직접 비교가 가능하다.
  H100이면 8 SM 단위·132 SM이라 **격자 재설계 + λ0 재측정**이 선행돼야 하고, E2 사전등록도 그대로는 성립하지
  않는다(성능 판정이 아니라 **미결 상태** — 사용자 결정 + 규칙층 감사 사항).
- **디스크**: 모델 1개(Nano-9B-v2 ≈ 17 GB) + 실행당 telemetry 수백 MB–수 GB ⇒ 100 GB 이상 여유 권장.

## 3. 이식 작업 (비용이 작은 순서)

1. **스케줄러 제거는 대부분 이미 돼 있다.** 활성 스크립트가 SLURM 변수에 폴백을 갖고 있다:
   `scripts/r2_eval/r2_eval.sbatch:55` `run_index="${SLURM_ARRAY_TASK_ID:-${PDMUX_RUN_INDEX:-0}}"`,
   같은 파일 `:57` `job_${SLURM_JOB_ID:-local}`, `results/r2_eval/e2_sticky_prereg/e2_sticky.sbatch:54`
   `JOBID="${SLURM_JOB_ID:-local$$}"`. `#SBATCH` 줄은 `bash`로 실행하면 주석이다(`r2_eval.sbatch:27`이 이 실행
   경로를 이미 상정하고 있다). ⇒ **배열 작업 = `PDMUX_RUN_INDEX=i bash <script>` 루프**. 새 러너가 할 일은
   루프·로그 리다이렉트·종료코드 수집·아티팩트 경로 고정뿐이고, `results/<campaign>/` 레이아웃과 telemetry
   스키마를 그대로 두면 기존 분석·판정 코드를 손대지 않아도 된다.
2. **절대경로 치환 — 활성 경로만 하면 된다**: `scripts/bootstrap/install_engine.sh`(2) ·
   `scripts/bootstrap/sync_engine_tree.sh`(1, 이미 `SGLANG_ENGINE_DEV`로 덮어쓸 수 있음) ·
   `scripts/r2_eval/engine_bench_runner.sh`(2) · `scripts/r2_eval/r2_eval.sbatch`(2) ·
   `results/r2_eval/e2_sticky_prereg/e2_sticky.sbatch`(4, `#SBATCH --output/--error` 포함) · 테스트 15줄.
   루트 환경변수 하나(`PDMUX_ROOT`)로 정리하면 끝난다. `results/` 아래 과거 캠페인 5,473줄은 **재실행할 것만** 고친다.
3. **서버 부트스트랩을 한 명령으로**: torch/CUDA 스택 → SGLang v0.5.10 소스 → `sync_engine_tree.sh` →
   `env/devtree_manual_edits.patch` → 모델·trace 내려받기 → CPU 회귀
   (★정정 2026-09-21, doc-steward: "140 tests"는 이전 머신[KISTI Neuron] 값 —
   로컬 PC 정본 기준선은 598 tests, `env/cpu_regression_baseline_2026-09-18.md`) → 부팅 스모크.
   대여 인스턴스는 매번 새로 뜨므로 **컨테이너 이미지로 굳히는 편이 안전**하다(패키지 버전은
   `env/venv_packages_2026-09-17.txt`; py3.14 shim은 `env/sitecustomize.py`).
4. **결과 회수 규약**: 실행 뒤 `rsync`로 `results/<campaign>/`를 로컬로 내리고 sha256을 남긴다. 서버는 언제든
   반납되므로 **실행 종료 = 회수 완료**까지를 한 작업으로 본다.
5. **Claude 도구 재작성**: `experiment-runner`를 "SLURM 제출 운영"에서 "**ssh 원격 실행·로그 triage·아티팩트 회수**"로
   바꾼다(`--comment`·파티션·`showappl`·`squeue` 규약 삭제). 나머지 대상·줄 수는
   `migration_claude_tools_retarget_2026-09-17.md` 표 그대로. `catch-up`은 `squeue` 대신 "원격 실행 중인 프로세스
   확인" 또는 "회수된 결과 확인"으로 바꾼다.
6. **예산 규율**: 시간당 과금이므로 기존 OVERRIDE(등록 예산 GPU-시간)를 그대로 쓰되 단가·한도를 새로 적는다.
   지금까지 누적 ≈462 GPU-시간이라는 사실이 대여 비용 추정의 출발점이다.

## 4. 로컬에서의 순서

1. 작업 클론 자리를 정하고(예: `~/Experiments/KISTI/prefill-layer-alloc`) origin을 GitHub로 지정. 검증용
   `repo-test`는 지우거나 그 클론을 작업본으로 승격(origin이 번들 파일을 가리키므로 `git remote set-url` 필요).
2. `tools/claude/sync_claude_tools.sh --install`로 `.claude` 도구 설치, 메모리 디렉터리를 새 작업 루트의 slug로 복사.
3. 모델·trace 확보(`D_misc/hf_hub_revisions.tsv`, `hf_cache_raw_traces.tar.zst`).
4. GPU 대여 전 §2 확인(A100이면 비교 가능 / H100이면 재측정 전제 / Blackwell이면 코드 확장 필요).
5. 부트스트랩·러너 작성 → CPU 회귀 + 부팅 스모크로 기판 검증.
6. E2·λ0을 그 기판에서 다시 등록할지 판단(사용자 결정 + 규칙층 감사).

## 5. 바뀌지 않는 것

HE0·layer-type 死·정책 순위·Claim D/E 등급·방법론 게이트는 실행 환경과 무관하게 그대로다. 기판이 바뀌어
재측정이 필요해지는 항목(λ\*, E2 등록 격자)은 **결론이 아니라 미결 항목**으로 남는다.
