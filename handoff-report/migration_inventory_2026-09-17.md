# 이양 인벤토리 — `/scratch/ehmoon/whlee` 전체의 git 밖 파일 (2026-09-17)

> 짝 문서: `session_handoff_2026-09-17.md`(이양 체크포인트·복원 절차),
> `migration_claude_tools_retarget_2026-09-17.md`(Claude 도구 머신 의존 참조 줄 단위 목록).
> 생성 도구: `tools/migration/inventory_workspace.py`(전수 목록) ·
> `tools/migration/pack_migration_bundle.sh`(묶기·검증) · `tools/migration/scan_machine_refs.py`.

## 1. 방법

- `/scratch/ehmoon/whlee` 아래 **모든 파일**을 걸어(`os.scandir`, 심볼릭 링크는 따라가지 않음)
  `prefill-layer-alloc`·`SSM-Scope`의 `git ls-files`에 있는 파일(6,489개, 0.73 GiB)을 뺐다.
- 남은 파일을 규칙(첫 일치)으로 분류했다. **미분류(UNCLASSIFIED) 0개.**
- 파일 단위 전체 목록: 번들 `INVENTORY/untracked_files.tsv.gz`(경로·바이트·규칙·결정·번들 위치).
- 교차검증: 인벤토리의 `pla_results_untracked` 경로 집합 = 결과 아카이브 35개를 풀어 얻은
  경로 집합(11,162개, 차이 0). `pla_other_untracked`+`external_zenodo_and_logs` 집합과
  `D_misc/repo_untracked_misc` 사이 차이는 **이 인벤토리 작업이 새로 만든 3파일뿐**이다
  (이 문서와 두 도구 — 같은 커밋으로 git에 들어간다).

## 2. 전체 분류표 (git 밖 파일만)

| 규칙 | 파일 수 | 원본 크기 | 결정 | 번들 위치 |
|---|---:|---:|---|---|
| `pla_results_untracked` — `workspace/engine-port/results/**` 원시 telemetry·로그 | 11,162 | 41.86 GiB | **필수** | `C_results_untracked/` (압축 1.72 GB) |
| `pla_git_dir` — `prefill-layer-alloc/.git` | 710 | 0.09 GiB | **필수** | `A_git/prefill-layer-alloc.bundle` (75 MB, `.git/config` 제외) |
| `hf_raw_traces` — `hf_cache/raw/**` 벤치 입력 trace | 9 | 0.74 GiB | **필수** | `D_misc/hf_cache_raw_traces.tar.zst` (167 MB) |
| `pla_other_untracked` — 루트 `logs/`·`slurm/logs/`·루트 `.out/.err/.jsonl`·`reports/figures/*.png/pdf`·`deprecated/logs` 등 | 740 | 0.03 GiB | **필수** | `D_misc/repo_untracked_misc.tar.zst` (27 MB, 아래 줄과 합본) |
| `claude_workspace_tools` — `CLAUDE.md`·`.claude/agents`·`.claude/skills` | 11 | <1 MB | **필수** | `B_claude/claude_memory_tools.tar.gz` (git `tools/claude/`에도 사본) |
| `external_zenodo_and_logs` — `external/muxwise-zenodo{,.zip}`·clone 로그 | 2,149 | 0.03 GiB | 권장 | `D_misc/repo_untracked_misc.tar.zst` |
| `hf_converted_configs` — `hf_cache/mamba2-2.7b-sglang`·`mamba-codestral-7b-sglang` | 17 | 3.6 MB | 권장 | `D_misc/hf_converted_model_dirs.tar.gz` (가중치는 hub 링크로만 기록) |
| `engine_dev_tree` — `sglang_engine_dev/` | 2,269 | 0.02 GiB | 권장 | `E_env/sglang_engine_dev_src.tar.zst` |
| `engine_venv_config` — venv `pyvenv.cfg`·`sitecustomize.py` | 2 | <1 MB | 권장 | `E_env/` (+ git `env/sitecustomize.py`) |
| `vllm_bench_scripts` — `vllm_bench/` | 3 | <1 MB | 권장 | `D_misc/vllm_bench.tar.gz` |
| `vscode_settings` — `.vscode/settings.json` | 1 | <1 MB | 선택 | `D_misc/workspace_dotfiles.tar.gz` |
| `hf_hub_models` — `hf_cache/hub/models--*` | 312 | **148.52 GiB** | 선택 (재다운로드 권장) | 기본 미포함 — §4, 필요 시 `hf_models` 단계 |
| `hf_hub_datasets` — `hf_cache/hub/datasets--THUDM--LongBench` | 5 | <1 MB | 선택 | 미포함(재다운로드; 실제 데이터는 `hf_cache/raw/longbench_cache`에 저장됨) |
| `vllm_venv_and_pip_cache` — `vllm_venv/`·`.pip_cache/` | 60,017 | 9.57 GiB | 불필요 | 재설치 |
| `pla_stray_venvs` — 저장소 루트 `bin/lib/include/pyvenv.cfg`(cuda12 pip-only venv)·`serving-eval/.venv` | 23,470 | 3.26 GiB | 불필요 | 재설치 |
| `build_caches` — triton 캐시·`__pycache__`·`.pytest_cache` | 70,446 | 1.80 GiB | 불필요 | 재생성 |
| `engine_venv` — `sglang_engine_venv/` 나머지 | 3,744 | 1.28 GiB | 불필요 | 패키지 목록 git `env/venv_packages_2026-09-17.txt` (purge로 이미 손상) |
| `hf_datasets_cache` — `hf_cache/{json,anon…share_gpt…,modules,xet}` | 17 | 1.21 GiB | 불필요 | HF datasets 캐시, 재다운로드 |
| `external_git_clones` — `external/{sglang-latest,muxwise,bullet}` | 11,118 | 0.46 GiB | 불필요 | 커밋 고정 재클론(`E_env/external_clone_pins.tsv`) |
| `vsix_installer` — `Microsoft.VisualStudio.Services.VSIXPackage` | 1 | 0.08 GiB | 불필요 | — |
| `ssm_scope_git_dir` — `SSM-Scope/.git` | 13 | 0.04 GiB | 불필요 | 별도 저장소, 원격과 차이 0·미추적 파일 0 |
| `hf_hub_other`·`hf_cache_meta` — HF 잠금·태그·메타 | 80 | <1 MB | 불필요 | — |

- 빈 디렉터리 `.agents/`·`.codex/`·`.slurm-agent/`·`.claude/worktrees/`, 바깥 `.git/`(빈 디렉터리)는 파일이 없다.
- **외부 클론 3개는 로컬 수정이 없다**: `git diff HEAD --diff-filter=M` 0건, 미추적 파일은 전부 purge가 이름을 바꾼
  `ToBeDelete_*` 또는 HEAD에 같은 경로가 있는 파일(=purge가 인덱스에서 밀어낸 원본)이다.

### whlee 밖이지만 함께 저장한 것

| 경로 | 결정 | 번들 위치 |
|---|---|---|
| `~/.claude/projects/-scratch-ehmoon-whlee/memory/` (메모리 15파일) | **필수** | `B_claude/claude_memory_tools.tar.gz` |
| `~/.claude/{settings.json,plans,skills}` | 권장 | 같은 아카이브 |
| `~/.claude/projects/-scratch-ehmoon-whlee/*.jsonl` 등 세션 기록 | 선택 | `B_claude/claude_transcripts.tar.zst` (100 MB) |
| `~/.claude/.credentials.json` | **저장 금지**(자격증명) | — 새 머신에서 다시 로그인 |

## 3. 다운로드할 목록 (우선순위순)

전부 `/scratch/ehmoon/whlee/_migration_2026-09-17/` 한 폴더에 있다(현재 약 2.1 GB).
공간이 부족하면 위에서부터 받는다.

1. `SHA256SUMS`, `MANIFEST.tsv`, `README_session_handoff_2026-09-17.md`, `pack_migration_bundle.sh`, `INVENTORY/`
2. `A_git/` — git bundle + `refs.txt`
3. `B_claude/claude_memory_tools.tar.gz`
4. `C_results_untracked/telemetry_jsonl/r2_eval.tar.zst`, `r2_correctness.tar.zst` — 현재 열린 트랙의 원자료
5. `C_results_untracked/` 나머지 전부
6. `D_misc/` 전부 (`hf_cache_raw_traces`·`repo_untracked_misc`·`hf_converted_model_dirs`·`hf_hub_revisions.tsv`·`vllm_bench`·`workspace_dotfiles`)
7. `E_env/` 전부
8. (선택) `B_claude/claude_transcripts.tar.zst`
9. (선택) HF 모델 — §4

## 4. HF 모델 가중치 (148.5 GiB, 기본 미포함)

| 저장소 | 크기 | 리비전(`refs/main`) | purge 손상 파일 | 권고 |
|---|---:|---|---:|---|
| `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base` | 16.57 GiB | `dc0661c829b1…` | 0 | 현재 열린 트랙(λ0·E2) 모델 — 새 환경에 인터넷이 없으면 `hf_models`로 묶어 가져갈 1순위 |
| `Zyphra/Zamba2-2.7B` | 4.96 GiB | `31afeeac4c66…` | 0 | 재다운로드 |
| `Zyphra/Zamba2-1.2B` | 4.53 GiB | `6b05bf29d1bb…` | 0 | 재다운로드 |
| `Zyphra/Zamba2-7B-Instruct` | 27.72 GiB | `3146688c5286…` | **3** | **로컬 사본 이미 불완전 — 재다운로드만 가능** |
| `nvidia/Nemotron-H-8B-Base-8K` | 30.20 GiB | `94ea861e008c…` | **10** | **로컬 사본 이미 불완전 — 재다운로드만 가능** |
| `tiiuae/Falcon-H1-7B-Instruct` | 14.14 GiB | `41e72f27effb…`(`refs/ToBeDelete_main`에서 읽음) | **8** | **로컬 사본 이미 불완전 — 재다운로드만 가능** |
| `tiiuae/Falcon-H1-3B-Base` | 5.87 GiB | `c096902c69be…` | 0 | 재다운로드 |
| `ibm-granite/granite-4.0-h-micro-base` | 5.95 GiB | `372ede0bc484…` | 0 | 재다운로드 |
| `mistralai/Mamba-Codestral-7B-v0.1` | 13.57 GiB | `4f086c08c1e0…` | 0 | 재다운로드(게이트 여부 미확인) |
| `state-spaces/mamba2-2.7b` | 5.03 GiB | `99b226cc377d…` | 0 | 재다운로드 |
| `Qwen/Qwen2.5-7B` · `Qwen2.5-3B` · `Qwen2.5-0.5B-Instruct` | 14.20 · 5.76 · 0.01 GiB | `d14972939875…` · `3aab1f1954e9…` · `7ae557604adf…` | 0 | 재다운로드 |
| `EleutherAI/gpt-neox-20b`(tokenizer) | 3.6 MB | `c292233c833e…` | 0 | 재다운로드 |
| `THUDM/LongBench`(dataset) | <1 MB | `5e628be450b7…` | **2** | 재다운로드 |

전체 40자리 리비전은 `D_misc/hf_hub_revisions.tsv`·`INVENTORY/hf_models.tsv`.

- 새 환경에서 받기: `huggingface-cli download <repo> --revision <40자리> --cache-dir <HF_HOME>/hub`(dataset은 `--repo-type dataset`).
- 이 머신에서 묶어 가기(손상 0인 저장소만 허용, 비압축 tar):
  `/usr/bin/bash tools/migration/pack_migration_bundle.sh hf_models /scratch/ehmoon/whlee/_migration_2026-09-17 models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base`
  → `F_hf_models/<repo>.tar`, 이어서 `sums` 다시 실행. 인자 없이 실행하면 크기·손상 표만 출력한다.

## 5. Claude 스킬·에이전트 — 이양 후 수정할 곳

줄 단위 목록은 `migration_claude_tools_retarget_2026-09-17.md`. 카테고리별 합계: 경로 30줄 · 스케줄러 36 ·
모듈·환경 15 · git 원격·자격증명 9 · 하드웨어·기관 8 · 메모리 경로 3 (한 줄이 여러 카테고리에 걸칠 수 있음).

| 파일 | 참조 줄 | 이양 후 할 일 |
|---|---:|---|
| `.claude/agents/experiment-runner.md` | 31 | **가장 큼.** SLURM 제출·`--comment` 정책·파티션·`module load`·venv 경로·A100 108 SM 서술 전부 새 클러스터 기준으로 재작성(스케줄러가 없으면 실행 규약 자체를 바꿔야 함) |
| `CLAUDE.md` | 21 | "환경 / 실행" 절(경로·모듈·SLURM·`--comment`)과 메모리 인덱스 경로, 워크스페이스 경로 서술 |
| `.claude/agents/git-committer.md` | 10 | 안팎 저장소 경로, 원격 URL 토큰 경고·push 규칙(새 자격증명 방식에 맞춰 재검토) |
| `.claude/skills/catch-up/SKILL.md` | 6 | `git -C` 경로들, `squeue` 호출. (참고: 30행 `git -C /scratch/ehmoon/whlee log`는 바깥이 git 저장소가 아니라 지금도 실패하는 줄) |
| `.claude/skills/handoff/SKILL.md` | 6 | `git -C` 경로들, 메모리 경로, push 금지 문구 |
| `.claude/agents/engine-porter.md` | 4 | dev tree 경로·sync 절차(수동편집 패치 단계 추가 검토) |
| `.claude/agents/doc-steward.md` | 1 | 메모리 인덱스 경로 |
| `tools/claude/sync_claude_tools.sh`·`test_sync_claude_tools.sh`·`README.md` | 5 | live 루트 기본값 `/scratch/ehmoon/whlee` — `CLAUDE_TOOLS_LIVE=<새 루트>`로 덮어쓰거나 기본값 수정 |
| `claims-auditor`·`result-analyst`·`venue-strategist`·`review-lead` | 0 | 수정 불필요 |

수정 순서(새 머신): live 파일 수정 → `CLAUDE_TOOLS_LIVE=<새 루트> tools/claude/sync_claude_tools.sh --import` →
`test_sync_claude_tools.sh` → 커밋 → `scan_machine_refs.py --live <새 루트>`로 남은 참조 확인.
메모리 topic 파일 안의 `/scratch/ehmoon` 언급(13줄)은 과거 기록이라 고치지 않는다.

저장소 전체로는 추적 파일 중 `.sbatch` 131개·`.sh` 51개·`.py` 102개 등이 절대경로를 갖는다(대부분 `results/` 아래
과거 캠페인 하네스 — 재실행할 것만 고치면 된다).

## 6. 전송 경로 확인 — 로컬 PC `163.239.23.156` (2026-09-17, 이 로그인 노드에서 확인)

| 점검 | 결과 |
|---|---|
| IP 소유(APNIC RDAP) | `163.239.0.0/16` **SOGANG-NET (Sogang University), KR** |
| 역방향 DNS | 없음(NXDOMAIN) |
| ping | 3/3 손실 (대조: `8.8.8.8`은 응답) |
| TCP 22 (ssh) | 6초 타임아웃 |
| **대조: 로그인 노드 → `github.com:22`** | **타임아웃** (`github.com:443`은 열림) |
| 이 노드의 외부 송신 IP | `150.183.150.99` (`neuron.ksc.re.kr` 중 하나) |

⇒ **Neuron 로그인 노드에서 바깥으로 나가는 SSH(22)는 대상과 무관하게 막혀 있다**(github.com도 실패). 따라서 이 머신에서
PC로 *보내는*(`scp`/`rsync` push) 방식은 PC 설정과 상관없이 안 된다. PC 쪽 ping·22 무응답은 PC/학내 방화벽 때문일 수도
있으나, 위 대조 때문에 이 점검만으로는 PC 상태를 판단할 수 없다.

**권장: PC에서 Neuron으로 접속해 받아오기(pull).**

```bash
# 로컬 PC(Linux/macOS/WSL). OTP 인증이 한 번만 필요하도록 rsync 한 번으로 받는다. 끊기면 같은 명령을 다시 실행(이어받기).
rsync -avP --partial ehmoon@neuron.ksc.re.kr:/scratch/ehmoon/whlee/_migration_2026-09-17/ ./migration_2026-09-17/
cd migration_2026-09-17 && sha256sum -c --quiet SHA256SUMS && echo OK      # macOS: shasum -a 256 -c SHA256SUMS
```

- Windows: WinSCP/FileZilla(SFTP, 호스트 `neuron.ksc.re.kr`)로 폴더 통째로 받은 뒤 WSL 또는 `Get-FileHash -Algorithm SHA256`으로 대조.
- PC가 학내망에서 외부 22번으로 나갈 수 있어야 한다(학교 방화벽 정책은 확인하지 않았다).
- Neuron에는 데이터 이동용 노드 `gdm01–03`(로그인 메시지 기준)이 있으나(로그인 노드에서 내부 접근 가능하다고 안내됨) 외부 직접 접속 가능 여부는 확인하지 않았다.
