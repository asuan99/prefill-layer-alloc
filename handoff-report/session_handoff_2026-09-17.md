# 세션 핸드오프 — 2026-09-17 (머신 이양 준비 · 중간 마무리)

> 이 문서는 **머신 이양용 체크포인트**다. 연구 결론·트랙 상태는 바뀌지 않았다 —
> 정본은 그대로 `PROJECT_STATUS.md`(최상단 배너 2026-09-15(6)) · `reports/CONSENSUS.md`
> rev79 · 직전 핸드오프 `session_handoff_2026-09-16.md`다.

## 이번 세션 요약

사용자가 이 머신(KISTI Neuron, 작업 루트 `/scratch/ehmoon/whlee`)을 더 이상 쓸 수 없게 되어
이양을 준비했다. **옮겨 갈 곳은 아직 정해지지 않았고 임시 저장소만 있다.** 이 세션의 작업은 전부
**GPU 0 · 새 측정 0 · 새 성능 판정 0**이다: ① git에 없는 자산 전수 조사 ② 엔진 런타임을
git + SGLang v0.5.10만으로 재구성할 수 있는지 바이트 단위로 실증 ③ 패키지 버전·모델 리비전·외부
클론 커밋 기록 ④ 저장 번들 스크립트 작성 + 아카이브 생성 ⑤ 이 문서.

## 연구 상태 스냅샷 (변경 없음 — 포인터만)

- 확정 결론(CLAUDE.md "현재 결론")·HE0·layer-type 死·정책 순위·Claim D/E 등급 **전부 불변**.
- **E2(sticky 분할 대조)**: 사전등록·감사 5회 `GO-with-caveats` 완료, **미실행**(GPU 지출 0·라벨 0).
  OVERRIDE는 유효·미소진이지만 **이 클러스터의 A100 108 SM·예산 기준으로 승인된 것**이다(아래 "이양 시 주의" 1).
- **E1a/E1b**: 사전등록 미작성.
- **실행 중 job 없음.** 워킹트리 clean(이 세션 커밋 전 기준).
- **SLURM 블로커 지속**: 2026-09-17에 `sbatch --test-only`(amd_a100nv_8, `--comment` 포함)로 재확인 —
  2026-09-15와 **같은 문구로 거부**(`Your account has expired or exceeded the allocated CPU time`).
  원인(만료 vs 한도초과)은 여전히 **미확정**.

## 측정 (GPU 0, 로그인 노드)

### M1. 엔진 dev tree는 git + v0.5.10 + 수동편집 패치로 바이트 동일하게 재구성된다

- 방법: `external/sglang-latest`의 태그 `v0.5.10`에서 `python/sglang`를 `git archive`로 꺼낸 새 사본에
  `sync_engine_tree.sh`(`SGLANG_ENGINE_DEV`로 사본 지정, 별도 lock·manifest)를 돌리고 live 트리
  (`/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang`)와 `diff -rq`(`__pycache__`,
  `*.orig`, 양쪽 모두 끊긴 심볼릭 링크 `.clang-format` 제외).
- **sync만으로는 5파일이 다르다**(음성 대조): `configs/__init__.py` · `layers/attention/triton_backend.py` ·
  `managers/scheduler.py` · `model_executor/model_runner.py` · `utils/hf_transformers_utils.py`.
  이는 이미 알려진 사실이다 — `PROJECT_STATUS.md` 방법론 게이트 #33 = `CONSENSUS.md` §3 항목48(2026-08-14,
  `env/dev_tree_edits.md` 항목 3·4·5·7 "sync가 재적용하지 않는 수동 편집") 참조. 추가로
  `scheduler.py`의 `PDMUX_LA_COORD` → `event_loop_pdmux_coord` 분기(R0c 계열, 이미 폐기된 경로)도 이 5파일에 들어 있다.
- 이 5파일의 차이를 **`workspace/engine-port/env/devtree_manual_edits.patch`**(117행)로 떠서
  v0.5.10 → sync → `patch -p1` 순으로 적용한 사본은 live 트리와 **`diff -rq` exit 0(동일)**.
- **sync 스크립트는 고치지 않았다**: sync를 바꾸면 등록된 실험(E2)의 런타임이 바뀐다 — engine-porter +
  감사 대상이다. 패치는 이양 복원용 산출물로만 둔다.

### M2. git 밖에만 있는 자산 (크기 = 원본, 압축 전)

| 항목 | 크기 | 재생성 가능? | 판정 |
|---|---:|---|---|
| `workspace/engine-port/results/**` 원시 telemetry `*.jsonl`(`workspace/engine-port/.gitignore:29`로 무시됨) | 약 41 GiB · 5,210 파일 | **GPU 재실행뿐**(누적 ≈462 GPU-h, 계정 블로커) | **필수** |
| 같은 디렉터리의 로그·`.out/.err`·csv 등 비-telemetry 무시 파일 | 약 0.6 GiB | 동일 | **필수** |
| Claude Code 메모리 `~/.claude/projects/-scratch-ehmoon-whlee/memory/` | 약 1 MB · 14 파일 | 불가 | **필수** |
| 저장소 안 기타 무시 파일(루트 `logs/`·`slurm/logs/`·루트 `.out/.err`·`reports/figures/*.png/pdf`·`external/muxwise-zenodo`) | 약 65 MB | 일부만 | 권장 |
| `hf_cache/raw/`(ShareGPT/LongBench 필터 사본 = 벤치 입력 trace) | 약 755 MB | 스크립트로 가능하나 upstream 변동 위험 | **필수**(후반 인벤토리에서 권장→필수로 상향) |
| 로컬 전용 git 커밋: 브랜치 `fix/gate14-tci-analyze`의 origin보다 앞선 4건(`8ad2d7c`·`fc87a17`·`7c11739`·`71cdb30`; main 미병합 7건 중) | — | 불가 | **필수**(git bundle에 포함) |
| Claude 세션 transcript(`~/.claude/projects/-scratch-ehmoon-whlee/*.jsonl` 등) | 약 346 MB | 불가 | 선택 |
| `sglang_engine_dev` 소스 | 43 MB | M1로 재구성 가능 | 보험(선택) |
| `hf_cache/hub` 모델 가중치 | 152 GB | HF 재다운로드(리비전 고정, 아래 M4) | **제외**(사용자 결정: 다른 장치에서 재다운로드) |
| `sglang_engine_venv`(1.4 G)·`vllm_venv`(8.8 G)·`.pip_cache`·triton 캐시·`serving-eval/.venv` | — | 재설치 | 저장 안 함 |
| `external/{sglang-latest,muxwise,bullet}` git 클론 | 510 MB | 커밋 고정 재클론(M4) | 저장 안 함 |
| `SSM-Scope/` | 623 MB | 별도 저장소, 워킹트리 clean·origin과 차이 0 | 저장 안 함 |
| `Microsoft.VisualStudio.Services.VSIXPackage` | 78 MB | VS Code 확장 설치본 | 저장 안 함 |

- `.claude/agents`·`.claude/skills`·루트 `CLAUDE.md`는 `tools/claude/sync_claude_tools.sh --check`로
  **11개 파일 동일**(추적 사본 최신) — git에 이미 있다. 번들에도 원본을 같이 넣었다.
- git 메인 브랜치 `main`은 origin과 차이 0.

### M3. scratch 정리(purge) 흔적 — 이미 진행 중

`ToBeDelete_` 접두사로 이름이 바뀐 파일이 관측됐다: `sglang_engine_venv` **4,616개**(예:
`sgl_kernel/ToBeDelete___init__.py`), `external/` 클론 약 1,580개, `serving-eval/.venv` 1,230개,
`hf_cache` 28개, `characterization`의 `__pycache__` 일부. **`workspace/engine-port/results/`와 추적 파일에서는
0개**(git status clean). Neuron `/scratch`의 미접근 파일 정리 정책과 일치하는 흔적이지만 정책 세부는
이 세션에서 확인하지 않았다. ⇒ **venv는 이미 온전하다고 볼 수 없고, 원시 결과도 복사를 미루면 같은 위험에 놓인다.**

### M4. 재구성용 고정값

- 패키지: `workspace/engine-port/env/venv_packages_2026-09-17.txt`(156개; 핵심 `torch==2.9.1+cu130` ·
  `triton==3.5.1` · `flashinfer-python==0.6.10` · `sglang-kernel==0.4.1+cu130` · `mamba_ssm==2.3.1` ·
  `causal_conv1d==1.5.3.post1` · `flash_attn==2.7.4.post1` · `transformers==5.8.0`, Python 3.14.2,
  conda 모듈 `pytorch_2.9.1_cuda13` 위 `--system-site-packages` venv). py3.14 호환 shim은 이미 추적 중
  (`workspace/engine-port/env/sitecustomize.py`).
- 외부 클론 커밋: `sglang-latest` `926140d789c8…`(https://github.com/sgl-project/sglang) ·
  `muxwise` `eeac5148f456…`(https://github.com/ykcombat/sglang) · `bullet` `371cb2f1b6c9…`
  (https://github.com/zejia-lin/BulletServe). 전체 해시는 번들의 `E_env/external_clone_pins.tsv`.
- HF 모델 리비전(`refs/main`): Nemotron-Nano-9B-v2-Base `dc0661c8…` · Nemotron-H-8B-Base-8K `94ea861e…` ·
  Zamba2-1.2B `6b05bf29…` · Zamba2-2.7B `31afeeac…` · Zamba2-7B-Instruct `3146688c…` ·
  Falcon-H1-3B-Base `c096902c…` · granite-4.0-h-micro-base `372ede0b…` · 그 외 전체는 번들의
  `D_misc/hf_hub_revisions.tsv`(Falcon-H1-7B-Instruct·LongBench는 `refs/main` 없음).

## 코드·문서 변경 (이 세션 커밋에 포함)

- `tools/migration/pack_migration_bundle.sh` — **신규**. git 밖 자산을 단계별 아카이브로 묶는다
  (`results` / `final` / `sums` / `verify`). 결과 telemetry는 **캠페인별 1개 아카이브**라 일부만 옮길 수 있고,
  재실행 시 완료된 캠페인은 건너뛴다. 소규모 캠페인(`r0a`, 21파일)으로 묶기→풀기→`cmp` 왕복 검증 통과.
- `workspace/engine-port/env/devtree_manual_edits.patch` — **신규**(M1).
- `workspace/engine-port/env/venv_packages_2026-09-17.txt` — **신규**(M4).
- 이 문서.

## 저장 번들 — `/scratch/ehmoon/whlee/_migration_2026-09-17/`

| 디렉터리 | 내용 | 우선순위 |
|---|---|---|
| `A_git/prefill-layer-alloc.bundle` + `refs.txt` | 모든 ref(로컬 전용 브랜치 커밋 포함). `.git/config`는 들어가지 않는다 | **필수** |
| `B_claude/claude_memory_tools.tar.gz` | 메모리 · 루트 `CLAUDE.md` · `.claude/` · `~/.claude/{settings.json,plans,skills}` (**`.credentials.json` 제외**) | **필수** |
| `C_results_untracked/results_nontelemetry.tar.zst` | 결과 디렉터리의 비-telemetry 무시 파일 | **필수** |
| `C_results_untracked/telemetry_jsonl/<campaign>.tar.zst` | 원시 telemetry, 캠페인별. 현재 열린 트랙의 외부 앵커는 **`r2_eval`**(λ0 job 908623 원자료 = E2 기준점)·**`r2_correctness`**(908534) | **필수**(용량이 모자라면 이 둘부터) |
| `D_misc/` | 저장소 기타 무시 파일 · `hf_cache/raw` trace · HF 리비전 · `vllm_bench` | 권장 |
| `E_env/` | dev tree 소스 · venv `pyvenv.cfg` · 외부 클론 커밋 | 권장 |
| `B_claude/claude_transcripts.tar.zst` | 세션 transcript | 선택 |
| `MANIFEST.tsv` · `SHA256SUMS` | 파일별 크기·체크섬 | **필수**(복사 후 `sha256sum -c`) |

압축 후 크기는 `MANIFEST.tsv`에 있다(체크섬은 이 문서에 적지 않는다 — 번들이 이 커밋 뒤에 만들어진다).
생성 직후 검증(2026-09-17): 결과 아카이브 35개를 모두 풀어 목록을 세면 **11,162개 = 원본 목록 11,162개**;
telemetry 원본 약 41 GiB → 압축 약 1.8 GB. git bundle은 `git bundle verify` 통과(6 refs, complete history).

## 새 머신 복원 절차

1. **저장소**: GitHub에서 새 자격증명으로 clone하거나 `git clone A_git/prefill-layer-alloc.bundle`.
   번들 clone 뒤 로컬 전용 브랜치 복구: `git branch fix/gate14-tci-analyze origin/fix/gate14-tci-analyze`
   (scratch 사본에서 clone→branch→`71cdb30` 확인 완료).
2. **결과**: 저장소 루트에서 `zstd -dc <archive> | tar -xf -`(경로가 저장소 기준 상대경로).
3. **Claude 도구·메모리**: `tar -xzf claude_memory_tools.tar.gz -C <임시>` 후
   - `CLAUDE.md`·`.claude/`는 새 작업 루트로(또는 `tools/claude/sync_claude_tools.sh --install`),
   - `memory/`는 `~/.claude/projects/<새 작업 루트 경로의 slug>/memory/`로 복사(slug는 경로에서 `/`를 `-`로 바꾼 이름).
4. **엔진**: 새 환경에서 torch/triton/flashinfer/sgl-kernel을 M4 버전(또는 그 환경의 등가물)으로 설치 →
   SGLang v0.5.10 `python/` 준비 → `SGLANG_ENGINE_DEV=<path>/python workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh`
   → `patch -p1 -d <path>/python < workspace/engine-port/env/devtree_manual_edits.patch` →
   `python -m unittest discover -s workspace/engine-port/tests -v`(이 머신 기준 140 tests OK).
5. **모델**: `hf_hub_revisions.tsv`의 리비전으로 HF에서 다시 받는다.
6. 원본을 지우기 전에 반드시 `pack_migration_bundle.sh verify <복사본 경로>` 또는 `sha256sum -c SHA256SUMS`.

## 이양 시 주의 — 환경이 바뀌면 그대로 성립하지 않는 것

1. **기판 의존**: 모든 서빙 결과와 E2 등록 격자(offered 8.0 / 3.0 / 1.15, λ\*(A)≈3.05·λ\*(B)≈0.696 req/s)는
   **이 클러스터의 A100(108 SM)·CUDA 13·green-context** 위에서 측정됐다. 다른 GPU/드라이버/SM 수에서
   E2를 "등록된 실험"으로 돌릴 수 있는지, λ0을 다시 재야 하는지는 **사용자 결정 + 규칙층 감사 사항**이다 —
   이 세션은 판정하지 않았다. OVERRIDE의 예산(1.803/2.338/3.0 GPU-h)도 이 클러스터 기준이다.
2. **하드코딩 경로**: 추적 파일 1,096개에 `/scratch/ehmoon` 5,782회(그중 `.sbatch/.sh/.py` 254개).
   `sync_engine_tree.sh`는 `SGLANG_ENGINE_DEV`로 바꿀 수 있지만 대부분의 sbatch·문서는 경로 치환이 필요하다.
3. **Neuron 전용 규약**: SLURM `--comment="field=efficientai;appl=pytorch"`, 파티션 `amd_a100nv_8`,
   `module load` 줄은 새 클러스터에 맞게 바꿔야 한다. CLAUDE.md "환경 / 실행" 절 전체가 이 머신 기준이다.
4. **dev tree 재현 범위**: sync manifest가 해시하는 15개 파일 밖의 수동 편집 5파일은 여전히 sync가 적용하지 않는다
   (M1). 새 환경에서 patch 단계를 빠뜨리면 Zamba2 부팅과 pdmux 경로(`ngram_embedding_info` 가드)가 달라진다.

## 추가 (2026-09-17 후반) — whlee 전체 인벤토리 · Claude 도구 수정 목록 · 전송 경로

사용자 요청("git에 없는 whlee 하위 파일 전부 정리, 스킬·에이전트도 이양 후 수정 필요")에 따라 작업했다.
정리 결과는 **`migration_inventory_2026-09-17.md`**, Claude 도구 줄 단위 목록은
**`migration_claude_tools_retarget_2026-09-17.md`** 에 있다. 요점:

- **whlee 전체 전수 분류**(`tools/migration/inventory_workspace.py`): git 밖 파일 22개 규칙, 미분류 0.
  인벤토리의 결과 파일 경로 집합 = 결과 아카이브를 풀어 얻은 집합(11,162개, 차이 0).
- **번들 추가**(`pack_migration_bundle.sh extra`): `D_misc/hf_converted_model_dirs.tar.gz`(수작업 mamba2/Codestral
  config·tokenizer), `D_misc/workspace_dotfiles.tar.gz`(`.vscode`), `INVENTORY/`. `D_misc/hf_hub_revisions.tsv`는
  purge가 `refs/main`을 바꿔 둔 저장소 2개(Falcon-H1-7B-Instruct·LongBench)의 리비전까지 읽도록 다시 만들었다.
- **HF 모델 purge 손상**: Nemotron-H-8B-Base-8K(10파일)·Zamba2-7B-Instruct(3)·Falcon-H1-7B-Instruct(8)·
  LongBench(2)는 blob/ref 이름이 이미 바뀌어 **로컬 사본을 옮겨도 쓸 수 없다 — 재다운로드만 가능**.
  가중치(148.5 GiB)는 **사용자 결정으로 제외** — 새 장치에서 `D_misc/hf_hub_revisions.tsv`의 리비전으로 다시 받는다.
- **외부 클론 3개 로컬 수정 0**(`git diff HEAD --diff-filter=M` 0) — 재클론으로 충분.
- **Claude 도구**: 머신 의존 참조는 `experiment-runner`(31줄)·`CLAUDE.md`(21)·`git-committer`(10)·
  catch-up/handoff 스킬(각 6)·`engine-porter`(4)·`doc-steward`(1)·sync 도구(5). 나머지 4개는 0.
  이양 대상이 정해진 뒤 고친다(지금은 목록만).
- **전송 경로**: 사용자 로컬 PC `163.239.23.156`(APNIC: SOGANG-NET, Sogang University)은 이 로그인 노드에서 ping·22번
  모두 무응답. 대조로 `github.com:22`도 타임아웃(443은 열림) ⇒ **Neuron 로그인 노드의 외부행 SSH가 막혀 있어 여기서
  PC로 보내는 방식은 불가**. PC에서 `neuron.ksc.re.kr`로 접속해 받는(pull) `rsync` 절차를 인벤토리 문서 §6에 적었다.

## 열린 항목 / 다음 세션 시작점

1. ★**사용자**: `_migration_2026-09-17/` 전체를 임시 저장소로 복사 → `sha256sum -c SHA256SUMS` 확인. purge가 진행 중이므로 미루지 말 것(M3).
2. 이양 대상이 정해지면: 복원 절차 1–6 → CPU 회귀 통과 확인 → 하드코딩 경로 처리 방식 결정.
3. E2 제출 여부는 **새 환경이 같은 기판인지**에 달려 있다("이양 시 주의" 1). 같은 Neuron 계정이 복구되는 경우에만 기존 OVERRIDE 그대로 재시도 가능.
4. E1a/E1b 사전등록(게이트 #6) 미작성 — 직전 핸드오프 열린 항목 4·5 그대로.
5. 브랜치 `fix/gate14-tci-analyze`(main 미병합 7건) — 2026-09-08 핸드오프 이후 상태 불변.

## 미완·주의

- 아카이브는 **이 머신의 현재 파일 그대로**다. purge로 이미 이름이 바뀐 파일(M3)은 venv·클론·캐시 쪽이라 번들 대상이 아니다.
- `longctx_conflict/probes/**/*telemetry.jsonl`은 `.gitignore`가 "재생성 가능"으로 분류한 것이지만 번들에는 포함했다(가장 큰 캠페인, 13.8 GB 원본).
- 이 세션은 서빙 결론·게이트·등급을 하나도 바꾸지 않았다 — claims-auditor 회부 대상 없음.
