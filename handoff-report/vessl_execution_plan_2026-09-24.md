# VESSL 클라우드 실행 전환 — 에이전트 검토 + 추가 작업 목록 (2026-09-24)

사용자 결정(2026-09-24): **GPU 실행은 VESSL 클라우드에서 한다.** `migration_plan_local_dev_remote_gpu_2026-09-18.md`의
"대여 서버(업체 미정)" 자리에 VESSL이 들어간다. 짝 문서는 `gpu_rental_checklist_2026-09-18.md`(기판 판정·B0–B5)다.

> **지위**: 새 측정 0건, 새 성능 판정 0건, GPU 지출 0. 이 문서의 내용은 (a) VESSL 공개 문서 확인(2026-09-24)과
> (b) `.claude/agents/*`·`.claude/skills/*`·활성 스크립트 대조뿐이다. 연구 결론·Claim 등급·게이트는 바뀌지 않는다.
> "VESSL A100이 KISTI A100과 같은 기판인가"는 **이 문서가 판정하지 않는다**(§4, 사용자 결정 + 규칙층 감사).

> ★**정정(같은 날)**: §1·§3-2·§3-3은 **구 플랫폼 문서(docs.vessl.ai)** 기준이다. 실제로 쓸 VESSL Cloud(docs.cloud.vessl.ai,
> `vesslctl`, Workspace/Job/Volume)의 사실·가격(A100 SXM $1.55/h)·이미지(managed py3.13 base 위 파생)·회수 경로는
> **`vessl_cloud_setup_plan_2026-09-24.md`가 대체**한다. §2(에이전트 검토)·§4(사용자 결정)는 유효.

## 1. VESSL 실행 모델 — 확인한 것과 미확인인 것

**문서로 확인(docs.vessl.ai, 2026-09-24)**

| 항목 | 내용 | 우리에게 의미 |
|---|---|---|
| 실행 단위 | **Run** = Kubernetes 위 컨테이너 1개. YAML 하나(`image`·`resources.preset`·`import`·`mount`·`export`·`run[].command`·`env`)로 정의, `vessl run create -f <yaml>` | 서버에 ssh로 들어가 스크립트를 돌리는 모델이 아니다. **매 실행이 새 컨테이너** |
| 대화형 | **Workspace** + SSH(`vessl ssh-key add`, `vessl workspace ssh`, 수동 `ssh -p <port> vessl@tcp.<cluster>.vessl.ai`). Run도 `vessl run ssh` 가능 | B0–B5 기판 검증·디버깅은 Workspace로 할 수 있다 |
| 입력 | `import`: `git://`, `vessl-model://`, `vessl-dataset://`, `vessl-artifact://`, `s3://`, `volume://vessl-storage/<vol>` | 코드는 git(특정 커밋), 모델·trace는 볼륨/모델 저장소 |
| 출력 | `export`: `vessl-artifact://`, `volume://vessl-storage/`, `s3://` 등 — **실행 완료 후 업로드** | 로컬 회수는 "rsync"가 아니라 **export → 로컬 다운로드 → sha256** |
| 로그 | `vessl run logs <id> -f`, `vessl run read`, `vessl run list`, `vessl run terminate` | 모니터 명령이 `squeue/sacct`를 대체 |
| 비용 | AP-KR on-prem 클러스터: **A100 80GB SXM ×1 = $1.80/h**, **H100 80GB SXM5 ×1 = $2.99/h**, on-demand만(spot 없음) | E2 예산 1.803 GPU-h(하드캡 3.0) ≈ $3.2–5.4. 누적 ≈462 GPU-h 규모면 ≈$830 |
| 분할 GPU | 플랫폼이 **MIG·소수 GPU preset**을 지원한다(`specs` 문서) | ★**반드시 전체 GPU 1장 preset**을 골라야 한다(green context는 GPU 전체 전제) |

**미확인(접속 후 B0/B5로 확정할 것 — 코드 상수·추측으로 채우지 말 것)**

1. 노드 **NVIDIA 드라이버 버전**. 우리 스택은 `torch==2.9.1+cu130`·`sglang-kernel==0.4.1+cu130`
   (`env/venv_packages_2026-09-17.txt`)라 green context 요구(≥12.4)보다 **CUDA 13 드라이버 요구가 더 빡빡하다**.
   드라이버가 CUDA 13을 못 받치면 cu12x 스택으로 재빌드해야 하고, 그건 **또 하나의 기판 변수**다.
2. **노드 단독 점유 여부**(같은 노드의 다른 GPU에서 남의 작업이 도는가 — 호스트 CPU·PCIe·전력 공유).
3. 컨테이너 권한: `nvidia-smi` 조회는 되겠지만 **클럭/전력 고정(`-lgc`/`-pl`)은 거의 확실히 불가** → §3-4.
4. Run이 **실패·terminate될 때도 `export`가 수행되는가.** 안 된다면 중간 결과는 컨테이너와 함께 사라진다.
5. Workspace 디스크가 stop/start 사이에 유지되는 경로(`/root` 등)와 용량.
6. 볼륨 → 로컬 다운로드 CLI의 정확한 문법(`vessl storage …`; `vessl volume`은 deprecated로 표시됨).
7. HF·PyPI·GitHub 아웃바운드, private 레지스트리/저장소 credential 등록 방식.

## 2. 에이전트·스킬 검토 결과

### 2-1. `experiment-runner` (실행 직결 — 가장 많이 고쳐야 함)

| # | 위치 | 문제 | 조치 |
|---|---|---|---|
| R1 | "지금의 지형" 3번째 bullet | 로컬 거부 사유를 여전히 "`get_arch_constraints` major 6–9 → `ValueError`"로 적음. **2026-09-21 정정(`CONSENSUS §3` 항목278) 미반영** — 이 분기는 `manual_divisions` 경로에서 발화하지 않는다 | CLAUDE.md 개정문과 같은 3사유(sgl_kernel sm120 빌드 없음·16GB·게이트1)로 교체 |
| R2 | B절 "기판 확인 먼저" | "H100은 8 SM 단위", "Blackwell은 코드 확장 필요"를 사실로 적음 — 둘 다 정정 대상(입도는 B3 실측, Blackwell은 미지) | 체크리스트 §1 판정표를 인용하도록 교체 |
| R3 | "실패 시그니처" green context 거부 | `Unsupported compute capability`를 기판 거부 신호로 적음 — 우리 경로에선 그 메시지가 나오지 않는다. 실제로 볼 것은 `sgl_kernel.spatial` import 실패("Ensure CUDA Driver >= 12.4")·`cuGreenCtxCreate` 오류·**요청≠실현 SM** | 시그니처 교체 + B3(realized 확인) 연결 |
| R4 | "실험 규약" | **n≥5**. CLAUDE.md 게이트3·result-analyst는 **n≥4** | 정본(n≥4)과 맞추거나, "운영 목표 n≥5 / 판정 하한 n≥4"로 명시 분리 |
| R5 | B절 전체 | "ssh/rsync로 부트스트랩·실행·회수"를 전제 — VESSL Run은 **컨테이너 1회성 + export** 모델 | B절을 VESSL 절로 재작성(§3-3) |
| R6 | 규약 "GPU clock/power 고정" | 컨테이너에서 강제 불가 | "고정" → "**매 실행 기록 + arm 간 동일성 검증**"(§3-4) |
| R7 | — | 이미 있는 `scripts/run/array_runner.sh`(커밋 `95e542b`, 스케줄러 없는 배열 러너)를 언급하지 않음 | Run 안의 실행 진입점으로 명시 |
| R8 | "예산" | 단가·장부 위치가 없음 | preset·단가($1.80/h)·run id별 소진을 `results/<campaign>/budget.tsv` 류에 기록 |
| R9 | frontmatter description | "대여 GPU 서버가 정해지면" | "VESSL Run/Workspace 실행·회수"로 갱신 |

### 2-2. 나머지 에이전트·스킬

| 대상 | 문제 | 조치 |
|---|---|---|
| `engine-porter` "코드 지형" | "로컬 GPU에서는 PD-mux 경로가 **엔진 단계에서 거부**" — R1과 같은 정정 미반영. dev tree를 "실행 머신에서 새로 만든다"는 전제도 VESSL에선 **이미지 빌드 시점**으로 바뀐다 | 문구 정정 + "dev tree는 이미지에 굽고 manifest sha256을 이미지 라벨·결과 디렉터리에 남긴다" 추가 |
| `result-analyst` | 기판 동일성 확인이 워크플로에 없음(현재 cudagraph·backend·seed·commit만 대조) | 5단계에 **GPU 모델·드라이버·노드 id·이미지 digest·clock/power 기록이 arm 간 동일한지** 추가. 다르면 confounded |
| `claims-auditor` | confound 목록에 **기판 혼합**(KISTI A100 ↔ VESSL A100, 노드 간)이 없음(`기판|substrate|node` grep 0건) | confound 항목 추가: "옛 기판 결론을 새 기판 수치로 보강/반박", "arm이 다른 노드·드라이버에서 측정됨" |
| `git-committer` | 커밋 금지 목록에 VESSL 자격증명이 없음 | `~/.vessl/`·access token·run YAML의 `secret: true` 값·레지스트리 credential 추가 |
| `catch-up` 스킬 §2 | "로컬 5060 Ti는 PD-mux가 **거부**"(정정 미반영) + 실행 중 작업 확인 수단 없음 | 사유 정정 + `vessl run list`(실행 중/최근 Run)·미회수 export 확인으로 교체 |
| `handoff` 스킬 | 실행 id 기록 형식이 SLURM job id 전제 | VESSL run id·preset·이미지 digest·회수 여부를 기록 항목으로 |
| `CLAUDE.md` "환경 / 실행" | "대여 서버(업체 미정)" | VESSL 확정 반영, 원격 실행 규약은 experiment-runner로 위임 한 줄 |
| `doc-steward`·`venue-strategist` | 실행 경로와 무관 | 변경 불필요 |

모든 툴 파일 수정 후 `tools/claude/sync_claude_tools.sh --import` → `test_sync_claude_tools.sh` → 커밋(CLAUDE.md 규약).

**새 에이전트는 만들지 않는 것을 권고한다.** 실행 규약은 experiment-runner 한 곳에 두고, 반복 절차
(launch·fetch·verify)는 에이전트 산문이 아니라 **스크립트**(`scripts/vessl/`)로 고정한다
(교훈 (66) "규칙을 산문으로 고정하면 구멍이 난다").

## 3. VESSL 때문에 새로 생기는 작업 (우선순위 순)

### 3-1. [P0] 계약·preset 확인 (GPU 0, 사용자/VESSL 문의)

- `A100 80GB SXM ×1` **전체 GPU preset** 이름 확정(MIG·소수 preset 배제). H100을 고르면 체크리스트 §4 비용이 붙는다.
- §1 미확인 1–7 답변 확보. 특히 **드라이버 버전(CUDA 13 가능 여부)**과 **실패 시 export 동작**.

### 3-2. [P0] 실행 이미지 (가장 큰 신규 작업)

VESSL은 매번 새 컨테이너라 migration plan §3-3의 "이미지로 굳히는 편이 안전"이 **필수**가 된다.

- `Dockerfile`(예: `workspace/engine-port/env/docker/`): CUDA 13.0 기반 → **Python 3.14.2** → `torch 2.9.1+cu130`·
  `triton 3.5.1`·`flashinfer 0.6.10`·`transformers 5.8.0`·`sglang-kernel 0.4.1+cu130`(`venv_packages_2026-09-17.txt`) →
  SGLang **v0.5.10 소스** → `sync_engine_tree.sh` → `devtree_manual_edits.patch` → `sitecustomize.py`.
- ★패키지 목록은 캡처 시점에 purge가 진행 중이던 **best-effort** 목록이다(파일 헤더). 이미지 빌드 후 CPU 회귀가
  로컬 기준선(598 tests / failures 26 / errors 13, `env/cpu_regression_baseline_2026-09-18.md`)과 같은지로 검증.
- 이미지 digest와 sync manifest sha256을 **매 실행 결과 디렉터리에 기록**(재현성의 1차 근거).
- 레지스트리(Docker Hub private / GHCR 등) + VESSL `credential_name` 등록 — 사용자 결정.
- 모델 가중치(Nano-9B-v2 ≈17 GB)·trace는 이미지에 넣지 말고 **VESSL 볼륨에 한 번 올려 `import`**(리비전은
  `D_misc/hf_hub_revisions.tsv` 고정). 매 Run마다 HF에서 받으면 시간·과금·리비전 드리프트가 생긴다.

### 3-3. [P1] Run 템플릿 + 로컬 러너 (`scripts/vessl/`)

- `run_template.yaml`: `image`(digest 고정) · `resources.preset`(전체 GPU) · `import: /code ← git://…@<commit>`,
  `/models ← volume://…` · `export: /code/.../results/<campaign> → volume://vessl-storage/<campaign>` ·
  `run: array_runner.sh <target> <spec>` · `env: PDMUX_*`, `TRITON_CACHE_DIR`(실행별 분리).
- Run 첫 단계에서 **B0 기계 사실**(`nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version,clocks.max.sm,power.limit`,
  노드명, 이미지 digest)을 `results/<campaign>/job_<id>/substrate.json`으로 떨군다.
- 로컬 `vessl_launch.sh`(커밋 확인 → yaml 렌더 → `vessl run create -f` → run id 기록) /
  `vessl_fetch.sh`(export 다운로드 → `results/<campaign>/` 원위치 → `sha256sum` 매니페스트 → 성공해야 "완료").
- 저장소 접근: `git://` import가 private GitHub을 받으려면 credential 등록 필요. 대안은 로컬에서 커밋 고정
  tarball을 볼륨에 올리는 방식 — 어느 쪽이든 **커밋 해시를 결과에 기록**.
- 실패 시 export가 안 된다면(§1-4), Run 안에서 셀 단위로 결과를 볼륨/S3에 주기적으로 밀어내는 훅이 필요하다.

### 3-4. [P1] 측정 규약 조정 (claims-auditor 사전 검토 대상)

- **clock/power**: 고정 불가 → 매 실행 기록 + arm 간 동일성 검사. 같은 E2 등록 안의 대조 arm은 **같은 Run(같은 노드)
  안에서** 돌리는 E2-α 설계가 여기서 더 중요해진다.
- **노드 간 변동**: Run마다 다른 노드에 배정될 수 있다. 쌍(pair)은 같은 Run 안에서 구성하고, 여러 Run에 걸친
  반복은 노드 id를 공변량으로 남긴다. 기준선 분산(게이트 3)을 **VESSL에서 새로 측정**하기 전엔 옛 ±값을 쓰지 않는다.
- 비용 장부: run id · preset · 시작/종료 · 청구 단위 · $ — E2 이식성 `REFUTED`의 3경로 중 "과금 모형" 경로가 이것.

### 3-5. [P1] 기판 bring-up (체크리스트 §3 B0–B5를 VESSL로 번역)

Workspace에서 대화형으로: B0 기계 사실 → B1 이미지 기동 → B2 CPU 회귀 → **B3 SM 입도 probe**
(`green_readout.py`·`%smid` probe로 **realized** 확인, 생성 성공 ≠ 요청 SM 수신) → B4 부팅 스모크(`triage/pdmux_a100_smoke.yml`)
→ **B5는 VESSL에선 "Run 1개를 export→로컬 fetch→sha256까지 끝까지"**. B5 통과 전 캠페인 제출 금지.

## 4. 사용자 결정 사항 (판정 아님 — 이 문서는 답하지 않는다)

1. **기판 동등성**: VESSL `A100 80GB SXM`은 SKU·SM 수(108)가 같을 수 있으나 노드·드라이버·전력·호스트가 다르다.
   옛 λ\*·격자·분산을 그대로 쓸지, 소규모 재앵커(예: λ0 일부 셀 재현)로 확인할지 — 사용자 결정 + claims-auditor 규칙층 감사.
   그 전까지는 CLAUDE.md "기판 주의"대로 **수치를 섞지 않는다**.
2. **E2 새 OVERRIDE**: 기존 OVERRIDE는 이식성 `REFUTED`로 무효. VESSL 단가·예산 집행기로 재작성 + 승인 필요.
3. 이미지 레지스트리·저장소 import 방식(credential 등록 vs tarball 업로드).
4. bring-up을 Workspace(시간 과금, 대화형)로 할지 Run만으로 할지.

## 5. 권장 순서

1. §3-1 문의·preset 확정 (GPU 0)
2. §2 에이전트·스킬 문구 정정 → `sync --import` → 커밋 (GPU 0)
3. §3-2 이미지 빌드 → 로컬 CPU 회귀로 기준선 대조 (GPU 0)
4. §3-3 템플릿·러너 작성, dry-run(렌더만) (GPU 0)
5. §3-4 규약 조정안 → claims-auditor 사전 감사 (GPU 0)
6. §3-5 bring-up B0–B5 (첫 GPU 지출, 소량)
7. §4 결정 후 E2/λ0 재등록 여부 판단
