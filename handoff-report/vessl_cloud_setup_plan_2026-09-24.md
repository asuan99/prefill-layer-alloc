# VESSL Cloud 세팅 계획 — Workspace·Storage 생성 + 실험 자원의 컨테이너 이미지화 (2026-09-24)

짝 문서: `vessl_execution_plan_2026-09-24.md`(에이전트 검토, 같은 날) · `gpu_rental_checklist_2026-09-18.md`(B0–B5) ·
`migration_plan_local_dev_remote_gpu_2026-09-18.md` · `vllm_baseline_stack_2026-09-18.md`.

> **지위**: 새 측정 0 · 새 성능 판정 0 · GPU 지출 0. 근거는 **VESSL Cloud 공식 문서(docs.cloud.vessl.ai, 2026-09-24 원문
> 다운로드)**와 로컬 이양 번들(`~/Experiments/KISTI/_migration_2026-09-17/`) 대조뿐이다. 기판 동등성·E2 재등록은
> 판정하지 않는다(사용자 결정 + 규칙층 감사).
>
> ★**정정**: 같은 날 앞 문서(`vessl_execution_plan_2026-09-24.md`) §1은 **구 플랫폼 문서(docs.vessl.ai — `vessl run`,
> Run YAML, `import/export`)**를 근거로 했다. 실제로 쓸 **VESSL Cloud는 다른 제품**이다(CLI `vesslctl`, 단위는
> Workspace/Job/Volume, YAML `export` 없음). 그 문서의 가격($1.80/h)·CLI·회수 경로 서술은 이 문서로 대체한다.
> 에이전트 검토 결과(§2)는 유효하다.

## 1. VESSL Cloud 사실 (1차: 문서 원문)

| 항목 | 문서 내용 | 출처 페이지 |
|---|---|---|
| CLI | `vesslctl` — 설치 `curl -fsSL https://api.cloud.vessl.ai/cli/install.sh \| bash`, `vesslctl auth login`(브라우저 OAuth) | `cli/cheatsheet` |
| 자원 조회 | `vesslctl cluster list`, `vesslctl resource-spec list --usable-only` → slug 예 `a100-sxm-1`(12 CPU·80 GiB RAM·$1.55/h) | `cli/commands/cluster` |
| 가격 | **A100 SXM 80GB $1.55/h**, **H100 SXM 80GB $2.39/h**, on-demand만(spot "Coming Soon"), 2026-03-16 기준 | `pricing/gpu-instances` |
| Workspace | 대화형 컨테이너(SSH·JupyterLab). 사용자 **root**. `create/start/pause/terminate/ssh/logs`. **Paused = GPU $0, 스토리지만 과금** | `member/workspace/*`, `cli/commands/workspace` |
| Workspace 격리 | **"Shared node, containerized"**, 호스트·커널 접근 없음, 드라이버는 VESSL 관리(패치도 VESSL이 적용) | `vm-clusters/vs-workspace`, `guides/platform/managed-software-stack` |
| Job | 명령 하나를 끝까지 실행 후 종료. 상태 `scheduling→running→succeeded/failed/terminated`. **자동 재시도 없음**. 과금은 running 동안만 | `member/job/overview` |
| Job 출력 | **컨테이너 1회성 — 마운트한 볼륨 밖에 쓴 것은 종료 시 사라짐**(succeeded여도) | `member/job/create` |
| Job 설정 재사용 | `vesslctl job export <slug> > job.json` → `vesslctl job create -f job.json` | `cli/commands/job` |
| Cluster storage | CephFS/NVMe, **리전 1곳에 묶임**, terminate 뒤에도 유지, RWX, ≈150 MB/s, **$0.20/GiB/월(실사용)**. 리전 선택해 콘솔에서 생성 | `member/volume/overview`, `admin/storage/create-volume` |
| Object storage | S3 기반 POSIX 마운트, 전 클러스터 공유, RWX, **$0.03/GiB/월**. **`/root`에 마운트 불가** | 같음 |
| 로컬↔볼륨 전송 | `vesslctl volume upload/download/ls/token`은 **Object 볼륨만**. Cluster 볼륨은 워크스페이스에 마운트해 안에서 적재(wget/HF, SSH rsync, Object 경유) | `cli/commands/volume`, `member/volume/load-data` |
| Temporary storage | 컨테이너마다 할당, **stop 시 삭제**. 한도 초과 시 컨테이너 퇴출 | `guides/platform/resource-limits` |
| Custom image | Workspace에 쓰려면 **SSHD + Python 필수**. Job은 private registry 자격증명·pull policy 지원 | `member/workspace/create`, `member/job/create` |
| Managed image | `quay.io/vessl-ai/torch:2.9.1-cuda13.0.1-py3.13-slim` 등 | `guides/platform/managed-software-stack` |
| Secret | org 단위 암호화, **Job에만 주입**(`--secret HF_TOKEN`), Workspace는 미지원 | `member/settings/secrets` |
| API 한도 | `--cmd` 256 KiB, env 값 8 KiB, env 128쌍 초과 시 4xx | `member/volume/load-data` |
| 크레딧 | 잔액 ≤0이면 create 차단. Workspace는 -$10까지 버퍼 | `cli/commands/job`, `member/workspace/create` |
| 워크로드 토큰 | 모든 Workspace에 `VESSLCTL_ACCESS_TOKEN`이 env로 들어 있음 | `cli/commands/workspace` |

**문서에 없는 것(접속 후 실측)**: 노드 NVIDIA 드라이버 버전(→CUDA 13 지원 여부), 같은 노드 다른 GPU의 이웃 부하,
`nvidia-smi` 클럭/전력 제어 가능 여부(컨테이너 격리상 불가 추정), shm 크기, temporary storage 크기(콘솔 카드에 표시),
managed slim 이미지에 sshd가 있는지.

**사용자 확인(2026-09-24)**: 선택 가능한 **A100 SXM 80GB 리전은 `us-west-2` 하나뿐**이다. 따라서 리전은 결정 사항이 아니라
**고정 조건**이다. 결과: Cluster 볼륨도 `us-west-2`에 만든다 · 모델/HF 다운로드는 리전 안이라 빠르지만 로컬(한국)↔VESSL 간
SSH·rsync·`volume download`는 태평양 왕복(RTT 대략 100 ms대, 미실측)이라 대화형 작업은 느리고 대용량 전송은 리전 안에서
끝내는 편이 낫다 · CPU-only 스펙이 `us-west-2` 클러스터에도 있는지 `resource-spec list`로 확인(문서 예시 `betelgeuse-na`
= us-west-2에 A100 SXM·L40S·CPU-Only가 함께 있음 — 예시일 뿐 실측 아님).

## 2. 현재 실험 자원 목록 → VESSL 배치

| 자원 | 현재 위치 | 크기 | 배치 |
|---|---|---:|---|
| 저장소 코드·설정(`*.yml`, 스크립트, 하네스) | 로컬 git | — | Cluster 볼륨에 **커밋 고정 checkout**(자주 바뀜). 커밋 해시를 결과에 기록 |
| SGLang v0.5.10 소스 | `~/Experiments/KISTI/sglang_engine_dev`(태그 `v0.5.10`), 번들 `E_env/sglang_engine_dev_src.tar.zst` | 수 MB | **이미지**(느리게 바뀜) |
| 엔진 패치(`sync_engine_tree.sh` + `devtree_manual_edits.patch`) | 저장소 | — | **이미지 빌드 단계**에서 적용, manifest sha256을 이미지 라벨로 |
| 파이썬 패키지 | `env/venv_packages_2026-09-17.txt`(py3.14.2, best-effort 목록) | — | **이미지**. 단 py3.13 조합은 먼저 검증(§4 1단계) |
| py3.14 shim `env/sitecustomize.py` | 저장소 | — | 이미지에 포함 — ★base 기본 Python이 3.14.6이라 **활성**(§9 실측) |
| 모델 가중치 | HF, 리비전 고정 `D_misc/hf_hub_revisions.tsv` | Nano-9B-v2 17.8 GB(현 트랙). 15개 전부 ≈178 GB | **Cluster 볼륨** `/data/hf`. 현 트랙(E2/λ0)에 필요한 것만 먼저 |
| 변환 모델 디렉터리 | `D_misc/hf_converted_model_dirs.tar.gz`(mamba2-2.7b·codestral-7b sglang 변환본) | 1.1 MB(가중치 제외 목록) | 필요 시 Cluster 볼륨 |
| trace·데이터셋 | `D_misc/hf_cache_raw_traces.tar.zst` | 175 MB | Cluster 볼륨 `/data/traces` |
| triton·flashinfer JIT 캐시 | 실행 시 생성 | — | 실행별로 분리한 디렉터리(과거 캐시 경합 전례). 위치는 Cluster 볼륨 |
| 과거 원시 결과 `C_results_untracked/` | 로컬 번들 | 1.7 GB | **로컬에만**(분석은 CPU). VESSL에 올리지 않는다 |
| vLLM 스택 | 별도 venv였음 | — | **별도 이미지**(torch 핀 충돌, `vllm_baseline_stack` §1) |
| 외부 clone(BulletServe·MuxWise) | `E_env/external_clone_pins.tsv` | — | 지금은 보류 |

## 3. 스토리지 설계

```
Cluster storage  pdmux-cs  (A100과 같은 리전)  → 마운트 /data  (Workspace·Job 둘 다 같은 경로)
  /data/repo/<commit>/              저장소 checkout (커밋별 디렉터리, 덮어쓰지 않음)
  /data/hf/                         HF_HOME (리비전 고정 다운로드)
  /data/traces/                     trace 원본
  /data/cache/<run_id>/{triton,flashinfer}
  /data/results/<campaign>/job_<id>/  실행 중 기록 위치
  /data/env/                        (1단계 bring-up용 venv; §4)
Object storage   pdmux-out → 마운트 /out   (로컬 회수 전용)
  /out/<campaign>/job_<id>/  + SHA256SUMS
```

- 실행 중에는 **Cluster 볼륨에 쓴다.** 컨테이너가 죽어도 남고 Object보다 빠르다. S3 FUSE에 telemetry를 바로 쓰면
  관측자 효과 게이트(<3%)를 건드릴 수 있다.
- 실행이 끝나면 Job이 `/data/results/...`를 `/out/...`로 복사하고 `sha256sum`을 만든다. 로컬에서
  `vesslctl volume download pdmux-out <local> --remote-prefix <campaign>/` 후 `sha256sum -c`를 통과해야 **완료**다.
  Cluster 볼륨은 CLI로 직접 받을 수 없어 이 2단 구조가 필요하다.
- 비용 감각: Nano-9B-v2 + trace + 결과 수십 GB ≈ 월 $10–20. 모델 15개를 전부 올리면 ≈$36/월.
- Cluster 볼륨은 리전에 묶인다. A100이 `us-west-2`뿐이므로 **`pdmux-cs`는 `us-west-2`에 만든다**(다른 리전 GPU로는 마운트 불가).

## 4. 컨테이너 이미지 — 2단계로 만든다

**왜 2단계인가**: 정본 패키지 목록은 py3.14.2에서 purge 중에 캡처한 best-effort 목록이고, managed 이미지는 py3.13이다.
처음부터 Dockerfile을 쓰면 의존성 해소를 GPU 과금 중 빌드/실패 반복으로 찾게 된다. 먼저 동작하는 조합을 찾고, 그다음 굳힌다.

### 1단계 — bring-up(Workspace + Cluster 볼륨 venv, 탐색용)

1. **CPU-only Workspace**($0.20/h)를 A100과 같은 리전에 managed torch 이미지로 만들고 `pdmux-cs:/data`를 마운트한다.
   패키지 설치·SGLang 소스 배치·패치·모델/trace 다운로드는 GPU가 필요 없다.
   - `python -m venv --system-site-packages /data/env/pdmux-v1`(managed torch 2.9.1 재사용) →
     `sgl-kernel 0.4.1+cu130`·`flashinfer 0.6.10`·`triton 3.5.1`·`transformers 5.8.0` 등 핀 설치 →
     `pip install -e <sglang v0.5.10>/python --no-deps` → `sync_engine_tree.sh` → `devtree_manual_edits.patch`.
   - CPU 회귀를 돌려 로컬 기준선(598 tests / failures 26 / errors 13, `env/cpu_regression_baseline_2026-09-18.md`)과 대조.
   - `pip freeze > /data/env/pdmux-v1.lock`, manifest sha256 보존.
2. Workspace를 pause하고, **A100 Workspace**를 같은 볼륨으로 띄워 B0(기계 사실)·B3(SM 입도 realized probe)·B4(부팅 스모크)를 한다.
   끝나면 즉시 pause 또는 terminate(시간 과금).
3. 이 단계 산출물 = 동작이 검증된 **lock 파일 + manifest**. 이 venv로는 측정 캠페인을 돌리지 않는다(가변 환경).

### 2단계 — 측정용 고정 이미지(Job 전용)

```dockerfile
FROM quay.io/vessl-ai/torch:2.9.1-cuda13.0.1-py3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server git patch && rm -rf /var/lib/apt/lists/*
COPY env/pdmux-v1.lock /opt/pdmux/requirements.lock      # 1단계 pip freeze 결과
RUN pip install --no-deps -r /opt/pdmux/requirements.lock
COPY sglang-v0.5.10/ /opt/sglang/                         # 태그 v0.5.10 소스
COPY prefill-layer-alloc/workspace/engine-port/ /opt/pdmux/engine-port/   # sync 스크립트·src·patch만
RUN pip install -e /opt/sglang/python --no-deps \
 && SGLANG_ENGINE_DEV=/opt/sglang/python bash /opt/pdmux/engine-port/scripts/bootstrap/sync_engine_tree.sh /opt/pdmux/manifest.sha256 \
 && patch -p1 -d /opt/sglang/python < /opt/pdmux/engine-port/env/devtree_manual_edits.patch
COPY prefill-layer-alloc/workspace/engine-port/env/sitecustomize.py /opt/pdmux/
LABEL pdmux.repo_commit=<commit> pdmux.sglang_tag=v0.5.10 pdmux.manifest_sha256=<sha>
```

- 이미지에는 **엔진과 의존성만** 넣는다. 실험 스크립트·설정은 `/data/repo/<commit>`에서 읽고 커밋을 기록한다.
  둘을 분리하면 하네스를 고칠 때마다 이미지를 다시 빌드하지 않아도 된다.
- Workspace에서도 쓸 이미지라 `openssh-server`를 넣는다(문서 요건). Job만 쓸 거면 빼도 된다.
- **빌드 위치**: 로컬 PC에 Docker가 없다(`docker: command not found`, 2026-09-24 확인). 선택지는
  (a) 로컬에 Docker 설치(sudo 필요, GPU 불필요) 또는 (b) GitHub Actions → GHCR. 둘 다 **레지스트리 결정**이 필요하다.
  private이면 Job의 레지스트리 자격증명 등록이 필요하다(콘솔 지원 명시, CLI 플래그는 문서에 없음).
- Job에는 태그가 아니라 **digest(`@sha256:…`)**로 지정하고 `--image-pull-policy IfNotPresent`. digest를 결과에 기록한다.
- vLLM 기준선은 같은 방식으로 **별도 이미지**(`vllm_baseline_stack` §3에서 버전 결정 후).

## 5. 생성 절차 (명령 순서)

```bash
# 로컬 1회
curl -fsSL https://api.cloud.vessl.ai/cli/install.sh | bash
vesslctl auth login
vesslctl ssh-key add --name wonho-local --generate      # 또는 콘솔 Settings > SSH Keys
vesslctl cluster list
vesslctl resource-spec list --usable-only               # a100-sxm-1 / cpu-only slug·리전 확인
echo "$HF_TOKEN" | vesslctl secret create HF_TOKEN HF_TOKEN --from-stdin   # Job 전용

# 스토리지 (Cluster는 콘솔: Cluster storage > Create new volume > 리전 선택)
vesslctl storage list                                    # Object storage 백엔드 slug 확인
vesslctl volume create --name pdmux-out --storage <object-storage-slug> --teams <team>

# 1단계 bring-up
vesslctl workspace create --name pdmux-build --cluster <cluster> --resource-spec <cpu-only> \
  --image quay.io/vessl-ai/torch:2.9.1-cuda13.0.1-py3.13-slim \
  --cluster-volume <pdmux-cs>:/data --object-volume <pdmux-out>:/out --ssh-key <key>
vesslctl workspace ssh <slug>        # 여기서 §4 1단계 작업, 코드는 rsync로 /data/repo/<commit>에 올림
vesslctl workspace pause <slug>

# 측정 Job (2단계 이미지 이후)
vesslctl job create -n <campaign>-<n> -r a100-sxm-1 -i <registry>/pdmux-sglang@sha256:<digest> \
  --cluster-volume <pdmux-cs>:/data --object-volume <pdmux-out>:/out \
  --secret HF_TOKEN --env PDMUX_CAMPAIGN=<c> --env PDMUX_ROOT=/data/repo/<commit>/.. \
  --cmd "bash /data/repo/<commit>/prefill-layer-alloc/workspace/engine-port/scripts/vessl/job_entry.sh" --tag <campaign>
vesslctl job logs <slug> -f
vesslctl job export <slug> > results/<campaign>/job_<id>/vessl_job.json   # 실행 설정을 저장소 쪽에 보존
```

`job_entry.sh`(새로 작성): ① B0 기계 사실을 `substrate.json`으로(`nvidia-smi` name·driver·compute_cap·clocks·power.limit,
호스트명, 이미지 digest, 저장소 커밋 — ★**env 전체를 덤프하지 말 것**: `VESSLCTL_ACCESS_TOKEN`·`HF_TOKEN`이 들어 있다)
② 실행별 `TRITON_CACHE_DIR`/flashinfer 캐시 분리 ③ `array_runner.sh` 또는 캠페인 스크립트 ④ `/data/results`→`/out` 복사 +
`SHA256SUMS` ⑤ 실패해도 ④는 수행(`trap`).

## 6. 측정 규약에 걸리는 VESSL 특성 (claims-auditor 사전 검토 대상)

- **공유 노드**: 같은 노드의 다른 GPU에서 남의 작업이 돌 수 있다(호스트 CPU·PCIe·전력). 대조 arm은 **같은 Job 안에서**
  돌리고(E2-α 방식), 노드 호스트명을 공변량으로 남긴다. 기준선 분산은 VESSL에서 새로 잰다.
- **클럭/전력 고정 불가(추정)** → 실행마다 기록하고 arm 간 동일성을 검사한다.
- **드라이버를 VESSL이 패치한다** → 캠페인 도중 드라이버가 바뀔 수 있다. `substrate.json`의 드라이버가 캠페인 안에서
  바뀌면 그 캠페인은 기판 혼합이다.
- **Job은 재시도가 없다** → 실패 셀 재실행은 새 Job이고, 재실행 사실을 기록해야 한다(사후 선택 방지).
- **예산**: 선불 크레딧을 넣으면 잔액 ≤0일 때 Job 생성이 막힌다. 이것을 E2 하드캡(3.0 GPU-h ≈ $4.65)의 **외부 집행기**로
  쓸 수 있는지는 E2 이식성 `REFUTED`의 "예산 집행기" 경로와 함께 감사받을 후보다(여기서 판정하지 않음).
  실행 중인 Workspace는 -$10까지 계속 과금된다는 점도 같이 본다.

## 7. 사용자 결정 사항

1. ~~리전~~ → `us-west-2`로 고정(A100 유일 리전, 사용자 확인 2026-09-24)
2. **이미지 레지스트리**(GHCR private / Docker Hub / quay)와 빌드 위치(로컬 Docker 설치 vs GitHub Actions)
3. HF 모델 업로드 범위(현 트랙만 vs 15개 전부)
4. 기판 동등성 처리 · E2 새 OVERRIDE (앞 문서 §4와 같음)

## 8. 권장 순서

1. `vesslctl` 설치·로그인·ssh-key·`cluster/resource-spec list`로 `us-west-2` 클러스터 slug·`a100-sxm-1`/CPU-only slug 확정 (비용 0)
2. `pdmux-cs`(Cluster)·`pdmux-out`(Object) 생성, HF secret 등록 (스토리지 과금 시작)
3. CPU-only Workspace에서 1단계 env 구성 + 모델/trace 적재 + CPU 회귀 (≈$0.20/h)
4. A100 Workspace에서 B0·B3·B4 (첫 GPU 지출, $1.55/h, 끝나면 즉시 pause)
5. lock 파일로 2단계 이미지 빌드·push, `job_entry.sh` 작성
6. Job 하나로 **B5**(끝까지 회수 + sha256) — 통과 전 캠페인 금지
7. §7-4 결정 후 캠페인 등록

## 9. 로컬 Docker 설치 + managed base 이미지 실측 (2026-09-24, GPU 0)

- 로컬: Docker Engine **29.8.1**, buildx **v0.37.1**, storage driver overlayfs, cgroup v2. (설치 시 NVIDIA devtools apt 저장소
  `NO_PUBKEY F60F4B3D7FA2AF80`로 `apt-get update`가 실패해 `&&` 뒤 설치가 한 번 건너뛰어졌음 — Docker와 무관)
- base: `quay.io/vessl-ai/torch:2.9.1-cuda13.0.1-py3.13-slim` → **digest `sha256:f361ea0d36dae3b692e393e25ec6e7910bc61af80739cd735606865bb19fe4a6`**
  (태그는 바뀔 수 있으므로 Dockerfile `FROM`은 이 digest로 고정), amd64, **비압축 23.4 GB**(torch 계층만 5.73 GB), Ubuntu 24.04.3, user root,
  `ENTRYPOINT /bin/bash`, `CMD start-notebook.sh`(JupyterLab).
- ★**태그의 `py3.13`은 기본 Python이 아니다.** `PATH` 선두의 `/opt/conda/bin/python` = **Python 3.14.6**이고 torch·triton이 여기에
  설치돼 있다. `/usr/bin/python3`(3.13.15)는 시스템 Python일 뿐이다. ⇒ 정본 환경(py3.14.2)과 **같은 3.14 계열**이며,
  `env/sitecustomize.py` shim은 no-op이 아니라 **활성**이다(§4의 "py3.13에선 no-op" 서술은 이 실측으로 대체). 3.14.2↔3.14.6
  패치 버전 차이는 기록 대상.
- base에 이미 있는 것: torch **2.9.1+cu130**·triton **3.5.1**(정본과 일치), cuDNN 9.13, NCCL 2.28.3(pip `nvidia-nccl-cu13` 2.27.7),
  **nvcc 13.0**(flashinfer JIT 가능), **sshd**·git·gcc/g++·make·patch·curl·wget. 없는 것: **rsync·ninja**·cmake.
- 정본 목록(147개) 대조: 일치 50 · 버전 다름 24(대부분 anyio·certifi 등 부수 패키지, numpy 2.5.1↔2.5.2, setuptools 70.2↔83.0) ·
  **없음 73**(sglang 계열·flashinfer 0.6.10·sglang-kernel 0.4.1+cu130·transformers 5.8.0·huggingface-hub 1.14.0 등).
  정확 재현하려면 lock에서 24개 차이를 **정본 버전으로 되돌려 설치**할지(권고) base 버전을 받아들일지 결정·기록한다.
- ★**컴파일형 CUDA 확장 3개**(`causal_conv1d 1.5.3.post1`·`mamba_ssm 2.3.1`·`flash_attn 2.7.4.post1`, 정본에서 `[system]`)는
  py3.14 + torch 2.9.1 + cu130용 미리 빌드된 wheel이 없을 가능성이 높아 소스 컴파일이 필요할 수 있다(미확인). 우리 코드
  (`engine-port/{src,scripts,benchmarks,tests}`)에서 직접 import 0건, 엔진 트리에서는 `flash_attn`만 선택적 import
  (`hf_transformers_utils.py` 호환 shim, `kimi_vl_moonvit.py`). ⇒ **처음엔 빼고 빌드**, B4 부팅 스모크에서 필요성이 드러날
  때만 `TORCH_CUDA_ARCH_LIST=8.0`·`MAX_JOBS` 제한으로 로컬 컴파일(로컬 RAM 31 GiB라 flash-attn 병렬 컴파일은 OOM 주의).
- push 비용: base 계층을 우리 레지스트리로 **처음 한 번 전부 올려야** 한다(압축 크기는 미측정, 비압축 23.4 GB). 이후 push는
  바뀐 계층만. 레지스트리 무료 용량 한도를 넘을 수 있으니 레지스트리 선택 시 확인.

## 10. 로컬 이미지 초안 빌드 결과 (`pdmux-sglang:draft2`, 2026-09-24, GPU 0)

파일: `workspace/engine-port/env/docker/{Dockerfile.pdmux-sglang, requirements.pdmux-sglang.txt, build_image.sh}` (미커밋).
빌드 입력: base digest `sha256:f361ea0d…` · SGLang `git archive v0.5.10`(commit `1519acf3`) · engine-port = repo `b1c739c`의
`src/`·`scripts/bootstrap/`·`env/` (build_image.sh가 미커밋 변경을 거부). 결과 이미지 비압축 27.0 GB.

- **sglang-kernel**: PyPI `0.4.1`은 **CUDA 12 빌드**(`libcudart.so.12`·`libcublas.so.12` 링크, draft1에서 실측) ⇒ 정본
  `0.4.1+cu130` wheel을 `github.com/sgl-project/whl/releases/download/v0.4.1/…+cu130…whl`에서 **sha256
  `9164b8fc…c5c0` 고정**으로 설치. draft2에서 모든 `.so`가 `libcudart.so.13`/`libcublas.so.13` 링크 확인.
  로더(`sgl_kernel/load_utils.py`)는 cc==90이면 `sm90/`, **그 외(A100 sm80 포함)·GPU 없음이면 `sm100/common_ops`**를
  로드 ⇒ A100에서 실제 로드 성공 여부는 **B4에서 확인**(GPU 없는 컨테이너에선 로드 실패가 정상).
- **패키지 대조**: 정본 147개 중 **138개 버전 일치**, 차이 2(pip 도구·sglang 소스 설치 버전 표기 `0.0.0.dev0`), 부재 7
  (의도적 제외: 컴파일형 확장 3·PyPI 부재 3[`cuda-tile`·`pf-sparse`·`zmq`]·`uv`). `pip check` 경고 1건:
  `flashinfer-python 0.6.10 requires cuda-tile` — `import flashinfer` 정상(실측), 기록만.
- **엔진 트리 동일성**: 이미지 manifest 25개 해시 = 로컬 dev tree manifest(md5 of hashes 일치). 트리 전체 `diff -rq`
  차이는 `_version.py`(editable 설치 생성물)와 로컬 쪽 깨진 심볼릭 링크 `.clang-format` 뿐.
- **CPU 회귀(같은 커밋, 같은 날 호스트 재실행과 비교)**:

  | | 호스트 miniconda py3.13.9 | 이미지 draft2 py3.14.6 |
  |---|---|---|
  | tests | 598 | **662** |
  | failures / errors / skipped | 27 / 13 / 2 | **21 / 6 / 1** |

  이미지의 실패 28개는 **전부 호스트에도 있는 실패**(신규 실패 0). 호스트에만 있는 14개는 이미지에서 통과하거나
  수집된다 — 호스트는 7개 테스트 모듈이 import 단계에서 죽어(64개 미수집) 수가 적다. 남은 공통 실패는
  GPU/libcuda 부재(`common_ops`), 로컬 미존재 파일(`scripts/r2_eval/lambda_star.example.json`), HF 캐시 부재(Zamba2) 부류.
  ⚠ 컨테이너를 root로 돌리며 사용자 소유 저장소를 bind mount하면 git `dubious ownership`으로 19건이 추가 실패한다
  (1차 실행) — `git config --global --add safe.directory '*'` 후 재실행한 값이 위 표. VESSL 볼륨 checkout에서도
  소유자가 다를 수 있으니 `job_entry.sh`에 같은 설정을 넣는다.
- **남은 일**: 레지스트리 결정·push(첫 push는 base 계층 포함) → B4 부팅 스모크에서 `sm100/common_ops`·green context·
  flash_attn/mamba_ssm 필요 여부 확인.

## 11. `VESSL_A100_CONTEXT.md`(작업 루트, 사용자 제공, 2026-09-24) 대조 재검토

해당 문서는 VESSL A100 Workspace에서 **Driver API 프로브(`gc_probe.cu`)로 실측**한 환경 제약이다. 이 계획의 미확인 항목 중
일부를 해소하고, 일부는 여전히 미확인으로 남긴다. (아래 "우리 경로"는 `sgl_kernel.spatial.create_greenctx_stream_by_value`
경유 SGLang PD-mux다 — 프로브는 raw Driver API라 **같은 API 계층이지만 같은 코드 경로는 아니다**.)

★**참고(2026-09-22 코드 근거 감사 `reports/audit/2026-09-22_scope_lineage/REPORT.md` §6 I-1, doc-steward
등재 2026-09-24)**: 이 프로젝트의 config 55/59는 `manual_divisions`(arch 분기 미경유)를 쓰지만 나머지
4개(byte-identical `pdmux_a100_smoke.yml`)는 자동 격자(`divide_sm`, `get_arch_constraints` 경유)를 쓴다.
**VESSL A100은 major 8(cc 8.0)로 KISTI A100과 같으므로 이 상수도 동일하게 적용된다** — 이 실행 계획은
영향받지 않는다. 다른 major(H100/Blackwell)로 갈 때만 §4(대여 전 확인)의 재설계 전제가 걸린다.

**해소된 것**
- 드라이버 **580.105.08 (CUDA 13.0)** ⇒ cu130 스택(torch 2.9.1+cu130·sglang-kernel 0.4.1+cu130) 구동 조건 충족. §1 "드라이버 미확인" 해소.
- A100-SXM4-80GB, 108 SM, MIG Disabled, max SM clk 1410 MHz, power limit 400 W, K8s 컨테이너.
- green context 생성·스트림 실행·**단일 파티션 SM 격리 성립**(864 블록을 띄워도 16 SM에서만 실행). ⇒ KISTI에서의 기존 검증
  (`results/bcg_probe`, `results/smid_census`, `green_readout.py`)과 같은 결론이 이 기판에서도 **단일 파티션 수준으로는** 재현.
- **클럭 고정·전력/persistence/ECC 변경 불가, ncu 카운터(`ERR_NVGPUCTRPERM`)·nsys `--gpu-metrics-devices` 불가** ⇒ §6의 "추정"이
  실측으로 확정. engine-port의 `src/`·`scripts/`·`benchmarks/`에는 `ncu`·`-lgc/-pl/-pm` 사용이 **0건**(grep) — 우리 측정 경로는 영향 없음.

**여전히 미확인(B3/B4 전 프로브 필요)** — 문서 §6과 우리 체크리스트 B3가 겹치는 부분
1. cc 8.0 파티션 입도(홀수·소수 요청의 올림). 우리 격자 (92,16)(84,24)(74,34)(64,44)는 전부 짝수 ≥16이라 **16만 확인된 현재로선
   24·34·44·64·74·84·92가 요청대로 잡히는지 미확인** — realized 확인 필수(교훈: pin은 target이 아니라 realized로 검증).
2. **두 파티션 동시 실행 시 smid 비중첩** — PD-mux 성립의 직접 근거. 미확인.
3. **CUDA Graph 캡처·replay 중 격리 유지** — KISTI에선 P0-A로 확인됐으나 이 기판에선 미확인. 운영점이 cudagraph-ON이라 필수.
4. nsys CUPTI Activity 트레이스 — 우리가 쓸 계획은 없으나 kernel_mech류 재개 시 필요.
5. MPS — 우리 PD-mux 경로와 무관. 기준선에 MPS arm을 넣을 경우에만 별도 모드로.

**계획에 반영할 변경**
- `job_entry.sh`: ① `nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active
  --format=csv -lms 100`을 실행 내내 백그라운드 기록 → throttle reason ≠ 0 구간이 있으면 run에 플래그
  ② `substrate.json`에 **GPU UUID**·`clocks.max.sm`·`power.limit`·`nvcc --version` 추가, 이전 실행과 UUID/드라이버가 다르면 경고
  ③ 긴 스윕은 **이미 끝난 셀 건너뛰기(재개)** — Workspace는 다른 노드에서 재시작될 수 있고 Job은 재시도가 없다.
- **`%smid`는 재매핑되지 않는다**(16 SM 파티션 → 28–43). 우리 `smid_l0_census.py`는 파티션 크기를 `len(set(union))`로 세고
  0-기반 연속을 가정하지 않는다(0-기반 `range()`는 selftest 합성 데이터에만 있음) ⇒ 호환. `mamba_ssm`은 이미지에서 제외돼 있고
  우리 실행 경로가 쓰지 않음(`ssm_kernel_sm_partition_2026-09-21.md` §2) ⇒ 문서가 경고한 smid/SM 수 기반 분배 문제는 현 경로에 해당 없음.
- `substrate` 비교: KISTI A100 측 동일 항목(모델·clk·power limit·드라이버)을 이양 번들 `a_substrate`에서 대조해 기판 동등성 판단
  자료로 쓴다(판정은 사용자 + 규칙층 감사).

**⚠ 정본과 어긋나는 서술 1건 → 해소(2026-09-24, 사용자 확인: 새 방향 아님, 실행 검토용 문구) — `VESSL_A100_CONTEXT.md` §1을 whole-phase 단일 파티션 + layer type은 offline cost model 입력으로만 쓴다는 문구로 교정함.** 원 지적:: 문서 §1 연구 목적의 "layer type(attention vs SSM)에 따라 SM
할당을 다르게 가져가는 정책을 평가"는 정본 결론 "**layer-type 런타임 정책은 全형태 死**(서빙 직접 측정, CLAUDE.md·CONSENSUS)"와
충돌한다. "SSM decode의 SM floor" 같은 **특성화 측정**은 정본과 양립하지만, layer-type **정책**을 다시 평가하려면 기판이 바뀐
재측정이라는 명시적 사용자 결정 + 사전등록이 필요하다. 이 문서를 코드 작성 지침으로 쓰기 전에 §1 문구를 정리할 것을 권고.
