# VESSL Cloud 운영 모델 — 실행 구성·워크스페이스·Job 전달 (2026-10-02)

> **지위**: 운영 규약 + 스크립트 초안. 새 측정 0 · GPU 지출 0 · Claim 등급 변경 0.
> - VESSL 사실표(가격·볼륨 종류·격리·한도)는 `vessl_cloud_setup_plan_2026-09-24.md` §1·§9–§12에 그대로 둔다.
> - 이 문서가 **대체하는 부분**: 같은 문서 §3(스토리지 배치)·§5(생성 절차의 Job 부분), `vessl_execution_plan_2026-09-24.md` §3-3(Run 템플릿·러너).
> - 근거: 로컬 `vesslctl` 2026.09.22-01의 `--help`·`vesslctl skill show`(VESSL 공식 에이전트 스킬) 원문, 저장소 스크립트 대조, 로컬 모의 실행(§6).
> - 기판 동등성·E2 재등록·예산 시나리오는 **판정하지 않는다**(사용자 결정).

## 1. 실행 구성 판단 (원칙 6개)

1. **측정은 Job으로만 한다. Workspace는 bring-up·프로브·디버깅 전용이다.**
   - Workspace는 사람이 고친 가변 환경이고, 대화형 세션이 같은 GPU를 건드린다.
   - Job은 이미지 digest + 커밋으로 환경이 고정되고, 끝나면 사라져서 "누가 무엇을 바꿨나"가 남지 않는다.
2. **한 비교의 모든 arm은 한 Job 안에서 돈다**(E2-α 방식).
   - VESSL은 공유 노드이고 클럭을 고정할 수 없으며, 드라이버를 플랫폼이 패치한다(`VESSL_A100_CONTEXT.md` §4–5).
   - 따라서 arm 차이와 노드 차이를 분리하려면 같은 노드·같은 드라이버 안에서 arm을 블록 무작위 순서로 섞어야 한다.
   - 반복(n)은 여러 Job으로 채우고, 노드(hostname·GPU UUID)를 공변량으로 남긴다.
3. **셀은 묶되 Job 하나는 짧게 유지한다.**
   - 부팅마다 이미지 pull·모델 로드·cudagraph 캡처 오버헤드가 붙는다. 그래서 셀을 한 Job에 묶는다.
   - 다만 Job은 자동 재시도가 없다. 실패 손실을 제한하도록 **Job당 ≲3–4 GPU-h**로 쪼개고, 끝난 셀 건너뛰기(재개)를 캠페인 스크립트에 둔다.
4. **네 가지 입력을 분리한다.** 각각 바뀌는 속도가 다르다.

   | 입력 | 담는 곳 | 고정 수단 |
   |---|---|---|
   | 엔진+의존성 | 컨테이너 이미지 | `@sha256` digest |
   | 코드·하네스·설정 | 커밋 git bundle → Object 볼륨 | 커밋 해시 |
   | 모델·trace | Cluster 볼륨 `/data/hf` | HF 리비전(`hf_hub_revisions.tsv`) |
   | 결과 반출 | Object 볼륨 `/io/results` | `SHA256SUMS` + `DONE` |

5. **제출은 비용 행위다.**
   - 순서는 `launch.sh`(dry-run)로 명령·단가·job name을 보여 주고 → 사용자가 승인하면 → `--submit`.
   - VESSL 공식 스킬의 규칙("`job create` 전 정확한 명령 + 시간당 비용을 보이고 승인 대기")과 같다.
   - 크레딧을 선불로 충전해 두면 잔액 ≤0일 때 생성이 막힌다. 이것이 사실상의 하드캡이다. 단 실행 중인 Workspace는 −$10까지 과금된다.
6. **회수까지가 한 작업이다.**
   - `fetch.sh`가 DONE 마커와 sha256 검증을 통과해야 "완료"다.
   - Job이 succeeded여도 회수하지 않은 결과는 결과가 아니다.

## 2. 스토리지·워크스페이스 구성

```
Cluster volume  pdmux-cs   (us-west-2 고정 — A100 SXM 유일 리전)          → /data
  /data/hf/                          HF_HOME. 모델(리비전 고정) + raw/ (trace: 기존 스크립트의 $HF_HOME/raw 규약)
  /data/runs/<job_name>/             Job 실행 중 쓰는 곳: meta/ logs/ cache/{triton,xdg,flashinfer} results/
  /data/substrate_history.tsv        Job마다 GPU UUID·드라이버·호스트 1줄 (캠페인 내 드라이버 변경 경고)
  /data/env/                         bring-up용 venv (측정에 쓰지 않음)
Object volume   pdmux-io   (setup plan의 pdmux-out을 양방향으로 개명)      → /io
  /io/code/<commit>/{repo.bundle, repo.bundle.sha256, job_entry.sh}         로컬 → Job
  /io/results/<job_name>/{DONE, SHA256SUMS, meta/, logs/, results/}         Job → 로컬
```

- **Cluster 볼륨**은 빠르고(CephFS) 리전에 묶이며, 로컬 CLI로 직접 올리거나 받을 수 없다. 그래서 대용량 입력(모델)과 실행 중 쓰기에 쓴다.
- **Object 볼륨**은 `vesslctl volume upload/download`가 되는 유일한 경로다. 그래서 로컬과 주고받는 창구로 쓴다.
- telemetry를 S3 FUSE에 직접 쓰면 관측자 효과가 생긴다. 그래서 Object 볼륨에는 Job이 끝난 뒤 한 번만 복사한다.

| Workspace | 스펙 | 용도 | 규칙 |
|---|---|---|---|
| `pdmux-build` | CPU-only, 이미지 = 측정 이미지 digest | 모델·trace를 `/data/hf`에 리비전 고정으로 적재, CPU 회귀, 이미지 동작 확인 | Workspace에는 secret 주입이 안 된다. gated 모델 토큰은 수동 `hf auth login` 후 반드시 logout |
| `pdmux-probe` | A100 SXM ×1 | B0 기계 사실 · B3 SM 입도 realized probe · B4 부팅 스모크 · 실패 Job 재현 디버깅 | 작업이 끝나면 **즉시 pause**. GPU 시간 과금 |

- 두 Workspace 모두 생성 시 `--ssh-key`를 붙인다. 생성 후에는 키를 붙일 수 없다(VESSL 스킬 원문).
- 세션을 시작할 때 `vesslctl workspace list`로 켜진 Workspace가 방치돼 있지 않은지 확인한다.

## 3. Job 전달 방식 (`workspace/engine-port/scripts/vessl/`)

```
[로컬] 커밋 ──► launch.sh (dry-run: 명령·$/h·job name 출력)
                 └─ 사용자 승인 ─► launch.sh --submit
                        ├─ tracked 변경 있으면 거부 (HEAD만 실행)
                        ├─ git bundle(HEAD) + job_entry.sh → /io/code/<commit>/ (이미 있으면 생략)
                        ├─ vesslctl job create  (이미지 digest, /data·/io 마운트, PDMUX_* env, tag=campaign)
                        └─ results/<campaign>/vessl_jobs.tsv 에 1줄 (커밋 대상)
[VESSL Job] job_entry.sh
   1 bundle sha256 확인 → 정확한 커밋 checkout (PDMUX_COMMIT과 일치 검사)
   2 engine-port/results → /data/runs/<job>/results 심볼릭 (추적 파일 보존, 쓰기는 영속 볼륨으로)
   3 sync_engine_tree.sh (이 커밋의 src로 엔진 트리 재동기화, manifest 기록)
     devtree_manual_edits.patch가 이미지 빌드 때와 다르면 중단 → 이미지 재빌드 필요
   4 substrate.json (GPU 이름·UUID·드라이버·cc·클럭·전력·MIG·호스트·이미지·커밋) + clk.csv(100 ms) 백그라운드
   5 HF_HOME=/data/hf (offline), Job별 triton/flashinfer 캐시 → 대상 스크립트 실행 (PDMUX_ARRAY_SPEC면 array_runner)
   6 trap: 성공·실패와 무관하게 meta/logs + 이번 실행이 만든 results 파일만 /io/results/<job>/ 복사
          → SHA256SUMS → DONE(마지막)
[로컬] vesslctl job show/logs <name> 로 모니터 ──► fetch.sh <campaign> <job_name>
         ├─ DONE 없으면 "미완" · sha256 실패면 중단
         ├─ results/ 병합 (기존 파일은 덮어쓰지 않음, 내용이 다르면 충돌 보고 후 중단)
         └─ meta·logs → results/<campaign>/vessl/<job>/, cost.txt(컨테이너 시간×단가 하한), ledger fetched=yes
```

- **secret은 Job에 넣지 않는다.** 모델은 미리 `/data/hf`에 올려 두고 offline으로 돈다.
  - `launch.sh --env`는 `PDMUX_*` 노브만 받는다.
  - `job_entry.sh`는 env를 덤프하지 않는다. env에는 `VESSLCTL_ACCESS_TOKEN`이 들어 있다.
- **재실행**: 자동 재시도가 없다. 재실행은 새 job name(커밋·시각 포함)으로 하고 ledger에 남긴다. 실패 셀만 골라 다시 돌린 사실은 결과 보고에 반드시 적는다(사후 선택 방지).
- **이미지를 재빌드할 때**: `devtree_manual_edits.patch`·`requirements.pdmux-sglang.txt`·SGLang 버전·base digest가 바뀐 경우다.
  - `src/` 변경은 Job 안의 런타임 sync가 반영하므로 재빌드가 필요 없다.
  - 2026-10-02 확인: draft2(빌드 커밋 `b1c739c`) 이후 `env/`(docker 제외)·`src/`·`scripts/bootstrap/` 변경이 0이라 draft2 내용은 여전히 유효하다.
- **비용 장부**: `vessl_jobs.tsv`(제출)와 `cost.txt`(실행 시간 하한)를 쓴다. 스케줄링·이미지 pull 중 과금은 컨테이너 시간에 안 잡히므로 `vesslctl billing show`로 대조한다.

## 4. 기존 캠페인 스크립트 이식 (실행 전 필수)

- `r2_eval.sbatch`처럼 **env로 경로를 받는 스크립트**(`PDMUX_PROJECT_ROOT`/`PDMUX_ROOT`, `PDMUX_RESULT_ROOT`)는 그대로 돈다.
- **KISTI 경로·`module load`·venv `source`가 하드코딩된 스크립트**는 그대로 돌지 않는다. 예: `results/r2_eval/e2_sticky_prereg/e2_sticky.sbatch:46–51`.
  - 이식 방법: `ROOT="${PDMUX_PROJECT_ROOT:?}"`로 바꾸고 `module`/`source` 줄을 제거한다(이미지가 환경을 제공한다).
  - 이것은 사전등록된 스크립트의 변경이므로 **새 캠페인 = 새 사전등록**의 일부로 다룬다(E2 이식성 `REFUTED`와 같은 원리).
- `job_entry.sh`가 `$PROJECT/hf_cache → /data/hf` 심볼릭을 만들어 옛 `$ROOT/hf_cache` 규약도 받는다.

## 5. 현재 상태와 블로커 (2026-10-02 실측)

| 항목 | 상태 | 해소 |
|---|---|---|
| `vesslctl` | **로그인 완료**(org `MLSysLab`, team `HybridLLM`, 토큰 만료 2026-10-03 15:35 KST). 볼륨 0·Workspace 0 | 만료되면 다시 `vesslctl auth login` |
| 이미지 | 로컬 draft2(26.7 GB, label commit `b1c739c`). wonho가 docker 그룹에 들어감(2026-10-02) | §6-1의 3단계로 HEAD 기준 재빌드 |
| **레지스트리** | **GHCR private으로 결정(2026-10-02)** — push 전까지는 여전히 블로커 | §6-1 절차 |
| 볼륨 | `pdmux-cs`·`pdmux-io` 생성 여부 미확인 | 로그인 후 `vesslctl volume list` |
| 예산·크레딧 | 시나리오 미결정(`setup_plan` §12: A ≈$10–15 … D ≈$35–55) | 사용자 결정 |
| 스크립트 | `job_entry.sh`·`launch.sh`·`fetch.sh` 작성, 로컬 모의 실행 통과(§6) | 첫 실제 Job(B5)으로 검증 |

## 6. 스크립트 검증 수준 (정직하게)

- **로컬 모의 실행(2026-10-02)**: 가짜 `nvidia-smi`·`nvcc`·`vesslctl`과 임시 볼륨 디렉터리로 돌렸다.
  - 정상 경로: 정확한 커밋 checkout, array 2셀 실행, 새 결과 파일만 반출, `SHA256SUMS` 통과, env 유출 0.
  - 실패 경로: 대상이 없어 rc=2여도 DONE·반출이 수행됐다.
  - fetch: 병합 성공, 내용이 다른 기존 파일은 충돌 보고 후 rc=4로 멈췄다.
  - launch: dry-run 출력, `PDMUX_*`가 아닌 env 거부, 빈 설정 거부를 확인했다.
- **실제 VESSL에서 미검증**(B5에서 확인할 것):
  - `volume download`가 `--remote-prefix`를 벗기는지(fetch.sh는 두 경우를 모두 처리한다)
  - tag 문자 규칙(밑줄)
  - Object 볼륨 FUSE 위의 rsync·sha256 동작
  - `FLASHINFER_WORKSPACE_BASE` 변수명(flashinfer JIT 캐시 분리)
  - Job 컨테이너의 `/opt/pdmux` 쓰기 가능 여부
  - `nvidia-smi -lms` 동작

## 6-1. 이미지 레지스트리 = GHCR private (2026-10-02 사용자 결정)

**요금(GitHub 공식 문서 `billing/concepts/product-billing/github-packages`, 2026-10-02 확인)**
- "Container image storage and bandwidth for the Container registry is **currently free**." 정책이 바뀌면 최소 1개월 전에 통지한다고 적혀 있다.
  ⇒ 27 GB private 이미지라도 **지금은 추가 요금이 없다.**
- 일반 GitHub Packages 한도(Free 500 MB 저장·월 1 GB 전송)는 Container registry에는 현재 적용되지 않는다.
  통지가 오면 재검토한다(대안: Docker Hub, 또는 VESSL 쪽 레지스트리).
- VESSL 쪽: 이미지 pull은 Job 시간(=GPU 과금) 안에 일어난다. 노드 캐시(`IfNotPresent`)가 없으면 Job마다 수 분이 붙는다. 이것이 Job을 묶는 이유 중 하나다(§1-3).

**제약(GitHub 문서 `working-with-the-container-registry`)**
- **레이어당 10 GB 한도**: 최대 레이어가 5.73 GB(비압축, base의 torch 계층)라 통과한다.
- **업로드 10분 타임아웃(레이어당)**: 압축 후 수 GB 레이어를 10분 안에 올리려면 대략 40–80 Mbps 이상의 업로드가 필요하다(압축 크기·회선 미측정).
  첫 push에서 타임아웃이 나면 두 가지를 먼저 확인한다: 그 레이어만 재시도되는지, docker의 `max-concurrent-uploads`를 1로 낮춰 대역을 몰아줄 수 있는지.
- 새로 push한 패키지는 기본이 **private**이고 저장소와 자동 연결되지 않는다. 그래서 Dockerfile에
  `org.opencontainers.image.source=https://github.com/asuan99/prefill-layer-alloc` 라벨을 추가했다.

**push 결과(2026-10-02)**: `ghcr.io/asuan99/pdmux-sglang:b27c527` → digest `sha256:f3166a7a4427bf87d7417387c222162a854ed3ba64d3c1b44c11f32a439536a4`
(빌드 커밋 `b27c527`, SGLang `1519acf3`). push는 약 4분 걸렸고 10분 타임아웃에 걸리지 않았다. `scripts/vessl/vessl.env`에 기록했다(볼륨 slug는 아직 placeholder).

**절차** (토큰 값은 사용자만 입력한다 — 대화·파일·커밋에 남기지 않는다)
1. (사용자, GitHub) Personal access token (classic)을 **1개** 만든다(2026-10-02 사용자 결정: 분리하지 않음).
   - scope는 `write:packages` 하나만 체크한다(읽기 권한 포함). `repo`·`delete:packages` 등 나머지는 체크하지 않는다.
   - 만료는 실험 기간(예: 90일)으로 두고, 실험이 끝나면 폐기한다.
   - 같은 토큰을 로컬 push와 VESSL pull 등록에 함께 쓴다. VESSL에서 유출되면 이미지를 덮어쓰는 것까지 가능하다는 점은 감수한다.
2. (로컬) `echo <PAT> | docker login ghcr.io -u asuan99 --password-stdin`
   - 사용자가 직접 입력한다. 셸 히스토리에 남지 않게 앞에 공백을 붙이거나 `read -s`를 쓴다.
3. (로컬) 깨끗한 HEAD에서 이미지를 다시 빌드한다(라벨 커밋을 정확히 남기기 위해):
   `bash workspace/engine-port/env/docker/build_image.sh ghcr.io/asuan99/pdmux-sglang:<commit7>`
   - 엔진 입력이 draft2와 같으므로 마지막 계층들 외에는 캐시를 탄다.
4. (로컬) `docker push ghcr.io/asuan99/pdmux-sglang:<commit7>`
   - 첫 push는 base 계층 포함 ≈23–27 GB(비압축 기준)다.
   - 끝나면 digest를 확인한다: `docker inspect --format '{{index .RepoDigests 0}}' ghcr.io/asuan99/pdmux-sglang:<commit7>`
5. (VESSL) Job이 private 이미지를 받는 방법 — ★**미해결**. 2026-10-02 현재 VESSL Cloud 공식 문서(job/workspace 생성·secrets·문서 색인)
   어디에도 private registry 자격증명 등록 방법이 없다. 폼에는 Managed/Custom 탭만 있고, `vesslctl job create`에도 관련 플래그가 없다.
   ⇒ `vessl_cloud_setup_plan_2026-09-24.md` §1의 "Job은 private registry 자격증명·pull policy 지원"은 **현재 문서로 확인되지 않는다**(정정).
   진행 순서: (a) 콘솔 Job 생성 화면의 Custom 탭과 org Settings에서 해당 메뉴를 직접 확인한다. (b) 메뉴가 없으면 GHCR 패키지를 public으로 전환하거나(엔진 패치 공개, 사용자 결정) VESSL 지원팀에 문의한다.
6. `scripts/vessl/vessl.env`의 `PDMUX_VESSL_IMAGE=ghcr.io/asuan99/pdmux-sglang@sha256:<digest>`를 채우고 커밋한다(digest만 들어가고 비밀은 없다).

## 7. 첫 실행까지의 순서

1. (사용자) `vesslctl auth login`을 하고 org/team을 정한다. 레지스트리를 결정하고 docker 그룹을 추가한다.
2. (비용 0) `vesslctl resource-spec list --usable-only -o json`으로 `a100-sxm-1`·CPU-only slug와 단가를 확정한다.
   그다음 `vessl.env.example`을 `vessl.env`로 채워 커밋한다.
3. (스토리지 과금 시작) `pdmux-cs`(콘솔, us-west-2)와 `pdmux-io`를 만든다.
4. 이미지를 push하고 digest를 확정한다 → `pdmux-build`에서 모델(Nano-9B-v2 등 현 트랙만)과 trace를 `/data/hf`에 적재한다 → CPU 회귀를 돌린다.
5. (첫 GPU 지출) `pdmux-probe`에서 B0·B3(24·34·44·64·74·84·92 SM이 요청대로 잡히는지 realized 확인, 두 파티션 smid 비중첩, cudagraph 중 격리)·B4를 하고, 즉시 pause한다.
6. **B5**: 소형 스모크 Job 하나를 `launch.sh`로 제출 → `fetch.sh` 통과까지. **B5 통과 전에는 캠페인을 제출하지 않는다.**
7. 기판 재앵커(λ0 재측정 등)와 E-1 사전등록·규칙층 감사 → 캠페인.

## 8. 사용자 결정 사항

1. ~~레지스트리~~ → GHCR private으로 결정(2026-10-02)
2. 예산 시나리오와 크레딧 충전액
3. 모델 업로드 범위(현 트랙만 권고)
4. 기판 동등성 처리와 E2 새 OVERRIDE(기존과 같음)
