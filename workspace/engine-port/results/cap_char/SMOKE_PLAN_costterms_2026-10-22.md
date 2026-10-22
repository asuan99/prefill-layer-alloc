# Workspace smoke 절차서: 비용 항 측정 하네스 (stage 1) — 2026-10-22

> 지위: 절차서 초안(CPU 문서 작업). **비측정·인용 불가** — smoke에서 나온 어떤 수치도 성능 주장·사전등록 입력 근거가 아니다(ramp 길이는 "상한 재설정의 설계 입력"으로만 쓰며, 값은 인용 금지 프로브 표지를 단다). 제출·Workspace 시작은 이 문서가 하지 않는다: `pdmux-probe` 시작은 비용 행위라 사용자 승인 뒤 그 명령 하나에만.
> 근거: PREREG_COSTMODEL_TERMS_MEAS_rev3 §7-4(smoke 상한 약 1.5 h [유도]), §12 항목 11·13, 결정 29(O-18 smoke 후 ramp 재설정). 하네스 = `costterms_harness/`(engine-porter 소유, 이 절차서는 읽기만).

## 1. 목적 (이 4가지만; 그 밖은 하지 않는다)
1. **실제 ramp 길이 측정** — T1 burst에서 첫 decode_forward 이후 prefill batch가 끝날 때까지의 decode step 수(`ramp_records`, `ramp_steps_ub`와의 비교)를 실엔진에서 관측해, 등록된 ramp 상한(σ = 1, 가짜 엔진은 상한의 약 25–45%였음)이 보수적인지 재설정할 입력을 얻는다. `RAMP_EXCEEDED_BOUND` 여부를 기록.
2. **J-B2 16k 채우기 완료 여부** — T4 채우기(L = 16k, cap 48/CA/CB 중 최소 1 boot)가 부팅 풀 안에서 끝까지 채워지는지, 시간(≈ 6 min/회 가정 대비).
3. **libnuma** — Job 경로가 이미지 `…@sha256:1e0d335b…`(libnuma1 포함)에서 `import sgl_kernel`을 통과하는지. Workspace는 구 이미지(`@c85fc198`)일 수 있어 apt 편차가 생기므로 **Workspace 편차와 Job 경로 사실을 분리해 기록**한다(아래 §3).
4. **실엔진 기록 스키마 확인** — 가짜 엔진이 만든 `events.jsonl`/셀 기록/`DONE`/`SHA256SUMS`가 실엔진 telemetry(TEL phase event 쌍, decode_forward 레코드, 배너 s·k·층 수, 실현 SM)와 같은 필드·단위로 나오는지, `ct_runner.py analyze`가 실데이터에서 에러 없이 도는지(수치 판정 없음).

## 2. 구성 (최소)
- 호스트: Workspace `pdmux-probe`(A100 x1). 측정 Job이 아니다. 끝나면 `vesslctl workspace pause`, 시작 전 `vesslctl workspace list`로 방치 확인.
- 코드: 커밋된 HEAD를 번들로 받아 `install_runtime.sh`(sync + manual edits) → **ENGINE_DIGEST(현행 30엔트리 `99bbef91…`) 일치 확인을 먼저**(하네스에 확인 단계가 아직 없으므로 `engine_digest.py check --manifest … --sealed-file …`를 손으로 실행; 불일치면 중단).
- libnuma 확인: `ldconfig -p | grep libnuma`. 없으면 `apt-get update && apt-get install -y libnuma1`을 하고 `SMOKE_DEVIATION.txt`에 쓴다(2026-10-04·10-20·10-21 선례).
- 실행 방법: `costterms_job.sh`는 봉인 전 거부(rc 16)이므로 smoke에서 `PDMUX_COSTTERMS_SEALED=1`을 **켜지 않는다**. 대신 `ct_runner.py run`을 직접 부르되 이 명령도 SEALED를 요구하므로, smoke 전용 우회가 필요하다 → 하네스 변경 요청 H-1(§5). 우회가 들어오기 전에는 smoke를 시작하지 않는다.
- boot 최소 구성(3 boot):
  - **S-A** Nano, PD-mux, 고정 D = 44, cap 128, cudagraph ON: T1 셀 (8, 2k)와 (32, 2k) 각 1회(ramp 관측용, O가 가장 작은 것), T2 1개 L, T2′ 1개 L. TEL(계측) ON.
  - **S-B** J-B2 T4: `jb2_capCA_L16384`(또는 L = 16k에 해당하는 boot 1개)만 — 채우기 16k 완료·시간·풀 크기 확인.
  - **S-C** J-H1 변형 a4 dummy(vanilla) 1 boot: 변형 디렉터리 생성(`prepare_variants`)과 dummy 부팅·T1 (8, 2k) 1회.
- 모든 boot은 `PDMUX_HEARTBEAT_S`와 무관(Workspace). 서버 로그·`clk.csv`·`nvidia-smi` 스냅샷을 남긴다.

## 3. 판독 항목 (기록만, 판정 없음)
| 항목 | 기록 |
|---|---|
| ramp | 셀별 `ramp_records`, `ramp_steps_ub`, 라벨(`RAMP_EXCEEDED_BOUND` 여부), burst의 prefill batch 수 m, 첫 decode_forward까지 시간 |
| 16k 채우기 | 채우기 완료 여부·소요·풀 토큰·`INFEASIBLE_BY_RULE` 판정이 부팅 풀 값으로 재평가된 결과 |
| libnuma | Workspace: 설치 전 import 결과(실패 시 stderr 마지막 줄)·설치 여부. Job 경로: 이미지 digest가 `1e0d335b`임을 `job show`/`PDMUX_IMAGE`로 확인(실행 없이 확인 가능한 사실) |
| 스키마 | 각 산출물 필드 목록과 가짜 엔진 기록의 차집합; analyzer 에러 0 |
| 부수 | 배너 s·k·층 수 vs 등록 조성표, 실현 SM = 요청 SM(D 44 → 64/44), 부팅 시간, 셀 1개 실제 소요(§7-1 모형 "셀 오버헤드 15.5 s" 대조용 설계 입력) |

## 4. 상한·종료
- 전체 상한 **약 1.5 GPU-h**(PREREG §7-4). 자체 상한: 3 boot 합쳐 90 min `timeout`; boot 하나가 25 min 안에 ready 로그를 못 내면 그 boot은 실패로 기록하고 다음으로 넘어간다(원인 분류만: OOM / CUDA graph capture / green ctx / libnuma / triton cache).
- 종료: 산출물 `sha256sum`, 로컬로 회수(`vesslctl` 볼륨 다운로드, 결과는 `results/cap_char/costterms_smoke_<UTC>/`), **Workspace pause**. 이 smoke는 `fetch.sh`/`vessl_jobs.tsv` 경로가 아니므로 ledger에 올리지 않고, 첫 줄에 "Workspace smoke, 비측정" 표지를 단다.

## 5. 비측정·인용 불가 명시
- Workspace에서 낸 수치는 측정 결과가 아니다(운영 모델: 측정 = Job만). ramp·시간·풀 값은 상한 재설정과 Job 분할 재확인의 **입력**이고, 결과 문서·COSTMODEL·CONSENSUS에 인용하지 않는다.
- 대조 arm이 같은 Job 안에 있다는 보장이 없고 클럭 비고정이므로 arm 간 비교도 하지 않는다.
- smoke에서 하네스가 고쳐지면 그 수정은 새 커밋이며, 봉인(S1)은 smoke 결과를 반영한 뒤에 한다(§12 항목 11 → 13 순서).
- 이 smoke가 채우는 COSTMODEL 칸: **미지정**(CM-2·CM-4 항 측정 Job의 사전 점검).

## 6. 하네스에 필요한 변경 요청 (engine-porter, 하네스 디렉터리 — 이 문서가 고치지 않음)
- **H-1** smoke 전용 모드: 봉인 없이 Workspace에서 `run`을 허용하되 출력 폴더에 `SMOKE_UNSEALED` 표지를 강제하고 plan의 (boot, 셀) 부분집합 선택(예: `--boots`, `--cells-per-boot N`)을 받을 것. 현재 `ct_runner.py run`은 SEALED/TEST_MODE 외 경로가 없고 셀 부분집합 옵션이 없다.
- **H-2** ENGINE_DIGEST 확인: `costterms_job.sh`에 `engine_digest.py check --manifest ${DATA}/runs/${PDMUX_JOB_NAME}/meta/runtime_source_manifest.sha256 --sealed-file <harness>/ENGINE_DIGEST.txt`와 rc 13(`ABORT_ENGINE_DIGEST`)을 추가(M3 `mg_job_common.sh:151-161` 형식; 현재 harness에는 digest 확인이 전혀 없다). 봉인 파일에는 `99bbef91…`(30엔트리)를 넣고, 확인 전까지 UNSEALED 자리표시.
- **H-3** 원자료 위치: 기본 출력이 `results/cap_char/costterms_runs/<name>`(= `${RUN}/results` 안)라 `job_entry.sh`의 trap이 시작 마커 이후 만든 **모든 파일**(boot별 서버 로그·telemetry 포함)을 `/io/results`로 export한다. 결정 23 관례(큰 원자료는 클러스터 볼륨에만)를 따르려면 원자료를 `/data/runs/<job>/raw/`(`PDMUX_COSTTERMS_OUT`)에 쓰고, `results/` 아래에는 `analysis.json`·`plan.json`·`events` 요약·`SHA256SUMS`·`RUN_DONE`·boot별 DONE manifest(파일명→sha)만 둘 것. `engine_gate_decomp.sh:3`이 같은 방식.
- **H-4** 재개: 현재 boot 단위 `DONE`(`ct_runner.py` run_boot) 재개이고 셀 단위가 아니다. 또 새 Job은 새 job name이라 `${RUN}`이 달라 재개가 되지 않는다 → 재개하려면 `PDMUX_COSTTERMS_OUT`을 고정 경로(`/data/costterms/<job>_r<round>`)로 주고 그 경로를 `launch.sh --env`로 넘겨야 한다. 그 규약과 "재개한 사실을 결과 보고에 적는다"를 문서화하고, 셀 단위 DONE이 필요하면 추가.
- **H-5** `costterms_job.sh`의 J-H2·J-C(2단계) 추가 시 `launch.sh`의 하드캡 표시표(`*/costterms_job.sh` 블록, 현재 J-H1 245 · J-A r1 243 · r2 234 · J-B2 78)도 같이 고칠 것(표시 전용, 실제 cap은 하네스가 강제).
- **H-6** `ct_runner.py`가 쓰는 ramp 상한 σ를 smoke 결과로 재설정할 때 `ct_spec.o_tokens` 변경은 spec digest를 바꾸므로 봉인 전에 한 번에 반영.
