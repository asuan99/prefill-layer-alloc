# BRINGUP Phase B — pdmux-sbase Workspace (2026-11-02)

측정 아님(bring-up). 성능 수치·판정 없음. 아래 시간·ms 값은 첫 호출(cold) 관측이며 어떤 측정 결과로도 인용하지 않는다.

## 환경
- Workspace `pdmux-sbase` = `wsp-3osfxrydotsv`, A100 SXM ×1, 클러스터 `cluster-betelgeuse`(us-west-2), 볼륨 `/data`(pdmux-cs)·`/io`(pdmux-io).
- 이미지: `ghcr.io/asuan99/sglang-vllm-runtime@sha256:a5f9bf0f742de31f0de3f1cc2bc9fb352eec537dd4c746d9753d6d104087ea2b`
- 커밋 d7c6c5b3f08e5fb9ba43ab457b1adc2cf7a34abe — 로컬 `git bundle`을 scp로 올려 `/opt/pdmux/prefill-layer-alloc`에 `git clone --no-checkout` + `checkout`(job_entry.sh step 1과 같은 방식). 작업 트리의 untracked 파일은 번들에 불포함.
- 기판: driver 580.178.04, GPU UUID `GPU-26bdb1ab-b9cb-3278-0415-2e2f9b40572c`, host `main-wsp-3osfxrydotsv-0`. freeze sha: vLLM venv `adff2ff0…`(= `VLLM_VENV_FREEZE_SHA256`와 일치), SGLang `45c39de2…`(= `SGLANG_FREEZE_SHA256`와 일치).
- 모델: `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base` rev `dc0661c8…`, `HF_HOME=/data/hf`, `HF_HUB_OFFLINE=1`(sb_spec와 같은 id+revision 방식).
- 산출물 원본(Cluster 볼륨): `/data/sbase_bringup/`(서버 로그 `srv_*.log`, `tree_*.json` 스탬프, `out_*.json` 출력, `probe_e.jsonl`, `g_*.jsonl`, `install_runtime.log`).
- 소요: Workspace 요청 가능 시점까지 약 21분 재시도(용량 없음 4회 후 5번째 시도에서 슬러그 오류, 아래) → 생성 06:36:41Z → running 약 10분 이내 → 검증 약 25분 → pause 요청 07:04:58Z. **최종 상태: `standby`(paused) 확인** (`vesslctl workspace list`, 3개 모두 standby).

## 용량·생성 기록
- `resourcespec-a100x1` availability: `unavailable`. 시도 06:15:36, 06:20:42, 06:25:58, 06:30:58 UTC 모두 "no available capacity". 06:36:19 시도는 availability가 열려(`low`) 용량 검사를 통과했으나 **`--cluster betelgeuse-na`(이름) 때문에 "Resource not found"**로 실패(생성 안 됨). `--cluster cluster-betelgeuse`(슬러그)로 고쳐 06:36:41에 생성 성공. 지시문의 명령에 있던 클러스터 인자가 슬러그가 아니라 이름이었다(하네스/문서 정정 사항: 앞으로 슬러그 사용). 생성된 Workspace는 1개.
- `--ssh-key`를 붙이지 않았으나 SSH 접속은 로컬 `~/.ssh/id_ed25519`로 됐다(조직 기본 키 추정; 운영 모델 §2 문구 "생성 후 키 부착 불가"와는 이번 사례에서 문제 없음).

## 항목별 결과

| 항목 | 판정 | 근거 |
|---|---|---|
| (a) install_runtime + 보존 | **PASS** | `PDMUX_REQUIRE_PRISTINE=1` install rc=0. `engine_tree_pristine_status.txt`: `status=OK`, `ref_overridden=0`, `origin=copied`, `tree_manifest_sha256=1bd84d884e53468ccfd65f38806ea02cd855cc08d6ac18d31f88bac48a1c3dd3`(= 참조), entries=2258, extras=`sglang/_version.py`만, reason 빈 값. 마커 0은 pristine `boot` 스탬프의 `engine_tree_markers_found=[]`로 별도 확인. patched 트리: sync manifest sha `0c8e5d1a…`, manual edits `applied`. |
| (b) boot --check-only | **PASS** | pristine: `[engine_tree] pristine root=/opt/pdmux/sglang_pristine/python manifest=1bd84d884e53` rc=0, `engine_tree_sglang_init`가 pristine 경로. patched: `root=/opt/pdmux/sglang_engine_dev/python manifest=525c6bce568d` rc=0, markers 6종(필수 3개 포함: `pdmux_role_is_thread_local`·`maybe_create_holb_probe`·`maybe_install_chunk_probe`). pristine+`PDMUX_TRUE_DUAL_WORKER=1` → `ENGINE_TREE_REFUSED … ['PDMUX_TRUE_DUAL_WORKER']` rc=6. 주: `PDMUX_PROJECT_ROOT` 미설정이면 pristine이 rc=6으로 거부(job_entry.sh가 export하므로 Job에서는 무관; Workspace ssh 세션 전용 주의). |
| (c) SG-stock pristine 부팅 | **PASS** | SG-stock argv(radix OFF) pristine 부팅, `/health` 200(139 s), 고정 프롬프트 5개 greedy 32토큰 저장(`out_sgstock.json`). `server_args=` 줄에 `enable_mixed_chunk=False`·`chunked_prefill_size=8192` 키 존재(§5 regex 대상 OK). 선택: `attention_backend='flashinfer'`(`prefill/decode_attention_backend=None`), `mamba_backend='triton'`, `disable_cuda_graph=False`, `cuda_graph_max_bs=256`(capture bs [1,2,4,8,12,16,24,32,40,48], "Capture cuda graph end"), decode 로그 `cuda graph: True`, piecewise는 `--disable-piecewise-cuda-graph`로 꺼짐, `mem_fraction_static=0.836`, overlap ON(`disable_overlap_schedule=False`), `enable_pdmux=False`, `enable_metrics=False`. 출력은 base 모델이라 반복성(예: "Paris. Paris is a city…")이 높음 — 비교 가드로는 유효. |
| (d) `--enable-metrics` | **PASS** | pristine 재부팅(84 s), `sglang:time_to_first_token_seconds` before `{sum 4.5281, count 1}`(warmup 요청 포함) → 요청 1개 뒤 `{sum 4.6617, count 2}`: count +1, sum +0.134 증가. |
| (e) 패치 트리 + 프로브(PD-mux OFF) | **PASS** (관측 2건 공시) | `PDMUX_DECODE_PROBE_PATH`+`PDMUX_PROBE_EXTEND_TIME=1`, patched 부팅(104 s). 프로브 235줄: `decode_forward` 215, `extend_forward` 10, `decode_probe_window` 9, `decode_probe_boot` 1(`extend_time_enabled=true`, `arm_enable_pdmux=false`). `extend_forward`: 요청 512/2048/64토큰에 대응하는 `extend_tokens`=512·2048·64, `n_reqs`=1, `extend_gpu_ms` 전부 양수, `stream_key`=94189604612944 단일값(boot 안 상수). 5개 프롬프트 출력 `output_ids`가 (c)와 **5/5 토큰 동일**. 공시: ① 첫 호출 레코드(`extend_tokens` 7·1·512)의 `extend_gpu_ms`가 2–3.6 s로 큼 — cold(JIT/autotune) 시간으로 추정, 측정은 warmup 뒤 창이어야 함. ② `extend_tokens=2048` 레코드에서 `extend_gpu_ms`(347.8) > `extend_host_ms`(164.3): GPU event 구간이 host 구간보다 길다(overlap 경로에서 선행 GPU 작업 대기 포함 가능성) — 해석은 engine-porter 몫, r_ser 계산 전에 확인 권고. |
| (f) vLLM 0.31.0 | **PASS** (불일치 공시, 프로파일 엔드포인트 조건 공시) | venv 인터프리터·`PYTHONPATH`/`PDMUX_*` 제거(`env -i`), `CUDA_HOME`·`TORCH_CUDA_ARCH_LIST=8.0`, 부팅 `/health` 200(165 s). 로그: `Using FLASH_ATTN attention backend`(후보 FLASH_ATTN·FLASHINFER·TRITON_ATTN·FLEX_ATTENTION), `cudagraph_mode=FULL_AND_PIECEWISE`(PIECEWISE 15 + FULL 캡처), `enable_chunked_prefill=True`, `enable_prefix_caching=False`, compile mode `VLLM_COMPILE`(inductor), Mamba SSU triton, KV 2,442,462 토큰. `max_num_batched_tokens`는 로그에 직접 안 찍힘 → 코드 기본값(A100, OPENAI_API_SERVER)=**2048**(`arg_utils.py` 2879–2884행에서 도출; 로그의 compile range (1, 2048)과 일치) — 이 값은 로그 채취가 아니라 코드 도출임을 표기할 것. `/metrics` `vllm:time_to_first_token_seconds` count 5→6, sum +0.0351(증가 확인). 토큰 일치율 vs (c): 프롬프트별 id 완전 일치 **2/5**(#3, #5), 공통 접두 길이 1·12·32·30·32 / 32(#1·#2·#4 이탈) — 기록만, 실패 아님(엔진·커널 차이로 greedy 이탈 예상 범위; 원인 미조사). `/start_profile`·`/stop_profile`: 기본 부팅에서는 **404 Not Found**(라우터가 `profiler_config.profiler`가 있을 때만 등록, `vllm/entrypoints/serve/profile/api_router.py`). `--profiler-config '{"profiler":"torch","torch_profiler_dir":…}'`로 재부팅하면 둘 다 **200**이고 `.pt.trace.json.gz` 2개·`profiler_out_0.txt` 생성. ⇒ §5-2는 계측 boot 전용으로 이 플래그가 필요하다(측정 arm 인자에는 넣지 않음). |
| (g) sb_loadgen vLLM 클라이언트 | **PASS** | `sb_loadgen.replay("openai", …)`로 `/v1/completions` 스트리밍 1요청(128 입력·16 출력). `t_sent_mono`·`token_t_mono`(16개)·`t_finished_mono` 채워짐, `error=''`, `http_status=200`, `token_itl_ms` 15개, 순서 `t_sent < token[0] ≤ token[-1] ≤ t_finished`. (프로파일러 설정 boot에서 실행; 기본 부팅과 엔진 설정 차이는 profiler config뿐.) |
| (h) SGLang `/generate` 스트리밍 | **PASS** | `sb_loadgen.replay("sglang", …)`(→`ph_loadgen.replay_natural`, `abs_mono=True`) 1요청, pristine 부팅. 같은 레코드 키 모두 채워짐, 오류 0, 순서 OK. (첫 호출이라 TTFT가 크다 — cold, 측정 아님.) |

## 실패 분류
- FAIL 항목 없음.
- 하네스/문서 정정 후보: ① 클러스터 인자는 이름(`betelgeuse-na`)이 아니라 슬러그 `cluster-betelgeuse`(용량 검사가 먼저 통과하면 "Resource not found"로 나타난다). ② `pkill -f sglang`은 같은 ssh 명령줄에 걸려 세션을 죽인다(`[l]aunch_server` 패턴 사용).
- 엔진 쪽 확인 요청(engine-porter): (e) 공시 ②의 `extend_gpu_ms` > `extend_host_ms`.
- vLLM 출력 불일치 3/5는 §5 가드의 "공시" 항목으로만 기록.

## 규약 준수
- Job 생성 0, Workspace 생성 1회(성공 1, 용량 부족/슬러그 오류로 생성 안 된 시도 5회). 측정 수치 보고 없음. secret은 Workspace에 주입하지 않았고 env를 출력하지 않았다.
- 모델 로드는 offline(`HF_HUB_OFFLINE=1`), 캐시(Triton·FlashInfer·vLLM)는 boot별 `/data/sbase_bringup/cache/*`로 분리.
- 이 bring-up 출력은 `/data/sbase_bringup`에 남아 있어 pdmux-build/pdmux-sbase에서 재확인 가능. 커밋하지 않음.

이 Job이 채우는 COSTMODEL 칸: 미지정 (Job 아님, 기준선 Phase B bring-up).
