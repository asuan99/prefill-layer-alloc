# engine-port environment and reproduction handoff

연구 결론과 다음 실험의 정본은
[`../../PROJECT_STATUS.md`](../../PROJECT_STATUS.md)와
[`reports/paper/`](../../reports/paper)다. 이 문서는 환경 복구만 다룬다.

## Runtime

- editable tree: `/scratch/ehmoon/whlee/sglang_engine_dev/python`
- virtualenv: `/scratch/ehmoon/whlee/sglang_engine_venv`
- SGLang base: v0.5.10
- cluster stack: PyTorch 2.9.1+cu130, CUDA 13, A100 108 SM
- Zamba2 serving: bfloat16, Triton attention backend

실행 전에 tracked PD-mux source와 thread-local role patch를 적용하고 hash
manifest를 만든다.

```bash
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh
python -m unittest discover -s workspace/engine-port/tests -v
```

CPU 회귀는 **노드 구분 없이** 돌려도 된다 — 로그인 노드에서 **140 tests OK
/ 약 73초**(대부분이 `import sglang.srt.managers.scheduler` 70.6초, 2026-08-11
세션 측정). "컴퓨트 노드에서 돌려야 한다"는 재현되지 않았다; 과거 관측된
7분 stall은 **Lustre 콜드 캐시로 추정**되며 코드 사실로 확정된 것은 아니다.
별개로 `results/p1_gates/gate2/g2s_run.sbatch:91`은 **이미 컴퓨트 노드에서**
전체 회귀를 차단 게이트로 돌린다(실패 시 `exit 5`) — 이 sbatch를 제출하면
로그인 노드 실행과 별개로 컴퓨트 노드에서도 같은 회귀가 다시 통과해야 한다.

`sync_engine_tree.sh`는 dual runtime, mixin, profile, controller, telemetry와
`parallel_state.py` thread-local role patch를 설치한다. 실험 manifest에는
생성된 SHA-256 파일을 보존한다.

## 하드웨어 노드 이질성 (2026-08-14 신설)

★**로그인 노드 `glogin01`과 컴퓨트 노드(`amd_a100nv_8` 파티션, gpu36–43 등)는
GPU SKU가 다르다** — `glogin01`은 **A100 80GB PCIe**(1935 GB/s), 컴퓨트
노드는 **A100 80GB SXM4**(2039 GB/s, 400W TDP+NVLink)다. 컴퓨트 노드끼리는
동질(`scontrol show node gpu36 gpu40 gpu43` 전부 `AvailableFeatures=
A100-80GB_8,hwperf`). 발견 경위: `workspace/engine-port/results/
smsplit_realized/PREREG_SMSPLIT_REALIZED_2026-08-14.md`(addendum 1·2)가
두 노드에서 `torch.cuda.get_device_name()`을 각각 기록해 차이를 드러냈고,
이 오식별이 `results/s8_scaleup/TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md`에
실제로 들어가 있었음을 §11 addendum으로 정정했다(사양 BW 1935→2039 GB/s,
성능 판정은 불변).

**실무 규칙**: green-context 생성 등 **드라이버 층만 건드리는 프로브는
SLURM 할당 없이 `glogin01`에서 검증 가능**하다(비용 0) — 단 **하드웨어
사양(대역폭·클럭·SM 개수 등)에 의존하는 판정은 반드시 컴퓨트 노드에서
재확인**하라. `nvidia-smi -q` 등 하드웨어 사양 인용은 **측정이 실제로
실행된 노드**에서 읽어 job 아티팩트에 기록해야 한다 — 로그인 노드에서
읽은 값을 컴퓨트 노드 캠페인 문서에 쓰지 말 것(`reports/CONSENSUS.md`
§3 항목46 추가 사례, `PROJECT_STATUS.md` "방법론 게이트" #32 参照).

## Runtime modes

| 환경 변수 | 의미 |
|---|---|
| unset | legacy baseline |
| `PDMUX_DUAL_WORKER=1` | R1 observer 재현 전용; architecture separation 아님 |
| `PDMUX_TRUE_DUAL_WORKER=1` | 두 host issue thread를 사용하는 R2 experimental path |
| `PDMUX_R2_POLICY=fixed\|generic\|hybrid` | safe-boundary controller 선택 |
| `PDMUX_MODEL_PROFILE=<json>` | Hybrid policy의 versioned offline profile |
| `PDMUX_TELEMETRY_PATH=<path>` | 모든 arm에 같은 async telemetry |
| `PDMUX_RUN_ID`, `PDMUX_WORKLOAD_ID` | trace 식별자 |

True dual은 thread-local role capability가 없으면 fail-fast한다. GPU correctness
gate를 통과하기 전에는 기본값을 켜지 않는다.

## SLURM `--comment` (제출 필수)

2026-08-12 18:00 KST 이후 뉴론은 응용 분야·프로그램명이 없는 job을 **제출 거부**한다.
모든 job script는 `#SBATCH` 블록 안(첫 실행 라인 위)에 아래를 갖는다:

```bash
#SBATCH --comment="field=efficientai;appl=pytorch"
```

`field=efficientai`(Efficient & Scalable AI Systems) · `appl=pytorch`(SGLang은
`showappl` 목록에 없어 PyTorch 기반으로 신고 — `vllm`은 다른 엔진이라 쓰지 않는다).
인터랙티브는 `salloc ... --comment="field=efficientai;appl=pytorch"`.
**CLI에서 `--comment`를 덮어쓰지 말 것** — script directive가 있어도 CLI 인자가
이를 무효화해 제출이 거부된다. **새로 만드는 job script도 같은 형식**을 따른다.
저장소 내 job 제출 경로 124개(`.sbatch` 97 + `.sh` 26 + `.tmp` 1 + CLI 4경로
`submit.sh`/`run_pipeline.sh`/`submit_size_sweep.sh`/`interact_scheduler.sh`)는
2026-08-14 커밋 `31b3e96`으로 전부 적용됐다. 검사·정정(`86179ec`, 내용 기반
스캔이라 `.sh`로 저장된 job script도 잡는다):

```bash
python3 workspace/engine-port/scripts/bootstrap/check_sbatch_comment.py [--fix]
```

역사적 `PREREG_*.md`가 옛 형식(`--comment=pytorch`)을 그대로 인용하는 것은
그 시점 사전등록 기록을 보존하기 위한 **의도적 조치**이며 손대지 않는다.

## R2 campaign

```bash
workspace/engine-port/scripts/r2_eval/generate_campaign.sh
PDMUX_CAMPAIGN=workspace/engine-port/results/r2_eval/campaign.json \
  sbatch --array=0-<N-1> workspace/engine-port/scripts/r2_eval/r2_eval.sbatch
```

기본 runner는 CUDA Graph를 켠 SGLang server와 immutable SSE trace load
generator를 사용한다. 필요하면 `PDMUX_R2_RUNNER` 또는
`PDMUX_ENGINE_BENCH_RUNNER`로 pinned adapter를 교체한다. 각 array entry는
immutable run record, source hash, telemetry를 별도 디렉터리에 남긴다.
과거 manual dev-tree edit 목록은
[`env/dev_tree_edits.md`](env/dev_tree_edits.md)에 역사 기록으로 유지한다.
