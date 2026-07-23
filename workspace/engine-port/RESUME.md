# engine-port environment and reproduction handoff

연구 결론과 다음 실험의 정본은
[`../../PROJECT_STATUS.md`](../../PROJECT_STATUS.md)와
[`reports/paper/`](reports/paper/)다. 이 문서는 환경 복구만 다룬다.

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

`sync_engine_tree.sh`는 dual runtime, mixin, profile, controller, telemetry와
`parallel_state.py` thread-local role patch를 설치한다. 실험 manifest에는
생성된 SHA-256 파일을 보존한다.

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
