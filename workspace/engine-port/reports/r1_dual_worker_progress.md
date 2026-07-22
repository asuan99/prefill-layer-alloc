# R1 dual-worker 정리 및 실행 진행 보고서

작성 기준: 2026-07-23

## 목적

기존 PD-mux scheduler 안에서 prefill/decode 역할별 상태와 공유 GPU
partition lease를 기록하는 실험 경로를 검증한다. 이 변경은 기본 경로를
바꾸지 않으며 `PDMUX_DUAL_WORKER=1`일 때만 활성화된다.

## 산출물 위치

| 구분 | 경로 | 목적 |
|---|---|---|
| runtime source | `src/multiplex/dual_worker.py` | worker state, phase coordinator, GPU arbiter |
| scheduler integration | `src/multiplex/multiplexing_mixin.py` | 기존 event loop에 opt-in 연결 |
| test | `tests/test_dual_worker.py` | CUDA 없이 phase/lease/telemetry 회귀 검증 |
| launcher | `scripts/r1_dual_worker/*.sbatch` | direct 및 no-SMI 제출 |
| environment | `scripts/bootstrap/install_engine.sh` | CUDA stack 보존형 의존성 bootstrap |
| reports | `reports/r1_dual_worker/job_<id>.md` | job별 상태와 실패 원인 |
| raw results | `results/r1_dual_worker/job_<id>/` | manifest, server log, trace, SLURM stdout/stderr |

## 실행 결과

| Job | 상태 | 핵심 원인 | 보존 위치 |
|---:|---|---|---|
| 862456 | FAILED | `openai_harmony` 누락, port `37056` 충돌 후 server 종료 | `job_862456/` |
| 862501 | FAILED | 이전 제출본의 `SCRIPT_DIR` 미정의 (`set -u`) | `job_862501/` |
| 862502 | FAILED | 제출 시점의 이전 script 경로가 `/var/spool/...`에서 사라짐 | `job_862502/` |

위 세 job은 end-to-end 성능 결과가 아니며, 정책 비교의 근거로 사용하지
않는다. 실패 로그와 자동 보고서는 재현성·디버깅 자료로만 보존한다.

## 현재 코드 상태

- `PhaseCoordinator`가 완료 request를 `COMPLETE`로 표시하고 scheduler에서
  사라진 request metadata를 prune한다.
- launcher는 job ID 기반 port 탐색, server start retry, `model_info` 확인,
  Python/torch/sglang manifest 기록을 수행한다.
- `no_smi` 제출은 동일한 runner를 호출하되 `R1_USE_NVIDIA_SMI=0`을 설정한다.
- 기본 scheduler 경로와 per-layer switching 경로는 변경하지 않는다.

## 검증

```bash
python workspace/engine-port/tests/test_dual_worker.py
python -m compileall -q workspace/engine-port/src/multiplex/dual_worker.py \
  workspace/engine-port/tests/test_dual_worker.py
```

CPU-only unit test 6개가 통과한다. 이 환경에는 `pytest`가 설치되어 있지
않아 pytest 실행 결과는 없다. GPU serving 재검증은 새 canonical launcher로
별도 제출해야 한다.

## 다음 실행

```bash
sbatch workspace/engine-port/scripts/r1_dual_worker/r1_dual_worker.sbatch
sbatch workspace/engine-port/scripts/r1_dual_worker/r1_dual_worker_no_smi.sbatch
```

새 job의 보고서는 `reports/r1_dual_worker/`, 원자료는
`results/r1_dual_worker/job_<id>/`에 둔다.
