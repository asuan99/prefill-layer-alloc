# Observer agent deprecation

상태: **Removed (구현체 삭제 완료)**
적용일(deprecated): 2026-07-22
제거일(removed): 2026-07-24

## 2026-07-24 최종 제거

observer(=`workspace/slurm-agent/`, 코드명 "SLURM Experiment Observer") 구현체·서비스·
런타임 산출물을 **전부 삭제**했다. 사유: 상시 systemd 데몬이 `sacct`/`squeue`를 사용자
전체 대상으로 무필터 스캔해 **프로젝트와 무관한 job까지 보고서를 자동 생성**(불필요한 결과
제시)했고, deprecated 상태에서도 계속 실행 중이었다. 진행 중 job에 대한 자동 보고도 불요로 판단.

제거한 것:
- systemd user unit `slurm-experiment-agent.service` (stop + disable + unit 파일 삭제).
- `workspace/slurm-agent/` 전체(`slurm_agent.py`·래퍼·`.service`·`reports/job_*.md`).
- 런타임 상태 `.slurm-agent/`(lock/pid/log/`job_*.json`) 및 `.gitignore`의 `.slurm-agent/` 항목.
- `workspace/characterization/experiments/slurm/{run_pipeline,submit,submit_size_sweep}.sh`의
  observer 경유 제출(`$AGENT submit --`)을 **직접 `sbatch --parsable`** 로 치환.

대체: 완료 job 보고서 생성은 **on-demand로 `experiment-runner` 에이전트**가 담당(상시 데몬
아님, 프로젝트 실험 아티팩트만 대상). 상세는 `.claude/agents/experiment-runner.md`.

복구가 필요하면 deprecated commit `1eec8b8` 및 원격 `observer-agent` branch에서 출발한다.

---

## (이력) 2026-07-22 deprecation

`workspace/slurm-agent/`와 characterization 제출 경로에 추가되었던 observer agent는 현재 기본 실험 경로에서 제외한다. observer의 watcher/제출 래퍼만 제거하며, dual-worker 본체와 R1 재현 launcher·telemetry는 engine-port 연구 산출물로 유지한다.

## 사유

- 기존 SLURM 제출 경로에 별도 watcher/등록 상태를 추가해 운영 복잡도를 높였다.
- 기존 observer 경유 R1 preflight 실행은 `rg` 미설치로 중단되어 검증된 실험 결과를 만들지 못했다. launcher에는 `rg` fallback을 유지한다.
- observer agent가 제공하는 기능을 현재 연구 결론이나 재현 경로의 필수 요소로 채택하지 않는다.

## 현재 원칙

- 실험 제출은 observer agent 없이 기존 `sbatch` 경로를 사용한다.
- `.slurm-agent/`, observer 로그는 저장소 산출물로 보존하지 않는다. R1 launcher 자체와 그 실험 결과는 별도 관리한다.
- observer 구현을 다시 사용할 경우 deprecated commit `1eec8b8` 및 원격 `observer-agent` branch를 출발점으로 별도 검토한다.
