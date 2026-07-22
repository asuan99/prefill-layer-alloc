# Observer agent deprecation

상태: **Deprecated**
적용일: 2026-07-22

`workspace/slurm-agent/`와 characterization 제출 경로에 추가되었던 observer agent는 현재 기본 실험 경로에서 제외한다. observer의 watcher/제출 래퍼만 제거하며, dual-worker 본체와 R1 재현 launcher·telemetry는 engine-port 연구 산출물로 유지한다.

## 사유

- 기존 SLURM 제출 경로에 별도 watcher/등록 상태를 추가해 운영 복잡도를 높였다.
- 기존 observer 경유 R1 preflight 실행은 `rg` 미설치로 중단되어 검증된 실험 결과를 만들지 못했다. launcher에는 `rg` fallback을 유지한다.
- observer agent가 제공하는 기능을 현재 연구 결론이나 재현 경로의 필수 요소로 채택하지 않는다.

## 현재 원칙

- 실험 제출은 observer agent 없이 기존 `sbatch` 경로를 사용한다.
- `.slurm-agent/`, observer 로그는 저장소 산출물로 보존하지 않는다. R1 launcher 자체와 그 실험 결과는 별도 관리한다.
- observer 구현을 다시 사용할 경우 deprecated commit `1eec8b8` 및 원격 `observer-agent` branch를 출발점으로 별도 검토한다.
