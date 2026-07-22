# Observer agent deprecation

상태: **Deprecated**
적용일: 2026-07-22

`workspace/slurm-agent/`와 characterization 제출 경로에 추가되었던 observer agent는 현재 기본 실험 경로에서 제외한다. main에는 observer 연동 코드와 dual-worker telemetry를 유지하지 않으며, 해당 구현은 Git history와 `observer-agent` 원격 branch에서 참고용으로만 보존한다.

## 사유

- 기존 SLURM 제출 경로에 별도 watcher/등록 상태를 추가해 운영 복잡도를 높였다.
- R1 preflight 실행은 `rg` 미설치로 중단되어 검증된 실험 결과를 만들지 못했다.
- observer agent가 제공하는 기능을 현재 연구 결론이나 재현 경로의 필수 요소로 채택하지 않는다.

## 현재 원칙

- 실험 제출은 observer agent 없이 기존 `sbatch` 경로를 사용한다.
- `.slurm-agent/`, observer 로그, R1 runtime manifest는 저장소 산출물로 보존하지 않는다.
- observer 구현을 다시 사용할 경우 deprecated commit `1eec8b8` 및 `observer-agent` branch를 출발점으로 별도 검토한다.
