# SLURM Experiment Observer

이 agent는 pipeline을 실행하지 않는다. 현재 큐와 최근 `sacct` accounting을
주기적으로 확인하고, 종료된 job의 스크립트·로그·결과 파일을 모아 보고서를 만든다.

```bash
cd /scratch/ehmoon/whlee/prefill-layer-alloc
workspace/slurm-agent/agentctl.sh start
workspace/slurm-agent/agentctl.sh status
workspace/slurm-agent/agentctl.sh poll
```

observer는 다음을 수행한다.

- `squeue`에서 기존/신규 raw `sbatch` job 발견
- `scontrol` 및 job-name으로 실험 스크립트 추적
- `sacct`에서 observer가 내려간 동안 종료된 job backfill
- 종료 상태와 원자료를 `reports/job_<JOB_ID>.md`로 정리
- `summary.csv`가 있으면 정책별 종합 판정, 시나리오별 `legacy → candidate`
  비교표, 변화율, 시사점, 다음 단계를 자동 생성
- 종료 시 기존 보고서가 있어도 최신 결과를 다시 읽어 보고서를 재생성

성능 보고서의 수치와 판정은 먼저 구조화된 결과 분석기가 생성한다. LLM을
서술 단계에 연결할 경우에도 이 정책 비교표와 원자료 경로를 입력으로 사용해,
표에 없는 수치나 근거가 보고서에 추가되지 않도록 한다.

`agentctl.sh start`는 detached observer를 시작한다. 프로세스가 세션 종료 시
사라지는 환경에서는 user systemd 또는 tmux에서 `observe.sh --poll 30`을 실행해야
한다. observer 로그는 `.slurm-agent/observer.log`다.

이 클러스터에서는 user systemd를 사용할 수 있으므로 영구 실행은 다음처럼
설정한다.

```bash
systemctl --user link /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/slurm-agent/slurm-experiment-agent.service
systemctl --user daemon-reload
systemctl --user enable --now slurm-experiment-agent.service
systemctl --user status slurm-experiment-agent.service
```

서비스가 시작되면 pipeline을 실행하지 않고 observer만 실행한다.

특정 job 보고서를 다시 만들 수 있다.

```bash
workspace/slurm-agent/agentctl.sh report 862381
```
