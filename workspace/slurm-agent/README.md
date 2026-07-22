# SLURM 실험 결과 보고 Agent

이 도구는 `sbatch` 제출 정보를 기록하고, 잡이 끝나면 `squeue`/`sacct`를
확인하여 실험 목적과 결과 파일을 정리한 Markdown 보고서를 생성한다. 기본
보고서 생성은 Python 표준 라이브러리만 사용하므로 별도 API key가 필요 없다.

## 사용법

```bash
cd /scratch/ehmoon/whlee/prefill-layer-alloc
workspace/slurm-agent/submit.sh workspace/engine-port/triage/r1_dual_worker.sbatch
workspace/slurm-agent/submit.sh slurm/measure_prefill_sm.sh
```

`submit.sh`는 `sbatch`와 같은 인자를 그대로 받는다. 제출 직후 다음 정보가
`.slurm-agent/job_<JOB_ID>.json`에 저장된다.

- job ID와 제출 시각
- 제출 디렉터리와 원래 `sbatch` 명령
- 스크립트의 `#SBATCH` 메타데이터
- 실험 경로, E0/E1 같은 라벨, Python 실행 명령으로부터 추론한 목적

제출 래퍼는 독립 observer를 분리해 실행하므로 터미널을 계속 점유하지 않는다.
observer는 현재 큐의 raw `sbatch` job도 주기적으로 발견하고, 잡의 종료를 감지하면
`workspace/slurm-agent/reports/job_<JOB_ID>.md`를 만든다.

```bash
workspace/slurm-agent/slurm_agent.py list
tail -f .slurm-agent/watcher.log
```

observer를 명시적으로 실행하려면 다음을 사용한다. 이 프로세스는 pipeline을
제출하거나 실행하지 않고, job 발견·종료 감시·결과 보고만 수행한다.

```bash
workspace/slurm-agent/observe.sh --poll 30
```

이미 `sbatch`로 제출한 잡도 등록할 수 있다.

```bash
# 특정 잡 하나
workspace/slurm-agent/slurm_agent.py adopt 862381

# 현재 사용자의 큐에 있는 모든 잡
workspace/slurm-agent/slurm_agent.py discover
```

`adopt`는 `scontrol`에서 실제 제출 스크립트, job name, 작업 디렉터리와 로그
경로를 읽어 기존 잡을 registry에 복원한다. 등록된 잡은 새로 제출한 잡과 동일하게
종료 후 보고서가 생성된다.

로그인 노드 정책이 백그라운드 프로세스를 정리하는 클러스터라면 watcher를
`tmux`/`screen` 안에서 직접 실행한다.

```bash
tmux new -s slurm-agent
workspace/slurm-agent/watch.sh --poll 30
```

## LLM Agent 검토를 추가하는 선택 설정

구조화 보고서 생성은 항상 수행되고, 아래 환경 변수를 지정했을 때만 LLM 검토가
추가된다. `{report}`, `{metadata}`, `{output}`은 도구가 실제 경로로 치환한다.
LLM은 결과를 읽고 보고서만 작성해야 하므로 read-only 모드 사용을 권장한다.

```bash
export SLURM_AGENT_CMD='codex exec --ephemeral --sandbox read-only --ask-for-approval never --cd /scratch/ehmoon/whlee/prefill-layer-alloc --output-last-message {output}'
workspace/slurm-agent/submit.sh slurm/measure_prefill_sm.sh
```

LLM 검토는 `job_<JOB_ID>.agent.md`로 별도 저장되고 원래 보고서의 `Agent 검토`
절에도 붙는다. Codex 인증이 없거나 실행에 실패해도 기본 보고서 생성에는 영향을
주지 않는다.

## 직접 보고서 재생성

```bash
workspace/slurm-agent/slurm_agent.py report 123456
workspace/slurm-agent/slurm_agent.py watch-once
```

자동 추론은 스크립트와 SLURM 메타데이터를 근거로 한 요약이다. 정상 종료는 실험
가설의 성공을 의미하지 않으며, 보고서의 결과 수치와 로그를 사람이 최종 검토해야
한다.

## 전체 v2 파이프라인

전체 파이프라인도 각 GPU 하위 잡을 agent에 등록하도록 연결되어 있다.

```bash
DETACH=1 MAXQ=2 POLL=30 env -u BASH_ENV \
  bash workspace/characterization/experiments/slurm/run_pipeline.sh
```

파이프라인 시작 시 기존 큐 잡을 먼저 adopt하고, 이후 제출되는 E1/E2/E3/E5/E4
잡은 자동 등록한다. 파이프라인 로그는 `logs/run_pipeline_*.log`, agent 로그는
`.slurm-agent/watcher.log`에서 확인한다. 동일 파이프라인이 이미 실행 중인지 먼저
확인해야 한다.
