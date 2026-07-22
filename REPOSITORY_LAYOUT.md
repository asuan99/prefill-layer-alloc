# Repository layout

이 저장소는 실험 트랙을 유지하면서 산출물의 목적을 분리한다. 새 파일은
가능하면 해당 트랙 아래의 canonical 경로에 둔다.

| 목적 | canonical 위치 | 포함하는 것 |
|---|---|---|
| 진행·설계·판정 보고서 | `workspace/<track>/reports/` | 연구 진행, 설계, 해석, 실패 원인, 인수인계 |
| 실험 원자료·실행 결과 | `workspace/<track>/results/<campaign>/` | CSV/JSON, SLURM 로그, manifest, job별 산출물 |
| 재적용 가능한 소스 | `workspace/<track>/src/` | 엔진 패치 미러, 모델·연산·공통 라이브러리 |
| 단위 테스트 | `workspace/<track>/tests/` | CPU-only 및 회귀 테스트 |
| 실행·제출 스크립트 | `workspace/<track>/scripts/` | 환경 bootstrap, `sbatch`, 분석 실행 드라이버 |

현재 트랙은 다음과 같다.

- `workspace/characterization/`: 커널·SM 특성화와 시뮬레이터
- `workspace/engine-port/`: SGLang 포팅과 실제 서빙 검증
- `workspace/serving-eval/`: serving 평가 하네스
- `workspace/shared/`: 트랙 공통 설정

## engine-port R1 예시

R1 dual-worker 산출물은 목적별로 다음에 모여 있다.

```text
workspace/engine-port/
├── src/multiplex/                 # dual-worker runtime state
├── tests/test_dual_worker.py     # CPU-only regression tests
├── scripts/
│   ├── bootstrap/install_engine.sh
│   └── r1_dual_worker/            # direct/no-smi SLURM launchers
├── reports/r1_dual_worker/        # job별 자동 보고서와 진행 보고서
└── results/r1_dual_worker/        # job별 manifest, stdout/stderr, server trace
```

`triage/`와 기존 `results/<campaign>/*.sbatch`는 과거 P-series/R-series
재현 경로이므로 이번 정리에서 일괄 이동하지 않았다. 새 실험은 `scripts/`에
제출 스크립트를 두고, 결과는 반드시 해당 campaign의 `results/` 아래에
기록한다. 루트의 SLURM `.out`/`.err` 파일은 커밋하지 않는다.
