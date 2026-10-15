# NOTICE — 로컬 엔진 트리 `sglang_engine_dev` 동기화 (2026-10-15, main 세션)

대상: amd·opchar 트랙(및 로컬에서 엔진 미러 시험을 돌리는 모든 세션)

- 2026-10-15부터 `~/Experiments/KISTI/sglang_engine_dev`는 main(merge `dc06c870`) 트리로 동기화됐다. ENGINE_DIGEST `82a5fc5b…`(= `sglang_engine_dev_multigpu`, 32개 파일 바이트 동일). 사용자 승인 결정.
- 바뀐 파일: `sglang/srt/multiplex/{multiplexing_mixin,telemetry,holb_probe,chunk_probe}.py`(merge src), `pdmux_context.py`(device assert 패치), `multigpu_guard.py`(신규).
- **영향**: amd·opchar worktree의 `scripts/bootstrap/sync_engine_tree.sh`(구버전), mirror·sync 계열 시험 11–12개 모듈, 로컬 `r2_eval.sbatch`는 `SGLANG_ENGINE_DEV`가 비어 있으면 이 경로를 기본값으로 읽거나 **덮어쓴다**.
  - 읽기: mirror 시험이 실패한다(amd `test_probe_flush_durability.TestMirrorSync` 실측 FAIL — "telemetry.py and the installed runtime diverged").
  - 쓰기: 구버전 sync가 main 트리의 4파일을 옛 버전으로 덮고 `pdmux_context.py`·`multigpu_guard.py`는 남겨 **어느 manifest와도 안 맞는 섞인 트리**가 된다.
- **할 일**: 로컬 실행 전에 자기 사본 트리를 만들고 `SGLANG_ENGINE_DEV=<자기 사본>/python`을 반드시 지정한다. VESSL 경로(`substrate_probe.sh`의 `/opt/pdmux`)는 영향 없다.
- 되돌리기(필요 시, main 세션 몫): scratchpad 백업 `devtree_pre_sync_backup/targets.tar` — 세션 전용 경로라 영구 보존은 아니다.
