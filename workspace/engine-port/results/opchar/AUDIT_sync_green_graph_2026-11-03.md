# 감사: green stream graph replay 뒤 device sync 조기 반환 (2026-11-03)

claims-auditor 1회차. 결과 원본은 `workspace/engine-port/results/vessl_gpu_obs_probe_2026-11-02/`(opchar 브랜치, 1 MB 넘는 trace는 미커밋).

## 판정
- **관측 CONFIRMED(범위 한정)**: VESSL A100, 드라이버 580.178.04, torch 2.9.1+cu130, 단순 graph body, replay가 스트림의 마지막 작업. 이때 `torch.cuda.synchronize()`는 green stream replay 완료를 기다리지 않는다. 데이터를 읽어 확인한 것은 아니고 타이밍 추론이다.
- **정본 수치·claim 변경 없음**: CONSENSUS §1, CM-1…CM-4, opchar Safe wording, partition_power, A0 라벨.
- 580.105.08에서도 존재: PLAUSIBLE. replay가 마지막 작업일 때만 발생: PLAUSIBLE. 다음 replay가 이전 launch를 기다림: PLAUSIBLE. 기전: NOT-YET-SUPPORTED.

## 서빙 지표가 영향받지 않는 근거
- 토큰은 `multiplexing_mixin.py:1494`의 `decode_stream.synchronize()` 뒤에만 나간다.
- `results/engine_gate/pdmux-policy-gate-smoke-71f5006-20261028064810/on/tel.jsonl`(580.178.04, cudagraph ON): 레코드 17개에서 `total_cuda_synchronize` 120 고정, 같은 구간 `total_stream_synchronize` 6,927→49,016. 이 boot 1개의 실측이고, 다른 캠페인은 코드 경로 읽기에 의존한다.
- Event 측정은 `query()`가 거짓이면 `elapsed_time`이 예외를 낸다(`InlineEvent.h:125-128`). 확인한 헤더는 2.9.0+cu128·2.10.0+cu128이며 2.9.1+cu130 원본은 미확인.

## 영향받은 값 (둘 다 정본에 없음)
- `results/kernel_mech/stage0ppp/stage0ppp_a0_probe.py:205-209`의 `L4_wall_ns`: replay 1회분(약 5%) 짧음. 분석에서 사용하지 않음(grep 0건).
- `scripts/vessl/gpu_obs_green_load.py`의 `graph_replay_*` elapsed: 약 2%(1/50) 짧음.

## 인용 금지
- "이 관측은 서빙 TTFT·ITL·decode step 수치가 과소 측정됐다는 뜻이 아니다."
- `graph_replay_*` elapsed와 `L4_wall_ns`를 replay당 시간으로 나누지 않는다.
- 기전은 미확립이고 데이터 경합은 관측된 적이 없다.

## 규약 제안
green stream의 graph 시간은 `stream.synchronize()` 또는 query 게이트 Event로만 잰다.

## 미해결
- 호출자 미추적: `mem_cache/memory_pool.py:933-965,1657-1681`(형태상 경합), `managers/scheduler_update_weights_mixin.py:153`.
- 미검사: `copy_done`의 PD-mux 설정 경로, implfix handoff `_ho` 분기 내부 순서, multigpu·AMD 트리, KISTI 런 sync-audit, loader·hi*·disaggregation·lmcache·multimodal(grep만).
- GPU 프로브 후보(서버 불필요): (1) replay 뒤 event·eager·blocking `.item()` 후 device sync, eager Triton·flashinfer 대조 (2) device sync 직후 읽은 값과 stream sync 후 값 비교, 다노드 graph 포함 (3) replay 호출별 host 시간 (4) 580.105.08 재실행, 노드 선택이 가능할 때만.
