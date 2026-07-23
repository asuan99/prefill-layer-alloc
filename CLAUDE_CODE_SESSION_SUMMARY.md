# Claude Code 세션 산출물 통합 정리

> **HISTORICAL HANDOFF (2026-07-22):** R1/R2 이전 상태다. 현재 연구 방향,
> dual-worker 판정과 실험 gate는 [`PROJECT_STATUS.md`](PROJECT_STATUS.md)를
> 따른다.

작성 기준: 2026-07-22
범위: `prefill-layer-alloc` 저장소 내부의 연구·실험·엔진 포팅 산출물

> 이 문서는 `prefill-layer-alloc`에서 여러 Claude Code 세션을 통해 누적된 코드, 실험, 로그, 보고서를 하나의 인수인계 문서로 정리한 것이다. 최신 실엔진 측정 결과를 현재 결론으로 삼고, 과거의 긍정 결과는 역사적 기록으로 분리한다.

## 1. 한눈에 보는 현재 결론

| 질문 | 현재 판정 |
|---|---|
| Layer-type(attention/SSM) 기반 정적 SM 공간 분할이 유효한가? | **아니다.** 실제 서빙에서 동시실행 또는 agnostic 정책을 안정적으로 넘지 못했다. |
| Layer-aware decode SM 예약이 agnostic 예약보다 우수한가? | **아니다.** 4개 hybrid 모델의 실제 측정에서 저부하는 동률, 부하에서는 agnostic이 우세했다. |
| SLO-aware 동적 SM 제어가 최적 정적 split을 넘는가? | **아니다.** 관대한 SLO와 tight chat SLO 모두에서 decode-heavy static이 우세했다. |
| 실패 원인은 단순한 switch 오버헤드인가? | **아니다.** 잘못된 split 위치, decode starvation, prefill/decode scheduler state 결합이 핵심이다. |
| 단일 GPU에서 남은 headroom을 활용할 수 있는가? | 제한적이다. TTFT와 ITL을 독립적으로 최적화한 oracle headroom은 coupling tax 때문에 단일 108-SM GPU에서 사용할 수 없다. |
| 실무 기본 정책은? | **Peak decode 부하 기준 decode-heavy static split을 고정**한다. 검증된 static profile이 없으면 agnostic PD-mux를 기본값으로 둔다. |

중요한 정정: [`reports/PAPER_RESULTS.md`](reports/PAPER_RESULTS.md)와 [`reports/FINAL_SUMMARY.md`](reports/FINAL_SUMMARY.md)에는 시뮬레이터 기반으로 layer-aware 예약이 Zamba2에서 **1.37–2.02×** 우세하다는 결과가 있다. 이후 SGLang v0.5.10 실서빙 검증에서 이 예측은 뒤집혔다. 시뮬레이션의 fine-grained per-layer 창과 실제 prefill 실행 granularity의 차이를 놓친 것이 원인이다.

## 2. 저장소 산출물 지도

### 2.1 초기 연구·프로파일링·시뮬레이션

- `src/`: 메모리 footprint, 시각화, 분석용 기반 코드
- `stage1_sm_scaling/`: SM scaling sweep, kernel/layer 민감도, SSM 측정
- `stage2_overhead/`: Green Context와 SM 재구성 오버헤드 측정
- `stage3_hm_eval/`: 정책 A/B/C 및 hybrid 모델 평가
- `configs/`, `slurm/`: 모델 구성과 실험 제출 스크립트
- `reports/`: 실험 설계, 시뮬레이터, 모델별 분석, 논문용 결과
- `profile_data_*`, `results/`: 수집된 CSV, 측정 결과, 분석 산출물

초기 가설은 attention과 SSM의 자원 비대칭을 Green Context 기반 SM 분할로 활용하는 것이었다. 마이크로벤치와 시뮬레이션에서는 layer-aware 예약의 가능성이 보였지만, 실제 엔진 측정 전의 결과였다.

### 2.2 `workspace/engine-port` — 실제 SGLang 포팅·서빙 검증

- `workspace/engine-port/scripts/`: 환경 bootstrap 및 현재 실험 제출 스크립트
- `workspace/engine-port/triage/`: 과거 pdmux 부팅·모델 포팅·기초 serving harness 재현 경로
- `workspace/engine-port/env/`: 외부 개발 트리와 venv에 적용한 패치 기록
- `workspace/engine-port/results/`: SLURM 로그, 정책별 결과, 재분석 스크립트
- `workspace/engine-port/reports/`: 최신 정본, 연구 아크, postmortem, SLO 분석
- `workspace/engine-port/RESUME.md`: 환경·패치·부팅·재현 상태
- [`REPOSITORY_LAYOUT.md`](REPOSITORY_LAYOUT.md): 진행 보고서·결과·소스·실행 스크립트 배치 규칙
- [`workspace/engine-port/reports/r1_dual_worker_progress.md`](workspace/engine-port/reports/r1_dual_worker_progress.md): R1 현재 실행 상태와 실패 job 기록

검증 환경은 A100-SXM4-80GB(108 SM), SGLang v0.5.10, CUDA 13, torch 2.9.1, `sglang_kernel 0.4.1+cu130`이다. venv와 editable SGLang 개발 트리는 저장소 밖에 있으므로 재현 시 `RESUME.md`를 먼저 확인해야 한다.

기본 부팅 재현:

```bash
cd prefill-layer-alloc
sbatch workspace/engine-port/triage/p1_0_pdmux_boot.sbatch
```

## 3. 연구 흐름과 결론 변화

| 시기 | 작업 | 결과와 의미 |
|---|---|---|
| 2026-05–06 | 초기 SM 분할 가설 | SSM/attention의 SM 민감도 차이를 Green Context 공간 분할로 활용하려 했다. |
| 2026-06 | 시뮬레이션·마이크로벤치 | temporal hybrid에서 layer-aware 예약 1.37–2.02× goodput이라는 중간 결론을 얻었다. 정적 공간 분할은 이미 약세였다. |
| 2026-07-02–05 | SGLang/A100 엔진 포팅 | pdmux, Zamba2/NemotronH, Green Context, layer-aware 정책을 실제 엔진에 연결했다. 초기 단일·불완전 측정은 가능성을 남겼다. |
| 2026-07-05–13 | 4개 hybrid 및 coordinated per-layer 검증 | NemotronH, Zamba2, Granite-4, Falcon-H1에서 simulated advantage가 실제 goodput advantage로 전이되지 않음을 확인했다. |
| 2026-07-14–17 | SLO-aware 동적 controller | feedforward, binding-first, saturation-hold, feasibility gate를 시험했다. 변화 trace n≥4에서 best-static을 넘지 못했고 noise와 metric cliff를 분리했다. |
| 2026-07-18–19 | tight SLO 직접 재측정 | chat SLO(300/50ms)로 controller를 실제 재튜닝해 동적 우위를 반증했다. rate 8에서 d44 73.2% 대 bind+GATE 44.3% attainment였다. |
| 2026-07-19–22 | 극단 workload·oracle·coupling 분석 | disjoint-feasibility region은 존재하지만 동적 정책은 여전히 패배했다. 독립 TTFT/ITL oracle의 +16% headroom은 116 SM을 요구해 단일 108-SM GPU에서 불가능했다. |

현재 최신 변경은 temporal intervention과 TTFT/ITL coupling을 시각화하는 단계까지 반영되어 있다.

## 4. 결과 상태 분류

### 현재 정본

- [`workspace/engine-port/reports/CONSENSUS.md`](workspace/engine-port/reports/CONSENSUS.md): 확정·철회·미완료 상태의 단일 기준
- [`workspace/engine-port/reports/research_arc.md`](workspace/engine-port/reports/research_arc.md): 가설 변화의 연구 서사
- [`workspace/engine-port/reports/per_layer_type_postmortem.md`](workspace/engine-port/reports/per_layer_type_postmortem.md): per-layer-type 구제 시도의 실패 chain
- [`workspace/engine-port/reports/interactive_slo_retune_plan.md`](workspace/engine-port/reports/interactive_slo_retune_plan.md): tight-SLO 직접 재측정
- [`workspace/engine-port/RESUME.md`](workspace/engine-port/RESUME.md): 엔진 포팅 환경·패치·부팅·재현 상태

### 역사적 참고자료

- [`reports/PAPER_RESULTS.md`](reports/PAPER_RESULTS.md): 실엔진 반증 전의 논문용 결과
- [`reports/FINAL_SUMMARY.md`](reports/FINAL_SUMMARY.md): 시뮬레이션 결론과 이후 정정이 함께 기록된 capstone
- `reports/layer_aware_benefit_report.md`: Zamba2 시뮬레이션 goodput 이득
- `reports/vllm_validation.md`, `reports/framework_comparison.md`: vLLM baseline과 simulator calibration
- [`workspace/engine-port/reports/policy_comparison.md`](workspace/engine-port/reports/policy_comparison.md): 정책 taxonomy와 단계별 비교
- [`workspace/engine-port/reports/system_vs_engine_vs_sim.md`](workspace/engine-port/reports/system_vs_engine_vs_sim.md): simulator·engine·full-system fidelity 차이

이 문서들은 당시의 가설과 수정 경위를 설명하는 데 유용하지만, 현재 정책 결론을 인용할 때는 `CONSENSUS.md`를 우선한다.

### 폐기·재사용 금지

- `deprecated_reports/` 전체
- `workspace/engine-port/triage/DEPRECATED_gil_client.md`
- stationary ShareGPT r8 단독 결과와 n이 작은 정책 비교 결과
- `PAPER_RESULTS.md`의 “layer-aware 1.37–2.02× 실효 이득” 문장

폐기 문서는 삭제하지 않고 이력 보존용으로 남겨져 있다. 최신 결과와 충돌할 수 있으므로 새 분석의 근거로 직접 재사용하지 않는다.

## 5. 현재 남은 작업

### 우선순위 1 — long-context 실 trace 검증

현재 실서빙 반증의 대부분은 short-context ShareGPT trace에 기반한다. [`longcontext_trace_plan.md`](workspace/engine-port/reports/longcontext_trace_plan.md)는 L≈3k 이상에서 prefill-side lever가 다시 열릴 가능성과, 반대로 granularity/coupling이 여전히 이를 삼킬 가능성을 함께 기록한다.

실행 전 블로커:

- Zamba2-2.7B context limit 4096
- long prefill에서 goodput SLO가 모든 정책에서 0이 될 가능성
- 모델·baseline·SLO를 다시 측정해야 하므로 기존 수치와 직접 연결할 수 없음

### 우선순위 2 — 일반화 범위 확장

- A100 외 GPU에서 decode-heavy static 권고 재검증
- 7B 초과 모델과 다른 temporal hybrid 측정
- 실제 사용자 trace의 context-length/mix 분포 반영

### 우선순위 3 — disaggregation 후속 연구

Oracle 분석에서 관찰된 TTFT⊗ITL headroom은 단일 GPU split이 아니라 별도 prefill/decode device pool에서만 실현될 가능성이 높다. 이 방향은 layer-aware 부활을 전제하지 않고, coupling이 제거된 PD-disaggregation 정책으로 별도 정의해야 한다.

## 6. 다음 작업 시 지켜야 할 방법론

- 정책 비교는 stationary benchmark가 아니라 변화 trace를 사용한다.
- 결론 전 baseline 분산을 먼저 측정하고 n≥4를 확보한다.
- dynamic 결과에는 `switch_count`, split 체류 분포, TTFT/ITL p50/p95/p99를 함께 기록한다.
- goodput처럼 SLO 임계 지시함수에 의존하는 지표는 용량과 TTFT 분포를 함께 보고한다.
- 여러 라운드 결과를 합칠 때 duration은 `max()`가 아니라 합산한다.
- SLO를 바꿔 평가할 때는 기존 controller를 재스코어하지 말고 해당 SLO로 controller를 재튜닝해 직접 측정한다.
- 시뮬레이터의 per-layer 이득을 실엔진의 step/layer granularity 이득으로 해석하지 않는다.

## 결론

Claude Code 세션들의 `prefill-layer-alloc` 산출물은 단순한 layer-aware 구현 결과가 아니다. SM 분할 가설을 세우고, 시뮬레이션으로 가능성을 확인하고, 실제 SGLang 엔진으로 포팅한 뒤, 여러 hybrid 모델과 SLO에서 적용 범위를 좁혀 **단일 GPU에서는 decode-heavy static/agnostic 정책이 더 신뢰할 수 있다**는 결론까지 도달한 연구 기록이다. 다음 핵심 질문은 layer-aware 자체의 재시도가 아니라 long-context와 disaggregation에서 이 결론의 유효 경계를 확인하는 것이다.
