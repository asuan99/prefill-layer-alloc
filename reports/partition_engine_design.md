# partition/layer_aware 실증 엔진 — 스코핑 (착수 전 설계)

작성일: 2026-06-22 · 목적: [queue_simulator_design §15](queue_simulator_design.md)가 남긴 마지막 gap — *partition/layer_aware_protect의 end-to-end 실증* — 을 어떻게 닫을지 스코핑. **이 문서는 설계만, 코드 착수는 보류(사용자 지시).**

관련: [vllm_validation](vllm_validation.md)(fused baseline 실측 검증 완료) · [queue_simulator_design §14](queue_simulator_design.md)(layer_aware sim 결과)

---

## 1. 닫으려는 gap

sim(§11–14)·vLLM 실측(vllm_validation)이 닫은 것 = **fused(=vLLM 기본) baseline의 현실성**. 아직 *미실증*인 것 = **partition(`dynamic_protect`) / `layer_aware_protect`가 실제로 fused를 이기는가** — 이건 어떤 프레임워크에도 구현이 없어 sim 예측으로만 존재. 이 엔진이 그 예측(§14: temporal 하이브리드서 layer_aware 2× goodput)을 *실측*으로 확인/반증한다.

## 2. 왜 "vLLM에 직접"(경로 B)이 1순위가 아닌가 — vLLM 0.22.1 코드 근거

- **실행 단위가 단일 fused forward**: `gpu_model_runner.py:3692`가 prefill+decode를 이어붙인 mixed batch를 모델에 *한 번* 통과. partition은 이걸 두 동시 스트림으로 *쪼개야* 함 → 코어 설계 역행.
- **SM 공간분할 API 전무**: `cuGreenCtx`/`set_active_thread_percentage`/SM partition 코드 없음.
- **PD-disagg은 cross-instance**(KV connector): 별도 인스턴스 간 KV 전송이지 same-GPU SM multiplexing 아님.
- ∴ vLLM의 fusion이 곧 우리가 이기려는 `co_schedule` baseline. partition을 vLLM에 넣는 건 vLLM의 중심 최적화를 되돌리는 *논문급 수술*(경로 B). **검증이 목적이면 과한 길.**

## 3. 경로 비교

| | 방법 | 실현 | 노력 | vLLM 신뢰성 |
|--|--|--|--|--|
| A | MPS 2엔진 + same-GPU KV connector | prefill/decode 분할만, **layer_aware 불가** | 中 | 高(프로덕션 위) |
| B | vLLM model runner를 green-ctx 2-stream으로 수술 | 충실(layer_aware 포함) | **大** | 最高 |
| **C(추천)** | **레포 green-ctx 커널 재사용 최소 serving 하네스** | **충실(layer_aware 포함)** | **中** | 中(프로토타입) |

**추천 = C.** 이유: partition·layer_aware를 *충실히* 실증하면서, 분할 커널·동시실행·큐 루프가 **레포에 이미 다 있어** 노력이 中. vLLM과 안 싸움. (A는 layer_aware 불가 + same-GPU disagg 비표준; B는 과대.)

## 4. Path C 설계 — "성능-충실(performance-faithful)" 서빙 하네스

### 4.1 스코프 경계 (중요)
**정확한 출력 토큰을 만드는 correctness 엔진이 아니다.** 목적은 실측 TTFT/TBT/goodput이므로 — *실제 커널을 실제 크기의 KV/ssm-state 버퍼 위에서 continuous-batching 스케줄대로 실행해 latency를 측정*하면 충분. (프로젝트 전체 방법론이 성능 microbench라 일관. 출력 정확성은 불요 → 노력 大 절감.)

### 4.2 재사용 자산 (이미 존재)
- **green-ctx SM 분할**: `e4_concurrent/_green_ctx.py::create_two_partitions(n_first_sm)` → 공간 배타 2 파티션의 동시 스트림 (`cuGreenCtxCreate`). `measure_concurrent`/`time_solo`.
- **실커널**: `common/kernels.py::build_*` — qkv/in/out_proj, ssm_scan, attn_prefill, decode_ssm, decode_attn(+_opt flash). `load_layer_cfg`.
- **루프·스케줄러**: `e6_queue_sim/simulator.py`(continuous-batching + decoupled scheduler), `workload.py`(Poisson 도착), `run_queue_sim.py::metrics`(goodput@SLO). **여기서 LUT `step_ms`를 *실커널 실행*으로 교체**하는 게 핵심 작업.

### 4.3 새로 만들 것
1. **per-request 상태 캐시**: attn KV 버퍼(req별, ctx까지 증가) + ssm recurrent state(req별 고정 크기). step마다 decode가 갱신. (정확값 불요, 실크기·실접근만.)
2. **실행 백엔드** (`step_exec.py`): 한 step을 받아 (prefill 토큰들 ∥ decode batch) 를
   - `co_schedule`/`fused`: 한 스트림에 순차(또는 mixed) — baseline.
   - `dynamic_protect`(agnostic): decode를 *예약 파티션*, prefill을 나머지 파티션에 `measure_concurrent`식 동시 실행.
   - `layer_aware_protect`: 레이어 타입별로 — attn 레이어 decode만 예약 파티션, ssm 레이어는 full 공유. (temporal: 레이어 순회; spatial(falcon): 매 레이어 attn+ssm 동시라 적용 불가 — sim과 동일 결론 확인용.)
   실제 wall-clock으로 step_ms 반환 → simulator 루프에 주입.
3. **드라이버** `run_real_engine.py`: 모델·정책·λ·SLO 스윕, 실측 TTFT/TBT/goodput CSV. v2/§14와 동일 grid.

### 4.4 대상
- **zamba2_2.7b**(temporal, 9a+45s) — layer_aware 핵심 케이스.
- **falcon_h1_3b**(spatial) — layer_aware 미적용 대조.
- 정책: fused·co_schedule·agnostic_protect·layer_aware_protect (sim과 1:1).

## 5. 작업 분해 & 노력

| # | 작업 | 산출 | 추정 |
|--|--|--|--|
| 1 | per-request KV/ssm-state 캐시 + 갱신 | `state_cache.py` | 0.5d |
| 2 | step 실행 백엔드(4정책, green-ctx 동시) | `step_exec.py` | 1.5d |
| 3 | simulator step_ms→실커널 주입, 루프 연결 | simulator 패치 | 0.5d |
| 4 | 드라이버 + SLURM 잡 | `run_real_engine.py`+sbatch | 0.5d |
| 5 | 캘리브레이션(아래 §6-a) + 디버그 | — | 1d |
| 6 | 스윕 실행 + 분석 + 보고서 | results+report | 1d |

**합 ~5 GPU-일**(큐 대기 별도). 리스크 버퍼 포함 ~1주.

## 6. 검증 계획 (3단)
- **(a) 캘리브레이션**: 엔진의 `fused` 모드가 **vLLM 실측(job 783863) TPOT/throughput을 재현**해야(±~30%). 안 되면 커널/상태크기 보정. ← 이게 통과해야 partition 결과가 신뢰됨.
- **(b) 가설 검증**: `layer_aware_protect` vs `agnostic_protect` vs `co_schedule` 실측 goodput@SLO. **sim §14 예측(zamba2 SLO≥50ms서 layer_aware 2×)이 실측에서 재현되는가.**
- **(c) sim 대조**: 실측 ITL/goodput을 sim §14와 셀단위 비교 → sim의 정량 신뢰도 최종 판정.

## 7. 리스크 & 미지수
- **green-ctx 안정성**: `create_two_partitions`는 E4에서 동작 검증됨(microbench). full-model 루프·반복 파티션 전환에서의 안정성·오버헤드는 미지수 → §6-a에서 조기 노출.
- **layer_aware 파티션 전환 오버헤드**: per-layer green-ctx stream 전환 비용. sim은 ~0.4%로 추정(§14), 실측 필요 — *이 엔진이 그걸 실제로 재는 게 핵심 가치*.
- **상태 캐시 충실도**: KV/ssm-state 실크기·실접근 패턴이 vLLM paged와 다를 수 있음(연속 vs paged). 캘리브레이션(§6-a)으로 보정.
- **결과가 음성일 수 있음**: 실측서 layer_aware가 sim만큼 안 이길 수 있다(파티션 전환 오버헤드·실커널 경합이 sim 가정보다 클 때). 그것도 valid한 결론(sim 정량 과대 판정).

## 8. 스코프 경계 (정직)
- 이건 **연구 프로토타입(성능-충실)**이지 프로덕션/정확성 엔진이 아니다. partition/layer_aware *아이디어*를 실엔진 루프에서 실증/반증하는 게 목적.
- "real vLLM 통합"(경로 B)은 이게 양성이면 정당화되는 *별도 future work*. 이 엔진의 양성 결과 = B 투자에 대한 evidence; 음성 = B 불필요(아이디어 자체가 실엔진서 안 됨).
- SGLang 등 타 프레임워크 비교는 fused baseline 한정으로 이미 충분(둘 다 fused mixed-batch).
