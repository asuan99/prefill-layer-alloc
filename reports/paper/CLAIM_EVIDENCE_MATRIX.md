# Claim–evidence matrix

최종 갱신: 2026-07-23

| Claim | 현재 판정 | Existing evidence | Missing evidence | Required experiment |
|---|---|---|---|---|
| A. Hybrid composition, context, active load에 따라 decode demand가 변한다 | 부분 지지 | Zamba2 context knee; synthetic/ShareGPT의 best split 이동; 다중 모델 batch/context characterization | 동일 CUDA Graph 운영점의 joint surface, GQA/composition 통제, held-out accuracy | P3 full-model profile, feature ladder, leave-one-workload/model-family-out |
| B. layer-level reconfiguration은 ITL critical path와 CUDA Graph를 훼손한다 | 강한 지지, 현 구현 범위 한정 | coordinated TPOT 약 42→124 ms, 최적화 후 약 85 ms; sub-step drain; graph incompatibility | 다중 모델 반복과 timeline attribution | B7 반복, CUDA Graph on/off, Nsight synchronization timeline |
| C. decode starvation은 TTFT도 악화시킨다 | running-batch 경로 강함; KV 경로 부분 | D16 TTFT 7.24 s/ITL 61.9 ms 대 D24 1.21 s/39.9 ms; admission capacity 관측 | time-aligned KV occupancy와 admission reason | D16/D24 paired replay, structured KV/full/mamba occupancy, mediation timeline |
| D. execution-state separation은 single-worker coupling을 줄인다 | 미검증 | R1은 observer라 해당 증거가 아님 | 실제 두 host loop에서의 fixed-split 비교 | legacy fixed 대 true dual fixed, 동일 telemetry/seed/graph |
| E. Hybrid-informed decode floor가 generic/global static보다 높은 SLO goodput을 낸다 | 미검증 핵심 가설 | 없음 | architecture control, generic policy, static/oracle, profile generalization | B0–B8, P4 profile ablation, W1–W9 및 real trace |

## 주장 제한

- Claim B는 A100/SGLang green-context implementation에 한정한다.
- Claim C는 KV telemetry 전까지 “shared running-batch/capacity congestion”으로
  표현하고 KV causal chain을 확정하지 않는다.
- Claim D는 true dual fixed가 architecture gate를 통과한 뒤에만 사용한다.
- Claim E는 B6가 B1과 B5를 모두 유의하게 이긴 경우에만 사용한다.
- D/E가 실패하면 A–C의 characterization 및 negative result를 논문의 중심으로
  유지한다.
