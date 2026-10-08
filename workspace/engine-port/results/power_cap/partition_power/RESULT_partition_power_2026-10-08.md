# 파티션별 전력 관측 프로브 — VESSL Job `job-78xs9733hcjy` (2026-10-08)

Job `pdmux-partition-power-20261008052457`, A100-SXM4-80GB ×1, 공개 이미지 digest 고정, object volume `/io`(원시 회수).
스크립트 `probe_partition_power.py`, 그림 `plot_partition_power.py`. 원시: `job-78xs9733hcjy/` + `SHA256SUMS`. **합성 커널 측정이며 서빙 엔진 측정이 아니다.**

## 질문과 설계

보드 전력계는 GPU 전체 하나다. 그래도 green-context 파티션(prefill 쪽 SM vs decode 쪽 SM)별 기여를 관측할 수 있는가?
방법은 **통제된 배제**다. 각 파티션의 작업을 혼자·함께 돌려 보드 증가분을 설계로 귀속한다(파티션 전력계는 없다).

- prefill-like = bf16 GEMM 4096³(compute-bound), decode-like = bf16 GEMV 16384²×16(memory-bound).
- 분할: `sgl_kernel` `create_greenctx_stream_by_value`로 (P,D) = (92,16), (64,44), (16,92). 드라이버 read-back smCount가 전부 목표와 일치.
- 각 분할마다 4 s씩 유휴 / P만 / D만 / P+D, 그리고 **D 연속 + P 200 ms ON/OFF 토글** 4 s. 전체-GPU(108 SM, 기본 스트림) P만·D만 기준을 둔다.
- 센서: NVML 버퍼 샘플(~20 ms, 6,153개)과 에너지 카운터(20 ms 폴링). 유휴 82 W.

## 결과

| 분할 | P만 ΔW | D만 ΔW | P+D ΔW | P+D / (P만+D만) | P 처리량(함께/혼자) | D 처리량(함께/혼자) | 토글: P ON / OFF |
|---|---|---|---|---|---|---|---|
| 전체 GPU | 318 | 275 | — | — | — | — | — |
| P92/D16 | 316 | 108 | 315 | **0.74** | 0.93 | 0.87 | 405 / 187 W |
| P64/D44 | 272 | 235 | 318 | **0.63** | 0.86 | 0.84 | 406 / 310 W |
| P16/D92 | 79 | 277 | 308 | **0.87** | 0.83 | 0.90 | 399 / 360 W |

(ΔW = 카운터 기준 보드 전력 − 유휴, 각 상태 4 s 중 가운데 3.3 s 평균)

## 판정 (기술 — 합성 부하 한정)

1. **혼자 돌 때 파티션별 전력은 분명히 구분된다.** 전력은 그 파티션의 SM 수를 따른다. P만: 92 SM 316 W → 64 SM 272 W → 16 SM 79 W. D만: 16 SM 108 W → 44 SM 235 W → 92 SM 277 W.
2. **20 ms 샘플로 prefill 버스트가 시간 영역에서 보인다.** decode가 계속 도는 위에 prefill을 200 ms씩 켜고 끄면 보드 전력이 계단으로 오르내린다(+218 / +96 / +40 W, 분할 의존). 한 샘플 안에 반응한다.
3. **함께 돌면 가산성이 깨진다(0.63–0.87).** 두 파티션이 동시에 바쁘면 보드 전력이 **400 W 한계에 붙는다**(P+D 상태 전부 397–400 W). 혼자 값의 합은 400 W를 넘으므로 chip-wide power cap이 잘라낸다.
   ⇒ 동시 실행 중에는 보드 전력을 파티션별로 나눌 수 없다(식별 불가). 파티션별 귀속은 배제 설계(혼자 돌린 상태)에서만 성립한다.
4. 동시 실행 시 처리량은 두 파티션 모두 0.83–0.93×로 떨어진다. 원인 후보는 메모리 대역폭 경합과 power cap 클럭 하강인데, **이번 Job은 클럭·throttle reason을 기록하지 않아 둘을 분리하지 못한다.**
5. 이전 power cap 평가와 연결된다. 기본 400 W 한계가 이미 두 파티션을 묶는 **공유 자원**으로 작동한다(합성 포화 부하 기준). 실제 서빙 Job에서 SW power cap 노출은 27–49 %였다(`RESULT_power_cap_passive_2026-10-07.md`).

## 한계 · 금지

- 합성 커널(포화 부하)이다. 서빙 엔진의 prefill/decode 커널·batch·동거 비율과 다르다. 서빙 결론으로 옮기지 않는다(게이트 1).
- n = 분할당 1회(Job 1개). 수치는 기술용이고 판정 근거가 아니다.
- "SM 분할 = 전력 분할" 금지 문안은 유지한다. 혼자 돌 때만 SM 수와 비례 관계가 보이고, 함께 돌면 cap이 결합한다.
- 다음에 이 질문을 닫으려면 같은 설계에 `clocks.sm`·throttle reason 20 ms 기록과 반복 n ≥ 4를 붙인다.

## 그림

- `figures/figA_power_increments.png`: 분할별 P만 / D만 / P+D 증가분과 "혼자 합" 점선(가산성)
- `figures/figB_toggle_timeseries.png`: decode 연속 + prefill 200 ms 토글의 20 ms 보드 전력
- `figures/figC_throughput.png`: 혼자 vs 함께 처리량(간섭)
