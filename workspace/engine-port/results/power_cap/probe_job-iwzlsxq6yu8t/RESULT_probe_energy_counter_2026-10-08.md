# 에너지 카운터 프로브 — VESSL Job `job-iwzlsxq6yu8t` (2026-10-08)

Job `pdmux-energy-probe-20261008014846`, A100-SXM4-80GB ×1(드라이버 580.105.08), 공개 이미지 digest 고정, 볼륨 없음.
스크립트 `../probe_energy_counter.sh`. 상태 `succeeded`. 회수: `job_logs.txt`, `job_show.json`, `SHA256SUMS`.

## 판정: VESSL 컨테이너에서 GPU 보드 에너지 측정 **가능**

| 항목 | 관측 |
|---|---|
| NVML `nvmlDeviceGetTotalEnergyConsumption` | 읽힘(권한 불필요), 단위 mJ |
| 카운터 갱신 주기(유휴) | 2 s 동안 19회 변화, 중앙 간격 103 ms |
| 유휴 5 s | 카운터 449.2 J(89.8 W) vs power.draw 적분 440.0 J → 적분/카운터 98.0 % |
| bf16 matmul 부하 20 s | 카운터 7,900.2 J(394.6 W) vs 적분 7,852.5 J → 99.4 % |
| 유휴 5 s | 카운터 458.7 J(91.7 W) vs 적분 466.6 J → 101.7 % |
| `power.draw` vs `power.draw.instant` | 이 A100·드라이버에서 100 ms 샘플 적분이 두 필드 모두 같은 값 |
| 호스트 RAPL(`/sys/class/powercap`) | 컨테이너에 없음 — CPU·DRAM 에너지는 측정 불가(공유 노드라 귀속도 불가) |

결론: 카운터 차분이 window 에너지의 1차 측정이 되고, 기존 Job의 `power.draw` 적분은 수십 초 이상 window에서 ±2 % 안의 대리값이다
(20 s 부하에서 −0.6 %; 짧은 유휴 window의 ±2 %는 1 s 평균·경계 효과). 범위: GPU 보드 전체만 — prefill/decode 파티션 분리 불가.

## 반영
- `scripts/vessl/job_entry.sh`: 모든 이후 Job이 `meta/energy.csv`(epoch_s, energy_mJ, 100 ms)를 기록(best-effort, pynvml 없으면 `energy_sampler.log`에 사유만 남김). 기존 Job·봉인된 사전등록 스크립트는 불변.
- `results/power_cap/analysis_2026-10-07/pc_common.py`·`pc_energy.py`: `energy.csv`가 있으면 카운터 차분(선형 보간, 범위 밖 외삽 금지)을 우선 사용하고 적분 대비 %를 함께 기록, 없으면 기존 적분.

## 정정 (2026-10-08, 센서 대역폭 프로브 `job-absefkt9e88x`)

위 "적분은 ±2 % 안의 대리값"은 **정상 부하에서만** 확인된 것이다. 사각파 부하에서 100 ms 스냅샷 적분은 50 ms 주기에 −27 %(aliasing)까지 벗어났고,
카운터는 모든 주기에서 일관됐다. 카운터를 1차 측정으로 쓰고, 적분 기반 기존 수치에는 이 caveat를 붙인다. 상세 `../sensor_bw/RESULT_sensor_bandwidth_2026-10-08.md`.
