# 고지 — `job_entry.sh` 전력·에너지 샘플러 추가 (N2 감사 대상, 2026-10-08)

**성격: 기판 변경 고지. N2 사전등록·판정 규칙·러너·축약기 무편집.** 다음 N2 감사 회차(rev11 델타)가 이 고지를 기판 caveat로 다루도록 남긴다.
사용자 결정(2026-10-08): power cap 트랙의 "A안" — N2 Job에 샘플러를 얹어 분할 크기별 decode-only 전력을 **부산물(기술)**로 얻는다.

## 무엇이 바뀌었나

`workspace/engine-port/scripts/vessl/job_entry.sh`가 타깃 스크립트 실행 전에 백그라운드 Python 프로세스 하나를 더 띄운다(기존 `nvidia-smi -lms 100` → `meta/clk.csv` 샘플러와 별개).
- `meta/energy.csv`: NVML 보드 누적 에너지 카운터(mJ), 100 ms 폴링.
- `meta/power_samples.csv`: NVML 버퍼 전력 샘플(`nvmlDeviceGetSamples` `NVML_TOTAL_POWER_SAMPLES`, 드라이버 타임스탬프, ~20 ms 간격), 200 ms마다 비움.
- `meta/sm_clk_samples.csv`: NVML 버퍼 SM 클럭 샘플(`NVML_PROCESSOR_CLK_SAMPLES`). 미지원이면 생략하고 `energy_sampler.log`에 사유만 남긴다(로컬 RTX 5060 Ti는 미지원, VESSL A100은 미확인).
- 전부 **읽기 전용** NVML 조회이며 GPU 설정을 바꾸지 않는다. 실패해도 Job을 실패시키지 않는다(best-effort, 오류 로그 20줄 상한).
- 끄기: `PDMUX_ENERGY_SAMPLER=0`(기본은 켜짐, 변수 미설정). ★N3식 러너 허용목록은 **설정된** 미등록 `PDMUX_*`에 abort하므로, 끄려면 N2 러너 허용목록에 이 변수를 등록해야 한다.

## 비용 (로컬 실측, 2026-10-08)

| 항목 | 값 |
|---|---|
| CPU | 20 s 동안 1.6 s(≈ 한 코어의 8 %; VESSL cgroup quota 11 CPU의 ≈ 0.7 %) |
| 메모리 | RSS ≈ 34 MB |
| 호출 비용 | 에너지 카운터 1회 ≈ 1.9 ms CPU(로컬) — 10 Hz |
| 산출 크기 | 시간당 power_samples ≈ 4–5 MB, energy ≈ 1 MB |

VESSL A100에서의 호출 비용은 미측정이다. GPU 커널·SM에는 관여하지 않는다(NVML은 드라이버 조회).

## N2 감사가 판단할 것 (처방 아님)

1. 이 프로세스가 N2의 측정량(엔진측 decode step 시간 S-per p50/p95/p99, 특히 b = 1 짧은 step)에 CPU 경합으로 영향을 줄 수 있는가 — 같은 Job 안 on/off 대조가 없으면 "영향 없음"은 주장할 수 없다.
2. 기판 기록(`substrate.json`·CPU stat)에 샘플러 on/off를 명시할지.
3. 켜 둘지(`A안`), 끌지(`PDMUX_ENERGY_SAMPLER=0` 등록), 또는 일부 유닛에서만 끄고 대조할지.

## N2 판정과의 관계

전력 산출물은 N2의 판정 입력이 **아니다**. power cap 트랙(`results/power_cap/`)이 N2 Job 회수 뒤 기술 분석만 한다(분할 크기별 decode-only 전력, batch·context별 J/step).
근거 문서: `results/power_cap/serving_power_preview/REVIEW_serving_engine_power_2026-10-08.md` §3 A, `results/power_cap/sensor_bw/RESULT_sensor_bandwidth_2026-10-08.md`.
