# 세션 핸드오프 — 2026-10-13 (power cap / 에너지 관측 트랙, 작업 기간 2026-10-07 ~ 2026-10-12)

## 이번 세션 요약

사용자가 `prefill-layer-alloc`와 병행해 GPU power cap의 개입여지를 물었다. 결론은 두 갈래다.
**power cap을 처치로 쓰는 것은 현재 기판(VESSL Job/Workspace 컨테이너)에서 불가**하다(실측). **에너지·전력의 관측은 권한 없이 가능**하다.
이 판단을 프로브 Job 5개로 확정했다. Job 공통 스크립트에 전력·에너지 샘플러를 넣었다(이후 모든 Job에 적용). N2 등록 창을 그대로 쓰는 전력 분석기를 준비했다.
교수님 보고 문서에는 감사를 거친 최소안만 반영했다. 사용자 결정으로 KISTI는 앞으로 고려 대상이 아니며, 관련 메모리를 정리했다.
**새 성능 판정 0건, 정본(PROJECT_STATUS·CONSENSUS) 무편집, Claim 등급·게이트 불변.**

## 결정·측정

| 날짜 | 무엇 | 결론(기술 수준) | 근거·원자료 |
|---|---|---|---|
| 2026-10-07 | 수동 재분석(0 GPU-h) | 기존 VESSL 데이터의 cap 중 클럭 하락은 −3…−4 %뿐이라 cap 처치 정보가 없다. J/good은 SLO cliff에서 발산한다. 전력 ≠ 에너지(E2 A 쌍 8/8). claims-auditor 감사 `MIXED`, 치명 0, 문구 전부 반영 | `results/power_cap/RESULT_power_cap_passive_2026-10-07.md`, `AUDIT_RESULT_passive_2026-10-07.md` |
| 2026-10-07 | 선행연구 | PEACE(SoCC'26)가 일반 기제를 선점했다. 남는 공백은 단일 GPU PD-mux × chip-wide cap뿐이고 워크숍급 | `reports/power_cap_viability_2026-10-07.md` §4 |
| 2026-10-07 | 컨테이너 권한 프로브 | uid 0이지만 CAP_SYS_ADMIN이 없다. no-op `-pl 400`·NVML set 모두 `Insufficient Permissions`. 설정 가능 범위는 100–400 W | `results/power_cap/probe_job-x3lzellrkm8z/` |
| 2026-10-07 | VESSL 경로 조사 | root 경로는 VM Cluster뿐이다(passthrough VM, H100 이상, 8 GPU/노드, 선불). 그곳의 `-pl` 동작은 미확인 | viability §2-1 |
| 2026-10-08 | 에너지 카운터 프로브 | NVML 누적 카운터를 권한 없이 읽는다. 정상 부하에서 적분과 −2.0…+1.7 % 일치. RAPL 없음 | `results/power_cap/probe_job-iwzlsxq6yu8t/` |
| 2026-10-08 | 센서 대역폭 프로브 2회(사각파) | 폴링 `power.draw`와 카운터는 ~100 ms 갱신. **NVML 버퍼 샘플은 ~20 ms**(계단응답 12–34 ms, 평균 편향 없음, 권한 불필요). 스냅샷 적분은 50 ms 주기에서 −27…−33 % aliasing. 카운터는 연속 적분이라 구간 총량이 정확 | `results/power_cap/sensor_bw/RESULT_sensor_bandwidth_2026-10-08.md`, 그림 `sensor_bw/figures/` |
| 2026-10-08 | 파티션 전력 프로브(합성 GEMM/GEMV, n=1) | 혼자 돌면 SM 수에 따라 구분된다. 함께 돌면 보드가 400 W cap에 붙어 가산성이 0.63–0.87로 깨지므로, 동시 실행 중 파티션 귀속은 식별 불가 | `results/power_cap/partition_power/RESULT_partition_power_2026-10-08.md`, 그림 `partition_power/figures/` |
| 2026-10-08 | 서빙 미리보기(기존 N3 B r3, 0 GPU-h) | 유휴 89 / decode만 265–296 / 동거 365–369 W(cap 92–97 %). 상태와 실현 분할이 겹쳐(sticky OFF) 파티션 효과는 분리 불가 | `results/power_cap/serving_power_preview/` |
| 2026-10-08 | 사용자 "A 진행" | N2에 샘플러를 얹는 경로를 택했다. N2 고지를 남겼고, rev12부터 §3-3·N2C-61로 흡수됐다 | `results/n2_static_grid/NOTICE_substrate_power_sampler_2026-10-08.md` |
| 2026-10-12 | 사용자 결정 | **KISTI는 앞으로 고려하지 않음.** 재활용 전제 메모리를 정리했다 | 메모리 `kisti-closed` |
| 2026-10-12 | 교수님 보고 문서 | 최소안 4곳(현황 1문장·트랙 표 1행·"두→세 트랙"·결정 요청 1항목)을 반영했다. 반영 후 claims-auditor 감사의 NEEDS-FIX 3건을 전부 고쳤다(rev 14) | Claude Docs "Hybrid PD-mux 진행 보고 (교수님)" |

## 코드·문서 변경 (전부 커밋됨)

| 커밋 | 내용 |
|---|---|
| `2196642` | `scripts/vessl/job_entry.sh`: NVML 샘플러를 추가했다. 산출은 `meta/energy.csv`·`power_samples.csv`·`sm_clk_samples.csv`이고, opt-out은 `PDMUX_ENERGY_SAMPLER=0`이다. 로컬 비용은 한 코어 ~8 %, RSS ~34 MB. N2 고지도 같은 커밋 |
| `7a7fa10` | power_cap 트랙 전체: 분석 스크립트·프로브 스크립트·RESULT/AUDIT/PREREG 초안/REVIEW·그림·SHA256SUMS. 원시 프로브 CSV는 로컬에 보관하고 해시는 커밋했다 |
| `677111a` | REVIEW 문서의 샘플러 상태 문구 갱신 |
| `cea838d` | `results/power_cap/n2_decode_power/pc_n2_decode_power.py`: N2 등록 창(`n2_probe_reduce.cell_windows`)으로 W/C/A2별 20 ms 전력·J/step을 계산하고, 단독 vs 동거 대비를 낸다. selftest는 합성 데이터 + N2 실제 harness·fake engine 종단, 무전력 변이 사살 |
| (다른 세션) `dc59129` | `energy.csv`가 3열(epoch, mono, mJ)로 바뀌어 분석기 reader를 고쳤다. `tests/test_pc_energy_layouts.py` 추가 |

사전등록 초안 `results/power_cap/PREREG_PC_CAP_REV0_DRAFT_2026-10-07.md`(rung C)는 실행 불가 상태 그대로다.

## 열린 항목 / 다음 세션 시작점

1. **N2 전력 데이터가 이미 있다**(이번 세션 이후 다른 세션이 N2 Job을 실행). 아래 명령은 아직 돌리지 않았다.
   - Job-PROFILE `pdmux-n2-static-grid-a5de2d6-20261012021243`: energy 131k행, power_samples 677k행
   - Job-EA `pdmux-n2-static-grid-7e09cc3-20261013022149`: energy 65k행, power_samples 336k행
   - 두 Job 모두 **`sm_clk_samples.csv`가 헤더뿐**이다. A100이 버퍼 클럭 샘플을 주지 않는 것으로 보인다. cap 하강은 100 ms `clk.csv`로 본다.
   - 원시 CSV는 `.gitignore` 대상이라 로컬에만 있다.
   ```bash
   cd workspace/engine-port/results/power_cap/n2_decode_power
   python3 pc_n2_decode_power.py --run-dir <N2 attempt/boot 디렉터리> --meta ../../n2_static_grid/vessl/<job>/meta
   ```
   - PROFILE은 decode 단독(W 해당), EA는 W/C/A2가 다 있다. EA 쪽에서 "같은 분할 쌍의 단독 vs prefill 동거" 대비가 나온다.
   - **N2 결과 감사가 PENDING인 동안 에너지 수치를 판정·보고에 쓰지 않는다.** 실행해도 기술 산출로만 둔다.
   - N2의 Job 레이아웃(attempt/boot 디렉터리 이름, marks/wall/tel 파일 위치)이 fake 시도와 같은지 먼저 확인한다.
2. **사용자 결정 대기**:
   - power cap 처치용 기판 확보 여부. 선택지는 VESSL VM Cluster(H100 이상)나 루트 외부 VM이며, KISTI는 제외다. 교수님 문서의 결정 요청 항목에 들어가 있다.
   - B안(prefill만 상태까지 보는 전용 서빙 Job) 여부
   - J/req 보조 열 병기
   - venue(워크숍급)
3. `reports/power_cap_viability_2026-10-07.md`의 기판 선택지(73행·§8)에 "KISTI 계정 복구"가 남아 있다. 사용자 지시 범위가 메모리라 손대지 않았다. 원하면 같은 기준으로 정리한다.

## 미완·주의

- **claims-auditor 미경유**: 센서 대역폭·파티션 전력·서빙 미리보기 RESULT 문서. 전부 기술 수준이고 정본 인용은 금지다.
- 파티션 전력은 합성 부하·n=1·클럭 미기록이다. 동시 실행 처리량 하락(0.83–0.93×)의 원인(메모리 대역폭 vs cap)은 분리하지 못했다.
- 금지 문안은 메모리 `power-cap-track` §금지 문안을 따른다. 예: "SM 분할 = 전력 분할", "cap은 decode에 무해", "sticky ON이 에너지 절감", 기판·Job 간 J/* 비교.
- 교수님 문서의 "아래 수치는 모두 KISTI A100" 문장 아래에 나중에 VESSL 에너지 수치를 넣으면 기판을 따로 표기해야 한다.
- Workspace `pdmux-probe`(A100)·`pdmux-build`(CPU)는 `standby` 상태다. 이번 트랙과는 무관하고, 이전부터 존재했다.

## GPU 실행 기록 (VESSL, 이번 세션 제출분)

전부 공개 이미지 `ghcr.io/asuan99/sglang-runtime@sha256:1e0d335b…`, `resourcespec-a100x1`이다. 표준 런처(`launch.sh`)를 거치지 않은 프로브라 캠페인 장부·`DONE`·`cost.txt`가 없다. 회수는 `vesslctl job logs/show`(+ object volume 다운로드)로 했고, 결과 디렉터리에 `SHA256SUMS`가 있다.

| job name | slug | 내용 | 상태 | 회수 |
|---|---|---|---|---|
| pdmux-power-cap-probe-20261007083542 | job-x3lzellrkm8z | 컨테이너 권한(`-pl`/`-lgc`/CapEff) | succeeded | `probe_job-x3lzellrkm8z/` |
| pdmux-energy-probe-20261008014846 | job-iwzlsxq6yu8t | 에너지 카운터 vs 적분 | succeeded | `probe_job-iwzlsxq6yu8t/` |
| pdmux-sensor-bw-probe-20261008045030 | job-absefkt9e88x | 사각파 센서 대역폭 v1 | succeeded | `sensor_bw/job-absefkt9e88x/` |
| pdmux-sensor-bw-v2-20261008050054 | job-tkgxusf12n0v | v2(버퍼 샘플) | succeeded | `sensor_bw/job-tkgxusf12n0v/` |
| pdmux-partition-power-20261008052457 | job-78xs9733hcjy | green-ctx 파티션 전력(합성) | succeeded | `partition_power/job-78xs9733hcjy/` |

미회수 Job과 켜 둔 Workspace는 없다(이번 세션 기준).
