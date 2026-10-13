> 이 날짜 파일에는 두 세션의 핸드오프가 있다: (A) N2(3-a) 세션(아래 첫 부분) · (B) power_cap 세션(커밋 d90f226 원문, 아래 두 번째 부분). 2026-10-13 N2 세션이 처음에 (B)를 덮어써서 원문을 복원해 합쳤다.

# 세션 핸드오프 — 2026-10-13 (2026-10-11 시작, 10-12·10-13 속행; VESSL A100 + 로컬 CPU)

## 이번 세션 요약

2026-10-11 핸드오프(rev17 진행 중)에서 이어받아 N2(3-a)를 **사전등록 → 봉인 → 실측 → 결과 감사 → 정본 반영**까지 끝냈다. rev17 규칙 재봉인, 하네스 PROFILE 봉인, Job-PROFILE 실행(첫 제출은 preflight에서 실패 → H17-10 수리 → 재제출 완주), PROFILE이 "판정을 바꾸지 않는 실험"이었다는 사용자 지적 뒤 동거 관문을 세웠다가 감사에서 철회(안 0), Job-EA 하네스 구현 → rev18 규칙 재봉인 → PROFILE+EA 하네스 봉인 → **Job-EA 완주**, 결과 감사 `死因 0`(라벨 8/8 `COLOC_SLOWER_GE3` CONFIRMED(scoped), 크기는 전력 한도와 교락)까지 진행했다. 그 밖에 VESSL 로그 진행 표시, 작업 검토(review-lead), 진행 점검 아티팩트 갱신을 했다. **새 성능·정책 판정 0건**, HE0·게이트 #6·Claim D/E 불변. 정본은 rev95.

## 결정·측정

**사용자 결정(시간순)**
- 2026-10-11: N3 shape A **보류**(N2 먼저); G-10 = VD 셀 중단 시 **그 boot만 종료하고 계속**; CG 카나리아(H17-3) 채택, H17-4/5 미채택(N2C-79); 장부는 봉인 바이트 확정 뒤 1회(메모리 `ledger-runs-last`), 러너 단축은 다음 재개봉에.
- 2026-10-12: power_cap 미커밋 2파일 커밋 후 제출; job_entry 로그 진행 표시; 리뷰 후속 1–3 진행; Job-PROFILE이 판정을 바꾸지 않는다는 지적 → 동거 관문 + Job-EA 준비 병행 → 관문 감사 `NO-GO` 뒤 **안 0(관문 폐기, 동거율은 기술값만)**; 블라인드 노출 처리 = **K(33,15)·C 창·κ\*≥3.0 동결, 개정 창 폐쇄**; §10-2 외삽 + AON_CONTAINMENT 거부율 병기; **E14-1 하드캡 16,100 s 재승인**; EA 의미 카나리아(H18-R1) 채택, H18-R2/R3 미채택.
- 2026-10-13: Job-EA 제출 승인.

**측정**
- Job-PROFILE `pdmux-n2-static-grid-a5de2d6-20261012021243`: 완주, 러너 `PROFILE_PARTIAL` 56/60 — 빠진 4점은 하네스 결함이 아니라 등록 ŝ 모형이 "첫 span 단독(216/216)"을 빠뜨려 fail-closed가 발화(`CHECK_profile_a5de2d6_2026-10-12.md`, N2C-88); throttle 측정 창 겹침 0/264. **경량 확인만**(판정을 바꾸지 않는 실험이라는 사용자 판단). 그림 `results/n2_static_grid/figures/fig1–3`.
- 작업 B(관문 아님): N3 shape B 동거율 f(`coresidency_a_decode`)·g 기술값(`n3_capacity/CORESIDENCY_B_2026-10-12.md`) — f의 66–89%가 prefill 커널 완료 대기 구간, f = W-dec 점유와 코드 항등.
- **Job-EA `pdmux-n2-static-grid-7e09cc3-20261013022149`**: 64 복제 전부 창 성립, 러너 값 비트 단위 재현. 결과 감사 `VERDICT_result_n2_ea_7e09cc3_2026-10-13.md`: ρ_off·ρ_on 8/8 `COLOC_SLOWER_GE3` CONFIRMED(scoped); ρ ≈ 1.13–1.18은 **동거 + 전력 한도 클럭 하락의 합성값**(C 창 64/64 SW power cap 371–382 W/400 W, A 창 0/128) — 전력 무관 간섭 크기 `NOT-YET-SUPPORTED`; 클럭 성분 최대로 빼도 64/64 > ln 1.03(최소 0.069, 존재만, 크기 인용 금지); "D·b·c 무관"은 사후 기술값(NOT-YET-SUPPORTED); host 발사 비용 가설 반증; merge-then-filter 확인(batch 합류 효과 아님); DS3 8/8. 허용 문안 (P)(A)(B)(C)(D)(K)·금지 32–42·N2C-102…108은 판정서와 정본 `PROJECT_STATUS.md` (a⁶).

## 코드·문서 변경 (전부 커밋됨)

주요 커밋(이 세션, 다른 세션 커밋 제외): `5e29990` rev17 규칙 봉인 · `97a91b5` 하네스 PROFILE 봉인 · `52a3b97` energy.csv mono_s · `a5de2d6` H17-10 재봉인 + 실패 Job 기록 · `7e43c61` job_entry 로그 거울 · `b184b10` 회수물 · `dc59129` 리뷰 D1–D3/D7 · `85fd126`·`5a35d57` 정본 rev94 · `0ec7922` 그림 · `4fb0899` 관문 철회 · `5f25bad` 경량 확인 · `e1c3f0c` 작업 B · `b1de10c` Job-EA 하네스 체크포인트 · `5ea2a60` rev18 규칙 봉인(`56473d6f…`) · `07dd368` H18 수리 + EA 카나리아 · `7e09cc3` 하네스 PROFILE+EA 봉인(`9257845f…`) · `49c933b` Job-EA 회수 + EA 결과 봉인 · `074d1d5` 재집계 · `fa454c4` 결과 감사 · `b035291` 정본 rev95.
- `workspace/engine-port/results/n2_static_grid/`: PREREG rev18, 하네스 전체(EA 경로, D4–D6, H17/H18 수리, 카나리아 2종), 러너 단축(`--no-targeted`, `--jobs`), `ledgers/`(rev17·rev18 core, rev18 H·JH 장부 원문), 판정서 10여 건.
- `scripts/vessl/job_entry.sh`: `[PHASE]`/`[T]`/`[HB]` 진행 표시, `tee -p`, 로깅 노브 unset(운영 문서 §12).
- `power_cap/n2_decode_power/pc_n2_decode_power.py` 3열 energy.csv + `tests/test_pc_energy_layouts.py`.
- 진행 점검 아티팩트 https://claude.ai/artifact/33Bfk4YtxKbd2pt61yyjcW (버전 3, 2026-10-12) — **Job-EA 결과 미반영**(갱신 필요 시 다음 세션).

## 열린 항목 / 다음 세션 시작점

1. **N2 후속 측정 결정(사용자)** — 결과 감사 §10, 처방 아님: F-1 같은 boot 안 전력 정합 대조 arm(C_burn; VESSL은 `-pl`/`-lgc` 불가가 실측돼 전력 정합 방식만, 서버 내 green-ctx burner 주입 경로·≈376 W 도달성 미확인), F-2 `clocks.mem`·NVML 버퍼 SM 클럭 수집, F-3 운영 rate trace에서 동거/단독 비·cap 노출(N2C-107을 닫는 유일 경로), F-4 무관성은 확장 격자 + 등록 동등성 라벨로만.
2. **Job-PROFILE 셀·boot 동거 4추정량 산출 여부** — 규칙상 시점(EA 결과 봉인 뒤)은 왔으나 블라인드 노출 2건(N2C-87·98)·봉인 의미 한계(N2C-95) 때문에 claims-auditor 확인 후 산출 권고.
3. **사다리 방향(사용자)** — "분할해도 동거 감속이 남는다"는 성분 확인됨. 정책 차이로 이어지는지는 서빙 실증(S0′ → S1 → 5-b). 계속 vs 정리.
4. 정본 `§1 행 38`(N2 Job-EA 성분 측정 행)은 doc-steward 판단으로 신설 — 원치 않으면 되돌림.
5. 대기 중인 오래된 사용자 결정(아티팩트 §9): 기판 동등성, 5-b·4-a arm, A11 모델 hold-out·대조 모델, 정의 사각지대 D-1…D-12, E2 후속, N3 확정 조건 ①–④.
6. 비차단 정리(다음 하네스 회차): `n2_probe_run.ENG` env 우선, `test_n2_job.REPO` 삭제; N2C-90 러너 의미는 하네스 장부 겨냥 대조로만 검증 가능.

## 미완·주의

- **블라인드 노출 2건**(분석자 N2C-87, 감사자 N2C-98) — E-α 인용 시 "블라인드가 지켜졌다" 금지. 감사 지시에 원자료 경로 금지를 명시하는 관행 유지.
- Job-EA 크기(ρ ≈ 1.16)를 "간섭 크기"로 인용 금지(N2C-102); 정책·static·d_min 함의 금지(금지 37).
- 인용금지 검사가 원자료(clk.csv·JSON) 숫자에서 오탐 — 문서 위반 아님. PROJECT_STATUS 기존 17건·CONSENSUS 기존 10건은 미정리(`[CS-OK]` 여부 사용자 결정).
- 다른 세션 트랙의 VESSL Job(`pdmux-multigpu-*` 2건 running, `pdmux-opchar-*` succeeded)은 이 세션 소관 아님.
- Workspace `pdmux-probe`(A100)·`pdmux-build`(CPU) 둘 다 standby.

## GPU 실행 기록(VESSL)

| job name | 캠페인 | 커밋 | 이미지 | 상태 | fetch | cost.txt |
|---|---|---|---|---|---|---|
| `pdmux-n2-static-grid-52a3b97-20261011182211` | n2_static_grid PROFILE | `52a3b97` | `sglang-runtime@1e0d335b…` | failed exit 4 (preflight, 셀 0, 결과 아님 N2C-85) | DONE+sha256 OK (`a5de2d6`에 기록) | 0.011 h |
| `pdmux-n2-static-grid-a5de2d6-20261012021243` | n2_static_grid PROFILE | `a5de2d6` | 같음 | succeeded exit 0 | DONE+sha256 OK (`b184b10`) | 3.801 h |
| `pdmux-n2-static-grid-7e09cc3-20261013022149` | n2_static_grid EA | `7e09cc3` | 같음 | succeeded exit 0 | DONE+sha256 OK (`49c933b`) | 1.893 h |

---

# (B) power_cap 세션 핸드오프 — 원문 복원(커밋 d90f226)

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
