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
