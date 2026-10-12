# 하네스층 감사 판정서 — PREREG_MULTIGPU rev6 하네스 (M2g·M3), 커밋 999bf55

> 감사자: claims-auditor(읽기 전용, GPU 0), 2026-10-12. 판정문 요지를 메인 세션이 저장했다(필수 수정·caveat 문안은 원문 그대로).
> 감사자 직접 확인: 테스트 4파일 166/166 OK, `g5a_selector_equiv.py` PASS(1026 상태, 불일치 0), `mg_label --selftest`(변이 4종 KILLED), `m3-plan` boot 70·셀 365, `m3-budget` λ=prior 29.96 h·TP 0.7 33.15 h·확장 57–61 h, `m2g-budget` 5.48 h.

## 0. 판정
| 범위 | 등급 | 요지 |
|---|---|---|
| **M2g 하네스** | **`GO-with-caveats`** | 死因 0. 봉인 전 필수 수정 F1–F8. F1(G3 순서)·F2(G3b 생존)는 비정상 경로 fail-open — 수정 없이 봉인하면 死因 |
| **M3 하네스** | **`GO-with-caveats`(단일 Job 한정)** | 死因 0. 봉인 전 필수 수정 F9–F17. F10(확장 미완료 폴백)은 조건부 반전 — 수정 없이 봉인하면 N2. M3는 M2g 산출물(H7 표, H9 g, PASS 표 sha) 뒤에만 봉인. **블록 분할을 고르면 이 판정 미적용**(새 하네스·감사 필요) |

"이 하네스로 실행해도 된다"는 뜻이지 M2g PASS나 M3 스케일링 판정 능력을 뜻하지 않는다.

## 1. 등록 대응
문자 그대로 일치: 상수 전부, Q_c·g·r, HI/LO/KNEE, 확장 트리거(OK 셀 하나라도 g<0.10 ⇔ arm 비-HI_SATURATED), 공동 x_lo, λ\* 중앙값, G_sym 1차, bootstrap·추세 재표집, 크기 규칙, 앵커, GPU_HETEROGENEOUS, HARNESS_LIMITED, FRONTEND(역할 문자열 dev tree 존재 확인), E-time, G5(c)(i)–(iii), §3-1 FAIL 단위, G0(ii) `PD-mux with tp_size=`(`multigpu_guard.py:123`), G0(iii)(`pdmux_context.py:118`), G4 `Capture cuda graph begin/end`(`model_runner.py:2451/2465`), 실행 순서·arm 순열.
불일치·누락: G1 '비정상 종료 0' 절이 Job 경로에서 항상 False(`m2g_correctness_gate.py:903`), G1 노출 비율 미산출, G3b 생존이 마지막 step 이후 미검사, G5(a) GPU 기술치 boot가 `mg_plan.m2g_boots`에 없음, M2g G5(c)·G2·G1-UNINFORMATIVE 플래그 M3 미승계, GPU_CONTAMINATED가 config 전체로 확대.

## 2. 반전 시험 (실제 반전: F1·F2 비정상 경로, F10 조건부)
| 표면 | 변화 | 근거 |
|---|---|---|
| warm-up, server seed | 없음 | 창 밖·arm 대칭, temperature 0 |
| boot 입장 추정 | 라벨 불변, 잘리는 꼬리만 변화 | +49 s×70 boot ≈ +0.95 h → DP4 LO·NUMA·nsys·앵커 절단 |
| 확장 미완료 폴백 | **조건부 반전(F10)** | 확장 시나리오에서 TP2 fused_sym ext boot 입장 거부·LO 입장 → G_sym(TP2) 정의 → TP 추세 {1,8}→{1,2,8}, UNDEFINED → 라벨 |
| G3 실행 순서(Job 경로 맨 끝) | **비정상 경로 반전(F1)** | TP hang 1.73 h + G3b drain 1.08 h → 8.3 h > 8 h → G3 미시도 → TP 축 전체 제외(`mg_drive.py:194-207`) |
| G3b 서버 생존 | **비정상 경로 반전(F2)** | 생존 판정이 G3b 이전 step만 봄(`mg_drive.py:273-276`, `mg_boot.py:379-402`) |
| teardown poll, G5(c)(나) 시점, LO ext 태그, G2_ALIGN_MIN, G2 Δ 정의, watchdog | 없음 | — |
| G1 prompt | 예보 불가 → caveat | — |
| assemble_pass_table 어댑터 vs M2gRun | F1–F4 원인 | 테스트된 M2gRun을 Job이 안 씀 |
| bootstrap 통계량 | 없음 | 차이 < 0.5%, 조합 35개 |
| fused HI 미완 시 LO 전체 생략 | 정상 경로 없음 → MGH-11 | — |
| P-DP 미검정 시 broken=False | 규칙층 공백 → MGH-8 + F9 | — |
| boot 안 순서 무작위 | 예보 불가 → MGH-10 | — |

## 3. 항등식 위험
G5(a)는 엔진 원문 실행(가짜 torch·AST 추출, capability 읽기 guard, 변이 7 KILLED) — 항등식 아님. mg_label 변이 기대 클래스는 손으로 정함, **단 S7은 검사 대상 `g_of`로 문턱 위치를 맞춤**(`mg_label.py:349-355`) → MGH-13. mg_realized_mix 기대값 손 계산. **Job 경로(`mg_drive.run_m2g` + `assemble_pass_table`)는 AssemblePassTable 3건과 dry-run으로만 덮임**(F8).

## 4. M2g 첫 실행 공허 경로
G0(ii)(iii) 문자열 존재·rank별 RuntimeError → rc≠0(정상). 거부 boot가 정지하면 1800 s 후 FAIL → TP 축 제외(예산 최대 6×1800 s). G4 PASS는 group별 캡처 미증명(MGH-1). **piecewise ON fused 미부팅**: 캡처 실패 → G0(i) fused FAIL → (mode,N) 전체 제외, N1이면 M3 공허(R-0). proctitle 누락 시 FRONTEND 조용히 False(F13). **NVML 불가 → 포함 처리**(`mg_boot.py:182-183`, `mg_drive.py:293-294`)(F5). torchrun 없음/timeout → `mg_drive` 붕괴(F1). G3 스크립트 예외 → FAIL(실제는 NOT_TESTED, 교훈 21)(F1). nsys 없음 → 기술치만 손실. ENGINE_DIGEST manifest에 절대 경로(`sync_engine_tree.sh:187-215`) → 로컬 sha 봉인 시 Job `ABORT_ENGINE_DIGEST`(F7).

## 5. 예산·MGP-14
- §6-1 "≈290"은 산술 오류: 336 + 기술 29 = **365**(문서 追記).
- M3 λ=prior 29.96 h(여유 2.4분), VESSL 부대비용 미보정(N3 실측 +49 s/unit) → 'within cap' 사실상 불성립(F15).
- 정의역: (i) λ=prior·무확장이면 DP4 LO → NUMA → nsys → **N=1 2회차(앵커)** 순 절단, TP·DP 정의점 ≥3 유지·라벨 유지, 단 드리프트 측정 장치인 앵커가 잘릴 가능성 높음(앵커 이동은 규칙 변경 = 사용자 결정+델타 감사). (ii) 확장 시나리오(어느 arm λ ≳ 1.42·prior): 모형 57.4 h, 30 h 시점 TP·DP 정의 N {1,8} → 1차 추세 4개 TREND_UNDEFINED, P-DP 미검정.
- M2g 5.48 h, G3b drain 미반영(+0.5 h, TP 0.7이면 +1.1 h), 정상 경로 8 h 안.

## 6. 봉인 절차
1. ENGINE_DIGEST: (a) 같은 이미지·봉인 커밋 Workspace bring-up으로 `install_runtime.sh` manifest sha, 또는 (b) 경로 없는 형태(첫 열 hash 27개 digest 또는 항목별 대조, 앞 25개는 직전 VESSL manifest와 대조).
2. 봉인 묶음: 하네스 19파일 + `pdmux_agnostic_manual.yml`, `pdmux_r2.yml`, **추가** `results/r2_correctness/r2_correctness_client.py`, `triage/pdmux_a100_smoke.yml`, `benchmarks/pdmux_eval/analyze.py`, PREREG·VERDICT 전부, 테스트 4파일 sha, 봉인 트리 기준 g5a PASS JSON.
3. PASS 표: M2g 회수 → `results/multigpu/M2G_PASS_TABLE.json`·`.sha256` 커밋 → M3는 env sha = 커밋 `.sha256` = 파일 바이트 sha일 때만 진행(현재 env만, `m3_job.sh:159-163`). M3 봉인은 이 커밋 뒤.

## 7. 死因 0

## 8. 봉인 전 필수 수정
**M2g**
- **F1** G3를 서빙 boot보다 먼저. torchrun preflight 또는 `python -m torch.distributed.run`, `TimeoutExpired`·`FileNotFoundError` 처리, 조합별 stdout/stderr 저장, "디렉터리 있음·rank 파일·HANG 없음" = `NOT_TESTED(HARNESS_ERROR)`.
- **F2** 모든 boot 마지막 step 뒤 `poll()`·`/health` 생존 기록, G3b 생존·G1 비정상 종료가 이를 읽음, `cli_score_g1` 하드코딩 False 제거.
- **F3** Job 경로 G1 노출 비율(G1 창 E-time).
- **F4** G5(a) GPU 기술치 boot(N=1 agnostic auto config)를 `m2g_boots`에 추가(빼려면 등록 追記).
- **F5** NVML 불가 시 `CONTAM_UNVERIFIED` 명시(권고: preflight ABORT).
- **F6** g5a PASS를 봉인 트리 기준 기록.
- **F7** ENGINE_DIGEST 봉인 방법.
- **F8** M2gRun을 Job 경로로 쓰거나 Job 경로 실행형 테스트(순서·생존·노출) 추가.
**M3**
- **F9** P_DP 3값(BROKEN/HELD/UNTESTED), UNTESTED면 TP 라벨 공시 강제.
- **F10** 확장 발화 후 확장 HI 미완료 arm = `HI_NOT_JUDGED(EXT_NOT_RUN)`, 확장 미완료 config는 boot2(LO) 입장 금지.
- **F11** 오염·메모리 플래그를 (config, round, block) 단위로, 앵커·NUMA·gpu_dev가 round 1 main G를 지우지 않게, OR 합산(`mg_label.py:776-800`).
- **F12** MIX_FAILED·E_TIME_UNDEFINED를 estimand 항목에 전파(`mg_label.py:530-535, 678-680`).
- **F13** FRONTEND·throttle·send-lag 미측정과 역할 누락을 명시 플래그로.
- **F14** M2g config 플래그(TP_NUMERIC_DIVERGENT, G1_UNINFORMATIVE, M2g MEMORY_MISMATCH)를 LABELS로 승계, arm 결손 시 G5(c) 의미 통일.
- **F15** 예산 모형 재보정(VESSL 부대비용, arm별 ratio, 확장 분기, 30 h 절단 시 추세 정의역 예보표) → §6-1 사용자 결정.
- **F16** REGISTRATION_SHA256 보강(§6-2).
- **F17** PASS 표 sha를 커밋 파일과 대조.
- 문서: §6-1 "≈290" → "365" 追記.

## 9. caveat (문자 그대로 승계)
- **MGH-1** "M2g G4 PASS는 rank별 `Capture cuda graph begin/end` 쌍이 짝을 이루고 실패 문자열이 없다는 뜻이다. PD-mux의 SM group별 캡처는 따로 기록되지 않는다. 'TP PD-mux의 모든 SM group cudagraph가 캡처됐다'고 쓰지 않는다."
- **MGH-2** "G3 PASS는 PyNccl 두 communicator가 짝·홀 rank에서 역순으로 발행돼도 plain·green(16/92, 44/64) stream에서 500회 동안 hang과 정수합 오류가 없었다는 뜻이다(진행형 watchdog 120 s). custom all-reduce, cudagraph replay, 실제 모델은 범위 밖이다. 'TP PD-mux에 통신 hang이 없다'고 쓰지 않는다."
- **MGH-3** "G2의 Δ는 TP=1 자기 greedy 출력을 teacher로 했을 때 위치별 각 run의 top-1 logprob 차이다(teacher 토큰의 logprob 차이가 아니다). G2는 제외를 만들지 않는다. `TP_NUMERIC_DIVERGENT`·`G2_NOT_TESTED`는 M3 TP 결과 옆에 반드시 병기한다."
- **MGH-4** "G1의 등록 prompt 64개는 `r2_correctness_client` seed 20260911의 S tier 32개와 O tier 32개 텍스트를 동시성 32로 보낸 것이다(S/O 프로토콜이 아니다). G1 PASS를 다른 prompt 집합이나 길이로 확장하지 않는다."
- **MGH-5** "G3b PASS는 완료 요청 > 0, 서버 생존, 600 s 송신 창 마지막 60 s 안 토큰 진행이다. TP8 G3b의 피크 동시 연결(약 7.5–10k)은 H10 로컬 프로브(≤약 3k)가 검증한 범위 밖이다. G3b의 g는 그 셀의 send-lag·throttle을 함께 보고하기 전에는 'Q_c가 실데이터로 검증됐다'(MGP-33 해소)의 근거로 쓰지 않는다."
- **MGH-6** "하네스 dry-run 29.96 h(λ=prior)는 VESSL 실측 부대비용으로 보정되지 않은 모형값이다. 'M3가 30 h 안에 끝난다'고 쓰지 않는다. 하드캡 절단 순서는 DP4 LO → NUMA → nsys → N=1 2회차(앵커)다. 앵커가 잘리면 `ANCHOR_DRIFT` 미발화를 '드리프트 없음'으로 쓰지 않는다."
- **MGH-7** "N=1에서 어느 arm이든 λ*/prior가 약 1.42 이상이면 공동 확장이 발화한다. 그 경우 단일 30 h Job에서는 TP와 DP의 정의 N이 {1,8}만 남아 1차 추세 라벨 네 개가 모두 TREND_UNDEFINED가 될 수 있다(예보이며 측정이 아니다). 이 결과를 'G가 N에 따라 변하지 않는다'나 'P-DP 성립'으로 쓰지 않는다."
- **MGH-8** "DP 추세가 TREND_UNDEFINED이거나 DP 축이 M2g에서 제외되면 P-DP는 '깨지지 않음'이 아니라 '미검정'이다. 그때 TP 라벨은 'P-DP 미검정' 공시 없이 인용하지 않는다."
- **MGH-9** "seed 짝 CI와 추세 CI의 통계량은 '재표집한 seed들의 중앙값 비'다. n=4라 서로 다른 재표집 조합은 35개뿐이다. CI 폭을 정밀도 주장으로 쓰지 않는다."
- **MGH-10** "boot 안 shape·seed 순서는 (config, arm)마다 무작위다(미등록 하네스 선택). 그래서 seed 짝을 이루는 두 셀은 boot 안 위치가 다르고, seed 짝 CI는 boot 내 위치 효과를 포함한다."
- **MGH-11** "fused HI 4 seed 중 하나라도 OK가 아니면 그 (config, shape)의 LO를 네 arm 모두 돌리지 않는다(등록 문언보다 엄격한 하네스 선택). 이 경로로 미정의가 된 G는 사유를 따로 표기하며, LO 실패나 'G<1'로 읽지 않는다."
- **MGH-12** "`PARTITION_NOT_EXERCISED` 미표기는 PD-mux arm의 HI 4 seed E-time이 모두 OK일 때만 '분할이 실행됐다'를 뜻한다. MIX_FAILED가 하나라도 있으면 분할 실행 여부는 미정이다."
- **MGH-13** "M2g의 G1 PASS 판정은 동시성 32 배치 비결정성을 N=1 대조로만 보정한다. `mg_label` 변이 검사 S7은 검사 대상 함수로 문턱 위치를 맞춘 시나리오라, S7에서 대조군이 통과하는 것은 독립 증거가 아니다."
- **MGH-14** "M2g PASS 표가 포함한 (mode, N, arm)은 '이 모델·이 prompt 집합·이 길이에서 G0–G5를 통과했다'는 뜻일 뿐이다. 'TP/DP PD-mux가 정확하다' 일반으로 쓰지 않는다."

## 10. 자기 검사·위험
- F1–F17 모두 CPU 코드·문서 수정. F7(a)만 Workspace bring-up 1회(측정 아님).
- 앵커를 NUMA·nsys 앞으로 옮기는 것은 처방이 아니라 규칙 변경(사용자 결정 + 델타 감사).
- **R-0(위험)**: piecewise ON NemotronH fused 미부팅(MGP-36). N1 fused 기동 실패 시 M3 통째로 공허. 사전 Workspace 프로브(fused cp8192, N=1, boot만) 여부는 사용자 결정.

## 11. 다음 gate
F1–F17 반영·국소 재확인 → ENGINE_DIGEST 봉인·M2g 봉인 묶음 커밋 → M2g dry-run → 사용자 승인 → 제출 → 회수 → PASS 표·sha 커밋 → M3 하네스 봉인 → 예산 재예보로 사용자 결정(단일 Job/블록 분할) → M3 제출.
