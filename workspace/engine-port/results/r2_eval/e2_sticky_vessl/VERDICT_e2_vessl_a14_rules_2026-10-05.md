# 규칙층 감사: E2 VESSL 재실행 追記 A14 (R1 새 20 boot · P3 (ii) · O1–O4), claims-auditor, 2026-10-05

> 읽기 전용, GPU 0. 감사자 최종 보고를 메인 세션이 옮겼다. 성능 판정 0건, Claim 등급 변경 0건.

## 최종 판정: **`GO-with-caveats`** (死因 0)
제출 승인이 아니다. 제출 전 필수 수리 4건 → 하네스층 봉인 → presubmit 차단 2건 확인 → 커밋 → dry-run 승인. 새 caveat **E2VC-12…16**.

## 0. 독립 실측
- HEAD `69fbbb0`, 추적 파일 변경 0, 새 파일 5개 sha 일치. 1회차 봉인 7행 + OVERRIDE `131abfff…`(= `git show 9a29f84:`) 불변.
- 테스트 재실행 89/89(r2 41 + r1 48). helper r2/r1 셀프테스트 OK, `--verify-digests-r2` OK(6 sealed).
- 패치 재구성: 1회차 러너 + 초안 패치 2개 vs r2 diff = A14 §2-1 나열 변경뿐(settle 위치, P3 helper 호출, mamba 48, 재개 거부, digest 확인, 주석·배너).
- r1→r2 diff에서 SRV/WARM/BENCH·노브·P13 heredoc·추정량·라벨 호출·`--units`/`--expect`/`HARD_CAP_S`/`T_cap`/`P14_MARGIN_S` 바이트 동일.
- 엔진 측정 경로: `git diff e6f424e HEAD -- src configs e2_sticky_prereg scripts/bootstrap env` 빈 diff.

## 1. P3 (ii) 사후 완화(교훈 251 양방향)
- 완화 표면(토큰 상대차 ≤ 1e-4): 1회차 키 rel = 1.102e-6 PASS; 변이(mamba 47/48·전부 47·0.8% 차·786,431·배너 누락·helper 부재·거짓 PASS) FAIL; 중화 변이 검출.
- 판정량 경로 독립 확인: `FixedPolicy.decide`(controller.py:88-104)는 snapshot 미사용, CoarseGrainedController 미경유, sticky/upstream 분할(mixin:1169-1195)은 `decode_bs`/`_sticky_fixed_idx`만, `kv_occupancy` 0.85/0.90은 generic/hybrid 전용. "3 토큰이 achieved에 닿는 경로 없음" 반증 실패 — 단 코드 경로 탐색뿐(2차 경로 배제 증명 불가, A14 자기 공시 적정).
- **1e-4는 사용자 결정 상수(자유 표면)**. 새 사실: 같은 모델·mem 0.82에서 **6그룹 config는 2,721,290**(KISTI 907959 `srv_L1`, 908179 `srv_L2`, N3 probe A1/A2) vs 5그룹 2,722,025 → rel 2.70e-4. 1e-4는 5↔6그룹 drift를 잡고 1e-3은 놓치나 `CONFIG_DIGEST`가 먼저 abort.
- 조임 표면(mamba=48 정확, 하한 786,432): 관측 36 boot 전부 48, 최소 토큰 2,722,022 ≫ 786,432, B KV 수요 상한 396,288. 거짓 abort 없음. (a)는 사실상 config 가드.
- 토큰 수준값 2,722,025 비강제 → E2VC-7 병기 의무 유지.

## 2. O1–O4
- **O4 위치(settle → P14 앞)**: P14 문턱식 바이트 불변, 평가 시점만 이동. 코너별 입장 수 불변(normal/observed/worst 20/20, worst+settle 20/20 최소 여유 568 s[앞]/748 s[뒤], capped+settle 16/16). worst+settle 12,717 s = 3.533 GPU-h < 13,800 s·4.0 h.
- O1 창 20000–20799, 겹침식(`hi≥EPH_LO ∧ lo≤EPH_HI`) 옳음, exit 5. 단 검사가 preflight **뒤**라 범위가 다르면 preflight 비용(≤600 s ≈$0.25) 손실(위험 낮음).
- O2 `ss -Htan`: TIME_WAIT 변이에서 구 검사 놓침·신 검사 잡음. SGLang 내부 TCPStore 포트(port+100…1000, 미확인)도 전 상태 검사가 덮음.
- O3 `^sglang::` 기존 순서 뒤, bench 클라이언트 비일치.
- watchdog 패턴 미수정·PDEATHSIG 의존 공시 사실.
- **결함(caveat)**: O4 정지 조건 `memory.used ≤ base+64 MiB`(MiB 해상도)는 −3 토큰급(≈40–80 KB) 잔여를 **해상하지 못함**. "O4가 H1을 겨냥"은 예방으로 불성립 → E2VC-14.

## 3. 측정 경로 불변·재개 제거
§0 확인. `PDMUX_E2V_RESUME_FROM` 설정 시 등록 전 exit 2; 가드 제거 변이는 20 boot 진행(항등식 아님); 재개 코드 잔존 0. "재개는 digest로 이미 막힘"(OVERRIDE 46c85d71→131abfff) 1회차 코드로 확인.

## 4. A14 문안
- λ0 줄 번호 정정 맞음(`lambda0.sbatch:316-321`, `lambda0_vessl.sh:372-377`, `e2_sticky.sbatch:200-205`, `e1_cap_vessl.sh:188-194`, `substrate_probe.sh:405-409` 모두 39100창+`ss -ltn`; `n3_capacity.sh:271,280` 수리됨). KISTI ephemeral "미확인" 사실.
- P3 전례 = N3 rev3 D2와 동일 사실(KISTI 11 boot 전부 …025 재집계 일치). R1 사유 독립. D23/E2V-11·E2VC 승계 있음.
- 결함: E2VC-7 첫 문장을 문언 승계하면 새 Job에서 지시 대상 틀어짐 → E2VC-16. 사소: OVERRIDE R2 §5-1 두 Job 합계 $4.74는 금액 합, GPU-h 기준 $4.73(정보용).

## 5. digest 체계
helper r2가 A14·r2 러너 고정, Job 내 preflight 전 `--verify-digests-r2` → `ABORT_REG_DRIFT`, 순환 없음. helper r2 자신은 Job 내 비강제(1회차와 같은 설계; 커밋+봉인+`test_constants`로 보호). `SEALED_R1`에 `test_e2_vessl.py` 포함은 의도(가용성 문제일 뿐). 판정서 추가·`DESIGN_A1_REV2_STICKY` 수리는 digest·glob에 안 걸림.

## 6. 예산(재계산 일치)
등록 보통 1.991 / 관측 보정 2.242(a_r2 ON 375 s 가정) / 최악 2.533 / 최악+settle 3.533 / capped 3.591(19 boot) / 하드캡 4.000 GPU-h. 1회차 0.957–0.970 합산 3.20–3.21 / 4.49–4.50 / 4.96–4.97. `BUDGET_TABLE.json`은 게이트 미사용.

## 7. 테스트
41/41·48/48 재현. P3·O1·O2·O4 위치·kill 층·재개 가드 모두 대조 변이 실재, null 대조 있음. 미수행 공시: HEAD export 대조, 컨테이너 dry-run, DRYRUN에서 settle·실제 bind.

## 8. presubmit
doc_facts 1건(853 vs 940) 잔존, OVERRIDE R2 미포함 — 제출 전 수리 필수.

## 9. 1회차 결과 열람 후 설계 변경 여부
셀·seed·arm·순서·R1–R6·문턱·추정량·T_cap·하드캡 **불변**(바이트). 바뀐 것: (a) P3 abort 조건(라벨 아님) (b) R1(digest로 강제된 선택) (c) settle 간격. 1회차 a_r4 d_i/OFF −29.25…−30.09%(범위 0.84%p)로 부호 반전 예보 불가 — "불리한 4짝을 버리고 다시 굴렸다" 경로를 지우려면 두 Job 병기가 의무 → E2VC-13.

## 반전 시험 표
| 자유 표면 | 민 범위 | 판정 변화 | 근거 |
|---|---|---|---|
| P3 (c) 허용오차 | 0…1e-3 | 1회차 키: <1.102e-6 abort, ≥ PASS; 1e-5/1e-3 PASS 불변; 라벨 변화 0 | rel 1.102e-6; KISTI·λ0 rel 0 |
| 같은 표면(config drift) | 1e-4→1e-3 | 5↔6그룹 drift 놓치나 `CONFIG_DIGEST` 선행 abort | 2,721,290 vs 2,722,025 |
| P3 (a) mamba | "한 값"↔"=48" | 36 boot 전부 48 | — |
| P3 (b) 하한 | 0↔786,432 | 최소 2,722,022 | B 수요 396,288 |
| O4 위치 | 뒤↔앞 | 입장 수 불변 | 여유 568/748 s |
| settle 상한 | 0…180×20 | 20 boot 입장, 3.533<4.0 | 12,717<13,800 |
| R1↔R2 | 새 Job↔재사용 | 부호 반전 경로 없음(라벨 미계산) | d/OFF −29.25…−30.09% |
| ON 375 s 가정 | 375↔worst | 게이트 미사용 | — |
반전 0행 → 전부 caveat.

## 제출 전 필수 수리(A14 바이트 불변)
1. **doc_facts**: `DESIGN_A1_REV2_STICKY_2026-08-25.md` 회귀 수를 커밋 시점 최종값으로(doc-steward) → presubmit 차단이 정확히 M4R·TC1 2건 확인 → 출력을 OVERRIDE R2 §1-1에 문자 그대로 追記.
2. **하네스 봉인 표**: 새 5파일 + 1회차 7+1행 sha를 커밋 **후** 재확인.
3. **HEAD export 대조**: 커밋 후 `git archive HEAD` 트리에서 `e2_vessl_plan_r2.py --verify-digests-r2` + `e2_vessl_plan.py --verify-digests` OK(CPU).
4. **사용자 재확인**: OVERRIDE R2 §7 ⚠️(금액·presubmit 현황이 승인 뒤 작성)를 필수로 — 제출 창에서 §5-1 표(보통 2.242 / 최악+settle 3.533 / 하드캡 4.0 GPU-h, 두 Job 합계)를 다시 보이고 확인 기록.

## caveat(결과 문서·정본 승계) — E2VC-1…11, E2V-1…11, E2C-36…39, 1회차 OVERRIDE §2-1·§4 유지 + 신규
- **E2VC-12 (P3 (ii) arm 상관)**: "토큰 키가 둘 이상이었으면 결과 문서는 boot명(arm·seed·순번)별 값을 싣고 서로 다른 키가 arm을 완전히 가르는지 적는다. 가르면 'P3 (ii)가 arm 간 KV 풀 차이(≤1e-4)를 통과시켰다'고 명시한다. 1회차 이탈 1건은 OFF boot였고 ON 4 boot는 모두 2,722,025였다."
- **E2VC-13 (1회차 병기 의무)**: "새 Job의 a_r4 R2 라벨 옆에 1회차 중단 Job a_r4 4짝 기술 통계(d_i/OFF −29.25…−30.09%, 4/4 음수, 라벨 없음)를 반드시 나란히 적는다. 방향이 다르면 머리말에 적고 '새 Job이 1회차를 대체/정정했다'고 쓰지 않는다. 1회차 숫자로 R1–R5를 계산하지 않는다(E2V-11)."
- **E2VC-14 (O4 해상도)**: "O4 settle 정지 조건(`memory.used ≤ base+64 MiB`)은 −3 토큰급(≈40–80 KB) 잔여를 해상하지 못한다. 'O4가 H1 잔여를 막았다/배제했다'고 쓰지 않는다. 실효는 `SETTLE.txt` NVML 바이트 기록이며 그 판별력도 미검증. 재발이 없으면 'O1–O4가 −3 토큰을 해결했다'가 아니라 '이 Job에서는 이탈이 관측되지 않았다'고만 쓴다."
- **E2VC-15 (허용오차 출처·포트 잔여 위험)**: "1e-4는 사용자 결정 상수이며 데이터 보정이 아니다. 판별 근거는 Job 내 이탈 1.10e-6(PASS)과 5↔6그룹 2.70e-4(FAIL) 두 점뿐. O1/O2는 포트 충돌 확률을 줄일 뿐 0이 아니다. BOOT_FAILED가 한 번이라도 나면 재개가 없으므로 그 셀의 R2는 n<4 `UNRESOLVED`, `PORT_RANGE.txt`·`BOOT_FAILURES.txt` 병기."
- **E2VC-16 (E2VC-7 지시 대상)**: "E2VC-7 첫 문장 '이 Job 전에 GPU 부팅 이력이 없다'는 1회차 중단 Job에 대한 진술이다. 새 Job의 엔진 측정 코드는 e6f424e 이후 diff 0이며 1회차에서 11 boot를 거쳤으나 그 boot들은 측정 타당성 근거가 아니다(라벨 0). P3 수준값 병기 의무 유지."

## 봉인 sha 표(2026-10-05 감사자 실측)
| 파일 | sha256 |
|---|---|
| `PREREG_E2_VESSL_A14_2026-10-05.md` | `524a08ea7b19ca238e4b69197d61f088a2b003a87f7f75df558ce21f907b2c5a` (= `A14_DIGEST`) |
| `e2_sticky_vessl_r2.sh` | `d4a146a9f8df65fe22470fe2a689e054d4a5b7161dd58b75c2e2a51b89999fb9` (= `RUNNER_R2_DIGEST`) |
| `e2_vessl_plan_r2.py` | `a7a6feb095290ab9ecbac5013ca908eae431edc6744e255d0f1116d7a7fd8fdb` |
| `tests/test_e2_vessl_r2.py` | `c5957813df38148885dba844e62296f4fa50ca6c5827bb3c447a628473028bba` |
| `OVERRIDE_E2_VESSL_R2_2026-10-05.md` | `a5b2b51e32368eaf3fdc9cda8ecc47c5c3504a528b33fb1d4b762122fef687e0` (봉인 예외: §1-1·§7 追記 예정) |
| 1회차 7행 + OVERRIDE `131abfff…` | 불변 |

## 반증 실패 기록
r2 = 1회차 + 패치 2 + 공시 변경 · 측정 명령줄·노브·셀·추정량·라벨·T_cap·하드캡 불변 · 엔진 측정 코드 diff 0 · 3 토큰 경로 없음(코드) · 36 boot 키 재집계 · 줄 번호 5건 · N3 D2 동일 · 재개 가드 대조 변이 · O4 위치 입장 수 불변 · 예산 6층 · 테스트 89/89 · presubmit doc_facts 재현. 실행 전이라 성능 주장 아님.

## 실현가능성 자기검사(게이트 #113)
수리 1–4는 로컬 CPU 또는 사용자 확인, A14 바이트 불변. E2VC-12/13/15 근거 파일은 r2 러너가 이미 생성. 이 판정서는 digest·glob에 안 걸림. 처방 중 명령줄·arm·규칙 변경 0건.
