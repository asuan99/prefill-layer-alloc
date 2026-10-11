# 판정서 — N2(3-a) rev17 좁은 규칙 델타 감사 + Job-PROFILE 하네스층 감사 (2026-10-11)

> 감사자: claims-auditor (읽기 전용). 저장: 메인 세션(2026-10-11), 본문은 감사 보고 그대로.
> 대상: `workspace/engine-port/results/n2_static_grid/` 작업 트리(미커밋) — PREREG 감사 바이트 sha256 `0dbdfffbe17f8aaaf4f514a27889a21a4bed2d88aa1392f53c1ab514a9b19322`, `test_n2_probe_reduce.py` `a796980c…`, `n2_mutants.py` `ffb4d6f3…`. 기준 = 규칙 봉인 `fcc0b9e`(digest `fbb13b15…`), 하네스 `1037655`(미봉인).
> 변이 러너(`n2_mutants.py`, `n2_job_mutants.py`)는 돌리지 않았다. 직접 실행: `test_n2_harness`, `test_n2_job`, `n2_reach_tests` 단위 시험, `n2_plan` CLI 검사, `n2_realized_mix --positive-control`.
> 새 성능 판정 0건 — 등록 감사다.

## 0. 결론 요약

| 대상 | 등급 | 死因 |
|---|---|---|
| **규칙 봉인 갱신(rev17)** | **`GO-with-caveats`** — 조건: R17-1…R17-3 문언 수리 + `@@CORE17@@` 채움을 같은 봉인 갱신 커밋에 넣는다(§3) | **0** |
| **하네스 봉인(Job-PROFILE)** | **아직 불가(보류)** — 선행 조건 미충족 + 2단 봉인 미등록(R17-2); N1–N4 해당 死因은 아님 | **0** |

판정 경로(K·창·라벨·예산·한도) 불변 주장은 반증되지 않았다(계산기·축약기 바이트 동일). 새 시험 2개는 J28·J29를 코드상 사살한다. 낡은 수치 문언 2곳이 남아 있고 §21-31 자기검사가 그것을 놓쳤다(R15-2 계열 재발). 하네스에서는 fail-open 2건(greedy `output_ids` 부재 H17-1, VD 정지 시 F-N2-7P 거짓 FALSIFIED H17-2)을 찾았다.

## 1. 규칙 델타 감사

### 1-1. "판정 경로 불변" 검증
- `n2_feasibility.py` sha256 `1ee3338d…ab0c60`, `n2_probe_reduce.py` `8ef0c8cc…4282d` = `REGISTRATION_SHA256.txt`(fcc0b9e) 규칙 절과 같다. `N2_REGISTERED.json`(`e7d743c8…`)도 같아 재생성 불필요 주장이 성립한다. `n2_plan --budget PROFILE` = cap_s 36,700(`PROFILE_b`).
- 바뀐 규칙 파일은 PREREG, `test_n2_probe_reduce.py`(+14줄, 시험 2개), `n2_mutants.py`(J28·J29, `JUDGMENT_IDS` 1–29, docstring) 셋뿐이고 판정 코드 diff는 0이다.
- **`CTX_UNREALIZED` 문언 정정(§6-2·§9-1)**: 코드(`alone_status`, `ctx.get(seg[0])`, seg = `_alone(...)[TRIM:]`)와 일치한다. 귀결이 "라벨만"이라는 주장도 확인했다 — 두 사유 모두 `n2_window.PREDICATE_FAILURE`·같은 VD이며 점·D·dsens 라벨·예보 어디서도 구별되지 않는다(`n2_analyze`의 `CTX_UNREALIZED` 참조 0). 다만 사유 귀속 의미가 바뀐다: trim 앞 시작이 (c−96, c−80]인 실패 창은 원인이 종료 쪽이어도 `CTX_UNREALIZED`가 된다 → caveat N2C-76.

### 1-2. 새 시험이 AU1·AU3(J28·J29)를 사살하는가 (변이 머리로 적용)

| 변이 | 앵커(유일성) | 사살 시험 | 기전 |
|---|---|---|---|
| J28 `else 0.0`→`else 1.0` | `n2_probe_reduce.py:267`, 1회 | `test_pe_time_zero_weight_is_fail_closed` | 창 128개가 iter_seq 짝수뿐 → 연속 쌍 0 → den=0; 원본 0.0, 변이 1.0 → `assertEqual(…, 0.0)` 실패. **사살** |
| J29 `<=`→`<` | `:331`, 1회 | `test_alone_prefill_at_window_endpoints` | `_inject`는 **짝 없는** `prefill_span_start`를 넣어 `spans`·t_burst_end·창을 바꾸지 않고 `span_ts`에만 들어간다; 시각 = `w[0].t_launch`·`w[-1].t_sync` 정확히 일치; 원본 n_pf=1 → `D_UNREALIZED`, 변이 n_pf=0 → VALID. **사살** |

한계(공시): H16-1 처방은 "가중 합 0 창 → `D_UNREALIZED`/`COLOC_UNREALIZED`" **라벨 수준** 시험이었고, 반영된 것은 `_pe_time` **함수 수준**이다. 두 호출 지점이 같은 함수를 쓰고 호출 지점 변이(J21·J26 계열)가 기존 장부로 묶여 있어 등가 생존 경로는 찾지 못했다. 실제 사살은 메인 세션 `--set core` 결과로 확정한다(`@@CORE17@@`).

**`@@CORE17@@` 채움 조건**: (a) 194행 = 192+2, J28·J29 `KILLED`, 예상 밖 생존 0, BROKEN 0, `.pyc` 0; (b) CTRL skip **정확히 1**(`test_repo_registered_file_is_current`) — 인자 `N3_RAW_DIR` 누락 시 임시 디렉터리 기본 경로가 없어 `test_positive_control_n3`도 skip되는데 CTRL은 OK로 나온다(러너가 skip 수를 검사하지 않음, R17-3); (c) 장부 시작 시각 > 세 파일 마지막 수정 시각(S-2 출처 형식).

### 1-3. N3 경로 의존과 등록 문언
- `test_n2_probe_reduce.N3_DIR` = `N2_N3_DIR` 또는 `../n3_capacity/n3b_pdmux-n3-capacity-38ed3f4-20261006092813`, `skipUnless(isdir)`.
- `vessl/<job>` 원본 디렉터리는 중첩 구조라 `glob(tel_g_*.jsonl)`이 비어 `set(th)=∅`로 **실패**(fail-closed, 메인 세션 관찰과 일치). 존재하지 않는 경로는 **skip**(조용한 통과).
- §18-2-b는 n3b_ 경로·manifest `bec2cba5…`를 "기본값"으로 적지만 (i) `vessl/` 원본이 유효 입력이 아님 (ii) "skip 0이어야 장부 유효"가 등록 문언에 없다. §15 rev16 기록 문장 "`N2_N3_DIR` 또는 저장소 상대 경로"도 모호함을 남긴다 → R17-3.

### 1-4. 계약 공백 15항목 중 판정을 바꿀 수 있는 정책 성격 항목

| # | 항목 | 판정을 바꾸는가 | 사용자 결정 |
|---|---|---|---|
| **7** | **G-10: VD 중단 → Job 정지** | 라벨 **부호**는 안 바뀐다(남은 boot 결측 → `UNRESOLVED_N_BELOW_4`/`D_PROFILE_PARTIAL` 쪽으로만; 4/4 규칙이라 거짓 GE3/INVERTED 없음). 그러나 **커버리지·비용은 크게 바뀐다** — `stream_cancel_failed`(§19-5 미실증)·`client_exception`·`cell_timeout` 1회로 ~10 h Job이 그 자리에서 끝난다. 또 **F-N2-7P가 거짓 FALSIFIED**(H17-2) | **필요(확인 권고)** — 근거가 프로브용임을 공시만 하고 메인 세션이 "진행" 범위로 정했다. 대안("VD → 그 boot만 종료, 다음 boot 진행")도 부호를 못 뒤집으므로 비용 대 커버리지 선택이다 |
| 2 | 셀 마감 기준점 = 셀 구간 시작 | 아니다(승인 문언 "셀 구간 캡"과 일치) | 불필요 |
| 9 | OBSERVER 우선순위 | 아니다(§11-4 리터럴과 같음) | 불필요 |
| 10 | 예보 조작화 | 예보는 판정 입력 아님; F-N2-7P는 #7과 결합해 결함(H17-2) | 불필요(수리만) |
| 11 | §7-3 PROFILE 4추정량 = Job-EA 결과 봉인 뒤 | 아니다(블라인드 강화) | 불필요 |
| 4 | greedy 비교 = 텍스트 **와** output_ids | 기술 라벨(abort 아님); 구현이 output_ids 부재 시 fail-open(H17-1) | 불필요(수리만) |
| 13 | "취소 OFF 경로 바이트 동일" 삭제 | 아니다(비교할 두 경로가 등록에 없음, 타당) | 불필요 |
| 1,3,5,6,8,12,14,15 | 구현 세부 | 아니다(6 RETRACT 경계 양쪽 귀속은 fail-closed) | 불필요 |

### 1-5. 낡은 문언·수치 대조
- 로더 기준: 계산기 58·축약기 85(=143)·하네스 36·Job 48·reach 35. `git ls-files` 43 = 7+16+2+17+1. 분류표 하네스 16 = `n2_plan.HARNESS_FILES` 16(같은 집합).
- **불일치(R17-1)**: §18-2-a 표 991행 `test_n2_probe_reduce.py | 규칙 시험(83, …)` → 85; 997행 `test_n2_harness.py | 하네스 시험(35)` → 36. §21-31 "141·35(하네스)·83(축약기)은 rev16 기록 문장과 §21 이력에만 남는다"는 이 두 곳이 **현행 정본 표**에 있으므로 **거짓**. 970행 "이 판(rev16)의 좁은 델타 감사"는 낡은 지시어.
- **공시 누락(R17-1 부속)**: rev17 작업 트리에서 `test_n2_job`은 **44/48**(실측) — `TestRegSource.test_sealed_file_loads`·`test_rule_seal_tamper_detected`, `TestCapacityScript.test_dryrun_in_symlinked_job_layout`·`test_fake_job_end_to_end`가 rev16 상수로 `ABORT_REG_SOURCE` 실패. 의도된 실패이고 §21-30이 "규칙 봉인 검사 불일치로 실패"는 적었으나, "Job 48" 옆에 "봉인 갱신 커밋 전 44/48, 실패 4건 = 위 목록"을 적어야 한다.
- 커밋 `1037655` 메시지 "reach 36"은 오기(로더 35) — 이력 문서라 수리 대상 아님(PREREG가 이미 보고).

### 1-6. 규칙층 반전 시험 표 (rev17이 만든 자유 표면)

| 자유 표면 | 민 범위 | 판정 변화 | 근거 |
|---|---|---|---|
| `CTX_UNREALIZED` 기준점(trim 앞↔뒤) | 경계 16 iteration 전체 = trim 앞 시작 (c−96, c−80] | 라벨 이름만 `WINDOW_SHORT`↔`CTX_UNREALIZED`; VALID 집합·점·D·dsens 라벨·예보 불변 | 둘 다 `PREDICATE_FAILURE`·VD, 분석기 참조 0 → **N2C-76** |
| G-10(VD 뒤 정지↔계속) | 두 끝 | 라벨 부호 불변; F-N2-7P HELD→FALSIFIED(예보, 판정 입력 아님) | `n2_analyze.py:364-365`가 `adm == n_units`만 보고 `stopped` 사유를 안 봄 → **H17-2 + N2C-77** |
| 셀 마감 기준점(구간 시작↔컨테이너 시작) | preflight 최대 600 s + cold 60 s | 입장 boot 수 불변(최악 코너 OBS44 직전 여유 ≈3,721 s > 660 s) | 표 14-2-P |
| 경계 초 RETRACT 귀속 | 한쪽↔양쪽 | fail-closed 방향, 반전 없음 | — |
| J28·J29 규약 | 0.0↔1.0, ≤↔< | 시험이 사살 → 등록 상태에서 자유 표면 아님 | §1-2 |

→ **반전 0건.** N1–N4 해당 없음, 死因 0.

## 2. 하네스층 감사 (Job-PROFILE)

### 2-1. 하네스 봉인 선행 조건

| 조건 | 상태 | 근거 | Job-EA 이월 |
|---|---|---|---|
| H15-3 ρ_off 조립·`AON_CONTAINMENT` | **미충족** | `n2_job`·`n2_capacity.sh`가 EA 거부(`ABORT_EA_NOT_IMPLEMENTED`) | **가능 — 단 등록 필요(R17-2)**: 현 §18-2는 단일 하네스 봉인 선행 조건으로 적음 |
| H15-4 블라인드 범위 | **충족** | `n2_realized_mix.estimators` PROFILE scope≠window → `BlindedScope`; 계약 #11; JH68 사살; `blind.coloc_statistics_computed=false`; 실행 범위 N2C-73 | — |
| H15-5 `ABORT_REG_SOURCE` + 변이 | **충족** | `load_registered`(파일 바이트 sha) + `verify_rule_seal`(7파일 재해시·digest·REGISTRATION 줄); JH01·JH73; rev17 트리에서 rc=3 실측(의도대로) | — |
| H15-5 warm-up 마크·C1 고정 | **충족** | `n2_marks.WARMUP_ARITY`, `test_n2_harness:105-116`이 `ARITY ∪ WARMUP_ARITY` 고정; 36/36 OK(실측) | — |
| H15-5 `--set harness` H01–H32 재실행 | **미충족** | 새 규칙 digest 위에서 재실행 필요(§18-2-c) | 불가(PROFILE 봉인 필수) |
| H14-1 하네스 구현 | **PROFILE 부분 충족** | 10개 파일 구현, EA stub | EA 부분 이월 가능(R17-2) |
| E14-2 개정 창 감사 + N2C-71 | 해당 없음(미사용) | Job-EA 개정 창에서만 발화 | 이월(본래 EA) |
| E14-3 OFF E-α 창 추출 변이 | **PROFILE 무관** | 규칙 쪽 경로(H15-2 시험·J군)는 일부 묶임; 하네스 EA 추출은 stub | 이월 |

### 2-2. 코드 집행 대조

| 항목 | 집행 | 근거 |
|---|---|---|
| `n2_plan` 규칙·하네스 봉인 검사 | 예 | `--verify-registration` rc=3, `--verify-harness-seal` UNSEALED rc=3; 셸은 측정 시 abort, TESTMODE에서만 note; 테스트 노브는 `PDMUX_COMMIT`(job_entry 필수) 있으면 거부 |
| 외부 digest | 예 | `EXTERNAL_DIGESTS_OK`(config `01675b2d…`, analyze `1be49511…`, profile `d7b6a807…` = §18-2-b) |
| 3층 하드캡 | 예 | (1) `n2_job` 입장 STOP(최악 코너, 첫 거부에서 정지, JH08) (2) 감시자 `SECTION_T0+36,700` s TERM → 90 s 뒤 KILL·pkill (3) `timeout -k 60 (CAP+1800)`; 마감 기준점 = preflight 뒤 셀 구간 시작(계약 #2) |
| `n2_window` 봉인 `cell_windows` import | 예 | `spec_from_file_location(REDUCER_FILE)` → `RED.cell_windows(...)`(`n2_window.py:40,261`) |
| `n2_realized_mix` λ0 양성 대조 | 예(로컬) | `--positive-control` → `OK []`(실측); Job 안에서는 원자료가 git 밖이라 **skip** → N2C-78 |
| `n2_analyze` D-민감도 | 예 | `dsens_label` = §9-6 의사코드(4/4, ln 1.03, 경계 ±LN3 UNRESOLVED); 같은 round만 짝(비짝 `dsens_x` ValueError); 108은 OFF16·OFF44 합산; 출력은 개수만; 코드 생성 문장에 "attention보다"·기전 문구 0 |
| 블라인드 | 예(코드 범위) | H15-4; 원자료 열람 자체는 막을 수 없음(N2C-73) |

### 2-3. 실엔진 전용 항목 — 첫 boot fail-closed 여부

| 항목 | 첫 boot fail-closed | 판단 |
|---|---|---|
| sticky ENABLED 로그(정확히 1회, index) + green read-out(6/6 그룹, smCount) | **예** — 매 boot `check_wiring`/`check_green` 실패 시 exit 6 | 충족 |
| capture bs ⊇ {1,8,32,48} | **예** — 매 boot exit 3 | 충족 |
| sticky ON 6그룹 decode 줄 `cuda graph: True` | **아니오** — 분석 단계 `cg_evidence`에서만(전 셀 `CG_UNPROVEN` fail-closed이나 **Job 종료 뒤**); §19-7·8 수용 위험으로 등록됨 | H17-3(권고) |
| `/generate` `output_ids` | **아니오** — 러너는 `text or output_ids`만 검사; 분석기 `x.get("output_ids") == y.get("output_ids")`는 양쪽 None이면 True → 텍스트만으로 `BYTE_IDENTICAL`. 계약 #4 위반 | **H17-1(필수)** |
| RETRACT 로그 `#retracted_reqs: N` | **불가**(부재가 정상 상태). 등록 엔진 트리(v0.5.10)에서 실제 출력 여부는 로컬 미검증(로컬 설치 다른 판 sglang에는 이 형식 없음 — 참고일 뿐). KV 수요 0.60M ≪ 2.72M이라 영향 예상 작음 | H17-4(권고) 또는 N2C-79 |
| 로그 TZ | 부분 — boot 기록 `tz_gmtoff_s`, 없으면 **분석 호스트** localtime 대체(`n2_window.py:113`); 어긋나도 CG 카나리아로 `CG_UNPROVEN` 셀 무효 | H17-5(권고) |

## 3. 결론

- **규칙 봉인 갱신: `GO-with-caveats`, 死因 0.** 판정 경로 불변(sha 일치, 판정 코드 diff 0), 새 시험이 J28·J29 사살(코드 대조), 반전 0건. 조건: R17-1…R17-3 + `@@CORE17@@` 채움을 봉인 갱신 커밋에 함께 넣는다. 그 diff는 감사 바이트 `0dbdfffb…`에서 **R17 항목 문장과 `@@CORE17@@` 자리만** 바뀌어야 하며 그 밖 바이트가 바뀌면 이 GO는 무효다.
- **하네스 봉인: 아직 불가(보류), 死因 0.** 필요: (1) 위 규칙 봉인 갱신 (2) R17-2(2단 봉인 등록) (3) H17-1·H17-2·H17-6·H17-7 (4) 그 뒤 하네스층 델타 재감사.

### 필수 수리 — 규칙 (R17-n)
- **R17-1** §18-2-a 991행 → "규칙 시험(85, …)", 997행 → "하네스 시험(36)", 970행 "이 판(rev16)" → "이 판(rev17)"; §21-31 "141·35·83은 이력에만" 문장을 정정 뒤 사실이 되도록 재작성; §21-30에 "봉인 갱신 커밋 전 `test_n2_job` 44/48, 의도된 실패 4건(TestRegSource 2·TestCapacityScript 2)" 공시.
- **R17-2** §18-2 하네스 봉인 행을 **2단**으로 등록: "하네스 봉인(PROFILE)" 선행 조건 = H15-4·H15-5(H01–H32 재실행 포함)·H14-1(PROFILE 부분)·H17-1/2/6/7; "하네스 봉인(EA)" 선행 조건 = H15-3·H14-1(EA 부분)·E14-2·E14-3. Job-PROFILE dry-run은 PROFILE 절, Job-EA dry-run은 두 절 모두 요구. (현 문언대로면 PROFILE 봉인이 자기 등록을 위반.)
- **R17-3** §18-2-b·§15에 등록: N3 양성 대조의 유효 입력은 **n3b_ 평탄 디렉터리(manifest `bec2cba5…`)뿐**, fetch 원본 `vessl/<job>/`은 유효 입력 아님; 규칙 봉인 장부는 **저장소 제자리 실행 skip 0**, `--set core`는 **CTRL skip 정확히 1**일 때만 유효.

### 필수·권고 수리 — 하네스 (H17-n)
- **H17-1(필수)** `n2_analyze.sticky_correctness`: 양쪽 모든 출력의 `output_ids`가 비어 있지 않은 리스트일 때만 `BYTE_IDENTICAL`, 아니면 `STICKY_CORRECTNESS_UNRESOLVED_D{d}`(why = output_ids absent). 러너는 첫 greedy boot에서 `output_ids` 부재를 기록하고 등록된 방식(라벨 또는 exit 9 중 하나, §18-2-d #4에 명시)으로 처리. 대응 JH 변이(None==None 경로) 추가.
- **H17-2(필수)** F-N2-7P: `JOB_LEDGER.stopped ∉ {None, "budget"}` 또는 `job_abort` 존재 → `UNRESOLVED`. 시험·변이 추가.
- **H17-3(권고·비용 보호)** boot마다 warm-up 구간 `Decode batch` 줄이 ≥ 1개이고 모두 `cuda graph: True`가 아니면 Job abort(`ABORT_CG_CANARY`, exit 3). warm-up 8요청 × 64토큰이라 실현 가능. 채택 시 §18-2-d #16으로 **이번 rev17 안에서** 등록(abort 규칙 추가는 계약).
- **H17-4(권고)** preflight에서 설치 엔진 트리에 `#retracted_reqs` 로그 형식 문자열 grep, 없으면 abort. 미채택 시 N2C-79.
- **H17-5(권고)** boot 기록에 `tz_gmtoff_s`가 없으면 분석 호스트 시간대로 대체하지 말고 그 boot 전 셀 `RECORDER_UNPROVEN`.
- **H17-6(필수, R17-2 구현)** `REGISTRATION_SHA256.txt` 하네스 절에 범위 태그(PROFILE/EA); `n2_capacity.sh`는 EA 절이 없으면 EA 거부(현행 stub 거부를 봉인 검사로 승격).
- **H17-7(필수)** 규칙 봉인 갱신 뒤 `n2_plan.RULE_SEAL_DIGEST`·`RULE_FILES` 갱신, 새 digest 위에서 시험 3종 + reach + `--set harness`(H01–H32) + JH 76 재실행, CTRL skip 수 기록.

### 사용자 결정 필요
- **계약 #7 G-10 Job 적용**(VD 중단 → Job 정지 ↔ 해당 boot만 종료 후 계속). 라벨 부호는 어느 쪽이든 불변 — ~10 h Job의 커버리지 대 비용 선택. 현재 값은 메인 세션이 "진행" 범위에서 고른 것.

### 신규 caveat (문자 그대로)
- **N2C-76**: "`CTX_UNREALIZED`와 `WINDOW_SHORT`의 구분은 trim 16 뒤 첫 단독 iteration의 ctx_p50 > c − 80이라는 코드 기준이며, trim 앞 시작 ctx가 (c − 96, c − 80]인 실패 창은 실제 실패가 종료 쪽(가장 앞선 요청의 O_P 도달)에서 났어도 `CTX_UNREALIZED`로 분류된다 — `CTX_UNREALIZED` 개수를 '시작 ctx 쪽 실패의 개수'로 인용하지 않는다."
- **N2C-77**: "G-10 Job 적용(VALUE_DEPENDENT 중단 뒤 Job 정지) 때문에 미입장 boot는 예산 정지와 다른 사유로 생길 수 있다 — `JOB_LEDGER.json`의 `stopped`가 `budget`이 아니면 F-N2-7P를 등록 예산 예보의 성패로 인용하지 않으며, 그 정지로 n < 4가 된 라벨을 '효과 없음'으로 쓰지 않는다."
- **N2C-78**: "Job 안 프리플라이트의 N3 θ 양성 대조와 λ0 실현 혼합 양성 대조는 원자료가 git 밖이라 Job 트리에서 skip된다 — `PREFLIGHT_TESTS.txt` 통과를 양성 대조 통과로 인용하지 않는다; 양성 대조의 근거는 로컬 봉인 장부 실행 기록(skip 0)뿐이다."
- **N2C-79**(H17-4 미채택 시만): "RETRACT 판정은 서버 로그 줄 `#retracted_reqs: N`의 존재에 의존하며 등록 엔진 트리에서 그 형식이 실제로 출력되는지는 실행 전 확인되지 않았다 — RETRACT 0건을 '재시도 없음'의 증거로 인용하지 않는다."
- N2C-75 미등재 **승인** — 전제("시험으로 묶이지 않았다")가 거짓이 됐고, 남는 인용 제한(데이터로 0건 보고)은 모든 경험적 주장에 공통이라 별도 caveat 불필요(교훈 276: 승계 아닌 재계산).

### 확인한 파일
- `PREREG_N2_3A_2026-10-07.md`(§6-2, §11-2, §11-4, §13, §18-2, §18-2-a…d, §21-30, §21-31)
- `n2_probe_reduce.py:248-333`, `test_n2_probe_reduce.py:14-15,271,503-557`, `n2_mutants.py:215-217,250,309-312`
- `n2_plan.py:1-100,153-280,600-690`, `n2_capacity.sh`, `n2_job.py:1-30,225-400`, `n2_window.py:105-260`, `n2_analyze.py:200-320,362-365`, `n2_realized_mix.py:20-48,121-144`, `n2_loadgen.py:39,587-609`, `n2_marks.py:27-32`, `test_n2_harness.py:105-116`, `test_n2_job.py:623`
- `REGISTRATION_SHA256.txt`(fcc0b9e), `VERDICT_n2_3a_rev16_seal_2026-10-09.md`, `VERDICT_n2_3a_rev14_delta_2026-10-09.md:292-302`
