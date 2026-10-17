# VERDICT: preflight 판정기 r2(`940c1d7a`)와 rev12 하네스 봉인 좁은 재발행 (2026-10-17, claims-auditor, 1회차)

> claims-auditor(읽기 전용) 최종 보고를 main 세션이 저장(2026-10-17). 규칙층 감사(사전등록 설계 심사). 저장소 무편집, 재실행은 scratchpad 사본(`…/scratchpad/audit_ca1/`)만, fixture 로그 sha 감사 전후 동일(`FIXTURE_UNCHANGED`).
> 원칙: 런 `pdmux-preflight-cl3-68f386b-20261016114954`의 라벨은 기록대로 `PREFLIGHT_FAIL_NEW`. 이 판정서는 그 런의 비변이 로그에 r2 라벨을 산출하지 않았다. 반전 시험은 아래 정의한 **반사실 사본**에서만.
> 감사 횟수: A·B 모두 1회차(상한 2). 2회차는 이 판정 이후 diff만.

## 0. 판정 요약

| 대상 | 등급 | 死因 | 필수 수리 | 제출 전 조건 |
|---|---|---|---|---|
| A. 판정기 r2 `940c1d7a`(F3·F5·F6·사전 선언 5건·결론 범위) | **GO-with-caveats** | 0 | 코드 수리 없음 | K-1–K-4 |
| B. rev12 하네스 봉인 좁은 재발행(`test_mg_harness.py` 1행 + `harness_seal_digest`) | **GO-with-caveats** | 0 | **R-B1, R-B2**(적용 전) | 같음 |

`GO-with-caveats`는 실행 승인이며 preflight gate를 닫았다는 뜻이 **아니다**.

## 1. 재실행 (사본, 로컬 CPU, python 3.13.9)

- `preflight_judge_selftest.py` **53/53 ok**. `preflight_judge_fixture_selftest.py`(68f386b 로그, 읽기 전용) **39/39 ok**. `preflight_judge_mutants.py` CONTROL_OK, **21/21 KILLED**(fixture 단독 J01·J02·J03·J05·J06·J07·J15 7종 — 커밋 메시지와 일치).
- 구조화 러너 동치(5모듈 44 id: engine_digest·mg_rev12_seal·holb_probe[import 오류]·mg_symlink_layout·loadgen_outcome): `python -m unittest -v`와 `pf_unittest_runner.py` 텍스트 동일(시간 제외), 엄격 텍스트 상태 = JSON 44/44, `load_rep` problems 0.
- 패치 `F2_F4_candidate.patch`(sha `7192b2af…`)를 `git archive HEAD` 사본에 적용 → `test_mg_harness.py` `f97cc945…`(예정 일치). `seal_new.txt` 재계산: harness 59행 digest `cd00fe05…`, rule 2행 `7e4e5e19…`(불변), 기존 봉인과 diff 정확히 2줄(65·84행).
- F2 양방향(fake 캐시 `refs/main = dc0661c8…`): 패치 전 HEAD — unset OK / fake **FAIL**(`'ABORT_MODEL_REVISION' not found`, 컨테이너와 같은 메시지). 패치 후 + 새 봉인 — unset·fake 모두 통과. 패치 후 + 새 봉인으로 `test_mg_harness`·`test_mg_rev12_seal`·`test_holb_probe` 185건 실행, 실패 2는 사본 환경(`.git` 없음 1, `sglang` 없음 1). 패치 + **옛** 봉인이면 `test_mg_rev12_seal` 2건 실패(`bytes differ test_mg_harness.py`, 등록 목록 불일치) — 의도된 fail-closed.

## 2. 반전 시험 (대상 A)

**반사실 사본**: 68f386b `logs/{pre,post}.rep{1,2}.log` 사본에서 post 두 반복의 `test_mg_harness.JobScriptsDryRun.test_m3_dry_run_every_job`만 `ok`로 바꾸고 실패 블록 삭제("F2가 고쳐졌다" 가정), 그 밖은 바이트 그대로. 상태 원천 = 텍스트(엄격 파서, fixture에 JSON 없음). 판정기 = 감사한 r2 바이트 또는 표에 적은 한 줄 변이. **68f386b 런의 라벨이 아니다.**

| 자유 표면 | 민 범위 | 판정 변화 | 근거 |
|---|---|---|---|
| 사전 선언 목록 | 0건 / 1건(overrunning) / e1 2건만 | **UNVERIFIABLE** | holes 2/1/2, 전부 `L2_UNVERIFIED`(n3 overrunning·control_both, PRE `FAIL`) |
| 〃 | n3 PRE 실패 2건 / n3 3건 / watchdog 뺀 4건 / **등록 5건** | **PASS** | holes 0, NEW 0, CHANGED 0; L2 10/10, 9/9, 10/10, 9/9 |
| 〃 해석 | — | 판정은 **사용자 결정 §8 (가)/(나) 경계**에서만 뒤집힘. 남은 재량(watchdog·e1 포함 여부)은 이 자료에서 라벨 불변 → **死因 아님**(자유도는 사용자 결정이 정했고 목록은 sha 고정) → C-A2 | — |
| 스펙 판정(`PDMUX_CPU_ONLY`/`PDMUX_SPEC` × GPU 가시성) | 4조합 | CPU 플래그 + GPU 보임 → INVALID. 플래그 없음 → 선언 미적용 → UNVERIFIABLE(holes 2). PASS로 기울일 수 없음(`launch.sh:60`이 `PDMUX_SPEC` 주입) | selftest S8 결정표 8행 |
| F6(L2는 PRE 정확히 `ok`일 때만) | 등록 / 끔 | 끄면 선언 없이도 **PASS**(holes 0, L2 10/12) — 등록이 더 엄격, non-PASS 쪽으로만 움직임. 死因 아님 | 변이 J01 |
| F5 마스킹 | 없음 / 등록(arm 루트만) / 과잉(`/data/runs/\S+`) | 없음 → **FAIL_NEW**(CHANGED 8), 등록 = 과잉 → PASS. 8건은 같은 아카이브 파일 FNF(분석서 §3-1), 과잉과의 차이는 다른 파일 경로에서만 생기고 잡힘(합성 S5b, J03 KILLED). 死因 아님, 교차 누출은 C-A7 | — |
| F3 파싱 | 엄격 / 첫 토큰 / JSON | fixture에서 엄격 = 관대(같은 라벨), 합성 S3이 가름(J04 KILLED). Job은 JSON + `--require-structured`. 엄격이 보수적. 死因 아님 | — |
| (diff 밖) `PDMUX_PF_REPS` | 1 / 2 | 1이면 반복 간 flaky 대조 소멸(r1부터) → K-3 | `preflight_judge.py:162` |

새 자유도 미사용: 등록 판정기 바이트와 한 줄 변이(mutant ledger J01–J04와 같은 앵커)만, 반사실 변경은 F2 가정 하나.

## 3. 질문별 답

- **Q1 `hermetic_env` 범위**: 의도된 상속 단절 없음. `mg_job_common.sh:51`이 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`을 스스로 export(하위 프로세스에서 재설정). `mg_runner.sh`·`engine_bench_runner.sh` HF 변수 참조 0(grep). 패치 전 로컬 기본값(`$DATA/hf`)·패치 후(`td/hf_empty`) 모두 존재하지 않는 경로라 로컬 동작 동일. 단정은 부분문자열(`ABORT_MODEL_REVISION`)뿐. 실행 근거: 패치 후 unset·fake 양쪽 `test_mg_harness` 전체 통과(환경 실패 1 제외). **한계(C-B2)**: 밀폐는 `test_mg_harness` 5곳에만 — 같은 비밀폐 필터가 `test_mg_rev12.py:1509`(M3JobScript), `test_mg_rev12_seal.py:136`에 남음; 컨테이너에서 실제 캐시 모델 config를 읽음(mamba dtype이 placeholder 대신 실제 값), 68f386b에서 통과라 판정 무관이나 기계 의존; `test_mg_rev12.py`는 봉인 행이라 이번 범위에서 미수정이 맞음. 덤으로 사라지는 것: 컨테이너 런은 "refs/main이 등록 revision과 일치하면 노트 없음" 분기를 **우연히** 덮었고 그 분기를 의도적으로 덮는 시험은 없음(grep 0; 실제 M3 Job B0가 실행). 권고(비필수): fake 캐시로 노트 부재를 단정하는 짝 시험.
- **Q2 하드웨어 비밀폐를 가리는가**: 가리지 않음(독립 축). `ABORT_SUBSTRATE`는 `nvidia-smi`·`score-b0`·NVML(`mg_job_common.sh:113-149`), `HF_HOME` 무관. 로컬 실측(RTX 5060 Ti 1장)에서도 "gpu_count 1 != 2"로 노트 발생 → **A100 SXM 400W 2장 이상** 기계에서는 J1·J2 노트가 사라져 실패. 현 계획에서 `test_mg_harness`를 GPU x2+에서 도는 경로 0(`m3_job.sh`·`mg_job_common.sh`는 unittest 미호출, `cpu_probe.sh`는 이 모듈 미실행) → C-B3.
- **Q3 분모 12 → 9**: 허용. 자유 표면이나 반전(§2)에서 死因 없음. "L2 never reproduced"는 "≥1건 재현" 검사라 분모 변화로 느슨해지지 않음(반사실 9/9). n3 3건은 POST에서 **어차피 검증 불가**(L2 drift abort가 예산 강제 단계보다 먼저, 분석서 §5-2 1) → 잃는 것은 PRE 정상성 비트뿐; `test_n3_capacity.py`·`n3_capacity.sh`는 PRE `7b15214b` → POST `e43749c9` 코드 변경 0(CLOSURE 문서 1줄만). watchdog 1건 [코드 + 로그 추론]: 단정은 `rc==0`·`wall<50`뿐, 같은 CPU PRE에서 overrunning `FAKE_BENCH`가 빈 문자열(첫 unit 전 deadline 소진) → watchdog PRE `ok`는 **공허 통과**로 보는 게 합리적, F6 근거로 부적합하므로 선언 포함이 오히려 정당. 단 선언 머리말 "4/4 deterministic repetitions"는 이 1건에 **사실 아님**(PRE ok×2) → C-A3. e1 2건: 양 arm 같은 서명 실패(4/4), 선언 없어도 라벨 영향 없었음; 잃는 것은 "POST에서만 생기는 e1 회귀" 검출; `test_e1_cap_vessl.py`·`results/e1_cap_vessl/` PRE→POST 변경 0 → 현재 반전 없음, 최종 POST가 이 경로를 바꾸면 전제 붕괴 → K-2.
- **Q4 봉인 diff**: 기계적 의미 보존 — rule digest `7e4e5e19…` 불변(재계산 일치), 하네스 실행 바이트(`m3_job.sh`·`mg_*.py`·`mg_job_common.sh` 등) 행 불변(diff 2줄), 바뀐 행은 **시험 파일** 1행, 옛 digest `323c2081…`는 봉인 파일 자신 외 저장소 어디서도 인용 없음(git grep). 빠진 것 둘: **(R-B1)** 계획 바이트 `seal_new.txt`(sha `503f3de8…`)에 **재발행 기록 없음** — 머리말 "basis: harness-layer audit x2" 그대로라 감사받지 않은 `test_mg_harness.py` 바이트가 그 근거 아래 섬. **(R-B2)** 순서 — `test_mg_rev12_seal.py`는 PRE에 없음; POST에 패치만 + 봉인 옛것이면 POST에서만 실패해 **NEW → FAIL_NEW**(fail-closed지만 2.7 h 낭비).
- **Q5a 러너 동치**: 코드상 동치 — `sys.path[0]` 둘 다 cwd, discover top-level 기본 같음, `warnings='default'` 같음, 같은 `TextTestRunner(verbosity=2)`라 py3.14 색상 여부 같음(파일 리다이렉트, tty 아님), 시험 본문이 러너 `sys.argv`·`__main__`을 읽는 곳 0(grep의 `sys.argv`는 하위 스크립트 문자열 안). 실측 로컬 44 id 동일(커밋 보고 1375 동일, 전체 재실행 안 함). 컨테이너 첫 실행은 다음 런 — 어긋나면 ran·상태·실패 블록 교차검증이 INVALID, PASS 쪽 실패 경로 없음 → C-A6.
- **Q5b JSON 권위의 약점**: fail→ok(실패 블록 ≠ JSON 실패 → INVALID), skip→ok(raw skip ≠ JSON → INVALID), 누락(텍스트 상태 id가 JSON에 없음 → INVALID), 개수(ran 불일치 → INVALID), 비정상 종료(JSON 없음 → INVALID). 남는 경로: 텍스트 상태 불가독 + 실패 블록 없음인 시험을 JSON이 ok라 하는 경우 = 러너 버그, 변이 J18–J20이 덮음. 근본 한계: 두 기록은 **같은 `TestResult`**에서 나와 독립 증인 아님; unittest 공통 모드 오류(시험이 자기 실패를 삼킴 등)는 `python -m unittest`에서도 같음 → C-A6.
- **Q5c schema v2**: gate 정확성으로는 충분(gate는 verdict·digest·image·post_commit·gate 경로만, `mg_plan.py:689-735`). 범위 추적으로는 **불충분** — gate는 선언 적용 여부·선언 sha가 감사받은 `23f60658…`인지 모름; `merge_trial_tools/`는 7개 gate 경로 밖이라 감사 뒤 도구·선언을 바꿔도 (iii) 미발화 → K-1(provenance 도구 sha 대조)로 닫음. 코드 pin을 요구하지 않는 이유: pin을 넣을 스크립트 자체가 pin되지 않아 순환; 결정적 근거는 Job이 남기는 `meta/provenance.txt` sha.

## 4. confound 카탈로그 대조

| 번호 | 해당 |
|---|---|
| 7 (micro·tenancy 아티팩트) | **해당** — 선언 5건은 3.5 vCPU·스로틀 17.6 % 기판의 시작 비용 문제, 서빙·정확성 신호 아님 → "미검증" 공시가 맞음 |
| 2 변형 | **기전 귀속 위험** — e1 2건 "CPU timing"은 기전 미분해(평가서 §7-7), 가짜 run이 끝까지 돌고 컷 미발생, 다른 기전(고아 프로세스가 파이프를 붙잡음 등) 가능 → C-A3 |
| 10 (변수 동시 변경) | **해당** — r2는 F3·F5·F6·선언을 한 번에 바꿈, §2가 분리; F6과 선언은 같은 n3 id에 반대 방향, 순효과 PASS → C-A2 |
| 11 (기판 혼합) | **해당(범위)** — CPU 컨테이너 PASS를 GPU 경로·cc 8.0 변종 증거로 쓰지 않음 → C-A5 |
| 6 유사 | 개정 판정기를 68f386b에 소급 안 함, 반사실 수치는 예보이지 라벨 아님 |
| 1·3·4·5·8·9 | 해당 없음(성능·정책 주장 0) |

## 5. 필수 수리 (B) · 제출 전 조건 (A·B)

- **R-B1 (필수, 봉인 적용 전)**: `REGISTRATION_rev12_SEAL.txt` 재발행본에 주석 행(`#`, 봉인 바이트 아님) 추가 — 재발행일 2026-10-17, 바뀐 행 `tests/test_mg_harness.py` `8392b4ca…` → `f97cc945…`, 이전 `harness_seal_digest 323c2081…`, rule digest 불변, 사유 F2(HF 밀폐 5곳; F4는 봉인 밖 `test_holb_probe.py`), 패치 sha `7192b2af…`, 하네스 실행 바이트 불변, 이 판정서 경로. 머리말 basis 줄에 이 판정서 추가. 주석만이라 digest 불변(`parse`가 `#` 행 무시).
- **R-B2 (필수)**: F2/F4 패치와 봉인 재발행을 **같은 커밋**에(최소한 재발행 커밋 ≤ 최종 POST). 근거: `test_mg_rev12_seal`은 PRE `7b15214b`에 없음(`git cat-file` 확인) → 패치만 든 POST면 NEW → FAIL_NEW.
- **K-1**: 다음 Job `meta/provenance.txt` 도구 sha가 감사 바이트와 같아야 함. 하나라도 다르면 그 런 라벨을 "r2 판정"으로 인용하지 않고 2회차(diff만)로.

| 파일 | 감사 sha256 |
|---|---|
| `preflight_cl3.sh` | `d9cb3755…` |
| `preflight_judge.py` | `f4b7e884…` |
| `sigcmp.py` | `0677d07f…` |
| `pf_unittest_runner.py` | `6a01bf65…` |
| `preflight_cl3_l2_ids.txt` | `d10620d0…` |
| `preflight_cl3_targets.txt` | `a79153f5…` |
| `preflight_cl3_cpu_timing_declared.txt` | `23f60658…` |

- **K-2**: 아래 출력에 **코드 파일 변경 0**(n3 CLOSURE `.md` 제외). 아니면 선언 전제 붕괴 → 사용자 재결정.
  `git diff --stat 7b15214b <최종 POST> -- workspace/engine-port/tests/test_e1_cap_vessl.py workspace/engine-port/tests/test_n3_capacity.py workspace/engine-port/results/e1_cap_vessl workspace/engine-port/results/n3_capacity`
- **K-3**: `launch.sh` dry-run에서 `PDMUX_PF_REPS` 미지정 또는 ≥2, `PDMUX_PF_MODULES` 미지정, 스펙 `resourcespec-a100cpu`, `PDMUX_PF_POST` = R-B2 커밋 이후 최종 POST.
- **K-4**: 최종 POST 엔진 digest가 `82a5fc5b…`일 때만 결론 범위 문장(GPU 경로는 M2g·M3 내부 gate가 덮음) 성립. 아니면 rev12 §19-6대로 사용자 재질의.
- 권고(비필수): `PREFLIGHT_LABEL`은 stderr를 합쳐 써서 첫 줄이 pynvml 경고였음(68f386b) — `head -1`로 읽지 말 것. 반복 간 상태(ok↔skip) 대조와 rep2 target coverage는 r1부터 비어 있음(C-A8).

## 6. 승계 caveat (문자 그대로)

- **C-A1** "CPU 스펙 preflight PASS는 사전 선언 5건(`preflight_cl3_cpu_timing_declared.txt`, sha256 `23f60658…`)을 판정에서 뺀 결과다. 이 5건 — `test_e1_cap_vessl.TestBudgetEnforcer` 2건, `test_n3_capacity.TestBudgetEnforcer` 3건 — 은 미검증이며, PASS를 예산 강제·워치독·deadline 컷이 POST에서 동작한다는 근거로 인용하지 않는다."
- **C-A2** "판정기 r2는 런 `68f386b`를 본 뒤 정한 규칙이다. 반사실 사본에서 사전 선언은 UNVERIFIABLE을 PASS로 바꾸는 바로 그 자유도였고(n3 PRE 실패 2건 → `L2_UNVERIFIED` 구멍 2개 제거), 이는 사용자 결정 §8 (나)의 의도된 효과다. 'r2가 판정을 더 정확하게 만들어 PASS가 났다'고 쓰지 않는다."
- **C-A3** "'CPU timing'이라는 원인은 n3 2건(PRE에서 가짜 bench가 한 번도 뜨지 않음)에 대해서는 로그로 뒷받침되는 추론이고, e1 2건(가짜 run이 끝까지 돌고 컷되지 않음)에 대해서는 기전 미분해다. watchdog 1건은 4/4 결정적 실패가 아니다(PRE ok×2 — CPU에서 공허 통과로 추정, POST L2)."
- **C-A4** "CPU 스펙에서 L2 재현은 선언 3건을 뺀 9개 id 기준이다. 'L2 12건 재현'으로 쓰지 않는다."
- **C-A5** "CPU 스펙 PASS는 CPU 컨테이너(GPU 없음, `sgl_kernel` `sm100` 변종, CUDA 커널 실행 0)의 단위 스위트 결과다. GPU 실행 경로는 PASS 범위 밖이고, 최종 POST 엔진 digest가 `82a5fc5b…`일 때만 M2g PASS와 M3 Job 내부 gate가 덮는다. 'preflight PASS가 M3 실행 바이트를 보증한다'(MGH12-7)·'merge 회귀 0건'(C-4)이라고 쓰지 않는다."
- **C-A6** "구조화 상태 JSON과 `-v` 텍스트는 같은 `TestResult`에서 나온 기록이라 독립 증인이 아니다. 교차검증은 전사·파싱 오류를 잡을 뿐 unittest 수준의 공통 오류를 잡지 않는다. 러너의 컨테이너 첫 실행은 다음 런이다."
- **C-A7** "F5 마스킹은 arm 루트(`…/pf_work/pre|post`)를 지운다. 한 arm이 다른 arm의 경로를 읽는 교차 누출은 서명으로 구분되지 않는다."
- **C-A8** "반복 간 일치 검사는 실패 서명만 비교하고 상태(ok↔skip) 차이는 보지 않으며, CL-3 대상 coverage는 post rep1만 본다(r1부터의 동작, r2가 바꾸지 않음)."
- **C-B1** "rev12 하네스 봉인 재발행(2026-10-17)은 `tests/test_mg_harness.py` 1행과 `harness_seal_digest`만 바꿨다. 하네스 실행 바이트와 규칙 digest `7e4e5e19…`는 불변이다. 바뀐 시험 바이트는 하네스층 감사 2회가 아니라 이 판정서(claims-auditor 1회차)가 다뤘다."
- **C-B2** "F2 밀폐는 `test_mg_harness.py`의 5곳에만 적용됐다. `test_mg_rev12.py:1509`·`test_mg_rev12_seal.py:136`은 여전히 상속된 `HF_HOME`을 읽으며, 컨테이너에서는 실제 모델 캐시가 그 시험들의 dry-run 내용을 바꾼다(68f386b에서 결과는 통과)."
- **C-B3** "`test_mg_harness.JobScriptsDryRun`의 통과는 'A100 SXM 2장 미만 기계'에서만 의미가 있다. 이 시험은 하드웨어에 대해 밀폐되어 있지 않아 x2 이상 A100 노드에서는 노트가 사라져 실패할 수 있다. `HF_HOME` 고정은 이 축을 다루지 않는다."

## 7. 허용 / 금지 문안

허용: "판정기 r2(`940c1d7a`)는 claims-auditor 1회차 규칙층 감사에서 死因 0, `GO-with-caveats`를 받았다. selftest 53/53, fixture selftest 39/39, 변이 21/21 KILLED가 감사자 사본에서 재현됐다." · "반전 시험에서 F3·F5·F6·스펙 판정의 남은 자유도는 판정을 PASS 쪽으로 뒤집지 못했다. 사전 선언은 사용자 결정 §8 (나)의 경계에서만 판정을 바꿨다." · "F2 패치는 fake 모델 캐시(`refs/main` = 등록 revision)에서 실패하던 `test_m3_dry_run_every_job`을 통과시키고, 캐시가 없을 때도 통과한다(양방향, 로컬 CPU)." · "봉인 재발행 계획은 규칙 digest를 바꾸지 않고, diff는 2줄이며, digest를 재계산해 `cd00fe05…`와 일치했다."

금지: "68f386b 런은 r2 판정기로 보면 PASS다"(또는 같은 뜻) · "CPU preflight가 예산 강제·워치독이 POST에서 동작함을 확인했다" · "CPU timing이 e1 실패의 원인으로 확인됐다" · "L2 12건 재현" · "구조화 러너가 컨테이너에서 `python -m unittest`와 동치임을 확인했다"(다음 런 전) · "F2로 `test_mg_harness`가 기계와 무관해졌다" · "rev12 하네스 봉인이 하네스층 감사 2회로 전부 검증됐다" · "preflight gate 통과"/"M3 제출 가능"(preflight 런 전).

## 8. 다음 단계

A·B 1회차. 2회차는 R-B1·R-B2 적용 바이트, K-1 sha 변경 시, 다음 런 뒤 라벨 해석에 대해 diff만. 2회차에도 死因이 남으면 사용자 결정.
