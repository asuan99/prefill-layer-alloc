# PREFLIGHT 결과 요약 (2026-10-17, experiment-runner) — 판정: PREFLIGHT_FAIL_NEW

Job `pdmux-preflight-cl3-9c3ea47-20261017113917` (slug `job-r7u67zy6nbly`), 스펙 `resourcespec-a100cpu`, POST `9c3ea47d`, PRE `7b15214b`.
Job 상태 `failed`는 라벨 FAIL_NEW의 rc=1 (target rc=1, 설계대로). 컨테이너 경과 11428 s(약 3.17 h; 예상 2.7 h 초과, suite timeout 7200 s 내).
회수: DONE(exit_code=1)+`sha256sum -c SHA256SUMS` 통과. fetch.sh는 `__pycache__/*.pyc` 2건(sigcmp, preflight_judge; 로컬 기존 pyc와 바이트 불일치)을
CONFLICT로 보고하고 exit 4 — 소스 충돌 아님. 나머지는 `cp -an`(비덮어쓰기)로 수동 병합, 로컬 pyc 보존. 병합 충돌 0 기준의 엄밀한 "완료"는 이 pyc 예외를 달고 있다.

## 시작 직후 점검
CPU 모드 줄, compat `LD_LIBRARY_PATH`(/usr/local/cuda-13.0/compat), venv_mode=image_python_prefix(/opt/conda), spec_decision `cpu_spec=1 gpu_seen=0 invalid=`,
구조화 러너(`status_source` = json x4, `pf_status/v1`) 모두 확인. 교차검증 INVALID 없음(`invalid: []`, notes=[]).

## K-1 (provenance.txt 대 감사 sha) — 7/7 일치
preflight_cl3.sh d9cb3755, preflight_judge.py f4b7e884, sigcmp.py 0677d07f, pf_unittest_runner.py 6a01bf65, l2_ids d10620d0, targets a79153f5, declared 23f60658.
따라서 이 런의 라벨은 "r2 판정"으로 인용 가능.

## judge.json
- label PREFLIGHT_FAIL_NEW. pre ran=989 failing=28, post ran=1487 failing=22 (rep1=rep2 동일 요약).
- NEW 2: `test_multigpu_launcher.JobEntrySubstrate.test_all_gpus_recorded`, `...test_one_gpu_keeps_old_fields`. CHANGED 0. HOLE 0.
- L2 재현 9/9 (선언 3건 제외 9 id 기준). fixed_in_post 17. 선언 제외 5건(sha 23f60658, applied=True).
- CL-3 32개 커버리지: A_import9 9/9, B_holb4 4/4, C_venv17 17/17, D_zamba2 1/1, E_rev12 1/1.
- 판정기 selftest PASS.

## NEW 2건 원인 (진단, 수리는 engine-porter)
KeyError 'gpu_count' / TypeError 'NoneType' (r["gpu"]). 시험의 `_heredoc("substrate.json")`이 `job_entry.sh`에서 "substrate.json"을 포함하는 **첫** heredoc을
고르는데, POST가 추가한 CPU-only 분기 heredoc(line 160, `"gpu": None`, `substrate.json` 기록)이 GPU 스탬프 heredoc(line 178)보다 앞에 있어 그것이 선택된다.
환경(CPU 컨테이너) 의존이 아니라 CPU-only 분기 추가와 시험 마커의 비양립. 재제출은 하지 않았다.

## 복사
라벨이 PREFLIGHT_PASS가 아니므로 `results/multigpu/`로 PREFLIGHT_PASS.json 복사 안 함 (해당 경로에 기존 PREFLIGHT_PASS* 없음). 커밋 없음.
`PREFLIGHT_PASS.json`(내용 verdict=PREFLIGHT_FAIL_NEW) 파일명은 FAIL에서도 생성되므로 파일 존재가 통과를 뜻하지 않는다.

## 아티팩트
`results/preflight_cl3/pdmux-preflight-cl3-9c3ea47-20261017113917/`(judge.json, PREFLIGHT_LABEL, meta/, logs/ *.status.json),
`results/preflight_cl3/vessl/pdmux-preflight-cl3-9c3ea47-20261017113917/`(meta, logs, DONE, SHA256SUMS, job_show_at_fetch.txt; cost.txt는 수동 병합이라 미생성).
분석 대기: 추가 해석은 result-analyst/claims-auditor 몫.

## 승계 caveat (VERDICT_preflight_judge_r2 §6, 문자 그대로; 라벨이 FAIL이라 PASS 인용용은 아니나 이 런에 적용되는 범위 병기)
- **C-A1** "CPU 스펙 preflight PASS는 사전 선언 5건(`preflight_cl3_cpu_timing_declared.txt`, sha256 `23f60658…`)을 판정에서 뺀 결과다. 이 5건 — `test_e1_cap_vessl.TestBudgetEnforcer` 2건, `test_n3_capacity.TestBudgetEnforcer` 3건 — 은 미검증이며, PASS를 예산 강제·워치독·deadline 컷이 POST에서 동작한다는 근거로 인용하지 않는다."
- **C-A2** "판정기 r2는 런 `68f386b`를 본 뒤 정한 규칙이다. 반사실 사본에서 사전 선언은 UNVERIFIABLE을 PASS로 바꾸는 바로 그 자유도였고(n3 PRE 실패 2건 → `L2_UNVERIFIED` 구멍 2개 제거), 이는 사용자 결정 §8 (나)의 의도된 효과다. 'r2가 판정을 더 정확하게 만들어 PASS가 났다'고 쓰지 않는다."
- **C-A3** "'CPU timing'이라는 원인은 n3 2건(PRE에서 가짜 bench가 한 번도 뜨지 않음)에 대해서는 로그로 뒷받침되는 추론이고, e1 2건(가짜 run이 끝까지 돌고 컷되지 않음)에 대해서는 기전 미분해다. watchdog 1건은 4/4 결정적 실패가 아니다(PRE ok×2 — CPU에서 공허 통과로 추정, POST L2)."
- **C-A4** "CPU 스펙에서 L2 재현은 선언 3건을 뺀 9개 id 기준이다. 'L2 12건 재현'으로 쓰지 않는다."
- **C-A5** "CPU 스펙 PASS는 CPU 컨테이너(GPU 없음, `sgl_kernel` `sm100` 변종, CUDA 커널 실행 0)의 단위 스위트 결과다. GPU 실행 경로는 PASS 범위 밖이고, 최종 POST 엔진 digest가 `82a5fc5b…`일 때만 M2g PASS와 M3 Job 내부 gate가 덮는다. 'preflight PASS가 M3 실행 바이트를 보증한다'(MGH12-7)·'merge 회귀 0건'(C-4)이라고 쓰지 않는다."
- **C-A6** "구조화 상태 JSON과 `-v` 텍스트는 같은 `TestResult`에서 나온 기록이라 독립 증인이 아니다. 교차검증은 전사·파싱 오류를 잡을 뿐 unittest 수준의 공통 오류를 잡지 않는다. 러너의 컨테이너 첫 실행은 다음 런이다." (이 런이 그 컨테이너 첫 실행: 구조화 러너 정상 동작, INVALID 없음. 단 "python -m unittest와 동치" 확인 문안은 여전히 금지.)
- **C-A7** "F5 마스킹은 arm 루트(`…/pf_work/pre|post`)를 지운다. 한 arm이 다른 arm의 경로를 읽는 교차 누출은 서명으로 구분되지 않는다."
- **C-A8** "반복 간 일치 검사는 실패 서명만 비교하고 상태(ok↔skip) 차이는 보지 않으며, CL-3 대상 coverage는 post rep1만 본다(r1부터의 동작, r2가 바꾸지 않음)."
- **C-B2** "F2 밀폐는 `test_mg_harness.py`의 5곳에만 적용됐다. `test_mg_rev12.py:1509`·`test_mg_rev12_seal.py:136`은 여전히 상속된 `HF_HOME`을 읽으며, 컨테이너에서는 실제 모델 캐시가 그 시험들의 dry-run 내용을 바꾼다(68f386b에서 결과는 통과)."
- **C-B3** "`test_mg_harness.JobScriptsDryRun`의 통과는 'A100 SXM 2장 미만 기계'에서만 의미가 있다. 이 시험은 하드웨어에 대해 밀폐되어 있지 않아 x2 이상 A100 노드에서는 노트가 사라져 실패할 수 있다. `HF_HOME` 고정은 이 축을 다루지 않는다."
(C-B1은 봉인 재발행 항목이라 이 런 결과와 무관, 생략.) 금지 문안 중 "preflight gate 통과"/"M3 제출 가능"은 이 런에 대해서도 쓰지 않는다 — 라벨이 FAIL_NEW.

이 Job이 채우는 COSTMODEL 칸: 미지정
