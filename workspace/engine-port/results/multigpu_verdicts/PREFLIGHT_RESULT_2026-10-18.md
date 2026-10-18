# PREFLIGHT 결과 요약 (2026-10-18, experiment-runner) — 판정: PREFLIGHT_PASS

Job `pdmux-preflight-cl3-2aaf2aa-20261018014410` (slug `job-uprrdqr7kjr2`), 스펙 `resourcespec-a100cpu`, POST `2aaf2aae`, PRE `7b15214b`, reps=2, modules=<all>(K-3 충족).
Job 상태 `succeeded`(target rc=0). 컨테이너 경과 11094 s(약 3.08 h; suite timeout 7200 s 내). 이미지 `@sha256:1e0d335b…`.
회수: DONE(exit_code=0) + SHA256SUMS 검증 통과(50 OK + logs/meta 12 OK). fetch.sh는 `__pycache__` 2건을 충돌 검사에서 제외("existing ones kept"), 그 2건 pyc는 SHA 불일치(로컬 기존 pyc 보존). 소스 충돌 0.

## 시작 직후 점검
CPU-only 모드 줄, compat `LD_LIBRARY_PATH`(/usr/local/cuda-13.0/compat), venv_mode=image_python_prefix(/opt/conda), spec_decision `cpu_spec=1 gpu_seen=0 invalid=`, 판정기 SELFTEST PASS, l2 12/targets 32/declared 5 bad [], 구조화 러너 status_source = json x4. 모두 정상.

## K-1 — 7/7 일치
preflight_cl3.sh d9cb3755, preflight_judge.py f4b7e884, sigcmp.py 0677d07f, pf_unittest_runner.py 6a01bf65, l2_ids d10620d0, targets a79153f5, declared 23f60658 (`meta/provenance.txt`). r2 판정으로 인용 가능.
## K-4
`merge_tree_engine_digest` = `82a5fc5b5f587464e0df8037b64791900a013f61024d128e451818793526781c` → 결론 범위 문장 성립. (K-2는 이 런에서 미점검: 사전 확인 항목, main 세션 몫.)

## judge.json
- label `PREFLIGHT_PASS`. invalid [] . pre ran=989 failing=28, post ran=1487 failing=20. NEW 0, CHANGED 0, HOLE 0, l2_unverified [].
- L2 재현 9/9 (선언 3건 제외 9 id 기준). fixed_in_post 17. 선언 제외 5건(sha 23f60658, applied=True).
- CL-3 커버리지: A_import9 9/9, B_holb4 4/4, C_venv17 17/17, D_zamba2 1/1, E_rev12 1/1 (32/32).
- 직전 런의 NEW 2건(`JobEntrySubstrate`)은 이번에 없음. arm_tree_diff_unexplained / unmodelled_patch_changes 0바이트.

## 복사 (바이트 그대로, cmp 동일, 커밋 안 함)
`results/multigpu/PREFLIGHT_PASS.json` sha256 `18c42ebc35f1a551e3fa8115f7f71b6b2665ceac89e2c944a9a73f66db35cc63`
`results/multigpu/PREFLIGHT_PASS.json.sha256` sha256 `606d4a29523b99d9c772da07a5bb1a15c109a3531c86f238a5e17c5d17f591a7` (내용 = 위 json sha). 회수본 SHA256SUMS와 일치. judge.json sha `1a97865f…`.

## 커밋 전 게이트 참고 (로컬, 참고용)
- `mg_plan.py committed-clean . results/multigpu/PREFLIGHT_PASS.json` → `NOT_COMMITTED_CLEAN … not tracked at HEAD`(rc 1).
- `mg_plan.py preflight-gate . PASS.json <image>` → `ABORT_PREFLIGHT_NOT_PASSED (i) PREFLIGHT_PASS.json / .sha256 not tracked/clean`(커밋 전이라 예상된 결과; image 인자 일치 시 (ii)는 통과). 커밋 후 재실행 필요.
- 커밋은 하지 않았다.

## 승계 caveat (VERDICT_preflight_judge_r2 §6, 문자 그대로)
- **C-A1** "CPU 스펙 preflight PASS는 사전 선언 5건(`preflight_cl3_cpu_timing_declared.txt`, sha256 `23f60658…`)을 판정에서 뺀 결과다. 이 5건 — `test_e1_cap_vessl.TestBudgetEnforcer` 2건, `test_n3_capacity.TestBudgetEnforcer` 3건 — 은 미검증이며, PASS를 예산 강제·워치독·deadline 컷이 POST에서 동작한다는 근거로 인용하지 않는다."
- **C-A2** "판정기 r2는 런 `68f386b`를 본 뒤 정한 규칙이다. 반사실 사본에서 사전 선언은 UNVERIFIABLE을 PASS로 바꾸는 바로 그 자유도였고(n3 PRE 실패 2건 → `L2_UNVERIFIED` 구멍 2개 제거), 이는 사용자 결정 §8 (나)의 의도된 효과다. 'r2가 판정을 더 정확하게 만들어 PASS가 났다'고 쓰지 않는다."
- **C-A3** "'CPU timing'이라는 원인은 n3 2건(PRE에서 가짜 bench가 한 번도 뜨지 않음)에 대해서는 로그로 뒷받침되는 추론이고, e1 2건(가짜 run이 끝까지 돌고 컷되지 않음)에 대해서는 기전 미분해다. watchdog 1건은 4/4 결정적 실패가 아니다(PRE ok×2 — CPU에서 공허 통과로 추정, POST L2)."
- **C-A4** "CPU 스펙에서 L2 재현은 선언 3건을 뺀 9개 id 기준이다. 'L2 12건 재현'으로 쓰지 않는다."
- **C-A5** "CPU 스펙 PASS는 CPU 컨테이너(GPU 없음, `sgl_kernel` `sm100` 변종, CUDA 커널 실행 0)의 단위 스위트 결과다. GPU 실행 경로는 PASS 범위 밖이고, 최종 POST 엔진 digest가 `82a5fc5b…`일 때만 M2g PASS와 M3 Job 내부 gate가 덮는다. 'preflight PASS가 M3 실행 바이트를 보증한다'(MGH12-7)·'merge 회귀 0건'(C-4)이라고 쓰지 않는다."
- **C-A6** "구조화 상태 JSON과 `-v` 텍스트는 같은 `TestResult`에서 나온 기록이라 독립 증인이 아니다. 교차검증은 전사·파싱 오류를 잡을 뿐 unittest 수준의 공통 오류를 잡지 않는다. 러너의 컨테이너 첫 실행은 다음 런이다." (이 런이 컨테이너 첫 실행 이후 두 번째; 교차검증 INVALID 없음. "python -m unittest와 동치" 확인 문안은 여전히 금지.)
- **C-A7** "F5 마스킹은 arm 루트(`…/pf_work/pre|post`)를 지운다. 한 arm이 다른 arm의 경로를 읽는 교차 누출은 서명으로 구분되지 않는다."
- **C-A8** "반복 간 일치 검사는 실패 서명만 비교하고 상태(ok↔skip) 차이는 보지 않으며, CL-3 대상 coverage는 post rep1만 본다(r1부터의 동작, r2가 바꾸지 않음)."
- **C-B2** "F2 밀폐는 `test_mg_harness.py`의 5곳에만 적용됐다. `test_mg_rev12.py:1509`·`test_mg_rev12_seal.py:136`은 여전히 상속된 `HF_HOME`을 읽으며, 컨테이너에서는 실제 모델 캐시가 그 시험들의 dry-run 내용을 바꾼다(68f386b에서 결과는 통과)."
- **C-B3** "`test_mg_harness.JobScriptsDryRun`의 통과는 'A100 SXM 2장 미만 기계'에서만 의미가 있다. 이 시험은 하드웨어에 대해 밀폐되어 있지 않아 x2 이상 A100 노드에서는 노트가 사라져 실패할 수 있다. `HF_HOME` 고정은 이 축을 다루지 않는다."
- **C-B1**(봉인 재발행 항목): "rev12 하네스 봉인 재발행(2026-10-17)은 `tests/test_mg_harness.py` 1행과 `harness_seal_digest`만 바꿨다. …" — 이 런 결과와 직접 관계는 없고 참고만.
금지 문안 유지: "L2 12건 재현", "CPU preflight가 예산 강제·워치독을 확인했다", "preflight PASS가 M3 실행 바이트를 보증한다". 허용 범위의 "preflight gate 통과"는 PASS 파일 커밋 + `mg_plan.py preflight-gate`가 `PREFLIGHT_GATE_OK`를 낸 뒤에만.

## 아티팩트
`results/preflight_cl3/pdmux-preflight-cl3-2aaf2aa-20261018014410/`(judge.json, PREFLIGHT_LABEL, PREFLIGHT_PASS.json(.sha256), meta/, logs/),
`results/preflight_cl3/vessl/pdmux-preflight-cl3-2aaf2aa-20261018014410/`(meta, logs, DONE, SHA256SUMS, cost.txt, job_show_at_fetch.txt).
분석 대기: 추가 해석은 result-analyst/claims-auditor 몫.

이 Job이 채우는 COSTMODEL 칸: 미지정
