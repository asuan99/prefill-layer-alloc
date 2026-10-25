# 하네스층 감사 r1 — costterms 하네스 대 PREREG rev3.7(미봉인), 커밋 `5b394f81`, 2026-10-26

> claims-auditor(읽기 전용) 판정을 main 세션이 옮겨 적었다. 같은 대상 최대 2회 중 1회차. r2 범위 = MAJOR-1·2, m1·m2·m4·m5·m6 수리 diff와 신규 변이.

## 전체 판정
**smoke 진행 가능. 봉인(S1)·본 측정 제출은 수리 뒤.** 死因 0, MAJOR 2, MINOR 13.
직접 확인: 시험 60/60 OK(232 s), 실행 단계 변이 15개 표본 재실행 — 전부 crash가 아니라 oracle 메시지로 검출, ct\_timemodel 값·하드캡 세 곳 일치, J-A 실행 셀 106개(context 상향으로 바뀐 셀은 (8, 64k)뿐), 엔진 소스 대조(`prefill_sms` = 목표값, span 종료 이벤트 lockstep에서도 호출, SGLang은 입력 + max\_new ≥ context\_len 거절).

## MAJOR
- **MAJOR-1** `ct_fit.py:266-279` `client_secondary_table`: 키 `(model, mode, bs, c)`에 창마다 덮어씀 → 마지막 창만 남고, J-C TP2와 cap 48 앵커(`real_a4|D44`), J-A1 cap 불변성 boot와 D44 cap 128 boot가 서로 덮어씀. `agreement_labels`(`:249-263`)도 cap·tp 없이 묶음. T1 2차(§1-7)·§4 ③ H 대비 인용 대상 오염. 변이 `o_mismatch_excluded`(`ct_mutants.py:304`)는 값을 보지 않음. 수리: 키 = `_tkey(boot)`·bs·c, 값 = VALID 창 `client_gap_ms` 중앙값, agreement 키에 cap·tp, 변이 2개(마지막 창 유지 / TP2·N=1 합침).
- **MAJOR-2** `costterms_job.sh:41-44`·`ct_runner.py:1622`: 봉인 플래그 `PDMUX_COSTTERMS_SEALED=1` 자기 신고뿐 — 하네스·spec·config digest를 실행 시 검사하지 않음(§12 항목 13 위반 경로). 수리: 봉인 때 `HARNESS_DIGEST.txt`(파일별 sha + spec\_sha256 + config sha) 커밋, 실행 전 검사 `ABORT_HARNESS_DIGEST`(rc 13 계열), 1바이트 변경 변이.

## MINOR
- m1 OFF 진단 (16, 16k)에 ON 짝 없음(`ct_spec.py:225` vs `BS_GRID` `:70`) → `instr_effects`가 조용히 빼서 비교 셀 5→4, `INSTR_EFFECT_SMALL`이 유리해짐. 수리: ON D44·108v에 (16, 16k) 추가(사용자 결정) 또는 비교 셀 수 산출 + A-49 공시.
- m2 격자 boot 서버 사망: DONE 기록으로 재개 시 남은 셀 미실행(`ct_runner.py:767-781`), boot 전체 unusable(`:986`)로 사망 전 VALID 셀도 빠짐, 미실행 셀 행 없음. 수리: 등록 셀마다 `NOT_RUN_AFTER_SERVER_DEATH` 행, 사망 전 셀 적합 포함 여부는 새 해석 항목 + 사용자 확인.
- m3 T3 `RAMP_EXCEEDED_BOUND` 문턱이 a4·tp1 구조로 계산(`ct_extract.py:729`) vs 등록 O는 변형 구조(`ct_runner.py:90`) → J-H2 변형 라벨 거짓 양성.
- m4 T2·T2′·장문 셀에 엔진 측 잔여 요청 검사 없음(`ct_extract.py:506-515`, `:565-582`) — 108u 잔여 decode 시 prefill이 P 64로. 수리: [t\_send, t\_first] 안 `bs_real`이 0(T2·장문)/1(T2′) 아니면 무효, 또는 `wait_engine_idle`.
- m5 `ABORT_CONTEXT`가 거의 항등식(`ct_extract.py:230-232`가 argv 되풀이 값 비교, 가짜 엔진이 clamp 값을 ServerArgs에 씀 `ct_fake_engine.py:264`), T1 실행 가능성 재계산이 계획 context 사용(`ct_runner.py:722-726`), `feasible_cell` `>` vs SGLang `>=`(`ct_spec.py:485`, `tokenizer_manager.py:814`). 수리: `final.context_len` 기준, 가짜 엔진은 argv 되풀이.
- m6 계측 매트릭스 위반 boot가 적합에 남음(`ct_runner.py:698`, `:786`, `:986-987`). 수리: `INSTR_MATRIX_VIOLATION` → unusable 또는 ABORT.
- m7 재개 segment: 새 T3 segment가 σ′ 기준 셀 미재실행(`ct_runner.py:800-831`), segment 간 `by_d`(`:1155`)·J-C `by`(`:1433`)·`next()` 선택 덮어씀.
- m8 장문 boot 사이 GPU 상태 확인 없음 → `AFTER_PRIOR_DEATH` 라벨.
- m9 미등록 노브(`--disable-overlap-schedule`·`--disable-piecewise-cuda-graph`·bf16·`cuda_graph_max_bs` 기본·burst 같은 프롬프트·셀 사이 0.5 s, `ct_runner.py:371-384`, `:779`) → A-49 이후 명시.
- m10 T1·T2 셀에 (wall, monotonic) 쌍 없음(`ct_client.py:170-216`), 재개 시 이전 clk.csv 누락(`costterms_job.sh:79`).
- m11 하드캡에 셋업·분석 시간 없음, vesslctl wall-clock 캡 없음(J-C는 셋업 ≤ 53분이어야 6 h 안).
- m12 원자료가 클러스터 볼륨에만 → SHA256SUMS 검증 보관 단계.
- m13 Workspace smoke가 ENGINE\_DIGEST 미검사(smoke 값이 봉인 입력을 정함) → 결과를 smoke 출력에 기록.

## 해석 항목
- A-42(J-A T1·T2·OFF boot 전부 context 131,072): **OK, T1만 올리는 안보다 우월**(쌍이 노브 하나만 다름). 단 flashinfer cudagraph `kv_indices` 버퍼가 max\_bs × context\_len int32로 커짐(`flashinfer_backend.py:519-528`) → caveat.
- A-46 108u 장문 boot `PDMUX_R2_FIXED_DSM`=44: 장문 셀에 동시 decode 없음 → 쓰이지 않음, 라벨은 `sticky_index`로 "108u". 위험은 `boot.json` 오독뿐.
- 반전 시험(A-42 범위·A-9 NA·A-5/A-46 FIXED\_DSM): 판정 불변 → caveat.

## 가짜 엔진이 숨기는 가정(smoke 점검 목록)
clamp context를 ServerArgs에 씀 · 입장 회계가 하네스 최악 예약식과 같음(T4는 구현 사실 확인) · `mixed` 모드 없음 · abort 즉시 소멸 · 스트림 청크 정확히 1토큰 · TP 두 rank 기록 동일(`TP_RANK_RECORDS_DIFFER` e2e 미시험). smoke 점검: 랭크별 sticky 줄·KV 배너 수, abort 뒤 스트림 종료, bs 96 클라이언트 GIL 지연(confound 7).

## 승계 문장(인용 금지 형태)
- "J-A의 T1·T2·OFF 값은 context 131,072 boot의 값이며, KV 풀 −0.2% 외에 context에 비례하는 비정적 버퍼(flashinfer cudagraph kv_indices)도 함께 바뀐다 — context 73,728 boot 값과 같은 조건이라고 쓰지 않는다."
- "jc_tp2_long_108u의 PDMUX_R2_FIXED_DSM=44는 단일 요청 장문 셀에서 쓰이지 않는 값이며, 이 boot는 D44가 아니다."
