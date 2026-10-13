# 하네스층 한정 감사 — M2g 수리 a41e725 (2026-10-13)

> 감사자: claims-auditor(읽기 전용, GPU 0). 요지를 메인 세션이 저장(caveat 원문 그대로).

## 판정: `GO-with-caveats` — J4·J8 재제출 승인, 死因 0, 필수 수정 0
"G1·G2를 닫았다"는 뜻 아님. 회귀 271 OK, 수리 9건 되돌림 변이 9/9 KILLED(항등식 아님). 엔진 대조: `_process_input_top_logprobs` [None] 선행·마지막 pop → logprob_start_len=len(pids)에서 항목 j(j≥1)가 cont[j] 예측, teacher 정렬 일치; /tokenize·/generate 모두 add_special_tokens=True; /health는 SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION off면 생성 없이 200. J2 원자료 재채점(정렬 검증 예보, **인용 금지**): 제외 위치 64(프롬프트당 1), TP=1 teacher agreement 0.998.

## 항목
1. G1 step별 생존: 등록 의미("G1 행 범위") 보존, 새 공허 PASS 없음(G1은 첫 step, 단일 Popen 재기동 없음, expect_step 거부, 옛 J4/J8 산출물은 새 채점기에서 NOT_TESTED). 잔여: G1 직후 지연 사망이 G2 NOT_TESTED로만 보임 → C1.
2. /tokenize 경로: 꼬리 16 위치+길이 대조는 타당한 fail-closed, 앞부분 같은 길이 치환은 TP↔TP 비교 무영향.
3. 위치 0 제외: 등록과 충돌 없음, 대가 cont[0] 미비교 → C2.
4. 결측 → NOT_TESTED: 교훈 21 정합, 단 결측 1개면 관측된 발산까지 덮임(G1 원칙과 비대칭) → C3(권고: 존재 레코드로 임계 초과면 FAIL).
5. G2 동시성 32→1(미등록): 반전 미실증 → C4.
6. J4·J8만 재실행, J1·J2 유지: 등록(Job 단위 재실행, 같은 Job 원칙) 정합, 수리는 G1 생존·G2만. J2 G1 4/4 PASS는 더 엄격한 규칙으로 얻음. J2 G2 TP2는 NOT_TESTED 유지(사후 재채점 금지) → C5.

## 반전 표
G2 동시성 32→1: 미실증(동시 32 바닥 p99 0.0592, 임계 0.237, TP2 p99 0.0598 PASS; 직렬이면 임계 0.05 근처로 하락 가능). 결측 문턱: 구성 가능·실측 불가(G2 FAIL은 표기 전용). 위치 0: 없음. G1 생존 시점: J4/J8 FAIL → 재측정 대상(정당한 원인 제거).

## caveat (문자 그대로 승계)
- **C1**: "J4·J8 재실행의 G1 '비정상 종료 0'은 각 boot의 G1 step 직후 생존만 뜻한다. fused boot가 G1 뒤 G2 단계에서 죽은 경우는 G1·제외 판정에 반영되지 않고 G2 NOT_TESTED로만 나타난다(boot 끝 기록 `alive_after_boot_end`는 기술치)."
- **C2**: "G2는 프롬프트당 continuation 위치 1…127만 비교한다. 첫 continuation 토큰(프롬프트 마지막 위치의 예측)은 비교되지 않았다."
- **C3**: "G2 NOT_TESTED에는 'TP=N 레코드 일부 결측' 사례가 포함되며, 이 경우 존재하는 레코드의 발산 여부는 판정에 쓰이지 않았다."
- **C4**: "TP4·TP8 G2(바닥·시험 모두)는 직렬(동시 1) 송신에서 측정됐다. 바닥은 배치 구성 잡음을 포함하지 않으며, 동시 32 송신의 G2와 비교하거나 TP 추세로 잇지 않는다. TP2 G2는 미측정(NOT_TESTED)이다."
- **C5**: "M2g PASS 표 J1·J2는 수리 전 하네스(boot 끝 생존 규칙, G2 동시 32)로, J4·J8은 수리 후 하네스로 산출됐다. 네 표는 서로 다른 커밋의 산물이다."

## 재제출·봉인
1. 이 판정서 커밋 후 dry-run → 승인 → submit(ENGINE_DIGEST 일치 기대). 2. J4·J8 표는 새 Job 산출물만(옛 99b5e5e·d9e7743 재조립·재채점 금지). 3. `results/multigpu/M2G_PASS_TABLE_J{1,2,4,8}.json`+`.sha256` 커밋(J1·J2는 313be0a·2fc3880 산출물 바이트 그대로), M3 3자 대조. 4. 결과 확인: `alive_after_g1_*.json` 존재·step 이름, `g2_tf_*_r*.client.json` errors 0, fused TP4/TP8 boot 끝 생존(기술치).
