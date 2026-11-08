# 설계 결함·예측 오류 정리와 반복 패턴 수정 (2026-11-01)

> ★결과 감사 r1(2026-11-08) GO-with-edits — 판정서 `reports/audit/2026-11-08_unaudited_r1/AUDIT_design_defects_policy_memos_dstar_r1_2026-11-08.md`, 수정 전 인용 금지 목록 적용. 본문은 고치지 않았다.

> 지위: 검토 메모(사용자 요청, 미감사). 새 측정 0, 판정 변경 0. 대상 = 정책 캠페인(CAP1-R·CAP2·TUNE)과 통합 정책 측정(policy_implfix P·H·R·E3), GPU gate rev0.2, D\* 예측식.
> 근거: 각 결과 감사 r1(`policy_cap1r/AUDIT_cap1r_result_r1_2026-10-29.md`, `policy_cap2/AUDIT_cap2_sustain_result_r1_2026-10-29.md`, `policy_tune/AUDIT_tune_result_r1_2026-10-29.md`, implfix `policy_implfix/AUDIT_harness_policy_implfix_r1_2026-10-31.md`·`AUDIT_result_policy_implfix_r1_2026-11-01.md`), 메모 `policy_tune/MEMO_HE0_cell_load_redefinition_2026-10-30.md`.
> 라벨: [감사] = 감사 r1에서 확정된 것 · [정리] = 이 메모가 묶은 것(감사 미경유).

## 1. 심각도 순 (판정 무효 > GPU Job 낭비 > 정본 오염 위험)

| 순위 | 결함 | 영향 | 분류 | 출처 |
|---|---|---|---|---|
| S1 | 과부하 셀(HI = 1.5×λ\*)이 FEAS 혼잡 가드를 닫아 동적 arm을 D44에 고정 | HE0 판정 불가. TUNE·EVAL·H의 동적 arm 전부 같은 분할 null. 루프 수정으로 안 풀림 | 계획+셋업 | [감사] TUNE (2), HE0 메모 |
| S2 | 계측 boot 축약(`pc_reduce`·`dose_reduce`·E3 handoff 카운터)이 boot 전체 텔레메트리를 읽어 warmup·가드 프로브 혼입 | P 전제 검사(r·dose)·E3 문턱 전부 오염. P Job 판정 없이 종료 | 셋업(하네스·테스트) | [감사] implfix 결과 감사 1·2·8 |
| S3 | 승계 상수(cap 48·D44·budget 65536·h 1.5)를 Nano에서 재결정하지 않고 사용 | S1의 직접 원인. R Job에서 D34–64 동률이라 D44 근거 없음 | 계획 | [정리] HE0 메모 3 |
| S4 | HI 구간 threshold-goodput이 구간 길이에 묶임(ill-posed), ±5% cliff 검사가 시간축 절벽을 못 잡음 | H −4%는 크기 인용 불가. TUNE 감사에서 알고도 P·H 설계에 승계 | 계획+셋업(게이트 정의) | [감사] TUNE (4), implfix §4 |
| S5 | P 판정 쌍 검출력 부족(sd(ln) 0.030 > 필요 0.0186) | S2 없이도 UNRESOLVED였을 가능성. 하네스 감사가 caveat로만 적음 | 계획 | [감사] implfix "다음에 필요한 것" 1 |
| S6 | λ\*를 120요청 버스트로 정의, 지속 완료율과 차이 — ★정오(2026-11-02): 지속 완료율은 ≈3.0이 아니라 **5.06–5.34**(CAP2 감사 스크립트 단위 오류, `policy_cap2/RECHECK_sustain_throughput_2026-11-02.md`); 버스트 대비 약 1.4배, LO 2.74는 그 0.51–0.54배 | HI(10.94)는 여전히 지속 불가; "LO가 지속 처리율 경계"는 철회 | 계획 | [감사] CAP1-R caveat 1·2, CAP2 (3)+정오 |
| S7 | E1·E2 묶음 arm만, E2 단독 arm 예산 때문에 제거 | H −4% 귀속 불가 | 계획 | [감사] implfix B-8 |
| S8 | E3를 run당 전환 5–9회 arm에서 측정 | handoff가 goodput에 드러날 기회 없음 | 계획 | [감사] implfix §7 |
| S9 | D\* 예측식 경성 ITL 문턱 가정 → E-6·E-7 MISS, D64 +7% 미실현 | 정상적 예측 실패(봉인 예측). CM-1 재검토 대상 | 모형 | [감사] implfix §6 |
| S10 | GPU gate 스코어러 per_flag E3 롤업이 봉인 문언보다 좁음 | E3 UNRESOLVED인 채 성능 측정 진입 | 셋업 | [감사] gate rev0.2 §0 |
| S11 | 결과 문서 초고의 사후 서술 3건(CM-2 γ 1.3–1.7, CM-4 두 모드, PC_TAILS) REFUTED | 감사가 잡아 정본 오염 없음. 매번 감사 의존 | 셋업(에이전트 프롬프트) | [감사] implfix 결과 감사 3·4 |
| S12 | λ\*_W3 여유 1요청 · 동률 규칙 미등록 · CAP2 문서 오기 | 판정 불변, caveat 처리 | 계획(경미) | [감사] CAP2 (1), CAP1-R (1) |

## 2. 반복 패턴 — 원인 분류와 수정

### P1. 셀을 과부하에 두는 선택이 반복 (S1·S4·S5·S6)
- 계획 오류. 셋업이 막지 않음: result-analyst 게이트 4는 "명시"만 요구, experiment-runner 규약에 부하 위치 항 없음, claims-auditor 카탈로그에 "과부하 → 컨트롤러 고정" 없음.
- 수정(적용 2026-11-01): experiment-runner 규약 "부하 위치"(HI = best static attainment 0.6–0.85) · result-analyst 게이트 4 차단형(TTFT≤SLO 비율 < 0.5 구간은 부호만) · claims-auditor confound 12 · PREREG 필수 항 "검출력 사전 계산".
- 남은 것: HE0 셀 재정의 원칙 (a)–(d) 사용자 결정. 이것 없이는 규약만 바뀌고 측정은 안 바뀐다.

### P2. 계측 boot가 측정 arm을 대변한다는 가정 (S2·S10)
- 셋업 오류. 테스트 69개가 규칙만 검증, 입력 모집단 미검증. 하네스 감사 1회가 결정 규칙(K1–K5)만 보고 파이프라인은 안 봄. 문언↔스코어러 대조 테스트 없음.
- 수정(적용): experiment-runner "분석기도 phase 분리" + 변이 테스트 요건 · claims-auditor "파이프라인 모집단 층"(라벨별 입력 필터 표, 미충족 NO-GO) · confound 13 · 봉인 문언↔롤업 대조.
- 남은 것: 다음 PREREG의 축약 함수에 부하 창 인자 구현(봉인 코드는 수정 금지라 새 PREREG에서).

### P3. 승계값 위에 쌓기 (S3·S6·S7·S8)
- 계획 오류. 측정 우선 원칙(결정 38)이 "승계값으로 일단"으로 작동. right-sizing 기준이 비용이지 답하는 질문 수가 아님.
- 수정(적용): experiment-runner "승계 상수"(판정 셀은 재결정 뒤 봉인, 순서 cap → λ\* → h → TUNE → EVAL) · "처치 발생 횟수" 등록 · PREREG 필수 항 "답하는 질문 목록"(예산 부족 시 arm이 아니라 질문을 줄임).

### P4. caveat이 설계 수정을 대체 (S5·S12·S1)
- 셋업 오류(2026-09-10 등급 개정의 부작용). GO-with-caveats = 실행 승인이라 검출력 미달을 알고도 제출. caveat가 정본으로 누적.
- 수정(적용): claims-auditor `GO-with-caveats(검출력)` 등급(실행 승인 아님, 예상 결과 명시 후 사용자 결정) · caveat 5개 초과 시 설계 재작성 분기.

### P5. 분석가의 사후 서술 (S11)
- 셋업 오류. result-analyst "회의 기본값" ↔ "cost model 절 빈칸 금지"가 충돌해 사후 해석 압력.
- 수정(적용): cost model 절 두 칸(사전 예측 잔차 표 / `[사후·미감사]` 태그) · "이 Job이 측정하지 않은 것" 절 신설, 빈칸 금지는 그쪽으로.

### P6. 예측식 가정 실패 (S9) — 정상. 절차만
- D\* 식 수정 = 새 봉인 + 새 Job 1회. `PROPOSAL_dstar_model_2026-10-30.md` §7-2 절차를 그렇게 고정.

## 3. 적용 내역 (2026-11-01)
- `.claude/agents/experiment-runner.md`: 규약 3항(부하 위치·승계 상수·처치 발생 횟수) + phase 분리 분석기 요건 + "PREREG 필수 항" 절.
- `.claude/agents/result-analyst.md`: 게이트 4 차단형 · "측정하지 않은 것" 절 · cost model 절 2칸.
- `.claude/agents/claims-auditor.md`: confound 12·13 · 파이프라인 모집단 층 · `GO-with-caveats(검출력)` · caveat 상한.
- 정본 사본: `tools/claude/` (`sync_claude_tools.sh --import`).

## 4. 적용하지 않은 것 (사용자 결정 또는 새 PREREG 몫)
- HE0 셀 재정의 원칙 (a)–(d) · 승계 상수 재결정 Job(cap → λ\*) · 축약 함수 부하 창 구현 + 변이 테스트 · D\* 식 수정 절차 문서화 · MEMORY.md 갱신(promote 몫).
