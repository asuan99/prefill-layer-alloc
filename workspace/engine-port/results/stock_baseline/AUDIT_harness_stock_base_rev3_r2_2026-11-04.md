# 하네스 제한 재감사 rev3 r2 (diff fdb7c4ba..6f74bf7f, 2026-11-04; claims-auditor 보고 요지, main 세션 전사) — rev3의 2회째(상한)

> 판정: **NO-GO (N1 변형: 변이 가드 부재)** — 판정 셀 축약 `cell_runs`(`sb_analyze.py:438-443`, `cell == c ∧ boot_role == "meas"`)의 셀 조건을 지워도 테스트 90개가 전부 통과(픽스처가 셀 하나뿐). 실제 J2에서는 50%·80% 셀이 같은 arm·seed라 셀 조건이 빠지면 50% 셀 분석이 80% run으로 덮인다. 현재 코드는 맞게 재지만 가드가 비어 있다(rev1부터 있던 구멍, 앞선 감사 3회가 놓침). 처방 없음(상한).

## 닫힘
- r1 死因(Q1-SG 띠): **닫힘**(띠 재선택 없이 삭제).
- Q2 부호 전용: 문서·코드 일치(M91·M92·M101), C12 부재 맞음.
- PM-rep 플라시보: Q2 "하나라도 무효면 BOOT_NOISE"(코드 any) 일치, 계열 매핑 등록 상수라 자유도 0, argv 동일성 fail-closed(M99; env 비교 삭제 변이는 생존 — 사소). PD-mux 플라시보 sd 예보 부재는 규칙 입력 아님.

## 핵심 예보(첫 번째 Job raw 기준)
- PM-ref·VL-stock은 판정 셀 rate 범위(≤ 5.836)에서 두 seed 모두 attainment 1.00 → **Q4는 두 W1 셀 모두 `BOTH_SATURATED` 예상**. PD-mux 플라시보가 Q4에 할 수 있는 일은 BOOT_NOISE로 바꾸는 것뿐.
- Q2·Q3·Q5·Q8: 셀마다 약 60% BOOT_NOISE, 나머지 부호만. **정본에 올라갈 goodput 라벨은 사실상 0.**
- W3 셀 PM-rep: Q7 키는 Q2·Q3뿐이라 PM-rep은 무효 조건만 추가(+43.4 min), Q7 도달 확률 하락.
- Q4 caveat 수: 엄격히 읽으면 6(1·2·3·10·15·16) > 5 → 정본 비승격. 문장 16을 3에 합치거나 10을 SG 전용으로 읽으면 5.

## 인용 금지 문장(문자 그대로)
- 16: "VL-stock과 PM-ref34의 boot 간 잡음은 이 캠페인 어디에서도 재지 않았다. Q3(vLLM 대 SGLang)은 SGLang 쪽 플라시보만, Q4(PD-mux 대 vLLM)는 PD-mux D44 쪽 플라시보(PM-rep 대 PM-ref)만 통과 조건으로 쓰므로, Q3·Q4의 부호·라벨에는 vLLM boot 하나의 우연이 섞일 수 있고, PM-ref34 쌍은 D44 복제로 대신 검사됐다."
- 검출력 공시: "첫 번째 Job에서 PM-ref와 VL-stock은 판정 셀 rate 범위(5.836 req/s 이하)에서 두 seed 모두 attainment 1.00이었으므로, Q4는 두 W1 셀 모두 BOTH_SATURATED가 예상 결과다. PD-mux 플라시보 sd 예보는 등록되지 않았다."

## 사용자 결정 필요
(a) 50%·80% 셀 분리 테스트 1개 추가 + 재봉인, Q4 정본 비승격 전제로 실행(권고) / (b) (a) + W3 셀 PM-rep 제거(네 번째 Job 297.9 → 254.5 min) / (c) 세 번째·네 번째 Job 보류(첫 번째 Job raw가 "기준 용량 아래에서는 PD-mux·vLLM goodput이 안 갈린다"를 이미 예보).
