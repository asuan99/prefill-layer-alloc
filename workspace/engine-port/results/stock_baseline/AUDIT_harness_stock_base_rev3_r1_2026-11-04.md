# 하네스 제한 감사 rev3 r1 (diff 9e12a4e9..c394b799, 2026-11-04; claims-auditor 보고 요지, main 세션 전사)

> 판정: **NO-GO (N3)** — 死因 1건: §8 Q1-SG 예측 띠 [2.5, 4.8]에 "거짓"이 없다(아래쪽 사다리 rung 4개가 전부 띠 안; 2^8 패턴 전수에서 MISS 불가). 처방(1회): Q1-SG 행 삭제(PREREG §8, `SP.Q1_SG_BAND`, `score_predictions` 블록, 테스트), 띠 재선택 금지. 처방 반영 후 등급 상한 = GO-with-caveats(검출력), 실행 승인 아님.

## 사후 등록 위험 표(요지)
SG 제외 기준 용량 → 판정 셀 rate가 SG 실패 영역(5.5–7.3)으로 올라가 Q2 ln ≈ 1.5–2.2(**PD-mux 유리**), 데이터 전 등록이면 달랐음 · Q2만 일반 라벨은 **비대칭·자기 모순**(코드 주석 사유가 Q2의 SG arm에도 해당) · 아래쪽 사다리 rung은 데이터 본 뒤 선택(해 낮음) · refcell PM-ref·Q3 지연 지표·W3 floor 이동은 중립.

## Q2 크기 = ITL 절벽값(confound 5), N1 아님(B5가 부호 기전을 보여 줌)
ln(PM/SG) 5.516: 1.507/1.504; ITL SLO +5% → 0.740/1.466, +10% → **0.029**/0.702(ABOVE 소멸). ITL 축 cliff 검사를 유효성 사슬에 넣으면 그 검사가 자유 표면이 되어 반전(±5%: 5.516만 INVALID, ±10%: 전 rung INVALID) → 넣지 말 것. 권고: Q2를 부호 전용에 포함. 일반 라벨 유지 시 caveat 6 → 정본 비승격.

## 플라시보 검출력
proxy(SG-off vs SG-stock, 같은 seed, n=6) sd(ln) **0.113** = 예보 0.006의 19배(SG good 요청 26–40건뿐). K3 무효 확률 n=4: sd 0.113 → **0.60**. K3은 셀 단위라 Q4(PM vs VL)도 함께 소멸, P-3 UNSCORABLE. 예상 결과 = R2 셀 50–60% BOOT_NOISE.

## 기타
refcell PM-ref 0.967: 10% 이상 저하만 검출, 빨라진 기판·SG 정지 길이 이동은 미검출(C15) · W3 floor 이동 문제 없음 · 시간 모형 2.7배 보수적, 문제 없음 · Q1 ladder_down은 두 Job 혼합(스탬프 필터 없음, C14).

## 인용 금지 문장(문자 그대로)
- C12(Q2 일반 라벨일 때만): "Q2(PM-ref 대 SG-stock)의 goodput ln 비 크기는 손대지 않은 SGLang의 요청 내부 ITL p95가 60 ms SLO 바로 위(63–66 ms)에 몰린 ITL 축 절벽 값이다. 첫 번째 Job raw에서 ITL SLO를 5% 올리면 5.516 req/s의 ln(PM/SG)가 1.507 → 0.740(seed 201), 10% 올리면 0.029로 떨어지므로 Q2의 % 크기는 인용하지 않는다."
- C13: "판정 셀 rate는 첫 번째 Job에서 손대지 않은 SGLang의 실패를 본 뒤 정한 기준 용량 정의(rev3, SG 제외)에서 나왔으므로, 80% 셀 rate가 5.516 req/s 이상이면 P-3의 HIT(+)는 첫 번째 Job 데이터로 이미 정해진 결과이며 사전등록 적중으로 인용하지 않는다."
- C14: "Q1의 SGLang 버스트 λ\*는 첫 번째 Job(GPU bc0be109)의 5.516 이상 점과, 기준 셀이 없는 아래쪽 사다리 Job(다른 GPU일 수 있음)의 점을 한 pick 규칙으로 합친 값이며, 아래쪽 rung은 첫 번째 Job 결과를 본 뒤 정했다."
- C15: "rev3 기준 셀(PM-ref 7.29491 req/s)은 PM-ref 용량 쪽 기판 저하(약 10% 이상)만 잡고, 빨라진 기판(최대 차 0.033)과 손대지 않은 SGLang의 도착당 정지 길이 이동(PM-ref 간극 45 ms는 SLO에서 25% 떨어져 있음)은 잡지 못한다."
- 검출력 공시: "판정 셀의 플라시보(SG-rep 대 SG-stock)는 SG가 자기 ITL 용량 위에 놓일 수 있는 rate에서 잰다. 첫 번째 Job proxy sd 0.113은 등록 예보 0.006의 19배이므로, BOOT_NOISE 셀은 기판 잡음의 증거로 인용하지 않는다."

## 사용자 결정 필요
(a) Q2 부호 전용 + Q1-SG 삭제 + 재봉인 후 실행, BOOT_NOISE 위험 공시(감사 권고) / (b) (a) + PD-mux 플라시보 arm(PM-rep) 추가로 Q4·Q2 플라시보를 SG 절벽 잡음에서 분리(셀당 4 run 추가, 새 rev) / (c) J1-down·J1b까지만, J2 보류.
