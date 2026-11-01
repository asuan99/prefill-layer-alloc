# 하네스 감사 r1 — 기준선 Phase B 사전등록 rev1.1 + 하네스 (2026-11-02, claims-auditor 보고 요지; main 세션 전사)

> 단일 패스(결정 38). 판정: **NO-GO** — 死因 A(지속 용량 규칙, N1·N2 수치), B(용량 아래 셀 goodput 쌍 = 처리량 항등식, N1), C(mixed-chunk 발화 검사·P-4 정의역 퇴화, N1·N3), D(봉인 문언 ≠ 스코어러 5건). 전부 GPU 0으로 수정 가능, 처방은 기존 raw(CAP1-R·CAP2)로 실현 확인.

## 死因과 처방
- **A. 지속 용량 판정**(`sb_runner.sustain_eval`): 분자 "창 안 완료", 분모 "창 안 도착" → 첫 창(채움 과도)·마지막 창(600 s 이후 도착 몰림, 완료 미집계)이 구조적으로 깎임. 정상 서버 모형(CAP1-R D44 E2E 분포, 푸아송 300회) 통과 확률 3.65 req/s 0.037 · 5.84 0.027 · 7.29 0.013. → 처방: 도착 창별 "600 s 안 도착 요청 중 E2E ≤ 60 s 완료 비율 ≥ 0.95", 600 s 이후 도착 제외, TTFT 2배 조건 유지. 처방 규칙으로 정상 서버 통과 1.00, CAP2 과부하 raw 4 run 전부 실패(0.96–1.0 → 0.1–0.33 → 0). 변이 1개 추가.
- **B. 용량 아래 셀 goodput 쌍**: 두 arm이 전부 SLO 안 처리 → goodput = n/duration, ln 비 = drain 차이뿐 → "3% 이내 동등"이 구조적. CAP1-R 용량 아래 rung에서 D44/D34·D44/D64 ln 비 전부 |≤0.010|. Q8(패치 트리 OFF 대 손대지 않은 트리)에서 토큰당 10% 느려도 "동등"이 나옴. 고부하 셀 "best 0.6–0.85" 예보도 기존 데이터와 불일치(PD-mux arm은 5.516·6.343에서 attainment 1.00). → 처방: 쌍의 두 arm 모두 4 seed attainment ≥ 0.98이면 `BOTH_SATURATED`(EQUIV 금지); Q8·Q5에 지연 공동 지표(seed별 paired ln TPOT 평균·TTFT p50·요청 내부 ITL p95 중앙, 같은 3% 규칙) 등록.
- **C.(i) mixed-chunk 발화 검사**: "running ≥ 1인 Prefill batch 로그 줄"은 `--enable-mixed-chunk`와 무관하게 출력(v0.5.10 `scheduler_metrics_mixin.py:389–394`) → 처치 측정 불가. 처방: 서버 로그 server_args `enable_mixed_chunk=True` 확인 + proxy를 "mixing 기회 수"로 개명해 SG-stock 음성 대조 병기. **(ii) Q6·P-4**: 계측 60요청에서 W1 길이 몫 512±25% 3.7%·2k 2.4%·8k 0% → 기대 span 2.2/1.4/0 < 등록 "각 ≥ 10". 처방: 8k 제거, n < 10이면 `UNSCORABLE`, 필요 시 계측 boot에 512·2k 합성 프롬프트 각 10개.
- **D. 문언 ≠ 코드**: 기준 용량 "7개 arm 최소" vs 코드 5개(SG-rep·PM-ref34 제외, PM-ref34는 CAP1-R 6.343 승계) · §3(가) 4개 vs 5개 · §5-0(a) "상수 같을 필요 없음" vs 코드 |SG−VL| ≤ 5 ms 요구 · §6 "8k ≥ 10" 잔존 · PM-ref argv 묶음(mem 0.6·overlap OFF·chunked 끔·seed 1) 본문 부재(confound 10). 처방: 문언 정합 + caveat.

## 반전 시험 표(요지)
지속 분모·창 처리: **반전**(정상 PASS→FAIL) · 0.95 문턱: 첫 창 0.925–0.934라 0.90/0.95에서 갈림 · TTFT 2배: 불변 · 지속 격자 {1.0,0.8,0.5}: 격자가 최솟값 arm을 정할 수 있음(기준 용량 39% 차, caveat) · 20 ms: 반전 불가(goodput 영향 ≤1.3%) · K3: 데이터 전 불가 · EQUIV sd<0.0186: 정확, 엔진 간 sd 예보 0.02라 Q3·Q4 EQUIV 사실상 도달 불가(검출력 caveat) · W3 rate: PM-ref 용량 위 가능(caveat).

## 파이프라인 모집단 표(요지)
지속 λ\* 입력 < 100%(마지막 창 혼입) → A · 기준 용량 5 arm vs 문언 7 → D · TTFT 오프셋(b)는 저부하 셀만 · r_ser: 다른 stream extend가 경고 없이 제외(fail-closed 아님 → 한 gap이라도 > 0이면 Q6 INVALID) · mixed-chunk proxy는 처치 무관 → C.

## 특별 검토 ①–⑩ 요지
트리 스탬프는 항등식 아님(boot 시 `find_spec` 해석 위치 검사) · 승계 상수 공시 OK, PM argv 묶음 누락 · Q8 estimand는 맞으나 B · 네 번째 Job: W3 용량 0.225→290분·0.15→361·0.1→468(하드캡 320) → 사다리 뒤 남은 시간 가드 필요; 두 번째 Job도 기준 용량 2.5면 278분 · "측정하지 않은 것"이 질문 목록과 대조 안 함, 건너뛴 boot 누락 → Q1–Q8 행 강제 + DONE.json status · fail-closed: pristine 보존·freeze sha·오프셋 미계산 OK; freeze sha는 이미지 파일 해시이지 실행 시점 site-packages 아님(caveat).

## 차단 수정 5건(CPU)
1. 지속 판정 교체 + 변이 · 2. `BOTH_SATURATED` + Q8·Q5 지연 공동 지표 · 3. mixed-chunk server_args 확인·proxy 개명·8k 제거·P-4 n<10 UNSCORABLE·다른 stream extend → Q6 INVALID · 4. 문언 정합(최솟값 arm 5개 사유, 사다리 수, smoke 5 ms 규칙, Q6 격자, PM argv 묶음, 고부하 셀 예보 갱신) · 5. "측정하지 않은 것" Q1–Q8 강제 + DONE.json, 네 번째 Job 남은 시간 가드.

## 인용 금지 문장(결과 문서 승계, 문자 그대로)
1. "이 캠페인의 PD-mux 대 stock 라벨은 PM-ref 구성 묶음(D44 또는 D34, mem 0.6, overlap OFF, chunked prefill 끔, cap 48, budget 65536, 모두 KISTI 승계) 대 SGLang 기본값의 비교이며, PD-mux 단독 효과도 static PD-mux 최적도 아니다."
2. "판정 셀 rate는 다른 Job(다른 GPU)에서 잰 지속 용량에서 왔고, 지속 점 격자 {1.0, 0.8, 0.5}가 최솟값 arm을 정할 수 있으므로, 셀의 상대 부하는 arm·Job마다 다르다. 기준 셀 비교는 용량 경계 rung에서 잰 것이라 기판 동일성의 증거가 아니다."
3. "두 엔진의 TTFT 시간 기준은 저부하 셀에서만 확인됐고 ITL 측정의 동등성은 확인되지 않았다. 엔진 간 쌍의 EQUIV는 n=4에서 사실상 도달 불가다(sd 예보 0.02 > 0.0186)."
4. "Q8 결과는 이 셀의 goodput·지연 범위에서만 성립하며, 수정 트리 OFF가 커널 수준에서 upstream과 같다는 근거가 아니다(causal_conv1d int64는 코드상 다름)."
5. "W3 셀 rate는 SG-stock 용량만으로 정했으므로, PD-mux arm이 그 rate에서 용량 위였다면 W3 라벨은 부호만 인용하고 크기는 인용하지 않는다. r_ser·T_dec는 수정 트리·플래그 OFF 계측 boot 범위다."
Q4 caveat 5개 = 상한. 하나 더 붙으면 Q4는 정본에 올리지 않고 설계 재작성.

## 정본 밖 발견(확인 필요)
`policy_cap2/RESULT_cap2_sustain_2026-10-29.md` §4 "완료 처리율 약 3.0 req/s, 마지막 완료 1405–1475 s"가 raw(`cap_char/policy_runs/pdmux-policy-cap2-e90e233-20261029023336/sustain/seg/*.jsonl`)와 충돌: raw 4 run 마지막 완료 820–865 s, 평균 완료 5.06–5.34 req/s, 600 s 안 완료 3148(D44 s201). CAP2 감사 r1의 "2.97–3.12"와도 불일치 → result-analyst 재계산 필요. 이 PREREG §7의 "기준 용량 3.0 가정"도 영향.

## 사용자 결정 필요
차단 수정 5건 반영 뒤: (a) 재감사 없이 변이 테스트 통과를 조건으로 봉인(권고) / (b) diff만 보는 제한 재감사 1회 / (c) 고정 셀 대신 arm별 상대 rate로 재설계.
