# AUDIT — RESULT_power_cap_passive_2026-10-07.md (claims-auditor, 읽기 전용, 2026-10-07)

판정 요약: **치명 0건.** 문서는 자칭대로 기술적이나 §0(3)·§2 s0·§1-2(E2)·§3 "동거 채널" 문구가 코드·데이터보다 멀리 갔다 → 전부 RESULT 감사 반영판에 수정됨.

| 항목 | 판정 | 요지 |
|---|---|---|
| 1 regress 항등성 | CONFIRMED(+추가) | 같은 파서 규약 재구현 = 항등 검사. 에너지·J/* 열은 외부 대조 없음. N3/E2 window = TIMING(bench duration보다 18–21 s 김), λ0 = bench 길이. |
| 2 §3 정렬 | PLAUSIBLE | (a) `decode_log_interval=40`(`server_args.py:418`), 두 카운터가 같은 루프에서 0부터 — 짝 정당. (b) bracket은 O+d(로그 지연)를 묶음, ms 편향. (c) selftest는 5 s 에피소드·25 %·전체 반전만 — 부분 오정렬 소효과 희석 미검증; 동거 플래그↔0x4 95/99 % 일치가 독립 교차 증거. (d) guard를 올리면 깨끗한 층이 사라짐(0.5 s: n=23, 1 s: 0) — 전부 전환 꼬리 표본. |
| 3 공선성·"클럭 채널 없음" | REFUTED(문구) / CONFIRMED(범위 한정) | 동거 = (64,44) 42,091 / 비동거 = (0,108) 33,522 — SM 수·동거 교락. bs1 (0,108) 비 1.0017(CI≈[1.0016,1.0019])는 "+0.2 %"이지 "변화 없음" 아님. 0.77 = N3C-16. |
| 4 KISTI | (a) 기술로만 PLAUSIBLE (b) REFUTED (c) CONFIRMED | (a) util 마스크는 작업량 정규화 아님, s8p/deconf는 다른 캠페인; +139(+135 아님); "1410 고정"은 deconf 행만. (b) Stage 0은 D108 앵커=16 SM(분할 상태)·명목 셀 60–97 % 유지 ⇒ near-cap은 "전력=분할 증인"의 반대 증거. (c) coupled/deconf 라벨 yml과 일치. |
| 5 E2 | 부호 CONFIRMED / 문구 위반 2곳 | 8/8 쌍 부호 양. 크기 1.03–1.17(창 의존). "37 %"·B "같은 실현 분할"(e_time 없음, n=2 R4″)·§2-3 E2 문장(#14) 삭제. |
| 6 인용정지 | CONFIRMED 0건 | near-miss: e_time 3–14 %는 추정량 명시 덕에 허용(13.5/85.6/91.9 계열 옆에 두지 말 것); "2.99 달성"에 48-cap 한정어. |
| 7 금지 목록 | 불완전 → 추가 | 동거 간섭 1.5–1.6× / 전력=분할 증인 / prefill 축>decode 축 / J/* Job·기판 간 비교 / sticky ON 에너지 / 부하↑효율↑ 권고 / "cap은 prefill 때" 일반화 / rate 순 추세 / 클럭 채널 없음 일반화. |

평가서가 인용 가능한 것(범위 한정): ① N3 한 Job J/req 2.7×↓(0.10→0.77, TIMING 규약; rate=순서 교락) ② J/good cliff·분모 의존(정성) ③ E2 A 쌍 내 전력·J/req 반대 부호 8/8(정규화 교훈) ④ KISTI deconf P16 near-cap 0.00 vs np·P92 0.4–0.8(캠페인별 기술) ⑤ **기존 VESSL 데이터의 cap 중 클럭 하락은 −3…−4 %뿐 ⇒ power cap 처치를 알려 주는 데이터 없음, `-pl` 기판 필요**(핵심) ⑥ bs1 (0,108) +0.2 % @ −2.8 %(n=1,035, 전환 인접).

§3을 결정할 실험: `-pl` 가능한 기판에서 shape B D44 운영점을 한 Job에 400/300/250 W arm으로, cliff 밖 고정 rate, n≥4 교차 반복, PHASE_EVENTS=1, mem clock 기록, 실현 (prefill_sms, decode_sms)×bs 층화 step 시간을 1차 결과로, paired CI.

(전문은 세션 기록; 이 파일은 요약 등재본.)
