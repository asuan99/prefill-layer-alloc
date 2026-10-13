# 결과 감사 판정서 — N2(3-a) Job-EA `pdmux-n2-static-grid-7e09cc3-20261013022149` (claims-auditor, 2026-10-13, 읽기 전용)

> 저장: 메인 세션. 판정·허용/금지 문안·caveat는 감사 보고 그대로, 근거 서술은 요지. 기준 PREREG rev18(규칙 봉인 `56473d6f…`), `EA_RESULT_SEAL.json` 재해시 일치(`1af88702…`/`35d015e8…`, 의미 한계 N2C-95). 블라인드: Job-PROFILE 원자료·요약 미열람, 신규 노출 0(N2C-87·98 병기). 성분 측정 결과 감사 — 성능·정책 판정 아님.

## 0. 한 줄 판정
**死因 0, 등록 라벨 반전 0.** ρ_off·ρ_on 8/8 `COLOC_SLOWER_GE3`, `COLOAD_NOT_JOINED`, `DS3_MISMATCH_OBSERVED_8`, `NEG_CONTROL_EA_HELD`, 예보 6건 HELD — 기계적 `CONFIRMED`. 클럭 성분을 등록 가정 밖 최대(탄성 1, 창별 실측 클럭)로 빼도 64/64 짝 > ln 1.03(최소 0.069). 해석 3곳 강등·정정: "ρ ≈ 1.16 = 동거 간섭 크기"는 (a) 이 기판 전력 한도 포함 합성 감속 `CONFIRMED(scoped)` / (b) 전력 무관 간섭 크기 `NOT-YET-SUPPORTED`로 분리; "ρ는 D·b·c·prefill 강도에 무관"은 `NOT-YET-SUPPORTED`("prefill 진행량 5배" 프레이밍 `REFUTED`); result-analyst의 "`-lgc` 가능 여부 미확인"은 사실오류(2026-10-07 프로브 rc=4).

## 1. 라벨·주장별 판정
| # | 주장 | 판정 |
|---|---|---|
| 1 | ρ_off 8/8 GE3 | CONFIRMED(스코프: 이 Job·GPU `3be4c7da`·A-agn·legacy·CG ON·cap 48·E-α 8셀·지속 backlog) |
| 2 | ρ_on 8/8 GE3 | CONFIRMED(클럭 보정 후 최소 0.069) |
| 3 | ρ ≈ 1.16 = 동거 간섭 크기 | 분리: (a) 합성 감속 CONFIRMED(scoped) / (b) 전력 무관 간섭 크기 NOT-YET-SUPPORTED — C 창 64/64 SW power cap(0.94–1.00, 1325–1361 MHz, 371–382 W), A1·A2 128/128 cap 밖(1410 MHz) |
| 4 | 클럭 성분 ≤ ln 0.048 | PLAUSIBLE(조건부), 정정: 창별 최대 ln(1410/1325)=0.062; 탄성 ≤1·HBM 클럭 불변 가정(`clocks.mem` 미기록) |
| 5 | D·b·c 주변 차 <3% | CONFIRMED(산술 기술값) |
| 6 | ρ가 D·b·c에 무관(발견) | NOT-YET-SUPPORTED(사후 동등성, D 두 수준, 공통 cap) |
| 7 | 곱셈적 ⇒ 고정 오버헤드 아님 | 고정 가산 반증 CONFIRMED(scoped), 단 #6과 같은 관측의 재진술(독립 증거 아님) |
| 8 | 증가분은 decode 드레인 안 | CONFIRMED(scoped); 판별 근거는 issue ≈ wait |
| 9 | prefill 호스트 발사 비용 가설 REFUTED | CONFIRMED(scoped, 발사·고정 호스트 비용 한정) |
| 10 | 다른 분할의 prefill 실행이 decode 드레인을 늘림 | PLAUSIBLE; 채널 귀속 금지 |
| 11 | prefill 진행량 5배에도 ρ 동일 | 프레이밍 REFUTED(시간 단위 ≈1.4배, D와 공변), 주장 NOT-YET-SUPPORTED |
| 12 | COLOAD_NOT_JOINED → batch 합류 효과 아님 | CONFIRMED(scoped) |
| 13 | DS3_MISMATCH_OBSERVED_8 | CONFIRMED(scoped: 8셀·지속 backlog·이 기판); 방향은 cap에 오염되지 않음 |
| 14 | NEG_CONTROL_EA_HELD | CONFIRMED |
| 15 | 예보 6건 HELD | 기계적 CONFIRMED; F-N2-3은 방향만(설계 대역 [1.3,1.6] 크기 빗나감, 실측 1.14–1.17) |
#9 유사 경로 경고: 이 ρ를 "동적 제어가 진 이유"·"switch/동거 오버헤드"로 쓰지 마라(HE0 원인은 positioning·entanglement 확정).

## 2. 반전 시험 표
| 자유 표면 | 민 범위 | 판정 변화 | 근거 |
|---|---|---|---|
| 클럭 성분 제거 | 탄성 0→1, 창별 | 라벨 불변(64/64), 크기 변함 | 보정 후 평균 0.100, 최소 0.069, 최대 0.121; 한 짝 반전에 탄성 ≥1.77 필요 |
| 분모 A_on | 보간→A1/A2 단독 | 불변 | 최소 0.115/0.131 |
| C 창 | 처음 128→W_co 전체 | 불변 | 0.123–0.165 |
| 판정량 | ρ_off↔ρ_on | 불변 | 최대 |차| 0.019 |
| 외삽 | 허용→엄격 | 불변 | 0/64 |
| 추정량 | S-per→S-dev | 불변 | 중앙 차 0.0029 |
| 문턱 | 3%→10% | 불변 | 최소 0.1198 > ln 1.10 |
| cap 밖 iteration 부분집합(진단) | — | 불변, 분리 근거 아님 | 211/41,910 iteration, 11복제 중앙 +0.0006 |
반전 0 → 死因 없음, 클럭 표면은 caveat(N2C-102·103).

## 3. power cap 범위
외부 오염 아님(자기 부하가 만든 공유 전력 봉투 결합 채널). (i) 운영 대표성(DS-3) 질문에선 처치의 일부, (ii) 비-SM 공유 자원 간섭 질문에선 confound, (iii) 다른 power limit·기판 이식 불가. "운영점에서 항상 동반"은 NOT-YET-SUPPORTED(E-α는 전력 최대화 부하). 허용 문안 둘로 분리(§6 A·B), 전력 무관 간섭 크기 문안은 불허.

## 4. 기여 2·DS-3에 대해
말할 수 있음: DS-3 불일치가 D16·D44 양쪽에서 관측(스코프 진술); green-context 분할 아래에서도 동거 prefill이 decode step을 늦추는 결합 채널이 존재(성분 관측, "분할 = 격리" 아님의 성분 수준 반례); 절대 증가분은 A_on 비례(기술값). 말할 수 없음: SM 배분·static·d_min·SLO goodput 함의, layer-type 재배분 경계, 굶김 연쇄 증거, 예측기 ×1.16 보정, hybrid 조성 귀속, "DS-3 해소".

## 5. result-analyst 보고서 정정
R-1 `-lgc`/`-rgc`는 2026-10-07 `job-x3lzellrkm8z`에서 rc=4(CAP_SYS_ADMIN 없음) — 클럭 고정 대조는 이 기판에서 불가(`-lmc` 미시험). R-2 클럭 성분 상한 창별 0.062, HBM 클럭 미측정. R-3 "5배"는 iteration 단위, 시간 단위 ≈1.4배·D 공변. R-4 무관성 패턴은 기술값으로 강등, 곱셈성은 독립 증거 아님. R-5 `sm_clk_samples.csv` 0행·미지원 사유 미기록(NOTICE 계약 불일치) — 공시. R-6 `ln_rho_*_uncapped` 64중 1개만 값.

## 6. 허용 문안 (문자 그대로)
- (E-α 셀 라벨) §16-1 E-α 틀을 `N2_EA_LABELS.txt` 값으로 채운 셀별 문장 8개 + 아래 (P) + §16-2 필수 병기(N2C-87·89, NEG_CONTROL_EA, K·κ\*·ctx_mid(C)·정지 수, Job 이름·GPU UUID).
- **(P)** "Job-EA의 C 창(64/64)은 GPU SW power cap 아래(SM 1325–1361 MHz, 20 ms 평균 371–382 W, 한도 400 W)였고, A1·A2 창(128/128)은 cap 밖(1410 MHz)이었다. ρ의 분자만 이 클럭 하락에 노출됐다."
- **(A)** "이 Job(VESSL A100-SXM4-80GB `GPU-3be4c7da…`, 기본 power limit 400 W)의 E-α 8셀에서, 지속 prefill backlog와 동거한 decode step 시간(S-per 중앙값)은 같은 ctx의 sticky ON 단독보다 ln ρ_off 0.120–0.164(ρ ≈ 1.13–1.18, 셀 평균 1.15–1.17)만큼 길었다. 이 값은 동거와 그에 동반된 전력 한도 클럭 하락(C 창 64/64 cap 노출)의 합성값이며, 두 성분은 분리되지 않았다."
- **(B)** "SM 클럭에 대한 탄성 ≤ 1과 HBM 클럭 불변을 가정하고(메모리 클럭은 미기록), 창별 실측 클럭 하락을 전부 빼도 64/64 짝의 ln ρ는 ln 1.03을 넘었다(최소 0.069). `COLOC_SLOWER_GE3`의 존재는 클럭 성분만으로 설명되지 않는다. 이 문장은 크기 진술이 아니다."
- **(C)** "C 창의 증가분은 호스트 간격(중앙 +0.29 ms)이 아니라 decode 발사부터 decode stream 동기화까지의 구간에 있었다. prefill 발사 iteration과 완료 대기 iteration의 S-per는 같았다(ln 비 중앙 −0.0004). 동거 창의 decode bs는 W_co 전체에서 b였다(S2). 8셀 안에서 D·b·c 주변 ln ρ 차는 모두 ln 1.03 미만이었다(기술값, 등록 판정량 아님)."
- **(D)** "측정 성립 E-α 복제 64개 중 0개에서 서명 JOINED가 나왔다(S3 736/736 = b + n_done). `max_new_tokens = 1` co-load 요청이 decode iteration에 참여하지 않았다는 코드 독해와 일치한다." / "`DS3_MISMATCH_OBSERVED_8` — E-α 8셀 밖의 동거 대표성은 미검증이다." / "`NEG_CONTROL_EA_HELD`(n 64, OFF A1·A2 idx5 최솟값 1.0)."
- **(K)** "K·C·κ\* 요구는 rev14 등록값 그대로이며 노출 뒤에도 바뀌지 않았다(N2C-87·N2C-98 병기)."

## 7. 금지 문안 (신규 32–42, §16-3 1–31 승계)
32 "ρ ≈ 1.16은 동거 간섭의 크기다", "green context 분할은 decode에 약 16%의 간섭을 남긴다"를 cap 병기 없이; "공유 메모리 계층 간섭이 16%(또는 10.5%)다". 33 "클럭 성분은 ln ρ의 1/3", "전력 무관 간섭은 ρ ≈ 1.105", (B)의 0.069·0.100을 크기로. 34 "power cap은 측정 오염/제거하면 ρ는 X", "운영점에서 동거는 항상 power cap을 유발". 35 "ρ는 D·b·c(또는 prefill 강도)에 무관/불변"을 발견으로, "곱셈성과 D 무관성이 서로를 뒷받침". 36 "prefill 진행량이 5배 달라도 ρ는 같다", "prefill 강도와 무관". 37 "decode에 SM을 더 줘도 동거 감속은 안 줄어든다/준다", ρ에서 static·d_min·정책·재배분 경계 도출 일체, 굶김 연쇄·HE0 원인 인용. 38 "단독 프로파일 × 1.16(× 1.105) 보정", "ρ 보정 예측기". 39 "cap 밖 동거 iteration에서도 ρ가 같으므로 클럭 성분은 0". 40 "호스트 오버헤드는 ρ와 무관"을 일반형으로. 41 이 Job의 ρ·클럭·전력 수치를 Job-PROFILE·다른 VESSL Job·KISTI 수치와 비교·결합·비율화. 42 "`-lgc`로 클럭 고정 대조"를 VESSL 실행 계획으로.

## 8. 새 caveat (문자 그대로)
- **N2C-102** "Job-EA의 C 창(64/64)은 GPU SW power cap 아래(SM 1325–1361 MHz, 20 ms 평균 371–382 W, 한도 400 W)였고 A1·A2 창(128/128)은 cap 밖(1410 MHz)이었다. 이 노출은 외부 요인이 아니라 동거 부하 자신이 만든 것이며 처치와 완전히 공변한다. ρ는 '동거 + 전력 한도 클럭 하락'의 합성값이다. 다른 power limit·다른 기판으로 이식하지 않고, 전력과 무관한 간섭의 크기로 쓰지 않는다. N2C-20(boot 간 열·클럭)은 이 창 간·boot 안 노출을 다루지 않는다."
- **N2C-103** "클럭 성분의 상한(창별 ln(1410/f) ≤ 0.062)과 그로부터의 라벨 견고성(보정 후 최소 ln ρ 0.069)은 SM 클럭 탄성 ≤ 1과 HBM 클럭 불변을 가정한다. 메모리 클럭은 기록되지 않았고, NVML 버퍼 SM 클럭 표본(`sm_clk_samples.csv`)은 0행이었으며 미지원 사유도 남지 않았다. 클럭 정렬 해상도는 100 ms다. 클럭 성분의 하한과 크기는 미정이다."
- **N2C-104** "E-α 8셀의 D·b·c 주변 ln ρ 차가 모두 ln 1.03 미만이라는 것은 사후 기술값이다. 등록된 동등성 판정량이 아니고, D는 {16, 44} 두 수준뿐이며, C 창 클럭 하락이 셀 간에 거의 같아 공통 cap이 무관성의 일부를 만들 수 있다. '증가분 ∝ A_on(곱셈성)'은 같은 관측의 재진술이지 독립 증거가 아니다."
- **N2C-105** "동거 iteration당 prefill 진행량(약 130–630 tok/iter)의 차이는 주로 decode iteration 길이에서 온다. 시간 단위 prefill 처리율 차는 약 1.4배이며 D(prefill 64 대 92 SM)와 완전히 공변한다. prefill 강도는 이 Job에서 D와 별개의 축으로 처치되지 않았다."
- **N2C-106** "C 창 중 최근접 클럭 표본이 1410 MHz인 iteration(전체의 0.5%, 복제당 5–10개)에서의 S-per 비교는 사후 진단이다. 100 ms 정렬 오차와 cap 전이 구간 선택 편향이 있어 클럭 성분의 분리 근거가 아니다."
- **N2C-107** "E-α 동거는 8192-token co-load를 연속으로 쌓은 지속 backlog로, 전력 소모를 최대화하는 부하다. 운영 부하(간헐 prefill, 낮은 rate)에서 동거와 cap이 동반되는 비율은 이 Job에서 측정되지 않았다. '운영점에서 동거는 항상 cap을 동반한다'도 '운영점에서는 cap이 없으니 ρ의 클럭 성분은 빠진다'도 쓰지 않는다."
- **N2C-108** "F-N2-3 `HELD`는 방향(ρ_on > 1) 예보의 적중이다. 설계 예보 대역 [1.3, 1.6](§17-14)은 크기에서 빗나갔다(실측 1.14–1.17). '예보가 맞았다'를 크기 적중으로 쓰지 않는다."

## 9. 교훈 후보 (doc-steward 회부)
1 기판 잡음으로 분류한 변수도 처치와 완전히 공변할 수 있다 — 기판 telemetry는 판정량의 분자·분모 창 단위로 정렬해 공변 여부부터 검사. 2 같은 관측의 재진술을 두 번째 증거로 세지 마라(교훈 9 계열). 3 iteration 단위 강도는 iteration 길이와 섞인다 — 시간 단위로, 처치 축 공변 먼저. 4 처방 실현가능성은 기존 프로브 산출물부터 대조(게이트 #113 재확인).

## 10. 후속 측정 (처방 아님)
VESSL 컨테이너는 `-pl`·`-lgc`·`-rgc` 불가(실측) → 동일 클럭 대조는 전력 정합 방식만. F-1 같은 boot 안 전력 정합 대조 arm(C_burn: decode 단독 + 보완 분할 레지스터 상주 FMA burner로 C 창 클럭 ±1%·cap ≥0.94 정합; ln(C_burn/A_on) = 전력·클럭 성분, ln(C/C_burn) = prefill 고유 나머지; 실현 조건 미확인: 서버 프로세스 안 green-ctx stream burner 주입 경로, burner가 ≈376 W 도달 가능성; 여전히 성분 측정). F-2 다음 Job에서 `clocks.mem`·NVML 버퍼 SM 클럭 수집(또는 사유) 확인. F-3 운영 rate trace 부하에서 동거/단독 S-dev 비·창별 cap 노출(DS-3 운영 대표성, N2C-107을 닫는 유일 경로, 별도 사전등록). F-4 무관성 주장은 b ∈ {1,48}, c 512, D24/34 확장 + 등록 동등성 라벨로만. 기판 변경(VM Cluster H100)은 새 캠페인, 이 Job과 비교 금지.

## 11. 불변 선언
게이트 #6, λ0R-8(iii), HE0, layer-type 死, 정책 순위, Claim D/E(미검증), 기여 2 "가능한 경계" 등급, 기여2→3 인과 연결 규칙, gate #13/#16, C2 인용정지 불변. 새 성능 판정·정책 판정 0건. N2(3-a)는 완결되지 않았다(N2C-10).
