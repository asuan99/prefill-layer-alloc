# claims-auditor 판정 2 — 보고용 "핵심 기여 3개" 카드 (2026-09-28, 읽기 전용)

대상: `handoff-report/artifacts_2026-09-28/pdmux_roadmap_v17.html` 6절의 **직전 판(v15)** 및 검토용 v4 3절. 판정 뒤 교체 문구가 v16/v17·v5/v6에 반영됐다.

## 사실 오류 4건(전부 교정 완료)
1. **"KV가 작아 배치 여유가 크다" REFUTED** — CONS:2593 "'hybrid는 KV가 적다'는 통념은 이 격자에서 거짓"(Ha8 601 MiB > T8 70 MiB, 8.6×). 구속 자원은 `max_mamba_cache_size = max_running_requests = 48`(SSM 상태 풀), KV는 6.6× 과공급(PROJECT_STATUS:1573-1578, CONS:325, §3 항목23). 3-b에는 transformer 비교군 없음. → 문장 삭제.
2. **"검증된 프로파일이 없으면 균등" REFUTED(오역)** — 정본 문구는 "agnostic" = decode 배치 크기 문턱 표 조회(upstream `temporary demo`, REPORT:337/390), 균등 분할 아님. → 세 곳 교정.
3. **"운영 규칙을 4-c로 4모델에서 확인" REFUTED** — 4-c는 PD-mux vs fused(C1)이며 고정 배분 규칙을 시험하지 않음. 검토용 ⑥ 행("3-b, 5-a")과도 모순. → 삭제.
4. **"prefill에서 두 층의 SM 민감도가 같다(측정됨)" REFUTED(표현)** — L=256 [1.24,1.34], L=512 1.198, ≈1.0은 L≥1024뿐(CONS:3441); no-cudagraph·triton·Zamba2-2.7B·n_indep=1·UNAUDITED, Nemotron-H 이전 금지. "나눌 이유 없음"의 근거는 민감도가 아니라 서빙 실측(42→124 ms)과 `R_policy≈1.03`. → 근거를 서빙 실측으로 교체.

## 과장(교정 완료)
- decode 최소 SM이 "attention 층 때문에 움직인다"는 단정 → 질문형(3-a). 반대 증거: CONS:2592-2593(C2, 8B 4 arm에서 SM 민감도 모델 무관, 스텝 트래픽 68–94 %가 weight-sweep), CONS:3443(§1-5 HE2 decode non-binding), "attn 비중 5→79 %" 인용 금지(MEMO B1). 원인 귀속엔 모델 hold-out 필요(2–4단계에 없음).
- "처음으로 하나의 비용 지도" → 삭제. VP:111(SSM characterization arXiv 2507.12442), VP:115 "부분적이나 실재". prefill 근거(KISTI·Zamba2·graph OFF)와 decode 근거(VESSL·Nano-9B·graph ON)를 한 지도로 묶으면 게이트 1 위반. "다른 GPU로 이전"은 기판 주의와 충돌.
- 기여 2 제목 "작동하는 경계" → "가능한 경계"(재배분이 이득을 낸 측정 없음, HE0는 반대). 층 종류 재배분 "확정"에 "현 구현 범위 한정"(CEM:585) · 반응형 "우리가 시험한 구조" 한정(Bullet·DuetServe는 반대 주장, VP:115-123) · 동거 측정 출처는 4-a만(4-b는 계기, RR D-1) + "hybrid에서" 한정(dense 선행 M4).
- 기여 3 "구성 정보만으로 초기 배분" → 확장으로 강등. B6는 전체 모델 프로파일 기반 dynamic(EXPERIMENT_ROADMAP:1477), 구성 기반 비용 모델은 별도 사전등록 확장(MEMO §2 말미), hold-out 없이는 식별 불가(RR §0-5). 5-a 격차 <3 %면 5-b 미실행이라 두 결론 모두 부재(RR §3 P11).
- 맺음 "2–4단계로 완성" → "완성될 예정(2단계 통과 조건)". 기여 1의 원인 귀속 부분은 2–4단계로 완성되지 않음.

## B. 핵심 기여 선택
기여 1·2는 증거 구성(특성화 + 기전적 부정 결과 + 가이드, RR §2 경로 A, VP:491)과 맞음. **기여 3은 "모델 구조 기반 선택"으로 두면 심사 통과 어려움**(5-b 조건부·Claim E 미검증·MuxWise 워크로드별 표 선점). 권고 채택: **"운영 가이드와 그 여유의 크기"** — 확정된 운영 규칙 + 5-a 격차(= B2 oracle 상한, EXPERIMENT_ROADMAP:1473/1489; 어떤 선택기든 얻을 수 있는 이득의 상한이라 5-b 결과와 무관하게 긍정 산출물). 5-b는 확장 한 줄.

## C. 검토용 정합성(교정 완료)
- 매핑 REFUTED → "기여 1 = ①④ + 3-b(사실 기록), 기여 2 = ②③⑤⑧, 기여 3 = ⑥ + 5-a 여유 + ⑨(조건부), ⑦ 방법론 별도".
- ⑥ "확정(③에서 도출)"인데 계층은 "이번 실험으로 완성" → "이미 확보"로 이동. ⑤ 출처 "4-a, 4-b" → 4-a. ⑦ 13.64 %에 "등록 규약 a_r4" 병기(교훈 252·E2C-8′).
- ⑨/5-b 판정 기준: 미등록 [제안] arm을 기준에서 제외, "B1·B5 대비 paired CI 3 % 이상 + non-target 영역 >3 % regression 없음"(EXPERIMENT_ROADMAP:1486). B6 ≠ "구성만 쓰는 것".
- "[제안] 부하 인덱스 배분표 arm"은 **이미 측정한 agnostic 기준선과 같은 기제** → "agnostic 기준선(MuxWise 저자 upstream 기본 선택기, 저자 주석상 임시 데모) 108 SM 표 재현 arm"으로 개명.
- "Bullet Fig. 20a 각주 확인 필요" → 이미 판정됨(REPORT:422/427: 구성별 지속시간 타임라인, 집계값 아님).
- 대응표 "C1 (Claim A 계열)" 오기 → C1은 Claim A 아님(CEM:584).

## 범위 밖 관찰
- 보고용 H3-b 차트(8B decode SM 비 2.39–3.11×, 모델별 점추정)는 "순서·격차 주장 안 함(인용 정지)" 문구를 달고 있으나, transformer·pure SSM도 비슷한 민감도를 보이므로 기여 1을 "attention 층 때문"으로 되돌리면 이 차트와 충돌한다(RR V-char "8B C2 2.36–2.91× 인용정지").
