# 규칙층 경량 델타 감사 — PREREG_MULTIGPU rev10 §17 (커밋 9b4e28a), 2026-10-13

> 감사자: claims-auditor(읽기 전용, GPU 0). 요지를 메인 세션이 저장(caveat 문안 원문 그대로).

## 0. 결론
M2g `GO-with-caveats`(변경 없음), **M3 `GO-with-caveats`**(死因 0, D10·D11 닫힘), 종합 `GO-with-caveats`. 실행 승인이며 "M3가 TP 스케일링을 판정할 수 있다"는 뜻이 아님(MGP-56). 다음 gate = 하네스층 H-1–H-5.

## 1. 처방 등록 확인
P10(§17-1 문언 일치, shape별 상태 전수, BROKEN > UNTESTED > HELD) → D10 닫힘. P11(a) 상수·결과·dry-run 불일치 시 봉인 불가 → 닫힘. P11(b) 등록, 덧붙은 비절단 문장은 P9-c 재서술 → D11 닫힘. H-6 충족.

## 2. P11(a) 재계산
Python 3.13.9로 J2·J4·J8 모두 ['DP','TP'] 일치. 하네스 `mg_plan.py config-order`도 DP→TP 출력(JOB_CONFIGS 초기 순서 TP 먼저). 2원소 shuffle은 RNG로 swap 여부만 결정 — 초기 리스트 [DP,TP]면 반대, 정수 seed면 swap 없음. 직전 예보(TP 먼저)와 달라도 판정 영향 없음(MGP-49′ 중립 문언, 死因은 미등록이었음, 이제 관측 전 고정 상수, 초기 순서 의존은 dry-run 대조가 잡음).

## 3. [DP,TP] × S3 절단 (m3-forecast 모형 예보)
J2: TP2.boot1ext 3–4·boot2 4 잘림 → G_TP(2) 미정의. J4: TP4.boot1ext 1–2·boot2 4 잘림 → 미정의. J8: TP8.boot2 1–4 잘림 → 정의 불확실. DP는 모든 calib에서 완주, S1·S2는 절단 없음. S3에서 TP 점 {1} 또는 {1,8} → TREND_UNDEFINED, DP 라벨은 P-DP 판정에만. 반대 순서면 DP 잘림 → P-DP UNTESTED → P11(b) 전단으로 TP 인용 불가. **S3에서는 어느 순서든 TP 방향 인용 불가**, [DP,TP]가 P-DP 판정 가능이라 정보상 우월 → caveat.

## 4. 반전 시험
config 순서(고정 상수, 반사실 [TP,DP]도 TP 인용 불가 동일), P10 BROKEN vs MGP-7 크기 규칙(우선순위 조항이 문자 그대로 읽기 강제, fail-closed), 절단으로 점 빠졌으나 추세 정의(대체값 없음, N_max 불변) — 반전 없음.

## 5. caveat (문자 그대로 승계)
- **MGP-56**: "rev10 등록 순서(J2·J4·J8 = [DP, TP])에서 공동 확장(S3)이 일어나면 예보상 TP2·TP4의 LO가 전부 잘리고 TP8 LO도 일부 잘려 TP 추세는 `TREND_UNDEFINED`다. 이 경우 이 캠페인은 TP 스케일링 방향을 내지 못하며, DP 라벨은 P-DP 판정에만 쓴다. 'TP 추세 없음'·'TP가 확장되지 않는다'로 쓰지 않는다."
- **MGP-57**: "P10에 따라 크기 3% 미만의 DP 방향 라벨도 그 shape를 BROKEN으로 만들고 MGP-13을 발화시킨다(fail-closed). 이 경우 'DP 용량비가 N에 따라 변한다'로 쓰지 않는다. 쓸 수 있는 문장은 'P-DP 예측이 등록 규칙상 기각되어 TP 라벨 인용이 정지됐다'뿐이다."
- **MGP-58**: "`TRUNCATED` 점이 빠진 상태로 정의된 추세(TP·DP 모두)는 빠진 N 목록과 절단 사유를 라벨 옆에 병기한다. 잘리지 않은 등록 격자 전체에서 추세가 같았을 것이라고 쓰지 않는다."
- 승계: MGP-43′·49′·50′·51·52·53·54·55.

## 6. 편집 지적
E-1 §16-2.2 "DP2·DP8 LO 잘림" 문장과 §17-2 충돌 → 취소선 + 우선순위 §17 > §16. E-2 §15-1 "TP 먼저" 취소선. E-3 하네스 `PREREG_REV = 10`, mg_plan 머리말 rev 표기. (E-1·E-2 메인 세션 반영 완료)

## 7. 다음 gate
하네스층 H-1–H-5 변이 KILLED + 무변이 대조, H-2에 "JOB_CONFIGS 초기 순서 [DP,TP]" 변이본(dry-run 대조에서 KILLED) 추가. H-7 권고.
