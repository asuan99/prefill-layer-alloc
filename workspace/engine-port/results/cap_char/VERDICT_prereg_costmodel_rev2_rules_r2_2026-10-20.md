# VERDICT — PREREG_COSTMODEL_TERMS rev2 규칙층 감사 2회차 (claims-auditor, 2026-10-20) = **NO-GO, 감사 상한 도달 → 사용자 결정**

diff 범위만 봤다. 수치는 CPU로 다시 계산했다(풀 33.04 GB, s 143.5 MB, k 16,379 B). [CS-OK] 33.04는 풀 GB 값(우연 일치)

## 1회차 항목
- **死因 1(T4 예약 항): RESOLVED.** 비율 0.506 / 0.684 / 0.800 / 0.889가 재현됐다. C\_B {118, 93, 68, 24}, n\_B {79, 78, 64, 24}, Llama {32, 12}도 맞다.
  - caveat: n\_A(cap) 식의 형식이 B와 비대칭이다. cap 48·64k 칸이 27→24가 되지만 라벨은 바뀌지 않는다.
- **shape C 60 대 83: 83으로 해소.** "60"은 1회차 감사서가 출력을 두 번 센 오류다. 다만 §6-2의 "어느 읽기든 KV가 cap\_V보다 먼저 구속한다"는 거짓이다. R\_자연이 4,096에 가까우면 둘이 같아진다(101 대 101).
- **死因 2(arm별 ±2 rung): RESOLVED.** 사다리 범위 ±0.365가 V\_COARSE 경계 0.223보다 넓어 오판 구간이 비었다. 다만 "보정 예측도 ±3% 안에서 이동한다"는 문장은 틀렸다. 이동량에 상한이 없고, 31%를 넘으면 P2가 사다리 밖으로 나간다. 이 문장은 삭제해야 한다.
- MAJOR 1 PARTIAL: N1 귀무 예측기가 bs=cap\_V(99–127)로 조회하는데 격자 최대가 96이라 항상 외삽된다. shape C 동시 수는 cap\_V가 아니라 KV 상한 약 83이어야 한다.
- MAJOR 2 PARTIAL: |ln| 중앙값은 위로 편향된다(n=2). shape C 두 Job의 앵커 보정 규칙이 등록되지 않았다.
- MAJOR 4 PARTIAL: 아래 死因 B 참조.
- MAJOR 7 RESOLVED(조건부): I-4 vanilla 경로 방출에 달려 있다.
- MAJOR 8 PARTIAL: q\_stall의 예측기 역할과 `SIGMA_NOT_COLLECTED`일 때의 σ 기본값이 없다.
- MAJOR 10 RESOLVED.
- 미등재 B PARTIAL: T2′로 대체할 때 σ′ = 동거/T2′를 등록하지 않으면 간섭을 두 번 센다.
- MINOR 전부 RESOLVED.

## 수리로 새로 생긴 死因 (교훈 276 패턴 재현)
- **死因 A: 예측 중심 사다리 × P1 `CLIFF_ADJACENT`(SP R9-2).** rung 0이 예측값이므로 예측이 정확할수록 a(rung0)≈0.90이 되어 CLIFF에 걸린다. 그래서 판정이 정확도에 대해 비단조적이다. 오차 0이면 UNRESOLVED, 8%면 PASS, 20%면 COARSE가 된다. 단위당 PASS 자격(Δa 0.2) 확률은 e=0에서 0.05, e=0.08에서 0.97이다. 근본 처방은 P1의 CLIFF 조건을 "보간 CI가 ±10% 안"으로 바꾸는 것이고, SP R9-2 이탈임을 공시한다.
- **死因 B: shape C의 f\_ext가 10%를 넘는 것이 사실상 확실하다.** bs 약 83의 padded 값은 88 또는 96이다. c̄≈9.1k인데 격자의 볼록껍질 c\_max는 bs 96에서 8,192다. 그러면 사전에 V\_PASS\_UNREACHABLE이 되고, min 집계 때문에 전체 V가 COARSE 이상으로 올라갈 수 없다(V\_PASS 정의역이 공집합). 원인은 cap 128 부팅의 KV 0.95 규칙이 shape C 점유를 격자에서 잘라낸 데 있다.
- 쟁점 14: ±2 rung 산술은 맞다. V\_PASS\_UNREACHABLE은 정직한 장치이지만 死因 B·min 집계와 결합하면 과잉이 된다. Δa의 출처(예측 곡선인지 실측인지)가 정해지지 않았다. 해상도 5%는 SP R9-6(3%) 이탈이며 공시되지 않았다.
- **J-A·J-B·J-C(항 측정) diff에는 死因이 없다.** 死因은 V 규칙층에만 있다.

## 반전 시험
| 자유 표면 | 민 범위 | 판정 변화 |
|---|---|---|
| 사다리 중심(rung 0 = 예측) | e 0 → 0.08 | UNRESOLVED → PASS |
| bs\_padded / 볼록껍질 | 64 → 96 | shape C PASS 가능 → 불가 |
| R\_자연(new\_token\_ratio) | 0 → 4,096 | KV 구속 → cap 구속 |
| n\_A 형식 | doc → 일관 | 변화 없음(caveat) |

## 사용자 결정 선택지
- **(a) 권고**: 항 측정(T1–T4, J-A/J-B/J-C)만 caveat를 붙여 봉인하고, V는 항 데이터를 받은 뒤 별도 PREREG로 새로 감사받는다. GPU 비용은 0이다. 데이터를 본 뒤 구조를 고르는 위험은 새 PREREG에서 공시한다.
- (b) 감사 없이 rev3를 승인한다. 내용: P1 CLIFF를 보간 CI ⊂ ±10%로 교체, 라벨은 셀별로 보고, N1 bs를 96으로 클램프, shape C용 T1 셀을 cap 96 부팅으로 추가(약 +1 GPU-h).
- (c) V를 기술값으로 낮추고 상한을 V\_COARSE로 명시한다.

## 봉인 전 [미등록] 체크리스트
ENGINE\_DIGEST(G-α/β/γ 통과값)·config digest·분할 gate 문서 경로 · I-4·I-6 필드, T\_span\_iter, SPAN\_TOKENS, σ 엔진 필드 · σ 기본값, q\_stall 역할 · σ′ 정의, T2′ L 범위 밖 처리 · cap\_V(max\_new\_tokens, new\_token\_ratio 정의 시점, L̄\_자연=L̄\_in) · "KV 먼저 구속" 문장 정정, 83 · n\_A 형식 · Δa 출처, 3% 대 5% 공시 · 캡처 bs 목록, bs\_padded, N1 클램프 · shape C 앵커 보정 규칙, 부호 있는 중앙값 · "±3% 이동" 문장 삭제 · 기존 목록(X\_T1·O, m(L), 채우기 출력 길이, 외삽 수준, 소거 변형, O-9, 적합 기준, 실현 SM 허용, Llama 짝 셀, 하드캡, O-4·6·8·10·11).
