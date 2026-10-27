# 논문 positioning — A+B만 vs A+B+C (venue-strategist, 2026-10-27)

> 읽기·웹 조사만, 새 측정 0. 짝 감사: `workspace/engine-port/results/anchor_audit/VERDICT_claimC_online_predictive_2026-10-27.md`.
> A = 비용 모델로 워크로드별 static 선택 + oracle 후회(V-O) · B = phase 경계 여지(H-D) · C = 비용 모델 구동 iteration 단위 예측형 SM 배분(hybrid 귀속).

## 요지
- 현재 A·B 측정값 0(V-O·H-D 동결, 결정 38). "A+B 논문"의 실체는 비용 항 특성화 + 기존 확정 결론뿐.
- 사용자 우려("C 없으면 빈약")는 타당. 단 C의 형태는 **DuetServe(ICML'26, attention-aware roofline으로 iteration 지연 예측 → SM 분할)** 가 transformer에서 이미 수행 → C의 신규성은 **hybrid 귀속**에만 달림. CM-1 hybrid 고유 항은 미입증.
- 웹 검색상 hybrid SM 분할 PD 스케줄러는 부재(빈자리 ≠ 자명하지 않음).

## 선행 (웹 확인)
| 선행 | venue | adaptive | 비용 모델 | hybrid | 관계 |
|---|---|---|---|---|---|
| Bullet 2504.19516 | **미확인**(기존 문서 ASPLOS'26 재확인 필요) | 예 | 예 | 아니오 | C 원형 |
| MuxWise/Drift 2504.14489 | ASPLOS'26 | 예(공개 코드는 문턱표) | 예 | 아니오 | 최근접 |
| **DuetServe 2511.04791** | **ICML'26 (PMLR v306)** | 예 | roofline 예측 | 아니오 | **C와 형태 동일, 필수 baseline** |
| Nexus 2507.06608 · semi-PD 2504.19867 · DynaServe 2504.09285 | 미확인 | 예 | 예/불명/부분 | 아니오 | 동계열 |
| DistServe · Splitwise · Sarathi-Serve | OSDI'24 · ISCA'24 · OSDI'24 | 정적 | 시뮬/프로파일 | 아니오 | static baseline |
| **Vidur 2405.05465** | MLSys'24 | 아니오 | 예측 모델+Search | 아니오 | **A와 형식 동일** |
| SLIM 2607.29575 | 미확인 | 아니오 | semi-analytical | 아니오 | A 동계열 |
| Marconi 2411.19379 | MLSys'25 | 예(캐시) | 예 | **예** | hybrid 서빙 선례 |
| SQD 2609.13134 | preprint | 아니오 | 분석 | **예** | AMD 문서 |
| DUET 2603.15530 | DAC'26(재확인 필요) | HW | roofline | 예 | — |
| Elastic KV 2608.23658 | preprint | — | — | 아니오 | 순수 negative, 미게재 |

## 학회 fit
| 학회 | A+B만 | C: hybrid 귀속 승리 | C: negative(충실 재현 동반) |
|---|---|---|---|
| MLSys | 약~중 | **강** | 중 |
| EuroSys / SIGOPS ATC | 약 | 중~강 | 약~중 |
| OSDI/SOSP | 약 | 약~중 | 약 |
| ASPLOS | 약 | 중(Nsight 필요) | 약 |
| ISCA/MICRO/HPCA · SC | 약 | 약 | 약 |
| ISPASS/IISWC | **중~강** | 중 | 중~강 |
| 워크숍 | 강 | — | 강 |

선례: 특성화가 top에 붙을 땐 constructive 장치 동반(POLCA ASPLOS'24, Vidur MLSys'24). 순수 negative(Elastic KV)는 미게재. ⇒ **모델이 결정을 바꾼다는 증거(A의 후회 측정)** 가 최소 장치.

마감(2026-10-27): MLSys 2027 10-30(불가) · OSDI'27 12-01/12-08(C 불가) · EuroSys'27 가을·ASPLOS'27 지남 · ATC'27·ISCA'27·ISPASS'27 CFP 미공개(추정만).

## C가 negative일 때
가치: DuetServe식 예측형 최소-SM이 hybrid로 이득을 옮기지 못함 + HE0 범위를 예측형까지 확장. **조건: 같은 하네스에서 순수 attention 모델로는 예측형이 static을 이긴다는 재현.** 위험: lockstep 기판(결정 F4) 아티팩트 반박 · "결정만 다르고 성능 같다"는 A+B 강도 · hold-out 서빙 비용 불가피.

## 리뷰어 비판 상위 5 + 최소 실험
A+B만: ① Vidur-Search와 같다 → V-O 소거 변형(ctx 항 0, s=0, ρ=1)별 후회로 hybrid 항이 결정을 바꾸는지 ② 기판·모델 혼합 → Nano에서 HE0 핵심 셀 재측정 또는 절 분리 ③ untuned fused·Sarathi/DuetServe 부재 → G0 tuned fused·chunked-prefill arm ④ 동적 패배 = 기판 아티팩트 → H-D + Bullet §4.5 방향 프레이밍 ⑤ 변화 trace 대표성 → 실 trace 재생 1종.
A+B+C: ① 허수아비 generic → P_gen+recal(Bullet 두 점+online 재보정) + DuetServe roofline판, 순수 attention에서 이기는 재현 ② hybrid 귀속 → hold-out + 조성 스윕 a4–a31 단조성 ③ 노이즈 → n≥4·CI·≥3% ④ "disaggregation 쓰면 된다" → GPU 수 맞춘 2-GPU PD 분리 1점 ⑤ 오버헤드·lockstep → 컨트롤러 구동 전환 경로 측정(CM-3 빈칸) + Nsight timeline.

## 권고
1. H-D(값쌈) → 2. V-O(A) → 3. C 게이트: H-D 여지 > 3% 또는 V-O phase별 oracle 분기 시 착수, 첫 마일스톤 = 순수 attention 재현.
투고: A+B+C로 SIGOPS ATC'27 또는 EuroSys'28 봄(날짜 미검증), 대안 ISPASS/IISWC(A+B)·MLSys 2028.
리스크: DuetServe·Bullet 대비 증분 공격 · hybrid 고유 항 미성립 시 A 차별성 동반 약화(CM-5 cap≡슬롯은 할당기 종속) · PD-mux 기전 미귀속.
금지: "C가 이긴다" · "hybrid가 예측기를 틀리게 만든다" · MuxWise 2.20× 등 선행 수치 배율 비교.
정정 후보(promote 몫): `reports/paper/venue_positioning.md` — Bullet venue 미확인, DuetServe ICML'26 확인.

Sources: arxiv.org/abs/2504.19516 · arxiv.org/html/2504.14489v1 · arxiv.org/abs/2511.04791v1 · proceedings.mlr.press/v306/gao26m.html · arxiv.org/html/2507.06608v5 · arxiv.org/abs/2504.19867 · arxiv.org/html/2504.09285v2 · arxiv.org/pdf/2405.05465 · arxiv.org/abs/2607.29575 · proceedings.mlsys.org (Marconi 2025) · arxiv.org/abs/2609.13134 · arxiv.org/abs/2603.15530 · arxiv.org/abs/2507.12442 · arxiv.org/abs/2608.23658 · mldeadlines.com/conference/mlsys-2027 · usenix.org/node/321091 · asplos-conference.org/?p=8512 · atc.sigops.org/2026/cfp.html
