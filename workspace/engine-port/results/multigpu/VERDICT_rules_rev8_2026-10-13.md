# 규칙층 델타 감사 — PREREG_MULTIGPU rev8 §15 (구성별 적정 크기 Job), 커밋 5152b33, 2026-10-13

> 감사자: claims-auditor(읽기 전용, GPU 0). 요지를 메인 세션이 저장(caveat·필수 조건 문안은 원문 그대로).

## 0. 결론
| 층 | 등급 | 死因 |
|---|---|---|
| M2g(rev8 배분) | `GO-with-caveats` | 0 (MGP-44·47·48 승계) |
| M3(rev8 Job 효과 처리) | **`NO-GO`** | **1 (D9, N2)** — 처방은 등록 문장 3개 + J1 캡 상수 1개, 설계 변경 불필요 |
| 종합 | **`MIXED`** | |

## 1. D9 (N2): G(1) 미정의 시 교차 Job 추세가 안전장치 없이 인용 가능
rev8에서 G\_sym(1)(J1)이 Job 효과 검사 기준·크기 규칙 분모·추세 N=1 점을 모두 맡는데, §4-4의 rev1 문장("G(1) 미정의면 방향 라벨만, MAGNITUDE_UNDEFINED")이 남아 {2,4,8}이 서로 다른 Job인 rev8에서 Job 효과 검사·3% 크기 규칙·TREND_JOB_SENSITIVE가 모두 꺼진 채 방향 라벨 인용 가능.
도달 경로: (1) J1 하드캡 — S3 공동 확장 시 J1 8.68–9.16 h > 8 h, N1 LO fused_sym NOT_ATTEMPTED → G\_sym(1) 미정의. (2) J1 CPU quota 절벽 — x1 cpu.max 11 코어, N3 r3·r4 x1 91셀 Δnr_throttled/Δnr_periods 중앙 0.63%·최대 1.61%·1% 초과 3셀(평균 1.3 코어), x8 M2g 2회차 0.028% → HARNESS_LIMITED 문턱 바로 위. (3) J1 기동·인프라 실패.
반전: λ0 seed 범위/평균 A 0.54%, Job 간 λ\* 차 1.95%. G(1) 미정의 시 DP {2,4,8}에 J8만 +2% → 기울기 ≈0.0099/doubling, seed bootstrap CI(Job 분산 미포함) 0 초과 → G_INCREASES 인용 가능 → P-DP 깨짐·MGP-13 발화(원인 Job 효과 단독). G(1) 정의 시 2% < 3%라 미인용.
같은 계열: (b) 보정 추세 크기 규칙 미등록(δ 3.5%, 원 1.05 vs 보정 1.0145 → 인용 반전), (c) TREND_JOB_SENSITIVE에서 P-DP 상태 미등록.
**처방**: P9-a G\_sym(1, shape) 미정의면 그 shape 모든 N-추세 방향 라벨 인용 불가(`G1_UNDEFINED_NO_JOB_CHECK`), 모든 J_N `JOB_EFFECT_UNTESTED`, P-DP `P_DP_UNTESTED`. P9-b 플래그 발화 시 원·보정 추세 둘 다 크기 규칙 통과 + 라벨 동일일 때만 방향 인용. P9-c P-DP가 TREND_JOB_SENSITIVE·TREND_UNDEFINED·G(1) 미정의면 P_DP_UNTESTED, MGP-13 처리 한 문장 등록. P9-d J1 하드캡 ≥ 10 h.

## 2. 반전 표 요지
G(1) fallback·J1 캡 8→10 h·보정 추세 크기 규칙·TREND_JOB_SENSITIVE P-DP: 반전. 앵커 문턱 3%(seed 0.54%라 해상력 충분)·앵커 rung·bootstrap 방식·shape B 보정·G0(ii) 교차 적용: 미반전(원자료 없음 포함). config 순서 문장 모순("TP 먼저" vs "무작위"): 정의역만 변화. 기판 gate `"A100" in name`은 PCIe·40GB 구별 못함(`substrate_probe_helper.py:245`).

## 3. 중점 답 요지
- G\_sym은 arm 대칭 곱셈 성분만 상쇄. 비대칭 기전 실재(SW power cap 노출 부하 의존: N3 7.5%→47%, λ0 A 27.2%·B 49.4%; quota ∝ N).
- GPU0 단독 앵커는 노드 비대표(M2g 2회차 GPU별 throttle 표본 GPU0 5085 vs GPU5 28). SKU 보증 없음. PCIe/NVLink 혼합은 confound #11.
- M2g Job별 G1-ctl·G2 바닥은 개선. G0(ii) 교차 적용은 코드 근거, TP4·8 다중 rank 종료 미시험.
- §15-5 정정: MGH-20 해당 없음 맞음. **MGH-7·MGP-14 "해당 없음"은 틀림**(S3에서 전 Job 캡 초과, DP8 LO 4/4 절단) → MGP-49로 대체, MGP-14 존속. **MGH-6 부분 존속**(미보정 모형). C-1 공변량에 quota·SKU·NUMA 누락. C-9 "배제" 불충족(MGP-42). 2회차 앵커·NUMA 폐지 → MGP-45.
- 예산: S1 audited 120.0 GPU-h·최장 J2 8.76 h, VESSL 보정 123.6–129.5 GPU-h·9.0–9.4 h, S2 132.9–142.5 GPU-h·9.8–10.5 h, S3 222–237 GPU-h(전 Job 캡 초과). M2g Job당 4 h 여유.
- **MGH-22 승계**: fused_sym↔agnostic만으로 TP4 1.34%·TP8 1.29% > 1% → G\_sym(TP4/8) 미정의 예보, TP 1차 추세 {1,2}만 → TREND_UNDEFINED, J4·J8 TP 절반(약 44 GPU-h) 기술치만. 규칙 변경은 사후 변경(사용자 결정·델타 감사·공시) — 감사자는 처방하지 않음.

## 4. confound: #11 기판 혼합(핵심, N과 Job·노드·quota·테넌시·SKU 완전 별칭), #10, #5(HARNESS_LIMITED 1%가 x1 분포 안), #7, #3(N마다 Job n=1).

## 5. caveat (문자 그대로 승계)
- **MGP-42**: "rev8에서 N과 Job(노드·시점·CPU quota·테넌시)은 완전히 별칭이다. §15-3 앵커는 Job 효과 중 'GPU 0 단독 N=1 성분'만 검사하며 GPU 간 통신·노드 내 다른 GPU·N에 비례하는 quota와 Job의 상호작용은 검사하지 않는다. `JOB_EFFECT` 미발화를 'Job 효과 없음'·'N 간 G 차이는 N 효과'·'C-9의 Job 효과 배제 충족'으로 쓰지 않는다; 미발화의 뜻은 'N=1 GPU0에서 G\_sym·λ\*(fused\_sym, A) 차이 ≤ 3%'뿐이다."
- **MGP-43**: "VESSL Job의 CPU quota는 GPU 수에 비례한다(x1 실측 11 코어, x8 실측 88 코어, x2·x4 미실측). J1의 G(1)은 11 코어, J\_N 앵커는 11N 코어에서 잰 N=1 값이다. x1 셀 throttle 비율은 N3 r3·r4 91셀에서 중앙 0.63%·최대 1.61%·1% 초과 3셀이고 x8 Job 전 구간은 0.028%다. J1 셀의 `HARNESS_LIMITED`를 'GPU 용량 한계'로, J1↔앵커 차이를 순수 'Job 효과'로 쓰지 않는다."
- **MGP-44**: "B0 gate는 GPU 이름의 'A100' 포함·cc 8.0·108 SM만 본다(A100-PCIe·40GB도 통과). 각 Job의 GPU 이름·power limit·`topo -m`이 J8과 같지 않은 Job의 점을 같은 N-곡선에 잇지 않는다. `TOPOLOGY_NOT_NVLINK` Job의 TP 점은 NVLink Job과 같은 추세에 넣지 않는다."
- **MGP-45**: "rev8은 N=1 2회차 앵커와 NUMA 블록을 없앴다. 앵커는 각 Job 맨 앞 1회이고 config는 그 뒤 최대 약 9 h(모형) 동안 측정된다. 앵커 보정 추세는 Job 안 정상성 가정 위의 값이며 Job 안 드리프트와 NUMA 교차 효과는 측정되지 않았다. 같은 Job 안 TP↔DP 차이를 mode 효과로 쓰지 않는다."
- **MGP-46**: "앵커는 shape A, arm {fused\_sym, agnostic}만 잰다. shape B 추세와 보조 라벨 G\_def·G\_d44,sym은 Job 효과가 검사되지 않은 값(`JOB_EFFECT_UNTESTED`)이며, A 앵커 비로 B·보조 비를 보정한 값은 '교차 shape/arm 가정 하 보정'으로만 쓴다."
- **MGP-47**: "J8 GPU 편차 블록은 동시 8-서버 부하 하 기술치이고 rev8에는 결과 분기(`GPU_HETEROGENEOUS`)가 없다. J2·J4 노드의 GPU 간 편차는 측정되지 않았다. 앵커(GPU 0)를 그 노드의 N=1 대표로 쓰지 않는다(C-2 재활성)."
- **MGP-48**: "G0(ii) 거부는 J2(TP=2)에서만 시험했다. TP=4·8 적용 근거는 `multigpu_guard.tp_pdmux_violations`가 `tp_size > 1`의 순수 함수라는 코드 독해이며 TP=4·8 다중 rank의 비정상 종료는 시험되지 않았다. 'TP 전 크기에서 금지 모드 거부 확인'으로 쓰지 않는다."
- **MGP-49**(MGH-6·MGH-7 대체, MGP-14 존속): "Job별 예보(λ=prior: J1 4.0–4.3 h, J2 8.8–9.4 h, J4 8.7–9.4 h, J8 7.9–8.6 h, 합 120–130 GPU-h)는 VESSL 실측 보정이 없는 `mg_plan` 모형값이다. 공동 확장(S3)에서는 J1 8.7–9.2 h > 8 h, J2 17.9–19.0 h·J4 16.2–17.2 h·J8 14.1–15.2 h > 14 h로 예보되어 N=1 LO·DP2·DP8 LO가 잘린다. '120 GPU-h로 끝난다', '모든 Job이 캡 안'으로 쓰지 않는다."
- **MGP-50**: "MGH-22 예보(TP4·TP8 fused\_sym↔agnostic `max_total_num_tokens` 상대차 1.34%·1.29%)가 rev8에서도 성립하면 G\_sym(TP4)·G\_sym(TP8)은 미정의이고 TP 1차 추세는 `TREND_UNDEFINED`다. 이 경우 J4·J8 TP 측정은 기술치만 낸다."

## 6. 하네스 필수 조건
1. P9-a–d 반영. 2. Job 효과 기준 = J1 boot1(확장 전 1.6× rung) G\_sym·λ\*(fused_sym, A); 앵커가 HI_SATURATED 아님·HARNESS_LIMITED·CLIENT_ERROR면 JOB_EFFECT_UNTESTED. 3. 보정 추세 bootstrap은 앵커 seed 색인 공동 재표집. 4. config 순서 모순 해소·로그 기록. 5. 기판 gate: `--expected-gpus` = N, 이름 정확히 "NVIDIA A100-SXM4-80GB", power limit 400 W, N≥2면 전 쌍 NV#, 어긋나면 ABORT_SUBSTRATE(최소 MGP-44대로 곡선 제외). 6. Job별 cpu.max·셀별 throttle 공변량 병기. 7. PREREG_REV 8, Job별 HARD_CAP_S(J1 ≥ 10 h; J2·J4·J8 사용자 결정 — S3 덮으려면 약 19/17.2/15.2 h, 아니면 MGP-49), Job별 EXPECTED_GPUS, anchor2·numa·nsys 제거·앵커 2 boot 추가, GPU 편차 블록 위치·분기 부재 등록. 8. 변이 KILLED: (i) G(1) 미정의 방향 인용 (ii) 확장 rung과 앵커 비교 (iii) 다른 Job 앵커로 보정 (iv) 보정 추세 크기 규칙 제거 (v) 이름 부분 일치 gate. 9. M3 J_N은 자기 PASS 표와 J2 표 3자 대조, J2 표 없으면 TP NOT_TESTED·DP 진행. 10. M2g Job별 N=1 앵커 boot에 G1-ctl용 fused·agnostic·d44 N=1 arm 모두 포함.
선택(사용자 결정): J1을 x2 스펙에서 GPU0만 사용(quota·노드 종류를 J2와 맞춤, 약 +4 GPU-h).

## 7. 반증 실패 공시
앵커 문턱 3%·arm 대칭 상쇄·G0(ii) 교차 적용·M2g 배분은 "통과"가 아니라 "원자료 없음". 비율 수준 교차 Job 효과는 GPU가 다른 x1 Job ≥3개에서 N=1 fused_sym·agnostic shape A HI 4 seed씩 재야 판정 가능.
