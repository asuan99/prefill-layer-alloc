# 실험 카드 필요·불필요 검토 + 논문 기여·기대 결과 정리 (2026-09-27, rev2 — claims-auditor 감사 반영)

> **지위**: 분석·계획. **GPU 지출 0 · 새 측정 0 · 새 성능 판정 0.** 연구 결론·Claim 등급·게이트 불변.
> rev1(같은 날)은 claims-auditor 적대 감사에서 Q1 부분 REFUTED(D-8 상태 낡음·D-1 서술 오류·D-10 "완료" 오류·D-5 인용 오류·번호 체계 충돌),
> Q2 D-1 축소 REFUTED·D-8→P11 전용 REFUTED, Q4 residency 문구 REFUTED, Q5 등급 오류를 받았다. rev2는 그 교체 문구·삭제 목록을 반영한다.
> 출처: `reports/synthesis_2026-09-22/EXPERIMENT_DESIGN_2026-09-22.md`, `reports/paper/{CLAIM_EVIDENCE_MATRIX,EXPERIMENT_ROADMAP,venue_positioning}.md`,
> `reports/audit/2026-09-22_scope_lineage/REPORT.md`, `handoff-report/{vessl_cloud_setup_plan_2026-09-24,design_memo_span_step_boundary_costmodel_2026-09-24}.md`.
> ★번호 체계: VESSL 실행 단계는 **V-0 기판 / V-B5 회수 / V-char 특성화 / V-cap 용량 / V-pol 정책**으로 부르고, 설계 메모의 술어 P0–P12·로드맵 P1–P6과 분리한다.

## 0. 선행 게이트 (모든 KISTI 수치 재사용의 전제)
1. **기판 동등성 판정**(사용자 + 규칙층 감사): VESSL A100 SXM4(driver 580/CUDA 13.0, K8s, 클럭 고정 불가) vs KISTI A100. 판정 전엔 KISTI 수치(d44·λ\*·격자)를 VESSL 가이드라인에 대입하지 않는다. **λ\* 일치는 동등성 근거가 아니다**(교훈 247).
2. **n 독립성**: 한 job 안 다중 boot ≠ 독립 n(B3C-3). 비교 캠페인은 **독립 job ≥2 × n≥4**. P1-opint의 n=5도 동일 job·노드였다.
3. **모델별 용량 먼저**(게이트 6): V-cap은 Nano-9B 두 shape뿐 — D-3의 다른 모델은 λ\*가 예산에 없다.
4. **관측자 효과**(HOLB G5 UNDETERMINED): probe-off 대조를 같은 job에, 전 arm telemetry 구성 동일.
5. **조성(composition) 주장엔 모델 hold-out ≥2–3개**(단일 모델로는 식별 불가).
6. **SLO별 재튜닝**, 재채점 금지.
7. **적용 범위**: radix OFF · ctx ≤16K · 합성 shape(short/long 분리). radix ON으로 결론 이식 금지.

## 1. 카드별 검토

| 카드 | 검증 대상 | 봉사 기여 | 필요도 | GPU-h | 근거·조건 |
|---|---|---|---|---:|---|
| **V-0 = D-0** 기판 점검 | green-context 입도·두 파티션 비중첩·cudagraph 중 격리(realized). L0 유사물(드라이버 보고값, 하드웨어 층 아님) | 전부의 전제(게이트 1) | 필수 | 0.05–0.1 | 실패하면 "기판 이식성"이 결과 |
| D-6-a span 발화 예보 | 층 단위 prefill 발사가 꺼져 있는가 | Claim B 스코프 | 완료(스크립트 미등록) | 0 | 09-22 census + 09-24 P0 재집계. 인용 전 스크립트 저장소 등록·selftest 필요 |
| D-7-a 부분 중첩 분할 코드 조사 | HE0 vs Bullet §4.4 정면 대조 가능성 | C3 방어(CONSENSUS §5 열린 항목 11, 회부 중) | 필수(GPU 0) | 0 | "불가"면 D-7-b 설계 안 함 |
| D-10 정본 정정 | I-1·I-2·I-3·M2 협소화 | — | **부분 완료** | 0 | I-1·§3 280–283·§5-11은 b330ae8. **미완**: I-2(`residency_census` README:95·`.py:55` 줄 인용 표류), I-3(`multiplexing_mixin.py:451-453` 로그 문구), `venue_positioning.md` §0.2(4) M2 협소화·§1 C1 문장 낡음 |
| **D-5** residency 분리 프로브 | PART 상태 비용을 "SM 수"와 "동거"로 분해 | residency 축(정본 §1 rev44 분할 상태 비용의 재확인) | 필수 | 0.4–0.6 | ★dense 선행(MuxWise §3.3.1·Bullet §3.2.3)이 동거 경합을 **이미 보고**(감사 M4) — 우리 몫은 **hybrid 재확인 + 체류 양 계수**. 조건: `PDMUX_TRACE_FORCE_PREFILL` force-off 대조를 같은 job에(관측자 α, D-2 대체) |
| **D-1** E2 sticky 대조 | sticky가 realized 분할을 고정하는가(계측 타당성) | 기여 아님 — fixed-D arm 계기 | 필수(등록 원안 그대로) | **1.803**(최악 2.338, 하드캡 3.0) | 등록 설계(R1–R6·추정량·seed·arm·20 boot)는 **유효**(판정서 `:17-19`). VESSL 전제 = A13 이식 追記(경로·런처·벽시계 집행기 한정) + **새 OVERRIDE(사용자, 보류 중)** + B0/B1/B4 선행. **축소하지 않는다**(R1 전 seed·R5 seed 산포·R4″·Q2 n=4가 깨짐). E2C-36~39 승계 |
| **V-char** 특성화(신규) | 운영점 decode step 시간(SM×bs×ctx), 전환 비용·bubble(메모 P12) | C4(로드맵 P3 축소판) | 필수 | 0.8–1.5 | 인용 금지된 옛 decode-floor 값을 대체할 운영점 데이터. **성분 측정 — 정책 결론(sticky 손익) 도출 금지**(N4/2F9). 모델 수 명시(hold-out) |
| **V-cap** λ\* 재측정 | VESSL 용량(Nano-9B shape A·B) | 게이트 6 | 필수 | 1.4–1.8 | λ0 재등록. caveat: "λ\*(A)는 D44 측정이 아니다", 17자리 인용 금지 |
| **D-3** Gate 3 | PD-mux(묶음 처치) vs fused, 운영점, NemotronH·Falcon-H1 | C1 | 필수 | 3.0(2모델) / ≈6(4모델 VESSL, 용량 포함) | 2 KISTI + 2 VESSL을 "4모델"로 합치면 게이트 1 위반 — 동등성 판정 없으면 4모델 전부 VESSL |
| **D-8** Q-A regret | 고정 split의 ITL regret 프론티어 | 계측 기록 | **완주(2026-09-11, 10.641 GPU-h)** | 0 추가 | 계측 기록으로만 등재, 인용 금지 80건 승계. **argmin·포락선 산출 금지 등록이라 P11에 쓸 수 없다** |
| **P11**(신규, 별도 사전등록) | 워크로드별 오라클 static − peak-decode 규칙 global static 갭 | Claim E 정의역·상금 크기 | 조건부(P9 선행) | ≈Q-A급(≈10) | 최소 설계: V-cap 선행 · 전체 격자 static 5 arm · short/long 분리 워크로드 ≥2(각각 Claim E 정의역 후보로 선언) · λ\* 분수 부하 2점 · n≥4 × job≥2 · 정본 goodput 술어 · SLO별 재튜닝 · **오라클 선택 job 1, 평가 job 2 분리**(winner's curse). 먼저 CPU 단계(기존 자료)로 정의역 존재 확인 |
| D-2 E1-b/c Gate 2-S | "SM 분할 자체" 귀속 | C1 귀속 | 보류(조건부) | 0.5–0.7 | 조건 ① 논문에 "SM 분할 자체/PD 분리 자체" 미사용 ② force-off 대조를 D-5에 흡수 ③ Gate 2-S 성분 격리는 트레이드오프(ITL p95↑·TTFT p95↓)임을 병기 |
| D-6-b span budget 스윕 | span 입도가 정책 레버인가 | Claim B 보강 | 불필요(현 범위) | 1.0–1.5 | 근거: 현 범위는 Claim B 협소형만 주장. **layer 단위 prefill 진행도 제어(MuxWise N_PL·Bullet L_exe)는 미시험**으로 명시(M3) |
| D-4 %smid 하드웨어 프로브 | Gate 2 SM 귀속(하드웨어 층) | C1 귀속 | 보류 | 미산정 | 재설계 필요. D-0은 L0 유사물이라 대체하지 못한다 |
| D-9 P2 architecture | true-dual coupling 감소(Claim D) | H-Arch | 조건부 | 미산정 | 경로 B에서만 |
| **V-pol** = P9 정책 캠페인 | B6 vs B1·B5 | H-Policy(Claim E) | 조건부 | 10–20 | P11 갭 ≥3%(정의역 비어 있지 않을 때) 그리고 D-9 통과 시 |

## 2. 논문 기여 후보 — 증거 수준별 (감사 교체 문구)

**A. KISTI A100 기판 결과로 성립(재측정 없이 재사용 가능 — 단 "KISTI 기판 결과"로 표기, VESSL 수치와 혼합 금지)**
- **Claim B**(강한 지지, 현 구현·green-context·Zamba2 경로 한정, 헤드라인 아님): layer 경계마다 SM 파티션을 재구성하는 layer-type 정책은 死. 근거 coordinated per-type TPOT 42→124 ms(OPT 85 ms), **양 arm no-cudagraph(eager) 측정**. 운영점에서는 per-type이 cudagraph와 양립하지 않아 진입 불가. offline 고정은 결정 비용만 없애고 실행시 (D) 비용은 남음. layer 단위 prefill 진행도 제어(MuxWise/Bullet 방식)는 미시험. libsmctrl 대조·Nsight timeline 없음(VESSL에서 ncu 불가).
- **C3 = HE0**(KISTI A100, Zamba2-2.7B 단일 모델, 변화 trace, 예산 구속 분리 격자 p+d=108, n≥4, 관대 3 s·tight 300/50 ms): shared running-batch coupling 하의 reactive single-worker 분할 레버 단독 동적 제어는 best decode-heavy static을 못 넘는다. 기전 positioning + entanglement(기판 무관성은 "후보"). 달성 가능한 천장은 미확정. Bullet §4.4와의 관계는 회부 중(D-7-a 선행).

**B. VESSL 실험으로 완성**
- **C1(관측형)**: 이 엔진에서 PD-mux(묶음 처치)를 켜면 기본 설정 fused보다 꼬리 SLO goodput이 좋다 — 술어·모델·워크로드 한정, 기전 귀속 미확립(SM 분할 자체 = NOT-YET-SUPPORTED, aux 두 플래그는 원인에서 배제, 귀속 상한은 pdmux 서브시스템). D-3 기대 [제안]. 4모델 문장은 같은 기판 4모델 측정 또는 동등성 판정 후에만.
- **C4 특성화**: decode floor의 ctx·load 의존을 운영점에서 — V-char. composition 의존은 모델 hold-out이 있을 때만 주장.
- **residency 축**: "정의·분모·규약을 갖춘 집계 추정량으로서 residency(PD 동거 wall-clock 비율)를 보고한 PD-mux 선행은 확인되지 않는다(조사 범위 DuetServe·MuxWise·Nexus·Bullet). 각주: Bullet Fig. 20a는 SM 구성별 지속시간 막대 타임라인, MuxWise GPU utilization은 Nsight active-SM 비율(residency 아님)." "SM 수 vs 동거" 분해는 dense 선행이 동거 경합을 이미 보고 — 우리 몫은 hybrid 재확인 + 체류 양 계수. 정당화는 트렌드가 아니라 내부 타당성.
- **guideline**(정본 문구): peak decode 부하 기준 decode-heavy static 고정(검증된 profile 없으면 agnostic).
- **방법론**: realized-vs-target 검증, 기판 이식성(V-0), 계측 타당성(E2).

**C. 조건부 upside**
- **H-Policy**: MuxWise의 워크로드별 분할표가 "워크로드별 static 선택"을 이미 선점 — 남는 차이는 **B6(조성 프로파일) vs B5(generic)**뿐. offline floor 예측은 여기(Tier C). 조건: P11 정의역 ≠ ∅ ∧ 갭 ≥3% → D-9 → V-pol B6 > B1·B5.
- **H-Arch**: Bullet 재발명 위험, hybrid 한정 + coupling 정량 기전으로 차별화.

투고 경로: 경로 A(characterization + 기전 negative + guideline). 경로 B는 P11이 열 때만.

## 3. 기대 결과 트리

| 실험 | 기대(등급) | 나오면 | 반대면 |
|---|---|---|---|
| V-0 | 입도·비중첩·cudagraph 격리 성립 **[제안]**(VESSL 미확인) | KISTI 격자 재사용 후보(동등성 별도) | 이식성 결과, 격자 재설계 |
| V-cap | Nano-9B A·B의 λ\* **[제안]**; 값 자체가 산출물 | 이후 부하를 λ\* 분수로 정의 | (없음 — 측정 자체가 목적). λ\* 일치를 동등성 근거로 쓰지 않음 |
| V-char | decode step 시간의 SM×bs×ctx 곡면 **[제안]**; 방향 예측 없음(철회된 Stage 0 "SM-무감각" 형태 회피, 8B C2 2.36–2.91× 인용정지) | C4 운영점 특성화 | — |
| D-5 | H-동거 1.00–1.15× / 혼합 1.15–1.5×(분해 추가 필요) / H-SM수 1.5–1.9× **[제안]** | H-동거 = hybrid 재확인 | H-SM수 = §1 rev44 서술 자기 반증 |
| D-3 | 부호·자릿수 **[제안]** | C1 완성(같은 기판 조건) | 모델 의존으로 기록 |
| D-1 | ON >90% / OFF <20% **[등록]**(원안 그대로 실행 시만) | fixed-D 계측 타당 | sticky 결함 = 엔진 신규 사실 |
| P11 | 정의역 존재 여부 + 갭 **[제안]** | 갭 ≥3%: V-pol 진행 근거 | 정의역 ∅ 또는 갭 <3%: **그 정의역 한정으로** Claim E 진행 근거 없음 |
| V-pol | B6 > B1·B5 ≥3% | 경로 B | 경로 A 유지, D/E는 negative architecture result |

## 4. 예산 (rev2)
| 구성 | GPU-h | 비용 |
|---|---:|---:|
| 경로 A 최소: V-0 · V-B5 · V-char · V-cap · D-5 · D-1(원안) · D-3(2모델) | ≈7.8–9.5 | ≈$12–15 |
| + D-3 4모델 VESSL(용량 포함, 동등성 판정 없을 때) | +3 → ≈11–12.5 | ≈$17–19 |
| × 1.5–2 여유(VESSL 오버헤드·재실행) | ≈12–25 | ≈$19–39 |
| P11(별도 등록, ≈Q-A급) | +≈10(+여유) | +≈$16–31 |
| V-pol(조건부) | +10–20 | +$15–31 |
| 제외: D-2·D-6-b(1.5–2.2) + D-4(미산정) | | |

## 5. doc-steward 회부 (감사 부수 발견)
- `venue_positioning.md` §1 C1 문장(`:398-401` "4모델 전부서 cudagraph-ON")과 §0.2(4)(`:322` "하나도 없다")가 정본(`CONSENSUS.md:3439`, 감사 M2)보다 낡음.
- D-10 잔여: I-2·I-3.
