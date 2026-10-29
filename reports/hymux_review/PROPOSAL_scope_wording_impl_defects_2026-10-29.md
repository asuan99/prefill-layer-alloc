# 정본 범위 정정 제안 — PD-mux 루프 구조 의존 (2026-10-29)

> 지위: **제안**(결정 54 ②). 정본(CONSENSUS·COSTMODEL·PROJECT_STATUS·CLAUDE.md)은 고치지 않았다. 반영은 promote 1회, 반영 전 문구 diff에 대해 claims-auditor 2차(마지막) 감사.
> 근거: 감사 r1 `workspace/engine-port/results/impl_defects/AUDIT_scope_loop_dependence_r1_2026-10-29.md`(행 번호 = 이 문서의 #), Phase 0 `…/impl_defects/PHASE0_CPU_CHECK_2026-10-28.md`.
> **새 성능 판정 0 · §6 등급 변경 0 · 부호가 바뀌는 결론 0.** 바뀌는 것은 문구 범위뿐이다.

## 0. 공통 원칙

1. 이 프로젝트의 serving 결론은 전부 **SGLang 공개 PD-mux 루프 하나**(lockstep 단일 프로세스, 고정 토큰 예산 65536, 진행 중 prefill drain 후 전환, prefill 단일 배치·선점 없음, 비sticky 전 SM 복귀)에서 나왔다. 이 루프는 MuxWise 논문 시스템(N_PL 시간 맞춤, 남은 층 GreenContext 이동, 층 경계 선점)·Bullet(별도 프로세스, step별 동기화, SM 부분 공유)과 다르다.
2. 루프와 무관한 성분 측정(해석 수식, T1/T2 SM 스케일링, C2 2.36–2.91×, CPU 0.014%, agnostic 경로 s≤0.04 ms)은 유지한다.
3. "SGLang 공개 PD-mux 구현"을 MuxWise 논문 시스템과 동일시하지 않는다. MuxWise upstream 공개 코드(PR #10692)와 이 루프의 대응은 **미대조**다(대조 전까지 "공개 코드에도 N_PL·선점이 없다"고 쓰지 않는다).

## 1. 문구 교체 (OVERBROAD)

| # | 위치 | 현재 | 제안 |
|---|---|---|---|
| 1 | CONSENSUS §0 `:3666-3667` | 그 위의 똑똑한 정책은 전부 실패 / layer-type 기반 정책은 全형태 死 | SGLang 공개 PD-mux 루프에서 시험한 정책(층 경계 SM 전환 계열, split index 반응형 컨트롤러)은 best static을 넘지 못했다. |
| 2 | CONSENSUS §1-3 `:3682`, CLAUDE.md "현재 결론" 2, MEMORY 색인 | layer-type 런타임 정책 全형태 死 | KISTI A100·Zamba2-2.7B·triton·no-cudagraph에서, SGLang 공개 PD-mux 루프 위에 구현한 "층 종류 경계마다 SM 분할을 바꾸는" 정책 계열(coord R0d·OPT·v4·PF)은 모두 agnostic·tuned static에 졌다(TPOT 42→124 ms, 최적화 후 85 ms). 층 종류를 span 크기(시간 예산)로만 쓰는 정책, 박자 영역, handoff, preemption, cudagraph-ON 구현은 시험하지 않았다. |
| 3 | §1-15 `:3719` | 살아있는 offline 역할은 오직 whole-phase floor | …이 구현의 층 경계 SM 전환은 offline 고정이어도 실행 시 비용을 낸다. step 안 SM 분할을 바꾸지 않는 층 종류 활용(span 크기·발행 시점)은 이 판정 범위 밖이다. |
| 6 | CLAUDE.md "현재 결론" 3, §1-17 `:3717`, COSTMODEL `:73` | 실패 원인 = switch overhead 아님 | 컨트롤러 CPU 비용과 agnostic 경로 전환 비용은 무시 가능하다. 컨트롤러 전환의 진행 중 drain(`:1344`)은 미측정이며, 상한상 bind+GATE에는 작고(≤0.72%) slo arm에서는 배제할 수 없다. bind+GATE의 열위는 positioning으로 설명된다. |
| 10 | §1-24 `:3698`, COSTMODEL `:50`·`:86` 답 4 | ITL 꼬리 = monolithic prefill 정지 — 구조적, 고칠 수 없다 | KISTI·Zamba2-2.7B·triton의 이 루프에서 ITL 꼬리는 최장 프롬프트 prefill과 시간상 겹치는 정지가 지배했고 크기는 D에 단조였다. 정지 기전은 미확정이며 VESSL·Nano에서는 관측되지 않았다. PD-mux와 chunked prefill의 비양립은 엔진 사실이다. |
| 11 | COSTMODEL CM-4 `:21`·`:54` | 동거 decode 간격은 span 시간에 거의 반응하지 않음 | (기존 문안 뒤) 이 run들은 적체 영역(span > 동거 decode step, prefill 미동기 발행)이었고, 이 영역에서 무반응은 루프 구조상 예상되는 값이다. 박자 영역에서는 측정되지 않았다. |
| 13 | COSTMODEL "이미 답한 것" 1 `:82` (계획 D1) | 결정 비용 무시 가능 → 성패는 예측 정확도 | 전환 연산과 결정 계산은 무시 가능하다. 정책의 성패는 예측 정확도와 실행 루프 구조(발행·입장·전환 drain)에 달려 있으며, 후자는 공개 루프 하나에서만 측정됐다. 진행 중 전환 drain(조작 비용)은 미측정. |
| 14 | COSTMODEL "이미 답한 것" 2 `:83`, CM-2 `:39` | 층 종류 차이는 런타임 분할 레버로 쓸 수 없다 | 층 종류 차이는 층 경계 SM 재분할 레버로는 쓸 수 없다(이 구현, Diff B≈1, Zamba2·triton·no-cudagraph micro). 층 종류별 비용 차(Diff A, L에 따라 크게 열림)를 span 시간 예산에 쓰는 레버는 미시험이다(CM-2 열린 질문 `:43`과 정합). |

## 2. 범위 구절 추가 (NEEDS SCOPE)

| # | 위치 | 추가 구절 |
|---|---|---|
| 4 | §1-1 / Claim G `:3680`·`:8786`, CLAUDE.md "현재 결론" 1 (계획 D3) | "SGLang 공개 PD-mux 구현(lockstep 단일 프로세스 루프, budget 65536, 적체 영역) 활성화는 …". MuxWise 논문 시스템과 동일시 금지. |
| 5 | §1-7·§1-17 HE0 `:3686`·`:3722`, CLAUDE.md "현재 결론" 3 (계획 D2) | "…SGLang 공개 PD-mux 루프(결정 = prefill 배치 입장 시점, 전환 = 진행 중 prefill drain, 이 trace 배치의 94–95%는 span 1개)에서…" |
| 7 | §1-4 `:3683`, Claim C `:8782` | "(cap 구속, prefill 입장 직렬 루프에서)" |
| 8 | §1-6 `:3685`, §1 실전 권고 `:3729`, CLAUDE.md "현재 결론" 4 | "…비sticky SGLang PD-mux 루프(decode-only·prefill-only 구간은 전 SM 복귀)에서는 decode-heavy static 라벨의 과다공급 비용이 동거 구간에만 생긴다. 라벨은 목표 D이지 실현 할당이 아니다(결정 49)." 기본값은 결정 40/43대로 Nano에서 재결정. |
| 9 | §1-13 `:3711` | "(이 trace의 LO에서 prefill 배치 94–95%가 span 1개이고 비sticky 복귀가 있는 루프에서)" |
| 12 | §1-38 / CM-4 ρ `:3727`, COSTMODEL `:51-52` (계획 D5) | "(적체 영역 측정, 8k 프롬프트·예산 65536의 8층 혼합 span)" |
| 15 | §1-16–§1-20 `:3714-3721`, Claim F `:8785` | "(SGLang 공개 PD-mux 루프의 static·split index 컨트롤러 기준)" |

## 3. 계획서 D4·D6 처리

- D4(§1-24 기전 미확정, drain 후보) → #10에 흡수.
- D6(venue_positioning·관련연구 "MuxWise upstream 공개 코드 기반, 논문 디스패처 미포함") → `reports/paper/`는 동결 이력이므로 고치지 않는다. 앞으로 쓰는 관련연구·원고 문안에만 적용하고, "논문 디스패처 미포함"은 PR #10692 대조 뒤에 확정한다.

## 4. 이 정정이 열어 두는 실험 (감사 r1 §3, 실행은 사용자 승인)

1. 층 종류 span 정책: {budget 65536, time, time_typed} × static D44, 박자 영역 셀(512×D44 또는 E2 depth 1), Nano·VESSL·cudagraph-ON, n≥4 paired. 판정 time_typed 대 time |Δ goodput| ≥3%, CI가 0 제외.
2. B1 크기: 진단 PREREG 새 rev(8k, 동적 arm 진행 중 전환 유도, PHASE_DIAG drain ms) → Nano HE0 핵심 셀, handoff ON/OFF 별도 arm.
3. 비sticky 편향: D{24,44} × sticky{OFF,ON}, LO·HI, n≥4.
4. CM-4 영역: 경계 4셀 + 512×D44에서 gpu_ms 짝, 기울기 재추정.
5. §1-24 KISTI 정지: 결정 46으로 판별 불가 — 문구만 교정.
