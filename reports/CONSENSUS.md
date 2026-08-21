# CONSENSUS — engine-port PD-mux 연구의 합의점 (2026-07-19 historical)

> **현재 전체 정본:** [`../PROJECT_STATUS.md`](../PROJECT_STATUS.md).
> 이 문서는 dual-worker/R2 이전까지 확정된 phase separation, layer-granular
> negative result, entanglement, single-worker dynamic 결과의 정본으로 유지한다.

최종 갱신: 2026-08-21 rev41 (doc-steward — ★**gate #13 job-축 캠페인
완주 + 양 arm `PASS`**(등록 11.52 GPU-hr, 실측 11.66 GPU-hr) 반영 +
S0(a) 초판 결론 **철회**(claims-auditor REFUTED) + G18 rate-축 프로브
**트랙 보류(HOLD)** + kernel_mech rev3 차단 2건 설계 본문 수리 +
rev40 자신의 P1 판정 과잉 인용 정정[아래 追記(6) 참조]). §3 항목
71–74 신설(블라인딩 자기인용 자기거부 · 부분 격자 확장 산물 ·
비용 비정합 대조 · 태그가 오염을 못 막음). §4에 신규 living-doc
행 4개(G13_RESULTS·S0A_VERDICT·G18 HOLD 문서·kernel_mech rev3).
★★쓸 수 없는 문장 불변: **"gate #13을 닫았다"**(865533 batch⊗regime
앨리어스 불변) · **`Δ_batch`를 측정했다**(배치 축 없음, `CONC=16`
단일) · `grand_mean_r`(M8 3.0613·Ha8 3.1229) arm 비교·2.91/3.058/
3.114 대조. σ_alloc은 2노드·계수 0.400·유효 df≈1로만 표집(정본
gate #13(1)의 ≥3노드 부분 미충족 확정). 부수: 비용 상수 144s/부팅이
실측 145.8s로 검증돼 rev3 §6-8 미해결 항목 해소. **새 성능 판정
0건**(gate #13은 분산 측정) · 결론 철회 1건(S0(a)) · 등급 변경
0건 · 정책 순위 변경 0건 · GPU 지출(세션 합계) 11.72 GPU-hr(캠페인
11.66 + 취소 프로브 0.06). 상세 `PROJECT_STATUS.md` 최상단 배너,
`handoff-report/session_handoff_2026-08-21.md`.
이전 rev40: 2026-08-20 (doc-steward — **P1 프로브 판정 반영:
`UNAVAILABLE (CUPTI×GREEN-CONTEXT)`**(job 886718, `--exclusive
--constrain=hwperf`, node gpu38, 1분55초, 동반 프로브 886752 포함
**GPU 지출 0.032 GPU-hr**). §3 항목52에 追記(6) 신설 — "미확인
리스크"였던 CUPTI×green-context 귀속이 **두 겹 대조**(다리 간:
greenctx exit=9/control exit=0 같은 GEMM 정상 수집·다리 내부: 같은
프로세스에서 green ctx 밖 RNG 커널만 성공)로 관측 근거를 얻어
**실증됨**으로 갱신됐다(단 내부 기전 미분리·A100-SXM4-80GB/driver
580.105.08/ncu 2025.3.1.0/CUDA 13.0.2/이 클러스터 한정, 이식 금지).
⇒ **Stage B(ncu 커널 내부 카운터를 SM 제한 하에서 수집)는 이 기판에서
구성상 불가로 확정 → `kernel_mech` rev3는 Stage A 전용으로 범위
축소**(4세션 이월의 실질 원인 해소). ★★★**이것은 "성능" 판정이
아니라 "도구 타당성" 판정이다** — 깨진 것은 프로파일링이지 green
context 실행이 아니다(`realized_sm=16`으로 정상 실현). **새 성능
판정 0건 · 등급 변경 0건 · 정책 순위 변경 0건 · GPU 지출 0.032
GPU-hr.** 상세 `workspace/engine-port/results/kernel_mech/p1_probe/
P1_VERDICT_2026-08-20.md`(동반 프로브 검토 `P1_886752_REVIEW_
2026-08-20.md`), `PROJECT_STATUS.md` "8B decode-SM 민감도 측정
노트"(실증 확정 배너)·"다음 실험 gate" #11 레지스트리(kernel_mech
P1 프로브 행 신설)·#17(kernel_mech rev3 스코프 확정).
이전 rev39: 2026-08-19 (doc-steward — **gate #13 rev2 규칙층
NO-GO(死因 3건) → rev3 재감사 GO-with-caveats(등록 11.52 GPU-hr,
미제출·여전히 사전등록 아님) 반영 + gate #16 이차 표적(rate 축)
규칙층 발견(rate↓는 목표 (i) 과부하 이탈엔 옳으나 목표 (ii)
`D_itl`/`Δ` 식별엔 반대 방향) + `PROJECT_STATUS.md` "다음 실험
gate" #17의 후보 목록 정정(2건→4건, gate #16 rate 축·S-6 누락 발견)
+ 방법론 게이트 1건 신설(§3 항목70, 방법론 게이트 #50과 대응 —
게이트 #9의 열세 번째 재발: 자기 수리를 검증하는 검사 자신이
항등식이었던 사례 + 변이 테스트 게이트 제안 + B7 재현성 지뢰).
**새 성능 판정 0건 · 캠페인 등급 변경 0건 · 정책 순위 변경 0건 ·
GPU 지출 0(이번 세션 job 제출 0건).**
G16 정본 승격(rev37·§1-33)·G17/E-B1 NO-GO(rev38)는 각각 2026-08-17·
2026-08-18에 이미 완료됐다. 이번 세션은 그 이후를 감사·수리한다 —
전부 규칙층/설계층 판정이고 데이터 캠페인은 실행되지 않았다(gate
#13 rev3는 **밴드 점(11.52 GPU-hr)이 정해졌을 뿐 사전등록 문서도
쓰이지 않았고**(수리 자체가 미감사), gate #16 rate 축은
규칙층 초안 상태다). ★**"gate #16을 닫았다"·"gate #13을 닫았다"고
쓰지 말 것 — 둘 다 불변**: gate #13은 rev3로도 batch⊗regime
앨리어스와 노드·날짜 축 미배선 때문에 닫히지 않고, gate #16은
원문 문턱 판본이 여전히 rate 축에 열려 있다(이번 세션이 rate↓로는
그 축이 닫히지 않음을 추가로 확인했을 뿐이다). 상세
`workspace/engine-port/results/s8_scaleup/
DESIGN_G13_JOB_BATCH_REV3_2026-08-19.md`, `workspace/engine-port/
results/slo_sched/DESIGN_G18_RATE_AXIS_2026-08-19.md`,
`PROJECT_STATUS.md` "다음 실험 gate" #16·#17(갱신)·#11 레지스트리·
"방법론 게이트" #50(신설).
이전 rev38: 2026-08-18 (doc-steward — **G17(payoff 밴드+
`gap_upper` 비식별+sticky 레버)·E-B1(shadow price) 설계 규칙층
NO-GO 2건 반영 + 방법론 게이트 3건 신설(§3 항목67·68·69, 방법론
게이트 #47–49와 대응) + `EXPERIMENT_PLAN_2026-08-18.md` stale 표시.
새 성능 판정 0건 · 캠페인 등급 변경 0건 · 정책 순위 변경 0건 · GPU
지출 0(이번 세션 job 제출 0건).**
G16 정본 승격(rev37·§1-33)은 전날 이미 완료됐다. 이번 세션은 그
결론을 뒤집거나 확장하려는 시도 5건(G17·E-B1·`g2_0` d34 "골짜기"·
용량축·`out`축)을 시도했고 **전부 실패**했다 — 설계 NO-GO 2건 +
메인 세션 자체 재검으로 REFUTED 3건(claims-auditor 회부 없음). ★G17
死因 3건(S2 격자가 `Δ_SLO≥0`을 데이터와 무관하게 강제·sticky가 동거
아닌 단독-at-D를 잼·estimand 검열), ★E-B1 死因(판정 가능 창이 대수로
공집합, `max_running_requests=48` 캡). ★★메인 세션이 정본 인용금지
(33.04, CONSENSUS §33 승격금지(vii))를 등재 다음 날 위반 [CS-OK] → 기계적
차단 도구 신설(§3 항목67). ★★★`results/slo_sched/`를 G16 전용
디렉터리로 오인해 사실 오류 산출·정정(§3 항목68). ★★★정본이 이미
`ILL-POSED`로 판정한 `g2_0_hard` rA5를 판정서를 안 읽고 재분석(§3
항목69, 방법론 게이트 #18 최강 사례). 상세
`handoff-report/session_handoff_2026-08-18.md`,
`PROJECT_STATUS.md` "다음 실험 gate" #17(신설)·"방법론 게이트"
#47–49(신설).
이전: 2026-08-17 rev37 (doc-steward — **G16 캠페인 완료(4블록,
jobs 884336/884410/884411/884412) + claims-auditor 적대 감사 반영,
`G16_RESULTS_2026-08-17.md` rev2 정본 승격. 새 §1-33(신설) + §3
항목65·66(신설, 방법론 게이트 #45·46과 대응). 새 성능 판정 0건 ·
캠페인 등급 변경 0건 · 정책 순위 변경 0건 · GPU ≈3.9 GPU-hr(캠페인
자체, 이번 세션 신규 지출 0 — 감사·문서 반영만).**
양 phase `ITL_SATURATED`(사전등록 §6이 규정한 "정보 있는 음성").
★**"gate #16을 닫았다"고 쓰지 말 것 — 불변**: 닫히는 것은 R2
결정량②(§1-32)의 **재정식화판**뿐이고, 원문 문턱 판본은 **rate 축**에
그대로 남는다. H-1~H-8 판정·exact band·A-2 residency 스코프·
`gap_upper` 비식별 구조 등 전문은 §1-33. 상세
`workspace/engine-port/results/slo_sched/G16_RESULTS_2026-08-17.md`
(rev2)·`PREREG_G16_RULES_REV3_2026-08-16.md`(addendum C/D/E)·
`PROJECT_STATUS.md` "다음 실험 gate" #16(갱신)·"방법론 게이트" #45·46.
이전 rev36: 2026-08-16 (doc-steward, 세션4 — **G16 스모크 2회
통과 + 하네스 결함 2건 해소 + 사전등록 분석기 `g16_analyze.py` 작성·
감사[조건부 GO]·반영 + 블록 1(job 884336) 제출[PENDING] + 등급 변화
후보 1건(H18 PLAUSIBLE→CONFIRMED, 설계 속성) + 정본 자신의 인용 결함
1건 정정(ncu/CUPTI, §3 항목52 追記(5)) + 설계 감사 2건 NO-GO(gate #13·
kernel_mech rev2) + 방법론 게이트 3건 신설(§3 항목62–64). 새 성능
판정 0건 · 캠페인 등급 변경 0건 · GPU 0.272 GPU-hr(스모크 2회) ·
job 제출 1건(884336, PENDING).**
(1) **G16 진행상황은 rev31 관례에 따라 `PROJECT_STATUS.md` "다음
실험 gate" #11 레지스트리에만 등재**(CONSENSUS 신규 항목 없음) —
스모크 2회(884292→하네스 결함 2건 발견→`37cf6b8` 수정→884320
`G16_SMOKE_OVERALL=PASS`), 분석기 작성(커밋 `4f05dce`)→claims-auditor
조건부 GO→반영(F1–F4/S1/S6, 커밋 `8311d6c`), 블록 1 제출·PENDING.
★기존 배너 불변: **"gate #16을 닫았다"고 쓰지 말 것**(재정식화판만
닫음, rate 축 원문 문턱 판본 잔존). (2) **H18 PLAUSIBLE→CONFIRMED**
(addendum B-3 승격 조건 충족 — arm별 `boot_s` 884292
32.7/32.7/32.7초·884320 32.8/32.7/32.8초로 위치-0 arm이 cold-cache
페널티를 안 문다. 하네스 설계 속성이지 성능 판정 아님, PROJECT_STATUS.md
등재). (3) ★★**정본 자신의 인용 결함 정정**(kernel_mech rev2 감사
F3에서 발견, §3 항목52 追記(5)) — 정본이 `_ncu_target.py:73-74`
("ncu profiling always runs at full GPU")만 인용하고 그 **다섯 줄
위(68-71)**의 CUPTI×green-context 비호환 사유를 인용하지 않아 왔다.
참이면 kernel_mech Stage B는 이 기판에서 구성상 불가하나, `error
code 9`엔 저장소가 3가지 경합 귀속을 갖고 있어 **아직 미확인
리스크로만 등재**(판별기 = Stage 0′ P1 프로브, 발화 시 라벨은
`UNAVAILABLE (CUPTI×GREEN-CONTEXT)`). (4) **설계 감사 2건 NO-GO**
(GPU 0, PROJECT_STATUS.md 레지스트리 등재, CONSENSUS 신규 항목 없음
— C2-R rev1/G16 registry-only 관례를 따름): **gate #13** job/node축
설계(`DESIGN_G13_JOB_BATCH_2026-08-16.md`) — 지정 1차 결정량
"between-job SD of r은 부분 항등식"·"Ha8 검정력≈0" 둘 다 REFUTED,
UB95 규칙이 부팅 잡음을 job 축 통제로 오판. **kernel_mech rev2**
(`DESIGN_KERNEL_MECH_REV2_2026-08-16.md`) — `wave_eff`가 ncu
메트릭이 아닌 수제 유도를 되살림(게이트 #36 死因 재생)·§3.2 부호
반대로 위험구간 통과·`ncu --pid` 존재하지 않는 옵션. 둘 다
**"닫혔다"고 쓰지 말 것**. (5) **방법론 게이트 3건 신설**(전부
아래 §3): #42(항목62) 게이트가 자기 실패를 성공으로 라벨링(교훈
항목21의 거울상) · #43(항목63) 분석기가 사전등록 기호를 조용히
재정의하면 자기 대조를 깨고 중심 산출물을 침묵시킴(+메인 세션
오진 1건 기록) · #44(항목64) 양성대조의 빈 서명 구멍. 상세
`handoff-report/session_handoff_2026-08-16.md` §4-0–4-7.
이전 rev35: 2026-08-16 (doc-steward — **gate #16 사전등록 규칙→
하네스 2단 감사 완주 반영(`workspace/engine-port/results/slo_sched/
PREREG_G16_RULES_REV3_2026-08-16.md` rev3+addendum A+B) + engine-porter
이관 1건 해소(`he2_bench.sbatch:92`, 커밋 `dadb851`) + 방법론 게이트
2건. 새 성능 판정 0건 · 등급 변경 0건 · GPU 지출 0 · job 제출 0
(스모크 미제출).**
(1) **G16 사전등록 자체는 rev31 관례에 따라 `PROJECT_STATUS.md` "다음
실험 gate" #11 레지스트리에만 등재**(CONSENSUS 신규 항목 없음) — 규칙
감사 GO → 하네스 감사 NO-GO(4개 사유, `PIN_GATE=0.80`이 7 arm 전부
배제·H3 검출이 항등식 포함) → addendum A 수정 10/10 → 재감사 GO →
addendum B 수정 5/5, **미제출**(≈4.0–5.5 GPU-hr, 스모크 0.2–0.35).
결정량 재정식화(`Δ_SLO` 단일 60ms 점 → 사다리 함수, 아래 §3 항목60
(ii) 참조) 결과 payoff는 ITL SLO≲58.6ms 구간에만 존재함이 도출됐다.
★**"gate #16을 닫았다"고 쓰지 말 것** — 닫는 것은 R2 결정량②(§1-32)의
**재정식화판**이고, 원문 문턱 판본은 **rate 축**(gate #16 이차 표적)에
남는다(rev3 §1 C1). (2) **`he2_bench.sbatch:92` 게이트 #7 버그 잔존분
해소**(커밋 `dadb851`, `dur=max(dur,d)`→`dur+=d`) — rev34가 "engine-
porter 이관 목록에 추가"라 적었던 항목이 이번 세션에 실제로 수정됨.
과거 `HE2_RESULT` 라인 영구 인용 금지는 **불변**, 정본 §1-19/§1-20
숫자는 이미 `sum(dur)` 독립 재계산으로 무사 확인돼 재확인 불요(§1-19
문구 갱신). (3) **방법론 게이트 2건**(둘 다 아래 §3): (a) 게이트 #9
**열한 번째 재발**(항목60) — 감사 대상이 아니라 **검증하는 쪽**
(메인 세션의 독립 검증 스크립트·claims-auditor 자신의 1차 권고 자기
반증)에서도 항등식이 재발. (b) **신규 #41 — telemetry는 공짜
관찰자가 아니다**(항목61) — `observe_scheduler`가 모든 sync마다
실행돼 `--disable-overlap-schedule`에서 임계경로 비용을 지불하고,
2026-07-15/18 sgptv 격자(HE0/HE2 헤드라인 다수의 근거)는 telemetry
없이 돌아 telemetry-ON 캠페인과의 절대값 직접 비교가 빌드 드리프트+
계측 오버헤드의 합이 됨 ⇒ ⚠️**기존 "긴장 A(HE2 vs C2)"가 정확히 이
패턴(telemetry OFF vs ON)이며 그 미통제 인자 목록에 telemetry 유무가
없었음을 목록으로만 등재**(정정은 다음 세션). (4) 부수: 커밋
`17fcac3`(제목 `gate #16 dynamic-vs-static grid harness`)은 G16이
dynamic-vs-static 실험이 아니므로(그건 HE0 트랙, rev3 §10-7이 명시
분리) 제목 오류 — 본문은 정확, history 재작성 안 함, `PROJECT_STATUS.md`
"다음 실험 gate" #11 G16 행에 문서 층 정정 메모만 남김. 상세
`handoff-report/session_handoff_2026-08-16.md` §14–19.** 이전
rev34: 2026-08-16 (doc-steward — **R1·R2 재분석 정본 승격
(`workspace/engine-port/results/slo_sched/ORACLE_REANALYSIS_2026-08-16.md`
rev2 = result-analyst 산출 + claims-auditor 적대 감사, GPU 0 · 새 서빙
실험 0건). 새 성능 판정 0건 · 등급 상향 0건 · 등재 내역 = provenance
확정 1건(R1, §1-19/§1-20) + 음성 결과 1건(R2, 신규 §1-32) + 코드 결함
1건(영구 인용 금지) + 방법론 게이트 2건.**
(1) **§1-19 "+2.1%" provenance 확정** — 독립 재현이 phase-mean
**+2.28/+2.35%**(정본 표기와 정합, "+2.1%"은 반올림 2자리 값끼리 계산한
결과였음이 확인됨)와 하네스 자신의 pooled trace-level **+5.88/+6.12%**
(phase B duration이 pooled 가중치의 ~78%)로 갈림을 확정하고, **정본
goodput 술어(TTFT≤3s ∧ 요청-내부 token-ITL p95≤60ms)로 재채점하면 phase
B가 5 arm×2 rep 전부 joint 0/192 완전분리 — 오라클 자체가 미정의**임을
확정(claims-auditor 1차 산출 그대로 재현, 독립 재확인 완료).
(2) **§1-20 "+16%"에 caveat 2건 추가** — (a) 정본술어로도 견딤
(**+16.54%**, legacy 재현 +16.23%)이나 ITL 도너가 d24/d34 완전 동률
(99.7396%)이라 **"116"은 tie-break 의존**(d34 택하면 126), (b) TTFT
임계 ±10%(41점 스캔)에서 **+0.00%~+19.75%로 비단조 요동**, 구조(도너
d16 vs d24+)는 41점 중 39점만 불변. (3) ★**신규 §1-32 — R2 음성 결과
등재**: `sgptv{Lo,Hi}` rate-swing 격자(n=4, 정본술어)에서 **"+16%" 크기는
재현되지 않는다**(어떤 재표집·부분집합에서도 최대 +5.64%) — 그러나
**"116>108 coupling tax가 없다"는 등재 금지**다: 결정량
`SM합>108 ⟺ D_itl>D_ttft`가 HI의 TTFT-argmax=격자 최대 decode arm(d44)
이라 **항등식**(검정력 0)이었다. (4) ★**방법론 게이트 #9 열 번째 재발 +
게이트 #18 사례**(둘 다 §3 항목59 신설) — rev1이 위 항등식을 증거로
제시했고, 그 항등식은 `PRIZE_SIZE_ARGUMENT_2026-08-16.md` §2.3(2)에
**이미 문자 그대로 적혀 있었다**(저장소가 이미 가진 진단을 자기 산출물에
적용하지 못함). (5) §3 항목39에 追記(재현 경로 결손 3회차, 이번엔 등재
시점에 닫힘) — `unpaired_bootstrap_ci` import-only·미호출·
`bootstrap_over_arms` 죽은 코드였던 rev1 결손이 rev2에서 실제 배선되고
claims-auditor 독립 재구현값과 일치 확인됨. (6) ★**코드 결함 영구
등재**: `he2_bench.sbatch:92`가 아직 `dur=max(dur,d)`(게이트 #7 버그
잔존, `sharegpt_vary_bench.sbatch:92`만 수정됨) ⇒ **`he2_*.out`의
`HE2_RESULT` 라인은 정확히 약 3× 부풀려져 있다 — 인용 영구 금지**(단
정본 §1-19/§1-20 숫자 자체는 `sum(dur)` 독립 재계산으로 무사함이
확인됨). `he2_bench.sbatch:92` 수정을 engine-porter 이관 목록에 추가.
"다음 실험 gate" #14(R1)·#15(R2)를 **"실행 완료"**로 갱신(#15는 부분
달성 — 결정량①만, 비-과부하 판본 미달성) + **#16(HI에 d54 이상 arm을
n≥4로 추가) 신설**을 최우선 gate로. 상세
`ORACLE_REANALYSIS_2026-08-16.md`(rev2), `PROJECT_STATUS.md` "다음
실험 gate" #14–16·"방법론 게이트" #40, `handoff-report/
session_handoff_2026-08-16.md`.** 이전
rev33: 2026-08-16 (doc-steward — **두 병행 세션 정합성 반영:
"상금 크기" 논증(`PRIZE_SIZE_ARGUMENT_2026-08-16.md`, rev3, claims-auditor
적대 감사 완료)을 §4 살아있는 문서 표에 등재 + 그 감사가 찾은 인용금지·
강등 전파 결손 5건(원 4건 + 코디네이터 지적 1건) 정정. **새 성능 판정
0건 · 정책 순위 변경 0건 · GPU 지출 0 · 어떤 claim의 등급도 변경 없음
— 표기·전파·정합성 정리만.** (1) §1-4 "7.24s" 인용금지를
`CLAIM_EVIDENCE_MATRIX.md` Claim C·`venue_positioning.md`(§0.1·C3)에
전파(원 미전파에 `PRIZE_SIZE_ARGUMENT` rev1이 실제로 걸렸었음, rev2에서
자체 정정). (2) §1-19에 §1-20과 동일 근거의 강등 배너 + 집계(phase-mean
vs pooled)·술어(legacy mean-ITL vs 정본 p95) 재작성 대상 표시 추가(숫자
자체는 R1 재분석 완료 전까지 미승격, "다음 실험 gate" #14 신설). (3)
§5-5·§1-17 본문에 스코프 배너 부착 — "천장 확정"이 뜻하는 것은 **달성된**
정책 계열(HE0)뿐, **달성 가능한 천장**이 아님(§3 항목57 신설). (4)
`EXPERIMENT_ROADMAP.md`의 국소 ε 밴드(16→24/44→92)에 인용정지 (a) 표기
부착. (5) `C2R_RESULTS_2026-08-16.md` §0·§7이 같은 날 §3 항목56(A)가
금지한 percentile CI를 caveat 없이 헤드라인으로 쓰고 있던 것을 발견·
역전파(t(5) 병기 + 과소피복 사유 + JSON 포인터 강등) — §3 항목58(신설)로
"인용금지는 원 아티팩트 문서로도 역전파하라"를 방법론 게이트화(#39).
"다음 실험 gate" #14(R1: he2 재채점 정식 등재)·#15(R2: sgptv
TTFT⊗ITL 분해) 신설, 게이트 #9 아홉 번째 재발 대응 양성대조 조건
2건 병기. §3 항목57(HE0⇏천장소멸)도 신설, 방법론 게이트 #38. 상세
`handoff-report/session_handoff_2026-08-16.md` §4-5,
`PRIZE_SIZE_ARGUMENT_2026-08-16.md`.** 이전
rev32: 2026-08-16 (doc-steward — **C2-R 캠페인 결과 등재:
M8·Ha8의 신규 점추정 2건 등재, 기존 값 교체 아님(대체값일 뿐 심판이
아니다) · C2 등급 무변경(CONFIRMED scoped) · 인용정지 (a)(arm별 ε·순위)·
(b)(깨끗한 셀 CI·"n=4") 둘 다 유효(해제 0건) · 기존 Δ·p값·크기 인용 셀
(Zamba2 r2 단일)은 한 글자도 안 바뀐다.** 사전등록 `PREREG_C2R_RULES_REV2_
2026-08-15.md`(rev2 GO) 집행 결과(jobs 883574=M8·883575=Ha8, 각 12부팅) —
result-analyst 분석 + claims-auditor 적대 감사 완료, 메인 세션은 운영
지표(부팅 성공·H7·실현률)만 직접 확인. **정본 인용 문구(claims-auditor
지정, 그대로 채택)**: "`r_M8(16) = 3.058`, `r_Ha8(16) = 3.114`(ctx1024,
realized SM16/SM92, `decode_bs=16`, job 883574/883575, gpu43, 1시간,
`n_indep=6` 부팅). 동반 구간은 **within-job 부팅 구간이며 재현
불확실성이 아니다** — 보수적으로 **t(5) [3.056, 3.060] · [3.077,
3.155]**를 쓰고, **job/node/날짜 축은 미측정(n=1)**임을 병기한다."
★**percentile 부트스트랩 CI는 정본 본문에 쓰지 않는다** — 감사자 실측상
**일관되게 과소피복**(M8 1.36×·Ha8 1.55× 더 좁음), 원자료 JSON 포인터로만
남긴다. seed∈{1,2,3,99} SD 변동 ≤1.5%(부트스트랩 seed는 결과를 만들지
않음)이나 이는 **재표집 잡음**만 배제할 뿐 **재표집되는 모집단**(boot
단위, job/node/day 축 부재)의 문제는 그대로다. (2) Ha8은 실현률 0.80에
12/12 미달(arm의 성질로 보임, 감사 N5 미해결). (3) ★★★양성대조가
**항등식**이었다(방법론 게이트 #9 **아홉 번째 재발**) — 대조는
헤드라인이 안 쓰는 코드 경로만 실행했고 표적값 자체가 같은 루프의 또
다른 복사본 산출물, 공백을 메운 것은 claims-auditor의 독립 재구현
— 단 ★그 재구현 코드(`indep.py`·`legacy.py`)는 **스크래치에만 있고
저장소에 없다**(재현 경로 미보존, 2026-08-14 E-1a errata와 동형).
(4) 엔진 빌드 정정 — 865493 대비 매니페스트 11→15 파일, hot path 3종
추가(`scheduler.py`·`holb_probe.py`·`zamba2.py`). (5) guard 고원 —
채택 구간이 전부 미관측 창 안이라 (16,16)은 보간. **등재 금지**:
"C2-R이 2.36–2.91×보다 위"(범주 오류, batch/job 분해 불가)· 인용정지
(b) 해제(범주 오류, T8·Hs8 미재측정)·N1/N2 해결(교차-job 대조 0건,
미해결)·C2 등급 변경. 상세 §3 항목56(신설)·18(追記). 이전
rev31: 2026-08-16 (doc-steward — **새 성능 판정 0건 · 정책 주장
0건 · 기존 Δ·p값·크기 인용 셀(Zamba2 r2 단일)·등급은 한 글자도 안 바뀐다
— 등재 3건뿐: (a) C2-R 사전등록(rev1 NO-GO/rev2 GO)을 `PROJECT_STATUS.md`
"다음 실험 gate" #11 레지스트리에 추가(CONSENSUS 신규 항목 없음, 레지스트리는
PROJECT_STATUS 전용이라는 기존 관례 유지), (b) §3 항목55(신설) + 방법론
게이트 #37 신설 — "적대 감사에는 합격 기준과 단일 판정 질문을 함께
줘라"(⚠️메인 세션 진단, claims-auditor 감사 없음 — "현재 작업가설"로
인용), (c) §3 항목50 addendum3(신설) — 붕괴 keepalive 프롬프트 토큰 수를
"1793"에서 실측값 **1794**로 표기 정정(1792 초과·HTTP 400 전량 거부라는
결론·기전은 완전히 불변), 같은 정정을 §1 항목29(B-1)에도 반영.** 이전
rev30: 2026-08-15 (doc-steward — **새 성능 판정 0건 · 정책 주장
0건 — result-analyst의 C2 헤드라인 job 구성 감사를 claims-auditor가
적대 검증(완료, 반증 3건 포함), 기존 Δ·p값·크기 인용 셀(Zamba2 r2
단일)·등급은 한 글자도 안 바뀐다.**
(A) **rev29 자신의 서술 오류 정정**: rev29 (A)·`PROJECT_STATUS.md` G-1·
`NOTES_D54_ANCHOR_2026-08-03.md`가 job 865533의 keepalive 붕괴를
**"Ha8 arm 한정"**으로 적었으나, 원자료는 **4 arm(Ha8·Hs8·M8·T8) 전
20셀 전부**임을 보인다(셀당 거부 23,404–23,741건, `keepalive_done=0`).
어제 등재한 서술이 바로 다음 감사에서 범위 오류로 드러난 사례 — 정정
자체와 함께 "등재 직후 재발" 사실을 기록한다(§3 항목50 追記2). 이 A건은
claims-auditor 적대 검증의 대상이 아니었고(B/C만 검증 대상) 그대로
유지된다.
(B) **C2 헤드라인(2.36–2.91×, 4 arm) job provenance 감사 —
claims-auditor 적대 검증 완료, 판정=등급 유지 + 인용 정지 2건
신설(§3 항목54).** **생존(반증 실패, 그대로 인용 가능)**: 헤드라인
4셀 중 Ha8·M8·Hs8은 양 다리 100% job 865533 단독, T8도 SM92 다리는
865533 단독(1차 원인=하네스 `t0_monotonic_s` 결손, keepalive 붕괴
아님); 58 매칭 셀의 조건부 ITL 차 −0.18%±0.62%(최대 2.20%)는
claims-auditor의 3중 추가 반증 시도(헤드라인-다리만 재집계·`tok_idx`
매칭·`pf_bs` 매칭)에서도 살아남았다. **반증됨**: (i) "깨끗한 런
단독 재계산 T8 2.388×·Hs8 2.687×를 CI·n=4와 함께 인용 가능" —
T8 b16의 865493/d16 rep1–4가 실은 **서버 부팅 1회**(다리당
`n_indep=1`, rep는 의사반복)라 CI는 실제로 약 2배 넓다([2.368,2.408]),
점추정만 `n_indep=1` 명시 하에 인용 가능. (ii) "Ha8은 b12가 양 다리
0건이라 FINDINGS 지침을 구조적으로 만족 불가" — 귀속 오류, SM92
다리는 오히려 깨끗한 런(865493)이 b16까지 도달한다(33,344건); 사실은
"865533에서 미도달, 865493은 SM92에서 b16 도달"뿐. **신규 발견(감사가
추가)**: arm별 ε 순서(T8<Hs8<Ha8<M8)는 슬라이스 산물 — 4 arm 공통
batch(b=1)에서는 순서가 완전히 뒤집힌다(Ha8<Hs8<M8<T8, 폭
0.118→0.038) ⇒ **arm별 ε·arm 간 순위 인용 정지**. T8/M8/Hs8은 b12,
Ha8은 자기 최대 b9를 쓰는 **이질적 batch 추출 규칙**이었다는 것도
확인. **종합 판정(claims-auditor)**: "레버 존재, 2.3–2.9× 대역,
4 arm 전부"는 **CONFIRMED(scoped) 유지**, 단 **인용 정지 2건 신설**
(arm별 ε·순위 / 깨끗한 셀 CI·"n=4" 표기). 상세 §3 항목54.
(C) **게이트 #12 ctx 스코프 서술 — REFUTED, 재정정(§3 항목51 追記).**
result-analyst의 원 addendum("수치는 ctx4096 한정, 기전은 ctx-무관")은
claims-auditor의 독립 재집계로 **반증**됐다 — ctx1024의 SM92
max(decode_bs)는 865493 21/23/21/23·865533 16(전 arm)으로 ctx4096의
4/12보다 훨씬 높아, ctx1024 자체가 "기전이 ctx-무관하게 항상
구속한다"는 서술의 **반례**다. **정정된 명제**: B-폐쇄는 **λ(L)이
작을 때만 구속한다**(prefill 서비스율 λ가 prompt 길이 L의 함수이고,
L이 짧으면 λ가 커 폐쇄가 안 걸릴 수 있다) — "ctx-무관"이 아니라
**regime-의존**. 상세는 이 문서 §3 항목50(追記2)·51(追記, 재정정)·
54(신설, 재정정), `PROJECT_STATUS.md` "8B decode-SM 민감도 측정
노트"(재정정)·"다음 실험 gate" #12(재정정)·G-1(追記2),
`workspace/engine-port/results/s8_scaleup/
AUDIT_C2_HEADLINE_JOB_COMPOSITION_2026-08-15.md`(1차 산출,
result-analyst — 위 반증 3항목 포함 원문 보존), claims-auditor 적대
검증(원자료 파일 위치 미확정, 다음 세션 편입 요망),
`NOTES_D54_ANCHOR_2026-08-03.md`(Addendum 2, A건만 대상), `reports/
paper/CLAIM_EVIDENCE_MATRIX.md`(Claim A 각주, 재정정). ⚠️**provenance**:
B/C는 **result-analyst(1차) + claims-auditor(적대 검증) 산출**이고
**메인 세션(doc-steward)은 독립 재확인한 것이 없다**. 이전 rev29:
2026-08-15 (doc-steward — **새 성능 판정
0건 · 정책 주장 0건 — 세션 종료 정본 반영 4건, 기존 Δ·p값·크기 인용
셀(Zamba2 r2 단일)·등급은 한 글자도 안 바뀐다.** (A) keepalive 오염이
§3 항목50 caveat(ii)를
구체화(addendum, §3 항목28과 대응 — 새 사실 아니라 기전 특정)+
`NOTES_D54_ANCHOR_2026-08-03.md`/`PROJECT_STATUS.md` G-1의 "byte-identical/
engine-tree churn" 서술에 dated 정정(mtime 증거로 865493이 그 파일을 쓸 수
없었음을 확인). (B) decode batch 도달성의 구조적 폐쇄를 실험 게이트로
등재(§3 항목51, **claims-auditor 산출·메인 세션은 T8 행 하나만 재확인**,
provenance 명시). (C) ncu/nsys 도구 사실 4건으로 "커널 단위 측정 0건"에
운영점·green-ctx 스코프 주석(§3 항목52). (D) 방법론 게이트 #35·#36
신설(§3 항목53). 상세는 이 문서 §3 항목50(追記)·51·52·53(신설),
`PROJECT_STATUS.md` "다음 실험 gate" #12(신설)·"8B decode-SM 민감도 측정
노트"(스코프 주석 추가)·"방법론 게이트" #35·#36, 메모리
`deconfound-measurement-lessons.md` 항목35·36·`scale-8b-sm-sensitivity.md`,
원자료 `handoff-report/session_handoff_2026-08-15.md`. 이전 rev28:
2026-08-14 (doc-steward — **새 성능 판정 0건 — E-1a
Tier 2 처리(claims-auditor 판정 T2-1 조건부 채택 §3 항목50·T2-2 보류
[사전등록 addendum]·T2-3/T2-4 기각[T2-4는 §1-25 追記로 재확인만])
+ RESUME.md push 프레이밍 정정, 기존 Δ·p값·크기 인용 셀(Zamba2 r2
단일)·등급은 한 글자도 안 바뀐다.** 상세는 이 문서 §3 항목50(신설)·
§1-25(追記), `PROJECT_STATUS.md` "다음 실험 gate" #11(레지스트리 자체는
불변, T2 시리즈는 그 NO-GO 판정을 뒤집지 않는다), `workspace/engine-port/
results/bsweep_regime/PREREG_E1_BSWEEP_REGIME_2026-08-14.md`(addendum,
사전등록 자체는 여전히 미제출)·`E1A_ARTIFACT_ERRATA_2026-08-14.md`(신설)·
`workspace/engine-port/RESUME.md`(Git push 이력 절 정정, CONSENSUS
비대상). 이전 rev27: 2026-08-14 (doc-steward
— **새 성능 판정 0건 — 정본 결함 정정 2건 + Tier 1 등재 5건, 기존
Δ·p값·크기 인용 셀(Zamba2 r2 단일)·등급은 한 글자도 안 바뀐다.**
**(A) ε 표기 정정** —
`PROJECT_STATUS.md` "8B decode-SM 민감도 측정 노트"의 "C2 구간 평균
ε≈0.48–0.56"이 변환식(`ε=log(ratio)/log(92/16)`)과 불일치했다(정답
[0.491,0.611]). 2026-08-14 E-1a(`workspace/engine-port/results/
bsweep_regime/e1a_preanalysis_2026-08-14.json`)의 원자료 직접
재산출(T8 0.492·Hs8 0.543·Ha8 0.599·M8 0.610)이 변환식 재계산과
일치해 **ε≈0.49–0.61로 정정**한다. 옛 값은 오히려 **인용 금지된
SM16/SM108 열**(비 2.60–2.96×, `FINDINGS_8B_2026-07-28.md` §2)을
`log(108/16)`로 변환한 [0.488,0.568]과 더 가까워 **claims-auditor
가설(인용 금지 열의 수치 혼입)이 지지된다.** ITL 비 2.36–2.91×(C2
CONFIRMED scoped) 자체는 **불변** — 바뀐 것은 파생 ε 표기뿐이다.
같은 정정을 `reports/paper/CLAIM_EVIDENCE_MATRIX.md`의 동일 인용에도
반영(Tier 2 전파, 원문 미덮어쓰기·괄호 addendum). **(B) Ha8 B-라벨
정정** — `TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md` §3.4의 Ha8 행이
`B` 열에 9를 적으면서 `B×KV`/`B×2·state`는 **B=12로 계산돼 있었다**
(검산: `5.734=12×0.4779`, B=9라면 4.301이어야 함). 행 라벨(B=9,
§5·§6.1·§6.5와 일치하는 실제 측정 B)로 재계산하면 Ha8 weight share는
70.1%가 아니라 **75.80%**다. §6.1의 Ha8 924 GB/s는 이미 **올바른
B=9 트래픽(29.10 GB)**을 쓰고 있어 **영향 없음**(총 31.449 GB로
계산하면 998.7 GB/s가 나와 다르다) — §3.4와 §6.1이 서로 다른 B를
쓰고 있던 **문서 내부 불일치**였다. 헤드라인 "스텝 트래픽 68–94%가
weight"(하한 M8 68.6%)·achieved_BW 하한 45.4%(=924/2039, rev26)·
그 밖의 어떤 판정도 **불변**. §12(신규 addendum)로 원문 미덮어쓰기
정정. **(C) Tier 1 등재 5건**: (1) **방법론 게이트 #34(신설, §3
항목49)** — "규칙 먼저 감사, 하네스는 그 다음" 2단 규율: 2026-08-14
사전등록 4건(LTSM P1·E1-b/c·`%smid` R0·E-1) 감사 결과 3 NO-GO·1
CONDITIONAL-GO, 차단 결함이 예외 없이 하네스/스코어러 구현 층이었고
셋(LTSM P1·E1-b/c·E-1)은 **도구 자신이 거짓 음성/양성을 산출**했다
— E-1은 메인 세션 자신이 작성해 같은 결함을 냈다는 점에서 개인
부주의가 아니라 **구조적 실패 모드**임을 보인다(메모리
`deconfound-measurement-lessons.md` 항목34와 대응). (2)
`PROJECT_STATUS.md` "다음 실험 gate" #11에 **미제출·감사 차단
사전등록 4건 레지스트리** 등재(목적=다음 세션의 재제출 방지, "가설
반증"으로 오독 금지). (3) engine-porter 이관 신규 3건 — `sgl_kernel/
spatial.py`(venv본) 래퍼가 `res[2]/res[3]`(realized SM 수)를 버림
(C++ `greenctx_stream.cu:84-97`은 반환), `smid_l0_census.py:410`
(`census_kernel.cache[dev]`, 존재하지 않는 속성 추정)이 죽은 코드에
`:523/:529` 가드가 `None`을 fail-open, `g2s_e1b_premise.py:295-333`이
`M<36`에서 `frac`을 조회하지 않음 — 기존 이관 4건은 여전히 미해결임을
재확인만(코드 미수정). (4) **C2 `sd_rep(ε)` 실측값으로 수입값 교체**
(방법론 게이트 #32의 새 사례, §3 항목46 追記) — "rep 간 sd 0.01–0.07"을
E-1a 원자료 직접측정(44→92: Ha8 **0.0102**·T8 **0.0083**)으로
대체한다, ITL 수준 rep CV는 0.2–0.7%. (5) `%smid` R0 §0.1 payoff
항목1(`log(108/34)` 분모 정정)은 **정정 대상 없음이 이미 rev26에
반영**돼 있음을 재확인(중복 등재 안 함). 상세 `PROJECT_STATUS.md`
"8B decode-SM 민감도 측정 노트"·"다음 실험 gate" #11·"방법론 게이트"
#32(追記)·#34(신설), §3 항목46(追記)·49(이 문서), `reports/paper/
CLAIM_EVIDENCE_MATRIX.md`, `TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md`
§12(addendum), `workspace/engine-port/RESUME.md`(운영 사실 — git push
이력·병렬 세션·PAT 노출, CONSENSUS 비대상이라 여기엔 판정 없음).
이전 rev26: 2026-08-14 (doc-steward — **새 성능 판정 0건 —
정본 정정 1건 + 신규 결과 등재 1건, 기존 결론은 뒤집히지 않는다(오히려
강화 방향).** **(A) 하드웨어 오식별 정정** —
`workspace/engine-port/results/s8_scaleup/TRAFFIC_ROOFLINE_DIAGNOSTIC_
2026-08-11.md`(§6.2·§6.5·§9·부록A)가 `nvidia-smi -q`로 하드웨어를 읽은
노드가 **로그인 노드(glogin01, `NVIDIA A100 80GB PCIe`)**였는데, 실제
측정 캠페인(jobs 865289/865533)은 **컴퓨트 노드**(`sacct`로 확인한
gpu36·gpu38·gpu40)에서 돌았고 컴퓨트 노드는 **SXM4**다(job 882374가
gpu43에서 `torch.cuda.get_device_name()`=`NVIDIA A100-SXM4-80GB` 직접
관측, `scontrol show node`로 gpu36·gpu40·gpu43 동일 feature
`A100-80GB_8,hwperf` 확인 — 파티션 동질). `s8_scaleup/`의 job 아티팩트
전체(`*.out *.log *.err *.json *.jsonl`)에 `"A100 80GB PCIe"` 문자열은
**0건** — 하드웨어 식별이 측정이 일어나지 않은 기계에서 읽혀 컴퓨트
노드 결과 문서에 들어갔다. **정정**: 사양 BW **1935→2039 GB/s**(SXM4
mem clock 1593 MHz), achieved_BW 비율 **48–61%→45.4–57.7%**(×0.949),
ridge point **161→153** FLOP/byte, decode AI≈8.2는 ridge의
**5.1%→5.4%**. **어떤 판정도 뒤집히지 않는다** — "고-SM 평탄화를 HBM
포화로 서술 금지"의 근거는 비율이 더 낮아져 **더 강해지고**,
"compute-bound 아님"도 불변. 방법론 게이트 #32(항목46)에 **새 사례로
追記**(숫자 층이 아니라 하드웨어 식별 층, #32를 만든 바로 그 문서가
같은 종류의 오류를 두 번 냄) — 실무 규칙 추가: 하드웨어 사양은 측정이
실제로 실행된 노드에서 읽어 job 아티팩트에 기록하라(로그인 노드와
컴퓨트 노드가 다른 SKU일 수 있다 — 이 클러스터가 실제로 그렇다).
**(B) 신규 결과 등재** — E-3 realized SM count 프로브(사전등록
`workspace/engine-port/results/smsplit_realized/
PREREG_SMSPLIT_REALIZED_2026-08-14.md`, GPU 비용 ≈0): `pdmux_context.
divide_sm()`으로 얻은 `(74,34)`·`(54,54)`·C2 스윕 5지점(92/16, 84/24,
64/44, 54/54, 16/92) 전 7지점에서, green-context 원시함수의 realized
반환값이 요청값과 **정확히 일치**했다(Δ=0, `realized_sum=108` 전 대상)
— glogin01(PCIe)·컴퓨트 노드(job 882374, SXM4) 두 하드웨어에서 레코드
완전 일치, 판정 `REQUEST_EQUALS_DRIVER_REPORTED_PARTITION`. **허용
문장**: "드라이버가 보고하는 green-context 파티션은 이 격자에서
요청값과 일치하며 반올림이 없다(드라이버 자기보고 층)." **금지**: 이는
**드라이버 자기보고**이지 하드웨어 실행 층이 아니다 — §1-1 Gate 1
블록의 기존 인용 금지("실현 파티션을 **측정**했다" 등)는 **그대로
유지**되고, `%smid`의 SM id 집합 disjointness 질문은 전진 0, 기존
캠페인 판정문에 사후 부착 금지, 성능·정책 주장 0건. 부수 함의: `%smid`
R0 §0.1의 `log(108/34)` 분모 정정 payoff는 정정 대상이 없음이 확인(34는
요청값이자 드라이버 보고 realized 값), F2 "다섯 번째 세계"는 개수
층에서는 관측되지 않음(id 집합 층 미측정). 이 프로브가 §(A)의 하드웨어
오식별을 발견한 결정적 근거다. **신규 방법론**: §3 항목46에 새 사례
追記(하드웨어 식별 층 재발). 상세 `PROJECT_STATUS.md` "8B decode-SM
민감도 측정 노트"·"확정된 결과" 1번(E-3 블록)·"방법론 게이트" #32,
§1-1(이 문서, E-3 블록)·§3 항목46(追記), `TRAFFIC_ROOFLINE_DIAGNOSTIC_
2026-08-11.md` §11(addendum)·`FINDINGS_8B_2026-07-28.md` §7 C-5
(addendum)·`workspace/engine-port/results/smsplit_realized/
PREREG_SMSPLIT_REALIZED_2026-08-14.md`(addendum 2)·`workspace/
engine-port/RESUME.md`(환경 사실).
이전 rev25: 2026-08-14 (doc-steward — **새 성능 판정 0건.
문서 층 정리 — 기존 Δ·p값·크기 인용 셀·등급은 한 글자도 바뀌지
않는다.** (1) **§3 항목47(신규)**: 방법론 게이트 #26("배관 스모크")의
발동 조건이 "캠페인" 단수라 <1 GPU-hr 조각으로 쪼개면 회피된다는
구멍 — 2026-08-11 세션이 검토만 하고 제출하지 않은 GPU 실험 4건
(P1/E1-b·c/`%smid` P1+P2/G1-a, 합 ≈1.15–1.65 GPU-hr, 개별로는 전부
문턱 아래이며 같은 매니페스트 트립와이어를 공유해 독립이 아님)에서
확인 — 트리거를 "한 배치로 제출되는, 신규/변경 코드를 공유하는
캠페인들의 합"으로 개정 권고(追記, 원 게이트 대체 아님). (2) **§3
항목48(신규)**: `sync_engine_tree.sh`가 sha256으로 해싱하는 파일은
정확히 **15개**뿐이라 "매니페스트 N/N sha 일치"는 그 15파일의 바이트
동일성만 보증한다 — `pdmux_context.py`(`(74,34)` 등 파티션 기대값의
출처, `divide_sm()` 정의)·`sgl_kernel/spatial.py`(green-context
원시함수)는 매니페스트 밖이고, `src/patches/`의 패치 5개 중 2개
(`nemotron_h_forward_split_prefill.patch`·
`triton_backend_mambaish_vheaddim.patch`)는 어느 실행 경로에서도
적용되지 않으며, `env/dev_tree_edits.md` 항목 3·4·5·7(수동 편집)·
6·8·9(스크립트 자신이 "still manual copies"라 자백)도 매니페스트
밖 — Gate 1/`gate1b`/`gate1c`/Gate 2-S 트립와이어의 "N/N sha 바이트
동일" 재현성 주장 범위를 이 15파일로 명시적으로 좁힌다(**기존
판정을 뒤집지 않음** — 커버 밖 드리프트가 실제 있었다는 증거 없음;
"패치 미적용"도 "그 기능이 런타임에 없다"를 함의하지 않음, 수동
편집으로 이미 반영됐을 수 있어 확인된 것은 "sync가 보장하지 않는다"
까지). (3) `PROJECT_STATUS.md` 방법론 게이트 **#9·#25 본문**에
섹션 헤더가 이미 명시했던 追記 2건(§3 항목18 별건 사례·항목39
여덟 번째 재발)이 누락돼 있던 정본 결함을 이 문서 원문 대조로
복원(핸드오프가 1건이라 보고했으나 실제 2건이었음). (4)
`workspace/engine-port/RESUME.md`(RESUME.md는 프로젝트 루트가
아니라 이 경로에만 있음)의 CPU 회귀 지시에 노드 구분·소요 시간을
반영 — 로그인 노드에서 140 tests OK/73초(대부분이
`import sglang.srt.managers.scheduler` 70.6초), "컴퓨트 노드
필수"는 재현 안 됨, 과거 7분 stall은 Lustre 콜드캐시 추정(코드
사실 아님), `g2s_run.sbatch:91`이 이미 컴퓨트 노드에서 전체
회귀를 차단 게이트(`exit 5`)로 돌린다는 사실 병기. (5) 2026-08-14
커밋 `31b3e96`/`86179ec`(KISTI `--comment` 정책 저장소 전체
적용 + conformance checker)를 `PROJECT_STATUS.md`에 신규 "운영
규약" 절로 등재(`RESUME.md`에는 이미 있었음). 상세
`PROJECT_STATUS.md` "운영 규약"·"방법론 게이트" #9·#25·#26·#33,
`workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh`·
`workspace/engine-port/RESUME.md`·
`handoff-report/session_handoff_2026-08-13.md` §2.5·§4.
이전 rev24: 2026-08-11 (doc-steward — **트래픽·roofline 진단
(`../workspace/engine-port/results/s8_scaleup/
TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md`, result-analyst, GPU 0 —
기존 아티팩트+모델 config 계산만) 정본 반영. 새 성능 판정 0건,
C2("decode SM 민감도 2.36–2.91×, scoped")의 등급·수치 불변 —
기존 주장이 더 약화되고 스코프가 더 좁아지는 방향.** 계기는
사용자가 "4 arm이 왜 비슷한 SM 민감도인가/hybrid는 KV가 적을
텐데/mamba 층을 제대로 재나/A100 대역폭이 weight traffic으로
제한된다는 게 납득 안 된다"고 제기한 반박이며, 그중 2건이 계산으로
확인되고 메인 세션의 "교차점(L\*) 가설"은 반증됐다. **핵심**: (1)
`FINDINGS_8B_2026-07-28.md` §2.1의 "1차 weight-traffic 추정(M
5.40/T 6.17/H 7.66 GB, H/M=1.42)"이 실은 **3B급 다른 캠페인의
수치**이고 기준(체크포인트 바이트 vs 호출-인지 트래픽)도 혼합돼
있었음을 체크포인트 실측 3자리 일치로 확정 — 이미 NOT-YET-
SUPPORTED인 C2b(hybrid 급락=Zamba2 성질)를 **더 약화**(되살리지
않음). (2) "decode SM 민감도는 모델-무관"의 원인이 아키텍처
동질성이 아니라 이 측정점(B≈9–12·L≈1.0–1.5k)에서 스텝 트래픽의
68–94%가 weight-sweep이기 때문임을 규명 — B/L 확장 이식 금지.
★**"hybrid는 KV가 적다"는 통념은 이 격자에서 거짓**(per-seq 캐시
Ha8 601 MiB > M8 260 > Hs8 117 > T8 70 MiB, Ha8이 T8의 8.6×). (3)
SM92에서도 achieved_BW가 사양 대역폭(A100 80GB **PCIe** 1935
GB/s — SXM 2039 아님)의 48–61%뿐이라 **고-SM 평탄화를 HBM 포화로
서술하는 것 금지**(진짜 원인 미식별); decode 축(AI≈8.2,
memory-bound) 결론을 prefill 축(AI≈1035, compute-bound)으로
이식 금지. 신규 방법론 항목 2건(§3 항목18에 일곱 번째 재발
追記[roofline 탄력도 정합=항등식, 산출자 자수]·항목46 신설[타
캠페인 보조 수치는 기준 검증 후 수입]). 상세 `../PROJECT_STATUS.md`
"8B decode-SM 민감도 측정 노트"·"방법론 게이트" #9·#32, §3
항목18·46, `FINDINGS_8B_2026-07-28.md` §7.
이전 rev23: 2026-08-11 (doc-steward — **E1 addendum(Gate 2-S
전제 결정량, jobs 877756/877757 재집계, GPU 0) 정본 반영 —
claims-auditor 적대 감사 완료, 4셀 전부 `VERIFIED_AT_SAMPLED_
INSTANTS`. 판정 = 승격이 아니라 등급 하향된 조건부 채택.** E1이
자신의 §2.5에서 주장한 "G1-b/G1-c보다 밀도 페널티를 피했다"는
**반증**됐다 — 결정 관련 pop-A 관측 수는 오히려 G1-b/G1-c 대비
5–6× 적다(1,279·1,496 vs 6,402·8,920, 메인 세션이 원 telemetry에서
pop-A[decode_bs>0 ∧ prefill_active_bs>0] 건수를 독립 재계산해
확인). E1의 실질 우위는 밀도가 아니라 **반복수(n=1→10)와 셀
일치**뿐이다. 채택은 조건 7개 전부(하나라도 누락 시 무효) 하에서만
유효 — 밀도 정정 병기·bound 병기·복합 규칙 분할 출처 명시·post-hoc
자백 4셀 전부·§5.6.1 불변 재확인 등, §1-1 전문 참조. 새 적대적
bound(`sup decode_bs ≤ in-system+Poisson(λ·dt)`, 새 자유모수 0,
q=1e-9) = 25/41/24/27 — **Zamba2 r3는 q=1e-6에서 이미 37≥36으로
문턱을 배제하지 못하는 유일한 셀**(같은 모델 G1-b rate6 실측
초과[max40]·이 캠페인 자신의 capscan[max31] 병행 근거, 메인 세션
독립 재확인). **성능 결론은 한 글자도 안 바뀐다**(Δ=+13.95/
+24.88/+16.04/+19.11ms·4셀 Holm 후 최대 p=1.88e-06·9-셀 `S1-C`·
크기 인용 셀 Zamba2 r2 하나 — `premise`는 `nine_cell`·`gate_label`·
`paired_t`[`g2s_analyze.py:1153-1159`] 어디에도 미입력, 코드 확인).
**무료 대조(감사자 등재)**: T′의 pooled max(decode_bs)=13/23/10/12가
A4의 14/23/10/13과 ±1 이내로 일치(메인 세션 telemetry 직접
재계산) — 전제가 의존하는 부하 동등성의 직접 증거인데 addendum
자신은 산출하지 않았다. **별건 발견(engine-porter 소관, 코드
수정 안 함)**: `g2s_analyze.py:84-89`의 `PREMISE_LABEL`과 아카이브
`g2s_report_granite-40-h-micro-base_877757.json`의 `premise_labels`가
여전히 Granite r3·r4를 `"unverified"`로 고정 — G1-c/E1 이전 상태의
동결 스냅샷임을 정오표로 명시하고, 재현성 보존을 위해 스코어러는
지금 패치하지 않는다(향후 재실행 시 갱신 요구사항으로 등재).
신규 방법론 항목 2건(§3 항목44·45) 신설 + 항목18·39에 재발
追記(각각 게이트#9·#25 새 사례). 상세 §1-1(2026-08-11 E1
addendum 블록)·§3 항목44·45, `PROJECT_STATUS.md` "확정된 결과"
1번·"방법론 게이트" #30·#31. 원자료 `workspace/engine-port/
results/p1_gates/gate2/{PREREG_G2S_E1_ADDENDUM_2026-08-11.md,
g2s_e1_premise.py, g2s_e1_premise_877756_877757.json}`(수정
금지·인용만).
이전 rev22: 2026-08-11 (doc-steward — **G1-c(job 877974, 0.10
GPU-hr) 정본 반영 — Gate 2-S Granite r3·r4 §5.6.1 명명 제한
조건부 해제. 새 성능 판정 아님, "크기 인용 셀 확대" 아님.**
세션 초반 "전제 VERIFIED가 되면 크기 인용 가능 셀이 1→3개로
는다"는 서술은 원자료·코드 대조로 **반증**됐다 — 크기 인용을
막는 것은 `premise` 라벨이 아니라 독립 산출되는 F-계열 gate이고
(`g2s_analyze.py:1157-1161`, 유일 소비처는 `name_for()`), 둘은
서로 다른 값이다. **이번 승격이 바꾸는 것은 명명 층 하나뿐**:
Granite r3·r4에서 "간헐 전달(엔진 기본 궤적)"·"A4형" 명명이
필수조건 6건 충족 하에 허용되고, **크기 인용 가능 셀은 여전히
Zamba2 r2 하나**다(불변). rev20의 Δ=+13.95/+24.88/+16.04/
+19.11ms·4셀 Holm 후 최대 p=1.88e-06·9-셀 `S1-C` 문장은 **한
글자도 바뀌지 않는다**(premise는 그 산출 경로에 미입력). 같은
회차에 `PREREG_GATE2S_2026-08-09.md` §8.9의 Zamba2 근거표가
`gate1/PREREG_G1B_2026-08-07.md:146`이 인용 금지한 pop-C
시간가중 분수를 인용하고 있던 기존 정본 결함을 사후 addendum으로
정정(원문 미덮어쓰기)했고, 신규 방법론 항목 3건(§3 항목41–43)을
등재했다. 상세 §1-1(2026-08-11 G1-c 블록)·§3 항목41–43,
`PROJECT_STATUS.md` "확정된 결과" 1번·"방법론 게이트" #27–29.
원자료 `workspace/engine-port/results/p1_gates/gate1/
{PREREG_G1C_2026-08-11.md, gate1c_result_877974.txt,
gate1c_analyze.py, runtime_source_manifest_gate1c_877974.sha256,
manifest_diff_gate1c_877974.txt}`(수정 금지·인용만).
이전 rev21: 2026-08-11 (doc-steward — **§3 항목40 신설[대형
캠페인 제출 전 배관 스모크 규율, PROJECT_STATUS "방법론 게이트"
#26과 대응] — 새 성능 판정 아님, Gate 2-S 1차 실행 실패(jobs
877107/877109, 6.40 GPU-hr, primary 0개)와 그 재발을 막은 배관
스모크(job 877593, 0.11 GPU-hr)로부터의 방법론 등재.** 상세는
§3 항목40 본문 참조. **동시에 `reports/paper/`(`CLAIM_EVIDENCE_
MATRIX.md`·`EXPERIMENT_ROADMAP.md`·`DOCUMENT_STATUS.md`)를
rev14–rev20 구간(2026-08-06~2026-08-11)과 동기화했다** — Claim
A–F 행 자체는 무변경(각 rev가 반복 확인한 대로 이 구간이 그
행들에 인용되지 않는다, 아래 rev14–20 기록 참조). 대신 그 세
문서에 지금까지 없었던 **"P1 트랙"**(PD-mux 자체 vs fused, Gate
1/G1-b/Gate 2 rev4/E-A/R2′/Gate 2-S) 절을 신설해 rev14–20의 현재
판정과 Gate 2-S 제한 7건을 이관했다 — 이 동기화 자체는 CONSENSUS의
새 결론이 아니라 상위 정본(reports/paper/)이 하위 정본(이 문서)을
누락 없이 반영하도록 맞춘 것이다. 상세는 각 문서 changelog.
이전 rev20: 2026-08-11 (doc-steward — ★★★★★★★**Gate 2-S 첫
유효 결과(jobs 877756/877757, 6.35 GPU-hr) — claims-auditor 적대
감사 "조건부 등재 가(可)". §1-1의 기존 NOT-YET-SUPPORTED를
대체하지 않는 별도 트랙(Gate 2-S)의 산출이며, 2026-08-10에 넣은
"귀속 스코프" 주석과 정합하게 배치한다.** 둘 다 exit 0 /
`MEASURED_AND_SCORED`, `design_conformance.conformant=True`(전
셀 n=10, 양 rate, 5 arm), 19/19 블록. **이력(반드시 병기)**: 1차
실행(jobs 877107/877109, 6.40 GPU-hr)은 하네스 결함 6건(§2.7,
`session_handoff_2026-08-10.md`)으로 사전등록 primary를 **0개**
냈다 — 사전등록 설계는 무결했고 실패는 전부 하네스 층이었다.
`PREREG_GATE2S_2026-08-09.md`(rev6)는 claims-auditor 설계 감사
**4회**(NO-GO 3 → GO-with-changes)를 거친 것. 채점 result-analyst
(원자료 200행 독립 재계산), 감사 claims-auditor(원자료에서 α 독립
재현, 소수 3자리 일치 — 메인 세션이 g2s_report_*.json 원자료로 핵심
수치(Δ 4값·Holm p·Fieller 3.098·premise verified/unverified·
frac_idx1) 추가 독립 확인, 전용 감사 아티팩트 파일 위치는 미확인).

★**정본에 넣는 문장(감사자가 확정한 허용 범위, 축약·확장 금지)**:
이 엔진·이 격자(Zamba2-2.7B r{2,3} triton / Granite-4.0-h-micro-base
r{3,4} flashinfer, in2000/out96, A100 108 SM, cudagraph ON, n=10
paired, jobs 877756/877757)에서, pdmux 서브시스템·이벤트 루프·
split-prefill·역할별 backend를 고정한 채 `PDMUX_R2_FIXED_DSM`만
108→34로 바꿔 SM을 `(74,34)`로 분할하면, **요청-내부 ITL p95의
평균(α)이 4셀 전부에서 감소**하고(Δ = C−T′ = **+13.95/+24.88/
+16.04/+19.11 ms**, 4셀 Holm 후 **최대 보정 p = 1.88e-06**,
β=pooled p99도 부호·유의성 일치) **같은 4셀에서 TTFT p95는
악화**한다(−183 ~ −536 ms) — 9-셀 표 좌표는 16블록 전부
**`S1-C`**(트레이드오프)다. Zamba2 r2·r3는 "`FixedPolicy(34)`·
sticky OFF = 엔진 기본 궤적" 명명이 허용되고(전제 검증됨, Gate 1
rev15/job 875293), **Granite r3·r4는 그 명명이 금지**되어
"`FixedPolicy(34)`·sticky OFF 구성"으로만 부른다(전제 미검증,
G1-c 미실행 — in-job 단방향 반증기는 4셀 `unchanged`이며 **이는
검증이 아니다**). **크기 인용이 허용된 셀은 F-계열 미발화 셀
Zamba2 r2 하나뿐**이고 그 값은 α **−38.8%**[−41.6,−36.0] · TTFT
p95 **+51.5%**[+35.8,+67.2]다(부호 규약 = T′ 대 C, 음수 = 분할
우수). ★**이 효과는 분포의 꼬리에 한정된다** — 같은 대비에서
요청-내부 **q≤0.7 분위수는 4셀 전부 반대 부호로 유의**하며
(q0.5: −0.85/−2.58/−0.91/−1.68 ms), 기전은 "드문 대형 prefill
유발 decode 스톨을 균일한 소폭 decode 지연으로 교환"이다.
★**처치는 decode-active 시간의 35–40%만 실현**됐고(T′ idx1 frac
0.355–0.405), **100% 실현 arm(T, sticky ON)의 α 효과는 오히려
더 작다** ⇒ **어떤 dose 외삽도 금지**. "무분할(C)"은 **명시적
분할 부재**를 뜻하며 잔여 차감은 미측정이다(Zamba P-carve
NOT-EQUIVALENT = 등가 미확립이며, 차감이 있더라도 C·T′가 같은
yml·같은 green context로 부팅하므로 이 대비에서는 공통항으로
소거된다). 이 결과를 **§1-1의 A4-vs-fused 격차의 성분·기여분·
분해로 서술하지 않으며**, "PD 분리 자체가 원인"으로 승격하지
않고, **SLO goodput·용량·fused 대비 우열에 대해서는 아무것도
말하지 않는다**(같은 job에서 미조율 fused A2가 요청 p95 ITL에서
C를 이기는 셀이 있고 TTFT p95는 4셀 전부 A2가 우수하다 — 부호만,
크기 인용 금지). "SM 분할만 켠 fused arm"은 이 엔진에서 구성할
수 없다.

⚠️**함께 박을 금지 6건**: ① `component_share` 인용 금지(Granite
r3 Fieller **3.098**="몫 310%"가 아티팩트에 있으나 코드 자신이
"항등식, 결과 아님"이라 표시 — 게이트 #6). ② **두 report json의
`headline.text` 문자열 인용 금지**(m=2 Holm으로 계산한 것을 "4셀
Holm 후"라 인쇄 = 거짓 provenance — 메인 세션이 `holm` 필드
`size_this_job=2` vs 텍스트 "4셀"을 직접 대조해 확인). ③ "분할이
ITL을 개선한다"를 **꼬리 한정 없이** 쓰기 금지. ④ Δ^cont
(secondary)를 헤드라인으로 승격 금지(T는 mean e2e가 최대 **+41%**
악화). ⑤ **T′의 F-B disjunct (i) 미평가**(§8.6) 병기. ⑥ **인용
셀이 지정 셀이 아니라 구조적 저용량 복제 셀**이며 그 선별 필터의
**귀무 발화율이 40.1%**(1−0.95¹⁰, 메인 세션 재계산 일치)임을
병기.

**신규 방법론 항목**(전부 새 성능 판정 아님): (i) §4.5의 두
안전장치가 같은 3개 꼬리 통계량만 보고, `CONVENTION-SENSITIVE`는
12지표 중 9개 비트 동일(항등식) — **§3 항목18/방법론 게이트#9**
("게이트 자신이 항등식") **새 사례**. (ii) ★**방법론 게이트#20
진짜 재발**: `g2s_analyze.py:63`의 `MIN_COVERAGE=0.98`이
`gate1b_analyze.py:49`에서 **상수만 수입되고 규칙은 미구현**
(원본 `:250`은 실제 사용) — **"식별자 수입 ≠ 거동 수입"**, **§3
항목34** 새 사례(방향은 저자 불리, 4셀 전부 253–427 에피소드라
수치 결과는 없으나 provenance 기록 필수). (iii) 신규(§3 항목38):
`any()` over n reps 형태의 스크린은 귀무 발화율이
**1−(1−α)ⁿ**이며 인용 자격을 사실상 무작위 배정한다(여기선
40.1%). (iv) 신규(§3 항목39): §6.3-6의 관측자 대칭 보고 항목
(`dropped_events`/`writer_error`)이 **산출되지 않았다** — 감사자가
오프라인 복구해 전부 0 확인, 간극은 실재. 부수(문구 구속용, 정본
수치 인용 금지): pdmux 가족 내부 관측자 부하 비대칭(C가 T′보다
`runtime_snapshot` 2.6–3.8% 많고 방향이 효과에 유리, G5
UNDETERMINED라 상한 없음).

⚠️**등재 불가 항목**: e2e/concurrency 수치, Granite 전제의
"검증됨" 승격, dose 보정치, goodput 해석. 상세 §1-1(2026-08-11
Gate 2-S 결과 블록, C 스코프 주석 뒤)·§3 항목18·34(개정)·38·39,
`PROJECT_STATUS.md` "확정된 결과" 1번·"방법론 게이트" #9·#20
(개정)·#24·#25 동반 갱신. 원자료 `workspace/engine-port/results/
p1_gates/gate2/g2s_report_zamba2-27b_877756.json`·
`g2s_report_granite-40-h-micro-base_877757.json`(수정 금지·인용만).
`CLAIM_EVIDENCE_MATRIX.md`는 대조 확인 결과 이 항목을 인용한
서술이 없어 갱신 대상 없음(확인 완료, 2026-08-11).
이전 rev19: 2026-08-10 (doc-steward — 같은 캠페인 계열의 세 번째
정본 반영 건. **A(재발 카운트 갱신)·B(도구 규율)·C(설계 감사 스코프
결론)·D(코드 사실 문구 하향) 네 갈래 — 전부 새 성능 판정 아님.**

**A. 방법론 게이트 #21의 일곱 번째 재발 — 이번엔 코드가 거짓 음성을
냈다**(engine-porter 발견, 메인 세션 사건 경위 확인): job 876699
(T4-1, 2026-08-09~10)가 `--time=1:00:00`에서 TIMEOUT됐다 — 사이징이
아니라 **단일 호출 스톨**(9 부팅 중 7개가 ~8분에 정상 완료, `ON +
chunk512`의 첫 실제 multi-chunk generate(2552 토큰, 5-chunk)가
**~52분간 무응답**, 스케줄러 로그 0줄·CUDA 에러·watchdog 미발화;
같은 job의 `OFF + chunk512`는 동일 프롬프트를 동일 경로로 ~1초에
완료. 서버 로그 마지막 활동 23:34:42, SLURM TIME LIMIT kill
00:26:19 — 메인 세션이 두 시각을 직접 대조해 ~52분 간격을
확인했다. **핵심**: 그 상태의 아티팩트에 대해 옛 `g2det_analyze.py`
가 **`REFUTED — reduction-order is not the (sole) cause`를 반환하고
있었다** — ON chunk512의 `n_total=0`(arm 미실행)인데 `on_clean =
n_total > 0 and ...`이 False로 떨어져 REFUTED 분기로 통과했다.
**52분짜리 멈춤이 실질적 음성 결과로 발표될 뻔했다.** **이것이
재발 1–6과 다른 점**: 1–6은 라벨·해석 오류였고 사람이 문서에서
잡았다. 이번은 **분석 코드가 거짓 음성 판정을 산출**했고, 막은
것은 도구가 아니라 **실행자의 규율**이다(experiment-runner가
`INCOMPLETE`로 보고, 사전등록이 열거하지 않은 조건이라며 채점을
거부). 도구가 사람보다 관대했다. **사후 완화가 아님**: 사전등록의
REFUTED 조건은 "ON이 **어느 rep에서든** self-mismatch ≥1"인데 rep이
0개면 그런 관측 자체가 없다 — 옛 코드는 사전등록 규칙의 재해석이
아니라 **버그**였다. 수정(2026-08-10) 후 `NO VERDICT (MEASUREMENT
ABSENT)`를 반환하고, 데이터가 있을 때의 CONFIRMED/REFUTED 분기는
5-케이스 매트릭스로 **불변** 확인됐다. ⇒ **§3 항목35(2026-08-09
신설, 방법론 게이트 #21)의 재발 카운트를 6→7로 갱신**하고 이
구별을 그 항목에 직접 追記한다(신규 항목 아님, 아래 §3 항목35
개정판 참조).

**B. 공유 하네스의 무한 대기(도구 규율)**: `g2_holb_phaseA_lib.sh`
의 `greedy_call`이 **`--max-time` 없는 raw curl**이었고, 이
디렉터리의 **모든 캠페인**(g2ctrl/g2ea/g2holb/g2det)이 이 경로를
쓴다. 소비자 10개를 전수 확인한 결과 **4개가 타임아웃을
`FAIL`/`SMOKE_FAIL` 계열로 채점**하고 있었다(A의 `g2det_analyze.py`
포함, `g2_holb_phaseA_lib.sh:102-139` 주석에 사건 경위 기록됨,
직접 확인). 수정: `--connect-timeout 10 --max-time 180`(env
`G2_GREEDY_CONNECT_TIMEOUT`/`G2_GREEDY_MAX_TIME`로 재정의 가능),
curl 종료코드 28을 `STATUS=TIMEOUT`으로 `STATUS=ERROR`와 구별,
**SHA를 아예 방출하지 않아** mismatch 채점이 구조적으로 불가능,
사이드카 `.status.json`. 기본값 180s는 **측정 근거**로 정당화됐다
— 이 디렉터리 아카이브 응답 **n=64**(jobs 874602/874628/874633/
874635/875344/875346/875610/875611/876699, 4 arm × 2 모델)의
서버측 `e2e_latency`가 median 0.978s / p90 2.534s / **max 6.605s**
(최악 관측의 27배, 중앙값의 184배) — 메인 세션이 코드 주석의
근거 문단을 직접 대조해 확인. 정상 경로는 아카이브 64개 + 합성
실패 9종 재생으로 **73/73 byte-identical** 검증(engine-porter
보고, 메인 세션 미재현 — raw curl 응답 재생 스위트라 별도
아티팩트 경로 미확인). ⇒ 새 §3 항목37·방법론 게이트 #23으로
신설(아래) — A와 뿌리 사건은 같으나(job 876699) **레슨은
다르다**: A는 "분석 코드가 없는 데이터에 REFUTED를 내렸다", B는
"공유 인프라가 무경계 대기를 10개 소비자에 전파했다".

**C. Gate 2-S 3라운드 설계 감사의 구조적 결론 — §1-1 귀속 스코프
주석(대체 아님)**: `PREREG_GATE2S_2026-08-09.md`가 claims-auditor
감사 **3회 전부 NO-GO**를 받았다(rev1→rev2: 통계층 / rev2→rev3:
게이트 인식론 / rev3→rev4: 귀무 채택형) — 메인 세션이 파일 직접
확인. **3연속이 같은 자리(§5.5 앵커 발화 조건)에서 죽었고**, 3차
감사가 근본 원인을 **문서 내부 모순**으로 특정했다(파일 §0.0.B
직접 확인): §0.1이 "두 pdmux arm 사이 등가 마진에는 외부 앵커가
없다"고 이미 확립했는데 §5.5는 정확히 그 등가를 앵커 조건으로
요구했다. 정량(3차 감사, §0.0.B 원문 대조): rev3 §5.3의 **암묵
등가 마진 = 0.715 σ_D**, 이 설계의 **primary MDE = 0.995 σ_D**
⇒ 대리 허용오차가 검출한계의 **0.72배**. 표준 처방(TOST+사전등록
마진)은 §0.1 정면 위반이고, 마진을 MDE의 1/4로 낮추려면 **n≈64**
(현재 10)가 필요하다. ⇒ **정본에 기록하는 명제(문구 고정)**:

> **"P1 이득 중 'SM 분할 자체의 몫'을 두 pdmux arm 사이의 성능-층
> 등가검정으로 귀속하는 경로는, 이 프로젝트의 예산 범위에서 닫히지
> 않는다**(n≈64 필요, 현행 설계 n=10). 원리적 불가능이 아니라
> 이 경로·이 예산에서의 불가능이다.**"**

⚠️**과장 금지**: "귀속이 원리적으로 불가능하다"로 쓰지 않는다.
이것은 **미실행 사전등록에 대한 설계 감사**이지 측정 결과가
아니다 — 등급어를 그에 맞춘다(§1-1의 NOT-YET-SUPPORTED는
**대체하지 않고**, 위 인용문을 스코프 주석으로 **덧붙인다** — "아직
안 됐다"와 "이 경로로는 안 된다"는 다른 진술). **함께 기록(감사가
깨뜨리려 시도했으나 실패한 것 = 견고한 것, §0.0.A 직접 확인)**:
arm 구조·Δ_split estimand의 내부 타당성, T·C 드레인 대칭, 9-셀
판정표의 저자 불리 셀 실재, 부팅 단위 프로브 근거(n_eff=10),
예산 산술. 3차 감사 원문(파일 §0.0.A 직접 인용): **"이 캠페인이
죽어야 할 이유는 측정 층이 아니라 귀속 층에만 있다."** rev4는
§5.5 앵커 발화 조건 자체를 제거해 이 특정 귀속 하위목표에서
후퇴했다(패치 없음, env 조합만 재구성) — **Gate 2-S가 폐기된
것은 아니다.**

**D. 코드 사실 문구 하향(메인 세션 자기정정)**: 메인 세션이 이
세션 중 "`(0,108)` idx에서는 두 역할 모두 전체 108 SM에
접근한다"고 서술했다. **검증된 것은 "green context가 아니라
평범한 `torch.cuda.Stream` 쌍이다"까지**다(`pdmux_context.py:
124-138`, 메인 세션 코드 직접 확인 — idx 0과 idx `len-1`은
`torch.cuda.Stream(gpu_id)`이고 중간 division만
`create_greenctx_stream_by_value`). green ctx 생성이 primary
context의 SM을 깎는지는 **미측정 물리 명제**다(격리 측정 수단
이던 P-b 프로브는 공선성 때문에 삭제됨, engine-porter 보고 —
메인 세션 미재검증). ⇒ **전수 검색 결과(2026-08-10) 이 문구는
canon·파생 문서 어디에도 들어가지 않았다** — 정정 대상 문구는
없다. 향후 인용 규칙으로만 등재: **"(0,108) idx에 대해서는
'명시적 분할이 적용되지 않는다(잔여 차감 여부는 미측정)'로만
쓴다"** — "두 역할이 전체 108 SM에 접근한다"류의 문구 금지.

상세는 §1-1(2026-08-10 세 번째 정본 반영 건 A/B/C/D 블록)·§3
항목35(개정)·37, `PROJECT_STATUS.md` "확정된 결과" 1번·"방법론
게이트" #21(개정)·#23 동반 갱신. `CLAIM_EVIDENCE_MATRIX.md`는
대조 확인 결과 이 항목을 인용한 서술이 없어 갱신 대상 없음(확인
완료, 2026-08-10).
이전 rev18: 2026-08-09 (doc-steward — 같은 캠페인 계열의 두 번째
정본 반영 건. **A(재채점)·B(사실 정정)·C(도구 규율) 세 갈래 — 전부
새 성능 판정 아님.**

**A. HOLB G5 재채점(result-analyst, 2026-08-09, jobs
874601/874602/874632/874633/874635)**: 원자료
`workspace/engine-port/results/p1_gates/gate2/
g2holb_g5_tost_rescore_2026-08-09.json`(72셀 전량, 3 job × 4 arm ×
6 응답변수, n=5 paired) + `g2holb_g5_tost_rescore.py`·
`g2holb_g5_tost_summary.py`. **판정: G5 = 미결정(UNDETERMINED),
저장된 `G5=False`를 대체한다.** 3% 초과가 통계적으로 지지되는 셀
**0/72**(미보정 최소 p_exceed=0.138, job별 Holm 후 최소 조정
p=1.000 3 job 전부), 등가 입증 **29/72**, 검정력 부족 **43/72**.
G5는 프로브 관측자 효과가 3%를 초과함을 입증하지 못했고, 3% 이내임도
입증하지 못했다 — 저장된 `G5=False`는 귀무-채택형 연언(`PASS =
|효과|<3% ∧ CI∋0`)의 실패를 기록한 것이지 프로브 유해성의 입증이
아니다. **구 규칙이 노이즈를 보상했다**(874635 agnostic ttft_p95
−0.77%±29.14%, 95% CI [−36.95,+35.41] → 구 규칙 PASS, 메인 세션
독립 재확인) — 역방향(구 규칙 FAIL·±3% 등가 실제 입증)도 **4셀**
발생(예: 874633 plain itl_p95 −1.15%±0.72%, p_TOST=0.0023). **설계
층**: n=5·δ=3%·α=0.05/side에서 TOST 발화 산술 천장 SD<3.147%인데
72셀 중 **35셀(49%)**이 그 위. 80% 검정력 필요 n 중앙값 **9**(변수별
request_throughput 4 / itl_p50·mean_e2e_ms 7 / itl_p95 22 /
ttft_p50 29 / **ttft_p95 89**, ⚠️SD가 df=4 추정이라 필요 n은 자릿수
수준 의미만). **`PREREG_GATE2` §14.2의 "프로브가 무해함이 입증됐다고
쓰지 마라"는 해제되지 않는다** — 동시에 반대 오독("3% 넘게 유해함이
입증됐다")도 근거 없음이 확정됐다. §14.3의 귀인("주로 점추정이 3%를
넘는 조합이 실재하기 때문")은 **부분적으로만 참**(그런 셀은 실재하나
그중 하나도 초과가 지지되지 않는다) — 두 문장 병기 필수. **잔여
교락**: 874602 agnostic은 5/5 rep 전부 `order='off on'`(무작위화
불균형) ⇒ 그 arm의 등가 판정은 조건부. **등가 판정 29건은 전부
paired-t 정규 가정 위**(n=5 분포무가정 두측 p 하한 2/32=0.0625).

**B. job 874601의 `G3=False` 라벨 정정(메인 세션 원자료 직접 확인)**:
`g2holb_report_zamba2_874601.json`은 4 arm 전부 `result:"FAIL"`이나
`sha_off:null, sha_on:null`·`self_repro_off/on:true`다 — 출력이
갈라진 게 아니라 sha 추출이 응답 스키마를 못 읽은 것(`KeyError:
'text'`)이고 Phase B가 실행되지 않았다. 재실행 874633/874635는
`method_off/on:"output_ids"`로 sha 양측 일치 → **PASS**(위 A절
근거로 이미 사용됨). 874632는 Phase A 중 SLURM CANCELLED(데이터
없음). ⇒ "874601 G3 실패" 라벨은 **하네스 실패**로 정정한다. **이
프로젝트 서명 오류(측정 실패를 게이트 실패로 라벨링)의 여섯 번째
재발**이다 — 핸드오프 2026-08-09 §1이 다섯 번(텔레메트리 드롭
카운터 자기검열 / UNSCOREABLE을 강등으로 읽음 / HTTP 400을 G3
FAIL로 / G5의 귀무-채택형 기준(=위 A절과 같은 사건 계열) / 프로브의
stderr 오염)으로 셌다. **canon에 이 패턴을 다루는 기존 번호가
없어(전수 검색 확인) §3 항목35로 신규 등재**(중복 신설 아님 — 이후
재발은 이 항목의 카운트만 갱신).

**C. E-A 커밋 산출물의 scipy 부재 폴백(engine-porter 발견 + 메인
세션 노출범위 실측, 범위 한정)**: `g2ea_report_*.json`은 scipy 없는
인터프리터에서 생성돼 `t_cdf()`가 Student-t가 아니라 정규 CDF로
조용히 폴백했다(저장된 `p_tost`가 정확히 1.0인 이유,
`g2ea_analyze.py:84-87,156-159`). **메인 세션이 노출 범위를 직접
측정**: `raw_ci` 폭에서 역산한 임계값이 **8개 비교 전부**(2 job ×
2 rate × 2 cmp, coordinator가 인용한 4개 포함, 전부 df=9)
implied_t = **2.2621… = t(.975, df=9)**(하드코드 표값과 일치, 1.96
아님) — **CI(raw_ci)는 오염되지 않았다.** 노출은 (i) `p_tost` 값
자체(이 캠페인은 관측치가 0.05 경계에서 멀어 `NOT_EQUIVALENT`
판정에 영향 없음)와 (ii) `t_ppf`의 경우 **df>10이거나 표에 없는
p**에 한정된다(`t_cdf`는 표가 아예 없어 scipy 부재 시 모든 df에서
근사값이라는 점은 (i)에 포함되되 원인 층이 다름을 기록). ⇒ **등재
방식 = "정본 수치 정정"이 아니라 도구 규율 항목**(게이트 #14
계열 — 통계 라이브러리의 조용한 폴백은 아티팩트에 기록되지 않는다;
분석 재현 시 인터프리터 환경을 아티팩트에 남겨라), §3 항목36
신설. **과장 금지 — 이 캠페인에서 오염된 인용 수치는 없다.**

상세는 §1-1(A/B/C 블록, E-A 블록 뒤)·§3 항목35·36, `PROJECT_STATUS.md`
"확정된 결과" 1번·"방법론 게이트" #21·#22 동반 갱신. `CLAIM_EVIDENCE_
MATRIX.md`는 대조 확인 결과 이 항목을 인용한 서술이 없어 갱신 대상
없음(확인 완료, 2026-08-09).
이전 rev17: 2026-08-09 (doc-steward — ★★★★★**Gate 2 rev4 본
캠페인(jobs 875344/875346, 2026-08-07 실행, 5.86 GPU-hr) 정본 반영
복구 — R1′/R2′ 확립.** 새 성능 판정 아님 — 이미 완료된 결과의 정본
누락 복구다. Primary(A3=`chunk512` vs A4=`agnostic`)는 5셀
(Zamba2 r2·r3, Granite r3·r4·r6) 전부 `Rprime4`(A4 유의 우세, TOST
등가 미발화) — chunk512는 pdmux를 대체하지 못한다(크기 인용 가능
셀은 F-E 스크린 통과분인 Zamba2 r2·Granite r3뿐). **Secondary —
R1′/R2′(A2=`plainaux` vs A4)**: rev4 §1.1의 등식 A2=A4−pdmux
(realized server_args 차이는 `enable_pdmux`·`pdmux_config_path` 두
필드뿐)이므로 이 비교는 §1-1의 "3-플래그 묶음 처치" 교락 중 pdmux
고유분을 분리한다. 메인 세션이 2026-08-09 원자료(`per_arm_x60`,
paired-t n=10)로 독립 재현: Zamba2 r2 **+0.834**[+0.781,+0.887]·r3
**+0.930**[+0.897,+0.963], Granite r3 **+0.855**[+0.816,+0.894]·r4
**+0.944**[+0.920,+0.968](이상 부호 10/10, F-E clear), r6
**+0.960**[+0.929,+0.991](**부호만** — agnostic 측 F-E flagged).
sign-flip 순열 p = 2/1024 = **0.001953125**(5셀 공통 하한). A2는
A1(`plain`)과 사실상 같고(5셀 |A1−A2|≤0.023) 그 부호는 **A4에
불리한 핸디캡 방향** ⇒ aux 플래그(`--chunked-prefill-size -1`·
`--disable-overlap-schedule`) 단독으로는 pdmux 이득이 재현되지
않는다 — §1-1 "3-플래그 묶음 처치" 교락 중 이 두 플래그는 원인에서
**배제**된다. ★**provenance(인용 시 필수 병기)**: 이 A2-vs-A4
비교는 **사전등록 분석기 `g2_analyze.py`가 계산하지 않는다** —
`tost_equivalence` 호출은 코드 전체 1회뿐(`:543`)이고 입력은
A3-vs-A4뿐(`:538-539`); A2는 `:734`에 서술 문장으로만 등장한다.
**arm·n·raw 데이터는 사전등록(rev4)이지만 이 비교 자체는 저장된
primary 산출물(`per_arm_x60`) 위의 사후 계산**이다(코드 확인
2026-08-09, §3 항목34 신설 — 항목33 "사후 지정 셀 이동"의 형제
사례). ★**천장 포화(해석 제한)**: fused arm X_60 평균이
0.72–0.99·agnostic이 0.00–0.12로 양쪽 포화 근접 ⇒ **부호는
견고하나 크기는 포화 구간의 값**이며 E-A 블록이 이미 건 "기전
해석 금지"가 여기도 적용된다. ★**해금 범위의 상한(overclaim
금지)**: 해소되는 것은 §1-1의 "3-플래그 묶음 처치" 교락뿐이다.
귀속의 상한은 **"pdmux 서브시스템 전체"**(green-context SM 분할 +
전용 이벤트 루프 + split-prefill)이고 **"SM 분할 자체"는 여전히
미분리**다 — `--enable-pdmux`가 A2의 `event_loop_normal()`을
대체해 pdmux 전용 이벤트 루프를 통째로 켜기 때문이다(§1.1).
**"PD 분리 자체가 원인"으로 승격 금지** — §1-1의
NOT-YET-SUPPORTED 등급은 불변, Gate 2 본 질문은 한 눈금도
전진하지 않는다. ★**감사 provenance 정정**: 세션 핸드오프
(`handoff-report/session_handoff_2026-08-09.md` §2.5)는 이 결과를
"[감사 완료]"로 표기했으나, 2026-08-09 저장소 전체 검색(`gate2/`
디렉터리·전체 `*.md`)으로 이 R1′/R2′ 비교를 다루는 claims-auditor
감사 아티팩트가 확인되지 않는다 ⇒ **"감사 완료"로 인용하지
않는다.** 현재 등급 = **메인 세션 원자료 독립 재현 확인
(2026-08-09), claims-auditor 감사 기록 위치 미확인.** ⚠️**부수
정정**: `PREREG_GATE2_2026-08-06.md`(§14 addendum)는 커밋
`c47fad0`로 반영돼 **워킹트리 클린**(2026-08-09 확인) — 아래
rev16·§4 "살아있는 문서" 표가 기록한 "워킹트리 미커밋" 상태는
**stale, 정정**(§11-3 감사 면제 상태는 별개로 불변 — "감사 통과
설계"로는 여전히 인용 금지). 상세 §1-1(rev4 본 캠페인 블록, E-A
블록 앞)·§3 항목34, 원자료 `workspace/engine-port/results/
p1_gates/gate2/g2_report_zamba2-27b_875344.json`·
`g2_report_granite-40-h-micro-base_875346.json`(`per_arm_x60`).
`PROJECT_STATUS.md` "확정된 결과" 1번·"방법론 게이트" #20 동반
갱신. `CLAIM_EVIDENCE_MATRIX.md`는 대조 확인 결과 이 항목을 인용한
서술이 없어 갱신 대상 없음(확인 완료, 2026-08-09).
이전 rev16: 2026-08-09 (doc-steward — ★★★★★★**E-A(mixed-chunk
레버, jobs 875654/875657/875661, 2026-08-08~09) 반영 — claims-auditor
감사 완료, 판정 조건부(문구 강한 제한). 새 성능 판정 아님 — fused-측
조율 가능성에 대한 진단, Gate 2 본 질문("PD 분리 자체" 귀속)은
전진하지 않았다.** Stage 1(875654)에서 `--enable-mixed-chunk`가
Zamba2-2.7B·Granite-4.0-h-micro-base×{8192,512} 4콤보 전부에서
realized 확인(3중 증인: `/server_info`·`internal_states[0]`·HOLB
`holb_open`), 기능 스모크 8/8. E-A(875657/875661, n=10 paired,
cudagraph-ON, arm 순서 무작위, gpu42/gpu40)의 사전등록 primary 두
비교×4셀=8건 전부 TOST 등가 미발화·부호 전부 agnostic 우세(10/10,
분포무가정 정확 p=0.00195) ⇒ **"조율된 fused가 pdmux를 대체한다"는
발화하지 않았고 논문 신규성 축 전환은 없다.** 유일한 인용 가능
정량치(Granite rate3, cmp1, F-E clear·G-2 무관): X_60(plainmix)−
X_60(agnostic)=+0.855[+0.814,+0.896], n=10, 부호 10/10 — 단 **사후
지정 셀 이동**(r4→r3, 선행 캠페인 진단을 본 뒤)의 산물이라 이동이
없었으면 인용 가능 셀 0개였고, **같은 셀에서 TTFT 항은 통과**한다
(plainmix가 −19.2%[−0.256,−0.129] 더 좋음 — "TTFT 비열등 0건"은
거짓, 1/8). **레버 격리**: +0.855 중 mixed-chunk 기여분은
**+0.010[+0.0045,+0.0155]=1.2%**뿐(4셀 범위 1.2–10.6%), 나머지는
rev4/§1-1이 이미 확립한 untuned-fused-vs-pdmux 격차 — X_60
천장근접(0.85–0.99)으로 하향편향, HOLB 관측자 효과 잔차와 같은
자릿수라 **기전 해석 금지**. **기전(신규, 서빙 직접 측정)**:
mixed-chunk ON 시 prefill과 병합된 decode가 `ForwardMode.MIXED`
extend 경로로 재라우팅되고(cudagraph 제외 경로), MIXED step
지속시간이 병합 decode 수에 선형(Zamba2 ≈33ms/req, Granite
≈4.2ms/req) — 요청별 ITL이 MIXED step time과 같아지고 backlog가
`max_running_requests`·mamba usage를 포화시켜 admission을 막고
TTFT가 발산한다(정본 §1 얽힘 死因의 **새 트리거**이지 새 기전
아님). **cudagraph 가설은 반증**(순수 DECODE step은 양 arm 동일
지속시간·cuda graph True, piecewise 전 arm OFF) — 일어난 것은
"decode eager 강등"이 아니라 "decode의 extend 경로 재라우팅"이며,
이 정정으로 rev4 §1.1의 "A3는 piecewise도 함께 바꾸는 ≥2-기전
묶음" 캐비어트는 **이 두 모델에선 런타임 실측으로 무효**(런타임
비활성 확인, 인용 금지 해제). **G-2(재현성, correctness 아님)**:
`--chunked-prefill-size 512` Granite가 16-동시 greedy에서 비트단위
재현성 상실(16 중 2 요청, 반복 단위 62↔322 토큰에서 결정론적
분기, 주기 3의 31개 지점, 3회 독립 재현, 음성대조 4종 통과, mamba
state 이월과 불정합) — correctness 결함 아님. **T2(fused 조율
소진 주장 금지)**: 정본이 지목한 두 레버(`--chunked-prefill-size`·
`--enable-mixed-chunk`)는 시험됐고 격차를 못 닫혔으나, 같은 기전
축의 **미시험 노브 최소 6개**(realized 값 실측 — `--prefill-max-
requests`=None, `--num-continuous-decode-steps`=1,
`--chunked-prefill-size` 2048/4096, `--max-running-requests`=48
(실제 포화점), `--schedule-conservativeness`=1.0,
`--mamba-scheduler-strategy`=no_buffer, +적용성 미확인
`--enable-prefill-delayer`)가 미시험이라 **"fused 조율 공간
소진"은 정본 등재 금지**. **Gate 2 본 질문 전진 없음**: A2
(`plainaux`)를 뺐으므로 "PD 분리(SM 분할) 자체" 귀속은 한 눈금도
전진하지 않았고 §1-1 NOT-YET-SUPPORTED는 불변, 875344/875346의
A2로 교차-job 메우기도 사전등록이 명시 금지
(`PREREG_G2EA_2026-08-07.md:66-67`). **인용 금지 13건**·**신규
방법론 항목 3건**(primary뿐 아니라 보고되는 모든 블록에 게이트를
걸어라 / 과부하 arm과 정상 arm을 같은 rate에서 비교한 수치는
시스템 상수가 아니다 / 사후 지정 셀 이동은 부호를 안 바꿔도 인용
가능성을 만들 수 있다)은 §1-1(E-A 블록)·§3 항목31–33 전문 참조.
**후속 게이트**(우선순위순): T4-1(deterministic-inference
재확인) → T3-1/T3-2(MIXED 배치 조성 계측·micro 스윕) →
E-C(등지속가능-rate 대조, T1 정식 종결) → E-D(미시험 fused 레버
스윕, T2 종결) → T3-3(backend 교차, 8× 비대칭 귀속) → T4-2
(비퇴화 프롬프트 G-2). ⚠️**상위 사전등록
`PREREG_GATE2_2026-08-06.md`는 워킹트리 미커밋 수정 상태**
(§14 addendum) — §4 "살아있는 문서" 표 갱신. ★**정정(2026-08-09,
rev17)**: 이후 커밋 `c47fad0`으로 반영돼 워킹트리 클린 확인, 위 참조.
상세 §1-1(E-A 블록,
이 파일)·§3 항목31–33, 원자료 `workspace/engine-port/results/
p1_gates/gate2/`(`PREREG_G2EA_2026-08-07.md`·`g2ea_report_*.json`·
`g2earun_8756{57,61}.out`·`g2eaprobe_875654*`, 수정 금지·인용만).
`PROJECT_STATUS.md` "확정된 결과" 1번·"방법론 게이트" #17–19·
"다음 실험 gate" #10 동반 갱신. `CLAIM_EVIDENCE_MATRIX.md`는 대조
확인 결과 이 항목을 인용한 서술이 없어 갱신 대상 없음(확인 완료).
이전 rev15: 2026-08-07 (doc-steward — ★★★★★**G1-b(job 875293,
2026-08-07) 사전등록 철회 규칙 발화 — rev14 §1-1의 "이 격자에서
정책은 단일 분할에 고정됐다" 문장 철회.** [experiment-runner
사전등록·실행, `workspace/engine-port/results/p1_gates/gate1/
PREREG_G1B_2026-08-07.md`·`gate1b_result_875293.txt`]. **새 성능
판정 아님 — Gate 1 진단 산출 1의 사실 정정, 사전등록 철회 규칙
발화.** Zamba2-2.7B·agnostic·cudagraph-ON·`TRACE_EVERY=32`·
`FORCE_PREFILL=1`·n=1, 873944의 `MAIN_RATES` 순서 그대로
rate{2,3,4,6}을 한 부팅 안에서 재현(gpu41, 8분 38초). **양성대조
PASS**(rate2·3이 874478을 재현: pop A frac 1.0000, max decode_bs
9·23). **rate 2·3·4**는 pop A 시간가중 100% `(74,34)`, `max(decode_
bs)` 9·23·18(문턱 36 미만) — 인용 가능 셀(Zamba2 r2·r3)에서는
rev14 문장이 여전히 참이다. **rate 6에서만** `(54,54)`가 시간가중
**8.37%**(2.089s/24.974s, max decode_bs 40) 등장 ⇒ 사전등록 철회
규칙(`frac((54,54))≥0.01`, `gate1b_analyze.py`에 하드코딩) **발화**
— rev14의 "단일 분할 고정"·"`(54,54)`는 0회 선택" 문장을 **철회**
(원문은 §1-1에 취소선으로 보존). ⚠️**과잉 철회 금지**: 거짓이 된
것은 **격자 전체에 대한 무제한 주장**뿐, 인용 가능 셀은 불변.
⚠️**rate 4(18)가 rate 3(23)보다 낮다 — 비단조·원인 미해명**(순차
부팅의 순서 효과 또는 큐잉 동역학 가능). ⚠️**rate 6 경계 근접**:
pop A 에피소드 5개뿐, max decode_bs 40이 문턱 36을 11%만 초과 —
정본은 Zamba2 r4·r6을 이미 "⚠용량 초과" 셀로 표시 중이다.
Population C 절대 수치는 여전히 인용 금지(Gate 1 감사 caveat
승계). 매니페스트는 874478과 **바이트 동일 트리가 아님**(HOLB
프로브 설치로 `holb_probe.py`·`scheduler.py` +2줄, 0줄 변경 —
동등성 근거는 **G3**, jobs 874635/874633, output_ids 바이트 동일
2모델×4arm). `(74,34)`·`(54,54)`는 여전히 selector 라벨(하드웨어
SM 수 아님). ★신규 방법론 항목(§3 항목30): **스코프가 좁은 주장을
확장할 때는 인접한 한 점이 아니라 원 격자를 전부 재현하라** —
rate 4만 돌렸다면 `NO_EVIDENCE`가 나와 반대 방향(격자 전체로
확장)의 오류를 저질렀을 것이다, 873944의 전 격자 {2,3,4,6}을
그대로 재현한 설계가 그것을 막았다. `PROJECT_STATUS.md` "확정된
결과" 1번·"방법론 게이트" #16·"다음 실험 gate" #10(G1-b) 동반
갱신. `CLAIM_EVIDENCE_MATRIX.md`는 대조 확인 결과 이 항목을
인용한 서술이 없어 갱신 대상 없음(확인 완료). §1-1(Gate 1 블록)
아래 상세.
이전 rev14: 2026-08-06 (doc-steward — ★★★★★**Gate 1(job 874478)
조건부 채택 반영** [claims-auditor 감사 2026-08-06, `workspace/
engine-port/results/p1_gates/gate1/`]. **새 성능 판정 아님 —
진단 전용.** §1-1(P1)에 selector-level 파티션 라벨 관측 결과를
조건부·부분 해금 블록으로 추가(전문은 §1-1 참조) — decode-busy∧
prefill-in-flight 구간의 시간가중 라벨은 Zamba2 rate{2,3}·
agnostic v1·cudagraph-ON에서 `(74,34)` 100%였으나 그 조건 자체가
코드상 항등식에 가까워 판별력이 사실상 없다(방법론 게이트 #9
다섯 번째 재발, 아래 §3 항목28). `(54,54)`는 이 격자에서 0회
선택. rate 4·6·Granite는 여전히 미측정 ⇒ Granite rate4 +27.0%
셀에는 파티션 문장 금지. "PD 분리 자체" 기전 귀속은 여전히
NOT-YET-SUPPORTED(Gate 2 소관, 불변). 시간가중 step-function
추정량의 두 함정(케이던스 불변성을 물리적 불변성으로 오독 /
비인접 구간 dt 오염)을 §3 항목29로 신설. 커버리지 가드 논거는
좁은 형태(coverage 실패는 UNLOCK을 만들지 않는다)만 살아남고
일반 원칙으로 승격하지 않는다. `PROJECT_STATUS.md` "확정된 결과"
1번·"방법론 게이트" #9·#15·"다음 실험 gate" #10(G1-a–d) 동반
갱신. `CLAIM_EVIDENCE_MATRIX.md`는 대조 확인 결과 갱신 대상 없음.
이전 rev13: 2026-08-06 (doc-steward — ★★★★**§1-1(P1) 통계 방법 층
정정** [claims-auditor Gate 2 설계 감사 2회 + result-analyst 독립 재현,
2026-08-06, `workspace/engine-port/results/p1_gates/verify/`]. **새
성능 판정 아님 — 기존 정본 수치의 통계 방법 층 정정이다.** `paired_
bootstrap_ci`(n=5 percentile bootstrap of mean)는 coverage 0.840(명목
95%의 한쪽 오류율 ≈8.0%, n=5 자체가 원인, n=4 0.798/n=6 0.859/n=8
0.888)이라 소표본 판정에 부적합함이 2출처 독립 확인됐다 — 저장소
안에 `m3_analyze.py`·`tfgate_analyze.py`가 이미 독립으로 t-CI로
전환한 동일 진단이 있었는데 정본 라이브러리·P1 판정서는 반영하지
않았다(도구 규율 실패, 신규 방법론 게이트 #27). primary를 t-CI로
교체해 재채점: **Granite rate3(+11.7%)는 t-CI가 0을 포함(p=0.0515)
→ 인용 목록에서 제외**(미검증, 철회 아님) ⇒ **인용 가능 정량치
4개→3개(Zamba2 rate2 +40.5%/rate3 +185.8%, Granite rate4 +27.0%)로
축소**. "임계 사다리 40–300ms 전 구간 부호 불변"도 정정 — 부호
불변은 **T∈[40,113.0)ms뿐**이고 그 위에서 인용 4셀 중 3셀(Zamba2 r2,
Granite r3·r4)이 술어 포화로 음전환, 끝까지 유지되는 유일한 인용
셀은 **Zamba2 r3**. "rep 부호 5/5"도 정정 — **Granite rate2는
3/5**(효과의 87%가 rep2 1점), 나머지 7셀은 5/5 유지. **P1의 방향
자체(agnostic>fused, Zamba2 r2·r3·Granite r4)는 어떤 방법으로도
불변** — 강등되는 것은 "전 셀"·"5/5"·"사다리 전 구간"·"Granite r3
수치"뿐. E1(`s8_frontier/e1_analyze.py:492,1286`) 사전등록 결정
규칙도 같은 undercoverage를 상속하므로 `PROJECT_STATUS.md` "다음
실험 gate" #8에 제출 선행조건으로 등재. `CLAIM_EVIDENCE_MATRIX.md`는
이 수치를 인용하는 서술이 없어 갱신 대상 없음(대조 확인 완료).
`PROJECT_STATUS.md` "확정된 결과" 1번·"방법론 게이트" #14·"다음
실험 gate" #8 동반 갱신. §1-1(rev13)·§3 항목27 아래 상세. 이전
rev12: 2026-08-05 (doc-steward — ★★★**§1-1(P1) 갱신: P1 운영점
대조(cudagraph-ON) 감사 반영 [claims-auditor 2026-08-05, jobs 873944/873945,
`results/p1_opint/`]. 새 성능 판정 아님(감사자 판정을 정본화만 함).**
정본 goodput 술어로 채점하면 agnostic v1이 fused를 Zamba2-2.7B·
Granite-4.0-h-micro-base **2모델 전부·rate 2–6에서** 이기나(정상상태 셀
한정 인용), 사전등록 mean-ITL 술어로는 Granite 부호가 뒤집힌다(단
위반 0건=항등식이라 강등 채택 안 함) ⇒ "Zamba2 확인"·"Granite 강등"
둘 다 등재 금지. "항상 이득" 철회(pdmux는 정상상태 decode를 5–45%
늦춤), 기전은 3-플래그 묶음 처치라 "PD 분리 자체" 귀속 NOT-YET-
SUPPORTED, 실현 파티션 미측정. NemotronH·Falcon-H1 미측정이라
"4모델 전부" 문구 폐기. 신규 방법론 게이트 #12(임계 사다리+큐-성장
검정)·#13(묶음 처치) + #6 새 사례(위반 0건 술어=항등식) 등재.
`PROJECT_STATUS.md` "확정된 결과" 1번·"방법론 게이트"·"다음 실험
gate" 동반 갱신, `CLAIM_EVIDENCE_MATRIX.md`는 이 항목을 인용한 기존
서술이 없어 갱신 대상 없음(확인 완료). §1-1(P1) 아래 상세. 이전 rev11:
2026-08-05 (★★★★**고-D 대조(job 873921) 판정 반영
[claims-auditor 2026-08-05 감사 — §0 종결 CONFIRMED (scoped) 등급 불변,
사전등록 밴드 [12,13]ms는 REFUTED, 새 성능 판정 0건]**: α(T8 d92=(P16,D92),
sticky ON, ShareGPT rate 2, n=4 블록, cudagraph ON, gpu41)가 실행됐다.
pooled per-token ITL p50 = **11.26ms**(telemetry-path)/**11.32ms**
(raw-itls path), `E1_DECODE_REALIZED`=1.000/0.998/0.998/1.000, `n_err=0`.
사전등록 3-밴드 규칙 적용 시 **INDETERMINATE**(12–13 밴드 미달·28–30
붕괴 밴드와도 거리 큼) — **붕괴 분기는 REFUTED**, §0 종결은 유지된다.
★**밴드 자신이 REFUTED**: [12,13]ms는 §1-31이 스스로 금지한
**C2→sticky 이식**으로 도출됐고, α 내부 엔진측 step 회귀로 예측한 C2
동작점(12.44–12.80)이 관측(12.875)과 0.9–3.8%만 어긋나 기록된 계통
오프셋(−5.4%)으로 격차가 소진된다 — **워크로드 불일치**(decode-busy
ctx_p50 중앙 1291 vs 287 tok, decode batch 11.31 vs 4.51,
closed-loop+keepalive vs open-loop)가 원인이며 새 기전이 아니다. 대신
α는 §0 종결을 분쟁 필드(`decode_sms`) 내부 재진술에서 **결과(outcome)
축 앵커**로 옮긴다: block-matched ON d92/OFF d16(872077)=
**1.0293[1.0210,1.0376]**, ON d16/OFF d16=**2.6320[2.6193,2.6447]**.
**S3(하드웨어 부여 층)는 여전히 닫히지 않는다**(D92/D108을 0.6–2.9%밖에
못 벌려 검정력 없음). **다음 gate 재정렬**: (δ) 같은 바이너리 OFF
arm(10% 미만 교차-job 비교 인용의 신규 선행조건) → (β) →
(γ), (α′) sticky ON을 C2 클라이언트로 1–2블록(γ와 병렬). **신규 방법론
항목 §3-26**(예측 밴드도 이식 금지 규칙의 적용 대상). 정본 반영:
`CONSENSUS.md` §1-31(이 파일)·§3-26, `PROJECT_STATUS.md` "8B
decode-SM 프론티어" "2026-08-05(α)" 소절·"E1은 열리지 않는다"
항목·"다음 실험 gate" #8. 원자료
`workspace/engine-port/results/s2_sticky/s2a_pooled_873921.txt`·
`s2a_T8_873921_result.txt`·`s2_sticky_d92.sbatch`(prereg 규칙
:403-408, 전용 분석 md 아직 미작성). 이전 rev10: 2026-08-05 (★★**ceiling-censoring 진단(§1-13 각주의
직접 검정) claims-auditor 감사 완료 — 새 성능 판정 0건, §1-13/HE0 판정
자체는 불변.** result-analyst의 `CEILING_CENSORING_DIAG_2026-08-05.md`
(UNAUDITED)를 감사 — **헤드라인 논증(§3.4 임계 사다리·§5 근거5) REFUTED**
(T=20/30/60 세 점만으로 부호가 뒤집힌다는 서사는 검정력 0인 9-vs-9
이벤트, 방법론 게이트 #6의 새 사례이자 §3 항목13의 6번째 재발, §3
항목24 신설), **§3.5 후반("[55,65) split-불변 모드가 >60 질량을
지배") REFUTED**, **§4 C2 정성 대조 REFUTED**(항목21의 3회차 재발, §3
항목25 신설, ε 정합 요구 0.63–0.78 vs 정본 0.48–0.56), **§3.6 "gpu39
3중 사다리" 사실오류로 폐기**(d24 rep1 실제 노드=gpu43, `sgptv_852341.
out` 확인) — **line 89도 사실오류**(07-17 배치는 여전히 3× 버그 값,
수정은 07-18부터, `sgptv_856889.out`/`sgptv_859005.out` 확인). **단
각주의 결론(각주 자체) 은 REFUTED가 아니라 다른 증거로 CONFIRMED**:
LO goodput은 이중 절단(처리량=도착률, pass=SLO 여유)이고, LO에도 ITL/
TTFT 레버가 실재한다(요청별 ITL p95 p90 5.6 SD·중앙값 3.2 SD·TTFT p50
3.3 SD) — 단 SLO 예산 단위로 HI의 1/14, 통계량에 따라 부호가 뒤집혀
"어느 split이 LO에서 좋다"는 functional 없이 정의되지 않는다(§1-13
각주 전면 교체). **§4.1/§4.2 부수 발견 중 F(HI spread=legacy scorer,
d16 n=1, 정본 술어로는 +716%)는 허가, E(LO n 상이) 원안은 귀속 오류로
불허하고 재작성본만 허가**(spread 0.067→0.087 재산출, "spread<rep SD"
로 재서술, "d24 rep1 12건이 spread 전부"라는 귀속은 REFUTED — bind
no-gate n=4가 같은 spread를 만든다). **노드 교락 방어 신규 기록**: arm
내부 노드효과(0.05–0.33ms) vs arm효과(1.73ms)=3–19%, 완전매칭 대조
(d34/d44 rep41-43, 같은 노드 gpu38) 부호 유지. 도구 결함 2건 기록
(`analyze.py:183` request_slice가 분자만 자르고 duration은 안 자름 —
engine-porter 후속 항목; duration 합산 `:180`은 코드로 확인, 옳음).
살아남은 것(감사자가 깨려 시도했으나 못 깬 것): pass-절단 산술·워크로드
바이트 동일성·phase 경계 청결성·duration 합산·legacy 스코어러 확인.
`CEILING_CENSORING_DIAG_2026-08-05.md` 원문에 정정 표시 추가(보존,
UNAUDITED→AUDITED 배너 갱신). 정본 반영: `CONSENSUS.md` §1-13·§3
항목24·25(이 파일), `PROJECT_STATUS.md` "다음 실험 gate" #9.
`PROJECT_STATUS.md:434-443`(C2 scope 문구)·§3 항목21과의 정합 확인
(3회차 재발로 등재, 정본 자체는 불변). 이전 rev9: 2026-08-05 (★★★**§1-31 신설[claims-auditor CONFIRMED
(scoped), doc-steward 기록 — §0 최상위 열린 항목 종결(behavioural), 새
성능 판정 0건]**: S2(job 873015, sticky ON, T8 d16/d54, ShareGPT rate 2,
n=8 블록, cudagraph ON)를 2026-08-05 독립 재현. pooled per-token ITL p50 =
**28.92ms**(d16, t95[28.81,29.02])/**12.04ms**(d54, t95[11.96,12.12]) —
사전등록 `[28,34]ms` 안, `split_frac` 라벨 미사용으로 재현. §1-28/§1-30
§0의 이분법이 **behavioural하게 종결**: (i) 872077의 `decode_sms==16`
라벨은 하드웨어 형태로 REFUTED·라벨 형태로 CONFIRMED(872077 d16이
decode-busy 시간의 96.2%를 D108에서 보냄) / (ii) C2의 28–31ms=셀 배치
성질은 DISFAVOURED(keepalive 없이·batch 2.6배 작게 0.93×로 재현) — 살아남는
답 (iii)은 `split_frac≥0.90`이 D-파티션 클래스를 격리·완결 못한다는 것.
**scope: selector-level, 하드웨어 SM 부여 직접 프로브(S3) 없음**
(`decode_sms`는 `arbiter.sm_counts[stream_index]` 재진술).
`E1_DECODE_REALIZED`(시간가중) = 0.9990±0.0017(d16)/0.9994±0.0008(d54),
16/16 cell-block ≥0.995. **기전 독립 도출**: `runtime_snapshot`이
개수-서브샘플(`TRACE_EVERY=32`)이고 decode-busy 조건부 케이던스가 4 arm
전부 정확히 16 decode step(0.177–0.465s)이라 ITL 구간 하나가 스냅샷
1/16개를 걸침 ⇒ 기대 순도 ≈6%, S0-R mode 분해의 6.6–9%와 일치. 배치 기여는
세 추정 모두 2.3–4.7%뿐(노드/바이너리 상한은 d54 companion ≤1.097×) ⇒
아티팩트 배제. ★**d54 companion은 사전등록 [13,16] 미달**(관측 12.03) —
`PREREG_S2` §5.3에 이 조합의 규칙 없음, **인용 시 필수 동반**. `DESIGN.md`
§4.3.12(f) 판별 예측(T8≈1.85)도 빗나갔으나(관측 2.402) 판별 arm(Ha8)이
미제출이라 **설계상 미판정**(모형 반증 아님). ★**이 런에는 결과 게이트가
0개**(`AMBIG_FRAC`·`MIN_N_SPLIT`·§3.1 일치검사는 sticky ON 하 항등식) —
**방법론 게이트 #9의 네 번째 재발**(§3-23 신설). `S2_ANALYSIS_2026-08-04.md`
의 "Instrument check" 문단(케이던스 ~2.0ms 서술)에 **정정 표시**(원문
보존) — decode-idle 값을 decode-busy로 오인, 90–260× 오차. **E1은 이번
회차로 열리지 않는다 — 4개 독립 사유**: `G_LEVER`/`G_FLAT` 여전히
UNDETERMINED(post-sticky sd ~14× 붕괴로 임계 사전등록 시 null 채택 편향
위험) / sticky 기판이 estimand를 바꿈(decode-busy 시 prefill이 벽시계
~77% 유휴, co-located 예산 배분이 아니라 단일-테넌트 측정에 가까움) /
prefill 축 미통제(TTFT p50 46.3→63.2ms 무기전) / 음성대조 구조적 부재(d16
UNSPLIT n=0/8, "레버 부재"가 아니라 "이 estimand 하 정의 불가")+S3 미실행.
**긴장 A(HE2 vs C2)는 전혀 닫히지 않았다.** C2 자체(레버 존재,
2.36–2.91×, scoped)의 등급·수치는 **불변**(자기완결적 캠페인, 이 종결의
영향 밖) — 바뀐 것은 "C2와 E1 격자가 같은 물리량을 재는가"라는 상위
질문뿐이고 이제 그쪽으로 confirmed. 다음 gate: (α) sticky-ON 고-D 대조
셀(prereg p50→12–13ms, 반증 가능한 유일 실험) / (β) OFF+`TRACE_FORCE_
PREFILL=1` 1블록 / (γ) S3(하드웨어 층) / (δ) 같은 바이너리 OFF arm(등재
선행조건 아님). 상세 `../PROJECT_STATUS.md` "8B decode-SM 프론티어"
"2026-08-05" 소절, `../workspace/engine-port/results/s2_sticky/
S2_REPLICATION_2026-08-05.md`(전문), `results/s8_frontier/DESIGN.md`
§4.3.16]. 이전 rev8: 2026-08-04 (★★**GROUP A/B 감사 반영[claims-auditor +
result-analyst X2′, doc-steward 기록 — 새 성능 판정 0건, 전부 등급
강등·스코프 축소·개념 등재]**: §1-3(Diff A/B) — 계측 결함 2종(버킷
비대칭 + 누산기 러닝평균) 확인, "WIDE 스윕으로 확증"·"L=2000선
lever는 L↓서 열림" **철회/REFUTED**(steady L=256 밴드 [1.24,1.34]·
L=512 1.198·L=2000 1.0081[1.0078,1.0084]), 정책 단위 환산 시 1.032/
1.031로 소멸(비독립), §1-3 판정 자체는 불변(기전 서사만 변경). §1-5 —
"knee=108을 floor에 대입하지 말 것" 오독 방지 각주 신설 + job 858811
계측 결함 3건 확인(버킷 비대칭·`full`-첫-arm 편향·물리 불변량 위반)으로
"attn 비중 5%→79%" 등 인용 금지. §1-13 — 천장 절단(ceiling censoring)
개념 각주(LO goodput은 도착률에 절단돼 지표 무신호, 레버 부재 아님).
§1-1(P1) — no-cudagraph 비운영점 한정·rate1 동률·cudagraph가 fused
死因(TPOT>60ms)을 제거한다는 반대 증거 병기(검증 실험 예정). §1-17
(positioning) — "모든 수정이 static 수렴"의 '모든'은 reactive 계열
각주(Step D/F 온라인 feedforward는 이미 시험, offline decode-floor
예측만 미시험). §1-20(P6, disaggregation +16%) — 정본 자신이 근거
캠페인을 n=1~2·overload-only·underpowered로 이미 기록 ⇒ "확정"→
"n=1~2, 미확증" 강등. §3에 항목20(층위 구분 3종: 절대≠상대 민감도·
천장절단·baseline 상이)·21(C2→벡터1 탄력도 정합 REFUTED, 격자 이전
금지 재실증)·22(재사용 인용 금지: "d16 1.056 vs d24 6.240"=폐기벤치
최량rep, 집계 Diff A 교차점=U_H 가정 의존 판정불가) 신설.
`PROJECT_STATUS.md`(확정된 결과 1·8B decode-SM 민감도 측정 노트 C2
국소탄력도)·`reports/paper/CLAIM_EVIDENCE_MATRIX.md`(Claim A C2 부기)·
`research_arc.md`(§S3 정정 재정정·L=2000 REFUTED·mamba 역행 크기정정
9.05%→2.59%·Diff A 지수정정 1.916/0.954·d16 6.240 인용금지 각주)·
`longcontext_trace_plan.md`(Diff A 지수정정)·
`workspace/engine-port/results/prefill_knee/diffA_vs_diffB_table.md`
(no-cudagraph micro 라벨 + L=2000 오염 표시)에 동반 반영. 근거
`reports/layertype_dynamic_POSITIVE_2026-08-04.md`·`_NEGATIVE_
2026-08-04.md`·`_JUNCTION_2026-08-04.md`(종합 문서, 정본 아님) +
`workspace/engine-port/results/prefill_knee/AGGREGATE_COMPOSITION_
2026-08-04.md`(REV.2, UNAUDITED 배너 유지)]. 이전 rev7: 2026-08-03
(★★★§1-30 신설[같은 날 4차 속행, doc-steward
기록 — **성능 판정 0건, GPU 런은 별도 제출 중·결과 없음**]: §1-28 §0의
이분법((i)/(ii))이 **유지 불가**임이 확인됐다. E1 SPLIT 모집단이 이봉이고
윗봉이 C2 셀별 p50과 1–2% 일치, 아랫봉은 같은 job UNSPLIT과 통계적으로
동일이라는 감사자 재프레이밍(자기감사, 방법론 교훈 12)을 result-analyst가
독립 실행으로 재검증(`S0R_REPLICATION_2026-08-03.md`) — 행 1·3 재현, 행
2는 순서만(d24−d16 미해결), **행 5(클럭) 미발화**(최적 δ=+0.10s서도 slow
share 11.4%). ★**행 4(음성대조)가 강한 형태를 죽였다**: UNSPLIT(108 SM)
클래스도 전 셀에서 이봉이고 느린 봉 위치가 셀을 따라간다(d16 8,893개 중
7,903개=88.9%가 UNSPLIT 라벨) ⇒ SPLIT은 배타가 아니라 **농축**(2.33–
3.24×). **세 번째 후보 (iii)**이 실측으로 문서화됐다: 두 job은 같은
축이나 `split_frac≥0.90`이 D 파티션 실행 토큰을 순수하지도 완전하지도
않게 잡는다. 남은 두 읽기(셀 수준 현상 vs 클럭 오프셋 누출)는 **오프라인
분리 불가** — S2(GPU, 별도 제출 중, 결과 없음)가 인과 시험. **철회 3건**
(메인 세션이 같은 날 앞서 씀): "§0 stands as written"·"aggregation-
invariant"·"11.09는 집계 단위 미기록"(**틀림** — 산출자는
`m3_conditional.report_conditional` [3] `sp_p50=11.0905`,
`m3_conditional.py:158-161,251-262,316-329`에 문서화, 없는 것은 stdout
저장분뿐). **재사용 계측 결함 2건**: `c2_anchor.py` 표 [5]가 M8 전체·
Ha8 d16을 조용히 누락(`meta`가 `t0_monotonic_s` 분기 안에서만 채워짐,
c2_anchor.py:181-187 — Ha8 d16은 **돌았다**, `itl_ms_p50=112.84` n=5200,
빠진 건 텔레메트리 앵커뿐) · mode estimator 60ms 상한은 arm-이식 불가
(Ha8은 토큰의 0.16%만 창 안). **증거 수준**: 강한 형태(레버=D-SM 실행)는
**채택 불가**(음성대조 반증), 약한 형태(이봉·농축 2.33–3.24×)는
**재현됨(독립성 부분적** — 추정량=감사자 제안, 사전등록=메인 세션,
실행만 독립**)**. §0의 (i)/(ii)는 **여전히 미판정**(이제 3지선다).
게이트 S1(§4.3.13)은 "부분 실현" 분기가 없어 **현 상태로 실행 불가**.
`G_LEVER`/`G_FLAT`는 §4.3.12(d) 그대로 UNDETERMINED. 상세
`../PROJECT_STATUS.md` "8B decode-SM 프론티어" "2026-08-03(4차)" 소절,
`results/s8_frontier/DESIGN.md` §4.3.15, 사전등록 3건(`PREREG_S0_AXIS_
2026-08-03.md`·`PREREG_S0R_MODE_2026-08-03.md`·
`PREREG_S2_STICKY_ITL_2026-08-03.md`), 재현 판정
`S0R_REPLICATION_2026-08-03.md`]. 이전 rev6(★★★§1-28/§1-29 신설[같은 날
3차 속행, doc-steward 기록]: `c2_anchor.py`로 시도한 C2→`G_LEVER` 앵커
도출을 claims-auditor가 감사 —
**주장 1만 생존, 2–5 전부 REFUTED/NOT-YET-SUPPORTED**. ★**§0 신규 결정적 발견**:
같은 arm·같은 서버 플래그·매칭 batch에서 C2 865493과 872077(E1 격자)의 "decode
16 SM" per-token ITL p50이 **2.6× 다름**(28.79ms vs 11.09ms) — 872077의
`decode_sms==16`이 실제 16-SM 실행이 아니거나(DESIGN §4.3.11 미검증 잔여층),
C2의 28–31ms가 decode-SM 비용이 아니라 그 셀 배치 성질이거나 둘 중 하나이며
**이 층이 872077 전체와 sticky 결과가 딛고 선 바닥**(최상위 열린 항목).
`G_LEVER`/`G_FLAT`는 §4.3.12(d) 그대로 **미결정 유지**(이 시도로 해소 안 됨).
D=54 앵커 측정(jobs 872920/872921)은 제출 17분 뒤 취소(감사가 전제 폐기 +
독립 발견인 keepalive 토큰 초과로 s8_scaleup 재현 불가 확인) — 취소됐으나
설계는 재사용 가능. 독립 수렴으로 **C2의 높은 residency는 decode 파티션
제어가 아니라 keepalive 워크로드 장치의 산물**임을 확인(3경로 독립 도달).
정본 정정 2건: "파티션 활성률 0.66–0.93"은 count-weighted(시간가중은
0.99+); 108 SM 시간은 warmup 아니라 drain 전용. 상세
`../PROJECT_STATUS.md` "8B decode-SM 프론티어" "2026-08-03(3차)" 소절,
`results/s8_frontier/DESIGN.md` §4.3.13–4.3.14,
`results/s8_scaleup/NOTES_D54_ANCHOR_2026-08-03.md`]. 이전 rev5(★★§1-27
신설[같은 날 2차 속행, doc-steward 기록]:
(I) claims-auditor가 §1-26의 `A_free` 결함을 대체하는 **조건부 per-token
추정량**(`m3_conditional.py`)으로 estimand를 이관 — SPLIT/UNSPLIT 라벨(≥0.90/
≤0.10, 사이는 배제), primary `p95(SPLIT)` 비, UNSPLIT control(대비 정의상 0)
[AUDITED, blocking-threshold 스윕(3)만 UNAUDITED — 감사자가 자기 산출을 자기가
감사한 형태]. (II) engine-porter가 `PDMUX_STICKY_PARTITION` 구현 완료 —
decode-busy 시 무분할 fallback 우회, decode-empty 시엔 의도적 release(hold
아님), OFF는 short-circuit으로 patch 전과 byte-identical, correctness gate
전부 PASS(CPU 회귀 40 tests + sticky 단위 테스트 12 + GPU smoke job 872800
byte-identical 출력), realized 관측(n=1) `E1_DECODE_REALIZED` OFF 0.0839→
ON 1.0000. **구현 완료 ≠ 성능 주장 성립.** (III) sticky 런 사전등록: 872077
소급 재분석은 DIAGNOSTIC 전용, primary 1개 선언, 게이트 4종, **`G_LEVER`/
`G_FLAT`는 미결정으로 기록**(스케일 불일치로 기존 1.5/1.15 이전 불가). 상세
`../PROJECT_STATUS.md` "8B decode-SM 프론티어" "2026-08-03(2차)" 소절,
`results/s8_frontier/DESIGN.md` §4.3.10–4.3.12]. 이전 rev4: ★★★§1-26
신설[같은 날 속행: 메인 세션이 세운
"decode 실현 4–19%가 g를 attenuate했다"는 보정 가설을 claims-auditor가 REFUTED —
control-arm reductio(T8에 같은 보정 적용 시 corrected g 21–29×로 C2를 10배
위반)·de-engagement 직접 실험(w=0에서도 g 거의 불변)·"A(108) 셀 무관" 가정의
실측 위반(UNSPLIT-only 부분집합만으로 T8 헤드라인 재현) 3중. 동시에 872077의
NO VERDICT 사유가 "CI 폭 부족"에서 **"estimand 미식별"**로 확장됨 —
`initialize_stream_groups`가 마지막 (0,108) 무분할 그룹을 항상 덧붙이는 기판에서는
"decode가 D SM에서 돌았다"⟺"prefill이 동시에 실행 중이었다"가 같은 사건이라 이
격자의 어떤 통계도 decode-SM 탄력도와 prefill 간섭을 분리 못 함(§1-24와 결합).
`g = A_free(d16)/A_free(d54)`는 **이 격자 한정 은퇴**(sticky partition 기판
수정 전까지 인용 금지), 블록 증설 재실행은 **선행 금지**. `A_free` 추정량
자체도 결함 확인(blocking 필터가 prefill 작업의 74–77%를 통과시켜 §1-24 stall이
d16–d54까지 오염 범위 확장 + 극단 percentile 퇴화) + arm 비교의 decode-batch-size
미제거 교락. §3에 항목15(항등식에서 파생된 양을 자유 모수처럼 나누지 마라) 신설.
상세는 `../PROJECT_STATUS.md` "8B decode-SM 프론티어"·`results/s8_frontier/
DESIGN.md` §4.3.9]. 이전 rev3: ★★§1-25 신설[pin 게이트가 항등식이었고, decode 축은 라벨이 4–19%만 실현 — claims-auditor 회부 + 77,688 스냅샷 독립 재현]. 이전 rev2: ★§1-24 신설[M4: ITL 꼬리 = monolithic prefill, 구조적] + §1-23 따름정리 정정[`--max-mamba-cache-size`는 공통 절대상수가 아니라 `= cap` 규칙]. 2026-08-01 실험 4건은 claims-auditor 회부 완료 — 2 REFUTED / 2 NOT-YET-SUPPORTED / 1 CONFIRMED, 상세·철회 목록은 `../PROJECT_STATUS.md` 해당 절의 supersession 박스. 이전 rev1: 진행 상태 갱신만, 결론 개정 아님 — §1에 항목23
"`--max-running-requests`는 arm 계열마다 다른 손잡이 · `kv_mamba_occupancy=1.0`
은 항등식" 신설(**소스 읽기로 검증되는 코드 사실**, 성능 판정 아님) + §3에
항목14 "항등식을 증거로 쓰지 마라 — 이 양이 재려는 것과 논리적으로 독립인가"
신설(13번과 같은 뿌리, 실패 모드는 다름). 2026-08-01 실행된 E1 전제 실험
4건(jobs 870295/870296/870297 용량 스캔, 870301 batch-cap)은 **전부
claims-auditor 미통과 = 이 문서에 결론으로 올리지 않는다** — 상태 기록은
`../PROJECT_STATUS.md` "열린 긴장"의 "2026-08-01 실험 4건" 소절에만 있고
**인용 금지**다. §1-4 얽힘의 KV 갈래는 등급 불변(occupancy 데이터는 생겼으나
de-confound 안 됨 — `../PROJECT_STATUS.md` "확정된 결과" KV 항목).
이전: 2026-07-29 (진행 상태 갱신만, 결론 개정 아님 — §3에 항목13
"집계 단위를 먼저 정하고 추정 대상과 맞는지 논증하라" 신설(9·10번을 특수
사례로 흡수) + §1에 항목22 "green-context 분할은 decode가 비면 무분할로
auto-revert — 셀 라벨은 목표이지 실현 배분 아님" 신설(관측 사실, 성능 판정
아님). `results/s8p_prefill/`(prefill 축 SM 민감도) 완료·claims-auditor 미통과
(정본 인용 금지 유지), `results/s8_frontier/`(E1) 하네스 구축 완료·본 스윕
미실행. 상세는 `../PROJECT_STATUS.md`). 이전: 2026-07-28 (★★★claims-auditor가
`workspace/engine-port/results/
s0_deconfound/DESIGN.md` §5 사전등록 게이트를 집행 — **부분 GO**. **C1
CONFIRMED**: §1-21이 인용한 Stage 0 D108 무경합 앵커는 코드 버그로 실제로는
decode 16 SM이었음을 3중 독립 증거로 확인 → §1-21 판정2(NULL)·판정3(게이트
non-binding)을 **철회**, 판정1(CONFOUNDED)만 생존, §5-6 long-ctx open item을
"게이트 실패로 보류"에서 **"게이트 미실행"**으로 복원. §3-9(방법론 교훈)는
정정이 아니라 **재작성**(무경합 앵커·음성대조 모두 고장났던 사실을 반영). **C2
CONFIRMED(scoped)**: prefill 16 SM 고정 시 decode ITL SM16→SM92 2.36–2.91×,
4 arm 모델-무관(8B 측정 노트, `../PROJECT_STATUS.md` "8B decode-SM 민감도"
절). **C2b("hybrid 급락=Zamba2 성질") NOT-YET-SUPPORTED로 강등.** §1·§1-5·
§1-7·HE0/HE2는 **철회하지 않는다** — 긴장 2건(HE2 vs C2, r0c 부분 복권)을
열린 항목으로 기록. 상세는 아래 §1-21·§3-9·§5-6, `../PROJECT_STATUS.md`).
이전: 2026-07-26 (★★**Stage 0(long-ctx L−2 게이트)**: 운영점 decode
SM-무감각을 hybrid·pure-Transformer·pure-Mamba·ctx≤16k 전부로 확장 확인 —
non-binding, long-ctx 충돌 가설 이 regime서 붕괴, HE0/벡터1 ctx-무관으로 강화 —
★★★2026-07-28 이 판정의 핵심 근거가 철회됨, 위 참조. 상세 §1-21). 2026-07-19
(★★**§1-16 반증 — tight SLO로 컨트롤러를 실제 재튜닝하면 동적은 best-static에 크게 열위(§1-17). SLO 엄격도와 무관하게 decode-heavy static 지배 확정.** 2026-07-18 §1-16의 "tight→동적 우위"는 재스코어 아티팩트로 격하). 2026-07-17 (변화-trace n≥4 — HE0 견고 확정 + 게이트=auto-tuner 규명).
과거 보고서는 `deprecated_reports/`로 이관(이력 보존용, 내용은 당시 시점 기준이라 현재 결론과 충돌할 수 있음). ★**2026-07-24**: 저장소 전체 격리처를 단일 `deprecated/`로 통합하면서 이 디렉터리는 [`../deprecated/reports/quarantine_engine_port/`](../deprecated/reports/quarantine_engine_port)로 물리 이동했다(내용·판정 불변, 경로만 변경). ★**2026-07-24 스코프 정정(claims-auditor 감사, doc-steward)**: §5-7의 "⇒ 동적 제어 트랙 완전 종결" 표현이 [`../PROJECT_STATUS.md`](../PROJECT_STATUS.md)(Claim D/E = 미검증) 및 §1-20(decoupled substrate에 +16% headroom 실재)과 모순돼 **overclaim으로 범위 축소**. §1-7·§1-17(single-worker·SM-split·reactive 제어, n≥4)의 확정성은 불변. 상세는 §5-7·§5-8. ★★**2026-07-24 구조적 정정 추가(engine-porter 코드 리뷰, 읽기전용, [`r2_decoupling_review_2026-07-24.md`](r2_decoupling_review_2026-07-24.md))**: §5-8(a)가 §1-20의 decoupled substrate 후보로 언급했던 `PDMUX_TRUE_DUAL_WORKER=1`은 file:line 근거로 **control-plane dual-worker(호스트 스레드/큐/role만 분리)일 뿐, running batch/KV/SM은 전면 공유**로 확인됨 — decoupled substrate에 **해당하지 않는다**. 상세는 §5-8(a). ★**2026-07-25 §5-8(c) 업데이트**: HE0-reopen 벡터1(G2.0 disjoint 스윕, n≥4 재시도)이 (c)의 underpowered 상태를 부분적으로 검증 — **ILL-POSED at rA5 판정**(escape hatch 근거로 "지지 안 됨"이나 "종결"도 아님, §1-20과는 무관한 별도 축). 상세는 §5-8(c). ★**2026-07-25 §5-8(c) 추가 업데이트(de-cliff stage-1, 실험 2026-07-24~25·기록 2026-07-25)**: rA5 절벽을 벗어난 rA2에서 disjoint를 찾지 못함(`d54`가 양 phase 동시 커버) — **PLAUSIBLE closure이나 CONFIRMED 아님**(claims-auditor 반증 3항목 + narrow-rA 확증 sweep 선행 필수). 한 눈금 전진이지 종결 아님. 상세는 §5-8(c). ★★**2026-07-25 §5-8(c) 최종 업데이트(narrow-rA 확증, `g2_0_rasweep`+`g2_0_raconf`)**: g2_0_rasweep(120 job, off-cliff sub-band rate≤2.75서 disjoint 재확인 없음)이 전이대를 rate 3.0–3.5로 좁혔고, claims-auditor pre-registered 24-job 확증 열 `g2_0_raconf`(rate{3.5,3.75}×{d44,d54}×n6)가 결정 규칙을 충족 — **companion collapse**(rate3.5: d44 0.953±0.035≈d54 0.948±0.035; rate3.75: d44 0.932±0.042<**d54 0.948±0.062**, REOPEN 전제 양쪽 붕괴). **벡터1(short-ctx disjoint) = CONFIRMED closure(scoped)로 종결** — PROJECT_STATUS "벡터1" 절·`g2_0_raconf/raconf_final_verdict_2026-07-25.md` 참조. §5-8(c)의 미결 갈래 (c)는 이제 닫혔다(scope=short-ctx drained; long-ctx·§1-20 spatial decoupling은 별도 미결).

---

## 0. 한 줄

**PD-mux(prefill↔decode SM 분할)는 이득이나, 그 위의 "똑똑한 정책"은 전부 실패했다.**
layer-type 기반 정책은 全형태 死. 동적(SLO-aware/binding-first/feasibility-gate) 제어는 **best-static을 못 넘는다 — 관대·tight SLO 양쪽에서 확정**. 관대 SLO(TTFT 3s): d44 3.220±0.013 > bind+GATE 3.132±0.019, 5.4σ. ★**tight SLO(chat 300/50ms)로 컨트롤러를 실제 재튜닝해도 열위, 오히려 격차 확대**: rate8 attainment d44 **73.2%** ≫ d34 49.6 > bind+GATE 44.3 > bind 40.6 (28.9%p≈10σ, §1-17). 컨트롤러가 tight TTFT에 반응해 prefill-ward 이동→decode 굶김→§1-4 얽힘 트랩→TTFT 악화.
★**한때(2026-07-18, §1-16) tight SLO 재스코어에서 "동적 우위"로 보였으나, 컨트롤러를 실제 재튜닝한 직접 측정(§1-17)이 이를 반증** — 재스코어는 3s-튜닝 컨트롤러의 정착 static 위치를 사후 채점한 아티팩트였다. ⇒ **"decode-heavy static 지배"는 SLO 엄격도와 무관한 결론.**
**최적 split은 모델 상수가 아니라 *decode 부하*의 함수**이며, 실전 권고는 **peak decode 부하 기준 decode-heavy static 고정**.
살아남은 동적의 유일한 값어치는 **성능이 아니라 견고성**(게이트가 트랩 붕괴를 막음: 1/4 → 0/5; 정체는 **틀린 static에 조기 수렴하는 auto-tuner**).

---

## 1. 확정 결론 (robust — 노이즈·재현성 검증 통과)

| # | 결론 | 근거 |
|---|---|---|
| 1 | ~~**PD 분리 자체는 항상 이득**~~ → ★반증/정정(2026-08-05, claims-auditor, jobs 873944/873945) **PD-mux 활성화는 운영점에서도 꼬리 SLO goodput 이득 — 술어·모델·워크로드 한정, 기전 귀속 미확립** | agnostic이 fused를 4모델 전부서 이김. ★**스코프 축소(2026-08-04, claims-auditor)**: 이 4-모델 캠페인은 **전부 `--disable-cuda-graph`**(no-cudagraph 비운영점, `triage/p1_7_bench_one.sbatch:42`)이고 **rate 1에서는 동률**(도착률 천장)이며 **운영점(cudagraph-ON) 대조는 어느 모델에서도 측정된 적 없다**. ★**반대 증거 신규**: fused의 死因은 **TPOT > 60ms 임계 초과**(Granite rate4 TPOT 61.21)인데 **cudagraph가 그 벽을 제거한다**(plain TPOT 62.70→13.51ms, rate4 82.41→**54.04ms=60ms SLO 통과**, `workspace/engine-port/results/cudagraph_probe/cudagraph_results.md` Probe 1 — Zamba2 단일모델 관측이라 Granite에 직접 이식은 아니나 死因 메커니즘이 cudagraph로 해소 가능함을 시사) ⇒ **"PD 분리 자체는 항상 이득"이 운영점에서 축소되거나 소멸할 가능성**. 검증 실험(2모델×{plain,agnostic}×cudagraph-ON×n≥4) 진행 예정, 결과 없음. 상세 [layertype_dynamic_POSITIVE_2026-08-04.md](layertype_dynamic_POSITIVE_2026-08-04.md) §2.0. ★★★**반증/정정(2026-08-05, claims-auditor 감사, jobs 873944/873945, Zamba2-2.7B·Granite-4.0-h-micro-base, n=5 paired, 사전등록 `results/p1_opint/PREREG.md`, 판정서 `results/p1_opint/P1_OPINT_RESULT_2026-08-05.md`[claims-auditor 감사 반영본])**: 위 "축소되거나 소멸할 가능성"은 낡았다 — 운영점(cudagraph-ON) 대조가 처음 측정됐고, **소멸하지 않았으나 "확인"으로 올라가지도 않는다.** 정본 goodput 술어(게이트 #4: TTFT≤3s ∧ 요청 내부 token-ITL p95≤60ms)로 채점하면 `--enable-pdmux`(agnostic v1)가 fused(plain)를 **두 모델 전부·rate 2–6 전 셀에서** 이긴다(rep 부호 5/5, paired CI 0 배제). **인용 가능한 정량치는 두 arm이 모두 정상상태(큐 성장·런길이 표류 없음)인 셀뿐이다**: Zamba2 rate2 **+40.5%**[CI +0.431,+0.620 req/s], rate3 **+185.8%**; Granite rate3 **+11.7%**, rate4 **+27.0%**. 임계 사다리 40–300ms 전 구간 부호 불변(=metric cliff 아님, ★신규 방법론 게이트 #12: 임계 지시함수 판정은 임계 사다리와 큐-성장 검정으로 견고성을 보여라). **Zamba2 rate4·6과 Granite rate6은 한쪽 이상이 용량 위**여서 **부호만** 인용한다. Granite rate2는 경계(+3.4%, 임계 48ms로 내리면 소멸). ★**술어 의존(중요)**: 사전등록이 채택한 mean-ITL(TPOT) 술어로는 Granite에서 부호가 뒤집힌다(−0.3~−1.6%, CI 0 배제). 그러나 그 술어는 Granite rate3·4에서 **위반 요청이 0건이라 goodput ≡ throughput**(항등, ★방법론 게이트 #6의 새 사례: 판별력 0인 술어에 "차이<3%⇒강등" 규칙을 적용하면 무신호가 강등으로 둔갑한다)이고 남는 차이는 **3% 하한 미만**이다. ⇒ **"Granite에서 P1 전면 강등"은 채택하지 않는다.** 채택하는 것은 "**mean-ITL 술어는 이 regime의 Granite에서 fused의 死因을 잡지 못한다**"뿐이다. 마찬가지로 Zamba2의 사전등록-술어 유의 셀(r4·r6)은 전부 절벽 위라 **그 경로로는 "확인"이 성립하지 않는다** — ⇒ **"Zamba2 P1 운영점 확인(사전등록 지표)" / "Granite P1 전면 강등" 두 문장 모두 정본 등재 금지.** ★**"항상 이득"은 철회한다 — 비용이 실재한다**: pdmux는 정상상태 per-token decode를 **5–45% 늦추고**(중앙 token-ITL 9.67→10.50 / 22.57→29.30ms[Zamba2], 6.80→7.63 / 8.96→12.98ms[Granite]), 저부하 TTFT p50를 체계적으로 악화시키며(Zamba2 r2 0.164→0.230s), Granite raw 처리량은 **−0.3~−2.6%**(CI 0 배제)다. **이득은 꼬리, 비용은 중앙이다.** ⚠️**기전 귀속 미확립(3-플래그 + 미조율 baseline, ★신규 방법론 게이트 #13: arm 대조가 엔진 제약으로 다중 플래그를 강제하면 그것은 묶음 처치다)**: 이 엔진에서 `--enable-pdmux`는 `--chunked-prefill-size -1`·`--disable-overlap-schedule`을 **assert로 강제**하므로(`sglang/srt/server_args.py:6125-6137`) 측정된 처치는 세 플래그 **묶음**이다(`p1op_run.sbatch:55`). 또한 fused arm은 측정 대상 실패 모드에 대해 **미조율**이다 — 관측 stall은 prefill 배치 자체이고(클러스터 지속 166ms@r2→474ms@r4, 케이던스 1.1–1.9/s), `--chunked-prefill-size` 축소와 `--enable-mixed-chunk`(`sglang/srt/managers/scheduler.py:2525-2541`)가 정확히 그 축의 fused-측 레버인데 **둘 다 기본값**이다. ⇒ **"PD 분리(SM 분할) 자체가 원인"은 NOT-YET-SUPPORTED.** 현재 지지되는 것은 "**이 엔진에서 PD-mux를 켜면 기본 설정 fused보다 꼬리 SLO goodput이 좋다**"이다. ⚠️**실현 파티션 미측정**: `PDMUX_TELEMETRY_PATH` 미설정으로 realized `(prefill_sms, decode_sms)` 재집계 불가 ⇒ **파티션·동시성 기전 문장은 정본 금지**(Stage 0 D108 전례), arm 수준 대조만 유효. ⚠️**2026-08-04 반대 증거(위 Probe 1) 정정**: "cudagraph가 fused의 死因(TPOT>60ms)을 제거한다"는 **중앙값 근거**였다. 이번 측정에서 plain rate4는 중앙 TPOT 48.5ms인데도 요청의 **31%**가 mean-ITL SLO를, **86%**가 정본 술어를 위반한다. **중앙값이 SLO 아래 ⇏ 요청 통과.** cudagraph가 제거한 것은 **per-step 벽**이고 남은 것은 **blocking 벽**이다. **위생(허가)**: `boot_ok=1` 24/24, `CORRECTNESS=PASS` 24/24, 서버 로그 traceback/CUDA error 0건, **양 arm cudagraph ON**, piecewise는 **양 모델·양 arm 모두 OFF**(대칭, 교락 아님), 페어링 무결(input_lens 40/40 완전 일치), arm 순서 무작위화 로그 확인. **단 rate 순서는 미무작위화**(`p1op_run.sbatch:148` 고정 2→3→4→6, arm 대조는 paired라 무영향)이고 **n=5는 동일 job·동일 노드 반복**(CI는 노드 내 재현성이지 노드·날짜 간 재현성이 아니다). **등재 금지**: "Zamba2 P1 운영점 확인(사전등록 지표)"/"Granite P1 전면 강등"(위 사유, 양쪽 다); r4·r6 크기 **+41.8%/+398%/+501%/+634%/+174.9%**(런길이 의존 + 캠페인 간 2.2× 불일치, 2026-07-13 probe3 agnostic r4 1.721 vs 3.834); 용량 수치(plain≈4/agnostic≈5, Granite 7.5 vs 6.5)를 n=1 지시값 이상으로(Granite r8 비단조=drain-tail 아티팩트); 파티션·동시성 기전 문장·split 수치(★2026-08-06 Gate 1로 **부분·조건부 해소** — Zamba2 rate{2,3}·decode-busy∧prefill-in-flight 구간의 selector 라벨 `(74,34)`만 인용 가능, rate 4·6·Granite는 여전히 금지, 아래 Gate 1 결과 블록 caveat 전체 필수 동반); "PD 분리 자체" 기전 귀속(Gate 1로도 해소 안 됨 — Gate 2 해소 전); NemotronH/Falcon-H1 확장·ShareGPT/변화-trace 확장(미측정); **2026-07-13 `cudagraph_probe` 수치(위 Probe 1)와 이 캠페인 수치의 직접 대조**(격자 간 이전 금지, 게이트 #11). **다음 gate**(우선순위순): ~~Gate 1 telemetry 재현런~~ → ✅**완료(2026-08-06, job 874478) — 조건부 채택**, 상세는 아래 ★★★★★ Gate 1 블록 참조(부분·조건부 해금, G1-a–d 후속 등재) → Gate 2 4-arm 분해(plain/plain+chunked-1+no-overlap/plain+chunked512/agnostic × rate{2,3}(+4) × n=5, 사전등록 판별: `plain+aux≈agnostic`(3% 이내)면 §1-1 **플래그 아티팩트로 붕괴**, `plain+chunk512≈agnostic`이면 §1-1을 "PD-mux는 head-of-line blocking을 없애는 여러 수단 중 하나"로 재작성 — **논문 신규성 축이 바뀐다**) → Gate 3 나머지 2모델(NemotronH·Falcon-H1) 운영점 대조("4모델 전부" 인용 전제, 안 하면 §1-1은 영구히 2모델 문장) → Gate 4 sustainable-rate n≥4 직접측정(r4/r6 크기 인용 전제, 현재 후순위). **scope(축약 금지)**: {Zamba2-2.7B(triton, ctx4096)·Granite-4.0-h-micro-base(flashinfer, ctx8192), A100 108-SM green-context, **cudagraph-ON**, `--disable-radix-cache --mem-fraction-static 0.82 --max-running-requests 48`, `random-ids` **in2000/out96**(prefill:decode 토큰비 ≈21:1), 정상상태 단일-rate 격자 {2,3,4,6}, 120 프롬프트, n=5 paired(동일 job·동일 노드), agnostic v1(`pdmux_a100_smoke.yml`, sm_group_num 4)}. **NemotronH·Falcon-H1은 운영점 미측정 ⇒ "4모델 전부"는 더 이상 쓸 수 없다.** ShareGPT·변화 trace로 확장 금지(게이트 #2). torch 2.9.1에서 pdmux는 엔진 자체 경고 대상(`server_args.py:6141-6147`). 상세 `workspace/engine-port/results/p1_opint/P1_OPINT_RESULT_2026-08-05.md`, 신규 방법론 게이트 전문은 `../PROJECT_STATUS.md` "방법론 게이트" #6 새 사례·#12·#13, gate 목록 전문은 같은 문서 "다음 실험 gate" ★★★★**통계 방법 층 정정(2026-08-06, claims-auditor Gate 2 설계 감사 2회 + result-analyst 독립 재현, `workspace/engine-port/results/p1_gates/verify/`) — 새 성능 판정 아님, primary를 t-CI로 교체해 재채점한 결과.** `paired_bootstrap_ci`(n=5 percentile bootstrap of mean, BCa·studentization 없음)는 실 coverage **0.840**(100k MC, 명목 95%의 한쪽 오류율 ≈8.0%=명목 3.2배, 원인은 seed·정규성이 아니라 **n=5 그 자체**)이라 소표본 판정에 부적합 — 저장소 안에 `m3_analyze.py`·`tfgate_analyze.py`가 이미 독립으로 t-CI로 전환한 동일 진단이 존재했다(도구 규율 실패, 신규 방법론 게이트 #27). t-CI로 재채점 시 **Granite rate3(+11.7%)는 t-CI [−0.0032,+0.6132]가 0을 포함(p=0.0515)** → 인용 목록에서 제외(**미검증으로 재분류, 철회 아님** — boot CI는 여전히 0 배제), ⇒ **인용 가능 정량치는 4개→3개(Zamba2 rate2 +40.5%/rate3 +185.8%, Granite rate4 +27.0%)로 축소**. "임계 사다리 40–300ms 전 구간 부호 불변(=metric cliff 아님)"도 정정 — 부호가 유지되는 구간은 **T∈[40,113.0)ms뿐**이고 그 위에서 인용 가능 4셀 중 3셀(Zamba2 r2, Granite r3, Granite r4)이 음으로 뒤집힌다(뒤집힘의 정체는 절벽이 아니라 **술어 포화** — 뒤집히는 셀은 T≥150에서 양 arm 위반 0/0, goodput≡throughput; ★§3-24가 REFUTED한 "검정력 0인 임계 사다리"의 재발), 부호가 끝까지 유지되는 유일한 인용 가능 셀은 **Zamba2 r3**. "rep 부호 5/5"도 정정 — **Granite rate2는 실제로 3/5**(per-rep diff −0.0026/**+0.2789**/+0.0321/−0.0035/+0.0164, 효과의 87%가 rep2 한 점), 나머지 7셀은 5/5 유지 확인. n=5 paired 정확 부호뒤집기 순열검정의 두측 p 하한 = **2/32=0.0625**이므로 이 프로젝트의 n=5 paired 셀은 분포무가정으로 p<0.05에 원리적으로 도달 불가(기존 "CI가 0 배제" 서술은 전부 모수 가정 의존이었다는 사실을 명시). ★**P1의 방향 자체는 살아남는다**: agnostic > fused(꼬리에서)는 Zamba2 r2·r3, Granite r4에서 어떤 방법으로도 유효하다 — 강등되는 것은 "전 셀"·"5/5"·"사다리 전 구간"·"Granite r3 수치"뿐, 과잉 강등 아님. **열린 불일치(반영 보류)**: "Granite rate2는 경계(48ms로 내리면 소멸)" 문장은 방향이 반대라는 지적(임계를 내리면 오히려 커짐, T=22→+30.35%)이 있으나 "48ms" 수치의 출처가 `P1_OPINT_RESULT_2026-08-05.md`·`PREREG.md` 어디에도 없어 수치는 유지하고 "출처 미확인·방향 불일치 지적 있음(2026-08-06), 확인 전 인용 주의" 표시만 추가한다(단 Granite rate2는 A-4 경로로 이미 사실상 무신호로 반영됨). 신규 방법론 게이트 **#27**(n≤8 반복에서 `paired_bootstrap_ci`/`unpaired_bootstrap_ci` 구간을 판정에 쓰지 않는다, primary=t-CI) 등재, E1(`s8_frontier/e1_analyze.py:492,1286`) 사전등록 결정 규칙도 같은 undercoverage(net-positive 방향 편향, n=4 coverage 0.798)를 상속하므로 `PROJECT_STATUS.md` "다음 실험 gate" #8에 **제출 선행조건**으로 등재. `CLAIM_EVIDENCE_MATRIX.md`는 이 수치를 인용하는 서술이 없어 갱신 대상 없음(대조 확인 완료). 상세 `PROJECT_STATUS.md` "확정된 결과" 1번·"방법론 게이트" #14·§3 항목27(이 파일), 원자료 `workspace/engine-port/results/p1_gates/verify/`(스크립트·JSON·로그 전체).★★★★★**Gate 1(job 874478, 2026-08-06) 조건부 채택 — claims-auditor 감사, 새 성능 판정 0건, 진단 전용.** Zamba2-2.7B·agnostic v1·cudagraph-ON·rate{2,3}·n=1의 873944 텔레메트리 재현런에서, 엔진이 실제로 구성한 분할표는 `[(108,0),(74,34),(54,54),(0,108)]`이었고(`gate1_srv_874478.log:32`), **decode-busy ∧ prefill-in-flight 구간의 selector 라벨은 시간가중 100.00%가 `(74,34)`**였다(pooled 51.9 s, 78 에피소드, 3,820 스냅샷, 반례 0/40,062). ⚠️ **이 통계는 판별력이 사실상 없다** — `event_loop_pdmux`에서 prefill 어드미션(`multiplexing_mixin.py:1004`)과 `adjust_stream_groups()`(`:1080`) 사이에 telemetry sync가 존재하지 않으므로, "pop A ∧ idx∉{1,2}"는 **관측 가능한 상태가 아니다**(방법론 게이트 #9 다섯 번째 재발, 아래 §3 항목28). 이 문장은 "코드가 그렇게 쓰여 있고 실행도 그대로 했다"이지 "측정으로 알아냈다"가 아니다. ~~**★실질 산출 1 — 이 격자에서 정책은 단일 분할에 고정됐다**: 전 런 `decode_running_batch_size` 최댓값 **23**(문턱 `decode_bs_divisor=36`, `pdmux_a100_smoke.yml:6`) ⇒ **`(54,54)`는 0회 선택**. 인용 가능한 파티션 수치는 **`(74,34)` 하나뿐**이다.~~ → ★★★★★**철회(2026-08-07, G1-b, job 875293, 사전등록 철회 규칙 발화, 새 성능 판정 아님 — 사실 정정)**: 위 무제한(격자 전체) 문장은 거짓이다. rate **2**(max decode_bs 9)·**3**(23)·**4**(18)는 pop A 시간가중 100% `(74,34)`로 문턱 36 미만이라 **인용 가능 셀(Zamba2 r2·r3)에서는 "단일 분할"이 여전히 참**이지만, **rate 6에서만** `(54,54)`가 시간가중 **8.37%**(2.089s/24.974s, max decode_bs 40) 등장해 사전등록 철회 규칙(`frac((54,54))≥0.01`, `gate1b_analyze.py`에 하드코딩)이 발화했다 — 거짓이 된 것은 **격자 전체에 대한 무제한 주장**뿐이다(과잉 철회 금지). ⚠️**rate 4(18)가 rate 3(23)보다 낮다 — 비단조·원인 미해명**(단일 부팅 순차 실행의 순서 효과 또는 큐잉 동역학 가능성, §3 항목30). ⚠️**rate 6 경계 근접**: pop A 에피소드 5개뿐, max decode_bs 40이 문턱 36을 11%만 초과 — 정본은 Zamba2 r4·r6을 이미 "⚠용량 초과" 셀로 이미 표시 중이다. Population C 절대 수치는 Gate 1 감사 caveat 승계(인용 금지, 개수·에피소드만). `(74,34)`·`(54,54)`는 여전히 selector 라벨(하드웨어 SM 수 아님). G1-b 매니페스트는 874478과 바이트 동일 트리가 아님(HOLB 프로브 +2줄, 0줄 변경, 동등성 근거=G3). 상세 위 rev15 헤더, 원자료 `PREREG_G1B_2026-08-07.md`·`gate1b_result_875293.txt`(수정 금지·인용만). **★실질 산출 2 — duty cycle(창 시간 기준)**: prefill/decode 동시 in-flight **27–38%**(추정량 경계 오염으로 구간 제시), decode 단독 `(0,108)` 32.2%, prefill 단독 `(108,0)` 3.5%, **완전 idle 26.3%**. **필수 동반 caveat**: (a) **selector-level·behavioural**이며 하드웨어 SM 부여 프로브(S3) **미실행** — `(74,34)`는 요청값이지 실현된 하드웨어 SM 수가 아니다(Stage 0 D108 전례); (b) **Zamba2 rate{2,3} 한정** — 873944의 `MAIN_RATES`는 {2,3,4,6}이고 rate 4·6은 미측정이며 그 셀은 decode batch가 커져 `(54,54)`로 넘어갈 수 있다; (c) **Granite-4.0-h-micro-base는 전혀 미측정**; (d) n=1, 노드 gpu38(873944는 gpu41); (e) 다른 rate/워크로드/모델로의 이전 금지(게이트 #11). **⇒ §1-1의 "실현 파티션 미측정" 문구는 완전 해제가 아니라 부분·조건부 해제로 대체한다**: decode-busy∧prefill-in-flight 상태의 selector 라벨은 Zamba2 rate{2,3}·agnostic v1·cudagraph-ON에서 `(74,34)`(prefill 74 SM/decode 34 SM 요청값) 하나로 인용 가능하나 여전히 selector-level(S3 미실행)이고, **rate 4·6 셀과 Granite 전체는 여전히 미측정** — **특히 Granite rate4 +27.0%(위 인용 가능 정량치)에는 이 파티션·동시성 문장을 붙이지 않는다**(Gate 1이 그 셀을 커버하지 않음). **"PD 분리 자체" 기전 귀속은 Gate 1로 해금되지 않는다 — 여전히 NOT-YET-SUPPORTED(Gate 2 소관, 불변)**. **인용 금지(신규, 감사자 열거)**: "실현 파티션을 **측정**했다"(항등식) / "prefill 74 SM·decode 34 SM에서 **실행**됐다"(S3 미실행, green-context 반올림 미확인) / rate 4·6·Granite에 대한 파티션 문장 / "**동시성** 기전" 일반(fused arm 대응 계측이 원리적으로 부재 — `plain`은 `--enable-pdmux`가 없어 파티션 텔레메트리가 없다) / "음성대조 C의 2–3% 누출이 gate가 항등식이 아님을 보인다"(누출 10/10이 prefill-완료 전이 sync의 결정적 lag) / "**C = 0.9708/0.9779**" 수치 자체(10개 스냅샷 위, dt 4–10× 과대) / "케이던스 8↔32에서 불변"(추정량 산술 항등식: 누출 개수 ÷4 × dt ×4) / "874465 실패 원인 = bounded queue 포화"(**미확증** — grid 결측 0 + 급정지는 오히려 writer-thread 종료를 시사) / "pooled A = 51.9 s 겹침"(상한, 하한 36.7 s) / "런의 30%가 full-prefill 파티션"(그중 88%가 순수 idle). **커버리지 가드 논거 정정(부기 2, 좁은 형태만 인용)**: `PREREG_GATE1_2026-08-06.md`의 "coverage guard는 단방향으로만 더 엄격해진다"는 감사 결과 **거짓**이다 — `gate1_analyze.py:181-190`의 `severity()` 기준으로 `frac<0.90`이면 `NO_UNLOCK`이 됐을 창이 `coverage<0.98`이면 `UNMEASURABLE`로 **승격(완화)**될 수 있다. 살아남는 것은 좁은 주장뿐: **coverage 실패가 UNLOCK을 만드는 경로는 없다.** 임계 0.98도 근거 없음(관측된 두 점이 80.11%와 ~100%라 어느 값이든 결과 동일) — 이번 판정엔 무작동(`MIN_COVERAGE=0` 재실행 시 출력 동일). **이 논거는 정본 일반 원칙으로 승격하지 않는다**(이 job의 부기 텍스트에 대한 국소 정정일 뿐). **다음 gate 갱신** — Gate 1 완료(조건부 채택) → **G1-a**(≈0.2 GPU-hr, engine-porter): `multiplexing_mixin.py:1005`/`:1080` 사이 관측 전용 sync 1회 추가 ⇒ "prefill in-flight ∧ stale idx"를 관측 가능하게 해 주 조건에 판별력을 부여, 결정량=어드미션-후/adjust-전 구간의 시간 비율+절대 ms → **G1-b**: ✅**완료(2026-08-07, job 875293) — 철회 규칙 발화**(rate 6에서 `(54,54)` 시간가중 8.37% 관측, rate 4는 NO_EVIDENCE), 상세는 위 rev15 헤더·본 항목 "★실질 산출 1" 철회 블록 참조 → **G1-c**: Granite(873945 복제) → **G1-d**: S3 하드웨어 프로브(`%smid` 샘플링/CUPTI) → **하네스**: `gate1_analyze.py`에 grid-completeness 검정 상시화(`trace_forced==False ⟹ si==1 ∨ si%TRACE_EVERY==0`, 결측 수 출력, 이번 "유실 없음"의 실제 근거는 coverage가 아니라 이 검정이었음) + engine-porter 이관(`telemetry.py`의 `writer_error` 로깅, SIGKILL 경로에서 미호출되는 `close()`) → (기존, 불변) Gate 2 4-arm 분해 → Gate 3 나머지 2모델 → Gate 4 sustainable-rate. **위생 확인(전부 통과)**: 매니페스트 13파일 SHA-256이 873944와 바이트 일치, 2026-08-05 이후 변경된 `.py`가 정확히 그 13개(커버 밖 드리프트 없음 — ★2026-08-14 스코프 주석[§3 항목48]: 이 "13파일"·"커버"는 그 시점 `sync_engine_tree.sh`가 **해시하는** 파일 집합만을 뜻한다. `pdmux_context.py`·`sgl_kernel/spatial.py`처럼 sync가 애초에 관할하지 않는 런타임 파일은 이 확인의 범위 밖이며, 이 위생 확인은 그런 파일의 드리프트 여부에 대해 아무것도 말하지 않는다 — 판정 자체를 뒤집지 않음, 주장 범위만 명시), `architecture` 필드 40,062/40,062="legacy", `stream_index↔sms` 불일치 0, cudagraph ON, `CORRECTNESS=PASS`. ⚠️**노드 불일치**: Gate 1=gpu38, 873944=gpu41. 상세 `PROJECT_STATUS.md` "다음 실험 gate" #10(Gate 1 항목), 원자료 `workspace/engine-port/results/p1_gates/gate1/`(`gate1_result_874478.txt`·`PREREG_GATE1_2026-08-06.md`·`gate1_analyze.py`·`gate1_telemetry_874478.jsonl`·` ★★★★★**(2026-08-07 실행, 2026-08-09 정본 반영 복구, doc-steward) Gate 2 rev4 본 캠페인(jobs 875344/875346, 5.86 GPU-hr) — 정본 누락 복구.** 새 성능 판정 아님 — 이미 완료된 결과의 정본 반영 누락을 메운다. 원자료 `workspace/engine-port/results/p1_gates/gate2/g2_report_zamba2-27b_875344.json`·`g2_report_granite-40-h-micro-base_875346.json`(rev4 사전등록 `PREREG_GATE2_2026-08-06.md` §14 적용, arm A1=`plain`/A2=`plainaux`/A3=`chunk512`/A4=`agnostic`). **Primary(§4.1/5.2, A3 vs A4 TOST)**: 5셀(Zamba2 r2·r3, Granite r3·r4·r6) 전부 `Rprime4`(A4 유의 우세, 등가 미발화, `p_tost=1.0`) — **chunk512는 pdmux를 대체하지 못한다.** F-E(정상상태) 스크린 통과로 크기 인용 가능한 셀은 **Zamba2 r2·Granite r3 둘뿐**(나머지 3셀은 chunk512측 F-E 발화, 부호만). **Secondary — R1′/R2′(§5.2 표 마지막 행, A2=`plainaux` vs A4=`agnostic`)**: rev4 §1.1의 등식 A2=A4−pdmux(realized server_args 차이는 `enable_pdmux`·`pdmux_config_path` 두 필드뿐)이므로 이 비교는 §1-1의 "3-플래그 묶음 처치" 교락 중 pdmux 고유분을 분리한다. 메인 세션이 2026-08-09 원자료(`per_arm_x60`)에서 독립 재현(paired-t, n=10; 셀: mean(A2−A4, X_60) [95% CI], 부호, F-E(plainaux/agnostic)) — Zamba2 r2: **+0.8341** [+0.7809,+0.8872], 10/10, clear; Zamba2 r3: **+0.9297** [+0.8968,+0.9626], 10/10, clear; Granite r3: **+0.8550** [+0.8161,+0.8939], 10/10, clear; Granite r4: **+0.9442** [+0.9201,+0.9682], 10/10, clear; Granite r6: **+0.9600** [+0.9292,+0.9908], 10/10, **agnostic 측 F-E flagged — 부호만**. sign-flip 순열 p = 2/1024 = **0.001953125**(n=10 분포무가정 두측 하한, 5셀 공통). **묶음 기여 분해**: A2는 A1(`plain`)과 사실상 같고(5셀 |A1−A2|≤0.023) 그 부호는 **A4에 불리한 핸디캡 방향**이다 ⇒ `--chunked-prefill-size -1`·`--disable-overlap-schedule` 두 플래그만으로는 pdmux 이득이 재현되지 않는다 — §1-1 "3-플래그 묶음 처치" 교락 중 이 두 플래그는 **원인에서 배제**된다. ★★**provenance(인용 시 필수 동반)**: 이 A2-vs-A4 비교는 **사전등록 분석기 `g2_analyze.py`가 계산하지 않는다** — `tost_equivalence` 호출은 코드 전체에서 1회뿐(`:543`)이고 그 입력은 A3-vs-A4(`chunk512`-`agnostic`, `:538-539`)뿐이다. A2는 `:734`에 서술 문장으로만 등장한다("A1/A2/A4 data … remain valid regardless"). 즉 **arm·n·raw 데이터는 사전등록(rev4)이지만, 이 비교 자체는 저장된 primary 산출물(`per_arm_x60`) 위에서의 사후 계산**이다(코드 확인 2026-08-09). §3 항목34로 등재(항목33 "사후 지정 셀 이동"의 형제 사례). ★★**천장 포화(해석 제한)**: fused arm(plain/plainaux/chunk512) X_60 평균이 0.72–0.99(천장 근접), agnostic이 0.00–0.12(바닥 근접)이다 ⇒ **부호는 견고하나 크기는 포화 구간의 값**이며, E-A 블록이 이미 건 "기전 해석 금지"가 여기도 적용된다. ★★**해금 범위의 상한(overclaim 금지)**: 해소되는 것은 §1-1의 **"3-플래그 묶음 처치" 교락뿐**이다. 귀속의 상한은 **"pdmux 서브시스템 전체"**(green-context SM 분할 + 전용 이벤트 루프 + split-prefill)이고, **"SM 분할 자체"는 여전히 미분리**다 — `--enable-pdmux`가 A2의 `event_loop_normal()`을 대체해 pdmux 전용 이벤트 루프를 통째로 켜기 때문이다(§1.1). **"PD 분리 자체가 원인"으로 승격 금지** — §1-1의 NOT-YET-SUPPORTED 등급은 불변, Gate 2 본 질문은 한 눈금도 전진하지 않는다. ★**Granite r6**: agnostic 측 F-E flagged=true라 부호만(10/10, +), 크기 인용 금지. ★**감사 provenance(등급 정정)**: 세션 핸드오프(`handoff-report/session_handoff_2026-08-09.md` §2.5)는 이 R1′/R2′ 결과를 "[감사 완료]"로 표기했으나, 2026-08-09 저장소 전체 검색(`gate2/` 디렉터리·전체 `*.md`)으로 이 결과를 다루는 claims-auditor 감사 아티팩트가 확인되지 않는다. ⇒ **"감사 완료"로 인용하지 않는다.** 현재 등급 = **메인 세션 원자료 독립 재현 확인(2026-08-09)**, claims-auditor 감사 기록 위치 미확인.(같은 캠페인의 primary R3′/R4′와 §11-3 감사 면제 사실은 §14.1 addendum에 이미 기록돼 있고 그 부분은 인용 가능 — 감사 미확인은 이 R1′/R2′ 비교 자체에 한정된다.) 상세는 위 rev17 헤더, `PROJECT_STATUS.md` "확정된 결과" 1번·"방법론 게이트" #20·`CLAIM_EVIDENCE_MATRIX.md`(대조 확인, 갱신 대상 없음) 동반 갱신 참조. ★★★★★★**E-A(mixed-chunk 레버, jobs 875654/875657/875661, 2026-08-08~09) — claims-auditor 감사 완료, 판정 조건부(문구 강한 제한). 새 성능 판정 아님 — fused-측 조율 가능성에 대한 진단이다.** 원자료 `workspace/engine-port/results/p1_gates/gate2/`(`PREREG_G2EA_2026-08-07.md`·`g2ea_report_*.json`·`g2earun_8756{57,61}.out`·`g2eaprobe_875654*`, 수정 금지·인용만). **A. 레버 실현**: Stage 1(job 875654)에서 `--enable-mixed-chunk`가 Zamba2-2.7B·Granite-4.0-h-micro-base × {8192, 512} 4콤보 전부에서 realized로 확인됐다(세 독립 증인: `/server_info` top, `internal_states[0]`, HOLB `holb_open`). 기능 스모크 8/8 정상. ⚠️프로브 스크립트의 `PROBE_RESULT` 필드는 stderr 오염 버그(`g2ea_mixed_chunk_probe.sbatch:205`의 `2>&1`)로 `AVAILABLE_BUT_BURST_ERROR`를 잘못 출력했다 — 판정은 원 burst JSON 8건 직접 재파싱에 근거한다. **B. 주 결과(부호)**: E-A(jobs 875657/875661, n=10 paired, cudagraph-ON, arm 순서 무작위, 노드 gpu42/gpu40)에서 사전등록 primary 두 비교 × 4셀 = **8건 전부 TOST 등가가 발화하지 않았고, 방향은 전부 agnostic 우세**(부호 10/10, 분포무가정 정확 p = 2/1024 = 0.00195). ⇒ 사전등록 규칙상 **"조율된 fused가 pdmux를 대체한다"는 발화하지 않았고, 논문 신규성 축 전환은 없다.** ⚠️ 8건 중 7건은 F-E 발화(비정상상태) ⇒ **부호만 인용 가능.** ⚠️ cmp2(A3m)의 Granite 두 셀은 G-2 실패로 상속된 §2.1 폐기 규칙 하 **무효**다. **C. 유일한 인용 가능 정량치**: Granite-4.0-h-micro-base rate 3, cmp1: **X_60(plainmix) − X_60(agnostic) = +0.855 [95% CI +0.814, +0.896]**, n=10 paired, 부호 10/10. 양 arm 정상상태(duration 41.0s vs ideal 40.0, 실현 throughput 2.938 vs 제공 3, F-E 1.026/0.973, 런길이 불변 검정 N=60→120에서 TTFT p95 338→406 ms = 1.20×). ⚠️ **이 셀이 지정 셀이 된 것은 선행 캠페인 진단을 본 뒤의 사후 이동(r4→r3)이며, r4를 유지했다면 이 캠페인의 인용 가능 셀은 0개였다** (`PREREG_G2EA_2026-08-07.md` §3 자기인지 위험 #2). ⚠️ **같은 셀에서 TTFT 항은 통과한다** — plainmix가 agnostic보다 TTFT p95가 **19.2% 좋다**(CI [−0.256, −0.129]). 탈락은 ITL 항에서만 일어난다. **"TTFT 비열등 0건"은 거짓**이다(1/8). **D. 레버 격리**(C와 반드시 병기 — 없으면 C가 오해를 부른다): 위 +0.855 중 mixed-chunk 기여분은 **+0.010 [+0.0045, +0.0155] = 1.2%**뿐이다. 나머지 98.8%는 untuned fused와 pdmux 사이의 기존 격차(X_60(plain) − X_60(agnostic) = +0.845 [+0.802, +0.888])이며 rev4/§1-1이 이미 확립한 양이다. 4셀 전체에서 mixed-chunk 기여분은 **1.2–10.6%**. ⇒ **E-A의 primary는 mixed-chunk 레버를 격리하지 않는다.** ⚠️ X_60은 fused arm 전부 0.85–0.99로 천장 근접(정본 §1-13 ceiling-censoring 재발) ⇒ mixed-chunk 고유효과 크기는 **하향편향**. ⚠️ 그 크기(+0.010)는 HOLB 관측자 효과 잔차(rev4 §14.3: itl_p95 arm별 −9.9%~+105%)와 **같은 자릿수**이므로 **기전 해석 금지.** **E. 기전(신규, 서빙 직접 측정)**: mixed-chunk가 켜지면 prefill과 함께 스케줄된 decode가 `ForwardMode.MIXED` extend 경로로 재라우팅된다(`schedule_batch.py:1874`; `MIXED`는 `is_cuda_graph()`에서 제외 — `forward_batch_info.py:166-173`). HOLB 실측: MIXED step 지속시간이 병합 decode 요청 수에 **선형 증가** — Zamba2 252→1,844 ms(bs 0→48, ≈33 ms/요청), Granite 129→247 ms(bs 0→24, ≈4.2 ms/요청). 순수 DECODE step은 각각 9.4 / 6.8 ms. MIXED step 하나가 running 요청 전부를 1 토큰씩만 전진시키므로 요청별 ITL이 MIXED step time과 같아지고, backlog가 자라 `max_running_requests=48`·`mamba usage 1.00`을 포화시켜 admission을 막고 TTFT가 발산한다. ⇒ **정본 §1이 동적 제어에 대해 확립한 얽힘 死因(decode 굶김→ITL↑→batch 정체→admission 차단→TTFT 폭발)의 새 트리거이지 새 기전이 아니다.** **F. cudagraph 가설 반증 + 정본 부수 정정**: "mixed 배치가 decode CUDA graph를 못 써서 decode가 eager로 떨어진다"는 **설명으로서 반증**: (i) plainmix의 남은 순수 DECODE step은 `cuda graph: True`이고 중앙 지속시간이 plain과 동일(9.5 vs 9.4 ms Zamba2, 6.9 vs 6.8 Granite); (ii) 페널티가 배치 크기 비례라 그래프 launch 오버헤드보다 두 자릿수 큼; (iii) piecewise CUDA graph는 두 모델 10개 boot 전부 런타임 비활성이라 arm 간 비대칭 아님. ⇒ 실제로 일어난 것은 "decode의 eager 강등"이 아니라 **"decode 작업의 extend 커널 경로 재라우팅"**이다. ★ **부수 정정**: rev4 §1.1의 "A3는 `piecewise_cuda_graph_max_tokens` 8192→512도 함께 바꾸는 ≥2-기전 묶음" 캐비어트는 **이 두 모델에선 런타임 실측으로 무효**다(전 arm piecewise OFF). 해당 인용 금지를 해제하되 근거를 명시하라. **G. G-2(재현성 — correctness 아님)**: `--chunked-prefill-size 512`를 건 Granite-4.0-h-micro-base(flashinfer, `enable_deterministic_inference=False`)는 16-동시 greedy 요청에서 **비트단위 재현성이 깨진다** — 16 중 2 요청이 퇴화 반복 루프의 반복 단위 한 토큰(62↔322)에서 결정론적으로 갈라진다(불일치 위치 = 정확히 주기 3의 31개 지점, 출력 길이 96 동일, 두 변종은 arm 간 바이트 동일). 3회 독립 재현(875346·875611·875661), 음성대조 `plain`·`plainaux`·`plainmix`·`agnostic` 전부 통과. **이것은 correctness 결함이 아니라 reproducibility 결함이며, 엔진은 배치 형태 간 비트 재현성을 약속하지 않는다.** mamba state 이월과는 **정합하지 않는다**(그 경우 인덱스 0부터 고정 불일치 + 전혀 다른 연속이 예상되나, 관측은 반복 구조 완전 보존). **H. 스코프(축약 금지)**: {Zamba2-2.7B(triton, ctx4096) · Granite-4.0-h-micro-base(flashinfer, ctx8192), A100 108-SM, **cudagraph-ON / piecewise-OFF**, `--disable-radix-cache --mem-fraction-static 0.82 --max-running-requests 48`, `random-ids` in2000/out96, 정상상태 단일-rate {Zamba2 2,3 / Granite 3,4}, 120 프롬프트, n=10 paired(동일 job·동일 노드), agnostic v1, HOLB 프로브 전 arm ON}. **상위 사전등록 rev4의 §11-3(3차 감사)은 면제됐다 — "감사를 통과한 설계"가 아니다**(§14.1 자백). **T2(가장 무거움) — "fused 조율 공간 소진" 주장 금지**: 정본 `CONSENSUS.md:368`의 *"…가 정확히 그 축의 fused-측 레버인데 둘 다 기본값이다"*는 **대표 레버 지목이지 완전성 주장이 아니다.** 같은 기전 축의 **미시험 노브 최소 6개**(realized 값 실측): `--prefill-max-requests`(현재 `None`=무제한, `server_args.py:3959`) · `--num-continuous-decode-steps`(현재 1, `:5559`) · `--chunked-prefill-size` 2048/4096 · `--max-running-requests`(현재 48 = 실제 포화점) · `--schedule-conservativeness`(1.0, `:4029`) · `--mamba-scheduler-strategy`(`no_buffer`, `:5082`). (+`--enable-prefill-delayer`는 적용성 미확인.) ⇒ 쓸 수 있는 최대치: *"정본이 지목한 두 레버는 시험됐고 둘 다 격차를 닫지 못했다 — 다만 fused 측 조율 공간이 소진됐다는 뜻은 아니다."* **Gate 2 본 질문 전진 없음**: A2(`plainaux`)를 뺐으므로 **"PD 분리(SM 분할) 자체" 귀속은 한 눈금도 전진하지 않았고 §1-1의 NOT-YET-SUPPORTED는 불변**이다. `PREREG_G2EA_2026-08-07.md:66-67`이 A2와의 교차-job 비교를 명시 금지했으므로 875344/875346의 A2로 메우는 것도 금지. **인용 금지 목록(13건, 정본에 그대로 등재)**: 1. "등부하에서 mixed-chunk는 지연을 N배 악화시킨다"(78×·6.2×·24–35× **전부**) — 런길이 의존 실측(N 60→120에서 2.3–6.2× 변동). 2. throughput `1.938→1.303`·`0.530`·`1.070`, TTFT p95 `28,359`·`154,601`·`67,298 ms` — 무플래그 `secondary` 블록 출신, 불안정 큐 과도상태. 3. F-E 발화 셀의 X_60 크기(cmp1 Zamba2 r2·r3, Granite r4; cmp2 전 셀). 4. **"TTFT 비열등 0건"** — 거짓(1/8, C 참조). 5. "fused 조율 레버 소진" / "T1이 닫혔다" / "조율된 fused는 pdmux를 대체할 수 없다"(일반형). 6. "두 레버 모두 실패"를 **E-A 캠페인 내부 결과로** 제시 — A3-vs-A4는 E-A 사전등록 비교가 아니다(사후 계산이며 F-E clear ∧ G-2 PASS를 동시 만족하는 셀은 **Zamba2 r2 하나**: +0.879 [+0.859, +0.899]). 7. "chunked-prefill 조율이 …" 문구 — 단 F의 정정 반영 후에는 금지 근거가 약해지므로 F를 먼저 반영할 것. 8. "chunked_prefill_size=512가 Granite에서 **잘못된 출력**을 낸다" — reproducibility ≠ correctness. 9. cmp2 Granite r3·r4의 어떤 판정도 — 상속 §2.1 폐기 규칙 미적용. 10. "mixed-chunk 붕괴는 Zamba2라는 모델의 성질" — 모델과 attention backend 동시 변경(confound #10). ⚠️ 선례 `[[scale-8b-sm-sensitivity]]`의 "hybrid 급락=Zamba2 성질" 강등과 동형. 11. "PD 분리 자체" / "pdmux 필요성 확립" / "Gate 2 전진". 12. "감사를 통과한 사전등록". 13. MIXED 한계비용 8× 비대칭(33 vs 4.2 ms)의 **원인 귀속** — 미해명으로 등재. **신규 방법론 항목(§3 항목31–33 등재)**: (31) "게이트를 지표에 걸 때는 primary뿐 아니라 보고되는 모든 블록에 걸어라." F-E 집행 수정이 primary 두 비교에만 적용되고 `secondary` 블록(throughput/goodput/ttft_p95/itl_p95)은 무방비였다 — 그리고 실제로 **그 무방비 경로에서 인용이 일어났다**(`g2ea_analyze.py:662-680`). (32) "과부하 arm과 정상 arm을 같은 제공 rate에서 비교한 수치는 시스템 상수가 아니다." 런길이 의존 실측: 프롬프트 60→120에서 지연 2.3–6.2× 변동, 정상 셀만 1.20×. ⇒ 비정상상태 셀은 부호만, 크기는 지속가능 rate 대조(E-C) 후에만. (33) "사후 지정 셀 이동은 부호를 안 바꿔도 인용 가능성을 만들 수 있다." r4→r3 이동이 없었다면 이 캠페인의 인용 가능 셀은 0개였다. **후속 게이트 등재(우선순위순, `PROJECT_STATUS.md` "다음 실험 gate")**: T4-1 deterministic-inference 재확인(<0.5 GPU-hr, 폐기 규칙 해제) → T3-1/T3-2 MIXED 배치 조성 계측·micro 스윕(<1 GPU-hr each, 기전 확정) → E-C 등지속가능-rate 대조(≈4 GPU-hr, T1 정식 종결) → E-D 미시험 fused 레버 스윕(≈8 GPU-hr, T2 종결) → T3-3 backend 교차(≈2, 8× 비대칭 귀속) → T4-2 비퇴화 프롬프트 G-2(≈1). ⚠️**상위 사전등록 `PREREG_GATE2_2026-08-06.md`의 §14 addendum은 커밋 `c47fad0`으로 반영돼 워킹트리 클린이다(2026-08-09 확인) — "워킹트리 미커밋" 기록은 stale, 정정.** §11-3(3차) 감사 면제 상태(§14.1)는 별개로 불변이라 "감사 통과 설계"로는 여전히 인용 금지. `results/p1_gates/` 이하 raw dump 수정 금지(인용만). 상세 위 rev17 헤더·rev16 헤더, `PROJECT_STATUS.md` "확정된 결과" 1번·"방법론 게이트" #17–20·"다음 실험 gate" #10(E-A 항목), 원자료 위 경로. ★★★**(2026-08-09, 두 번째 정본 반영 건 — A/B/C 세 갈래, 전부 새 성능 판정 아님)** **A. HOLB G5 재채점**(result-analyst, jobs 874601/874602/874632/874633/874635, 원자료 `workspace/engine-port/results/p1_gates/gate2/g2holb_g5_tost_rescore_2026-08-09.json` 72셀 전량, 3 job × 4 arm × 6 응답변수, n=5 paired): **판정 G5 = 미결정(UNDETERMINED), 저장된 `G5=False`를 대체.** 3% 초과 통계적 지지 셀 **0/72**(미보정 최소 p_exceed=0.138, job별 Holm 후 최소 조정 p=1.000 3 job 전부), 등가 입증 **29/72**, 검정력 부족 **43/72** — G5는 관측자 효과가 3%를 넘음을 입증하지 못했고 3% 이내임도 입증하지 못했다. 저장 `G5=False`는 귀무-채택형 연언(`|효과|<3% ∧ CI∋0`)의 실패이지 프로브 유해성의 입증이 아니다. **구 규칙이 노이즈를 보상했다**(874635 agnostic ttft_p95 −0.77%±29.14%, 95% CI [−36.95,+35.41] → 구 규칙 PASS, 메인 세션 독립 재확인) — 역방향(구 규칙 FAIL·±3% 등가 실제 입증)도 **4셀**(예: 874633 plain itl_p95 −1.15%±0.72%, p_TOST=0.0023). **설계 층**: n=5·δ=3%·α=0.05/side에서 TOST 발화 산술 천장 SD<3.147%인데 72셀 중 **35셀(49%)**이 그 위, 80% 검정력 필요 n 중앙값 **9**(request_throughput 4 / itl_p50·mean_e2e_ms 7 / itl_p95 22 / ttft_p50 29 / **ttft_p95 89**, ⚠️SD가 df=4 추정이라 필요 n은 자릿수 수준 의미만). `PREREG_GATE2` §14.2의 "프로브가 무해함이 입증됐다고 쓰지 마라"는 **해제되지 않는다** — 동시에 반대 오독("3% 넘게 유해함이 입증됐다")도 근거 없음. §14.3의 귀인("주로 점추정이 3%를 넘는 조합이 실재하기 때문")은 **부분적으로만 참**(그런 셀은 실재하나 그중 하나도 초과가 지지되지 않는다) — 두 문장 병기 필수. **잔여 교락**: 874602 agnostic은 5/5 rep 전부 `order='off on'`(무작위화 불균형) ⇒ 그 arm의 등가 판정은 조건부. 등가 판정 29건은 전부 paired-t **정규 가정** 위(n=5 분포무가정 두측 p 하한 2/32=0.0625). **B. job 874601의 `G3=False` 라벨 정정**(메인 세션 원자료 직접 확인): `g2holb_report_zamba2_874601.json`은 4 arm 전부 `result:"FAIL"`이나 `sha_off:null, sha_on:null`·`self_repro_off/on:true` — 출력이 갈라진 게 아니라 sha 추출이 응답 스키마를 못 읽은 것(`KeyError:'text'`)이고 Phase B가 실행되지 않았다. 재실행 874633/874635는 `method_off/on:"output_ids"`로 sha 양측 일치 → **PASS**(위 A절 근거로 이미 사용됨). 874632는 Phase A 중 SLURM CANCELLED(데이터 없음). ⇒ "874601 G3 실패" 라벨은 **하네스 실패**로 정정. **이 프로젝트 서명 오류(측정 실패를 게이트 실패로 라벨링)의 여섯 번째 재발**(핸드오프 2026-08-09 §1이 다섯 번 — 텔레메트리 드롭 카운터 자기검열/UNSCOREABLE을 강등으로 읽음/HTTP 400을 G3 FAIL로/G5의 귀무-채택형 기준(위 A절과 같은 사건 계열)/프로브의 stderr 오염). **canon에 이 패턴을 다루는 기존 번호가 없어(전수 검색 확인) §3 항목35로 신규 등재**(중복 신설 아님 — 이후 재발은 이 항목 카운트만 갱신). **C. E-A 커밋 산출물의 scipy 부재 폴백**(engine-porter 발견 + 메인 세션 노출범위 실측, 범위 한정): `g2ea_report_*.json`은 scipy 없는 인터프리터에서 생성돼 `t_cdf()`가 Student-t가 아니라 정규 CDF로 조용히 폴백(저장 `p_tost`가 정확히 1.0인 이유, `g2ea_analyze.py:84-87,156-159`). **메인 세션이 노출 범위 직접 측정**: `raw_ci` 폭 역산 임계값이 **8개 비교 전부**(2 job×2 rate×2 cmp, coordinator 인용 4개 포함, 전부 df=9) implied_t=**2.2621…=t(.975,df=9)**(하드코드 표값과 일치, 1.96 아님) — **CI(raw_ci)는 오염되지 않았다.** 노출은 (i) `p_tost` 자체(이 캠페인은 관측치가 0.05 경계에서 멀어 판정 무영향)와 (ii) `t_ppf`의 df>10·표에 없는 p에 한정(`t_cdf`는 표가 아예 없어 scipy 부재 시 모든 df에서 근사값이라는 점은 (i)에 포함되되 원인 층이 다름을 기록). ⇒ **"정본 수치 정정"이 아니라 도구 규율 항목**(게이트 #14 계열, §3 항목36 신설). **과장 금지 — 이 캠페인에서 오염된 인용 수치는 없다.** 상세 위 rev18 헤더·§3 항목35·36, `PROJECT_STATUS.md` "확정된 결과" 1번·"방법론 게이트" #21·#22 동반 갱신, 원자료 `workspace/engine-port/results/p1_gates/gate2/`(`g2holb_g5_tost_rescore_2026-08-09.json`·`g2holb_g5_tost_rescore.py`·`g2holb_g5_tost_summary.py`·`g2holb_report_zamba2_874601.json`, 수정 금지·인용만). ★★★**(2026-08-10, 세 번째 정본 반영 건 — A/B/C/D, 전부 새 성능 판정 아님)** **A. 방법론 게이트 #21의 일곱 번째 재발**(job 876699, T4-1): `--time=1:00:00` TIMEOUT, 원인은 사이징이 아니라 단일 호출 스톨 — `ON+chunk512` 첫 multi-chunk generate(2552 토큰, 5-chunk)가 ~52분 무응답(스케줄러 로그 0줄, CUDA 에러·watchdog 미발화; 서버 로그 마지막 활동 23:34:42, SLURM kill 00:26:19, 메인 세션이 두 시각 직접 대조), 같은 job `OFF+chunk512`는 동일 프롬프트 ~1초 완료. **핵심**: 옛 `g2det_analyze.py`가 그 아티팩트에 `REFUTED`를 반환하고 있었다 — ON chunk512 `n_total=0`(미실행)인데 `on_clean = n_total > 0 and ...`이 False로 떨어져 REFUTED로 통과, **52분 멈춤이 실질적 음성 결과로 발표될 뻔했다.** 재발 1–6과 다른 점: 1–6은 라벨·해석 오류를 사람이 잡았으나 이번은 **분석 코드가 거짓 음성을 산출**했고 막은 것은 experiment-runner의 채점 거부(`INCOMPLETE`)였다 — 도구가 사람보다 관대했다. 사후 완화 아님(사전등록 REFUTED 조건은 "ON이 어느 rep에서든 self-mismatch≥1"인데 rep 0개는 그 관측 자체가 없다 — 옛 코드는 버그였다). 수정(2026-08-10) 후 `NO VERDICT (MEASUREMENT ABSENT)` 반환, CONFIRMED/REFUTED 분기는 5-케이스 매트릭스로 불변. ⇒ **§3 항목35(가트#21)의 재발 카운트 6→7 갱신**(아래 개정판, 신규 항목 아님). **B. 공유 하네스의 무한 대기(도구 규율, 신규 §3 항목37/게이트#23)**: `g2_holb_phaseA_lib.sh`의 `greedy_call`이 `--max-time` 없는 raw curl이었고 이 디렉터리의 모든 캠페인(g2ctrl/g2ea/g2holb/g2det)이 이 경로를 쓴다 — 소비자 10개 전수 확인 결과 4개가 타임아웃을 `FAIL`/`SMOKE_FAIL` 계열로 채점 중이었다(A의 `g2det_analyze.py` 포함). 수정: `--connect-timeout 10 --max-time 180`(env 재정의 가능), `STATUS=TIMEOUT`을 `STATUS=ERROR`와 구별, SHA 미방출로 mismatch 채점 구조적 불가, 사이드카 `.status.json`. 180s는 측정 근거(이 디렉터리 아카이브 응답 n=64의 서버측 `e2e_latency` median 0.978s/p90 2.534s/max 6.605s, 최악의 27배·중앙값의 184배 — 메인 세션이 코드 주석 근거 문단 직접 대조) — 정상 경로는 아카이브 64개+합성 실패 9종 재생 73/73 byte-identical 검증(engine-porter 보고, 메인 세션 미재현). A와 뿌리 사건은 같으나(job 876699) 레슨은 다르다: A="분석 코드가 무데이터에 REFUTED", B="공유 인프라가 무경계 대기를 10개 소비자에 전파". **C. Gate 2-S 3라운드 설계 감사 — §1-1 귀속 스코프 주석(대체 아님)**: `PREREG_GATE2S_2026-08-09.md`가 claims-auditor 3회 전부 NO-GO(rev1→2 통계층/rev2→3 게이트 인식론/rev3→4 귀무 채택형, 메인 세션 파일 직접 확인) — 3연속이 같은 자리(§5.5 앵커 발화 조건)에서 죽었고 3차 감사가 근본 원인을 문서 내부 모순으로 특정(§0.1 "두 pdmux arm 사이 등가 마진에는 외부 앵커가 없다" vs §5.5가 정확히 그 등가를 앵커 조건으로 요구). 정량(§0.0.B 원문 대조): rev3 §5.3 암묵 등가 마진 = **0.715 σ_D**, primary MDE = **0.995 σ_D** ⇒ 대리 허용오차가 검출한계의 **0.72배**; 표준 처방(TOST+마진)은 §0.1 위반, 마진을 MDE 1/4로 낮추려면 **n≈64**(현재 10). ⇒ **정본 문구(고정)**: **"P1 이득 중 'SM 분할 자체의 몫'을 두 pdmux arm 사이의 성능-층 등가검정으로 귀속하는 경로는, 이 프로젝트의 예산 범위에서 닫히지 않는다(n≈64 필요, 현행 설계 n=10). 원리적 불가능이 아니라 이 경로·이 예산에서의 불가능이다."** ⚠️**과장 금지**(귀속이 원리적으로 불가능하다로 쓰지 않는다) — **이 문장은 위 §1-1의 NOT-YET-SUPPORTED를 대체하지 않고 스코프 주석으로 덧붙는다**("아직 안 됐다"≠"이 경로로는 안 된다"), 등급어는 미실행 사전등록의 설계 감사에 맞춘다(측정 결과 아님). **감사가 깨뜨리려 시도했으나 실패한 것(견고, §0.0.A 직접 확인)**: arm 구조·Δ_split estimand 내부 타당성, T·C 드레인 대칭, 9-셀 판정표의 저자 불리 셀 실재, 부팅 단위 프로브 근거(n_eff=10), 예산 산술. 3차 감사 원문: **"이 캠페인이 죽어야 할 이유는 측정 층이 아니라 귀속 층에만 있다."** rev4는 §5.5 앵커 조건 자체를 제거해 이 하위목표에서 후퇴(패치 0줄, env 재구성만) — **Gate 2-S가 폐기된 것은 아니다.** **D. 코드 사실 문구 하향(메인 세션 자기정정)**: 이 세션 중 "`(0,108)` idx에서는 두 역할 모두 전체 108 SM에 접근한다"고 서술한 바 있다. 검증된 것은 "green context가 아니라 평범한 `torch.cuda.Stream` 쌍"까지뿐(`pdmux_context.py:124-138`, 메인 세션 직접 확인 — idx 0·idx `len-1`은 `torch.cuda.Stream(gpu_id)`, 중간 division만 `create_greenctx_stream_by_value`) — green ctx 생성이 primary context SM을 깎는지는 미측정 물리 명제(격리 수단 P-b 프로브는 공선성으로 삭제됨, engine-porter 보고). **전수 검색(2026-08-10) 결과 이 문구는 canon·파생 문서 어디에도 없다** — 정정 대상 없음, 향후 인용 규칙만 등재: (0,108) idx는 "명시적 분할이 적용되지 않는다(잔여 차감 여부는 미측정)"로만 쓴다. 상세 §3 항목35(개정)·37, `PROJECT_STATUS.md` "확정된 결과" 1번·"방법론 게이트" #21(개정)·#23, 원자료 `workspace/engine-port/results/p1_gates/gate2/`(`g2det_876699.out`·`g2det_876699.err`·`g2det_analyze.py`·`g2_holb_phaseA_lib.sh`·`PREREG_GATE2S_2026-08-09.md`, 수정 금지·인용만). ★★★★★★★**(2026-08-11) Gate 2-S 첫 유효 결과(jobs 877756/877757, 6.35 GPU-hr) — claims-auditor 적대 감사 "조건부 등재 가(可)". 별도 트랙 산출, §1-1 대체 아님, 위 C(2026-08-10 귀속 스코프 주석) 뒤에 배치.** 둘 다 exit 0/`MEASURED_AND_SCORED`, `design_conformance.conformant=True`(전 셀 n=10, 5 arm), 19/19 블록. 1차 실행(877107/877109, 6.40 GPU-hr)은 하네스 결함 6건으로 primary 0개 — 설계는 무결, 실패는 하네스 층뿐. 사전등록은 claims-auditor 설계 감사 4회(NO-GO 3 → GO-with-changes). **정본 문장(감사자 확정 범위, 그대로)**: 이 엔진·이 격자(Zamba2-2.7B r{2,3} triton / Granite-4.0-h-micro-base r{3,4} flashinfer, in2000/out96, A100 108 SM, cudagraph ON, n=10 paired)에서, pdmux 서브시스템·이벤트 루프·split-prefill·역할별 backend를 고정한 채 `PDMUX_R2_FIXED_DSM`만 108→34로 바꿔 SM을 `(74,34)`로 분할하면 **요청-내부 ITL p95 평균(α)이 4셀 전부 감소**(Δ=C−T′=**+13.95/+24.88/+16.04/+19.11 ms**, 4셀 Holm 후 최대 보정 p=1.88e-06, β=pooled p99도 부호·유의성 일치)하고 **같은 4셀에서 TTFT p95는 악화**(−183~−536 ms) — 9-셀 좌표는 16블록 전부 `S1-C`(트레이드오프). Zamba2 r2·r3는 "`FixedPolicy(34)`·sticky OFF=엔진 기본 궤적" 명명 허용(전제 검증, Gate 1 rev15/875293), **Granite r3·r4는 그 명명 금지**(전제 미검증, G1-c 미실행, in-job 반증기 `unchanged`는 검증 아님). **크기 인용 허용 셀=F-계열 미발화 Zamba2 r2 하나뿐**: α **−38.8%**[−41.6,−36.0]·TTFT p95 **+51.5%**[+35.8,+67.2](부호=T′ 대 C, 음수=분할 우수). ★**효과는 꼬리 한정**(q≤0.7 분위수는 4셀 전부 반대 부호로 유의, q0.5: −0.85/−2.58/−0.91/−1.68 ms; 기전="드문 대형 prefill 유발 decode 스톨을 균일한 소폭 decode 지연으로 교환"). ★**처치는 decode-active 시간의 35–40%만 실현**(T′ idx1 frac 0.355–0.405)되며 **100% 실현 arm(T, sticky ON)의 α 효과가 오히려 더 작아 어떤 dose 외삽도 금지**. "무분할(C)"=명시적 분할 부재일 뿐 잔여 차감 미측정(Zamba P-carve NOT-EQUIVALENT, 단 C·T′가 같은 yml·같은 green context로 부팅해 이 대비에서는 공통항 소거). **§1-1의 A4-vs-fused 격차의 성분·기여분·분해로 서술 금지, "PD 분리 자체가 원인" 승격 금지, SLO goodput·용량·fused 우열은 무언급**(같은 job에서 미조율 A2가 C를 요청 p95 ITL에서 이기는 셀 있음, TTFT p95는 4셀 전부 A2 우수 — 부호만). "SM 분할만 켠 fused arm"은 이 엔진에서 구성 불가. ⚠️**금지 6건**: ①`component_share` 인용 금지(Granite r3 Fieller 3.098="몫 310%"가 존재하나 코드 자신이 항등식이라 표시). ②두 report json `headline.text` 인용 금지(m=2 Holm을 "4셀 Holm 후"로 오기, 메인 세션이 `holm._family.size_this_job=2` 대조로 확인). ③"분할이 ITL 개선" 꼬리 한정 없이 쓰기 금지. ④Δ^cont 헤드라인 승격 금지(T mean e2e 최대 +41% 악화). ⑤T′ F-B disjunct(i) 미평가 병기. ⑥인용 셀이 지정 셀 아닌 구조적 저용량 복제 셀, 선별 필터 귀무 발화율 40.1%(1−0.95¹⁰, 메인 세션 재계산 일치) 병기. **신규 방법론**: (i) §3 항목18/게이트#9("게이트 자신이 항등식") 새 사례 — §4.5 두 안전장치가 같은 3 꼬리통계만 보고, CONVENTION-SENSITIVE 12지표 중 9개 비트동일. (ii) §3 항목34/게이트#20 새 사례 — `MIN_COVERAGE=0.98` 상수만 수입, 규칙 미구현("식별자 수입≠거동 수입"). (iii) 신규 §3 항목38 — `any()` over n reps 스크린 귀무 발화율=1−(1−α)ⁿ. (iv) 신규 §3 항목39 — §6.3-6 관측자 대칭 보고(`dropped_events`/`writer_error`) 미산출, 감사자 오프라인 복구. 상세 위 rev20 헤더·§3 항목18·34(개정)·38·39, 원자료 `g2s_report_zamba2-27b_877756.json`·`g2s_report_granite-40-h-micro-base_877757.json`. ★★★★★★★★**(2026-08-11, G1-c, job 877974, 0.10 GPU-hr) Gate 2-S Granite r3·r4 명명 제한 조건부 해제 — 새 성능 판정 아님, 크기 인용 셀 확대 아님.** G1-c(Gate 1/G1-b의 Granite 자매 job, 873945 격자 {2,3,4,6} 전부 재현, `PDMUX_TRACE_FORCE_PREFILL=1`·`TRACE_EVERY=32`, 원자료 `results/p1_gates/gate1/gate1c_result_877974.txt`)가 §8.9 전제 판정 규칙(frac((54,54))(pop A 시간가중)<0.01 ∧ max(decode_bs)<36 ⇒ VERIFIED)을 rate 3·4 양쪽에서 발화시켰다: max(decode_bs)=10/10(36 미만)·frac_5454=0.0000/0.0000, 구조적 근항등식 양성대조(pop_A_time_weighted_frac(TARGET_A))=1.0000 4 rate 전부 PASS, grid-completeness 결측 0/expected. 트리 근거는 `results/p1_gates/gate2/runtime_source_manifest_s_granite-40-h-micro-base_877757.sha256`와 15/15 바이트 동일(diff 확인, 메인 세션 재확인) — 873945 상속(G3, 재검증 안 함) 근거뿐 아니라 877757 자신과의 직접 동일성으로 강화됨. ⇒ 해제되는 것은 `PREREG_GATE2S_2026-08-09.md` §5.6.1 `name_for()`의 **명명 층 하나뿐**: Granite r3·r4에서 "간헐 전달(엔진 기본 궤적)"·"A4형" 명명이 허용된다. **크기 인용 자격은 불변**(코드 확인, `g2s_analyze.py:1157-1161` — `premise` 필드는 `nine_cell`·`gate_label`(F-계열) 산출에 입력되지 않고 결과 dict에 나란히 기록만 됨) — Granite r3는 T′ F-계열 발화, r4는 C·T′ F-계열 발화 상태라 **`SIGN ONLY, MAGNITUDE NOT CITABLE`이 유지**된다 ⇒ **크기 인용 가능 셀은 여전히 Zamba2 r2 하나뿐**(세션 초반 "1→3개로 는다" 서술은 원자료로 반증, 과장 정정). **조건부 해제 필수조건 6건(하나라도 누락 시 해제 무효)**: ① 명명 층 한정(부호 서술·헤드라인 명명에만, 크기 인용 불변) ② Granite r3·r4 인용마다 F-계열 gate 병기(`F-SERIES FIRED ⇒ SIGN ONLY, MAGNITUDE NOT CITABLE`, r3=T′·r4=C·T′) ③ 경험적 내용은 하나 — "동거 구간 realized max(decode_bs)=10(r3)/10(r4)<문턱 36"이고 frac((54,54))=0은 그 **코드 귀결**(`multiplexing_mixin.py:900-909`: idx==2 ⟺ decode_bs≥36, 항등식)이지 독립 증거가 아니므로 두 수치를 독립 증거처럼 병기 금지 ④ 증거 등급 명시(`job 877974, n=1, 873945 격자 복제, PDMUX_TRACE_FORCE_PREFILL=1[관측 밀도 32배, 관측자 부하 상한 없음 — G5 UNDETERMINED], Gate 2-S 셀에서의 직접 관측 아님, selector-level·S3 미실행`) ⑤ 트리 근거는 877757 기준(위 문단) — "873945와 바이트 동일 아님(G3 상속)"만 쓰면 오도 ⑥ 아래 Zamba2 근거표 동시 정정. **pop A `t_total_s` 인용 금지(신규, result-analyst)**: 경계 구간 dt가 interior의 11–18배라 5–29% 상향 편향 — G1-c 사전등록의 "A/B 그대로 인용 가능" 허용은 과대 허용이었다. 정직한 bracket(참고용, 크기 결론 아님): r2 [11.07,15.57]s·r3 [15.55,19.10]s·r4 [16.24,19.71]s·r6 [17.26,18.17]s. pop C 절대·상대 분수는 종전대로 인용 금지. ⚠️**해제 후에도 금지인 문장 8건**: ①"크기 인용 가능 셀이 3개가 됐다"/Granite r3·r4의 Δ·CI·%·β 크기 인용 ②"전제가 검증됐으므로 4셀이 동질적이다"(F-계열·검정력 축 비동질성 유지) ③"A4와 T′는 동일하다"(확립된 것은 인덱스 사상 동일 + decode_bs<36 관측뿐, A4는 `_slo_on=False`라 EMA·`controller_decision`·`_r2_decide_idx` 경로 자체가 없다) ④"Granite에서 (54,54)는 도달 불가"(부하가 커지면 도달 — Zamba2 r6에서 실제 발생, 게이트#16 거울상) ⑤"엔진 기본 궤적=물리적으로 prefill 74 SM/decode 34 SM"(selector 라벨, 잔여 차감 미측정, S3 미실행) ⑥"frac_5454=0과 max_decode_bs<36이 각각 전제를 지지한다"(항등식, ③ 재확인) ⑦877974의 TTFT/TPOT/ITL/goodput 어떤 수치도 인용 금지(n=1, 진단 전용) ⑧"Granite에서는 G1-b식 철회가 일어나지 않는다"의 무제한 서술(n=1·이 격자·이 워크로드 스코프 필수, 게이트#16 거울상). ★**성능 결론 불변**: `premise` 라벨은 Δ·paired t CI·Holm·9-셀·F-계열 gate 산출에 미입력(코드 확인) — 위 rev20의 Δ=+13.95/+24.88/+16.04/+19.11ms(4셀 Holm 후 최대 p=1.88e-06)·9-셀 좌표 전부 `S1-C`·크기 인용 허용 셀 Zamba2 r2 하나 문장은 **한 글자도 바뀌지 않는다**. ★**`PREREG_GATE2S_2026-08-09.md` §8.9 정오표(사후 addendum, 원문 미덮어쓰기)**: 기존 §8.9 표의 Zamba2 r2·r3 "전제 상태" 근거란이 **pop-C 시간가중 분수(비동거 구간 `(0,108)` 98%/99%)를 인용**하고 있었는데, 이 양은 `gate1/PREREG_G1B_2026-08-07.md:146`이 few-snapshot dt 팽창을 이유로 **명시적으로 인용 금지**한 것이다(Gate 1 감사 확정) — Zamba2·Granite 양 모델 공통으로 §8.9 표의 근거란을 "**max(decode_bs)<36**" 하나로 좁힌다(보수적 방향; "Granite 기준을 낮춰 맞춘 것"이 아니라 기존 Zamba2 표기가 과다 인용이었던 것의 정정). load-bearing 아님(비동거 분기 `multiplexing_mixin.py:911-912,923`는 decode_bs 비의존 코드 항등식이라 A4·T′ 공통이므로 이 정정으로 §8.9의 판정 자체는 바뀌지 않음). 정본 반영: 위 rev22 헤더·§1-1(이 블록)·§3 항목41(신규, 결정량의 밀도 의존성)·항목42(신규, 실험 payoff는 코드로 검증 후 정당화)·항목43(신규, `compute_coverage`류는 내부 구멍에 맹목). `PROJECT_STATUS.md` "확정된 결과" 1번·"방법론 게이트" #27–29 동반 갱신. `CLAIM_EVIDENCE_MATRIX.md`는 대조 확인 결과 이 항목을 인용한 서술이 없어 갱신 대상 없음. 원자료 `workspace/engine-port/results/p1_gates/gate1/{PREREG_G1C_2026-08-11.md, gate1c_result_877974.txt, gate1c_analyze.py, runtime_source_manifest_gate1c_877974.sha256, manifest_diff_gate1c_877974.txt}`(수정 금지·인용만). ★★★★★★★**(2026-08-11, Gate 2-S E1 addendum, jobs 877756/877757 재집계[GPU 증분 0], claims-auditor 적대 감사 완료) E1 채택 — 새 성능 판정 아님, 등급 하향된 조건부 채택(4셀 전부).** E1(사전등록 `PREREG_G2S_E1_ADDENDUM_2026-08-11.md`, 구현 `g2s_e1_premise.py`, 결과 `g2s_e1_premise_877756_877757.json`)이 §8.9 전제를 Gate 2-S 자신의 셀에서 n=10 직접 산출한 pooled `max(decode_running_batch_size)`(threshold 36, `gate1b_analyze.py:135-137` 동사[verbatim] 수입, 새 자유모수 0)로 재검증했다: Zamba2 r2=**14**(여유 22)·r3=**23**(여유 13)·Granite r3=**10**(여유 26)·r4=**13**(여유 23), 4셀 전부 무결성 I1–I7 통과·`VERIFIED_AT_SAMPLED_INSTANTS` 발화. ★**이번 반영은 승격이 아니라 등급 하향된 조건부 채택이다 — E1이 §2.5에서 스스로 주장한 "G1-b/G1-c 대비 밀도 페널티를 같은 크기로 받지 않는다/이것이 이 addendum이 존재할 수 있는 이유다"는 claims-auditor 감사로 반증됐다.** 채택은 **조건 7개 전부**(하나라도 누락 시 채택 무효) 하에서만 유효: ① 상태명은 항상 `VERIFIED_AT_SAMPLED_INSTANTS`(축약형 "VERIFIED" 단독·"전제가 검증됐다" 단독 서술 금지). ② **밀도 서술 정정 의무** — E1 인용 시마다 다음을 병기: *"E1의 우위는 반복수(n=1→10)와 셀 일치이며, 결정 관련 pop-A 관측 수는 G1-b/G1-c 대비 5–6× 적다(1,279·1,496 vs 6,402·8,920 — pop-A[decode_bs>0 ∧ prefill_active_bs>0] 건수를 `g2s_*_telemetry_agnostic_r{2,3,4}_*.jsonl`에서 직접 재계산해 확인: Zamba2 r2 684+r3 595=1,279, Granite r3 726+r4 770=1,496)."* prereg §2.5의 "밀도 페널티가 같은 크기로 적용되지 않는다/이것이 이 addendum이 존재할 수 있는 이유다"는 **인용 금지**(과소 서술, 반증됨). ③ **bound 병기 의무** — `sup decode_bs ≤ in-system + Poisson(λ·dt)`(새 자유모수 0, claims-auditor 산출) q=1e-9에서 Zamba2 r2/r3·Granite r3/r4 순으로 **25 / 41 / 24 / 27**을 셀별 병기한다. **Zamba2 r3는 가장 적대적 bound에서 문턱을 배제하지 못하는 유일한 셀**임을 명시(q=1e-6에서 이미 **37 ≥ 36**). ④ **복합 규칙 분할 명시** — 수입 규칙은 `frac(idx2)<0.01 AND max(decode_bs)<36`인데 E1은 앞 절반(`frac`)을 §8.9.1에 위임하고 뒤 절반(`max`)만 계산했다 — **두 절반의 출처를 모두** 적는다(감사자 독립 확인: 4셀 전부 `stream_index==2` 0건, §8.9.1 자체 산출). ⑤ **post-hoc 자백을 4셀 전부에** — 작성자가 이 addendum 작성 시점에 이미 (앞선) G1-b의 Zamba2 9/23 값을 알고 있었으므로 Zamba2를 prospective로 표기하지 않는다(Granite 2셀만 §0에서 사후 인지를 자백한 것은 불완전했다). ⑥ **§5.6.1 명명·F-계열·크기 인용 자격 불변** 재확인 문구 동반 — `premise`는 `nine_cell()`·`gate_label()`·`paired_t()`(`g2s_analyze.py:1153-1159`) 어디에도 입력되지 않는다(감사자 코드 확인). ⑦ 아래 "성능 결론 불변"과 "무료 대조"를 함께 적는다. **Zamba2 r3(margin 최소, 인용 시 강제 병기)**: "Zamba2 r3의 여유는 13(문턱의 36%)으로 4셀 중 최소이며, in-system+도착률 상한(q=1e-6)에서 37≥36이라 **샘플되지 않은 순간의 문턱 도달을 배제하지 못하는 유일한 셀**이다. 같은 모델·같은 arm이 G1-b 격자 rate 6에서 실제 초과했고(max 40, frac((54,54))=8.37%), 이 캠페인 자신의 capscan(agnostic, rate 1–6)에서도 max 31(seed7, 메인 세션 재확인)/in-system 40까지 올라간다. 문턱은 이 모델에서 도달 가능한 영역에 있다." **강등하지 않는 이유**: 강등하면 §8.9(rev5 이래)의 Zamba2 r3 "검증됨"이 먼저 무너진다(근거가 G1-b의 같은 값 23·같은 여유 13·더 나쁜 n=1) — r3만 강등은 정본 자기모순이 된다. 여유 기준의 정식 도입은 **별건 결정**이며 그 결정의 첫 대상은 E1이 아니라 §8.9의 Zamba2 r3 명명 허가임을 함께 기록한다. **rev22 필수조건④ 대체(감사자 제시안, 요지 유지)**: ④ 증거 등급 = (i) `job 877974(G1-c), n=1, 873945 격자 복제, FORCE_PREFILL=1 — pop-A 밀도 높음(8,920)`; (ii) `job 877757(Gate 2-S 자신), n=10, 그 셀·A4 직접 관측, FORCE_PREFILL=0 — pop-A 밀도 낮음(1,496=(i)의 1/6)`. **두 관측은 서로를 대체하지 않는다** — (i)은 밀도, (ii)는 반복수·셀 일치를 준다. 어느 쪽도 "샘플되지 않은 순간"을 배제하지 않으며(`TRACE_EVERY=32`, 실효 16반복당 1회), 판정은 `VERIFIED_AT_SAMPLED_INSTANTS` 등급이다. (ii)는 (i)의 값을 Granite r4에서 10→13으로 **상향 정정**한다(n=1 max는 pooled max의 구조적 하한). selector-level·S3 미실행 불변. **구 rev22 필수조건④ 문자열("Gate 2-S 셀에서의 직접 관측이 없다")는 이제 금지 문장으로 전환**(아래). ★**성능 결론 불변**: `premise`는 `nine_cell()`·`gate_label()`·`paired_t()`(`g2s_analyze.py:1153-1159`) 어디에도 인자로 안 들어간다(감사자 코드 확인) — Δ=+13.95/+24.88/+16.04/+19.11ms(4셀 Holm 후 최대 p=1.88e-06)·9-셀 좌표 전부 `S1-C`·크기 인용 허용 셀 Zamba2 r2 하나·TTFT p95 악화 −183~−536ms **전부 불변**. **무료 대조(신규, 감사자 등재)**: 같은 셀에서 T′의 pooled max(decode_bs)=13/23/10/12 vs A4 14/23/10/13(±1 이내 일치, 메인 세션이 `g2s_*_telemetry_pdmux_split34_nosticky_r{2,3,4}_*.jsonl`에서 직접 재계산해 확인) — 전제가 의존하는 **부하 동등성의 직접 증거**인데 이 addendum 자신은 산출하지 않은 값이다. **금지 문장 추가 6건**(rev22의 금지 8건·필수조건 6건은 전부 유효, 추가분): ①"VERIFIED" 단독·"전제가 검증됐다" 단독 ②"E1은 G1-c보다 밀도가 높다/밀도 페널티를 피했다"(반증됨) ③"이 캠페인 자신의 데이터가 G1-c를 대체한다"(대체 아님 — 밀도 vs 반복수는 서로 다른 축) ④`max_decode_bs`를 부하·용량·동시성 대리 지표로 사용(모델 간·rate 간 M 비교 금지 포함) ⑤"Gate 2-S 셀에서의 직접 관측이 없다"(구 rev22 필수조건④ 문자열 — 이제 금지 대상) ⑥E1-b·E1-c 실행 전에 "전제가 이 캠페인에서 검증됐다"고 쓰는 것. ★★**(3) g2s_analyze.py 문서-스코어러 불일치(E1과 독립, 방법론 게이트#20 새 인스턴스, engine-porter 소관 — doc-steward는 코드 수정 안 함)**: `g2s_analyze.py:84-89`의 `PREMISE_LABEL`이 Granite r3·r4를 여전히 `"unverified"`로 하드코딩하고 있고(주석 "Granite has NO realized observation (G1-c not run)"도 stale), 아카이브된 `g2s_report_granite-40-h-micro-base_877757.json`의 `premise_labels`도 `unverified`다 — **rev22/이번 rev23은 문서 층에서만 승격**했으므로 원자료를 직접 읽는 인용자는 정본과 반대되는 값을 얻는다. 처리: (i) 알려진 결함으로 등재(이 문단) (ii) 아카이브 아티팩트의 `premise_labels`는 **G1-c/E1 이전 상태로 동결된 스냅샷**임을 못박는 정오표(이 문단이 그것) (iii) 향후 어떤 재실행에서든 `PREMISE_LABEL`·아카이브 `premise_labels` 갱신을 요구사항으로 등재. **스코어러 자체는 지금 수정하지 않는다**(rev20 결과를 낸 감사 통과 스코어러 — 조용히 패치하면 재실행 결과가 아카이브 json과 어긋나 재현성이 깨진다). **신규 방법론**: §3 항목44(경로 없음 재사용 경계)·45(bound와 점추정 구별) 신설(아래). 항목18·39에 새 사례 追記(아래). **후속 실험(전부 미실행)**: E1-a(사전등록 후 4셀×5arm×capscan 전체에 bound 적용, 제3자/잠긴 스크립트 채점 권장, GPU 0)·**E1-b**(★결정적, Gate 2-S 4셀 A4를 `FORCE_PREFILL=1`·n≥4로 재실행 — 밀도와 셀 일치를 동시 만족하는 유일한 설계, ≈0.3–0.5 GPU-hr)·**E1-c**(★필수, 양성대조: Zamba2 A4 rate6·n≥4·force=1 — G1-b 40[초과] vs 이 캠페인 capscan 31[미초과]인 문턱-교차 구간, 없으면 `VERIFIED_AT_SAMPLED_INSTANTS`는 "한 번도 발화한 적 없는 스크린의 통과"에 불과·게이트#15, ≈0.2 GPU-hr)·E1-d(Zamba2 r3 여유 13의 정면 검정, E1-b에 포함 가능). 상세는 `PROJECT_STATUS.md` "다음 실험 gate" #10 G1-c 하위(갱신), `CLAIM_EVIDENCE_MATRIX.md`/`EXPERIMENT_ROADMAP.md` "P1 트랙" 절. 원자료 `workspace/engine-port/results/p1_gates/gate2/{PREREG_G2S_E1_ADDENDUM_2026-08-11.md, g2s_e1_premise.py, g2s_e1_premise_877756_877757.json}`(수정 금지·인용만) ★★★**(2026-08-14, E-3 realized SM count 프로브, GPU 비용 ≈0[glogin01 무비용]+job 882374) 드라이버가 요청 SM 개수를 반올림 없이 그대로 보고한다 — 새 성능 판정 아님, 위 Gate 1 블록의 인용 금지는 그대로 유지된다.** 사전등록 `workspace/engine-port/results/smsplit_realized/PREREG_SMSPLIT_REALIZED_2026-08-14.md`(§0–§6+addendum 1·2). `pdmux_context.divide_sm()`을 직접 호출해 얻은 `(74,34)`·`(54,54)`·C2 스윕 5지점(92/16, 84/24, 64/44, 54/54, 16/92) 전 7지점에서, `torch.ops.sgl_kernel.create_greenctx_stream_by_value`의 realized 반환값이 요청값과 **정확히 일치**했다(Δ=0, `realized_sum=108` 전 대상, `n_returned=4`) — 로그인 노드(glogin01, `NVIDIA A100 80GB PCIe`)와 컴퓨트 노드(job 882374, gpu43, `NVIDIA A100-SXM4-80GB`) 두 하드웨어에서 레코드가 완전히 동일. 판정 문자열 `REQUEST_EQUALS_DRIVER_REPORTED_PARTITION`. **허용 문장**: "드라이버가 보고하는 green-context 파티션은 이 격자에서 요청값과 일치하며 반올림이 없다(드라이버 자기보고 층)." **금지(전부 유지)**: 이것은 **드라이버 자기보고**이지 "실현 파티션을 측정했다"/"prefill 74 SM·decode 34 SM에서 실행됐다"가 아니다 — 위 Gate 1 블록의 인용 금지는 **그대로 유지**된다(S3 하드웨어 실행 층·`%smid`의 SM id 집합 disjointness는 여전히 미측정). 기존 캠페인 판정문(873944/874478/875293/877974/877756/877757)에 사후 부착 금지, Gate 2 본 질문("PD 분리 자체" 귀속)은 한 눈금도 전진하지 않는다. **부수 함의**: `%smid` R0 사전등록 §0.1의 payoff 항목 1(`g2s_analyze.py:1488`의 `log(108.0/34.0)` 분모 정정)은 **정정 대상이 없음이 확인**됐다(34는 요청값이자 드라이버 보고 realized 값). `%smid` F2의 "다섯 번째 세계"(disjoint ∧ ∪⊊D)는 **개수 층에서는 관측되지 않는다**(`realized_sum=108` 전 대상, id 집합 층은 여전히 미측정). ★**부수 발견(방법론 게이트 #32 새 사례로도 등재)**: 이 프로브가 `glogin01`(A100 80GB **PCIe**)과 컴퓨트 노드(A100 **SXM4**)의 하드웨어 SKU가 다름을 직접 관측해, "8B decode-SM 민감도 측정 노트"의 트래픽·roofline 하드웨어 정정(2026-08-14)의 결정적 근거가 됐다(§3 항목46 追記). 정본 `PROJECT_STATUS.md` "확정된 결과" 1번(E-3 블록)·"8B decode-SM 민감도 측정 노트"·"방법론 게이트" #32, 원자료 `workspace/engine-port/results/smsplit_realized/{smsplit_realized_glogin01_2026-08-14.json, smsplit_realized_882374.json}`(수정 금지·인용만) |
| 2 | **운영점 = cudagraph-ON** | decode wall 제거(TPOT 41→12ms), goodput ~1.5–2×↑. 기존 no-cudagraph 수치는 전부 하한 |
| 3 | ★**layer-type 런타임 정책 全형태 死** | 근거는 **서빙 직접 측정**: 4-모델서 agnostic 4/4 승 + **coordinated per-type 구현이 TPOT 42→124ms**. **(B,L) 2D knee**: 재배분 lever **Diff B ≈ 1.0 (L≥8000, 0.96–1.04)**. ⚠️**정정(2026-07-17)**: **L=2000선 Diff B≈1.35**(B1 1.38/B48 1.34)이고 **격자가 실 서빙 regime(ShareGPT 98%가 L<2k)을 안 덮음** ⇒ **"lever 부재" 기전은 long-context 한정·짧은 L엔 외삽**. 결론은 서빙 측정이 지탱하며, 짧은 L의 死因은 **(D) granularity**로 추정. 시각화 `results/prefill_knee/diffA_vs_diffB.png`. **✅ WIDE 스윕(L 256–32768×B 1–16, 2026-07-18, jobs 857371/857477)으로 확증**: lever는 **L≤512서 실제로 열림**(Diff B 256→1.42/512→1.22), L≥1024 ≈1.0; 기전=짧은 L서 둘 다 SM 미활용(mamba 44SM 포화); **그래도 死**(stakes sub-ms/layer ≪ (D) 42→124ms, batch 무영향). `knee2d_wide.png`. decode-side는 lever 있으나 sub-step (D)drain + **cudagraph 비양립**. §14 예약도 fixed d16으로 degenerate. ★★**강등(2026-08-04, claims-auditor+result-analyst X2′ 재집계, 4,104 ZBPT줄/285셀 블록평균 역산, `workspace/engine-port/results/prefill_knee/AGGREGATE_COMPOSITION_2026-08-04.md` REV.2, UNAUDITED 배너 유지)**: 계측 결함 2종 확인 — (1) **버킷 비대칭**(`src/models/zamba2.py:163` `_zt("attn")`=RadixAttention 코어만·qkv/o_proj/MLP 제외 vs `:259` `_zt("mamba")`=mixer 전체) (2) **누산기 러닝평균**(`:391-394` `_zt_acc`가 emit 시 리셋 안 됨 + `knee2d_wide.sbatch:120`·`decode_knee_vs_ctx.sbatch:90`의 `tail -1`). attn·mamba를 나란히 재구성하면 **둘 다 부풀려지나 attn은 mamba의 1/20~1/2뿐**(비순환 확증: 두 독립 job이 같은 셀서 보고 mamba 111.578 vs 116.648=1.0454×인데 steady 84.012 vs 83.941=1.0008×). ⇒ **"WIDE 스윕으로 확증"은 철회**: steady 재계산 **L=256은 단일값 인용 금지, 밴드 [1.24, 1.34]로만**(추정량 의존, 보고 1.42는 warm-up 편향) / **L=512 1.198**(보고 1.22). **"L=2000선 Diff B≈1.35 ⇒ lever는 L↓에서 열린다"는 REFUTED** — steady **1.0081** [1.0078, 1.0084](보고 1.32/1.38은 런 간 4.5% 불일치, steady는 두 독립 job이 0.08%로 일치); **B=48 행은 shape 혼합 셀이라 별도 인용 불가**. ★정책 단위(모듈 전체)로 환산하면 `R_policy ≈ 1 + w_attn·(DiffB−1)`(`w_attn`=attn/(attn+mamba/6), L=256서 9.6%) ⇒ **1.032/1.031로 소멸** — 단 이는 독립 증거가 아니라 `w_attn`이 작아 U-K가 구조적으로 1로 끌리는 결과다. ⚠️**n_indep=1**(WIDE는 셀당 런 1개) — 모든 CI는 **런 내 블록 정밀도이지 재현성이 아니다**. **판정 자체는 불변**: "layer-type 런타임 정책 全형태 死"는 서빙 직접 측정(4모델 agnostic 4/4 승, coordinated TPOT 42→124ms)이 지탱하며 이 강등의 영향을 받지 않는다 — 바뀌는 것은 **기전 서사**뿐이다("lever는 있었는데 (D)가 삼켰다" → "정책 단위에서 lever가 애초에 없었다", negative가 더 깨끗해짐). ★**Diff A/B를 인용하는 모든 정본 문장에 "no-cudagraph micro" 라벨 필수**(`knee2d.sbatch:61`·`knee2d_wide.sbatch:82` 둘 다 `--disable-cuda-graph`, 기존 미기재). 상세 [layertype_dynamic_NEGATIVE_2026-08-04.md](layertype_dynamic_NEGATIVE_2026-08-04.md)·[layertype_dynamic_POSITIVE_2026-08-04.md](layertype_dynamic_POSITIVE_2026-08-04.md) |
| 4 | ★**얽힘(entanglement)** | prefill·decode가 running batch(`max_running_requests`)·KV 공유 → **decode 굶김 → ITL↑ → batch 정체 → prefill admission 차단 → TTFT 폭발**. 실측: **d16은 prefill에 92SM(최대)를 주고도 TTFT 7.24s**, d24(84SM)는 1.21s. ⚠️**인용금지(2026-08-04, `layertype_dynamic_POSITIVE_2026-08-04.md:159`) — 이 "7.24s" magnitude는 폐기 벤치 n=1**(stationary ShareGPT rate 8, 방법론 게이트 #2로 폐기된 벤치·§1-14의 절벽 rate)이라 **인용 금지**다. **방향**(decode 굶김→prefill admission 차단이 TTFT를 악화시킨다)은 변화 trace n≥4가 지지하므로 그대로 유지한다. ★**doc-steward 정정(2026-08-16)**: 이 인용금지가 `CLAIM_EVIDENCE_MATRIX.md` Claim C·`venue_positioning.md`(§0.1·C3)에 전파되지 않았던 결손을 이번 rev에서 바로잡았다(세 곳 모두 배너 추가) — `PRIZE_SIZE_ARGUMENT_2026-08-16.md` rev1이 바로 이 미전파에 걸려 §2.3에서 이 수치를 반례로 오용했었다(rev2에서 자체 정정, 상세 §3 항목58) |
| 5 | ★**최적 split = 부하 의존 (이동함)** | `최적 D_sm = max(모델 floor[attn-decode knee], 부하항[∝ λ×output_len])`. ★**floor 자체가 ctx 의존 (2026-07-18, job 858811)**: decode step의 attn 비율이 ctx 따라 이동(ctx256=5%→ctx16k=79%)해 **whole-decode SM-민감도가 1.1×(ctx256, SM-free)→10.5×(ctx16k, SM-hungry)**, 최적 decode SM knee **16→44→108→108**. ⇒ 짧은 ctx=decode에 SM 조금·긴 ctx=많이. (per-type 분할 아님=offline predictor 입력; `results/r0c/decode_knee_vs_ctx.png`). ⚠️**단 '민감도'는 triton/no-cudagraph 마이크로벤치 값**: decode-attn은 원리상 memory-bound지만 이 커널은 **HBM 대역폭 미포화(MLP-limited)라 108 SM까지 ~선형 스케일**(효율 44→108서도 ≈1.0). **운영점(cudagraph)선 HE2가 decode non-binding으로 관측** ⇒ 운영점 magnitude는 열린 질문. `decode_attn_saturation.png`. 저-decode-부하(synthetic o32/o96)=d16 / 실 trace(ShareGPT r8)=**d24·d44**. **d16 1.056 vs d24 5.28 = 5× 격차로 노이즈(±1.3) 압도**. ★**오독 방지 각주(2026-08-04, claims-auditor)**: 이 "knee"는 **decode-only·예산 무제약 cost 곡선의 argmin**이며 예산 제약 하 최소 필요 D가 아니다 — `최적 D_sm = max(floor[knee], 부하항)`에 knee=108을 대입하면 **항상 D=108 ⇒ prefill 0 SM**이 되어 PD-mux가 성립하지 않는다(§1-5 내부 자기모순). **이 knee를 floor에 대입하지 말 것.** ★★**계측 결함 3건 확인 + 인용 금지(2026-08-04, claims-auditor, job 858811)**: (i) 위 §1-3과 동일한 버킷 비대칭(`_zt("attn")`=코어만 vs `_zt("mamba")`=mixer 전체) ⇒ "attn 비중"은 decode step의 조성이 **아니다**; (ii) `results/r0c/decode_knee_vs_ctx.sbatch:45` `SM_LIST=(full 44 24 16 8)`로 **`full`이 항상 첫 arm** ⇒ warm-up 편향이 **모든 비의 분모**에 걸림; (iii) **물리 불변량 위반** — mamba SSD decode는 ctx에 O(1)이어야 하는데 sm44에서 **1.98× 산포**, `full`(108 SM)이 sm44보다 **2.34× 느림**(ctx256 65.655 vs 28.057). ⇒ **"attn 비중 5%→79%"·"ctx256 = 1.1× SM-free"·"knee 16→44→108→108" 인용 금지.** 보정 시 ctx256 민감도 **1.140 → 2.513×**. 재측정 진행 중(계측 수정 후 4 ctx × 5 SM × n=3, 미제출). 상세 [layertype_dynamic_POSITIVE_2026-08-04.md](layertype_dynamic_POSITIVE_2026-08-04.md) §2.4·[layertype_dynamic_JUNCTION_2026-08-04.md](layertype_dynamic_JUNCTION_2026-08-04.md) §1.1 |
| 6 | ★**비대칭** | decode **과다공급**=저부하서 거의 무해 / **과소공급**=고부하서 파국 ⇒ **최악 phase 기준 decode-heavy static이 두 phase 모두 안전 → 지배** |
| 7 | ★**동적이 best-static을 못 넘음 (HE0)** — **n≥4 견고, SLO 엄격도 무관 (§1-17로 tight까지 확정)** | **변화 trace**(유효 벤치). **d44 3.220±0.013 (n=4)** > **d34 3.171±0.025 (n=4)** > **bind+GATE 3.132±0.019 (n=9)** > bind no-gate 2.934±0.306 (n=4). d44↔bind+GATE 격차 **0.088 = 5.4 pooled-σ**. (n≥4: d24 3.039±0.130 / slo 2.964±0.025 / d16 2.846±0.055). ※ 전부 **TRUE goodput** — 구 보고값(9.649 등)은 하네스 3× 부풀림, `f921ae8`서 수정, **순위 불변**. ★**tight SLO(chat 300/50)로 재튜닝해도 동일**(§1-17: d44 73.2%≫bind+GATE 44.3%) — 관대 SLO 한정 아님 |
| 8 | **switch overhead는 병목이 아님** | switch 2회로 static 매칭한 rep 존재; **slo(5sw) < bind(21sw)** ⇒ 손실은 (A)overhead 아니라 **(B)positioning** |
| 9 | **§B의 +18%는 confound** | no-cudagraph(비운영점) + vs d44(최적 아닌 static) — best-static 대비가 아니었음 |
| 10 | ★**feasibility 게이트 = 동적 제어가 아니라 "undershooting auto-tuner"** (2026-07-17 규명) | **구조**: 로그상 `2→3`(d24→d34) **1회 decode-ward 이동 후 prefill-ward 복귀를 113회 전부 거부**(`bs=47 ≥ 0.85×48` 상시 참) ⇒ **d34에 영구 고정 = one-way ratchet**. **수치**: bind+GATE **3.132 (n=9)** ≈ **d34-static 3.171** − 0.039(정착 비용). ★**그런데 틀린 static으로 수렴** — 최적은 **d44(3.220)**. 정지 규칙(decode가 더는 급하지 않음: tpot<51ms)이 **최적점 못 미쳐 발동해 ratchet이 조기 정지** |
| 11 | ★**게이트의 가치 = 성능이 아니라 견고성 (트랩 방지)** | **유효 벤치(d44 ±0.013 = 노이즈 없음이 증명된 벤치)에서**: no-gate **2.934±0.306, 1/4 붕괴(2.405, sw=10)** vs gate **3.132±0.019 (n=9), 0/9 붕괴, 분산 16× 타이트**. ⇒ **그 붕괴는 시스템 노이즈가 아니라 컨트롤러 탓**(§2-1 부분 복권). 단 **게이트는 동적을 *안전*하게 만들 뿐 static은 여전히 못 이김** |
| 12 | ★**컨트롤러 CPU 오버헤드 = 死 (직접 계측)** | `SLO-CTLCOST`(v7 이벤트루프 활성 경로 계측): **mean 32–36µs, max 267µs, 누적 ~34ms / ≥1000 call**. 최악의 단일 호출조차 **decode 한 step(ITL p50 ~30ms)의 0.9%**, 누적은 **wall clock의 0.014%**. ⇒ "컨트롤러가 도는 것만으로 이벤트 루프를 지연시킨다"는 가설 **명시적 반증**. 과거 "bind가 switch=0인데 static 미달"은 CPU 비용이 아니라 **§5-4 시스템 노이즈** 탓 |

| 21 | ★★**Stage 0(long-ctx L−2 게이트, 2026-07-26) — ★★★2026-07-28 판정2/판정3 철회(C1 CONFIRMED), 판정1만 생존** | 3-arm coupled-운영점 스윕(M=pure Mamba2-2.7B 음성대조·H=Zamba2-2.7B hybrid·T=Qwen2.5-3B 양성대조, ctx{4k,8k,16k}, decode-SM{16,44,92}+108-ref, jobs 864230+864601, PIN_CHECK 전부 PASS). **판정1(CONFOUNDED, CONFIRMED, 생존)**: raw ITL(D16/D44/D92) 곡선은 decode-SM binding이 아니라 prefill 경합/entanglement 아티팩트 — 이는 prefill=108−D가 항상 공변하는 설계상 사실이라 D108 앵커의 유효성과 무관하게 참이다. 원 **판정2(NULL, CONFIRMED)**: 유일 de-confounded 대조 D16 vs D108 = 1.00±0.01, 3 arm×3 ctx 전부 ⇒ 운영점 decode는 16→108 SM에 무감각. 원 **판정3**: long-ctx 충돌 가설 붕괴, HE0/벡터1이 ctx-무관으로 강화. ★★★**반증(2026-07-28, claims-auditor 사전등록 게이트 집행, C1 CONFIRMED)** — 판정2·판정3 철회: "D108(무경합 앵커)"은 **실제로는 decode 16 SM**이었다. 3중 독립 증거: (i) 코드 기전 — `manual_divisions=[92,16,0]`의 세 번째 값 0이 legacy auto-path threshold로 읽혀 `decode_bs>=0`이 항상 참 → 항상 stream_idx 1=(92,16) 선택(`src/multiplex/multiplexing_mixin.py:725-742`); (ii) realized telemetry 재집계 — decode-active 샘플의 79–96%가 (92,16)(9/9 셀); (iii) telemetry와 독립인 클라이언트 서명 — D108/D16=0.992–1.001(9/9 셀)인데 D92는 3.4–3.6× 빠름(108이 92보다 느릴 수 없음). ⇒ "D16 vs D108=1.00±0.01"은 **동일 조건 반복측정**. ★**"3중 삼각검증" 표현도 철회** — 무경합 앵커는 고장, 음성 대조 M의 전제("decode O(1) recurrent라 SM-bound 불가")도 틀렸음이 확인됨(context 길이의 O(1)이지 SM 수의 O(1)이 아니었다 — `../PROJECT_STATUS.md` "8B decode-SM 민감도" C2 참조), de-batch 논거는 미감사 — 1/3만 남는다. D16/D44/D92의 **pin 자체**는 realized 기준 유효함 유지. **HE0/HE2/§1-5/§1-7은 철회하지 않는다** — 대신 §5-6이 "게이트 미실행"으로 복원되고, 열린 긴장 2건(HE2 vs C2, r0c 부분 복권)이 `../PROJECT_STATUS.md`에 기록된다. ★scope 한정(필수, 판정1엔 여전히 적용): {M/H/T 2.7–3B, triton, cudagraph-ON green-context pdmux, ctx≤16k, coupled 하네스, one-shot 32-conc burst}. 상세 [`stage0_verdict_2026-07-26.md`](stage0_verdict_2026-07-26.md)(원 판정, 위 항목들로 철회됨), `../workspace/engine-port/results/s0_deconfound/PARTITION_RESIDENCY_STAGE0.md`(C1 근거) |

| 22 | **green-context 분할은 decode가 비면 무분할로 auto-revert한다 — 관측 사실, 성능 판정 아님(2026-07-29)** | 코드: `multiplexing_mixin.py:726,745-748`. 결과적으로 **셀 라벨 `[P,D]`는 목표(target)이지 실현(realized) 배분이 아니다**. 실측(`results/s8_frontier/` job 866066, T8, 시간가중 직접 집계): 목표 `[92,16]`(=d16) 셀은 prefill-active 시간의 **85%만** target `(92,16)`에서 돌고 **15%는 무분할 `(108,0)`**에서 돌았다(2.030s 중 0.311s); 목표 `[16,92]`(=d92) 셀은 **100%** target에서 돌았다(47.144s 중 47.091s, 무분할 0.053s=0%). 이 비대칭은 **prefill이 빠른 셀일수록 크다**(같은 뿌리에서 셀별 동시성도 갈린다: 시간가중 `concurrent_time_frac` d16 ~1.3% / d44 4.9% / d92 25–40%, prefill에 SM을 많이 줄수록 prefill이 빨리 끝나 decode와 덜 겹친다). ⇒ **파티션 스윕 결과는 목표 배분이 아니라 실현 배분의 시간가중 분포와 함께 보고해야 한다**(§3-11의 활성률 게이트와 결합). Stage 0(§1-21)의 D108 앵커 실패와 **같은 구조**(라벨 vs 실현)이나 **원인은 다르다** — 그건 legacy auto-path의 threshold 오독이라는 설정 버그, 이건 **정책이 설계대로 동작한 결과**(decode-empty 시 무분할 fallback은 의도된 경로) |

| 23 | ★★**(2026-08-02, 코드 사실) `--max-running-requests`는 arm 계열마다 다른 손잡이이고, `kv_mamba_occupancy=1.0`은 항등식이다 — 성능 판정 아님** | 코드: `sglang/srt/model_executor/model_runner_kv_cache_mixin.py:223-229`. `disable_radix_cache ∧ max_running_requests is not None`이면 **`max_mamba_cache_size = max_running_requests`**(앞 분기 `:218-222`는 `--max-mamba-cache-size` 명시 시, 뒤 `else`는 가용 메모리 ratio 기반 — 이 분기는 s8/E1 계열에서 한 번도 타지 않았다). E1/s8 캠페인이 정확히 그 조건이므로 **SSM 포함 arm(M8/Ha8/Hs8)에서 cap은 admission + mamba state pool 크기를 동시에** 움직이고, **T8(순수 Transformer)에서는 admission만** 움직인다. ⇒ (i) **T8에서 잰 cap 효과는 hybrid로 이전 불가**, (ii) cap을 실험 파라미터로 쓰려면 `--max-mamba-cache-size`를 명시 고정해 두 축을 분리해야 한다. ★**정정(2026-08-02, M5 재설계)**: 그 고정은 **전 arm 공통 절대상수가 아니라 `= cap` 규칙**이어야 한다 — slot당 SSM state 비용이 arm마다 달라(M8 0.255 / Ha8 0.141 / Hs8 0.096 GB) 공통 절대상수는 arm마다 다른 메모리 분할을 강제하는 **새 cross-arm 교락**이 된다(특히 Ha8의 attention KV pool은 이미 48×ctx의 52%만 잡혀 있어 mamba pool 증가분이 곧장 거기서 나온다). 명시 고정은 `:218` 분기를 타므로 항등식을 깨는 목적은 `= cap`으로도 완전히 달성된다. **따름정리**: pool 크기 = cap이므로 batch가 cap에 닿으면 `kv_mamba_occupancy`는 **정의상 1.0** — 이 값을 "hybrid는 메모리가 구속한다"의 근거로 쓸 수 없다(§3-14의 사례 2). §1-4 얽힘의 KV 갈래 등급은 **불변**: 이 regime(ctx 4k ShareGPT)에서 지지되는 것은 좁게 "attention KV(`kv_full_occupancy`)가 어느 arm에서도 구속 근처에 없었다"뿐이며 그 관측치 자체는 **claims-auditor 미통과(인용 금지)**다(`../PROJECT_STATUS.md` "확정된 결과" KV 항목·"방법론 게이트" #5) |
| 24 | ★★**(2026-08-02, M4 — GPU 0) 이 기판의 ITL 꼬리는 decode step time이 아니라 *monolithic prefill이 decode를 멈춘 시간*이 지배한다 — 그리고 그건 구조적이다** | 절대 토큰-방출 시각 재구성(arrival replay + TTFT + 누적 ITL) 결과, rate 2에서 stall probe **17개 중 16개**에서 어떤 요청이 **stall 전 구간 동안 prefill 중**이었고 그 요청은 거의 항상 그 probe의 **최장 프롬프트**(2469–2776 tok)였다. 크기는 고정 프롬프트에서 D에 **단조**(Ha8 seed1: d24 167.7 → d44 225.6 → d54 263.6 → **d92 865.6 ms**) — prefill SM = 108−D 가 줄기 때문. ⇒ **conjunctive goodput의 ITL 항은 decode-SM 레버와 반대 부호로 움직이는 항을 내장하고 있다.** ★**고칠 수 없다**: `server_args.py:6130`이 `enable_pdmux`일 때 `chunked_prefill_size == -1`을 **하드 assert**한다("PD-Multiplexing is not compatible with chunked prefill") ⇒ un-chunked prefill은 연구 대상 기판의 **전제**이지 설정 실수가 아니며, `venue_positioning.md` §0.1의 **(A) green-context 종속** 버킷에 속한다(DuetServe가 libsmctrl로 우회한 바로 그 비용 계열). **따름정리**: ITL 보고는 `A_all`(등록 SLO 항)과 `A_free`(≥1024 tok prefill 창과 겹치는 ITL 제외)를 **병기**한다. `A_free`는 d92에서는 여전히 오염(prefill 16 SM이라 짧은 프롬프트도 막음)이라 **d16–d54에서만 해석**한다. 사전등록 = `results/s8_frontier/DESIGN.md` §4.3.8(a). ⚠️2026-08-01 핸드오프의 기전 추측이 **옳았고** 2026-08-02 감사자의 REFUTED는 **틀린 검정**을 썼다 — prefill 중인 요청은 아직 decode를 안 하므로 자기 stall을 볼 수 없고, 그 요청의 max ITL이 작은 것은 가설과 모순이 아니다 |
| 25 | ★★★**(2026-08-03, claims-auditor + 독립 재현) decode 축은 라벨이 4–19%만 실현된다 — 그리고 기존 pin 게이트는 항등식이었다** | **(a) pin 게이트 = 항등식.** telemetry 120파일·prefill-active 스냅샷 **77,688개**에서 `prefill_sms ≠ target ⟺ decode_running_batch_size == 0`이 **양방향 위반 0건**(off-target∧decode-empty 50,328 / on-target∧decode-busy 27,360). `multiplex/multiplexing_mixin.py:773,792-794`가 decode batch가 비면 **설계대로** 무분할로 떨어뜨리므로(§1-22가 이미 '의도된 경로'로 기록), `e1_pin_check.py`의 시간가중 pin 게이트는 파티션 제어가 아니라 **'prefill in-flight 중 decode가 안 비어 있던 시간 몫'**을 잰다 ⇒ **방법론 게이트 #6 위반**(정확성을 강제하려고 만든 게이트가 저질렀다). 따름정리로 **조건부 pin = 27,360/27,360 = 1.000 정확** — **파티션은 질문이 성립하는 곳에서 완전 실현**(job 872077 재채점 64/64 PASS). **(b) 정작 게이트가 없던 축 = decode.** decode-active 시간 중 `decode_sms == D` 비율(872077, n=8/셀): T8 d16 **0.038**/d24 0.047/d44 0.082/d54 0.093, Ha8 d16 0.104/d24 0.110/d44 0.148/d54 0.187 ⇒ **셀 라벨의 decode 분할은 decode 작업시간의 4–19%만 실현되고 81–96%는 무분할 108 SM**. ⇒ (i) E1 격자는 **지속적 decode-SM 배분을 주지 않으므로** C2가 잰 물리량(prefill 16 고정·decode 연속 D)과 **다른 양**이다 — 긴장 A를 이 격자의 비(比)로 닫을 수 없다; (ii) 희석 계수가 **셀마다 다르다**(T8 0.038→0.093, Ha8 0.104→0.187) ⇒ `A_free(d16)/A_free(d54)`는 SM 수준과 **분할 engagement 비율을 동시에 움직인다 = 추정량 내부 교락**. **Stage 0 D108(라벨≠실현, §1-21)의 decode 축 판본**이며, §1-22가 요구한 실현분포 보고가 prefill 축에만 적용돼 있었다. 신규 게이트 `E1_COND_PIN`·`E1_DECODE_REALIZED`(`e1_pin_check.py`, 2026-08-03)로 코드화, 상세 `results/s8_frontier/DESIGN.md` §4.3.8(h). ★追記(2026-08-14, E-1a Tier 2 T2-4, claims-auditor 기각 — **재확인, 새 발견 아님**): decode 축의 대응 항등식(`decode_sms≠108 ⟺ split_prefill_batch is not None ⟺ prefill_active_batch_size>0`, `src/multiplex/multiplexing_mixin.py:881-923`이 분기 술어이고 `:209-219`가 이를 스스로 문서화)도 코드로 재확인됐다 — 위 (a)의 prefill 축 항등식과 표리(表裏) 관계일 뿐이며 정책 판정에 영향 없음 |

| 26 | ★★★**(2026-08-03, claims-auditor, 같은 날 §1-25 속행) `g`는 이 격자에서 은퇴한다 — 희석-보정 가설 REFUTED, NO VERDICT 사유가 "estimand 미식별"로 확장, `A_free` 자체가 결함, arm 비교에 미제거 교락** | **(A) 희석 attenuation 가설 REFUTED.** 메인 세션이 세운 모형("`E1_DECODE_REALIZED` 4–19% ⇒ `A_free(dD)=w_D·A(D)+(1−w_D)·A(108)` 혼합, 보정 시 Ha8 g≈1.62–1.70")을 3중으로 반증: (i) **control-arm reductio** — 같은 보정식을 T8에 적용하면 corrected g **21–29×**(C2의 2.36–2.91×를 10배 위반, 요구 ITL p95 352–360ms 대 실측 30.67ms); (ii) **de-engagement 직접 실험**(같은 셀 unsplit 분포에서 engagement를 낮춤) — `A_free` 변화 **1–11%뿐**, w=0에서도 g 거의 그대로(Ha8 1.173/T8 1.796); (iii) 핵심 가정 "`A(108)` 셀 무관"이 실측 위반 — Ha8 SPLIT-only 0.920[0.842,0.998](CI가 1 배제, 부호 반대), **T8 헤드라인은 UNSPLIT-only(decode SM 대비가 정의상 0인 부분집합)에서 그대로 재현**(1.795[1.589,2.001] ≈ ALL 1.837). 죽은 것은 보정이지 §1-25가 확립한 "engagement가 낮다"는 전제 자체가 아니다(세 계측기 교차확인으로 견고). **(B) 872077 NO VERDICT 사유 확장.** 코드 사실: `pdmux_context.py:initialize_stream_groups`가 `SM_COUNTS=[(108,0)]+divisions+[(0,108)]`를 하드코딩하고 `multiplexing_mixin.py:773,792-794`가 prefill 비-in-flight 시 무조건 `(0,108)`로 되돌린다 — 즉 이 기판에서 **"decode가 D SM에서 돌았다"⟺"prefill이 동시에 in-flight였다"는 같은 사건**이다. §1-24(ITL 꼬리=monolithic prefill, 크기가 108−D에 단조)와 결합하면 `g`는 사전에 **"decode-SM 탄력도 라벨을 단 prefill-SM 탄력도"**일 것이 예상되고, 실측이 그와 일치(UNSPLIT-only에서 T8 헤드라인 재현)한다 — **n으로 해결되지 않는 설계 결함**. `g = A_free(d16)/A_free(d54)`는 **이 격자 한정 은퇴**(sticky-partition 기판 수정 전 인용 금지), 블록 8→12–16 증설 재실행은 **선행 금지**. ⚠️"Ha8에 decode-SM 레버가 없다"는 CONFIRMED 아님 — 현 데이터는 그 질문에 답하지 못한다, 긴장 A(HE2 vs C2)는 전혀 닫히지 않았다. **(C) `A_free` 추정량 자체의 결함.** `e1_m3_control.sbatch:281-306`: `PREFILL_BLOCK_TOK=1024` 필터가 이 워크로드 요청의 4–5%만 걸러 prefill 작업의 **74–77%가 필터를 통과**(제거되는 ITL은 전체의 ~2%뿐) ⇒ §1-24가 확정한 monolithic-prefill stall이 **d16–d54까지 오염 범위 확장**(§1-24 원문의 "d92만 오염" 한정을 넓힘). 요청의 27.5–29.5%가 output≤25 토큰이라 내부 p95가 사실상 max ITL로 퇴화 — 평균의 선형 혼합 항등식이 극단 분위수에 성립하지 않음(비단조 응답으로 실증). `A_free`는 blocking-제거본이 아니라 대부분 monolithic-prefill stall로 이루어진 극단꼬리 통계다. **(D) arm 비교의 미제거 교락.** 공통 rate 2에서 T8 conc 12.8/decode batch 4.5/ITL p50 ~11ms 대 Ha8 conc 30.6/batch 15.8/~30ms — 양 arm 모두 off-cliff 평탄역(metric cliff 아님)이나 decode batch size가 memory/compute-bound 여부를 결정하는 공변량이라 arm과 완전 교락 ⇒ "attributable to the arm" 문구는 **현재 허용 안 됨**(통제는 realized concurrency/decode batch를 맞춘 rate여야 함); 이 교락은 관측 *방향*을 설명하지 않으므로(batch 큰 쪽이 오히려 무반응) 대안 설명이 아니라 **미제거 교락**으로만 기록. 부수: d16은 `sm_group_num:3`, d54는 4(guard row)로 셀마다 green context 수가 다르고 d54의 `decode_sms==44`는 telemetry에 미관측(guard row 미선택) — 행동 교락 아니나 셀 간 차이. `DESIGN.md` §4.3.8(h)의 "A re-run under RULE_BOUNDS would settle it"은 **이제 틀렸다** — sticky partition 구현 전 재실행은 같은 estimand-미식별 문제를 반복한다. 다음 gate: `PDMUX_STICKY_PARTITION` 구현(engine-porter, correctness gate) → sticky 격자 1회(872077 대조 8 block) → 사전등록 판별 예측(T8≈1.85·Ha8≈0.92 vs Ha8≈1.6, CI 비중첩) → `E1_DECODE_REALIZED≥0.90`이 sticky에서는 항등식이 아닌 **진짜 게이트**. 상세 `results/s8_frontier/DESIGN.md` §4.3.9. **부수(UNAUDITED, 인용 금지 — result-analyst 산출, claims-auditor 미통과)**: `m3_decode_empty.py` 분석이 §1-25의 희석이 decode-empty가 아니라 **prefill 부재**(decode-active 시간에 조건부라 decode-empty는 정의상 분자·분모에 안 들어감)로 나온다고 보고하며, 죽은 telemetry 필드 4개(`decode_ready_queue_depth`·`active_decode_sequences`·`decode_idle_ratio`/`prefill_idle_ratio`)를 코드 근거로 식별한다 — **claims-auditor 미통과이나 부하를 올려 engagement를 높이는 방향이 지지되지 않는다는 결론만은 감사자와 독립적으로 수렴해 그 좁은 항목만 AUDITED로 인용 가능**(rate를 올리면 prefill·decode가 비례해 늘고 split-eligible iteration 수는 셀 무관 194–214로 거의 일정) |
| 27 | ★★**(2026-08-03, 같은 날 2차 속행) `A_free` 대체 추정량으로 estimand 이관 완료[AUDITED, (3)만 예외] + `PDMUX_STICKY_PARTITION` 구현·correctness gate 통과[구현 사실, 성능 판정 아님]** | **(I) 조건부 per-token 추정량**(`m3_conditional.py`, §1-26의 `A_free` 결함을 대체). 단위 = 개별 ITL 구간 1개(요청별 집계 없음), 라벨 `split_frac(a,b)≥0.90`(SPLIT)/`≤0.10`(UNSPLIT)/사이는 양쪽에서 배제(AMBIGUOUS 0.3–1.4%), 통계량은 셀-블록별 **직접 분위수**(primary `p95(SPLIT)`, control `p95/p50(UNSPLIT)` — 두 셀 모두 108 SM이라 대비가 정의상 0). **[AUDITED]**: T8 d16 pooled per-token p95 **11.60**(`A_free`는 28.31, 2배 이상 바깥 꼬리 — 요청의 27.5–29.5%가 outlen≤25라 `A_free`가 사실상 max ITL로 퇴화, `A_free`는 요청 ~9.6개에 얹히는 반면 새 추정량은 셀-블록당 1,390–7,045 토큰). client↔telemetry 시계정렬은 `phase=="benchmark"` 필터 **금지**(그 마커는 warm-up 요청에 발화, 실제 probe 시작이 아님); `ALIGN_R_MIN=0.95` 미달은 flag-only(조용히 배제 금지) — 배제하면 오히려 대비가 커짐(LOO 실측, T8 1.688→1.822). **[UNAUDITED — 감사자가 이번 턴에 새로 생산해 자기 산출을 자기가 감사한 형태, 별도 확증 전 인용 금지]**: `PREFILL_BLOCK_TOK` 스윕(1024→512→256→0)에 **무릎이 없고** 임계 0에서 d16-vs-d54 대비가 두 arm 모두 소멸(0.986/1.005) — 임계는 자유 모수가 아니라 답을 정하는 손잡이. 권고: **`A_free` 은퇴**, `A_all` 유지 병기, **primary 라벨 = realized partition**(`decode_sms==D`, persisted state variable이라 견고); secondary(prefill overlap) 라벨은 872077이 `PDMUX_TRACE_FORCE_PREFILL=0`이라 **계측 결손**(SPLIT 라벨 토큰 중 "overlap-free"로 나오는 비율 Ha8 66.3/38.8%·T8 76.1/27.1% — 물리적으로 불가능해야 할 값, 곧 계측 과소표집의 증거). **(II) `PDMUX_STICKY_PARTITION`**(`multiplexing_mixin.py` +151/−17): decode-busy 시 무분할 fallback을 우회(`if not running_batch.is_empty() and (split_prefill_batch or sticky_partition_enabled)`), **decode-empty 시엔 의도적으로 index 0 release**(hold 아님 — 보호할 decode 작업이 없고 `E1_DECODE_REALIZED`가 decode-active 가중이라 가중치 0). OFF는 short-circuit으로 **패치 전과 byte-identical**(독립 재구현 pre-patch selector와 전 격자 동등성 테스트로 확인). cudagraph 보존(스트림별 캡처 유지, eager fallback 없음). `PDMUX_LA_COORD`/`PDMUX_SLO_SCHED`/`PDMUX_FIXED_DECODE_SM_FILE`/비-`fixed` `PDMUX_R2_POLICY`와의 조합은 init `RuntimeError`로 거부(반쪽 sticky 방지). **correctness gate 전부 PASS**: CPU 회귀(40 tests, sync manifest SHA-256 일치) + sticky 단위 테스트 12건 + **GPU smoke(job 872800, Ha8 d16)**: 고정 프롬프트 6개 greedy 출력이 OFF/ON **byte-identical**. **realized 관측(게이트 아님, n=1)**: sticky OFF `E1_DECODE_REALIZED=0.0839`(기존 동작 재현) vs **ON=1.0000**(사전등록 ≥0.90 초과, 튜닝한 것 없음) — decode-busy 스냅샷이 ON에서 `(idx1,92,16)` 139/139, OFF에서 무분할 진입 146회. **구현 완료 ≠ 성능 주장 성립** — throughput/latency/goodput/`g` 그 무엇도 주장되지 않는다. **(III) sticky 런 사전등록**: 872077 소급 재분석은 **DIAGNOSTIC 전용**(re-score 금지, 설계 판정의 근거로만), primary 통계량 1개(`p95(SPLIT)` 비) 선언, 게이트 4종(`ALIGN_R_MIN`·`E1_DECODE_REALIZED≥0.90`·`AMBIG_FRAC` 상한·`MIN_N_SPLIT` 하한), **★`G_LEVER`/`G_FLAT`는 미결정으로 기록**(기존 1.5/1.15는 `A_free` 스케일이라 그대로 이전 불가 — C2 측정범위(2.36–2.91×)에 묶는 안이 논거는 있으나 E1은 D 범위가 좁고 상보적(P+D=108)이라 그대로 못 씀, sticky 런 제출 전 별도 사전등록 필요), sticky 후 primary 모집단이 `SPLIT ∧ BLOCK-FREE`로 이동함을 미리 등록, trace-force ON 재허용 여부는 **열린 설계 쟁점**으로 미결정 기록. 판별 예측(§4.3.9 승계) 불변: prefill 주도면 T8≈1.85·Ha8≈0.92, 희석 가설이 옳았다면 Ha8≈1.6(CI 비중첩, 8 block으로 구분 가능). 상세 `results/s8_frontier/DESIGN.md` §4.3.10–4.3.12 |

| 28 | ★★★**(2026-08-03, 같은 날 3차 속행, claims-auditor) C2 → `G_LEVER` 앵커 경로 = 폐기, `G_LEVER`/`G_FLAT`는 §4.3.12(d)대로 UNDETERMINED 유지** | `c2_anchor.py`(신규, 미추적)로 시도한 5개 주장을 감사 — **주장 1(realized 검증)만 CONFIRMED, 2–5 전부 REFUTED/NOT-YET-SUPPORTED**. **★§0 결정적 발견(신규, 최상위 열린 항목)**: 같은 arm·같은 서버 플래그(`s8_sweep.sbatch:141-146` vs `e1_m3_control.sbatch:212-218`)·매칭 batch에서 "decode 16 SM" per-token ITL p50이 **2.6× 다르다**(C2 865493 SPLIT bin5 **28.79ms** ≈ `FINDINGS_8B` §2 28.48ms, vs **872077(E1 격자) d16 11.09ms**) ⇒ 둘 중 하나가 거짓: (i) 872077의 `decode_sms==16`이 실제 16-SM 하드웨어 실행이 아니다(`DESIGN.md` §4.3.11이 명시적으로 미검증으로 남긴 잔여층 — green context 생성이 하드웨어 SM 부여를 보장하는지 재프로브 안 함), 또는 (ii) C2의 28–31ms가 decode-SM 비용이 아니라 그 셀 배치(`[16,16,76-idle]`+상시 keepalive prefill 동거)의 성질이다. **어느 쪽이든 C2 비를 E1 격자로 이식 불가** — confound #1의 실측 대 실측 재현. **872077 전체와 이번 세션 sticky 결과가 딛고 선 바닥**이므로 최우선 열린 항목으로 기록. **주장 1** — CONFIRMED이나 서술 2건 정정 필수: (a) "파티션 활성률 0.66–0.93"은 `realized_pin_check.py`의 **스냅샷 개수 가중**이지 시간가중 all-busy(**0.803–0.958**)가 아님(방법론 게이트 #4 위반, 이 감사 한 건에서 3회), (b) 108 SM 시간은 warmup이 아니라 **창 밖 drain 전용**(warmup도 @D≈1.000, 기전은 오히려 강해짐). **구조적 함의(신규)**: in-window residency와 UNSPLIT 표본은 구성상 여집합 ⇒ `E1_DECODE_REALIZED≥0.90`을 통과하는 런엔 **음성대조가 정의상 존재할 수 없다**(865493 UNSPLIT n=0, sticky ON smoke D108 0초) — §4.3.12(e)(i) 확장 필요. **주장 2**(primary p95→p50) — 관측 CONFIRMED·기전 REFUTED(오염원은 monolithic prefill이 아니라 **파티션 전환 인접 구간**, 사건의 83–87%가 전환 0.5s 이내)·**처방 REFUTED 3중**(①같은 배제를 SPLIT에 적용해도 ≤2%만 이동 ②오염 기전이 sticky ON서 소멸 ③872077 실측 `T8 sp_p50=0.996[0.990,1.002]`로 p50 전환 시 양성대조조차 1.00이 돼 어떤 `G_LEVER>1`도 발화 불가=**캠페인을 구조적 NO VERDICT로 확정**하는 처방, 게다가 같은 p95 음성대조가 872077에서 반대 방향으로 깨짐) ⇒ **primary=`p95(SPLIT)` 유지, p50은 secondary, 전환-근접은 진단**. **주장 3**(편향=하한) NOT-YET-SUPPORTED(가산/곱셈 미식별 + 증거가 항을 0으로 만듦 — C2 분할 셀 전부 1410MHz 고정, 클럭 하락은 무분할 np뿐이라 **E1/sticky가 낼 throttling 비용을 C2는 안 냄**=반대 방향 경고). **주장 4**(`G_LEVER=1.41`) REFUTED(끝점 선택만으로 [1.41,2.40] 전 구간 도달 가능, 2.02는 게이트-FAIL job 4셀 포함, 1.41이 872077 T8 CI 하한 1.227을 가로지름). **주장 5**(`G_FLAT=1.25`) 방법 PLAUSIBLE·숫자 REFUTED(LOO 8개 실측 t95 반폭 0.303⇒1.30인데 Ha8 점추정 1.340>1.30로 자기 데이터서 뒤집힘, sticky가 사후 sd를 낮춰 사전-sticky 기반 임계는 관대해지는 방향=귀무 오수용 편향). `n_indep=1`(865493↔865533은 keepalive 설정이 달라 replicate 아님=**세 번째 pseudo-replication**). **regime 매칭 정정**: 872077의 12.8=concurrency, C2의 12.7=decode batch — 단위 맞추면 T8 2.8× 어긋나고 Ha8이 잘 맞음(앞선 서술과 정반대). **남은 경로(계획만, 미실행)**: `G_LEVER`는 (α)sticky 파일럿 T8 양성대조 효과크기 또는 (β)arm 간 대비 `g_T8/g_Ha8`(batch-매칭 rate 필요) — **둘 다 감사자 발안이라 독립 사전등록 필요**; `G_FLAT`는 사후-sticky sd 측정 후 TOST 동등성 마진(역시 독립 사전등록); 감사자 제안 게이트 S1(≈1 GPU-시간, T8 sticky ON d16+d54 C2복제 vs E1복제 2 block, 3갈래 판정) 미실행. 상세 `results/s8_frontier/DESIGN.md` §4.3.13 |
| 29 | ★★**(2026-08-03, 같은 날 3차 속행) D=54 앵커 측정 취소(jobs 872920/872921) — keepalive 재현성 결함[코드/로그로 직접 검증, 미감사] + C2 high-residency=워크로드 장치 산물[독립 수렴, AUDITED]** | 기록 `results/s8_scaleup/NOTES_D54_ANCHOR_2026-08-03.md`(미추적 신규) + 하네스 4파일(미추적, `s8_sweep_d54.sbatch`·`pdmux_p16_d54.yml`·`d54_block_ratio.py`·`runtime_source_manifest_d54.sha256`) — **취소됐으나 설계(d16+d54 동일 캠페인, 4 block, 셀 순서 block 패리티 교대)는 재사용 가능**. **B-1[미감사, 코드·로그로 직접 검증]**: `s8_keepalive_prompt_224.txt`가 **1794 토큰**(★2026-08-16 정정, 구 1793 — 표기만 정정, §3 항목50 addendum3 참조)인데 `CTXCAP=1792` ⇒ 모든 keepalive가 HTTP 400. **865493은 byte-identical한 같은 파일로 `keepalive_errors=0`**이었고 두 srv.log 모두 `CTXCAP=1792` 동일 출력 ⇒ **2026-07-27 이후 엔진 트리 churn으로 context-length 거부가 엄격해졌거나 off-by-one이 이동**(원인 미규명, 자명한 수정 `KEEPA_REPS≤223` 미적용) — **s8_scaleup 캠페인 전체의 재현 불가 요인**이므로 인용 시 경고 필수. 결과: co-residency ~90–100%→**31–34%** 붕괴, 측정된 전 셀 `REALIZED_PIN` FAIL(T8 blk1 d16 0.316/d54 0.628, Hs8 d16 0.342/d54 0.577). **B-2[AUDITED — 독립 수렴]**: sticky OFF는 prefill in-flight일 때만 목표 분할을 유지(`_init_sticky_partition` docstring, `multiplexing_mixin.py:206-231`) ⇒ **C2의 높은 residency는 decode 파티션 제어가 아니라 keepalive 포화라는 워크로드 장치의 산물**. 세 경로 독립 도달: (A) 코드 읽기 (B) keepalive 사망 시 실측 붕괴(위 B-1) (C) 이번 run block-1 telemetry 교차표(`prefill_active>0` @D=**0.975–0.996**, @108=**0.000**, 4파일 — §1-28의 감사자 0.943–0.996/0.0000과 같은 모양이나 **다른 대조**(워크로드 장치 실패 전후)로 도달). ⇒ §1-26(B) estimand 미식별이 **C2에도 그대로 상속**됨 — C2 앵커가 죽는 **세 번째 이유**(§0 축 불일치·estimand 미식별 상속에 이어). 한계: 이 캠페인은 **np(무분할) 셀을 안 돌려** §1-28 주장3의 "np만 1290MHz 하락" 관측을 확증 못 함(분할 셀은 1396–1410MHz 평평, 일관은 하나 독립 확인은 아님). 상세 `results/s8_frontier/DESIGN.md` §4.3.14 |
| 30 | ★★★**(2026-08-03, 같은 날 4차 속행) §1-28 §0의 이분법이 유지 불가 — 세 번째 후보가 실측으로 문서화됨, 오프라인 분리 불가, GPU(S2) 대기 — 성능 판정 0건** | §0은 (i) 872077의 `decode_sms==16`이 실제 16-SM 실행이 아니다 / (ii) C2의 28–31ms가 셀 배치 성질이다 중 하나가 거짓이라고 적었다. claims-auditor가 `FINDINGS_S0_AXIS_2026-08-03.md`를 감사하며(자기감사, 방법론 교훈 12) 이 이분법이 "E1의 `split_frac≥0.90`이 D 파티션 실행 토큰을 올바로 분리한다"는 전제 위에 서 있으며, T8의 세 공유 셀 전부에서 그 전제가 깨진다는 재프레이밍을 냈다 — E1 SPLIT 모집단은 **이봉**이고, 윗봉이 C2 셀별 SPLIT p50과 **1–2% 일치**(d16 0.992/d24 0.991/d44 1.013), 아랫봉은 **같은 job의 UNSPLIT과 통계적으로 동일**(d16 비 1.011). **독립 재현(result-analyst, `S0R_REPLICATION_2026-08-03.md`)**: claims-auditor도 아니고 사전등록을 쓴 세션도 아닌 분석자가 감사된 `m3_conditional.py` 프리미티브만 재사용 선언하고 자체 C2 리더·mode estimator를 새로 작성, 생산자 자체(`label_probe` element-wise, `s0dc_client`의 자기 기록 20/20 정확)에 대조해 실행 — 행 1·3 **재현**(윗봉/C2 비 0.992–1.013 flat, 빠른봉/UNSPLIT 비 1.000–1.023), 행 2(슬로우-쉐어가 D에 단조 증가)는 **순서만 재현**(d24−d16 스텝이 block scatter 안에서 미해결, Δ=+2.37±3.52pp n=8, t=1.90<t_crit 2.365; 감사자가 인용한 수준값 8.95/13.60/23.62/29.38%는 사전등록이 고정한 모집단(`a_free_only=True`)이 아니라 `False` 모집단에서 나온 것 — **사전등록 자체의 내부 불일치**, 방향은 불변), **행 5(클럭 lag sweep)는 발화하지 않음**(최적 δ=+0.10s에서도 slow share d16 11.4%/d44 27.0%, 90% 문턱에 크게 못 미침; δ 절대값≥0.25s에선 UNSPLIT 배경(2.7–3.1%)으로 수렴 — 라벨이 순수 클럭 잡음은 아니되 0.05s 스케일에서 취약함을 동시에 보임). ★**행 4(음성대조)가 강한 형태를 죽였다 — 이 회차의 핵심 결과.** 같은 mode estimator를 **UNSPLIT(108 SM) 모집단**에 적용하면 T8 전 셀에서 동일한 슬로우 모드가 나타나고 그 위치가 셀을 정확히 따라간다(33.88→22.12→15.62→14.12ms, D=16→24→44→54; d54에서는 SPLIT·UNSPLIT 슬로우 모드가 수치까지 동일, 14.12=14.12). 사전등록 falsifier("~31ms 모드가 ~9% 점유")는 d16/d24/d44에서 문자 그대로는 발화 안 함(UNSPLIT share 2.71/3.28/4.84%, 5% 미만 — d54는 5.61%로 발화)이지만 **그 falsifier가 지키려던 실질은 확인된다**: d16의 슬로우 토큰 8,893개 중 **7,903개(88.9%)가 UNSPLIT 라벨**이고 SPLIT은 759개(8.5%)뿐이다. 농축(클래스 내 슬로우 비율/모집단 base rate)은 **SPLIT 2.33–3.24×, UNSPLIT 0.78–0.93×**(d16–d54 전 구간) — `split_frac≥0.90`은 슬로우 모드를 **격리**하는 게 아니라 **농축**시킨다. 순도(SPLIT 토큰의 ~93%가 빠른 모드)도 완전성(SPLIT이 그 job 슬로우 토큰의 8.5%만 포획)도 성립하지 않는다. ⇒ **세 번째 후보 (iii)**: 두 job은 같은 축이나 라벨이 순수하지도 완전하지도 않다 — **§0은 더 이상 이분이 아니라 3지선다이며, 오프라인으로 분리 불가**. 살아남는 두 읽기(셀 수준 현상 vs 클럭 오프셋 누출)는 S2(GPU, 별도 제출 중, 결과 없음)만이 인과적으로 분리 가능 — prefill-overlap 컬럼도 같은(어쩌면 shift된) 클럭에서 계산되므로 arbitrate 불가. **이 층은 §1-25가 이미 기록한 시간 층 희석(`E1_DECODE_REALIZED` 4–19%) 안쪽의 두 번째 층**이고, §1-21(target-vs-realized)·§1-26(B)(estimand 미식별)와 같은 계열이다. **철회 3건**(메인 세션이 같은 날 앞서 씀, `FINDINGS_S0_AXIS_2026-08-03.md` 배너와 동일): "§0 stands as written"(과잉 해석 — 보인 건 3개 인접 집계에서 p50≈11ms뿐), "aggregation-invariant"(정확한 문장은 "11.06ms 단일 모드가 지배적이라 집계 선택에 둔감" — mode dominance이지 estimand identification 아님), "11.09는 집계 단위 미기록"(**틀림** — 산출자는 `m3_conditional.report_conditional` 리포트 [3] `sp_p50=11.0905`, n=11,124, `a_free_only=True`, `m3_conditional.py:158-161,251-262,316-329`에 문서화됨; 없는 것은 그 stdout 저장분뿐 — "저장된 출력이 없다"와 "추정량이 미기록"은 다른 실패 모드다). **재사용 가치 있는 계측 결함 2건**: ★`c2_anchor.py` 표 [5] "UNCONDITIONED CELL SUMMARY"가 **M8 전체와 Ha8 d16을 조용히 누락**(`meta`가 `"t0_monotonic_s" in s` 분기 안에서만 채워지는데, `c2_anchor.py:181-187`, 표 [5]는 앵커가 필요 없음에도) — 865493 앵커 현황은 T8·Hs8=5셀 전부/Ha8=d16 없음/M8=전무이고, **Ha8 d16 rep은 실제로 돌았다**(`itl_ms_p50=112.84`, n=5,200) — "측정 부재"가 아니라 "텔레메트리 앵커 부재"; mode estimator의 60ms 상한은 **arm-이식 불가**(Ha8은 토큰의 0.16%만 창 안, 미해명 ~87ms 스파이크가 구조상 상한 위). **방법론 교훈(§3 신규, 아래 참조)**: 자기가 검증하려는 코드를 복사한 게이트는 항등식에 가깝다(S0 gate 1이 `label_probe`를 복사해 자기 자신과 대조, `wmean`은 무대조) · 여집합 클래스에 음성대조를 걸어라(행 4가 세 차례 놓친 것을 한 줄로 잡음, §3 항목 9와 뿌리는 같고 방향 반대). **증거 수준**: 강한 형태(레버=D-SM 실행 식별)는 **채택 불가**(음성대조 반증), 약한 형태(이봉·윗봉 일치·아랫봉=UNSPLIT·농축 2.33–3.24×)는 **재현됨(독립성 부분적 — 추정량은 감사자 제안, 사전등록은 메인 세션, 실행만 독립, 인용 시 이 스코프 문구 동반 필수)**. §0의 (i)/(ii)는 **여전히 미판정**. 게이트 S1(§4.3.13)은 "부분 실현" 분기가 없어 **현 상태로 실행 불가**(4번째 분기 필요). `G_LEVER`/`G_FLAT`는 §4.3.12(d)대로 **UNDETERMINED 유지**(이번 회차로도 미해소). 상세 `../PROJECT_STATUS.md` "8B decode-SM 프론티어" "2026-08-03(4차)" 소절, `results/s8_frontier/DESIGN.md` §4.3.15, 사전등록 `PREREG_S0_AXIS_2026-08-03.md`·`PREREG_S0R_MODE_2026-08-03.md`·`PREREG_S2_STICKY_ITL_2026-08-03.md`, 재현 판정 `S0R_REPLICATION_2026-08-03.md` |
| 31 | ★★★**(2026-08-05, claims-auditor CONFIRMED scoped) S2(job 873015) 독립 재현 — §1-28/§1-30 §0 최상위 열린 항목이 behavioural하게 종결, E1은 열리지 않음, 성능 판정 0건** | S2(sticky ON, T8 d16/d54, ShareGPT rate 2, n=8 블록, cudagraph ON, gpu37) pooled per-token ITL p50 = **28.92ms**(d16, t95[28.81,29.02])/**12.04ms**(d54, t95[11.96,12.12]) — 사전등록 `[28,34]ms` 안, `split_frac` 라벨 미사용으로 재현(`PREREG_S2_STICKY_ITL_2026-08-03.md` §4 row 1 발화). §0 이분법 종결: **(i)**(872077 `decode_sms==16`=실제 16-SM 실행 아님)은 **하드웨어 형태 REFUTED·라벨 형태 CONFIRMED**(872077 d16이 decode-busy 시간의 96.2%를 D108에서 보냄, 3.8%만 D16) — `decode_sms`는 `arbiter.sm_counts[stream_index]` 재진술(`dual_worker.py:608-623`)일 뿐 하드웨어 SM 부여 직접 프로브(S3)는 여전히 미실행. **(ii)**(C2 28–31ms=셀 배치 성질)은 **DISFAVOURED**(keepalive 없는 open-loop ShareGPT·decode batch 2.6배 작은 조건에서 C2의 0.93×=5.4% 빠르게 재현). 살아남는 답 **(iii)**: `split_frac≥0.90`이 D-파티션 클래스를 격리·완결 못함(sticky ON에선 이 문제 자체가 SPLIT≈전체 인구가 돼 무의미해짐, d16 100.000%/d54 99.969% SPLIT). `E1_DECODE_REALIZED`(시간가중) = **0.9990±0.0017**(d16)/**0.9994±0.0008**(d54), 16/16 cell-block ≥0.995(pre-patch 872077 = 0.038/0.093). **기전 독립 도출**: `runtime_snapshot`이 개수-서브샘플(`PDMUX_DUAL_WORKER_TRACE_EVERY=32`)이고 양 캠페인 모두 `PDMUX_TRACE_FORCE_PREFILL=0`이라, decode-busy 조건부 스냅샷 케이던스가 4 arm 전부 정확히 16 decode step(0.177–0.465s) — ITL 구간 하나가 스냅샷 1/16개를 걸침 ⇒ 872077 d16 SPLIT 모집단 기대 순도 ≈6%, S0-R mode 분해의 독립 경로 값 6.6–9%와 일치. **아티팩트 배제**: 배치 기여는 bin-matching 4.7%/within-run slope 2.7%/블록간 회귀 2.3–3.2% 세 추정 모두 소폭, 노드(gpu36→gpu37)·바이너리(`multiplexing_mixin.py` 1파일 차)·캠페인날짜 전역효과는 d54 companion ≤1.097×로 상한. ★**d54 companion은 사전등록 [13,16]ms 미달**(관측 12.03, CI 전체 13 미만) — `PREREG_S2` §5.3에 "primary 적중+companion 미스" 규칙 없음, 사후분석은 원인을 sticky 교란(부호 반대로 배제)이 아니라 [13,16] 구간 도출 자체의 cross-cell 외삽 오류로 귀속(C2 d44=14.68을 d54 대용 사용, C2 자체 곡선으로 직접 외삽하면 12.79로 이미 하한 미만) — **인용 시 필수 동반**. `DESIGN.md` §4.3.12(f) 판별 예측(T8≈1.85)도 관측 2.402로 빗나갔으나 **판별 arm(Ha8) 미제출**이라 **설계상 미판정**(모형 반증 아님). ★**이 런에는 결과(outcome) 게이트가 0개**: `AMBIG_FRAC`·`MIN_N_SPLIT`·`PREREG_S2` §3.1 일치검사(`p50(SPLIT)`≈`p50(all)`)는 `E1_DECODE_REALIZED≥0.90` 통과 시 SPLIT≈전체 인구가 되므로 sticky ON 하에서 항등식, `E1_DECODE_REALIZED`는 arm 간엔 비항등식이나 ON arm 안에서는 `stream_idx=_sticky_fixed_idx` 코드 불변식, `ALIGN_R`은 계측 flag — 어떤 게이트도 "28.92 vs 11" 결과를 사전 제약하지 않음. **방법론 게이트 #9의 네 번째 재발**로 §3-23에 등재(세 번째 재발은 `S2_ANALYSIS_2026-08-04.md` §3의 §3.1-only 항등식 발견, 이번은 런 전체로 확장). `S2_ANALYSIS_2026-08-04.md:61-64`의 "Instrument check" 문단(케이던스 ~2.0ms 서술)에 **정정 표시**(원문 보존, 삭제 아님) — decode-idle 값을 decode-busy로 오인, 90–260× 오차·방향도 반대. **§0 최상위 열린 항목 해제**: "미해소 3지선다"→"CONFIRMED(scoped)로 종결, 단 하드웨어 층 미프로브"로 전환. C2 자체(레버 존재, 2.36–2.91×, scoped)의 등급·수치는 **불변**(자기완결적 4-arm matched-batch 캠페인, 이 종결의 영향 밖) — 바뀐 것은 "C2와 E1/sticky 격자가 같은 물리량을 재는가"라는 상위 질문뿐(이제 그쪽으로 confirmed, 단 `G_LEVER`/`G_FLAT` 미결이라는 별개 이유로 C2→sticky 이식·앵커는 계속 금지). **E1은 4가지 독립 사유로 이번 회차에도 열리지 않는다**: (1) `G_LEVER`/`G_FLAT` 여전히 UNDETERMINED(post-sticky 블록 sd가 pre-sticky 대비 ~14× 붕괴해 pre-sticky 산포 기반 임계는 null 채택 편향), (2) sticky 기판이 estimand를 바꿈(decode-busy 시 prefill이 벽시계 ~77% 유휴 — co-located `[108−D,D]` 예산 배분이 아니라 단일-테넌트 decode 측정에 가까움), (3) prefill 축 미통제(d16 TTFT p50 46.3→63.2ms, 기전 주장 없음), (4) 음성대조 구조적 부재(d16 UNSPLIT n=0/8블록)+S3 미실행. ⇒ **긴장 A(HE2 vs C2)는 전혀 닫히지 않았다.** **(α) 고-D 대조 셀은 2026-08-05 실행됐다(job 873921, T8 d92=(P16,D92), sticky ON, ShareGPT rate 2, n=4 블록, cudagraph ON, gpu41).** 관측 pooled per-token ITL p50 = **11.26ms**(telemetry-path, per-block t95 [11.049,11.467]) / **11.32ms**(raw-itls path, t95 [11.066,11.568]), `E1_DECODE_REALIZED` = 1.000/0.998/0.998/1.000, `n_err=0`. 사전등록 규칙(`s2_sticky_d92.sbatch:403-408`) 적용 시 **INDETERMINATE**(12–13 밴드에 5.7 block-sd 미달, 28–30 붕괴 밴드에서 ~106 block-sd 이격). **붕괴 분기는 REFUTED** — §0 종결은 유지되며 등급은 **CONFIRMED (scoped) 불변**이다. ★**밴드 부검(필수 동반) — 사전등록 밴드 [12,13]은 잘못 도출됐다(REFUTED).** 앵커 C2 d92=12.88은 재계산으로 정확하나(pooled raw 12.875, n=239,659), **C2와 sticky 격자는 파티션만 같고 워크로드가 다르다**: decode-busy ctx_p50 중앙 **1291 vs 287 tok**, decode batch 평균 **11.31 vs 4.51**, closed-loop+keepalive vs open-loop. α 런 자신의 엔진측 step 회귀(`t = 10.883 + 0.0991·batch + 0.00034·ctx`, n=2,140)로 C2 동작점을 예측하면 12.44–12.80ms로 C2 관측 12.875와 잔차 0.9–3.8%이며, 여기에 이미 기록된 캠페인 계통 오프셋(−5.4%, 아래 "등재 금지"의 "28.92는 구간 중앙에서
견고" 항목 참조)을 더하면 격차가 사실상 소진된다. ⇒ **격차는 새 기전이 아니라 통제되지 않은 워크로드 격차다.** 이 밴드는 이 항목이 스스로 금지한 **C2→sticky 이식**을 예측에 사용한 것이며, 정본이 이미 보유한 더 가까운 앵커(872077의 D108 우세 10.98, α와 블록별 byte-matched·batch 4.56·ctx 283)를 쓰면 예측은 11.0–11.4로 관측과 일치했다. **"α가 §0를 수치적으로 확증했다"는 서술 금지.** ★**α의 실질 기여(순환성 제거).** α는 §0 종결의 근거를 **분쟁 중인 필드(`decode_sms`) 내부의 시간-가중 재진술**에서 **결과(outcome) 축 앵커**로 옮긴다: 블록별 byte-matched trace·batch-matched(4.51 vs 4.56)·ctx-matched(287 vs 283) 조건에서 ON d92/OFF d16(872077) = **1.0293 [1.0210, 1.0376]**, ON d16/OFF d16 = **2.6320 [2.6193, 2.6447]**, ON d54/ON d92 = **1.0658 [1.0586, 1.0731]**. prefill 비공존 조건 엔진측 per-step은 ON d92 **11.048** vs OFF (P0,D108) **10.979**(+0.6%). 872077 d16의 진짜 D16 질량은 라벨 분할이 아니라 **client ITL의 3.43%가 [20,45]ms(중앙 31.7ms)**로 나타나며 실현 시간점유 3.5%와 일치한다. ★**872077 안에서 SPLIT p50(11.08)과 UNSPLIT p50(10.98)의 차이는 1%인데 같은 기판의 진짜 D16 vs D92 대비는 163%다 — 라벨은 사실상 아무것도 분리하지 않았다**(이것이 (iii)의 결과-축 재진술이다). 붕괴 분기는 세 다리로 독립 반증된다: α의 d92 pin(11.3) / sticky **이전** 바이너리 C2의 실현 (P16,D16) 91–96%에서 31.05ms / 872077 자신의 느린 모드 31.7ms. **동반 필수**: 이 비교는 노드(gpu36/gpu37/gpu41)·바이너리(1파일)·날짜를 건너며, **3% 이하 차이는 그 오프셋 안이므로 정밀 일치로 읽지 않는다**(같은 바이너리 OFF arm(δ) 미실행). ★**스코프(유지·강화). S3(하드웨어 부여 층)는 α로 닫히지 않는다.** α는 D92와 D108을 **0.6–2.9%**밖에 벌리지 못하므로 "하드웨어가 92 SM을 부여했다"를 검정할 **검정력이 구조적으로 없다**. α가 배제한 것은 저-SM 가설(2.63×)이지 고-SM 내부 구분이 아니다. 이 항목의 "selector-level, 하드웨어 직접 프로브 없음" 스코프 문구는 **그대로 유지**한다. **다음 gate(재정렬, 2026-08-05)**: **(δ) 같은 바이너리 OFF arm — 승격: "선행조건 아님" → "10% 미만 교차-job 비교를 인용하려면 필수".** 1블록 d92(가능하면 d16도), sticky flag만 OFF, 같은 노드·같은 날. 위 실질 기여의 1.029·1.006 비교가 딛고 선 계통 오프셋을 처음으로 측정한다. ~7 GPU-min. **(β) OFF 1블록 `PDMUX_TRACE_FORCE_PREFILL=1`** — 유지(높음), (iii)의 유일한 양적 다리 직접 검정. 밴드는 **같은 job 내부 값에서만** 뽑을 것. **(α′) 신규(~10 GPU-min)** — sticky ON d92를 **C2의 클라이언트로**(C1024 closed-loop conc16 + keepalive) 1–2블록. 사전등록 예측 **12.4–12.9ms**(α 내부 회귀에서 도출). 적중하면 "C2와 sticky 격자가 같은 물리량을 양적으로도 잰다"가 처음 성립하고 격자 이전 금지 근거 일부가 해제되며, 빗나가면 **C2 앵커는 영구 은퇴**다. **(γ) S3** — 유일하게 남은 스코프 구멍, 우선순위는 δ/β/α′ 뒤. 제출 순서 권고: **δ(7분) → β(7분) → γ, α′는 γ와 병렬**. **등재 금지**: "slow-mass 잔차 2.03×"(REFUTED, count/time-share 단위 불일치), "하드웨어가 16 SM을 부여했다"(미검증), "S2는 A/B다"/"sticky의 인과 효과"(before/after, 매니페스트 1파일 차), "음성대조 통과"(d16 UNSPLIT n=0=검정 불능), `g`/`G_LEVER`/`G_FLAT`/decode-SM 탄력도/goodput/HE0/긴장A 일체(p95비 2.227은 값+CI+"판정 없음" 동반해서만), "28.92는 구간 중앙에서 견고"(하단에서 3.3%, 자기 측정 계통 오프셋(−5.4%)과 같은 크기 여유). ★**추가(α, 2026-08-05)**: "α가 §0를 수치적으로 확증"(REFUTED, 위 밴드 부검 참조) · "11.26≈11.09는 정밀 일치"(교차-job 오프셋 안, 사후 통계량 선택) · "고-D에서 라벨 순도가 다르다"(85/138,588, n 과소) · "α가 하드웨어 층을 닫았다"(D92-D108 판별력 0.6%) · α의 TTFT 수치를 이용한 일체의 성능/프론티어 판정. ★**통제 확인(α).** 매니페스트: 873015 vs 873921 공유 11파일 해시 전부 동일 ⇒ **같은 sticky 바이너리**. 서버 인자 354키 중 차이 4개(`port`·`random_seed`·`pdmux_config_path`·`internal_states`), **양쪽 cudagraph ON**, backend triton 동일. 워크로드: 블록별 `input_lens`/`output_lens` sha256이 872077·873015·873921 전부 동일 ⇒ block-paired 성립. **미통제**: 노드(gpu36/37/41)·캠페인 날짜·클라이언트 seed 경로 ⇒ 3% 이하 비교의 허용오차 미상(δ 사유). 신규 방법론 항목은 §3-26(예측 밴드도 이식 금지 규칙의 적용 대상) 참조. 상세 `../PROJECT_STATUS.md` "8B decode-SM 프론티어" "2026-08-05(α)" 소절, 원자료 `workspace/engine-port/results/s2_sticky/s2a_pooled_873921.txt`·`s2a_T8_873921_result.txt`·`s2_sticky_d92.sbatch`(전용 분석 md 아직 미작성), `../workspace/engine-port/results/s2_sticky/S2_REPLICATION_2026-08-05.md`(§0 종결 전문), `results/s8_frontier/DESIGN.md` §4.3.16 |
| 32 | ★★**(2026-08-16, result-analyst 재분석 R2 + claims-auditor 적대 감사, GPU 0 · 새 서빙 실험 0건) `sgptv{Lo,Hi}` rate-swing 격자(n=4, 정본술어)에서 §1-20의 "+16%" 크기는 재현되지 않는다 — 음성 결과, §1-20 등급 불변** | rate-swing ShareGPT 격자(`sgptv{Lo,Hi}_*_L3H12`, Zamba2-2.7B, cudagraph-ON, 정본술어, 라운드 duration 합산, arm당 `n_indep`=4=별도 SLURM job=별도 서버 부팅 4회)에서 TTFT-pass⊗ITL-p95-pass 분해 DECOUPLED oracle의 best-static 대비 이득은 HI(rate 12)에서 **점추정 +1.67%**, arm-resample 부트스트랩 10000 [+0.21, +5.64], 도너-내 paired t(3) [−3.67, +8.01], rep 단위 jackknife(n=3) 범위 +0.33~+3.55%, pooled trace-level +0.24% — **어떤 재표집·부분집합에서도 한 자릿수**로 +16%와 자릿수가 다르다. **§1-20을 반증하지 않는다**(다른 워크로드 격자이며 he2 격자 안에서는 +16.23%(legacy)·+16.54%(정본술어)가 그대로 재현된다, §1-20 참조). ⚠️★**"116>108 coupling tax가 없다"는 이 격자에서 확인할 수 없다 — 등재 금지.** `SM합=(108−D_ttft)+D_itl>108 ⟺ D_itl>D_ttft`인데 이 격자의 HI TTFT-argmax가 **격자 최대 decode arm(d44)** 이므로 `D_itl≤D_ttft`가 **대수적으로 강제**된다(부트스트랩 TTFT 도너 10000/10000이 d44) — "10000 draw 중 0건 초과"는 증거가 아니라 이 항등식의 재표현(검정력 0)이다. **d54/d64 arm이 미실행인 한 coupling tax의 부재는 미확정.** ⚠️**"TTFT 도너와 ITL 도너가 일치한다"도 등재 금지** — jackknife 4개 중 2개, 그리고 노드·날짜 정합 부분집합(gpu38·2026-07-17: d44 3 rep vs d34 4 rep)에서 ITL 도너가 **d34로 갈린다**(그 경우 SM 98, 이득 +3.08~+3.55%로 3% 게이트를 넘는다). 부트스트랩 ITL 도너 분포는 d44 60.3% / d34 39.7%. 정본 §1-13 각주 F가 이미 "HI d34 3.405±0.478 vs d44 3.613±0.571은 분리 불가"라고 적고 있다. **기전(방향만, 크기 주장 아님)**: he2·sgptv 두 격자 모두 과부하이므로(he2 A 1.45–1.82×, he2 B 3.14–3.70×, sgptv HI 2.15–2.35×) **과부하 여부는 판별축이 아니다.** 두 격자 모두 TTFT-pass 순위=달성 처리량 순위가 8/8 arm 완전 일치하고, 처리량 최대 split이 뒤집히는 것은 요청당 **prefill:decode 토큰비**가 **he2 A 2047:32≈64:1 vs sgptv 341:237≈1.4:1(44.39×)**로 다르기 때문 — "TTFT는 prefill SM이 산다"는 워크로드의 **prefill:decode 작업비 의존**이며 "SLO regime"·"과부하 정도" 의존이 아니다. HI TTFT-pass의 decode-SM 단조 증가(57.08→66.38→68.92→70.71, d44−d34=+1.79%p, unpaired boot [+0.917,+2.625], Welch t [+0.48,+3.10])는 TTFT 임계 1000–7000ms 전 구간에서 argmax 불변이고 ITL 임계를 참조하지 않으므로 ITL 절벽의 산물이 아니다. **필수 동반 한정(생략 시 인용 무효) 10건**: (1) 격자 절단 — d44가 최대 decode arm(d54·d64·d74 config 존재하나 미실행), HI TTFT 증분 감속 중(+9.29→+2.54→+1.79%p)이나 ITL은 포화(+1.33%p, SD 8.8/11.1) ⇒ d54에서 118>108 재출현 가능, 결정량②는 정보량 0. (2) HI는 과부하 2.15–2.35×(threshold-goodput이 런 길이 의존, ill-posed) — `PRIZE_SIZE_ARGUMENT` §5가 요구한 "비-과부하 판본"은 만들어지지 않음. (3) HI는 ITL 축 절벽 위(요청별 ITL p95 중앙값/SLO 0.98–1.00, ±10% 밴드 질량 79.6–85.3%; 임계 55→60ms에서 pass율 d34 20.54→92.12%·d44 18.71→93.46%=5ms에 +72~+75%p) — ITL 도너 정체(55→d34,60→d44,70+→d24)와 크기는 인용 금지. (4) HI 이득 +1.67%는 단일 rep(rep42, job 856890) 견인 — 도너-내 paired 오차 4값 [0.14,0.28,7.67,0.60], 중앙값 0.44%. (5) LO는 천장 절단 — joint 99.1–99.6%, ITL argmax가 4 rep에서 4개 arm으로 흩어짐, LO 부트스트랩은 draw의 67.2%가 108 초과이므로 결정량②를 LO로 확장 금지. (6) arm×node/날짜 교락+시간순 블록 배치+호스트 co-tenancy — d34는 gpu38·07-17 16:59–17:14 단일창 n=4, d44는 3/4가 gpu38·07-17 16:51–16:59로 d34보다 전부 앞선 블록(인터리브 아님), d16/d24/slo는 gpu37·07-18, 교차-노드 대조 0건, 856889/890·856891/892·856917/918이 각각 동일 호스트 동시 실행(856929만 단독). (7) §1-20 대조 극의 크기는 인용 금지 — he2 A "+16%"는 TTFT 임계 ±10%(41점)에서 0.00%~+19.75%로 비단조 요동하고 구조도 39/41점만 불변("+16.2% vs +1.67%" 크기 대조는 성립하지 않으며, 구조 대조도 39/41 강건성 한정, §1-20 참조). (8) 양성대조 표기는 "48/48"이 아니라 "47 독립+PC6 1건은 PC4와의 항등식"(`identity_proof`가 8/8 셀 비트 동일 assert) — `itl_p95_pass_frac`(ITL 도너 결정량, HI 작동구간 17–100%)에 독립 대조 0건, `joint_pass_frac` 무대조, 임계 사다리 52개 SLO 설정이 대조 밖(전부 스크립트가 `ACKNOWLEDGED_CONTROL_GAPS`로 코드 등재). (9) `HE2_RESULT` 라인 인용 영구 금지(§1-19 참조, `he2_bench.sbatch:92` 게이트 #7 버그 잔존). (10) mix축(§1-18)·he2 격자로 확장 금지. ★**방법론 게이트 #9 열 번째 재발 + 게이트 #18 사례**: 결정량②의 항등식은 `PRIZE_SIZE_ARGUMENT_2026-08-16.md` §2.3(2)에 이미 문자 그대로 적혀 있었다 — 저장소가 이미 가진 진단을 자기 산출물에 적용하지 못한 사례(§3 항목59). 상세 `../workspace/engine-port/results/slo_sched/ORACLE_REANALYSIS_2026-08-16.md` §3, `oracle_reanalysis_2026_08_16.py`, `oracle_reanalysis_2026-08-16.json` |
| 33 | ★★★**(2026-08-17, G16 캠페인 완료 — 4블록 jobs 884336/884410/884411/884412, claims-auditor 적대 감사 완료, GPU≈3.9 GPU-hr, 정책 순위 변경 0건) gate #16 재정식화판 결정량 산출 — 양 phase `ITL_SATURATED`(정보 있는 음성), SLO payoff 구간(≲58.6ms)은 무판정** | 사전등록 `PREREG_G16_RULES_REV3_2026-08-16.md`(§4/§6/§11 + addendum C/D/E) 축자 실행, 결과 `G16_RESULTS_2026-08-17.md`(rev2, claims-auditor 감사 반영). **H-1~H-8 판정**: H-1 CONFIRMED(안정성 술어 필수) · H-2 **식별 CONFIRMED / "처리량 최적" 라벨 REFUTED** · H-3 PLAUSIBLE(조건부) · H-4 CONFIRMED(여유 술어 필수) · H-5 (a) CONFIRMED/(b) REFUTED(항등식)/(c) CONFIRMED(재서술) · H-6 전반부 CONFIRMED/배율 REFUTED · H-7 **NOT-YET-SUPPORTED** · H-8 CONFIRMED. **핵심 사실**: (1) HI `M_ttft` argmin = 내부점 **d44**(4/4 블록·부트스트랩 1.000, 표류·곡률 보정 후 불변) — ★그러나 **"처리량 최적"이라 부르지 않는다**: 달성 처리량 argmax는 **d64**(5.6096 vs d44 5.5361 req/s, +1.33%, 페어드 t(3) CI [+0.049,+0.098], 0 배제; goodput argmax도 d44 3.7973 vs d64 3.7752로 +0.58%뿐이라 "goodput 최적"도 헤드라인 아님). (2) `Δ_SLO`는 **exact band로만** 인용: `[58.533,58.625)`→**+20 TAX_POSITIVE**(폭 0.092ms) · `[58.625,60.461)`→−10 · `[60.461,83.952)`→−20. **이 캠페인의 유일한 payoff 구간(≲58.6ms)은 무판정**이다 — 도달 경계의 `Δ_SLO`는 대수적으로 `delta`(=`D_itl`분포)와 동일하고 P(부호>0)=0.632. 60ms 운영점의 `Δ_SLO=−10`은 **구 4-arm 하위격자만으로도 동일**(감사 재계산) — 확장 3 arm이 산 것은 값이 아니라 `D_ttft`의 절단 해제뿐(구 격자에서는 `TRUNCATED`·`sign_forced=True`). (3) LO는 `ITL_UNCONSTRAINED`이나 **ITL은 decode SM에 악화**(d16/d24 12.27–12.30ms vs d44–d74 14.19–14.49ms) — 정보량은 **약 3.1× 여유**일 뿐(최악 arm 14.49ms가 사다리 바닥 45ms의 1/3이라 판정이 사실상 강제됨). (4) HI 판정 안정성 **P(gap_upper>δ)=0.266**(블록 재표집) · LO 0.000 — HI의 `ITL_SATURATED`·K9 `NO_MORE_BLOCKS`는 약 4회 중 1회 반대로 나올 결정(관측 격차 0.634ms은 참 평균 완전 동일의 기대 range와 사실상 같으나 최대 ~2ms 실차는 배제 못함). (5) A-2 조건부 라벨 크기(시간가중, result-analyst 실측, `residency_scope_2026-08-17/`): **d16 7.67±0.26% → d74 20.28±0.77%**(t(3) 95% CI, 28/28 부팅 [7.44,21.33]%) — 불균형 배율은 **(분모,창,arm집합) 3개 명시 필수**(`W_tim_all` d16→d74 **2.65×**[LO창 3.22×·HI창 2.34×], A-2 자신의 count 분모 `W_cnt_da` **3.52×**) — ★이전 문헌의 "2.0–2.65×"는 서로 다른 arm 집합(`U` 부분집합 1.98× vs 전 arm 2.65×)을 한 범위로 합친 **오기**였다. A-2 원문 baseline "53–68%"는 이 캠페인에서 **재현되지 않음**(선택 스크립트가 저장소에 없어 재계산 불가) — **비교 대상 아닌 인용문으로만 취급**(`g16_analyze.py` 스코프 문자열도 같은 수치를 담고 있어 doc-steward가 코드 주석으로 병기). (6) `gap_upper` 비식별은 **구조적 사실**(혼합구조+측정된 노출 이질성 `w(d74)−w(d64)`=+4.57±0.39pp(LO)·+5.15±1.27pp(HI), 희석계수 `1/w̄`=9.5×(LO)·4.3×(HI)) — ★단 "노출차만으로 gap이 정확히 재현된다"는 검사는 **식 1개·자유모수 1개의 대수적 항등식**(반증가능성 0)이라 **증거로 인용 금지**(§3 항목65). 과결정판(전 4 arm)은 반증됨(블록 재표집 20,000회 단조 순서 0회, P=0.0000, 균등 1/24=0.042) — 단 개별 쌍은 전부 CI 0 포함이라(HI `d74−d64`=+0.634±1.493 등) **결합(joint) 진술로만** 인용, "어떤 두 arm이 다르다"로 쓰지 않는다. **정직 고지 2건(필수 병기)**: (i) 동률 가드가 데이터 도착(04:54–09:31) **약 5.8시간 후**(15:19–15:21) 적용됨(addendum E-1-c가 "블록 도착 전"이라던 기존 기록을 **철회·정정** — 완화 근거 3건: 캠페인 리포트 JSON 산출 0건이라 결정량 미산출·PC 판정 수정 전후 완전 동일·이 데이터엔 동률이 원리상 부재[빈 ITL 요청 0건]). (ii) **블록 2–4의 동시 배치**(blk2·blk3이 gpu41에서 동시 실행, 시작·종료 초 단위 일치)는 **메인 세션이 블록 2–4를 한 배치로 제출한 것이 원인**이다 — 사전등록 실행계획(블록 1 실소요 확인 후 순차 제출)을 따랐다면 발생하지 않았다. ⇒ **`n_indep(placement)=3`**(gpu42 단독 1·gpu41 동시쌍 1·gpu41 단독 1)이지 4가 아니다 — K3의 ≥80%는 4회 실행 빈도이지 4개 독립 배치에 대한 것이 아니며, 교란 크기는 이 4블록으로 산정 불가(블록 제외·재가중·사후 블록 추가 금지). **스코프**: Zamba2-2.7B·ShareGPT·부하점 2개(LO rate3/HI rate12)·green-context SM 분할 단일 기판. ★★**정본 승격 금지**(claims-auditor 지정, 전부): "처리량 최적 split=d44" · exact band 없는 "tax 없음/10 SM 여유" 단독 인용 · `Δ_SLO`를 1ms 사다리 3행만으로 인용 · "노출차만으로 gap 전체가 재현된다"·`Δ`=6.6/12.3ms를 측정치처럼 · "순서가 반증한다"를 쌍 진술로 · "2.0–2.65×"(오기)·"배치 효과 ±9% bound"(§2.1과 모순) · split 상태 decode batch 절대값 33.04/20.39/19.49(재현 경로 없음) · 분석기 `MANDATORY_BAND=[+0.34%,−1.92%]` 문자열(구 격자 하드코딩, 이 캠페인 실제 밴드는 +0.77%/−2.29%) · **"gate #16을 닫았다"** · HE0 부활 · `delta` 단독 · 07-15/18 절대값 비교 · C2 인용정지 (a)(b) 해제. 상세 `workspace/engine-port/results/slo_sched/G16_RESULTS_2026-08-17.md`(rev2)·`PREREG_G16_RULES_REV3_2026-08-16.md` addendum C/D/E, `PROJECT_STATUS.md` "다음 실험 gate" #16(갱신)·"방법론 게이트" #45·46(신설), `CONSENSUS.md` §3 항목65·66(신설) |

| 13 | ★★**HE0의 구조적 이유 — 두 regime의 최적이 *충돌하지 않는다*** | **TRUE per-phase goodput**: 정책간 spread가 **LO(rate 3) 0.067 (2.3%) vs HI(rate 12) 1.187 (43%)** ⇒ **차별의 ~95%가 과부하 phase에서 발생**. LO는 split에 **무관심**(prefill-heavy 극단 d16 2.861 ≈ decode-heavy 극단 d44 2.858 = 구분 불가) ⇒ **LO엔 쫓아갈 최적점이 없고, HI의 최적은 LO에서도 공짜**(§1-6 비대칭의 정량 확인). ⇒ **"항상 HI 최적"=decode-heavy static이 정의상 최선**이고 동적은 과도만 지불. **동적이 이기려면 regime 간 최적이 *충돌*해야 하는데 이 워크로드엔 그 구간이 없다**. ⚠️**정정 이력**: 2026-07-18(§1-16)엔 "이 논증은 관대 SLO 한정, tight선 HI 최적이 동적"이라 봤으나, **§1-17(직접 재튜닝)이 반증** — tight SLO에서도 HI 최적은 **고정 decode-heavy(d44)**이고 동적은 얽힘 트랩으로 열위. ⇒ **이 논증은 tight SLO에서도 성립**(SLO 엄격도 무관). ★**각주(2026-08-05, ceiling-censoring 진단 + claims-auditor 감사 — 2026-08-04 각주 교체)**: LO goodput은 이중으로 절단돼 있다 — 처리량 인자는 도착률에(도착 span 199.0s vs duration 209.7s, `throughput ≤ 2.861` 전 arm), pass 인자는 SLO 여유에(`pass ∈ [0.973, 0.998]` 16런 전부). 후자는 런을 늘려도 남는다. **LO에 레버는 실재한다**: `요청별 ITL p95의 p90` 53.36±1.32 → 46.54±1.09 ms (**5.6 SD**, 4점 단조), 중앙값 12.183±0.131 → 13.912±0.216 (3.2 SD), `TTFT p50` 73.56 → 79.61 ms (3.3 SD). ⇒ **"LO는 split에 무관심"은 지표의 무신호이지 레버의 부재가 아니다.** ⚠️ **단 LO 레버는 SLO 예산 단위로 HI의 1/14이고 통계에 따라 부호가 뒤집힌다**(중앙값은 d16 우세, 초과질량은 T<15.5에서 d16 우세·T>15.5에서 d44 우세) — **"어느 split이 LO에서 좋다"는 functional 없이 정의되지 않는다**. §1-13의 "HI 최적이 LO에서 공짜"는 **goodput functional 한정 참**. **HE0 불변.** ⚠️**배제(진단서 원안 중 REFUTED, 인용 금지)**: T=20/30/60 임계 사다리 논증 전체(진단서 §3.4·§5 근거5) · "[55,65) split-불변 모드가 >60 질량을 지배"(§3.5 후반) · C2 정성 대조(§4 "T4 참고 대조") · "gpu39 3중 사다리"(§3.6, line 286 — d24 rep1의 실제 노드는 gpu43이지 gpu39가 아님, line 32 사실오류). 상세 정정 이력은 `workspace/engine-port/results/slo_sched/CEILING_CENSORING_DIAG_2026-08-05.md`(AUDITED, 정정 표시 포함, 원문 보존). ★**각주(2026-08-05, F — 허가)**: §1-13의 "HI spread 1.187 (43%)"은 **하네스 인라인 스코어러의 legacy mean-ITL 판정**(`sharegpt_vary_bench.sbatch:94` `m=sum(I[i])/len(I[i])`)이고 **d16은 n=1**이다. 정본 술어(요청 내 ITL p95, n=4)로는 HI d16 0.443±0.016 vs d44 3.613±0.571 = **+716%**(legacy 2.859±0.241 vs 3.924±0.046 = +37%). **순위·부호·"차별의 대부분이 HI"라는 결론은 불변, 오히려 강화**(정본 술어 n=4로 HI 몫 **97.3%**, legacy 91.9%). ⚠️ 단 정본 술어에서 HI는 rep 분산이 커져 **d34 3.405±0.478 vs d44 3.613±0.571은 분리 불가(0.4 SD)** — "+716%"는 **d16↔d44 쌍에만** 쓴다. ★**각주(2026-08-05, E — 재작성본, 진단서 원안은 귀속 오류로 불허)**: §1-13의 LO 컬럼은 **arm마다 n이 다르다**(`bench_noise_root_cause.md:74-86`: d44/d34/bind-nogate n=4, bind+GATE n=9, **d16/d24/slo n=1**). spread 0.067의 두 끝점(d16 2.861, d24 2.794)이 **둘 다 n=1**이고, "d16 2.861 ≈ d44 2.858"은 **n=1 값과 n=4 평균의 비교**다. 정본 술어 n=4 재산출: d16 2.831±0.038 / d24 2.761±0.140 / d34 2.835±0.025 / d44 2.848±0.007 ⇒ **spread 0.087 (3.1%)**, 그러나 **d24 한 arm의 rep SD(0.140)만으로 spread 전체를 삼킨다**(bind-nogate SD 0.110도 마찬가지). ⇒ **"LO는 split에 무관심"의 정량 근거는 spread 값이 아니라 "spread < rep SD"라는 부등식으로 다시 써야 한다.** 결론 불변. ⚠️ 진단서가 쓴 "spread 0.067 전부가 d24 rep1의 TTFT>3s 12건에서 나왔다"는 **REFUTED** — bind no-gate가 n=4로 2.795±0.110에 앉아 있어 d24를 지워도 spread 0.066으로 사실상 불변이다. 이 귀속을 정본에 쓰지 마라. ★**각주(2026-08-05, 노드 교락 방어)**: d16 = gpu39×1 + **gpu37×3**, d44 = gpu39×1 + **gpu38×3** ⇒ arm과 노드·날짜가 3/4 rep에서 교락(진단서의 "gpu39 3중 사다리" 방어는 사실오류로 못 씀 — 위 배제 항목 참조). arm 내부에서 노드 효과를 직접 추정하면 d16 **0.047ms** / d44 **0.33ms** vs arm 효과 **1.73ms** ⇒ 노드 효과는 arm 효과의 **3–19%**(`frac(>20)`에서는 ≤6%). 완전 매칭 대조 존재: **d34/d44 rep41-43은 같은 노드(gpu38)·같은 날·같은 rep 인덱스**로 paired 가능하며 부호 유지. |
| 14 | ★**stationary r8의 "시스템 노이즈" = 메트릭 절벽 (외인성 아님)** | 워크로드 4런 전부 동일(fingerprint), 하부 섭동은 **thru 3%·ITL 8%**뿐인데 goodput 2× — **r8이 TTFT≈SLO(3s) 경계에 앉아** 3% 결손이 TTFT 평탄역을 1.5s→3.7s로 밀어 임계선을 넘김. **3=견고/8=불안정/12=견고** ⇒ 경계 regime만 불안정. 상세 [bench_noise_root_cause.md](bench_noise_root_cause.md) |

| 20 | ★★★**Oracle 재구성(TTFT⊗ITL 분해): headroom은 +2%가 아니라 +16% — 단 그건 disaggregation 몫 (사용자 지적, 2026-07-19)** | §1-19의 "+2.1%"는 per-static oracle이라 **coupled 절충점만** 봄. **goodput을 TTFT-pass ⊗ ITL-pass로 분해**(사용자 지적)하면 진짜 headroom이 보임: phase A(양 SLO 동시 binding)서 **d16 TTFT-pass 57.8%(ITL 실패) / d24+ ITL 100%(TTFT 하락)** — **DECOUPLED oracle**(d16의 TTFT ⊗ d24의 ITL)=**57.8% = best static 49.7% 대비 +16%**. ★**그러나 그 headroom은 92 prefill SM(d16-TTFT) + 24 decode SM(100% ITL) = 116 SM > 108 요구 = coupling TAX(8 SM 초과)라 단일-GPU 불가**(+ 얽힘이 batch로 추가 결합). ⇒ **두 개의 다른 천장**: 단일-GPU **동적**=coupled ceiling **+2%**(못 이김) / **decoupling=disaggregation ceiling +16%**(별도 디바이스 풀서만). **진짜 headroom은 디바이스 간에 존재, 단일-GPU split(동적이든)엔 없음.** `oracle_corrected.png`. ★★**등급 강등(2026-08-04, claims-auditor)**: 이 오라클(+16%)은 §1-19(극단 disjoint mix, jobs 860497–518)의 데이터를 그대로 재사용하는데, 정본 자신이 §5(열린 항목 (c), 아래)에서 그 캠페인을 **n=1~2·overload-only·underpowered**로 이미 기록하고 있다(§2-4 방법론 "n≥4 없이 정책 결론 금지"에 못 미침). ⇒ **"+16%"는 "확정"이 아니라 n=1~2, 미확증으로 강등한다.** 결합 가정(서로 다른 arm의 주변 pass율을 합성해 DECOUPLED oracle을 만든 것)도 별도로 미검증이다. 판정(단일-GPU와 disaggregation은 별개 천장)의 **방향**은 §1-4/§1-6 얽힘·비대칭 기전과 정합해 그대로 두나, **"+16%"라는 magnitude는 인용 시 이 caveat 동반 필수**. ★★**R1 rev2 확정(2026-08-16, result-analyst 독립 재현 + claims-auditor 적대 감사)**: 정본술어(p95)로 재채점해도 **+16.54%**로 견딘다(legacy 재현 +16.23%, §1-20 자기 술어와 정합). ⚠️단 **ITL 도너가 d24/d34 완전 동률(99.7396%)** — d34를 택하면 SM 합은 **126**(108 초과폭 8→18)이라 **"116"은 tie-break 의존**이며 애초에 대수적 재진술이다(§1-32 결정량② 참조). ★**절벽 위 크기 요동(rev2 신설, 등재 금지 대상)**: TTFT 임계 3000ms ±10%(2700–3300ms, 15ms 스텝 41점)를 훑으면 이득이 **+0.00%(2715/2745/2760/2775/2790ms)~+19.75%(3270ms, 3300ms는 +19.42%)로 비단조 요동**하며, 구조(TTFT도너 d16/ITL도너 d24/SM=116)는 41점 중 **39점만 불변**(2760·2775ms에서 TTFT도너가 d24로 붕괴해 SM=108) — **"+16.2%"라는 크기 자체를 단독 헤드라인 숫자로 인용하지 말 것**, §1-32의 sgptv HI(+1.67%)와의 "+16.2% vs +1.67%" 크기 대조도 **성립하지 않는다**(he2 쪽이 요동 범위 자체가 0~20%). 인용 가능한 것은 **구조 대조**뿐: he2 A는 두 도너가 격자 반대 끝(d16 vs d24+)에, sgptv HI는 같은 끝(d44 근방)에 있다. 상세 [layertype_dynamic_POSITIVE_2026-08-04.md](layertype_dynamic_POSITIVE_2026-08-04.md) §2.5(P6), `../workspace/engine-port/results/slo_sched/ORACLE_REANALYSIS_2026-08-16.md` §2-3 |
| 19 | ★★★**disjoint-feasibility region은 존재하나 동적은 거기서도 패배 — 이유는 conjunctive SLO의 구조 (사용자 극단-도전 검증, 2026-07-19)** | 사용자 논리(어떤 static도 양 phase 두 SLO 동시충족 못 하는 workload 필연 존재)를 극단 mix(A prefill-heavy in2048/o32@8, **B decode-heavy in2048/o512@5=긴ctx라 ITL-binding**)로 실증: **median-feasibility DISJOINT 확인**(feasible-A={d16} ∩ feasible-B={d24,d34}=∅; job 860497–518). ★**그런데 동적 여전히 패배**: graded goodput서 per-phase 최적이 **인접**(A→d24, B→d34)이라 **ORACLE 동적조차 best-static +2.1%뿐**(d24가 양 phase 근최적: A 2.73=최적, B 1.01 vs 1.09), **reactive bind는 −20.6%**(오배치, 양 phase 실패). ★**깊은 이유**: conjunctive SLO(TTFT∧ITL)가 동적을 **동기부여**(d16 최고TTFT·d44 최고ITL)하는 바로 그 힘이 **각 phase 최적을 중간 compromise로 당김**(d16은 A서 ITL벽·d44는 B서 TTFT벽) → 서로 다른 phase 최적이 인접 → 단일 중간 static이 양쪽 서빙. ★**게다가 이 region은 OVERLOAD서만 존재**(전 정책 gp 0.99–1.87, 대다수 SLO 실패): 용량 이하=전부 통과(static 자명)·이상=전부 실패(static 최소손실). **동적이 유용하게 이기는 operating regime 없음.** `extreme_disjoint.png`. ★★**강등 배너 추가(doc-steward, 2026-08-16, §1-20과 같은 근거로 소급 적용)**: 이 행의 "+2.1%"도 §1-20의 "+16%"와 **같은 캠페인**(jobs 860497–518)이며 **같은 n=1~2·overload-only** 조건이다 — §2-4 방법론("n≥4 없이 정책 결론 금지")에 못 미치고, 이미 §5-8(c)가 이 캠페인을 underpowered로 기록하고 있었으나 이 행 자체엔 배너가 없었다. 추가로 **claims-auditor 1차 재계산(독립 재확인 전, `PRIZE_SIZE_ARGUMENT_2026-08-16.md` §2.4)이 "+2.1%"가 집계 단위(phase 무가중 평균 vs 하네스 자신의 pooled trace-level, +5.9~6.1%)와 술어(legacy mean-ITL vs 정본 goodput p95 술어)에 이중으로 의존하며, 정본 술어로 재채점하면 phase B가 전 arm 0.0%가 돼 오라클 자체가 미정의됨을 보였다** — 숫자는 **아직 정본으로 승격하지 않는다**(선행 재분석 R1, `../PROJECT_STATUS.md` "다음 실험 gate" #14). 정본 숫자 교체는 R1 완료 후. ★★**R1 완료(2026-08-16, result-analyst 독립 재현 + claims-auditor 적대 감사, `ORACLE_REANALYSIS_2026-08-16.md` rev2) — 재현 확정, 등급 상향 아님**: phase-mean **+2.28%/+2.35%**(정본 "+2.1%"과 정합 — "+2.1%"은 반올림 2자리 값끼리 계산한 결과였음이 확인됨) vs 하네스 자신이 정의한 pooled trace-level `gpC=(g_A+g_B)/(d_A+d_B)` **+5.88%/+6.12%**(phase B duration이 pooled 가중치의 ~78%를 먹기 때문에 갈림). **정본 goodput 술어(TTFT≤3s ∧ 요청-내부 token-ITL p95≤60ms)로 재채점하면 phase B는 5 arm×2 rep 전부 joint 0/192로 완전분리 — 오라클 자체가 미정의**(TTFT-pass 요청 중 최소 ITL-p95 106.1–106.3ms, ITL-p95-pass 요청 중 최소 TTFT 14.73/14.75s). ⚠️★**하네스 결함 발견(같은 재현) — `he2_bench.sbatch:92`가 아직 `dur=max(dur,d)`**(게이트 #7 버그 잔존, `sharegpt_vary_bench.sbatch:92`만 `dur+=d`로 수정됨) ⇒ **`he2_*.out`의 `HE2_RESULT` 라인은 정확히 약 3× 부풀려져 있다**(예: `he2_860498.out` d24 rep81 A `gp=8.125` vs 참값 2.717) — **`HE2_RESULT` 라인 인용 영구 금지**. 단 **정본 §1-19/§1-20 숫자 자체는 `sum(dur)` 기준 독립 재계산으로 무사함이 확인됐다**(공표 2자리까지 재현: A `[2.03,2.73,1.86,1.23]`, B `[0.28,1.01,1.09,0.74]`, bind combined 1.485). `n_indep`=2·overload-only(phase A offered 8/s vs achieved 4.39–5.52/s=1.45–1.82×, phase B offered 5/s vs 1.35–1.59/s=3.14–3.70×)·arm×node 부분교락(d34/bind rep82만 gpu42, 나머지 8/10 job은 gpu40) 불변. ★★**해소(2026-08-16, 커밋 `dadb851`)**: `he2_bench.sbatch:92`가 `dur+=d`로 수정됨(engine-porter 이관 완료) — 과거 `HE2_RESULT` 라인 인용 영구 금지는 **불변**(과거 로그 자체는 오염된 채 남음), 이 §1-19/§1-20 숫자 자체는 위 `sum(dur)` 독립 재계산으로 이미 무사함을 확인했으므로 재실행·재확인 불요. 상세 `../workspace/engine-port/results/slo_sched/ORACLE_REANALYSIS_2026-08-16.md` §2 |
| 18 | ★★**mix-스윙 트레이스서도 동적 패배 — 최적은 좁은 중간대만 스윙 (사용자 도전 검증, 2026-07-19)** | 기존 결론은 rate만 변하는 fixed-mix 트레이스 한정이었음. **mix-스윙**(phaseA prefill-heavy in2048/o32 ⇄ phaseB decode-heavy in256/o512, static sweep+bind, job 860452–470) 실측: **최적이 d24(A)↔d34(B)로 *좁게만* 스윙**(d16↔d44 아님). ★**prefill-heavy phase를 d16이 안 이김**(d24 5.006 > d16 3.016 > d44 1.564). 기전=**goodput=TTFT-SLO ∧ ITL-SLO가 반대로 당김**: d16 최고 TTFT(1.47s)·최악 ITL(56ms, 60벽 근접); d44 반대(ITL 22ms·TTFT 3.64s로 3s 실패); **중간 d24가 둘 다 충족→승**. **단일 중간 static d24가 양 phase 근최적**(A 5.006=최적, B 2.854 vs 최적 2.870=0.6%차)이라 **combined d24 3.930 ≫ bind 3.419**. ⚠️caveat: phaseB 포화(thru 2.9<offered 5)·n=2; **어떤 static도 양 phase서 두 SLO 동시충족 불가한 극단 mix는 미검증(동적의 남은 문)**. `mixswing.png` |
| 17 | ★★**동적이 지는 이유 = 오버헤드 아니라 *positioning* (실패 지점 규명, 2026-07-19)** | "오버헤드>이득"은 이미 반박(switch~0 §1-8, CPU 0.014% §1-12). 로그가 실패 지점을 정확히 보임: **최적=dec_sm 44(d44, throughput·goodput 양쪽 1위)인데 컨트롤러는 dec_sm 16–24(평균 22)서 진동하며 44에 절대 도달 못 함 = decode-STARVED**. 기전 = **reactive**(TPOT 스파이크 후에야 decode에 SM)+**symmetric**(두 slack 대등화)이라 decode가 잠깐 괜찮아지면 즉시 prefill로 회수 → 구조적으로 decode-heavy 최적에 누적 불가. 손실은 **switch 비용이 아니라 앉은 위치**. ★**risk/reward 18:1**: prefill-ward 이동의 LO 이득 ≤2.3%(§1-13 LO split-무관) vs HI 오배치 손실 ≤41.5% ⇒ 매 스위치가 나쁜 베팅. ★**모든 수정(anchor·비대칭 penalty·이동 중단)이 "44에 앉기"=static으로 수렴** — gate(ratchet, 34서 정지)가 best-dynamic이나 undershoot. **동적은 안 움직여 static과 *tie*가 상한, 이길 regime 없음**(§1-13). `why_dynamic_loses.png`. ★★**스코프 부착(doc-steward, 2026-08-16)**: 이 "상한·이길 regime 없음"이 확정하는 것은 **달성된**(observed) 정책 계열 — single-worker·SM-split·reactive 제어(HE0) — 뿐이다. **달성 가능한 천장**(어떤 lever로도 못 넘는 이론적 상한)은 다른 명제이며 별도로 확정된 바 없다(§5-8(a)(b)가 dual-worker·non-SM-split lever를 열린 항목으로 유지). 두 명제를 같은 문장으로 혼동하지 말 것(§3 항목57, `PRIZE_SIZE_ARGUMENT_2026-08-16.md` §3). ★**각주(2026-08-04, claims-auditor)**: 위 "모든 수정"의 **'모든'은 reactive 계열**이다. 비-reactive는 이미 시험됐다: Step D `PDMUX_SLO_LFF`(context-length feedforward, `multiplexing_mixin.py:825-832`) = **HD0, n=3(underpowered)**, Step F `sat_predict`(포화 예측 트리거)도 시험됨. ⇒ **온라인 feedforward는 시험돼 net win 아님(n=3, n≥4 재시험 미실행). offline 모델-프로파일 기반 decode-floor 예측(Claim E)만 미시험.** 본문 정정 불필요 |
| 16 | ★**정책 차이는 throughput이 아니라 SLO-attainment 효과 (2026-07-19)** | 같은 변화-trace 런을 **throughput(SLO-무관 req/s)**으로 재정렬: **스프레드 3.4%** (d44 3.776 > d34 3.759 > bind+GATE 3.745 > bind 3.700 > slo 3.696 > d24 3.677 > d16 3.654) vs **goodput 스프레드 13.1% (4×)**. ⇒ **모든 split이 GPU를 거의 동일하게 포화**시키고, split이 정하는 건 "몇 개 완료"가 아니라 "어느 요청이 TTFT 벽에 부딪히나"(SLO attainment 77.9–85.3%). 순위는 안 뒤집힘(d44 양쪽 1위). ★단 **d16이 양쪽 최하** — 얽힘이 raw throughput도 소량(3.4%) 깎음(decode 굶김→batch 정체→admission 차단→완료↓); d16 goodput 결손의 **~1/4는 실 throughput 손실·~3/4는 SLO attainment**. `throughput_vs_goodput.png` |
| 15 | ★**(D) granularity = *실행시* 비용이지 *결정시* 비용 아님 (2026-07-18)** | per-layer-type SM 분할을 **offline predictor/floor로 고정해도 死**. (D)는 "누가 split을 정하나(런타임 vs offline)"가 아니라 "한 forward *안에서* 파티션이 layer 경계마다 바뀌나"의 문제 — offline 고정값이라도 실행 시 **attn↔mamba 경계마다 green-ctx 재분할 필요**(step 파편화·sync 직렬화·overlap 감소, S2서 **TPOT 42→124ms**). offline은 **결정 오버헤드만** 제거·**실행 파편화 비용은 그대로**. ⇒ **살아있는 offline 역할은 오직 whole-phase floor**(step 내내 단일 파티션, composition으로 크기만 결정 = §1-5). ★**비용 분해(2026-07-18 확인)**: green-ctx는 시작 시 `initialize_stream_groups`로 **전부 pre-created**(스위치=인덱싱; 생성비용 없음). 실제 스위치 비용 = 경계마다 `stream.synchronize()` **드레인**. 이를 GPU측 wait_stream 순서화로 교체(`PDMUX_LA_COORD_OPT`)하면 **124→85ms(갭 ~47% 회수)**나 **여전히 패배**(agnostic 42ms 평탄). **잔차 = 구조적 오버랩 손실**(monolithic prefill이 window 0만 오버랩, 윈도우수 무관·모델 독립) + **cudagraph 비양립**(step 중간 green-ctx 전환 캡처 불가→eager 강제→운영점 진입 불가). ⇒ **"싼 전환"으론 절반만 없앰; 나머지 절반은 pre-created로도 불가.** `results/a_substrate/` |

| 16 | ★**HE0는 goodput SLO 엄격도에 의존 — tight SLO에선 동적이 best-static과 대등~약우위 (2026-07-18)** | **기존 벤치 재분석**(job 재제출 없음, per-request `input_lens`/`ttfts`/`itls` 재스코어; `results/slo_sched/lengthnorm_slo_reanalysis.md`, `reanalyze_lengthnorm_slo.py`). **sanity**: fixed-3s 재현이 §1-7과 정확 일치(d44 3.220±0.013). ★**fixed-tight sweep**(길이 무관, 순수 엄격도): 승자 = **d44@{3.0,2.0,1.5,1.0,0.75s} → d34@0.5s → bind+GATE@0.335s**, 교체 임계 **TTFT 0.5–0.75s**(=실 prefill mean~112ms의 3–5×). ★**축은 길이-비례성 아니라 엄격도**: 같은 ~335ms 평균예산서 flat SLO(bind 2.593) ≈ 길이비례 SLO(bind 2.582) = 둘 다 동적 승(길이비례 여부 무영향). **n≥4 baseline**(d24/d16/slo, job 859005–859059)서도 norm-k(2/3/4) 전부 bind+GATE ①. **강도(보수적)**: bind+GATE vs **d44 +0.073(~3σ, 유의)** / vs best-static **d34 +0.042(~1.5σ, 대등)**, floor변형선 d34≈bind 무승부 ⇒ **"동적이 압도"가 아니라 "동적이 best-static과 대등~약우위, decode-heavy static 지배는 반증"**. **기전**(phase 분해): 반전은 HI(과부하) phase에서만 — 관대SLO=완료율 지배(decode throughput=decode-heavy 승) / tight SLO=first-token 반응성 지배(부하 중 prefill 저글링하는 동적 승, decode-heavy static은 prefill 굶겨 꼴찌권). **한계**: Zamba2 short-ctx·ShareGPT p99 2776tok 한정, 재분석은 3s 벤치 데이터 재스코어(인터랙티브 TTFT를 직접 attain 측정한 건 아님). ★**실무 관행 조사로 지위 강화 (2026-07-18, `serving_slo_survey.md`)**: 프로덕션 인터랙티브 TTFT P99 = **chat 300ms·voice 150ms·code 100ms·RAG 400ms** = **전부 tight regime(동적 승)**; 우리 정본 3s는 표에서 **"batch async" 행**에 정확 대응 ⇒ **"static 지배"는 배치 서빙 한정, 인터랙티브 주류는 동적 regime**. SLO를 배수로 sweep(DistServe "SLO scale")은 표준 방법론이고 "엄격할수록 구조/반응성 이점이 드러남"도 알려진 패턴(DistServe: strict→disaggregation). ⚠️★**이 "동적 우위"는 §1-17(직접 재튜닝 측정)에서 아티팩트로 반증됨 — 재스코어는 컨트롤러 *행동*을 못 봤다** |
| 17 | ★★**§1-16 반증 — tight SLO로 컨트롤러를 *실제 재튜닝*하면 동적은 best-static에 크게 열위 (2026-07-19)** | §1-16은 3s-튜닝 컨트롤러의 궤적을 tight SLO로 *사후 재스코어*(행동 불변)한 것. 이번엔 **컨트롤러 SLO를 chat(TTFT 300/ITL 50ms)로 실제 설정**해 직접 서빙(`interactive_bench.sbatch`, jobs 860415–860514). **용량 = rate 7–8**(rate≤6 무관심·≥10 전붕괴), 판정은 경계 **rate 8, n=4 attainment%**: **d44 73.2±4.8 ≫ d34 49.6±3.9 > bind+GATE 44.3±2.7 > bind 40.6±0.4**. ★**d44 vs bind+GATE = 28.9%p ≈ 10σ**. static 단조(decode SM↑=attain↑: d16 33<d24 41<d34 50<d44 73), **동적은 2위 static(d34)도 못 넘음**. **기전**: 컨트롤러가 tight TTFT에 반응해 prefill-ward 이동(switch 24–30)→decode 굶김→§1-4 얽힘 트랩→batch 정체→TTFT 악화(bind TTFT p90 1.6s vs d44 0.44s). §1-16이 상상한 "tight→prefill 반응성 유리"가 실제론 **역효과**. bind+GATE>bind는 게이트가 trap 억제(§1-11 재확인, rep2 feas=3서 게이트 미발동→bind급 하락=반증실험). ★**결론: decode-heavy static이 관대 SLO(§1-7)뿐 아니라 tight SLO에서도 지배, 오히려 격차 더 큼(§1-4·§1-6이 tight서 더 극명). §1-16의 조건부화는 취소 — SLO 엄격도와 무관하게 static 지배.** code(100/25)는 무경쟁 66%로 HT-neg(물리 불가). 상세 [interactive_slo_retune_plan.md](interactive_slo_retune_plan.md) §9 |

**실전 권고**: **peak decode 부하 기준 decode-heavy static split 고정**(이 워크로드선 d44급). 동적 불요 — **관대(3s)·tight(chat 300ms) SLO 양쪽에서 확정**(§1-7·§1-17).
**게이트를 굳이 쓴다면**: 수동 튜닝 없이 안전한 static을 자동으로 찾는 **auto-tuner**로서만 값어치(최적에 미달; tight SLO선 trap 억제로 bind보다 낫지만 여전히 static 미달).

---

## 2. ★철회·불확실 (2026-07-17 분산 측정으로 무너진 것)

| # | 이전 주장 | 현재 상태 |
|---|---|---|
| 1 | "stationary bimodal(6.24↔2.22)은 **양성피드백 트랩** 때문" | ★**stationary 벤치 한정 과잉 귀속 — 철회 유지**(static d24도 switch=0인데 6.32↔3.10 붕괴 ⇒ 거기선 노이즈와 분리 불가). ★**그러나 트랩 자체는 2026-07-17 부분 복권**: **유효 벤치(변화 trace)** 에서 **d44가 ±0.013 = 노이즈 없음이 증명된 조건**인데도 **no-gate만 1/4 붕괴(TRUE 2.405 vs 정상 3.10–3.12), gate는 0/9** ⇒ 거기서의 붕괴는 **컨트롤러 탓이 맞다**(§1-11). **정정된 주장**: "트랩은 실재하고 게이트가 막는다 — 단 stationary 벤치의 bimodal은 그 증거가 못 된다" |
| 2 | "d24-static은 ±0.039로 안정" | **n=2의 운.** 실제 **5.282 ± 1.302 (n=4, min 3.102)** |
| 3 | "feasibility 게이트가 트랩을 없애 **성능 회복**" | ★**2026-07-17 유효 벤치서 분해 — 절반 확정·절반 반증.** **견고성은 확정**(no-gate 2.934±0.306·1/4 붕괴 → gate 3.132±0.019 (n=9)·0/9, 16× 타이트 = §1-11). **성능 회복은 반증**(gate 3.132 < d34 3.171 < **d44 3.220**; §1-10 = ratchet이 틀린 static에 조기 정지). ⇒ "**트랩은 없애나 성능은 여전히 static 미달**" (구 stationary 수치 5.928/5.282/5.269는 노이즈 교란이라 폐기) |
| 4 | 최근 n=1~3 정책 비교 다수 | **underpowered** — 베이스라인 ±1.3이 정책 차이를 삼킴. 재측정 없이 인용 금지 |
| 6 | (내 가설) "stationary 분산 = **GPU 클럭/전력 throttling**" | ★**철회 (2026-07-17)** — 불필요. 설명 대상은 TTFT 3.5×가 아니라 **throughput 3%**였고, 증폭기는 **SLO 임계 절벽**이었다(§1-14). 잔여 3%(co-tenant/클럭/페이지캐시)는 상존·무해 |
| 5 | 초기 SLO track "isolation 오버헤드 0"(2.318≡2.319) | **주의 플래그** — 당시도 n이 작았다면 같은 함정. 재확인 전까지 약한 근거로 취급 |

---

## 3. 방법론 (교훈 — 앞으로 필수)

1. ★**stationary ShareGPT r8 = 정책 비교 벤치로 부적합·폐기.** 동일 프롬프트(`--seed` 고정)·switch 0인 static조차 **±1.302** → 신호를 삼킴.
2. ★**변화 trace(rate 3↔12, 3라운드 평균) = 유효 벤치** (±0.02). **정책 비교는 이걸로.**
3. **베이스라인 분산을 먼저 측정**하고 시작. **n≥4** 없이 정책 결론 금지.
4. **dynamic 결과엔 항상 `switch_count` + split 체류분포 + TTFT/ITL p50/p95/p99 병기.**
5. **Switch decomposition**: `Net = Σ(B positioning) − (A switch × drain)`. (A)와 (B)를 분리 귀속.
6. ★**메트릭이 임계 지시함수(goodput=TTFT≤SLO)면 절벽을 피해 측정할 것** — 용량을 먼저 재고, TTFT 평탄역이 SLO 임계에 걸치는 rate는 **3% 섭동을 2× 신호로 증폭**한다. 과부하 구간의 threshold-goodput은 **런 길이 의존 = ill-posed**.
7. ★**여러 라운드를 한 파일에 append하는 하네스는 분모를 반드시 합산**(`dur+=d`; `max()`는 라운드 수만큼 부풀림 — `f921ae8`서 3× 버그로 실현).
8. **변수는 하나씩** (pf_urg와 dwell 동시 변경 → 해석 불가였던 전례).
9. ★**coupled 스윕은 반대편 축을 공변시켜 confound되기 쉽다**(2026-07-26,
   Stage 0). decode-SM을 낮추며 동시에 prefill-SM을 높이는(또는 그 역) 하네스는
   관측 곡선이 스윕 축 자체의 효과인지 반대쪽에서 늘어난 경합/de-batch의 효과인지
   구분 못 한다. 이 결론 자체는 유효하다(§1-21 판정1, 생존). ★★★**재작성
   (2026-07-28, claims-auditor 감사 후) — 아래 문장은 정정이 아니라 대체다.**
   원래 여기 있던 "음성 대조 + 무경합 앵커의 조합이 confound를 identify했다"는
   서술은 **거짓이었다** — Stage 0에서 그 두 기구는 **둘 다 고장 나 있었다**:
   "무경합 앵커"(D108)는 실제로는 decode 16 SM이었고(코드 버그로 legacy
   auto-path가 항상 (92,16) 선택), "음성 대조"(pure-Mamba M)의 전제 자체가
   틀렸다("decode는 O(1) recurrent라 SM-bound 불가"의 O(1)은 context 길이에
   대한 것이지 SM 수에 대한 것이 아니었다 — 8B 재측정에서 M도 T·H와 동일 밴드로
   SM-민감했다). 이 확산에서 얻는 교훈 3개로 대체한다:
   - **pin은 policy target이 아니라 realized 파티션으로 검증한다**
     (`runtime_snapshot`의 `(prefill_sms, decode_sms)`, `dual_worker.py:608`).
     비용 0. Stage 0은 controller가 지정한 값(target)만 확인하고 실제로
     선택된 stream_index(realized)를 확인하지 않아 D108이 D16이었음을
     놓쳤다.
   - **음성대조는 그 축에 binding 불가능함이 독립 입증된 뒤에만 음성대조다.**
     "Mamba decode는 O(1)"이라는 직관을 검증 없이 음성 대조의 자격으로 썼다가,
     그 O(1)이 다른 축(context 길이)에 대한 것이었음이 드러나며 대조 자체가
     무효화됐다.
   - **"policy OFF = 중립 기준선"은 legacy fallback 경로 때문에 조용히
     깨진다.** `PDMUX_R2_POLICY`가 unset이면 legacy `adjust_stream_groups`가
     `manual_divisions`의 threshold 필드(0)를 조건문으로 오독해 의도한 값과
     다른 파티션을 고른다 — "정책을 껐다"가 "분할을 안 했다"를 뜻하지 않는다.
   상세 [`stage0_verdict_2026-07-26.md`](stage0_verdict_2026-07-26.md) §5(원
   교훈, 철회됨), `../workspace/engine-port/results/s0_deconfound/
   PARTITION_RESIDENCY_STAGE0.md`(재작성 근거).
10. ★★**(2026-07-28) pin은 realized로 검증한다.** 위 항목 9의 첫 소항목과 동일 —
    `PROJECT_STATUS.md` "방법론 게이트(신규)" 참조. ★**아래 13번의 특수
    사례**(target-vs-realized 집계 단위 불일치)로 재분류(2026-07-29).
11. ★★**(2026-07-28) 파티션 활성률을 사전등록 게이트로 삼는다.** green-context
    분할은 split-prefill 동거 중에만 유효하고, 비면 무분할로 되돌아간다 —
    최소 활성률(예: ≥0.60)을 실험 전에 정하고 미달 셀은 폐기한다. ★**정정
    필요(2026-07-29)**: 이 활성률은 **시간가중**으로 재정의해야 한다 — 아래
    13번 참조. 스냅샷 **개수** 기반 활성률은 실제값을 최대 16–26× 과소평가할
    수 있다(`results/s8_frontier/` 실측: 동시성 0.015 → 시간가중 0.25–0.40).
12. ★★**(2026-07-28) keepalive는 짧은 prefill을 자주.** 긴 keepalive는 활성률을
    오히려 떨어뜨린다(실측 0.66–0.93 → 0.32–0.63) — 긴 prefill 윈도우 동안
    decode가 진행되지 않아 시간적으로 분리된다.
13. ★★★**(2026-07-29) 집계 단위를 먼저 정하고, 그 단위가 추정 대상과 맞는지
    논증하라.** 위 9번(coupled 스윕 confound)·10번(pin은 realized로 검증)의
    상위 개념 — 그 둘을 이 원칙의 특수 사례로 흡수하되 문구는 지우지 않는다.
    `results/s8_frontier/`(E1 프론티어) 하네스 구축 중 **집계 단위가 답을
    5번 바꿨고 전부 반대 결론을 낼 뻔했다**:
    1. **target vs realized 파티션**(Stage 0 무효화, 위 §1-21).
    2. **게이트 모집단** `prefill_active OR decode_active` vs 조건부 — 설계상
       정상인 decode-only 무분할 윈도우를 pin 실패로 셈: pin **0.029 FAIL →
       0.976 PASS**.
    3. **스냅샷 개수 가중 vs 시간 가중** — `runtime_snapshot`이 이벤트루프
       iteration당 발화해 235ms prefill 스텝과 11ms decode 스텝이 같은 무게를
       가짐: 동시성 **0.015 → 0.25–0.40**(16–26×).
    4. **drain 꼬리 포함 duration vs 도착 구간만의 duration** —
       `achieved_rps` **3.40 → arrival_rps 8.96**.
    5. **스냅샷 vs 에피소드**, 그리고 그 안에서 다시 **개수 vs 시간** —
       d16 pin_frac **0.583 → 0.847**.

    **따름정리**: 하나의 추정량으로 두 질문에 답하지 마라. "그 파티션에서
    실행됐는가"(게이트)는 **시간 가중**으로 답하고, "이 요청의 지연은 어느
    파티션 것인가"(귀속)는 **요청별 bracket**으로 답한다. 후자는 전자의
    데이터를 대부분 버리므로(대부분의 스냅샷이 요청 경계 안쪽이 아니라
    사이에 놓임) **게이트에 쓰면 검정력이 무너진다**(예: d16 요청-bracket
    귀속 n_episodes=8, lower95=0.554 → FAIL — 사유는 검정력 부족이지 잘못된
    SM 아님). 상세 `../workspace/engine-port/results/s8_frontier/` 하네스
    설계 문서(결함 5종 수정 기록).
14. ★★★**(2026-08-02) "이 양이 내가 재려는 것과 논리적으로 독립인가"를 먼저
    물어라 — 항등식을 증거로 쓰지 마라.** 13번(집계 단위 선확정)과 같은
    뿌리이나 실패 모드가 다르므로 별도 항목으로 둔다: 13번은 *같은 양을 어떻게
    세는가*의 문제이고, 이것은 *그 양이 애초에 답을 담고 있는가*의 문제다.
    2026-07-31~08-01 세션에서 철회된 주장 8건 중 **3건이 정의·항등식을 증거로
    착각**한 것이었다:
    1. `arrival_rps`는 seed로부터 RNG replay로 **재생성된** 값
       (`e1_capacity_scan.sbatch:200-207`)이라 `(seed, n)`만의 결정론적
       함수 — "5셀 전부 동일"은 **항등식**이고 서버에 대한 정보량이 0이다.
    2. `kv_mamba_occupancy = 1.0`은 pool 크기가 `--max-running-requests`와
       같아서 생기는 **항등식**(§1-23).
    3. "ITL 구속 rate ⟂ d92 off-cliff"는 ITL 구속 rate가 **공집합**이라
       **공허참**(T8은 전 구간·전 룽에서 요청의 ≥93%가 ITL 통과).
    나머지 3건은 **한 arm/셀에서 잰 것을 일반화**한 것이었다(batch-cap은 T8
    2셀만 측정). ⇒ 새 지표를 증거로 올리기 전에 **그 지표가 실험 설정으로부터
    해석적으로 결정되는 값이 아닌지** 먼저 확인한다. `../PROJECT_STATUS.md`
    "방법론 게이트" #6과 동일 항목.
15. ★★★**(2026-08-03) 항등식에서 파생된 양을 자유 모수처럼 나누지 마라 —
    14번과 같은 뿌리, 이번엔 GPU를 쓰기 전에 잡혔다.** §1-26(A). `E1_DECODE_
    REALIZED`(§1-25)는 pin 게이트와 마찬가지로 항등식 경계(prefill in-flight
    ⟺ decode busy)에서 파생된 시간 몫이다. 이걸 "decode-SM 레버가 A(D)에
    engagement 비율 `w`만큼만 반영된다"는 **자유 모수**로 취급해 `A_free(dD)
    =w·A(D)+(1−w)·A(108)`을 역산하면, `w`가 실은 §1-24(prefill=108−D)가
    이미 결정한 duty cycle이라 **역산 대상과 역산 도구가 같은 양**이 되고
    보정치는 자기충족적이다. 검정 순서가 재사용 가치: (i) 같은 보정을
    **대조군(레버가 있다고 이미 알려진 arm)에 적용**해 알려진 값을 위반하는지
    본다(control-arm reductio) — 위반하면 모형 사망; (ii) 모형이 요구하는
    잠재 관측치(여기선 split-조건부 ITL)를 **직접 계산**해 실측과 대조한다;
    (iii) 모형의 핵심 가정을 **부분집합 분해**로 직접 검정한다(여기선
    UNSPLIT-only에서 헤드라인이 그대로 재현 = "A(108) 셀 무관" 가정의 반례).
    셋 다 GPU 데이터 없이 기존 텔레메트리 재사용만으로 수행됐다.
16. ★★**(2026-08-03, 같은 날 3차 속행) 진단과 처방을 같은 턴에 하면 처방은
    자기가 감사한 것이다 — 두 번째 실증.** §1-27에서 claims-auditor가
    `A_free` 결함(진단, [AUDITED])을 낸 뒤 같은 회부에서 대체 추정량 설계
    (처방, [UNAUDITED])까지 했던 것과 **같은 실패 모드가 이번엔 반대
    방향에서 재현됐다**: result-analyst가 §1-28의 p95 음성대조 오염을
    **발견**(진단)한 뒤 primary를 p50으로 **바꾸자는 처방**까지 같은
    분석에서 냈고, claims-auditor가 그 처방만 세 겹으로 반증했다(관측은
    CONFIRMED, 처방은 REFUTED — §1-28 주장 2). **규율: 진단자와 처방자를
    분리하라.** ★**같은 회차에 성공 사례도 하나 나왔다**: 경합 가설이
    공유하는 전제를 검정 대상에 명시적으로 넣는 규율(항목 10과 같은
    뿌리)이 두 번째로 작동 — 회부서에 "축이 같다"는 전제를 명시한 덕에
    §0의 2.6× 모순이 GPU를 쓰기 전에 드러났고, 감사자가 "p95 오염 vs
    정상"의 공유 전제(*"SPLIT 집단이 decode-SM 대비를 담고 있다"*) 자체를
    찾아 반증했다.
17. ★★**(2026-08-03, 같은 날 3차 속행) 끝점 선택이 임계를 정한다 —
    "격자 실측값이라 임의 상수 없음"은 방어가 되지 않는다.** §1-28 주장
    4(`G_LEVER=1.41`)에서: 후보 임계가 C2 측정 격자{16,24,44,92}의 실측값
    중 하나라는 것만으로는 자유 모수가 아니라고 주장할 수 없다 — **끝점만
    골라도 [1.41, 2.40] 전 구간에 도달 가능**하기 때문에, "어느 두 점을
    비교점으로 쓸지"가 그 자체로 숨은 자유도다. §4.3.10(3)이 이미 다른
    맥락(blocking 임계 스윕)에서 거부한 것과 같은 종류의 논거가 문턱값
    선택에서 재등장한 사례로 기록한다.
18. ★★★**(2026-08-03, 같은 날 4차 속행) 자기가 검증하려는 코드를 복사한
    게이트는 항등식에 가깝다 — 대조는 생산자에 걸어라.** `s0_axis_check.py`의
    gate 1은 `m3_conditional.label_probe`의 라벨링 루프를 복사해 같은
    입력에 대조했다 — 정작 새로 넣은 값(`wmean`, 평균 batch 필드)은 무엇과도
    대조되지 않았다. gate 2는 28.79ms를 만든 바로 그 함수(`c2_anchor.collect`)를
    호출해 그 값을 "검증"했다 — 구조적으로 순환이다. `S0R_REPLICATION_
    2026-08-03.md`의 gate G-B가 대안을 보여준다: 같은 코드 경로를 공유하는
    다른 분석 스크립트가 아니라 **생산자 자체**(`s0dc_client`의 자기 기록
    per-rep summary)에 대조해 20/20 정확히 일치시켰다. ⇒ 게이트를 쓸 때는
    "이 게이트가 재검증 대상 코드를 그대로 복사했는가"를 먼저 묻는다.
    ★**새 사례(2026-08-11, Gate 2-S 첫 유효 결과, claims-auditor)**:
    `PREREG_GATE2S_2026-08-09.md` §4.5의 두 "독립" 안전장치가 실제로는
    같은 3개 꼬리 통계량만 보고 있었고, 그중 `CONVENTION-SENSITIVE`
    판정은 12개 지표 중 9개에서 **비트 동일**(항등식) 산출이었다 —
    형식상 두 개의 게이트지만 정보량은 하나에 가깝다. 상세 §1-1
    (2026-08-11 Gate 2-S 블록).
    ★**별건 사례(2026-08-11, E1 addendum, claims-auditor — E1 자신의
    책임 아님)**: `g2s_analyze.py:640-660`의 `falsifier()` docstring이
    "Rules IMPORTED from gate1b_analyze.py"라 적었으나 실제로는 pop-A
    분류 규칙을 **인라인 복사**했고, 생산자(`gate1b_analyze.py`)가
    적용하는 `phase=="benchmark"` 필터를 **적용하지 않는다**. 이
    캠페인은 telemetry 100%가 benchmark phase라 수치 결과에는 영향이
    없었으나(우연), 주석이 실제 코드 계통과 다르다 — engine-porter
    이관 대기(코드 미수정).
    ★★★**일곱 번째 재발(2026-08-11, 트래픽·roofline 진단, result-analyst
    자수) — 게이트가 아니라 "확증 서술"이 항등식이었다.** `s8_scaleup`
    decode-SM 스윕의 스텝 트래픽(`bytes_step`)은 SM 파티션과 무관하게
    config·체크포인트만으로 정의되므로 achieved_BW = bytes_step/ITL은
    **정의상** `|ε_BW| ≡ |ε_ITL|`이다. roofline 역산의 국소 탄력도
    (16→24 0.83–0.90, 44→92 0.16–0.41)가 C2의 기존 ITL 탄력도(16→24
    0.77–0.88, 44→92 0.09–0.35)와 "정합"한 것은 **독립 확증이 아니라
    항등식** — roofline이 실제로 더한 정보는 **절대 수준**(achieved_BW가
    사양 대역폭의 48–61%)뿐이다. 6번째 재발까지는 전부 사후 감사가
    잡았으나, 이번엔 진단을 산출한 result-analyst 자신이 실행 중
    문서 안에서 자수했다(`TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md`
    §6.4). 상세 `../PROJECT_STATUS.md` "8B decode-SM 민감도 측정 노트"
    C-4·"방법론 게이트" #9(일곱 번째 재발).
    ★★★**아홉 번째 재발(2026-08-16, C2-R 캠페인, claims-auditor 적대
    감사) — 이번엔 "양성대조" 자체가 항등식이었다.** `s8_c2r_score.py`의
    `cmd_poscontrol`(:270)이 대조에 쓴 코드 경로(`discover("legacy",...)`
    → `score(...)`의 기본값 `keep_slack=False`)는 헤드라인이 실제로
    쓰는 경로(`cmd_c2r`:333, `discover("c2r",...)` →
    `score(..., keep_slack=True)`)와 **다르다** — 대조는 헤드라인
    경로를 **한 번도 실행하지 않았다.** 게다가 대조의 표적값(T8
    2.388·Hs8 2.687) 자체가 `s8_batch_matched.py` 루프의 또 다른
    verbatim 복사본(`audit_c2_job_composition.py`)의 산출물이다 —
    "독립 검증"이 세 번째 복사본이 첫 번째 복사본과 일치하는지를 잰
    것에 가깝다. 배제된 오류는 전사(transcription)·t0 조인·pooling
    로직뿐이며, 실제로 남은 공백(헤드라인 경로 자체의 검증)을 메운
    것은 이 대조가 아니라 **claims-auditor의 독립 재구현**(해당 코드
    미import, 새 스크립트로 3.057990/3.114061 및 24개 부팅의 n·median
    전부 재현)이다. 상세 §3 항목56(C).
19. ★★★**(2026-08-03, 같은 날 4차 속행) 여집합 클래스에 음성대조를 걸어라 —
    항목 9와 뿌리는 같고 방향은 반대.** 이 자료를 세 차례(원 C2→`G_LEVER`
    감사, 첫 §0 axis check, 이 세션 자신의 첫 프레이밍) 통과했지만 아무도
    "같은 mode estimator를 UNSPLIT(여집합)에도 적용해본다"는 한 줄을 하지
    않았다 — `PREREG_S0R_MODE_2026-08-03.md`의 행 4가 그것을 했고, SPLIT의
    "특이적" 슬로우 모드가 실은 UNSPLIT에도 같은 위치·크기로 존재함을
    드러내 §1-28 §0의 강한 재프레이밍을 죽였다. 항목 9(게이트가 여집합을
    세는 바람에 *실수로* 실패)와 같은 뿌리이나 방향이 반대다: 이번엔
    여집합을 **일부러** 재서 라벨의 배타성을 검정했고, 그 검정이
    **성공**했다(세 차례의 앞선 통과가 놓친 것을 잡음). 상세 §1-30,
    `DESIGN.md` §4.3.15(c)–(d).
20. ★★**(2026-08-04, claims-auditor+result-analyst X2′) 세 가지 층위 구분을
    명시적으로 등재한다 — 이전엔 문장으로 존재하지 않았다(grep 확인).**
    (a) **절대 민감도 ≠ 상대 민감도** — 동시에 참일 수 있다(예: C2의 절대
    ITL 개선과 §1-13의 상대 goodput spread는 서로 다른 양이며 서로를
    반증하지 않는다). (b) **천장 절단(ceiling censoring)** — 지표가 도착률
    등 외적 상한에 잘려 정책 간 차이를 볼 수 없는 상태. §1-13 원자료: rate
    3(LO)에서 199/200·200/200·200/200 통과, goodput 2.86 ≈ 도착률 3의
    95%. *"저부하 phase의 goodput은 도착률에 절단돼 있어 정책을 구분할 수
    없다. 따라서 §1-13의 'LO는 split에 무관심'은 지표의 무신호이지 레버의
    부재가 아니다. LO에서 레버를 보려면 goodput이 아니라 ITL/TTFT 분포를
    직접 봐야 한다."* 초판이 이를 "민감도≠구속"이라 불렀으나 **이름이
    부정확**했다 — 실제 기전은 지표 절단이지 물리적 non-binding이 아니다.
    (c) **baseline이 다르면 % 비교가 성립하지 않는다** — "PD 분리 이득"
    (§1-1)은 **fused 대비**이고 "+2% 천장"(§1-20 coupled ceiling)은
    **best-static 대비**다, 같은 축 위의 숫자처럼 직접 비교하지 말 것.
    상세 [layertype_dynamic_JUNCTION_2026-08-04.md](layertype_dynamic_JUNCTION_2026-08-04.md).
21. ★★**(2026-08-04, claims-auditor) C2 탄력도를 다른 격자로 이전하지
    마라 — 규율의 재실증.** "C2 국소 탄력도가 벡터1(g2_0_raconf)의
    d44→d54 전이를 예측한다"는 주장을 감사가 REFUTED: 벡터1의 d44→d54는
    C2의 44→92 구간 안이고 그 구간 국소 ε=0.09–0.35이므로 예측
    +1.9~7.4%인데 실측은 15.0% = **2–8× 빗나감**(끝점 풀링 ε을 동작점
    밖에서 끌어온 산물 = 항목17의 재발). 추가로 (i) estimand
    불일치(벡터1=요청별 token-ITL p95의 중앙값, C2=파티션×batch 매칭
    per-token ITL p50), (ii) 변수 동시 변경(`d44=[64,44]`→`d54=[54,54]`는
    decode+10 AND prefill−10의 합성), (iii) §0(2.6× 미해소, §1-28/§1-30)
    격자 간 이전 금지 규율 위반. ⇒ **"C2를 다른 격자로 이전하지 마라"는
    이미 §1-28/§1-30이 세운 규율이며, 이번 시도는 그 규율이 왜 있는지를
    한 번 더 보여준 사례일 뿐 새 발견이 아니다.**
22. ★★**(2026-08-04, claims-auditor+result-analyst X2′) 재사용 인용 금지
    목록(끝점/최량-rep 선택 + 집계 단위 미정 재발).** (i) `research_arc.md`
    S9(§1-5 서사)의 **"d16 1.056 vs d24 6.240 (5.9×)"**는 폐기 벤치
    (stationary r8, 방법론 게이트 #2)의 n=4 중 **최댓값**이다 — 정본 §1-5는
    **5.282±1.302**(n=4)를 쓴다. 끝점/최량 rep 선택(항목17) + 폐기 벤치
    사용(게이트 #2)의 **이중 위반**이므로 "6.240"·"5.9×"는 인용 금지,
    "5.282±1.302"만 인용한다(원문은 역사 기록으로 보존, 삭제하지 않음).
    (ii) `diffA_vs_diffB_table.md`의 **집계 Diff A(≈12k–18k 교차점)·
    "mamba 1.28× 지배" 등 집계-조성 서술은 인용 금지** — X2′ 판정:
    미계측 GEMM(U_H) 효율 가정 하나가 교차점을 3k↔30k로 움직인다
    (@23.9 TFLOP/s면 11,792 / @150이면 3,052 / MLP 귀속 제외 시 29,576)
    ⇒ 판정 불가, "집계 교차점" 수치는 측정이 아니라 **명명 선택의
    결과**다. `s_G ≈ s_M`도 자기산출·여집합 음성대조 없음 ⇒ 공허참
    가능성 배제 못 함, 인용 금지. 상세
    [layertype_dynamic_POSITIVE_2026-08-04.md](layertype_dynamic_POSITIVE_2026-08-04.md) §5·
    [layertype_dynamic_JUNCTION_2026-08-04.md](layertype_dynamic_JUNCTION_2026-08-04.md).
23. ★★★**(2026-08-05, claims-auditor, S2 재현) 사전등록 게이트가 전부
    통과했다는 것이 "결과가 제약됐다"를 뜻하지 않는다 — 방법론 게이트 #9의
    네 번째 재발.** S2(job 873015)는 `AMBIG_FRAC`·`MIN_N_SPLIT`·
    `PREREG_S2` §3.1 일치검사가 16/16 cell-block 전부 PASS했지만, 세
    게이트 모두 `E1_DECODE_REALIZED≥0.90` 통과라는 **같은 전제조건 아래서
    항등식**이다(그 전제가 성립하면 SPLIT이 이미 인구의 100.000%/99.969%가
    되므로 `p50(SPLIT)=p50(all)` 검사는 실패할 수 없다). `E1_DECODE_
    REALIZED` 자신도 arm 간(OFF vs ON)에는 정보를 담지만, **ON arm 안에서는
    `stream_idx=_sticky_fixed_idx`라는 코드 불변식**이라 게이트로서
    기능하지 않는다. ⇒ 이 런에는 **"28.92가 나올지 11이 나올지"를 사전에
    제약한 게이트가 0개였다** — 관측치의 타당성과는 별개로, "사전등록
    게이트를 전부 통과했다"는 문장을 "결과가 게이트에 의해 제약됐다"로
    읽지 않는다. 세 번째 재발은 `S2_ANALYSIS_2026-08-04.md` §3이 §3.1
    하나에 대해 이미 기록했다("third recurrence of methodology gate #9
    in this campaign line") — 이번은 그보다 넓은 형태로, 런 전체의 결과
    게이트 집합이 구조적으로 공집합임을 확인한 것이다. 상세 §1-31,
    `../workspace/engine-port/results/s2_sticky/S2_REPLICATION_
    2026-08-05.md` §7, `results/s8_frontier/DESIGN.md` §4.3.16.
24. ★★**(2026-08-05, ceiling-censoring 진단 + claims-auditor 감사)
    `frac(reqITLp95>T)`은 SLO 임계 T 근처에서 LO 판정에 쓰지 마라 —
    방법론 게이트 #6(metric cliff)의 새 사례이자 집계 단위 교훈(항목13)의
    6번째 재발.** LO(rate 3)에서 `frac(reqITLp95>60)`은 **2400 요청 중
    9건 대 9건**이고(상대효과 CI [−55.6%, +55.6%] — T=20에서 측정된
    −49.8%를 포함하므로 **검정력 0**), T=57 −38.9% → T=59 +88.2% →
    T=61 −44.4%로 **3ms 창 안에서 부호가 널뛴다**(CDF 수직 지점). 게다가
    그 9건의 7–8건이 **각 job 첫 라운드의 launch index ≤27** 요청이며,
    `middle60` slice에선 **0건**이다. ⇒ 같은 임계 근처에서 부호가
    안정적이지 않은 초과질량 지표는, 몇 건 안 되는 이벤트가 부호를 정하고
    있다는 뜻이므로 헤드라인으로 쓰지 않는다 — §1-13에서는 요청별 ITL
    p95의 p90/중앙값과 TTFT p50(§1-13 각주, 3.2–5.6 SD)만 인용한다.
25. ★★**(2026-08-05, ceiling-censoring 진단 + claims-auditor 감사)
    C2를 다른 격자로 이전하지 마라 — 항목21의 3회차 재발.** 진단서
    (`CEILING_CENSORING_DIAG_2026-08-05.md` §4 "T4 참고 대조")의 C2 정성
    대조("HI는 방향·자릿수 일치, LO는 방향 반대"로 caveat을 우회하려던
    시도) 전체가 **REFUTED**다 — `PROJECT_STATUS.md`(8B decode-SM 민감도
    절, "44 이상 구간에 이 비를 적용하지 말 것"·"C2를 다른 격자로 이전하지
    말 것")와 이 문서 항목21의 **정면 위반**이며, 주장된 1.9–2.2×는
    ε≈0.63–0.78을 요구하는데 정본 국소 탄력도 ε는 0.48–0.56이라
    (44/16)^ε = **1.62–1.75**다. 항목21(2026-08-04, 벡터1 d44→d54 전이
    예측 시도)에 이은 **3회차 재발**로 기록한다 — "C2를 다른 격자로 이전
    하지 마라"는 규율이 세 번째로 그 필요성을 보여준 사례일 뿐 새 발견이
    아니다. 상세 위 §1-13 각주.
26. ★★★**(2026-08-05, α 밴드 부검) 예측 밴드도 이식 금지 규칙의 적용
    대상이다. 밴드는 가장 가까운 기판(같은 trace·같은 batch·같은
    client)에서 뽑아라. 자기가 금지한 이전을 자기 사전등록 예측에 쓰면,
    실험이 성공해도 규칙은 실패한다.** §1-31이 C2→sticky 이식을 명시적으로
    금지해 놓고, 같은 문서가 유일한 반증 실험(고-D 대조, job 873921)의
    사전등록 밴드 [12,13]ms를 C2 점추정(d92=12.88)에서 그대로 뽑았다 —
    정본이 이미 보유한 더 가까운 앵커(872077의 D108 우세 10.98, batch·ctx
    byte-matched)를 썼다면 예측은 11.0–11.4로 관측(11.26/11.32)과
    일치했을 것이다. 실험 자체는 정상 작동했다(붕괴 분기 REFUTED, §0
    CONFIRMED (scoped) 불변) — 실패한 것은 **밴드 도출 규율**이지 실험도
    §0 판정도 아니다. 항목1(격자 이전 confound)의 변종이자 항목8(끝점
    선택이 임계를 정한다)의 재발: 사전등록이 사후 반증을 막지 못하는 것은
    등록 시점의 숫자가 틀린 원천에서 나왔을 때뿐이다. 상세 §1-31.
27. ★★★★**(2026-08-06, claims-auditor Gate 2 설계 감사 2회 + result-analyst
    독립 재현) n≤8 소표본에서 percentile bootstrap CI를 판정에 쓰지
    마라 — 저장소가 이미 두 번 같은 진단을 냈는데 정본 라이브러리가 안
    바뀌었다.** `paired_bootstrap_ci`/`unpaired_bootstrap_ci`
    (`benchmarks/pdmux_eval/analyze.py:115-142`)는 n=5 percentile
    bootstrap of the mean(BCa·studentization 없음)이라 명목 95% 구간의
    실제 coverage가 **0.840**(100k trial MC, 정규 0.83948 / 이 캠페인
    rep-diff 경험분포 0.82539 / 왜도 있을 때 0.72301)에 불과하다 —
    명목 2.5%인 한쪽 오류율이 **≈8.0%**(3.2배). n별 coverage: n=4
    **0.798** / n=5 **0.840** / n=6 **0.859** / n=8 **0.888**(전부 n이
    커져도 서서히만 개선, seed 고정을 풀어도 0.8397로 불변 — **원인은
    seed도 정규성도 아니라 n=5 그 자체**). `unpaired_bootstrap_ci`도
    동일 결함(n=4/arm coverage 0.8556, Welch t 0.9590). 폭 비 =
    **0.624**(이론값 0.6314) — 과거 감사에서 인용된 0.58은 **오기**이니
    정정한다. **구조적 사실**: n=5 paired 정확 부호뒤집기 순열검정의
    두측 p 하한 = **2/32=0.0625**이므로 n=5 paired 셀은 분포무가정으로
    p<0.05에 원리적으로 도달 불가하다 — 이 프로젝트가 "CI가 0을
    배제한다"고 쓴 판정은 전부 모수(정규) 가정에 의존해 왔다는 뜻이다.
    ★**같은 진단이 저장소 안에 이미 두 번 독립으로 존재했다**:
    `results/s8_frontier/m3_analyze.py:39-57`(2026-08-02, n=4 79.8%/
    94.5% 수치까지 정확히 일치)와 `results/e1_traceforce/
    tfgate_analyze.py:89-92` — 두 캠페인 모두 **로컬로 percentile
    bootstrap을 버리고 t-CI로 갈아탔는데**, 그 규율이 정본 방법론
    라이브러리(`analyze.py`)와 P1 판정서(`results/p1_opint/
    P1_OPINT_RESULT_2026-08-05.md`)에는 전파되지 않았다. ⇒ 이건 통계
    지식의 문제가 아니라 **도구 규율(tooling discipline)의 실패**다 —
    같은 교훈을 두 번 재발견하고도 공용 라이브러리를 고치지 않으면
    세 번째 캠페인이 또 같은 함정에 빠진다. `PROJECT_STATUS.md`
    "방법론 게이트" #14, 재채점 결과는 §1-1(rev13) 참조. 상세 원자료
    `workspace/engine-port/results/p1_gates/verify/`.
28. ★★★★★**(2026-08-06, Gate 1, claims-auditor) 사전등록이 실행 전에
    "이건 항등식에 가깝다"고 자수했는데도 그 게이트로 판정을 냈다 —
    자수는 면죄가 아니다.** 방법론 게이트 #9(자기가 검증하려는 코드를
    복사한 게이트는 항등식에 가깝다, §3 항목18)의 **다섯 번째 재발**.
    `PREREG_GATE1_2026-08-06.md`는 제출 전부터 코드를 직접 추적해
    `adjust_stream_groups()`의 분기 구조상 "decode-busy ∧ prefill
    in-flight"가 곧 `idx∈{1,2}`와 사실상 동치임을 스스로 기록했다
    (§"게이트가 항등식인가"). 그런데도 그 조건으로 job 874478을
    돌리고 시간가중 100.00%를 "실질 산출"로 보고했다. **이번 재발의
    새 각도**: 이전 네 번은 사후에(결과를 본 뒤) 항등식임이 드러났지만,
    이번엔 **사전등록 문서 자신이 실행 전에 항등식 위험을 명시적으로
    자백**했다 — 그런데도 "그래도 돌린다"는 판단이 그 자백을 판정의
    면책 사유로 썼다. **항등식임이 사전에 확인되면 게이트를 고치거나
    실험을 바꿔야지, 자수만 해두고 원안대로 실행해서는 안 된다.**
    이 job이 그나마 정보를 준 것은 주 조건이 아니라 **여집합 두
    갈래**(B/C, 방법론 게이트 #10)였다는 사실이 이 교훈을 뒷받침한다
    — 판별력은 자수한 항등식 조건이 아니라 자수하지 않은 부분에서
    나왔다. 상세 §1-1(Gate 1 블록), `PREREG_GATE1_2026-08-06.md`
    §"게이트가 항등식인가".
29. ★★★★★**(2026-08-06, Gate 1, claims-auditor) 시간가중
    step-function 추정량의 두 가지 함정 — 둘 다 이번에 실증.**
    (i) **케이던스 불변성을 물리적 불변성의 증거로 쓰지 마라(항등식).**
    샘플링 케이던스를 k배 성기게 하면 이벤트 수는 대략 ÷k, 행당 dt는
    대략 ×k가 되어 시간가중 합(=Σ count×dt)이 근사적으로 불변한다 —
    `gate1_analyze.py`의 음성대조 C에서 "케이던스 8↔32에서 불변"이라는
    관측은 이 산술 항등식의 재현일 뿐, 서버의 실제 동작이 케이던스에
    둔감하다는 증거가 아니다. (ii) **행의 dt를 "다음 기록 행까지"로
    주면 비인접 구간이 오염된다.** 스냅샷 사이 간격이 서브샘플링으로
    벌어지면, 마지막 스냅샷의 "지속 시간"이 실제로는 그 뒤에 일어난
    다른 상태 전이까지 흡수해버린다 — pop A의 `t_total`이 이 오염으로
    rate2에서 35.0%(8.83 s/90 스텝, 단일 최대 1282.8 ms), rate3에서
    23.7% 부풀려졌다. `frac`(비율)은 분자·분모가 같은 오염을 공유해
    상쇄되므로 강건하지만, **`t_total`(절대량)은 강건하지 않다** —
    절대 시간을 인용할 때는 오염 방향과 크기를 반드시 병기한다. 상세
    §1-1(Gate 1 블록), `PROJECT_STATUS.md` "방법론 게이트" #15.
30. ★★★★★**(2026-08-07, G1-b, job 875293) 스코프가 좁은 주장을
    확장할 때는 인접한 한 점이 아니라 원 격자를 전부 재현하라 —
    양이 단조라는 보장이 없다.** rev14 §1-1의 "단일 분할 고정"은
    Zamba2 rate{2,3}만으로 얻은 결론이었다. `max(decode_running_
    batch_size)`가 rate 2→3에서 9→23으로 급증해 "rate 4에서 이미
    문턱(36) 근처"로 정성적으로 기대됐으나, **실제 rate 4는 18로
    rate 3보다 낮았다**(비단조, 원인 미해명). G1-b가 **rate 4만**
    돌렸다면 `max decode_bs=18 < 36`·`frac((54,54))=0`이 나와
    사전등록의 "확장" 규칙이 발화하고, 정본 문장의 범위를 873944의
    전 격자(rate 2–6)로 **넓히는 반대 방향의 오류**를 저질렀을
    것이다 — 실제로는 rate 6에서 `(54,54)`가 8.37% 등장해 그
    확장이 거짓임이 드러났다. 873944의 `MAIN_RATES` 전 격자
    `{2,3,4,6}`을 그대로(순서까지) 재현한 사전등록 설계가 이
    비단조성을 놓치지 않게 막았다. ⇒ 실무 규칙: 스코프 확장
    실험은 **경계 인접 한 점이 아니라 원 캠페인의 전 격자를
    재현**하고, 중간값에 대해 단조성을 가정하지 않는다. 상세
    §1-1(G1-b 철회 블록), `PREREG_G1B_2026-08-07.md`,
    `PROJECT_STATUS.md` "방법론 게이트" #16.
31. ★★★★**(2026-08-09, E-A, jobs 875654/875657/875661, claims-auditor)
    게이트를 지표에 걸 때는 primary뿐 아니라 보고되는 모든 블록에
    걸어라.** E-A의 F-E(정상상태 판정) 집행 수정이 사전등록 primary
    두 비교에만 적용되고 `secondary` 블록(throughput/goodput/
    ttft_p95/itl_p95)은 무방비였다 — 그리고 실제로 **그 무방비
    경로에서 인용이 일어났다**(`g2ea_analyze.py:662-680`, 인용
    금지 목록 항목2). 게이트를 설계할 때 "이 분석 스크립트가
    출력하는 모든 블록"을 나열하고 각각에 같은 정상상태 필터를
    적용했는지 확인하지 않으면, 판정 로직이 옳아도 우회 경로로
    비정상상태 수치가 새어나간다. 상세 §1-1(E-A 블록),
    `workspace/engine-port/results/p1_gates/gate2/PREREG_G2EA_
    2026-08-07.md`.
32. ★★★★**(2026-08-09, E-A) 과부하 arm과 정상 arm을 같은 제공
    rate에서 비교한 수치는 시스템 상수가 아니다.** E-A 8건 중
    7건은 F-E(정상상태 검정)가 발화해 부호만 인용 가능했다 —
    런길이 의존을 직접 실측하면 같은 셀·같은 arm 쌍의 지연차가
    프롬프트 60→120에서 **2.3–6.2×** 움직이는데, 유일한 정상상태
    셀(Granite r3)에서는 N=60→120이 1.20×에 그친다. ⇒ 지연·처리량
    격차의 "크기"를 헤드라인화하려면 먼저 그 셀이 정상상태인지
    검정하고, 아니라면 **부호만** 인용하고 크기는 지속가능-rate
    직접 대조(예: E-C)로 미룬다. 상세 §1-1(E-A 블록).
33. ★★★★**(2026-08-09, E-A) 사후 지정 셀 이동은 부호를 안
    바꿔도 인용 가능성을 만들 수 있다.** E-A의 유일한 인용 가능
    정량치(Granite rate3 +0.855)는 사전등록 이후 **선행 캠페인
    진단을 본 뒤 r4→r3로 재지정된 셀**에서 나왔다 — r4를
    유지했다면 이 캠페인은 인용 가능 셀 0개로 끝났을 것이다
    (`PREREG_G2EA_2026-08-07.md` §3 자기인지 위험 #2). 사후
    이동이 결과의 방향을 바꾸지 않았더라도, 그 이동 자체가
    "이 캠페인에 인용 가능한 무언가가 있다"는 존재 명제를 만든
    선택압이었다는 사실은 인용 시 반드시 병기한다. 상세
    §1-1(E-A 블록) C·D.
34. ★★★★**(2026-08-09, Gate 2 rev4 본 캠페인 R1′/R2′ 정본 반영 중
    발견) 사전등록 분석기가 계산하지 않는 비교는, 그 arm·n·raw
    데이터가 사전등록됐더라도 사후 비교다.** `PREREG_GATE2_2026-08-06.md`
    (rev4) §5.2 판정표는 "R1′/R2′ — A2 vs A4에 같은 TOST/우열 검정을
    적용"이라 적어 A2(`plainaux`)를 arm으로 사전등록했지만, 실제
    스코어러 `g2_analyze.py`는 `tost_equivalence`를 코드 전체에서
    **1회만** 호출하고(`:543`) 그 입력은 A3-vs-A4(`chunk512`-
    `agnostic`, `:538-539`)뿐이다 — A2는 `:734`에 서술 문장으로만
    등장한다("A1/A2/A4 data … remain valid regardless"). **"arm이
    사전등록 표에 있다"와 "그 arm 쌍의 비교가 사전등록 스코어러에
    구현돼 있다"는 다른 명제다.** 항목33("사후 지정 셀 이동")의
    형제 사례 — 이번엔 셀이 아니라 **비교 자체**가 저장된 primary
    산출물(`per_arm_x60`) 위에서 사후 계산됐다(메인 세션, 2026-08-09).
    결과의 방향(A4 우세, 부호 10/10)이 바뀌지는 않았으나, 실무
    규칙: 사전등록 판정표에 적힌 비교마다 그것을 실제로 계산하는
    스코어러 코드 줄 번호를 대조하라 — 표에 문구가 있다고 스코어러
    코드에도 구현이 있다고 가정하지 마라. 상세 §1-1(rev4 본 캠페인
    블록).
    ★**새 사례(2026-08-11, Gate 2-S 첫 유효 결과, claims-auditor) —
    "식별자 수입 ≠ 거동 수입".** `g2s_analyze.py:63`가
    `MIN_COVERAGE=0.98`이라는 **상수**를 `gate1b_analyze.py:49`에서
    가져왔지만, 그 상수를 실제로 집행하는 **규칙**은 가져오지
    않았다(원본 코드에서 이 상수를 쓰는 곳은 `gate1b_analyze.py:250`
    뿐이고 그 로직은 이식되지 않았다). 이 캠페인에서는 4셀 전부
    253–427 에피소드로 여유가 있어 수치 결과에 영향은 없었으나,
    provenance 결함 자체는 실재한다 — 상수 이름이 같다고 그 상수가
    하는 일까지 같이 넘어온 것은 아니다. 상세 §1-1(2026-08-11
    Gate 2-S 블록).
35. ★★★★**(2026-08-09, job 874601 G3 라벨 정정 — 최초 정식 등재)
    측정 실패를 게이트 실패로 라벨링하지 마라 — 이 프로젝트의
    서명 오류이며 이 항목까지 6회 재발했다.** `g2holb_report_
    zamba2_874601.json`은 4 arm 전부 `result:"FAIL"`로 저장됐으나
    실제로는 sha 추출기가 응답 스키마를 못 읽은 `KeyError:'text'`
    (하네스 결함)였고 Phase B 자체가 실행되지 않았다 — 출력이
    갈라졌다는 증거(정확성 반증)가 전혀 아니다. **이 패턴은 이
    프로젝트에서 이번이 처음이 아니다** — 핸드오프
    `session_handoff_2026-08-09.md` §1이 이미 **5회**를 서술로
    기록했다(텔레메트리 드롭 카운터 자기검열 / `UNSCOREABLE`을
    강등으로 읽음 / HTTP 400을 G3 FAIL로 / G5의 귀무-채택형 기준
    (§3 항목 미부여 상태로 위 rev18 A절에서 재채점됨) / 프로브의
    stderr 오염) — **그러나 canon(§3·"방법론 게이트")에는 이
    패턴을 다루는 번호가 이번까지 없었다**(전수 검색 확인, 2026-08-09)
    ⇒ 이번이 **최초 정식 등재**이며 카운트는 소급 반영해 **6**으로
    시작한다. 실무 규칙: 게이트/correctness 스크립트가 `FAIL`을
    반환하면, 그 전에 **하네스 자체가 데이터를 만들어냈는지**
    (null 필드·예외·스키마 불일치)부터 확인하라 — `FAIL`이라는
    문자열은 "가설이 거짓"과 "측정이 실패"를 구분하지 않는다.
    상세 §1-1(2026-08-09 두 번째 정본 반영 건 B절).
    ★**갱신(2026-08-10) — 일곱 번째 재발, 질적으로 다른 종.** job
    876699(T4-1)에서 `--time=1:00:00` TIMEOUT의 진짜 원인은 `ON+
    chunk512`의 첫 multi-chunk generate가 ~52분 무응답한 단일 호출
    스톨이었다(같은 job `OFF+chunk512`는 동일 프롬프트 ~1초). **옛
    `g2det_analyze.py`가 그 상태(ON chunk512 `n_total=0`)에 대해
    `REFUTED — reduction-order is not the (sole) cause`를 반환하고
    있었다**(`on_clean = n_total > 0 and ...`이 False로 떨어져
    REFUTED 분기로 통과) — 52분짜리 하네스 스톨이 실질적 음성
    결과로 발표될 뻔했다. **재발 1–6과의 질적 차이**: 1–6은
    라벨·해석 오류였고 **사람이** 문서/보고 단계에서 잡았다. 이번
    (7)은 **분석 코드 자신이 거짓 음성(REFUTED)을 산출**했고, 막은
    것은 도구가 아니라 experiment-runner가 `INCOMPLETE`로 보고하며
    사전등록이 열거하지 않은 조건이라고 채점을 거부한 **실행자
    규율**이었다 — 도구가 사람보다 관대했던 최초 사례. **사후
    완화가 아니다**: 사전등록 REFUTED 조건은 "ON이 어느 rep에서든
    self-mismatch ≥1"인데 rep 0개면 그런 관측 자체가 없다 — 옛
    코드는 규칙의 재해석이 아니라 **버그**였다. 수정(2026-08-10,
    `g2det_analyze.py`)은 이 경우 `NO VERDICT (MEASUREMENT
    ABSENT)`를 반환하도록 가드를 추가했고, 데이터가 있을 때의
    CONFIRMED/REFUTED 분기(5-케이스 매트릭스)는 **불변**임을
    확인했다. **재발 카운트: 6 → 7.** 상세 §1-1(2026-08-10 세 번째
    정본 반영 건 A절), 원자료 `workspace/engine-port/results/
    p1_gates/gate2/g2det_876699.out`·`g2det_876699.err`·
    `g2det_analyze.py`.
36. ★★★**(2026-08-09, engine-porter 발견 + 메인 세션 노출범위
    실측) 통계 라이브러리의 조용한 폴백은 아티팩트에 기록되지
    않는다 — 분석 재현 시 인터프리터 환경을 아티팩트에 남겨라.**
    `g2ea_analyze.py:84-87`는 `scipy` import 실패를 조용히 흡수하고
    (`_scipy_stats=None`), `t_cdf()`(`:156-159`)는 그 경우 **모든
    df에서** Student-t 대신 정규 CDF로 근사한다 — 커밋된
    `g2ea_report_*.json`의 `p_tost`가 정확히 `1.0`인 것이 그 흔적
    이다. `t_ppf()`(`:142-153`)는 df 1–10·p∈{.95,.975}에 한해서만
    하드코드 표로 정확하고 그 밖은 무조건 `1.96`을 반환한다. 산출물
    JSON은 이 인터프리터 상태를 **어디에도 기록하지 않는다** —
    메인 세션이 `raw_ci` 폭을 역산해서야(implied_t = 2.2621…
    = t(.975,df=9), 1.96이 아님) CI 자체는 df=9 표값과 일치해
    오염되지 않았음을 사후 확인할 수 있었다. **게이트 #14 계열**
    (n≤8 `paired_bootstrap_ci` undercoverage, §3 항목27)과 뿌리가
    같다 — 통계 함수의 정확도가 **실행 환경(라이브러리 가용성)에
    조건부**인데 그 조건이 산출물에 남지 않으면, 몇 달 뒤 같은
    파이프라인을 다른 인터프리터에서 돌린 사람은 자신이 다른
    숫자를 재현하고 있다는 것조차 모른다. 실무 규칙: 통계 계산에
    쓰는 라이브러리가 선택적 의존성이면 (i) import 성공 여부를
    산출 JSON에 필드로 남기고, (ii) fallback 경로가 근사임을
    stdout에 경고로 남기고, (iii) fallback의 유효 df 범위를
    코드 주석이 아니라 데이터로 노출하라. **이 사건 자체는 정본
    수치를 오염시키지 않았다**(이 캠페인의 관측치가 0.05 경계에서
    멀어 판정 불변) — 등재하는 것은 수치 정정이 아니라 이 도구
    규율뿐이다. 상세 §1-1(2026-08-09 두 번째 정본 반영 건 C절).
37. ★★★**(2026-08-10, engine-porter 발견 + 메인 세션 코드 직접
    확인) 공유 하네스의 무한 대기 — 도구 규율, 항목35와 뿌리 사건은
    같으나 레슨은 다르다.** `g2_holb_phaseA_lib.sh`의 `greedy_call`
    이 **`--max-time` 없는 raw curl**이었고, 이 디렉터리의 **모든
    캠페인**(g2ctrl/g2ea/g2holb/g2det)이 이 함수를 공유해서 쓴다.
    소비자 10개를 전수 확인한 결과 **4개가 타임아웃을
    `FAIL`/`SMOKE_FAIL` 계열로 채점**하고 있었다(항목35의 `g2det_
    analyze.py` 포함). 수정: `--connect-timeout 10 --max-time
    180`(env `G2_GREEDY_CONNECT_TIMEOUT`/`G2_GREEDY_MAX_TIME`로
    재정의 가능), curl 종료코드 28을 `STATUS=TIMEOUT`으로
    `STATUS=ERROR`와 구별, **SHA를 아예 방출하지 않아** mismatch
    채점이 구조적으로 불가능, 사이드카 `.status.json`. 기본값 180s
    는 **측정 근거**로 정당화됐다 — 이 디렉터리 아카이브 응답
    **n=64**(jobs 874602/874628/874633/874635/875344/875346/
    875610/875611/876699, 4 arm × 2 모델)의 서버측 `e2e_latency`가
    median 0.978s / p90 2.534s / **max 6.605s**(최악 관측의 27배,
    중앙값의 184배) — 메인 세션이 `g2_holb_phaseA_lib.sh:102-139`
    주석의 근거 문단을 코드에서 직접 대조해 확인. 정상 경로는
    아카이브 64개 + 합성 실패 9종 재생으로 **73/73 byte-identical**
    검증됐다(engine-porter 보고 — 메인 세션은 이 재현 스위트 자체를
    독립 재실행하지 않음, 근거 아티팩트 경로 미확인이라 캐비어트로
    남긴다). **실무 규칙**: 공유 하네스 함수 하나가 여러 독립
    캠페인의 correctness 채점 경로에 들어가면, 그 함수의 실패 모드
    (특히 무경계 대기)는 **한 캠페인이 아니라 그 함수를 쓰는 모든
    소비자의** 위험이다 — 소비자별로 타임아웃을 막지 말고 공유
    지점에서 한 번 막아라. 상세 §1-1(2026-08-10 세 번째 정본 반영
    건 B절), 원자료 `workspace/engine-port/results/p1_gates/gate2/
    g2_holb_phaseA_lib.sh`.
38. ★★★**(2026-08-11, Gate 2-S 첫 유효 결과, claims-auditor 적대
    감사) `any()` over n reps 형태의 스크린은 귀무 발화율이
    1−(1−α)ⁿ이며, 인용 자격을 사실상 무작위로 배정한다.** Gate 2-S
    의 인용 가능 셀 선별 필터가 이 형태였고, α=0.05·n=10에서
    귀무(효과 없음) 상황에서도 **40.1%**(=1−0.95¹⁰, 메인 세션
    재계산 일치)의 확률로 "이 셀은 인용 가능"이 발화한다 — 인용
    가능 셀이 나머지 3셀과 물리적으로 다르다는 근거가 아니라 다중
    시행의 산술적 산물일 수 있다는 뜻이다. 실무 규칙: 여러 rep·셀에
    "하나라도 조건을 만족하면" 식의 OR형 스크린을 인용 자격 게이트로
    쓸 때는 그 스크린 자체의 귀무 발화율을 먼저 계산하고, 그 값이
    무시할 수 없이 크면(여기서처럼 40%대) 그 사실을 인용 시 병기하라
    — 결과 자체를 무효화하지는 않되(부호·크기는 그대로 유효), "이
    셀이 지정 셀"이라는 특권적 지위는 취소된다. 상세 §1-1(2026-08-11
    Gate 2-S 블록 금지 항목⑥).
39. ★★★**(2026-08-11, Gate 2-S 첫 유효 결과, claims-auditor 적대
    감사) 코드가 산출하기로 되어 있는 진단 필드가 실제로는 산출되지
    않을 수 있다 — 관측자 대칭 보고도 예외가 아니다.** `PREREG_
    GATE2S_2026-08-09.md` §6.3-6이 명시한 관측자 대칭 진단 필드
    (`dropped_events`/`writer_error`)가 이번 캠페인 산출물에
    **산출되지 않았다** — 감사자가 원자료에서 오프라인으로 복구해
    전부 0임을 확인했다(간극 자체는 실재했다는 뜻, 값이 0이었던 것은
    결과일 뿐 사전등록이 요구한 자동 산출 경로가 작동했다는 증거가
    아니다). 실무 규칙: 사전등록이 "이 필드를 보고한다"고 적었다고
    그 필드가 실제로 산출됨을 가정하지 마라 — 채점 전에 산출물
    스키마를 사전등록 문서와 직접 대조하고, 빠진 필드는 (있었다면
    나왔을 값이 아니라) **산출 경로의 결함**으로 기록하라. 상세
    §1-1(2026-08-11 Gate 2-S 블록).
    ★**여덟 번째 재발(2026-08-11, E1 addendum, 직전 회차 등재
    직후)**: `PREREG_G2S_E1_ADDENDUM_2026-08-11.md` §4가 명시한
    서술 전용 진단("rep 경계에 인접한 행이 pop A인 건수")이
    산출되지 않았다 — `g2s_e1_premise.py:28` docstring은 `classify()`
    를 수입 목록에 적었으나 실제 코드에서 한 번도 호출되지 않는다.
    판정에는 영향이 없었으나(그 진단은 서술 전용이지 판정 입력이
    아님), 같은 서명이 바로 앞 회차(항목39 자신)에 정식 등재된
    직후 재발했다는 사실은 이 실패 모드가 "안다고 없어지지 않는다"는
    걸 보여준다.
    ★**세 번째 재발, 그러나 이번엔 등재 시점에 닫혔음(2026-08-16,
    R2 재분석 rev1→rev2)**: `oracle_reanalysis_2026_08_16.py` rev1은
    §1-32의 `+1.79%p CI [+0.917,+2.625]`를 **산출하지 못했다** —
    `unpaired_bootstrap_ci`가 import만 되고 한 번도 호출되지 않았고
    `bootstrap_over_arms`는 죽은 코드였다(JSON에도 없었다). 2026-08-14
    E-1a·2026-08-16 C2-R와 동형인 패턴이 다시 나타난 것 — 다만 이번엔
    claims-auditor 적대 감사가 rev1 제출 **직후**(같은 등재 사이클
    안에서) 잡았고, rev2가 실제로 배선해 25개 비교를 JSON으로 산출
    (AST 스캔으로 미참조 함수 0건 확인)했으며 감사의 독립 재구현값과
    소수 셋째 자리까지 일치함을 확인했다. 재발 카운터는 올리되
    **"미해결"로 남기지 않는다** — 닫힌 사례로 기록. 상세
    `../workspace/engine-port/results/slo_sched/
    ORACLE_REANALYSIS_2026-08-16.md` rev1→rev2 변경표.
40. ★★★**(2026-08-11, Gate 2-S 1차 실행 실패 후속) 대형 캠페인
    제출 전 배관 스모크를 규율로 등재한다 — 스모크의 PASS는
    성능 판정에 아무 정보도 주지 않는다.** Gate 2-S 1차 실행(jobs
    877107/877109, 6.40 GPU-hr, `COMPLETED exit 0:0`)은 스코어러가
    `KeyError: 'itls'`로 죽어 사전등록 primary를 **0개** 냈다
    (하네스 결함 6건, §1-1 2026-08-10 세 번째 정본 반영 건 참조).
    재실행 전에 돌린 **배관 스모크**(job 877593, 6분 40초 =
    **0.11 GPU-hr**, n=1)가 결함 6(rep glob이 telemetry 파일을
    삼키고, **분류기 자신이 오분류**해 exit 21[GPU 재실행 필요]을
    냈어야 할 자리에 20을 내던 문제)을 잡았다 — 분류기를 그대로
    믿고 재실행했다면 6.40 GPU-hr가 다시 낭비됐을 것이다(**약
    60배** 비용 절감, `session_handoff_2026-08-10.md` §2.6–2.7).
    **발동 조건(구체 — 조건 없는 규율은 지켜지지 않는다)**: 캠페인의
    하네스·스코어러 코드가 **직전 감사 통과 캠페인과 diff가 있고**
    (신규 작성 포함), 예상 GPU 비용이 **≥1 GPU-hr**(스모크 통상
    비용 0.1–0.2 GPU-hr의 대략 10배 이상)이면, 본 제출 전
    **최소 rep(n=1–2) 스모크**를 선행한다. 스모크는 (a) 실제 실행
    경로로 (b) 실제 스코어러가 소비할 산출 스키마를 만들고 (c)
    계약 위반(필드 누락·행 수·타입) 0건을 확인한 뒤에만 본
    캠페인을 제출한다. **면제**: 하네스·스코어러가 이전 감사 통과
    캠페인과 diff 0(바이트 동일)이면 생략 가능. **이 게이트는 결과
    채점 게이트가 아니라 배관 게이트다** — PASS는 그 자체로 어떤
    성능 판정도 licence하지 않는다(§0.0류 항등식으로 재해석 금지).
    ★**따름정리(실무 규율)**: 스모크의 판정 코드 자신을 무조건
    신뢰하지 마라 — 이 사례에서도 분류기 자신이 결함 6(오분류)을
    냈고 잡은 것은 사람(메인 세션)의 원자료 직접 대조였다(항목35·
    방법론 게이트#21과 같은 뿌리: "판정 코드가 스스로 오판할 수
    있다"). 상세 `session_handoff_2026-08-10.md` §2.6–2.7, 원자료
    `workspace/engine-port/results/p1_gates/gate2/
    g2ssmoke_verdict_877593.txt`·`g2ssmoke_877593.out`·
    `g2ssmoke_manifest_877593.txt`·
    `g2ssmoke_scoreout_zamba2-27b_877593.txt`(수정 금지·인용만).
    `PROJECT_STATUS.md` "방법론 게이트" #26 동반 갱신.
41. ★★**(2026-08-11, G1-c, job 877974) 결정량의 밀도 의존성을 먼저
    따져라 — "sparse 텔레메트리로는 낼 수 없다"는 주장은 어느
    통계량에 대한 것인지 먼저 밝혀야 한다.** §8.9.1의 in-job
    반증기가 VERIFIED를 못 내는 것은 설계상 단방향이기 때문만이
    아니라 실제로 밀도(`PDMUX_TRACE_FORCE_PREFILL=0`, ~1/32)에도
    의존했다. 그런데 `max(decode_bs)`는 스케줄 그리드를 population과
    무관하게 표본추출하므로 **sparse 텔레메트리에서도 나온다** —
    Gate 2-S 자신의 A4 원자료(877756/877757)에서 사후 산출한 값이
    r3=10·r4=13(rep별 [9,13], claims-auditor 산출)이었다. 즉 "전제
    검증의 경험적 절반(`max_decode_bs<36`)은 새 GPU 캠페인 없이 이미
    지불돼 있었다" — G1-c가 실제로 추가한 것은 `frac((54,54))`의
    **고밀도 직접 관측**과 양성대조뿐이다. 실무 규칙: "이 라벨은
    오직 새 캠페인만 낼 수 있다"는 서술은 반드시 **어느 통계량**에
    대해 참인지 한정하라 — 밀도에 둔감한 통계량과 민감한 통계량을
    뭉뚱그리면 실험의 필요성을 과장하게 된다. 상세 §1-1(2026-08-11
    G1-c 블록), `PROJECT_STATUS.md` "방법론 게이트" #27.
42. ★**(2026-08-11, G1-c) 실험이 무엇을 풀어주는지가 코드 사실인지
    추정인지, 실행 전에 코드로 확인한 뒤 정당화하라.** 이번 세션
    초기 동기 서술("전제가 VERIFIED되면 Gate 2-S 크기 인용 셀이
    1→3개로 는다")은 실행 후 원자료·코드 대조로 반증됐다 — 크기
    인용을 막는 것은 `premise` 라벨이 아니라 독립 산출되는 F-계열
    gate였다(`g2s_analyze.py:1157-1161`). 이번엔 실험 자체가
    0.10 GPU-hr로 저렴해 피해가 작았지만, 원칙은 비용 규모와
    무관하다: 실험을 정당화하는 문장이 "이 실험이 풀어줄 것"이라고
    서술하는 대상은, 실행 전에 그 서술을 산출 코드와 직접 대조해
    코드 사실인지 서술자의 추정인지를 구별해야 한다 — 대조하지
    않으면 예산·시간을 실제로 풀리지 않는 것에 쓸 위험이 있다.
    (관련: 항목19 "사전등록 표의 비교 문구와 스코어러 구현은 다른
    명제다" — 이번은 그 패턴이 분석 결과가 아니라 **실험 동기
    서술**로 옮겨간 사례.) 상세 §1-1(2026-08-11 G1-c 블록),
    `PROJECT_STATUS.md` "방법론 게이트" #28.
43. **(2026-08-11, G1-c, engine-porter 이관 대기) `compute_coverage`
    류(span 기반: `(min(last,t1)−max(first,t0))/(t1−t0)`) 지표는
    내부 구멍에 맹목이다.** `gate1c_analyze.py:90-99`가 rate별 창의
    첫·마지막 스냅샷만으로 "coverage=100.00%"를 냈으나 실제로는 창
    내부에 유의한 공백이 존재했다(예: rate2 내부 최대 gap
    2.749s = 창의 4.10%). 이번 판정에는 영향이 없었다(양성대조·
    grid-completeness가 독립적으로 결측 0을 확인) — **무해했던
    것은 우연이지 정의의 방어력이 아니다.** 실무 규칙: span 기반
    coverage를 보고할 때는 **최대 내부 gap**을 함께 병기하라.
    `pdmux_eval/analyze.py`로 이 지표를 이관할 때 반영할 것. 상세
    §1-1(2026-08-11 G1-c 블록), `PROJECT_STATUS.md` "방법론 게이트"
    #29.
44. ★★**(2026-08-11, E1 addendum, claims-auditor) 저장소가 이미
    "이 경로는 없다"고 명시적으로 써 둔 지점 옆에서 다른 통계량이
    낸 "경로"는, 자기인지 자백만으로는 재감사를 면제받지 않는다 —
    스코어러의 부정 선언문(negative invariant string)을 grep해
    대조하는 것이 감사 절차의 일부여야 한다.** `g2s_analyze.py:682`의
    `falsifier()`는 결과 문자열에 문자 그대로 `"NO UPGRADE PATH
    EXISTS -- state stays 'unchanged'"`를 기록해 둔다 — 이것은
    `frac(idx2)` 통계량에 대한 설계 불변식(§8.9.1 안전장치)이다. E1은
    **다른** 통계량(`max(decode_bs)`)으로 사실상의 "업그레이드"를
    냈고, prereg 자신도 §1에서 "이 통계량엔 §8.9.1의 단방향 안전성이
    자동 상속되지 않는다"고 자백하며 재감사를 요청했다 — 그 자백이
    있었기 때문에 이번엔 위험이 감사로 흡수됐다. 실무 규칙: (i)
    사전등록/addendum이 기존 negated-invariant를 우회하는 새
    통계량을 도입하면, 그 통계량이 실제로 독립적인지(다른 표본·다른
    가정·다른 자유모수)를 **코드 대조로** 확인하기 전까지는 감사자가
    최우선으로 검토할 항목으로 표시한다. (ii) 자기인지 자백(§0류
    경고문)은 필요조건이지 충분조건이 아니다(항목9 "자수는 면죄
    아님"의 구체화). 상세 §1-1(2026-08-11 E1 addendum 블록),
    `PROJECT_STATUS.md` "방법론 게이트" #30.
45. ★★**(2026-08-11, E1 addendum, claims-auditor) 구간 안에서
    관측되지 않은 sup을 bound하려면, 표본들 사이의 최대 변동(인접
    샘플 차이)이 아니라 그 구간에 대해 실제로 성립하는 확률모델에서
    유도한 상한을 써야 한다 — 전자는 점추정이고 후자만 bound다.**
    메인 세션이 처음 제안한 "인접 관측 사이 최대 변화량"류 bound는
    claims-auditor가 기각했다 — 그것은 관측된 값들 사이의 국소
    변동성을 재는 점추정 성격의 양이지, 32-스텝 샘플링 간격
    내부에서 도달했을 수 있는 진짜 sup의 확률적 상한이 아니다.
    감사자가 제시한 실제 bound(`sup ≤ in-system + Poisson(λ·dt)`,
    도착과정 가정만 추가·자유모수는 그대로 0)는 셀별로 25/41/24/27을
    내며, 이 중 Zamba2 r3(41)는 q=1e-6에서 이미 문턱 36을
    넘는다 — 잘못된 bound를 썼다면 이 경계 사례를 놓쳤을 것이다.
    실무 규칙: "bound"라고 부르는 양을 볼 때마다 그것이 어떤 확률
    모델의 꼬리 분위수인지, 아니면 그저 관측값들의 최대 차이인지
    구별하고, 후자를 전자로 오인해 채택 근거로 쓰지 않는다. 상세
    §1-1(2026-08-11 E1 addendum 블록), `PROJECT_STATUS.md` "방법론
    게이트" #31.
46. ★★**(2026-08-11, 트래픽·roofline 진단, result-analyst) 다른
    캠페인·다른 스케일에서 수입한 보조 수치는 기준(basis)이 같은지
    검증하라 — 항목34("사전등록 표의 비교 문구와 스코어러 구현은
    다른 명제다")의 숫자 층 변형.** `workspace/engine-port/results/
    s8_scaleup/FINDINGS_8B_2026-07-28.md` §2.1의 "1차 weight-traffic
    추정(M 5.40/T 6.17/H 7.66 GB, H/M=1.42)"은 2026-07-28·2026-08-04
    두 차례 감사를 통과했으나, 실은 **7-8B 캠페인 문서에 3B급 모델
    (state-spaces/mamba2-2.7b, Qwen2.5-3B, Zyphra/Zamba2-2.7B)의
    수치가 섞여 들어와 있었다** — 체크포인트 파일 크기·safetensors
    텐서 합을 실측 대조해 3자리까지 정확히 일치시켜 출처를 확정.
    게다가 M·T는 체크포인트 바이트인데 H만 호출-인지(shared block
    재독출 곱한) 트래픽이라 **기준까지 혼합**돼 1.42×가 만들어졌다
    (같은 3B 격자·같은 기준이면 0.985). 실무 규칙: 보조 수치를
    논증에 수입할 때는 (i) 어느 캠페인·어느 모델 스케일의 것인지
    (ii) 무슨 기준으로 낸 것인지를 수치 옆에 명시하고, 같은 문서
    안의 다른 수치와 기준이 다르면 비율을 만들기 전에 통일하라.
    상세 `PROJECT_STATUS.md` "방법론 게이트" #32, `TRAFFIC_ROOFLINE_
    DIAGNOSTIC_2026-08-11.md` §7.
    ★**새 사례(2026-08-14, doc-steward, E-3 realized SM count 프로브의
    부수 발견) — 이번엔 숫자 층이 아니라 하드웨어 식별 층이고, #32를
    만든 바로 그 문서가 같은 종류의 오류를 두 번 냈다.** 위 §2.1 수치
    혼입을 잡아낸 바로 그 `TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md`가
    `nvidia-smi -q`로 하드웨어를 식별한 노드(로그인 노드 glogin01,
    `A100 80GB PCIe`)가 실제 측정이 실행된 노드(컴퓨트 노드 gpu36/
    gpu38/gpu40, SXM4)와 **달랐다** — `s8_scaleup/` job 아티팩트
    전체에 `"A100 80GB PCIe"` 문자열이 **0건**(job 882374로 컴퓨트
    노드가 `NVIDIA A100-SXM4-80GB`임을 직접 확인, `scontrol show node`로
    gpu36·gpu40·gpu43 동일 feature 확인). 정정: 사양 BW 1935→2039 GB/s,
    achieved_BW 비율 48–61%→45.4–57.7%, ridge 161→153 FLOP/byte — 어떤
    판정도 뒤집히지 않으나(오히려 강화) 출처 층이 하나 더 있음을
    보여준다. 실무 규칙 추가: **하드웨어 사양을 인용할 때는 측정이
    실제로 실행된 노드에서 읽어 job 아티팩트에 기록하라** — 로그인
    노드와 컴퓨트 노드가 다른 SKU일 수 있다(이 클러스터가 실제로
    그렇다: glogin01=PCIe, gpu36–43=SXM4). 상세 `PROJECT_STATUS.md`
    "방법론 게이트" #32(追記), `TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md`
    §11(addendum).
    ★**새 사례 2(2026-08-14, E-1a, B-4) — 이번엔 basis 미검증 수입이
    맞았던 사례(성공 사례로 등재).** C2 caveat이 인용해 온 "rep 간
    sd 0.01–0.07"은 basis(어느 캠페인·어느 쌍인지)가 적혀 있지 않은
    수입값이었다. 2026-08-14 E-1a가 원자료(`bsweep_regime/
    e1a_preanalysis_2026-08-14.json`의 `T3_sd_eps`)에서 **직접
    측정**해 `sd_rep(ε, 44→92)` = Ha8 0.0102·T8 0.0083(ITL 수준 rep
    CV 0.2–0.7%)로 대체했다 — 44→92 쌍·Ha8/T8 2 arm·이 격자
    한정이라고 스코프도 함께 명시. 실무 규칙(강화): basis를 모르는
    수입값을 발견하면 **금지만 하지 말고, 가능하면 그 자리에서
    직접 재측정해 대체**하라 — 이번엔 GPU 비용 없이(기존
    캠페인 원자료 재분석) 가능했다. 상세 `PROJECT_STATUS.md` "8B
    decode-SM 민감도 측정 노트"(rep 분산 실측값 단락).
47. ★★★**(2026-08-14, doc-steward, 2026-08-11 세션의 GPU 실험 4건
    검토 재해석 — 새 측정 아님, 규율 개정 권고) 게이트의 발동 조건
    주어가 단수 "캠페인"이면 <1 GPU-hr 조각으로 쪼개 회피할 수
    있다.** 방법론 게이트 #26(대형 캠페인 제출 전 배관 스모크, 항목
    40)의 발동 조건은 "예상 GPU 비용이 ≥1 GPU-hr"였다. 2026-08-11
    세션이 준비했다가 사전등록 부재로 제출하지 않은 GPU 실험 4건
    (P1 · E1-b/E1-c · `%smid` P1+P2 · G1-a)은 개별 예상 비용이
    0.1–0.7 GPU-hr로 각각 문턱 아래지만 **합은 ≈1.15–1.65 GPU-hr**
    로 문턱을 넘는다 — 트리거가 캠페인 단위인 채로 있으면 넷 다
    스모크 없이 통과한다. 이 넷은 우연히 같은 시점에 준비된 것이
    아니라 실제로 결합돼 있다: 같은 매니페스트 트립와이어
    (`g2s_run.sbatch:81-89`의 `added==2 ∧ removed==0`,
    `gate1b_run.sbatch`·`gate1c_run.sbatch`의 `N_CHANGED==0 ∧
    N_ADDED==2`)를 공유하고, "dev 트리를 바꾸는 것은 G1-a 하나뿐이라
    나머지 셋(P1/E1-b·c/`%smid`)은 매니페스트 무변경이라 동시 제출이
    코드 사실로 안전하다"는 관찰 자체가 이 넷을 한 배치로 묶는
    근거였다(`handoff-report/session_handoff_2026-08-13.md` §2.5
    "GPU 실험 4건 실행 준비 검토"). 실무 규칙(개정): 배관 스모크
    발동 조건의 트리거 단위를 "캠페인 하나"에서 **"한 배치로
    제출되는, 신규/변경 코드를 공유하는 캠페인들의 합"**으로
    바꾼다 — 그 합이 ≥1 GPU-hr이면, 배치 안에서 하네스·스코어러에
    신규/변경분이 있는 캠페인마다 최소 rep 스모크를 선행한다(직전
    감사 통과 캠페인과 diff 0인 캠페인은 기존 면제 유지). 항목40
    (게이트 #26 본문)은 대체하지 않고 이 항목으로 追記한다 — 이
    개정을 적용해 실제로 스모크를 새로 돌린 사례는 아직 없다.
    상세 `PROJECT_STATUS.md` "방법론 게이트" #26(追記).
48. ★★★**(2026-08-14, doc-steward, 코드 직접 확인 — 새 측정 아님)
    "매니페스트 N/N sha 일치"는 런타임이 바이트 동일함을 함의하지
    않는다 — 매니페스트가 실제로 덮는 범위를 먼저 확인하라.**
    `workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh`는
    정확히 **15개 파일만** sha256으로 해시한다(`:99-115`). 그 밖에서
    런타임 거동에 관여하는 것으로 확인된 것: (i)
    `sglang/srt/multiplex/pdmux_context.py` — `(74,34)` 등 파티션
    기대값을 만드는 `divide_sm()`이 여기 있고
    `multiplexing_mixin.py:28`가 이 파일을 import하지만, sync는 이
    파일을 **설치도 해시도 하지 않는다**(dev 트리 mtime 2026-04-06,
    sync가 만지는 형제 파일들의 2026-08-11과 불일치; 저장소
    `src/multiplex/`에 이 파일 자체가 없다). (ii)
    `sgl_kernel/spatial.py`(green-context 생성 원시함수) — dev
    트리(`sglang_engine_dev`)에는 아예 없고 venv 사이트패키지
    (`sglang_engine_venv/lib/python3.14/site-packages/sgl_kernel/`)
    에만 있다 — sync의 관할 밖. (iii) `src/patches/`의 패치 5개 중
    sync가 참조하는 것은 3개뿐(`pdmux_thread_local_role.patch`
    `:39`·`holb_probe_scheduler_hooks.patch`:`64`·
    `mamba2_pure_ssm_arch.patch`:`89`) — `nemotron_h_forward_split_
    prefill.patch`·`triton_backend_mambaish_vheaddim.patch`는
    저장소 전체 검색으로도 **어느 실행 경로에서도 적용되지
    않는다**(`env/dev_tree_edits.md` 항목6:35·항목7:43 자신의 서술도
    이 두 patch를 "Full method" 참고용으로만 가리킬 뿐 sync 대상으로
    적지 않는다). (iv) `env/dev_tree_edits.md` 항목 3·4·5·7
    (`hf_transformers_utils.py` 레지스트리·`configs/__init__.py`·
    `model_runner.py`·`triton_backend.py`의 Zamba2/v_head_dim 관련
    수동 편집)은 sync가 재적용도 해시도 하지 않는 **수동 편집**이고,
    항목 6·8·9(`models/{nemotron_h,falcon_h1,granitemoehybrid}.py`)
    는 sync 스크립트 자신의 주석(`:73-75` "still manual copies")이
    명시적으로 자백하며 해시 목록에도 없다.
    ⇒ **재현성 주장의 범위를 좁힌다**: "매니페스트 N/N sha 일치"가
    보증하는 것은 그 매니페스트가 나열한 **정확히 그 파일들**의
    바이트 동일성뿐이다 — 캠페인 간 "동등한 코드에서 실행됐다"는
    주장(Gate 1/`gate1b`/`gate1c`/Gate 2-S 트립와이어가 근거로 쓰는
    "13/13"·"15/15 sha 바이트 동일", 예: 아래 Gate 1 결과 블록의
    "매니페스트 13파일 SHA-256이 873944와 바이트 일치 … 커버 밖
    드리프트 없음")는 이 매니페스트가 나열한 파일 범위로 한정해서
    읽는다. **과잉 강등 금지 — 두 가지 구별 필수**: (a) 이것은
    **기존 성능 판정을 뒤집지 않는다**. 지금까지의 캠페인 사이에
    실제로 커버 밖 드리프트가 있었다는 증거는 없다(그런 주장도 하지
    않는다) — 좁아지는 것은 "매니페스트가 무엇을 증명하는가"라는
    **주장의 범위**뿐이다. (b) "패치 2개가 sync 경로에서 미적용"은
    "그 기능이 런타임에 없다"를 함의하지 않는다 — `nemotron_h.py`/
    `falcon_h1.py`처럼 해당 변경이 수동 편집으로 이미 트리에
    반영돼 있을 수 있다(사실 `dev_tree_edits.md` 항목6·7이 정확히
    그렇다고 기록한다). 확인된 것은 오직 "sync 스크립트가 그것을
    보장하지 않는다"까지다. 실무 규칙: 캠페인 간 코드 동등성을
    주장할 때는 그 근거로 쓰는 매니페스트/트립와이어가 **실제로
    무엇을 해싱하는지**(전체 소스 트리인지, 지정된 부분집합인지)를
    먼저 코드로 확인하고, 그 부분집합 밖의 런타임 의존성(외부
    패키지·미러 안 되는 헬퍼 모듈·수동 편집)이 있는지 별도로
    점검한다. 상세 `PROJECT_STATUS.md` "방법론 게이트" #33, 원자료
    `workspace/engine-port/scripts/bootstrap/
    sync_engine_tree.sh`(`:39-116`)·`workspace/engine-port/env/
    dev_tree_edits.md`(항목 3–9)·
    `handoff-report/session_handoff_2026-08-13.md` §4 "정본 등재
    후보 2건".
49. ★★★**(2026-08-14, doc-steward, 4건 사전등록 감사 결과 종합 —
    새 실험 아님) 사전등록은 규칙과 하네스를 한 번에 감사받으면
    안 된다 — 규칙 먼저 감사받고, 통과한 규칙에 대고 지은 하네스를
    다시 감사받아라.** 2026-08-14 하루에 사전등록 **4건**(LTSM P1 ·
    Gate 2-S E1-b/c · `%smid` R0 · E-1)이 감사를 받아 **3건 NO-GO,
    1건 CONDITIONAL-GO**가 났다. 넷 다 차단 결함이 **예외 없이
    스코어러·하네스 구현 층**이었고(결정 규칙 자체의 하자가 아니라),
    **GPU 0으로 발견 가능**했다. 더 나쁘게는 셋(LTSM P1·E1-b/c·E-1)에서
    **코드 자신이 잘못된 라벨을 산출**했다 — 사람이 사후에 오독한 것이
    아니라 도구가 실행 중 거짓 음성/양성을 냈다(항목35의 6·7번째
    재발과 같은 부류지만, 이번엔 **4건이 같은 날 동시에** 나타나
    패턴이 개별 사고가 아님을 보여준다). ★**개인 부주의가 아니라는
    증거**: 이 넷 중 E-1(`bsweep_regime/PREREG_E1_BSWEEP_REGIME_
    2026-08-14.md`)은 **메인 세션 자신이 작성**했고, 다른 세 건과
    **같은 종류의 결함**(하네스 배선 부재·수치 basis 미검증)을
    냈다 — 특정 에이전트나 특정 작성자의 습관이 아니라 "규칙과
    구현을 한 사전등록 문서에 같이 적으면 감사가 규칙 층에서 소진돼
    구현 층 결함이 새어나간다"는 **구조적 실패 모드**다. 실무 규칙:
    사전등록을 두 단계로 나눠 감사한다 — **(i) 결정 규칙만 먼저
    감사**(입력·임계값·판정 로직이 사전에 명시한 질문에 답하는가),
    **(ii) 통과한 규칙에 맞춰 하네스/스코어러를 구현한 뒤 그 구현을
    별도로 다시 감사**(GPU 0 스모크로 각 분기가 실제로 발화하는지
    확인). 한 번에 제출하면 감사자가 규칙의 타당성에 주의를 쓰는
    동안 구현 버그가 나란히 통과한다. 메모리
    `deconfound-measurement-lessons.md` 항목34와 대응. 상세
    `PROJECT_STATUS.md` "다음 실험 gate" #11(4건 레지스트리)·"방법론
    게이트" #34.
50. `POST-HOC · NOT BLIND · NOT PRE-REGISTERED · PRODUCER-INCOMPLETE.`
    ★★**(2026-08-14, E-1a Tier 2 T2-1, claims-auditor 조건부 채택 —
    §1-1 성능 절에는 반영하지 않는다, 새 정책 주장 아님)** 사후 회귀
    `S(B)=k+b·Δw`(`Δw`=T8−Ha8 weight-traffic share 차)의 절편 `|k|`는
    12개 적합 전부에서 **[0.051, 0.082]**이고 B=1 포함 여부로 부호가
    뒤집힌다(전체 −0.0574 ↔ B=1 제외 +0.0661). 반드시 다음을 병기:
    (i) B=1은 과도(transient)가 아니라 동거 prefill이 최대인
    정상상태라 `k`를 arm 상수 오프셋으로 해석 금지 — 조성/간섭 항의
    흡수항이다; (ii) Ha8 arm은 B축이 job축과 완전 교락(B=7·9는 job
    865533 전용, B=15·16은 job 865493 위주) ⇒ `k`·`b`가 캠페인 효과와
    분리 안 됨; (iii) job 간 sd(log ITL)=0.0052(n=20, 두 job이 공존하는
    셀에서만 추정 가능)는 회귀가 쓰는 Ha8 10개 B-포인트 중 **8개
    (B=2,6,7,8,9,10,13,16)에서는 추정 자체가 불가**(그 셀엔 job이
    하나뿐)— 남은 2개(B=1,15)만 부분적으로 이중job; (iv) 이 추정량은
    사전등록이 요구한 paired가 아니라 독립표본(Welch) t다; (v)
    "4개 적합 중 3" 류 요약은 어느 4개인지 명시 없이 인용 금지(12개
    k-CI 중 0을 포함하는 것은 7개). ★**weight 서사는 방향 불문 전면
    금지**: `Δw`는 `log B`·`√B`와 강공선(r≥0.978, 실측 r(log
    B)=0.9996·r(√B)=0.9999; 사전등록 계획 격자 {9,12,24,40,48}에서도
    r(log B)=0.998) ⇒ 이 설계는 "weight share가 원인"과 "B에 단조인
    임의의 원인"을 판별하지 못한다 — **지지도 반증도 금지**(직교
    조작 설계만 답한다, 예: 같은 B에서 L만 바꿔 조성 `c`를 움직이는
    설계). 상세 `workspace/engine-port/results/bsweep_regime/
    PREREG_E1_BSWEEP_REGIME_2026-08-14.md`(2026-08-14 addendum, 사전등록
    자체는 여전히 미제출)·`E1A_ARTIFACT_ERRATA_2026-08-14.md`(아티팩트
    결함 4건, engine-porter/result-analyst 이관).

    ★★**(2026-08-15 addendum, doc-steward — caveat 정확화, 등급·수치
    변경 아님) caveat(ii) "job축 교락"의 정체 확인.** job **865533**은
    keepalive 프롬프트(`s8_keepalive_prompt_224.txt`, **1794**(★2026-08-16
    정정, 구 1793 — addendum3 참조) 토큰 >
    컨텍스트 상한 1792)가 Ha8 arm **전 5셀**에서 사실상 100% 거부돼
    (d16 23,662·d24 23,689·d44 23,705·d92 23,729·np 23,675건/셀,
    `s8_deconf_Ha8_C1024_d{16,24,44,92,np}_865533_srv.log`, 메인 세션
    직접 재확인) **prefill 동거가 죽은 런**이다(`CO_RESIDENT_frac` d44:
    865533 **0.349** vs 865493 **0.662**,
    `s8_deconf_Ha8_C1024_865{493,533}_result.txt:68`). caveat(ii)의
    "job축 교락"은 곧 **"동거 정상(865493) vs keepalive-사망(865533)"의
    교락**이라는 구체적 형태다 — 동거는 §1-1·§1-26이 이미 확립한 대로
    green-context 파티션이 **실현되는 조건 그 자체**다. 이것은 새
    사실이 아니라 §3 **항목28**(2026-08-03)의 "`n_indep=1`
    (865493↔865533은 keepalive 설정이 달라 replicate 아님)" 판정의
    **기전을 특정**한 것이다(당시는 "설정이 다르다"까지, 이번에
    **1794**(구 1793)>1792 토큰 초과라는 정확한 원인·크기 확인) —
    `k`·`b` 비식별
    판정 자체는 이 발견으로 **강화**되며 **등급 변경 아님**.
    ⚠️**별건 dated 정정(원문 미덮어쓰기)**: `results/s8_scaleup/
    NOTES_D54_ANCHOR_2026-08-03.md` Finding 1과 이를 그대로 복제한
    `PROJECT_STATUS.md` "8B decode-SM 민감도 측정 노트" G-1의
    "byte-identical to the one 865493 used"·"engine-tree churn between
    2026-07-27 and today[08-03]" 서술 — 865493의 전 20개 arm/cell
    srv.log 중 최종 파일은 **2026-07-27 23:05:55**에 끝나는데
    `s8_keepalive_prompt_224.txt`의 파일시스템 mtime은 그보다 **47분
    뒤인 23:53:11**이다(메인 세션 직접 확인) ⇒ **865493은 그 시각의
    파일을 쓸 수 없다**(job이 이미 종료). 865533(865493 종료 직후
    같은 날 밤 실행, Ha8 d44 srv.log 종료 2026-07-28 00:53:25)이
    이미 같은 붕괴를 보이므로, 깨짐 시점은 08-03이 아니라 **늦어도
    2026-07-27 23:53경**이다(원인 자체는 여전히 미규명). 두 문서
    모두에 addendum 필요 — `NOTES_D54_ANCHOR_2026-08-03.md`에는
    별도 addendum 절로, `PROJECT_STATUS.md` G-1에는 인접 정정
    문단으로 추가(아래 참조). 상세 `PROJECT_STATUS.md` "다음 실험
    gate" #12, `handoff-report/session_handoff_2026-08-15.md` §2.7.

    ★★★**(2026-08-15 addendum2, doc-steward — result-analyst
    `AUDIT_C2_HEADLINE_JOB_COMPOSITION_2026-08-15.md` 반영, 등급·수치
    변경 아님, "감사 대기[claims-auditor 병행 중]" 아님 — job 구성
    사실이라 판정 대상 밖) 바로 위 문단(같은 날 addendum) 자체의
    범위 오류 정정.** 위 문단과 `PROJECT_STATUS.md` G-1·
    `NOTES_D54_ANCHOR_2026-08-03.md`가 job 865533의 keepalive 붕괴를
    **"Ha8 arm 전 5셀"**로 적었다. 원자료 재확인 결과 **4 arm
    (Ha8·Hs8·M8·T8) 전 20셀 전부**가 같은 붕괴를 보인다 — srv.log
    거부 건수/셀: Ha8 23,662–23,729·Hs8 23,404–23,472·M8
    23,608–23,741·T8 23,409–23,431(`s8_deconf_{arm}_C1024_d{16,24,44,
    92,np}_865533_srv.log`), 클라이언트 확증(첫 rep, d16) 865533은
    4 arm 전부 `keepalive_done=0, keepalive_errors≈5,880–5,943`,
    865493은 4 arm 전부 `keepalive_done=199–528, keepalive_errors=0`.
    **어제(2026-08-15) 등재한 서술이 바로 다음 감사에서 범위 오류로
    드러난 사례** — 새 게이트를 신설하지는 않으나 기존 게이트 #25
    ("대형 캠페인 제출 전 배관 스모크")와 유사한 패턴("등재한 그날
    다음 감사에서 스코프가 틀렸음이 드러난다")으로 기록해 둔다. 판정
    함의는 없음(job 865533이 keepalive-사망이라는 사실 자체·`n_indep=1`
    판정·항목28/50의 등급은 전부 불변) — 바뀌는 것은 "어느 arm이"의
    범위뿐이다. 상세 `workspace/engine-port/results/s8_scaleup/
    AUDIT_C2_HEADLINE_JOB_COMPOSITION_2026-08-15.md` §3.2,
    `NOTES_D54_ANCHOR_2026-08-03.md` Addendum 2, `PROJECT_STATUS.md`
    G-1(追記2)·§3 항목54(신설, 이 감사의 본 결과).

    ⚠️**addendum3(2026-08-16, doc-steward — 메인 세션 실측, `s8_c2r.sbatch`
    H2 assert 구현 과정, 새 성능 판정 아님, 사소한 표기 정정)**: 위
    addendum·addendum2와 §1(항목29 B-1)·`PROJECT_STATUS.md`가 붕괴
    keepalive 프롬프트를 **"1793 토큰"**으로 적었으나, C2-R 하네스(H2,
    `n_tok ≤ CTXCAP−256` assert) 구현 중 arm별 토크나이저로 재측정한
    결과 실제 토큰 수는 **1794**다(M8·Ha8 동일값). **1792 초과라는
    결론·기전(HTTP 400 전량 거부)은 완전히 불변** — 숫자 표기만
    정정한다. 상세 `workspace/engine-port/results/s8_scaleup/
    PREREG_C2R_RULES_REV2_2026-08-15.md`, `PROJECT_STATUS.md` G-1
    dated정정3.
51. ★★**(2026-08-15, doc-steward 등재 — claims-auditor 산출, 메인
    세션 미재현[provenance 명시]) decode batch 도달성의 구조적 폐쇄 —
    실험 게이트로 등재, 성능 판정 아님.** ctx4096 텔레메트리 전수
    재집계: prefill 16 SM 고정 + `--chunked-prefill-size -1` 격자에서
    SM92 셀의 max(decode_bs)가 T8 4·M8 4·Hs8 4·Ha8 12로 무너진다
    (SM44는 T8 8·M8 8·Hs8 6·Ha8 16) — SM92에서 B≥9 도달률은 T8·
    M8·Hs8 **0.0%**, Ha8 **3.0%**뿐. ⚠️**provenance**: 메인 세션이
    독립 확인한 것은 이 중 **T8 행 하나**뿐(`workspace/engine-port/
    results/bsweep_regime/PREREG_E1_REV3_2026-08-15.md:150` "T8은
    d92에서 B가 4를 넘은 적이 없다[d44는 8]") — M8·Hs8·Ha8 행과
    도달률 %는 재현하지 않았다. 제시된 기전: prefill SM을 고정하면
    prefill 서비스율 `λ`가 상한이 되고 Little's law
    (`B_decode=λ·T_decode`)로 동시성을 올려도 B가 오르지 않는다
    (출력 토큰↑ 레버는 Zamba2 `max_position_embeddings=4096`이 막음).
    **등재 명제(앞으로의 모든 B축 설계에 적용)**: 이 기판(prefill SM
    고정·unchunked)에서는 decode batch를 제공 동시성으로 임의로
    끌어올릴 수 없다. E-1 계열 rev1–rev3(2026-08-15, 死因표)이 이
    제약으로 닫혔다 — rev4는 prefill SM을 풀거나 다른 B 통제 수단이
    선행돼야 한다. 상세 `PROJECT_STATUS.md` "다음 실험 gate" #12,
    `handoff-report/session_handoff_2026-08-15.md` §2.8·§4.2.

    ★★**(2026-08-15 addendum, doc-steward — result-analyst 1차 산출을
    claims-auditor가 적대 검증해 반증, 등재 명제 자체가 수정됨) ctx1024는
    "기전은 ctx-무관"의 반례다 — REFUTED, 정정된 명제로 교체.**
    result-analyst의 원 addendum(같은 날 앞선 초안)은 위 표(T8/M8/Hs8
    max(decode_bs)=4, Ha8=12, ctx4096 job 865311)에 "**수치만 ctx4096
    한정이고 기전(prefill SM 고정 → λ 상한 → Little's law)은
    ctx-무관**"이라 적었으나, claims-auditor의 독립 재집계가 이를
    **반증**했다: **ctx1024(job 865493/865533)의 SM92 실현 스냅샷
    max(decode_bs)는 865493 21/23/21/23, 865533 (전 arm) 16**으로,
    ctx4096의 4(T8/M8/Hs8)/12(Ha8)보다 훨씬 높다 — 즉 **같은 "prefill
    16 SM 고정" 기판에서 ctx만 바꿨는데 폐쇄가 사실상 풀린다**, 이는
    "기전이 ctx와 무관하게 항상 구속한다"는 서술을 정면으로 반박하는
    반례다. **정정된 명제**: B-폐쇄(prefill SM 고정 하 decode batch
    상한)는 **λ(L)이 작을 때만 구속한다** — 여기서 `λ(L)`은 prefill
    서비스율이고 L(prompt 길이)의 함수다. ctx4096(L 길다)처럼 prefill이
    느려 λ가 작으면 Little's law 상한이 낮게 걸려 폐쇄가 강하게
    구속하고, ctx1024(L 짧다)처럼 prefill이 빨라 λ가 크면 상한이
    실질적으로 안 걸린다. **이 정정은 "기전이 존재한다" 자체를
    부정하지 않는다**(ctx4096에서 관측된 폐쇄 자체는 유효) — 부정되는
    것은 "그 기전이 ctx에 무관하게 보편적으로 적용된다"는 일반화뿐이다.
    **등재 명제 개정**: "이 기판에서는 decode batch를 제공 동시성으로
    임의로 끌어올릴 수 없다"는 문장에 **"단, λ(L)이 이 폐쇄를 구속할
    만큼 작을 때만"** 을 필수 조건으로 추가한다 — ctx4096(job 865311)
    격자 밖으로 수치·기전 둘 다 무조건 이식 금지. 상세
    `workspace/engine-port/results/s8_scaleup/
    AUDIT_C2_HEADLINE_JOB_COMPOSITION_2026-08-15.md` §8 항목3(1차,
    반증됨) + claims-auditor 적대 감사(2026-08-15, 산출물 위치는
    아직 원자료 파일 미확정 — provenance만 등재, 재현 스크립트는
    N4/후속 실험으로 이관), `PROJECT_STATUS.md` "다음 실험 gate"
    #12(追記, 재정정).
52. ★**(2026-08-15, doc-steward 등재 — 메인 세션이 벤더 문서·메트릭
    DB·아카이브 로그로 직접 확인) ncu/nsys 도구 사실 4건 — "커널
    단위 측정 0건"에 운영점·green-ctx 스코프 주석, 성능 판정 아님.**
    (1) `launch__waves_per_multiprocessor` 메트릭 설명(`ncu
    --query-metrics-collection launch --chip ga100`, 설치본
    2025.3.1.0, 드라이버 580.105.08, 메인 세션 직접 재확인): "When
    using green contexts, this metric is scaled with the number of
    SMs used by the green context." — green ctx 하에서도 wave 지표를
    그대로 쓸 수 있다(요건 ncu 2024.3+/드라이버 560+, 충족). (2)
    `nsys profile --help`(설치본 2025.3.1.0, 메인 세션 직접 재확인)의
    `--cuda-graph-trace` 기본값은 `graph`이고 "node activities will
    not be collected" — **운영점(cudagraph-ON)에서 커널 노드를 보려면
    `=node` 명시가 필요**. (3) `--exclusive`는 ncu 요구사항이 아니다
    — 직렬화 락은 per-device, GPU는 `--gres=gpu:1`로 이미 할당되며
    카운터 게이트 `hwperf`는 A100 노드 전체의 기본 feature(`sinfo`:
    `gpu[30-33,36-43] A100-80GB_8,hwperf`, 메인 세션 직접 재확인),
    아카이브 ncu 로그에 `ERR_NVGPUCTRPERM` **0건**(메인 세션 직접
    재확인). (4) 선행 ncu 시도가 **이미 있다** —
    `workspace/characterization/src/profiling/ncu_runner.py` +
    아카이브 **8 job**(`logs/archived/ncu_{717984,720116,724102,
    726120,727029,729105,735985,735986}.*`), `error code 9`
    **1,986건** + 메트릭 정규식 실패 **1,920건**(메인 세션 직접
    재확인). `_ncu_target.py:73-74`가 "ncu profiling always runs at
    full GPU"라 명시하고 wave를 해석적으로 계산 ⇒ **green context
    하에서 잰 적이 없다**. ⇒ `PROJECT_STATUS.md` "8B decode-SM 민감도
    측정 노트"의 "커널 단위 측정이 0건이라 미식별"에 **"운영점·
    green-ctx 한정으로 참이며, full-GPU 합성 커널 프로파일링은
    2026년 초 시도돼 대량 실패했다"** 스코프 주석을 단다 — "미실행"과
    "시도했으나 실패"는 다른 명제다. 상세 `PROJECT_STATUS.md` "8B
    decode-SM 민감도 측정 노트", `handoff-report/session_handoff_
    2026-08-15.md` §2.9.

    ★★★**追記(5) (2026-08-16, doc-steward 등재 — kernel_mech rev2 설계
    감사[claims-auditor, 판정 NO-GO] F3에서 발견, 새 성능 판정 아님)
    정본 자신의 인용 결함 — 위 (4)가 인용한 문장의 다섯 줄 위를
    누락했다.** 위 (4)가 인용한 "ncu profiling always runs at full
    GPU"(`_ncu_target.py:73-74`)는 **그 이유가 다섯 줄 위(68-71)에
    적혀 있는데 정본이 지금까지 그 이유를 인용하지 않았다**:

    > ```
    > # Do NOT call smctrl.set_sm_count() here.
    > # CUPTI (the API ncu uses for hardware counter collection) is incompatible with
    > # CUDA Green Contexts. Restricting SMs via Green Context while ncu is attached
    > # causes "Failed to prepare kernel for profiling / Unknown Error on device 0."
    > ```
    > (`workspace/characterization/src/profiling/_ncu_target.py:68-71`,
    > 메인 세션 직접 재확인)

    "ncu는 full-GPU에서만 돈다"는 *design intent*(결과)이고, 위
    인용이 그 *원인*(CUPTI×green-context 비호환)이다. **참이면
    kernel_mech Stage B(green-context 하 커널 단위 프로파일링)는
    이 기판에서 구성상 불가**하다 — 이 정정은 위 (1)–(4)가 세운
    "green-ctx·운영점 하의 커널 프로파일링은 시도된 적 없다"는
    결론을 뒤집지 않는다(그건 여전히 참이다), 다만 "시도하면
    될 것이다"로 읽는 것을 막는다.

    ★★**단, 아직 미확인 리스크로만 등재한다** — 같은 `error code 9`
    (위 (4)의 1,986건)에 저장소가 **3가지 경합 귀속**을 갖고 있다:
    (a) 메트릭 이름 불일치(`run_ncu_profile.py:213`) (b) cuda12 타겟
    (`run_ncu_profile.sh:37-41`) (c) CUPTI×green-context(이 문단).
    판별기는 **Stage 0′ P1 프로브**(kernel_mech rev2 §8, ≈GPU 0)이며
    아직 실행되지 않았다. 발화 시 라벨은 **`UNAVAILABLE (CUPTI×
    GREEN-CONTEXT)` = 도구 한계**이지 게이트 실패도 기전 증거도
    아니다(방법론 게이트 #21). 상세 `PROJECT_STATUS.md` "8B
    decode-SM 민감도 측정 노트" 정정 배너, `workspace/engine-port/
    results/kernel_mech/DESIGN_KERNEL_MECH_REV2_2026-08-16.md` F3,
    `handoff-report/session_handoff_2026-08-16.md` §4-4(b).

    ★★★★**追記(6) (2026-08-20, 메인 세션 — P1 프로브 job 886718
    실행 결과, 새 성능 판정 아님, 도구 타당성 판정) 위 追記(5)의
    "판별기"가 실행됐다: `UNAVAILABLE (CUPTI×GREEN-CONTEXT)`.**
    `--exclusive --constrain=hwperf`, node gpu38, 1분55초
    (`COMPLETED 0:0`, 동반 프로브 886752 포함 GPU 지출 0.032
    GPU-hr). greenctx 다리에서 위 追記(4)가 인용한 시그니처
    (`Failed to prepare kernel for profiling` / `Unknown Error on
    device 0` / exit 9)가 **정확히** 재현됐다. **두 겹의 대조**로
    (c) 귀속이 (a)·(b)로부터 깨끗이 분리된다: (i) **다리 간** —
    control(같은 GEMM `ampere_bf16_s16816gemm_bf16_128x256_ldg8_
    f2f_stages_64x3_nn`, full GPU)은 에러 0건·60행 정상 수집. (ii)
    ★**다리 내부** — 같은 프로세스·같은 ncu 호출에서 green ctx
    **밖**(`set_sm_count()` 이전 launch)의 RNG 커널
    (`distribution_elementwise_grid_stride_kernel`)은 8행 성공
    수집됐고, green ctx **위** GEMM만 실패했다 — job·노드·할당·권한
    (control 성공이 `ERR_NVGPUCTRPERM` 부재를 확증)·ncu 호출·메트릭·
    클럭 정책(`--clock-control none`)·타깃 스크립트·커널·차원이
    전부 동일했다. ★★2026-08-21 정정(doc-steward, 원문 P1_VERDICT §2
    대조): 원문 표현은 **"두 겹의 대조가 같은 방향을 가리킨다"**
    이지 "변인은 하나뿐"이 아니다 — 다리 간(i) 대조는 `realized_sm`
    16 vs 108도 함께 바뀌고, 다리 내부(ii) 대조는 커널 종류 자체가
    다르다(RNG 초기화 vs GEMM)라서 어느 쪽도 변인을 하나로 완전히
    좁히지 못한다("유일한 변인은 … 뿐이었다"는 이 배너의 과잉
    인용이었다, `PROJECT_STATUS.md`에도 동일 과잉이 있어 함께 정정
    했다). ⇒ **(c) CUPTI×green-context가 이 구성에 한해 관측 근거를
    얻었다** — `_ncu_target.py:68-71`은 더 이상 코드 작성자의
    주장이 아니다.

    ★**지킬 서술 한계(overclaim 금지)**: (1) 이건 **도구 타당성
    판정이지 성능 판정이 아니다**. (2) 깨진 것은 **프로파일링이지
    green context 실행이 아니다** — `realized_sm=16`으로 정상
    실현됐고 커널도 실행됐다(동반 프로브 886752는 `DONE`까지
    완주). (3) **내부 기전은 미분리**다 — 관측은 "green-context
    스트림 위 커널의 프로파일링이 실패한다"까지이고, 그 실패가
    CUPTI 비호환인지 스트림/컨텍스트 처리의 다른 층인지는 이
    프로브가 가르지 않는다. `UNAVAILABLE (CUPTI×GREEN-CONTEXT)`
    라벨은 하네스가 붙인 이름이다. (4) **스코프 한정** —
    A100-SXM4-80GB·driver 580.105.08·ncu 2025.3.1.0·CUDA 13.0.2·
    이 클러스터(`amd_a100nv_8`) 한정, 다른 버전·기판으로 이식
    금지. (5) 위 (a)·(b) 경합 귀속(메트릭 이름 불일치·cuda12 타겟)은
    **여전히 유효** — 옛 `workspace/characterization/` 아카이브
    8-job 배치(전부 full GPU, cuda12 venv 불일치)를 설명하는
    것으로 남는다. 이번 P1은 모듈을 표준화한 동일-툴킷 구성에서
    green-context만 격리했으므로 (a)·(b)를 반증하지 않는다 — 서로
    다른 실패 배치에 대한 설명이다.

    **함의**: (i) ⇒ **`kernel_mech` rev3는 Stage B를 폐기하고
    Stage A 전용으로 범위를 좁힌다** — GO 경로였던 "문서 수정 8건"
    중 Stage B 대상 최소 5건(메트릭 교체·부착 구조 재작성·클럭
    제어 이관 등)이 적용 대상 소멸(4세션 이월의 실질 원인 해소).
    (ii) **트리의 wave 수치는 이 기판에서 원리상 측정일 수 없다**
    — kernel_mech rev2 감사 F1(`wave_eff`는 ncu 메트릭이 아님/폐기
    선언한 수제 유도를 1차 결정량으로 되살림)을 독립적으로
    뒷받침한다. (iii) 부수 확정 — 선례 스크립트 `run_ncu_profile.sh:
    15-17`의 권한 근거("SLURM job 환경이 카운터를 켜준다 — batch면
    interactive와 달리 열린다")는 **불충분**하다: 동반 프로브
    886752(non-exclusive **batch**)가 `ERR_NVGPUCTRPERM`을 받았다
    — 실제 구분선은 **exclusive(+hwperf)**다. 아카이브 ncu 로그
    8건 전수 재확인: `ERR_NVGPUCTRPERM` 0건은 전부 exclusive+hwperf
    계열이고, 그중 실제 데이터를 수집한 유일한 성공 사례
    (`ncu_729105`, 88행)에도 green/`set_sm_count`/`--sm-counts`
    언급이 **0건**이다 — ⇒ **이 저장소는 green context 하에서
    카운터를 수집한 적이 한 번도 없었다**(P1이 재탕이 아니라 진짜
    미해결 질문이었다는 확인). 상세 `PROJECT_STATUS.md` "8B
    decode-SM 민감도 측정 노트" 실증 확정 배너(2026-08-20)·"다음
    실험 gate" #11 레지스트리 kernel_mech P1 프로브 행·#17
    2026-08-20 갱신, `workspace/engine-port/results/kernel_mech/
    p1_probe/P1_VERDICT_2026-08-20.md`,
    `P1_886752_REVIEW_2026-08-20.md`.

53. ★**(2026-08-15, doc-steward 등재 — E-1 rev1–rev3+kernel_mech 4회
    설계 전부 감사 차단에서 도출, 새 실험 아님) 방법론 게이트 #35·
    #36 신설.** **#35 "개정판에서 손잡이 값을 유지한 채 유도 서사만
    바꾸지 마라."** rev2의 `δ=0.0610/3=0.0203`이 rev3에서 "오차예산"
    유도로 갈아 끼워졌으나 숫자는 `δ=0.020`(반올림)으로 그대로였다
    (`workspace/engine-port/results/bsweep_regime/PREREG_E1_REV2_
    2026-08-15.md:202`·`PREREG_E1_REV3_2026-08-15.md:194`, 메인 세션
    직접 재확인) — 그 0.0610 자체가 §3 항목50이 지금 오염원으로
    특정한 A의 앵커다. **#36 "타당성은 도구 문서·메트릭 DB로 확인한
    뒤 설계하라."** kernel_mech §3이 항목52(1)이 반증한(즉 존재하지
    않는) green-context wave-분모 오염을 피하려 수제 카운터 층으로
    내려가 `wave_eff≡1` 항등식을 만들었다(방법론 게이트 #9 여덟
    번째 재발) — 확인 비용은 로그인 노드 명령 1줄, GPU 0이었다.
    메모리 `deconfound-measurement-lessons.md` 항목35·36과 대응.
    상세 `PROJECT_STATUS.md` "방법론 게이트" #35·#36,
    `workspace/engine-port/results/bsweep_regime/PREREG_E1_REV{2,3}_
    2026-08-15.md`(미커밋), `workspace/engine-port/results/
    kernel_mech/`(미커밋).
54. `ADVERSARIAL AUDIT COMPLETE (2026-08-15) — claims-auditor 판정:
    등급 유지 + 인용 정지 2건 신설.`
    ★★★**(2026-08-15, result-analyst 1차 산출 + claims-auditor 적대
    검증, doc-steward 등재, 메인 세션은 독립 재확인 없음) C2
    헤드라인(2.36–2.91×, 4 arm) job 구성 감사 — 등급 CONFIRMED(scoped)
    유지, 신규 인용 정지 2건, 새 성능 판정·정책 주장 아님.**
    result-analyst가 `FINDINGS_8B_2026-07-28.md`의 헤드라인 4셀 각각의
    job 구성을 원자료(telemetry+raw ITL+`t0_monotonic_s`+srv.log)에서
    재구성했고, claims-auditor가 그 산출을 적대 검증해 **일부는
    확증, 일부는 반증**했다.

    **[생존 — claims-auditor가 반증 시도했으나 실패, 그대로 인용 가능]**
    - **헤드라인 4셀 중 3셀(Ha8·M8·Hs8)은 양 다리 모두 job 865533
      단독**, **T8도 SM92 다리는 865533 단독**(SM16 다리만
      865493 61%+865533 39%, 두 job 중앙값 차 +0.14%). 1차 원인은
      keepalive 붕괴가 아니라 **하네스 결손**(865493의 M8 전 5셀·
      Ha8 d16 셀에 `t0_monotonic_s` 부재, 클라이언트 패치가 job
      도중 적용됨) — 데이터가 오염된 게 아니라 애초에 존재하지
      않는다.
    - **58개 매칭 셀의 교락 크기**: 조건부 ITL 중앙값 차 평균
      −0.18%±0.62%(SD), 최대 |2.20%|. claims-auditor가 (i) 헤드라인
      다리만 재집계(−0.00%±0.58%) (ii) `tok_idx` 매칭(비 변화 ≤0.2%)
      (iii) 동거 prefill 강도(`pf_bs`) 매칭(≤1%) 세 가지로 추가
      반증을 시도했으나 **모두 실패** — 이 수치는 항등식이 아니며
      생존한다.
    - 독립 스크립트 재실행 결과가 저장소 JSON과 **byte-identical**
      (2026-08-14 E-1a식 아티팩트 결함 없음).

    **[반증됨 — 아래 (D)(E)(F)로 대체]**
    - ~~"교락은 작으므로 깨끗한 런 단독 재계산 T8 2.388×[2.379,2.397]·
      Hs8 2.687×[2.686,2.689](양 다리 4 rep)를 CI와 함께 인용 가능"~~
      → **REFUTED, (D) 참조.**
    - ~~"Ha8은 b12 슬라이스가 양 다리 모두 0건이라 FINDINGS §2
      지침을 구조적으로 만족할 수 없다"~~ → **귀속 오류, (E) 참조.**
    - ~~"위 표(gate #12, ctx4096)의 수치는 ctx4096 한정이나 기전은
      ctx-무관"~~ → **REFUTED, 항목51 追記(재정정) 참조.**

    **(D) ★신규 인용 정지 1 — arm별 ε 및 arm 간 순위·격차.**
    헤드라인 슬라이스로 계산한 ε(=ln(ratio)/ln(5.75)) 순서 **T8
    .492 < Hs8 .543 < Ha8 .599 < M8 .610**은 **슬라이스 선택의
    산물**이다. 4 arm이 전부 존재하는 유일한 공통 batch(b=1)에서는
    **Ha8 .480 < Hs8 .514 < M8 .515 < T8 .518**로 **순서가 완전히
    뒤집히고 폭이 0.118→0.038로 좁아진다**. `FINDINGS_8B_2026-07-28.md`
    §2.1 자신이 "Ha8의 SM 민감도는 오히려 최저(2.54–2.75×)"라 적어
    헤드라인 순서와 모순한다. ⇒ **arm별 ε 값과 arm 간 순위/격차는
    이 시점부터 인용 정지** — "레버 존재, 4 arm 전부"까지만 인용
    가능하고 "어느 arm이 더 민감한가"는 인용 금지. ⚠️**하방 소비자
    추적 필요**(아직 미확인, 다음 세션 과제): `bsweep_regime/`
    E-1a의 `T6_PC1`이 이 ε 순서를 소비 중이고, `reports/figures/
    canon.py:308-335`도 확인 대상.

    **(E) ★신규 인용 정지 2 — 깨끗한 셀 CI·"n=4" 표기.** 원 산출의
    "T8 2.388×[2.379,2.397]·Hs8 2.687×[2.686,2.689], 양 다리 4 rep"는
    **CI·"n=4"로 인용 불가**. 이유: cluster는 (job, cell, rep)인데
    T8 b16의 865493/d16 rep1–4는 **서버 부팅 1회**에서 나왔다 — 즉
    **다리당 `n_indep=1`이고 rep1–4는 의사반복(pseudo-replication)**
    (같은 부팅·같은 파티션 실현 이벤트, 독립 시행이 아님). 보고된
    CI [2.379,2.397]은 토큰/rep 내부 잡음만 잡은 것이며, rep 간
    변동을 t(3)로 반영하면 **≈[2.368,2.408](±0.85%), 약 2배 넓다**.
    원 산출 §6의 "양 다리 4 rep = 게이트 n≥4 충족"은 **거짓**이다.
    ⇒ **점추정 T8≈2.388×·Hs8≈2.687×는 `n_indep=1` 명시 하에 인용
    가능**하나, CI·"n=4"·"게이트 충족" 표기는 인용 정지. "2.39–2.69×
    (2/4 arm)"이라는 문구를 마치 제대로 된 구간처럼 적지 말 것.

    **(E') Ha8 batch 도달성 — 귀속 정정(구조적 제약 아님, 사실만
    기술).** Ha8 SM16 다리는 job 865533(붕괴 런)에서 최대
    batch=11까지만 관측되고(b12–16 0건), 865493(깨끗한 런)은 d16
    셀 자체에 `t0_monotonic_s` 결손이 있어 대응 데이터가 아예 없다.
    **SM92 다리는 오히려 865493(깨끗한 런)이 batch=16까지
    도달한다**(33,344건, 전부 865493 공급). 즉 batch 12–16 슬라이스가
    없는 것은 architecture/기판의 **구조적 도달 불가가 아니라**
    (i) 865533의 낮은 batch 상한(붕괴 job) + (ii) 865493 SM16
    다리의 t0 결손(하네스 문제)의 조합이다 — "865533에서 미도달,
    865493에서는 SM92 다리가 b16 도달"로만 서술한다.

    **(F) ★신규 caveat — 조건화 자체가 붕괴 경로를 제거한다.**
    realized-SM으로 조건화하면 동거율이 **구성상 ≈1로 강제**된다
    (green-ctx 분할은 동거 중에만 실현되므로, §1-1·§1-26 기존
    확립). 즉 58개 매칭 셀은 **붕괴의 주 경로가 조건화로 이미
    제거된 뒤의 잔차**다 — 조건화가 안 걸린 SM108 셀에서는 실제로
    **−2.2%/−1.26%**의 차이가 관측된다. 또한 58개 매칭 셀 중
    **헤드라인 다리(SM16/SM92) 자체는 17개뿐**이고(M8 0개,
    Ha8-SM16 0개), 나머지는 SM24/44 등 비-헤드라인 파티션이다.

    **(G) 이질적 batch 추출 규칙(신규 확인).** T8/M8/Hs8 헤드라인은
    b12를 쓰고, Ha8은 **자신이 도달한 최댓값 b9**(=Ha8 자신의 최대
    비)를 쓴다. 비는 전 arm에서 batch에 단조증가(weight-sweep
    포화 방향)하므로 **"2.36–2.91×"는 이질적 batch-추출 규칙의
    나열**이지 매칭 비교가 아니다. b12에서 실제로 매칭되는 3
    arm(T8/M8/Hs8, 이미 헤드라인 값과 동일)의 범위는 **2.366–2.909
    (23% 폭)**. 4 arm 공통 batch(b=1)의 2.31–2.48은 **"수렴 확증"으로
    과잉 해석 금지** — b=1은 weight-sweep 트래픽이 지배적인 지점이라
    (§7 C-2) arm 간 근접이 **준-항등**에 가깝다(독립 확증 아님).

    **(H) 해소 실험(등재, 미실행)**: keepalive 프롬프트 ≤1792
    토큰(context 상한 이하) + 4 arm × {d16,d92} × **독립 서버 부팅
    4회씩**(rep 아님, 셀당 재부팅) + concurrency 계단으로 b≥12
    강제 — job-불변성·batch 매칭·`n_indep≥4`를 한 설계로 동시에
    해소한다.

    **(I) errata(신규).** 원 산출 보고서의 tok_idx 민감도 표·rep별
    중앙값 SD 표·§4.2(a) 교차표·§8.3 ctx1024/4096 비교 수치 4종은
    저장소에 남은 `audit_c2_job_composition.py`가 **그대로 산출하지
    않는다**(수동 계산/전사로 추정) — 다음 재현 시도가 코드 부재를
    발견하기 전에 미리 기록.

    **(J) Stage 0 D108 전례와의 구분.** 이 사례는 Stage 0(§1-21,
    C1 CONFIRMED)와 **다른 실패 모드**다 — Stage 0은 **라벨 오류**
    (D108이라 적힌 두 arm이 실은 같은 조건)였고, 여기는 **라벨이
    맞고**(865493/865533·SM16/SM92가 실제로 다른 job/파티션) 교락이
    **직접 측정**됐다. 두 사례를 같은 유형으로 인용하지 말 것.

    **종합 판정 (claims-auditor, 2026-08-15)**: **(ii) 더 강한 조치
    필요, 단 부분적** — "레버 존재, 2.3–2.9× 대역, 4 arm 전부"는
    **CONFIRMED(scoped) 유지**. 여기에 **인용 정지 2건 추가**: (a)
    arm별 ε·arm 간 순위/격차 (D), (b) 깨끗한 셀 CI·"n=4" 표기 (E).
    등급 자체는 안 바뀐다 — 인용 가능 범위만 좁아진다.

    ⚠️★**provenance(최종)**: 이 항목은 **claims-auditor 적대 감사
    완료(2026-08-15)** 상태다 — "감사 대기"가 아니다. 산출 주체는
    **result-analyst(1차)** + **claims-auditor(적대 검증)**이고,
    **메인 세션(doc-steward)의 독립 재확인은 없다**. 상세
    `workspace/engine-port/results/s8_scaleup/
    AUDIT_C2_HEADLINE_JOB_COMPOSITION_2026-08-15.md`(1차, 위 반증된
    3항목 포함 원문 보존)(+ `audit_c2_job_composition.py`·
    `audit_c2_job_composition_2026-08-15.json`), claims-auditor
    적대 검증 결과(원자료 파일 위치 미확정 — 다음 세션이 저장소에
    편입할 것), `PROJECT_STATUS.md` "8B decode-SM 민감도 측정
    노트"(追記, 재정정)·"다음 실험 gate" #12(追記, 재정정)·
    `FINDINGS_8B_2026-07-28.md` §8(신설, 재정정)·`reports/paper/
    CLAIM_EVIDENCE_MATRIX.md`(Claim A 각주, 追記, 재정정).
55. ★**(2026-08-16, doc-steward 등재 — 메인 세션 진단, ⚠️claims-auditor
    감사 없음 — "현재 작업가설"로 인용, 새 실험 아님) 방법론 게이트 #37
    신설: 적대 감사에는 합격 기준과 단일 판정 질문을 함께 줘라.**
    C2-R 사전등록이 5연속 감사 차단(E-1 rev1/rev2/rev3·kernel_mech·
    C2-R rev1, 방법론 게이트 #34·#35·#36)을 끝내고 **rev2에서 처음
    GO**를 받았다(2026-08-15, claims-auditor, `PROJECT_STATUS.md`
    "다음 실험 gate" #11). 차이는 설계 개선만이 아니라 **감사 의뢰
    방식** — 감사 범위를 단일 질문("이 설계가 정지 (b)에 답하는가")
    으로 한정하고 그 외 발견을 NO-GO가 아니라 caveat로 접수하도록
    바꿨다. 결과: caveat **10건**이 나왔고 그중 **5건**이 하네스를
    바꿔 차단이 아니라 개선으로 흡수됐다. ⚠️**부수 진단(등재의 핵심
    절반)**: rev1이 5표적 NO-GO를 받은 진짜 원인은 감사가 과했던
    것이 **아니라** 메인 세션이 실제 결손(2 arm, M8·Ha8)보다 훨씬 큰
    설계(4 arm ε-순서 캠페인)를 냈던 것이다 — 감사는 그 과잉을
    걷어냈을 뿐이고, GPU 지출 없이 재려던 것 5개(b*=16·부팅 분산·
    PIN 통과율·D3 계산가능성·백엔드 바운드 부재)를 **기존 데이터로
    답해줬다**. ⇒ 이 게이트가 말하는 것은 "감사를 약하게 하라"가
    **아니라** "감사에 판정 기준을 주라"다. **항목49(게이트 #34,
    규칙→하네스 2단 감사 순서)와의 구분**: #34는 **언제** 감사하는지
    (순서)를 다루고, 이 게이트는 한 번의 감사 요청 자체의 **스코프
    경계**(단일 질문+합격기준+caveat 라우팅)를 다룬다 — 다른 축이라
    별도 번호(#37)로 등재한다. 실무 규칙: 적대 감사 의뢰 시 (i) 설계가
    답해야 하는 단일 판정 질문을 명시, (ii) 그 질문의 합격 기준
    (임계값·통과 조건)을 함께 제공, (iii) 질문 범위 밖 발견은 NO-GO가
    아니라 caveat로 접수하도록 요청한다. ★**provenance**: 메인 세션
    (doc-steward) 직접 진단이며 **claims-auditor 감사를 거치지
    않았다** — 항목49·54 등 이 문서의 다른 §3 항목과 달리 적대 검증
    미완료 상태로 등재한다. 메모리 `deconfound-measurement-
    lessons.md` 항목38과 대응. 상세 `PROJECT_STATUS.md` "방법론
    게이트" #37, `handoff-report/session_handoff_2026-08-16.md` §4-1.
56. `C2-R CAMPAIGN COMPLETE (2026-08-16) — jobs 883574(M8)/883575(Ha8),
    result-analyst 분석 + claims-auditor 적대 감사 완료.`
    ★★★**(2026-08-16, doc-steward 등재 — result-analyst 산출 +
    claims-auditor 적대 검증, 메인 세션은 운영 지표만 직접 확인) M8·Ha8의
    신규 점추정 2건 등재, 기존 값 교체 아님 — C2 등급 무변경
    (CONFIRMED scoped), 인용정지 (a)(b) 둘 다 유효.** 사전등록
    `PREREG_C2R_RULES_REV2_2026-08-15.md`(rev1 NO-GO 5표적 → rev2 GO,
    §3 항목55/방법론 게이트#37 참조) 집행 결과.

    **(A) 점추정·구간(등재 가능 범위 한정) — ★정본 인용 문구
    (claims-auditor 지정, 그대로 채택, percentile CI를 t(5)로 대체)**:

    > `r_M8(16) = 3.058`, `r_Ha8(16) = 3.114` (ctx1024, realized
    > SM16/SM92, `decode_bs=16`, job 883574/883575, gpu43, 1시간,
    > `n_indep=6` 부팅). 동반 구간은 **within-job 부팅 구간이며 재현
    > 불확실성이 아니다** — 보수적으로 **t(5) [3.056, 3.060] · [3.077,
    > 3.155]**를 쓰고, **job/node/날짜 축은 미측정(n=1)**임을 병기한다.

    ★**percentile 부트스트랩 CI([3.0566,3.0595]/[3.0848,3.1354])는
    정본 본문에 쓰지 않는다** — claims-auditor 실측 결과 이 값들이
    **일관되게 과소피복**한다(부팅-평균 비의 t(5) 구간 대비 M8
    1.36×·Ha8 1.55× 더 좁음). percentile 값은 **원자료 JSON
    (`C2R_RESULTS_2026-08-16.json`)에 있다는 포인터로만** 남긴다.
    **최외곽 통계 단위는 boot이지 job/node/day가 아니다** —
    between-job SD는 **미측정**이다.

    **(A′) seed 민감도(등재 가능, 짧게)**: `seed∈{1,2,3,99}`에서 M8
    CI95 [3.0566,3.0595/6]·Ha8 CI95 [3.0843–3.0850,3.1354–3.1356], SD
    변동 ≤1.5% ⇒ **부트스트랩 seed는 결과를 만들지 않는다**(사전등록
    seed=1 선택은 무해). ⚠️**이것을 "CI가 견고하다"로 읽지 마라** —
    seed 안정성은 **재표집 잡음**만 배제할 뿐, 문제는 **재표집되는
    모집단**이다(최외곽 단위가 boot이고 job/node/day 축이 통째로
    빠져 있다). "seed에 안정적"과 "population을 대표한다"는 다른
    명제이며, 이 구분은 방법론 게이트 #9(항등식/자기확인) 계열의
    오독을 막기 위한 것이다.

    **(B) D3 사실(보고 전용, 게이트 아님)**: Ha8은 실현률 0.80에 12/12
    셀 전부 미달한다(d16 0.508–0.765, d92 0.735–0.792). 865493의
    Ha8(.682/.760)도 같은 대역 ⇒ **캠페인 결함이 아니라 arm의 성질로
    보인다.** 감사 N5("실현률 80%+ 조건에서의 비")는 **여전히 미해결**.

    **(C) ★★★양성대조가 항등식이었다 — 방법론 게이트 #9 아홉 번째
    재발(§3 항목18 追記 참조).** `s8_c2r_score.py`의 `cmd_poscontrol`
    (:270)이 `score(discover("legacy", jobs=jobs))`를 **`keep_slack`
    인자 없이** 호출한다 — `score()`/`_intervals()`의 기본값은
    `keep_slack=False`인데, 헤드라인 경로 `cmd_c2r`(:333)은
    `score(discover("c2r", jobs=jobs), keep_slack=True)`를 쓴다. 즉
    대조는 **legacy 분기 + `keep_slack=False`** 코드 경로만 실행했고
    **헤드라인이 실제로 쓰는 c2r 분기 + `keep_slack=True` 경로는 한
    번도 실행하지 않았다.** 게다가 대조의 표적값(T8 2.388·Hs8 2.687)
    자체가 `s8_batch_matched.py`의 또 다른 verbatim 복사본
    (`audit_c2_job_composition.py`)의 산출물이다 — 즉 "독립 검증"이
    같은 estimand 루프의 세 번째 복사본이 첫 번째 복사본과 일치하는지를
    확인한 것에 가깝다. **배제된 오류는 전사(transcription)·t0 조인·
    pooling 로직뿐**이며, 그 공백(헤드라인 경로 자체의 검증)을 실제로
    메운 것은 이 대조가 아니라 **claims-auditor의 독립 재구현**이다
    (해당 코드를 import하지 않고 새로 짠 스크립트가 3.057990/3.114061
    및 24개 부팅의 n·median을 전부 재현) — 이 provenance를 정확히
    적을 것.

    **(C′) ★provenance errata — 재현 경로가 저장소에 없다.**
    claims-auditor의 독립 재구현 스크립트(`indep.py`·`legacy.py`)는
    **스크래치에만 있고 저장소에는 없다** — "claims-auditor가
    3.057990/3.114061을 독립 재현했다"는 사실 자체는 성립하지만,
    **그 재현 경로가 저장소에 보존돼 있지 않다.** 2026-08-14 E-1a의
    `VERDICT`/`T6_PC1` 미산출 전례(`E1A_ARTIFACT_ERRATA_2026-08-14.md`)와
    같은 형태다 — **헤드라인 경로를 검증한 것은 저장소 밖 재구현이며,
    그 코드는 보존되지 않았다.**

    **(D) ★엔진 빌드 정정**: 865493(정본 헤드라인 앵커) 대비
    `runtime_source_manifest*.sha256`가 **11→15 파일**로 늘었고, 추가
    3종이 hot path다 — `scheduler.py`(HOLB hook을 `run_batch` 3경로에
    삽입, 커밋 `4e4e01d`, 2026-08-07) · `holb_probe.py` ·
    **`zamba2.py`**(= Ha8 자신의 forward, 커밋 `7de5336`, 2026-08-06).
    865493 시점 해시는 **무증명**. ⇒ **"C2-R과 865493의 차이는 플래그
    2개뿐"이라는 서술은 거짓**이며, 두 캠페인을 섞는 모든 비교(아래
    (F) 포함)의 각주에 이 3파일을 명시할 것.

    **(E) guard 고원 — C2 전반에 적용되는 estimand 성질.** `r`은
    guard∈[2.5,∞)에서 불변(Ha8 3.1141→3.1133→3.1132), M8은
    guard∈[1.0,3.0]에서 완전 무감이다. 진짜 한계는 반대쪽이다 —
    **guard≤0.75면 셀이 빈다.** 정본 estimand의 채택 구간
    (`GUARD=3.0`)이 전부 **0.7–6.3초 미관측 창 안**에 있다 ⇒
    **(16,16) 점 자체가 관측이 아니라 보간**이다. 이는 C2-R만의
    성질이 아니라 정본 estimand(`GUARD=3.0`, `s8_batch_matched.py`
    이래 전 C2 캠페인 공통)의 성질이므로, C2를 인용할 때 전반적으로
    병기한다.

    **(F) 전-구간 강건성**: 조건화(guard/batch 등) 없이 계산해도
    `r_M8=3.055`(−0.1%)·`r_Ha8=3.106`(−0.3%) — 위 점추정과 사실상
    같다.

    **[등재 금지 — claims-auditor가 명시적으로 막음]**

    - ❌ **"C2-R 값이 정본 2.36–2.91×보다 위"** — **범주 오류.** C2-R의
      d16 다리는 `b∈{1,15,16}`에만 존재하고 **M8 b12·Ha8 b9가 통째로
      없어** batch/job 분해가 불가능하다. 단조성도 국소 위반 4건(M8
      865533 b11 2.9309>b12 2.9091; C2-R b15 3.0719>b16 3.0580).
      ⇒ 새 값은 기존 값을 **대체가 아니라 병기**한다 — "위/아래"
      서술 금지.
    - ❌ **인용정지 (b) 해제** — **범주 오류.** (b)는
      `PROJECT_STATUS.md` "8B decode-SM 민감도 측정 노트"·
      `CONSENSUS.md`의 **T8·Hs8 깨끗한 셀**(CI·"n=4" 표기) 인용에
      걸린 것이고, C2-R은 T8·Hs8을 재측정하지 않았다(사전등록 §1 —
      애초에 재측정할 이유가 없음). 새로 생긴 것은 **M8·Ha8의 새 인용
      대상**뿐이다. **(b)는 그대로 유효.**
    - ❌ **N1/N2 해결** — b=1 증거는 준-항등 체제(§7 C-2, weight-sweep
      지배) · 대조 상대가 붕괴 job 865533 · 셀이 분모 다리일 뿐이며,
      헤드라인 SM16 다리는 두 arm 다 교차-job 대조 **0건**이다.
      검정력도 분해능 ≈1%(§6.1의 b=1·SM92 재현 폭) = M8 CI의 약
      20배. **N1/N2는 여전히 미해결.**
    - ❌ **C2 등급 변경** — **CONFIRMED(scoped) 그대로.**

    **(G) 다음 실험 gate(신설, `PROJECT_STATUS.md` "다음 실험 gate"
    #13)**: (1) job/node 축 — 동일 커밋·15파일 매니페스트로 ≥4 job×
    ≥3 노드×≥2 날짜, arm M8·Ha8, d16/d92, b16, job당 3부팅(다리당
    12). 1차 결정량 = **between-job SD of r**. **이게 없으면 어떤 CI도
    인용 불가.** (2) batch vs job 분해 — 한 job 안에서 conc 계단
    (4/8/12/16)으로 b=9·12·16을 공존시켜 within-job `r(b)`를 얻은 뒤
    865533의 b9/b12와 대조. **이 둘 전에는 "3.06 vs 2.91"에 어떤
    판정도 내리지 않는다.**

    ⚠️★**provenance**: 캠페인 jobs **883574**(M8)·**883575**(Ha8, 각
    12 부팅) · 스모크 **883351/883545/883563** · 분석
    **result-analyst** · 적대 감사 **claims-auditor** · **메인 세션은
    운영 지표(부팅 성공·H7·실현률)만 직접 확인**, 점추정·CI 산출은
    독립 재현하지 않았다. 상세 `workspace/engine-port/results/
    s8_scaleup/C2R_RESULTS_2026-08-16.md`(+ `C2R_RESULTS_2026-08-16.json`·
    `C2R_SENSITIVITY_2026-08-16.json`·`C2R_GUARDSCAN_2026-08-16.json`·
    `C2R_POSCONTROL_2026-08-16.json`), `PREREG_C2R_RULES_REV2_
    2026-08-15.md`, `PROJECT_STATUS.md` "8B decode-SM 민감도 측정
    노트"(2026-08-16 addendum)·"다음 실험 gate" #11(갱신)·#13(신설),
    `handoff-report/session_handoff_2026-08-16.md`.

57. ★★**(2026-08-16, doc-steward 등재 — "상금 크기" 논증 감사,
    claims-auditor 기준3 CONFIRMED에서 도출) 달성된 동적의 열위(HE0)와
    달성 가능한 천장은 다른 명제다 — 정본 자신이 이 둘을 혼동한 문장을
    두 곳 이상 갖고 있었다.** HE0(§1-7 5.4σ·§1-17 ≈10σ)는 *이 저장소가
    실제로 만들어 시험한* 동적 제어(single-worker·SM-split·reactive)가
    best-static을 못 넘는다는 강한 확정이다. 그러나 §5-5·§1-17 본문이
    "천장이 static 매칭으로 확정… payoff 없음"·"static과 tie가 상한,
    이길 regime 없음"이라 적을 때 이는 **관측된 하한**을 **이론적 상한**
    처럼 서술한다 — 정본 자신이 §5-8(a)(b)에서 dual-worker(Claim D,
    서빙 0건)·non-SM-split lever(구현 전무)라는 미탐색 통로를 열어둔
    채로다. 실무 규칙: "달성된 X가 Y를 못 넘는다"와 "달성 가능한 X의
    천장이 Y 근처다"를 같은 문장에 섞지 말고, 후자를 주장하려면 그
    자체의 증거(예: §1-13 각주 E/F의 scoped 천장 진술, n=4·정본술어)를
    따로 인용하라. 두 §1-17/§5-5 행에 스코프 배너 부착(위 참조).
    상세 `PRIZE_SIZE_ARGUMENT_2026-08-16.md` §3.

58. ★★**(2026-08-16, doc-steward 등재 — "상금 크기" 논증 감사 + 코디네이터
    지적, 2회 재발) 인용금지·강등 결정은 그 수치를 만든 원 아티팩트
    문서로도 역전파하라 — 정본 본문에 다는 것만으로는 부족하다.**
    같은 실패 모드가 이 세션 안에서 이미 2회 발생했다: (i) §1-4가
    "7.24s" 인용금지를 걸었으나(2026-08-04) `CLAIM_EVIDENCE_MATRIX.md`
    Claim C·`venue_positioning.md`(§0.1·C3)로 12일간 전파되지 않았고,
    `PRIZE_SIZE_ARGUMENT_2026-08-16.md` rev1이 실제로 이 결손에 걸려
    해당 수치를 반례로 오용했다(rev2에서 자체 정정). (ii) §3 항목56(A)가
    percentile 부트스트랩 CI를 "정본 본문 인용 금지, t(5)로 대체"라고
    **당일** 결정했으나, 그 수치를 처음 산출한 원 아티팩트 문서
    (`workspace/engine-port/results/s8_scaleup/C2R_RESULTS_2026-08-16.md`
    §0 "한 줄"·§7 D2 표)는 여전히 percentile CI를 caveat 없이 헤드라인
    으로 들고 있었다(같은 날 발견·정정). 두 사례 모두 **정본에는 이미
    올바른 판정이 있었는데, 사람이 실제로 읽고 인용할 다른 문서(소비
    문서 또는 원 아티팩트)가 그 판정을 모르는 채로 남아 있었다** — 축은
    다르지만(하나는 정본→소비 문서 방향, 하나는 정본→원자료 문서
    방향) 실패 형태가 같다. 항목18("저장소가 같은 진단을 두 번 냈는데
    정본이 안 바뀌면 도구 규율 실패")과는 **축이 다르다** — 항목18은
    "진단이 반복되는데 정본이 그대로"이고, 이 항목은 "정본은 이미
    갱신됐는데 소비처·원자료가 그대로"다. 실무 규칙: 인용금지·강등을
    등재할 때는 (a) 그 수치를 인용하는 모든 소비 문서(matrix·roadmap·
    positioning 등)와 (b) 그 수치를 처음 산출한 원 아티팩트 문서 양쪽에
    체크리스트로 전파를 확인하라 — 정본 문구 자체를 고치는 것만으로는
    안 끝난다. 상세 `PRIZE_SIZE_ARGUMENT_2026-08-16.md` §7,
    `handoff-report/session_handoff_2026-08-16.md` §4-5.

59. ★★★**(2026-08-16, result-analyst R2 재분석 + claims-auditor 적대
    감사, GPU 0 — 방법론 게이트 #9 열 번째 재발 + 게이트 #18 사례)
    결정량 자체가 항등식이었고, 그 항등식은 저장소가 이미 문서로
    갖고 있었다.** R2(§1-32)의 결정량②("`SM합 = (108−D_ttft)+D_itl`이
    108을 초과하는가")는 `SM합>108 ⟺ D_itl>D_ttft`인데, `sgptv` HI의
    TTFT-argmax가 **이 격자의 최대 decode arm(d44)**이므로
    `D_itl≤44=D_ttft`가 **데이터와 무관하게 확률 1로 성립**한다 —
    부트스트랩 "10000 draw 중 0건이 108 초과"는 검정력 0의 재진술이며
    이를 증거로 제시한 것은 **게이트 #9의 열 번째 재발**이다. ★**게다가
    이 항등식은 새로 만든 것도 아니었다** — `reports/
    PRIZE_SIZE_ARGUMENT_2026-08-16.md` §2.3(2)에 `> 108 ⟺
    D_itl > D_ttft`가 **이미 문자 그대로 적혀 있었고**, 메인 세션도
    result-analyst도 그것을 결정량 설계에 적용하지 않았다 — 이는
    **게이트 #18**(저장소가 같은 진단을 두 번 냈는데 정본이 안 바뀌면
    도구 규율 실패)의 사례이기도 하다 — 단 이번엔 "정본이 안 바뀌었다"가
    아니라 **"이미 문서화된 항등식을 설계 단계에서 못 알아봤다"**는
    변종이다. 실무 규칙: 결정량을 설계하기 전에 그 결정량이 데이터
    분포와 무관하게 참/거짓이 되는 극단 사례(여기서는 "TTFT-argmax가
    격자 경계일 때")를 먼저 대수적으로 점검하고, 저장소 안에 이미 같은
    부등식이 문서화돼 있는지 grep하라. 상세
    `../workspace/engine-port/results/slo_sched/
    ORACLE_REANALYSIS_2026-08-16.md` §3-4, `PROJECT_STATUS.md`
    "방법론 게이트" #40.

60. ★★★**(2026-08-16, gate #16 사전등록 2단 감사 — 방법론 게이트 #9
    열한 번째 재발, 새 성능 판정 아님, GPU 0) 게이트 #9는 감사
    대상이 아니라 검증하는 쪽에서도 재발한다.** 두 사례가 같은
    감사 라운드에 나왔다. **(i) 메인 세션(검증자)**: `g16_arm_order.
    py --verify`의 "모든 arm 평균 위치 3.000000 PASS"를 **독립 검증
    근거로 사용자에게 보고**했으나, 역순쌍으로 만든 **어떤** 순열
    조합에서도 항상 성립하는 항등식이었다(모듈 자신의 docstring이
    자인, claims-auditor가 정정). **(ii) 감사 자신**: claims-auditor가
    1차 감사에서 `Δ_SLO` 단일 60ms 점 결정량을 대체안으로 권고했고
    메인 세션이 rev2에서 채택했으나, 후속 산출에서 감사 스스로 그
    권고를 반증했다 — `S_itl`이 58.65/58.85/60.20ms 위의 거의 평탄한
    곡선 위 **문턱 지시함수**라 "확인 카탈로그 #5(metric cliff)"가
    새 결정량에 그대로 재발했고, rev3에서 사다리 함수로 교체됐다
    (C2→C2′). 이전까지 게이트 #9의 재발은 전부 **생산자**(사전등록·
    분석 코드·산출 문서 저자) 쪽이었다(#40=열 번째가 가장 최근
    사례) — 이번 둘은 **검증자·감사자** 쪽에서 나온 첫 사례다.
    실무 규칙: "독립 검증 PASS"·"감사가 권고한 대체 결정량"도
    데이터와 무관하게 항상 참이 되는 극단 사례가 있는지 먼저
    점검하라 — 검증·감사라는 역할 자체가 게이트 #9 면역을 주지
    않는다. 상세 `workspace/engine-port/results/slo_sched/
    PREREG_G16_RULES_REV3_2026-08-16.md` addendum A-4,
    `handoff-report/session_handoff_2026-08-16.md` §15.6,
    `PROJECT_STATUS.md` "방법론 게이트" #9(열한 번째 재발).

61. ★★**(2026-08-16, gate #16 사전등록 재감사, claims-auditor 신규
    결함 N7 — 새 성능 판정 아님, GPU 0) telemetry는 공짜 관찰자가
    아니다.** `multiplexing_mixin.py:470-473`을 통과하면
    `dual_worker.py:566-602 observe_scheduler`가 **모든 sync마다**
    실행된다(쓰기만 1/32 서브샘플) — 비용은 `O(waiting_queue +
    batch)` 파이썬 작업이고, 스케줄러 스레드가 임계경로인
    `--disable-overlap-schedule`에서는 이 오버헤드가 그대로 지연에
    얹힌다. ★**2026-07-15/18 sgptv 격자(HE0/HE2 다수 헤드라인의
    근거, §1-13·§1-19·§1-20·§1-32)는 telemetry 없이 돌았다**
    (`sharegpt_vary_bench.sbatch`·`he2_bench.sbatch`에
    `PDMUX_TELEMETRY_PATH` 미설정) — telemetry가 켜진 캠페인(G16
    신규 하네스 등)과의 **절대값 직접 비교는 빌드 드리프트 + 계측
    오버헤드의 합**이라 분리 불가하다. G16 **내부** 비교(전 arm
    동일 계측)는 이 결함의 영향 밖이나, 절대값을 07-15/18과 직접
    비교하는 것은 금지된다. ⚠️**기존 결과 스코프 영향 가능(정정
    아님, doc-steward 목록만 등재) — "긴장 A(HE2 vs C2)"가 정확히 이
    패턴이다**: HE2 쪽 하네스(`he2_bench.sbatch`·
    `sharegpt_vary_bench.sbatch`, §1-13·§1-19·§1-20 근거)는 telemetry
    OFF, C2/S2/sticky/Gate 1·2-S 쪽 하네스(`s8_scaleup/*`·
    `s2_sticky/*`·`p1_gates/*`)는 telemetry ON이며, 이미 §3 항목31이
    기록한 "캠페인 계통 오프셋 −5.4%"(S2/α 분석)의 미통제 인자
    목록(노드·바이너리·날짜·워크로드)에 **telemetry 유무는 없다** —
    이 오프셋의 일부 또는 전부가 계측 오버헤드일 가능성은 아직
    검토된 적이 없다. 다음 세션 result-analyst/claims-auditor 판단
    대상이며, 이번 세션엔 정정하지 않는다(GPU 0·doc-steward 스코프
    밖). 분리하려면 telemetry-OFF 대조 부팅이 필요하다. ★★**정정
    (2026-08-21, doc-steward — S-6 검정력 계산,
    `workspace/engine-port/results/slo_sched/S6_POWER_2026-08-21.md`,
    GPU 0·새 성능 판정 0건)**: 위 "n≥2(≈0.3 GPU-hr)"는 **반증**됐다
    — 필요한 n은 δ와 사전값의 함수이며, 이 격자의 구속 셀(HI
    `M_ttft`)에서 최소 **5쌍**(δ=5.4% 참조값, 점추정)~**18쌍**(보수
    사전값 UB95)이고, 부팅 단가는 실측 **0.1272 GPU-hr/부팅**
    (457.8s — B-2가 적은 값의 2배). ★단 **S-6 설계 자체는 규칙층
    감사 `NO-GO`**(死因 4건, 전부 GPU 0의 문면 수리로 해소 가능하며
    하네스 착수 전에 잡힘)이지 **계측 축 분리가 불가능하다는
    반증이 아니다**(방법론 게이트 #21과 혼동 금지 — 이건 설계
    반려이지 측정 실패도 실험 방향 반증도 아니다). **등록된 n은
    아직 없다.** 상세 `workspace/engine-port/results/slo_sched/
    PREREG_G16_RULES_REV3_2026-08-16.md` addendum B-2(N7, 정정 표시
    병기), `handoff-report/session_handoff_2026-08-16.md` §15.5,
    `PROJECT_STATUS.md` "방법론 게이트" #41(2026-08-21 정정)·"다음
    실험 gate" #17 2026-08-21 갱신(2차).

62. ★★**(2026-08-16, 세션4, doc-steward 등재 — G16 스모크 884292
    하네스 감사에서 발견, 방법론 게이트 #21의 거울상, GPU 0·새 성능
    판정 아님) 게이트가 자기 실패를 성공으로 라벨링할 수 있다.**
    §3 항목35(교훈 항목21, "측정 실패를 게이트 실패로 라벨링 마라")는
    코드가 **성공한 측정을 실패로** 오라벨한 사례들이었다. 이번은
    그 **거울상**이다: G16 스모크 체커 `9_H2_COUNT_AND_RESIDENCY`가
    `controller_summary.json`을 못 열면 `except Exception:
    print("{}")`로 삼키고 **PASS를 찍었고**, `G16_SMOKE_OVERALL`
    논리곱이 **item 9를 아예 참조하지 않았다** ⇒ 884292가
    `residency_fraction={}`(측정이 사실상 전무한 상태)를 출력하면서
    `do submit`을 말할 수 있었다. 이 경로는 addendum A-2가 요구하는
    "arm 라벨 = 동거 구간의 명목 split" 조건부 라벨의 **크기 기준선을
    캠페인 내내 비워 놓을 수 있는** 종류의 결함이었다.
    반영(커밋 `37cf6b8`): `9a`(스냅샷 카운트)/`9b`(residency)로
    분리해 파일 부재·파싱실패·키 부재·빈 dict·퇴화 전부 FAIL로 판정,
    `OVERALL` 논리곱에 편입. 검증: 신규 `g16_smoke9_check.py`(8케이스)의
    case H가 **실제 884292 아티팩트에 새 게이트를 걸어 PASS→FAIL로
    뒤집는 것**을 양성대조로 사용. 실무 규칙: 게이트/스모크 체커
    작성 시 (i) 예외를 삼키는 모든 `except` 블록이 FAIL로 귀결하는지,
    (ii) 전체 판정의 논리곱/논리합이 **정의된 모든 item을 실제로
    참조하는지**를 별도로 assert하라. 상세 `PROJECT_STATUS.md`
    "방법론 게이트" #42, `handoff-report/session_handoff_2026-08-16.md`
    §4-2.

63. ★★**(2026-08-16, 세션4, doc-steward 등재 — G16 사전등록 분석기
    `g16_analyze.py` 감사, claims-auditor F1, GPU 0·새 성능 판정
    아님) 분석기가 사전등록 기호를 조용히 재정의하면 자기 대조를
    깨고 중심 산출물을 침묵시킨다.** `g16_analyze.py` 최초 구현이
    §4가 bare argmin으로 **정의**한 `D_itl`을 §6의 K1 식별 게이트를
    통과할 때만 값을 돌려주는 것으로 **조용히 재정의**했다 — §6은
    식별을 판정의 *조건*으로만 얹을 뿐 §4 기호를 재정의하지 않았는데도.
    부작용: (a) 자기 양성대조 **PC-C가 깨짐**(§7 원문 "1차 추정량으로도
    `D_ttft=d44·D_itl=d44`"가 추정량에 스코프를 명시했는데, 재정의판
    분석기는 미식별 시 `Δ=None`을 돌려줘 이 표적을 통과 못함).
    (b) 미식별 시 `forced_cell='undetermined'`가 돼 **§3 강제표(설계의
    중심 산출물)가 가장 개연적인 결과에서 자동 침묵**했다. 반영(F1,
    커밋 `8311d6c`) = bare argmin 복원 + K1을 `delta_citable`/verdict의
    **인용 게이트**로 분리 → PC-C 5/5 통과·§3 강제표 복원.
    ★**메인 세션 오진 1건 기록**(재발 방지용): 이 결함을 처음
    발견했을 때 *"PC-C 표적이 §6 게이트를 통과 못 하므로 **사전등록
    텍스트 결함**"*이라 보고했는데 **절반 틀렸다** — 사전등록 §4·§7은
    문제 없었고, 결함은 **분석기가 §4 기호를 게이트-조건부로 재정의한
    것**이었다. 근거로 든 문장도 estimand를 혼동했다("ORACLE §3-4
    재현"이 실은 문턱술어 도너(0.595/0.405)와 위치통계량 도너
    (0.638/0.362)의 우연한 근접값 비교였다 — 게이트 #9/§3 항목39가
    지목한 바로 그 추론 패턴, "숫자가 맞으니 같은 경로"). 일치하는
    것은 **결론**(d44/d34 비식별)이지 추정량이 아니다. 실무 규칙:
    분석기 감사에서 "결과가 사전등록 문구와 다르다"를 발견하면
    **먼저 분석기가 그 문구의 기호를 실제로 그대로 구현했는지 확인**
    하고 나서 사전등록 결함으로 보고하라. 상세 `PROJECT_STATUS.md`
    "방법론 게이트" #43, `handoff-report/session_handoff_2026-08-16.md`
    §4-3.

64. ★**(2026-08-16, 세션4, doc-steward 등재 — G16 사전등록 분석기 PC-B
    감사, 방법론 게이트 #9 계열, GPU 0·새 성능 판정 아님) 양성대조의
    빈 서명 구멍.** `g16_analyze.py`의 최초 PC-B(양성대조) 구현이
    실제로 **빈 서명 집합으로도 통과**할 수 있었다 — 대조가
    estimand를 아예 실행하지 않아도 "통과"를 찍을 수 있는 구조였다
    (방법론 게이트 #9 아홉 번째 재발·§3 항목56(C)의 C2-R 양성대조가
    다른 코드 경로를 타면서도 표적값을 재현했던 사례와 같은 계열의
    취약점). 반영: (a) 서명을 **런타임에 실제 인자·분기로부터 기록**
    하도록 바꾸고, (b) **"빈 서명 집합은 통과 불가"** 규칙을 추가하고,
    (c) **`PC-B-neg`**(kwargs 하나만 바꿔 비교가 실제로 차이를
    잡아내는지 확인하는 음성 방향 대조)를 신설했다. 실무 규칙:
    양성대조 설계 시 "표적값 일치"만이 아니라 (i) 서명이 비어 있지
    않은지, (ii) 인자를 바꾸면 대조가 실제로 반응하는지(음성 대조)를
    함께 assert하라 — 이 둘이 없으면 양성대조가 항상 통과하는 항등식
    으로 퇴화할 수 있다. 상세 `PROJECT_STATUS.md` "방법론 게이트" #44,
    `handoff-report/session_handoff_2026-08-16.md` §4-3.

65. ★★★**(2026-08-17, G16 결과 문서 §2.4 감사, 방법론 게이트 #40/#9
    열두 번째 재발, GPU 0·새 성능 판정 아님) 감사 목적으로 쓴 검사
    자신이 항등식일 수 있다 — 비식별성의 "증거"가 비식별성의
    "정의"였다.** `gap_upper`(HI/LO 도너 미식별의 근거 통계량)가
    순수 노출차만으로 설명되는지 확인하려던 검사가, 상태-혼합 모형
    `M_itl(a) = w(a)·S(a) + (1−w(a))·U(a)`에서 **arm당 관측치 1개**
    (주변 평균)·**미지수 2개**(`S(a)`·`U(a)`)를 가정 2개(`U(d64)=U(d74)`,
    `S−U=Δ`)로 강제한 뒤 `Δ = gap_upper / (w(d74)−w(d64))`를 푸는
    것이었다 — **식 1개·자유모수 1개**라 `Δw ≠ 0`이기만 하면
    **관측값과 무관하게 항상 유일해를 갖는다**(반증 가능성 0). 초판
    결과 문서가 이것을 *"노출차만으로 gap이 정확히 재현된다"*는
    **발견**처럼 적었으나, 감사가 "이 검사는 비식별성의 *정의*를
    비식별성의 *증거*로 제시한 것"이라 정정했다(G16_RESULTS §2.4).
    이전 재발(§3 항목18·39·56·59·60 등, 게이트 #9 열한 차례)은 전부
    **생산자 또는 검증자**가 만든 항등식이었으나, 이번은 **감사·진단
    목적으로 설계된 검사 자체**가 항등식이었던 첫 사례다. 실무 규칙:
    새 진단·감사 검사를 설계할 때도 "이 검사가 데이터와 무관하게
    항상 같은 결론을 내는 자유도가 있는가"를 먼저 대수적으로
    점검하라 — 감사자의 역할이 항등식 면역을 주지 않는다(§3 항목60의
    "검증자 쪽 재발"과 같은 계열, 이번엔 대상이 진단 도구 자체).
    상세 `workspace/engine-port/results/slo_sched/
    G16_RESULTS_2026-08-17.md` §2.4, `PROJECT_STATUS.md` "방법론
    게이트" #45.

66. ★★**(2026-08-17, G16 결과 문서 §1.3 감사, GPU 0·새 성능 판정
    아님) 사다리 해상도가 사전등록된 payoff 구간을 은폐할 수 있다.**
    G16 사전등록(K8)은 `Δ_SLO` 사다리를 45–80ms, **1ms** 스텝으로
    등록했다. 초판 결과 문서는 이 사다리(45–58ms `UNREACHED` / 59–60ms
    미세 변화 / 60ms+ `NEGATIVE`)만으로 *"60ms에서 tax 없음"*을
    헤드라인으로 냈으나, 분석기 자신이 함께 산출한
    `s_itl_exact_bands`는 **폭 0.092ms짜리 밴드**
    (`[58.533, 58.625)`)에서 `Δ_SLO`의 부호가 **+20(`TAX_POSITIVE`)**
    로 뒤집힘을 보였다 — 1ms 격자는 이 밴드를 **볼 수 없다**(분석기가
    *"cite the exact bands"* 경고를 찍었는데도 초판이 사다리만
    인용했다). 하필 이 밴드가 사전등록(rev3 §4-1)이 *"이 캠페인의
    SLO-관련 payoff는 전적으로 이 구간(≲58.6ms)에 있다"*고 예언한
    바로 그 구간이었다 — **가장 중요한 구간에서 가장 거친 격자를
    썼다.** 실무 규칙: SLO/문턱 스윕을 사다리(고정 스텝)로 등록하더라도,
    결정량이 스텝보다 좁은 폭에서 부호를 바꿀 수 있는 계단함수
    (예: `S_itl`처럼 min-연산 기반 정수값)라면 **exact breakpoint를
    별도 산출·병기**하도록 사전등록 자체에 강제하라 — 사다리는 사람이
    읽기 위한 요약일 뿐 판정의 정본이 아니다. 상세
    `workspace/engine-port/results/slo_sched/G16_RESULTS_2026-08-17.md`
    §1.3, `PROJECT_STATUS.md` "방법론 게이트" #46.

67. ★★★**(2026-08-18, 메인 세션 자기정정, GPU 0·새 성능 판정 아님,
    항목41의 열두 번째 재발) 인용금지는 정본 등재만으로 전파되지
    않는다.** 위 §1 항목33이 "split 상태 decode batch 절대값
    33.04/20.39/19.49(재현 경로 없음)"의 인용을 승격금지(vii)로 못박은 [CS-OK]
    **바로 다음 날**, 메인 세션이 새 분석 문서에서 이 값을 다시
    인용했다 — 사람이 매번 정본 전문을 기억해 새 글과 대조하는 방식이
    실패한다는 것이 실증됐다. 대응: 사람의 기억에 의존하지 않는
    **기계적 차단 도구**
    `workspace/engine-port/scripts/discipline/check_citation_stops.py`
    + 레지스트리 `citation_stops.tsv` 신설. 1차 설계(저장소 전체
    스캔)는 40파일 30건 오탐(사전등록·승격금지 목록 자체가 금지
    문구를 정당하게 열거해야 함)으로 **폐기**됐고, 실패 모드에 맞춰
    **커밋에 새로 추가되는 줄(diff `+`)만** 검사하는 형태로 재설계했다
    — `[CS-OK]` 마커로 의식적 예외 인정, `audit_*/`·`scripts/
    discipline/` 경로 면제, 레지스트리가 비면 통과가 아니라 **에러로
    죽는다**(빈 서명 구멍 방지, 항목47과 같은 계열). 양방향 검증:
    실제 위반(33.04) 재현 exit 1 · 이 감사 산출물 자체(5,061줄) 통과 · [CS-OK]
    숫자 우연 충돌(오탐) 1건을 레지스트리 주석으로 등재. ★**한계
    병기**: 추가 줄만 보므로 **트리에 이미 있는 위반은 못 잡는다** —
    재발 방지이지 소급 감사가 아니다. 실무 규칙: 인용금지 등재는
    "정본에 적었다"이지 "전파됐다"가 아니다 — 등재 당일 이후의 **모든
    신규 문서 작성 행위**가 각자 독립된 실패 지점이다. 상세
    `workspace/engine-port/scripts/discipline/citation_stops.tsv`,
    `PROJECT_STATUS.md` "방법론 게이트" #47.
    ★추가 사례(2026-08-21, doc-steward — **위반 아님, 선제 정정이라
    재발로 신규 등재하지 않음**): `S6_POWER_2026-08-21.md`가
    반증한 B-2의 "n≥2(≈0.3 GPU-hr)"가 세 곳(`CONSENSUS.md` §3
    항목61·`PROJECT_STATUS.md` 방법론 게이트 #41·원 사전등록
    `PREREG_G16_RULES_REV3_2026-08-16.md:587`)에 2026-08-16부터
    그대로 남아 있었다는 사실 자체가 이 항목이 경고한 패턴(정본
    등재는 전파를 보장하지 않는다)의 거울상이다 — 다만 이번엔 그
    stale 값이 **새 문서에 실제로 잘못 인용되기 전에** 잡혀 세 곳을
    동시에 정정했다.

68. ★★**(2026-08-18, 메인 세션 자기정정, GPU 0·새 성능 판정 아님)
    결과 디렉터리 이름을 캠페인 이름으로 오인하지 마라.**
    `results/slo_sched/`는 "G16 캠페인 디렉터리"가 아니라 **12개
    캠페인의 447개 bench 파일이 섞인 공유 디렉터리**다(prefix 분해:
    `iact` 77 · `g16` **68** · `lffShort`/`lffLong` 46 · `he2A`/`he2B`
    42 · `sgptvLo`/`sgptvHi` 33 · `sgpt`/`jl` 22 · `mixA`/`mixB` 6).
    이 오인이 2026-08-18에 **"G16만 밴드질량 이상치"라는 사실 오류**를
    만들었다 — `g16_` prefix로만 필터해 전수 집계하면 G16 HI
    밴드질량은 **67.47**이고, θ=60 부근 밴드질량을 계산 가능한 형제
    서브캠페인 3개가 **같거나 위**였다. 실무 규칙: 캠페인 단위 통계를
    집계하기 전에 디렉터리가 실제로 그 캠페인 전용인지 확인하고,
    공유 디렉터리라면 파일명 prefix(또는 매니페스트)로 서브캠페인을
    분해한 뒤 집계하라 — "디렉터리 = 캠페인"은 검증 없이 가정하면 안
    되는 명제다. 상세 `handoff-report/session_handoff_2026-08-18.md`,
    `PROJECT_STATUS.md` "방법론 게이트" #48.

69. ★★★**(2026-08-18, 메인 세션 자기정정, GPU 0·새 성능 판정 아님,
    방법론 게이트 #18 최강 사례) 정본이 이미 `ILL-POSED`로 판정한
    측정점을 재분석에 쓰지 마라.** 2026-08-18에 메인 세션은
    `g2_0_full` phase A(rA5 운영점)의 평균에서 "d34 골짜기"(`d44-d34`
    +33.59pp)를 발견처럼 보고했으나, **같은 계열의 `g2_0_hard/
    hardened_disjoint_verdict_2026-07-25.md`가 바로 그 rA5 운영점을
    이미 `ILL-POSED`로 판정**해 두었고(TTFT 3s-cliff bimodality,
    d44/d54 견고성 순위가 sweep 간 완전 반전), `g2_0_full/
    disjoint_verdict_2026-07-24.md` 자신도 같은 d34 rep1/rep4의 t50
    3063/3385ms를 "cliff" 사례로 **이름까지 붙여** 이미 기록해 두었다
    — "골짜기"로 재해석된 것이 그 문서가 이미 "절벽"이라 이름 붙인
    바로 그 관측치였다. 재현 캠페인 `g2_0_hard`(독립 재현)에서는 이
    차이의 **부호가 반전**되고(원 문서 페어드 CI [−22.6,+89.8], 0
    포함), 애초에 통계적으로 유의하지 않았다. 이전 게이트 #9 재발
    11회는 대체로 "결정량 자체가 항등식"이었던 데 비해, 이번은
    **"판정할 자유가 애초에 없는 측정점(정본이 이미 ill-posed로 닫은
    지점)을 판정 대상으로 다시 연 것"**이라 게이트 #18의 최강 사례로
    등재한다 — 이번엔 정본이 "안 바뀐" 게 아니라 **판정서를 읽지
    않고 지나쳤다.** 실무 규칙: 원자료 디렉터리를 재분석 대상으로
    열기 전, 그 디렉터리(또는 그 캠페인을 다루는 CONSENSUS/
    PROJECT_STATUS 절)에 이미 `*_verdict_*.md`류 판정서가 있는지
    먼저 찾아 읽어라 — 원자료가 남아 있다는 사실이 "판정 미정"을
    뜻하지 않는다. 상세 `workspace/engine-port/results/g2_0_hard/
    hardened_disjoint_verdict_2026-07-25.md`,
    `workspace/engine-port/results/g2_0_full/
    disjoint_verdict_2026-07-24.md`, `PROJECT_STATUS.md` "방법론
    게이트" #49.

70. ★★★**(2026-08-19, gate #13 rev3 재감사, claims-auditor, GPU 0·새
    성능 판정 아님, 게이트 #9의 열세 번째 재발) 자기 수리를 검증하는
    검사 자신이 반증 불가능한 항등식일 수 있다.**
    `DESIGN_G13_JOB_BATCH_REV3_2026-08-19.md` 초판은 감사가 지목한
    死因 3건(F2 가드 문턱 오류·Ha8 σ_boot의 부팅 1개 의존·`power2()`의
    exact-F 하드와이어)을 규칙층에서 수리한 뒤, 그 수리를 "검증"한다는
    명목으로 검사 3건(구 PC8·구 S3·구 S4)을 대조/자기검사 집합에 넣고
    `all_pass=true`로 계수했다. 재감사가 확인한 바, 셋 다 **데이터와
    무관하게 항상 참인 항등식**이었다 — PC8(`F₀.₀₅<1`)은
    `F₀.₀₅(d1,d2)=1/F₀.₉₅(d2,d1)`이고 `F₀.₉₅>1`이 항상 성립하므로
    어떤 격자에서도 위반 불가, 구 S3는 `_ub_exact_f`의 대수 구조상
    `UB≤0 ⟺ msb≤fq_lo·msw`가 구성상 참(게다가 `_oc`를 호출하지 않고
    가드 식을 복사 재구현해 실제 가드 줄의 오타를 못 잡는 설계였다),
    구 S4는 `msb>msw`면 `v>0`이라 검사 대상 사건이 애초에 발생
    불가능했다. rev2가 이미 겪은 실수(F-역함수 항등식을 `identity_
    only`로 분리해 회피)와 **같은 실수의 재발**이며, 이번엔 "수리를
    검증하는 검사 자신"이 항등식이었다는 점에서 게이트 #9의 새로운
    변종이다. **해소**: S3·S4를 `_oc`를 **실제로 호출**하고 `_oc`가
    쓰지 않는 독립 경로(해석적 F 분포)와 대조하도록 재작성 —
    σ_job=0에서 exact-F 가드는 정의상 0.05 확률로, Satterthwaite
    가드는 `P(F≤1)`로 발동해야 하고, 관측치가 그 이론값과 일치하는지
    검사한다. **변이 테스트로 반증 가능성을 증명**했다: 가드를 rev2의
    `msb≤msw`로 되돌린 사본에서 재작성된 S3가 **0.53 vs 기대 0.05로
    FAIL**한다(S2도 FAIL) — 구 항등식판 S3는 이 변이본에서도 통과했을
    것이므로 애초에 아무것도 검증하고 있지 않았다. ⇒ **신규 게이트
    후보**: **"자기 수리를 검증하는 검사는 그 수리를 되돌린 변이본에서
    반드시 실패해야 한다"** — 항등식 혼입을 사람이 매번 대수로
    확인하지 않고 기계적으로 걸러내는 유일한 방법이며, 이번에 실제로
    작동해 반증 불가능한 검사 3건을 `all_pass`에서 분리시켰다(재구성
    후 실제 반증 가능한 검사는 대조 19건 + 자기검사 7건 — 초판이 쓴
    "20건/4건"은 항등식을 포함한 계수였다).
    ★**부수 등재(B7) — 재현성 지뢰**: rev2 문서 상단이 지시한 재현
    명령(`design_g13_stats.py rev2 --out DESIGN_G13_STATS_REV2_
    2026-08-17.json`)은 rev3가 같은 함수(`_oc`/`power2`/`seq2`)를
    **공유**하는 코드 경로를 수리했기 때문에, 그 명령을 그대로
    실행하면 **정본 rev2 JSON을 수리된 숫자로 덮어써** rev2가 어떤
    수치를 보고했는지의 기록 자체를 지울 수 있었다. `design_g13_
    stats.py`가 기존 파일 덮어쓰기를 **거부**하도록 수정(`--force`로만
    우회)하고, rev2 문서 상단에 그 사실을 경고 배너로 남겼다 — 사전등록
    문서가 지시하는 재현 명령 자체가 그 문서를 무효화할 수 있다는
    일반적 위험의 구체 사례. 실무 규칙: (i) 규칙 수리를 검증하는
    자기검사를 설계할 때, "이 검사가 통과하는 데 필요한 최소 조건이
    무엇인가"를 먼저 묻고 그 조건이 데이터에 의존하는지 확인하라(의존
    안 하면 항등식이다). (ii) 가능하면 검사가 수리 전 코드에서 반드시
    실패함을 실제로 실행해 확인하라(변이 테스트) — "논리적으로
    실패해야 한다"는 주장만으로는 부족하다(구 S3가 그 실패 사례).
    (iii) 개정판이 이전 판의 산출 파일을 재생성 명령으로 덮어쓸 수
    있는 공유 코드 경로가 있는지 확인하고, 산출 파일 쓰기 경로에
    덮어쓰기 방지 가드를 기본값으로 둬라. 상세 `workspace/engine-port/
    results/s8_scaleup/DESIGN_G13_JOB_BATCH_REV3_2026-08-19.md`
    §4.1·§6, `DESIGN_G13_JOB_BATCH_REV2_2026-08-17.md`(재현성 지뢰
    경고 배너), `PROJECT_STATUS.md` "방법론 게이트" #50.

71. ★★**(2026-08-21, gate #13 캠페인 해제 절차, 메인 세션, GPU 0·새
    성능 판정 아님) 블라인딩 감시 문자열을 설명하는 문서가 그 문자열
    자체를 인용하면 자기 자신을 건다.** `g13_analyze.py --unblind`는
    `PREREG_G13_2026-08-20.md`에 `AUDIT_STATUS: PENDING` 감시 문자열이
    있으면 거부하도록 짜여 있었는데, 첫 해제 시도에서 그 문서 §2가
    블라인딩 기전을 **설명하며 감시 문자열을 그대로 인용**해 자기
    자신을 걸었다. **fail-closed 방향**이라 안전했다 — 거짓 unblind를
    허용하는 실패가 아니라 정당한 unblind를 거부하는 실패였다. 해소:
    분석기 SHA(`01448211…`)를 지키기 위해 문서 쪽 인용을 우회
    서술로 재작성. 증거 `G13_ANALYSIS_2026-08-21_REFUSED.json` 보존.
    실무 규칙: 자동화된 감시 문자열을 설명 목적으로 문서에 적을
    때는 감시기가 검색하는 정확한 형태로 인용하지 말 것(paraphrase).
    상세 `workspace/engine-port/results/s8_scaleup/G13_RESULTS_
    2026-08-21.md` §5, `PROJECT_STATUS.md` "방법론 게이트" #51.

72. ★**(2026-08-21, seq3 8티어 확장, 메인 세션) 격자를 한 점만 넓히고
    멈추면 그 자체가 격자 산물이다.** seq3 결론("sequential은 예약액을
    못 낮춘다")은 감사가 8티어×2arm×2m으로 깨뜨리려다 실패해 생존
    했으나, 메인 세션이 격자를 **k=14 한 점만** 추가하고 멈춘 탓에
    최저가(**k=13/3.12**, 총액 **10.80** — 인용값 11.04는 stale)와
    격자 산물 크기(**0.72** — 인용값 0.48은 stale, m축까지 넓히면
    1.76)가 부정확하게 인용됐다. 방법론 게이트 #16(스코프 확장은
    원 격자를 전부 재현하라, 2026-08-07)의 재발이지만 이번엔 "확장을
    아예 안 함"이 아니라 **"확장을 부분적으로만 하고 멈춤"** 이라는
    새 형태다. 실무 규칙: 격자를 넓히는 실험은 넓힌 축의 전 범위를
    스윕하거나, 못 하면 그 사실 자체를 결과에 명시하라(부분 확장에서
    나온 최저가·산포 인용 금지). 상세 `handoff-report/session_
    handoff_2026-08-21.md` §2.3, `PROJECT_STATUS.md` "방법론 게이트" #52.

73. ★**(2026-08-21, S0(a) 감사, claims-auditor) 비용 정합 없이 두
    대조를 견주면 분모가 틀린다.** S0(a) 초판이 *"C-g의 전제가
    성립하지 않는다"*고 결론 낸 판정선은 **1 부팅쌍짜리** half-split
    대조를 **2 부팅쌍짜리**가 필요한 부팅-간 대조의 SD와 직접
    비교했다 — 같은 예산(부팅쌍 N개)에서 두 설계의 SE가 다른
    상수(`s_alt/√N` 대 `2σ_boot/√N`)로 스케일한다는 사실을 놓쳤다.
    비용 정합 판정선(`s_alt` 대 `2σ_boot`)으로 재판정하면 답이
    뒤집힌다(부팅당 8구간 교대는 이기고 4·2구간은 진다). 실무 규칙:
    서로 다른 측정 설계를 비교하는 판정선을 세울 때는 두 설계가
    "같은 예산에서 무엇을 추정하는가"를 먼저 대수로 적고, 그 식에서
    분모(표준오차 스케일)가 실제로 같은지 확인하라(다르면 비용
    정합 변환 후 비교). 상세 `workspace/engine-port/results/
    s8_scaleup/S0A_VERDICT_2026-08-20.md` §0, `PROJECT_STATUS.md`
    "방법론 게이트" #53.

74. ★★**(2026-08-21, G18 rate-축 정본 오염 발견, 메인 세션) 태그는
    오염을 막지 못한다 — 분석기의 glob/regex를 직접 확인해야 한다.**
    `g18_probe_feeder.sh`가 프로브 run id에 `g16_` 접두사를 재사용
    했는데, `g16_analyze.py:953`의 글롭+regex가 그 id를 정상 매치하고
    mode 필터는 `smoke`만 걸러 **완주한 프로브가 G16 정본 블록으로
    조용히 편입될 뻔했다** — "태그가 막아준다"는 메인 세션 자신의
    주석은 코드를 확인하지 않고 쓰인 거짓이었다. 실제 오염은
    **0건**(발견 즉시 접두사를 `g18probe_`로 바꿔 구조적으로 차단,
    검증 완료). 실무 규칙: 이름 규칙(prefix/tag)으로 캠페인을
    격리한다고 주장하기 전에 소비하는 분석기의 실제 매칭 코드
    (glob 패턴·정규식)를 읽고 그 이름이 실제로 안 걸리는지 확인
    하라 — 이름 규칙 자체는 검증되지 않은 약속일 뿐이다. 방법론
    게이트 #48(결과 디렉터리 이름 오인)의 자매 사례. 상세
    `workspace/engine-port/results/slo_sched/G18_RATE_VALUE_
    2026-08-20.md` §6, `PROJECT_STATUS.md` "방법론 게이트" #54.

---

## 4. 살아있는 문서 (이것만 참조)

| 문서 | 역할 |
|---|---|
| **CONSENSUS.md** (이 문서) | **정본** — 확정/철회 상태 |
| **`per_layer_type_postmortem.md`** | ★**per-layer-type 구제 시도→기각 kill-chain**(시도 A–H, 두 조건 C1 lever·C2 착취 관점) |
| **`research_arc.md`** | ★**연구 아크 전사** — 출발점→현재의 단계별 시작논의/촉발지표/해소지표/부정사유(수치 포함). CONSENSUS의 서사 짝. **§S-M = 서빙 수준 반증 지점(등급별) · 측정 환경(E1–E4) · 결론의 유효 경계** |
| **`longcontext_trace_plan.md`** | ★**계획 문서**(L−1 이상은 측정 전) — 실 trace를 long-context로 전환하는 문제. 동기(Diff A는 L≥3k서 열림) vs 정직한 반론(Diff B는 long-L서 닫힘) · 하드 블로커(Zamba2 ctx 4096 / goodput SLO 붕괴) · 단계 L0–L3. ★★★**L−2(Stage 0) 게이트는 2026-07-26 non-binding으로 "실행·확정"됐다고 기록했으나 2026-07-28 claims-auditor 감사(C1 CONFIRMED)로 무효 — 게이트는 사실상 미실행이었다**(아래 `stage0_verdict_2026-07-26.md` 참조), L−1 이상은 "게이트 실패로 보류"가 아니라 "게이트 미실행" |
| **`stage0_verdict_2026-07-26.md`** | ★★★**Stage 0(long-ctx L−2) 원 판정 — 2026-07-28 판정2/판정3 철회(C1 CONFIRMED, §1-21)**. coupled 스윕 confound 진단(판정1)만 생존, "de-confounded D16≡D108 null" 결과는 D108 앵커가 실은 decode 16 SM이었음이 확인돼 무효. 이력 보존용, 새 분석 근거로 재인용 금지(단독으로는) — 재인용 시 §1-21 전문과 병기 |
| `results/slo_sched/lengthnorm_slo_reanalysis.md` | ★**길이-정규화/tight SLO 재계측**(§1-16) — HE0가 SLO 엄격도 의존임을 기존 벤치 재분석으로 확정. 스크립트 `reanalyze_lengthnorm_slo.py` |
| **`serving_slo_survey.md`** | ★**실 서빙 SLO 관행 조사**(§1-16 후속) — 프로덕션 인터랙티브 TTFT(chat 300/voice 150/code 100/RAG 400ms)가 전부 tight regime; 우리 3s=batch async. DistServe SLO-scale sweep=표준. goodput 메트릭 비판 |
| **`interactive_slo_retune_plan.md`** | ★**tight-SLO 컨트롤러 재튜닝 + P90-attainment 직접 측정**(§1-17) — §9에 최종 결과(HT0 확정, d44≫동적). 하네스 `results/slo_sched/interactive_bench.sbatch` |
| `../../workspace/engine-port/results/s8_frontier/DESIGN.md` | ★★**E1 프론티어 하네스 설계·전사**(§4.3.1–4.3.16) — 사전등록·버그 수정·게이트 이력의 정본. §4.3.9=`g` 격자 한정 은퇴, §4.3.10=`A_free` 대체 조건부 추정량[AUDITED, blocking 스윕만 UNAUDITED], §4.3.11=`PDMUX_STICKY_PARTITION` 구현 사실, §4.3.12=sticky 런 사전등록(`G_LEVER`/`G_FLAT` 미결정), §4.3.13=C2→`G_LEVER` 앵커 경로 폐기[AUDITED, §0 신규 최상위 열린 항목], §4.3.14=D=54 측정 취소+keepalive 재현성 결함[일부 미감사]+C2 residency 워크로드 장치 산물[AUDITED], §4.3.15=§0 이분법 유지 불가·세 번째 후보 실측 문서화[감사자 재프레이밍 UNAUDITED, result-analyst 독립 재현 PARTIAL INDEPENDENCE, 성능 판정 0건, S2(GPU) 대기], **§4.3.16=S2(job 873015) 독립 재현 CONFIRMED(scoped) — §0 최상위 열린 항목 behavioural 종결, E1은 4가지 독립 사유로 미개방, 성능 판정 0건[claims-auditor CONFIRMED, 2026-08-05]** |
| `../../workspace/engine-port/results/s2_sticky/S2_REPLICATION_2026-08-05.md` | ★★★**S2(job 873015) 독립 재현 전문 — claims-auditor CONFIRMED(scoped), 2026-08-05.** §0 3지선다의 behavioural 종결(pooled ITL p50 d16 28.92ms/d54 12.04ms, 사전등록 구간 적중), 기전(스냅샷 샘플링 케이던스) 독립 도출, d54 companion 미스·§4.3.12(f) 판별예측 설계상 미판정·결과 게이트 0개(방법론 게이트 #9 네 번째 재발)·`S2_ANALYSIS_2026-08-04.md` 계측 오류 정정 기록, "등재 금지" 표 포함. **재현 스크립트는 아직 리포지토리 밖(에이전트 scratchpad)** — 이관 필요, 별도 작업 |
| `../../workspace/engine-port/results/s2_sticky/S2_ANALYSIS_2026-08-04.md` | result-analyst의 S2 1차 분석(정본 아님, claims-auditor 재현으로 대부분 확인됨) — **61–64행 "Instrument check" 문단은 2026-08-05 정정 표시됨(원문 보존)**: 케이던스 ~2.0ms는 decode-idle 값, decode-busy 조건부는 0.177–0.465s. 정정 전 그 문단 인용 금지, 나머지 수치는 유효 |
| `../../workspace/engine-port/results/s8_scaleup/NOTES_D54_ANCHOR_2026-08-03.md` | ★**D=54 앵커 취소 기록**(미추적) — keepalive 토큰 초과로 인한 s8_scaleup 재현성 결함[미감사, 코드/로그 직접 검증] + C2 high-residency=워크로드 장치 산물 독립 수렴 3경로[AUDITED]. 상세는 `DESIGN.md` §4.3.14·CONSENSUS §1-29 |
| `bench_noise_root_cause.md` | ★벤치 노이즈 근본원인(메트릭 절벽)·3× 하네스 버그·HE0 구조적 이유 |
| `../../workspace/engine-port/results/slo_sched/CEILING_CENSORING_DIAG_2026-08-05.md` | ★★**§1-13 각주 직접 검정 — claims-auditor AUDITED(2026-08-05), 정본 반영 완료.** 헤드라인 논증(§3.4 임계 사다리·§5 근거5·§3.5 "지배" 서술·§4 C2 정성 대조·§3.6 "gpu39 3중 사다리")은 REFUTED(정정 표시, 원문 보존); 각주 자신의 결론(LO 이중 절단·레버 실재)은 다른 증거로 CONFIRMED. 상세 §1-13·§3 항목24·25 |
| `realtrace_findings_and_open_branches.md` | 실 trace 검증 + 얽힘 기전 + 남은 갈래(트리거/행동모델) 상세 |
| `slo_aware_scheduling_design.md` | SLO-aware track 설계·실측 전사(§C–§HE2-3, Step D/E/F/G) |
| `policy_comparison.md` | 정책 taxonomy·메커니즘 (⚠️ §goodput 수치는 no-cudagraph·구벤치 — §1·§2 우선) |
| `system_vs_engine_vs_sim.md` | fidelity ladder (sim/engine/full-system 편향 분리) |
| `prefill_vs_decode_execution.md` | prefill/decode 실행 특성·knee 기초 |
| `sm_policy_report.html` | layer-aware 원자료 기록(死 트랙, 이력용. §coordinated 개정 미완 = stale) |
| `paper/venue_positioning.md` | ★**전략 문서(2026-07-24, 2026-07-25 §0.1 갱신)** — 투고 positioning. claim/evidence 아님, CONSENSUS/matrix 판정을 인용·요약만 함. 새 증거 근거로 쓰지 말 것. §0.1 = 신규성 축(disaggregation→multiplexing) 정정 + negative (A)/(B) 분해 + green-context vendor-primitive 방어 + 벡터2 게이트(cross-substrate 이식 철회 → Transformer-control on green-context) |
| `spatial_decoupling_design_review_2026-07-25.md` | §1-20 disaggregation 기판 실현가능성 설계 검토. **§1(신규성)은 위 `venue_positioning.md` §0.1로 부분 supersede**(문서 상단 HISTORICAL 노트), §2–§5(SGLang v0.5.10 기판 file:line, long-ctx 게이트 연결점)는 유효 |
| `../../workspace/engine-port/results/p1_opint/P1_OPINT_RESULT_2026-08-05.md` | P1 운영점(cudagraph-ON) 대조 판정서(jobs 873944/873945). **본문의 percentile-bootstrap CI·"임계 사다리 전 구간 부호 불변"·"rep 부호 5/5" 서술은 2026-08-06 통계 방법 층 정정(§1-1 rev13·§3 항목27)으로 갱신됨 — 재인용 시 CONSENSUS §1-1(rev13) 병기 필수**, 원 문서는 수정하지 않음(이력 보존) |
| `../../workspace/engine-port/results/p1_gates/verify/` | ★★★★**2026-08-06 통계 방법 층 정정의 1차 산출물**(claims-auditor Gate 2 감사 2회 + result-analyst 독립 재현) — `verify_c1_coverage.py`(bootstrap coverage MC)·`verify_c2_c3.py`/`verify_c2_pvalues.py`(t-CI 재채점)·`verify_c3_boundary.py`(임계 사다리·vacuity 경계)·`verify_c1_unpaired.py`/`verify_c1_othern.py`(unpaired·타 n coverage) + `*.json`/`*.log`. **Gate 1/Gate 2가 진행 중인 디렉터리이므로 편집 금지, 인용만** |
| `../../workspace/engine-port/results/p1_gates/gate1/` | ★★★★★**Gate 1(job 874478, 2026-08-06) 원자료 — claims-auditor 감사 완료, 조건부 채택, 새 성능 판정 0건.** `gate1_result_874478.txt`(판정 출력)·`PREREG_GATE1_2026-08-06.md`(사전등록+부기1·2)·`gate1_analyze.py`(coverage guard 포함 분석기)·`gate1_telemetry_874478.jsonl`·`gate1_srv_874478.log`(근거 인용원, `:32`가 분할표 근거). 874465(1차 시도, 절단된 창)도 같은 디렉터리에 보존(비교용). **전문은 CONSENSUS §1-1 Gate 1 블록·§3 항목28·29 — 수정 금지, 인용만.** ★★★★★**같은 디렉터리에 G1-b(job 875293, 2026-08-07) 후속 — 사전등록 철회 규칙 발화, 새 성능 판정 아님.** `PREREG_G1B_2026-08-07.md`(사전등록)·`gate1b_result_875293.txt`(판정 출력, rate{2,3,4,6}). rate6에서 `(54,54)` 시간가중 8.37% 관측 ⇒ Gate 1의 "단일 분할 고정" 문장을 **철회**(rate2·3·4 인용 가능 셀은 불변). **전문은 CONSENSUS §1-1(rev15, "★실질 산출 1" 철회 블록)·§3 항목30 — 수정 금지, 인용만.** ★★★★★★★★**같은 디렉터리에 G1-c(job 877974, 2026-08-11) 후속 — Gate 2-S Granite r3·r4 §5.6.1 명명 제한 조건부 해제, 새 성능 판정 아님, 크기 인용 셀 확대 아님(불변, Zamba2 r2 하나).** `PREREG_G1C_2026-08-11.md`(사전등록)·`gate1c_result_877974.txt`(판정 출력, rate{2,3,4,6})·`gate1c_analyze.py`(분석기)·`runtime_source_manifest_gate1c_877974.sha256`(877757과 15/15 바이트 동일)·`manifest_diff_gate1c_877974.txt`. **전문은 CONSENSUS §1-1(rev22, 2026-08-11 G1-c 블록)·§3 항목41–43 — 수정 금지, 인용만** |
| `../../workspace/engine-port/results/p1_gates/gate2/` | ★★★★★**Gate 2 rev4 본 캠페인(jobs 875344/875346, 2026-08-07) 원자료 — 메인 세션 원자료 독립 재현 확인(2026-08-09), claims-auditor 감사 기록 위치 미확인, 새 성능 판정 아님(정본 반영 복구).** `g2_report_zamba2-27b_875344.json`/`g2_report_granite-40-h-micro-base_875346.json`(`per_arm_x60` 포함 판정 산출)·`g2run_875344.out`/`g2run_875346.out`(실행 로그)·`g2_analyze.py`(분석기, `:538-543`이 primary A3-vs-A4 계산, `:734`가 A2 서술 위치 — R1′/R2′는 이 스크립트가 계산 안 함). **전문은 CONSENSUS §1-1(본 캠페인 블록, rev17)·§3 항목34 — 수정 금지, 인용만.** ★★★★★★**E-A(mixed-chunk 레버, jobs 875654/875657/875661, 2026-08-08~09) 원자료 — claims-auditor 감사 완료, 판정 조건부(문구 강한 제한), 새 성능 판정 아님(fused-측 조율 진단).** `PREREG_G2EA_2026-08-07.md`(사전등록+§14 자기인지 위험)·`g2ea_report_zamba2-27b_875657.json`/`g2ea_report_granite-40-h-micro-base_875661.json`(판정 산출)·`g2earun_875657.out`/`g2earun_875661.out`(실행 로그)·`g2eaprobe_875654*`(Stage 1 레버 실현 확인, HOLB jsonl 포함)·`g2ea_analyze.py`(분석기, `:662-680`이 방법론 게이트 #31의 근거). **전문은 CONSENSUS §1-1(E-A 블록, rev16)·§3 항목31–33 — 수정 금지, 인용만.** ⚠️**같은 디렉터리의 `PREREG_GATE2_2026-08-06.md`는 커밋 `c47fad0`으로 §14 addendum이 반영돼 워킹트리 클린이다(2026-08-09 확인 — "미커밋" 기록은 stale, 정정)** — 단 §11-3(3차) 감사는 면제된 채 제출됐다는 사실(§14.1)은 불변이라 "감사를 통과한 사전등록"으로는 여전히 인용 금지(인용 금지 목록 항목12). G2CTRL(`PREREG_G2CTRL_2026-08-07.md`)도 커밋 상태 확인됨(2026-08-09) — 동일 정정. ★★★**HOLB G5 재채점 + 874601 G3 라벨 정정(jobs 874601/874602/874632/874633/874635, 2026-08-09) — 새 성능 판정 아님.** `g2holb_g5_tost_rescore_2026-08-09.json`(72셀 전량 재채점 산출, G5=UNDETERMINED)·`g2holb_g5_tost_rescore.py`·`g2holb_g5_tost_summary.py`·`g2holb_report_zamba2_874601.json`(sha 추출 `KeyError` 하네스 결함 원자료). **전문은 CONSENSUS §1-1(rev18, 2026-08-09 두 번째 정본 반영 건 A/B절)·§3 항목35 — 수정 금지, 인용만.** ★★★**job 876699(T4-1, 2026-08-09~10) TIMEOUT 사후분석 + 공유 하네스 무한대기 수정 + Gate 2-S 설계 감사(2026-08-10) — 새 성능 판정 아님.** `g2det_876699.out`/`g2det_876699.err`(52분 스톨·SLURM TIMEOUT 원자료)·`g2det_analyze.py`(2026-08-10 수정, `on_clean` NO-VERDICT 가드)·`g2_holb_phaseA_lib.sh`(`:102-139`이 사건 경위·180s 근거 주석)·`PREREG_GATE2S_2026-08-09.md`(rev1–4, 3회 NO-GO 감사 이력, §0.0.A/B). **전문은 CONSENSUS §1-1(rev19, 2026-08-10 세 번째 정본 반영 건 A/B/C/D 블록)·§3 항목35(개정)·37 — 수정 금지, 인용만.** ★★★★★★★**Gate 2-S 첫 유효 결과(jobs 877756/877757, 2026-08-11) 원자료 — claims-auditor 적대 감사 "조건부 등재 가", 새 성능 판정 아님(별도 트랙, §1-1 대체 아님).** `g2s_report_zamba2-27b_877756.json`/`g2s_report_granite-40-h-micro-base_877757.json`(판정 산출, `design_conformance.conformant=true`)·`g2s_status_zamba2-27b_877756.json`/`g2s_status_granite-40-h-micro-base_877757.json`(`MEASURED_AND_SCORED`)·`g2s_analyze.py`(`:63`이 방법론 게이트#20 새 사례의 근거)·1차 실패 이력 `g2s_report_*_877107/877109`(하네스 결함, primary 0개). **전문은 CONSENSUS §1-1(rev20, 2026-08-11 블록)·§3 항목18(개정)·34(개정)·38·39 — 수정 금지, 인용만.** ★★★★★★★★**G1-c(job 877974, 2026-08-11) 반영 — `PREREG_GATE2S_2026-08-09.md` §8.9에 사후 addendum(정오표, 원문 미수정) 추가.** addendum은 (a) Granite r3·r4 전제 VERIFIED 승격 (b) 그 승격이 명명 층에만 미치고 크기 인용 자격(F-계열 gate 산출)에는 미치지 않음을 코드로 확인 (c) Zamba2 근거표의 pop-C 인용을 제거하고 양 모델 공통 근거를 `max(decode_bs)<36`으로 좁힘 (d) G1-c의 `t_total_s` 인용 금지 추가, 4가지를 담는다. **전문은 CONSENSUS §1-1(2026-08-11 G1-c 블록, rev22)·§3 항목41–43 — addendum 자체는 사전등록 판정 규칙을 바꾸지 않음, 인용만.** ★★★★★★★**E1 addendum(jobs 877756/877757 재집계, GPU 증분 0, 2026-08-11) — claims-auditor 적대 감사 완료, 판정=등급 하향된 조건부 채택(4셀 전부), 새 성능 판정 아님.** `PREREG_G2S_E1_ADDENDUM_2026-08-11.md`(사전등록, §0의 post-hoc 자백 포함)·`g2s_e1_premise.py`(구현, `gate1b_analyze.py`에서 통계량·문턱 동사 수입 — `g2s_analyze.py` 미수정·미import)·`g2s_e1_premise_877756_877757.json`(4셀 pooled/rep별 max·무결성 진단·상태, 4셀 전부 `VERIFIED_AT_SAMPLED_INSTANTS`). E1 §2.5가 주장한 "G1-b/G1-c보다 밀도 페널티를 덜 받는다"는 **반증**됐다(결정 관련 pop-A 관측 수 5–6× 적음) — 실질 우위는 반복수(n=10)·셀 일치뿐. **전문은 CONSENSUS §1-1(2026-08-11 E1 addendum 블록)·§3 항목44·45(신규)·18·39(재발 追記) — `g2s_analyze.py`는 재현성 보존을 위해 수정하지 않음(`PREMISE_LABEL`·아카이브 `premise_labels`가 Granite r3·r4를 여전히 `unverified`로 고정하는 것은 알려진 결함으로 등재, 문서 층에서만 승격됐다는 정오표 병기 필수), 수정 금지·인용만.** |
| `../PRIZE_SIZE_ARGUMENT_2026-08-16.md` | ★**"상금 크기"(P-dyn 추가 이득) 논증 — rev3, 정본 rev32 반영본, 2026-08-16 doc-steward 등재(생성은 다른 병행 세션, GPU 지출 0).** claims-auditor 적대 감사 완료(기준1 CONFIRMED(강화)·기준2 PLAUSIBLE(조건부)·기준3 CONFIRMED). 결론 = 원 사양(C2 탄력도+도달 가능 영역+§1-20 coupling tax 조립)은 **성립 불가**(축 A는 §3 항목17·21·25가 REFUTED한 격자 이전의 4회차, C2-R 재확인으로 **원리상 영구 정지**) — 대신 §1-13 각주 E/F(n=4, 정본술어)의 **scoped 천장 진술**(LO/HI argmax 동일 static d44, 완전 예지 오라클 이득=점추정 0)을 복원. ★부수: §1-19 "+2.1%"가 claims-auditor 1차 재계산(당시 **독립 재확인 전**)으로 phase-mean/pooled 집계·legacy/정본 술어에 따라 +2.3%~+6.1%~미정의(phase B 전 arm 0%)로 갈리는 **집계·술어 산물**임이 드러남 — **이 재계산은 2026-08-16 R1(아래 `ORACLE_REANALYSIS_2026-08-16.md` 행)에서 result-analyst 독립 재현으로 확정됐다**(§1-19 참조, 등급 상향 아님). doc-steward 라우팅으로 §1-4·§1-19·§1-20 소비처(matrix·roadmap)·§5-5·§1-17에 인용금지/스코프 정정 5건 전파(이 표 항목이 그 정정을 기록한 rev). 새 성능 판정 0건·정책 순위 변경 0건. |
| `../../workspace/engine-port/results/slo_sched/ORACLE_REANALYSIS_2026-08-16.md` | ★★**R1(he2 재채점 provenance 확정)·R2(sgptv TTFT⊗ITL 분해) — rev2, result-analyst 산출 + claims-auditor 적대 감사 완료(2026-08-16), GPU 0 · 새 서빙 실험 0건, 정본 등재 완료(doc-steward, 같은 날).** 위 `PRIZE_SIZE_ARGUMENT_2026-08-16.md` §5가 지정한 선행 재분석. **R1**: §1-19 "+2.1%" provenance 확정(phase-mean +2.28/+2.35% ↔ pooled +5.88/+6.12%, 정본술어로는 오라클 미정의) + §1-20 "+16%"가 정본술어에서도 견딤(+16.54%)이나 ITL 도너 동률로 "116"이 tie-break 의존 + TTFT 임계 ±10%에서 크기 0.00~+19.75% 요동(§1-19·§1-20 참조). **R2**: §1-32(신규) — "+16%" 크기는 sgptv rate-swing 격자에서 재현되지 않으나(최대 +5.64%) "116>108 coupling tax 없음"은 결정량이 항등식이라 등재 금지(d54/d64 미실행). **코드 결함**: `he2_bench.sbatch:92`가 게이트 #7 버그(`dur=max(dur,d)`) 잔존 — `HE2_RESULT` 라인 인용 영구 금지(정본 숫자 자체는 `sum(dur)` 독립 재계산으로 무사). rev1→rev2 사이 claims-auditor 지적 6건 + 재현 경로 결손 1건(3회차, 등재 시점에 닫힘) 반영. **새 성능 판정 0건 · 등급 상향 0건.** 스크립트 `oracle_reanalysis_2026_08_16.py`, 수치 `oracle_reanalysis_2026-08-16.json`. 정본 반영 = §1-19·§1-20·§1-32(신규)·§3 항목39(追記)·59(신설), `PROJECT_STATUS.md` "다음 실험 gate" #14–16(신설)·"방법론 게이트" #40(신설) |
| `../../workspace/engine-port/results/slo_sched/G16_RESULTS_2026-08-17.md` | ★★★**G16 캠페인 결과(rev2, claims-auditor 적대 감사 완료, 2026-08-17) — R2 결정량②(§1-32)의 재정식화판 산출, 양 phase `ITL_SATURATED`, 정본 등재 완료(doc-steward, 같은 날).** H-1~H-8 판정·exact `Δ_SLO` band·A-2 residency 스코프 크기·`gap_upper` 비식별 구조 전문은 §1-33. 사전등록 `PREREG_G16_RULES_REV3_2026-08-16.md`(addendum C/D/E) · 수치 `G16_RESULTS_2026-08-17.json`(`analyzer_sha256=2c9626f2…`) · 감사 재현 경로 `audit_g16_results_2026-08-17/`·`audit_g16_prereg_c93_2026-08-17/`·`audit_g16_saturation_2026-08-16/`·`residency_scope_2026-08-17/`. **새 성능 판정 0건 · 정책 순위 변경 0건.** ★**"gate #16을 닫았다"고 쓰지 말 것**(원문 문턱 판본은 rate 축에 잔존). 정본 반영 = §1-33(신설)·§3 항목65·66(신설), `PROJECT_STATUS.md` "다음 실험 gate" #16(갱신)·"방법론 게이트" #45·46(신설) |
| `../../workspace/engine-port/results/s8_scaleup/G13_RESULTS_2026-08-21.md` | ★★★**gate #13 job-축 캠페인 결과(2026-08-21, 메인 세션, 사전등록·블라인드·감사 완주) — 양 arm `PASS`(강한 지지(범위 한정)), 정본 등재 완료(doc-steward, 같은 날).** `σ̂_job`(M8 0.063%·Ha8 0.190%)이 설명 대상 격차(5.12%·9.20%)를 3×UB95 기준 각각 18.4배·4.9배 못 미친다. 사전등록 `PREREG_G13_2026-08-20.md`(§10 감사 부록) · 실행기록 `G13_CAMPAIGN_LOG_2026-08-21.md` · 원자료 `G13_ANALYSIS_2026-08-21.json`+`_REFUSED.json`(자기거부 증거). **성능 판정 0건**(분산 측정)·HE0 불변. ★**"gate #13을 닫았다"고 쓰지 말 것**(불변) · **`Δ_batch` 미측정**(배치 축 없음) · 노드 성분 계수 0.400·유효 df≈1(≥3노드 부분 미충족) · `grand_mean_r` arm 비교·2.91/3.058/3.114 대조 금지(인용정지 (a)(b) 해제 없음). 정본 반영 = `PROJECT_STATUS.md` "다음 실험 gate" #11 레지스트리(신설 행)·#17(2026-08-21 갱신)·"방법론 게이트" #51(신설) |
| `../../workspace/engine-port/results/s8_scaleup/S0A_VERDICT_2026-08-20.md` | ★★**S0(a) 부팅 내부 잔차 측정 — rev2(2026-08-21 doc-steward 등재, claims-auditor 감사 REFUTED로 초판 결론 철회).** *"C-g의 전제가 성립하지 않는다"*는 **철회**, 비용 정합 재판정 결과는 **"C-g 이득은 교대 속도의 함수"**(부팅당 8구간 교대는 승, 4·2구간은 패). `r`이 창 의존량임을 신규 등재(Ha8 30s 3.025 vs 60s 3.116, 정본 채택구간 밖). ★**게이트가 아니다**·gate #13 불변(S0(a)는 임계경로 밖). 정본 반영 = `PROJECT_STATUS.md` "다음 실험 gate" #17(2026-08-21 갱신)·"방법론 게이트" #53(신설) |
| `../../workspace/engine-port/results/slo_sched/{PREREG_G18_PROBE_2026-08-20.md,G18_RATE_VALUE_2026-08-20.md}` | ★**gate #16 rate-축 내부 구간 프로브 — 2026-08-21 doc-steward 등재, 트랙 보류(HOLD).** rev1 규칙층 REFUTED → rev2 NO-GO(새 死因 5건) → 사용자 결정으로 보류(gate #13 완주 후 재개 판단). ★**정본 오염 경로 발견·수리**(`g16_analyze.py:953` 글롭 매치, 실제 오염 0건) — `G18_RATE_VALUE_2026-08-20.md` §6이 그 대응. `G18_PROBE_STOP` 존재, 피더 재기동 금지. **gate #16 불변**(닫히지 않음). 정본 반영 = `PROJECT_STATUS.md` "다음 실험 gate" #17(2026-08-21 갱신)·"방법론 게이트" #54(신설) |
| `../../workspace/engine-port/results/kernel_mech/DESIGN_KERNEL_MECH_REV3_2026-08-20.md` | ★**kernel_mech rev3(Stage A 전용) — §3 차단 2건 설계 본문 수리(2026-08-21 doc-steward 등재).** §3.1 셀 라벨을 realized로 판정(시간가중 조건+혼합구간 폐기) · §3.2 `gap_frac` 항등식 가드(union 정의+변이 테스트). 후보 (vi) 클럭 2차 강등. ★**이 수리 자체는 미감사**(다음 세션 재감사 대상, 사전등록으로 아직 안 넘어감). Stage B 폐기(P1 프로브, 2026-08-20)는 불변. 정본 반영 = `PROJECT_STATUS.md` "다음 실험 gate" #17(2026-08-21 갱신) |
| `../../workspace/engine-port/results/slo_sched/S6_POWER_2026-08-21.md` | ★**S-6 telemetry-OFF 대조 검정력 계산 — 규칙층 입력, claims-auditor 감사 `NO-GO`(2026-08-21), 등록 `n` 없음, 정본 등재 완료(doc-steward, 같은 날).** `PREREG_G16_RULES_REV3_2026-08-16.md` addendum B-2의 "n≥2(≈0.3 GPU-hr)"를 **반증**(필요 n은 δ·사전값의 함수, 구속 셀 HI `M_ttft`에서 5–18쌍, 부팅 단가 실측 0.1272 GPU-hr) — 대체 단일값은 등재 금지, 함수 형태로만 인용. 死因 4건(OFF 다리가 하네스·분석기에서 구조적으로 탈락·비오염 논증이 이름 약속뿐[교훈 #57 재발]·estimand 단위 불일치[절대 ms vs 상대 %]·pooling 판단 근거가 귀무분포 없는 범위통계량) 전부 GPU 0의 문면 수리로 해소 가능, 하네스 착수 **전**에 잡힘(설계 반려이지 계측 축 분리 자체의 반증 아님). **payoff는 1 arm(d44) 공통 시프트 크기로 축소**(he2/sgptv 계열 절대값 이식 불가, arm×telemetry 교호작용은 1 arm으로 측정 불가). 원자료 `S6_POWER_2026-08-21.json`, 스크립트 `s6_power.py`, 커밋 `ace05b8`. 정본 반영 = §3 항목61(정정)·67(追記)·`PROJECT_STATUS.md` "방법론 게이트" #41(정정)·"다음 실험 gate" #17(2026-08-21 갱신 2차) |

`deprecated_reports/`(2026-07-24부터 [`../deprecated/reports/quarantine_engine_port/`](../deprecated/reports/quarantine_engine_port)) = 초기 triage·포팅·모델별 평가·구 핸드오프·구 리포트. **이력 보존용, 현재 결론과 충돌 가능.**

---

## 5. 열린 항목

1. ~~**변화-trace 기반 재검증**~~ → ✅ **완료 (2026-07-17, jobs 856889–856975)**. n≥4 캠페인으로 **HE0 견고 확정**(§1-7, 5.4σ) + **게이트 정체 규명**(§1-10/11).
2. ~~**게이트 정교화 필요?**~~ → ✅ **성격이 바뀜**: 게이트는 지능적 제어가 아니라 **auto-tuner**(§1-10). 살릴 값어치가 있다면 **ratchet의 조기 정지 수정**(정지 규칙이 d34에서 멈춰 d44를 놓침) — 단 그래봐야 천장은 "best-static 매칭"이라 payoff는 *튜닝 자동화*뿐.
3. ~~**컨트롤러 CPU 오버헤드**~~ → ✅ **직접 계측으로 死 (2026-07-17, jobs 857111/2)** = §1-12.
4. ~~**시스템 노이즈의 정체**~~ → ✅ **종결 (2026-07-17)** = **노이즈가 아니라 메트릭 절벽**. 상세 [bench_noise_root_cause.md](bench_noise_root_cause.md).
   워크로드는 4런 전부 **동일**(fingerprint 일치)이고, 하부 섭동은 **throughput 3%·ITL 8%뿐**. 증폭기는 **r8이 하필 TTFT≈SLO(3s) 경계에 앉은 것** — 과부하 큐의 TTFT 평탄역이 3% 결손에 1.5s→3.7s로 이동해 **임계선을 넘음 → goodput 반토막**. rate로 재현: **3=견고(200/200×3) / 8=불안정(400,400,357,206) / 12=견고(142,141,142)** ⇒ 불안정한 건 **경계 regime뿐**. ★**GPU 클럭 throttling 가설 철회**(3.5×가 아니라 3%만 설명하면 됨). ★**자원 격리 불필요했음**. **stationary 부활 조건**: 용량(d24≈6.3/s) 아래 rate에서 측정하거나, 임계 지시함수 대신 TTFT 분포/용량 지표 사용(현 goodput은 과부하서 런 길이 의존 = ill-posed).
5. ~~(낮음) 얽힘-aware 행동모델의 정밀화~~ → **천장이 "static 매칭"으로 확정**(관대·tight 양쪽, §1-7·§1-17). 동적 upside는 **모든 SLO 엄격도에서 닫힘**. payoff 없음. ★★**스코프 부착(doc-steward, 2026-08-16)**: "천장"·"payoff 없음"은 §1-7·§1-17이 측정한 **달성된**(observed) 정책 계열(single-worker·SM-split·reactive) 한정이다 — **달성 가능한 천장**(HE0가 함의하지 않는 다른 명제)까지 "확정"으로 읽지 말 것(아래 §8(a)(b)가 그 통로를 여전히 열어둔다). §3 항목57, `PRIZE_SIZE_ARGUMENT_2026-08-16.md` §3.
7. ✅**완료·종결 (2026-07-19)**: **tight-SLO 서브트랙.** (b) SLO 관행 조사(`serving_slo_survey.md`): 인터랙티브 주류(chat/voice/code, TTFT 100–400ms)가 tight regime. (a) **컨트롤러 재튜닝 직접 측정**([interactive_slo_retune_plan.md](interactive_slo_retune_plan.md) §9, jobs 860415–860514): chat(300/50)으로 컨트롤러 실제 재튜닝 → **§1-17 = HT0 확정**(d44 73.2%≫bind+GATE 44.3%, 10σ). ★**§1-16의 "tight 동적 우위"는 재스코어 아티팩트로 반증** — 재스코어는 컨트롤러 *행동*을 못 봤다(방법론 교훈: SLO를 목적함수로 바꾸는 실험은 반드시 컨트롤러를 그 SLO로 재튜닝해 직접 측정). code(100ms)는 HT-neg(무경쟁 66%, 물리 불가). ⇒ **single-worker·SM-split·reactive 동적 제어(SLO-aware/binding-first/feasibility-gate)는 관대(3s, §1-7, n≥4)·tight(chat 300/50, §1-17, n=4) SLO 양쪽에서 best-static을 못 넘는다 — 이 범위는 확정, 반증 실패.**
   ★**정정(2026-07-24, claims-auditor 감사)**: 위 "⇒ 동적 제어 트랙 완전 종결"이라는 이전 표현은 **overclaim이라 철회**(취소선 아님, 이 정정으로 대체) — [`../PROJECT_STATUS.md`](../PROJECT_STATUS.md)는 "완전 종결"을 말하지 않으며 Claim D(true dual-worker coupling 감소)·Claim E(Hybrid-informed policy)를 **미검증**으로 명시적으로 열어둔다. §1-20(oracle 분해)도 canon 스스로 disaggregation ceiling +16%가 decoupled substrate엔 열려 있다고 정량화해 "완전 종결"과 정합하지 않았다. **종결된 것은 위 좁은 범위(single-worker·SM-split·reactive)뿐**이며, 미결 항목은 §5-8 참조.
6. ★**long-context 실 trace로의 전환** (사용자 발의 2026-07-17) — **계획 단계**. 근거: **모든 서빙 반증이 short-context**(ShareGPT 98%가 L<2k)이고 창립 동기의 Diff A는 **L≈3k에서 교차해 열린다** ⇒ 결론의 **유효 경계**가 컨텍스트 축에서 미확인. 단 **layer-aware 부활 경로 아님**(Diff B는 long-L서 ≈1.0으로 닫힘). 실제 stake = **최적 static 위치 · 얽힘 병목의 KV 재편 · HE0 반전(혼합 trace에서만 가능)**. 블로커: **Zamba2-2.7B ctx 4096**(모델 교체 필수 → 전 baseline 재측정) · **goodput SLO가 long prefill서 붕괴**(전 정책 0). 상세·단계 게이트 [longcontext_trace_plan.md](longcontext_trace_plan.md).
   ★★**Stage 0(L−2) 게이트 실행 — non-binding, 2026-07-26 (★★★2026-07-28
   철회, C1 CONFIRMED)**(§1-21, [`stage0_verdict_2026-07-26.md`](stage0_verdict_2026-07-26.md)):
   2026-07-26엔 이 항목이 전제하는 "decode floor가 ctx로 상승해 운영점서
   binding해진다"가 **ctx≤16k에서는 반증**됐다고 봤으나(D16≡D108(1.00±0.01)),
   근거였던 D108 앵커가 실은 decode 16 SM이었음이 claims-auditor 감사(C1
   CONFIRMED, §1-21)로 확인돼 이 판정을 **철회**한다. **L−2 게이트는 사실상
   아무것도 측정하지 않았다** — 따라서 `longcontext_trace_plan.md` §6의 게이트
   규칙이 예정한 "L−1 이상은 진행 근거 없어 멈춘다"는 **"게이트 실패"가 아니라
   "게이트 미실행"**으로 정정한다(재개 권고 아님, 판정 부재라는 뜻). 이 항목
   (long-context 실 trace 전환)은 **판정 이전 상태로 되돌아가 여전히 열려
   있다** — 남은 미측정은 ctx≤16k을 포함한 전 구간(L−2 재시도부터).
8. ★**미결(종결 아님, 2026-07-24 스코프 정정으로 신설)**: §5-7의 "완전 종결"은 아래 세 갈래를 배제하지 않는다.
   - **(a) dual-worker(decoupled) 동적** — [`../PROJECT_STATUS.md`](../PROJECT_STATUS.md) Claim D("역할별 queue/host issue loop/CUDA stream을 실제로 분리하면 single-worker control-plane coupling을 줄일 수 있다")는 **서빙 측정 0건**(증거 수준 "미검증"). §1-20이 정량화한 **+16% disaggregation ceiling**(92 prefill SM + 24 decode SM = 116 > 108 = 단일-GPU coupling tax로 불가능)은 **별도 디바이스 풀 + hybrid state transfer를 갖춘 decoupled substrate에만 열려 있다**. ★★**정정(2026-07-24, engine-porter 코드 리뷰, 읽기전용, [`r2_decoupling_review_2026-07-24.md`](r2_decoupling_review_2026-07-24.md))**: `PDMUX_TRUE_DUAL_WORKER=1`는 이 decoupled substrate에 **해당하지 않는다** — file:line 근거로 확인한바 두 host issue thread/role별 task queue/immutable `ExecutionContext`/thread-local role(ContextVar)만 분리하는 **control-plane dual-worker**이고, running batch(`max_running_requests`)·KV/mamba pool·SM 파티션(`SharedGpuArbiter`의 단일 `stream_index`)은 **전면 공유**된다(92+24=116의 별도 device pool이 아니라 ≤108 단일 coupled index). §1-4 死因 얽힘(공유 running-batch+KV)의 substrate가 불변이므로 이 구현은 **구성상 +16% headroom에 도달 불가**하며 coupled ceiling(+2%, §1-20) 위에 앉아 있다. state-transfer 경로·mamba conv/ssm state migration은 **코드에 전무**(스캐폴딩조차 없음). ⇒ **Claim D는 "control-plane coupling 감소"로만 유의미하게 측정 가능**, "얽힘 깨기"로 팔 수 없다. 부가: admission latch(`r2_admission_limited`)에 **known-latent stale-True 버그** 확인 — split batch가 None으로 배수되면 재평가 경로가 없어 latch가 True로 고착되어 prefill admission을 영구 차단할 수 있다(clear 경로 부재). **사용자 결정으로 현재 수정하지 않고 보류.** R2는 GPU correctness gate를 통과한 이력이 없다(`results/r2_eval/` 디렉터리 미생성, `architecture=true_dual` telemetry 전무). 이 headroom이 (다른 substrate에서) 실현되는지는 여전히 미검증. 게이트는 PROJECT_STATUS "다음 실험 gate" §2 참조.
   - **(b) 비-SM-split lever** — §1-4 얽힘의 死因은 **공유 running-batch capacity·KV**(SM 분할 자체가 아님). admission-control 또는 KV-aware한 lever로 이 死因을 직접 겨냥하는 시도는 **구현조차 되지 않았다**. 지금까지 종결된 것은 전부 *SM-split* 기반 컨트롤러(SLO-aware/binding-first/feasibility-gate)뿐.
   - **(c) 충돌 regime 워크로드** — "동적이 이길 disjoint-feasibility escape hatch가 없다"는 결론은 §1-18(mix-스윙, **n=2**)·§1-19(극단 disjoint, **n=1~2, overload-only**)의 **underpowered 탐침**에 근거하며, 이 자체가 §2-4 방법론("n≥4 없이 정책 결론 금지")에 못 미친다. long-context 혼합 trace(§5-6의 stake 중 "HE0 반전")도 아직 미측정.
     ★**HE0-reopen 벡터1(n≥4 재시도, 2026-07-24 실행·2026-07-25 판정)**: G2.0 short-ctx disjoint 스윕(Zamba2-2.7B, ctx4096, Phase A in2048/o32 vs Phase B in2048/o512–1024, rA5)으로 (c)를 n≥4로 재검증 시도. **g2_0_full**(n=4/mode, `../workspace/engine-port/results/g2_0_full/disjoint_verdict_2026-07-24.md`)는 razor-thin real disjoint를 보고(feasible-A={d16,d44} ∩ feasible-B={d54}=∅, d54 Phase-A 0.852 fragile). **g2_0_hard**(n=6–10/mode, hardening axis 2개 + claims-auditor 독립 재채점, `../workspace/engine-port/results/g2_0_hard/hardened_disjoint_verdict_2026-07-25.md`) 판정: **ILL-POSED at rA5, escape hatch 근거로 "지지 안 됨"이나 "종결"도 아님.** 근거: (i) 동일 byte-identical Phase-A 워크로드에서 d44/d54 견고성 순위가 sweep 간 완전 반전(d44 0.969→0.695, d54 0.852→0.938), pool(n=10) 시 둘 다 ~0.86–0.90로 통계적 구분 불가 — 이는 §1-14 "메트릭 절벽" 기전과 동일한 TTFT 3s-cliff bimodality. (ii) "축1서 feasible-B가 d34로 넓어져 disjoint 소멸"은 물리적 완화가 아니라 **per-request ITL-p95 percentile-window 아티팩트**(OB512→1024서 ITL 중앙값은 상승하는데 p95만 하락 — 초기 고정 스파이크가 더 긴 출력에 희석). ⇒ 견고한 disjoint도 견고한 공유 static도 입증 안 됨. **§1-20(spatial coupling-tax, 116>108, disaggregation +16%)과는 무관 — 별도 축(시간적 disjoint-feasibility vs 공간적 SM-budget), §1-20은 이 결과로부터 영향 없음.** **방법론 교훈 강화**: 또 절벽 위에서 측정(§3 gate #6 위반) — g2_0_hard가 gate #6을 지키려 워크로드를 바꿨으나(axis1) 그 자체가 새 percentile 절벽에 올라앉아 실패; feasibility 판정 전 **capacity de-cliff 선행** 필수, "결과가 바뀌었는가"가 아니라 "메트릭이 여전히 절벽/percentile 경계에 앉았는가"로 de-cliff 여부를 검증해야 함. **de-cliff 재스윗 pending**(rA 추가 인하로 Phase-A p90≪3s 확보 → 출력-길이 불변 ITL 지표 → ≥2-SM-step 간극 n≥6 paired). (c)는 여전히 **열려 있음** — 이번 라운드로도 확정도 반증도 안 됨.
   ⇒ **정확한 종결 범위**: single-worker·SM-split·reactive 동적 제어가 관대·tight SLO 양쪽에서 best-static을 못 넘는다는 것만 확정. **(c) 충돌 regime(short-ctx)은 2026-07-25 `g2_0_raconf` 확증으로 CONFIRMED closure(scoped) — 아래 참조**. 트랙의 나머지(dual-worker architecture (a), non-SM-split lever (b), **long-ctx 충돌 regime**)는 여전히 **열려 있음**.

   ★**de-cliff stage-1 완료(실험 2026-07-24~25, 기록 2026-07-25, jobs 863880–863948,
   `../workspace/engine-port/results/g2_0_decliff/decliff_verdict_2026-07-25.md`)**:
   위 `g2_0_hard`가 권고한 de-cliff 재스윗의 1단계(capacity de-cliff only, feasibility
   판정 아님). `rA{2,3,3.5,4}×{d16,d44,d54}`를 n=3(coarse)로 스캔한 결과 **`rA=2`만
   전 split에서 clean off-cliff**(mean p90<2.0s ∧ std p90<0.15s); `rA≥3`은 전부
   여전히 bimodal(`g2_0_full`/`g2_0_hard`와 같은 절벽 재현). `rA=2`를 n=6으로 확증.
   **판정 = "견고한 off-cliff disjoint를 찾지 못함 = PLAUSIBLE closure, CONFIRMED
   아님"**: 유일한 clean off-cliff 지점 `rA2`에서 static `d54`가 양 phase를 동시
   커버(Phase-A frac_good 0.974±, TTFT p99≤1028ms; Phase-B ITL-p95 42.3±0.10ms,
   frac_good 1.0, 출력-불변 지표로도 견고) — `feasible-A={d16,d44,d54} ∩
   feasible-B={d54} = {d54} ≠ ∅`. 후보 disjoint는 `rate≥3.5`(전부 bimodal 절벽,
   n=3)에서만 재등장 — 새로 해소된 regime이 아니라 기존 절벽의 재확인.
   ★**"종결" 불가(claims-auditor 반증)**: (a) split→TTFT gradient가 off-cliff에서도
   살아있음(`rA2` p99 `d16` 712±37ms vs `d54` 1028±101ms, `t≈7.2`, 단조) → rate↑
   시 `d54`가 절벽을 먼저 넘는 disjoint 발생 경로 미배제; (b) `d54`-배제 onset
   (~rate 3.0–3.5)이 정확히 **미측정·n=3·bimodal 전이대**라 논증만으론 못 닫음;
   (c) 핵심 등식 "`binding-A` ⟺ `on-cliff`"는 "`d54` 배제"와 "`d54` 절벽-flicker"를
   혼동한 **미증명 경험명제**. **선행 필수(pre-registered stage-2, 진행 중, 미실행)**:
   narrow `rA{2.25,2.5,2.75,3.0,3.25}×{d16,d44,d54(+d34)}` n≥6 확증 sweep — 반증
   표적은 어떤 rate에서 `d54` 견고히 <0.7(p90>3s, unimodal) ∧ `d44`/`d16` 동시에
   견고히 ≥0.95·off-cliff(p90<2s)이면 disjoint 실재 → **벡터1 REOPEN**. **scope
   한정**(필수): {Zamba2-2.7B, ctx4096, Phase A in2048/o32, Phase B in2048/o512@rB4,
   triton attn+mamba, `disable-radix-cache`, cudagraph-ON, A100 108-SM
   green-context pdmux, SLO=TTFT 3s ∧ per-req ITL-p95 50ms, inter-phase drain된
   순차 2-phase} — "hybrid엔 disjoint 없음"으로 일반화 불가. **drain caveat**:
   closure는 얽힘 억제(drain) 조건 관측 = 필요조건 bound이지 hot varying-trace
   (Claim C 얽힘) 실증 아님. ★**§1-20 방화벽 유지**: 시간적 disjoint(단일 static이
   양 phase를 시간축에서 커버)와 공간적 coupling-tax(92+24=116>108, disaggregation
   +16%)는 별개 축 — "단일 static으로 충분 ⟹ coupling tax 없음"으로 새지 않는다.
   §1-20은 이 결과로부터 영향 없음. de-cliff stage-1 시점엔 (c)가 **한 눈금
   전진**(rA5 ILL-POSED → rA2 PLAUSIBLE-not-CONFIRMED closure)이었으나 여전히
   **열려 있었다** — stage-2 확증 sweep 전까지 어떤 방향의 결론도 채택하지 않았다.

   ★★**(c) 최종 종결 — CONFIRMED closure(scoped), 2026-07-25**: 위 stage-2
   확증 sweep 두 단계가 완료됐다. **`g2_0_rasweep`**(120 job,
   `rA{2.25,2.5,2.75,3.0,3.25}×{d16,d34,d44,d54}×n6`)이 off-cliff sub-band
   (rate≤2.75)에서 disjoint를 재확인하지 못해 전이대를 rate 3.0–3.5로
   좁혔다(raw jsonl만 존재, 별도 verdict 미작성 — provenance
   `../workspace/engine-port/results/g2_0_rasweep/`). 그 좁혀진 창을 겨눈
   claims-auditor pre-registered **`g2_0_raconf`**(24 job, `rate{3.5,3.75}×
   {d44,d54}×n6`, 결정 규칙: 어떤 rate서든 `d54` 견고히 <0.7(p90>3s, unimodal)
   ∧ `d44`/`d16` 동시에 견고히 ≥0.95·off-cliff(p90<2s)면 disjoint 실재→REOPEN,
   아니면 `d54`가 양 phase를 동시 커버하는 companion collapse면 CONFIRMED)가
   **companion collapse로 판정**: rate3.5 `d44` 0.953±0.035 ≈ `d54`
   0.948±0.035(d54 failTTFT=0/6); rate3.75 `d44` 0.932±0.042 < **`d54`
   0.948±0.062**(d54가 오히려 높음). REOPEN 전제 둘 다 붕괴(`d54`는 어느
   rate서도 <0.7이 아니고, `d44`도 어느 rate서도 견고히 ≥0.95가 아님 — 웜업성
   TTFT-blowup이 rep 하나에서 0.844–0.875까지 끌어내리며, 이 blowup은
   **split-대칭적**이라 disjoint를 만들지 않는다). `d54`는 Phase B의 유일
   feasible split(`d44` ITL-p95 50.7ms로 50ms SLO 초과, `frac_good` 0.188;
   `d54`는 44.1ms, `frac_good` 1.000, SD=0)이면서 Phase A도 `d44`와 대등하게
   커버 → 단일 split이 양 phase를 시간축에서 커버 → **disjoint 없음, 최종
   확정**. **magnitude는 ill-posed(metric cliff)나 순위(d54≈d44, d54
   미선-배제)는 견고**; Phase-B `d44` 0.188은 50ms 경계 위라 magnitude
   fragile·방향 견고. scope는 위와 동일({Zamba2-2.7B, ctx4096, Phase A
   in2048/o32, Phase B in2048/o512@rB4, ..., drain된 순차 2-phase,
   **rate_A≤3.75**} 한정, "hybrid엔 disjoint 없음"으로 일반화 금지). §1-20
   방화벽 불변(시간적 disjoint와 공간적 coupling-tax는 별개 축). **남은
   방향**: long-context(decode floor 상승 영역, §1-5) 재검증, §1-20 spatial
   decoupling. ★**2026-07-26 갱신, ★★★2026-07-28 철회**: 2026-07-26엔 전자가
   Stage 0(L−2) 게이트로 부분 실행돼 "ctx≤16k에서는 decode floor가 운영점서
   상승하지 않아(§1-21) 이 항목도 ctx-무관으로 강화되는 방향"이라고 봤으나,
   그 근거(D108 무경합 앵커)가 claims-auditor 감사(C1 CONFIRMED, §1-21)로
   무효 확인돼 **철회**한다. **이 항목은 다시 미검증**(Stage 0이 "게이트
   실패"가 아니라 "게이트 미실행"이었으므로 강화도 약화도 아니고 원점).
   §1-20 spatial decoupling은 여전히 미실행. 상세
   [`../workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md`](../workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md),
   [`stage0_verdict_2026-07-26.md`](stage0_verdict_2026-07-26.md).
9. ★★★**(2026-08-03, 신설, ★★★같은 날 4차 속행 갱신) 최우선 — §1-28 §0의
   이분법을 어떻게 가를 것인가 — 이제 3지선다, 오프라인 분리 불가, S2(GPU)
   대기.** `DESIGN.md` §4.3.11이 명시적으로 미검증으로 남긴 잔여층("green
   context를 `(92,16)`으로 만들면 하드웨어가 실제로 그 SM 수를 부여했는가")이
   여전히 **872077 전체와 sticky 결과가 딛고 선 바닥**이다. ★**4차 속행에서
   세 번째 후보 (iii)이 실측으로 문서화됐다**(§1-30): 두 job은 같은 축이나
   `split_frac≥0.90`이 D 파티션 실행 토큰을 순수하지도 완전하지도 않게
   잡는다(E1 SPLIT 모집단이 이봉, 윗봉=C2 p50과 1–2% 일치, 아랫봉=같은
   job UNSPLIT, 슬로우 토큰의 88.9%가 UNSPLIT 라벨) — result-analyst의
   독립 재현(`S0R_REPLICATION_2026-08-03.md`)이 재확인. 남은 두 읽기(셀
   수준 현상 vs 클럭 오프셋 누출)는 **오프라인으로 분리 불가**이고, S2(GPU,
   `PREREG_S2_STICKY_ITL_2026-08-03.md`, 별도 제출 중·결과 없음)가 인과
   시험이다. 감사자 제안 **게이트 S1은 현 상태로 실행 불가**(3갈래 판정에
   "부분 실현" 분기가 없음 — 4번째 분기 추가 필요, `DESIGN.md` §4.3.15(e)).
   **`G_LEVER`/`G_FLAT` 사전등록**은 §4.3.12(d) 그대로 UNDETERMINED
   유지(4차 속행으로도 해소되지 않음) — 다음 시도는 감사자 발안 (α)/(β)에 대한
   **독립 사전등록**이 선행돼야 한다(감사자가 자기 발안의 승인 주체일 수
   없다). 상세 `DESIGN.md` §4.3.13–4.3.15, §1-30.

   ★★★**종결(2026-08-05, claims-auditor CONFIRMED scoped) — 이 항목은
   "미해소 3지선다"에서 "CONFIRMED(scoped)로 종결, 단 하드웨어 층
   미프로브"로 상태가 바뀐다.** S2(job 873015)의 독립 재현이 §1-28/§1-30의
   3지선다를 behavioural하게 닫았다: (i)는 하드웨어 형태 REFUTED·라벨
   형태 CONFIRMED, (ii)는 DISFAVOURED, 살아남는 답 (iii)("라벨이 순수·
   완전하지 않다")의 기전이 스냅샷 샘플링 케이던스(개수-서브샘플 +
   decode-busy 조건부 1/16)에서 독립적으로 도출됐다. **`DESIGN.md`
   §4.3.11이 남긴 하드웨어 SM 부여 프로브(S3)는 여전히 미실행** — 이
   항목이 완전히 닫힌 것은 아니고 스코프가 축소된 것이다. **`G_LEVER`/
   `G_FLAT`는 여전히 UNDETERMINED**이고, **E1은 4가지 독립 사유(§4.3.12(d)
   미결·sticky 기판의 estimand 전환·prefill 축 미통제·음성대조 구조적
   부재+S3 미실행)로 열리지 않는다** — **긴장 A(HE2 vs C2)는 전혀 닫히지
   않았다.** 다음 gate는 (α) sticky-ON 고-D 대조 셀(반증 가능한 유일
   실험) 우선. 상세 §1-31, `../workspace/engine-port/results/s2_sticky/
   S2_REPLICATION_2026-08-05.md`, `DESIGN.md` §4.3.16.
