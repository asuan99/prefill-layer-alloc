# `prefill-layer-alloc` project status

최종 갱신: 2026-08-20 (doc-steward — **P1 프로브 판정 반영:
`UNAVAILABLE (CUPTI×GREEN-CONTEXT)`**(job **886718**, `--exclusive
--constrain=hwperf`, node gpu38, 1분55초, `COMPLETED 0:0` — 동반
프로브 job 886752 포함 **GPU 지출 0.032 GPU-hr**). greenctx 다리에서
문서화된 시그니처(`Failed to prepare kernel for profiling` / `Unknown
Error on device 0` / exit 9)가 **정확히** 재현됐고, **두 겹 대조**로
귀속이 깨끗하다 — (a) **다리 간**: greenctx exit=9 / control(full GPU)
exit=0·에러 0건·같은 GEMM(`ampere_bf16_...`) 정상 수집(60행). (b)
★**다리 내부**: 같은 프로세스·같은 ncu 호출에서 green ctx **밖**
RNG 커널은 수집 성공(8행)하고 green ctx **위** GEMM만 실패 — 변인은
"커널이 green-context 스트림 위인가" 하나뿐(job·노드·권한·ncu 호출·
메트릭·클럭 정책·타깃·커널·차원 전부 동일). ⇒ **Stage B(ncu 커널
내부 카운터를 SM 제한 하에서 수집)는 이 기판에서 구성상 불가 확정
→ `kernel_mech` rev3는 Stage A 전용으로 범위 축소**(rev2 감사가
지정한 "문서 수정 8건" 중 Stage B 대상 최소 5건이 적용 대상 소멸,
4세션 이월의 실질 원인 해소). ★**`_ncu_target.py:68-71`의 CUPTI×
green-context 비호환 주석 — 정본이 지금까지 "미확인 리스크"로만
등재해 온 것이 이제 관측 근거로 뒷받침돼 실증됨으로 갱신한다**(단
아래 서술 한계 참조 — 내부 기전 미분리, A100-SXM4-80GB·driver
580.105.08·ncu 2025.3.1.0·CUDA 13.0.2·이 클러스터 한정, 다른
기판으로 이식 금지). 부수 확정: 선례 스크립트 `run_ncu_profile.sh:
15-17`의 권한 근거 문장("batch면 권한이 열린다")이 **불충분**함을
동반 프로브(job 886752, non-exclusive batch가 `ERR_NVGPUCTRPERM`로
거부됨)로 실측 — 실제 구분선은 **exclusive(+hwperf)**. ★★★**이것은
"성능 판정"이 아니라 "도구 타당성 판정"이다** — 깨진 것은
프로파일링이지 green context 실행이 아니다(`realized_sm=16`으로
정상 실현, 886752는 `DONE`까지 완주). **새 성능 판정 0건 · 등급
변경 0건 · 정책 순위 변경 0건 · GPU 지출 0.032 GPU-hr.** 상세
`workspace/engine-port/results/kernel_mech/p1_probe/
P1_VERDICT_2026-08-20.md`(동반 프로브 검토 `P1_886752_REVIEW_
2026-08-20.md`, 원자료 `job_{886718,886752}/`), 아래 "8B decode-SM
민감도 측정 노트"(갱신)·"다음 실험 gate" #11 레지스트리(kernel_mech
행 갱신)·#17(kernel_mech rev3 스코프 확정).
이전: 2026-08-19 (doc-steward — **gate #13 rev2 규칙층 NO-GO(死因
3건) → rev3 재감사 GO-with-caveats(등록 11.52 GPU-hr, 미제출·여전히
사전등록 아님) + gate #16 이차 표적(rate 축) 규칙층 발견(rate↓는 목표
(i) 과부하 이탈엔 옳으나 목표 (ii) `D_itl`/`Δ` 식별엔 반대 방향 —
`gap_upper` rate3 0.304ms→rate12 0.634ms, δ=1.0ms) + "다음 실험 gate"
#17의 후보 목록 정정(2건→4건, gate #16 rate 축·S-6 누락 발견) +
방법론 게이트 #50 신설(게이트 #9 열세 번째 재발 — 자기 수리를
검증하는 검사가 그 자체로 항등식이었던 사례 + "수리를 검증하는 검사는
그 수리를 되돌린 변이본에서 반드시 실패해야 한다" 게이트 제안 +
재현성 지뢰 B7). **GPU 지출 0 · 새 성능 판정 0건 · 정책 순위 변경
0건 · 등급 변경 0건**(전부 규칙층/설계층 판정). 상세 아래 "다음 실험
gate" #16·#17(갱신)·"방법론 게이트" #50(신설).
이전: 2026-08-18 (doc-steward — **G17(payoff 밴드+`gap_upper`
비식별+sticky 레버)·E-B1(shadow price) 설계 NO-GO 반영(둘 다 규칙층) +
방법론 게이트 3건 신설(#47–49) + `EXPERIMENT_PLAN_2026-08-18.md` stale
표시 + 미감사 설계 2건(gate #13 rev2·S-6)·분석기 누적 미감사 명시. 새
성능 판정 0건 · 정본 등급 변경 0건 · 정책 순위 변경 0건 · GPU 지출
0(이번 세션 job 제출 0건).**
G16 정본 승격(§1-33, gate #16)은 전날(2026-08-17) 이미 완료됐다. 이번
세션은 그 결론을 뒤집거나 확장하려는 시도 **5건**(G17·E-B1·`g2_0`
d34 "골짜기"·용량축·`out`축)을 진행했고 **전부 실패**했다 — 설계
NO-GO 2건 + 메인 세션 재검으로 REFUTED 3건. ★**G17 NO-GO**(死因 3건:
S2 격자 `U={d44..d74}`에서 `D_ttft=44=S_min(U)`라 `Δ_SLO≥0`이 데이터와
무관하게 강제됨[격자 축소만으로 `P(부호>0)` HI 0.632→0.875] · sticky는
동거 시간이 아니라 단독-at-D를 잼 · 사전등록 estimand가 검열됨).
★**E-B1 NO-GO**(판정 가능 창이 대수로 공집합 — `max_running_requests=48`
캡 때문에 고원 arm의 포화 시 ITL 실패율 `q`가 완전포화에서도 5%
미만). ★★**정본 인용금지를 메인 세션이 위반**(33.04, 등재 다음 날) [CS-OK]
→ 기계적 차단 도구 `scripts/discipline/check_citation_stops.py` 신설
(추가 줄만 검사, 한계 병기). ★★★**`results/slo_sched/` 디렉터리를
G16 캠페인으로 오인**해 "G16만 밴드질량 이상치"라는 사실 오류를
만들었다가 전수 집계로 정정(디렉터리는 12캠페인 447파일 혼합).
★★★**정본이 이미 `ILL-POSED`로 판정한 측정점(`g2_0_hard/
hardened_disjoint_verdict_2026-07-25.md`의 rA5)을 그 판정서를 안 읽고
재분석**해 "d34 골짜기"를 발견처럼 보고했다가 자체 재현에서 부호
반전으로 반증. 상세 `handoff-report/session_handoff_2026-08-18.md`,
아래 "다음 실험 gate" #17(신설)·"방법론 게이트" #47–49(신설).
이전: 2026-08-17 (doc-steward — **G16 캠페인 완료 반영(4블록,
jobs 884336/884410/884411/884412) — R2 결정량②(§1-32)의 재정식화판
산출, 양 phase `ITL_SATURATED`(정보 있는 음성). claims-auditor 적대
감사 반영본(`G16_RESULTS_2026-08-17.md` rev2) 정본 승격. 방법론
게이트 2건 신설(#45·46). 새 성능 판정 0건 · 캠페인 등급 변경 0건 ·
정책 순위 변경 0건 · GPU 지출 0(이번 세션은 문서 반영만, 캠페인
자체는 이미 완료된 ≈3.9 GPU-hr).**
★**"gate #16을 닫았다"고 쓰지 말 것 — 불변**: 닫히는 것은 결정량②의
**재정식화판**뿐이고 원문 문턱 판본은 **rate 축**에 잔존한다.
★**payoff 구간(ITL SLO≲58.6ms)은 이 캠페인으로도 무판정** —
exact band `[58.533,58.625)`(폭 0.092ms)에서 `Δ_SLO`가 `TAX_POSITIVE`
로 뒤집히나 그 부호 자체는 P(부호>0)=0.632로 미식별. 아래 "다음
실험 gate" #16(갱신)·#11 레지스트리 G16 행(4차 갱신) 참조. 상세
`reports/CONSENSUS.md` §1-33(신설)·§3 항목65·66(신설).
이전: 2026-08-16 (doc-steward, 세션4 — **G16 트랙 진행상황 갱신(스모크 2회
통과·하네스 결함 2건 해소·사전등록 분석기 작성+감사[조건부 GO]+반영·블록 1
제출[PENDING]) + 등급 변화 후보 1건(H18 PLAUSIBLE→CONFIRMED, 하네스 설계 속성이지
성능 판정 아님) + 정본 자신의 인용 결함 1건 정정(ncu/CUPTI) + 설계 감사 2건 NO-GO
등재(gate #13·kernel_mech rev2) + 방법론 게이트 3건 신설(#42–44). 새 성능 판정
0건 · 캠페인 등급 변경 0건 · GPU 0.272 GPU-hr(스모크 2회, 블록 1은 별도 제출 중이며
결과 미회수) · 커밋 8건(로컬 `main`, push 0건).**
(1) **G16 진행** — 아래 "다음 실험 gate" #11 레지스트리 G16 행 참조. 스모크 2회
(884292 구 하네스 sha `ce68de98…` → 하네스 결함 2건 발견 → `37cf6b8` 수정 →
884320 신 하네스 sha `e75a6f37…` → `G16_SMOKE_OVERALL=PASS`), 사전등록 분석기
`g16_analyze.py` 신규 작성(rev3 §4·§6·§7·addendum A-2/A-3/B-5 축자 구현) →
claims-auditor 감사(조건부 GO, 기준1·2 PLAUSIBLE(조건부)/기준3 CONFIRMED) →
반영(F1–F4/S1/S6, 커밋 `8311d6c`), **블록 1(job `884336`) 제출 — 세션 종료 시
PENDING**(클러스터 혼잡). ★기존 배너 불변: **"gate #16을 닫았다"고 쓰지 말 것**
(닫는 것은 R2 결정량②의 재정식화판, 원문 문턱 판본은 rate 축에 잔존).
(2) ★**H18 PLAUSIBLE→CONFIRMED**(`PREREG_G16_RULES_REV3_2026-08-16.md` addendum
B-3의 승격 조건 = 스모크 `boot_s` 대조 — 충족): arm별 `boot_s`가 884292
32.7/32.7/32.7초 · 884320 32.8/32.7/32.8초로 **위치-0 arm이 cold-cache 페널티를
전혀 안 문다**. 이것은 하네스 warm-up 설계가 의도대로 동작함을 확인한 **설계
속성**이며 정책·성능 판정이 아니다.
(3) ★★**정본 자신의 인용 결함 정정**(kernel_mech rev2 감사 F3에서 발견) — 상세는
아래 "8B decode-SM 민감도 측정 노트" 스코프 주석 및 `reports/CONSENSUS.md` §3
항목52(4) 追記(5) 참조. `_ncu_target.py:68-71`의 "CUPTI(ncu가 쓰는 API)는 CUDA
Green Contexts와 비호환"을 그동안 인용하지 않고 다섯 줄 아래(73-74)의 "ncu
profiling always runs at full GPU"만 인용해왔다 — 참이면 kernel_mech Stage B는
이 기판에서 구성상 불가하나, ★**아직 미확인 리스크**로만 등재한다(같은
`error code 9`에 저장소가 3가지 경합 귀속을 갖고 있음, 아래 참조).
(4) **설계 감사 2건 NO-GO 등재**(GPU 0) — **gate #13** job/node축 설계
(`DESIGN_G13_JOB_BATCH_2026-08-16.md`): 정본이 지정한 1차 결정량 "between-job
SD of r은 부분 항등식"이 REFUTED(비-축 교차-job 데이터에서 두 다리 job 편차가
사실상 독립), UB95 규칙이 `MS_B≤MS_W`면 자동 PASS라 부팅잡음이 job 신호를
삼킨 런도 "통제됨"으로 선언(Ha8 P(pass)=0.394, 그중 93%가 `σ̂=0` 퇴화). **kernel_mech
rev2**(`DESIGN_KERNEL_MECH_REV2_2026-08-16.md`): `wave_eff`가 ncu 메트릭이
아닌 수제 유도를 1차 결정량으로 되살림(게이트 #36 死因 재생) · §3.2 축퇴 대수
부호 반대로 게이트가 위험구간을 통과시킴 · `ncu --pid` 존재하지 않는 옵션.
★두 건 다 **"닫혔다"고 쓰지 말 것**(gate #13도 gate #16과 동형 — healthy
체제에 필요한 대조 batch가 원 아티팩트에 없어 batch⊗regime 앨리어스).
(5) **방법론 게이트 #42–44 신설**(아래 "방법론 게이트" 참조): #42 게이트가
자기 실패를 성공으로 라벨링(교훈 항목21의 거울상) · #43 분석기가 사전등록
기호를 조용히 재정의하면 자기 대조를 깨고 중심 산출물을 침묵시킴(+메인
세션 오진 1건 기록) · #44 양성대조의 빈 서명 구멍.
상세 `handoff-report/session_handoff_2026-08-16.md` §4-0–4-7.
이전: 2026-08-16 (doc-steward — **C2-R 캠페인 결과 등재: M8·Ha8의
신규 점추정 2건 등재, 기존 값 교체 아님 · C2 등급 무변경(CONFIRMED
scoped) · 인용정지 (a)(arm별 ε·순위)·(b)(깨끗한 셀 CI·"n=4") 둘 다 유효
(해제 0건) · 기존 Δ·p값·크기 인용 셀(Zamba2 r2 단일)은 한 글자도 안
바뀐다.** 사전등록 `PREREG_C2R_RULES_REV2_2026-08-15.md`(rev2 GO)
집행(jobs 883574=M8·883575=Ha8, 각 12부팅) — result-analyst 분석 +
claims-auditor 적대 감사 완료, 메인 세션은 운영 지표(부팅 성공·H7·
실현률)만 직접 확인. **정본 인용 문구(claims-auditor 지정)**:
"`r_M8(16) = 3.058`, `r_Ha8(16) = 3.114`(ctx1024, realized SM16/SM92,
`decode_bs=16`, job 883574/883575, gpu43, 1시간, `n_indep=6` 부팅).
동반 구간은 **within-job 부팅 구간이며 재현 불확실성이 아니다** —
보수적으로 **t(5) [3.056, 3.060] · [3.077, 3.155]**를 쓰고,
**job/node/날짜 축은 미측정(n=1)**임을 병기한다." ★percentile
부트스트랩 CI([3.0566,3.0595]/[3.0848,3.1354])는 **정본 본문에 쓰지
않는다**(일관되게 과소피복 — M8 1.36×·Ha8 1.55× 더 좁음, 원자료
JSON 포인터로만). seed∈{1,2,3,99} SD 변동 ≤1.5%(부트스트랩 seed는
결과를 만들지 않음)이나 **재표집 잡음**만 배제할 뿐 **재표집되는
모집단**(boot 단위, job/node/day 축 부재)의 문제는 그대로다. **핵심
5건**: (1) 위 인용 문구. (2) Ha8은 실현률 0.80에 12/12 셀 미달(arm의
성질로 보임, 감사 N5 미해결). (3) ★★★양성대조가 **항등식**이었다 —
방법론 게이트 **#9 아홉 번째 재발**: 대조 코드 경로(`legacy`+
`keep_slack=False`)는 헤드라인 경로(`c2r`+`keep_slack=True`)와
다르고, 표적값 자체가 같은 루프의 또 다른 복사본
(`audit_c2_job_composition.py`) 산출물 — 공백을 메운 것은
claims-auditor의 독립 재구현이나, ★그 재구현 코드(`indep.py`·
`legacy.py`)는 **스크래치에만 있고 저장소에 없다**(재현 경로 미보존,
2026-08-14 E-1a errata와 동형). (4) ★엔진 빌드 정정 —
865493 대비 매니페스트 11→15 파일, hot path 3종 추가(`scheduler.py`·
`holb_probe.py`·`zamba2.py`) ⇒ "차이는 플래그 2개뿐"은 거짓. (5) guard
고원 — 채택 구간이 전부 0.7–6.3s 미관측 창 안이라 (16,16)은 관측이
아니라 보간. **등재 금지(claims-auditor 지정)**: "C2-R이 정본
2.36–2.91×보다 위"(범주 오류, batch/job 분해 불가)·인용정지 (b)
해제(범주 오류, T8·Hs8 미재측정)·N1/N2 해결(교차-job 대조 0건, 미해결)·
C2 등급 변경. "다음 실험 gate" #11 갱신(실행 완료로 표기) + #13 신설
(job/node 축·batch vs job 분해 — 이 둘 전엔 "3.06 vs 2.91" 판정 금지).
상세 `reports/CONSENSUS.md` rev32·§3 항목56(신설)·18(追記, 아홉 번째
재발), "8B decode-SM 민감도 측정 노트"(2026-08-16 addendum), 메모리
`scale-8b-sm-sensitivity.md`·`deconfound-measurement-lessons.md`
항목39, `workspace/engine-port/results/s8_scaleup/
C2R_RESULTS_2026-08-16.md`.
이전: 2026-08-16 (doc-steward — **새 성능 판정 0건 · 정책 주장 0건 ·
기존 Δ·p값·크기 인용 셀(Zamba2 r2 단일)·등급 무변경 — 등재 3건.**
(1) C2-R 사전등록(rev1 NO-GO 5표적, rev2 ★GO — 이 세션 유일한 GO)을
"다음 실험 gate" #11 레지스트리에 추가(미제출, 스모크 job 883351 대기).
(2) 방법론 게이트 **#37 신설** — "적대 감사에는 합격 기준과 단일 판정
질문을 함께 줘라, 범위 없는 적대 검토는 항상 NO-GO를 산출하며 그건 설계
품질 신호가 아니다"(C2-R rev1→rev2로 5연속 감사 차단이 끝난 데서 도출).
⚠️**부수 진단 필수 병기**: rev1의 진짜 문제는 감사 과잉이 아니라 메인
세션의 설계 과잉(4 arm인데 실제 결손은 2 arm)이었다. ⚠️**provenance**:
메인 세션 직접 진단, **claims-auditor 감사 없음** — "현재 작업가설"로
인용. (3) 사소한 표기 정정 — job 865533 붕괴 keepalive 프롬프트 토큰
수를 "1793"에서 실측값 **1794**로 정정(M8·Ha8 토크나이저 동일, `s8_c2r.
sbatch` H2 구현 중 실측) — **1792 초과·HTTP 400 전량 거부라는 결론·
기전은 완전히 불변**. 상세 `reports/CONSENSUS.md` rev31·§3 항목55(신설)·
50 addendum3, 메모리 `deconfound-measurement-lessons.md` 항목38,
`handoff-report/session_handoff_2026-08-16.md` §4-2.
이전: 2026-08-15 (doc-steward — **새 성능 판정 0건 · 정책 주장
0건 — result-analyst의 C2 헤드라인 job 구성 감사를 claims-auditor가
적대 검증 완료(반증 3건 포함). 등급 무변경(CONFIRMED scoped),
인용 정지 2건 신설.** (A) 같은 날 앞선 addendum(G-1) 자신의 서술
오류 정정 — job 865533의 keepalive 붕괴를 "Ha8 arm 한정"으로
적었으나 원자료는 **4 arm(Ha8·Hs8·M8·T8) 전 20셀 전부**임(G-1
追記2, claims-auditor 검증 대상 아님, 그대로 유지). (B) C2
헤드라인(2.36–2.91×, 4 arm) job provenance 감사 — **claims-auditor
적대 검증 완료**. 생존(반증 실패, 인용 가능): 헤드라인 4셀 중
3셀(Ha8·M8·Hs8)은 양 다리 100% job 865533 단독, T8도 SM92 다리는
865533 단독(원인=`t0_monotonic_s` 하네스 결손, keepalive 붕괴
아님); 58 매칭 셀의 조건부 ITL 차 −0.18%±0.62%(최대 2.20%)는
3중 추가 반증 시도에서도 생존. **반증됨**: "깨끗한 런 단독 재계산
T8 2.388×·Hs8 2.687×를 CI·n=4와 함께 인용 가능" — T8 b16의
865493/d16 rep1–4가 실은 **서버 부팅 1회**(다리당 `n_indep=1`, rep는
의사반복)라 CI는 실제로 ≈2배 넓다; "Ha8은 b12가 양 다리 0건이라
FINDINGS 지침을 구조적으로 만족 불가" — 귀속 오류, SM92 다리는
오히려 깨끗한 런(865493)이 b16까지 도달(33,344건). **신규 발견**:
arm별 ε 순서는 슬라이스 산물 — 4 arm 공통 batch(b=1)에서는 순서가
완전히 뒤집힌다(폭 0.118→0.038). **판정 = 등급 CONFIRMED(scoped)
유지 + 인용 정지 2건 신설**(arm별 ε·순위 / 깨끗한 셀 CI·"n=4"
표기, "8B decode-SM 민감도 측정 노트" 재정정). (C) "다음 실험
gate" #12의 "수치는 ctx4096 한정, 기전은 ctx-무관" 서술은
**claims-auditor 독립 재집계로 반증**됨 — ctx1024(SM92 max bs
865493 21/23/21/23·865533 16)가 그 기전의 반례다. **정정된 명제**:
B-폐쇄는 **λ(L)이 작을 때만 구속**(regime-의존, ctx-무관 아님,
재정정). 상세 위 "8B decode-SM 민감도 측정 노트"(재정정)·"다음
실험 gate" #12(재정정)·G-1(追記2), `reports/CONSENSUS.md` rev30·§3
항목50 追記2·51 追記(재정정)·54(신설, 재정정), `workspace/
engine-port/results/s8_scaleup/
AUDIT_C2_HEADLINE_JOB_COMPOSITION_2026-08-15.md`(1차 산출,
result-analyst — 반증 3항목 포함 원문 보존), claims-auditor 적대
검증(원자료 파일 위치 미확정, 다음 세션 편입 요망). ⚠️provenance:
B/C는 result-analyst(1차)+claims-auditor(적대 검증) 산출, 메인
세션 독립 재확인 없음.
⚠️**이 상단 changelog의 공백 기록**: 2026-08-14(2차) 이후 같은 날
(2026-08-15, commit `2e2ddc1`)에 이미 G-1 addendum·"다음 실험 gate"
#12(신설)·ncu/nsys 스코프 주석·방법론 게이트 #35·#36이 본문에
반영됐으나 이 상단 헤더에는 별도 항목으로 기록되지 않았다(내용은
`reports/CONSENSUS.md` rev29 changelog 참조) — 누락을 여기 짚어
두고, 이번 갱신부터 다시 매 반영을 상단에 기록한다.
이전: 2026-08-14 (doc-steward, 2차 — **정본 정정 1건 + 신규 결과
등재 1건. 새 성능 판정 0건. 기존 결론은 뒤집히지 않는다(오히려 강화
방향).** **(A) 하드웨어 오식별 정정**: `TRAFFIC_ROOFLINE_DIAGNOSTIC_
2026-08-11.md`가 `nvidia-smi -q`를 **로그인 노드(glogin01, `A100 80GB
PCIe`)**에서 읽었으나 실제 캠페인(jobs 865289/865533)은 **컴퓨트
노드**(gpu36/38/40, `sacct` 확인)에서 돌았고 컴퓨트 노드는 **SXM4**다
(job 882374, gpu43, `torch.cuda.get_device_name()`=`NVIDIA
A100-SXM4-80GB` 직접 관측; `s8_scaleup/` job 아티팩트 전체에 `"A100
80GB PCIe"` 문자열 0건 — 하드웨어 식별이 측정이 일어나지 않은 기계에서
읽혔다). 정정: 사양 BW 1935→2039 GB/s, achieved_BW 비율
48–61%→45.4–57.7%, ridge 161→153 FLOP/byte, decode AI 5.1%→5.4%.
**어떤 판정도 안 뒤집힘 — 오히려 강화**(비율이 더 낮아져 "고-SM
평탄화≠HBM 포화" 근거 강해짐). 방법론 게이트 #32에 새 사례 追記
(하드웨어 식별 층 재발 — #32를 만든 문서 자신이 같은 종류 오류를 두
번 냄). **(B) 신규 결과 등재**: E-3 realized SM count 프로브(사전등록
`workspace/engine-port/results/smsplit_realized/
PREREG_SMSPLIT_REALIZED_2026-08-14.md`, GPU 비용 ≈0) — `(74,34)`·
`(54,54)`·C2 스윕 5지점 전 7지점에서 green-context 원시함수의 realized
반환값이 요청값과 **정확히 일치**(Δ=0), glogin01·컴퓨트 노드(job
882374) 두 하드웨어에서 레코드 완전 일치, 판정
`REQUEST_EQUALS_DRIVER_REPORTED_PARTITION`. **드라이버 자기보고**이지
하드웨어 실행 층이 아니므로 §1-1 Gate 1 블록의 기존 인용 금지("실현
파티션을 측정했다" 등)는 **그대로 유지**, `%smid`의 SM id 집합
disjointness 질문은 전진 0, 성능·정책 주장 0건. 부수 함의: `%smid` R0
§0.1의 `log(108/34)` 분모 정정 payoff는 정정 대상 없음이 확인(34는
요청값=드라이버 보고값). 상세 아래 "확정된 결과" 1번(E-3 블록)·"8B
decode-SM 민감도 측정 노트"·"방법론 게이트" #32(追記), `CONSENSUS.md`
rev26·§1-1(E-3 블록)·§3 항목46(追記), `TRAFFIC_ROOFLINE_DIAGNOSTIC_
2026-08-11.md` §11·`FINDINGS_8B_2026-07-28.md` §7 C-5·`workspace/
engine-port/results/smsplit_realized/PREREG_SMSPLIT_REALIZED_
2026-08-14.md`(addendum 2)·`workspace/engine-port/RESUME.md`.
이전: 2026-08-14 (doc-steward — **문서 층 정리, 새 성능 판정
0건.** Δ·p값·크기 인용 셀·등급은 한 글자도 바뀌지 않는다. (A) 방법론
게이트 #26에 追記 — 발동 조건("캠페인" 단수)이 <1 GPU-hr 조각으로
쪼개면 회피됨을 2026-08-11 세션의 GPU 실험 4건 검토(P1/E1-b·c/
`%smid` P1+P2/G1-a, 합 ≈1.15–1.65 GPU-hr, 개별로는 전부 문턱 아래)에서
확인, 트리거를 "한 배치로 제출되는 신규/변경 코드 공유 캠페인들의
합"으로 개정 권고. (B) 신규 방법론 게이트 #33 — `sync_engine_tree.sh`가
sha256 해시하는 파일은 정확히 **15개**뿐이고, `pdmux_context.py`
(`(74,34)` 등 파티션 기대값의 출처)·`sgl_kernel/spatial.py`(green-ctx
원시함수)는 매니페스트 밖, `src/patches/`의 패치 5개 중 2개는 코드
전체에서 미적용 확인 ⇒ "매니페스트 N/N sha 일치" 재현성 주장의 범위를
그 15파일로 명시적으로 좁힌다(기존 판정 뒤집기 아님, 커버 밖 드리프트
증거도 없음). (C) 방법론 게이트 #9·#25 본문에 **섹션 헤더가 이미
명시했던 追記 2건이 누락**돼 있던 정본 결함(핸드오프가 1건이라 했으나
실제 2건)을 `CONSENSUS.md` §3 항목18·39 원문 대조로 복원. (D)
`workspace/engine-port/RESUME.md`의 CPU 회귀 지시(루트가 아니라
`workspace/engine-port/RESUME.md`임)에 노드 구분·소요 시간(로그인
노드 140 tests OK/73초, 대부분이 scheduler import 70.6초 — "컴퓨트
노드 필수"는 재현 안 됨, 7분 stall은 Lustre 콜드캐시 추정) +
`g2s_run.sbatch:91`이 이미 컴퓨트 노드에서 전체 회귀를 차단 게이트로
돈다는 사실을 반영. (E) 2026-08-14 커밋 `31b3e96`/`86179ec`(KISTI
`--comment` 정책 전 저장소 적용 + conformance checker)를 이 문서에
신규 "운영 규약" 절로 등재(`RESUME.md`에는 이미 있었음). 상세 아래
"운영 규약"·"방법론 게이트" #9·#25·#26·#33, `CONSENSUS.md` rev25·§3
항목47·48.
이전: 2026-08-11 (doc-steward — **트래픽·roofline 진단 반영
(`workspace/engine-port/results/s8_scaleup/
TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md`, result-analyst, GPU 0 —
기존 아티팩트+모델 config 계산만). 새 성능 판정 0건, C2("decode
SM 민감도 2.36–2.91×, scoped")의 등급·수치는 불변 — 기존 주장이
더 약화되고 스코프가 더 좁아지는 방향.** 계기 = 사용자의 반박 4건
중 2건이 계산으로 확인되고 메인 세션의 "교차점(L\*) 가설"은
반증됨. **C-1**: §2.1의 "1차 weight-traffic 추정(M 5.40/T 6.17/
H 7.66 GB, H/M=1.42)"이 실은 **3B급 다른 캠페인의 수치**이고
기준(체크포인트 바이트 vs 호출-인지 트래픽)도 혼합돼 있었음을
체크포인트 실측 3자리 일치로 확정 — 이미 NOT-YET-SUPPORTED인
C2b(hybrid 급락=Zamba2 성질)를 **더 약화**(되살리지 않음). **C-2**:
"decode SM 민감도는 모델-무관"의 원인이 아키텍처 동질성이 아니라
이 측정점(B≈9–12·L≈1.0–1.5k)에서 스텝 트래픽의 68–94%가
weight-sweep이기 때문임을 규명 — B/L 확장 이식 금지. ★**"hybrid는
KV가 적다"는 통념은 이 격자에서 거짓**(per-seq 캐시 Ha8 601 MiB
> M8 260 > Hs8 117 > T8 70 MiB, Ha8이 T8의 8.6×). **C-4(신규
금지)**: SM92에서도 achieved_BW가 사양 대역폭(A100 80GB **PCIe**
1935 GB/s — SXM 2039 아님)의 48–61%뿐이라 **고-SM 평탄화를 HBM
포화로 서술하는 것 금지**(진짜 원인 미식별); decode 축(AI≈8.2,
memory-bound)의 이 결론을 prefill 축(AI≈1035, compute-bound)으로
이식 금지. 신규 방법론 항목 2건(#9 일곱 번째 재발[roofline 탄력도
정합=항등식, 산출자 자수]·#32 신설[타 캠페인 보조 수치는 기준
검증 후 수입]) 등재. 상세 아래 "8B decode-SM 민감도 측정 노트"·
"방법론 게이트" #9·#32, `CONSENSUS.md` §3 항목18(追記)·46,
`workspace/engine-port/results/s8_scaleup/
FINDINGS_8B_2026-07-28.md` §7.
이전: 2026-08-11 (doc-steward — **G1-c(job 877974, 0.10
GPU-hr) 반영 — Gate 2-S Granite r3·r4 "엔진 기본 궤적"·"A4형"
명명 제한 조건부 해제. 새 성능 판정 아님, "크기 인용 셀이
늘었다"도 아님(반드시 정정).** 세션 초반의 동기 서술("전제
VERIFIED가 되면 크기 인용 가능 셀이 1→3개로 는다")은 원자료·
코드 대조로 **반증**됐다 — 크기 인용을 막는 것은 `premise`
라벨이 아니라 독립 산출되는 F-계열 gate이고(`g2s_analyze.py:
1157-1161`, 유일 소비처는 `name_for()`), Granite r3(T′)·r4(C·
T′) 모두 여전히 F-계열 발화 상태라 **`SIGN ONLY, MAGNITUDE NOT
CITABLE`이 유지**된다. ⇒ **이번 승격이 바꾸는 것은 명명 층
하나뿐**이고, **크기 인용 가능 셀은 여전히 Zamba2 r2 하나**다
(불변). rev20(2026-08-11 Gate 2-S 첫 유효 결과)의 Δ=+13.95/
+24.88/+16.04/+19.11ms(4셀 Holm 후 최대 p=1.88e-06)·전 9-셀
`S1-C` 문장은 **한 글자도 바뀌지 않는다**. 동시에
`PREREG_GATE2S_2026-08-09.md` §8.9의 Zamba2 근거표가 Gate 1이
인용 금지한 pop-C 시간가중 분수를 인용하고 있던 기존 정본 결함을
사후 addendum으로 정정했고(원문 미덮어쓰기), 신규 방법론 항목
3건(#27–29)을 등재했다. 상세는 아래 "확정된 결과" 1번·"방법론
게이트" #27–29, `CONSENSUS.md` §1-1(2026-08-11 G1-c 블록, rev22)·
§3 항목41–43.
이전: 2026-08-11 (doc-steward — **"방법론 게이트" #26 신설**
[대형 캠페인 제출 전 배관 스모크 규율, `CONSENSUS.md` §3 항목40과
대응] — 새 성능 판정 아님. Gate 2-S 1차 실행(jobs 877107/877109,
6.40 GPU-hr)이 하네스 결함 6건으로 사전등록 primary 0개를 낸 뒤,
재실행 전 돌린 배관 스모크(job 877593, 0.11 GPU-hr)가 같은 재발을
막았다(≈60배 비용 절감) — 전문은 아래 "방법론 게이트" #26 참조.
**동시에 `reports/paper/`(`CLAIM_EVIDENCE_MATRIX.md`·
`EXPERIMENT_ROADMAP.md`·`DOCUMENT_STATUS.md`)를 이 문서·
`CONSENSUS.md` rev14–rev20(2026-08-06~2026-08-11)과 동기화했다**
— Claim A–F 행 자체는 무변경(이 구간이 그 행들에 인용되지 않음,
`CONSENSUS.md` rev14–20 changelog가 반복 확인). 대신 그 세 문서에
없었던 **"P1 트랙"**(PD-mux 자체 vs fused, Gate 1/Gate 2/E-A/
Gate 2-S) 절을 신설해 아래 "확정된 결과" 1번의 현재 판정과 Gate
2-S 인용 제한 7건을 이관했다(문서 자체 changelog에 상세). 이
동기화는 새 결론이 아니라 상위 정본(reports/paper/)이 이 문서를
누락 없이 반영하도록 맞춘 것이다.
이전: 2026-08-11 (doc-steward — ★★★★★★★**Gate 2-S 첫 유효
결과(jobs 877756/877757, 6.35 GPU-hr) — claims-auditor 적대 감사
"조건부 등재 가(可)". §1-1의 기존 NOT-YET-SUPPORTED를 대체하지
않는 별도 트랙(Gate 2-S)의 산출이며, 2026-08-10에 넣은 "귀속
스코프" 주석과 정합하게 배치한다.** 둘 다 exit 0/
`MEASURED_AND_SCORED`, `design_conformance.conformant=True`(전 셀
n=10, 양 rate, 5 arm), 19/19 블록. **이력(반드시 병기)**: 1차
실행(jobs 877107/877109, 6.40 GPU-hr)은 하네스 결함 6건으로
사전등록 primary를 **0개** 냈다 — 사전등록 설계는 무결했고 실패는
전부 하네스 층이었다. `PREREG_GATE2S_2026-08-09.md`(rev6)는
claims-auditor 설계 감사 **4회**(NO-GO 3 → GO-with-changes)를
거친 것. 채점 result-analyst(원자료 200행 독립 재계산), 감사
claims-auditor(원자료에서 α 독립 재현, 소수 3자리 일치 — 메인
세션이 g2s_report_*.json 원자료로 핵심 수치(Δ 4값·Holm p·Fieller
3.098·premise verified/unverified·frac_idx1) 추가 독립 확인, 전용
감사 아티팩트 파일 위치는 미확인).

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
12지표 중 9개 비트 동일(항등식) — **방법론 게이트#9**("게이트
자신이 항등식") **새 사례**. (ii) ★**방법론 게이트#20 진짜 재발**:
`g2s_analyze.py:63`의 `MIN_COVERAGE=0.98`이 `gate1b_analyze.py:49`
에서 **상수만 수입되고 규칙은 미구현**(원본 `:250`은 실제 사용) —
**"식별자 수입 ≠ 거동 수입"**, 게이트#20 새 사례(방향은 저자
불리, 4셀 전부 253–427 에피소드라 수치 결과는 없으나 provenance
기록 필수). (iii) 신규(방법론 게이트#24): `any()` over n reps
형태의 스크린은 귀무 발화율이 **1−(1−α)ⁿ**이며 인용 자격을
사실상 무작위 배정한다(여기선 40.1%). (iv) 신규(방법론 게이트#25):
§6.3-6의 관측자 대칭 보고 항목(`dropped_events`/`writer_error`)이
**산출되지 않았다** — 감사자가 오프라인 복구해 전부 0 확인, 간극은
실재. 부수(문구 구속용, 정본 수치 인용 금지): pdmux 가족 내부
관측자 부하 비대칭(C가 T′보다 `runtime_snapshot` 2.6–3.8% 많고
방향이 효과에 유리, G5 UNDETERMINED라 상한 없음).

⚠️**등재 불가 항목**: e2e/concurrency 수치, Granite 전제의 "검증됨"
승격, dose 보정치, goodput 해석. 상세 `reports/CONSENSUS.md`
rev20·§1-1(2026-08-11 Gate 2-S 결과 블록)·§3 항목18·34(개정)·38·39,
"확정된 결과" 1번·"방법론 게이트" #9·#20(개정)·#24·#25 동반 갱신.
원자료 `workspace/engine-port/results/p1_gates/gate2/
g2s_report_zamba2-27b_877756.json`·`g2s_report_granite-40-h-micro-base_
877757.json`(수정 금지·인용만). `CLAIM_EVIDENCE_MATRIX.md`는 대조
확인 결과 이 항목을 인용한 서술이 없어 갱신 대상 없음(확인 완료).
이전: 2026-08-10 (doc-steward — 같은 캠페인 계열의 세 번째 정본
반영 건. **A(재발 카운트 갱신)·B(도구 규율)·C(설계 감사 스코프
결론)·D(코드 사실 문구 하향) 네 갈래 — 전부 새 성능 판정 아님.**

**A. 방법론 게이트 #21의 일곱 번째 재발 — 이번엔 코드가 거짓 음성을
냈다**(engine-porter 발견, 메인 세션 사건 경위 확인): job 876699
(T4-1)가 `--time=1:00:00` TIMEOUT — 원인은 사이징이 아니라 단일
호출 스톨(9 부팅 중 7개 ~8분 정상 완료, `ON+chunk512`의 첫 실제
multi-chunk generate(2552 토큰, 5-chunk)가 ~52분 무응답, 스케줄러
로그 0줄·CUDA 에러·watchdog 미발화; 서버 로그 마지막 활동 23:34:42
vs SLURM kill 00:26:19, 메인 세션 직접 대조; 같은 job `OFF+
chunk512`는 동일 프롬프트 ~1초 완료). **핵심**: 옛 `g2det_
analyze.py`가 그 아티팩트에 `REFUTED — reduction-order is not the
(sole) cause`를 반환 중이었다(ON chunk512 `n_total=0`인데
`on_clean = n_total > 0 and ...`이 False로 떨어져 REFUTED 분기로
통과) — **52분짜리 멈춤이 실질적 음성 결과로 발표될 뻔했다.**
재발 1–6은 라벨·해석 오류를 사람이 잡았으나, 이번(7)은 **분석
코드가 거짓 음성을 산출**했고 막은 것은 experiment-runner의
채점 거부(`INCOMPLETE`)였다 — 도구가 사람보다 관대했다. 사후
완화 아님(사전등록 REFUTED 조건은 "ON이 어느 rep에서든
self-mismatch≥1", rep 0개는 그 관측 자체가 없어 옛 코드는
버그였다). 수정(2026-08-10) 후 `NO VERDICT (MEASUREMENT ABSENT)`
반환, CONFIRMED/REFUTED 분기(5-케이스 매트릭스)는 불변. ⇒
**`CONSENSUS.md` §3 항목35(방법론 게이트 #21)의 재발 카운트 6→7
갱신**(신규 항목 아님, 항목 자체에 追記).

**B. 공유 하네스의 무한 대기**(도구 규율, 신규 §3 항목37/게이트
#23): `g2_holb_phaseA_lib.sh`의 `greedy_call`이 `--max-time` 없는
raw curl이었고 이 디렉터리의 모든 캠페인(g2ctrl/g2ea/g2holb/g2det)
이 이 함수를 공유한다 — 소비자 10개 전수 확인 결과 4개가
타임아웃을 `FAIL`/`SMOKE_FAIL` 계열로 채점 중이었다(A의
`g2det_analyze.py` 포함). 수정: `--connect-timeout 10 --max-time
180`(env 재정의 가능), `STATUS=TIMEOUT`을 `STATUS=ERROR`와 구별,
SHA 미방출로 mismatch 채점 구조적 불가, 사이드카 `.status.json`.
180s는 측정 근거(아카이브 응답 n=64의 서버측 `e2e_latency` median
0.978s/p90 2.534s/**max 6.605s**, 최악의 27배·중앙값의 184배 —
메인 세션이 코드 주석 근거 문단 직접 대조) — 정상 경로는 아카이브
64개+합성 실패 9종 재생 73/73 byte-identical 검증(engine-porter
보고, 메인 세션 미재현). A와 뿌리 사건은 같으나(job 876699) 레슨은
다르다.

**C. Gate 2-S 3라운드 설계 감사 — §1-1 귀속 스코프 주석(대체
아님)**: `PREREG_GATE2S_2026-08-09.md`가 claims-auditor 감사 3회
전부 NO-GO(rev1→2 통계층/rev2→3 게이트 인식론/rev3→4 귀무 채택형,
메인 세션 파일 직접 확인) — 3연속이 같은 자리(§5.5 앵커 발화
조건)에서 죽었고 3차 감사가 근본 원인을 문서 내부 모순으로 특정
(§0.1 "두 pdmux arm 사이 등가 마진에는 외부 앵커가 없다" vs §5.5가
정확히 그 등가를 앵커 조건으로 요구). 정량: rev3 §5.3 암묵 등가
마진 = **0.715 σ_D**, primary MDE = **0.995 σ_D** ⇒ 대리
허용오차가 검출한계의 **0.72배**; 표준 처방(TOST+마진)은 §0.1
위반, 마진을 MDE 1/4로 낮추려면 **n≈64**(현재 10). ⇒ **정본
문구(고정, §1-1에 스코프 주석으로 덧붙임 — 기존 NOT-YET-SUPPORTED
대체 아님)**: "P1 이득 중 'SM 분할 자체의 몫'을 두 pdmux arm 사이의
성능-층 등가검정으로 귀속하는 경로는, 이 프로젝트의 예산 범위에서
닫히지 않는다(n≈64 필요, 현행 설계 n=10). 원리적 불가능이 아니라
이 경로·이 예산에서의 불가능이다." ⚠️과장 금지("귀속이 원리적으로
불가능하다"로 쓰지 않는다) — 미실행 사전등록의 설계 감사이지 측정
결과가 아니다. **감사가 깨뜨리려다 실패한 것(견고)**: arm
구조·Δ_split estimand 내부 타당성, T·C 드레인 대칭, 9-셀 판정표의
저자 불리 셀 실재, 부팅 단위 프로브 근거, 예산 산술. 3차 감사
원문: "이 캠페인이 죽어야 할 이유는 측정 층이 아니라 귀속 층에만
있다." rev4는 §5.5 앵커 조건만 제거(패치 0줄) — **Gate 2-S 폐기
아님.**

**D. 코드 사실 문구 하향**(메인 세션 자기정정): 이 세션 중 "`(0,
108)` idx에서는 두 역할 모두 전체 108 SM에 접근한다"고 서술한 바
있다. 검증된 것은 "green context가 아니라 평범한 `torch.cuda.
Stream` 쌍"까지뿐(`pdmux_context.py:124-138`, 메인 세션 직접
확인) — green ctx 생성이 primary context SM을 깎는지는 미측정
물리 명제(격리 수단 P-b 프로브는 공선성으로 삭제됨). 전수 검색
(2026-08-10) 결과 이 문구는 canon·파생 문서 어디에도 없어 정정
대상 없음 — 향후 인용 규칙만 등재: (0,108) idx는 "명시적 분할이
적용되지 않는다(잔여 차감 여부는 미측정)"로만 쓴다.

상세 `reports/CONSENSUS.md` rev19·§1-1(2026-08-10 세 번째 정본
반영 건 A/B/C/D 블록)·§3 항목35(개정)·37, "확정된 결과" 1번·
"방법론 게이트" #21(개정)·#23 동반 갱신. `CLAIM_EVIDENCE_MATRIX.md`
는 대조 확인 결과 인용 서술 없어 갱신 대상 없음(확인 완료).
이전: 2026-08-09 (doc-steward — 같은 캠페인 계열의 두 번째 정본
반영 건. **A(재채점)·B(사실 정정)·C(도구 규율) 세 갈래 — 전부 새
성능 판정 아님.** **A. HOLB G5 재채점**(result-analyst, jobs
874601/874602/874632/874633/874635, `workspace/engine-port/
results/p1_gates/gate2/g2holb_g5_tost_rescore_2026-08-09.json`
72셀 전량, 3 job × 4 arm × 6 응답변수, n=5 paired): **판정 G5 =
미결정(UNDETERMINED), 저장된 `G5=False`를 대체.** 3% 초과 통계적
지지 셀 **0/72**(미보정 최소 p_exceed=0.138, job별 Holm 후 최소
조정 p=1.000 3 job 전부), 등가 입증 **29/72**, 검정력 부족
**43/72** — G5는 관측자 효과가 3%를 넘음도, 3% 이내임도 입증하지
못했다. 저장 `G5=False`는 귀무-채택형 연언(`|효과|<3% ∧ CI∋0`)의
실패이지 프로브 유해성의 입증이 아니다. **구 규칙이 노이즈를
보상했다**(874635 agnostic ttft_p95 −0.77%±29.14%, 95% CI
[−36.95,+35.41] → 구 규칙 PASS) — 역방향(구 규칙 FAIL·±3% 등가
실제 입증)도 **4셀**(예: 874633 plain itl_p95 −1.15%±0.72%,
p_TOST=0.0023). **설계 층**: n=5·δ=3%·α=0.05/side TOST 발화
산술 천장 SD<3.147%인데 72셀 중 **35셀(49%)**이 그 위, 80%
검정력 필요 n 중앙값 **9**(변수별 4–89, **ttft_p95=89**가 최대).
`PREREG_GATE2` §14.2의 "프로브가 무해함이 입증됐다고 쓰지 마라"는
**해제되지 않는다**(반대 오독도 근거 없음). §14.3의 귀인은
**부분적으로만 참**(초과 실재 조합은 있으나 지지되는 것은 0개).
**잔여 교락**: 874602 agnostic은 5/5 rep `order='off on'`(무작위화
불균형). **B. job 874601의 `G3=False` 라벨 정정**(메인 세션 원자료
직접 확인): 4 arm 전부 `result:"FAIL"`이나 `sha_off/on:null`·
`self_repro:true` — sha 추출 `KeyError:'text'`(하네스 결함)이지
correctness 반증이 아니다. 재실행 874633/874635는 PASS. **측정
실패를 게이트 실패로 라벨링한 이 프로젝트 서명 오류의 6번째
재발**(핸드오프 2026-08-09 §1이 5번 서술, canon 최초 정식 등재).
**C. E-A 산출물의 scipy 부재 폴백**(engine-porter 발견 + 메인 세션
노출범위 실측, 범위 한정): `g2ea_analyze.py`가 scipy 없이 실행돼
`t_cdf()`가 정규 CDF로 폴백(저장 `p_tost=1.0`의 원인). 메인
세션이 `raw_ci` 8개 전부 implied_t=2.2621…=t(.975,df=9)임을
확인 — **CI는 오염 안 됨**, 판정도 불변(관측치가 0.05 경계에서
멀어서). **정본 수치 정정 아님 — 도구 규율 항목**(게이트 #14
계열). 상세 `reports/CONSENSUS.md` rev18·§1-1(A/B/C 블록)·§3
항목35·36, 원자료 `workspace/engine-port/results/p1_gates/gate2/`.
`CLAIM_EVIDENCE_MATRIX.md`는 대조 확인 결과 인용 서술 없어 갱신
대상 없음(확인 완료). 방법론 게이트 #21·#22 신설(아래).
이전: 2026-08-09 (doc-steward — ★★★★★**"확정된 결과" 1번(P1)
Gate 2 rev4 본 캠페인(jobs 875344/875346, 2026-08-07 실행, 5.86
GPU-hr) 정본 반영 복구 — R1′/R2′ 확립.** [`workspace/engine-port/
results/p1_gates/gate2/g2_report_zamba2-27b_875344.json`·
`g2_report_granite-40-h-micro-base_875346.json`·`PREREG_GATE2_
2026-08-06.md`(rev4)]. **새 성능 판정 아님 — 이미 완료된 결과의
정본 반영 누락 복구다.** Primary(A3=`chunk512` vs A4=`agnostic`,
§4.1/5.2)는 5셀(Zamba2 r2·r3, Granite r3·r4·r6) 전부 `Rprime4`(A4
유의 우세, TOST 등가 미발화) — chunk512는 pdmux를 대체하지
못한다(크기 인용 가능 셀은 F-E 스크린 통과분인 Zamba2 r2·
Granite r3뿐). **Secondary — R1′/R2′(A2=`plainaux` vs A4)**:
rev4 §1.1의 등식 A2=A4−pdmux(realized 차이는 `enable_pdmux`·
`pdmux_config_path` 두 필드뿐)이므로 이 비교는 §1-1의 "3-플래그
묶음 처치" 교락 중 pdmux 고유분을 분리한다. 메인 세션이
2026-08-09 원자료(`per_arm_x60`, paired-t n=10)로 독립 재현:
Zamba2 r2 +0.834[+0.781,+0.887]·r3 +0.930[+0.897,+0.963], Granite
r3 +0.855[+0.816,+0.894]·r4 +0.944[+0.920,+0.968](이상 부호 10/10,
F-E clear), r6 +0.960[+0.929,+0.991](부호만 — agnostic 측 F-E
flagged). sign-flip 순열 p = 2/1024 = **0.001953125**(5셀 공통
하한). A2는 A1(`plain`)과 사실상 같고(5셀 |A1−A2|≤0.023) 부호는
**A4에 불리한 핸디캡 방향** ⇒ aux 플래그(`--chunked-prefill-size
-1`·`--disable-overlap-schedule`) 단독으로는 pdmux 이득이
재현되지 않는다 — 이 두 플래그는 §1-1 "3-플래그 묶음 처치"
교락에서 **원인 배제**. ★**provenance(필수 병기)**: 이 비교는
**사전등록 분석기 `g2_analyze.py`가 계산하지 않는다**(primary
`tost_equivalence` 호출은 A3-vs-A4 1회뿐, `:538-543`; A2는 `:734`
서술문뿐) — arm·데이터는 사전등록이나 **비교 자체는 저장된
`per_arm_x60` 위의 사후 계산**(방법론 게이트 #20, CONSENSUS §3
항목34, 항목33의 형제 사례). ★**천장 포화**: fused 0.72–0.99·
agnostic 0.00–0.12로 양쪽 포화 근접 ⇒ 부호는 견고, **크기는 기전
해석 금지**(E-A와 동일 캐비어트). ★**해금 상한(overclaim 금지)**:
해소는 §1-1 "3-플래그 묶음 처치" 교락뿐 — 귀속 상한은 **"pdmux
서브시스템 전체"**(SM분할+전용 이벤트 루프+split-prefill),
**"SM 분할 자체"는 여전히 미분리**(`--enable-pdmux`가 이벤트
루프를 통째로 교체) ⇒ **"PD 분리 자체가 원인" 승격 금지**,
§1-1 NOT-YET-SUPPORTED 불변, Gate 2 본 질문 전진 없음.
★**감사 provenance 정정**: 핸드오프(`session_handoff_2026-08-09.md`
§2.5)의 "[감사 완료]" 표기는 저장소 전체 검색으로 대응
claims-auditor 아티팩트가 확인되지 않아 **"메인 세션 원자료
독립 재현 확인(2026-08-09), 감사 기록 위치 미확인"으로 등급
정정**(감사 완료로 인용 금지). ⚠️**부수 정정**: 아래(이전 항목)
`PREREG_GATE2_2026-08-06.md`가 "워킹트리 미커밋"이라 적은 것은
**stale**이다 — 커밋 `c47fad0`으로 반영돼 현재 워킹트리 수정
추적 파일 0건(2026-08-09 확인, `git status`). §11-3 감사 면제
상태(§14.1)는 별개로 불변. 상세 `reports/CONSENSUS.md`
rev17·§1-1(rev4 본 캠페인 블록, E-A 블록 앞)·§3 항목34, 원자료
`workspace/engine-port/results/p1_gates/gate2/`. `CLAIM_EVIDENCE_
MATRIX.md`는 대조 확인 결과 이 항목을 인용한 서술이 없어 갱신
대상 없음(확인 완료). 방법론 게이트 #20 신설(아래).
이전: 2026-08-09 (doc-steward — ★★★★★★**"확정된 결과" 1번(P1)
E-A(mixed-chunk 레버, jobs 875654/875657/875661) 반영 — claims-auditor
감사 완료, 판정 조건부(문구 강한 제한).** [`workspace/engine-port/
results/p1_gates/gate2/PREREG_G2EA_2026-08-07.md`·`g2ea_report_*.json`·
`g2earun_8756{57,61}.out`·`g2eaprobe_875654*`]. **새 성능 판정 아님 —
fused-측 조율 가능성에 대한 진단이다.** `--enable-mixed-chunk`는
Zamba2-2.7B·Granite-4.0-h-micro-base×{8192,512} 4콤보 전부에서
realized로 확인됐으나(Stage 1, job 875654), E-A(n=10 paired,
cudagraph-ON) 사전등록 primary 두 비교×4셀=8건 전부 TOST 등가
미발화·부호 전부 agnostic 우세(10/10, p=0.00195) ⇒ **"조율된
fused가 pdmux를 대체한다"는 발화하지 않았다.** 유일한 인용 가능
정량치(Granite rate3, cmp1) X_60(plainmix)−X_60(agnostic)=+0.855
[+0.814,+0.896] 중 mixed-chunk 고유 기여분은 **1.2%(+0.010)뿐**,
나머지 98.8%는 rev4/§1-1이 이미 확립한 untuned-fused-vs-pdmux
격차 ⇒ **E-A의 primary는 레버를 격리하지 않는다.** 같은 셀에서
TTFT 항은 오히려 plainmix가 19.2% 좋음("TTFT 비열등 0건"은 거짓).
기전(신규, 서빙 직접 측정): mixed-chunk가 decode를 `ForwardMode.
MIXED` extend 경로로 재라우팅해 MIXED step time이 병합 decode 수에
선형 증가(Zamba2 ≈33ms/req, Granite ≈4.2ms/req) — cudagraph 가설은
반증되고, 정본 §1 얽힘 死因의 새 트리거로 확인. G-2(재현성, 16-동시
greedy에서 비트단위 재현성 상실)는 correctness 결함 아님, 3회 독립
재현. ★**T2("fused 조율 소진") 주장 금지**: 정본이 지목한 두 레버는
시험됐으나 같은 기전 축의 미시험 노브 최소 6개가 남아 "조율 공간
소진"은 등재 금지. ★**Gate 2 본 질문("PD 분리 자체" 귀속) 전진
없음**(A2 미제출, §1-1 NOT-YET-SUPPORTED 불변). 인용 금지 13건,
신규 방법론 게이트 #17–19(CONSENSUS §3 항목31–33과 대응) 등재.
⚠️상위 사전등록 `PREREG_GATE2_2026-08-06.md`는 워킹트리 미커밋
수정 상태. ★**정정(2026-08-09, 위 최신 항목)**: 이후 커밋
`c47fad0`으로 반영돼 워킹트리 클린 확인, 이 문구는 stale.
`CLAIM_EVIDENCE_MATRIX.md`는 이 항목을 인용한 서술이
없어 갱신 대상 없음(대조 확인 완료). 상세 `reports/CONSENSUS.md`
rev16·§1-1(E-A 블록)·§3 항목31–33, "확정된 결과" 1번·"방법론
게이트" #17–19·"다음 실험 gate" #10(E-A) 동반 갱신.
이전: 2026-08-07 (doc-steward — ★★★★★**"확정된 결과" 1번(P1)
Gate 1 산출 1 철회 — G1-b(job 875293) 사전등록 철회 규칙 발화**
[experiment-runner, `workspace/engine-port/results/p1_gates/gate1/
PREREG_G1B_2026-08-07.md`·`gate1b_result_875293.txt`]. **새 성능
판정 아님 — Gate 1 진단 산출 1의 사실 정정.** rev14/Gate 1(job
874478)이 정본에 올린 "이 격자에서 정책은 단일 분할 `(74,34)`에
고정됐다"(Zamba2 rate{2,3} 한정 관측을 무제한으로 서술)는 873944의
전 격자 `{2,3,4,6}`을 그대로 재현한 결과 **rate 6에서 `(54,54)`가
시간가중 8.37%** 등장해 사전등록 철회 규칙(`frac((54,54))≥0.01`)이
발화, **철회**됐다. rate 2·3·4는 여전히 pop A 시간가중 100%
`(74,34)`(max decode_bs 9·23·18, 문턱 36 미만)라 **인용 가능 셀
(Zamba2 r2·r3)에서는 여전히 참** — 거짓이 된 것은 격자 전체
무제한 주장뿐(과잉 철회 아님). ⚠️rate 4(18)가 rate 3(23)보다
낮은 비단조(원인 미해명). 신규 방법론 게이트 #16(스코프 확장은
인접 한 점이 아니라 원 격자를 전부 재현하라, CONSENSUS §3 항목30과
대응) 등재. `CLAIM_EVIDENCE_MATRIX.md`는 이 항목을 인용한 서술이
없어 갱신 대상 없음(대조 확인 완료). 상세 `reports/CONSENSUS.md`
§1-1(rev15)·§3 항목30, "확정된 결과" 1번·"방법론 게이트" #16·
"다음 실험 gate" #10(G1-b) 동반 갱신.
이전: 2026-08-06 (doc-steward — ★★★★**"확정된 결과" 1번(P1) 통계
방법 층 정정**[claims-auditor Gate 2 설계 감사 2회 + result-analyst
독립 재현, `workspace/engine-port/results/p1_gates/verify/`]. **새
성능 판정 아님 — 기존 정본 수치의 통계 방법 층 정정이다.** `paired_
bootstrap_ci`(n=5 percentile bootstrap of mean)는 실 coverage
0.840(명목 95%의 한쪽 오류율 ≈8.0%, 원인은 n=5 자체, n=4 0.798/
n=6 0.859/n=8 0.888)이라 소표본 판정에 부적합함이 2출처 독립
확인됐다 — 저장소 안에 `m3_analyze.py`·`tfgate_analyze.py`가 이미
독립으로 t-CI로 전환한 동일 진단이 있었는데 정본 라이브러리·P1
판정서는 반영 안 함(도구 규율 실패, 신규 방법론 게이트 #14). primary를
t-CI로 교체해 재채점: **Granite rate3(+11.7%)는 t-CI가 0을 포함
(p=0.0515) → 인용 목록에서 제외**(미검증, 철회 아님) ⇒ **인용 가능
정량치 4개→3개(Zamba2 rate2 +40.5%/rate3 +185.8%, Granite rate4
+27.0%)로 축소**. "임계 사다리 40–300ms 전 구간 부호 불변"도 정정 —
부호 불변은 **T∈[40,113.0)ms뿐**이고 그 위에서 인용 4셀 중 3셀
(Zamba2 r2, Granite r3·r4)이 술어 포화로 음전환, 끝까지 유지되는
유일한 인용 셀은 **Zamba2 r3**. "rep 부호 5/5"도 정정 — **Granite
rate2는 3/5**(효과의 87%가 rep2 1점), 나머지 7셀은 5/5 유지. **P1의
방향 자체(agnostic>fused, Zamba2 r2·r3·Granite r4)는 어떤 방법으로도
불변** — 강등되는 것은 "전 셀"·"5/5"·"사다리 전 구간"·"Granite r3
수치"뿐. E1(`s8_frontier/e1_analyze.py:492,1286`) 사전등록 결정 규칙도
같은 undercoverage를 상속하므로 "다음 실험 gate" #8에 제출
선행조건으로 등재. `CLAIM_EVIDENCE_MATRIX.md`는 이 수치를 인용하는
서술이 없어 갱신 대상 없음(대조 확인 완료). MEMORY.md·
`slo-aware-scheduling-track.md`·`deconfound-measurement-lessons.md`에
이 항목을 반영. 상세 `reports/CONSENSUS.md` §1-1(rev13)·§3 항목27.
이전: 2026-08-05 (같은 날 4차 속행, doc-steward — ★★★**"확정된 결과"
1번(P1) 갱신: P1 운영점(cudagraph-ON) 대조 감사 반영[claims-auditor
2026-08-05, jobs 873944/873945, `results/p1_opint/`]. 새 성능 판정 아님,
감사자 판정을 정본화만 함.** 정본 goodput 술어로 채점하면 agnostic v1이
fused를 Zamba2-2.7B·Granite-4.0-h-micro-base **2모델 전부·rate 2–6
정상상태 셀에서** 이기나(Zamba2 rate2 +40.5%/rate3 +185.8%, Granite
rate3 +11.7%/rate4 +27.0%, CI 0 배제), 사전등록 mean-ITL 술어로는
Granite 부호가 뒤집힌다(위반 0건=항등식이라 강등 채택 안 함) ⇒
**"Zamba2 확인"·"Granite 강등" 둘 다 등재 금지**, "항상 이득" 철회
(정상상태 decode를 5–45% 늦춤), 기전은 3-플래그 묶음 처치라 "PD 분리
자체" 귀속 NOT-YET-SUPPORTED, 실현 파티션 미측정, **NemotronH·
Falcon-H1 미측정이라 "4모델 전부" 문구 폐기**(2모델로 축소). 신규
방법론 게이트 #12(임계 사다리+큐-성장 검정)·#13(묶음 처치) + #6 새
사례(위반 0건 술어=항등식) 등재. 후속 Gate 1–4를 "다음 실험 gate" #10
으로 등재. `CLAIM_EVIDENCE_MATRIX.md`는 이 항목을 인용한 기존 서술이
없어 갱신 대상 없음(대조 확인 완료). MEMORY.md·`slo-aware-scheduling-
track.md`에 이 항목과 앞서 미반영이던 α(job 873921) 판정을 함께 반영.
상세 `reports/CONSENSUS.md` §1-1(전문). 이전: 2026-08-05 (같은 날 3차 속행, ★★★★claims-auditor 감사 완료 —
**고-D 대조(job 873921, "α") 판정 반영. §0 종결 CONFIRMED (scoped) 등급
불변, 사전등록 밴드 [12,13]ms는 REFUTED, 새 성능 판정 0건.** α(T8 d92=
(P16,D92), sticky ON, ShareGPT rate 2, n=4 블록, cudagraph ON, gpu41)가
실행됐다: pooled per-token ITL p50 = **11.26ms**(telemetry-path)/
**11.32ms**(raw-itls path), `E1_DECODE_REALIZED`=1.000/0.998/0.998/1.000,
`n_err=0`. 사전등록 3-밴드 규칙(`s2_sticky_d92.sbatch:403-408`) 적용 시
**INDETERMINATE**(12–13 밴드 미달·28–30 붕괴 밴드와도 거리 큼) —
**붕괴 분기는 REFUTED**, §0 종결은 유지. ★**밴드 [12,13]ms 자신이
REFUTED**: §1-31이 스스로 금지한 **C2→sticky 이식**으로 도출됐고, α
내부 엔진측 step 회귀로 예측한 C2 동작점(12.44–12.80)이 관측(12.875)과
0.9–3.8%만 어긋나 기록된 계통 오프셋(−5.4%)으로 격차가 소진된다 —
**워크로드 불일치**(decode-busy ctx_p50 중앙 1291 vs 287 tok, decode
batch 11.31 vs 4.51, closed-loop+keepalive vs open-loop)가 원인이며 새
기전이 아니다. 대신 α는 §0 종결을 분쟁 필드(`decode_sms`) 내부 재진술
에서 **결과(outcome) 축 앵커**로 옮긴다: block-matched ON d92/OFF
d16(872077)=**1.0293[1.0210,1.0376]**, ON d16/OFF d16=
**2.6320[2.6193,2.6447]**. **S3(하드웨어 부여 층)는 여전히 닫히지
않는다**(D92/D108을 0.6–2.9%밖에 못 벌려 검정력 없음). **E1은 4가지
사유로 여전히 열리지 않고, α는 오히려 (3) prefill 축 미통제 사유를
강화한다**(d92 TTFT p50 162.7–210.8ms vs d16 63.2–66.1ms, 무기전).
**다음 gate 재정렬**: (δ) 같은 바이너리 OFF arm 승격(10% 미만
교차-job 비교 인용의 신규 선행조건) → (β) → (γ), (α′) sticky ON을
C2 클라이언트로 1–2블록(γ와 병렬). 신규 방법론 항목(예측 밴드도 이식
금지 규칙의 적용 대상) 등재. 상세 `reports/CONSENSUS.md` §1-31·§3-26,
아래 "8B decode-SM 프론티어" "2026-08-05(α)" 소절·"다음 실험 gate"
#8. 원자료 `workspace/engine-port/results/s2_sticky/
s2a_pooled_873921.txt`·`s2a_T8_873921_result.txt`·`s2_sticky_d92.sbatch`
(전용 분석 md 아직 미작성). 이전: 2026-08-05 (같은 날 2차 속행, ★★claims-auditor 감사 완료 —
**ceiling-censoring 진단(`CEILING_CENSORING_DIAG_2026-08-05.md`)이 §1-13
각주("LO는 split에 무관심 = 지표 무신호")를 직접 검정. 새 성능 판정 0건,
HE0/§1-13 판정 자체는 불변.** 진단서 헤드라인(T=20/30/60 임계 사다리,
"[55,65) split-불변 모드가 >60 질량 지배", C2 정성 대조, "gpu39 3중
사다리")은 **전부 REFUTED**(검정력 0인 9-vs-9 이벤트·사실오류 2건·C2
caveat 위반의 3회차 재발). 그러나 **각주 자신의 결론은 다른 증거로
생존·강화**: LO goodput은 이중 절단(처리량=도착률에, pass=SLO 여유에
고정)이고, LO에도 요청별 ITL p95(p90 5.6 SD·중앙값 3.2 SD)·TTFT p50
(3.3 SD) 레버가 실재한다 — 단 SLO 예산 단위로 HI의 1/14이고 부호가
functional에 따라 뒤집혀 "어느 split이 LO에서 좋다"는 정의되지 않는다.
§1-13의 "HI 최적이 LO에서 공짜"는 goodput functional 한정으로만 참.
부수 정정 2건 허가(HI spread 43%는 legacy mean-ITL 스코어러 수치이고
정본 술어로는 +716%; LO spread는 arm마다 n이 달라 0.067→0.087 3.1%로
재산출, "spread<rep SD"로 재서술) + 노드 교락 방어(arm 내 노드효과는
arm효과의 3–19%) + 도구 결함 1건(`analyze.py:183` request_slice가
duration을 안 자름, engine-porter 후속) 기록. 상세 `reports/CONSENSUS.md`
§1-13·§3 항목24·25, 아래 "다음 실험 gate" #9. 이전: 2026-08-05 (같은 날
1차, ★★★claims-auditor CONFIRMED (scoped) — **§0 최상위
열린 항목이 behavioural하게 종결됐다. 새 성능 판정은 없다.** S2(job 873015,
sticky ON, T8 d16/d54, ShareGPT rate 2, n=8 블록, cudagraph ON, gpu37)를
2026-08-05 독립 재현: pooled per-token ITL p50 = **28.92ms**(d16, t95
[28.81,29.02]) / **12.04ms**(d54, t95[11.96,12.12]) — 사전등록
`[28,34]ms` 안, `split_frac` 라벨을 전혀 쓰지 않고 재현. §1-28/§1-30 §0의
이분법이 종결: **(i)**(872077 `decode_sms==16`이 실제 16-SM 실행이
아니다)는 **하드웨어 형태 REFUTED·라벨 형태 CONFIRMED**(872077 d16이
decode-busy 시간의 96.2%를 실제로는 D108에서 보냄) — `decode_sms`는
선택기 인덱스 재진술일 뿐 하드웨어 SM 부여 직접 프로브(S3)는 여전히
미실행. **(ii)**(C2 28–31ms=셀 배치 성질)는 **DISFAVOURED**(keepalive
없이·decode batch 2.6배 작게 C2의 0.93×로 재현). 살아남는 답 **(iii)**:
`split_frac≥0.90`이 D-파티션 클래스를 격리·완결 못한다 — 기전이 스냅샷
샘플링 케이던스(decode-busy 조건부 1/16 구간만 포착)에서 독립적으로
도출됐다(기대 순도 ≈6%, S0-R mode 분해 6.6–9%와 일치). d54 companion은
사전등록 [13,16]ms를 미달(관측 12.03, 원인=구간 도출 자체의 외삽
오류) — **인용 시 필수 동반**. §4.3.12(f) 판별 예측(T8≈1.85)도 관측
2.402로 빗나갔으나 판별 arm(Ha8) 미제출이라 **설계상 미판정**(모형
반증 아님). ★**이 런에는 결과 게이트가 0개**(전부 sticky ON 하 항등식
또는 코드 불변식) — **방법론 게이트 #9의 네 번째 재발**로 등재.
`S2_ANALYSIS_2026-08-04.md`의 "Instrument check" 문단(61-64행)에
정정 표시(원문 보존) — decode-idle 케이던스를 decode-busy로 오인,
90–260× 오차. **§0 최상위 열린 항목**은 "미해소 3지선다"→
"CONFIRMED(scoped)로 종결, 단 하드웨어 층 미프로브"로 전환. C2
자체(레버 존재, 2.36–2.91×, scoped)의 등급·수치는 **불변**. **E1은
4가지 독립 사유로 여전히 열리지 않는다**: `G_LEVER`/`G_FLAT` 여전히
UNDETERMINED / sticky 기판이 estimand를 바꿈(decode-busy 시 prefill
92 SM이 벽시계 ~77% 유휴 — co-located 예산 배분이 아니라 단일-테넌트
측정에 가까움) / prefill 축 미통제(TTFT p50 46.3→63.2ms 무기전) /
음성대조 구조적 부재(d16 UNSPLIT n=0/8블록)+S3 미실행. ⇒ **긴장
A(HE2 vs C2)는 전혀 닫히지 않았다.** 다음 gate: (α) sticky-ON 고-D
대조 셀(사전등록 예측 p50→12–13ms, 유일한 값싼 반증 실험) 우선. 상세는
아래 "8B decode-SM 프론티어" "2026-08-05" 소절,
[`results/s2_sticky/S2_REPLICATION_2026-08-05.md`](workspace/engine-port/results/s2_sticky/S2_REPLICATION_2026-08-05.md)
(전문), `reports/CONSENSUS.md` §1-31·§3-23, `results/s8_frontier/DESIGN.md`
§4.3.16. 이전: 2026-08-03 (★★★같은 날 4차 속행, doc-steward 기록 — **상태
기록, 성능 판정 아님. GPU 런(S2)은 별도로 제출 중이고 결과는 아직 없다.**
§1-28/§0이 세운 이분법((i) 872077의 `decode_sms==16`이 실제 16-SM 실행이
아니다 / (ii) C2의 28–31ms가 셀 배치 성질이다)이 **유지 불가**임이
확인됐다 — 두 job이 같은 축에 있다는 점은 유지되나, `split_frac≥0.90`이
D 파티션 실행 토큰을 식별하지 못한다는 **세 번째 후보 (iii)**가 실측으로
문서화됐다. claims-auditor가 `FINDINGS_S0_AXIS_2026-08-03.md`를 감사하며
낸 재프레이밍(자기감사, 방법론 교훈 12) — E1 SPLIT 모집단은 **이봉**이고
윗봉이 C2 셀별 p50과 1–2% 일치(d16 0.992/d24 0.991/d44 1.013), 아랫봉은
같은 job UNSPLIT과 통계적으로 동일(1.011) — 을 **result-analyst가 독립
실행으로 재검증**(`S0R_REPLICATION_2026-08-03.md`, claims-auditor도
사전등록 작성 세션도 아님): 행 1·3 재현, 행 2는 순서만(d24−d16 미해결,
t=1.90<2.365), **행 5(클럭 lag)는 미발화**(최적 δ=+0.10s에서도 slow
share 11.4%). ★**행 4(음성대조)가 강한 형태를 죽였다**: UNSPLIT(108 SM)
클래스도 전 셀에서 이봉이고 느린 봉 위치가 셀을 따라간다(33.88→22.12→
15.62→14.12ms, D 16→24→44→54; d16 슬로우 토큰 8,893개 중 **7,903개
(88.9%)가 UNSPLIT 라벨**) ⇒ SPLIT은 배타가 아니라 **농축**(2.33–3.24×).
남은 두 읽기(셀 수준 현상 vs 클럭 오프셋 누출)는 **오프라인으로 분리
불가** — S2(GPU, 별도 제출 중, 결과 없음)가 인과 시험. **철회 3건**
(메인 세션이 같은 날 앞서 씀): "§0 stands as written"·"aggregation-
invariant"·"11.09는 집계 단위 미기록"(**틀림** — 산출자는
`m3_conditional.report_conditional` [3] `sp_p50=11.0905`,
`m3_conditional.py:158-161,251-262,316-329`에 문서화, 없는 건 stdout
저장분뿐). **재사용 계측 결함 2건**: `c2_anchor.py` 표 [5]가 M8 전체·
Ha8 d16을 조용히 누락(865493 앵커: T8·Hs8=5셀 전부/Ha8=d16 없음/M8=전무,
Ha8 d16 rep은 **돌았다** `itl_ms_p50=112.84` n=5200 — 측정 부재가 아니라
텔레메트리 앵커 부재) · mode estimator 60ms 상한은 **arm-이식 불가**
(Ha8은 토큰의 0.16%만 창 안). **증거 수준**: 강한 형태(레버=D-SM 실행
식별)는 **채택 불가**(음성대조 반증), 약한 형태(이봉·농축)는 **재현됨
(독립성 부분적** — 추정량은 감사자 제안, 사전등록은 메인 세션, 실행만
독립**)**. §0의 (i)/(ii)는 **여전히 미판정**(이제 3지선다). 게이트
S1(§4.3.13)은 "부분 실현" 분기가 없어 **현 상태로 실행 불가**.
`G_LEVER`/`G_FLAT`는 §4.3.12(d) 그대로 UNDETERMINED(이번 회차로도
미해소). 상세는 아래 "8B decode-SM 프론티어" "2026-08-03(4차)" 소절,
`reports/CONSENSUS.md` §1-30·§3-18·§3-19, `results/s8_frontier/DESIGN.md`
§4.3.15, 사전등록 3건(`PREREG_S0_AXIS_2026-08-03.md`·`PREREG_S0R_MODE_
2026-08-03.md`·`PREREG_S2_STICKY_ITL_2026-08-03.md`), 재현 판정
`S0R_REPLICATION_2026-08-03.md`. 이전(같은 날 3차 속행, doc-steward
기록 — 두 갈래 완료, 둘 다 앞선 2차 속행의 일부를 정정한다. **(1) C2 → `G_LEVER` 앵커 경로
= 폐기**(`c2_anchor.py`, claims-auditor 감사: 주장 1만 생존, 2–5 REFUTED/
NOT-YET-SUPPORTED). **`G_LEVER`/`G_FLAT`는 §4.3.12(d) 그대로 UNDETERMINED
유지**(이 시도로 해소 안 됨). ★**§0 신규 결정적 발견(최상위 열린 항목)**:
같은 arm·같은 서버 플래그·매칭 batch에서 C2(865493)와 872077(E1 격자)의
"decode 16 SM" per-token ITL p50이 **2.6× 다르다**(28.79ms vs 11.09ms) —
872077의 `decode_sms==16`이 실제 16-SM 하드웨어 실행이 아니거나(green
context 생성이 SM 부여를 보장하는지 미재프로브), C2의 28–31ms가 decode-SM
비용이 아니라 그 셀 배치 성질이거나 둘 중 하나이며 **872077 전체와 sticky
결과가 딛고 선 바닥**이다. 주장 1(realized 검증)은 CONFIRMED이나 서술 2건
정정 필수(활성률 0.66–0.93은 count-weighted, 시간가중은 0.99+; 108 SM
시간은 warmup 아니라 drain 전용). 주장 2(primary p95→p50)는 관측
CONFIRMED·기전/처방 REFUTED(p50 전환은 872077 T8 양성대조조차 1.00으로
만들어 **캠페인을 구조적 NO VERDICT로 확정**하는 처방) ⇒ **primary=
`p95(SPLIT)` 유지**. 주장 3–5(편향=하한/`G_LEVER=1.41`/`G_FLAT=1.25`)는
NOT-YET-SUPPORTED/REFUTED/REFUTED. **(2) D=54 앵커 측정 취소**(jobs
872920/872921, 제출 17분 뒤 취소) — 감사가 전제를 폐기했고, 독립적으로
**keepalive 토큰 초과**(`s8_keepalive_prompt_224.txt` **1794** tok(★2026-08-16
정정, 구 1793 — 표기만 정정, 아래 G-1 dated정정3) > `CTXCAP`
1792)로 `s8_scaleup` 캠페인 전체가 재현 불가임을 발견[미감사, 코드/로그
직접 검증]. **독립 수렴[AUDITED]**: C2의 높은 residency는 decode 파티션
제어가 아니라 keepalive 워크로드 장치의 산물(3경로 독립 도달) — C2 앵커가
죽는 세 번째 이유. 취소됐으나 캠페인 설계(d16+d54, 4 block)는 재사용 가능.
상세는 아래 "8B decode-SM 프론티어" "2026-08-03(3차)" 소절,
`reports/CONSENSUS.md` §1-28·§1-29, `results/s8_frontier/DESIGN.md`
§4.3.13–4.3.14, `results/s8_scaleup/NOTES_D54_ANCHOR_2026-08-03.md`. 이전
(같은 날 2차 속행, doc-steward 기록 — 두 갈래 완료.
**(I) claims-auditor의 추정량 이관**: 아래 1차 속행이 확인한 `A_free`
결함을 대체하는 **조건부 per-token 추정량**(`results/s8_frontier/
m3_conditional.py`) — 단위는 개별 ITL 구간 1개, SPLIT(`split_frac≥0.90`)/
UNSPLIT(`≤0.10`) 라벨(사이는 배제), primary `p95(SPLIT)` 비 + UNSPLIT
control(대비가 정의상 0). **[AUDITED]**: `A_free`가 요청 ~9.6개에 얹히는
극단꼬리 통계였음을 확인(T8 d16 pooled p95 11.60 vs `A_free` 28.31),
client↔telemetry 정렬은 `phase=="benchmark"` 필터 **금지**(그 마커는
warm-up 요청 발화, probe 경계 아님) 규율 확정, `ALIGN_R_MIN=0.95`
flag-only(배제하면 대비가 오히려 커짐, LOO 실측). **[UNAUDITED — 감사자가
자기 산출을 자기가 감사, 별도 확증 전 인용 금지]**: `PREFILL_BLOCK_TOK`
임계 스윕(1024→0)에 무릎 없음·임계 0에서 대비 소멸(임계는 자유 모수가
아니라 답을 정하는 손잡이). 권고 = `A_free` 은퇴, primary 라벨 = realized
partition(`decode_sms==D`). **(II) engine-porter의 `PDMUX_STICKY_PARTITION`
구현 완료 + correctness gate 통과**[구현 사실, 성능 판정 아님]:
decode-busy 시 무분할 fallback 우회, decode-empty 시엔 의도적 index-0
release(hold 아님), OFF는 short-circuit으로 patch 전과 byte-identical(독립
재구현 pre-patch selector 대비 전 격자 동등성 테스트), cudagraph 보존.
**correctness gate 전부 PASS**: CPU 회귀 40 tests + sticky 단위 테스트
12건 + **GPU smoke(job 872800, Ha8 d16)** 고정 프롬프트 6개 greedy 출력
OFF/ON byte-identical. **realized 관측(n=1, 성능 아님)**:
`E1_DECODE_REALIZED` sticky OFF 0.0839(기존 동작 재현) → **ON 1.0000**
(사전등록 ≥0.90 초과, 튜닝 없음). **구현 완료 ≠ 성능 주장 성립.** **(III)**
sticky 런 사전등록 기록 — 872077 소급 재분석은 **DIAGNOSTIC 전용**(재분석
사후 채택 금지), primary 1개(`p95(SPLIT)` 비) 선언, 게이트 4종, **`G_LEVER`/
`G_FLAT`는 미결정으로 기록**(스케일 불일치로 기존 1.5/1.15 그대로 이전
불가). 상세는 아래 "8B decode-SM 프론티어" "2026-08-03(2차)" 소절,
`reports/CONSENSUS.md` §1-27, `results/s8_frontier/DESIGN.md`
§4.3.10–4.3.12. 이전(같은 날 1차 속행 — 2026-08-03 이른 회차가 세운
"pin 게이트=항등식·decode 실현 4–19%"(§1-25급) 위에서, 메인 세션이 그
희석을 `g = A_free(d16)/A_free(d54)`의 **보정 모형**으로 확장했다가
claims-auditor가 **REFUTED**시켰다 — control-arm reductio(T8에 같은 보정
적용 시 corrected g 21–29×로 C2를 10배 위반)·de-engagement 직접 실험
(w=0에서도 g 거의 불변, 1–11%만 이동)·"A(108) 셀 무관" 가정의 실측 위반
(UNSPLIT-only 부분집합만으로 T8 헤드라인이 그대로 재현) 3중. 동시에
job **872077**의 NO VERDICT 사유가 "CI 폭 부족"에서 **"estimand
미식별"**로 확장됨 — `initialize_stream_groups`가 마지막 무분할 그룹을
항상 덧붙이는 이 기판에서는 "decode가 D SM에서 돌았다"와 "prefill이
동시에 in-flight였다"가 같은 사건이라, 어떤 통계도 decode-SM 탄력도와
prefill 간섭을 분리 못 한다(§1-24와 결합, n으로 해결 안 됨). **`g`는 이
격자 한정 은퇴**(sticky-partition 기판 수정 전 인용 금지), **블록 증설
재실행은 선행 금지**. `A_free` 추정량 자체도 결함(blocking 필터가 prefill
작업의 74–77%를 통과시켜 stall 오염이 d16–d54까지 확장 + 극단 percentile
퇴화) + arm 비교의 decode-batch-size 미제거 교락도 확인. 부수(UNAUDITED,
정본 인용 금지): result-analyst의 `m3_decode_empty.py` 진단이 희석 원인을
decode-empty가 아니라 **prefill 부재**로 재귀속하고 죽은 telemetry 필드
4개를 식별 — 단 "부하를 올려도 engagement가 안 는다"는 결론만은
claims-auditor와 독립 수렴해 그 좁은 항목만 인용 가능. 상세는 아래
"8B decode-SM 프론티어" 절·`CONSENSUS.md` §1-26·`results/s8_frontier/
DESIGN.md` §4.3.9. 이전: 2026-08-02 (진행 상태 갱신만, 결론 개정 아님 — 2026-08-01에 실행된
E1 전제 실험 4건(jobs **870295**=M8 / **870296**=Ha8 / **870297**=Hs8 용량
스캔, **870301**=T8 batch-cap)의 결과를 **상태로만** 기록. ⚠️**이 4건은 전부
claims-auditor 미통과 = 정본·논문 인용 금지**이며 등급어는 **미검증**이다
(아래 "열린 긴장"의 `results/s8_frontier/` 절). 새로 확정으로 올린 것은
**소스 읽기로 검증되는 코드 사실 2건뿐**(`--max-running-requests`가 arm
계열마다 다른 손잡이 · `kv_mamba_occupancy=1.0`은 항등식 — "방법론 게이트"
#5)이고, 여기서 **방법론 게이트 #6**("항등식을 증거로 쓰지 마라")을 신설했다.
이 세션에 세웠다가 claims-auditor에 반증돼 **철회한 6건**은 "철회된 가설"
절에 기록. E1 본 스윕은 **미제출**(설계 위험 3중, "다음 실험 gate" #8).
세션 전문은
[`handoff-report/session_handoff_2026-08-02.md`](handoff-report/session_handoff_2026-08-02.md)).
이전: 2026-07-31 (진행 상태 갱신만, 결론 개정 아님 — `results/s8p_prefill/`
**완료**[claims-auditor 미통과, 정본 인용 금지 유지]·`results/s8_frontier/`(E1)
하네스 구축 완료·본 스윕 미실행. 하네스 전제 job 867231(T8 용량 스캔)·867298
(관측자 효과 게이트)은 2026-07-29 세션 핸드오프에는 PENDING으로 기록됐으나
★**2026-07-31 `sacct` 재확인 결과 둘 다 2026-07-29 21:15–22:41에 이미
COMPLETED**(핸드오프의 예측이 backfill로 빗나감). ★**2026-07-31 정정**: 같은 날
앞선 갱신은 "분석/판정 파일 0건"이라고 적었으나 **오기** — 두 job 모두 **분석이
job 안에서 이미 실행**되어 결과가 `e1cap_T8_867231_result.txt`·
`tfgate_T8_867298_result.txt`에 있고, 판정은 `handoff-report/
session_handoff_2026-07-29.md` §10에 기록돼 있다. 없는 것은 별도 FINDINGS 문서뿐이다.
관측자 효과 게이트 = **조건부 통과**(d92에서만 `itl_p95 +2.00%` t95 [+0.78,+3.21] —
결정 지표 위 비대칭 교란) ⇒ **본 스윕은 `PDMUX_TRACE_FORCE_PREFILL=0`, pin 검증만
별도 ON 런으로 분리**(사전등록: `results/s8_frontier/DESIGN.md` §4.7.1).
⚠️**T8 용량 스캔 = ITL 축이 전 구간 non-binding**(claims-auditor 2026-07-31
확정, 아래 "열린 긴장"). ★**이 절의 2026-07-31 초판이 쓴 "5셀 동시 off-cliff rate
부재 = §4.2/§9.1 escalation 발동"은 부정확해 철회** — rate≲2에서는 d92 포함 전 셀이
plateau 위에 있어 §4.2가 정의한 "D16과 D92 안전대 비중첩"은 엄밀히는 발동하지 않았다.
실제 명제는 **"공통 off-cliff band(≲2–3 req/s)가 ITL 항이 조금이라도 움직이는 영역과
완전히 분리돼 있다"**이다. knee 수치도 정정(첫 교차 기준): d16 12.6 / d24 16.0 /
d44 16.0 / d54 **8.45** / d92 **2.80** req/s(초판의 4.10은 "임계 아래 마지막 점"을
쓴 값이며 곡선이 단조가 아니라 두 정의가 크게 갈린다). **견고한 것은 d92 knee가
나머지보다 3배 이상 낮다는 순서**(세 임계 × 두 x축에서 불변).
★신규 방법론 게이트(집계 단위 선확정 — "방법론 게이트"
절 #4) + green-context auto-revert 실현-배분 관측 사실(`reports/CONSENSUS.md`
§1-22) 등재. 상세는 아래 "열린 긴장"·"다음 실험 gate"·"방법론 게이트" 절).
이전: 2026-07-28 (★★claims-auditor가
사전등록 게이트(`workspace/engine-port/
results/s0_deconfound/DESIGN.md` §5)를 집행 — **부분 GO**. **C1 CONFIRMED**:
Stage 0(2026-07-26)이 인용한 "D108 무경합 앵커"는 코드 버그로 **실제로는 decode
16 SM**이었음이 3중 독립 증거(코드 기전·telemetry 재집계·클라이언트 시그니처)로
확인 ⇒ Stage 0 판정2(NULL)·판정3(게이트 non-binding)을 **철회**, 판정1(raw ITL(D)
스윕 = CONFOUNDED)만 생존, long-ctx L−1 이상은 "게이트 실패로 보류"가 아니라
**"게이트 미실행"**으로 복원. **C2 CONFIRMED(scoped)**: prefill을 16 SM에 고정한
채 decode-SM만 올리면 ITL이 **2.36–2.91× 개선**(4 arm, 모델-무관) — 측정 노트로
정본 진입하되 **정책 이득이 아님**(프론티어 `[108−D,D]` 미측정). **C2b("hybrid
급락=Zamba2 성질") NOT-YET-SUPPORTED**로 강등. **Claim A 등급 변경 없음**(부분
지지), **HE0/HE2/§1-5/§1-7 철회 안 함** — 대신 긴장 2건을 열린 항목으로 기록.
상세는 아래 "Stage 0" 절·"8B decode-SM 민감도 측정 노트" 절·"열린 긴장" 절·
"다음 실험 gate"). 이전: 2026-07-26 (Stage 0[long-ctx L−2 게이트]: 운영점 decode
SM-무감각을 hybrid·16k ctx까지 확장 확인 — non-binding, long-ctx 충돌 가설 이
regime서 붕괴, HE0/벡터1 ctx-무관으로 강화 — ★★2026-07-28 이 판정의 핵심 근거가
철회됨, 위 참조). 2026-07-25 (벡터1[G2.0 short-ctx disjoint
conflict-regime]: CONFIRMED closure, scoped — narrow-rA 확증 sweep g2_0_raconf
완료; 같은 날 논문 positioning 판정[multiplexing 신규성 축 + Transformer-control
게이트, cross-substrate 이식 프레이밍 철회] "다음 실험 gate" #6 추가). 이 문서가
프로젝트 전체의 유일한 현재 상태
정본이다. 이전 문서와 충돌하면 이 문서와
[`reports/paper/`](reports/paper)의
판정을 우선한다.

## 논문 방향

Layer composition을 runtime scheduling boundary로 사용하지 않는다. Hybrid
구조는 coarse-grained decode floor를 예측하는 offline model profile로만
사용한다.

- **H-Architecture:** 역할별 queue, host issue loop, CUDA stream을 실제로
  분리하면 single-worker control-plane coupling을 줄일 수 있다.
- **H-Policy:** model profile과 runtime load로 예측한 decode floor가 시간적으로
  상보적인 near-saturation workload에서 generic dynamic 및 global static보다
  높은 SLO goodput을 제공한다.

두 가설은 아직 검증되지 않았다. 구현 완료와 성능 주장 성립을 구분한다.

## 확정된 결과

1. ~~여러 Hybrid 모델에서 prefill/decode resource separation은 fused
   execution보다 유리한 operating point를 제공한다.~~ → ★반증/정정
   (2026-08-05) **PD-mux 활성화는 운영점에서도 꼬리 SLO goodput 이득 —
   술어·모델·워크로드 한정, 기전 귀속 미확립.**

   ★**스코프 축소(2026-08-04, claims-auditor)**: 이 4-모델 캠페인은
   **전부 `--disable-cuda-graph`**(no-cudagraph 비운영점,
   `triage/p1_7_bench_one.sbatch:42`)이고 **rate 1에서는 동률**(도착률
   천장)이며, **운영점(cudagraph-ON) 대조는 어느 모델에서도 측정된 적
   없다.** ★**반대 증거 신규**: fused의 死因은 **TPOT > 60ms 임계
   초과**(Granite rate4 TPOT 61.21)인데 **cudagraph가 그 벽을
   제거한다**(plain TPOT 62.70→13.51ms, rate4 82.41→**54.04ms=60ms SLO
   통과**, `workspace/engine-port/results/cudagraph_probe/
   cudagraph_results.md` Probe 1 — Zamba2 단일모델 관측이라 Granite에
   직접 이식은 아니나 死因 메커니즘이 cudagraph로 해소 가능함을
   시사한다) ⇒ **이 항목이 운영점에서 축소되거나 소멸할 가능성이
   있다.** 검증 실험(2모델×{plain,agnostic}×cudagraph-ON×n≥4, ≈6
   GPU-hr) 진행 예정, 결과 없음. 상세 `reports/CONSENSUS.md` §1-1,
   `reports/layertype_dynamic_POSITIVE_2026-08-04.md` §2.0.

   ★★★**반증/정정(2026-08-05, claims-auditor 감사, jobs 873944/873945,
   Zamba2-2.7B·Granite-4.0-h-micro-base, n=5 paired,
   `workspace/engine-port/results/p1_opint/`)**: 위 "축소되거나 소멸할
   가능성"은 낡았다 — 운영점(cudagraph-ON) 대조가 처음 측정됐고,
   **소멸하지 않았으나 "확인"으로 올라가지도 않는다.** 정본 goodput
   술어(TTFT≤3s ∧ 요청 내부 token-ITL p95≤60ms)로 채점하면 agnostic
   v1이 fused를 **2모델 전부·rate 2–6 정상상태 셀에서** 이긴다(Zamba2
   rate2 +40.5%, rate3 +185.8%; Granite rate3 +11.7%, rate4 +27.0%,
   paired CI 0 배제). 그러나 **사전등록 mean-ITL 술어로는 Granite
   부호가 뒤집힌다**(그 셀들은 위반 요청 0건이라 goodput≡throughput
   항등이라서 강등을 채택하지 않는다) ⇒ **"Zamba2 P1 운영점 확인"·
   "Granite P1 전면 강등" 두 문장 모두 정본 등재 금지.** **"항상
   이득"은 철회한다** — pdmux는 정상상태 per-token decode를 5–45%
   늦추고 Granite raw 처리량은 −0.3~−2.6%다(**이득은 꼬리, 비용은
   중앙**). **기전은 3-플래그 묶음 처치**다(`--enable-pdmux`가
   `--chunked-prefill-size -1`·`--disable-overlap-schedule`을 assert로
   강제, `sglang/srt/server_args.py:6125-6137`)이고 fused arm은 관측
   실패모드(prefill 배치 stall)에 미조율 baseline이라 **"PD 분리
   자체가 원인"은 NOT-YET-SUPPORTED** — 지지되는 것은 "이 엔진에서
   PD-mux를 켜면 기본 설정 fused보다 꼬리 SLO goodput이 좋다"뿐이다.
   실현 파티션 미측정(`PDMUX_TELEMETRY_PATH` 미설정)이라 파티션·동시성
   기전 문장은 여전히 금지. **NemotronH·Falcon-H1은 운영점 미측정 ⇒
   "4모델 전부"는 더 이상 쓸 수 없다**(scope는 2모델로 축소).
   신규 방법론 게이트 **#12**(임계 지시함수 판정은 임계 사다리·큐-성장
   검정으로 견고성을 보여라)·**#13**(arm 대조가 엔진 제약으로 다중
   플래그를 강제하면 그것은 묶음 처치다) + 기존 **#6의 새 사례**(위반
   0건인 술어의 goodput은 throughput의 다른 이름이다) 등재(아래
   "방법론 게이트"). 인용 금지 목록·후속 Gate 1–4·전체 scope는
   `reports/CONSENSUS.md` §1-1(전문) 참조, 판정서
   `workspace/engine-port/results/p1_opint/P1_OPINT_RESULT_2026-08-05.md`.

   ★★★★**통계 방법 층 정정(2026-08-06, claims-auditor Gate 2 설계 감사
   2회 + result-analyst 독립 재현, `workspace/engine-port/results/
   p1_gates/verify/`). 새 성능 판정 아님 — 위 수치들의 통계 방법 층
   정정이다.** `paired_bootstrap_ci`(`benchmarks/pdmux_eval/
   analyze.py:115-142`)는 n=5 percentile bootstrap of the mean(BCa·
   studentization 없음)이라 실 coverage가 **0.840**(100k trial MC)에
   불과해 명목 95%의 한쪽 오류율이 **≈8.0%**(명목 3.2배)다 — 원인은
   고정 seed도 정규 가정도 아니라 **n=5 그 자체**(n=4 coverage 0.798/
   n=6 0.859/n=8 0.888). `unpaired_bootstrap_ci`도 동일 결함(n=4/arm
   0.856). ★**저장소 안에 이미 같은 진단이 두 번 독립으로 존재했다**
   (`results/s8_frontier/m3_analyze.py`·`results/e1_traceforce/
   tfgate_analyze.py`가 각자 로컬로 t-CI로 전환)는데 **정본 라이브러리와
   이 P1 판정서는 percentile bootstrap을 계속 썼다** — 통계 문제가
   아니라 **도구 규율 실패**. primary를 t-CI로 교체해 재채점: **Granite
   rate3(+11.7%)는 t-CI [−0.0032,+0.6132]가 0을 포함(p=0.0515)** →
   인용 목록에서 제외(**미검증으로 재분류, 철회는 아니다** — boot CI는
   여전히 0을 배제) ⇒ **인용 가능 정량치는 4개→3개(Zamba2 rate2
   +40.5%/rate3 +185.8%, Granite rate4 +27.0%)로 축소**. "임계 사다리
   40–300ms 전 구간 부호 불변"도 정정 — 부호가 유지되는 구간은
   **T∈[40,113.0)ms뿐**이고 그 위에서 인용 가능 4셀 중 3셀(Zamba2 r2,
   Granite r3, Granite r4)이 음으로 뒤집힌다(뒤집힘의 정체는 절벽이
   아니라 **술어 포화** — 뒤집히는 셀은 T≥150에서 양 arm 위반 0/0,
   그때 goodput 효과는 raw throughput 효과와 소수 4자리까지 항등,
   ★§3-24가 REFUTED한 "검정력 0인 임계 사다리"의 재발); 부호가 끝까지
   유지되는 유일한 인용 가능 셀은 **Zamba2 r3**. "rep 부호 5/5"도
   정정 — **Granite rate2는 실제로 3/5**(per-rep diff −0.0026/
   **+0.2789**/+0.0321/−0.0035/+0.0164, 효과의 87%가 rep2 한 점),
   나머지 7셀은 5/5 유지 확인. n=5 paired 정확 부호뒤집기 순열검정의
   두측 p 하한 = **2/32=0.0625**이므로 이 프로젝트의 n=5 paired 셀은
   분포무가정으로 p<0.05에 원리적으로 도달 불가하다(기존 "CI가 0
   배제"는 전부 모수 가정 의존이었다는 뜻). ★**P1의 방향 자체는
   살아남는다**: agnostic > fused(꼬리에서)는 Zamba2 r2·r3, Granite
   r4에서 어떤 방법으로도 유효하다 — 강등되는 것은 "전 셀"·"5/5"·
   "사다리 전 구간"·"Granite r3 수치"뿐, 과잉 강등은 아니다. **열린
   불일치(반영 보류)**: "Granite rate2는 경계(48ms로 내리면 소멸)"
   문장은 방향이 반대라는 지적(임계를 내리면 오히려 커짐)이 있으나
   "48ms" 수치의 출처가 `P1_OPINT_RESULT_2026-08-05.md`·`PREREG.md`
   어디에도 없어 수치는 유지하고 "출처 미확인·방향 불일치 지적 있음
   (2026-08-06), 확인 전 인용 주의" 표시만 추가한다. 신규 방법론
   게이트 **#14**(아래 "방법론 게이트" 절, CONSENSUS §3 항목27과
   대응 — n≤8 반복에서 `paired_bootstrap_ci`/`unpaired_bootstrap_ci`
   구간을 판정에 쓰지 않는다, primary=t-CI) 등재 — E1
   (`s8_frontier/e1_analyze.py:492,1286`) 사전등록 결정 규칙도 같은
   undercoverage(net-positive 방향 편향)를 상속하므로 아래 "다음
   실험 gate" #8에 **제출 선행조건**으로 등재. 상세 `reports/
   CONSENSUS.md` §1-1(rev13)·§3 항목27, 원자료 `workspace/engine-port/
   results/p1_gates/verify/`.

   ★★★★★**Gate 1(job 874478, 2026-08-06) 조건부 채택**[claims-auditor 감사,
   `workspace/engine-port/results/p1_gates/gate1/`]. **새 성능 판정 아님 —
   진단 전용.** Zamba2-2.7B·agnostic v1·cudagraph-ON·rate{2,3}·n=1의
   873944 텔레메트리 재현런에서, decode-busy ∧ prefill-in-flight 구간의 selector
   라벨은 시간가중 100.00%가 `(74,34)`였다(pooled 51.9 s, 78 에피소드, 3,820 스냅샷).
   ⚠️**이 통계는 판별력이 사실상 없다** — prefill 어드미션과 `adjust_stream_groups()`
   사이에 telemetry sync가 없어 "pop A ∧ idx∉{1,2}"는 관측 가능한 상태가 아니다(방법론
   게이트 #9 다섯 번째 재발). ~~**실질 산출 1**: 이 격자에서 정책은 단일 분할
   `(74,34)`에 고정됐다(`decode_running_batch_size` 최댓값 23 < 문턱 36 ⇒
   `(54,54)` 0회 선택).~~ → ★★★★★**철회(2026-08-07, G1-b, job 875293) —
   사전등록 철회 규칙 발화, 새 성능 판정 아님(진단 산출 1의 사실 정정).**
   rate 2·3·4는 여전히 pop A 시간가중 100% `(74,34)`(max decode_bs 9·23·18,
   문턱 36 미만)지만 **rate 6에서 `(54,54)`가 시간가중 8.37%** 등장해 위
   무제한(격자 전체) 문장이 거짓임이 확인됐다 — **인용 가능 셀(Zamba2
   r2·r3)에서는 여전히 참**, 거짓이 된 것은 격자 전체 주장뿐(과잉 철회
   금지). ⚠️rate 4(18)가 rate 3(23)보다 낮은 비단조(원인 미해명). 상세
   `reports/CONSENSUS.md` §1-1(rev15, "★실질 산출 1" 철회 블록)·§3 항목30.
   **실질 산출 2**: duty cycle(창 시간 기준): 동시 in-flight 27–38%, decode
   단독 32.2%, prefill 단독 3.5%, 완전 idle 26.3%. ⇒ 위 §1-1의 "실현 파티션
   미측정"은 **부분·조건부
   해제**로 바뀐다: Zamba2 rate{2,3}·agnostic v1·cudagraph-ON의
   decode-busy∧prefill-in-flight 구간에서 `(74,34)`(selector-level, S3
   하드웨어 프로브 미실행) 하나만 인용 가능 — **rate 4·6과 Granite 전체는 여전히 미측정**이며, 특히
   위 인용 가능 정량치 중 **Granite rate4 +27.0%에는 이 파티션 문장을 붙이지 않는다.** **"PD
   분리 자체" 기전 귀속은 여전히 NOT-YET-SUPPORTED**(Gate 2 소관, Gate 1로 해소 안 됨).
   인용 금지 목록(신규, 감사자 열거 — "측정했다"/"실행됐다" 등 항등식·S3-미실행 표현 포함 10건)· 커버리지
   가드 논거 정정(좁은 형태만 채택: coverage 실패가 UNLOCK을 만드는 경로는 없다, 단 이 논거는 정본
   일반 원칙으로 승격하지 않는다)·후속 Gate G1-a–d 전문은 `reports/CONSENSUS.md` §1-1
   Gate 1 블록·§3 항목28·29 참조. 아래 "방법론 게이트" #9·#15, "다음 실험 gate"
   #10(Gate 1 항목) 동반 갱신. ★**G1-b(2026-08-07) 철회 후속** — 아래
   "방법론 게이트" #16, "다음 실험 gate" #10(G1-b) 동반 갱신.

   ★★★★★**(2026-08-07 실행, 2026-08-09 정본 반영 복구) Gate 2 rev4 본
   캠페인(jobs 875344/875346, 5.86 GPU-hr) — 정본 누락 복구.** 새 성능
   판정 아님 — 이미 완료된 결과의 정본 반영 누락을 메운다. 원자료
   `workspace/engine-port/results/p1_gates/gate2/g2_report_
   zamba2-27b_875344.json`·`g2_report_granite-40-h-micro-base_
   875346.json`(rev4 사전등록 `PREREG_GATE2_2026-08-06.md` §14 적용,
   arm A1=`plain`/A2=`plainaux`/A3=`chunk512`/A4=`agnostic`).
   **Primary(§4.1/5.2, A3 vs A4 TOST)**: 5셀(Zamba2 r2·r3, Granite
   r3·r4·r6) 전부 `Rprime4`(A4 유의 우세, 등가 미발화, `p_tost=1.0`)
   — **chunk512는 pdmux를 대체하지 못한다.** F-E(정상상태) 스크린
   통과로 크기 인용 가능한 셀은 **Zamba2 r2·Granite r3 둘뿐**(나머지
   3셀은 chunk512측 F-E 발화, 부호만). **Secondary — R1′/R2′(§5.2
   표 마지막 행, A2=`plainaux` vs A4=`agnostic`)**: rev4 §1.1의 등식
   A2=A4−pdmux(realized server_args 차이는 `enable_pdmux`·
   `pdmux_config_path` 두 필드뿐)이므로 이 비교는 §1-1의 "3-플래그
   묶음 처치" 교락 중 pdmux 고유분을 분리한다. 메인 세션이
   2026-08-09 원자료(`per_arm_x60`)에서 독립 재현(paired-t, n=10):
   Zamba2 r2 **+0.834**[+0.781,+0.887]·r3 **+0.930**[+0.897,+0.963],
   Granite r3 **+0.855**[+0.816,+0.894]·r4 **+0.944**[+0.920,+0.968]
   (이상 부호 10/10, F-E clear), r6 **+0.960**[+0.929,+0.991](**부호만**
   — agnostic 측 F-E flagged). sign-flip 순열 p = 2/1024 =
   **0.001953125**(n=10 분포무가정 두측 하한, 5셀 공통). **묶음
   기여 분해**: A2는 A1(`plain`)과 사실상 같고(5셀 |A1−A2|≤0.023)
   그 부호는 **A4에 불리한 핸디캡 방향**이다 ⇒
   `--chunked-prefill-size -1`·`--disable-overlap-schedule` 두
   플래그만으로는 pdmux 이득이 재현되지 않는다 — §1-1 "3-플래그
   묶음 처치" 교락 중 이 두 플래그는 **원인에서 배제**된다.
   ★**provenance(인용 시 필수 동반)**: 이 A2-vs-A4 비교는
   **사전등록 분석기 `g2_analyze.py`가 계산하지 않는다** —
   `tost_equivalence` 호출은 코드 전체에서 1회뿐(`:543`)이고 그
   입력은 A3-vs-A4(`chunk512`-`agnostic`, `:538-539`)뿐이다. A2는
   `:734`에 서술 문장으로만 등장한다. 즉 **arm·n·raw 데이터는
   사전등록(rev4)이지만, 이 비교 자체는 저장된 primary 산출물
   (`per_arm_x60`) 위에서의 사후 계산**이다(코드 확인 2026-08-09).
   §3 항목34로 등재(항목33 "사후 지정 셀 이동"의 형제 사례).
   ★**천장 포화(해석 제한)**: fused arm X_60 평균이 0.72–0.99
   (천장 근접), agnostic이 0.00–0.12(바닥 근접)이다 ⇒ **부호는
   견고하나 크기는 포화 구간의 값**이며, E-A 블록이 이미 건 "기전
   해석 금지"가 여기도 적용된다. ★**해금 범위의 상한(overclaim
   금지)**: 해소되는 것은 §1-1의 **"3-플래그 묶음 처치" 교락뿐**
   이다. 귀속의 상한은 **"pdmux 서브시스템 전체"**(green-context
   SM 분할 + 전용 이벤트 루프 + split-prefill)이고, **"SM 분할
   자체"는 여전히 미분리**다 — `--enable-pdmux`가 A2의
   `event_loop_normal()`을 대체해 pdmux 전용 이벤트 루프를 통째로
   켜기 때문이다. **"PD 분리 자체가 원인"으로 승격 금지** —
   §1-1의 NOT-YET-SUPPORTED 등급은 불변, Gate 2 본 질문은 한
   눈금도 전진하지 않는다. ★**감사 provenance(등급 정정)**: 세션
   핸드오프(`handoff-report/session_handoff_2026-08-09.md` §2.5)는
   이 결과를 "[감사 완료]"로 표기했으나, 2026-08-09 저장소 전체
   검색(`gate2/` 디렉터리·전체 `*.md`)으로 이 결과를 다루는
   claims-auditor 감사 아티팩트가 확인되지 않는다. ⇒ **"감사
   완료"로 인용하지 않는다.** 현재 등급 = **메인 세션 원자료 독립
   재현 확인(2026-08-09)**, claims-auditor 감사 기록 위치 미확인
   (같은 캠페인의 primary R3′/R4′와 §11-3 감사 면제 사실은 §14.1
   addendum에 이미 기록돼 있고 그 부분은 인용 가능 — 감사 미확인은
   이 R1′/R2′ 비교 자체에 한정된다). 상세 `reports/CONSENSUS.md`
   rev17·§1-1(rev4 본 캠페인 블록, E-A 블록 앞)·§3 항목34, 원자료
   `workspace/engine-port/results/p1_gates/gate2/`. 아래 "방법론
   게이트" #20 신설. `CLAIM_EVIDENCE_MATRIX.md`는 대조 확인 결과
   이 항목을 인용한 서술이 없어 갱신 대상 없음(확인 완료).

   ★★★★★★**E-A(mixed-chunk 레버, jobs 875654/875657/875661,
   2026-08-08~09) — claims-auditor 감사 완료, 판정 조건부(문구 강한
   제한). 새 성능 판정 아님 — fused-측 조율 가능성에 대한 진단이다.**
   `--enable-mixed-chunk`는 Zamba2-2.7B·Granite-4.0-h-micro-base×
   {8192,512} 4콤보 전부에서 realized로 확인됐다(Stage 1, job 875654,
   3중 증인). E-A(jobs 875657/875661, n=10 paired, cudagraph-ON, arm
   순서 무작위)의 사전등록 primary 두 비교×4셀=**8건 전부 TOST 등가
   미발화·부호 전부 agnostic 우세**(10/10, 분포무가정 정확 p=0.00195)
   ⇒ **"조율된 fused가 pdmux를 대체한다"는 발화하지 않았고, 논문
   신규성 축 전환은 없다.** ⚠️8건 중 7건은 F-E(정상상태 검정) 발화라
   부호만 인용 가능. **유일한 인용 가능 정량치**(Granite rate3,
   cmp1, F-E clear·G-2 무관): X_60(plainmix) − X_60(agnostic) =
   **+0.855**[+0.814,+0.896], n=10, 부호 10/10 — 단 이 셀은 선행
   캠페인 진단을 본 뒤의 **사후 이동**(r4→r3)이며, r4를 유지했다면
   인용 가능 셀은 0개였다. **같은 셀에서 TTFT 항은 오히려 통과**한다
   (plainmix가 agnostic보다 TTFT p95 19.2% 좋음 — "TTFT 비열등 0건"은
   거짓, 1/8). **레버 격리**: +0.855 중 mixed-chunk 고유 기여분은
   **+0.010[+0.0045,+0.0155]=1.2%**뿐(4셀 범위 1.2–10.6%), 나머지
   98.8%는 rev4/§1-1이 이미 확립한 untuned-fused-vs-pdmux 격차
   (+0.845) ⇒ **E-A의 primary는 mixed-chunk 레버를 격리하지
   않는다** — X_60이 fused arm 전부 천장 근접(0.85–0.99)이라 고유
   효과는 하향편향, HOLB 관측자 효과 잔차와 같은 자릿수라 기전 해석
   금지. **기전(신규, 서빙 직접 측정)**: mixed-chunk ON 시 prefill과
   병합된 decode가 `ForwardMode.MIXED` extend 경로로 재라우팅되고
   (cudagraph 제외 경로), MIXED step 지속시간이 병합 decode 수에
   선형(Zamba2 ≈33ms/req, Granite ≈4.2ms/req) — backlog가
   `max_running_requests`·mamba usage를 포화시켜 admission을 막고
   TTFT가 발산한다(정본 §1 얽힘 死因의 **새 트리거**이지 새 기전
   아님). **cudagraph 가설은 반증**(순수 DECODE step은 양 arm 동일
   지속시간·cuda graph True, piecewise 전 arm OFF) — rev4 §1.1의
   "A3는 piecewise도 함께 바꾸는 ≥2-기전 묶음" 캐비어트는 **이 두
   모델에선 런타임 실측으로 무효**(인용 금지 해제). **G-2(재현성,
   correctness 아님)**: `--chunked-prefill-size 512` Granite가
   16-동시 greedy에서 비트단위 재현성 상실(16 중 2 요청, 반복 단위
   62↔322 토큰, 3회 독립 재현, 음성대조 4종 통과) — mamba state
   이월과는 불정합, correctness 결함 아님. ★**T2("fused 조율 공간
   소진") 주장 금지**: `reports/CONSENSUS.md:368`이 지목한 두 레버
   (`--chunked-prefill-size`·`--enable-mixed-chunk`)는 시험됐고 둘
   다 격차를 못 닫혔으나, 같은 기전 축의 **미시험 노브 최소 6개**
   (`--prefill-max-requests`·`--num-continuous-decode-steps`·
   `--chunked-prefill-size` 2048/4096·`--max-running-requests`·
   `--schedule-conservativeness`·`--mamba-scheduler-strategy`, realized
   값 실측)가 남아 "fused 조율 공간 소진"은 등재 금지. ★**Gate 2 본
   질문("PD 분리 자체" 귀속) 전진 없음**: A2(`plainaux`)를 뺐으므로
   §1-1의 NOT-YET-SUPPORTED는 불변, 875344/875346의 A2로 교차-job
   메우는 것도 사전등록이 명시 금지. **인용 금지 13건**(런길이 의존
   수치·F-E 발화 셀 크기·"TTFT 비열등 0건"·"조율 소진"류·"두 레버 모두
   실패"의 E-A 귀속·cmp2 Granite 무효·"잘못된 출력" 등)·**신규 방법론
   항목 3건**(primary뿐 아니라 모든 보고 블록에 게이트를 걸어라/과부하
   arm과 정상 arm 비교는 시스템 상수가 아니다/사후 지정 셀 이동은
   인용 가능성을 만들 수 있다, 아래 "방법론 게이트" #17–19)은
   `reports/CONSENSUS.md` §1-1(E-A 블록)·§3 항목31–33 전문 참조.
   **후속 게이트**(우선순위순): T4-1(deterministic-inference
   재확인)→T3-1/T3-2(MIXED 배치 조성 계측·micro 스윕)→E-C(등지속가능
   -rate 대조, T1 정식 종결)→E-D(미시험 fused 레버 스윕, T2 종결)→
   T3-3(backend 교차)→T4-2(비퇴화 프롬프트 G-2). ⚠️상위 사전등록
   `PREREG_GATE2_2026-08-06.md`는 **워킹트리 미커밋 수정 상태**(§14
   addendum) — `reports/CONSENSUS.md` §4 "살아있는 문서" 표에 기록됨.
   ★**정정(2026-08-09, 위 최신 항목)**: 이후 커밋 `c47fad0`으로
   반영돼 워킹트리 클린 확인(추적 파일 수정 0건), 이 문구는 stale
   — §11-3 감사 면제 상태(§14.1)는 별개로 불변.
   전문 `reports/CONSENSUS.md` rev16·§1-1(E-A 블록)·§3 항목31–33,
   원자료 `workspace/engine-port/results/p1_gates/gate2/`
   (`PREREG_G2EA_2026-08-07.md`·`g2ea_report_*.json`·`g2earun_8756{57,61}.
   out`·`g2eaprobe_875654*`, 수정 금지·인용만).

   ★★★**(2026-08-09, 두 번째 정본 반영 건 — A/B/C, 전부 새 성능
   판정 아님)**

   **A. HOLB G5 재채점**(result-analyst, jobs
   874601/874602/874632/874633/874635, 원자료 `workspace/
   engine-port/results/p1_gates/gate2/g2holb_g5_tost_rescore_
   2026-08-09.json` 72셀 전량, 3 job × 4 arm × 6 응답변수, n=5
   paired): **판정 G5 = 미결정(UNDETERMINED), 저장된 `G5=False`를
   대체한다.** 3% 초과 통계적 지지 셀 **0/72**(미보정 최소
   p_exceed=0.138, job별 Holm 후 최소 조정 p=1.000 3 job 전부),
   등가 입증 **29/72**, 검정력 부족 **43/72** — G5는 프로브의
   관측자 효과가 3%를 넘음을 입증하지 못했고 3% 이내임도 입증하지
   못했다. 저장된 `G5=False`는 귀무-채택형 연언(`PASS = |효과|<3%
   ∧ CI∋0`)의 실패를 기록한 것이지 프로브 유해성의 입증이 아니다.
   **구 규칙이 노이즈를 보상했다**: 874635 agnostic ttft_p95 =
   **−0.77%±29.14%**, 95% CI [−36.95,+35.41] → 구 규칙 **PASS**
   (메인 세션 독립 재확인). 역방향도 발생 — 구 규칙 FAIL인데 ±3%
   등가가 실제로 입증된 셀 **4개**(예: 874633 plain itl_p95
   −1.15%±0.72%, p_TOST=0.0023; 나머지는 874633 plainaux/itl_p50·
   plainaux/mean_e2e_ms, 874635 plainaux/request_throughput).
   **설계 층(전향적 제약)**: n=5·δ=3%·α=0.05/side에서 TOST 발화
   산술 천장은 SD<3.147%인데 **72셀 중 35셀(49%)**이 그 위에
   있었다. 80% 검정력 필요 n 중앙값 **9**(변수별 request_throughput
   4 / itl_p50·mean_e2e_ms 7 / itl_p95 22 / ttft_p50 29 /
   **ttft_p95 89**, ⚠️SD가 df=4 추정이라 필요 n은 자릿수 수준
   의미만). **금지 유지**: `PREREG_GATE2` §14.2의 "프로브가
   무해함이 입증됐다고 쓰지 마라"는 **해제되지 않는다** — 동시에
   반대 오독("3% 넘게 유해함이 입증됐다")도 근거 없음이 확정됐다.
   §14.3의 귀인("주로 점추정이 3%를 넘는 조합이 실재하기 때문")은
   **부분적으로만 참**(그런 셀은 실재하나 그중 하나도 초과가
   지지되지 않는다) — 두 문장 병기 필수. **잔여 교락**: 874602
   agnostic은 5/5 rep 전부 `order='off on'`(무작위화 불균형) ⇒
   그 arm의 등가 판정은 조건부. 등가 판정 29건은 전부 paired-t
   **정규 가정** 위(n=5 분포무가정 두측 p 하한 2/32=0.0625).

   **B. job 874601의 `G3=False` 라벨 정정**(메인 세션 원자료 직접
   확인): `g2holb_report_zamba2_874601.json`은 4 arm 전부
   `result:"FAIL"`이나 `sha_off:null, sha_on:null`·
   `self_repro_off/on:true`다 — 출력이 갈라진 게 아니라 sha
   추출이 응답 스키마를 못 읽은 것(`KeyError:'text'`)이고 Phase B가
   실행되지 않았다. 재실행 874633/874635는 `method_off/on:
   "output_ids"`로 sha 양측 일치 → **PASS**(위 A절 근거로 이미
   사용됨). 874632는 Phase A 중 SLURM CANCELLED(데이터 없음). ⇒
   "874601 G3 실패" 라벨은 **하네스 실패**로 정정한다. **이
   프로젝트 서명 오류(측정 실패를 게이트 실패로 라벨링)의 여섯
   번째 재발**이다 — 핸드오프 2026-08-09 §1이 다섯 번(텔레메트리
   드롭 카운터 자기검열 / `UNSCOREABLE`을 강등으로 읽음 / HTTP
   400을 G3 FAIL로 / G5의 귀무-채택형 기준(위 A절과 같은 사건
   계열) / 프로브의 stderr 오염)으로 셌다. **canon에 이 패턴을
   다루는 기존 번호가 없어(전수 검색 확인) `CONSENSUS.md` §3
   항목35로 신규 등재**(중복 신설 아님 — 이후 재발은 이 항목
   카운트만 갱신).

   **C. E-A 커밋 산출물의 scipy 부재 폴백**(engine-porter 발견 +
   메인 세션 노출범위 실측, 범위 한정): `g2ea_report_*.json`은
   scipy 없는 인터프리터에서 생성돼 `t_cdf()`가 Student-t가 아니라
   정규 CDF로 조용히 폴백했다(저장된 `p_tost`가 정확히 1.0인 이유,
   `g2ea_analyze.py:84-87,156-159`). **메인 세션이 노출 범위를
   직접 측정**: `raw_ci` 폭에서 역산한 임계값이 **8개 비교
   전부**(2 job × 2 rate × 2 cmp, 전부 df=9) implied_t =
   **2.2621… = t(.975, df=9)**(하드코드 표값과 일치, 1.96 아님) —
   **CI(raw_ci)는 오염되지 않았다.** 노출은 (i) `p_tost` 값
   자체(이 캠페인은 관측치가 0.05 경계에서 멀어 판정에 영향
   없음)와 (ii) `t_ppf`의 df>10·표에 없는 p에 한정된다(`t_cdf`는
   표가 아예 없어 scipy 부재 시 모든 df에서 근사값이라는 점은
   (i)에 포함되되 원인 층이 다름을 기록). ⇒ **등재 방식 = "정본
   수치 정정"이 아니라 도구 규율 항목**(게이트 #14 계열 — 통계
   라이브러리의 조용한 폴백은 아티팩트에 기록되지 않는다; 분석
   재현 시 인터프리터 환경을 아티팩트에 남겨라), `CONSENSUS.md`
   §3 항목36 신설. **과장 금지 — 이 캠페인에서 오염된 인용 수치는
   없다.**

   상세 `reports/CONSENSUS.md` rev18·§1-1(A/B/C 블록)·§3 항목35·36,
   원자료 `workspace/engine-port/results/p1_gates/gate2/`
   (`g2holb_g5_tost_rescore_2026-08-09.json`·`g2holb_g5_tost_rescore.py`·
   `g2holb_g5_tost_summary.py`·`g2holb_report_zamba2_874601.json`,
   수정 금지·인용만). 아래 "방법론 게이트" #21·#22 신설.
   `CLAIM_EVIDENCE_MATRIX.md`는 대조 확인 결과 이 항목을 인용한
   서술이 없어 갱신 대상 없음(확인 완료).

   ★★★**(2026-08-10, 세 번째 정본 반영 건 — A/B/C/D, 전부 새 성능
   판정 아님)**

   **A. 방법론 게이트 #21의 일곱 번째 재발 — 이번엔 코드가 거짓
   음성을 냈다**(engine-porter 발견, 메인 세션 사건 경위 확인):
   job 876699(T4-1)가 `--time=1:00:00`에서 TIMEOUT됐다. 원인은
   사이징이 아니라 **단일 호출 스톨**이다 — 9개 부팅 중 7개가
   ~8분에 정상 완료됐고, `ON + chunk512`의 첫 실제 multi-chunk
   generate(2552 토큰, 5-chunk)가 **~52분간 무응답**(스케줄러
   로그 0줄, CUDA 에러·watchdog 미발화 — 서버 로그 마지막 활동
   23:34:42, SLURM TIME LIMIT kill 00:26:19, 메인 세션이 두 시각을
   직접 대조해 확인)했다. 같은 job의 `OFF + chunk512`는 동일
   프롬프트를 동일 경로로 ~1초에 완료했다. **핵심**: 그 상태의
   아티팩트에 대해 옛 `g2det_analyze.py`가 **`REFUTED — reduction-
   order is not the (sole) cause`를 반환하고 있었다** — ON
   chunk512의 `n_total=0`(arm 미실행)인데 `on_clean = n_total > 0
   and ...`이 False로 떨어져 REFUTED로 통과한 것이다. **52분짜리
   멈춤이 실질적 음성 결과로 발표될 뻔했다.** 이것이 재발 1–6과
   다른 점: 1–6은 라벨·해석 오류였고 사람이 문서에서 잡았다.
   이번은 **분석 코드가 거짓 음성 판정을 산출**했고, 막은 것은
   도구가 아니라 experiment-runner가 `INCOMPLETE`로 보고하며
   사전등록이 열거하지 않은 조건이라고 채점을 거부한 **실행자의
   규율**이었다 — 도구가 사람보다 관대했다. 사후 완화가 아니다:
   사전등록의 REFUTED 조건은 "ON이 어느 rep에서든 self-mismatch
   ≥1"인데 rep이 0개면 그런 관측 자체가 없다 — 옛 코드는 사전등록
   규칙이 아니라 **버그**였다. 수정 후 `NO VERDICT (MEASUREMENT
   ABSENT)`를 반환하고, 데이터가 있을 때의 CONFIRMED/REFUTED는
   5-케이스 매트릭스로 불변 확인됐다. ⇒ `CONSENSUS.md` §3
   항목35(방법론 게이트 #21)의 **재발 카운트를 6→7로 갱신**한다
   (신규 항목 아님).

   **B. 공유 하네스의 무한 대기**(도구 규율): `g2_holb_phaseA_lib.sh`
   의 `greedy_call`이 **`--max-time` 없는 raw curl**이었고, 이
   디렉터리의 **모든 캠페인**(g2ctrl/g2ea/g2holb/g2det)이 이 경로를
   쓴다. 소비자 10개를 전수 확인한 결과 **4개가 타임아웃을
   `FAIL`/`SMOKE_FAIL` 계열로 채점**하고 있었다(A의 `g2det_
   analyze.py` 포함). 수정: `--connect-timeout 10 --max-time
   180`(env 재정의 가능), `STATUS=TIMEOUT`을 `STATUS=ERROR`와
   구별, **SHA를 아예 방출하지 않아** mismatch 채점이 구조적으로
   불가능, 사이드카 `.status.json`. 기본값 180s는 **측정 근거**로
   정당화됐다 — 이 디렉터리 아카이브 응답 **n=64**의 서버측
   `e2e_latency`가 median 0.978s / p90 2.534s / **max 6.605s**
   (최악 관측의 27배)다. 정상 경로는 아카이브 64개 + 합성 실패
   9종 재생으로 **73/73 byte-identical** 검증됐다. ⇒
   `CONSENSUS.md` §3 항목37·방법론 게이트 #23으로 신설(아래).

   **C. Gate 2-S 3라운드 설계 감사의 구조적 결론 — §1-1 귀속 스코프
   주석(대체 아님)**: `PREREG_GATE2S_2026-08-09.md`가 claims-auditor
   감사 **3회 전부 NO-GO**를 받았다(rev1 통계층 / rev2 게이트
   인식론 / rev3 귀무 채택형). **3연속이 같은 자리(§5.5 앵커 발화
   조건)에서 죽었고**, 감사자가 근본 원인을 **문서 내부 모순**으로
   특정했다: §0.1이 "두 pdmux arm 사이 등가 마진에는 외부 앵커가
   없다"고 이미 확립했는데 §5.5는 정확히 그 등가를 앵커 조건으로
   요구한다. 정량: rev3 §5.3의 **암묵 등가 마진 = 0.715 σ_D**, 이
   설계의 **primary MDE = 0.995 σ_D** ⇒ 대리 허용오차가 검출한계의
   **0.72배**. 표준 처방(TOST+사전등록 마진)은 §0.1 정면 위반이고,
   마진을 MDE의 1/4로 낮추려면 **n≈64**(현재 10)가 필요하다. ⇒
   **정본에 기록할 명제(문구를 정확히 지킨다)**:

   > **"P1 이득 중 'SM 분할 자체의 몫'을 두 pdmux arm 사이의
   > 성능-층 등가검정으로 귀속하는 경로는, 이 프로젝트의 예산
   > 범위에서 닫히지 않는다**(n≈64 필요, 현행 설계 n=10). 원리적
   > 불가능이 아니라 **이 경로·이 예산에서의 불가능이다.**"

   ⚠️**과장 금지**: "귀속이 원리적으로 불가능하다"로 쓰지 않는다.
   이것은 **미실행 사전등록에 대한 설계 감사**이므로 측정 결과가
   아니다 — 등급어를 그에 맞춘다. **이 문장은 아래 §1-1의
   NOT-YET-SUPPORTED를 대체하지 않고 스코프 주석으로 덧붙는다**
   ("아직 안 됐다"와 "이 경로로는 안 된다"는 다른 진술). **감사가
   깨뜨리려 시도했으나 실패한 것(견고한 것)**: arm 구조·Δ_split
   estimand의 내부 타당성, T·C 드레인 대칭, 9-셀 판정표의 저자
   불리 셀 실재, 부팅 단위 프로브 근거, 예산 산술. 감사자 원문:
   **"이 캠페인이 죽어야 할 이유는 측정 층이 아니라 귀속 층에만
   있다."** rev4는 §5.5 앵커 발화 조건을 제거해 이 하위목표에서
   후퇴했다(패치 0줄, env 조합만 재구성) — **Gate 2-S가 폐기된
   것은 아니다.**

   **D. 코드 사실 문구 하향**(메인 세션 자기정정): 메인 세션이 이
   세션 중 "`(0,108)` idx에서는 **두 역할 모두 전체 108 SM에
   접근한다**"고 서술했다. **검증된 것은 "green context가 아니라
   평범한 `torch.cuda.Stream` 쌍이다"까지**이고(`pdmux_context.py:
   124-138`, 직접 확인 — idx 0과 idx `len-1`은 `torch.cuda.
   Stream(gpu_id)`, 중간 division만 `create_greenctx_stream_by_
   value`), green ctx 생성이 primary context SM을 깎는지는
   **미측정 물리 명제**다(격리 측정 수단이던 P-b 프로브는 공선성
   때문에 삭제됨). **전수 검색(2026-08-10) 결과 이 문구는 canon이나
   파생 문서 어디에도 들어가지 않았다** — 정정 대상 문구는 없다.
   향후 인용 규칙으로만 등재: (0,108) idx는 **"명시적 분할이
   적용되지 않는다(잔여 차감 여부는 미측정)"로만 쓴다**(과장형
   금지).

   상세 `reports/CONSENSUS.md` rev19·§1-1(2026-08-10 세 번째 정본
   반영 건 A/B/C/D 블록)·§3 항목35(개정)·37, 원자료
   `workspace/engine-port/results/p1_gates/gate2/`(`g2det_876699.out`·
   `g2det_876699.err`·`g2det_analyze.py`·`g2_holb_phaseA_lib.sh`·
   `PREREG_GATE2S_2026-08-09.md`, 수정 금지·인용만). 아래 "방법론
   게이트" #21(개정)·#23 신설. `CLAIM_EVIDENCE_MATRIX.md`는 대조
   확인 결과 이 항목을 인용한 서술이 없어 갱신 대상 없음(확인
   완료).

   ★★★★★★★**(2026-08-11) Gate 2-S 첫 유효 결과(jobs 877756/877757,
   6.35 GPU-hr) — claims-auditor 적대 감사 "조건부 등재 가(可)".
   별도 트랙(Gate 2-S)의 산출이며 §1-1의 NOT-YET-SUPPORTED를
   대체하지 않는다(위 2026-08-10 "귀속 스코프" 주석과 정합하게
   배치).** 둘 다 exit 0/`MEASURED_AND_SCORED`,
   `design_conformance.conformant=True`(전 셀 n=10, 양 rate, 5
   arm), 19/19 블록. **이력**: 1차 실행(jobs 877107/877109, 6.40
   GPU-hr)은 하네스 결함 6건으로 사전등록 primary를 **0개** 냈다 —
   설계는 무결했고 실패는 전부 하네스 층이었다. 사전등록
   `PREREG_GATE2S_2026-08-09.md`(rev6)는 claims-auditor 설계 감사
   **4회**(NO-GO 3 → GO-with-changes)를 거쳤다. 채점
   result-analyst(원자료 200행 독립 재계산), 감사 claims-auditor
   (원자료에서 α 독립 재현, 소수 3자리 일치 — 메인 세션도
   g2s_report_*.json 원자료로 핵심 수치를 추가 독립 확인, 전용 감사
   아티팩트 파일 위치는 미확인).

   ★**정본 문장(감사자가 확정한 허용 범위, 축약·확장 금지)**: 이
   엔진·이 격자(Zamba2-2.7B r{2,3} triton / Granite-4.0-h-micro-base
   r{3,4} flashinfer, in2000/out96, A100 108 SM, cudagraph ON,
   n=10 paired, jobs 877756/877757)에서, pdmux 서브시스템·이벤트
   루프·split-prefill·역할별 backend를 고정한 채
   `PDMUX_R2_FIXED_DSM`만 108→34로 바꿔 SM을 `(74,34)`로 분할하면,
   **요청-내부 ITL p95의 평균(α)이 4셀 전부에서 감소**하고(Δ =
   C−T′ = **+13.95/+24.88/+16.04/+19.11 ms**, 4셀 Holm 후 **최대
   보정 p = 1.88e-06**, β=pooled p99도 부호·유의성 일치) **같은
   4셀에서 TTFT p95는 악화**한다(−183 ~ −536 ms) — 9-셀 표 좌표는
   16블록 전부 **`S1-C`**(트레이드오프)다. Zamba2 r2·r3는
   "`FixedPolicy(34)`·sticky OFF = 엔진 기본 궤적" 명명이 허용되고
   (전제 검증됨, Gate 1 rev15/job 875293), **Granite r3·r4는 그
   명명이 금지**되어 "`FixedPolicy(34)`·sticky OFF 구성"으로만
   부른다(전제 미검증, G1-c 미실행 — in-job 단방향 반증기는 4셀
   `unchanged`이며 **이는 검증이 아니다**). **크기 인용이 허용된
   셀은 F-계열 미발화 셀 Zamba2 r2 하나뿐**이고 그 값은 α
   **−38.8%**[−41.6,−36.0] · TTFT p95 **+51.5%**[+35.8,+67.2]다
   (부호 규약 = T′ 대 C, 음수 = 분할 우수). ★**이 효과는 분포의
   꼬리에 한정된다** — 같은 대비에서 요청-내부 **q≤0.7 분위수는
   4셀 전부 반대 부호로 유의**하며(q0.5: −0.85/−2.58/−0.91/−1.68
   ms), 기전은 "드문 대형 prefill 유발 decode 스톨을 균일한 소폭
   decode 지연으로 교환"이다. ★**처치는 decode-active 시간의
   35–40%만 실현**됐고(T′ idx1 frac 0.355–0.405), **100% 실현
   arm(T, sticky ON)의 α 효과는 오히려 더 작다** ⇒ **어떤 dose
   외삽도 금지**. "무분할(C)"은 **명시적 분할 부재**를 뜻하며
   잔여 차감은 미측정이다(Zamba P-carve NOT-EQUIVALENT = 등가
   미확립이며, 차감이 있더라도 C·T′가 같은 yml·같은 green
   context로 부팅하므로 이 대비에서는 공통항으로 소거된다). 이
   결과를 **§1-1의 A4-vs-fused 격차의 성분·기여분·분해로
   서술하지 않으며**, "PD 분리 자체가 원인"으로 승격하지 않고,
   **SLO goodput·용량·fused 대비 우열에 대해서는 아무것도 말하지
   않는다**(같은 job에서 미조율 fused A2가 요청 p95 ITL에서 C를
   이기는 셀이 있고 TTFT p95는 4셀 전부 A2가 우수하다 — 부호만,
   크기 인용 금지). "SM 분할만 켠 fused arm"은 이 엔진에서 구성할
   수 없다.

   ⚠️**함께 박을 금지 6건**: ① `component_share` 인용 금지(Granite
   r3 Fieller **3.098**="몫 310%"가 아티팩트에 있으나 코드 자신이
   "항등식, 결과 아님"이라 표시). ② **두 report json의
   `headline.text` 문자열 인용 금지**(m=2 Holm으로 계산한 것을
   "4셀 Holm 후"라 인쇄 = 거짓 provenance — 메인 세션이 `holm`
   필드 `size_this_job=2` vs 텍스트 "4셀"을 직접 대조해 확인). ③
   "분할이 ITL을 개선한다"를 **꼬리 한정 없이** 쓰기 금지. ④
   Δ^cont(secondary)를 헤드라인으로 승격 금지(T는 mean e2e가
   최대 **+41%** 악화). ⑤ **T′의 F-B disjunct (i) 미평가**(§8.6)
   병기. ⑥ **인용 셀이 지정 셀이 아니라 구조적 저용량 복제 셀**
   이며 그 선별 필터의 **귀무 발화율이 40.1%**(1−0.95¹⁰, 메인
   세션 재계산 일치)임을 병기.

   **신규 방법론 항목**(전부 새 성능 판정 아님): (i) `CONSENSUS.md`
   §3 항목18/방법론 게이트#9("게이트 자신이 항등식") **새 사례**
   — §4.5의 두 안전장치가 같은 3개 꼬리 통계량만 보고,
   `CONVENTION-SENSITIVE`는 12지표 중 9개 비트 동일. (ii) ★§3
   항목34/방법론 게이트#20 **새 사례**("식별자 수입 ≠ 거동 수입")
   — `g2s_analyze.py:63`의 `MIN_COVERAGE=0.98`이
   `gate1b_analyze.py:49`에서 상수만 수입되고 규칙(원본 `:250`)은
   미구현(방향은 저자 불리, 4셀 전부 253–427 에피소드라 수치
   결과는 없음). (iii) 신규 §3 항목38/방법론 게이트#24: `any()`
   over n reps 스크린은 귀무 발화율이 1−(1−α)ⁿ(여기선 40.1%). (iv)
   신규 §3 항목39/방법론 게이트#25: §6.3-6 관측자 대칭 보고
   (`dropped_events`/`writer_error`)가 산출되지 않았다(감사자
   오프라인 복구, 전부 0 확인).

   ⚠️**등재 불가 항목**: e2e/concurrency 수치, Granite 전제의
   "검증됨" 승격, dose 보정치, goodput 해석. 상세
   `reports/CONSENSUS.md` rev20·§1-1(2026-08-11 Gate 2-S 결과
   블록)·§3 항목18·34(개정)·38·39, 원자료 `workspace/engine-port/
   results/p1_gates/gate2/g2s_report_zamba2-27b_877756.json`·
   `g2s_report_granite-40-h-micro-base_877757.json`(수정 금지·
   인용만). 아래 "방법론 게이트" #9·#20(개정)·#24·#25 신설.
   `CLAIM_EVIDENCE_MATRIX.md`는 대조 확인 결과 이 항목을 인용한
   서술이 없어 갱신 대상 없음(확인 완료).

   ★★★★★★★★**(2026-08-11, G1-c, job 877974, 0.10 GPU-hr) Gate 2-S
   Granite r3·r4 명명 제한 조건부 해제 — 새 성능 판정 아님, 크기
   인용 셀 확대 아님.** G1-c(Gate 1/G1-b의 Granite 자매 job, 873945
   격자 {2,3,4,6} 전부 재현, `PDMUX_TRACE_FORCE_PREFILL=1`·
   `TRACE_EVERY=32`, 원자료 `results/p1_gates/gate1/gate1c_result_
   877974.txt`)가 §8.9 전제 판정 규칙(frac((54,54))(pop A 시간가중)
   `<0.01` ∧ max(decode_bs)`<36` ⇒ VERIFIED)을 rate 3·4 양쪽에서
   발화시켰다: max(decode_bs)=10/10(36 미만)·frac_5454=0.0000/
   0.0000, 구조적 근항등식 양성대조(pop_A_time_weighted_frac
   (TARGET_A))=1.0000이 4 rate 전부 PASS, grid-completeness 결측
   0/expected. 트리 근거는 `results/p1_gates/gate2/
   runtime_source_manifest_s_granite-40-h-micro-base_877757.sha256`
   와 15/15 바이트 동일(diff 확인, 메인 세션 재확인) — 873945
   상속(G3, 재검증 안 함) 근거뿐 아니라 877757 자신과의 직접
   동일성으로 강화됨.

   ⇒ 해제되는 것은 `PREREG_GATE2S_2026-08-09.md` §5.6.1
   `name_for()`의 **명명 층 하나뿐**이다: Granite r3·r4에서 "간헐
   전달(엔진 기본 궤적)"·"A4형" 명명이 허용된다. **크기 인용
   자격은 불변**(코드 확인, `g2s_analyze.py:1157-1161` —
   `premise` 필드는 `nine_cell`·`gate_label`(F-계열) 산출에
   입력되지 않고 결과 dict에 나란히 기록만 됨) — Granite r3는 T′
   F-계열 발화, r4는 C·T′ F-계열 발화 상태라 **`SIGN ONLY,
   MAGNITUDE NOT CITABLE`이 유지**된다 ⇒ **크기 인용 가능 셀은
   여전히 Zamba2 r2 하나뿐**(세션 초반 "1→3개로 는다" 서술은
   원자료로 반증, 과장 정정).

   **조건부 해제 필수조건 6건(하나라도 누락 시 해제 무효)**: ①
   명명 층 한정(부호 서술·헤드라인 명명에만, 크기 인용 불변) ②
   Granite r3·r4 인용마다 F-계열 gate 병기(`F-SERIES FIRED ⇒
   SIGN ONLY, MAGNITUDE NOT CITABLE`, r3=T′·r4=C·T′) ③ 경험적
   내용은 하나 — "동거 구간 realized max(decode_bs)=10(r3)/10(r4)
   `<`문턱 36"이고 frac((54,54))=0은 그 **코드 귀결**
   (`multiplexing_mixin.py:900-909`: idx==2 ⟺ decode_bs≥36, 항등식)
   이지 독립 증거가 아니므로 두 수치를 독립 증거처럼 병기 금지 ④
   증거 등급 명시(`job 877974, n=1, 873945 격자 복제,
   PDMUX_TRACE_FORCE_PREFILL=1[관측 밀도 32배, 관측자 부하 상한
   없음 — G5 UNDETERMINED], Gate 2-S 셀에서의 직접 관측 아님,
   selector-level·S3 미실행`) ⑤ 트리 근거는 877757 기준(위 문단) —
   "873945와 바이트 동일 아님(G3 상속)"만 쓰면 오도 ⑥ 아래 Zamba2
   근거표 동시 정정.

   **pop A `t_total_s` 인용 금지(신규, result-analyst)**: 경계
   구간 dt가 interior의 11–18배라 5–29% 상향 편향 — G1-c
   사전등록의 "A/B 그대로 인용 가능" 허용은 과대 허용이었다.
   정직한 bracket(참고용, 크기 결론 아님): r2 [11.07,15.57]s·r3
   [15.55,19.10]s·r4 [16.24,19.71]s·r6 [17.26,18.17]s. pop C
   절대·상대 분수는 종전대로 인용 금지.

   ⚠️**해제 후에도 금지인 문장 8건**: ①"크기 인용 가능 셀이
   3개가 됐다"/Granite r3·r4의 Δ·CI·%·β 크기 인용 ②"전제가
   검증됐으므로 4셀이 동질적이다"(F-계열·검정력 축 비동질성 유지)
   ③"A4와 T′는 동일하다"(확립된 것은 인덱스 사상 동일 +
   decode_bs<36 관측뿐, A4는 `_slo_on=False`라 EMA·
   `controller_decision`·`_r2_decide_idx` 경로 자체가 없다)
   ④"Granite에서 (54,54)는 도달 불가"(부하가 커지면 도달 —
   Zamba2 r6에서 실제 발생, 게이트#16 거울상) ⑤"엔진 기본
   궤적=물리적으로 prefill 74 SM/decode 34 SM"(selector 라벨,
   잔여 차감 미측정, S3 미실행) ⑥"frac_5454=0과
   max_decode_bs<36이 각각 전제를 지지한다"(항등식, ③ 재확인)
   ⑦877974의 TTFT/TPOT/ITL/goodput 어떤 수치도 인용 금지(n=1,
   진단 전용) ⑧"Granite에서는 G1-b식 철회가 일어나지 않는다"의
   무제한 서술(n=1·이 격자·이 워크로드 스코프 필수, 게이트#16
   거울상).

   ★**성능 결론 불변**: `premise` 라벨은 Δ·paired t CI·Holm·
   9-셀·F-계열 gate 산출에 미입력(코드 확인) — 위 rev20의
   Δ=+13.95/+24.88/+16.04/+19.11ms(4셀 Holm 후 최대
   p=1.88e-06)·9-셀 좌표 전부 `S1-C`·크기 인용 허용 셀 Zamba2 r2
   하나 문장은 **한 글자도 바뀌지 않는다**.

   ★**`PREREG_GATE2S_2026-08-09.md` §8.9 정오표(사후 addendum,
   원문 미덮어쓰기)**: 기존 §8.9 표의 Zamba2 r2·r3 "전제 상태"
   근거란이 **pop-C 시간가중 분수(비동거 구간 `(0,108)` 98%/99%)를
   인용**하고 있었는데, 이 양은 `gate1/PREREG_G1B_2026-08-07.md:
   146`이 few-snapshot dt 팽창을 이유로 **명시적으로 인용
   금지**한 것이다(Gate 1 감사 확정) — Zamba2·Granite 양 모델
   공통으로 §8.9 표의 근거란을 "**max(decode_bs)<36**" 하나로
   좁힌다(보수적 방향; "Granite 기준을 낮춰 맞춘 것"이 아니라
   기존 Zamba2 표기가 과다 인용이었던 것의 정정). load-bearing
   아님(비동거 분기 `multiplexing_mixin.py:911-912,923`는
   decode_bs 비의존 코드 항등식이라 A4·T′ 공통이므로 이 정정으로
   §8.9의 판정 자체는 바뀌지 않음).

   정본 반영: `reports/CONSENSUS.md` rev22·§1-1(이 블록)·§3
   항목41(신규, 결정량의 밀도 의존성)·항목42(신규, 실험 payoff는
   코드로 검증 후 정당화)·항목43(신규, `compute_coverage`류는
   내부 구멍에 맹목). 아래 "방법론 게이트" #27–29 신설.
   `CLAIM_EVIDENCE_MATRIX.md`는 대조 확인 결과 이 항목을 인용한
   서술이 없어 갱신 대상 없음(확인 완료). 원자료 `workspace/
   engine-port/results/p1_gates/gate1/{PREREG_G1C_2026-08-11.md,
   gate1c_result_877974.txt, gate1c_analyze.py,
   runtime_source_manifest_gate1c_877974.sha256,
   manifest_diff_gate1c_877974.txt}`(수정 금지·인용만).

   ★★★★★★★**(2026-08-11, Gate 2-S E1 addendum, jobs 877756/877757
   재집계, GPU 증분 0, claims-auditor 적대 감사 완료) E1 채택 — 새
   성능 판정 아님, 등급 하향된 조건부 채택(4셀 전부).** E1(사전등록
   `PREREG_G2S_E1_ADDENDUM_2026-08-11.md`, 구현 `g2s_e1_premise.py`,
   결과 `g2s_e1_premise_877756_877757.json`)이 §8.9 전제를 Gate 2-S
   자신의 셀에서 n=10 직접 산출한 pooled `max(decode_running_
   batch_size)`(threshold 36, `gate1b_analyze.py:135-137` 동사
   수입, 새 자유모수 0)로 재검증했다: Zamba2 r2=**14**(여유 22)·
   r3=**23**(여유 13)·Granite r3=**10**(여유 26)·r4=**13**(여유
   23), 4셀 전부 무결성 I1–I7 통과·`VERIFIED_AT_SAMPLED_INSTANTS`
   발화.

   ★**이번 반영은 승격이 아니라 등급 하향된 조건부 채택이다 — E1이
   §2.5에서 스스로 주장한 "G1-b/G1-c 대비 밀도 페널티를 같은
   크기로 받지 않는다/이것이 이 addendum이 존재할 수 있는
   이유다"는 claims-auditor 감사로 반증됐다.** 채택은 **조건 7개
   전부**(하나라도 누락 시 채택 무효) 하에서만 유효:
   ① 상태명은 항상 `VERIFIED_AT_SAMPLED_INSTANTS`(축약형 "VERIFIED"
   단독·"전제가 검증됐다" 단독 서술 금지).
   ② **밀도 서술 정정 의무** — E1 인용 시마다 다음을 병기: *"E1의
   우위는 반복수(n=1→10)와 셀 일치이며, 결정 관련 pop-A 관측 수는
   G1-b/G1-c 대비 5–6× 적다(1,279·1,496 vs 6,402·8,920)."*
   prereg §2.5의 해당 서술은 **인용 금지**(과소 서술, 반증됨).
   ③ **bound 병기 의무** — `sup decode_bs ≤ in-system + Poisson
   (λ·dt)`(새 자유모수 0, claims-auditor 산출) q=1e-9에서 Zamba2
   r2/r3·Granite r3/r4 순으로 **25 / 41 / 24 / 27**을 셀별
   병기한다. **Zamba2 r3는 가장 적대적 bound에서 문턱을 배제하지
   못하는 유일한 셀**임을 명시(q=1e-6에서 이미 **37 ≥ 36**).
   ④ **복합 규칙 분할 명시** — 수입 규칙은 `frac(idx2)<0.01 AND
   max(decode_bs)<36`인데 E1은 앞 절반을 §8.9.1에 위임하고 뒤
   절반만 계산했다 — 두 절반의 출처를 모두 적는다(감사자 독립
   확인: 4셀 전부 `stream_index==2` 0건).
   ⑤ **post-hoc 자백을 4셀 전부에** — 작성자가 이미 G1-b의
   Zamba2 9/23을 알고 있었으므로 Zamba2를 prospective로 표기하지
   않는다.
   ⑥ **§5.6.1 명명·F-계열·크기 인용 자격 불변** 재확인 문구
   동반 — `premise`는 `nine_cell()`·`gate_label()`·`paired_t()`
   (`g2s_analyze.py:1153-1159`) 어디에도 입력되지 않는다(감사자
   코드 확인).
   ⑦ 아래 "성능 결론 불변"·"무료 대조"를 함께 적는다.

   **Zamba2 r3(margin 최소, 인용 시 강제 병기)**: "Zamba2 r3의
   여유는 13(문턱의 36%)으로 4셀 중 최소이며, in-system+도착률
   상한(q=1e-6)에서 37≥36이라 **샘플되지 않은 순간의 문턱 도달을
   배제하지 못하는 유일한 셀**이다. 같은 모델·같은 arm이 G1-b
   격자 rate 6에서 실제 초과했고(max 40, frac((54,54))=8.37%),
   이 캠페인 자신의 capscan(agnostic, rate 1–6)에서도 max
   31/in-system 40까지 올라간다. 문턱은 이 모델에서 도달 가능한
   영역에 있다." **강등하지 않는 이유**: 강등하면 §8.9(rev5
   이래)의 Zamba2 r3 "검증됨"이 먼저 무너진다(근거가 G1-b의 같은
   값 23·같은 여유 13·더 나쁜 n=1) — r3만 강등은 정본
   자기모순이다. 여유 기준의 정식 도입은 **별건 결정**이며 그
   결정의 첫 대상은 E1이 아니라 §8.9의 Zamba2 r3 명명 허가임을
   함께 기록한다.

   **rev22 필수조건④ 대체(감사자 제시안)**: ④ 증거 등급 = (i)
   `job 877974(G1-c), n=1, 873945 격자 복제, FORCE_PREFILL=1 —
   pop-A 밀도 높음(8,920)`; (ii) `job 877757(Gate 2-S 자신), n=10,
   그 셀·A4 직접 관측, FORCE_PREFILL=0 — pop-A 밀도 낮음
   (1,496=(i)의 1/6)`. **두 관측은 서로를 대체하지 않는다** —
   (i)은 밀도, (ii)는 반복수·셀 일치를 준다. 판정은
   `VERIFIED_AT_SAMPLED_INSTANTS` 등급이다. (ii)는 (i)의 값을
   Granite r4에서 10→13으로 **상향 정정**한다(n=1 max는 pooled
   max의 구조적 하한). selector-level·S3 미실행 불변. **구
   필수조건④("Gate 2-S 셀에서의 직접 관측이 없다")는 이제 금지
   문장으로 전환된다.**

   ★**성능 결론 불변**: Δ=+13.95/+24.88/+16.04/+19.11ms(4셀
   Holm 후 최대 p=1.88e-06)·9-셀 좌표 전부 `S1-C`·크기 인용
   허용 셀 Zamba2 r2 하나·TTFT p95 악화 −183~−536ms **전부
   불변**(코드 확인). **무료 대조(감사자 등재)**: 같은 셀에서
   T′의 pooled max(decode_bs)=13/23/10/12 vs A4 14/23/10/13
   (±1 이내 일치) — 전제가 의존하는 부하 동등성의 직접
   증거인데 addendum 자신은 산출하지 않았다.

   **금지 문장 추가 6건**(rev22의 금지 8건·필수조건 6건 전부
   유효, 추가분): ①"VERIFIED" 단독·"전제가 검증됐다" 단독
   ②"E1은 G1-c보다 밀도가 높다/밀도 페널티를 피했다"(반증됨)
   ③"이 캠페인 자신의 데이터가 G1-c를 대체한다"(밀도 vs
   반복수는 서로 다른 축) ④`max_decode_bs`를 부하·용량·동시성
   대리 지표로 사용(모델 간·rate 간 비교 금지 포함) ⑤"Gate 2-S
   셀에서의 직접 관측이 없다"(이제 금지) ⑥E1-b·E1-c 실행 전에
   "전제가 이 캠페인에서 검증됐다"고 쓰는 것.

   ★★**별건 발견(engine-porter 소관, 코드 수정 안 함)**:
   `g2s_analyze.py:84-89`의 `PREMISE_LABEL`이 Granite r3·r4를
   여전히 `"unverified"`로 하드코딩하고 있고, 아카이브된
   `g2s_report_granite-40-h-micro-base_877757.json`의
   `premise_labels`도 `unverified`다 — rev22/rev23은 문서
   층에서만 승격했으므로 원자료를 직접 읽는 인용자는 정본과
   반대되는 값을 얻는다. 처리: (i) 알려진 결함으로 등재 (ii)
   아카이브 아티팩트는 G1-c/E1 이전 상태의 동결 스냅샷임을
   정오표로 명시 (iii) 향후 재실행 시 갱신 요구사항으로 등재.
   스코어러는 재현성 보존을 위해 지금 패치하지 않는다.

   정본 반영: `reports/CONSENSUS.md` rev23·§1-1(이 블록)·§3
   항목44(신규, 경로 없음 재사용 경계)·항목45(신규, bound와
   점추정 구별)·항목18·39(재발 追記). 아래 "방법론 게이트"
   #30·#31 신설. 원자료 `workspace/engine-port/results/
   p1_gates/gate2/{PREREG_G2S_E1_ADDENDUM_2026-08-11.md,
   g2s_e1_premise.py, g2s_e1_premise_877756_877757.json}`
   (수정 금지·인용만).

   ★★★**(2026-08-14, E-3 realized SM count 프로브, GPU 비용 ≈0 —
   glogin01 무비용 + job 882374) 드라이버가 요청 SM 개수를 반올림
   없이 그대로 보고한다 — 새 성능 판정 아님, 위 Gate 1 블록의 인용
   금지는 그대로 유지된다.** 사전등록 `workspace/engine-port/
   results/smsplit_realized/PREREG_SMSPLIT_REALIZED_2026-08-14.md`
   (§0–§6 + addendum 1·2). `pdmux_context.divide_sm()`을 직접
   호출해 얻은 `(74,34)`·`(54,54)`·C2 스윕 5지점(92/16, 84/24,
   64/44, 54/54, 16/92) 전 7지점에서, `torch.ops.sgl_kernel.
   create_greenctx_stream_by_value`의 realized 반환값이 요청값과
   **정확히 일치**했다(Δ=0, `realized_sum=108` 전 대상,
   `n_returned=4`) — 로그인 노드(glogin01, `NVIDIA A100 80GB
   PCIe`)와 컴퓨트 노드(job 882374, gpu43, `NVIDIA
   A100-SXM4-80GB`) 두 하드웨어에서 레코드가 완전히 동일. 판정
   문자열 `REQUEST_EQUALS_DRIVER_REPORTED_PARTITION`. **허용
   문장**: "드라이버가 보고하는 green-context 파티션은 이 격자에서
   요청값과 일치하며 반올림이 없다(드라이버 자기보고 층)."
   **금지(전부 유지)**: 이것은 **드라이버 자기보고**이지 "실현
   파티션을 측정했다"/"prefill 74 SM·decode 34 SM에서 실행됐다"가
   아니다 — 위 Gate 1 블록의 인용 금지는 **그대로 유지**된다(S3
   하드웨어 실행 층·`%smid`의 SM id 집합 disjointness는 여전히
   미측정). 기존 캠페인 판정문(873944/874478/875293/877974/
   877756/877757)에 사후 부착 금지, Gate 2 본 질문("PD 분리 자체"
   귀속)은 한 눈금도 전진하지 않는다. **부수 함의**: `%smid` R0
   사전등록 §0.1의 payoff 항목 1(`g2s_analyze.py:1488`의
   `log(108.0/34.0)` 분모 정정)은 **정정 대상이 없음이
   확인**됐다(34는 요청값이자 드라이버 보고 realized 값). `%smid`
   F2의 "다섯 번째 세계"(disjoint ∧ ∪⊊D)는 **개수 층에서는
   관측되지 않는다**(`realized_sum=108` 전 대상, id 집합 층은
   여전히 미측정). ★**부수 발견(방법론 게이트 #32 새 사례로도
   등재)**: 이 프로브가 `glogin01`(A100 80GB **PCIe**)과 컴퓨트
   노드(A100 **SXM4**)의 하드웨어 SKU가 다름을 직접 관측해, 아래
   "8B decode-SM 민감도 측정 노트"의 트래픽·roofline 하드웨어
   정정(2026-08-14)의 결정적 근거가 됐다. 정본 `reports/
   CONSENSUS.md` rev26·§1-1(이 블록, E-3)·§3 항목46(追記), 원자료
   `workspace/engine-port/results/smsplit_realized/
   {smsplit_realized_glogin01_2026-08-14.json,
   smsplit_realized_882374.json}`(수정 금지·인용만).
2. 현재 A100/SGLang green-context substrate에서 layer-boundary resource
   switching은 sub-step drain과 synchronization을 일으켜 decode TPOT을 약
   `42→124 ms`로 악화시켰다. 최적화 후에도 약 `85 ms`였다.
3. decode starvation은 active sequence 체류시간과 shared running-batch capacity를
   증가시켜 prefill admission과 TTFT를 악화시킨다. 대표적으로 D16은 D24보다
   prefill SM이 많지만 TTFT가 `7.24 s` 대 `1.21 s`였다.
4. 적절한 static split은 workload와 context/load에 따라 이동한다.
5. 기존 single-worker SLO-aware/binding-first dynamic은 valid varying trace에서
   best static을 넘지 못했다.

KV congestion은 plausible mechanism이지만 기존 artifact에 구조화된 KV occupancy가
없어 아직 독립적인 causal claim이 아니다.

★**갱신(2026-08-02) — occupancy 데이터는 생겼으나 de-confound가 안 된 상태다.**
`results/s8_frontier/`(2026-08-01, 미감사)에서 처음으로 arm별 occupancy가
기록됐다. 그러나 hybrid arm에서 관측된 `kv_mamba_occupancy = 1.0000`은
**메모리 구속의 증거가 아니라 항등식**이다 — 이 캠페인의 설정
(`--disable-radix-cache` ∧ `--max-running-requests 48`)이
`sglang/srt/model_executor/model_runner_kv_cache_mixin.py:223-229`의 분기를
타서 `max_mamba_cache_size = max_running_requests`가 되므로 pool 크기 = cap
이고, batch가 cap에 닿으면 정의상 1.0이다(코드 사실, "방법론 게이트" #5).
⇒ **이 데이터로 hybrid에서 "메모리 vs 스케줄링" 중 무엇이 구속적인지 판정
불가.** 이 regime(ctx 4k ShareGPT)에서 지지되는 것은 좁다: **attention
KV(`kv_full_occupancy`)는 어느 arm에서도 구속 근처에 없었다** — T8 0.032 /
M8 0.145 / Ha8 0.279 / Hs8 0.011(2026-08-01 캠페인 관측치, **claims-auditor
미통과 = 인용 금지**). Claim C의 등급은 변경 없음(running-batch 경로 강함,
KV 경로 부분).

### 벡터1 (G2.0 short-ctx disjoint conflict-regime escape hatch) — CONFIRMED closure (scoped)

6. short-ctx band(아래 scope)에는 동적 제어가 이길 수 있는 disjoint-feasibility
   conflict regime(어떤 static도 두 phase를 동시에 못 커버하는 워크로드)이 **없다**.
   `reports/CONSENSUS.md` §5-8(c) 미결 갈래 (c)를 최종적으로 닫는다.

실험 계열(2026-07-24 실행 시작·2026-07-25 최종 판정): `reports/CONSENSUS.md`
§5-8(c)("충돌 regime 워크로드" — 동적이 이길 disjoint-feasibility escape hatch가
있는가)를 n≥4로 재검증하는 G2.0 short-ctx 스윕(Zamba2-2.7B). 1차 라운드
(g2_0_full/g2_0_hard)는 **ILL-POSED at rA5**로 판정됐다: g2_0_full이 찾은
razor-thin real disjoint(feasible-A={d16,d44} ∩ feasible-B={d54}=∅)는 g2_0_hard
hardening 스윕에서 재현되지 않았고(TTFT 3s-cliff bimodality, n=10 pool 시 d44/d54
둘 다 ~0.86–0.90로 통계적 구분 불가), "disjoint 소멸" 관측은 별도의 ITL-p95
percentile-window 아티팩트였다.

**de-cliff stage-1(jobs 863880–863948) 완료**: `rA{2,3,3.5,4}×{d16,d44,d54}`를
스캔해 `rA=2`만 clean off-cliff임을 확인(`rA≥3`은 전부 여전히 bimodal)하고
`rA=2`를 n=6으로 확증 — 유일한 clean off-cliff 지점에서 static `d54`가 양
phase를 동시 커버, 단 **PLAUSIBLE closure, CONFIRMED 아님**(claims-auditor 반증
3항목: off-cliff에서도 살아있는 split→TTFT gradient·d54 배제 onset이 미측정
전이대·"binding-A⟺on-cliff" 미증명).

**narrow-rA 확증 sweep 완료 — CONFIRMED로 승격**: `g2_0_rasweep`(120 jobs,
`rA{2.25,2.5,2.75,3.0,3.25}×{d16,d34,d44,d54}×n6`)이 off-cliff sub-band
(rate≤2.75)에서 disjoint 부재를 재확인해 전이대를 rate 3.0–3.5로 좁혔고,
그 창을 겨눈 **pre-registered 24-job 확증 열 `g2_0_raconf`**(rate{3.5,3.75}×
{d44,d54}×n=6, 결정규칙: 어떤 rate서든 d54 견고히 <0.7(p90>3s, unimodal) ∧
d44/d16 동시에 견고히 ≥0.95·off-cliff(p90<2s)면 disjoint 실재→REOPEN, 아니면
d54가 양 phase 동시 커버하는 companion collapse면 CONFIRMED)가 **companion
collapse로 판정**:

| rate | split | frac_good mean±SD (n=6) | 비고 |
|---|---|---|---|
| 3.5 | d44 | 0.953 ± 0.035 | |
| 3.5 | d54 | 0.948 ± 0.035 | failTTFT=0/6 (Phase-A도 d44와 통계적 동률) |
| 3.75 | d44 | 0.932 ± 0.042 | |
| 3.75 | d54 | **0.948 ± 0.062** | **d54가 d44보다 높음** |

REOPEN 전제 둘 다 붕괴(d54는 어느 rate서도 <0.7이 아니고, d44도 어느 rate서도
견고히 ≥0.95가 아님: rep 하나가 warm-up성 TTFT-blowup으로 0.844–0.875까지
떨어짐 — 이 blowup은 **split-대칭적**이라 disjoint를 만들지 않음). d54는 Phase B의
유일 feasible split(d44 ITL-p95 50.7ms로 50ms SLO 초과, `frac_good` 0.188;
d54는 44.1ms, `frac_good` 1.000, SD=0)이면서 Phase A도 d44와 대등하게 커버 →
단일 split(d54)이 양 phase를 시간축에서 커버 → **disjoint 없음, 최종 확정**.
상세 per-rep 표·기전·caveat:
[`workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md`](workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md).

★**필수 caveat(overclaim 방지)**: **magnitude는 ill-posed, 순위는 견고** —
Phase-A frac_good≈0.95는 웜업성 TTFT/ITL tail-event(metric cliff)가 결정해
run-length 의존이나, "d54≈d44·d54 미선-배제"라는 **순위**는 견고하다. Phase-B
d44 0.188은 50ms 경계 바로 위라 magnitude는 fragile하나 방향(d44는 decode 못
커버)은 견고하다.

**scope 한정(필수)**: {Zamba2-2.7B, ctx4096, Phase A in2048/o32, Phase B
in2048/o512@rB4, triton attn+mamba, disable-radix-cache, cudagraph-ON, A100
108-SM green-context pdmux, SLO=TTFT 3s ∧ per-req ITL-p95 50ms, inter-phase
drain된 순차 2-phase, rate_A≤3.75} — **"hybrid엔 disjoint 없음"으로 일반화
금지**. **drain caveat**: closure는 얽힘 억제(drain) 조건 관측 = 필요조건
bound이지 hot varying-trace(Claim C 얽힘) 실증 아님. Claim D/E와 §1-20(spatial
coupling-tax, 92+24=116>108, disaggregation +16% headroom)에는 영향 없음 —
시간적 disjoint(단일 static이 시간축에서 양 phase를 커버)와 공간적
coupling-tax는 별개 축이며 "단일 static으로 충분 ⟹ coupling tax 없음"으로
새지 않는다.

**남은 방향(벡터1 종결이 열어두는 것)**: (i) **long-context**(decode floor가
ctx 상승에 따라 올라가므로 — CONSENSUS §1-5 — 충돌이 발생할 수 있는 영역).
★**2026-07-26 갱신, ★★2026-07-28 철회**: 2026-07-26엔 이 방향의 전제(운영점서
decode floor가 ctx로 상승해 binding해지는가)를 Stage 0 게이트가 "ctx≤16k에서는
상승하지 않는다"로 닫았다고 봤으나, **그 근거(D108 무경합 앵커)가 2026-07-28
claims-auditor 감사(C1)로 무효 확인**됐다(아래 "Stage 0" 절). ⇒ **이 전제는 다시
미검증으로 되돌아간다** — "게이트 실패로 보류"가 아니라 "게이트 미실행". (ii)
**§1-20 spatial decoupling**(별도 device pool disaggregation, +16% headroom)은
아직 실행되지 않았다.

상세 verdict:
[`workspace/engine-port/results/g2_0_full/disjoint_verdict_2026-07-24.md`](workspace/engine-port/results/g2_0_full/disjoint_verdict_2026-07-24.md),
[`workspace/engine-port/results/g2_0_hard/hardened_disjoint_verdict_2026-07-25.md`](workspace/engine-port/results/g2_0_hard/hardened_disjoint_verdict_2026-07-25.md),
[`workspace/engine-port/results/g2_0_decliff/decliff_verdict_2026-07-25.md`](workspace/engine-port/results/g2_0_decliff/decliff_verdict_2026-07-25.md),
[`workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md`](workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md).

### Stage 0 (long-ctx L−2 게이트, 2026-07-26) — ★★반증(2026-07-28, claims-auditor C1 CONFIRMED)

7. 운영점(cudagraph-ON, green-context pdmux)에서 decode ITL은 decode-SM(16→108,
   6.75×)에 **무감각(non-binding)**하다 — pure-Transformer(Qwen2.5-3B, 양성
   대조)·pure-Mamba(Mamba2-2.7B, 음성 대조)·hybrid(Zamba2-2.7B, 타깃) **전부**,
   ctx {4k, 8k, **16k**} 전부에서. 유일한 de-confounded 대조 **D16 vs D108(prefill
   경합 0인 두 점) = 1.00 ± 0.01, 3 arm × 3 ctx 전부**.

실험(jobs 864230[H/T]·864601[M], `workspace/engine-port/results/stage0_xctrl/`,
PIN_CHECK 전부 PASS): raw coupled ITL(D) 스윕(D16/D44/D92)은 **CONFOUNDED로
판정**됐다 — D16이 D108과 9셀 전부 ≤0.5% 동일(6.75× SM 증가가 무이득), 최속점 D92는
비단조(prefill을 16 SM으로 굶기는 지점), 그리고 **음성 대조 M**(decode가 O(1)
recurrent라 원리상 SM-bound 불가)이 H와 동형의 "민감도"(2.4×대)를 보이는 것 자체가
그 곡선이 decode-SM이 아니라 prefill 경합/batch-entanglement를 재고 있다는 증거다.
D108 무경합 앵커만이 이 confound를 우회한다.

★★**반증(2026-07-28, claims-auditor 사전등록 게이트 집행, C1 CONFIRMED)**: 위
"D108 무경합 앵커"는 **실제로는 decode 16 SM이었다** — 3중 독립 증거: (i)
**코드 기전** — `manual_divisions=[92,16,0]`의 세 번째 값 0이 legacy auto-path의
threshold로 읽혀 `decode_bs>=0`이 항상 참이 되고 그 결과 **항상 stream_idx
1=(92,16)이 선택**된다(`src/multiplex/multiplexing_mixin.py:725-742`); (ii)
**realized telemetry 재집계** — decode-active 샘플의 **79–96%가 (92,16)**
파티션에서 돌았다(9/9 셀); (iii) **telemetry와 독립인 클라이언트 서명** —
`D108/D16 = 0.992–1.001`(9/9 셀)인데 D92는 D16보다 3.4–3.6× 빠르므로, 108 SM이
92 SM보다 느릴 수 없다는 물리로부터 telemetry 없이도 D108이 실은 D16과 동일
조건이었음이 확인된다. ⇒ **"D16 vs D108 = 1.00±0.01"은 동일 조건의 반복측정**이었다.

**판정2(NULL, "decode SM-무감각")·판정3(게이트 non-binding, "long-ctx 충돌
가설 붕괴·HE0/벡터1 ctx-무관 강화")를 철회한다. 판정1(raw ITL(D) 스윕 =
CONFOUNDED)만 생존**한다(prefill=108−D 공변은 설계상 사실이라 D108 앵커의
유효성과 무관하게 참). ★**"3중 삼각검증" 표현도 철회** — 무경합 앵커는
고장, 음성 대조 M의 전제("decode O(1) recurrent라 SM-bound 불가")도 **틀렸다**
(그 O(1)은 context 길이에 대한 것이지 SM 수에 대한 것이 아니었다 — 아래 "8B
decode-SM 민감도 측정 노트" C2 참조), de-batch 논거는 미감사 — 1/3만 남는다.
D16/D44/D92 각 division의 **pin 자체**(controller 지정값이 realized로도 그
값이었다는 것)는 유효함이 유지된다 — 무효화되는 것은 **D108 앵커 하나뿐**이다.

**연쇄 반영**: `reports/CONSENSUS.md` §1-21 판정2/판정3, `reports/
longcontext_trace_plan.md` §0.6·H_L4·H_L5·L−2 행, `reports/paper/
CLAIM_EVIDENCE_MATRIX.md` Claim A의 Stage 0 evidence 인용, `workspace/
engine-port/results/s0_deconfound/DESIGN.md` §1.1의 "D108: keepalive 0 →
prefill 경합 없음" 표(거짓 — 실제로는 82–96%가 (92,16) 동거; 이 문서가 "재인용
시 필수"로 지정했으므로 정본 재인용 시 반드시 무효 표시)도 함께 철회한다.
**long-ctx 트랙은 "게이트 실패로 보류"가 아니라 "게이트 미실행"으로 복원**한다
— L−2가 실은 아무것도 측정하지 않았으므로 L−1 이상이 멈출 근거가 사라졌다
(재개하라는 뜻은 아니다 — 판정이 없다는 뜻).

상세 [`reports/stage0_verdict_2026-07-26.md`](reports/stage0_verdict_2026-07-26.md)
(원 판정, 위 항목들로 철회됨), `workspace/engine-port/results/s0_deconfound/
PARTITION_RESIDENCY_STAGE0.md`(C1 근거), `workspace/engine-port/results/
s0_deconfound/DESIGN.md`(재측정 설계 — §1.1 표만 무효, 나머지 유효).

### 8B decode-SM 민감도 측정 노트 (scoped, 2026-07-28) — C2 CONFIRMED(scoped)/C2b NOT-YET-SUPPORTED

claims-auditor 판정(사전등록 게이트, `workspace/engine-port/results/
s0_deconfound/DESIGN.md` §5): 아래 측정은 **정책 결론이 아니라 레버 존재를
확립하는 측정 노트**로만 정본에 진입한다. scope 문구는 축약하지 않고 그대로
인용한다.

> {Mamba-Codestral-7.3B / Zamba2-7B / Nemotron-H-8B / Qwen2.5-7B, A100 80GB TP1,
> cudagraph-ON green-context pdmux, `--disable-overlap-schedule
> --chunked-prefill-size -1 --disable-radix-cache`, max-running-requests 48,
> **ctx1024**, conc16 closed-loop, out512, n=4 rep} 조건에서, **prefill을 16
> SM에 고정한 채** decode 파티션만 16→92 SM으로 올리면 decode ITL p50이
> **2.36–2.91×**(4 arm, rep 간 sd 0.01–0.07) 개선된다. 이는 **decode 측
> 등량곡선**이며 `[prefill,decode,idle]=[16,16,76]…[16,92,0]`로 저-D 셀이
> SM을 일부러 놀린다 — **정책 비교가 아니다.** 실제 정책은
> `prefill_SM+decode_SM ≤ 108`을 받으므로 판단 대상은 프론티어 **ITL(D) vs
> TTFT(108−D)**이고 **그것은 미측정**이다. 정본의 실패 기전(얽힘: decode
> 굶김→ITL↑→batch 정체→prefill admission 차단→TTFT 폭발)은 decode-ITL
> 지표에 원리상 보이지 않는다. ⇒ **레버의 존재만 확립하며, 레버를 움직여
> SLO goodput이 나아진다는 근거가 아니다(게이트 #1). HE0(동적 < best
> static)를 되살리지 않는다** — HE0의 死因은 레버 부재가 아니라 positioning
> + 얽힘이었으므로 바뀌는 것은 negative의 **설명**뿐이다. **SM16→SM108(=np)
> 비율은 인용 금지**(분할 자체가 없어 prefill 할당·동시상태·SM clock이
> 함께 바뀜: prefill_active_bs 0.6–0.8 vs 4.5–6.7, clock 1293–1396 vs
> 1396–1403 MHz). ctx는 1024만 귀속 측정됐다(ctx4096은 엔진측 ITL-EWMA
> 프록시로 Ha8 3.46→3.57×, M8 2.56×로 유지 관찰 — **보조 증거**; 8k/16k
> prefill-고정은 미측정). Nemotron-H는 flashinfer, 나머지는 triton(측정
> offset +2.3%, n=1 스모크).

★**국소 탄력도 부기(2026-08-04, claims-auditor)**: 이 **2.36–2.91×는
16→92 끝점 비**이며 구간 평균 ε≈0.48–0.56이다. **국소 탄력도는 16→24
에서 0.77–0.88, 44→92에서 0.09–0.35로 4× 다르다 — 44 이상 구간에 이
비를 적용하지 말 것.**(근거: C2 자신의 batch-matched 표.) 이 caveat
위반 사례가 이미 한 건 있었다 — "C2 탄력도가 벡터1(g2_0_raconf)의
d44→d54 전이(15.0%)를 예측한다"는 시도는 44→92 국소 ε(0.09–0.35)로
+1.9~7.4%를 예측해 **2–8× 빗나가 REFUTED**됐다(estimand 불일치 +
변수 동시 변경도 중복 위반, `reports/CONSENSUS.md` §3 항목21). **C2를
다른 격자로 이전하지 말 것.**

★★**정정(2026-08-14, doc-steward, A-1 — 값 표기 정정만, 새 성능 판정
아님)**: 위 "구간 평균 ε≈0.48–0.56"은 **변환식과 불일치**했다. 정의
`ε = log(ratio)/log(92/16)`, `log(92/16)=1.749200`을 C2 자신의 헤드라인
끝점 비(SM16→92) **2.36×(T8)·2.91×(M8)**에 대입하면 `ln(2.36)/1.7492
=0.491`, `ln(2.91)/1.7492=0.611` ⇒ **ε≈0.491–0.611**이다. 독립 재확인:
2026-08-14 E-1a(`workspace/engine-port/results/bsweep_regime/
e1a_preanalysis_2026-08-14.json`, `T6_PC1_anchor_check`)가 원자료에서
arm별 ε(16→92, C2 자신의 헤드라인 B)를 직접 산출한 값도 **T8 0.492 ·
Hs8 0.543 · Ha8 0.599 · M8 0.610 = [0.492, 0.610]**로 수렴한다(변환식
재계산·원자료 직접 측정 둘 다 일치). **정정: ε≈0.48–0.56 → ε≈0.49–0.61.**
옛 값의 출처를 역산하면 **인용 금지된 SM16/SM108 열**
(`workspace/engine-port/results/s8_scaleup/FINDINGS_8B_2026-07-28.md`
§2의 batch=1/4/8 표, 비 2.54–2.96×, 극단은 T8 2.60×·M8 2.96×)을
`log(108/16)=1.9095`로 변환한 **[0.488, 0.568]**과 훨씬 더 가깝다
(SM16→92 정답 구간 [0.491,0.611]과는 안 맞고, 금지된 SM16→108 구간과는
거의 일치) — **claims-auditor 가설(인용 금지 열의 수치가 본문에
섞여 들어왔다)이 지지된다**(방법론 게이트 #32의 새 사례, B-4 참조).
**과잉 정정 금지**: ITL 비 2.36–2.91×(SM16→92, C2 CONFIRMED scoped)
**자체는 불변** — 바뀐 것은 그로부터 파생된 ε 표기뿐이다. 16→24
(0.77–0.88)·44→92(0.09–0.35) 국소 탄력도도 **별도 산출이라 불변**(이번
정정과 무관, 재사용 금지 caveat 그대로 유지). 상세
`e1a_preanalysis_2026-08-14.json`의 `T6_PC1_anchor_check`(band
[0.49,0.61])·`SC1_pooled_reproduces_s8_batch_matched`, `reports/
CONSENSUS.md` rev27·§3 항목46(追記).

★★**rep 분산 실측값으로 수입값 교체(2026-08-14, E-1a, 방법론 게이트
#32의 새 사례)**: 위 caveat 서술이 인용해 온 "rep 간 sd 0.01–0.07"은
basis 미검증 수입값이었다. 오늘 원자료에서 **직접 측정**했다(44→92
쌍, `e1a_preanalysis_2026-08-14.json`의 `T3_sd_eps`): `sd_rep(log ITL
p50)` 개별 SM점 — Ha8 SM44 0.0044 / SM92 0.0061, T8 SM44 0.0040 / SM92
0.0047 ⇒ **`sd_rep(ε, 44→92)` = Ha8 **0.0102** · T8 **0.0083****. ITL
수준 rep 간 CV(변동계수)는 **0.2–0.7%**다. 캠페인 간(부팅·하네스
v2/v3) 효과는 matched 20셀 대조에서 평균 +0.0001·sd 0.0052(최대
1.1%)로 작다. ⇒ 이 실측값이 향후 인용 시 "0.01–0.07"(basis 미기재
수입값)을 대체한다. ⚠️**스코프 한정**: 이 값은 **44→92 쌍·Ha8/T8 2
arm·이 격자(ctx1024, C2 자신의 헤드라인 B) 한정**이며 다른 SM 쌍이나
다른 격자로 이식 금지. 상세 `e1a_preanalysis_2026-08-14.json`의
`T3_sd_table`·`T3_sd_eps`, `reports/CONSENSUS.md` §3 항목46(追記).

**C2 인용 규율**: 구간(2.36–2.91×)으로 인용, 단일 소수점 금지(bin 선택으로
점추정이 ±0.1 이동: Hs8 2.58 vs 2.67). **C2b("hybrid 급락=Zamba2 additive
성질", Hs8/M8=0.86)는 NOT-YET-SUPPORTED** — 모델간 절대비교(파라미터·형상·
tokenizer 동시 상이) + backend 교차(offset 근거가 20초 스모크 n=1) + 제시
기전(weight-traffic 추정)에 Hs8 데이터가 아예 없고 방향도 반대라 강등한다.
인용 시 scope 문구: "「Hs8/M8=0.86, Ha8/M8=1.57–1.69」는 모델간 절대비용의
**통제되지 않은 관찰**이다(파라미터 수·형상·tokenizer·backend 동시 상이).
아키텍처 계열(additive vs substitutive)에 대한 **기전 주장으로 쓰지 않는다.**"
★**정정(2026-08-11, 트래픽·roofline 진단)**: 위 "제시 기전(weight-traffic 추정)"이
인용하던 수치(M 5.40/T 6.17/H 7.66 GB, H/M=1.42)는 **7-8B arm의 것이 아니라 3B급
모델(mamba2-2.7b/Qwen2.5-3B/Zamba2-2.7B)의 것이고 기준(체크포인트 바이트 vs
호출-인지 트래픽)도 혼합돼 있었다** — 근거 수치 자체가 다른 스케일에서 왔음이
확인되어 C2b의 강등을 **더 약화**시킨다(되살리지 않음). 대체 근거로 새로 계산된
"Ha8 W_step=파라미터 바이트의 1.50×"를 승격하는 것도 금지(같은 arm-간 절대비교
confound 그대로). 상세 아래.

★★**정정/추가(2026-08-11, 트래픽·roofline 진단, GPU 0 — 기존 아티팩트+모델 config
계산만, 새 성능 판정 아님)**: `workspace/engine-port/results/s8_scaleup/
TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md`가 위 C2("decode SM 민감도는 실재하고
모델-무관")에 **스코프 주석**을 추가한다 — "모델-무관"의 원인은 아키텍처 동질성이
아니라 이 측정점(B≈9–12·L≈1.0–1.5k)에서 decode 스텝 트래픽의 68–94%가
아키텍처-무관 weight sweep이기 때문이다(계열 구분 항인 KV·state는 6–31%뿐).
**B나 L을 크게 키운 격자로 이 결론을 이식 금지.** ★**"hybrid는 KV가 적다"는 통념은
이 격자에서 거짓**이다(per-seq 캐시 바이트, L=1282: Ha8 601 MiB > M8 260 MiB > Hs8
117 MiB > T8 70 MiB — Ha8이 T8의 8.6×; mamba2 temporal state는 `d_inner×d_state`
fp32라 L-불변이나 절대 크기가 작지 않고, Zamba2는 GQA가 아니라 MHA head_dim 224×13층).
메인 세션이 제시한 "측정점이 T8-KV=M8-state 교차점(L\*) 근처라서 닮았다"는 설명은
**반증**됐다(L\*=4,750토큰, 측정점은 그 27% 지점). ★★**신규 금지 문구**: SM92에서도
achieved_BW(계산 트래픽÷측정 ITL)는 사양 대역폭(A100 80GB **PCIe**, 1935 GB/s — SXM
2039 아님)의 **48–61%뿐**이라, **고-SM(44→92) 평탄화를 HBM 대역폭 포화로 서술하는
것을 금지**한다. 진짜 원인(wave quantization/층 직렬 사슬/커널 점유율/cudagraph
직렬화)은 커널 단위 측정이 0건이라 미식별. 대조: 1026토큰 prefill은 AI≈1035로
compute-bound이므로 **이 decode 결론을 prefill 축(`s8p_prefill`)으로 이식 금지**
(기존 이식 금지 목록에 사유 1건 추가). C2 자체의 등급·수치(2.36–2.91×, scoped)는
**불변**.

★★★**정정(2026-08-14, doc-steward, E-3 realized SM count 프로브의 부수 발견 —
방법론 게이트 #32의 하드웨어 층 재발, 새 성능 판정 아님, 위 2026-08-11 진단의
등급·"고-SM 평탄화를 HBM 포화로 서술 금지" 판정은 불변·오히려 강화)**: 위
2026-08-11 진단(`TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md`)이 `nvidia-smi -q`로
읽은 하드웨어(`A100 80GB PCIe`, 사양 BW 1935 GB/s)는 **로그인 노드
(glogin01)에서 읽은 값**이었다 — 그러나 이 캠페인(jobs 865289/865533)은
**컴퓨트 노드**(gpu36/gpu38/gpu40, `sacct` 확인)에서 돌았고 컴퓨트 노드는
**SXM4**다(job 882374가 gpu43에서 `torch.cuda.get_device_name()` =
`NVIDIA A100-SXM4-80GB` 직접 관측, `scontrol show node`로 gpu36·gpu40·gpu43
동일 feature `A100-80GB_8,hwperf` 확인 — 파티션 동질). `s8_scaleup/` job
아티팩트(`*.out *.log *.err *.json *.jsonl`) 전체에 `"A100 80GB PCIe"` 문자열은
**0건**이다 — 하드웨어 식별이 측정이 일어나지 않은 기계에서 읽혀 들어온
것. **정정**: 사양 BW **1935→2039 GB/s**(SXM4 mem clock 1593 MHz), achieved_BW
비율 **48–61%→45.4–57.7%**(×0.949), ridge point **161→153** FLOP/byte,
decode AI≈8.2는 ridge의 **5.1%→5.4%**. **어떤 판정도 뒤집히지 않는다** —
"고-SM 평탄화를 HBM 포화로 서술 금지"의 근거는 비율이 더 낮아져 **더
강해지고**, "compute-bound 아님"도 불변. ⚠️SXM4는 400W TDP+NVLink(PCIe는
300W) — 전력·열 특성이 다르므로 향후 평탄화 원인(wave quantization/층
직렬 지연/점유율/cudagraph 직렬화) 분석에서 PCIe 특성을 전제하면 오도된다
(현재 문서는 전력 근거를 쓴 곳이 없어 **깨지는 주장 없음**). 상세
`TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md` §11(addendum), `FINDINGS_8B_
2026-07-28.md` §7 C-5(addendum), `reports/CONSENSUS.md` rev26·§3 항목46
(追記), "방법론 게이트" #32(追記) 아래.

★**스코프 주석(2026-08-15, doc-steward — 메인 세션이 벤더 문서·메트릭 DB·
아카이브 로그로 직접 확인, 새 성능 판정 아님)**: 위 문단 "진짜 원인(wave
quantization/층 직렬 사슬/커널 점유율/cudagraph 직렬화)은 커널 단위 측정이
0건이라 미식별"은 **운영점·green-ctx 한정으로만 참**이다. (1)
`launch__waves_per_multiprocessor`는 green context 하에서도 "scaled with
the number of SMs used by the green context"로 그대로 쓸 수 있다(ncu
2025.3.1.0/드라이버 580.105.08 요건 충족, `ncu --query-metrics-collection
launch --chip ga100`). (2) `nsys --cuda-graph-trace` 기본값 `graph`는
"node activities will not be collected" — 운영점(cudagraph-ON)에서 커널
노드를 보려면 `=node` 명시가 필요할 뿐 원리적 불가가 아니다. (3)
`--exclusive`는 ncu 요구사항이 아니다(직렬화 락은 per-device, `hwperf`는
A100 노드 전체 기본 feature). (4) 선행 ncu 시도가 **이미 있다**
(`workspace/characterization/src/profiling/ncu_runner.py` + 아카이브 8
job, `error code 9` 1,986건 + 메트릭 정규식 실패 1,920건) — 단
`_ncu_target.py:73-74`가 "ncu profiling always runs at full GPU"라
명시해 **green context 하에서 잰 적은 없다**. ⇒ "커널 단위 측정 0건"은
"미실행"이 아니라 "**full-GPU 합성 커널 프로파일링은 2026년 초 시도돼
대량 실패했고, green-ctx·운영점 하의 커널 프로파일링은 시도된 적 없다**"로
정정해서 읽는다. 상세 `reports/CONSENSUS.md` §3 항목52, `handoff-report/
session_handoff_2026-08-15.md` §2.9.

★★★**정정 배너(2026-08-16, doc-steward 등재 — kernel_mech rev2 설계
감사[claims-auditor, 판정 NO-GO] F3에서 발견, 새 성능 판정 아님)**: 바로 위
문단 (4)가 인용한 "ncu profiling always runs at full GPU"(`_ncu_target.py:
73-74`)는 그 **다섯 줄 위(68-71)**에 이유가 적혀 있는데, 정본이 지금까지
그 이유를 **인용하지 않았다**:

> ```
> # Do NOT call smctrl.set_sm_count() here.
> # CUPTI (the API ncu uses for hardware counter collection) is incompatible with
> # CUDA Green Contexts. Restricting SMs via Green Context while ncu is attached
> # causes "Failed to prepare kernel for profiling / Unknown Error on device 0."
> ```
> (`workspace/characterization/src/profiling/_ncu_target.py:68-71`, 메인
> 세션 직접 재확인)

즉 정본이 설계 자체를 죽이는 문장의 **다섯 줄 아래만 인용해 왔다** — "ncu는
full-GPU에서만 돈다"는 *결과*(design intent)이고, 위 인용이 그 *원인*(CUPTI×
green-context 비호환)이다. **이것이 참이면 kernel_mech Stage B(green-ctx 하
커널 단위 프로파일링)는 이 기판에서 구성상 불가**하다.

★★**단, 아직 미확인 리스크로만 등재한다**(★2026-08-20 갱신 — 이 문단의
"미확인" 판정은 P1 프로브로 **실증 쪽으로 좁혀졌다**, 단정 아님. 아래
배너 참조) — 같은 `error code 9`
(1,986건, 아카이브 8 job)에 저장소가 **3가지 경합 귀속**을 갖고 있다: (a)
메트릭 이름 불일치(`run_ncu_profile.py:213`) (b) cuda12 타겟
(`run_ncu_profile.sh:37-41`) (c) CUPTI×green-context(이 문단). 판별기는
**Stage 0′ P1 프로브**(kernel_mech rev2 §8, ≈GPU 0)다. 발화 시 라벨은
**`UNAVAILABLE (CUPTI×GREEN-CONTEXT)` = 도구 한계**이지 게이트 실패도
기전 증거도 아니다(방법론 게이트 #21 "측정 실패를 게이트 실패로 라벨링
마라"). 상세 `reports/CONSENSUS.md` §3 항목52 追記(5),
`workspace/engine-port/results/kernel_mech/DESIGN_KERNEL_MECH_REV2_
2026-08-16.md` F3, `handoff-report/session_handoff_2026-08-16.md` §4-4(b).

★★★★**실증 확정 배너(2026-08-20, doc-steward 등재 — P1 프로브 job
886718 결과 반영, 새 성능 판정 아님, 도구 타당성 판정)**: 바로 위
문단이 "판별기"로 지정한 Stage 0′ P1 프로브가 **실행됐다**
(`--exclusive --constrain=hwperf`, node gpu38, 1분55초). 결과는
**정확히 (c)** — greenctx 다리에서 문서화된 시그니처(`Failed to
prepare kernel for profiling` / `Unknown Error on device 0` / exit
9)가 그대로 재현됐고, **두 겹 대조**로 귀속이 깨끗하다: (a) **다리
간** — control(같은 GEMM, full GPU)은 에러 0건·60행 정상 수집. (b)
★**다리 내부** — 같은 프로세스·같은 ncu 호출에서 green ctx **밖**
RNG 커널(8행)은 성공하고 green ctx **위** GEMM만 실패, 나머지 변수는
전부 동일(job·노드·권한·ncu 호출·메트릭·클럭 정책·타깃·커널·차원).
⇒ **(c) CUPTI×green-context가 이 구성에 한해 관측 근거를 얻었다** —
`_ncu_target.py:68-71`은 더 이상 "코드 작성자의 주장"이 아니다.

★**그럼에도 지킬 서술 한계(overclaim 금지)**: (i) **성능 판정이
아니라 도구 타당성 판정**이다. (ii) 깨진 것은 **프로파일링이지
green context 실행이 아니다** — `realized_sm=16`으로 정상
실현됐고 커널도 실행됐다(동반 프로브 886752는 `DONE`까지 완주). (iii)
**내부 기전은 미분리** — 관측은 "green-context 스트림 위 커널의
프로파일링이 실패한다"까지이고, `UNAVAILABLE (CUPTI×GREEN-CONTEXT)`
라벨은 하네스가 붙인 이름이지 CUPTI 비호환 대 스트림/컨텍스트 처리
다른 층을 이 프로브가 갈랐다는 뜻은 아니다. (iv) **스코프 한정** —
A100-SXM4-80GB·driver 580.105.08·ncu 2025.3.1.0·CUDA 13.0.2·이
클러스터(`amd_a100nv_8`) 한정, 다른 버전·기판으로 이식 금지. (v) (a)·
(b) 경합 귀속(메트릭 이름 불일치·cuda12 타겟)은 **저장소의 다른
아카이브 배치**(옛 `workspace/characterization/` 트랙, full-GPU,
cuda12 venv 불일치)를 설명하는 것으로 남는다 — 이번 P1은 **모듈을
표준화한 동일-툴킷 구성**에서 순수 green-context 원인만 격리했으므로
(a)·(b)와 모순되지 않는다(서로 다른 실패 배치를 설명).

⇒ **`kernel_mech` rev3는 Stage B를 폐기하고 Stage A 전용으로 범위를
좁힌다**(GO 경로였던 "문서 수정 8건" 중 Stage B 대상 최소 5건이
적용 대상 소멸). ★**트리의 wave 수치는 이 기판에서 원리상 측정일 수
없다**는 kernel_mech rev2 감사 F1(`wave_eff`는 ncu 메트릭이 아님)을
독립적으로 뒷받침한다. 부수 확정: 선례 스크립트 `run_ncu_profile.sh:
15-17`의 권한 근거("batch면 권한이 열린다")는 **불충분**함이
실측으로 드러났다 — 동반 프로브 886752(non-exclusive batch)가
`ERR_NVGPUCTRPERM`을 받았고, 아카이브 ncu 로그 8건 전수 조사 결과
`ERR_NVGPUCTRPERM` 0건은 전부 `--exclusive --constrain=hwperf`
계열이었다(실제 구분선 = exclusive+hwperf) — 그리고 그 8건 중
실제 데이터를 수집한 유일한 성공 사례(`ncu_729105`, 88행)조차
**green context 언급이 0건**이라 이 저장소는 **green context 하에서
카운터를 수집한 적이 한 번도 없었다**(P1이 진짜 미해결 질문이었다는
확인). 상세 `workspace/engine-port/results/kernel_mech/p1_probe/
P1_VERDICT_2026-08-20.md`, `P1_886752_REVIEW_2026-08-20.md`,
`reports/CONSENSUS.md` §3 항목52 追記(6), "다음 실험 gate" #11
레지스트리(kernel_mech 행 갱신)·#17(kernel_mech rev3 스코프 확정).

`ADVERSARIAL AUDIT COMPLETE (2026-08-15) — claims-auditor 판정: 등급
유지 + 인용 정지 2건 신설.`
★★★**(2026-08-15, result-analyst 1차 산출 + claims-auditor 적대
검증 — 새 성능 판정·정책 주장 아님, 등급 CONFIRMED(scoped) 유지)**:
헤드라인 "2.36–2.91×" 4셀의 job 구성을 원자료에서 직접 재구성했고,
claims-auditor가 그 산출을 적대 검증해 **일부는 확증, 일부는
반증**했다.

**생존(반증 시도 실패, 인용 가능)**: **3셀(Ha8·M8·Hs8)은 양 다리
모두 job 865533 단독, T8도 SM92 다리는 865533 단독**(SM16 다리만
865493 61%+865533 39%) — 1차 원인은 keepalive 붕괴가 아니라
**하네스 결손**(865493은 M8 전 5셀·Ha8 d16 셀에 `t0_monotonic_s`가
없어 귀속 구간 0건). 같은 `(arm, 실현SM, decode batch)`를 채운
58개 매칭 셀에서 조건부 ITL 차 **−0.18%±0.62%(SD), 최대 2.20%**는
claims-auditor의 3중 추가 반증 시도(헤드라인-다리만 재집계·`tok_idx`
매칭·동거 prefill 강도 `pf_bs` 매칭)에서도 살아남았다(각각 ≤0.58%·
≤0.2%·≤1%). 독립 스크립트 재실행 결과는 저장소 JSON과 byte-identical.

**반증됨(REFUTED)**: (i) "깨끗한 런(865493) 단독 재계산 T8
2.388×[2.379,2.397]·Hs8 2.687×[2.686,2.689](양 다리 4 rep)를 CI·
n=4와 함께 인용 가능" — **T8 b16의 865493/d16 rep1–4는 실은 서버
부팅 1회**(다리당 `n_indep=1`, rep1–4는 의사반복)라 보고된 CI는
토큰/rep 내부 잡음만 잡은 것이고, rep 간 변동을 t(3)로 반영하면
**≈[2.368,2.408](±0.85%), 약 2배 넓다** — "양 다리 4 rep = 게이트
n≥4 충족"은 거짓. ⇒ 점추정 T8≈2.388×·Hs8≈2.687×는 `n_indep=1`
명시 하에서만 인용 가능, CI·"n=4"·"게이트 충족" 표기는 **인용
정지**. (ii) "Ha8은 b12 슬라이스가 양 다리 모두 0건이라 문서 자신의
'batch 12–16 대표값' 지침을 구조적으로 만족할 수 없다" — 귀속
오류. Ha8 SM16 다리는 job 865533(붕괴 런)에서 최대 b11까지만
관측되고 865493(깨끗한 런)은 d16 셀 자체에 t0 결손이 있어 대응
데이터가 없지만, **SM92 다리는 오히려 865493(깨끗한 런)이 batch=16
까지 도달한다**(33,344건, 전부 865493). 즉 도달 불가는 구조적
제약이 아니라 (a) 865533의 낮은 batch 상한 + (b) 865493 SM16
다리의 t0 결손의 조합이다 — "865533에서 미도달, 865493은 SM92에서
b16 도달"로만 서술한다. (iii) **게이트 #12 ctx 스코프 서술**("수치는
ctx4096 한정, 기전은 ctx-무관") — REFUTED, 아래 "다음 실험 gate"
#12 재정정 참조.

**신규 인용 정지(claims-auditor가 추가로 확인)**: **(a) arm별 ε 및
arm 간 순위/격차 인용 정지** — 헤드라인 슬라이스의 ε(=ln(ratio)/
ln(5.75)) 순서 T8 .492 < Hs8 .543 < Ha8 .599 < M8 .610은 **슬라이스
선택의 산물**이다. 4 arm 공통 batch(b=1)에서는 **Ha8 .480 < Hs8
.514 < M8 .515 < T8 .518**로 순서가 완전히 뒤집히고 폭이
0.118→0.038로 좁아진다 — `FINDINGS_8B_2026-07-28.md` §2.1 자신이
"Ha8의 SM 민감도는 오히려 최저(2.54–2.75×)"라 적어 헤드라인
순서와 모순한다. ⚠️하방 소비자 추적 필요(미확인): `bsweep_regime/`
E-1a의 `T6_PC1`, `reports/figures/canon.py:308-335`. **(b) 깨끗한
셀 CI·"n=4" 표기 인용 정지**(위 (i) 참조). 추가로 **이질적 batch
추출 규칙** 확인: T8/M8/Hs8은 b12, Ha8은 자신의 최댓값 b9를 쓴다
— "2.36–2.91×"는 매칭 비교가 아니라 이질적 규칙의 나열이다. b12
매칭 3 arm(T8/M8/Hs8, 헤드라인과 동일)의 범위는 2.366–2.909(23%
폭). **4 arm 공통 batch(b=1)의 2.31–2.48은 "수렴 확증"으로 과잉
해석 금지** — b=1은 weight-sweep 트래픽 지배 지점(§7 C-2)이라
arm 간 근접이 준-항등에 가깝다. **realized-SM 조건화 자체가
동거율을 구성상 ≈1로 강제**하므로(green-ctx는 동거 중에만 실현)
58개 매칭 셀은 붕괴 경로가 조건화로 이미 제거된 잔차다 — 조건화가
안 걸린 SM108 셀에서는 실제로 −2.2%/−1.26%가 관측된다. 58셀 중
헤드라인 다리(SM16/SM92) 자체는 17개뿐(M8 0, Ha8-SM16 0).

**종합 판정(claims-auditor, 2026-08-15)**: **등급 CONFIRMED(scoped)
유지 + 인용 정지 2건 신설**(arm별 ε·순위 / 깨끗한 셀 CI·"n=4" 표기).
"레버 존재, 2.3–2.9× 대역, 4 arm 전부"는 그대로 인용 가능. **해소
실험(등재, 미실행)**: keepalive ≤1792 토큰 + 4 arm × {d16,d92} ×
독립 서버 부팅 4회씩(rep 아님) + concurrency 계단으로 b≥12 강제 —
job-불변성·batch 매칭·`n_indep≥4`를 한 설계로 동시 해소. **Stage 0
D108 전례와 다르다**: Stage 0은 라벨 오류(§1-21), 이번은 라벨이
맞고 교락이 직접 측정됐다 — 같은 유형으로 인용 금지.

⚠️**provenance(최종)**: "감사 대기"가 아니라 **claims-auditor 적대
감사 완료(2026-08-15)** 상태다. 산출 주체는 **result-analyst(1차)
+ claims-auditor(적대 검증)**이고 **메인 세션 독립 재확인 없음**.
상세 `workspace/engine-port/results/s8_scaleup/
AUDIT_C2_HEADLINE_JOB_COMPOSITION_2026-08-15.md`(1차, 위 반증
3항목 포함 원문 보존), claims-auditor 적대 검증(원자료 파일 위치
미확정, 다음 세션 편입 요망), `reports/CONSENSUS.md` §3
항목54(신설, 재정정), `FINDINGS_8B_2026-07-28.md` §8(신설,
재정정), `reports/paper/CLAIM_EVIDENCE_MATRIX.md`(Claim A 각주,
재정정).

`C2-R CAMPAIGN COMPLETE (2026-08-16) — jobs 883574(M8)/883575(Ha8),
결정 = M8·Ha8 신규 점추정 2건 등재(기존 값 교체 아님), C2 등급 무변경.`
★★★**(2026-08-16, result-analyst 산출 + claims-auditor 적대 검증,
메인 세션은 운영 지표만 직접 확인) 사전등록 `PREREG_C2R_RULES_REV2_
2026-08-15.md`(rev2 GO) 집행 결과 — M8·Ha8의 신규 점추정 2건 등재,
기존 값 교체 아님. C2 등급 무변경(CONFIRMED scoped). 인용정지 (a)
(arm별 ε·순위)·(b)(깨끗한 셀 CI·"n=4") 둘 다 유효(해제 0건).**

**정본 인용 문구(claims-auditor 지정, 그대로 채택)**: "`r_M8(16) =
3.058`, `r_Ha8(16) = 3.114`(ctx1024, realized SM16/SM92,
`decode_bs=16`, job 883574/883575, gpu43, 1시간, `n_indep=6` 부팅).
동반 구간은 **within-job 부팅 구간이며 재현 불확실성이 아니다** —
보수적으로 **t(5) [3.056, 3.060] · [3.077, 3.155]**를 쓰고,
**job/node/날짜 축은 미측정(n=1)**임을 병기한다." ★percentile
부트스트랩 CI([3.0566,3.0595]/[3.0848,3.1354])는 **정본 본문에 쓰지
않는다** — 일관되게 과소피복(M8 1.36×·Ha8 1.55× 더 좁음), 원자료
JSON(`C2R_RESULTS_2026-08-16.json`) 포인터로만 남긴다. between-job
SD는 **미측정**이다.

**seed 민감도**: `seed∈{1,2,3,99}`에서 M8 CI95 [3.0566,3.0595/6]·
Ha8 CI95 [3.0843–3.0850,3.1354–3.1356], SD 변동 ≤1.5% ⇒ 부트스트랩
seed는 결과를 만들지 않는다(사전등록 seed=1 선택은 무해). ⚠️이것을
"CI가 견고하다"로 읽지 말 것 — seed 안정성은 **재표집 잡음**만
배제할 뿐, **재표집되는 모집단**(boot 단위, job/node/day 축 부재)의
문제는 그대로다.

D3(보고 전용, 게이트 아님): Ha8은 실현률 0.80에 12/12 셀 미달(d16
0.508–0.765, d92 0.735–0.792, 865493 .682/.760과 같은 대역) — arm의
성질로 보이며 감사 N5는 미해결이다.

★★★**양성대조가 항등식이었다(방법론 게이트 #9 아홉 번째 재발, 아래
"방법론 게이트" #9 참조)** — 대조가 실행한 코드 경로(`s8_c2r_score.py`
`cmd_poscontrol`:270, `legacy`+`keep_slack=False`)는 헤드라인이
실제로 쓰는 경로(`cmd_c2r`:333, `c2r`+`keep_slack=True`)와 다르고,
표적값(T8 2.388·Hs8 2.687) 자체가 같은 estimand 루프의 또 다른
복사본(`audit_c2_job_composition.py`) 산출물이다. 배제된 오류는
전사·t0 조인·pooling뿐이며, 공백(헤드라인 경로 자체의 검증)을 실제로
메운 것은 대조가 아니라 **claims-auditor의 독립 재구현**(해당 코드
미import, 새 스크립트로 3.057990/3.114061 및 24 부팅 n·median 전부
재현)이다. ★**provenance errata**: 그 재구현 스크립트(`indep.py`·
`legacy.py`)는 **스크래치에만 있고 저장소에 없다** — 사실 자체(독립
재현됨)는 성립하지만 **재현 경로가 저장소에 보존돼 있지 않다**
(2026-08-14 E-1a `VERDICT`/`T6_PC1` 미산출 전례와 동형: 헤드라인
경로를 검증한 것은 저장소 밖 재구현이며, 그 코드는 보존되지 않았다).

★엔진 빌드 정정 — 865493 대비 `runtime_source_manifest*.sha256`가
**11→15 파일**로 늘었고, 추가 3종이 hot path다(`scheduler.py`의
HOLB hook을 `run_batch` 3경로에 삽입[커밋 `4e4e01d`, 2026-08-07]·
`holb_probe.py`·**`zamba2.py`**[= Ha8 자신의 forward, 커밋
`7de5336`, 2026-08-06]). 865493 시점 해시는 **무증명** ⇒ "C2-R과
865493의 차이는 플래그 2개뿐"이라는 서술은 **거짓**이며, 두 캠페인을
섞는 모든 비교의 각주에 이 3파일을 명시할 것.

guard 고원(C2 전반에 적용되는 estimand 성질): `r`은 guard∈[2.5,∞)에서
불변(Ha8 3.1141→3.1133→3.1132)이나 **guard≤0.75면 셀이 빈다** —
정본 estimand의 채택 구간(`GUARD=3.0`)이 전부 **0.7–6.3초 미관측
창 안**에 있어 **(16,16) 점은 관측이 아니라 보간**이다.

전-구간 강건성: 조건화 없이 계산해도 `r_M8=3.055`(−0.1%)·
`r_Ha8=3.106`(−0.3%) — 위 점추정과 사실상 같다.

**등재 금지(claims-auditor가 명시적으로 막음)**: (i) **"C2-R 값이
정본 2.36–2.91×보다 위"** — 범주 오류, C2-R의 d16 다리는
`b∈{1,15,16}`뿐이라 M8 b12·Ha8 b9가 아예 없어 batch/job 분해가
불가능하고 단조성도 국소 위반(4건) — **병기만, 대체·순서 서술
금지**. (ii) **인용정지 (b) 해제** — 범주 오류, (b)는
`PROJECT_STATUS.md`·`CONSENSUS.md`의 T8·Hs8 깨끗한 셀 표기에 걸린
것이고 C2-R은 그 둘을 재측정하지 않았다 — **(b)는 그대로 유효**,
새로 생긴 것은 M8·Ha8의 새 인용 대상뿐. (iii) **N1/N2 해결** — b=1은
준-항등 체제·대조 상대가 붕괴 job(865533)이며, 헤드라인 SM16 다리는
두 arm 다 교차-job 대조 **0건**(검정력 분해능 ≈1% = M8 CI의 약
20배) — **N1/N2는 여전히 미해결**. (iv) **C2 등급 변경 없음**
(CONFIRMED scoped 그대로).

다음 실험 gate(아래 #13 신설): (1) job/node 축(동일 커밋·15파일
매니페스트로 ≥4 job×≥3 노드×≥2 날짜, 1차 결정량 = between-job SD of
r) — **이게 없으면 어떤 CI도 인용 불가**. (2) batch vs job 분해(한
job 안에서 conc 계단 4/8/12/16으로 b=9·12·16 공존, within-job
`r(b)`를 865533의 b9/b12와 대조). **이 둘 전에는 "3.06 vs 2.91"에
어떤 판정도 내리지 않는다.**

⚠️★**provenance**: 캠페인 jobs 883574(M8)·883575(Ha8, 각 12부팅) ·
스모크 883351/883545/883563 · 분석 result-analyst · 적대 감사
claims-auditor · **메인 세션은 운영 지표(부팅 성공·H7·실현률)만
직접 확인**, 점추정·CI 산출은 독립 재현하지 않았다. 상세
`reports/CONSENSUS.md` §3 항목56, `workspace/engine-port/results/
s8_scaleup/C2R_RESULTS_2026-08-16.md`, `PREREG_C2R_RULES_REV2_
2026-08-15.md`.

상세 [`workspace/engine-port/results/s8_scaleup/FINDINGS_8B_2026-07-28.md`](workspace/engine-port/results/s8_scaleup/FINDINGS_8B_2026-07-28.md)
(§6에 (a)"레버 존재≠정책 이득" (b)"HE0를 되살리지 않는다" 명시, §3 retraction을
"SM108이 근소하게 빠르다"에서 "비교 불가"로 강화 — 2026-07-28 doc-steward 반영;
§7에 2026-08-11 트래픽·roofline 정정 C-1–C-4 추가), [`workspace/engine-port/
results/s8_scaleup/TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md`](workspace/engine-port/results/s8_scaleup/TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md)
(진단 전문, arm-간 achieved_BW 순위 비교는 여전히 금지).

## 열린 긴장 (2026-07-28, claims-auditor 지정 — HE2/§1-5/§1-7/HE0 철회 아님)

- **긴장 A (HE2 vs C2)**: C2가 맞는데 왜 HE2(운영점서 decode 최적 split=static·
  불변, 동적이 anchor 무관 패)는 평탄했나? 유력 가설 = HE2는
  `prefill+decode≤108` 예산 제약 하에서 D를 움직였으므로 decode 이득이 prefill
  손실+얽힘으로 상쇄된다 — **레버는 있으나 예산 제약 하 net-positive가 아닐 수
  있다**는 뜻. **이것은 가설이며 미측정**이고, 정확히 아래 E1(프론티어 실험)의
  대상이다. ★**갱신(2026-08-05)**: §0(C2와 E1 격자가 같은 물리량을 재는지)이
  S2(job 873015) 독립 재현으로 **behavioural CONFIRMED (scoped)** 종결됐으나
  (위 "8B decode-SM 프론티어" "2026-08-05" 소절), 이는 §0만 닫을 뿐 **긴장 A는
  전혀 닫지 않는다** — E1이 열리지 않는 4가지 독립 사유(`G_LEVER`/`G_FLAT`
  UNDETERMINED·sticky 기판의 estimand 전환·prefill 축 미통제·음성대조 구조적
  부재+S3 미실행)가 그대로 남아 있다.
- **긴장 B (r0c 부분 복권)**: `reports/CONSENSUS.md` §1-5가 r0c의 no-cudagraph
  decode-knee(ctx256 1.1×→ctx16k 10.5×)를 "운영점 magnitude는 열린 질문"으로
  강등했는데, C2(cudagraph-ON 서빙, ctx1024, 2.36–2.91×)가 그 곡선 위에 앉는다
  ⇒ "열린 질문"이 일부 닫히는 **방향**이다. **"정합"까지만 쓴다 — "확증"으로
  쓰지 않는다.**
- **`results/s8p_prefill/`(prefill 축 SM 민감도) — 완료(2026-07-29), 정본 인용
  금지 유지**(claims-auditor 미통과). 판정서
  [`FINDINGS_PREFILL_2026-07-29.md`](workspace/engine-port/results/s8p_prefill/FINDINGS_PREFILL_2026-07-29.md).
  한 줄 요약(등급어 없이 REAL scoped·미감사로만 인용): "prefill 축 SM 민감도 =
  기울기 비 **4.74–5.16×**, 탄력도 ε **0.89–0.94**, 4 arm 모델-무관". 아래
  "다음 실험 gate" #8에 claims-auditor 반증 축과 함께 기록. ★**이식 금지 사유
  추가(2026-08-11, 트래픽·roofline 진단)**: decode 축(ctx1024, L≈1282)의 산술강도는
  AI≈8.2 FLOP/byte(memory-bound, ridge 161의 5%)인 반면 1026토큰 prefill은
  AI≈1035(compute-bound, ridge의 6.4×)다 — **두 축은 서로 다른 roofline 영역에
  있으므로 decode 축의 achieved_BW/포화 판정(위 "8B decode-SM 민감도 측정 노트"
  C-4)은 이 prefill 캠페인으로 이식 금지**. 반대 방향(prefill 축 곡률 결론을
  decode 축으로 이식)도 마찬가지로 금지.
- **`results/s8_frontier/`(E1 8B 프론티어) — 하네스 구축 완료, 본 스윕
  미실행**. ★**2026-08-02 갱신**: 2026-08-01에 나머지 3 arm 용량 스캔
  (870295/870296/870297)과 T8 batch-cap 실험(870301)이 완료됐고, 그 결과가
  **E1 설계 자체에 대한 판정**으로 이어졌다(아래 "2026-08-01 실험 4건" 소절).
  **본 스윕은 여전히 미제출**이다. 용량 스캔(job **867231**, T8,
  5셀)·관측자 효과 게이트(job
  **867298**, `results/e1_traceforce/`)는 세션 핸드오프(2026-07-29) 작성
  시점엔 PENDING으로 기록됐으나, ★**2026-07-31 doc-steward 갱신 시 `sacct`
  재확인 — 둘 다 COMPLETED**(867231: 2026-07-29 21:15:25–22:41:31 / 867298:
  21:16:56–21:50:21). ★**2026-07-31 정정** — 같은 날 앞선 갱신의 "분석/판정
  파일 0건"은 **오기**다. 두 job 모두 **분석 단계가 job 스크립트 안에서 이미
  실행**되어 결과가 `e1cap_T8_867231_result.txt`(100 probe)·
  `tfgate_T8_867298_result.txt`(`=== PAIRED SUMMARY ===`)에 있고, 판정은
  [`handoff-report/session_handoff_2026-07-29.md`](handoff-report/session_handoff_2026-07-29.md)
  §10에 기록돼 있다. 부재한 것은 별도 FINDINGS 문서뿐이며, **재실행은 불필요**하다.
  - **관측자 효과 게이트 = 조건부 통과.** d16은 전 지표 t95 CI가 0을 포함,
    d92만 `itl_p95` **+2.00%** [+0.78, +3.21] · `itl_p99` −1.28% · `itl_mean`
    +0.56%가 0을 배제. 교란의 크기(≤2%)보다 **위치**가 문제 — 하필 결정 규칙이
    임계하는 ITL-p95이고 D-격자의 한쪽 끝에서만 난다. ⇒ **본 스윕은
    `PDMUX_TRACE_FORCE_PREFILL=0`(기본값)으로 돌리고, pin 검증은 같은 arm/cell/
    rate/seed의 짧은 ON 런으로 분리**한다(사전등록 = `results/s8_frontier/
    DESIGN.md` §4.7.1). ⚠️ 그 job의 `PIN_CHECK`는 `traceforce_gate.sbatch`의
    옛 인자 순서 때문에 전부 크래시해 **pin 데이터가 없다**(paired summary는
    별도 분석기 산출이라 무영향). 인자 순서는 **2026-07-31 수정 완료**.
  - ⚠️★**T8 용량 스캔은 §4.2/§9.1이 사전등록한 escalation 분기를 발동시켰다.**
    ★**2026-07-31 claims-auditor가 이 항목의 초판을 정정** — "5셀 동시
    off-cliff rate 부재"는 부정확하다(rate≲2에서 d92 포함 전 셀이 plateau 위).
    knee(첫 교차, plateau 2×): d16 12.6 / d24 16.0 / d44 16.0 / d54 **8.45** /
    d92 **2.80** req/s. **견고한 것은 d92 knee가 나머지보다 3배 이상 낮다는
    순서**(세 임계 × 두 x축 불변, 기전도 확실: 저부하 TTFT plateau가 181 vs
    47–62ms이므로 같은 rate에서 ρ가 3–4배). 실제로 성립하는 명제는 **"공통
    off-cliff band(≲2–3 req/s)가 ITL 항이 움직이는 영역과 완전히 분리돼
    있다"**이며, 셀별로 다른 rate를 골라 우회하는 것은 §4.2가 금지한 rate-교락
    이므로 여전히 불가하다. 상세 = 아래 "열린 긴장".
  - ★★**T8의 ITL 항은 전 rate·전 사다리 룽에서 non-binding**(claims-auditor
    2026-07-31, C-E CONFIRMED). 요청 단위 직접 집계로 rate 1–32 전 구간에서
    `ITL-p95 ≤ 50ms`를 요청의 **≥93%**가 통과한다. ⇒ T8에서는 **어떤 결과가
    나와도 C2 판정 불가**(§4.3.4가 사전등록한 `ITL-NONBINDING` 취급). ⚠️단
    **"그러므로 E1의 판정력이 M8/Ha8/Hs8에 걸린다"는 NOT-YET-SUPPORTED** —
    같은 batch cap이 그 arm들에도 걸려 같은 계단 구조가 생기고, Ha8은 외삽상
    d16 ≈149ms로 **전 룽 초과**(모든 셀에서 구속 ⇒ conjunctive goodput 전멸
    ⇒ 역시 판정 불가) 가능성이 있다. §4.3.4에 대칭 플래그
    **`ITL-ALWAYS-BINDING`이 없다** = 미등록 실패 모드(신설 필요).
    ★**2026-08-02 후속**: 플래그는 2026-07-31에 `DESIGN.md` §4.3.5(a-1)로
    **신설·사전등록**됐고, 2026-08-01 스캔에서 **Ha8이 전 룽
    `ITL-ALWAYS-BINDING`으로 실제 발동**했다(위 "2026-08-01 실험 4건" (3),
    **미감사**) — 위 "판정력이 M8/Ha8/Hs8에 걸린다"는 그만큼 부분 반대
    증거를 얻었다(철회 기록은 "철회된 가설" 절).
  - ★★**미등록 하네스 상수가 ITL 축을 단독 결정한다 — `--max-running-requests
    48`**(`e1_capacity_scan.sbatch:143`, `e1_sweep.sbatch:178`). `DESIGN.md`에
    **단 한 번도 등장하지 않는다**(grep 0건). 증거: telemetry의
    `decode_running_batch_size`가 d16/d24/d44/d54 전부 정확히 48에서 절단
    (d92만 38 — prefill admission이 먼저 막혀 cap 미도달 ⇒ **d92의 ITL은 다른
    셀과 비교 불가**), 그 시점 `kv_occupancy = 0.024`(2.4%, 자원 강제 아님 —
    ⚠️**T8 한정 관측**이며 "메모리는 어느 arm에서도 구속하지 않는다"로
    일반화 금지: hybrid arm의 mamba pool은 cap과 항등이라 별개 문제다,
    "방법론 게이트" #5),
    d16 ITL-p95가 rate 12→32에서 50.2–50.6ms로 완전 평탄. ⇒ 셀별 ITL "천장"
    {d16 50.6 / d24 37.9 / d44 26.2 / d54 23.9 / d92 19.1}은 **모델 성질이
    아니라 설정 성질**이며(★2026-08-01 batch-cap 실험이 T8 2셀에서 이를 직접
    시험 — 위 "2026-08-01 실험 4건" (1), **미감사**), "60ms 도달 불가"의
    scope가 달라진다.
  - ★★★**decode 측 realized-partition duty cycle이 D와 공변한다**(Stage 0
    C1과 같은 종, 이번엔 decode 축). `decode_sms`를 decode-active 구간에서
    시간가중하면 **라벨 D SM에서 보낸 시간 비율 = d16 0.110 / d24 0.112 /
    d44 0.166 / d54 0.201 / d92 0.518** — 나머지는 무분할 108 SM이다. 즉
    **D 축이 "decode SM 양"과 "그 제한이 걸리는 시간 비율" 두 변수를 동시에
    움직인다.** 사전등록 `E1_PIN_GATE`는 **prefill 측만** 검사하므로 이걸 못
    잡는다(`D=16(P92) pin_frac=0.950`은 prefill이 92 SM인지의 지표). 배치·
    파티션을 동시에 맞추면(b=48, `decode_sms==target`) d16 28.8 / d24 22.2 /
    d44 19.4 / d54 18.4ms로 **d16이 27% 더 느리다**(혼합 집계의 22.6보다 큼).
    ⇒ **decode 측 duty-cycle 게이트 신설 필요**(수치 자체도 스냅샷 샘플링
    기반 1차 근사라 엔진측 누적 시간으로 재측정 대상).
  사전등록 SLO·결정 규칙은 아래 "다음 실험 gate" #8 참조.

  #### 2026-08-01 실험 4건 — ⚠️**전부 claims-auditor 미통과, 등급 = 미검증, 정본·논문 인용 금지**

  > ★★**2026-08-02 감사 완료 — 아래 절의 상당수가 이미 철회·대체됐다. 이 절을
  > 인용하기 전에 반드시 이 박스를 먼저 읽을 것.** claims-auditor 회부 결과
  > **2 REFUTED / 2 NOT-YET-SUPPORTED / 1 CONFIRMED**, 후속 M1/M2/M4로 다음이
  > 확정됐다(정본 = `results/s8_frontier/DESIGN.md` §4.3.7–§4.3.8 +
  > `FINDINGS_M1_M2_2026-08-02.md`):
  >
  > - **(1) "천장 = 설정 성질" → NOT-YET-SUPPORTED.** 1 arm·2셀에서 모델 변이
  >   0인 설계로 "모델 성질이 아니다"를 결론할 수 없고, d44의 +1.3ms도 실은
  >   CI가 0을 배제한다(=설정×셀 상호작용). 더구나 **rate 16 = 공통 off-cliff
  >   대역의 5.7배**라 E1 운영구간 밖이다. **재실험하지 않고 폐기.**
  > - **(2) "cap 96 ≈ 192" → REFUTED.** `frac(ITL≤60)` −0.091[−0.166,−0.016],
  >   TTFT p50 +6.75ms 모두 0 배제. d44 cap96 행은 누락이 아니라 미보고이고
  >   넣으면 비단조.
  > - **(3) mamba pool 항등식 → CONFIRMED**(코드 분기 + 서버 로그
  >   `max_mamba_cache_size: 48` + telemetry의 1/48 양자화, 3방향 독립).
  >   단 "1.0000 = 포화"는 **max 통계**였고 실제로는 스냅샷의 0.006–0.08%.
  > - **(4) "HEADLINE NONE = cap 아티팩트" → REFUTED.** 운영대역 실측 동시성
  >   12–44 < cap 48 ⇒ **cap이 구속할 수 없다**. 다만 결론 자체
  >   (`HEADLINE-ELIGIBLE RUNGS = NONE`)는 **살아남았다** — 공통 rate·두
  >   estimand·두 seed 전부에서 재확인. **arm×룽 표는 rate-confound로 폐기**
  >   (T8만 rate 12, 나머지 rate 2 — §4.3.5(b) 명시 위반), `--common-rate`
  >   출력으로 대체.
  > - **(5) knee — "네 arm 공통 2.80" 철회.** knee의 치역이 probe 격자뿐이라
  >   일치가 부분적으로 강제된다(게이트 #6). 9-변형 집합은 arm마다 다름.
  >   **살아남는 것은 순서**(d92가 먼저 무너짐: T8/M8/Hs8 9/9, Ha8 8/9).
  >   Hs8 d16 knee 9.09의 취약성은 실재하나 **원인 귀속이 틀렸다** — 그
  >   이상치를 지워도 knee 불변.
  > - **(6) stall 원인 규명(M4, GPU 0).** 17개 stall probe 중 **16개**에서
  >   최장 프롬프트의 **monolithic prefill**이 stall 전 구간을 덮는다. 크기는
  >   prefill SM = 108−D 이므로 **D에 단조**. `enable_pdmux`가
  >   `chunked_prefill_size == -1`을 하드 assert하므로 **기판 구조이지 버그가
  >   아니다**(venue positioning의 (A) green-context 종속 버킷).
  >
  > E1 본 스윕은 여전히 미제출이며, 대신 **M3 Transformer-control 대조**를
  > 사전등록·제출했다(job **872077**, §4.3.8(c)).

  아래는 **상태 기록**이며 "확정된 결과"가 아니다. 원자료 =
  `workspace/engine-port/results/s8_frontier/`. 4 job 전부 COMPLETED,
  probe 오류 0(870295/870296/870297 각 100 probe, 870301 24 probe).
  판정서는 아직 없다(`e1bcap_T8_870301_result.txt` 등 in-job 분석 산출물만
  존재). 반증 대상 목록은 `handoff-report/session_handoff_2026-08-02.md`
  "열린 항목" 1번.

  - **(1) batch-cap 실험(job 870301, T8 × {d16,d44} × cap{48,96,192} × 4
    seed) — 미검증**. 셀별 ITL "천장"은 **모델 성질이 아니라 설정 성질**로
    보인다: d16 ITL-p95가 cap 48→96→192에서 **49.7 → 59.0 → 62.0 ms**,
    seed-paired Δ(48→192) = **+12.2 ± 0.7 ms**(4/4 seed), d44는 **+1.3 ±
    0.7**. decode batch가 48 → 60–75로 자란 뒤 cap 192에서도 그 자리에서
    정지 ⇒ **cap 48만 실제로 구속**하고 96 이상은 도착·서비스율이 정하는
    자연 평형. ⚠️**scope: T8(순수 Transformer) 2셀만 측정 — 이 문구 없이
    인용 금지**(hybrid 이전 불가, 사유는 "방법론 게이트" #5). 부수 관측:
    같은 d16에서 TTFT-p50이 cap 48일 때 **612 ± 273 ms**인데 cap을 풀면
    **114 ± 11 ms**로 내려간다 ⇒ **cap 48이 TTFT를 5.4× 악화**시키고 있었고,
    같은 설정이 ITL은 좋아 보이게 만들었다. cap이 숨은 admission control로
    작동해 **두 축을 반대 방향으로 동시에 왜곡**한 것이다.
  - **(2) 4-arm 용량 knee(jobs 870295/870296/870297 + 기존 867231) —
    미검증**. 첫 교차 기준 knee(req/s):

    | arm | d16 | d24 | d44 | d54 | **d92** | 구속 셀 |
    |---|---|---|---|---|---|---|
    | T8 | 12.6 | 16.0 | 16.0 | 8.45 | **2.80** | d92 |
    | M8 | 5.60 | 5.60 | 5.60 | 4.20 | **2.80** | d92 |
    | Ha8 | 5.60 | 5.60 | 4.20 | 3.08 | **2.80** | d92 |
    | Hs8 | 9.09* | 12.6 | 9.09 | 8.45 | **2.80** | d92 |

    **d92가 네 arm 전부에서 구속 셀**이고 knee가 2.80으로 일치 ⇒ 공통
    off-cliff 상한 2.80 req/s. ★**기존 `results/s8_frontier/DESIGN.md`
    §4.3.5(b)에 *추측으로* 적어둔 "느린 decode arm(M8/Ha8)은 반대로 d16을
    먼저 잃을 것"은 지지되지 않는다** — 모델과 무관하게 d92의 prefill 16
    SM이 먼저 무너진다(사전등록에 추측으로 표시해둔 덕에 손해 없음).
    `*` Hs8 d16의 9.09는 견고하지 않다(한 점이 임계를 2% 초과하고 다음
    점이 회귀; 같은 셀 `arr=12.61 seed=1`에 미조사 이상치 1건).
  - **(3) 룽 분류 — 미검증**. 사전등록 사다리 {50, 60, 80} ms가 **네 arm
    전부에서 `HEADLINE-ELIGIBLE RUNGS = NONE`**이다(T8/Hs8 = 50ms
    CLIFF-HAZARD·60/80ms NONBINDING; M8 = 50ms ALWAYS-BINDING·60/80ms
    CLIFF-HAZARD; **Ha8 = 전 룽 `ITL-ALWAYS-BINDING`**). 2026-07-31에
    신설한 `ITL-ALWAYS-BINDING` 플래그가 **등록 몇 시간 뒤 Ha8에서 실제로
    발동**했다 — 없었다면 Ha8의 conjunctive goodput 0이 "어떤 D도 못 이김
    = 레버 net-negative"로 오독됐을 것이다(`degenerate_goodput_guard()`가
    결정 규칙을 차단).
  - **(4) E1 설계 위험(미검증, 판정 아님)**: (a) 사전등록 사다리가 as-run
    설정에서 **네 arm 전부 판정 불가**, (b) 그 as-run 설정 자체가
    **왜곡으로 증명됨**(cap 48, 위 (1)), (c) 제외 규칙(d92 knee 2.80, 전
    arm)이 **어떤 동작점에서도 decode-rich 끝을 제거** — C2의 레버가 사는
    바로 그 끝. ⇒ **E1이 `DESIGN.md` §4.3.5(b)가 사전등록해둔 분기
    "E1 as designed cannot reach this question"(설계상 이 질문에 도달할 수
    없다)으로 갈 위험이 높다.** 이는 **실패가 아니라 미리 적어둔 분기**이며,
    본 스윕에 GPU를 쓰기 전에 알아낸 것이다. 본 스윕 **미제출**.
  - ⚠️**as-run 상수 경고**: `PROBE_TARGET_S`는 문서화된 20이 아니라 **8**로
    867231·870295–297·870301이 전부 돌았다. **비교 런을 추가할 때 반드시
    8로 맞출 것**(현재는 submit 라인에 명시돼 있다).

  #### 2026-08-03 (같은 날 속행) — `g`가 이 격자에서 은퇴한다: 희석
  attenuation REFUTED, NO VERDICT 사유 확장, `A_free` 결함, arm 교락

  > ★★★**출처 구분(overclaim 금지).** A–D = **[AUDITED]**(claims-auditor가
  > 872077 원자료 telemetry 64 + bench 64를 독립 재분석해 판정, 정본 인용
  > 가능). E = **[UNAUDITED]**(result-analyst 산출, claims-auditor 미통과,
  > **정본 인용 금지** — 명시된 한 항목만 예외).

  - **(A) 메인 세션의 "희석 attenuation" 주장 — REFUTED.** 이날 앞선 회차의
    `E1_DECODE_REALIZED`(4–19%, 위 gate #7·`CONSENSUS.md` §1-25)를 근거로
    "`A_free(dD)=w_D·A(D)+(1−w_D)·A(108)` 혼합이고 그게 `g`를 1 쪽으로
    attenuate시킨다"는 보정 모형을 세워 Ha8 보정치 ≈1.62–1.70을 역산했다.
    **판정 = REFUTED**, 독립 증거 3줄: (i) **control-arm reductio** — 같은
    보정식을 T8에 적용하면 corrected g **21–29×**(A108∈{12,14,15}, b∈{0,2}) —
    정본 C2(SM16→92 2.36–2.91×)를 더 좁은 16→54 구간에서 **10배 위반**하고,
    모형이 요구하는 T8 d16 split-조건부 ITL p95(352–360ms)가 실측
    **30.67ms**와 10배 어긋난다. (ii) **de-engagement 직접 실험**(split
    라벨 토큰을 같은 셀 unsplit 분포에서 재추출해 engagement를 `f·w`로
    낮춤) — `A_free` 변화는 **1–11%뿐**(Ha8 d16 113.51→112.29 −1.1%, Ha8
    d54 107.45→95.74 −10.9%, T8 d16 28.31→26.97 −4.7%, T8 d54 15.38→15.02
    −2.3%). **w=0에서 g = Ha8 1.173 / T8 1.796** — 헤드라인이 거의 그대로
    남는다. (iii) 핵심 가정 "`A(108)` 셀 무관"이 실측에 반한다 — `A_free`
    형태를 하위 모집단에 적용 시 Ha8 ALL 1.068[0.947,1.190] / **SPLIT-only
    0.920[0.842,0.998]**(CI가 1 배제, **부호 반대**) / UNSPLIT-only
    1.146[0.997,1.296]; **T8 헤드라인 효과 전부가 decode SM 대비가 정의상
    0인 UNSPLIT-only 모집단에서 재현**(1.795[1.589,2.001] ≈ ALL 1.837).
    confound 유형 = #1(서빙 직접 측정을 오프라인 산술 모형으로 대체) +
    **#6**(항등식에서 파생된 `w`를 자유 모수처럼 나눔, "방법론 게이트" #6
    새 사례로 등재). **살아남은 것**: engagement가 낮다는 §1-25의 전제
    자체는 견고(스냅샷·event-driven·토큰 기준 세 계측기 교차확인) — 죽은
    것은 **보정**뿐. 집계 단위(게이트 #5) 부호는 확정(시간-몫 > 토큰-몫 ⇒
    시간가중 engagement는 과대평가)됐으나 2차항이 이미 모형을 죽인다.
  - **(B) 872077 — NO VERDICT 사유를 "규칙 모호"에서 "estimand 미식별"로
    확장.** 코드 사실: `pdmux_context.py:initialize_stream_groups`가
    `SM_COUNTS=[(108,0)]+divisions+[(0,108)]`를 하드코딩하고
    `multiplexing_mixin.py:773,792-794`가 prefill 비-in-flight 시 무조건
    `real_sm_group_num-1`=plain `(0,108)`로 되돌린다 ⇒ **"decode가 D SM에서
    돌았다"와 "prefill이 동시에 실행 중이었다"는 이 기판에서 같은 사건**이다.
    이 격자의 어떤 통계도 decode-SM 탄력도와 prefill 간섭을 분리 못 한다.
    §1-24(ITL 꼬리=monolithic prefill, 크기가 108−D에 단조)와 결합하면 `g`는
    사전에 **"decode-SM 탄력도 라벨을 단 prefill-SM 탄력도"**일 것이
    예상되고, 실측이 그와 일치(UNSPLIT-only서 헤드라인 재현)한다. **이는
    n으로 해결되지 않는 설계 결함이다.** 기록: 872077의 NO VERDICT 지위는
    **유지**하되 사유를 위와 같이 확장한다. **`g = A_free(d16)/A_free(d54)`는
    이 격자 한정 은퇴**(sticky partition 기판 수정 전까지 인용 금지). 블록
    8→12–16 증설 재실행은 **선행 금지**(참고: 현 mean/sd 유지 가정 시 상한
    ≤1.15 확률은 n=12 43%/16 55%/24 75%/32 87%/40 93%였으나, 기판 수정 후
    sd가 달라지므로 이 표는 사전에 무효). ⚠️**"Ha8에 decode-SM 레버가
    없다"는 CONFIRMED가 아니다** — 현 데이터는 그 질문에 답하지 못한다.
    **긴장 A(HE2 vs C2)는 전혀 닫히지 않았다.** `CLAIM_EVIDENCE_MATRIX.md`/
    `EXPERIMENT_ROADMAP.md`에도 이 상태(등급 임의 변경 없이, 닫히지
    않았다는 사실만)를 반영한다.
  - **(C) `A_free` 추정량 자체의 결함(E1 하네스 전반에 걸림).**
    `e1_m3_control.sbatch:281-306`을 읽은 결과: (i) blocking 필터가 작동하지
    않는다 — `PREFILL_BLOCK_TOK=1024`인데 이 워크로드는 요청의 4–5%만
    ≥1024 tok이고 그 prefill 토큰 몫은 23–26%뿐이라 **prefill 작업의
    74–77%가 필터를 통과**(실제 제거되는 ITL은 전체의 ~2%). §1-24가 확정한
    monolithic prefill stall(원문은 "d92만 오염"으로 한정)이 **d16–d54까지
    오염 범위가 확장됨**을 기록한다 — T8 d16 꼬리를 만드는 요청 input은
    221–804 tok로 전부 임계 아래. (ii) 요청의 27.5–29.5%가 output≤25
    토큰이라 내부 p95가 사실상 max ITL로 퇴화 — 평균의 선형 혼합 항등식이
    극단 분위수에 성립하지 않는다(비단조 응답 Ha8 d54
    107.45→101.89→102.84→97.75가 실증). ⇒ **`A_free`는 blocking 제거본이
    아니라 대부분이 monolithic-prefill stall로 이루어진 극단꼬리 통계다.**
    추정량 교체가 후속 로드맵 항목(아래).
  - **(D) arm 간 비교의 미제거 교락.** 공통 rate 2에서 T8 conc 12.8/decode
    batch 4.5/ITL p50 ~11ms 대 Ha8 conc 30.6/batch 15.8/~30ms. 양 arm 모두
    off-cliff 평탄역(0.88–1.07)이라 metric cliff는 아니나, decode batch
    size가 decode step의 memory-bound/compute-bound 여부를 결정하는
    공변량이라 arm과 완전 교락 ⇒ "attributable to the arm" 문구는 **현재
    허용되지 않는다**(통제는 arrival rate가 아니라 realized concurrency/
    decode batch를 맞춘 rate). ⚠️이 교락이 관측 *방향*을 설명하지는
    않으므로(batch 큰 쪽이 오히려 무반응) **대안 설명이 아니라 미제거
    교락**으로 기록한다. 부수: d16은 `sm_group_num:3`, d54는 4(guard
    row)로 셀마다 green context 수가 다르고, d54에서 `decode_sms==44`는
    전 telemetry에서 미관측(guard row 미선택) — 행동 교락은 아니나 셀 간
    차이로 기록.
  - **(E) [UNAUDITED — 정본 인용 금지] result-analyst의 decode-empty 진단.**
    산출 스크립트 `results/s8_frontier/m3_decode_empty.py`(재실행 가능).
    claims-auditor 미통과이나 (B)와 독립적으로 수렴하는 부분이 있어 기록
    가치가 있다. ★희석의 원인은 decode 공백이 아니라 **prefill 부재**다
    (`E1_DECODE_REALIZED`는 decode-active 시간에 조건부라 decode-empty는
    정의상 분자·분모 어디에도 안 들어감; 실측 기여 T8 −0.0006±0.0046/Ha8
    −0.0001±0.0028 = 0, block-paired 분해에서 gradient의 100%가 prefill
    점유율 gradient로 설명됨, 잔차 CI가 0 포함). 부하창 내 decode-empty는
    **0.6–1.3%뿐**(원자료의 15–19%는 클라이언트 warmup→dataset 준비 갭 +
    종료 후 꼬리로 인한 **측정창 아티팩트** — 클라이언트 `duration`으로
    앵커한 부하창에서 분석해야 함). ★게이트 #6 필드 감사 — 죽은 telemetry
    필드 4개(코드 근거 포함): `decode_ready_queue_depth` 항등 0(dual-worker
    가드 안에서만 채워지는데 872077은 `architecture=="legacy"`),
    `active_decode_sequences`≡`decode_running_batch_size`,
    `decode_idle_ratio`/`prefill_idle_ratio` 항등 0.0(`controller.py:55`에서
    선언만 되고 대입 없음), `running_batch_occupancy`≡min(1,drb/48),
    `prefill_admission_blocked`≡(pqd>0∧pab==0). 집계 단위(게이트 #5)
    재확인: decode-empty 시간몫이 시간가중 0.0056–0.0127 vs 개수
    0.333–0.424(30–60× 차이), 분해 비율(A 지배)은 세 단위 모두 강건.
    arm별 realized 천장이 워크로드 성질(ΣTTFT/decode-busy, T8
    0.120–0.154/Ha8 0.381–0.558)이라는 관측은 클라이언트 측 양이라
    텔레메트리 계측 문제에 면역 — `g`의 arm 간 비교에 D의 교락과 별개인
    추가 축. UNDETERMINED: realization gradient의 크기(스냅샷 dt p95
    195–805ms가 prefill span과 동 자릿수라 물리량으로 인용 금지) 및
    t_pa 기반 prefill span 증가와 클라이언트 TTFT 증가의 미해소 모순(해소법
    = prefill batch start/end 이벤트 직접 emit 후 재측정). ★**단 이 항목
    하나만은 claims-auditor와 독립적으로 수렴해 [AUDITED]로 인용 가능**:
    부하를 올려 engagement를 높이는 방향은 데이터가 지지하지 않는다
    (decode는 이미 부하 중 ~99% busy; rate를 올리면 prefill·decode가
    비례해 늘고, 셀 간 `w` 차이조차 arrival이 아니라 prefill이 108−D에서
    느려져 생긴 것; split-eligible iteration 수는 셀 무관 194–214로 거의
    일정).

  **다음 실험 gate(2026-08-03 1차 속행 시점 계획)**: 감사자 권고 순서 —
  1–3 = 이 절 자체를 정본에 기록(완료, GPU 0), 4 = `PDMUX_STICKY_PARTITION`
  구현 + correctness gate(engine-porter, ~0.25h — **구현 완료 ≠ 성능 주장
  성립**), 5 = sticky 격자 1회(872077 동일 설계 8 block, ~3.0h; 872077이
  non-sticky 대조군 ⇒ 총 ~3.3 GPU-hour), 6 = 블록 증설은 **4·5 이후에만**.
  ★**4는 아래 "2026-08-03(2차)" 소절에서 완료됐다 — 5는 여전히 미제출**.

  #### 2026-08-03 (같은 날 2차 속행) — estimand 이관 완료 + `PDMUX_STICKY_PARTITION`
      구현·correctness gate 통과

  > 출처 구분(overclaim 금지). **(I)**는 claims-auditor의 산출이며 (1)(2)는
  > **[AUDITED]**, **(3)만 [UNAUDITED]**(감사자가 이번 턴에 새로 생산한
  > 것을 감사자 자신이 감사한 형태 — 별도 확증 전 정본 인용 금지). **(II)**는
  > engine-porter의 **구현 사실**이며 성능 판정이 아니다. **(III)**은
  > 사전등록이며 결과가 아니다. 원자료·코드:
  > `results/s8_frontier/m3_conditional.py`(신규, 미추적),
  > `src/multiplex/multiplexing_mixin.py`(+151/−17), `tests/
  > test_sticky_partition.py`(신규), `results/sticky_smoke/`(신규). 전문
  > `results/s8_frontier/DESIGN.md` §4.3.10–4.3.12.

  **(I) `A_free`를 대체하는 조건부 per-token 추정량.**

  1. **[AUDITED] 정의.** 단위 = 개별 ITL 구간 1개(요청별 내부 집계 없음 —
     이것이 `A_free`의 두 병리, 요청별-p95 이중극단·outlen≤25 퇴화를 피하는
     전부). 라벨: 구간 `[a,b]`(client 시계, 스냅샷 계단함수 기준)에
     `split_frac(a,b) = (decode_sms==D였던 시간)/(b−a)`,
     **SPLIT := ≥0.90 / UNSPLIT := ≤0.10 / AMBIGUOUS := 사이(0.3–1.4%) →
     강제분류 없이 양쪽 배제**. 통계량은 셀-블록별 **직접 분위수**(primary
     `p95(SPLIT)`, secondary `p50(SPLIT)`, **control `p95/p50(UNSPLIT)`**,
     mean 병기). ★**UNSPLIT control이 핵심 안전장치**: 두 셀 모두 108 SM이라
     decode-SM 대비가 **정의상 0** — 여기서 비가 1이 아니면 그 차이는
     decode SM이 만든 게 아니다. 대비량은 block 내 paired, 8-block
     block-clustered t-구간(`TCRIT[8]=2.365`, percentile bootstrap 금지 —
     n=8 커버리지 79.8% 기지). 회피 근거: T8 d16 pooled per-token p95
     **11.60**인데 `A_free`=**28.31**(2배 이상 바깥 꼬리); 요청의
     27.5–29.5%가 outlen≤25; `A_free` outer-p95는 요청 ~9.6개에 얹히는 반면
     새 추정량은 셀-블록당 1,390–7,045 토큰. ⚠️**이 추정량은 `A_free`보다
     잘 정의됐을 뿐 사전등록된 SLO 항이 아니다** — `A_all`(등록 SLO 항)은
     계속 병기.
  2. **[AUDITED] client↔telemetry 시계 정렬.** `phase_marker(phase==
     "benchmark")` 앵커 → `replay_arrivals(seed,rate,n)`을
     `e1_m3_control.sbatch:284-287`과 bit-identical 복제 → client 계단함수
     구성 → telemetry `decode_running_batch_size`와 Pearson r 최대화.
     **lag이 13.9–35.9s인 이유**: `benchmark` 마커는 서버가 처음 본
     **warm-up 요청**에 발화하고(`multiplexing_mixin.py:386-398`) 이어
     데이터셋 토크나이즈가 실제 probe 시작을 미룬다 ⇒ **마커는 탐색
     앵커일 뿐 probe 경계가 아니므로 `load_telemetry`는 `phase==
     "benchmark"` 필터를 걸면 안 된다**. 감사자 자기정정: "정렬 약한
     probe 1개(T8 d16 b1, r=0.882)"는 24개 spot-check 결과였고 전수
     64 probe에서는 **2개**(T8 d16 b6, r=0.930 추가) — **보고 수치는 전부
     8 block 전수 계산이라 변경 없음**, 정정 대상은 서술뿐. **규칙**:
     `ALIGN_R_MIN=0.95`, 미달은 ALIGN-WEAK로 flag하고 포함본/제외본 둘 다
     보고(조용한 drop 금지) — 근거: (a) `r`이 block 내 paired라 probe
     하나 배제 = block 전체 소실 = `n_indep` 8→7, (b) 오정렬은 라벨을
     무작위화해 두 하위모집단을 pooled로 끌어당길 뿐(분리를 약화만 시킴,
     편향이 보수적), (c) 실측 LOO(`sp_p95` 비): T8 full 1.688 → b1 제외
     **1.822**, b6 제외 1.640 ⇒ **가장 약한 block을 빼면 대비가 오히려
     커진다**. Ha8 full 1.340, LOO 1.219–1.375(전 probe r≥0.993).
  3. **[UNAUDITED — 감사자 자신이 이번 턴에 새로 생산, 자기 산출을 자기가
     감사한 셈. 정본 인용 전 별도 확증 필요] Blocking 필터 임계 스윕.**
     `PREFILL_BLOCK_TOK` 1024→512→256→0(keep_frac/pooled-p95/`A_free`-form,
     ms): Ha8 d16 0.979/98.66/**113.51** → 0.827/33.50/**34.63**; Ha8 d54
     0.960/88.01/**107.45** → 0.756/33.01/**35.11**; T8 d16
     0.994/11.60/**28.31** → 0.928/11.44/**11.65**; T8 d54
     0.984/14.01/**15.38** → 0.877/11.38/**11.59**. **무릎(knee)이 없고**,
     임계 0에서 d16-vs-d54 대비가 두 arm 모두 **소멸**(Ha8 0.986, T8
     1.005) ⇒ 임계는 자유 모수가 아니라 **답을 정하는 손잡이**이고 사후
     선택 금지(임계 0은 estimand를 "prefill이 전혀 없던 조용한 순간의
     ITL"로 바꾸고 그 선택이 부하와 상관되므로 선택 편향). ★이 표는
     telemetry·시계정렬을 전혀 쓰지 않아 조건부 분석에 대한 계측 반론에는
     면역이다(그 점만은 강함).

  **권고**: `A_free` **은퇴**(임계 재조정 아님), `A_all` 유지·병기.
  **primary 라벨 = realized partition**(`decode_sms==D`, persisted state
  variable이라 조밀 idle-spin 스냅샷에서도 견고). secondary 라벨(prefill
  overlap, `prefill_overlap_frac(a,b)=구간 중 prefill_active_batch_size>0
  였던 시간 몫`, BLOCK-FREE := ≤0.0)은 872077이 `PDMUX_TRACE_FORCE_
  PREFILL=0`이라 **계측 결손**(prefill-active가 wall의 4.9%인데 스냅샷의
  0.07%): SPLIT 라벨 토큰 중 "overlap-free"로 나오는 비율이 Ha8 d16/d54
  66.3%/38.8%, T8 76.1%/27.1%(물리적으로 SPLIT면 prefill in-flight여야
  하므로 이 불일치분이 곧 계측 결손).

  **재구현 시 갈라지는 3곳**(코드 주석에도 명시): ① `replay_arrivals`는
  sbatch와 bit-identical, ② `load_telemetry`는 `phase=="benchmark"` 필터
  **미적용**, ③ `report_deengagement`는 rng를 전체 중첩 루프에 하나만
  생성해 **루프 순서 의존**(보고값 재현엔 순서 고정 필요; 프로덕션 흡수 시
  `(arm,cell,block,f)`별 seed로 바꾸면 값이 MC 잡음만큼 움직임).

  **(II) [구현 사실 — 성능 주장 아님] `PDMUX_STICKY_PARTITION` 구현 완료,
  correctness gate 통과.** 전부 `src/multiplex/multiplexing_mixin.py`
  (+151/−17, 4 hunks).

  - **되돌림 경로 5곳 전수 처리**: `adjust_stream_groups`:904의 핵심
    되돌림(decode busy·prefill 부재 → plain (0,108))을
    `if not running_batch.is_empty() and (split_prefill_batch or
    sticky_partition_enabled)`로 가드해 sticky ON에선 이 경로에 도달하지
    않음. :906의 decode-**empty** → plain (108,0)은 **의도적 미변경**.
    `event_loop_pdmux`:1010–1012 트리거도 미변경(#2와 일관성 유지 주석).
    `event_loop_pdmux_coord` + `PDMUX_LA_COORD` 조합은 init에서
    `RuntimeError`로 거부(반쪽 sticky 방지). SLO 분기·v7 `_tgt`는 prefill
    span 중에만 도달하므로 미변경.
  - **decode-empty 시 index 0으로 release(hold 아님)** — 이유 3: (a)
    `E1_DECODE_REALIZED`가 decode-active 시간 가중이라 이 구간은 가중치 0,
    (b) 보호할 decode 작업이 없어 D SM을 prefill로부터 놀리기만 함(upside
    0), (c) 경로 #3을 유효하게 유지해 기전의 두 반쪽을 일관 유지. smoke
    실측: decode-empty 스냅샷이 양 arm 모두 `(0,108,0)`(OFF 10812 / ON
    12046) ⇒ **두 arm은 decode-busy 모집단에서만 다르다**.
  - **OFF 바이트 동등**: 유일 변경이 `split_prefill_batch or
    sticky_partition_enabled`이고 flag OFF면 short-circuit으로 patch
    이전과 정확히 동일. `test_off_matches_pre_patch_selector`가 **독립
    재구현한 pre-patch selector**와 3 config × decode_bs{0,1,4,47,48,96} ×
    {prefill 무/유} 전 격자에서 동등 assert. 텔레메트리 코드 미변경.
  - **cudagraph 보존**: `cuda_graph_runner.py`가 `f"{stream_idx}_{bs}"`로
    키잉하고 `capture()`가 모든 stream-group 인덱스를 캡처 ⇒ division
    인덱스 유지 시 캡처된 그래프 재생, eager fallback 없음.
  - **모드 상호작용**: sticky ON은 init에서 `PDMUX_LA_COORD`,
    `PDMUX_SLO_SCHED`, `PDMUX_FIXED_DECODE_SM_FILE`, 비-`fixed`
    `PDMUX_R2_POLICY`, `real_sm_group_num<3`을 `RuntimeError`로 거부.
    허용: `PDMUX_R2_POLICY=fixed`(target 인덱스를 init에서 해석), 무정책,
    `PDMUX_DUAL_WORKER`, `PDMUX_TRUE_DUAL_WORKER`.
  - **텔레메트리 = realized 확인**: `_dual_worker_sync(stream_idx)` →
    `observe_scheduler` → `arbiter.select_partition(stream_index)` →
    `metrics()`가 `arbiter.sm_counts[arbiter.stream_index]` 반환 — CUDA
    스트림을 고르는 바로 그 인덱스이므로 `decode_sms`는 라벨이 아니라
    **decode가 실제로 돈 green context의 SM 수**(그래서 patch 이전 런이
    셀 라벨이 아닌 D108을 92–96% 보였던 것). ⚠️**미재검증 잔여 스코프**:
    "green context를 `create_greenctx_stream_by_value`로 만들면 하드웨어가
    그 SM 수를 실제로 부여한다" 단계는 이 패치가 바꾸지 않았고 재probe하지
    않았다.
  - **correctness gate 결과**: (1) CPU 회귀 sticky OFF **PASS** — 40
    tests(기존 28+신규 12), `sync_engine_tree.sh` 후 manifest SHA-256 일치
    (`multiplexing_mixin.py =
    59eaafb4ac61cc09ad8d28f663c495cf6a0e850435c15b873547b8a6e5a7d20a`,
    동기화 스크립트 변경 불필요). (2) sticky 단위 테스트 12건 **PASS**
    (설치된 런타임에서 로드하므로 sync 실행 여부까지 assert). (3)
    thread-local role patch 정상. (4) **GPU smoke PASS** — job **872800**,
    `amd_a100nv_8`/gpu38, ~9분. Ha8=Zamba2-7B-Instruct, d16
    (`PDMUX_R2_POLICY=fixed`, `PDMUX_R2_FIXED_DSM=16`), server args를
    `e1_m3_control.sbatch`에서 그대로 복사, 두 부팅이
    `PDMUX_STICKY_PARTITION`만 다름. 고정 프롬프트 6개 greedy(`temperature
    0`, `max_new_tokens 48`) → **OFF/ON 출력 6개 전부 byte-identical**.
  - **★realized 관측(게이트가 아니라 관측)**: probe = ShareGPT 100
    prompts, rate 2, seed 1, `PDMUX_TRACE_FORCE_PREFILL=0`,
    `e1_pin_check.py:compute_decode_realized`.

    | arm | `E1_DECODE_REALIZED` | decode-active 히스토그램 |
    |---|---|---|
    | sticky OFF | **0.0839** | D108 66.7s, D16 6.1s |
    | sticky ON | **1.0000** | D16 164.3s (D108 부재) |

    OFF는 기존 동작 재현(872077 Ha8 d16 = 동일 추정기로 0.104–0.127). ON은
    **1.0000, 사전등록 0.90 초과이며 이를 맞추려 튜닝한 것 없음**. 스냅샷
    교차확인: decode-busy 스냅샷이 ON에서 `(idx1,92,16)` 139/139, OFF에서
    `(idx2,0,108)` 146 / `(idx1,92,16)` 8 — ON에서 index 2 진입 없음.
    ⚠️**해석 없는 주의**: decode-active wall time이 두 arm에서 다르다
    (72.8s vs 164.3s) — arm이 서로 다른 파티션에서 decode를 돌리기
    때문이다. **n=1, 미반복, 성능 측정 아님.** ★**구현 완료 ≠ 성능 주장
    성립.** throughput/latency/goodput/`g`에 대한 어떤 진술도 허용되지
    않는다. 이 패치가 주장하는 것은 **`E1_DECODE_REALIZED`가 항등식이기를
    멈추고 진짜 게이트가 됐다**는 것뿐이다.
  - **신규 파일**: `tests/test_sticky_partition.py`,
    `results/sticky_smoke/sticky_smoke.sbatch`,
    `results/sticky_smoke/stksmoke_Ha8_d16_872800_result.txt` + smoke
    아티팩트(telemetry.jsonl 2개, 각 ~14MB, git 미추적 권고).
    `pdmux_context.py`는 미변경(`[(108,0)]+divisions+[(0,108)]` 유지 —
    sticky는 trailing 그룹을 제거하지 않고 회피).

  **(III) sticky 런 사전등록 — 기록하되 `G_LEVER`/`G_FLAT`는 비워 둠.**

  - **(a) 872077 소급 적용의 지위 = DIAGNOSTIC 전용, 판정 금지.** 이
    추정량은 872077을 본 뒤 선택됐으므로 그 데이터로 판정을 채택하면
    confound(re-score vs re-tune)다 — §4.3.8(c)의 RULE_BOUNDS
    forward-only와 같은 논리. 소급 적용은 "estimand 미식별"이라는 **설계
    판정의 근거 자료**로만 쓴다. `m3_conditional.py`가 실행 끝마다 이
    문장을 출력한다.
  - **(b) primary 1개 고정**: `p95(SPLIT tokens)` 비를 primary로 선언,
    나머지(p50/mean/UNSPLIT)는 secondary — 6개 통계량이 있어 선언 없이
    가면 사후 선택(multiplicity)이 열린다.
  - **(c) 게이트 4종**(등록 전 각각이 estimand와 논리적으로 독립인지 재확인
    — 방법론 게이트 #6/#7, `MIN_PA_SNAPSHOTS`가 estimand의 여집합을 셌던
    실수 반복 금지): `ALIGN_R_MIN` 0.95(flag-only) / `E1_DECODE_REALIZED`
    ≥0.90 per cell-block(sticky에서 항등식이 아니게 되므로 비로소 진짜
    게이트) / `AMBIG_FRAC` 상한(872077 관측 0.003–0.014) / `MIN_N_SPLIT`
    cell-block당 하한(872077 관측 1,390–7,045, sticky에서 늘어야 정상).
  - **(d) ★`G_LEVER`/`G_FLAT` = 미결정.** 기존 1.5/1.15는 `A_free`(이중
    극단) 스케일 값이고 새 추정량은 per-token 분위수라 스케일이 다르다 —
    옮기는 것 자체가 사후 재단. 살아있는 논거: 새 추정량이 C2와 같은
    축(per-token ITL 분위수)이므로 `G_LEVER`를 C2 측정범위(**2.36–2.91×**,
    prefill 16 SM 고정·SM16→SM92)에 묶는 것이 원리적으로 정당화 가능하나,
    E1은 D 범위가 16→54로 좁고 **complementary**(prefill=108−D가 함께
    움직임)라 C2 값을 그대로 쓸 수 없다. **결정 근거를 문서에 남긴 뒤
    sticky 런 제출 전에 사전등록해야 할 열린 항목**으로 기록한다.
  - **(e) sticky에서 달라지는 것 2가지 — 미리 등록**: (i) UNSPLIT
    하위모집단이 비거나 매우 작아진다(목적) ⇒ `un_*` 행을 필수 산출로
    요구하지 말고 정의 불가 시 NaN, 판정은 SPLIT 계열로(UNSPLIT이 여전히
    크면 sticky 미적용이므로 `E1_DECODE_REALIZED` 게이트가 먼저 잡음).
    (ii) **primary 모집단이 `SPLIT ∧ BLOCK-FREE`로 이동**한다(sticky
    이전엔 두 라벨이 사실상 중복이지만 이후엔 decode-SM을 prefill 간섭에서
    분리하는 유일한 모집단) — 이 이동을 미리 등록해야 사후 선택이 아니다.
  - **(f) 판별 예측(주장 A 최종 검정, §4.3.9 승계 불변)**: sticky에서
    꼬리가 prefill 주도면 Ha8≈0.92 / T8≈1.85 쪽으로, 주장 A가 옳았다면
    Ha8≈1.6 쪽으로. CI 비중첩 ⇒ **8 block으로 구분 가능**. 블록 증설은
    이 판별 **이후에만** 검토.
  - **(g) 열린 설계 쟁점(claims-auditor 판단으로 남김, 결론 내지 않음)**:
    primary 모집단이 `SPLIT ∧ BLOCK-FREE`로 가면 §4.7.1의 trace-force 금지가
    **구속 조건**이 되는데(prefill overlap 라벨의 계측 결손 때문),
    §4.3.8(h)의 SCHED-only 분해가 trace-force의 *시스템* 효과를 mean
    +0.001(sign 4+/2−)로 재귀속했으므로 **trace-force ON 재허용 여부는
    sticky 설계에서 다시 판단할 사안**이다.

  상세 전문 `results/s8_frontier/DESIGN.md` §4.3.10(estimator)–§4.3.11
  (implementation)–§4.3.12(pre-registration), `reports/CONSENSUS.md`
  §1-27.

  #### 2026-08-03 (같은 날 3차 속행) — C2 → `G_LEVER` 앵커 경로 폐기
      [AUDITED] + D=54 측정 취소[일부 미감사, 코드/로그 직접 검증]

  > `c2_anchor.py`(신규, 미추적)로 시도한 앵커 도출을 claims-auditor가
  > 감사해 **주장 1만 생존, 2·3·4·5 전부 반증/미지지**로 판정했다.
  > **`DESIGN.md` §4.3.12(d)의 "미결정"은 그대로 유지된다** — 이 시도는
  > 그 항목을 닫지 못했다. 원자료·코드: `results/s8_frontier/
  > c2_anchor.py`(§7 재현에 필요), `results/s8_scaleup/
  > NOTES_D54_ANCHOR_2026-08-03.md`(미추적). 전문
  > `results/s8_frontier/DESIGN.md` §4.3.13–4.3.14.

  **A. §0 결정적 발견(신규, 가장 중요) — 축이 같다는 전제가 실측으로
  거짓.** 같은 arm·같은 서버 플래그(`s8_sweep.sbatch:141-146` vs
  `e1_m3_control.sbatch:212-218` 직접 대조로 동일 확인)·매칭 batch에서
  **"decode 16 SM"의 per-token ITL이 2.6× 다르다**:

  | 출처 | 조건 | decode@16SM p50 |
  |---|---|---|
  | C2 865493 (SPLIT, batch bin 5) | prefill 16, decode 16 | **28.79 ms** |
  | C2 `FINDINGS_8B` §2 batch=4 | prefill 16, decode 16 | 28.48 ms |
  | **872077 (E1 격자)** d16, decode batch 4.5 | prefill 92, decode 16 | **11.09 ms** |

  둘 중 하나가 거짓: **(i)** 872077의 `decode_sms==16`이 실제 16-SM 실행이
  아니다 — `DESIGN.md` **§4.3.11이 명시적으로 미검증으로 남긴 잔여
  층**(green context 생성 → 하드웨어 SM 부여를 재프로브 안 함), 또는
  **(ii)** C2의 28–31 ms가 decode-SM 비용이 아니라 그 셀 배치(`[16,16,
  76-idle]` + 상시 동거 keepalive prefill)의 성질이다. ★**이 층이 이제
  872077 전체와 이번 세션의 sticky 결과가 딛고 선 바닥**이다. (i)이 참이면
  **Stage 0급 정정**이 된다. **최상위 열린 항목**으로 기록 — confound
  #1(sim→serving 전이 실패)의 **실측 대 실측** 재현이자 confound #9(라벨≠
  실현)의 변종이다.

  **B. 주장 1(실현 검증) — 결론 CONFIRMED, 서술 2건은 정본 정정 대상.**
  비순환 재검정(구간 집합을 쓰지 않고 telemetry만으로 client `t0` 기준
  3분할): 865493 측정창 @D **0.989–1.000** / @108 0.000–0.007, warmup도
  @D≈1.000, 창 밖 drain은 @108 0.716–0.895. 865533은 측정창에서도 @D
  0.30–0.73 **FAIL**. **정본 정정 2건(필수)**: (1) **"파티션 활성률
  0.66–0.93"은 `results/s8_scaleup/realized_pin_check.py`의 스냅샷
  개수 가중 + `phase=="benchmark"` 필터 값**이다. 시간가중 all-busy는
  0.803–0.958로 **정본 범위와 일치하지 않는다** — 인용 시 "count-weighted
  0.66–0.93 / 측정창 시간가중 0.99+"로 병기(방법론 게이트 #4 위반 사례,
  이 감사 한 건에서만 3회). (2) **108 SM 시간은 warmup이 아니라 창 밖
  drain 전용**이다(warmup도 @D≈1.000). 기전 서술을 고치면 결론은 **더
  강해진다**.

  **C. 구조적 함의 — sticky 런에는 음성대조가 정의상 없다.** in-window
  residency와 UNSPLIT 표본은 구성상 여집합(SPLIT+UNSPLIT+AMBIG ≈ 1) ⇒
  **`E1_DECODE_REALIZED ≥ 0.90`을 통과하는 런에는 음성대조 표본이 존재할
  수 없다**(865493 UNSPLIT n=**0**; sticky ON smoke는 D108 **0초**).
  `DESIGN.md` §4.3.12(e)(i)를 이 사실로 확장하고, 사전등록에 명시해야
  한다 — `MIN_PA_SNAPSHOTS`가 estimand의 여집합을 셌던 것과 같은 구조
  (방법론 게이트 #7).

  **D. 주장 2 — primary `p95` 유지, p50 전환 REFUTED.** 관측(UNSPLIT
  집단 d16/d44 비가 p95에서 T8 1.662·Hs8 1.374, p50은 0.997–1.002)은
  재현되나: **기전 귀속 REFUTED** — monolithic prefill stall이 아니라
  **파티션 전환 인접 구간**(`>3×median` 사건의 83–87%가 전환 0.5s 이내,
  배제 시 1.662→1.041 단조 감쇠; `prefill_active>0` @108 = **0.0000**이라
  M4 서명으로 설명 불가). **처방 REFUTED, 세 겹** — ① 같은 배제를
  SPLIT(=estimand)에 적용해도 p95는 **≤2%만** 이동(오염원이 전이 안 됨)
  ② 오염 기전이 sticky ON에서 **소멸** ③ ★**872077 실측 `T8 sp_p50 =
  0.996 [0.990, 1.002]`** ⇒ p50으로 바꾸면 **양성대조조차 1.00**이라 어떤
  `G_LEVER>1`에서도 발화 불가 = 캠페인을 **구조적 NO VERDICT로 확정**하는
  처방, 게다가 872077에서는 같은 p95 음성대조가 **반대 방향**으로 깨진다
  (`T8 un_p95 = 0.880 [0.853, 0.907]`, `Ha8 un_p95 = 1.377 > sp_p95
  1.340`). forward-only 구속으로도 구제 안 됨: RULE_BOUNDS가 정직할 수
  있었던 이유는 "872077에서 발화하지 않는 쪽"이었기 때문인데, p50 전환은
  반대로 T8 헤드라인 1.688을 0.996으로 **지우는** 방향이고 데이터를 본
  뒤 제안됐다(confound #6/#12). ⇒ **primary = `p95(SPLIT)` 유지, p50은
  secondary, 전환-근접 진단은 게이트가 아닌 진단으로 병기.**

  **E. 주장 3·4·5.** **주장 3**(편향 부호=하한) NOT-YET-SUPPORTED —
  (ii) 가산/곱셈 **미식별**(Hs8은 비 일정, T8은 차·비 모두 불일정; 두 job이
  keepalive 길이·개수를 **동시 변경** = confound #10), (iii) **증거가
  항을 0으로 만든다** — C2 분할 셀은 **전부 1410 MHz 고정**이고 클럭이
  떨어지는 건 **무분할 np뿐**(T8 median **1290**, p10 1275) ⇒ **E1/sticky
  (prefill 108−D + decode D = 108 전부 가동)가 낼 throttling 비용을 C2는
  안 낸다**는 **반대 방향 경고**로 기록(비에 대한 효과는 미측정). (i)
  span 절단만 생존. **주장 4**(`G_LEVER`=1.41) REFUTED — "한 격자 스텝"은
  사후 정당화(끝점 선택만으로 [1.41, 2.40] 전부 도달 가능), 부차값 2.02는
  게이트-FAIL job(865533) 4셀을 포함해 내적 비일관, 1.41이 872077 T8 CI
  하한 1.227을 가로지른다. **주장 5**(`G_FLAT`=1.25) 방법 PLAUSIBLE(§4.3.8
  선례와 같은 종류의 논거, re-score 저촉 아님) / **숫자 REFUTED** — LOO
  8개 실측 시 실제 t95 반폭 0.303 ⇒ 1.30이고 **Ha8 점추정 1.340 > 1.30**
  이라 규칙이 자기 데이터에서 뒤집힌다. 더 큰 문제: sticky가 `n_split`을
  한 자릿수 이상 늘려 **사후 sd가 떨어지므로** 사전-sticky sd 기반
  `G_FLAT`은 **관대해지는 방향 = 귀무 오수용** 편향(정본이 반복해 당한
  방향). **`n_indep` = 1**(865493↔865533은 keepalive 설정이 달라
  replicate가 아니라 다른 조건 — **세 번째 pseudo-replication**).
  **regime 불일치 정정**: 872077의 12.8은 **concurrency**, C2의 12.7은
  **decode batch**다 — 같은 단위로 맞추면 **T8이 2.8× 어긋나고**(11.2–
  12.7 vs decode batch 4.5) **Ha8이 잘 맞는다**(13.0–13.4 vs 15.8, 앞선
  서술과 정확히 반대).

  **F. 남은 경로(계획으로만 기록, 미실행).** `G_LEVER`: C2 앵커 폐기.
  감사자 대안 (α) sticky 파일럿의 T8 양성대조 실측 분포 + 효과크기 논증,
  (β) 절대 임계 제거하고 arm 간 대비 `g_T8/g_Ha8`의 block-paired CI가
  1을 배제하는지(스케일 자유, batch-매칭 rate 필요) — ★**둘 다 감사자
  발안 ⇒ 독립 사전등록 필요**(감사자가 자기 발안의 승인 주체가 될 수
  없다). `G_FLAT`: 사후-sticky 파일럿에서 sd 측정 후 결정하되, 순수 power
  임계 대신 TOST 동등성 마진으로 바꾸고 근거를 실질 유의성으로 논증(감사자
  발안 ⇒ 독립 사전등록). 파일럿 block은 본 런에 재사용 금지. 감사자 제안
  **게이트 S1**(≈1 GPU-시간, 미실행): T8 단일, d16+d54, sticky ON,
  `PDMUX_R2_POLICY=fixed`, 워크로드 2종을 같은 부팅 계열에서 — (a) C2
  복제(폐루프 conc16, 고정 1024/512, keepalive 8) vs (b) E1 복제(ShareGPT
  rate 2), 2 block. 판정 3갈래: C2 재현 O + E1 ~1.0 → 워크로드/batch 귀속
  / 둘 다 ~1.0 → C2 28–31 ms가 셀 배치 성질(C2 스코프 문구 수정 사안) /
  둘 다 큼 → 872077 d16 라벨 미실현(Stage 0급 정정). 필수 계측: 셀별 SM
  clock, realized decode batch, `prefill_active` 동거율. ⚠️ 하네스 확장
  필요(`s0dc_client.py`는 폐루프 합성, E1은 ShareGPT trace replay + 개루프
  rate ⇒ 같은 서버에 두 클라이언트를 순차 투입하도록 `run_cell` 확장
  필요, E1 클라이언트 CLI 계약 미조사).

  **G. D=54 측정 취소(jobs 872920/872921) + 재현성 결함.** 기록
  `results/s8_scaleup/NOTES_D54_ANCHOR_2026-08-03.md`(미추적 신규,
  하네스 파일 `s8_sweep_d54.sbatch`·`pdmux_p16_d54.yml`·
  `d54_block_ratio.py`·`runtime_source_manifest_d54.sha256`도 미추적) —
  **취소됐으나 설계(d16+d54 동일 캠페인, 4 block, 셀 순서 block 패리티
  교대)는 유효하므로 재사용 가능.** **G-1 [미감사, 코드·로그로 직접 검증
  가능] keepalive 토큰 초과 = s8_scaleup 재현 불가 요인.**
  `s8_keepalive_prompt_224.txt`가 **1794 토큰**(★2026-08-16 정정, 구
  1793 — 표기만 정정, 아래 dated정정3)인데 `CTXCAP = CTX+OUTTOK+
  256 = 1792` ⇒ 모든 keepalive가 `HTTP 400`. **865493은 byte-identical한
  같은 파일로 `keepalive_done≈500, keepalive_errors=0`**이었고 두 srv.log
  모두 `CTXCAP=1792`를 찍는다 ⇒ **2026-07-27 이후 엔진 트리 churn으로
  context-length 거부가 엄격해졌거나 off-by-one이 이동**(원인 미규명,
  자명한 수정 `KEEPA_REPS ≤ 223` 미적용). ⇒ **`s8_scaleup` 캠페인 전체의
  재현 불가 요인**이므로 해당 캠페인을 참조하는 곳에 경고를 남긴다. 결과:
  co-residency가 865493의 ~90–100% → **31–34%** 붕괴, 측정된 전 셀
  `REALIZED_PIN` FAIL(T8 blk1 d16 0.316 / d54 0.628, Hs8 d16 0.342 / d54
  0.577).

  ⚠️**dated 정정(2026-08-15, doc-steward — 메인 세션 mtime 직접 확인, 원문
  미덮어쓰기, 새 성능 판정 아님)**: 위 "865493은 byte-identical한 같은
  파일로 성공"·"2026-07-27 이후 엔진 트리 churn"은 두 가지가 틀렸다.
  (1) **865493은 이 파일을 쓸 수 없다** — 865493의 전 20개 arm/cell
  srv.log 중 최종 파일은 **2026-07-27 23:05:55**에 끝나는데
  `s8_keepalive_prompt_224.txt`의 파일시스템 mtime은 그보다 **47분 뒤인
  23:53:11**이다(job이 이미 종료된 뒤). (2) **깨짐은 08-03이 아니라 이미
  07-27 밤에 있었다** — 같은 밤 865493 종료 직후 실행된 **865533**의
  Ha8 arm이 **전 5셀**(d16/d24/d44/d92/np)에서 이미 같은 컨텍스트-초과
  거부를 셀당 23,662–23,729건 냈다(`s8_deconf_Ha8_C1024_d*_865533_
  srv.log`, `CO_RESIDENT_frac` 붕괴 0.662→0.349). ⇒ "07-27~08-03 사이
  churn" 가설은 시점을 08-03으로 오귀속한다 — 근거는 늦어도 **07-27
  23:53경**부터 이미 있었다(원인 자체는 여전히 미규명). **865493과
  865533은 같은 캠페인의 반복측정(replicate)이 아니다** — keepalive
  실현 조건이 다른 별개 런이며, 이 사실은 §3 항목28(2026-08-03)이
  이미 "n_indep=1"로 등재해 둔 것이다. 상세 `reports/CONSENSUS.md` §3
  항목50 addendum(2026-08-15), `NOTES_D54_ANCHOR_2026-08-03.md`
  addendum(원문 보존, 신설 예정).

  ⚠️**dated 정정2(2026-08-15, doc-steward — result-analyst
  `AUDIT_C2_HEADLINE_JOB_COMPOSITION_2026-08-15.md` §3.2 반영, 원문
  미덮어쓰기, 새 성능 판정 아님)**: 바로 위 문단(같은 날 등재)이 job
  865533의 keepalive 붕괴를 **"Ha8 arm이 전 5셀에서"**로 적었다.
  원자료 재확인 결과 **4 arm(Ha8·Hs8·M8·T8) 전 20셀 전부**가 같은
  붕괴를 보인다: srv.log 거부 건수/셀 — Ha8 23,662–23,729·Hs8
  23,404–23,472·M8 23,608–23,741·T8 23,409–23,431, 클라이언트
  확증(첫 rep, d16) 865533은 4 arm 전부 `keepalive_done=0,
  keepalive_errors≈5,880–5,943`(865493은 4 arm 전부 `keepalive_done=
  199–528, keepalive_errors=0`). **어제 등재한 서술이 바로 다음
  감사에서 범위 오류로 드러난 사례** — 판정 함의는 없음(865533이
  keepalive-사망이라는 사실·`n_indep=1`·G-1의 등급은 전부 불변, 바뀌는
  것은 "어느 arm이"의 범위뿐). 상세 `reports/CONSENSUS.md` §3
  항목50 追記2, `NOTES_D54_ANCHOR_2026-08-03.md` Addendum 2.

  ⚠️**dated 정정3(2026-08-16, doc-steward — 메인 세션 실측, `s8_c2r.sbatch`
  H2 assert 구현 과정, 새 성능 판정 아님)**: 위·아래에서 "1793 토큰"으로
  적은 `s8_keepalive_prompt_224.txt`의 실제 토큰 수는 **1794**다(M8·Ha8
  토크나이저로도 동일값). C2-R 하네스(H2, `n_tok ≤ CTXCAP−256` assert)
  구현 중 arm별 토크나이저로 재측정해 발견 — **1792 초과라는 결론과
  기전(HTTP 400 전량 거부)은 불변**, 숫자 표기만 정정한다. 상세
  `workspace/engine-port/results/s8_scaleup/PREREG_C2R_RULES_REV2_
  2026-08-15.md`, `reports/CONSENSUS.md` §3 항목50 addendum3.

  **G-2 [AUDITED — 독립 수렴] C2의 높은 residency는 파티션 제어가
  아니라 워크로드 장치의 산물.** sticky OFF에서는 prefill이 in-flight일
  때만 목표 분할이 유지되므로(`_init_sticky_partition` docstring,
  `multiplexing_mixin.py:206-231`) **keepalive 포화가 C2 물리의 하중
  부재**였다. 세 경로가 독립적으로 같은 결론에 도달: (A) 코드 읽기 (B)
  keepalive 사망 시 co-residency 실측 붕괴(~90–100% → 31–34%) (C) 이번
  run block-1 telemetry 교차표(`prefill_active>0` @`decode_sms==D` =
  **0.975–0.996**, @108 = **0.000**(4파일) — claims-auditor가
  865493/865533에서 낸 0.943–0.996 / 0.0000과 같은 모양이나 **다른
  대조**(워크로드 장치 실패 전후)로 도달). ⇒ `CONSENSUS.md` §1-26(B)의
  estimand 미식별이 **C2에도 그대로 상속**됨 — **C2 앵커가 죽은 세 번째
  이유**(§0 축 불일치 · estimand 미식별 상속 · 물리를 떠받친 게 워크로드
  장치). 한계(정직 기록): 이 캠페인은 **np(무분할) 셀을 안 돌려** 감사자의
  "np만 1290 MHz 하락" 관측을 **확증하지 못한다**(분할 셀은 1396–1410 MHz
  평평).

  상세 전문 `results/s8_frontier/DESIGN.md` §4.3.13(C2 앵커 감사)–§4.3.14
  (D=54 취소·keepalive 재현성), `reports/CONSENSUS.md` §1-28·§1-29.

  #### 2026-08-03 (같은 날 4차 속행) — §0의 이분법이 유지 불가: 세 번째
  후보가 실측으로 문서화됨, 오프라인 분리 불가, GPU(S2) 대기

  > ★★★**이 소절은 상태 기록이지 성능 판정이 아니다. GPU 런(S2)은 별도로
  > 제출 중이고 결과가 없다 — throughput/latency/goodput/`g` 그 무엇도
  > 이 소절에서 주장되지 않는다.** 원자료·코드(전부
  > `results/s8_frontier/`): 1차 사전등록
  > [`PREREG_S0_AXIS_2026-08-03.md`](workspace/engine-port/results/s8_frontier/PREREG_S0_AXIS_2026-08-03.md),
  > 스크립트·원출력
  > [`s0_axis_check.py`](workspace/engine-port/results/s8_frontier/s0_axis_check.py)/
  > [`S0_AXIS_CHECK_2026-08-03.txt`](workspace/engine-port/results/s8_frontier/S0_AXIS_CHECK_2026-08-03.txt),
  > 판정서(★RETRACTION BANNER 포함)
  > [`FINDINGS_S0_AXIS_2026-08-03.md`](workspace/engine-port/results/s8_frontier/FINDINGS_S0_AXIS_2026-08-03.md),
  > 재현 사전등록
  > [`PREREG_S0R_MODE_2026-08-03.md`](workspace/engine-port/results/s8_frontier/PREREG_S0R_MODE_2026-08-03.md),
  > 독립 재현 판정
  > [`S0R_REPLICATION_2026-08-03.md`](workspace/engine-port/results/s8_frontier/S0R_REPLICATION_2026-08-03.md)
  > (result-analyst, claims-auditor도 사전등록 작성 세션도 아님),
  > 다음-gate 사전등록
  > [`PREREG_S2_STICKY_ITL_2026-08-03.md`](workspace/engine-port/results/s8_frontier/PREREG_S2_STICKY_ITL_2026-08-03.md).
  > 전문 `results/s8_frontier/DESIGN.md` §4.3.15.

  **A. §0의 이분법이 유지 불가 — 세 번째 후보 (iii)이 실측으로
  문서화됨.** §0(위 "2026-08-03(3차)")은 (i) 872077의 `decode_sms==16`이
  실제 16-SM 실행이 아니다 / (ii) C2의 28–31ms가 셀 배치 성질이다 중
  하나가 거짓이라고 적었다. 이 이분법은 "E1의 `split_frac≥0.90`이 D
  파티션 실행 토큰을 올바로 분리한다"는 전제 위에 서 있는데, 그 전제가
  T8의 세 공유 셀 전부에서 실측으로 깨진다: E1 SPLIT 모집단은 **이봉**이고,
  윗봉이 C2 셀별 SPLIT p50과 **1–2% 일치**(d16 0.992/d24 0.991/d44
  1.013), 아랫봉은 **같은 job의 UNSPLIT과 통계적으로 동일**(d16 비
  1.011). 이 발견은 claims-auditor가 `FINDINGS_S0_AXIS_2026-08-03.md`를
  감사하며 낸 것으로 **자기감사**(같은 턴에 진단+재프레이밍, 방법론 교훈
  12)였고, 그래서 별도 확증 전 인용 금지였다.

  **독립 재현(result-analyst, `S0R_REPLICATION_2026-08-03.md`)**:
  claims-auditor도 아니고 사전등록(`PREREG_S0R_MODE_2026-08-03.md`)을
  쓴 세션도 아닌 분석자가, 감사된 `m3_conditional.py` 프리미티브만
  프리미티브별로 재사용 선언(`label_probe`·`tokens`·`sel_split`/
  `sel_unsplit`·`pctl` 등)하고 C2 리더·mode estimator·lag-shift 드라이버는
  전부 새로 작성, **생산자 자체**에 대조해 실행(gate G-A: 내 `wfrac` ≡
  `label_probe`의 내부 `frac`, 39,557/39,557 정확; gate G-B: 내 C2 재구성
  ≡ `s0dc_client`의 자기 기록 20/20 정확 — 이전 회차 gate 2가 감사자 지적한
  순환성이 여기엔 적용되지 않음).

  - **행 1**(윗봉/C2 SPLIT p50 비 ∈[0.95,1.05], D에 flat) **재현**:
    d16/d24/d44 = 0.992/0.991/1.013, flat. 단 **비교대상이 중요**: 배치
    매칭 bin-5 comparand(28.79) 대비로는 1.072/1.060/1.079로 **band
    밖**(그래도 flat). C2의 SPLIT 라벨은 거의 공허(T8 d16 91,073개 중
    99.75%가 SPLIT, UNSPLIT=0) — 이 행은 사실상 E1의 3.7% 소수 모드를
    C2의 전체 모집단과 비교하는 셈이고, `CONSENSUS.md` §1-29 B-2(C2
    high-residency=워크로드 장치 산물)를 상속한다.
  - **행 2**(슬로우-쉐어가 D에 단조 증가) **순서만 재현**: pre-registered
    모집단(`a_free_only=True`)에서 6.82→9.48→17.68→22.55%, 방향은 단조.
    Paired-within-block(n=8)으로 보면 d24−d16 스텝(+2.37±3.52pp,
    t=1.90<t_crit 2.365)과 d54−d44 스텝(+3.53±4.10pp)이 **block scatter
    안에서 미해결**, d44−d24만 확실히 해결(+8.85±4.57pp, t=5.48). ★**사전
    등록 자체의 내부 불일치**를 발견: 감사자가 인용한 수준값(8.95/13.60/
    23.62/29.38%)은 사전등록 §2가 고정한 `a_free_only=True` 모집단이
    아니라 `False` 모집단의 것 — 데이터 불일치가 아니라 사전등록 문서
    안의 불일치. 방향(단조)은 어느 모집단에서든 재현.
  - **행 3**(빠른봉 ≈ UNSPLIT, ∈[0.97,1.03]) **재현**: 1.000–1.023,
    d16 p50/p50=1.011로 감사자 수치와 정확 일치.
  - ★**행 4(음성대조) — 이 회차의 핵심 결과, 강한 형태를 죽였다.** 같은
    mode estimator를 **UNSPLIT(108 SM) 모집단**에 적용: T8 전 셀에서
    동일한 슬로우 모드가 나타나고 그 위치가 셀을 정확히 따라간다(33.88→
    22.12→15.62→14.12ms, D=16→24→44→54; d54에서는 SPLIT·UNSPLIT 슬로우
    모드가 수치까지 동일, 14.12=14.12). 사전등록 falsifier("~31ms 모드가
    ~9% 점유")는 문자 그대로는 d16/d24/d44에서 발화 안 함(UNSPLIT share
    2.71/3.28/4.84%, 5% 미만 — d54는 5.61%로 발화)이지만, **그 falsifier가
    지키려던 실질은 확인된다**: d16의 슬로우 토큰 8,893개 중 **7,903개
    (88.9%)가 UNSPLIT 라벨**이고 SPLIT은 759개(8.5%)뿐이다. 농축(클래스
    내 슬로우 비율/모집단 base rate)은 **SPLIT 2.33–3.24×, UNSPLIT
    0.78–0.93×**(d16–d54 전 구간) — `split_frac≥0.90`은 슬로우 모드를
    **격리**하는 게 아니라 **농축**시킨다. 순도(SPLIT 토큰의 ~93%가
    빠른 모드)도 완전성(SPLIT이 그 job 슬로우 토큰의 8.5%만 포획)도
    성립하지 않는다.
  - **행 5(클럭 lag sweep) — 미발화**: δ∈[−0.40,+0.40]s 스윕에서 최적
    δ=+0.10s일 때도 slow share는 d16 11.4%/d44 27.0%뿐(90% 문턱에 크게
    못 미침 — falsifier는 ≥90%). δ 절대값≥0.25s에선 UNSPLIT 배경(2.7–3.1%)으로
    수렴 — 라벨이 순수 클럭 잡음은 아니라는 증거(무작위 라벨이면 base
    rate에 항상 있어야 함)이면서, 동시에 0.05s 스케일에서 취약함을 보인다.

  ⇒ **세 번째 후보 (iii)**: 두 job은 같은 축이나, 872077은 진짜 D-SM
  실행을 담고 있되 `split_frac≥0.90`이 그것을 순수하지도 완전하지도 않게
  분리한다 — 일부가 UNSPLIT으로 새어 들어간다. **§0은 더 이상 (i)/(ii)
  이분이 아니라 3지선다이며, 어느 쪽도 오프라인으로 분리 불가**. 살아남는
  두 읽기 — (a) 셀 수준 현상(파티션과 무관하게 존재) vs (b) 클럭 오프셋을
  통한 D-실행의 누출 — 는 **S2(GPU, 별도 제출 중, 결과 없음)만이 인과적으로
  분리 가능**하다. prefill-overlap 컬럼도 같은(어쩌면 shift된) 클럭에서
  계산되므로 arbitrate 불가(그 컬럼 자체도 `CONSENSUS.md` §1-27이 이미
  계측 결손으로 기록). **이 층은 §1-25가 기록한 시간 층 희석
  (`E1_DECODE_REALIZED` 4–19%) 안쪽의 두 번째 층**이며, Stage 0의
  target-vs-realized(§1-21)·§1-26(B)의 estimand 미식별과 같은 계열이다.

  **B. 철회 3건(메인 세션이 같은 날 앞서 씀) — `FINDINGS_S0_AXIS_
  2026-08-03.md`의 RETRACTION BANNER와 동일.**

  1. "§0 stands as written" — 과잉 해석. 보인 건 3개 인접 집계에서
     E1의 p50이 ≈11ms라는 것뿐이며, §0의 (i)/(ii)를 판정하지 못했다.
  2. "aggregation-invariant" — 정확한 문장은 "11.06ms 단일 모드가
     지배적이라 집계 선택에 둔감"이다. 집계 간 일치는 **mode dominance를
     확립하지 estimand identification을 확립하지 않는다** — 더 약한
     주장이다.
  3. "11.09는 집계 단위 미기록" — **틀림, 판단 문제가 아니라 사실
     오류다.** 산출자는 `m3_conditional.report_conditional` 리포트 [3],
     T8 d16 `sp_p50` = **11.0905**(n=11,124, `a_free_only=True`),
     `m3_conditional.py:158-161,251-262,316-329`에 문서화돼 있다. 없는
     건 그 산출을 낸 명령(`python3 m3_conditional.py --job 872077 --only
     conditional`)의 **stdout 저장분뿐**이다. **"저장된 출력이 없다"와
     "추정량이 미기록"은 다른 실패 모드다** — 혼동하지 않는다.

  **C. 재사용 가치 있는 계측 결함 2건.**

  1. ★`c2_anchor.py` 표 [5] "UNCONDITIONED CELL SUMMARY"가 **M8 전체와
     Ha8 d16을 조용히 누락**한다 — `meta`가 `"t0_monotonic_s" in s` 분기
     안에서만 채워지는데(`c2_anchor.py:181-187`), 표 [5]는 앵커가 필요
     없는 표다. §4.3.13이 이미 산문으로 지적한 결함("앵커 없는 arm은
     빈 행을 내는데, 이는 측정 부재가 아니라 앵커 부재")을 정확한
     코드 위치로 추적한 것. 865493의 앵커 현황을 정확히 다시 쓰면:
     **T8·Hs8 = 5셀 전부 / Ha8 = d16 없음 / M8 = 전무**. Ha8 d16 rep은
     **실제로 돌았다**(서버측 telemetry `itl_ms_p50=112.84`, n=5,200) —
     빠진 건 C2 관례로 bin을 매길 클라이언트측 앵커뿐, 측정 자체가
     아니다. 이 스크립트의 "셀 요약" 표를 인용할 때는 빈 칸을 **앵커
     부재**로 읽어야지 **0값**으로 읽으면 안 된다.
  2. mode estimator의 window `(1.15×p50, 60]ms`는 **arm-이식 불가**다.
     T8(p50≈11ms) 기준으로 고정됐는데 Ha8(p50≈31.7ms)에서는 사실상
     censoring이 된다: Ha8 d16 SPLIT 토큰의 **0.16%(약 47개)**만 창
     안에 들어오고, `share_above`류 지표로는 15.7%다. 미해명 ~87ms
     스파이크를 포함한 Ha8의 슬로우 질량 대부분이 **상한 위에 구조적으로
     존재**한다 — Ha8이 무엇이든 이 추정량은 그것을 볼 수 없다. p50
     단위로 상한을 정의했다면 피했을 문제이며, S0-R은 사후 조정(재튜닝)을
     올바르게 거부하고 결함만 기록했다.

  **D. 방법론 게이트(아래 "방법론 게이트" #9·#10으로 등재, `CONSENSUS.md`
  §3-18·§3-19와 동일).**

  - **자기가 검증하려는 코드를 복사한 게이트는 항등식에 가깝다.**
    `s0_axis_check.py`의 gate 1은 `m3_conditional.label_probe`의 라벨링
    루프를 복사해 같은 입력에 대조했다 — 새로 넣은 값(`wmean`)은 무엇과도
    대조되지 않았다. gate 2는 28.79ms를 만든 함수(`c2_anchor.collect`)를
    호출해 그 값을 "검증"했다 — 순환. S0-R의 gate G-B가 대안을 보인다:
    같은 코드 경로를 공유하는 다른 분석 스크립트가 아니라 **생산자 자체**
    (`s0dc_client`의 자기 기록)에 대조해 20/20 정확 일치시켰다.
  - ★**여집합 클래스에 음성대조를 걸어라.** 세 차례(원 C2→`G_LEVER` 감사,
    첫 §0 axis check, 이 세션 자신의 첫 프레이밍)가 놓친 것을 "같은
    estimator를 UNSPLIT에도 적용해본다"는 한 줄이 잡았다. `PROJECT_
    STATUS.md`·`CONSENSUS.md` §3 항목 9(게이트가 여집합을 세는 바람에
    *실수로* 실패)와 뿌리는 같고 방향은 반대 — 이번엔 여집합을
    **일부러** 재서 라벨의 배타성을 검정했고 **성공**했다.
  - "저장된 출력이 없다"를 "추정량이 미기록"으로 읽지 않는다(위 B.3).

  **E. 증거 수준(정확히).** 감사자 재프레이밍의 **강한 형태**(레버=D-SM
  실행 식별)는 **채택 불가**(음성대조가 반증). **약한 형태**(이봉·윗봉
  일치·아랫봉=UNSPLIT·농축 2.33–3.24×)는 **재현됨, 단 독립성은 부분적**
  — 추정량은 감사자 제안, 사전등록은 메인 세션, **실행만 독립** — 인용
  시 이 스코프 문구를 동반해야 한다. §0의 (i)/(ii)는 **여전히 미판정**.
  게이트 S1(§4.3.13)은 "부분 실현" 분기가 없어 **현 상태로 실행
  불가**(4번째 분기 필요). `G_LEVER`/`G_FLAT`는 §4.3.12(d)대로
  **UNDETERMINED 유지**(이번 회차로도 미해소). **성능 주장 0건** —
  throughput/latency/goodput/`g` 무엇도 이 소절에서 주장되지 않는다.

  상세 전문 `results/s8_frontier/DESIGN.md` §4.3.15, `reports/CONSENSUS.md`
  §1-30·§3-18·§3-19.

  #### 2026-08-05 — S2(job 873015) 독립 재현: §0 최상위 열린 항목
  behavioural 종결 [claims-auditor CONFIRMED (scoped)], E1은 여전히
  열리지 않음, 성능 판정 0건

  > ★★★**출처.** claims-auditor가 2026-08-05에 result-analyst의
  > [`S2_ANALYSIS_2026-08-04.md`](workspace/engine-port/results/s2_sticky/S2_ANALYSIS_2026-08-04.md)를
  > 독립 재현·감사해 CONFIRMED (scoped) 판정과 정본 기록 허가를 냈다.
  > 전문 doc-steward 영속화:
  > [`results/s2_sticky/S2_REPLICATION_2026-08-05.md`](workspace/engine-port/results/s2_sticky/S2_REPLICATION_2026-08-05.md).
  > 아래는 그 허가분의 **등급·scope를 축약하지 않은** 요약이다. **새 성능
  > 판정은 이 소절에 없다.**

  1. **주 결과.** S2(job 873015, sticky ON, T8 d16/d54, ShareGPT rate 2,
     n=8 블록, cudagraph ON, gpu37): pooled per-token ITL p50 =
     **28.92 ms**(d16; per-block 28.91±0.13, t95 [28.81, 29.02]) /
     **12.04 ms**(d54; 12.05±0.09). d16 값은 사전등록 `[28,34] ms` 안이며,
     **논쟁 대상인 `split_frac` 라벨을 전혀 쓰지 않고** 재현된다 —
     317,342개 클라이언트 ITL 구간의 raw median이다. **PREREG_S2 §4 row 1
     발화.**
  2. **선택기는 decode-active 시간의 사실상 전부에서 목표 division을
     유지했다**: `E1_DECODE_REALIZED`(시간가중, decode-busy) =
     **0.9990±0.0017**(d16) / **0.9994±0.0008**(d54), 16/16 cell-block
     ≥0.995, 동일 추정량이 pre-patch job 872077에서는 0.0380/0.0925. ON에서
     decode-busy ∧ 108 SM 스냅샷 **0건**, d54 guard row (64,44) **미선택**.
     ★**이는 선택기 인덱스를 인증할 뿐 하드웨어 SM 부여를 인증하지
     않는다** (`decode_sms`는 `arbiter.sm_counts[stream_index]`의
     재진술, `dual_worker.py:608-623`; sticky ON에서 `stream_idx =
     _sticky_fixed_idx`는 코드 불변식, `multiplexing_mixin.py:882-891`).
  3. **§1-28 §0의 이분법은 거짓으로 닫힌다(behavioural)**: (i)는
     **하드웨어 형태로 REFUTED**, **라벨 형태로 CONFIRMED**; (ii)는
     **DISFAVOURED**(28–31 ms 수준이 keepalive 없는 open-loop ShareGPT에서,
     decode batch 2.6× 작게, C2의 0.93×로 재현); 살아남는 답은 **(iii)** —
     `split_frac≥0.90`은 D-파티션 클래스를 격리하지도 완결하지도 못한다.
     **Scope: behavioural·selector-level. 부여 SM 수의 직접 프로브는
     없다(S3 미실행).**
  4. ★ **(iii)의 기전 — 계측에서 독립 도출.** `runtime_snapshot`은
     개수-서브샘플링(`PDMUX_DUAL_WORKER_TRACE_EVERY=32`)이고 양 캠페인
     모두 `PDMUX_TRACE_FORCE_PREFILL=0`이다. **decode-busy 조건부** 스냅샷
     케이던스는 네 arm 전부 **정확히 16 decode step**(872077 d16 0.177s /
     d54 0.176s; 873015 d16 0.465s / d54 0.193s). 따라서 ITL 구간 하나는
     스냅샷 **1/16개**를 걸치며, 872077 d16의 SPLIT 모집단(11,124 토큰)
     전체가 8블록 합 **~120개 스냅샷**에서 번져 나온다 — **기대 순도
     ≈6%**로, S0-R의 mode 분해가 다른 경로로 얻은 6.6–9%와 일치한다.
  5. **11.09 → 28.92 ms는 배치 아티팩트도 노드/바이너리 아티팩트도
     아니다.** decode-batch 기여는 세 독립 추정에서 **2.3–4.7%**(bin
     matching 4.7% / within-run slope 2.7% / 블록간 회귀 2.3–3.2%). 전역
     효과(노드 gpu36→gpu37, 1파일 바이너리 차, 캠페인 날짜)는 **d54
     companion이 ≤1.097×로 상한**을 준다(같은 before/after에서 10.97→
     12.04). 독립적으로, ~29–34 ms 수준은 **872077 자신 안에 gpu36·
     pre-patch 바이너리로 존재**한다 — 그 job의 클라이언트측 slow mode가
     셀을 따라간다(d16 34.0 / d24 22.0 / d44 15.0 / d54 14.2 ms).
  6. **워크로드 페어링은 정확하다**: `input_lens`와 `output_lens`가 4개
     job×cell × 8 블록 전부 sha256 동일; server args 19키 일치(**양쪽
     cudagraph ON**), `random_seed`만 상이. 매니페스트는
     `multiplexing_mixin.py` 1개 파일만 다르다(872077 = pre-sticky-patch
     바이너리) — 나머지 10개 런타임 파일은 해시 동일.
  7. ★ **이 런에는 결과(outcome) 게이트가 0개다**: `AMBIG_FRAC`·
     `MIN_N_SPLIT`·PREREG §3.1 일치검사는 sticky ON 하에서 **항등식**,
     `E1_DECODE_REALIZED`는 arm 간에는 비항등식이나 **ON arm 안에서는
     코드 불변식**, `ALIGN_R`은 계측 flag다. ⇒ 사전등록 게이트 중 어느
     것도 "28.92가 나올지 11이 나올지"를 제약하지 않았다. **방법론
     게이트 #9의 네 번째 재발**(세 번째 재발은 `S2_ANALYSIS_2026-08-04.md`
     §3이 §3.1 하나에 대해 이미 기록; 이번은 런 전체 결과 게이트 집합이
     구조적으로 공집합이라는 더 넓은 형태). 아래 "방법론 게이트"·
     `reports/CONSENSUS.md` §3-23에 등재.
  8. **d54 companion은 사전등록 구간 [13,16]을 빗나갔다**(12.03, CI 전체가
     13 미만), 그리고 **PREREG_S2에는 "primary 적중 + companion 미스"에
     대한 규칙이 없다**. 사후분석은 원인을 sticky 교란(부호 반대라 배제)이
     아니라 [13,16] 구간 도출 자체의 cross-cell 외삽 오류로 귀속한다(C2
     d44=14.68을 d54 대용으로 씀, C2 자체 곡선으로 직접 외삽하면 이미
     12.79로 하한 미만). **이 문장은 §0 관련 어떤 인용에도 동반해야
     한다.**
  9. **같은 런의 다른 사전등록(`DESIGN.md` §4.3.12(f))의 헤드라인 판별
     예측(T8≈1.85)은 빗나갔고**(관측 2.22 [2.21, 2.24]), **판별을 담당하는
     arm(Ha8)은 제출 격자에 없다** ⇒ §4.3.12(f)는 **설계상 미판정**이며,
     그 앵커(1.85/0.92/1.6)는 **은퇴한 `g`/`A_free` 통화**로 쓰였다.
     **모형 반증으로 읽을 수 없다.**
  10. **`S2_ANALYSIS_2026-08-04.md:61-64`의 "Instrument check" 문단은
      사실 오류다** — "텔레메트리 케이던스 ~2.0ms"는 **decode-idle** 값이고
      decode-busy 조건부로는 0.177–0.465s다. 따라서 "11ms 구간이 5.6
      스냅샷을 걸친다"는 실제로 **0.06개**이며 **90–260× 틀렸고 방향
      주장도 반대**다. 그 문서 해당 문단에 **정정 표시를 달았다**(원문
      보존, `S2_ANALYSIS_2026-08-04.md:65-83`), **정정 전 인용 금지**.
  11. **★ 최상위 열린 항목 해제.** §0은 "미해소 3지선다"에서 **"CONFIRMED
      (scoped)로 종결, 단 하드웨어 층 미프로브"**로 전환한다. C2 자체
      (레버 존재, 2.36–2.91×, scoped)의 등급·수치는 **불변**(자기완결적
      4-arm matched-batch 캠페인, 이 종결의 영향 밖) — 바뀐 것은 "C2와
      E1/sticky 격자가 같은 물리량을 재는가"라는 상위 질문뿐이고, 이제
      그쪽으로 confirmed됐다(단 `G_LEVER`/`G_FLAT` 미결이라는 별개 이유로
      C2→sticky 이식·앵커는 계속 금지).
  12. **★ E1은 열리지 않는다 — 네 가지 독립 사유**(감사자 판정, 그대로
      등재):
      1. `G_LEVER`/`G_FLAT` 여전히 **UNDETERMINED**(§4.3.12(d)).
         post-sticky 블록 sd가 0.022로 붕괴(pre-sticky t95 half-width
         0.303 대비 ~14×) ⇒ pre-sticky 산포로 교정한 임계는 **null 채택
         편향**. 임계 사전등록 미결.
      2. ★ **sticky 기판이 질문을 바꾼다** — decode-busy면 (92,16) 유지 =
         **prefill SM 92개가 벽시계 ~77% 유휴**. E1의 프론티어 질문은
         예산 제약 `[108−D, D]` 하의 **co-located 배분**인데 sticky arm은
         **단일-테넌트 decode 측정에 가깝다** ⇒ **다른 estimand**.
      3. **prefill 축 미통제** — d16에서 TTFT p50이 46.3 → 63.2ms로
         움직였고 기전 주장 없음. E1은 ITL(D)와 TTFT(108−D)를 동시에
         요구한다. ★**(2026-08-05, α 실행 후 강화)** 고-D 대조(job
         873921, d92)는 이 사유를 **완화하지 않고 오히려 강화**한다 —
         d92 TTFT p50 **162.7–210.8ms**(4 블록) vs d16 63.2–66.1ms로,
         prefill SM을 16까지 줄인 대가가 여전히 통제되지 않은 채
         관측된다. 상세 아래 "2026-08-05(α)" 소절.
      4. 음성대조 부재(d16 UNSPLIT n=0/8블록, 구조적으로 정의 불가) +
         게이트 S1 여전히 실행 불가("부분 실현" 분기 부재) + **하드웨어
         부여 층 미프로브**.

      ⇒ **긴장 A(HE2 vs C2)는 이번 회차로 전혀 닫히지 않았다.**
  13. **다음 실험 gate 갱신**(감사자 우선순위, ★2026-08-05 실행 완료 —
      아래 "2026-08-05(α)" 소절로 대체됨, 원문은 그 시점 사전등록으로
      보존): **(α) sticky-ON 고-D 대조
      셀**(~25–30 GPU-min, 4 block, decode≈92 SM 고정 division) —
      **사전등록 예측 p50 → 12–13ms**(C2 d92 재계산 12.88). 29ms 근처면
      **수준을 만든 것이 SM 수가 아니라 처치 자체**이고 §0 판정 전체가
      무너진다. **유일하게 이 판정을 반증할 수 있는 값싼 실험.** 이어서
      (β) OFF 1블록 `TRACE_FORCE_PREFILL=1`(~7 GPU-min, 샘플링 법칙 직접
      검정, 예측: D16 시간 share 3.8%→~10%, SPLIT 순도 ~6%→~1.0), (γ)
      S3(하드웨어 층), (δ) 같은 바이너리 OFF arm은 **1블록만·등재
      선행조건 아님**.

  **등재 금지**(사유 함께): "slow-mass 잔차 2.03×"/"(iii)는 양적
  미종결"(REFUTED — count-share vs time-share 단위 불일치, 매칭 단위 비
  0.97) · "하드웨어가 16 SM을 부여했다"(미검증, `decode_sms`는 선택기
  재진술, S3 미실행) · "S2는 A/B다"/"sticky의 인과 효과"(before/after,
  인과 읽기는 ≤1.097× 상한 논증만 허가) · "음성대조 통과"/"배타성
  회복"(d16 UNSPLIT n=0/8블록=검정 불능, 클래스가 물리적으로 부재) ·
  `g`·`G_LEVER`·`G_FLAT`·decode-SM 탄력도·goodput·HE0·긴장 A
  일체(PREREG_S2 §5.3, §4.3.12(d); p95 비 2.222는 값+CI+"판정 없음,
  UNSPLIT control n=0" 동반해서만) · "28.92는 구간 중앙에서 견고"(하단에서
  3.3%, 자기 측정 계통 오프셋(−5.4%)과 같은 크기 여유).

  상세 전문 `workspace/engine-port/results/s2_sticky/
  S2_REPLICATION_2026-08-05.md`, `results/s8_frontier/DESIGN.md` §4.3.16,
  `reports/CONSENSUS.md` §1-31·§3-23.

  #### 2026-08-05(α) — 고-D 대조 셀(job 873921): 사전등록 밴드 [12,13]ms
  REFUTED, §0 종결 CONFIRMED (scoped) 불변, 새 성능 판정 0건

  > ★★★★**출처.** claims-auditor가 2026-08-05에 위 항목 13의 사전등록
  > (α)을 실행한 결과를 감사했다. **성능 판정은 이 소절에 없다.**

  1. **주 결과.** α(job 873921, T8 d92=(P16,D92), sticky ON, ShareGPT
     rate 2, n=4 블록, cudagraph ON, gpu41): pooled per-token ITL p50 =
     **11.26ms**(telemetry-path, per-block t95[11.049,11.467]) /
     **11.32ms**(raw-itls path, t95[11.066,11.568]). `E1_DECODE_REALIZED`
     = 1.000/0.998/0.998/1.000(4블록), `n_err=0`(4블록). 사전등록 규칙
     (`s2_sticky_d92.sbatch:403-408`) 적용 시 **INDETERMINATE**(12–13
     밴드에 5.7 block-sd 미달, 28–30 붕괴 밴드에서 ~106 block-sd 이격).
     **붕괴 분기는 REFUTED** — §0 종결은 유지되며 등급은 **CONFIRMED
     (scoped) 불변**이다.
  2. **밴드 부검(필수 동반) — 사전등록 밴드 [12,13]은 잘못 도출됐다
     (REFUTED).** 앵커 C2 d92=12.88은 재계산으로 정확하나(pooled raw
     12.875, n=239,659), **C2와 sticky 격자는 파티션만 같고 워크로드가
     다르다**: decode-busy ctx_p50 중앙 **1291 vs 287 tok**, decode
     batch 평균 **11.31 vs 4.51**, closed-loop+keepalive vs open-loop.
     α 런 자신의 엔진측 step 회귀(`t = 10.883 + 0.0991·batch +
     0.00034·ctx`, n=2,140)로 C2 동작점을 예측하면 12.44–12.80ms로 C2
     관측 12.875와 잔차 0.9–3.8%이며, 여기에 이미 기록된 캠페인 계통
     오프셋(−5.4%, 위 "등재 금지"의 "28.92는 구간 중앙에서 견고" 항목
     참조)을 더하면 격차가 사실상 소진된다. ⇒ **격차는 새 기전이 아니라
     통제되지 않은 워크로드 격차다.** 이 밴드는 §1-31 자신이 금지한
     **C2→sticky 이식**을 예측에 사용한 것이며, 정본이 이미 보유한 더
     가까운 앵커(872077의 D108 우세 10.98, α와 블록별 byte-matched·
     batch 4.56·ctx 283)를 쓰면 예측은 11.0–11.4로 관측과 일치했다.
     **"α가 §0를 수치적으로 확증했다"는 서술 금지.**
  3. **α의 실질 기여(순환성 제거).** α는 §0 종결의 근거를 **분쟁 중인
     필드(`decode_sms`) 내부의 시간-가중 재진술**에서 **결과(outcome)
     축 앵커**로 옮긴다: 블록별 byte-matched trace·batch-matched(4.51
     vs 4.56)·ctx-matched(287 vs 283) 조건에서 ON d92/OFF d16(872077) =
     **1.0293 [1.0210, 1.0376]**, ON d16/OFF d16 = **2.6320 [2.6193,
     2.6447]**, ON d54/ON d92 = **1.0658 [1.0586, 1.0731]**. prefill
     비공존 조건 엔진측 per-step은 ON d92 **11.048** vs OFF (P0,D108)
     **10.979**(+0.6%). 872077 d16의 진짜 D16 질량은 라벨 분할이 아니라
     **client ITL의 3.43%가 [20,45]ms(중앙 31.7ms)**로 나타나며 실현
     시간점유 3.5%와 일치한다. ★872077 안에서 SPLIT p50(11.08)과
     UNSPLIT p50(10.98)의 차이는 1%인데 같은 기판의 진짜 D16 vs D92
     대비는 163%다 — **라벨은 사실상 아무것도 분리하지 않았다**(이것이
     (iii)의 결과-축 재진술이다). 붕괴 분기는 세 다리로 독립 반증된다:
     α의 d92 pin(11.3) / sticky **이전** 바이너리 C2의 실현 (P16,D16)
     91–96%에서 31.05ms / 872077 자신의 느린 모드 31.7ms. **동반 필수**:
     이 비교는 노드(gpu36/gpu37/gpu41)·바이너리(1파일)·날짜를 건너며,
     **3% 이하 차이는 그 오프셋 안이므로 정밀 일치로 읽지 않는다**(같은
     바이너리 OFF arm(δ) 미실행).
  4. **스코프(유지·강화). S3(하드웨어 부여 층)는 α로 닫히지 않는다.** α는
     D92와 D108을 **0.6–2.9%**밖에 벌리지 못하므로 "하드웨어가 92 SM을
     부여했다"를 검정할 **검정력이 구조적으로 없다**. α가 배제한 것은
     저-SM 가설(2.63×)이지 고-SM 내부 구분이 아니다. §1-31의
     "selector-level, 하드웨어 직접 프로브 없음" 스코프 문구는 **그대로
     유지**한다.
  5. **다음 gate(재정렬, 2026-08-05)**: **(δ) 같은 바이너리 OFF arm —
     승격: "선행조건 아님" → "10% 미만 교차-job 비교를 인용하려면
     필수".** 1블록 d92(가능하면 d16도), sticky flag만 OFF, 같은
     노드·같은 날. 위 3번의 1.029·1.006 비교가 딛고 선 계통 오프셋을
     처음으로 측정한다. ~7 GPU-min. **(β) OFF 1블록
     `PDMUX_TRACE_FORCE_PREFILL=1`** — 유지(높음), (iii)의 유일한
     양적 다리 직접 검정. 밴드는 **같은 job 내부 값에서만** 뽑을 것.
     **(α′) 신규(~10 GPU-min)** — sticky ON d92를 **C2의 클라이언트로**
     (C1024 closed-loop conc16 + keepalive) 1–2블록. 사전등록 예측
     **12.4–12.9ms**(α 내부 회귀에서 도출). 적중하면 "C2와 sticky
     격자가 같은 물리량을 양적으로도 잰다"가 처음 성립하고 격자 이전
     금지 근거 일부가 해제되며, 빗나가면 **C2 앵커는 영구 은퇴**다.
     **(γ) S3** — 유일하게 남은 스코프 구멍, 우선순위는 δ/β/α′ 뒤.
     제출 순서 권고: **δ(7분) → β(7분) → γ, α′는 γ와 병렬.**
  6. **통제 확인.** 매니페스트: 873015 vs 873921 공유 11파일 해시 전부
     동일 ⇒ **같은 sticky 바이너리**. 서버 인자 354키 중 차이 4개
     (`port`·`random_seed`·`pdmux_config_path`·`internal_states`),
     **양쪽 cudagraph ON**, backend triton 동일. 워크로드: 블록별
     `input_lens`/`output_lens` sha256이 872077·873015·873921 전부
     동일 ⇒ block-paired 성립. **미통제**: 노드(gpu36/37/41)·캠페인
     날짜·클라이언트 seed 경로 ⇒ 3% 이하 비교의 허용오차 미상(δ 사유).

  **등재 금지(α, 추가)**: "α가 §0를 수치적으로 확증"(REFUTED, 위 2번
  참조) · "11.26≈11.09는 정밀 일치"(교차-job 오프셋 안, 사후 통계량
  선택) · "고-D에서 라벨 순도가 다르다"(85/138,588, n 과소) · "α가
  하드웨어 층을 닫았다"(D92-D108 판별력 0.6%) · α의 TTFT 수치를 이용한
  일체의 성능/프론티어 판정.

  신규 방법론 항목: **예측 밴드도 이식 금지 규칙의 적용 대상이다. 밴드는
  가장 가까운 기판(같은 trace·같은 batch·같은 client)에서 뽑아라.
  자기가 금지한 이전을 자기 사전등록 예측에 쓰면, 실험이 성공해도
  규칙은 실패한다.** `reports/CONSENSUS.md` §3-26 참조.

  상세 원자료 `workspace/engine-port/results/s2_sticky/
  s2a_pooled_873921.txt`·`s2a_T8_873921_result.txt`·`s2_sticky_d92.sbatch`
  (전용 분석 md 아직 미작성), `reports/CONSENSUS.md` §1-31·§3-26.

## 철회된 가설

- attention/SSM layer별 static resource partition이 보편적으로 유리하다.
- layer-granular switching이 phase-granular allocation보다 유리하다.
- 기존 single-worker dynamic이 best static을 이긴다.
- 과거 simulation/no-CUDA-Graph 결과의 1.37–2.02× layer-aware goodput 향상이
  현재 real-engine 논문 결과다.
- R1 `PDMUX_DUAL_WORKER=1`이 독립 worker architecture를 구현했다.
- ★**(2026-07-28) Stage 0 "운영점 decode SM-무감각(non-binding), ctx≤16k·
  hybrid 전부"** — D108 무경합 앵커가 실은 decode 16 SM이었음이 확인되어
  (C1 CONFIRMED, 코드/telemetry/클라이언트 서명 3중 증거) 철회. 상세는 위
  "Stage 0" 절.
- ★**(2026-08-02) E1 하네스 세션에서 세웠다가 claims-auditor에 반증돼 철회한
  6건**(전부 2026-07-31~08-01 세션 내부 주장, 정본에 확정으로 오른 적은
  없다 — 되살아나지 않도록 여기 보이게 남긴다):
  - **"seed divergence = 도착 draw의 성질"**(C-A **REFUTED**) — 근거로 든
    `arrival_rps`가 측정값이 아니라 seed로부터 RNG replay로 재생성된 값
    (`e1_capacity_scan.sbatch:200-207`)이라 `(seed, n)`만의 결정론적 함수 ⇒
    "5셀 전부 동일"은 **항등식**이고 서버 정보량 0. 진짜 기전은
    `bench_serving.py:1705`의 `random.seed()`를 `datasets/sharegpt.py:98`의
    `random.shuffle`이 소비해 **seed가 프롬프트 집합 자체를 바꾸는 것**.
    seed 비-pooling 결정은 유지, **이유만 교체**(두 seed = 서로 다른 워크로드).
  - **"ITL 구속 rate ⟂ d92 off-cliff(배타성)"**(C-D **REFUTED**) — T8은
    rate 1–32 전 구간·전 룽에서 요청의 ≥93%가 ITL을 통과해 "ITL 구속 rate"가
    **공집합** ⇒ 배타성 명제가 **공허참**.
  - **"그러므로 E1의 판정력이 M8/Ha8/Hs8에 걸린다"**(**NOT-YET-SUPPORTED**) —
    2026-08-01 룽 분류에서 Ha8이 **전 룽 ITL-ALWAYS-BINDING**으로 나와 부분
    반대 증거가 생겼다.
  - **"cap↑이 T8·Hs8의 사다리를 살리고 M8·Ha8을 악화시킨다"** — **철회**.
    cap은 hybrid에서 admission이 아니라 **다른 손잡이**다("방법론 게이트" #5).
  - **"메모리는 한 번도 구속하지 않음"** — **scope 오류로 철회**. batch-cap은
    **T8 하나만** 측정했다.
  - **knee 수치·framing 정정**(이미 커밋 `d1f157a`로 정본 반영, 중복 확인만):
    d92 4.10 → **2.80**, d54 14 → **8.45**; "5셀 동시 off-cliff rate 부재"는
    부정확 → 성립 명제는 **"공통 off-cliff band(≲2–3 req/s)가 ITL 항이
    움직이는 영역과 분리돼 있다"**.
- ★★★**(2026-08-03, 같은 세션 속행) 메인 세션이 세운 "희석 attenuation"
  가설 — claims-auditor REFUTED**(전문은 위 "8B decode-SM 프론티어" 절
  "2026-08-03" 소절 (A)): "`E1_DECODE_REALIZED` 4–19%이므로 `A_free(dD)
  =w_D·A(D)+(1−w_D)·A(108)` 혼합이 872077의 `g`를 attenuate시켰고, Ha8
  보정치는 ≈1.62–1.70이다"는 control-arm reductio(T8 보정 시 21–29×로
  C2를 10배 위반)·de-engagement 직접 실험(w=0에서도 g 1–11%만 이동)·
  "A(108) 셀 무관" 가정의 실측 위반(UNSPLIT-only에서 T8 헤드라인 재현)
  3중으로 반증됐다. 확정으로 오른 적 없는 이 세션 내부 주장이나, 되살아나지
  않도록 여기 보이게 남긴다. **살아남은 것**: engagement가 낮다는 §1-25의
  전제 자체는 견고 — 죽은 것은 **보정**뿐.
- ★★★**(2026-08-03, 같은 날 3차 속행) `c2_anchor.py`의 C2→`G_LEVER`
  앵커 도출 주장 2·3·4·5 — claims-auditor REFUTED/NOT-YET-SUPPORTED**
  (전문은 위 "8B decode-SM 프론티어" "2026-08-03(3차)" 소절 D–E). 확정으로
  오른 적 없는 이 세션 내부 시도이나, 되살아나지 않도록 남긴다: "primary를
  p95→p50으로 바꿔야 한다"(처방 REFUTED — 872077 T8 양성대조가 p50에서
  0.996으로 무너져 캠페인을 구조적 NO VERDICT로 확정하는 처방이었다),
  "`G_LEVER`=1.41"(REFUTED — 끝점 선택만으로 [1.41,2.40] 도달 가능),
  "`G_FLAT`=1.25"(REFUTED — LOO 실측 t95 반폭이 1.30이라 Ha8 1.340에
  뒤집힘), "편향 부호는 하한"(NOT-YET-SUPPORTED — C2 분할 셀은 클럭 하락이
  없고 무분할 np만 떨어져 반대 방향 경고로 재귀속). **살아남은 것**: 주장
  1(실현 검증)만 CONFIRMED(단 서술 2건 정정 — 활성률은 count-weighted,
  108 시간은 drain 전용). `G_LEVER`/`G_FLAT`는 여전히 UNDETERMINED. ★**§0
  신규 발견**(C2와 872077의 "decode 16 SM" ITL이 2.6× 다름)은 이 세션
  내부 주장이 아니라 **최상위 열린 항목**으로 별도 기록(위 "8B decode-SM
  프론티어" 절 A).
- ★★★**(2026-08-03, 같은 날 4차 속행) 메인 세션이 §0 axis check 첫
  프레이밍에서 쓴 문장 3건 — claims-auditor 반증/정정, `FINDINGS_S0_AXIS_
  2026-08-03.md`의 RETRACTION BANNER와 동일**(전문은 위 "8B decode-SM
  프론티어" "2026-08-03(4차)" 소절 B). 확정으로 오른 적 없는 이 세션
  내부 문장이나, 되살아나지 않도록 남긴다: "§0 stands as written"(과잉
  해석 — 실제로 보인 건 3개 인접 집계에서 E1 p50이 ≈11ms라는 것뿐, §0의
  (i)/(ii)를 판정하지 못함), "aggregation-invariant"(정확한 문장은
  "11.06ms 단일 모드가 지배적이라 집계 선택에 둔감" — mode dominance이지
  estimand identification 아님), "11.09는 집계 단위 미기록"(**틀림** —
  산출자는 `m3_conditional.report_conditional` 리포트 [3]
  `sp_p50=11.0905`(n=11,124, `a_free_only=True`),
  `m3_conditional.py:158-161,251-262,316-329`에 문서화됨; 없는 건 그
  stdout 저장분뿐 — "저장된 출력이 없다"와 "추정량이 미기록"은 다른
  실패 모드다). **살아남은 것**: §0의 세 번째 후보 (iii)이 실측으로
  문서화됐다는 것과, 이분법이 3지선다로 바뀌었다는 것 — 죽은 건 그
  이분법을 조기에 "그대로 확정"하려 한 세 문장뿐.

## R1 판정

`job_862512`는 **R1 observer-path A/B**다. 기존 scheduler queue/batch의 alias와
logical arbiter를 추가했을 뿐 동일 `event_loop_pdmux`에서 실행됐다. dual arm에만
동기식 JSONL bookkeeping이 있었고 CUDA Graph가 꺼졌으며 server seed가 달랐고
각 arm은 한 번만 실행됐다.

따라서 “4 improved / 0 worse / 3 mixed” 판정과 decode-heavy 성능 차이의
dual-worker 인과 귀속을 철회한다. 모든 scenario는 방향성 관측, 통계·인과
미확정이다. 전체 수치는
[`R1_REANALYSIS.md`](reports/paper/R1_REANALYSIS.md)에
보존한다.

## 현재 구현

- `PDMUX_DUAL_WORKER=1`: R1 observer 재현 전용
- `PDMUX_TRUE_DUAL_WORKER=1`: 두 long-lived host issue thread, role task queue,
  immutable execution context, safe-boundary resource lease를 사용하는 R2 경로
- `PDMUX_TELEMETRY_PATH`: architecture와 무관하게 동일한 비동기 telemetry 사용
- versioned `HybridModelProfileV1`, conservative floor estimator, fixed/generic/
  Hybrid controller가 구현되어 `PDMUX_R2_POLICY`로 engine safe-boundary에 연결됨
- W1–W9 deterministic trace, paired campaign manifest, request token-ITL p95
  goodput와 paired bootstrap 분석 도구가 구현됨

True dual 경로는 module-global PD-mux role을 thread-local `ContextVar`로 바꾸는
tracked patch를 필수로 요구하며, 적용되지 않은 runtime에서는 fail-fast한다.
GPU correctness/performance 검증 전에는 production-ready로 분류하지 않는다.
현재 controller의 online ITL p95는 최근 decode-iteration wall-time window의
추정치이며 request token-level p95는 load generator에서 별도로 계산한다.

### 코드 리뷰 스코프 정정 (2026-07-24)

읽기 전용 코드 리뷰([`reports/r2_decoupling_review_2026-07-24.md`](reports/r2_decoupling_review_2026-07-24.md),
engine-porter, file:line 근거)가 확인한 구조: `PDMUX_TRUE_DUAL_WORKER=1`은
**control-plane dual-worker**다 — 두 host issue thread, role별 task queue,
immutable `ExecutionContext`, thread-local role(ContextVar)만 분리한다.
**data/resource plane은 전면 공유**된다: running batch(`max_running_requests`)는
단일 scheduler 속성이고 완료된 prefill을 같은 running batch로 in-place merge하며,
KV/mamba pool도 단일 객체, SM 파티션도 `SharedGpuArbiter`가 하나의
`stream_index`만 추적한다(92+24=116의 별도 device pool이 아니라 ≤108 단일
coupled index). 확정된 결과 §3의 死因 얽힘이 사는 substrate(공유
running-batch+KV)를 이 구현은 **구성상 깰 수 없다** — 관측될 win/loss는
host-thread overlap(control-plane)에 귀속되며, 별도 device pool disaggregation과
hybrid mamba/SSM state transfer가 필요한 headroom에는 도달 불가하다. 이 두
경로는 코드에 **미구현**이다(state-transfer 경로 전무, mamba conv/ssm state
migration 스캐폴딩조차 없음). 따라서 **Claim D는 "control-plane coupling
감소"로 범위를 축소**한다 — "얽힘을 깬다"는 프레이밍으로 쓰지 않는다. R2는 GPU
correctness gate를 통과한 이력이 없다(`results/r2_eval/` 디렉터리 미생성,
`architecture=true_dual` telemetry 전무). 부가로 admission
latch(`r2_admission_limited`)에 **known-latent 버그**가 코드 근거로 확인됐다:
split batch가 None으로 배수되면 재평가 경로가 없어 latch가 True로 고착되어
prefill admission을 영구 차단할 수 있다(clear 경로 부재) — **사용자 결정으로
현재 수정하지 않고 보류**한다.

## 증거 수준

| Claim | 상태 |
|---|---|
| A. composition/context/load-dependent decode demand (★2026-07-26 Stage 0의 "운영점 decode SM-무감각" rider는 ★★2026-07-28 claims-auditor 감사(C1 CONFIRMED, D108 앵커 무효)로 철회 — 대신 C2(scoped): prefill 16 SM 고정 시 decode ITL SM16→SM92 2.36–2.91×, 4 arm 모델-무관, "8B decode-SM 민감도 측정 노트" 참조. 등급 변경 없음 — 레버 존재만 확립, 정책 이득 근거 아님) | 부분 지지 |
| B. layer-level reconfiguration의 critical-path 손상 | 강한 지지, 현 substrate 한정 |
| C. decode starvation의 TTFT entanglement | running-batch 경로 강함, KV 경로 부분 |
| D. true dual-worker가 coupling 감소 (★2026-07-24 코드 리뷰로 control-plane 범위로 축소, 위 "코드 리뷰 스코프 정정" 참조) | 미검증 |
| E. Hybrid-informed policy가 generic/static보다 우수 | 미검증 |

## 다음 실험 gate

1. 대칭 telemetry의 observer effect가 3% 미만이고 paired CI가 0을 포함해야 한다.
2. 동일 fixed split에서 true dual이 legacy 대비 decode progress/ITL/queue age를
   개선하며 throughput regression이 3%를 넘지 않아야 Claim D를 채택한다.
3. estimator의 95% upper-bound coverage가 95% 이상, under-reservation epoch가
   1% 이하여야 한다.
4. target applicability 영역에서 proposed가 B1/B5보다 paired CI 기준 유의하고
   effect가 3% 이상이어야 Claim E를 채택한다.
5. 벡터1(disjoint conflict-regime escape hatch, CONSENSUS §5-8(c)): **CONFIRMED
   closure (scoped, 2026-07-25)** — g2_0_full → g2_0_hard → g2_0_decliff →
   g2_0_rasweep → g2_0_raconf(pre-registered 24-job 확증 열, 결정 규칙 충족)로
   short-ctx band(scope는 위 "벡터1" 절 참조)에 견고한 disjoint 없음을 최종
   확정. 더 이상의 게이트 없음(트랙 종결) — 남은 방향은 (i) long-context
   재검증(decode floor 상승 영역, 미실행), (ii) §1-20 spatial coupling-tax/
   decoupled substrate(별도 트랙, 아래 항목 2 참조). 어느 쪽도 아직 실험
   설계·게이트가 없다.
6. **논문 positioning(2026-07-25, venue-strategist prior-art 조사,
   [`reports/paper/venue_positioning.md`](reports/paper/venue_positioning.md)
   §0.1)**: 신규성 축은 disaggregation이 아니라 **co-located multiplexing**
   (DuetServe/MuxWise/SGLang-pdmux/Nexus/Bullet 대조) — 부분적 신규성 실증,
   방어 자산은 DuetServe(libsmctrl·Transformer서 동적 승)와의 정량적 상반.
   negative를 (A) green-context 종속(Claim B, 헤드라인 금지) / (B)
   mechanism-independent 후보(Claim A/C, lever-weakness·entanglement)로 분리.
   ★**green-context = 배포 가능한 유일 vendor primitive(CUDA 12.4+)** →
   "libsmctrl 쓰면 되잖아"는 배포 불가 research curiosity로 반박(약점 아님).
   ⚠️**초판의 "cross-substrate serving 이식 make-or-break" 프레이밍은 철회**
   (MPS=정적·프로세스별, libsmctrl=비-vendor·세대귀속 → 이식 불필요·부적합).
   대신 (B)를 **기존 green-context 위에서** 닫는 vendor-substrate 3수: 새 게이트
   = **Transformer-control 대조**(같은 green-context+SLO, drain 상쇄 → 동적
   flip이 모델서 갈리면 hybrid 귀속 식별) + roofline lever-weakness microbench
   (기존 r0c) + 기측정 entanglement 귀속(switch≈0). long-ctx(위 5번 (i))는
   ctx-regime 경계용(별도 질문). 상세는
   [`reports/paper/EXPERIMENT_ROADMAP.md`](reports/paper/EXPERIMENT_ROADMAP.md)
   "벡터2"(TC-series) 절.
7. **long-context Stage 0(L−2) 게이트: ★★철회(2026-07-28, C1 CONFIRMED)** —
   2026-07-26엔 "실행 완료, non-binding"으로 기록했으나, 근거였던 D108 무경합
   앵커가 실은 decode 16 SM이었음이 확인돼(위 "Stage 0" 절) 판정2/판정3이
   철회됐다. **L−2 게이트는 사실상 아무것도 측정하지 않았다** — 따라서
   L−1 이상(SLO 재정의·모델 교체 baseline·시간축/공간축 충돌 스윕)은 "게이트
   실패로 보류"가 아니라 **"게이트 미실행"**으로 되돌아간다(재개 권고 아님,
   판정 부재라는 뜻). 상세
   [`reports/stage0_verdict_2026-07-26.md`](reports/stage0_verdict_2026-07-26.md)
   (원 판정, 철회됨).
8. **8B decode-SM 프론티어 실험 E1 (2026-07-28 지정, 2026-07-29 하네스 구축
   완료·본 스윕 미실행)** — 위 "8B decode-SM 민감도 측정 노트"(C2)가 확립한
   레버가 예산 제약 하에서 net-positive인지 판정하는 gate. 설계: `[108−D,
   D]`(D∈{16,24,44,54,92}) + best-static 대조, 4 arm, **offered-rate 고정**
   (closed-loop 금지), 용량 선측정 후 off-cliff rate 선택, **n≥4**, paired
   bootstrap, TTFT p50/p95/p99 + request-내부 token-ITL p95 + conjunctive
   goodput 보고. **사전등록 게이트 2개**: realized 파티션 점유율 ≥0.80, 파티션
   활성률 ≥0.60(2026-07-29 시간가중으로 정정 — 아래 "방법론 게이트" #4 참조).
   **사전등록 결정규칙**: 어떤 D가 best static을 conjunctive goodput에서 ≥3%
   이기고 paired CI가 0을 배제하면 채택, 아니면 C2는 "ITL 레버는 있으나 예산
   제약 하 net-negative"로 확정한다. **사전등록 SLO(2026-07-29, 사용자 지적으로
   개정)**: 1차 **ITL-p95 = 60ms 고정**(근거는 데이터 적합이 아니라
   `serving_slo_survey.md` chat-class + §1-17 선례), 사다리 {50,60,80}ms
   민감도 병기. 초안의 150ms는 8B 측정 ITL p50이 전 arm·전 D에서 그 아래라
   ITL 항이 non-binding해져 conjunctive goodput이 TTFT-only로 붕괴시키므로
   폐기됨. TTFT SLO는 ≥1 셀에서 binding + 모든 셀 p95로부터 ≥15% 마진, 없으면
   "TTFT 축 ill-posed"로 보고한다. **게이트 #8(동적 컨트롤러 규율의 재스코어
   금지)은 전 셀 `FixedPolicy`인 E1의 사전등록 사다리 재스코어에는 적용되지
   않음**을 명시.

   ★★★★**제출 선행조건 신설(2026-08-06, 통계 방법 층 정정)**: 위 결정규칙의
   "paired bootstrap"·`s8_frontier/e1_analyze.py:492,1286`가 구현한 자체
   paired percentile bootstrap(seed=20260728, n≥4)은 방법론 게이트 #14가
   확인한 것과 같은 undercoverage(n=4 coverage 0.798, 한쪽 오류율 ≈10%)를
   가진다 — 그리고 결정규칙이 "≥3% ∧ CI 0 배제 ⇒ WINS" 단방향이라 이
   undercoverage는 **오직 net-positive로 해소되는 방향으로만** 편향된다.
   E1은 아직 미제출이므로, **제출 전에 이 결정규칙의 CI를 t-CI(또는 동등한
   커버리지 보정)로 교체**하는 것을 새 선행조건으로 등재한다(engine-porter
   구현 소관, 여기선 게이트 등재만). 상세 `reports/CONSENSUS.md` §1-1
   (rev13)·§3 항목27.
   - **E2**: ctx∈{1024,4096,16384}로 확장, t0 패치된
     `workspace/engine-port/results/s8_scaleup/s0dc_client.py` 사용.
   - **E3**: duty-cycle 2수준(짧은 keepalive vs 긴 keepalive) 설계상 종결 —
     추가 실행 불필요, "짧은 prefill을 자주" 규율만 준수.
   - **E4**: C2b("hybrid 급락=Zamba2 성질")는 현존 체크포인트로 통제된 비교가
     불가능하므로 — 파라미터·형상·tokenizer가 동시에 다름 — **주장 폐기가
     정직한 수순**이다(추가 실험으로 구제하지 않는다).
   **완료**: `results/s8p_prefill/`(prefill 축 SM 민감도) — 2026-07-29 판정,
   정본 인용 금지 유지(claims-auditor 미통과). 요약: 기울기 비 4.74–5.16×,
   탄력도 ε 0.89–0.94, 4 arm 모델-무관(REAL scoped, 미감사). claims-auditor
   반증 대상(위 "열린 긴장" 참조): 게이트 미달 5셀의 strict 재귀속 충분성·
   곡률 크기가 attention FLOP 예측의 5배(기전 미상)·두 게이트가 동일 사건이라
   사전등록 강도가 1개분인 것·`--probe-conc 2` 사전등록 미실행.
   **진행 중(2026-07-29 신규)**: `results/s8_frontier/` 하네스 구축 완료(양축
   판정기·분석기·config 5, seed-per-rep·rep-고정-per-cell 정책), 측정 방법론
   결함 5종(집계 단위) 발견·수정 완료. 하네스 전제 job **867231**(T8 5셀 용량
   스캔, `RATES="1 2 3 4 6 8 12 16 24 32"`×2 seed)·**867298**(`results/
   e1_traceforce/` 관측자 효과 게이트, ABBA n=4)은 세션 핸드오프(2026-07-29)
   시점 PENDING으로 기록됐으나 ★**2026-07-31 doc-steward 갱신 시 `sacct` 재확인
   — 둘 다 COMPLETED**(867231: 21:15:25–22:41:31 / 867298: 21:16:56–21:50:21,
   둘 다 2026-07-29). ★**2026-07-31 정정: "분석/판정 파일 0건"은 오기**이며 두
   job 모두 in-job 분석이 완료돼 있다(위 "8B decode-SM 민감도 측정 노트" 절의
   정정 참조) — 관측자 효과 게이트 **조건부 통과**(본 스윕 force-trace OFF,
   pin 검증만 분리: `DESIGN.md` §4.7.1), 용량 스캔은 **§4.2/§9.1 escalation
   분기 재정의**(knee 첫 교차: d16 12.6 / d24 16.0 / d44 16.0 / d54 8.45 /
   d92 2.80 req/s; 성립 명제 = "공통 off-cliff band ≲2–3 req/s가 ITL 항이
   움직이는 영역과 분리"). 남은 전제는 이 해소이지 두 job의 재실행이
   아니다 — 본 스윕은 (i) 867231 용량 스캔 분석으로 SLO 사다리 확정, (ii) 867298
   분석으로 관측자 효과 게이트 통과 여부(=`PDMUX_TRACE_FORCE_PREFILL` 점화
   여부) 확정 — **(i)(ii) 모두 2026-07-31 완료**. ⇒ **다음 액션은 (iii)
   escalation 해소**(공통 off-cliff band와 ITL 구속 영역의 분리를 어떻게
   처리할지 결정)이며,
   나머지 3 arm(M8/Ha8/Hs8) 용량 스캔은 그 결정이 스캔 설계를 바꿀 수 있으므로
   그 뒤에 제출한다.
   - ★**2026-08-02 상태 갱신 — 위 "그 뒤에 제출한다"는 이미 집행됐다.**
     2026-08-01에 3 arm 용량 스캔(jobs **870295**=M8 / **870296**=Ha8 /
     **870297**=Hs8, 각 100 probe, 오류 0)과 T8 batch-cap 실험(job
     **870301**, {d16,d44}×cap{48,96,192}×4 seed, 24 probe, 오류 0)이
     완료됐다. 결과 요약·수치는 위 "열린 긴장"의 **"2026-08-01 실험 4건"**
     소절에 있으며 ⚠️**전부 claims-auditor 미통과 = 미검증, 인용 금지**다.
     ⚠️단 위 (iii) **escalation 해소는 선행되지 않았다** — 3 arm 스캔은
     그 결정 전에 제출됐고, 결과는 escalation을 **더 넓혔다**(d92 knee 2.80이
     네 arm 전부에서 구속 ⇒ 공통 off-cliff 상한이 arm-무관하게 2.80).
     escalation 해소는 여전히 **열린 항목**이다.
   - ★**본 스윕은 미제출이며, 그 판단 근거는 세 가지가 동시에 성립하기
     때문이다**: (a) 사전등록 사다리 {50,60,80}ms가 as-run 설정에서 **네 arm
     전부 판정 불가**(HEADLINE-ELIGIBLE RUNGS = NONE), (b) 그 as-run 설정
     자체가 **왜곡으로 증명됨**(`--max-running-requests 48`이 ITL·TTFT 두
     축을 반대 방향으로 왜곡), (c) 제외 규칙(d92 knee 2.80, 전 arm)이 **어떤
     동작점에서도 decode-rich 끝을 제거** — C2 레버가 사는 끝. ⇒ **E1은
     `results/s8_frontier/DESIGN.md` §4.3.5(b)가 사전등록한 "설계상 이 질문에
     도달할 수 없다" 분기로 갈 위험이 높다.** 이는 실패가 아니라 **미리
     적어둔 분기**이다. 긴장 A(HE2 vs C2)는 이 경우 E1으로 닫히지 않는다.
   - **다음 액션 후보(미결정, 사전등록 필요)**: (i) `--max-mamba-cache-size`를
     전 arm 공통 상수로 명시 고정해 cap을 순수 admission 손잡이로 되돌리기
     (E1 correctness에 필요, "방법론 게이트" #5), (ii) cap을 데이터 독립적
     규칙("batch가 cap-bound가 아닌 최소 cap")으로 사전등록 파라미터 승격
     — 단 (i)이 선행해야 하고 **arm별 cap 튜닝은 손잡이만 바꾼 SLO 쇼핑이라
     명시적으로 금지**, (iii) 2026-08-01 결과 4건의 claims-auditor 회부.
   - **하네스 결함·사전등록(전부 커밋 완료)**: 버그 #7/#7b/#8/#9 +
     `traceforce` `PIN_CHECK` 인자순서 수정, `DESIGN.md`
     §4.3.5(`ITL-ALWAYS-BINDING`·룽 4분류·on-cliff 셀 제외)·§4.3.6(미등록
     cap 상수·decode duty cycle)·§4.7.1(force-trace를 pin 검증 전용으로
     분리) 사전등록, `results/s8_frontier/decode_duty_check.py` 신설
     (prefill 게이트와 구조적으로 동일한 시간가중 추정량; **의도적으로
     cite-blocking 게이트가 아님** — §5가 게이트 2개를 사전등록한 뒤 데이터를
     보고 세 번째 임계를 더하면 게이트를 데이터에서 고르는 것이 된다).
   - ★**상태 갱신(2026-08-03, 같은 날 2차 속행)**: 위 (iii)은 이미 집행됐고
     (§1-26/§1-27), 이어서 `A_free`가 은퇴하고 조건부 per-token 추정량으로
     estimand가 이관됐으며(위 "8B decode-SM 프론티어" "2026-08-03(2차)"
     소절 (I)), `PDMUX_STICKY_PARTITION`이 구현·correctness gate 통과했다
     (동 소절 (II)). **다음 액션 = sticky 격자 1회 제출**(872077 동일 설계
     8 block, 872077이 non-sticky 대조) — 단 제출 전 **`G_LEVER`/`G_FLAT`
     사전등록이 미결 열린 항목**이다(동 소절 (III)(d)).
   - ★★**상태 갱신(2026-08-03, 같은 날 3차 속행)**: 위 "다음 액션"의
     `G_LEVER`/`G_FLAT` 미결 항목을 C2 데이터로 해소하려던 시도
     (`c2_anchor.py`)를 claims-auditor가 감사해 **경로 자체를 폐기**했다
     (위 "8B decode-SM 프론티어" "2026-08-03(3차)" 소절). **`G_LEVER`/
     `G_FLAT`는 여전히 UNDETERMINED**이며, 다음 시도는 감사자 발안 (α)/(β)
     에 대한 **독립 사전등록**이 선행돼야 한다. ★**더 시급한 선결 항목이
     새로 생겼다**: 같은 감사가 872077의 "decode 16 SM" ITL과 C2의
     "decode 16 SM" ITL이 **2.6× 다름**을 발견했고(§0), 이 층이
     872077·sticky 결과 전체가 딛고 선 바닥이다 — **sticky 격자 제출보다
     먼저 이 모순을 가려야 한다**(engine-porter의 하드웨어 SM 부여 직접
     검증 권고, 또는 감사자 제안 게이트 S1). D=54 앵커 측정(jobs
     872920/872921)은 이 감사와 별개로(그러나 독립 수렴하는 결론으로)
     제출 17분 뒤 취소됐다 — 취소됐으나 캠페인 설계는 재사용 가능,
     상세는 위 소절 G.
   - ★★★**상태 갱신(2026-08-03, 같은 날 4차 속행) — "이 모순을 가려야
     한다"는 선결 항목은 아직 닫히지 않았고, S2(GPU)가 별도로 제출
     중이다(결과 없음).** 위 §0의 이분법이 실측으로 **유지 불가**임이
     확인됐다 — 세 번째 후보 (iii)("두 job은 같은 축이나
     `split_frac≥0.90`이 순수하지도 완전하지도 않다")가 claims-auditor의
     재프레이밍(자기감사) + result-analyst의 독립 재현
     (`S0R_REPLICATION_2026-08-03.md`)으로 문서화됐다(위 "8B decode-SM
     프론티어" "2026-08-03(4차)" 소절 A). 재현의 음성대조(행 4)가 그
     재프레이밍의 **강한 형태**를 죽였으나(UNSPLIT도 같은 슬로우 모드를
     가짐), **약한 형태**(농축 2.33–3.24×)는 재현됐다. §0의 (i)/(ii)는
     여전히 미판정(이제 3지선다)이고 **오프라인으로는 더 분리 불가** —
     감사자 제안 게이트 S1도 "부분 실현" 분기가 없어 현 상태로 실행
     불가(4번째 분기 필요). **다음 액션 = S2**(`PREREG_S2_STICKY_ITL_
     2026-08-03.md`, T8 d16 주+d54 동반, sticky ON, `E1_DECODE_REALIZED
     ≥0.90` 게이트, per-token p50(SPLIT) 예측 [28,34]ms/[13,16]ms) —
     **이미 별도로 제출 중이며 결과는 아직 없다**. `G_LEVER`/`G_FLAT`는
     §4.3.12(d) 그대로 UNDETERMINED(이번 회차로도 미해소). 상세는 위
     소절, `results/s8_frontier/DESIGN.md` §4.3.15, `reports/CONSENSUS.md`
     §1-30.
   - ★★★★**상태 갱신(2026-08-05) — S2가 반환됐고, claims-auditor 독립
     재현이 §0을 behavioural CONFIRMED (scoped)로 닫았다. E1은 열리지
     않는다.** job 873015 pooled per-token ITL p50 = **28.92ms**(d16,
     사전등록 [28,34]ms 안) / **12.04ms**(d54, 사전등록 [13,16]ms
     미달 — 인용 시 필수 동반). §0의 (i)/(ii)/(iii) 3지선다가 (i) 하드웨어
     형태 REFUTED·라벨 형태 CONFIRMED / (ii) DISFAVOURED / (iii) 채택,
     기전이 스냅샷 샘플링 케이던스에서 독립 도출로 종결됐다(위 "8B
     decode-SM 프론티어" "2026-08-05" 소절). **다음 액션 = (α) sticky-ON
     고-D 대조 셀**(decode≈92 SM, ~25–30 GPU-min, 4 block, 사전등록 예측
     p50→12–13ms — 29ms 근처가 나오면 이 종결 전체가 무너지는 유일한
     값싼 반증 실험), 이어서 (β) OFF 1블록 `TRACE_FORCE_PREFILL=1`(~7
     GPU-min), (γ) S3(하드웨어 SM 부여 직접 프로브), (δ) 같은 바이너리
     OFF arm(1블록, 등재 선행조건 아님). `G_LEVER`/`G_FLAT`는 §4.3.12(d)
     그대로 UNDETERMINED. **긴장 A(HE2 vs C2)는 전혀 닫히지 않았다** —
     E1을 열려면 α~γ와 별개로 `G_LEVER`/`G_FLAT` 독립 사전등록, sticky
     estimand를 co-located 예산 배분으로 되돌릴 설계, prefill 축 통제가
     모두 필요하다. 상세는 위 소절, `workspace/engine-port/results/
     s2_sticky/S2_REPLICATION_2026-08-05.md`, `results/s8_frontier/
     DESIGN.md` §4.3.16, `reports/CONSENSUS.md` §1-31·§3-23.
   - ★★★★★**상태 갱신(2026-08-05, α 실행) — (α)가 실행됐다. §0 종결은
     CONFIRMED (scoped)로 불변, 사전등록 밴드 [12,13]ms는 REFUTED다.
     새 성능 판정 없음.** job 873921(T8 d92, sticky ON, n=4 블록,
     gpu41) pooled per-token ITL p50 = **11.26ms**(telemetry-path)/
     **11.32ms**(raw-itls path). 사전등록 3-밴드 규칙 적용 시
     **INDETERMINATE**(12–13 밴드 미달, 28–30 붕괴 밴드와도 거리 큼) —
     **붕괴 분기는 REFUTED**. ★그러나 **밴드 [12,13]ms 자신이
     REFUTED**다 — C2→sticky 이식(§1-31이 스스로 금지한 조작)으로
     도출됐고, α 내부 엔진측 회귀로 예측한 C2 동작점(12.44–12.80)이
     관측(12.875)과 0.9–3.8%만 어긋나 계통 오프셋(−5.4%)으로 소진 —
     격차는 새 기전이 아니라 통제되지 않은 워크로드 격차(closed-loop+
     keepalive vs open-loop). α의 실질 기여는 §0 종결을 결과(outcome)
     축 앵커로 옮긴 것(block-matched ON d92/OFF d16=
     1.0293[1.0210,1.0376], ON d16/OFF d16=2.6320[2.6193,2.6447]).
     **S3(하드웨어 부여 층)는 여전히 닫히지 않는다**(D92/D108 0.6–2.9%
     만 벌림, 검정력 없음) — §1-31의 scope 문구는 그대로 유지. **다음
     gate 재정렬**: (δ) 같은 바이너리 OFF arm 승격(10% 미만 교차-job
     비교 인용의 신규 선행조건) → (β) → (γ), (α′) sticky ON을 C2
     클라이언트로 1–2블록(γ와 병렬). 신규 방법론 항목(예측 밴드도
     이식 금지 규칙의 적용 대상, `reports/CONSENSUS.md` §3-26) 등재.
     상세는 위 "8B decode-SM 프론티어" "2026-08-05(α)" 소절, 원자료
     `workspace/engine-port/results/s2_sticky/s2a_pooled_873921.txt`·
     `s2a_T8_873921_result.txt`·`s2_sticky_d92.sbatch`, `reports/
     CONSENSUS.md` §1-31·§3-26.

9. **ceiling-censoring 진단 후속 (2026-08-05, claims-auditor 감사 후
   지정)** — §1-13 각주(LO goodput 이중 절단)의 잔여 불확실성 해소.
   새 성능 판정 없음, HE0/§1-13 판정 자체는 불변. (1) **warm-up 제거
   재측정**: 각 라운드 앞 30 요청 폐기 후 200개 재측정, n≥6. 사전등록
   예측: 폐기 후 LO ITL-fail=0이면 T=60 대조는 정의상 소멸하고 각주는
   pass-절단 산술만으로 선다. (2) **[55,60) 슬로우 모드 정체**: 토큰의
   3.23%가 55–59ms이고 4 arm 불변(1.2 SD) — **LO 판정의 실질 바닥**.
   후보(`mamba_track_interval=256`/`decode_log_interval=40`/host GC)를
   **한 번에 하나씩** 4셀×n=3, 동시 변경 금지(방법론 게이트 #10). (3)
   **노드 교락 해소**: 같은 노드 interleaved n≥4로 paired 정당화(현재는
   d16 rep44-46/d44 rep41-43이 arm과 노드·날짜가 교락, `CONSENSUS.md`
   §1-13 각주 "노드 교락 방어" 참조 — 완전매칭 대조(d34/d44 rep41-43,
   gpu38)는 이미 부호를 재현). (4) **LO 레버의 정책 이득 여부**는 게이트
   #7대로 tight ITL SLO 재튜닝 + 직접 서빙 측정으로만 판단한다 — 초과질량
   표는 근거가 아니다. 상세 `workspace/engine-port/results/slo_sched/
   CEILING_CENSORING_DIAG_2026-08-05.md`(AUDITED, 정정 표시 포함),
   `reports/CONSENSUS.md` §1-13·§3 항목24·25.
   **도구 결함(engine-porter 후속, 정책 결론과 무관)**:
   `workspace/engine-port/benchmarks/pdmux_eval/analyze.py:183`의
   `request_slice`가 요청만 자르고 `duration`(:180, 합산은 올바름)은
   안 잘라 — 분포 통계엔 무해하나 slice로 goodput을 계산하면 즉시
   틀린다. docstring 경고 추가 필요, 아직 미수정.
10. **P1 운영점(cudagraph-ON) 대조 후속 (2026-08-05, claims-auditor 지정,
    우선순위순)** — 위 "확정된 결과" 1번/`reports/CONSENSUS.md` §1-1의
    P1 감사(jobs 873944/873945)가 연 4개 gate. 새 성능 판정 없음, 각
    gate 자체가 판정 대상.
    - ~~**Gate 1 — telemetry 재현런**~~ → ✅**완료(2026-08-06, job
      874478) — claims-auditor 감사, 조건부 채택[진단 전용, 새 성능
      판정 0건].** decode-busy ∧ prefill-in-flight 구간의 시간가중
      selector 라벨 100.00%가 `(74,34)`. §1-1의 파티션·기전 금지
      문구는 **부분·조건부 해금**(Zamba2 rate{2,3}·agnostic v1·
      cudagraph-ON, selector-level 한정) — rate 4·6·Granite는 여전히
      미측정·여전히 금지, "PD 분리 자체" 기전 귀속은 Gate 2 소관으로
      불변. 전문 `CONSENSUS.md` §1-1 Gate 1 블록·§3 항목28·29, 원자료
      `workspace/engine-port/results/p1_gates/gate1/`. 이 gate가
      연 후속 4개:
      - **G1-a**(≈0.2 GPU-hr, engine-porter): `multiplexing_mixin.py:
        1005`/`:1080` 사이에 관측 전용 sync 1회 추가 ⇒ "prefill
        in-flight ∧ stale idx"가 관측 가능해져 주 조건에 판별력이
        생긴다. 결정량 = 어드미션-후/adjust-전 구간의 시간 비율과
        절대 ms. **미착수 — 2026-08-11 G1-c 완료로 착수 가능**(관측자
        혼입 회피 목적으로 G1-c 뒤로 순서를 미뤄뒀던 것, 우선순위
        재평가는 engine-porter 소관).
      - ~~**G1-b**(≈0.2 GPU-hr): 같은 하네스에 rate 4·6 창 추가.
        결정량 = `max(decode_running_batch_size) ≥ 36` 여부 + pop A
        시간가중 hist. `(54,54)`가 등장하면 위 "단일 분할" 문장
        즉시 철회.~~ → ✅**완료(2026-08-07, job 875293) — 철회 규칙
        발화, 새 성능 판정 아님.** rate 2·3·4는 pop A 시간가중 100%
        `(74,34)`(max decode_bs 9·23·18, 문턱 36 미만, 인용 가능
        셀은 불변) — **rate 6에서만 `(54,54)`가 시간가중 8.37%**
        (2.089s/24.974s, max decode_bs 40, pop A 에피소드 5개뿐,
        경계 근접) 등장해 사전등록 철회 규칙 발화 ⇒ 위 "확정된
        결과" 1번·`reports/CONSENSUS.md` §1-1(rev15)의 "이 격자에서
        정책은 단일 분할에 고정됐다"(격자 전체 무제한 주장)를
        **철회**. ⚠️rate 4(18)가 rate 3(23)보다 낮은 비단조(원인
        미해명, §3 항목30). 양성대조 PASS(rate2·3이 874478 재현).
        전문 `CONSENSUS.md` §1-1(rev15)·§3 항목30, 원자료
        `workspace/engine-port/results/p1_gates/gate1/
        gate1b_result_875293.txt`·`PREREG_G1B_2026-08-07.md`.
      - ~~**G1-c**: Granite(873945 복제) — Granite 전체 미측정 상태
        해소.~~ → ✅**완료(2026-08-11, job 877974, 0.10 GPU-hr) —
        Gate 2-S Granite r3·r4 §8.9 전제 VERIFIED, 새 성능 판정
        아님.** rate 3·4 모두 frac((54,54))(pop A 시간가중)=0.0000·
        max(decode_bs)=10<36, 양성대조(구조적 근항등식) 4 rate
        전부 PASS. **해제되는 것은 §5.6.1 명명 층뿐**(간헐 전달/
        "엔진 기본 궤적"·"A4형" 명명 허용) — **크기 인용 자격은
        불변**(코드 확인, `g2s_analyze.py:1157-1161`: `premise`는
        Δ·CI·Holm·F-계열 gate 산출에 미입력), Granite r3·r4는
        F-계열 발화 상태라 `SIGN ONLY, MAGNITUDE NOT CITABLE` 유지
        — **크기 인용 가능 셀은 여전히 Zamba2 r2 하나**(세션
        초반 "3개로 는다" 서술은 반증됨). 조건부 해제 필수조건
        6건·해제 후 금지 문장 8건은 위 "확정된 결과" 1번·
        `CONSENSUS.md` §1-1(2026-08-11 G1-c 블록) 참조. 원자료
        `workspace/engine-port/results/p1_gates/gate1/
        {PREREG_G1C_2026-08-11.md, gate1c_result_877974.txt,
        gate1c_analyze.py}`.
        - ~~**후속 E1**(GPU 0, CPU ≈2분, 권장, ⚠️blind 아님) — Gate
          2-S 자신의 셀 슬라이스 A4 텔레메트리에서 rate·rep별
          `max(decode_bs)`를 직접 산출.~~ → ✅**완료(2026-08-11,
          jobs 877756/877757 재집계, GPU 증분 0) — 채택, 등급
          하향된 조건부 채택(4셀 전부).** Zamba2 r2=14(여유
          22)·r3=23(여유 13)·Granite r3=10(여유 26)·r4=13(여유
          23), 4셀 전부 `VERIFIED_AT_SAMPLED_INSTANTS`. **E1 §2.5가
          주장한 "밀도 페널티를 피했다"는 반증됨** — 실질 우위는
          반복수(n=10)·셀 일치뿐(결정 관련 pop-A 관측 수는 G1-b/
          G1-c 대비 오히려 5–6× 적음, 1,279·1,496 vs 6,402·8,920).
          조건 7개 전부 하에서만 유효, Zamba2 r3는 q=1e-6 bound에서
          이미 37≥36으로 문턱 미배제(4셀 중 유일). 성능 결론
          불변. 전문·인용 제한은 위 "확정된 결과" 1번·
          `CONSENSUS.md` §1-1(2026-08-11 E1 addendum 블록) 참조.
          원자료 `workspace/engine-port/results/p1_gates/gate2/
          {PREREG_G2S_E1_ADDENDUM_2026-08-11.md, g2s_e1_premise.py,
          g2s_e1_premise_877756_877757.json}`.
          - **E1-a**(GPU 0) — §2 bound를 사전등록 후 4셀×5arm×
            capscan 전체에 적용, 꼬리 분위수 하나(권장 1e-6)
            사전 고정. ⚠️감사자가 이미 값을 봤으므로 제3자/잠긴
            스크립트 채점 또는 저자 불리 방향 확정용으로만.
            **미실행.**
          - **E1-b**(★결정적, ≈0.3–0.5 GPU-hr) — Gate 2-S 4셀 A4를
            `FORCE_PREFILL=1`·**n≥4**로 재실행 — 밀도와 셀 일치를
            동시 만족하는 유일한 설계(구 "후속 E2"의 상위
            버전). G5 UNDETERMINED이므로 force-mode 섭동을 α로
            동반 보고. **미실행.**
          - **E1-c**(★필수, ≈0.2 GPU-hr) — 양성대조: Zamba2 A4
            rate6·n≥4·force=1. G1-b 40(초과) vs 이 캠페인 capscan
            31(미초과) = 문턱을 가로지르는 구간. 없으면
            `VERIFIED_AT_SAMPLED_INSTANTS`는 "한 번도 발화한 적
            없는 스크린의 통과"에 불과(게이트#15). **미실행.**
          - **E1-d**(E1-b에 포함 가능) — Zamba2 r3 여유 13의 정면
            검정(force=1·n≥6·상위 꼬리 직접 추정). **미실행.**
        - ~~**후속 E2**(GPU ≈0.7 hr, 차선) — G1-c를 Gate 2-S 부팅
          구조 그대로 n=3 재실행.~~ → **E1-b로 대체·상위 설계로
          흡수**(force=1·n≥4, 위 참조). **미실행.**
        - ~~**후속 E3**(GPU ≈0.7 hr, 선택, E1 통과 시 불필요) —
          `PDMUX_TRACE_FORCE_PREFILL` 1 vs 0 paired 관측자 부하
          상한 직접 측정.~~ → E1 통과(조건부 채택)로 **불필요
          확정**. 남는 관측자-효과 질문은 E1-b가 force=1 섭동을
          α로 보고하며 부분 흡수.
      - **G1-d**: S3 하드웨어 프로브(`%smid` 샘플링 / CUPTI) —
        selector-level→hardware-level 격상.
      - **하네스**: `gate1_analyze.py`에 grid-completeness 검정
        상시화(`trace_forced==False ⟹ si==1 ∨ si%TRACE_EVERY==0`,
        결측 수 출력) — coverage guard는 꼬리만 잡고 중간 구멍을
        못 잡는다(이번 "유실 없음"의 실제 근거는 coverage가 아니라
        이 검정이었다, 결측 0/22,252). engine-porter 이관:
        `telemetry.py`의 `writer_error` 로깅 + SIGKILL 경로에서
        미호출되는 `close()`. ★신규(2026-08-11, G1-c 후속): (1)
        `g2s_*_telemetry_<arm>_r<rate>_<job>.jsonl` 생성 코드에 rep
        경계 마커 추가(현재 glob 순서가 rep1,rep10,rep2…로
        비정렬 + 최대 1176.957s gap — 이번엔 >5s gap이 전부 pop B에
        charge돼 무해했으나 설계가 아니라 운이다). (2)
        `compute_coverage`(span 기반)에 최대 내부 gap 병기
        (`gate1c_analyze.py:90-99`가 rate2 내부 2.749s=4.10% 공백을
        "100.00%"로 보고 — 이번 판정엔 무영향, 정의가 과대진술,
        `CONSENSUS.md` §3 항목43).
    - **Gate 2 — 4-arm 분해(1 job/모델, ≈1.5 GPU-hr)** [원 설계,
      이력 보존 — 실행판은 `PREREG_GATE2_2026-08-06.md` rev4로
      3회 개정(τ=60·n=10·5셀·Holm 보정)됐고 그 실행판이 2026-08-07
      jobs 875344/875346으로 **완료**됐다. 아래 원문은 최초 설계
      의도만 참조, 수치(n=5·rate{2,3})는 rev4가 아니라 이 문단
      한정]: arm =
      {`plain`, `plain+chunked-1+no-overlap`, `plain+chunked512`,
      `agnostic`} × rate **{2,3}**(+선택 4) × n=5, 동일 seed·노드·job.
      정본 술어 primary. **사전등록 판별 예측(제출 전 고정)**:
      `plain+aux ≤ plain` ⇒ pdmux 기여 하한 = 관측 격차 ⇒ §1-1 "PD
      분리" 문구 부분 해금 / `plain+aux ≈ agnostic`(3% 이내) ⇒ §1-1
      **플래그 아티팩트로 붕괴** / `plain+chunk512`가 agnostic의 3%
      이내 ⇒ §1-1은 "**PD-mux는 head-of-line blocking을 없애는 여러
      수단 중 하나**"로 재작성해야 하며 **논문 신규성 축이 바뀐다**(이
      게이트의 진짜 스테이크). 부트 게이트로
      `plain+--enable-mixed-chunk` 1회 가용성 확인.
      ★★**정정(2026-08-09, doc-steward — 정본 반영 누락 복구):
      Gate 2 본 설계(A2=`plainaux` 포함 4-arm)는 미실행이 아니라
      2026-08-07에 이미 실행됐다** — jobs 875344(Zamba2-2.7B)/
      875346(Granite-4.0-h-micro-base), 5.86 GPU-hr. 아래 "상태
      갱신(2026-08-09)"이 쓴 "Gate 2 본 설계는 여전히 미실행"은
      **그 문장을 쓴 시점에 이미 사실과 달랐다**(875344/875346이
      그보다 이틀 전 완료돼 있었으나 정본에 반영되지 않았던 것) —
      이번 정정으로 해소. 결과: **Primary**(A3=`chunk512` vs
      A4=`agnostic`)는 5셀 전부 A4 유의 우세(chunk512는 pdmux를
      대체 못함, 크기 인용 가능은 Zamba2 r2·Granite r3뿐).
      **Secondary R1′/R2′**(A2 vs A4, §5.2 표 마지막 행)는 §1-1의
      "3-플래그 묶음 처치" 교락 중 pdmux 고유분을 분리 —
      Zamba2 r2 +0.834·r3 +0.930, Granite r3 +0.855·r4 +0.944
      (부호 10/10, F-E clear), r6 +0.960(부호만). **귀속 상한은
      "pdmux 서브시스템 전체"이고 "SM 분할 자체"는 여전히
      미분리 — "PD 분리 자체" 귀속(§1-1 NOT-YET-SUPPORTED)은
      한 눈금도 전진하지 않는다**(이 결론 자체는 아래 "상태
      갱신(2026-08-09)"과 일치, 다만 그 근거였던 "설계 미실행"은
      틀렸다 — 실행은 됐으나 A2-vs-A4 비교가 사전등록 스코어러의
      계산 범위 밖이었을 뿐). 전문 `reports/CONSENSUS.md`
      rev17·§1-1(rev4 본 캠페인 블록)·§3 항목34, 원자료
      `workspace/engine-port/results/p1_gates/gate2/g2_report_
      zamba2-27b_875344.json`·`g2_report_granite-40-h-micro-base_
      875346.json`.

      ★**상태 갱신(2026-08-09) — E-A(jobs 875654/875657/875661)가
      이 Gate 2 설계의 부트 게이트(`plain+--enable-mixed-chunk`
      가용성)와 그 레버 하나(mixed-chunk)의 효과 격리를 실행했다.**
      E-A 결과: mixed-chunk 고유 기여는 유일한
      인용 가능 셀에서 관측 격차의 1.2%뿐(범위 1.2–10.6%, 나머지는
      기존 pdmux-vs-untuned-fused 격차) ⇒ "조율된 fused가 pdmux를
      대체한다"는 사전등록 규칙상 발화하지 않았다(부호 10/10
      agnostic 우세, p=0.00195) — 단 8건 중 7건이 F-E(정상상태)
      미달이라 부호만 인용 가능. 새 기전(신규, 서빙 직접 측정):
      mixed-chunk가 decode를 `ForwardMode.MIXED` extend 경로로
      재라우팅해 배치 크기 비례로 ITL이 악화(cudagraph 가설은
      반증). ★**T2("fused 조율 공간 소진") 주장 금지 확정**: 같은
      기전 축에 미시험 노브 최소 6개가 남아 있다. 후속(T4–T3–E
      순서, 우선순위): **T4-1**(deterministic-inference 재확인,
      <0.5 GPU-hr) → **T3-1/T3-2**(MIXED 배치 조성 계측·micro
      스윕, <1 GPU-hr each, 기전 확정) → **E-C**(등지속가능-rate
      대조, ≈4 GPU-hr, T1 정식 종결) → **E-D**(미시험 fused 레버
      스윕—`--prefill-max-requests`·`--num-continuous-decode-steps`·
      `--chunked-prefill-size` 2048/4096·`--max-running-requests`·
      `--schedule-conservativeness`·`--mamba-scheduler-strategy`,
      ≈8 GPU-hr, T2 종결) → **T3-3**(backend 교차, ≈2 GPU-hr, MIXED
      한계비용 8× 비대칭 귀속) → **T4-2**(비퇴화 프롬프트 G-2,
      ≈1 GPU-hr). ~~Gate 2 본 설계(A2 포함)는 이 순서 뒤에도 별도로
      남아 있다.~~ → ★**정정(2026-08-09)**: Gate 2 본 설계(A2 포함
      4-arm)는 이미 2026-08-07(jobs 875344/875346)에 실행 완료됐다
      — 위 "정정(2026-08-09, doc-steward — 정본 반영 누락 복구)"
      참조. 남은 것은 그 결과의 caveat(provenance·천장 포화·감사
      기록 미확인)뿐, 재실행 불필요. 인용 금지 13건·신규 방법론
      게이트 #17–19는
      `reports/CONSENSUS.md` §1-1(E-A 블록)·§3 항목31–33 참조.
      원자료 `workspace/engine-port/results/p1_gates/gate2/`
      (`PREREG_G2EA_2026-08-07.md`·`g2ea_report_*.json`·
      `g2earun_8756{57,61}.out`·`g2eaprobe_875654*`).
    - **Gate 3 — 나머지 2모델(NemotronH·Falcon-H1) 운영점 대조**:
      "4모델 전부"를 다시 쓰고 싶을 때만 필요. 안 하면 §1-1은 영구히
      2모델 문장.
    - **Gate 4 — sustainable-rate 직접 측정(n≥4)**: r4/r6 크기·용량
      주장을 인용하고 싶을 때만. 현재 권고가 "크기 인용 안 함"이라
      후순위. 하려면 도착창 ≫ drain-tail이 되도록 프롬프트 수를
      rate에 비례(§4.2.1 관례 참조).

11. ★**미제출·감사 차단 사전등록 레지스트리(2026-08-14, B-2, 2026-08-16
    C2-R 추가, 2026-08-16 G16 추가, 2026-08-16 세션4 G13·kernel_mech rev2 추가,
    2026-08-18 G17·E-B1 추가, 2026-08-19 G13 rev2·rev3 추가)**
    — 이 표의 목적은 판정 기록이 아니라
    **다음 세션이 같은 설계를 그대로 재제출하는 것을 막는 것**이다.
    ★**2026-08-19 갱신 — 11행 → 13행**(G13 rev2·rev3 추가, 아래 표
    "G13 job/node축" 행 바로 아래). 아래 13행 중 **10행은 미제출·감사
    차단**이며 "가설이 반증됐다"는 뜻이
    **아니다** — 실험이 애초에 돌지 않았거나(LTSM P1·E-1·C2-R rev1)
    결정 규칙이 자기 데이터를 못 읽는 상태(E1-b/c)이거나 검사가 죽은
    코드(`%smid` R0)이거나 1차 결정량 자체가 REFUTED(gate #13 원설계·
    rev2)·수제 유도가 부활(kernel_mech rev2)·**결정량이 격자 선택으로
    대수적으로 강제되거나 판정 창이 대수로 공집합**(G17·E-B1)해서라서
    **결론 자체가 아직 없다.**
    ★**예외 3건 — C2-R
    rev2·G16·G13 rev3이 이 표에서 GO(또는 GO-with-caveats)를
    받았다**(각각 2026-08-15·2026-08-16·2026-08-19,
    claims-auditor, 범위 한정 감사) — 감사 차단이 아니라 **미제출**일
    뿐이며(C2-R rev2는 스모크 job 883351 대기 중이었다가 아래처럼
    실행 완료됨, G16은 아직 스모크도 미제출, G13 rev3는 재감사만
    받았고 **하네스층 감사·사전등록 문서 자체가 아직 없다**), 이 표에
    남기는 이유는
    "재제출 방지"가 아니라 "다음 세션이 뭘 제출해야 하는지"를 가리키기
    위해서다. ★**G13 rev3은 다른 두 예외와 지위가 다르다** — GO가
    아니라 **GO-with-caveats**이고, 위 "다음 실험 gate" #17 정정이
    적었듯 **이 캠페인은 어느 밴드 점을 사든 gate #13을 닫지 못한다**
    (batch⊗regime 앨리어스 + 노드·날짜 축 미배선, 아래 표 참조). 전문은
    각 PREREG/DESIGN 문서에 있다(여기선 포인터 + 조건 개수만,
    복사 금지). ★**2026-08-16 갱신**: C2-R rev2가 **실행 완료**됐다
    (jobs 883574/883575) — 결과는 M8·Ha8 신규 점추정 2건(§3 항목56/
    CONSENSUS, "8B decode-SM 민감도 측정 노트" 2026-08-16 addendum),
    C2 등급 무변경·인용정지 (a)(b) 둘 다 유효. 이 표의 GO 판정 자체가
    사후 유효했음이 확인됐으나, 결과의 판정력은 §3 항목56(G)/"다음
    실험 gate" #13(신설)의 job/node 축·batch-vs-job 분해 전에는
    제한적이다.

    ★**2026-08-16 갱신(2차, G16)**: gate #16(HI에 d54 이상 decode arm을
    n≥4로 추가, "다음 실험 gate" #16)의 사전등록이 규칙→하네스 2단
    감사(게이트 #34)를 완주했다 — 규칙 감사 **GO** → 하네스 감사
    **NO-GO**(4개 사유, 그중 규칙층 결함 2건 포함 — `PIN_GATE=0.80`이
    7 arm 전부를 배제·H3 검출이 코드상 항등식) → addendum A + 수정
    10/10 → **재감사 GO(스모크 제출 가능)** → addendum B + 수정 5/5.
    **미제출**(예상 비용 ≈4.0–5.5 GPU-hr, 스모크 0.2–0.35).
    ★★**등재 시 반드시 병기**: 이 사전등록이 닫는 것은 R2 결정량②
    (§1-32, `SM합>108`)의 **재정식화판**(`Δ_SLO` 사다리 함수)이고,
    원문 문턱 판본은 **rate 축**(gate #16 이차 표적, HI rate를 용량
    근처로 낮추는 것)에 그대로 남는다 — **"gate #16을 닫았다"고 쓰지
    말 것**(`PREREG_G16_RULES_REV3_2026-08-16.md` §1 C1). ~~분석기는
    아직 존재하지 않는다~~ — **정정, 아래 3차 갱신 참조(작성 완료)**.
    ★부수 정정:
    커밋 `17fcac3`(하네스 6파일)의 제목이 `gate #16 dynamic-vs-static
    grid harness`인데 **G16은 dynamic-vs-static 실험이 아니다**(그건
    HE0 트랙, rev3 §10-7이 명시적으로 분리) — 본문은 정확, 제목만
    오류(history 재작성 안 함, 이 문장이 문서 층 정정 메모).

    ★**2026-08-16 갱신(3차, 세션4)**: 위 2차 갱신이 "분석기 미존재"로
    남긴 결손이 이번 세션에 해소됐다. (i) **스모크 2회**(884292 구
    하네스 sha `ce68de98…` → **하네스 결함 2건 발견** → `37cf6b8` 수정
    → 884320 신 하네스 sha `e75a6f37…` → `G16_SMOKE_OVERALL=PASS`, 비용
    0.143+0.129=0.272 GPU-hr). 결함 1(측정 실패) = addendum A-6/N3
    blacklist-unset이 하네스 내부 변수 `PDMUX_EVAL_DIR`까지 지워
    `sys.path` heredoc이 죽어 `residency_fraction={}` — **`G16_EVAL_DIR`로
    rename**해 해소, FULL 7×4 전체에 무조건 발동하는 경로였음. 결함
    2(★게이트 오설정, 교훈 항목21의 거울상) = 스모크 체커 `9_H2_COUNT_
    AND_RESIDENCY`가 `except Exception: print("{}")`로 삼키고 PASS를
    찍었고 `G16_SMOKE_OVERALL` 논리곱이 item 9를 아예 참조하지 않아
    884292가 `residency_fraction={}`를 출력하면서 통과할 수 있었음 —
    `9a`/`9b`로 분리해 실제 판정 + `OVERALL` 편입으로 해소. ⚠️884292와
    블록 1은 서로 다른 하네스 텍스트로 돌았다(884292 인용 시 병기 필수).
    ★두 스모크 공통: `2_REALIZED_PROBE_EXACT_D74`·`8b_..._D64` 둘 다
    PASS(7 arm 폐기 조건 미발동) · H18 CONFIRMED 승격 근거(위 (2) 참조) ·
    `residency_fraction`(884320, 시간가중) d44
    `{0:0.600, 44:0.045, 108:0.355}`·d64 `{0:0.598, 64:0.067, 108:0.336}`·
    d74 `{0:0.601, 74:0.081, 108:0.317}` — ★이 수치는 **해석 대기
    상태로만 기록**한다(addendum A-2의 "decode-active 표본의 53–68%"와
    분모가 다르고[decode-active 표본 vs 전체 bench 시간] NP=40/ROUNDS=1
    스케일이라 메인 세션이 판정하지 않음, FULL 회수 후 result-analyst/
    claims-auditor 라우팅).
    (ii) **사전등록 분석기 `g16_analyze.py` 작성 완료**(커밋 `4f05dce`) —
    rev3 §4(estimand)·§6(판정 9종·K1·δ·적응규칙)·§7(PC-A~E)·§11(K1–K10) +
    addendum A-2/A-3/B-5(a)2/B-5(a)3 축자 구현, 전부 정본 `pdmux_eval`
    호출. claims-auditor 감사 = **조건부 GO**(기준1·2 PLAUSIBLE(조건부)/
    기준3 CONFIRMED) — 완전 독립 재구현으로 PC-A/D/E 표적 다수를 소수
    셋째 자리까지 재현했으나 F1–F4/S1/S6 6건 지적. **반영 완료**(커밋
    `8311d6c`): F1 bare argmin 복원(K1을 인용 게이트로 분리, PC-C
    5/5 통과 복원) · F2 게이트 대조 실패 시 `campaign`을 `SUPPRESSED`로
    대체 · F3 `grid_hygiene`(`UNBALANCED_GRID`/`INSUFFICIENT_BLOCKS`) ·
    F4 `analyzer_sha256` 자체 기록 · S1 **K8 1ms 사다리가 0.203ms
    밴드(`S_itl=44`, §4-1이 지목한 유일한 payoff 구간)를 통째로 건너뜀**
    → `s_itl_exact_bands` 병기로 해소 · S6 비단조 격자용
    `S_itl_upward_closed`. ★**분석기는 재감사 미실시**(F1–F4 반영분
    자체는 claims-auditor 재검증을 거치지 않음, 여전히 조건부 GO).
    ⚠️메인 세션 오진 1건 기록: *"PC-C 표적 `D_itl=d44`가 §6 K1 게이트를
    통과 못 하므로 사전등록 텍스트 결함"*이라 보고했으나 **절반 오류**
    — 진짜 결함은 분석기가 §4 기호를 게이트-조건부로 조용히 재정의하고
    §7에 없는 4번째 하위주장을 스스로 만들어 실패한 것(아래 방법론
    게이트 #43 참조).
    (iii) **블록 1(job `884336`) 제출** — 세션 종료 시 **PENDING**(8/8
    노드 mix/alloc, 234 job pending). K1(≥3/4 블록)은 블록 1개로는
    원리상 판정 불가 — 결정량 계산은 4블록 완주 후.

    ★**2026-08-17 갱신(4차) — 캠페인 완료 + claims-auditor 적대 감사
    완료, 결과 정본 승격(`reports/CONSENSUS.md` §1-33·§3 항목65·66).**
    4블록 전부 `COMPLETED`(jobs 884336/884410/884411/884412,
    28/28 부팅 `exact=True`, 빌드 드리프트 0, ≈3.9 GPU-hr) →
    `g16_analyze.py`(`analyzer_sha256=2c9626f2…`, C-9(3) 재감사분·
    E-2 등재조건 3건 반영 후 최종) 실행 →
    `G16_RESULTS_2026-08-17.md`(rev1)에 대해 claims-auditor 적대 감사
    → **rev2**(반영, 이 표의 최종 판정). **판정 요약(H-1~H-8)**: H-1
    CONFIRMED(안정성 술어 필수)·H-2 식별 CONFIRMED/"처리량 최적"
    REFUTED·H-3 PLAUSIBLE(조건부)·H-4 CONFIRMED(여유 술어 필수)·
    H-5 (a) CONFIRMED/(b) REFUTED(항등식)/(c) CONFIRMED(재서술)·H-6
    전반부 CONFIRMED/배율 REFUTED·H-7 NOT-YET-SUPPORTED·H-8 CONFIRMED.
    **양 phase `ITL_SATURATED`**(정보 있는 음성, 사전등록 §6 예정된
    결과) — HI `M_ttft` argmin은 내부점 d44로 깨끗이 식별되나(4/4
    블록·부트 1.000), 유일한 payoff 구간(ITL SLO≲58.6ms)에서
    `Δ_SLO`는 exact band(폭 0.092ms)에서만 부호가 뒤집히고 그 부호
    자체는 P(부호>0)=0.632로 **무판정**이다. **정직 고지**: 동률
    가드가 데이터 도착 5.8시간 후 적용(철회·정정), 블록 2–4 동시
    배치는 메인 세션의 제출 방식이 원인(사전등록 순차 제출을
    따랐다면 미발생). 재현 경로 `audit_g16_results_2026-08-17/`
    (독립 재구현, 하네스 3× 부풀림·로더 결함 없음 확인)·
    `residency_scope_2026-08-17/`(A-2 조건부 라벨 크기 실측)·
    `audit_g16_saturation_2026-08-16/`(K11 시뮬)·
    `audit_g16_prereg_c93_2026-08-17/`(C-9(3) 재감사). ★**"gate #16을
    닫았다"고 쓰지 말 것 — 불변**(원문 문턱 판본은 rate 축 잔존).
    전문·정본 승격 금지 목록은 `reports/CONSENSUS.md` §1-33.

    | 사전등록 | 판정 | 치명 결함(요약) |
    |---|---|---|
    | LTSM P1 (`workspace/engine-port/results/ltsm_probe/PREREG_LTSM_P1_PROBE_2026-08-14.md`) | NO-GO(9조건 중 다수 불충족) | `PDMUX_FIXED_DECODE_SM_FILE` 미export ⇒ SM 축 부재. 임계값이 문서 3곳에서 서로 다른 3값 |
    | E1-b/c (`workspace/engine-port/results/p1_gates/gate2/PREREG_G2S_E1B_E1C_2026-08-14.md`) | NO-GO(8조건 중 다수 불충족) | 스코어러 P2가 `M<36`에서 `frac`을 조회하지 않아 REFUTED 판정을 삼킴. 결정량 5개 중 4개가 발화 불가 |
    | `%smid` R0 (`workspace/engine-port/results/smid_census/PREREG_SMID_R0_2026-08-14.md`) | CONDITIONAL-GO(5조건) | 런타임 PTX 검사가 죽은 코드(`smid_l0_census.py:410`의 `JITFunction.cache` 미존재 속성 참조) + 스코어러 fail-open(`:523/:529` 가드가 `None`을 통과) |
    | E-1 (`workspace/engine-port/results/bsweep_regime/PREREG_E1_BSWEEP_REGIME_2026-08-14.md`) | NO-GO(F1–F9) | T8 arm에 SM 순회 훅 부재. `R≈2.1` 유도 입력 3개 오류 |
    | C2-R rev1 (`workspace/engine-port/results/s8_scaleup/PREREG_C2R_RULES_2026-08-15.md`) | NO-GO(5표적) | 결정량("단일 `b*`에서의 arm 간 순서")이 자기가 고치려던 슬라이스 아티팩트를 재생산 — 순서가 b의 함수이고 곡선이 교차(T8 b=1 꼴찌 2.476→b=16 1등 2.388). `b*` 규칙이 분모 미정의로 무이빨 문턱(E1-b/c `M<36` 동형). D3가 표적 arm에서 계산 불가. PIN 0.80을 게이트로 쓰면 3/8 탈락·Ha8 전멸 |
    | ★C2-R rev2 (`workspace/engine-port/results/s8_scaleup/PREREG_C2R_RULES_REV2_2026-08-15.md`) | ★**GO**(2026-08-15, claims-auditor, 범위 한정 — **이 표에서 유일한 GO**) → **★★실행 완료(2026-08-16, jobs 883574/883575)** | 손잡이가 아니라 결정량 자체를 교체(게이트 #35 준수): 4 arm→2 arm(M8·Ha8)·순서→arm별 비의 citability·`b*`=16 고정·PIN을 보고축으로 강등·D3 삭제·정지 (a)는 영구 정지로 포기. 하네스 `s8_c2r.sbatch`+`s8_c2r_client.py`(커밋 `19b8853`). **결과**: `r_M8(16)=3.058`·`r_Ha8(16)=3.114`(t(5) [3.056,3.060]·[3.077,3.155], within-job boot 구간·`n_indep=6`, percentile CI[3.0566,3.0595]/[3.0848,3.1354]는 과소피복이라 본문 인용 금지·원자료 포인터만), 양성대조가 항등식(방법론 게이트#9 아홉 번째 재발, 재구현 스크립트는 저장소 밖), C2 등급 무변경. 전문 `C2R_RESULTS_2026-08-16.md`, `CONSENSUS.md` §3 항목56 |
    | ★★G16 rev3 (`workspace/engine-port/results/slo_sched/PREREG_G16_RULES_REV3_2026-08-16.md`, rev1·rev2는 SUPERSEDED 배너 부착 보존 + addendum C/D/E) | ★★**GO** → 스모크 2회 통과 → **캠페인 완료(2026-08-17, 4블록, jobs 884336/884410/884411/884412)** → `G16_RESULTS_2026-08-17.md` rev1 → **claims-auditor 적대 감사 반영 rev2(정본 승격)** | 규칙층: `Δ=D_itl−D_ttft` 강제표를 설계보다 먼저 명시(`TRUNCATED`/`TRUNCATED_LOW` 사전 선언), 결정량 3종 선언적 병존(1차-A/B/C+2차, 게이트 #35 준수), 원 격자 전부 재실행+블록설계(정방향/역순 쌍, R2 caveat #5 제거), 양성대조 5종. 하네스층 NO-GO 4건은 addendum A로 해소, 재감사가 N7(텔레메트리 비용)을 addendum B로 추가 발견. `Δ_SLO` 단일 60ms 점 결정량을 **사다리 함수**로 교체(C2→C2′) — payoff는 ITL SLO≲58.6ms 구간에만 존재, 사다리 해상도(1ms)가 이 payoff 구간(exact band 폭 0.092ms)을 은폐할 뻔함(방법론 게이트 #46). **결과(2026-08-17)**: 양 phase `ITL_SATURATED`(정보 있는 음성) — HI `M_ttft` argmin=d44(식별 CONFIRMED, "처리량 최적" REFUTED), payoff 구간 `Δ_SLO` 부호 무판정(P=0.632). `gap_upper` 비식별 확인 검사 자신이 항등식이었음이 감사로 드러남(방법론 게이트 #45). H-1~H-8 판정·정직 고지 2건(동률 가드 지연·블록 동시배치 원인)은 `CONSENSUS.md` §1-33. ★**"gate #16을 닫았다"고 쓰지 말 것**(재정식화판만 닫힘, 원문 문턱 판본은 rate 축 잔존). 상세 `handoff-report/session_handoff_2026-08-16.md` §15·§4-1–4-3, `CONSENSUS.md` §1-33 |
    | ★G13 job/node축 (`workspace/engine-port/results/s8_scaleup/DESIGN_G13_JOB_BATCH_2026-08-16.md`) | ★**NO-GO**(2026-08-16, claims-auditor — 지목 주장 2건 둘 다 REFUTED) | (a) "between-job SD of r은 부분 항등식" — REFUTED, 저장소의 유일한 비-축 교차-job 데이터에서 두 다리 job 편차가 사실상 **독립**(비 축 RMS 0.804%≈√2×0.581%), b16에 가장 가까운 셀(T8 b15)은 오히려 반-상쇄(+1.345%) — 문서 §2("구조적으로 0")와 §4(그 비-상쇄 값을 σ_job 사전값으로 사용) 두 절이 동시에 참일 수 없음. (b) "Ha8 검정력 사실상 0(1.01)" — REFUTED, `1.01`은 아카이브 스크립트가 산출하지 않는 값(JSON은 `0.562`, `power()` `target_pct` 기본값 미덮음); 실제 **P(pass)=0.394**, 그중 **93%가 `σ̂=0` 퇴화** ⇒ UB95 규칙이 `MS_B≤MS_W`면 자동 PASS라 **부팅 잡음이 job 신호를 삼킨 런을 "job 축 통제됨"으로 선언**(피복률 Ha8 0.638). F5 표본산정이 df=1·타 arm·b=1 사전값에 걸려 b≥15 셀만 쓰면 M8 결론 전부 FAIL로 뒤집힘(3.06→1.83). F6 재현 경로가 결정 표를 산출 안 함. ★**"gate #13을 닫았다"고 쓰지 말 것**(gate #16과 동형 — healthy 체제 b9/b12 대조 batch가 원 아티팩트에 없어 batch⊗regime 앨리어스). 감사 독립 재구현 5조각 저장소 보존(커밋 `1d57cb6`, `audit_g13_independent_2026-08-16/`). 상세 "다음 실험 gate" #13, `handoff-report/session_handoff_2026-08-16.md` §4-4(a) |
    | ★G13 rev2 (`workspace/engine-port/results/s8_scaleup/DESIGN_G13_JOB_BATCH_REV2_2026-08-17.md`, ★**SUPERSEDED 배너 부착** — rev3를 읽어라) | ★**규칙층 NO-GO**(2026-08-19, claims-auditor, 게이트 #34 1단) | 死因 3건: (A) F2 가드 문턱이 틀렸다 — exact-F 상한의 실제 퇴화점은 `MS_B ≤ MS_W`가 아니라 `MS_B ≤ F₀.₀₅(df_B,df_W)·MS_W`(이 격자에서 `F₀.₀₅`=0.34–0.45)인데, 상한이 양수·유한·피복 정상인 `MS_B/MS_W∈(0.34,1.0]` 구간을 통째로 "측정 실패"로 버렸다(**방법론 게이트 #21의 역방향 재발**). (B) Ha8의 `σ_boot=1.5684%`가 부팅 1개(blk5, 잭나이프 2.46×, 나머지 5개는 0.90–0.98×)의 산물이고 그 부팅은 `C2R_SENSITIVITY_2026-08-16.json`의 `named_exclusion`에 **이미 등재**돼 있었다(**교훈 #31의 내부 재발** — σ_job엔 밴드 3중화, σ_boot엔 민감도 0건인 비대칭). (C) `power2()`가 exact-F를 하드와이어해 §0 헤드라인이 사전등록 규칙(피복 조건부 Satterthwaite/exact-F 선택)의 작동 특성이 아니었다. 상세 `DESIGN_G13_JOB_BATCH_REV3_2026-08-19.md` §1, "다음 실험 gate" #17 |
    | ★★G13 rev3 (`workspace/engine-port/results/s8_scaleup/DESIGN_G13_JOB_BATCH_REV3_2026-08-19.md`) | ★★**GO-with-caveats**(2026-08-19 재감사) — **미제출**(등록 = **11.52 GPU-hr**), **여전히 사전등록 아님**(하네스층 감사 미실시) | 死因 A/B/C 전부 규칙층에서 수리(GPU 0). 수리 후 **전 밴드·전 arm에서 채택 계획의 창이 60 s**가 돼 `σ_boot ∝ 1/√T` 가정이 소멸하고 S0(a)가 임계경로 밖으로 나갔다(재감사 CONFIRMED). ★**밴드 점 판정 = `hi_chi2_upper`**(M8 3.84+Ha8 7.68=**11.52 GPU-hr**) — `mid`(6.72)의 붕괴 σ_boot 1.652%는 점추정 1.568%에서 여유 5%뿐이고 n=6 CI [0.979,3.847]% 대부분에서 무너져 P(pass) 0.21로 `UNDETERMINED` 위험(비대칭: prior 과대추정은 비용만, 과소추정은 캠페인 무효). ★재감사가 메인 세션 수리에서 blocking 4건(B1–B4)을 잡았다: B1 `seq2()`가 死因 A 미수리 상태로 §2.4가 인용(수리 후 Ha8 0.795/0.719/0.466, 인용값 "<0.4"는 2배 stale) · B2 **항등식 3건**(구 PC8·구 S3·구 S4)을 `all_pass`에 편입(**교훈 #9 열세 번째 재발**, `_oc` 실호출 + **변이 테스트**로 반증 가능한 검사로 재작성해 해소) · B3 PC5가 가드 수리에 원리상 무감인데 "수리 검증 증거"로 오독 · B4 `named_exclusion`이 실제로는 부팅 2개(`blk2`,`blk5`)인데 1개로 서술. 전부 수리 완료(커밋 `dee6c80`·`75e7ba3`). ★**이 캠페인은 어느 밴드 점을 사든 gate #13을 닫지 못한다**(batch⊗regime 앨리어스 — healthy 체제 b12=**0**/3064 vs collapsed **58**/1173 + 정본 #13(1)의 **노드·날짜 축 미배선**, job 축만). 산출 `DESIGN_G13_STATS_REV3_2026-08-19.json`. **"gate #13을 닫았다"고 쓰지 말 것 — 불변.** 상세 `DESIGN_G13_JOB_BATCH_REV3_2026-08-19.md` §2.2.1·§4.1·§5, "다음 실험 gate" #17 |
    | ★kernel_mech rev2 (`workspace/engine-port/results/kernel_mech/DESIGN_KERNEL_MECH_REV2_2026-08-16.md`) | ★**NO-GO**(2026-08-16, claims-auditor — 기준1 REFUTED·기준2 PLAUSIBLE·기준3 REFUTED) | F1 `wave_eff`가 ncu 메트릭이 아님(ga100 `--query-metrics` 확인) — 폐기 선언한 수제 유도를 1차 결정량으로 되살림(게이트 #36 死因이 이름만 바꿔 생존). F2 §3.2 축퇴 대수 부호가 반대라 게이트가 위험구간(D=16)을 정확히 통과시킴(게이트 오설정). F3 ★★`_ncu_target.py:68-71`의 CUPTI×green-context 비호환 문장 발견 — 정본이 그 다섯 줄 아래(73-74)만 인용해온 결함 발견(위 "8B decode-SM 민감도 측정 노트" 정정 배너 참조, 참이면 Stage B 전체가 이 기판에서 구성상 불가하나 error code 9는 3가지 경합 귀속이 있어 미확인 리스크로만 등재). F4 `ncu --pid` 부착 옵션이 존재하지 않아 §9-2 재발방지 구조 실행 불가. F5 ncu 기본값 `--clock-control base`가 후보(vi)를 클럭 핀으로 박고 직렬화가 후보(v)의 동거를 소멸시킴. 기준3: provenance 12건이 아니라 감사 출처 11+rev2 자작 1, 재구성이 제약 3건을 느슨화 방향으로 떨어뜨림. GO 경로 = 문서 수정 8건 + Stage 0′ 4프로브(≈40–50분). 상세 `handoff-report/session_handoff_2026-08-16.md` §4-4(b) |
    | ★★kernel_mech P1 프로브 (`workspace/engine-port/results/kernel_mech/p1_probe/p1_greenctx_ncu.sbatch`, 결과 `P1_VERDICT_2026-08-20.md`) | ★★**`UNAVAILABLE (CUPTI×GREEN-CONTEXT)`**(2026-08-20, 메인 세션, job 886718 — 도구 타당성 판정, 성능 판정 아님) — F3의 "미확인 리스크"를 해소 | greenctx 다리: 문서화된 시그니처(exit 9) 정확히 재현. **두 겹 대조**로 귀속 확정 — (a) 다리 간: control(full GPU, 같은 GEMM)은 에러 0건·60행 정상 수집. (b) 다리 내부: 같은 프로세스에서 green ctx 밖 RNG 커널(8행)은 성공, green ctx 위 GEMM만 실패 — 변인은 "green-context 스트림 위인가" 하나뿐. ⇒ **Stage B(SM 제한 하 ncu 커널 내부 카운터) 구성상 불가 확정 → kernel_mech rev3는 Stage A 전용으로 범위 축소**(문서 수정 8건 중 Stage B 대상 최소 5건 적용 대상 소멸). 동반 프로브 886752(non-exclusive)가 `ERR_NVGPUCTRPERM`으로 실패 → 선례 스크립트 `run_ncu_profile.sh:15-17`의 권한 근거("batch면 열린다")가 불충분함을 반증, 실제 구분선은 exclusive+hwperf. GPU 0.032 GPU-hr(886718+886752). ★서술 한계: 성능 판정 아님·green context 실행 자체는 정상(`realized_sm=16`)·내부 기전 미분리·A100-SXM4-80GB/driver 580.105.08/ncu 2025.3.1.0/CUDA 13.0.2/이 클러스터 한정. 상세 `P1_VERDICT_2026-08-20.md`, `P1_886752_REVIEW_2026-08-20.md`, `reports/CONSENSUS.md` §3 항목52 追記(6) |
    | ★★G17 payoff 밴드 (`workspace/engine-port/results/slo_sched/DESIGN_G17_PAYOFF_BAND_2026-08-17.md`) | ★★**규칙층 NO-GO**(2026-08-18, claims-auditor, gate #34 stage 1 — `audit_g17_rules_2026-08-18/` a1–a8) | 死因 3건: (a) `a1_restricted_grid.py` — 제안한 S2 격자 `U={d44,d54,d64,d74}`에서 `D_ttft=44=S_min(U)`가 **양 phase 모두** §3의 `FORCED`(`Δ≥0`) 셀을 재생산 — 결정량이 데이터 관측 전에 격자 선택만으로 부호 강제(같은 4블록을 원 7-arm 격자로 두면 `P(부호>0)` HI 0.632, `U`로 좁히면 0.875 — 격자가 판정을 만든다). (b) sticky 레버 estimand가 **동거(co-residency) 시간이 아니라 단독-at-D 시간(`S_solo`)만** 재는 것으로 확인(`a6_estimand_structure.py`) — 손잡이가 설계 의도와 다른 양을 조작. (c) `M_itl`(요청별 token-ITL p95의 중앙값) estimand가 **이봉 분포에서 검열**됨 — `U` 위 p95는 0.341ms인데 요청별 평균은 2.062ms로 대표성이 없다(`a6`·`a8`). `K1` 순위 규칙도 블록 수 N이 늘수록 식별 확률이 **떨어지는 반직관 성질**(`a2_block_power.py`) 발견. ★**"gate #16을 닫았다"고 쓰지 말 것**(불변, G17은 §1-32/§1-33 재정식화판을 더 좁힌 하위 시도). 상세 `handoff-report/session_handoff_2026-08-18.md` |
    | ★★E-B1 shadow price (`workspace/engine-port/reports/DESIGN_EB1_SHADOW_PRICE_2026-08-18.md`) | ★★**규칙층 NO-GO**(2026-08-18, gate #34 stage 1 감사 — `audit_eb1_rules_2026-08-18/` window_exists 등 7스크립트) | 死因: **판정 가능 창이 대수로 공집합**. `max_running_requests=48`이 모든 28부팅에서 decode 배치를 하드캡해 HI(12 req/s)가 이미 포화(최악) ITL 분포를 관측하는데, 포화 시 ITL-p95 실패율 `q=P(ITLp95>60\|saturated)`가 arm별 ≈0.01–0.10(d34 최저)로 **거의 전부 5% 미만** — `window_exists.py`가 "어떤 rate에서도 조정 가능 창 진입 불가"(`ITL_AXIS_FEASIBLE_AT_ANY_RATE=False`, 다수 arm)를 산출. rate를 낮추면 포화 모집단이 희석돼 `q`가 더 내려갈 뿐이라 구제 불가능. 상세 `handoff-report/session_handoff_2026-08-18.md` |

    감사 보고서(있는 것만): E1-b/c ↔
    `workspace/engine-port/results/p1_gates/gate2/AUDIT_E1B_E1C_2026-08-14.md`,
    C2-R rev1→rev2 ↔ `PREREG_C2R_RULES_REV2_2026-08-15.md` 본문(rev1
    死因 5건을 rev2가 자체 기록, 별도 감사 보고서 파일 없음). LTSM
    P1·`%smid` R0·E-1은 별도 감사 보고서 파일 없음(판정은 각 PREREG
    문서 자체 및 2026-08-14 세션 기록). G13·kernel_mech rev2는 별도
    감사 보고서 파일 없음(판정은 메인 세션이 handoff §4-4에 기록,
    claims-auditor 산출물 자체는 아직 저장소 미편입 — G13 감사 독립
    재구현만 `audit_g13_independent_2026-08-16/`로 보존됨). G17·E-B1은
    별도 감사 보고서 파일 없음(판정 근거는 각 감사 스크립트 세트
    `audit_g17_rules_2026-08-18/`·`audit_eb1_rules_2026-08-18/`의
    `README.txt`+`*.json` 자체, `handoff-report/session_handoff_
    2026-08-18.md`가 종합). **G13 rev2 NO-GO·rev3 재감사(GO-with-caveats)는
    별도 감사 보고서 파일 없음**(판정 근거는 `DESIGN_G13_JOB_BATCH_REV2_
    2026-08-17.md` 상단 SUPERSEDED 배너와 `DESIGN_G13_JOB_BATCH_REV3_
    2026-08-19.md` 본문[死因·B1–B9 반영 내역이 문서 자체에 축자 기록],
    산출물 `DESIGN_G13_STATS_REV2_2026-08-17.json`·
    `DESIGN_G13_STATS_REV3_2026-08-19.json`). ★**방법론 게이트 #34(신규, B-1)
    참조** — 위 표 첫 **4행**(2026-08-14 동시 감사)은 전부 사전등록
    **규칙**이 아니라 **하네스/스코어러 구현 층**에서 차단됐고, 셋(LTSM
    P1·E1-b/c·E-1)은 **도구 자신이 잘못된 라벨을 산출**했다(사람이
    아니라 코드가 거짓 음성/양성). C2-R(rev1 NO-GO→rev2 GO)은 별도 날짜
    (2026-08-15)의 후속 트랙 — 방법론 게이트 #37(신설, 아래) 참조. 상세
    `reports/CONSENSUS.md` rev27·§3 항목49.

    **engine-porter 이관(신규 3건, 2026-08-14, 위 감사에서 발견 — 코드
    미수정)**:
    - ★`sgl_kernel/spatial.py`(venv 사이트패키지본, `create_greenctx_
      stream_by_value` 래퍼)가 `res[0]`/`res[1]`(스트림 포인터)만
      쓰고 **realized SM 수(`res[2]`/`res[3]`)를 버린다** — C++
      (`greenctx_stream.cu:84-97`, `smCountA`/`smCountB`를 반환벡터
      3·4번째 원소로 채움)은 값을 반환하고 있다. E-3(`smsplit_realized_
      probe.py:73-78`)는 래퍼를 우회해 `torch.ops.sgl_kernel.
      create_greenctx_stream_by_value`를 직접 호출해 `res[2]`/`res[3]`를
      읽었다.
    - `smid_l0_census.py:410`이 `census_kernel.cache[dev]`(JITFunction의
      존재하지 않는 속성으로 추정)를 읽어 런타임 PTX 검사가 예외로
      빠지고, `:523`/`:529`의 가드(`runtime_ptx_smid_sites == 0`·
      `runtime_spin_back_edge is False`)가 그 결과인 `None`을 통과시켜
      (`== 0`·`is False` 둘 다 `None`에 거짓) **fail-open**한다.
    - `g2s_e1b_premise.py:295-333`이 `m < THRESHOLD(36)`일 때 `frac`을
      계산하지 않고 `measurable=False, frac_5454=None`으로 채운다(항등식
      전제 — 코드 주석은 "max<36 ⇒ frac(idx2)=0 by identity"라 정당화하나,
      AUDIT_E1B_E1C가 이를 REFUTED 삼킴으로 판정).

    (기존 이관 4건 — `SGLANG_ZAMBA_TIMING=0` truthy · `g2s_analyze.py:84-89`
    `PREMISE_LABEL` · `compute_coverage` 내부 gap · `g2s_*` rep 경계
    마커 — 은 이번 세션에서도 **여전히 미해결**임을 확인만 함, 코드
    미수정.)

12. ★★**(2026-08-15, doc-steward 등재 — claims-auditor 산출, 메인 세션은
    일부만 재확인[아래 명시] — 새 실험 아님) decode batch 도달성의 구조적
    폐쇄(ctx4096 격자에서 관측 — ★2026-08-15 재정정: regime-의존, "ctx-무관
    기전" 아님, 아래 재정정 참조) — B축 설계 시 λ(L)이 작은 격자에
    적용되는 제약.** ctx4096 텔레메트리
    전수 재집계: prefill 16 SM 고정 + `--chunked-prefill-size -1` 격자에서
    SM92 셀의 max(decode_bs)가 T8 4·M8 4·Hs8 4·Ha8 12로 무너진다(SM44는
    T8 8·M8 8·Hs8 6·Ha8 16) — SM92에서 B≥9 도달률은 T8·M8·Hs8 **0.0%**,
    Ha8 **3.0%**뿐.

    | arm | d44 maxB | d92 maxB | d92에서 B≥9 |
    |---|---|---|---|
    | T8 | 8 | 4(p10=p90=4) | 0.0% |
    | M8 | 8 | 4 | 0.0% |
    | Hs8 | 6 | 4 | 0.0% |
    | Ha8 | 16 | 12 | 3.0% |

    ⚠️**provenance**: 메인 세션이 독립 확인한 것은 이 중 **T8 행 하나**뿐
    (`workspace/engine-port/results/bsweep_regime/PREREG_E1_REV3_
    2026-08-15.md:150` "T8은 d92에서 B가 4를 넘은 적이 없다[d44는 8]") —
    M8·Hs8·Ha8 행과 도달률 %는 재현하지 않았다. 기전(메모리 아님 — T8 KV
    풀 910,624 tok로 충분): prefill SM을 고정하면 prefill 서비스율 `λ`가
    상한이 되고 Little's law(`B_decode=λ·T_decode`)로 동시성을 올려도 B가
    오르지 않는다(출력 토큰↑ 레버는 Zamba2 `max_position_embeddings=4096`이
    막음). **등재 명제**: 이 기판(prefill SM 고정·unchunked)에서는 decode
    batch를 제공 동시성으로 임의로 끌어올릴 수 없다 — 목표 B가 도달
    가능한지를 **프로브로 먼저 확인**한 뒤에만 격자를 설계한다(rev2·rev3의
    §3.3 도달성 프로브가 이 교훈을 반영). E-1 계열 rev1–rev3(死因표는
    "방법론 게이트" #35 참조)이 이 제약으로 이 기판에서는 닫혔다 —
    rev4는 prefill SM을 풀거나 다른 B 통제 수단이 먼저 필요하다. 상세
    `reports/CONSENSUS.md` §3 항목51, `handoff-report/session_handoff_
    2026-08-15.md` §2.8·§4.2.

    ★★**(2026-08-15 재정정, doc-steward — result-analyst 1차 addendum을
    claims-auditor가 적대 검증해 반증, 등재 명제 자체가 수정됨) "기전은
    ctx-무관"은 REFUTED — ctx1024가 반례다.** result-analyst의 원
    addendum은 "위 표(max(decode_bs)=T8/M8/Hs8 4·Ha8 12)는 ctx4096
    격자의 값이지만 기전(prefill SM 고정 → λ 상한 → Little's law)은
    ctx-무관"이라 적었으나, claims-auditor의 독립 재집계가 이를
    **반증**했다: **ctx1024(job 865493/865533, C2의 헤드라인 격자)의
    SM92 실현 스냅샷 max(decode_bs)는 865493 21/23/21/23, 865533
    (전 arm) 16**으로 ctx4096의 4(T8/M8/Hs8)/12(Ha8)보다 훨씬 높다 —
    같은 "prefill 16 SM 고정" 기판에서 ctx만 바뀌었는데 폐쇄가
    사실상 풀린다는 것은 "기전이 ctx와 무관하게 항상 구속한다"는
    서술의 **반례**다. **정정된 명제**: B-폐쇄는 **λ(L)이 작을
    때만 구속한다** — `λ(L)`은 prefill 서비스율이고 프롬프트 길이
    L의 함수다. ctx4096(L 길다)처럼 prefill이 느려 λ가 작으면
    Little's law 상한이 낮게 걸려 폐쇄가 강하게 구속하고,
    ctx1024(L 짧다)처럼 prefill이 빨라 λ가 크면 상한이 실질적으로
    안 걸린다. ctx4096에서 관측된 폐쇄 자체는 유효하나, "그 기전이
    ctx에 무관하게 보편적으로 적용된다"는 일반화는 **철회**한다.
    구체적 수치(4/4/4/12, 도달률 %)도 여전히 ctx4096 한정이며
    다른 (B,L) 격자로 이식 금지(게이트 #31). 상세 `workspace/
    engine-port/results/s8_scaleup/
    AUDIT_C2_HEADLINE_JOB_COMPOSITION_2026-08-15.md` §8 항목3(1차,
    반증됨), claims-auditor 적대 검증(원자료 파일 위치 미확정),
    `reports/CONSENSUS.md` §3 항목51 追記(재정정).

13. ★★★**(2026-08-16, doc-steward 등재 — claims-auditor 지정, C2-R
    캠페인 결과에서 도출) C2-R의 M8·Ha8 신규 점추정을 정본 헤드라인과
    비교하려면 아래 두 실험이 선행돼야 한다 — 그 전에는 "3.06 vs
    2.91" 판정을 내리지 않는다.**
    (1) **job/node 축** — 동일 커밋·15파일 매니페스트로 **≥4 job ×
    ≥3 노드 × ≥2 날짜**, arm M8·Ha8, d16/d92, b16, job당 3부팅
    (다리당 12). 1차 결정량 = **between-job SD of r**. ★**이게
    없으면 어떤 CI도 인용 불가** — C2-R의 24부팅은 전부 gpu43
    단일 노드·단일 job·1시간 창 안이라 최외곽 단위가 boot이지
    job/node/day가 아니다(between-job SD 미측정).
    (2) **batch vs job 분해** — **한 job 안에서** conc 계단
    (4/8/12/16)으로 b=9·12·16을 공존시켜 within-job `r(b)`를 얻은
    뒤 865533의 b9/b12와 대조. C2-R의 d16 다리는 `b∈{1,15,16}`뿐이라
    M8 b12·Ha8 b9가 통째로 없어 지금은 batch/job 분해가 원리상
    불가능하다.
    ⇒ 두 실험 전에는 C2-R 값(3.058/3.114)과 정본 헤드라인(2.909/2.852)
    사이에 "위/아래"·"교체"·"확증/반증" 어느 서술도 쓰지 않는다 — **병기만
    한다.** 상세 `reports/CONSENSUS.md` §3 항목56(G), `workspace/
    engine-port/results/s8_scaleup/C2R_RESULTS_2026-08-16.md` §10.

    ★★★**추가(2026-08-16, 세션4, claims-auditor — gate #13 설계 감사
    NO-GO, 위 "다음 실험 gate" #11 레지스트리 표 참조)**: 위 (1)이
    지정한 1차 결정량 **"between-job SD of r"을 실제로 구현한
    `DESIGN_G13_JOB_BATCH_2026-08-16.md`가 감사에서 NO-GO**를 받았다
    — 이 설계 자체를 다음 세션이 그대로 재제출하지 말 것. (a) 그
    설계 문서 §2의 "between-job SD of r은 부분 항등식"이라는 주장은
    **REFUTED**(저장소의 유일한 비-축 교차-job 데이터에서 두 다리
    job 편차가 사실상 독립). (b) 문서의 UB95 검정력 규칙이
    `MS_B≤MS_W`면 자동 PASS라 **부팅 잡음이 job 신호를 삼킨 런도
    "job 축 통제됨"으로 오판**(Ha8 P(pass)=0.394, 그중 93%가
    `σ̂=0` 퇴화, 피복률 0.638) — "이게 없으면 어떤 CI도 인용 불가"라는
    위 경고를 실제로 지키는 검정력 규칙은 **아직 설계되지 않았다**.
    (c) b≥15 셀만 쓰면 M8 결론이 전부 뒤집힘(3.06→1.83)이라 표본
    산정 자체가 취약. ★**"gate #13을 닫았다"고 쓰지 말 것**(gate #16과
    동형 — healthy 체제 대조 batch가 원 아티팩트에 없어 batch⊗regime
    앨리어스). 다음 시도는 이 UB95 규칙을 **부팅 잡음이 아니라 실제
    job 신호를 검정력 있게 잡는 규칙**으로 재설계해야 한다(퇴화
    `σ̂=0` 케이스를 PASS가 아니라 별도 라벨로 분리 권고). 감사 독립
    재구현 5조각은 저장소에 보존됨(`audit_g13_independent_2026-08-16/`,
    커밋 `1d57cb6`).

14. ★**(2026-08-16, doc-steward 등재 — "상금 크기" 논증 §5 R1, GPU 0)
    he2 재채점의 정식 등재 — ✅실행 완료(2026-08-16, result-analyst
    독립 재현 + claims-auditor 적대 감사, `ORACLE_REANALYSIS_2026-08-16.md`
    rev2).** 결정량 (i)(ii) 둘 다 산출됨: (i) COMBINED goodput
    phase-mean **+2.28/+2.35%**(정본 "+2.1%"과 정합) vs 하네스 자신의
    pooled trace-level **+5.88/+6.12%**, (ii) 정본술어(TTFT≤3s ∧
    요청-내부 token-ITL p95≤60ms) 재채점 — **phase B는 5 arm×2 rep
    전부 joint 0/192, 오라클 자체가 미정의**임을 확인. 부수 발견:
    `he2_bench.sbatch:92`가 게이트 #7 버그(`dur=max(dur,d)`) **잔존** —
    `he2_*.out`의 `HE2_RESULT` 라인은 약 3× 부풀려짐(**인용 영구
    금지**, 단 정본 §1-19/§1-20 숫자 자체는 `sum(dur)` 독립 재계산으로
    무사 확인됨). 산출물 반영: `reports/CONSENSUS.md` §1-19(정정
    완료)·§1-20(tie-break·±10% 절벽 caveat 추가). 양성대조 조건(아래
    #15와 공통, 게이트 #9 아홉 번째 재발 대응) 충족 확인 — 표적값
    "2.73"·"1.01 vs 1.09" 원문 재현 + pooled/정본 p95 양쪽 실행.
    ★★**engine-porter 이관 해소(2026-08-16, 커밋 `dadb851`)**:
    `he2_bench.sbatch:92`가 `dur=max(dur,d)` → `dur+=d`로 수정됨(게이트
    #7 버그 잔존분, 같은 수정이 2026-07-17에 `sharegpt_vary_bench.sbatch`
    에만 적용되고 12일간 미전파됐던 것). 과거 `he2_*.out`의
    `HE2_RESULT` 라인 **영구 인용 금지는 불변**(과거 로그 자체는
    여전히 오염된 채 남는다), 정본 §1-19/§1-20 숫자는 이미 `sum(dur)`
    독립 재계산으로 무사 확인돼 재실행·재확인 불요. engine-porter
    이관 목록 **8건 → 7건**(잔여 7건 = "다음 실험 gate" #11 서술의
    기존 이관 4건 + 신규 3건).
15. ★**(2026-08-16, doc-steward 등재 — "상금 크기" 논증 §5 R2, GPU 0)
    `sgptv{Lo,Hi}_{d16,d24,d34,d44}_rep*.jsonl`(arm당 n=4)의 TTFT⊗ITL
    분해 — ✅실행 완료(부분 달성, result-analyst + claims-auditor 적대
    감사, 같은 rev2).** §1-13 각주 E/F의 point-argmax scoped 천장
    진술을 **분해 오라클**로 격상 시도: **결정량①(oracle vs
    best-static 이득)은 산출됨** — 점추정 +1.67%, 재표집 전 구간
    최대 +5.64%로 "+16%" 크기는 **재현되지 않는다**(신규
    `reports/CONSENSUS.md` §1-32, REAL 음성 결과 scoped). **결정량②
    ("116>108 coupling tax 없음")는 미달성** — `SM합>108 ⟺
    D_itl>D_ttft`가 TTFT-argmax=격자 최대 decode arm(d44)이라
    **항등식(검정력 0)**이었음이 드러남(방법론 게이트 #9 열 번째
    재발 + 게이트 #18 사례, `reports/CONSENSUS.md` §3 항목59 신설).
    `PRIZE_SIZE_ARGUMENT` §5가 요구한 **"비-과부하 판본"도 미달성**
    (HI가 2.15–2.35× 과부하). 사전등록 게이트(joint-pass 열린구간
    확인) 충족, 양성대조 조건 충족(#14와 동일). 상세
    `reports/PRIZE_SIZE_ARGUMENT_2026-08-16.md` §5,
    `ORACLE_REANALYSIS_2026-08-16.md` §3.
    ⚠️#14·#15 모두 게이트 #34(규칙 먼저 감사→하네스 그 다음)·#37(합격
    기준+단일 판정 질문 동반)·#39(인용금지 역전파)·#40(신설, 결정량
    자체 항등식 사전 점검) 적용 대상.
16. ★★**(2026-08-16, doc-steward 등재 — R2 결과의 유일한 구조적
    약점에서 도출, 최우선 gate) HI에 d54 이상 decode arm을 n≥4로
    추가하라.** #15의 결정량②가 미확정으로 남은 유일한 이유는 격자
    경계 절단(TTFT-argmax가 이미 최대 decode arm d44)이다 — 이것만
    지우면 "coupling tax 존재/부재"를 처음으로 판별력 있게 물을 수
    있다. d54/d64/d74 config는 이미 존재(미실행). 이차 표적: **HI
    rate를 용량(≈5.5/s) 근처로 낮춰** 과부하 + ITL 절벽을 동시
    이탈시켜 R2 **크기**(현재 인용 금지)를 인용 가능하게 만들 것.
    **후속 캠페인에 `PDMUX_TELEMETRY_PATH` 배선을 필수 선행 조건으로
    건다** — residency·dwell이 R1·R2 하네스 양쪽에서 영구 결손이라
    `controller_summary` 계산이 불가능했다(같은 실패를 3번째 반복하지
    말 것). 상세 `reports/PRIZE_SIZE_ARGUMENT_2026-08-16.md` §5,
    `ORACLE_REANALYSIS_2026-08-16.md` §5.

    ★**2026-08-17 갱신 — 실행 완료(4블록, jobs 884336/884410/884411/
    884412), claims-auditor 적대 감사 반영(`G16_RESULTS_2026-08-17.md`
    rev2). 상태는 "닫힘"이 아니라 "②의 재정식화판 산출 완료, payoff
    구간은 무판정 + 원문 문턱 판본은 rate 축(위 이차 표적)에
    잔존".** d54/d64/d74 arm을 n=4로 추가한 결과 격자 경계 절단은
    해소돼 HI `M_ttft` argmin이 내부점 **d44**로 4/4 블록·부트스트랩
    1.000 식별됐다(단 "처리량 최적"은 REFUTED — 달성 처리량 argmax는
    **d64**, +1.33%). 그러나 양 phase가 `ITL_SATURATED`로 귀결돼
    `Δ`(선호 기반) 자체는 여전히 미식별이고, **재정식화된 결정량
    `Δ_SLO`도 이 캠페인의 유일한 payoff 구간(ITL SLO≲58.6ms,
    exact band 폭 0.092ms)에서 무판정**이다(P(부호>0)=0.632). 60ms
    운영점의 `Δ_SLO=−10`(tax 없음, 10 SM 여유)은 **구 4-arm
    하위격자만으로도 동일한 값**이라 확장이 산 것은 "격자 경계 절단
    해제"라는 판별력이지 새 수치가 아니다. 이차 표적(HI rate 하향)은
    **미실행** — gate #16의 원문 문턱 판정 판본은 여전히 rate 축에
    열려 있다. 상세 `reports/CONSENSUS.md` §1-33(신설), "다음 실험
    gate" #11 레지스트리 G16 행(4차 갱신, 아래).

    ★**2026-08-19 갱신 — 이차 표적(HI rate 하향)을 설계로 옮기려다,
    그 표적이 서로 반대 방향으로 당기는 두 목표를 묶고 있음을
    발견했다(규칙층 초안, 감사 전, GPU 0). gate #16은 여전히 닫히지
    않는다.** 정본 문구 *"HI rate를 용량(≈5.5/s) 근처로 낮춰 과부하
    + ITL 절벽 동시 이탈"*은 두 목표를 한 문장에 묶고 있다: **(i)
    과부하 이탈**(threshold-goodput의 ill-posedness 해소, 방법론
    게이트 #6)에는 `rate↓`가 **옳다**. **(ii) ITL 절벽 이탈 →
    `D_itl`/`Δ` 식별**에는 `rate↓`가 ★**반대 방향**이다 — 정본
    `G16_RESULTS_2026-08-17.json`의 사전등록 결정량 `gap_upper`(상위
    arm 간 `M_itl` 최대 스프레드, 식별 문턱 δ=1.0ms)가 **rate 3에서
    0.304ms, rate 12에서 0.634ms**로 **부하와 함께 커진다**(`g16_
    analyze.py:541-586`). 즉 식별에 가까워지는 방향은 rate↑이지
    rate↓가 아니다. 분석기 자신의 `_saturation_gap`으로 블록별
    재계산해도 방향은 동일(LO 평균 0.415 / HI 평균 1.300)하나,
    ★**HI 평균은 blk1=2.780 하나가 끌어올리는 단일-사건 의존**이라
    (gate #13 rev3 감사가 잡은 것과 구조적으로 동형) **결정 근거는
    pooled 값(0.304/0.634)**이다. ★**단조성은 주장하지 않는다** —
    측정점이 (3, 12) 두 개뿐이고 그 구간의 비단조 가능성은 배제되지
    않았다(노리려면 사전등록 선행). 부수 확정: "용량 ≈5.5/s"는
    **arm 의존 밴드**(G16 achieved: d16 5.084 ↔ d64 5.610)이고
    **d16에겐 이미 과부하**다 — basis 검증은 G16 자신의 아티팩트로
    통과(sgptv 수입 불요, 교훈 #31). ⇒ **실행 가능한 것은 목표 (i)뿐**
    (R2 크기의 인용 가능한 판본, `Δ`/`D_itl` 식별은 1차 결정량에서
    명시적으로 배제). 이 발견은 "다음 실험 gate" #17의 후보 목록에
    누락돼 있었다(아래 #17 2026-08-19 갱신 참조). ★**"gate #16을
    닫았다"고 쓰지 말 것 — 불변**(이 문서는 gate #16을 닫지 않고,
    rate↓로는 닫히지 않는다는 것만 보였다). 상세 `workspace/
    engine-port/results/slo_sched/DESIGN_G18_RATE_AXIS_2026-08-19.md`,
    커밋 `6263d84`.

17. ★★★**(2026-08-18, doc-steward 등재 — G16을 뒤집거나 확장하려는
    시도 5건이 전부 실패한 데서 도출, 새 성능 판정 0건) 열린 실행
    가능 후보는 gate #13 rev2와 kernel_mech rev3뿐이다 — 계획
    재작성이 다음 세션의 첫 일이다.**
    (a) **G17(payoff 밴드+`gap_upper` 비식별+sticky 레버) — ★규칙층
    NO-GO**(`workspace/engine-port/results/slo_sched/
    DESIGN_G17_PAYOFF_BAND_2026-08-17.md`, claims-auditor gate #34
    stage 1 감사 `audit_g17_rules_2026-08-18/`). 死因 3건: ① 제안한
    S2 격자 `U={d44,d54,d64,d74}`에서 `D_ttft=44=S_min(U)`라
    **`Δ_SLO≥0`이 데이터 관측 전에 격자 선택만으로 강제**된다(같은
    4블록으로 원 7-arm 격자 대신 `U`로 좁히면 `P(부호>0)`가 HI
    0.632→0.875로 뜀 — 격자가 판정을 만든다). ② sticky 레버 estimand가
    **동거(co-residency) 시간이 아니라 단독-at-D 시간(`S_solo`)만**
    잰다는 것이 확인됐다 — 손잡이가 설계 의도와 다른 양을 조작.
    ③ **사전등록 estimand(`M_itl`)가 검열**돼 있다 — token-ITL
    분포가 이봉이라 `U` 위 p95는 0.341ms인데 요청별 mean은 2.062ms로
    대표성이 없다. `K1` 순위 규칙도 블록 수가 늘수록 식별 확률이
    떨어지는 반직관 성질이 발견됐다(`a2_block_power.py`).
    (b) **E-B1(shadow price로 "무엇이 SLO를 묶는가") — ★규칙층
    NO-GO**(`workspace/engine-port/reports/
    DESIGN_EB1_SHADOW_PRICE_2026-08-18.md`, gate #34 stage 1 감사
    `audit_eb1_rules_2026-08-18/`). 死因: **판정 가능 창이 대수로
    공집합**이다 — `max_running_requests=48`이 전 28부팅에서 decode
    배치를 하드캡해 HI(rate 12)가 이미 포화(최악) ITL 분포를 관측하고,
    포화 시 ITL-p95 실패율 `q`가 arm별로 대략 d34 0.012·d44 0.021·
    d64 0.043 수준(일부 arm은 최대 0.10대)이라 **완전포화에서도 거의
    전부 5% 미만** — rate를 낮추면 포화 모집단이 희석돼 `q`가 더
    내려갈 뿐이다. ⇒ 고원 arm은 **어떤 rate에서도 판정 불가**
    (`window_exists.py` `ITL_AXIS_FEASIBLE_AT_ANY_RATE=False`, 다수
    arm).
    (c) ★**`workspace/engine-port/reports/EXPERIMENT_PLAN_2026-08-18.md`
    는 stale — 재작성 필요.** P1(G17)이 NO-GO, P2(L축)는 표적이
    `L`→`out`으로 바뀌었다가 그것마저 아래 (d)에서 REFUTED됐다.
    (d) **같은 세션의 자체 재검 4건이 REFUTED, 1건은 조건부 생존**
    (claims-auditor 회부 없이 메인 세션 재검, 인용 시 반드시 술어
    전문 동반): "θ=60은 판별력 없음"은 **REFUTED**(θ=60에서 d16 vs
    d34가 82pp 차이) · split은 높이 아닌 **분율**이라는 주장은
    **REFUTED**(인용한 0.154→0.091은 절벽이 아니라 고원 내부[d34→d74]
    비교였고 고원 내부 상관은 부호 반대) · **"좁아지는 축은 `out`"은
    REFUTED(부호 반대)**(out=32→고원 3/3, out=512→2/3) · **`g2_0_full`
    phase A의 "d34 골짜기"(+33.59pp)는 REFUTED** — 페어드 CI
    [−22.6,+89.8](0 포함)이고, 독립 재현 `g2_0_hard`에서 **부호
    반전**을 보였으며, ★**정본이 이미 `g2_0_hard/
    hardened_disjoint_verdict_2026-07-25.md`에서 rA5 운영점을
    `ILL-POSED`(TTFT 3s-cliff bimodality)로 판정**해 두었고
    `g2_0_full/disjoint_verdict_2026-07-24.md`도 같은 d34 rep1/rep4를
    절벽 사례(t50 3063/3385ms)로 **이름까지 붙여** 이미 기록해 뒀다
    — 그 판정서를 먼저 읽지 않고 재분석한 것이 원인(방법론 게이트
    #49 신설, 최강 사례). 조건부로 살아남은 것(술어 전문 동반 시에만):
    문턱 위치는 **HI 한정·θ=60**에서만 CONFIRMED이고 **LO는 정반대**.
    (e) **미감사로 남은 설계 2건**: **gate #13 rev2**
    (`workspace/engine-port/results/s8_scaleup/
    DESIGN_G13_JOB_BATCH_REV2_2026-08-17.md`, 9.24 GPU-hr, **규칙층
    감사 미실행** — 이번 세션에 유일하게 감사 안 받은 설계, 다음
    세션 첫 후보) · **S-6 telemetry 대조**
    (`workspace/engine-port/results/slo_sched/
    DESIGN_S6_TELEMETRY_CONTRAST_2026-08-17.md`, 규칙층 초안만 —
    검정력 계산으로 `n` 확정이 선행돼야 한다). ★**2026-08-19
    정정 — gate #13 rev2는 감사받았고 NO-GO였다**(아래 2026-08-19
    갱신 참조), "다음 세션 첫 후보"라는 문구는 실현됐으나 결과가
    남긴 것은 rev3다.
    (f) **분석기 누적 미감사**: `g16_analyze.py`의 K11 교체·동률
    가드·E-2 가산 패치·`boot_s` 유도가 전부 **결정량 산출 후**
    감사받았다(addendum C-9(3)만 산출 *전* 감사였다) — 재감사 여부는
    다음 세션 판단 대상.
    ★**"gate #16을 닫았다"고 쓰지 말 것 — 불변**(G17은 그 재정식화판을
    더 좁히려던 하위 시도였을 뿐, 이번 NO-GO가 gate #16 판정에 영향을
    주지 않는다). ★**HE0 확장 금지** — 이 세션 전 캠페인·전 재분석에
    동적 arm 0개. 상세 `handoff-report/session_handoff_2026-08-18.md`.

    ★★★**2026-08-19 갱신(doc-steward — 위 (e)의 목록이 gate #16
    이차 표적을 빠뜨리고 있었다는 것과, gate #13 rev2 감사·rev3
    재감사 결과를 반영. 새 성능 판정 0건, GPU 지출 0)** ★**후보
    목록 정정 — "열린 실행 가능 후보는 gate #13 rev2와 kernel_mech
    rev3뿐"이라는 위 표제 진술은 틀렸다.** item #16 본문이 스스로
    "이차 표적(HI rate 하향)은 미실행 — 원문 문턱 판본은 rate 축에
    열려 있다"고 적어 두었는데 이 항목이 그것을 빠뜨렸다. ★**후보는
    4개다**: (i) **gate #13 rev3**(아래) · (ii) **kernel_mech rev3**
    (문서 8건 미수정 — ★2026-08-20 갱신, "상태 불변"이라는 이 문구는
    더 이상 참이 아니다: P1 프로브가 범위를 Stage A 전용으로 확정했다,
    아래 참조) · (iii) **gate #16 rate 축**(위
    "다음 실험 gate" #16 2026-08-19 갱신 — 목표 (i) 과부하 이탈만
    표적, `Δ`/`D_itl` 식별은 명시 배제, 규칙층 초안·재감사 전) ·
    (iv) **S-6 telemetry 대조**(위 (e), `n` 확정 선행).

    **gate #13 rev2 → 규칙층 NO-GO**(2026-08-19, claims-auditor,
    게이트 #34 1단): 死因 3건 — (A) F2 가드가 `MS_B ≤ MS_W`라는
    **틀린 문턱**(exact-F 상한의 실제 퇴화점은 `MS_B ≤
    F₀.₀₅(df_B,df_W)·MS_W`, `F₀.₀₅`=0.34–0.45)이라 상한이 양수·유한·
    피복 정상인 구간을 통째로 "측정 실패"로 버렸다(방법론 게이트
    #21의 역방향 재발). (B) Ha8의 `σ_boot=1.5684%`가 부팅 1개(blk5)
    산물이고 그 부팅은 `C2R_SENSITIVITY_2026-08-16.json`의
    `named_exclusion`에 이미 등재돼 있었다(교훈 #31의 내부 재발).
    (C) `power2()`가 exact-F를 하드와이어해 §0 헤드라인이 사전등록
    규칙의 작동 특성이 아니었다. `DESIGN_G13_JOB_BATCH_REV2_
    2026-08-17.md`에 **SUPERSEDED 배너 부착 완료**.

    **gate #13 rev3 → GO-with-caveats**(2026-08-19 재감사) —
    ★**등록 = 11.52 GPU-hr**(`hi_chi2_upper` 밴드 점). `mid`(6.72)의
    붕괴 σ_boot 1.652%는 점추정에서 여유 5%뿐이고 n=6 CI
    [0.979,3.847]% 대부분에서 무너져 P(pass) 0.21로 캠페인 전액
    손실 위험(비대칭: prior 과대추정은 비용만, 과소추정은 캠페인
    무효). 수리 후 **전 밴드·전 arm에서 창이 60s**가 돼
    `σ_boot ∝ 1/√T` 가정과 S0(a) 선행조건이 소멸했다. 재감사가
    메인 세션 수리의 blocking 4건(B1–B4)을 추가로 잡았다 — 상세
    "다음 실험 gate" #11 레지스트리 G13 rev3 행. ★★**여전히
    사전등록 아님**(수리 자체가 미감사) — 그리고 ★**이 캠페인은
    어느 밴드 점을 사든 gate #13을 닫지 못한다**(batch⊗regime
    앨리어스 + 정본 #13(1)의 노드·날짜 축 미배선, job 축만).
    **"gate #13을 닫았다"고 쓰지 말 것 — 불변.**

    ★**방법론 게이트 — 교훈 #9의 13번째 재발**: gate #13 rev3
    초판이 반증 불가능한 명제 3건(구 PC8·S3·S4)을 대조/자기검사로
    계수해 `all_pass`에 넣었다 — rev2가 자기 F-역함수 항등식을
    `identity_only`로 분리해 피했던 바로 그 실수다. 재감사가 잡았고,
    S3/S4는 `_oc`를 실제로 호출해 해석적 F 값과 대조하도록
    재작성됐으며 **변이 테스트**(가드를 rev2로 되돌린 사본에서
    S3가 0.53 vs 기대 0.05로 FAIL)로 반증 가능성을 증명했다 — 옛
    항등식판은 그 변이를 통과했을 것이다. 신규 게이트 후보: **"자기
    수리를 검증하는 검사는 그 수리를 되돌린 변이본에서 반드시
    실패해야 한다"**(방법론 게이트 #50 신설, 아래). 부수 등재:
    **B7 재현성 지뢰** — rev2 문서가 지시한 재현 명령이 정본 rev2
    JSON을 수리된 숫자로 덮어쓸 수 있었다(공유 코드 경로).
    `--out`이 기존 파일 덮어쓰기를 거부하도록 수정 + rev2에 배너.

    상세 `workspace/engine-port/results/s8_scaleup/
    DESIGN_G13_JOB_BATCH_REV3_2026-08-19.md`, `DESIGN_G13_STATS_
    REV3_2026-08-19.json`, 커밋 `dee6c80`·`75e7ba3`. ★**"gate #16을
    닫았다"·"gate #13을 닫았다" 금지 — 둘 다 불변**. ★**HE0 확장
    금지** — 이 세션 전 캠페인·전 재분석에 동적 arm 0개.
    `results/slo_sched/`는 12캠페인 447파일 혼합(오인 금지 불변).
    인용정지 전부 유효.

    ★★★★**2026-08-20 갱신(doc-steward — P1 프로브 job 886718 결과
    반영, GPU 지출 0.032 GPU-hr, 새 성능 판정 0건) 후보 (ii) kernel_mech
    rev3의 범위가 정해졌다 — "문서 8건 미수정, 상태 불변"은 더 이상
    맞지 않는다.** `Stage 0′ P1 프로브`(위 두 항목이 판별기로 지정해
    둔 그것)가 실행됐고 판정은 **`UNAVAILABLE (CUPTI×GREEN-CONTEXT)`**
    — greenctx 다리가 문서화된 시그니처(exit 9)로 정확히 실패하고
    control(full GPU, 같은 GEMM)은 에러 0건으로 성공했으며, ★같은
    프로세스 안에서 green ctx 밖 커널은 성공·green ctx 위 커널만
    실패해 귀속이 깨끗하다. ⇒ **Stage B(SM 제한 하 ncu 커널 내부
    카운터 수집)는 이 기판에서 구성상 불가로 확정** — rev2 감사가
    지정한 "문서 수정 8건" 중 Stage B 대상 최소 5건이 적용 대상
    소멸해 **rev3는 Stage A 전용으로 재작성하면 되고 작업량이 크게
    준다**(4세션 이월의 실질 원인 해소). ★이 발견은 gate #13·gate #16
    어느 쪽도 닫지 않는다(그 둘은 이 P1과 독립) — **후보는 여전히 4개**,
    다만 (ii)의 실행 계획이 좁아졌다. ★★서술 한계: 이건 **도구 타당성
    판정이지 성능 판정이 아니다**, green context 실행 자체는 정상
    (`realized_sm=16`), 내부 기전(CUPTI 대 다른 층)은 미분리, 스코프는
    A100-SXM4-80GB·driver 580.105.08·ncu 2025.3.1.0·CUDA 13.0.2·이
    클러스터 한정(이식 금지). 상세 위 "8B decode-SM 민감도 측정 노트"
    실증 확정 배너, "다음 실험 gate" #11 레지스트리 kernel_mech P1
    프로브 행, `workspace/engine-port/results/kernel_mech/p1_probe/
    P1_VERDICT_2026-08-20.md`, `reports/CONSENSUS.md` §3 항목52
    追記(6). ★**"gate #16을 닫았다"·"gate #13을 닫았다" 금지 — 둘 다
    여전히 불변**(이 발견은 그 둘과 무관).

실험·통계·fallback의 상세 정본은
[`EXPERIMENT_ROADMAP.md`](reports/paper/EXPERIMENT_ROADMAP.md)다.

## 운영 규약 (SLURM 제출, 2026-08-14 신설)

**2026-08-12 18:00 KST 이후 뉴론은 `--comment="field=<field>;appl=<program>"`이
없는 job 제출을 거부한다.** 이 프로젝트 값은 `field=efficientai`(Efficient &
Scalable AI Systems) · `appl=pytorch`(SGLang은 `showappl` 목록에 없어 PyTorch
기반으로 신고 — `vllm`은 다른 엔진이라 쓰지 않는다). 필수 조건:

- 지시문은 **`#SBATCH` 블록 안, 첫 실행 라인보다 위**에 있어야 유효하다 —
  Slurm은 첫 비주석 실행 라인 이후의 `#SBATCH`를 무시하므로, 그 아래 있는
  줄은 `grep`엔 걸려도 제출은 거부된다.
- **CLI에서 `--comment`를 덮어쓰지 말 것** — script directive가 있어도 CLI
  인자가 이를 무효화해 거부로 이어진다. 인터랙티브는
  `salloc --partition=amd_a100nv_8 --gres=gpu:1
  --comment="field=efficientai;appl=pytorch"`.
- 새로 만드는 job script도 같은 형식을 따른다.
- 검사·자동 수정: `python3 workspace/engine-port/scripts/bootstrap/
  check_sbatch_comment.py [--fix]`(내용 기반 스캔 — `*.sbatch` glob이 아니라
  `^#SBATCH`로 스크립트를 찾아 `.sh`로 저장된 job script도 잡는다. 첫 실행
  라인 아래 지시문은 `UNPARSED`로 별도 분류).
- 2026-08-14 커밋 `31b3e96`(97 `.sbatch` + 26 `.sh` + `.tmp` 1 + CLI 4경로
  `submit.sh`/`run_pipeline.sh`/`submit_size_sweep.sh`/`interact_scheduler.sh`
  전부 적용)·`86179ec`(conformance checker 추가)로 저장소 전체(124개 job
  제출 경로) 적용 완료. **역사적 `PREREG_*.md`는 옛 형식(`--comment=pytorch`)을
  그대로 인용하며 의도적으로 손대지 않는다**(그 시점에 사전등록된 그대로를
  기록하는 것이 목적, 커밋 메시지가 이를 명시).

전문·환경 셋업은 [`RESUME.md`](workspace/engine-port/RESUME.md) "SLURM
`--comment` (제출 필수)" 참조.

## 방법론 게이트 (2026-07-28 신설, 2026-07-29 #4, 2026-08-02 #5·#6, 2026-08-03 #7 추가·#6 사례 추가·(3차 속행) #8 추가·(4차 속행) #9·#10 추가, 2026-08-05 #9 네 번째 재발 기록·#11 추가·(P1 운영점 대조 감사) #6 새 사례 추가·#12·#13 신설, 2026-08-06 #14 신설[통계 방법 층 정정, CONSENSUS §3 항목27과 대응]·(Gate 1) #9 다섯 번째 재발 기록·#15 신설[시간가중 step-function 추정량의 두 함정, CONSENSUS §3 항목28·29와 대응], 2026-08-07 (G1-b) #16 신설[스코프 확장은 원 격자를 전부 재현하라, CONSENSUS §3 항목30과 대응], 2026-08-09 (E-A) #17–19 신설[게이트를 모든 보고 블록에 걸어라·과부하 arm 비교는 시스템 상수가 아니다·사후 지정 셀 이동, CONSENSUS §3 항목31–33과 대응]·(Gate 2 rev4 본 캠페인 R1′/R2′ 정본 반영 복구) #20 신설[사전등록 분석기가 계산하지 않는 비교는 사후 비교다, CONSENSUS §3 항목34와 대응], 2026-08-09 (HOLB G5 재채점·874601 라벨 정정·scipy 폴백) #21 신설[측정 실패를 게이트 실패로 라벨링 마라 — 6번째 재발, 최초 정식 등재, CONSENSUS §3 항목35와 대응]·#22 신설[통계 라이브러리의 조용한 폴백은 아티팩트에 기록되지 않는다, CONSENSUS §3 항목36과 대응], 2026-08-10 (job 876699 T4-1 TIMEOUT 사후분석·공유 하네스 무한대기 수정) #21에 일곱 번째 재발 追記[이번엔 분석 코드가 거짓 음성(REFUTED)을 산출, CONSENSUS §3 항목35 개정과 대응]·#23 신설[공유 하네스 함수의 무경계 대기는 그 함수를 쓰는 모든 소비자의 위험이다, CONSENSUS §3 항목37과 대응], 2026-08-11 (Gate 2-S 첫 유효 결과, jobs 877756/877757, claims-auditor 적대 감사) #9에 여섯 번째 재발 追記[형식상 두 게이트가 같은 정보를 잼, CONSENSUS §3 항목18 개정과 대응]·#20에 새 사례 追記["식별자 수입 ≠ 거동 수입", CONSENSUS §3 항목34 개정과 대응]·#24 신설[any() over n reps 스크린의 귀무 발화율 1−(1−α)ⁿ, CONSENSUS §3 항목38과 대응]·#25 신설[사전등록이 명시한 진단 필드가 산출되지 않을 수 있다, CONSENSUS §3 항목39와 대응], 2026-08-11 (Gate 2-S 1차 실행 실패 후속, doc-steward) #26 신설[대형 캠페인 제출 전 배관 스모크 규율 — 0.11 GPU-hr 스모크(job 877593)가 6.40 GPU-hr 오판(jobs 877107/877109) 재발을 막음, CONSENSUS §3 항목40과 대응], 2026-08-11 (G1-c, job 877974) #27 신설[결정량의 밀도 의존성을 먼저 따져라, CONSENSUS §3 항목41과 대응]·#28 신설[실험이 무엇을 풀어주는지가 코드 사실인지 추정인지 실행 전에 구별하라, CONSENSUS §3 항목42와 대응]·#29 신설[`compute_coverage`류는 내부 구멍에 맹목이다, CONSENSUS §3 항목43과 대응], 2026-08-11 (E1 addendum, jobs 877756/877757, claims-auditor 적대 감사) #25에 여덟 번째 재발 追記[직전 회차 등재 직후 재발, CONSENSUS §3 항목39 追記와 대응]·#9에 별건 사례 追記[`falsifier()` docstring이 "IMPORTED"라 적었으나 인라인 복사·phase 필터 미적용, CONSENSUS §3 항목18 追記와 대응]·#30 신설[부정 선언문("NO UPGRADE PATH EXISTS") 옆의 새 통계량은 자기인지 자백만으로 재감사를 면제받지 않는다, CONSENSUS §3 항목44와 대응]·#31 신설[bound는 확률모델의 꼬리 분위수여야 하고 인접 관측 최대 차이(점추정)와 구별하라, CONSENSUS §3 항목45와 대응], 2026-08-11(트래픽·roofline 진단, result-analyst, GPU 0) #9에 일곱 번째 재발 追記[서술자 자수 — roofline 탄력도 정합은 항등식, CONSENSUS §3 항목18 追記와 대응]·#32 신설[다른 캠페인·다른 스케일의 보조 수치는 기준(basis) 검증 후 수입하라, CONSENSUS §3 항목46과 대응], 2026-08-14 (doc-steward, 문서 층 정리 — 새 성능 판정 0건) #9·#25 본문에 헤더가 이미 명시했던 追記 2건이 누락돼 있던 것을 CONSENSUS §3 항목18·39 원문 대조로 복원·#26에 追記[발동 조건을 "캠페인 하나"에서 "한 배치로 제출되는 신규/변경 코드 공유 캠페인들의 합"으로 개정 권고, CONSENSUS §3 항목47과 대응]·#33 신설[매니페스트 N/N sha 일치는 런타임 바이트 동일함을 함의하지 않는다 — 재현성 주장 범위를 매니페스트가 실제로 덮는 15파일로 한정, CONSENSUS §3 항목48과 대응], 2026-08-14 (E-3 realized SM count 프로브의 부수 발견, doc-steward, 2차) #32에 새 사례 追記[하드웨어 식별 층 재발 — glogin01(PCIe)에서 읽은 하드웨어를 컴퓨트 노드(SXM4) 캠페인에 잘못 귀속, CONSENSUS §3 항목46 追記와 대응], 2026-08-14 (4건 사전등록 감사 종합, doc-steward, 3차) #32에 새 사례 追記[C2 sd_rep(ε)를 basis 미검증 수입값에서 E-1a 원자료 직접측정값으로 교체 — 실패가 아니라 성공 사례, CONSENSUS §3 항목46 追記와 대응]·#34 신설[사전등록은 규칙 먼저 감사받고 하네스는 그 다음 별도로 감사받아라 — 2단 규율, LTSM P1·E1-b/c·%smid R0·E-1 4건 동시 발견, CONSENSUS §3 항목49와 대응], 2026-08-15 (doc-steward, E-1 rev1–rev3 + kernel_mech 4회 설계 전부 감사 차단에서 도출) #35 신설[개정판에서 손잡이 값을 유지한 채 유도 서사만 바꾸지 마라 — rev2 δ=0.0610/3=0.0203이 rev3에서 "오차예산" 유도로 갈아 끼워졌으나 숫자는 δ=0.020(반올림)으로 그대로였음, CONSENSUS §3 항목53과 대응]·#36 신설[타당성은 도구 문서·메트릭 DB로 확인한 뒤 설계하라 — kernel_mech §3이 존재하지 않는 green-context wave-분모 오염을 피하려다 `wave_eff≡1` 항등식을 만듦(#9 여덟 번째 재발), 확인 비용은 로그인 노드 명령 1줄·GPU 0, CONSENSUS §3 항목53과 대응], 2026-08-16 (doc-steward, C2-R rev1→rev2 5연속 감사 차단 종결에서 도출, 메인 세션 진단·claims-auditor 감사 없음) #37 신설[적대 감사에는 합격 기준과 단일 판정 질문을 함께 줘라 — 범위 없는 적대 검토는 항상 NO-GO를 산출하며 그건 설계 품질 신호가 아니다(단, rev1의 진짜 문제는 감사 과잉이 아니라 메인 세션의 설계 과잉이었음을 병기), CONSENSUS §3 항목55와 대응], 2026-08-16 (2차, C2-R 캠페인 결과, jobs 883574/883575, claims-auditor 적대 감사) #9에 아홉 번째 재발 追記[이번엔 "양성대조" 자체가 항등식 — 대조 코드 경로가 헤드라인 경로와 다르고 표적값 자체가 같은 루프의 또 다른 복사본 산출물, CONSENSUS §3 항목18 追記·56(C)와 대응], 2026-08-16 (3차, doc-steward — "상금 크기" 논증 감사 반영) #38 신설[달성된 동적의 열위(HE0)와 달성 가능한 천장은 다른 명제다 — 정본 §5-5·§1-17 본문이 이 둘을 혼동한 문장을 갖고 있었다(스코프 배너로 정정), CONSENSUS §3 항목57과 대응], 2026-08-16 (4차, doc-steward — C2R_RESULTS.md 역전파 결손, 코디네이터 지적) #39 신설[인용금지·강등 결정은 그 수치를 만든 원 아티팩트 문서로도 역전파하라 — 2회 재발("7.24s": CONSENSUS §1-4 본문→CLAIM_EVIDENCE_MATRIX Claim C·venue_positioning 미전파, 12일 소요; percentile CI: CONSENSUS §3 항목56(A)→C2R_RESULTS_2026-08-16.md §0/§7 미전파, 당일 발견), CONSENSUS §3 항목58과 대응], 2026-08-16 (5차, doc-steward — R1·R2 재분석 정본 승격, `ORACLE_REANALYSIS_2026-08-16.md` rev2, result-analyst + claims-auditor 적대 감사, GPU 0) #40 신설[결정량 자체가 항등식일 수 있다 — R2의 결정량②("SM합>108")가 `D_itl>D_ttft`와 동치인데 TTFT-argmax가 격자 최대 decode arm이라 데이터와 무관하게 거짓이었다(게이트 #9 열 번째 재발), 게다가 이 항등식은 `reports/PRIZE_SIZE_ARGUMENT_2026-08-16.md` §2.3(2)에 이미 문자 그대로 적혀 있었는데 결정량 설계에 적용되지 않았다(게이트 #18 사례) — 결정량 설계 전 "데이터와 무관하게 참/거짓이 되는 극단 사례가 있는가"를 먼저 대수적으로 점검하고 저장소를 grep하라, CONSENSUS §3 항목59(신설)·39(追記, 재현 경로 결손 3회차·이번엔 등재 시점에 닫힘)와 대응], 2026-08-16 (6차, doc-steward — gate #16 사전등록 2단 감사 완주 반영, GPU 0·job 제출 0) #9에 열한 번째 재발 追記[이번엔 검증하는 쪽에서 재발 — 독립 검증 스크립트의 "PASS"가 순열 항등식이었고, claims-auditor 자신이 권고한 대체 결정량도 스스로 반증(C2→C2′), CONSENSUS §3 항목60(신설)과 대응]·#40 본문 문단 복구(헤더엔 있었으나 번호 문단 누락, 새 판정 아님)·#41 신설[telemetry는 공짜 관찰자가 아니다 — 2026-07-15/18 sgptv 격자는 telemetry 없이 돌아 이후 telemetry-ON 캠페인과의 절대값 직접 비교가 빌드 드리프트+계측 오버헤드의 합이 됨, "긴장 A(HE2 vs C2)"가 이 패턴의 기존 사례일 수 있어 목록만 등재(정정은 다음 세션), CONSENSUS §3 항목61(신설)과 대응]), 2026-08-16 (세션4, doc-steward — G16 스모크 2회 + 분석기 작성·감사·반영 + 병행 설계 감사 2건 NO-GO에서 도출, GPU 0.272 GPU-hr[스모크만]·새 성능 판정 0건) #42 신설[게이트가 자기 실패를 성공으로 라벨링할 수 있다 — 교훈 항목21의 거울상, G16 스모크 체커 9번이 예외를 삼키고 PASS를 찍었고 `OVERALL` 논리곱이 그 항목을 아예 참조하지 않았다, CONSENSUS §3 항목62와 대응]·#43 신설[분석기가 사전등록 기호를 조용히 재정의하면 자기 대조를 깨고 중심 산출물을 침묵시킨다 — `g16_analyze.py`가 §4의 `D_itl`을 §6 게이트-조건부로 재정의해 PC-C가 깨지고 §3 강제표가 미식별 시 자동 침묵했다(+메인 세션의 "사전등록 텍스트 결함" 오진 1건 기록), CONSENSUS §3 항목63과 대응]·#44 신설[양성대조의 빈 서명 구멍 — 대조가 estimand를 아예 실행하지 않았는데 통과할 수 있는 설계는 게이트 #9 계열의 취약점이다, `PC-B-neg` + 빈 서명 집합 통과 불가 규칙으로 해소한 사례, CONSENSUS §3 항목64와 대응]), 2026-08-17 (doc-steward — G16 캠페인 완료 반영, claims-auditor 적대 감사, GPU 0·이번 세션 신규 지출 0) #45 신설[감사 목적으로 쓴 검사 자신이 항등식일 수 있다 — 게이트 #40/#9 열두 번째 재발, `gap_upper`의 "노출차만으로 재현된다" 검사가 식 1개·자유모수 1개라 데이터와 무관하게 항상 풀린다, CONSENSUS §3 항목65와 대응]·#46 신설[사다리 해상도가 사전등록된 payoff 구간을 은폐할 수 있다 — 1ms 사다리가 폭 0.092ms 밴드의 부호 반전을 못 봐 초판이 "60ms에서 tax 없음"을 헤드라인으로 냄, CONSENSUS §3 항목66과 대응]), 2026-08-18 (doc-steward — G17·E-B1 규칙층 NO-GO 감사 + 메인 세션 자체 인용금지 위반 발견 + `results/slo_sched/` 디렉터리 오인 발견·정정 + 정본 `ILL-POSED` 판정 무시 재발견에서 도출, 새 성능 판정 0건) #47 신설[인용금지는 정본 등재만으로 전파되지 않는다 — CONSENSUS §33 승격금지(vii)가 33.04 인용을 금지한 다음 날 메인 세션이 새 문서에서 위반(교훈 항목41의 열두 번째 재발) [CS-OK], 기계적 차단 도구 `scripts/discipline/check_citation_stops.py`(추가 줄만 검사, 한계 병기) 신설로 대응, CONSENSUS §3 항목67과 대응]·#48 신설[결과 디렉터리 이름을 캠페인 이름으로 오인하지 마라 — `results/slo_sched/`는 G16 전용이 아니라 12캠페인 447 bench 파일 혼합(g16 68개뿐)인데 이를 오인해 "G16만 밴드질량 이상치"라는 사실 오류를 만듦, 전수 집계로 정정(G16 HI 67.47, 형제 3개가 같거나 위), CONSENSUS §3 항목68과 대응]·#49 신설[정본이 이미 `ILL-POSED`로 판정한 측정점을 재분석에 쓰지 마라 — 게이트 #18 최강 사례, `g2_0_hard/hardened_disjoint_verdict_2026-07-25.md`가 rA5를 TTFT 3s-cliff bimodality로 이미 `ILL-POSED` 판정했고 `g2_0_full/disjoint_verdict_2026-07-24.md`도 같은 d34 rep1/rep4를 절벽 사례로 이름까지 붙여 기록해 뒀는데, 그 판정서를 안 읽고 "d34 골짜기"를 발견처럼 재분석·보고(페어드 CI 0 포함·독립 재현에서 부호 반전으로 자체 반증), CONSENSUS §3 항목69와 대응], 2026-08-19 (doc-steward — gate #13 rev3
재감사, claims-auditor, GPU 0·새 성능 판정 0건) #50 신설[게이트 #9의
열세 번째 재발 — 자기 수리를 검증하는 검사 자신이 반증 불가능한
항등식일 수 있다, rev3 초판이 구 PC8·S3·S4를 `all_pass`에 편입했으나
셋 다 코드 구조상 항상 참이었다(재감사가 `_oc` 실호출 + 변이 테스트로
재작성해 해소, 변이본에서 S3가 실제로 FAIL함을 확인). 신규 게이트:
"자기 수리를 검증하는 검사는 그 수리를 되돌린 변이본에서 반드시
실패해야 한다" — 항등식 혼입을 기계적으로 걸러내는 유일한 방법이며
이번에 실제로 작동했다. 부수 등재(B7): rev2 문서가 지시한 재현 명령이
정본 rev2 JSON을 수리된 숫자로 덮어쓸 수 있는 재현성 지뢰였다 —
`--out`이 기존 파일 덮어쓰기를 거부하도록 수정, CONSENSUS §3
항목70과 대응]

Stage 0/8B de-confound 감사에서 확인된 실패 모드로부터 도출된 3개 항목(1–3),
E1 하네스 구축에서 도출된 상위 원칙(4), 그리고 2026-08-01 캠페인에서 나온
코드 사실 1건(5)과 메타 교훈 1건(6), 그리고 2026-08-03 M3 캠페인에서 나온 항목(7).
`CLAUDE.md`의 기존 8개 게이트에 추가로, 이 정본에 등재한다.

7. ★★★**게이트를 만들 때, 그 게이트가 재는 양이 "게이트가 통과시키려는 조건"과
   논리적으로 독립인지 먼저 증명하라 — 게이트 자신이 #6을 위반할 수 있다.**
   2026-08-03에 `e1_pin_check.py`의 시간가중 pin 게이트가 **항등식**임이 확인됐다
   (telemetry 120파일·77,688 스냅샷에서 `prefill_sms ≠ target ⟺
   decode_running_batch_size == 0`, 양방향 위반 0). 즉 "파티션 제어 실패"로 채점한
   것이 전부 **설계된 decode-empty auto-revert**였다(`multiplexing_mixin.py:773,
   792-794`; `CONSENSUS.md` §1-22가 이미 "의도된 경로"로 기록). 이 게이트는
   **정확성을 강제하려고 만든 것**이고, 그 자신이 게이트 #6을 위반했다.
   ★**역방향 사례도 같은 세션에서 나왔다**: 표본 부족을 막으려 만든
   `MIN_PA_SNAPSHOTS`가 하필 **추정 대상의 여집합**(off-target 스핀이 스냅샷의
   대부분)이라 **핀이 잘 걸린 probe일수록 미측정 판정**을 받았다(UNMEASURABLE
   20/20이 pin 최상위 두 seed에 집중, 최하위 두 seed는 0건).
   ⇒ 실무 규칙: (i) 게이트를 코드로 쓰기 전에 **통과 조건과 측정량의 결합분포를
   손으로 적어보고**, 한쪽이 다른 쪽을 함의하는지 확인한다. (ii) 새 게이트는
   **기존 게이트와 병기**한다(제거하면 이전 보고서가 재현 불가). (iii) 게이트가
   무엇을 **여집합으로** 세고 있지 않은지 확인한다.
   ★**따름정리(별개 발견)**: 같은 감사에서 **decode 축은 게이트가 아예 없었고**,
   셀 라벨의 decode 분할이 decode 작업시간의 **4–19%만 실현**됨이 드러났다
   (`CONSENSUS.md` §1-25). §1-22의 "라벨≠실현" 요구가 **prefill 축에만** 적용돼
   있었던 것 — 축이 둘이면 게이트도 둘이어야 한다.

1. **pin은 policy target이 아니라 realized 파티션으로 검증한다.** telemetry의
   `runtime_snapshot`이 보고하는 `(prefill_sms, decode_sms)`(코드 근거
   `dual_worker.py:608`)를 매 실험에서 재집계해 controller가 지정한 값과
   실제로 일치하는지 확인한다 — 검증 비용은 0이며, 이걸 생략해서 Stage 0의
   D108 앵커가 실은 D16임을 놓쳤다. ★**아래 4번의 특수 사례**(target-vs-realized
   집계 단위 불일치)로 재분류.
2. **파티션 활성률을 사전등록 게이트로 삼는다.** green-context 분할은
   split-prefill 동거 중에만 유효하고, 비면 legacy `adjust_stream_groups`가
   무분할로 되돌아간다 — 활성률이 낮으면 셀 평균이 목표 파티션과 무분할의
   혼합이 된다. 실험 전에 최소 활성률(예: ≥0.60)을 정해두고 미달 셀은 폐기한다.
   ★**2026-07-29 정정 필요**: 이 활성률은 반드시 **시간가중**으로 재정의한다
   (아래 4번 참조) — 스냅샷 **개수** 기반 활성률은 실제값을 16–26× 과소평가할
   수 있다(E1 하네스에서 실측).
3. **keepalive는 짧은 prefill을 자주 넣는 방향으로 설계한다.** 긴 keepalive는
   활성률을 오히려 떨어뜨린다(실측: 0.66–0.93 → 0.32–0.63) — 긴 prefill
   윈도우 동안 decode가 진행되지 않아 시간적으로 분리되기 때문이다.
4. ★★**(2026-07-29) 집계 단위를 먼저 정하고, 그 단위가 추정 대상과 맞는지
   논증하라.** 위 1–3번을 포괄하는 상위 원칙(1번을 특수 사례로 흡수, 지우지
   않음). `results/s8_frontier/` 하네스 구축 중 **집계 단위가 답을 5번 바꿨고
   전부 반대 결론을 낼 뻔했다**:
   1. target vs realized 파티션(위 1번, Stage 0 무효화의 원인).
   2. 게이트 모집단 `prefill_active OR decode_active` vs 조건부 —
      설계상 정상인 decode-only 무분할 윈도우를 pin 실패로 셈: pin **0.029
      FAIL → 0.976 PASS**.
   3. 스냅샷 **개수** 가중 vs **시간** 가중 — `runtime_snapshot`이 이벤트루프
      iteration당 발화해 235ms prefill 스텝과 11ms decode 스텝이 같은 무게를
      가짐: 동시성 **0.015 → 0.25–0.40**(16–26×).
   4. drain 꼬리를 포함한 duration vs 도착 구간만의 duration —
      `achieved_rps` **3.40 → arrival_rps 8.96**.
   5. 스냅샷 vs 에피소드, 그리고 그 안에서 다시 개수 vs 시간 —
      d16 pin_frac **0.583 → 0.847**.

   **따름정리**: 하나의 추정량으로 두 질문에 답하지 마라. "그 파티션에서
   실행됐는가"(게이트, 예: 파티션 점유율/활성률)는 **시간 가중**으로 답해야
   하고, "이 요청의 지연은 어느 파티션 것인가"(귀속)는 **요청별 bracket**으로
   답해야 한다. 후자는 전자의 데이터를 대부분 버리므로(대부분의 스냅샷이
   요청 경계 안쪽이 아니라 사이에 놓임) **게이트에 쓰면 검정력이 무너진다**
   (예: d16 요청-bracket 귀속 n_episodes=8, lower95=0.554 → FAIL — 이유는
   검정력 부족이지 잘못된 SM이 아니다). 상세
   [`workspace/engine-port/results/s8_frontier/`](workspace/engine-port/results/s8_frontier/)
   설계 문서(하네스 내 결함 5종 수정 기록).
5. ★★**(2026-08-02, 코드 사실 — 성능 주장 아님) `--max-running-requests`는
   arm 계열마다 다른 손잡이다.** 근거:
   `sglang/srt/model_executor/model_runner_kv_cache_mixin.py:223-229` —
   `disable_radix_cache ∧ max_running_requests is not None`이면
   `max_mamba_cache_size = max_running_requests`로 설정된다(그 앞 분기
   `:218-222`는 `--max-mamba-cache-size`가 명시된 경우, 뒤의 `else`는 가용
   메모리 ratio 기반). E1/s8 계열 캠페인이 정확히 그 조건이므로 **SSM을
   포함한 arm(M8/Ha8/Hs8)에서 cap은 admission과 mamba state pool 크기를
   동시에** 움직이고, **T8(순수 Transformer)에서는 admission만** 움직인다.
   ⇒ (i) **T8에서 잰 cap 효과는 hybrid로 이전 불가**, (ii) cap을 실험
   파라미터로 쓰려면 `--max-mamba-cache-size`를 전 arm 공통 상수로 **명시
   고정**해 두 축을 분리해야 한다.
   **따름정리 — `kv_mamba_occupancy = 1.0`은 메모리 구속의 증거가 아니라
   항등식이다**(pool 크기 = cap이므로 batch가 cap에 닿으면 정의상 1.0).
   hybrid arm에서 관측된 1.0을 "hybrid는 메모리가 구속한다"의 근거로 쓰지
   않는다. 위 "확정된 결과"의 KV congestion 항목과 함께 읽을 것.
6. ★★★**(2026-08-02) "이 양이 내가 재려는 것과 논리적으로 독립인가"를 먼저
   물어라 — 항등식을 증거로 쓰지 마라.** 위 4번(집계 단위 선확정)과 같은
   뿌리이나 실패 모드가 다르므로 별도 항목으로 둔다. 2026-07-31~08-01
   세션에서 철회된 주장 8건 중 **3건이 정의·항등식을 증거로 착각**한
   것이었다:
   1. `arrival_rps`는 seed로부터 RNG replay로 재생성된 값이라 "5셀 전부
      동일"이 **항등식**(서버에 대한 정보량 0).
   2. `kv_mamba_occupancy = 1.0`은 pool 크기가 cap과 같아서 생기는
      **항등식**(위 5번).
   3. "ITL 구속 rate ⟂ d92 off-cliff"는 ITL 구속 rate가 **공집합**이라
      **공허참**.
   나머지 3건은 **한 arm/셀에서 잰 것을 일반화**한 것이다(batch-cap =
   T8 2셀만). ⇒ 새 지표를 증거로 올리기 전에 **그 지표가 실험 설정으로부터
   해석적으로 결정되는 값이 아닌지** 먼저 확인한다.
   4. ★★★**(2026-08-03) 새 사례 — 항등식에서 파생된 양을 자유 모수처럼
      나누지 마라.** `E1_DECODE_REALIZED`(#7이 확인한 조건부 pin의 결과물,
      4–19%)를 "decode-SM 레버가 `A(D)`에 engagement 비율 `w`만큼만
      반영된다"는 **자유 모수**로 취급해 `A_free(dD)=w·A(D)+(1−w)·A(108)`을
      역산하는 보정 모형을 세웠다가 claims-auditor에 REFUTED됐다 — `w`는
      실은 §1-24(prefill=108−D)가 이미 결정한 duty cycle이라 역산 대상과
      역산 도구가 같은 양이었다. **잡은 순서(재사용 가치)**: (i) 같은
      보정을 레버가 있다고 이미 알려진 **대조군에 적용**해 알려진 값을
      위반하는지 본다(control-arm reductio, T8 corrected g 21–29× vs
      정본 C2 2.36–2.91×로 즉시 사망) — GPU 없이, 기존 872077 텔레메트리
      재사용만으로 수행. 상세 `CONSENSUS.md` §1-26·§3-15, `results/
      s8_frontier/DESIGN.md` §4.3.9.
   5. ★★**(2026-08-05, claims-auditor, P1 운영점 대조 감사) 새 사례 —
      위반 0건인 술어의 goodput은 throughput의 다른 이름이다.** P1
      운영점 대조(jobs 873944/873945)에서 Granite rate3·4는 사전등록
      mean-ITL SLO 위반 요청이 600건 중 0~1건이라 그 술어의
      `goodput = good/total ≈ 1`이고 `goodput/throughput` 자체가
      **항등식**이다. "차이 <3% ⇒ 헤드라인 아님"(게이트 #3) 규칙을 이
      판별력 0인 술어에 그대로 적용하면 **무신호가 강등으로 둔갑한다**
      (다른 술어(p95 token-ITL)로는 같은 셀이 +11.7%~+27.0%, CI 0
      배제). ⇒ goodput 지표를 강등 근거로 쓰기 전에 그 지표가 해당
      셀에서 **판별력을 갖는지**(위반 요청이 실재하는지) 먼저 확인한다.
      상세 `CONSENSUS.md` §1-1, `results/p1_opint/
      P1_OPINT_RESULT_2026-08-05.md` §1.2·§3.4.
8. ★★**(2026-08-03, 같은 날 3차 속행) 끝점 선택이 임계를 정한다 — 그리고
   진단과 처방을 같은 턴에 하면 처방은 자기가 감사한 것이다(두 번째 실증).**
   `c2_anchor.py`의 `G_LEVER=1.41` 시도(위 "8B decode-SM 프론티어"
   "2026-08-03(3차)" 소절 E)에서: 후보 임계가 C2 측정 격자{16,24,44,92}의
   실측값이라는 것만으로는 자유 모수가 아니라고 주장할 수 없다 — **끝점만
   골라도 [1.41, 2.40] 전 구간에 도달 가능**하기 때문에 "어느 두 점을
   비교점으로 쓸지"가 그 자체로 숨은 자유도다(§4.3.10(3)이 이미 다른
   맥락에서 거부한 논거의 재등장). ★**동시에 §7의 실패 모드가 반대 방향에서
   재현됐다**: result-analyst가 p95 음성대조 오염을 **발견**(진단)한 뒤
   primary를 p50으로 바꾸자는 **처방**까지 같은 분석에서 냈고,
   claims-auditor가 그 처방만 세 겹으로 반증했다(관측은 CONFIRMED, 처방은
   REFUTED — 위 소절 D). **규율: 진단자와 처방자를 분리하라.** ★**같은
   회차에 성공 사례도 있었다**: 경합 가설이 공유하는 전제("SPLIT 집단이
   decode-SM 대비를 담고 있다")를 검정 대상에 명시적으로 넣는 규율이
   §0의 2.6× 모순을 GPU 쓰기 전에 드러냈다. 상세 `CONSENSUS.md` §3-16·
   §3-17, `results/s8_frontier/DESIGN.md` §4.3.13.
9. ★★★**(2026-08-03, 같은 날 4차 속행) 자기가 검증하려는 코드를 복사한
   게이트는 항등식에 가깝다 — 대조는 생산자에 걸어라.** `s0_axis_check.py`의
   gate 1은 `m3_conditional.label_probe`의 라벨링 루프를 그대로 복사해
   같은 입력에 대조했다 — 정작 새로 넣은 값(`wmean`, 평균 batch 필드)은
   아무것과도 대조되지 않았다. gate 2는 28.79ms를 만든 바로 그 함수
   (`c2_anchor.collect`)를 호출해 그 값을 "검증"했다 — 구조적으로
   순환이다. 대조는 **같은 코드 경로를 공유하는 다른 분석 스크립트가
   아니라 생산자 자체**에 걸어야 한다는 것을 `S0R_REPLICATION_2026-08-03.md`
   의 gate G-B(내 C2 재구성 ≡ `s0dc_client`의 자기 기록 20/20 정확)가
   보인다. 상세 `CONSENSUS.md` §3-18, `results/s8_frontier/DESIGN.md`
   §4.3.15(d).
   ★**네 번째 재발(2026-08-05, claims-auditor, S2 재현) — 이번엔 게이트가
   순환이 아니라 통째로 항등식이었다.** S2(job 873015)의 사전등록 게이트
   `AMBIG_FRAC`·`MIN_N_SPLIT`·`PREREG_S2` §3.1 일치검사가 16/16
   cell-block 전부 PASS했지만, `E1_DECODE_REALIZED≥0.90`이 통과하는 순간
   SPLIT이 이미 인구의 100.000%/99.969%가 되므로 세 게이트 전부 **실패할
   수 없는 조건**이었다(`p50(SPLIT)=p50(all)`도 정의상 성립). `E1_
   DECODE_REALIZED` 자신도 ON arm 안에서는 `stream_idx=_sticky_fixed_idx`
   코드 불변식이다. ⇒ **이 런에는 결과를 사전에 제약한 게이트가 0개였다**
   — 관측치(28.92ms)의 타당성과는 별개로, "게이트를 전부 통과했다"를
   "결과가 게이트에 의해 제약됐다"로 읽지 않는다. 세 번째 재발은
   `S2_ANALYSIS_2026-08-04.md` §3이 §3.1 하나에 대해 이미 기록했다
   ("third recurrence of methodology gate #9 in this campaign line") —
   이번은 런 전체의 결과 게이트 집합이 구조적으로 공집합이라는 더 넓은
   형태다. 상세 `CONSENSUS.md` §1-31·§3-23, `workspace/engine-port/
   results/s2_sticky/S2_REPLICATION_2026-08-05.md` §7.
   ★★★★★**다섯 번째 재발(2026-08-06, Gate 1, job 874478) — 사전등록이
   실행 전에 "이건 항등식에 가깝다"고 자수했는데도 그 게이트로 판정을
   냈다. 자수는 면죄가 아니다.** `PREREG_GATE1_2026-08-06.md`는 제출
   전부터 코드를 직접 추적해 `adjust_stream_groups()`의 분기 구조상
   "decode-busy ∧ prefill in-flight"가 곧 `idx∈{1,2}`와 사실상
   동치임을 스스로 기록했다(§"게이트가 항등식인가"). 그런데도 그
   조건으로 job 874478을 돌리고 시간가중 100.00%를 "실질 산출"로
   보고했다. **이전 네 번과 다른 각도**: 이전에는 사후에(결과를 본 뒤)
   항등식임이 드러났지만, 이번엔 **사전등록 문서 자신이 실행 전에
   항등식 위험을 명시적으로 자백**했는데도 "그래도 돌린다"는 판단이
   그 자백을 판정의 면책 사유로 썼다. **항등식임이 사전에 확인되면
   게이트를 고치거나 실험을 바꿔야지, 자수만 해두고 원안대로
   실행해서는 안 된다.** 이 job이 그나마 정보를 준 것은 주 조건이
   아니라 여집합 두 갈래(B/C, 아래 #10)였다는 사실이 이 교훈을
   뒷받침한다 — 판별력은 자수한 항등식 조건이 아니라 자수하지 않은
   부분에서 나왔다. 상세 `CONSENSUS.md` §1-1(Gate 1 블록)·§3 항목28,
   `workspace/engine-port/results/p1_gates/gate1/
   PREREG_GATE1_2026-08-06.md` §"게이트가 항등식인가".
   ★★★**여섯 번째 재발(2026-08-11, Gate 2-S 첫 유효 결과,
   claims-auditor 적대 감사) — 이번엔 "독립"이라 이름 붙인 두 게이트가
   같은 정보를 재는 것이었다.** `PREREG_GATE2S_2026-08-09.md` §4.5의
   두 안전장치가 실제로는 같은 3개 꼬리 통계량만 보고 있었고, 그중
   `CONVENTION-SENSITIVE` 판정은 12개 지표 중 9개에서 **비트 동일**
   (항등식) 산출이었다 — 형식상 별개 게이트 두 개지만 정보량은
   하나에 가까웠다. 상세 `CONSENSUS.md` §1-1(2026-08-11 Gate 2-S
   블록)·§3 항목18(개정).
   ★★★**일곱 번째 재발(2026-08-11, 트래픽·roofline 진단, 서술자 자수) —
   이번엔 게이트가 아니라 "확증 서술"이 항등식이었다.** `bytes_step`이
   SM 파티션에 무관(config·체크포인트에서만 정의)이므로 achieved_BW =
   bytes_step/ITL은 **정의상** `|ε_BW| ≡ |ε_ITL|`이다. roofline 역산이
   낸 국소 탄력도(16→24 0.83–0.90, 44→92 0.16–0.41)가 C2의 기존 ITL
   탄력도(16→24 0.77–0.88, 44→92 0.09–0.35)와 "정합"하는 것은 **독립
   확증이 아니라 항등식**이다 — roofline이 실제로 더하는 정보는 **절대
   수준**(achieved_BW가 사양 대역폭의 48–61%)뿐이다. 이번엔 진단을 쓴
   result-analyst 자신이 산출 문서 안에서 이를 자수했다(`TRAFFIC_
   ROOFLINE_DIAGNOSTIC_2026-08-11.md` §6.4) — 6번째 재발까지는 모두
   사후 감사가 잡았으나, 이번은 산출자 자신이 실행 중 인지했다는 점이
   다르다. 상세 `CONSENSUS.md` §3 항목18(추가 追記), 아래 "8B decode-SM
   민감도 측정 노트" C-4.
   ★**[본문 복구: 2026-08-14, doc-steward — 이 섹션 헤더(위)는
   이미 "#9에 별건 사례 追記"를 명시하고 있었으나 본문에 실제
   텍스트가 누락돼 있었다. `CONSENSUS.md` §3 항목18을 정본으로
   삼아 원문 그대로 복원한다. 새 성능 판정 아님.]**
   ★**별건 사례(2026-08-11, E1 addendum, claims-auditor — E1 자신의
   책임 아님)**: `g2s_analyze.py:640-660`의 `falsifier()` docstring이
   "Rules IMPORTED from gate1b_analyze.py"라 적었으나 실제로는 pop-A
   분류 규칙을 **인라인 복사**했고, 생산자(`gate1b_analyze.py`)가
   적용하는 `phase=="benchmark"` 필터를 **적용하지 않는다**. 이
   캠페인은 telemetry 100%가 benchmark phase라 수치 결과에는 영향이
   없었으나(우연), 주석이 실제 코드 계통과 다르다 — engine-porter
   이관 대기(코드 미수정). 상세 `CONSENSUS.md` §3 항목18(별건 追記).
   ★★★**아홉 번째 재발(2026-08-16, C2-R 캠페인, claims-auditor 적대
   감사) — 이번엔 "양성대조" 자체가 항등식이었다.** `s8_c2r_score.py`의
   `cmd_poscontrol`(:270)이 대조에 쓴 코드 경로(`discover("legacy",...)`
   → `score(...)`의 기본값 `keep_slack=False`)는 헤드라인이 실제로
   쓰는 경로(`cmd_c2r`:333, `discover("c2r",...)` →
   `score(..., keep_slack=True)`)와 **다르다** — 대조는 헤드라인 경로를
   **한 번도 실행하지 않았다.** 게다가 대조의 표적값(T8 2.388·Hs8
   2.687) 자체가 `s8_batch_matched.py` 루프의 또 다른 verbatim
   복사본(`audit_c2_job_composition.py`)의 산출물이다 — "독립 검증"이
   세 번째 복사본이 첫 번째 복사본과 일치하는지를 잰 것에 가깝다.
   배제된 오류는 전사·t0 조인·pooling 로직뿐이며, 실제로 남은 공백
   (헤드라인 경로 자체의 검증)을 메운 것은 이 대조가 아니라
   **claims-auditor의 독립 재구현**(해당 코드 미import, 새 스크립트로
   3.057990/3.114061 및 24개 부팅의 n·median 전부 재현)이다. 상세
   `CONSENSUS.md` §3 항목18(追記)·56(C), "8B decode-SM 민감도 측정
   노트" 2026-08-16 addendum.
   ★★★**열한 번째 재발(2026-08-16, gate #16 사전등록 2단 감사) —
   이번엔 감사 대상이 아니라 검증하는 쪽에서 두 번 나왔다.** (i)
   메인 세션이 `g16_arm_order.py --verify`의 "모든 arm 평균 위치
   3.000000 PASS"를 **독립 검증 근거로 사용자에게 보고**했으나,
   역순쌍으로 만든 **어떤** 순열 조합에서도 항상 성립하는 항등식이었다
   (모듈 자신의 docstring이 자인, claims-auditor가 정정). (ii)
   claims-auditor가 1차 감사에서 권고한 `Δ_SLO` 단일 60ms 점
   결정량을, 후속 산출에서 **감사 자신이 스스로 반증**했다(`S_itl`이
   58.65/58.85/60.20ms 위의 문턱 지시함수라 확인 카탈로그 #5[metric
   cliff]가 새 결정량에 그대로 재발, C2→C2′로 사다리 함수 교체).
   #40(2026-08-16, R2 결정량② 항등식)이 이미 "열 번째 재발"로
   기록됐으므로 이 항목은 **열한 번째**다 — 단 #40은 항등식을
   **산출한 쪽**의 사례였고, 이번은 그 항등식을 **검증하려던 쪽**
   (독립 검증 스크립트·감사 자신)이 재발시켰다는 점이 다르다. 실무
   규칙: "독립 검증 PASS"·"감사가 권고한 대체 결정량"도 그 자체가
   데이터와 무관하게 항상 참인지 먼저 점검하라 — 검증자·감사자라는
   역할이 게이트 #9 면역을 주지 않는다. 상세 `CONSENSUS.md` §3
   항목60(신설), `PREREG_G16_RULES_REV3_2026-08-16.md` addendum A-4,
   `handoff-report/session_handoff_2026-08-16.md` §15.6.
   노트" 2026-08-16 addendum.
10. ★★★**(2026-08-03, 같은 날 4차 속행) 여집합 클래스에 음성대조를
    걸어라 — #7과 뿌리는 같고 방향은 반대.** 이 자료를 세 차례(원
    C2→`G_LEVER` 감사, 첫 §0 axis check, 이 세션 자신의 첫 프레이밍)
    통과했지만 아무도 "같은 mode estimator를 UNSPLIT(여집합)에도
    적용해본다"는 한 줄을 하지 않았다 — `PREREG_S0R_MODE_2026-08-03.md`의
    행 4가 그것을 했고, SPLIT의 "특이적" 슬로우 모드가 실은 UNSPLIT에도
    같은 위치·크기로 존재함을 드러내 §0의 강한 재프레이밍을 죽였다.
    #7(게이트가 여집합을 세는 바람에 *실수로* 실패)과 같은 뿌리이나
    방향이 반대다: 이번엔 여집합을 **일부러** 재서 라벨의 배타성을
    검정했고, 그 검정이 **성공**했다(세 차례의 앞선 통과가 놓친 것을
    잡음). 상세 `CONSENSUS.md` §3-19, `results/s8_frontier/DESIGN.md`
    §4.3.15(c)–(d).
11. ★★★**(2026-08-05, α 밴드 부검) 예측 밴드도 이식 금지 규칙의 적용
    대상이다. 밴드는 가장 가까운 기판(같은 trace·같은 batch·같은
    client)에서 뽑아라. 자기가 금지한 이전을 자기 사전등록 예측에
    쓰면, 실험이 성공해도 규칙은 실패한다.** §1-31이 C2→sticky 이식을
    명시적으로 금지해 놓고, 같은 항목이 유일한 반증 실험(고-D 대조,
    job 873921)의 사전등록 밴드 [12,13]ms를 C2 점추정(d92=12.88)에서
    그대로 뽑았다 — 정본이 이미 보유한 더 가까운 앵커(872077의 D108
    우세 10.98, batch·ctx byte-matched)를 썼다면 예측은 11.0–11.4로
    관측(11.26/11.32)과 일치했을 것이다. 실험 자체는 정상 작동했다
    (붕괴 분기 REFUTED, §0 CONFIRMED (scoped) 불변) — 실패한 것은
    **밴드 도출 규율**이지 실험도 §0 판정도 아니다. #1(격자 이전
    confound)의 변종이자 #8(끝점 선택이 임계를 정한다)의 재발. 상세
    `CONSENSUS.md` §1-31·§3-26.
12. ★★**(2026-08-05, claims-auditor, P1 운영점 대조 감사) 임계
    지시함수 판정은 임계 사다리와 큐-성장 검정으로 견고성을 보여라.**
    "절벽 플래그"는 술어에 의존해 뒤집힌다 — P1(jobs 873944/873945)의
    같은 셀이 사전등록 mean-ITL 술어로는 off-cliff, 정본 p95 token-ITL
    술어로는 on-cliff로 갈렸다. 임계 하나로 절벽 여부를 판정하지 말고
    임계 사다리(예: 40–300ms)에서 부호가 유지되는지, 그리고 요청
    launch 순서 전반부/후반부 위반율이 안정적인지(큐 성장 검정)를
    함께 보고한다. 게이트 #6(metric cliff 회피)의 실행 절차를
    구체화한다. 상세 `CONSENSUS.md` §1-1, `results/p1_opint/
    P1_OPINT_RESULT_2026-08-05.md` §1.4–1.5.
13. ★★**(2026-08-05, claims-auditor, P1 운영점 대조 감사) arm 대조가
    엔진 제약으로 다중 플래그를 강제하면 그것은 묶음 처치다.** P1에서
    `--enable-pdmux`는 `--chunked-prefill-size -1`·
    `--disable-overlap-schedule`을 assert로 강제한다
    (`sglang/srt/server_args.py:6125-6137`) — "정책 A 대 정책 B"로
    보고한 대조가 실은 **세 플래그 묶음**과 단일 플래그의 대조였다.
    **운영 주장**("이 설정을 켜면 좋아진다")은 이 상태로도 가능하나,
    **기전 주장**("원인은 X 플래그다")은 분해 arm(각 플래그를 개별
    적용한 중간 arm) 없이는 불가하다. 상세 `CONSENSUS.md` §1-1,
    `results/p1_opint/P1_OPINT_RESULT_2026-08-05.md` §5-2.
14. ★★★★**(2026-08-06, claims-auditor Gate 2 설계 감사 2회 +
    result-analyst 독립 재현) n≤8 반복에서 `paired_bootstrap_ci`/
    `unpaired_bootstrap_ci`(`benchmarks/pdmux_eval/analyze.py:115-142`)
    의 구간을 판정에 쓰지 않는다 — primary는 t-CI, bootstrap은
    병기만.** n=5 percentile bootstrap of the mean(BCa·studentization
    없음)의 실 coverage는 **0.840**(100k trial MC)뿐이라 명목 95%의
    한쪽 오류율이 **≈8.0%**(명목 3.2배)다 — 원인은 seed 고정도 정규
    가정도 아니라 **n=5 그 자체**(n=4 coverage 0.798/n=6 0.859/n=8
    0.888, seed를 풀어도 0.8397로 불변). `unpaired_bootstrap_ci`도
    동일 결함(n=4/arm 0.8556, Welch t 0.9590). **n=5 paired 정확
    부호뒤집기 순열검정의 두측 p 하한 = 2/32=0.0625**이므로 n=5
    paired 셀은 분포무가정으로 p<0.05에 원리적으로 도달 불가하다.
    ★**이미 저장소 안에 같은 진단이 두 번 독립으로 존재했다**
    (`results/s8_frontier/m3_analyze.py`·`results/e1_traceforce/
    tfgate_analyze.py`가 각자 로컬로 t-CI 채택) — 정본 라이브러리와
    P1 판정서에는 반영되지 않고 있었다는 뜻이며, 이는 통계 문제가
    아니라 **도구 규율 실패**다. 재채점 결과(P1 §1-1)와 E1 사전등록
    결정 규칙 제출 선행조건은 "다음 실험 gate" #8 참조. 상세
    `CONSENSUS.md` §1-1(rev13)·§3 항목27, 원자료 `workspace/
    engine-port/results/p1_gates/verify/`.
15. ★★★★★**(2026-08-06, Gate 1, job 874478, claims-auditor) 시간가중
    step-function 추정량의 두 가지 함정 — 둘 다 이번에 실증.**
    (i) **케이던스 불변성을 물리적 불변성의 증거로 쓰지 마라(항등식).**
    샘플링 케이던스를 k배 성기게 하면 이벤트 수는 대략 ÷k, 행당 dt는
    대략 ×k가 되어 시간가중 합(Σ count×dt)이 근사적으로 불변한다 —
    `gate1_analyze.py`의 음성대조 C에서 "케이던스 8↔32에서 불변"이라는
    관측은 이 산술 항등식의 재현일 뿐, 서버의 실제 동작이 케이던스에
    둔감하다는 증거가 아니다. (ii) **행의 dt를 "다음 기록 행까지"로
    주면 비인접 구간이 오염된다.** 스냅샷 간격이 서브샘플링으로
    벌어지면 마지막 스냅샷의 "지속 시간"이 그 뒤에 일어난 다른 상태
    전이까지 흡수한다 — pop A의 `t_total`이 이 오염으로 rate2에서
    35.0%(8.83 s/90 스텝, 단일 최대 1282.8 ms), rate3에서 23.7%
    부풀려졌다. `frac`(비율)은 분자·분모가 같은 오염을 공유해 상쇄
    되므로 강건하지만, **`t_total`(절대량)은 강건하지 않다** — 절대
    시간을 인용할 때는 오염 방향과 크기를 반드시 병기한다. 상세
    `CONSENSUS.md` §1-1(Gate 1 블록)·§3 항목29, 원자료
    `workspace/engine-port/results/p1_gates/gate1/gate1_analyze.py`.
16. ★★★★★**(2026-08-07, G1-b, job 875293) 스코프가 좁은 주장을
    확장할 때는 인접한 한 점이 아니라 원 격자를 전부 재현하라 —
    양이 단조라는 보장이 없다.** Gate 1(rev14 §1-1)의 "단일 분할
    고정"은 Zamba2 rate{2,3}만으로 얻은 결론이었다. `max(decode_
    running_batch_size)`가 rate 2→3에서 9→23으로 급증해 "rate 4에서
    이미 문턱(36) 근처"로 기대됐으나 **실제 rate 4는 18로 rate 3보다
    낮았다**(비단조, 원인 미해명). G1-b가 **rate 4만** 돌렸다면
    `NO_EVIDENCE`가 나와 사전등록의 "확장" 규칙이 발화하고, 정본
    문장의 범위를 873944 전 격자(rate 2–6)로 **넓히는 반대 방향의
    오류**를 저질렀을 것이다 — 실제로는 rate 6에서 `(54,54)`가
    8.37% 등장해 그 확장이 거짓임이 드러났다. 873944의 `MAIN_RATES`
    전 격자 `{2,3,4,6}`을 그대로(순서까지) 재현한 사전등록 설계가
    이 비단조성을 놓치지 않게 막았다. ⇒ 실무 규칙: 스코프 확장
    실험은 경계 인접 한 점이 아니라 원 캠페인의 전 격자를 재현하고,
    중간값에 대해 단조성을 가정하지 않는다. 상세 `CONSENSUS.md`
    §1-1(rev15)·§3 항목30, 원자료 `workspace/engine-port/results/
    p1_gates/gate1/gate1b_result_875293.txt`·`PREREG_G1B_2026-08-07.md`.
17. ★★★★**(2026-08-09, E-A, jobs 875654/875657/875661, claims-auditor)
    게이트를 지표에 걸 때는 primary뿐 아니라 보고되는 모든 블록에
    걸어라.** E-A의 F-E(정상상태 판정) 집행 수정이 사전등록 primary
    두 비교에만 적용되고 `secondary` 블록(throughput/goodput/
    ttft_p95/itl_p95)은 무방비였다 — 그리고 실제로 그 무방비 경로에서
    인용이 일어났다(`g2ea_analyze.py:662-680`). 게이트를 설계할 때는
    그 분석 스크립트가 출력하는 모든 블록을 나열하고 각각에 같은
    정상상태 필터가 적용됐는지 확인해야 한다 — 그러지 않으면 판정
    로직이 옳아도 우회 경로로 비정상상태 수치가 새어나간다. 상세
    `CONSENSUS.md` §1-1(E-A 블록)·§3 항목31.
18. ★★★★**(2026-08-09, E-A) 과부하 arm과 정상 arm을 같은 제공
    rate에서 비교한 수치는 시스템 상수가 아니다.** E-A 8건 중 7건은
    F-E(정상상태 검정)가 발화해 부호만 인용 가능했다 — 런길이
    의존을 직접 실측하면 같은 셀·같은 arm 쌍의 지연차가 프롬프트
    60→120에서 2.3–6.2× 움직이는데, 유일한 정상상태 셀(Granite
    r3)에서는 N=60→120이 1.20×에 그친다. ⇒ 지연·처리량 격차의
    크기를 헤드라인화하려면 먼저 정상상태를 검정하고, 아니면 부호만
    인용하고 크기는 지속가능-rate 직접 대조(E-C)로 미룬다. 상세
    `CONSENSUS.md` §1-1(E-A 블록)·§3 항목32.
19. ★★★★**(2026-08-09, E-A) 사후 지정 셀 이동은 부호를 안 바꿔도
    인용 가능성을 만들 수 있다.** E-A의 유일한 인용 가능 정량치
    (Granite rate3 +0.855)는 사전등록 이후 선행 캠페인 진단을 본
    뒤 r4→r3로 재지정된 셀에서 나왔다 — r4를 유지했다면 이 캠페인은
    인용 가능 셀 0개로 끝났을 것이다(`PREREG_G2EA_2026-08-07.md`
    §3 자기인지 위험 #2). 사후 이동이 결과의 방향을 바꾸지 않았어도,
    그 이동 자체가 "이 캠페인에 인용 가능한 무언가가 있다"는 존재
    명제를 만든 선택압이었다는 사실은 인용 시 반드시 병기한다. 상세
    `CONSENSUS.md` §1-1(E-A 블록)·§3 항목33.
20. ★★★★**(2026-08-09, Gate 2 rev4 본 캠페인 R1′/R2′ 정본 반영
    복구 중 발견) 사전등록 분석기가 계산하지 않는 비교는, 그
    arm·n·raw 데이터가 사전등록됐더라도 사후 비교다.**
    `PREREG_GATE2_2026-08-06.md`(rev4) §5.2 판정표는 "R1′/R2′ —
    A2 vs A4에 같은 TOST/우열 검정을 적용"이라 적어 A2(`plainaux`)를
    arm으로 사전등록했지만, 실제 스코어러 `g2_analyze.py`는
    `tost_equivalence`를 코드 전체에서 1회만 호출하고(`:543`) 그
    입력은 A3-vs-A4(`chunk512`-`agnostic`, `:538-539`)뿐이다 — A2는
    `:734`에 서술 문장으로만 등장한다. **"arm이 사전등록 표에
    있다"와 "그 arm 쌍의 비교가 사전등록 스코어러에 구현돼 있다"는
    다른 명제다.** 항목19(§3 항목33 "사후 지정 셀 이동")의 형제
    사례 — 이번엔 셀이 아니라 비교 자체가 저장된 primary 산출물
    (`per_arm_x60`) 위에서 사후 계산됐다. 실무 규칙: 사전등록
    판정표에 적힌 비교마다 그것을 실제로 계산하는 스코어러 코드
    줄 번호를 대조하라 — 표에 문구가 있다고 스코어러 코드에도
    구현이 있다고 가정하지 마라. 상세 `CONSENSUS.md` §1-1(rev4 본
    캠페인 블록)·§3 항목34.
    ★**새 사례(2026-08-11, Gate 2-S 첫 유효 결과, claims-auditor
    적대 감사) — "식별자 수입 ≠ 거동 수입".** `g2s_analyze.py:63`가
    `MIN_COVERAGE=0.98`이라는 **상수**를 `gate1b_analyze.py:49`에서
    가져왔지만, 그 상수를 실제로 집행하는 **규칙**은 가져오지
    않았다(원본에서 이 상수를 쓰는 곳은 `gate1b_analyze.py:250`
    뿐이고 그 로직은 이식되지 않았다). 이번 캠페인은 4셀 전부
    253–427 에피소드로 여유가 있어 수치 결과에 영향은 없었으나,
    provenance 결함 자체는 실재한다 — 상수 이름이 같다고 그 상수가
    하는 일까지 같이 넘어온 것은 아니다. 상세 `CONSENSUS.md`
    §1-1(2026-08-11 Gate 2-S 블록)·§3 항목34(개정).
21. ★★★★**(2026-08-09, job 874601 G3 라벨 정정 — 최초 정식 등재)
    측정 실패를 게이트 실패로 라벨링하지 마라 — 이 프로젝트의
    서명 오류이며 이 항목까지 6회 재발했다.** `g2holb_report_
    zamba2_874601.json`은 4 arm 전부 `result:"FAIL"`로 저장됐으나
    실제로는 sha 추출기가 응답 스키마를 못 읽은 `KeyError:'text'`
    (하네스 결함)였고 Phase B 자체가 실행되지 않았다 — 출력이
    갈라졌다는 증거(정확성 반증)가 전혀 아니다. 이 패턴은 이번이
    처음이 아니다 — 핸드오프 `session_handoff_2026-08-09.md` §1이
    이미 5회를 서술로 기록했다(텔레메트리 드롭 카운터 자기검열 /
    `UNSCOREABLE`을 강등으로 읽음 / HTTP 400을 G3 FAIL로 / G5의
    귀무-채택형 기준(위 항목20 뒤 A절에서 재채점됨) / 프로브의
    stderr 오염) — 그러나 canon(`CONSENSUS.md` §3·이 목록)에는
    이 패턴을 다루는 번호가 이번까지 없었다(전수 검색 확인,
    2026-08-09) ⇒ 이번이 **최초 정식 등재**이며 카운트는 소급
    반영해 **6**으로 시작한다. 실무 규칙: 게이트/correctness
    스크립트가 `FAIL`을 반환하면, 그 전에 하네스 자체가 데이터를
    만들어냈는지(null 필드·예외·스키마 불일치)부터 확인하라 —
    `FAIL`이라는 문자열은 "가설이 거짓"과 "측정이 실패"를 구분하지
    않는다. 상세 `CONSENSUS.md` §1-1(2026-08-09 두 번째 정본 반영
    건 B절)·§3 항목35.
    ★**갱신(2026-08-10) — 일곱 번째 재발, 질적으로 다른 종.** job
    876699(T4-1)에서 `--time=1:00:00` TIMEOUT의 진짜 원인은
    `ON+chunk512`의 첫 multi-chunk generate가 ~52분 무응답한 단일
    호출 스톨이었다(같은 job `OFF+chunk512`는 동일 프롬프트 ~1초).
    **옛 `g2det_analyze.py`가 그 상태(ON chunk512 `n_total=0`)에
    대해 `REFUTED`를 반환하고 있었다**(`on_clean = n_total > 0
    and ...`이 False로 떨어져 REFUTED 분기로 통과) — 52분짜리
    하네스 스톨이 실질적 음성 결과로 발표될 뻔했다. **재발 1–6과의
    질적 차이**: 1–6은 라벨·해석 오류였고 사람이 문서·보고 단계에서
    잡았다. 이번(7)은 **분석 코드 자신이 거짓 음성(REFUTED)을
    산출**했고, 막은 것은 experiment-runner가 `INCOMPLETE`로
    보고하며 사전등록이 열거하지 않은 조건이라고 채점을 거부한
    실행자 규율이었다 — 도구가 사람보다 관대했던 최초 사례. 사후
    완화가 아니다(사전등록 REFUTED 조건은 "ON이 어느 rep에서든
    self-mismatch≥1", rep 0개는 그 관측 자체가 없어 옛 코드는
    버그였다). 수정(2026-08-10)은 `NO VERDICT (MEASUREMENT
    ABSENT)`를 반환하도록 가드를 추가했고, 데이터가 있을 때의
    CONFIRMED/REFUTED 분기(5-케이스 매트릭스)는 불변임을 확인했다.
    **재발 카운트: 6 → 7.** 상세 `CONSENSUS.md` §1-1(2026-08-10
    세 번째 정본 반영 건 A절), 원자료 `workspace/engine-port/
    results/p1_gates/gate2/g2det_876699.out`·`g2det_876699.err`·
    `g2det_analyze.py`.
22. ★★★**(2026-08-09, engine-porter 발견 + 메인 세션 노출범위
    실측) 통계 라이브러리의 조용한 폴백은 아티팩트에 기록되지
    않는다 — 분석 재현 시 인터프리터 환경을 아티팩트에 남겨라.**
    `g2ea_analyze.py:84-87`는 `scipy` import 실패를 조용히 흡수하고,
    `t_cdf()`(`:156-159`)는 그 경우 모든 df에서 Student-t 대신
    정규 CDF로 근사한다 — 커밋된 `g2ea_report_*.json`의 `p_tost`가
    정확히 `1.0`인 것이 그 흔적이다. `t_ppf()`(`:142-153`)는 df
    1–10·p∈{.95,.975}에 한해서만 하드코드 표로 정확하고 그 밖은
    무조건 `1.96`을 반환한다. 산출물 JSON은 이 인터프리터 상태를
    어디에도 기록하지 않는다 — 메인 세션이 `raw_ci` 폭을 역산해서야
    (implied_t = 2.2621… = t(.975,df=9), 1.96이 아님) CI 자체는
    df=9 표값과 일치해 오염되지 않았음을 사후 확인할 수 있었다.
    **게이트 #14 계열**(n≤8 `paired_bootstrap_ci` undercoverage,
    `CONSENSUS.md` §3 항목27)과 뿌리가 같다 — 통계 함수의 정확도가
    실행 환경(라이브러리 가용성)에 조건부인데 그 조건이 산출물에
    남지 않으면, 나중에 같은 파이프라인을 다른 인터프리터에서
    돌린 사람은 자신이 다른 숫자를 재현하고 있다는 것조차 모른다.
    실무 규칙: 통계 계산에 쓰는 라이브러리가 선택적 의존성이면
    (i) import 성공 여부를 산출 JSON에 필드로 남기고, (ii)
    fallback 경로가 근사임을 stdout에 경고로 남기고, (iii)
    fallback의 유효 df 범위를 코드 주석이 아니라 데이터로 노출하라.
    **이 사건 자체는 정본 수치를 오염시키지 않았다**(이 캠페인의
    관측치가 0.05 경계에서 멀어 판정 불변) — 등재하는 것은 수치
    정정이 아니라 이 도구 규율뿐이다. 상세 `CONSENSUS.md` §1-1
    (2026-08-09 두 번째 정본 반영 건 C절)·§3 항목36.
23. ★★★**(2026-08-10, engine-porter 발견 + 메인 세션 코드 직접
    확인) 공유 하네스의 무한 대기 — 항목21(게이트 #21)과 뿌리
    사건은 같으나 레슨은 다르다.** `g2_holb_phaseA_lib.sh`의
    `greedy_call`이 **`--max-time` 없는 raw curl**이었고, 이
    디렉터리의 **모든 캠페인**(g2ctrl/g2ea/g2holb/g2det)이 이
    함수를 공유해서 쓴다. 소비자 10개를 전수 확인한 결과 **4개가
    타임아웃을 `FAIL`/`SMOKE_FAIL` 계열로 채점**하고 있었다(항목21
    의 `g2det_analyze.py` 포함). 수정: `--connect-timeout 10
    --max-time 180`(env `G2_GREEDY_CONNECT_TIMEOUT`/`G2_GREEDY_
    MAX_TIME`로 재정의 가능), curl 종료코드 28을 `STATUS=TIMEOUT`
    으로 `STATUS=ERROR`와 구별, **SHA를 아예 방출하지 않아**
    mismatch 채점이 구조적으로 불가능, 사이드카 `.status.json`.
    기본값 180s는 **측정 근거**로 정당화됐다 — 이 디렉터리 아카이브
    응답 **n=64**(jobs 874602/874628/874633/874635/875344/875346/
    875610/875611/876699, 4 arm × 2 모델)의 서버측 `e2e_latency`가
    median 0.978s / p90 2.534s / **max 6.605s**(최악 관측의 27배,
    중앙값의 184배) — 메인 세션이 `g2_holb_phaseA_lib.sh:102-139`
    주석의 근거 문단을 코드에서 직접 대조해 확인. 정상 경로는
    아카이브 64개 + 합성 실패 9종 재생으로 **73/73 byte-identical**
    검증됐다(engine-porter 보고 — 메인 세션은 이 재현 스위트 자체를
    독립 재실행하지 않음, 근거 아티팩트 경로 미확인이라 캐비어트로
    남긴다). **실무 규칙**: 공유 하네스 함수 하나가 여러 독립
    캠페인의 correctness 채점 경로에 들어가면, 그 함수의 실패 모드
    (특히 무경계 대기)는 한 캠페인이 아니라 그 함수를 쓰는 모든
    소비자의 위험이다 — 소비자별로 타임아웃을 막지 말고 공유
    지점에서 한 번 막아라. 상세 `CONSENSUS.md` §1-1(2026-08-10
    세 번째 정본 반영 건 B절)·§3 항목37, 원자료 `workspace/
    engine-port/results/p1_gates/gate2/g2_holb_phaseA_lib.sh`.
24. ★★★**(2026-08-11, Gate 2-S 첫 유효 결과, claims-auditor 적대
    감사) `any()` over n reps 형태의 스크린은 귀무 발화율이
    1−(1−α)ⁿ이며, 인용 자격을 사실상 무작위로 배정한다.** Gate
    2-S의 인용 가능 셀 선별 필터가 이 형태였고, α=0.05·n=10에서
    귀무(효과 없음) 상황에서도 **40.1%**(=1−0.95¹⁰, 메인 세션
    재계산 일치)의 확률로 "이 셀은 인용 가능"이 발화한다 — 인용
    가능 셀이 나머지 3셀과 물리적으로 다르다는 근거가 아니라 다중
    시행의 산술적 산물일 수 있다는 뜻이다. 실무 규칙: 여러
    rep·셀에 "하나라도 조건을 만족하면" 식의 OR형 스크린을 인용
    자격 게이트로 쓸 때는 그 스크린 자체의 귀무 발화율을 먼저
    계산하고, 그 값이 무시할 수 없이 크면 그 사실을 인용 시
    병기하라 — 결과 자체를 무효화하지는 않되(부호·크기는 그대로
    유효), "이 셀이 지정 셀"이라는 특권적 지위는 취소된다. 상세
    `CONSENSUS.md` §1-1(2026-08-11 Gate 2-S 블록 금지 항목⑥)·§3
    항목38.
25. ★★★**(2026-08-11, Gate 2-S 첫 유효 결과, claims-auditor 적대
    감사) 코드가 산출하기로 되어 있는 진단 필드가 실제로는 산출되지
    않을 수 있다 — 관측자 대칭 보고도 예외가 아니다.**
    `PREREG_GATE2S_2026-08-09.md` §6.3-6이 명시한 관측자 대칭
    진단 필드(`dropped_events`/`writer_error`)가 이번 캠페인
    산출물에 **산출되지 않았다** — 감사자가 원자료에서 오프라인
    으로 복구해 전부 0임을 확인했다(간극 자체는 실재했다는 뜻,
    값이 0이었던 것은 결과일 뿐 사전등록이 요구한 자동 산출 경로가
    작동했다는 증거가 아니다). 실무 규칙: 사전등록이 "이 필드를
    보고한다"고 적었다고 그 필드가 실제로 산출됨을 가정하지 마라
    — 채점 전에 산출물 스키마를 사전등록 문서와 직접 대조하고,
    빠진 필드는 산출 경로의 결함으로 기록하라. 상세 `CONSENSUS.md`
    §1-1(2026-08-11 Gate 2-S 블록)·§3 항목39.
    ★**[본문 복구: 2026-08-14, doc-steward — 이 섹션 헤더(위)는
    이미 "#25에 여덟 번째 재발 追記"를 명시하고 있었으나 본문에
    실제 텍스트가 누락돼 있었다. `CONSENSUS.md` §3 항목39를
    정본으로 삼아 원문 그대로 복원한다. 새 성능 판정 아님.]**
    ★**여덟 번째 재발(2026-08-11, E1 addendum, 직전 회차 등재
    직후)**: `PREREG_G2S_E1_ADDENDUM_2026-08-11.md` §4가 명시한
    서술 전용 진단("rep 경계에 인접한 행이 pop A인 건수")이
    산출되지 않았다 — `g2s_e1_premise.py:28` docstring은 `classify()`
    를 수입 목록에 적었으나 실제 코드에서 한 번도 호출되지 않는다.
    판정에는 영향이 없었으나(그 진단은 서술 전용이지 판정 입력이
    아님), 같은 서명이 바로 앞 회차(항목39 자신)에 정식 등재된
    직후 재발했다는 사실은 이 실패 모드가 "안다고 없어지지 않는다"는
    걸 보여준다. 상세 `CONSENSUS.md` §3 항목39(追記).
26. ★★★**(2026-08-11, Gate 2-S 1차 실행 실패 후속) 대형 캠페인
    제출 전 배관 스모크를 규율로 등재한다 — 스모크의 PASS는 성능
    판정에 아무 정보도 주지 않는다.** Gate 2-S 1차 실행(jobs
    877107/877109, 6.40 GPU-hr, `COMPLETED exit 0:0`)은 스코어러가
    `KeyError: 'itls'`로 죽어 사전등록 primary를 **0개** 냈다(하네스
    결함 6건, 위 "확정된 결과" 1번 2026-08-10 세 번째 정본 반영 건
    참조). 재실행 전에 돌린 **배관 스모크**(job 877593, 6분 40초 =
    **0.11 GPU-hr**, n=1)가 결함 6(rep glob이 telemetry 파일을
    삼키고, **분류기 자신이 오분류**해 exit 21[GPU 재실행 필요]을
    냈어야 할 자리에 20을 내던 문제)을 잡았다 — 분류기를 그대로
    믿고 재실행했다면 6.40 GPU-hr가 다시 낭비됐을 것이다(**약 60배**
    비용 절감, `handoff-report/session_handoff_2026-08-10.md`
    §2.6–2.7). **발동 조건(구체 — 조건 없는 규율은 지켜지지
    않는다)**: 캠페인의 하네스·스코어러 코드가 **직전 감사 통과
    캠페인과 diff가 있고**(신규 작성 포함), 예상 GPU 비용이
    **≥1 GPU-hr**(스모크 통상 비용 0.1–0.2 GPU-hr의 대략 10배
    이상)이면, 본 제출 전 **최소 rep(n=1–2) 스모크**를 선행한다.
    스모크는 (a) 실제 실행 경로로 (b) 실제 스코어러가 소비할 산출
    스키마를 만들고 (c) 계약 위반(필드 누락·행 수·타입) 0건을
    확인한 뒤에만 본 캠페인을 제출한다. **면제**: 하네스·스코어러가
    이전 감사 통과 캠페인과 diff 0(바이트 동일)이면 생략 가능.
    **이 게이트는 결과 채점 게이트가 아니라 배관 게이트다** —
    PASS는 그 자체로 어떤 성능 판정도 licence하지 않는다. ★**따름
    정리(실무 규율)**: 스모크의 판정 코드 자신을 무조건 신뢰하지
    마라 — 이 사례에서도 분류기 자신이 결함 6(오분류)을 냈고 잡은
    것은 사람(메인 세션)의 원자료 직접 대조였다(위 항목21과 같은
    뿌리). 상세 `handoff-report/session_handoff_2026-08-10.md`
    §2.6–2.7, `CONSENSUS.md` §3 항목40, 원자료 `workspace/
    engine-port/results/p1_gates/gate2/g2ssmoke_verdict_877593.txt`·
    `g2ssmoke_877593.out`·`g2ssmoke_manifest_877593.txt`·
    `g2ssmoke_scoreout_zamba2-27b_877593.txt`(수정 금지·인용만).
    ★**追記(2026-08-14, doc-steward — 2026-08-11 세션이 검토만 하고
    미제출한 4건에서 나온 규율 개정, 새 측정 아님) 트리거 주어가
    단수 "캠페인"이면 <1 GPU-hr 조각으로 쪼개 회피할 수 있다.**
    2026-08-11 세션이 준비했다가 사전등록 부재로 제출하지 않은 GPU
    실험 4건(P1 · E1-b/E1-c · `%smid` P1+P2 · G1-a)은 개별 예상
    비용이 각각 0.1–0.7 GPU-hr로 문턱(≥1 GPU-hr) 아래이지만 **합은
    ≈1.15–1.65 GPU-hr**로 문턱을 넘는다 — 트리거가 캠페인 단위인 채로
    있으면 넷 다 스모크 없이 통과한다. 이 넷은 서로 독립이 아니다:
    같은 매니페스트 트립와이어(`g2s_run.sbatch:81-89`의
    `added==2 ∧ removed==0`, `gate1b_run.sbatch`·`gate1c_run.sbatch`의
    `N_CHANGED==0 ∧ N_ADDED==2`)를 공유하고, "트리 무변경 상태에서
    셋 동시 제출이 코드 사실로 안전"하다는 사실 자체가 이 넷을 한
    묶음으로 묶는 근거였다(`handoff-report/
    session_handoff_2026-08-13.md` §2.5 "GPU 실험 4건 실행 준비
    검토"). **개정**: 발동 조건의 트리거 단위를 "캠페인 하나"에서
    **"한 배치로 제출되는, 신규/변경 코드를 공유하는 캠페인들의
    합"**으로 바꾼다 — 그 합이 ≥1 GPU-hr이면, 배치 안에서
    하네스·스코어러에 신규/변경분이 있는 캠페인마다 최소 rep
    스모크를 선행한다(직전 감사 통과 캠페인과 diff 0인 캠페인은
    기존 면제 유지). 원 게이트 #26 본문(위)은 대체하지 않고 이
    追記로 보완한다 — 이 개정을 적용해 실제로 스모크를 새로 돌린
    사례는 아직 없다. 상세 `CONSENSUS.md` §3 항목47.
27. ★★**(2026-08-11, G1-c, job 877974) 결정량의 밀도 의존성을 먼저
    따져라 — "sparse 텔레메트리로는 낼 수 없다"는 주장은 어느
    통계량에 대한 것인지 먼저 밝혀야 한다.** §8.9.1의 in-job
    반증기가 VERIFIED를 못 내는 것은 설계상 단방향이기 때문만이
    아니라 실제로 밀도(`PDMUX_TRACE_FORCE_PREFILL=0`, ~1/32)에도
    의존했다. 그런데 `max(decode_bs)`는 스케줄 그리드를 population과
    무관하게 표본추출하므로 sparse 텔레메트리에서도 나온다 — Gate
    2-S 자신의 A4 원자료(877756/877757)에서 사후 산출한 값이
    r3=10·r4=13(rep별 [9,13], claims-auditor 산출)이었다. 즉 "전제
    검증의 경험적 절반(`max_decode_bs<36`)은 새 GPU 캠페인 없이 이미
    지불돼 있었다" — G1-c가 실제로 추가한 것은 `frac((54,54))`의
    고밀도 직접 관측과 양성대조뿐이다. 실무 규칙: "이 라벨은 오직
    새 캠페인만 낼 수 있다"는 서술은 반드시 어느 통계량에 대해
    참인지 한정하라. 상세 `CONSENSUS.md` §1-1(2026-08-11 G1-c
    블록)·§3 항목41.
28. ★**(2026-08-11, G1-c) 실험이 무엇을 풀어주는지가 코드 사실인지
    추정인지, 실행 전에 코드로 확인한 뒤 정당화하라.** 이번 세션
    초기 동기 서술("전제가 VERIFIED되면 Gate 2-S 크기 인용 셀이
    1→3개로 는다")은 실행 후 원자료·코드 대조로 반증됐다 — 크기
    인용을 막는 것은 `premise` 라벨이 아니라 독립 산출되는 F-계열
    gate였다(`g2s_analyze.py:1157-1161`). 실험 자체는 0.10 GPU-hr로
    저렴해 피해가 작았지만 원칙은 비용 규모와 무관하다: 실험을
    정당화하는 문장이 "이 실험이 풀어줄 것"이라고 서술하는 대상은
    실행 전에 산출 코드와 직접 대조해 코드 사실인지 서술자의
    추정인지를 구별해야 한다. 상세 `CONSENSUS.md` §1-1(2026-08-11
    G1-c 블록)·§3 항목42.
29. **(2026-08-11, G1-c, engine-porter 이관 대기) `compute_coverage`
    류(span 기반) 지표는 내부 구멍에 맹목이다.** `gate1c_analyze.py:
    90-99`가 rate별 창의 첫·마지막 스냅샷만으로
    "coverage=100.00%"를 냈으나 실제로는 창 내부에 유의한 공백이
    존재했다(예: rate2 내부 최대 gap 2.749s = 창의 4.10%). 이번
    판정에는 영향이 없었다(양성대조·grid-completeness가 독립적으로
    결측 0을 확인) — 무해했던 것은 우연이지 정의의 방어력이 아니다.
    실무 규칙: span 기반 coverage를 보고할 때는 최대 내부 gap을
    함께 병기하라. `pdmux_eval/analyze.py`로 이 지표를 이관할 때
    반영할 것. 상세 `CONSENSUS.md` §1-1(2026-08-11 G1-c 블록)·§3
    항목43.
30. ★★**(2026-08-11, E1 addendum, claims-auditor) 저장소가 이미
    "이 경로는 없다"고 명시적으로 써 둔 지점 옆에서 다른 통계량이
    낸 "경로"는, 자기인지 자백만으로는 재감사를 면제받지
    않는다.** `g2s_analyze.py:682`의 `falsifier()`는
    `"NO UPGRADE PATH EXISTS -- state stays 'unchanged'"`를
    문자 그대로 기록해 둔다(`frac(idx2)` 통계량에 대한 §8.9.1
    안전장치). E1은 다른 통계량(`max(decode_bs)`)으로 사실상의
    업그레이드를 냈고, prereg 자신도 §1에서 "이 통계량엔 §8.9.1의
    단방향 안전성이 자동 상속되지 않는다"고 자백하며 재감사를
    요청했다 — 그 자백 덕에 이번엔 위험이 감사로 흡수됐다. 실무
    규칙: (i) 사전등록/addendum이 기존 negated-invariant를
    우회하는 새 통계량을 도입하면 그 독립성을 코드 대조로 확인할
    때까지 감사자 최우선 검토 항목으로 표시한다. (ii) 자기인지
    자백은 필요조건이지 충분조건이 아니다(#9의 구체화). 상세
    `CONSENSUS.md` §1-1(2026-08-11 E1 addendum 블록)·§3 항목44.
31. ★★**(2026-08-11, E1 addendum, claims-auditor) 구간 안에서
    관측되지 않은 sup을 bound하려면 표본 사이 최대 변동(점추정)이
    아니라 그 구간에 성립하는 확률모델에서 유도한 상한을 써야
    한다.** 메인 세션이 처음 제안한 "인접 관측 사이 최대 변화량"
    류 bound는 claims-auditor가 기각했다 — 국소 변동성을 재는
    점추정이지 32-스텝 샘플링 간격 내부에서 도달했을 수 있는 진짜
    sup의 확률적 상한이 아니다. 감사자가 제시한 실제 bound
    (`sup ≤ in-system + Poisson(λ·dt)`, 자유모수 0)는 셀별로
    25/41/24/27을 내며 Zamba2 r3(41)는 q=1e-6에서 이미 문턱 36을
    넘는다 — 잘못된 bound였다면 이 경계 사례를 놓쳤을 것이다.
    실무 규칙: "bound"라 부르는 양이 확률모델의 꼬리 분위수인지
    관측값들의 최대 차이일 뿐인지 구별하라. 상세 `CONSENSUS.md`
    §1-1(2026-08-11 E1 addendum 블록)·§3 항목45.
32. ★★**(2026-08-11, 트래픽·roofline 진단, result-analyst 자수) 다른
    캠페인·다른 스케일에서 수입한 보조 수치는 기준(basis)이 같은지
    검증하라 — 항목20("식별자 수입 ≠ 거동 수입")의 숫자 층 변형.**
    `FINDINGS_8B_2026-07-28.md` §2.1의 "1차 weight-traffic 추정(M
    5.40/T 6.17/H 7.66 GB, H/M=1.42)"은 감사를 **두 차례**(2026-07-28,
    2026-08-04) 통과했으나, 실은 **7-8B 캠페인 문서에 3B급 모델
    (state-spaces/mamba2-2.7b, Qwen2.5-3B, Zamba2-2.7B)의 수치가
    섞여 들어와 있었고**, 게다가 M·T는 체크포인트 바이트인데 H만
    호출-인지(재독출 곱한) 트래픽이라 **기준까지 혼합**돼 1.42×라는
    비가 만들어졌다(같은 3B 격자·같은 기준이면 0.985). 체크포인트
    파일 크기·safetensors 텐서 합을 실측 대조해 3자리까지 정확히
    일치시켜 출처를 확정했다. 실무 규칙: 보조 수치를 논증에 수입할
    때는 (i) 어느 캠페인·어느 모델 스케일의 것인지 (ii) 무슨 기준
    (체크포인트 바이트/호출-인지 트래픽/다른 단위)으로 낸 것인지를
    수치 옆에 명시하고, 같은 문서 안의 다른 수치와 기준이 다르면
    비율을 만들기 전에 통일하라. 상세 `CONSENSUS.md` §3 항목46,
    `workspace/engine-port/results/s8_scaleup/
    TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md` §7.
    ★**새 사례(2026-08-14, doc-steward, E-3 realized SM count 프로브의
    부수 발견) — 이번엔 숫자 층이 아니라 하드웨어 식별 층이고, #32를
    만든 바로 그 문서가 같은 종류의 오류를 두 번 냈다.** 위 §2.1
    수치 혼입을 잡아낸 바로 그 `TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md`
    가 `nvidia-smi -q`로 하드웨어를 식별한 노드(로그인 노드 glogin01,
    `A100 80GB PCIe`)가 실제 측정이 실행된 노드(컴퓨트 노드 gpu36/
    gpu38/gpu40, SXM4)와 **달랐다** — `s8_scaleup/` job 아티팩트
    전체에 `"A100 80GB PCIe"` 문자열이 **0건**(job 882374로 컴퓨트
    노드가 `NVIDIA A100-SXM4-80GB`임을 직접 확인, `scontrol show
    node`로 gpu36·gpu40·gpu43 동일 feature 확인). 정정: 사양 BW
    1935→2039 GB/s, achieved_BW 비율 48–61%→45.4–57.7%, ridge
    161→153 FLOP/byte — 어떤 판정도 뒤집히지 않으나(오히려 강화)
    출처 층이 하나 더 있음을 보여준다. 실무 규칙 추가: **하드웨어
    사양을 인용할 때는 측정이 실제로 실행된 노드에서 읽어 job
    아티팩트에 기록하라** — 로그인 노드와 컴퓨트 노드가 다른
    SKU일 수 있다(이 클러스터가 실제로 그렇다: glogin01=PCIe,
    gpu36–43=SXM4). 상세 `CONSENSUS.md` §3 항목46(追記),
    `TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md` §11(addendum).
    ★**새 사례 2(2026-08-14, E-1a, B-4) — 이번엔 성공 사례**: C2
    caveat이 인용해 온 "rep 간 sd 0.01–0.07"은 basis 미기재 수입값
    이었다. 2026-08-14 E-1a가 원자료(`bsweep_regime/
    e1a_preanalysis_2026-08-14.json`)에서 직접 측정해 `sd_rep(ε,
    44→92)` = Ha8 0.0102·T8 0.0083(ITL 수준 rep CV 0.2–0.7%)로
    대체했다 — 44→92 쌍·Ha8/T8 2 arm·이 격자 한정. 실무 규칙 강화:
    basis 미상 수입값을 발견하면 금지만 하지 말고 가능하면 그
    자리에서 원자료로 직접 재측정해 대체하라(이번엔 GPU 비용 0).
    상세 `CONSENSUS.md` §3 항목46(追記2), "8B decode-SM 민감도 측정
    노트" rep 분산 실측값 단락.
33. ★★★**(2026-08-14, doc-steward, 코드 직접 확인 — 새 측정 아님)
    "매니페스트 N/N sha 일치"는 런타임이 바이트 동일함을 함의하지
    않는다 — 매니페스트가 실제로 덮는 범위를 먼저 확인하라.**
    `sync_engine_tree.sh`는 정확히 **15개 파일만** sha256으로
    해시한다(`:99-115`). 그 밖에서 런타임 거동에 관여하는 것으로
    확인된 것: (i) `sglang/srt/multiplex/pdmux_context.py` —
    `(74,34)` 등 파티션 기대값을 만드는 `divide_sm()`이 여기
    있고 `multiplexing_mixin.py:28`가 이 파일을 import하지만, sync는
    이 파일을 **설치도 해시도 하지 않는다**(dev 트리 mtime
    2026-04-06, sync가 만지는 형제 파일들의 2026-08-11과 불일치;
    저장소 `src/multiplex/`에 이 파일 자체가 없다). (ii)
    `sgl_kernel/spatial.py`(green-context 생성 원시함수) — dev
    트리(`sglang_engine_dev`)에는 아예 없고 venv 사이트패키지
    (`sglang_engine_venv/lib/python3.14/site-packages/sgl_kernel/`)
    에만 있다 — sync의 관할 밖. (iii) `src/patches/`의 패치 5개 중
    sync가 참조하는 것은 3개뿐(`pdmux_thread_local_role.patch`
    `:39`·`holb_probe_scheduler_hooks.patch`:`64`·
    `mamba2_pure_ssm_arch.patch`:`89`); `nemotron_h_forward_split_
    prefill.patch`·`triton_backend_mambaish_vheaddim.patch`는
    저장소 전체 검색으로도 **어느 실행 경로에서도 적용되지
    않는다**(`env/dev_tree_edits.md` 항목6:35·항목7:43 자신의
    서술도 이 두 patch를 "Full method" 참고용으로만 가리킬 뿐 sync
    대상으로 적지 않는다 — self-consistent). (iv)
    `env/dev_tree_edits.md` 항목 3·4·5·7(`hf_transformers_utils.py`
    레지스트리·`configs/__init__.py`·`model_runner.py`·
    `triton_backend.py`의 Zamba2/v_head_dim 관련 수동 편집)은
    sync가 재적용도 해시도 하지 않는 **수동 편집**이고, 항목 6·8·9
    (`models/{nemotron_h,falcon_h1,granitemoehybrid}.py`)는 sync
    스크립트 자신의 주석(`:73-75` "still manual copies")이 명시적
    으로 자백하며 해시 목록에도 없다.
    ⇒ **재현성 주장의 범위를 좁힌다**: "매니페스트 N/N sha 일치"가
    보증하는 것은 그 매니페스트가 나열한 **정확히 그 파일들**의
    바이트 동일성뿐이다 — 캠페인 간 "동등한 코드에서 실행됐다"는
    주장(Gate 1/`gate1b`/`gate1c`/Gate 2-S 트립와이어가 근거로
    쓰는 "13/13"·"15/15 sha 바이트 동일")은 이 15파일 범위로
    한정해서 읽는다. **과잉 강등 금지 — 두 가지 구별 필수**:
    (a) 이것은 **기존 성능 판정을 뒤집지 않는다**. 지금까지의
    캠페인 사이에 실제로 커버 밖 드리프트가 있었다는 증거는
    없다(그런 주장도 하지 않는다) — 좁아지는 것은 "매니페스트가
    무엇을 증명하는가"라는 **주장의 범위**뿐이다. (b) "패치 2개가
    sync 경로에서 미적용"은 "그 기능이 런타임에 없다"를 함의하지
    않는다 — `nemotron_h.py`/`falcon_h1.py`처럼 해당 변경이 수동
    편집으로 이미 트리에 반영돼 있을 수 있다(사실 `dev_tree_edits.md`
    항목6·7이 정확히 그렇다고 기록한다). 확인된 것은 오직 "sync
    스크립트가 그것을 보장하지 않는다"까지다. 상세 `CONSENSUS.md`
    §3 항목48, 원자료 `workspace/engine-port/scripts/bootstrap/
    sync_engine_tree.sh`(`:39-116`)·`workspace/engine-port/env/
    dev_tree_edits.md`(항목 3–9)·`handoff-report/
    session_handoff_2026-08-13.md` §4 "정본 등재 후보 2건".
34. ★★★**(2026-08-14, doc-steward, 4건 사전등록 감사 결과 종합 —
    새 실험 아님) 사전등록은 규칙과 하네스를 한 번에 감사받으면
    안 된다 — 규칙 먼저 감사받고, 통과한 규칙에 대고 지은 하네스를
    다시 감사받아라.** 같은 날 사전등록 4건(LTSM P1·Gate 2-S
    E1-b/c·`%smid` R0·E-1)이 감사받아 **3건 NO-GO, 1건
    CONDITIONAL-GO**가 났고, 차단 결함이 예외 없이 **스코어러·하네스
    구현 층**이었으며 전부 **GPU 0으로 발견 가능**했다. 셋(LTSM
    P1·E1-b/c·E-1)은 **코드 자신이 잘못된 라벨을 산출**했다(사람이
    아니라 도구가 거짓 음성/양성). ★**개인 부주의가 아니라는 증거**:
    E-1은 **메인 세션 자신이 작성**했고 다른 세 건과 **같은 종류의
    결함**(하네스 배선 부재·basis 미검증)을 냈다 — 특정 작성자의
    습관이 아니라 **구조적 실패 모드**다. 실무 규칙: 사전등록을
    두 단계로 감사한다 — **(i) 결정 규칙만 먼저 감사**(입력·임계값·
    판정 로직이 사전에 명시한 질문에 답하는가), **(ii) 통과한 규칙에
    맞춰 지은 하네스/스코어러를 별도로 다시 감사**(GPU 0 스모크로
    각 분기가 실제로 발화하는지 확인). 규칙과 하네스를 한 번에
    제출하면 감사가 규칙 층에 소진돼 구현 층 결함이 통과한다. 상세
    `CONSENSUS.md` §3 항목49, `PROJECT_STATUS.md` "다음 실험 gate"
    #11(4건 레지스트리), 메모리 `deconfound-measurement-lessons.md`
    항목34.
35. ★**(2026-08-15, doc-steward, E-1 rev1–rev3+kernel_mech 4회 설계
    전부 감사 차단에서 도출) 개정판에서 손잡이 값을 유지한 채 유도
    서사만 바꾸지 마라.** rev2가 등가-분기 마진 `δ=0.0610/3=0.0203`을
    썼고, 감사가 그 유도를 반증하자 rev3는 유도를 "오차예산"(측정
    잔차+L 부작용 채널 상한 합)으로 완전히 갈아 끼웠다 — 그런데 결과
    숫자는 `δ=0.020`(0.0203의 반올림)으로 **그대로**였다
    (`workspace/engine-port/results/bsweep_regime/PREREG_E1_REV2_
    2026-08-15.md:202`·`PREREG_E1_REV3_2026-08-15.md:194`). 유도가
    바뀌었는데 숫자가 안 바뀌는 것 자체는 우연일 수 있으나, 이 경우
    그 숫자(0.0610)의 **출처**가 바로 §3 항목50이 지금 오염원(job축
    교락 = keepalive 사망)으로 특정한 그 `S(16)−S(9)` 앵커였다 —
    유도를 바꾸는 작업이 앵커 자체의 타당성 재검증으로 이어지지
    않았다. 실무 규칙: 개정에서 유도/서사를 바꿀 때는 결과 숫자가
    안 바뀌었다면 **그것이 우연인지, 숫자가 바뀐 유도로도 재검증되지
    않은 채 이월된 것인지**를 명시적으로 구분해 기록한다. 상세
    `CONSENSUS.md` §3 항목53, `handoff-report/session_handoff_
    2026-08-15.md` §2.6·§4.
36. ★**(2026-08-15, doc-steward, kernel_mech 설계 감사 차단에서
    도출) 타당성은 도구 문서·메트릭 DB로 확인한 뒤 설계하라 — 확인
    비용이 로그인 노드 명령 1줄일 때도 생략될 수 있다.**
    `kernel_mech` §3은 "green context 하에서 wave 지표 분모가 전체
    108 SM으로 고정돼 오염된다"는 함정을 **가장 중요한 함정**이라
    부르며 이를 피하려고 수제 카운터 층으로 내려갔고, 거기서
    `wave_eff≡1` 항등식을 만들었다(방법론 게이트 #9 여덟 번째 재발).
    그런데 `ncu --query-metrics-collection launch --chip ga100`을
    실행해 `launch__waves_per_multiprocessor`의 설명문을 읽으면
    "When using green contexts, this metric is scaled with the
    number of SMs used by the green context"라고 **명시**돼 있다 —
    그 오염은 이 툴체인(ncu 2025.3.1.0)에 **존재하지 않는다**. 설계
    §0은 "도구 층 확인(GPU 0)" 표까지 갖췄으나 **버전 문자열만**
    확인하고 메트릭 설명문은 읽지 않았다. 실무 규칙: 새 계측 설계가
    "도구의 알려진 한계를 피한다"고 주장할 때는 그 한계의 근거를
    **그 도구 자신의 문서/메트릭 DB에서 직접 인용**해 §0에 병기한다
    (버전 번호 확인만으로는 부족하다) — 확인 비용이 GPU 0·명령
    1줄이라는 사실이 생략을 정당화하지 않는다. 상세 `CONSENSUS.md`
    §3 항목52·53, `handoff-report/session_handoff_2026-08-15.md`
    §2.9·§4.
37. ★**(2026-08-16, doc-steward 등재 — 메인 세션 진단, claims-auditor
    감사 없음[provenance 명시, 아래 참조]) 적대 감사에는 합격 기준과
    단일 판정 질문을 함께 줘라 — 범위 없는 적대 검토는 항상 NO-GO를
    산출하며, 그건 설계 품질 신호가 아니다.** E-1 rev1/rev2/rev3 ·
    kernel_mech · C2-R rev1이 **5연속 감사 차단**됐다(방법론 게이트
    #34·#35·#36, 위 "다음 실험 gate" #11). C2-R rev2가 통과한 것
    (2026-08-15, claims-auditor GO, 위 "다음 실험 gate" #11)은 설계가
    나아져서만이 아니라 **감사 범위를 단일 질문("이 설계가 정지 (b)에
    답하는가")으로 한정하고, 그 외 발견을 NO-GO가 아니라 caveat로
    접수**하도록 의뢰를 바꿨기 때문이다 — 실제로 caveat **10건**이
    나왔고 그중 **5건**이 하네스를 바꿔 **차단 대신 개선으로
    흡수**됐다. ⚠️**부수 진단(반드시 병기 — 한쪽만 적으면 왜곡)**:
    rev1의 진짜 문제는 감사가 과했던 것이 **아니라** 메인 세션이 필요
    이상으로 크게 설계한 것이었다(실제 결손은 2 arm(M8·Ha8)뿐인데 4
    arm ε-순서 캠페인을 설계했다) — 감사는 그 과잉을 걷어냈고, GPU로
    재려던 것 5개(b*=16·부팅 분산·PIN 통과율·D3 계산가능성·백엔드
    바운드 부재)를 **기존 데이터로 답해줬다**. ⇒ 이 게이트는 "감사를
    약하게 하라"가 **아니라** "감사에 판정 기준을 주라"다. 기존 게이트
    #34(사전등록을 **규칙 감사 → 하네스 감사**로 순서상 2단 분리하라)
    와는 다른 축이다 — #34는 **언제** 감사하는지(순서), 이 게이트는
    한 번의 감사 요청 자체의 **스코프 경계**(질문+합격기준+caveat
    라우팅)를 다룬다. 실무 규칙: 적대 감사를 의뢰할 때는 (i) 설계가
    답해야 하는 **단일 판정 질문**을 명시하고, (ii) 그 질문의 **합격
    기준**(임계값·통과 조건)을 함께 주고, (iii) 질문 범위 밖 발견은
    **NO-GO가 아니라 caveat로 접수**하도록 요청한다. ★**provenance**:
    이 게이트는 메인 세션(doc-steward) 직접 진단이며 **claims-auditor
    감사를 거치지 않았다** — 적대 검증 전이므로 "확정 방법론"이 아니라
    "현재 작업가설"로 인용한다. 메모리 `deconfound-measurement-
    lessons.md` 항목38과 대응. 상세 `reports/CONSENSUS.md` §3 항목55,
    `handoff-report/session_handoff_2026-08-16.md` §4-1.

38. ★★**(2026-08-16, doc-steward 등재 — "상금 크기" 논증 감사,
    claims-auditor 기준3 CONFIRMED에서 도출) 달성된 동적의 열위(HE0)와
    달성 가능한 천장은 다른 명제다.** 정본이 이 둘을 혼동한 문장을
    두 곳 이상 갖고 있었다 — `reports/CONSENSUS.md` §5-5("천장이
    static 매칭으로 확정… payoff 없음")·§1-17("static과 tie가 상한,
    이길 regime 없음")이 **관측된 하한**(HE0: 이 저장소가 실제로
    시험한 single-worker·SM-split·reactive 제어가 best-static을 못
    넘는다, §1-7 5.4σ·§1-17 ≈10σ)을 **이론적 상한**처럼 서술한다 —
    정본 자신이 §5-8(a)(b)에서 dual-worker(Claim D, 서빙 0건)·
    non-SM-split lever(구현 전무)라는 미탐색 통로를 열어둔 채로다.
    실무 규칙: "달성된 X가 Y를 못 넘는다"와 "달성 가능한 X의 천장이
    Y 근처다"를 같은 문장에 섞지 말 것 — 후자를 주장하려면 그 자체의
    증거(예: §1-13 각주 E/F의 scoped 천장 진술, n=4·정본술어)를 따로
    인용하라. 두 행에 스코프 배너 부착 완료(§5-5·§1-17). 상세
    `reports/CONSENSUS.md` §3 항목57, `reports/
    PRIZE_SIZE_ARGUMENT_2026-08-16.md` §3.
39. ★★**(2026-08-16, doc-steward 등재 — "상금 크기" 논증 감사 + 코디네이터
    지적, 2회 재발) 인용금지·강등 결정은 그 수치를 만든 원 아티팩트
    문서로도 역전파하라 — 정본 본문에 다는 것만으로는 부족하다.** 같은
    실패 모드가 한 세션 안에서 2회 발생했다: (i) `CONSENSUS.md` §1-4가
    "7.24s" 인용금지를 걸었으나(2026-08-04) `reports/paper/
    CLAIM_EVIDENCE_MATRIX.md` Claim C·`reports/paper/
    venue_positioning.md`(§0.1·C3)로 **12일간** 전파되지 않았고,
    `reports/PRIZE_SIZE_ARGUMENT_2026-08-16.md` rev1이 실제로 이
    결손에 걸려 그 수치를 반례로 오용했다(rev2에서 자체 정정). (ii)
    `CONSENSUS.md` §3 항목56(A)가 percentile 부트스트랩 CI를 "정본
    본문 인용 금지, t(5)로 대체"라고 **당일** 결정했으나, 그 수치를
    처음 산출한 원 아티팩트 문서(`workspace/engine-port/results/
    s8_scaleup/C2R_RESULTS_2026-08-16.md` §0 "한 줄"·§7 D2 표)는
    여전히 percentile CI를 caveat 없이 헤드라인으로 들고 있었다(같은
    날 발견·정정). 두 사례 모두 **정본에는 이미 올바른 판정이
    있었는데, 사람이 실제로 읽고 인용할 다른 문서(소비 문서 또는 원
    아티팩트)가 그 판정을 모르는 채로 남아 있었다.** 방법론 게이트
    #21("측정 실패를 게이트 실패로 라벨링 마라", CONSENSUS §3 항목18
    대응)과는 **축이 다르다** — #21/항목18은 "진단이 반복되는데
    정본이 안 바뀐다"이고, 이 게이트는 "정본은 이미 갱신됐는데
    소비처·원자료가 그대로다"다. 실무 규칙: 인용금지·강등을 등재할
    때는 (a) 그 수치를 인용하는 모든 소비 문서(matrix·roadmap·
    positioning 등)와 (b) 그 수치를 처음 산출한 원 아티팩트 문서
    양쪽에 체크리스트로 전파를 확인하라 — 정본 문구 자체를 고치는
    것만으로는 안 끝난다. 상세 `reports/CONSENSUS.md` §3 항목58,
    `reports/PRIZE_SIZE_ARGUMENT_2026-08-16.md` §7,
    `handoff-report/session_handoff_2026-08-16.md` §4-5.
40. ★★★**(2026-08-16, doc-steward 등재 — R1·R2 재분석 정본 승격,
    result-analyst + claims-auditor 적대 감사, GPU 0 — ★본문 복구:
    이 항목은 위 헤더에 "#40 신설"로 이미 명시돼 있었으나 번호가
    붙은 본문 문단이 누락돼 있었다, 새 판정 아님) 결정량 자체가
    항등식일 수 있고, 그 항등식이 저장소에 이미 문서로 있었을 수
    있다.** R2(§1-32)의 결정량②("`SM합=(108−D_ttft)+D_itl`이 108을
    초과하는가")는 `SM합>108 ⟺ D_itl>D_ttft`인데, `sgptv` HI의
    TTFT-argmax가 이 격자의 최대 decode arm(d44)이라 `D_itl≤D_ttft`가
    **데이터와 무관하게 확률 1로 성립**한다 — 부트스트랩 "10000 draw
    중 0건이 108 초과"는 검정력 0의 재진술이며 이를 증거로 제시한
    것은 **게이트 #9의 열 번째 재발**이다. 게다가 이 항등식은
    `reports/PRIZE_SIZE_ARGUMENT_2026-08-16.md` §2.3(2)에 **이미
    문자 그대로 적혀 있었는데** 결정량 설계 단계에서 적용되지
    않았다(**게이트 #18 사례**). 실무 규칙: 결정량을 설계하기 전에
    그 결정량이 데이터 분포와 무관하게 참/거짓이 되는 극단 사례를
    먼저 대수적으로 점검하고, 저장소 안에 이미 같은 부등식이
    문서화돼 있는지 grep하라. 상세 `reports/CONSENSUS.md` §3
    항목59, `workspace/engine-port/results/slo_sched/
    ORACLE_REANALYSIS_2026-08-16.md` §3-4.
41. ★★**(2026-08-16, gate #16 사전등록 재감사, claims-auditor 신규
    결함 N7) telemetry는 공짜 관찰자가 아니다 — 계측 유무가 바뀐
    비교를 드리프트/앵커로 쓰지 마라.** `multiplexing_mixin.py:
    470-473`을 통과하면 `dual_worker.py:566-602 observe_scheduler`가
    **모든 sync마다** 실행된다(쓰기만 1/32 서브샘플) — 비용은
    `O(waiting_queue + batch)` 파이썬 작업이고, 스케줄러 스레드가
    임계경로인 `--disable-overlap-schedule`에서는 이 오버헤드가
    그대로 지연에 얹힌다. ★**2026-07-15/18 sgptv 격자(HE0/HE2 다수
    헤드라인의 근거, §1-13·§1-19·§1-20·§1-32)는 telemetry 없이
    돌았다**(`sharegpt_vary_bench.sbatch`·`he2_bench.sbatch`에
    `PDMUX_TELEMETRY_PATH` 미설정) — telemetry를 켠 캠페인(G16 신규
    하네스, R2 dual-worker 평가 등)과의 **절대값 직접 비교는 빌드
    드리프트 + 계측 오버헤드의 합**이라 분리 불가하다. G16 **내부**
    비교(전 arm 동일 계측)는 이 결함의 영향 밖이나, G16과 07-15/18의
    절대값 비교는 금지. ⚠️**기존 결과 스코프 영향 가능(정정 아님,
    doc-steward 목록만) — "긴장 A(HE2 vs C2)"가 정확히 이 패턴이다**:
    HE2 쪽 하네스(`he2_bench.sbatch`·`sharegpt_vary_bench.sbatch`)는
    telemetry OFF, C2/S2/sticky/Gate 1·2-S 쪽 하네스(`s8_scaleup/*`·
    `s2_sticky/*`·`p1_gates/*`)는 telemetry ON이며, 이미 기록된
    "캠페인 계통 오프셋 −5.4%"(§3 항목31, S2/α 분석)의 미통제 인자
    목록(노드·바이너리·날짜·워크로드)에 **telemetry 유무가 없다** —
    다음 세션 result-analyst/claims-auditor가 판단할 대상이며 이번
    세션엔 정정하지 않는다. 분리하려면 telemetry-OFF 대조 부팅
    n≥2(≈0.3 GPU-hr)가 필요. 상세 `PREREG_G16_RULES_REV3_2026-08-16.md`
    addendum B-2(N7), `reports/CONSENSUS.md` §3 항목61,
    `handoff-report/session_handoff_2026-08-16.md` §15.5.

42. ★★**(2026-08-16, 세션4, doc-steward 등재 — G16 스모크 884292 하네스
    감사에서 발견, 방법론 게이트 #21의 거울상) 게이트가 자기 실패를
    성공으로 라벨링할 수 있다.** 교훈 항목21("측정 실패를 게이트
    실패로 라벨링 마라")은 코드가 **성공한 측정을 실패로** 오라벨한
    사례들이었다 — 이번은 그 **거울상**: G16 스모크 체커
    `9_H2_COUNT_AND_RESIDENCY`가 `controller_summary.json`을 못
    열면 `except Exception: print("{}")`로 삼키고 **PASS를 찍었고**,
    `G16_SMOKE_OVERALL` 논리곱이 **item 9를 아예 참조하지 않았다** —
    ⇒ 884292가 `residency_fraction={}`(측정이 사실상 전무한 상태)를
    출력하면서 `do submit`을 말할 수 있었다. 실무 규칙: 스모크/게이트
    체커를 작성할 때는 (i) 예외를 삼키는 모든 `except` 블록이 FAIL로
    귀결하는지, (ii) 전체 판정(`OVERALL` 등)의 논리곱/논리합이 **정의된
    모든 item을 실제로 참조하는지**를 별도로 assert하라 — 체커
    함수가 존재한다는 사실만으로 그 결과가 전체 판정에 반영된다고
    가정하지 마라. 검출 방법: 신규 `g16_smoke9_check.py`(8케이스)가
    **실제 884292 아티팩트에 새 게이트를 걸어 PASS→FAIL로 뒤집는
    것**(case H)을 양성대조로 사용해 수정을 검증했다. 상세
    `reports/CONSENSUS.md` §3 항목62, `handoff-report/session_handoff_
    2026-08-16.md` §4-2.

43. ★★**(2026-08-16, 세션4, doc-steward 등재 — G16 사전등록 분석기
    `g16_analyze.py` 감사에서 발견, claims-auditor F1) 분석기가
    사전등록 기호를 조용히 재정의하면 자기 대조를 깨고 중심 산출물을
    침묵시킨다.** `g16_analyze.py` 최초 구현이 §4가 bare argmin으로
    **정의**한 `D_itl`을 §6의 K1 식별 게이트를 통과할 때만 값을
    돌려주는 것으로 **조용히 재정의**했다 — §6은 식별을 판정의
    *조건*으로만 얹을 뿐 §4 기호를 재정의하지 않았는데도. 부작용은
    두 가지였다: (a) 자기 양성대조 **PC-C가 깨짐**(§7 원문은 "1차
    추정량으로도 `D_ttft=d44·D_itl=d44`"라고 스코프를 추정량에
    명시했는데, 재정의판 분석기는 미식별 시 `Δ=None`을 돌려줘 이
    표적을 통과 못 함). (b) 미식별 시 `forced_cell='undetermined'`가
    돼 **§3 강제표(설계의 중심 산출물)가 가장 개연적인 결과에서
    자동 침묵**했다. 반영(F1, 커밋 `8311d6c`) = bare argmin 복원 +
    K1을 `delta_citable`/verdict의 **인용 게이트**로 분리(기호와
    인용 자격을 분리) → PC-C 5/5 통과·§3 강제표 복원.
    ★**메인 세션 오진 1건 기록(재발 방지용)**: 이 결함을 처음
    발견했을 때 *"PC-C 표적이 §6 게이트를 통과 못 하므로 **사전등록
    텍스트 결함**"*이라 보고했는데 **절반 틀렸다** — 사전등록 §4·§7은
    문제 없었고, 결함은 **분석기가 §4 기호를 게이트-조건부로 재정의한
    것**이었다. 근거로 든 문장도 estimand를 혼동했다("ORACLE §3-4
    재현"이 실은 문턱술어 도너 대 위치통계량 도너의 우연한 근접값
    비교였다 — 게이트 #9/§3 항목39가 지목한 바로 그 추론 패턴).
    실무 규칙: 분석기 감사에서 "결과가 사전등록 문구와 다르다"를
    발견하면 **먼저 분석기가 그 문구의 기호를 실제로 그대로 구현했는지
    확인**하고 나서 사전등록 결함으로 보고하라 — 순서를 바꾸면 오진이
    반복된다. 상세 `reports/CONSENSUS.md` §3 항목63,
    `handoff-report/session_handoff_2026-08-16.md` §4-3.

44. ★**(2026-08-16, 세션4, doc-steward 등재 — G16 사전등록 분석기 PC-B
    감사, 방법론 게이트 #9 계열) 양성대조의 빈 서명 구멍.**
    `g16_analyze.py`의 최초 PC-B(양성대조) 구현이 실제로 **빈 서명
    집합으로도 통과**할 수 있었다 — 즉 대조가 estimand를 아예 실행
    하지 않아도 "통과"를 찍을 수 있는 구조였다(방법론 게이트 #9
    아홉 번째 재발, C2-R 양성대조가 실제로 다른 코드 경로를 타면서도
    표적값을 재현했던 사례와 같은 계열의 취약점). 반영: (a) 서명을
    **런타임에 실제 인자·분기로부터 기록**하도록 바꾸고, (b)
    **"빈 서명 집합은 통과 불가"** 규칙을 추가하고, (c)
    **`PC-B-neg`**(kwargs 하나만 바꿔 비교가 실제로 차이를 잡아내는지
    확인하는 음성 방향 대조)를 신설했다. 실무 규칙: 양성대조를
    설계할 때는 "표적값 일치"만이 아니라 (i) 서명이 비어 있지 않은지,
    (ii) 인자를 바꾸면 대조가 실제로 반응하는지(음성 대조)를 함께
    assert하라 — 이 둘이 없으면 양성대조가 항상 통과하는 항등식으로
    퇴화할 수 있다. 상세 `reports/CONSENSUS.md` §3 항목64,
    `handoff-report/session_handoff_2026-08-16.md` §4-3.

45. ★★★**(2026-08-17, G16 결과 문서 §2.4 감사, 게이트 #40/#9 열두
    번째 재발) 감사 목적으로 쓴 검사 자신이 항등식일 수 있다 —
    비식별성의 "증거"가 비식별성의 "정의"였다.** G16의 `gap_upper`
    (HI/LO ITL 도너 미식별의 근거 통계량)가 순수 노출차만으로
    설명되는지 확인하려던 검사는, 상태-혼합 모형
    `M_itl(a) = w(a)·S(a) + (1−w(a))·U(a)`에서 arm당 관측치 1개(주변
    평균)·미지수 2개(`S(a)`·`U(a)`)를 가정 2개로 강제한 뒤
    `Δ = gap_upper / (w(d74)−w(d64))`를 푸는 것이었다 — **식 1개·
    자유모수 1개**라 노출차가 0이 아니기만 하면 **관측값과 무관하게
    항상 유일해를 갖는다**(반증 가능성 0). 초판 결과 문서가 이것을
    "노출차만으로 gap이 정확히 재현된다"는 **발견**처럼 적었으나,
    감사가 "이 검사는 비식별성의 *정의*를 *증거*로 제시한 것"이라
    정정했다. 이전 열한 차례의 게이트 #9 재발은 전부 **생산자 또는
    검증자**가 만든 항등식이었으나, 이번은 **감사·진단 목적으로
    설계된 검사 자체**가 항등식이었던 첫 사례다. 실무 규칙: 새
    진단·감사 검사를 설계할 때도 "이 검사가 데이터와 무관하게 항상
    같은 결론을 내는 자유도가 있는가"를 먼저 대수적으로 점검하라 —
    감사자의 역할이 항등식 면역을 주지 않는다. 상세
    `reports/CONSENSUS.md` §3 항목65,
    `workspace/engine-port/results/slo_sched/G16_RESULTS_2026-08-17.md`
    §2.4.

46. ★★**(2026-08-17, G16 결과 문서 §1.3 감사) 사다리 해상도가
    사전등록된 payoff 구간을 은폐할 수 있다.** G16 사전등록(K8)은
    `Δ_SLO` 사다리를 45–80ms, 1ms 스텝으로 등록했다. 초판 결과
    문서는 이 사다리만으로 "60ms에서 tax 없음"을 헤드라인으로
    냈으나, 분석기 자신이 함께 산출한 `s_itl_exact_bands`는 **폭
    0.092ms짜리 밴드**에서 `Δ_SLO`의 부호가 `TAX_POSITIVE`로
    뒤집힘을 보였다 — 1ms 격자는 이 밴드를 볼 수 없다(분석기가
    "cite the exact bands" 경고를 찍었는데도 초판이 사다리만
    인용했다). 하필 이 밴드가 사전등록이 "이 캠페인의 SLO-관련
    payoff는 전적으로 이 구간에 있다"고 예언한 바로 그 구간이었다 —
    가장 중요한 구간에서 가장 거친 격자를 쓴 것이다. 실무 규칙:
    SLO/문턱 스윕을 고정 스텝 사다리로 등록하더라도, 결정량이
    스텝보다 좁은 폭에서 부호를 바꿀 수 있는 계단함수(예: min-연산
    기반 정수값)라면 exact breakpoint를 별도 산출·병기하도록
    사전등록 자체에 강제하라 — 사다리는 사람이 읽기 위한 요약일
    뿐 판정의 정본이 아니다. 상세 `reports/CONSENSUS.md` §3 항목66,
    `workspace/engine-port/results/slo_sched/G16_RESULTS_2026-08-17.md`
    §1.3.

47. ★★★**(2026-08-18, doc-steward 등재 — 메인 세션 자기정정, GPU 0·새
    성능 판정 아님) 인용금지는 정본 등재만으로 전파되지 않는다.**
    `CONSENSUS.md` §33 승격금지(vii)가 "split 상태 decode batch
    절대값 33.04/20.39/19.49(재현 경로 없음)"의 인용을 금지한 [CS-OK]
    **바로 다음 날**, 메인 세션이 새 분석 문서에서 이 값을 다시
    인용했다(방법론 게이트 #41의 열두 번째 재발 — 인용금지·강등
    결정은 그 값을 만든 원 아티팩트뿐 아니라 **소비하는 모든 새
    문서**에도 역전파돼야 하는데, 사람이 매번 CONSENSUS 전문을
    기억해 대조하는 방식은 실패율이 실증됐다). 실무 대응: 사람의
    기억에 의존하지 않는 **기계적 차단 도구**
    `workspace/engine-port/scripts/discipline/check_citation_stops.py`
    + 레지스트리 `citation_stops.tsv` 신설. 설계 이력: 1차 설계(저장소
    전체 스캔)는 40파일 30건 오탐(사전등록·승격금지 목록 자체가 금지
    문구를 정당하게 열거해야 하므로)으로 **폐기**됐다 — 실패 모드에
    맞춰 재설계: **커밋에 새로 추가되는 줄(diff `+`)만** 검사,
    `[CS-OK]` 마커로 의식적 예외 인정, `audit_*/`·`scripts/discipline/`
    경로 면제, 레지스트리가 비면 통과가 아니라 **에러로 죽는다**(빈
    서명 구멍 방지, 교훈 항목47과 같은 계열). 양방향 검증: 실제
    위반(33.04) 재현 시 exit 1, 이 감사 산출물 자체(5,061줄) 통과, [CS-OK]
    숫자 우연 충돌(오탐) 1건 발견해 레지스트리 주석으로 등재. ★**한계
    병기**: 추가 줄만 보므로 **트리에 이미 있는 위반은 못 잡는다** —
    이 도구는 재발 방지이지 소급 감사가 아니다. 실무 규칙: 인용금지를
    등재할 때 "정본에 적었다"를 "전파됐다"와 동일시하지 마라 — 등재
    당일 이후의 **모든 신규 문서 작성 행위**가 그 등재를 어길 수
    있는 별도의 실패 지점이다. 상세
    `workspace/engine-port/scripts/discipline/citation_stops.tsv`,
    `reports/CONSENSUS.md` §3 항목67.

48. ★★**(2026-08-18, doc-steward 등재 — 메인 세션 자기정정, GPU 0·새
    성능 판정 아님) 결과 디렉터리 이름을 캠페인 이름으로 오인하지
    마라.** `results/slo_sched/`는 "G16 캠페인 디렉터리"가 아니라
    **12개 캠페인의 447개 bench 파일이 섞인 공유 디렉터리**다(prefix로
    분해하면 `iact` 77 · `g16` **68** · `lffShort`/`lffLong` 46 ·
    `he2A`/`he2B` 42 · `sgptvLo`/`sgptvHi` 33 · `sgpt`/`jl` 22 ·
    `mixA`/`mixB` 6, 나머지는 스크립트·감사 아티팩트). 이 디렉터리를
    "G16 자체"로 오인한 것이 2026-08-18에 **"G16만 밴드질량
    이상치(anomalous)"라는 사실 오류**를 만들었다 — 전 파일을 `g16_`
    prefix로만 필터해 전수 집계한 결과 G16 HI 밴드질량은 **67.47**이고,
    형제 서브캠페인 3개(θ=60 부근 밴드질량을 계산 가능한 것들)가
    **같거나 위**였다(비교 대상 아니라는 뜻이 아니라 "G16이 유독
    이상하다"는 원 주장이 성립하지 않는다는 뜻). 실무 규칙: 캠페인
    단위로 통계를 집계하기 전에 **디렉터리가 실제로 그 캠페인
    전용인지 `ls`로 먼저 확인**하고, 공유 디렉터리라면 파일명
    prefix(또는 매니페스트)로 서브캠페인을 분해한 뒤 집계하라 —
    "디렉터리 = 캠페인"은 검증 없이 가정하면 안 되는 명제다. 상세
    `handoff-report/session_handoff_2026-08-18.md`, `reports/
    CONSENSUS.md` §3 항목68.

49. ★★★**(2026-08-18, doc-steward 등재 — 메인 세션 자기정정, GPU 0·새
    성능 판정 아님, 방법론 게이트 #18 최강 사례) 정본이 이미
    `ILL-POSED`로 판정한 측정점을 재분석에 쓰지 마라.** 2026-08-18에
    메인 세션은 `g2_0_full` phase A(rA5 운영점)의 평균에서 "d34
    골짜기"(d44 대비 +33.59pp, `d44-d34`)를 발견했다고 보고했으나,
    **같은 디렉터리 계열의 `g2_0_hard/
    hardened_disjoint_verdict_2026-07-25.md`가 바로 그 rA5 운영점을
    이미 `ILL-POSED`로 판정**(TTFT 3s-cliff bimodality, d44/d54
    견고성 순위가 sweep 간 완전 반전 확인)해 두었고, `g2_0_full/
    disjoint_verdict_2026-07-24.md` 자신도 같은 d34 rep1/rep4의
    t50 3063/3385ms를 "cliff" 사례로 **이름까지 붙여** 이미 기록해
    두었다 — "골짜기"로 재해석된 것이 실은 그 문서가 이미 "절벽"이라
    이름 붙인 바로 그 관측치였다. 재현 캠페인 `g2_0_hard`(독립
    재현)에서는 이 차이의 **부호가 반전**되고(원 문서
    페어드 CI [−22.6,+89.8], 0을 포함), 페어드 t-검정은 애초에 유의하지
    않았다. **이전 11회 재발(게이트 #9)이 대체로 "결정량 자체가
    항등식"이었던 것과 달리, 이번은 "판정할 자유가 애초에 없는
    측정점(정본이 이미 ill-posed로 닫은 지점)을 판정 대상으로 다시
    연 것"**이라 게이트 #18("저장소가 같은 진단을 두 번 냈는데 정본이
    안 바뀌면 도구 규율 실패")의 최강 사례로 등재한다 — 이번엔 정본이
    "안 바뀐" 게 아니라 **판정서를 읽지 않고 지나쳤다.** 실무 규칙:
    임의의 원자료 디렉터리를 재분석 대상으로 열기 전에, 그 디렉터리
    안(또는 그 캠페인을 다루는 CONSENSUS/PROJECT_STATUS 절)에 이미
    `*_verdict_*.md`류 판정서가 있는지 먼저 찾아 읽어라 — 원자료
    파일이 남아 있다는 사실은 그 데이터가 "판정 미정"이라는 뜻이
    아니다. 상세 `workspace/engine-port/results/g2_0_hard/
    hardened_disjoint_verdict_2026-07-25.md`,
    `workspace/engine-port/results/g2_0_full/
    disjoint_verdict_2026-07-24.md`, `reports/CONSENSUS.md` §3
    항목69.

50. ★★★**(2026-08-19, gate #13 rev3 재감사, claims-auditor, GPU 0·새
    성능 판정 아님, 게이트 #9의 열세 번째 재발) 자기 수리를 검증하는
    검사 자신이 반증 불가능한 항등식일 수 있다.**
    `DESIGN_G13_JOB_BATCH_REV3_2026-08-19.md` 초판은 감사가 지목한 死因
    3건(F2 가드 문턱 오류·Ha8 σ_boot의 부팅 1개 의존·`power2()`의
    exact-F 하드와이어)을 규칙층에서 수리한 뒤, 그 수리를 "검증"한다는
    명목으로 검사 3건(구 PC8·구 S3·구 S4)을 대조/자기검사 집합에 넣고
    `all_pass=true`로 계수했다. 재감사가 확인한 바, 셋 다 **데이터와
    무관하게 항상 참인 항등식**이었다 — PC8(`F₀.₀₅<1`)은
    `F₀.₀₅(d1,d2)=1/F₀.₉₅(d2,d1)`이고 `F₀.₉₅>1`이 항상 성립하므로
    어떤 격자에서도 위반 불가, 구 S3는 `_ub_exact_f`의 대수 구조상
    `UB≤0 ⟺ msb≤fq_lo·msw`가 구성상 참(게다가 `_oc`를 호출하지 않고
    가드 식을 복사 재구현해 실제 가드 줄의 오타를 못 잡는 설계였다),
    구 S4는 `msb>msw`면 `v>0`이라 검사 대상 사건이 애초에 발생 불가능
    했다. 이것은 rev2가 이미 겪은 실수(F-역함수 항등식을 `identity_
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
    경고 배너), `reports/CONSENSUS.md` §3 항목70.
