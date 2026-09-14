# R2 experiment roadmap

최종 갱신: 2026-09-14(3)(doc-steward — **job 908534(D-none
대조) 결과 감사 등재 + λ0 rev5 규칙층 감사 등재**: "수리가
907959 OOM의 원인" = `PLAUSIBLE(조건부)` → **`CONFIRMED(scoped)`
승급**(결과 감사 §3 (A)(B) 두 문장만 인용 가능, "지배 원인"
거부) — 그러나 **P2는 여전히 시작할 수 없다**: 승급으로 안
닫히는 것 11항에 **Claim D 선결 0건 폐쇄·P2 블로커 3개
불변**(λ0·W4 λ\* 실측 부재·게이트#6)이 포함된다.
λ0 rev5 = 규칙층 감사 **`GO-with-caveats`**(死因 0, 신규 발견
λ5C-1: shape B의 F5 앵커 자격 술어가 이 구성에서 구조적으로
도달 불가) — **실행 승인이며 게이트#6을 닫은 것이 아니다**,
GPU 신규 지출 0·**아직 미제출**(engine-porter 코드 권고 5건
반영 대기). GPU 장부 갱신(트랙 누적 1.107500→**1.322778
GPU-h**, 등록 상한 대비 +9.28%) + 4번째 미등록 축 갱신
(gpu38→43→40→41, 게이트#233 여전히 열림) + 신규 게이트 7건
(#241–247). 새 성능 판정 0건·Claim D/E 등급 불변·HE0·
layer-type 死·정책 순위·stake #1 전부 불변. `CONSENSUS.md`
rev76→**rev77**. 상세 "P2" 절(아래)·`CLAIM_EVIDENCE_MATRIX.md`
Claim D 행(2026-09-14(3) 갱신)·`PROJECT_STATUS.md` 최상단
배너(2026-09-14(4)).
이전: 2026-09-14(2)(doc-steward — **job 908179 결과 등재**:
`VERDICT PASS`(등록 튜플 최초 실행, 0.165556 GPU-h) — "수리가
OOM을 고쳤다"는 `PLAUSIBLE(조건부)`(`CONFIRMED` 아님, 결정적
`none`-guard 대조 arm 부재). **Claim D 선결 #5는 새로 닫힌
게 아니라 (Zamba2,triton) 한정 술어가 두 번째 (모델,백엔드,
ctx) 쌍으로 스코프만 확장**됐다 — Claim D 등급 불변(미검증),
**`PASS`는 P2 착수를 승인하지 않는다**(NP-8, 남은 블로커 =
λ0 `NO-GO`·λ\* 미측정·게이트 #6). GPU 장부 드리프트 정정
(`sacct` 기준 통일, 트랙 누적 1.107500 GPU-h) + 4번째 미등록
축(노드/물리GPU) 발견 + 신규 게이트 6건(#232–237). 새 성능
판정 0건·Claim D/E 등급 불변·HE0·정책 순위·stake #1 전부
불변. `CONSENSUS.md` rev74→**rev75**. 상세 "P2" 절(아래)·
`CLAIM_EVIDENCE_MATRIX.md` Claim D 행(2026-09-14(2) 갱신)·
`PROJECT_STATUS.md` 최상단 배너(2026-09-14(2)).
이전: 2026-09-14(doc-steward — **2026-09-13 저녁~2026-09-14
새벽 세션 정본 반영 1회 패스**: "P2" 절의 W4 열린 항목을
**사용자 결정(phase별 독립 λ\*)으로 종결**(구현 `lambda_star.py`,
미커밋; W5/W6는 단일 shape 아닌 phase-교대 스트림이라 별도
규칙층 감사 필요) + P2가 여전히 GPU correctness 선결(job 907959
`FAIL`=true-dual OOM)에 막혀 있음을 갱신 + 엔진 grad-guard
결함 발견+수리(커밋 `a9cd8dd`) + job 908020(등록 밖)·job
908179(작성 중 RUNNING→종료 시 `COMPLETED` exit 0, verdict
미독·결과 미등재) 등재. ★이 갱신은 직전
2026-09-13(3)(λ0 캠페인 0단계 `NO-GO`, "P2" 절 열린 항목
최초 등재)의 상단 배너 미갱신도 함께 catch-up한다 — 그
사이 본문 내용은 이미 반영돼 있었다. 새 성능 판정 0건·
Claim D/E 등급 불변(둘 다 미검증)·HE0·정책 순위·stake #1
전부 불변. `CONSENSUS.md` rev73→**rev74**. GPU: 이 트랙
신규 지출 0(모두 이미 지출된 R2 correctness 트랙 결과의
재등재). 상세 `CLAIM_EVIDENCE_MATRIX.md` Claim D 행
(2026-09-14 갱신)·`PROJECT_STATUS.md` 최상단 배너(2026-09-14).
이전: 2026-09-13(2)(doc-steward — **세 번째 사용자 결정**
(attention 백엔드 triton→flashinfer 전환) + engine-porter
Nano-9B-v2-Base 모델 지원 검증 **`GO`**(job 905835 재인용, GPU 0
이 세션) + λ* 기존 측정 확인(job 905835 `c_capacity.sbatch`,
arm별 5× 상이·단일 스칼라 설계 문제 신규 등재) + provenance
manifest 확장 사실(커밋 `87213a9`, 이미 커밋) + 신규 게이트 1건
(#195). X1의 발견·907100·907456 결론은 **(Zamba2-2.7B, triton)
한정 동결**, 새 (Nano-9B-v2-Base, flashinfer) 쌍은 correctness
게이트를 새로 쌓아야 한다(스코프 튜플에 `attention_backend=
flashinfer`·`model=nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base`·
`context_length=16384` 필요). "공통 방법" λ* 절과 "P2" 절
(아래)을 갱신했다. 새 성능 판정 0건·Claim D/E 등급 불변(둘 다
미검증)·HE0·정책 순위·stake #1 전부 불변. `CONSENSUS.md`
rev71→**rev72**. GPU: 이 트랙 신규 지출 0(job 905835 1.11
GPU-h는 `longctx_conflict` 트랙 기존 지출). 상세
`CLAIM_EVIDENCE_MATRIX.md` Claim D 행(2026-09-13(2) 갱신)·
`PROJECT_STATUS.md` 최상단 배너(2026-09-13(2)).
이전: 2026-09-12(3)(doc-steward — **X1 결과 등재**: job
907456(0.154 GPU-h) 완주, claims-auditor 결과 감사
**`CONFIRMED(scoped)`**[F1–F4 전부 충족·PASS 및 교차-잡 8/96
불일치 독립 재현·귀속 성립]. 메인 세션 해석 문장 1건
**`REFUTED`**(O06이 D44 probe prefill 민감도를 보정한다는
문장 — 뒤집힌 두 단위 모두 비분할 idx 5 decode, D44에서 돈
교란된 decode는 0/32). "24/24 교란"은 단위 지시함수(실제
스텝 커버리지 82.5%). C 층 "불일치 8→6 감소"는 Σ(등가류−1)
10→10 불변이라 성립하지 않는다. 새 게이트 5건(#172–176,
#176=긍정 사례). **X1은 어떤 Claim D 선결도 닫지 않는다.** "P1/
P2" 절(아래)의 X1 항목을 결과로 갱신하고 §12의 다음 실험
1–5를 열린 항목(전부 미승인)으로 등재했다. 새 성능 판정 0건·
Claim D 등급 불변(미검증)·HE0·정책 순위·stake #1 전부 불변.
`CONSENSUS.md` rev67→**rev68**. GPU: 이 트랙 0.28→**0.43
GPU-h**. 전문·line-by-line 근거는 `CLAIM_EVIDENCE_MATRIX.md`
Claim D 행(2026-09-12(3) 갱신)·"주장 제한" 참조.
이전: 2026-09-12(2)(doc-steward — **A) Claim E 선결
코드↔로드맵 불일치 3건 해소**[커밋 `3f188bb`, engine-porter, GPU 0,
사용자 결정: "3건 모두 로드맵이 정본"] + **B) X1 사전등록 규칙층만
등재**[커밋 `b3ef62d`·`19b9d8d` — **X1 job 실행 중, 결과 미등재**]).
"Controller defaults" 절(아래)의 2026-09-12(1차) 追記가 지적한 3건
불일치는 전부 코드로 해소됐다(추가로 새로운 긴장 2건이 등재됨,
사용자 결정 대기). "P1/P2" 절의 X1 설명은 사전등록 rev1(SUPERSEDED)
→ rev2(claims-auditor `GO-with-caveats`)로 갱신했다 — X1은 아직
실행되지 않았다(job 실행 중, 결과는 별도 등재). 전문·line-by-line
근거는 `CLAIM_EVIDENCE_MATRIX.md` "주장 제한" Claim E 항목·Claim D
행(2026-09-12(2) 갱신) 참조. Claim D/E 등급 무변경(둘 다 미검증)·새
성능 판정 0건·arm 순위 0건·HE0 불변. `CONSENSUS.md` rev66→**rev67**
(§3 항목190·191 신설[게이트#170·#171, X1 감사 기원] + 항목177
追記[게이트#157 완료] — A는 코드 사실 등재이며 새 분석 결론이
아니므로 직전 세션 선례대로 CONSENSUS에 반영하지 않되, B[X1 규칙층
감사의 신규 게이트 2건]만 반영).
이전: 2026-09-12(doc-steward — **engine-porter 코드 사실 3건
등재**, 2026-09-11 확인·file:line 근거, **성능 판정 아님·수정 없음·
사용자 결정 대기**) — "Controller defaults" 절(아래)의 평가 주기·
admission 명세가 실제 컨트롤러/프로파일 코드와 3곳에서 불일치함을
확인해 그 절 바로 아래에 追記했다(Claim E 착수 전 해소 필요). 전문·
line-by-line 근거는 `CLAIM_EVIDENCE_MATRIX.md` "주장 제한" Claim E
항목(2026-09-12 추가) 참조. Claim E 등급 무변경(미검증)·새 성능
판정 0건·arm 순위 0건·HE0 불변. `CONSENSUS.md` rev 변경 없음(rev66
유지, doc-steward 판단 — 코드 사실 등재이며 새 분석 결론 아님).
이전: 2026-09-11(3)(doc-steward — **R2 true-dual GPU
correctness 트랙** — 2026-09-11 admission latch stale-True
버그 수정(`02918e8`, 보류 해제·사용자 R2 복귀 결정) 이후 재실행
job 907032(0.12 GPU-h)는 split-prefill ownership race로
**FAIL**, 경합 수정 `874873b`+판정 규칙 사전 고정 하네스 v2
`e16e93f` 이후 job 907100(≈0.16 GPU-h, 커밋 `ea99191`)은
**PASS**. claims-auditor 결과 감사(`workspace/engine-port/
results/r2_correctness/audit_r2corr_2026-09-11/VERDICT.md`)가
S16 순차+O8 단일-probe 중첩 프로토콜 한정으로 `CONFIRMED(scoped)`
— 동시 부하(C 층) 포함 무스코프 "같은 토큰"은 관측으로
`REFUTED`(3/32 arm-분리 불일치). **Claim D 등급 무변경(미검증).**
닫힌 선결 #5(cudagraph-ON, scoped)·부분 해소 #2·#1(latch)은
코드 수정뿐. 인용 금지 R2C-1…16+필수 병기 P-1…7(VERDICT §5
문자 승계, 인용 금지 총계 **116건**). 신규 방법론 게이트
3건(#167–169)+追記 3건(G-3→#50, G-5→#1·#114, G-6→#95). 후속
실험 X1–X3·하네스 결함 H1–H3은 **미실행·미승인**(아래 "P1/P2"
절). 새 성능 판정 0건·arm 순위 0건·정책 순위 변경 0건·HE0
불변. GPU 이번 0.28 GPU-h(이 트랙 최초 지출). 상세
`PROJECT_STATUS.md` "다음 실험 gate" 항목1–4 追記, `CONSENSUS.md`
§5-8(a) 追記(rev66)·§3 항목187–189.
이전: 2026-09-11(2)(doc-steward — **Q-B 사전등록
`PREREG_QB_LOOPGAIN_2026-09-11.md`가 규칙층+결과 감사
`GO-with-caveats`(死因 0·반전 0/23표면)를 받고 **Q-B′**로
개명.** 원 Q-B(프로브 C 교락 14.953×/23.713×)는 Little
항등식(도착 비×체류 비)으로 소진, 공통-λ 설계가 도착 채널을
끈다. Stage 0(GPU 0, 기존 Q-A+P7 원자료, 신규 측정량 `E`
하나)은 "탐색적 계측 기록(사후 정의)"으로만 등재, Stage 1
(≈2.9 GPU-h)은 실행 허가·비권고로 **미실행**. 인용 금지
QB-1…20(총 100건=Q-A 80+QB 20) + 필수 병기 ⓐ–ⓕ. 신규 게이트
4건(#163–166)+追記 3건(#159·#160·#83). 아래 "longctx_conflict
트랙 로드맵" §2(Q-B′) 갱신. 새 성능 판정 0건·arm 순위 0건·정책
순위 변경 0건·HE0 불변·stake #1 불변. GPU 0, 트랙 누적 **15.42
GPU-h 불변**. 상세 `PROJECT_STATUS.md` longctx_conflict 행,
`CONSENSUS.md` §5-6 追記(rev65)·§3 항목182–186.
이전: 2026-09-11(doc-steward — **Q-A `REGRET` 캠페인(jobs
906504–906507)이 4/4 `COMPLETED 0:0`로 완주(10.641 GPU-h,
cudagraph-ON 192/192) + claims-auditor read-only 결과 감사
(`audit_qa_result_2026-09-11/VERDICT.md`)로 raw 독립 재구현
전 항목 4자리 일치 확인.** 등록 허가 `sign_agree=4`는 **0칸
발화**(rev6이 제출 전 예보한 결과, ⛔"효과 없음" 금지) — 지배
원인은 네 추정량의 실제 부호 불일치(무순서 36칸 중 17칸).
결함 9건 중 D1·D2·D3 정정+D9 라벨 강등을 조건으로 **정본 등재
= 조건부 가(可, "계측 기록"으로만)**, 조건 전부 충족·이미
커밋(`bc0033c`). ⛔인용 금지 12건 신규(총 80건) — 특히 N-4
(σ_req-free 결과로 rev8을 "틀렸다"고 쓰기 금지, 게이트#32
양방향)·N-5(같은 λ의 arm은 처치 강도 미매칭). d92 `μ_p` 0.1885
(n=1)→0.1834±0.0002(n=4, 격자·ρ는 0.1885 불변). 아래
"longctx_conflict 트랙 로드맵" §1(Q-A) 追記. 새 성능 판정
0건·arm 순위 0건·정책 순위 변경 0건·HE0 불변. GPU 장부:
트랙 누적 4.78→**15.42 GPU-h**. 신규 게이트 4건(#158–161).
상세 `PROJECT_STATUS.md` longctx_conflict 행, `CONSENSUS.md`
§5-6 追記(rev64)·§3 항목178–181.
이전: 2026-09-10(doc-steward, 5차 세션 — **캠페인(jobs
906504–906507)이 PENDING/RUNNING인 동안 Q-A 분석기 작성 중
result-analyst가 4추정량의 "축"(어느 배열의 범함수인가)이
rev7까지 미등록이었음을 발견 — rev8이 축 P(풀링 토큰 ITL)를
등록해 봉인(`GO-with-caveats`, 死因0·반전0/10표면), rev9가
코드-only 분석 상수 4건 등록+erratum 3건 정정.** 승계 인용
금지 총 68건(rev1–7 55+rev8 13), 신규 게이트 8건(#149–156)+
기존#130 追記+도구-버전관리 게이트#157. ⚠️캠페인 결과는 여전히
**0건** — 이 갱신은 어떤 결과·기대도 예고하지 않는다. 새 성능
판정 0건·arm 순위 0건·정책 순위 변경 0건·HE0 불변. 아래
"longctx_conflict 트랙 로드맵" §1 追記. 상세 `PROJECT_STATUS.md`
longctx_conflict 행, `CONSENSUS.md` §5-6 追記(rev63)·§3 항목
169–177.
이전: 2026-09-10(doc-steward, 4차 세션 — **Q-A(longctx_conflict
트랙 후계 질문)가 규칙층 감사 계보 rev2→rev6[`GO-with-caveats`,
死因0·반전0, 이 트랙 최초]→rev7(제출 판본)을 거쳐 캠페인 제출됨**
[`probes/qa_regret.sbatch`, jobs 906504–906507, 승인 예산 ≈10.1
GPU-h(10.4에서 Stage A 폐지로 감액), **제출 상태 PENDING·결과
0건**]. 아래 "longctx_conflict 트랙 로드맵" §1 갱신. ⚠️결과는
여전히 0건 — "예정"에서 "제출됨·미도착"으로만 바뀐다, 결론 예고
없음. 새 성능 판정 0건·arm 순위 0건·정책 순위 변경 0건·HE0
불변. 상세 `PROJECT_STATUS.md` longctx_conflict 행, `CONSENSUS.md`
§5-6 追記(rev62)·§3 항목139–168.
이전: 2026-09-10(doc-steward — **`longctx_conflict` 트랙(P0–P6/벡터1/
벡터2/P1 트랙과 별개, `CONSENSUS.md` §5-6 소관) 상태를 처음으로 이 로드맵에
반영 — 신규 "longctx_conflict 트랙 로드맵" 절 신설(아래 "공통 방법" 앞).
stake #1("최적 static split 위치가 워크로드 모양에 따라 움직이는가")은 이
기판(A100 108-SM, `prefill SM+decode SM=108` 엔진 강제)에서 **구매 불가로
종결**(반증 아님 — 스코프 선언, `audit_p6_rules_2026-09-09/VERDICT.md`
Y1-f). 후계 질문 2건 신설: **Q-A**(고정 split의 regret 프론티어, 사전등록
초안 `PREREG_QA_REGRET_2026-09-10.md` 존재·**미회부·미제출·예산 미승인
[≈6.8 GPU-h 산정]**) · **Q-B**(SM-split 액추에이터 자기상쇄 루프 이득,
사전등록 없음, 2순위). ⚠️Q-A는 아직 결과 0건 — "예정"으로만 기록, 결론
예고 없음. 새 성능 판정 0건·arm 순위 0건·정책 순위 변경 0건.** 상세
`CONSENSUS.md` §5-6(2026-09-09 rev58/rev59), `PROJECT_STATUS.md`
longctx_conflict 관련 절.
이전: 2026-08-22(doc-steward — **`%smid` R0(job 889631,
claims-auditor `CONFIRMED(scoped)`) 결과 반영 — 아래 §"P1 트랙"
항목6(S3/`%smid`)에 갱신 문단 추가.** id 집합 층에서
`GLOBALLY_CONSISTENT_LABEL` 확인(엔진 없는 별도 프로세스 eager
census, L0 유사물). **이 항목은 닫히지 않는다** — "무분할(C)" 잔여
SM 차감·서빙/cudagraph-ON 운영점 전달은 여전히 미측정, 새 성능
판정 0건. 상세 `CONSENSUS.md` §1-1(R0 addendum)·§3 항목76·77,
`CLAIM_EVIDENCE_MATRIX.md:704`.
이전: 2026-08-16(doc-steward — **"상금 크기" 논증 감사가 찾은 인용정지
전파 결손 정정, 새 성능 판정 0건.** 아래 §"8B decode-SM 민감도 측정 노트"의
국소 ε 밴드(16→24 0.77–0.88 vs 44→92 0.09–0.35)에 **인용정지 (a) 표기**
부착 — 이 밴드는 arm별 국소 탄력도의 min/max이므로 인용정지 (a)(arm별
ε·arm 간 순위/격차, `CONSENSUS.md` §3 항목54)의 숫자 인용과 동치다.
생존 범위는 격자 **내부** 정성 서술("고-SM 구간에서 국소 탄력도가 뚜렷이
낮다")뿐, arm 라벨·숫자는 인용 금지. 상세 `PRIZE_SIZE_ARGUMENT_2026-08-16.md`
§6, `CONSENSUS.md` rev33.
이전: 2026-08-14(doc-steward — **E-3 realized SM count 프로브
(GPU 비용 ≈0) 반영 — "S3/`%smid`" 로드맵 항목의 §0.1 payoff 1건이
정정 대상 없음으로 확인(추가 追記). 새 성능 판정 아님, 이 항목의
필요성·우선순위 불변(SM id 집합 disjointness는 여전히 열려있음).**
상세 `CLAIM_EVIDENCE_MATRIX.md` "P1 트랙" 절 항목7, `CONSENSUS.md`
rev26·§1-1(2026-08-14 E-3 블록).
이전: 2026-08-11(doc-steward — **E1 addendum(jobs
877756/877757 재집계, GPU 0) 완료 반영 — "P1 트랙 로드맵" 절의
G1-c 후속 E1을 완료 항목으로 이동, 후속 E1-a/E1-b(★결정적)/
E1-c(★필수)/E1-d 신설(구 E2/E3는 E1-b/불필요로 흡수). 새 성능
판정 아님, Gate 2-S 판정 자체 무변경, 크기 인용 셀은 여전히
Zamba2 r2 하나.** ★E1 채택은 승격이 아니라 **등급 하향된 조건부
채택**이다 — E1이 스스로 주장한 "밀도 페널티를 피했다"는 반증됨
(결정 관련 pop-A 관측 수 G1-b/G1-c 대비 5–6× 적음). 상세
`CLAIM_EVIDENCE_MATRIX.md` "P1 트랙" 절, `CONSENSUS.md`
§1-1(2026-08-11 E1 addendum 블록, rev23)·§3 항목44·45.
이전: 2026-08-11(doc-steward — **G1-c(job 877974) 완료 반영
— "P1 트랙 로드맵" 절의 G1-c를 완료 항목으로 이동, 후속 E1/E2/E3
신설, G1-a "착수 가능"으로 갱신. 새 성능 판정 아님, Gate 2-S 판정
자체 무변경, 크기 인용 셀은 여전히 Zamba2 r2 하나(3개로 늘지
않음).** 상세 `CLAIM_EVIDENCE_MATRIX.md` "P1 트랙" 절,
`CONSENSUS.md` §1-1(2026-08-11 G1-c 블록, rev22)·§3 항목41–43.
이전: 2026-08-11(doc-steward — **정본 동기화, rev14–rev20
(2026-08-06~2026-08-11) 반영 완료. 아래 P0–P6/벡터1/벡터2 본문
(2026-08-03 이전 작성분)은 무변경** — 그 구간은 "8B decode-SM
프론티어"(E1/sticky/C2, Claim A 소관) 또는 "SLO-aware 동적 제어"
(Claim D/E 소관) 로드맵이고, rev14–20이 다루는 것은 **다른 트랙**
(PD-mux 자체 vs fused, Gate 1/Gate 2/E-A/Gate 2-S — "P1")이라
어느 기존 절과도 겹치지 않는다. 이 문서에 P1 로드맵이 지금까지
없었으므로 **신규 "P1 트랙 로드맵" 절(벡터2 뒤)을 신설**해 다음
gate 대기열을 이관했다.** 이전: 2026-08-03(같은 세션 4차 속행 — doc-steward 기록, **상태 기록
· 성능 판정 0건 · GPU 런(S2)은 별도 제출 중·결과 없음**. 3차 속행이 연
§0의 이분법((i)/(ii))이 **유지 불가**로 판정됐다 — `split_frac≥0.90`이
D 파티션 실행 토큰을 순수하지도 완전하지도 않게 잡는다는 **세 번째 후보
(iii)**가 실측으로 문서화됨(claims-auditor 자기감사 재프레이밍 +
result-analyst 독립 재현 `S0R_REPLICATION_2026-08-03.md`: 행 1·3 재현,
행 2 순서만, 행 5(클럭) 미발화, ★행 4(음성대조)가 UNSPLIT도 같은 슬로우
모드를 가짐을 보여 강한 형태를 죽임 — 살아남는 건 농축 2.33–3.24×뿐).
남은 두 읽기는 **오프라인 분리 불가**, S2가 인과 시험. **철회 3건**
(메인 세션이 같은 날 앞서 씀): "§0 stands as written"·"aggregation-
invariant"·"11.09는 집계 단위 미기록"(**틀림** — 산출자는
`m3_conditional.report_conditional` [3] `sp_p50=11.0905`,
`m3_conditional.py:158-161,251-262,316-329`에 문서화). **재사용 계측
결함 2건**: `c2_anchor.py` 표 [5]가 M8 전체·Ha8 d16을 조용히 누락
(측정 부재 아니라 텔레메트리 앵커 부재) · mode estimator 60ms 상한이
arm-이식 불가. **게이트 정의(3.4.4 결정규칙) 변경 없음.**
`G_LEVER`/`G_FLAT`는 여전히 UNDETERMINED. 상세
`../../PROJECT_STATUS.md` "8B decode-SM 프론티어" "2026-08-03(4차)"
소절, `../CONSENSUS.md` §1-30·§3-18·§3-19,
`../../workspace/engine-port/results/s8_frontier/DESIGN.md` §4.3.15,
사전등록 3건(`PREREG_S0_AXIS_2026-08-03.md`·`PREREG_S0R_MODE_2026-08-03.md`·
`PREREG_S2_STICKY_ITL_2026-08-03.md`). 이전(같은 세션 3차 속행 —
doc-steward 기록. C2 →
`G_LEVER`/`G_FLAT` 앵커 도출 시도(`c2_anchor.py`)를 claims-auditor가
감사해 **경로 폐기**(주장 1만 CONFIRMED, 2–5 REFUTED/NOT-YET-SUPPORTED).
**`G_LEVER`/`G_FLAT`는 여전히 UNDETERMINED**, 다음 시도는 감사자 발안
(α)(β)에 대한 독립 사전등록이 선행돼야 함. ★**신규 최상위 열린 항목**:
C2(865493)와 872077(E1 격자)의 "decode 16 SM" per-token ITL이 같은 arm·
서버 플래그·매칭 batch에서 **2.6× 다름**(28.79ms vs 11.09ms) — 872077의
`decode_sms==16`이 실제 16-SM 하드웨어 실행인지 미검증(`DESIGN.md`
§4.3.11의 잔여층), 또는 C2 값이 셀 배치 성질인지 미해소. **sticky 격자
제출보다 이 모순 해소가 선행돼야 한다.** D=54 앵커 측정(jobs 872920/
872921)은 keepalive 재현성 결함으로 취소, 독립 수렴으로 C2 high-residency
=워크로드 장치 산물임을 확인. **게이트 정의(3.4.4 결정규칙) 변경 없음.**
상세 `../../PROJECT_STATUS.md` "8B decode-SM 프론티어" "2026-08-03(3차)"
소절, `../CONSENSUS.md` §1-28·§1-29,
`../../workspace/engine-port/results/s8_frontier/DESIGN.md`
§4.3.13–4.3.14. 이전(같은 날 2차 속행 — doc-steward 기록. (I)
claims-auditor가 §1-26/여기 아래 기록된 `g` 은퇴의 근거였던 `A_free`
결함을 대체하는 **조건부 per-token 추정량**(`m3_conditional.py`)으로
estimand를 이관[AUDITED, blocking-threshold 스윕만 UNAUDITED — 감사자
자기산출 자기감사]. (II) engine-porter가 `PDMUX_STICKY_PARTITION`
구현·correctness gate 통과[구현 사실, **구현 완료 ≠ 성능 주장 성립**] —
CPU 회귀 40 tests + sticky 단위 테스트 12 + GPU smoke(job 872800)
byte-identical 출력. sticky 격자 런은 **여전히 미제출**, 제출 전
`G_LEVER`/`G_FLAT` 사전등록이 미결 열린 항목. **게이트 정의(3.4.4 결정규칙)
변경 없음.** 상세 `../../PROJECT_STATUS.md` "8B decode-SM 프론티어"
"2026-08-03(2차)" 소절, `../CONSENSUS.md` §1-27,
`../../workspace/engine-port/results/s8_frontier/DESIGN.md`
§4.3.10–4.3.12. 이전(같은 날 1차 속행) — job **872077**(M3)의 NO VERDICT
사유가 "CI 폭 부족"에서 **"estimand 미식별"**로 확장됨을 P6에 기록. 기판이
prefill 비-in-flight 시 항상 무분할로 되돌아가므로 이 격자에서는
"decode가 D SM에서 돌았다"⟺"prefill이 동시 in-flight였다"가 같은 사건이라
`g=A_free(d16)/A_free(d54)`는 **이 격자 한정 은퇴**(sticky-partition
기판 수정 전 인용 금지, 블록 증설 재실행 선행 금지). 메인 세션이 세운
"decode 실현 4–19%가 `g`를 attenuate했다"는 보정 가설도 claims-auditor에
**REFUTED**. 다음 gate = `PDMUX_STICKY_PARTITION` 구현 → sticky 격자
1회(872077 대조) → 사전등록 판별 예측(T8≈1.85·Ha8≈0.92 vs Ha8≈1.6).
게이트 정의(3.4.4 결정규칙) 변경 없음. 상세 `../../PROJECT_STATUS.md`
"8B decode-SM 프론티어" "2026-08-03" 소절, `../CONSENSUS.md` §1-26. 이전
rev2: ★P6의 E1 "설계 위험 3중" 중 (b)(c)를 claims-auditor 회부 결과로 **정정**하고, E1 대신 제출된 M3 Transformer-control 대조[job 872077]를 기록. 게이트 정의 변경 없음. 이전 rev1: 진행 상태 갱신만 — P6에
"E1 상태(2026-08-02)" 추가: 전제 실험 4건 완료(2026-08-01, **claims-auditor
미통과 = 인용 금지**), **본 스윕 미제출**, 설계 위험 3중으로 E1이 사전등록
분기 "설계상 이 질문에 도달할 수 없다"로 갈 위험, 선행 사전등록
`--max-mamba-cache-size` 공통 상수 고정). 이전:
2026-07-28(★★★claims-auditor 감사 — P6의 Stage 0/L−2 게이트 판정
[non-binding]을 **철회**(C1 CONFIRMED: D108 앵커가 실은 decode 16 SM). L−2는
"실행 완료"가 아니라 "게이트 미실행"으로 정정. 대신 8B decode-SM 민감도 측정
노트[C2, scoped]와 그 프론티어 게이트 E1–E4를 추가 — 아래 P6 절 갱신). 이전:
2026-07-26(P6 long-context Stage 0/L−2 게이트 실행 완료 — non-binding,
아래 P6 절 갱신 — ★2026-07-28 철회, 위 참조). 2026-07-25(벡터2 재프레이밍 — cross-substrate serving 이식
[XS-series]을 "불필요·부적합"으로 하향 후 새 게이트 "Transformer-control on
green-context"[TC-series]로 교체, positioning 판정)

## 벡터1 (disjoint conflict-regime escape hatch) — 별도 트랙, CONFIRMED closure (scoped, 종결)

이 항목은 P0–P6 Claim D/E gate 배선 밖의 별도 트랙이었다(`reports/CONSENSUS.md`
§5-8(c) 추적, `PROJECT_STATUS.md` "벡터1" 절, `reports/paper/CLAIM_EVIDENCE_MATRIX.md`
Claim F 참조). **2026-07-25 CONFIRMED closure(scoped)로 종결** — 아래 5단계
sweep 계열의 최종 판정.

1. **g2_0_full**(2026-07-24, 1차 스윕) — razor-thin disjoint(feasible-A ∩
   feasible-B = ∅) 발견.
2. **g2_0_hard**(2026-07-24/25, hardening 재스윕) — **ILL-POSED at rA5**: 1차
   disjoint가 재현되지 않음(TTFT 3s-cliff bimodality), "disjoint 소멸" 관측은
   별도 ITL-p95 percentile-window 아티팩트로 판명.
3. **g2_0_decliff stage-1**(jobs 863880–863948) — `rA{2,3,3.5,4}×{d16,d44,d54}`
   스캔, `rA=2`만 clean off-cliff(n=6 확증). 유일한 clean 지점에서 static `d54`가
   양 phase 동시 커버(`feasible-A={d16,d44,d54} ∩ feasible-B={d54} = {d54} ≠ ∅`)
   하나 **PLAUSIBLE closure, CONFIRMED 아님**(claims-auditor 반증 3항목: off-cliff
   에서도 살아있는 split→TTFT gradient·d54 배제 onset 미측정 전이대·
   "binding-A⟺on-cliff" 미증명).
4. **g2_0_rasweep**(120 job, `rA{2.25,2.5,2.75,3.0,3.25}×{d16,d34,d44,d54}×n6`) —
   off-cliff sub-band(rate≤2.75)에서 disjoint 재확인 없음, 전이대를 rate
   3.0–3.5로 좁힘.
5. **g2_0_raconf**(claims-auditor pre-registered 24-job 확증 열, `rate{3.5,3.75}×
   {d44,d54}×n6`) — 결정 규칙(어떤 rate서든 d54 견고히 <0.7(p90>3s, unimodal) ∧
   d44/d16 동시에 견고히 ≥0.95·off-cliff(p90<2s)면 disjoint 실재→REOPEN, 아니면
   d54가 양 phase 동시 커버하는 companion collapse면 CONFIRMED)를 **companion
   collapse로 판정**: rate3.5 d44 0.953±0.035≈d54 0.948±0.035(d54 failTTFT=0/6);
   rate3.75 d44 0.932±0.042 < **d54 0.948±0.062**(d54가 더 높음). REOPEN 전제
   양쪽 붕괴 — d54는 어느 rate서도 <0.7이 아니고 d44도 어느 rate서도 견고히
   ≥0.95가 아님(웜업성·split-대칭적 TTFT-blowup). d54는 Phase B 유일 feasible
   split이면서 Phase A도 d44와 대등하게 커버 → 단일 split이 양 phase를 시간축
   에서 커버 → **disjoint 없음, 최종 확정**.

★**필수 caveat**: magnitude는 ill-posed(metric cliff, run-length 의존)이나
**순위(d54≈d44, d54 미선-배제)는 견고**. Phase-B d44 0.188은 50ms 경계 바로 위라
magnitude fragile·방향 견고. scope는 {Zamba2-2.7B, ctx4096, Phase A in2048/o32,
Phase B in2048/o512@rB4, triton attn+mamba, disable-radix-cache, cudagraph-ON,
A100 108-SM green-context pdmux, SLO=TTFT 3s ∧ per-req ITL-p95 50ms, inter-phase
drain된 순차 2-phase, rate_A≤3.75}에 한정 — **"hybrid엔 disjoint 없음"으로
일반화 금지**. closure는 얽힘 억제(drain) 조건 관측 = 필요조건 bound이지 hot
varying-trace(Claim C 얽힘) 실증 아님.

이 트랙의 결과는 Claim D/E나 §1-20(spatial coupling-tax)에 영향을 주지 않는다 —
시간적 disjoint와 공간적 coupling-tax는 별개 축. **남은 방향(후속, 미실행)**:
(i) long-context(decode floor가 ctx 상승에 따라 올라가는 영역 — CONSENSUS
§1-5 — 충돌이 발생할 수 있음, 모델/ctx 교체 필요), (ii) §1-20 spatial
decoupling(별도 device pool disaggregation, +16% headroom). 상세 verdict:
`workspace/engine-port/results/g2_0_full/disjoint_verdict_2026-07-24.md`,
`workspace/engine-port/results/g2_0_hard/hardened_disjoint_verdict_2026-07-25.md`,
`workspace/engine-port/results/g2_0_decliff/decliff_verdict_2026-07-25.md`,
`workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md`.

## 벡터2 (substrate-robustness 식별) — Transformer-control on green-context (2026-07-25)

`reports/paper/venue_positioning.md` §0.1(2026-07-25, venue-strategist
prior-art 조사)의 판정: 이 논문의 central negative는 substrate-robustness 축으로
두 갈래다 — **(A) green-context 종속**(layer-aware 死·cudagraph 비양립, Claim
B) vs **(B) mechanism-independent 후보**(lever-weakness=mamba decode
SM-둔감·entanglement·decode 비대칭, Claim A/C).

⚠️★**2026-07-25 하향·재프레이밍(사용자 지적) — cross-substrate serving 이식
(XS0/XS1/XS2)은 불필요·부적합.** 이 절의 초판은 두 번째 substrate(libsmctrl/MPS)
serving 이식을 "make-or-break 필수 게이트"로 걸었으나 **철회**한다. 이식은
불필요할 뿐 아니라 부적합하다:

- **MPS**: SM 파티션이 프로세스별·정적 → 런타임 동적 PD-mux 불가 +
  단일-프로세스 `event_loop_pdmux`의 멀티-프로세스 전면 재구조화 비용이 실익
  초과.
- **libsmctrl**: NVIDIA 비제공 리버스-엔지니어링(per-arch SM 마스킹) →
  하드웨어 세대·드라이버 귀속(driver-580 BLOCKED가 증거). 배포 근거에 비-vendor·
  비-이식 의존성을 들이는 셈.
- ★**green-context = 배포 primitive 방어**: NVIDIA 공식 fine-grained SM
  primitive는 green-context 하나(CUDA Green Contexts 12.4+)뿐. ⇒ "libsmctrl
  쓰면 되잖아"의 답 = "green-context가 배포 가능한 유일 vendor primitive다.
  DuetServe/Bullet의 동적-승은 libsmctrl(세대 귀속·비-vendor) 위에서만 성립 →
  libsmctrl에서 hybrid 동적이 이겨도 이식 불가한 research curiosity이지 배포
  가이드라인의 반례가 아니다." 따라서 (A) layer-aware 死는 green-context-bound로
  정직히 스코프하고, 그 스코프를 libsmctrl 비-이식성이 오히려 받쳐준다.

따라서 XS0/XS1/XS2(별도 substrate serving 이식)는 **실행하지 않는다.** 대신
Risk 2(모델 vs substrate 귀속)를 **기존 green-context 위에서** 닫는 값싼
식별 실험 3수로 교체한다:

- ★**신규 게이트 = Transformer-control on green-context**: 순수 Transformer
  (예: Qwen/Llama)를 기존 pdmux(green-context)에 통과시켜 hybrid와 **같은
  green-context + 같은 conjunctive-SLO**에서 대조한다. drain 비용은 두 모델에
  동일하게 작용 → **상쇄**. 이 벡터의 ID prefix `TC`(Transformer-Control)는
  P4 baseline ID `B0`–`B8`와 별개다.
  - **TC0**: 순수 Transformer 모델(Qwen/Llama류)을 기존 green-context pdmux에
    배선(모델 로딩·correctness gate).
  - **TC1**: hybrid와 동일 워크로드/SLO/split-grid에서 reactive dynamic vs
    decode-heavy static을 측정.
  - **결정 규칙(사전 등록)**: **Transformer 동적-승 ∧ hybrid 동적-패 → flip은
    substrate·메트릭 고정 하에 모델(hybrid) 귀속 확정**(Risk 2 닫힘, 진짜 식별).
    **둘 다 동적-패 → negative는 hybrid가 아니라 메트릭(conjunctive-SLO
    goodput)+배포-primitive(green-context) 탓으로 재프레이밍**(여전히 유효하나
    다른 기여).
- **보강 (실행 불필요·기존 데이터)**: (2) **lever-weakness = roofline
  microbenchmark**(r0c SM-민감도 데이터 보유) — mamba decode SM-둔감은
  연산강도 성질이라 primitive-robust; "libsmctrl이 고친다"는 반론은
  drain(=(A))에만 닿고 lever(Claim A)엔 안 닿음. (3) **헤드라인 HE0는 이미
  entanglement 귀속으로 측정 완료**(`switch_count`≈0·컨트롤러 0.014% 직접 계측
  → 동적-패가 overhead/drain 탓 아님) → drain-아티팩트 반론은 (A)에만 닿고
  헤드라인 무관.

벡터1(disjoint conflict-regime, 시간축 disjoint-feasibility)과는 무관한 별개
트랙. 벡터2는 이제 파티셔닝 primitive 불변성을 **cross-substrate 이식이 아니라
green-context 위 모델-대조(+기존 microbench/telemetry)**로 식별한다.

## P1 트랙 로드맵 (PD-mux 자체 vs fused) — P0–P6/벡터1/벡터2 밖, 별도 트랙
(2026-08-11 신설)

벡터1·벡터2와 마찬가지로 이 트랙도 아래 "단계와 stop/go gate"(P0–P6, Claim
D/E 소관)와 별개다 — 여기서 묻는 질문은 "이 엔진에서 PD-mux(공간 SM 분할)를
켜는 것 자체가 fused보다 나은가, 왜 그런가"다. 정본은 `PROJECT_STATUS.md`
"확정된 결과" 1번·"다음 실험 gate" #10, `reports/CONSENSUS.md` §1-1이며 이
절은 다음 gate 대기열만 이관한다(전문·수치는 위 두 정본, 요약은
`CLAIM_EVIDENCE_MATRIX.md` "P1 트랙" 절 참조).

### 지금까지의 순서 (완료분)

1. P1 운영점(cudagraph-ON) 대조(2026-08-05, jobs 873944/873945) — "PD 분리
   자체는 항상 이득" 철회, "PD-mux 켜면 이득(2모델·꼬리)" 확립.
2. **Gate 1**(2026-08-06, job 874478) → **G1-b**(2026-08-07, job 875293) —
   selector-level 파티션 라벨, Zamba2 rate{2,3} 조건부 해금.
3. **Gate 2 rev4**(2026-08-07, jobs 875344/875346) — chunk512 vs pdmux(부정),
   R1′/R2′(aux 플래그 배제, 2026-08-09 정본 반영 복구).
4. **E-A**(2026-08-08~09, jobs 875654/875657/875661) — mixed-chunk 레버 진단
   (고유 기여 1.2%, "fused 조율 소진" 주장 금지).
5. **Gate 2-S 설계**(2026-08-09~10, `PREREG_GATE2S_2026-08-09.md`) —
   claims-auditor 감사 4회(NO-GO×3 → GO-with-changes). 1차 실행(jobs
   877107/877109)은 하네스 결함 6건으로 primary 0개, 배관 스모크(job
   877593, 0.11 GPU-hr)가 재발 방지(`PROJECT_STATUS.md` "방법론 게이트"
   #26).
6. **★Gate 2-S 첫 유효 결과**(2026-08-11, jobs 877756/877757) — SM 분할
   성분만 격리한 첫 대조: ITL p95 개선 vs TTFT p95 악화(트레이드오프),
   크기 인용은 Zamba2 r2 1셀. **인용 시 `CLAIM_EVIDENCE_MATRIX.md` "P1
   트랙" 절의 제한 7건을 반드시 함께 적용.**
7. **★G1-c**(2026-08-11, job 877974, 0.10 GPU-hr) — Granite(873945 복제)로
   Gate 1/Gate 2-S가 Granite에 대해 남긴 "전제 미검증" 상태를 해소. Gate
   2-S §8.9 Granite r3·r4 전제 **PREMISE VERIFIED**(max(decode_bs)=10/10
   <36, frac((54,54))=0/0). **해제되는 것은 명명 층 하나뿐**(§5.6.1
   `name_for()`) — **크기 인용 자격은 불변**(`g2s_analyze.py:1157-1161`
   코드 확인: `premise`는 F-계열 gate 산출에 미입력), Granite r3·r4는
   여전히 F-계열 발화 상태라 `SIGN ONLY, MAGNITUDE NOT CITABLE` 유지 ⇒
   **크기 인용 가능 셀은 여전히 Zamba2 r2 하나뿐**(세션 초반 "1→3개로
   는다" 예상은 원자료로 반증). 상세 `CLAIM_EVIDENCE_MATRIX.md` "P1 트랙"
   절 5번·인용 제한 7건 4번(갱신).
8. **★E1 addendum**(2026-08-11, jobs 877756/877757 재집계, GPU 증분 0,
   claims-auditor 적대 감사 완료) — G1-c 후속 "E1"을 실행: Gate 2-S §8.9
   전제를 그 캠페인 자신의 셀에서 n=10 직접 산출한 pooled `max(decode_bs)`
   로 재검증(4셀 전부 `VERIFIED_AT_SAMPLED_INSTANTS`: Zamba2 r2=14/r3=23,
   Granite r3=10/r4=13). **채택 — 승격이 아니라 등급 하향된 조건부
   채택.** E1이 §2.5에서 스스로 주장한 "G1-b/G1-c보다 밀도 페널티를 덜
   받는다"는 **반증**됐다 — 결정 관련 pop-A 관측 수는 오히려 5–6× 적다
   (1,279·1,496 vs 6,402·8,920). 실질 우위는 밀도가 아니라 **반복수
   (n=1→10)와 셀 일치**뿐 — **E1은 G1-c를 대체하지 않는다**(서로 다른
   축). 적대적 bound(q=1e-9) = 25/41/24/27, **Zamba2 r3는 q=1e-6에서
   이미 37≥36으로 문턱 미배제인 유일한 셀**. 성능 결론(Δ·Holm·9-셀
   `S1-C`·크기 인용 셀 Zamba2 r2 하나) 불변. 상세 `CLAIM_EVIDENCE_
   MATRIX.md` "P1 트랙" 절 6번, `CONSENSUS.md` §1-1(2026-08-11 E1
   addendum 블록)·§3 항목44·45.

### 다음 gate (우선순위순)

1. **E1-a**(GPU 0) — §2 bound를 사전등록 후 4셀×5arm×capscan 전체
   적용, 꼬리 분위수 하나(권장 1e-6) 사전 고정. ⚠️감사자가 이미 값을
   봤으므로 제3자/잠긴 스크립트 채점 또는 저자 불리 방향 확정용으로만.
   **미실행.**
2. **E1-b**(★결정적, ≈0.3–0.5 GPU-hr) — Gate 2-S 4셀 A4를
   `FORCE_PREFILL=1`·**n≥4**로 재실행 — 밀도와 셀 일치를 동시 만족하는
   유일한 설계(구 "G1-c 후속 E2"의 상위 버전). G5 UNDETERMINED이므로
   force-mode 섭동을 α로 동반 보고. **미실행.**
3. **E1-c**(★필수, ≈0.2 GPU-hr) — 양성대조: Zamba2 A4 rate6·n≥4·
   force=1. G1-b 40(초과) vs 이 캠페인 capscan 31(미초과) = 문턱을
   가로지르는 구간. 없으면 `VERIFIED_AT_SAMPLED_INSTANTS`는 "한 번도
   발화한 적 없는 스크린의 통과"에 불과(게이트#15). **미실행.**
4. **E1-d**(E1-b에 포함 가능) — Zamba2 r3 여유 13의 정면 검정(force=1·
   n≥6·상위 꼬리 직접 추정). **미실행.** (구 "G1-c 후속 E3"는 E1 채택
   으로 불필요 확정, 관측자-효과 잔여 질문은 E1-b가 부분 흡수.)
5. **G1-a**(≈0.2 GPU-hr, engine-porter) — 관측 전용 sync 1회
   (`multiplexing_mixin.py:1005`/`:1080` 사이). 관측자 혼입 회피로
   G1-c 뒤로 순서를 미뤄뒀던 것, **2026-08-11 G1-c 완료로 착수 가능**.
   G1-d(S3 하드웨어 프로브, 아래 항목과 통합)도 같은 그룹, 미착수.
6. **S3 / `%smid` 직접 하드웨어 SM 프로브**(별건, 이 트랙 어떤 캠페인의
   선행조건도 아님, `PREREG_GATE2S_2026-08-09.md` §0.0.B) — "무분할(C)"의
   잔여 SM 차감 여부·selector-level→hardware-level 격상. ★2026-08-14
   (E-3 realized SM count 프로브, GPU 비용 ≈0)이 §0.1의 payoff 항목 1
   (`log(108/34)` 분모 정정)이 정정 대상 없음을 확인했다(34는 요청값이자
   드라이버 보고 realized 값) — **이 항목 자체의 우선순위·필요성은
   불변**(SM id 집합 disjointness는 여전히 개수 층 프로브가 못 여는
   질문). `%smid` 재설계 시 §0.1 재작성 필요. ★★**2026-08-22 갱신 —
   `%smid` R0(job 889631, claims-auditor `CONFIRMED(scoped)`)가
   실행돼 id 집합 층에서 `GLOBALLY_CONSISTENT_LABEL`을 확인했다**
   (엔진·모델·요청 없는 별도 프로세스의 eager census, "물리 SM
   인덱스" 아닌 "전역 일관 라벨"). ★**이 항목은 여전히 닫히지
   않는다** — R0는 L0 유사물이라 이 항목이 실제로 묻는 것("무분할
   (C)"의 잔여 SM 차감 여부, selector-level→hardware-level 격상,
   서빙·cudagraph-ON 운영점 전달)은 미측정. 상세 `CONSENSUS.md`
   §1-1(R0 addendum)·§3 항목76·77, `CLAIM_EVIDENCE_MATRIX.md:704`,
   `PROJECT_STATUS.md` "다음 실험 gate" #11 레지스트리 `%smid` R0 행.
7. **F-B(ii) 재설계** — Gate 2-S 인용-셀 선별 필터의 귀무 발화율이 40.1%
   (1−0.95¹⁰)임이 확인됐다(방법론 게이트 #24). 다음 사전등록에서 대체할
   것, **이번 결과에 소급 적용 금지**.
8. **Gate 3**(≈1.5 GPU-hr/모델) — NemotronH·Falcon-H1 운영점 대조. "4모델
   전부"를 다시 쓰고 싶을 때만 필요, 안 하면 §1-1은 영구히 2모델 문장.
9. **Gate 4**(sustainable-rate 직접 측정, n≥4) — r4/r6 크기·용량 주장을
   인용하고 싶을 때만. 도착창 ≫ drain-tail이 되도록 프롬프트 수를 rate에
   비례(§4.2.1 관례).
10. E-A 잔여 사다리(T3-1/T3-2/T3-3/T4-2/E-C/E-D, 우선순위·비용은
   `PROJECT_STATUS.md` "확정된 결과" 1번 E-A 블록 참조) — fused-측 조율
   가능성 진단 계열, Gate 2 본 질문에는 직접 기여하지 않음.

## longctx_conflict 트랙 로드맵 (stake #1 종결 + 후계 질문 Q-A/Q-B′)
(2026-09-10 신설)

벡터1·벡터2·P1 트랙과 마찬가지로 이 트랙도 아래 "단계와 stop/go gate"(P0–P6,
Claim D/E 소관) 밖의 별도 트랙이다. 정본은 `reports/CONSENSUS.md` §5-6(전체
경과)·`PROJECT_STATUS.md`이며, 이 절은 로드맵 상태만 이관한다(수치·死因 전문은
정본에서 확인).

### stake #1 — 이 기판에서 구매 불가로 종결(반증 아님, 스코프 선언)

*"최적 static split 위치가 워크로드 모양에 따라 움직이는가"*는 `longctx_conflict`
트랙의 규칙층 사전등록 계열(누적 17연속 `NO-GO`, `PREREG_SWEEP` rev1/rev2·
`PREREG_P6_ITL_ORDER` 포함)이 직접 검증을 시도했으나, 이 기판에서는 **구매 불가로
종결**됐다. 근거 = `prefill SM + decode SM = 108`이 엔진 강제라 `μ_p(D)`(포화
처리율)와 `itl(·,D)`(ITL 반응)를 분리할 자유도가 없고, 실측된 두 단조성이
서로 상쇄가 아니라 **보강**하므로 순서 역전이 구조적으로 관측될 수 없다
(`../../workspace/engine-port/results/longctx_conflict/audit_p6_rules_
2026-09-09/VERDICT.md` Y1-f, `../CONSENSUS.md` §5-6 (C)).

⚠️**이 판정을 인용할 때 절대 쓰면 안 되는 문장**: *"장문(long-context)에서
최적 split 위치가 안 움직인다"* — 이것은 반증이 아니라 **이 공통-부하 설계로는
답할 수 없다는 스코프 선언**이며, 저 문장으로 쓰면 즉시 방법론 게이트 위반이다.
"구매 불가"로만 서술한다. (관련 positioning 함의는
[`venue_positioning.md`](venue_positioning.md) §5 2026-09-10 추가 참조 — 이
질문의 긍정 답은 이미 최근접 선행[MuxWise]의 제품 전제로 배포돼 있다.)

### 후계 질문 (Q-A 캠페인 완주[계측 기록으로만 등재] · Q-B′ 규칙층+결과 감사 `GO-with-caveats`[탐색적 계측 기록으로만 등재, Stage 1 미실행] — 우선순위순)

1. **Q-A — 고정 split의 regret 프론티어**(1순위). 질문을 "어느 위치가
   최적인가"에서 "하나로 고정하면 얼마를 잃는가"(쌍 페어드 차 행렬 estimand,
   `min`·argmin·포락선 산출 사전등록 문자로 금지)로 바꿔 stake #1을 다시
   묻지 않고 우회한다.

   ★★**(2026-09-10, 4차 세션, doc-steward 갱신) 규칙층 감사 계보 완주 +
   캠페인 제출** — rev2(`.../longctx_conflict/audit_qa_rules_2026-09-10/
   VERDICT.md`) `NO-GO` → rev3 `NO-GO` → rev4 `NO-GO` → rev5 `NO-GO`
   (전부 반전 시험 死因 N2, 트랙 계열 누적 18→21연속[승계 장부, 게이트#110
   미검증]) → **rev6(`.../audit_qa_rev6_2026-09-10/VERDICT.md`)
   `GO-with-caveats`(死因 0·반전 0/6표면) — 이 트랙 최초의 `GO-with-caveats`,
   결정 규칙 6개를 유지한 채 반전 시험을 통과한 첫 설계**(P7식 "결정 규칙
   0개로 우회"와 구분) → rev7(`PREREG_QA_REGRET_REV7_2026-09-10.md`,
   T1–T6 문면 수리 이행, 제출 판본).

   **캠페인 제출·결과 없음**: `.../longctx_conflict/probes/qa_regret.sbatch`,
   **jobs 906504–906507**(라운드당 1개), 승인 예산 **≈10.1 GPU-h**(초안
   ≈6.8→등록 10.4, Stage A 폐지로 10.4→10.1로 감액 — `𝒜(λ,B):=𝒜(λ,A)` 확정
   이후 그 산출이 Stage B 포화 bench와 중복). **제출 상태 PENDING·완료 0/4·
   결과 0건.** ⚠️**Q-A는 여전히 결과가 전혀 없다** — 이 로드맵 항목을
   인용할 때 "제출됨·미도착" 이상의 어떤 결론도 예고하지 않는다.

   **결과 문서가 승계할 인용 금지 55건**(rev6 §6의 14건 + rev2–rev5의 41건,
   전문은 각 VERDICT 경로)의 필수 병기 3종: ① 미처치 바닥이 arm-무관
   (median-ITL 최소 13.016–13.079ms·최소 TTFT 0.883–0.903s, 기전
   `CONSENSUS §3 항목26`) ② `|𝒜(λ)|`이 처치 강도의 순감소 함수 ③ 저부하
   외삽 노출 18/19 셀(94.7%). stake #1 구조 판정은 이 캠페인의 어떤 판본
   으로도 열리지 않는다(`Ê`·argmin·순위·포락선·"어느 arm이 최적" 산출
   사전등록 문자로 금지). 상세 `CONSENSUS.md` §5-6 追記(rev62)·§3 항목
   139–168, `PROJECT_STATUS.md` longctx_conflict 행·"방법론 게이트"
   #119–148.

   ★★**(2026-09-10, 5차 세션, doc-steward 갱신) 캠페인 PENDING 중
   "실행 중" 분석 정의 등록(rev8·rev9)** — 캠페인(jobs 906504–906507)이
   여전히 PENDING/RUNNING인 동안, 분석기 작성 과정에서 **4추정량이 어느
   배열의 범함수인지가 rev7까지 미등록**이었음이 드러났다(미등록 축
   R[요청별 `itl95`]에서는 유일한 등록 허가 `sign_agree=4`가 3쌍×3 N
   전부 발화, 등록 축 P[풀링 토큰 ITL]에서는 0회). **rev8**(`audit_qa_
   rev8_2026-09-10/VERDICT.md`) **`GO-with-caveats`(死因 0·반전
   0/10표면)** — 축 P 등록이 **사후 선택이 아니라 정합성 복원**임을
   4갈래로 논증(rev7 §2·rev6 §1(d) 연역·rev6 §3 제출-전 예보값 일치·
   이해상충 방향이 허가를 죽이는 쪽·시각순서). **rev9**가 처방 이행(코드-
   only 상수 4건[`σ̂_req` 페어드 재표집·`B=10000`/seed1·모드 값 정렬·
   `UNTREATED_TOL=1.10×ITL_SOLO`] 등록·1시드 rung 대칭 진단 병기[등록식
   불변]·산출7 구현·rev8 erratum 3건 정정). ★★승계 인용 금지 총
   **68건**(rev1–7 55 + rev8 13). ⚠️**캠페인 결과는 여전히 0건** — 이
   갱신은 분석 정의 문면 결함의 정정일 뿐 어떤 결론도 예고하지 않는다.
   신규 게이트 8건(#149–156, `CONSENSUS §3` 항목169–177) + 기존#130
   追記(ε 중복) + 도구-버전관리 게이트#157. 새 성능 판정 0건·arm 순위
   0건·정책 순위 변경 0건·HE0 불변. 상세 `workspace/engine-port/results/
   longctx_conflict/{PREREG_QA_REGRET_REV8_2026-09-10.md,
   PREREG_QA_REGRET_REV9_2026-09-10.md,
   audit_qa_rev8_2026-09-10/VERDICT.md}`.
   ★★**(2026-09-11, doc-steward 갱신) 캠페인 완주 + 독립 결과
   감사 — Q-A는 이제 "제출됨·미도착"이 아니라 결과가 있다.**
   jobs 906504–906507이 **4/4 `COMPLETED 0:0`**로 완주했다
   (`RESULT_QA_REGRET_2026-09-11.md`): main 192/192·포화 20/20·
   부팅 16/16·`SELFCHECK_PASSED` 4/4·manifest 4라운드 동일·
   등록 정합 192=192·교차검증 최대 5.847e-06·벽시계 38,308s=
   **10.641 GPU-h**(예산 대비 +2.3%), 운영점 cudagraph-ON
   (192/192).

   claims-auditor가 결과 문서를 **read-only** 적대 감사
   (`audit_qa_result_2026-09-11/VERDICT.md`) — `qa_analyze.py`를
   쓰지 않고 raw `bench_*.jsonl` 192 + `sat_*.jsonl` 20에서 등록
   정의를 독립 재구현해 Δ행렬·CI·`sign_agree`·분산성분·모드·
   포화·makespan·span **전 항목 4자리 일치**(불일치 0건), 해석
   규율 준수 확인. 사실-보고 절 결함 9건 중 **D1**(§15 #1이
   σ_req-free `Δ(d44,d54)`를 "6칸"으로 적었으나 §6.3[옳음]은
   **8칸**[1·1·2·1·2·2·3·3] — 자기모순)·**D2**(§13.2 `1.958×`·
   "변동≤1.6%"가 네 독법 어디서도 재현 안 됨, 실제로는 4라운드
   중 라운드 1 단독 값)·**D3**(§15 #5가 `gap≥4ms` 항등식을
   "작동하는 참인 주장"으로 서술 — 게이트#9 18번째 재발)은
   등재 전 정정 필수, **D9**(산출 8 라벨 과잉)는 라벨 강등
   필수였고 **전부 원자료 문서에 이미 반영·커밋**(`bc0033c`).

   **정본 등재 = 조건부 가(可) — "계측 기록"으로만**(조건 전부
   충족). 등록 estimand(72칸×4추정량)는 전수 산출됐고 **등록
   허가 `sign_agree=4`는 어느 칸에서도 발화하지 않았다**(무순서
   36칸 최대 3, 단 2칸이며 대칭 진단에서 2로 하락 — rev6이
   제출 전에 적어 둔 예보와 일치). ⛔이 결과를 "효과가 없다"·
   "차이가 관측되지 않았다"로 옮겨 적는 것은 금지(rev8 6-1).
   허가 미발화의 지배 원인은 이중계상이 아니라 **네 추정량
   사이의 실제 부호 불일치**다(무순서 36칸 중 **17칸**, `σ_req`
   빼면 **22칸**). `n_boot=4`(df=3)에서 `σ_boot`는 여전히
   **미검출**(양수5·미검출59·퇴화43, 비퇴화 상한 21건 0.03–
   1.40%). d92 `μ_p` 갱신: 0.1885(n=1)→**0.1834±0.0002**(n=4,
   격자·`ρ`는 등록값 0.1885 불변). 미처치 바닥은 arm-무관:
   **13.0064–13.0158ms(0.072%)·TTFT 0.8768–0.8785s(0.20%)**
   (rev6 6-8 인용 수치는 그보다 높은 선행 자료 값). 등록 축(P)
   과 진단 축(R) 사이 144 점추정 중 49건(34.0%) 부호 반대
   (갈릴 수 있는 108건 중 45.4%).

   ★★신규 인용 금지 12건(N-1…N-12, 문자 승계, 총 **80건**
   [rev1–8 68+12]) 중 특히 **N-4**: ⛔σ_req-free 결과로 "rev8
   판정서가 틀렸다"·"서사가 바뀐다"로 쓰지 마라 — 그 투영은
   rev8이 **P7 부하·모양 A·`homog6`·N 재표집 한정**으로 스코프
   했고 게이트#32는 **양방향**이다. 쓸 수 있는 건 "이 캠페인
   으로 전이되지 않는다"까지. **N-5**: ⛔같은 `λ`의 arm들은
   처치 강도가 맞춰져 있지 않다(`ρ(d92)/ρ(D)=4.754×` 항등식) —
   `Δ` 행렬 어떤 칸도 "split만 다른 비교"가 아니다.

   stake #1 구조 판정은 이 캠페인의 어떤 판본으로도 열리지
   않는다(불변). ★★GPU 장부: 트랙 누적 **4.78 → 15.42 GPU-h**
   (이 트랙 최초의 완주 캠페인). ★★신규 게이트 4건(#158–161,
   판정서 §11 G-a/G-b/G-c + 메인 세션 발의 1건, 기존 #119–157
   과 대조해 중복 없음): #158(G-a, 폐기 절이 산출 목록에 고아로
   남을 수 있다)·#159(G-b, 요약 표가 본문과 다른 수를 실을 수
   있다)·#160(G-c, 인용 금지 문자 승계+같은 문서 재긍정, 게이트
   #9 18번째 재발)·#161(요약 통계 출처가 전체가 아니라 부분일
   수 있다). 새 성능 판정 0건·arm 순위 0건·정책 순위 변경
   0건·HE0 불변. 상세 `workspace/engine-port/results/longctx_
   conflict/{RESULT_QA_REGRET_2026-09-11.md, audit_qa_result_
   2026-09-11/VERDICT.md}`.
2. **Q-B′ — 공통 λ에서 decode 평균 ITL 차의 노출·인구 합성 표준화
   분해**(경로 cf 지정, 2순위, 원 **Q-B**[SM-split 액추에이터
   자기상쇄 루프 이득]의 개명 후계).

   ★★**(2026-09-11(2), doc-steward 갱신) 사전등록 규칙층+결과
   감사 `GO-with-caveats`**(`audit_qb_rules_2026-09-11/VERDICT.md`,
   死因 0·N1–N4 미발화·반전 0/23표면) — 등재 전 필수 수리
   D1–D9(전부 GPU 0, 문면·계산기 편집) 이행을 조건으로 "가(可)".

   **명칭 변경 Q-B → Q-B′**: 원 Q-B가 가리킨 프로브 C 교락 수치
   (out=96 14.953×=`X`비4.998×체류비2.992, out=384 23.713×=
   5.194×4.565)는 **Little 항등식**으로 raw에서 소진 확인됐고,
   핸드오프가 요구한 공통 λ 설계 자체가 이 교락의 도착 채널을
   구조적으로 끈다(`dL/dD`가 Q-A `Δ_mean`의 재척도로 붕괴, 36/36
   차이 0.0ms로 재현). "루프 이득"·"액추에이터"·"자기상쇄"·
   "되돌린다"는 인용에서 퇴역 — 정적 arm 사이에는 되먹임 루프가
   없다.

   **Stage 0**(GPU 0, 기존 Q-A 192+P7 36 bench에서 계산, 신규
   측정량은 토큰 노출 계측기 `E` 하나): 위약·정렬 검사로 타당성
   확인(위약 최대 0.039ms, 구간 조기종료 변이에서 0.704ms로
   실패 ⇒ 작동하는 검사). `E`는 `ρ`의 재진술이 아니다(ρ-대리
   R² 0.529). 1차 성분 `K@D`/`S@D′`는 39 pair-cell 중 32칸에서
   CI가 0을 배제하나, 분해는 경로 의존(`G` 중앙 cf 0.809/sf
   0.410, 외삽 없는 e-only에서도 0.815/0.516 — 상호작용
   `I`[39/39 음]에서 옴). **등재는 "탐색적 계측 기록(사후
   정의)"으로만**(게이트#150형 방어 자산 없음).

   **Stage 1(≈2.9 GPU-h)은 실행 허가되나 저자·감사 모두
   비권고**(하류 소비자 없음 — HE0·stake #1 사용 금지, 확증도
   예측 0개라 구조적으로 불가) — **미실행**.

   ★★인용 금지 QB-1…QB-20(소재 = 사전등록 §12·판정서 §5.2,
   전문 비승계) + 필수 병기 ⓐ–ⓕ(ⓐ–ⓒ는 Q-A에서 승계). **인용
   금지 총계 100건**(Q-A 계보 80+QB 20). 신규 방법론 게이트
   4건(#163–166, `CONSENSUS §3` 항목182–186) + 追記 3건(#159·
   #160·#83/교훈88). stake #1 구조 판정·HE0 불변(이 판정서도
   "어떤 게이트도 닫지 않는다"고 명시). 새 성능 판정 0건·arm
   순위 0건·정책 순위 변경 0건. GPU 지출 0(트랙 누적 **15.42
   GPU-h 불변**). 상세 `PROJECT_STATUS.md` longctx_conflict 행,
   `CONSENSUS.md` §5-6 追記(rev65)·§3 항목182–186,
   `workspace/engine-port/results/longctx_conflict/{PREREG_QB_
   LOOPGAIN_2026-09-11.md, audit_qb_rules_2026-09-11/VERDICT.md}`.

## 공통 방법

- 먼저 B1 sustainable SLO rate `lambda*`를 모델별로 측정한다.
  ★**정정(2026-09-13, doc-steward, X3 규칙층 감사 §10.3 부수
  발견)**: `benchmarks/pdmux_eval/generate_campaign.sh:10`[HIST]
  의 `sustainable_rate`가 그때는 **측정값이 아니라 기본값
  `4`**로 코드에 박혀 있었다 — 워크로드가 전부 이 값의 분수로
  정의되므로(W1 0.60·W3 0.80·W8 0.90·W9 1.10) 이 산문이 요구한
  측정이 실행되지 않으면 "near saturation"·"overload" 라벨
  자체가 틀린다(게이트 #6 "용량 먼저 측정" 위반이었다). **모델별
  λ* 측정을 R2/Claim D 캠페인의 0번째 단계로 등재**(아래 "P2"
  절·`PROJECT_STATUS.md` 최상단 배너[2026-09-13] "D" 참조).
  ★**추기(2026-09-13(2), doc-steward, engine-porter 모델 지원
  검증 부수 발견)**: λ*는 **이미 측정돼 있다**(새 도구 불필요) —
  job 905835의 `c_capacity.sbatch`가 같은 체크포인트(flashinfer,
  ctx 8192, PD-mux fixed D)에서 **D16 0.933 req/s·D44 0.675·
  D92 0.187**(전부 `KNEE_BRACKETED`)을 측정했다. 단 λ*는
  **arm별 5× 차이**가 나는데 이 스크립트는 W1–W9에 **단일
  스칼라**를 쓰고 입력 길이는 32× 차이(W3 256-in vs W2/W4/W5
  8192-in)가 나므로 "기본값 미측정"보다 **더 깊은 설계 문제**
  로 열어둔다 — "단일 스칼라 λ*를 W1–W9가 공유하는 것이 무엇을
  뜻하는지"가 캠페인 0단계의 사전등록 대상이다. 1차 캠페인
  (Claim D)에 필요한 shape는 W3(256in/512out)·W4(prefill
  8192/64+decode 64/512) 둘뿐이며, 기존 측정은 W4 prefill
  phase에 가깝고 W3와는 무관(λ*가 더 높을 것). 상세
  `PROJECT_STATUS.md` 최상단 배너(2026-09-13(2)) "C" 절.

  ★★**정정(2026-09-13(3), doc-steward, λ0 사전등록 규칙층
  감사 §S8/N3 — `.../lambda0_prereg/VERDICT_lambda0_rules_
  2026-09-13.md`)**: 위 "W4(prefill 8192/64+decode 64/512)"의
  **"decode 64/512"는 오독이다**. `workloads.py:159-160`을
  실행해 확인: W4 decode phase = **(256, 512)**이고 이는 W3
  전부와 **정확히 같다**(근사 아님) — 저장소 전체에 (in 64,
  out 512)는 존재하지 않는다. 오독 출처는 `WorkloadSpec.
  output_distribution = "64/512 by phase"`(phase별 **출력**
  길이 서술)를 입력 길이로 읽은 것. ⇒ 1차 캠페인(Claim D)에
  필요한 shape는 3개가 아니라 **2개뿐**: (256,512)[=W3=W4
  decode phase]·(8192,64)[=W4 prefill phase].

  ★★**캠페인 0단계(λ*) 사전등록 `NO-GO`(2026-09-13(3),
  claims-auditor 규칙층 감사, GPU 0, **미실행**)**: 판정서
  `.../lambda0_prereg/VERDICT_lambda0_rules_2026-09-13.md`
  (358행). 死因 **N2**(client `--seed` 미등록 — 비포화 셀의
  achieved/offered는 포화도가 아니라 도착 실현 계수 `1/Ē`;
  `bench_serving.py:1706,948`; 905835 12/12 셀 동일 방향
  −2.0σ; seed 40개 모의 실패율 15–28%; `1.96/√(N−1)≤0.05`가
  요구하는 N≥1537은 실현 불가 + shape A 사다리 4점 미등록으로
  브래킷 판정 반전 예시 3건) + **N3**(위 정정에서 확인된 정의역
  `∅`). 제출 차단 운영 결함 3건(`UNRESOLVED` 도달 불가·`shape`
  KeyError·sbatch/analyzer 부존재). ★측정 격자·승계 규칙
  (R1/R5)은 건강함을 보관 cell JSON 9개 직접 투입으로 비트
  단위 재현해 확인.

  ★★★**최우선 파생 사실 — 이 단계는 게이트 #6을 닫지 못한다**:
  905835 d44를 정본 goodput 술어(TTFT≤3000ms ∧ 요청내부
  token-ITL p95≤60ms)로 재채점하면 **0.59·λ*에서 goodput
  53.8%·0.89·λ*에서 5.8%·1.27·λ*에서 0.8%** ⇒ **λ*_SLO <
  0.59×λ*_throughput**(비 추정 2.3–3.4×). 위 "먼저 B1
  sustainable **SLO** rate를 측정한다"는 산문 정의를 이
  사전등록은 **throughput 포화로 교체**했다 — 이 교체를
  **정정으로 등재**한다: **λ*_SLO는 여전히 측정되지 않았다**,
  측정된 것은 λ*_throughput뿐이다.

  ★★★**두 λ*는 W4를 파라미터화하지 못한다 — 상호 배타다**:
  λ*=0.675 주입 → prefill phase 0.79×λ*(B) ✓ / decode phase
  **0.21–0.25×λ*(A)** ✗; λ*=2.1 주입 → decode 0.79×λ*(A) ✓ /
  prefill **2.47×λ*(B)** ✗; 캠페인 기본값 4 → prefill
  **4.7×λ*(B)**. **어떤 단일 스칼라도 두 phase를 동시에 0.80×로
  만들 수 없다.** ⇒ **★열린 항목(사용자 결정 대기)**: W4를
  아래 "P2" acceptance에서 쓰려면 `generate_campaign.sh`의
  "9 워크로드에 단일 `PDMUX_SUSTAINABLE_RATE`" 설계를
  phase별 독립 λ*로 바꿔야 한다 — 그렇게 개정할지, W4를 1차
  캠페인 범위에서 제외할지, 아니면 두 λ* 중 하나만으로
  W4 전체를 근사(그리고 그 근사를 명시적으로 최강 caveat로
  달아)할지는 **아직 결정되지 않았다**.

  권고 사다리(GPU 0 재감사 통과 시): shape A(8192-in 대응 없음,
  실은 256-in) **{1.1, 1.8, 3.0, 4.9, 8.0}**, shape B(8192-in)
  **{0.45, 0.62, 0.85, 1.15}**. ctx 16384/mem 0.82는 구속 자원
  (`max_mamba_cache_size=max_running_requests=48`, KV 6.6×
  과공급)을 바꾸지 않으므로 앵커 0.675의 provisional 강등은
  충분. 우선순위 권고: 새 모델 correctness 게이트 + `--request
  -rate inf` 2셀(≈0.3–0.4 GPU-h, N2 死因을 구조적으로 소멸시킴)
  을 λ* 사다리 재감사보다 먼저.

  인용 금지 Q1–Q5(게이트 #6 미충족·correctness 미확인·W4
  파라미터화 불가·B1≠시스템 용량·W1/5/6/7/8/9 라벨 불변) +
  필수 병기 6항, 판정서 §5·§6 문자 승계. 신규 방법론 게이트
  5건(#196–200, `CONSENSUS.md` §3 항목216–220). **overclaim
  금지**: λ*는 측정되지 않았다(NO-GO) — 905835 값은 기존
  측정(8192-in/96-out·ctx 8192·mem 0.80)이며 새 캠페인의 λ*가
  아니다. 새 성능 판정 0건·GPU 0·Claim D/E 등급 불변. 상세
  `PROJECT_STATUS.md` 최상단 배너(2026-09-13(3)), `CONSENSUS.md`
  rev73·§3 항목216–220.

  ★**반증(2026-09-14, doc-steward — engine-porter W4 구현 확인,
  커밋 `8507cee`, 등재 계기: 정본 인용 등록성 검사)**: 위
  `generate_campaign.sh:10`[HIST] 인용은 **경로도 틀렸고**
  (실제 파일은 `scripts/r2_eval/generate_campaign.sh`이며
  `benchmarks/pdmux_eval/` 아래가 아니다) **지금은 존재하지 않는
  코드도 인용한다.** `sustainable_rate` 기본값 `4`는 삭제됐다 —
  `generate_campaign.sh:30-46`이 `PDMUX_LAMBDA_STAR_TABLE`
  미설정 시 fail-closed(`exit 2`)로 대체하고, `PDMUX_SUSTAINABLE_
  RATE` 자체도 명시적으로 거부한다(단일 스칼라로는 W4의 phase별
  용량 5×차를 표현할 수 없어 per-shape 테이블로 교체,
  `benchmarks/pdmux_eval/lambda_star.py`) — 이 개정 자체는 이미
  아래 "### P2" "열린 항목 종결(2026-09-13/14)"에 등재돼 있고,
  이 문단은 **위 원문 인용의 경로·기본값 서술만** 최신 상태로
  정정한다. **달라지지 않은 것**: 모델별 λ\*가 실측됐다는 뜻은
  아니다 — 캠페인 0단계 λ0 사전등록은 여전히 `NO-GO`(게이트 #6
  잔류). 이 인용이 그동안 정정되지 않았던 이유: `check_line_
  citations.py`의 이 문서 `__scope__`가 `controller.py`/
  `profile.py`(+`workloads.py`) 패턴만 등록해 `generate_
  campaign.sh` 인용은 원래부터 검사 대상 밖이었다(신규 게이트
  #238, `PROJECT_STATUS.md` "방법론 게이트" 참조).
- configuration당 최소 5회, paired CI가 0을 교차하거나 variance가 크면 10회
  이상 수행한다.
- 모든 pair는 동일 immutable trace/hash, workload seed, server seed를 사용하고
  node 내 실행 순서를 randomize한다.
- warm-up/correctness/benchmark phase를 분리한다.
- CUDA Graph, backend, GPU clock/power, KV capacity, max-running을 고정한다.
- mean, median, SD, paired bootstrap 95% CI와 percent effect를 보고한다.
- request의 TTFT와 request 내부 token-level ITL p95가 모두 SLO 이하여야 primary
  goodput에 포함한다. 기존 request-mean-ITL score는 secondary로만 보존한다.
- 3% 미만 차이는 headline improvement로 사용하지 않는다.

## 단계와 stop/go gate

### P0 — 상태/R1 정정

`PROJECT_STATUS.md`, claim matrix, R1 재분석, stale banner를 정본으로 반영한다.

### P1 — 계측 observer effect

동일 fixed split에서 다음 네 arm을 paired AB/BA로 비교한다.

1. legacy, telemetry off
2. legacy, symmetric telemetry on
3. R1 observer state on, trace off
4. R1 observer state와 trace on

각 overhead effect가 3% 미만이고 CI가 0을 포함해야 진행한다. 실패 시 buffer,
sampling, writer를 수정하고 architecture 비교를 보류한다.

### P2 — Architecture

- legacy fixed D24/D44
- true dual fixed D24/D44

Decode-heavy/alternating에서 decode progress, ITL 또는 oldest queue age가
유의하게 개선되고 throughput regression이 3% 이하일 때만 Claim D를 채택한다.
그렇지 않으면 true dual은 negative architecture result로 남긴다.

★★★**★열린 항목(2026-09-13(3), doc-steward, λ0 캠페인 0단계
사전등록 규칙층 감사에서 도출, 사용자 결정 대기)**: 위
acceptance의 W4 워크로드는 두 phase(prefill (8192,64)·decode
(256,512))를 `generate_campaign.sh`의 **단일** `PDMUX_
SUSTAINABLE_RATE`로 스케일한다. 실측/추정 λ*(A≈2.1–2.5,
B=0.675)로 계산하면 **어떤 단일 스칼라도 두 phase를 동시에
0.80×λ*로 만들 수 없다**(λ*=0.675 → prefill 0.79×✓/decode
0.21–0.25×✗; λ*=2.1 → decode 0.79×✓/prefill 2.47×✗) — 상세
`.../lambda0_prereg/VERDICT_lambda0_rules_2026-09-13.md` §3
(1a). **P2가 W3+W4×{B1,B4}로 실행되려면 다음 중 하나를 사용자가
결정해야 한다**: (a) W4 정의를 phase별 독립 λ*로 개정, (b) W4를
1차 캠페인 범위(위 "P2")에서 제외하고 W3만 진행, (c) 두 λ* 중
하나로 W4 전체를 근사하고 그 근사를 결과 문서에 최강 caveat로
명시. **미결정 상태에서는 P2를 W4 포함으로 실행하지 않는다.**

★★★**열린 항목 종결(2026-09-13/14, 사용자 결정 — 위 (a) 채택)**:
W4 워크로드 정의를 **phase별 독립 λ\***로 개정한다. 구현물
`lambda_star.py`(신설)·`PDMUX_SUSTAINABLE_RATE` fail-closed·
campaign 스키마 v2→v3 — **미커밋**. ★신규 사실(구현 중 발견):
W5/W6는 단일 shape가 아니라 **한 Poisson 스트림에서 두 shape를
교대**하므로 per-shape rate가 존재하지 않아 이 fail-closed
설계에서는 기본 거부된다 — 도입된 혼합 정의(교대 스트림에
대한 rate 정의)는 **사전등록된 적이 없다**, W5/W6를 P2/Claim E
어느 쪽이든 돌리려면 별도 규칙층 감사가 선행돼야 한다.

★★★**P2는 여전히 GPU correctness 선결에 막혀 있다(2026-09-13/14
갱신)**: 새 (Nano-9B-v2, flashinfer) 쌍의 R2 correctness 게이트
job 907959가 **`FAIL`**(true-dual OOM, `CONFIRMED(scoped)`
귀속)했다 — 사전등록 §8이 이 FAIL로 **P2 착수를 명시적으로
차단**한다. 원인은 엔진 실물 결함(worker thread가
`inference_mode` 밖에서 autograd 켠 채 forward, NemotronH
[n_groups=8]만 노출)이었고 수리(`_activate_role_context`
grad-guard, 커밋 `a9cd8dd`)는 완료됐으나 **아직 GPU로
검증되지 않았다**. 검증 사전등록(재실행 `rerun_prereg/`)은
rev1 `NO-GO`→rev2 `GO-with-caveats`(★사후 철회)→job
908020(등록 밖 실행, Zamba2-2.7B/triton/ctx4096)→rev3
`NO-GO`→**rev4 `GO-with-caveats`**(死因 0건)를 거쳐 제출
가능 상태가 됐고, **job 908179가 그 등록 튜플(Nano-9B-v2/
flashinfer/ctx16384/D44, 수리 반영 엔진)의 최초 실행으로
보인다 — 배너 작성 시작 시 RUNNING이었고, 작성 완료 시점
`sacct` 확인 결과 **`COMPLETED`(exit 0, 4 boot 전부 완주)로
이미 종료돼 있었으나 이 갱신은 `verdict.txt`를 의도적으로
읽지 않았다** — 결과는 다음 정본 반영 패스에서 결과 감사를
거쳐 등재한다.** ★필수 병기(RA3-1 승계): 이 검증
회차는 어느 결과가 나와도 "grad-guard 수리가 OOM의 지배
원인이었다"는 인과 주장을 닫지 못한다 — PASS/FAIL 어느
쪽이든 그 자체로 P2 착수를 자동 승인/거부하지 않으며, 결과가
나오면 별도 감사를 거쳐 P2 재개 여부를 판단한다. 새 성능
판정 0건·GPU 0(이 문단은 트랙 누적 재등재)·**Claim D 등급
불변(미검증)**. 상세 `PROJECT_STATUS.md` 최상단
배너(2026-09-14), `reports/CONSENSUS.md` rev74.

★★★**job 908179 결과 등재(2026-09-14(2)) — `VERDICT PASS`이나
P2는 여전히 시작할 수 없다**: 등록 튜플의 최초 실행이
`PASS`했다(0.165556 GPU-h, C1 스코프 술어 (1)–(11) 전부 참,
비트 단위 재현). TD 2/2가 907959에서 자신을 죽였던 바로 그
14-seq/10,125-token 배치를 완주했다. ★★**"수리가 OOM을
고쳤다"는 `PLAUSIBLE(조건부)`일 뿐 `CONFIRMED`가 아니다** —
결정적 `PDMUX_WORKER_GRAD_GUARD=none` 대조 arm이 없고(무수정
하네스로는 실행 불가), F-a1은 원인 무차별이라 반증 시도 4건
실패가 확증을 만들지 않는다. ★★**Claim D 선결 #5는 이 job으로
새로 닫힌 게 아니다** — 이미 (Zamba2-2.7B, triton) 한정으로
닫혀 있던 술어("S/O층 토큰 id 동일 + cudagraph[decode 한정]
유지")가 두 번째 (Nano-9B-v2, flashinfer, ctx16384) 쌍으로
**스코프만 확장**됐을 뿐이다. 선결 #1·#3·#4a·#4b 상태 불변,
#2는 부분 해소. **Claim D 등급 = 미검증, 불변.** **`PASS`는
P2 착수를 승인하지 않는다**(NP-8) — 남은 블로커는 **λ0
사전등록 `NO-GO`(캠페인 0단계 λ\* 미측정)·W4 워크로드 정의
(위에서 phase별 독립 λ\*로 종결됐으나 λ\* 실측 자체가 아직
없음)·방법론 게이트 #6(용량 먼저 측정) 셋 다 그대로**다.
부수: 4번째 미등록 축(노드/물리 GPU, gpu38→gpu40) 발견 —
메모리 채널은 닫혔으나(양 job 5 boot `avail mem` 사다리
완전 동일) 축 존재 자체가 사전등록 §10에 없었다. ★GPU 장부
드리프트 정정: `sacct` 기준으로 통일해 R2 correctness 트랙
누적 = **1.107500 GPU-h**(등록 0.955000/등록 밖 0.152500,
이전 배너의 "0.78583" 소계는 반올림 오차 −0.00361 GPU-h를
안고 있었음 — 상세 `PROJECT_STATUS.md` "G" 절). 인용 금지
A8179-1…7. 신규 게이트 6건(#232–237). 새 성능 판정 0건·
Claim D/E 등급 불변(둘 다 미검증)·HE0·정책 순위·stake #1
전부 불변. 상세 `PROJECT_STATUS.md` 최상단 배너
(2026-09-14(2)), `reports/CONSENSUS.md` rev75.

★★★**job 908534(D-none 대조) 결과 감사 등재(2026-09-14(4)) —
"수리가 907959 OOM의 원인" `PLAUSIBLE(조건부)` →
`CONFIRMED(scoped)` 승급이나, P2는 여전히 시작할 수 없다**:
같은 job 안에서 `PDMUX_WORKER_GRAD_GUARD` 축만
`inference_mode`→`none`으로 이동시키면 907959의 OOM이 같은
39-배치 도착열·같은 rung·바이트 동일 OOM 원문으로 재현되고
(가드 없는 조건 3/3 사망, 가드 있는 조건 4/4 완주), 매칭
배치에서 peak 차 `+8,107,034,112 B`(산포 천장의 1,804배).
인용 가능 문장은 결과 감사(`.../dnone_prereg/
VERDICT_result_908534_2026-09-14.md`) §3의 (A)(B) 두 개뿐이며
**"지배 원인"이라는 표현은 승인되지 않았다**(측정된 것은
충분성+필요성뿐, 분산 분해 미측정). ★★**승급으로 닫히지
않는 것 11항 중 P2에 직결되는 것**: **Claim D 선결 0건
폐쇄**(908534는 grad-guard 귀속을 강화했을 뿐 Claim D의
S/O 프로토콜 선결과는 무관) · **P2 블로커 3개 불변**(λ0
사전등록·W4 λ\* 실측 부재·게이트 #6) · **H-Architecture/
H-Policy 무진전**(구현 결함의 수리 확인이며 dual-worker 구조
자체의 이득 증거가 아니다) · 성능 판정 0건 · 모델(Zamba2/
Falcon-H1/Granite-4)·백엔드·ctx 일반화 없음 · guard-none 축
n=1(구간추정 금지). **Claim D 등급 = 미검증, 불변.**
GPU 장부: R2 correctness 트랙 누적 = **1.322778 GPU-h**(등록
1.170278/등록 밖 0.152500, 등록 상한 대비 +9.28%). 4번째
미등록 축 갱신(gpu38→43→40→**41**) — within-job 비교엔
교락하지 않으나 **게이트 #233은 여전히 열려 있다**(cross-job
비교는 노드 교락). 신규 게이트 7건(#241–247, `CONSENSUS.md`
§3 항목261–267).

★★★**λ0 rev5 규칙층 감사 등재(2026-09-14(4)) — `GO-with-caveats`
(死因 0)이나 게이트 #6은 닫히지 않는다**: 계보 rev1–rev4
4연속 `NO-GO` → rev5 死因 0건. ★이번 회차 최대 발견
**λ5C-1**: shape B의 F5 앵커 자격 술어(`#running-req ≥ 48`)가
이 구성(`max_prefill_tokens=16384`가 8192-토큰 요청 2개를
prefill 배치 상한으로 만듦)에서 **어떤 부하로도 도달
불가**다(실현 동시성 57.43/64·median TTFT 90.69s로 backlog
확인) ⇒ (a) 셀 B의 F5 실격은 규칙의 올바른 적용이나 "셀이
포화하지 못했다"로 바꿔 쓸 수 없고, (b) `ANCHORED` 분지는 이
shape/cap의 모든 미래 instrument run에서 구조적으로 사용
불가(→ "다시 재서 앵커를 살린다"는 경로가 아니다), (c) 미달
원인은 미확정. ★★**게이트 #6은 이 결과로도 닫히지
않는다**(λ5C-8) — probe C d44 4셀 정본 술어 재채점 goodput
53.8%/5.8%/0.8%(ITL 조건은 4셀 전부 통과, 구속은 전적으로
TTFT). rev4 판정서의 `I3b=11/11/2`는 **인용 금지**(`wc -l`
체계로는 2/2/2). `FALLBACK`이 1차 등록, `ANCHORED`는
반사실. 예산 최악 코너 3.509 GPU-h(요청 3.60 ≤ 벽시계
4.50h). **GPU 신규 지출 0**(rev1–rev5 전부) — **이 사전등록은
아직 제출되지 않았다**(사용자가 0단계 제출을 승인했으나,
제출은 engine-porter의 코드 권고 5건[λ5A-2·3·4·9·11] 반영
이후로 예정). 새 성능 판정 0건·Claim D/E 등급 불변(둘 다
미검증)·HE0·layer-type 死·정책 순위·stake #1·게이트#13/#16
전부 불변. 상세 `PROJECT_STATUS.md` 최상단 배너
(2026-09-14(4)), `reports/CONSENSUS.md` rev77.

★★★**job 908623(λ0 0단계) 결과 감사 등재(2026-09-14(5)) — 11셀
완주·死因 0이나 게이트 #6은 여전히 닫히지 않는다**: gpu38,
`sacct` COMPLETED 01:23:29=5,009s, **1.391389 GPU-h**(λ0 트랙
최초 GPU 지출, 등록 보통 1.348의 1.032×). 라벨은 감사자 무수정
재실행으로 **전 아티팩트 바이트 재현**(digest 11/11 일치, 변이
54/54 blocked, 반전 시험 7표면 전부 λ\* 값 불변) — 기계적
산물 CONFIRMED. shape A(256,512) `KNEE_BRACKETED`
**λ\*(A)≈3.05 req/s**(±0.6%, n=2, @창 131.01s·N=400) · shape
B(8192,64) `KNEE_BRACKETED` **λ\*(B)≈0.696 req/s**(±0.3%, n=2,
@창 286.85s·N=200), 양쪽 `SEED_REPEAT_HOLDS` — ★라벨이 인쇄한
17자리는 인용 금지(λ0R-7).

★★**최대 신규 발견 — 실현 분할 재집계로 `runtime_snapshot`
(`multiplex/dual_worker.py:608`, 그 순간 선택된 division)을 세
독립 추정량으로 집계하면 λ\*(A)는 D44 측정이 아니다**:
λ\*(A)를 공급한 셀(a_r4)의 decode-busy 시간 **85.6–95.0%가
비분할 `(0,108)`**이고 D44 점유는 **4.5–13.5%뿐**이다(λ\*(B)
공급 셀 b_r3는 반대로 90.7–91.9%가 D44). 기전:
`multiplexing_mixin.py:1199-1200`(decode 배치가 있는데
split-prefill이 in-flight가 아니면 무조건 (0,108))
+`lambda0.sbatch:92-94`가 `PDMUX_STICKY_PARTITION`을 unset. 이는
새 기전이 아니라 `CONSENSUS.md` §1-25·§1-26(B)가 이미 등재한
`E1_DECODE_REALIZED`(4–19%)의 재현이며, 정본이 같은 이유로 `g`를
은퇴시킨 것과 **같은 구조의 교락**이다. ⇒ **NPC-I(caveat)가
shape A에서 반증으로 승격**(λ0R-8): (i) λ\*(A)를
"decode_sm=44 용량"으로 인용 금지 (ii) "split을 바꾸면 5×
변한다"(905835 계열)를 shape A에 이식 금지 (iii)
★★**λ\*(A)로 B4(true-dual)를 B1에 정규화하는 설계는
교락된다** — true-dual이 바꾸는 것이 바로 prefill·decode
동시성, 즉 실현 혼합비 자신이다.

★★**P2 블로커 갱신**: **①λ0 `NO-GO` 해소**(rev1–4 4연속
`NO-GO`→rev5 `GO-with-caveats`→job 908623 `COMPLETED`, 측정
실패 0). ②W4 λ\* 실측 부재는 **부분 이동·미해소(4겹)** — (1)
정의가 틀렸다: 측정된 것은 `throughput_saturation`, 요구되는
것은 `slo_sustainable`(하네스 자신의 `lambda_star.py`
docstring이 명기 — "does NOT close gate #6") ⇒ ②③은 같은
블로커의 두 이름 (2) Q3(상호 배타) 그대로: λ\*=0.697→decode
phase 0.23×, λ\*=3.053→prefill phase 4.38× (3) 위 실현 분할
발견이 새 장벽: 두 λ\*는 **서로 다른 실현 분할**에서 측정돼
"한 arm의 두 phase 용량"조차 성립하지 않는다 (4) NPC-I: B4
포화 상한 미측정. ③게이트#6은 **불변**(λ5C-8이 이 job 자신의
수치로 재확인 — 정본 goodput 술어 재채점 결과 ITL은 11셀 전부
통과, 구속은 전적으로 TTFT; shape B는 0.45 req/s 미만에서만
λ\*_SLO가 있고 그보다 낮은 셀이 없어 브래킷 안 됨, shape A는
인용 가능 셀 2개뿐이라 하한만 λ\*_SLO(A)≥1.754 req/s).
**실제로 움직인 만큼**: W3+W4가 요구하는 두 shape에 대해
`lambda_star.py`의 `source="measured"` 항목을 거짓말 없이
채울 수 있게 됐다(이전엔 `exit 2`). 부수: 폐기된 옛 기본값
4 req/s는 측정 대비 A 1.310×·B 5.737× 과대. **P2 착수 남는
조건 3개**: (a) λ\*_SLO 두 shape 직접 측정 (b) λ\*를 실현
분할 혼합비와 함께 등록 + `PDMUX_STICKY_PARTITION=1` 대조 (c)
B4 포함 시 B4의 λ\* 또는 정규화 논증. ★금지:
`PDMUX_ALLOW_UNMEASURED_LAMBDA_STAR=1`로 ③ 우회(허용되나
게이트 #6을 닫지 않는다).

그 밖 필수 병기: ★**λ\*(B) 0.697240 vs 인용금지 λ_inf(B)
0.695587(+0.2377%)의 "일치"는 N-8을 되살리지 않는다**(λ0R-1,
이번 회차 최대 위험) — 차이가 이 측정 자신의 seed 재현 산포
(0.310%)보다 작아 통계적으로 구별 불가, N-8은 자격 조항이라
사후 독립 측정으로 충족되지 않으며(λ5C-1 재확인, 두 번째
job에서 복제), 상한/피상한 순서도 미약히 반대로 깨지고(λ_inf는
상한 프로브였는데 측정 λ\*(B)가 초과), B 사다리 사전이 그 값을
겨냥해 설계돼 순환 위험도 있다 — 합법적으로 남는 것은 하네스
교차 일관성 서술뿐(λ0R-2). λ\*(A)는
`--max-running-requests 48`이 구속(λ0R-3, 하드웨어 상한
아님). arm 비동일성 1파일(`multiplexing_mixin.py`, 커밋
`a9cd8dd`) — 무영향은 코드 독해 추론이며 측정된 null 아님.
"제출 전 커밋" 절차가 어겨졌으나(F6 `uncommitted=31`, 커밋은
job 시작 14분 뒤) 실행 바이트 HEAD 일치로 死因 아님(λ0R-10).
`ACH_HI`를 조이는 방향이 미등록 자유 표면(λ0R-6, 최단 반전
+3.3%). GPU 장부: λ0 트랙 신규 **1.391389 GPU-h**(이 트랙
최초 지출), R2 correctness 트랙 불변 1.322778 GPU-h. 신규
게이트 6건(#248–253, `CONSENSUS.md` §3 항목268–273). **새
성능 판정 0건**(arm 비교 0·shape A vs B 비교 금지[in·out
동시 변경]·policy 레버 0·hybrid/generic·true-dual 미발화)·
Claim D/E 등급 불변(둘 다 미검증)·HE0·layer-type 死·정책
순위·stake #1·게이트#13/#16·C2 인용정지 전부 불변. 상세
`PROJECT_STATUS.md` 최상단 배너(2026-09-14(5)),
`reports/CONSENSUS.md` rev78.

★★★**다음 실험 순서 = 사용자 결정 (2026-09-15) — `b → a`**: job 908623 결과 감사 §7이 처방한
실험 중 **(b) E2를 먼저, 그 다음 (a) E1a → E1b**로 진행한다.
- **1순위 (b) E2** — `PDMUX_STICKY_PARTITION=1` **한 노브만** 바꾼 대조로 실현 분할 혼합비를
  고정해 **4-A / λ0R-8을 확정**한다(λ\*(A)를 공급한 셀의 D44 점유가 8.6–17.2%였고 지배 division이
  `(0,108)` 82.8–91.0%였다는 발견). 설계 그대로 5 boot ≈ **0.49 GPU-h**.
- **2순위 (a) E1a → E1b** — λ\*_SLO 두 shape 직접 측정으로 **게이트 #6**에 접근.
  E1a(shape B, 24 boot, ≈**1.55** GPU-h) → E1b(shape A, 28 boot, ≈**2.74** GPU-h), 합 ≈4.29 GPU-h
  (1 job에 들어가지 않아 분할된 설계).
★**이 순서는 감사자 권고와 다르다** — 결과 감사 §7은 *"우선순위는 **E1a**"*(B가 절벽이 가장 낮고
W4 prefill phase를 직접 구속)라 적었다. 사용자가 E2 우선으로 정했으므로 그 결정을 등재하고 진행한다
(근거의 합리성: 4-A는 λ\*(A)의 인용 범위와 **true-dual 정규화 설계 전체**를 교락시키므로 먼저 닫으면
E1b의 설계가 달라질 수 있다).
★★**E2 착수 전 반드시 사전등록이 문자로 고정할 설계 결정 1건**: 감사자의 E2 문안은
*"E1a/E1b의 부분집합을 반복"* 이고 판정량 (ii)가 *"같은 offered에서의 λ\*/goodput 변화"* 인데
**sticky-OFF 대조가 필요하고 E1이 아직 없다.** ⇒ **(E2-α, 권고)** E2가 **자기 sticky-OFF arm을 같은
job 안에** 들고 간다(≈**0.98 GPU-h**) — **노드/물리 GPU 축이 움직이지 않는다**(게이트 #233은 아직
열려 있고, D-none 회차에서 "같은 job 안의 대조"가 귀속을 닫은 바로 그 이유다). **(E2-β)** 기존 job
908623의 셀을 대조로 쓰면 싸지만(≈0.49) **cross-job**이라 노드 축이 교락된다(908623 = gpu38).
**선행조건(전부 미충족)**: 새 사전등록(판정량이 0단계와 다르다) · 규칙층 감사 · **새 범위 한정
OVERRIDE + 사용자 승인**(기존 2건은 job 908534·908623으로 **소진**, presubmit의 M4R·TC1 차단 2건은
여전히 살아 있다) · ★OVERRIDE 한 창 규율에 **"제출 전 커밋" 단계 추가**(λ0R-10이 그 누락을 등재했다).
**보류**: E3은 감사자가 초안을 **자기 철회**하고 수정안조차 "이 기판에서 수행 불가" 가능성을 남겼으며,
E4는 선행 correctness(R2C-2)가 미충족이라 처방되지 않았다 ⇒ **둘 다 이 순서에 넣지 않는다.**
새 성능 판정 0건 · GPU 0(이 등재 자체) · Claim D/E 등급 불변.

★追記(2026-09-11, doc-steward, R2 true-dual GPU correctness
트랙) — 위 P1/P2는 여전히 미실행 성능 게이트다. 별도로
**correctness 게이트**(P1/P2 성능 기준과 무관)가 job 907100에서
S16 순차+O8 단일-probe 중첩 프로토콜 한정으로 `PASS`했다
(claims-auditor `CONFIRMED(scoped)`, `workspace/engine-port/
results/r2_correctness/audit_r2corr_2026-09-11/VERDICT.md`).
직전 job 907032(admission latch 수정 `02918e8` 직후 재실행)는
split-prefill ownership race로 두 boot 모두 **FAIL**했고, 경합
수정 `874873b`+하네스 v2 `e16e93f` 이후 907100이 PASS했다. 새
성능 판정 0건, 위 P1/P2 acceptance는 여전히 미평가(observer
effect·decode progress/ITL·throughput regression 전부 미측정).
인용 금지 R2C-1…16·필수 병기 P-1…7(VERDICT §5, 문자 승계) —
요지는 무스코프 "같은 토큰"/"동시 부하 동치" 금지(R2C-1/2),
C 층 불일치(3/32 arm-분리, C01·C10·C19)의 원인을 "TD 결함"·
"무해 노이즈" 양쪽으로 단정 금지(R2C-3).

**후속 실험(X2·X3·하네스 결함은 여전히 미실행·미승인, 사용자 판단
대기 — X1은 아래 참조)**:
- **X1**: 민감도 양성대조 + C 층 기전 판별(0.154 GPU-h, **완료**).
  원래 계획 ~~`SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS=true`로
  4 boot(L/TD/L/TD) 재실행~~은 **SUPERSEDED**(사전등록 rev1, 死因:
  `MIN_BLOCK_KV=32` 커널 양자화 미반영). 유효 **rev2**
  (`--triton-attention-num-kv-splits 2` CLI 단일 노브 + 엔진 소스
  `38c1aca` 고정)가 job 907456(gpu42, 2026-09-12 19:11:22–19:20:36
  KST, 0.154 GPU-h, commit `19b9d8d`, manifest 17/17 907100과
  바이트 동일)으로 완주했다. ★**2026-09-12(3) 결과 갱신**:
  claims-auditor 결과 감사(`workspace/engine-port/results/
  r2_correctness/audit_x1_2026-09-12/VERDICT.md`, 357행) 총괄
  **`CONFIRMED(scoped)`** — 등록 예보 F1–F4 전부 충족, `VERDICT
  PASS`와 교차-잡 불일치 8/96(S05·O06, 4 boot 전부 동일)이 독립
  재현되며 공허성 없음, 귀속 성립(귀무대조 2건: `907032→907100`
  S 0/32 + 비등록 `907032→907456` S 2/32 전부 S05@25). 단 메인
  세션이 쓰려던 해석 문장 하나는 **`REFUTED`**됐다 — "O06이 D44를
  밟는 probe prefill의 민감도를 보정한다"는 거짓이다(뒤집힌 S05·
  O06 둘 다 **비분할 `stream_index 5`(0/108)**에서 돌았고, D44
  [idx 4]에서 돈 유일한 교란된 decode[O 층 background 행]는
  **0/32**로 뒤집히지 않았다 ⇒ **X1은 D44 운영점의 측정층 민감도를
  보정하지 않는다**). "24/24 교란"은 단위 지시함수였다 — 실제
  **decode 스텝 커버리지는 1,142/1,384 = 82.5%**이고 비복사형
  단기 6단위(S00 5/63 등)가 전부 부분 커버리지다. F3: C01
  arm-분리 소멸은 §1.7 기전과 정합이나(확증 아님) "arm-분리 = 0"의
  2/3(C10·C19)는 **L2가 옛 TD 값으로 이탈**한 결과이고 L-L 불일치는
  4→5로 증가했다 — 불일치 단위 수는 8→6이나 **Σ(등가류−1)은 10→10
  불변**(C00·C16은 3류→4류) ⇒ "C 불일치가 줄었다"는 서술 금지.
  필수 병기 X1P-1′·2′·3′·5′·6′·8·9 + P-1 추가분·인용 금지
  X1C-10…14는 `PROJECT_STATUS.md` 최상단 배너(2026-09-12(3))·
  `CLAIM_EVIDENCE_MATRIX.md` Claim D 행에 문자 그대로 등재돼
  있다. **X1은 Claim D 선결을 하나도 닫지 않는다**(#1·#2·#3·#4a·
  #4b·#5 상태 불변, #5는 재확인이나 등급 변화 없음, #2는 오히려
  더 약함). GPU: R2 correctness 트랙 0.28→**0.43 GPU-h**. 신규
  방법론 게이트 5건: **#172**(양성대조의 도달 범위는 비교 단위
  지시함수가 아니라 실제 교란 스텝/토큰 분포로 등록하라, 게이트#9
  계열 3번째 층) · **#173**(양성대조가 인증 운영점 위에 떨어졌는지
  확인하라) · **#174**(등가류 "불일치 단위 수"는 총량이 아니다 —
  Σ(등가류−1)을 병기하라) · **#175**(provenance 보강을 교란 arm
  에만 넣지 마라) · **#176**(**긍정 사례** — 교란 노브 자신에
  대해 target이 아니라 realized를 요구하라, args 덤프 +
  cudagraph capture 메모리 변화 2채널로 충족).

  **X1 이후 다음 실험(판정서 §12, 전부 미승인·사용자 판단 대기)**:
  1. **C-tier 귀무대조**(`R2C_ORDER="L L L L"`, ≈0.154 GPU-h) —
     무교란·동일 arm에서 32단위 등가류 분할·Σ(classes−1)·
     pairwise 불일치를 측정한다. F3을 해석 가능하게 만드는
     유일한 값싼 길이며 arm-분리 판정에 분모를 준다(현재 n=2/arm,
     P≈0.20).
  2. `R2C_ORDER="TD TD TD TD"` 1 job(+0.154 GPU-h) ⇒ 합쳐
     n=4/arm(프로젝트 게이트 3 충족), paired 비교. ★**철회
     (2026-09-12(5), B3 판정서 §8)**: 괄호 문장은 **거짓**이다 —
     한 job의 4 boot은 같은 노드·같은 물리 GPU·같은 warmup·같은
     `TRITON_CACHE_DIR`을 공유하는 순차 실행이라 독립 런이
     아니고, 프로젝트 게이트 3(n≥4)은 *독립 런* n≥4를 요구한다.
     인용 금지 **B3C-3**. 이 항목은 `L L L L`+`TD TD TD TD` 2
     job(0.31 GPU-h) 사전등록 "B3"로 구체화됐고 **`NO-GO`**(死因
     N2, GPU 0·미실행) — 상세 `PROJECT_STATUS.md` 최상단 배너
     (2026-09-12(5)) "B. B3" 절.
  3. **D44 resident decode 동치를 보려면 새 층 O′ + 판정 규칙 v3
     사전등록 필수**(≈0.16 GPU-h) — 엔진은 prefill이 없으면 idx
     4를 떠나므로 probe가 decode하는 동안에도 제3의 장문 prefill을
     계속 투입해야 한다. 이것은 X1 후속이 아니라 **선결 #2를
     넓히는 실험**이고 1·2를 먼저 하지 않으면 또 n=2로 끝난다.
  4. **GPU 0으로 지금 가능**: preflight에 스텝 커버리지 출력
     추가·비교기에 Σ(classes−1) 출력 추가·죽은 텔레메트리 필드
     2건(`decode_step_count`·`worker_overlap_ratio`) 처리
     (engine-porter)·사전등록 템플릿에 "교란 도달 위치=인증하려는
     운영점인가" 항목 추가.
  5. **권고하지 않음**: cap을 더 낮추거나 다른 수치 구성으로 X1
     반복(교란이 떨어지는 위치가 안 바뀌므로 같은 한계의 결과가
     또 나온다).

  ★★追記(2026-09-12(5), doc-steward — 항목1이 "B3" 사전등록으로
  구체화·제출됐고, 후속 설계 "OS"가 규칙층 감사를 받았다. 둘 다
  **미실행**, GPU 0): **B3**(`L L L L`+`TD TD TD TD` 2 job, 0.31
  GPU-h) 판정 = **`NO-GO`**(死因 N2, 반전 3건: E4를 n_22 조건부로
  읽으면 907456 n_22=0에서 라벨 반전·F4 여집합 31–59% 무라벨·빈
  `phase_c`만으로 측정 실패가 결론 발화). 신규 게이트
  G-B3-1…5(`#180–184`, `CONSENSUS.md` §3 항목200–204). 判定서
  원문 `workspace/engine-port/results/r2_correctness/b3_prereg/
  VERDICT_b3_rules_2026-09-12.md`.

  **OS**(순서 교환, `R2C_ORDER="L TD TD L"`, 0.154 GPU-h, 규칙층
  만) 판정 = **`GO-with-caveats`**(死因 0, 자유표면 20개 전수
  반전 0건) + 차단 D1–D8. 무수정 checker를 이 순서로 실제
  실행해 `VERDICT PASS` 확인(B3가 죽던 자리). 핵심 caveat:
  arm 경계 `{1,4}|{2,3}`이 동시에 "외곽-중앙" 위치 모양이라 arm과
  구별되지 않음(C24 1건, 균등 기대 1.33 미만) · 엔진 소스 핀은
  장식이 아니라 하중재(작업 트리 `dual_worker.py` +172줄 미커밋
  이었음 — A1 커밋으로 해소) · "L이 항상 먼저 조건을 제거한다"는
  B3 §7의 주장은 **철회**(L1은 여전히 위치 1). 신규 게이트
  G-OS-1…5(`#185–189`, §3 항목205–209). 판정서 원문
  `.../os_prereg/VERDICT_os_rules_2026-09-12.md`.

  **★구매 순서 권고 변경: A1 커밋 → X3 → OS.** OS는 어떤 선결도
  안 닫고(등록 스스로 선언) 재현 대상(907100)이 균등 귀무에서
  1/3 확률 사건이라, 아래 X3(선결 #4b 토큰 쪽 절반 + 성능 트랙
  전체 해제 조건)를 먼저 산다.

- **X2**: 비기본 split D16(≈0.16 GPU-h). `R2C_DSM=16`으로
  B8/O3의 FixedPolicy 변별력을 확보하고 B5/B6가 방문하는 split
  까지 스코프를 넓힌다.
- **X3**: 러너 설정 인증. ctx/dirname 버그 수정 후 러너 서버
  인자 튜플로 하네스를 1회 실행(관측 플래그 ON/OFF 대조) — 선결
  #4b(observer effect)의 토큰 쪽 절반. ★**추기(2026-09-12(5))**:
  이 항목의 "ctx/dirname 버그"가 A1 커밋(`09a8075`)으로
  수리됐다 — `r2_eval.sbatch:11`의 spool-copy 경로 결함 +
  `PDMUX_CONTEXT_LENGTH` 미배선 + `context_limit.py` 사전
  스크리닝 신설. X3 실행 자체는 여전히 미실행(GPU 0)이지만
  선행조건은 이제 충족됐다 — 위 구매 순서 권고에서 최우선.
- **하네스 결함(게이트 아님, engine-porter 이관)**: H1
  (INPUT_IDENTITY가 개수만 비교, sha 미검사, 미수정) ·
  ~~H2(`task_count`가 `set_result` 뒤 증가해 Δ+1 지연)~~ →
  **수리 완료(A2, 커밋 `ae7830e`)**: `finally`에서 future 해소
  전에 공개하도록 변경(의미 불변, 가시화 시점만 앞당김) ·
  H3(`host_worker_overlap_ratio`의 1024개 절단·수명 분모로
  동시성 지표 부적합)는 **재정의하지 않고**(기존 값 비트 동일
  고정) 올바른 창 기반 지표 5개를 새 이름으로 A2가 추가 ·
  ~~H4(`decode_step_count`가 전 boot·전 스냅샷 0, 죽은 필드)~~ →
  **A2가 방출 제거**(유일한 호출부가 R1 observer 가드라 항상 0) ·
  ~~H5(`worker_overlap_ratio`가 TD boot에서도 전부 0.0)~~ →
  **A2가 죽은 선언 제거**(살아있는 형제 `host_worker_overlap_
  ratio`만 남김). `controller.py:54-55`의 같은 결함
  (`prefill_idle_ratio`/`decode_idle_ratio`)은 정본이 "죽어
  있다는 증거"로 인용 중이라 **의도적으로 미수정**, 테스트로
  고정.

- ★**A4 부수 결과(GPU 0)**: `controller.py`·`profile.py`
  line-citation 앵커 38항목/21키가 이제
  `workspace/engine-port/scripts/discipline/line_citations.json`
  으로 버전관리 안에 있다 — 이 로드맵/`PROJECT_STATUS.md`/
  `CLAIM_EVIDENCE_MATRIX.md`의 controller.py/profile.py 인용은
  `check_line_citations.py --check`가 자동 검증한다(50→88
  compared, 0 violation).

- ★**사용자 결정 대기(2026-09-12(5))**: A1이 만든
  `results/r2_eval` 러너로 실제 캠페인을 생성하면 **405 run 중
  약 270이 현 구성으로 실행 불가**하다 — W2/W4/W5(135 run)가
  8192-토큰 프롬프트를 내보내 Zamba2-2.7B ctx 4096으로 서빙
  불가·B2/B8(90 run)이 `requires_offline_oracle`로 exit 2·
  B6(45 run)이 `policy_adapter.sh:57`
  `PDMUX_MODEL_PROFILE_PATH: unbound variable`. **Claim E
  캠페인은 모델 교체 또는 워크로드 교체 결정 없이는 돌 수
  없다** — 아래 "Controller ablation" 절과 함께 다음 세션/
  사용자 판단 대기.

  ★★**정정 + X3 규칙층 판정 + 사용자 결정 2건(2026-09-13,
  doc-steward)**: 위 "약 270"은 **중복 계수**였다. X3 사전등록
  규칙층 감사(claims-auditor, GPU 0, `.../x3_prereg/
  VERDICT_x3_rules_2026-09-13.md`, 死因 0·차단 D1–D13,
  **`GO-with-caveats`**, 미실행)가 포함-배제로 재계산: **고유
  차단 = 225**(135[W2/4/5]+90[B2/8]+45[B6] −
  30[W2/4/5×B2,B8] − 15[W2/4/5×B6]), **실행 가능 = 180**
  (W1·W3·W6·W7·W8·W9 × 5 rep × B0·B1·B3·B4·B5·B7). 사실 정정
  3건: F1이 측정 안 한 것을 측정했다고 말함(러너 코드를 한 줄도
  실행하지 않는다 — 선결 #2 "러너 설정"은 X3가 안 닫음) · F2의
  "eval 계측 구성으로 전이"가 3중 과대(관측자 2/4만 시험·엔진
  소스 핀 대 캠페인 HEAD 불일치·C층 제외) · F3 크기 오류
  (probe-boot 26/32=81% 적중, 진짜 이유는 32칸 전칭 요구).
  감사자 자기 철회(3회차 누적): 자기 OS §8 "X3는 성능 트랙
  전체를 연다"를 철회 — 블로커 3개 중 X3가 제거하는 것은
  **0개**. 신규 게이트 G-X3-1…5(`#190–194`).

  ★★**사용자 결정 2건(같은 세션, X3를 즉시 무효화)**:
  **(1) 모델 교체** — Zamba2-2.7B → `NemotronHForCausalLM`
  Nano-9B-v2-Base(ctx 131072). **X3 사전등록(rev1)은 SUPERSEDED**
  (Zamba2 기준 설계, 아티팩트는 이력 보존+무효 배너). OS
  사전등록도 같은 취급 대상(재무효화는 안 함, 아직 미구매).
  907100·907456·X1 결론은 Zamba2-2.7B 한정으로 동결, 새 모델은
  R2 correctness 게이트를 새로 쌓아야 한다. Nano-9B-v2는 서빙된
  적 없음 — engine-porter CPU-only 검증 중, GO/NO-GO 전 GPU
  계획 없음. ★교차 트랙 위험: TC1 트랙이 이미 NemotronH+
  `triton`=부팅 거부(`CONSENSUS.md` §3 항목103/게이트#83)를
  확인했는데 R2 correctness 조건 튜플은 `--attention-backend
  triton`이라 재확인 필요. provenance 구멍:
  `sync_engine_tree.sh`가 nemotron_h를 수동 복사한다면서 manifest
  17항목엔 zamba2·mamba2뿐(engine-porter 이관 중).
  **(2) 1차 캠페인 범위 = Claim D+P3까지** — 아래 "P2"
  acceptance는 `B1`/`B4` arm과 W3+W4 워크로드만 필요하므로
  고유 차단 225 중 135(B2/8/6)는 Claim E/oracle 쪽이며 Claim D와
  독립이다. **실행 순서**: **(0) λ* 측정(위 "공통 방법" 정정
  참조) → (1) 새 모델 R2 correctness 게이트 → (2) P1 observer
  effect → (3) P2(W3+W4×{B1,B4}×5rep) → (4) P3 offline profile
  → (5) Claim E(B6)**, 각 단계 개별 사전등록+규칙층 감사 대상.
  비용 감각(추정, 미측정): 실행 가능 180 run×~3분 ≈ **9–13
  GPU-h**(Zamba2-2.7B 기준, 9B 모델은 더 비쌈) vs 트랙 누적
  0.43 GPU-h.

  ★★★**(0)의 상태 갱신(2026-09-13(3), doc-steward, λ0 사전등록
  규칙층 감사)**: 위 "(0) λ* 측정"은 `NO-GO`로 판정됐다(死因
  N2+N3, GPU 0·미실행 — 상세 위 "공통 방법" 절). 재감사 통과
  전제로 **권고 순서를 (0a) 새 모델 correctness 게이트 +
  `--request-rate inf` 2셀(≈0.3–0.4 GPU-h, N2 死因 구조적
  소멸) → (0b) λ* 사다리 재감사·측정으로 개정**한다 — 즉 위
  "(0)→(1)"의 순서를 사실상 뒤집는다(correctness가 λ* 사다리
  설계를 추측에서 측정으로 바꾸기 때문). 추가로 **(3) P2**는
  W4를 포함하려면 위 "P2" 절의 열린 항목(단일 λ*가 W4 두
  phase를 동시에 파라미터화 못 함)이 **사용자 결정**으로 먼저
  풀려야 한다.

  ★★**세 번째 사용자 결정 + engine-porter 모델 지원 검증 `GO`
  (2026-09-13(2), doc-steward, GPU 0 이 세션)**: `--attention-
  backend triton`이 NemotronH에서 엔진에 거부됨(CPU-only
  `ServerArgs` 재현, `server_args.py:1959` assert)을 확인 —
  사용자가 **flashinfer로 전환**을 결정했다. 두 R2 하네스가
  triton을 하드코딩하고 있었다(`r2_correctness.sbatch`·
  `engine_bench_runner.sh`, 커밋 `87213a9`로 백엔드 노브 신설,
  **기본값은 `triton` 불변** — Zamba2 재현 보존, NemotronH arm은
  명시적으로 `flashinfer` 지정 필요). **X1의 발견(`triton_
  attention_num_kv_splits`)과 907100·907456 결론은 (Zamba2-2.7B,
  triton) 한정으로 동결**되고, 새 (모델, 백엔드) 쌍은
  correctness 게이트를 새로 쌓아야 한다(스코프 튜플에
  `attention_backend=flashinfer`·`model=nvidia/NVIDIA-
  Nemotron-Nano-9B-v2-Base`·`context_length=16384` 명시, 1
  job·9B 기준 ≈0.2–0.3 GPU-h 추정). 게이트#83/`CONSENSUS.md`
  §3 항목103(TC1 job 896776) 追記 — 같은 백엔드 강제 상호배타가
  **Nano-9B-v2-Base에서도 재현**됐다.

  **engine-porter 모델 지원 검증 = `GO`**: job 905835(2026-09-09,
  1.11 GPU-h, `longctx_conflict` 트랙 기존 지출 — 재인용, 이
  세션 신규 GPU 지출 아님)가 Nano-9B-v2-Base를 flashinfer·PD-mux
  ON·cudagraph ON·ctx 8704·D∈{16,44,92}로 **12/12 cell 전부
  boot**(BOOT_FAILED 0)했다. 가중치 완전(4 shard 16.56 GiB,
  index 341/341 일치, Σparams 8.888B) + config↔가중치 전수 대조
  + hybrid KV `cell_size=16 KiB/token` 확인, 메모리 제약 아님.
  **ctx 권고 = 16384**(131072 아님 — 트레이스 최대 요구
  8320토큰의 2× 여유). 잠복 상류 위험 3건(미수정, 등재만):
  `mamba2_cache_params`의 미선언 `self.n_groups` 의존·
  `config.expand` stale(미사용)·`piecewise_cuda_graph_disabled_
  model_archs` 목록 부재(cps>0 arm 생기면 재개). **overclaim
  금지**: 아직 아무 correctness 게이트도 돌지 않았다 — 있는
  것은 부팅·서빙 사실과 capacity 측정뿐.

  **λ* = 이미 측정돼 있음**(위 "공통 방법" 절 참조, job 905835
  `c_capacity.sbatch`): D16 0.933 req/s·D44 0.675·D92 0.187,
  arm별 5× 차이인데 단일 스칼라 캠페인 설계는 열린 문제.

  **provenance 구멍 해소**(커밋 `87213a9`, 이미 커밋, GPU 0):
  모델 구현이 manifest 안으로(17→24항목, 신규 7줄 append·기존
  순서 바이트 동일). ★**이 결함의 실증**: job 905835의 manifest
  는 17줄·`models/`가 mamba2·zamba2뿐인 채로 NemotronH를 12
  boot 서빙 — 모델 구현 provenance 0이었다. **신규 게이트
  #195**: "서빙한 모델의 구현 파일이 provenance manifest에
  없으면 그 캠페인은 모델 축에서 귀속 불가다." correctness
  하네스 ctx 하드코딩도 함께 제거(`R2C_CTX`, Zamba2 4096 재현
  보존)·`campaign.json` schema v2(model·context_length)·
  `cuda_graph` 읽히게 배선. 전체 463 tests OK(신규 37). 상세
  `PROJECT_STATUS.md` 최상단 배너(2026-09-13(2)), `CONSENSUS.md`
  rev72·§3 항목215(+항목103 追記), `CLAIM_EVIDENCE_MATRIX.md`
  Claim D 행. **커밋 금지**(이번 등재는 문서뿐, 코드는 커밋
  `87213a9`로 이미 완료).

상세 `workspace/engine-port/results/r2_correctness/{job_907032/,
job_907100/, job_907456/, audit_r2corr_2026-09-11/VERDICT.md,
audit_x1_2026-09-12/VERDICT.md, b3_prereg/, os_prereg/,
x3_prereg/}` §6–§7.3·§11. 정본 반영: `PROJECT_STATUS.md`
최상단 배너(2026-09-13)·"다음 실험 gate" 항목1–4 追記,
`CONSENSUS.md` rev70→rev71·§3 항목210–214, `CLAIM_EVIDENCE_
MATRIX.md` Claim D 행·"주장 제한" 갱신. **추가 반영
(2026-09-13(2))**: `results/longctx_conflict/probes/{ccap_
905835.out, c_905835/C_LABEL.json, PREREG_CAPACITY_
2026-09-09.md}`, `PROJECT_STATUS.md` 최상단 배너(2026-09-13(2)),
`CONSENSUS.md` rev71→rev72·§3 항목215(신설)+항목103(追記),
`CLAIM_EVIDENCE_MATRIX.md` Claim D 행 갱신.

### P3 — Offline profile/estimator

- CUDA Graph on, 실제 full-model decode
- D16/D24/D34/D44/D108
- batch 1/4/8/16/32/48
- context 256/1K/4K/8K, 지원 시 16K
- point당 warm-up 후 30 step 이상, 5회 이상

Profile은 model/revision/config hash, engine commit, GPU/driver/backend/graph,
attention/SSM count와 ratio, GQA metadata, latency percentile/residual을 기록한다.

Acceptance:

- upper-bound empirical coverage ≥95%
- under-reservation epoch ≤1%
- median over-reservation ≤한 state
- unseen/stale profile은 D44 fallback, live violation은 D108/admission limit

### P4 — Baseline와 profile ablation

| ID | Policy |
|---|---|
| B0 | vanilla continuous batching |
| B1 | training workload에서 선택한 one global static |
| B2 | per-workload offline best static oracle |
| B3 | 기존 single-worker dynamic |
| B4 | true dual fixed |
| B5 | true dual generic dynamic |
| B6 | true dual Hybrid-informed dynamic |
| B7 | layer-granular negative baseline |
| B8 | future trace와 transition/dwell cost를 아는 offline oracle |

Profile ladder는 no profile, model-size only, attention-ratio aware, full Hybrid
profile 순으로 동일 held-out point/trace에서 평가한다.

Claim E acceptance:

- target 영역에서 B6가 B1/B5보다 paired CI 기준 유의하고 ≥3% 개선
- non-target 영역에서 >3% regression 없음
- B2/B8은 upper bound로만 보고하며 이를 이긴다고 주장하지 않음

### P5 — Mechanism

대표 target/non-target point에서 Nsight Systems/Compute로 kernel timeline,
GPU idle gap, stream/event overlap, graph replay, SM active, occupancy, Tensor Core,
DRAM/L2를 수집한다.

### P6 — Workload

★**long-context(W5 등)의 논문적 역할(2026-07-25 positioning 판정,
`venue_positioning.md` §0.1(4))**: `longcontext_trace_plan.md`의 long-ctx
트랙(Stage 0/L−2 → L3/L3s)은 negative→가이드라인 전환과 "언제 유효한가" 경계
획정, 그리고 granularity 비용의 모델-composition 독립성(ctx-불변 구조적
성질)을 보이는 데 유효하다. **substrate 귀속(Risk 2)은 long-ctx가 아니라 위
벡터2(green-context 위 Transformer-control 대조 + roofline microbench + 기측정
entanglement 귀속)가 닫는다** — 별도 substrate serving 이식은 불필요·부적합으로
철회됐으므로, long-ctx가 "이식의 대체재"일 필요도 없다. 두 트랙은 서로 다른
질문(long-ctx=ctx-regime 경계, 벡터2=primitive/모델 귀속)을 담당한다.

★★**Stage 0(L−2) 게이트 실행(2026-07-26) — non-binding, ★★★2026-07-28 철회
(claims-auditor 감사, C1 CONFIRMED)**
(`../stage0_verdict_2026-07-26.md`, jobs 864230+864601): 2026-07-26엔 운영점서
decode SM-무감각이 hybrid·pure-Transformer·pure-Mamba 전부, ctx≤16k 전부로
확인됐다고 기록했으나, 근거였던 "D108 무경합 앵커"가 실은 decode 16 SM이었음이
3중 독립 증거(코드 기전·telemetry 재집계·클라이언트 서명)로 확인돼 **판정을
철회**한다. **L−2는 SM-binding 여부를 측정한 적이 없다** — "negative→가이드라인
전환" 역할은 실현되지 않았고, `longcontext_trace_plan.md` §6의 L−1 이상은
"게이트 실패로 보류"가 아니라 **"게이트 미실행"**이다.

★★**대신(2026-07-28) 8B decode-SM 민감도 측정 노트 — C2 CONFIRMED(scoped)**
(`../../workspace/engine-port/results/s8_scaleup/FINDINGS_8B_2026-07-28.md`,
jobs 865289–865533): prefill을 16 SM에 고정한 채 decode-SM만 16→92로 올리면
decode ITL이 **2.36–2.91×**(4 arm, 모델-무관) 개선된다 — Stage 0가 주장하던
"SM-무감각"과 정반대 방향. 단 이는 **decode 측 등량곡선**(예산 제약
`prefill+decode≤108` 없음)이라 **레버 존재만 확립**하며 정책 이득 근거가
아니다. ★**국소 탄력도 부기(2026-08-04, claims-auditor)**: 이 비율은
**16→92 끝점 비**이며 **국소 탄력도는 16→24 0.77–0.88 vs 44→92
0.09–0.35로 4× 다르다 — 44 이상 구간에 그대로 적용하지 말 것**
(`../../PROJECT_STATUS.md` "8B decode-SM 민감도 측정 노트"). ⚠️**인용정지
(a) 표기(doc-steward, 2026-08-16)** — 이 arm별 국소 ε 밴드(0.09–0.35)는
**인용정지 (a)**(arm별 ε·arm 간 순위/격차, `../CONSENSUS.md` §3 항목54)의
숫자 인용과 동치다. arm 라벨·숫자 없이 정성 서술("고-SM 구간이 저-SM
구간보다 국소 탄력도가 뚜렷이 낮다")만 인용 가능. 상세
`../PRIZE_SIZE_ARGUMENT_2026-08-16.md` §6. **claims-auditor가 지정한 프론티어 게이트 E1**(`[108−D,D]` 스윕,
D∈{16,24,44,54,92}+best-static 대조, 4 arm, offered-rate 고정, n≥4, 사전등록
파티션 점유율≥0.80·활성률≥0.60 게이트, 결정규칙: best static 대비 conjunctive
goodput ≥3% 개선 & paired CI가 0 배제)가 이 레버가 예산 제약 하 net-positive인지
판정한다 — 병행 게이트 E2(ctx 확장)·E3(duty-cycle, 설계상 종결)·E4(C2b는 통제
불가하므로 주장 폐기). 상세 `../PROJECT_STATUS.md` "8B decode-SM 민감도 측정
노트"·"열린 긴장"·"다음 실험 gate" #8.

★★**E1 상태(2026-08-02) — 본 스윕 미제출, 설계 위험 3중**. 2026-08-01에 전제
실험 4건이 완료됐다(jobs **870295**=M8/**870296**=Ha8/**870297**=Hs8 용량
스캔 각 100 probe, **870301**=T8 batch-cap 24 probe, 전부 오류 0; 원자료
`../../workspace/engine-port/results/s8_frontier/`). ⚠️**이 4건은 전부
claims-auditor 미통과 = 미검증, 인용 금지**이며 수치는 `../PROJECT_STATUS.md`
"열린 긴장"의 "2026-08-01 실험 4건" 소절에만 둔다. 로드맵 차원에서 기록할
것은 **판정이 아니라 설계 위험**이다: (a) 사전등록 사다리 {50,60,80}ms가
as-run 설정에서 **네 arm 전부 헤드라인 룽 없음**, (b) 그 as-run 설정
(`--max-running-requests 48`)이 ITL·TTFT 두 축을 반대 방향으로 왜곡함이
직접 실험으로 드러남, (c) on-cliff 제외 규칙(d92 knee 2.80, 전 arm)이 **어떤
동작점에서도 decode-rich 끝을 제거** — C2 레버가 사는 끝.

> ★★★**정정(2026-08-02, claims-auditor 회부 + M1/M2/M4 후속) — 위 세 다리 중
> 둘이 무너졌고, 그래서 "설계상 도달 불가" 종결은 정당화되지 않는다.**
> - **(c) 기각.** C2 측정(`s8_scaleup/FINDINGS_8B_2026-07-28.md` §2)에서
>   SM16→44 구간이 log-range의 **75–81%**를 차지한다 ⇒ d92 하나 제외는 레버가
>   사는 끝이 아니라 **마지막 15–25%**만 자른다. 게다가 "공통 knee 2.80"
>   자체가 철회됐다(knee의 치역이 probe 격자뿐이라 일치가 부분 강제 —
>   살아남는 건 **순서**뿐, `results/s8_frontier/DESIGN.md` §4.3.7).
> - **(b) 운영구간 밖.** cap 왜곡은 rate 16에서만 관측됐고, 운영대역(≤2.80)
>   실측 동시성은 12–44 < cap 48이라 **cap이 구속할 수 없다** ⇒ E1에 적용 안 됨.
> - **(a)만 생존**하되 arm×룽 표는 **rate-confound로 폐기**(T8만 rate 12).
>   공통 rate로 재계산하면 `HEADLINE-ELIGIBLE RUNGS = NONE`은 **유지**된다
>   (두 estimand × 두 seed 전부).
>
> ⇒ **(A) 제출도 (B) 종결도 시기상조**로 판정하고, 대신 **M3
> Transformer-control 대조**를 사전등록·제출했다(job **872077**, §4.3.8(c)).
> M3는 goodput 이득이 아니라 **"ITL 축이 D에 반응하기는 하는가"**를 묻는다 —
> 그 질문이 긍정이어야 프론티어 질문이 성립하기 때문이다. 오프라인 예비값:
> blocking 제거 후 d16→d54 기울기가 T8 2.03× 대 Ha8 1.03×/0.89×.
> **선행 필수 정정**: `--max-mamba-cache-size`는 **전 arm 공통 절대상수가
> 아니라 `= cap` 규칙**으로 고정해야 한다 — slot당 비용이 arm마다 달라
> (M8 0.255 / Ha8 0.141 / Hs8 0.096 GB) 절대상수는 arm마다 다른 메모리 분할을
> 강제하는 **새 cross-arm 교락**이 된다(`../CONSENSUS.md` §1-23 따름정리 정정).
>
> ★★★**정정(2026-08-03, claims-auditor, 같은 세션 속행) — M3(872077)의
> NO VERDICT 사유가 "CI 폭 부족"에서 "estimand 미식별"로 확장됐다.** 코드
> 사실: `pdmux_context.py:initialize_stream_groups`가 마지막에 무조건
> `(0,108)` 무분할 그룹을 덧붙이고 prefill이 비-in-flight면 그리로 되돌아간다
> (`multiplexing_mixin.py:773,792-794`) ⇒ 이 격자에서는 **"decode가 D SM에서
> 돌았다"와 "prefill이 동시 in-flight였다"가 같은 사건**이다. §1-24(ITL 꼬리
> =monolithic prefill, 크기가 108−D에 단조)와 결합하면 `g`는 사전에
> **"decode-SM 탄력도 라벨을 단 prefill-SM 탄력도"**일 것이 예상되고, 실측
> (UNSPLIT-only 부분집합만으로 T8 헤드라인 재현)이 그와 일치한다. **이는 n을
> 늘려도 해결되지 않는 설계 결함**이라 `g = A_free(d16)/A_free(d54)`는 **이
> 격자 한정 은퇴**(sticky-partition 기판 수정 전 인용 금지), 블록 8→12–16
> 증설 재실행은 **선행 금지**. 동시에 메인 세션이 세운 "decode 실현 4–19%가
> `g`를 attenuate했다"는 보정 가설도 **REFUTED**(control-arm reductio: T8에
> 같은 보정 적용 시 corrected g 21–29×로 C2를 10배 위반; de-engagement 직접
> 실험에서 w=0에도 g 1–11%만 이동; "A(108) 셀 무관" 가정이 UNSPLIT-only
> 부분집합 분해로 반증). ⚠️"Ha8에 레버가 없다"는 CONFIRMED 아님 — **긴장
> A(HE2 vs C2)는 전혀 닫히지 않았다.** 다음 gate: `PDMUX_STICKY_PARTITION`
> 구현(engine-porter, correctness gate) → sticky 격자 1회(872077 동일 설계
> 8 block, non-sticky 대조로 872077 사용) → 사전등록 판별 예측(prefill
> 주도라면 T8≈1.85·Ha8≈0.92로 하강, 희석 가설이 옳았다면 Ha8≈1.6로 상승 —
> CI 비중첩이라 8 block으로 구분 가능) → `E1_DECODE_REALIZED≥0.90`이
> sticky에서는 **진짜 게이트**(더 이상 항등식 아님). 상세 `../CONSENSUS.md`
> §1-26, `../../PROJECT_STATUS.md` "8B decode-SM 프론티어" "2026-08-03"
> 소절, `../../workspace/engine-port/results/s8_frontier/DESIGN.md` §4.3.9.
>
> ★★**(2026-08-03, 같은 세션 2차 속행) 위 두 다음-gate가 모두 진행됐다 —
> estimand 이관 완료, 구현 완료, 런은 아직 미제출.** claims-auditor가
> `A_free`를 대체하는 **조건부 per-token 추정량**(`m3_conditional.py`,
> `DESIGN.md` §4.3.10)을 만들었다[AUDITED, blocking-threshold 스윕만
> UNAUDITED]. engine-porter가 `PDMUX_STICKY_PARTITION`을 구현했고(`DESIGN.md`
> §4.3.11) correctness gate 전부 PASS(CPU 회귀 40 tests + sticky 단위
> 테스트 12 + GPU smoke job 872800, 6개 고정 프롬프트 OFF/ON greedy 출력
> byte-identical) — **구현 완료 ≠ 성능 주장 성립**이며, realized 관측(n=1)
> `E1_DECODE_REALIZED` OFF 0.0839 → **ON 1.0000**(사전등록 게이트 ≥0.90
> 초과)만 기록됐다. **sticky 격자 런은 여전히 미제출** — `DESIGN.md`
> §4.3.12가 사전등록한 4개 게이트·primary 통계량은 확정했으나
> **`G_LEVER`/`G_FLAT`는 미결정으로 남겨 두었다**(기존 1.5/1.15는 `A_free`
> 스케일이라 새 추정량에 그대로 이전 불가 — C2 측정범위[2.36–2.91×]에
> 묶는 안이 논거는 있으나 E1의 D 범위[16→54]가 좁고 상보적[P+D=108]이라
> 확정 전 별도 사전등록 필요). 판별 예측(위 문단)은 불변. 상세
> `../CONSENSUS.md` §1-27, `../../PROJECT_STATUS.md` "8B decode-SM
> 프론티어" "2026-08-03(2차)" 소절, `DESIGN.md` §4.3.10–4.3.12.
>
> ★★★**(2026-08-03, 같은 세션 3차 속행) `G_LEVER`/`G_FLAT`의 미결정을
> C2 데이터로 닫으려던 시도 — 경로 폐기, UNDETERMINED 그대로.**
> `c2_anchor.py`로 시도한 5개 주장을 claims-auditor가 감사: **주장
> 1(realized 검증)만 CONFIRMED**(서술 2건 정정 필요 — 활성률은 count
> 가중, 108 SM 시간은 drain 전용), **주장 2(primary p95→p50)는 관측
> CONFIRMED·처방 REFUTED**(p50 전환 시 872077 T8 양성대조조차 1.00으로
> 무너져 캠페인을 구조적 NO VERDICT로 만드는 처방이었다 — **primary는
> `p95(SPLIT)` 유지**), **주장 3–5는 NOT-YET-SUPPORTED/REFUTED/REFUTED**
> (`G_LEVER=1.41`은 끝점 선택만으로 [1.41,2.40] 전 구간 도달 가능해
> REFUTED, `G_FLAT=1.25`는 LOO 실측 반폭이 1.30인데 Ha8 1.340으로
> 자기 데이터서 뒤집혀 REFUTED). ★**§0 신규 최상위 열린 항목**: 같은
> arm·서버 플래그·매칭 batch에서 C2(865493)와 872077의 "decode 16 SM"
> per-token ITL이 **2.6× 다르다**(28.79ms vs 11.09ms) — 872077의
> `decode_sms==16`이 실제 하드웨어 16-SM 실행인지(`DESIGN.md` §4.3.11의
> 미검증 잔여층) 또는 C2 값이 셀 배치 성질인지 미해소, **872077 전체와
> sticky 결과가 딛고 선 바닥**. **sticky 격자 제출은 이 모순 해소 이후로
> 미룬다.** 별도로, D=54 앵커 측정(jobs 872920/872921)이 keepalive 토큰
> 초과(재현성 결함)로 취소됐고, 독립 수렴으로 **C2의 높은 residency는
> decode 파티션 제어가 아니라 keepalive 워크로드 장치의 산물**임이
> 확인됐다(C2 앵커가 죽는 세 번째 이유). `G_LEVER`/`G_FLAT`는
> **UNDETERMINED로 유지**, 다음 시도는 감사자 발안 (α) sticky 파일럿
> 양성대조 효과크기 또는 (β) arm 간 대비 `g_T8/g_Ha8`(batch-매칭
> rate)에 대한 **독립 사전등록**이 선행돼야 한다(감사자가 자기 발안의
> 승인 주체일 수 없다). 상세 `../CONSENSUS.md` §1-28·§1-29,
> `../../PROJECT_STATUS.md` "8B decode-SM 프론티어" "2026-08-03(3차)"
> 소절, `DESIGN.md` §4.3.13–4.3.14.
>
> ★★★**(2026-08-03, 같은 세션 4차 속행) §0의 이분법이 유지 불가로 판정
> — 세 번째 후보 (iii) 실측 문서화, 오프라인 분리 불가, GPU(S2) 별도
> 제출 중·결과 없음. 성능 판정 0건.** §0은 (i) 872077의 `decode_sms==16`
> 이 실제 16-SM 실행이 아니다 / (ii) C2의 28–31ms가 셀 배치 성질이다 중
> 하나가 거짓이라는 이분법이었다. claims-auditor가 `FINDINGS_S0_AXIS_
> 2026-08-03.md`를 감사하며(자기감사, 방법론 교훈 12) 이 이분법이
> "`split_frac≥0.90`이 D 파티션 실행 토큰을 올바로 분리한다"는 전제 위에
> 서 있고 그 전제가 T8 세 공유 셀 전부에서 깨진다는 재프레이밍을 냈다 —
> E1 SPLIT 모집단이 이봉이고 윗봉이 C2 셀별 p50과 1–2% 일치, 아랫봉은
> 같은 job UNSPLIT과 통계적으로 동일. **독립 재현**(result-analyst,
> `S0R_REPLICATION_2026-08-03.md`, claims-auditor도 사전등록 세션도
> 아님, 감사된 `m3_conditional.py` 프리미티브만 프리미티브별 재사용
> 선언·생산자 자체에 gate 대조): 행 1·3 재현, 행 2는 순서만(d24−d16
> 미해결), **행 5(클럭 lag) 미발화**(최적 δ=+0.10s서도 slow share
> 11.4%). ★**행 4(음성대조)가 강한 형태를 죽였다**: 같은 estimator를
> UNSPLIT(108 SM)에 적용하면 T8 전 셀에서 동일한 슬로우 모드가 나타나고
> 그 위치가 셀을 따라간다(33.88→22.12→15.62→14.12ms, D 16→24→44→54;
> d16 슬로우 토큰 8,893개 중 7,903개=88.9%가 UNSPLIT 라벨) ⇒ SPLIT은
> 배타가 아니라 **농축**(2.33–3.24×). ⇒ **세 번째 후보 (iii)**: 두
> job은 같은 축이나 라벨이 순수하지도 완전하지도 않다 — §0은 이제
> 3지선다이며 오프라인으로 분리 불가. 남은 두 읽기(셀 수준 현상 vs 클럭
> 오프셋 누출)는 **S2**(GPU, `PREREG_S2_STICKY_ITL_2026-08-03.md`, 별도
> 제출 중, 결과 없음)만이 인과적으로 분리 가능. **철회 3건**(메인
> 세션이 같은 날 앞서 씀, `FINDINGS_S0_AXIS_2026-08-03.md` 배너와
> 동일): "§0 stands as written"·"aggregation-invariant"·"11.09는 집계
> 단위 미기록"(**틀림** — 산출자는 `m3_conditional.report_conditional`
> [3] `sp_p50=11.0905`, n=11,124, `a_free_only=True`,
> `m3_conditional.py:158-161,251-262,316-329`에 문서화, 없는 건 stdout
> 저장분뿐). **재사용 계측 결함 2건**: `c2_anchor.py` 표 [5]가 M8 전체·
> Ha8 d16을 조용히 누락(`meta`가 `t0_monotonic_s` 분기 안에서만 채워짐,
> `c2_anchor.py:181-187` — Ha8 d16은 돌았다, `itl_ms_p50=112.84`
> n=5200, 빠진 건 텔레메트리 앵커뿐) · mode estimator 60ms 상한은
> arm-이식 불가(Ha8은 토큰의 0.16%만 창 안). **증거 수준**: 강한
> 형태(레버=D-SM 실행)는 **채택 불가**(음성대조 반증), 약한 형태(이봉·
> 농축)는 **재현됨(독립성 부분적** — 추정량은 감사자 제안, 사전등록은
> 메인 세션, 실행만 독립**)**. 게이트 S1(§4.3.13)은 "부분 실현" 분기가
> 없어 **현 상태로 실행 불가**(4번째 분기 필요). `G_LEVER`/`G_FLAT`는
> §4.3.12(d) 그대로 **UNDETERMINED 유지**(이번 회차로도 미해소).
> **게이트 정의(3.4.4 결정규칙) 변경 없음.** 상세 `../CONSENSUS.md`
> §1-30·§3-18·§3-19, `../../PROJECT_STATUS.md` "8B decode-SM 프론티어"
> "2026-08-03(4차)" 소절, `DESIGN.md` §4.3.15.

| ID | 고정 workload |
|---|---|
| W1 | input 2K/output 128, Poisson 0.60 lambda* |
| W2 | input 8K/output 64, 10초 4× burst+30초 drain, 3 cycles |
| W3 | input 256/output 512, 0.80 lambda* |
| W4 | W2형/W3형 30초 phase를 3 cycles |
| W5 | 1K/8K context 교대, 지원 시 16K |
| W6 | input 2K, output 32/512 교대 |
| W7 | 0.20 lambda* |
| W8 | 0.90/0.95 lambda* |
| W9 | 1.10/1.25/1.50 lambda* |

ShareGPT와 현재 cache된 LongBench를 우선 사용한다. coding/agentic trace는
출처·license·전처리 규칙이 확정된 뒤 추가한다.

Applicability map은 average combined demand, peak-minus-average demand,
prefill/decode pressure temporal correlation으로 만들고 B6−B1 goodput effect를
색으로 표시한다. `peak sum>1`, `average≤1`, pressure 교대 영역을 사전 정의한
target으로 사용한다.

## Controller ablation

| Ablation | 검증 claim/mechanism |
|---|---|
| dual worker 제거 | D |
| dynamic 제거/fixed | architecture 대 policy |
| profile 및 feature ladder | A, E |
| runtime context 제거 | A |
| ITL slack 제거 | E의 SLO protection |
| feasibility gate 제거 | unsafe downshift |
| hysteresis/dwell 제거 | oscillation |
| safety margin 제거 | under-reservation |
| static floor | runtime load term |
| layer-level switching | B |
| CUDA Graph off | B와 operating-point sensitivity |
| chunked prefill on/off | orthogonal composability |

## Controller defaults

- steady: D16/D24/D34/D44; emergency D108
- 정상 평가 케이던스 = `max(4 decode iterations, 100 ms)`(**둘 다** 충족해야
  발화, `_dwell_satisfied`와 같은 AND). ★`or bucket change`는 **미구현**이다
  — 서빙 경로 `_r2_decide_idx`가 `bucket_changed`를 넘기지 않고
  `src/multiplex`에 bucket 정의가 없다. 파라미터는 API·테스트 호환과 "이
  절이 미구현임"을 남기기 위해 유지한다.
- ITL p95 > SLO **또는** KV/running-batch 점유 ≥ 85%면 케이던스를 기다리지
  않고 **그 iteration에 즉시** 재평가한다(`evaluation_due`의 독립 disjunct,
  위 케이던스 항과 별개 조항). 이 즉시 평가는 **upshift 전용**(target을
  내리지 않으며 `downshift_streak`을 0으로 리셋한다)이고, `safe_boundary`가
  거짓이면 전환하지 않고 `requested_decode_sms`만 기록한 뒤 다음 iteration에
  재시도한다. 긴급 평가도 epoch 시계를 재시작하므로 **다음 정상 평가는 긴급
  평가 시점 기준 1 epoch 후**다.
- downshift only below 0.75×SLO for 3 epochs
- dwell `max(8 decode steps, 200 ms)`; upshift exempt
- **(a)** `target ≥ D108` **이고** 예측 상한 `upper_bound_itl_ms` > ITL SLO,
  **또는 (b)** KV 점유 ≥ 90%, **또는 (c)** running-batch 점유 ≥ 90% — 이 중
  하나가 **연속 2 epochs**(증가는 케이던스 epoch당 최대 1회) 지속되면
  admission 제한. ★(a)의 `target ≥ D108`은 장식이 아니다: 호환 프로파일이
  없으면 `upper_bound_itl_ms = inf`가 되어 (a)의 부등식이 무조건 참이 되므로,
  `target ≥ D108` 결합이 **프로파일 없는 hybrid arm의 무조건 throttle을
  막는 안전장치**다(순수 OR 금지). 제한은 설정한 평가 이후의 비평가(HOLD)
  iteration에도 유지되고, 이후 평가 또는 런타임 backstop
  `release_admission_limit()`이 해제한다.

★★追記(2026-09-12, doc-steward — engine-porter 코드 사실 3건,
2026-09-11 확인·file:line 근거, **성능 판정 아님·수정 없음·사용자
결정 대기**): 위 명세 두 줄이 실제 컨트롤러/프로파일 코드
(`workspace/engine-port/src/multiplex/`)와 다음 지점에서 불일치한다
— 어느 쪽(코드/이 로드맵)이 옳은지는 판정하지 않는다.

1. **평가 주기 연산이 반대다.** 위 `evaluate every max(4 decode
   iterations, 100 ms)`(:975, `max`=둘 다 채운 뒤 발화)와 달리, 코드
   `evaluation_due()`(`controller.py:124-130`)는 `bucket_changed OR
   경과시간≥100ms OR 경과iteration≥4`, 즉 **셋 중 먼저 오는 것**(`OR`/
   min 의미)에 발화한다.
2. **admission 제한의 지속 기간이 사실상 ~1 iteration이다.** 위
   `D108 risk 또는 occupancy 90%가 2 epochs 지속되면 admission
   제한`(:979)은 트리거 조건만 적고 있으나, 트리거된 뒤의 지속은
   코드상 짧다 — `stabilize()`가 `evaluation_due()==False`인 매
   iteration마다 `SplitDecision(current, current, HOLD)`을 반환하고
   (`controller.py:148-149`) 이 반환값의 `admission_limited`는
   dataclass 기본값 `False`다(`controller.py:68`). `_r2_decide_idx`
   (`multiplexing_mixin.py:430-441`)가 이 값으로
   `self.r2_admission_limited`를 매 호출 무조건 덮어쓰고, 그 호출은
   split-prefill in-flight 또는 admission-recheck 경로
   (`multiplexing_mixin.py:1066-1080`)로 거의 매 iteration
   일어난다 — 즉 latch가 평가 iteration에 `True`가 된 바로 다음
   (비평가) iteration에 `False`로 되돌아간다.
3. **hybrid profile 호환 검사가 `engine_commit`을 포함해 거의 항상
   fallback이다.** `HybridModelProfileV1.is_compatible()`
   (`profile.py:98-114`)이 `engine_commit`(:102, repo HEAD —
   `scripts/r2_eval/r2_eval.sbatch:26`이 `git rev-parse HEAD`로
   설정)을 포함한 6개 필드 엄격 일치를 요구해, 엔진과 무관한 문서
   커밋만으로도 `ConservativeDecodeFloorEstimator.estimate()`
   (`profile.py:268-279`)가 `upper_bound_itl_ms=inf`·`fallback=True`로
   떨어진다 — B6(hybrid) arm이 기본값으로 profile 없이 도는 구성이
   될 수 있다.

**Claim E(위 "P4" 절·"Controller ablation" 절 A/E 행) 착수 전 해소
필요 — 전부 고치지 않고 표시만 해 둔다(사용자 결정 대기).** Claim E
등급(미검증)·B0–B8 순위·새 성능 판정 전부 무변경. 전문·인용 근거는
`CLAIM_EVIDENCE_MATRIX.md` "주장 제한" Claim E 항목(2026-09-12 추가)
참조.

★★追記(2026-09-12(2), doc-steward — 위 3건 해소 완료, 커밋
`3f188bb`, engine-porter, GPU 0, 사용자 결정: "3건 모두 이 로드맵이
정본"): 위 1–3은 전부 코드로 해소됐다.

1. `evaluation_due()`가 이제 이 로드맵(:975)의 `max(4 iterations,
   100 ms)` 의미(`bucket_changed OR (경과시간≥100ms AND
   경과iteration≥4)`)로 동작한다(`controller.py:129-142`).
2. admission 제한 지속을 컨트롤러 상태로 보존하고 HOLD가 그 값을
   승계한다(`controller.py:127,174-185,219-220`) + `release_
   admission_limit()` 신설(`multiplexing_mixin.py:1809-1817`,
   FixedPolicy는 no-op).
3. profile 호환 축에서 `engine_commit`을 제외하고 실제 임포트된
   엔진 모듈 8개의 내용 해시 `engine_source_hash`로 교체
   (`profile.py:44,113-160,367-481`, `:102`의 `engine_commit` 엔트리는
   더 이상 존재하지 않는다).

검증: 신규 테스트 38개 + 수리를 되돌린 변이 전부 실패 확인, 전체
**343 tests OK**(직전 305), `check_line_citations.py` 0 violation.

★**새로 생긴 긴장 2건 — 미해결(사용자 결정 대기)**: (1) 위
"immediate safe-boundary upshift"(:976)가 해결 ①로 인해 최악
반응 지연이 **길어질 수 있다**(평가 주기가 이제 둘 다 충족해야
발화) — 이 패치가 Claim E 거동에 실제 영향을 줄 수 있는 유일한
항목. (2) 위 "D108 risk **또는** occupancy 90%"(:979)는 코드에서
`AND` 합성(`controller.py:210-218`)이라 occupancy 90%만으로는
제한이 걸리지 않는다(접속사는 그대로 둠 — 2026-09-12(1차) 등재는
지속 기간만 문제 삼았음). 둘 다 어느 쪽이 옳은지는 판정하지 않는다.
★★**해소됨(2026-09-12(4), 아래 追記 참조)** — 원문은 역사로
보존하고 상태만 갱신한다.

깨진 줄 인용 갱신: `controller.py:124-130 → :129-142` ·
`:148-149 → :174`(+`:180-185`) · `:174-183 → :210-219` ·
`:183 → :219` · `:207 → :244` · `:222 → :259` · `profile.py:
98-114 → :113-160` · `profile.py:268-279 → :314-325`.
★★**이 인용은 다시 이동했다 — 아래 追記(2026-09-12(4)) 참조**
(이 줄은 역사 보존용). ★**추가 갱신(2026-09-14, doc-steward
— engine-porter 등록성 검사에서 발견)**: 위 목록이 `scripts/
r2_eval/r2_eval.sbatch:26 → :69-70`(`PDMUX_ENGINE_COMMIT`
export, 커밋 `09a8075`가 이동시킴)를 빠뜨렸다 — 이 파일은
`check_line_citations.py`의 이 문서 `__scope__`(controller.py/
profile.py/workloads.py 패턴만) 밖이라 위 8건과 함께 잡히지
않았다(신규 게이트 #238, `PROJECT_STATUS.md` "방법론 게이트"
참조). 인용된 결함(`engine_commit`이 hybrid 호환 6개 필드
엄격 일치에 포함)은 위 항목3의 해소로 이미 없다 — 이 문단은
그 해소된 문장이 남겨 둔 하위 인용 하나만 정정한다.

Claim E 등급(미검증)·B0–B8 순위·새 성능 판정 전부 무변경. 상세
`CLAIM_EVIDENCE_MATRIX.md` "주장 제한" Claim E 항목(2026-09-12(2)
갱신), `PROJECT_STATUS.md` 최상단 배너 B 항목.

★★追記(2026-09-12(4), doc-steward — 위 "새로 생긴 긴장 2건"
해소 완료, engine-porter A안 구현, GPU 0, **작업트리 미커밋**,
사용자 승인): 위 두 긴장은 전부 해소됐다. 위 5줄짜리 명세
(`steady`·케이던스·즉시 upshift·downshift·dwell·admission)는
이미 이 문서 상단 "Controller defaults" 절 본문에서 최신
문안으로 교체됐다(engine-porter 제안 문안, 문자 그대로 적용).
요지:

1. (1)의 해소: `_live_underprediction`(ITL p95>SLO 또는
   KV/running 점유≥85%)을 단일 정의원 헬퍼로 추출해
   `evaluation_due()`를 `bucket_changed or _live_underprediction
   (snapshot) or _cadence_due(snapshot)`의 **독립 disjunct
   셋**으로 재조립했다(`controller.py:196-216`). `:975`(정상
   케이던스, `_cadence_due`, `:168-179`)와 `:976`(즉시 upshift,
   `:151-166`)이 이제 독립 조항이라 위반이 난 그 iteration에
   즉시 재평가한다 — 최악 반응 지연이 다시 짧아졌다. 이 즉시
   평가는 **upshift 전용**(target을 내리지 않고
   `downshift_streak`을 0으로 리셋)이고, `safe_boundary`가
   거짓이면 전환 없이 `requested_decode_sms`만 기록한 뒤 다음
   iteration에 재시도한다. 다음 정상 평가는 긴급 평가 시점
   기준 1 epoch 후다.
2. **①의 필수 동반 수리(신규)**: overload streak 증가를 별도
   epoch 마커(`last_overload_epoch_s/_iteration`,
   `_overload_epoch_elapsed` `:181-194`)로 케이던스 epoch당
   최대 1회로 게이팅했다. 없으면 긴급 평가가 매 iteration
   발화해 연속 2 iteration(≈27–85ms)만에 admission 제한이
   걸렸을 것이다(로드맵 "2 epochs"[≥400ms]와 다르고, 이
   프로젝트에서 TTFT 폭발을 일으키는 레버). 과부하 해제 시 즉시
   0 리셋은 유지, `release_admission_limit()`(`:218-232`)이
   epoch 시계도 재시작한다(`:231-232`).
3. (2)의 해소: `:979`를 `(target≥D108 AND upper_bound>SLO) OR
   kv≥0.90 OR running≥0.90`으로 갱신했다(`controller.py:
   311-318`). 죽어 있던 점유율 90% 단독 트리거가 살아났고
   (이전엔 `target≥108` AND가 전부를 막아 발화 불가), 동시에
   `upper_bound=inf`(프로파일 비호환 fallback,
   `fallback_decode_sms=44`)가 저점유에서 무조건 throttle하는
   순수 OR은 명시적으로 금지했다(B6 arm이 조용히 제한되는 것
   방지). 두 방향 다 테스트로 고정. "코드/로드맵 중 옳은 쪽"을
   판정한 것이 아니라 둘 다 이 로드맵 산문으로 흡수했다.
4. `bucket_changed`는 **미구현임이 코드에 명시**됐다 —
   `_r2_decide_idx`(`multiplexing_mixin.py:436-440`)가 이
   인자를 넘기지 않고 `src/multiplex`에 bucket 정의가 없다.
   파라미터는 API·테스트 호환과 "이 절이 미구현"이라는 기록을
   위해 유지한다(`stabilize` bucket docstring, `controller.py:
   249-262`). 즉시성은 ①이 제공한다.
5. 감사 가능성: `SplitDecision.off_cadence`(신설, `:78`) +
   `controller_decision` 텔레메트리 필드
   (`multiplexing_mixin.py:452`)로 긴급 경로가 케이던스 밖에서
   발화했는지를 기록한다. 기존 필드 의미·스키마 변경 0.

검증: **전체 CPU 362 tests OK**(직전 343, +19), 변이 트리 12종
(M1–M12)에서 각 수리가 실패함을 확인, `check_line_citations.py
--check --all` = 50 compared **0 violation**(전·후, doc-steward
독립 재실행 확인). manifest 2항목 변경(`controller.py
bd9e6660…→a4fde4d3…`, `multiplexing_mixin.py
a3b9a4e9…→0fe9d570…`). **FixedPolicy/Claim D 경로 불변**(6개
이름 부재를 테스트로 고정, 변이 M8이 잡음), 공용 경로 편집은
텔레메트리 kwarg 1개뿐이고 `r2_correctness_check.py`는
`rec.get(...)`이라 B5 POLICY 검사 무영향.

★새 사실 1(**미측정**, 열린 항목/후속 결정 후보로만 등재): dwell은
upshift에 면제(`:977`)이므로 위반이 지속되면 상태 사다리가
연속 iteration마다 한 칸 오른다 — 기본 케이던스에서
`D16→24→34→44→108`이 4 iteration(≈40ms), 이전 판본은 4
epochs(≥400ms)였다. 칸마다 green-context 드레인 1회다. **성능
영향은 측정된 바 없다**(generic/hybrid GPU 측정 0건). "upshift를
epoch당 한 칸으로 제한할 것인가"는 아직 결정되지 않았다 —
성능 주장으로 인용 금지.

★새 사실 2(프로세스 교훈): 세션 시작 시 dev tree가 stale이어서
`sync_engine_tree.sh`(→ module load + venv 활성화 포함 CLAUDE.md
부팅 절차) 전에는 테스트 다수가 실패한다. 343/362 기준선은
그 절차 이후에만 재현된다(doc-steward 독립 재확인: 절차 없이
`python -m unittest discover`는 220개만 수집·다수 실패, 절차대로
하면 362 OK) — CLAUDE.md 부팅 절차의 근거 사례.

깨진 줄 인용 갱신(engine-porter 제공, 위 줄이 다시 갱신됨):
`controller.py:129-142 → :196-216` · `:127 → :148` ·
`:174-185 → :263-274` · `:210-218 → :311-326` · `:219 → :327` ·
`:219-220 → :327-328` · `:244 → :352` · `:259 → :368`. 불변:
`:68`, `:17`, `profile.py` 전부(파일 미수정), `multiplexing_
mixin.py` 17개 인용(줄 수 1879 유지). 신규 앵커:
`_live_underprediction :151-166` · `_cadence_due :168-179` ·
`_overload_epoch_elapsed :181-194` · `off_cadence` 필드 `:78` ·
계산 `:277` · release epoch 리셋 `:231-232` · `stabilize`
bucket docstring `:249-262` · 텔레메트리 필드
`multiplexing_mixin.py:452`.

신규 방법론 게이트 3건(#177–179, `CONSENSUS.md` §3
항목197–199). Claim E 등급(미검증)·B0–B8 순위·새 성능 판정
전부 무변경. `CONSENSUS.md` rev68→**rev69**(신규 게이트가
사유). 상세 `PROJECT_STATUS.md` 최상단 배너(2026-09-12(4)),
`CLAIM_EVIDENCE_MATRIX.md` "주장 제한" Claim E 항목
(2026-09-12(4) 갱신).
