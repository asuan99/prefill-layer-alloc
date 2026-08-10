# CONSENSUS — engine-port PD-mux 연구의 합의점 (2026-07-19 historical)

> **현재 전체 정본:** [`../PROJECT_STATUS.md`](../PROJECT_STATUS.md).
> 이 문서는 dual-worker/R2 이전까지 확정된 phase separation, layer-granular
> negative result, entanglement, single-worker dynamic 결과의 정본으로 유지한다.

최종 갱신: 2026-08-11 rev21 (doc-steward — **§3 항목40 신설[대형
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
| 1 | ~~**PD 분리 자체는 항상 이득**~~ → ★반증/정정(2026-08-05, claims-auditor, jobs 873944/873945) **PD-mux 활성화는 운영점에서도 꼬리 SLO goodput 이득 — 술어·모델·워크로드 한정, 기전 귀속 미확립** | agnostic이 fused를 4모델 전부서 이김. ★**스코프 축소(2026-08-04, claims-auditor)**: 이 4-모델 캠페인은 **전부 `--disable-cuda-graph`**(no-cudagraph 비운영점, `triage/p1_7_bench_one.sbatch:42`)이고 **rate 1에서는 동률**(도착률 천장)이며 **운영점(cudagraph-ON) 대조는 어느 모델에서도 측정된 적 없다**. ★**반대 증거 신규**: fused의 死因은 **TPOT > 60ms 임계 초과**(Granite rate4 TPOT 61.21)인데 **cudagraph가 그 벽을 제거한다**(plain TPOT 62.70→13.51ms, rate4 82.41→**54.04ms=60ms SLO 통과**, `workspace/engine-port/results/cudagraph_probe/cudagraph_results.md` Probe 1 — Zamba2 단일모델 관측이라 Granite에 직접 이식은 아니나 死因 메커니즘이 cudagraph로 해소 가능함을 시사) ⇒ **"PD 분리 자체는 항상 이득"이 운영점에서 축소되거나 소멸할 가능성**. 검증 실험(2모델×{plain,agnostic}×cudagraph-ON×n≥4) 진행 예정, 결과 없음. 상세 [layertype_dynamic_POSITIVE_2026-08-04.md](layertype_dynamic_POSITIVE_2026-08-04.md) §2.0. ★★★**반증/정정(2026-08-05, claims-auditor 감사, jobs 873944/873945, Zamba2-2.7B·Granite-4.0-h-micro-base, n=5 paired, 사전등록 `results/p1_opint/PREREG.md`, 판정서 `results/p1_opint/P1_OPINT_RESULT_2026-08-05.md`[claims-auditor 감사 반영본])**: 위 "축소되거나 소멸할 가능성"은 낡았다 — 운영점(cudagraph-ON) 대조가 처음 측정됐고, **소멸하지 않았으나 "확인"으로 올라가지도 않는다.** 정본 goodput 술어(게이트 #4: TTFT≤3s ∧ 요청 내부 token-ITL p95≤60ms)로 채점하면 `--enable-pdmux`(agnostic v1)가 fused(plain)를 **두 모델 전부·rate 2–6 전 셀에서** 이긴다(rep 부호 5/5, paired CI 0 배제). **인용 가능한 정량치는 두 arm이 모두 정상상태(큐 성장·런길이 표류 없음)인 셀뿐이다**: Zamba2 rate2 **+40.5%**[CI +0.431,+0.620 req/s], rate3 **+185.8%**; Granite rate3 **+11.7%**, rate4 **+27.0%**. 임계 사다리 40–300ms 전 구간 부호 불변(=metric cliff 아님, ★신규 방법론 게이트 #12: 임계 지시함수 판정은 임계 사다리와 큐-성장 검정으로 견고성을 보여라). **Zamba2 rate4·6과 Granite rate6은 한쪽 이상이 용량 위**여서 **부호만** 인용한다. Granite rate2는 경계(+3.4%, 임계 48ms로 내리면 소멸). ★**술어 의존(중요)**: 사전등록이 채택한 mean-ITL(TPOT) 술어로는 Granite에서 부호가 뒤집힌다(−0.3~−1.6%, CI 0 배제). 그러나 그 술어는 Granite rate3·4에서 **위반 요청이 0건이라 goodput ≡ throughput**(항등, ★방법론 게이트 #6의 새 사례: 판별력 0인 술어에 "차이<3%⇒강등" 규칙을 적용하면 무신호가 강등으로 둔갑한다)이고 남는 차이는 **3% 하한 미만**이다. ⇒ **"Granite에서 P1 전면 강등"은 채택하지 않는다.** 채택하는 것은 "**mean-ITL 술어는 이 regime의 Granite에서 fused의 死因을 잡지 못한다**"뿐이다. 마찬가지로 Zamba2의 사전등록-술어 유의 셀(r4·r6)은 전부 절벽 위라 **그 경로로는 "확인"이 성립하지 않는다** — ⇒ **"Zamba2 P1 운영점 확인(사전등록 지표)" / "Granite P1 전면 강등" 두 문장 모두 정본 등재 금지.** ★**"항상 이득"은 철회한다 — 비용이 실재한다**: pdmux는 정상상태 per-token decode를 **5–45% 늦추고**(중앙 token-ITL 9.67→10.50 / 22.57→29.30ms[Zamba2], 6.80→7.63 / 8.96→12.98ms[Granite]), 저부하 TTFT p50를 체계적으로 악화시키며(Zamba2 r2 0.164→0.230s), Granite raw 처리량은 **−0.3~−2.6%**(CI 0 배제)다. **이득은 꼬리, 비용은 중앙이다.** ⚠️**기전 귀속 미확립(3-플래그 + 미조율 baseline, ★신규 방법론 게이트 #13: arm 대조가 엔진 제약으로 다중 플래그를 강제하면 그것은 묶음 처치다)**: 이 엔진에서 `--enable-pdmux`는 `--chunked-prefill-size -1`·`--disable-overlap-schedule`을 **assert로 강제**하므로(`sglang/srt/server_args.py:6125-6137`) 측정된 처치는 세 플래그 **묶음**이다(`p1op_run.sbatch:55`). 또한 fused arm은 측정 대상 실패 모드에 대해 **미조율**이다 — 관측 stall은 prefill 배치 자체이고(클러스터 지속 166ms@r2→474ms@r4, 케이던스 1.1–1.9/s), `--chunked-prefill-size` 축소와 `--enable-mixed-chunk`(`sglang/srt/managers/scheduler.py:2525-2541`)가 정확히 그 축의 fused-측 레버인데 **둘 다 기본값**이다. ⇒ **"PD 분리(SM 분할) 자체가 원인"은 NOT-YET-SUPPORTED.** 현재 지지되는 것은 "**이 엔진에서 PD-mux를 켜면 기본 설정 fused보다 꼬리 SLO goodput이 좋다**"이다. ⚠️**실현 파티션 미측정**: `PDMUX_TELEMETRY_PATH` 미설정으로 realized `(prefill_sms, decode_sms)` 재집계 불가 ⇒ **파티션·동시성 기전 문장은 정본 금지**(Stage 0 D108 전례), arm 수준 대조만 유효. ⚠️**2026-08-04 반대 증거(위 Probe 1) 정정**: "cudagraph가 fused의 死因(TPOT>60ms)을 제거한다"는 **중앙값 근거**였다. 이번 측정에서 plain rate4는 중앙 TPOT 48.5ms인데도 요청의 **31%**가 mean-ITL SLO를, **86%**가 정본 술어를 위반한다. **중앙값이 SLO 아래 ⇏ 요청 통과.** cudagraph가 제거한 것은 **per-step 벽**이고 남은 것은 **blocking 벽**이다. **위생(허가)**: `boot_ok=1` 24/24, `CORRECTNESS=PASS` 24/24, 서버 로그 traceback/CUDA error 0건, **양 arm cudagraph ON**, piecewise는 **양 모델·양 arm 모두 OFF**(대칭, 교락 아님), 페어링 무결(input_lens 40/40 완전 일치), arm 순서 무작위화 로그 확인. **단 rate 순서는 미무작위화**(`p1op_run.sbatch:148` 고정 2→3→4→6, arm 대조는 paired라 무영향)이고 **n=5는 동일 job·동일 노드 반복**(CI는 노드 내 재현성이지 노드·날짜 간 재현성이 아니다). **등재 금지**: "Zamba2 P1 운영점 확인(사전등록 지표)"/"Granite P1 전면 강등"(위 사유, 양쪽 다); r4·r6 크기 **+41.8%/+398%/+501%/+634%/+174.9%**(런길이 의존 + 캠페인 간 2.2× 불일치, 2026-07-13 probe3 agnostic r4 1.721 vs 3.834); 용량 수치(plain≈4/agnostic≈5, Granite 7.5 vs 6.5)를 n=1 지시값 이상으로(Granite r8 비단조=drain-tail 아티팩트); 파티션·동시성 기전 문장·split 수치(★2026-08-06 Gate 1로 **부분·조건부 해소** — Zamba2 rate{2,3}·decode-busy∧prefill-in-flight 구간의 selector 라벨 `(74,34)`만 인용 가능, rate 4·6·Granite는 여전히 금지, 아래 Gate 1 결과 블록 caveat 전체 필수 동반); "PD 분리 자체" 기전 귀속(Gate 1로도 해소 안 됨 — Gate 2 해소 전); NemotronH/Falcon-H1 확장·ShareGPT/변화-trace 확장(미측정); **2026-07-13 `cudagraph_probe` 수치(위 Probe 1)와 이 캠페인 수치의 직접 대조**(격자 간 이전 금지, 게이트 #11). **다음 gate**(우선순위순): ~~Gate 1 telemetry 재현런~~ → ✅**완료(2026-08-06, job 874478) — 조건부 채택**, 상세는 아래 ★★★★★ Gate 1 블록 참조(부분·조건부 해금, G1-a–d 후속 등재) → Gate 2 4-arm 분해(plain/plain+chunked-1+no-overlap/plain+chunked512/agnostic × rate{2,3}(+4) × n=5, 사전등록 판별: `plain+aux≈agnostic`(3% 이내)면 §1-1 **플래그 아티팩트로 붕괴**, `plain+chunk512≈agnostic`이면 §1-1을 "PD-mux는 head-of-line blocking을 없애는 여러 수단 중 하나"로 재작성 — **논문 신규성 축이 바뀐다**) → Gate 3 나머지 2모델(NemotronH·Falcon-H1) 운영점 대조("4모델 전부" 인용 전제, 안 하면 §1-1은 영구히 2모델 문장) → Gate 4 sustainable-rate n≥4 직접측정(r4/r6 크기 인용 전제, 현재 후순위). **scope(축약 금지)**: {Zamba2-2.7B(triton, ctx4096)·Granite-4.0-h-micro-base(flashinfer, ctx8192), A100 108-SM green-context, **cudagraph-ON**, `--disable-radix-cache --mem-fraction-static 0.82 --max-running-requests 48`, `random-ids` **in2000/out96**(prefill:decode 토큰비 ≈21:1), 정상상태 단일-rate 격자 {2,3,4,6}, 120 프롬프트, n=5 paired(동일 job·동일 노드), agnostic v1(`pdmux_a100_smoke.yml`, sm_group_num 4)}. **NemotronH·Falcon-H1은 운영점 미측정 ⇒ "4모델 전부"는 더 이상 쓸 수 없다.** ShareGPT·변화 trace로 확장 금지(게이트 #2). torch 2.9.1에서 pdmux는 엔진 자체 경고 대상(`server_args.py:6141-6147`). 상세 `workspace/engine-port/results/p1_opint/P1_OPINT_RESULT_2026-08-05.md`, 신규 방법론 게이트 전문은 `../PROJECT_STATUS.md` "방법론 게이트" #6 새 사례·#12·#13, gate 목록 전문은 같은 문서 "다음 실험 gate" ★★★★**통계 방법 층 정정(2026-08-06, claims-auditor Gate 2 설계 감사 2회 + result-analyst 독립 재현, `workspace/engine-port/results/p1_gates/verify/`) — 새 성능 판정 아님, primary를 t-CI로 교체해 재채점한 결과.** `paired_bootstrap_ci`(n=5 percentile bootstrap of mean, BCa·studentization 없음)는 실 coverage **0.840**(100k MC, 명목 95%의 한쪽 오류율 ≈8.0%=명목 3.2배, 원인은 seed·정규성이 아니라 **n=5 그 자체**)이라 소표본 판정에 부적합 — 저장소 안에 `m3_analyze.py`·`tfgate_analyze.py`가 이미 독립으로 t-CI로 전환한 동일 진단이 존재했다(도구 규율 실패, 신규 방법론 게이트 #27). t-CI로 재채점 시 **Granite rate3(+11.7%)는 t-CI [−0.0032,+0.6132]가 0을 포함(p=0.0515)** → 인용 목록에서 제외(**미검증으로 재분류, 철회 아님** — boot CI는 여전히 0 배제), ⇒ **인용 가능 정량치는 4개→3개(Zamba2 rate2 +40.5%/rate3 +185.8%, Granite rate4 +27.0%)로 축소**. "임계 사다리 40–300ms 전 구간 부호 불변(=metric cliff 아님)"도 정정 — 부호가 유지되는 구간은 **T∈[40,113.0)ms뿐**이고 그 위에서 인용 가능 4셀 중 3셀(Zamba2 r2, Granite r3, Granite r4)이 음으로 뒤집힌다(뒤집힘의 정체는 절벽이 아니라 **술어 포화** — 뒤집히는 셀은 T≥150에서 양 arm 위반 0/0, goodput≡throughput; ★§3-24가 REFUTED한 "검정력 0인 임계 사다리"의 재발), 부호가 끝까지 유지되는 유일한 인용 가능 셀은 **Zamba2 r3**. "rep 부호 5/5"도 정정 — **Granite rate2는 실제로 3/5**(per-rep diff −0.0026/**+0.2789**/+0.0321/−0.0035/+0.0164, 효과의 87%가 rep2 한 점), 나머지 7셀은 5/5 유지 확인. n=5 paired 정확 부호뒤집기 순열검정의 두측 p 하한 = **2/32=0.0625**이므로 이 프로젝트의 n=5 paired 셀은 분포무가정으로 p<0.05에 원리적으로 도달 불가(기존 "CI가 0 배제" 서술은 전부 모수 가정 의존이었다는 사실을 명시). ★**P1의 방향 자체는 살아남는다**: agnostic > fused(꼬리에서)는 Zamba2 r2·r3, Granite r4에서 어떤 방법으로도 유효하다 — 강등되는 것은 "전 셀"·"5/5"·"사다리 전 구간"·"Granite r3 수치"뿐, 과잉 강등 아님. **열린 불일치(반영 보류)**: "Granite rate2는 경계(48ms로 내리면 소멸)" 문장은 방향이 반대라는 지적(임계를 내리면 오히려 커짐, T=22→+30.35%)이 있으나 "48ms" 수치의 출처가 `P1_OPINT_RESULT_2026-08-05.md`·`PREREG.md` 어디에도 없어 수치는 유지하고 "출처 미확인·방향 불일치 지적 있음(2026-08-06), 확인 전 인용 주의" 표시만 추가한다(단 Granite rate2는 A-4 경로로 이미 사실상 무신호로 반영됨). 신규 방법론 게이트 **#27**(n≤8 반복에서 `paired_bootstrap_ci`/`unpaired_bootstrap_ci` 구간을 판정에 쓰지 않는다, primary=t-CI) 등재, E1(`s8_frontier/e1_analyze.py:492,1286`) 사전등록 결정 규칙도 같은 undercoverage(net-positive 방향 편향, n=4 coverage 0.798)를 상속하므로 `PROJECT_STATUS.md` "다음 실험 gate" #8에 **제출 선행조건**으로 등재. `CLAIM_EVIDENCE_MATRIX.md`는 이 수치를 인용하는 서술이 없어 갱신 대상 없음(대조 확인 완료). 상세 `PROJECT_STATUS.md` "확정된 결과" 1번·"방법론 게이트" #14·§3 항목27(이 파일), 원자료 `workspace/engine-port/results/p1_gates/verify/`(스크립트·JSON·로그 전체).★★★★★**Gate 1(job 874478, 2026-08-06) 조건부 채택 — claims-auditor 감사, 새 성능 판정 0건, 진단 전용.** Zamba2-2.7B·agnostic v1·cudagraph-ON·rate{2,3}·n=1의 873944 텔레메트리 재현런에서, 엔진이 실제로 구성한 분할표는 `[(108,0),(74,34),(54,54),(0,108)]`이었고(`gate1_srv_874478.log:32`), **decode-busy ∧ prefill-in-flight 구간의 selector 라벨은 시간가중 100.00%가 `(74,34)`**였다(pooled 51.9 s, 78 에피소드, 3,820 스냅샷, 반례 0/40,062). ⚠️ **이 통계는 판별력이 사실상 없다** — `event_loop_pdmux`에서 prefill 어드미션(`multiplexing_mixin.py:1004`)과 `adjust_stream_groups()`(`:1080`) 사이에 telemetry sync가 존재하지 않으므로, "pop A ∧ idx∉{1,2}"는 **관측 가능한 상태가 아니다**(방법론 게이트 #9 다섯 번째 재발, 아래 §3 항목28). 이 문장은 "코드가 그렇게 쓰여 있고 실행도 그대로 했다"이지 "측정으로 알아냈다"가 아니다. ~~**★실질 산출 1 — 이 격자에서 정책은 단일 분할에 고정됐다**: 전 런 `decode_running_batch_size` 최댓값 **23**(문턱 `decode_bs_divisor=36`, `pdmux_a100_smoke.yml:6`) ⇒ **`(54,54)`는 0회 선택**. 인용 가능한 파티션 수치는 **`(74,34)` 하나뿐**이다.~~ → ★★★★★**철회(2026-08-07, G1-b, job 875293, 사전등록 철회 규칙 발화, 새 성능 판정 아님 — 사실 정정)**: 위 무제한(격자 전체) 문장은 거짓이다. rate **2**(max decode_bs 9)·**3**(23)·**4**(18)는 pop A 시간가중 100% `(74,34)`로 문턱 36 미만이라 **인용 가능 셀(Zamba2 r2·r3)에서는 "단일 분할"이 여전히 참**이지만, **rate 6에서만** `(54,54)`가 시간가중 **8.37%**(2.089s/24.974s, max decode_bs 40) 등장해 사전등록 철회 규칙(`frac((54,54))≥0.01`, `gate1b_analyze.py`에 하드코딩)이 발화했다 — 거짓이 된 것은 **격자 전체에 대한 무제한 주장**뿐이다(과잉 철회 금지). ⚠️**rate 4(18)가 rate 3(23)보다 낮다 — 비단조·원인 미해명**(단일 부팅 순차 실행의 순서 효과 또는 큐잉 동역학 가능성, §3 항목30). ⚠️**rate 6 경계 근접**: pop A 에피소드 5개뿐, max decode_bs 40이 문턱 36을 11%만 초과 — 정본은 Zamba2 r4·r6을 이미 "⚠용량 초과" 셀로 이미 표시 중이다. Population C 절대 수치는 Gate 1 감사 caveat 승계(인용 금지, 개수·에피소드만). `(74,34)`·`(54,54)`는 여전히 selector 라벨(하드웨어 SM 수 아님). G1-b 매니페스트는 874478과 바이트 동일 트리가 아님(HOLB 프로브 +2줄, 0줄 변경, 동등성 근거=G3). 상세 위 rev15 헤더, 원자료 `PREREG_G1B_2026-08-07.md`·`gate1b_result_875293.txt`(수정 금지·인용만). **★실질 산출 2 — duty cycle(창 시간 기준)**: prefill/decode 동시 in-flight **27–38%**(추정량 경계 오염으로 구간 제시), decode 단독 `(0,108)` 32.2%, prefill 단독 `(108,0)` 3.5%, **완전 idle 26.3%**. **필수 동반 caveat**: (a) **selector-level·behavioural**이며 하드웨어 SM 부여 프로브(S3) **미실행** — `(74,34)`는 요청값이지 실현된 하드웨어 SM 수가 아니다(Stage 0 D108 전례); (b) **Zamba2 rate{2,3} 한정** — 873944의 `MAIN_RATES`는 {2,3,4,6}이고 rate 4·6은 미측정이며 그 셀은 decode batch가 커져 `(54,54)`로 넘어갈 수 있다; (c) **Granite-4.0-h-micro-base는 전혀 미측정**; (d) n=1, 노드 gpu38(873944는 gpu41); (e) 다른 rate/워크로드/모델로의 이전 금지(게이트 #11). **⇒ §1-1의 "실현 파티션 미측정" 문구는 완전 해제가 아니라 부분·조건부 해제로 대체한다**: decode-busy∧prefill-in-flight 상태의 selector 라벨은 Zamba2 rate{2,3}·agnostic v1·cudagraph-ON에서 `(74,34)`(prefill 74 SM/decode 34 SM 요청값) 하나로 인용 가능하나 여전히 selector-level(S3 미실행)이고, **rate 4·6 셀과 Granite 전체는 여전히 미측정** — **특히 Granite rate4 +27.0%(위 인용 가능 정량치)에는 이 파티션·동시성 문장을 붙이지 않는다**(Gate 1이 그 셀을 커버하지 않음). **"PD 분리 자체" 기전 귀속은 Gate 1로 해금되지 않는다 — 여전히 NOT-YET-SUPPORTED(Gate 2 소관, 불변)**. **인용 금지(신규, 감사자 열거)**: "실현 파티션을 **측정**했다"(항등식) / "prefill 74 SM·decode 34 SM에서 **실행**됐다"(S3 미실행, green-context 반올림 미확인) / rate 4·6·Granite에 대한 파티션 문장 / "**동시성** 기전" 일반(fused arm 대응 계측이 원리적으로 부재 — `plain`은 `--enable-pdmux`가 없어 파티션 텔레메트리가 없다) / "음성대조 C의 2–3% 누출이 gate가 항등식이 아님을 보인다"(누출 10/10이 prefill-완료 전이 sync의 결정적 lag) / "**C = 0.9708/0.9779**" 수치 자체(10개 스냅샷 위, dt 4–10× 과대) / "케이던스 8↔32에서 불변"(추정량 산술 항등식: 누출 개수 ÷4 × dt ×4) / "874465 실패 원인 = bounded queue 포화"(**미확증** — grid 결측 0 + 급정지는 오히려 writer-thread 종료를 시사) / "pooled A = 51.9 s 겹침"(상한, 하한 36.7 s) / "런의 30%가 full-prefill 파티션"(그중 88%가 순수 idle). **커버리지 가드 논거 정정(부기 2, 좁은 형태만 인용)**: `PREREG_GATE1_2026-08-06.md`의 "coverage guard는 단방향으로만 더 엄격해진다"는 감사 결과 **거짓**이다 — `gate1_analyze.py:181-190`의 `severity()` 기준으로 `frac<0.90`이면 `NO_UNLOCK`이 됐을 창이 `coverage<0.98`이면 `UNMEASURABLE`로 **승격(완화)**될 수 있다. 살아남는 것은 좁은 주장뿐: **coverage 실패가 UNLOCK을 만드는 경로는 없다.** 임계 0.98도 근거 없음(관측된 두 점이 80.11%와 ~100%라 어느 값이든 결과 동일) — 이번 판정엔 무작동(`MIN_COVERAGE=0` 재실행 시 출력 동일). **이 논거는 정본 일반 원칙으로 승격하지 않는다**(이 job의 부기 텍스트에 대한 국소 정정일 뿐). **다음 gate 갱신** — Gate 1 완료(조건부 채택) → **G1-a**(≈0.2 GPU-hr, engine-porter): `multiplexing_mixin.py:1005`/`:1080` 사이 관측 전용 sync 1회 추가 ⇒ "prefill in-flight ∧ stale idx"를 관측 가능하게 해 주 조건에 판별력을 부여, 결정량=어드미션-후/adjust-전 구간의 시간 비율+절대 ms → **G1-b**: ✅**완료(2026-08-07, job 875293) — 철회 규칙 발화**(rate 6에서 `(54,54)` 시간가중 8.37% 관측, rate 4는 NO_EVIDENCE), 상세는 위 rev15 헤더·본 항목 "★실질 산출 1" 철회 블록 참조 → **G1-c**: Granite(873945 복제) → **G1-d**: S3 하드웨어 프로브(`%smid` 샘플링/CUPTI) → **하네스**: `gate1_analyze.py`에 grid-completeness 검정 상시화(`trace_forced==False ⟹ si==1 ∨ si%TRACE_EVERY==0`, 결측 수 출력, 이번 "유실 없음"의 실제 근거는 coverage가 아니라 이 검정이었음) + engine-porter 이관(`telemetry.py`의 `writer_error` 로깅, SIGKILL 경로에서 미호출되는 `close()`) → (기존, 불변) Gate 2 4-arm 분해 → Gate 3 나머지 2모델 → Gate 4 sustainable-rate. **위생 확인(전부 통과)**: 매니페스트 13파일 SHA-256이 873944와 바이트 일치, 2026-08-05 이후 변경된 `.py`가 정확히 그 13개(커버 밖 드리프트 없음), `architecture` 필드 40,062/40,062="legacy", `stream_index↔sms` 불일치 0, cudagraph ON, `CORRECTNESS=PASS`. ⚠️**노드 불일치**: Gate 1=gpu38, 873944=gpu41. 상세 `PROJECT_STATUS.md` "다음 실험 gate" #10(Gate 1 항목), 원자료 `workspace/engine-port/results/p1_gates/gate1/`(`gate1_result_874478.txt`·`PREREG_GATE1_2026-08-06.md`·`gate1_analyze.py`·`gate1_telemetry_874478.jsonl`·` ★★★★★**(2026-08-07 실행, 2026-08-09 정본 반영 복구, doc-steward) Gate 2 rev4 본 캠페인(jobs 875344/875346, 5.86 GPU-hr) — 정본 누락 복구.** 새 성능 판정 아님 — 이미 완료된 결과의 정본 반영 누락을 메운다. 원자료 `workspace/engine-port/results/p1_gates/gate2/g2_report_zamba2-27b_875344.json`·`g2_report_granite-40-h-micro-base_875346.json`(rev4 사전등록 `PREREG_GATE2_2026-08-06.md` §14 적용, arm A1=`plain`/A2=`plainaux`/A3=`chunk512`/A4=`agnostic`). **Primary(§4.1/5.2, A3 vs A4 TOST)**: 5셀(Zamba2 r2·r3, Granite r3·r4·r6) 전부 `Rprime4`(A4 유의 우세, 등가 미발화, `p_tost=1.0`) — **chunk512는 pdmux를 대체하지 못한다.** F-E(정상상태) 스크린 통과로 크기 인용 가능한 셀은 **Zamba2 r2·Granite r3 둘뿐**(나머지 3셀은 chunk512측 F-E 발화, 부호만). **Secondary — R1′/R2′(§5.2 표 마지막 행, A2=`plainaux` vs A4=`agnostic`)**: rev4 §1.1의 등식 A2=A4−pdmux(realized server_args 차이는 `enable_pdmux`·`pdmux_config_path` 두 필드뿐)이므로 이 비교는 §1-1의 "3-플래그 묶음 처치" 교락 중 pdmux 고유분을 분리한다. 메인 세션이 2026-08-09 원자료(`per_arm_x60`)에서 독립 재현(paired-t, n=10; 셀: mean(A2−A4, X_60) [95% CI], 부호, F-E(plainaux/agnostic)) — Zamba2 r2: **+0.8341** [+0.7809,+0.8872], 10/10, clear; Zamba2 r3: **+0.9297** [+0.8968,+0.9626], 10/10, clear; Granite r3: **+0.8550** [+0.8161,+0.8939], 10/10, clear; Granite r4: **+0.9442** [+0.9201,+0.9682], 10/10, clear; Granite r6: **+0.9600** [+0.9292,+0.9908], 10/10, **agnostic 측 F-E flagged — 부호만**. sign-flip 순열 p = 2/1024 = **0.001953125**(n=10 분포무가정 두측 하한, 5셀 공통). **묶음 기여 분해**: A2는 A1(`plain`)과 사실상 같고(5셀 |A1−A2|≤0.023) 그 부호는 **A4에 불리한 핸디캡 방향**이다 ⇒ `--chunked-prefill-size -1`·`--disable-overlap-schedule` 두 플래그만으로는 pdmux 이득이 재현되지 않는다 — §1-1 "3-플래그 묶음 처치" 교락 중 이 두 플래그는 **원인에서 배제**된다. ★★**provenance(인용 시 필수 동반)**: 이 A2-vs-A4 비교는 **사전등록 분석기 `g2_analyze.py`가 계산하지 않는다** — `tost_equivalence` 호출은 코드 전체에서 1회뿐(`:543`)이고 그 입력은 A3-vs-A4(`chunk512`-`agnostic`, `:538-539`)뿐이다. A2는 `:734`에 서술 문장으로만 등장한다("A1/A2/A4 data … remain valid regardless"). 즉 **arm·n·raw 데이터는 사전등록(rev4)이지만, 이 비교 자체는 저장된 primary 산출물(`per_arm_x60`) 위에서의 사후 계산**이다(코드 확인 2026-08-09). §3 항목34로 등재(항목33 "사후 지정 셀 이동"의 형제 사례). ★★**천장 포화(해석 제한)**: fused arm(plain/plainaux/chunk512) X_60 평균이 0.72–0.99(천장 근접), agnostic이 0.00–0.12(바닥 근접)이다 ⇒ **부호는 견고하나 크기는 포화 구간의 값**이며, E-A 블록이 이미 건 "기전 해석 금지"가 여기도 적용된다. ★★**해금 범위의 상한(overclaim 금지)**: 해소되는 것은 §1-1의 **"3-플래그 묶음 처치" 교락뿐**이다. 귀속의 상한은 **"pdmux 서브시스템 전체"**(green-context SM 분할 + 전용 이벤트 루프 + split-prefill)이고, **"SM 분할 자체"는 여전히 미분리**다 — `--enable-pdmux`가 A2의 `event_loop_normal()`을 대체해 pdmux 전용 이벤트 루프를 통째로 켜기 때문이다(§1.1). **"PD 분리 자체가 원인"으로 승격 금지** — §1-1의 NOT-YET-SUPPORTED 등급은 불변, Gate 2 본 질문은 한 눈금도 전진하지 않는다. ★**Granite r6**: agnostic 측 F-E flagged=true라 부호만(10/10, +), 크기 인용 금지. ★**감사 provenance(등급 정정)**: 세션 핸드오프(`handoff-report/session_handoff_2026-08-09.md` §2.5)는 이 R1′/R2′ 결과를 "[감사 완료]"로 표기했으나, 2026-08-09 저장소 전체 검색(`gate2/` 디렉터리·전체 `*.md`)으로 이 결과를 다루는 claims-auditor 감사 아티팩트가 확인되지 않는다. ⇒ **"감사 완료"로 인용하지 않는다.** 현재 등급 = **메인 세션 원자료 독립 재현 확인(2026-08-09)**, claims-auditor 감사 기록 위치 미확인.(같은 캠페인의 primary R3′/R4′와 §11-3 감사 면제 사실은 §14.1 addendum에 이미 기록돼 있고 그 부분은 인용 가능 — 감사 미확인은 이 R1′/R2′ 비교 자체에 한정된다.) 상세는 위 rev17 헤더, `PROJECT_STATUS.md` "확정된 결과" 1번·"방법론 게이트" #20·`CLAIM_EVIDENCE_MATRIX.md`(대조 확인, 갱신 대상 없음) 동반 갱신 참조. ★★★★★★**E-A(mixed-chunk 레버, jobs 875654/875657/875661, 2026-08-08~09) — claims-auditor 감사 완료, 판정 조건부(문구 강한 제한). 새 성능 판정 아님 — fused-측 조율 가능성에 대한 진단이다.** 원자료 `workspace/engine-port/results/p1_gates/gate2/`(`PREREG_G2EA_2026-08-07.md`·`g2ea_report_*.json`·`g2earun_8756{57,61}.out`·`g2eaprobe_875654*`, 수정 금지·인용만). **A. 레버 실현**: Stage 1(job 875654)에서 `--enable-mixed-chunk`가 Zamba2-2.7B·Granite-4.0-h-micro-base × {8192, 512} 4콤보 전부에서 realized로 확인됐다(세 독립 증인: `/server_info` top, `internal_states[0]`, HOLB `holb_open`). 기능 스모크 8/8 정상. ⚠️프로브 스크립트의 `PROBE_RESULT` 필드는 stderr 오염 버그(`g2ea_mixed_chunk_probe.sbatch:205`의 `2>&1`)로 `AVAILABLE_BUT_BURST_ERROR`를 잘못 출력했다 — 판정은 원 burst JSON 8건 직접 재파싱에 근거한다. **B. 주 결과(부호)**: E-A(jobs 875657/875661, n=10 paired, cudagraph-ON, arm 순서 무작위, 노드 gpu42/gpu40)에서 사전등록 primary 두 비교 × 4셀 = **8건 전부 TOST 등가가 발화하지 않았고, 방향은 전부 agnostic 우세**(부호 10/10, 분포무가정 정확 p = 2/1024 = 0.00195). ⇒ 사전등록 규칙상 **"조율된 fused가 pdmux를 대체한다"는 발화하지 않았고, 논문 신규성 축 전환은 없다.** ⚠️ 8건 중 7건은 F-E 발화(비정상상태) ⇒ **부호만 인용 가능.** ⚠️ cmp2(A3m)의 Granite 두 셀은 G-2 실패로 상속된 §2.1 폐기 규칙 하 **무효**다. **C. 유일한 인용 가능 정량치**: Granite-4.0-h-micro-base rate 3, cmp1: **X_60(plainmix) − X_60(agnostic) = +0.855 [95% CI +0.814, +0.896]**, n=10 paired, 부호 10/10. 양 arm 정상상태(duration 41.0s vs ideal 40.0, 실현 throughput 2.938 vs 제공 3, F-E 1.026/0.973, 런길이 불변 검정 N=60→120에서 TTFT p95 338→406 ms = 1.20×). ⚠️ **이 셀이 지정 셀이 된 것은 선행 캠페인 진단을 본 뒤의 사후 이동(r4→r3)이며, r4를 유지했다면 이 캠페인의 인용 가능 셀은 0개였다** (`PREREG_G2EA_2026-08-07.md` §3 자기인지 위험 #2). ⚠️ **같은 셀에서 TTFT 항은 통과한다** — plainmix가 agnostic보다 TTFT p95가 **19.2% 좋다**(CI [−0.256, −0.129]). 탈락은 ITL 항에서만 일어난다. **"TTFT 비열등 0건"은 거짓**이다(1/8). **D. 레버 격리**(C와 반드시 병기 — 없으면 C가 오해를 부른다): 위 +0.855 중 mixed-chunk 기여분은 **+0.010 [+0.0045, +0.0155] = 1.2%**뿐이다. 나머지 98.8%는 untuned fused와 pdmux 사이의 기존 격차(X_60(plain) − X_60(agnostic) = +0.845 [+0.802, +0.888])이며 rev4/§1-1이 이미 확립한 양이다. 4셀 전체에서 mixed-chunk 기여분은 **1.2–10.6%**. ⇒ **E-A의 primary는 mixed-chunk 레버를 격리하지 않는다.** ⚠️ X_60은 fused arm 전부 0.85–0.99로 천장 근접(정본 §1-13 ceiling-censoring 재발) ⇒ mixed-chunk 고유효과 크기는 **하향편향**. ⚠️ 그 크기(+0.010)는 HOLB 관측자 효과 잔차(rev4 §14.3: itl_p95 arm별 −9.9%~+105%)와 **같은 자릿수**이므로 **기전 해석 금지.** **E. 기전(신규, 서빙 직접 측정)**: mixed-chunk가 켜지면 prefill과 함께 스케줄된 decode가 `ForwardMode.MIXED` extend 경로로 재라우팅된다(`schedule_batch.py:1874`; `MIXED`는 `is_cuda_graph()`에서 제외 — `forward_batch_info.py:166-173`). HOLB 실측: MIXED step 지속시간이 병합 decode 요청 수에 **선형 증가** — Zamba2 252→1,844 ms(bs 0→48, ≈33 ms/요청), Granite 129→247 ms(bs 0→24, ≈4.2 ms/요청). 순수 DECODE step은 각각 9.4 / 6.8 ms. MIXED step 하나가 running 요청 전부를 1 토큰씩만 전진시키므로 요청별 ITL이 MIXED step time과 같아지고, backlog가 자라 `max_running_requests=48`·`mamba usage 1.00`을 포화시켜 admission을 막고 TTFT가 발산한다. ⇒ **정본 §1이 동적 제어에 대해 확립한 얽힘 死因(decode 굶김→ITL↑→batch 정체→admission 차단→TTFT 폭발)의 새 트리거이지 새 기전이 아니다.** **F. cudagraph 가설 반증 + 정본 부수 정정**: "mixed 배치가 decode CUDA graph를 못 써서 decode가 eager로 떨어진다"는 **설명으로서 반증**: (i) plainmix의 남은 순수 DECODE step은 `cuda graph: True`이고 중앙 지속시간이 plain과 동일(9.5 vs 9.4 ms Zamba2, 6.9 vs 6.8 Granite); (ii) 페널티가 배치 크기 비례라 그래프 launch 오버헤드보다 두 자릿수 큼; (iii) piecewise CUDA graph는 두 모델 10개 boot 전부 런타임 비활성이라 arm 간 비대칭 아님. ⇒ 실제로 일어난 것은 "decode의 eager 강등"이 아니라 **"decode 작업의 extend 커널 경로 재라우팅"**이다. ★ **부수 정정**: rev4 §1.1의 "A3는 `piecewise_cuda_graph_max_tokens` 8192→512도 함께 바꾸는 ≥2-기전 묶음" 캐비어트는 **이 두 모델에선 런타임 실측으로 무효**다(전 arm piecewise OFF). 해당 인용 금지를 해제하되 근거를 명시하라. **G. G-2(재현성 — correctness 아님)**: `--chunked-prefill-size 512`를 건 Granite-4.0-h-micro-base(flashinfer, `enable_deterministic_inference=False`)는 16-동시 greedy 요청에서 **비트단위 재현성이 깨진다** — 16 중 2 요청이 퇴화 반복 루프의 반복 단위 한 토큰(62↔322)에서 결정론적으로 갈라진다(불일치 위치 = 정확히 주기 3의 31개 지점, 출력 길이 96 동일, 두 변종은 arm 간 바이트 동일). 3회 독립 재현(875346·875611·875661), 음성대조 `plain`·`plainaux`·`plainmix`·`agnostic` 전부 통과. **이것은 correctness 결함이 아니라 reproducibility 결함이며, 엔진은 배치 형태 간 비트 재현성을 약속하지 않는다.** mamba state 이월과는 **정합하지 않는다**(그 경우 인덱스 0부터 고정 불일치 + 전혀 다른 연속이 예상되나, 관측은 반복 구조 완전 보존). **H. 스코프(축약 금지)**: {Zamba2-2.7B(triton, ctx4096) · Granite-4.0-h-micro-base(flashinfer, ctx8192), A100 108-SM, **cudagraph-ON / piecewise-OFF**, `--disable-radix-cache --mem-fraction-static 0.82 --max-running-requests 48`, `random-ids` in2000/out96, 정상상태 단일-rate {Zamba2 2,3 / Granite 3,4}, 120 프롬프트, n=10 paired(동일 job·동일 노드), agnostic v1, HOLB 프로브 전 arm ON}. **상위 사전등록 rev4의 §11-3(3차 감사)은 면제됐다 — "감사를 통과한 설계"가 아니다**(§14.1 자백). **T2(가장 무거움) — "fused 조율 공간 소진" 주장 금지**: 정본 `CONSENSUS.md:368`의 *"…가 정확히 그 축의 fused-측 레버인데 둘 다 기본값이다"*는 **대표 레버 지목이지 완전성 주장이 아니다.** 같은 기전 축의 **미시험 노브 최소 6개**(realized 값 실측): `--prefill-max-requests`(현재 `None`=무제한, `server_args.py:3959`) · `--num-continuous-decode-steps`(현재 1, `:5559`) · `--chunked-prefill-size` 2048/4096 · `--max-running-requests`(현재 48 = 실제 포화점) · `--schedule-conservativeness`(1.0, `:4029`) · `--mamba-scheduler-strategy`(`no_buffer`, `:5082`). (+`--enable-prefill-delayer`는 적용성 미확인.) ⇒ 쓸 수 있는 최대치: *"정본이 지목한 두 레버는 시험됐고 둘 다 격차를 닫지 못했다 — 다만 fused 측 조율 공간이 소진됐다는 뜻은 아니다."* **Gate 2 본 질문 전진 없음**: A2(`plainaux`)를 뺐으므로 **"PD 분리(SM 분할) 자체" 귀속은 한 눈금도 전진하지 않았고 §1-1의 NOT-YET-SUPPORTED는 불변**이다. `PREREG_G2EA_2026-08-07.md:66-67`이 A2와의 교차-job 비교를 명시 금지했으므로 875344/875346의 A2로 메우는 것도 금지. **인용 금지 목록(13건, 정본에 그대로 등재)**: 1. "등부하에서 mixed-chunk는 지연을 N배 악화시킨다"(78×·6.2×·24–35× **전부**) — 런길이 의존 실측(N 60→120에서 2.3–6.2× 변동). 2. throughput `1.938→1.303`·`0.530`·`1.070`, TTFT p95 `28,359`·`154,601`·`67,298 ms` — 무플래그 `secondary` 블록 출신, 불안정 큐 과도상태. 3. F-E 발화 셀의 X_60 크기(cmp1 Zamba2 r2·r3, Granite r4; cmp2 전 셀). 4. **"TTFT 비열등 0건"** — 거짓(1/8, C 참조). 5. "fused 조율 레버 소진" / "T1이 닫혔다" / "조율된 fused는 pdmux를 대체할 수 없다"(일반형). 6. "두 레버 모두 실패"를 **E-A 캠페인 내부 결과로** 제시 — A3-vs-A4는 E-A 사전등록 비교가 아니다(사후 계산이며 F-E clear ∧ G-2 PASS를 동시 만족하는 셀은 **Zamba2 r2 하나**: +0.879 [+0.859, +0.899]). 7. "chunked-prefill 조율이 …" 문구 — 단 F의 정정 반영 후에는 금지 근거가 약해지므로 F를 먼저 반영할 것. 8. "chunked_prefill_size=512가 Granite에서 **잘못된 출력**을 낸다" — reproducibility ≠ correctness. 9. cmp2 Granite r3·r4의 어떤 판정도 — 상속 §2.1 폐기 규칙 미적용. 10. "mixed-chunk 붕괴는 Zamba2라는 모델의 성질" — 모델과 attention backend 동시 변경(confound #10). ⚠️ 선례 `[[scale-8b-sm-sensitivity]]`의 "hybrid 급락=Zamba2 성질" 강등과 동형. 11. "PD 분리 자체" / "pdmux 필요성 확립" / "Gate 2 전진". 12. "감사를 통과한 사전등록". 13. MIXED 한계비용 8× 비대칭(33 vs 4.2 ms)의 **원인 귀속** — 미해명으로 등재. **신규 방법론 항목(§3 항목31–33 등재)**: (31) "게이트를 지표에 걸 때는 primary뿐 아니라 보고되는 모든 블록에 걸어라." F-E 집행 수정이 primary 두 비교에만 적용되고 `secondary` 블록(throughput/goodput/ttft_p95/itl_p95)은 무방비였다 — 그리고 실제로 **그 무방비 경로에서 인용이 일어났다**(`g2ea_analyze.py:662-680`). (32) "과부하 arm과 정상 arm을 같은 제공 rate에서 비교한 수치는 시스템 상수가 아니다." 런길이 의존 실측: 프롬프트 60→120에서 지연 2.3–6.2× 변동, 정상 셀만 1.20×. ⇒ 비정상상태 셀은 부호만, 크기는 지속가능 rate 대조(E-C) 후에만. (33) "사후 지정 셀 이동은 부호를 안 바꿔도 인용 가능성을 만들 수 있다." r4→r3 이동이 없었다면 이 캠페인의 인용 가능 셀은 0개였다. **후속 게이트 등재(우선순위순, `PROJECT_STATUS.md` "다음 실험 gate")**: T4-1 deterministic-inference 재확인(<0.5 GPU-hr, 폐기 규칙 해제) → T3-1/T3-2 MIXED 배치 조성 계측·micro 스윕(<1 GPU-hr each, 기전 확정) → E-C 등지속가능-rate 대조(≈4 GPU-hr, T1 정식 종결) → E-D 미시험 fused 레버 스윕(≈8 GPU-hr, T2 종결) → T3-3 backend 교차(≈2, 8× 비대칭 귀속) → T4-2 비퇴화 프롬프트 G-2(≈1). ⚠️**상위 사전등록 `PREREG_GATE2_2026-08-06.md`의 §14 addendum은 커밋 `c47fad0`으로 반영돼 워킹트리 클린이다(2026-08-09 확인) — "워킹트리 미커밋" 기록은 stale, 정정.** §11-3(3차) 감사 면제 상태(§14.1)는 별개로 불변이라 "감사 통과 설계"로는 여전히 인용 금지. `results/p1_gates/` 이하 raw dump 수정 금지(인용만). 상세 위 rev17 헤더·rev16 헤더, `PROJECT_STATUS.md` "확정된 결과" 1번·"방법론 게이트" #17–20·"다음 실험 gate" #10(E-A 항목), 원자료 위 경로. ★★★**(2026-08-09, 두 번째 정본 반영 건 — A/B/C 세 갈래, 전부 새 성능 판정 아님)** **A. HOLB G5 재채점**(result-analyst, jobs 874601/874602/874632/874633/874635, 원자료 `workspace/engine-port/results/p1_gates/gate2/g2holb_g5_tost_rescore_2026-08-09.json` 72셀 전량, 3 job × 4 arm × 6 응답변수, n=5 paired): **판정 G5 = 미결정(UNDETERMINED), 저장된 `G5=False`를 대체.** 3% 초과 통계적 지지 셀 **0/72**(미보정 최소 p_exceed=0.138, job별 Holm 후 최소 조정 p=1.000 3 job 전부), 등가 입증 **29/72**, 검정력 부족 **43/72** — G5는 관측자 효과가 3%를 넘음을 입증하지 못했고 3% 이내임도 입증하지 못했다. 저장 `G5=False`는 귀무-채택형 연언(`|효과|<3% ∧ CI∋0`)의 실패이지 프로브 유해성의 입증이 아니다. **구 규칙이 노이즈를 보상했다**(874635 agnostic ttft_p95 −0.77%±29.14%, 95% CI [−36.95,+35.41] → 구 규칙 PASS, 메인 세션 독립 재확인) — 역방향(구 규칙 FAIL·±3% 등가 실제 입증)도 **4셀**(예: 874633 plain itl_p95 −1.15%±0.72%, p_TOST=0.0023). **설계 층**: n=5·δ=3%·α=0.05/side에서 TOST 발화 산술 천장 SD<3.147%인데 72셀 중 **35셀(49%)**이 그 위, 80% 검정력 필요 n 중앙값 **9**(request_throughput 4 / itl_p50·mean_e2e_ms 7 / itl_p95 22 / ttft_p50 29 / **ttft_p95 89**, ⚠️SD가 df=4 추정이라 필요 n은 자릿수 수준 의미만). `PREREG_GATE2` §14.2의 "프로브가 무해함이 입증됐다고 쓰지 마라"는 **해제되지 않는다** — 동시에 반대 오독("3% 넘게 유해함이 입증됐다")도 근거 없음. §14.3의 귀인("주로 점추정이 3%를 넘는 조합이 실재하기 때문")은 **부분적으로만 참**(그런 셀은 실재하나 그중 하나도 초과가 지지되지 않는다) — 두 문장 병기 필수. **잔여 교락**: 874602 agnostic은 5/5 rep 전부 `order='off on'`(무작위화 불균형) ⇒ 그 arm의 등가 판정은 조건부. 등가 판정 29건은 전부 paired-t **정규 가정** 위(n=5 분포무가정 두측 p 하한 2/32=0.0625). **B. job 874601의 `G3=False` 라벨 정정**(메인 세션 원자료 직접 확인): `g2holb_report_zamba2_874601.json`은 4 arm 전부 `result:"FAIL"`이나 `sha_off:null, sha_on:null`·`self_repro_off/on:true` — 출력이 갈라진 게 아니라 sha 추출이 응답 스키마를 못 읽은 것(`KeyError:'text'`)이고 Phase B가 실행되지 않았다. 재실행 874633/874635는 `method_off/on:"output_ids"`로 sha 양측 일치 → **PASS**(위 A절 근거로 이미 사용됨). 874632는 Phase A 중 SLURM CANCELLED(데이터 없음). ⇒ "874601 G3 실패" 라벨은 **하네스 실패**로 정정. **이 프로젝트 서명 오류(측정 실패를 게이트 실패로 라벨링)의 여섯 번째 재발**(핸드오프 2026-08-09 §1이 다섯 번 — 텔레메트리 드롭 카운터 자기검열/UNSCOREABLE을 강등으로 읽음/HTTP 400을 G3 FAIL로/G5의 귀무-채택형 기준(위 A절과 같은 사건 계열)/프로브의 stderr 오염). **canon에 이 패턴을 다루는 기존 번호가 없어(전수 검색 확인) §3 항목35로 신규 등재**(중복 신설 아님 — 이후 재발은 이 항목 카운트만 갱신). **C. E-A 커밋 산출물의 scipy 부재 폴백**(engine-porter 발견 + 메인 세션 노출범위 실측, 범위 한정): `g2ea_report_*.json`은 scipy 없는 인터프리터에서 생성돼 `t_cdf()`가 Student-t가 아니라 정규 CDF로 조용히 폴백(저장 `p_tost`가 정확히 1.0인 이유, `g2ea_analyze.py:84-87,156-159`). **메인 세션이 노출 범위 직접 측정**: `raw_ci` 폭 역산 임계값이 **8개 비교 전부**(2 job×2 rate×2 cmp, coordinator 인용 4개 포함, 전부 df=9) implied_t=**2.2621…=t(.975,df=9)**(하드코드 표값과 일치, 1.96 아님) — **CI(raw_ci)는 오염되지 않았다.** 노출은 (i) `p_tost` 자체(이 캠페인은 관측치가 0.05 경계에서 멀어 판정 무영향)와 (ii) `t_ppf`의 df>10·표에 없는 p에 한정(`t_cdf`는 표가 아예 없어 scipy 부재 시 모든 df에서 근사값이라는 점은 (i)에 포함되되 원인 층이 다름을 기록). ⇒ **"정본 수치 정정"이 아니라 도구 규율 항목**(게이트 #14 계열, §3 항목36 신설). **과장 금지 — 이 캠페인에서 오염된 인용 수치는 없다.** 상세 위 rev18 헤더·§3 항목35·36, `PROJECT_STATUS.md` "확정된 결과" 1번·"방법론 게이트" #21·#22 동반 갱신, 원자료 `workspace/engine-port/results/p1_gates/gate2/`(`g2holb_g5_tost_rescore_2026-08-09.json`·`g2holb_g5_tost_rescore.py`·`g2holb_g5_tost_summary.py`·`g2holb_report_zamba2_874601.json`, 수정 금지·인용만). ★★★**(2026-08-10, 세 번째 정본 반영 건 — A/B/C/D, 전부 새 성능 판정 아님)** **A. 방법론 게이트 #21의 일곱 번째 재발**(job 876699, T4-1): `--time=1:00:00` TIMEOUT, 원인은 사이징이 아니라 단일 호출 스톨 — `ON+chunk512` 첫 multi-chunk generate(2552 토큰, 5-chunk)가 ~52분 무응답(스케줄러 로그 0줄, CUDA 에러·watchdog 미발화; 서버 로그 마지막 활동 23:34:42, SLURM kill 00:26:19, 메인 세션이 두 시각 직접 대조), 같은 job `OFF+chunk512`는 동일 프롬프트 ~1초 완료. **핵심**: 옛 `g2det_analyze.py`가 그 아티팩트에 `REFUTED`를 반환하고 있었다 — ON chunk512 `n_total=0`(미실행)인데 `on_clean = n_total > 0 and ...`이 False로 떨어져 REFUTED로 통과, **52분 멈춤이 실질적 음성 결과로 발표될 뻔했다.** 재발 1–6과 다른 점: 1–6은 라벨·해석 오류를 사람이 잡았으나 이번은 **분석 코드가 거짓 음성을 산출**했고 막은 것은 experiment-runner의 채점 거부(`INCOMPLETE`)였다 — 도구가 사람보다 관대했다. 사후 완화 아님(사전등록 REFUTED 조건은 "ON이 어느 rep에서든 self-mismatch≥1"인데 rep 0개는 그 관측 자체가 없다 — 옛 코드는 버그였다). 수정(2026-08-10) 후 `NO VERDICT (MEASUREMENT ABSENT)` 반환, CONFIRMED/REFUTED 분기는 5-케이스 매트릭스로 불변. ⇒ **§3 항목35(가트#21)의 재발 카운트 6→7 갱신**(아래 개정판, 신규 항목 아님). **B. 공유 하네스의 무한 대기(도구 규율, 신규 §3 항목37/게이트#23)**: `g2_holb_phaseA_lib.sh`의 `greedy_call`이 `--max-time` 없는 raw curl이었고 이 디렉터리의 모든 캠페인(g2ctrl/g2ea/g2holb/g2det)이 이 경로를 쓴다 — 소비자 10개 전수 확인 결과 4개가 타임아웃을 `FAIL`/`SMOKE_FAIL` 계열로 채점 중이었다(A의 `g2det_analyze.py` 포함). 수정: `--connect-timeout 10 --max-time 180`(env 재정의 가능), `STATUS=TIMEOUT`을 `STATUS=ERROR`와 구별, SHA 미방출로 mismatch 채점 구조적 불가, 사이드카 `.status.json`. 180s는 측정 근거(이 디렉터리 아카이브 응답 n=64의 서버측 `e2e_latency` median 0.978s/p90 2.534s/max 6.605s, 최악의 27배·중앙값의 184배 — 메인 세션이 코드 주석 근거 문단 직접 대조) — 정상 경로는 아카이브 64개+합성 실패 9종 재생 73/73 byte-identical 검증(engine-porter 보고, 메인 세션 미재현). A와 뿌리 사건은 같으나(job 876699) 레슨은 다르다: A="분석 코드가 무데이터에 REFUTED", B="공유 인프라가 무경계 대기를 10개 소비자에 전파". **C. Gate 2-S 3라운드 설계 감사 — §1-1 귀속 스코프 주석(대체 아님)**: `PREREG_GATE2S_2026-08-09.md`가 claims-auditor 3회 전부 NO-GO(rev1→2 통계층/rev2→3 게이트 인식론/rev3→4 귀무 채택형, 메인 세션 파일 직접 확인) — 3연속이 같은 자리(§5.5 앵커 발화 조건)에서 죽었고 3차 감사가 근본 원인을 문서 내부 모순으로 특정(§0.1 "두 pdmux arm 사이 등가 마진에는 외부 앵커가 없다" vs §5.5가 정확히 그 등가를 앵커 조건으로 요구). 정량(§0.0.B 원문 대조): rev3 §5.3 암묵 등가 마진 = **0.715 σ_D**, primary MDE = **0.995 σ_D** ⇒ 대리 허용오차가 검출한계의 **0.72배**; 표준 처방(TOST+마진)은 §0.1 위반, 마진을 MDE 1/4로 낮추려면 **n≈64**(현재 10). ⇒ **정본 문구(고정)**: **"P1 이득 중 'SM 분할 자체의 몫'을 두 pdmux arm 사이의 성능-층 등가검정으로 귀속하는 경로는, 이 프로젝트의 예산 범위에서 닫히지 않는다(n≈64 필요, 현행 설계 n=10). 원리적 불가능이 아니라 이 경로·이 예산에서의 불가능이다."** ⚠️**과장 금지**(귀속이 원리적으로 불가능하다로 쓰지 않는다) — **이 문장은 위 §1-1의 NOT-YET-SUPPORTED를 대체하지 않고 스코프 주석으로 덧붙는다**("아직 안 됐다"≠"이 경로로는 안 된다"), 등급어는 미실행 사전등록의 설계 감사에 맞춘다(측정 결과 아님). **감사가 깨뜨리려 시도했으나 실패한 것(견고, §0.0.A 직접 확인)**: arm 구조·Δ_split estimand 내부 타당성, T·C 드레인 대칭, 9-셀 판정표의 저자 불리 셀 실재, 부팅 단위 프로브 근거(n_eff=10), 예산 산술. 3차 감사 원문: **"이 캠페인이 죽어야 할 이유는 측정 층이 아니라 귀속 층에만 있다."** rev4는 §5.5 앵커 조건 자체를 제거해 이 하위목표에서 후퇴(패치 0줄, env 재구성만) — **Gate 2-S가 폐기된 것은 아니다.** **D. 코드 사실 문구 하향(메인 세션 자기정정)**: 이 세션 중 "`(0,108)` idx에서는 두 역할 모두 전체 108 SM에 접근한다"고 서술한 바 있다. 검증된 것은 "green context가 아니라 평범한 `torch.cuda.Stream` 쌍"까지뿐(`pdmux_context.py:124-138`, 메인 세션 직접 확인 — idx 0·idx `len-1`은 `torch.cuda.Stream(gpu_id)`, 중간 division만 `create_greenctx_stream_by_value`) — green ctx 생성이 primary context SM을 깎는지는 미측정 물리 명제(격리 수단 P-b 프로브는 공선성으로 삭제됨, engine-porter 보고). **전수 검색(2026-08-10) 결과 이 문구는 canon·파생 문서 어디에도 없다** — 정정 대상 없음, 향후 인용 규칙만 등재: (0,108) idx는 "명시적 분할이 적용되지 않는다(잔여 차감 여부는 미측정)"로만 쓴다. 상세 §3 항목35(개정)·37, `PROJECT_STATUS.md` "확정된 결과" 1번·"방법론 게이트" #21(개정)·#23, 원자료 `workspace/engine-port/results/p1_gates/gate2/`(`g2det_876699.out`·`g2det_876699.err`·`g2det_analyze.py`·`g2_holb_phaseA_lib.sh`·`PREREG_GATE2S_2026-08-09.md`, 수정 금지·인용만). ★★★★★★★**(2026-08-11) Gate 2-S 첫 유효 결과(jobs 877756/877757, 6.35 GPU-hr) — claims-auditor 적대 감사 "조건부 등재 가(可)". 별도 트랙 산출, §1-1 대체 아님, 위 C(2026-08-10 귀속 스코프 주석) 뒤에 배치.** 둘 다 exit 0/`MEASURED_AND_SCORED`, `design_conformance.conformant=True`(전 셀 n=10, 5 arm), 19/19 블록. 1차 실행(877107/877109, 6.40 GPU-hr)은 하네스 결함 6건으로 primary 0개 — 설계는 무결, 실패는 하네스 층뿐. 사전등록은 claims-auditor 설계 감사 4회(NO-GO 3 → GO-with-changes). **정본 문장(감사자 확정 범위, 그대로)**: 이 엔진·이 격자(Zamba2-2.7B r{2,3} triton / Granite-4.0-h-micro-base r{3,4} flashinfer, in2000/out96, A100 108 SM, cudagraph ON, n=10 paired)에서, pdmux 서브시스템·이벤트 루프·split-prefill·역할별 backend를 고정한 채 `PDMUX_R2_FIXED_DSM`만 108→34로 바꿔 SM을 `(74,34)`로 분할하면 **요청-내부 ITL p95 평균(α)이 4셀 전부 감소**(Δ=C−T′=**+13.95/+24.88/+16.04/+19.11 ms**, 4셀 Holm 후 최대 보정 p=1.88e-06, β=pooled p99도 부호·유의성 일치)하고 **같은 4셀에서 TTFT p95는 악화**(−183~−536 ms) — 9-셀 좌표는 16블록 전부 `S1-C`(트레이드오프). Zamba2 r2·r3는 "`FixedPolicy(34)`·sticky OFF=엔진 기본 궤적" 명명 허용(전제 검증, Gate 1 rev15/875293), **Granite r3·r4는 그 명명 금지**(전제 미검증, G1-c 미실행, in-job 반증기 `unchanged`는 검증 아님). **크기 인용 허용 셀=F-계열 미발화 Zamba2 r2 하나뿐**: α **−38.8%**[−41.6,−36.0]·TTFT p95 **+51.5%**[+35.8,+67.2](부호=T′ 대 C, 음수=분할 우수). ★**효과는 꼬리 한정**(q≤0.7 분위수는 4셀 전부 반대 부호로 유의, q0.5: −0.85/−2.58/−0.91/−1.68 ms; 기전="드문 대형 prefill 유발 decode 스톨을 균일한 소폭 decode 지연으로 교환"). ★**처치는 decode-active 시간의 35–40%만 실현**(T′ idx1 frac 0.355–0.405)되며 **100% 실현 arm(T, sticky ON)의 α 효과가 오히려 더 작아 어떤 dose 외삽도 금지**. "무분할(C)"=명시적 분할 부재일 뿐 잔여 차감 미측정(Zamba P-carve NOT-EQUIVALENT, 단 C·T′가 같은 yml·같은 green context로 부팅해 이 대비에서는 공통항 소거). **§1-1의 A4-vs-fused 격차의 성분·기여분·분해로 서술 금지, "PD 분리 자체가 원인" 승격 금지, SLO goodput·용량·fused 우열은 무언급**(같은 job에서 미조율 A2가 C를 요청 p95 ITL에서 이기는 셀 있음, TTFT p95는 4셀 전부 A2 우수 — 부호만). "SM 분할만 켠 fused arm"은 이 엔진에서 구성 불가. ⚠️**금지 6건**: ①`component_share` 인용 금지(Granite r3 Fieller 3.098="몫 310%"가 존재하나 코드 자신이 항등식이라 표시). ②두 report json `headline.text` 인용 금지(m=2 Holm을 "4셀 Holm 후"로 오기, 메인 세션이 `holm._family.size_this_job=2` 대조로 확인). ③"분할이 ITL 개선" 꼬리 한정 없이 쓰기 금지. ④Δ^cont 헤드라인 승격 금지(T mean e2e 최대 +41% 악화). ⑤T′ F-B disjunct(i) 미평가 병기. ⑥인용 셀이 지정 셀 아닌 구조적 저용량 복제 셀, 선별 필터 귀무 발화율 40.1%(1−0.95¹⁰, 메인 세션 재계산 일치) 병기. **신규 방법론**: (i) §3 항목18/게이트#9("게이트 자신이 항등식") 새 사례 — §4.5 두 안전장치가 같은 3 꼬리통계만 보고, CONVENTION-SENSITIVE 12지표 중 9개 비트동일. (ii) §3 항목34/게이트#20 새 사례 — `MIN_COVERAGE=0.98` 상수만 수입, 규칙 미구현("식별자 수입≠거동 수입"). (iii) 신규 §3 항목38 — `any()` over n reps 스크린 귀무 발화율=1−(1−α)ⁿ. (iv) 신규 §3 항목39 — §6.3-6 관측자 대칭 보고(`dropped_events`/`writer_error`) 미산출, 감사자 오프라인 복구. 상세 위 rev20 헤더·§3 항목18·34(개정)·38·39, 원자료 `g2s_report_zamba2-27b_877756.json`·`g2s_report_granite-40-h-micro-base_877757.json` |
| 2 | **운영점 = cudagraph-ON** | decode wall 제거(TPOT 41→12ms), goodput ~1.5–2×↑. 기존 no-cudagraph 수치는 전부 하한 |
| 3 | ★**layer-type 런타임 정책 全형태 死** | 근거는 **서빙 직접 측정**: 4-모델서 agnostic 4/4 승 + **coordinated per-type 구현이 TPOT 42→124ms**. **(B,L) 2D knee**: 재배분 lever **Diff B ≈ 1.0 (L≥8000, 0.96–1.04)**. ⚠️**정정(2026-07-17)**: **L=2000선 Diff B≈1.35**(B1 1.38/B48 1.34)이고 **격자가 실 서빙 regime(ShareGPT 98%가 L<2k)을 안 덮음** ⇒ **"lever 부재" 기전은 long-context 한정·짧은 L엔 외삽**. 결론은 서빙 측정이 지탱하며, 짧은 L의 死因은 **(D) granularity**로 추정. 시각화 `results/prefill_knee/diffA_vs_diffB.png`. **✅ WIDE 스윕(L 256–32768×B 1–16, 2026-07-18, jobs 857371/857477)으로 확증**: lever는 **L≤512서 실제로 열림**(Diff B 256→1.42/512→1.22), L≥1024 ≈1.0; 기전=짧은 L서 둘 다 SM 미활용(mamba 44SM 포화); **그래도 死**(stakes sub-ms/layer ≪ (D) 42→124ms, batch 무영향). `knee2d_wide.png`. decode-side는 lever 있으나 sub-step (D)drain + **cudagraph 비양립**. §14 예약도 fixed d16으로 degenerate. ★★**강등(2026-08-04, claims-auditor+result-analyst X2′ 재집계, 4,104 ZBPT줄/285셀 블록평균 역산, `workspace/engine-port/results/prefill_knee/AGGREGATE_COMPOSITION_2026-08-04.md` REV.2, UNAUDITED 배너 유지)**: 계측 결함 2종 확인 — (1) **버킷 비대칭**(`src/models/zamba2.py:163` `_zt("attn")`=RadixAttention 코어만·qkv/o_proj/MLP 제외 vs `:259` `_zt("mamba")`=mixer 전체) (2) **누산기 러닝평균**(`:391-394` `_zt_acc`가 emit 시 리셋 안 됨 + `knee2d_wide.sbatch:120`·`decode_knee_vs_ctx.sbatch:90`의 `tail -1`). attn·mamba를 나란히 재구성하면 **둘 다 부풀려지나 attn은 mamba의 1/20~1/2뿐**(비순환 확증: 두 독립 job이 같은 셀서 보고 mamba 111.578 vs 116.648=1.0454×인데 steady 84.012 vs 83.941=1.0008×). ⇒ **"WIDE 스윕으로 확증"은 철회**: steady 재계산 **L=256은 단일값 인용 금지, 밴드 [1.24, 1.34]로만**(추정량 의존, 보고 1.42는 warm-up 편향) / **L=512 1.198**(보고 1.22). **"L=2000선 Diff B≈1.35 ⇒ lever는 L↓에서 열린다"는 REFUTED** — steady **1.0081** [1.0078, 1.0084](보고 1.32/1.38은 런 간 4.5% 불일치, steady는 두 독립 job이 0.08%로 일치); **B=48 행은 shape 혼합 셀이라 별도 인용 불가**. ★정책 단위(모듈 전체)로 환산하면 `R_policy ≈ 1 + w_attn·(DiffB−1)`(`w_attn`=attn/(attn+mamba/6), L=256서 9.6%) ⇒ **1.032/1.031로 소멸** — 단 이는 독립 증거가 아니라 `w_attn`이 작아 U-K가 구조적으로 1로 끌리는 결과다. ⚠️**n_indep=1**(WIDE는 셀당 런 1개) — 모든 CI는 **런 내 블록 정밀도이지 재현성이 아니다**. **판정 자체는 불변**: "layer-type 런타임 정책 全형태 死"는 서빙 직접 측정(4모델 agnostic 4/4 승, coordinated TPOT 42→124ms)이 지탱하며 이 강등의 영향을 받지 않는다 — 바뀌는 것은 **기전 서사**뿐이다("lever는 있었는데 (D)가 삼켰다" → "정책 단위에서 lever가 애초에 없었다", negative가 더 깨끗해짐). ★**Diff A/B를 인용하는 모든 정본 문장에 "no-cudagraph micro" 라벨 필수**(`knee2d.sbatch:61`·`knee2d_wide.sbatch:82` 둘 다 `--disable-cuda-graph`, 기존 미기재). 상세 [layertype_dynamic_NEGATIVE_2026-08-04.md](layertype_dynamic_NEGATIVE_2026-08-04.md)·[layertype_dynamic_POSITIVE_2026-08-04.md](layertype_dynamic_POSITIVE_2026-08-04.md) |
| 4 | ★**얽힘(entanglement)** | prefill·decode가 running batch(`max_running_requests`)·KV 공유 → **decode 굶김 → ITL↑ → batch 정체 → prefill admission 차단 → TTFT 폭발**. 실측: **d16은 prefill에 92SM(최대)를 주고도 TTFT 7.24s**, d24(84SM)는 1.21s |
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
| 25 | ★★★**(2026-08-03, claims-auditor + 독립 재현) decode 축은 라벨이 4–19%만 실현된다 — 그리고 기존 pin 게이트는 항등식이었다** | **(a) pin 게이트 = 항등식.** telemetry 120파일·prefill-active 스냅샷 **77,688개**에서 `prefill_sms ≠ target ⟺ decode_running_batch_size == 0`이 **양방향 위반 0건**(off-target∧decode-empty 50,328 / on-target∧decode-busy 27,360). `multiplex/multiplexing_mixin.py:773,792-794`가 decode batch가 비면 **설계대로** 무분할로 떨어뜨리므로(§1-22가 이미 '의도된 경로'로 기록), `e1_pin_check.py`의 시간가중 pin 게이트는 파티션 제어가 아니라 **'prefill in-flight 중 decode가 안 비어 있던 시간 몫'**을 잰다 ⇒ **방법론 게이트 #6 위반**(정확성을 강제하려고 만든 게이트가 저질렀다). 따름정리로 **조건부 pin = 27,360/27,360 = 1.000 정확** — **파티션은 질문이 성립하는 곳에서 완전 실현**(job 872077 재채점 64/64 PASS). **(b) 정작 게이트가 없던 축 = decode.** decode-active 시간 중 `decode_sms == D` 비율(872077, n=8/셀): T8 d16 **0.038**/d24 0.047/d44 0.082/d54 0.093, Ha8 d16 0.104/d24 0.110/d44 0.148/d54 0.187 ⇒ **셀 라벨의 decode 분할은 decode 작업시간의 4–19%만 실현되고 81–96%는 무분할 108 SM**. ⇒ (i) E1 격자는 **지속적 decode-SM 배분을 주지 않으므로** C2가 잰 물리량(prefill 16 고정·decode 연속 D)과 **다른 양**이다 — 긴장 A를 이 격자의 비(比)로 닫을 수 없다; (ii) 희석 계수가 **셀마다 다르다**(T8 0.038→0.093, Ha8 0.104→0.187) ⇒ `A_free(d16)/A_free(d54)`는 SM 수준과 **분할 engagement 비율을 동시에 움직인다 = 추정량 내부 교락**. **Stage 0 D108(라벨≠실현, §1-21)의 decode 축 판본**이며, §1-22가 요구한 실현분포 보고가 prefill 축에만 적용돼 있었다. 신규 게이트 `E1_COND_PIN`·`E1_DECODE_REALIZED`(`e1_pin_check.py`, 2026-08-03)로 코드화, 상세 `results/s8_frontier/DESIGN.md` §4.3.8(h) |

| 26 | ★★★**(2026-08-03, claims-auditor, 같은 날 §1-25 속행) `g`는 이 격자에서 은퇴한다 — 희석-보정 가설 REFUTED, NO VERDICT 사유가 "estimand 미식별"로 확장, `A_free` 자체가 결함, arm 비교에 미제거 교락** | **(A) 희석 attenuation 가설 REFUTED.** 메인 세션이 세운 모형("`E1_DECODE_REALIZED` 4–19% ⇒ `A_free(dD)=w_D·A(D)+(1−w_D)·A(108)` 혼합, 보정 시 Ha8 g≈1.62–1.70")을 3중으로 반증: (i) **control-arm reductio** — 같은 보정식을 T8에 적용하면 corrected g **21–29×**(C2의 2.36–2.91×를 10배 위반, 요구 ITL p95 352–360ms 대 실측 30.67ms); (ii) **de-engagement 직접 실험**(같은 셀 unsplit 분포에서 engagement를 낮춤) — `A_free` 변화 **1–11%뿐**, w=0에서도 g 거의 그대로(Ha8 1.173/T8 1.796); (iii) 핵심 가정 "`A(108)` 셀 무관"이 실측 위반 — Ha8 SPLIT-only 0.920[0.842,0.998](CI가 1 배제, 부호 반대), **T8 헤드라인은 UNSPLIT-only(decode SM 대비가 정의상 0인 부분집합)에서 그대로 재현**(1.795[1.589,2.001] ≈ ALL 1.837). 죽은 것은 보정이지 §1-25가 확립한 "engagement가 낮다"는 전제 자체가 아니다(세 계측기 교차확인으로 견고). **(B) 872077 NO VERDICT 사유 확장.** 코드 사실: `pdmux_context.py:initialize_stream_groups`가 `SM_COUNTS=[(108,0)]+divisions+[(0,108)]`를 하드코딩하고 `multiplexing_mixin.py:773,792-794`가 prefill 비-in-flight 시 무조건 `(0,108)`로 되돌린다 — 즉 이 기판에서 **"decode가 D SM에서 돌았다"⟺"prefill이 동시에 in-flight였다"는 같은 사건**이다. §1-24(ITL 꼬리=monolithic prefill, 크기가 108−D에 단조)와 결합하면 `g`는 사전에 **"decode-SM 탄력도 라벨을 단 prefill-SM 탄력도"**일 것이 예상되고, 실측이 그와 일치(UNSPLIT-only에서 T8 헤드라인 재현)한다 — **n으로 해결되지 않는 설계 결함**. `g = A_free(d16)/A_free(d54)`는 **이 격자 한정 은퇴**(sticky-partition 기판 수정 전 인용 금지), 블록 8→12–16 증설 재실행은 **선행 금지**. ⚠️"Ha8에 decode-SM 레버가 없다"는 CONFIRMED 아님 — 현 데이터는 그 질문에 답하지 못한다, 긴장 A(HE2 vs C2)는 전혀 닫히지 않았다. **(C) `A_free` 추정량 자체의 결함.** `e1_m3_control.sbatch:281-306`: `PREFILL_BLOCK_TOK=1024` 필터가 이 워크로드 요청의 4–5%만 걸러 prefill 작업의 **74–77%가 필터를 통과**(제거되는 ITL은 전체의 ~2%뿐) ⇒ §1-24가 확정한 monolithic-prefill stall이 **d16–d54까지 오염 범위 확장**(§1-24 원문의 "d92만 오염" 한정을 넓힘). 요청의 27.5–29.5%가 output≤25 토큰이라 내부 p95가 사실상 max ITL로 퇴화 — 평균의 선형 혼합 항등식이 극단 분위수에 성립하지 않음(비단조 응답으로 실증). `A_free`는 blocking-제거본이 아니라 대부분 monolithic-prefill stall로 이루어진 극단꼬리 통계다. **(D) arm 비교의 미제거 교락.** 공통 rate 2에서 T8 conc 12.8/decode batch 4.5/ITL p50 ~11ms 대 Ha8 conc 30.6/batch 15.8/~30ms — 양 arm 모두 off-cliff 평탄역(metric cliff 아님)이나 decode batch size가 memory/compute-bound 여부를 결정하는 공변량이라 arm과 완전 교락 ⇒ "attributable to the arm" 문구는 **현재 허용 안 됨**(통제는 realized concurrency/decode batch를 맞춘 rate여야 함); 이 교락은 관측 *방향*을 설명하지 않으므로(batch 큰 쪽이 오히려 무반응) 대안 설명이 아니라 **미제거 교락**으로만 기록. 부수: d16은 `sm_group_num:3`, d54는 4(guard row)로 셀마다 green context 수가 다르고 d54의 `decode_sms==44`는 telemetry에 미관측(guard row 미선택) — 행동 교락 아니나 셀 간 차이. `DESIGN.md` §4.3.8(h)의 "A re-run under RULE_BOUNDS would settle it"은 **이제 틀렸다** — sticky partition 구현 전 재실행은 같은 estimand-미식별 문제를 반복한다. 다음 gate: `PDMUX_STICKY_PARTITION` 구현(engine-porter, correctness gate) → sticky 격자 1회(872077 대조 8 block) → 사전등록 판별 예측(T8≈1.85·Ha8≈0.92 vs Ha8≈1.6, CI 비중첩) → `E1_DECODE_REALIZED≥0.90`이 sticky에서는 항등식이 아닌 **진짜 게이트**. 상세 `results/s8_frontier/DESIGN.md` §4.3.9. **부수(UNAUDITED, 인용 금지 — result-analyst 산출, claims-auditor 미통과)**: `m3_decode_empty.py` 분석이 §1-25의 희석이 decode-empty가 아니라 **prefill 부재**(decode-active 시간에 조건부라 decode-empty는 정의상 분자·분모에 안 들어감)로 나온다고 보고하며, 죽은 telemetry 필드 4개(`decode_ready_queue_depth`·`active_decode_sequences`·`decode_idle_ratio`/`prefill_idle_ratio`)를 코드 근거로 식별한다 — **claims-auditor 미통과이나 부하를 올려 engagement를 높이는 방향이 지지되지 않는다는 결론만은 감사자와 독립적으로 수렴해 그 좁은 항목만 AUDITED로 인용 가능**(rate를 올리면 prefill·decode가 비례해 늘고 split-eligible iteration 수는 셀 무관 194–214로 거의 일정) |
| 27 | ★★**(2026-08-03, 같은 날 2차 속행) `A_free` 대체 추정량으로 estimand 이관 완료[AUDITED, (3)만 예외] + `PDMUX_STICKY_PARTITION` 구현·correctness gate 통과[구현 사실, 성능 판정 아님]** | **(I) 조건부 per-token 추정량**(`m3_conditional.py`, §1-26의 `A_free` 결함을 대체). 단위 = 개별 ITL 구간 1개(요청별 집계 없음), 라벨 `split_frac(a,b)≥0.90`(SPLIT)/`≤0.10`(UNSPLIT)/사이는 양쪽에서 배제(AMBIGUOUS 0.3–1.4%), 통계량은 셀-블록별 **직접 분위수**(primary `p95(SPLIT)`, control `p95/p50(UNSPLIT)` — 두 셀 모두 108 SM이라 대비가 정의상 0). **[AUDITED]**: T8 d16 pooled per-token p95 **11.60**(`A_free`는 28.31, 2배 이상 바깥 꼬리 — 요청의 27.5–29.5%가 outlen≤25라 `A_free`가 사실상 max ITL로 퇴화, `A_free`는 요청 ~9.6개에 얹히는 반면 새 추정량은 셀-블록당 1,390–7,045 토큰). client↔telemetry 시계정렬은 `phase=="benchmark"` 필터 **금지**(그 마커는 warm-up 요청에 발화, 실제 probe 시작이 아님); `ALIGN_R_MIN=0.95` 미달은 flag-only(조용히 배제 금지) — 배제하면 오히려 대비가 커짐(LOO 실측, T8 1.688→1.822). **[UNAUDITED — 감사자가 이번 턴에 새로 생산해 자기 산출을 자기가 감사한 형태, 별도 확증 전 인용 금지]**: `PREFILL_BLOCK_TOK` 스윕(1024→512→256→0)에 **무릎이 없고** 임계 0에서 d16-vs-d54 대비가 두 arm 모두 소멸(0.986/1.005) — 임계는 자유 모수가 아니라 답을 정하는 손잡이. 권고: **`A_free` 은퇴**, `A_all` 유지 병기, **primary 라벨 = realized partition**(`decode_sms==D`, persisted state variable이라 견고); secondary(prefill overlap) 라벨은 872077이 `PDMUX_TRACE_FORCE_PREFILL=0`이라 **계측 결손**(SPLIT 라벨 토큰 중 "overlap-free"로 나오는 비율 Ha8 66.3/38.8%·T8 76.1/27.1% — 물리적으로 불가능해야 할 값, 곧 계측 과소표집의 증거). **(II) `PDMUX_STICKY_PARTITION`**(`multiplexing_mixin.py` +151/−17): decode-busy 시 무분할 fallback을 우회(`if not running_batch.is_empty() and (split_prefill_batch or sticky_partition_enabled)`), **decode-empty 시엔 의도적으로 index 0 release**(hold 아님 — 보호할 decode 작업이 없고 `E1_DECODE_REALIZED`가 decode-active 가중이라 가중치 0). OFF는 short-circuit으로 **패치 전과 byte-identical**(독립 재구현 pre-patch selector와 전 격자 동등성 테스트로 확인). cudagraph 보존(스트림별 캡처 유지, eager fallback 없음). `PDMUX_LA_COORD`/`PDMUX_SLO_SCHED`/`PDMUX_FIXED_DECODE_SM_FILE`/비-`fixed` `PDMUX_R2_POLICY`와의 조합은 init `RuntimeError`로 거부(반쪽 sticky 방지). **correctness gate 전부 PASS**: CPU 회귀(40 tests, sync manifest SHA-256 일치) + sticky 단위 테스트 12건 + **GPU smoke(job 872800, Ha8 d16)**: 고정 프롬프트 6개 greedy 출력이 OFF/ON **byte-identical**. **realized 관측(게이트 아님, n=1)**: sticky OFF `E1_DECODE_REALIZED=0.0839`(기존 동작 재현) vs **ON=1.0000**(사전등록 ≥0.90 초과, 튜닝한 것 없음) — decode-busy 스냅샷이 ON에서 `(idx1,92,16)` 139/139, OFF에서 무분할 진입 146회. **구현 완료 ≠ 성능 주장 성립** — throughput/latency/goodput/`g` 그 무엇도 주장되지 않는다. **(III) sticky 런 사전등록**: 872077 소급 재분석은 **DIAGNOSTIC 전용**(re-score 금지, 설계 판정의 근거로만), primary 통계량 1개(`p95(SPLIT)` 비) 선언, 게이트 4종(`ALIGN_R_MIN`·`E1_DECODE_REALIZED≥0.90`·`AMBIG_FRAC` 상한·`MIN_N_SPLIT` 하한), **★`G_LEVER`/`G_FLAT`는 미결정으로 기록**(기존 1.5/1.15는 `A_free` 스케일이라 그대로 이전 불가 — C2 측정범위(2.36–2.91×)에 묶는 안이 논거는 있으나 E1은 D 범위가 좁고 상보적(P+D=108)이라 그대로 못 씀, sticky 런 제출 전 별도 사전등록 필요), sticky 후 primary 모집단이 `SPLIT ∧ BLOCK-FREE`로 이동함을 미리 등록, trace-force ON 재허용 여부는 **열린 설계 쟁점**으로 미결정 기록. 판별 예측(§4.3.9 승계) 불변: prefill 주도면 T8≈1.85·Ha8≈0.92, 희석 가설이 옳았다면 Ha8≈1.6(CI 비중첩, 8 block으로 구분 가능). 상세 `results/s8_frontier/DESIGN.md` §4.3.10–4.3.12 |

| 28 | ★★★**(2026-08-03, 같은 날 3차 속행, claims-auditor) C2 → `G_LEVER` 앵커 경로 = 폐기, `G_LEVER`/`G_FLAT`는 §4.3.12(d)대로 UNDETERMINED 유지** | `c2_anchor.py`(신규, 미추적)로 시도한 5개 주장을 감사 — **주장 1(realized 검증)만 CONFIRMED, 2–5 전부 REFUTED/NOT-YET-SUPPORTED**. **★§0 결정적 발견(신규, 최상위 열린 항목)**: 같은 arm·같은 서버 플래그(`s8_sweep.sbatch:141-146` vs `e1_m3_control.sbatch:212-218`)·매칭 batch에서 "decode 16 SM" per-token ITL p50이 **2.6× 다르다**(C2 865493 SPLIT bin5 **28.79ms** ≈ `FINDINGS_8B` §2 28.48ms, vs **872077(E1 격자) d16 11.09ms**) ⇒ 둘 중 하나가 거짓: (i) 872077의 `decode_sms==16`이 실제 16-SM 하드웨어 실행이 아니다(`DESIGN.md` §4.3.11이 명시적으로 미검증으로 남긴 잔여층 — green context 생성이 하드웨어 SM 부여를 보장하는지 재프로브 안 함), 또는 (ii) C2의 28–31ms가 decode-SM 비용이 아니라 그 셀 배치(`[16,16,76-idle]`+상시 keepalive prefill 동거)의 성질이다. **어느 쪽이든 C2 비를 E1 격자로 이식 불가** — confound #1의 실측 대 실측 재현. **872077 전체와 이번 세션 sticky 결과가 딛고 선 바닥**이므로 최우선 열린 항목으로 기록. **주장 1** — CONFIRMED이나 서술 2건 정정 필수: (a) "파티션 활성률 0.66–0.93"은 `realized_pin_check.py`의 **스냅샷 개수 가중**이지 시간가중 all-busy(**0.803–0.958**)가 아님(방법론 게이트 #4 위반, 이 감사 한 건에서 3회), (b) 108 SM 시간은 warmup이 아니라 **창 밖 drain 전용**(warmup도 @D≈1.000, 기전은 오히려 강해짐). **구조적 함의(신규)**: in-window residency와 UNSPLIT 표본은 구성상 여집합 ⇒ `E1_DECODE_REALIZED≥0.90`을 통과하는 런엔 **음성대조가 정의상 존재할 수 없다**(865493 UNSPLIT n=0, sticky ON smoke D108 0초) — §4.3.12(e)(i) 확장 필요. **주장 2**(primary p95→p50) — 관측 CONFIRMED·기전 REFUTED(오염원은 monolithic prefill이 아니라 **파티션 전환 인접 구간**, 사건의 83–87%가 전환 0.5s 이내)·**처방 REFUTED 3중**(①같은 배제를 SPLIT에 적용해도 ≤2%만 이동 ②오염 기전이 sticky ON서 소멸 ③872077 실측 `T8 sp_p50=0.996[0.990,1.002]`로 p50 전환 시 양성대조조차 1.00이 돼 어떤 `G_LEVER>1`도 발화 불가=**캠페인을 구조적 NO VERDICT로 확정**하는 처방, 게다가 같은 p95 음성대조가 872077에서 반대 방향으로 깨짐) ⇒ **primary=`p95(SPLIT)` 유지, p50은 secondary, 전환-근접은 진단**. **주장 3**(편향=하한) NOT-YET-SUPPORTED(가산/곱셈 미식별 + 증거가 항을 0으로 만듦 — C2 분할 셀 전부 1410MHz 고정, 클럭 하락은 무분할 np뿐이라 **E1/sticky가 낼 throttling 비용을 C2는 안 냄**=반대 방향 경고). **주장 4**(`G_LEVER=1.41`) REFUTED(끝점 선택만으로 [1.41,2.40] 전 구간 도달 가능, 2.02는 게이트-FAIL job 4셀 포함, 1.41이 872077 T8 CI 하한 1.227을 가로지름). **주장 5**(`G_FLAT=1.25`) 방법 PLAUSIBLE·숫자 REFUTED(LOO 8개 실측 t95 반폭 0.303⇒1.30인데 Ha8 점추정 1.340>1.30로 자기 데이터서 뒤집힘, sticky가 사후 sd를 낮춰 사전-sticky 기반 임계는 관대해지는 방향=귀무 오수용 편향). `n_indep=1`(865493↔865533은 keepalive 설정이 달라 replicate 아님=**세 번째 pseudo-replication**). **regime 매칭 정정**: 872077의 12.8=concurrency, C2의 12.7=decode batch — 단위 맞추면 T8 2.8× 어긋나고 Ha8이 잘 맞음(앞선 서술과 정반대). **남은 경로(계획만, 미실행)**: `G_LEVER`는 (α)sticky 파일럿 T8 양성대조 효과크기 또는 (β)arm 간 대비 `g_T8/g_Ha8`(batch-매칭 rate 필요) — **둘 다 감사자 발안이라 독립 사전등록 필요**; `G_FLAT`는 사후-sticky sd 측정 후 TOST 동등성 마진(역시 독립 사전등록); 감사자 제안 게이트 S1(≈1 GPU-시간, T8 sticky ON d16+d54 C2복제 vs E1복제 2 block, 3갈래 판정) 미실행. 상세 `results/s8_frontier/DESIGN.md` §4.3.13 |
| 29 | ★★**(2026-08-03, 같은 날 3차 속행) D=54 앵커 측정 취소(jobs 872920/872921) — keepalive 재현성 결함[코드/로그로 직접 검증, 미감사] + C2 high-residency=워크로드 장치 산물[독립 수렴, AUDITED]** | 기록 `results/s8_scaleup/NOTES_D54_ANCHOR_2026-08-03.md`(미추적 신규) + 하네스 4파일(미추적, `s8_sweep_d54.sbatch`·`pdmux_p16_d54.yml`·`d54_block_ratio.py`·`runtime_source_manifest_d54.sha256`) — **취소됐으나 설계(d16+d54 동일 캠페인, 4 block, 셀 순서 block 패리티 교대)는 재사용 가능**. **B-1[미감사, 코드·로그로 직접 검증]**: `s8_keepalive_prompt_224.txt`가 **1793 토큰**인데 `CTXCAP=1792` ⇒ 모든 keepalive가 HTTP 400. **865493은 byte-identical한 같은 파일로 `keepalive_errors=0`**이었고 두 srv.log 모두 `CTXCAP=1792` 동일 출력 ⇒ **2026-07-27 이후 엔진 트리 churn으로 context-length 거부가 엄격해졌거나 off-by-one이 이동**(원인 미규명, 자명한 수정 `KEEPA_REPS≤223` 미적용) — **s8_scaleup 캠페인 전체의 재현 불가 요인**이므로 인용 시 경고 필수. 결과: co-residency ~90–100%→**31–34%** 붕괴, 측정된 전 셀 `REALIZED_PIN` FAIL(T8 blk1 d16 0.316/d54 0.628, Hs8 d16 0.342/d54 0.577). **B-2[AUDITED — 독립 수렴]**: sticky OFF는 prefill in-flight일 때만 목표 분할을 유지(`_init_sticky_partition` docstring, `multiplexing_mixin.py:206-231`) ⇒ **C2의 높은 residency는 decode 파티션 제어가 아니라 keepalive 포화라는 워크로드 장치의 산물**. 세 경로 독립 도달: (A) 코드 읽기 (B) keepalive 사망 시 실측 붕괴(위 B-1) (C) 이번 run block-1 telemetry 교차표(`prefill_active>0` @D=**0.975–0.996**, @108=**0.000**, 4파일 — §1-28의 감사자 0.943–0.996/0.0000과 같은 모양이나 **다른 대조**(워크로드 장치 실패 전후)로 도달). ⇒ §1-26(B) estimand 미식별이 **C2에도 그대로 상속**됨 — C2 앵커가 죽는 **세 번째 이유**(§0 축 불일치·estimand 미식별 상속에 이어). 한계: 이 캠페인은 **np(무분할) 셀을 안 돌려** §1-28 주장3의 "np만 1290MHz 하락" 관측을 확증 못 함(분할 셀은 1396–1410MHz 평평, 일관은 하나 독립 확인은 아님). 상세 `results/s8_frontier/DESIGN.md` §4.3.14 |
| 30 | ★★★**(2026-08-03, 같은 날 4차 속행) §1-28 §0의 이분법이 유지 불가 — 세 번째 후보가 실측으로 문서화됨, 오프라인 분리 불가, GPU(S2) 대기 — 성능 판정 0건** | §0은 (i) 872077의 `decode_sms==16`이 실제 16-SM 실행이 아니다 / (ii) C2의 28–31ms가 셀 배치 성질이다 중 하나가 거짓이라고 적었다. claims-auditor가 `FINDINGS_S0_AXIS_2026-08-03.md`를 감사하며(자기감사, 방법론 교훈 12) 이 이분법이 "E1의 `split_frac≥0.90`이 D 파티션 실행 토큰을 올바로 분리한다"는 전제 위에 서 있으며, T8의 세 공유 셀 전부에서 그 전제가 깨진다는 재프레이밍을 냈다 — E1 SPLIT 모집단은 **이봉**이고, 윗봉이 C2 셀별 SPLIT p50과 **1–2% 일치**(d16 0.992/d24 0.991/d44 1.013), 아랫봉은 **같은 job의 UNSPLIT과 통계적으로 동일**(d16 비 1.011). **독립 재현(result-analyst, `S0R_REPLICATION_2026-08-03.md`)**: claims-auditor도 아니고 사전등록을 쓴 세션도 아닌 분석자가 감사된 `m3_conditional.py` 프리미티브만 재사용 선언하고 자체 C2 리더·mode estimator를 새로 작성, 생산자 자체(`label_probe` element-wise, `s0dc_client`의 자기 기록 20/20 정확)에 대조해 실행 — 행 1·3 **재현**(윗봉/C2 비 0.992–1.013 flat, 빠른봉/UNSPLIT 비 1.000–1.023), 행 2(슬로우-쉐어가 D에 단조 증가)는 **순서만 재현**(d24−d16 스텝이 block scatter 안에서 미해결, Δ=+2.37±3.52pp n=8, t=1.90<t_crit 2.365; 감사자가 인용한 수준값 8.95/13.60/23.62/29.38%는 사전등록이 고정한 모집단(`a_free_only=True`)이 아니라 `False` 모집단에서 나온 것 — **사전등록 자체의 내부 불일치**, 방향은 불변), **행 5(클럭 lag sweep)는 발화하지 않음**(최적 δ=+0.10s에서도 slow share d16 11.4%/d44 27.0%, 90% 문턱에 크게 못 미침; δ 절대값≥0.25s에선 UNSPLIT 배경(2.7–3.1%)으로 수렴 — 라벨이 순수 클럭 잡음은 아니되 0.05s 스케일에서 취약함을 동시에 보임). ★**행 4(음성대조)가 강한 형태를 죽였다 — 이 회차의 핵심 결과.** 같은 mode estimator를 **UNSPLIT(108 SM) 모집단**에 적용하면 T8 전 셀에서 동일한 슬로우 모드가 나타나고 그 위치가 셀을 정확히 따라간다(33.88→22.12→15.62→14.12ms, D=16→24→44→54; d54에서는 SPLIT·UNSPLIT 슬로우 모드가 수치까지 동일, 14.12=14.12). 사전등록 falsifier("~31ms 모드가 ~9% 점유")는 d16/d24/d44에서 문자 그대로는 발화 안 함(UNSPLIT share 2.71/3.28/4.84%, 5% 미만 — d54는 5.61%로 발화)이지만 **그 falsifier가 지키려던 실질은 확인된다**: d16의 슬로우 토큰 8,893개 중 **7,903개(88.9%)가 UNSPLIT 라벨**이고 SPLIT은 759개(8.5%)뿐이다. 농축(클래스 내 슬로우 비율/모집단 base rate)은 **SPLIT 2.33–3.24×, UNSPLIT 0.78–0.93×**(d16–d54 전 구간) — `split_frac≥0.90`은 슬로우 모드를 **격리**하는 게 아니라 **농축**시킨다. 순도(SPLIT 토큰의 ~93%가 빠른 모드)도 완전성(SPLIT이 그 job 슬로우 토큰의 8.5%만 포획)도 성립하지 않는다. ⇒ **세 번째 후보 (iii)**: 두 job은 같은 축이나 라벨이 순수하지도 완전하지도 않다 — **§0은 더 이상 이분이 아니라 3지선다이며, 오프라인으로 분리 불가**. 살아남는 두 읽기(셀 수준 현상 vs 클럭 오프셋 누출)는 S2(GPU, 별도 제출 중, 결과 없음)만이 인과적으로 분리 가능 — prefill-overlap 컬럼도 같은(어쩌면 shift된) 클럭에서 계산되므로 arbitrate 불가. **이 층은 §1-25가 이미 기록한 시간 층 희석(`E1_DECODE_REALIZED` 4–19%) 안쪽의 두 번째 층**이고, §1-21(target-vs-realized)·§1-26(B)(estimand 미식별)와 같은 계열이다. **철회 3건**(메인 세션이 같은 날 앞서 씀, `FINDINGS_S0_AXIS_2026-08-03.md` 배너와 동일): "§0 stands as written"(과잉 해석 — 보인 건 3개 인접 집계에서 p50≈11ms뿐), "aggregation-invariant"(정확한 문장은 "11.06ms 단일 모드가 지배적이라 집계 선택에 둔감" — mode dominance이지 estimand identification 아님), "11.09는 집계 단위 미기록"(**틀림** — 산출자는 `m3_conditional.report_conditional` 리포트 [3] `sp_p50=11.0905`, n=11,124, `a_free_only=True`, `m3_conditional.py:158-161,251-262,316-329`에 문서화됨; 없는 것은 그 stdout 저장분뿐 — "저장된 출력이 없다"와 "추정량이 미기록"은 다른 실패 모드다). **재사용 가치 있는 계측 결함 2건**: ★`c2_anchor.py` 표 [5] "UNCONDITIONED CELL SUMMARY"가 **M8 전체와 Ha8 d16을 조용히 누락**(`meta`가 `"t0_monotonic_s" in s` 분기 안에서만 채워지는데, `c2_anchor.py:181-187`, 표 [5]는 앵커가 필요 없음에도) — 865493 앵커 현황은 T8·Hs8=5셀 전부/Ha8=d16 없음/M8=전무이고, **Ha8 d16 rep은 실제로 돌았다**(`itl_ms_p50=112.84`, n=5,200) — "측정 부재"가 아니라 "텔레메트리 앵커 부재"; mode estimator의 60ms 상한은 **arm-이식 불가**(Ha8은 토큰의 0.16%만 창 안, 미해명 ~87ms 스파이크가 구조상 상한 위). **방법론 교훈(§3 신규, 아래 참조)**: 자기가 검증하려는 코드를 복사한 게이트는 항등식에 가깝다(S0 gate 1이 `label_probe`를 복사해 자기 자신과 대조, `wmean`은 무대조) · 여집합 클래스에 음성대조를 걸어라(행 4가 세 차례 놓친 것을 한 줄로 잡음, §3 항목 9와 뿌리는 같고 방향 반대). **증거 수준**: 강한 형태(레버=D-SM 실행 식별)는 **채택 불가**(음성대조 반증), 약한 형태(이봉·윗봉 일치·아랫봉=UNSPLIT·농축 2.33–3.24×)는 **재현됨(독립성 부분적 — 추정량은 감사자 제안, 사전등록은 메인 세션, 실행만 독립, 인용 시 이 스코프 문구 동반 필수)**. §0의 (i)/(ii)는 **여전히 미판정**. 게이트 S1(§4.3.13)은 "부분 실현" 분기가 없어 **현 상태로 실행 불가**(4번째 분기 필요). `G_LEVER`/`G_FLAT`는 §4.3.12(d)대로 **UNDETERMINED 유지**(이번 회차로도 미해소). 상세 `../PROJECT_STATUS.md` "8B decode-SM 프론티어" "2026-08-03(4차)" 소절, `results/s8_frontier/DESIGN.md` §4.3.15, 사전등록 `PREREG_S0_AXIS_2026-08-03.md`·`PREREG_S0R_MODE_2026-08-03.md`·`PREREG_S2_STICKY_ITL_2026-08-03.md`, 재현 판정 `S0R_REPLICATION_2026-08-03.md` |
| 31 | ★★★**(2026-08-05, claims-auditor CONFIRMED scoped) S2(job 873015) 독립 재현 — §1-28/§1-30 §0 최상위 열린 항목이 behavioural하게 종결, E1은 열리지 않음, 성능 판정 0건** | S2(sticky ON, T8 d16/d54, ShareGPT rate 2, n=8 블록, cudagraph ON, gpu37) pooled per-token ITL p50 = **28.92ms**(d16, t95[28.81,29.02])/**12.04ms**(d54, t95[11.96,12.12]) — 사전등록 `[28,34]ms` 안, `split_frac` 라벨 미사용으로 재현(`PREREG_S2_STICKY_ITL_2026-08-03.md` §4 row 1 발화). §0 이분법 종결: **(i)**(872077 `decode_sms==16`=실제 16-SM 실행 아님)은 **하드웨어 형태 REFUTED·라벨 형태 CONFIRMED**(872077 d16이 decode-busy 시간의 96.2%를 D108에서 보냄, 3.8%만 D16) — `decode_sms`는 `arbiter.sm_counts[stream_index]` 재진술(`dual_worker.py:608-623`)일 뿐 하드웨어 SM 부여 직접 프로브(S3)는 여전히 미실행. **(ii)**(C2 28–31ms=셀 배치 성질)은 **DISFAVOURED**(keepalive 없는 open-loop ShareGPT·decode batch 2.6배 작은 조건에서 C2의 0.93×=5.4% 빠르게 재현). 살아남는 답 **(iii)**: `split_frac≥0.90`이 D-파티션 클래스를 격리·완결 못함(sticky ON에선 이 문제 자체가 SPLIT≈전체 인구가 돼 무의미해짐, d16 100.000%/d54 99.969% SPLIT). `E1_DECODE_REALIZED`(시간가중) = **0.9990±0.0017**(d16)/**0.9994±0.0008**(d54), 16/16 cell-block ≥0.995(pre-patch 872077 = 0.038/0.093). **기전 독립 도출**: `runtime_snapshot`이 개수-서브샘플(`PDMUX_DUAL_WORKER_TRACE_EVERY=32`)이고 양 캠페인 모두 `PDMUX_TRACE_FORCE_PREFILL=0`이라, decode-busy 조건부 스냅샷 케이던스가 4 arm 전부 정확히 16 decode step(0.177–0.465s) — ITL 구간 하나가 스냅샷 1/16개를 걸침 ⇒ 872077 d16 SPLIT 모집단 기대 순도 ≈6%, S0-R mode 분해의 독립 경로 값 6.6–9%와 일치. **아티팩트 배제**: 배치 기여는 bin-matching 4.7%/within-run slope 2.7%/블록간 회귀 2.3–3.2% 세 추정 모두 소폭, 노드(gpu36→gpu37)·바이너리(`multiplexing_mixin.py` 1파일 차)·캠페인날짜 전역효과는 d54 companion ≤1.097×로 상한. ★**d54 companion은 사전등록 [13,16]ms 미달**(관측 12.03, CI 전체 13 미만) — `PREREG_S2` §5.3에 "primary 적중+companion 미스" 규칙 없음, 사후분석은 원인을 sticky 교란(부호 반대로 배제)이 아니라 [13,16] 구간 도출 자체의 cross-cell 외삽 오류로 귀속(C2 d44=14.68을 d54 대용 사용, C2 자체 곡선으로 직접 외삽하면 12.79로 이미 하한 미만) — **인용 시 필수 동반**. `DESIGN.md` §4.3.12(f) 판별 예측(T8≈1.85)도 관측 2.402로 빗나갔으나 **판별 arm(Ha8) 미제출**이라 **설계상 미판정**(모형 반증 아님). ★**이 런에는 결과(outcome) 게이트가 0개**: `AMBIG_FRAC`·`MIN_N_SPLIT`·`PREREG_S2` §3.1 일치검사(`p50(SPLIT)`≈`p50(all)`)는 `E1_DECODE_REALIZED≥0.90` 통과 시 SPLIT≈전체 인구가 되므로 sticky ON 하에서 항등식, `E1_DECODE_REALIZED`는 arm 간엔 비항등식이나 ON arm 안에서는 `stream_idx=_sticky_fixed_idx` 코드 불변식, `ALIGN_R`은 계측 flag — 어떤 게이트도 "28.92 vs 11" 결과를 사전 제약하지 않음. **방법론 게이트 #9의 네 번째 재발**로 §3-23에 등재(세 번째 재발은 `S2_ANALYSIS_2026-08-04.md` §3의 §3.1-only 항등식 발견, 이번은 런 전체로 확장). `S2_ANALYSIS_2026-08-04.md:61-64`의 "Instrument check" 문단(케이던스 ~2.0ms 서술)에 **정정 표시**(원문 보존, 삭제 아님) — decode-idle 값을 decode-busy로 오인, 90–260× 오차·방향도 반대. **§0 최상위 열린 항목 해제**: "미해소 3지선다"→"CONFIRMED(scoped)로 종결, 단 하드웨어 층 미프로브"로 전환. C2 자체(레버 존재, 2.36–2.91×, scoped)의 등급·수치는 **불변**(자기완결적 4-arm matched-batch 캠페인, 이 종결의 영향 밖) — 바뀐 것은 "C2와 E1/sticky 격자가 같은 물리량을 재는가"라는 상위 질문뿐(이제 그쪽으로 confirmed, 단 `G_LEVER`/`G_FLAT` 미결이라는 별개 이유로 C2→sticky 이식·앵커는 계속 금지). **E1은 4가지 독립 사유로 이번 회차에도 열리지 않는다**: (1) `G_LEVER`/`G_FLAT` 여전히 UNDETERMINED(post-sticky 블록 sd가 pre-sticky 대비 ~14× 붕괴해 pre-sticky 산포 기반 임계는 null 채택 편향), (2) sticky 기판이 estimand를 바꿈(decode-busy 시 prefill이 벽시계 ~77% 유휴 — co-located `[108−D,D]` 예산 배분이 아니라 단일-테넌트 decode 측정에 가까움), (3) prefill 축 미통제(d16 TTFT p50 46.3→63.2ms, 기전 주장 없음), (4) 음성대조 구조적 부재(d16 UNSPLIT n=0/8블록)+S3 미실행. ⇒ **긴장 A(HE2 vs C2)는 전혀 닫히지 않았다.** **(α) 고-D 대조 셀은 2026-08-05 실행됐다(job 873921, T8 d92=(P16,D92), sticky ON, ShareGPT rate 2, n=4 블록, cudagraph ON, gpu41).** 관측 pooled per-token ITL p50 = **11.26ms**(telemetry-path, per-block t95 [11.049,11.467]) / **11.32ms**(raw-itls path, t95 [11.066,11.568]), `E1_DECODE_REALIZED` = 1.000/0.998/0.998/1.000, `n_err=0`. 사전등록 규칙(`s2_sticky_d92.sbatch:403-408`) 적용 시 **INDETERMINATE**(12–13 밴드에 5.7 block-sd 미달, 28–30 붕괴 밴드에서 ~106 block-sd 이격). **붕괴 분기는 REFUTED** — §0 종결은 유지되며 등급은 **CONFIRMED (scoped) 불변**이다. ★**밴드 부검(필수 동반) — 사전등록 밴드 [12,13]은 잘못 도출됐다(REFUTED).** 앵커 C2 d92=12.88은 재계산으로 정확하나(pooled raw 12.875, n=239,659), **C2와 sticky 격자는 파티션만 같고 워크로드가 다르다**: decode-busy ctx_p50 중앙 **1291 vs 287 tok**, decode batch 평균 **11.31 vs 4.51**, closed-loop+keepalive vs open-loop. α 런 자신의 엔진측 step 회귀(`t = 10.883 + 0.0991·batch + 0.00034·ctx`, n=2,140)로 C2 동작점을 예측하면 12.44–12.80ms로 C2 관측 12.875와 잔차 0.9–3.8%이며, 여기에 이미 기록된 캠페인 계통 오프셋(−5.4%, 아래 "등재 금지"의 "28.92는 구간 중앙에서
견고" 항목 참조)을 더하면 격차가 사실상 소진된다. ⇒ **격차는 새 기전이 아니라 통제되지 않은 워크로드 격차다.** 이 밴드는 이 항목이 스스로 금지한 **C2→sticky 이식**을 예측에 사용한 것이며, 정본이 이미 보유한 더 가까운 앵커(872077의 D108 우세 10.98, α와 블록별 byte-matched·batch 4.56·ctx 283)를 쓰면 예측은 11.0–11.4로 관측과 일치했다. **"α가 §0를 수치적으로 확증했다"는 서술 금지.** ★**α의 실질 기여(순환성 제거).** α는 §0 종결의 근거를 **분쟁 중인 필드(`decode_sms`) 내부의 시간-가중 재진술**에서 **결과(outcome) 축 앵커**로 옮긴다: 블록별 byte-matched trace·batch-matched(4.51 vs 4.56)·ctx-matched(287 vs 283) 조건에서 ON d92/OFF d16(872077) = **1.0293 [1.0210, 1.0376]**, ON d16/OFF d16 = **2.6320 [2.6193, 2.6447]**, ON d54/ON d92 = **1.0658 [1.0586, 1.0731]**. prefill 비공존 조건 엔진측 per-step은 ON d92 **11.048** vs OFF (P0,D108) **10.979**(+0.6%). 872077 d16의 진짜 D16 질량은 라벨 분할이 아니라 **client ITL의 3.43%가 [20,45]ms(중앙 31.7ms)**로 나타나며 실현 시간점유 3.5%와 일치한다. ★**872077 안에서 SPLIT p50(11.08)과 UNSPLIT p50(10.98)의 차이는 1%인데 같은 기판의 진짜 D16 vs D92 대비는 163%다 — 라벨은 사실상 아무것도 분리하지 않았다**(이것이 (iii)의 결과-축 재진술이다). 붕괴 분기는 세 다리로 독립 반증된다: α의 d92 pin(11.3) / sticky **이전** 바이너리 C2의 실현 (P16,D16) 91–96%에서 31.05ms / 872077 자신의 느린 모드 31.7ms. **동반 필수**: 이 비교는 노드(gpu36/gpu37/gpu41)·바이너리(1파일)·날짜를 건너며, **3% 이하 차이는 그 오프셋 안이므로 정밀 일치로 읽지 않는다**(같은 바이너리 OFF arm(δ) 미실행). ★**스코프(유지·강화). S3(하드웨어 부여 층)는 α로 닫히지 않는다.** α는 D92와 D108을 **0.6–2.9%**밖에 벌리지 못하므로 "하드웨어가 92 SM을 부여했다"를 검정할 **검정력이 구조적으로 없다**. α가 배제한 것은 저-SM 가설(2.63×)이지 고-SM 내부 구분이 아니다. 이 항목의 "selector-level, 하드웨어 직접 프로브 없음" 스코프 문구는 **그대로 유지**한다. **다음 gate(재정렬, 2026-08-05)**: **(δ) 같은 바이너리 OFF arm — 승격: "선행조건 아님" → "10% 미만 교차-job 비교를 인용하려면 필수".** 1블록 d92(가능하면 d16도), sticky flag만 OFF, 같은 노드·같은 날. 위 실질 기여의 1.029·1.006 비교가 딛고 선 계통 오프셋을 처음으로 측정한다. ~7 GPU-min. **(β) OFF 1블록 `PDMUX_TRACE_FORCE_PREFILL=1`** — 유지(높음), (iii)의 유일한 양적 다리 직접 검정. 밴드는 **같은 job 내부 값에서만** 뽑을 것. **(α′) 신규(~10 GPU-min)** — sticky ON d92를 **C2의 클라이언트로**(C1024 closed-loop conc16 + keepalive) 1–2블록. 사전등록 예측 **12.4–12.9ms**(α 내부 회귀에서 도출). 적중하면 "C2와 sticky 격자가 같은 물리량을 양적으로도 잰다"가 처음 성립하고 격자 이전 금지 근거 일부가 해제되며, 빗나가면 **C2 앵커는 영구 은퇴**다. **(γ) S3** — 유일하게 남은 스코프 구멍, 우선순위는 δ/β/α′ 뒤. 제출 순서 권고: **δ(7분) → β(7분) → γ, α′는 γ와 병렬**. **등재 금지**: "slow-mass 잔차 2.03×"(REFUTED, count/time-share 단위 불일치), "하드웨어가 16 SM을 부여했다"(미검증), "S2는 A/B다"/"sticky의 인과 효과"(before/after, 매니페스트 1파일 차), "음성대조 통과"(d16 UNSPLIT n=0=검정 불능), `g`/`G_LEVER`/`G_FLAT`/decode-SM 탄력도/goodput/HE0/긴장A 일체(p95비 2.227은 값+CI+"판정 없음" 동반해서만), "28.92는 구간 중앙에서 견고"(하단에서 3.3%, 자기 측정 계통 오프셋(−5.4%)과 같은 크기 여유). ★**추가(α, 2026-08-05)**: "α가 §0를 수치적으로 확증"(REFUTED, 위 밴드 부검 참조) · "11.26≈11.09는 정밀 일치"(교차-job 오프셋 안, 사후 통계량 선택) · "고-D에서 라벨 순도가 다르다"(85/138,588, n 과소) · "α가 하드웨어 층을 닫았다"(D92-D108 판별력 0.6%) · α의 TTFT 수치를 이용한 일체의 성능/프론티어 판정. ★**통제 확인(α).** 매니페스트: 873015 vs 873921 공유 11파일 해시 전부 동일 ⇒ **같은 sticky 바이너리**. 서버 인자 354키 중 차이 4개(`port`·`random_seed`·`pdmux_config_path`·`internal_states`), **양쪽 cudagraph ON**, backend triton 동일. 워크로드: 블록별 `input_lens`/`output_lens` sha256이 872077·873015·873921 전부 동일 ⇒ block-paired 성립. **미통제**: 노드(gpu36/37/41)·캠페인 날짜·클라이언트 seed 경로 ⇒ 3% 이하 비교의 허용오차 미상(δ 사유). 신규 방법론 항목은 §3-26(예측 밴드도 이식 금지 규칙의 적용 대상) 참조. 상세 `../PROJECT_STATUS.md` "8B decode-SM 프론티어" "2026-08-05(α)" 소절, 원자료 `workspace/engine-port/results/s2_sticky/s2a_pooled_873921.txt`·`s2a_T8_873921_result.txt`·`s2_sticky_d92.sbatch`(전용 분석 md 아직 미작성), `../workspace/engine-port/results/s2_sticky/S2_REPLICATION_2026-08-05.md`(§0 종결 전문), `results/s8_frontier/DESIGN.md` §4.3.16 |

| 13 | ★★**HE0의 구조적 이유 — 두 regime의 최적이 *충돌하지 않는다*** | **TRUE per-phase goodput**: 정책간 spread가 **LO(rate 3) 0.067 (2.3%) vs HI(rate 12) 1.187 (43%)** ⇒ **차별의 ~95%가 과부하 phase에서 발생**. LO는 split에 **무관심**(prefill-heavy 극단 d16 2.861 ≈ decode-heavy 극단 d44 2.858 = 구분 불가) ⇒ **LO엔 쫓아갈 최적점이 없고, HI의 최적은 LO에서도 공짜**(§1-6 비대칭의 정량 확인). ⇒ **"항상 HI 최적"=decode-heavy static이 정의상 최선**이고 동적은 과도만 지불. **동적이 이기려면 regime 간 최적이 *충돌*해야 하는데 이 워크로드엔 그 구간이 없다**. ⚠️**정정 이력**: 2026-07-18(§1-16)엔 "이 논증은 관대 SLO 한정, tight선 HI 최적이 동적"이라 봤으나, **§1-17(직접 재튜닝)이 반증** — tight SLO에서도 HI 최적은 **고정 decode-heavy(d44)**이고 동적은 얽힘 트랩으로 열위. ⇒ **이 논증은 tight SLO에서도 성립**(SLO 엄격도 무관). ★**각주(2026-08-05, ceiling-censoring 진단 + claims-auditor 감사 — 2026-08-04 각주 교체)**: LO goodput은 이중으로 절단돼 있다 — 처리량 인자는 도착률에(도착 span 199.0s vs duration 209.7s, `throughput ≤ 2.861` 전 arm), pass 인자는 SLO 여유에(`pass ∈ [0.973, 0.998]` 16런 전부). 후자는 런을 늘려도 남는다. **LO에 레버는 실재한다**: `요청별 ITL p95의 p90` 53.36±1.32 → 46.54±1.09 ms (**5.6 SD**, 4점 단조), 중앙값 12.183±0.131 → 13.912±0.216 (3.2 SD), `TTFT p50` 73.56 → 79.61 ms (3.3 SD). ⇒ **"LO는 split에 무관심"은 지표의 무신호이지 레버의 부재가 아니다.** ⚠️ **단 LO 레버는 SLO 예산 단위로 HI의 1/14이고 통계에 따라 부호가 뒤집힌다**(중앙값은 d16 우세, 초과질량은 T<15.5에서 d16 우세·T>15.5에서 d44 우세) — **"어느 split이 LO에서 좋다"는 functional 없이 정의되지 않는다**. §1-13의 "HI 최적이 LO에서 공짜"는 **goodput functional 한정 참**. **HE0 불변.** ⚠️**배제(진단서 원안 중 REFUTED, 인용 금지)**: T=20/30/60 임계 사다리 논증 전체(진단서 §3.4·§5 근거5) · "[55,65) split-불변 모드가 >60 질량을 지배"(§3.5 후반) · C2 정성 대조(§4 "T4 참고 대조") · "gpu39 3중 사다리"(§3.6, line 286 — d24 rep1의 실제 노드는 gpu43이지 gpu39가 아님, line 32 사실오류). 상세 정정 이력은 `workspace/engine-port/results/slo_sched/CEILING_CENSORING_DIAG_2026-08-05.md`(AUDITED, 정정 표시 포함, 원문 보존). ★**각주(2026-08-05, F — 허가)**: §1-13의 "HI spread 1.187 (43%)"은 **하네스 인라인 스코어러의 legacy mean-ITL 판정**(`sharegpt_vary_bench.sbatch:94` `m=sum(I[i])/len(I[i])`)이고 **d16은 n=1**이다. 정본 술어(요청 내 ITL p95, n=4)로는 HI d16 0.443±0.016 vs d44 3.613±0.571 = **+716%**(legacy 2.859±0.241 vs 3.924±0.046 = +37%). **순위·부호·"차별의 대부분이 HI"라는 결론은 불변, 오히려 강화**(정본 술어 n=4로 HI 몫 **97.3%**, legacy 91.9%). ⚠️ 단 정본 술어에서 HI는 rep 분산이 커져 **d34 3.405±0.478 vs d44 3.613±0.571은 분리 불가(0.4 SD)** — "+716%"는 **d16↔d44 쌍에만** 쓴다. ★**각주(2026-08-05, E — 재작성본, 진단서 원안은 귀속 오류로 불허)**: §1-13의 LO 컬럼은 **arm마다 n이 다르다**(`bench_noise_root_cause.md:74-86`: d44/d34/bind-nogate n=4, bind+GATE n=9, **d16/d24/slo n=1**). spread 0.067의 두 끝점(d16 2.861, d24 2.794)이 **둘 다 n=1**이고, "d16 2.861 ≈ d44 2.858"은 **n=1 값과 n=4 평균의 비교**다. 정본 술어 n=4 재산출: d16 2.831±0.038 / d24 2.761±0.140 / d34 2.835±0.025 / d44 2.848±0.007 ⇒ **spread 0.087 (3.1%)**, 그러나 **d24 한 arm의 rep SD(0.140)만으로 spread 전체를 삼킨다**(bind-nogate SD 0.110도 마찬가지). ⇒ **"LO는 split에 무관심"의 정량 근거는 spread 값이 아니라 "spread < rep SD"라는 부등식으로 다시 써야 한다.** 결론 불변. ⚠️ 진단서가 쓴 "spread 0.067 전부가 d24 rep1의 TTFT>3s 12건에서 나왔다"는 **REFUTED** — bind no-gate가 n=4로 2.795±0.110에 앉아 있어 d24를 지워도 spread 0.066으로 사실상 불변이다. 이 귀속을 정본에 쓰지 마라. ★**각주(2026-08-05, 노드 교락 방어)**: d16 = gpu39×1 + **gpu37×3**, d44 = gpu39×1 + **gpu38×3** ⇒ arm과 노드·날짜가 3/4 rep에서 교락(진단서의 "gpu39 3중 사다리" 방어는 사실오류로 못 씀 — 위 배제 항목 참조). arm 내부에서 노드 효과를 직접 추정하면 d16 **0.047ms** / d44 **0.33ms** vs arm 효과 **1.73ms** ⇒ 노드 효과는 arm 효과의 **3–19%**(`frac(>20)`에서는 ≤6%). 완전 매칭 대조 존재: **d34/d44 rep41-43은 같은 노드(gpu38)·같은 날·같은 rep 인덱스**로 paired 가능하며 부호 유지. |
| 14 | ★**stationary r8의 "시스템 노이즈" = 메트릭 절벽 (외인성 아님)** | 워크로드 4런 전부 동일(fingerprint), 하부 섭동은 **thru 3%·ITL 8%**뿐인데 goodput 2× — **r8이 TTFT≈SLO(3s) 경계에 앉아** 3% 결손이 TTFT 평탄역을 1.5s→3.7s로 밀어 임계선을 넘김. **3=견고/8=불안정/12=견고** ⇒ 경계 regime만 불안정. 상세 [bench_noise_root_cause.md](bench_noise_root_cause.md) |

| 20 | ★★★**Oracle 재구성(TTFT⊗ITL 분해): headroom은 +2%가 아니라 +16% — 단 그건 disaggregation 몫 (사용자 지적, 2026-07-19)** | §1-19의 "+2.1%"는 per-static oracle이라 **coupled 절충점만** 봄. **goodput을 TTFT-pass ⊗ ITL-pass로 분해**(사용자 지적)하면 진짜 headroom이 보임: phase A(양 SLO 동시 binding)서 **d16 TTFT-pass 57.8%(ITL 실패) / d24+ ITL 100%(TTFT 하락)** — **DECOUPLED oracle**(d16의 TTFT ⊗ d24의 ITL)=**57.8% = best static 49.7% 대비 +16%**. ★**그러나 그 headroom은 92 prefill SM(d16-TTFT) + 24 decode SM(100% ITL) = 116 SM > 108 요구 = coupling TAX(8 SM 초과)라 단일-GPU 불가**(+ 얽힘이 batch로 추가 결합). ⇒ **두 개의 다른 천장**: 단일-GPU **동적**=coupled ceiling **+2%**(못 이김) / **decoupling=disaggregation ceiling +16%**(별도 디바이스 풀서만). **진짜 headroom은 디바이스 간에 존재, 단일-GPU split(동적이든)엔 없음.** `oracle_corrected.png`. ★★**등급 강등(2026-08-04, claims-auditor)**: 이 오라클(+16%)은 §1-19(극단 disjoint mix, jobs 860497–518)의 데이터를 그대로 재사용하는데, 정본 자신이 §5(열린 항목 (c), 아래)에서 그 캠페인을 **n=1~2·overload-only·underpowered**로 이미 기록하고 있다(§2-4 방법론 "n≥4 없이 정책 결론 금지"에 못 미침). ⇒ **"+16%"는 "확정"이 아니라 n=1~2, 미확증으로 강등한다.** 결합 가정(서로 다른 arm의 주변 pass율을 합성해 DECOUPLED oracle을 만든 것)도 별도로 미검증이다. 판정(단일-GPU와 disaggregation은 별개 천장)의 **방향**은 §1-4/§1-6 얽힘·비대칭 기전과 정합해 그대로 두나, **"+16%"라는 magnitude는 인용 시 이 caveat 동반 필수**. 상세 [layertype_dynamic_POSITIVE_2026-08-04.md](layertype_dynamic_POSITIVE_2026-08-04.md) §2.5(P6) |
| 19 | ★★★**disjoint-feasibility region은 존재하나 동적은 거기서도 패배 — 이유는 conjunctive SLO의 구조 (사용자 극단-도전 검증, 2026-07-19)** | 사용자 논리(어떤 static도 양 phase 두 SLO 동시충족 못 하는 workload 필연 존재)를 극단 mix(A prefill-heavy in2048/o32@8, **B decode-heavy in2048/o512@5=긴ctx라 ITL-binding**)로 실증: **median-feasibility DISJOINT 확인**(feasible-A={d16} ∩ feasible-B={d24,d34}=∅; job 860497–518). ★**그런데 동적 여전히 패배**: graded goodput서 per-phase 최적이 **인접**(A→d24, B→d34)이라 **ORACLE 동적조차 best-static +2.1%뿐**(d24가 양 phase 근최적: A 2.73=최적, B 1.01 vs 1.09), **reactive bind는 −20.6%**(오배치, 양 phase 실패). ★**깊은 이유**: conjunctive SLO(TTFT∧ITL)가 동적을 **동기부여**(d16 최고TTFT·d44 최고ITL)하는 바로 그 힘이 **각 phase 최적을 중간 compromise로 당김**(d16은 A서 ITL벽·d44는 B서 TTFT벽) → 서로 다른 phase 최적이 인접 → 단일 중간 static이 양쪽 서빙. ★**게다가 이 region은 OVERLOAD서만 존재**(전 정책 gp 0.99–1.87, 대다수 SLO 실패): 용량 이하=전부 통과(static 자명)·이상=전부 실패(static 최소손실). **동적이 유용하게 이기는 operating regime 없음.** `extreme_disjoint.png` |
| 18 | ★★**mix-스윙 트레이스서도 동적 패배 — 최적은 좁은 중간대만 스윙 (사용자 도전 검증, 2026-07-19)** | 기존 결론은 rate만 변하는 fixed-mix 트레이스 한정이었음. **mix-스윙**(phaseA prefill-heavy in2048/o32 ⇄ phaseB decode-heavy in256/o512, static sweep+bind, job 860452–470) 실측: **최적이 d24(A)↔d34(B)로 *좁게만* 스윙**(d16↔d44 아님). ★**prefill-heavy phase를 d16이 안 이김**(d24 5.006 > d16 3.016 > d44 1.564). 기전=**goodput=TTFT-SLO ∧ ITL-SLO가 반대로 당김**: d16 최고 TTFT(1.47s)·최악 ITL(56ms, 60벽 근접); d44 반대(ITL 22ms·TTFT 3.64s로 3s 실패); **중간 d24가 둘 다 충족→승**. **단일 중간 static d24가 양 phase 근최적**(A 5.006=최적, B 2.854 vs 최적 2.870=0.6%차)이라 **combined d24 3.930 ≫ bind 3.419**. ⚠️caveat: phaseB 포화(thru 2.9<offered 5)·n=2; **어떤 static도 양 phase서 두 SLO 동시충족 불가한 극단 mix는 미검증(동적의 남은 문)**. `mixswing.png` |
| 17 | ★★**동적이 지는 이유 = 오버헤드 아니라 *positioning* (실패 지점 규명, 2026-07-19)** | "오버헤드>이득"은 이미 반박(switch~0 §1-8, CPU 0.014% §1-12). 로그가 실패 지점을 정확히 보임: **최적=dec_sm 44(d44, throughput·goodput 양쪽 1위)인데 컨트롤러는 dec_sm 16–24(평균 22)서 진동하며 44에 절대 도달 못 함 = decode-STARVED**. 기전 = **reactive**(TPOT 스파이크 후에야 decode에 SM)+**symmetric**(두 slack 대등화)이라 decode가 잠깐 괜찮아지면 즉시 prefill로 회수 → 구조적으로 decode-heavy 최적에 누적 불가. 손실은 **switch 비용이 아니라 앉은 위치**. ★**risk/reward 18:1**: prefill-ward 이동의 LO 이득 ≤2.3%(§1-13 LO split-무관) vs HI 오배치 손실 ≤41.5% ⇒ 매 스위치가 나쁜 베팅. ★**모든 수정(anchor·비대칭 penalty·이동 중단)이 "44에 앉기"=static으로 수렴** — gate(ratchet, 34서 정지)가 best-dynamic이나 undershoot. **동적은 안 움직여 static과 *tie*가 상한, 이길 regime 없음**(§1-13). `why_dynamic_loses.png`. ★**각주(2026-08-04, claims-auditor)**: 위 "모든 수정"의 **'모든'은 reactive 계열**이다. 비-reactive는 이미 시험됐다: Step D `PDMUX_SLO_LFF`(context-length feedforward, `multiplexing_mixin.py:825-832`) = **HD0, n=3(underpowered)**, Step F `sat_predict`(포화 예측 트리거)도 시험됨. ⇒ **온라인 feedforward는 시험돼 net win 아님(n=3, n≥4 재시험 미실행). offline 모델-프로파일 기반 decode-floor 예측(Claim E)만 미시험.** 본문 정정 불필요 |
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
| `../../workspace/engine-port/results/p1_gates/gate1/` | ★★★★★**Gate 1(job 874478, 2026-08-06) 원자료 — claims-auditor 감사 완료, 조건부 채택, 새 성능 판정 0건.** `gate1_result_874478.txt`(판정 출력)·`PREREG_GATE1_2026-08-06.md`(사전등록+부기1·2)·`gate1_analyze.py`(coverage guard 포함 분석기)·`gate1_telemetry_874478.jsonl`·`gate1_srv_874478.log`(근거 인용원, `:32`가 분할표 근거). 874465(1차 시도, 절단된 창)도 같은 디렉터리에 보존(비교용). **전문은 CONSENSUS §1-1 Gate 1 블록·§3 항목28·29 — 수정 금지, 인용만.** ★★★★★**같은 디렉터리에 G1-b(job 875293, 2026-08-07) 후속 — 사전등록 철회 규칙 발화, 새 성능 판정 아님.** `PREREG_G1B_2026-08-07.md`(사전등록)·`gate1b_result_875293.txt`(판정 출력, rate{2,3,4,6}). rate6에서 `(54,54)` 시간가중 8.37% 관측 ⇒ Gate 1의 "단일 분할 고정" 문장을 **철회**(rate2·3·4 인용 가능 셀은 불변). **전문은 CONSENSUS §1-1(rev15, "★실질 산출 1" 철회 블록)·§3 항목30 — 수정 금지, 인용만** |
| `../../workspace/engine-port/results/p1_gates/gate2/` | ★★★★★**Gate 2 rev4 본 캠페인(jobs 875344/875346, 2026-08-07) 원자료 — 메인 세션 원자료 독립 재현 확인(2026-08-09), claims-auditor 감사 기록 위치 미확인, 새 성능 판정 아님(정본 반영 복구).** `g2_report_zamba2-27b_875344.json`/`g2_report_granite-40-h-micro-base_875346.json`(`per_arm_x60` 포함 판정 산출)·`g2run_875344.out`/`g2run_875346.out`(실행 로그)·`g2_analyze.py`(분석기, `:538-543`이 primary A3-vs-A4 계산, `:734`가 A2 서술 위치 — R1′/R2′는 이 스크립트가 계산 안 함). **전문은 CONSENSUS §1-1(본 캠페인 블록, rev17)·§3 항목34 — 수정 금지, 인용만.** ★★★★★★**E-A(mixed-chunk 레버, jobs 875654/875657/875661, 2026-08-08~09) 원자료 — claims-auditor 감사 완료, 판정 조건부(문구 강한 제한), 새 성능 판정 아님(fused-측 조율 진단).** `PREREG_G2EA_2026-08-07.md`(사전등록+§14 자기인지 위험)·`g2ea_report_zamba2-27b_875657.json`/`g2ea_report_granite-40-h-micro-base_875661.json`(판정 산출)·`g2earun_875657.out`/`g2earun_875661.out`(실행 로그)·`g2eaprobe_875654*`(Stage 1 레버 실현 확인, HOLB jsonl 포함)·`g2ea_analyze.py`(분석기, `:662-680`이 방법론 게이트 #31의 근거). **전문은 CONSENSUS §1-1(E-A 블록, rev16)·§3 항목31–33 — 수정 금지, 인용만.** ⚠️**같은 디렉터리의 `PREREG_GATE2_2026-08-06.md`는 커밋 `c47fad0`으로 §14 addendum이 반영돼 워킹트리 클린이다(2026-08-09 확인 — "미커밋" 기록은 stale, 정정)** — 단 §11-3(3차) 감사는 면제된 채 제출됐다는 사실(§14.1)은 불변이라 "감사를 통과한 사전등록"으로는 여전히 인용 금지(인용 금지 목록 항목12). G2CTRL(`PREREG_G2CTRL_2026-08-07.md`)도 커밋 상태 확인됨(2026-08-09) — 동일 정정. ★★★**HOLB G5 재채점 + 874601 G3 라벨 정정(jobs 874601/874602/874632/874633/874635, 2026-08-09) — 새 성능 판정 아님.** `g2holb_g5_tost_rescore_2026-08-09.json`(72셀 전량 재채점 산출, G5=UNDETERMINED)·`g2holb_g5_tost_rescore.py`·`g2holb_g5_tost_summary.py`·`g2holb_report_zamba2_874601.json`(sha 추출 `KeyError` 하네스 결함 원자료). **전문은 CONSENSUS §1-1(rev18, 2026-08-09 두 번째 정본 반영 건 A/B절)·§3 항목35 — 수정 금지, 인용만.** ★★★**job 876699(T4-1, 2026-08-09~10) TIMEOUT 사후분석 + 공유 하네스 무한대기 수정 + Gate 2-S 설계 감사(2026-08-10) — 새 성능 판정 아님.** `g2det_876699.out`/`g2det_876699.err`(52분 스톨·SLURM TIMEOUT 원자료)·`g2det_analyze.py`(2026-08-10 수정, `on_clean` NO-VERDICT 가드)·`g2_holb_phaseA_lib.sh`(`:102-139`이 사건 경위·180s 근거 주석)·`PREREG_GATE2S_2026-08-09.md`(rev1–4, 3회 NO-GO 감사 이력, §0.0.A/B). **전문은 CONSENSUS §1-1(rev19, 2026-08-10 세 번째 정본 반영 건 A/B/C/D 블록)·§3 항목35(개정)·37 — 수정 금지, 인용만.** ★★★★★★★**Gate 2-S 첫 유효 결과(jobs 877756/877757, 2026-08-11) 원자료 — claims-auditor 적대 감사 "조건부 등재 가", 새 성능 판정 아님(별도 트랙, §1-1 대체 아님).** `g2s_report_zamba2-27b_877756.json`/`g2s_report_granite-40-h-micro-base_877757.json`(판정 산출, `design_conformance.conformant=true`)·`g2s_status_zamba2-27b_877756.json`/`g2s_status_granite-40-h-micro-base_877757.json`(`MEASURED_AND_SCORED`)·`g2s_analyze.py`(`:63`이 방법론 게이트#20 새 사례의 근거)·1차 실패 이력 `g2s_report_*_877107/877109`(하네스 결함, primary 0개). **전문은 CONSENSUS §1-1(rev20, 2026-08-11 블록)·§3 항목18(개정)·34(개정)·38·39 — 수정 금지, 인용만.** |

`deprecated_reports/`(2026-07-24부터 [`../deprecated/reports/quarantine_engine_port/`](../deprecated/reports/quarantine_engine_port)) = 초기 triage·포팅·모델별 평가·구 핸드오프·구 리포트. **이력 보존용, 현재 결론과 충돌 가능.**

---

## 5. 열린 항목

1. ~~**변화-trace 기반 재검증**~~ → ✅ **완료 (2026-07-17, jobs 856889–856975)**. n≥4 캠페인으로 **HE0 견고 확정**(§1-7, 5.4σ) + **게이트 정체 규명**(§1-10/11).
2. ~~**게이트 정교화 필요?**~~ → ✅ **성격이 바뀜**: 게이트는 지능적 제어가 아니라 **auto-tuner**(§1-10). 살릴 값어치가 있다면 **ratchet의 조기 정지 수정**(정지 규칙이 d34에서 멈춰 d44를 놓침) — 단 그래봐야 천장은 "best-static 매칭"이라 payoff는 *튜닝 자동화*뿐.
3. ~~**컨트롤러 CPU 오버헤드**~~ → ✅ **직접 계측으로 死 (2026-07-17, jobs 857111/2)** = §1-12.
4. ~~**시스템 노이즈의 정체**~~ → ✅ **종결 (2026-07-17)** = **노이즈가 아니라 메트릭 절벽**. 상세 [bench_noise_root_cause.md](bench_noise_root_cause.md).
   워크로드는 4런 전부 **동일**(fingerprint 일치)이고, 하부 섭동은 **throughput 3%·ITL 8%뿐**. 증폭기는 **r8이 하필 TTFT≈SLO(3s) 경계에 앉은 것** — 과부하 큐의 TTFT 평탄역이 3% 결손에 1.5s→3.7s로 이동해 **임계선을 넘음 → goodput 반토막**. rate로 재현: **3=견고(200/200×3) / 8=불안정(400,400,357,206) / 12=견고(142,141,142)** ⇒ 불안정한 건 **경계 regime뿐**. ★**GPU 클럭 throttling 가설 철회**(3.5×가 아니라 3%만 설명하면 됨). ★**자원 격리 불필요했음**. **stationary 부활 조건**: 용량(d24≈6.3/s) 아래 rate에서 측정하거나, 임계 지시함수 대신 TTFT 분포/용량 지표 사용(현 goodput은 과부하서 런 길이 의존 = ill-posed).
5. ~~(낮음) 얽힘-aware 행동모델의 정밀화~~ → **천장이 "static 매칭"으로 확정**(관대·tight 양쪽, §1-7·§1-17). 동적 upside는 **모든 SLO 엄격도에서 닫힘**. payoff 없음.
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
