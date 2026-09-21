# `prefill-layer-alloc` project status

최종 갱신: 2026-09-21(doc-steward — **2026-09-18~2026-09-21
로컬 세션 산출물 정본 반영**, GPU 0·새 성능 판정 0건. 저장소는
로컬 PC로 이양 완료, GPU 실행 불가 상태 지속). ★**A100 GPU
지출 0 · 새 성능 판정 0건 · Claim D/E 등급 불변 · HE0·layer-type
死·정책 순위·게이트 #6/#13/#16·stake #1 전부 불변**(재도출
없음, 아래 "이전"[2026-09-17] 포함 전 회차 계승).

**(1) `CLAUDE.md` 정정(작업 루트, `tools/claude/workspace/
CLAUDE.workspace.md` 동기화 완료)** — `handoff-report/
gpu_rental_checklist_2026-09-18.md` §0이 upstream SGLang
`v0.5.10` 소스를 1차 확인해, "PD-mux가 엔진 단계에서 major
10+ `ValueError`로 거부된다"는 옛 서술이 **자동 격자
(`divide_sm`) 경로 한정**이고 이 프로젝트가 전적으로 쓰는
**`manual_divisions` 경로에서는 미발화**함을 발견했다 —
claims-auditor가 `VERDICT_e2_substrate_portability_2026-09-18.md`
Q4에서 독립 검증(설치 wheel `sglang-0.5.10.post1`·`0.5.3rc0`
대조, 구조 바이트 동일) **`CONFIRMED(스코프 한정)`** 처리.
**결론(로컬 GPU에서 서빙 실험을 돌리지 않는다)은 불변** — 실효
사유는 ①설치된 `sgl_kernel`에 sm120 대응 빌드가 없어 import
자체가 깨짐(2026-09-18 로컬 실측 `undefined symbol:
_ZNK3c106SymInt22maybe_as_int_slow_pathEv`) ②16GB로 9B 모델이
안 올라감 ③**게이트 1**(기판이 다르면 수치가 주장 근거가 못
됨) 셋뿐이다. ★단 "그러므로 major 10+에서 돌아간다"로 확장하면
**거짓**이다 — 제약은 `sgl_kernel.spatial.
create_greenctx_stream_by_value`→CUDA green context API 층으로
**이동할 뿐 사라지지 않고**, 그 층의 실제 SM 입도는
**미실측**이다(신규 방법론 게이트 #258, `CONSENSUS.md` §3
항목278). H100 "8 SM 단위" 코드 상수도 같은 스코프 배너를
받았다(§4 대여 전 확인 절, `CLAUDE.md` 갱신).

**(2) E2 기판 이식성 — `REFUTED`(예산), 새 OVERRIDE는 사용자
결정 사항, 아직 미승인.** `VERDICT_e2_substrate_portability_
2026-09-18.md`(claims-auditor 규칙층 감사)의 결론: A100→A100
기판 변경 자체는 **死因을 만들지 못했다**(반전 시험에서 등록
판정을 뒤집는 수치 없음) — 그러나 E2 OVERRIDE의 "설계·예산
불변이면 재승인 불요" 조항이 요구하는 "예산 불변"이 **세 경로로
깨진다**: ①SLURM `--time` 하드 캡이라는 **집행기**가 없어지면
승인된 3.0 GPU-h 상한이 집행되지 않음(최악 ≈3.3–3.4 GPU-h
가능) ②908623 아티팩트 mtime 실측으로 산출된 예산 상수
(A=145.5·B=106.5s)가 호스트 의존 ③무상 SLURM 할당→시간당
과금이면 초과의 의미(잡 종료 vs 금전)가 달라짐 — 따라서
**`REFUTED`: 기존 OVERRIDE 그대로는 유효하지 않다, 새 OVERRIDE
필요**(신규 방법론 게이트 #259, `CONSENSUS.md` §3 항목279).
판정서가 제안한 **신규 caveat E2C-36…E2C-39**(offered 사다리는
A100 D44 전용이라 다른 기판 부하 사다리로 해석 금지 · 이 캠페인의
어떤 게이트도 기판 변화를 탐지하지 않음 · realized
green-context SM 미측정 · SLURM 밖 실행 시 3.0 GPU-h 하드 캡
미집행)는 **판정서 참조로만 승계**한다 — E2 사전등록 본문
(`PREREG_E2_STICKY_*rev1–3*`)은 등록 무결성 보존을 위해 이
회차에서 **편집하지 않았다**. ★★**새 OVERRIDE 승인 여부는
사용자 결정 사항이다 — doc-steward는 승인하지 않았고 승인할
수 없다.** 계정 복구 후 E2를 A100 대여 서버에서 실행하려면
이 새 OVERRIDE가 먼저 필요하다(기존 OVERRIDE는 여전히 KISTI
Neuron 재개 시나리오에만 유효·미소진).

**(3) 로컬 CPU 회귀 정본 기준선 정정.** `workspace/engine-port/
env/cpu_regression_baseline_2026-09-18.md` — dev tree 재구성
후 **정본 검증 인터프리터 = miniconda base `python3`**, 결과
**598 tests / failures=26 / errors=13 / skipped=3 / ~296초**
(`PDMUX_ROOT` env var 지정 시 failures=21). "이전 머신[KISTI
Neuron] 기준 140 tests / 약 73초"는 로컬 PC에 그대로 적용되지
않는다 — 3곳 정정: `handoff-report/
migration_plan_local_dev_remote_gpu_2026-09-18.md:61`,
`handoff-report/session_handoff_2026-09-17.md:127`,
`tools/claude/agents/experiment-runner.md:29`(및 동기화된
`.claude/agents/experiment-runner.md`). 잔존 실패는 전부
①`sgl_kernel`/GPU 부재(12건, 이 CPU 박스에서 원천 불가) ②
`PDMUX_ROOT` 미설정(23건, fail-closed 설계 그대로 — 6건은
`PDMUX_ROOT` 지정만으로 해소) ③venv 부재로 인한 서버 부팅
전제 실패(17건, 이 세션 지시 범위 밖) ④pre-existing gap
(`lambda_star.example.json` 미존재, 4건)로 분류됨 — **코드
결함 0건 발견, 전부 배관/문서 상태**.

**(4) 기타 등재(살아있는 문서, 새 분석 없음)**: `handoff-report/
vllm_baseline_stack_2026-09-18.md`(vLLM 대조군 스택·의존성
정리), `workspace/engine-port/results/local_smoke_5060ti_
2026-09-18/`(로컬 RTX 5060 Ti vLLM 3B 스모크 — **게이트 1
배너 포함, 어떤 결론의 근거도 아님**), `handoff-report/
ssm_kernel_sm_partition_2026-09-21.md`(SSM 커널×green-context
SM 분할 코드 사실 정리, §7 grid 정책·§8 `sgl_kernel` 소스
검토 — 코드 사실 수집, 새 성능 판정 아님).

**(5) venue-strategist residency prior-art 조사(2026-09-21,
웹 검색 시점 2026-09-21, `reports/paper/venue_positioning.md`
§0.2 신설).** 핵심 판정: **"워크로드 트렌드가 residency(한
디바이스에서 prefill·decode 동시 in-flight인 wall-clock
비율)를 키운다"는 주장할 수 없다** — 토큰 비(input:output)는
1차 근거가 풍부하고 강하게 prefill 쪽으로 이동 중이나(Azure
LLM Inference Dataset 2023/2024·Mooncake FAST'25·TraceLab
arXiv 2606.30560·GitHub Copilot 특성화 arXiv 2608.00101·
vLLM AgentX 블로그 2026-09-08·OpenRouter arXiv 2601.10088),
**residency 자체(PD 동거 wall-clock 비율)를 직접 측정해 보고한
공개 1차 데이터는 0건**이다(PD-mux 선행 DuetServe·MuxWise·
Nexus·Bullet 전부 미보고 — **신규 문헌 공백, 주장 가능**).
prefix caching 적중률(Copilot 98%·TraceLab 95.7%·AgentX
>96%·Mooncake ~40%)과 reasoning 토큰 비중 상승(ServeGen
NSDI'26·OpenRouter)은 **반대 방향 증거로 정직하게 등재**.
★**그럼에도 residency 축이 정당한 이유는 트렌드가 아니라 내부
타당성**: DuetServe(arXiv 2511.04791v2) 원문이 "prefill
isolation은 prefill이 iteration latency의 큰 비중일 때
최유리, decode-heavy regime은 원래 prefill-decode 경합이
적다"고 명시하는데, 저장소 자체 진단 셋(E2C-21의 shape A
prefill 유휴 86–89% · longctx_conflict 2F9 · λ0 실현 분할
불일치)이 정확히 그 regime을 가리킨다 — residency 축을 여는
것은 기여 확대가 아니라 **"unfavorable regime을 골랐다"는
비판에 대한 타당성 수리**다. **§6 MuxWise "게재처·수치 검증
필요" 해소**: **ASPLOS'26 확정, DOI 10.1145/3779212.3790236**,
평균 2.20×/최대 3.06× goodput, 주 testbed 8×A100-80GB(+H100/
H200) — 저장소 기존 자기정정("`sharegpt.yml`/`loogle.yml`
합계 132 SM = H100/H200급, A100 108 SM 아님")과 정합. 학회:
**MLSys 2027 마감 2026-10-30 20:00 UTC**(10쪽, 별도 abstract
마감 없음)로 확인 — GPU 부재+SLURM 블로커로 **이번 사이클
비현실적**; EuroSys/ATC가 현재 자산과 가장 정합. **쓸 수 없는
주장(금지 목록)**: "트렌드가 residency를 키운다" ·
"residency-large에서 동적 제어가 이길 것"(근거 0, H_L1은 반대
예측) · "layer-aware가 long-ctx에서 부활"(§2-(a) 이미 배제) ·
"1.53–1.94×가 Nexus 8–10×보다 작으니 우리가 낫다"(기전·기판
상이) · "ServeGen/Copilot trace로 실험했다/할 수 있다" ·
"chunked prefill이 residency를 늘린다"(가설, 근거 없음).
`reports/longcontext_trace_plan.md` §5/§8도 이 조사를 반영해
갱신(trace 후보에 TraceLab·Azure-2024 추가, radix-cache 결정이
residency 트랙 선결 게이트로 승격 — 미결정이라는 사실 자체는
불변).

**(6) 미커밋 코드(engine-porter/experiment-runner 소관, 기록만)**:
`PDMUX_ROOT` 이식 4스크립트 + 테스트 10파일, 신규
`workspace/engine-port/scripts/run/array_runner.sh` — 이
회차의 CPU 회귀 재실행(위 (3))이 이 변경을 **다른 세션이
동시 진행 중인 미커밋 작업**으로 확인했을 뿐, doc-steward는
이 코드를 검토·수정하지 않았다.

정본 반영: `CONSENSUS.md` rev79→**rev80**(§3 항목278–279 신설,
§4 "살아있는 문서" 표 갱신), `reports/paper/venue_positioning.md`
§0.2 신설+§6 갱신, `reports/longcontext_trace_plan.md` §5/§8
갱신. **정본끼리 모순 0건**(이번 회차 발견 없음). **커밋
금지**(사용자 승인 전까지 작업 트리에만 반영 — 커밋은
git-committer 소관, 사용자 지시 대기).

이전: 2026-09-17(doc-steward — **머신 이양 준비
체크포인트**, GPU 0·새 측정 0·새 성능 판정 0). 사용자가 이
머신(KISTI Neuron)을 더 이상 쓸 수 없어 이양 준비 — **이양
대상 미정**, 임시 저장소만 존재. 확정 결론·HE0·layer-type 死·
정책 순위·Claim D/E 등급·게이트#6·게이트#13/#16·stake #1·아래
"이전"(2026-09-15(6)) 전부 **불변**(재도출 없음). ★**SLURM
블로커 재확인**: 2026-09-17 `sbatch --test-only`가 2026-09-15와
**동일 문구로 거부**(계정 만료/한도초과, 원인 여전히 **미확정**);
OVERRIDE(E2)는 유효·미소진 그대로. 신규 추적 파일:
`workspace/engine-port/env/devtree_manual_edits.patch`(sync
미적용 수동편집 5파일 — 방법론 게이트 #33=`CONSENSUS.md` §3
항목48의 재확인, 새 판정 아님) · 같은 디렉터리
`venv_packages_2026-09-17.txt` ·
`tools/migration/pack_migration_bundle.sh`. ★**기판 의존(OPEN
USER DECISION, 판정 아님)**: E2 등록 격자·λ\*·OVERRIDE 예산은
이 클러스터 A100(108 SM) 기판 기준이다 — 다른 기판에서 E2를
"등록된 실험"으로 그대로 돌릴 수 있는지는 **미결정**(사용자+
규칙층 감사 사항, 이 세션은 판정 안 함). 전문
`handoff-report/session_handoff_2026-09-17.md`. **커밋 금지**
(핸드오프 커밋에 함께 묶인다).

이전: 2026-09-15(6)(doc-steward — 작업 수행일 2026-09-15,
문서 반영 2026-09-16 — **E2(sticky 분할 대조) 트랙 신설**[설계
**E2-α**(대조 arm 같은 job), 20 boot, 등록 예산 **1.803
GPU-h**(최악 2.338, 하드캡 3.0), 규칙층 3회차+하네스층 2회차=
감사 5회 전부 최종 `GO-with-caveats`(死因 0) — **아직 실행
안 됨, GPU 지출 0·라벨 0건**] + ★**신규 블로커**[SLURM 계정이
2026-09-15부터 전 파티션 제출 거부(`--test-only`도 거부,
association·QOS 한도 비어 있음, 파일시스템 쿼터 정상 ⇒ KISTI
외부 계정 관리 시스템, 원인은 만료 vs 한도초과 **미확정**·정황은
만료 쪽. 마지막 접수 job 908623(2026-09-14T21:30 정상 완주).
누적 1,153 jobs/3,698.3 CPU-h] + OVERRIDE 장부 갱신[기존 2건
소진(D-none 908534·λ0 908623), **3번째(E2) 사용자 승인·미소진**,
계정 복구 시 재승인 불필요] + 신규 방법론 게이트 4건[#254–257,
`CONSENSUS.md` §3 항목274–277] + E2C-19-b 회부 처리[정본 인용
금지 정정, 아래 "C" 참조]). GPU 장부 불변(R2 correctness
1.322778·λ0 1.391389 GPU-h, **E2 신규 지출 0**). 새 성능 판정
0건·**Claim D/E 등급 불변**(둘 다 미검증)·HE0·layer-type 死·
정책 순위·stake #1·게이트#6·게이트#13/#16·C2 인용정지·P2
블로커 ①②③ 전부 불변. `CONSENSUS.md` rev78→**rev79**(판단
근거는 아래 "E. 정본 반영" 참조). **커밋 금지**(핸드오프 커밋에
함께 묶인다).

## A. E2(sticky 분할 대조) 트랙 신설 — 준비 완료, 미실행

핸드오프 `handoff-report/session_handoff_2026-09-16.md`(작업일
2026-09-15)가 이 회차의 사실 원본이다. 사용자 등재 순서 b→a에
따라 **E2**(sticky 분할이 λ0에서 발견된 실현 분할 불일치를
바꾸는지 대조)를 먼저 준비했다. 설계 **E2-α**(대조 arm=sticky
OFF를 같은 job 안에서 측정) 채택 근거 3개: ① 게이트 #233(미등록
노드 축 gpu38→43→40→41)이 열려 있어 cross-job 비교는 노드
교락을 새로 지고 ② job 908534(D-none)가 귀속을 닫은 방법이
같은 job 안 대조였고 ③ cross-job은 `multiplexing_mixin.py`
1파일 차이 병기 의무를 발생시킨다.

**등록 격자**: `a_r4`(offered 8.0)×4 seed · `a_r2`(3.0)×4 seed ·
`b_r3`(1.15)×2 seed, 각×{OFF,ON}=**20 boot**, seed=λ0
`choose_seeds` 순위 1–4(4386/4162/251/2630).

**감사 5회, 전부 GPU 0**(전문은 핸드오프 "감사 5회" 표):
규칙층 rev1 `NO-GO`(F1: 음성대조가 확률 1 FAIL) → rev2 `NO-GO`
(F1′ E-time 규약 미결정·F2′ E-iter 비재현 공시가 거짓) → rev3
**`GO-with-caveats`**(死因 0, 追記 A1–A10) — 하네스층 rev1
`NO-GO`(H-F1 P7 CAPPED 미전파·H-F2 R5 무발화·H-F3 P3 발화 불가)
→ rev2 **`GO-with-caveats`**(死因 0, 追記 A11–A12). **신규 등록
caveat E2C-1…E2C-35.** 최대 위험 3개: **E2C-1**(sticky ON이
decode SM 108→44 *와* prefill 경계 드레인 제거를 동시에 바꿔
부호 반대) · **E2C-8′**(추정량 열은 규약 없이 인용 불가, 아래
"C" 참조) · **E2C-21**(shape A에서 sticky ON이 바꾸는 decode-busy
가중 시간의 86–89%가 prefill 유휴 구간 — Q2가 사는 것은
*"멀티플렉싱할 prefill이 없는 동안 decode를 44 SM에 묶어 둔
비용"*까지, **정책 서술 금지**).

**자기 공시(이 세션이 저지르고 정정)**: 규칙 정본 파일 안에
출처 허위 2건(rev2 §4-3의 "E-iter 4규약 전부 재현 실패" 거짓
서술, rev3 §4-2의 초 단위 역산 기재) — 상세 "C" 및 아래 신규
게이트 #255.

**사용자 결정 2건**: ①"1번으로 진행"(E2-α 채택+사전등록+규칙층
감사) ②"진행"(OVERRIDE 발효, 예산 3층 1.803/2.338/3.0 GPU-h·
사는 것(R1·R2·R3)·못 사는 것 전부(게이트#6 미해소·Claim D/E
불변·새 성능 판정 0건·λ0R-8(iii) 미해소·E2C-21) 명시,
`OVERRIDE_E2_SUBMIT_2026-09-15.md` §7).

**커밋**: `b917bfc`(13파일 5,638행: 사전등록 4·판정서 5·하네스
3·OVERRIDE 1) · `dff7bb7`(제출 거부 기록).

## B. ★블로커 — SLURM 계정 전 파티션 제출 거부 (프로젝트 전체 영향)

한 창 규율 1–3단계(presubmit 재확인·셀프테스트 3종·커밋
`b917bfc`)를 통과한 뒤 4단계 `sbatch`에서 스케줄러가 계정
사유로 거부했다:

```
sbatch: error: Your account has expired or exceeded the allocated CPU time.
Please contact the account manager. (account@ksc.re.kr, 042-869-0597)
```

읽기전용 진단(3파티션·`--test-only` 포함) 결과: **계정 전체
차단**(스크립트·파티션·`--comment` 무관), SLURM association
`GrpTRESMins`/`MaxTRESMins`·QOS `normal` 제한 **비어 있음**(SLURM
회계 한도 아님), 파일시스템 쿼터 정상(home 20.55G/64G·scratch
300.8G/100T) ⇒ 차단 주체는 **KISTI 외부 계정 관리 시스템**(클러스터
쪽 우회 불가). 누적 사용 2026-01-01~ **1,153 jobs·3,698.3
CPU-시간≈462.3 GPU-시간**(월별: 05월 2,037/07월 575/08월
554/09월 209 CPU-h). **원인 판별은 미확정**(메시지가 만료·한도
초과를 구별 못 함) — 정황은 만료 쪽(9월 사용량이 오히려 적고,
마지막 접수 job 908623이 2026-09-14T21:30, 거부는 09-15로 날짜
경계에서 끊김)이나 **확정으로 인용 금지**.

★**OVERRIDE는 유효·미소진** — 설계·예산 불변이므로 계정 복구
후 **새 승인 없이** 1·2단계 재실행+4단계 재시도만 하면 된다.

## C. E2C-19-b — CONSENSUS.md 인용 금지 정정(doc-steward 회부 처리)

`CONSENSUS.md` 머리(2026-09-14(5)·rev78 문단)의 §1-25·§1-26(B)
재현 문장에 든 **13.5 / 85.6 / 91.9**(a_r4 D44 점유·(0,108)
점유·b_r3 D44 점유)는 λ0 판정서 §4-A **2열(시간가중)**의
값이며, 그 열은 위 "A"의 **E2C-8′**로 **인용 금지**가 됐다(3회차
독립 탐색 10/160/256변형 전부 재현 실패, 최소 최대오차
1.44pp). 등록 규약(다음 비-startup 스냅샷까지·무캡)의 같은 셀
값은 **a_r4 13.64%** · **b_r3 97.63%**로 눈금이 다르다 — 두
값 모두 인용 시 규약명("등록 규약, 다음 비-startup 스냅샷까지·
무캡")을 병기할 것. `CONSENSUS.md`·`CLAIM_EVIDENCE_MATRIX.md`·
`EXPERIMENT_ROADMAP.md`에 인용 금지 배너 부착 완료(아래 "E"
참조), 원 판정서(`VERDICT_result_lambda0_908623_2026-09-14.md`
§4-A)는 이력 보존을 위해 수정하지 않는다.

## D. 신규 방법론 게이트 4건 (#254–257)

전부 **새 성능 판정 0건**. 대응 `CONSENSUS.md` §3
항목274–277(전문은 그쪽 참조):

- **#254**: 추정량 열은 규약을 명시하지 않고는 인용할 수
  없다(E2C-8′) — "재현되지 않는다"는 서술도 규약 전수 탐색 없이
  쓰면 그 자체가 출처 허위다(rev2 §4-3이 실제로 그랬다).
- **#255**: 계산하지 않고 적은 수는 규칙 정본 파일 안에서 가장
  위험하다(교훈80 재발, 이번 회차 안에서 2번) — E-iter 비재현
  거짓 공시(하네스 셀프테스트 기대값을 포기시킬 뻔함) + E-time
  역산 기재(정직한 하네스를 확률 1로 제출 차단시켰을 뻔함).
- **#256**: 변이 검증에는 무변이 대조군이 필수다 — 단 그 대조군은
  거짓 사살은 잡아도 거짓 생존(등가 변이)은 못 잡는다(감사자
  스스로 겪은 함정을 공시).
- **#257**: 수리에는 되돌림 회귀 검사를 함께 넣어라(교훈53
  강화) — 회귀 검사 없이 커밋된 수리 하나가 되돌림 변이에서
  SURVIVED했다가 검사 추가 후에야 KILLED됐다.

## E. 정본 반영

`CONSENSUS.md` rev78→**rev79**(§3 항목274–277 신설[게이트
#254–257] + §3 항목268 追記[E2C-19-b 인용 금지 정정] + 머리
문단 인용 금지 배너 + §5 신규 열린 항목10[E2 상태]).
`reports/paper/CLAIM_EVIDENCE_MATRIX.md`·`EXPERIMENT_ROADMAP.md`
에 같은 인용 금지 배너 부착(85.6/91.9/13.5% 시간가중 열
전파분, "해당 맥락만"). `MEMORY.md` 포인터 갱신,
`memory/slo-aware-scheduling-track.md`에 신규 `## 2026-09-15`
절, `memory/deconfound-measurement-lessons.md`에 항목252–255
신설(topic 파일 내부 순번), 신규 메모리 파일
`memory/slurm-account-blocker.md`(환경 사실, `type: project`) +
`MEMORY.md` 한 줄 추가. **커밋 금지**(핸드오프 커밋에 함께
묶인다).

**overclaim 금지**: (1) E2는 **아무것도 측정하지 않았다** —
`STICKY_REALIZES_A` 등 어떤 라벨도 아직 존재하지 않는다,
"sticky가 실현 분할을 고정한다"류 서술 금지. (2)
`GO-with-caveats`는 **실행 승인**이지 결과가 아니다. (3) 게이트
#6·P2 블로커 ①②③·Claim D/E 등급·HE0·layer-type 死·정책 순위·
stake #1 전부 재도출 금지(무변).

이전: 2026-09-14(5)(doc-steward — ★job 908623(λ0 0단계)
결과 감사 등재[**11셀 완주, `KNEE_BRACKETED` 양쪽,
`SEED_REPEAT_HOLDS` 양쪽**, λ\*(A)≈3.05 req/s(±0.6%,n=2)@창
131.01s,N=400 · λ\*(B)≈0.696 req/s(±0.3%,n=2)@창 286.85s,N=200 —
**17자리 인용 금지**, 라벨은 기계적 산물 CONFIRMED(전 아티팩트
바이트 재현, digest 11/11 일치), 死因 0·성능 판정 0건] + ★★가장
중요한 신규 발견[**λ\*(A)는 D44 측정이 아니다** — 실현 분할은
대부분 비분할 (0,108)(D44 점유 4.5–13.5%뿐 — ★2026-09-15 정정:
이 13.5%는 시간가중 열이며 E2C-8′로 인용 금지, 등록 규약 값은
a_r4 13.64%·b_r3 97.63%[위 "2026-09-15(6)" "C" 참조]), 정본
§1-25·§1-26(B)의 재현, NPC-I가 shape A에서 caveat→반증으로
승격, λ\*(A)로 B4/true-dual을 B1에 정규화하는 설계는 교락됨을
등재] + P2 블로커 갱신[**① λ0 `NO-GO` 해소** · ②W4 λ\* 실측
부재 부분 이동·미해소(4겹) · ③게이트#6 불변] + GPU 장부
갱신[λ0 트랙 신규 **1.391389 GPU-h**(이 트랙 최초 지출), R2
correctness 트랙 불변 1.322778 GPU-h] + 신규 방법론 게이트
6건[#248–253, `CONSENSUS.md` §3 항목268–273]). 새 성능 판정
0건·**Claim D/E 등급 불변**(둘 다 미검증)·HE0·layer-type 死·
정책 순위·stake #1·게이트#13/#16·C2 인용정지 전부 불변.
`CONSENSUS.md` rev77→**rev78**(판단 근거는 아래 "G. 정본 반영"
참조). **커밋 금지**(핸드오프 커밋에 함께 묶인다).

## A. job 908623 결과 — λ0 0단계 완주, 死因 0 · 성능 판정 0건

**job 908623**(gpu38, `sacct COMPLETED` 01:23:29=5,009s, **1.391389
GPU-h** — 등록 보통 1.348의 1.032×, 최악 코너 3.509의 0.397×) =
`PREREG_LAMBDA0_REV5_2026-09-14.md`(`8367b6ec…`)+구속 追記
(`632c2d78…`)+규칙층 판정서(`33dac116…`)+`OVERRIDE_LAMBDA0_
SUBMIT_2026-09-14.md`의 실행. 11셀 전부 완주(`missing_cells=[]`,
`UNRESOLVED` 0, abort 0). 등급 판정의 정본은 결과 감사
`VERDICT_result_lambda0_908623_2026-09-14.md`(607행, 감사자가
파일로 직접 작성 — 전사 충실성 공시 불필요).

**측정값**:
- shape A (256,512): `KNEE_BRACKETED`, **λ\*(A) ≈ 3.05 req/s
  (±0.6%, n=2) @ 창 131.01 s, N=400**
- shape B (8192,64): `KNEE_BRACKETED`, **λ\*(B) ≈ 0.696 req/s
  (±0.3%, n=2) @ 창 286.85 s, N=200**
- 양쪽 `SEED_REPEAT_HOLDS`. ★**라벨이 인쇄한 17자리(예:
  `3.0531296871219915`) 인용 금지**(λ0R-7) — 위 형식으로만
  인용하고 (창 길이, N)을 반드시 병기.

**라벨은 기계적 산물 CONFIRMED**: `LAMBDA0_LABEL.json`·
`cell_*.json` 11개·`plan.json`·`REACHABILITY.txt`·
`RUNNING_REQ_RECOUNT.txt`·`mutation_check.txt` 전부 감사자
무수정 재실행으로 **바이트 재현**, 등록 digest 11/11이 디스크·
HEAD 양쪽과 일치, 변이 54/54 blocked·escape 0. 반전 시험 7표면
전부 **λ\* 값을 움직이지 못함**(움직이는 것은 "존재" 라벨뿐,
최단 이동도 등록값에서 +3.3% 이상, 아래 "F" 게이트#253).

## B. ★★가장 중요한 신규 발견 — 실현 분할 혼합비: **λ\*(A)는 D44 측정이 아니다**

`tel_*.jsonl`의 `runtime_snapshot`(`prefill_sms`/`decode_sms`, 출처
`multiplex/dual_worker.py:608` = **그 순간 선택된 division**,
target 아님)을 세 독립 추정량으로 집계한 결과:

- **λ\*(A)를 공급한 셀(a_r4)**: D44 `(64,44)` 점유 **8.6% /
  13.5% / 17.2%**뿐, 지배 division은 **`(0,108)` 91.0% / 85.6% /
  82.8%**.
- **λ\*(B)를 공급한 셀(b_r3)**: `(64,44)` 점유 **90.7% / 91.9% /
  100%**.

기전(쉬핑 코드): `multiplexing_mixin.py:1199-1200` — decode
배치가 있는데 split-prefill이 in-flight가 아니면 **무조건
(0,108)을 고른다**. `lambda0.sbatch:92-94`가
`PDMUX_STICKY_PARTITION`을 unset해 sticky 수리도 꺼져 있다.

★**이것은 새 기전 발견이 아니라 `CONSENSUS.md` §1-25·§1-26(B)의
재현이다** — 정본이 이미 decode 축은 라벨이 **4–19%만 실현**
(`E1_DECODE_REALIZED`)한다고 등재하고 그 때문에 `g`를 이 격자
한정 은퇴시켰다. shape A의 4.5–8.6%는 그 밴드 안이고, shape B의
43–91%는 밴드 밖(8192-토큰 prefill이 거의 항상 in-flight라서).

⇒ **NPC-I(caveat)가 shape A에서 반증으로 승격**: *"λ\*는
B1(legacy, fixed D44) 한정"*은 **shape A에서 실현 조건으로
거짓**이다. 파생 귀결 3가지(**λ0R-8**, 필수 병기):
(i) λ\*(A)를 *"decode_sm=44 용량"*으로 인용 금지.
(ii) *"split을 바꾸면 5× 변한다"*(905835 계열)를 shape A에 이식
금지 — A는 D44 점유가 5–14%뿐이라 훨씬 덜 반응할 것으로
**예상**되나 미측정.
(iii) ★**λ\*(A)로 B4(true-dual)를 B1에 정규화하는 설계는
교락된다** — true-dual이 바꾸는 것이 바로 prefill·decode
동시성, 즉 실현 혼합비 자신이며, 정본이 `g`를 은퇴시킨 것과
**같은 구조의 교락**이다.

## C. 그 밖의 신규 발견 (요약, 전문은 결과 감사 §2·§4)

- **λ\*(A)는 `--max-running-requests 48`이 구속한 값이다**(λ0R-3,
  필수 병기) — a_r2/a_r3/a_r4 전부 `#running-req` 정확히 48,
  `#queue-req` 최대 217. 하드웨어 상한이 아니다. shape B는 이
  상한에 닿지 않는다(2/48, 병목은 `max_prefill_tokens=16384`).
- **게이트 #6의 크기(감사자 산출, 등록 판정 아님)**: 정본
  goodput 술어로 이 job의 per-request 기록을 재채점 — ITL
  조건은 11셀 전부 100% 통과 ⇒ 구속은 전적으로 TTFT. shape B는
  0.45 req/s 미만에서만 λ\*_SLO가 있고 사다리에 그보다 낮은
  셀이 없어 아래에서 브래킷 안 됨. shape A는 인용 가능 셀이
  0.36×·0.59×뿐이라(둘 다 100%) **하한만**(λ\*_SLO(A) ≥ 1.754
  req/s), 절벽은 2.7× 폭으로만 갇힌다. **λ5C-8의 "비
  2.3–3.4×"는 shape B 한정**(A는 0.983×까지 100%) — A에
  일반화 금지(λ0R-9).
- ★**λ\*(B) 0.697 vs 인용금지 λ_inf(B) 0.696(+0.2377%)의
  "일치"는 N-8을 되살리지 않는다**(λ0R-1, 이번 회차 최대
  위험) — (i) 차이가 이 측정 자신의 seed 재현 산포(0.310%)보다
  작아 통계적으로 구별 불가 (ii) N-8은 수치 조항이 아니라
  자격 조항이라 사후 독립 측정으로 충족되지 않는다 (iii)
  순서가 반대로 깨진다(λ_inf는 상한 프로브였는데 측정
  λ\*(B)가 그것을 초과, shape A는 순서 유지) (iv) 순환(B
  사다리 사전이 그 값을 겨냥해 설계됨). 합법적으로 남는 것은
  하네스 교차 일관성 서술뿐(λ0R-2, 필수 병기 5개 조건과 함께).
- **λ5C-5 미구속**: A 창 하단 여유 3.3%는 구속하지 않았고,
  blind spot `[3.0]`은 라벨·λ\*에 영향 0(반전 시험 확인).
  "offered 3.0에서 포화가 시작됐다"는 그 셀이 실제로는 배수만으로
  설명돼 **쓸 수 없다**(λ0R-4).
- **λ5C-6 미해소**: 측정값은 정의역 안이지만(로그 위치 36.4%)
  **B 타당범위 로그폭 56.7%에서는 이 설계가 라벨을 못 낸다** —
  이번 결과가 이 한계를 해소하지 않는다.
- **arm 비동일성**: manifest 25항목 중 정확히 1개
  (`multiplexing_mixin.py`, 커밋 `a9cd8dd`)가 907959와 다르다.
  무영향 근거는 **코드 독해 추론이며 측정된 null이 아니다**
  (양 arm 공통 메모리 계측이 이 job에서 발화하지 않음, 직접
  확인) — 교차 일관성 문장에 병기 필수(λ0R-2 조건 (iv)).
- **절차**: "제출 전 커밋" 선행조건이 어겨졌다(F6
  `uncommitted=31`, 커밋은 job 시작 14분 뒤 `b60dc64`). 실행
  바이트가 현 HEAD와 11/11 일치·mtime 전부 제출 이전·감사자
  재실행이 전 아티팩트 바이트 재현 ⇒ **死因 아님, 절차
  caveat**(λ0R-10) — 다음 OVERRIDE 한 창 규율에 커밋 단계를
  명시할 것.
- **6번째 자유 표면(λ0R-6)**: `ACH_HI`를 **조이는** 방향(>0.9816
  ⇒ B 상실, >0.9889 ⇒ A도)이 등록 변이 목록에 없다(등록 변이는
  느슨하게 하는 방향만 시험) — 실행 시점 재량은 아니다(selftest
  rc≠0).
- **비하중 정정**: rev5 판정서 `N_SHA=15`는 실제 16;
  `plan.json`이 여전히 `"rev": 4`(읽는 코드 없음, 판정 무관).

## D. P2 블로커 상태 갱신

| 블로커 | 상태 |
|---|---|
| ① λ0 `NO-GO` | ★**해소** |
| ② W4 λ\* 실측 부재 | **부분 이동 · 미해소**(4겹) |
| ③ 게이트 #6 | **불변 — 해소 안 됨** |

②를 "닫혔다"고 쓸 수 없는 4겹 이유: (1) ★**정의가 틀렸다** —
측정된 것은 `throughput_saturation`, 요구되는 것은
`slo_sustainable`(하네스 자신이 `lambda_star.py` docstring에
명기 — declaring throughput_saturation은 게이트 #6을 닫지
않는다) ⇒ **②와 ③은 같은 블로커의 두 이름**. (2) Q3(상호
배타) 그대로 — λ\*=0.697→decode phase 0.23×, λ\*=3.053→
prefill phase 4.38×. (3) ★위 "B" 절이 새 장벽 — 두 λ\*는
**서로 다른 실현 분할**에서 측정됐다(A 4.5–8.6% / B 90.7%
D44) ⇒ "한 arm의 두 phase 용량"조차 성립하지 않는다. (4)
NPC-I — B4 포화 상한 미측정.

**실제로 움직인 만큼**: W3+W4가 요구하는 정확히 그 두 shape에
대해 `lambda_star.py`의 `source="measured"`+
`definition="throughput_saturation"` 항목을 거짓말 없이 채울 수
있게 됐다(그 전엔 `generate_campaign.sh`가 `exit 2`). 부수:
폐기된 옛 기본값 4 req/s는 측정 대비 A 1.310×·B 5.737× 과대.

**P2 착수 남는 조건 3개**: (a) λ\*_SLO 두 shape 직접 측정(게이트
#6) (b) λ\*를 실현 분할 혼합비와 함께 등록 +
`PDMUX_STICKY_PARTITION=1` 대조 (c) B4 포함 시 B4의 λ\* 또는
정규화 논증. ★**금지**:
`PDMUX_ALLOW_UNMEASURED_LAMBDA_STAR=1`로 ③ 우회(하네스는
허용하나 게이트 #6을 닫지 않는다).

## E. GPU 장부 갱신 (정확히)

```
λ0 트랙(신설, 이 트랙 최초 GPU 지출)
  job 908623 (sacct 01:23:29=5,009s, gpu38)   1.391389 GPU-h
  ────────────────────────────────────────────
  누적                                         1.391389 GPU-h
  (등록 보통 1.348의 1.032×, 최악 코너 3.509의 0.397×)

R2 correctness 트랙 (불변)
  등록 1.170278 + 등록 밖 0.152500 = 1.322778 GPU-h

longctx_conflict 트랙 15.42 GPU-h — 별개 장부, 불변.
```

## F. 신규 방법론 게이트 6건 (#248–253)

전부 **새 성능 판정 0건**. 대응 `CONSENSUS.md` §3
항목268–273(전문은 그쪽 참조):

- **#248**: "fixed D44에서 측정했다"는 target이 아니라 realized
  혼합비로 검증하라 — λ\*(A)를 공급한 셀의 D44 점유가
  8.6–17.2%였고 지배 division은 `(0,108)`이었다(정본 §1-25·
  §1-26(B)의 재현, Stage 0 "D108 앵커가 실은 16 SM"과 같은
  계열).
- **#249**: 두 수가 잡음보다 작게 일치하는 것은 정밀도 증거가
  아니다 — λ\*(B) vs λ_inf(B) 차 0.2377% < seed 재현차 0.310%,
  게다가 상한/피상한 순서가 깨지면 그것은 반증 방향의 증거다.
- **#250**: 실격은 자격 조항이므로 사후 독립 측정으로 충족되지
  않는다(N-8·λ5C-1 재확인).
- **#251**: "제출 전 커밋"을 OVERRIDE의 한 창 규율에 명시적으로
  넣어라 — 이번 회차 커밋이 job 시작 14분 뒤였고 F6가
  `uncommitted=31`을 기록했다.
- **#252**: 감사자 처방도 다중 노브를 움직일 수 있다 — 감사자가
  자기 처방(E3, `--max-running-requests` 48↔96 대조)을 스스로
  철회했다: 이 빌드에서 `max_mamba_cache_size =
  max_running_requests`(정본 §3 항목23)라 네 노브가 동시에
  움직인다.
- **#253**: 문턱을 조이는 방향도 변이 목록에 넣어라(λ0R-6:
  `ACH_HI` 상향이 미등록 자유 표면).

## G. 정본 반영

`CONSENSUS.md` rev77→**rev78**(§3 항목268–273 신설[게이트
#248–253]). `reports/paper/CLAIM_EVIDENCE_MATRIX.md` Claim D
행에 job 908623 문단 추가, `EXPERIMENT_ROADMAP.md` "P2" 절에
job 908623 문단 + P2 블로커 상태(①해소·②③불변) 갱신.
`MEMORY.md` 포인터 갱신, `memory/slo-aware-scheduling-track.md`에
신규 `## 2026-09-14(5)` 절, `memory/
deconfound-measurement-lessons.md`에 항목246–251 신설(topic
파일 내부 순번 — `CONSENSUS.md` §3 항목과는 별개 축). **커밋
금지**(핸드오프 커밋에 함께 묶인다).

**overclaim 금지**: (1) 성능 판정 0건 — arm 비교 0(전 셀이 같은
arm, 변한 것은 offered rate·shape뿐), shape A vs B 비교 금지
(in·out 동시 변경), policy 레버 0, hybrid/generic 미발화 ⇒
H-Policy 무관, true-dual 미발화 ⇒ H-Architecture 무관. (2)
"λ\*를 측정했으므로 W3/W4 부하 라벨이 참이 됐다"는 여전히 쓸 수
없다(λ5C-8, 게이트 #6 불변). (3) P2 블로커 ①의 해소는 **P2
착수 승인이 아니다** — ②③이 남아 있고 ②는 "부분 이동"일 뿐
새 장벽(§B의 실현 분할 불일치)이 추가됐다. (4) Claim D/E 등급·
HE0·layer-type 死·정책 순위·stake #1·게이트#13/#16·C2 인용정지
전부 재도출 금지.

이전: 2026-09-14(4)(doc-steward — ★job 908534(D-none 대조
회차) 결과 감사 등재[`PLAUSIBLE(조건부)` → **`CONFIRMED(scoped)`**
승급, 인용 가능 문장은 판정서 §3 (A)(B) 두 개뿐, "지배 원인"
서술은 감사가 명시 거부, 승급으로 안 닫히는 것 11항 동시 등재] +
λ0 rev5 규칙층 감사 등재[**`GO-with-caveats`**(死因 0), 신규
발견 λ5C-1(앵커 자격 술어가 이 shape/cap에서 구조적으로 도달
불가), GPU 신규 지출 0·**아직 미제출**] + GPU 장부 갱신[R2
correctness 트랙 1.107500→**1.322778 GPU-h**(등록
1.170278/등록 밖 0.152500), 등록 상한 대비 +9.28%] + 4번째
미등록 축 갱신(gpu38→43→40→41, 게이트 #233 여전히 열림) +
신규 방법론 게이트 7건[#241–247, `CONSENSUS.md` §3
항목261–267]). 새 성능 판정 0건·**Claim D 선결 0건 폐쇄·P2
블로커 3개 불변**(λ0·W4 λ\* 실측 부재·게이트 #6)·Claim D/E
등급 불변(둘 다 미검증)·HE0·layer-type 死·정책 순위·stake
#1·게이트#13/#16·C2 인용정지 전부 불변. `CONSENSUS.md`
rev76→**rev77**(판단 근거는 아래 "G. 정본 반영" 참조). **커밋
금지**(핸드오프 커밋에 함께 묶인다).

## A. job 908534 결과 — "수리가 907959 OOM 원인" `PLAUSIBLE(조건부)` → **`CONFIRMED(scoped)`** 승급

**job 908534**(gpu41, `GPU-dd853b73-…`, 0.215278 GPU-h[`sacct`
00:12:55=775s], D-none 대조 회차) = 등록 튜플에서 같은 job 안에
`PDMUX_WORKER_GRAD_GUARD=none`(DN) arm을 추가한 회차,
**`VERDICT PASS`**(무수정 채점기 재실행 `verdict.txt` 0 byte
차이). 사전등록 계보: rev1(`b5d1a244…`) `NO-GO` → 규칙층
판정서(`VERDICT_dnone_rules_2026-09-14.md`, `530ac31b…`) →
**rev2**(`PREREG_DNONE_2026-09-14_rev2.md`, `11b63095…`) +
구속 追記(`PREREG_DNONE_rev2_ADDENDUM_2026-09-14.md`,
`5a0fbde8…`) → rev2 규칙층 판정(`VERDICT_dnone_rev2_2026-09-14.md`,
`ec8502e7…`, `GO-with-caveats` 死因 0) → 범위 한정
OVERRIDE(`OVERRIDE_DNONE_SUBMIT_2026-09-14.md`, `c0fa107c…`,
사용자 명시 승인, 이 job으로 소진) → 실행. ★**등급 판정의
정본은 결과 감사
`VERDICT_result_908534_2026-09-14.md`(`e54fcd90…`)** — 원자료
`RESULT_908534_RAW_2026-09-14.md`(`cb833630…`)·F-n2
`FN2_PEAK_ANALYSIS_908534_2026-09-14.md`(`845abb69…`)·채점서
`SCORING_908534_2026-09-14.md`(`aa8e49b5…`)+정정 追記
`SCORING_908534_CORRECTIONS_2026-09-14.md`(`6e1259f0…`) 전부
독립 재검증·재계산됨.

★★★**정본에 쓸 수 있는 문장은 아래 (A)·(B) 두 개뿐이다(전사,
요약 아님)**:

> **★인용 가능한 유일한 문장 (A)** — *"(model
> `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base` =
> `NemotronHForCausalLM` 56층 `n_groups=8` · backend
> flashinfer(0.6.10이라고 보고한 설치본) · ctx 16384 ·
> `mem-fraction-static 0.82` · `max-running-requests 48` ·
> `--disable-radix-cache --chunked-prefill-size -1
> --disable-overlap-schedule --random-seed 1` · split fixed
> **D44**(realized 64/44 SM, green idx 4) · **cudagraph
> decode-ON / prefill-OFF** · architecture `true_dual` · 엔진
> src 매니페스트 `e2a97b42…` · 하네스 `ab55c07c…` · 채점기
> `ec355e17…` · client `95e10b49…` · seed 1 · A100-SXM4-80GB
> 1장, node gpu41, GPU-UUID `dd853b73…`, job 908534)에서,
> **같은 job 안에서 `PDMUX_WORKER_GRAD_GUARD` 축만**
> `inference_mode` → `none`으로 이동시키면 job 907959의 OOM이
> **같은 39-배치 도착열 · 같은 rung(6245 완주 후 다음 배치) ·
> 635 byte 바이트 동일한 OOM 원문 · 24 프레임 중 22개가 바이트
> 동일하고 남은 2개는 같은 파일·같은 함수의 줄번호만 다른
> 스택**으로 재현된다. 같은 job의 수리 arm 2 boot은 같은
> 도착열을 41/41 완주했다."*

> **★인용 가능한 유일한 문장 (B)** — *"이 튜플에서 **grad-guard
> 축 단독 이동이 OOM의 발생과 부재를 양방향으로 결정한다**:
> 가드 없는 조건 **3/3 사망**(907959 TD1·TD2 + 908534 DN1,
> OOM 원문 바이트 동일), 가드 있는 조건 **4/4 완주**(908179
> TD1·TD2 + 908534 TD1·TD2). 토큰·seq 매칭된 마지막 공통
> 배치(`#new-token 6245`, `#new-seq 8`)에서 peak 차 =
> **`+8,107,034,112 B` (= 7.5503 GiB)**, 같은-설정 boot 산포
> 천장 `4,494,336 B`의 **1,804배**; 동반 상태까지 완전 일치하는
> 배치(`#new-token 3241`)에서도 **`+5,013,701,632 B` (=
> 4.6694 GiB)**."*

★★★**"지배 원인"이라는 표현은 감사가 명시 거부했다** — 측정된
것은 이 튜플 안에서의 **충분성 + 필요성(양방향 결정)** 뿐이고
분산 분해·기여도 비율은 측정되지 않았다. "지배 원인" 대신 위
문장 (B)를 쓴다.

## B. 승급으로 닫히지 않는 것 (11항, 판정서 §3 목록 그대로 등재)

1. **기전 미측정** — 이 회차 데이터가 단조 층별 축적을
   **지지하지 않는다**(chunk 40 짝지은 Δ=+0.0613 GiB, 다른
   chunk는 +1.26–+6.46 GiB로 톱니형).
2. **`R(T)` 잔류 모형은 같은 데이터가 반증**(Δ/R 0.772–89.969)
   — 어떤 크기 주장도 이 모형에 기대 쓸 수 없다.
3. **guard-none 축 n=1** — 구간추정(±/CI/p/σ) 금지, 계측기의
   guard 축 해상력 양성대조 없음.
4. **907959 자신의 가드 실현은 미측정**(`a9cd8dd^` 코드 사실 +
   908534 thread-local성 실측으로부터의 **연역**).
5. **`a9cd8dd` 전체의 대조가 아니다**(DN-9) — 단 OOM 원문 635
   byte 동일로 계측의 할당 중성성이 1회 실증(n=1).
6. **하네스 축 미폐쇄**(sha 1개 이동) — 단 within-job DN↔TD
   비교에는 교락하지 않는다.
7. **모델·백엔드·ctx·split 일반화 없음** — Zamba2/Falcon-H1/
   Granite-4의 "n_groups=1 구조적 면역"은 코드 독해이며
   미측정(RR-3 소급 금지 유효·강화: 과거 Zamba2 회차 "OOM
   없음"은 오버헤드·잔류 크기가 0이 아니라 **미측정**이라는
   뜻).
8. **크래시 배치 번호는 cliff-민감**(여유 143 MiB, 재현 시 다른
   rung일 수 있다).
9. **Claim D 선결 0건 폐쇄 · P2 블로커 3개 불변**(λ0 `NO-GO`·
   W4 λ\* 실측 부재·게이트 #6).
10. **H-Architecture / H-Policy 어느 쪽도 진전 없음** —
    true-dual **구현 결함의 수리 확인**이며 dual-worker 구조의
    이득 증거가 아니다. **성능 판정 0건.**
11. **확정 결론 전부 불변**(HE0·layer-type 死·정책 순위·stake
    #1·게이트 #13/#16·C2 인용정지).

## C. λ0 rev5 — `GO-with-caveats`(死因 0), 5차 감사

**대상** `PREREG_LAMBDA0_REV5_2026-09-14.md`(`8367b6ec…`) +
구속 追記 `PREREG_LAMBDA0_REV5_ADDENDUM_2026-09-14.md`
(`9af1c141…`) — 계보 rev1–rev4 4연속 `NO-GO` → 이 rev5 **死因
0건**, 판정서 `VERDICT_lambda0_rev5_2026-09-14.md`(`33dac116…`).
★**GPU 신규 지출 0**(rev1–rev5 전부 규칙층 감사·변이 시험만,
CPU만) — **이 사전등록은 아직 제출되지 않았다**(사용자가
0단계 제출을 승인했으나[최악 3.509 GPU-h], 제출은
engine-porter의 코드 권고 5건[λ5A-2·3·4·9·11] 반영 이후로
예정).

★★★**이번 회차 최대 발견 — λ5C-1(필수병기)**:

> **I3b의 셀별 `#running-req` 최댓값 2는 "부하가 모자랐다"는
> 뜻이 아니다.** 같은 레코드가 실현 클라이언트 동시성
> **57.43/64**·**median TTFT 90.69 s**를 적는다(엔진은 셀
> 내내 backlog 상태였다). prefill 지배 shape에서 엔진의
> *decode* 배치는 구성상 ~2다 — `max_prefill_tokens=16384`가
> 8192-토큰 요청 **2개**를 prefill 배치 상한으로 만들고, KV
> 예산(`max_total_num_tokens=2721290`)은 구속하지 않았다.
> probe C도 같은 shape 계열에서 concurrency 2.10을 독립적으로
> 읽었다. ⇒ **`#running-req ≥ 48`은 이 구성의 shape B에서 어떤
> 부하로도 도달 불가**다. **(a)** 셀 B의 F5 실격은 규칙의
> 올바른 적용이나 "셀이 포화하지 못했다"로 바꿔 쓸 수 없고,
> **(b)** `ANCHORED` 분지는 이 shape/cap의 모든 미래 instrument
> run에서 **구조적으로 사용 불가**(→ "I3를 다시 재서 앵커를
> 살린다"는 경로가 아니다), **(c)** 미달 원인은 **미확정**으로
> 남는다.

그 외 caveat: **rev4 판정서의 `I3b = 11/11/2`는 인용
금지**(`splitlines()` 산물, 생산자 `wc -l` 체계로는 **2/2/2**,
bare `\r` 116개) — 판정 방향은 불변. `FALLBACK`이 **1차
등록**이고 `ANCHORED`는 반사실. 분지 전환이 shape B 13점 중
5점·A 12점 중 3점의 지도 라벨을 뒤집는다. 예산 최악 코너
**3.509 GPU-h**(요청 3.60 ≤ 벽시계 4.50h, 여유 0.991h). ★**게이트
#6은 어떤 결과에서도 닫히지 않는다**(λ5C-8) — probe C d44 4셀
정본 술어 재채점 goodput 53.8%/5.8%/0.8%, **ITL 조건은 4셀
전부 100% 통과** ⇒ 구속은 전적으로 TTFT. λ5C-5(A 창 하단 여유
3.3%)·λ5C-6(B 창은 등록 타당범위 로그폭의 43%만 덮는다).

## D. GPU 장부 갱신 (정확히)

```
R2 correctness 트랙 (sacct 기준)
  이전 누적                                        1.107500 GPU-h
  + job 908534 (sacct 00:12:55=775s, gpu41)        0.215278 GPU-h
  ────────────────────────────────────────────────────────────
  등록 (0.955000 + 0.215278)                       1.170278 GPU-h
  등록 밖 (908020 불변)                             0.152500 GPU-h
  총계                                              1.322778 GPU-h

초과 공시(분모 구분 필수, 결과 감사 §4 정정):
  등록 상한 0.197 GPU-h(=709.2s) 대비   +9.28%
  예보 708s 대비                        +9.46%

λ0 트랙: GPU 신규 지출 0(rev1–rev5 전부 규칙층 감사, 미실행) — 별개 장부.
longctx_conflict 트랙 15.42 GPU-h — 별개 장부, 불변.
```

## E. 4번째 축 갱신 — gpu38→43→40→**41**, 게이트 #233 여전히 열림

노드/물리 GPU 축이 907959(gpu38)→908179(gpu40)에 이어 이번
908534에서 **gpu41**로 다시 이동했다(경로: gpu38→43→40→41).
within-job DN↔TD 비교(문장 (A)(B)의 근거)는 같은 job·같은
노드이므로 이 축에 교락하지 않지만, ★**게이트 #233은 닫히지
않았다** — cross-job 비교는 여전히 노드 교락 상태이고, 채점서의
"게이트 #233의 목적 달성" 서술은 결과 감사가 **과대**로
판정했다.

## F. 신규 방법론 게이트 7건 (#241–247)

전부 **새 성능 판정 0건**. 대응 `CONSENSUS.md` §3
항목261–267(전문은 그쪽 참조):

- **#241**(D-none 결과 감사): 승계 술어 목록은 캠페인 설계가
  바뀌면 문자 그대로 거짓이 된다 — SCOPE (4)·(7)이 문자
  그대로 거짓이었고(사후 자유도는 0), rev2 본문·구속 追記·
  규칙층 판정서 2건 전부가 놓쳤다(rev1을 죽인 사인 DNR-2와
  같은 형태).
- **#242**: "보수적 변경"이라는 자기 점검은 도달성을 보장하지
  않는다 — 구속 追記 A2가 트리거 문자열을 정의와 모순되게
  추가해 강한 독법에서 `RECOVERED-*`를 구조적으로 도달 불가로
  만든다.
- **#243**: 다른 캠페인의 문턱을 역할 반대로 끌어오지 마라 —
  F-n2의 "0.3·R 문턱 충족"은 rev2에 없는 문턱이고 θ=0.3825는
  다른 캠페인(`PREREG_RERUN:787`)에서 반대 역할로 쓰인 값이다.
- **#244**: 로그 인쇄 시점과 계측 epoch을 섞으면 토큰 비매칭이
  된다 — 단 이번 정정("DN1이 ep39 도중 사망"은 거짓, 실은
  완주)은 결과를 **강화**했다(창 길이 비대칭 교락 소멸).
- **#245**: 자기보고 수치를 저장소 전역 단조 카운터에 묶지
  마라 — 전체 CPU 스위트 수가 한 세션에 네 번(295→641→645→
  681→697) 낡았다. 진리원 선택이 틀렸다(수리는 열린 항목).
- **#246**: 동시 편집 중인 트랙의 중간 상태를 게이트 사실로
  등재하면 거짓이 된다 — "제출 시점 696/4 실패"는 제출 창
  실측(697/697 OK)과 다르다. 세 수치(681/1·696/4·697/0)를
  날짜 붙은 계열로 병기해야 하고, 제출 창 실측은 제3자 재현
  불가.
- **#247**(λ0 rev5, λ5C-1): 앵커 자격 술어가 그 shape/cap에서
  **구조적으로 도달 불가**일 수 있다 — "부하가 모자랐다"로
  오독하지 말고 재시도를 경로로 등록하지 마라.

## G. 정본 반영

`CONSENSUS.md` rev76→**rev77**(§3 항목261–267 신설[게이트
#241–247]). `reports/paper/CLAIM_EVIDENCE_MATRIX.md` Claim D
행에 job 908534 문단 추가, `EXPERIMENT_ROADMAP.md` "P2" 절에
job 908534 문단 + λ0 rev5 상태 갱신. `MEMORY.md` 포인터 갱신,
`memory/slo-aware-scheduling-track.md`에 신규 `## 2026-09-14(4)`
절, `memory/deconfound-measurement-lessons.md`에 항목239–245
신설(topic 파일 내부 순번 — `CONSENSUS.md` §3 항목과는 별개
축), `memory/engine-port-p0-triage.md`에 grad-guard 결함의
인과 확정(스코프 문장 그대로) 갱신. **커밋 금지**(핸드오프
커밋에 함께 묶인다).

**overclaim 금지**: (1) 인용 가능한 문장은 위 "A" 절의 (A)(B)
두 개뿐이며 "지배 원인"이라는 표현은 승인되지 않았다. (2)
승급은 "B" 절의 11항을 닫지 않는다 — 특히 Claim D 선결 0건
폐쇄·P2 블로커 3개 불변·성능 판정 0건. (3) λ0 rev5
`GO-with-caveats`는 **실행 승인이며 게이트 #6을 닫은 것이
아니다** — 아직 GPU가 지출되지 않았다. (4) 게이트 #233은 이
회차로도 닫히지 않았다.

이전: 2026-09-14(3)(doc-steward — 정본 인용 부패 4건 수정
[engine-porter 등록성 검사 발견: `generate_campaign.sh` 경로
오기+주장 자체 반증 1건, `r2_eval.sbatch` 줄 드리프트 2건 —
전부 [HIST]+dated 追記로 처리, 삭제 없음] + 신규 방법론 게이트
3건 등재[#238–240, 출처=engine-porter 측정: 정규식 엔진
불일치·git-HEAD 자기부정 대조·presubmit 비-read-only] +
presubmit 차단 2건을 현재 블로커로 등재[다음 R2 job 제출은
범위 한정 OVERRIDE 문서+사용자 승인 선행조건] + M4R `sm_match`
"why" 부분정정 이후 미재유도 플래그). 새 성능 판정 0건·Claim
D/E 등급 불변(둘 다 미검증)·HE0·정책 순위·stake #1·게이트
#13/#16·C2 인용정지·Zamba2/triton 동결 전부 불변. `CONSENSUS.md`
rev75→**rev76**(판단 근거는 아래 "D. 정본 반영" 참조). **커밋
금지**(핸드오프 커밋에 함께 묶인다).

## A. 정본 인용 부패 4건 수정 (engine-porter 등록성 검사에서 발견)

engine-porter가 `check_line_citations.py` 등록 가능성을 검사하다
발견한 4건 — 전부 **현재 트리에 대해 거짓**이어서 그대로 스냅샷
등록할 수 없었다(등록하면 검사기가 거짓 주장을 영구 인증하게
된다 — 게이트#96의 항등식-레지스트리 실패 재발). 삭제 없이
[HIST]+dated 追記로 처리했다:

1. **`generate_campaign.sh:10`**(`PROJECT_STATUS.md`·
   `CLAIM_EVIDENCE_MATRIX.md`·`EXPERIMENT_ROADMAP.md` 3곳)의
   "`sustainable_rate` 기본값 `4`가 미측정" — **주장 자체가
   이제 거짓**이다: W4 구현(커밋 `8507cee`)이 그 기본값을
   fail-closed(`exit 2`)로 삭제했다. 경로 오기(`benchmarks/
   pdmux_eval/` 아래가 아니라 `scripts/r2_eval/` 아래)도 함께
   정정. **λ\* 자체가 미측정이라는 별개 사실**(게이트 #6·λ0
   `NO-GO`)은 불변이며 지우지 않았다.
2. **`r2_eval.sbatch:26`**(`PDMUX_ENGINE_COMMIT` export, 3곳
   동일 인용) — 커밋 `09a8075`로 `:69-70`으로 이동. 인용된
   결함(hybrid profile 호환 검사가 `engine_commit`을 엄격
   일치에 포함)은 이미 커밋 `3f188bb`로 해소돼 각 문서에
   기록돼 있었으나, 그 해소 문단이 이 하위 인용까지는 갱신하지
   않고 있었다.
3. **`r2_eval.sbatch:100`**(`PROJECT_STATUS.md`만, 유일한
   `context_limit` 호출 지점) — 커밋 `87213a9`+W4로 `:180`으로
   이동. ★재검증 플래그(판정 아님): `r2_correctness.sbatch`가
   이 패스 시점 engine-porter의 별도 진행 중 개정 대상이고 그
   파일에 이미 `context_limit` 호출(`:244`)이 나타나 있어,
   "유일한 호출 지점"·"ctx 4096 리터럴" 서술은 그 개정 완료
   후 재확인이 필요할 수 있다(이 패스는 `r2_correctness.sbatch`
   에 손대지 않았다).
4. **범위 제한이 부패를 숨겼다**: 위 3건이 게이트에 안 잡힌
   이유는 `PROJECT_STATUS.md`·`CLAIM_EVIDENCE_MATRIX.md`·
   `EXPERIMENT_ROADMAP.md`의 line-citation `__scope__`가
   `controller.py`/`profile.py`(+`workloads.py`) 패턴만
   등록해 `generate_campaign.sh`·`r2_eval.sbatch` 인용은
   원래부터 검사 대상 밖이었기 때문이다 — 이 사실 자체를
   신규 게이트로 등재한다(아래 "B" 참조).

## B. 신규 방법론 게이트 3건 등재 (#238–240, 출처=engine-porter 측정)

전부 **새 성능 판정 0건**. 요지(전문은 "방법론 게이트" 절):

- **#238**: 감사자의 교정 처방(정규식) 자체가 실행 엔진(GNU
  grep -E vs PCRE)에 따라 12/12 또는 4/12 또는 0/12로 갈리는
  실효 항등식이었다 — 정규식은 실행 엔진까지 고정해 검증하라.
  검증된 대체형 확보. 대응 `CONSENSUS.md` §3 항목258.
- **#239**: 개정-전 대조가 `git show HEAD:…`에 걸려 있어 W4
  커밋 순간 `HEAD`가 개정후 파일이 돼 대조가 자살(2 에러) —
  세션 시작 시점 전체 스위트가 이미 red(645 tests, errors=2)
  였음도 함께 등재. 대응 `CONSENSUS.md` §3 항목259.
- **#240**: `presubmit.py`가 부르는 `design_reachability.py`가
  각 spec의 `out` 경로를 매 실행마다 다시 쓴다 — 규율 검사
  도구를 read-only로 전제하지 마라(이번 4회는 바이트 동일
  확인됐으나 과거 다른 세션 미커밋 작업 파괴 전례 있음). 대응
  `CONSENSUS.md` §3 항목260.

## C. presubmit 차단 2건 — 현재 블로커로 등재 (결정은 사용자 몫)

engine-porter 판정: `m4r_confinement/reachability_spec.json`
(`SINGLE_LABEL_FORCED`)와 `tc1_model_attrib/reach_spec_rev3_A.
json`(`RESTRICTIONS_INERT`)은 **stale이 아니라 살아 있는 설계
결함**이고 후속 spec이 없어 `superseded_by`로 닫을 수 없다.
⇒ **★현재 블로커로 등재**: 다음 R2 job 제출은 이 2건 때문에
**범위 한정 OVERRIDE 문서 + 사용자 명시 승인을 선행 조건으로
한다** — 전례는 두 번 다 그 형태였다(`results/cp_baseline/
OVERRIDE_VPROBE_SUBMIT_2026-09-01.md:10-11,40,42`·
`OVERRIDE_P1_SUBMIT_2026-08-28.md:44,81`, 둘 다 도구 차단범위
축소를 명시 금지). doc-steward는 OVERRIDE 문서를 작성하지
않는다(사용자 승인 전).

부수: M4R spec의 `sm_match` 제약 `why` 문자열("28 files/39,849
intervals, 0 cases")이 `PROJECT_STATUS.md:8665` 追記
(2026-09-09)로 **⟸ 방향에서만 참**(반례 1,643건, 35.8%)으로
부분 정정됐는데, **아무도 이 제약을 재유도하지 않았다** — 이
사실을 여기 병기한다(제약 재유도는 이 패스 범위 밖).

## D. 정본 반영

`CONSENSUS.md` rev75→**rev76**(§3 항목258–260 신설[게이트
#238–240] + 인용 부패 4건 [HIST]/정정 반영 — 항목 신설 없이
해당 문서 자체[PROJECT_STATUS·CLAIM_EVIDENCE_MATRIX·
EXPERIMENT_ROADMAP]에 직접 표시). `reports/paper/{CLAIM_
EVIDENCE_MATRIX,EXPERIMENT_ROADMAP}.md` 4곳 갱신(위 "A" 목록).
`MEMORY.md` 포인터 갱신, `memory/deconfound-measurement-
lessons.md`에 항목236–238 신설(topic 파일 내부 순번 —
`CONSENSUS.md` §3 항목과는 별개 축). **커밋 금지**(핸드오프
커밋에 함께 묶인다).

GPU 장부: 이 패스 **신규 GPU 지출 0**(문서 등재·인용 정정
작업만, engine-porter의 측정은 각자 자기 GPU 0 보고에 따름).
R2 correctness 트랙 누적 **1.107500 GPU-h**(등록 0.955000/
등록 밖 0.152500) 불변. `longctx_conflict` 15.42 GPU-h 불변.

이전: 2026-09-14(2)(doc-steward — ★job 908179 결과 감사
등재[`PASS`, C1 스코프 술어 (1)–(11) 전부 참, "수리가 OOM을
고쳤다" = `PLAUSIBLE(조건부)`·`CONFIRMED` 아님] + Claim D 선결
#5는 새로 닫힌 게 아니라 스코프만 두 번째 (모델,백엔드,ctx)
쌍으로 확장 + GPU 장부 드리프트 정정[sacct 기준으로 통일] +
4번째 미등록 축(노드/물리 GPU) 발견 + 신규 게이트 6건
[#232–237]). 새 성능 판정 0건·Claim D/E 등급 불변(둘 다
미검증)·HE0·정책 순위·stake #1·게이트#13/#16·C2 인용정지·
Zamba2/triton 동결 전부 불변. **PASS는 P2 착수를 승인하지
않는다**(NP-8 — 남은 블로커: λ0 `NO-GO`·W4·게이트 #6).
`CONSENSUS.md` rev74→rev75(판단 근거는 위 2026-09-14(2)
"J. 정본 반영" 참조).

## A. job 908179 결과 — `VERDICT PASS`, 등록 튜플 최초 실행

**job 908179**(gpu40, `GPU-94efac99-…`, 0.165556 GPU-h[`sacct`
00:09:56], commit `83d8cb9d`) = 등록 튜플(Nano-9B-v2-Base/
flashinfer/ctx16384/D44)의 최초 실행, **`VERDICT PASS`** —
무수정 채점기 재실행으로 **비트 단위 재현**(`verdict.txt`·
`r2_correctness_report.json` 양쪽 diff 0바이트). 결과 감사
(`.../audit_908179_2026-09-14/VERDICT.md`, claims-auditor)
판정: **C1 스코프 일치 술어 (1)–(11) 전수 재검증 — 전부
참** ⇒ `NO_VERDICT_SCOPE`가 아니라 **등록된 실험**(908020의
등록 밖 사고가 이 회차에서는 재발하지 않음, 스코프 가드가
실제로 작동). ★단 (9)·(11)의 등록된 한계는 유효(RA4-3·
RA3-8 — "GPU 0: " 확인은 src 청결의 충분조건일 뿐 필요조건
아님, 6/6 나머지 축은 여전히 기본값 우연 일치에 의존, `C3`
예산 0).

## B. 예보 채점 (사전등록 §6, rerun rev4)

| 예보 | 결과 |
|---|---|
| F-a1(4채널 무크래시) | **참** |
| F-a2a/F-a2b(기록 전용) | 충족(성능 주장 근거로 인용 금지) |
| F-b1(기울기 축) | **대역 A′**(`S=0.003211` MiB/token, θ=0.3825 — 부호 반대·크기 397배 작음) |
| F-b2(10,125 배치 메모리) | **대역 A**(`Δ=−48.37 MiB`, ε=1.00 GiB, `D` 미발화) |
| F-c(게이트 위생) | 충족 |
| F-d(가드 실현) | 충족(단 RR-11/12·A8179-3 제한 유효 — 스냅샷 단위, task 전수 아님) |
| F-e/E5(inference-tensor 예외) | 미발화(원시 `RuntimeError` 0건) |

## C. ★★"수리가 OOM을 고쳤다" = `PLAUSIBLE(조건부)` — `CONFIRMED` 아님

**TD 2/2가 job 907959에서 자기를 죽였던 바로 그 14-seq/
10,125-token 배치를 완주**했다(`srv_TD1.log:2579`·
`srv_TD2.log:2580`). 찬성 증거 6건(완주·헤드룸 바이트 동일·
피크 0차·계측기 5MiB 해상력 실증·가드 실현·나머지 축에
원인 없음) — 그러나 **결정적 arm(`PDMUX_WORKER_GRAD_
GUARD=none` 대조)이 없고**, F-a1은 원인 무차별(이 트랙 4 job
중 1건은 OOM 0건인데도 F-a1이 거짓이 되는 구조), 축 3개
(mixin sha·하네스 sha·계측 플래그)가 동시 이동해 이 job
단독으로는 귀속이 안 닫힌다. **반증 시도 4건 전부 실패했으나
반증 실패는 확증이 아니다.** `CONFIRMED(scoped)`로 올리려면
`none` 대조 arm 1회가 필요하고, ★그 실험은 **무수정으로는
실행 불가**(하네스가 `PDMUX_*`를 전부 unset·채점기가
`startswith("TD")`로만 arm을 갈라 `none` boot을 넣으면
설계상 무조건 `FAIL`) — 노브 추가 + "예측된 FAIL은 정보이며
재제출 금지 대상 아님"을 등록하는 새 사전등록이 선행돼야
한다.

## D. 메모리 축 — 3자 완전 동일은 측정, 산포는 arm 아닌 boot 효과

`peak(TD1) = peak(TD2) = peak(L2) = 73,342,324,736 B`(**차 0
B**), `peak(L1)`만 `+96.73 MiB`. 진단서가 예측한 "수리 前 TD가
12.61 GiB 초과"의 **부재가 측정됐다**(아티팩트 가설 7종 전수
배제, 특히 **9/36 epoch에서 완전 arm 분리[+5.17…+20.08
MiB]가 관측**돼 계측기가 5 MiB 수준에서 arm 차이를 해상함이
**같은 job 안에서 실증**됨 — 이것이 없었다면 "0 B 일치"는
계측기가 눈이 먼 결과일 수도 있었다).

★**Δ는 전량이 L1의 505 토큰 더 큰 배치(10,630 vs 10,125)에서
나온다** — arm 차이가 아니다(`peak(L1)−peak(L2)`의 절반이
정확히 `Δ`). ★**L1의 이탈은 arm 효과가 아니라 boot 수준
산포**다 — 3 job 교차 검증: 907959는 L2가, 908020(등록
밖)은 L1·TD1이, 908179는 L1이 이탈했다(arm도 위치도 고정되지
않음). 인용 금지 **A8179-1**("Δ<0 ⇒ TD가 메모리를 덜 쓴다"는
거짓)·**A8179-2**("L1 peak 높음=legacy arm 성질"은 거짓,
배치열 분할이 arm 경계를 가로지름).

## E. Claim D 선결 #5 — 새로 닫힌 게 아니라 스코프만 확장

PASS가 인증하는 **한 문장**(RA4-7 계열): *"위 튜플에서
`PDMUX_TRUE_DUAL_WORKER=1` 경로와 legacy 루프가 S층·O층에서
같은 토큰 id를 내고 그동안 cudagraph가 두 arm 모두 켜진 채
유지된다."* ★cudagraph는 **decode 한정**(prefill 41/41줄
`cuda graph: False`, 양 arm). **판정**: 선결 #5는 **이미
(Zamba2-2.7B, triton) 한정으로 닫혀 있던 술어가 두 번째
(모델, 백엔드, ctx) 쌍으로 스코프 확장된 것**이지 새로 닫힌
게 아니다. 선결 #1·#3·#4a·#4b 상태 불변, #2는 S/O 프로토콜
범위에서 **부분 해소**. **Claim D 등급 = 미검증, 불변.**
**PASS는 P2 착수를 승인하지 않는다**(NP-8 — 남은 블로커:
λ0 `NO-GO`·W4 워크로드 정의·게이트 #6).

## F. ★4번째 축 발견 — 907959↔908179 사이 노드/물리 GPU도 이동

사전등록 §10은 이동 축을 3개(mixin sha·하네스 sha·계측
플래그)로 셌으나 실제로는 **노드·물리 GPU(gpu38/
`GPU-4f60982e-…` → gpu40/`GPU-94efac99-…`)도 동시에 움직였고
이 축은 등록되지 않았다**. 그 축의 메모리 채널은 닫혔다 —
두 job 5 boot 전부에서 `avail mem` 사다리(78.69/62.10/13.98/
11.59 GB)가 **완전 동일**해 "GPU 여유가 더 있어서 살았다"는
배제된다. ★그러나 **채널이 닫힌 것과 축의 존재가 등록됐던
것은 다르다** — 세지 않은 축은 닫을 대상으로도 등록되지
않는다(G-8179-2).

## G. ★GPU 장부 드리프트 정정 — sacct 기준으로 통일

이전 배너(2026-09-14 1차)의 "0.43→0.93833 GPU-h"는 내부적으로
**자기 모순**을 담고 있었다: 등록 소계 `0.78583`은 `0.43`
(907032+907100+907456의 **반올림값**) + `0.35583`(907959
정확값)의 합이었는데, `sacct` 정확값(907032 7:13+907100
9:34+907456 9:14=1,561초=**0.433611**)을 쓰면 그 소계는
`0.78944`가 되어 **−0.00361 GPU-h** 어긋난다. 이 감사가
발견해 doc-steward 판단을 요청했다 — **이번부터 `sacct` 정확
초 단위를 유일 기준으로 채택**한다(과거 배너의 원문은 이력
그대로 보존, 되돌려 고치지 않는다):

```
R2 correctness 트랙 (sacct 기준, 2026-09-14(2)부터 유일 기준)
  등록(907032 7:13 + 907100 9:34 + 907456 9:14
       + 907959 21:21 + 908179 9:56 = 3,438s)   0.955000 GPU-h
  등록 밖(908020, 9:09 = 549s)                   0.152500 GPU-h
  누적                                            1.107500 GPU-h
longctx_conflict 트랙 15.42 GPU-h — 별개 장부, 불변
```

## H. 인용 금지·신규 게이트

**신규 인용 금지 A8179-1…7**(전문은 판정서 §14, 정본에는
존재·출처만 등재 — 요지: Δ<0≠TD 우위·L1 peak≠legacy 성질·
F-d "94.1%"는 스냅샷 단위이지 task 전수 아님·C층 6/32는
동시부하 판정 근거 아님·2채널 정렬검사 이 job서 검출력
0·O 티어 "구성상 결정적" 정당화는 triton 유도라 flashinfer
기판서 미재유도·"908179가 907959를 재현/반증했다"는 거짓[수리
前 arm 없음]). **필수 병기 A8179-P1…P7**(§13). **신규 게이트
6건 #232–237**(대응 `CONSENSUS.md` §3 항목252–257, 요지 아래
"방법론 게이트" 목록 참조 — **#232는 긍정 사례**: "세 boot이
바이트 동일"을 보고하려면 그 계측기가 같은 job에서 arm
차이를 해상함을 먼저 보여라).

## I. 승계 총량 갱신 · 감사자 자기 제한

sha 핀 승계 총량이 **96건 + 전사 의무 2건**(NP·NPC·N·A908·
RRC·RA3·RA4·RR 8계열)으로 늘었다. ★**RA4-6 유효**: 다음 회차
전방 강제 조항은 **44건만** 요구해 **40건이 무강제**로
남는다. 감사자 자기 제한: 이 판정서의 신규 분석 4종(연역적
정렬 채널·arm 분리 집계·교차-job 배치열 분할표·505-토큰
분해)은 **전부 사후 분석이며 등록 예보가 아니다** — 관측
기록으로만 인용한다.

## J. 정본 반영

`CONSENSUS.md` rev74→**rev75**(§3 항목252–257 신설[게이트
#232–237]) — 판단 근거: job 908179 결과 등재(Claim D 선결 #5
스코프 확장) + GPU 장부 드리프트 정정(사용자 지시 판단
사항) + 4번째 미등록 축 발견 + 신규 게이트 6건. `reports/
paper/CLAIM_EVIDENCE_MATRIX.md` Claim D 행에 새 문단 추가,
`EXPERIMENT_ROADMAP.md`는 P2 블로커 목록 갱신(GPU
correctness 선결은 이제 "PASS이나 P2 미승인"으로 정정).
`MEMORY.md` 포인터 갱신, `memory/{deconfound-measurement-
lessons,slo-aware-scheduling-track}.md`에 이번 회차 항목
신설(항목230–235, topic 파일 내부 순번 — `CONSENSUS.md` §3
항목과는 별개 축). **커밋 금지**(핸드오프 커밋에 함께
묶인다).

GPU 장부: 이 배너 자체는 **신규 GPU 지출 0**(이미 지출된
job 908179의 0.165556 GPU-h를 정본에 처음 등재할 뿐). R2
correctness 트랙 누적(sacct 기준) = **1.107500 GPU-h**(등록
0.955000/등록 밖 0.152500). `longctx_conflict` 15.42 GPU-h
불변.

**overclaim 금지**: (1) "수리가 OOM을 고쳤다"는 `PLAUSIBLE
(조건부)`일 뿐 `CONFIRMED`가 아니다 — `none` 대조 arm 없이는
승격 불가. (2) Claim D 선결 #5는 **새로 닫힌 게 아니라
스코프만 확장**됐다 — Claim D 등급은 여전히 미검증. (3)
`PASS`는 P2 착수를 승인하지 않는다(NP-8). (4) 907100/907456/
X1의 (Zamba2, triton) 한정 결론은 이 job과 **무관하게 동결**
그대로다.

이전: 2026-09-14(doc-steward — ★2026-09-13 저녁~2026-09-14
새벽 세션 산출물 정본 반영 1회 패스[그 세션 자체는 산출물
다수·정본 미반영 상태였음]: 사용자 결정 4건 등재 + R2
correctness 트랙 GPU 실측 2건 등재[job 907959 `FAIL`=true-dual
OOM `CONFIRMED(scoped)`·job 908020 `PASS`-등록밖, 트랙 누적
0.43→**0.93833 GPU-h**] + ★엔진 실물 결함 발견+수리[worker
thread grad-guard 누락, 커밋 `a9cd8dd`] + λ_inf(A) 측정+
λ_inf(B) 인용 금지(N-8)+N-7 정정 + 사전등록 재감사 이력 4+4
연속 등재[rerun rev1→rev4 `GO-with-caveats`·λ0 rev1→rev4 전부
`NO-GO`] + job 908179(작성 시작 시 RUNNING → 작성 완료 시
`sacct` 확인 결과 `COMPLETED` exit 0, ★verdict 미독·미감사 —
결과는 다음 패스) + 신규 방법론 게이트 31건(#201–231,
`CONSENSUS.md` §3 항목221–251 신설). 새 성능 판정 0건·이 패스 GPU 신규지출
0(전부 이미 지출된 세션 결과의 재등재)·Claim D/E 등급 불변
(둘 다 미검증)·HE0·정책 순위·stake #1·게이트 #13/#16·C2
인용정지 전부 불변. `CONSENSUS.md` rev73→**rev74**(판단
근거는 아래 "I. 정본 반영" 참조). **커밋 금지**(핸드오프
커밋에 함께 묶인다).

## A. 사용자 결정 4건 (이번 세션)

1. **W4 워크로드 = phase별 독립 λ\*** — `EXPERIMENT_ROADMAP.md`
   P2의 열린 항목("단일 λ\*가 W4의 두 phase를 동시에
   파라미터화할 수 없다", 2026-09-13(3) 등재)을 종결하는
   방향으로 채택. 구현물 `lambda_star.py`(신설)·
   `PDMUX_SUSTAINABLE_RATE` fail-closed·campaign 스키마
   v2→v3 — **미커밋**. ★신규 사실: W5/W6는 단일 shape가
   아니라 **한 Poisson 스트림에서 두 shape를 교대**하므로
   per-shape rate가 존재하지 않아 기본 거부되며, 도입된 혼합
   정의는 **사전등록된 적이 없다**(W5/W6 실행 전 별도 규칙층
   감사 필요).
2. **옵션 A 수리 + arm 대칭 메모리 계측 → 게이트 재실행** —
   아래 "C" 절의 grad-guard 결함(H1)에 대한 대응. 텐서
   의미론까지 legacy와 맞추기 위해 `no_grad`가 아니라
   `inference_mode` 선택.
3. **rev4 → 재감사 → 실행** — 재실행 사전등록의 검정력 공시
   (RA3-1: "이 캠페인은 어느 결과가 나와도 H1을 닫지 못한다")를
   받아들이고, **게이트 라벨 자체를 산출물**로 진행(H1의 인과
   판정과는 분리 — 아래 "E"·"F" 필수 병기 참조).
4. (직전 세션 승계, 2026-09-13(2)) 모델·백엔드·ctx =
   Nano-9B-v2-Base / flashinfer / 16384(불변).

## B. GPU 실측 2건 — R2 correctness 트랙 누적 **0.93833 GPU-h**

```
R2 correctness 트랙 누적          0.93833 GPU-h
  ├ 등록된 회차(907032·907100·907456·907959)   0.78583
  └ ★등록 밖 실행(908020, 스코프 불일치)        0.15250  ← 어떤 게이트도 진전시키지 않았다
longctx_conflict 트랙 15.42 GPU-h — 별개 장부, 불변
```

**job 907959**(0.35583 GPU-h, gpu38, `VERDICT FAIL`
`TD1:B2_NO_CRASH`·`TD2:B2_NO_CRASH`) — true-dual boot 2/2가
legacy가 **같은 job에서 완주한 바로 그 14-seq/10,125-token
split-prefill 배치**에서 `torch.OutOfMemoryError`. S16(6/6
쌍)·O8(6/6 쌍·32/32 probe-boot) 불일치 0(측정 사실 기록, PASS가
아니므로 "동등성 인증"으로는 쓸 수 없다 — 사전등록 §2가 인증문을
PASS 전용으로 제한). 결과 감사(`.../audit_907959_2026-09-13/
VERDICT.md`) 판정: 귀속 **`CONFIRMED(scoped)`**("이 스코프
튜플·이 배치에서 true-dual boot 2/2가 죽고 legacy 2/2는 같은
[더 큰] 배치를 완주했다") / 기전 일반화 **`NOT-YET-SUPPORTED`**
(사전등록 §8이 원인 분해 회차 없는 일반 귀속을 명시적으로
금지). 인용 금지·필수 병기 N-1…N-13(전문은 판정서 §12, 정본에는
존재·출처만 등재). 이 FAIL은 **P2/Claim D를 차단**하나(사전등록
§8), λ0 0단계·P1은 이 FAIL로는 차단되지 않는다(legacy 경로만
쓰므로) — 단 λ0는 아래 "E"의 별건 死因으로 이미 막혀 있었다.

**job 908020**(0.15250 GPU-h, gpu43, `VERDICT PASS`) —
★**등록 밖 튜플**로 돌았다: 제출 명령이 `R2C_MODEL`/
`R2C_ATTN_BACKEND`/`R2C_CTX` 세 env를 빠뜨렸고 하네스 기본값이
Zamba2-2.7B/triton/ctx4096이었다(§9의 명령이 §0-b 스코프 튜플을
실현하지 못하는 **문서 내부 모순**, `rerun_prereg/
PREREG_RERUN_2026-09-13.md`). 결과 감사(`.../
audit_908020_2026-09-14/VERDICT.md`) 판정: **이 job은
사전등록의 실행이 아니다**(등록된 어떤 예보도 채점 대상이
아님). ★결정적 신규 사실: **job 907100(수리 前, Zamba2 기판)의
TD1이 이미 12,369-token 배치를 완주**했다 — 908020의 "TD1
8,497 > 907959 천장 6,245"를 수리의 증거로 쓰는 것은
**`REFUTED`**(두 층: 모델 비교 불가 + 같은 모델·수리 前에 이미
초과). 인용 금지 **A908-1…7**(전문 판정서, 정본에는 존재만
등재) — 특히 **A908-7**: 이 job을 승인했던 `rerun_prereg` rev2
판정서의 등급 `GO-with-caveats`는 **인용 불가**(사후 재감사에서
사인 `NO-GO`였어야 함이 밝혀짐, 死因 N3).

## C. ★엔진 실물 사실 — true-dual worker grad-guard 결함 + 수리

진단서(`.../audit_907959_2026-09-13/DIAGNOSIS_true_dual_
oom.md`, engine-porter, GPU 0, 코드 무수정)의 **주 가설 H1**
(산정, GPU 측정 前 — CPU 측정으로 코드 사실은 확인됨):
true-dual의 role worker thread가 `event_loop_pdmux`의
`@torch.inference_mode()` **밖**(thread-local이라 스레드
경계를 못 넘음)에서 돌아 모델 forward를 **autograd가 켜진 채**
실행하고 있었다. 노출 경로는 `mixer2_rms_norm_gated.py:97`의
bare `nn.Parameter`(`.data` 없이 그대로 사용) 단 한 곳이고, 그
줄은 `mamba_n_groups != 1`일 때만(`forward_native` 분기)
실행된다 ⇒ **NemotronH(n_groups=8)만 노출, Zamba2·Falcon-H1·
Granite(전부 n_groups=1)는 구조적으로 면역**(커널 분기가
`.data`를 씀). 산정 `R(T)=T×1.2750 MiB`(층당 retain 바이트)가
예측한 천장 6.5k–7.7k 토큰과 관측(6,245 통과/10,125 실패)이
일치. ★**수리 = `_activate_role_context`에 grad guard**
(`inference_mode` 선택 — `no_grad` 아님, legacy와 텐서
의미론까지 대칭 유지), 커밋 `a9cd8dd`. 동반 계측:
`runtime_snapshot`에 메모리 9키를 양 arm 대칭으로 추가
(`PDMUX_MEM_TELEMETRY` 기본 1), 관측자 효과 0.084%(벽시계).

★**소급 주장 금지**: 이전 TD job(907100/907456/X1, 전부
Zamba2)들은 이 OOM 기전에 **구조적으로 면역**이었으나, "worker가
autograd 켜진 채 돈다"는 성질 자체는 **그 job들에도 있었고**,
그로 인한 arm 비대칭 autograd 부기 오버헤드는 **미측정**이다.

★**감사 정정(RA4-1, 최우선)** — 재실행 rev3 사전등록 §6-5의
"`inference_mode` × `forward_native`(bare Parameter,
`n_groups≠1`) 조합은 GPU에서 한 번도 실행된 적이 없다"는
**거짓이다**: job 907959의 legacy 2 boot(NemotronH,
n_groups=8)이 바로 그 조합을 `@torch.inference_mode()` 루프
안에서 각 56.0초·decode step 2,604/2,634·`cuda graph: True`·
`request_errors=0`·`crash_free=True`로 **이미 무오류 완주**했다.
E5(inference-mode 하드에러 트리거 계열)는 "미지의 위험에 대한
안전망"이 아니라 "**이미 통과한 조합**에 대한 값싼 안전망"으로
재분류해야 한다.

## D. λ 측정 (job 907959 계측, `instrument/` 원자료)

- **λ_inf(A) = 3.0939 req/s**(shape (256,512), decode-heavy) —
  **legacy·warm-up boot·cudagraph ON 한정**, decode 줄
  3,880 중 2,814(72.5%)가 엔진 상한 48에 도달한 closed-loop
  상한 프로브. **분할 혼합비 미측정**(warm-up boot은
  텔레메트리를 안 씀, NPC-I). 등록 λ\*(A) 사전추정(2.10–2.51)보다
  23–47% 큼 — λ_inf≥λ\*이므로 형식 모순은 없음.
- ★**λ_inf(B) = 0.6956 req/s는 인용 불가(N-8)** — 사전등록
  §5-I3 (D4) 레시피로 셀별 재계산한 결과 shape (8192,64)
  셀(I3b)의 `#running-req` 최댓값은 **2**(줄 종류 무관 시 11,
  앞 셀 drain tail 제거 시 2 — 세 규약 전부 <48)이지, 하네스가
  기록한 전역 최댓값 48(**셀 A에서만 온 값**)이 아니다. F5("두
  셀 각각 48 도달")는 셀 B에서 반증됐다.
- ★**N-7(정정)**: I2(동시성 1) 프로브의 TTFT 43.03 ms·ITL
  median 12.96 ms는 **D44 값이 아니라 비분할(108 SM) 값**이다
  — 엔진은 prefill∧decode가 **동시에** 활성일 때만 green idx
  4(D44)를 쓰고 그 외엔 idx 0/5(비분할)에서 돈다(같은 job
  scored boot 텔레메트리 12,458 스냅샷: idx0 11,725·**idx4
  592**·idx5 141; 동시성 1에서는 중첩이 구성상 불가능). λ0 rev1
  판정서 `:151`("B=1은 pdmux 분할 미적용 구간")이 **옳았다** —
  2026-09-13(2)~(3) 배너의 관련 서술을 이 사실로 정정.

## E. 사전등록·감사 이력 (두 트랙, 전부 이번 세션)

**재실행 사전등록**(`rerun_prereg/`, 목적: grad-guard 수리[C절]
후 같은 스코프 튜플로 907959 OOM 재현 여부 확인): rev1 `NO-GO`
(死因 N2×2) → rev2 `GO-with-caveats` → **job 908020 실행**(등록
밖, 위 "B" 참조) → rev2 등급 **★철회**(A908-7, 사후 재감사로
올바른 등급은 `NO-GO`[死因 N3]였음이 밝혀짐) → rev3 `NO-GO`
(死因 N2 — C1 스코프 일치 술어 항목 (9)["`src_dirty:` 다음
줄이 비어 있음"]의 참 분지가 하네스 산출물 어디서도 성립하지
않음: git이 0줄을 내면 "다음 줄"은 항상 다음 명령[`nvidia-smi`]
의 출력이므로 직해하면 캠페인 전체가 `NO_VERDICT_SCOPE`로
확정되고 C3 재실행 예산 0에 의해 **비가역**) → **rev4
`GO-with-caveats`**(死因 0건, 자유 표면 13개 반전 0/13, 등록
caveat **RA4-1…12**). ★**RA3-1 반영(필수 병기)**: 이 재실행
회차는 **어느 결과가 나와도 H1을 닫지 못한다**(F-a1이 원인
무차별+공허 참 가능, F-a2/F-b는 양쪽 `UNREALIZED` 분지 존재).
**제출 가능 상태**(rev4 §6) — 단 제출 전 커밋(sbatch·scope
guard·job 907959/908020 아티팩트)·`git status --porcelain --
.../src`=0줄·테스트 재확인(62/62·13/13·12/12·line-citation
88/0) 6항목 완료 후. → **job 908179가 이 rev4의 등록 튜플
최초 실행이다**(아래 "F" 참조).

**λ0(캠페인 0단계 λ\*) 사전등록**(`lambda0_prereg/`): rules
`NO-GO`(死因 N2+N3, 2026-09-13(3)에 이미 정본 등재) → rev2
`NO-GO`(死因 N2 — 비포화 셀의 "영점" 기준선이 등록에 없고,
실제 값은 `1/(1+L/span)`로 shape에 따라 0.93–1.02 사이를
움직여 shape A 라벨을 뒤집음 + fallback 분기가 미등록 재량이라
shape B 라벨도 뒤집힘) → rev3 `NO-GO`(死因 N3 — 등록 예보
"shape B = `KNEE_BRACKETED`"의 참 분지가 **정의역 자체가
공집합**: 계획이 shape B의 rung 4개 **전부**를 저측 후보로
인증하는데 R0 배수 가드가 인증된 모든 셀에 걸려 R1 고측
문턱과 상호배타적 요구를 만듦 — ★이 死因은 **감사자 자신의
rev2 처방[D16/D17]이 만든 것**임을 자기 철회로 기록[6회차
누적]) → **rev4 `NO-GO`**(死因 2건, 같은 한 곳: **N1**[F5
앵커 술어 조건 (b)가 live에서 실효 항등식 — 생산 하네스가
전역 최댓값 한 줄만 쓰는데 술어는 "셀별 전부 ≥48"을 요구하는
것처럼 읽혀 실격 경로가 죽어 있음, 907959 계측을 그대로
넣으면 `ANCHORED`·실격 0건] + **N2**[그 술어를 셀별로 수리하면
분지가 `ANCHORED→FALLBACK`으로 바뀌고 **shape B 13격자점 중
5점·shape A 12점 중 3점**의 라벨이 뒤집힘, 반전 4점은 등록
커버리지 밴드 안, 예산 2.592→3.509 GPU-h]). **4연속 `NO-GO`,
GPU 지출 0**(전부 규칙층, 원자료+CPU만 사용). 인용 금지
**L1–L3·R3C-1…4·R4C-1…6**(전문은 각 판정서, 정본에는 존재만
등재). rev3 死因은 rev2 처방 자신의 granularity 결함이었고
rev4 死因은 직전(907959) 결과 감사가 이미 "제출 전 필수"로
등록해 둔 수리(§13-D1)를 rev4가 반영하지 않은 것 — **양쪽
모두 GPU 0·처방 1곳·수리 후 재감사에서 `GO-with-caveats`
가능**으로 판정서가 평가.

## F. ★진행 중(→종료) — job 908179 (결과는 이 패스에 없음)

**job 908179**(gpu40, `r2corr`, SubmitTime 11:10:09 →
StartTime 11:10:15, `TimeLimit 02:30:00`, `--comment
"field=efficientai;appl=pytorch"`)는 이 배너 작성을 **시작할
때는 RUNNING**이었다(`squeue`/`scontrol show job` 재현). ★이
패스를 **마무리하는 시점 재확인 결과 `sacct` 기준
`COMPLETED`(exit 0:0, 11:10:15→11:20:11, 9분56초)로 이미
종료돼 있었다** — `exit 0`은 sbatch 스크립트 자체가 정상
종료했다는 뜻일 뿐 correctness verdict(`PASS`/`FAIL`)와는
별개이며, **이 판정서는 그 verdict를 읽지도 감사하지도
않았다**(지침에 따라 의도적으로 미수행). 이는 위 "E"의 rerun
rev4 `GO-with-caveats` 제출 체크리스트를 거친
**등록 튜플(Nano-9B-v2-Base/flashinfer/ctx16384/D44,
grad-guard 수리 반영 엔진)의 최초 실행**으로 보인다
(`job_908179/` 디렉터리에 `boots.txt`·4 boot 전부[L1/L2/TD1/TD2]
의 `gen_*.json`·`srv_*.log`·`tel_*.jsonl`·`green_*.json`·
`verdict.txt`·`verdict_rule.txt`·`provenance.txt` 등 **전체
산출물이 이미 존재** — job이 4 boot 전부를 완주했다는 뜻이며,
`verdict.txt` 자체의 내용은 이 패스가 **의도적으로 읽지
않았다**). **★이 job의 결과(FAIL/PASS, S/O 불일치, OOM 재현
여부)는 의도적으로 이번 정본 반영 패스에 넣지 않는다 —
job이 이미 완주했으므로 다음 세션 시작 즉시 결과 감사를
우선 실행해야 한다** — 다음 패스(핸드오프)에서 결과
감사(claims-auditor)를 거쳐 등재한다. 이 시점에서 미리 쓸 수
있는 것은 없음(RA3-1이 이미 경고: 어느 결과가 나와도 H1을
닫지 못한다 — "성공"이어도 "실패"여도 그 자체로 성능 판정이나
아키텍처 우열 결론이 되지 않는다).

## G. 신규 방법론 게이트 31건(#201–231)

번호 이어서(직전 #200). 6개 판정서(907959 결과감사·908020
결과감사·rerun rev3·rerun rev4·λ0 rev3·λ0 rev4)가 낸 원후보
34건(G-X 6·G-908020 5·G-λ3 5·G-λ4 6·G-RA3 6·G-RA4 6)을 중복·
근접 계열 통합해 **31건**으로 등재(통합·번호 부여는
doc-steward 판단, 출처는 각 항목에 병기). 전문은 이 문서 아래
"방법론 게이트" 목록(신규 항목 201–231)과 `CONSENSUS.md` §3
항목221–251 참조. 요지만:

- **#201** — "진단 전용" 라벨의 정의역(불일치 수 한정 vs 그
  층 전체 사건)을 사전등록 문안에 명시하라(긍정 사례 — 907959
  규칙은 명시돼 있어 재량 0이었다). §3 항목221.
- **#202** — 퍼-셀 판정을 요구하는 예보는 생산 하네스가 실제로
  퍼-셀 산출물을 만드는지 확인하라(전역 집계 오독은 실격 경로를
  실효 항등식으로 만든다 — 907959 F5·λ0 rev4 N1 공통 재발).
  §3 항목222.
- **#203** — 포화 판정 계기는 그 shape의 실제 병목 축과 맞춰라
  (decode-슬롯 계기는 prefill-토큰-지배 shape에서 구조적으로
  도달 불가할 수 있다). §3 항목223.
- **#204** — 분할(green-context) 라벨을 수치에 붙이기 전에 그
  구간이 실제로 prefill∧decode 동시성 위에 있었는지 확인하라
  (동시성-1 프로브는 구성상 분할 인덱스를 못 밟는다, N-7).
  §3 항목224.
- **#205** — verdict 리포트가 인쇄하는 카운터 이름이 게이트
  술어가 실제로 읽는 값과 같은지 확인하라. §3 항목225.
- **#206** — 오류로 비어버린 출력끼리의 "불일치 0"은 결정성이
  아니다(비교 함수가 오류 레코드를 등가류에 넣으면 죽은 arm은
  자기 자신과 항상 일치한다). §3 항목226.
- **#207** — 사전등록의 "제출 명령"은 그 문서의 스코프 튜플을
  실현하는지 한 줄씩 대조한 뒤에만 승인하라(하네스 기본값이
  튜플과 다르면 명령 자체가 死因). §3 항목227.
- **#208** — rev 개정 시 "불변"이라 표에 적은 축이 실행
  절차에서 조용히 사라졌는지 확인하라(삭제는 diff·grep에 안
  잡힌다, 교훈 89 확장). §3 항목228.
- **#209** — 인적 실패점 하나를 명령줄에서 제거했다면 같은
  명령줄의 나머지 인자 전부에 같은 검사를 적용하라. §3 항목229.
- **#210** — 스코프 불일치 처분(전용 라벨 + 병기 의무 + 유한
  재실행 예산)을 사전등록에 반드시 넣어라 — 안 넣으면 "등록
  실험이 돌았는가"가 라벨을 본 뒤의 사람 재량이 된다. §3 항목230.
- **#211** — 사전등록의 1차 판정 술어를 처치-이전 엔진+같은
  기판의 기존 원자료에 먼저 먹여 그 기판에서 검정력이 0인지
  확인하라(양성통제의 소극판). §3 항목231.
- **#212** — 계획 단계의 "자격"과 실행 단계의 "역할"을
  구분하라 — 자격 인증에 걸리는 가드가 그 자격이 아닌 역할로
  쓰인 같은 셀에도 걸릴 수 있다. §3 항목232.
- **#213** — 결정 규칙의 문턱 도달가능성은 등록된 모든 verdict
  라벨 각각에 대해, pooled가 아니라 실제로 돌 시나리오별로
  사전 계산하라. §3 항목233.
- **#214** — 같은 문서의 두 문장이 같은 대상에 상호배타적
  수치 요구를 하고 있는지 대조하라. §3 항목234.
- **#215** — 분지 존재 증명 테스트에 손으로 만든 레코드를
  쓰지 마라(실제 인증 경로에서 존재할 수 없는 셀은 그 분지를
  증명하지 못한 채 통과를 준다). §3 항목235.
- **#216** — 변이 하네스 CONTROL의 비공허성은 논증이 아니라
  주입(실행 경로를 깨서 CONTROL이 진짜 실패하는지)으로 보여라.
  §3 항목236.
- **#217** — 직전 감사가 "제출 전 필수"로 등록한 수리를 반영
  안 한 판본을 볼 때는, 그 수리가 분지를 바꾼다는 사실과 두
  분지의 결과를 실행 전에 등록하라. §3 항목237.
- **#218** — 변이 하네스의 모듈-변이 라우팅표는 그 모듈 상수가
  실제로 영향을 주는 모든 판정 산출물(reachability selftest
  포함)로 보내라. §3 항목238.
- **#219** — 같은 상수를 두 파일이 각자 정의하면 죽은 쪽을
  삭제하라(특히 그 주석이 이미 철회된 규칙을 적고 있을 때,
  게이트#179 계열). §3 항목239.
- **#220** — "블록이 비어 있다"를 게이트 술어로 쓰지 마라(빈
  출력은 줄을 안 남기므로 "다음 줄"은 언제나 다음 명령의
  것). §3 항목240.
- **#221** — 재감사에서 직전 판정서 문안 일부만 재검증하면,
  재검증 성공 항목이 나머지 미검증 항목의 신뢰를 위조한다
  (게이트#110 심화). §3 항목241.
- **#222** — 철회된 판정서의 "등급"과 "caveat 목록"은 별개다
  — 철회 시 함께 소실되는 항목을 열거하라. §3 항목242.
- **#223** — "엄한 실패 지점은 정확히 N개"라는 사실은 설치본
  에서 전수 열거해 등록하라(부분 열거는 처치층 우연을 가설
  판정으로 둔갑시킨다). §3 항목243.
- **#224** — cross-substrate로 같은 수치가 나오면 "조밀해서"로
  설명하지 마라(밀도는 실현 가능성만 설명한다). §3 항목244.
- **#225** — fail-closed 스코프 가드의 발화 자체를 provenance
  술어로 추가하는 것은 거의 항등식이다 — 남는 고유 정보만
  명시하고 "독립 채널"이라 부르지 마라. §3 항목245.
- **#226** — "이 조합은 GPU에서 한 번도 실행된 적 없다"는
  주장 전에 같은 job의 대조 arm을 먼저 조회하라. §3 항목246.
- **#227** — 예외 계열(pattern) 일반화 시 토큰을 같은
  불완전 열거에서 다시 뽑지 마라 — 설치본 전수에서 재유도
  하라. §3 항목247.
- **#228** — 파생 집계(기저율·백분율)를 철회할 때 문서
  전역에서 grep해 함께 철회하라(국소 철회는 처분절에 살아
  남는다). §3 항목248.
- **#229** — 사전등록의 "제출 전 커밋 목록"은 감사 시점의
  `git status`로 재생성하라. §3 항목249.
- **#230** — 승계 사슬의 "전방 강제" 조항은 자기 요약절의
  항목 수와 대조하라(누수 구조 확인). §3 항목250.
- **#231** — 참 분지가 공집합이던 술어를 수리한 뒤에는 교체
  문안을 실제 아티팩트 전수(음성대조 포함)에 먹여 도달가능성을
  실행으로 보여라(긍정 사례). §3 항목251.

## H. 인용 금지 목록 (존재만 등재 — 전문은 출처 판정서)

이번 세션이 등록한 인용 금지·필수 병기 목록은 규모가 크다.
정본에는 **목록의 존재와 출처 파일만** 적는다(전문은 판정서에
둔다, 지침 준수):

| 계열 | 건수 | 출처 판정서 |
|---|---|---|
| N-1…13 | 13 | `.../audit_907959_2026-09-13/VERDICT.md` §12 |
| A908-1…7 | 7 | `.../audit_908020_2026-09-14/VERDICT.md` §9 |
| RRC-1…13 | 13 | `.../rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md`(등급 철회, caveat 자체는 부분 생존 — RA3-5 참조) |
| RA3-1…12 | 12 | `.../rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md` §4 |
| RA4-1…12 | 12 | `.../rerun_prereg/VERDICT_rerun_rev4_2026-09-14.md` §4 |
| L1–L3 | 3 | `.../lambda0_prereg/VERDICT_lambda0_rev2_2026-09-13.md` §7 |
| R3C-1…4 | 4 | `.../lambda0_prereg/VERDICT_lambda0_rev3_2026-09-13.md` §7 |
| R4C-1…6 | 6 | `.../lambda0_prereg/VERDICT_lambda0_rev4_2026-09-14.md` §7 |
| Q1–Q5 | 5 | `.../lambda0_prereg/VERDICT_lambda0_rules_2026-09-13.md` §5(2026-09-13(3)에 이미 정본 등재, 불변) |

★가장 자주 재인용될 3건만 여기 그대로 적는다(나머지는 위 표의
출처를 인용): **RA4-7**("게이트 라벨이 닫는 것은 H1과 별개다 —
어떤 `PASS`도 'S층·O층 토큰 id 동일 + cudagraph 유지' 한 문장만
인증하고, 907100/907456이 닫았던 선결은 [Zamba2, triton]
한정이다") · **RA3-1**("재실행 회차는 어느 결과가 나와도 H1을
닫지 못한다") · **N-8**("λ_inf(B)=0.6956은 포화 상한으로 인용
불가").

## I. 정본 반영

`CONSENSUS.md` rev73→**rev74**(§3 항목221–251 신설[게이트
#201–231]) — 판단 근거: 신규 방법론 게이트 31건은 그 자체로
CONSENSUS rev 사유이며, 여기에 (1) R2 correctness 트랙 GPU
실측 2건[FAIL+등록밖 PASS] (2) 엔진 실물 결함 발견+수리[grad
guard] (3) 4+4연속 사전등록 재감사 이력 (4) λ 측정치 갱신+
인용정정(N-7/N-8)을 더해 단순 게이트 등재보다 훨씬 무거운
갱신이다. `reports/paper/{CLAIM_EVIDENCE_MATRIX,
EXPERIMENT_ROADMAP}.md` 갱신(Claim D 행에 새 GPU 실측·엔진
결함·λ 측정 문단 추가, P2 절에 job 908179 진행중 표시).
`MEMORY.md` 포인터 갱신, `memory/{deconfound-measurement-
lessons,slo-aware-scheduling-track}.md` 갱신(항목 신설 — topic
파일 내부 순번, `CONSENSUS.md` §3 항목과는 별개 축).
`memory/engine-port-p0-triage.md`는 이번 세션 내용과 무관해
갱신하지 않는다(그 파일은 2026-07 layer-type P0/P1 트리아지
전용, 이번 세션은 R2/dual-worker/λ\* 트랙 — 정본 = [[slo-aware-
scheduling-track]]). **커밋 금지**(이번 등재는 문서 작업만이며
핸드오프 커밋에 함께 묶인다).

GPU 장부: 이 배너 자체는 **신규 GPU 지출 0**(이미 지출된 이
세션의 결과[907959 0.35583 + 908020 0.15250 GPU-h]를 정본에
재등재할 뿐). R2 correctness 트랙 누적 0.43→**0.93833 GPU-h**.
`longctx_conflict` 15.42 GPU-h 장부 불변.

**overclaim 금지**: (1) job 907959 FAIL의 귀속은 "이 스코프
튜플·이 배치"에 scoped이며 "true-dual이 일반적으로 OOM으로
죽는다"로 일반화 금지(NOT-YET-SUPPORTED). (2) grad-guard 수리
(C절)는 **아직 GPU로 검증되지 않았다** — job 908179가 그
검증이며 결과는 다음 패스. (3) λ_inf(B)=0.6956은 인용 불가
(N-8). (4) rerun rev4 `GO-with-caveats`는 게이트 라벨
산출 허가일 뿐 "true-dual 아키텍처가 옳다"가 아니다(RA4-7).
Claim D·E 등급은 **둘 다 여전히 미검증**이다.

이전: 2026-09-13(3)(doc-steward — ★캠페인 0단계(λ*)
사전등록 `NO-GO` 등재[死因 N2+N3, GPU 0·미실행] + 감사 파생
사실 7건 등재[게이트 #6 미충족이 최우선] + 2026-09-13(2) 배너의
W4 decode phase 오독 정정[(64,512)→(256,512)] + 신규 게이트
5건[#196–200] + `EXPERIMENT_ROADMAP.md` P2에 열린 항목 신설
[단일 λ*가 W4 phase 둘을 동시에 파라미터화 못 함, 사용자 결정
대기]). 새 성능 판정 0건·GPU 0·Claim D/E 등급 불변(둘 다
미검증)·HE0·정책 순위·stake #1 전부 불변. `CONSENSUS.md`
rev72→rev73(판단 근거는 아래 "F. 정본 반영" 참조).

## A. ★정정: 2026-09-13(2) 배너의 W4 decode phase 오독

아래 "이전" 절(2026-09-13(2) "C")이 "W4 = prefill phase
(8192,64) + decode phase (64,512)"라고 적었는데 **거짓이다**.
λ0 사전등록 규칙층 감사(claims-auditor, GPU 0)가
`benchmarks/pdmux_eval/workloads.py:159-160`을 실제로 실행해
확인했다:

```
W4 n=96
   (in=256, out=512) phase=decode   n=48
   (in=8192, out=64) phase=prefill  n=48
```

⇒ **W4 decode phase = (256, 512)**이고 저장소 전체에 (in 64,
out 512)는 **존재하지 않는다**. 오독 출처 = `WorkloadSpec.
output_distribution = "64/512 by phase"`(phase별 **출력** 길이
서술)를 입력 길이로 읽은 것 — `reports/paper/EXPERIMENT_
ROADMAP.md`의 워크로드 정의표(`W4 | W2형/W3형 30초 phase를 3
cycles`)는 애초에 옳았고, 잘못은 2026-09-13(2) 세션의 서술
추가분에만 있었다. ★**shape (256,512)는 W3 전부이자 W4 decode
phase 그 자체이며 근사가 아니다** — 1차 캠페인(Claim D)에
필요한 shape는 3개가 아니라 **2개뿐**: (256,512)[=W3=W4 decode
phase]·(8192,64)[=W4 prefill phase]. 이 정정을
`EXPERIMENT_ROADMAP.md`·`CLAIM_EVIDENCE_MATRIX.md` Claim D 행·
이 문서 아래 "이전"(2026-09-13(2)) 절 원문에 **역사 보존 + 정정
표시**로 부착했다(삭제 없음).

## B. 캠페인 0단계(λ\*) 사전등록 `NO-GO`(GPU 0, 미실행)

판정서 원문 전사 = `workspace/engine-port/results/r2_eval/
lambda0_prereg/VERDICT_lambda0_rules_2026-09-13.md`(358행, sha
`fe75287b…`). 대상 사전등록 `PREREG_LAMBDA0_2026-09-13.md`(sha
`fa4bb517…`, **미실행**).

**死因 2건**:

- **N2(반전 확인, 수치 있음)**: ①client `--seed`가 미등록 —
  `bench_serving.py:1706 np.random.seed(args.seed)` +
  `:948 np.random.exponential(1/rate)`이므로 비포화 셀의
  achieved/offered는 **포화도가 아니라 도착 실현 계수 `1/Ē`**다.
  905835 원자료 **12/12 셀 전부** 도착 구간이 명목보다 15–18%
  짧았다(평균 −2.0σ) ⇒ ach/off 1.03–1.19. 다른 seed 시뮬레이션
  (40개)에서는 용량의 53%인 셀이 "포화" 오라벨(seed 24)되거나
  0.90<x<0.95 사각지대에 떨어져 브래킷이 성립 불가(seed 18/34)
  — **실패율 15–28%**. ★**N을 늘려도 못 고친다**:
  `1.96/√(N−1)≤0.05`는 **N≥1537**을 요구한다. ②shape A 사다리
  4점의 값이 등록에 적혀 있지 않다 — R1은 최저·최고점만 쓰므로
  사다리 선택만으로 `BRACKETED↔NOT_BRACKETED`가 뒤집히는 예시
  3건이 제시됐다.
- **N3(예측 도달 불가)**: 위 "A"의 오독 때문에 "두 shape의
  λ\*는 5% 이내" 예보의 **정의역이 공집합**이었다.

제출 차단 운영 결함 3건(전부 GPU 0): `UNRESOLVED`가 live
경로에서 **도달 불가**(glob이 결손 셀을 건너뛰어 boot 실패가
`KNEE_NOT_BRACKETED`로 나온다 — 등록 문안과 코드 불일치, 교훈
21·게이트 #21 위반) · `rec["shape"]`가 기존 analyzer 출력에
없어 **KeyError** · **sbatch·per-cell analyzer 자체가
부존재**(제출할 실행체가 없었다). 조건 D1–D15(전부 GPU 0)
전문은 판정서 §4.

★이 `NO-GO`는 "설계가 틀렸다"가 아니다 — **측정 격자와 승계된
규칙(R1/R5)은 건강함을 감사가 확인**했다(보관 cell JSON 9개를
직접 투입해 `C_LABEL.json`의 R1·R5를 비트 단위 재현). D1–D15
적용 후 재감사에서 `GO`/`GO-with-caveats`가 가능하다고 본다.

## C. ★이 감사가 만든 사실 7건(정본 가치가 큰 순서)

1. ★★**이 단계는 방법론 게이트 #6("용량 먼저 측정")을 닫지
   못한다.** 게이트 #6이 요구하는 것은 **지표 절벽 대비** 용량인데
   λ\*_throughput은 그 절벽을 위치시키지 않는다. 905835 d44
   원자료를 **정본 goodput 술어**(TTFT≤3000ms ∧ 요청-내부
   token-ITL p95≤60ms)로 재채점: **0.59·λ\*에서 goodput
   53.8%·0.89·λ\*에서 5.8%·1.27·λ\*에서 0.8%** ⇒ **λ\*_SLO
   < 0.59×λ\*_throughput**(비 추정 2.3–3.4×). **"0.60·λ\*"는
   이미 SLO 바깥이다.** `EXPERIMENT_ROADMAP.md:652`의 정본
   정의는 "sustainable **SLO** rate"인데 이 사전등록이
   **throughput 포화로 교체**했다 — 이 교체를 **정정으로
   등재**한다(아래 "D" 필수 병기 (2)).
2. ★★**두 λ\*는 W4를 파라미터화하지 못한다 — 상호 배타다**:
   λ\*=0.675 주입 → prefill phase 0.79× ✓ / decode phase
   **0.21–0.25×** ✗; λ\*=2.1 → decode 0.79× ✓ / prefill
   **2.47×** ✗; 기본값 4 → prefill **4.7×**. **어떤 단일
   스칼라도 두 phase를 동시에 0.80으로 만들 수 없다** ⇒
   **Claim D의 P2가 W4를 쓰려면 워크로드 정의 자체(phase별
   독립 λ\*)를 고쳐야 한다** — `EXPERIMENT_ROADMAP.md` P2 절에
   **사용자 결정 대기**로 등재(아래 "F").
3. **λ\*(A) ≈ 2.1–2.5 req/s**(절대 경계 [1.08, 7.15]) — 엔진
   로그 decode step 회귀 `step(B) = 19.82 + 0.5174·B ms`와
   구속 자원(`max_mamba_cache_size = max_running_requests = 48`;
   KV는 `max_total_num_tokens = 2,618,868`로 6.6× 과공급)에서
   유도. 권고 사다리 shape A **{1.1, 1.8, 3.0, 4.9, 8.0}**,
   shape B **{0.45, 0.62, 0.85, 1.15}**(사전등록 ×1.2 상단은
   자기 예보 방향에서 깨졌다 — probe C 실제 상단은 ×1.30).
4. **ctx 16384 / mem 0.82는 용량을 바꾸지 않는다**(구속이
   mamba cache 48이고 KV 6.6× 과공급) ⇒ 앵커 0.675의
   provisional 강등은 **충분하다**.
5. **R4(MODEL_HOLDS)의 닫힌 형태는 항등식이었다** —
   `pred = μ_ach·out·itl_load`로 환원되어 추정량(λ\*)을 입력으로
   요구, `obs/pred`는 사실상 `mean_itl/itl_p50`를 검정했다. λ0이
   R4를 승계하지 않은 것은 **옳다**(되살리지 말 것).
6. **R1/R5 승계는 CONFIRMED** — 보관 cell JSON 9개 직접 투입
   재현(d16 0.9331188685063582 등, `C_LABEL.json`과 비트 단위
   일치).
7. **우선순위 권고 변경**: **새 모델 correctness 게이트 +
   `--request-rate inf` 2셀(≈0.3–0.4 GPU-h)을 먼저** 산다 —
   `bench_serving.py:943-945`가 `rate == inf`에서 sleep을 전부
   건너뛰어 Poisson 실현이 존재하지 않으므로 **N2 死因이
   구조적으로 소멸**한다. ★감사자 자기 한계 고지 승계:
   closed-loop(concurrency 64) 포화율이 open-loop와 같다고 볼
   근거와 다를 근거가 둘 다 있으므로(905835 d44는 Concurrency
   24.08 중 22.75가 대기) **λ\*의 대체가 아니라 사다리 양 끝을
   고정하는 상한 프로브로만** 등록해야 한다.

## D. 인용 금지 Q1–Q5 + 필수 병기 6항(판정서 §5·§6 문안 그대로)

> **Q1** — 이 캠페인 0단계는 어떤 결과가 나오더라도 방법론
> 게이트 #6을 닫지 못한다. 905835 원자료를 정본 술어로
> 재채점하면 shape B(8192-in)의 SLO 절벽은 0.59·λ\*_throughput
> 아래에 있다(0.59×에서 53.8%, 0.89×에서 5.8%, 1.27×에서
> 0.8%). "λ\*를 측정했으므로 W3/W4의 부하 라벨이 참이 되었다"는
> 문장은 **쓸 수 없다**.

> **Q2** — λ\*(B1)의 측정은 (Nano-9B-v2, flashinfer) 쌍의 R2
> correctness를 **어느 정도도 확인하지 않는다**. 이 단계와 job
> 905835는 legacy 루프+fixed split만 발화시켰고 `PDMUX_TRUE_
> DUAL_WORKER`는 이 모델에서 **한 번도 실행된 적이 없다**.
> 907100·907456·X1의 결론은 (Zamba2-2.7B, triton) 한정으로
> 동결이고, X1의 민감도 시연은 `triton_attention_num_kv_splits`가
> flashinfer에 없으므로 이식되지 않는다.

> **Q3** — 이 단계가 내놓는 두 λ\*는 W4를 파라미터화하지
> 못한다 — 상호 배타적이다(위 "C-2"). W4를 Claim D의 P2에서
> 쓰려면 워크로드 정의 자체(phase별 독립 λ\*)를 고쳐야 하며,
> 그것을 이 단계가 주지 않는다.

> **Q4** — "B1의 용량이 시스템의 용량"이 아니다. 워크로드
> 이름(W8 "near saturation", W9 "overload")은 **B1에서만** 참인
> 서술이고, B4에서 같은 trace는 자기 용량의 0.2×일 수도 5×일
> 수도 있다. 워크로드 이름을 regime 서술로 인용하는 것을
> 금지한다.

> **Q5** — shape A·B 두 shape를 쟀다는 것이 W1·W5·W6·W7·W8·W9의
> 부하 라벨을 고치지 않는다. 그 워크로드들의 shape는 미측정으로
> 남으며, λ\*는 shape 의존적이다.

**필수 병기 6항**(결과가 나오면 문자 그대로 붙인다 — 이번
회차는 미실행이므로 선반영만): (1) λ\*는 B1(legacy, fixed D44)
한정 throughput 포화율이다 — split을 바꾸면 같은 모델·같은
shape에서 5× 변한다(D16 0.933/D44 0.675/D92 0.187). B4의 λ\*는
측정되지 않았다. (2) 정본 술어 goodput은 이 단계에서 측정되지
않았다 — TTFT·ITL SLO를 쓰지 않는다. (3) 상단 셀의 TTFT/ITL
백분위는 인용 불가(의도적 과포화 셀, 지연 수치가 아니라 처리율
plateau용). (4) `switch_count`·split 체류분포는 이 단계의
판정에 쓰이지 않았다(fixed split, 동적 제어 없음). (5) 포화
셀의 achieved는 도착 실현에 무관하나(λ\*는 robust), 비포화
셀의 ach/off는 도착 실현 계수다(905835에서 1.03–1.19, 12/12
동일 방향). (6) 이 단계는 `sglang.bench_serving`(Poisson)을
쓰고 캠페인은 `pdmux_eval.trace_loadgen`(고정 도착시각,
`input_ids=[1]*n`)을 쓴다 — λ\*는 다른 하네스에서 측정된
상수로 캠페인을 파라미터화한다.

## E. 신규 방법론 게이트 5건(#196–200)

번호 이어서(직전 #195). 전문은 이 문서 아래 "방법론 게이트"
목록(신규 항목 196–200) 참조, 요지:

- **#196** — 비포화 셀의 achieved/offered는 포화도가 아니라
  도착 실현 계수다(Poisson 부하에서 명목 rate를 분모로 쓰는
  문턱 규칙은 client seed가 사다리 전체에 공통모드로 들어가
  실현 가능한 N으로는 신뢰 도달이 불가능하다, N≥1537 요구).
  대응 `CONSENSUS.md` §3 항목216.
- **#197** — 사다리 규칙이 최저·최고점만 쓰면 중간점은 예보가
  아니다 — 사다리 값을 수치로 등록하지 않으면 그 예보는 자유
  표면이고 판정을 뒤집을 수 있다. 대응 §3 항목217.
- **#198** — 예보의 정의역은 생성기를 실행해 확인하라 — 워크로드
  명세 문자열(예: `output_distribution`)을 입력 shape로 읽으면
  예보가 공집합 위에 선다. 대응 §3 항목218.
- **#199** — 닫힌 형태 추정식이 자기 추정량(예측 대상 자체)을
  입력으로 요구하면 그것은 항등식이지 예측 모델이 아니다(R4
  유형 재발). 대응 §3 항목219.
- **#200** — `UNRESOLVED`(측정 실패) 경로가 live 코드에서 실제로
  도달 가능한지 주입 실험으로 확인하라 — glob 기반 셀 수집은
  결손 셀을 조용히 빼고 규칙 판정(실패가 아닌)을 내보낼 수 있다.
  대응 §3 항목220.

## F. 정본 반영

`CONSENSUS.md` rev72→**rev73**(§3 항목216–220 신설[게이트
#196–200]) — 판단 근거: 신규 방법론 게이트 5건은 그 자체로
CONSENSUS rev 사유(기존 선례)이며, 여기에 (1) 캠페인 0단계
`NO-GO`(死因 2건, GPU 0) (2) 정본 게이트 #6 미충족이라는
1순위 사실 (3) W4 상호배타 발견(Claim D P2 열린 항목) (4)
2026-09-13(2) 배너의 사실 오류 정정을 더해 단순 게이트 등재보다
무거운 갱신이다. `reports/paper/{CLAIM_EVIDENCE_MATRIX,
EXPERIMENT_ROADMAP}.md` 갱신(Claim D 행·"공통 방법" λ\* 절·P2
열린 항목: **"단일 λ\*가 W4의 두 phase를 동시에 파라미터화할 수
없음 — 워크로드 정의 변경 여부 사용자 결정 대기"**), `MEMORY.md`
포인터 갱신, `memory/{deconfound-measurement-lessons,
slo-aware-scheduling-track}.md` 갱신(항목 신설 — topic 파일
내부 순번, `CONSENSUS.md` §3 항목과는 별개 축). **커밋 금지**
(이번 등재는 문서 작업만이며 핸드오프 커밋에 함께 묶인다).

GPU 장부: 이 배너 전체 **GPU 0**(감사는 기존 원자료+CPU만
사용, 캠페인 자체 미실행). `longctx_conflict` 15.42 GPU-h·R2
correctness 0.43 GPU-h 장부 둘 다 불변.

**overclaim 금지**: λ\*는 **측정되지 않았다**(사전등록이
`NO-GO`). 905835의 값(D16 0.933/D44 0.675/D92 0.187)은
**8192-in/96-out·ctx 8192·mem 0.80** 조건의 기존 측정이며 새
캠페인의 λ\*가 아니다.

이전: 2026-09-13(2)(doc-steward — ★**세 번째 사용자 결정**
(attention 백엔드 `triton`→`flashinfer` 전환) + engine-porter의
Nano-9B-v2-Base **모델 지원 검증 `GO`**(job 905835 재인용, GPU 0
이 세션) + λ*가 이 모델에서 **이미 측정돼 있음**을 확인(arm별
5× 상이, 단일 스칼라 캠페인 설계 문제 신규 등재) +
provenance manifest 확장 사실 등재(커밋 `87213a9`, 이미 커밋) +
신규 게이트 1건(#195) + 게이트#83/§3 항목103 追記(교차-트랙
위험이 Nano-9B-v2에서 재현)). 새 성능 판정 0건·GPU 0(이 세션
신규 지출 0 — job 905835의 1.11 GPU-h는 `longctx_conflict`
트랙의 2026-09-09 기존 지출이며 이 등재는 그 사실을 모델-지원
증거로 재인용할 뿐)·Claim D/E 등급 불변(둘 다 미검증)·HE0·
정책 순위·stake #1 전부 불변. `CONSENSUS.md` rev71→**rev72**
(판단 근거는 아래 "E. 정본 반영" 참조).

## A. ★사용자 결정(세 번째): attention 백엔드 triton→flashinfer 전환

`--attention-backend triton`은 NemotronH에서 **엔진이 거부**한다
(CPU-only `ServerArgs` 구성에서 재현: `AssertionError:
NemotronHForCausalLM does not support triton attention backend, as
the first layer might not be an attention layer`,
`server_args.py:1959`). 두 R2 하네스(`r2_correctness.sbatch`·
`engine_bench_runner.sh`)가 모두 triton을 하드코딩하고 있었다.
사용자가 **flashinfer 전환**을 택했다(job 905835가 이미 그
조합으로 12/12 boot한 선례가 존재 — 아래 "B" 참조).

★**귀결을 명시 등재**(숨기지 않음):

- **X1의 발견은 triton 전용이므로 새 캠페인으로 이식되지 않는다**
  — `triton_attention_num_kv_splits`는 flashinfer에 존재하지
  않는다. X1이 시연한 "측정층이 축약순서급 교란에 반응한다"는
  하한은 **(Zamba2-2.7B, triton) 조합 한정**으로 동결한다.
- **907100·907456의 correctness 결론도 (Zamba2-2.7B, triton)
  한정**이며, 새 (모델, 백엔드) 쌍에서는 correctness 게이트를
  **새로 쌓아야** 한다(1 job, 9B 기준 ≈0.2–0.3 GPU-h 추정).
- 이것은 이미 정본에 있는 **교차-트랙 위험**(게이트 #83/
  `CONSENSUS.md` §3 항목103, TC1 job 896776 — NemotronH+triton
  부팅 거부)이 **Nano-9B-v2에서 재현**된 사례다 — 그 항목에
  追記했다(아래 "E").
- 새 게이트의 조건 튜플은 Zamba2 튜플과 **다르다**(모델·백엔드·
  ctx 전부) — P-1 계열 스코프 튜플에 `attention_backend=
  flashinfer`·`model=nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base`·
  `context_length=16384`를 넣어야 한다.
- `r2_correctness.sbatch`·`engine_bench_runner.sh`(커밋
  `87213a9`)에 백엔드 노브(`R2C_ATTN_BACKEND`/
  `PDMUX_ATTENTION_BACKEND`)가 생겼고 **기본값은 `triton` 불변**
  (Zamba2 재현 보존). NemotronH arm은 명시적으로 `flashinfer`를
  줘야 한다.

## B. 모델 지원 검증 결과 = **`GO`**(engine-porter, GPU 0)

- ★**기록된 GPU 사실**: `results/longctx_conflict/probes/
  ccap_905835.out`(job 905835, 2026-09-09, **1:06:42 = 1.11
  GPU-h**, gpu42)가 `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base`를
  **12 cell 전부 boot**(BOOT_FAILED 0), PD-mux ON·cudagraph
  ON·**flashinfer**·ctx 8704·D∈{16,44,92}로 서빙했다. "서빙
  가능한가"는 예측이 아니라 이 job의 사실이다.
- 가중치 완전(4 shard 16.56 GiB, `model.safetensors.index.json`
  **341/341** 일치, 누락 0, 전부 BF16, Σ params 8.888B). config↔
  가중치 전수 대조 통과(56층 패턴 ↔ 모듈 실측 일치,
  `full_attention_layer_ids=[14,21,30,39]`). **`head_dim=128`이
  config에 명시돼 있어야 한다**(40×128=5120 ≠ hidden 4480이므로
  fallback이면 틀린다 — 9B에서 load-bearing). hybrid KV
  `cell_size` = **16 KiB/token**(8B와 동일). MoE/MTP/SWA/MLA
  전부 부재. tokenizer 오프라인 로드 OK. `forward_split_prefill`
  존재(PD-mux 선결).
- 메모리: 가중치 16.56 GiB + mamba **136.90 MiB/req**(9B는 8B의
  1.41×) × 48 = 6.42 GiB + KV pool ≈41.9 GiB ⇒ **제약 아님**.
- ★**ctx 권고 = 16384**(131072 아님): 트레이스 최대 요구가
  **8320 토큰**(W5 8192+128)이라 2× 여유이고, 측정하지 않은
  131072 체제를 run record가 주장하지 않게 된다. correctness
  게이트에는 `R2C_CTX=16384`를 **명시**해야 한다(기본 유도는
  131072를 낸다).
- 잠복 상류 위험 3건(미수정, 등재만): ①`NemotronHConfig.
  mamba2_cache_params`가 **선언되지 않은 `self.n_groups`**에
  의존하며 `config.json`이 우연히 `n_groups`를 싣고 있어서만
  동작한다(없는 config면 `AttributeError`) ②`config.expand`가
  Nano-9B에서 **stale**(2 ⇒ d_inner 8960, 실제 10240) — 현재
  미사용이나 d_inner를 expand에서 유도하는 코드가 생기면 조용히
  틀린다 ③`NemotronHForCausalLM`이
  `piecewise_cuda_graph_disabled_model_archs`에 **없다** —
  `DECISION_PIECEWISE_OFF_2026-09-02.md`가 기록한 크래시가
  `--chunked-prefill-size -1`(capture 목록이 비는 것)로만
  가려져 있고 **cps>0 arm이 생기면 재개**된다.
- **overclaim 금지**: 새 모델에서 **아직 아무 correctness
  게이트도 돌지 않았다**. "Nano-9B에서 PD-mux가 정확하다"고 쓰지
  않는다 — 있는 것은 job 905835의 **부팅·서빙 사실**과 capacity
  측정(아래 "C")뿐이다.

## C. ★λ\*: 이 모델에서 **이미 측정돼 있다**(새 도구 불필요) —
단 단일 스칼라는 설계 문제

`c_capacity.sbatch`(사전등록 `PREREG_CAPACITY_2026-09-09.md` +
규칙층 감사 완료, `KNEE_BRACKETED` 존재 규칙·warmup 폐기·arm별
ladder·측정실패≠규칙실패)가 **같은 체크포인트**에서 측정했다
(job 905835, 8192-in/96-out, flashinfer, ctx 8192, PD-mux fixed
D): **D16 0.933 req/s · D44 0.675 · D92 0.187**(전부
`KNEE_BRACKETED`, D16은 `R5 MEASURED`).

- ★**λ\*는 arm별·shape별이다** — D에 따라 **5× 차이**(0.187 vs
  0.933). 그런데 `generate_campaign.sh`는 W1–W9에 **단일 스칼라
  λ\***를 쓰고 워크로드 입력 길이는 **32× 차이**가 난다(W3
  256-in vs W2/W4/W5 8192-in). ⇒ "기본값 4가 미측정"보다 **더
  깊은 설계 문제**이며, 열린 항목으로 등재한다: **"단일 스칼라
  λ\*를 W1–W9가 공유하는 것이 무엇을 뜻하는지"가 캠페인 0단계의
  사전등록 대상이다.**
- 1차 캠페인(Claim D)에 필요한 shape는 3개뿐: **W3 = (256 in,
  512 out) 전부** · **W4 = prefill phase (8192, 64) + decode
  phase (64, 512)**, 30s 페이즈 × 3 cycle. 기존 측정(8192-in/
  96-out)은 **W4의 prefill phase에 가깝고 W3와는 무관**이다
  (256-in은 prefill이 싸고 512-out이 지배 ⇒ λ*가 훨씬 높을
  것).

  ★★**정정(2026-09-13(3), doc-steward, λ0 사전등록 규칙층 감사
  §S8/N3 — 이 문서 최상단 배너 "A" 절 참조)**: 위 "decode phase
  (64, 512)"는 **오독이다**. `workloads.py:159-160`을 실행해
  확인하면 W4 decode phase = **(256, 512)**이고, 이는 W3
  전부와 **정확히 같다**(근사 아님) — 저장소 전체에 (in 64,
  out 512)는 존재하지 않는다. ⇒ 1차 캠페인에 필요한 shape는
  3개가 아니라 **2개뿐**: (256,512)[=W3=W4 decode phase]·
  (8192,64)[=W4 prefill phase]. 기존 측정(8192-in/96-out)이
  "W4 prefill phase에 가깝다"는 서술 자체는 옳다.
- 비용 감각: job 905835는 12 cell에 **1.11 GPU-h**(≈5.6분/
  cell). 참조 arm 1개 × 2 shape × 3–4 rate point ≈ **0.6–0.75
  GPU-h** 추정.
- **새 λ\* 도구를 만들지 않는다**: `e1_capacity_scan.sbatch`
  (일반 rate 격자·2 seed·warmup 폐기·추세 기반 elbow) 재타겟
  또는 `c_capacity.sbatch` ladder 확장이 권고다.

## D. 부수로 들어온 구현 사실 (커밋 `87213a9`, 이미 커밋됨) +
신규 게이트 #195

- **모델 구현이 manifest 안으로(17 → 24항목)**:
  `models/{nemotron_h,falcon_h1,granitemoehybrid}.py` **설치분**
  + `configs/{nemotron_h,falcon_h1,granitemoehybrid}.py`·
  `configs/mamba_utils.py` **해시 전용**. 신규 7줄은 뒤에
  append되어 **기존 17줄·순서 바이트 동일**. ★**읽기 규칙**:
  2026-09-13 이전 manifest는 같은 17파일을 검증하고 이후 것은
  7줄이 더 많다 ⇒ 교차-잡 서술은 **"기존 17 일치 + 신규 7항목
  존재"**로 쓰고 **"17/17 동일"로 쓰지 않는다**. 해시 전용
  근거: `configs/nemotron_h.py`가 `hybrid_override_pattern →
  layers_block_type`을, `mamba_utils.Mamba2StateShape`가 그것을
  `mamba_cache_per_req`로 바꾸므로 **λ\* 라벨이 딛고 선 값**이다.
  ★**이 결함의 실증**: job 905835의 manifest는 **17줄이고
  `models/` 항목이 `mamba2`·`zamba2`뿐**인데 그 job은
  **NemotronH를 12 boot 서빙했다** — 모델 구현 provenance가
  **0**이었다. 이것을 방법론 게이트로 등재한다(번호 이어서):

  **#195**(신규) — **서빙한 모델의 구현 파일이 provenance
  manifest에 없으면 그 캠페인은 모델 축에서 귀속 불가다.**
  대응 `CONSENSUS.md` §3 항목215(신설).

- correctness 하네스의 ctx 하드코딩 제거(`R2C_CTX` 또는 모델
  config 유도, 실패 시 `exit 2`; **Zamba2는 4096 유지로
  907100·907456 재현 보존**, 테스트 16건). `campaign.json`에
  `model`·`context_length` 추가(`schema: pdmux.campaign/v2`),
  `cuda_graph`는 **읽히게** 배선(모순 시 exit 2) — 제거하지
  않은 이유는 cudagraph-ON/OFF가 정본의 **운영점 축**이기
  때문. 전체 **463 tests OK**(직전 426, 신규 37), line-citation
  88/0.

## E. 정본 반영

`CONSENSUS.md` rev71→**rev72**(§3 항목215 신설[게이트 #195] +
항목103 追記[게이트#83, 교차-트랙 위험 Nano-9B-v2 재현] 반영)
— **판단 근거**: 신규 방법론 게이트 1건 + 기존 §3 항목103에
대한 실증 재현(追記)이 이번 세션 산출이므로 기존 선례("신규
방법론 게이트는 그 자체로 CONSENSUS rev 사유")를 그대로
적용한다. 여기에 더해 이번 회차는 (1) 사용자의 세 번째 구조적
결정(attention 백엔드 전환, 하위 트랙 다수에 파급) (2)
engine-porter의 모델 지원 검증 `GO`(이 프로젝트의 첫 Nano-9B-v2
서빙 가능성 확인) (3) λ* 미측정 선결의 부분 해소(측정은
존재하나 단일 스칼라 설계 문제로 재개방)를 포함해 단순 코드
사실 등재보다 무거운 갱신이다. `reports/paper/
{CLAIM_EVIDENCE_MATRIX,EXPERIMENT_ROADMAP,DOCUMENT_STATUS}.md`
갱신, `workspace/engine-port/RESUME.md` 갱신(백엔드 노브 등재),
`MEMORY.md`·`memory/{deconfound-measurement-lessons,
slo-aware-scheduling-track}.md` 갱신(항목 신설 — topic 파일
내부 순번, `CONSENSUS.md` §3 항목과는 별개 축). **커밋 금지**
(이번 등재는 문서 작업만; 코드 변경은 커밋 `87213a9`로 이미
완료돼 있고 이 세션에서 추가 커밋을 만들지 않는다).

GPU 장부: 이 배너 전체 **GPU 0**(job 905835의 1.11 GPU-h는
`longctx_conflict` 트랙의 기존 지출로 그 트랙 누적 15.42 GPU-h
안에 이미 포함돼 있다 — 재확인만, 이중 계상 아님). `longctx_
conflict` 15.42 GPU-h·R2 correctness 0.43 GPU-h 장부 둘 다
불변.

이전: 2026-09-13(doc-steward — **★정본 산술 정정(최우선,
"약 270"은 중복계수 → 고유 225/가능 180) + X3 사전등록 규칙층
감사 `GO-with-caveats` 등재(死因 0·차단 D1–D13, GPU 0·미실행,
아래 "C"로 같은 세션 안에서 즉시 무효화) + 사용자 결정 2건
(①모델 교체 Zamba2-2.7B→NemotronH Nano-9B-v2-Base ②1차 캠페인
범위=Claim D+P3) + 새 선결 1건(λ*/`sustainable_rate` 미측정) +
신규 게이트 5건(G-X3-1…5, #190–194)**). 새 성능 판정 0건·GPU
0·Claim D/E 등급 불변(둘 다 미검증)·HE0·정책 순위·stake #1
전부 불변. `CONSENSUS.md` rev70→**rev71**(판단 근거는 아래
"E. 정본 반영" 참조).

## A. ★정본 산술 정정 (최우선)

2026-09-12(5)이 등재한 **"생성 405 run 중 약 270이 현 구성으로
실행 불가"는 중복 계수**였다. X3 사전등록 규칙층 감사
(claims-auditor, GPU 0, `workspace/engine-port/results/
r2_correctness/x3_prereg/VERDICT_x3_rules_2026-09-13.md` §10.1)
가 포함-배제로 재계산했다:

- **고유 차단 = 225** = 135(W2/W4/W5, 8192-토큰 프롬프트가
  Zamba2-2.7B ctx 4096 초과) + 90(B2/B8, `requires_offline_
  oracle`) + 45(B6, `PDMUX_MODEL_PROFILE_PATH: unbound`) −
  30(W2/4/5 × {B2,B8} 교집합) − 15(W2/4/5 × B6 교집합).
- **실행 가능 = 180** = W1·W3·W6·W7·W8·W9(6종) × 5 rep ×
  B0·B1·B3·B4·B5·B7(6종).

아래 "D-1" 절(2026-09-12(5) 배너 원문, "약 270")·
`EXPERIMENT_ROADMAP.md`(구 `:845–853`)·`CLAIM_EVIDENCE_MATRIX.md`
Claim D 행에 **정정 追記**를 부착했다(원문 보존 + 정정 표시,
삭제 없음). 이 정정 수치는 아래 "C-2"(1차 캠페인 범위 결정)의
근거이기도 하다.

## B. X3 사전등록 규칙층 판정 `GO-with-caveats`(GPU 0,
**미실행**) — 아래 "C"로 즉시 무효화

판정서 원문 전사 = `workspace/engine-port/results/r2_correctness/
x3_prereg/VERDICT_x3_rules_2026-09-13.md`(303행, sha
`1ed37b65…`). 대상 사전등록 `PREREG_X3_RUNNERCONF_2026-09-13.md`
(sha `cf887161…`).

**판정**: 死因 0(자유표면 **14개** 전수, 등록 라벨 반전
**0건**) + 차단 **D1–D13**(전부 GPU 0). 긍정 확인: ★**무수정
판정 규칙(`r2_correctness_check.py`, `ec355e17…`)을 "907100에서
`trace_forced` 레코드만 제거한" 반사실 입력으로 실제 실행**해
예보 `NO_VERDICT_UNREALIZED`를 재현(`failures=[] infra=[]`,
실패 하위검사는 **O2(+O3) 한정**, O1·O4는 32/32 성립,
**B7·B8 통과**).

★**사실 정정 3건**(등재 필수):

1. **F1이 측정하지 않은 것을 측정했다고 말했다** — X3 job은
   **러너 코드를 한 줄도 실행하지 않는다**(ctx 4096은 당시
   `r2_correctness.sbatch:177` 리터럴, `pdmux_eval.context_limit`
   호출처는 `r2_eval.sbatch:180`[갱신 2026-09-14 — 원문은 `:100`,
   커밋 `87213a9`+W4가 이동시켰다; 이 파일은 이 문서의
   `check_line_citations.py` `__scope__` 밖이라 정정되지 않고
   있었다, 신규 게이트 #238] 단 1곳). ⇒ **선결 #2의 "러너
   설정" 항목은 X3가 닫지 않는다.** ★**재검증 필요 플래그
   (2026-09-14, doc-steward — 관찰만, 판정 아님)**: `r2_correctness.
   sbatch`는 이 패스 시점에 engine-porter가 진행 중인 별도
   개정(이 패스는 그 파일에 손대지 않는다) 대상이고, 그 파일에
   이미 `python -m pdmux_eval.context_limit`을 호출하는 줄(`:244`)
   이 나타나 있다 — 완결되면 "단 1곳"과 "ctx 4096 리터럴" 둘 다
   재확인이 필요할 수 있다(현재는 미결정, engine-porter WIP 종료
   후 별도 세션에서 판단).
2. **F2의 "eval 캠페인 계측 구성으로 이전" 주장은 3중 과대** —
   (i) 관측자 4개 중 **2개만** 시험하고 **남긴 쪽이 boot당 더
   자주 발화**(Decode 로그 2580–2617줄 vs 제거되는 forced 스냅샷
   1263–2285건) (ii) 엔진 소스가 38c1aca로 핀되는데 캠페인은
   HEAD를 돈다(`src/multiplex` 4파일 +564/−39, `dual_worker.py`
   +180 = TD hot loop) (iii) 타이밍→토큰 결합이 실측된 유일한
   층 **C는 제외**됐다(같은 구성 L-L에서도 4/32 불일치).
3. **기전 크기 오류**: 사전등록의 "15–19건 중 하나가 8개 창에
   들어와야"는 희소성을 암시했으나 실제 창별 적중은 **probe-boot
   26/32 = 81%**다. F3가 성립하는 이유는 희소성이 아니라
   **32칸 전칭 요구**다.

**묶음(seed 1000 + forced OFF)은 confound #10이 아니다** —
greedy argmax·`top_k=1`·기본 `fcfs`·고정 warmup ids·모델 로드 후
seeding으로 **seed→토큰 경로가 없다**(sglang 소스 8곳 인용,
판정서 §4). 단 "두 변인 모두 증명된다"는 과대 — seed 쪽은
**코드 연역**이다(D12). 엔진 소스 핀의 충분성은 `sha256sum -c
job_907100/runtime_source_manifest.sha256` **실측**으로
확인했다(핀 전 13 OK / 4 FAILED = 정확히 핀 대상 4개).

필수 병기 **X3P-1…7** · 인용 금지 **X3C-1…9**(판정서 §12·§13,
문자 그대로 승계): X3P-1(스코프 튜플)·X3P-2(잔여 관측자 —
더 자주 발화하는 쪽이 남았다)·X3P-3(엔진 소스 축 — 캠페인
바이너리는 미인증)·X3P-4(검정력·양성대조 결손)·X3P-5(판정
단어 — 구성상 그렇다)·X3P-6(교차-잡 귀속의 비대칭)·X3P-7
(승계: R2C-1…16·P-1…7·X1P-1′…9·X1C-10…14·B3C-1…5·OSP-1…6·
OSC-1…8 전부 불변, 성능 수치 인용 전면 금지, Claim D/E 등급
불변·HE0·정책 순위·stake #1 불변). X3C-1…9 = "X3가 러너
설정/eval 계측 구성을 인증했다"·"서버가 러너 인자로 뜬다는
것을 보였다"·"`0/96`이 계측 무해함을 보였다"·"`PASS`가 나오면
선결 #2가 넓어진다"·"`NO_VERDICT_UNREALIZED`가 게이트 실패다"·
"X3가 선결 #4b를 닫았다/절반을 닫았다"·"O-only 불일치가 토큰을
바꾼다는 증거"·`cross_job_compare.txt`의 `reading (F2)` 문단
인용·907100/907456과의 지연·스루풋·GPU-h·decode step 수 비교 —
**전부 인용 금지**(전문은 판정서 §13).

신규 방법론 게이트 5건 **G-X3-1…5**(번호 이어서 **#190–194**,
`CONSENSUS.md` §3 항목210–214):

1. **G-X3-1**(게이트 #1의 3번째 층) — 관측자를 끄는 실험은
   "끈 관측자"와 "남긴 관측자"의 발화율을 함께 등록하라.
2. **G-X3-2**(게이트 #110/G-OS-1 대칭, 긍정 사례) — "구성상
   그렇게 된다"는 예보는 논증하지 말고 무수정 판정 규칙을 그
   구성의 반사실 입력으로 실제 실행해 보여라.
3. **G-X3-3**(#174·G-OS-4 인접) — 감사·강화된 분석 도구를
   다른 처치에 재사용할 때 그 도구가 *인쇄하는* 해석 문장의
   부호가 반대인지 확인하라.
4. **G-X3-4** — 귀무대조의 적용 범위를 층별로 적어라
   (`907032→907100`은 **S층 전용**).
5. **G-X3-5**(우선순위 규율) — "트랙을 연다"는 주장은 블로커
   목록과 대조한 뒤에만 쓰라.

★**감사자 자기 철회(3회차 누적 4건)**: 판정서 §8·§10에서
자기 OS 판정서(§8) 문장 1건을 추가 철회했다 — **"X3는 성능
트랙 전체를 연다"는 거짓**이다. 캠페인 블로커 3개(W2/W4/W5
ctx·B2/B8 oracle·B6 hybrid profile 산출물 저장소 전체 0건) 중
X3가 제거하는 것은 **0개**다.

## C. ★사용자 결정 2건 (2026-09-13) — 모델 교체와 1차 캠페인 범위

### C-1. 모델 교체: Zamba2-2.7B → NemotronH Nano-9B-v2-Base

사용자가 결정: `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base`
(`NemotronHForCausalLM`, `max_position_embeddings=131072`,
hf_cache 보유)로 4모델 캠페인의 Zamba2 슬롯을 교체한다. 근거:
W2/W4/W5가 **8192 토큰 입력**을 내보내는데 **Zamba2 전
계열(1.2B/2.7B/7B)이 ctx 4096**이라 위 "A" 절의 135 run이
구조적으로 불가였다. ctx 131072면 **트레이스 무수정** 실행이
가능하다. 정본 4모델의 NemotronH는 `Nemotron-H-8B-Base-8K`
(ctx 8192)이며, 그것으로 가면 8192+64가 `ctx−2`를 넘어
**여전히 트레이스 1% 축소가 필요**했다 — 사용자는 Nano-9B-v2를
택했다.

★**이 결정의 귀결(등재)**:

- **X3 사전등록(rev1)은 이 결정으로 무효화**된다 — Zamba2
  기준 설계이고, X3 감사 §10.3이 "모델 교체면 X3는 전량
  무효(ctx·커널 shape·프로토콜 전부 변경)"라고 이미 예고했다.
  아티팩트(`x3_prereg/PREREG_X3_RUNNERCONF_2026-09-13.md`·
  `VERDICT_x3_rules_2026-09-13.md`·`forced_sample_dependency*`)
  는 **이력으로 보존**하고 상단에 SUPERSEDED/무효 배너를
  달았다. GPU 0 지출, 결과 없음 — 위 "B" 절의 방법론 게이트
  5건·X3P/X3C 문안은 **방법론 지식으로는 유효**하나 이 job
  자체는 다시 사지 않는다.
- **OS 사전등록(`GO-with-caveats`, 2026-09-12(5))도 Zamba2
  기준**이므로 같은 무효화 대상일 가능성을 여기 표시한다(C 층
  등가류·초점 단위 C01·C10·C19는 Zamba2 907100의 관측이다) —
  OS 자신을 지금 무효로 재등재하지는 않는다(아직 아무도 OS를
  구매 결정한 적이 없어 "취소"할 대상이 없다), 단 향후 누구든
  OS를 사려 하면 이 스코프 동결부터 대조해야 한다.
- **907100·907456·X1의 모든 결론은 Zamba2-2.7B 한정으로
  동결**된다(R2C-1…16·X1P-*·X1C-* 전부 불변, 다른 모델로 이식
  금지는 이미 등재돼 있다 — 이번 등재는 그 동결을 재확인·명시할
  뿐이다). **새 모델에서는 R2 correctness 게이트를 새로 쌓아야
  한다**(907100 등가물이 존재하지 않는다).
- **미검증 선결(신설)**: Nano-9B-v2는 이 프로젝트가 **한 번도
  서빙한 적 없다**. engine-porter가 CPU-only 지원 검증
  중이며(config 필드 대조·가중치 완전성·kv-cache cell_size·
  메모리 산술) **GO/NO-GO가 나오기 전에는 어떤 GPU 계획도
  확정이 아니다**. (overclaim 금지: "Nano-9B-v2로 돌릴 수
  있다"고 쓰지 않는다.)
- ★★**교차 트랙 위험(신규 발견, 등재 필수)**: R2 correctness
  게이트의 조건 튜플은 `--attention-backend triton`을
  포함한다(X3P-1). 그런데 **TC1 트랙(2026-08-28)이 이미
  NemotronH 계열+`triton`=부팅 거부**(`server_args.py:1959`
  하드 assert, job 896776)를 확인했다 — 백엔드 강제 상호배타
  교락(신규 게이트 #83/`CONSENSUS.md` §3 항목103, 교훈 88
  계열). **engine-porter의 CPU-only 검증에 이 boot-refusal
  재확인이 포함돼 있는지 이 doc-steward 등재 시점에는
  불확실**하다 — GO/NO-GO 판정 전에 반드시 확인해야 한다. 만약
  Nano-9B-v2도 `triton`으로 안 뜨면 R2 correctness 게이트
  자체를 다른 attention backend로 재설계해야 하고, 그러면
  `--attention-backend triton` 조건에 묶인 §1-3의 결과들과
  마찬가지로 새 게이트의 조건 튜플 자체가 Zamba2 판본과
  달라진다(단순 "모델만 교체" 이상의 재설계).
- ★**provenance 구멍(신설)**: `sync_engine_tree.sh:94`가
  "`nemotron_h`/`falcon_h1`/`granitemoehybrid`는 수동 복사"라고
  적고 있고 `runtime_source_manifest.sha256` **17항목에 모델
  파일은 zamba2·mamba2뿐**이다 ⇒ 지금 상태로 NemotronH
  캠페인을 돌리면 **모델 구현이 해시 귀속에서 빠진 채
  측정된다**. engine-porter가 manifest 이관 중(항목 수
  17→N 증가와 기존 job 비교 절차에 대한 영향을 함께 보고할
  예정) — 진행 중, 결과 없음.

### C-2. 1차 캠페인 범위: Claim D + P3(profile)까지

사용자가 결정: 1차 캠페인은 **Claim D + P3(profile)까지**로
제한한다. 구조적 사실:

- **Claim D의 성능 게이트 P2는 `B1`(legacy fixed)과
  `B4`(true_dual fixed) 두 arm만 필요**하고, 로드맵 P2
  acceptance가 요구하는 워크로드는 **decode-heavy(W3) +
  alternating(W4)**뿐이다.
- 따라서 위 "A" 절 고유 차단 225 중 대부분(B2/B8/B6 = 135)은
  **Claim E·oracle 쪽이며 Claim D와 독립**이다.
- **P3는 B6(hybrid)의 선결**이고 hybrid profile JSON은 저장소
  전역 **0건**이다(`find` 결과 0, X3 감사가 확인).

**실행 순서(로드맵에 등재)**: **(0) λ\* 측정 → (1) 새 모델 R2
correctness 게이트 → (2) P1 observer effect(paired ON/OFF) →
(3) P2 = W3+W4 × {B1,B4} × 5 rep → (4) P3 offline profile →
(5) Claim E(B6)**. 각 단계는 **개별 사전등록 + 규칙층 감사**
대상이다. 지금까지 완료된 것은 **0단계뿐**(그리고 0단계도
아래 "D"처럼 아직 미실행).

## D. ★새 선결 1건 등재(이번에 발견) — λ*(`sustainable_rate`)
미측정

`generate_campaign.sh:10`[HIST]의 **`sustainable_rate`(λ\*)가
측정값이 아니라 기본값 `4`**였다. 워크로드가 전부 λ\*의 분수로
정의되므로(W1 0.60·W3 0.80·W8 0.90·W9 1.10) λ\*가 틀리면
"near saturation"·"overload" 라벨 자체가 틀린다 — **프로젝트
방법론 게이트 #6("용량 먼저 측정")의 직접 위반**이었다(`reports/
paper/EXPERIMENT_ROADMAP.md` "공통 방법"이 이미 "먼저 B1
sustainable SLO rate λ\*를 모델별로 측정한다"고 적고 있으나
이 코드 경로에서는 그 절차가 **실행되지 않았다**). ⇒ **모델별
λ\* 측정이 캠페인의 0번째 단계**임을 위 "C-2" 실행 순서에
명시했다.

★**반증(2026-09-14, doc-steward — engine-porter W4 구현 확인,
커밋 `8507cee`, 등재 계기: 정본 인용 등록성 검사)**: 위 기본값
`4`는 더는 존재하지 않는다 — `generate_campaign.sh:30-46`이
`PDMUX_LAMBDA_STAR_TABLE` 미설정 시 fail-closed(`exit 2`)로
대체했고 `PDMUX_SUSTAINABLE_RATE` 자체도 명시적으로 거부한다
(단일 스칼라로 W4 phase별 용량 5×차를 표현할 수 없어 per-shape
테이블로 교체, `benchmarks/pdmux_eval/lambda_star.py`). 상세는
이미 이 배너의 2026-09-14 세션 "A. 사용자 결정 4건" 항목1
(W4=phase별 독립 λ\*)과 `EXPERIMENT_ROADMAP.md` "### P2" "열린
항목 종결(2026-09-13/14)"에 등재돼 있다 — 이 문단은 **이 자리의
원문 인용만** 최신 상태로 정정한다. **달라지지 않은 것**:
모델별 λ\*가 실측됐다는 뜻은 아니다 — 캠페인 0단계 λ0 사전등록은
여전히 `NO-GO`(위 "B. 캠페인 0단계(λ*) 사전등록 `NO-GO`" 참조,
게이트 #6 잔류). 이 인용이 그동안 정정되지 않았던 이유:
`check_line_citations.py`의 이 문서 `__scope__`가 `controller.py`/
`profile.py`(+`workloads.py`) 패턴만 등록해 `generate_campaign.sh`
인용은 검사 대상 밖이었다(신규 게이트 #238, 아래 "방법론 게이트"
참조).

비용 감각(추정, 미측정): 실행 가능 180 run × run당 ~3분 ≈
**9–13 GPU-h**(Zamba2-2.7B 기준 추정이며 **9B 모델은 더
비싸다**) vs 이 트랙 누적 지출 **0.43 GPU-h**(오늘 GPU 0으로
불변).

## E. 정본 반영

`CONSENSUS.md` rev70→**rev71**(§3 항목210–214 신설
[G-X3-1…5], 정정 1건 + 사용자 결정 2건 + 새 선결 1건 반영) —
**판단 근거**: 신규 방법론 게이트 5건이 이번 세션 산출이므로
2026-09-12(2)가 확립한 선례("신규 방법론 게이트는 그 자체로
CONSENSUS rev 사유")를 그대로 적용한다. 여기에 더해 이번
회차는 (1) 정본 산술 정정(고유 225/가능 180) (2) 사용자의
구조적 결정 2건(모델 교체·캠페인 범위, 하위 트랙 다수에 파급)
(3) X3 사전등록의 같은 세션 내 발행→즉시 무효화를 포함해 단순
코드 사실 등재보다 무거운 갱신이다. `reports/paper/
{CLAIM_EVIDENCE_MATRIX,EXPERIMENT_ROADMAP,DOCUMENT_STATUS}.md`
갱신, `MEMORY.md`·`memory/{deconfound-measurement-lessons,
slo-aware-scheduling-track}.md` 갱신(항목188–192 신설 —
topic 파일 내부 순번, `CONSENSUS.md` §3 항목과는 별개 축).
**X3·OS·B3 전부 미실행이므로 어떤 성능 결과도 등재하지
않았다.**

GPU 장부: 이 배너 전체 **GPU 0**. `longctx_conflict` 15.42
GPU-h·R2 correctness 0.43 GPU-h 장부 둘 다 불변.

이전: 2026-09-12(5)(doc-steward — **정정 2건(최우선) +
B3/OS 사전등록 규칙층 판정 등재(둘 다 미실행, GPU 0) + A1/A2/A4
구현 등재(커밋 `09a8075`·`ae7830e`·`3153260`, GPU 0)**). 새
성능 판정 0건·Claim D/E 등급 불변(둘 다 미검증)·HE0·정책 순위·
stake #1 전부 불변. `CONSENSUS.md` rev69→**rev70**(판단 근거는
아래 "F. 정본 반영" 참조).

## A. 정정 2건 (최우선)

1. **★철회**: 2026-09-12(3)이 등재한 "B3 두 job 합쳐 n=4/arm ⇒
   프로젝트 게이트 3(n≥4) 충족"(X1 판정서 §12-2 유래 문장, 아래
   "다음 실험 gate" 항목4·`EXPERIMENT_ROADMAP.md`·`CONSENSUS.md`
   에도 동일 문구로 승계돼 있었음)을 claims-auditor가 B3 판정서
   §8에서 **철회**했다: 한 job의 4 boot은 같은 노드·같은 물리
   GPU·같은 warmup·같은 `TRITON_CACHE_DIR`을 공유하는 **순차
   실행**이라 독립 런이 아니고, 프로젝트 게이트 3은 *정책 비교
   베이스라인 분산*을 위한 **독립 런 n≥4**를 요구한다. 해당
   문장에 취소선이 아니라 ★철회 표시를 부착했다(아래 "다음 실험
   gate" 항목4·`EXPERIMENT_ROADMAP.md`·`CONSENSUS.md` 3곳 전부).
   인용 금지 **B3C-3** 신설.
2. **산술 정정**: X1 결과 판정서(`workspace/engine-port/results/
   r2_correctness/audit_x1_2026-09-12/VERDICT.md`) §3.3 표의
   907100 행 "cross-arm 합 22"는 **25**가 맞다(7+7+4+7; 6쌍
   pairwise 합 32 − within[L-L 4+TD-TD 3=7] = 25 — 22를 쓰면
   7+22=29≠32로 그 표가 자기모순이다. 907456 행의 cross 16은
   원래 맞다). B3 판정서 §1·OS 판정서 §1이 각각 **독립
   재계산**으로 확인했다. 이 수는 `CONSENSUS.md`·이 문서·
   `CLAIM_EVIDENCE_MATRIX.md` 본문에 합계로 직접 인용된 적은
   없었다(전부 "cross 4–7/32" 범위 인용에 그침) — 근원 판정서
   자체(`audit_x1_2026-09-12/VERDICT.md` 상단)에 정정 追記를
   부착해 향후 인용의 근거를 고쳤다.

## B. B3 사전등록 `NO-GO` (GPU 0, **미실행**)

판정서 원문 전사 = `workspace/engine-port/results/r2_correctness/
b3_prereg/VERDICT_b3_rules_2026-09-12.md`(236행, sha
`e6c5019d…`). 설계 = 동일-arm 4 boot × 2 job(`L L L L`/
`TD TD TD TD`, 0.31 GPU-h) — 위 "다음 실험" 항목1·2가 구체화된
것. **死因 N2 발화(반전 3건, 전부 수치 있음)**:

1. **R1**: E4(불일치 2-2 단위 수)를 원 카운트가 아니라 **2-2 단위
   총수(n_22)로 조건부** 읽으면, **실제 4-boot job 907456의
   n_22=0**에서 "귀무로 재현 안 됨"(분기1)이 "분모 0, 측정
   불가"(`NOT-MEASURED`)로 뒤집힌다.
2. **R2**: F4 문턱 "≥2"의 여집합에 **등록 라벨이 없다** — 무라벨
   비중이 n_22=2/job에서 59.3%, 3에서 46.1%(이 구간의 최빈값)로,
   분기1("어느 job이든")과 분기2("두 job 모두")가 **비상보
   수량자**라 사후 어느 쪽으로도 채울 수 있다.
3. **R4**: 한 boot의 `phase_c`만 비워도(허용조건이 **파일 4개
   존재**만 검사, 레코드 존재는 안 봄) 스크립트가 `units=0 …
   E4=0`을 인쇄해 **측정 실패가 분기2("귀무로 재현되지 않았다")를
   발화**시킨다(실증).

부수 발견: selftest 변이 **4종 중 3종 통과**(E4의 모양-이름 배선을
바꾼 변이 `M2`가 통과하면서 907100에서 E4를 **3 대신 0**으로
출력 — 결론 자체가 무검증) · **TD-only job은 `s_ref_ok`의 공허
참**(`r2_correctness_check.py:471`, `all(...)` over `l_boots=[]`)
**때문에 같은 사건이 `L L L L`=`NO_VERDICT_INFRA` /
`TD TD TD TD`=`VERDICT FAIL`로 갈려 기록**된다(S 토큰 1개 섭동으로
실증) · TD-only job에서는 B6(퇴화 검사)가 job 수준 계산이라
**아예 계산되지 않는다**.

신규 게이트 **G-B3-1…5**(번호 이어서 `#180–184`, `CONSENSUS.md`
§3 항목200–204): (1) 직전 회차의 허위-0 수리를 새 스크립트에
옮길 때는 granularity를 재도출하라(X1 D8의 "파일 4개 미만"
가드가 "빈 phase_c"에서 재발) (2) selftest는 핵심 추정량의
*이름-모양 배선* 자체를 assert해야 한다(`M2`가 결론을 무검증으로
통과시킴) (3) 귀무대조의 문턱은 **그 귀무에서의 발화 확률과
함께** 등록하라("E4≥2"는 균등 귀무에서 n_22=4일 때 40.7%로
발화) (4) 공허 수량자가 arm에 따라 verdict 라벨을 바꿀 수
있다(`all()` over `[]`가 TD-only에서만 다른 라벨을 만듦) (5)
**감사자의 "유일한 값싼 길" 처방은 그 자신이 사전등록 심사를
받기 전까지는 설계 권고가 아니다**(자기 적용 — 감사자 자신의
X1 §12-1이 조건부 귀무 기존 존재·분모 이중성·가설 A/B 동시 소거를
셋 다 놓쳤고, §12-2는 위 정정1의 거짓 주장을 담았다).

**구매 권고(규칙층 등급과 별개 축)**: rev2가 되어도 B3를 먼저
사지 않는다 — 같은 0.154 GPU-h로 `R2C_ORDER="L TD TD L"` 1
job(아래 "C. OS")이 arm/position을 job 내부에서 3-way로 가르고
동시에 운영점 게이트 런을 1→2로 늘리는 유일한 선택지다(B3 2
job은 구성상 게이트 런을 **0개** 추가한다 — 양쪽 다
`NO_VERDICT_INFRA`).

## C. OS(순서 교환) 사전등록 `GO-with-caveats` (규칙층만, **미실행**)

대상 `.../os_prereg/PREREG_OS_ORDERSWAP_2026-09-12.md`(sha
`512c7443…`), 판정서 원문 전사 `.../os_prereg/
VERDICT_os_rules_2026-09-12.md`(290행, sha `4c7cfa94…`). 설계 =
`R2C_ORDER="L TD TD L"` 1 job(0.154 GPU-h). 판정: **死因 0**(자유
표면 **20개** 전수 시험, 등록 라벨 반전 **0건**), **차단 조건
D1–D8**. 긍정 확인: 무수정 checker를 이 순서로 **실제 실행**해
6쌍 해소·`O_null_control_complete=True`·`B6=0/56`·**`VERDICT
PASS`** 확인(B3가 구성상 죽던 자리에서 전 경로가 돈다).

★**필수 등재 사실 3건**:

1. **D2 — 핀은 장식이 아니라 하중재다**: 작업 트리
   `src/multiplex/dual_worker.py`가 38c1aca 대비 **+172줄**(A2의
   H2가 계측 발행을 `future.set_result` 앞으로 옮긴 **true-dual
   hot loop** 변경, manifest 17항목 중 2번)이고 **미커밋**이다.
   핀(`git checkout 38c1aca -- .../src/multiplex`) 없이 OS를
   돌리면 부팅 순서와 엔진 소스를 동시에 바꾼 것이 돼 confound
   #10으로 무효다. (★D1은 **이미 해소**: 아래 "D" 항목의 A1
   작업이 커밋 `09a8075`·`ae7830e`·`3153260`으로 들어가
   `git status --porcelain -- src/multiplex`가 비었다 — OS
   판정서가 지적한 "미커밋 상태" 위험은 이 doc-steward 등재
   시점에는 이미 사라졌다.)
2. **D3 — ★철회**: "`L TD TD L`이 'L이 항상 먼저'라는 잠복
   조건을 제거한다"는 **거짓**이다(출처는 감사자 자신의 B3 §7,
   OS 판정서에서 철회). L1은 이 순서에서도 여전히 위치 1이다.
   제거되는 것은 **엄격 교대·동일 arm 비인접·마지막 boot이
   TD** 셋뿐이고, 제거되지 **않는** 것은 **L-first·
   TD-never-first·warmup은 항상 legacy·위치1≡L1**이다.
3. **D4 — 가장 중요한 과학적 caveat**: `{1,4}|{2,3}`(이 순서의
   arm 경계)은 동시에 **"외곽 vs 중앙" 위치 모양**이라 이 설계는
   둘을 **구별하지 못한다**. 907100에서 이 모양의 2-2 단위는
   **1건(C24)**뿐이고 균등 귀무 기대치 1.33(n_22=4) 미만이라
   증거는 없다. 데이터에 보이는 유일한 위치 구조는 **위치1
   고립**(싱글턴 907100+907456 합산 7건 vs 위치2/3/4의 3/4/4,
   χ²=2.0·df=3·**p≈0.57, 미확립**)이며, 그것은 **3-1을 만들어
   focus-3 단위(m)에서 제외**된다.

**필수 병기 OSP-1…6 · 인용 금지 OSC-1…8**(판정서 §10·§11에서
문자 그대로 승계, 실행 시 그대로 등재할 것): 특히 **OSP-1**(이
job이 실행되면 그 결과는 "n=2를 채웠다"가 아니라 "운영 수치
구성에서 서로 다른 부팅 순서의 두 번째 게이트 런"이라는
**커버리지 진술**일 뿐이다) · **OSP-4**(F3의 p=0.037은 3 초점
단위의 **독립성 가정**에 전적으로 의존 — n_eff=1이면 1/3;
family-wise[arm 또는 parity]는 2/27=0.074; **재현 대상인
907100 관측 자신이 균등 귀무에서 1/3 확률 사건**) · **OSP-5**
(F3[초점 3단위]과 F4[전체 32단위 2-2]는 **항상 함께** 인용,
초점 3단위는 907100의 arm-비지지 유일 단위 C24를 배제한다) ·
**OSC-4**(n≥4/게이트 3 충족 주장 금지, B3C-3 승계) ·
**OSC-5**(선결 #2[GPU correctness 동치] 진전/닫힘 주장 금지 —
부팅 순서를 바꾼 두 번째 PASS는 같은 S16+O8 프로토콜의 커버리지
확장일 뿐, 동시 부하[C 층] 동치는 이 job에서도 성립하지 않는다).

신규 게이트 **G-OS-1…5**(번호 이어서 `#185–189`, `CONSENSUS.md`
§3 항목205–209): (1) 직전 판정서의 코드 사실은 **그 사실이
참이었던 트리 상태와 함께** 인용하라(B3 §10/R10의 "값은
어차피 같다"는 `HEAD` vs `38c1aca` 비교였을 뿐, 그 사이
`dual_worker.py` +172줄이 들어와 명제가 거짓이 됨 — 게이트#110의
실패 사례) (2) "소스를 핀하라"는 처방은 **미커밋 작업 확인
후에만** 실행 가능하다(`git checkout -- path`는 reflog 없이
파괴) (3) 교락 해소 설계는 **깨는 교락**만이 아니라 **새로
만드는 교락**을 열거해야 한다(부팅 순서를 바꾸면 arm 경계가
다른 위치 모양[외곽-중앙]으로 이동할 뿐) (4) 사후 선택된 단위의
사전등록 재현은 **재현 대상 자신의 귀무 확률**(1/3)과 **선택에서
배제된 반대 증거 단위**(C24)를 함께 등록해야 한다 (5) **긍정
사례** — "구성상 게이트가 도는가"는 논증하지 말고 판정 규칙을
그 구성으로 **실제 실행**해서 보여라(무수정 checker를 907100
아티팩트+OS 순서 `boots.txt`로 실행해 `VERDICT PASS` 확인).

★**구매 순서 권고 변경**: **A1 커밋 → X3 → OS**. 근거: OS는
등록 스스로 인정하듯 **어떤 선결도 닫지 않고**, 재현하려는
907100 관측 자신이 **1/3 확률 사건**이며, m=0으로 아무것도 못
재고 끝날 확률이 **미추정**(운영점 선행 표본 n=1). 반면 **X3는
선결 #4b(observer effect)의 토큰 쪽 절반이자 성능 트랙 전체
(`results/r2_eval`)의 해제 조건**이다. `EXPERIMENT_ROADMAP.md`
"P1/P2"·R2 correctness 절의 실험 순서를 이에 맞춰 갱신했다.

## D. A1·A2·A4 구현 (커밋 `09a8075`·`ae7830e`·`3153260`, GPU 0)

- **A1 러너 수리**(`09a8075`): `r2_eval.sbatch:11`의
  `dirname "${BASH_SOURCE[0]}"`가 **sbatch의 spool 사본** 때문에
  `engine_root`를 scratch로 잘못 잡던 결함(재현 확인) → 고정
  절대경로(`r2_correctness.sbatch:99`의 기존 선례를 따름).
  (a)sbatch (b)로그인 (c)array 3경로 테스트 + 되돌린 변이 음성
  대조 포함. **ctx**: `PDMUX_CONTEXT_LENGTH`는 읽는 곳 1개·쓰는
  주체 0개였고, 이제 sbatch가 **서빙 모델 `config.json`에서
  유도**한다(Zamba2-2.7B→4096). 신규 `context_limit.py`가 trace를
  사전 스크리닝해 `input+output > ctx−2`(엔진이 `max_new_tokens`를
  조용히 줄여 적은 decode 작업량으로 채점되는 경우)까지 거부.
  `engine_bench_runner.sh`가 `server_args.txt`를 남기고
  `PDMUX_DRY_RUN=1`을 지원.
- **A2 H2–H5**(`ae7830e`): H2 `task_count` 가시화 시점 정정(의미
  불변) · H3은 **재정의 없이 새 이름 5개 추가**(기존 값 비트
  동일 고정) · H4 `decode_step_count` 방출 제거(R1 observer
  가드라 항상 0이었음) · H5 `worker_overlap_ratio` 죽은 선언
  제거(살아있는 형제 `host_worker_overlap_ratio`만 남김).
  **판정 규칙 무영향**을 `SNAP_KEYS`를 텍스트로 읽어 생산자
  존재를 검증하는 테스트로 고정. `controller.py:54-55`의 동일
  결함(`prefill_idle_ratio`/`decode_idle_ratio`)은 **정본이 그
  줄을 "죽어 있다는 증거"로 인용하고 있어 일부러 남기고
  테스트로 고정**.
- **A4**(`3153260`): `controller.py`·`profile.py` 앵커
  38항목/21키 등록(50 → **88 compared, 0 violation**), 2줄
  밀기 변이에서 **38/38 발화**. 도구 결함 2건 수리: (i) `CITE`가
  콤마 목록을 파싱 못해 실제로 이동한 앵커 2개가 구조적으로
  안 보였다 (ii) 고칠 수 없는 역사 인용 51건 때문에 부분 등록을
  매니페스트 `__scope__`로 선언(감시 범위는 좁히되 검사 강도는
  낮추지 않음, 테스트로 고정). 매니페스트 =
  `workspace/engine-port/scripts/discipline/line_citations.json`
  — 이 파일이 이제 controller.py/profile.py 줄 인용의
  앵커-검증 SSOT다.
- 전체 CPU 회귀 **426 tests OK**(직전 362).

### D-1. ★사용자 결정 대기 항목으로 등재 (가장 중요)

**생성되는 405 run 중 약 270이 현 구성으로 실행 불가**다: **W2/
W4/W5(135 run)** 트레이스가 **8192 토큰 프롬프트**를 내보내
Zamba2-2.7B ctx 4096으로 서빙 불가 · **B2/B8(90 run)**
`requires_offline_oracle`로 exit 2 · **B6(45 run)**
`policy_adapter.sh:57`에서 `PDMUX_MODEL_PROFILE_PATH: unbound
variable`. ⇒ **Claim E 캠페인(`results/r2_eval`)은 "모델 교체
또는 워크로드 교체"라는 실험 설계 결정 없이는 돌 수 없다.**
`EXPERIMENT_ROADMAP.md`·이 문서의 열린 항목에 **사용자 결정
대기**로 명시했다(아래 "다음 실험 gate" 항목4 追記 참조).

★★**정정(2026-09-13, X3 규칙층 감사 §10.1, doc-steward 등재
— 이 문서 최상단 배너(2026-09-13) "A" 절 참조)**: 위 "약 270"은
**중복 계수**였다 — 포함-배제로 재계산하면 **고유 차단 =
225**(135+90+45 − 30[W2/4/5×B2,B8] − 15[W2/4/5×B6]),
**실행 가능 = 180**(W1·W3·W6·W7·W8·W9 × 5 rep ×
B0·B1·B3·B4·B5·B7). 원문(위 "약 270")은 이력 보존을 위해
그대로 두고 이 정정으로 대체하지 않는다. 같은 세션에서
**모델 교체 결정**(Zamba2-2.7B→NemotronH Nano-9B-v2-Base) +
**1차 캠페인 범위 결정**(Claim D+P3까지)이 함께 내려졌다 —
전문은 최상단 배너(2026-09-13) "C" 절.

### D-2. 잠복 결함(미수정, 등재만)

`r2_eval.sbatch`에 `#SBATCH --output/--error` 없음(로그이 **저장소
루트**에 떨어짐 — CLAUDE.md 금지) · `module load`·`HF_HOME`/
`HF_HUB_OFFLINE` 없음(정확성 게이트와 **다른 모델 스냅샷**을
읽을 수 있음) · `TRITON_CACHE_DIR` 없음(array 공유) ·
`campaign.json`에 `model`·`context_length` 없음 · `cuda_graph:
true`는 **읽히지 않는 장식** · eval 러너는 `--decode-log-interval`
을 안 줘서 **per-boot cudagraph 증거가 없다** ·
`trace_loadgen.py:90-91`이 예외를 삼켜 **부분 실패 arm도 채점
가능한 요약**을 냄 · `engine_bench_runner.sh:12` 포트 공식이
array 동시 스케줄 시 충돌 가능.

### D-3. 정본의 인용 drift (engine-porter 보고, doc-steward가 고침)

검토 결과: `controller.py`/`profile.py` 줄 인용의 **최종 교정본
자체는 이미 이 문서·`CLAIM_EVIDENCE_MATRIX.md`·
`EXPERIMENT_ROADMAP.md`의 2026-09-12(4) 追記에 정확히 실려 있다**
(예: `evaluation_due→:196-216`, `stabilize` HOLD 분기
`→:263-274`, overload 트리거 `→:311-326` — 세 문서 모두 동일).
실제로 **미수정 상태였던 것은 그 정정과 별개로 존재하던
"그대로 유효" 서브리스트**(`controller.py:55,65,68,80-94,87`)
하나뿐이다 — 이제 위 A/B 항목(이 문서 상단)과
`CLAIM_EVIDENCE_MATRIX.md`에서 정정했다: `:87`은 **빈 줄**이라
제외, `:65`는 `admission_limited`가 아니라
`upper_bound_itl_ms`(그 줄은 `:68`). 추가로 `dual_worker.py:619`가
`decode_running_batch_size`로 인용된 자리(`workspace/engine-port/
results/s8p_prefill/FINDINGS_PREFILL_2026-07-29.md:126`, 정본
계층 밖의 결과 문서)는 실제로 **`:618`**이다(선행 결함, 이
doc-steward가 직접 대조 확인) — 정본 4문서 어디에도 이 인용이
없어 canon 수정은 불필요하나, 근거 결과 문서 쪽 정정은 사용자
판단 대기(해당 문서는 "이력 보존" 원칙상 이 doc-steward가
독자적으로 손대지 않았다).

GPU 장부: 이 배너 전체 **GPU 0**(B3/OS 미실행, A1/A2/A4는 코드
작업). `longctx_conflict` 15.42 GPU-h·R2 correctness 0.43 GPU-h
장부 둘 다 불변.

## F. 정본 반영

`CONSENSUS.md` rev69→**rev70**(§3 항목200–209 신설[G-B3-1…5·
G-OS-1…5], 정정 2건 반영) — **판단 근거**: 신규 방법론 게이트
10건(#180–189)이 이번 세션 산출이므로 2026-09-12(2)가 확립한
선례("신규 방법론 게이트는 그 자체로 CONSENSUS rev 사유")를
그대로 적용한다. 여기에 더해 이번 회차는 (1) 이전 정본 문장의
**철회**(B3C-3) (2) 근원 판정서의 **산술 오류 정정**을 포함해,
단순 코드 사실 등재보다 무거운 갱신이다. `reports/paper/
{CLAIM_EVIDENCE_MATRIX,EXPERIMENT_ROADMAP,DOCUMENT_STATUS}.md`
갱신, `MEMORY.md`·`memory/{deconfound-measurement-lessons,
slo-aware-scheduling-track}.md` 갱신(항목178–187 신설). **OS·B3는
미실행이므로 어떤 결과도 등재하지 않았다.**

이전: 2026-09-12(4)(doc-steward — **2026-09-12(2)가
"새로 생긴 긴장 2건 — 미해결"로 등재한 항목이 engine-porter
A안 구현으로 둘 다 해소**[사용자 승인, GPU 0, **작업트리
미커밋**]. ①로드맵 `:976` 즉시성 회복 — `_live_underprediction`
술어(ITL p95 > SLO 또는 KV/running 점유 ≥ 85%)를 단일 정의원
헬퍼로 추출해 `evaluation_due`의 독립 disjunct로 승격, 위반이
난 그 iteration에 즉시 재평가(`:975` 정상 케이던스는 별개 조항
그대로 유지). ②overload streak epoch 게이트 신설(①의 필수
동반 수리 — 없으면 긴급 평가가 매 iteration 발화해 연속 2
iteration[≈27–85ms]만에 admission 제한이 걸렸을 것). ③`:979`
분리-disjunct 갱신 — 죽어 있던 점유율 90% 단독 트리거 복원 +
`target≥D108 AND upper_bound>SLO` 결합 유지로 profile-less
hybrid arm 무조건 throttle을 막는 순수 OR 금지. ④`bucket_
changed` 미구현 사실을 코드 docstring에 고정(서빙 경로가
넘기지 않고 `src/multiplex`에 정의도 없음). ⑤감사 가능성 필드
2건 신설(`SplitDecision.off_cadence` + 텔레메트리
`controller_decision.off_cadence`). 검증: 전체 CPU **362 tests
OK**(직전 343, +19), 변이 12종(M1–M12) 전부 실패 확인,
`check_line_citations.py --check --all` = 50 compared **0
violation**(전·후, 이 doc-steward가 독립 재실행해 확인). 새
방법론 게이트 3건(#177–179). **새 사실 2건**(사다리 상승 속도
변화[미측정, 열린 항목으로만 등재] · stale dev tree 프로세스
교훈). 새 성능 판정 0건·Claim D/E 등급 불변(둘 다 미검증)·HE0·
정책 순위·stake #1 전부 불변. `CONSENSUS.md` rev68→**rev69**
(신규 게이트 3건이 사유 — 판단 근거는 아래).

★★**구현 상세(engine-porter, `workspace/engine-port/src/
multiplex/{controller.py,multiplexing_mixin.py}`, 커밋 없음 —
작업트리 수정만)**:

1. `_live_underprediction(snapshot)`(`controller.py:151-166`,
   staticmethod, **단일 정의원**) = `measured_itl_p95_ms >
   itl_slo_ms or kv_occupancy>=0.85 or running_batch_occupancy
   >=0.85`. 호출자 둘: `evaluation_due`(케이던스를 기다리지 않고
   즉시 재평가시키기 위해)와 `stabilize`(그 상향 target을
   고르기 위해) — 사본을 두면 트리거와 동작이 따로 드리프트하는
   결함(게이트#166 계열)이 되므로 정의원을 하나로 강제한다.
2. `_cadence_due(snapshot)`(`controller.py:168-179`) = 순수
   `max(4 iterations, 100 ms)` **AND**(`_dwell_satisfied`와 같은
   합성) — 첫 호출은 `-inf`/`-10**12` sentinel로 무조건 due.
3. `evaluation_due(snapshot, bucket_changed=False)`
   (`controller.py:196-216`) = `bucket_changed or
   _live_underprediction(snapshot) or _cadence_due(snapshot)` —
   **:975**(정상 케이던스)와 **:976**(즉시 upshift)이 산문의
   "두 개의 별개 조항"이라는 것을 그대로 코드 구조로 반영(하나로
   접으면 한쪽이 조용히 죽는다는 것이 아래 게이트 #177의 근거).
   `bucket_changed`는 서빙 경로에서 항상 `False`(아래 4항).
4. **overload streak epoch 게이트**: `_overload_epoch_elapsed
   (snapshot)`(`controller.py:181-194`)가 `_cadence_due`와 같은
   `max(min_epoch_iterations, min_epoch_s)` AND를 별도 마커
   (`last_overload_epoch_s`/`_iteration`, `:277` 부근에서
   `off_cadence` 계산 직후 갱신되지 않고 `stabilize`의 overload
   판정부[`:311-326`]에서만 갱신)로 유지해, 긴급(off-cadence)
   평가가 이 카운터를 부풀리지 못하게 막는다 — **케이던스
   epoch당 최대 1회 증가**. 과부하 해제 시 `overload_streak`은
   즉시 0(불변), `release_admission_limit()`(`controller.py:
   218-232`)이 이제 `last_overload_epoch_s/_iteration`도 함께
   리셋(`:231-232`)해 시계를 재시작한다.
5. `:979` 분리-disjunct(`stabilize`, `controller.py:311-318`) =
   `(target>=emergency_state and upper_bound_itl_ms>itl_slo_ms)
   or kv_occupancy>=0.90 or running_batch_occupancy>=0.90`.
   `target>=emergency_state` 결합은 장식이 아니라 profile
   비호환 fallback(`upper_bound_itl_ms=inf`, `fallback_decode_
   sms=44`)에서 순수 OR이 저점유에서도 무조건 throttle하는 것을
   막는 안전장치 — 코드 주석(`:300-310`)에 이유를 그대로 남겼다.
6. `stabilize`의 bucket docstring(`controller.py:249-262`)이
   `bucket_changed`가 **미구현**임을 명시적으로 기록 — 유일한
   서빙 호출부 `_r2_decide_idx`(`multiplexing_mixin.py:436-440`)
   가 이 인자를 넘기지 않고 `src/multiplex`에 bucket 정의가
   없다. 파라미터를 지우지 않은 이유: API·테스트 호환 + "이 절이
   미구현"이라는 사실 자체를 코드에서 지우지 않기 위해서다.
7. 감사 가능성: `SplitDecision.off_cadence`(신설, `:78`, AUDIT
   ONLY 주석) + 텔레메트리 `controller_decision.off_cadence`
   (`multiplexing_mixin.py:452`, `# :976 fired before the
   cadence` 주석). 기존 필드 의미·스키마 변경 0.

검증 상세: 신규/확장 테스트가 위 6개 이름
(`controller`·`evaluation_due`·`_live_underprediction`·
`_cadence_due`·`_overload_epoch_elapsed`·`overload_streak`)의
**FixedPolicy 비보유**를 고정(`test_controller_defaults.py:
572-602`, `hasattr` 루프) — 이 doc-steward가 파일을 읽어 6개
이름 전부 확인. **FixedPolicy/Claim D 경로는 여전히 불변**
(변이 M8이 이 불변을 어긴 변이를 잡는다). 공용 경로(`_r2_decide_
idx`)의 편집은 텔레메트리 kwarg `off_cadence` 1개 추가뿐이고,
`r2_correctness_check.py`는 `d.get("policy")`처럼 `.get()`으로
개별 필드를 뽑아 쓰므로 B5 POLICY 불변식 검사는 무영향(이
doc-steward가 재확인). 파일 해시(작업트리, 커밋 없음):
`controller.py` → `a4fde4d3bc30b94796fb94d942be0ecc1d587bf1c
b8c2189fc88e5d3106534e1`, `multiplexing_mixin.py` →
`0fe9d570f151698cb4120320bea8bea1111dff052e179085349b97bcbf1
a1697`(이 doc-steward가 `sha256sum`으로 재계산해 확인).
`profile.py`는 **파일 자체 미수정**(481줄 불변).
`multiplexing_mixin.py`는 총 **1879줄 불변**(17개 기존 인용
자리는 그대로).

★★**새 사실 1 — 사다리 상승 속도 변화(미측정, 열린 항목/후속
결정 후보로만 등재)**: dwell은 upshift에 면제(로드맵 `:977`)
이므로, 위반이 iteration마다 지속되면 상태 사다리가 **연속
iteration마다 한 칸씩** 오를 수 있다 — 기본 케이던스 정수에서
`D16→24→34→44→108`이 **4 iteration(≈40ms)**, 이전 판본(위반이
다음 정상 케이던스까지 대기)에서는 **4 epochs(≥400ms)**였다.
칸마다 green-context 드레인 1회다. **성능 영향은 측정된 바
없다**(generic/hybrid GPU 측정 0건, 이 세션도 GPU 0) — 어떤
성능 주장으로도 인용하지 말 것. 열린 항목: "upshift 속도를
epoch당 한 칸으로 제한할 것인가"는 **아직 결정되지 않았다**
(다음 세션/사용자 판단 대기, 아래 "다음 실험 gate" 참조).

★★**새 사실 2 — 프로세스 사실(방법론 교훈, 새 게이트 번호는
매기지 않음 — 이미 CLAUDE.md 부팅 절차로 성립된 규율의 확인
사례)**: 이 세션 시작 시 dev tree가 **stale**이어서
`sync_engine_tree.sh` 실행 전에는 7개 테스트가 실패했다. **343
→362 기준선은 sync 이후에만 재현된다.** 이 doc-steward가 독립
재현: module load + venv 없이 `python -m unittest discover`를
바로 돌리면 **220개만 수집되고 다수 실패**하지만, CLAUDE.md
부팅 절차(`module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2
gcc/15.2.0` → venv 활성화)를 따르면 **362 tests OK**로
재현된다 — CLAUDE.md 부팅 절차가 장식이 아니라 이 세션의
구체 사례로 다시 확인된 것이다.

★★**로드맵 산문 3줄 교체(engine-porter 제공 문안, 문자 그대로
적용) — `reports/paper/EXPERIMENT_ROADMAP.md` "Controller
defaults"**: 기존 `:975`(케이던스)·`:976`(즉시 upshift)·
`:979`(admission) 세 줄을 대체하고, 기존 追記(2026-09-12
불일치 3건 + 긴장 2건)는 역사로 보존하되 "해소됨"으로 상태를
갱신했다. 전문은 `EXPERIMENT_ROADMAP.md` "Controller defaults"
절 追記(2026-09-12(4)) 참조.

★★**깨진 줄 인용 갱신(engine-porter 제공, 이 doc-steward가
독립 재확인)**: `controller.py`: `:129-142 → :196-216`
(`evaluation_due`) · `:127 → :148`(`self.admission_limited =
False`) · `:174-185 → :263-274`(HOLD 반환 블록) · `:210-218 →
:311-326`(overload 판정+streak 증가) · `:219 → :327`
(`admission_limited = self.overload_streak >= 2`) · `:219-220 →
:327-328`(위 줄 + `self.admission_limited = admission_limited`)
· `:244 → :352`(UNSAFE 반환의 `admission_limited=`) · `:259 →
:368`(최종 반환의 `admission_limited=`). 불변: `:68`, `:17`,
`profile.py` 전부(파일 미수정), `multiplexing_mixin.py` 17개
인용(줄 수 1879 유지). 신규 앵커: `_live_underprediction
:151-166` · `_cadence_due :168-179` · `_overload_epoch_elapsed
:181-194` · `off_cadence` 필드 `:78` · 계산 `:277` · release
epoch 리셋 `:231-232` · `stabilize` bucket docstring `:249-262`
· 텔레메트리 필드 `multiplexing_mixin.py:452`. 영향 문서: 이
문서 위 2026-09-12(2) 블록의 "새로 생긴 긴장 2건"·"정본 문서의
깨진 줄 인용 갱신" 항목(아래에 ★해소 포인터 부착) ·
`reports/paper/CLAIM_EVIDENCE_MATRIX.md` "Claim E 착수 전
코드↔로드맵 불일치 3건" 절 · `reports/paper/
EXPERIMENT_ROADMAP.md` "Controller defaults" 절.

★★**신규 방법론 게이트 3건**: **#177**("산문 사양의 두 조항
[케이던스·즉시성]을 한 술어로 합치면 한쪽이 조용히 죽는다" —
2026-09-12(2) 시점 코드는 즉시성[`:976`]을 케이던스[`:975`]
안에서만 계산해 최악 반응 지연이 한 epoch로 늘어났었다;
`evaluation_due`를 두 개의 독립 disjunct로 되돌린 것이 수리다,
`CONSENSUS.md` §3 항목197[신설]) · **#178**("긴급 경로를
추가할 때 그 경로가 다른 카운터[여기선 overload streak]의 시간
단위를 바꾸지 않는지 확인하라" — 긴급 평가가 케이던스 epoch을
우회하면서 "2 epochs 지속"을 "2 iterations"로 은근히 재정의할
뻔했다[≈400ms→≈27-85ms], `CONSENSUS.md` §3 항목198[신설]) ·
**#179**("미구현 절을 산문에 남겨 두면 구현된 것처럼 인용된다"
— `bucket_changed`/"or bucket change"[`:975`] 사례: 파라미터가
API에 존재하고 테스트가 통과한다고 해서 서빙 경로가 그 값을
채운다는 뜻이 아니다, `CONSENSUS.md` §3 항목199[신설]).

★해소(2026-09-12(4)) — 아래 2026-09-12(2) 블록의 "★새로 생긴
긴장 2건 — 미해결로 등재(사용자 결정 대기, 고치지 않음)"과
"정본 문서의 깨진 줄 인용 갱신" 두 항목은 **위 구현으로 전부
해소됐다**(사용자 승인 A안). 원문은 역사로 보존, 뒤집지 않는다
— 상태만 "미해결"→"해소"로 갱신.

`CONSENSUS.md` rev68→**rev69** — **판단 근거**: 이번 변경은
GPU 0·코드 사실 + 규칙 정합(새 성능/분석 결론 아님)이라는 점만
보면 2026-09-12(1차) 선례("코드 사실 등재는 매트릭스·로드맵·이
문서 3개로 충분, CONSENSUS rev 불필요")를 따라야 하지만,
2026-09-12(2)가 이미 확립한 후속 선례("규칙층 감사·구현
검토가 산출한 신규 방법론 게이트는 그 자체로 CONSENSUS rev
사유")를 그대로 적용한다 — 이번 세션이 신규 게이트 3건
(#177–179)을 산출했으므로 그 선례를 따라 rev를 올린다(코드
사실 자체[①–⑤]는 그 선례에서도 단독으로는 rev 사유가 아니다).
정본 반영: `CONSENSUS.md` §3 항목197–199(신설), `reports/
paper/{CLAIM_EVIDENCE_MATRIX,EXPERIMENT_ROADMAP}.md` "Claim
E 착수 전 코드↔로드맵 불일치 3건"·"Controller defaults" 절
갱신, `MEMORY.md` 포인터 갱신·`memory/deconfound-measurement-
lessons.md` 항목175–177 신설·`memory/slo-aware-scheduling-
track.md` 새 절 `## 2026-09-12(4)` 신설. 상세는 이 배너 자체
(원자료는 작업트리 코드·테스트, 커밋 없음 — GPU 캠페인 아니므로
`results/<campaign>/` 산출물 없음).

GPU 장부: 이 트랙 변경 없음(GPU 0). `longctx_conflict` 15.42
GPU-h·R2 correctness 0.43 GPU-h 장부 둘 다 불변.

이전: 2026-09-12(3)(doc-steward — **X1 결과 등재**: job
907456(gpu42, 0.154 GPU-h) 완주 + claims-auditor 결과 감사
**`CONFIRMED(scoped)`**[등록 예보 F1–F4 전부 충족, `VERDICT
PASS`·교차-잡 불일치 8/96 독립 재현, 귀속 성립]. 단 메인 세션이
쓰려던 해석 문장 1건은 **`REFUTED`**(D44 운영점 측정층 민감도
미보정). 새 방법론 게이트 5건(#172–176, #176=긍정 사례) 신설.
새 성능 판정 0건·Claim D 등급 불변(미검증)·HE0·정책 순위·
stake #1 전부 불변. `CONSENSUS.md` rev67→**rev68**.

**사실 기록** — job 907456, node gpu42, GPU UUID
`GPU-7a590213`, 2026-09-12 19:11:22–19:20:36 KST, 9m14s =
**0.154 GPU-h**, commit `19b9d8d`(`src/multiplex`는 `38c1aca`
복원, `runtime_source_manifest.sha256` 17/17 907100과 바이트
동일), `--triton-attention-num-kv-splits 2`, checker sha
`ec355e17…` 무수정, 사전등록 sha `54e76c9a…`. claims-auditor
결과 감사 원문 전사 = `workspace/engine-port/results/
r2_correctness/audit_x1_2026-09-12/VERDICT.md`(357행, sha256
`f1d8e64e91f2a251fdbbd76a2947bb853709efcbdff570219ced5fda53
aeedc7`). 총괄 **`CONFIRMED(scoped)`** — 등록 예보 F1–F4 전부
충족, PASS와 교차-잡 8/96 불일치는 정확히 재현되고 공허성
없음(TD 경로·비복사·작업량 일치·8,448 토큰 비교 불일치 0·퇴화
0/56), 귀속 성립(귀무대조 2건: 등록 `907032→907100` S 0/32 +
감사자가 추가한 비등록 `907032→907456` S 2/32 전부 S05@25),
엔진 소스 manifest 17/17 바이트 동일.

★★**정본 등재 스코프 문장(판정서 §8, 문자 그대로 승계)**:

> "X1 민감도 양성대조: job 907456(2026-09-12, 0.154 GPU-h,
> node gpu42, commit `19b9d8d` — `src/multiplex`는
> `38c1aca`로 복원되어 `runtime_source_manifest.sha256`
> 17항목이 job 907100과 바이트 동일, 판정 규칙
> `r2_correctness_check.py` sha `ec355e17…` 무수정, 사전등록
> `PREREG_X1_SENSITIVITY_2026-09-12_rev2.md` sha
> `54e76c9a…`). 네 boot 전부에 `--triton-attention-num-kv-
> splits 2`를 준 **비운영 수치 구성**에서도 `VERDICT PASS`
> (S 6쌍 + O 6쌍 = 8,448 토큰 비교 불일치 0; 4 boot 전부
> booted·crash-free·cudagraph-ON[decode 10,384줄 전부 True,
> false 0]·path·policy·overlap 100% idx 4·green realized
> 64/44·O1–O4 8/8·퇴화 0/56). 등록 예보 F1–F4 전부 충족,
> claims-auditor 결과 감사 **`CONFIRMED(scoped)`**. 교차-잡
> 진단(907100→X1, 분모 96 온전·제외 0·index-0 불일치 0)에서
> **2단위 S05·O06이 4 boot 전부 동일하게 뒤집혔다**(first
> divergence 25·11, 꼬리 38/64·35/48 토큰 계속 상이, 토큰
> 값도 4 boot 동일). 귀무대조 2건이 귀속을 받친다:
> `907032→907100`(노드·GPU·커밋 상이, 무교란) **S 0/32**,
> `907032→907456`(노드명 동일·물리 GPU 상이, 교란) **S
> 2/32 전부 S05@25**. 이로써 **측정층 민감도의 하한**이
> 시연됐고, 범위는 세 가지로 한정된다: (i) 교란은 decode
> attention에만 걸리고 **prefill(extend)은 비트 동일**,
> (ii) 뒤집힌 두 단위의 decode는 **비분할 stream_index
> 5(0/108)**에서 돌았고 **D44(idx 4)에서 돈 비교 대상
> 계산은 여전히 probe prefill뿐이며 미교란**, (iii) D44에서
> 돈 유일한 **교란된** decode(O 층 background 행, bs=1 구간
> 112/159 step)는 **0/32로 뒤집히지 않았다**. 따라서 **X1은
> D44 운영점의 측정층 민감도를 보정하지 않는다.** 교란의
> 도달 범위는 단위 기준 24/24지만 **decode 스텝 기준
> 1,142/1,384(82.5%)**이고, 직전 회차가 가장 민감하다고
> 지목한 비복사형 단기 6단위는 부분 커버리지(S00 5/63·S03
> 6/63·S02 10/63·S01 24/63·S04 36/63·S13 55/63)다. C 층
> (진단 전용): C01의 arm-분리 소멸(4 boot 전부 = 907100 TD
> 값)은 §1.7 기전과 **정합이며 확증 아님**; 그러나 arm-분리
> 0의 2/3는 C10·C19에서 **L2가 옛 TD 값으로 이탈**해
> `{L1}`/`{L2,TD1,TD2}`가 된 결과이고 L-L 불일치는 4→5로
> 증가했다. 불일치 단위 수 8→6이지만 **등가류 초과수
> Σ(classes−1)는 10→10 불변**(C00·C16은 3류→4류). **성능
> 판정 0건, Claim D 등급 불변(미검증), 선결 #1·#2·#3·#4a·
> #4b·#5 상태 불변, R2C-1…16·P-1…7·HE0·정책 순위·stake #1
> 불변. X1은 어떤 게이트도 닫지 않는다."**

★★**§9 인용 금지 추가분(X1C-10…14, 판정서 원문 그대로
승계)**:

- **X1C-10** "X1이 **D44(PD-mux 운영 분할)에서의** 측정층
  민감도를 시연했다 / 보정했다". 허용형: "비분할
  `stream_index 5` decode에서 시연됐고, D44에서 돈 교란된
  decode(O 층 background 행)는 **0/32로 뒤집히지 않았다**;
  D44 64-SM 측의 probe prefill은 구성상 미교란이다."
- **X1C-11** "X1의 교란이 비교 단위 **전 구간**에 걸렸다" /
  "24/24는 교란 **커버리지**다". (24/24는 '어떤 step에서라도
  다르다'는 단위 지시함수다. 스텝 커버리지는 **1,142/1,384
  = 82.5%**이고 S00은 **5/63**, 즉 마지막 5토큰만 교란됐다.)
- **X1C-12** "C 층 불일치가 **줄었다**" / "8→6 감소가
  교란의 효과를 보여 준다". (**Σ(등가류−1)은 10→10 불변**,
  C00·C16은 3류→**4류**로 더 파편화, L-L은 **4→5로 증가**.
  '감소'라는 서술 자체를 쓰지 않는다.)
- **X1C-13** "arm-분리 = 0이 §1.7 기전을 **지지**한다".
  (arm-분리 0의 2/3인 C10·C19는 **L2가 옛 TD 값으로
  이탈**해 분할선이 `{L1}` vs `{L2,TD1,TD2}`로 바뀐 결과다.
  TD가 legacy로 수렴한 것이 아니다. 지지되는 범위는 **C01
  한 건**이고 boot 2/arm에서 P≈0.20이다.)
- **X1C-14** "**8/96 = 8.3%**가 측정층 민감도(검출률)다"
  또는 그 수에서 파생한 어떤 율. (독립 비교 수는 **24**이고
  4 boot-쌍은 같은 결정론적 비교의 반복이다 — first
  divergence와 토큰 값이 4/4 동일. 단위별 교란 커버리지
  8%–100%와 argmax 마진 차이가 섞여 있어 어떤 율도
  추정량이 아니다.)

★★**§10 필수 병기(X1P-1′·2′·3′·5′·6′·8·9 + P-1 추가분,
판정서 원문 그대로 승계)**:

- **X1P-1′** "X1의 수치 구성은 `--triton-attention-num-kv-
  splits **2**`(907100은 `8`, 4 boot 전부 server args에
  기록)이며 **운영점이 아니다**. 운영 기본은 8 + 커널
  휴리스틱이다. X1의 within-job PASS는 운영 수치 구성의
  correctness를 확장 인증하지 않는다."
- **X1P-2′** "X1의 교란은 decode attention에만 걸린다
  (`forward_extend`에 `num_kv_splits` 참조 0건; 실측
  index-0 불일치 0/96). 따라서 **probe prefill의 민감도는
  X1 이후에도 미보정**이다. 나아가 뒤집힌 두 단위의 decode는
  `stream_index 5`에서 돌았고, D44에서 돈 교란된 decode(bg
  행)는 **0/32**다 — **X1은 D44에서의 민감도를 전혀
  시연하지 않았다.**"
- **X1P-3′** "X1의 교란은 **단위 기준 24/24에 도달**하지만
  (무교란 0), 그 지표는 '어떤 decode step에서라도
  `kv_len_per_split`이 다르다'는 지시함수다. **스텝 기준
  커버리지는 1,142/1,384(82.5%)**이고 단위별로 S00 5/63·
  S03 6/63·S02 10/63·S01 24/63·S04 36/63·S13 55/63, 나머지
  18단위 100%다. 직전 회차 §1.6이 '뒤집힐 확률이 가장
  높다'고 분류한 비복사형 단기 6단위가 **전부 부분
  커버리지**다. (4-gram 복사형 분류값은 직전 회차 §1.6에서
  **인용**한 것이며, `gen_*.json`에 prompt token id가 없어
  output↔prompt 4-gram은 이번에 독립 재계산하지 못했다 —
  내가 계산한 output 자기-4-gram 반복률은 다른 지표다.)"
- **X1P-5′** "교차-잡 비교는 **진단 전용**이다. 귀무대조는
  두 건이다: 등록된 `907032→907100` S `compared=32 /
  mismatch=0`(gpu42→gpu38, GPU UUID 상이, `02918e8`→
  `38c1aca`, 무교란)과 **비등록 추가** `907032→907456` S
  `compared=32 / mismatch=2`(전부 S05@25). O 층에는
  귀무대조가 없다(907032에 O 층 부재). 불일치는
  `first_divergence ≥ 1`일 때만 교란에 귀속하고, 실제로
  index-0 불일치는 0/96이다."
- **X1P-6′** "C 층 수치는 어떤 게이트·기전·정책 판정에도
  쓰지 않는다(boot 2/arm, 균등 귀무 P≈0.20). 불일치 단위 수
  8→6은 **총량 감소가 아니다** — Σ(등가류−1) 10→10 불변,
  C00·C16 3류→4류, L-L 4→5."
- **X1P-8**(신설) "907100의 provenance는 env를 기록하지
  않는다(env 기록은 X1 sbatch가 신설). 따라서 baseline에서
  `SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS`가 unset이었다는
  것은 **아티팩트로 검증 불가**다. set이었다면 X1은 두 가지를
  동시에 바꾼 것이 된다(static fill→휴리스틱 ∧ cap 8→2,
  confound #10). 이 불확실성은 뒤집힘의 존재 귀속을 바꾸지
  않으나 사전등록 교란표의 전제를 바꿀 수 있다."
- **X1P-9**(신설) "교란의 **실현** 증거(args 덤프 외 2번째
  채널): cudagraph capture `mem usage`가 5/5 boot에서
  **0.73 GB → 0.65 GB**, `avail mem` 13.32 → 13.40 GB로
  바뀌었다. 부호·차수는 `attn_logits`/`attn_lse`/cudagraph
  버퍼가 `max_kv_splits` 8→2로 줄어든 것과 맞다. 닫힌 형태
  추정(≈39 MB)과 관측(≈80 MB)의 배수 2는 미해소이므로
  **정성적 실현 확인으로만** 인용한다."
- **P-1 추가**: 스코프 튜플에 `triton_attention_num_kv_
  splits=2`, node gpu42(`GPU-7a590213`), commit `19b9d8d`
  (`src/multiplex`는 `38c1aca` 복원, manifest 17/17 동일)를
  넣는다.

★★신규 방법론 게이트 5건 — **#172**(G-X1-1, 교훈 9 계열의
3번째 층. 양성대조의 도달 범위는 "비교 단위 지시함수"가 아니라
"교란이 실제로 걸린 계산 스텝/토큰의 분포"로 등록하라 — 자기
적용: 이 결함의 출처는 claims-auditor 1단 판정서 자신의 D1이다,
`CONSENSUS.md` §3 항목192[신설]) · **#173**(G-X1-2, 게이트#1·
#114·#166 확장, 양성대조가 게이트가 인증하려는 운영점 위에
떨어졌는지 확인하고 떨어지지 않았으면 "그 운영점에서는
미보정"을 등록 문안에 넣어라, §3 항목193[신설]) · **#174**
(G-X1-3, 등가류 "불일치 단위 수"는 불일치 총량이 아니다 —
Σ(등가류−1) 또는 분할 전체를 병기하라, §3 항목194[신설]) ·
**#175**(G-X1-4, 게이트#80 인접, provenance 보강을 교란
arm에만 넣으면 비대칭 기록이 되어 baseline 전제를 사후 검증
불가로 만든다, §3 항목195[신설]) · **#176**(G-X1-5, **긍정
사례**, 게이트#1의 처치 측 적용 — 교란 노브 자신에 대해
"target이 아니라 realized"를 요구하라, X1은 (a) server args
덤프+`SAME_ACROSS_BOOTS` 검사, (b) cudagraph capture 메모리
변화 두 채널로 충족했다, §3 항목196[신설]).

★**engine-porter 이관 항목(게이트 아님, 열린 항목으로 등재)**:
(i) `decode_step_count`가 전 boot·전 스냅샷 0(죽은 필드) — 이를
작업량 지표로 쓰면 0을 얻는다. (ii) `worker_overlap_ratio`가 TD
boot에서도 전부 0.0이고, 실제 값은 이름이 비슷한
`host_worker_overlap_ratio`(X1 TD 최대 0.0024/0.0017)에만 있음
— 게이트 #166(G-2) 계열 재발 위험.

★**다음 실험(판정서 §12, 전부 미승인·사용자 판단 대기, 아래
"다음 실험 gate" 항목4 追記에도 등재)**:

1. **C-tier 귀무대조**(`R2C_ORDER="L L L L"`, ≈0.154 GPU-h) —
   무교란·동일 arm에서 32단위 등가류 분할·Σ(classes−1)·
   pairwise 불일치를 측정한다. F3을 해석 가능하게 만드는 유일한
   값싼 길이며 arm-분리 판정에 분모를 준다(현재 n=2/arm,
   P≈0.20). ★**추적(2026-09-12(5))**: 이 항목이 아래 "B3"
   사전등록(`L L L L`+`TD TD TD TD` 2 job)으로 구체화·제출됐고
   규칙층 감사 `NO-GO`를 받았다(GPU 0, 미실행) — 상세 아래
   2026-09-12(5) 배너.
2. `R2C_ORDER="TD TD TD TD"` 1 job(+0.154 GPU-h) ⇒ 합쳐
   n=4/arm(프로젝트 게이트 3 충족), paired 비교. ★**철회
   (2026-09-12(5), B3 판정서 §8, 아래 2026-09-12(5) 배너 참조)**:
   괄호 문장 "프로젝트 게이트 3(n≥4) 충족"은 **거짓**이다 — 한
   job의 4 boot은 같은 노드·같은 물리 GPU·같은 warmup·같은
   `TRITON_CACHE_DIR`을 공유하는 순차 실행이라 독립 런이 아니고,
   게이트 3은 *정책 비교 베이스라인 분산*을 위한 **독립 런**
   n≥4를 요구한다. 인용 금지 **B3C-3**. 이 항목은 이후
   `L L L L`+`TD TD TD TD` 2 job(0.31 GPU-h) 사전등록으로
   구체화됐고 **`NO-GO`**(死因 N2, GPU 0·미실행) 판정을 받았다 —
   상세 아래 2026-09-12(5) 배너 "B. B3" 절.
3. **D44 resident decode 동치를 보려면 새 층 O′ + 판정 규칙 v3
   사전등록 필수**(≈0.16 GPU-h) — 엔진은 prefill이 없으면 idx
   4를 떠나므로 probe가 decode하는 동안에도 제3의 장문 prefill을
   계속 투입해야 한다. 이것은 X1 후속이 아니라 **선결 #2를
   넓히는 실험**이고 1·2를 먼저 하지 않으면 또 n=2로 끝난다.
4. **GPU 0으로 지금 가능**: preflight에 스텝 커버리지 출력
   추가·비교기에 Σ(classes−1) 출력 추가·죽은 텔레메트리 필드
   2건 처리(engine-porter)·사전등록 템플릿에 "교란 도달
   위치=인증하려는 운영점인가" 항목 추가.
5. **권고하지 않음**: cap을 더 낮추거나 다른 수치 구성으로 X1
   반복(교란이 떨어지는 위치가 안 바뀌므로 같은 한계의 결과가
   또 나온다).

GPU 장부: 이 트랙 0.28 → **0.43 GPU-h**(longctx_conflict 15.42
GPU-h 장부와 별개). Claim D 등급 불변(미검증). 선결 #1–#5 전부
상태 불변(#5 cudagraph-ON은 재확인이나 등급 변화 없음, #2는
오히려 X1의 within-job 시험이 더 약함). R2C-1…16·P-1…7·HE0·
정책 순위·stake #1 전부 불변. **X1은 어떤 게이트도 닫지
않는다.**

`CONSENSUS.md` rev67→**rev68** — **판단 근거**: 직전
2026-09-12(2)(A/B/C)는 코드 사실 등재·X1 규칙층(제출 전) 감사
였으나, 이번은 **GPU 캠페인 실행 결과 + claims-auditor 결과
감사 판정**(F1–F4 충족·`CONFIRMED(scoped)`·메인 세션 해석 문장
1건 `REFUTED`)이라 이전 결과-등재 세션들의 관례(Q-A 캠페인
rev64, R2 true-dual correctness 결과 rev65/66)를 따라 rev를
올린다. 정본 반영: `CONSENSUS.md` §3 항목192–196(신설)·§5-8(a)
追記(rev68), `reports/paper/CLAIM_EVIDENCE_MATRIX.md` Claim D
행·"주장 제한" 갱신, `EXPERIMENT_ROADMAP.md` "P1/P2" 절 갱신,
`MEMORY.md` 포인터 갱신·`memory/deconfound-measurement-
lessons.md` 항목170–174 신설·`memory/slo-aware-scheduling-
track.md` 새 절 `## 2026-09-12(3)` 신설(결과 등재). 상세
`workspace/engine-port/results/r2_correctness/{job_907456/,
audit_x1_2026-09-12/VERDICT.md}`.

이전: 2026-09-12(2)(doc-steward — **A) 게이트 #157(도구
파일 버전관리 이관) 완료**[커밋 `4027257`, GPU 0] · **B) Claim E
선결 코드↔로드맵 불일치 3건 해소**[커밋 `3f188bb`, GPU 0,
engine-porter — 사용자 결정: 3건 모두 "로드맵이 정본"] · **C) X1
사전등록 rev1→rev2 규칙층 감사 `GO-with-caveats` 등재**[커밋
`b3ef62d`·`19b9d8d` — **X1 job 실행 중, 결과는 등재하지
않는다**]. 새 성능 판정 0건·arm 순위 0건·Claim D/E 등급 무변경
(둘 다 미검증)·HE0·정책 순위·stake #1 전부 불변.

**A) 게이트 #157 완료** — outer 워크스페이스
`/scratch/ehmoon/whlee`가 git 저장소가 아니라는 사실(`.git`이
빈 디렉터리)이 게이트 #157의 미해결 부분이었다(아래 "방법론
게이트" #157 참조). `.claude/agents/*.md`(7)·`.claude/skills/*/
SKILL.md`(3)·루트 `CLAUDE.md`, 총 **추적 사본 11개**를 inner
repo `tools/claude/`로 이관해 버전관리 안에 넣었다. 동기화·검증:
`tools/claude/sync_claude_tools.sh`(`--check` 기본·`--import`
live→추적·`--install` 추적→live·`--manifest`), 해시 목록
`tools/claude/claude_tools.manifest.sha256`, 변이 테스트
`tools/claude/test_sync_claude_tools.sh` **10케이스 전부 통과**
(수정·삭제·신규추가·추적사본삭제 검출 + `--install`/`--import`
복구, 해시뿐 아니라 **파일 집합**도 비교 — 교훈89 대응). `--import`
직후 `--check`는 항등식이라 정보 0이라는 한계를 README에 명시.
심볼릭 링크 방식은 **기각**(에이전트·스킬 디스커버리가 링크를
따라가는지 이 세션에서 검증 불가 ⇒ 실패 시 다음 세션 에이전트 7개
소실 위험). **사실 정정**: `.claude/agents/git-committer.md`의
"outer는 별도 repo, 두 repo는 별개 커밋" 서술을 "저장소는 하나
(inner)뿐, outer는 git 저장소가 아님 + 툴 파일 변경 시 `--import`
후 `tools/claude/` 커밋" 절차로 교체, 루트 `CLAUDE.md`에도 같은
취지 1블록 추가. 앵커 해시: `claims-auditor.md` =
`e7c2dedfe53527a84a0990c50607135d9ba4200d85a151f919aa7f81e7ed93b8`
(**변경 없음**)·`git-committer.md` =
`68797f68781f1c21900045d0b57b3ea9c1995f4c5052032836fa2f8d76a2ada1`
(이번 정정으로 변경)·`workspace/CLAUDE.workspace.md`(=루트
`CLAUDE.md`) =
`478c83bc2604d743f8eab811236995fc796ca98552a5e4c972b0b9c861f4a076`.
앞으로는 개별 해시를 산문에 붙여 넣는 대신 **manifest 파일 +
커밋을 인용**한다(아래 "방법론 게이트" #157 갱신 참조).

**B) Claim E 선결 3건 해소**(커밋 `3f188bb`) — 2026-09-12(1차)가
등재한 "불일치 3건(수정 없이 표시만)"을 사용자 결정("3건 모두
로드맵이 정본")에 따라 engine-porter가 코드로 해소했다: ①
admission 제한 지속을 컨트롤러 상태로 보존하고 HOLD가 승계
(`controller.py:127,174-185,219-220`) + `release_admission_
limit()` 신설(`multiplexing_mixin.py:1809-1817`, duck-typed
⇒ FixedPolicy는 no-op) — stale-True 게이트 테스트 불변 통과,
"영구 True 퇴화" 변이도 검출 ② `evaluation_due()`를 로드맵
`max(4 iterations, 100 ms)` 의미(`bucket_changed OR (시간 AND
iteration)`)로 수정(`controller.py:129-142`) ③ profile 호환
검사에서 `engine_commit`(repo HEAD)을 엄격 일치 집합에서 제외
(provenance 기록용으로만 남김)하고, **실제 임포트된 엔진 모듈
8개의 내용 해시** `engine_source_hash`로 교체(`profile.py:44,
113-160,367-481`) — 하위호환 **fail-closed**(해시 없는 프로파일은
명시 reason과 함께 fallback), `profile_cli`에 `engine-source-hash`
서브커맨드 + 자동 채우기 거부(허위 provenance 방지). 저장소에
기존 프로파일 JSON 0건 ⇒ 재생성 대상 없음. 선택 근거(manifest
아닌 임포트된 바이트를 씀): manifest는 설치 시점의 경로 기반
주장이라 array task 공유 dev tree·재sync·다른
`SGLANG_ENGINE_DEV`에서 "같은 manifest인데 로드된 바이트는
다름"이 가능 — `engine_commit`과 같은 대리물 실패 모드. ★**추기
(2026-09-14, doc-steward — engine-porter 등록성 검사)**: 위 ③이
해소한 원문 발견(아래 2026-09-12 "이전:" 절 ③)의 하위 인용
`scripts/r2_eval/r2_eval.sbatch:26`(`engine_commit` export
지점)도 이후 `:69-70`으로 이동했다(커밋 `09a8075`) —
`r2_eval.sbatch`가 이 문서의 `check_line_citations.py`
`__scope__`(controller.py/profile.py/workloads.py 패턴만) 밖이라
이 해소 당시 갱신되지 않았다(신규 게이트 #238, 아래 "방법론
게이트" 참조). 검증:
신규 테스트 38개(`test_controller_defaults.py` 27 +
`test_r2_admission_persistence.py` 11, 후자는 실제
`event_loop_pdmux`를 fake로 구동) + 수리를 되돌린 변이 M1–M6 +
mixin 2종 전부 실패 확인. **전체 343 tests OK**(직전 305).
`check_line_citations.py --check --all` = 50 compared,
**0 violation**(전·후). **FixedPolicy/Claim D 경로 불변**
(테스트로 고정) — `controller` 속성 없음 ⇒ 해제 훅 no-op,
`evaluation_due`/`stabilize` 미호출, `RuntimeEnvironment`·profile
미사용. R2 correctness 하네스의 `B5_POLICY` 불변식도 그대로
성립.

★**새로 생긴 긴장 2건 — 미해결로 등재(사용자 결정 대기, 고치지
않음)**: (1) 로드맵 `:976`의 "ITL 위반 또는 85% KV/batch 점유율
즉시 upshift"가 결정 ②로 인해 **덜 즉각적**이 됐다 — upshift
분기(`controller.py:189-204`)는 평가 iteration에만 돌고, 평가
주기가 이제 `max(100ms, 4 iter)`(둘 다 충족)라 최악 반응 지연이
이전(먼저 오는 것)보다 **길어진다**. bucket 변화가 안 걸리면
"immediate"는 성립하지 않는다. ★이것은 결정 ②가 **만든** 긴장이고,
이 패치에서 Claim E 거동에 실제로 영향을 줄 수 있는 유일한
항목이다. (2) 로드맵 `:979`의 트리거 합성 — 산문은 "D108 risk
**또는** occupancy 90%"인데 코드는 `target>=108 AND
(upper_bound>SLO or kv>=0.90 or running>=0.90)`
(`controller.py:210-218`) — occupancy 90%만으로는 제한이 걸리지
않는다. 2026-09-12(1차) 등재는 *지속 기간*만 문제 삼았으므로
접속사는 그대로 뒀다. 둘 다 어느 쪽(코드/로드맵)이 옳은지는
판정하지 않는다.

★**정본 문서의 깨진 줄 인용 갱신**(engine-porter 제공, 반드시
반영): `controller.py:124-130 → :129-142` · `:148-149 → :174`
(+반환 블록 `:180-185`) · `:174-183 → :210-219` · `:183 → :219`
· `:207 → :244` · `:222 → :259` · `profile.py:98-114 → :113-160`
· **`profile.py:102`("engine_commit" 엔트리)는 더 이상 존재하지
않음** · `profile.py:268-279 → :314-325`. 그대로 유효:
`controller.py:55,65,68,80-94,87` · `multiplexing_mixin.py:
430-441,1066-1080,1758`. ★**정정(2026-09-12(5), doc-steward,
engine-porter D-3 보고 반영 — 이 문단 전체가 2026-09-12(4) 재작성
이후 STALE임을 알리는 아래 (2)블록 주석이 있음에도, "그대로
유효" 서브리스트 자체엔 개별 지적이 없어 오독 위험이 남아
있었다)**: `controller.py:87`은 현재 **빈 줄**(`MultiplexingPolicy`
Protocol과 `FixedPolicy` class 정의 사이의 구조적 공백)이라
"유효"한 인용 대상이 아니다 — 목록에서 제외. `:65`는
`SplitDecision.upper_bound_itl_ms` 필드이지 `admission_limited`가
아니다(그 필드는 `:68`, 이 구분의 출처는
`workspace/engine-port/results/switch_cost/audit_switch_cost_
2026-08-22/VERDICT.md:27`). 수정된 "그대로 유효" 집합:
`controller.py:55`(`decode_idle_ratio`)·`:65`
(`upper_bound_itl_ms`)·`:68`(`admission_limited`)·`:78`
(`off_cadence`, 2026-09-12(4) 신설이나 구조상 이 목록과 같은
성격)·`:80-94`(`FixedPolicy`) — **`:87` 제외**. 영향 문서: 이
문서 위 A/B 항목 ·
`reports/paper/CLAIM_EVIDENCE_MATRIX.md` Claim E 행·"주장 제한" ·
`reports/paper/CLAIM_EVIDENCE_MATRIX.md` Claim E 행·"주장 제한" ·
`EXPERIMENT_ROADMAP.md` "Controller defaults" · `reports/
r2_decoupling_review_2026-07-24.md:84` · `workspace/engine-port/
results/s8_frontier/DESIGN.md:3460,3466`(engine-porter가 코드
사실을 확인, doc-steward는 정본 인용만 갱신 — DESIGN.md 파일
자체는 이 doc-steward 갱신 범위 밖).

★★**해소(2026-09-12(4), doc-steward, GPU 0, engine-porter A안
구현, 작업트리 미커밋)**: 위 "새로 생긴 긴장 2건"과 "정본 문서의
깨진 줄 인용 갱신" 두 항목은 **둘 다 해소됐다**. (1)은
`_live_underprediction`을 `evaluation_due`의 독립 disjunct로
승격해 즉시성을 되돌리고 overload streak epoch 게이트를 동반
수리로 신설해 해결했다. (2)는 점유율 90% 단독 트리거를
복원했으나 `target≥D108 AND upper_bound>SLO` 결합은 profile-less
hybrid arm의 무조건 throttle을 막는 안전장치로 **의도적으로
유지**했다(순수 OR 금지) — "코드/로드맵 중 옳은 쪽" 판정이
아니라 둘 다 로드맵 산문으로 흡수했다. 줄 인용은 이번 구현으로
다시 이동했다(위 2026-09-12(3) 배너보다 더 위, 이 문서 최상단
2026-09-12(4) 블록의 "깨진 줄 인용 갱신" 참조 — 이 (2)블록의
인용은 **더 이상 유효하지 않다**, 역사 보존용으로만 남긴다).
원문은 뒤집지 않는다(역사 보존). 상세는 최상단
2026-09-12(4) 블록.

**C) X1 규칙층만 등재(결과 없음 — job 실행 중)**(커밋 `b3ef62d`·
`19b9d8d`) — 사전등록 rev1은 **제출 전 감사에서 반증**돼
**SUPERSEDED**(배너 부착, 보존). 死因: "교란됨"을 행별 split
수로 판정했으나 커널이 `MIN_BLOCK_KV=32`로 한 번 더 양자화해
(`decode_attention.py:35,98,553`) split 수가 달라도
`kv_len_per_split`이 같으면 **비트 동일** ⇒ rev1의 "강제 6 ⇒
24/24 교란"은 거짓(실효 **18/24**), 무교란 6개가 하필 4-gram
≤0.04 비복사형이었다. 판정서 초안(env var만 ⇒ cap 8)은 **2/24**로
더 나빴다.

유효 판본 **rev2**: `--triton-attention-num-kv-splits 2`
**CLI 단일 노브**(env var 미사용 ⇒ 관측 가능·`SAME_ACROSS_BOOTS`
자동 검사·confound "변수 동시 변경" 제거), 실효 **24/24 교란**.
통제 강화 1건 추가: 엔진 소스를 `38c1aca`(907100이 돌린 바이트,
8/8 해시 확인)로 되돌린 뒤 제출하고 job의 manifest로 사후
검증한다.

claims-auditor 감사 판정 **`GO-with-caveats`**(死因 0, 반전 0/7
표면) + 차단 조건 D1–D4·D6–D9 반영 완료. 판정서 **원문 전사** =
`workspace/engine-port/results/r2_correctness/x1_prereg/
VERDICT_x1_rules_2026-09-12.md`(228행).

★★**필수 병기 X1P-1…7 / 인용 금지 X1C-1…9를 문자 그대로
등재**(판정서 §7, 전문은 판정서에서 확인, 요지):

- **X1P-1** X1의 수치 구성(`triton_attention_num_kv_splits=6`)은
  **운영점이 아니다**(운영 기본 = 8 + 커널 휴리스틱). within-job
  PASS는 운영 수치 구성의 correctness를 확장 인증하지 않는다.
- **X1P-2** 교란은 **decode attention에만** 걸린다. prefill
  (extend) 경로는 비트 동일이므로(`forward_extend`에 `num_kv_
  splits` 참조 0건), 판정서 §2.3이 "비교 대상 중 D44 64-SM에서
  돈 유일한 계산"으로 지목한 **probe prefill의 민감도는 X1
  이후에도 미보정**이다.
- **X1P-3** 교란은 비교 단위 24개 중 **18개(실효 기준, `MIN_
  BLOCK_KV=32` 양자화 반영)**에만 도달한다. S00·S01·S02·S03·
  S04·S13 6개(= 96 unit-pair 중 24개)는 구성상 비트 동일이어서
  불일치를 만들 수 없다. 교란된 단위는 전부 4-gram≥0.98 복사형,
  ≤0.04 비복사형 7개 중 6개가 무교란.
- **X1P-4** X1의 within-job 동치 시험은 **907100보다 약하다**.
  강제 split은 decode attention을 행 단위 배치 불변으로 만들고
  O1의 전제를 자동 성립시켜, arm-특이 타이밍이 토큰을 바꿀 수
  있는 경로 하나를 제거한 뒤 동치를 확인한다.
- **X1P-5** 교차-잡 비교는 **진단 전용**이다. 귀무대조는
  `907032→907100` S 층 `compared=32/mismatch=0`(노드·커밋 상이,
  교란 없음)이며, **O 층에는 귀무대조가 없다**. 불일치는
  `first_divergence≥1`일 때만 교란에 귀속한다.
- **X1P-6** C 층 수치는 등가류 서술뿐이고 어떤 게이트·기전·정책
  판정에도 쓰지 않는다. boot 2개/arm에서 "arm-분리 소멸"은
  균등 귀무로도 P≈0.20이다. 강제 split은 decode attention의
  배치 의존성을 제거하므로 **C 불일치 총수 감소는 교란의 예상된
  부작용**이다.
- **X1P-7** X1은 성능을 측정하지 않는다. 새 성능 판정 0건. Claim
  D 등급 불변(미검증), 선결 #1·#2·#3·#4a·#4b·#5 상태 불변(#2는
  여전히 S/O 프로토콜 부분 해소). R2C-1…16·P-1…7·HE0·정책
  순위·stake #1 불변.

인용 금지 **X1C-1**"X1이 correctness를 더 넓게 인증했다"/"두
수치 구성에서 확인됐으므로 TD≡legacy가 강해졌다" · **X1C-2**
"비교기(측정층)의 민감도가 보정됐다"(허용형: "불일치가 난
층·단위에서 decode attention 수치 교란에 대한 민감도의 하한이
시연됐다") · **X1C-3** "X1 0 mismatch는 TD와 legacy가 정말
같다는 추가 증거"(C 층 8/32 불일치가 이미 이 엔진·프롬프트
계열에서 ULP급 섭동이 argmax를 뒤집을 수 있음을 보여 준다 ⇒
0은 "이 복사형 단위들의 마진을 넘지 못했다"로 읽는다) ·
**X1C-4** "24/24 비교 단위를 교란했다"/"무교란 단위는 없다" ·
**X1C-5** "`fill_` 값 하나만 바꿨다"(`MAX_KV_SPLITS` constexpr·
grid z·버퍼 shape가 함께 바뀐다) · **X1C-6** "C01 arm-분리
소멸이 §1.7 기전을 확증한다"/"X1이 C 층 불일치 원인을
규명했다" · **X1C-7** "C 층 결과가 동시 부하 동치(R2C-2)를
완화한다" · **X1C-8** "X1이 Claim D 선결을 닫았다/진전시켰다"
또는 "r2_eval 설정·D16/24/34·generic/hybrid·다른 모델·TP≥2로
확장됐다" · **X1C-9** X1 아티팩트(또는 907100↔X1 교차)에서
**어떤 지연·스루풋·GPU-h 비교 수치도 인용 금지**(n=1, 비페어,
교차-잡, 두 수치 구성이 동시에 다름).

**X1은 Claim D 선결을 하나도 닫지 않는다**(#1·#2·#3·#4a·#4b·#5
상태 불변). GPU 장부: 이 트랙(R2 correctness) **0.28 GPU-h**
지출 불변, X1 ≈0.16 GPU-h **예정**(job 실행 중 — **결과 도착
후 갱신**, 지금은 미기재). `longctx_conflict` 15.42 GPU-h
장부와 별개.

★신규 방법론 게이트 — **#170**(X1 감사 §1(a), 양성대조의
"교란됨" 판정은 상위 API 근사[행별 split 수]가 아니라 **커널이
실제로 쓰는 granularity**[`MIN_BLOCK_KV` 양자화]로 해야 한다 —
게이트#9 계열 **스무 번째 재발**[gate #40이 열 번째, gate #163이
19번째였던 것과 같은 형태로 새 게이트 번호를 받는다], `CONSENSUS.md`
§3 항목190[신설]) · **#171**(X1 D4/rev2 설계, 교란 노브를 도입하는
실험은 그 교란과 무관한 배경 축[엔진 소스/커밋 등]을 이전 참조
실행과 고정해 드리프트를 통제하라 — 노브 하나만 진짜로 바뀌게
하라, `CONSENSUS.md` §3 항목191[신설]). "도구 파일이 버전관리 밖이면
규칙 자신에 이력이 없다"는 새 번호를 매기지 않는다 — 위 A항목이
바로 **기존 게이트 #157**의 완료 사례이므로 그 항목에 追記한다
(중복 방지, 아래 "방법론 게이트" #157 참조).

★기존 불변 배너 전부 승계(HE0·정책 순위·gate #13/#16·switch-
cost "닫았다" 금지·C2 인용정지(a)(b)·`CONSENSUS §1-24`·게이트
#14 "닫았다" 금지·stake #1 구조 판정, 교훈88). `CONSENSUS.md`
rev66→**rev67**(§3 항목190·191 신설[게이트 #170·#171 대응] +
항목177 追記[게이트#157 완료], X1 등재는 §5-8(a) 追記) —
**판단 근거**: A(도구 이관)·B(Claim E 코드 수정)
는 코드/도구 사실 등재이며 새 분석 결론이 아니므로, 직전 세션
(2026-09-12 1차)이 세운 선례("코드 사실 등재는 매트릭스·로드맵·
이 문서 3개로 충분, CONSENSUS rev 불필요")를 그대로 따라
CONSENSUS에 반영하지 않는다. C(X1 규칙층 감사가 산출한 신규
방법론 게이트 2건)만 CONSENSUS rev 사유로 판단했다 — 이는 이전
세션들(Q-A rev8/9, Q-B′ rev65)이 GPU 결과 없이도 규칙층 감사의
신규 게이트를 즉시 CONSENSUS §3에 반영해 온 선례를 따른 것이다.
정본 반영: `reports/paper/{CLAIM_EVIDENCE_MATRIX,
EXPERIMENT_ROADMAP}.md` Claim D/E 절 갱신, `MEMORY.md` 포인터
갱신·`memory/deconfound-measurement-lessons.md` 항목168–169
신설. 상세 `workspace/engine-port/results/r2_correctness/
x1_prereg/{PREREG_X1_SENSITIVITY_2026-09-12.md,
VERDICT_x1_rules_2026-09-12.md}`, `tools/claude/README.md`.

이전: 2026-09-12(doc-steward — **engine-porter 코드 사실 4건
등재**[Claim E 설계 3건 + 도구 1건], 2026-09-11 확인·file:line
근거, **성능 판정 아님·수정 없음·사용자 결정 대기**, GPU 0). Claim
E(generic/hybrid R2 정책) 착수 전 해소가 필요한 코드↔로드맵 불일치
3건이 `reports/paper/{CLAIM_EVIDENCE_MATRIX,EXPERIMENT_ROADMAP}.md`
에 등재됐다 — ①admission 제한 latch가 평가 차례가 아닌 호출마다
기본값 `False`로 재덮어써 사실상 ~1 iteration만 유지
(`workspace/engine-port/src/multiplex/controller.py:68,124-130,
148-149`; `multiplexing_mixin.py:430-441,1066-1080`) ②평가 주기
연산이 코드(`controller.py:124-130`, `OR`=먼저 오는 것)와 로드맵
(`max(4 iterations, 100 ms)`=둘 다 채운 뒤)에서 반대 ③hybrid
profile 호환 검사(`profile.py:98-114,268-279`)가 `engine_commit`
(repo HEAD, `scripts/r2_eval/r2_eval.sbatch:26`)[HIST]을 포함해
엔진과 무관한 문서 커밋만으로도 fallback(`upper_bound_itl_ms=inf`)
에 빠질 수 있음(B6 arm이 기본값으로 profile-less가 될 수 있음) —
★**이미 해소됨, 위 2026-09-12(2) "B) Claim E 선결 3건 해소" 참조**
(커밋 `3f188bb`가 `engine_commit`을 엄격 일치에서 제외; 인용
`r2_eval.sbatch:26`도 이후 `:69-70`으로 이동, 2026-09-14 정정).
도구
사실 1건(게이트 #157 계열 — `/scratch/ehmoon/whlee/.claude/agents/
git-committer.md:25-27`이 outer 워크스페이스를 별도 git 저장소로
서술하나 실제로는 `.git`이 빈 디렉터리라 저장소가 아님)은 아래
게이트 #157 항목에 追記. **전부 고치지 않고 표시만 해 둔다**(사용자
결정 대기). Claim D/E 등급 무변경(둘 다 미검증)·새 성능 판정
0건·arm 순위 0건·정책 순위 변경 0건·HE0·stake #1 전부 불변.
`CONSENSUS.md` rev **변경 없음**(rev66 유지 — 코드 사실 등재이며
새 분석 결론이 아니라 이 3개 문서(매트릭스·로드맵·이 문서) 등재로
충분하다고 판단, doc-steward). 상세는 `reports/paper/
CLAIM_EVIDENCE_MATRIX.md` "주장 제한" Claim E 항목,
`EXPERIMENT_ROADMAP.md` "Controller defaults" 절 追記, 아래 게이트
#157 追記(전부 2026-09-12).
이전: 2026-09-11(3) **(doc-steward — R2 true-dual GPU
correctness 트랙 등재, 새 성능 판정 0건·Claim D 등급 불변
[미검증]·HE0·정책 순위·stake #1 전부 불변, 이 세션은 문서
등재분[GPU 0] — 트랙 자체는 오늘 0.12+0.16≈**0.28 GPU-h**
기지출[이 트랙 최초 지출, longctx_conflict 트랙 누적 **15.42
GPU-h**와는 별개 장부])** — 2026-09-11 커밋 `02918e8`가
admission latch(`r2_admission_limited`) stale-True 버그를
수정했다(2026-07-24 사용자 결정으로 걸린 보류를 **해제**,
사용자 R2 복귀 결정에 따름). 재실행 job 907032(gpu42, 0.12
GPU-h)는 **FAIL** — true-dual 두 boot(TD1/TD2) 모두 split-prefill
ownership race로 크래시했다(구현 사실). 경합 수정 커밋
`874873b`, 이어 판정 규칙을 TD 출력이 생성되기 전에 사전
고정한 하네스 v2 커밋 `e16e93f`. job 907100(19:37–19:47 KST,
≈0.16 GPU-h, 아티팩트는 커밋 `ea99191`)은 **PASS**했다.

claims-auditor가 이 결과를 read-only 결과 감사했다(`workspace/
engine-port/results/r2_correctness/audit_r2corr_2026-09-11/
VERDICT.md`) — 제안 문장은 그대로 정본에 쓸 수 없으나 PASS
자체는 공허하지 않고, 스코프를 S16(순차)+O8(단일 probe·단일
background 중첩) 프로토콜로 고치면 **`CONFIRMED(scoped)`**다.
무스코프("동시 부하[C 층] 포함, 같은 토큰")는 관측으로
**`REFUTED`**다 — C01·C10·C19 3/32 요청이 L1=L2≠TD1=TD2
arm-분리 패턴을 보이며, 그 불일치를 "TD 결함"으로 돌리는 것은
**`NOT-YET-SUPPORTED`**다(C01은 triton KV-split 휴리스틱의
arm-특이 타이밍으로 산술 설명됨, C10·C19는 재구성 못함). **Claim
D 등급 영향은 0이다(미검증 유지)** — 이 job이 닫는 Claim D
선결은 #5(cudagraph-ON 호환, decode 한정·scoped) 하나, 부분
해소는 #2(GPU correctness, S/O 프로토콜 범위)뿐이다. #1(admission
latch)은 코드 수정(`02918e8`)만 있고, FixedPolicy에서는 그
경로가 한 번도 발화하지 않아(`admission_limited` 185개 결정 중
0, `r2_admission` 이벤트 0) **GPU에서의 발화는 이번에도 관측되지
않았다**.

등재 한 줄(판정서 §7.2, 문자 승계): *"R2 true-dual GPU
correctness: job 907100 PASS, `CONFIRMED(scoped)`, S/O 프로토콜
한정. 인용 금지 R2C-1…16과 필수 병기 P-1…7을 판정서에서 문자
그대로 승계한다. 새 성능 판정 0건, Claim D 등급 불변, HE0
불변."*

★**정정 追記**: 아티팩트 커밋 `ea99191`의 메시지 "since B8/O3
confirmed the realized split was D44"는 **R2C-9**로 정정한다 —
직접 근거는 `controller_decision` 텔레메트리(target=current=44,
60/60)이고, B8/O3는 표본 기반 검사라 근거로 쓰기에 부정확했다
(커밋 자체는 재작성하지 않는다).

★★인용 금지 R2C-1…16(문자 소재 = VERDICT §5, 전문은 정본에
복사하지 않음) 중 요지: **R2C-1** 무스코프 "같은 토큰" 금지
(허용형: "S16 순차·O8 단일-probe 중첩 프로토콜에서 토큰 동일") ·
**R2C-2** "동시/서빙 부하에서 TD≡legacy" 금지(C 층 3/32 arm-분리
불일치) · **R2C-3** C 층 불일치를 "TD 결함"·"무해한 노이즈"
양쪽 다로 단정 금지(원인 미확정) · **R2C-5** "토큰 동치가 D44
실현을 확인한다" 금지(동치 검사는 파티션 오결합을 보지 못함) ·
**R2C-11** "스레드 안전성·경합 부재가 검증됐다" 금지(host 중첩
13.5/19.4ms는 하한) · **R2C-14** 무스코프 "Claim D 증거"·"Claim
D 선결 해소" 금지. 필수 병기 P-1…7(스코프 튜플·D44 계산 분할·
C 층 수치·unsafe 기전·host 중첩 크기·하네스-러너 편차·새 성능
판정 0건)도 문자 승계, 전문은 VERDICT §5. 인용 금지 총계
**116건**(기존 100+R2C 16, 트랙별 계수는 총계와 별개로 유지:
Q-A 계보 80+QB 20+R2C 16).

★★신규 방법론 게이트 3건(#167–169, 기존 #1–166과 대조해 중복
없음) — **#167**(G-1, 진단 층으로 강등해도 주장 스코프에서는
빠지지 않는다) · **#168**(G-2, 같은 이름의 텔레메트리 필드가
arm마다 다른 술어로 채워질 수 있다) · **#169**(G-4, 워커 경로를
탔다는 것[태스크 카운터 증가]은 동시 실행의 증거가 아니다).
追記 3건(신규 번호 없음): **G-3**(#50/`CONSENSUS §3` 항목70
追記, 동치 게이트의 검출력은 스코어러 변이 테스트로 보증되지
않는다 — #50의 측정 층 판본) · **G-5**(#1·#114/`CONSENSUS`
§1-9·§3 항목130 追記, 파티션 실현은 "창 안에 분할이 있었다"가
아니라 "비교 대상 산출의 어느 계산이 그 분할 위에서 돌았는가"로
적어라) · **G-6**(#95/§3 항목115 追記, 검증 표적이 설정의 기본
동작과 같으면 실현 검사는 두 가설을 구별하지 못한다 — confound
#10 사례). 하네스 결함 H1–H3(INPUT_IDENTITY가 개수만 비교·
`task_count`가 `set_result` 뒤에 증가해 Δ+1 지연·overlap ratio
1024개 절단+수명 분모)은 게이트가 아니라 engine-porter 이관
사항으로만 등재.

후속 실험 X1(민감도 양성대조+C 기전 판별)·X2(비기본 split
D16)·X3(러너 설정 인증), 각 ≈0.16 GPU-h — **전부 미실행·
미승인**(사용자 판단 대기). `reports/paper/EXPERIMENT_ROADMAP.md`
"P1/P2" 절에 열린 항목으로만 등재.

★기존 불변 배너 전부 승계(HE0·정책 순위·gate #13/#16 "닫았다"
금지·switch-cost "닫았다" 금지·C2 인용정지(a)(b)·`CONSENSUS
§1-24`·게이트 #14 "닫았다" 금지·stake #1 구조 판정). 정본
반영: `CONSENSUS.md` rev65→**rev66**(§5-8(a) 追記·§3 항목
187–189 신설), 이 문서 아래 "다음 실험 gate" 항목1–4 追記·
"방법론 게이트" #167–169 신설(+#1·#50·#95·#114 追記),
`reports/paper/{CLAIM_EVIDENCE_MATRIX,EXPERIMENT_ROADMAP}.md`
Claim D/R2 절 갱신, `MEMORY.md` 포인터 갱신·`memory/slo-aware-
scheduling-track.md`[2026-09-11(3)]·`memory/deconfound-
measurement-lessons.md` 항목165–167 신설. 상세 `workspace/
engine-port/results/r2_correctness/{job_907032/, job_907100/,
audit_r2corr_2026-09-11/VERDICT.md}`.

이전: 2026-09-11(2) **(doc-steward — Q-B′ 사전등록
규칙층+결과 감사 `GO-with-caveats` 등재, 새 성능 판정 0건·arm
순위 0건·정책 순위 변경 0건·HE0 불변·stake #1 불변, GPU 0,
트랙 누적 **15.42 GPU-h 불변**)** — claims-auditor가
`PREREG_QB_LOOPGAIN_2026-09-11.md`(후계 질문 Q-B의 사전등록,
751행)를 규칙층+결과 감사 겸으로 심사했다(`audit_qb_rules_
2026-09-11/VERDICT.md`): **死因 0·N1–N4 미발화·반전 0/23표면**
(초안 14표면 재검+감사 추가 9표면). 등재 전 필수 수리 D1–D9
(전부 GPU 0) 이행을 조건으로 "가(可)".

**명칭 변경 Q-B → Q-B′**("공통 λ에서 decode 평균 ITL 차의
노출·인구 합성 표준화 분해, 경로 cf 지정") — 원 Q-B("SM-split
액추에이터 자기상쇄 루프 이득")가 가리킨 프로브 C 교락 수치
(out=96 14.953×, out=384 23.713×)는 **Little 항등식**(도착
비×체류 비)으로 소진됨이 raw에서 확인됐고, 공통-λ 설계 자체가
원 교락의 도착 채널을 구조적으로 끈다. "루프 이득"·"액추에이터"·
"자기상쇄"·"되돌린다"는 인용에서 퇴역(정적 arm 사이엔 되먹임
루프가 없다).

**Stage 0**(기존 Q-A 192+P7 36 bench에서 계산, **GPU 0**)의
유일한 신규 측정량은 토큰 노출 계측기 `E`다 — 위약·정렬 검사로
타당성 확인(위약 최대 0.039ms, 구간 조기종료 변이에서 0.704ms
로 실패 ⇒ 검사가 작동함을 입증). `E`는 `ρ`의 재진술이 아니다
(ρ-대리 R² 0.529). **등재는 "탐색적 계측 기록(사후 정의)"으로만**
(게이트#150형 방어 자산 없음, 1차 경로 지정은 "값을 본 뒤의
지정"으로 병기, 부호 진술 0개 유지). **Stage 1(≈2.9 GPU-h)은
실행 허가되나 저자·감사 모두 비권고**(하류 소비자 없음, 확증도
예측 0개라 구조적으로 불가) — **미실행, GPU 0.**

★★인용 금지 QB-1…QB-20(소재 `PREREG_QB_LOOPGAIN_2026-09-11.md`
§12·판정서 §5.2) + 필수 병기 ⓐ–ⓕ. **인용 금지 총계 100건**
(Q-A 계보 80+QB 20). 신규 방법론 게이트 4건(**#163–166**, 기존
#119–162와 중복 없음) — #163(G-α, 분해 성분을 대수적 주항으로
회귀한 R²는 항등식 잔차 크기다, 게이트#9 계열 19번째 재발)·
#164(G-β+G-γ 병합, 비고유 분해의 경로 지정 방향 귀결은 대수로
정해질 수 있다 — 지정 전 계산·공개, 거친 층에서 외삽 여부 확인)·
#165(G-ζ, 층 세분 강건성 시험은 지지 결손으로 가짜 민감도를
만든다 — off-support 질량 병보고)·#166(G-η, 인계된 질문이
항등식으로 소진되면 정식 판본은 다른 질문이다 — 개명·소진
명시, Q-B→Q-B′ 사례). 追記 3건(신규 번호 없음): #159(G-ε 사례)·
#160(G-δ 사례)·#83/교훈88(G-θ 사례) — 본문은 아래 "방법론
게이트" 절 참조.

★직전 핸드오프 열린 항목5(`CONSENSUS §3` 후보 항목182 미등재,
게이트#162 대응)를 이번 세션에서 **등재 처리**한다 —
`CONSENSUS.md` rev65가 항목182(게이트#162)·183(#163)·184(#164)·
185(#165)·186(#166)을 신설.

★기존 불변 배너 전부 승계. GPU 지출 이번 0, 트랙 누적 **15.42
GPU-h 불변**. ★R2 쪽(job 907032 FAIL·경합 수정 `874873b`·
하네스v2 `e16e93f`)은 재실행 결과가 오면 별도 rev에서 등재 —
이번엔 `reports/paper/CLAIM_EVIDENCE_MATRIX.md` Claim D 행에
latch 수정 `02918e8` 완료(보류 해제, 사용자 R2 복귀 결정)라는
코드 사실 한 줄만 반영(성능 판정 아님). 정본 반영: `CONSENSUS.md`
rev64→**rev65**(§5-6 追記·§3 항목182–186 신설), 이 문서 "다음
실험 gate" longctx_conflict 행 追記·"방법론 게이트" #163–166
신설(+#159·#160 追記), `reports/paper/{CLAIM_EVIDENCE_MATRIX,
EXPERIMENT_ROADMAP}.md` Q-B′ 관련 절 갱신, `MEMORY.md` 포인터
갱신·`memory/slo-aware-scheduling-track.md`[2026-09-11(2)]·
`memory/deconfound-measurement-lessons.md` 항목161–164 신설.
상세 `workspace/engine-port/results/longctx_conflict/
{PREREG_QB_LOOPGAIN_2026-09-11.md, audit_qb_rules_2026-09-11/
VERDICT.md}`.

이전: 2026-09-11 **(doc-steward — Q-A `REGRET` 캠페인(jobs
906504–906507) 완주 + 독립 결과 감사 등재, 새 성능 판정 0건·arm
순위 0건·정책 순위 변경 0건·HE0 불변·stake #1 불변, 이 세션은
문서 등재분만[GPU 0] — 캠페인 자체는 **10.641 GPU-h** 기지출)**
— 캠페인이 **4/4 `COMPLETED 0:0`**로 완주했다: main 192/192·
포화 20/20·부팅 16/16·`SELFCHECK_PASSED` 4/4·manifest 4라운드
동일·등록 정합 192=192·교차검증 최대 5.847e-06·벽시계
38,308s = **10.641 GPU-h**(예산 대비 +2.3%). 운영점은
**cudagraph-ON**(192/192 `disable_cuda_graph=False`).

claims-auditor가 결과 문서(`RESULT_QA_REGRET_2026-09-11.md`)를
**read-only**로 적대 감사했다(`audit_qa_result_2026-09-11/
VERDICT.md`) — `qa_analyze.py`를 **쓰지 않고** raw
`bench_*.jsonl` 192개 + `sat_*.jsonl` 20개에서 등록 정의를
독립 재구현해 Δ행렬·CI·`sign_agree`·분산성분·모드·포화·
makespan·span을 재계산, **전 항목 4자리 일치**(불일치 0건).
해석 규율(arm 순위·argmin/argmax·"효과 없음" 금지 등)도 실질적
으로 준수됐음을 확인. 그러나 **사실-보고 절에 재현되지 않는
수치 결함 9건(D1–D9)**을 발견 — **D1**(§15 #1이 σ_req-free
`Δ(d44,d54)`를 "6칸[1·1·2·2·2·3]"으로 적었으나 실제는 §6.3이
옳은 **8칸[1·1·2·1·2·2·3·3]** — 자기 문서 내 모순), **D2**
(§13.2의 `1.958×`와 "라운드 간 변동 ≤1.6%"는 네 독법[`split_
fired` 1.9683·`f_time` 1.9865·arm평균 1.9486/1.9601, d16 변동
2.45%/2.29%] **어디에서도 재현되지 않음**), **D3**(§15 #5가
`gap≥4ms` **항등식**을 "작동하는 주장 … 참"으로 서술 — §16이
문자 승계한 rev8 6-7과 자기모순, **게이트#9의 18번째 재발**)은
등재 전 정정 필수였고, **D9**(산출 8 라벨 `STRUCTURALLY
VACANT[대상 부재]`는 과잉 — 정확 처분은 "미이행, 사유: rev5
§0/§1이 보간 기저[referent]를 폐기해 등록 기저가 남지 않음")는
라벨 강등 필수였다. **이 정정은 이미 원자료 문서에 반영·
커밋됐다**(`bc0033c`, 2026-09-11, D7[도구 미커밋] 포함 전부
이행) — doc-steward는 **정본(canon)에만** 반영한다.

**정본 등재 가부 = 조건부 가(可) — "계측 기록"으로만**(감사
판정서 §9, 조건 = D1·D2·D3 정정 + D9 라벨 강등 + `qa_analyze.
py`/결과 JSON/결과 MD 커밋, **전부 충족**). 등재 문장은 판정서
§9-1(정본 가능 10건)·§9-2(인용 금지 N-1…N-12, 문자 승계)·
§9-3(기존 기재 변경 3건)을 **그대로** 따른다(새 해석·새 판정
추가 0건).

★★**정본 가능 10건(§9-1, 요지 — 수치는 원문 그대로)**: ①
캠페인 완주 통계·**10.641 GPU-h**·cudagraph-ON 192/192 ② 등록
estimand(72칸×4추정량=288) 전수 산출, 등록 허가 `sign_agree=4`
는 **0칸 발화**(무순서 36칸 최대 3, 단 2칸이며 그 2칸도 대칭
진단에서 2로 내려간다 — rev6이 제출 전에 적어 둔 예보와 일치)
③ ⛔이 결과를 "효과가 없다"·"차이가 관측되지 않았다"로 옮겨
적는 것은 금지(rev8 6-1 문자 승계) ④ 허가 미발화의 지배 원인은
**이 캠페인 안에서** 이중계상 항이 아니라 **네 추정량 사이의
실제 부호 불일치**(무순서 36칸 중 **17칸**에서 두 추정량이 각각
CI로 0을 배제한 채 반대 부호, `σ_req`를 빼면 **22칸**으로 증가)
⑤ `n_boot=4`(df=3)에서 `σ_boot` **미검출**(양수5·미검출59·
퇴화43, 인용 가능은 비퇴화 상한 21건 **0.03–1.40%**뿐, ⛔"부팅
효과 없다" 금지) ⑥ 계측 건전(도착 창 예보 135.76–158.82s ↔
관측 135.81–158.98s, 상대오차 평균 **0.2236%**, `max_running_
requests=48` 비구속[최대 `L_total` 14.311]) ⑦ d92 `μ_p` 갱신
0.1885(n=1)→**0.1834±0.0002**(n=4, −2.7%, **격자·`ρ`는 등록값
0.1885 불변**) ⑧ d16/d44/d54 셀의 **144/152(94.74%)**가 저장소
미측정 `X` 영역 아래(검정력 예보 전부가 외삽) ⑨ 미처치 바닥은
arm-무관: **13.0064–13.0158ms(0.072%)·TTFT 0.8768–0.8785s
(0.20%)**(rev6 6-8 인용 수치는 선행 자료 값이고 이 캠페인
실측이 그 아래) ⑩ 등록 축(P)·진단 축(R) 사이 **144 점추정 중
49건(34.0%)** 부호 반대(갈릴 수 있는 108건 중 **45.4%**) — 추정량
축은 이 estimand에서 답을 바꾸는 자유 표면.

★★**인용 금지 12건 신규(N-1…N-12, 문자 승계, 소재 = 판정서
§9-2) — 총 80건**(rev1–8 68 + 이 결과 감사 12, `RESULT_QA_
REGRET_2026-09-11.md` §16에 전문 등재). 특히 **N-4**: ⛔이
캠페인의 σ_req-free 결과를 "rev8 판정서가 틀렸다"·"서사가
바뀐다"로 쓰지 마라 — 그 투영은 rev8 판정서 자신이 **P7 부하·
모양 A·`homog6`·N 재표집 한정**으로 스코프했고 게이트#32는
**양방향**이라 이 캠페인이 그 수치를 "틀렸다"고 판정할 자격도
없다. 쓸 수 있는 것은 *"그 투영은 이 캠페인으로 전이되지
않는다"*까지. **N-5**: ⛔같은 `λ`의 arm들은 처치 강도가 맞춰져
있지 않다(`ρ(d92)/ρ(D) = μ_p(D)/μ_p(d92)` 항등식 **4.754×**;
λ=0.09에서 0.110↔0.512) — **`Δ` 행렬의 어떤 칸도 "split만
다른 비교"가 아니다**(confound #10의 유일한 실재 항목). 나머지
N-1·N-2·N-3·N-6·N-7·N-8·N-9·N-10·N-11·N-12는 §16/§9-2에
문자 그대로 등재돼 있다(옮겨 적지 않음, 소재만).

★기존 기재 변경 3건(§9-3): `μ_p_ref(d92)` 인용처에 **0.1834±
0.0002(n_boot=4)** 병기(정의는 0.1885 불변) · rev6 6-8의
미처치 바닥 TTFT 범위 인용처에 *"선행 자료 값이며 이 캠페인
실측은 0.8768–0.8785s로 그보다 낮다"* 병기 · "새 성능 판정
0건·arm 순위 0건·정책 순위 변경 0건·HE0 불변·stake #1 불변"이
**감사 전수 확인으로 참**임을 등재.

★★**GPU 장부 갱신**: 트랙 누적 **4.78 → 15.42 GPU-h**(이
캠페인 10.641 GPU-h 완주분 가산) — 이 트랙 **최초의 완주된
캠페인**(그 전까지는 전부 계측 프로브이거나 PENDING).

★★신규 게이트 4건(#158–161, 판정서 §11 G-a/G-b/G-c + 메인
세션 발의 1건, 기존 #119–157과 대조해 중복 없음 확인) —
**#158**(G-a, 절[§]을 폐기할 때 그 절을 입력으로 쓰는 산출
목록 항목이 고아로 남을 수 있다 — 폐기는 산출 목록까지 전파
확인해야 한다, 교훈89의 산출-목록 판본. rev5가 rev2 §2.4를
폐기하며 산출 8을 남겼고 감사 3회를 통과) · **#159**(G-b,
결과 문서의 요약 표가 본문 절과 다른 수를 실을 수 있다 —
요약 표는 본문에서 기계 파생하거나 교차 대조 자기점검을
넣어라. 실사례 D1: 8칸↔6칸) · **#160**(G-c, 인용 금지를
문자 승계해 놓고 같은 문서의 다른 절이 그 금지 대상을 "참인
주장"으로 재긍정할 수 있다 — 승계 목록과 본문 주장의 술어
대조 필요, 게이트#9의 **18번째 재발**. 실사례 D3) · **#161**
(메인 세션 발의: 요약에 쓴 통계의 출처 표본이 요약이 주장하는
전체 표본이 아니라 부분[예: 라운드 1개]일 수 있다 — "전 라운드
재현"이라는 서술과 원자료를 직접 대조해야 드러난다. 실사례:
D2의 `1.958×`가 4라운드 중 라운드 1 단독 값이었음. G-b[#159]
와 인접하나 판정 대상이 다르다[개수 불일치 vs 출처 표본
불일치]로 별개 유지).

★기존 불변 배너 전부 승계(HE0·정책 순위·gate #13/#16 "닫았다"
금지·switch-cost "닫았다" 금지·C2 인용정지(a)(b)·`CONSENSUS
§1-24`·게이트#14 "닫았다" 금지·stake #1 구조 판정 — **감사가
전수 확인**). 정본 반영: `CONSENSUS.md` rev63→**rev64**(§5-6
追記·§3 항목178–181 신설), `PROJECT_STATUS.md` "다음 실험
gate" longctx_conflict 행 追記·"방법론 게이트" #158–161 신설,
`reports/paper/{CLAIM_EVIDENCE_MATRIX,EXPERIMENT_ROADMAP,
DOCUMENT_STATUS}.md` Q-A 관련 절 갱신(등급 변경 0건),
`MEMORY.md` 포인터 갱신·`memory/slo-aware-scheduling-track.md`
[2026-09-11]·`memory/deconfound-measurement-lessons.md` 항목
157–160 신설. 상세 `workspace/engine-port/results/longctx_
conflict/{RESULT_QA_REGRET_2026-09-11.md, audit_qa_result_
2026-09-11/VERDICT.md, probes/qa_analyze.py, probes/qa_regret_
result.json}`.

이전: 2026-09-10 **(5차 세션, doc-steward — Q-A rev8·rev9
"실행 중" 분석 정의 등록 + 그 판정서(rev8) 등재, 새 성능 판정
0건·arm 순위 0건·정책 순위 변경 0건·HE0 불변, GPU 0)** — 캠페인
(jobs 906504–906507)이 아직 PENDING인 동안, result-analyst가
분석기(`probes/qa_analyze.py`)를 작성하며 **4추정량(`pooled_p95`·
`median`·`mean`·`trim10`)이 어느 배열의 범함수인지 rev7까지
사전등록 문면에 없었다**는 것을 자기점검(`qa_analyze_selfcheck.
json`)으로 발견했다. rev8이 그 축을 **`PRIMARY_AXIS := 풀링
토큰 ITL 배열`**로 등록해 문면 결함을 메웠고(★핵심 발견: 미등록
축 R[요청별 `itl95`]에서는 캠페인의 유일한 등록 허가
`sign_agree=4`가 3쌍×3 N 전부에서 발화하는데, rev6/rev7이 실제로
쓴 등록 축 P[풀링 토큰]에서는 **한 번도 발화하지 않는다** — 즉
rev7 제출 시점까지 이 축 자체가 `N2`급 자유 표면이었다), rev8을
규칙층 재감사에 회부했다.

★**rev8 판정 = `GO-with-caveats`**(死因 0·반전 **0/10 표면**,
`audit_qa_rev8_2026-09-10/VERDICT.md`) — A1(축 P 등록)은 **사후
선택이 아니라 정합성 복원**이다(3+1갈래 근거: ① rev7 §2·rev6
§1(d)의 연역[단 `mean`은 균일성으로만 덮임] ② ★**rev6 §3이
제출 전에 적어 둔 예보 12값 자체가 축 P의 값**[사전등록이 자기도
모르게 미등록 정의를 특정해 뒀다] ③ 이해상충 방향이 반대[축 R을
골랐다면 허가가 전면 발화했을 것을, A1은 **자기 캠페인의 허가를
죽이는 쪽**을 골랐다] ④ 시각순서[자기점검 18:16:42 < 첫 bench
18:17:49 < rev8 18:22:08]). ⚠️단 그 대가로 `Δ(d44,d54)`에서는
rev8 자신이 "이중계상"이라 인정한 `σ̂_req²/8` 항 하나가 허가를
단독 결정하는 상태가 됐다(빼면 4/4, 넣으면 2/4–3/4).

rev9는 판정서 처방 P2–P5(전부 비용 0·GPU 0)를 이행 — 코드에만
있던 분석 상수 4건(`σ̂_req` **페어드** 재표집·부트스트랩
`B=10000`/seed 1·모드 **값 정렬**·`UNTREATED_TOL=1.10×ITL_SOLO`)
을 현재 구현 그대로 등록, 1시드 rung 대칭 진단 병기(등록식
불변), 등록 산출 7 구현, ★**rev8 erratum 3건 정정**(§A3
"3.5–3.9배"는 반폭÷SE **단위 오류** — 옳게는 SE 기준
1.06–1.22×[`median`@`Δ(d44,d54)`만 4.33×] · CI 상단 전정밀도
1.713666[rev8의 1.71373은 SD 재반올림] · 표 열이름 `pooled_p95`
행 부정확).

★★**승계 인용 금지 총계 68건**(rev1–rev7 55건 + rev8 판정서
13건, 전문은 옮기지 않는다 — 소재만: `PREREG_QA_REGRET_
REV{1..7}_2026-09-10.md`·`audit_qa_{rules,rev3,rev4,rev5,rev6}_
2026-09-10/VERDICT.md`[55건], `audit_qa_rev8_2026-09-10/
VERDICT.md` §6[13건]). 결과 문서가 **문자 그대로** 승계할 핵심
3건: ⓐ 부호 진술을 못 하더라도 "효과 없음"으로 쓰기 금지(원인이
쌍마다 다르다 — `Δ(d16,·)`는 `median` 효과 −0.2165ms·`σ_seed`
10.79ms, `Δ(d44,d54)`는 이중계상 항 하나) ⓑ `sign_agree=4`를
"통과 가능했던 문턱"으로 쓰기 금지(등록 투영 9칸 전부 ≤3/4) ⓒ
1시드 rung(λ∈{0.25,0.42,0.1162,0.70})의 `sign_agree`를 다른
rung과 같은 자로 비교 금지(A2가 反보수, 최소 여유 0.25% < MC
오차 0.7%). **캠페인 결과·기대는 이 세션도 0건 기록**(jobs
906504–907 여전히 PENDING/RUNNING, 트랙 누적 4.78 GPU-h 불변).

★★rev8 판정서 §8의 신규 방법론 게이트 후보 9건(α–ι) 중 **8건을
신규 게이트로 등재(#149–#156)**, **1건(ε)은 기존 #130(중복)에
사례 追記** — ε("고정 시드 성분을 n으로 나누지 마라")는 A2의
死因(공통 trace를 공유하는 4부팅에 `σ̂_req²`을 나눠 SE 최대
2배 과소평가)이 #130("자유 표면을 '동결'한 것을 '제거'라 쓰지
마라 — 동결은 분산을 반복 축에 앨리어스된 편향으로 바꾼다")과
같은 통계적 오류(반복해도 평균화되지 않는 고정/공유 성분을 `n`
으로 나눔)의 새 사례이기 때문. 최고 전이가치 3건: **#149**(α,
추정량 **이름**을 등록하는 것은 추정량을 등록하는 것이 아니다
— 어느 배열의 범함수인가가 판정을 뒤집는다[같은 원자료에서
`median`이 −0.2165 ↔ +23.0220, 허가 0/9칸 ↔ 9/9칸]) ·
**#150**(β, 사전등록이 자기 예보 수치를 적어 두면 그 수치가
미등록 정의를 사후에 특정한다 — 결함이 아니라 방어 자산) ·
**#151**(γ, 분석 정의를 실행 중에 고쳐야 한다면 그 수정이
"허가를 여는 쪽"인지 "닫는 쪽"인지 수치로 먼저 보고하라 —
닫는 쪽이면 이해상충 방어 성립, 여는 쪽이면 자동 死因). 나머지
5건: **#152**(δ, 정렬을 단언하는 괄호는 검사가 아니다 — 위반
시 처분과 함께 등록하라[47셀 중 33셀에서 위반]) · **#153**(ζ,
"보수적이니 유지"를 주장할 땐 보수성 배율을 판정 단위[SE
기준]로 계산하라 — 반폭÷SE는 `t`를 곱한 값) · **#154**(η,
결정 마진이 부트스트랩 MC 오차보다 작으면 그 칸의 판정은
seed의 함수다) · **#155**(θ, "분석기가 등록 격자와 일치한다"는
등록 산출 전부가 구현됐다는 뜻이 아니다) · **#156**(ι, 실행 중
관측을 사전등록에 적을 때의 봉인은 "n=1·판정 금지"가 아니라
"그 관측이 문제의 선택과 직교함을 수치로 보이는 것"이다).
기존 #119–148·§3 항목139–168과 대조해 중복 없음을 확인했다
(δ는 "동률/tie-break" 게이트[#145]와, θ는 "compute_coverage류는
내부 구멍에 맹목"[초기 게이트#29]과 인접하나 판정 대상이 달라
별개로 유지). 전문·계보는 `CONSENSUS.md` §3 항목169–177.

★★★**도구 파일이 버전관리 밖이라는 사실을 등재(신규 게이트
#157)**: 게이트#118 개편의 **작동하는 절반**인
`/scratch/ehmoon/whlee/.claude/agents/claims-auditor.md`
(63→109줄, SHA-256
`47e800cf6b663752730c492137f06cab7ede58d29c38dea6f55211656835ae02`
— **2026-09-11 doc-steward 追記**: 경로 1줄 정정(존재하지 않는
`workspace/engine-port/reports/CONSENSUS.md` → `reports/
CONSENSUS.md`), 규칙 내용 불변. 새 SHA-256 =
`e7c2dedfe53527a84a0990c50607135d9ba4200d85a151f919aa7f81e7ed93b8`)
이 있는 워크스페이스 루트 `/scratch/ehmoon/whlee`는 **git
저장소가 아니다**(`.git`이 빈 디렉터리, `git status` →
`fatal: not a git repository`) ⇒ 오늘 이 저장소(`prefill-layer-
alloc`)에 만들어진 커밋 **`d869751`·`e2c2ba1`에 그 파일이 들어갈
수 없었다**(별도 워크스페이스). ★이것은 **교훈 항목97**("게이트
등재돼도 도구가 안 고치면 재발")의 **뒷면**이다 — 이번엔 도구는
고쳐졌는데 **그 수정이 기록(버전관리)에 없다**. 신규 게이트
**#157**: *"게이트의 작동하는 절반이 버전관리 밖에 있으면 그
게이트는 드리프트 탐지·재현이 불가능하다"* — 이 SHA-256을
정본에 박아 두는 것 자체가 임시 완화책(다음 세션이 이 값과
다르면 무단 수정 탐지 가능)이며, 근본 해법(그 파일을 어느
저장소로 편입할지)은 사용자 소관으로 남긴다.

★★追記(2026-09-12, doc-steward, GPU 0 — 게이트 #157 계열, 도구
사실 추가 사례): 위와 같은 형태의 결함이 **다른 도구 파일에도
있다** — `/scratch/ehmoon/whlee/.claude/agents/git-committer.md:
25-27`은 outer 워크스페이스(`/scratch/ehmoon/whlee`)를 "별도
`.git`, 원격 있음"인 저장소로 서술하나(engine-porter, 2026-09-11
확인), 위에서 이미 확인한 대로 그 경로의 `.git`은 **빈
디렉터리**이고 `git status`는 `fatal: not a git repository`를
반환한다(doc-steward 2026-09-12 재확인, `ls -la`·`git status`
직접 실행). 2026-09-11 doc-steward가 `claims-auditor.md`의 오기
경로 문자열은 정정했으나(위 참조), `git-committer.md`의 이
서술(경로 문자열이 아니라 에이전트 동작 지침 문장)은 그 정정
때 고치지 않고 **표시만 해 두었던 항목**이다 — **아직 고치지
않고 보류**(도구 파일 버전관리 이관 여부와 함께 사용자 결정
대기). 신규 게이트 번호 없음(#157 자체의 추가 사례). GPU 0·새
성능 판정 0건. 정본 대응은 이 항목(#157)뿐이며 `CONSENSUS.md`
rev는 변경하지 않는다(원 게이트#157/§3 항목177은 그대로 유지,
이번 追記는 이 문서에만 등재 — 코드 사실 확인이며 새 분석
결론 아님).

★기존 불변 배너 전부 승계(HE0·정책 순위·gate #13/#16 "닫았다"
금지·switch-cost "닫았다" 금지·C2 인용정지(a)(b)·`CONSENSUS
§1-24`·게이트#14 "닫았다" 금지·stake #1 구조 판정). GPU 지출
이번 세션 **0**(전부 문서 등재 — 캠페인은 여전히 PENDING/
RUNNING). 정본 반영: `CONSENSUS.md` rev62→**rev63**(§5-6
追記·§3 항목169–177 신설·항목150 追記), `PROJECT_STATUS.md`
"다음 실험 gate" longctx_conflict 행 追記, `reports/paper/
EXPERIMENT_ROADMAP.md` Q-A 행 갱신(등급 변경 0건),
`MEMORY.md`·`memory/slo-aware-scheduling-track.md`
[2026-09-10d]·`memory/deconfound-measurement-lessons.md`
항목148–156 신설(+항목129 追記). 상세
`workspace/engine-port/results/longctx_conflict/{PREREG_
QA_REGRET_REV8_2026-09-10.md, PREREG_QA_REGRET_REV9_2026-09-10.md,
audit_qa_rev8_2026-09-10/VERDICT.md}`.

이전: 2026-09-10 **(4차 세션, doc-steward — `longctx_conflict`
Q-A 트랙 rev2→rev7 규칙층 감사 계보 전체 등재 + 캠페인 제출 반영,
새 성능 판정 0건·arm 순위 0건·정책 순위 변경 0건·HE0 불변, GPU
0[이번 세션 문서 등재분])** — Q-A(고정 split의 regret 프론티어,
쌍 페어드 차 행렬 estimand, `min`·argmin·포락선 산출 금지)
사전등록이 **이 트랙 최초의 `GO-with-caveats`**를 받았다.

**계보**: rev2(`.../longctx_conflict/audit_qa_rules_2026-09-10/
VERDICT.md`, 대상 `PREREG_QA_REGRET_2026-09-10.md`) `NO-GO`(N2,
반전 5건 F1–F5 — 실행 상한 `λ≤0.95·s_min·μ_p`가 셀 실행 자격을
난수[`span_factor`]의 함수로 만들어 부팅이 서로의 반복이 아니게
됨, `n_boot`가 칸마다 2/2/1/0으로 붕괴, 트랙 계열 누적 **18연속**)
→ rev3(`audit_qa_rev3/VERDICT.md`) `NO-GO`(반전 3건 V1–V3 —
포락선 정의역이 측정량이라 실행 격자를 고정해도 라운드 의존,
반전 크기가 rev2의 1.52ms에서 **25.52ms[16.8×]**로 확대) →
rev4(`audit_qa_rev4/VERDICT.md`) `NO-GO`(반전 3건 — `min`이
고르는 기준 arm이 추정량의 함수[`pooled_p95` argmin=d54 12/12
↔ `median` argmin=**d16 10/12**], 시드 동결이 "제거"가 아니라
rung-앨리어스 편향으로 전환[σ_seed/σ_boot **3.30×**, F=1.276<
4.066 미검출], 모양 B의 `𝒜`가 측정량의 함수[여유 1.05–3.1σ]) →
rev5(`audit_qa_rev5/VERDICT.md`) `NO-GO`(반전 1건 W1[세 다리]
— `c_valley`의 두 등록 구성요소[유도규칙 ↔ `{14,16,18}` 격자]가
서로 반대 라벨: `d16_r1_o384`는 유도규칙 not-pinned ↔ 격자
PINNED×3, `d92_r0_o96`는 **격자 안에서도** `u` 0.6171↔0.9868로
뒤집힘) → **rev6(`audit_qa_rev6/VERDICT.md`) `GO-with-caveats`
(死因 0·반전 0/6표면)** — 6개 자유 표면[동률 tie-break·히스토그램
빈 폭·`gap` 순위 검사·`gap<4ms` 게이트·`u` 정의역·`sign_agree`
집계 수준]을 등록 허용범위 끝까지 밀었으나 판정이 안 바뀜(W1은
실제로 죽음: `gap<4ms` 게이트가 §1(a) 정의상 **항등식**이라
절대 발화 못 하지만, d92는 미등록 제3경로로 여전히 배제되어
결과가 안 바뀜) → rev7(제출 판본, 판정서 §6 문면 수리 T1–T6
이행 + erratum 5건 정정, 측정 계획 변경 0). ★★**"N연속 `NO-GO`"
장부는 rev6에서 끊긴다** — 단 반드시 병기: **P7처럼 결정 규칙을
0개로 만들어 감사를 우회한 것이 아니라 결정 규칙 6개를 유지한
채 반전 시험을 통과한 첫 설계**다(rev6 판정서 §0·꼬리말). 승계
장부상 연속 카운트는 rev2 18 → rev3 19 → rev4 20 → rev5 21로
이어졌으나 각 판정서가 "독립 검증값 아님"(게이트#110)을 자기
등재했다.

★★**캠페인 제출(결과 아직 없음)**: `probes/qa_regret.sbatch`,
**jobs 906504–906507**(라운드당 1개, `--time=04:00:00`), 사용자
승인 예산 **≈10.1 GPU-h**(Stage A 폐지로 10.4→10.1 — rev7 §9:
`𝒜(λ,B):=𝒜(λ,A)` 확정 이후 Stage A 산출이 Stage B 포화 bench의
`n_boot=4`와 중복). main 192 bench + 포화 20. **제출 상태
PENDING, 완료 0/4·결과 0건** — ⛔**어떤 결과·기대도 기록하지
않는다.** 트랙 누적 GPU-h(**4.78**)에는 아직 가산하지 않는다
(완료 후 doc-steward가 갱신).

★계측 자산 3건 승계: **`span_factor(seed,N)`** 예보식(`probes/
span_factor.py` — 첫 `N−1`개 표준지수난수 평균, P7 12시드+
프로브 C 9셀 = 21표본 평균 |오차| 0.21%·0.084%, ⚠️`--random-
range-ratio 1.0`에서만 성립[0.9면 −17%]) · **`X` 예보식**
(`duration=max(span+ttft_last+decode_last, N/μ_p+decode_last)`,
48표본 평균 |오차| 0.21–0.24%, rev3 판정서) · **`RESULT_QA_
LADDER_LOO_2026-09-10.md`**(형식상 PASS이나 판정 **`VOID`**—
lever 0.90–0.996·귀무 최근접-앵커 예측기가 12조합 중 4개서
보간보다 정확·측정이 등록 정의역 밖에서만 이뤄짐, "사다리 밀도가
충분함이 확인됐다"로 인용 금지).

★★★**정본 정정 3건(판정서가 직접 뒤집은 것)**: ① rev4 §3-C
*"d92 정의역[X∈0.07–0.19]서 전 arm이 미처치 모드 ≈13.2ms로
수렴"*은 **미측정**이다 — `ITL_SOLO_S=0.013037`은 P2 `FLOOR`가
**d44 단독** 부팅으로 잰 상수이고(원문서가 "재확인 안 했음"을
자기 기록), 저장소에 `X<0.42`의 d92 외 arm 표본이 **0건**이다.
⛔**"d92 regret은 원래 식별 불가였다"로 쓰면 게이트 위반.** ②
rev6 §2-(1) *"d92는 간격 1.9ms의 좁은 이봉"*은 **거짓**이다 —
d92는 **단봉**이고 1.96ms는 지지집합 전폭이며, "13.04↔15.0"은
**서로 다른 셀**의 단봉이다. ③ **게이트#117(`σ_seed`는 부하다)은
페어드 `Δ` estimand에 전이되지 않는다** — 실현 도착 스팬의
설명력이 `R²` **0.116/0.130**뿐이다(원자료 `L_decode`의
0.95–0.99와 대조). estimand마다 재검증하라는 것이 교훈이다.

★결과 문서·정본이 문자 그대로 승계할 **필수 병기 3종**(인용
금지 55건[rev6 14건+rev2–rev5 41건]의 소재는 각 VERDICT 경로,
전문 인용 대신 소재만 적는다): ① 미처치 바닥이 **arm-무관**
(요청별 median-ITL 최소값 13.016–13.079ms·최소 TTFT 0.883–
0.903s 전 arm 동일, 기전 `CONSENSUS §3 항목26`=`multiplexing_
mixin.py:952-953`의 `(0,108)` 복귀) ② `|𝒜(λ)|`이 **처치 강도의
순감소 함수**(4-arm rung = 최저 두 부하, 1-arm[λ=0.70] = 최고
부하) ③ **저부하 외삽 노출 18/19 셀**(94.7%, 등록 19셀 중
18개가 저장소 최저 관측 `X=0.4186` 아래). ★**stake #1 구조
판정 불변** — 이 캠페인은 어떤 판본을 사도 `Ê`·argmin·순위·
포락선·"어느 arm이 최적" 산출을 사전등록 문자로 금지해 stake
#1을 열지 못한다.

★★신규 방법론 게이트 **30건(#119–#148)**, 원 후보 **35건**
(rev2[rules] 4·rev3 6·rev4 8·rev5 9·rev6 8 — 각 판정서 말미
"신규 방법론 게이트 후보" 절)을 중복 5쌍 병합해 등재했다: (i)
"기준 원소가 추정량의 함수"(rev3+rev4) (ii) "명목 노브 위 계산은
측정축에서 무효"(rev3+rev4) (iii) "재사용 상수의 출처 확인"
(rev3+rev4) (iv) "순환·상수공유 문턱은 항등식"(rules+rev6) (v)
"직전 caveat가 다음 판본에서 누락"(rev5+rev6, **2회 관측 재발**).
★특히 3건은 이 세션 최고 전이가치: **#119**(문턱을 판정층에서
지우면 자유 표면은 아래층[표본 자격]으로도 내려간다) ·
**#130**(자유 표면을 "동결"한 것을 "제거"라 쓰지 마라 — 동결은
분산을 반복 축에 앨리어스된 편향으로 바꾼다) · **#143**(감사
처방의 자기검사가 과잉일 수 있다 — 값을 하나로 고정하면 그
표면은 사라지므로 봉인이 불필요하다). **전 목록·병합 출처는
`CONSENSUS.md` §3 항목139–168.**

★기존 불변 배너 전부 승계(HE0·정책 순위·gate #13/#16 "닫았다"
금지·switch-cost "닫았다" 금지·C2 인용정지(a)(b)·`CONSENSUS
§1-24`·게이트#14 "닫았다" 금지·stake #1 구조 판정). GPU 지출
이번 세션 **0**(전부 문서 등재 — 캠페인은 제출만 됐고 미완료).
정본 반영: `CONSENSUS.md` rev61→**rev62**(§5-6 追記·§3 항목
139–168 신설), `PROJECT_STATUS.md` "다음 실험 gate"
longctx_conflict 행 追記, `reports/paper/CLAIM_EVIDENCE_MATRIX.md`·
`EXPERIMENT_ROADMAP.md` Q-A 행 갱신(등급 변경 0건). 상세
`workspace/engine-port/results/longctx_conflict/{PREREG_
QA_REGRET_2026-09-10.md, PREREG_QA_REGRET_REV{3,4,5,6,7}_
2026-09-10.md, audit_qa_rules_2026-09-10/VERDICT.md, audit_qa_
rev{3,4,5,6}_2026-09-10/VERDICT.md, RESULT_QA_LADDER_LOO_
2026-09-10.md, probes/{span_factor.py,qa_regret.sbatch}}`.

이전: 2026-09-10 **(3차 세션, doc-steward — 정본 번호 불일치
확정 정리, 새 성능 판정 0건·등급 변경 0건·GPU 0)** — "저장소가
같은 진단을 두 번 냈는데 정본이 안 바뀌면 도구 규율 실패"라는
교훈이 **게이트 #14(`CONSENSUS.md` §3 항목27)**와 **"게이트
#18"/"항목18"** 두 라벨로 병존해 온 것을 확정한다 —
**정본 번호는 오직 #14/§3 항목27뿐이다.** 사실 확인: 실제 게이트
#18="과부하 arm 비교는 시스템 상수가 아니다"(§3 항목32), 실제
`CONSENSUS.md` §3의 항목18="자기가 검증하려는 코드를 복사한
게이트는 항등식"(게이트#9 계열) — 둘 다 이 개념과 무관한 별개
교훈으로 이미 점유돼 있었다. 오인용 근원은 `memory/deconfound-
measurement-lessons.md`(장기 메모리)의 제3의 독립 내부 순번(그
파일 자체 1–117)에서 이 개념이 우연히 그 파일 자신의 18번째
항목과 일치했던 것이 `CONSENSUS.md` §3의 항목18과 혼동된 것.
전수 grep 조사로 이 절 곳곳(아래 "방법론 게이트" #40·#49·#97·
#101·#103·#115 서술 포함)·`CONSENSUS.md` §3 항목27·58·59·69·
117·123·135·`MEMORY.md`·양쪽 memory topic 파일의 재발 사례를
모두 찾아 원문은 보존한 채 오인용 정정 태그를 追記했다(각 사례의
판정 자체는 유효, 바뀐 것은 번호 표기뿐). `CONSENSUS.md` §3
머리말에 두 번호 축의 대응 규칙을 명문화했다. 상세
`CONSENSUS.md` §3 머리말·항목27 追記, rev61. ★**새 성능 판정 0건
· arm 순위 0건 · 정책 순위 변경 0건 · HE0 불변.**

이전: 2026-09-10 **(2차 세션, doc-steward — 연구 규율 변경
반영, 성능 판정 아님·GPU 0)** — 사용자 결정(2026-09-10)으로
규칙층 적대 감사(사전등록 설계 심사)의 차단 기준이 **"자유
표면이 존재한다"에서 "그 자유도가 등록 판정을 실제로
뒤집는가"(반전 시험)로 전환**되고, 등급에 **`GO-with-caveats`**
(死因 0·등록 caveat ≥1, 실행 승인 — 선례 gate #13 rev3)가
부활했다. 배경: `longctx_conflict` 트랙 계열 규칙층 감사
**17연속 `NO-GO`**(통과한 유일한 설계는 결정 규칙을 0개로 만든
순수 계측) 이후 `handoff-report/session_handoff_2026-09-10.md`
"열린 항목" #3의 "감사 프로세스에 정지 조건이 없다" 진단을
사용자가 채택. **도구가 이미 고쳐졌다**: `.claude/agents/
claims-auditor.md`에 "## 규칙층 감사 (사전등록 설계 심사) —
등급과 차단 기준" 절 신설(63→109줄, 반전 시험 절차 +
`GO`/`GO-with-caveats`/`NO-GO`[N1–N4] 판정 기준 인코딩) —
**교훈 항목97**(게이트 등재돼도 도구가 안 고치면 재발, 게이트
#14의 4번째 재발)의 직접 적용. **남는 위험**: 반전 계산 자체가
새 자유 표면이 될 수 있다(`PREREG_P6_ITL_ORDER` 판정서 전례) —
도구 절의 "반전 계산에 쓴 원자료·격자·추정량을 명시하라"로
방어했으나 완전 해소 아님. **제1원칙·기존 confound 카탈로그·
인용 금지 목록·HE0는 불변이며, 기존 17연속 `NO-GO` 판정들은
소급 재판정하지 않는다**(그 死因들이 새 기준에서도 `NO-GO`였을
것이라는 서술은 검증되지 않았으므로 쓰지 않는다). 신규 게이트
**#118**(대응 `CONSENSUS §3` 항목138). ★**새 성능 판정 0건 ·
arm 순위 0건 · 정책 순위 변경 0건 · HE0 불변 · GPU 0.** 상세
`.claude/agents/claims-auditor.md`, `handoff-report/
session_handoff_2026-09-10.md` "열린 항목" #3.

이전: 2026-09-10 **(1차 세션, doc-steward, Step 1[job 905958]+P7[job
905994] 계측 반영, 커밋 `7cd2d3f`·`b3b8b91`)** — `longctx_conflict`
트랙 계측 2건, **전부 순수 계측**: Step 1(0.244 GPU-h)이 死因
Y4(공통 `W` 실현 불가)를 arm 집합 수리로 닫았고, P7(2.606 GPU-h)이
이 트랙 **최초의 자기 워크로드 부팅간 분산 상한**을 샀다(`itl95`
2.6–3.2%·`μ_p` 2.4–6.1%). ★★후속 설계를 바꾸는 발견: **`σ_seed`는
잡음이 아니라 부하**(실현 도착률 폭 1.50×가 `L_decode` 산포의
지배 성분, R²=0.95–0.99) — 검정력 계산엔 `span_hat`을 공변량으로
넣어야 한다. 신규 인용 금지 3건 + 방법론 게이트 2건(#116–117)
신설. **새 성능 판정 0건 · arm 순위 0건 · 정책 순위 변경 0건 ·
HE0 불변 · stake #1 미답(구매 불가 종결 상태 유지, 변경 없음).**

**(L) Step 1**(job 905958, 0.244 GPU-h, `s1_armfix.sbatch`):
`pdmux_homog6.yml`(6그룹/division 4행, `D=54` 추가)이 **`BOOT_OK`
4/4**, 배너 6행 예상과 정확히 일치, division 비용 **0.0273%**
(homog5 대비). `μ_p` 실측 d16 **0.9143**·d44 **0.6666**·d54
**0.5772**(신규)·d92 **0.1885** req/s — 대조 3 arm이 homog5 앵커
**±2% 안**(0.980/0.987/1.010, 5→6 그룹 구조 변경 무해), d54가
감사 외삽 0.566과 **2.0% 일치**. ⇒ 등록 규칙 `W := 0.85 × min_D
μ_p`가 `W=0.491`·`ρ=0.537/0.737/0.851`(전 arm `(0.5,1.0)`)·도착
창 183.3s 전 arm 동일을 산출 ⇒ **rev2가 처방하고 P6에서 실현
불가였던 "공통 `W`"가 처음으로 성립, 死因 Y4(항목129/게이트#113)
닫힘.**

★**라벨러 정정**: `S1_LABEL.json`의 `mu_p_spread=4.849×`는
**부팅한 4 arm 전부**에서 계산돼 목표가 가리키는 양이 아니다
(게이트 #21 역방향 — 라벨러 결함이지 실험 실패 아님). 목표
≤2.4×는 P7 등록 arm 집합 **`{16,44,54}`**에 적용되며 **1.584×
PASS**(`max/min`은 중간 arm 추가로 줄지 않음 — 사후 선택 아님,
`PREREG_P7 §2`가 이 job **이전**에 등록).

**(M) P7 `ITL_SIGMA`**(job 905994, 2.606 GPU-h, 사전등록 예측
0건·결정 규칙 0건 — 순수 계측): **12/12 부팅·main 36/36·sat
12/12·단일 노드 gpu37·교차검증 최대 2.66e-06·실패 0건.** ★★핵심
산출 — **`σ_boot`는 `n_boot=4`(df=3)에서 분해되지 않는다**: 3
arm×2 지표 **전부 분산성분 원값이 음수**라 0으로 클립됐다 ⇒
**"`σ_boot=0`"은 점추정이 아니라 "미검출"**이며, 인용해야 할
것은 F-검정 기반 **95% 상한**: `itl95` d16 **3.2%**·d44
**2.6%**·d54 **2.7%**(부하 보정 후 2.9/2.7/3.0%), `μ_p`
(n_boot=4) 0.8962/0.6632/0.5736·σ 95% 상한 **6.1%/2.5%/2.4%** —
Step 1의 `n_boot=1` 값을 **0.5–2.0%로 재현**. ⚠️**`d44`/`d54`의
`L_decode` σ_boot 상한 "0.0%"는 상한이 아니라 퇴화**(`F<F₀.₀₅`) —
인용 금지.

**(N) ★★★후속 설계를 바꾸는 발견 — `σ_seed`는 잡음이 아니라
부하다**: 12 시드의 실현 도착 스팬 계수 **0.770–1.154** ⇒ 실현
도착률 **0.425–0.638 req/s**(명목 0.491, 폭 **1.50×**). 실현
도착률로 회귀하면 `L_decode`가 **R²=0.95–0.99**로 설명되고 잔차
`σ_seed`가 **15–26%→1.6–4.2%**로 붕괴. ⇒ **검정력 계산에
`σ_seed`를 그대로 쓰면 안 되고 `span_hat`(또는 실현 도착률)을
bench별 공변량으로 넣어야 한다.** `F`가 6개 중 2개에서 `F₀.₀₅`
아래인 것은 부팅 평균이 시드 3개로 도착 변동을 상쇄하는 **설계
결과**(라운드 평균 스팬 계수 폭 ±2.4%)이지 모형 위반 아님 —
위치·warmup 교락 배제(편차가 위치가 아니라 그 시드의 스팬 계수를
따름). ★**`RESULT_RULEPOWER_2026-09-08.md §7`·부록A와 교차참조
필수**: 그 문서의 검정력 표는 `SIG_BOOT`(부팅×split 랜덤효과,
실현 페어드 SD 7.1–24.3%)를 입력으로 쓰며 "goodput 페어드 SD를
다른 캠페인에서 수입하지 말 것"(게이트#32)을 이미 명시한다.
**P7의 `σ_seed`/`σ_boot`는 그 `SIG_BOOT`와 다른 양·다른
워크로드**이므로 수치 대입은 금지(같은 게이트#32)이나, **"원시
표본 산포를 검정력 입력으로 쓰기 전에 그것이 처치-상관 공변량
(부하)으로 설명되는지 회귀로 먼저 확인하라"는 방법론은
`RESULT_RULEPOWER`류의 모든 향후 검정력 재계산에 직접
적용된다**(신규 게이트 #117).

**(O) 계측 자산 갱신**: `span_hat` 복원식이 **독립 표본 36건**
(P7 main bench)에서 평균 |오차| **0.207%** — `audit_sweep_rev2
§R5`의 12셀 0.077%와 합쳐 **두 캠페인 48 표본에서 검증**.

**(P) ★★신규 인용 금지 3건**: ① P7의 TTFT 확률지배 위반 0(전
`T`, 격자 409점)은 **관측 사실일 뿐** — arm 순위·정책 해석 인용
금지(이미 등재된 `μ_p` 단조성의 귀결이며 P6가 `E2`로 등록하려다
"전제의 재확인"으로 `NO-GO` 받은 명제, 항목127/게이트#111). ②
`d44`/`d54`의 `L_decode` σ_boot 상한 "0.0%"는 상한이 아니라
퇴화(`F<F₀.₀₅`) — 인용 금지(위 M). ③ **"부팅 효과가 없다" 주장
금지** — `n_boot=4`에서 미검출일 뿐이며, 정본의 V-probe `d44`
6.13%·AF-1 4.35×를 **반증하지 않는다**(다른 워크로드·ctx,
게이트#32는 양방향).

**(Q) 다음 실험 gate**: **stake #1은 여전히 구매 불가로 종결
상태 유지**(위 rev58 (F) 불변). 후계 질문 **Q-A**(고정 split의
regret 프론티어)·**Q-B**(SM-split 액추에이터 자기상쇄 루프
이득)가 이제 **검정력 계산 가능**해졌음을 등재(P7의 σ 95%
상한 + `span_hat` 공변량 사용). 실행 계획 없음 — 다음 세션 몫.

**(R) 실행 전 자기 수정 1건(기록)**: `PREREG_P7 §3.1` — 등록
동작점이 전 arm 비포화라 `μ_p`가 측정 불가능함을 **실행 전**
메인 세션이 자기 발견 ⇒ 포화 bench를 추가(주 bench 36개 설계·
순서·시드·`N` 불변, 판정 규칙이 없어 판정 영향 0). ★이는
게이트#111("사전등록 예측은 실행 전 기존 원자료로 예보")의
정신을 실행 확인 측면에서 따른 **긍정 사례**이나, 새 死因·새
함정이 아니라 사전등록 절차가 의도대로 작동한 사례이므로
**신규 게이트 번호는 매기지 않는다**(doc-steward 판단).

★★신규 방법론 게이트 2건: **#116**(분산성분이 음수면 0 클립은
점추정이 아니라 미검출 — 산출물은 F-검정 95% 상한이며 `F<F₀.₀₅`
셀은 상한 자체가 퇴화한다) · **#117**(표본별 산포가 잡음처럼
보여도 처치-상관 공변량[부하]일 수 있다 — 회귀로 설명력 확인
후 검정력 입력에서 분리하라, `RESULT_RULEPOWER` 계열 검정력
재계산에 직접 적용). 대응 `CONSENSUS §3` 항목136·137(신설).
★기존 불변 배너 전부 승계(HE0·정책 순위·gate #13/#16 "닫았다"
금지·switch-cost "닫았다" 금지·C2 인용정지(a)(b)·`CONSENSUS
§1-24`·job 893663 일반화 금지·게이트#14 "닫았다" 금지). GPU
지출 이번(2026-09-09~10) **2.850 GPU-h**(Step1 0.244+P7 2.606),
트랙 누적 **≈4.78 GPU-h**. 정본 반영: `CONSENSUS.md` rev58→
**rev59**, "다음 실험 gate" longctx_conflict 행 追記, "방법론
게이트" #116–117 신설. 상세 `workspace/engine-port/results/
longctx_conflict/{RESULT_S1_ARMFIX_2026-09-09.md,
RESULT_P7_ITL_SIGMA_2026-09-10.md, PREREG_P7_ITL_SIGMA_
2026-09-09.md, probes/p7_905994/P7_RESULT.json,
probes/s1_905958/S1_LABEL.json}`.

이전: 2026-09-09 **(2차 세션, 같은 날 연속, doc-steward)** —
`longctx_conflict` 트랙 GPU 계측 프로브 1건(프로브 C `CAPACITY`,
job 905835, **1.11 GPU-h**, 12/12셀·`UNRESOLVED` 0) + 규칙층
사전등록 **3판본**(`PREREG_SWEEP` rev1·rev2·`PREREG_P6_ITL_ORDER`)
**전부 `NO-GO`**(트랙 계열 누적 **17연속**) + ★★**stake #1 구조
판정 등재**(이 기판의 공통-`W` 설계로는 답할 수 없다 — **스코프
선언, 반증 아님**) + ★★**선행연구 사실 확인**(저장소 내
`muxwise/{sharegpt.yml,loogle.yml}`가 이미 워크로드별 SM 분할표를
출하 중) + 계측 자산 2건 승계(`L_decode_exact`·`span_hat`) + 정정
3건 + 신규 방법론 게이트 후보 11건 **전부 등재**(#104–114) +
★★**게이트 #115 신설**(전사 누락 자체를 교훈으로 등재) +
★사실 정정 2건(GPU 누적치·"17연속"의 올바른 독해). ★새 성능
판정 0건 · arm 순위 0건 · 정책 순위 변경 0건 · 등급 변경 0건 ·
**HE0 불변.** ★**정정(같은 세션, doc-steward 자기 정정)**:
이 배너는 최초 **#104–106·#111–114 7건만 등재**(#107–110은
`audit_sweep_rev2_2026-09-09/VERDICT.md`로 판정서를 옮기며
그 절이 전사 누락돼 어느 커밋 문서에도 정의문이 없었음)로
작성됐으나, 원인 확인 후 판정서 원문에서 해당 절을 복원해
**#107–110도 등재**했다(아래 (J)·게이트 #115).

**(A) 프로브 C `CAPACITY`(job 905835, 1.11 GPU-h, 12/12셀,
`UNRESOLVED` 0건)**: `PREREG_CAPACITY_2026-09-09.md`(⚠️미감사
계측 프로브 — 아래 §5 caveat 준수). R1 `KNEE_BRACKETED` d16/d44/
d92 **전부 통과**(`ach/off`가 rate 상승에 따라 0.95↑→0.90↓로
하강: d16 1.170→1.080→0.765, d44 1.142→1.088→0.785, d92
1.189→1.116→0.775). **R2 `P_C1`(out=96 max `L_decode`<3.0)
`REFUTED`**(max=**3.974**, `d16_r2_o96`) — *"`R`이 decode 인구
천장을 정한다"*는 이로써 반증됐다. **R3 `P_C2`(out=384 max≥3.0)
`SUPPORTED`**(max=**22.826**, `d16_r1_o384`). R4 `MODEL_HOLDS`는
수치상 통과(obs/pred 0.981/1.007/0.988)했으나 **대수적 항등식**
으로 판명(`obs/pred ≡ (out−1)/out × itl_mean/itl_p50`, `floor_ref`·
`itl_solo`가 전개상 소거) — **모형 검증이 아니다.** R5 `MU_P_D16`
**MEASURED 0.9331 req/s**(사전 2점외삽 0.94, 오차 0.7%). 실측
`μ_p(D)`(0.187/0.675/0.933, D=92/44/16)는 prefill SM에 사실상
선형(108 SM 외삽 1.101 vs 무경합 실측 1.098, 0.3% 일치). ★**후보
교락(미감사, 감사 회부 대상)**: 각 arm을 **자기 포화점**에서 재도
`L_decode`가 decode SM을 **덜** 받은 arm에서 **더 크다**(out=96:
0.266→3.974=**14.9×**, out=384: 0.963→22.826=**23.7×**) — decode
SM을 줄이면 남는 108 SM 예산이 prefill로 가 `μ_p`가 올라 decode
**도착(수요)**이 늘고, 동시에 decode ITL(체류시간)도 느려져
**공급↓·수요↑가 같은 방향으로 겹친다.** `R`을 낮춰도 완화되지
않고 **커진다.**

**(B) ★★정정 배너 — `RESULT_CAPACITY_PRECOMP_2026-09-09.md`
§3·§4.1·§4.2 철회**(프로브 C 실행 직후, 같은 날 자기 철회): §3의
"자유 파라미터 0개의 예측"은 예측이 아니라 Little 법칙의
재진술(R4와 동일한 항등식)이었고, §4.1·§4.2의 *"`R∈[0.30,0.45]`가
decode 인구를 1.22–2.33으로 못 박는다"*는 위 R2 `P_C1_REFUTED`로
**직접 반증**됐다. §1(정확 추정량 정의)·§2(실측)·§4.3(교락
후보)·§4.4(미측정 목록)는 유효 유지.

**(C) `PREREG_SWEEP`(rev1) — `NO-GO`**(`audit_sweep_rules_
2026-09-09/VERDICT.md`, 死因5[W1–W5]·차단13, 트랙 계열 누적
**15연속**): `(R,W)×D` 2요인, `ctx` 고정, SLO goodput(`T_slo=
9.1s=10×floor_ref`·`I_slo=60ms`). 死因 **W1**(`ARGMAX_MOVES`
결정 규칙이 이미 측정된 ITL95 표의 함수 — 게이트 #9의 15번째
재발) · **W2**(절벽 회피 절차가 답 선택 절차와 동치, 실제 채점
셀에서 §2.1 자신의 기준 위반) · **W3**(등록 `n=6`이 부팅이 아니라
시드 — 인용 출처가 정반대로 명시) · **W4**(死因 3F1이 축 이름만
`ctx→out`으로 바뀌어 안 닫힘) · **W5**(등록 규칙 4개 중 3개가
항등식·자동통과·GPU-0 결정가능). ★**부록 A(메인 세션 독립
재검증)**: 감사 헤드라인 2건(**95.93%**·`[47.0,86.9)`)은
**미재현**(82.10%·다른 구간, 동률 처리 자유도 탓) — **인용
금지**. 그러나 등록 운영점 `(9.1s,60ms)`에서 1·2위 goodput 차
**0.5%p**(TTFT p50 8.93 vs 8.90s)가 직접 확인돼 **`NO-GO`는
오히려 강화** — TTFT는 goodput **크기**엔 절벽이 아니나
**`argmax`엔 칼날**.

**(D) `PREREG_SWEEP_REV2` + `PREREG_P5_ITL_LOCATE` — `NO-GO`**
(`audit_sweep_rev2_2026-09-09/VERDICT.md`, 死因6[X1–X6]·차단16,
트랙 계열 누적 **16연속**): 단일 문턱 `I_slo` 대신 격자 사각형
`[1,70]s×[10,100]ms` 지도를 산출로 등록했으나 死因 **X1**(`T1
REDUCIBLE`이 §0 두 단조성 전제의 연역 — 공통 `W` 반사실
`A=0.966–0.996`, "측정"이 아니라 "전제 재확인") · **X2**(격자
4모서리가 새 문턱, 같은 자료에서 `A` 0.359↔0.846) · **X3**
(de-alias 불성립: arm 산포 2.9–23.7× > 중첩대역 폭 1.43×;
`T4`가 후처치 변수 위의 게이트라 ITT 선언 뒷문 위반) · **X4**
(`n_boot` 위임이 순환은 아니나 무력) · **X5**(같은 격자점에서
`N`이 arm간 3.8× 상이) · **X6**(`N` 제약의 근거였던 RNG 모형
불일치가 **존재하지 않음** — `randint(L,L+1)`이 MT19937 상태를
소비하지 않음을 확인 ⇒ **`N∈{45,84,170}` 제약 철회**). ★**계측
자산 확정**: `span_hat = duration − (ttfts[-1] + Σ itls[-1])`
(도착 스팬 복원)이 12셀 전수 평균 |오차| **0.0767%**(최대
0.1166%)로 검증 — "기존 하네스로 도착 스팬 관측 불가"라던 진단은
**철회**.

**(E) `PREREG_P6_ITL_ORDER` — `NO-GO`**(`audit_p6_rules_
2026-09-09/VERDICT.md`, 死因7[Y1–Y7]·차단13, 트랙 계열 누적
**17연속**): 사전등록 예측 E1(`itl95_cell` arm 순서가 두 `R`에서
동일)·E2(공통 `W`에서 TTFT 확률지배)를 2×2 결과표로 실행 전
고정. 死因 **Y1**(E1·E2가 rev2 §0의 전제 그 자체 — **메인 세션
독립 재현 E1 16/16 만장일치**[`d92<d44<d16`, 4 rate-rung×4
추정량]) · **Y2**(E2 등록 정의역 `𝒯`가 **9판본 중 6판본에서
공집합**, 무제한 `T`에서 위반 0.000% ⇒ E2의 실제 출력공간
={`UNDEFINED`,`TRUE`(예정)}, `거짓`은 도달 불가) · **Y3**(`𝒯`
자기정의가 처치 강도의 **감소함수** — `|𝒯|` 최대가 처치 null인
`D=44`) · **Y4**(rev2가 처방한 "공통 `W`"가 이 arm 집합의 `μ_p`
확산 4.99×에서 자기 제약과 양립 불가 — **처방자[rev2]도 함께
죽는다**) · **Y5**(맞춘 것이 도착 창이 아니라 bench 길이, 실제
도착 창 3.2× 상이) · **Y6**(`N`이 판정에 없다는 주장이 특별변론
— `F_92` 자신이 `N`의 함수, 3.2×) · **Y7**(2×2 밖 라벨 4개,
모달 예상 출력이 `E2_UNDEFINED`). ★**부록 C(메인 세션 독립
재검증)**: 이번엔 감사 헤드라인 **2건 전부 재현**(E1 16/16, E2
`𝒯=∅` 6/9판본 + 무제한 `T` 위반 0.000%) — 인용 금지 없음(단
아래 신규 배너 3건 병기 필수).

**(F) ★★stake #1 구조 판정(스코프 선언 — 반증 아님, 등재)**:
*"최적 static split 위치가 워크로드 모양에 따라 움직이는가"*
(`CONSENSUS §5-6` stake #1)는 **이 기판의 공통-`W` 설계로는 답할
수 없다.** `prefill SM + decode SM = 108`이 엔진 강제 제약이라
`μ_p(D)`와 `itl(·,D)`를 분리할 수 없고, 두 단조성(`μ_p`가 prefill
SM에 증가·ITL이 decode SM에 감소)이 **상쇄가 아니라 보강**하므로
순서 역전이 **구조적으로 도달 불가**하다(교훈 항목88: 엔진이
강제하는 상호배타 지원영역은 설계로 못 넘는 교락). 독립 근거
2건: `audit_p6_rules` Y1-(f) · venue-strategist 재평가("조건부
결정적" — 답이 두 실측 단조성 + 결합 지시함수 술어 + 실질 2점
격자에서 연역됨). ⚠️**"장문에서 최적점이 안 움직인다"로 쓰면
즉시 게이트 위반이다** — "반증"이 아니라 **"이 기판에서 구매
불가"**로만 서술할 것.

**(G) ★★선행연구 사실(확인, 새 판정 아님)**: 저장소 내
`workspace/engine-port/external/muxwise/{sharegpt.yml,
loogle.yml}`가 **같은 하드웨어·같은 엔진에서 워크로드별로 다른
SM 분할표를 이미 출하**한다(메인 세션 직접 확인) — ShareGPT
`sm_group_num:8`·`decode_bs=1`에서 decode **20 SM**, LooGLE
`sm_group_num:5`·같은 조건에서 decode **52 SM**. 필드
`decode_bs_threshold`("minimum decode batch size at which this
group should be selected")가 **분할점이 decode 배치에 따라
이동함을 자료구조로 인코딩**한다. ⇒ stake #1의 긍정 답은 최근접
선행이 **이미 전제·배포 중**이다.

★★**정정 追記(2026-09-10, 2차 세션, doc-steward, 두 yml 직접
재확인)**: 위 "같은 하드웨어"는 **틀렸다**. `sharegpt.yml`
(`sm_group_num:8`, 6행 `[112,20,1][104,28,5][96,36,10][80,52,15]
[64,68,20][56,76,25]`)과 `loogle.yml`(`sm_group_num:5`, 3행
`[80,52,1][64,68,5][56,76,10]`) **전 행이 `prefill_sm+decode_sm
=132`**다 — H100/H200급 132-SM 다이 대상이며 **우리 A100 108-SM
기판이 아니다.** 승계되는 것은 **구조적 사실뿐**이다: 최근접
선행이 워크로드별로 다른 SM 분할표를 출하하고
`decode_bs_threshold` 필드로 "분할점이 decode 배치에 따라
이동"을 자료구조에 인코딩한다는 것 — **SM 수치(decode 20 vs
52)는 132-SM 다이의 값이라 우리 108-SM 격자(`D∈{16,44,54,92}`)와
비교·이식 금지.** ★★**게이트 #14/#18 사례(신규 번호 없음)**: 이
표(loogle 3행 포함)는 **2026-08-28**
`reports/impl_vs_external_pdmux_2026-08-28.md` §2.5–2.6에 **이미
문서화**돼 있었다 — 정본이 12일간 흡수하지 않다가 2026-09-09에
"선행연구 발견"으로 재기록됐다(`CONSENSUS.md` §3 항목27 追記
참조, "방법론 게이트" 항목14 追記도 함께). 상세는
`CONSENSUS.md` §5-6 追記(2026-09-10)와 동일, `reports/paper/
venue_positioning.md` §5(2026-09-10 1차 세션, 이미 이 caveat
반영)·`CLAIM_EVIDENCE_MATRIX.md:264`도 동일 정정 반영.

**(H) 계측 자산 2건(승계 권장, 감사 3회가 인정)**: **`L_decode_
exact` = Σᵢ Σ(itlsᵢ)/duration**(Little 적분형, 클라이언트측 —
telemetry 스냅샷·도착 timestamp 미사용 ⇒ 구간 부과 아티팩트
[게이트 #103]에 **구조적 면역**, 12셀 교차검증 상대오차 최대
**1.845e-06**) · **`span_hat` = duration − (ttfts[-1] + Σ itls
[-1])**(도착 스팬 복원, 12셀 전수 평균 |오차| **0.0767%**).

**(I) 정정 3건**: ① `RESULT_CAPACITY_PRECOMP §3·§4.1·§4.2` 철회
(위 B) ② `audit_sweep_rules` W5-b의 "0.5% 일치" 검증 **무효**
(계산-대-계산, 자기저작 픽스처)이나 **결론은 별개 논거로 생존**
(시드가 arm간 공유돼 `LOAD_REALIZED`가 arm별로 다르게 실패할 수
없음 = 처치 검출력 0) ③ 프로브 C `REPORT.md §5-B`의 "0.5% 이내"
검증은 **저부하 셀 한정 우연**(12셀 평균 |오차| 13.91%)이나
**결론은 `span_hat`으로 오히려 강화**(스팬 계수 0.8259–0.8715가
관측으로 확인).

**(J) 신규 방법론 게이트 — 후보 11건 전부 등재(#104–114) +
게이트 #115 신설**: 최초 점검 시 **#107–110**의 정의문이 어느
커밋 문서에도 없었다(`audit_p6_rules` C5·P6 §5가 존재하는
것처럼 인용만 함) — 원인은 claims-auditor가 반환한
`PREREG_SWEEP_REV2` 감사 판정서 원문의 "신규 방법론 게이트
후보 4건(#107–110)" 절이 메인 세션이 그 판정서를
`audit_sweep_rev2_2026-09-09/VERDICT.md`로 전사하는 과정에서
**통째로 누락**된 것이었다(파일 자체는 정상, 다른 절은 온전).
doc-steward의 등재 절차(인용-정의 상호참조 대조)가 이 누락을
잡아냈고, 판정서 원문에서 해당 절을 복원해(파일에 복원 경위
배너 포함) **#107–110도 등재**했다 — 이제 후보 11건 **전부
등재**(#104–114). ★★이 전사 누락 자체를 **게이트 #115**로
신설(신규 등재 절차가 발견한 자기 결함, 게이트 #14와 구분[★오인용
정정 2026-09-10, doc-steward: 원문은 "게이트 #18"이라 썼으나
오인용이다 — 정본 번호는 게이트#14/`CONSENSUS.md` §3 항목27뿐.
상세 `CONSENSUS.md` §3 항목27 追記] — #14는 *존재하는 진단의
전파 실패*, #115는 *최초 기록 단계의 데이터 손실*).

**(K) ★사실 정정 2건**(작업 지시문 서술 정정, doc-steward
확인): ① 이 트랙 누적 GPU는 **≈1.93 GPU-h**(1차 세션 프로브
5건 0.818 + 이번 프로브 C 1.11) — "프로브 C가 트랙 유일 GPU
실측"은 오기. ② **"17연속 `NO-GO`"는 이 질문(stake #1)에 대한
17회 시도가 아니다** — `longctx_conflict` 트랙 자체의 판정서는
**8건**(L2 rev1·rev2, L1 rev3, RATIO rev4, dutycycle 결과감사,
SWEEP rev1·rev2, P6)뿐이고 나머지 9건은 `cp_baseline`/D1 트랙에서
상속된 계보 카운트다. ⇒ 진단상 중요한 것은 **서로 다른 두 질문이
같은 死因 서명(게이트 #9류 항등식 재발)으로 죽었다**는 사실이다.

★기존 불변 배너 전부 승계 + ★★**신규 인용 금지**: `RESULT_
CAPACITY_PRECOMP §3·§4.1·§4.2` · 프로브 C `MODEL_HOLDS`를 "모형
검증"으로 · `f_time`을 `L_decode_exact` 병기 없이 · rev1 감사
95.93%·`[47.0,86.9)` · rev1 W5-b "0.5% 일치" · `PREREG_P5 §3-A`
"0.05%" · rev2 §3/§7-3의 82.1%/95.9% · P6 §1.2의 "±0.8% ⇒
과도상태 분율 동일" · P6 §0의 "구조적으로 재발할 수 없다" · P6
감사 판정서 예보 수치(45.20/22.75/15.16·49.64/23.53/16.55ms)를
§R5–§R6 모형·검증·마진 병기 없이(d92 적합 R²=0.19로 기울기
미식별). HE0·정책 순위·gate #13/#16 "닫았다" 금지·switch-cost
"닫았다" 금지·C2 인용정지 (a)(b)·`CONSENSUS §1-24`·job 893663
일반화 금지·게이트 #14 "닫았다" 금지 전부 불변. 정본 반영:
`CONSENSUS.md` rev57→**rev58**. 상세 `workspace/engine-port/
results/longctx_conflict/{PREREG_CAPACITY_2026-09-09.md,
RESULT_CAPACITY_PRECOMP_2026-09-09.md, probes/c_905835/REPORT.md,
PREREG_SWEEP_2026-09-09.md, audit_sweep_rules_2026-09-09/
VERDICT.md, PREREG_SWEEP_REV2_2026-09-09.md,
PREREG_P5_ITL_LOCATE_2026-09-09.md, audit_sweep_rev2_2026-09-09/
VERDICT.md, PREREG_P6_ITL_ORDER_2026-09-09.md,
audit_p6_rules_2026-09-09/VERDICT.md}`,
`handoff-report/session_handoff_2026-09-09.md`.

이전: 2026-09-09 **(1차 세션, doc-steward)** — `longctx_conflict`
트랙 규칙층 사전등록 **두 판본 추가**(`PREREG_L1` rev3·`PREREG_RATIO`
rev4) **전부 `NO-GO`**(트랙 계열 누적 **14연속**) + 결과 감사 1건
(`RESULT_DUTYCYCLE_G16`)도 **`NO-GO`**((B) 방향만 생존, (C)·(D)
철회) + ★★**이 트랙 최초로 GPU 지출**(계측 프로브 5건, **0.818
GPU-h**, 전부 등록 라벨 산출: `BOOT_OK`·`SPLIT_FIRES`·
`HOMOG_FREE`·계측 4/4 통과·1지지/2**VOID**/3지지) + ★**정본
정정**(`CLAIM_EVIDENCE_MATRIX.md`·`CONSENSUS.md`의 M4R
"G16 28파일 39,849구간 전수 반례 0" biconditional이 **한 방향만
참**임을 확인·정정) + 신규 방법론 게이트 3건(아래 #101–103).
★새 성능 판정 0건 · 정책 순위 변경 0건 · arm 순위 0건 ·
등급 변경 0건 · **HE0 불변.**

**자유 표면 이동 계보(갱신)**: 자 → 추정량 → 격자 `m` →
목표→할당 → 게이트의 표본화율·문턱 → 판정 면의 미정규화 축 →
**처치 듀티사이클 `f`**(신규, 이 세션).

**(A) 규칙층 2판본, 전부 `NO-GO`**: `PREREG_L1_2026-09-08.md`
(rev3, `audit_l1_rules_2026-09-08/VERDICT.md`, 死因4·차단11,
트랙 계열 누적 13연속) — 단일 판정 답: 판정 면 `S_norm`이 TTFT
결합항만 `floor_ref(L)`로 정규화하고 ITL 결합항은 절대값
(30–100ms)으로 남겨, 도착률 규칙 `rate(L)=ρ/floor_ref(L)`이
decode 동시성·요청당 monolithic-prefill 스톨 횟수를 ctx와
**≈15× 공선**으로 묶는다(死因 3F1) — `ARGMAX_MOVES`가 "SM
재배분 최적점이 ctx 때문에 움직였다"와 "부하 체제가 ctx를 따라
움직였다"를 구분 못 함. `PREREG_RATIO_2026-09-08.md`(rev4,
`audit_ratio_rules_2026-09-08/VERDICT.md`, 死因3·차단12, 트랙
계열 누적 **14연속**) — 단일 판정 답: 자유 표면이 **처치가
실제로 걸려 있는 시간 비율 `f`**로 옮겨 앉았다(死因 R1) — 1차
대조 앵커 셀(A)에서 `f_A≈0.004`(등록 문턱 `f>0.20`을 **51배
차이로 자기 기각**), iso-C 대조가 동시에 **100× anti-iso-`f`**.
결과 감사 `RESULT_DUTYCYCLE_G16_2026-09-08.md`도 `NO-GO`
(`audit_dutycycle_2026-09-08/VERDICT.md`, 死因3·차단7) — (B)
방향(듀티사이클이 arm에 단조)만 생존, **(C)(`f∝1/prefill_SM`
정량 일치)·(D)(드리프트 0) 철회**(死因 R3: `f≈ρ_pf` 전제 자체가
같은 파일에서 반증 — g16 28파일 전수 decode-active 구간 39,861
중 `split∧pab>0` 2,951·**`split∧pab==0` 1,643**(split의
35.8%)·`nonsplit∧pab>0` 0, 기전=이벤트 루프 record-skew).

**(B) ★이 트랙 최초 GPU 지출 — 계측 프로브 5건, 0.818 GPU-h**
(`PREREG_PROBES_2026-09-08.md`, 사용자 지적 *"추정으로 재현성
부족을 이유로 실행 안 하는 것처럼 보인다"*에서 착수): P3
`BOOT_SANITY`(job 905701, `BOOT_OK`, 균질화 config 3 arm 전부
부팅, 배너 5행 동일) · P1 `SPLIT_FIRES`(job 905707, d44 f_time
0.9921/d92 0.9982, `split∧pab=0` 0건 ⇒ Nemotron+flashinfer·
동시성>1에서 처치 발화 확인) · P3-C `DIVISION_COST`(job 905710,
Δ`max_total_num_tokens` 0.051% ⇒ `HOMOG_FREE`, 차단 B12 닫힘) ·
P2 `FLOOR`(job 905712, `floor_ref` 0.1237/0.4638/0.9104/1.8216s·
`itl_solo` 12.94–13.11ms·게이트 4/4 통과) · P4 `SKEW`(job
905713, 예측1 지지/예측2 **VOID**[등록 문턱 3×가 기준 0.9722
에서 최대 가능 1.029×이므로 산술적으로 도달 불가 — "철회"가
아니라 무효, `ADJUDICATION_P4_2026-09-09.md`]/예측3 지지[sticky
ON `pab==0` 68배]). ★§4-C 결합 검정(P2 전에 등록): `ρ_pf =
1.0×0.9104=0.910` vs 관측 `f=0.9921`, 비 **1.090**∈[0.85,1.15]
⇒ 지지(⚠️바닥을 108 SM에서 쟀는데 arm은 prefill 64 SM이라 참
`ρ_pf`는 더 크고 포화일 수 있어 판별력 약함). ★**설계를
무너뜨린 실측**: `itl_solo` 실측 **13.0ms**(설계 가정 40ms의
1/3) ⇒ `RATIO` 셀 좌표 재계산 시 `C` 상한 **0.703→0.229**
(死因 R3 3배 악화).

**(C) ★★정본 정정 — `CLAIM_EVIDENCE_MATRIX.md:264`·
`CONSENSUS.md` M4R 항목(§4 registry·§1-25 인접)의
*"`decode_sms==D ⟺ prefill_active_batch_size>0` … G16 28파일
39,849구간 전수(반례 0)"*는 **한 방향만 참**이다.** 메인 세션
전수 재집계(같은 28파일, `audit_dutycycle_2026-09-08/VERDICT.md`
死因 R3·§메인세션 독립 재검증 표): `split∧pab>0` 2,951 ·
**`split∧pab==0` 1,643**(35.8%) · `nonsplit∧pab>0` 0 — **⟸
방향은 반례 0(그대로 유지)**, **⟹ 방향은 1,643건 반례**.
⚠️**코드층 분기 술어와 telemetry층 관측을 구분**한다 — 반증되는
것은 **스냅샷 기록 시점의 동치**(이벤트 루프 record-skew:
`split_prefill_batch = None` 직후 같은 iteration의 마지막 sync가
발화)이지 코드 술어 자체(결정 시점 진리값)가 아니다. **M4R
F1/F1′의 판정(estimand 미식별, `NO-GO`)은 뒤집히지 않는다** —
방향 한정만 추가된다. 정정 반영 위치: 아래 감사 판정서 표(M4R
rev2 행)·"다음 실험 gate" #11 registry(m4r_confinement 행)·
`reports/paper/CLAIM_EVIDENCE_MATRIX.md:264`·`reports/
CONSENSUS.md`(§4 registry m4r_confinement 행).

**(D) 방향(합의) — 다음 회차 설계, 미등록·미감사**: `R≈0.3–0.45`
(prefill·decode 둘 다 바쁜 영역)에서 knee 이상 부하로 1요인
(ctx·out 고정, 경계 횡단 회피) static split 스윕. 근거: P1이
이 영역 근처(ctx 8192)에서 층위 1·5 해소 확인(f≈0.99, arm 간
편차 1.006×, g16의 2.66×와 대조). `n≥6`(`RESULT_RULEPOWER_
2026-09-08.md` 부록 A: 4셀·3대조 n=6 검정력 0.868, n=4는
0.124). rate는 knee 이상 — 자체 용량 선측정 필요.

★**신규 방법론 게이트 3건**(아래 "방법론 게이트" #101–103):
(i) 비율 문턱은 그 양의 천장/바닥을 먼저 계산하라(게이트 #9의
사전등록 판본, 실사례: 기준 0.9722·최대 가능 1.029×·등록 문턱
3×) (ii) 두 편향이 반대 방향으로 표류하면 그 곱이 이론 예측과
일치할 수 있다(실사례 `2.659=3.517×0.757`, 지수 0.98/1.26/1.92)
(iii) 정본이 이미 바뀌었는데 새 설계가 그 정본을 안 읽는 실패
(§1-26(B)가 2026-08-03에 등재한 기전·처방을 2026-09-08 `RATIO`
설계가 미적용, 게이트 #18[★오인용 정정 2026-09-10: 정본=게이트#14/
`CONSENSUS.md` §3 항목27 — 상세 §3 항목27 追記]보다 나쁜 형태).

⚠️**아직 감사받지 않은 주장(정본 아님, 판정처럼 쓰지 말 것)**:
*"§1-20의 `+2%` coupled ceiling 자체가 층위 1 아티팩트일 수
있다"* — claims-auditor 미회부, 미결/후속 항목으로만 등재.

★기존 불변 배너 전부 승계: HE0·정책 순위·gate #13/#16 "닫았다"
금지·switch-cost "닫았다" 금지·C2 인용정지 (a)(b)·`CONSENSUS
§1-24`·W1 SLO-쌍 수치표·크기 인용 금지(대수적 변환 포함)·
"장문에서 PD-mux가 decode를 보호한다" 정본 승격 금지·job 893663
일반화 금지·게이트 #14 "닫았다" 금지·7.24s 인용 금지[CS-OK]. 정본
반영: `CONSENSUS.md` rev56→**rev57**. 상세 `workspace/
engine-port/results/longctx_conflict/`(신규 산출물 전부),
`handoff-report/session_handoff_2026-09-09.md`.

이전: 2026-09-08 **(2차 세션, 같은 날 연속, doc-steward)** —
★★★**게이트 #14 도구 수리를 별도 브랜치 `fix/gate14-tci-analyze`에서
착수(main 미병합)** + ★★**`longctx_conflict` 트랙 신설**(사용자
발의로 long-context 우선순위화) — 규칙층 적대 감사 **3회**(도구
설계 1회·트랙 rev1·rev2 각 1회) **전부 `NO-GO`**. GPU 지출 **0 ·
새 성능 판정 0건 · arm 순위 0건 · 정책 순위 변경 0건 · 등급 변경
0건.**

**(A) 게이트 #14 도구 수리(브랜치, main 미병합)**: `benchmarks/
pdmux_eval/analyze.py`에 t-CI 3함수(`paired_t_ci`/`unpaired_t_ci`/
`t_crit_for`, `cp_baseline/d1_predicates.py`에서 이식) 추가·CLI
판정을 t-CI로 전환·bootstrap은 `companion_bootstrap`으로 병기.
검증(메인 세션 독립 재실행): 행동 보존(5,100키 불일치 0)·회귀
314 tests OK·변이 4/4 killed. 설계서 규칙층 감사 **`NO-GO`**(死因
4·차단 12) — 死因 F1(판정 경로에 n 하한이 없어 **n=2에서
headline=True**, 게이트 #3 하한 *추가* 변이를 이 수리의 새 테스트가
**죽임**)만 수리, F2–F4·차단 12건은 열려 있다. ★**"게이트 #14를
닫았다"는 여전히 금지** — main은 원본 397줄 그대로다(사용자 지시:
"기존 라이브러리 유지, 다른 브랜치로"). `reports/AUDIT_DEBT_
2026-08-23.md` §10.3 P0-4 상태를 **부분 이행**으로 갱신. ★신규
방법론 게이트 후보 3건(아래 "방법론 게이트" #98–100): (i) 판정
경로의 하한 부재는 추정량 교체로 드러나지 않는다(강화 변이를
그 수리가 낸 테스트 자신이 죽인다) (ii) 감사 지적 자체가 과장일
수 있다("venv에 scipy 없음"은 문자 그대로 참, 과장된 것은 함의 —
같은 전제가 7파일 10지점에 상속, 미수리) (iii) 구간 부과 규칙이
순간 상태를 지속 상태로 바꾼다(`compute_decode_realized`가
스냅샷 dt 전체를 bin에 부과, 실측 25.70 s가 표본화율의 함수).

**(B) `longctx_conflict` 트랙 신설**: `CONSENSUS §5-6`의 long-context
전환 항목 재개 시도, 규칙층 rev1(死因8·차단11)·rev2(死因9·차단12)
**둘 다 `NO-GO`**(트랙 계열 누적 **12연속**). ★**Stage 0 잔류
GPU-0 재측정**(정본 함수 `e1_pin_check::compute_decode_realized`
직접 재실행)이 스윕 27셀 realized 0.6986–0.9994(≥0.95 20/27,
파티션은 발화했다)와 D108 9셀 realized≤0.0437(히스토그램 16 SM
94–100%, `D16≡D108`의 원인은 **"둘 다 16"**)을 확인 — §1-21/C1과
**같은 방향**. ★★**`bin0` 판별**: 이 세션이 등록했던 신규 confound
`C-R`(잔류 결손이 처치·대조 축에 정렬)을 **같은 세션이 스스로
강등** — 지배 성분은 기전이 아니라 **구간 부과 아티팩트**(순간
상태 64개에 표본 간격 0.406 s씩 부과 = 25.70 s, `prefill_active`
0/64). ★**감사 2F9(신규 등재)**: decode-only+prefill 유휴+sticky는
PD-mux 운영점이 아니다(성분 측정, `[[scale-8b-sm-sensitivity]]`와
같은 형태 — 레버 존재만, HE0 안 되살림). **방향 판정**: (ㄴ) **L1
직행(ITT)** 채택(§5-6 stake #1 "최적 static 위치가 움직이는가"에
직접 답, 실현 게이트 불필요라 `bin0`/표본화율이 판정 경로 밖으로
빠짐) — ⚠️**ITT 재프레이밍은 미감사**이고 *"HE0가 ITT다"*는
**메인 세션의 독해이지 정본 자기 규정이 아니다.** (ㄱ) 축소
L−2는 보류(기각 사유는 비용 아닌 감사 2F9 — "비용 때문"이라던
이전 진술은 철회, 실측 부하는 10–40 GPU-h). **L−2는 여전히 "게이트
미실행"**(2026-07-28 C1 CONFIRMED 이래 상태 불변). 신규 금지문
7건(아래 "방법론 게이트" #14 追記·"다음 실험 gate" #7·#11 참조).

★기존 불변 배너 전부 승계: HE0·정책 순위·gate #13/#16 "닫았다"
금지·switch-cost "닫았다" 금지·C2 인용정지 (a)(b)·`CONSENSUS
§1-24`·W1 크기 인용 금지(대수적 변환 포함)·"장문에서 PD-mux가
decode를 보호한다" 정본 승격 금지·job 893663 일반화 금지. 정본
반영: `CONSENSUS.md` rev55→**rev56**. 상세 `workspace/engine-port/
results/tooling_gate14/`(브랜치 `fix/gate14-tci-analyze`에만 존재,
main 미병합), `workspace/engine-port/results/longctx_conflict/`
(main), `handoff-report/session_handoff_2026-09-08.md` "2차 세션" 절.

이전: 2026-09-08 **(1차 세션, doc-steward)** — ★★★**cp_baseline/D1 트랙
규칙층 3판본(rev1→rev2→rev3) + 적대 감사 3회, 전부 `NO-GO`**
(트랙 누적 **10연속**) + ★★**게이트 #14 재발 기전 확정**(도구가
게이트와 정반대를 안내) — GPU 지출 **0**.** D4(정본 1건 부분
정정, 아래) 이행 뒤 D1("운영점을 사지 말고 SLO 축을 sweep한다")의
규칙층을 GPU 0으로 설계해 적대 감사에 부쳤다. 각 회차 단일 판정
답이 자유 표면의 이동 경로를 그대로 이름 붙였다 — **자(yardstick,
rev1 死因 U1–U8) → 추정량(rev2 死因 V1–V10) → 격자 크기 `m`
(rev3 死因 W1–W10)**: `call_sign`이 부트스트랩을 못 쓰게는 막았으나
그 부호를 나르는 판정 임계값 `t_crit`이 이름 없는 호출자 인자이고,
유일한 등록 호출부는 보정 없는 `T_CRIT_UNCORRECTED=2.571`을
넘긴다 — 보정 강도를 정하는 `m`은 등록되지 않은 실현량(실측
t 3.810[m=4]↔8.592[m=142]). 판정서: `workspace/engine-port/
results/cp_baseline/{audit_d1_rules_2026-09-07,
audit_d1_rules_2nd_2026-09-07, audit_d1_rules_3rd_2026-09-08}/
VERDICT.md`(2·3회차는 감사 에이전트가 파일 기록 불가 — 메인
세션이 반환문을 기록, §9는 메인 세션 독립 재검증으로 별도 표시).

감사가 요구한 GPU-0 선행 계산 두 건 이행(`RESULT_D1_A1A2_
2026-09-07.md`). **A1(자 편향)**: AF-1 실측 바닥 × 캠페인
토크나이저 버킷 점유로 등록 쌍(`cp2048` vs `d44`)의 차등 편향
= **TTFT −5.07% / ITL p95 +5.97%** — 두 축이 **반대 부호**이고
둘 다 `PRACTICAL_FLOOR`(3%)를 넘어 **단일 자(`plain`)가 등록
예측(TTFT 축 교차)의 형태를 무부하에서 만들 수 있었음**을
확정(감사·메인 세션 양쪽 독립 재현). **A2(검정력)**: `D5`(트랙
축소) **발동 조건 불성립**(이 체제 사다리 양 끝 22–58%는 마진
3%짜리 질문의 가격[n≈20]보다 한 자릿수 배 큼) — 단 부수로
**Stage V(n=2)는 `t(1)=12.706`이라 MDE 70.98%로 통계적으로
빈 게이트**임을 확정(rev2에서 그것은 수리가 아니라 삭제였다).

★★**게이트 #14가 한 달째 도구에서 반대로 안내되고 있음을 확정**
(`FINDING_GATE14_TOOLING_2026-09-08.md`). 9회차(rev2) 死因
V2(percentile bootstrap n=4 참-귀무 위양성 ~20%)는 신규 발견이
아니라 `PROJECT_STATUS.md` 게이트 **#14**(2026-08-06, `CONSENSUS
§3` 항목27)의 **네 번째 재발**(항목18 계열[★오인용 정정 2026-09-10:
"항목18"이 아니라 §3 항목27 자신의 계열이다]) — 정본 coverage(n=4
.798/n=5 .840/n=6 .859/n=8 .888)를 이 세션이 독립 재현(.802/
.838/.860/.882, 같은 draw t-CI .949–.954). ★**재발 기전**:
`benchmarks/pdmux_eval/analyze.py`가 ① t-CI 함수를 **아예 제공
하지 않고** ② docstring `:153`이 오히려 `paired_bootstrap_ci`를
**권장**(n≤8 제한 언급 0) ③ CLI 경로 `:379`가 **n 가드 없이**
호출 — **게이트는 문서에 있고 도구는 정반대를 말한다.**
`reports/AUDIT_DEBT_2026-08-23.md` §10에 도구 부채로 등재, 게이트
#14 본문에 追記(`PROJECT_STATUS.md`·`CONSENSUS §3` 항목27 양쪽).
⚠️**도구는 고치지 않았다** — 여러 트랙이 공유하고 파급이 트랙
밖이라 **사용자 승인 필요**. ⚠️**스코프 미확인(열린 질문)**:
다른 트랙이 인용한 CI가 실제로 어느 추정량에서 나왔는지 이
세션은 **조사하지 않았다**.

★**메인 세션 자기 정정 1건**(게이트 #80 사례, 3회차 死因 W5):
2026-09-07에 "감사의 `len >= cps` 진단은 틀렸다(79.73%)"라고
보고한 것 **자체가 틀렸다** — `(정본 규약, >=)`와 `(기본 규약,
>)`가 **둘 다 79.6748%**로 관측 동치인데, `>=`를 **기본 규약
에서** 재서 규약과 부등호를 동시에 바꾼 뒤 부등호 가설을
기각했다(confound #10의 진단 판본, ★메인 세션 자신이 위반).
거짓 진술이 `d1_predicates.py` 주석(게이트 #80이 "가장 위험"
이라 지목한 자리)에 하루 있었고 2026-09-08 철회·교체.
★**등록 상수 5개는 전부 옳다 — 철회된 것은 원인 진술뿐.**

★신규 방법론 게이트 5건(#93–97 = `CONSENSUS §3` 항목113–117,
전부 이 세션이 직접 측정·확인, 상세 아래 "방법론 게이트"): **#93**
다중비교 보정은 강도 `m`을 함께 등록해야 완결된다(실측 t
3.810[m=4]↔8.592[m=142], 격자를 좁힐수록 검정력이 오르는
역인센티브) · **#94** 기각과 동등성은 반대 방향의 보수성을
요구한다(한 임계값이 둘 다 정하면 한쪽은 틀린다, 보정 기준
동등성 필요 n=10/50/78/267) · **#95** 두 설명이 관측 동치인데
한쪽을 기각하려면 변수 하나만 바꿔야 한다(★메인 세션 자신의
위반이 실측 사례) · **#96** 판별식을 자기 저작 픽스처로 대체
하면 양성대조가 된다(원 대상 문서가 디스크에 있으면 반드시
거기 적용, 3회차 감사 실측 4/5) · **#97** 게이트가 정본에
등재돼도 도구가 반대로 안내하면 재발한다(항목18[=`CONSENSUS.md`
§3 항목27, ★오인용 정정 2026-09-10]의 강화형,
게이트 #14의 네 번째 발화 — 규율은 도구에 심어야 전파된다).

★D4(정본 1건 부분 정정, `reports/longcontext_trace_plan.md`
§4.3) — *"연속 길이-정규화 임계 `a+b·L`은 계보 없음"* 중
**"계보 없음"이라는 사실 주장만** 정정(Etalon/Metron이 `D_p(L)`
profiling fitting을 명시 처방[`[venue 미확인]`], LoongServe가
input-normalized latency 사용 — Orca/vLLM의 "normalized latency"
는 출력 정규화라 다른 축, 경고 유지). 선형 폐형식 기각·§4.3의
실무 처방(절대 class SLO + slowdown 병기, floor 실측 필수)은
유효 유지 — 철회 아님, 삭제 0건, 인용·계보 정정뿐. 전파 grep —
상위 정본이 이 문구를 인용한 적 없어 `CONSENSUS §3` 등재 불요.

★★**원래 질문(chunked prefill vs PD-mux 정책 비교)은 이 세션에서도
0건 측정** — 트랙 누적 **10연속** `NO-GO`. **GPU 지출 0 · 새
성능 판정 0건 · arm 순위 0건 · 정책 순위 변경 0건 · 등급 변경
0건.** HE0·정책 순위·gate #13/#16 "닫았다" 금지·switch-cost
"닫았다" 금지·C2 인용정지 (a)(b)·`CONSENSUS §1-24`·**"TTFT
6000/ITL 60에서 d44가 2.4배" 인용 금지**(대수적 변환 포함)·
*"장문에서 PD-mux가 decode를 보호한다"* 정본 승격 금지 전부
불변. 정본 반영: `CONSENSUS.md` rev54→**rev55**. 상세
`workspace/engine-port/results/cp_baseline/{PREREG_D1_REV3_
2026-09-08.md, RESULT_D1_A1A2_2026-09-07.md,
FINDING_GATE14_TOOLING_2026-09-08.md}`,
`handoff-report/session_handoff_2026-09-08.md`.**

이전: 2026-09-07 (doc-steward — ★★★**cp_baseline 트랙 AF-1
완주 = 이 트랙 첫 실질 라벨 + 경로 (a) 폐쇄 사유 확정** — 규칙층
rev6이 이 트랙 최초 `GO`를 받은 뒤 하네스·스모크 2회(계측 결함
2건 수리)를 거쳐 6 boot(0.88 GPU-hr, 11 job)로 등록 분석기를
**1회** 실행, **`STRATUM_DEPENDENT` → 분기 `b`**를 냈다(사다리만
`intact`↔`pruned`, TTFT 축 단 한 행이 실체 — 분기 `b`의 등록
처방인 사다리 전수 공표에서 ITL 축은 어느 조합에서도 못 떨어짐).
등록 예측 5건 중 3건이 틀림(§5.1이 "틀리는 것이 결과"로 등록해
산출물). 이어 사용자 지시로 "층을 고정할 근거"를 제안했으나
규칙층 감사 `NO-GO`(死因 7건, 단일질문답 "이름만 바꿨다")를
받아 수용했고, 1차 출처 재조회 + 관련연구 조사(392줄)로
**경로 (a)("외부 앵커 좌표 하나를 운영점으로 등록")가 닫힌
이유는 우리 측정이 아니라 그런 좌표가 존재하지 않기 때문**임을
확정했다(DistServe OSDI'24 자신이 *"there exists no available
SLO settings"* 라 적고, long-context에 인터랙티브 100–400ms를
적용한 논문은 문헌 조사에서 0건). ★신규 방법론 게이트 3건: **#90**
금지문 목록 자신이 거짓 진술을 담을 수 있다(死因 U4, 게이트
#75=항목95의 금지문 판본) · **#91** 모집단 선별이 답을 정할 수
있다 — 분위수는 모집단을 명명해야 뜻을 갖는다(死因 U2, AF-1
풀=코퍼스 상위 2.12%라 `anchor=unique`는 사슬이 아니라 풀 선택이
정함[코퍼스 p99에선 `multiple`], 게이트 #81=항목101의 모집단
판본) · **#92** 외부 앵커를 쓰는 트랙은 1차 출처 스냅샷을 남겨야
한다(`serving_slo_survey.md`가 URL만 남겨 재조회 필요, 재조회가
서베이 §2 인용보다 많이 찾아냈고 DistServe SLO-scale 오인용
1건도 함께 드러남). ★**정정**: `serving_slo_survey.md` §2의
"DistServe SLO scale = 무경쟁 단일 요청 실행 지연의 배수"
귀속이 1차 출처(arXiv 2401.09670v3)와 불일치(본문은 **Table 1
절대 SLO의 선형 배수**) — 무경쟁 지연을 분모로 쓰는 것은
Splitwise·Mooncake·LoongServe. 서베이 문서에 정정 배너 부착
(`CONSENSUS.md` 본문은 이 구체 문구를 인용한 적이 없어 grep으로
확인 후 별도 수정 불요). ★overclaim 금지 재확인 — 이 세션
**새 성능 판정 0건**: `STRATUM_DEPENDENT`는 계측·설계 판정이지
정책 판정이 아니고, `d44` 꼬리 SD 4.35배는 관측이지 arm 순위가
아니다(arm이 묶임). HE0·정책 순위·gate #13/#16 "닫았다" 금지·
switch-cost "닫았다" 금지·C2 인용정지 (a)(b)·`CONSENSUS §1-24`
전부 불변. 정본 반영: `CONSENSUS.md` rev53→**rev54**. 상세
`workspace/engine-port/results/cp_baseline/`,
`handoff-report/session_handoff_2026-09-07.md`.**

이전: 2026-09-03 (doc-steward — ★★★**cp_baseline 트랙 5·6회차
규칙층 감사(누적 6연속, 둘 다 `NO-GO`) + 구조적 원인 첫 측정** —
2026-09-01 저녁 개시 3일 연속 세션. 5회차(CP-2 rev1) 死因 **H1–H8**,
그림자 **81%**; 6회차(CP-2 rev2) 死因 **F1–F8**, 그림자
**88%(트랙 최고)** — 단일 판정 질문 답: *"수리의 절반이 조건
등록이 아니라 조건 삭제였다. 삭제는 감사에 안 잡힌다 — 없는 것은
grep되지 않기 때문."* GPU **18 job 5.78 GPU-hr**(전부 `COMPLETED
0:0`) — **새 성능 판정 0건**. ★★**W1**(Nemotron-Nano-9B-v2,
32/32셀, 1.56 GPU-hr)이 이 트랙이 다섯 번 죽은 **구조적 원인을
처음 측정**: `d44`와 chunked-prefill(cp) 두 가족이 **서로 다른
SLO 다리에 묶여 있어** 등록 사다리 안에서 비교 부호가 뒤집힌다
(TTFT2000/ITL40=cp 우세 ↔ TTFT6000/ITL60=d44 2.4×, **n=1 —
순위 아님, 구조 관측**). **V-probe**(8 job, 2.16 GPU-hr)가 5회차
死因 H1(무처치 분산이 3% 마진을 삼킴)을 확증(job-내 Δ SD
`fused_default` 1.99%/`d44` 6.13%, ⚠️arm×노드 완전 교락이라 두
SD 비교 금지). **ShareGPT 전수 센서스**(92,824행, GPU 0)는
`cps 8192` 초과 프롬프트가 캡 4000 하 **0건**·무제한도 56건뿐임을
확인 — 정본 trace에서 chunked-prefill 처치가 **발화한 적이
없었다**. **correctness gate**(1.20 GPU-hr)는 arm 간 56 비교 중
55 바이트 동일로 통과(동시성은 미검증). **모델 부팅 스모크**
(0.36 GPU-hr)는 게이트 #83의 백엔드 강제가 **`nemotron_h`
계열에만** 걸림을 확인(`granite`·`Falcon-H1`은 양쪽 백엔드에서
돎). **설계 결정(사용자)**: piecewise CUDA graph 전 arm OFF(부수로
F3/G4 교락 닫힘) · W1용 모델을 Nemotron-Nano-9B-v2+flashinfer로
전환(★정본 정책 결과[Zamba2-2.7B/ctx4096]와 직접 연결 단절).
★**도구·장부 결함 2건**: `presubmit.py`가 **read-only가
아님**(실행이 타 트랙 인증서 재작성, 그 부작용으로 **다른 세션의
미커밋 작업이 소실**됐다가 이 세션이 재생성해 복구 — 이후
회차는 이 도구를 돌리지 않음) · **4회차 死因 G8이 정본 장부에서
2026-09-01 등재 시점부터 누락**돼 있었음을 발견해 이 세션이
정정(G2·G3·G7·G8 미해소, G4는 rev4에서 수리됨·미감사로 정정).
★신규 방법론 게이트 2건(#88 삭제는 조건 등록이 아니라 grep에
안 잡힌다·#89 규율 도구도 read-only를 자체검증 안 하면 타 트랙을
조용히 바꿀 수 있다). ★★**원래 질문(chunked prefill vs pdmux
정책 비교)은 이 트랙 전체에서 여전히 0건 측정** — 이 세션은
"왜 0건인가"를 측정으로 답했을 뿐이다. **새 성능 판정 0건 · 등급
변경 0건 · 정책 순위 변경 0건**. HE0·정책 순위·gate #13/#16
"닫았다" 금지·switch-cost "닫았다" 금지·C2 인용정지 (a)(b)·
`CONSENSUS §1-24` 전부 불변. 정본 반영: `CONSENSUS.md`
rev52→**rev53**. 상세 `workspace/engine-port/results/cp_baseline/`,
`handoff-report/session_handoff_2026-09-03.md`.**

이전: 2026-09-01 (doc-steward — ★★★**cp_baseline(chunked-prefill
baseline) 트랙 정본 반영** — 사용자 요청("정책 비교군 정리 +
chunked prefill을 비교 대상으로 추가")에서 출발한 캠페인이
**규칙층 적대 감사 4회 전부 `NO-GO`**(CP-1 rev1 死因9 ·
rev2 死因7[71%가 직전 수리의 그림자] · CP-0 rev1[3회차] 死因6 ·
CP-0 rev3[4회차] 死因8[24건 중 11건=46%가 순수 전파 실패, 게이트
#80 정량 재확인])를 받고 정책 판정 트랙에서 계측·축 검증 트랙으로
내려앉았다. GPU 4 job **1.36 GPU-hr**(899768 0.248 · 900053 0.246 ·
900054 0.139 · 900067 0.729, 전부 `COMPLETED 0:0`) — **새 성능
판정 0건**. **P1 계측**: 1차(899768) `P1_BLOCKED_INSTRUMENT`(원인
= 계측 로직이 아니라 하네스×텔레메트리 상호작용, launcher가
SIGTERM 없이 SIGKILL로 회수) → 3중 수리 후 재실행(900053)
`P1_ACCEPTED` 7/7 PASS. ★**F3 기전 관측 확인**(그전까지 코드
독해뿐): `--chunked-prefill-size`가 prefill 예산과 **piecewise
CUDA graph 캡처 범위를 동시에** 결정 — cps −1에서 캡처 목록
**len 0** 관측(job 900053). ⚠️**스코프: Zamba2-2.7B · 이 트리 ·
arm당 부팅 n=1** — 정본 pdmux arm(전부 cps −1)에 대한 파급은
**미검증 관측일 뿐 판정 아님**, 기존 결론 수정 없음. **G1(용량
축 식별가능성) 프로브**(900067): 등록 술어(threshold 0.80,
격자{2,4,8})는 두 arm 모두 `CAPACITY_BRACKETED`를 내나 **달성
처리량이 rate 8에서도 단조 상승**(fused +12.4%/cp512 +7.0%) ⇒
**이 축·이 격자·이 임계로는 포화가 식별되지 않는다**(계측 축
판정, arm·정책 판정 아님). `band_vs_sd`(plateau 술어) 미등록.
★신규 방법론 게이트 4건(#84–87)+#70·#80 追記 2건(자기검사 오라클
부재[등록 오라클 필요]·판본이력표↔본문 모순·전파실패 정량화
46%·신규 규율 도구 `check_version_sweep.py` S1–S8). ★**날짜
정정**: `results/cp_baseline/` 산출물 다수가 파일명·본문 날짜를
**2026-08-28로 오기**(실제 작성/실행일 2026-09-01, 세션 시작
시점 기존 파일 관례 승계) — 커밋 4건·`presubmit_registry.json`이
경로를 인용해 **개명하지 않고** 각 파일 상단 배너 +
`DATE_CORRECTION_NOTE.md`로 정정. presubmit override 2건(사용자
명시 승인, M4R `SINGLE_LABEL_FORCED`·TC1 `RESTRICTIONS_INERT`
전역 차단과 별개). ★★**원래 질문(chunked prefill vs pdmux 정책
비교)은 이 트랙 전체에서 여전히 0건 측정**. ★★이 세션 자체 GPU
지출은 위 4 job **1.36 GPU-hr**뿐(이미 집행) · 새 성능 판정 0건 ·
등급 변경 0건 · 정책 순위 변경 0건 · HE0·정책 순위·gate #13/#16
"닫았다" 금지·switch-cost "닫았다" 금지·C2 인용정지 (a)(b)·
`CONSENSUS §1-24` 전부 불변. 정본 반영: `CONSENSUS.md`
rev51→**rev52**. 상세 `workspace/engine-port/results/cp_baseline/`,
`handoff-report/session_handoff_2026-09-01.md`.**

이전: 2026-08-28 (같은 날 2차, doc-steward — ★★★`deprecated_v2/
README.md` 초안 **비준 완료**(§7 신설) — 정본 등재 커밋 `1c6bb59`
**이후** 발견분 6건 정본 반영: **(i) 백엔드 강제 교락**(Nemotron-H
계열+`triton`=부팅 거부 / Zamba2+`flashinfer`=스케줄러 사망
2/2[job 896776]) — 두 모델 계열의 동작 가능 백엔드가 **서로소**라
"모델 고정·백엔드만 변경" 셀을 만들 수 없음 **(ii) TC1 모델
전환**(Zamba2-2.7B→Nemotron-Nano-9B-v2, 대조 arm Qwen2.5-7B, 부팅
스모크 jobs 896760/764/767이 배관 확인: ctx 131072·cudagraph
캡처·긴 프롬프트서 컨트롤러 이동 2회) **(iii) TC1 P3 결과**(jobs
896689/896690, `MEASUREMENT_ABSENT` — 등록된 두 뿔보다 나쁜
**"셋째 뿔"**: Qwen2.5-3B는 배선 결함이 아니라 **물리**로 동시성이
발생하지 않음[TTFT p50 44.9ms vs Zamba2 1344.8ms]) **(iv) TC1 rev3
규칙층 감사**(`NO-GO`, 死因 **F8–F12**, 단일 판정질문 답 =
**"네 번째 고리다"** — 옳은 국소 수리 셋이 각각 새 강제를 만듦,
`audit_tc1_rules_rev3_2026-08-28/VERDICT.md`) **(v) 도구 자기
감사**(commit `5d180a6` — `design_reachability.py`가 자기 spec을
`DISCRIMINATING`으로 잘못 통과시켰음이 드러나 `RESTRICTIONS_INERT`·
`RESTRICTION_DROPPED_UNEXPLAINED`·`PRIOR_UNREGISTERED` 신설,
레지스트리 **append-only** 전환, `TOOL_REV 2` — TC1 rev3 spec은
이제 `DISCRIMINATING`이 아니라 **`BLOCKED`**) **(vi) 모델 로스터**
신설(`workspace/engine-port/results/model_roster/
MODEL_ROSTER_2026-08-28.md`). ★★**신규 게이트 #83**(아래 "방법론
게이트" 참조 — 엔진이 강제하는 상호배타 지원영역은 설계 선택으로
못 넘는 교락이다) — ★검증 결과 게이트 **#10의 실제 정본 텍스트는
"여집합 클래스에 음성대조"이지 "변수 동시 변경"이 아니다**(다수
사전등록 문서가 #10을 후자로 오인용해 온 기존 드리프트를 이번에
확인, 수정은 이 문서 범위 밖). ⇒ #83은 #10의 **변종이 아니라
별개**로 등재하며, 가장 가까운 기존 선례는 항목5의
`--max-running-requests` 교차-arm 결합(2026-08-02). `CONSENSUS §3
항목103`과 대응(넷째 체계 = 메모리 topic
`deconfound-measurement-lessons.md` 항목88). ★★追記(2026-09-11
(2), doc-steward — Q-B′ 규칙층 감사 G-θ 사례): QB-11("엔진 강제
합 108 아래 표준화 반사실은 어느 할당에도 대응하지 않는다",
인용 금지)은 이 게이트의 해석 판본이다. 상세
`workspace/engine-port/results/longctx_conflict/audit_qb_rules_
2026-09-11/VERDICT.md` §5.2(QB-11). ★**백엔드 스코프
배너 부착**: `CONSENSUS.md` §1-3(Diff A/B·교차점·`R_policy`) 행 +
아래 "8B decode-SM 민감도 측정 노트"(C2 국소 탄력도, triton
3-arm+flashinfer Hs8 혼합 재확인) — **`--attention-backend
triton`+Zamba2 계열 한정, Nemotron-H arm으로 이전 금지**. ★`knee2d`
계열 하네스 스테일(`results/prefill_knee/knee2d.sbatch:105`가
`ZBPT`를 grep하는데 계측은 `ZBPT2`를 내 `RESULT`가 빈 채
`boot_ok=1`로 "성공"처럼 끝남)을 **수리 대상**으로
`reports/AUDIT_DEBT_2026-08-23.md`에 신규 등재(정본 아님, 작업
목록). ★**조율 미해결 항목으로만 기록**: `design_reachability.py`
경로 이원화(저장소 루트 vs `workspace/engine-port/`)는 동시 세션이
루트 경로로 호출 중이라 **이번에도 이동하지 않음**. ★★이 비준
세션 자체 GPU 지출 **0** · 새 성능 판정 **0건** · 등급 변경 0건 ·
정책 순위 변경 0건 · HE0·정책 순위·gate #13/#16 "닫았다" 금지·
switch-cost "닫았다" 금지·C2 인용정지 (a)(b)·`CONSENSUS §1-24`
전부 불변. ★`deprecated_v2/`는 **정본이 아니다** — 스코프를 명시해
인용을 규율하는 장치일 뿐이며 편입되지 않는다. 정본 반영:
`CONSENSUS.md` rev50→**rev51**. 상세 `deprecated_v2/README.md` §7.**

이전: 2026-08-28 (같은 날 1차, doc-steward — ★★★**TC1(모델귀속)·M4R(confinement)
두 트랙 규칙층 감사 각 2회(rev1·rev2), 전부 `NO-GO`** + GPU 실측
1건(job **896565**, ≈0.20 GPU-hr, `F2_CONFIRMED`) + GPU 0 프로브
5건 + 제출 게이트(`presubmit.py`) 신설로 도구 4종 첫 등록 실행
지점 확보. 이 세션 **새 성능 판정 0건**. GPU 지출 = **0.20
GPU-hr(job 896565)뿐** · HE0·정책 순위·gate #13/#16 "닫았다"
금지·switch-cost "닫았다" 금지·C2 인용정지 (a)(b) 전부 불변.**

★**규칙층 감사 4회, 전부 `NO-GO`**:

| 판본 | 판정서 | 요지 |
|---|---|---|
| TC1 rev1(2026-08-27) | `workspace/engine-port/results/tc1_model_attrib/audit_tc1_rules_2026-08-27/VERDICT.md` | 死因 4·차단 18. F1(sticky+bind 상호배타 ⇒ `MEASUREMENT_ABSENT` 강제)·F2(anchor=argmax면 컨트롤러가 그 자리에 앉아 `NO_FLIP_BOTH_LOSE` 강제)·F3(estimand 3종 중 미지정)·F4(부호는 재척도 불변 방어가 대수적으로 거짓) |
| M4R rev1(2026-08-27) | `workspace/engine-port/results/m4r_confinement/audit_m4r_rules_2026-08-27/VERDICT.md` | 死因 4·차단 16. ★**F1 — 노출변수 `R`이 정본 등재 항등식(`decode_sms≠108 ⟺ prefill_active_batch_size>0`)과 aliased**: G16 confined 스냅샷 100.0%가 decode 44 SM(SM을 맞추면 R이 1.762→1.097로 붕괴, 전 셀 `R_MIN` 미달). F2(정본 §1-26(B) 선행금지·GPU-0 전제 붕괴)·F3(집계 자유가 실제로 세 라벨 전부를 냄)·F4(`CONFINEMENT_CONFIRMED` 도달 불가) |
| TC1 rev2(2026-08-28) | `.../tc1_model_attrib/audit_tc1_rules_rev2_2026-08-28/VERDICT.md` | 死因 3(F5–F7)·차단 12(B19–B30). 단일 질문 답 = **"국소였다"**(rev1 수리는 지목 좌표에서 옳았으나 파급 미재도출 — `visits_argmax`가 `NO_FLIP_BOTH_LOSE`를 `CONTROLLER_DEGENERATE`로 이름만 바꿈, δ 재도출 없이 anchor 원복[F6], §7 세 대조 전부 반증불가[F7]) |
| M4R rev2(2026-08-28) | `.../m4r_confinement/audit_m4r_rules_rev2_2026-08-28/VERDICT.md` | 死因 4(F1′–F4′)·차단 12(B1′–B12′). **`sm_match` 가드가 엔진이 강제하는 항등식**(28파일 39,849구간 반례 0)이고 "SM 매칭"은 구간의 왼쪽 끝점에서만 성립 — 자유 다리는 파티션 복원 지연 구간 자체(65–87%가 앞뒤 모두 108 SM 고립표본), batch 층화 시 108/D 서명 재출현(1.061–1.387). 7셀 중 6셀이 자기 가드에 막히고 유일한 생존 셀은 `RESIDUAL_INCONCLUSIVE`. ★★**정정 追記(2026-09-09, doc-steward, `longctx_conflict` 트랙 `audit_dutycycle_2026-09-08/VERDICT.md` 死因 R3)**: 이 "반례 0"은 **⟸ 방향만 참**이다 — G16 28파일 전수 재집계로 `split∧pab==0`이 **1,643건**(split의 35.8%) 확인, `nonsplit∧pab>0`은 0건 그대로. 반증되는 것은 **스냅샷 기록 시점의 동치**(record-skew)이지 코드 분기 술어 자체가 아니며, F1′의 estimand-미식별 판정(`NO-GO`)은 불변 |

★**GPU 실측 — F2 anchor 확정 프로브 job 896565**(2026-08-28, gpu39, ≈0.20
GPU-hr, `COMPLETED 0:0`, 사전등록 실행 전 등록 → 축자 채점) —
**`F2_CONFIRMED`**(`SW=0`·`gpC=3.232`, `workspace/engine-port/
results/tc1_model_attrib/probes/f2_verdict_896565.md`). ★★**중요
정정(rev2 재감사 B29)**: 서버 로그 실측 `SLO-BIND` **0줄** /
`SLO-FEAS refused` **152줄** ⇒ 컨트롤러는 **죽은 게 아니라 살아서
막혔다**(메모리항목 21의 거울상 — `SW=0`은 그 둘을 구별 못 함,
사전등록에 liveness 절 부재). 판정 자체는 유효하나 **크기 인용
금지**(n=1), 정본 d44-static과 비교 금지.

★**GPU 0 프로브 5건**: (1) TC1 F1(`f1_verdict.json`) ⇒
`REFUSED_AT_INIT` — `PDMUX_STICKY_PARTITION`+`PDMUX_SLO_SCHED`가
`multiplexing_mixin.py:298-303`에서 `RuntimeError`, 코드 사실·성능
판정 아님. (2) B17 인덱스 사상(`B17_ANCHOR_INDEX_MAP.md`) —
`multiplexing_mixin.py:328-333` 유래, idx 1=d16…**4=d44**…5=d54.
다른 `pdmux_*.yml`로 이식 금지. (3) M4R alias 프로브
(`alias_verdict.json`) ⇒ ★**`EXPOSURE_ALIASED`, 28/28 셀,
`P(decode_sms==D|confined)`=1.0000**(시간가중) — §1-26(B)가 문장으로
예측한 실패의 실측 확인. (4) M4R `R_matched` 프로브
(`rmatched_verdict.json`) ⇒ `RMATCHED_VIABLE` 7/7, 드레인 25ms
포화(★rev2 재감사 F4′가 이 "포화"를 반증 — (0,200)ms에 데이터 질량
0, 실제 연산은 16-iteration 묶음 1개 삭제). (5) 설계층 도달가능성
(`REACHABILITY_FINDING_2026-08-28.md`) — 아래 참조.

★★**설계층 도달가능성 — 두 트랙 모두 캠페인 구매 불가**(GPU 0,
`scripts/discipline/design_reachability.py`): M4R rev2 = 실질
라벨 3종 중 **`RESIDUAL_INCONCLUSIVE` 하나만** 낼 수 있음
(`SINGLE_LABEL_FORCED`). TC1 rev2 시나리오 A(argmax=d44, 정본
2회 지지) = **낼 수 있는 실질 라벨 0개**(`NOTHING_PURCHASABLE`)
— 캠페인 50 job 전체의 정보량이 Stage 1 argmax 추첨(d34↔d44 격차
1.5%=δ의 절반) 하나에 걸려 있었다. ★신규 방법론 교훈(아래
"방법론 게이트" #81): **도달가능성 검사가 격자 안에서만 돌면,
설계가 답을 미리 정해 놓아도 통과한다** — 두 트랙에서 독립 발화.

★**제출 게이트 신설**(`workspace/engine-port/scripts/discipline/
presubmit.py`+`presubmit_registry.json`+`PRESUBMIT_CHECKLIST.md`)
— 규율 도구 4종(`check_line_citations`·`check_doc_facts`·
`check_citation_stops`·`design_reachability`)의 **첫 등록된 실행
지점**(감사 B11이 2026-08-26에 지적한 결손 해소). 현재 실행:
`exit=1`(제출 금지) — TC1 rev3 `DISCRIMINATING`(OK) / M4R rev2
`SINGLE_LABEL_FORCED`(BLOCK). ★`design_reachability.py`는
저장소 **루트** `scripts/discipline/`에 있다(기존 도구는
`workspace/engine-port/scripts/discipline/`) — **동시 세션이 그
경로로 호출 중이라 지금 옮기지 않았다**(doc-steward 소관, 다음
세션 조율).

★**인용 규율 발견**: 저장소 번호 체계는 **넷**이다(`CLAUDE.md`
#1–8 / `PROJECT_STATUS` #1–80 / `CONSENSUS §3` #1–100 / 메모리
topic #1–80) — 다수 사전등록이 둘로만 셈. `results/kernel_mech/
stage0ppp/stage0ppp_a0_rule.py:2`의 `(gate #66)`은 **오인용**이다
(PS게이트 #66 = "포크는…", rules-as-code는 **CONSENSUS §3 항목81
= PS게이트 #61**) — ★그 파일은 완주 실험 A0(job 892556)의 동결
규칙 정본이라 **제자리 수정 금지**, 아래 kernel_mech A0 행에
정오표만 追記(파일은 무수정). ★그리고 TC1 rev1의 정정문 자체가
또 틀렸다(항목66이라 적고 `VERIFIED` 딱지까지 — rev2가 정정).
"오인용 3건"은 하계다(PREREG+rule 15파일 ~50건, 감사 확인).

★**TC1 rev3 작성 완료(2026-08-28) — ★감사 대기, 판정 없음.**
`PREREG_TC1_RULES_REV3_2026-08-28.md`·`tc1_rule_rev3.py`.
도달가능성 재계산(`reach_verdict_rev3_A.json`) = 시나리오 A에서
`DISCRIMINATING`(ctrl_H를 `blocked`로 재정의 — job 896565의
152건 refusal을 "퇴화"가 아니라 "시도·봉쇄"로 인정). **rev3에
대해 "규칙층을 통과했다"는 어떤 문장도 쓰지 않는다** — 규칙층
감사 미실행.

★★**불변 재확인**: `CONSENSUS §1-24`는 반증되지 않았다. 두 M4R
감사 모두 *"이 채널로는 그 질문에 답할 수 없다"*고만 말한다.
`R_matched` 1.00–1.16도, batch 층화판 1.06–1.39도 판정이 아니다.

정본 반영: `CONSENSUS.md` rev49→**rev50**(§3 항목101–102 신설 +
항목35·100 追記, §4 living-doc 행 4개 신설) · 아래 "다음 실험
gate" #11 레지스트리에 tc1_model_attrib·m4r_confinement·규율
도구(presubmit) 3행 신설 · "방법론 게이트" #81–82 신설. ★**금지
문장 승계·신설**: (TC1) *"TC1이 규칙층을 통과했다"* ·
*"`PDMUX_STICKY_PARTITION`이 M3의 estimand 미식별을 해소한다"*
(컨트롤러 arm엔 미정의) · *"anchor를 argmax에 두면 SLO 쇼핑이
아니므로 중립적이다"* · *"`visits_argmax`가 F2를 닫았다"*(이름만
바뀜) · *"TC1 rev2가 `INCONCLUSIVE`를 피한다"*(P≈0.66) (M4R)
*"M4R이 규칙층을 통과했다"* · *"M4R은 GPU 0이다"*(★rev1이 이미
금지, rev2가 위반 — B10′) · *"`sm_match` 가드가 aliasing을
차단한다"* · *"`R_matched`≈1이므로 강등 비용은 SM 감소 그
자체다"*(`RESIDUAL_ABSENT` 도달 불가, 7/7 CI가 1 포함) (공통)
*"도달가능성 검사를 통과했으므로 제출할 수 있다"* ·
*"봉쇄는 측정 실패다"*. ★★**불변**: gate #13/#16 "닫았다" 금지
· switch-cost "닫았다" 금지 · HE0 · 정책 순위 · C2 인용정지
(a)(b) 전부 유지. **GPU 지출 = 0.20 GPU-hr(job 896565)뿐 · 새
성능 판정 0건 · 등급 변경 0건 · 정책 순위 변경 0건.** 상세
`workspace/engine-port/results/{tc1_model_attrib,m4r_confinement}/`,
`workspace/engine-port/results/REACHABILITY_FINDING_2026-08-28.md`,
`workspace/engine-port/scripts/discipline/PRESUBMIT_CHECKLIST.md`.

이전: 2026-08-26 (doc-steward — ★★★**kernel_mech A1(엔진 기판)
sticky 재설계(rev2) 규칙층 감사 **3회, 전부 `NO-GO`**(1·3회차
死因 없음·차단만, ★2회차 판정서는 파일로 저장되지 않음) + ★A1
스모크 **job 893663**(GPU ≈0.014 GPU-hr, 채점 판정 0건·세 항목
PASS, ★n=1) + NSL **③②E-B 묶음**이 그 트랙 첫 규칙층 감사를 받아
`NO-GO`(★死因 4건) + 규율 도구 2종 신설(`check_line_citations.py`·
`check_doc_facts.py`) + 신규 방법론 교훈 5건 등재(NSL 감사 §⑥
후보 3건 + 3회차 A1 감사 명명 1건, `CONSENSUS.md` §3 항목96–100).
이 세션 GPU 지출 = **0.014 GPU-hr(job 893663)뿐** · 새 성능 판정
0건 · HE0·정책 순위·기존 인용정지 전부 불변.**

★**kernel_mech A1 sticky 재설계**(`workspace/engine-port/results/
kernel_mech/DESIGN_A1_REV2_STICKY_2026-08-25.md`, rev1 문서에
SUPERSEDED 배너) — §2 다리를 시간창 조인 대신 **부팅 5개 분리**
(B-U/B-S16/B-G/B-S16′[nsys-OFF]/B-D92). 규칙 신설·개정: ★**1차
추정량 규칙을 새 파일로 신설** `a1/a1_primary_rule.py`(3,110,400
세계/정합 166,194 — **A0 규칙 `stage0ppp_a0_rule.py`(`RULE_REV=4`)
는 동결**, 제자리 수정 시 `a0_verdict_892556.json`이 고정한
`rule_sha256=583c4ab0…`가 깨져 완주 실험의 채점 해시가 무효화됨)
· `a1/a1_q3k1_rule.py`(`RULE_REV=3`, 414,720 세계). 절차 3 —
Q3 채널 = **`decode_iterations`**(`decode_step_count`는 저장소
0/1,958,528 스냅샷, 전수 `architecture=legacy`) 확정,
`a1/DECISION_A1_Q3_CHANNEL_2026-08-25.md`.

★**규칙층 감사 3회, 전부 `NO-GO`**:

| 회차 | 판정서 | 핵심 |
|---|---|---|
| 1(2026-08-25) | `audit_a1_rev2_rules_2026-08-25/VERDICT.md` | 死因 A1 미해소(A2는 조건부 소멸) · 런킬러 B1′(A0 규칙 미개정인데 문서는 개정됐다 보고)·B2(게이트 #21이 `launches` 축에서 재개방)·B3(§5.1a 0.99→0.95 완화가 반증된 전제 위 — sticky 부팅 21/21이 ≥0.9956, 잔차 0.1–0.4%뿐, 완화폭의 1/10–1/50)·B4(자기모순 3건) · ★**B5 — "Ha8+sticky 부팅은 선례가 없다"가 거짓**(job **872800**이 이미 그 job이고, rev2 §1이 인용한 `0.0839→1.0000`이 바로 그 파일의 두 줄) |
| 2(2026-08-25) | ★**파일로 저장되지 않음** — 요지는 `DESIGN_A1_REV2_STICKY_2026-08-25.md` §6.8·§7 + 커밋 `3359fa3`/`7b7b007`/`189acfc`에 있음 | P1(B10 primary 미적용)·P2(게이트 #21이 primary에서 재발)·★P5(**인용 도구가 거짓 인용을 인증**)·P7–P12 |
| 3(2026-08-25) | `audit_a1_rev2_rules_3rd_2026-08-25/VERDICT.md` | 런킬러 4건: **R1**(부팅 5개/두 규칙 세계모형은 여전히 3개뿐, B-D92 양성 답에 라벨 없음)·**R2·R3**(상태표·거짓 금지문 3회째 재발)·**R4**(출처 허위 3회째 — ★이번엔 그것을 막으려 만든 새 도구 자신이 인증) · ★★**형태 명명 — "수리는 국소, 주장은 전역"**(각 수리는 지목된 좌표에서 실재하나 파급을 재도출 않아 다음 회차 결함 대부분이 직전 수리의 그림자) + 부수형 "세계모형이 실험설계 성장을 못 따라간다". ★단 "무한 반복 아님"(4회차가 다시 `NO-GO`면 규칙이 아니라 문서 갱신 규율의 실패) |

★**GPU 실측 — A1 스모크 job 893663**(2026-08-26, ≈0.014 GPU-hr,
`workspace/engine-port/results/kernel_mech/a1_smoke/
RESULT_A1_SMOKE_893663_2026-08-26.md`) — **채점 판정 0건**(배관
스모크, 게이트 #25), 세 항목 PASS: **(a)** `E1_DECODE_REALIZED(16)
= 1.0000`(batch-synchronous, `t_decode_active=35.5s`, hist `D16`
만 — ★§5.1a의 모집단 간극을 닫음, 재감사 P11 우려 해소) **(b)**
`decode_iterations` 0→415 **(c)** `PDMUX_GREEN_READOUT` 첫 부팅
실행, sticky division decode 스트림에서 드라이버가 **`smCount=16`**
확인(3회차 감사 R1이 신설한 `GREEN_TARGET_CONFIRMED`의 첫 관측,
`D=16` 한 점뿐 — `D=92`는 미관측). ★★**한정 엄수**: n=1·35.5초.
*"batch-sync에서 realized는 1.0이다"* 일반화 금지 — 사는 것은
*"기전이 존재하는 모집단에서 게이트가 물리적 사실 때문에 발화하지
않는다"* 뿐.

★**NSL ③②E-B 묶음 + 그 트랙 첫 규칙층 감사** — `NO-GO`, ★死因
4건(`workspace/engine-port/results/nsl_lever/
audit_nsl_bundle_2026-08-26/VERDICT.md`). **D1** `WORKLOAD_NOT_
SATURATING`이 실질 성공(등록 rate-3 부팅 4개, ≈0.7 GPU-hr)을
삼킴 · **D2** TOOLLIMIT 유일 술어(`SITES_INCOMPLETE`)에 관측
채널 없음 · **D3** 추정량이 **순서통계량**(`cap` 검사가 매 반복
최상단에서 평가돼 KV는 cap 불발 시만 도달 + `OTHER` 출구 누락 +
부동소수 비대칭 `1.0-0.80=0.19999999999999996`으로 `cap_share=
0.2`와 `0.8`이 다르게 채점, ★그 대칭 검사를 **격자 인공물**[0.20·
0.80이 격자에 없어서 생긴 "단독구속 0"]을 근거로 직접 삭제) ·
**D4** cap 축 제거가 답을 미리 정함. ★**반증 실패(살아남은 것)**:
③ 코드 사실 전부 정확 · ②의 사이트 열거(5개/도달 2개)·
`pp_max_micro_batch_size` 정정이 **주장보다 강함**(실현 cap으로
채워짐, 구조적으로 강건). ★★**§4.1 헤드라인 자체가 거짓으로
확인**: A1 3회차 표는 6 가족인데 NSL이 4개만 베껴 옮겼고 빠진
3개(상태오보·거짓금지문·출처허위)가 바로 **그 회차 런킬러** —
이 묶음 안에도 그대로 있었다.

★**규율 도구 2종 신설**(산문 규율 반복 실패 가족을 기계로) —
`scripts/discipline/check_line_citations.py`(+`line_citations.json`
50건, 단위 테스트 16) 지문 기반 드리프트 검출·정정 인용 제시·
재베이스 거부·bare 인용/`[HIST]`/고아 키 탐지. `scripts/
discipline/check_doc_facts.py`(11 facts/14 occurrences, 첫 실행
7건 적발) 문서 자기-아티팩트 수치를 진리원과 대조. ★두 도구 다
**한계가 즉시 노출됨**: line-citations는 bare `:NNN`·쉼표 목록
미포착 + `--snapshot`이 편집 파일 기준선을 조용히 덮어써 거짓
인증 가능(재베이스 거부+고아 탐지로 부분 수리) · doc-facts는 표
안 수치를 못 읽어 §4.1 헤드라인 오류를 못 잡음. ★두 도구 모두
사전등록 "제출 전 체크리스트"에 아직 등재 안 됨(감사 B11).

★**신규 방법론 교훈 5건 등재**(doc-steward 판단 — NSL 감사 §⑥
후보 3건은 이미 실행된 감사에서 재현 가능한 코드 사실로 도출됐고
3회차 A1 감사의 형태 명명도 그 자신의 사슬 판단표로 뒷받침되므로
등재) — `CONSENSUS.md` §3 항목96–100, 아래 "방법론 게이트" #76–80.
항목35(gate #21)에 NSL D1을 재발로, 항목93(gate #73)에 mirror-check
삭제 재발(교훈 #46+#78의 두 번째 사례)을 각각 짧게 追記.

★**이 세션이 저지르고 정정한 것(8건, 다음 세션 승계)** — (1)
"Ha8+sticky 선례 없다"가 거짓(job 872800이 그 job, 그 파일 수치를
인용하면서 주어를 선례없다 적음) (2) §5.1a 게이트를 5% 느슨하게
열었다가(크기 안 잼) 철회 (3) 금지문 12건이 수리 진행 중 참이 됨
(문서가 자기 코드에 대한 참인 문장을 금지) (4) 인용 도구가 거짓
인증(`--snapshot` 재베이스, 재베이스 거부+고아 탐지로 수리) (5)
NSL §4.1이 A1 6가족 중 4개만 베끼고 그 회차 런킬러 3개를 누락
(6) 게이트 #21이 반대 방향으로 열림(NSL D1, `n_firings=4000`이어도
`saturated="no"`면 버림) (7) 거울 대칭 검사 삭제가 버그를 숨김
(근거 자체가 격자 인공물) (8) 파일 하나를 스크립트 실수로 삭제
후 git으로 즉시 복원.

정본 반영: `CONSENSUS.md` rev48→**rev49**(§3 항목96–100 신설 +
항목35·93 追記, §4 living-doc 행 신설) · 아래 "다음 실험 gate" #11
레지스트리 kernel_mech A1 행·NSL-1 행에 追記 · "방법론 게이트"
#76–80 신설(+ #21·#73 追記). ★**금지 문장 승계·신설**: (kernel_mech)
*"A0 규칙이 A1을 그대로 채점한다"* · *"부팅 분리가 死因 A2를
닫았다"*(무조건형) · *"0.99를 요구하면 게이트가 물리적 사실 때문에
발화한다/0.95가 교정됐다"* · *"Ha8+sticky 부팅은 선례가 없다"* ·
*"스모크가 통과했으므로 A1을 제출할 수 있다"* · *"batch-sync에서
realized는 1.0이다"*(일반형) · *"rev2/rev3가 규칙층을 통과했다"* ·
(NSL) *"cap이 문다/안 문다"* · *"①이 잰 정본 셀이다"*(cap arm
하나뿐) · *"NSL이 admission 축을 쟀다"*. ★★**불변**: gate #13/#16
"닫았다" 금지 · switch-cost "닫았다" 금지 · HE0 · 정책 순위 · C2
인용정지 (a)(b) 전부 유지. **GPU 지출 = 0.014 GPU-hr(job 893663)
뿐 · 새 성능 판정 0건 · 등급 변경 0건 · 정책 순위 변경 0건.** 상세
`handoff-report/session_handoff_2026-08-26.md`, `workspace/
engine-port/results/kernel_mech/{DESIGN_A1_REV2_STICKY_2026-08-25.md,
audit_a1_rev2_rules_2026-08-25/VERDICT.md,
audit_a1_rev2_rules_3rd_2026-08-25/VERDICT.md,
a1_smoke/RESULT_A1_SMOKE_893663_2026-08-26.md}`, `workspace/
engine-port/results/nsl_lever/audit_nsl_bundle_2026-08-26/
VERDICT.md`.

이전: 2026-08-25 (doc-steward — ★★★**kernel_mech Stage 0‴ A0
완주[KSET_CONSTRUCTIBLE, job 892554→892556, 기판 한정] + NSL ①
구간묶기[BRACKET_DECIDES] + NSL E-A[ARITHMETIC_CONFIRMED, job
892561] 메인 세션 등재분 검증·정규화 + kernel_mech A1(엔진 기판)
규칙층 감사 `NO-GO`(★死因 2건, 이 감사 사슬 최초) 신규 등재[메인
세션 미등재분, doc-steward가 채움] + `reports/CONSENSUS.md` 표
무결성 결함 2건 수리[unescaped `\|` 2쌍·orphan 개행 1건, §1 항목1·
항목31]. 이 정규화 세션 GPU 지출 0 · 인용하는 세션 결과 GPU 지출
0.124 GPU-hr(job 892554/892556/892561, 이미 집행) · 새 성능 판정
0건 · HE0·정책 순위 불변.**

★**Stage 0‴ A0 완주**(job **892554→892556**, GPU 0.021 GPU-hr) —
판정 `KSET_CONSTRUCTIBLE`, basis `ctx+stream`. ★★**기판 한정 필수**
(`substrate="synthetic_probe_graph"`, 노드 5개) — **엔진 decode
그래프로 전이되지 않는다**. `frac=1.000`이라 `Q1_FRAC` 문턱은
미관측. ★첫 실행 892554의 `TRACE_TRUNCATED`는 sbatch 순서 버그였고
하네스가 배관 실패를 실질 판정으로 바꾸지 않았다(게이트 #21이 실행
에서 처음 지켜진 사례). 상세 아래 "다음 실험 gate" #11 레지스트리
kernel_mech Stage 0‴ A0 행.

★**NSL ①**(GPU 0) — `BRACKET_DECIDES`, cap-binding 참 술어 T가
하계 [0.080,0.119] 상계 구간에 갇힘(`can_run_list`가 telemetry에
없어 점 술어 계산 불가). ★**NSL E-A**(job **892561**, GPU 0.103
GPU-hr, 요청 0건) — `ARITHMETIC_CONFIRMED`: **KV 예산은 cap이
아니라 mamba pool을 따른다**(대조 c48m96=c96 오차 0), 슬롯당 393
KV 토큰. ⇒ `--max-mamba-cache-size` 고정이 ③(손잡이 순화)의
처방으로 작동함이 실증됐으나 **B3는 닫히지 않는다**(크기 +2.9%만
얻음). 둘 다 성능 판정 0건·점 술어("cap이 문다/안 문다") 여전히
금지. 상세 아래 레지스트리 NSL-1 행.

★★★**kernel_mech A1(엔진 기판) 규칙층 감사 `NO-GO`**(2026-08-25,
claims-auditor, ★死因 2건 A1·A2 — 이 감사 사슬 최초) — A0의 합성
그래프 전이 간극을 엔진 decode 그래프에서 닫으려던 설계. 死因·
차단·재설계 권고(**`PDMUX_STICKY_PARTITION`** 기반 부팅 분리)는
아래 레지스트리 신설 행 참조. rev7·rev8 차단(B1–B5·B7/C1–C10)은
A0·A1 어느 쪽도 건드리지 않아 전부 불변.

★**NVTX 근거 정정 승계 확인**(2026-08-24 등재분) — rev5–rev8·B6·
Stage 0″ **6개 문서 전부**에 정정 배너 부착 확인 완료(doc-steward
재검증). ★**표 무결성 수리**(`reports/CONSENSUS.md`): §1 항목1의
`\|A1−A2\|`·`\|효과\|`(절댓값 표기) 이스케이프 누락 2건, §1 항목31이
원시 개행으로 두 물리 줄에 걸쳐 있던 결함 1건 — 전부 셀 병합·이스케이프
로 수리(내용 변경 없음, 렌더링만 정정). 저장소 전체 재스캔 완료,
잔여 결함 0건.

이전: 2026-08-24 (doc-steward — ★★★**F2 양성대조 결과[결과-audit
C4 폐쇄] + kernel_mech rev8 규칙층 재감사 NO-GO[死因 없음, Stage 0″
하위-후보 신설] + NSL-1 rev3 규칙층 재감사 NO-GO[死因 H1·H2] +
정본 술어 서술(HE0 불변 재확인) + 메인 세션 반복 실패 3회 등재
반영. GPU 지출 = 결과 인용분 0.017 GPU-hr뿐(job 891612) · 이 등재
세션 자체는 0. 새 성능 판정 0건.**

★**F2 양성대조 — job 891612**(2026-08-24, gpu43, 1분03초, exit
`0:0`, GPU 0.017 GPU-hr, 판정 `REPLAY_IS_THE_WRITER`, 인용 대상
`workspace/engine-port/results/bcg_probe/p0a_f2_verdict_891612.json`
뿐[게이트 #56]). 같은 green-decode 스트림 위 두 레그가 replay
유무 **한 단계만** 다름: `capture_only`(25/25 캡처·replay 0회) →
**census 라벨 0개** / `capture_replay`(25/25 캡처·25회 replay) →
**34개**(890893 라벨 집합과 일치). ⇒ ★**890893 결과-audit이 남긴
유일한 전제(§5-C4, "census 텐서를 쓴 것은 replay다")가 닫힌다** —
890893의 집합 술어가 "아무도 replay하지 않은 텐서" 위에서 공허하게
만족됐을 가능성이 배제됐다. ★**닫는 것은 C4 하나뿐**: 성능 주장
0건·게이트 0건·한정에 대해 무언·구멍 C·R4 무답. shim은 인용
하네스에서 subclass(재구현 아님)했고, `GraphLaunchShim.__getitem__`
소스가 890893 실행 커밋(`0ba9394`)과 HEAD에서 동일(doc-steward
독립 재확인)함을 확인했다.

★**`kernel_mech` rev8 규칙층 재감사** — `NO-GO`(차단 D1–D9, ★死因
없음, `workspace/engine-port/results/kernel_mech/
audit_kernel_mech_rev8_2026-08-23/VERDICT.md`). C1 미폐쇄(신규
`A5′_KERNEL_COUNT`가 균일 오귀속에 무력 — 카운트 불변인데
`gap_frac` +75% 오차가 이상률 0.000으로 통과) · C8 수리는 진짜(이
트랙 첫 모수화 변이 통과). 값어치 = ★**Stage 0′는 오늘 구매 자체가
불가능**(P7=NVTX 패치 오버헤드, 패치 미작성) — 감사 권고는 더 작은
`Stage 0″`(P0 + NVTX 없는 P3a, ≈0.05–0.1 GPU-hr, ★엔진 패치 0)만
값한다는 것.

★**`Stage 0″` 사전등록 신설**(`workspace/engine-port/results/
kernel_mech/PREREG_STAGE0PP_2026-08-23.md`, ★**미감사**, 규칙층
초안, GPU 지출 0, 미제출) — nsys가 green-context 스트림 위 커널을
볼 수 있는지(node row 방출·stream/context+launch correlation id
존재)를 예/아니오 3문항 + 상수 1개로 판정, 실패 시
`TOOL_CANNOT_DEFINE_K_SET`로 트랙이 이 기판 한정으로 종결.

★**NSL-1 rev3 규칙층 재감사** — `NO-GO`(死因 **H1·H2**, 차단
H3–H11, `workspace/engine-port/results/nsl_lever/
audit_nsl1_rules_rev3_2026-08-23/VERDICT.md`). ★**H1**: rev3이 rev2
판정서의 attainment %(pp)를 **ITL 밀리초로 오독**하고 중심 논증을
세웠다 — 실측하면 **HI에서 ITL 다리가 TTFT보다 더 세게 문다**(통과
12.08% vs 30.46%). ★**H2**: 유일한 생존 전제 "cap이 문다"가
**미증명** — Little 법칙 동시성은 in-system 인구이지 running
batch가 아니고 rate 10·12에서 cap 48을 초과(63.96·83.06). ★**금지
문장 신설 5건**(판정서 §9, 아래 참조).

★★**정본 술어에 대한 서술 등재(판정서 §7 — 성능 판정 아님)**: 정본
변화-trace 하네스가 실제로 채점하는 다리는 게이트 #4의 p95가 아니라
**`mean`-ITL**이고, 그 다리가 이 운영점에서 제거하는 양은
**0.00–1.21 pp**뿐이라 정본 변화-trace goodput은 **TTFT 통과율과
경험적으로 구분 불가**하다. ★★**그러나 HE0는 흔들리지 않는다**(세
갈래 확인: (i) 정본이 이미 p95로 재채점했고 그 술어에서 ITL 다리는
지배적 판별항[순위 보존·강화] (ii) §1-17[tight SLO]은 rate 8에서
측정[바닥 regime 아님] (iii) rate 12×chat 300/50은 정본이 명시
기각한 조합) — 같은 진단의 세 번째 재발(`goodput ≡ throughput
항등`·`joint 0/192 완전분리`). 부수: 요청-내부 ITL p95 자체가
monolithic prefill 때문에 **58–60ms에 이봉 모드**를 가져 **자기
metric cliff**를 가질 수 있음을 확인(PLAUSIBLE, CONFIRMED 아님) —
게이트 #6(CLAUDE.md)·#12(PS 내부)를 ITL 다리에도 적용 권고.

★**메인 세션 반복 실패 3회 등재**(`reports/AUDIT_DEBT_2026-08-23.md`
§6) — 같은 오류(검증 칸에 "무엇을 했다"가 아니라 "무엇일
것이다"를 적음)가 kernel_mech rev7 C1·P0-A rev8 헤더["H1–H7 전부
닫았다"]·kernel_mech rev8 D1에서 **3회** 재발(★3번은 1번을
수리하는 항목 안에서 재발) — 기존 게이트(#62, "이력표에 검증 방법
병기")는 형식상 지켰으나 그 칸 내용이 예측이었다는 점에서
**신규 게이트로 승격**(#70). ★doc-steward 판단: NSL-1 rev3의
H1(단위 오독)도 이 메타-패턴의 네 번째 사례로 볼 수 있으나 구체
기전은 별도 계열(라벨/단위 오해, #67)이 더 정확히 포착 — 두 계열
모두에 교차 등재.

정본 반영: `CONSENSUS.md` rev46→**rev47**(§1-1 F2 addendum·§1-7
정본 술어 addendum·§3 항목87–90 신설·§4 living-doc 행 5개 신설) ·
아래 "다음 실험 gate" #11 레지스트리에 F2(P0-A 행 追記)·kernel_mech
rev8 행 신설·Stage 0″ 행 신설·NSL-1 rev3 행 追記 · #17 후보 상태
갱신(2026-08-24) · "방법론 게이트" #67–70 신설. ★**금지 문장
승계·신설**: *"구멍 C가 닫혔다"* · *"P0-A가 R4에 답했다"* ·
*"cudagraph-ON 운영점에서 한정이 유지된다"* · *"기판이 R0와
일치한다"* · *"양성 하한 통과"*(문턱은 1) · *"rev7/rev8이 차단만
고치면 GO다"* · *"kernel_mech 트랙을 닫았다"* · *"Stage 0″가 트랙을
열었다"* · *"NSL-1이 admission 축을 쟀다"* · *"cap 축이 무력함이
확인됐다"* · *"rev3이 규칙층을 통과했다"* · *"cap은 TTFT 다리에만
작용한다"* · *"HI에서 ITL 다리는 안 문다"* · *"HI 동시성 44.19/48
이므로 cap은 문다"* · *"rate 6–8은 정본이 절벽으로 판정한 대역"* ·
*"정본 goodput은 TTFT-only 지표였으므로 HE0가 흔들린다"*. ★★**불변**:
gate #13/#16 "닫았다" 금지 · switch-cost "닫았다" 금지 · HE0 · 정책
순위 · C2 인용정지 (a)(b) 전부 유지. GPU 지출 = **0.017 GPU-hr**
(job 891612뿐, 이 등재 세션 자체는 0) · **새 성능 판정 0건 · 등급
변경 0건 · 정책 순위 변경 0건.** 상세 `workspace/engine-port/results/
bcg_probe/p0a_f2_verdict_891612.json`, `workspace/engine-port/
results/kernel_mech/{audit_kernel_mech_rev8_2026-08-23/VERDICT.md,
PREREG_STAGE0PP_2026-08-23.md}`, `workspace/engine-port/results/
nsl_lever/audit_nsl1_rules_rev3_2026-08-23/VERDICT.md`,
`reports/AUDIT_DEBT_2026-08-23.md` §6.

이전: 2026-08-23 (2차 세션, doc-steward — ★★★**P0-A 결과 정본
등재 + kernel_mech rev7 규칙층 NO-GO + NSL-1 신규 트랙 반영. 등재
세션 자체 GPU 지출 0, 인용하는 P0-A 결과는 이미 0.054 GPU-hr
지출.** ★**P0-A(cudagraph replay × green-context SM 한정) 결과
도착**(job **890893**, 2026-08-23, gpu43, GPU 0.054 GPU-hr) — 판정
`CONFINEMENT_PRESERVED_THROUGH_GRAPH_REPLAY`, claims-auditor
**`CONFIRMED with conditions`**(조건 C1–C8, `workspace/engine-port/
results/bcg_probe/audit_p0a_result_890893_2026-08-23/VERDICT.md`).
S3·G1-d의 **또 하나의 L0 관측**(R0의 자매 관측) — ★**이 결과는
성능 판정이 아니다**, S3·G1-d는 여전히 닫히지 않는다(서빙·cudagraph
미측정). 확정 문구 ①–⑦은 위 "확정된 결과" Gate 1 블록(S3)의 P0-A
addendum·`reports/CONSENSUS.md` §1-1(같은 자리)에 그대로 옮겼다.
★**kernel_mech rev7(전체) 규칙층 감사 `NO-GO`**(차단 C1–C10, ★死因
없음 — rev6와 같은 계열의 국소·명세층 NO-GO, 감사 권고 = **2단 분할
구매**: Stage 0′ ≈0.2–0.4 GPU-hr는 값하나 Stage A 12부팅+`U_infl`
4부팅[≈0.85–1.2 GPU-hr]은 조건부). ★**신규 트랙 NSL-1**(non-SM-split
admission lever — ★`E-1`[`results/bsweep_regime/`]과 **무관**, 이름
충돌 경고 필수) — **rev1 규칙층 `NO-GO`**(死因 F1–F3: 결정량이 데이터
관측 전에 자기제조[G17 死因의 문자 그대로의 형태]·등록 arm×워크로드
×SLO×메트릭 조합이 저장소에 존재한 적 없음[rev1이 gate #13 근거를
오인용]·cap↔실현 D 앨리어스) → **rev2 작성**(F1–F15 수리 시도,
★**미감사**). ★**신규 인용금지 1건** — NSL-1 rev1이 2026-08-01 E1
전제 실험 4건(870295/870296/870297/870301, claims-auditor 미통과)을 [CS-OK]
인용해 `citation_stops.tsv`에 기계 규칙 신설, 신설 즉시 rev1을
소급 적발(교훈 #41 계열 재발). ★**감사 부채 목록 신설**
(`reports/AUDIT_DEBT_2026-08-23.md` — 층별 전수 + 우선순위, 포인터는
아래 "다음 실험 gate" #11 헤더). 정본 반영: `CONSENSUS.md`
rev45→**rev46**(§1-1[Gate 1 블록] P0-A addendum, §4 living-doc 행
3개 신설) · 아래 "다음 실험 gate" #11 레지스트리에 P0-A 행 갱신
(결과 반영)·kernel_mech rev7 행 신설·NSL-1 행 신설·AUDIT_DEBT
포인터 · #17 후보 상태 갱신 · `reports/paper/
CLAIM_EVIDENCE_MATRIX.md:704`(S3, P0-A를 두 번째 L0 관측으로 追記).
★**금지 문장 승계·신설**: *"구멍 C가 닫혔다"* · *"P0-A가 R4에
답했다"* · *"cudagraph-ON 운영점에서 한정이 유지된다"* · *"한정은
캡처 시점에 박힌다"* · *"물리 SM"* · *"기판이 R0와 일치한다"* ·
*"양성 하한 155/476 통과"*(문턱은 1) · *"하네스가 옳다고
판정됐다"* · *"rev7이 C1–C10만 고치면 GO다"* · *"5연속 NO-GO ⇒
트랙 종결"* · *"NSL-1이 admission 축을 쟀다"* · *"§5-8(b)
admission 갈래를 닫았다"*. ★★**불변**: gate #13/#16 "닫았다" 금지 ·
switch-cost "닫았다" 금지 · HE0 · 정책 순위 · C2 인용정지 (a)(b)
전부 유지. GPU 지출 = **0**(이 등재 세션 자체) · **새 성능 판정
0건 · 등급 변경 0건 · 정책 순위 변경 0건.** 상세 `workspace/
engine-port/results/bcg_probe/audit_p0a_result_890893_2026-08-23/
VERDICT.md`, `workspace/engine-port/results/kernel_mech/
audit_kernel_mech_rev7_2026-08-23/VERDICT.md`, `workspace/
engine-port/results/nsl_lever/audit_nsl1_rules_2026-08-23/
VERDICT.md`.

이전: 2026-08-23 (1차 세션, doc-steward — ★★★**규칙층 5라운드, GPU
지출 0·job 제출 0건 — 직전 핸드오프 §3 권고 순서(A-1→A-2→B-1)를
그대로 집행.** 미측정 실험 등록부 상위 3항목(A-1 `P0-A` 사전등록
rev1→rev6 · A-2 `S-6` rev1→rev4 · B-1 kernel_mech rev7의 B6)을
순서대로 집행했고 **전부 게이트 #34 1단계(규칙층)에서 멈췄다** —
GPU는 한 번도 쓰지 않았다. ★**핵심**: 규칙층 적대 감사(9회,
claims-auditor)가 **집행했으면 예산을 전액 날렸을 결함 2건**
(S-6 E4 — `set -u` 사망으로 즉시 `rc=127` · S-6 F2 — OFF 다리
채택률 0으로 3.146 GPU-hr 결정론적 전액 소실)과 **판정을 거짓으로
만들었을 결함 1건**(P0-A E2 — 합집합 카디널리티 문턱이 1–2 라벨
실 탈출을 영구 비가시화)을 잡았다. 결과: **P0-A**는 rev6에서
규칙층 수렴(감사 권고 — 다음 감사 예산은 §10 하네스층[게이트 #34
2단계]에, 0.17 GPU-hr 미집행) · **S-6**는 사용자 결정으로
**보류(HOLD)**(G18 선례와 같은 형태) · **kernel_mech B6**는
판정 완료(등록안이 예산 판단이 걸려 있던 항목 하나만 닫고, rev7
전체는 여전히 `NO-GO`). 신규 도구 2건(둘 다 자기검사+변이
테스트 포함, GPU 불요): `p0a_rule_totality.py`(516,096 세계·54
검사 PASS) · `s6_offleg_enumerate.py`(자기검사 19/19, `--emit`이
감사 독립 재실행과 바이트 동일). **새 성능 판정 0건 · 등급 변경
0건 · 정책 순위 변경 0건 · HE0 불변.** 정본 반영: `CONSENSUS.md`
rev44→**rev45**(§3 항목81–86 신설, ★**§1[확정 결론]은 건드리지
않음** — 이 세션은 성능 판정 0건) · 아래 "다음 실험 gate" #11
레지스트리에 P0-A·S-6·kernel_mech B6 행 신설(3행) · #17 후보
상태 갱신(S-6는 HOLD로 이탈, P0-A는 규칙층→하네스층 단계 이동) ·
"방법론 게이트" #61–66 신설(§3 항목81–86과 1:1 대응). ★**금지
문장 신설**(핸드오프 §5): *"구멍 C가 닫혔다"* · *"P0-A가 R4에
답했다"* · *"규칙층이 닫혔다"* · *"S-6가 계측 축을 분리했다"* ·
*"B6가 rev7을 열었다"*. ★★**불변**: gate #13/#16 "닫았다" 금지 ·
switch-cost "닫았다" 금지 · HE0 · 정책 순위 · C2 인용정지 (a)(b)
전부 유지. GPU 지출 = **0** · **job 제출 0건 · 새 측정 0건 · 새
성능 판정 0건 · 등급 변경 0건 · 정책 순위 변경 0건.** 상세
`handoff-report/session_handoff_2026-08-23.md`(172줄, 이 세션의
정본 요약), 사전등록 3건(`workspace/engine-port/results/
{bcg_probe/PREREG_P0A_2026-08-22.md, slo_sched/
PREREG_S6_2026-08-22.md, kernel_mech/
PREREG_B6_ELIGIBLE_WINDOW_2026-08-22.md}`), 감사 판정서 9건
(`{bcg_probe/audit_p0a_prereg_*, slo_sched/audit_s6_prereg_*}/
VERDICT.md`).

이전: 2026-08-22 (2차 세션, doc-steward — ★★★**전환 비용
재분석 6문장 정본 승격 — 메인 세션, GPU 지출 0·새 측정 0건**,
claims-auditor 감사 `조건부 승격`(대상 `workspace/engine-port/
results/kernel_mech/PROMOTION_DRAFT_SWITCH_2026-08-22.md`)→차단
B1–B3 전부 해소 후 확정 문구 6문장 등재. 원 문서
`STEP0_SWITCH_GAP_2026-08-22.md`의 헤드라인(`Δmed=0.910ms`를 "전환
기계 비용 상한"으로)은 **감사 `REFUTED`**로 보존(인용 금지)돼 있고,
승격된 6문장은 같은 채널을 **층화·매칭**해 다시 계산한 대조다.
★**핵심 결과**: switch overhead에 처음으로 **device-level 상한**이
생겼다(`s ≤ 0.04 ms/전환` · `d ≤ 0.07 ms/경계`, 가법성 가정·agnostic
`adjust_stream_groups` 경로 한정) — 그러나 인덱스 불변 경계(n=2,039)
의 간극 중앙값(1.81ms)이 전환 경계 두 부류 각각(`SW→PART` 1.41ms·
`SW→FULL` 0.88ms)보다 커서 **순서관계로는 전환 귀속이 비식별**이고,
진짜 기전은 **prefill 생애주기 경계**(admission +0.91·merge +0.45·
둘 다+인덱스불변 +1.29·요청은퇴 +0.34 ms, 매칭 증분)다. residency
(분할 상태 decode forward 지속시간, granite 1.53–1.66×·zamba2
1.73–1.94×, 풀링 1.70×는 인용 금지)는 전환 귀속 상한의 **≈10³배**
— 정본 §1-8·§1-17(positioning)과 방향 일치, **HE0 불변**. 재현
`promotion_metrics.py`(SHA `81c9d8a250ea5b06fed4f93bd810b089b272464
301b50386f5a445e0ef133974`), 검사 14/14+변이 3/3. 정본 반영:
`CONSENSUS.md` rev43→**rev44**(§1-8 판정어 개정[폐기 벤치 stationary
r8 근거 제거]·§1-12 追記·§1-15 각주 신설·§1 신규 행 34[문장2·4·5
원문]·§3 항목9/53 追記[14번째 재발=`phase` 항등식]·항목78–80 신설)·
`reports/paper/venue_positioning.md:178`·`CLAIM_EVIDENCE_MATRIX.md:
512`의 `switch_count≈0` overclaim 정정·`citation_stops.tsv` 4행
추가. 아래 "다음 실험 gate" #11 레지스트리에 switch-cost 행 신설·
"방법론 게이트" #58–60 신설(§3 항목78/79/80과 1:1 대응). ★★**불변**:
성능 판정 0건 · HE0 불변 · 정책 순위 0건 · Gate 2 귀속 전진 0 ·
C2 인용정지 (a)(b) 승계 · gate #13/#16 "닫았다" 금지 유지 ·
★**"switch-cost 트랙을 닫았다" 금지 신설** — 닫힌 것은 *"인덱스
변경 자체가 비쌀 수 있다"* 가설뿐이고 **컨트롤러 구동 전환 경로 ·
green→green 전환(0건 관측) · 포화 운영점**은 미측정이다(`alternate`
엔진 패치의 값어치는 하락 — 엔진이 이미 공짜 자연 대조[인덱스
불변 adjust 경계]를 갖고 있었다). GPU 지출 = **0** · **새 측정
0건 · 새 성능 판정 0건 · 등급 변경 0건 · 정책 순위 변경 0건.**
상세 `workspace/engine-port/results/kernel_mech/{PROMOTION_DRAFT_
SWITCH_2026-08-22.md, audit_{switch_cost,step0,promotion}_2026-08-22/
VERDICT.md, audit_kernel_mech_rev6_2026-08-22/VERDICT.md,
PROMOTION_METRICS_2026-08-22.json}`, `reports/CONSENSUS.md` rev44.

이전: 2026-08-22 (1차 세션, doc-steward — ★★**`%smid` R0 결과 도착·정본
등재**(job **889631**, gpu40, 2026-08-22T01:44:25–01:46:04, **0.0275
GPU-hr**, exit `0:0`, `git_head=28a972c`) — 사전등록
`results/smid_census/PREREG_SMID_R0_2026-08-14.md` + 2026-08-21
2층 감사(`GO-with-conditions`) 조건 4건 충족 후 제출된 결과를
claims-auditor가 **`CONFIRMED(scoped)` — 등재 가능(조건 4건)**으로
판정. ★**등재 헤드라인**: `%smid`는 이 프로세스·이 기판
(A100-SXM4-80GB, eager, 엔진 없는 별도 프로세스)에서 green
context를 건너는 **전역 일관 라벨**이다 — 상보 green 파티션 두
집합이 서로소(교집합 0)·합집합이 green ctx 생성 이전 평범
스트림 관측 id 집합과 집합으로 동일(`GLOBALLY_CONSISTENT_LABEL`,
`stop=false`, `sizes=[74,34]`↔`divide_sm(108,(8,0),2)` target
정확히 일치, idx2 `[54,54]`도 일치). ★**"물리 SM 인덱스"가 아니라
"전역 일관 라벨"**(§3.4) — 카디널리티를 계산 자원 비율로 환산
금지. **R2(기전, 서술 한정)**: 같은 스트림으로 green ctx 생성
전/후를 재면 `D \ S_post = ∅`(양쪽 포화, `min_hits=155`) — 이
프로세스·이 기판·**green 스트림이 유휴인 순간**에 한해 green
context *생성 자체*가 primary context 스트림의 도달 가능 SM id
집합을 줄이지 않았다(스코프 3개 필수: id 집합≠계산량 / 유휴 순간
한정 / 포화≠부재 증명). ★**아티팩트 결함 2건 신규 등재**: N1
(`.txt` 판정서가 `[:4000]` 절단으로 `stop`·`plain_control_detached`
·`control_status` 누락, `.json` 4777B vs `.txt` 4127B, grep 0건
확인) · N2(`setup.green_ctx_attached`의 **이름이 값의 부정** — 값
`false`가 "green ctx가 붙어 있다"는 뜻이라 `.txt`만 읽으면
정반대로 읽힌다) ⇒ **정본·논문은 `smid_l0_verdict_889631.json`만
인용, `.txt`/`.out` 인용 금지**. ★★**불변 배너(전 인용 지점
공통)**: 성능 판정 **0건** · `CONSENSUS.md` §1-1 "PD 분리 자체"
귀속 전진 0 · Gate 2-S 크기 인용 셀 1개 불변 · §1-1 인용 금지
그대로 · S3(`CLAIM_EVIDENCE_MATRIX.md:704`)·G1-d(이 문서
"확정된 결과" 1번, E-3 옆)는 **닫히지 않는다**(L0 유사물만 측정,
cudagraph-ON 운영점 전달 문장 금지) · HE0·정책 순위 불변 ·
**gate #13/#16 "닫았다" 금지 유지**. 방법론 게이트 #56 신설
(보고 필드 이름이 값의 부정일 수 있다 — 게이트 #21의 보고 층
변종) · #57 신설(결정론적 열거를 확률적 커버리지 증거로 쓰지
마라). GPU 지출 = **0.0275 GPU-hr**(이번 결과 도착분) · **새
성능 판정 0건 · 등급 변경 0건 · 정책 순위 변경 0건**. 상세
`workspace/engine-port/results/smid_census/{smid_l0_verdict_
889631.json, audit_smid_r0_2026-08-21/VERDICT.md}`, 아래 "다음
실험 gate" #11 레지스트리 `%smid` R0 행(2026-08-22 갱신)·#17
후보 상태.

이전: 2026-08-21 (2차 세션, doc-steward — ★★**kernel_mech rev3
재감사 → rev4-lite → rev5, 세 판본 연속 규칙층 `NO-GO`**(차단
5·8·7건, 전부 **하네스 착수·GPU 지출 전**에 규칙층에서 걸림 —
가장 깊은 층: *"커널 안인가 밖인가"* 판정이 **반사실 선택에
의존**한다, 대칭/비대칭 두 반사실이 같은 데이터에서 반대 판정을
내고 둘 다 똑같이 "정의". 사용자 판단 대기: rev6 / 보류 / 종결) +
★**`%smid` R0가 처음으로 제출됐다**(job **889631**, `amd_a100nv_8`,
`PENDING`, ★**결과는 아직 없다** — 2026-08-14 `CONDITIONAL-GO(5조건)`
원문이 저장소에 남지 않아 7일 만에 소실됐음을 확인, 2층 감사가
조건을 새로 도출해 이번엔 파일로 기록[`audit_smid_r0_2026-08-21/
VERDICT.md`], 지목 결함(fail-open 2건 + C1–C3) 수리 완료·제출).
방법론 게이트 #42에 새 사례 追記(공허참이 PASS를 낸다) · #55
신설(감사 산출물을 파일로 남기지 않으면 조건이 소실된다). GPU
지출 = job 889631 하나(예상 ≈0.17 GPU-hr, `PENDING`) · **새 성능
판정 0건 · 등급 변경 0건 · 정책 순위 변경 0건 · HE0 불변 · gate
#13/#16 "닫았다" 금지 불변.** 상세 `handoff-report/session_handoff_
2026-08-21b.md`, 아래 "다음 실험 gate" #17(2026-08-21 3차 갱신)·
"방법론 게이트" #42 追記·#55(신설).

이전: 2026-08-21 (1차 세션, doc-steward — ★**gate #13 job-축 캠페인 완주 +
양 arm `PASS`**(등록 11.52 GPU-hr 대비 실측 **11.66 GPU-hr**) 반영 +
S0(a) 초판 결론 **철회**(감사 REFUTED) + G18 rate-축 프로브 **트랙
보류(HOLD)** + kernel_mech rev3 차단 2건 설계 본문 수리 +
`PROJECT_STATUS.md` 자신의 P1 판정 과잉 인용 정정[위 참조]).

**gate #13 job-축 캠페인**(`workspace/engine-port/results/s8_scaleup/
G13_RESULTS_2026-08-21.md`·`G13_CAMPAIGN_LOG_2026-08-21.md`·
`PREREG_G13_2026-08-20.md` §10, 원자료 `G13_ANALYSIS_2026-08-21.json`)
— 32/32 job `COMPLETED`·288/288 부팅 성공·실패 시그니처 0건
(2026-08-20T17:32Z ~ 08-21T02:35Z). 사전등록 규칙("격차 ≥ 3×UB95")을
M8 **18.4배**·Ha8 **4.9배** 여유로 통과(감사 지정 최악 조건[노드
성분 복원]에서도 유지 — 11.6배·3.1배). ⇒ **job 축 할당 변동
(σ_alloc)만으로는 설명 대상 격차를 만들 수 없다**(등재 판정어 =
**강한 지지(범위 한정)** — 아래 한정 전부 붙는다). ★★쓸 수 없는
문장 불변: **"gate #13을 닫았다"**(865533 대조의 batch⊗regime
앨리어스 불변) · **이 캠페인은 `Δ_batch`를 재지 않았다**(배치 축
없음, `CONC=16` 단일 — 설계가 말한 두 항목 중 `σ_alloc` 절반만).
★**노드 한정 필수**: σ_alloc은 2노드(gpu36 24·gpu43 8)·단일 9시간
창 표집, 노드 성분은 **계수 0.400·유효 df≈1**로만 포함 — 정본
gate #13(1)의 **≥3 노드 부분 미충족 확정**(날짜 2일은 충족).
`grand_mean_r`(M8 3.0613·Ha8 3.1229)는 **정본 r 재추정 아님·arm
비교 금지·2.91/3.058/3.114 대조 금지**, 인용정지 (a)(b) 해제 없음.
★부수 확정: **비용 상수 144 s/부팅이 실측 145.8 s(+1.25%)로 검증**돼
rev3 §6-8 미해결 항목 해소. **성능 판정 0건**(분산 측정이지 성능
비교 아님)·HE0 불변·정책 순위 변경 0건.

**S0(a) 초판 결론 철회**(`workspace/engine-port/results/s8_scaleup/
S0A_VERDICT_2026-08-20.md` rev2, claims-auditor 감사 REFUTED) —
*"C-g(부팅 내 `--conc` 교대)의 전제는 측정 가능한 모든 lag에서 성립하지
않는다"*를 **철회**한다(측정은 재현되나 판정선이 분모를 틀렸다 — 1
부팅쌍짜리 대조를 2 부팅쌍짜리 SD와 견줌). 비용 정합(`s_alt` 대
`2σ_boot`)으로 재판정하면 **C-g 이득은 교대 속도의 함수**다 — 부팅당
**8구간** 교대는 이기고(Ha8 5.46×·M8 1.31×) **4·2구간은 진다**
(0.19–0.88×). 신규 등재(성능 판정 아님): 정보 없는 기준선은 1이
아니라 **√NBIN=2.83**, **`r`은 창 의존량**(Ha8 30s 3.025 vs 60s
3.116, 정본 채택구간 밖). ⇒ **gate #13 불변**(S0(a)는 임계경로 밖,
등록 계획의 비용·검정력에 영향 없음).

**G18 rate 축 트랙 보류**(`workspace/engine-port/results/slo_sched/
PREREG_G18_PROBE_2026-08-20.md` HOLD 배너, `G18_RATE_VALUE_2026-08-20.md`
§6) — rev1 규칙층 REFUTED → rev2도 **NO-GO**(6死因 중 2건만 제거,
새 死因 5건) → 사용자 결정으로 **트랙 보류**(gate #13 캠페인을 먼저
완주하고 재개 판단). 핵심 결론("지금 rate 값을 등록할 수 없다")은
감사 CONFIRMED이나 근거였던 1.26% 산포가 **귀무 기대치(1.207%,
n=4)와 구분 불가**로 갱신(값은 blk1 단일 부팅 산물, blk2–4만 쓰면
0.493%). 취소 6 job 중 1건만 3분36초 실행 = **0.06 GPU-hr**. ★**gate
#16은 불변** — 두 판본 어느 쪽도 닫히지 않았다.

**kernel_mech rev3**(`workspace/engine-port/results/kernel_mech/
DESIGN_KERNEL_MECH_REV3_2026-08-20.md` §3) — 차단 2건을 설계 본문에
수리(§3.1 셀 라벨을 realized로 판정[시간가중 조건 + 혼합구간 폐기] ·
§3.2 `gap_frac` 항등식 가드[union 정의 + 변이 테스트]). 후보 (vi)
클럭은 정본이 이미 *분할 셀 1396–1410MHz 평평*을 기록해 뒀으므로
**2차 강등**. Stage B 폐기(2026-08-20 이미 반영)는 불변, ★이 수리
자체는 **미감사**(다음 세션 재감사 대상).

GPU 지출(이번 세션 합계) **11.72 GPU-hr**(gate #13 캠페인 11.66 +
취소된 G18 프로브 0.06). 새 판정 1건(gate #13 job-축 분산 PASS,
성능 판정 아님) · 결론 철회 1건(S0(a)) · 등급 변경 0건 · 정책 순위
변경 0건. 상세 `handoff-report/session_handoff_2026-08-21.md`.

이전: 2026-08-20 (doc-steward — **P1 프로브 판정 반영:
`UNAVAILABLE (CUPTI×GREEN-CONTEXT)`**(job **886718**, `--exclusive
--constrain=hwperf`, node gpu38, 1분55초, `COMPLETED 0:0` — 동반
프로브 job 886752 포함 **GPU 지출 0.032 GPU-hr**). greenctx 다리에서
문서화된 시그니처(`Failed to prepare kernel for profiling` / `Unknown
Error on device 0` / exit 9)가 **정확히** 재현됐고, **두 겹 대조**로
귀속이 깨끗하다 — (a) **다리 간**: greenctx exit=9 / control(full GPU)
exit=0·에러 0건·같은 GEMM(`ampere_bf16_...`) 정상 수집(60행). (b)
★**다리 내부**: 같은 프로세스·같은 ncu 호출에서 green ctx **밖**
RNG 커널은 수집 성공(8행)하고 green ctx **위** GEMM만 실패 — ★★2026-08-21
정정(doc-steward, 원문 P1_VERDICT §2 대조): **두 겹의 대조가 같은
방향을 가리킨다**(원문 표현 그대로 복원 — 다리 간 대조는 `realized_sm`
16 vs 108도 함께 바뀌고, 다리 내 대조는 커널 종류 자체가 다르다[RNG
초기화 vs GEMM] ⇒ 어느 쪽도 변인을 하나로 완전히 좁히지 못한다;
"변인은 하나뿐"은 이전 배너의 **과잉 인용**이었다). ⇒ **Stage B(ncu 커널
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

   ★**`%smid` R0 결과 도착(2026-08-22, job 889631, gpu40, 0.0275
   GPU-hr, claims-auditor `CONFIRMED(scoped)`)** — 위 E-3가 자기보고
   층(드라이버가 요청값을 그대로 돌려줌)만 확인했던 것과 달리, R0는
   **엔진·모델·요청 없는 별도 프로세스**에서 eager 스트림 위 `%smid`
   census로 **id 집합 층**을 처음 쟀다: 상보 green 파티션 두 집합이
   서로소(교집합 0)이고 합집합이 green ctx 생성 이전 평범 스트림
   관측 id 집합과 집합으로 동일(`GLOBALLY_CONSISTENT_LABEL`,
   `sizes=[74,34]`가 `divide_sm(108,(8,0),2)` target과 정확히 일치).
   ★**"물리 SM 인덱스"가 아니라 "전역 일관 라벨"**이라고 읽어야
   한다(카디널리티→계산 자원 비율 환산 금지). 같은 스트림의 green
   ctx 생성 전/후 대조(R2)는 이 프로세스·이 기판·**green 스트림이
   유휴인 순간**에 한해 `D \ S_post = ∅`(도달 SM id 집합이 줄지
   않음)를 관측했다 — id 집합≠계산량·유휴 순간 한정·포화≠부재
   증명 3개 스코프 필수. ⇒ **S3·G1-d는 여전히 닫히지 않는다**(이건
   L0 유사물이지 서빙 중 SM id 확인이 아니고, cudagraph-ON 운영점
   전달 문장은 금지). 아티팩트 결함 N1(`.txt` 판정서 절단으로
   `stop`·`plain_control_detached`·`control_status` 누락)·N2
   (`green_ctx_attached` 필드명이 값의 부정)이 있어 **`.json`만
   인용**(`.txt`/`.out` 인용 금지). 상세 "다음 실험 gate" #11
   레지스트리 `%smid` R0 행(2026-08-22 갱신), 원자료
   `workspace/engine-port/results/smid_census/{smid_l0_verdict_
   889631.json, audit_smid_r0_2026-08-21/VERDICT.md}`(수정 금지·
   인용만), `reports/CONSENSUS.md` §1-1(이 블록, R0)·§3 항목76·77.

   ★★★**P0-A 결과 도착(2026-08-23, job 890893, gpu43, A100-SXM4-80GB
   cc (8,0), compute mode `Default`, GPU 0.054 GPU-hr, claims-auditor
   `CONFIRMED with conditions`[조건 C1–C8])** — R0(위)와 달리 **엔진의
   `cudagraph replay`가 green-context SM 한정을 전달하는지**를 잰 첫
   측정. 판정 `CONFINEMENT_PRESERVED_THROUGH_GRAPH_REPLAY`: `divide_sm
   (108,(8,0),2)[0]=(74,34)` 분할의 decode 절반(34) 스트림에 캡처한
   단일 토이 커널 그래프를 같은 스트림에서 replay했을 때, 관측된
   `%smid` 라벨 집합이 같은 스트림 eager census의 34-라벨 집합과
   **정확히 일치**(`Δ=E∪E_rev=∅`, 공허참 아님 — graph 레그
   33,480 block 관측 중 34-라벨 밖 0건, 라벨별 최소 히트 476[문턱은
   1]). 부수 도구 타당성 사실(신규): green-context 스트림 위 CUDA
   그래프 캡처·replay가 이 기판에서 **가능**(75/75, ★단 단일 커널
   노드·`pool=None`[엔진은 공유 pool]·`capture_error_mode="global"`
   한정, 엔진 캡처 경로 진술 아님). 서술 레그 2개(교차: green 캡처→
   plain replay=34, plain 캡처→green replay=108)는 **관측값만** —
   ★**기전 해석 금지**(H1 "한정이 캡처 시점에 결정" vs H2
   "`CUDAGraph.replay()`가 ambient 스트림이 아닌 곳에 launch"가 같은
   두 수를 예측, 비식별). ★**정정된 과잉 진술 2건, 그 정정된 형태로만
   인용**: (a) *"기판이 R0와 일치한다"*가 아니라 **"검사한 두 축
   (compute mode, census sha256)에서 R0와 불일치 없음"**(노드는
   gpu40→gpu43로 다름) (b) *"green-stream graph capture WORKS"*는
   위 §5 한정 없이 단독 인용 금지. 확정 문구 ①–⑦ 전문은
   `reports/CONSENSUS.md` §1-1(이 블록) P0-A addendum에 그대로.
   인용 대상 **`p0a_verdict_890893.json`뿐**(게이트 #56, `.txt` 인용
   금지; 부착은 `run_green_ctx_attached`만, `attachment_positive.
   observed`는 극성 반대[P2]; 드라이버 버전·경과시간·GPU-hr는
   `.json`에 없음[P1]). ⇒ **S3·G1-d는 여전히 닫히지 않는다**
   (P0-A도 R0와 같은 L0 유사물, 서빙·cudagraph 미측정 — 남는 질문
   R4/구멍 C는 엔진 층 L1 in-server boot census). 상세 "다음 실험
   gate" #11 레지스트리 P0-A 행, `reports/CONSENSUS.md` §1-1
   (이 블록, P0-A addendum), `reports/paper/
   CLAIM_EVIDENCE_MATRIX.md:704`(S3, 두 번째 L0 관측으로 追記),
   원자료 `workspace/engine-port/results/bcg_probe/{p0a_verdict_
   890893.json, audit_p0a_result_890893_2026-08-23/VERDICT.md}`
   (수정 금지·인용만).
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

★★★**백엔드 스코프 배너(2026-08-28 2차, doc-steward, `deprecated_v2/
README.md` 비준 반영)**: 위 국소 탄력도(16→24 = 0.77–0.88·44→92 =
0.09–0.35)와 헤드라인 2.36–2.91×는 **arm 계열에 걸쳐 `triton`(3
arm)·`flashinfer`(Hs8) 백엔드가 섞여 있다**(기존 인용정지 (a)가
가리키던 것과 같은 자리, B11 "cross-arm claims involving Hs8 must
carry that caveat"의 축소판). 이 세션에서 그 혼합이 **연구자 선택이
아니라 엔진 강제**임이 확인됐다(Nemotron-H 계열+`triton`=부팅 거부,
Zamba2+`flashinfer`=스케줄러 사망 2/2, job 896776) ⇒ **"백엔드를
고정하고 arm만 바꾼 대조"는 이 4-arm 격자 자체에 존재하지 않는다**
(신규 게이트 #83, `CONSENSUS §3 항목103`). ★C2의 등급·수치는
**불변**(이 배너는 스코프 명시일 뿐 반증이 아니다) — 단 Nemotron-H
계열(Nemotron-Nano 등)로 이 수치를 이전하지 말 것. 상세
`deprecated_v2/README.md` §1–§2.

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
  199–528, keepalive_errors=0`). **어제(2026-08-15) 등재한 서술이 바로 다음
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

   ★★追記(2026-09-11, doc-steward, R2 true-dual GPU correctness
   트랙) — 위 1–4는 Claim D/E **성능** 게이트이며, 이번에 통과한
   것은 **별개의 correctness 게이트**다(혼동 금지, 위 1–4 중
   어느 것도 이 job으로 평가되지 않았다 — observer effect·decode
   progress/ITL·throughput regression·estimator coverage는 전부
   미측정). 2026-09-11 admission latch stale-True 버그 수정
   (`02918e8`, 2026-07-24 보류 해제·사용자 R2 복귀 결정) 이후
   재실행 job 907032(0.12 GPU-h)는 true-dual 두 boot 모두
   **split-prefill ownership race로 크래시(FAIL)**했다(구현
   사실). 경합 수정 커밋 `874873b`, 이어 판정 규칙을 TD 출력이
   나오기 전에 사전 고정한 하네스 v2 커밋 `e16e93f`. job
   907100(19:37–19:47 KST, ≈0.16 GPU-h, 아티팩트 커밋 `ea99191`)
   이 **PASS**했고, claims-auditor 결과 감사(`workspace/
   engine-port/results/r2_correctness/audit_r2corr_2026-09-11/
   VERDICT.md`)가 S16 순차+O8 단일-probe 중첩 프로토콜 한정으로
   **`CONFIRMED(scoped)`** — 무스코프 "같은 토큰"(동시 부하[C 층]
   포함)은 관측으로 **`REFUTED`**(C01·C10·C19 3/32 arm-분리
   불일치, 원인 미확정, TD 결함 귀속은 **`NOT-YET-SUPPORTED`**).
   등재 한 줄(판정서 §7.2, 문자 승계): *"R2 true-dual GPU
   correctness: job 907100 PASS, `CONFIRMED(scoped)`, S/O
   프로토콜 한정. 인용 금지 R2C-1…16과 필수 병기 P-1…7을
   판정서에서 문자 그대로 승계한다. 새 성능 판정 0건, Claim D
   등급 불변, HE0 불변."* Claim D는 여전히 **미검증**. R2 GPU
   지출 오늘 0.12+0.16≈**0.28 GPU-h**(이 트랙 최초 지출). 선결
   상태(VERDICT §4): closes #5(cudagraph-ON 호환, scoped) ·
   부분 #2(GPU correctness, S/O 프로토콜 한정) · #1(latch)은
   코드 수정뿐이며 이 job에서 GPU 발화 관측 0(`admission_limited`
   185개 결정 중 0). ★**정정 추기**: 아티팩트 커밋 `ea99191`의
   문구 "since B8/O3 confirmed the realized split was D44"는
   **R2C-9**로 정정한다 — 직접 근거는 `controller_decision`
   텔레메트리(target=current=44, 60/60)이고 B8/O3는 표본 기반
   검사라 근거로 부정확했다(커밋 자체는 재작성하지 않음). 인용
   금지 R2C-1…16 요지: R2C-1(무스코프 "같은 토큰" 금지)·R2C-2
   ("동시/서빙 부하에서 TD≡legacy" 금지)·R2C-3(C 층 불일치를
   "TD 결함"도 "무해 노이즈"도로 단정 금지)·R2C-5(토큰 동치가
   D44 실현을 확인한다는 주장 금지)·R2C-11(스레드 안전성 검증
   완료 주장 금지)·R2C-14(무스코프 "Claim D 증거" 금지) 외
   전문은 VERDICT §5, 필수 병기 P-1…7도 동소. 신규 방법론 게이트
   3건(#167 G-1·#168 G-2·#169 G-4, 아래 "방법론 게이트" 절)·
   追記 3건(G-3→#50, G-5→#1·#114, G-6→#95). **후속 실험
   X1–X3(각≈0.16 GPU-h)·하네스 결함 H1–H3은 미실행·미승인**
   (사용자 판단 대기) — `reports/paper/EXPERIMENT_ROADMAP.md`
   "P1/P2" 절. 정본 반영: `CONSENSUS.md` §5-8(a) 追記(rev66),
   `reports/paper/CLAIM_EVIDENCE_MATRIX.md` Claim D 행·"주장
   제한" 갱신. 상세 `workspace/engine-port/results/r2_correctness/
   {job_907032/, job_907100/, audit_r2corr_2026-09-11/VERDICT.md}`.

   ★★追記(2026-09-12(2), doc-steward — X1 사전등록 규칙층만
   등재, **X1 job 실행 중이므로 결과는 등재하지 않는다**, GPU 0):
   위 "후속 실험 X1–X3" 중 X1(민감도 양성대조+C 기전 판별)의
   사전등록 rev1이 제출 전 감사에서 **반증**돼 SUPERSEDED됐다
   (死因: 행별 split 수 판정이 커널 실효 양자화 `MIN_BLOCK_KV=32`
   를 놓쳐 "강제 6 ⇒ 24/24 교란"이 거짓, 실효 **18/24**).
   유효 판본 rev2(`--triton-attention-num-kv-splits 2` CLI 단일
   노브 + 엔진 소스 `38c1aca` 고정)는 claims-auditor 규칙층 감사
   `GO-with-caveats`(死因 0, 반전 0/7표면, 차단 D1–D4·D6–D9
   반영 완료)를 받았다. 판정서 원문 전사 = `workspace/engine-port/
   results/r2_correctness/x1_prereg/VERDICT_x1_rules_2026-09-12.md`.
   필수 병기 X1P-1…7·인용 금지 X1C-1…9는 이 문서 최상단 배너
   "C) X1 규칙층만 등재" 항목에 문자 그대로 등재돼 있다. **X1은
   위 게이트 2("동일 fixed split에서 true dual이 legacy 대비…")
   를 포함해 어떤 Claim D 선결도 아직 평가하지 않는다** — job이
   아직 실행 중이다. GPU: R2 correctness 트랙 0.28 GPU-h 지출
   불변, X1 ≈0.16 GPU-h는 **결과 도착 후 갱신**. 신규 방법론
   게이트 #170·#171(아래 "방법론 게이트" 절). 정본 반영:
   `CONSENSUS.md` §5-8(a) 追記(rev67), `reports/paper/
   EXPERIMENT_ROADMAP.md` "P1/P2" 절 갱신.

   ★★追記(2026-09-12(3), doc-steward — X1 **결과** 등재, rev68,
   새 성능 판정 0건·Claim D 등급 불변[미검증]·HE0·정책 순위·
   stake #1 전부 불변, GPU 0.28→**0.43 GPU-h**): 위 X1 rev2가
   job 907456(0.154 GPU-h)으로 완주했고, claims-auditor 결과
   감사가 총괄 **`CONFIRMED(scoped)`**를 판정했다(등록 예보
   F1–F4 전부 충족·PASS 및 8/96 불일치 독립 재현·귀속 성립).
   전문·§8 정본 등재 스코프 문장·§9 인용 금지 X1C-10…14·§10
   필수 병기 X1P-1′·2′·3′·5′·6′·8·9는 이 문서 최상단 배너
   (2026-09-12(3))에 문자 그대로 등재돼 있다. 요지: 메인
   세션이 쓰려던 "O06이 D44를 밟는 probe prefill 민감도를
   보정한다"는 문장은 **`REFUTED`**다 — 뒤집힌 두 단위(S05·
   O06) 모두 비분할 `stream_index 5`에서 돌았고, D44에서 돈
   유일한 교란된 decode는 0/32로 뒤집히지 않았다. "24/24
   교란"은 단위 지시함수이고 실제 스텝 커버리지는 82.5%다.
   C 층 "불일치 감소"는 Σ(등가류−1) 10→10 불변으로 성립하지
   않는다. **X1은 위 게이트 2를 포함해 어떤 Claim D 선결도
   닫지 않는다**(#1·#2·#3·#4a·#4b·#5 상태 불변, #2는 오히려
   더 약함). 신규 방법론 게이트 5건 #172–176(아래 "방법론
   게이트" 절, #176=긍정 사례). GPU: R2 correctness 트랙
   0.28→**0.43 GPU-h**(이번 0.154). **다음 실험(판정서 §12,
   전부 미승인·사용자 판단 대기)**: (1) C-tier 귀무대조
   `R2C_ORDER="L L L L"`(≈0.154 GPU-h, F3을 해석 가능하게 만드는
   유일한 값싼 길) (2) `TD TD TD TD` 1 job(+0.154) ⇒ n=4/arm
   ★**철회(2026-09-12(5), B3 판정서 §8, 인용 금지 B3C-3)**: 한
   job의 4 boot은 독립 런이 아니라 게이트 3(n≥4)을 충족하지
   않는다 — 상세 이 문서 최상단 배너(2026-09-12(5)) "A. 정정".
   (3) D44 resident decode 동치엔 새 층 O′+판정 규칙 v3
   사전등록 필수(≈0.16, 선결 #2를 넓히는 실험, X1 후속 아님)
   (4) GPU 0: preflight 스텝 커버리지·비교기 Σ(classes−1) 출력·
   죽은 텔레메트리 2건 처리(engine-porter)·사전등록 템플릿
   항목 추가 (5) 권고 안 함: cap을 더 낮추거나 다른 수치
   구성으로 X1 반복. 정본 반영: `CONSENSUS.md` §5-8(a)
   追記(rev68)·§3 항목192–196(신설), `reports/paper/{CLAIM_
   EVIDENCE_MATRIX,EXPERIMENT_ROADMAP}.md` Claim D/"P1/P2" 절
   갱신.

   ★★追記(2026-09-12(5), doc-steward, GPU 0 — (1)이 "B3"
   사전등록[`L L L L`+`TD TD TD TD` 2 job, 0.31 GPU-h]으로
   구체화돼 **`NO-GO`**(死因 N2, 미실행)를 받았고, 후속 설계
   "OS"(`R2C_ORDER="L TD TD L"`, 0.154 GPU-h)는 **규칙층
   `GO-with-caveats`**(死因 0, 미실행)를 받았다. 또한 A1(러너
   spool-copy 결함 수리)·A2(dual-worker 텔레메트리 H2–H5)·A4
   (line-citation 앵커) 구현이 커밋 `09a8075`·`ae7830e`·
   `3153260`으로 완료됐다(GPU 0, 전체 CPU 426 tests OK). 전문·
   신규 게이트 G-B3-1…5(`#180–184`)·G-OS-1…5(`#185–189`)·
   구매 순서 권고 변경(A1 커밋→X3→OS)·사용자 결정 대기 항목
   (Claim E 캠페인 405 run 중 270 run 실행 불가, ★2026-09-13
   정정 — 고유 225/가능 180, 아래 追記 참조)은 이 문서
   최상단 배너(2026-09-12(5)) 참조. `CONSENSUS.md`
   rev69→**rev70**.

   ★★追記(2026-09-13, doc-steward, GPU 0 — X3 규칙층
   `GO-with-caveats`(死因 0·차단 D1–D13, 미실행) 등재 + 산술
   정정 + 사용자 결정 2건): X3 사전등록 규칙층 감사가 "약 270"
   중복 계수를 **고유 225/가능 180**으로 정정하고, 자기 OS §8
   문장 1건("X3는 성능 트랙 전체를 연다")을 철회했다(캠페인
   블로커 3개 중 X3가 제거하는 것은 0개). 사실 정정 3건(F1이
   측정 안 한 것을 측정했다고 말함·F2 3중 과대·F3 크기 오류
   [probe-boot 26/32=81%]), 신규 게이트 G-X3-1…5(`#190–194`,
   `CONSENSUS.md` §3 항목210–214). **같은 세션에서 사용자가
   즉시 X3를 무효화**했다 — 모델 교체 결정(Zamba2-2.7B→
   NemotronH Nano-9B-v2-Base, X3는 Zamba2 기준이라 SUPERSEDED)
   + 1차 캠페인 범위 결정(Claim D+P3까지, 실행 순서
   (0)λ*측정→(1)새 모델 R2 correctness→(2)P1→(3)P2[W3+W4×
   {B1,B4}×5rep]→(4)P3→(5)Claim E). 새 선결 1건: `generate_
   campaign.sh:10`의 `sustainable_rate`(λ*) 기본값 4 미측정
   (게이트 #6 위반). 전문·907100/907456/X1의 Zamba2-2.7B 스코프
   동결·NemotronH+triton 부팅 거부 교차 트랙 위험(§3 항목103/
   게이트#83)은 이 문서 최상단 배너(2026-09-13) 참조.
   `CONSENSUS.md` rev70→**rev71**.

   ★★追記(2026-09-12(4), doc-steward, Claim E 컨트롤러 코드,
   engine-porter A안 구현, GPU 0 — 이 항목4가 아니라 위 항목4의
   "Claim E" 게이트 자체와 인접한 별건이나, 같은 컨트롤러
   코드베이스라 여기 追記): 2026-09-12(2)가 미해결로 등재했던
   긴장 2건(즉시성 지연·occupancy 90% 단독 트리거 무발화)이 A안
   구현으로 해소됐다(전문은 최상단 배너[2026-09-12(4)]). **열린
   결정 후보(아직 판단하지 않음)**: dwell이 upshift에 면제이므로
   위반이 지속되면 상태 사다리가 연속 iteration마다 한 칸씩
   올라갈 수 있다(기본 케이던스에서 `D16→24→34→44→108`이 4
   iteration[≈40ms], 이전 판본은 4 epochs[≥400ms]) — **"upshift를
   케이던스 epoch당 한 칸으로 제한할 것인가"는 다음 세션/사용자
   판단 대기**이고, 성능 영향은 generic/hybrid GPU 측정 0건으로
   **미측정**이다(이 문항을 성능 주장으로 인용 금지). 신규
   방법론 게이트 3건 #177–179(`CONSENSUS.md` §3 항목197–199).
   `CONSENSUS.md` rev68→**rev69**. 새 성능 판정 0건·Claim E
   등급 불변(미검증)·HE0·정책 순위·stake #1 전부 불변.
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

   ★★**追記(2026-09-08, 2차 세션, `longctx_conflict` 트랙 재개 시도,
   doc-steward, GPU 0) — 여전히 "게이트 미실행"이다.** 사용자 발의로
   재개를 두 판본 설계했으나 규칙층 적대 감사 **2회 전부 `NO-GO`**
   (rev1 死因8·차단11 / rev2 死因9·차단12, 트랙 계열 누적 12연속).
   ★**Stage 0 잔류를 정본 함수(`e1_pin_check::compute_decode_realized`,
   재구현 아님)로 GPU 0 재실행**: 스윕 27셀 realized **0.6986–0.9994**
   (≥0.95 20/27) ⇒ **파티션은 발화했다**(rev1이 "decode-only라 거의
   실현 안 된다"고 예측한 것과 정반대, 그 예측은 철회) · D108 9셀
   realized **≤0.0437**, 히스토그램 **16 SM 94–100%** ⇒ `D16≡D108`의
   원인은 "둘 다 108"이 아니라 **"둘 다 16"**(§1-21/C1과 **같은
   방향**, 그것을 정본 함수로 재확인). ★★**신규 confound `C-R`을
   같은 세션이 스스로 강등**: 잔류 결손이 arm×ctx에 정렬됐다는 관측의
   지배 성분은 기전이 아니라 `compute_decode_realized`의 **구간 부과
   아티팩트**(순간 상태에 스냅샷 간격 전체를 부과 — `T_C16384_D16`
   bin0 스냅샷 64개 전부 연속 런 길이 1·`prefill_active` 0/64·부과
   dt median 0.4059s=전체 dt median 0.4063s ⇒ 64×0.406=25.70s).
   bin0 발생률의 arm 차이(T@16k 29.8% vs H@16k 0%)는 실재하나
   **원인 미확정**. ★**감사 2F9(신규 등재)**: decode-only+prefill
   유휴+sticky는 **PD-mux 운영점이 아니다**(성분 측정 — 정본이
   `[[scale-8b-sm-sensitivity]]`에서 이미 막은 것과 같은 형태,
   "레버 존재만, 정책 이득 아님, HE0 안 되살림"). **방향 판정**:
   (ㄴ) **L1 직행(ITT)** 채택 — 질문은 §5-6이 stake #1로 명시한
   "장문에서 최적 static split 위치가 움직이는가"; ITT는 노출을
   목표 설정값으로, `realized`는 병기 공변량(게이트 아님)으로 둬
   `bin0`·표본화율·`0.90 vs 0.95`가 판정 경로에서 사라진다(per-protocol이
   12연속 죽은 이유 = 비순응의 원인이 처치 자신이라는 collider).
   ⚠️**ITT 재프레이밍은 감사받지 않았다** — *"HE0가 ITT다"*는
   메인 세션의 독해이지 정본 자기 규정이 아니다. (ㄱ) 축소 L−2는
   보류(사지 않음): 기각 사유는 비용이 아니라 감사 2F9이며,
   "(ㄱ)은 비용 때문에 기각됐다"는 이전 진술은 **철회**(실측 부하
   기준 10–40 GPU-h, 이전 추정 100+ GPU-hr은 등록 부하 파라미터가
   4배 부풀린 것). **금지문 신설**: "게이트 #14를 닫았다"·"C-R이
   long-ctx 충돌의 증거다"·"장문에서 실현 할당이 무너진다"·"L−2가
   binding/non-binding이다"(미측정)·"(ㄱ)은 비용 때문에 기각됐다"
   (철회)·"bin0는 파티션 복원 지연이다"(인과 미확정)·"T가 장문에서
   decode를 더 자주 비운다"(반증 — decode 빈→참 전이 T4·H3·M3로
   동일). 상세 `workspace/engine-port/results/longctx_conflict/
   {PRIORITY_2026-09-08.md, PREREG_L2_2026-09-08.md,
   PREREG_L2_REV2_2026-09-08.md, RESULT_RESIDENCY_2026-09-08.md,
   FINDING_BIN0_2026-09-08.md, DIRECTION_2026-09-08.md,
   audit_l2_rules_2026-09-08/VERDICT.md,
   audit_l2_rules_2nd_2026-09-08/VERDICT.md}`.

   ★★**追記(2026-09-09, doc-steward, GPU 0.818 GPU-h[이 트랙
   최초 지출]) — 여전히 "게이트 미실행"이다, 방향은 좁혀졌다.**
   L1 직행(ITT) 설계를 rev3까지 밀었으나 규칙층 감사 `NO-GO`
   (死因 3F1: 판정 면 `S_norm`이 ITL 결합항을 정규화하지 않아
   도착률 규칙이 decode 동시성을 ctx와 ≈15× 공선으로 묶음, 트랙
   계열 누적 13연속). 대안으로 `PREREG_RATIO_2026-09-08.md`
   (요청당 prefill:decode 일 비 `ctx/out`를 처치 수정자로) 설계
   했으나 이 역시 `NO-GO`(死因 R1: 처치 실현 듀티사이클 `f`가
   앵커 셀에서 0.004, 등록 문턱 0.20을 51배 차이로 자기 기각,
   트랙 계열 누적 **14연속**). 프로브 5건(0.818 GPU-h)이 종이로
   못 얻던 것을 셋 닫았다: P1이 Nemotron+flashinfer·동시성>1
   에서 처치가 실제 발화함을 확인(`f_time` 0.99), P3/P3-C가
   균질화 config 비용 무시 가능함(0.051%)을 확인, P4가 sticky의
   기전(prefill 유휴 중에도 분할 유지)을 관측(`pab==0` 68배) —
   단 실측 `itl_solo`(13.0ms, 가정의 1/3)가 `RATIO` 설계의 `C`
   상한을 0.70→0.23으로 무너뜨려 死因 R3(peak decode 부하 축
   도달 불가)를 3배 악화시켰다. ★★**정본 정정**: `CLAIM_EVIDENCE_
   MATRIX.md:264`·`CONSENSUS.md` M4R 항목의 "G16 28파일 39,849
   구간 전수 반례 0" biconditional은 **한 방향만 참**(⟹ 방향
   1,643건 반례, split의 35.8% — 코드층 술어 자체가 아니라
   스냅샷 기록 시점의 record-skew 아티팩트, 상세는 위 "최종
   갱신" (C) 참조). **다음 방향(합의, 미등록)**: `R≈0.3–0.45`
   에서 knee 이상 부하로 1요인(ctx·out 고정) static split 스윕
   — 이 영역에서 층위 1·5가 해소됨(P1 실측 근거, f≈0.99 vs
   g16의 0.10–0.28). 상세 `workspace/engine-port/results/
   longctx_conflict/{PREREG_L1_2026-09-08.md,
   PREREG_RATIO_2026-09-08.md, audit_l1_rules_2026-09-08/
   VERDICT.md, audit_ratio_rules_2026-09-08/VERDICT.md,
   audit_dutycycle_2026-09-08/VERDICT.md,
   PREREG_PROBES_2026-09-08.md, probes/RESULT_PROBES_2026-09-08.md,
   probes/ADJUDICATION_P4_2026-09-09.md,
   RESULT_RULEPOWER_2026-09-08.md, QUESTION_COMPARISON_2026-09-08.md}`,
   `handoff-report/session_handoff_2026-09-09.md`.
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
    2026-08-18 G17·E-B1 추가, 2026-08-19 G13 rev2·rev3 추가, 2026-08-24
    F2[P0-A 행 追記]·kernel_mech rev8·Stage 0″·NSL-1 rev3[追記] 추가,
    2026-08-25 kernel_mech Stage 0‴ A0 완주 행 신설·A1(엔진 기판) 행
    신설·NSL-1 행 追記(① ·E-A) 추가, 2026-08-26 kernel_mech A1 행에
    sticky rev2 재설계·규칙층 감사 3회·스모크 job 893663 追記·NSL-1
    행에 ③②E-B 묶음 첫 규칙층 감사 追記 추가, 2026-08-28
    kernel_mech Stage 0‴ A0 행에 정오표 追記(gate #66 오인용, TC1
    rev1 감사 발견) · tc1_model_attrib·m4r_confinement·규율도구
    (presubmit) 3행 신설, 2026-08-28(같은 날 2차, doc-steward)
    tc1_model_attrib 행에 rev3 규칙층 감사(死因 F8–F12)·P3
    `MEASUREMENT_ABSENT`("셋째 뿔")·모델 전환(Zamba2-2.7B→
    Nemotron-Nano-9B-v2) 追記 · 규율도구(presubmit) 행에 도구
    자기감사(`RESTRICTIONS_INERT` 등 신설, TOOL_REV 2) 追記,
    2026-09-01(doc-steward) cp_baseline(chunked-prefill baseline)
    행 신설 — 규칙층 감사 4회 전부 `NO-GO`, GPU 4 job 1.36 GPU-hr
    · 새 성능 판정 0건, 2026-09-03(doc-steward) cp_baseline 행에
    5·6회차 규칙층 감사(둘 다 `NO-GO`, 死因 H1–H8/F1–F8, 그림자
    81%/88%) + V-probe·운영점 재배치·ShareGPT 센서스·W1·correctness
    gate·모델 부팅 스모크 측정(GPU 18 job 5.78 GPU-hr, 새 성능 판정
    0건) + G8 장부 누락 정정 + `presubmit.py` read-only 아님 발견·
    타 세션 데이터 소실 복구 追記, 2026-09-08(2차 세션, doc-steward)
    cp_baseline 행에 게이트 #14 도구 수리 상태(브랜치
    `fix/gate14-tci-analyze`, main 미병합) 追記 + `longctx_conflict`
    행 신설(규칙층 감사 2회 전부 `NO-GO`, 트랙 계열 누적 12연속))**
    — 이 표의 목적은 판정 기록이 아니라
    **다음 세션이 같은 설계를 그대로 재제출하는 것을 막는 것**이다.
    ★**감사 부채 목록(2026-08-23 신설, 2026-08-24 §6 반영 완료)** —
    이 표가 "재제출 방지"에
    집중하는 반면 "무엇이 아직 감사되지 않았는지"는 층별(규칙층·
    하네스층·분석기층·결과층)로 흩어져 있다. 전수·우선순위는
    `reports/AUDIT_DEBT_2026-08-23.md` 참조(정본 아님, 작업 목록 —
    1순위 P0-A 결과 등재[완료]·2순위 NSL-1 rev2 규칙층
    감사[rev3까지 완료]·3순위 `g16_analyze.py` 재감사[미실행]).
    §6(메인 세션 반복 실패 3회)은 "방법론 게이트" #70·CONSENSUS
    §3 항목90으로 승격 반영 완료(2026-08-24).
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

    ★★**2026-08-21 갱신 — G13 rev3도 실행 완료됐다**(jobs 32개,
    array feeder, 전량 `G13_FEEDER_STATE.tsv`) — 32/32 `COMPLETED`·
    288/288 부팅 성공, GPU 11.66 GPU-hr(등록 11.52 대비 +1.2%).
    사전등록·블라인드·감사 해제 절차를 완주해 **양 arm `PASS`**
    (강한 지지(범위 한정)) — 상세는 아래 표 "G13 job-축 캠페인 결과"
    행, "다음 실험 gate" #17 2026-08-21 갱신. ★**이 레지스트리에서
    G13 rev3을 제거하지 않는다** — GO-with-caveats 판정 자체는
    여전히 "재제출 방지" 대상 선례로 유효하고(다음에 batch 축·노드
    ≥3축을 다시 설계할 때 이 캠페인의 한정을 그대로 물려받는다),
    그리고 **★**2026-08-23 갱신 — 하네스·피더 층 감사 완료**(`results/s8_scaleup/audit_g13_harness_2026-08-23/VERDICT.md`, `CONFIRMED with conditions` H-1–H-5). **288 부팅 원자료에서 결정량까지 독립 재계산**해 등재값과 일치했고(기존 감사의 독립 재구현은 **합성 데이터만** 검증했다), 유일한 조작 기전(**블록 위치 고정효과**)을 2원 모형으로 실측해 **부재**(F=0.44/0.95)임을 보였다. **H-1·H-2·H-5는 미결로 승계**한다. ★분석기 SHA `01448211…` **변경 금지 조항 유지**(이번 감사도 그 SHA 한정). 종전 문구: 하네스층 자체(수리 8건 B1–B9)는 별도 감사를
    받은 적이 없다**(재감사 없이 이 캠페인의 분석기 SHA
    `01448211…`를 바꾸지 말 것). ★**G13 rev2**(9.24 GPU-hr 설계,
    규칙층 NO-GO)는 SUPERSEDED로 그대로 둔다 — rev3만 실행됐다.

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
    `g16_analyze.py`(`analyzer_sha256=bac783be…[★2026-08-23 정정 [CS-OK]: 폐기된 중간 판본을 지목하려면 그 값을 써야 하므로 의식적 면제 — 디스크·JSON 자기기록은 `bac783be…`다]`, C-9(3) 재감사분·
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
    | `%smid` R0 (`workspace/engine-port/results/smid_census/PREREG_SMID_R0_2026-08-14.md`) | CONDITIONAL-GO(5조건, 2026-08-14 — ★**원문이 저장소에 남지 않아 2026-08-21 확인 시점에 7일 만에 소실**, 레지스트리 이 줄 + `handoff-report/session_handoff_2026-08-15.md:68`·`:286` 두 줄뿐, 2026-08-14 날짜 핸드오프 파일 자체가 없다) → ★**2026-08-21, 2층 감사가 조건을 추측 재구성이 아니라 새로 도출, `GO-with-conditions`(차단 C1–C5, 파일 `audit_smid_r0_2026-08-21/VERDICT.md`로 이번엔 보존)** → 지목 결함 전부 수리(아래) → **제출(job 889631, `amd_a100nv_8`)** → ★★**2026-08-22, 결과 도착(gpu40, 01:44:25–01:46:04, 0.0275 GPU-hr, exit `0:0`) — claims-auditor `CONFIRMED(scoped)`, 등재 가능(조건 4건, 메인 세션이 ③④ 처리)**: `GLOBALLY_CONSISTENT_LABEL`(disjoint·union=108=`|D\|`·`sizes=[74,34]`=target=`divide_sm(108,(8,0),2)`, idx2 `[54,54]`도 일치) — "물리 SM 인덱스" 아닌 "전역 일관 라벨". R2(서술 한정, 3스코프 필수): `D\S_post=∅`(green ctx 생성이 primary 스트림 도달 SM id 집합을 줄이지 않음, eager·idle·이 기판 한정). 아티팩트 결함 N1(`.txt` `[:4000]` 절단으로 `stop`/`plain_control_detached`/`control_status` 누락)·N2(`green_ctx_attached` 필드명이 값의 부정) — **`.json`만 인용**. 불변: 성능 판정 0건·S3/G1-d 미종결·Gate 2 귀속 전진 0·HE0·gate #13/#16 "닫았다" 금지. \| 2026-08-14: 런타임 PTX 검사가 죽은 코드(`smid_l0_census.py:410`의 `JITFunction.cache` 미존재 속성 참조) + 스코어러 fail-open(`:523/:529` 가드가 `None`을 통과). ★2026-08-21 신규 지목·수리: **C1** 빈 census가 공허참 3중(∅∩∅=∅⇒disjoint·∅=∅⇒tiles_D·∅→∅를 포화로 판정)으로 `GLOBALLY_CONSISTENT_LABEL`+`stop=False`를 냄(메인 세션 직접 재현 확인, 아래 "방법론 게이트" #42 追記) — 양성 하한을 포화 검사 **앞**에 둬 수리 · **C2** 미포화 `S_post`에서 부재 주장(`lost_ids`) — 대상별 게이트로 수리 · **C3** 전달 실패(결과)와 green ctx 미부착(측정 실패)을 한 라벨로 합침 — 토큰 금지 축소+화이트리스트+AST 전이 검사로 수리 · **동일 fail-open의 P0-A 복사본**(`p0a_graph_sm_confinement.py:249`) — 별건 수리, ★이걸 안 잡았으면 모든 GPU 런이 `UNDETERMINED`로 끝났을 것(0.35 GPU-hr 절약) · **재감사(R2)**: green ctx 부착 전제가 테스트 0건이라 퇴화 변이 2종이 통과 — **양방향 전제**(green 부착 ∧ plain 대조 ≥1 미부착)로 수리. 최종 하네스: 자기검사 58/58·변이 7→28. ★2026-08-22 결과 감사 caveat(판단 완료): (a) 사전등록 §0.2 항목번호 오프바이원(원문 순서 1,2,3,4,6,5) — 이후 인용은 번호 대신 문구로 할 것(권고 등재) · (b) 게이트 후보 2건은 #56·#57로 신설(아래 "방법론 게이트") · (c) `smid_l0_run.sbatch:86`이 stage 4 rc 미검사(스코어링 실패가 exit 0으로 끝날 수 있음) — engine-porter 이관 항목으로 등재(미수정). 상세 `handoff-report/session_handoff_2026-08-21b.md` §2.2, 원자료 `workspace/engine-port/results/smid_census/smid_l0_verdict_889631.json`(수정 금지·인용만, `.txt`/`.out` 인용 금지) |
    | E-1 (`workspace/engine-port/results/bsweep_regime/PREREG_E1_BSWEEP_REGIME_2026-08-14.md`) | NO-GO(F1–F9) | T8 arm에 SM 순회 훅 부재. `R≈2.1` 유도 입력 3개 오류 |
    | C2-R rev1 (`workspace/engine-port/results/s8_scaleup/PREREG_C2R_RULES_2026-08-15.md`) | NO-GO(5표적) | 결정량("단일 `b*`에서의 arm 간 순서")이 자기가 고치려던 슬라이스 아티팩트를 재생산 — 순서가 b의 함수이고 곡선이 교차(T8 b=1 꼴찌 2.476→b=16 1등 2.388). `b*` 규칙이 분모 미정의로 무이빨 문턱(E1-b/c `M<36` 동형). D3가 표적 arm에서 계산 불가. PIN 0.80을 게이트로 쓰면 3/8 탈락·Ha8 전멸 |
    | ★C2-R rev2 (`workspace/engine-port/results/s8_scaleup/PREREG_C2R_RULES_REV2_2026-08-15.md`) | ★**GO**(2026-08-15, claims-auditor, 범위 한정 — **이 표에서 유일한 GO**) → **★★실행 완료(2026-08-16, jobs 883574/883575)** | 손잡이가 아니라 결정량 자체를 교체(게이트 #35 준수): 4 arm→2 arm(M8·Ha8)·순서→arm별 비의 citability·`b*`=16 고정·PIN을 보고축으로 강등·D3 삭제·정지 (a)는 영구 정지로 포기. 하네스 `s8_c2r.sbatch`+`s8_c2r_client.py`(커밋 `19b8853`). **결과**: `r_M8(16)=3.058`·`r_Ha8(16)=3.114`(t(5) [3.056,3.060]·[3.077,3.155], within-job boot 구간·`n_indep=6`, percentile CI[3.0566,3.0595]/[3.0848,3.1354]는 과소피복이라 본문 인용 금지·원자료 포인터만), 양성대조가 항등식(방법론 게이트#9 아홉 번째 재발, 재구현 스크립트는 저장소 밖), C2 등급 무변경. 전문 `C2R_RESULTS_2026-08-16.md`, `CONSENSUS.md` §3 항목56 |
    | ★★G16 rev3 (`workspace/engine-port/results/slo_sched/PREREG_G16_RULES_REV3_2026-08-16.md`, rev1·rev2는 SUPERSEDED 배너 부착 보존 + addendum C/D/E) | ★★**GO** → 스모크 2회 통과 → **캠페인 완료(2026-08-17, 4블록, jobs 884336/884410/884411/884412)** → `G16_RESULTS_2026-08-17.md` rev1 → **claims-auditor 적대 감사 반영 rev2(정본 승격)** | 규칙층: `Δ=D_itl−D_ttft` 강제표를 설계보다 먼저 명시(`TRUNCATED`/`TRUNCATED_LOW` 사전 선언), 결정량 3종 선언적 병존(1차-A/B/C+2차, 게이트 #35 준수), 원 격자 전부 재실행+블록설계(정방향/역순 쌍, R2 caveat #5 제거), 양성대조 5종. 하네스층 NO-GO 4건은 addendum A로 해소, 재감사가 N7(텔레메트리 비용)을 addendum B로 추가 발견. `Δ_SLO` 단일 60ms 점 결정량을 **사다리 함수**로 교체(C2→C2′) — payoff는 ITL SLO≲58.6ms 구간에만 존재, 사다리 해상도(1ms)가 이 payoff 구간(exact band 폭 0.092ms)을 은폐할 뻔함(방법론 게이트 #46). **결과(2026-08-17)**: 양 phase `ITL_SATURATED`(정보 있는 음성) — HI `M_ttft` argmin=d44(식별 CONFIRMED, "처리량 최적" REFUTED), payoff 구간 `Δ_SLO` 부호 무판정(P=0.632). `gap_upper` 비식별 확인 검사 자신이 항등식이었음이 감사로 드러남(방법론 게이트 #45). H-1~H-8 판정·정직 고지 2건(동률 가드 지연·블록 동시배치 원인)은 `CONSENSUS.md` §1-33. ★**"gate #16을 닫았다"고 쓰지 말 것**(재정식화판만 닫힘, 원문 문턱 판본은 rate 축 잔존). 상세 `handoff-report/session_handoff_2026-08-16.md` §15·§4-1–4-3, `CONSENSUS.md` §1-33 |
    | ★G13 job/node축 (`workspace/engine-port/results/s8_scaleup/DESIGN_G13_JOB_BATCH_2026-08-16.md`) | ★**NO-GO**(2026-08-16, claims-auditor — 지목 주장 2건 둘 다 REFUTED) | (a) "between-job SD of r은 부분 항등식" — REFUTED, 저장소의 유일한 비-축 교차-job 데이터에서 두 다리 job 편차가 사실상 **독립**(비 축 RMS 0.804%≈√2×0.581%), b16에 가장 가까운 셀(T8 b15)은 오히려 반-상쇄(+1.345%) — 문서 §2("구조적으로 0")와 §4(그 비-상쇄 값을 σ_job 사전값으로 사용) 두 절이 동시에 참일 수 없음. (b) "Ha8 검정력 사실상 0(1.01)" — REFUTED, `1.01`은 아카이브 스크립트가 산출하지 않는 값(JSON은 `0.562`, `power()` `target_pct` 기본값 미덮음); 실제 **P(pass)=0.394**, 그중 **93%가 `σ̂=0` 퇴화** ⇒ UB95 규칙이 `MS_B≤MS_W`면 자동 PASS라 **부팅 잡음이 job 신호를 삼킨 런을 "job 축 통제됨"으로 선언**(피복률 Ha8 0.638). F5 표본산정이 df=1·타 arm·b=1 사전값에 걸려 b≥15 셀만 쓰면 M8 결론 전부 FAIL로 뒤집힘(3.06→1.83). F6 재현 경로가 결정 표를 산출 안 함. ★**"gate #13을 닫았다"고 쓰지 말 것**(gate #16과 동형 — healthy 체제 b9/b12 대조 batch가 원 아티팩트에 없어 batch⊗regime 앨리어스). 감사 독립 재구현 5조각 저장소 보존(커밋 `1d57cb6`, `audit_g13_independent_2026-08-16/`). 상세 "다음 실험 gate" #13, `handoff-report/session_handoff_2026-08-16.md` §4-4(a) |
    | ★G13 rev2 (`workspace/engine-port/results/s8_scaleup/DESIGN_G13_JOB_BATCH_REV2_2026-08-17.md`, ★**SUPERSEDED 배너 부착** — rev3를 읽어라) | ★**규칙층 NO-GO**(2026-08-19, claims-auditor, 게이트 #34 1단) | 死因 3건: (A) F2 가드 문턱이 틀렸다 — exact-F 상한의 실제 퇴화점은 `MS_B ≤ MS_W`가 아니라 `MS_B ≤ F₀.₀₅(df_B,df_W)·MS_W`(이 격자에서 `F₀.₀₅`=0.34–0.45)인데, 상한이 양수·유한·피복 정상인 `MS_B/MS_W∈(0.34,1.0]` 구간을 통째로 "측정 실패"로 버렸다(**방법론 게이트 #21의 역방향 재발**). (B) Ha8의 `σ_boot=1.5684%`가 부팅 1개(blk5, 잭나이프 2.46×, 나머지 5개는 0.90–0.98×)의 산물이고 그 부팅은 `C2R_SENSITIVITY_2026-08-16.json`의 `named_exclusion`에 **이미 등재**돼 있었다(**교훈 #31의 내부 재발** — σ_job엔 밴드 3중화, σ_boot엔 민감도 0건인 비대칭). (C) `power2()`가 exact-F를 하드와이어해 §0 헤드라인이 사전등록 규칙(피복 조건부 Satterthwaite/exact-F 선택)의 작동 특성이 아니었다. 상세 `DESIGN_G13_JOB_BATCH_REV3_2026-08-19.md` §1, "다음 실험 gate" #17 |
    | ★★G13 rev3 (`workspace/engine-port/results/s8_scaleup/DESIGN_G13_JOB_BATCH_REV3_2026-08-19.md`) | ★★**GO-with-caveats**(2026-08-19 재감사) → ★★★**실행 완료(2026-08-21, 32/32 job·288/288 부팅, 결과는 아래 "G13 job-축 캠페인" 행)** | 死因 A/B/C 전부 규칙층에서 수리(GPU 0). 수리 후 **전 밴드·전 arm에서 채택 계획의 창이 60 s**가 돼 `σ_boot ∝ 1/√T` 가정이 소멸하고 S0(a)가 임계경로 밖으로 나갔다(재감사 CONFIRMED). ★**밴드 점 판정 = `hi_chi2_upper`**(M8 3.84+Ha8 7.68=**11.52 GPU-hr**) — `mid`(6.72)의 붕괴 σ_boot 1.652%는 점추정 1.568%에서 여유 5%뿐이고 n=6 CI [0.979,3.847]% 대부분에서 무너져 P(pass) 0.21로 `UNDETERMINED` 위험(비대칭: prior 과대추정은 비용만, 과소추정은 캠페인 무효). ★재감사가 메인 세션 수리에서 blocking 4건(B1–B4)을 잡았다: B1 `seq2()`가 死因 A 미수리 상태로 §2.4가 인용(수리 후 Ha8 0.795/0.719/0.466, 인용값 "<0.4"는 2배 stale) · B2 **항등식 3건**(구 PC8·구 S3·구 S4)을 `all_pass`에 편입(**교훈 #9 열세 번째 재발**, `_oc` 실호출 + **변이 테스트**로 반증 가능한 검사로 재작성해 해소) · B3 PC5가 가드 수리에 원리상 무감인데 "수리 검증 증거"로 오독 · B4 `named_exclusion`이 실제로는 부팅 2개(`blk2`,`blk5`)인데 1개로 서술. 전부 수리 완료(커밋 `dee6c80`·`75e7ba3`). ★**이 캠페인은 어느 밴드 점을 사든 gate #13을 닫지 못한다**(batch⊗regime 앨리어스 — healthy 체제 b12=**0**/3064 vs collapsed **58**/1173 + 정본 #13(1)의 **노드·날짜 축 미배선**, job 축만). 산출 `DESIGN_G13_STATS_REV3_2026-08-19.json`. **"gate #13을 닫았다"고 쓰지 말 것 — 불변.** 상세 `DESIGN_G13_JOB_BATCH_REV3_2026-08-19.md` §2.2.1·§4.1·§5, "다음 실험 gate" #17 |
    | ★★★**G13 job-축 캠페인 결과**(`workspace/engine-port/results/s8_scaleup/G13_RESULTS_2026-08-21.md`, 실행기록 `G13_CAMPAIGN_LOG_2026-08-21.md`, 원자료 `G13_ANALYSIS_2026-08-21.json`·`_REFUSED.json`, 전 job `G13_FEEDER_STATE.tsv`) | ★★★**양 arm `PASS`**(2026-08-21, 메인 세션 사전등록·블라인드·감사 절차 완주, 등재 판정어 = **강한 지지(범위 한정)**) — GPU **11.66 GPU-hr**(등록 11.52 대비 +1.2%) | 등록 규칙("격차 ≥ 3×UB95")을 M8 `gap/(3·UB95)`=**18.37**·Ha8=**4.95**로 통과(`σ̂_job` M8 0.0629%·Ha8 0.1898%, `UB95` M8 0.0928%·Ha8 0.6194%, 설명 대상 격차 M8 5.12%·Ha8 9.20%). 계획 이탈 0건·`balanced=True`·`CONTROLS: PASS`. ★**정정(2026-08-23, H-2): 아래 조건은 "최악"이 아니다** — 장치 성분이 표집되지 않았으므로 그 복원은 설계가 *관측한* 분산만 재팽창시킨다. ★대신 2026-08-23 감사의 **모형 무가정 천장**을 병기한다(임의효과 분해를 통째로 버리고 job 평균을 iid로만 취급): **M8 7.74 · Ha8 4.52**(노드복원 합산 **4.90 · 2.86**) ⇒ ★**PASS는 임의효과 모형에 의존하지 않는다.** 종전 문구: **감사 최악 조건(노드 성분 완전 복원, `σ_ν=σ̂_job/√0.400`)에서도 PASS 유지**(11.62·3.13, 단 이 조건에선 노드 성분 유효 df≈1 문제로 피벗 명목 피복이 성립 안 할 수 있어 크기 논증에 그침). 진단 2건 등재: (1) 실현 σ_boot이 등록 밴드점(`hi_chi2_upper`)에서 M8 2.66×·Ha8 0.28×로 반대 방향 이탈했으나 실현값 재시뮬에서도 합격선 유지 → rev3 §2.2.1이 `mid` 대신 `hi`를 산 결정이 사후 정당화(사후 관찰, 규칙 변경 아님). (2) `per_boot_n` 이분산 ★**정정(2026-08-23 하네스층 감사 H-3)**: 등재된 **풀링 15.3×는 두 다리를 섞은 값**이다. 다리별로는 M8 d16 **5.58×**·M8 d92 **3.12×**·Ha8 d16 **1.69×**·Ha8 d92 **1.79×**이고, 풀링값의 대부분은 **구조적 다리 차이**다(d92는 같은 창에서 구간을 `r≈3`배 빨리 쌓으므로 개수비가 **추정 대상 자체를 일부 포함**한다 — 게이트 #9 계열). 피벗이 실제로 기대는 등분산 가정은 *다리 내부* 가정이고 **등재값보다 3–5배 덜 위반**됐다 ⇒ **등재 진단이 비관 쪽으로 틀렸다**(판정 무영향). ★원인 귀속 완료: **피더 산물이 아니다**(전 job 동일 work item) — 여유 18배라 안 뒤집히나 여유 작은 후속 설계엔 가중 분석 필요(미결로 등재). ★★쓸 수 없는 문장(사전등록·불변): **"gate #13을 닫았다"**(865533 batch⊗regime 앨리어스 불변) · **`Δ_batch`를 측정했다**(배치 축 없음, `CONC=16` 단일 — 설계가 말한 두 항목 중 `σ_alloc` 절반만) · `grand_mean_r`(M8 3.0613·Ha8 3.1229) arm 비교·2.91/3.058/3.114 대조. ★**노드 한정**: 2노드(gpu36 24·gpu43 8)·단일 9시간 창·★**상시 동일노드 공존**(동시 실행 ≤2, **31/32 job**이 다른 캠페인 job과 시간 겹침 — 2026-08-23 감사 H-1, 방향은 inflating이라 PASS를 만들지 못한다)·★★**할당 GPU 식별자 미기록**(하네스가 `SLURM_NODELIST`만 찍고 index/uuid/pci를 한 번도 질의하지 않았다 — H-2). ★**표집된 적 없는 성분은 아래 `σ_ν` 복원으로 회복되지 않는다**, 노드 성분 계수 0.400·유효 df≈1 — 정본 #13(1)의 **≥3 노드 부분 미충족 확정**(날짜 2일 충족). 부수: **비용 상수 144s/부팅 검증**(실측 145.8s) → rev3 §6-8 미해결 항목 해소. `g13_analyze.py` 층 등록 양성대조 0건, 감사 독립 재구현으로 대체(**SHA `01448211…` 한정**, 재감사 없이 코드 변경 금지). 해제 절차 중 첫 `--unblind`가 문서 §2의 감시 문자열 자기인용으로 자기거부(fail-closed, 증거 `_REFUSED.json` 보존). **성능 판정 0건**·HE0 불변·정책 순위 변경 0건·인용정지 (a)(b) 해제 없음. |
    | ★kernel_mech rev2 (`workspace/engine-port/results/kernel_mech/DESIGN_KERNEL_MECH_REV2_2026-08-16.md`) | ★**NO-GO**(2026-08-16, claims-auditor — 기준1 REFUTED·기준2 PLAUSIBLE·기준3 REFUTED) | F1 `wave_eff`가 ncu 메트릭이 아님(ga100 `--query-metrics` 확인) — 폐기 선언한 수제 유도를 1차 결정량으로 되살림(게이트 #36 死因이 이름만 바꿔 생존). F2 §3.2 축퇴 대수 부호가 반대라 게이트가 위험구간(D=16)을 정확히 통과시킴(게이트 오설정). F3 ★★`_ncu_target.py:68-71`의 CUPTI×green-context 비호환 문장 발견 — 정본이 그 다섯 줄 아래(73-74)만 인용해온 결함 발견(위 "8B decode-SM 민감도 측정 노트" 정정 배너 참조, 참이면 Stage B 전체가 이 기판에서 구성상 불가하나 error code 9는 3가지 경합 귀속이 있어 미확인 리스크로만 등재). F4 `ncu --pid` 부착 옵션이 존재하지 않아 §9-2 재발방지 구조 실행 불가. F5 ncu 기본값 `--clock-control base`가 후보(vi)를 클럭 핀으로 박고 직렬화가 후보(v)의 동거를 소멸시킴. 기준3: provenance 12건이 아니라 감사 출처 11+rev2 자작 1, 재구성이 제약 3건을 느슨화 방향으로 떨어뜨림. GO 경로 = 문서 수정 8건 + Stage 0′ 4프로브(≈40–50분). 상세 `handoff-report/session_handoff_2026-08-16.md` §4-4(b) |
    | ★★kernel_mech P1 프로브 (`workspace/engine-port/results/kernel_mech/p1_probe/p1_greenctx_ncu.sbatch`, 결과 `P1_VERDICT_2026-08-20.md`) | ★★**`UNAVAILABLE (CUPTI×GREEN-CONTEXT)`**(2026-08-20, 메인 세션, job 886718 — 도구 타당성 판정, 성능 판정 아님) — F3의 "미확인 리스크"를 해소 | greenctx 다리: 문서화된 시그니처(exit 9) 정확히 재현. **두 겹 대조**로 귀속 확정 — (a) 다리 간: control(full GPU, 같은 GEMM)은 에러 0건·60행 정상 수집. (b) 다리 내부: 같은 프로세스에서 green ctx 밖 RNG 커널(8행)은 성공, green ctx 위 GEMM만 실패 — ★2026-08-21 정정(doc-steward, 원문 대조): **두 겹의 대조가 같은 방향을 가리킨다**(다리 간 대조는 `realized_sm` 16 vs 108도 함께 바뀌고, 다리 내 대조는 커널 종류 자체가 다르다[RNG 초기화 vs GEMM] — "변인은 하나뿐"은 원문 P1_VERDICT §2보다 조인 과잉 인용이었다). ⇒ **Stage B(SM 제한 하 ncu 커널 내부 카운터) 구성상 불가 확정 → kernel_mech rev3는 Stage A 전용으로 범위 축소**(문서 수정 8건 중 Stage B 대상 최소 5건 적용 대상 소멸). 동반 프로브 886752(non-exclusive)가 `ERR_NVGPUCTRPERM`으로 실패 → 선례 스크립트 `run_ncu_profile.sh:15-17`의 권한 근거("batch면 열린다")가 불충분함을 반증, 실제 구분선은 exclusive+hwperf. GPU 0.032 GPU-hr(886718+886752). ★서술 한계: 성능 판정 아님·green context 실행 자체는 정상(`realized_sm=16`)·내부 기전 미분리·A100-SXM4-80GB/driver 580.105.08/ncu 2025.3.1.0/CUDA 13.0.2/이 클러스터 한정. 상세 `P1_VERDICT_2026-08-20.md`, `P1_886752_REVIEW_2026-08-20.md`, `reports/CONSENSUS.md` §3 항목52 追記(6) |
    | ★★G17 payoff 밴드 (`workspace/engine-port/results/slo_sched/DESIGN_G17_PAYOFF_BAND_2026-08-17.md`) | ★★**규칙층 NO-GO**(2026-08-18, claims-auditor, gate #34 stage 1 — `audit_g17_rules_2026-08-18/` a1–a8) | 死因 3건: (a) `a1_restricted_grid.py` — 제안한 S2 격자 `U={d44,d54,d64,d74}`에서 `D_ttft=44=S_min(U)`가 **양 phase 모두** §3의 `FORCED`(`Δ≥0`) 셀을 재생산 — 결정량이 데이터 관측 전에 격자 선택만으로 부호 강제(같은 4블록을 원 7-arm 격자로 두면 `P(부호>0)` HI 0.632, `U`로 좁히면 0.875 — 격자가 판정을 만든다). (b) sticky 레버 estimand가 **동거(co-residency) 시간이 아니라 단독-at-D 시간(`S_solo`)만** 재는 것으로 확인(`a6_estimand_structure.py`) — 손잡이가 설계 의도와 다른 양을 조작. (c) `M_itl`(요청별 token-ITL p95의 중앙값) estimand가 **이봉 분포에서 검열**됨 — `U` 위 p95는 0.341ms인데 요청별 평균은 2.062ms로 대표성이 없다(`a6`·`a8`). `K1` 순위 규칙도 블록 수 N이 늘수록 식별 확률이 **떨어지는 반직관 성질**(`a2_block_power.py`) 발견. ★**"gate #16을 닫았다"고 쓰지 말 것**(불변, G17은 §1-32/§1-33 재정식화판을 더 좁힌 하위 시도). 상세 `handoff-report/session_handoff_2026-08-18.md` |
    | ★★E-B1 shadow price (`workspace/engine-port/reports/DESIGN_EB1_SHADOW_PRICE_2026-08-18.md`) | ★★**규칙층 NO-GO**(2026-08-18, gate #34 stage 1 감사 — `audit_eb1_rules_2026-08-18/` window_exists 등 7스크립트) | 死因: **판정 가능 창이 대수로 공집합**. `max_running_requests=48`이 모든 28부팅에서 decode 배치를 하드캡해 HI(12 req/s)가 이미 포화(최악) ITL 분포를 관측하는데, 포화 시 ITL-p95 실패율 `q=P(ITLp95>60\\|saturated)`가 arm별 ≈0.01–0.10(d34 최저)로 **거의 전부 5% 미만** — `window_exists.py`가 "어떤 rate에서도 조정 가능 창 진입 불가"(`ITL_AXIS_FEASIBLE_AT_ANY_RATE=False`, 다수 arm)를 산출. rate를 낮추면 포화 모집단이 희석돼 `q`가 더 내려갈 뿐이라 구제 불가능. 상세 `handoff-report/session_handoff_2026-08-18.md` |
    | ★★★switch-cost 재분석(`workspace/engine-port/results/kernel_mech/{DESIGN_SWITCH_COST_2026-08-22.md, STEP0_SWITCH_GAP_2026-08-22.md, PROMOTION_DRAFT_SWITCH_2026-08-22.md}`) | ★★★**GPU 0 · Step 0 헤드라인 `REFUTED`(claims-auditor) → 승격안 rev2 `조건부 승격` → 6문장 확정 등재(2026-08-22)** | 死因 없음(설계 자체가 재분석으로 성공) — 대신 초판 헤드라인(`Δmed=0.910ms`="전환 기계 비용 상한")이 **비식별 논증으로 REFUTED**(인덱스 불변 경계가 전환 경계보다 큼), 승격안이 문장 6개(순서관계·가법 상한 `s≤0.04ms`·생애주기 기전·residency 10³배 등)로 대체·확정. **후속 캠페인 불필요 권고**(감사 원문) — `alternate` 엔진 패치 1건의 값어치가 하락한다: **엔진이 이미 공짜 자연 대조(인덱스 불변 adjust 경계)를 갖고 있었다**. 미측정 항목 3개는 원리상 남는다: 컨트롤러 구동 전환 경로·green→green 전환(0건 관측)·포화 운영점. 상세 `reports/CONSENSUS.md` §1 신규 행 34, §1-8/§1-12/§1-15. |
    | ★★★P0-A cudagraph replay SM confinement(`workspace/engine-port/results/bcg_probe/{PREREG_P0A_2026-08-22.md, p0a_rule_totality.py}`) | rev1→rev6, 규칙층 감사 **5회 연속**(매 회차 신규 차단) → ★**5회차 감사 권고로 규칙층 수렴 → §10 하네스 재작성·제출**(job **890893**, 2026-08-23, gpu43, GPU 0.054 GPU-hr) → ★★★**결과 도착 — 판정 `CONFINEMENT_PRESERVED_THROUGH_GRAPH_REPLAY`, claims-auditor `CONFIRMED with conditions`(조건 C1–C8)** | rev1: B1–B7·N1–N6(빈 census가 `PRESERVED` — 메인 세션이 넣은 집합 술어가 `TOL`의 **하한** 안전망을 제거). rev2: C1–C8(같은 밴드의 **상한**도 제거돼 있었음). rev3: D1–D8·Q1–Q8(1라벨 잡음이 최고가 판정을 만듦·`P5` 편향 서술 오류·T4가 약화 변이를 못 잡음). rev4: E1–E7·P1–P7(★**이력표가 하지 않은 수리를 했다고 적음**, E6). rev5: F1–F4·N1–N9(`S(eager_green_prefill)`을 결정량으로 승격했으나 **게이트 0·열거 1값**[F1] · 본문이 아직 옛 규칙을 지시[F2]). **rev6**이 F1–F4를 반영해 규칙층 마지막 판(감사 미실시, §10 하네스 재작성이 선행 — launch-shim·`eager_green_prefill` 레그·신규 게이트 3개·원자료 스키마 변경 3건, divergence 20건). 결정량 최종형 `E_attrib := (S(gg)\S(eg)) ∩ S(eager_green_prefill)`(카디널리티·문턱 0개, `LOST`는 귀속으로만 성립). `P4c`(green pair 서로소 ∧ `D` tile)를 게이트로 걸자 `UNDETERMINED (...WITHIN TARGET)` 라벨(도달 세계 0)과 합집합 항(삭제해도 0 세계 변화)이 구조적으로 불필요해짐 — 기준선 커버리지 누락은 이제 `NOPAIR`로 더 이르고 정직하게 잡힘. `p0a_rule_totality.py`(516,096 세계·54 검사 PASS, 프로젝트 venv 필수) 신규. **등재 가능한 결과 0건**(사전등록은 결과가 아님, GPU 미집행). **금지 문장**: "구멍 C가 닫혔다"·"P0-A가 R4에 답했다"(R0 사전등록 §9가 이 질문을 L1 전용으로 등재했고 이 프로브는 L0 토이 그래프). 상세 `handoff-report/session_handoff_2026-08-23.md` §2.1 ★★★**2026-08-23(2차 세션) 결과 반영**: `%smid` 라벨 집합이 같은 스트림 eager census의 34-라벨 집합과 정확히 일치(`Δ=∅`, 공허참 아님 — 33,480 block 관측 중 34-라벨 밖 0건). 부수 도구 타당성: green-context 스트림 위 CUDA 그래프 캡처·replay가 이 기판에서 가능(75/75, 단일 커널 노드·`pool=None` 한정). 교차 레그 2개(34/108)는 관측값만 — 기전 해석(H1 캡처시점 vs H2 replay 스트림) **비식별**, 해석 문장 금지. 정정된 과잉 진술 2건은 정정된 형태로만 인용("기판이 R0와 일치"→"검사한 두 축에서 R0와 불일치 없음", "capture WORKS"→§5 한정 병기 필수). 인용 대상 `p0a_verdict_890893.json`뿐(게이트 #56). ★**S3·G1-d는 여전히 닫히지 않는다**(L0 유사물, 서빙·cudagraph 미측정). 확정 문구 ①–⑦ 전문·감사 판정서 = `workspace/engine-port/results/bcg_probe/audit_p0a_result_890893_2026-08-23/VERDICT.md`, `reports/CONSENSUS.md` §1-1(Gate 1 블록, P0-A addendum). GPU 지출 0.054 GPU-hr·새 성능 판정 0건·등급 변경 0건·정책 순위 변경 0건. 상세 `handoff-report/session_handoff_2026-08-23.md` §2.1 ★★★**2026-08-24 F2 양성대조 addendum**: job **891612**(gpu43, 1분03초, exit `0:0`, GPU 0.017 GPU-hr) — 판정 `REPLAY_IS_THE_WRITER`. `capture_only`(replay 0회) → census 0개 / `capture_replay`(replay 25회) → 34개(890893 라벨 집합과 일치) — 같은 스트림, replay 유무 한 단계만 다름. ⇒ 890893 결과-audit이 남긴 유일한 전제(§5-C4, "census를 쓴 것은 replay다")가 **닫힌다**(890893의 집합 술어가 "아무도 replay하지 않은 텐서" 위에서 공허하게 만족됐을 가능성 배제). ★**닫는 것은 C4 하나뿐**: 성능 주장 0건·게이트 0건·§6 스코프(동시부하 0·캡처당 replay 1회·decode 절반만 등)에 대해 무언·구멍 C·R4 무답 — 위 금지 문장 전부 승계. shim은 `p0a_graph_sm_confinement.py`에서 subclass(재구현 아님, `CaptureOnlyShim.__mro__[1] is P0A.GraphLaunchShim` 자기검사로 확인), `GraphLaunchShim.__getitem__` 소스가 890893 실행 커밋(`0ba9394`)과 이 job의 HEAD(`61adc343`)에서 동일(doc-steward 독립 재확인: `git show <rev>:<path>` 함수 소스 추출·해시 일치). 인용 대상 `p0a_f2_verdict_891612.json`뿐(게이트 #56). GPU 지출 0.017 GPU-hr·새 성능 판정 0건·등급 변경 0건·정책 순위 변경 0건. 상세 `workspace/engine-port/results/bcg_probe/{p0a_f2_verdict_891612.json, p0a_f2_positive_control.py}`, `reports/CONSENSUS.md` §1-1 F2 addendum |
    | ★★S-6 telemetry-OFF 대조(1-arm 포크, `workspace/engine-port/results/slo_sched/{PREREG_S6_2026-08-22.md, s6_offleg_enumerate.py}`) | `S6_POWER`(1회차) + rev1→rev4, 규칙층 감사 **연속 `NO-GO`**(총 5회차 — S6_POWER 死因4 포함) → ★★**2026-08-23, 사용자 결정으로 트랙 보류(HOLD)**(§10, `PREREG_S6_2026-08-22.md`) | rev1: C1–C4·V1–V8. rev2: D1–D7("C1–C4는 닫히지 않았다"). rev3: E1–E9, ★**E4=실행 불가**(`export TELEM_RC` unset 상태로 `set -u` 사망, 재현: unset+`set -u`→`rc=127` 즉사). rev4: F1–F10, ★**F2=OFF 부팅 채택률 0**(`export TELEM_RC=""`가 `set -u` 사망은 고치나 `g16_grid.sbatch:802`의 `[ "$TELEM_RC" -eq 0 ]` 요구를 못 만족해 OFF 다리 100% `artifact_invalid` — 그대로 제출하면 3.146 GPU-hr 결정론적 전액 소실) · **F1=E5(직전 3회차 차단)가 이력표·§9 합격기준에서 소거**(라벨 오류가 아니라 **차단 자체의 소거**, 3회차의 "이력표는 수리 기록으로 신뢰 불가"가 한 단계 악화). 근본원인(설계층 판정): 7-arm G16에 구조적으로 하드와이어된 하네스를 1-arm 대조로 포크하려 했고 매 회차 새 거부지점 발견(`SMOKE` 19개 중 17개 arm 리터럴 고정·`N_EXPECT_ARMS=7`+`exit 1`·`g16_analyze.py:971` `telem_rc==0` 하드코딩·`TELEM_RC` 정수비교+rc 소비처 9곳·`:196 ${1:?}`·`:234 exit 3`·파일명 리터럴 자기해싱) — 작문 품질이 아니라 아키텍처 신호. 나가면서 해소: `σ_log` 7.4698 vs 7.4644는 불일치가 아니라 두 추정 경로(원척도 로그정규 항등식 vs 로그변환 직접 SD) — 7.4698이 보수적이므로 등록 보증 0.9273 불변. 보존물(재개 시 손실 0): 사전등록 4판·감사 판정서 4건·기계 열거기 `s6_offleg_enumerate.py`(자기검사 19/19, `--emit`이 감사 독립 재실행과 **바이트 동일**, 포크를 버려도 재사용됨). 재개 조건 §10.4(F1–F10 전 19행·아키텍처 결정[포크 유지 vs 최소 하네스 신설]·드라이런[정적 assert 아님]·`TELEM_RC=0` 명시 대입+rc 소비처 9곳 전수). **금지 문장**: "S-6가 계측 축을 분리했다". 상세 `handoff-report/session_handoff_2026-08-23.md` §2.2 |
    | ★kernel_mech rev7의 B6 eligible-window feasibility(`workspace/engine-port/results/kernel_mech/PREREG_B6_ELIGIBLE_WINDOW_2026-08-22.md`) | ★**판정 완료** — 예산 판단이 걸려 있던 이 항목 하나만 닫는다. ★**rev7 전체는 여전히 `NO-GO`**(B1–B5·B7 불변, NVTX 엔진 패치 선행조건 그대로) | 감사의 `P≈0.07`은 간극 하나의 확률이라 결정량이 아니었음이 판명 — 결정량(3셀 전부 ≥10창)으로 재계산하면 rev6 워크로드 그대로는 `P(3셀 전부 통과)=0.238`(prefill이 step 2개를 가릴 때), 전액 손실 확률 76%(감사의 경보가 옳았음). 등록안: `NP=CONC=16` batch-synchronous 라운드×5, 예측 20창/셀(요구 10의 2.0배), 추가 비용 ≤0.20 GPU-hr, (a)prefill-free·(c)batch>0·(d)정상상태와 배치 등가가 확률이 아니라 **구조로**(`ignore_eos` 기본 True) 성립. 창의 정의: **maximal prefill-free run 1개 = 라운드 1개**(125-step run을 20-step 창 6개로 세는 것은 부트스트랩 분산 과소추정이므로 **금지**로 등록). 대가: 동거의 완전한 소멸·도착 과정 소멸·배치 축 고정. **금지 문장**: "B6가 rev7을 열었다". 상세 `handoff-report/session_handoff_2026-08-23.md` §2.3 |
    | ★★kernel_mech rev7(전체) 규칙층 감사(`workspace/engine-port/results/kernel_mech/DESIGN_KERNEL_MECH_REV7_2026-08-23.md`) | ★★**`NO-GO`(2026-08-23, claims-auditor, 게이트 #34 1단, 차단 C1–C10) — ★死因 없음**(rev6와 같은 계열의 국소·명세층 NO-GO) | C1 B5 미해결(`A3_NEGATIVE_INTER`·`A5_CONTAINMENT` 둘 다 시뮬레이터·실엔진 양쪽에서 발화 불가 — 교훈 #67 재발, 표에 "발화 가능"이라 적었으나 하지 않은 수리) · C2 `HOST_DOMINATED` 라벨이 CI 없는 점추정 비교인데 등록 순서상 최우선(동률 발화율 0.600) · C3 §1.3의 "n 축은 별개 명제"가 등록 규칙에서 거짓 · C4 §2.2 변이가 주장을 테스트하지 않음 · C5 §3.3 강등하며 실패 결과를 삭제, 판정표는 재사용 금지 rev6에 위임 · C6 §3.5 "전수"가 또 전수 아님(3연속) · C7 §1.1 정밀도 판정이 답을 가정해야 성립 · C8 두-셀 BCa 가속항 정규화 버그(결론 영향은 노이즈 이하, 하네스 상속 금지) · C9 등록 순서 4단계 중 `decide()`엔 3단계 · C10 값어치 문장이 워크로드 변경 전 판정 상속(정본 C2는 `B≈9–12` 스코프인데 캠페인은 `B=16` 고정). ★**값어치 판정 — 2단 분할 구매 권고**: Stage 0′만 선행(P0·P3a·P6·P7 도구 타당성 사실, ≈0.2–0.4 GPU-hr)은 값한다, Stage A 12부팅+`U_infl` 4부팅(≈0.85–1.2 GPU-hr)은 Stage 0′가 3조건(반폭<점추정·부팅간 CV 문턱·`N_win≥10`) 통과한 뒤에만 조건부. **금지 문장**: "rev7이 C1–C10만 고치면 GO다"·"5연속 NO-GO ⇒ 트랙 종결"(rev3–rev5 死因은 구조적, rev6·rev7엔 그 계열 없음 — B6은 위 행에서 이미 판정 완료). 감사 도구 9종 `audit_kernel_mech_rev7_2026-08-23/probes/`에 보존. GPU 지출 0. 상세 `workspace/engine-port/results/kernel_mech/audit_kernel_mech_rev7_2026-08-23/VERDICT.md`, `reports/CONSENSUS.md` §4 |
    | ★★kernel_mech rev8 규칙층 재감사(`workspace/engine-port/results/kernel_mech/{DESIGN_KERNEL_MECH_REV8_2026-08-23.md, audit_kernel_mech_rev8_2026-08-23/VERDICT.md}`) | ★★**`NO-GO`(2026-08-23, claims-auditor, 게이트 #34 1단, 차단 D1–D9) — ★死因 없음**(rev6/rev7과 같은 계열의 국소·명세층 NO-GO) | ★★D1 C1은 닫히지 않았다 — rev7이 새로 놓은 `A5′_KERNEL_COUNT`가 자기가 겨눈 위험(귀속 오류)에 **무력**하다: 스텝당 커널 5개×42스텝에 균일 off-by-one 오귀속을 넣으면 `gap_frac`이 **+75% 틀리는데** 이상률 0.000으로 통과(`probes/pF2.py`), 등록 상한 0.10은 "거의 안 거부" 쪽이라 거부/수용이 위험 크기와 무상관. `:173` "(발화 가능)"·`:180` "반드시 튄다"는 검증 없이 단정한 문장(교훈 #67/AUDIT_DEBT §6과 같은 형태, **같은 항목을 수리하다 재발**) · D2 §3.5-18 분기가 설계 자신이 최고라 적은 구간(관측 CV 0.0592=커널이 안 줄어듦, `cv_boot=1%`)을 예산 게이트가 **버림**(게이트 #21 역방향 재발) · D3–D9(대표): `n_boot` 축 없는 표로 등록된 행동이 실행 불가·§1.2가 JSON 값과 불일치하는데 "rev7에서 불변"이라 서술(C8 수리가 정확히 예측하는 방향으로 어긋남)·§3.5 "전수 19개"가 실은 4연속 전수 아님(`EPS1_SCEN=0.55`·`cv_win`·`ETA` 등 미스윕이면서 헤드라인을 결정)·§1.4 몬테카를로 표류를 수리 효과처럼 3자리로 제시(구분 불가, 0.5–0.7σ)·`HOST_DOMINATED` 도달성이 `decide()` 1단계가 실행되지 않는 셀에서 측정(반증 실패: 게이트 통과 증인으로 교체하면 40/40 재현, 강등이지 소멸 아님)·Stage 0′ 가격이 Stage 0′가 사려는 미지수(nsys 오버헤드)를 포함. ★**반증 실패(rev8이 실제로 고친 것)**: C8은 진짜(이 트랙 **첫 모수화 변이 통과**, 변이본 S6a/b/c 전부 FAIL 확인, delete-one ε 이동이 해석적 인공물과 6자리 일치)·C2/C5/C9/C10 적힌 대로 닫힘·C1 구조적 선언 자체는 옳음(무엇을 대신 놓았는지가 문제)·정본 오염 0(`check_citation_stops.py` 0 violation)·`--selftest` 14/14 PASS. ★★**값어치 판정 — Stage 0′조차 아직 아니다**: `:384` "nsys 오버헤드 ★미측정"인데 `:361`은 Stage 0′를 0.2–0.4 GPU-hr로 값매김(가격이 미지수를 포함) — ★**Stage 0′는 오늘 구매 자체가 불가능**(P7=NVTX 패치 오버헤드, 그 패치가 미작성). **더 작은 `Stage 0″`(P0 + NVTX 없는 P3a, ≈0.05–0.1 GPU-hr, 엔진 패치 0)만 값한다**(아래 행 참조, 자체 규칙층 통과 필요). **금지 문장**: "rev8이 C1–C10/D1–D9를 전부 수리했다"·"Stage 0′를 사면 트랙이 열린다"·"5연속 NO-GO ⇒ 종결"·"`A5′`가 귀속을 검증한다"·"kernel_mech 트랙을 닫았다". GPU 지출 0. 상세 `workspace/engine-port/results/kernel_mech/audit_kernel_mech_rev8_2026-08-23/VERDICT.md`, `reports/CONSENSUS.md` §4 |
    | ★`Stage 0″` 사전등록 신설(`workspace/engine-port/results/kernel_mech/PREREG_STAGE0PP_2026-08-23.md`) | ★★**규칙층 감사 `NO-GO`(2026-08-24, claims-auditor, 게이트 #34 1단, 차단 E1–E11) — ★死因 없음**, GPU 지출 0, 미제출. ★**후속 = `Stage 0‴ A0`**(아래 행) | rev8 감사 §3의 값어치 판정이 지정한 하위-후보 — `kernel_mech` rev1–rev8 전 판본의 결정량이 서 있는 `K_set(k)`(NVTX 범위 안 decode green-ctx 스트림 커널) 집합이 **이 기판에서 구성 가능한지 아무도 재 본 적이 없다**. 예/아니오 3문항(Q1 nsys가 green-ctx 커널을 node row로 방출하는가·Q2 그 row가 stream/context id **와** launch correlation id를 **둘 다** 갖는가·Q3 decode 스텝 경계를 커널만으로 식별 가능한가[판정 규칙 없음, 서술]) + 상수 1개(K1, 캡처가 `T_step` 중앙값에 거는 벽시계 배수) — P0 + NVTX 없는 P3a만, 부팅 1–2회·Ha8·cudagraph-ON·`D=92`(또는 44)·decode 전용 라운드·nsys `--cuda-graph-trace=node`·≈0.05–0.1 GPU-hr·엔진 패치 0. ~~등록 중단 규칙: Q1 또는 Q2가 "아니오"면 `TOOL_CANNOT_DEFINE_K_SET` → **kernel_mech 트랙은 이 기판 한정으로 여기서 끝난다**~~ ★★**이 중단 규칙은 감사 E1이 반증했다(2026-08-24) — 인용 금지**(rev1–rev8 결정량 전부 구성 불가로 등재, NVTX 패치 미착수, P1 프로브가 Stage B에 대해 한 것과 같은 형태의 도구 타당성 종결) — 어느 쪽도 가설의 반증이 아니다(게이트 #21). **금지 문장**: "Stage 0″가 트랙을 열었다"·"kernel_mech 트랙을 닫았다"(결과 도착 전). 새 성능 판정 0건. 상세 `workspace/engine-port/results/kernel_mech/PREREG_STAGE0PP_2026-08-23.md`, `reports/CONSENSUS.md` §4 ★★**2026-08-24 규칙층 감사 결과 追記**: `NO-GO`, 차단 **E1–E11**, 死因 없음. ★**E1 — `TOOL_CANNOT_DEFINE_K_SET`는 종결 조건이 아니라 재설계 조건**이다(생존 경로 4개 열거: (a) graph-launch API row 귀속[decode 스텝당 `replay()` 1회, `cuda_graph_runner.py:1161` — ★NVTX 불요] · (b) `--cuda-graph-trace=graph`+`GRAPH_TRACE` · (c) `streamId` 단독 · (d) device-시각 포함 귀속). `Q1=아니오`가 죽이는 것은 **1차 결정량**이고 **`Q2=아니오`는 아무것도 죽이지 않는다**. ★**E2 양성대조 0건**(P1은 두 겹·R0는 3중 대조였는데 이 설계는 0) ⇒ 부착 오지정·조기 종료·export 실패가 전부 "Q1=아니오"로 오독돼 **배관 실패가 트랙 종결로 등재**될 구조 · ★**E4 하네스가 틀렸다**(Q1·Q2는 **엔진이 필요 없다** — 엔진 없이 green ctx를 만들어 완주한 프로브가 저장소에 **셋**) · E5 `TOOL_CAN_...` 라벨도 과잉(캡처 시점 조인이면 필드는 있어도 멤버십 불가) · E6 등록값 부재 · E7 프로파일러 스위치·부착 대상 미등록(★서버/클라이언트 어느 쪽을 감싸는지 미등록 = 최대 위험) · E8 K1 순환 · ★**E9 근거 grep 스코프 오류** · ★E10 규칙이 산문뿐(전날 등재한 게이트 #66의 **즉시 재발**) · E11 하네스층 감사 생략. ★**감사가 GPU 0으로 Q2의 절반을 이미 답했다**(게이트 #36 부분 위반: 사전등록 §5-2가 스스로 "가장 값싼 표적"이라 지목하고도 미수행) — nsys 2025.3.2 export 스키마에 `CUPTI_ACTIVITY_KIND_KERNEL(contextId, greenContextId, streamId, correlationId, graphNodeId)`가 있고 stock 리포트가 그 컬럼을 SELECT한다. **금지 문장 신설**: *"`TOOL_CANNOT_DEFINE_K_SET`는 트랙 종결 조건이다"* · *"Q2는 도구 문서로 이미 답이 나왔으므로 프로브가 불필요하다"* · *"nsys는 green-context 커널을 낸다"*(A0 실행 전 금지). 상세 `workspace/engine-port/results/kernel_mech/audit_stage0pp_2026-08-23/VERDICT.md`|
    | ★★**NVTX 근거 정정**(`workspace/engine-port/results/kernel_mech/NVTX_EVIDENCE_CORRECTION_2026-08-24.md`) | ★★**근거 무효 — 결론은 다른 이유로 유지**(2026-08-24, 메인 세션 직접 확인, GPU 0) | kernel_mech **rev5·rev6·rev7·rev8 + B6 사전등록 + Stage 0″ 사전등록 6개 문서**가 상속한 *"`pdmux.decode_step` NVTX 방출이 엔진에 없다 — `grep -rn nvtx src/` = 0건"* 이 ★**틀린 트리**를 봤다: `workspace/engine-port/src/`는 **PD-mux 오버레이**이고, ★**실제로 도는 엔진**(`sglang_engine_dev/python/sglang/srt/`)에는 NVTX가 **4개 파일**에 이미 있다(`server_args.py:616,5384` `--enable-layerwise-nvtx-marker` · `utils/nvtx_pytorch_hooks.py` · `model_executor/model_runner.py:1196` · `batch_overlap/operations.py`). ★**결론(쓸 수 있는 스텝-경계 마커가 없다)은 유지되지만 이유가 다르다** — 그 훅은 `module.register_forward_pre_hook`/`register_forward_hook`(**모듈 forward hook**)이라 cudagraph **replay**에서 host forward가 안 돌아 **스텝마다 발화하지 않고**, 애초에 **layerwise**(스텝 경계 아님)다. `pdmux.decode_step` 마커 자체는 양 트리 어디에도 **없음**을 재확인. ★★**귀결(감사 E1-(a))**: decode 스텝당 `replay()`가 **정확히 1회**(`model_executor/cuda_graph_runner.py:1155-1161`)이므로 CUPTI graph-launch row **하나**가 **스텝 경계(host 시각) + `K_set(k)` 소속(`correlationId`)** 을 동시에 줄 수 있다 ⇒ rev6–rev8을 가로질러 이 트랙을 막아 온 **NVTX 엔진 패치 선행조건이 불필요할 수 있다**(★**미측정 가설** — ★**정정(2026-08-24 rev2 재감사 N13)**: 이전 문장 *"Stage 0‴ A0의 Q1·Q2b가 실측으로 답한다"* 는 **거짓이며 철회한다**. A0는 **엔진을 돌리지 않고 합성 spin 그래프**를 재므로 이 가설에 답할 수 없다 — A0가 사는 것은 **도구층 절반**(nsys가 green ctx replay의 노드 행을 내고 귀속·조인이 되는가)이고, **엔진 기판 절반은 A1**에 등록돼 있다). **금지 문장**: *"엔진에 NVTX가 없다"*(거짓) · *"NVTX 선행조건이 사라졌다"*(실측 전) · *"kernel_mech 트랙이 열렸다"*(rev7·rev8 `NO-GO` 불변). 6개 문서에 정정 배너 삽입 + 원 문장 **취소선 보존**(삭제 아님). 교훈 = 게이트 #31의 **grep 층 변종**: *"결론이 우연히 살아있는 것과 근거가 타당한 것은 다르다"*(5개 판본이 확인 없이 상속). GPU 0·새 성능 판정 0건 |
    | ★`Stage 0‴ A0` 사전등록 신설(`workspace/engine-port/results/kernel_mech/stage0ppp/{PREREG_STAGE0PPP_A0_2026-08-24.md, stage0ppp_a0_rule.py}`) | ★**규칙층 초안(★미감사)** — 게이트 #34 1단 **대기**, GPU 지출 0, 미제출, 하네스 미작성 | Stage 0″ 감사 §4 재설계의 이행. ★**엔진 없음**(감사 E4 — Q1·Q2는 엔진이 불필요하고, 엔진 없이 green ctx를 만들어 완주한 프로브가 저장소에 **셋**) · ★**대조 3겹**(감사 E2 — L1 full-GPU eager[양성대조] · L2 full-GPU graph[노드 행 존재 **+ 기대 노드 수를 측정해서 정의**, green 무관] · L3 green eager[green 가시성] · **L4 green graph=본 조건**) · ★**규칙을 코드로 고정 + 세계 전수 열거**(감사 E10/게이트 #66 — `stage0ppp_a0_rule.py`, sha256 `57241ce6010e32d3…`, **8,192 세계 × 라벨 11개 전부 도달 가능 · mutant 12개 전부 load-bearing(파라메트릭 약화 포함) · 판별 검사 3개가 지정 mutant에서 실제로 실패함을 `T8` 메타 검사로 실증**). ★**자기 검출 1건**: 초판 `T6b`가 **항등식**이어서 `T8`이 **첫 실행에서 잡았다**(게이트 #9의 17번째 재발, 이번엔 작성 중 자기 검출) → 위험한 방향(*ctx가 못 쓰는 세계가 ctx 근거로 **보고**되는가*)으로 교체. ★**설계 개선(2026-08-24, GPU 0 도구 문서 실측)**: `nsys profile --help`가 *"node를 고르면 graph 전체는 트레이싱되지 않는다"* ⇒ **`node`와 `graph` 입도는 상호 배타**이므로 4 다리를 **두 입도로 각각** 실행하고 `graph` 실행을 companion 규칙(32 세계·mutant 4개·판별검사 1개+메타)으로 채점 ⇒ ★`PRIMARY_ESTIMAND_UNCONSTRUCTIBLE`이 떠도 **E1-(b)의 생사를 이미 안다**. ★**답하지 않는 것**: `[:<launch origin>]`는 `host-only\|host-and-device`(호스트/디바이스 **코드** 기원)이며 **캡처/replay 시점을 가르지 않는다** ⇒ **E5 미해결**(이 오독을 명시 기록). ★**가격**: `--exclusive`·`--constrain=hwperf` **불필요**(CUDA API/커널 트레이스는 그 권한을 요구하지 않는다 — 그 권한은 `--gpuctxsw`·system-wide sampling·GPU metrics 전용) ⇒ 비exclusive `--gres=gpu:1`, ≈**0.03–0.05 GPU-hr**(선례 R0 0.0275·P1 0.032). 자유 모수 **18개 재열거**(Stage 0″의 "전수 6개"는 거짓이었다). **금지 문장**: *"`PRIMARY_ESTIMAND_UNCONSTRUCTIBLE`이면 트랙이 종결된다"*(E1 반증) · *"nsys는 green-context 커널을 낸다"*(★A0 실행 전 금지) · *"kernel_mech 트랙을 열었다/닫았다"*. ★**다음**: ① 규칙층 감사 → ② 하네스 작성 → ③ 하네스층 감사(2단) → ④ 제출. **②③ 없이 제출 금지**. 새 성능 판정 0건 ★★**2026-08-24 규칙층 감사 `NO-GO`(차단 X1–X13, ★死因 없음) 追記**: ★**정정(2026-08-24 규칙층 감사 X2)**: 초판 `T6b`는 **항등식이 아니었다**(`g_contra`가 깬다 — 메인 세션 재현 확인). `T8`이 잡은 것은 **검사↔지정 mutant 짝짓기 실패**이지 항등식이 아니며, 올바른 수리는 지정 mutant를 `g_contra`로 바꾸는 것이었다. *"게이트 #9의 17번째 재발을 자기 검출"* 은 **철회**한다 · ★**X1** 등록 문턱이 코드에 없다 — `JOIN_HIGH=0.95`는 `score()`에서 **한 번도 참조되지 않는 死코드**이고 `Q1_FRAC`은 **0.60/0.70/0.99 어느 값으로 바꿔도 자기검사 전부 통과**(메인 세션 재현) ⇒ *"규칙을 코드로 고정했다"* 는 **1차 문턱에 대해 성립하지 않는다** · ★**X3** `ctx=null\|parent` + `stream=match`인 세계(= green ctx 생성 실패·조용한 fallback)가 **최상위 긍정 라벨 `KSET_CONSTRUCTIBLE/stream_only`** 를 받는다 · ★**X4** companion이 부모 감사 E2(양성대조 0건)를 **그대로 재발** · ★**X5** 부분 export·절단·replay 일부 누락이 실질 라벨로 샌다(게이트 #21) · ★**X8** **전이 간극 미등록** — A0는 **합성 spin 그래프**라 **양성이 엔진 그래프로 전이되지 않는다**(A0+A1을 다 사도 2026-08-15 Stage 0 항목 1은 미구매) · ★**X11 선행 등록 승계 상실을 차단으로 채점** · X9 가격 **두 개가 함께 떠 있음**(0.03–0.05 vs `--time` 0.33) ⇒ 등록가는 **벽시계 상한 0.33 GPU-hr**. ★**값어치 판정**: A0는 **재도출이 아니라 지연된 이행**이며 초과분(Q2a/Q2b/대조 3겹)도 **과잉 구매 아님** — *"이 트랙의 가장 비싼 지출은 GPU가 아니라 이 프로브를 안 산 것"*. **X1·X3·X4·X5·X8을 닫은 뒤 사라**(전부 GPU 0). 상세 `workspace/engine-port/results/kernel_mech/audit_stage0ppp_a0_2026-08-24/VERDICT.md` ★**rev2 작성 완료(2026-08-24, ★재감사 대기)**: 감사가 구매 전제로 지목한 **X1·X3·X4·X5·X8** + 나머지 8건 반영 — 세계 **8,192→414,720** · 라벨 **11→15** · mutant **12→18** · 판별검사 **3→12** · ★**mutant 커버리지 8/12→18/18**(신설 `T10`, 반증 가능성 실측: 12검사 중 9개 제거 시 커버리지 실제 붕괴, 공허 검사 0건) · `JOIN_HIGH` 死코드 해소 + 문턱을 리터럴로 재선언하는 `T9a`/`T9b` + 값 변이 mutant · `green∈{matched,mismatched,absent}` 판정 분기(`absent`→`PROBE_INVALID`) · companion에 `l2g` 대조 + `E1B_GRAPH_TRACE_UNAVAILABLE` · `profile` 벡터 + **L2′ 후행 대조** + `TRACE_TRUNCATED` · `CAPTURE_FAILS_ONLY_UNDER_GREEN` · §0-1 **전이 간극**(A0는 필요조건 스크린, 양성은 엔진 기판으로 전이 안 됨) · §0-2 **선행 등록 승계표** · 등록가 **벽시계 상한 0.33 GPU-hr** · 권한 축 · 자유 모수 **25개** · 행 선택 술어·기대치 산출식 등록 · 자기검사 아티팩트 `selftest_rev2_2026-08-24.{txt,json}`. 규칙 sha256 `7d61d6bc821f4c83…`. ★**rev2는 미감사 — "rev2가 규칙층을 통과했다" 금지.** ★★**2026-08-24 rev2 재감사 `NO-GO`(차단 N1–N13, ★死因 없음) + rev3 작성(★재감사 대기)**: 구매 전제 5개 중 **완전히 닫힌 것은 X8 하나**, 나머지 넷은 **부분**이었고 ★★**그 넷 중 둘의 검증이 공허**했다. ★**N2** — X3의 검사 `T14`와 X5의 검사 `T15`가 **기존 `T4`에 논리적으로 함의**(반례 **0/18,662,400**, 메인 세션 재현) ⇒ **한계 구속 정확히 0세계**인데 변경 이력표는 `T4`가 이미 구속하던 집합 크기를 *"n=276,480/304,128 세계 구속"* 으로 적었다 (**두 줄 철회**). ★**N1** — **분기 순서가 무방비**: 순서 변이 4종이 라벨 **3,840–18,816개**를 바꾸는데 자기검사는 전부 통과했고, 측정조건 라벨 5종이 서로를 가려 **트레이스가 잘리고 green이 아예 없는 세계가 `GREENCTX_INVISIBLE`**(= *"nsys는 green 커널을 못 본다"*, 도구 능력 부정)로 채점됐다(메인 세션 증인 재현) — **X5의 수리 안에서 X5의 병이 재발**, 게이트 #21의 9회차. ★★게다가 **등록된 판별검사 `T17`이 올바른 수리를 금지**하고 있었다. 그 밖: N3 문턱이 **밴드**로만 고정(`Q1_FRAC` `(0.80,1.00]` · `JOIN_HIGH` `[0.95,1.00]`) · N4 `ctx="parent"`(**적극적 반대 증거**)가 최상위 긍정 라벨 통과 · N5·N7 companion 미이식·다리 수가 문서 안에서 3가지 · N6 새 축 2개의 **관측 채널 미등록** · N8 자유 모수 #15 무값인데 결정량이 거기 걸림 · N9 eager fallback 악화 · N10 `.json`을 **커밋된 스크립트가 만들 수 없음** · N11 nsys 문장 2건 과잉 독해 · N12 승계표 선택 인용 · N13 금지문 정본 전파 실패. ★**반증 실패(감사가 못 깬 것)**: `T10`은 항등식 아님(삭제 실험 12개 중 9개서 붕괴, 문서 기술과 **글자 그대로 일치**) · 항등식 검사 0건 · 부정합 세계가 라벨 도달성을 안 떠받침 · ★**§0-2 승계표 인용 줄번호 전부 정확**(*"틀린 트리 grep"*·*"차단된 상수 재사용"* 패턴 **없음**) · §13 수치가 하나 빼고 전부 재현 · nsys 주장 6개 축자 확인 · **死因 없음**. ★**rev3 수리 완료(미감사)**: 세계 **414,720→663,552** · 라벨 15→**17**(`ATTRIBUTION_DISCONFIRMED`·`KSET_STREAM_ONLY_ATTRIBUTION` 신설) · mutant 18→**24**(★**분기 순서 mutant 3종** 신설 = N1의 실질) · 판별검사 12→**16**(배타성 `T20a–e`) · ★**메타검사 `T19`**(모든 검사 쌍 함의 전수 → **함의 0쌍**) · **`OK`가 `ctx+stream` basis를 요구**(N8 — stream 일치는 구조적이라 증거가 아니다) · companion **5 다리·6,912 세계** · 관측 채널 등록(★nsys 값으로 green을 판정하면 Q2a와 **순환**이라 금지) · 자유 모수 **27개** · ★**커밋된 스크립트가 `.json`을 쓴다**. 규칙 sha256 `ca7f0fb615cdcd65…`, `all_pass:true`·`uncovered_mutants:[]`·`entailed_pairs:[]`. ★**rev3 작성 중 자기 검출 2건**(`_reaches_join` 오정의·`T20a/b` 과잉 요구)을 스위트가 **첫 실행에서 잡았고**, 남은 1건도 **검사↔mutant 짝짓기 실패**여서 순서 mutant 도입으로 해소했다. ★**rev3은 미감사 — "rev3이 규칙층을 통과했다" 금지.** 상세 `workspace/engine-port/results/kernel_mech/audit_stage0ppp_a0_rev2_2026-08-24/VERDICT.md` ★★★**2026-08-24 3차 감사 `NO-GO`(R1–R10, ★死因 없음) — 그러나 ★★"구매 저지 사유는 소멸했다", 권고 = (a) 지금 사라** + **rev4 작성(권고 이행 완료, ★미감사)**. ★**수렴 근거(메인 세션 재현)**: 배관 실패(`export`·`green`·`L1`·`L2′`·절단)가 **도구 능력 라벨이나 실질 라벨을 받는 세계 = 0건 / 663,552**(rev2 순서로는 **44,064건**) ⇒ 게이트 #21 위반 경로가 측정 가능한 의미에서 소멸. 감사 판정: 심각도 분포가 rev1 ★★★6 → rev2 ★★★2 → rev3 ★★★1로 가벼워지고 ★**실패 방향이 전환**됐다(이전 잔여는 배관 실패에서 **거짓 주장을 제조**할 수 있었고, rev3 잔여는 **전부 측정조건 라벨로 무너진다**) — *"강화가 순손실로 넘어가는 지점이 이 판본"*. 단 **개수는 수렴 안 함(13/13/10)이고 매 회차가 자기 수리 안에서 새 결함을 연다(3/3)**. ★★**R1(가장 무거움, 메인 세션이 만든 결함)**: 순서를 무결성 먼저로 옮기며 `truncated()`가 `l2=="ok"`를 요구하지 않아 **`NODE_TRACE_UNAVAILABLE`이 정합 세계에서 한 번도 발화 못 함**(도달 2,688세계 전부가 *"L2는 비었는데 L2′는 살아 있다"* = 자기모순) ⇒ **A0가 사려는 가장 그럴듯한 음성이 재실행 신호로 나왔다**. ★**R3**: `T19`가 **도달 불가 배정을 포함한 곱공간**에서 세어 `T22`의 **단독 구속 0**(실패 96쌍 전부를 `T4`가 잡음)을 못 봤다 — **N2 형태가 N2 수리 안에서 재발**. ★R2 신설 라벨 4개 금지 문구 0건(X7 재발) · R4 순서 5종 생존 · R5 자유 모수 5개 무값(★#15 스트림 배치 값이 **코드 주석에만**) · R6 §7이 rev2 이후 **한 글자도 안 바뀜**(분모 fallback에 축·라벨·가드 전무) · R7 검증 문장 2건이 rev3에서 거짓(rev2 수치 무단 승계) · R9 정합성 재계산 부재(**그 검사가 R1을 잡는다**) · R10 `graph` 실행 launch origin 기본값. ★**반증 실패**: `T19` 공허 아님(고의 함의 4종 주입 전부 검출) · 2단 탐색 건전·완전 · 문턱 밴드 **(0.89,0.91]/(0.94,0.95]**(이산 축 최소폭) · 1차 추정량 안 죽음(`KSET_CONSTRUCTIBLE` 도달 4, 전부 정합) · `.json` **바이트 동일 재현** · §0-2 승계표 **전 줄 정확**(rev3이 개선까지) · 死因 없음. ★★**rev4 = 권고 (a) 이행**: 세계 663,552→**1,327,104**, ★**정합 세계 33,088 등록**, ★**정합 세계 도달 불가 라벨 2개→0개**, 라벨 17→**18**(`EXPECTATION_UNVERIFIABLE`), mutant 24→**27**(순서 mutant **5종**), 검사 16→**17**, ★**`T19`를 실제 평가 공간(926,464 배정)으로 이전 + `T19b` 단독구속 신설 → 함의 0쌍·단독구속 최소 4**, ★**`T22` 삭제**(단독구속 0이 두 번 나와, 억지 검사 대신 *"`g_eager` 검출자는 `T4`·`T9a`"* 로 정직 기록), `T18`·`T20a/b/c` **iff 승격**, 자유 모수 **28개**(값 5개 신규 등록), 신설 라벨 4개 금지 문구 등록, `graph` 실행 **`:host-only`** 명시. 규칙 sha256 `23b7335557fcff34…`, `all_pass:true`·`uncovered_mutants:[]`·`entailed_pairs:[]`. ★**rev4 작성 중 자기 검출 3건**(`T22` 단독구속 0 · **`T18` 실종**[`T10`이 표면화] · 요약 줄 변수 그림자) — 전부 **스위트/실행이** 잡았다. ★★★**하드 스톱 등재**: *"이것이 마지막 규칙층 감사다. 다음 적대적 감사는 ③ 하네스층이다."* ★**rev4는 미감사 — "rev4가 규칙층을 통과했다" 금지.** 상세 `workspace/engine-port/results/kernel_mech/audit_stage0ppp_a0_rev3_2026-08-24/VERDICT.md` ★★★**2026-08-25 하네스층 감사(게이트 #34 2단) `NO-GO`(B1–B15, 死因 없음) → 수리 → ★제출·완주**(job **892554·892556**, GPU **0.021 GPU-hr**, 등록 상한 0.33). ★**하네스층 감사가 3차 규칙층 감사의 예측을 네 축 전부에서 적중**시켰다(*"남은 위험이 사는 곳은 하네스: `l2_post`·`profile`·`ctx`·`stream`"*). ★**런킬러 2건(메인 세션 재현)**: `triton.jit`이 `exec` 생성 함수를 못 받아 **커널을 하나도 못 만들었고**(`ValueError: @jit functions should be defined in a Python file`), L2·L2′가 **default 스트림에서 캡처**해 PyTorch가 거부(`CUDA graphs must be captured on a non-default stream.`, libtorch 문자열 실측). ⇒ **그날 제출했으면 GPU를 쓰고 `MEASUREMENT_ABSENT`만 나왔다.** ★**어댑터 결함 5건은 전부 같은 형태 — 하네스가 규칙이 내려야 할 답을 조용히 결정**: `l2_post`가 **L4에서 파생**돼 L4가 비면(=등록된 결정적 음성) `TRACE_TRUNCATED`(**답을 지운 뒤 그 답을 말하는 걸 금지**) · `profile`이 규칙 상수와 불일치 + **1-노드 그래프**라 모든 부분 세계가 붕괴해 **`Q1_FRAC`이 한 번도 평가 안 됨** · `ctx="parent"`를 *"green id가 여러 개"* 로 매핑(nsys 자신은 **0/NULL로 "green 아님"을 인코딩**하는데) ⇒ 등록된 반대 증거가 최상위 긍정 라벨로 감 · `l2_join`이 공허해 **R6 수리 전체가 도달 불가** · companion **호출 0회**로 `graph` 실행이 비용만 씀. ★**수리 검증 = 감사 지정 변이 4종 + 대조 2종 전부 뒤집힘**(L4=0→`UNCONSTR` · gctx=0→`DISCONF` · frac=0.900 균일→`OK` · L2 조인 실패→`EXPUNVER` · 건강→`OK` · replay 결손→`TRUNC`; **마지막 두 쌍이 갈리는 것이 X5가 `profile` 벡터를 도입한 목적**). `graph` 입도는 **이 제출에서 제외**하고 §4를 **A1로 이월**(GRAPH_TRACE에 커널 이름이 없어 L2/L2′ 분리가 B3와 같은 함정이 되고, 시험 불가 스키마에 맞춰 쓰는 것이 감사가 방금 벌한 실수 · 전이 간극이 E1-(b)에도 동일 적용). ★★**결과 = `KSET_CONSTRUCTIBLE`, basis `ctx+stream`**: L4 노드 행 **100/100**(`frac=1.000`, replay별 [5]×20) · `greenContextId = 4`(NULL/0 아님) · `streamId 149 == L3` · **조인율 1.000**(replay 100 / 캡처 0) · `green: realized 34 = target`(프로브 자신의 드라이버 호출, **nsys 아님**) · 대조 4겹 전부 건강(**L2′ = 100**). ★basis가 `ctx+stream`인 것이 핵심 — N8 이후 `OK`는 stream 단독으로는 안 난다. ★★**부모 감사 E5의 세계(필드는 있으나 캡처 시점에만 조인)는 이 기판·이 그래프에서 관측되지 않았다.** ★★★**첫 실행 892554가 더 중요할 수 있다**: 모든 축이 동일하게 깨끗했는데 **`TRACE_TRUNCATED`** 가 나왔고 원인은 `export="partial"` 하나 — **내 sbatch 순서 버그**(`nsys export`가 먼저 `.sqlite`를 만들어 `nsys stats`가 *"Use --force-export=true"* 로 exit 1). ⇒ **하네스가 배관 실패를 실질 판정으로 바꾸지 않고 측정조건 라벨을 냈다 — 게이트 #21이 실제 실행에서 처음으로 지켜졌다**(이 트랙이 판본 6개를 잃은 방식이 정확히 그 반대였다). **금지 문장**: *"kernel_mech 트랙이 열렸다/닫혔다"*(rev7·rev8 차단 전부 불변) · ★*"A0가 통과했으므로 엔진 decode 그래프에서도 node row가 난다"*(**전이 간극** — 노드 5 vs 수백, 캡처 경로, 동시 green 스트림 2개) · *"NVTX 선행조건이 사라졌다"*(스텝 경계 **미측정**, E1-(a)는 여전히 가설) · *"E1-(b)를 샀다"*(`companion_scored: false`) · *"구멍 C/R4에 답했다"* · *"`Q1_FRAC=0.90` 문턱이 시험됐다"*(`frac=1.000`이라 **문턱 근처 미관측**). ★**쓸 수 있는 문장은 기판 한정이다** — *"이 기판(A100/driver 580.105.08/CUDA 13.0.2/nsys 2025.3.2)에서, **엔진 없이 만든 합성 spin-커널 그래프**에 대해"* 를 떼면 안 된다. **다음 = A1(엔진 기판 Q1/Q2 재측정)**. 새 성능 판정 **0건**·HE0 불변. 상세 `workspace/engine-port/results/kernel_mech/{audit_a0_harness_2026-08-25/VERDICT.md, stage0ppp/RESULT_A0_892556_2026-08-25.md, stage0ppp/a0_verdict_892556.json}` ★**정오표(2026-08-28, TC1 rev1 규칙층 감사 §3.1이 발견, doc-steward 등재 — 파일은 동결이라 무수정)**: `stage0ppp_a0_rule.py:2`의 docstring `"fixed in code (gate #66)"`은 **오인용**이다 — `PROJECT_STATUS.md` 게이트 #66은 *"포크는 원본의 하드와이어를 상속한다"*(S-6)이고, rules-as-code 규율의 정본 위치는 **`CONSENSUS.md` §3 항목81 / 이 문서 게이트 #61**(P0-A rev1–rev6 + S-6 rev1–rev4, 2026-08-23)이다. `rule_sha256`이 `a0_verdict_892556.json`의 채점 해시를 고정하므로 **제자리 수정 금지** — 올바른 인용은 이 정오표를 통해서만 확인한다. ★TC1 rev1의 정정문 자체도 **또 틀렸다**(항목66이라 적고 `VERIFIED` 딱지, TC1 rev2가 정정) — 상세 `workspace/engine-port/results/tc1_model_attrib/audit_tc1_rules_2026-08-27/VERDICT.md` §3 |
    | ★신규 트랙 NSL-1 admission lever(`workspace/engine-port/results/nsl_lever/DESIGN_NSL1_ADMISSION_LEVER_2026-08-23.md`) | ★**rev1 규칙층 `NO-GO`**(2026-08-23, claims-auditor, 게이트 #34 1단, 死因 F1–F3) → **rev2 작성**(F1–F15 반영 시도, ★미감사) | ★★**이름 충돌 경고 — `E-1`(`results/bsweep_regime/PREREG_E1_BSWEEP_REGIME_2026-08-14.md`, 규칙층 `NO-GO`)과 무관**하다, 이 트랙 이름은 `NSL`(Non-SM-split Lever)뿐. §5-8(b)(admission/KV-aware lever로 §1-4 얽힘의 死因을 직접 겨냥)을 겨냥. 死因: **F1** 결정량 `argmax_{(D,cap)} goodput` vs `argmax_D goodput\|cap=48`이 부분집합 포함관계라 `Δ≥0`이 데이터 관측 전에 강제(귀무 하 위양성률 2/3, `E[Δ\|H0]`가 노이즈와 함께 증가 — 게이트 #20의 거울상, G17 死因과 문자 그대로 같은 형태) · **F2** 등록한 arm×워크로드×SLO×메트릭 조합이 저장소에 존재한 적 없음(변화 trace·tight-SLO 하네스는 2.7B 하드와이어인데 rev1은 7B `Ha8` 선택, 근거로 든 "gate #13이 부팅 분산을 실측"도 거짓[σ_job≠goodput 분산]) · **F3** cap↔실현 D 앨리어스(정본 §1-25/§1-26: cap이 in-flight prefill 수를 바꿔 실현 D를 직접 움직임, `(D,cap)`은 요인설계가 아니었음) · **F4** 인용금지 위반(2026-08-01 E1 전제 실험 4건 870295/870296/870297/870301을 "이미 확립된 것"으로 인용 — `citation_stops.tsv`에 기계 규칙 신설, 신설 즉시 rev1을 소급 적발). [CS-OK] F5–F15는 경미(차단, 페어링·다중성·검정력·7개 동시 변경 등). rev2 수리: 결합 argmax 폐기(D 고정+TOST 등가여백)·arm을 Zamba2-2.7B로 복귀(변화 trace·tight-SLO 하네스가 실제로 도는 모델)·실현 D 분포 셀별 필수 보고. **금지 문장**: "NSL-1이 admission 축을 쟀다"·"cap 축이 무력함이 확인됐다"·"§5-8(b) admission 갈래를 닫았다"·"rev2가 F1–F15를 전부 수리했다"(F3는 완화이지 소멸 아님, F11은 결과로 등록했을 뿐). GPU 지출 0. 상세 `workspace/engine-port/results/nsl_lever/audit_nsl1_rules_2026-08-23/VERDICT.md`, `reports/CONSENSUS.md` §4 \| ★★★**2026-08-23 rev3 재감사 addendum**(`workspace/engine-port/results/nsl_lever/{DESIGN_NSL1_REV3_2026-08-23.md, audit_nsl1_rules_rev3_2026-08-23/VERDICT.md}`) — **`NO-GO`, 死因 H1·H2, 차단 H3–H11**(선행 판정서 rev1·rev2는 보존, 무효화 아님). ★**H1(死因)**: rev3 §1.2가 rev2 판정서의 표(attainment %(pp))를 **ITL 밀리초로 오독**하고 그 위에 "cap은 TTFT 다리에만 작용한다" 중심 논증을 세웠다(같은 표의 같은 행이 §1.2에선 `ms`, §3에선 `pp`인 내부 모순) — 실측하면 **HI(rate 12, chat 300/50, d44)에서 ITL 다리 통과율(12.08%)이 TTFT 다리 통과율(30.46%)보다 낮다**(전 arm 동일: d16 35.58/10.46·d24 34.83/10.04·d34 33.75/14.67·d44 30.46/12.08, TTFT/ITL). ★**H2(死因)**: 유일한 생존 전제("HI 동시성 44.19/48=92%이므로 cap이 문다")가 **미증명** — 그 44.19는 Little 법칙 **in-system 인구**(대기+실행)이지 `--max-running-requests 48`이 제한하는 running batch가 아니다. 저장소 로그가 이 구분을 증명: rate 10·12에서 in-system 동시성이 **63.96·83.06**으로 cap 48을 크게 초과. 나머지(대표): **H3** ① 거부 논거가 §1-14 과잉 일반화(rate 8만 해당, chat SLO에서 rate 6·7은 절벽 아님[정본 `interactive_slo_retune_plan.md:130-142`로 직접 확증]) — 단 ①의 결론 자체도 완전히는 못 산다(지표 안정한 rate에선 cap이 무력, cap이 물 rate에선 지표가 절벽 위 — 긴장이 rate 축에서 좁혀졌을 뿐, 해가 없다) · **H5** rev3 자신의 등록 상수로 검정력이 재계산돼 있지 않음(2.2–3.4배 과소) · **H7**(신규 자유 모수) 요청-내부 ITL p95 추정기(NR/최근접순위 vs IN/선형보간) 미등록 — 등록 δ=1.22pp인데 추정기 선택이 9.92pp를 움직이고 바닥 regime에선 arm 순서까지 뒤집는다. ★★**§7 정본 술어 서술(성능 판정 아님, `reports/CONSENSUS.md` §1-7 addendum 반영)**: 정본 변화-trace 하네스(`sharegpt_vary_bench.sbatch:88,92`, `interactive_bench.sbatch`)가 실제로 채점하는 다리는 게이트 #4의 p95가 아니라 `mean`-ITL이고, 이 운영점(정본 SLO 3s/60ms)에서 그 다리가 제거하는 양은 0.00–1.21pp뿐이라 정본 변화-trace goodput은 TTFT 통과율과 경험적으로 구분 불가하다(이 하네스 채점으로 "정책이 ITL 다리를 통해 goodput을 바꿨다" 주장은 지지되지 않음). ★★그러나 **HE0는 흔들리지 않는다**(세 갈래: (i) 정본이 이미 p95로 재채점했고 그 술어에서 ITL 다리는 지배적 판별항[순위 보존·강화] (ii) §1-17[tight SLO]은 rate 8에서 측정[바닥 regime 아님] (iii) rate 12×chat 300/50은 정본이 명시 기각한 조합) — 저장소가 같은 진단을 이미 두 번 낸 것의 세 번째 재발(`goodput≡throughput 항등`·`joint 0/192 완전분리`). §6(d): 요청-내부 ITL p95 자체가 monolithic prefill 때문에 58–60ms에 이봉 모드를 가져 **자기 metric cliff**를 가질 수 있음(PLAUSIBLE, CONFIRMED 아님 — 서빙 직접 개입 미실시). **금지 문장 신설 5건**: "cap은 TTFT 다리에만 작용한다"·"HI에서 ITL 다리는 안 문다"·"HI 동시성 44.19/48이므로 cap은 문다"·"rate 6–8은 정본이 절벽으로 판정한 대역"·"정본 goodput은 TTFT-only 지표였으므로 HE0가 흔들린다". 값어치 판정: "질문은 여전히 산다. rev3은 rev2보다 뒤로 갔다 — 사지 마라"(결함을 발견으로 승격한 것이 이 저장소가 가장 비싸게 배운 실패 형태, 게이트 #21). 다음 회차 권고 1순위: H2를 프로브로 승격(telemetry 켜고 `batch_is_full` 발화율 직접 측정, 1부팅≈0.05 GPU-hr). GPU 지출 0·새 성능 판정 0건·등급 변경 0건·정책 순위 변경 0건·HE0 불변(재확인). 상세 `workspace/engine-port/results/nsl_lever/audit_nsl1_rules_rev3_2026-08-23/VERDICT.md`, `reports/CONSENSUS.md` §1-1·§1-7·§3 항목87–89 ★★★**NSL P0 cap-binding 규칙층 감사 `NO-GO` — ★死因 3건(B1·B2·B3), 차단 B4–B13** (2026-08-24, claims-auditor, GPU 0). ★**B1(死因)** 등록 항등식 `running_bs ≥ cap`은 **이 부팅에서 실행 불가능한 분기의 가드**다 — `scheduler.py:2366`이 `and self.chunked_req is not None`인데 하네스가 `--chunked-prefill-size -1`이라 `:890-891`이 `chunked_prefill_size=None`으로 만들고 `chunked_req ≡ None`이 된다(**메인 세션 직접 확인**). 살아 있는 cap 경로는 `:2433`뿐이고 조건은 **`running_bs + \|can_run_list\| ≥ cap`** — 등록 항등식은 그 **진부분집합**(`\|can_run_list\|=0` 특수해)이다. 게다가 `batch_is_full`은 **래치**라 스냅샷이 읽는 reason은 **다른 `running_bs`에서 세워진 것**일 수 있다. ⇒ H2와 **동형**(측정된 적 없는 사건을 인접 관측량으로 치환), 편향은 `CAP_NEVER_BINDS` 쪽으로 고정. ★**B2(死因)** 추정기(개수 가중 vs **시간 가중**)가 미등록인데, 감사가 **저장소에 이미 있는 동일 arm·동일 플래그 로그**(`results/slo_sched/g16_blk{1..4}_d44_boot1_*_telemetry.jsonl`)로 계산하니 **개수 가중 0.0014 vs 시간 가중 0.087 — 58배 차이**로 **등록 문턱 X=0.05의 양쪽**에 떨어진다(4/4 부팅 전부). 기전: 스냅샷 간격 비균일(p50 1.95 ms, p99 214 ms), 구간 밀도 **118배** 변동, `drbs==0` 스냅샷이 **97.5%**. ★**엔진 소스 자신이 이미 경고**(`multiplexing_mixin.py:508-520` *"fraction-of-snapshots … is biased … TIME-WEIGHTED … unaffected"*) ⇒ 교훈 #61/#63/#72의 **4번째 재발**. ★★**두 수치 어느 쪽도 물리 판정이 아니다**(B1로 술어 무효) — **추정기 민감도의 증거로만 인용**. ★**B3(死因)** `--max-running-requests`가 이 빌드에서 **순수 admission 손잡이가 아니다**: `model_runner_kv_cache_mixin.py:223-229`가 `disable_radix_cache ∧ max_running_requests` 조건에서 **`max_mamba_cache_size = max_running_requests // dp`** 로 대입한다(**메인 세션 직접 확인**) ⇒ cap 48→24는 **admission cap · req_to_token_pool 슬롯 · mamba state pool · (mamba 메모리 반환으로) attention KV 예산 넷을 동시에** 바꾼다 — 하필 그 넷이 프로브가 분리하겠다는 두 축이다. rev3 §5 **P1**(`--max-mamba-cache-size` 고정)이 이미 겨냥했는데 **P0가 버렸다**. ★**B4** 사려던 격자의 **절반이 이미 저장소에 있다**(`prefill_admission_block_reason` telemetry **989개**, `g16_*_d44_boot1_*`이 arm·플래그·cap 일치) — 인용 0줄, **재구매**. ★B7 `cap`을 **CLI 인자**로 읽어 rev3 P3의 fail-open 수리를 **되돌림**(메모리 교훈 *"pin은 target 아닌 realized로 검증"* 과 정면 충돌) · ★B8 부팅 상수 `145.8 s`가 rev3 감사 **H10이 차단한 값**이고 실측치가 같은 저장소에 있다(**32.8 s**, n=10, **4.4× 과대**) · ★B5 양성대조가 **깨질 수 있는 것을 시험 안 함**(항등식 오류가 두 arm에 같은 방향) · B6 *"자유 모수 전수 7개"* **거짓**(최소 6개 누락) · B9 `N_min=200`이 **cap-bound 셀에서 여유 17–44%뿐** ⇒ 측정실패 라벨이 물리 신호에 붙을 위험 · ★B10 `blocked`+reason이 **아무 정보도 더하지 않는다**(3집합이 실데이터서 동일, `KVBLOCK=0` 4/4) ⇒ 결정량이 **점유 지시함수로 붕괴** · B11 변화 trace→정상 rate 전환 논증 없음 · B12 게이트 #34 **2단 생략**(`nsl_lever/`에 `.sbatch` 0개) · B13 게이트 0이 공허. ★**반증 실패(감사가 못 깬 것)**: 세 코드 사실의 **줄번호 전부 정확** · **관측자 효과 없음**(`observe_scheduler`는 읽기 전용) · **실현 cap = 요청 cap**(48 정확) · **n=1은 병목 아님**(CV≈4%) · **인용정지 0위반** · §4·§5의 자기 지목 정직. ★★**값어치**: *"질문(H2)은 여전히 산다. 그러나 이 프로브는 사지 마라"* — 권고 순서: ①**GPU 0 재분석 먼저**(기존 g16 telemetry에 **시간 가중 추정기를 사전등록**해 계산, ★B1 수리 전엔 판정 아님) → ②**B1 수리 = 엔진 패치**(`:2369`/`:2434`/`:2476` 사이트별 카운터, `PDMUX_HOLB_PATH` 선례처럼 기본 OFF) → ③**B3 수리 = 설계 형태 변경**(mamba cache 고정+배너 실측 기록, 또는 cap 축 대조 포기) → ④B2·B5 → ⑤나머지. ★★**B3는 P0뿐 아니라 NSL-1 트랙 전체의 전제(*"cap = admission lever"*)에 대한 지적이다.** **금지 문장 신설 6건**: *"`running_bs ≥ cap`이 cap 경로의 발화 조건"* · *"batch_is_full만 보면 배치 크기로 cap/KV를 가른다"* · *"이 프로브는 엔진 패치가 필요 없다"* · *"CAPBLOCK=0.0015이므로 안 문다 / =0.087이므로 문다"* · *"cap 24 vs 48 대조가 cap 효과를 분리한다"* · *"프로브가 `CAP_NEVER_BINDS`를 등재했다"*. ★**신규 교훈 후보 3건**(doc-steward 판단): (A) *"시간비"라고 쓰기 전에 **분모가 시간인지 확인하라**(교훈 #61/#63/#72의 4번째 재발, 판정 반전)* · (B) ***손잡이가 순수한지 초기화 경로 전체로 증명하라*** · (C) ***프로브를 사기 전에 저장소를 grep하라** — 격자 절반이 이미 있었다(선행 등록 승계의 데이터층 변종)*. **성능 판정 0건 · HE0 불변.** 상세 `workspace/engine-port/results/nsl_lever/audit_nsl_p0_capbind_2026-08-24/VERDICT.md` ★**NSL P0 ① GPU-0 재분석 결과(2026-08-24, GPU 0, 새 성능 판정 0건)**: 감사 권고 ①을 이행 — ★**추정기와 결정 규칙을 계산 전에 사전등록**(`PREREG_NSL_P0_REANALYSIS_2026-08-24.md`)하고 기존 g16 telemetry(n=4 부팅)로 계산. ★**핵심 발견**: 감사 B1이 지목한 참 술어 `running_bs + \|can_run_list\| ≥ cap`의 `can_run_list`이 **telemetry 44필드 어디에도 없다**(전수 확인) ⇒ **기존 데이터로 계산 불가**. 대신 **구간으로 묶었다** — 하계 `L: running_bs ≥ cap`(∵`\|can_run_list\|≥0`) ⊆ T ⊆ 상계 `U: running_bs + queue_depth ≥ cap`(∵`scheduler.py:2415`가 waiting_queue를 순회해 `:2459`를 부르므로 `\|can_run_list\| ≤ len(waiting_queue)` = telemetry `prefill_queue_depth`). ★**P0의 등록 항등식이 바로 이 하계 L이었다** — 즉 P0는 T를 하계로 **대체**하고 동치라 선언했다. **결과 = `BRACKET_DECIDES`**(사전등록 §3): 4부팅 전부 L·U가 `X=0.05`의 **같은 쪽**, **T ∈ [0.080, 0.119]**, 관측 불가 기여분 **최대 2.6–3.9 pp**. ⇒ ★*"패치 없이는 원리적으로 아무것도 못 좁힌다"* 가 **거짓**이나 ★★**사건 계수는 여전히 엔진 패치 필요**(사전등록 §1이 미리 등재한 두 한계 — `batch_is_full`이 **래치**라 이 구간은 **상태 조건**을 묶을 뿐 **발화 사건**을 안 묶고, 스냅샷 지점과 `:2433` 실행 시점이 다르다 — 이 재분석이 **닫지 못한다**). ★**B2 재현 + 확장**: 시간 가중 L=0.080–0.088(≥X) vs 개수 가중 L=0.0014–0.0015(<X), **58–60배**, ★이번에 **상계 U도 같은 방향으로 뒤집힘**(0.112–0.119 vs 0.0019–0.0020) ⇒ 추정기는 **구간 전체**를 문턱 반대편으로 옮긴다. 감사 수치와 **0.5% 이내 일치**(재현 성공). ★**cap=48은 서버 배너 실현값**에서 읽음(B7 준수, CLI 아님). **닫힘: B2·B4·B7** · **부분: B1** · **미닫힘: B3**(cap 축을 아예 안 씀으로 회피)·B5·B6·B8–B13. **금지**: *"cap이 문다/안 문다"*(B1로 점 술어 무효 — 이 수치는 그 판정이 아니다) · *"이 재분석이 B1을 닫았다"* · *"구간이 좁으니 `\|can_run_list\|`는 무시해도 된다"*. 상세 `workspace/engine-port/results/nsl_lever/{RESULT,PREREG}_NSL_P0_REANALYSIS_2026-08-24.md`, `nsl_p0_reanalysis_2026-08-24.json` ★★**E-A 결과(job 892561, 2026-08-25, GPU 0.103 GPU-hr, 요청 0건, 성능 판정 구조적 불가)** — 사전등록 `PREREG_NSL_EA_CAPBANNER_2026-08-25.md`(★계산 전 작성), 판정 **`ARITHMETIC_CONFIRMED`**(P1∧P2∧P3∧C1). 배너 실측: cap 24/48/96 → `max_mamba_cache_size` 24/48/96, `max_total_num_tokens` **337,039 / 327,601 / 308,727**, 실현 cap = 요청 cap(세 셀 전부). ★**P2 완전 선형 — mamba 슬롯당 정확히 393 attention-KV 토큰**(24→48 −9,438 · 48→96 −18,874, 두 구간 일치). ★★**C1 대조가 결정적**: `c48m96`(cap **48** + `--max-mamba-cache-size 96`)의 KV 예산이 **`c96`과 정확히 동일한 308,727**(오차 0)이고 `c48`과 다르다 ⇒ ★**KV 예산을 움직이는 것은 cap이 아니라 mamba pool 크기**이고 cap은 기본값으로 그 pool을 크기 지정하는 **입력**일 뿐이다. ★★**따라서 ③(손잡이 순화)의 처방이 실제로 작동함이 실증됐다** — `--max-mamba-cache-size`를 전 셀 고정하면 KV 예산이 cap과 **분리된다**(NSL-1 **rev2 §3**이 등록했다가 P0 프로브가 되돌린 그 설계). ★교차검증: `c48`의 327,601이 기존 g16 서버 로그 배너와 **정확히 일치** ⇒ 부팅 구성이 역사적 구성을 재현. ★**P3로 조용한 클램프 없음 확인**(감사 주장 3-(3)이 경고한 `estimated`(≤4096) 절단이 이 범위에서 발생하지 않음). ★★**B3는 닫히지 않는다 — 오히려 크기를 얻었다**: cap 48→24 대조는 KV 예산을 **+9,438 토큰(+2.9%)** 함께 움직이므로 mamba 고정 없이 cap arm을 돌리면 그만큼이 교란이다. **금지**: *"cap이 문다/안 문다"*(B1로 점 술어 무효) · *"E-A가 B3를 닫았다"* · **어떤 지연·처리량·goodput 문장도**(요청 0건) · *"cap을 올리면 크래시한다"*(c96 정상 부팅) · *"실현 cap은 항상 요청과 같다"*(**{24,48,96}에서만** — 범위 밖 이식 금지) · *"슬롯당 393 토큰이 모델 무관 상수다"*(**Zamba2-2.7B·ctx4096·mem-frac 0.82 한정**). 새 성능 판정 **0건**·HE0 불변. 상세 `workspace/engine-port/results/nsl_lever/{RESULT_NSL_EA_892561_2026-08-25.md, nsl_ea_banner_892561.json}` \| ★★★**2026-08-26 ③②E-B 묶음 + 그 트랙 첫 규칙층 감사 `NO-GO`, ★死因 4건**(`workspace/engine-port/results/nsl_lever/audit_nsl_bundle_2026-08-26/VERDICT.md`) — ③ `PREREG_NSL_STEP3_KNOB_PURITY_2026-08-25.md`(R1–R6, cap 술어 상수 = `pp_max_micro_batch_size`, `ratio=1`은 `--disable-radix-cache`에서만) · ② `DESIGN_NSL_STEP2_ADMISSION_SITES_2026-08-25.md`(`batch_is_full` 사이트 5개, pdmux arm 도달 2개, ★초판이 `:2369`를 "도달"로 적었다 자기 정정) · E-B `nsl_eb/PREREG_NSL_EB_ATTRIBUTION_2026-08-25.md`+`nsl_eb/nsl_eb_rule.py`(13,440 세계, cap 스윕 제외 한 셀 귀속으로 축소, 24→8 부팅). **死因**: **D1** `WORKLOAD_NOT_SATURATING`이 실질 성공(등록 rate-3 부팅 4개, `g16_blk1_d44_boot1_884336` 재집계로 확인 — `drbs≥48` 20s 구간은 HI(rate12) 창 3개뿐이라 정상-rate 셀은 전부 `INSUFFICIENT_BOOTS`)을 삼킴, ≈0.7 GPU-hr이 채점 불가 셀에 등록 · **D2** TOOLLIMIT 유일 술어 `SITES_INCOMPLETE`에 관측 채널 없음(페이로드는 발화만 기록·telemetry는 `reason=blocked`일 때만·`batch_is_full`은 래치) · **D3** 추정량이 **순서통계량**(cap 검사가 매 반복 최상단에서 평가·`break`해 KV 검사는 cap 불발 시만 도달 + `AddReqResult.OTHER` 출구가 `batch_is_full`을 안 세워 사이트 목록에도 안 잡힘 + `1.0-0.80=0.19999999999999996`으로 `cap_share` 0.2/0.8이 다르게 채점되는데 ★그 거울 대칭 검사를 "단독구속 0"(격자에 0.20·0.80이 없어서 생긴 인공물)을 근거로 직접 삭제) · **D4** cap 축 제거(§2가 강한 형태 대신 약한 형태를 씀)가 답을 미리 정함. ★**반증 실패(살아남은 것)**: ③ 코드 사실 전부 정확(`:228`·`:250-255`·`:390-402`·`:404`·`:862-875` 대조) · ③ R2의 `iff` 참 · ②§1.1 사이트 열거·도달성 전부 옳음(다섯/둘) · ②§1.2가 **자기 주장보다 강함**(`pp_max_micro_batch_size`는 클램프 이후 실현 cap이라 ①의 `cap=48`이 구조적으로 강건) · E-A 수치 전부 재현·정본과 충돌 0. ★★**§4.1 헤드라인 자체가 거짓**: A1 3회차 사슬표는 6 가족인데 §4.1은 4개만 열거했고 뺀 3개(상태오보·거짓금지문·출처허위)가 **바로 그 회차 런킬러**(R2·R3·R4) — "교훈을 옮긴다고 선언하면서 교훈 자체를 잘못 베낀" 사례, "수리는 국소, 주장은 전역"의 재현. **신규 방법론 교훈 후보 3건**(§⑥, doc-steward가 CONSENSUS §3 항목97–99로 등재 — 상세 위 최종 갱신 절 참조): (A) 추정량이 코드 경로의 평가 순서에 의존하면 귀속이 아니라 순서통계량 (B) 계측 완전성은 사이트 목록이 아니라 기전 목록에서 판정 (C) de-confound 처방 자체가 추정량의 자유 모수일 수 있음(③의 `--max-mamba-cache-size` 고정이 KV 예산을 방향성 있게 이동). **금지 문장**: *"cap이 문다/안 문다"*(그대로 무효) · *"①이 잰 정본 셀이다"*(cap arm 하나뿐, ①·E-B는 mamba pool·KV예산·워크로드 형태 셋 다 다름) · *"NSL이 admission 축을 쟀다"* · *"②의 엔진 패치가 배선됐다"*(0줄) · *"묶음이 규칙층을 통과했다"*. 새 성능 판정 **0건**·HE0 불변. 권고 1–8(전부 GPU 0): 추정량 개명·`NOSAT` 조건부 강등·`unattributed` 전이 재정의·문턱 대칭 수리+거울 검사 복원·집계 규칙 코드화·정상 rate 2셀을 ①의 변화 trace로 교체·`M` 논증 등록·도구 확장(`check_doc_facts.py`가 표를 `LABELS`와 대조, `check_line_citations.py`가 bare `:NNN` 포착) → 배관 스모크 → 패치 → 하네스 → 규칙층 2단 → 제출 |
    | ★★★kernel_mech Stage 0‴ **A1**(엔진 기판, `workspace/engine-port/results/kernel_mech/{DESIGN_A1_ENGINE_SUBSTRATE_2026-08-25.md, a1/a1_q3k1_rule.py, audit_a1_rules_2026-08-25/VERDICT.md}`) | ★★★**규칙층 감사 `NO-GO` — ★★★死因 2건(A1·A2, 이 감사 사슬 최초)**(2026-08-25, claims-auditor, 게이트 #34 1단), GPU 0, 재설계 필요·미제출 | A0(합성 spin 그래프)가 등록한 전이 간극(§0-1)을 엔진 decode 그래프에서 닫으려는 설계. ★**死因 A1**: nsys `streamId`는 **리포트 로컬**이라 E-L3를 별도 부팅에 두면 `stream="match"`가 **원리적으로 불가** — 정합 세계 663,552 전부에서 `stream="mismatch"`가 `KSET_CONSTRUCTIBLE` **0건**(물리적 최선 세계도 `CONTRADICTORY_ATTRIBUTION`). ★**死因 A2**: E-L2와 E-L4가 **같은 커널·같은 이름**이라 A0의 행 선택 술어(고유 커널명)에 대응물이 없고, 남은 선택지는 (i) `streamId`/`greenContextId`=**Q2ae와 순환**(설계 자신이 금지) 또는 (ii) **미정의 telemetry↔nsys 시계 다리**. ★**런킬러 B1**: `decode_step_count`가 저장소 **전 telemetry에서 0**(`dual_worker.py:117-125`가 `PDMUX_DUAL_WORKER=1`에서만 발화, 이 설계는 런타임 모드 미등록 — 메인 세션 재현 확인). 그 밖 대표: **C1** `N_MIN_SPLIT_STEPS=200` 死코드(A0 감사 X1의 문자 그대로 재발) · **C2** ★**단위 오독 확정**(1 스냅샷=**16 decode step**, 교훈 #61/#63/#72의 5번째 재발) · **C5** `export="partial"`이 최상위 양성 라벨로 새고 **job 892554가 정확히 그 세계**에 떨어졌었음(A0 하네스감사가 이미 수리한 것) · **D5** `K1_HIGH_AT=2.00`의 **출처 허위**(rev8에 `K1` **0건** — 게이트 #31의 규칙 정본 파일 내부 재발). ★**감사 재설계 권고**: 死因 2건과 차단 절반이 "분할/무분할이 같은 런 안에서 같은 커널로 3% 듀티·95회 교차"라는 **한 뿌리**에서 나오고, ★★**`PDMUX_STICKY_PARTITION`**(구현 완료·correctness gate 전부 PASS·`E1_DECODE_REALIZED` 0.0839→1.0000, CONSENSUS 항목27(II))이 그 뿌리를 없앤다 — sticky ON이면 A2는 **부팅 분리로 소멸**하고 A1은 "같은 리포트 안 decode green 스트림"으로 재정의할 근거가 생긴다(단 신규 rule rev + 규칙층 재감사 1회 필요, "재감사 없음"은 포기). 권고 절차: ①§2·§5를 sticky 기반 부팅 분리로 재작성 ②`stream`·`green` 두 축의 엔진 응답자 명시 등록(`green`은 **서버 프로세스 안에서** `cuStreamGetGreenCtx`+`cuGreenCtxGetDevResource` 배선 필요) ③Q3 채널 결정(`decode_step_count` 모드 등록 vs `decode_iterations`, 어느 쪽이든 **비영 실증 먼저**) ④Q3 규칙 rev2(`N_min` 실참조·`export=partial` 가드·창별 축·검사 2개 이상[서로 다른 절편]) + K1 rev2(구간 4개·부팅쌍 축) + D1·D3 정정 ⑤규칙층 재감사 → 하네스 → 하네스층 감사 → 제출. **rev7·rev8 차단(B1–B5·B7/C1–C10)은 A0·A1 어느 쪽도 건드리지 않아 전부 불변.** **금지 문장**: "A0 규칙이 A1을 그대로 채점한다/바뀌는 것은 어댑터뿐"(`stream="mismatch"` 663,552 세계에서 `KSET_CONSTRUCTIBLE` 0건) · "엔진 telemetry `stream_index`/`decode_sms`가 분할 **실현**을 말한다"(허용 축소형만: "어느 파티션을 **선택**했는지") · "`N_min=200`이 규칙층에 코드로 등록됐다" · "Q3가 성립하면 NVTX 선행조건이 불필요함이 실측된다" · "`K1_HIGH_AT=2.00`은 rev8이 등록한 정지선이다"(rev8에 `K1` 0건) · "kernel_mech 트랙을 열었다/닫았다". 새 성능 판정 **0건**·HE0 불변. 상세 `workspace/engine-port/results/kernel_mech/{DESIGN_A1_ENGINE_SUBSTRATE_2026-08-25.md, audit_a1_rules_2026-08-25/VERDICT.md}` \| ★★★**2026-08-25 sticky 재설계 rev2 + 규칙층 감사 3회, 전부 `NO-GO`**(`workspace/engine-port/results/kernel_mech/{DESIGN_A1_REV2_STICKY_2026-08-25.md, audit_a1_rev2_rules_2026-08-25/VERDICT.md, audit_a1_rev2_rules_3rd_2026-08-25/VERDICT.md}`) — 死因 A2는 조건부 소멸(부팅 분리로 시간창 조인 불요)하나 **死因 A1은 자리를 옮기지 않았다**(A0 규칙 `RULE_REV=4`가 미개정, 1차 추정량 신규 규칙 `a1/a1_primary_rule.py`도 그 자리를 못 채움). ★**신규 런킬러**: A0 규칙이 개정 안 됐는데 문서는 개정됐다고 보고(1회차 B1′) · 게이트 #21이 `launches` 축·`primary`에서 재개방(1회차 B2, 2회차 P2) · §5.1a가 반증된 전제(21/21 sticky 부팅 ≥0.9956, 완화폭의 1/10–1/50) 위에서 게이트를 5% 느슨하게 열었다 철회(1회차 B3) · 문서 자기모순 3건(1회차 B4)·1회차와 다른 3건(3회차 R2·R3) · ★**"Ha8+sticky 부팅은 선례가 없다"가 거짓**(1회차 B5, job **872800**이 그 job이고 rev2가 인용한 `0.0839→1.0000` 수치가 바로 그 파일의 두 줄) · ★**인용 도구가 거짓 인용을 인증**(2회차 P5 — `check_line_citations.py --snapshot`이 편집 파일 기준선을 조용히 갱신, 3회차 R4에서 재발) · ★**2회차 판정서는 파일로 저장되지 않음**(요지는 `DESIGN_A1_REV2_STICKY_2026-08-25.md` §6.8·§7 + 커밋 `3359fa3`/`7b7b007`/`189acfc`). ★★**3회차가 형태를 명명**: **"수리는 국소, 주장은 전역"**(사슬 판단표 — 전사검사만 수렴, 게이트#21/미등록축/상태오보/출처허위는 회차마다 "이동") + 부수형 "세계모형이 실험설계 성장을 못 따라간다"(규칙 세계모형 3부팅 vs 등록 5부팅). ★단 "무한반복 아님"(핵심 규칙 로직은 실행으로 전부 검증됨, 4회차 재실패 시 그건 문서 갱신 규율의 실패). ★★**GPU 실측 — A1 스모크 job 893663**(2026-08-26, ≈0.014 GPU-hr, `a1_smoke/RESULT_A1_SMOKE_893663_2026-08-26.md`, **채점 판정 0건**·배관 스모크) — 세 항목 PASS: (a) `E1_DECODE_REALIZED(16)=1.0000`(batch-sync, `t_decode_active=35.5s`, hist `D16`만 — §5.1a 모집단 간극을 닫음) (b) `decode_iterations` 0→415 (c) `PDMUX_GREEN_READOUT` 첫 실행, sticky decode 스트림에서 드라이버 `smCount=16` 확인(`GREEN_TARGET_CONFIRMED` 첫 관측, `D=16` 한 점·`D=92` 미관측). ★★**한정**: n=1·35.5초, *"batch-sync realized는 1.0이다"* 일반화 금지. **금지 문장 신설**: 위와 동일 계열 전부 + *"부팅 분리가 死因 A2를 닫았다"*(무조건형) · *"스모크가 통과했으므로 A1을 제출할 수 있다"*(규칙층 3회 전부 `NO-GO`) · *"rev2/rev3가 규칙층을 통과했다"*. 새 성능 판정 **0건**·HE0 불변. 다음 = 권고 절차 이행(상태표 정합·§5.1a 재작성·A0 규칙 rev5·Q3 규칙 rev3·부팅 매트릭스 수정) → 4회차 규칙층 감사 |
    | ★★★TC1(모델귀속, `workspace/engine-port/results/tc1_model_attrib/`) | ★★규칙층 감사 **3회, 전부 `NO-GO`**(2026-08-27·2026-08-28×2, claims-auditor, 게이트 #34 1단) — rev1 死因4·차단18, rev2 死因3(F5–F7)·차단12(B19–B30, 단일 판정질문 답="국소였다"), **rev3 死因5(F8–F12)·차단10, 단일 판정질문 답="네 번째 고리다"**(옳은 국소 수리 셋이 각각 새 강제를 만듦 — F8 상호작용 재정식화가 주효과 축을 지워 `both_lose`/`both_win`이 같은 라벨로 접힘·F9 사구간 재척도 확대·F10 `control` 가드가 nuisance로 강제·F11 도달가능성 메타검사가 ctrl 축에 무력·F12 봉쇄 상태 estimand 비식별). GPU 실측 1건(job **896565**, ≈0.20 GPU-hr, `F2_CONFIRMED`) + GPU 0 프로브 4건(F1 `REFUSED_AT_INIT`·B17 인덱스맵·rev3 도달가능성·**P3 `MEASUREMENT_ABSENT`**). ★**P3(jobs 896689/896690, ≈0.28 GPU-hr, Qwen2.5-3B 대조 프로브)** — 사전등록된 두 뿔(`BOTH_BLOCKED`/`ASYMMETRIC_TREATMENT`)보다 나쁜 **"셋째 뿔"**: `BIND=0∧FEAS=0`로 `MEASUREMENT_ABSENT`, 원인은 배선이 아니라 **물리**(Qwen HI phase TTFT p50 44.9ms vs Zamba2 1344.8ms ⇒ split-prefill 동시성 자체가 발생 안 함, `running_batch`·`split_prefill_batch` 동시 성립 조건 미충족) — 대조 arm 후보의 **공변량 수준차**도 큼(`level_gap`=40.2%, F10 치환창 6–9%를 크게 벗어남). ⇒ **모델 귀속 대조는 긴 프롬프트(동시성 강제)가 필요**하다는 실측 근거가 됨. ★★**모델 전환**(Zamba2-2.7B[ctx 4096, 긴 프롬프트 원천 불가]→**Nemotron-Nano-9B-v2-Base**[8.89B, ctx 131072, attn4/mamba27/mlp25], 대조 arm **Qwen2.5-7B**) — 부팅 스모크(jobs 896760/764/767)가 배관 확인: `triton` 부팅 거부→`flashinfer`로 `boot_ok=1`·cudagraph 캡처·긴 프롬프트서 컨트롤러 이동 2회(짧은 프롬프트 0). ★★★**백엔드 강제 교락 발견**(job 896776, `deprecated_v2/README.md`) — Nemotron-H 계열+`triton`=부팅 거부, Zamba2+`flashinfer`=스케줄러 사망(2/2) ⇒ 두 모델 계열의 동작 가능 백엔드가 **서로소**, "모델 고정·백엔드만 변경" 셀 불가능(신규 게이트 #83/§3 항목103). ★★도구 자기감사(commit `5d180a6`) — `design_reachability.py`가 자기 spec을 잘못 통과시켰음이 드러나 `RESTRICTIONS_INERT` 등 신설(`TOOL_REV 2`) ⇒ **rev3 spec은 이제 `DISCRIMINATING`이 아니라 `BLOCKED`**(구 판정 철회) | Zamba2 컨트롤러를 자기 Stage-1 argmax(d44)에 anchor시켜 model-attribution 대조를 세우려는 설계. rev1 F2: anchor=argmax로 고정하면 컨트롤러가 그 자리에 서 버려(§1-10 ratchet 재확인) `NO_FLIP_BOTH_LOSE`가 기전적으로 강제됨 — job 896565가 실측 확정(`SW=0`·`gpC=3.232`), 단 서버 로그 `SLO-FEAS refused=152`로 "죽음"이 아니라 "봉쇄"임이 드러남(rev2 재감사 B29, ★메모리항목21의 거울상 — 사전등록에 liveness 절 부재로 판정서가 자기 최강 증거를 안 적었었음). rev2: `visits_argmax` 관측 축 신설로 F2 봉쇄를 우회하려 했으나 강제의 **이름만** 바뀜(F5=`CONTROLLER_DEGENERATE`, d34↔d44 격차 1.5%<δ/2가 판정을 가름) + anchor 원복하며 δ=3% 미재도출로 최빈 결과가 `INCONCLUSIVE`(관측 −2.73%에서 P≈0.66, F6) + §7 세 대조 전부 반증불가(F7). 설계층 도달가능성(GPU 0, `design_reachability.py`): 시나리오 A(argmax=d44, 정본이 §1-7·§1-33 두 번 지지)=**`NOTHING_PURCHASABLE`**(캠페인 50 job 전체가 d34↔d44 추첨 하나에 걸림), 시나리오 B(argmax=d34)만 `DISCRIMINATING`. rev3(`ctrl_H=blocked`로 재정의, "퇴화" 대신 "시도·봉쇄")이 시나리오 A를 `DISCRIMINATING`으로 되돌렸으나 규칙층 재감사가 F8–F12로 **`NO-GO`** — 단일질문 답 "네 번째 고리다"(옳은 수리가 새 강제를 만드는 패턴이 3회 연속). ★**Zamba2 arm 자체가 이제 스코프 밖으로 이동** — P3가 드러낸 물리적 한계(짧은 프롬프트는 동시성 자체가 없음) + ctx 4096 한계 때문에, rev4 이후는 (착수한다면) **Nemotron-Nano-9B-v2 arm에서 새로 설계**해야 하며 Zamba2 rev1–rev3의 δ·anchor·estimand 결론은 **이전 금지**(§1-31 이식 금지 규칙 적용). **금지 문장**: "TC1이 규칙층을 통과했다"·"`PDMUX_STICKY_PARTITION`이 M3의 estimand 미식별을 해소한다"(컨트롤러 arm엔 미정의)·"`visits_argmax`가 F2를 닫았다"(이름만 바뀜)·"d44 anchor가 더 낫다"·"n=1로 정책을 비교했다"·"TC1 rev2가 `INCONCLUSIVE`를 피한다"·"TC1 rev3이 규칙층을 통과했다"·★"P3가 배선 결함이다"(물리임, F10)·★"모델을 바꾸면 TC1이 재개된다"(rev1–rev3 死因은 anchor=argmax 구조 자체에서 나와 모델 전환으로 자동 해소되지 않음, rev4 신규 설계 필요). 새 성능 판정 0건·HE0 불변. 상세 `workspace/engine-port/results/tc1_model_attrib/{audit_tc1_rules_2026-08-27/VERDICT.md, audit_tc1_rules_rev2_2026-08-28/VERDICT.md, audit_tc1_rules_rev3_2026-08-28/VERDICT.md, probes/f2_verdict_896565.md, probes/f1_verdict.json, probes/B17_ANCHOR_INDEX_MAP.md, probes/P3_VERDICT_896689_896690.md, probes/PREREG_P3_QWEN_CONTROLLER_2026-08-28.md, reach_verdict_rev3_A.json}`, `deprecated_v2/README.md`, `workspace/engine-port/results/model_roster/MODEL_ROSTER_2026-08-28.md` |
    | ★★★M4R(confinement, `workspace/engine-port/results/m4r_confinement/`) | ★★규칙층 감사 2회, 전부 `NO-GO`(2026-08-27·2026-08-28, claims-auditor, 게이트 #34 1단) — rev1 死因4(F1–F4)·차단16, rev2 死因4(F1′–F4′)·차단12(B1′–B12′). GPU 0 프로브 2건: alias(`EXPOSURE_ALIASED`, 28/28 셀)·`R_matched`(`RMATCHED_VIABLE` 7/7, 드레인 "포화"는 rev2가 반증). presubmit: `SINGLE_LABEL_FORCED`(BLOCK, 제출 금지) | §1-24(monolithic-prefill stall)·§1-26(B)(decode-SM 확대재현) 후속. rev1 F1: 노출변수 `R`(confined vs free 시간가중 rate 비)이 정본 등재 항등식(`decode_sms≠108 ⟺ prefill_active_batch_size>0`, §1-25 追記·§1-26(B))과 aliased — G16 confined 스냅샷 100.0%가 decode=D SM, SM 맞추면 R이 1.762→1.097로 붕괴·전 셀 `R_MIN` 미달·**부호가 108/D와 같은 모양으로 D에 단조감소**(§3(a) confinement 예측과 반대 방향). F2: 정본 §1-26(B) 선행금지·"GPU 0" 전제 붕괴(sticky 신규측정만이 경로). rev2가 `R_matched`(SM 맞춘 대조)로 estimand를 이관했으나 F1′: `sm_match` 가드 자체가 엔진이 강제하는 항등식(28파일 39,849구간 반례 0)이고 "SM 매칭"은 구간의 **왼쪽 끝점**에서만 성립 — 자유 다리(`free@D`)는 65–87%가 앞뒤 모두 108 SM인 **파티션 복원 지연 구간** 자체다(16-iteration 표집 묶음이 노출을 희석). batch 층화 시 108/D 서명 재출현(1.061–1.387, rev1을 죽인 서명과 동형). F3′: 7셀 중 6셀이 자기 가드(batch parity/CI/coverage)에 막히고 유일 생존 d24는 `RESIDUAL_INCONCLUSIVE`(블록 클러스터 t-CI 7/7이 1을 포함). F4′: 드레인 "25ms 포화"는 데이터 질량이 (0,200)ms에 정확히 0이라 생긴 측정불가의 오독(실제 연산=16-iteration 묶음 1개 삭제, 명목값의 10–14배) — 대칭 컷 적용 시 R이 1.067–1.595로 상승. ★★**불변 재확인**: `CONSENSUS §1-24`는 반증되지 않았다 — 두 감사 모두 "이 채널로는 그 질문에 답할 수 없다"고만 말한다. `R_matched` 1.00–1.16도, batch 층화판 1.06–1.39도 판정이 아니다. **유일한 해소 경로 = `PDMUX_STICKY_PARTITION=1` 신규 측정**. **금지 문장**: "M4R이 규칙층을 통과했다"·"M4R은 GPU 0이다"(★rev1이 이미 금지했고 rev2 §1 F2 행이 위반 — B10′, 인용금지 승계 실패)·"`sm_match`가 aliasing을 차단한다"·"`R_matched`≈1이므로 강등비용은 SM감소 그 자체다"(`RESIDUAL_ABSENT` 도달 불가). 새 성능 판정 0건·HE0 불변·`CONSENSUS §1-24` 불변. 상세 `workspace/engine-port/results/m4r_confinement/{audit_m4r_rules_2026-08-27/VERDICT.md, audit_m4r_rules_rev2_2026-08-28/VERDICT.md, probes/alias_verdict.json, probes/rmatched_verdict.json}`. ★★**정정 追記(2026-09-09, doc-steward)**: F1′의 "28파일 39,849구간 반례 0"은 **⟸ 방향만 참** — `longctx_conflict/audit_dutycycle_2026-09-08/VERDICT.md`(死因 R3)의 같은 28파일 전수 재집계가 `split∧pab==0` **1,643건**(35.8%)을 확인했다(`nonsplit∧pab>0`은 0건 그대로). 이 정정은 **스냅샷 기록 시점 동치**(record-skew)에 한정되며 코드 분기 술어·F1′의 `NO-GO` 판정 자체는 불변이다. `CLAIM_EVIDENCE_MATRIX.md:264`도 동일 정정 |
    | ★규율 도구 4종 첫 제출 게이트(`workspace/engine-port/scripts/discipline/{presubmit.py, presubmit_registry.json, PRESUBMIT_CHECKLIST.md}` + 루트 `scripts/discipline/design_reachability.py`) | 신설(2026-08-28), GPU 0, 성능 판정 대상 아님 | 감사 B11(2026-08-26)이 지적한 "도구 4종에 등록된 실행 지점이 없다"를 해소. `design_reachability.py`는 TC1 rev2·M4R rev2 두 재감사가 공통으로 요구한 메타검사 — 규칙 파일의 `T2_reachable`(격자 내부 성질)과 달리 **등록된 설계 + 이미 측정된 데이터가 실제로 낼 수 있는 실질 라벨**을 묻는다(모든 제약에 출처 문자열 강제, 출처 없는 제약은 거부). 현재 실행(`python3 presubmit.py --registry presubmit_registry.json`): `exit=1`(제출 금지) — `line_citations`(50건 비교 0 위반)·`doc_facts`(11 facts 14 occurrences 0 위반)·`citation_stops`(0 위반) 전부 `OK`, `reachability`(★2026-08-28 2차 갱신, `TOOL_REV 2`) `reach_spec_rev3_A.json→BLOCKED`(**`RESTRICTIONS_INERT`, 구 판정 `DISCRIMINATING`에서 하향**) / `reachability_spec.json→SINGLE_LABEL_FORCED`(BLOCK, M4R rev2). ★신규 방법론 교훈(아래 "방법론 게이트" #81): 도달가능성 검사가 격자 안에서만 돌면 설계가 답을 미리 정해 놓아도 통과한다 — 두 트랙에서 각각 독립적으로 발화. ★★**도구 자기감사(2026-08-28 2차, commit `5d180a6`)**: TC1 rev3 규칙층 감사가 이 도구 자신을 겨눠 결함을 찾았다 — `design_reachability` rev1은 `ctrl_H`를 `blocked`/`mispositioned`/`reaches` 무엇으로 고정해도 라벨 함수가 안 바뀌어(`design_sub`가 `grid_sub/3`으로 균일 재척도) `unreachable_by_design: []`를 냈고, 이는 **제약이 아무것도 못 지운 것을 "손실 없음"으로 오보한 것**이었다. `TOOL_REV 2`가 이 형태에 `RESTRICTIONS_INERT`를 신설(+ `RESTRICTION_DROPPED_UNEXPLAINED` — rev3 spec이 rev2 판정을 결정했던 `sign_H` 제약을 커밋 메시지 "same evidence"라 적고 조용히 뺀 것을 잡음 + `PRIOR_UNREGISTERED` — rev3 헤드라인이 3개 사전 중 2개에서 사전확률 ≈0을 지님을 미등록). ★**레지스트리는 이제 append-only**(과거 판정을 지우지 않고 `TOOL_REV`로 버전화) — 다음에 도구를 고칠 때도 이전 판정을 덮어쓰지 말 것. ★**경로 미정본화**: `design_reachability.py`가 저장소 **루트** `scripts/discipline/`에 있고 기존 도구 셋(`check_line_citations.py`·`check_doc_facts.py`·`check_citation_stops.py`)은 `workspace/engine-port/scripts/discipline/`에 있다 — **동시 세션이 그 경로로 호출 중이라 지금 옮기지 않았다**(doc-steward 소관, 다음 세션이 양쪽 세션을 조율해 처리). `presubmit.py`는 두 위치를 모두 찾아 검사한다. **금지 문장**: "도달가능성 검사를 통과했으므로 제출할 수 있다"(다른 死因이 독립으로 남는다). 상세 `workspace/engine-port/scripts/discipline/PRESUBMIT_CHECKLIST.md`, `workspace/engine-port/results/REACHABILITY_FINDING_2026-08-28.md` |
    | ★★★cp_baseline(chunked-prefill baseline, `workspace/engine-port/results/cp_baseline/`) | ★★규칙층 감사 **4회, 전부 `NO-GO`**(2026-09-01, claims-auditor, 게이트 #34 1단) — CP-1 rev1 死因9(`audit_cp_rules_2026-08-28/VERDICT.md`) · CP-1 rev2 死因7(**71%가 직전 수리의 그림자**, `audit_cp_rules_2nd_2026-08-28/VERDICT.md`) · CP-0 rev1(3회차) 死因6(`audit_cp0_3rd_2026-08-28/VERDICT.md`) · CP-0 rev3(4회차) 死因8(24건 중 **11건[46%]이 순수 전파 실패** — 게이트 #80 "수리는 국소, 주장은 전역"의 정량 재확인, `audit_cp0_rev3_4th_2026-08-28/VERDICT.md`). GPU 4 job **1.36 GPU-hr**(899768 0.248·900053 0.246·900054 0.139·900067 0.729, 전부 `COMPLETED 0:0`) — **새 성능 판정 0건**(전부 계측·축 검증) | 사용자 요청("정책 비교군 정리 + chunked prefill을 비교 대상으로 추가하는 설계")에서 출발, 규칙층 감사 4회를 거치며 정책 판정 트랙에서 계측·축 검증 트랙으로 내려앉았다. **P1(계측 수용시험)**: 1차(job 899768) `P1_BLOCKED_INSTRUMENT`(원인은 계측 로직이 아니라 하네스×텔레메트리 상호작용 — launcher가 SIGTERM 유예 없이 SIGKILL로 자식을 회수해 telemetry flush가 못 돎, `launch_server.py:64`→`utils/common.py:1054`) → 3중 수리(벽시계 flush·shutdown handler 체이닝·kill 시퀀스) 후 재실행(job 900053) `P1_ACCEPTED` 7/7 PASS(`verdict_measurement=MEASUREMENT_CLEAN`). ★**F3 기전이 관측으로 확인됨**(그전까지 코드 독해뿐): `--chunked-prefill-size`가 prefill 예산과 **piecewise CUDA graph 캡처 범위를 동시에** 결정(`server_args.py:1259`→`:1397-1415`→`model_runner.py:2486-2491`) — realized 캡처 목록이 **cps −1에서 len 0**(cps512 30/4/512·cps4096 50/4/4096, job 900053) 관측. ⚠️**스코프: Zamba2-2.7B·이 트리·arm당 부팅 n=1.** ★**다른 트랙 파급 가능성(관측이지 판정 아님)**: 정본 pdmux arm은 assert로 전부 `chunked_prefill_size=-1`이므로 piecewise prefill graph가 꺼진 채 돌았을 수 있다 — **이것으로 기존 어떤 결론도 수정하지 않는다**(캠페인 시점 코드판본 미확인·hybrid에서 piecewise가 실제 engage하는지 미검증). 한정 2건: (a) P1-e phase B가 `exact_on`이 아니라 최소 문턱 채점(체커의 3번째 비대칭 — 앞의 둘은 수리됨, `n_chunk_events` 기대3/실측2가 verdict를 못 뒤집음) (b) P1-f n=3, mean_rel_diff +0.00113은 마진(±0.03) 안이나 **95% CI 양끝이 마진 밖** ⇒ "중립 입증"이 아니라 "n=3에서 중립과 부합"까지만 인용. **G1(용량 축 식별가능성) 프로브**(job 900067, ShareGPT 2arm×5rate×**seed 3**×1job, 전 arm `--disable-piecewise-cuda-graph`): 등록 술어(`ATTAINMENT_THRESHOLD=0.80`, 격자{2,4,8})는 두 arm 모두 `CAPACITY_BRACKETED`를 내나 **달성 처리량이 최상단 rate(8)에서도 단조 상승 중**(fused_default 6→8 +12.4%·cp512 6→8 +7.0%) ⇒ **이 축·이 격자·이 임계로는 포화가 식별되지 않는다** — arm이나 정책에 대한 판정이 아니라 **우리 계측 축에 대한 판정**(4회차 감사 死因 G1·G2의 예측을 측정으로 확인). `band_vs_sd`(plateau 판정 술어)는 **미확정**(미등록 — 결과를 본 뒤 채우면 사후 튜닝이라 다음 판본 선결조건으로 등재만). **presubmit override 2건**(899768·900053, 사용자 명시 승인 — `workspace/engine-port/results/cp_baseline/OVERRIDE_P1_SUBMIT_2026-08-28.md`, 범위·금지사항 기록. M4R `SINGLE_LABEL_FORCED`·TC1 `RESTRICTIONS_INERT` 전역 차단과는 **별개**, 다른 트랙 소관 미변경). ★**날짜 정정(doc-steward, 2026-09-01)**: 이 트랙 산출물 다수가 파일명·본문 날짜를 **2026-08-28로 오기**(실제 작성/실행일 2026-09-01) — 커밋 4건(`fe8781b`·`5907a33`·`94fa01a`·`6a6b843`)·`presubmit_registry.json`이 그 경로를 인용해 **개명하지 않고** 각 파일 상단에 배너 + `workspace/engine-port/results/cp_baseline/DATE_CORRECTION_NOTE.md`로 정정. **금지 문장**: "P1이 chunked prefill 정책을 비교했다"·"F3 관측이 정본 pdmux 캠페인의 piecewise 상태를 확정한다"(미검증 파급 가능성일 뿐)·"G1이 chunked prefill의 처리량/goodput을 판정했다"(arm을 재지 않음)·"CP-0 용량 축은 원리적으로 불가능하다"(이 격자·이 임계 한정일 뿐)·"CP-0 본체가 제출 가능하다"(4회차 死因 **G2·G3·G7·G8 미해소**·**G4는 rev4에서 수리됨·미감사**[미해소 아님], 닫힌 것은 G1[측정]·G5[오라클]·G6[접기규칙]뿐 — ★2026-09-01 등재 시점부터 이 줄·`CONSENSUS.md`·핸드오프가 G8을 어느 쪽으로도 세지 않은 **장부 누락**이었음을 2026-09-03 `SWEEP_FINDINGS_CP0_REV4_2026-09-01.md` §1로 확인해 이 줄에서 정정). ★★**원래 질문(chunked prefill vs pdmux 정책 비교)은 이 트랙 전체에서 여전히 0건 측정**. ★★★**追記(2026-09-03, doc-steward)**: 이 트랙 **5·6회차 규칙층 감사, 둘 다 `NO-GO`**(누적 6연속) — 5회차(CP-2 rev1, `audit_cp2_rules_5th_2026-09-01/VERDICT.md`) 死因 **H1–H8**, 그림자 **7/8(81%)**; 6회차(CP-2 rev2, `audit_cp2r2_rules_6th_2026-09-03/VERDICT.md`) 死因 **F1–F8**, 그림자 **7/8(88%, 트랙 최고)** — 단일 판정 질문 답: *"수리의 절반이 조건 등록이 아니라 조건 삭제였다. 삭제는 감사에 안 잡힌다 — 없는 것은 grep되지 않기 때문."*(방법론 게이트 #88). 이 세션 전체 GPU **18 job 5.78 GPU-hr**(V-probe 8job/2.16 GPU-hr·W1 스모크 4job/0.50·모델 부팅 스모크 2job/0.36·W1 본캠페인 2job/1.56·correctness gate 2job/1.20 — 규칙층 감사 자체는 GPU 0). **새 측정(전부 서빙 실증, 전부 n 제한 있음 — 정책 판정 아님)**: (i) **V-probe**(900411·900423–900436, `RESULT_VPROBE_2026-09-01.md`) — job-내 페어 무처치 Δ SD `fused_default` **1.99%** / `d44` **6.13%** ⇒ 5회차 死因 H1 확증(오히려 과소평가 — 스모크 1쌍 +0.39%가 반대 결론으로 오도했다가 n=4에서 뒤집힘). ⚠️**arm×노드 완전 교락**(`fused_default`=gpu42 전부, `d44`=gpu40 전부) — **두 SD의 비교는 금지**. (ii) **운영점 재배치**(`RESULT_POSITIONING_AND_WORKLOAD_2026-09-02.md`) — 통과율 6%→52%로 옮기면 SD 11.4%→0.7%(16배)나 **arm-중립 운영점이 없다**(HI에서 `fused` 71.8%가 [70,80)ms 모드·`d44` 81.8%가 [40,60)ms 모드, 등록 임계 60ms가 `d44` 모드 상단 1.2ms 위 ⇒ 통과율로 운영점을 고르는 행위 자체가 arm을 고르는 행위). (iii) **ShareGPT 전수 센서스**(92,824행 전수, GPU 0, `RESULT_SHAREGPT_CENSUS_2026-09-02.md`) — `cps 8192` 초과 프롬프트가 캡 4000 하 **0건**·무제한도 **56건(0.1%)** ⇒ 정본 trace에서 chunked-prefill 처치가 **발화한 적이 없다**. (iv) ★★**W1**(Nemotron-Nano-9B-v2, 32/32셀, `RESULT_W1_2026-09-02.md`) — 이 트랙이 다섯 번 죽은 구조적 원인을 처음 측정: A트랙(ShareGPT 장문꼬리) knee 1.6–2.0 req/s·B트랙(random 16k) knee 0.32–0.47 req/s 둘 다 도달, **knee 아래 4 arm 무구별**. `d44`는 요청별 ITL-p95가 전 rate서 29–32ms 고정(TTFT 561→18,559ms)인 반면 cp arm은 반대(ITL 13.5→331–395ms) ⇒ **두 가족이 서로 다른 SLO 다리에 묶여 있어 등록 사다리 안에서 비교 부호가 뒤집힌다**(TTFT2000/ITL40=cp 우세 ↔ TTFT6000/ITL60=d44 2.4×). **n=1 seed·arm당 부팅 1회 — 순위 아님, 구조 관측**. (v) **correctness gate**(902397·902407, `RESULT_CORRECTNESS_GATE_2026-09-03.md`) — 기준 `plain` 대비 56 비교 중 **55 바이트 동일**, 퇴화 0/14, 장문(8k·16k) 전 arm 일치, 유일 예외는 청크수 비단조(계통 오류 아닌 근사 동률 argmax 뒤집힘) — 사용자 수용. ★**동시성은 미검증**(요청 순차 전송). (vi) **모델 지원 표**(900731·900752, `../model_roster/RESULT_MODEL_BOOT_SMOKE_2026-09-02.md`) — piecewise OFF 시 3모델 모두 장문(8k) 서빙, ★**게이트 #83의 백엔드 강제는 `nemotron_h` 계열에만 걸린다**(`granite`·`Falcon-H1`은 triton·flashinfer 양쪽에서 돈다, 로스터 미확인 칸 해소). **설계 결정(사용자)**: **piecewise CUDA graph 전 arm OFF**(`DECISION_PIECEWISE_OFF_2026-09-02.md`, falcon_h1/nemotron_h 캡처 크래시가 근거) — 부수효과로 **F3/G4 교락이 닫힌다**(`cps`가 다시 단일 레버, 강제번들 4중→3중, 대가=CP가족 prefill 최적화 1종 제거). **모델을 Nemotron-Nano-9B-v2+flashinfer로 전환**(W1 전용, 로스터 1순위) — ★**정본 정책 결과(Zamba2-2.7B/ctx4096)와의 직접 연결은 단절**(로스터 §4-3 요구대로 명시, W1 수치를 정본 arm 비교로 전이 금지). **도구·장부 결함 2건**: ★`presubmit.py`가 **read-only가 아님**(5회차 감사 자기신고 — 실행이 `m4r_confinement/reachability_verdict.json`·`tc1_model_attrib/reach_verdict_rev3_A.json`을 재작성, 그 부작용으로 **타 세션의 미커밋 작업이 소실**됐다가 이 세션이 재생성해 복구 — 6회차부터는 이 도구를 **돌리지 않음**). ★**S6 파급**: `check_version_sweep.py` **S6**이 발화해 `PREREG_CP0_2026-08-28.md`의 감사 회차 표기를 5→6으로 스윕 — `SWEEP_FINDINGS §5-5`가 **미리 예고한 항목**이자 예고된 전파 실패가 도구로 잡힌 **첫 사례**. ★신규 방법론 게이트 **#88**(삭제는 조건 등록이 아니라서 grep에 안 잡힌다)·**#89**(등록된 규율 도구도 read-only를 자체 검증하지 않으면 타 트랙 상태를 조용히 바꿀 수 있다) 신설. **금지 문장 추가**: "V-probe의 두 SD를 비교해 arm이 더/덜 시끄럽다"(arm×노드 교락)·"W1이 chunked prefill과 PD-mux 중 하나를 골랐다"(n=1·순위 아님·SLO쌍 미정의)·"TTFT 6000/ITL60에서 d44가 2.4배 낫다"(부호가 SLO에 의존함을 보이는 표 자체를 인용 금지)·"correctness gate가 동시성까지 검증했다"(순차 전송뿐). ★★새 성능 판정 0건·등급 변경 0건·정책 순위 변경 0건. HE0·정책 순위·gate #13/#16 "닫았다" 금지·switch-cost "닫았다" 금지·C2 인용정지 (a)(b)·`CONSENSUS §1-24` 전부 불변. ★★**원래 질문(chunked prefill vs pdmux 정책 비교)은 여전히 0건 측정.** 정본 반영: `CONSENSUS.md` rev52→**rev53**, "방법론 게이트" #88–89 신설. 상세 `workspace/engine-port/results/cp_baseline/{RESULT_P1_899768_2026-08-28.md, RESULT_G1_900067_2026-09-01.md, RESULT_VPROBE_2026-09-01.md, RESULT_POSITIONING_AND_WORKLOAD_2026-09-02.md, RESULT_SHAREGPT_CENSUS_2026-09-02.md, RESULT_W1_2026-09-02.md, RESULT_CORRECTNESS_GATE_2026-09-03.md, DECISION_PIECEWISE_OFF_2026-09-02.md, SWEEP_FINDINGS_CP0_REV4_2026-09-01.md, audit_cp_rules_2026-08-28/VERDICT.md, audit_cp_rules_2nd_2026-08-28/VERDICT.md, audit_cp0_3rd_2026-08-28/VERDICT.md, audit_cp0_rev3_4th_2026-08-28/VERDICT.md, audit_cp2_rules_5th_2026-09-01/VERDICT.md, audit_cp2r2_rules_6th_2026-09-03/VERDICT.md, DATE_CORRECTION_NOTE.md}`, `../model_roster/RESULT_MODEL_BOOT_SMOKE_2026-09-02.md`, `handoff-report/session_handoff_2026-09-01.md`, `handoff-report/session_handoff_2026-09-03.md` ★★★★**追記(2026-09-07, doc-steward)**: **AF-1 완주 = 이 트랙 첫 실질 라벨**(`results/cp_baseline/RESULT_AF1_2026-09-07.md`, 6 boot[seed 11·23·37·53·67·71], 전 boot gpu42, Nemotron-Nano-9B-v2-Base, ctx 32768, 동시성 1, arm=`plain`/`cp2048`/`d44`) — 규칙층 rev6이 이 트랙 최초 `GO`를 받은 뒤 하네스·스모크 2회(계측 결함 2건 수리: **층별 첫 요청 오염**[`agg_within_boot`이 준-최대라 첫 요청이 그대로 boot 값이 됨, `cp2048` q=0.50에서 16.51× — `WARMUP_REQUESTS_PER_STRATUM = 2` DESIGN 상수로 등록, 추정량 불변] · **§9-1 반증 + 파티션 무발화 기전 확정**[동시성 1에서 분할 그룹 `(64,44)`가 선택 불가해 `d44`가 SM 분할을 한 번도 안 씀 — 판독기 실패가 아니라 참인 관측, 신규 한계 **L-10** 등록])를 거쳐 등록 분석기를 **1회** 실행, 판정 **`STRATUM_DEPENDENT` → 분기 `b`**(사다리만 `intact`↔`pruned`, neutrality·anchor는 p90·p99에서 동일). 등록 예측 5건 중 3건(1·2·4) 틀림 — §5.1이 "틀리는 것이 결과"로 등록해 산출물. 분기 `b`의 등록 처방(사다리 전수 공표, `FOLLOWUP_STRATUM_2026-09-07.md` §나) 이행: 36칸×2층×2모드 전수 — **ITL 축은 어느 조합에서도 한 칸도 못 떨어뜨린다** ⇒ `STRATUM_DEPENDENT`의 실체는 TTFT 축 단 한 행. ★이월 항목 해소: `d44` q=0.99 부팅간 SD 52.69ms = `plain`의 **4.35배**(귀속 안 함, arm이 묶임). ★호스트 동거 의심 → 페어 측정(어차피 버릴 boot 활용) → 기각(12셀 중 10셀 −1.2%~+0.2%, 부호도 반대라 스파이크 동거 원인도 배제, 지출 0.27 GPU-hr). 이어 사용자 지시로 "층 고정 근거" 제안 rev1을 규칙층 감사에 부쳤으나 **`NO-GO`(死因 7건, `audit_stratum_ground_2026-09-07/VERDICT.md`)** — 단일 판정질문 답 "이름만 바꿨다"(자유표면이 q에서 모집단으로 옮겨감), doc-steward 자신의 오류(U4)도 발견: 제안 §7 금지문 #2가 死因 U3(등록 예측 4가 q=0.99에서 False→True로 뒤집힘, 게이트 #8 위반)로 직접 반증됨. 1차 출처 재조회(`FOLLOWUP_STRATUM_2026-09-07.md` §가) — Spheron 표는 실재·건전이나 본문이 "**P99 prompt length**"로 독자 자신의 값을 평가하라고 직접 지시(2차 조회, worked example 512 tok) ⇒ **경로 (a)(외부 앵커 좌표를 운영점으로 등록)가 닫힌 것은 우리 측정이 아니라 그런 좌표가 존재하지 않기 때문**(AF-1 풀 p99=5,925 tok는 예시의 11.6배, 死因 U2는 반증이 아니라 날카로워짐). 관련연구 조사(`RELATEDWORK_LONGCTX_SLO_2026-09-07.md`, 392줄) — long-context 전략 4유형(class별 절대 SLO/무경쟁 배수/TTFT를 SLO에서 제거/프로파일 길이곡선) 중 **인터랙티브 100–400ms를 long-ctx에 적용한 논문 0건** 확인, 감사 死因은 문헌과 정합. 워크로드 검토(`WORKLOAD_REVIEW_2026-09-07.md`) — AF-1 풀=코퍼스(92,886행) 상위 **2.12%**, 코퍼스 p99(=캠페인 2,514 tok)에서는 `anchor=multiple`(rag·batch 통과·chat 초과 3.2%) ⇒ **`unique`는 오직 AF-1 풀의 p99에서만 나온다 = 사슬이 아니라 풀 선택이 답을 정함**. ShareGPT 제외 시 로컬 후보 0개, 외부 후보(Azure/BurstGPT/LMSYS/ServeGen)도 독립적으로 같은 방향(일반 워크로드를 등록하면 `anchor=multiple`). ★신규 방법론 게이트 3건(`PROJECT_STATUS.md` #90–92 = `CONSENSUS §3` 항목110–112: **#90** 금지문 목록 자신이 거짓 진술을 담을 수 있다[死因 U4, 게이트 #75=항목95의 금지문 판본] · **#91** 모집단 선별이 답을 정함 — 분위수는 모집단을 명명해야 뜻을 갖는다[死因 U2, 게이트 #81=항목101의 모집단 판본] · **#92** 외부 앵커를 쓰는 트랙은 1차 출처 스냅샷 의무[재조회가 서베이 §2 인용보다 많이 찾아냈고 오인용 1건도 발견]). ★**정정 전파**: `serving_slo_survey.md` §2의 DistServe SLO-scale 귀속("무경쟁 단일 요청 실행 지연의 배수")이 1차 출처와 불일치함을 재조회로 확인(본문은 **Table 1 절대 SLO의 선형 배수**, *"there exists no available SLO settings"* 자기시인 — 무경쟁 배수는 Splitwise·Mooncake·LoongServe) — 서베이 문서에 정정 배너 부착, `CONSENSUS.md`는 이 구체 문구를 인용한 적이 없어 grep 확인 후 별도 수정 불요. GPU **0.88 GPU-hr / 11 job**(전 AF-1 스모크·본캠페인 포함) — **새 성능 판정 0건**. **금지 문장 추가**: "경로 (a)가 우리 측정으로 닫혔다"(그런 좌표가 존재하지 않아서 닫힌 것)·"`STRATUM_DEPENDENT`가 정책 판정이다"·"AF-1이 chunked-prefill vs pdmux를 비교했다"(계측·설계 판정일 뿐)·"8,192 토큰 초과가 측정됐다"(미측정, 32K 바닥 ≈3,785ms는 외삽 ×5.5)·"동시성 1의 `L-10`을 경쟁 하로 이식한다". ★★새 성능 판정 0건·등급 변경 0건·정책 순위 변경 0건. HE0·정책 순위·gate #13/#16 "닫았다" 금지·switch-cost "닫았다" 금지·C2 인용정지 (a)(b)·`CONSENSUS §1-24` 전부 불변. ★★**원래 질문(chunked prefill vs pdmux 정책 비교)은 여전히 0건 측정** — 8회 연속 규칙층 `NO-GO`(제안 rev1 포함)는 다음 세션이 트랙 축소(D5)를 진지 검토할 신호로 등재됐다(`DIRECTION_2026-09-07.md`). 정본 반영: `CONSENSUS.md` rev53→**rev54**, "방법론 게이트" #90–92 신설. 상세 `workspace/engine-port/results/cp_baseline/{RESULT_AF1_2026-09-07.{md,json}, AF1_HARNESS_ADDENDA_2026-09-07.md, PREREG_AF1_2026-09-04.md, PROPOSAL_STRATUM_GROUND_2026-09-07.md, PROPOSAL_STRATUM_GROUND_2026-09-07_VERDICT_ACCEPTED.md, audit_stratum_ground_2026-09-07/VERDICT.md, FOLLOWUP_STRATUM_2026-09-07.md, WORKLOAD_REVIEW_2026-09-07.md, DESIGNNOTE_LONGCTX_SAFETY_2026-09-07.md, RELATEDWORK_LONGCTX_SLO_2026-09-07.md, DIRECTION_2026-09-07.md, af1_discarded/README.md}`, `reports/serving_slo_survey.md`, `handoff-report/session_handoff_2026-09-07.md`. ★★★★**追記(2026-09-08, doc-steward)**: D1("운영점을 사지 말고 SLO 축을 sweep한다") 규칙층 3판본(rev1→rev2→rev3) + 적대 감사 3회, **전부 `NO-GO`**(트랙 누적 **10연속**) — 단일 판정 답이 자유 표면의 이동 경로를 이름 붙였다: **자(rev1 死因 U1–U8) → 추정량(rev2 死因 V1–V10) → 격자 크기 `m`(rev3 死因 W1–W10)**. GPU-0 선행 계산 A1(자 편향: TTFT −5.07%/ITL p95 +5.97%, 반대 부호 둘 다 3% 초과 ⇒ 단일 자가 등록 예측을 무부하에서 만들 수 있었음)·A2(D5 발동 조건 불성립, 단 Stage V n=2는 MDE 70.98%로 빈 게이트) 이행. ★★게이트 #14가 도구에서 정반대로 안내됨을 확정(`analyze.py`가 t-CI 부재·percentile bootstrap 권장·n 가드 없음 — 네 번째 재발, `FINDING_GATE14_TOOLING_2026-09-08.md`, 도구는 미수리·사용자 승인 필요). 메인 세션 자기 정정 1건(발화율 원인 진술 — 규약·부등호 동시 변경 오류, 등록 상수 5개는 불변). 신규 게이트 5건 #93–97(§3 항목113–117). ★★새 성능 판정 0건·등급 변경 0건·정책 순위 변경 0건·GPU 지출 0. ★★★원래 질문(chunked prefill vs pdmux 정책 비교)은 이 세션에서도 0건 측정. 정본 반영: `CONSENSUS.md` rev54→**rev55**, "방법론 게이트" #93–97 신설+#14 追記. 상세 `workspace/engine-port/results/cp_baseline/{PREREG_D1_REV3_2026-09-08.md, RESULT_D1_A1A2_2026-09-07.md, FINDING_GATE14_TOOLING_2026-09-08.md, audit_d1_rules_2026-09-07/, audit_d1_rules_2nd_2026-09-07/, audit_d1_rules_3rd_2026-09-08/}`, `handoff-report/session_handoff_2026-09-08.md`. ★★★★**追記(2026-09-08, 2차 세션, 같은 날 연속, doc-steward)**: 게이트 #14 도구 수리를 사용자 지시로 **별도 브랜치 `fix/gate14-tci-analyze`에 착수**(main은 원본 397줄 그대로) — t-CI 3함수 추가·CLI 판정 t-CI 전환·bootstrap은 병기, 검증(메인 세션 독립 재실행) 행동보존 5,100키 불일치 0·회귀 314 OK·변이 4/4 killed. 설계서 규칙층 감사 `NO-GO`(死因4·차단12) — 死因 F1(판정 경로 n 하한 부재, n=2 headline=True)만 수리(`fc87a17`), F2–F4·차단 12건은 열림. **"게이트 #14를 닫았다" 여전히 금지**. `reports/AUDIT_DEBT_2026-08-23.md` §10.3 P0-4 상태를 **부분 이행**으로 갱신. 신규 게이트 후보 3건은 `PROJECT_STATUS.md` "방법론 게이트" #98–100(신설) 참조. GPU 지출 0. 상세 `workspace/engine-port/results/tooling_gate14/{DESIGN_ANALYZE_GATE14_2026-09-08.md, BACKLOG_2026-09-08.md, CORRECTIONS_2026-09-08.md, audit_design_2026-09-08/VERDICT.md}`(브랜치 전용). |
    | ★`longctx_conflict`(`workspace/engine-port/results/longctx_conflict/`) | ★★규칙층 감사 **2회, 전부 `NO-GO`**(2026-09-08, 2차 세션, claims-auditor, 게이트 #34 1단) — rev1(`audit_l2_rules_2026-09-08/VERDICT.md`) 死因8·차단11 / rev2(`audit_l2_rules_2nd_2026-09-08/VERDICT.md`) 死因9·차단12(트랙 계열 누적 **12연속**). GPU 0(전부 사전등록·감사·기존 텔레메트리 재분석) — 새 성능 판정 0건 | 사용자 발의로 `CONSENSUS §5-6`의 long-context 전환 항목 재개 시도(트랙 우선순위화). ★**Stage 0 잔류 GPU-0 재측정**(정본 함수 `e1_pin_check::compute_decode_realized` 직접 재실행, 재구현 아님) — 스윕 27셀 realized **0.6986–0.9994**(≥0.95 20/27, rev1 §2.3의 "decode-only라 거의 실현 안 된다" 예측을 **철회**시킴) · D108 9셀 realized **≤0.0437**, 히스토그램 **16 SM 94–100%**(`D16≡D108`의 원인은 "둘 다 108"이 아니라 **"둘 다 16"**, §1-21/C1과 **같은 방향**). ★★**신규 confound `C-R`을 같은 세션이 스스로 강등**(`FINDING_BIN0_2026-09-08.md`) — "잔류 결손이 처치(ctx)·대조(arm) 축에 정렬된 기전"으로 등록했던 것이 실은 `compute_decode_realized`의 **구간 부과 아티팩트**(순간 상태에 스냅샷 간격 전체를 부과 — `T_C16384_D16` bin0 64개 전부 연속 런 길이 1·`prefill_active` 0/64·부과 dt median 0.4059s=전체 dt median 0.4063s ⇒ 25.70s). bin0 발생률의 arm 차이(T@16k 29.8% vs H@16k 0%)는 실재하나 **원인 미확정**. ★**감사 2F9(신규 등재)**: decode-only+prefill 유휴+sticky는 **PD-mux 운영점이 아니다**(성분 측정, `[[scale-8b-sm-sensitivity]]`와 같은 형태 — 레버 존재만, 정책 이득 아님, HE0 안 되살림). **방향 판정**(`DIRECTION_2026-09-08.md`): (ㄴ) **L1 직행(ITT)** 채택 — §5-6 stake #1("최적 static 위치가 움직이는가")에 직접 답, 노출=목표 설정값·`realized`=병기 공변량(게이트 아님)이라 `bin0`·표본화율·`0.90 vs 0.95`가 판정 경로 밖으로 빠짐(per-protocol이 12연속 `NO-GO`를 낸 자리 = 비순응의 원인이 처치 자신인 collider). ⚠️**ITT 재프레이밍은 미감사**, *"HE0가 ITT다"*는 메인 세션의 독해이지 정본 자기 규정 아님. (ㄱ) 축소 L−2는 보류 — 기각 사유는 비용이 아니라 감사 2F9(이전 "(ㄱ)은 비용 때문에 기각됐다" 진술은 **철회**, 실측 부하 10–40 GPU-h). **L−2는 여전히 "게이트 미실행"**(2026-07-28 C1 CONFIRMED 이래 상태 불변, 위 "다음 실험 gate" #7 追記 참조). **금지 문장**: "게이트 #14를 닫았다"·"C-R이 long-ctx 충돌의 증거다"·"장문에서 실현 할당이 무너진다"·"L−2가 binding/non-binding이다"(미측정)·"(ㄱ)은 비용 때문에 기각됐다"(철회)·"bin0는 파티션 복원 지연이다"(인과 미확정)·"T가 장문에서 decode를 더 자주 비운다"(반증, 전이 T4·H3·M3 동일). ★★새 성능 판정 0건·등급 변경 0건·정책 순위 변경 0건. HE0·정책 순위·gate #13/#16 "닫았다" 금지·switch-cost "닫았다" 금지·C2 인용정지 (a)(b)·`CONSENSUS §1-24`·job 893663 일반화 금지 전부 불변. 정본 반영: `CONSENSUS.md` rev55→**rev56**(§5-6 追記), `PROJECT_STATUS.md` "다음 실험 gate" #7 追記. 상세 `workspace/engine-port/results/longctx_conflict/{PRIORITY_2026-09-08.md, PREREG_L2_2026-09-08.md, PREREG_L2_REV2_2026-09-08.md, RESULT_RESIDENCY_2026-09-08.{md,json}, FINDING_BIN0_2026-09-08.md, DIRECTION_2026-09-08.md, audit_l2_rules_2026-09-08/VERDICT.md, audit_l2_rules_2nd_2026-09-08/VERDICT.md}`, `handoff-report/session_handoff_2026-09-08.md` "2차 세션" 절. ★★**追記(2026-09-09, doc-steward)**: 위 (ㄴ) L1 직행(ITT) 설계를 rev3까지 밀었으나 규칙층 감사 `NO-GO`(死因 3F1: 판정 면 `S_norm`이 ITL 결합항을 정규화하지 않아 도착률 규칙이 decode 동시성을 ctx와 ≈15× 공선으로 묶음 — `audit_l1_rules_2026-09-08/VERDICT.md`, 트랙 계열 누적 13연속). 대안 `PREREG_RATIO_2026-09-08.md`(rev4, 요청당 prefill:decode 일 비를 처치 수정자로)도 `NO-GO`(死因 R1: 처치 실현 듀티사이클 `f`가 앵커 셀에서 0.004, 등록 문턱 0.20을 51배 차이로 자기 기각 — `audit_ratio_rules_2026-09-08/VERDICT.md`, 트랙 계열 누적 **14연속**). 결과 감사 `RESULT_DUTYCYCLE_G16_2026-09-08.md`(기존 g16 telemetry 재분석, 사용자 지시 "3번 우선 진행")도 `NO-GO`(`audit_dutycycle_2026-09-08/VERDICT.md`, 死因3·차단7) — (B) 방향(듀티사이클 arm 단조)만 생존, (C)(`f∝1/prefill_SM` 정량 일치)·(D)(드리프트 0) 철회(死因 R3: `f≈ρ_pf` 전제가 같은 파일에서 반증 — g16 28파일 전수 `split∧pab>0` 2,951·**`split∧pab==0` 1,643**[split의 35.8%]·`nonsplit∧pab>0` 0, 기전=이벤트 루프 record-skew). ★★**이 트랙 최초 GPU 지출**: 계측 프로브 5건(`PREREG_PROBES_2026-09-08.md`), **0.818 GPU-h** — P3 `BOOT_SANITY`(job 905701, `BOOT_OK`)·P1 `SPLIT_FIRES`(job 905707, d44/d92 f_time 0.9921/0.9982 ⇒ 처치 발화 확인)·P3-C `DIVISION_COST`(job 905710, Δ`max_total_num_tokens` 0.051% ⇒ `HOMOG_FREE`)·P2 `FLOOR`(job 905712, `floor_ref`·`itl_solo` 계측, 게이트 4/4 통과)·P4 `SKEW`(job 905713, 예측1 지지/예측2 **VOID**[등록 문턱 3×가 산술적으로 도달 불가, `probes/ADJUDICATION_P4_2026-09-09.md`]/예측3 지지). ★**설계를 무너뜨린 실측**: `itl_solo` 13.0ms(가정 40ms의 1/3) ⇒ `RATIO` `C` 상한 0.703→0.229(死因 R3 3배 악화). ★★**정본 정정**: `CLAIM_EVIDENCE_MATRIX.md:264`·`CONSENSUS.md` M4R 항목의 "G16 28파일 39,849구간 전수 반례 0" biconditional은 **한 방향만 참**(⟹ 방향 1,643건 반례, 35.8% — 코드층 술어가 아니라 스냅샷 기록 시점 record-skew 아티팩트, M4R F1/F1′ 판정은 불변). **다음 방향(합의, 미등록·미감사)**: `R≈0.3–0.45`(prefill·decode 둘 다 바쁜 영역)에서 knee 이상 부하로 1요인(ctx·out 고정) static split 스윕, `n≥6`(`RESULT_RULEPOWER_2026-09-08.md` 부록 A: n=6 검정력 0.868, n=4는 0.124). ★새 성능 판정 0건·정책 순위 변경 0건·HE0 불변. 정본 반영: `CONSENSUS.md` rev56→**rev57**, "방법론 게이트" #101–103(신설). 상세 `workspace/engine-port/results/longctx_conflict/{PREREG_L1_2026-09-08.md, PREREG_RATIO_2026-09-08.md, audit_l1_rules_2026-09-08/VERDICT.md, audit_ratio_rules_2026-09-08/VERDICT.md, audit_dutycycle_2026-09-08/VERDICT.md, RESULT_DUTYCYCLE_G16_2026-09-08.md, FINDING_DUTYCYCLE_2026-09-08.md, QUESTION_COMPARISON_2026-09-08.md, PREREG_PROBES_2026-09-08.md, probes/RESULT_PROBES_2026-09-08.md, probes/ADJUDICATION_P4_2026-09-09.md, RESULT_RULEPOWER_2026-09-08.md}`, `handoff-report/session_handoff_2026-09-09.md`. ★★**追記(2026-09-09, 2차 세션, doc-steward)**: 위 "다음 방향" 스윕을 `PREREG_SWEEP_2026-09-09.md`(rev1)로 밀기 전에 선측정 프로브 C `CAPACITY`(job 905835, 1.11 GPU-h, 12/12셀)를 먼저 돌렸다 — R2 `P_C1_REFUTED`(out=96 max `L_decode`=**3.974**, 문턱 3.0 초과)로 `RESULT_CAPACITY_PRECOMP §3·§4.1·§4.2`("`R`이 decode 인구 천장을 정한다")를 자기 철회시켰다. `PREREG_SWEEP` rev1은 `NO-GO`(`audit_sweep_rules_2026-09-09/VERDICT.md`, 死因5[W1–W5], 트랙 계열 누적 **15연속**) — `ARGMAX_MOVES`가 이미 측정된 ITL95 표의 함수(게이트 #9 15번째 재발). 단일 문턱을 격자 지도로 대체한 rev2 + `PREREG_P5_ITL_LOCATE`도 `NO-GO`(`audit_sweep_rev2_2026-09-09/VERDICT.md`, 死因6[X1–X6], 누적 **16연속**) — `T1 REDUCIBLE`이 §0 두 단조성 전제의 연역(공통 `W` 반사실 `A=0.966–0.996`). 예측 없는 순수 계측으로 좁힌 `PREREG_P6_ITL_ORDER`도 `NO-GO`(`audit_p6_rules_2026-09-09/VERDICT.md`, 死因7[Y1–Y7], 누적 **17연속**) — E1·E2가 rev2 §0의 전제 그 자체였음을 메인 세션 독립 재현이 확인(E1 16/16 만장일치, E2 `𝒯=∅` 6/9판본). ★★**stake #1 구조 판정(등재, 스코프 선언)**: *"최적 static split 위치가 워크로드 모양에 따라 움직이는가"*는 **이 기판의 공통-`W` 설계로는 답할 수 없다**(`prefill SM+decode SM=108`이 엔진 강제라 `μ_p(D)`와 `itl(·,D)`를 분리 불가, 두 단조성이 보강하므로 순서 역전이 구조적으로 도달 불가 — 교훈 항목88). ★★**선행연구 사실**: 저장소 내 `muxwise/{sharegpt.yml,loogle.yml}`가 이미 워크로드별 SM 분할표를 출하(ShareGPT decode 20 SM vs LooGLE decode 52 SM, `decode_bs_threshold` 필드). ★★**정정 追記(2026-09-10, doc-steward)**: 위 "같은 하드웨어"는 틀렸다 — 두 yml 전 행이 `prefill_sm+decode_sm=132`(H100/H200급 다이, 우리 A100 108-SM 아님), SM 수치(20 vs 52)는 108-SM 격자와 비교·이식 금지, 승계되는 것은 워크로드별 분할표 출하라는 구조적 사실뿐. 또한 이 표는 2026-08-28 `impl_vs_external_pdmux_2026-08-28.md` §2.5–2.6에 이미 문서화돼 있었다(정본이 12일 늦게 흡수 — 게이트 #14/#18 사례, `CONSENSUS.md` §3 항목27 追記). 계측 자산 2건 승계(`L_decode_exact`·`span_hat`). ★새 성능 판정 0건·arm 순위 0건·정책 순위 변경 0건·HE0 불변. GPU 지출 이번 세션 **1.11 GPU-h**(트랙 누적 ≈**1.93 GPU-h**). 정본 반영: `CONSENSUS.md` rev57→**rev58**, "방법론 게이트" #104–114(신설, 전부) + #115(신설). 상세 `workspace/engine-port/results/longctx_conflict/{PREREG_CAPACITY_2026-09-09.md, RESULT_CAPACITY_PRECOMP_2026-09-09.md, probes/c_905835/REPORT.md, PREREG_SWEEP_2026-09-09.md, audit_sweep_rules_2026-09-09/VERDICT.md, PREREG_SWEEP_REV2_2026-09-09.md, PREREG_P5_ITL_LOCATE_2026-09-09.md, audit_sweep_rev2_2026-09-09/VERDICT.md, PREREG_P6_ITL_ORDER_2026-09-09.md, audit_p6_rules_2026-09-09/VERDICT.md}`. ★★**追記(2026-09-09, 2차 세션 속행, doc-steward 자기 정정)**: 최초 이 追記는 게이트 **#104–106·#111–114 7건만** 등재했다 — `audit_sweep_rev2_2026-09-09/VERDICT.md`의 "신규 방법론 게이트 후보 4건(#107–110)" 절이 메인 세션 전사 과정에서 누락돼 있었기 때문(원인 확인·판정서 원문에서 복원 완료). **#107–110도 등재**했고, 이 전사 누락 자체를 **게이트 #115**로 신설했다("에이전트 반환문을 파일로 옮길 때 절 단위 누락이 발생할 수 있고, 등재 절차의 상호참조 대조가 이를 잡아낼 수 있다" — 게이트 #14[★오인용 정정 2026-09-10: 원문은 "게이트 #18"이라 썼으나 오인용 — 정본=게이트#14/`CONSENSUS.md` §3 항목27, 상세 §3 항목27 追記]와 구분: #14는 진단의 전파 실패, #115는 최초 기록 단계의 데이터 손실). 정본 반영: `CONSENSUS.md` §3 항목131–135(신설, #107–110·#115 대응). ★★追記(2026-09-09~10, doc-steward, Step 1[job 905958, 0.244 GPU-h]+P7[job 905994, 2.606 GPU-h] 계측 반영 — 전부 계측, 새 성능 판정 0건·arm 순위 0건·정책 순위 변경 0건·HE0 불변, stake #1 미답 불변): Step 1이 `pdmux_homog6.yml`(D=54 추가) `BOOT_OK` 4/4·division 비용 0.0273%로 死因 Y4(공통 `W` 실현 불가)를 닫음 — `μ_p` 0.9143/0.6666/0.5772(d54)/0.1885, 등록 규칙이 `W=0.491`·`ρ=0.537/0.737/0.851`·도착 창 183.3s 전 arm 동일을 산출. 라벨러 정정: `S1_LABEL.json`의 `mu_p_spread=4.849×`는 부팅한 4 arm 전부에서 계산된 값이라 목표가 가리키는 양이 아님(게이트#21 역방향) — 목표 ≤2.4×는 P7 arm 집합 `{16,44,54}`에 적용되며 1.584× PASS. P7이 이 트랙 최초 자기 워크로드 부팅간 분산 계측을 완주(12/12부팅·main 36/36·sat 12/12, 교차검증 최대 2.66e-06, 실패 0건) — **`σ_boot`는 `n_boot=4`(df=3)에서 분해되지 않음**(3arm×2지표 전부 분산성분 원값 음수→0클립=**미검출**), 산출물은 F-검정 95% 상한: `itl95` 2.6–3.2%(부하보정 2.7–3.0%)·`μ_p` 2.4–6.1%(Step1 `n_boot=1` 값을 0.5–2.0%로 재현). ★★후속 설계 함의: **`σ_seed`는 잡음이 아니라 부하다** — 시드별 실현 도착률 폭 1.50×가 `L_decode` 산포를 R²=0.95–0.99로 설명(잔차 1.6–4.2%로 붕괴) ⇒ 검정력 계산엔 `span_hat`(실현 도착률)을 bench별 공변량으로 넣어야 하며, 이는 `RESULT_RULEPOWER_2026-09-08.md §7`·부록A의 `SIG_BOOT` 기반 검정력 표(수치 수입은 게이트#32로 금지, 방법론만 적용)에 직접 걸린다. `span_hat` 복원식이 독립 표본 36건에서 평균 |오차| 0.207%(`audit_sweep_rev2 §R5`의 12셀 0.077%와 합쳐 두 캠페인 48표본 검증). 신규 인용 금지 3건: TTFT 확률지배 위반 0(전 `T`, 409점 격자)은 관측 사실일 뿐 arm 순위·정책 해석 금지 · `d44`/`d54`의 `L_decode` σ_boot 상한 "0.0%"는 상한이 아니라 퇴화(`F<F₀.₀₅`) · "부팅 효과가 없다" 주장 금지(미검출일 뿐, V-probe `d44` 6.13%·AF-1 4.35×를 반증하지 않음, 게이트#32 양방향). stake #1은 여전히 구매 불가로 종결 상태 유지 — 후계 질문 Q-A(고정 split regret 프론티어)·Q-B(SM-split 액추에이터 자기상쇄 루프 이득)가 P7의 σ 95% 상한 + `span_hat` 공변량으로 이제 검정력 계산 가능(실행 계획 없음, 다음 세션 몫). `PREREG_P7 §3.1`: 등록 동작점이 전 arm 비포화라 `μ_p` 측정 불가능함을 실행 전 자기 발견 ⇒ 포화 bench 추가(주 bench 설계 불변, 판정 영향 0, 신규 게이트 번호는 매기지 않음). 신규 방법론 게이트 2건 등재(#116 분산성분 음수 클립=미검출·F-상한 퇴화, #117 표본별 산포가 처치-상관 공변량일 수 있다). GPU 이번 2.850 GPU-h(트랙 누적 ≈4.78 GPU-h). 정본 반영: `CONSENSUS.md` rev58→**rev59**, §3 항목136–137(신설), "방법론 게이트" #116–117(신설). 상세 `workspace/engine-port/results/longctx_conflict/{RESULT_S1_ARMFIX_2026-09-09.md, RESULT_P7_ITL_SIGMA_2026-09-10.md, PREREG_P7_ITL_SIGMA_2026-09-09.md, probes/p7_905994/P7_RESULT.json, probes/s1_905958/S1_LABEL.json}`. ★★追記(2026-09-10, 4차 세션, doc-steward, GPU 0[문서 등재분]) — 후계 질문 Q-A(고정 split의 regret 프론티어)가 규칙층 감사 계보 rev2→rev6을 거쳐 **이 트랙 최초의 `GO-with-caveats`**를 받고 rev7(제출 판본)로 캠페인 제출됐다. 계보: rev2(`audit_qa_rules_2026-09-10/VERDICT.md`) `NO-GO`(N2, 반전5건 F1–F5 — 실행상한이 셀 실행 자격을 난수의 함수로 만듦, 트랙 계열 누적 18연속) → rev3(`audit_qa_rev3/VERDICT.md`) `NO-GO`(반전3건 V1–V3 — 포락선 정의역이 측정량, 반전 1.52→25.52ms[16.8×]) → rev4(`audit_qa_rev4/VERDICT.md`) `NO-GO`(반전3건 — `min` 기준 arm이 추정량의 함수[argmin `pooled_p95`=d54 vs `median`=d16], 시드 동결이 rung-앨리어스 편향으로 전환[σ_seed/σ_boot 3.30×], 모양 B `𝒜`가 측정량의 함수) → rev5(`audit_qa_rev5/VERDICT.md`) `NO-GO`(반전1건 W1세다리 — `c_valley` 유도규칙 ↔ `{14,16,18}` 격자가 반대 라벨) → **rev6(`audit_qa_rev6/VERDICT.md`) `GO-with-caveats`(死因0·반전0/6표면)** — W1이 실제로 죽음(`gap<4ms` 게이트는 §1(a) 정의상 항등식이라 발화 불가하나 d92는 미등록 제3경로로 여전히 배제돼 결과 불변) → rev7(T1–T6 문면 수리 이행, 측정 계획 변경 0). ★★"N연속 NO-GO" 장부는 rev6에서 끊기나 **결정 규칙 6개를 유지한 채 반전 시험을 통과한 첫 설계**이지 P7식 결정규칙 0개 우회가 아니다(승계 장부 카운트 rev2 18→rev3 19→rev4 20→rev5 21, 각 판정서가 "독립 검증값 아님"[게이트#110]을 자기 등재). **캠페인 제출·결과 없음**: `probes/qa_regret.sbatch`, jobs **906504–906507**(라운드당 1개), 승인 예산 **≈10.1 GPU-h**(Stage A 폐지로 10.4→10.1), main 192+포화 20 bench, **PENDING·완료 0/4·결과 0건**(⛔결과·기대 기록 금지, 트랙 누적 4.78에 미가산). 계측 자산 3건 승계: `span_factor(seed,N)`(21표본 평균|오차| 0.21%·0.084%, `--random-range-ratio 1.0` 전용) · `X` 예보식(48표본 평균|오차| 0.21–0.24%) · `RESULT_QA_LADDER_LOO_2026-09-10.md`(형식상 PASS·판정 `VOID`, "사다리 밀도 충분" 인용 금지). ★★★정본 정정 3건: rev4 §3-C "d92서 전 arm 미처치 모드 수렴"은 미측정("d92 regret 원래 식별불가" 인용 금지) · rev6 §2-(1) "d92 좁은 이봉[1.9ms]"은 거짓(d92는 단봉, 1.96ms=지지집합 전폭) · 게이트#117은 페어드 `Δ` estimand에 비전이(R² 0.116/0.130, 원자료 0.95–0.99와 대조). 필수 병기 3종(인용금지 55건 소재는 각 VERDICT): ① 미처치 바닥 arm-무관(median-ITL 최소 13.016–13.079ms·최소TTFT 0.883–0.903s, 기전 `CONSENSUS §3 항목26`) ② `|𝒜(λ)|` 처치강도 순감소함수 ③ 저부하 외삽노출 18/19셀(94.7%). stake #1 구조 판정 불변(`Ê`·argmin·순위·포락선·"최적 arm" 산출 금지). 신규 방법론 게이트 30건(#119–#148, 원후보 35건을 중복 5쌍 병합) — 최고 전이가치 3건: #119(문턱 제거 시 자유표면이 표본자격층으로 하강) · #130(동결≠제거, 분산이 반복축 앨리어스 편향으로 전환) · #143(감사 처방 자기검사도 과잉일 수 있음, 값 고정 시 봉인 불필요). 새 성능 판정 0건·arm 순위 0건·정책 순위 변경 0건·HE0 불변. 정본 반영: `CONSENSUS.md` rev61→**rev62**(§5-6 追記·§3 항목139–168 신설), `reports/paper/CLAIM_EVIDENCE_MATRIX.md`·`EXPERIMENT_ROADMAP.md` Q-A 행 갱신(등급 변경 0건). 상세 `workspace/engine-port/results/longctx_conflict/{PREREG_QA_REGRET_2026-09-10.md, PREREG_QA_REGRET_REV{3,4,5,6,7}_2026-09-10.md, audit_qa_rules_2026-09-10/VERDICT.md, audit_qa_rev{3,4,5,6}_2026-09-10/VERDICT.md, RESULT_QA_LADDER_LOO_2026-09-10.md, probes/{span_factor.py,qa_regret.sbatch}}`. ★★追記(2026-09-10, 5차 세션, doc-steward, GPU 0) — 캠페인(jobs 906504–906507)이 PENDING/RUNNING인 동안 result-analyst가 분석기를 작성하며 rev7까지 4추정량의 **축(어느 배열의 범함수인가)**이 미등록이었음을 발견(축 R[요청별 `itl95`]에서만 `sign_agree=4` 전면 발화, 등록 축 P[풀링 토큰]에서는 0회) — rev8이 축 P를 등록해 봉인(`GO-with-caveats`, 死因0·반전0/10표면, `audit_qa_rev8_2026-09-10/VERDICT.md`), rev9가 판정서 처방(코드-only 상수 4건 등록·1시드 rung 대칭 진단 병기·산출7 구현·erratum 3건 정정)을 이행. 승계 인용 금지 총 68건(rev1–7 55+rev8 13), 신규 게이트 8건(#149–156, `CONSENSUS §3` 항목169–177) + 기존#130 追記(ε 중복) + 도구-버전관리 게이트 #157. 결과·기대 여전히 0건, 트랙 누적 4.78 GPU-h 불변. 새 성능 판정 0건·arm 순위 0건·정책 순위 변경 0건·HE0 불변. 상세 `workspace/engine-port/results/longctx_conflict/{PREREG_QA_REGRET_REV8_2026-09-10.md, PREREG_QA_REGRET_REV9_2026-09-10.md, audit_qa_rev8_2026-09-10/VERDICT.md}`. ★★★追記(2026-09-11, doc-steward, GPU 0[문서 등재분, 캠페인 자체는 완주분 10.641 GPU-h 기지출]) — 캠페인(jobs 906504–906507)이 **4/4 `COMPLETED 0:0`**로 완주했다(`RESULT_QA_REGRET_2026-09-11.md`): main 192/192·포화 20/20·부팅 16/16·`SELFCHECK_PASSED` 4/4·manifest 4라운드 동일·등록 정합 192=192·교차검증 최대 5.847e-06·벽시계 38,308s=**10.641 GPU-h**(예산 대비 +2.3%), 운영점 cudagraph-ON(192/192). claims-auditor가 이 결과 문서를 read-only 적대 감사(`audit_qa_result_2026-09-11/VERDICT.md`) — `qa_analyze.py` 미사용, raw `bench_*.jsonl` 192+`sat_*.jsonl` 20에서 등록 정의를 독립 재구현해 Δ행렬·CI·`sign_agree`·분산성분·모드·포화·makespan·span 전 항목 4자리 일치(불일치 0건), 해석 규율 준수 확인. 결함 9건(D1–D9) 중 **D1**(§15 #1 8칸 수열 오기 — §6.3과 자기모순)·**D2**(§13.2 `1.958×`·"변동≤1.6%" 어느 독법에서도 재현 안 됨)·**D3**(§15 #5가 `gap≥4ms` 항등식을 "작동하는 참인 주장"으로 서술, 게이트#9 18번째 재발)은 등재 전 정정 필수, **D9**(산출 8 라벨 과잉)는 강등 필수였고 **전부 원자료 문서에 이미 반영·커밋됨**(`bc0033c`). **정본 등재 = 조건부 가(可, "계측 기록"으로만)** — 조건 전부 충족. 등재 내용은 판정서 §9-1(정본 가능 10건: ①완주·10.641GPU-h·cudagraph-ON ②`sign_agree=4` 0칸 발화 ③"효과없음" 전사 금지 ④지배원인=네 추정량 부호 불일치[무순서36칸 중17, σ_req빼면22] ⑤`σ_boot` n_boot=4 미검출[비퇴화상한 21건 0.03–1.40%] ⑥계측건전[예보오차평균0.2236%] ⑦d92 `μ_p` 0.1885→0.1834±0.0002[n=4, 격자·ρ는0.1885불변] ⑧저부하외삽144/152[94.74%] ⑨미처치바닥arm-무관13.0064–13.0158ms·TTFT0.8768–0.8785s ⑩축P↔R부호반전34.0%[갈릴수있는108중45.4%])·§9-2(인용금지N-1…N-12 문자승계, 총80건[rev1-8 68+12] — 특히N-4[σ_req-free 결과로 rev8을 "틀렸다" 판정 금지, 게이트#32양방향]·N-5[같은λ의arm은처치강도미매칭, ρ(d92)/ρ(D)=4.754×항등식, Δ행렬 어떤칸도 "split만다른비교" 아님])·§9-3(기존기재변경3건: μ_p_ref(d92) 0.1834±0.0002[n=4] 병기·rev6 6-8 TTFT범위에 "이캠페인실측이더낮음" 병기·"새성능판정0건·arm순위0건·정책순위변경0건·HE0불변·stake#1불변"이감사전수확인으로참임 등재)를 문자 그대로 따른다(새 해석·새 판정 추가 0건). ★★GPU장부: 트랙누적 **4.78→15.42 GPU-h**(이 트랙 최초의 완주 캠페인). ★★신규 게이트 4건(#158–161, 판정서§11 G-a/G-b/G-c+메인세션발의1건, 기존#119–157과중복없음확인) — #158(G-a, 절폐기시그절을입력으로쓰는산출목록항목이고아로남을수있다, 교훈89의산출-목록판본)·#159(G-b, 결과문서요약표가본문절과다른수를실을수있다)·#160(G-c, 인용금지문자승계+같은문서다른절재긍정, 게이트#9 18번째재발)·#161(요약통계출처표본이전체가아니라부분[1/4라운드]일수있다, G-b와인접하나판정대상달라별개). 기존 불변 배너 전부 승계(HE0·정책순위·gate#13/#16"닫았다"금지·switch-cost"닫았다"금지·C2인용정지(a)(b)·`CONSENSUS §1-24`·게이트#14"닫았다"금지·stake#1구조판정 — 감사가전수확인). 새 성능 판정 0건·arm 순위 0건·정책 순위 변경 0건·HE0 불변. 정본 반영: `CONSENSUS.md` rev63→**rev64**(§5-6追記·§3항목178–181신설), 이 표 행(이번 追記)·"방법론게이트" #158–161신설, `reports/paper/{CLAIM_EVIDENCE_MATRIX,EXPERIMENT_ROADMAP,DOCUMENT_STATUS}.md` Q-A행갱신(등급변경0건), `MEMORY.md`·`memory/slo-aware-scheduling-track.md`[2026-09-11]·`memory/deconfound-measurement-lessons.md` 항목157–160신설. 상세 `workspace/engine-port/results/longctx_conflict/{RESULT_QA_REGRET_2026-09-11.md, audit_qa_result_2026-09-11/VERDICT.md}`. ★★追記(2026-09-11(2), doc-steward, GPU 0) — 후계 질문 Q-B의 사전등록(`PREREG_QB_LOOPGAIN_2026-09-11.md`)이 claims-auditor 규칙층+결과 감사(`audit_qb_rules_2026-09-11/VERDICT.md`) `GO-with-caveats`(死因 0·반전 0/23표면)를 받았다. 명칭 변경 Q-B→Q-B′(공통 λ에서 decode 평균 ITL 차의 노출·인구 합성 표준화 분해, 경로 cf) — 원 Q-B의 프로브 C 교락 수치(14.953×/23.713×)는 Little 항등식(도착 비×체류 비)으로 소진됐다. Stage 0(GPU 0, 기존 Q-A+P7 원자료, 신규 측정량은 토큰 노출 계측기 `E` 하나)은 "탐색적 계측 기록(사후 정의)"으로만 등재 가능, Stage 1(≈2.9 GPU-h)은 실행 허가되나 저자·감사 모두 비권고로 미실행. 인용 금지 QB-1…QB-20(소재는 사전등록 §12·판정서 §5.2) + 필수 병기 ⓐ–ⓕ, 인용 금지 총계 100건(Q-A계보80+QB20). 신규 게이트 4건(#163–166) + 追記 3건(#159·#160·게이트#83/교훈88) — 전문은 위 "방법론 게이트" 절·`CONSENSUS.md` §5-6·§3 항목182–186 참조. stake #1 구조 판정·HE0·정책 순위 전부 불변. 새 성능 판정 0건·arm 순위 0건·정책 순위 변경 0건. GPU 지출 0(트랙 누적 15.42 GPU-h 불변). 상세 `workspace/engine-port/results/longctx_conflict/{PREREG_QB_LOOPGAIN_2026-09-11.md, audit_qb_rules_2026-09-11/VERDICT.md}`. |

    감사 보고서(있는 것만): E1-b/c ↔
    `workspace/engine-port/results/p1_gates/gate2/AUDIT_E1B_E1C_2026-08-14.md`,
    C2-R rev1→rev2 ↔ `PREREG_C2R_RULES_REV2_2026-08-15.md` 본문(rev1
    死因 5건을 rev2가 자체 기록, 별도 감사 보고서 파일 없음). LTSM
    P1·E-1은 별도 감사 보고서 파일 없음(판정은 각 PREREG 문서 자체
    및 2026-08-14 세션 기록). ★`%smid` R0의 **2026-08-14 판정(5조건)은
    여전히 별도 감사 보고서 파일 없음**(그 원문이 7일 만에 소실된
    이유이기도 하다) — 단 **2026-08-21 2층 감사부터는 파일로 남는다**
    (`workspace/engine-port/results/smid_census/audit_smid_r0_2026-08-21/
    VERDICT.md`, ★신규 규율 — 아래 "방법론 게이트" #55). G13·kernel_mech
    rev2는 별도 감사 보고서 파일 없음(판정은 메인 세션이 handoff §4-4에
    기록, claims-auditor 산출물 자체는 아직 저장소 미편입 — G13 감사
    독립 재구현만 `audit_g13_independent_2026-08-16/`로 보존됨).
    ★kernel_mech rev3 재감사·rev5는 파일로 남는다(`workspace/
    engine-port/results/kernel_mech/DESIGN_KERNEL_MECH_REV3_
    2026-08-20.md` §10-F 재감사 본문, `audit_kernel_mech_rev5_
    2026-08-21/VERDICT.md`) — rev4-lite 판정은 rev5 문서 서두에
    요약만 있고 별도 파일 없음. G17·E-B1은
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
    재발 + 게이트 #18 사례[★오인용 정정 2026-09-10: 정본=게이트#14/
    §3 항목27 — 상세 §3 항목27 追記], `reports/CONSENSUS.md` §3 항목59 신설).
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
    (f) ★**2026-08-23 해소 — 분석기층 재감사 완료**(`results/slo_sched/audit_g16_analyzer_2026-08-23/VERDICT.md`, `CONFIRMED with conditions` C1–C4). **7개 분석기 빌드**(사전등록 원본 → 현재 디스크본)로 같은 원자료를 재채점해 **전 결정량 불변**을 실증했고, ★**`delta_slo` 부트스트랩이 `g16_analyze.py`에 존재하지 않으므로** 정본 §1-33의 가장 하중 큰 두 문장(*"처리량 최적 REFUTED"*·*"payoff 무판정"*)은 **애초에 분석기 산출물이 아니라 독립 재도출물**이며 사후 패치 4건과 **구조적으로 무관**하다. ★**조건 4건**: **C1** 폐기 sha 인용 금지(기계 차단 등재) · ★**C2** *"PC-A..PC-E 전후 동일"*을 K11·동률가드·tie-break·`S_itl` 경계 무해성의 근거로 **인용 금지** — 변이 테스트로 **대조가 그 4건 전부에 눈멀었음이 실증**됐다(K11 44→34는 **LO 판정을 뒤집는데** 전 대조 통과), 근거는 **패치별 직접 재채점**을 인용하라 · ★**C3 K11 스코프**: LO의 `ITL_SATURATED`·K9 `NO_MORE_BLOCKS`는 **U 정의가 min SM ≥ 44일 때만** 성립(≥34이면 `UNIDENTIFIED`/`ADD_2_BLOCKS`), HI는 ≥34까지 · **C4** `PC-B`를 *"대조가 캠페인 코드 경로를 탔다"*의 근거로 인용 금지(`run_controls`가 `run_campaign`보다 먼저 돈다, §3 항목64와 **별건·미해결**). 종전 문구: **분석기 누적 미감사**: `g16_analyze.py`의 K11 교체·동률
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

    ★★★★★**2026-08-21 갱신(doc-steward — gate #13 job-축 캠페인
    완주 + 양 arm `PASS`, S0(a) 초판 결론 철회, G18 rate-축 트랙
    보류, kernel_mech rev3 차단 2건 수리 반영. GPU 지출 11.72
    GPU-hr[캠페인 11.66 + 취소 프로브 0.06], 새 성능 판정 0건 —
    이 캠페인은 분산 측정이지 성능 비교가 아니다)** ★**후보 (i)
    gate #13 rev3가 실행·완주됐다** — 사전등록(§10 감사 부록 포함)·
    블라인드·감사 해제를 전부 거쳐 M8·Ha8 **양 arm `PASS`**
    (`gap/(3·UB95)` 18.37·4.95, 감사 최악 조건에서도 11.62·3.13
    유지). 상세는 위 "다음 실험 gate" #11 레지스트리 "G13 job-축
    캠페인 결과" 행, 원자료 `G13_ANALYSIS_2026-08-21.json`. ★★쓸 수
    없는 문장 불변: **"gate #13을 닫았다"**(865533 batch⊗regime
    앨리어스 불변, 이 캠페인은 job 축만 쟀다) · **`Δ_batch`를
    측정했다**(배치 축 없음). 노드 성분은 2노드·계수 0.400·유효
    df≈1로만 표집(정본 #13(1) ≥3노드 부분 미충족 확정). 부수:
    ★비용 상수 144s/부팅이 실측 145.8s로 검증돼 rev3 §6-8 미해결
    항목 해소. ⇒ **후보 목록은 이제 3개**: (ii) **kernel_mech rev3**
    (차단 2건 설계 본문 수리 완료 — §3.1 셀 realized 판정·§3.2
    `gap_frac` 항등식 가드, ★이 수리 자체는 **미감사**, 다음
    세션 재감사 후 사전등록으로 넘어간다) · (iii) **gate #16 rate 축**
    (규칙층 초안·재감사 전, 불변) · (iv) **S-6 telemetry 대조**
    (`n` 확정 선행, 불변).

    ★**S0(a) 초판 결론 철회**(`workspace/engine-port/results/
    s8_scaleup/S0A_VERDICT_2026-08-20.md` rev2, claims-auditor 감사
    REFUTED) — *"C-g(부팅 내부 `--conc` 교대)의 전제는 측정 가능한
    모든 lag에서 성립하지 않는다"*를 **철회**한다. 초판 판정선이
    비용 미정합 비교였다(1 부팅쌍 대조를 2 부팅쌍짜리 SD와 견줌).
    비용 정합(`s_alt` 대 `2σ_boot`)으로 재판정한 결과: **C-g 이득은
    교대 속도의 함수**다 — 부팅당 8구간 교대는 이기고(Ha8 5.46×·
    M8 1.31×) 4·2구간은 진다(0.19–0.88×). 신규 등재(성능 판정
    아님): 정보 없는 기준선은 √NBIN=2.83(1 아님), ★**`r`은 창
    의존량**(Ha8 30s 3.025 vs 60s 3.116 — 정본 채택구간 밖). 답
    못 하는 것 = `--conc` 전환 과도응답(스모크 1 부팅으로 잴 수
    있음, §4). ⇒ **gate #13 불변**(S0(a)는 애초에 임계경로 밖이라
    이 철회가 gate #13 판정을 바꾸지 않는다).

    ★**G18 rate 축 — 트랙 보류**(`workspace/engine-port/results/
    slo_sched/PREREG_G18_PROBE_2026-08-20.md` HOLD 배너) — rev1
    규칙층 REFUTED(격자 축소 등 6死因) → rev2 **NO-GO**(2건만 제거,
    새 死因 5건) → 사용자 결정으로 **보류**(gate #13 완주 후 재개
    판단). ★**정본 오염 경로 발견·수리**: `g16_analyze.py:953`의
    글롭+regex가 프로브 run id를 정상 매치해 완주 프로브가 G16
    블록으로 조용히 편입될 뻔했다(태그가 막아준다는 주석은 거짓이었음)
    — run id 접두사를 `g16_`→`g18probe_`로 바꿔 구조적으로 차단(실제
    오염 0건). ★근거 수치 자체도 갱신: 인용해온 LO 산포 1.26%가
    귀무 기대치(n=4, E[range]=1.207%)와 구분 불가(값 자체도 blk1
    단일 부팅 산물, blk2–4만 쓰면 0.493%) — 결론 방향은 강화되나
    절대 문턱으로 못 씀. 검정력도 정정(비페어드 7.9→페어드 n≈15).
    취소 6 job 중 1건만 3분36초=0.06 GPU-hr. ★**gate #16은 불변**
    — 두 판본 어느 쪽도 닫히지 않았다.

    ★★**정본 자신의 결함 정정(방법론)**: 위 2026-08-20 갱신 문단과
    "다음 실험 gate" #11 레지스트리 kernel_mech P1 프로브 행이 *"변인은
    '커널이 green-context 스트림 위인가' 하나뿐"*이라 적었던 것은
    원본 `P1_VERDICT_2026-08-20.md` §2보다 **과잉**이었다 — 원문은
    "두 겹의 대조가 같은 방향을 가리킨다"로 더 조심스럽다(다리 간
    대조는 `realized_sm` 16 vs 108도 함께 바뀌고, 다리 내 대조는
    커널 종류 자체가 다르다). 위 두 곳 모두 원문 수위로 정정했다.

    상세 `handoff-report/session_handoff_2026-08-21.md`,
    `workspace/engine-port/results/s8_scaleup/{G13_RESULTS_2026-08-21.md,
    G13_CAMPAIGN_LOG_2026-08-21.md, S0A_VERDICT_2026-08-20.md}`,
    `workspace/engine-port/results/slo_sched/{PREREG_G18_PROBE_2026-08-20.md,
    G18_RATE_VALUE_2026-08-20.md}`, `workspace/engine-port/results/
    kernel_mech/DESIGN_KERNEL_MECH_REV3_2026-08-20.md`.

    ★★★★★★**2026-08-21 갱신(2차, doc-steward — S-6 검정력 계산
    완료 + 규칙층 감사 `NO-GO` 반영, GPU 지출 0, 새 성능 판정 0건)
    후보 (iv) S-6 telemetry 대조의 상태가 바뀐다 — "n 확정 선행,
    불변" 문구는 더 이상 정확하지 않다.** 검정력 계산 자체(`S6_POWER_
    2026-08-21.md`)는 완료돼 규칙층 감사(claims-auditor, 게이트 #34
    1단)로 넘어갔으나 판정은 **`NO-GO`** 다. 死因 4건 — ① OFF 다리가
    하네스·분석기에서 구조적으로 탈락(`g16_grid.sbatch:801-804`
    `TELEM_RC!=0`→`artifact_invalid`, 로더가 `telem_rc==0` 요구,
    이대로 지으면 `D_telem`은 OFF n=0의 항등식) ② 비오염 논증이
    이름 약속뿐(`g16_*` 글롭 매치, 교훈 #57 재발) ③ 결정량은 절대 ms
    차이인데 판정선·계산은 상대 %(등가성 대 정밀도 혼동) ④ pooling
    판단 근거가 귀무분포 없는 범위통계량(귀무 재계산 시 구속 셀
    관측치가 95백분위와 사실상 동률). ★**전부 GPU 0의 문면 수리로
    해소 가능하며 하네스 착수 전에 잡혔다 — 이건 설계 반려이지
    계측 축 분리 자체의 반증이 아니다**(방법론 게이트 #21과 혼동
    금지: 측정 실패를 게이트 실패로 라벨링하는 것의 거울상 오류 —
    여기선 반대로 설계 반려를 실험 방향 반증으로 오독하지 말 것).
    ★**등록된 n은 아직 없다.** B-2가 적었던 "n≥2(≈0.3 GPU-hr)"는
    이 계산으로 **반증**됐다 — 필요한 n은 δ와 사전값의 함수이며,
    구속 셀(HI `M_ttft`)에서 δ=5.4%(참조값, basis 미검증 수입 —
    §6) 기준 **최소 5쌍**(점추정)~**18쌍**(보수 UB95 사전값)이
    필요하고, 부팅 단가는 실측 **0.1272 GPU-hr/부팅**(457.8s, B-2가
    적은 값의 2배). 감사가 지정한 등록 후보(§7 권고 C를 대체)는
    1단 6쌍→캡 12쌍(≈3.25 GPU-hr, job 2개 분할 필요), 판정 셀 = HI
    `M_ttft` 단독, δ = 격자-고유값(argmin 여유 9.070% 또는 arm-평균
    CI 반폭 4.077% — 후자는 δ·σ가 같은 σ̂에서 나와 **"검정력 계산이
    아니라 항등식"이라고 라벨해야 함**) — 아직 **후보일 뿐, 등록
    아니다.** ⇒ **payoff 축소**: S-6가 닫을 수 있는 것은 **1 arm
    (d44)의 공통 시프트 크기**뿐이다 — 설계 §1은 빌드 축만 조건을
    걸었으나 실제로는 모델·워크로드·regime 축이 더 크게 막아
    he2/sgptv 계열 절대값으로 이식 불가하고, `arm×telemetry`
    교호작용(오버헤드가 결정축과 상관하는지)은 1 arm으로는 원리상
    측정 불가라 스코프 밖으로 등록해야 한다. 死因별 수리 경로(전부
    GPU 0)는 문서 §10에 있다. ★**정본 전파**: B-2의 "n≥2(≈0.3
    GPU-hr)"를 그대로 인용하던 세 곳 — `reports/CONSENSUS.md` §3
    항목61·이 문서 "방법론 게이트" #41·`PREREG_G16_RULES_REV3_
    2026-08-16.md:587`(addendum B-2 원문, 정정 표시 병기, 원문 보존)
    — 을 이 갱신과 함께 정정했다. ★**gate #13·gate #16 개폐 서술
    불변**(이 갱신과 무관) · **HE0·정책 순위·C2 인용정지 (a)(b)
    불변** · **새 성능 판정 0건.** 상세 `workspace/engine-port/
    results/slo_sched/S6_POWER_2026-08-21.md` §2·§4·§5·§9·§10,
    원자료 `S6_POWER_2026-08-21.json`, 스크립트 `s6_power.py`, 커밋
    `ace05b8`.

    ★★★★★★★**2026-08-21 갱신(3차, doc-steward — 2차 세션, GPU
    지출 0[kernel_mech]·0[미해결, %smid R0는 job 889631 하나
    `PENDING`], 새 성능 판정 0건) 후보 (ii) kernel_mech rev3의
    상태가 다시 바뀐다 — "이 수리 자체는 미감사(다음 세션 재감사
    대상)"이던 것이 재감사를 받았고, 그 뒤로도 두 판본이 더
    나와 **세 번 연속 규칙층 `NO-GO`** 다.** 순서: **rev3 재감사**
    (rev3 §10-F가 스스로 지목한 "이 수리 자체는 미감사" 결손을
    실행 — `NO-GO`, 차단 **5건**: cell B 문턱 0.98이 상속 워크로드에서
    도달 불가[실측 동거율 1.3/4.9/25–40%] · cell A 문턱이 D=16에서
    비변별적 · realized 추정량이 편향[개수 서브샘플링] · cell D에
    residency 검증 부재 · `gap_frac` 분모·인덱스집합 미정의) →
    **rev4-lite**(`NO-GO`, 차단 **8건**: `T_step`을 NVTX[=CPU
    타임라인]로 정의하고 "device-side"라 서술 · `r_K`·`gap_frac`이
    `T=K+G`로 종속 · **모든 게이트를 통과하며 틀린 쪽을 지목하는
    반례** · 문턱 0.85가 정본 ε 구간을 정확히 가름 · 부팅 62.5%가
    판정 미사용) → **rev5**(`NO-GO`, 차단 **7건**: ★**구조적 편향**
    — `S_K≥0`이라 `s>0.5`의 필요조건이 `G92>G44`인데 남긴 간극
    후보가 SM에 비례해 커지지 않아 `GAP_DOMINATED`가 도달 불가[간극
    일정 케이스 `s=0.000` 메인 세션 재확인] · ★엔진에 NVTX 방출이
    **0건**[`grep -rn nvtx workspace/engine-port/src/`, 메인 세션
    확인]이라 이 설계는 하네스가 아니라 **엔진 핫패스 패치**를
    요구하는데 예산·상태에 미계상 · 검정력 분석 0건 · §3.4 여유
    근거 삼중 무효 · `CONC`·`L`이 값 없이 고정 · `U_infl` 설계와
    예산이 서로 실행 불가). **세 판본 전부 하네스 착수·GPU 지출
    전에** 규칙층에서 걸렸다(GPU 0). ★**가장 깊은 층**(감사):
    *"커널 안인가 밖인가"* 판정이 **반사실 선택에 의존**한다 —
    대칭 반사실(`G_ideal := G(44)·44/92`)과 비대칭 반사실이 **같은
    데이터에서 반대 판정**을 내고(대칭으로 바꾸면 rev5 자기검증
    fixture는 같은 답, CE-1 반례만 뒤집힘: 0.811 ⇒ `GAP`, "정답")
    둘 다 똑같이 "정의"다. ⇒ **감사 권고**: 두 반사실을 **나란히
    사전등록**하고 갈리면 `COUNTERFACTUAL_SENSITIVE`를 결과로 보고
    (GO 전환 조건 ①). ★**부수 확정 2건**(이번 세션에서 세 판본에
    걸쳐 반복 확인): (1) **`t_launch`(후보 vii)는 cudagraph 운영점에서
    항등식**이라 폐기됐다(rev1 死因이 이름만 바꿔 rev3에서도 생존한
    것을 재확인 후 최종 폐기). (2) **엔진에 NVTX 방출이 0건**이라
    이 track이 전제해 온 "하네스만 지으면 된다"가 성립하지 않는다
    — `pdmux.decode_step` NVTX를 실제로 방출하려면 엔진 핫패스
    패치 + manifest 갱신 + correctness gate가 선행돼야 한다(engine-porter
    이관 대상, 미이관). ★**kernel_mech rev4/rev5는 둘 다 `NO-GO`
    상태로 커밋돼 있다 — 설계 근거로 재사용 금지.** ⇒ **사용자
    판단 대기**: rev6(감사 권고 ① 반영) / 트랙 보류 / 트랙 종결
    — 다음 세션의 첫 결정 사항.

    ★**후보 (iv) S-6 telemetry 대조 — 추가 상세(상태 불변,
    `NO-GO`).** 검정력 재계산이 지정한 등록 후보의 **최종 구간
    형태**가 확정됐다: **Stein 2단**(`X̄_N ± t_{.975,n₁−1}·S₁/√N`,
    1단 6쌍→캡 12쌍). ★**payoff는 감사 후 이미 좁아져 있다**(위
    2차 갱신 참조, 재확인만) — 이 격자 Δ는 he2/sgptv 절대값에서
    뺄 수 없고(모델·워크로드·regime 축), *"오버헤드가 결정축과
    상관"* 은 **1 arm으로 원리상 측정 불가**라 남는 것은 "공통
    시프트의 크기" 하나뿐이다. 死因 4건 수리(전부 GPU 0)는 아직
    실행되지 않았다 — **아직 후보일 뿐, 등록 아니다.**

    ★**신규 항목 — `%smid` R0가 처음으로 제출됐다(job **889631**,
    `amd_a100nv_8`, `PENDING`). ★결과는 아직 없다 — 다음 세션이
    읽어야 할 것은 아래 "다음 실험 gate" #11 레지스트리 `%smid` R0
    행이다.** 요지만: 2026-08-14 감사의 `CONDITIONAL-GO(5조건)`
    원문이 저장소 어디에도 남지 않아 **7일 만에 소실**됐음을 이번
    세션이 확인했다 — 등재는 레지스트리 한 줄 + 핸드오프 두 줄뿐
    (2026-08-14 날짜 핸드오프 파일 자체가 없다). 2층 감사가 조건을
    **추측 재구성이 아니라 새로 도출**했고, ★이번엔 그 판정서를
    **파일로 남겼다**(`workspace/engine-port/results/smid_census/
    audit_smid_r0_2026-08-21/VERDICT.md` — 판정어
    `GO-with-conditions`, 차단 C1–C5). 지목된 결함(fail-open 2건
    [`smid_l0_census.py`+P0-A 복사본] · C1 빈 census 공허참 PASS ·
    C2 미포화 부재주장 · C3 라벨 병합) 전부 이번 세션에 수리(자기검사
    58/58·변이 28) → 제출. ⇒ **후보 목록은 이제 사실상 0개 즉시
    실행 가능**(kernel_mech는 사용자 판단 대기, gate #16 rate 축은
    G18로 이미 실행돼 HOLD, S-6는 死因 수리 미착수) — `%smid` R0는
    이 4후보 목록 밖의 별도 트랙이다. 상세 `handoff-report/
    session_handoff_2026-08-21b.md` §2.2·§2.3, `results/kernel_mech/
    {DESIGN_KERNEL_MECH_REV3_2026-08-20.md §10-F 재감사,
    DESIGN_KERNEL_MECH_REV4_2026-08-21.md,
    DESIGN_KERNEL_MECH_REV5_2026-08-21.md,
    audit_kernel_mech_rev5_2026-08-21/VERDICT.md}`, `results/
    slo_sched/S6_POWER_2026-08-21.md`, `results/smid_census/
    {PREREG_SMID_R0_2026-08-14.md, audit_smid_r0_2026-08-21/
    VERDICT.md}`. ★**gate #13·gate #16 개폐 서술 불변**(이 갱신과
    무관) · **HE0·정책 순위·인용정지 불변** · **새 성능 판정 0건.**

    ★★**2026-08-22 갱신(doc-steward — `%smid` R0 결과 도착 +
    claims-auditor 결과 감사 `CONFIRMED(scoped)` 반영, GPU 지출
    0.0275 GPU-hr[결과 도착분], 새 성능 판정 0건) `%smid` R0(위
    별도 트랙)가 종결됐다 — 4후보 목록(gate #13 rev3는 이미 완주,
    kernel_mech rev3/gate #16 rate 축/S-6는 상태 불변)에는 영향
    없음.** 판정: `GLOBALLY_CONSISTENT_LABEL`(job 889631, 상세는
    위 "확정된 결과" 1번 R0 addendum·"다음 실험 gate" #11
    레지스트리 `%smid` R0 행 2026-08-22 갱신). ★**"물리 SM
    인덱스"가 아니라 "전역 일관 라벨"** — S3·G1-d(위 참조)는
    **닫히지 않는다**(이건 엔진 없는 별도 프로세스의 eager census,
    cudagraph-ON 운영점 전달 문장 금지). 아티팩트 결함 N1·N2로
    `.json`만 인용(`.txt`/`.out` 인용 금지, 아래 "방법론 게이트"
    #56). 감사 caveat 3건 처리: (a) 사전등록 §0.2 항목번호
    오프바이원 — 이후 문구로만 인용 권고(반영 완료, 이 문단) ·
    (b) 게이트 후보 2건 → #56·#57 신설(아래) · (c)
    `smid_l0_run.sbatch:86` stage 4 rc 미검사 → engine-porter
    이관 항목 등재(미수정, 위 #11 레지스트리 행 각주). ★**불변**:
    `CONSENSUS.md` §1-1 인용 금지·Gate 2-S 크기 인용 셀 1개·HE0·
    정책 순위·gate #13/#16 "닫았다" 금지 전부 그대로. 상세
    `reports/CONSENSUS.md` §1-1(R0 addendum)·§3 항목76·77·§4
    living-doc(smid_l0_verdict_889631.json 행), `reports/paper/
    CLAIM_EVIDENCE_MATRIX.md:704`(S3 항목, L0 유사물만 측정
    명시).

    ★★★★★★★★★**2026-08-23 갱신(doc-steward — 직전 핸드오프 §3
    권고 순서(A-1→A-2→B-1) 집행 결과 반영, GPU 지출 0·job 제출
    0건, 새 성능 판정 0건) 후보 목록에서 (iv) S-6가 빠지고 새
    트랙 P0-A가 규칙층→하네스층으로 단계 이동했다 — "4후보"라는
    표현은 이제 부정확하다.**
    (iv) **S-6 telemetry 대조 — ★사용자 결정으로 트랙 보류(HOLD,
    G18 선례와 같은 형태)**. `S6_POWER`(1회차) 이후 rev1→rev4가
    규칙층 감사에서 **5회 연속 `NO-GO`**를 받았고, 그중 2회
    (rev3 E4·rev4 F2)는 **메인 세션의 수리가 캠페인 자체를
    파괴**하는 결함이었다(`set -u` 즉사 / OFF 다리 채택률 0으로
    3.146 GPU-hr 결정론적 전액 소실). 근본원인은 문장 품질이
    아니라 **아키텍처**다 — 7-arm G16 하드와이어 하네스를 1-arm
    대조로 포크하려 했고 매 회차 새 거부지점이 나왔다(총 7건).
    상세는 위 "다음 실험 gate" #11 레지스트리 S-6 행, 재개 조건은
    `PREREG_S6_2026-08-22.md` §10.4.
    ★**신규 트랙 P0-A(cudagraph replay가 green-context SM 한정을
    전달하는가) — 규칙층 5회 감사 끝에 rev6에서 수렴, 다음은 §10
    하네스층 감사(게이트 #34 2단계)로 단계 이동.** 이 트랙은
    직전 세션의 "4후보" 목록에는 없었다(신규 착수) — 규칙층
    감사(claims-auditor)가 매 회차 신규 차단을 냈고(rev1 하한
    안전망 제거 → rev2 상한도 제거돼 있었음 → rev3 1라벨 잡음
    최고가 판정 → rev4 이력표가 하지 않은 수리를 적음 → rev5
    결정량이 게이트 0·열거 1값으로 승격됨), 5회차 감사가 "규칙층의
    남은 표면적이 한 점으로 수렴했다"고 판정해 **규칙층 감사를
    더 사지 말라고 권고**했다. 등재 가능한 결과는 **0건**(사전등록은
    결과가 아니고 GPU도 미집행). 상세는 위 "다음 실험 gate" #11
    레지스트리 P0-A 행.
    ⇒ **후보 목록 재정리**: (i) **P0-A**(§10 하네스 재작성 →
    2단계 감사 → 0.17 GPU-hr 집행, 세 단계 남음) · (ii)
    **kernel_mech rev3**(B6만 닫힘, rev7 전체는 여전히 `NO-GO` —
    B1–B5·B7 불변, NVTX 엔진 패치 선행조건, 사용자 판단 대기
    rev6/보류/종결 불변) · (iii) **gate #16 rate 축**(규칙층
    초안·재감사 전, 불변). **S-6는 이 목록에서 빠지고 별도
    HOLD 상태**(재개는 사용자 판단, §10.4 조건 충족 후).
    ★**gate #13·gate #16 "닫았다" 금지·switch-cost "닫았다" 금지
    불변** · **HE0·정책 순위·인용정지 전부 불변** · **새 성능
    판정 0건.** 상세 `handoff-report/session_handoff_2026-08-23.md`
    §2·§4.

    ★★★★★★★★★★**2026-08-23 갱신(2차 세션, doc-steward — P0-A
    결과 도착 + kernel_mech rev7(전체) 규칙층 감사 + NSL-1 신규
    트랙 반영. GPU 지출 0(등재 세션 자체) — 결과 자체는 위 P0-A
    행에서 이미 0.054 GPU-hr 지출됨, 새 성능 판정 0건) 후보 (i)
    P0-A가 결과 단계까지 실행·완료됐고, (ii) kernel_mech rev3의
    상태가 "전체 규칙층 감사 완료"로 갱신되며, 신규 후보 (iv)
    NSL-1이 추가된다.**
    (i) **P0-A**(§10 하네스 재작성 → 제출[job 890893] →
    ★**결과 도착 — `CONFINEMENT_PRESERVED_THROUGH_GRAPH_REPLAY`,
    claims-auditor `CONFIRMED with conditions`**) — 이 하위 질문은
    **실행·판정 완료**다(상세는 위 "다음 실험 gate" #11 레지스트리
    P0-A 행). ★단 **S3·G1-d(Gate 2 본 질문)는 여전히 닫히지
    않는다**(P0-A도 L0 유사물, 엔진 층 L1 in-server boot census는
    미측정 — R0 §9의 R4/구멍 C 그대로 열림). 이 항목은 더 이상
    "실행 대기 후보"가 아니라 "완료된 도구 타당성 결과"로 재분류.
    (ii) **kernel_mech rev7**(B6는 이미 판정 완료 — 위 참조) —
    ★**이번 세션에 rev7 전체(C1–C10)도 규칙층 감사를 받아 `NO-GO`**
    (死因 없음, 국소·명세층 10건). 감사 권고 = **2단 분할 구매**
    (Stage 0′ ≈0.2–0.4 GPU-hr는 값한다·Stage A는 조건부). "문서
    8건 미수정" 단계는 지났고, 다음 액션은 사용자 판단
    (Stage 0′ 선행 구매 / rev8 규칙층 재감사 / 보류 / 종결) —
    ★**rev4/rev5와 달리 死因이 없다는 점은 rev3 재감사 이래 불변**.
    (iii) **gate #16 rate 축**(규칙층 초안·재감사 전, 불변).
    (iv) ★**신규 후보 — NSL-1**(non-SM-split admission lever,
    CONSENSUS §5-8(b) 겨냥, `E-1`과 무관) — **rev1 규칙층 `NO-GO`**
    (死因 F1–F3) → **rev2 작성**(F1–F15 반영, ★미감사) — 다음
    세션 후보는 **rev2 규칙층 재감사**. S-6는 계속 목록 밖 별도
    HOLD.
    ★**감사 부채 목록**(`reports/AUDIT_DEBT_2026-08-23.md`)이
    이 4-후보 목록과 별도로 층별 미감사 항목을 전수 추적한다
    (G13 하네스층·`g16_analyze.py` 재감사·P0-A H1–H7 수리 감사 등
    — 이들은 "새 실험 후보"가 아니라 "기존 결과 위에 남은 감사
    부채"이므로 이 목록에 넣지 않는다).
    ★**gate #13·gate #16 "닫았다" 금지·switch-cost "닫았다" 금지
    불변** · **HE0·정책 순위·인용정지 전부 불변** · **새 성능 판정
    0건.** 상세 `workspace/engine-port/results/bcg_probe/
    audit_p0a_result_890893_2026-08-23/VERDICT.md`, `workspace/
    engine-port/results/kernel_mech/audit_kernel_mech_rev7_2026-08-23/
    VERDICT.md`, `workspace/engine-port/results/nsl_lever/
    audit_nsl1_rules_2026-08-23/VERDICT.md`, `reports/AUDIT_DEBT_
    2026-08-23.md`.

    ★★★★**2026-08-24 갱신(doc-steward — F2 양성대조 결과 + kernel_mech
    rev8 규칙층 재감사 + NSL-1 rev3 규칙층 재감사 반영. GPU 지출 =
    결과 인용분 0.017 GPU-hr뿐[job 891612], 새 성능 판정 0건) 후보
    (i)는 하위 감사 항목 하나가 닫히고, (ii)는 새 하위-후보로
    분기하고, (iv)는 또 한 판이 `NO-GO`를 받는다.**
    (i) **P0-A** — 여전히 "완료된 도구 타당성 결과"(890893)이며
    새로 실행 대기 후보가 아니다. ★단 그 결과-audit이 남긴 유일한
    전제(§5-C4)가 **F2 양성대조**(job 891612, GPU 0.017 GPU-hr,
    `REPLAY_IS_THE_WRITER`)로 **닫혔다** — 성능 판정도 아니고
    S3·G1-d를 전진시키지도 않는다(위 P0-A 행 추기 참조). 이제 이
    하위 질문에 남은 감사 부채는 없다.
    (ii) **kernel_mech** — rev8(전체)도 규칙층 감사에서 `NO-GO`
    (차단 D1–D9, ★死因 여전히 없음). 감사 자신이 "Stage 0′조차
    오늘 구매 불가"(NVTX 패치 미작성이 가격에 미지수로 들어감)라
    판정해 후보가 **`Stage 0″`**(P0 + NVTX 없는 P3a, ≈0.05–0.1
    GPU-hr, 엔진 패치 0, `PREREG_STAGE0PP_2026-08-23.md`)로
    **더 좁아졌다** — 단 이것도 ★**자체 규칙층 통과가 필요**하며
    아직 감사받지 않았다. 다음 세션 후보는 **Stage 0″ 규칙층
    감사**(게이트 #34 1단) — 통과하면 등록된 예/아니오 3문항으로
    이 트랙이 이 기판에서 계속될 수 있는지가 처음으로 판정된다.
    kernel_mech rev3(Stage A 전용 설계)은 여전히 사용자 판단
    대기(rev9/보류/종결) 상태로 불변.
    (iii) **gate #16 rate 축**(규칙층 초안·재감사 전, 불변).
    (iv) **NSL-1** — rev3도 규칙층 재감사에서 `NO-GO`(死因 H1·H2,
    차단 H3–H11). rev2에서 rev3로 가며 **결함을 발견으로 승격**한
    것이 死因이었다(게이트 #21) — 감사 자신의 1순위 권고는 rev4
    작성이 아니라 **H2를 프로브로 승격**하는 것(telemetry 켜고
    `batch_is_full` 발화율을 rate×cap 격자에서 직접 측정, 1부팅
    ≈0.05 GPU-hr) — "cap이 정말 문다"를 확인하기 전엔 이 트랙의
    유일한 생존 자산이 없다. 다음 세션 후보는 **H2 프로브**(rev4
    전면 재설계가 아니다). S-6는 계속 목록 밖 별도 HOLD.
    ★★부수(성능 판정 아님, HE0 재확인): NSL-1 rev3 §7이 정본
    변화-trace 하네스의 실제 채점 다리(`mean`-ITL)를 지적했으나
    세 갈래 확인으로 **HE0는 흔들리지 않는다**(위 CONSENSUS §1-7
    addendum 참조) — 이 4-후보 목록의 우선순위에 영향 없음.
    ★**감사 부채 목록**(`reports/AUDIT_DEBT_2026-08-23.md`)이
    이 4-후보 목록과 별도로 층별 미감사 항목을 전수 추적한다
    (G13 하네스층·`g16_analyze.py` 재감사 등 — §6[메인 세션
    반복 실패 3회]은 이번 갱신으로 "방법론 게이트" #70·CONSENSUS
    §3 항목90 승격 완료).
    ★**gate #13·gate #16 "닫았다" 금지·switch-cost "닫았다" 금지
    불변** · **"kernel_mech 트랙을 닫았다"·"Stage 0″가 트랙을
    열었다" 금지 신설** · **HE0·정책 순위·인용정지 전부 불변** ·
    **새 성능 판정 0건.** 상세 `workspace/engine-port/results/
    bcg_probe/p0a_f2_verdict_891612.json`, `workspace/engine-port/
    results/kernel_mech/{audit_kernel_mech_rev8_2026-08-23/
    VERDICT.md, PREREG_STAGE0PP_2026-08-23.md}`, `workspace/
    engine-port/results/nsl_lever/audit_nsl1_rules_rev3_2026-08-23/
    VERDICT.md`.

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

## 방법론 게이트 (2026-07-28 신설, 2026-07-29 #4, 2026-08-02 #5·#6, 2026-08-03 #7 추가·#6 사례 추가·(3차 속행) #8 추가·(4차 속행) #9·#10 추가, 2026-08-05 #9 네 번째 재발 기록·#11 추가·(P1 운영점 대조 감사) #6 새 사례 추가·#12·#13 신설, 2026-08-06 #14 신설[통계 방법 층 정정, CONSENSUS §3 항목27과 대응]·(Gate 1) #9 다섯 번째 재발 기록·#15 신설[시간가중 step-function 추정량의 두 함정, CONSENSUS §3 항목28·29와 대응], 2026-08-07 (G1-b) #16 신설[스코프 확장은 원 격자를 전부 재현하라, CONSENSUS §3 항목30과 대응], 2026-08-09 (E-A) #17–19 신설[게이트를 모든 보고 블록에 걸어라·과부하 arm 비교는 시스템 상수가 아니다·사후 지정 셀 이동, CONSENSUS §3 항목31–33과 대응]·(Gate 2 rev4 본 캠페인 R1′/R2′ 정본 반영 복구) #20 신설[사전등록 분석기가 계산하지 않는 비교는 사후 비교다, CONSENSUS §3 항목34와 대응], 2026-08-09 (HOLB G5 재채점·874601 라벨 정정·scipy 폴백) #21 신설[측정 실패를 게이트 실패로 라벨링 마라 — 6번째 재발, 최초 정식 등재, CONSENSUS §3 항목35와 대응]·#22 신설[통계 라이브러리의 조용한 폴백은 아티팩트에 기록되지 않는다, CONSENSUS §3 항목36과 대응], 2026-08-10 (job 876699 T4-1 TIMEOUT 사후분석·공유 하네스 무한대기 수정) #21에 일곱 번째 재발 追記[이번엔 분석 코드가 거짓 음성(REFUTED)을 산출, CONSENSUS §3 항목35 개정과 대응]·#23 신설[공유 하네스 함수의 무경계 대기는 그 함수를 쓰는 모든 소비자의 위험이다, CONSENSUS §3 항목37과 대응], 2026-08-11 (Gate 2-S 첫 유효 결과, jobs 877756/877757, claims-auditor 적대 감사) #9에 여섯 번째 재발 追記[형식상 두 게이트가 같은 정보를 잼, CONSENSUS §3 항목18 개정과 대응]·#20에 새 사례 追記["식별자 수입 ≠ 거동 수입", CONSENSUS §3 항목34 개정과 대응]·#24 신설[any() over n reps 스크린의 귀무 발화율 1−(1−α)ⁿ, CONSENSUS §3 항목38과 대응]·#25 신설[사전등록이 명시한 진단 필드가 산출되지 않을 수 있다, CONSENSUS §3 항목39와 대응], 2026-08-11 (Gate 2-S 1차 실행 실패 후속, doc-steward) #26 신설[대형 캠페인 제출 전 배관 스모크 규율 — 0.11 GPU-hr 스모크(job 877593)가 6.40 GPU-hr 오판(jobs 877107/877109) 재발을 막음, CONSENSUS §3 항목40과 대응], 2026-08-11 (G1-c, job 877974) #27 신설[결정량의 밀도 의존성을 먼저 따져라, CONSENSUS §3 항목41과 대응]·#28 신설[실험이 무엇을 풀어주는지가 코드 사실인지 추정인지 실행 전에 구별하라, CONSENSUS §3 항목42와 대응]·#29 신설[`compute_coverage`류는 내부 구멍에 맹목이다, CONSENSUS §3 항목43과 대응], 2026-08-11 (E1 addendum, jobs 877756/877757, claims-auditor 적대 감사) #25에 여덟 번째 재발 追記[직전 회차 등재 직후 재발, CONSENSUS §3 항목39 追記와 대응]·#9에 별건 사례 追記[`falsifier()` docstring이 "IMPORTED"라 적었으나 인라인 복사·phase 필터 미적용, CONSENSUS §3 항목18 追記와 대응]·#30 신설[부정 선언문("NO UPGRADE PATH EXISTS") 옆의 새 통계량은 자기인지 자백만으로 재감사를 면제받지 않는다, CONSENSUS §3 항목44와 대응]·#31 신설[bound는 확률모델의 꼬리 분위수여야 하고 인접 관측 최대 차이(점추정)와 구별하라, CONSENSUS §3 항목45와 대응], 2026-08-11(트래픽·roofline 진단, result-analyst, GPU 0) #9에 일곱 번째 재발 追記[서술자 자수 — roofline 탄력도 정합은 항등식, CONSENSUS §3 항목18 追記와 대응]·#32 신설[다른 캠페인·다른 스케일의 보조 수치는 기준(basis) 검증 후 수입하라, CONSENSUS §3 항목46과 대응], 2026-08-14 (doc-steward, 문서 층 정리 — 새 성능 판정 0건) #9·#25 본문에 헤더가 이미 명시했던 追記 2건이 누락돼 있던 것을 CONSENSUS §3 항목18·39 원문 대조로 복원·#26에 追記[발동 조건을 "캠페인 하나"에서 "한 배치로 제출되는 신규/변경 코드 공유 캠페인들의 합"으로 개정 권고, CONSENSUS §3 항목47과 대응]·#33 신설[매니페스트 N/N sha 일치는 런타임 바이트 동일함을 함의하지 않는다 — 재현성 주장 범위를 매니페스트가 실제로 덮는 15파일로 한정, CONSENSUS §3 항목48과 대응], 2026-08-14 (E-3 realized SM count 프로브의 부수 발견, doc-steward, 2차) #32에 새 사례 追記[하드웨어 식별 층 재발 — glogin01(PCIe)에서 읽은 하드웨어를 컴퓨트 노드(SXM4) 캠페인에 잘못 귀속, CONSENSUS §3 항목46 追記와 대응], 2026-08-14 (4건 사전등록 감사 종합, doc-steward, 3차) #32에 새 사례 追記[C2 sd_rep(ε)를 basis 미검증 수입값에서 E-1a 원자료 직접측정값으로 교체 — 실패가 아니라 성공 사례, CONSENSUS §3 항목46 追記와 대응]·#34 신설[사전등록은 규칙 먼저 감사받고 하네스는 그 다음 별도로 감사받아라 — 2단 규율, LTSM P1·E1-b/c·%smid R0·E-1 4건 동시 발견, CONSENSUS §3 항목49와 대응], 2026-08-15 (doc-steward, E-1 rev1–rev3 + kernel_mech 4회 설계 전부 감사 차단에서 도출) #35 신설[개정판에서 손잡이 값을 유지한 채 유도 서사만 바꾸지 마라 — rev2 δ=0.0610/3=0.0203이 rev3에서 "오차예산" 유도로 갈아 끼워졌으나 숫자는 δ=0.020(반올림)으로 그대로였음, CONSENSUS §3 항목53과 대응]·#36 신설[타당성은 도구 문서·메트릭 DB로 확인한 뒤 설계하라 — kernel_mech §3이 존재하지 않는 green-context wave-분모 오염을 피하려다 `wave_eff≡1` 항등식을 만듦(#9 여덟 번째 재발), 확인 비용은 로그인 노드 명령 1줄·GPU 0, CONSENSUS §3 항목53과 대응], 2026-08-16 (doc-steward, C2-R rev1→rev2 5연속 감사 차단 종결에서 도출, 메인 세션 진단·claims-auditor 감사 없음) #37 신설[적대 감사에는 합격 기준과 단일 판정 질문을 함께 줘라 — 범위 없는 적대 검토는 항상 NO-GO를 산출하며 그건 설계 품질 신호가 아니다(단, rev1의 진짜 문제는 감사 과잉이 아니라 메인 세션의 설계 과잉이었음을 병기), CONSENSUS §3 항목55와 대응], 2026-08-16 (2차, C2-R 캠페인 결과, jobs 883574/883575, claims-auditor 적대 감사) #9에 아홉 번째 재발 追記[이번엔 "양성대조" 자체가 항등식 — 대조 코드 경로가 헤드라인 경로와 다르고 표적값 자체가 같은 루프의 또 다른 복사본 산출물, CONSENSUS §3 항목18 追記·56(C)와 대응], 2026-08-16 (3차, doc-steward — "상금 크기" 논증 감사 반영) #38 신설[달성된 동적의 열위(HE0)와 달성 가능한 천장은 다른 명제다 — 정본 §5-5·§1-17 본문이 이 둘을 혼동한 문장을 갖고 있었다(스코프 배너로 정정), CONSENSUS §3 항목57과 대응], 2026-08-16 (4차, doc-steward — C2R_RESULTS.md 역전파 결손, 코디네이터 지적) #39 신설[인용금지·강등 결정은 그 수치를 만든 원 아티팩트 문서로도 역전파하라 — 2회 재발("7.24s": CONSENSUS §1-4 본문→CLAIM_EVIDENCE_MATRIX Claim C·venue_positioning 미전파, 12일 소요; percentile CI: CONSENSUS §3 항목56(A)→C2R_RESULTS_2026-08-16.md §0/§7 미전파, 당일 발견), CONSENSUS §3 항목58과 대응], 2026-08-16 (5차, doc-steward — R1·R2 재분석 정본 승격, `ORACLE_REANALYSIS_2026-08-16.md` rev2, result-analyst + claims-auditor 적대 감사, GPU 0) #40 신설[결정량 자체가 항등식일 수 있다 — R2의 결정량②("SM합>108")가 `D_itl>D_ttft`와 동치인데 TTFT-argmax가 격자 최대 decode arm이라 데이터와 무관하게 거짓이었다(게이트 #9 열 번째 재발), 게다가 이 항등식은 `reports/PRIZE_SIZE_ARGUMENT_2026-08-16.md` §2.3(2)에 이미 문자 그대로 적혀 있었는데 결정량 설계에 적용되지 않았다(게이트 #18 사례[★오인용 정정
2026-09-10: 정본=게이트#14/`CONSENSUS.md` §3 항목27 — 상세 §3
항목27 追記]) — 결정량 설계 전 "데이터와 무관하게 참/거짓이 되는 극단 사례가 있는가"를 먼저 대수적으로 점검하고 저장소를 grep하라, CONSENSUS §3 항목59(신설)·39(追記, 재현 경로 결손 3회차·이번엔 등재 시점에 닫힘)와 대응], 2026-08-16 (6차, doc-steward — gate #16 사전등록 2단 감사 완주 반영, GPU 0·job 제출 0) #9에 열한 번째 재발 追記[이번엔 검증하는 쪽에서 재발 — 독립 검증 스크립트의 "PASS"가 순열 항등식이었고, claims-auditor 자신이 권고한 대체 결정량도 스스로 반증(C2→C2′), CONSENSUS §3 항목60(신설)과 대응]·#40 본문 문단 복구(헤더엔 있었으나 번호 문단 누락, 새 판정 아님)·#41 신설[telemetry는 공짜 관찰자가 아니다 — 2026-07-15/18 sgptv 격자는 telemetry 없이 돌아 이후 telemetry-ON 캠페인과의 절대값 직접 비교가 빌드 드리프트+계측 오버헤드의 합이 됨, "긴장 A(HE2 vs C2)"가 이 패턴의 기존 사례일 수 있어 목록만 등재(정정은 다음 세션), CONSENSUS §3 항목61(신설)과 대응]), 2026-08-16 (세션4, doc-steward — G16 스모크 2회 + 분석기 작성·감사·반영 + 병행 설계 감사 2건 NO-GO에서 도출, GPU 0.272 GPU-hr[스모크만]·새 성능 판정 0건) #42 신설[게이트가 자기 실패를 성공으로 라벨링할 수 있다 — 교훈 항목21의 거울상, G16 스모크 체커 9번이 예외를 삼키고 PASS를 찍었고 `OVERALL` 논리곱이 그 항목을 아예 참조하지 않았다, CONSENSUS §3 항목62와 대응]·#43 신설[분석기가 사전등록 기호를 조용히 재정의하면 자기 대조를 깨고 중심 산출물을 침묵시킨다 — `g16_analyze.py`가 §4의 `D_itl`을 §6 게이트-조건부로 재정의해 PC-C가 깨지고 §3 강제표가 미식별 시 자동 침묵했다(+메인 세션의 "사전등록 텍스트 결함" 오진 1건 기록), CONSENSUS §3 항목63과 대응]·#44 신설[양성대조의 빈 서명 구멍 — 대조가 estimand를 아예 실행하지 않았는데 통과할 수 있는 설계는 게이트 #9 계열의 취약점이다, `PC-B-neg` + 빈 서명 집합 통과 불가 규칙으로 해소한 사례, CONSENSUS §3 항목64와 대응]), 2026-08-17 (doc-steward — G16 캠페인 완료 반영, claims-auditor 적대 감사, GPU 0·이번 세션 신규 지출 0) #45 신설[감사 목적으로 쓴 검사 자신이 항등식일 수 있다 — 게이트 #40/#9 열두 번째 재발, `gap_upper`의 "노출차만으로 재현된다" 검사가 식 1개·자유모수 1개라 데이터와 무관하게 항상 풀린다, CONSENSUS §3 항목65와 대응]·#46 신설[사다리 해상도가 사전등록된 payoff 구간을 은폐할 수 있다 — 1ms 사다리가 폭 0.092ms 밴드의 부호 반전을 못 봐 초판이 "60ms에서 tax 없음"을 헤드라인으로 냄, CONSENSUS §3 항목66과 대응]), 2026-08-18 (doc-steward — G17·E-B1 규칙층 NO-GO 감사 + 메인 세션 자체 인용금지 위반 발견 + `results/slo_sched/` 디렉터리 오인 발견·정정 + 정본 `ILL-POSED` 판정 무시 재발견에서 도출, 새 성능 판정 0건) #47 신설[인용금지는 정본 등재만으로 전파되지 않는다 — CONSENSUS §33 승격금지(vii)가 33.04 인용을 금지한 다음 날 메인 세션이 새 문서에서 위반(교훈 항목41의 열두 번째 재발) [CS-OK], 기계적 차단 도구 `scripts/discipline/check_citation_stops.py`(추가 줄만 검사, 한계 병기) 신설로 대응, CONSENSUS §3 항목67과 대응]·#48 신설[결과 디렉터리 이름을 캠페인 이름으로 오인하지 마라 — `results/slo_sched/`는 G16 전용이 아니라 12캠페인 447 bench 파일 혼합(g16 68개뿐)인데 이를 오인해 "G16만 밴드질량 이상치"라는 사실 오류를 만듦, 전수 집계로 정정(G16 HI 67.47, 형제 3개가 같거나 위), CONSENSUS §3 항목68과 대응]·#49 신설[정본이 이미 `ILL-POSED`로 판정한 측정점을 재분석에 쓰지 마라 — 게이트 #18 최강 사례[★오인용 정정 2026-09-10: 정본=게이트#14/`CONSENSUS.md` §3 항목27 — 상세 §3 항목27 追記], `g2_0_hard/hardened_disjoint_verdict_2026-07-25.md`가 rA5를 TTFT 3s-cliff bimodality로 이미 `ILL-POSED` 판정했고 `g2_0_full/disjoint_verdict_2026-07-24.md`도 같은 d34 rep1/rep4를 절벽 사례로 이름까지 붙여 기록해 뒀는데, 그 판정서를 안 읽고 "d34 골짜기"를 발견처럼 재분석·보고(페어드 CI 0 포함·독립 재현에서 부호 반전으로 자체 반증), CONSENSUS §3 항목69와 대응], 2026-08-19 (doc-steward — gate #13 rev3
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
항목70과 대응], 2026-08-21 (doc-steward — gate #13 job-축 캠페인
완주·양 arm PASS 해제 절차 + seq3 부분 격자 확장 + S0(a) 감사 +
G18 rate-축 오염 발견에서 도출) #51 신설[블라인딩 감시 문자열을
설명하는 문서가 그 문자열 자체를 인용하면 자기 자신을 건다 —
`g13_analyze.py --unblind`가 문서 §2의 자기인용으로 자기거부(fail
-closed 방향이라 안전), CONSENSUS §3 항목71과 대응]·#52 신설[격자를
한 점만 넓히고 멈추면 그 자체가 격자 산물이다 — seq3가 k=14 한 점만
추가해 최저가·총액·격자 산물 크기 3수치가 부정확 인용(게이트 #16의
새 변종: "확장 안 함"이 아니라 "부분 확장 후 정지"), CONSENSUS §3
항목72와 대응]·#53 신설[비용 정합 없이 두 대조를 견주면 분모가
틀린다 — S0(a) 초판이 1 부팅쌍 대조를 2 부팅쌍 SD와 비정합 비교해
결론이 REFUTED, `s_alt` 대 `2σ_boot`로 재정합하면 답이 뒤집힘,
CONSENSUS §3 항목73과 대응]·#54 신설[태그는 오염을 막지 못한다 —
분석기의 glob/regex를 직접 확인해야 한다 — `g16_analyze.py:953`이
`g16_` 접두사를 재사용한 G18 프로브 run id를 정상 매치할 뻔함(실제
오염 0건, 접두사를 `g18probe_`로 교체해 차단), 게이트 #48의 자매
사례, CONSENSUS §3 항목74와 대응]), 2026-08-21 (2차 세션, `%smid`
R0 2층 감사 실행에서 도출, doc-steward 신설) #55 신설[감사 산출물을
파일로 남기지 않으면 조건이 소실된다 — 2026-08-14 `CONDITIONAL-GO
(5조건)` 원문이 저장소 어디에도 남지 않아 7일 만에 소실됨을 확인,
CONSENSUS §3 항목75와 대응], 2026-08-22 (doc-steward — `%smid` R0
결과 도착 + claims-auditor 결과 감사 `CONFIRMED(scoped)` 반영) #56
신설[보고 필드 이름이 값의 부정일 수 있다 — 게이트 #21의 보고 층
변종, N1(`.txt` 절단으로 `stop` 등 3필드 누락)·N2(`green_ctx_
attached`가 값의 부정), CONSENSUS §3 항목76과 대응]·#57 신설[결정론적
열거를 확률적 커버리지 증거로 쓰지 마라 — D1 사다리가 완전균등
(min=max=155)인 것은 재표집 증거가 아니다, CONSENSUS §3 항목77과
대응]), 2026-08-22 (전환 비용 재분석, 메인 세션, GPU 0) #58
신설[분석층 라벨의 의미를 코드와 대조하지 않고 추정하지 마라 —
`gap_class=="strict"`의 실제 정의는 `pend_min>0`인데 서술은 "다른
스트림 forward 없음"이었다(항목76의 쌍둥이 사례), CONSENSUS §3
항목78과 대응]·#59 신설[거의-상쇄 잔차를 상한으로 팔 땐 잔차/최대셀
비·독립 재구현 산포·1차 단위 재표본 CI를 함께 적어라 — `s`는 셀 값
1% 이동에 상한이 15% 움직인다, CONSENSUS §3 항목79와 대응]·#60
신설[두 분리 최빈값의 풀링 중앙값은 물리량이 아니다 — 혼합비가
구조적으로 고정이면 "20/20 동부호"는 항등적 사실이다, CONSENSUS
§3 항목80과 대응], 2026-08-23 (P0-A rev1–rev6 + S-6 rev1–rev4
규칙층 5회 감사 각각, claims-auditor, GPU 0) #61 신설[규칙을
산문으로 고정하면 구멍이 난다 — 결정 규칙은 코드로 고정하고 세계를
전수 열거하라, `p0a_rule_totality.py`·`s6_offleg_enumerate.py`가
유일하게 통한 대응, CONSENSUS §3 항목81과 대응]·#62 신설[변경
이력표가 하지 않은 수리를 적을 수 있다 — 각 행에 검증 방법을
병기하라, S-6 rev4는 차단 자체를 이력표·합격기준에서 소거하는 더
나쁜 변종, CONSENSUS §3 항목82와 대응]·#63 신설[안전망을 제거하는
개정은 양쪽 끝을 모두 복원해야 한다 — P0-A rev1이 `TOL` 밴드의
하한만 복원해 상한 결함이 5회 연속 재발, CONSENSUS §3 항목83과
대응]·#64 신설[결정량으로 승격한 입력에는 게이트·앵커·열거를 함께
붙여라 — 결정량을 옮기는 것은 결함을 고치는 게 아니라 옮기는
것일 수 있다, CONSENSUS §3 항목84와 대응]·#65 신설[검사가
load-bearing해 보이려면 세계 공간이 그 게이트를 발화시킬 수
있어야 한다 — 삭제 변이만으로는 문턱 약화 변이를 못 잡는다,
CONSENSUS §3 항목85와 대응]·#66 신설[포크는 원본의 하드와이어를
상속한다 — 매 회차 새 거부 지점이 나오면 아키텍처 신호다, S-6가
4회차에 걸쳐 독립 거부 경로 7건 발견, CONSENSUS §3 항목86과 대응],
2026-08-23 (NSL-1 rev3, claims-auditor, GPU 0) #67 신설[표에 단위를
셀에 적어라 — 행 라벨을 셀의 단위로 읽지 마라, 판정 부호가 뒤집힌
가장 비싼 형태, CONSENSUS §3 항목87과 대응]·#68 신설["이 손잡이가
구속하는가"를 Little 법칙 인구로 답하지 마라 — 구속성은 막힌
사건으로 재라, CONSENSUS §3 항목88과 대응]·#69 신설[metric cliff는
TTFT 다리에만 있지 않다 — 요청-내부 ITL p95도 자기 절벽을 가질 수
있다, CONSENSUS §3 항목89와 대응], 2026-08-23 (AUDIT_DEBT §6,
메인 세션 반복 실패 3회 + doc-steward 승격 판단, GPU 0) #70
신설[변경/설계 검증 칸에는 "무엇을 했다"만 적어라 — "무엇일
것이다"는 검증이 아니다(#62의 강화형), 같은 세션 안에서 3회
재발(3번째는 1번을 수리하는 항목 안에서 재발), CONSENSUS §3
항목90과 대응], 2026-08-25 (kernel_mech Stage 0‴ A0/A1 + NSL
①/E-A, claims-auditor 적대 감사 7회, doc-steward 정규화) #71
신설[어댑터 축이 다른 축에서 파생되면 그 자체가 차단이다 —
`l2_post`가 L4에서 파생돼 L4가 비면(등록된 결정적 음성) 답을 지운
뒤 그 답을 말하는 것 자체를 금지했다, CONSENSUS §3 항목91과
대응]·#72 신설[트랙 자신의 사전등록이 지목한 최소비용 선행 프로브를
사지 않은 채 더 비싼 후속을 반복하지 마라 — kernel_mech가 "사전등록
전 필수"라 적은 15분짜리 프로브를 9일간 안 산 채 rev1–rev8을
썼다, CONSENSUS §3 항목92와 대응]·#73 신설[검사 중복은 양방향으로
위험하다 — 전체 iff-분할 위의 라벨별 검사 N개는 1개 분량의 정보만
나르고, 접으면 메타검사가 공허해질 수 있다, CONSENSUS §3 항목93과
대응]·#74 신설[결론이 우연히 살아있는 것과 근거가 타당한 것은
다르다 — 여러 문서가 같은 미확인 전제를 상속하면 결론이 옳아도
근거를 다시 확인하기 전엔 확인됐다고 쓰지 마라, kernel_mech 6개
문서가 틀린 트리 grep을 확인 없이 상속(게이트 #32의 grep 층 변종),
CONSENSUS §3 항목94와 대응]·#75 신설[출처 허위는 규칙 정본 파일
안에서 가장 위험하다 — 다음 판본이 검증 없이 그대로 승계할 수
있다, `K1_HIGH_AT = 2.00  # rev8 registered this`인데 rev8에 `K1`
0건, CONSENSUS §3 항목95와 대응], 2026-08-26 (kernel_mech A1
sticky 재설계 규칙층 감사 3회 + NSL ③②E-B 묶음 첫 규칙층 감사,
claims-auditor 적대 감사 4회 + doc-steward 승격 판단, GPU
0.014 GPU-hr[스모크 job 893663뿐]) #76 신설[인용/드리프트 검출
도구 자신이 거짓 인증을 낼 수 있다 — `check_line_citations.py
--snapshot`이 편집된 파일의 기준선을 조용히 갱신해
`a1_q3k1_rule.py:83`이 `N_MIN_DECODE_STEPS`를 가리킨다는 인증을
한 커밋 만에 발화시켰다(실제 `:105`), 재베이스 거부+고아 탐지로
부분 수리, CONSENSUS §3 항목96과 대응]·#77 신설[추정량이 코드
경로의 평가 순서에 의존하면 그것은 귀속이 아니라 순서통계량이다 —
NSL D3(a)(b), `cap` 검사가 매 반복 최상단에서 평가돼 KV 경로는
cap 불발 시만 도달하고 `OTHER` 출구는 사이트 목록에도 안 잡힌다,
계수 설계 전 제어흐름의 상호배제·평가 우선순위를 먼저 등록하라,
CONSENSUS §3 항목97과 대응]·#78 신설[계측 완전성은 사이트
목록이 아니라 기전 목록에서 판정하라 — NSL D2, `batch_is_full`을
세우는 자리와 admission을 막는 기전은 다른 집합이라 유일한
TOOLLIMIT 술어가 관측 불가/상시발화 중 어느 쪽인지를 하네스가
조용히 정한다, CONSENSUS §3 항목98과 대응]·#79 신설[de-confound
처방 자체가 추정량의 자유 모수일 수 있다(confound #10의 메타
형태) — NSL D3(c)/B2, `--max-mamba-cache-size` 고정처럼 손잡이를
순화하는 처방의 파라미터 선택이 측정 대상(KV 예산)을 방향성 있게
이동시킬 수 있다, CONSENSUS §3 항목99와 대응]·#80 신설["수리는
국소, 주장은 전역"(repair-local truth vs. document-global
claims) — 각 수리는 지목된 좌표에서 실재하나 그 파급을 재도출하지
않아 다음 회차 결함 대부분이 직전 수리의 그림자가 된다(A1 3회차
감사 명명, 이 세션 8건 자기수정 중 다수가 이 형태이고 NSL §4.1의
교훈-오사가 세션 내 재현), 부수형 "세계모형이 실험설계 성장을
못 따라간다", CONSENSUS §3 항목100과 대응], 2026-08-26(追記) #21에
NSL D1 재발 追記[`n_firings=4000`이어도 `saturated="no"`면 실질
성공(등록 rate-3 부팅 4개)을 `WORKLOAD_NOT_SATURATING`으로 버림 —
게이트가 반대 방향으로 열림, CONSENSUS §3 항목35 追記와 대응]·
#73에 두 번째 사례 追記[NSL D3(c) — 거울 대칭 검사(`cap_share`
0.2 vs 0.8)를 "단독구속 0"을 근거로 삭제했으나 그 0은 격자에
0.20·0.80이 없어 생긴 인공물이었다, CONSENSUS §3 항목93 追記와
대응], 2026-08-28 (doc-steward — TC1/M4R 규칙층 감사 각 2회 +
설계층 도달가능성 메타검사 신설 + 제출 게이트 신설에서 도출)
#81 신설[도달가능성 검사가 격자 안에서만 돌면 설계가 답을 미리
정해 놓아도 통과한다 — 게이트 #40(결정량 자체가 항등식일 수
있다)의 설계층 판본, `T2_reachable`류는 축 격자의 성질이고
실험이 만들 수 있는 세계는 그 부분집합인데 부분집합에서 실질
라벨이 0개(`NOTHING_PURCHASABLE`)거나 1개(`SINGLE_LABEL_FORCED`)면
결정량이 항등식이 아니어도 결정은 항등식이다, TC1 rev2(시나리오
A)·M4R rev2 두 트랙에서 독립 발화, CONSENSUS §3 항목101과 대응]·
#82 신설[번호 참조 체계가 넷(`CLAUDE.md` #1–8·`PROJECT_STATUS`
#1–80·`CONSENSUS §3` #1–100·메모리 topic #1–80)인데 사전등록이
둘로만 세면 "정정문" 자체도 오인용을 재생산한다 — TC1 rev1이
`stage0ppp_a0_rule.py:2`의 게이트 #66 오인용을 지적하며 쓴
정정문이 같은 문장 안에서 다시 틀렸다(항목66이라 적고 `VERIFIED`
딱지, 정답은 항목81/게이트 #61), 번호를 쓸 때는 항상 체계
이름을 병기하라, CONSENSUS §3 항목102와 대응], 2026-08-28(같은 날
2차, doc-steward — `deprecated_v2/README.md` 비준 + TC1 rev3
규칙층 감사(F8–F12) + 도구 자기감사(commit `5d180a6`)에서 도출)
#83 신설[**엔진이 강제하는 상호배타 지원영역은 설계 선택으로 못
넘는 교락이다** — TC1이 Zamba2-2.7B에서 Nemotron-Nano-9B-v2로
모델을 바꾸는 과정에서 Nemotron-H 계열+`triton`=부팅 거부,
Zamba2+`flashinfer`=스케줄러 사망(2/2, job 896776)이 확인돼
"모델 고정, 백엔드만 변경" 셀이 **존재할 수 없음**이 실측으로
드러났다. 이는 저자가 고른 confound(#5의 `--max-running-requests`
교차-arm 결합처럼 실험자가 두 손잡이를 하나로 묶은 경우)가
아니라 **엔진의 지원 행렬 자체가 두 축의 곱공간에서 서로소인
경우**라 어떤 사전등록 설계도 그 셀을 살 수 없다 — de-confound
처방(항목5 계열)이 통하지 않는 사례. ★검증: 이 게이트는 흔히
"게이트 #10(변수 동시 변경)"의 변종으로 불릴 뻔했으나, **#10의
실제 정본 텍스트는 "여집합 클래스에 음성대조를 걸어라"**(위 항목
10, 2026-08-03)이지 "동시 변경 금지"가 아님을 이번에 확인했다 —
그 오인용은 이 저장소 여러 문서(`CEILING_CENSORING_DIAG_2026-08-05.md:564`·
`bsweep_regime/PREREG_E1_REV3_2026-08-15.md:175`·이 문서 "다음
실험 gate" #9(ceiling-censoring 후속) 절 원문 "동시 변경 금지
(방법론 게이트 #10)")에 이미 산발적으로 있었다(수정은 이번 비준
범위 밖, 발견만 기록). ⇒ #83은 #10의 변종이 아니라 **별개
게이트**로 등재하며, 가장 가까운 기존 선례는 #5(`--max-running-
requests`가 admission과 mamba pool을 동시에 움직여 hybrid arm
cap 효과가 T8로 이전 불가한 사례) — 그러나 #5는 "한 캠페인 안에서
손잡이가 겹친다"이고 #83은 "두 캠페인을 잇는 셀 자체가 없다"라
스코프가 다르다. 실무 규칙: 모델×백엔드처럼 엔진이 검증하는
조합 축을 두 개 이상 다루는 설계는, 격자를 그리기 전에 **각
축의 실제 지원 조합(부팅 스모크)을 먼저 전수 확인**하고, 서로소로
드러나면 "고정" 대상 축을 지우는 대신 **그 축 자체가 비교
불가능함을 스코프 배너로 등재**하라(deprecated_v2/README.md §1이
그 최초 사례), CONSENSUS §3 항목103과 대응]), 2026-09-08
(cp_baseline/D1 트랙 규칙층 3판본[rev1→rev2→rev3] 적대 감사 3회
전부 `NO-GO`[트랙 누적 10연속] + 게이트 #14 재발 기전 확정에서
도출, claims-auditor + doc-steward, GPU 0) #14에 네 번째 재발
追記[게이트가 정본에 등재돼도 도구가 반대로 안내하면 재발한다
— 정본 coverage를 이 세션이 독립 재현, CONSENSUS §3 항목27
追記와 대응]·#93 신설[다중비교 보정은 강도 `m`을 함께 등록해야
완결된다, CONSENSUS §3 항목113과 대응]·#94 신설[기각과 동등성은
반대 방향의 보수성을 요구한다, CONSENSUS §3 항목114와 대응]·
#95 신설[두 설명이 관측 동치인데 한쪽을 기각하려면 변수 하나만
바꿔야 한다, CONSENSUS §3 항목115와 대응]·#96 신설[판별식을
자기 저작 픽스처로 대체하면 양성대조가 된다, CONSENSUS §3
항목116과 대응]·#97 신설[항목18[=§3 항목27, ★오인용 정정
2026-09-10]/#14의 강화형 — 규율은 도구에
심어야 전파된다, CONSENSUS §3 항목117과 대응])

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
   ★**追記(2026-09-11, G-5, R2 true-dual GPU correctness 판정서[`workspace/
   engine-port/results/r2_correctness/audit_r2corr_2026-09-11/VERDICT.md`]
   §6, claims-auditor, GPU 0)**: "realized로 검증하라"를 한 단계 더
   좁혀라 — 파티션 실현은 **"창 안에 그 분할이 있었다"가 아니라 "비교
   대상 산출의 어느 계산이 실제로 그 분할 위에서 돌았는가"**로 적어야
   한다. job 907100에서 D44 64-SM(prefill) 측은 probe prefill(비교
   대상)이 돌았지만, D44 44-SM(decode) 측에서는 **비교 대상에서 제외된
   bg decode만** 돌았다(probe decode 47 step은 비분할 plain stream
   idx 5에서 실행) — "overlap 스냅샷 100%가 idx 4(=D44)"라는 target-
   vs-realized 검사를 통과해도, 토큰 동치가 실제로 확인하는 범위는
   그보다 좁다. 대응 `CONSENSUS.md` §1-9(추기)·§3 항목130(추기).
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
    ★**追記(2026-09-08, cp_baseline/D1 트랙 규칙층 감사 9회차,
    claims-auditor + 메인 세션 독립 재현, GPU 0) — 네 번째 재발,
    이번엔 "등재만으로는 전파되지 않았다".** D1 rev2가 이 게이트를
    한 번도 참조하지 않은 채 처음부터 같은 percentile bootstrap을
    다시 채택했다(死因 V2, 정본 coverage n=4 .798/n=5 .840/n=6
    .859/n=8 .888을 이 세션이 독립 재현 — .802/.838/.860/.882).
    ★기전 확인: `benchmarks/pdmux_eval/analyze.py`(이 게이트가
    지목한 바로 그 파일)는 지금도 **t-CI 함수가 아예 없고**
    docstring `:153`이 `paired_bootstrap_ci`를 **권장**(n≤8 제한
    언급 0), CLI `:379`가 **n 가드 없이** 호출한다 — 게이트는
    이 문서에 있고 도구는 정반대를 말한다. 도구는 이번에도
    **고치지 않았다**(여러 트랙 공유, 파급이 트랙 밖 — 사용자
    승인 필요). 신규 게이트 #97(§3 항목117)로 별도 등재. 상세
    `workspace/engine-port/results/cp_baseline/
    FINDING_GATE14_TOOLING_2026-09-08.md`,
    `reports/AUDIT_DEBT_2026-08-23.md` §10.
    ★★**追記(2026-09-08, 2차 세션, 같은 날 연속, doc-steward) —
    수리가 브랜치에서 시작됐으나 게이트는 닫히지 않았다.** 사용자
    지시로 `benchmarks/pdmux_eval/analyze.py`에 t-CI 3함수를 별도
    브랜치 `fix/gate14-tci-analyze`에서 추가(main은 원본 397줄
    그대로, 사용자 지시: "기존 라이브러리 유지, 다른 브랜치로").
    검증(메인 세션 독립 재실행): 행동 보존(main 블롭 대조
    5,100키 불일치 0)·회귀 314 tests OK·변이 4/4 killed. 설계서
    규칙층 감사 **`NO-GO`**(死因4·차단12, `results/tooling_gate14/
    audit_design_2026-09-08/VERDICT.md`) — **死因 F1만 수리**
    (`fc87a17`): 판정 경로에 n 하한이 없어 수리 직후에도 **n=2에서
    `headline_improvement=True`**였고, 게이트 #3 하한을 *추가*하는
    변이(X9)를 이 수리가 함께 낸 새 테스트(`test_analyze_gate14.py:
    374-383`, n=3 pair로 `headline_improvement is True` 단언)가
    **직접 죽였다**(게이트 #9의 교과서적 형태 — 테스트가 수리가
    아니라 구현의 현 상태를 지킴). F2(`GATE14_DECISION_MIN_N=9`가
    판정과 무관한 죽은 상수)·F3(§3 "유일하게 남는 설계"가 거짓 —
    `d1_predicates.py`의 P5c형 코드객체 검사가 이미 더 강함)·
    F4(§6이 "이 수리 스코프 밖" 재검토 대상으로 든 인용값 자체가
    오귀속 — `analyze.unpaired_bootstrap_ci`가 아니라
    `oracle_reanalysis_2026_08_16.py`의 로컬 재표집 루프가 생산)
    + 차단 12건은 **열려 있다**(B1: CLI `level`이 새 연속 자유
    표면, B3: `GATE14_BOOTSTRAP_COVERAGE` 검사 0 등). ★**"게이트
    #14를 닫았다"는 여전히 금지** — main은 미변경이고 브랜치도
    `NO-GO`다. `reports/AUDIT_DEBT_2026-08-23.md` §10.3 P0-4 상태를
    **부분 이행**(브랜치 존재·main 미병합)으로 갱신. ★부수:
    감사 지적(B5, "venv에 scipy 없음"이 거짓)도 **과장**이었음을
    메인 세션이 재확인(venv 자체엔 없고 `~/.local`+
    `--system-site-packages`로 보임 — 문장은 참, 함의만 거짓) —
    같은 전제가 **7개 파일 10개 지점**에 상속(전부 미수리, 트랙
    밖 승인 필요). 신규 방법론 게이트 후보 3건은 아래 "방법론
    게이트" #98–100(신설) 참조. GPU 지출 0. 상세
    `workspace/engine-port/results/tooling_gate14/
    {DESIGN_ANALYZE_GATE14_2026-09-08.md, BACKLOG_2026-09-08.md,
    CORRECTIONS_2026-09-08.md, audit_design_2026-09-08/VERDICT.md}`
    (브랜치 `fix/gate14-tci-analyze` 전용, main 미병합).
    ★★**追記(2026-09-10, doc-steward, 사례 3, 다른 곳에서는
    "게이트 #18"로도 인용됨[★같은 날 후속 정정으로 확정: 이는
    오인용이었다 — 정본 번호는 게이트#14/`CONSENSUS.md` §3 항목27뿐.
    상세 §3 항목27 追記]) — 이번엔 "선행연구 발견"이 최초
    발견이 아니었다.** `longctx_conflict` 트랙 2026-09-09(2차
    세션)이 `muxwise/{sharegpt.yml, loogle.yml}`의 워크로드별 SM
    분할표를 §5-6·"다음 실험 gate" longctx_conflict 행에
    "선행연구 발견"으로 등재했으나, 같은 표(loogle 3행 포함)는
    **2026-08-28** `reports/impl_vs_external_pdmux_2026-08-28.md`
    §2.5–2.6에 이미 문서화돼 있었다 — 저장소가 같은 진단을
    **12일 간격으로 두 번** 냈고 정본은 첫 번째 시점에 흡수하지
    않았다. 이 항목의 패턴 그대로라 **새 게이트 번호는 매기지
    않고 이 항목의 사례로 등재**한다(doc-steward 판단). 동시에
    그 재발견 자체가 담고 있던 사실 오류("같은 하드웨어") — 두
    yml 전 행이 `prefill+decode=132`(H100/H200급, A100 108-SM
    아님) — 도 정정했다. 상세 이 문서 최상단 갱신 로그의
    2026-09-09(2차 세션) 항 (G) 항목 追記, `CONSENSUS.md` §3
    항목27 追記·§5-6 追記(2026-09-10).
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
    대응)[★오인용 정정 2026-09-10: 이 문단 전체가 이중으로 틀렸다 —
    (a) 이 문단이 실제로 비교하려는 개념("저장소가 같은 진단을 두
    번 냈는데 정본이 안 바뀌면 도구 규율 실패")은 게이트 #21이 아니라
    **게이트 #14**다. (b) 게이트 #21의 실제 대응은 §3 항목18이 아니라
    **§3 항목35**다(2026-08-09 신설, "측정 실패를 게이트 실패로
    라벨링 마라"). 올바른 비교 대상은 "게이트 #14/`CONSENSUS.md` §3
    항목27"이다. 상세 §3 항목27 追記]과는 **축이 다르다** —
    #21/항목18[=게이트#14/항목27]은 "진단이 반복되는데
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
    않았다(**게이트 #18 사례**[★오인용 정정 2026-09-10: 정본=
    게이트#14/`CONSENSUS.md` §3 항목27 — 상세 §3 항목27 追記]).
    실무 규칙: 결정량을 설계하기 전에
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
    세션엔 정정하지 않는다. 분리하려면 telemetry-OFF 대조 부팅이
    필요하다. ★★**정정(2026-08-21, doc-steward — S-6 검정력 계산,
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
    아직 없다.** 상세 `PREREG_G16_RULES_REV3_2026-08-16.md`
    addendum B-2(N7, 정정 표시 병기), `reports/CONSENSUS.md` §3
    항목61(정정)·67(追記), `handoff-report/session_handoff_2026-08-16.md`
    §15.5, 아래 "다음 실험 gate" #17 2026-08-21 갱신(2차).

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

    ★**(2026-08-21, 2차 세션, `%smid` R0 2층 감사에서 발견 — 극단형
    재발) 새 사례: 관측이 0인 것 자체가 공허참으로 PASS할 수 있다.**
    2026-08-16의 원형은 "예외를 삼켜 실패를 성공으로" 였고 이번은
    **더 깊다** — 예외가 나지 않고 코드가 정상 종료했는데, 입력이
    빈 집합이라 **논리식이 공허참으로 항상 참**이 된다: `%smid` R0
    스코어러가 빈 census(전 union이 ∅)를 받으면 `∅∩∅=∅`이므로
    `disjoint` 판정을 통과, `∅=∅`이므로 `tiles_D` 판정을 통과,
    `_saturated`가 `∅→∅`(증가 없음)를 "포화"로 판정해 D1 하드
    게이트까지 통과 — 그 결과 **관측 0건인 실행이
    `GLOBALLY_CONSISTENT_LABEL`+`stop=False`를 내고 R1·R2·R5 해석이
    개시될 수 있었다**(메인 세션 직접 재현 확인:
    `score(_synth([0,0], dsize=0))` → `GLOBALLY_CONSISTENT_LABEL`,
    `facts.sizes=[0,0]`). ★기존 fail-open 수리(죽은 코드 2건)가
    **손대지 않은 자리**였다 — 서로 다른 결함이 같은 파일에 공존.
    실무 규칙: (i) 게이트/스코어러를 설계할 때 "입력이 텅 비면 이
    판정이 자동으로 무엇을 내는가"를 **집합 연산마다** 대수로
    점검하라(공허참은 대개 원하는 방향으로 나온다 — 위험하다).
    (ii) 해소는 논리를 고치는 것이 아니라 **양성 하한을 포화·일치
    검사 앞에** 두는 순서 변경이다(관측 개수가 0이면 그 자체를
    별도 결과로 분리, 포화 판정에 도달시키지 않는다). 상세
    `workspace/engine-port/results/smid_census/audit_smid_r0_
    2026-08-21/VERDICT.md` C1, `handoff-report/session_handoff_
    2026-08-21b.md` §2.2.

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
    성능 판정 아님, 방법론 게이트 #18 최강 사례[★오인용 정정
    2026-09-10: 정본=게이트#14/`CONSENSUS.md` §3 항목27 — 상세 §3
    항목27 追記]) 정본이 이미
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
    연 것"**이라 게이트 #18[★오인용 정정 2026-09-10: 정본=게이트#14/
    `CONSENSUS.md` §3 항목27 — 상세 §3 항목27 追記]("저장소가 같은 진단을 두 번 냈는데 정본이
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
    ★**追記(2026-09-11, G-3, R2 true-dual GPU correctness 판정서
    §1.6·§6, claims-auditor, GPU 0 — 이 항목의 측정 층 판본)**:
    자기 수리의 검증 검사만 항등식일 수 있는 게 아니라, **동치
    게이트의 검출력(민감도)도 "스코어러가 항상-통과 변이본을
    잡아낸다"는 검사(하네스 v2의 e16e93f mutation test)만으로는
    보증되지 않는다** — 그 검사는 스코어러 층만 재고, 엔진+프롬프트
    라는 측정 층 자체의 민감도는 재지 않는다. job 907100의 O probe
    출력 8/8이 passage 그대로의 복사라 argmax 마진이 거의 전 구간
    작았는데도(복사형 출력 구간에서만 마진이 남음), "스코어러는
    항상-통과를 잡는다"는 확인은 이 사실을 드러내지 못했다. 실무
    규칙: 동치 게이트를 등록할 때는 스코어러 변이 테스트와 별개로,
    알려진 미세 수치 섭동이 실제로 불일치를 1건 이상 만드는지
    확인하는 **측정 층 양성대조**를 사전등록하라(예: X1,
    `SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS=true`). 대응
    `reports/CONSENSUS.md` §3 항목70(追記).

51. ★★**(2026-08-21, gate #13 캠페인 해제 절차, 메인 세션, GPU 0·새
    성능 판정 아님) 블라인딩 감시 문자열을 설명하는 문서가 그 문자열
    자체를 인용하면 자기 자신을 건다.** `g13_analyze.py --unblind`는
    `PREREG_G13_2026-08-20.md`에 `AUDIT_STATUS: PENDING` 감시 문자열이
    있으면 거부하도록 짜여 있었는데, 첫 해제 시도에서 그 문서 §2가
    블라인딩 기전을 **설명하며 감시 문자열을 그대로 인용**해 자기
    자신을 걸었다. **fail-closed 방향**이라 안전했다(거짓 unblind를
    허용하는 실패가 아니라 정당한 unblind를 거부하는 실패). 해소:
    분석기 SHA(`01448211…`)를 지키기 위해 문서 쪽 인용을 우회
    서술로 재작성. 증거 `G13_ANALYSIS_2026-08-21_REFUSED.json` 보존.
    실무 규칙: 자동화된 감시 문자열을 설명 목적으로 문서에 적을
    때는 감시기가 검색하는 정확한 형태로 인용하지 말 것(paraphrase).
    상세 `workspace/engine-port/results/s8_scaleup/G13_RESULTS_
    2026-08-21.md` §5, `reports/CONSENSUS.md` §3 항목71.

52. ★**(2026-08-21, seq3 8티어 확장, 메인 세션) 격자를 한 점만 넓히고
    멈추면 그 자체가 격자 산물이다.** seq3 결론("sequential은 예약액을
    못 낮춘다")은 감사가 8티어×2arm×2m으로 깨뜨리려다 실패해 생존
    했으나, 메인 세션이 격자를 **k=14 한 점만** 추가하고 멈춘 탓에
    최저가(**k=13/3.12**, 인용값 3.12 아닌 값이 아님 — 총액 **10.80**,
    인용값 11.04는 stale)·격자 산물 크기(**0.72**, 인용값 0.48은
    stale, m축까지 넓히면 1.76)가 전부 부정확하게 인용됐다. 방법론
    게이트 #16(스코프 확장은 원 격자를 전부 재현하라, 2026-08-07)의
    재발이지만 이번엔 "확장을 아예 안 함"이 아니라 **"확장을 부분적
    으로만 하고 멈춤"** 이라는 새 형태다. 실무 규칙: 격자를 넓히는
    실험은 넓힌 축의 전 범위를 스윕하거나, 못 하면 그 사실 자체를
    결과에 명시하라(부분 확장에서 나온 최저가·산포 인용 금지). 상세
    `handoff-report/session_handoff_2026-08-21.md` §2.3, `reports/
    CONSENSUS.md` §3 항목72.

53. ★**(2026-08-21, S0(a) 감사, claims-auditor) 비용 정합 없이 두
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
    s8_scaleup/S0A_VERDICT_2026-08-20.md` §0, `reports/CONSENSUS.md`
    §3 항목73.

54. ★★**(2026-08-21, G18 rate-축 정본 오염 발견, 메인 세션) 태그는
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
    2026-08-20.md` §6, `reports/CONSENSUS.md` §3 항목74.

55. ★★**(2026-08-21, 2차 세션, `%smid` R0 2층 감사 실행에서 도출,
    doc-steward 신설) 감사 산출물을 파일로 남기지 않으면 조건이
    소실된다.** 2026-08-14 감사가 `%smid` R0에 낸
    `CONDITIONAL-GO(5조건)`은 원문이 저장소 어디에도 남지 않았다 —
    "다음 실험 gate" #11 레지스트리의 **한 줄**과
    `handoff-report/session_handoff_2026-08-15.md`의 **두 줄**(개수와
    결함 2건만)이 전부였고, 정작 그 판정이 나온 2026-08-14 날짜의
    핸드오프 파일 자체가 **존재하지 않는다**. 2026-08-21에 다음
    세션이 이 조건들을 다시 읽으려 했을 때 **7일 만에 원문이
    소실**돼 있었다 — 2층 감사가 조건을 "복원"이 아니라 **처음부터
    새로 도출**해야 했다(운 좋게 결론은 겹쳤으나 그것을 보장하는
    장치가 없었다). ★이 저장소에는 같은 결손을 가진 설계가 **6건**
    더 있다(LTSM P1·E1-b/c·E-1·G13·kernel_mech rev2·G17/E-B1 —
    "다음 실험 gate" #11 레지스트리의 "감사 보고서(있는 것만)" 문단
    참조, 판정어만 레지스트리에 남고 조건 원문은 없다). 실무 규칙:
    (i) 규칙층·하네스층 감사가 차단/조건부-통과 판정을 낼 때는
    판정 요약을 정본에 등재하는 것과 **별개로** 감사 산출물 전문을
    파일로 저장한다(`results/<트랙>/audit_<이름>_<날짜>/VERDICT.md`
    형식 — 이번 세션이 `%smid` R0(`audit_smid_r0_2026-08-21/
    VERDICT.md`)·kernel_mech rev5(`audit_kernel_mech_rev5_2026-08-21/
    VERDICT.md`)에서 이 규율을 시작했다). (ii) 그 파일 안에 **왜
    이 파일이 존재하는지**(선행 판정의 소실 여부)를 명시해 다음
    세션이 같은 조사를 반복하지 않게 한다. (iii) 기존 6건에 이
    규율을 소급 적용할지는 **별건 판단 대상으로 등재**한다(이번
    세션은 하지 않았다 — 아래 "다음 실험 gate" #11 레지스트리
    참조). 상세 `workspace/engine-port/results/smid_census/
    audit_smid_r0_2026-08-21/VERDICT.md`(머리말 + caveat 4),
    `reports/CONSENSUS.md` §3 항목75, `handoff-report/
    session_handoff_2026-08-21b.md` §2.2.

56. ★★**(2026-08-22, `%smid` R0 결과 감사, claims-auditor, GPU
    0.0275 GPU-hr) 보고 필드의 이름이 값의 부정일 수 있다 — 극성
    필드는 이름과 값을 일치시키고 설명 문자열에 의존하지 마라
    (설명 문자열은 잘릴 수 있다).** `smid_l0_verdict_889631.json`의
    `setup.green_ctx_attached`는 값이 `green_ctx_is_null`이라
    **이름이 값의 정반대**다(`false`가 "green ctx가 붙어 있다"는
    뜻). 자매 필드 `plain_control_detached`는 정반대 규약(`true`가
    "안 붙어 있다")을 쓴다 — 같은 판정서 안에서 극성이 필드마다
    뒤집혀 있다. ★**합성 효과**: 이 결함이 N1(아래)과 겹치면
    `.txt`만 읽는 독자는 *"green pair에 green ctx가 안 붙었다"* =
    **진실의 정반대**로 읽는다. 같은 판정서의 **N1**: `.txt` 판정서가
    `smid_l0_census.py:1381`의 `[:4000]` 절단으로 잘려 `stop`·
    `plain_control_detached`·`control_status` 세 필드를 통째로
    누락한다(메인 세션 확인: `.txt` 4127B vs `.json` 4777B, grep
    0건). 실무 규칙: (i) 불린 필드 이름은 그 값이 참일 때의 의미로
    짓는다(`green_ctx_is_null`처럼 이미 그렇게 지은 필드와 이름만
    다른 `_attached`류를 병존시키지 않는다). (ii) 사람이 읽을
    판정서를 별도 포맷(`.txt`)으로 파생시킬 때 절단·필드 누락을
    자체 검사(누락 필드 목록을 `.json`과 대조)하지 않으면 그
    파생물은 인용 금지 대상이다. ⇒ **정본·논문·후속 문서는
    `smid_l0_verdict_889631.json`만 인용, `.txt`/`.out`은 인용
    금지**(이번 결과에 한정, 다른 판정서의 `.txt` 파생물도 같은
    점검 없이 인용 금지로 일반화하는 것은 별건 판단 대상). 상세
    `workspace/engine-port/results/smid_census/{smid_l0_verdict_
    889631.json, smid_l0_verdict_889631.txt}`, `reports/CONSENSUS.md`
    §3 항목76.

57. ★**(2026-08-22, `%smid` R0 결과 감사, claims-auditor) 결정론적
    열거를 확률적 커버리지 증거로 쓰지 마라.** R0의 D1 하드 게이트는
    관측 SM id 히트 분포가 그리드 점마다 완전 균등(`min_hits==
    max_hits==155`, `S_pre`/`S_post`/idx0/idx3 전부)함을 "사다리
    평탄"의 근거로 삼는다 — 그러나 이 스윕은 커널을 **결정론적으로
    반복 열거**하는 설계라 균등성은 **재표집(resampling) 하에서의
    안정성 증거가 아니다**(무작위 표본이 아니므로 신뢰구간·검정력
    개념이 적용되지 않는다). ★**등재만**(이번 세션은 R0 판정
    자체를 흔들지 않는다 — D1이 실제로 요구하는 것은 "그리드
    마지막 두 점에서 동일 집합"뿐이고 이 조건은 결정론적 열거로도
    충분히 검증된다, 단 "포화"라는 단어가 확률적 함의를 암시하지
    않도록 향후 서술에서 주의). 상세 `workspace/engine-port/results/
    smid_census/smid_l0_verdict_889631.json`(`D1` 블록), `reports/
    CONSENSUS.md` §3 항목77.

58. ★**(2026-08-22, 전환 비용 재분석, 메인 세션 — GPU 0) 분석층
    라벨의 의미를 코드와 대조하지 않고 추정하지 마라.** `holb_
    probe.py:71-72`에서 `gap_class=="strict"`의 실제 정의는
    `pend_min > 0`(간극 내내 decode 대기)이지 문서가 서술해 온
    "다른 스트림 forward 없음"이 아니다 — strict의 **7.16%**
    (11,114/155,141)가 실제로는 `n_other_fw>0`이다. #56(보고
    필드 이름이 값을 배반)의 분석층 쌍둥이 — 이번엔 필드 이름이
    아니라 **층 라벨 이름**이 코드 정의와 다른 것을 가리켰다.
    상세 `reports/CONSENSUS.md` §3 항목78, `workspace/engine-port/
    results/kernel_mech/audit_step0_2026-08-22/VERDICT.md` 결함1.
59. ★★**(2026-08-22, 전환 비용 재분석) 거의-상쇄 잔차를 상한으로
    팔지 마라 — 잔차/최대셀 비와 독립 구현 재현 산포를 함께
    적어라.** 가법 식별 `d+2s`가 최대 셀(1.29ms)의 **5.8%**뿐일 때,
    그 절반값을 "상한"으로 3자리까지 표기하면 허위 정밀도가
    된다(독립 재구현 3개 산포 ±13%, 절사평균으로 부호 반전, 20개
    중 2개 음수). 상한 등재 시 ① 잔차/최대셀 비율 ② 독립 재구현
    산포 ③ 아티팩트 단위 재표본 CI를 병기하고 유효숫자는 그
    산포가 지지하는 자릿수로 절사한다. 상세 `reports/CONSENSUS.md`
    §3 항목79, `workspace/engine-port/results/kernel_mech/
    audit_promotion_2026-08-22/VERDICT.md` 결함5.
60. ★★**(2026-08-22, 전환 비용 재분석) 두 분리 최빈값 또는 두
    분리 앨리어스 군집의 풀링 중앙값은 물리량이 아니다.** 혼합비가
    데이터 구조상 고정(모든 admission 뒤엔 반드시 merge, 48.5:51.5)
    이면 "20/20 아티팩트 부호 동일"은 반증 통과가 아니라 항등적
    사실이다 — 같은 함정이 모델≡노드≡job 앨리어스로 갈린 두
    residency 군집(granite/zamba2)의 풀링값(**인용 금지**)에도 적용된다.
    결론은 항상 모드/군집별로 다시 쓰고, 혼합비가 고정(=항등)인지
    자유(=경험적)인지 먼저 판별하라. 상세 `reports/CONSENSUS.md`
    §3 항목80, `workspace/engine-port/results/kernel_mech/
    audit_promotion_2026-08-22/VERDICT.md` 결함1·2.

61. ★★★**(2026-08-23, P0-A rev1–rev6 규칙층 5회 감사 + S-6
    rev1–rev4 규칙층 5회 감사, claims-auditor, GPU 0) 규칙을
    산문으로 고정하면 구멍이 난다 — 결정 규칙은 코드로 고정하고
    세계를 전수 열거하라.** P0-A는 평가 순서 결함이 **2회 연속**
    (rev4→rev5의 F1, rev5의 다음 판 F2가 같은 계열)으로 재발했고,
    S-6는 "OFF 다리를 전수 포함한다"는 산문 주장이 **5회 연속**
    실패했다(S6_POWER·rev1–rev4). 유일하게 통한 대응은 **기계화**
    였다 — P0-A의 `p0a_rule_totality.py`(516,096 세계·54 검사)와
    S-6의 `s6_offleg_enumerate.py`(자기검사 19/19)가 그것이고,
    4회차 감사가 후자의 `--emit` 산출물을 감사 자신의 독립
    재실행과 **바이트 동일**로 검증해 실제로 작동함을 확인했다.
    실무 규칙: 사전등록이 "모든 경우를 다뤘다"고 산문으로 주장하는
    지점마다 그 주장을 검증하는 열거기 스크립트를 짝지어라 —
    손 열거는 회차를 반복해도 수렴하지 않는다. 상세 `workspace/
    engine-port/results/bcg_probe/audit_p0a_prereg_{rev4,rev5}_
    2026-08-23/VERDICT.md`, `workspace/engine-port/results/
    slo_sched/audit_s6_prereg_{rev3,rev4}_2026-08-23/VERDICT.md`,
    CONSENSUS §3 항목81과 대응.

62. ★★★**(2026-08-23, 같은 두 감사에서 동시 발견, claims-auditor,
    GPU 0) 변경 이력표가 하지 않은 수리를 적을 수 있다 — 각 행에
    검증 방법을 병기하라.** 같은 세션에 **두 트랙에서 동시에**
    나온 실패다 — P0-A rev4의 이력표는 (i) "2회차가 삭제 지정한
    변이 행을 제거했다"고 적었으나 그 행은 §7에 그대로 있었고,
    (ii) "M1·M2·M5를 등재했다"고 적었으나 이름만 같고 내용이 전혀
    다른 3건이었다(5회차 감사가 "E6" 차단으로 지목). S-6 rev3의
    이력표도 같은 형태로 지적됐다(E1). ★**S-6 rev4는 더 나쁜
    변종을 냈다**: 이번엔 거짓 "수리" 라벨이 아니라 **차단
    자체를 이력표·§9 합격기준에서 소거**했다(F1 — E5가 두 곳
    모두에서 사라짐, 본문 검색은 무관한 `SMOKE5`만 나옴). 실무
    규칙: 개정판 이력표의 각 행에 "이 수리를 어떻게 검증했는가"
    (코드 검사명 또는 `grep` 명령)를 병기하고, 직전 회차가 지목한
    차단 개수와 이번 회차 이력표의 행 개수가 **정확히 일치하는지**
    별도로 대조한다(누락 탐지). 상세 `workspace/engine-port/
    results/bcg_probe/audit_p0a_prereg_rev5_2026-08-23/VERDICT.md`
    "먼저 E6을 정정한다" 절, `workspace/engine-port/results/
    slo_sched/audit_s6_prereg_rev4_2026-08-23/VERDICT.md` F1,
    CONSENSUS §3 항목82와 대응.

63. ★★**(2026-08-23, P0-A rev1→rev2, claims-auditor, GPU 0)
    안전망을 제거하는 개정은 양쪽 끝을 모두 복원해야 한다.**
    P0-A rev1이 `TOL`(카디널리티 허용오차 밴드)을 집합 술어로
    바꾸면서 **하한** 안전망만 복원했고(1회차 감사 B1–B7이 지적),
    rev2가 **상한**도 제거돼 있었음을 다시 지적했다(C1) — "이번
    수리가 다음 회차 최고 차단을 만든다"는 패턴이 P0-A만 5회
    연속이다. 실무 규칙: 대칭적으로 설계된 안전장치(상한/하한,
    양방향 게이트 등)를 개정할 때는 **한쪽을 고친 커밋 diff에서
    반대쪽도 같이 바뀌었는지 기계적으로 대조**한다(diff 좌우
    대칭 검사) — "이번엔 이 절반만 고친다"는 의도적 스코프 축소와
    "반대쪽을 잊었다"는 결함은 diff만 봐서는 구별되지 않으므로
    문서에 "반대쪽은 의도적으로 손대지 않았다"를 명시하지 않는 한
    결함으로 취급한다. 상세 `workspace/engine-port/results/
    bcg_probe/audit_p0a_prereg_2026-08-22/VERDICT.md`(1회차,
    B1–B7), `audit_p0a_prereg_rev2_2026-08-22/VERDICT.md`(2회차,
    C1), CONSENSUS §3 항목83과 대응.

64. ★★★**(2026-08-23, P0-A rev4→rev5, claims-auditor, GPU 0)
    결정량으로 승격한 입력에는 게이트·앵커·열거를 함께 붙여라.**
    rev4가 `LOST` 판별 문턱이었던 합집합 카디널리티 규칙을
    `E_attrib := E ∩ S(eager_green_prefill)`(귀속)으로 교체해 "1–2
    라벨 실 탈출이 영구 비가시화"라는 죽은 줄 알았던 결함(E2)을
    잡았다 — 그러나 rev5 감사(F1)가 확인한 것은 그 결함이
    **사라진 게 아니라 새로 승격된 입력 `S(eager_green_prefill)`
    자체로 이사**했다는 것이다: 이 입력엔 **게이트 0·앵커 0·세계
    공간 값 1개**뿐이라 1라벨 잡음이 여전히 최고가 판정을
    만들었다. 실무 규칙: 판별 로직을 "카디널리티/문턱"에서
    "집합 연산"으로 재정식화할 때, 그 연산에 새로 들어가는
    **모든 입력 집합**이 그 자체로 게이트(양성 하한)·앵커(고정
    기준값)·열거(세계 공간에서 취할 수 있는 값의 개수 ≥2)를
    갖는지 확인한다 — 결정량을 옮기는 것은 결함을 고치는 것이
    아니라 옮기는 것일 수 있다. 상세 `workspace/engine-port/
    results/bcg_probe/audit_p0a_prereg_rev5_2026-08-23/VERDICT.md`
    F1, CONSENSUS §3 항목84와 대응.

65. ★★**(2026-08-23, P0-A rev5, claims-auditor 자체 도구 실행,
    GPU 0) 검사가 load-bearing해 보이려면 세계 공간이 그 게이트를
    발화시킬 수 있어야 한다.** `p0a_rule_totality.py`가 스스로
    잡은 결함 2건 — (i) 포화 축이 서로 **묶여** 있어 `P5`(포화
    가드) 양쪽을 동시에 삭제해도 세계 공간에 **0 변화**만 생겨
    "가드가 있다"는 주장이 실은 무엇도 배제하지 않았고, (ii)
    삭제 변이 테스트만으로는 게이트를 **약화**시키는 변이(문턱을
    느슨하게 바꾸는 것)를 못 잡는다는 것을 감사가 `±8` 값
    변이로 실측해 "ALL PASS"를 냈다. 실무 규칙: 자기검사(self-check)
    스위트를 짤 때 (a) 각 게이트가 실제로 세계 공간의 어떤
    부분집합을 배제하는지 계산하고 그 크기가 0이 아닌지 확인하고
    (b) 삭제 변이뿐 아니라 **문턱을 느슨/엄격하게 바꾸는 모수화
    변이**도 포함한다(T4′). 상세 `workspace/engine-port/results/
    bcg_probe/audit_p0a_prereg_rev5_2026-08-23/VERDICT.md` "버틴
    것" 절, CONSENSUS §3 항목85와 대응.

66. ★★**(2026-08-23, S-6 rev1–rev4, claims-auditor, GPU 0) 포크는
    원본의 하드와이어를 상속한다 — 매 회차 새 거부 지점이 나오면
    아키텍처 신호다.** S-6는 7-arm G16 캠페인용으로 지어진 하네스
    (`g16_grid.sbatch`/`g16_analyze.py`)를 1-arm telemetry-OFF
    대조로 포크하려 했고, 4회차에 걸쳐 **독립적인 거부 경로
    7건**을 발견했다(`SMOKE` 19개 중 17개가 arm 리터럴로 고정·
    `N_EXPECT_ARMS=7`+`exit 1`·`g16_analyze.py:971`의 `telem_rc==0`
    하드코딩·`TELEM_RC` 정수 비교+소비처 9곳·`:196 ${1:?}`·
    `:234 exit 3`·파일명 리터럴 자기해싱) — 매 회차 문면을 고쳐도
    다음 회차가 새 거부 지점을 찾았다. 실무 규칙: 기존 캠페인
    하네스를 다른 arm 수·다른 축 설계로 포크할 계획이 있으면,
    **착수 전에** 하네스 코드에서 arm 개수·이름을 하드코딩한
    지점을 전수 grep하고 그 개수가 회차마다 새로 느는지를
    "설계 반려"가 아니라 "포크 대 재작성" 결정의 근거로 삼는다
    — 3회차 이상 반복되면 재작성이 더 싸다는 신호다(대가 = 기존
    하네스 계열과의 L1 비교가능성 상실). 상세 `workspace/
    engine-port/results/slo_sched/audit_s6_prereg_{rev2,rev3,rev4}_
    2026-08-23/VERDICT.md`, CONSENSUS §3 항목86과 대응.

67. ★★★**(2026-08-23, NSL-1 rev3, claims-auditor, GPU 0) 표에
    단위를 셀에 적어라 — 행 라벨을 셀의 단위로 읽지 마라(교훈
    #61·#63의 세 번째 재발이자 가장 비싼 형태, 판정 부호가
    뒤집혔다).** NSL-1 rev3 §1.2가 rev2 판정서의 표를 옮겨 적으며
    행 라벨 `stat="mean"/"p95"`(어떤 ITL 통계량으로 채점하는가)를
    셀의 **단위**로 오독하고 "cap은 TTFT 다리에만 작용한다"는
    중심 논증을 세웠다 — 실제 셀은 항상 goodput attainment
    %(pp)였고, 같은 표의 §1.2에선 `ms`, §3에선 `pp`로 쓰이는
    내부 모순이 이미 있었는데도 발견되지 않았다. 실측하면 결론이
    **정반대**다(HI에서 ITL 다리 통과율 12.08% < TTFT 다리
    통과율 30.46%). 실무 규칙: 다른 문서의 표를 옮길 땐 그 표를
    만든 원 코드를 대조해 행 라벨이 통계량 선택인지 출력 단위인지
    확인하고, 같은 표의 다른 절이 같은 수치를 다른 단위로 쓰고
    있지 않은지 내부 정합성부터 점검한다. 상세 `workspace/
    engine-port/results/nsl_lever/audit_nsl1_rules_rev3_2026-08-23/
    VERDICT.md` §2, CONSENSUS §3 항목87과 대응.

68. ★★**(2026-08-23, NSL-1 rev3, claims-auditor, GPU 0) "이
    손잡이가 이 운영점에서 구속하는가"를 Little 법칙 인구로 답하지
    마라 — 그 양은 손잡이가 제한하는 양이 아니고 같은 데이터에서
    손잡이 값을 초과할 수 있다. 구속성은 막힌 사건으로 재라.**
    NSL-1의 유일한 생존 전제("HI 동시성 44.19/48이므로 cap이
    문다")는 Little 법칙으로 복원한 in-system 인구(대기+실행)였다
    — `--max-running-requests 48`은 running batch만 제한하는데,
    저장소 로그에서 rate 10·12의 in-system 동시성이 63.96·83.06
    으로 cap 48을 크게 초과함을 확인했다(다른 양이라 구속의
    증거가 아니다). 실무 규칙: 어떤 손잡이가 실제로 발동해
    요청을 막았는지 주장하려면 간접 복원량이 아니라 그 손잡이
    자신이 발화하는 이벤트(예: `admission이 batch_is_full로 막힌
    스텝 시간비`)를 직접 세라. 상세 `workspace/engine-port/
    results/nsl_lever/audit_nsl1_rules_rev3_2026-08-23/VERDICT.md`
    §3(H2), CONSENSUS §3 항목88과 대응.

69. ★★**(2026-08-23, NSL-1 rev3, claims-auditor, GPU 0) metric
    cliff는 TTFT 다리에만 있는 것이 아니다 — 요청-내부 ITL p95도
    자기 절벽을 가질 수 있다. CLAUDE.md 게이트 #6·PS 내부 게이트
    #12를 ITL 다리에도 적용하라.** 요청-내부 token-ITL p95(최근접
    순위) 분포가 monolithic prefill(`--chunked-prefill-size -1`)
    때문에 58–60ms에 이봉 모드를 갖는다 — 정본 60ms 임계가 그
    모드 바로 위(d24: 55ms 10.2%→65ms 99.3%), chat SLO 50ms
    임계가 그 아래 바닥(전 arm 10–15%)에 앉는다. ★PLAUSIBLE —
    서빙 직접 개입으로 분리하지 않았으므로 CONFIRMED 아님. 실무
    규칙: percentile 다리를 절벽 없이 측정했다고 주장하려면 그
    다리에도 임계 사다리(예: 40–100ms)를 돌려 부호가 유지되는지
    확인한다 — TTFT 다리만 사다리를 돌리고 ITL 다리는 단일
    임계로 판정하면 비대칭 검증이다. 상세 `workspace/engine-port/
    results/nsl_lever/audit_nsl1_rules_rev3_2026-08-23/VERDICT.md`
    §6(d), CONSENSUS §3 항목89와 대응.

70. ★★★**(2026-08-23, AUDIT_DEBT §6, 메인 세션 자체 반복 실패 +
    doc-steward 승격 판단, GPU 0) 변경/설계 검증 칸에는 "무엇을
    했다"만 적어라 — "무엇일 것이다"는 검증이 아니다(게이트 #62의
    강화형).** 같은 오류가 이번 세션 메인 작업 안에서 3회
    재발했다: (1) kernel_mech rev7 처리표(C1 행)의 "(발화 가능)"
    — 실제로는 발화 불가 (2) `PREREG_P0A` rev8 헤더의 "신규 지적
    H1–H7은 rev8이 전부 닫았다" — H5는 안 닫혔다 (3) kernel_mech
    rev8 §2.1의 "(발화 가능)"+"귀속 사상이 어긋나면 카운트가
    반드시 튄다" — 균일 오귀속 반례에서 카운트 불변·`gap_frac`
    +75% 오차. ★★3번이 가장 나쁘다 — 1번을 수리하는 항목 안에서
    같은 형태로 재발했다. 세 건 모두 감사가 30분 안에 반례를
    만들었다(확인 비용 < 작성 비용). ★기존 게이트(#62, "이력표
    각 행에 검증 방법을 병기하라")와의 관계: 세 건 다 검증 칸
    자체는 있었다(#62 요구는 형식상 충족) — 문제는 그 칸에 적힌
    것이 검증이 아니라 **예측**이었다는 것. 요구를 "검증 칸의
    존재"에서 "그 칸 내용이 이미 실행된 것"으로 강화해 별도
    항목으로 등재한다. 실무 규칙: 검사를 신설하며 "발화 가능"
    이라 적으려면 그 검사를 발화시킨 입력을 같은 커밋에 넣어라 —
    발화 witness가 없으면 "발화 가능"이 아니라 "발화 가능성
    미확인"이라 적고, `UNFIREABLE`일 때의 등록 결과를 미리 적어라.
    따름: "전부 닫았다"는 항목 수를 세어 쓰지 말고 닫힌 항목만
    이름으로 열거하라(개수는 검증되지 않는다). 이 세션에서 실제로
    통한 대응 2건: P0-A H5(AST 형태 검사 → 런타임 값 결속으로
    교체)·`rev7_power.py` C8(변이 테스트를 같은 커밋에 포함).
    ★**doc-steward 부수 판단**: NSL-1 rev3의 H1(단위 오독, 위
    #67)도 "검증 없이 단정했다"는 점에서 이 메타-패턴의 네 번째
    사례로 볼 수 있다 — 그러나 그 구체 기전(라벨/단위 오해)은
    #67(교훈 #61·#63 계열)이 더 정확히 포착하므로, H1은 두 계열
    모두에 교차 등재하고 어느 한쪽으로 강제 흡수하지 않는다.
    상세 `reports/AUDIT_DEBT_2026-08-23.md` §6, CONSENSUS §3
    항목90과 대응.

    ★★**追記(2026-09-01, cp_baseline CP-0 rev3 규칙층 감사 4회차,
    claims-auditor, GPU 0) — 거울상 재발**: 이 게이트가 요구하는
    "검증 칸은 이미 실행된 것만 적어라"의 반대쪽 사례가 CP
    트랙에서 나왔다 — 판본 이력표(변경 이력 서술)가 **본문과
    직접 모순**됐다(감사 死因 L-l, 어느 절이 맞는지 서술만으로는
    판별 불가). 실무 규칙: 이력표와 본문이 같은 사실을 다르게
    서술하면 그 문서는 **감사 접수 대상이 아니다**(수리 여부를
    묻기 전에 반려) — 이력표 자체를 검증 가능한 단일 출처(코드가
    갱신하는 카운터)로 대체하는 것이 근본 해법이며, 아래 #80
    追記의 `check_version_sweep.py`가 그 시도다. 상세 `workspace/
    engine-port/results/cp_baseline/audit_cp0_rev3_4th_2026-08-28/
    VERDICT.md` 死因 L-l·"신규 방법론 교훈 후보" 6번, CONSENSUS
    §3 항목90 追記와 대응.

    ★★★**복원 항목 #71–80(2026-09-01, doc-steward)**: 아래 열 항목은
    이 절 제목(위 "## 방법론 게이트 (...)" 변경이력 산문)에는 신설
    당시부터 있었으나 **이 번호 매긴 열거에는 없었다** — engine-porter
    발견·claims-auditor 4회차 반박(정정: engine-porter가 옳다, :5845-5849는
    열거가 아니라 변경이력 산문)에 따른 구조 복원. 내용은 산문 원문을
    **그대로 전재**(재해석·재판정 없음) — 상세·전문은 각 항목이 가리키는
    `reports/CONSENSUS.md` §3 해당 번호에서 확인한다. 이 복원은 새 판정이
    아니며 등급·정책 순위·인용정지를 바꾸지 않는다.

71. ★**(2026-08-25, kernel_mech Stage 0‴ A0/A1 + NSL ①/E-A, claims-auditor
    적대 감사 7회, doc-steward 정규화) 어댑터 축이 다른 축에서 파생되면
    그 자체가 차단이다.** `l2_post`가 L4에서 파생돼 L4가 비면(등록된
    결정적 음성) 답을 지운 뒤 그 답을 말하는 것 자체를 금지했다.
    CONSENSUS §3 항목91과 대응.

72. ★**(2026-08-25, 상동) 트랙 자신의 사전등록이 지목한 최소비용
    선행 프로브를 사지 않은 채 더 비싼 후속을 반복하지 마라.**
    kernel_mech가 "사전등록 전 필수"라 적은 15분짜리 프로브를 9일간
    안 산 채 rev1–rev8을 썼다. CONSENSUS §3 항목92와 대응.

73. ★**(2026-08-25, 상동) 검사 중복은 양방향으로 위험하다.** 전체
    iff-분할 위의 라벨별 검사 N개는 1개 분량의 정보만 나르고,
    접으면 메타검사가 공허해질 수 있다(2026-08-26 NSL D3(c)에서
    두 번째 사례 追記 — 거울 대칭 검사 삭제 근거였던 "단독구속 0"이
    격자에 0.20/0.80이 없어 생긴 인공물이었다). CONSENSUS §3 항목93과
    대응.

74. ★**(2026-08-25, 상동) 결론이 우연히 살아있는 것과 근거가 타당한
    것은 다르다.** 여러 문서가 같은 미확인 전제를 상속하면 결론이
    옳아도 근거를 다시 확인하기 전엔 확인됐다고 쓰지 마라 —
    kernel_mech 6개 문서가 틀린 트리 grep을 확인 없이 상속했다
    (게이트 #32의 grep 층 변종). CONSENSUS §3 항목94와 대응.

75. ★**(2026-08-25, 상동) 출처 허위는 규칙 정본 파일 안에서 가장
    위험하다.** 산문이 아니라 코드 상수 안의 허위는 다음 판본이
    검증 없이 그대로 승계할 수 있다 — `K1_HIGH_AT = 2.00  # rev8
    registered this`인데 rev8에 `K1` 0건. CONSENSUS §3 항목95와 대응.

76. ★★**(2026-08-26, kernel_mech A1 sticky 재설계 규칙층 감사 3회
    + NSL ③②E-B 묶음 첫 규칙층 감사, claims-auditor 적대 감사 4회
    + doc-steward 승격 판단, GPU 0.014 GPU-hr[스모크 job 893663뿐])
    인용/드리프트 검출 도구 자신이 거짓 인증을 낼 수 있다.**
    `check_line_citations.py --snapshot`이 편집된 파일의 기준선을
    조용히 갱신해 `a1_q3k1_rule.py:83`이 `N_MIN_DECODE_STEPS`를
    가리킨다는 인증을 한 커밋 만에 발화시켰다(실제 `:105`) —
    재베이스 거부 + 고아 키 탐지로 부분 수리. ★이 게이트가 바로
    이번 세션(2026-09-01)의 CP-0 P1 line-citation 정정에서 지킨
    규율의 근거다. CONSENSUS §3 항목96과 대응.

77. ★★**(2026-08-26, 상동) 추정량이 코드 경로의 평가 순서에
    의존하면 그것은 귀속이 아니라 순서통계량이다.** NSL D3(a)(b) —
    `cap` 검사가 매 반복 최상단에서 평가돼 KV 경로는 cap 불발 시만
    도달하고 `OTHER` 출구는 사이트 목록에도 안 잡힌다. 계수 설계
    전 제어흐름의 상호배제·평가 우선순위를 먼저 등록하라. CONSENSUS
    §3 항목97과 대응.

78. ★★**(2026-08-26, 상동) 계측 완전성은 사이트 목록이 아니라
    기전 목록에서 판정하라.** NSL D2 — `batch_is_full`을 세우는
    자리와 admission을 막는 기전은 다른 집합이라, 유일한 TOOLLIMIT
    술어가 관측 불가인지 상시발화인지를 하네스가 조용히 정한다.
    CONSENSUS §3 항목98과 대응.

79. ★**(2026-08-26, 상동) de-confound 처방 자체가 추정량의 자유
    모수일 수 있다(confound #10의 메타 형태).** NSL D3(c)/B2 —
    `--max-mamba-cache-size` 고정처럼 손잡이를 순화하는 처방의
    파라미터 선택이 측정 대상(KV 예산)을 방향성 있게 이동시킬
    수 있다. CONSENSUS §3 항목99와 대응.

80. ★★★★**(2026-08-26, 상동) "수리는 국소, 주장은 전역"
    (repair-local truth vs. document-global claims).** 각 수리는
    지목된 좌표에서 실재하나 그 파급을 재도출하지 않으면 다음
    회차 결함 대부분이 직전 수리의 그림자가 된다(A1 3회차 감사
    명명; 그 세션 안에서 자기수정 8건 중 다수가 이 형태였고,
    NSL §4.1의 교훈-이식이 세션 내에서 재현). 부수형: "세계모형이
    실험설계 성장을 못 따라간다." ★이 게이트가 바로 이번 세션의
    "다른 트랙 파급(NSL 3문서 인용 정정)" 정리를 요구한 근거다.
    CONSENSUS §3 항목100과 대응.

    ★★★★**追記(2026-09-01, cp_baseline 규칙층 감사 4연속, claims-
    auditor, GPU 0) — 정량 재확인 + 새로운 하위 형태.** CP 트랙이
    이 패턴의 **네 번째 독립 확인**을 냈다: 4회차 감사(CP-0 rev3)가
    死因 8건 중 그림자 비율을 재계산해 **"24건 중 11건(46%)이
    순수 전파 실패"**(수리는 실행됐고 그것을 인용하는 자리를
    안 쓸었을 뿐, 새 사고 불필요)라고 정량화했고, 3회차 대비
    비율이 사실상 불변(그림자 死因 비중 50%대)임을 확인했다.
    ★**새 하위 형태**: 전파 실패 중 하나(§8의 "시험 3개" 갱신
    누락)는 **유일한 중단 규칙 자체를 무력화**했다 — 즉 전파
    실패는 산문 서식 문제가 아니라 **런킬러**(감사를 계속 통과
    시키지 못하게 막는 결함)가 될 수 있다. 4회차 최종 판정:
    *"rev1은 규칙이 문제였고 rev2도 규칙이 문제였고 rev3는 절반
    이상이 규율이다 — 이 이동 자체는 전진이지만, 규율 실패는
    감사로 못 고친다."* ★**대응 도구**: 이 세션이 `check_version_
    sweep.py`(S1–S8, `workspace/engine-port/results/cp_baseline/
    check_version_sweep.py`)를 신설해 회차 카운터·시험 개수·
    인용 좌표를 코드로 검사하는 시도를 시작했다(범위: cp_baseline
    산출물, 저장소 전역 아님 — 다른 트랙으로 이전하려면 그 트랙의
    규율도구 4종과 통합 검토 필요). 상세 `workspace/engine-port/
    results/cp_baseline/audit_cp0_rev3_4th_2026-08-28/VERDICT.md`
    "수리는 국소, 주장은 전역 + 4연속 NO-GO 판정" 절·"신규 방법론
    교훈 후보" 7번, CONSENSUS §3 항목100 追記와 대응.

81. ★★**(2026-08-28, TC1 rev2·M4R rev2 규칙층 감사 + 설계층
    도달가능성 메타검사, claims-auditor + doc-steward, GPU 0)
    도달가능성 검사가 격자 안에서만 돌면, 설계가 답을 미리
    정해 놓아도 통과한다.** 규칙 파일의 `T2_reachable`류는
    "라벨이 격자 어딘가에 나타나는가"를 묻는 **격자 내부
    성질**이라 설계층 도달불가를 원리적으로 못 잡는다(TC1 rev1
    B13·M4R rev2 F3′가 각각 이 구멍을 지적). `design_reachability.py`
    는 반대를 묻는다 — **등록된 설계와 이미 측정된 데이터가 실제로
    낼 수 있는 실질 라벨은 무엇인가**(모든 제약에 출처 문자열
    강제). 두 트랙에 돌린 결과: M4R rev2 = 실질 라벨 3종
    (`RESIDUAL_{PRESENT,ABSENT,INCONCLUSIVE}`) 중 **`RESIDUAL_
    INCONCLUSIVE` 하나만** 도달 가능(`SINGLE_LABEL_FORCED`) —
    7/7 셀의 블록 클러스터 t-CI가 1을 포함하고 `sm_match`·`iters`
    ·`cadence` 세 가드가 발화할 세계가 없어 판정이 데이터 도착
    전에 이미 고정돼 있었다. TC1 rev2 시나리오 A(argmax=d44,
    정본이 §1-7·§1-33 두 번 지지하는 좌표) = **낼 수 있는 실질
    라벨 0개**(`NOTHING_PURCHASABLE`) — 캠페인 50 job 전체의
    정보량이 Stage 1 argmax 추첨 하나(d34↔d44 격차 1.5% = 등록
    문턱 δ의 절반)에 걸려 있었다. **⇒ 결정량 자체가 항등식이
    아니어도(TC1의 `visits_argmax`, M4R의 `sm_match`는 둘 다
    엔진 코드가 강제하는 진짜 항등식이라 이 사례들은 게이트
    #40의 직접 재발이기도 하다) 도달가능성 검사가 격자 내부에만
    머물면 결정 자체가 항등식이 된다.** 실무 규칙: 사전등록
    산출물로 `reachability_spec.json`을 요구하고, 실질 라벨이
    0개(`NOTHING_PURCHASABLE`) 또는 1개(`SINGLE_LABEL_FORCED`)면
    **제출 금지**로 못 박아라(이번 세션에 `presubmit.py` 게이트로
    집행). 상세 `workspace/engine-port/results/
    REACHABILITY_FINDING_2026-08-28.md`, `workspace/engine-port/
    results/{tc1_model_attrib/reach_verdict_A.json,
    m4r_confinement/reachability_verdict.json}`, CONSENSUS §3
    항목101과 대응.

82. ★**(2026-08-28, TC1 rev1 규칙층 감사 §3, claims-auditor,
    GPU 0) 번호 참조 체계가 넷인데 사전등록이 둘로만 세면,
    "정정문" 자체도 오인용을 재생산한다.** 이 저장소는 번호
    체계가 넷이다 — `CLAUDE.md` "방법론 게이트"(#1–8) ·
    `PROJECT_STATUS.md` "방법론 게이트"(이 목록, #1–80대) ·
    `CONSENSUS.md` §3 교훈 항목(#1–100대) · 메모리 topic
    `deconfound-measurement-lessons` 항목(#1–80대). TC1 rev1
    사전등록 §0은 이를 둘로만 셌고, `results/kernel_mech/
    stage0ppp/stage0ppp_a0_rule.py:2`의 `(gate #66)`이 rules-as-code
    규율을 잘못 가리킨다는 지적(사실관계는 정확)을 하면서 그
    자신의 정정문에서 **"CONSENSUS §3 교훈 항목66"**이라고 다시
    썼다 — 정답은 **항목81**(대응 게이트는 `PROJECT_STATUS.md`
    게이트 #61)이다. 즉 오인용을 지적하는 문장이 **같은 문서·
    같은 단락 안에서** 새 오인용을 만들었다. 같은 저장소에서
    같은 파일이 동일 게이트 번호(`#4`)를 체계 A와 체계 B 양쪽
    의미로 섞어 쓴 사례도 확인됐다(`p1_gates/gate2/
    PREREG_GATE2S_2026-08-09.md`). 실무 규칙: 번호를 인용할 때는
    **항상 체계 이름을 병기**하라(`CLAUDE.md 게이트 #N` /
    `PS게이트 #N` / `CONSENSUS §3 항목N` / `메모리항목 N`) —
    "정정했다"는 서술 자체를 검증 없이 승계하지 마라(게이트
    #67/#70의 재발이기도 하다). 상세 `workspace/engine-port/
    results/tc1_model_attrib/audit_tc1_rules_2026-08-27/VERDICT.md`
    §3, CONSENSUS §3 항목102와 대응.

    ★★★**追記(2026-09-01, cp_baseline CP-1 rev2 규칙층 감사 2회차,
    claims-auditor, GPU 0)**: 이 게이트(#82)와 별개로, 번호 체계
    오인용이 **넷째 확인 사례**를 냈다 — CP 트랙 문서들이
    `PROJECT_STATUS.md` 게이트 #10(여집합 클래스 음성대조)을
    "동시 변경 금지"로 오독하려던 시도는 없었으나, `cp_rule.py`의
    임계 주석이 CLAUDE.md 게이트 #3("3% 미만 차이는 headline
    아님")을 "PROJECT_STATUS.md의 게이트"로 잘못 귀속했다(K6,
    `audit_cp_rules_2nd_2026-08-28/VERDICT.md`). 상세 CONSENSUS
    §3 항목102 追記.

83. (번호 예약 — TC1/M4R 트랙의 "백엔드 강제 교락" 게이트, 본문은
    이 문서 최상단 배너(2026-08-28 2차)·`CONSENSUS §3 항목103`에
    있음. 이 번호 매긴 목록에 문단 형태로 아직 옮겨지지 않은
    기존 결손 — TC1/M4R 소관, 이 세션[cp_baseline, 2026-09-01]
    에서 손대지 않음. 다음에 이 문단을 옮길 때 번호 충돌
    없도록 84 이상을 쓰지 말 것.)

84. ★★★**(2026-09-01, cp_baseline CP-0 rev3 규칙층 감사 4회차,
    claims-auditor, GPU 0) 자기검사의 기준선을 검사 대상 자신에서
    계산하면 외부 의미 반전을 원리적으로 못 잡는다 — 등록 오라클이
    필요하다.** `cp0_selftest.py`의 변이 검사가 `base`를 변이체
    자신에서 계산해 *"서로 다른 4값이 4셀에 놓이면 항등 아닌
    순열은 정의상 라벨을 바꾼다"*는 **항등식**만 확인했다(등록
    변이 X1–X3 전부 "DETECTED"로 생존했으나 외부 의미 반전은
    0건 검출 — 통과 자체가 항등식이 참임을 보인 것뿐). 4회차
    감사가 이를 지적한 뒤 메인 세션이 **`cp0_expected_labels.json`**
    (사람이 읽는 world→label 정답표, 검사 대상 코드와 독립)을
    별도 파일로 신설해 대응했다. 실무 규칙: 변이 검사를 설계할
    때 "기대값"을 검사 대상 함수를 호출해 만들지 말고, 그 함수가
    존재하기 전에 정할 수 있는 정답을 별도 아티팩트로 등록하고
    그것과 대조하라 — 게이트 #45(감사용 검사 자신의 항등식)·#9
    계열의 재발이지만 "기준선을 어디서 계산했는가"라는 새 축을
    지목한다는 점에서 별개로 등재한다. 상세 `workspace/engine-port/
    results/cp_baseline/audit_cp0_rev3_4th_2026-08-28/VERDICT.md`
    "신규 방법론 교훈 후보" 1번, CONSENSUS §3 항목104와 대응.

85. ★★★**(2026-09-01, 상동) 문턱을 이식 가능한 비율로 바꿔도 그
    문턱을 재는 격자가 절대 단위면 이식 불가성은 격자로 옮겨간
    것뿐이다.** CP-0 rev1이 `SATURATED_AT_FLOOR` 임계를 절대
    req/s에서 **비율**(도달률)로 바꿔 이식 가능성을 확보했다고
    주장했으나, 그 비율을 측정하는 격자 `{2,4,8} req/s`는 여전히
    **절대 단위**이고 워크로드마다 다르다(capscan 2096 tok/req vs
    ShareGPT 578 tok/req = 3.6배 차이 ⇒ 같은 절대 rate 격자가
    두 워크로드에서 서로 다른 부하 수준을 가리킨다). 게다가 이
    임계를 **유도한 데이터 자체를 검증 벡터로 재사용**해 반증
    불가능하게 만든 사례가 병존했다(0.72·0.795 두 후보 임계
    모두 자기 유도 데이터 위에서 생존). 실무 규칙: de-confound
    처방(절대→비율)이 결함을 **닫았는지 이동시켰는지**를 그
    처방이 딛고 선 다음 층(여기서는 격자)까지 추적해 판정하라 —
    그리고 임계를 유도하는 데 쓴 데이터는 그 임계의 검증
    데이터로 다시 쓰지 마라. 상세 `workspace/engine-port/
    results/cp_baseline/audit_cp0_rev3_4th_2026-08-28/VERDICT.md`
    "신규 방법론 교훈 후보" 2·4번, CONSENSUS §3 항목105와 대응.

86. ★★★**(2026-09-01, 상동) 연속 축 위에 문턱을 등록하기 전에 그
    통계량의 무처치(no-load) 분산을 먼저 재고, 유도한 판정 밴드가
    그 분산보다 넓은지 산술로 보여라 — 좁으면 그 축은 식별력이
    없다.** CP-0의 rate 축 판정에서 무부하 attainment의 seed 간
    SD가 **0.108**(일부 관측값은 1.0을 넘음 — 분자 duration과
    분모 nominal rate의 시간축 불일치)인데, knee 위치를 유도한
    밴드는 폭 **0.09**로 그 SD보다 **좁았다**(⇒ 밴드 안의 어떤
    지점을 골라도 잡음과 통계적으로 구분 안 됨). ★같은 논리가
    이 세션의 G1 프로브에서 **독립적으로 재확인**됐다 — `band_vs_sd`
    (plateau 판정)를 채우려 하니 fused_default(6→8 증가 0.610 ≫
    achieved SD 0.19)와 cp512(0.336 < achieved SD 0.435)가 **반대
    부호**로 갈렸고, 그래서 그 축을 미등록 상태로 남겼다(§ "다음
    실험 gate" #11 cp_baseline 행 참조). 실무 규칙: 연속 축의
    판정 문턱을 사전등록하려면 §0에 "무처치 SD"를 별도 산출물로
    요구하고, 유도 밴드/증가폭이 그 SD의 배수인지(예: >2×) 명시
    임계로 등록해야 판정이 가능하다. 상세 `workspace/engine-port/
    results/cp_baseline/{audit_cp0_rev3_4th_2026-08-28/VERDICT.md
    "신규 방법론 교훈 후보" 3번, RESULT_G1_900067_2026-09-01.md
    §B}`, CONSENSUS §3 항목106과 대응.

87. ★★**(2026-09-01, 상동) 상수를 검사 표면 밖 모듈로 옮기는 것은
    수리가 아니라 은폐일 수 있다.** `FLOAT_TOL`·`PROBE_RATES` 같은
    자유 상수가 규칙 모듈(`cp0_arm_rule.py` 등)에는 상수 분류
    검사를 통과하도록 등록됐으나, 같은 상수가 술어 모듈
    (`cp0_predicates.py`)로 옮겨진 뒤에는 그 검사가 **거기까지
    따라가지 않아** 사실상 검사 표면 밖으로 나갔다(감사 死因
    L-d). 게이트 #75("출처 허위는 규칙 정본 파일 안에서 가장
    위험하다")의 거울상 — 여기서는 허위 주석이 아니라 **검사
    커버리지 자체가 모듈 경계에서 끊긴 것**이 문제였다. 실무
    규칙: 상수를 새 모듈로 옮길 때는 그 상수를 검사하는 도구도
    같은 커밋에서 그 모듈을 훑도록 갱신하고, 안 그러면 "옮김"을
    "닫음"으로 서술하지 마라. 상세 `workspace/engine-port/
    results/cp_baseline/audit_cp0_rev3_4th_2026-08-28/VERDICT.md`
    死因 목록 L-d·"신규 방법론 교훈 후보" 5번, CONSENSUS §3
    항목107과 대응.

88. ★★★★**(2026-09-03, cp_baseline CP-2 rev2 규칙층 감사 6회차,
    claims-auditor, GPU 0) 삭제는 조건 등록이 아니라서 grep에
    안 잡힌다 — 수리를 "조건을 지웠다/범위를 좁혔다"로 서술하면
    감사가 못 잡는 死因을 만든다.** 이 회차의 단일 판정 질문 답이
    바로 이것이다: rev1의 死因 8건(H1–H8) 중 절반가량이 rev2에서
    "조건을 새로 등록"해서가 아니라 **조건을 삭제**해서 사라졌다
    — `H7`(범위로 양화한 중단 규칙)이 옳은 형태로 고쳐지는 과정에서
    rev1이 등록했던 **정본 스코어러·duration 합산·paired
    bootstrap·`N_BOOTS`·동반 공표 의무**가 rev2 본문에 **0회**
    등장한다(직접 grep 확인, 死因 F4) — "범위로 양화했다"는 옳은
    수리 서사가 실제로는 **더 작은 세계**를 묶었다. `H8`(estimand)은
    논증되지 않고 그냥 **삭제**됐다(rate 3개 × 사다리 36점을 단일
    `(ci_lo, ci_hi)`로 접는 경로가 어디에도 없다, 死因 F4). 이런
    삭제형 수리는 인용 검사(`check_line_citations`)·변이 검사
    양쪽에서 **구조적으로 안 잡힌다** — 두 도구 모두 "등록된 것이
    맞는가"만 검사하고 "이전에 있던 것이 없어졌는가"는 검사하지
    않는다(없는 대상을 grep할 수 없다). 실무 규칙: 판본 간 수리를
    감사할 때는 이전 판본이 등록했던 조항의 **전체 목록**을 만들고
    그 목록 각각이 새 판본에 **문자 그대로 존재하는지**를 개수로
    세라(`grep -c`) — "삭제됐다"를 "폐기하기로 했다"는 산문으로
    대체하지 마라. 대응 `CONSENSUS.md` §3 항목108(신설). 상세
    `workspace/engine-port/results/cp_baseline/
    audit_cp2r2_rules_6th_2026-09-03/VERDICT.md` 死因 F4·"단일
    판정 질문 직답" (2).

89. ★★★**(2026-09-01, cp_baseline CP-2 rev1 규칙층 감사 5회차,
    claims-auditor 자기 신고, GPU 0) 등록된 규율 도구 자신이
    read-only라고 이름 붙었어도, 실행 전 그 사실을 검증하지
    않으면 실제로는 타 트랙 상태를 조용히 바꿀 수 있다.** 5회차
    감사가 등록된 재현 절차의 마지막 줄(`presubmit.py --registry
    presubmit_registry.json`)을 그대로 실행한 결과, 그 도구가
    `m4r_confinement/reachability_verdict.json`과
    `tc1_model_attrib/reach_verdict_rev3_A.json` **두 개의 타
    트랙 추적 파일을 재작성**했다(`git checkout --`으로 즉시
    원상복구, 감사 대상 파일 md5는 불변). 같은 날 메인 세션에서는
    이 부작용이 **다른(동시 진행 중인) 세션의 미커밋 작업을
    실제로 소실**시켰고, 이 세션이 그 산출물을 재생성해 복구했다.
    ★게이트 #34(규칙 먼저·하네스 다음 2단 감사)·#26(대형 캠페인
    제출 전 배관 스모크) 계열의 새 하위 형태 — 두 게이트는 캠페인
    자신의 정확성을 겨누지만, 이번 결함은 **감사·제출 도구 자신이
    부작용원**이라는 점에서 별개다. 6회차는 이 결함을 반영해
    `presubmit.py`를 **아예 돌리지 않고** `design_reachability.py`만
    스크래치 경로로 격리해 돌렸다(그 결과 저장소에 파일 0개 생성).
    실무 규칙: "read-only"라고 문서화된 규율 도구도 실행 전
    `git status`로 대상 외 파일 변경 여부를 감시하거나, 결과
    아티팩트를 격리된 스크래치 경로에 쓰도록 강제하기 전까지는
    돌리지 마라 — 특히 여러 트랙이 동시에 진행 중일 때. 대응
    `CONSENSUS.md` §3 항목109(신설). 상세 `workspace/engine-port/
    results/cp_baseline/audit_cp2_rules_5th_2026-09-01/VERDICT.md`
    "감사자 자기 신고"·死因 L16, `handoff-report/
    session_handoff_2026-09-03.md` §4.1·§5.

90. ★★★**(2026-09-07, cp_baseline 층-근거 제안 rev1 규칙층 감사,
    claims-auditor, GPU 0) 금지문 목록 자신이 거짓 진술을 담을
    수 있다 — 산문 금지문은 어떤 자동 도구도 못 잡는다.** 오독을
    막으려 쓴 제안 §7의 금지문 #2(*"바뀌는 것은 어느 바닥을
    비교하느냐뿐"*)가 감사 死因 U3(등록 예측 4가 q=0.99에서
    False→True로 뒤집힘, 게이트 #8 정면 위반)로 직접 반증됐다
    (死因 U4) — 오독을 막으려고 등재한 문장 자신이 오독이었다.
    인용 검사·변이 검사는 코드·상수·라벨을 검사하지 산문 금지문
    내부의 논리적 함의를 검사하지 않는다. 게이트 #75(메모리 topic
    항목80=`CONSENSUS §3` 항목95, "출처 허위는 규칙 정본 파일
    안에서 가장 위험하다")의 **금지문 판본** — 코드 상수가 아니라
    산문 금지문이 같은 위험을 갖는다. 실무 규칙: 금지문을 등재하기
    전에 그 금지문이 참임을 보이는 반례 탐색을 최소 1회 직접
    수행하고, 그 탐색 결과를 같은 절에 병기하라. 상세
    `workspace/engine-port/results/cp_baseline/
    {PROPOSAL_STRATUM_GROUND_2026-09-07.md §7,
    audit_stratum_ground_2026-09-07/VERDICT.md 死因 U3·U4}`,
    `CONSENSUS §3` 항목110과 대응.

91. ★★★**(2026-09-07, 상동 + `FOLLOWUP_STRATUM_2026-09-07.md`
    §가·`WORKLOAD_REVIEW_2026-09-07.md`, doc-steward, GPU 0)
    모집단 선별이 답을 정할 수 있다 — 분위수는 모집단을 명명해야
    뜻을 갖는다.** 死因 U2. AF-1 풀(`ShareGPT_long2048_cap8192`,
    n=1,968)은 코퍼스(92,886행) 상위 **2.12%**라 등록된 네 층
    전부가 코퍼스 p98 위에 있었다(p98.09/p98.94/p99.79/p99.98)
    — **전형 요청을 한 번도 안 봤다.** 1차 출처가 지시하는 것은
    "당신의 P99"(코퍼스 p99 = 캠페인 2,514 tok)이고, 그 층에서
    재계산하면 `anchor = multiple`(rag·batch 통과, chat 초과
    3.2%)로 갈린다 — **`unique`는 오직 AF-1 풀의 p99에서만
    나온다.** 사전등록 §4.5의 사슬 논증 자체는 참이었지만
    (*"unique가 나오면 그것은 구조상 batch_async"*), **그 사슬이
    발화하는지 자체를 사슬이 아니라 풀 선택이 결정**했다. 메모리
    topic 항목86(=`PROJECT_STATUS.md` 게이트 #81=`CONSENSUS §3`
    항목101, "도달가능성 검사가 격자 안에서만 돌면 설계가 답을
    미리 정해도 통과한다")의 **모집단 판본** — 거기서는 실험
    설계 격자가 답을 미리 정했고, 여기서는 데이터 풀 선택이
    답을 미리 정했다. 실무 규칙: 분위수 기반 문턱을 등록할
    때는 그 분위수를 재는 모집단을 별도 산출물로 정의하고, 그
    모집단이 상위 스코프(코퍼스/실 트래픽)를 대표하는지 논증을
    병기하라 — 대표하지 못하면 라벨(`unique`/`multiple`)이
    채점 대상이 아니라 **풀 정의의 함수**다. 상세
    `workspace/engine-port/results/cp_baseline/
    WORKLOAD_REVIEW_2026-09-07.md` §A.4,
    `FOLLOWUP_STRATUM_2026-09-07.md` §가.3, `CONSENSUS §3`
    항목111과 대응.

92. ★★**(2026-09-07, 상동, doc-steward, GPU 0) 외부 앵커를 쓰는
    트랙은 1차 출처 스냅샷을 남겨야 한다 — URL만으로는 재조회가
    매번 다른 답을 낼 수 있다.** `serving_slo_survey.md:62`가
    Spheron 블로그 URL만 남기고 저장본·조회일·인용 스냅샷을
    남기지 않아, 이 세션이 같은 URL을 **다른 질문으로 두 번
    재조회**해야 했다 — 1차(*"표가 길이를 고정하는가"*)는
    "NO", 2차(*"길이·토큰을 언급한 문장을 전부 축자 인용"*)는
    14건 중 결정적 2건(*"P99 prompt length"* 직접 지시)을
    새로 찾아냈다. 재조회가 서베이 §2 인용보다 **많이**
    찾아냈을 뿐 아니라, 같은 재조회 과정에서 서베이 §2 자신의
    **DistServe SLO-scale 귀속 오인용**(*"무경쟁 단일 요청
    실행 지연의 배수"* — 실제는 Table 1 절대 SLO의 선형 배수,
    무경쟁 배수는 Splitwise/Mooncake/LoongServe)도 함께
    드러났다. 실무 규칙: 외부 문헌을 앵커로 쓰는 트랙은 인용
    시점에 (i) 조회일 (ii) 저장본 또는 스냅샷 해시 (iii) 인용한
    정확한 절/문장의 축자 인용을 함께 등재하라 — URL만으로는
    다음 세션의 재조회가 같은 답을 낸다는 보장이 없다. 상세
    `workspace/engine-port/results/cp_baseline/
    FOLLOWUP_STRATUM_2026-09-07.md` §가.4, `reports/
    serving_slo_survey.md` 정정 배너(2026-09-07), `CONSENSUS §3`
    항목112와 대응.

93. ★★★**(2026-09-08, cp_baseline/D1 트랙 규칙층 감사 3회차[rev3,
    트랙 10회차], claims-auditor + 메인 세션 독립 재현, GPU 0)
    다중비교 보정은 보정 강도를 정하는 양 `m`을 함께 등록해야
    완결된다 — 보정 도입 자체가 역인센티브를 만든다.** 死因
    W2. rev3이 Bonferroni 보정을 등록했으나 격자를 구성하는
    `m`(사다리·쌍·tolerance가 합성하는 비교 수)은 등록되지 않은
    실현량이다 — 실측 `t_crit(df=5)`: m=4일 때 **3.810**, m=40일
    때 **6.541**, m=142일 때 **8.592**(감사가 격자를 넓혀 재현).
    ⇒ **격자를 좁힐수록 보정이 약해지고 검정력이 오른다**(m=4의
    MDE 12.29% vs m=142의 27.71%) — 다중비교 보정을 도입하는
    설계자가 격자 크기를 스스로 정할 수 있으면, α가 명목상
    고정돼 있어도 실제 검정력이 설계자의 선택에 매달린다. 실무
    규칙: 다중비교 보정을 등록할 때는 (i) `m`을 만드는 fold를
    코드로 고정하고 (ii) 미보정값을 판정에 쓰면 반드시 실패하는
    변이를 붙이고 (iii) "격자를 좁히면 검정력이 오른다"는 역
    인센티브 문장을 명문으로 등재하라. 상세 `workspace/
    engine-port/results/cp_baseline/audit_d1_rules_3rd_2026-09-08/
    VERDICT.md` 死因 W1·W2, `CONSENSUS §3` 항목113과 대응.

94. ★★★**(2026-09-08, 상동) 기각과 동등성은 반대 방향의 보수성을
    요구한다 — 한 임계값이 둘을 다 정하면 한쪽은 반드시 틀린다.**
    死因 W3. rev3이 유의성 기각(부호 호출)과 동등성 선언
    (`band_binding='precise'`)에 **같은 보정 임계값**을 썼다 —
    보정 임계값(m=40, t=6.5414)에서 `precise`는 `SD/max(G) <
    1.123%`를 요구하는데 미보정 임계값에서는 `2.858%`였다. 보정
    기준으로 동등성에 도달하는 데 필요한 n은 SD 1.99/6.13/7.9/
    15%에서 각각 **10/50/78/267**(사전등록 §3.6이 등록한 미보정
    n은 5/17/28/99, 2칸은 산술 오류까지 겹침) — **격자를 넓혀
    검정력을 낮출수록 동등성 종료 라벨이 오히려 더 멀어진다.**
    기각은 큰 마진에서 보수적이어야 안전하고(1종 오류 억제),
    동등성은 좁은 마진에서 보수적이어야 안전한데(2종 오류/거짓
    동등 억제), 같은 `t_crit`으로 둘을 동시에 정하면 한쪽의
    보수성이 다른 쪽에서 반대 부호로 작동한다. 실무 규칙: 기각
    임계값과 동등성 임계값은 **별도로 등록**하고, 하나의 보정을
    둘 모두에 적용하는 설계라면 그 결정 자체를 논증으로 명문
    등재하라. 상세 `workspace/engine-port/results/cp_baseline/
    audit_d1_rules_3rd_2026-09-08/VERDICT.md` 死因 W3, `CONSENSUS
    §3` 항목114와 대응.

95. ★★★★**(2026-09-08, 상동, ★메인 세션 자신의 위반이 실측 사례)
    두 설명이 관측 동치인데 한쪽을 기각하려면 변수 하나만 바꿔야
    한다 — confound #10의 진단 판본.** 死因 W5. 2026-09-07에
    메인 세션이 "감사의 `len >= cps` 진단은 틀렸다(79.73%)"라고
    보고한 것 자체가 틀렸다 — 직접 재측정하면 cps 2048에서
    `(정본 규약 add_special_tokens=False, >=)`와 `(기본 규약
    add_special_tokens=True, >)`가 **둘 다 79.6748%**로 관측
    동치다. 메인 세션은 `>=`를 **기본 규약에서** 재서(79.73%)
    "부등호가 원인"이라 결론지었는데, 이는 **토크나이저 규약과
    부등호 두 변수를 동시에 바꾼 뒤** 한쪽(부등호)만 기각한
    것이다 — 참원인은 토큰화 규약(special token 1개)이었다.
    거짓 진술이 `d1_predicates.py` 주석(게이트 #80이 "출처
    허위는 규칙 정본 파일 안에서 가장 위험하다"고 지목한 바로
    그 자리)에 하루 있었고 2026-09-08 철회·교체(등록 상수 5개는
    불변, 원인 진술만 철회). 실무 규칙: 두 후보 설명이 같은
    관측값을 내는 지점을 찾았다면, 그중 하나를 기각하는 재측정은
    **정확히 한 변수만** 바꾼 대조로 설계하라 — 두 변수를 동시에
    바꾸면 어느 쪽을 기각했는지조차 알 수 없다. 상세
    `workspace/engine-port/results/cp_baseline/
    audit_d1_rules_3rd_2026-09-08/VERDICT.md` §9 死因 W5,
    `RESULT_D1_A1A2_2026-09-07.md` 정정 배너, `CONSENSUS §3`
    항목115와 대응.
    ★**追記(2026-09-11, G-6, R2 true-dual GPU correctness 판정서
    §2.5, claims-auditor, GPU 0 — confound #10 새 사례)**: 두
    설명이 관측 동치인 경우의 변종 — **검증 표적이 설정의 기본
    동작과 같으면, 그 설정에서의 실현 검사는 두 가설(정책 경로가
    실제로 액추에이션했다 vs 아무것도 안 해도 같은 값이 나왔다)을
    구별하지 못한다.** `pdmux_r2.yml`의 문턱이 전부 0이라 네이티브
    선택기도 overlap 시 idx 4(=D44)를 고르므로, D44에서는 B8/O3의
    idx 4 검사가 "FixedPolicy가 액추에이션했다"와 "아무 정책 없이도
    네이티브 선택기가 같은 분할을 골랐다"를 구별하지 못한다 —
    변별 증거는 `split_transition`(reason=fixed) 이벤트뿐이고,
    B8이 실제로 변별력을 갖는 것은 기본값이 아닌 D16/24/34에서뿐
    이다. 실무 규칙: 실현 검사를 설계할 때 "기본 동작이 이미 같은
    결과를 주는가"를 먼저 확인하고, 변별력이 필요하면 기본값이
    아닌 표적(예: X2, `R2C_DSM=16`)도 함께 돌려라. 대응
    `CONSENSUS §3` 항목115(追記).

96. ★★★**(2026-09-08, 상동) 판별식을 자기 저작 픽스처로 대체하면
    양성대조가 된다 — 원 대상 문서가 디스크에 있으면 반드시
    그것에 돌려라.** 死因 L8(반증 아님, 대체 부당 판정). rev2
    감사가 문서↔코드 대조 판별식(`D`-계열)을 검증할 때 자체
    제작한 합성 픽스처에 적용해 통과 여부를 판정했는데, 그
    판별식이 검사하려던 **원 대상 문서(rev1 사전등록)는 이미
    저장소·git에 존재**하고 규칙층 자기검사가 이미 로드하고
    있었다 — 자기 저작 픽스처로는 판별식이 실제 대상에 대해
    무엇을 잡고 무엇을 놓치는지 알 수 없다. 3회차 감사가 판별식을
    **rev1 실제 문서**에 직접 적용하자 요구 기준(5개 중 4개
    이상)을 충족하는 **4/5**가 나왔다(2회차는 자기 저작 픽스처로
    2/5를 내고 "요구 미달"로 판정했었다) — 원 데이터로 재실행
    했을 뿐인데 판정이 뒤집혔다. 실무 규칙: 코드↔문서 대조
    판별식(또는 인용/변이 검사)을 검증할 때, 검사 대상이 되는
    실제 문서가 저장소에 존재한다면 합성 픽스처가 아니라 **그
    문서에 직접** 적용해 판별식의 유효성을 확인하라 — 합성
    픽스처는 판별식이 "무언가에는" 반응한다는 것만 보이지,
    "찾아야 할 것을 찾는다"는 것을 보이지 않는다. 상세
    `workspace/engine-port/results/cp_baseline/
    audit_d1_rules_3rd_2026-09-08/VERDICT.md` §2 L8·§4,
    `CONSENSUS §3` 항목116과 대응.

97. ★★★★**(2026-09-08, 상동, 게이트 #14/항목18[=`CONSENSUS.md` §3
    항목27, ★오인용 정정 2026-09-10 — 상세 §3 항목27 追記]의 네 번째
    발화)
    게이트가 정본에 등재돼도 도구가 반대로 안내하면 재발한다 —
    규율은 도구에 심어야 전파된다.** 게이트 #14(2026-08-06)가
    이미 "n≤8에서 percentile bootstrap CI를 판정에 쓰지 말 것,
    primary는 t-CI"를 정본에 등재했는데, D1 rev2(2026-09-07)가
    **새 트랙에서 처음부터 다시 같은 선택**을 했다 — 死因 V2로
    잡힌 문제는 신규 발견이 아니라 게이트 #14의 재현(정본
    coverage n=4 .798/n=5 .840/n=6 .859/n=8 .888을 이 세션이
    독립 재현, .802/.838/.860/.882). ★기전 확인: `benchmarks/
    pdmux_eval/analyze.py`(게이트 #14가 지목한 바로 그 파일)를
    직접 읽자, ① t-CI 함수가 **아예 없고** ② docstring `:153`이
    오히려 `paired_bootstrap_ci`를 **권장**(n≤8 제한 언급 0)
    ③ CLI 경로 `:379`가 **n 가드 없이** 호출한다 — **게이트는
    문서에 있고 도구는 정반대를 말하며, 새 트랙이 참조하는 것은
    도구다.** 이번이 이 진단의 **네 번째 발화**(1–2: 게이트 #14
    이전 두 트랙이 각자 로컬로 t-CI 우회 / 3: P1 판정서가
    percentile을 계속 씀 → 게이트 #14 등재 / 4: 이번, 정본이
    바뀌었는데도 도구가 안 바뀌어서 재발). ⚠️도구는 이번에도
    **고치지 않았다**(여러 트랙 공유, 파급이 트랙 밖 — 사용자
    승인 필요). 실무 규칙: 게이트를 정본 문서에 등재하는 것과
    별개로, 그 게이트가 지목한 라이브러리 함수 자체에 (i) 준수
    함수를 추가하고 (ii) 위반 함수의 docstring에 게이트를 인용
    하고 (iii) 위반 호출을 n 가드로 거부/경고하게 만들어야
    전파가 보장된다 — 문서 등재만으로는 다음 트랙이 또 같은
    함정을 판다. 상세 `workspace/engine-port/results/cp_baseline/
    FINDING_GATE14_TOOLING_2026-09-08.md`, `reports/
    AUDIT_DEBT_2026-08-23.md` §10, `CONSENSUS §3` 항목117과 대응.

98. ★★★★**(2026-09-08, 2차 세션, 게이트 #14 도구 수리 설계서
    규칙층 감사, claims-auditor + 메인 세션 독립 재현, GPU 0)
    판정 경로의 하한 부재는 추정량 교체로 드러나지 않는다 —
    그리고 그 부재를 고치려는 변이를, 같은 수리가 낸 테스트
    자신이 죽일 수 있다.** 死因 F1. `analyze.py`에 t-CI를
    도입하며 부트스트랩→t로 추정량을 옮겼으나, 판정 경로
    (`headline_improvement`)에는 여전히 n 하한이 없다(코드
    주석은 "gate #3 still requires n>=4"라 적고 집행은 0건).
    실측: 수리 후에도 **n=2에서 `headline_improvement=True`**,
    n=3(`ci95_low=+1.91`, effect 25.3%)은 **n만이 거부 사유**가
    돼야 하는데 통과. 결정적으로, 게이트 #3 하한을 *추가*하는
    변이(`pairs>=4` 조건 삽입)를 이 수리가 함께 낸 새 테스트
    (n=3 페어로 `headline_improvement is True`를 단언)가 **직접
    킬**한다 — 즉 회귀 스위트가 "수리"가 아니라 "구현의 현재
    상태"를 보호한다(게이트 #9의 교과서적 재발). 실무 규칙:
    추정량을 게이트가 요구하는 방향으로 바꿨다고 해서 그 게이트가
    요구하는 **문턱**(n 하한 등)까지 집행된 것은 아니다 — 판정
    경로에 하한이 실제로 걸리는지는 **강화 방향 변이**(문턱을
    엄격화하는 변이)로 반드시 확인하고, 회귀 스위트가 그 변이를
    죽인다면 스위트 자체가 오래된 스냅샷을 지키고 있다는 신호로
    읽어라. 상세 `workspace/engine-port/results/tooling_gate14/
    audit_design_2026-09-08/VERDICT.md` 死因 F1, `CONSENSUS §3`
    항목118과 대응.

99. ★★★**(2026-09-08, 2차 세션, 상동) 감사 지적 자체가 과장일
    수 있고, 그 과장이 정정문으로 그대로 전파될 수 있다.** 死因
    아님(차단 B5, 메인 세션 재확인). 규칙층 감사가 설계서
    docstring의 *"there is no SciPy in the serving venv"*를
    "사실 거짓"이라 판정했고, 메인 세션은 그 라벨을 검증 없이
    설계서 배너에 그대로 옮겨 적었다. 직접 재확인한 결과 **venv
    자체의 site-packages에는 정말 scipy가 없다** — 보이는 이유는
    `pyvenv.cfg`의 `include-system-site-packages=true`로 사용자
    홈(`~/.local`)의 설치가 노출되기 때문이다. 즉 **문장은 문자
    그대로 참**이고, 거짓인 것은 독자가 끌어낼 함의("그러므로
    import할 수 없다")뿐이었다 — 감사의 판정 자체가 과장이었다.
    처방은 바뀌지 않는다(판정 코드가 사용자 홈의 unpinned 설치에
    의존해서는 안 되므로 stdlib 전용 설계는 오히려 근거가
    강해진다). 부수 발견: 같은 전제("scipy 없음")가 **7개 파일
    10개 지점**(`analyze.py` 2곳·`d1_predicates.py`·
    `e1a_analyze.py`·`g2det_analyze.py`·`g2ctrl_analyze.py`·
    `design_g13_stats.py` 3곳·`verify_g2_tau40_lib.py`)에
    상속돼 있으며 **전부 미수리**(트랙 밖 승인 필요). 실무 규칙:
    적대 감사의 死因·차단 지적도 판정 오류를 낼 수 있다 —
    "감사가 지적했다"는 것 자체가 사실 확인을 면제하지 않으며,
    특히 부정 존재 진술("X가 없다")은 검색 범위(venv 자체 vs
    `sys.path` 전체)를 먼저 명시해야 참/거짓이 정의된다. 상세
    `workspace/engine-port/results/tooling_gate14/
    CORRECTIONS_2026-09-08.md` C3, `CONSENSUS §3` 항목119와 대응.

100. ★★★**(2026-09-08, 2차 세션, `longctx_conflict` 트랙,
     메인 세션 자기 발견 + GPU-0 재측정, 게이트 #74/§1-1의
     시간가중 함정 계열) 구간 부과 규칙이 순간 상태를 지속
     상태로 바꿀 수 있다.** `e1_pin_check::compute_decode_realized`
     는 스냅샷 `i`에서 관측된 상태를 그 bin에 **`dt = t[i+1] −
     t[i]`(다음 스냅샷까지의 간격) 전체**로 부과한다 — 상태가
     실제로는 표본 간격보다 훨씬 짧게 스쳤어도 간격 전체가
     그 상태로 청구된다. 실측(`T_C16384_D16`, `PDMUX_DUAL_WORKER_
     TRACE_EVERY=32`): `stream_index=0`(bin0) 스냅샷 **64개
     전부 연속 런 길이 1**(두 번 연속으로 관측된 적이 없다) ·
     그 시점 `prefill_active_batch_size` **0/64**(경합 중이
     아니었다) · 부과된 dt median **0.4059 s** = 전체 decode-active
     dt median **0.4063 s**(사실상 동일) ⇒ 순간 상태 64개가
     표본 간격 하나씩을 부과받아 **25.70 s**로 부풀었다. 기전은
     스냅샷 필드 간 read-skew(`stream_index=0`은
     `running_batch.is_empty()`일 때만 설정되는데 표본된
     `decode_running_batch_size`는 0보다 크다 — 필드들이 같은
     시점에 표본되지 않는다). 실무 규칙: 상태-지속시간 추정량을
     문턱 게이트로 쓰기 전에, 그 추정량이 순간 상태에 표본 간격
     전체를 부과하는 구조인지 확인하라 — 부과 규칙을 고치지
     않으면 문턱값(이 경우 `0.90` vs `0.95`)의 의미가 **표본화율
     자체의 함수**가 돼 버린다(케이던스를 바꾸면 문턱 통과 여부가
     바뀔 수 있다는 뜻 — 게이트 #15의 "케이던스 불변성" 함정과
     자매 형태이나, 이번엔 불변성이 아니라 **부과량이 표본화율에
     비례해서 커지는** 반대 방향의 함정이다). 상세
     `workspace/engine-port/results/longctx_conflict/
     FINDING_BIN0_2026-09-08.md`, `CONSENSUS §3` 항목120과 대응.

101. ★★★**(2026-09-09, `longctx_conflict` 트랙, `PREREG_RATIO`
     사전등록 감사 死因 R1 + `ADJUDICATION_P4_2026-09-09.md`,
     claims-auditor + 메인 세션 자기 발견, GPU 0.818 GPU-h[P4])
     비율 문턱을 등록할 때는 그 양의 천장(또는 바닥)을 먼저
     계산하라 — 안 그러면 그 검정은 항등식이다(게이트 #9의
     사전등록 판본).** 상한이 1인 비율(`f_count`)에 "ON이 OFF의
     ≥3배"를 등록했으나, OFF 기준값이 **0.9722**로 이미 천장
     근처라 산술적으로 가능한 최대 배수는 `1.0/0.9722=1.029×`
     뿐이었다 — 데이터가 무엇이든 `ge_3x`는 거짓이 될 수밖에
     없었다. 실행 후 이를 "코드 독해 철회"로 읽었으나(P4_LABEL.json
     `prediction_2.reading`) 이는 **측정 실패를 게이트 실패로
     라벨링한 것**(게이트 #21, 이 저장소 8회+ 재발)이었고,
     `ADJUDICATION_P4_2026-09-09.md`가 **VOID**(기각 아님)로
     정정했다. 후속 프로브(P4-B)는 등록 전에 천장을 계산해
     기준 0.3(최대 가능 3.3×)에 문턱 2×를 걸어 도달 가능하게
     설계했다. 실무 규칙: 비율 문턱을 사전등록에 넣기 전에
     `(관측 가능한 최댓값 또는 최솟값) − 기준값`으로 도달가능
     구간을 계산하고, 등록 문턱이 그 구간 밖이면 **그 검정
     자체를 폐기**하라(귀책은 실행자가 아니라 등록자에게 있다).
     상세 `workspace/engine-port/results/longctx_conflict/
     probes/{P4_LABEL.json, ADJUDICATION_P4_2026-09-09.md}`,
     `CONSENSUS §3` 항목121과 대응.

102. ★★★**(2026-09-09, `longctx_conflict` 트랙, `RESULT_DUTYCYCLE_
     G16` 결과 감사, `audit_dutycycle_2026-09-08/VERDICT.md`
     死因 R1, claims-auditor + 메인 세션 독립 재검증, GPU 0)
     두 편향이 반대 방향으로 표류하면 그 곱이 이론 예측과
     일치할 수 있다 — 예측과의 일치는 최소 두 개의 독립
     가중에서 재현돼야 증거다.** 같은 telemetry를 카운트-가중
     하면 초선형 추세(비 3.517×)가 나오지만, 시간-가중하면 그
     초선형 추세를 arm에 단조 감소하는 **구간-부과 길이편향**
     (비 0.757, `compute_decode_realized`가 짧은 상태에 표본
     간격 전체를 부과)이 정확히 상쇄해 "선형"처럼 보이는 비
     **2.659(=3.517×0.757)**를 만든다. 정식 검정(자기 4-블록
     분산)으로는 d34 z=−3.49·d54 z=−4.01로 **기각**되고, 같은
     데이터를 다르게 가중하면 지수가 `b=0.982`(시간)/
     `1.263`(카운트)/`1.921`(prefill in-flight)로 갈린다 — 지수
     자체가 가중 선택의 산물이라 "물리 법칙과 일치"라고 쓸 수
     없다. 실무 규칙: 이론값과의 산술적 일치를 보고할 때는
     추정량을 최소 하나 더 바꿔 **같은 방향·같은 크기로
     재현**되는지 반드시 확인하라 — 한 가중에서만 맞아떨어지는
     지수는 우연한 상쇄일 수 있다. 상세 `workspace/engine-port/
     results/longctx_conflict/{RESULT_DUTYCYCLE_G16_2026-09-08.md,
     audit_dutycycle_2026-09-08/VERDICT.md}`, `CONSENSUS §3`
     항목122와 대응.

103. ★★★★**(2026-09-09, `longctx_conflict` 트랙, `PREREG_RATIO`
     설계 vs `CONSENSUS §1-26(B)`[2026-08-03] 대조, 메인 세션
     자기 발견 + claims-auditor 사전등록 감사 死因 R1, GPU 0)
     정본이 이미 바뀌었는데 새 설계가 그 정본을 안 읽으면,
     게이트 #18[★오인용 정정 2026-09-10: 정본=게이트#14/
     `CONSENSUS.md` §3 항목27 — 상세 §3 항목27 追記](같은 진단을 두 번 냄)보다 나쁜 형태의 재발이
     생긴다.** `CONSENSUS §1-26(B)`(2026-08-03)는 이미 "이
     기판에서 decode가 D SM에서 돌았다 ⟺ prefill이 동시에
     in-flight였다"는 항등식과 그 처방(`PDMUX_STICKY_PARTITION`
     신규 측정이 유일한 해소 경로)을 등재해 뒀다. 그러나
     2026-09-08 `PREREG_RATIO` 설계는 이 항목을 인용도 반박도
     하지 않은 채 같은 함정(처치 실현 듀티사이클이 셀마다
     기전적으로 다름)을 처음부터 다시 설계했다가 死因 R1로
     죽었다 — §1-26(B)를 읽었더라면 셀 A의 `f≈0.004`가 등록
     전에 산술로 예견 가능했다. 이것은 게이트 #18[★오인용 정정
     2026-09-10: 정본=게이트#14/`CONSENSUS.md` §3 항목27 — 상세 §3
     항목27 追記]("저장소가
     같은 진단을 두 번 냈는데 정본이 안 바뀌면 도구 규율
     실패")보다 나쁘다 — 이번엔 **정본이 이미 바뀌어 있었는데도**
     새 설계가 그것을 조회하지 않았다. 실무 규칙: 새 설계를
     사전등록하기 전에 그 설계의 노출변수·estimand가 언급하는
     코드 경로에 대해 `CONSENSUS.md`·`PROJECT_STATUS.md`를
     **파일명이 아니라 코드 위치(파일:줄)로 grep**하라 — 트랙
     이름이 다르다는 이유로 관련 항목을 건너뛰지 마라. 상세
     `workspace/engine-port/results/longctx_conflict/
     {PREREG_RATIO_2026-09-08.md, FINDING_DUTYCYCLE_2026-09-08.md,
     audit_ratio_rules_2026-09-08/VERDICT.md}`, `CONSENSUS §1-25
     追記·§1-26(B)`, `CONSENSUS §3` 항목123과 대응.

104. ★★★**(2026-09-09, `longctx_conflict` 트랙, `PREREG_SWEEP`
     rev1 규칙층 감사 死因 W1, `audit_sweep_rules_2026-09-09/
     VERDICT.md`, claims-auditor, GPU 0) 항등식 검사는 점수
     함수가 아니라 결정 규칙 위에서 하라.** 게이트 #9의 15번째
     재발. 사전등록 §3.1은 goodput 점수함수 `G`를 전개해
     "설계 상수가 안 들어간다"를 보였고 그 전개는 맞았으나,
     실제로 판정에 쓰인 것은 `G`가 아니라 **`argmax_D G`의 두
     `R` 간 불일치**였다. 전개하면 요청별 ITL95 분산이 arm 간
     격차보다 훨씬 작아 `1[p95(ITL_i)≤I_slo]`가 요청별 판별자가
     아니라 **셀 상수 지시함수**로 붕괴하고, `argmax_D G`가
     "이미 측정된 ITL95 표에서 최소 `D`를 고르는" 결정론적
     함수로 무너졌다. 실무 규칙: 결정량(점수함수)이 항등식이
     아님을 확인하는 것으로 충분하지 않다 — **그 점수함수 위에
     얹힌 argmax/기각/선택 규칙까지 전개해 데이터-무관 극단값이
     있는지 확인**하라. 정본 `CONSENSUS §3` 항목124(신설),
     `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#104, 신설).
     상세 `workspace/engine-port/results/longctx_conflict/
     {PREREG_SWEEP_2026-09-09.md, audit_sweep_rules_2026-09-09/
     VERDICT.md}`.

105. ★★★**(2026-09-09, `longctx_conflict` 트랙, `PREREG_SWEEP`
     rev1 규칙층 감사 死因 W2 + rev2 감사 死因 X2, claims-auditor,
     GPU 0) 절벽 회피 절차가 답 선택 절차와 동치일 수 있다**
     (게이트 #6의 사전등록 판본). rev1 §2.1은 `I_slo`를 "두
     관측값(47.0ms·87.1ms) 양쪽에서 멀리" 골라 60ms로 정했으나,
     `ARGMAX_MOVES` 발화 구간이 정확히 그 두 관측값 사이였다 —
     "두 관측값 양쪽에서 멀리"는 "발화 구간의 내부"와 같은 말이다.
     rev2가 단일 문턱 대신 격자 사각형 `[1,70]s×[10,100]ms`로
     대체해도 재발했다(死因 X2): 격자 4모서리가 판정량 `A`를
     0.359–0.846으로 흔드는 **새 문턱**이었다. 실무 규칙: 문턱을
     "두 관측값 사이를 피해" 고르는 절차는, 그 두 관측값의
     **순서 자체가 결론**일 때 결론을 강제한다 — 문턱 대신 지도·
     격자로 대체해도 그 지도의 경계(모서리·범위)가 같은 함정을
     재생산할 수 있다. 정본 `CONSENSUS §3` 항목125(신설),
     `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#105, 신설).
     상세 `workspace/engine-port/results/longctx_conflict/
     {audit_sweep_rules_2026-09-09/VERDICT.md,
     audit_sweep_rev2_2026-09-09/VERDICT.md}`.

106. ★★★**(2026-09-09, `longctx_conflict` 트랙, `PREREG_SWEEP`
     rev1 규칙층 감사 死因 W3, claims-auditor, GPU 0) 검정력을
     승계할 때는 `n`이 무엇의 개수인지 함께 승계하라**(게이트
     #3/#88 결합형). `PREREG_SWEEP`은 `RESULT_RULEPOWER_
     2026-09-08.md` 부록 A의 "`n=6` 검정력 0.868"을 인용해
     "1셀=1부팅=6 bench(시드만 다름)"로 설계했으나, 인용 출처
     본문은 *"구속 잡음은 요청 수가 아니라 **부팅×split 랜덤
     효과**다. 창을 늘리는 것으로는 못 산다 — **부팅 수로만
     산다**"*로 정확히 반대를 명시하고 있었다 — 부록 A의 `n`은
     **부팅 수**였다. `PREREG_P6_ITL_ORDER`가 이 게이트를 정확히
     이행(부팅 수·시드 수를 구분해 검정력 비승계를 명시)한 첫
     사례다. 실무 규칙: 다른 문서에서 검정력·SD를 수입할 때는
     그 표본 크기 `n`이 부팅인지 시드인지 요청인지를 **원문에서
     확인하고 같은 종류로만** 쓰라 — 형식상 숫자를 맞춰도 종류가
     다르면 검출력이 정반대로 무너질 수 있다. 정본 `CONSENSUS §3`
     항목126(신설), `PROJECT_STATUS.md` "방법론 게이트" 이
     항목(#106, 신설). 상세 `workspace/engine-port/results/
     longctx_conflict/{audit_sweep_rules_2026-09-09/VERDICT.md,
     PREREG_P6_ITL_ORDER_2026-09-09.md §4-C}`.

107. ★★★**(2026-09-09, `longctx_conflict` 트랙, `PREREG_SWEEP`
     rev2 규칙층 감사 死因 X2, `audit_sweep_rev2_2026-09-09/
     VERDICT.md`, claims-auditor, GPU 0) 문턱을 없앴다면 그
     자리에 무엇이 들어왔는지 세어라**(게이트 #9/#105의 격자
     판본). 단일 문턱 `I_slo`를 격자 사각형으로 대체하면 자유도
     는 사라지는 것이 아니라 **격자 모서리 개수만큼 늘어난다**
     (1→4) — 같은 원자료에서 판정량 `A`가 `[10,60]ms`→0.8458 /
     `[10,100]`→0.6832 / `[40,100]`→0.3591로 흔들렸다. "자유도
     0" 선언은 주장이 아니라 **격자 민감도 표**로 증명해야
     한다. 실무 규칙: 문턱→지도(격자) 치환으로 자유도를 없앴다고
     쓰기 전에, 격자 경계를 바꿔가며 판정량이 얼마나 흔들리는지
     표로 등록하라. 정본 `CONSENSUS §3` 항목131(신설),
     `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#107, 신설).
     상세 `workspace/engine-port/results/longctx_conflict/
     {PREREG_SWEEP_REV2_2026-09-09.md,
     audit_sweep_rev2_2026-09-09/VERDICT.md}`.

108. ★★★**(2026-09-09, `longctx_conflict` 트랙, `PREREG_SWEEP`
     rev2 규칙층 감사 死因 X6, claims-auditor, GPU 0) 직전
     감사의 오류가 다음 판본의 등록 상수로 승격될 수 있다 —
     감사 지적으로 도입된 제약은 그 지적의 근거를 독립
     재검증한 뒤에 등록하라**(게이트 #99의 다음 단계).
     `PREREG_P5`·rev2가 등록한 `N∈{45,84,170}` 제약은 "감사
     모형과 메인 세션 모형이 RNG 소비에서 갈린다"는 전제 위에
     섰으나 그 전제 자체가 **거짓**이었다(`randint(L,L+1)`은
     MT19937 상태를 소비하지 않음, 메인 세션 독립 재현으로
     확인) — 근거 없는 제약이 그대로 등록돼 **arm 간 `N` 3.8×
     불일치**라는 새 死因(X5)을 낳았다. ★**#113과의 구분**:
     **#108**은 *감사의 오류(틀린 전제)가 다음 판본의 등록
     상수로 승격되는 것*을 다루고("그 지적이 참인가"), **#113**
     은 *감사의 처방이 실현 가능한지*를 다룬다("그 지적을 따른
     설계가 서지는가") — 전자는 사실 검증 실패, 후자는 실현가능성
     검증 실패다. 실무 규칙: 이전 회차 감사가 지적한 제약을 다음
     판본에 그대로 이식하기 전에, 그 지적의 근거(코드·수치)를
     GPU 0으로 독립 재검증하라. 정본 `CONSENSUS §3` 항목132
     (신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#108,
     신설). 상세 `workspace/engine-port/results/longctx_conflict/
     {PREREG_P5_ITL_LOCATE_2026-09-09.md,
     audit_sweep_rev2_2026-09-09/VERDICT.md}`.

109. ★★★**(2026-09-09, `longctx_conflict` 트랙, `PREREG_SWEEP`
     rev2 규칙층 감사 死因 X3-d, claims-auditor, GPU 0) ITT
     선언은 estimand 절에서 지켜지고 규칙 절에서 깨질 수 있다
     — 후처치 변수를 "게이트"가 아니라 "해석 가능성 강등
     조건"으로 써도 그것도 조건화다.** rev2 §0은 ITT를
     선언(노출=설정값, 실현 파티션은 병기 공변량, 조건화
     금지)했으나 등록 규칙 `T4 OVERLAP_CHECK`는 처치 하류
     변수인 `L_decode_exact` 실측값으로 `R` 축 해석 가능성을
     강등했다 — 명목상 "게이트"가 아니어도 판정 경로에 처치
     하류 변수가 들어오면 estimand 선언과 무관하게 조건화가
     재발한다. 실무 규칙: 판정 경로에 들어오는 모든 양에 대해
     처치-상류/하류 여부를 규칙별로 표기하고, 하류 변수가
     판정(라벨 강등 포함)에 관여하면 그 규칙을 삭제하거나 별도
     진단으로 격리하라. 정본 `CONSENSUS §3` 항목133(신설),
     `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#109, 신설).
     상세 `workspace/engine-port/results/longctx_conflict/
     {PREREG_SWEEP_REV2_2026-09-09.md,
     audit_sweep_rev2_2026-09-09/VERDICT.md}`.

110. ★★★**(2026-09-09, `longctx_conflict` 트랙, `PREREG_SWEEP`
     rev2 규칙층 감사 死因 X4, claims-auditor, GPU 0) 검정력을
     승계할 때는 `n`의 종류에 더해 "어느 결정 규칙의"
     검정력인지도 승계하라 — 분산을 재는 프로브는 본 캠페인의
     판정량과 같은 통계량을 재야 한다**(게이트 #106의 나머지
     절반). `PREREG_P5`는 `σ(요청별 p95)`·`σ(L_decode)`·
     `σ(G)`를 쟀으나 rev2의 실제 판정량은 `A`(argmax 일치율)
     였다 — `σ(G)→Var(A)` 사상이 등록돼 있지 않아 그 분산으로
     rev2의 검정력을 정할 수 없었다(무력, 순환은 아님). 실무
     규칙: 선행 프로브를 설계할 때 "무엇의 분산을 잴 것인가"를
     그 분산이 뒷받침해야 할 본 캠페인의 최종 판정 통계량과
     동일하게 맞추라 — 상관된 다른 통계량의 분산은 대용물이
     아니다. 정본 `CONSENSUS §3` 항목134(신설), `PROJECT_
     STATUS.md` "방법론 게이트" 이 항목(#110, 신설). 상세
     `workspace/engine-port/results/longctx_conflict/
     {PREREG_P5_ITL_LOCATE_2026-09-09.md,
     audit_sweep_rev2_2026-09-09/VERDICT.md}`.

111. ★★★**(2026-09-09, `longctx_conflict` 트랙, `PREREG_P6_
     ITL_ORDER` 규칙층 감사 死因 Y1, `audit_p6_rules_2026-09-09/
     VERDICT.md`, claims-auditor, GPU 0) 사전등록 예측은 실행
     전에 기존 원자료로 예보하고 그 예보를 문서에 적어라 —
     예보가 만장일치(또는 정의 불능)면 그 예측은 산출이 아니다.**
     게이트 #104의 한 층 위 판본. `PREREG_P6`의 예측 E1·E2는
     `PREREG_SWEEP_REV2` §0이 이미 "참이라 문턱으로 피할 수 없는
     전제"로 등재해 둔 두 단조성 그 자체였다 — 감사가 기존
     원자료(프로브 C 12셀)만으로 미리 계산해 보니 E1은 **4
     rate-rung×4 추정량 16/16 만장일치**, E2는 등록 정의역
     `𝒯`가 **9판본 중 6판본에서 공집합**이었다. 즉 P6를 실제로
     실행하지 않고도(GPU 0) 그 판정을 예보할 수 있었다. 실무
     규칙: 사전등록 예측을 쓰기 전에 그 예측을 **기존 원자료로
     먼저 계산**해 보고, 결과가 만장일치이거나 정의역이 비어
     있으면 그 예측을 실행으로 사는 것은 이미 아는 답을 다시
     사는 것이다. 정본 `CONSENSUS §3` 항목127(신설),
     `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#111, 신설).
     상세 `workspace/engine-port/results/longctx_conflict/
     {PREREG_P6_ITL_ORDER_2026-09-09.md, audit_p6_rules_2026-09-09/
     VERDICT.md}`.

112. ★★★**(2026-09-09, `longctx_conflict` 트랙, `PREREG_P6_
     ITL_ORDER` 규칙층 감사 死因 Y2·Y3, claims-auditor, GPU 0)
     정의역을 데이터로 정의하면 그 정의역이 비어 있을 수 있고
     `∅`에는 판정이 없다.** `PREREG_P6`의 E2는 문턱 자유도를
     없애려고 정의역 `𝒯 := {T : 세 CDF 전부 ∈ (0.05,0.95)}`를
     데이터 자신의 지지집합으로 자기정의했으나, 등록 설계점에서
     `𝒯 = ∅`이 9판본 중 6판본·두 `R` 전부였다 — E2의 실제
     출력공간이 `{UNDEFINED, TRUE(예정됨)}`으로 무너지고
     `거짓`은 도달 불가했다. 게다가 `𝒯`의 크기(`|𝒯|`)가 **처치
     강도의 감소함수**였다(모든 `W`에서 최대가 처치 null인
     `D=44`) — 정의역이 estimand와 반대 방향으로 움직였다.
     자기정의 정의역은 문턱을 없애는 것이 아니라 **정의역
     경계로 자유도를 이동**시킬 뿐이다. 실무 규칙: 데이터로
     정의역을 정의할 땐 (i) `∅` 케이스의 판정을 사전등록에 함께
     적고 (ii) 정의역 크기가 처치 강도와 어느 방향으로 상관하는지
     실행 전에 계산하라(반대 방향이면 그 설계는 위험 신호다).
     정본 `CONSENSUS §3` 항목128(신설), `PROJECT_STATUS.md`
     "방법론 게이트" 이 항목(#112, 신설). 상세
     `workspace/engine-port/results/longctx_conflict/
     {PREREG_P6_ITL_ORDER_2026-09-09.md,
     audit_p6_rules_2026-09-09/VERDICT.md}`.

113. ★★★**(2026-09-09, `longctx_conflict` 트랙, `PREREG_P6_
     ITL_ORDER` 규칙층 감사 死因 Y4, claims-auditor, GPU 0)
     직전 판정서의 처방도 실현가능성을 검사하라 — 처방은 자유
     표면 이동의 새 통로다.**(교훈 항목99의 강화판본). rev2
     감사(`audit_sweep_rev2`)가 死因 X1을 피하려 "공통 `W`"를
     처방했고 `PREREG_P6`은 그 처방을 그대로 받아 설계했으나,
     이 arm 집합(`{16,44,92}`)의 `μ_p` 확산(4.99×)이 §1.1이
     스스로 등록한 두 제약(하한 `ρ≥0.6`·상한 drain 2배)을 동시에
     만족할 수 없어 **처방한 rev2 자신도 함께 죽었다** — 등록값
     `out=96, W=0.60`이 상한을 위반했는데(`ρ_92=3.21>2`) 위반
     라벨 없이 "기판의 성질"로 재서술됐고, `out=384, W=0.25`
     에서는 하한을 만족하는 유일 arm이 용량 사망 arm(d92)이었다.
     실무 규칙: 감사·리뷰가 다음 라운드에 특정 설계를 처방할
     때, 그 처방이 **처방 대상 자신이 이미 등록한 다른 제약과
     양립 가능한지**를 처방 시점에 GPU 0으로 검산하라 —
     처방자의 산술 오류는 피처방자가 그대로 물려받는다. ★**#108
     과의 구분**: #108은 *감사의 오류(틀린 전제)가 등록 상수로
     승격되는 것*, #113(이 항목)은 *감사의 처방이 실현 가능한지*
     를 다룬다 — 전제가 참이어도 처방은 실현 불가능할 수 있다.
     정본 `CONSENSUS §3` 항목129(신설), `PROJECT_STATUS.md`
     "방법론 게이트" 이 항목(#113, 신설). 상세
     `workspace/engine-port/results/longctx_conflict/
     {PREREG_SWEEP_REV2_2026-09-09.md, PREREG_P6_ITL_ORDER_
     2026-09-09.md, audit_p6_rules_2026-09-09/VERDICT.md}`.

114. ★★**(2026-09-09, `longctx_conflict` 트랙, `PREREG_P6_
     ITL_ORDER` 규칙층 감사 死因 Y5, claims-auditor, GPU 0)
     "창을 맞췄다"고 할 때 어떤 창인지 문자로 고정하라.**
     `PREREG_P6` §1.2는 6셀 전부 **bench 길이**를 179–182초로
     맞춰 "warmup·과도상태 분율이 arm 간 동일"이라 주장했으나,
     실제로 맞아야 했던 것은 **도착 창**이었다 — d16/d44의
     실현 도착 창(151.9–191.1s)과 d92의 도착 창(56.7s)이 3.2×
     달랐고, d92 bench의 70%가 무도착 drain(d16/d44는 ~3%)이라
     그 주장은 23× 거짓이었다(P6 자신의 §4-B 등록 상수가 반증).
     `duration`을 맞추면 요청 수(`N`)가 arm마다 달라지고, `N`을
     맞추면 `ρ`가 다른 arm은 도착 창이 어긋난다 — 두 창(요청
     드는 시간 vs bench가 끝나는 시간)은 일반적으로 동시에
     맞출 수 없다. 실무 규칙: "창을 맞췄다"·"조건을 동일화했다"
     류의 주장을 쓸 때는 **정확히 어떤 창(도착 창/관측 창/bench
     duration/N)을 맞췄는지 문자로 명시**하고, 그로 인해 다른
     창이 얼마나 벌어지는지 계산해 함께 적으라. 정본
     `CONSENSUS §3` 항목130(신설), `PROJECT_STATUS.md`
     "방법론 게이트" 이 항목(#114, 신설). 상세
     `workspace/engine-port/results/longctx_conflict/
     {PREREG_P6_ITL_ORDER_2026-09-09.md,
     audit_p6_rules_2026-09-09/VERDICT.md}`.
     ★**追記(2026-09-11, G-5, R2 true-dual GPU correctness
     판정서 §6, claims-auditor, GPU 0 — 파티션 실현의 "어떤 창"
     판본)**: 같은 교훈이 파티션 실현에도 적용된다 — "D44 위에서
     돌았다"고 쓸 때도 정확히 어떤 창(overlap 스냅샷 창/비교
     대상 산출이 실제로 실행된 창)을 말하는지 명시해야 한다.
     job 907100에서 overlap 스냅샷 창은 전부 idx 4였지만, 비교
     대상 계산이 실제로 그 분할 위에서 돈 창은 probe prefill뿐
     이었다(§2.3). 대응 `CONSENSUS §3` 항목130(追記).

115. ★★★**(2026-09-09, `longctx_conflict` 트랙, doc-steward 정본
     등재 절차 중 발견·메인 세션 자기 정정, GPU 0) 에이전트
     반환문을 파일로 옮길 때(전사) 절 단위로 통째로 누락될 수
     있고, 그것을 잡아내는 지점은 등재(정본화) 절차의 상호참조
     대조일 수 있다.** `audit_sweep_rev2_2026-09-09/VERDICT.md`
     원문에 있던 "신규 방법론 게이트 후보 4건(#107–110)" 절이
     메인 세션이 그 판정서를 파일로 옮기는 과정에서 통째로
     빠졌다 — 파일 자체는 정상적으로 존재했고 다른 절은 멀쩡
     했으므로 diff/lint류 도구로는 안 잡혔다. 발견 경로는
     doc-steward가 정본에 게이트 번호를 등재하며 *"#107–110을
     인용하는 문서가 여럿인데 그 정의문이 어느 커밋된 문서에도
     없다"*는 **상호참조 불일치**를 짚은 것뿐이었다. ★**게이트
     #18과의 구분**[★오인용 정정 2026-09-11: 여기서 말하는 "#18"의 정본은 **게이트 #14**(`CONSENSUS §3` 항목27)다. 2026-09-10 오인용 정정 물결이 이 항목 **내부** 인용을 놓쳤다 — 실제 게이트 #18은 "과부하 arm 비교는 시스템 상수가 아니다"(`§3` 항목32)]: **#14**는 *저장소에 이미 존재하는 진단*이
     상위 정본에 전파되지 않는 실패(진단은 살아있고 전파가
     실패)인 반면, 이것은 *저장소에 진단이 아예 기록되지
     못한*(전사 자체가 실패) 경우다 — **#14**의 상위 호환이 아니라
     **더 이른 단계의 실패**(전파 이전, 최초 기록 단계)다.
     실무 규칙: 감사·판정서 반환문을 파일로 옮긴 뒤, 그 파일이
     인용하거나 다른 문서에서 인용될 수 있는 게이트/항목 번호가
     전부 그 파일 안에 실재하는지 **정본 등재 직전에 기계적으로
     대조**하라 — "#N을 인용하는 곳"과 "#N을 정의하는 곳"의
     집합이 일치하는지 grep으로 확인하는 것이 최소 방어선이다.
     정본 `CONSENSUS §3` 항목135(신설), `PROJECT_STATUS.md`
     "방법론 게이트" 이 항목(#115, 신설). 상세
     `workspace/engine-port/results/longctx_conflict/
     audit_sweep_rev2_2026-09-09/VERDICT.md`(§ "신규 방법론
     게이트 후보 4건" 복원 기록 배너).

116. ★★★**(2026-09-10, `longctx_conflict` 트랙, P7 `ITL_SIGMA`
     계측[job 905994], 메인 세션 직접 분석, GPU 0[분석 자체])
     분산성분 추정이 음수로 나오면 그 추정량은 "0"이 아니라
     "미검출"이다 — `n`이 작을 때(예: `n_boot=4`⇒`df=3`) ANOVA류
     분산성분 분해는 원값이 음수가 될 수 있고, 이를 0으로
     클립하는 것은 점추정이 아니라 검출 실패다. 인용해야 할
     산출물은 F-검정 기반 95% 상한이며, `F<F₀.₀₅`인 셀은 그
     상한 자체가 퇴화(0.0%)한다 — 이것도 "효과 없음"이 아니라
     "상한이 정의되지 않음"으로 읽어야 한다.** P7에서 3 arm×2
     지표(`itl95`·`μ_p`) 전부 분산성분 원값이 음수였고,
     `L_decode` 지표 2개(d44/d54)는 `F<F₀.₀₅`로 상한 자체가
     퇴화했다(`RESULT_P7_ITL_SIGMA_2026-09-10.md §2`). 실무 규칙:
     부팅간(또는 임의 랜덤효과) 분산을 소수의 반복(`n_boot`가
     한 자릿수)으로 추정할 때는 (i) 분산성분 원값의 부호를
     반드시 보고하고 (ii) 음수/0-클립을 "효과 없음"이 아니라
     "이 `n`에서 미검출"로 명시하고 (iii) 산출물을 F-검정 기반
     95% 상한으로 대체하되 `F<F₀.₀₅` 셀은 별도로 "퇴화" 라벨을
     달아 인용을 금지하라. 정본 `CONSENSUS §3` 항목136(신설),
     `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#116, 신설).
     상세 `workspace/engine-port/results/longctx_conflict/
     {RESULT_P7_ITL_SIGMA_2026-09-10.md, probes/p7_905994/
     P7_RESULT.json}`.

117. ★★★**(2026-09-10, `longctx_conflict` 트랙, P7 `ITL_SIGMA`
     계측[job 905994], 메인 세션 직접 분석, GPU 0[분석 자체])
     표본별(예: seed별) 산포가 잡음처럼 보여도 그것이 처치-상관
     공변량(부하)일 수 있다 — 검정력·분산 입력으로 쓰기 전에
     회귀로 설명력을 먼저 확인하라.** P7의 `σ_seed`(`L_decode`
     기준 15–26%)는 처음엔 순수 도착 잡음처럼 보였으나, 시드별
     실현 도착 스팬 계수(0.770–1.154, 폭 1.50×)로 회귀하면
     `R²=0.95–0.99`가 설명되고 잔차가 1.6–4.2%로 붕괴했다 —
     지배 성분은 잡음이 아니라 **등록된 도착 실현의 변동**(부하
     그 자체)이었다. 실무 규칙: 표본 간(시드·반복 등) 산포가
     크게 나오면 그것을 즉시 "검정력을 갉아먹는 잡음"으로 쓰기
     전에, 그 산포를 설명할 수 있는 관측 가능한 공변량(여기선
     `span_hat`/실현 도착률)이 있는지 회귀로 확인하라 — 설명되면
     그 공변량을 검정력·분산 입력에서 분리하고 잔차만 잡음으로
     취급한다. ★**`RESULT_RULEPOWER_2026-09-08.md §7`·부록A와의
     교차참조**: 그 문서의 검정력 표는 `SIG_BOOT`(부팅×split
     랜덤효과, 실현 페어드 SD 7.1–24.3%)를 입력으로 쓰며
     "goodput 페어드 SD를 다른 캠페인에서 수입하지 말 것"
     (게이트#32)을 이미 명시했다 — P7의 `σ_seed`/`σ_boot`는 그
     `SIG_BOOT`와 **다른 양·다른 워크로드**이므로 수치 대입은
     여전히 금지되나(게이트#32), 이 항목의 방법론("원시 산포를
     검정력 입력으로 쓰기 전에 처치-상관 공변량으로 설명되는지
     회귀로 확인하라")은 `RESULT_RULEPOWER`류의 모든 향후 검정력
     재계산에 직접 적용된다. 정본 `CONSENSUS §3` 항목137(신설),
     `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#117, 신설).
     상세 `workspace/engine-port/results/longctx_conflict/
     {RESULT_P7_ITL_SIGMA_2026-09-10.md, RESULT_RULEPOWER_
     2026-09-08.md §7, probes/p7_905994/P7_RESULT.json}`.

118. ★★★**(2026-09-10, 연구 규율 변경 — 사용자 결정, doc-steward
     반영, GPU 0) 규칙층 적대 감사(사전등록 설계 심사)의 차단
     기준이 "자유 표면이 존재한다"에서 "그 자유도가 등록 판정을
     실제로 뒤집는가"(반전 시험)로 전환됐다.** 배경: `longctx_
     conflict` 트랙 계열 규칙층 감사가 **17연속 `NO-GO`**를 내는
     동안 통과한 유일한 설계는 결정 규칙을 0개로 만든 순수
     계측(Step 1·P7)이었다 — `handoff-report/session_handoff_
     2026-09-10.md` "열린 항목" #3이 이를 "감사 프로세스에 정지
     조건이 없다"로 진단, 사용자가 판정-반전 기준 전환을
     채택했다(선례: gate #13 rev3의 `GO-with-caveats`). 새 규칙:
     감사자는 지목한 자유 표면(문턱·격자·추정량·정의역·arm
     집합·동률 규약·가중)마다 그 자유도를 등록 허용 범위 끝까지
     밀어 등록 판정(라벨·부호·순위·분기)이 실제로 바뀌는지
     기존 원자료 또는 실행 전 예보로 수치와 함께 제시하는
     **반전 시험**을 수행해야 한다. 반전을 만들면 死因, 못
     만들면 死因이 아니라 **등록 caveat**(인용 금지 문장)로
     전환한다. 등급은 3단: `GO`(死因 0·caveat 0) /
     **`GO-with-caveats`**(死因 0·등록 caveat ≥1, 실행 승인,
     caveat는 결과 문서·정본이 그대로 승계할 인용 금지 문장으로
     문자 그대로 기술) / `NO-GO`(N1 estimand 부재/항등식 · N2
     반전 확인 · N3 예측 도달 불가[게이트#111·#112] · N4 제1원칙
     위반 중 하나 이상). ⚠️`GO-with-caveats`는 "주장 범위가
     좁아진다"이지 **"게이트를 닫았다"가 아니다**. 자기 적용
     규율도 명시됐다: 감사자의 처방도 실현가능성을 검사하고
     (게이트#113), 재감사 시 직전 지적을 등록 상수로 승격하지
     마라(게이트#110). **도구가 이미 고쳐졌다**: `.claude/
     agents/claims-auditor.md`에 "## 규칙층 감사 (사전등록 설계
     심사) — 등급과 차단 기준" 절 신설(63→109줄) — 이는 **교훈
     항목97**("게이트 등재돼도 도구가 안 고치면 재발", 게이트#14
     의 4번째 재발)의 직접 적용이다. **남는 위험(등재)**: 반전
     계산 자체가 새 자유 표면이 될 수 있다 — `PREREG_P6_
     ITL_ORDER` 판정서에서 실제로 자유 표면이 한 층 위(사전등록
     예측 자신·그 정의역)로 올라간 전례가 있다. 도구 절에 "반전
     계산에 쓴 원자료·격자·추정량을 명시하라"로 방어했으나
     **완전 해소는 아니다**. **이 변경은 제1원칙(정책 주장은
     반드시 서빙 실증)·기존 confound 카탈로그·인용 금지 목록·
     HE0를 전혀 건드리지 않는다.** ⚠️**기존 17연속 `NO-GO`
     판정들은 소급 재판정하지 않는다** — 그 死因들이 새 기준
     (N1–N4)에서도 `NO-GO`였을 것이라는 서술은 검증되지
     않았으므로 쓰지 않는다("소급 적용 없음"만 참). 대응
     `CONSENSUS §3` 항목138(신설). 상세 `.claude/agents/
     claims-auditor.md`, `handoff-report/session_handoff_
     2026-09-10.md` "열린 항목" #3.

119. ★★★**(2026-09-10, `longctx_conflict` Q-A 트랙, rev2 규칙층 감사, claims-auditor, GPU 0 — ★이 세션 최고 전이가치 ⓐ) 문턱을 판정층에서 지웠다고 자유 표면이 사라지는 게 아니다 — 아래(표본 자격)층으로 내려갈 수 있다.** Q-A rev2가 절대 문턱·예측·SLO 지시함수를 전부 걷어냈으나, 새로 들인 실행 상한 `λ≤0.95·s_min·μ_p`가 "어떤 셀을 측정할 자격이 있는가"를 판정하는 새 문턱이 됐다 — 게이트#107("문턱을 없앤 자리에 무엇이 들어왔는지 세라")의 **아래층 판본**. 실무 규칙: 판정 문턱을 제거했다면 그 자리에 들어온 실행/자격 규칙이 새 자유 표면인지 반드시 세어라. 정본 `CONSENSUS §3` 항목139(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#119, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rules_2026-09-10/VERDICT.md` §9-후보1.

120. ★★★**(2026-09-10, rev2, claims-auditor, GPU 0) 실행 자격 규칙의 입력에 저-`n` 난수량(도착 스팬 등)을 넣지 마라 — 부팅이 서로의 반복이 아니게 된다.** `span_factor(seed,N)`(첫 `N−1`개 지수난수 평균)는 `N=15`에서 상대 SD **26.7%**다. 이를 실행 자격(`λ≤0.95·s_min·μ_p`)에 넣으면 arm×rung 격자가 라운드마다 달라지고(`(5,4,4,2)/(4,4,3,0)/(5,4,4,1)/(4,4,3,1)`), `n_boot`가 칸마다 2/2/1/0으로 붕괴해 등록 검정력이 무의미해진다. 정본 `CONSENSUS §3` 항목140(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#120, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rules_2026-09-10/VERDICT.md` §9-후보2.

121. ★★★**(2026-09-10, rev2, claims-auditor, GPU 0) 포락선/`min`/argmin류 추정량은 "물리적 도달 가능"과 "관측된 정의역"을 하나의 정의로 등록하라.** 둘을 동시에("§1은 물리적 도달 가능, §2.4는 관측 `X` 범위") 등록하면 포락선이 실행 규칙의 함수가 되어 측정 가능한 양(1.52ms, 부팅 4/4 동부호)이 "0과 구별 불가"로 바뀔 수 있다(등록 분해능 7.2%의 95%를 이 아티팩트 하나가 삼킴). 정본 `CONSENSUS §3` 항목141(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#121, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rules_2026-09-10/VERDICT.md` §9-후보3.

122. ★★★**(2026-09-10, rev2+rev6, claims-auditor, GPU 0, 게이트#9 계열 재발[병합 항목]) 판정 규칙이 자기 정의 요소와 순환·상수공유 관계면 데이터와 무관하게 항상 같은 답을 내는 항등식이 될 수 있다.** 두 변종 관측: (a) `λ_cap := 0.95·s_min(N(λ_cap))·μ_p` — 자기 출력이 자기 인자로 들어가는 순환 정의, 고정점이 세 개(1.47× 확산)였다(rev2). (b) rev6의 `gap<4ms` 배제 게이트가 §1(a)의 모드-분리 제약과 **같은 상수 4ms**를 공유해 `gap≥4`가 정의상 항등식이 되어(47셀 전수 `<4` 0건) 절대 발화하지 못했다(rev6) — 처방된 수리 자신이 항등식이 된 사례. 실무 규칙: "결정론적이라 자유도 0"이라 쓸 때 (i) 그 함수의 인자에 자기 출력이 있는지 (ii) 문턱이 그 문턱의 대상을 만드는 규칙과 상수를 공유하는지 둘 다 확인하라. 정본 `CONSENSUS §3` 항목142(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#122, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rules_2026-09-10/VERDICT.md` §3-N2-c·§9-후보4, `.../audit_qa_rev6_2026-09-10/VERDICT.md` §0·§2-(6)·§8-후보R.

123. ★★★**(2026-09-10, rev3, claims-auditor, GPU 0) 정의역이 그 자체로 측정량이면, 실행 격자를 라운드-불변으로 고정해도 정의역은 고정되지 않는다.** rev3가 rev2의 死因(실행 자격=난수의 함수)을 제거해 λ 격자는 고정했으나, 포락선의 정의역(`X` 관측 범위)은 그 셀에 배정된 시드의 실현 도착 스팬을 포함하므로 여전히 라운드마다 움직였다(d44/d54 정의역 상단 0.402–0.534, 1.33× 산포) — rev2 死因의 16.8배 큰 반전(1.52→25.52ms)을 낳았다. 정본 `CONSENSUS §3` 항목143(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#123, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev3_2026-09-10/VERDICT.md` §4-N2-α·§9-후보1.

124. ★★★**(2026-09-10, rev3+rev4, claims-auditor, GPU 0[병합 항목]) `min`/argmin으로 여러 추정량을 "하나로 통일"해도, 그 답이 가리키는 기준 원소는 어떤 추정량을 썼는지의 함수로 남는다.** rev3: 같은 estimand에 두 추정량 정의(matched-λ 페어드 vs matched-`X` 포락선)를 동시 등록하면 나쁜 쪽(둘 중 위험이 큰 정의)이 실질 판정을 정한다. rev4: `min{itl95(D′):D′∈𝒜}`의 argmin 자체가 추정량의 함수였다 — 동일 셀에서 `pooled_p95`의 argmin은 d54(12/12), `median`의 argmin은 **d16(10/12)**. 실무 규칙: 항등식 검사를 점수함수가 아니라 결정 규칙(argmax/argmin) 위에서 하라는 기존 교훈의 기준-원소 판본. 정본 `CONSENSUS §3` 항목144(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#124, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev3_2026-09-10/VERDICT.md` §4-N2-β·§9-후보2, `.../audit_qa_rev4_2026-09-10/VERDICT.md` §4-N2-α·§11-후보A.

125. ★★★**(2026-09-10, rev3, claims-auditor, GPU 0) "인덱스"라고 쓸 때 그것이 무엇의 인덱스인지 문자로 고정하라.** rung 목록의 원소 수가 arm마다 다를 때(`i mod 3`의 `i`) "arm 자신의 rung 목록 인덱스"와 "공유 사다리 인덱스"라는 두 자연스러운 독해가 서로 다른 시드 배정을 주고, 그 결과 "arm 간 동일 — 자유도 0"이라던 등록 속성이 한쪽 독해에서 4/4 라운드 전부 거짓이었다. 정본 `CONSENSUS §3` 항목145(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#125, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev3_2026-09-10/VERDICT.md` §4-N2-γ·§9-후보3.

126. ★★★**(2026-09-10, rev3+rev4, claims-auditor, GPU 0[병합 항목]) 격자 기하(lever)나 상한을 명목 조작 변수(knob) 위에서 계산했다면 반드시 측정된(실현) 축 위에서 재계산해 검사하라.** rev3: 명목 `λ` 위의 lever(0.48, "정상 기하학")가 실현 `X` 위에서는 R1–R4가 +0.384/+0.985/**−0.185(비단조)**/+0.079로 흩어졌다. rev4: 명목 `0.95` 상한을 실현 스팬 계수로 나눈 유효 상한(`0.95/span_factor`)이 등록 6 rung 중 **4곳에서 이미 음수 여유**였다. 정본 `CONSENSUS §3` 항목146(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#126, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev3_2026-09-10/VERDICT.md` §1-V3·§9-후보4, `.../audit_qa_rev4_2026-09-10/VERDICT.md` §4-N2-β·§11-후보F.

127. ★★★**(2026-09-10, rev3+rev4, claims-auditor, GPU 0[병합 항목, 게이트#32 계열]) 재사용하는 상수의 출처(몇 arm에서, 어떤 `n`으로 측정됐는지)를 저장소에서 반드시 재확인하라.** rev3: "d92 정의역서 전 arm이 미처치 모드로 수렴"의 근거 `ITL_SOLO_S`가 **d44 한 arm만** 부팅한 상수였는데 "arm-무관 물리량"으로 두 세대에 걸쳐 인용됐다(원문서 자신이 "재확인 안 했음"을 기록). rev4: 저장소에 `n_boot>1`(P7) 측정이 이미 있는 상수를, 그와 다른 정의(`n_boot=1`)로 재계산한 값을 그대로 등록 상수로 승격했다(직전 판정서의 "median 1.4147"이 새 정의로는 5.1237이었음). 정본 `CONSENSUS §3` 항목147(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#127, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev3_2026-09-10/VERDICT.md` §3-(C)·§9-후보5, `.../audit_qa_rev4_2026-09-10/VERDICT.md` §3-(1)·§11-후보E.

128. ★★★**(2026-09-10, rev3, claims-auditor, GPU 0) 예산은 항목별로 검사하라 — 총액이 맞다고 항목이 정확한 건 아니다.** main bench 실현 예보가 등록보다 −8%, 포화 bench 실현 예보가 +160%로 반대 방향으로 어긋났으나 두 오차가 상쇄돼 총액(≈7.6 vs 등록 7.5)은 거의 일치했다 — 항목이 틀리면 노브를 자를 때 잘못된 항목을 자르게 된다. 정본 `CONSENSUS §3` 항목148(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#128, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev3_2026-09-10/VERDICT.md` §3-(B)·N9·§9-후보6.

129. ★★★**(2026-09-10, rev4, claims-auditor, GPU 0) argmin 산출 금지는 stake 보호이면서 동시에 진단 은폐 장치일 수 있다 — 금지하려면 "기준 원소 불변량 검사"를 반드시 별도로 산출해야 한다.** Q-A는 stake #1 보호를 위해 argmin arm 표를 산출 금지했는데, 그 금지 때문에 독자는 항목144(기준 원소가 추정량의 함수)의 발생 자체를 탐지할 수단이 없었다. 정본 `CONSENSUS §3` 항목149(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#129, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev4_2026-09-10/VERDICT.md` §11-후보B.

130. ★★★**(2026-09-10, rev4, claims-auditor, GPU 0 — ★이 세션 최고 전이가치 ⓑ) 자유 표면을 "동결"한 것을 "제거"라고 쓰지 마라 — 동결은 분산을 반복 축에 앨리어스된 편향으로 바꿀 뿐이다.** rev4가 rung당 시드를 1개로 동결하며 "시드 배정: 제거"라 등록했으나, 실제로는 시드 성분(σ_seed/σ_boot **3.30×**, d44)이 CI에서 사라졌을 뿐 estimand에서 사라진 게 아니다 — 그 rung의 4부팅 전부에 같은 시드 offset이 들어가 등록 `σ`가 구조적으로 그 편향을 볼 수 없게 된다. 정본 `CONSENSUS §3` 항목150(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#130, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev4_2026-09-10/VERDICT.md` §4-N2-β·§11-후보C.

★130 追記(2026-09-10, 5차 세션, rev8 판정서 §4-4, doc-steward): rev8/rev9의 A2(1시드 rung 등록 CI식 `√(σ̂_cell²/4+σ̂_req²/4)`)가 **같은 오류의 재발**임을 확인 — 4부팅이 **공통 trace**를 쓰는데 `σ̂_req²`을 `n=4`로 나눠 SE를 최대 2배 과소평가한다(A2 투영 3/4 ↔ 공통-trace 진단 1/4, 허가까지 최소 여유 0.25% < MC 오차 0.7%). 기전(반복해도 평균화되지 않는 공유/고정 성분을 표본 수로 나눠 SE를 인위적으로 줄임)이 본 항목과 동일하므로 별도 게이트 번호를 매기지 않고 이 항목에 追記한다(doc-steward 판단, CONSENSUS §3 "추기[2026-09-10, 5차 세션, rev8 판정서 §4-4]"와 대응). 상세 `.../longctx_conflict/audit_qa_rev8_2026-09-10/VERDICT.md` §4-4·§8-후보ε, `PREREG_QA_REGRET_REV9_2026-09-10.md` §2.

131. ★★★**(2026-09-10, rev4, claims-auditor, GPU 0, 게이트#32 계열) 분포가 이봉(혼합)이면 백분위·중앙값·절사평균은 서로 다른 절벽(모드 전환점)을 갖는다.** d16 셀의 미처치 가중 `u`(모드 비율)가 12시드에서 0.481–0.725로 0.50을 가로지르는데, `median`은 `u=0.5`에서, `trim10`은 `u=0.9`에서, `pooled_p95`는 `u=0.95`에서 모드를 갈아탄다 — "4추정량 전부 동부호"라는 확인은 모드 가중 `u`를 함께 보고하지 않으면 다른 부하로 이식 불가하다. 정본 `CONSENSUS §3` 항목151(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#131, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev4_2026-09-10/VERDICT.md` §4-N2-α·§11-후보D.

132. ★★★**(2026-09-10, rev4, claims-auditor, GPU 0) 사전등록 예보 표는 전수 재계산해서 실어라 — 부분 재계산은 표기 오류를 숨긴다.** 등록 예보 6개 중 1개(`span_factor(62,18)=1.1826`)가 실제로는 `span_factor(61,5)`의 값이었다(등록식으로는 `1.1063`, −6.45%) — 나머지 5개만 검산했다면 이 오차는 안 잡혔다. 정본 `CONSENSUS §3` 항목152(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#132, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev4_2026-09-10/VERDICT.md` §7-erratum1·§11-후보G.

133. ★★★**(2026-09-10, rev4, claims-auditor, GPU 0) 파생 요약이 정말 파생인지 검사하라 — 주 산출의 함수로 재계산해 값이 같은지 확인하라.** rev4 §1.1이 "처리율이 `ΔX` req/s 덜 든다"고 서술했으나 실제로 P7 36 bench 전부 `completed=90`(고정)이라 `X`는 순전히 `duration`의 재표현이었다 — `ΔX`가 독립 축이 아니라 이미 계상한 makespan 차의 개명이었다. 정본 `CONSENSUS §3` 항목153(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#133, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev4_2026-09-10/VERDICT.md` §8-인용금지8·§11-후보H.

134. ★★★**(2026-09-10, rev5, claims-auditor, GPU 0) 진단량을 도입할 때 그 정의 조건이 전 arm/전 조건에서 성립하는지 먼저 확인하라.** 모드-가중 진단 `u`는 분포에 골(valley)이 있어야 정의되는데, d92의 미처치·처치 모드 간격은 1.9ms로 등록 문턱 격자 폭(4ms)보다 좁아 **골이 없다** — d92는 애초에 이 진단량의 정의역 밖이었다. 정본 `CONSENSUS §3` 항목154(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#134, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev5_2026-09-10/VERDICT.md` §4-N2-α·§11-후보I.

135. ★★★**(2026-09-10, rev5, claims-auditor, GPU 0) 1차 정의와 "강건성용 병기 격자"가 서로 다른 답을 줄 수 있다 — 불일치 시 처분을 함께 등록하지 않으면 병기는 봉인이 아니라 자유도의 전시다.** `c_valley := 1.23×m̂_untreated`(유도 규칙)와 `{14,16,18}ms`(병기 격자)를 "둘 다 필수"로 등록했는데, 같은 셀에서 둘이 반대 라벨(not-pinned ↔ PINNED)을 줬다 — 병기가 오히려 새 자유도를 노출했다. 정본 `CONSENSUS §3` 항목155(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#135, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev5_2026-09-10/VERDICT.md` §1-W1·§11-후보J.

136. ★★★**(2026-09-10, rev5, claims-auditor, GPU 0) 1차 판정에 쓰이는 유도량은 반드시 산출 목록에 넣어라 — 없으면 판정이 등록 산출물로부터 재현 불가능하다.** 유도된 `c_valley`·`u` 값이 Q-A rev5의 산출표에 없어(산출 1은 "u(3 c_valley 값)"만 등재), 1차 라벨이 등록 산출물만으로 재현되지 않았다. 정본 `CONSENSUS §3` 항목156(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#136, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev5_2026-09-10/VERDICT.md` §4-N2-α·§11-후보K.

137. ★★★**(2026-09-10, rev5, claims-auditor, GPU 0) 라벨 문턱이 "잡으려던 현상"의 절벽 위치와 같은 축·같은 눈금인지 확인하라.** `u`의 mode-pinned 문턱(`0.90/0.10`)에 추정량 인자가 없어, 실제 절벽(median 0.50·trim10 0.90·pooled_p95 0.95)과 안 맞았다 — P7 36셀 중 발화 2셀뿐이었고, 갈림이 실제 일어나는 d16 12셀에는 0/12 발화했다(가장 안정한 값에만 걸리고 부호가 뒤집히는 값은 통과시킴). 정본 `CONSENSUS §3` 항목157(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#137, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev5_2026-09-10/VERDICT.md` §4-N2-β·§11-후보L.

138. ★★★**(2026-09-10, rev5, claims-auditor, GPU 0, 게이트#117의 estimand별 재검증 사례) 분산 성분이 부하임(게이트#117)은 estimand마다 다시 검증해야 한다 — 원자료 estimand에서 성립해도 페어드 차 estimand에서는 성립하지 않을 수 있다.** P7 `L_decode`(원자료)에서는 실현 도착 스팬이 `R²=0.95–0.99`를 설명했으나, 페어드 `Δ`(같은 시드·같은 λ에서 arm 간 차)에서는 `R²`가 **0.116/0.130**으로 붕괴했다 — 페어링이 이미 부하 성분 대부분을 상쇄하기 때문이다. 정본 `CONSENSUS §3` 항목158(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#138, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev5_2026-09-10/VERDICT.md` §3-(1)·§11-후보M.

139. ★★★**(2026-09-10, rev5, claims-auditor, GPU 0) 검정력을 표본 수 `n`으로 사려면 그 `n`이 만드는 자유도(df)와 임계값도 함께 계산하라.** 부팅 4×시드 2를 합친 "n=8" 설계는 `F₀.₀₅(3,4)=6.591`을 요구해, `n=4`(부팅만)의 `F₀.₀₅(3,8)=4.066`보다 **더 엄격한(더 약한) 검정**이 됐다 — n이 늘었는데 검정력은 개선되지 않을 수 있다. 정본 `CONSENSUS §3` 항목159(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#139, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev5_2026-09-10/VERDICT.md` §8-인용금지9·§11-후보N.

140. ★★★**(2026-09-10, rev5, claims-auditor, GPU 0) 분산 성분 추정치가 단일 표본(셀/시드)에 지배되는지 확인하라.** `σ_seed(d44)`의 크기는 12셀 중 단 1개 시드쌍(`r1s43`)이 지배했다(그 시드 관여 3쌍만 |ΔΔ| 0.55–0.65, 나머지 9쌍은 ≤0.22) — 분산성분 하나가 사실상 이상치 하나의 재서술일 수 있다. 정본 `CONSENSUS §3` 항목160(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#140, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev5_2026-09-10/VERDICT.md` §3-(1)·§11-후보O.

141. ★★★**(2026-09-10, rev5, claims-auditor, GPU 0) 격자 충실도(밀도)를 높이는 수리가 저부하/외삽 노출을 늘릴 수 있다 — 수리의 이득과 부작용을 함께 계산하라.** 시드 규칙(진짜 수리)이 격자를 라운드 불변으로 만든 부작용으로, 저장소 최저 관측 `X` 아래에서 측정되는 셀 비율이 rev4의 56%(9/16)에서 rev5의 **95%(18/19)**로 늘었다 — 가치이자 한계이므로 둘 다 등재해야 한다. 정본 `CONSENSUS §3` 항목161(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#141, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev5_2026-09-10/VERDICT.md` §9-S5·§11-후보P.

142. ★★★**(2026-09-10, rev5+rev6, claims-auditor, GPU 0[병합 항목, 2회 관측 재발]) 직전 판정서가 caveat로 등재한 항목이 다음 판본에서 누락될 수 있다 — caveat 승계는 번호 붙은 목록으로 관리하라.** rev4 판정서가 등재한 caveat("d92 칸은 `ρ` 격차 4.754×의 재진술에 가깝다")가 rev5 판정서 목록에서 누락됐다가(1차 발화), rev5 자신이 등재한 caveat("`|𝒜|`이 처치 강도의 감소함수")가 다시 rev6 목록에서 누락됐다(2차 발화, 같은 유형 재발) — 게이트#18/#115와 구분: 이건 결함 진단 손실이 아니라 **스코프 제한(caveat)의 승계 실패**다. 정본 `CONSENSUS §3` 항목162(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#142, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev5_2026-09-10/VERDICT.md` §11-후보Q, `.../audit_qa_rev6_2026-09-10/VERDICT.md` §3-N3·§8-후보Y.

143. ★★★**(2026-09-10, rev6, claims-auditor, GPU 0 — ★이 세션 최고 전이가치 ⓒ) 감사 처방의 자기검사가 과잉일 수 있다 — 값을 하나로 고정(결정론적 함수화)하면 그 자유 표면은 사라지므로 별도 봉인(강건성 검사)이 필요 없다.** rev5 판정서가 처방한 "히스토그램 빈 폭 1·4ms에서 `gap` 순위 불변 검사"(자기검사)를 rev6이 문자 그대로 이행했으나, 애초에 빈 폭이 **2ms 단일값으로 고정**된 설계에서는 그 표면이 존재하지 않았다 — 봉인이 (i) 판정 구동량이 아닌 잘못된 양을 재고 (ii) 미등록 동률 규약을 새로 만들며 (iii) 처분이 0줄이었다. 게이트#113(처방의 실현가능성 검사)의 대칭형: 처방자는 자기 처방이 **불필요해지는 경우**도 검사해야 한다. 정본 `CONSENSUS §3` 항목163(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#143, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev6_2026-09-10/VERDICT.md` §2-(6)·§8-후보S.

144. ★★★**(2026-09-10, rev6, claims-auditor, GPU 0) 강건성 검사의 대상량이 실제 판정 구동량과 같은지 확인하라 — 다른 양을 검사하면 검사가 통과해도 판정은 여전히 흔들릴 수 있다.** 항목163의 `gap` 순위 검사는 `gap`(모드 간격)의 순위를 쟀지만, 라벨을 실제로 움직이는 양은 `c_valley`였다(예: `gap` 5/6/4인데 `c_valley`는 16.0으로 고정 — 검사 대상과 판정 구동량이 독립적으로 움직임). 정본 `CONSENSUS §3` 항목164(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#144, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev6_2026-09-10/VERDICT.md` §2-(6)·§8-후보T.

145. ★★★**(2026-09-10, rev6, claims-auditor, GPU 0) 순위 불변 검사에는 반드시 동률(tie) 규약을 함께 등록하라.** 저장소 47셀·1,081쌍에서 강한 역전 8쌍, 동률↔엄격 전환 127쌍(11.7%)이 나와, 동률 규약을 어떻게 정하느냐에 따라 "불안정한 셀" 수가 6/47과 46/47 사이를 오갔다 — 격자 양자화가 있는 곳에서는 동률이 지배적 현상일 수 있다. 정본 `CONSENSUS §3` 항목165(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#145, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev6_2026-09-10/VERDICT.md` §1-C·§8-후보U.

146. ★★★**(2026-09-10, rev6, claims-auditor, GPU 0) 진단량이 "발화하는 사례"를 예보로 제시할 때 그 사례가 실제로 등록된 설계점(격자 위의 점)인지 확인하라.** `sign_agree=4`(네 추정량 전부 부호 일치)가 실제로 도달하는 유일한 예시는 `N=90`이었는데, `N=90`은 Q-A의 등록 rung이 아니었다(`N>64`인 유일한 등록 rung은 λ=0.70이며 그 rung은 단일 arm이라 애초에 `Δ`가 없다) — 이 캠페인은 어느 등록 칸에서도 부호 진술을 못 할 수 있다. 정본 `CONSENSUS §3` 항목166(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#146, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev6_2026-09-10/VERDICT.md` §2-(2)·§8-후보V.

147. ★★★**(2026-09-10, rev6, claims-auditor, GPU 0) 검정력 수치는 결정 규칙이 실제로 구속되는 항(최악 추정량/최악 조건)으로 보고하라 — 최선 추정량 기준 수치는 전체 분해능을 과소평가하게 오인시킨다.** 등록 규칙(§3)이 4추정량 전부를 요구하는데, 검정력 표(§6)는 `pooled_p95`(최선) 기준 반폭만 실었다 — 실제 구속 항인 `trim10`은 그보다 5.9× 크고(효과 대비 여유 3.2%), `median`은 반폭 자체가 정의상 더 크다. 정본 `CONSENSUS §3` 항목167(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#147, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev6_2026-09-10/VERDICT.md` §2-(3)·§8-후보W.

148. ★★★**(2026-09-10, rev6, claims-auditor, GPU 0) arm/조건 태그가 파일명에 없는 프로브는 "전수(exhaustive)" 조사에서 조용히 빠질 수 있다 — 태그 스킴 자체를 감사하라.** `probes/p2_905712`(실제로는 `PDMUX_R2_FIXED_DSM=44` 구성)의 파일명에 arm 태그가 없어, "저장소는 X≥0.4186에서만 측정했다"는 전수 조사가 이 프로브의 `X=0.3257`(ctx16384) 표본을 놓쳤다. 정본 `CONSENSUS §3` 항목168(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#148, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev6_2026-09-10/VERDICT.md` §2-(4)·§8-후보X.

149. ★★★**(2026-09-10, rev8, claims-auditor, GPU 0 — ★이 세션 최고 전이가치 ⓐ) 추정량의 이름을 등록하는 것은 추정량을 등록하는 것이 아니다 — 어느 배열의 범함수인지가 판정을 뒤집는다.** rev5 §6이 "4추정량(`pooled_p95`·`median`·`mean`·`trim10`)"이라고만 쓰고 어느 배열(축) 위에서 계산하는지 등록하지 않았다 — 저장소 기존 도구 `p7_analyze.itl95_estimators`는 혼합축(`pooled_p95`만 토큰축, 나머지는 요청별 `itl95`축)이었다. 같은 `Δ(d16,d54)` 셀에서 `median`이 축 P(등록, 풀링 토큰)에서는 **−0.2165**, 축 R(요청별)에서는 **+23.0220**로 **부호까지** 다르고, 캠페인의 유일한 등록 허가(`sign_agree=4`)가 축 R에서는 3쌍×3 N 전부 발화(9/9칸), 축 P에서는 0회(0/9칸) 발화했다. 실무 규칙: 추정량을 등록할 때 그 이름만이 아니라 "어느 배열의 원소에 적용하는 함수인가"를 문자로 못박아라. 정본 `CONSENSUS §3` 항목169(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#149, 신설). 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_rev8_2026-09-10/VERDICT.md` §0·§1-표A·§8-후보α.

150. ★★★**(2026-09-10, rev8, claims-auditor, GPU 0 — ★이 세션 최고 전이가치 ⓑ) 사전등록이 자기 예보 수치를 적어 두면 그 수치가 미등록 정의를 사후에 특정한다 — 결함이 아니라 방어 자산이다.** rev6 §3이 **제출 전에** 적어 둔 예보 12값이 축 P(rev8이 사후 등록한 정의)에서만 소수 4자리까지 정확히 재현되고, 축 R에서는 재현되지 않는다(`median` `Δ(d16,d54)`가 부호까지 반대) — 즉 사전등록 문서 자신이 뒤늦게 문면화될 정의를 이미 수치로 못박아 두고 있었다. 이 사실이 A1("사후 선택 아님")의 두 번째이자 가장 강한 근거였다. 실무 규칙: 미등록 자유도를 뒤늦게 문면화할 때, 그 정의가 기존에 적어 둔 예보·산출 수치와 정확히 일치하는지 대조하라 — 일치하면 사후 선택이 아니라 문면 결함의 정정이라는 독립 증거다. 정본 `CONSENSUS §3` 항목170(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#150, 신설). 상세 `.../audit_qa_rev8_2026-09-10/VERDICT.md` §0·§4-1(ii)·§8-후보β.

151. ★★★**(2026-09-10, rev8, claims-auditor, GPU 0 — ★이 세션 최고 전이가치 ⓒ) 분석 정의를 실행 중에 고쳐야 한다면, 그 수정이 "허가를 여는 쪽"인지 "닫는 쪽"인지를 수치로 먼저 보고하라.** rev8의 A1(축 P 등록)은 캠페인의 유일한 등록 허가(`sign_agree=4`)를 축 R 전면 발화(9/9칸)에서 축 P 최대 3/4(0/9칸)로 **닫는 쪽**을 골랐다 — 이것이 "자기 캠페인에 유리하게 사후 선택했다"는 의심에 대한 세 번째 독립 방어(이해상충 방향이 결론과 반대)였다. 실무 규칙: 실행 중(캠페인 진행 중) 분석 정의를 확정할 때는, 그 정의가 데이터 수집 계획과 무관함을 밝히는 것에 더해 **그 정의가 자기 결론에 유리한 방향인지 불리한 방향인지를 수치로 먼저 제시**하라 — 닫는 쪽(불리한 방향)이면 이해상충 방어가 성립하고, 여는 쪽(유리한 방향)이면 그 자체로 死因 후보가 된다. 정본 `CONSENSUS §3` 항목171(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#151, 신설). 상세 `.../audit_qa_rev8_2026-09-10/VERDICT.md` §0·§4-1(iii)·§8-후보γ.

152. ★★**(2026-09-10, rev9, claims-auditor 계산[rev8 §4-4]+메인 세션 등재, GPU 0) 정렬을 단언하는 괄호는 검사가 아니다 — 위반 시 처분과 함께 등록하라.** rev6 §1(a)가 괄호로 적어 둔 `m_low<m_high`(저모드가 미처치 바닥) 가정이 저장소 47셀 중 **33셀(70.2%)에서 최대 빈이 高모드**라 문자 그대로는 거짓이었다 — 값으로 정렬(swap)하는 현재 구현이 이를 해소하지만, 그 사실 자체를 등록하지 않으면 "정렬이 보장돼 있다"는 오해가 재발한다. 실무 규칙: 코드/문서가 괄호로 단언하는 정렬·순서 조건은 사전등록 시점에 저장소 원자료로 위반율을 계산해 함께 등록하라 — 단언만으로는 검사를 대체하지 못한다. 정본 `CONSENSUS §3` 항목172(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#152, 신설). 상세 `PREREG_QA_REGRET_REV9_2026-09-10.md` §1(c), `.../audit_qa_rev8_2026-09-10/VERDICT.md` §1-표C·§4-6(c).

153. ★★**(2026-09-10, rev8 erratum 1, claims-auditor 자기 정정, GPU 0) "보수적이니 유지"를 주장할 땐 보수성 배율을 판정 단위(SE 기준)로 계산하라 — 반폭÷SE는 `t`를 곱한 값이다.** rev8 §A3이 "등록형 반폭이 `σ_req`-free의 3.5–3.9배"라 썼으나 이는 **반폭÷SE의 단위 오류**(`t₃`가 곱해진 값)였다 — 옳은 SE 기준 배율은 1.06–1.22×(8개 중 7개), `median`@`Δ(d44,d54)`만 4.33×다. 실무 규칙: 두 SE(또는 두 CI 반폭)의 비율을 "보수성 배율"로 인용할 때, 그 비율이 SE 대 SE인지 CI 반폭 대 CI 반폭인지(후자는 자유도의 임계값이 곱해져 있음)를 명시하고, 판정에 실제로 쓰이는 단위(SE)로 환산해 보고하라. 정본 `CONSENSUS §3` 항목173(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#153, 신설). 상세 `PREREG_QA_REGRET_REV9_2026-09-10.md` §4-erratum1, `.../audit_qa_rev8_2026-09-10/VERDICT.md` §5-1·§4-4.

154. ★★**(2026-09-10, rev9 P3, claims-auditor 계산+메인 세션 등재, GPU 0) 결정 마진이 부트스트랩 Monte Carlo 오차보다 작으면 그 칸의 판정은 seed의 함수다.** 1시드 rung(λ=0.42) `Δ(d44,d54)`에서 등록식 대 공통-trace 진단의 `sign_agree` 허가까지 최소 여유가 `trim10` 반폭 0.442 대 효과 0.4409 = **0.25%**로, 부트스트랩 재실행 시의 MC 오차(`≈1/√(2B)≈0.7%`, `B=10000`)보다 **작다** — 즉 이 칸의 등록 판정은 부트스트랩 seed를 바꾸면 뒤집힐 수 있다. 실무 규칙: 등록 판정이 임계값에 얼마나 가까운지(마진)를 그 판정을 산출하는 절차 자체의 재현 오차(MC 오차 등)와 비교해, 마진이 그 오차보다 작으면 "판정이 확정적"이라고 쓰지 마라 — seed 의존성을 명시하라. 정본 `CONSENSUS §3` 항목174(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#154, 신설). 상세 `PREREG_QA_REGRET_REV9_2026-09-10.md` §2, `.../audit_qa_rev8_2026-09-10/VERDICT.md` §1-표G·H·§4-4.

155. ★★**(2026-09-10, rev9 P4, doc-steward 등재, GPU 0) "분석기가 등록 격자와 일치한다"는 등록 산출 전부가 구현됐다는 뜻이 아니다.** rev8까지 `registration_conformance`(격자·시드표·`N` 일치, 파일명 파서 mismatch 0)가 100% 통과했음에도, 등록 산출 9개 중 산출 7(런타임 매니페스트·`server_info` 동일성 보고)은 **미구현 상태로 남아 있었다** — rev9 P4가 뒤늦게 구현했다. 실무 규칙: 격자/시드/표본수 정합성 검사(conformance check)의 통과를 "사전등록이 요구한 산출물이 전부 나왔다"로 확대 해석하지 마라 — 산출물 목록을 별도로 대조하라. 정본 `CONSENSUS §3` 항목175(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#155, 신설). 상세 `PREREG_QA_REGRET_REV9_2026-09-10.md` §3, `.../audit_qa_rev8_2026-09-10/VERDICT.md` §4-6(e).

156. ★**(2026-09-10, rev8 §6, claims-auditor, GPU 0) 실행 중 관측을 사전등록에 적을 때의 봉인은 "n=1·판정 금지"가 아니라 "그 관측이 문제의 선택과 직교함을 수치로 보이는 것"이다.** rev8 §6이 라운드 1 첫 셀의 실행 중 관측(`u` 정의 불가·`pooled_p95` 요청-표집 반폭 108.2%)을 "n=1이므로 설계·성능 판정에 인용 금지"로만 봉인했으나, 감사(표 J)는 그것으로 충분하지 않다고 판정했다 — 실제 방어는 그 두 관측이 **축 선택(A1)과 구별 정보 0**임(둘 다 두 축에서 구성상 동일하거나 제3의 축의 함수)을 수치로 보인 것이었다. 실무 규칙: 사전등록 실행 중 관측을 기재할 때 "표본 크기가 작다"는 caveat만으로 오염 가능성을 봉인했다고 쓰지 말고, 그 관측이 문제되는 선택(설계 자유도)과 통계적으로 독립임을 별도로 논증하라. 정본 `CONSENSUS §3` 항목176(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#156, 신설). 상세 `.../audit_qa_rev8_2026-09-10/VERDICT.md` §1-표J·§4-5.

157. ★★★**(2026-09-10, 5차 세션, doc-steward, GPU 0 — 연구 도구 인프라 사실) 게이트의 "작동하는 절반"이 버전관리 밖에 있으면 그 게이트는 드리프트 탐지·재현이 불가능하다.** 게이트#118(규칙층 감사 등급 개편)의 실제 이행분은 `/scratch/ehmoon/whlee/.claude/agents/claims-auditor.md`(63→109줄, SHA-256 `47e800cf6b663752730c492137f06cab7ede58d29c38dea6f55211656835ae02` — **2026-09-11 doc-steward 追記**: 경로 1줄 정정[존재하지 않는 `workspace/engine-port/reports/CONSENSUS.md` → `reports/CONSENSUS.md`], 규칙 내용 불변. 새 SHA-256 = `e7c2dedfe53527a84a0990c50607135d9ba4200d85a151f919aa7f81e7ed93b8`)에 있는데, 그 파일이 속한 워크스페이스 루트(`/scratch/ehmoon/whlee`)는 **git 저장소가 아니다**(`.git`이 빈 디렉터리, `git status` → `fatal: not a git repository`) — 이 프로젝트(`prefill-layer-alloc`)의 오늘 커밋 `d869751`·`e2c2ba1`(둘 다 "규칙층 감사 규율 개편(게이트 #118)"을 언급)에 **이 파일 자체는 들어갈 수 없었다**(별도 워크스페이스). 이것은 **교훈 항목97**("게이트 등재돼도 도구가 안 고치면 재발")의 **뒷면**이다 — 이번엔 도구가 실제로 고쳐졌는데 그 수정이 기록(버전관리)에 없다. 실무 규칙: 감사·게이트 도구 파일을 프로젝트 정본에서 참조할 때는 그 파일이 실제로 버전관리 대상인지 확인하고, 아니라면 (i) 정본에 경로+체크섬을 박아 드리프트를 탐지 가능하게 하고 (ii) 근본 해법(그 파일을 대상 저장소로 편입할지)을 사용자에게 명시적으로 위임하라. 정본 `CONSENSUS §3` 항목177(신설), `PROJECT_STATUS.md` "방법론 게이트" 이 항목(#157, 신설). 상세 `PROJECT_STATUS.md` 최상단 배너(2026-09-10, 5차 세션). ★追記(2026-09-11, doc-steward, GPU 0 — 도구 사실 추가 사례): 같은 형태의 결함이 다른 도구 파일에도 있었다 — `.claude/agents/git-committer.md:25-27`이 outer 워크스페이스를 "별도 `.git`, 원격 있음"인 저장소로 서술했으나(engine-porter, 2026-09-11 확인) 실제로는 위와 같은 빈 `.git`이었다(doc-steward 재확인). 당시 표시만 해 두고 수정하지 않았다. ★★追記(2026-09-12(2), doc-steward, GPU 0 — 게이트 완료, 커밋 `4027257`): 이 게이트가 지적한 문제 자체가 해소됐다 — `.claude/agents/*.md`(7)·`.claude/skills/*/SKILL.md`(3)·루트 `CLAUDE.md`, 총 11개 추적 사본을 inner repo `tools/claude/`로 이관해 버전관리 안에 넣었다(동기화·검증 스크립트 `sync_claude_tools.sh` + 해시 매니페스트 + 변이 테스트 10케이스 전부 통과, 심볼릭 링크는 기각). `git-committer.md`의 "outer는 별도 repo" 서술도 정정했다(위 2026-09-11 추기 사례 해소). 이 완료로 도구 파일 3개(`claims-auditor.md`·`git-committer.md`·`CLAUDE.md`)가 이제 커밋으로 추적된다 — 앞으로 이 파일들을 인용할 때는 개별 SHA-256 대신 `tools/claude/claude_tools.manifest.sha256` + 커밋 해시를 인용한다. 상세 `PROJECT_STATUS.md` 최상단 배너(2026-09-12(2)) A항목, `tools/claude/README.md`.

, 2026-09-11 (Q-A `REGRET` 캠페인[jobs 906504–906507] 결과 문서에 대한
claims-auditor 결과 감사[`audit_qa_result_2026-09-11/VERDICT.md`], GPU 0)
**#158**(G-a) 신설[절(§)을 폐기할 때 그 절을 입력으로 쓰는 산출 목록
항목이 고아로 남을 수 있다 — 폐기는 산출 목록까지 전파 확인해야 한다.
`PREREG_QA_REGRET_2026-09-10.md`(rev5) §0이 rev2 §2.4(포락선 정의)를
폐기했는데 rev5 §6의 산출 8("내부 rung LOO 자기진단")은 바로 그 폐기된
§2.4를 보간 기저로 요구하는 산출이었고, 이 불일치가 이후 감사 3회를
통과했다 — 교훈 항목89("삭제는 조건 등록이 아니라서 grep에 안 잡힌다")
의 산출-목록 판본, CONSENSUS §3 항목178과 대응]·**#159**(G-b) 신설
[결과 문서의 요약 표가 본문 절과 다른 수를 실을 수 있다 — 요약 표는
본문에서 기계적으로 파생시키거나 교차 대조 자기점검을 넣어라.
`RESULT_QA_REGRET_2026-09-11.md` §15 #1이 σ_req-free `Δ(d44,d54)`를
"6칸"으로 요약했으나 §6.3 본문(옳음)은 8칸을 실었다(D1), CONSENSUS §3
항목179와 대응]·**#160**(G-c) 신설[인용 금지를 문자 그대로 승계해
놓고 같은 문서의 다른 절이 그 금지 대상을 "참인 주장"으로 재긍정할 수
있다 — 승계 목록과 본문 주장의 술어 대조가 필요하다(게이트#9의 18번째
재발). §16이 rev8 6-7("`gap<4ms` 0건은 항등식")을 문자 승계해 두고도
§15 #5가 같은 항등식을 "작동하는 참인 주장"으로 서술했다(D3), CONSENSUS
§3 항목180과 대응]·**#161** 신설[메인 세션 발의, doc-steward 등재 —
요약에 쓴 통계량의 출처 표본이 요약이 주장하는 전체 표본이 아니라
부분(예: 라운드 1개)일 수 있다. `1.958×`가 §13.2의 네 독법 어디서도
재현되지 않았고(D2) 실제로는 4라운드 중 라운드 1 단독 값이었다 —
"4 라운드 전부 재현"이라는 서술은 거짓이었다. #159(G-b)와 인접하나
판정 대상이 다르다(개수 불일치 vs 출처 표본 불일치)로 별개 유지,
CONSENSUS §3 항목181과 대응]. 이 4건은 기존 #119–157과 전수 대조해
중복 없음을 확인했다(#158은 교훈89의 산출-목록 판본이라 신규, #160은
게이트#9의 재발 사례이나 "인용금지 문자 승계 후 같은 문서 재긍정"이라는
특정 메커니즘은 기존 #9 재발 사례들[아이덴티티 검사·양성대조 등]과
달라 별도 등재). 이 감사는 read-only·GPU 0이며 캠페인 결과 자체(jobs
906504–906507, 4/4 `COMPLETED`, 10.641 GPU-h)에 대한 새 성능 판정을
0건 추가한다 — 자세한 내용은 위 "다음 실험 gate" longctx_conflict 행
2026-09-11 追記 및 `CONSENSUS.md` §3 항목178–181 참조.

★★**전파 규칙 고정(2026-09-11, doc-steward, GPU 0 — #118–157 마스터 절 누락 발견·백필 후속)**: 게이트는 이 문서 안에서 세 곳(최상단 헤더 블록의 해당 세션 서술·"다음 실험 gate" 표의 해당 행·본 "방법론 게이트" 절의 번호매김 열거)에 중복 서술될 수 있고, 세 곳을 독립적으로 갱신하면 드리프트한다 — **본 절의 번호매김 열거만이 정본**이다. 신규 게이트를 신설할 때는 세 곳 전부를 동시 갱신하고, 세 곳이 불일치하면 본 절의 번호매김 열거를 따르라. 상세·근거는 #162.

158. ★★★**(2026-09-11, claims-auditor 결과 감사[`audit_qa_result_2026-09-11/VERDICT.md`], GPU 0) 절(§)을 폐기할 때 그 절을 입력으로 쓰는 산출 목록 항목이 고아로 남을 수 있다 — 폐기는 산출 목록까지 전파 확인해야 한다.** `PREREG_QA_REGRET_2026-09-10.md`(rev5) §0이 rev2 §2.4(포락선 정의)를 명시적으로 폐기했는데, rev5 §6의 산출 목록 8행("내부 rung LOO 자기진단")은 바로 그 폐기된 §2.4를 보간 기저로 요구하는 산출이었다 — 이 불일치가 그 뒤 감사 3회(rev6·rev8·이 결과 감사 이전)를 통과했다. 교훈89("삭제는 조건 등록이 아니라서 grep에 안 잡힌다")의 산출-목록 판본. 대응 `CONSENSUS.md` §3 항목178. 상세 `workspace/engine-port/results/longctx_conflict/audit_qa_result_2026-09-11/VERDICT.md` §6·§11(G-a).

159. ★★★**(2026-09-11, claims-auditor 결과 감사, GPU 0) 결과 문서의 요약 표가 본문 절과 다른 수를 실을 수 있다 — 요약 표는 본문에서 기계적으로 파생시키거나 교차 대조 자기점검을 넣어라.** `RESULT_QA_REGRET_2026-09-11.md` §15 #1이 σ_req-free `Δ(d44,d54)`를 "6칸"으로 요약했으나 §6.3 본문(옳음)은 **8칸**[1·1·2·1·2·2·3·3]을 실었다(D1). 대응 `CONSENSUS.md` §3 항목179. 상세 `.../audit_qa_result_2026-09-11/VERDICT.md` §7(D1)·§11(G-b). ★追記(2026-09-11(2), Q-B′ 감사 G-ε 사례): 같은 형태가 재발했다 — `PREREG_QB_LOOPGAIN_2026-09-11.md` 초안 §3.6 표(ΔE/ρ/U 회귀)는 계산기에도 JSON에도 없었고, 그 표 안에서도 ΔE 행은 풀링 gap 정의로·ρ 행은 셀평균 gap 정의로 따로만 재현돼 정의가 섞여 있었다(초안 D4). 상세 `audit_qb_rules_2026-09-11/VERDICT.md` §3(a) D4.

160. ★★★**(2026-09-11, claims-auditor 결과 감사, GPU 0) 인용 금지를 문자 그대로 승계해 놓고 같은 문서의 다른 절이 그 금지 대상을 "참인 주장"으로 재긍정할 수 있다 — 게이트#9의 18번째 재발.** `RESULT_QA_REGRET_2026-09-11.md` §16이 rev8 6-7("`gap<4ms` 0건은 항등식")을 문자 승계해 두고도 §15 #5가 같은 항등식을 "작동하는 참인 주장"으로 서술해 자기모순을 냈다(D3). 대응 `CONSENSUS.md` §3 항목180. 상세 `.../audit_qa_result_2026-09-11/VERDICT.md` §7(D3)·§11(G-c). ★追記(2026-09-11(2), Q-B′ 감사 G-δ 사례): 코드의 금지 산출 선언도 실제 출력과 어긋날 수 있다 — `qb_forecast.py`:482의 `does_not_produce`가 `Δ` 부호 진술을 금지했는데도 같은 파일 :624–630의 `C_sign_pattern`이 `delta_mean`을 산출했다(초안 D1). 상세 `audit_qb_rules_2026-09-11/VERDICT.md` §3(a) D1.

161. ★★**(2026-09-11, 메인 세션 발의 + doc-steward 등재, GPU 0) 요약에 쓴 통계량의 출처 표본이 요약이 주장하는 전체 표본이 아니라 부분(예: 라운드 1개)일 수 있다.** `RESULT_QA_REGRET` §13.2가 인용한 `1.958×`는 네 독법 중 어디에서도 재현되지 않았고(D2), 실제로는 4라운드 중 라운드 1 단독 값이었다. #159(G-b)와 인접하나 판정 대상이 다르다(개수 불일치 vs 출처 표본 불일치). 대응 `CONSENSUS.md` §3 항목181. 상세 `.../RESULT_QA_REGRET_2026-09-11.md` §13.2, `.../audit_qa_result_2026-09-11/VERDICT.md` §7(D2).

162. ★★★**(2026-09-11, doc-steward, GPU 0 — 정본 관리 절차 사실) 같은 게이트 항목이 저장소 정본 문서 한 곳 안에서도 서로 다른 세 위치(헤더 블록 서술·"다음 실험 gate" 표 행·본 절 번호매김 열거)에 표현될 수 있고, 그중 하나만 갱신되면 나머지가 정본과 어긋난다 — 번호매김 열거만이 유일한 정본이어야 한다.** 2026-09-10 이전 세션들이 #118–157(CONSENSUS §3 항목139–177 대응, 40건)을 헤더 블록·"다음 실험 gate" 표(예: longctx_conflict 행)에서는 계속 인용해 두면서도 본 절의 번호매김 열거(1…118 다음 158…161)에는 한 번도 추가하지 않아, 오늘 이전까지 다음 세션의 catch-up이 게이트 40건을 통째로 놓칠 뻔한 상태였다(doc-steward가 오늘 발견·CONSENSUS §3 항목139–177 원문으로 백필해 해소, 새 해석·새 판정 추가 0건). 형태상 게이트#14의 追記("이미 저장소 안에 같은 진단이 두 번 독립으로 존재 — 정본 라이브러리에 반영 안 됨은 도구 규율 실패", 4번째 재발 "등재만으로는 전파되지 않았다")·게이트#97(도구가 안 고치면 재발)·게이트#115(에이전트 반환문을 파일로 옮기는 전사 단계의 절 단위 데이터 손실)와 인접하나, 이 항목이 짚는 실패 지점은 그것들과 다르다 — 진단 자체의 소실(#115)도 도구 미수리(#97)도 아니라, **정본 문서 자신의 아키텍처가 같은 사실을 세 곳에 중복 표현하도록 굳어 있었고 그중 무엇이 canonical인지 이 세션 이전까지 한 번도 명시적으로 선언된 적이 없었다**는 것이다. 실무 규칙: 신규 게이트를 신설할 때는 헤더 블록·"다음 실험 gate" 표·본 절 번호매김 열거 세 곳 전부를 동시 갱신하고, 세 곳이 불일치하는 것이 발견되면 본 절의 번호매김 열거를 정본으로 취급해 나머지를 그에 맞춘다. ⚠️정본 §3 항목 신설 없음(doc-steward 판단 — 이 작업은 `CONSENSUS.md`의 신규 분석 결론이 아니라 `PROJECT_STATUS.md` 내부 문서 계층 간 전파 수리이므로 CONSENSUS rev를 올리지 않는다; 필요 시 다음 CONSENSUS.md 개정에서 항목182 후보로 재평가). 상세: 이 문서 자신(헤더 블록 2026-09-10/09-11 서술·"다음 실험 gate" longctx_conflict 행·본 절 #118 이후 번호매김 열거), 원 소재 `reports/CONSENSUS.md` §3 항목139–177. ★追記(2026-09-11(2), doc-steward): 위 "항목182 후보로 재평가"가 이번 rev65(Q-B′ 등재와 동시에 실행된 CONSENSUS.md 개정)에서 이행됐다 — `CONSENSUS.md` §3 항목182로 등재 완료.

163. ★★★**(2026-09-11, claims-auditor 규칙층+결과 감사[Q-B′], GPU 0) 분해 성분을 그 성분의 대수적 주항으로 회귀한 R²는 "정보 내용"의 증거가 아니라 항등식 잔차의 크기다 — 게이트#9 계열 19번째 재발.** `K@D`를 `ΔE×(μ1_D−μ0_D)`로 적합한 R²=0.997은 `K@D ≡ ΔE·gap_D + (인구 합성항)`인 회계 항등식의 잔차(인구항 중앙 −0.132ms)가 작다는 것만 잰다 — "`E`가 `ρ`의 재진술이 아니다"의 실제 근거는 ρ-대리 R²=0.529·corr(ΔE,Δρ)=0.667(36쌍)뿐이다. 실무 규칙: 분해 성분의 "설명력"을 주장하려면 그 회귀가 항등식의 대수적 주항을 포함하는지 먼저 확인하라. 대응 `CONSENSUS.md` §3 항목183. 상세 `workspace/engine-port/results/longctx_conflict/audit_qb_rules_2026-09-11/VERDICT.md` §1.1(D3)·§5.2(QB-12).

164. ★★★**(2026-09-11, claims-auditor, GPU 0) 비고유(경로 의존) 분해에서 경로 지정의 방향 귀결은 값과 무관하게 대수로 정해질 수 있다(G-β+G-γ 병합).** 지정 전에 그 방향을 계산해 공개하고, 경로 차를 "외삽" 탓으로 돌리기 전에 두 경로가 모두 지지되는 거친 층에서도 그 차이가 남는지 확인하라. Q-B′에서 `I<0`이면 `G_cf>G_sf` ⇔ `Δ<0`이 대수적으로 성립하고, 외삽이 전혀 없는 e-only 조건화에서도 `G` 중앙값이 cf 0.815/sf 0.516으로 갈린다(차이는 상호작용에서 옴). 대응 `CONSENSUS.md` §3 항목184. 상세 `.../audit_qb_rules_2026-09-11/VERDICT.md` §2.1(#11)·§2.2(A2)·§3(a)(3).

165. ★★★**(2026-09-11, claims-auditor, GPU 0) 층을 세분하는 강건성·반전 시험은 지지 결손으로 채움 규칙의 산물을 "민감도"로 보고하게 할 수 있다.** 층 세분화 시험은 off-support 질량과 같은 층의 원자료 분포를 함께 보고하라. Q-B′ 감사가 (e,b)층에 토큰 위치 3분위를 추가하자 d16 λ=0.09 A 3칸에서 `K@D`가 65.9–73.6% 움직였으나, 노출 ITL 자체는 위치와 무관(d16 42.3/—/43.3ms)해 이 "반전"은 해당 3분위 노출 토큰 0개가 만든 채움 산물이었다(가짜 반전으로 판정, 死因 후보에서 제외). 대응 `CONSENSUS.md` §3 항목185. 상세 `.../audit_qb_rules_2026-09-11/VERDICT.md` §2.2(A6)·§4.4.

166. ★★★**(2026-09-11, claims-auditor, GPU 0) 인계된 질문의 정의가 항등식으로 소진되면 그 "정식 판본"은 같은 질문이 아니라 다른 질문이다 — 개명하고 원 질문의 소진을 등재 문장에 명시하라.** 원 Q-B(SM-split 액추에이터 자기상쇄 루프 이득, 프로브 C 교락 14.953×/23.713×)는 Little 항등식(도착 비×체류 비)으로 정확히 분해됐고, 핸드오프가 요구한 공통-λ 설계는 원 교락의 도착 채널 자체를 구조적으로 끈다 — 남는 비자명 내용(평균 ITL 차의 합성 표준화 분해)을 **Q-B′**로 개명해 등재한다. 대응 `CONSENSUS.md` §3 항목186. 상세 `.../audit_qb_rules_2026-09-11/VERDICT.md` §3(c)·§5.2(QB-15).

167. ★★★**(2026-09-11, `r2_correctness` 트랙, claims-auditor 결과 감사[`workspace/engine-port/results/r2_correctness/audit_r2corr_2026-09-11/VERDICT.md` §6, G-1], GPU 0) 진단 층으로 강등해도 그 층은 주장 스코프에서 빠지지 않는다.** 등록 규칙이 동시 부하(C) 층을 층 단위 귀무대조(L1-L2 불일치 4/32 ≠ 0)로 이미 진단 전용으로 강등해 뒀지만, 같은 조건을 **요청 단위**로 적용하면 C01·C10·C19(3/32)에서 L1=L2≠TD1=TD2라는 arm-분리 불일치가 발화한다 — 이것은 등록 판정의 死因은 아니지만(등록 범위 밖), PASS 문장의 스코프를 "통과한 층(S/O)"으로 한정해야 하는 사유다. 실무 규칙: 층 단위 귀무대조가 실패한 층을 강등할 때는 그 층의 요청별 등가류를 병기하고, PASS 선언문에 그 층이 배제됨을 명시하라. 기존 게이트와 중복 없음(#17[모든 보고 블록에 게이트를 걸어라]은 수치 누출, 이 항목은 판정 라벨의 스코프 누출; #20·#24·#142와도 무관). 대응 `CONSENSUS.md` §3 항목187(신설). 상세 `.../audit_r2corr_2026-09-11/VERDICT.md` §1.7·§6(G-1).

168. ★★★**(2026-09-11, `r2_correctness` 트랙, claims-auditor, GPU 0, G-2·게이트#20 인접) 같은 이름의 텔레메트리 필드가 arm마다 다른 술어로 채워질 수 있다.** FixedPolicy(legacy)는 `true_dual_worker_runtime is None` 조건에서 단락돼 `safe` 술어를 평가하지 않는 채 상수 True로 기록하는 반면, true-dual은 `arbiter.safe_to_switch()`로 in-flight CUDA 이벤트를 실제로 query한다 — 같은 `safe` 필드값(`unsafe_decisions`=0 대 30)이 "arm 간 거동 차이"를 뜻하지 않고 "같은 필드를 다른 술어로 채운 결과"일 뿐이다(60/60 unsafe가 전부 target=current=44인 no-op). #20의 "식별자 수입≠거동 수입"은 분석기 쪽 문제고, 이 게이트는 엔진 쪽 필드 의미 문제라 구분 유지. 실무 규칙: 두 arm의 텔레메트리 필드를 비교하기 전에 같은 술어가 양쪽에서 실제로 평가되는지 코드로 확인하라. 대응 `CONSENSUS.md` §3 항목188(신설). 상세 `.../audit_r2corr_2026-09-11/VERDICT.md` §3.2·§6(G-2).

169. ★★★**(2026-09-11, `r2_correctness` 트랙, claims-auditor, GPU 0, G-4) 워커 경로를 탔다는 것(태스크 카운터 증가)은 동시 실행의 증거가 아니다.** job 907100의 S/O 전 구간에서 두 worker 스레드의 `prefill_host_tasks`/`decode_host_tasks` 증가량이 기대값과 정확히 일치했지만(O4 충족), 벽시계 host-worker 중첩을 복원하면 S 층 0.000s, O 층 13.5/19.4ms(`host_worker_overlap_ratio`의 1024개 절단+서버 수명 분모 때문에 이조차 하한)뿐이었다 — "경로를 탔다"가 "동시에 실행됐다"를 함의하지 않는다. 실무 규칙: 동시성은 태스크 카운터가 아니라 중첩 시간과 경합 기회 수로 적어라. 하네스 결함으로 `host_worker_overlap_ratio`를 동시성 지표로 쓰지 말 것(engine-porter 이관, 게이트 아님). 대응 `CONSENSUS.md` §3 항목189(신설). 상세 `.../audit_r2corr_2026-09-11/VERDICT.md` §1.5·§6(G-4).

170. ★★★**(2026-09-12, `r2_correctness` 트랙, X1 사전등록 규칙층 감사, claims-auditor, GPU 0) 양성대조의 "교란됨" 판정은 상위 API가 보고하는 근사값이 아니라 커널이 실제로 쓰는 granularity로 해야 한다 — 게이트#9 계열 스무 번째 재발.** X1 사전등록 rev1의 pre-flight(`x1_preflight_splits.py`)는 `get_num_kv_splits_triton`이 돌려주는 **행별 split 수**만 비교했으나, 실제 축약 구간은 커널 안에서 `_MIN_BLOCK_KV=32`로 한 번 더 양자화된다(`kv_len_per_split = cdiv(cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV)*MIN_BLOCK_KV`, `decode_attention.py:35,98,553`) — split 수가 달라도 이 값이 같으면 결과는 비트 단위로 같다. 이 때문에 rev1의 핵심 수치("강제 6 ⇒ 24/24 교란, 무교란 0")는 거짓이었고 실효값은 **18/24 교란·72/96 unit-pair, 무교란 6/24**였다(그것도 하필 4-gram≤0.04 비복사형 7개 중 6개). 도구 자신의 docstring이 선언한 목적("does the planned perturbation actually change the decode attention reduction")을 자기 코드가 달성하지 못한 사례로, de-confound 교훈 9의 변종이다(gate #40이 "게이트#9 계열 열 번째 재발", gate #163이 "19번째 재발"이었던 것과 같은 형태로 새 게이트 번호를 받는다). 실무 규칙: 양성대조·항등식 검출 도구를 설계할 때는 API가 노출하는 파라미터가 아니라 그 파라미터가 실제로 소비되는 최종 계산(커널 내부 양자화·타일링 등)까지 추적해 "교란됨"을 정의하라. 대응 `CONSENSUS.md` §3 항목190(신설). 상세 `workspace/engine-port/results/r2_correctness/x1_prereg/VERDICT_x1_rules_2026-09-12.md` §1(a)·§6(D1).

171. ★★★**(2026-09-12, `r2_correctness` 트랙, X1 사전등록 rev2 설계 + claims-auditor 규칙층 감사, GPU 0) 교란 노브를 도입하는 실험은 그 교란과 무관한 배경 축(엔진 소스/커밋 등)을 이전 참조 실행과 고정해 드리프트를 통제하라 — 노브 하나만 진짜로 바뀌게 하라.** X1 rev1은 새 교란(decode KV-split 강제 고정)을 env var와 CLI arg 두 경로로 동시에 주입했다(제1원칙 confound 카탈로그 #10 "변수 동시 변경" 해당). 실효 기준으로 재계산하면 두 노브의 한계 기여 차이는 단 1/24 단위(S05)뿐이었으나, env var는 관측 가능성이 CLI arg보다 낮고(`SAME_ACROSS_BOOTS` 같은 자동 검사가 CLI arg만큼 보장되지 않음) 원인 귀속을 흐린다. rev2는 (i) env var를 버리고 `--triton-attention-num-kv-splits` CLI 인자 하나만 쓰고, (ii) 그와 별개로 엔진 소스를 이전 GO 판정(job 907100)이 돈 정확한 커밋(`38c1aca`, 8/8 해시 확인)으로 되돌린 뒤 새 교란을 도입해, 교란 효과와 "엔진이 907100 이후 드리프트했다"는 효과가 뒤섞이지 않게 했다. 실무 규칙: 새 교란을 도입하는 실험을 설계할 때는 (a) 교란 자체는 관측 가능한 단일 노브로 주입하고, (b) 그 교란과 무관한 배경 축(엔진 소스/커밋·설정 등)은 비교 대상이 되는 이전 참조 실행과 명시적으로 고정하라 — 그러지 않으면 "무엇이 바뀌어서 결과가 달라졌는가"를 사후에 분해할 수 없다. 대응 `CONSENSUS.md` §3 항목191(신설). 상세 `workspace/engine-port/results/r2_correctness/x1_prereg/VERDICT_x1_rules_2026-09-12.md` §1(d)·§6(D4)·§8.

172. ★★★**(2026-09-12(3), `r2_correctness` 트랙, X1 결과 감사[job 907456], claims-auditor, GPU 0.154 GPU-h, G-X1-1 — 게이트#9 계열의 3번째 층) 양성대조의 도달 범위는 "비교 단위 지시함수"가 아니라 "교란이 실제로 걸린 계산 스텝/토큰의 분포"로 등록하라.** 같은 양에서 granularity 오류가 세 번 일어났다: rev1은 행별 split 수(틀림) → 게이트#170(1단 감사)이 `kv_len_per_split`로 교정(옳음) → **rev2, 그리고 그 교정을 지시한 claims-auditor의 D1 자신이 "any-step 지시함수"에 머물러 단위 기준 24/24를 보고**(실제 decode 스텝 커버리지 1,142/1,384 = 82.5%, 최악 단위 S00 5/63). 검출력 배분이 여전히 숨었다. 자기 적용: 이 결함의 출처는 claims-auditor 자신의 1단 판정서 D1이다 — 처방자가 같은 층에서 다시 미끄러졌다. 대응 `CONSENSUS.md` §3 항목192(신설). 상세 `workspace/engine-port/results/r2_correctness/audit_x1_2026-09-12/VERDICT.md` §2.1·§11(G-X1-1).

173. ★★★**(2026-09-12(3), `r2_correctness` 트랙, X1 결과 감사, claims-auditor, GPU 0.154 GPU-h, G-X1-2 — 게이트#1·#114·#166 확장) 양성대조가 게이트가 인증하려는 운영점 위에 떨어졌는지 확인하고, 떨어지지 않았으면 "그 운영점에서는 미보정"을 등록 문안에 넣어라.** X1의 교란은 D44 resident 비교 계산 **0개**에 떨어졌다(prefill은 구성상 미교란, O 층 background 행은 bs=2 구간에서 비트 동일, bs=1 교란 구간[112/159 step, D44 상주]에서는 0/32로 뒤집히지 않음). 창 안에 그 파티션이 있었다는 것과 *교란이 그 파티션 위 계산에 실제로 닿았다*는 것은 다른 명제다. 대응 `CONSENSUS.md` §3 항목193(신설). 상세 `workspace/engine-port/results/r2_correctness/audit_x1_2026-09-12/VERDICT.md` §2.4·§11(G-X1-2).

174. ★★★**(2026-09-12(3), `r2_correctness` 트랙, X1 결과 감사, claims-auditor, GPU 0.154 GPU-h, G-X1-3, 신설) 등가류 "불일치 단위 수"는 불일치 총량이 아니다 — Σ(등가류−1) 또는 분할 전체를 병기하라.** X1의 C 층에서 불일치 단위 수는 907100의 8에서 6으로 줄었으나, 32단위 합 **Σ(등가류 수−1)은 10→10으로 불변**이었고 일부 단위(C00·C16)는 3류→**4류**로 오히려 더 갈렸다(L-L 불일치도 4→5로 증가). 단위 수만 보면 "교란이 불일치를 줄였다"는 반대 결론이 나온다. 대응 `CONSENSUS.md` §3 항목194(신설). 상세 `workspace/engine-port/results/r2_correctness/audit_x1_2026-09-12/VERDICT.md` §3.3·§11(G-X1-3).

175. ★★★**(2026-09-12(3), `r2_correctness` 트랙, X1 결과 감사, claims-auditor, GPU 0.154 GPU-h, G-X1-4 — 게이트#80 인접, 신설) provenance 보강을 교란 arm에만 넣으면 비대칭 기록이 되어 baseline 전제를 사후 검증 불가로 만든다.** X1 sbatch는 env var 덤프를 신설했으나 907100(baseline)에는 없어, baseline에서 `SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS`가 unset이었다는 것을 아티팩트로 증명할 수 없다(X1P-8). 실무 규칙: 보강 항목은 기준 arm에도 (재실행 없이 가능한 범위에서) 소급 기록하거나, "그 항목은 baseline에서 검증 불가"를 등록 문서에 명시하라. 뒤집힘의 존재 귀속 자체는 바뀌지 않으나 사전등록 교란표의 전제는 바뀔 수 있다. 대응 `CONSENSUS.md` §3 항목195(신설). 상세 `workspace/engine-port/results/r2_correctness/audit_x1_2026-09-12/VERDICT.md` §1.6·§11(G-X1-4).

176. ★★★**(2026-09-12(3), `r2_correctness` 트랙, X1 결과 감사, claims-auditor, GPU 0.154 GPU-h, G-X1-5 — 긍정 사례, 게이트#1의 처치 측 적용) 교란 노브 자신에 대해 "target이 아니라 realized"를 요구하라.** X1은 (a) server args 덤프 + `SAME_ACROSS_BOOTS` 검사, (b) cudagraph capture 메모리 변화(5/5 boot에서 `mem usage` 0.73→0.65 GB, `avail mem` 13.32→13.40 GB, 부호·차수가 `max_kv_splits` 8→2 축소와 정합) 두 채널로 이를 충족했다(정성적 실현 확인, 닫힌 형태 추정과 배수 2 불일치는 미해소). 앞으로 수치 구성 교란은 이 2채널 확인을 사전등록 항목으로 넣는다. 대응 `CONSENSUS.md` §3 항목196(신설). 상세 `workspace/engine-port/results/r2_correctness/audit_x1_2026-09-12/VERDICT.md` §5.1·§11(G-X1-5).

177. ★★★**(2026-09-12(4), Claim E 컨트롤러 코드, engine-porter 구현 + doc-steward 등재, GPU 0) 산문 사양의 두 조항(케이던스·즉시성)을 한 술어로 합치면 한쪽이 조용히 죽는다.** 2026-09-12(2) 시점의 `evaluation_due()`는 로드맵 `:976`(즉시 upshift)을 `stabilize`의 평가 분기 안에서만 계산해, 사실상 `:975`(정상 케이던스) 조건이 먼저 충족돼야만 `:976`을 볼 수 있었다 — "즉시"가 최악 한 epoch까지 늦어진다. 수리는 `:975`·`:976`을 `evaluation_due`의 **독립 disjunct** 두 개로 되돌리는 것이었다(`_cadence_due` OR `_live_underprediction`). 실무 규칙: 산문 사양이 "또는"으로 묶은 두 조항을 구현이 "그리고" 관계로(또는 한쪽이 다른 쪽의 전제 조건으로) 합성하면, 그 사양 위반은 유닛 테스트로는 드러나지 않고 반응 지연 같은 시간 단위 회귀로만 드러난다 — 두 조항을 각각의 정의원 헬퍼로 쪼개 disjunct로 조립하라. 대응 `CONSENSUS.md` §3 항목197(신설). 상세 `PROJECT_STATUS.md` 최상단 배너(2026-09-12(4)) 구현 상세 3항, `workspace/engine-port/src/multiplex/controller.py:151-166,168-179,196-216`.

178. ★★★**(2026-09-12(4), Claim E 컨트롤러 코드, engine-porter 구현 + doc-steward 등재, GPU 0) 긴급 경로를 추가할 때 그 경로가 다른 카운터의 시간 단위를 바꾸지 않는지 확인하라.** 게이트#177의 수리(즉시성 회복)만 단독으로 넣었으면 overload streak 증가 카운터가 케이던스 epoch이 아니라 **iteration마다** 증가해, 로드맵 `:979`의 "2 epochs 지속"(기본 케이던스에서 ≥400ms)이 실질적으로 "2 iterations"(≈27–85ms)로 조용히 재정의됐을 것이다 — admission 제한은 이 프로젝트에서 TTFT 폭발을 일으키는 실측 레버(HE0)이므로 이 축소는 보수적 오차가 아니라 위험 방향 오차다. 수리는 overload streak 증가를 별도 epoch 마커(`last_overload_epoch_s/_iteration`)로 게이팅해 케이던스 epoch당 최대 1회로 제한한 것이었다(①의 필수 동반 수리). 실무 규칙: 이벤트 기반 즉시 반응 경로를 추가할 때는, 그 경로가 우회하는 케이던스에 묶여 있던 다른 카운터(스트릭·재시도·백오프 등)를 함께 감사해 그 카운터의 "단위 시간"이 조용히 축소되지 않았는지 확인하라. 대응 `CONSENSUS.md` §3 항목198(신설). 상세 `PROJECT_STATUS.md` 최상단 배너(2026-09-12(4)) 구현 상세 4항, `workspace/engine-port/src/multiplex/controller.py:130-140[주석],181-194,311-326`.

179. ★★★**(2026-09-12(4), Claim E 컨트롤러 코드, engine-porter 구현 + doc-steward 등재, GPU 0) 미구현 절을 산문에 남겨 두면 구현된 것처럼 인용된다 — `bucket_changed`/"or bucket change" 사례.** 로드맵 `:975` "evaluate every `max(4 decode iterations, 100 ms)` **or bucket change**"의 후반절은 파라미터(`bucket_changed`)가 API·테스트에 존재하고 관련 테스트가 통과함에도 서빙 경로(`_r2_decide_idx`, `multiplexing_mixin.py:436-440`)가 이 인자를 **한 번도 넘기지 않고**, `src/multiplex`에 bucket 정의 자체가 없어 **미구현**이다. 파라미터를 지우면 "이 절이 미구현"이라는 사실 자체가 코드에서 사라지므로, 수리는 삭제가 아니라 `stabilize` docstring에 미구현임을 명시하는 것이었다. 실무 규칙: API 파라미터가 존재·테스트 통과한다는 것은 그 파라미터가 서빙 경로에서 채워진다는 증거가 아니다 — 로드맵/논문이 그 절을 인용하기 전에 실제 호출부가 그 인자를 전달하는지 grep으로 확인하고, 미구현이면 파라미터를 지우는 대신 "미구현"을 코드·문서 양쪽에 명시적으로 박아라. 대응 `CONSENSUS.md` §3 항목199(신설). 상세 `PROJECT_STATUS.md` 최상단 배너(2026-09-12(4)) 구현 상세 6항, `workspace/engine-port/src/multiplex/controller.py:249-262`, `multiplexing_mixin.py:436-440`.

180. ★★★**(2026-09-12(5), B3 사전등록 규칙층 감사, claims-auditor, GPU 0, G-B3-1 — 교훈 53·85 계열) 직전 회차의 허위-0 수리를 새 스크립트에 옮길 때는 granularity를 재도출하라.** X1 감사의 D8은 "gen 파일 4개 미만이면 NOT-RUN"을 요구했고, B3의 `c_tier_null_control.py`는 그것을 **파일 존재**로 그대로 복제했다 — 그래서 파일은 4개 있지만 한 boot의 `phase_c`가 **빈 리스트**인 경우(레코드 결손)에서 허위 0(`units=0 … E4=0`)이 그대로 재발한다. 수리는 국소적이었고, 같은 형태의 결함이 한 층 아래(파일 수준 → 레코드 수준)로 이동했을 뿐이다. 실무 규칙: 이전 회차의 수리를 새 코드베이스로 옮길 때는 "무엇을 막았는가"가 아니라 "어느 granularity에서 막았는가"를 재도출해 새 코드의 같은 granularity를 확인하라. 대응 `CONSENSUS.md` §3 항목200(신설). 상세 `workspace/engine-port/results/r2_correctness/b3_prereg/VERDICT_b3_rules_2026-09-12.md` §9(D1)·§12(G-B3-1).

181. ★★★**(2026-09-12(5), B3 사전등록 규칙층 감사, claims-auditor, GPU 0, G-B3-2 — 교훈 9의 또 한 층) selftest는 "핵심 추정량의 *정의*(이름-모양 배선)"를 assert해야 한다.** B3의 selftest는 `shape_of()`(어느 분할이 2-2인가)는 검사하지만, `report()`가 그 2-2 모양에 **이름**(`{1,3}|{2,4}` 등)을 붙이는 `named` dict 자체는 검사하지 않는다 — 그 이름 배선을 뒤바꾼 변이(`M2`)가 selftest를 통과하면서, 참조 데이터(907100)에 돌리면 핵심 결과 수치 E4를 **3 대신 0**으로 출력한다. B3의 결론 전부가 이 한 숫자에 의존하는데, 그 숫자의 정의 자체가 무검증이었다. 실무 규칙: 변이 내성 시험을 설계할 때는 "무엇을 계산하는가"(예: 분할 모양)뿐 아니라 "그 계산 결과에 어떤 이름/의미를 붙이는가"까지 assert 대상에 넣어라. 대응 `CONSENSUS.md` §3 항목201(신설). 상세 `.../b3_prereg/VERDICT_b3_rules_2026-09-12.md` §5·§9(D5)·§12(G-B3-2).

182. ★★★**(2026-09-12(5), B3 사전등록 규칙층 감사, claims-auditor, GPU 0, G-B3-3 — 게이트#18 인접) 귀무대조의 문턱은 그 귀무에서의 발화 확률과 함께 등록하라.** B3의 F4 문턱 "E4 ≥ 2"는 position 구조가 전무한 균등 귀무(1/3)에서도 907100의 n_22=4 조건에서 **40.7%**(n_22=3에서 25.9%)로 발화하며, 그 분기가 도달하는 결론은 하필 "기존 관측을 은퇴시키는" 쪽이다 — 문턱만 보면 마치 신중한 것 같지만 실제 오경보율은 높다. 실무 규칙: 반증 문턱(threshold)을 사전등록할 때는 그 문턱값뿐 아니라 "완전한 귀무(효과 0)에서 이 문턱이 발화할 확률"을 함께 계산해 등록 문안에 넣어라 — 특히 그 발화가 이끄는 결론이 기존 정본을 약화시키는 방향일 때 필수다. 대응 `CONSENSUS.md` §3 항목202(신설). 상세 `.../b3_prereg/VERDICT_b3_rules_2026-09-12.md` §2(R3)·§4·§12(G-B3-3).

183. ★★★**(2026-09-12(5), B3 사전등록 규칙층 감사, claims-auditor, GPU 0, G-B3-4, 신설) 공허 수량자가 arm에 따라 verdict 라벨을 바꿀 수 있다.** `r2_correctness_check.py:471`의 `s_ref_ok = all(... for a,b in combinations(l_boots,2))`는 `l_boots=[]`(TD-only job)에서 **공허 참**이 된다 — 같은 물리적 사건(S 층 불일치)이 `L L L L`에서는 `NO_VERDICT_INFRA`, `TD TD TD TD`에서는 `VERDICT FAIL`로 **다르게** 기록된다(S 토큰 1개 섭동으로 실증). 단일-arm 하위집합(동일-arm job)을 돌리는 모든 캠페인은 판정 규칙 안의 `all()`/`any()` 같은 수량자가 그 부분집합에서 공허해지지 않는지 사전에 실행으로 확인해야 한다. 실무 규칙: 판정 규칙에 `all(...)`/`any(...)`가 있으면, 그것이 순회하는 컬렉션이 특정 arm 구성에서 빈 리스트가 될 수 있는지 확인하고, 공허 참/공허 거짓이 검사 의미를 바꾸는지 실제 실행으로 검증하라. 대응 `CONSENSUS.md` §3 항목203(신설). 상세 `.../b3_prereg/VERDICT_b3_rules_2026-09-12.md` §6·§9(D4)·§12(G-B3-4).

184. ★★★**(2026-09-12(5), B3 사전등록 규칙층 감사, claims-auditor, GPU 0, G-B3-5, 자기 적용 — 게이트#113 계열) 감사자의 "유일한 값싼 길" 처방은 그 자신이 사전등록 심사를 받기 전까지는 설계 권고가 아니다.** X1 결과 감사 §12-1은 B3(동일-arm 귀무대조)를 "F3을 해석 가능하게 만드는 유일한 값싼 길"로 지목했으나, 그 처방 자체가 사전등록되자 (i) 조건부 균등 귀무가 이미 GPU 0으로 정본에 존재했다는 것 (ii) 분모가 하나가 아니라 둘(L·TD 두 과정)이고 결합 규칙이 없다는 것 (iii) 동일-arm 설계가 가설 A(TD가 다르게 계산)와 B(TD가 다르게 타이밍)를 동시에 지워 구별 불가하게 만든다는 것, 이 셋을 놓쳤다. 같은 판정서 §12-2는 위 게이트#180–183과 무관하게 별도로 사실과 다른 "n≥4 충족" 주장(항목178 참조·2026-09-12 세션 정정 1)을 담았다. 처방자가 같은 층(설계 자문)에서 다시 미끄러진 사례다. 실무 규칙: 감사자·설계자가 스스로 제안한 다음 실험은, 제안자 본인이 그 실험의 사전등록을 승인할 수 없다 — 별도 회차(또는 별도 역할)의 적대적 재심사를 거쳐야 "권고"의 지위를 얻는다. 대응 `CONSENSUS.md` §3 항목204(신설). 상세 `.../b3_prereg/VERDICT_b3_rules_2026-09-12.md` §8·§12(G-B3-5).

185. ★★★**(2026-09-12(5), OS 사전등록 규칙층 감사, claims-auditor, GPU 0, G-OS-1 — 게이트#110의 실패 사례) 직전 판정서의 코드 사실은 그 사실이 참이었던 트리 상태와 함께 인용하라.** B3 판정서 §10/R10은 "`FixedPolicy`가 `CoarseGrainedController`를 안 쓰므로 엔진 소스를 핀해도 값은 어차피 같다"고 적었는데, 이 명제는 **`HEAD` vs `38c1aca`** 비교에 대해서만 참이었다(그 시점 `dual_worker.py`는 두 트리에서 동일). OS 사전등록은 이 명제를 **트리 무관 상수**로 그대로 승계했으나, 그 사이 작업 트리에 `dual_worker.py` +172줄(A2의 H2 = true-dual hot loop 변경)이 미커밋 상태로 들어와 명제가 거짓이 됐다 — 핀을 건너뛰면 이제는 TD arm이 실제로 다른 코드로 돈다. 실무 규칙: "코드 X는 Y에 영향이 없다"는 감사 결론을 인용할 때는 그 결론이 성립했던 정확한 커밋/트리 상태를 함께 인용하고, 재인용 시점에 작업 트리가 그 상태와 같은지 재확인하라 — 감사 결론의 유효기간은 그것이 읽은 트리다. 대응 `CONSENSUS.md` §3 항목205(신설). 상세 `workspace/engine-port/results/r2_correctness/os_prereg/VERDICT_os_rules_2026-09-12.md` §9(D2)·§12(G-OS-1).

186. ★★★**(2026-09-12(5), OS 사전등록 규칙층 감사, claims-auditor, GPU 0, G-OS-2 — 게이트#113 계열) "소스를 핀하라"는 처방은 미커밋 작업의 존재를 확인한 뒤에만 실행 가능하다.** `git checkout <ref> -- <path>`는 해당 경로의 unstaged 변경을 **reflog 없이** 파괴한다. OS 사전등록이 요구한 `git checkout 38c1aca -- workspace/engine-port/src/multiplex`를 그대로 실행했다면, 미커밋 상태였던 A1/A2의 `dual_worker.py`(+172줄)·`controller.py` 작업이 복구 불가하게 사라졌을 것이다. 실무 규칙: 엔진 소스 핀을 요구하는 모든 사전등록은 "제출 전 `git status --porcelain` 전체를 기록하라"뿐 아니라 "핀 대상 경로에 미커밋 변경이 있으면 먼저 커밋 또는 `git stash`하라"를 명령문으로 등록 문안에 넣어야 한다. 대응 `CONSENSUS.md` §3 항목206(신설). 상세 `.../os_prereg/VERDICT_os_rules_2026-09-12.md` §9(D1)·§12(G-OS-2).

187. ★★★**(2026-09-12(5), OS 사전등록 규칙층 감사, claims-auditor, GPU 0, G-OS-3 — 게이트#172 인접) 교락 해소 설계는 '깨는 교락'만이 아니라 '새로 만드는 교락'을 열거해야 한다.** 부팅 순서를 `L TD L TD`(B3가 죽던 원래 순서)에서 `L TD TD L`로 바꾸면 arm 경계가 `{1,3}|{2,4}`에서 `{1,4}|{2,3}`으로 **이동**할 뿐이고, 후자는 동시에 "외곽 vs 중앙" 위치 모양이다 — 세 개의 가능한 2-2 모양(arm 정렬·패리티·인접)에 세 개의 물리적 해석을 1:1로 붙이는 것은 "위치 구조의 후보가 정확히 셋뿐"이라는 미등록 가정이었다. 실무 규칙: 순서/배치를 바꿔 한 교락(예: arm≡패리티)을 깨는 설계를 사전등록할 때는, 새 배치가 만들 수 있는 **다른** 위치 구조 후보(외곽-중앙, 인접, 고립 등)를 전수 열거하고 그중 어느 것과 여전히 교락되는지 명시하라. 대응 `CONSENSUS.md` §3 항목207(신설). 상세 `.../os_prereg/VERDICT_os_rules_2026-09-12.md` §3(3)·§12(G-OS-3).

188. ★★★**(2026-09-12(5), OS 사전등록 규칙층 감사, claims-auditor, GPU 0, G-OS-4, 신설) 사후 선택된 단위의 사전등록 재현은 정당하지만, (i) 재현 대상 자신의 귀무 확률과 (ii) 선택에서 배제된 반대 증거 단위를 함께 등록해야 한다.** OS의 F3은 907100에서 관측 후 선택된 초점 3단위(C01·C10·C19)가 새 job에서도 arm 모양에 떨어지는지를 사전등록해 재현하는 정당한 설계이지만, 그 자체로 등록해야 할 두 수치가 빠지면 과대해석된다: (i) 재현 대상인 907100의 관측(2-2 4건 중 3건이 한 모양) 자신이 균등 귀무에서 **P=1/3**인 사건이고, (ii) 초점 3단위 선택은 907100의 2-2 4건 중 arm을 지지하지 않은 유일한 단위(C24)를 **배제**한다 — 이 둘을 넣지 않으면 F3 재현 성공이 실제보다 9–30배 강해 보인다. 실무 규칙: 사후 관측에서 선택한 검정 단위를 사전등록으로 재현할 때는, 그 선택이 배제한 반대 증거와 재현 대상 자체의 귀무 확률을 등록 문안에 명시적으로 병기하라. 대응 `CONSENSUS.md` §3 항목208(신설). 상세 `.../os_prereg/VERDICT_os_rules_2026-09-12.md` §5.3(D7)·§12(G-OS-4).

189. ★★★**(2026-09-12(5), OS 사전등록 규칙층 감사, claims-auditor, GPU 0, G-OS-5 — 긍정 사례) "구성상 게이트가 도는가"는 논증하지 말고 판정 규칙을 그 구성으로 실제 실행해서 보여라.** OS 사전등록의 "혼합-arm 순서라 checker의 전 경로가 돈다"는 주장은 907100 아티팩트에 `boots.txt="L1 TD1 TD2 L2"`만 부여해 무수정 판정 규칙(`r2_correctness_check.py`)을 실제로 돌려 **6쌍 해소·`O_null_control_complete=True`·`B6=0/56`·`VERDICT PASS`**로 확인됐다 — B3의 같은 자리 주장("동일-arm이라 checker가 `NO_VERDICT_INFRA`로 떨어진다")도 실행해 보니 부분적으로 **틀렸다**는 것이 드러난 것(TD-only에서 `s_ref_ok` 공허 참 → `VERDICT FAIL`, 항목183 참조)과 대조된다. 실무 규칙: "이 하네스/판정 규칙이 이 구성에서 정상 경로를 탄다"는 주장은 코드를 읽고 논증하지 말고, 그 정확한 구성(부팅 라벨 순서 등)으로 무수정 도구를 실제로 1회 실행해 보여라 — 논증과 실행 결과가 갈릴 수 있다. 대응 `CONSENSUS.md` §3 항목209(신설). 상세 `.../os_prereg/VERDICT_os_rules_2026-09-12.md` §4(a)·§12(G-OS-5).

190. ★★★**(2026-09-13, X3 사전등록 규칙층 감사, claims-auditor, GPU 0, G-X3-1 — 게이트 #1의 3번째 층) 관측자를 끄는 실험은 "끈 관측자"와 "남긴 관측자"의 발화율을 함께 등록하라.** X3는 boot당 1263–2285건 발화하는 관측자(forced 샘플)를 끄고 2580–2617건 발화하는 관측자(`--decode-log-interval 1`)를 남긴다. "관측자 효과를 시험했다"는 서술은 남긴 쪽의 발화율을 적지 않으면 과대다. 대응 `CONSENSUS.md` §3 항목210(신설). 상세 `workspace/engine-port/results/r2_correctness/x3_prereg/VERDICT_x3_rules_2026-09-13.md` §1·§14(G-X3-1).

191. ★★★**(2026-09-13, X3 사전등록 규칙층 감사, claims-auditor, GPU 0, G-X3-2 — 게이트#110/G-OS-1 대칭, 긍정 사례) "구성상 그렇게 된다"는 예보는 논증하지 말고 무수정 판정 규칙을 그 구성의 반사실 입력으로 실제 실행해 보여라.** X3는 907100에서 `trace_forced` 레코드만 제거한 입력으로 무수정 `r2_correctness_check.py`를 돌려 `NO_VERDICT_UNREALIZED`·`failures=[]`·실패 하위검사 O2 한정까지 선취했다. 등록이 추정한 기전("15–19건 중 하나가 창에 들어와야")은 크기가 81% 적중으로 틀렸지만 결론은 맞았다 — 실행이 없었으면 둘을 구별할 수 없었다. 대응 `CONSENSUS.md` §3 항목211(신설). 상세 `.../x3_prereg/VERDICT_x3_rules_2026-09-13.md` §5·§14(G-X3-2).

192. ★★★**(2026-09-13, X3 사전등록 규칙층 감사, claims-auditor, GPU 0, G-X3-3, 신설 — #174·G-OS-4 인접) 다른 처치를 위해 감사·강화된 분석 도구를 재사용할 때, 그 도구가 *인쇄하는 해석 문장*이 새 처치에서 부호가 반대인지 확인하라.** 재사용 비교기(`x1_cross_job_compare.py`)가 아티팩트에 찍는 "`mismatch==0` → sensitivity not demonstrated"는 X1에서는 참이었으나 X3에서는 **0이 등록된 긍정 결과**라 의미가 반대다. 아티팩트에 박히는 문장은 나중에 정본으로 새어 들어간다 — 도구를 고칠 수 없으면(해시 유지) 인용 금지 문장으로 중화하라. 대응 `CONSENSUS.md` §3 항목212(신설). 상세 `.../x3_prereg/VERDICT_x3_rules_2026-09-13.md` §7·§14(G-X3-3).

193. ★★★**(2026-09-13, X3 사전등록 규칙층 감사, claims-auditor, GPU 0, G-X3-4, 신설) 귀무대조의 적용 범위를 층별로 적어라.** `907032→907100`은 **S층 전용** 귀무대조다(907032에 O층 부재). "교차-잡 불일치는 귀무대조로 처치에 귀속된다"는 문장은 O층에 대해서는 거짓이다. 대응 `CONSENSUS.md` §3 항목213(신설). 상세 `.../x3_prereg/VERDICT_x3_rules_2026-09-13.md` §2#4·§14(G-X3-4).

194. ★★★**(2026-09-13, X3 사전등록 규칙층 감사, claims-auditor, GPU 0, G-X3-5, 신설 — 우선순위 규율) "트랙을 해제한다"는 주장은 그 트랙의 블로커 목록을 열거하고 각각 제거되는지 대조한 뒤에만 쓰라.** 감사자 자신의 OS §8이 X3가 성능 트랙을 연다고 적었으나 블로커 3개(W2/4/5 ctx·B2/8 oracle 산출물·B6 hybrid profile 산출물) 중 실제로는 **0개**가 제거된다(자기 철회, 3회차 누적). 부수로 정본의 "약 270 run 불가"는 중복 계수였다(고유 225/가능 180). 대응 `CONSENSUS.md` §3 항목214(신설). 상세 `.../x3_prereg/VERDICT_x3_rules_2026-09-13.md` §10·§14(G-X3-5).

195. ★★★**(2026-09-13(2), engine-porter + doc-steward, GPU 0 — provenance manifest 확장에서 도출) 서빙한 모델의 구현 파일이 provenance manifest에 없으면 그 캠페인은 모델 축에서 귀속 불가다.** job 905835(`longctx_conflict` 트랙, 2026-09-09, `results/longctx_conflict/probes/c_905835/runtime_source_manifest.sha256`)는 17항목이고 `models/` 아래 항목은 `mamba2.py`·`zamba2.py`뿐인데, 이 job은 `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base`(`NemotronHForCausalLM`)를 12 boot 서빙했다 — 서빙된 모델 구현의 provenance가 **0**이었다(`sync_engine_tree.sh:94`가 그 파일을 "수동 복사"라고 적어 매니페스트 생성 대상에서 제외했었음). 커밋 `87213a9`가 `models/{nemotron_h,falcon_h1,granitemoehybrid}.py`를 설치 경로로 옮기고 `configs/{nemotron_h,falcon_h1,granitemoehybrid}.py`·`configs/mamba_utils.py`를 해시 전용으로 추가해 매니페스트를 17→24항목으로 넓혔다(신규 7줄은 뒤에 append, 기존 17줄·순서 바이트 동일). 해시 전용 근거: `configs/nemotron_h.py`가 `hybrid_override_pattern → layers_block_type`을, `mamba_utils.Mamba2StateShape`가 그것을 `mamba_cache_per_req`로 바꾸므로 λ* 라벨이 딛고 선 값이다. ★**읽기 규칙**: 2026-09-13 이전 job의 manifest는 같은 17파일을 검증하고 이후 job은 7줄이 더 많다 — 교차-잡 서술은 "기존 17은 여전히 일치 + 신규 7항목 존재"로 쓰고 "17/17 동일"로 쓰지 않는다. 대응 `PROJECT_STATUS.md` "방법론 게이트" #195(본 항목). 상세 `PROJECT_STATUS.md` 최상단 배너(2026-09-13(2)) "D" 절, 커밋 `87213a9` 메시지.

196. ★★★**(2026-09-13(3), 캠페인 0단계[λ*] 사전등록 규칙층 감사, claims-auditor, GPU 0, 死因 N2) 비포화 셀의 achieved/offered는 포화도가 아니라 도착 실현 계수다 — Poisson 부하에서 명목 rate를 분모로 쓰는 문턱 규칙은 client seed가 사다리 전체에 공통모드로 들어가 실현 가능한 N으로는 신뢰 도달이 불가능하다.** `bench_serving.py:1706 np.random.seed(args.seed)` + `:948 np.random.exponential(1/rate)`이므로 비포화 셀의 `achieved/offered`는 서비스 포화도가 아니라 `1/Ē`(Ē=도착 간격 표본평균)다 — seed가 프로세스당 한 번 고정돼 사다리 전체에 공통모드로 들어가고 셀을 늘려도 평균되지 않는다. job 905835 원자료 12/12 셀 전부 같은 방향(−15~−18%, 평균 −2.0σ)으로 치우쳐 있었고, seed 40개 모의에서 저측(0.95 문턱) 실패율이 15–28%였다(N=90/150/300). `1.96/√(N−1)≤0.05`가 요구하는 N≥1537은 이 프로젝트의 실현 가능한 boot 예산 밖이다. 실무 규칙: Poisson 도착 부하에서 명목 rate 대비 achieved 비를 포화 판정 문턱으로 쓰는 모든 사전등록은 client seed를 등록하거나(그래도 N2 소멸 안 됨) `--request-rate inf` 같은 결정적 포화 프로브로 그 판정을 대체해야 한다. 대응 `CONSENSUS.md` §3 항목216(신설). 상세 `workspace/engine-port/results/r2_eval/lambda0_prereg/VERDICT_lambda0_rules_2026-09-13.md` §1(S1)·§2(N2).

197. ★★★**(2026-09-13(3), 캠페인 0단계[λ*] 사전등록 규칙층 감사, claims-auditor, GPU 0, 死因 N2) 사다리 규칙이 최저·최고점만 쓰면 중간점은 예보가 아니다 — 사다리 값을 수치로 등록하지 않으면 그 예보는 자유 표면이고 판정을 뒤집을 수 있다.** λ0 등록 §2는 "shape A rate 4점(앵커가 없어 넓게)"라고만 적고 네 수를 적지 않았다. R1 브래킷 판정은 x_min·x_max에만 의존하므로 중간 2점은 애초에 판정에 기여하지 않는데도, 등록되지 않은 4점 전체가 자유 표면으로 남아 있으면 사후에 어떤 사다리를 고르느냐로 `BRACKETED↔NOT_BRACKETED`가 뒤집힌다(감사자가 λ*(A)≈2.1 가정하에 3가지 사다리 후보로 반전 예시 3건을 실제로 계산해 확인). 실무 규칙: rate 사다리를 쓰는 사전등록은 사다리의 모든 점을 숫자로 등록하고, 등록된 문턱과 그 수로 계산되는 브래킷 창을 사전등록 문서 자체에 명시하라(그 계산이 향후 원자료로 검증 가능해야 한다). 대응 `CONSENSUS.md` §3 항목217(신설). 상세 `.../lambda0_prereg/VERDICT_lambda0_rules_2026-09-13.md` §1(S2)·§2(N2).

198. ★★★**(2026-09-13(3), 캠페인 0단계[λ*] 사전등록 규칙층 감사, claims-auditor, GPU 0, 死因 N3) 예보의 정의역은 생성기를 실행해 확인하라 — 워크로드 명세 문자열을 입력 shape로 읽으면 예보가 공집합 위에 선다.** λ0 등록 §3은 "W4의 decode phase는 (64, 512)이고 이 단계는 (256, 512)로 근사한다"고 적었으나, `benchmarks/pdmux_eval/workloads.py:159-160`을 실제로 실행하면 W4 decode phase는 **(256, 512)** 그 자체다 — (in 64, out 512)는 저장소 어디에도 없다. 오독의 출처는 `WorkloadSpec.output_distribution = "64/512 by phase"`(phase별 **출력** 길이 서술)를 입력 shape로 읽은 것이었다. 이 오독은 이 사전등록 이전에 메인 세션의 다른 등재(`PROJECT_STATUS.md`·`CONSENSUS.md`·`EXPERIMENT_ROADMAP.md`·`CLAIM_EVIDENCE_MATRIX.md`의 2026-09-13(2) 배너)에도 이미 승계돼 있었다 — 정본 문서가 여러 개 동시에 같은 오독을 반복해도 코드 실행 전에는 드러나지 않는다. 실무 규칙: 워크로드/데이터 스펙 문자열에서 파생된 수치 예보는 그 문자열을 만드는 생성기 코드를 실행해 실제 산출을 눈으로 확인한 뒤에만 등록하라 — 산문 인용만으로는 정의역 존재를 보장하지 못한다. 대응 `CONSENSUS.md` §3 항목218(신설). 상세 `.../lambda0_prereg/VERDICT_lambda0_rules_2026-09-13.md` §1(S8)·§2(N3)·§3(4).

199. ★★★**(2026-09-13(3), 캠페인 0단계[λ*] 사전등록 규칙층 감사, claims-auditor, GPU 0 — R4 유형 재발) 닫힌 형태 추정식이 자기 추정량(예측 대상 자체)을 입력으로 요구하면 그것은 항등식이지 예측 모델이 아니다.** λ0 등록 §3이 예보 근거로 인용하려 한 `c_capacity_analyze.py`의 닫힌 형태 `L_decode=[(1−R)/R]·(itl_load/itl_solo)/s(D)`를 대입해 전개하면 `pred=μ_ach·out·itl_load`로 환원된다 — `μ_ach`가 곧 λ*이므로 이 식은 λ*를 예측하는 데 **λ*를 입력으로 요구**한다. `obs/pred`는 사실상 `mean_itl/itl_p50`를 검정한 것이었다(longctx_conflict 트랙 R4/MODEL_HOLDS와 같은 함정). λ0 등록이 R4를 승계하지 않은 것은 이 감사로 확인상 옳았다 — 되살리지 말 것. 실무 규칙: 닫힌 형태 추정식을 예측 도구로 재사용하기 전에 그 식을 대입 전개해 예측 대상 자신이 우변에 나타나는지 확인하라. 대응 `CONSENSUS.md` §3 항목219(신설). 상세 `.../lambda0_prereg/VERDICT_lambda0_rules_2026-09-13.md` §3(3).

200. ★★★**(2026-09-13(3), 캠페인 0단계[λ*] 사전등록 규칙층 감사, claims-auditor, GPU 0 — 게이트 #21 확장, 제출 차단급) `UNRESOLVED`(측정 실패) 경로가 live 코드에서 실제로 도달 가능한지 주입 실험으로 확인하라 — glob 기반 셀 수집은 결손 셀을 조용히 빼고 규칙 판정(실패가 아닌)을 내보낼 수 있다.** `lambda0_label.py:104`의 `for f in sorted(d.glob("cell_*.json"))`는 없는 셀 파일을 리스트에서 그냥 빠뜨리므로 `any(c is None)` 검사가 결코 참이 될 수 없고, 등록 §4가 "셀 JSON 결손 → `UNRESOLVED`, 측정 실패이며 규칙 실패가 아니다(교훈 21)"라고 적은 문안과 실제 코드가 불일치했다 — boot 실패가 조용히 `KNEE_NOT_BRACKETED`(규칙 실패)로 나온다. 감사자가 4점 사다리에서 셀 1개를 실제로 삭제하는 주입 실험으로 이 불일치를 재현했다(상단 결손 → `KNEE_BRACKETED`로 오히려 통과, 하단 2점만 결손 → `KNEE_NOT_BRACKETED`). 실무 규칙: "측정 실패는 X로 라벨한다"고 등록한 사전등록은 제출 전에 그 실패 라벨이 실제 코드 경로에서 도달 가능한지 결손 셀을 인위적으로 주입해 확인하라 — 산문 서술과 glob 기반 파일 수집의 암묵적 스킵 동작은 다른 것이다. 대응 `CONSENSUS.md` §3 항목220(신설). 상세 `.../lambda0_prereg/VERDICT_lambda0_rules_2026-09-13.md` §1(S9)·§6(c).

201. ★★★**(2026-09-13, job 907959 결과 감사, claims-auditor, GPU 0 — 긍정 사례) "진단 전용" 라벨의 정의역을 사전등록 문안에 명시하라 — 불일치 *수*만 가리키는지, 그 층에서 일어난 *모든 사건*을 가리키는지 구분하지 않으면 결과 해석 시점에 미등록 재량이 생긴다.** 907959의 verdict rule v2는 "C층은 진단 전용(mismatch counts are reported and never enter the verdict)"이라고 괄호로 정의역을 **불일치 수로 한정**해 두었고, 그 덕에 C층에서 일어난 서버 크래시(`B2_NO_CRASH`, 티어 수식 없는 퍼-boot 술어)가 판정에 들어가는 데 재량이 0이었다(감사자 반사실 CF1로 과결정 확인). 실무 규칙: 관측자·계층을 "진단 전용"으로 강등하는 모든 사전등록 문구는 그 강등이 정확히 무엇(수치/사건/전체)에 적용되는지 괄호로 명시하라. 대응 `CONSENSUS.md` §3 항목221(신설). 상세 `workspace/engine-port/results/r2_correctness/audit_907959_2026-09-13/VERDICT.md` §2·§14(1).

202. ★★★**(2026-09-13/14, job 907959 결과 감사 + λ0 rev4 규칙층 감사, claims-auditor, GPU 0) 퍼-셀 판정을 요구하는 예보는 생산 하네스가 실제로 퍼-셀 산출물을 만드는지 확인하라 — 소비자가 전역 집계 파일 하나를 셀별 값으로 오독하면 실격 경로가 실효 항등식이 된다.** newpair F5("두 셀 각각 48 도달")를 위해 하네스가 쓴 `I3_max_running_req.txt`는 `sort -n | tail -1`로 만든 **전역 최댓값 한 줄**뿐이었고, 그 파일을 그대로 소비한 `lambda0_lambda_inf.py`의 `min(vals) >= 48` 조건은 "어느 한 셀이라도 48에 닿았다"와 동치인 실효 항등식이었다(무수정 술어 + job 907959 실제 `instrument/`가 `ANCHORED`·실격 0건을 냄, 셀별 재계산 결과는 I3a=48/I3b=2). 실무 규칙: "퍼-셀 값이 필요하다"고 등록했다면 그 값을 실제로 만드는 생산 코드(하네스)를 실행해 파일 포맷을 확인한 뒤에만 소비 술어를 신뢰하라 — selftest가 통과하는 픽스처를 생산자가 한 번도 만들지 않으면 그 분지는 죽어 있다. 대응 `CONSENSUS.md` §3 항목222(신설). 상세 `.../audit_907959_2026-09-13/VERDICT.md` §6-4·§14(2); `.../lambda0_prereg/VERDICT_lambda0_rev4_2026-09-14.md` §2(N1)·G-λ4-1.

203. ★★★**(2026-09-13, job 907959 결과 감사, claims-auditor, GPU 0) 포화 판정 계기는 그 워크로드 shape의 실제 병목 축과 맞춰라 — decode-슬롯 계기는 prefill-토큰-지배 shape에서 구조적으로 도달 불가할 수 있다.** F5가 쓴 `#running-req ≥ 48`(mamba/decode 슬롯 계기)은 shape B(8192-in, prefill 지배)에서 진짜 구속인 `max_prefill_tokens=16384`를 반영하지 못해, 서버가 명백히 병목(대부분 구간 `#queue-req` 60 이상, 최대 62)인데도 계기 자체가 발화할 수 없었다. 실무 규칙: 포화/병목 판정 계기를 등록하기 전에 "이 계기가 이 shape의 지배 자원과 같은 축인가"를 확인하라(게이트 #200 계열의 자매 항목 — #200이 UNRESOLVED 경로 도달성을, 이 항목은 계기 자체의 shape 적합성을 다룬다). 대응 `CONSENSUS.md` §3 항목223(신설). 상세 `.../audit_907959_2026-09-13/VERDICT.md` §6-3·§14(3).

204. ★★★**(2026-09-13, job 907959 결과 감사, claims-auditor, GPU 0 — N-7) 분할(green-context SM) 라벨을 수치에 붙이기 전에 그 측정 구간이 실제로 prefill∧decode 동시성 위에 있었는지 확인하라 — 동시성-1 프로브는 구성상 분할 인덱스를 밟을 수 없다.** 엔진은 prefill과 decode가 **동시에** 활성인 구간에서만 green idx 4(D44)를 쓰고, 그 외(동시성 1처럼 항상 하나만 진행 중인 구간)에는 idx 0/5(비분할, 108 SM)에서 돈다(같은 job 텔레메트리 12,458 스냅샷 전수: idx0 11,725·idx4 592·idx5 141). newpair I2(동시성 1) 프로브의 TTFT/ITL을 "D44 값"으로 태깅한 사전등록 문구(NP-9/D6-ii)는 이 사실로 거짓임이 확인됐다 — 반대로 λ0 rev1 판정서가 이미 옳게 적어 둔 각주("B=1은 pdmux 분할 미적용 구간")가 있었다는 것도 함께 기록한다(정본 문구가 새 사전등록에 밀리는 사례). 실무 규칙: 분할 라벨을 수치에 붙일 때는 그 구간의 동시성(양쪽 phase가 동시에 in-flight였는가)을 텔레메트리로 직접 확인하라. 대응 `CONSENSUS.md` §3 항목224(신설). 상세 `.../audit_907959_2026-09-13/VERDICT.md` §6-2·§14(4)(N-7).

205. ★★★**(2026-09-13, job 907959 결과 감사, claims-auditor, GPU 0) verdict 리포트가 인쇄하는 카운터 이름이 게이트 술어가 실제로 읽는 값과 같은 것인지 확인하라 — 이름이 겹치면 "N건인데 왜 통과?"라는 오독을 유도한다.** `verdict.txt`의 `unsafe_decisions`(텔레메트리 `controller_decision.safe=False` 계수, TD 16건)는 B2가 실제로 검사하는 로그 정규식 `unsafe (automatic )?split transition`(값 0, 4 boot 전부)과 **다른 양**이다 — 16건 전부 `target=current=44`라 분할 전이 자체가 없었다(FixedPolicy 설계상 필연). 실무 규칙: 같은 리포트 안에 이름이 비슷한 두 카운터가 있으면, 게이트가 실제로 소비하는 필드명을 코드에서 직접 확인하고 그 대응관계를 결과 문서에 명시하라. 대응 `CONSENSUS.md` §3 항목225(신설). 상세 `.../audit_907959_2026-09-13/VERDICT.md` §3-3(A6)·§14(5).

206. ★★★**(2026-09-13, job 907959 결과 감사, claims-auditor, GPU 0) 요청 오류로 비어버린 출력끼리의 "불일치 0"은 결정성의 증거가 아니다 — 비교 함수가 오류 레코드를 등가류에 넣으면 죽은 arm은 자기 자신과 항상 일치한다.** `verdict.txt`의 `C within-arm TD1-TD2: 0`은 두 TD boot이 **같은 31건에서 똑같이 빈 출력**(서버 사망 후 `RemoteDisconnected`)을 냈기 때문에 나온 0이며, "TD가 결정적으로 재현됐다"의 증거가 아니다(항등식, 게이트 #9 계열 재발). 실무 규칙: 토큰-비교 등가류 계산에 오류 레코드가 섞여 있으면 그 등가류를 "결정성" 서술에 인용하기 전에 오류율을 먼저 확인하라. 대응 `CONSENSUS.md` §3 항목226(신설). 상세 `.../audit_907959_2026-09-13/VERDICT.md` §12(N-5)·§14(6).

207. ★★★**(2026-09-14, job 908020 결과 감사, claims-auditor, GPU 0) 사전등록의 "제출 명령"은 그 문서의 스코프 튜플을 실제로 실현하는지 한 줄씩 대조한 뒤에만 승인하라 — 하네스 기본값이 튜플과 다르면 명령 그 자체가 死因이다.** `rerun_prereg/PREREG_RERUN_2026-09-13.md` §9의 제출 명령은 `R2C_MODEL`/`R2C_ATTN_BACKEND`/`R2C_CTX` 세 env를 빠뜨렸는데, 하네스 기본값이 정확히 반대 모델(Zamba2-2.7B/triton/ctx4096)이었다 — 규칙층 감사 2회(rules·rev2) 모두 이 세 문자열을 0회 언급했고 rev2는 그 명령을 그대로 전사하며 승인했다. 결과 job 908020이 등록 밖 튜플로 돌았다(0.15250 GPU-h, 어떤 게이트도 진전 없음). 실무 규칙: 감사 체크리스트에 "등록 명령 ⊨ 등록 튜플" 항목을 넣고, 대조에 쓴 하네스 줄 번호를 판정서에 적어라(rerun rev3의 §6이 18/18 실현을 이 방식으로 확인한 것이 교정 사례). 대응 `CONSENSUS.md` §3 항목227(신설). 상세 `workspace/engine-port/results/r2_correctness/audit_908020_2026-09-14/VERDICT.md` §5.3(G-908020-1).

208. ★★★**(2026-09-14, job 908020 결과 감사, claims-auditor, GPU 0 — 교훈 89 확장) rev 개정 시 "불변"이라고 표에 적은 축이 실행 절차(제출 명령 등)에서 조용히 사라졌는지 확인하라 — 삭제는 diff에도 grep에도 안 잡힌다.** 올바른 제출 명령(세 env 포함)은 sha로 핀된 선행 문서(`newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:797-807`)에 이미 있었으나 rerun rev1/rev2로 복사되며 소실됐고, 같은 rerun 문서의 §0 표는 그 축을 "동일 | 동일 | 불변"으로 선언하고 있었다(문서 내부 모순). 실무 규칙: rev 계보를 이어받을 때 "불변"이라 표시한 축은 실행 절차(명령줄·sbatch·env 목록)에서도 실제로 나타나는지 직접 대조하라. 대응 `CONSENSUS.md` §3 항목228(신설). 상세 `.../audit_908020_2026-09-14/VERDICT.md` §5.3(G-908020-2).

209. ★★★**(2026-09-14, job 908020 결과 감사, claims-auditor, GPU 0 — 교훈 85 계열) 인적 실패점 하나를 명령줄 인자에서 제거했다면 같은 명령줄의 나머지 인자 전부에 같은 검사를 적용하라 — 처방은 국소, 결함은 계열이다.** newpair D11은 `--time` 인자에 대해 "빠뜨리면 조용히 망가지는 단일 인적 실패점"이라 진단하고 지시자로 내재화했으면서, 같은 명령줄의 모델 정체성 3개(env)에는 같은 검사를 적용하지 않았다 — 하네스 기본값이 정확히 반대 모델이었다. 실무 규칙: 명령줄의 한 인자를 fail-closed로 강화했다면, 그 강화 작업을 같은 명령줄의 다른 인자로 확장할지 여부를 반드시 검토·기록하라. 대응 `CONSENSUS.md` §3 항목229(신설). 상세 `.../audit_908020_2026-09-14/VERDICT.md` §5.3(G-908020-3).

210. ★★★**(2026-09-14, job 908020 결과 감사, claims-auditor, GPU 0 — 라벨 쇼핑 방지) 사전등록에 스코프 불일치 처분을 반드시 등록하라 — 넣지 않으면 "등록 실험이 돌았는가"가 라벨을 본 뒤의 사람 재량이 된다.** 908020 사건 이전에는 하네스·채점기 어디에도 §0-b 스코프 튜플을 강제하는 코드가 없었고, "등록 실험이 아니다"라는 판단 자체가 라벨(`PASS`)이 이미 보이는 상태에서 사람이 provenance를 읽어 내린 사후 판단이었다(구조가 라벨 쇼핑과 동일 — 908020이 `FAIL`이었어도 같은 논증이 같은 강도로 제기됐을지는 알 수 없다). 최소 요구: (i) provenance만으로 판정되는 기계적 일치 술어 (ii) 전용 라벨(`NO_VERDICT_SCOPE`, 게이트 실패 아님) (iii) 유한한 운영오류 재실행 예산 (iv) 불일치 job의 라벨·수치 병기 의무. 대응 `CONSENSUS.md` §3 항목230(신설). 상세 `.../audit_908020_2026-09-14/VERDICT.md` §2.3·§5.3(G-908020-4).

211. ★★★**(2026-09-14, job 908020 결과 감사 + rerun rev3, claims-auditor, GPU 0 — 양성통제의 소극판) 사전등록의 1차 판정 술어를 처치-이전 엔진 + 같은 기판의 기존 원자료에 먼저 먹여, 그 기판에서 검정력이 0인지 확인하라.** F-a1(4채널 무크래시)을 907100/907456(둘 다 수리 前, Zamba2)에 먹이면 4채널 전부 이미 참이었고, F-a2 문턱(6,245)은 907100 TD1의 12,369에 이미 초과돼 있었다 — 즉 이 사전등록은 어느 결과가 나와도 H1(grad-guard 수리가 OOM을 없앴다)을 닫지 못한다(RA3-1). 사전등록 §6-0은 이 검증을 907959에 대해서만 했고 실제로 돌아간(908020) 기판에 대해서는 하지 않았다. 실무 규칙: 판정 술어를 등록하기 전에 처치가 없는 기존 job에 그 술어를 먹여 "이미 참"이거나 "이미 초과"인 채널이 있는지 확인하라. 대응 `CONSENSUS.md` §3 항목231(신설). 상세 `.../audit_908020_2026-09-14/VERDICT.md` §5.3(G-908020-5); `workspace/engine-port/results/r2_correctness/rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md` §4(RA3-1).

212. ★★★**(2026-09-13, λ0 rev3 규칙층 감사, claims-auditor, GPU 0) 계획 단계의 "자격"(eligibility, "X가 될 수 있다")과 실행 단계의 "역할"(role, "X로 쓰인다")을 구분하라 — 자격 인증에 걸리는 가드가 그 자격이 아닌 역할로 쓰인 같은 셀에도 걸릴 수 있다.** λ0 rev3의 계획이 shape B의 rung 4개 **전부**를 "저측 후보가 될 수 있다"고 인증했는데, 저측 전용 배수 가드(`drain ≤ 2·L̂`)가 그 인증에 걸린 모든 셀에 무조건 적용돼 고측(포화) 증거로 쓰일 셀까지 측정 실패로 폐기했다(브래킷 판정 자체가 구조적으로 정의역 공집합). 실무 규칙: 계획 산출 필드가 "자격"인지 "역할"인지 이름과 문서에 구분해 적고, 자격 전용 가드를 역할 판정에 그대로 재사용하지 마라. 대응 `CONSENSUS.md` §3 항목232(신설). 상세 `workspace/engine-port/results/r2_eval/lambda0_prereg/VERDICT_lambda0_rev3_2026-09-13.md` §2·§8(G-λ3-1).

213. ★★★**(2026-09-13/14, λ0 rev3+rev4 규칙층 감사, claims-auditor, GPU 0) 결정 규칙(브래킷/사다리 판정)의 문턱 도달가능성은 등록된 모든 verdict 라벨 각각에 대해, pooled가 아니라 실제로 돌 시나리오별로 사전 계산하라 — 한 라벨만 계산하거나 여러 시나리오를 뭉뚱그리면 다른 라벨이 도달불가로 죽거나 표본 없는 시나리오가 발화한 것처럼 보일 수 있다.** rev3는 저측(κ) 도달가능성만 계산했고 그 계산이 만든 새 도달불가(`LADDER_TOO_HIGH`가 A·B 공통으로 영원히 도달불가)를 못 봤다. rev4의 도달가능성 assert는 7 시나리오를 pooled해 "4개 verdict 전부 정의역 비어있지 않다"고 했지만 시나리오별로는 5/7이 shape A에서 빈 정의역이었고 **실제로 도는 시나리오는 하나뿐**이었다. 대응 `CONSENSUS.md` §3 항목233(신설). 상세 `.../lambda0_prereg/VERDICT_lambda0_rev3_2026-09-13.md` §8(G-λ3-2); `.../VERDICT_lambda0_rev4_2026-09-14.md` §1(V4)·§8(G-λ4-3).

214. ★★★**(2026-09-13, λ0 rev3 규칙층 감사, claims-auditor, GPU 0) 같은 문서의 두 문장이 같은 대상(셀/조건)에 상호배타적인 수치 요구를 하고 있는지 대조하라.** λ0 rev3 §3.3("고측 rung은 배수 0.5–0.7×span[=120초]을 읽는다")과 §3.4("배수 ≤ 2·L̂=8.4초를 요구한다")가 같은 b_r3 셀에 대해 한 페이지 안에서 서로 배타적인 수치를 요구하고 있었다. 대응 `CONSENSUS.md` §3 항목234(신설). 상세 `.../VERDICT_lambda0_rev3_2026-09-13.md` §2(N3-2)·§8(G-λ3-3).

215. ★★★**(2026-09-13, λ0 rev3 규칙층 감사, claims-auditor, GPU 0 — 교훈 9 계열) 분지(branch) 존재를 증명하는 테스트에 손으로 만든 레코드를 쓰지 마라 — 실제 인증 경로에서 존재할 수 없는 셀을 픽스처로 쓰면 그 분지는 증명되지 않은 채 "통과" 표시만 얻는다.** `LADDER_TOO_HIGH`를 증명하는 테스트가 `LABEL._cell(r, 0.70)`을 썼는데 이 헬퍼는 `drain_ok=True`가 기본값이라, 실제 인증된 저측 rung에서는 존재할 수 없는 조합이었다 — e2e 배선 시험은 가드가 물 수 없는 유일한 구성(shape A)만 재현하고 있었다. 대응 `CONSENSUS.md` §3 항목235(신설). 상세 `.../VERDICT_lambda0_rev3_2026-09-13.md` §2·§8(G-λ3-4).

216. ★★★**(2026-09-13/14, λ0 rev3+rev4 규칙층 감사, claims-auditor, GPU 0) 변이(mutation) 하네스 CONTROL의 비공허성은 논증이 아니라 주입(injection)으로 보여라 — 실제 실행 경로(temp-dir 사본 등)를 인위적으로 깨서 CONTROL이 진짜로 실패하는지 확인해야 한다.** rev3는 `lambda0_mutation_check.py:156`이 CONTROL 결과를 즉시 덮어써 버려 죽은 CONTROL이었음을 지적만 했으나(死因은 아님), rev4는 실제로 아카이브 경로를 깨는 주입 실험으로 CONTROL이 `FAILS`·rc=2로 실패함을 **실행으로 확인**했다(긍정 사례, 게이트#113 자기 적용의 모범). 대응 `CONSENSUS.md` §3 항목236(신설). 상세 `.../VERDICT_lambda0_rev3_2026-09-13.md` §8(G-λ3-5); `.../VERDICT_lambda0_rev4_2026-09-14.md` §4-3(E3)·§8(G-λ4-6, 긍정 사례).

217. ★★★**(2026-09-14, λ0 rev4 규칙층 감사, claims-auditor, GPU 0) 직전 감사가 "제출 전 필수"로 등록해 둔 수리를 반영하지 않은 판본을 감사할 때는, 그 수리가 판정 분지를 바꾼다는 사실과 두 분지 각각의 결과를 실행 전에 등록하라 — 死因을 이미 아는 채로 무수리 분지를 고르면 반전-확인 死因(T2 계열)이 재발한다.** 907959 결과 감사 §13-D1이 λ0 rev3 제출 전 필수로 등록해 둔 F5 셀별 재계산 수리를 rev4가 반영하지 않았고, 그 수리를 넣으면 분지가 `ANCHORED→FALLBACK`으로 바뀌며 shape B 13격자점 중 5점·shape A 12점 중 3점의 라벨이 뒤집힘이 사후 계산으로 드러났다. 대응 `CONSENSUS.md` §3 항목237(신설). 상세 `.../VERDICT_lambda0_rev4_2026-09-14.md` §2(N2)·§8(G-λ4-2).

218. ★★★**(2026-09-14, λ0 rev4 규칙층 감사, claims-auditor, GPU 0) 변이 하네스의 모듈-변이 라우팅표는 그 모듈의 상수가 실제로 영향을 주는 모든 판정 산출물(reachability selftest 포함)로 보내라 — 라우팅에서 빠진 모듈은 사실상 무제한 escape를 허용한다.** `SELFTEST_OF`가 `plan`·`analyze` 변이를 reachability selftest로 보내지 않아, 감사자 독립 변이 19종 중 11종이 escape했고 그중 2종(`DRAIN_MODEL_TOL`, `MULT[B]` 최상단)은 shape B 지도 라벨을 실제로 이동시켰다. 대응 `CONSENSUS.md` §3 항목238(신설). 상세 `.../VERDICT_lambda0_rev4_2026-09-14.md` §4-3(E3)·§8(G-λ4-4).

219. ★★★**(2026-09-14, λ0 rev4 규칙층 감사, claims-auditor, GPU 0 — 게이트#179 계열 재발) 같은 상수를 두 파일이 각자 정의하고 있으면 죽은 쪽을 삭제하라 — 특히 그 주석이 이미 철회된 규칙을 적고 있을 때.** `lambda0_plan.py:118`의 `DRAIN_MODEL_TOL`은 죽은 상수이면서 주석이 아직 철회된 rev3 규칙을 인용하고 있었다. 대응 `CONSENSUS.md` §3 항목239(신설). 상세 `.../VERDICT_lambda0_rev4_2026-09-14.md` §6(F5)·§8(G-λ4-5).

220. ★★★**(2026-09-14, rerun rev3 규칙층 감사, claims-auditor, GPU 0) "블록이 비어 있다"를 게이트 술어로 쓰지 마라 — 빈 명령 출력은 파일에 줄을 남기지 않으므로 "그 다음 줄"은 언제나 그 다음 명령의 출력이다. 경계는 공백이 아니라 다음 마커의 존재로 등록하라.** rerun rev3의 C1 항목 (9)("`src_dirty:` 다음 줄이 비어 있음")는 `git status --porcelain`이 깨끗하면 0줄을 내므로 그 "다음 줄"이 항상 `nvidia-smi`의 출력이 되어, 직해하면 참 분지가 하네스 산출물 어디서도 성립하지 않는 死因이었다(rerun rev4가 "다음 줄이 `GPU 0: `로 시작한다"로 수정해 해소). 대응 `CONSENSUS.md` §3 항목240(신설). 상세 `workspace/engine-port/results/r2_correctness/rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md` §2·§9(G-RA3-1).

221. ★★★**(2026-09-14, rerun rev3 규칙층 감사, claims-auditor, GPU 0 — 게이트#110 심화) 재감사에서 직전 판정서 문안의 일부 항목만 재검증하면, 재검증에 성공한 항목이 나머지 미검증 항목까지 감사받은 것처럼 신뢰를 위조한다.** rerun rev3는 C1의 (7)·(10) 항목은 독립 재검증해 정정했지만 (9)는 "감사 문안에서 글자 그대로 승계"했고, 그 (9)가 바로 死因이었다 — 두 곳의 성공이 세 번째 곳의 무검증을 가려버렸다. 대응 `CONSENSUS.md` §3 항목241(신설). 상세 `.../VERDICT_rerun_rev3_2026-09-14.md` §2·§9(G-RA3-2).

222. ★★★**(2026-09-14, rerun rev3 규칙층 감사, claims-auditor, GPU 0) 철회된 판정서의 "등급"과 그 판정서가 등록한 "caveat 목록"은 별개다 — 등급을 철회할 때 caveat 중 어느 것이 함께 소실되는지 명시적으로 열거하라.** rerun rev2의 등급이 철회(A908-7)됐지만 그 철회 사유는 caveat RRC-1…13과 무관했는데, rev3 본문은 "RRC" 언급이 0회였다 — 결과적으로 RRC-1(§6-2-3 근거 문장이 거짓임을 이미 지적한 caveat)이 rev3에도 그대로 남아 있던 거짓 문장을 방치할 뻔했다. 대응 `CONSENSUS.md` §3 항목242(신설). 상세 `.../VERDICT_rerun_rev3_2026-09-14.md` §4(RA3-5)·§9(G-RA3-3).

223. ★★★**(2026-09-14, rerun rev3 규칙층 감사, claims-auditor, GPU 0) "이 라이브러리/설치본의 엄한 실패 지점(hard error)은 정확히 N개"라는 사실은 설치본에서 전수 열거해 등록하라 — 부분집합만 열거하면 처치층 코드의 우연한 커버리지가 가설 판정으로 둔갑한다.** rev3 §1.2(b)의 "inference-tensor 하드 에러는 정확히 둘"은 venv 실측으로 최소 4종임이 드러났고, 그중 하나(`Inference tensors do not track version counter.`)는 inference mode **안에서도** 발화해 완전성 근거가 거짓이었다. 대응 `CONSENSUS.md` §3 항목243(신설). 상세 `.../VERDICT_rerun_rev3_2026-09-14.md` §4(RA3-6)·§9(G-RA3-4).

224. ★★★**(2026-09-14, rerun rev3 규칙층 감사, claims-auditor, GPU 0) 서로 다른 기판(substrate)에서 같은 수치가 우연히 나왔을 때 "조합 공간이 조밀해서"로 설명하지 마라 — 밀도는 그 값의 실현 가능성만 설명할 뿐 왜 그 값이 반복됐는지는 설명하지 않는다.** 6,245가 907959(NemotronH)·908020(Zamba2) 양쪽에서 4항 부분합으로 실현 가능하다는 산술은 참이었지만, 실측 배치는 4항이 아니라 8-seq/7-seq였고 6,245로 가는 부분합이 72,489·35,597개나 돼 "부분합이다"라는 설명은 거의 무정보였다 — 진짜 근거는 "907100 TD1이 수리 前 같은 기판에서 12,369를 완주했다"는 사실 하나뿐이었다. 대응 `CONSENSUS.md` §3 항목244(신설). 상세 `.../VERDICT_rerun_rev3_2026-09-14.md` §4(RA3-3)·§9(G-RA3-5).

225. ★★★**(2026-09-14, rerun rev3 규칙층 감사, claims-auditor, GPU 0) fail-closed 스코프 가드를 하네스에 넣은 뒤 그 가드의 발화 자체를 provenance 확인 술어로 추가하는 것은 정보량이 거의 0이다 — 가드가 통과했다는 것은 부팅했다는 것과 거의 동치다. 남는 고유 정보만 명시하고 "독립 채널"이라 부르지 마라.** 가드가 fail-closed이므로 부팅한 모든 job은 자동으로 `scope_guard: OK`가 되고, 이 항목의 고유 정보는 spool-copy 검출 하나로 한정된다. 대응 `CONSENSUS.md` §3 항목245(신설). 상세 `.../VERDICT_rerun_rev3_2026-09-14.md` §4(RA3-8)·§9(G-RA3-6).

226. ★★★**(2026-09-14, rerun rev4 규칙층 감사, claims-auditor, GPU 0 — RA4-1) "이 조합은 GPU에서 한 번도 실행된 적 없다"는 주장을 쓰기 전에 같은 job의 대조 arm(control arm)을 먼저 조회하라 — 처치 축이 arm이면 대조 arm이 그 조합을 이미 돌렸을 수 있다.** rerun rev3 §6-5는 "`inference_mode`×`forward_native`(bare Parameter, n_groups≠1) 조합은 GPU에서 한 번도 실행된 적 없다"고 적었으나, 같은 job(907959)의 legacy 2 boot(NemotronH, n_groups=8)이 바로 그 조합을 이미 무오류로 완주하고 있었다. 대응 `CONSENSUS.md` §3 항목246(신설). 상세 `workspace/engine-port/results/r2_correctness/rerun_prereg/VERDICT_rerun_rev4_2026-09-14.md` §4(RA4-1)·§7(G-RA4-1).

227. ★★★**(2026-09-14, rerun rev4 규칙층 감사, claims-auditor, GPU 0) 개별 예외 문자열들을 하나의 "계열(pattern)"로 일반화할 때, 그 계열의 토큰을 원래 열거가 이미 불완전했던 같은 소스에서 다시 뽑지 마라 — 설치본 문자열 전수에서 재유도하라.** rev4가 E5 트리거를 열거에서 계열(정규식)로 일반화했지만, 설치본(`libtorch_cpu.so`)의 inference-tensor 하드 에러 중 최소 1종(`A view was created in inference mode and is being modified inplace in normal mode.`)을 여전히 놓쳤다. 대응 `CONSENSUS.md` §3 항목247(신설). 상세 `.../VERDICT_rerun_rev4_2026-09-14.md` §4(RA4-2)·§7(G-RA4-2).

228. ★★★**(2026-09-14, rerun rev4 규칙층 감사, claims-auditor, GPU 0) 파생 집계(기저율·백분율 등)를 철회할 때는 그 파생물을 문서 전체에서 grep해 함께 철회하라 — 국소 철회는 등록된 처분절이나 자기 적용 절에 원문 그대로 살아남아 다시 인용된다.** "61–68%"·"23–26/38" 철회가 문서 전역에 적용되지 않아 등록 처분절(`:783`)과 자기 적용절(`:1428`)에 그대로 생존해 있었다. 대응 `CONSENSUS.md` §3 항목248(신설). 상세 `.../VERDICT_rerun_rev4_2026-09-14.md` §4(RA4-4)·§7(G-RA4-3).

229. ★★★**(2026-09-14, rerun rev4 규칙층 감사, claims-auditor, GPU 0) 사전등록의 "제출 전 커밋 목록"은 작성 시점이 아니라 감사 시점의 `git status`로 재생성하라 — 이미 커밋된 항목이 목록에 남아 있으면 정작 sha로 핀한 원자료 디렉터리가 목록에서 빠질 수 있다.** §9-3의 11항목 커밋 목록 중 7항목은 이미 커밋돼 있었던 반면, §0-c가 sha로 핀한 `job_907959/`는 목록에 없고 untracked 상태였다. 대응 `CONSENSUS.md` §3 항목249(신설). 상세 `.../VERDICT_rerun_rev4_2026-09-14.md` §4(RA4-8)·§7(G-RA4-4).

230. ★★★**(2026-09-14, rerun rev4 규칙층 감사, claims-auditor, GPU 0) 승계 사슬의 "전방 강제"(다음 회차에 반드시 편입해야 할 항목) 조항은 자기 문서의 요약절이 실제로 등재한 항목 수와 대조하라 — 총 caveat 수보다 강제 조항이 적으면 그 차이만큼 무강제 누수가 생긴다.** §11.1은 65건 + 전사 의무 2건을 이 회차에 의무화했지만 RR-14(다음 회차 강제)는 RR 19 + RA3 12 + RRC 13 = 44건만 요구해 **40건이 무강제**로 남았다. 대응 `CONSENSUS.md` §3 항목250(신설). 상세 `.../VERDICT_rerun_rev4_2026-09-14.md` §4(RA4-6)·§7(G-RA4-5).

231. ★★★**(2026-09-14, rerun rev4 규칙층 감사, claims-auditor, GPU 0 — 긍정 사례) 참 분지가 공집합이던 술어를 수리한 뒤에는 교체 문안을 실제 아티팩트 전수(음성대조 포함)에 먹여 참/참/참/거짓 같은 구체적 출력으로 도달가능성을 보여라 — 논증만으로 끝내지 마라.** rerun rev4는 C1 (9)의 대체 문안을 907959·908020·907100(참) + 907456(src dirty, 거짓 — 음성대조)에 실제로 먹여 도달가능성을 실행으로 확인했다(rerun rev3의 死因을 감사자가 직접 실행해 닫은 모범 사례). 대응 `CONSENSUS.md` §3 항목251(신설). 상세 `.../VERDICT_rerun_rev4_2026-09-14.md` §1(U1)·§7(G-RA4-6).

232. ★★★**(2026-09-14, job 908179 결과 감사, claims-auditor, GPU 0 — 긍정 사례) "세 boot이 바이트 동일하다"를 보고할 때는 그 계측기가 같은 job 안에서 arm 차이를 해상함을 먼저 보여라 — 동일성은 "차이가 없다"와 "계측기가 눈이 멀었다"를 구별하지 못한다.** job 908179의 `peak(TD1)=peak(TD2)=peak(L2)`(차 0 B)는 **9/36 epoch에서 완전 arm 분리(+5.17…+20.08 MiB)가 같은 job 안에서 관측**된 뒤에야 "계측기가 5 MiB 수준에서 arm 차이를 해상할 수 있다"는 것이 실증돼, 0 B 일치를 "측정"이라 부를 근거가 생겼다(이 실증이 없었다면 0 B는 계측기 맹점의 산물일 수도 있었다). 대응 `CONSENSUS §3` 항목252(신설). 상세 `workspace/engine-port/results/r2_correctness/audit_908179_2026-09-14/VERDICT.md` §6·§16(G-8179-1).

233. ★★★**(2026-09-14, job 908179 결과 감사, claims-auditor, GPU 0) 사전등록의 "동시에 움직이는 축" 목록에 노드·물리 GPU를 넣어라 — 세지 않은 축은 닫을 대상으로도 등록되지 않는다.** rerun rev4 §10은 이동 축을 3개(mixin sha·하네스 sha·계측 플래그)로 셌으나 907959→908179 사이에 노드·물리 GPU(gpu38→gpu40, 다른 UUID)도 함께 이동했다. 그 축의 메모리 채널은 두 job 5 boot의 `avail mem` 사다리가 완전 동일함을 확인해 닫았지만, 애초에 그 축의 존재 자체가 사전등록에 없었다. 대응 `CONSENSUS §3` 항목253(신설). 상세 `.../audit_908179_2026-09-14/VERDICT.md` §8.1·§16(G-8179-2).

234. ★★★**(2026-09-14, job 908179 결과 감사, claims-auditor, GPU 0) arm-비교 추정량의 정의역이 arm마다 다른 처치 단위(예: 다른 크기의 배치) 위에 놓일 수 있는지 검사하라 — "정의역이 비어있지 않다"만으로는 부족하다.** F-b2의 정의역이 boot마다 서로 다른 배치를 지목했고(L1은 10,630-token, 나머지 셋은 10,125-token) 그 한 boot의 505-토큰 초과분이 arm-간 차이(`Δ`) 전량을 만들었다. 실무 규칙: ∅ 처분만으로는 부족하고 "같은 `#new-token`을 비교하고 있는가"도 처분 조건에 넣어야 한다. 대응 `CONSENSUS §3` 항목254(신설). 상세 `.../audit_908179_2026-09-14/VERDICT.md` §5.3·§16(G-8179-3).

235. ★★★**(2026-09-14, job 908179 결과 감사, claims-auditor, GPU 0) 정렬 검사를 등록할 때 그 검출점이 실현된 정의역 안에 실제로 있는지 확인하라 — 검출력이 등록 문서가 가정한 자리에 없으면 검사는 아무것도 하지 않는다.** rerun rev4의 2채널 epoch↔ordinal 정렬 검사는 907959의 유일한 검출점(ordinal 38, `#new-seq: 4`)에 의존했는데, job 908179는 정의역이 ordinal 36에서 끝나 그 검출점이 존재하지 않았다 — `#new-seq`가 정의역 전체에서 1이라 어떤 상수 시프트도 못 잡는 검사가 됐다(등록된 최악 케이스가 실제로 실현). 대응 `CONSENSUS §3` 항목255(신설). 상세 `.../audit_908179_2026-09-14/VERDICT.md` §4.2·§16(G-8179-4).

236. ★★★**(2026-09-14, job 908179 결과 감사, claims-auditor, GPU 0) 백엔드를 바꾸면 판정 규칙 본문의 정당화 문장도 함께 재유도하라 — 조건 자체가 충족돼도 그 조건이 "왜 결정적인가"의 근거는 이식되지 않을 수 있다(교훈88 계열, 규칙층 판).** O1 프로토콜의 독립성 조건("구성상 결정적")은 `triton_attention_num_kv_splits`에서 유도됐는데 job 908179는 flashinfer 기판이다. 조건 자체는 8/8 만족해 PASS는 영향받지 않지만, "왜 구성상 결정적인가"의 정당화는 이 기판에서 한 번도 재유도된 적이 없다. 대응 `CONSENSUS §3` 항목256(신설). 상세 `.../audit_908179_2026-09-14/VERDICT.md` §12(T13)·§16(G-8179-5).

237. ★★★**(2026-09-14, job 908179 결과 감사, claims-auditor, GPU 0 — 게이트#113 명령줄 판) "이 처방은 값싸다"고 쓰기 전에 하네스가 그 노브를 실제로 삼키는지 확인하라.** `PDMUX_WORKER_GRAD_GUARD=none` 대조 arm 제출은 `r2_correctness.sbatch:216`의 일괄 `PDMUX_*` unset 루프에 조용히 먹혀 값이 전달되지 않는다 — 노브 추가 배선 없이는 "다음 job 하나면 CONFIRMED로 올릴 수 있다"는 처방이 성립하지 않는다. 대응 `CONSENSUS §3` 항목257(신설). 상세 `.../audit_908179_2026-09-14/VERDICT.md` §8.5·§16(G-8179-6).

238. ★★★**(2026-09-14, engine-porter — E5 계열 RA4-2 처리[정본 인용 등록성 검사에서 부수 발견], GPU 0) 감사자의 교정 처방 자체가 실효 항등식일 수 있다 — 정규식은 실행 엔진까지 고정해 검증하라.** RA4-2가 처방한 `RuntimeError.*(?i:inference[ _](tensor|mode))|RuntimeError.*InferenceMode`의 `(?i:…)`는 PCRE 인라인 플래그이고 POSIX ERE에 없다 — 이 클러스터의 `/usr/bin/grep`(GNU grep 3.6) `-E`에서는 실측 12개 양성 중 **4개만** 잡히고(첫 대안이 리터럴 불일치가 되어 `RuntimeError.*InferenceMode`만 살아남음), 대안을 하나로 줄인 형태로는 **0/12**(아무것도 안 잡고 "E5 미발화"라고 보고할 수 있는 형태)다. 같은 명령이 셸에 따라 12 또는 4를 낸다(이 세션 셸의 `grep`은 `(?i:)`를 받는 ugrep 7.8.4 래퍼) — 감사자가 손으로 돌린 값이 어느 쪽인지는 기록에 없다. 검증된 대체형 `RuntimeError.*[Ii]nference[ _]?([Tt]ensors?|[Mm]ode)\b`는 `grep -E`/`grep -P`/`python re` 세 엔진 전부 12/12 양성·6/7 음성 침묵으로 동값이다. 부수 정정: RA4-2는 등록 계열이 놓치는 하드 에러가 1종이라 했으나 실측은 **3종**(X5·X6·X7, 전부 실행 확인). 대응 `CONSENSUS §3` 항목258(신설). 상세 `workspace/engine-port/results/r2_correctness/E5_FAMILY_RA4_2_2026-09-14.md`(§4·§5·§6) + 재현체 `e5_family_probe.py`(같은 디렉터리).

239. ★★★**(2026-09-14, engine-porter — 전체 테스트 스위트 감사[W4 커밋 직후], GPU 0) 변화를 통제하는 대조(control)를 그 변화가 움직이는 기준점(`HEAD`)에 걸면, 그 변화가 커밋되는 순간 대조가 자살한다.** `tests/test_lambda_star_per_shape.py`의 개정-전 대조는 `git show HEAD:…`를 exec해 개정 전 코드를 얻는데, W4가 커밋(`8507cee`)되는 순간 `HEAD`가 **개정 후** 파일이 돼 `ImportError`로 2 에러가 났다. 저자의 `try/except TypeError → skipTest` 가드는 모듈이 호출 전에 죽으므로 이 실패를 못 잡는다. 수리 = 고정 커밋 핀(`bddff6a`). ★부수 사실: **이 세션 시작 시점에 전체 스위트가 이미 red였다**(645 tests, errors=2)는데 아무도 보고 있지 않았다. 대응 `CONSENSUS §3` 항목259(신설). 상세 `workspace/engine-port/tests/test_lambda_star_per_shape.py:660,690,693-694,717-718`(읽기 확인만, 이 패스는 이 파일에 손대지 않았다 — engine-porter가 별도로 수리 중).

240. ★★★**(2026-09-14, engine-porter — 규율 도구 부작용 감사, GPU 0) 규율 검사 도구 자신이 read-only라고 전제하지 마라 — 검사가 아티팩트를 다시 쓸 수 있다.** `presubmit.py`가 호출하는 `design_reachability.py`가 각 spec의 `out` 경로를 매 실행마다 다시 쓴다(`scripts/discipline/design_reachability.py:161-162`) — 이번 4회 실행이 `m4r_confinement/reachability_verdict.json`·`tc1_model_attrib/reach_verdict_rev3_A.json`을 재작성했다(이번엔 `git status` 대조로 바이트 동일 확인됐으나, 다른 세션의 미커밋 작업을 파괴한 전례가 있다). 대응 `CONSENSUS §3` 항목260(신설). 상세 `scripts/discipline/design_reachability.py:161-162`.
