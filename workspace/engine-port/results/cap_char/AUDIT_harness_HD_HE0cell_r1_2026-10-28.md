# 하네스층 감사 r1 — PREREG_HD_HE0cell rev0.1 + policy_harness (2026-10-28)

- 감사자: claims-auditor(읽기 전용) · 회차 1/2 · **판정 봉인 불가**(死因 D1–D4, MAJOR M1–M7). r2(최종)는 수리 diff만.
- CPU: `tests.test_policy_harness` 44 OK, 변이 31/31 사망 — 그러나 D2는 테스트 통과 상태의 실제 결함(변이 미대표 영역).

## 死因
- **D1 엔진 digest 불일치** (`ph_job.sh:43`, PREREG:30): 하네스는 `costterms_harness/ENGINE_DIGEST.txt`(99bbef91…)와 대조, `job_entry.sh:119-141`은 PDMUX_COMMIT에서 설치 → 96ae3afa 이후면 3ab5295e… → 전 Job `ABORT_ENGINE_DIGEST`(rc 13). 수리: `policy_harness/ENGINE_DIGEST.txt` 신설(최종 트리 값), ph_digest 대상 포함, Q-7 확정. caveat: "OFF 경로 동일성은 정적 확인이며 측정되지 않았다. 비용 항(99bbef91) 수치와 결합하지 않는다."
- **D2 CAP trace가 arm마다 다름** (`ph_trace.py:127` `sk=(tag,seed,rate)`, tag에 D 포함 `ph_spec.py:142`): rate 1.0·seed 201에서 D16 첫 도착 0.238 s·522 tok vs D44 1.304 s·186 tok, hash 5개 상이 → §2:37 위반, λ*_best-static이 독립 잡음 최댓값으로 상향 편향 → HE0·H-D rate로 전파. 수리: `sk=(workload, seed, rate)` + CAP 테스트·변이.
- **D3 O-P6 미결 + 배선 불일치**: `plan_cap`(`ph_spec.py:145`)·`plan_hd`(`:230`)는 sticky 고정, `plan_eval`(`:175`)만 addendum 읽음. sticky는 prefill 휴지 중에도 D 유지(mixin:1163-1172), 동적 arm은 split prefill 있을 때만 제어하고 그 외 108-SM(mixin:1153-1157) → "sticky static 대 동적"은 변수 동시 변경(confound 10). 수리: CAP 전에 정해야 하므로 ph_spec 봉인 상수. **권고: 전 Job 비sticky**(동적과 같은 fallback, KISTI static d44도 비sticky), engine-porter가 `R2_POLICY=fixed` 비sticky 의미 확인·smoke. caveat: "static D는 co-residence 중의 목표값이며 실현 할당이 아니다."
- **D4 절단 용량을 λ*로 사용** (`ph_verdict.py:235-250`): ladder에 실패 rate 없으면 맨 위 rate가 λ* → q=1.0이 용량 이하 → H=0 항등에 가까움(교훈 9). 수리: argmax D가 `CAPACITY_BRACKETED`일 때만 인정, 아니면 ladder 이동 후 CAP 재실행 규칙 등록, `λ*=None` 시 `he0_rates` 사망 방지.

## MAJOR
- **M1** connector 상한 100(`ph_runner.py:80`, `ph_loadgen.py:55`): HI=1.5λ*면 구간 끝 시스템 내 요청 ≈67(+running 48) [유도] → 클라이언트 대기 누적, 동적 arm 입력 왜곡(confound 7). 수리: `connector_limit=0` + 테스트.
- **M2** switch_count·체류를 누적 로그 전체로 계산(`ph_runner.py:96-114,171-173`): warm-up·이전 run 포함, 체류는 boot부터. TUNE 두 번째 run 오염 → `select_tune` 동률 깨기 오염. 수리: run 시작·끝 byte offset으로 자르고 체류는 run 시작부터.
- **M3** TUNE 동적 arm seed별 재부팅 없음(`ph_spec.py:163-167`, §7-2) vs EVAL 재부팅(§4-4). 수리: boot 12개, 약 +26 min(206 min, cap 247 안).
- **M4** W3 SLO floor를 다른 Job(CAP2)에서 addendum으로(`ph_spec.py:203,213`) vs §2:35 "같은 Job" — confound 11. 권고: CAP2 floor 봉인 상수로 본문 수정, HD Job 자체 floor는 공변량 기록.
- **M5** correctness 검사 미구현(`ph_runner.py:7` 문서만), warm-up 10요청 ≈5 s vs 등록 60 s(§2:39). 수리: 첫 요청 출력 길이 검사 → 실패 시 boot INVALID, warm-up 60 s.
- **M6** 정상성 S1–S3 미구현(§3-3:67). S3은 runtime_snapshot 필요하나 Q-5 금지 → S1·S2 구현, S3은 결정 30 표 행.
- **M7** trace_hash에 arm 의존 request_id(`ph_trace.py:105,122`) → EVAL 같은 내용·다른 hash. 수리: arm 무관 내용 hash, 분석에서 arm 간 동일성 대조.

## MINOR
m1 "더 넓은 CI"를 min lo/max hi envelope로 구현(`ph_verdict.py:204-205`) → envelope로 등록 · m2 §12:214 "trace_loadgen.py 변경" 삭제(4aff6b30 원복, sha 8da91574…=rev12) · m3 §12:213 "KISTI bench_serving 자연 EOS"는 틀린 듯(`sharegpt_vary_bench.sbatch:44-50`에 `--disable-ignore-eos` 없음, 기본 ignore_eos=True — v0.5.10 재확인) → W1 고정 출력은 KISTI와 같음 · m4 HE0 라벨이 전체 goodput 사용(`ph_verdict.py:116`) 명시 등록 + LO/HI phase별 출력 · m5 `DRAIN_TIMEOUT` 의미 정의 · m6 H-D 게이트 5 항목(가상 oracle 전환 수·체류, p50/p95/p99)과 연속 변형 warm 10% 제외값 미계산 · m7 static 체류를 비sticky에선 "목표값"으로 · m8 ShareGPT sha 미고정(`ph_job.sh:47-48`), Job 간 `w1_lengths` sha 검사 · m9 static arm 누락 시 조용히 best-of-4(`ph_verdict.py:114-117`), boot 실패 rc 0(`ph_runner.py:248-251`).

## 빠진 변이
CAP trace tag 의존 · connector 상한 · 로그 자르기 · TUNE seed별 재부팅 · CAP/HD sticky 배선 · bracket 요구 · correctness 검사 · phase 완료 귀속 · `req_good` 빈 ITL→True · hash arm 독립성 · ENGINE_DIGEST 핀.

## 반전 시험
BELOW (b)만 요구: 귀무·5 arm 최댓값·n=4에서 승자 편향 ≈0.58σ < 구간 반폭 ≈2.25σ [유도] → 반전 없음, caveat · widest vs envelope: UNRESOLVED 쪽으로만 · h·동률 1%·ln 1.03: 원자료 없어 계산 불가 → 봉인 전 상수.
규칙층 caveat(인용 금지): "H-D의 H는 요청률 가중이라 W1 phase가 지배한다. λ 비 약 10:1이면 W3 비중 약 9%, W3 phase만의 여지가 PRESENT를 만들려면 약 33% 이상 [유도, §7 가정 λ 3/0.3]."

## 봉인 전 닫을 [미정] — 권고
O-P6 전 Job 비sticky(ph_spec 상수) · Q-7 최종 digest 전용 파일 · O-P5 rmin 산식 + bracket 실패 시 재실행 규칙 · W3 floor CAP2 상수 · O-P3 h=1.5(caveat "KISTI HI/λ* 배율 미상, 한 점") · O-P4 {2,3,4}×{3,6} · O-H1 {120,600} s · connector 0 · warm-up 60 s · W3_MAX_NEW 1024 · context 20480 · 판정 상수 ln 1.03·n=4·확장 규칙·CAP seed 2. 열린 채 허용: O-H3(미포함), q=0.8(조건부).

## 확인된 것
t(n−1) 짝 비교·bootstrap 없음 · 교차적합 구조 · 판정표·cliff N-1 · 게이트 7·phase 도착 기준·ITL p95 술어 · 사이클 1·seed 분리 · ph_loadgen이 봉인 trace_loadgen 무수정 동작 · Job 하드캡 ≤ 5.2 h · 결정 30 표 R-1·R-2·금지 문안.
