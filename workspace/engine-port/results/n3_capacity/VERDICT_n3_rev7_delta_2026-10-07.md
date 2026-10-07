# 판정서 — N3 사전등록 rev7 초안 규칙층 델타 감사 (2026-10-07)

감사자: claims-auditor(읽기 전용, GPU 0, 성능 판정 0, Claim 등급 변경 0). 대상: 미커밋 작업 트리
(`git diff HEAD`[=6113723] — `n3_plan.py`, `PREREG_N3_CAPACITY_2026-10-04.md` 머리말·§17-6,
`tests/test_n3_capacity.py`), 근거 `REV7_DRAFT_NOTES_2026-10-07.md`, `DRYRUN_rev7_2026-10-07.txt`,
원자료 3회차 Job `pdmux-n3-capacity-38ed3f4-20261006092813` `TIMING.tsv`·floor jsonl·`marks_floor_head.txt`,
2회차 Job `pdmux-n3-capacity-79999aa-20261006081638` floor 자료, E2 VESSL r2 `TIMING.tsv`(harness 대조용).

## 0. 판정

**`GO-with-caveats`** — 死因 0(N1–N4 해당 없음). 필수 수리 **R-1…R-6**(봉인 전, 전부 문서·테스트 배치·
기술 문자열 층; 측정·채점 경로 무영향). 등록 caveat **N3C-34…41**(아래 §5 문자 그대로 승계).
⚠ 이 판정은 "예산 모형 문언이 실행을 허용한다"는 뜻이다. 예보가 검증됐다는 뜻도, k=5/10이 해결된다는 뜻도,
게이트 #6을 닫는다는 뜻도 아니다.

死因이 없는 이유(요약): rev7이 바꾼 상수의 소비처는 `bench_s`→`unit_cost_s`→`need_s`/`simulate`(예산 입장·
selftest·예보표)뿐이다(`grep` 전수, `n3_analyze.py`·채점 경로 0곳). 예산은 결측(미시도/절단)만 바꿀 수 있고,
결측은 `capacity_rule`(n3_analyze.py:206–225)에서 **읽어야 할 rate의 n<4 → `UNRESOLVED_N_BELOW_4`**,
예산 절단 unit은 무효(n3_analyze.py:87, 러너 462–500)로 떨어진다 → 용량 라벨 값은 어떤 cap·상수에서도 바뀌지
않는다. 반전 시험(§2)에서 등록 판정을 뒤집는 자유도를 만들지 못했다.

## 1. 질의별 판정

| # | 질의 | 판정 | 근거 수치 |
|---|---|---|---|
| Q1 | 같은 41 unit 적합·검증 = 항등식? | **부분 항등식 `CONFIRMED`.** 0.997·중앙 +0.46%는 41 unit 중앙값으로 적합한 상수를 같은 41 unit에 대는 것이라 증거력이 거의 없다(교훈 9). 감사 독립 시간 분할(같은 Job 안): 전반 20 적합(170/23.1)→후반 21 예보 합계 **1.003**(범위 −5.0…+4.3%), 후반 적합(171/20.6)→전반 **0.988**(−6.7…+1.6%). 부대비용 처음 10 unit 중앙 169.0 vs 마지막 10 unit 169.5(Job 내 drift 없음). **Job 간·GPU 간 근거: 2회차 floor_head 1 unit뿐**(admit→health 167 vs 162 s, floor bench 창 50 vs 49 s) — **2·3회차는 같은 GPU `c36b3c0c`**. 다른 GPU에서의 예보는 근거 0. harness가 다르면 부대비용 자체가 다르다(E2 VESSL r2: 중앙 139 s, admit→health 41 s vs N3 170 s/104 s) | 위 |
| Q2 | 17,100 + 44/48 포기가 #261 재발? | **구조적 재발 가능 `CONFIRMED`, 라벨 오염 `REFUTED`.** UNITS 순서상 round 4(seed 4589)가 마지막·오름차순이라 어떤 예산 부족도 4589의 상위 rate에 몰린다. 스트레스 코너 44/48의 결측 = 4589 × 0.61·0.69·0.77 + floor_tail(3회차와 같은 모양). 3회차 실측 총시간 재생(결측 5 unit은 모형 보통 코너, floor_tail 219 s): cap 17,100에서 균등 둔화 ×1.08까지 48/48, ×1.10 floor_tail만 결측, ×1.15 0.69·0.77 결측. 결측 시 경로는 fail-closed(위 §0) — k=10이 다시 `UNRESOLVED`가 될 수 있을 뿐 잘못된 용량 라벨이 되지 않는다 | 재생 표는 §2 |
| Q3 | preflight 600→300 정당·오거부 위험? | **오거부 위험 `REFUTED`** — `PREFLIGHT_ALLOW_S`는 `simulate`의 시작 오프셋일 뿐 러너가 읽지 않는다(러너 입장은 `T0`=job_entry `start_utc`부터 실제 시계, n3_capacity.sh:90–95). 300은 예보를 낙관 쪽으로 옮긴다: 600이면 스트레스 코너 43/48, 300이면 44/48(보통 코너는 둘 다 48/48). 근거 n=2·같은 GPU(N3C-39) | |
| Q4 | W-5 대체가 원안 의도 구현? | **다른 술어 `CONFIRMED` + 초안의 전제 서술 거짓 `CONFIRMED`(R-1).** 초안은 "시계 앵커가 기록에 없다"고 했으나 러너는 unit마다 `marks_<unit>.txt`에 CLOCK_MONOTONIC `bench_start`/`bench_end`를 쓰고(n3_capacity.sh:453,459) 서버 이벤트와 같은 시계인지 `ABORT_CLOCK`(172)로 검사한다; 분석기도 이 표식으로 스냅샷을 자른다(n3_analyze.py:192–193). 3회차 floor: bench_start 1831996.564, bench_end 1832046.971, bench_serving 시험 요청 span start 1832012.777 — **표식 창 안이다**. 즉 원안 포함 술어는 이 Job에서 시험 요청 혼입을 **거부하지 못했을 것이다**(표식이 bench_serving 프로세스 전체를 감싼다). rev7 길이 술어는 이 경우 더 강하다(시험 요청 혼입 → gap < 0, `TestW5Rev7`로 KILLED). fail-open: 마지막 16이 전부 한 덩어리로 bench 밖에 있는 경우는 길이로 못 잡는다(비현실적, 원안이 잡는 유일한 경우). false-reject 여유: 실측 gap 0.89/0.88 s(n=2, 같은 GPU) vs 2.0 s; 첫 floor span 탈락+이물 혼입 시 gap ≈ 0.88+1.76 = 2.64 s → 거부(여유 0.64 s). shape A(floor 110 s, 다른 decode 꼬리)에는 미검증 | |
| Q5 | 재개 = `ABORT_RESUME`? 새 Job 권고 정합? | **rev7이면 `ABORT_RESUME` `CONFIRMED`**(러너 213–215: 등록 해시열 전체 diff). 감사 독립 확인: 등록 대상 파일은 38ed3f4→6113723에서 **바이트 불변**(diff stat 0) → **rev6 그대로 HEAD에서 재개하면 digest 조건 (b)는 통과한다**(Cluster 디렉터리 보존은 미확인). ★추가 사실: 이 판정서 이름 `VERDICT_n3_rev7_delta_*.md`가 등록 glob `VERDICT_n3_*.md`에 걸린다 → **이 파일을 커밋한 트리에서 제출하면 rev7을 버려도 rev6 재개가 `ABORT_RESUME`**이다. "새 Job 48 unit" 권고는 N3VR-5·6과 정합(재개 셀이 k=10을 직접 정하는 Job 혼합을 피한다). 단 새 Job도 Job 축 σ는 주지 않는다(Job n=1, E2VR-3 계열) | |
| Q6 | 변이 하중·무변이 대조 | 하중 있음, 단 결함 2. 되돌림 변이: overhead→121 단독 중앙 −15.7% KILLED, bench_fixed→0 단독 −7.1% KILLED, floor 40 KILLED(|40−49|>3), cap 14,400 KILLED(보통 42/48), 무변이 대조 통과(모듈 모드 79/79 OK). ① **보상 등가 변이(overhead 193·fixed 0)는 `TestRev7Budget` 전부 생존**(중앙 +0.46% 동일) — 부대/고정비 분할은 총시간 시험으로 식별되지 않고 스트레스 코너 고정점 44(→43)만이 잡는다(교훈 254). 시험 통과 대역: overhead 160–175, fixed 15–30. ② **`TestRev7Budget`·`TestW5Rev7`(13개)가 `if __name__ == "__main__": unittest.main()`(1345행) 뒤에 정의돼 스크립트 실행 시 조용히 빠진다**: `python3 tests/test_n3_capacity.py` = **66 tests OK**, `python3 -m unittest test_n3_capacity` = 79 OK(감사 실측). "79/79"는 모듈 모드에서만 참(R-2). `test_resume_..._digest_gate` 는 러너 diff를 실행하지 않는 사실 고정(문자열 존재 + 현재 해시 ≠ 등록 해시 — rev7 편집이 있는 한 항상 참) | |
| Q7 | 3회차 k=3 + 새 Job 결합 | 허용/금지 §4 | |

## 2. 반전 시험 표

원자료: 3회차 `TIMING.tsv`(43행, warm 총시간 41), floor jsonl·marks(2·3회차). 격자는 등록·초안이 스스로 제시한
범위만 썼다(새 격자 없음). 추정량은 초안의 정의(총시간 = 다음 admit − admit, 부대 = 총시간 − bench 창) 그대로.

| 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 |
|---|---|---|---|
| `HARD_CAP_S`(사용자 결정) | 16,200 / 17,100 / 18,000 | 라벨 값 **불변**(결측→UNRESOLVED·절단→무효). **preflight 분기는 바뀐다**: 16,200이면 스트레스 코너 41 < 고정점 44 → selftest·`TestUnits` 단언 실패 → `ABORT_SELFTEST n3_plan`(exit 4); 1.05 둔화 테스트도 실패. 18,000은 46 ≥ 44 통과 | 모형 48/48 유지 최대 균등 둔화: 16,200 ×1.020 · 17,100 ×1.082 · 18,000 ×1.143. 그리드 완결(floor_tail만 결측) 한계 ×1.034/×1.097/×1.159 |
| 부대비용 추정량(중앙 vs min/p90/max) | 163–192 s × fixed 17.9–28.3 s | 보통 코너 48/48 유지(192×28.3에서만 47), 스트레스 코너 46→40. 고정점 44가 같이 움직여야 함(N3C-37). 라벨 무영향 | 위 12점 격자 |
| `PREFLIGHT_ALLOW_S` | 224 / 300 / 600 | 러너 무영향. 스트레스 코너 44/44/43 | 위 |
| W-5 ON/OFF, slack | OFF / ON(2.0) | 실측 floor 2건 모두 `valid` 불변 | gap 0.89(2회차)·0.88(3회차) |
| 재생 균등 둔화(실측 총시간 기준) | ×1.00–1.15 @17,100 | 결측 위치만 이동(4589 상위 rate) — 라벨 경로 fail-closed | ×1.08 48/48 · ×1.10 47 · ×1.15 45(0.69·0.77·tail 결측) |

**반전 0건 → 死因 0.** 단 cap 대안은 "끼워 넣기"가 아니다(R-5).

## 3. 필수 수리 (봉인 전)

- **R-1 (W-5 전제 정정 — 출처 허위, 교훈 80/252)**: PREREG 머리말 rev7 줄·`n3_plan.py` W-5 주석·notes §4의
  "시계 앵커가 기록에 없다"를 삭제하고 사실로 교체: CLOCK_MONOTONIC `marks_<unit>.txt` 표식이 있고, 원안 포함 술어는
  3회차에서 시험 요청(1832012.777 ∈ [1831996.564, 1832046.971])을 거부하지 못했을 것이며, rev7은 그와 **다른 길이
  술어**로서 이 혼입을 잡는다. 사용자 결정 (iv)는 이 정정된 사실로 다시 제시한다(선택지: 길이 술어 단독 / 포함+길이
  병용 / OFF). 켤 경우 `floor_from` 결과의 `"rule"`에 W-5 적용을 표기(현재 항상 "rev6").
- **R-2**: `TestRev7Budget`·`TestW5Rev7`를 `if __name__ == "__main__":` 위로 옮긴다. 테스트 수 서술에 실행 방식을 붙인다.
- **R-3 (코드 내 거짓 기술 문자열)**: `budget_table()`의 `note`("per-boot overheads are λ0 VESSL measurements (E2 A13
  §5-1)", "preflight 600 s is UNMEASURED for N3")가 rev7에서 거짓이고 Job의 `BUDGET_TABLE.json`에 그대로 찍힌다.
  rev7 출처(3회차 Job TIMING, n·같은 GPU)로 교체.
- **R-4 (shape A 스코프)**: `HARD_CAP_S`는 공유 상수라 A의 cap도 4.0→4.75 h로 바뀐다. 머리말 "shape A 상수 불변"과
  `test_shape_a_constants_untouched_by_rev7`(cap 미검사)는 이 점에서 부정확하다. per-shape cap(A 14,400 유지)으로
  하거나, "rev7 cap은 A 지출을 승인하지 않는다(A는 §0·§15 별도 승인·재추정)"를 등록 문장으로 넣는다.
- **R-5 (cap–고정점 결합 공시)**: notes·머리말의 cap 대안(16,200·18,000)에 "고정점 `REV7_B_WORST_MIN_ADMITTED`
  (16,200→41, 18,000→46)와 1.05 둔화 테스트를 함께 바꿔야 하며 그렇지 않으면 preflight `ABORT_SELFTEST`"를 붙인다.
  cap 결정 뒤 고정점 재계산 = 또 하나의 digest 변경이다.
- **R-6 (재현 n 정정)**: bench 고정비 "40 unit, 중앙 22.6 s"는 감사 재현과 다르다(비floor 42 unit 중앙 22.48 s;
  grid 32 22.02 · 관측자 8 24.33 · bridge 2 20.83). 배제 규칙을 적거나 n을 고친다(값 23.0은 통과 대역 15–30 안이라
  판정 무영향). §8-3 표 교체는 초안이 이미 예고한 대로 봉인 전 수행.

권고(비필수): W-a 부대/고정비 분할을 총시간 외 독립 술어(admit→health, bench 창)로 각각 시험해 보상 등가 변이를
잡는다. W-b 재개 판단 전에 이 판정서 파일명이 `VERDICT_n3_*` glob에 걸린다는 사실을 사용자에게 먼저 알린다(§1 Q5).

## 4. 허용/금지 문안

허용형:
- "rev7 B 예산 상수는 3회차 Job(GPU `c36b3c0c`) 41 unit의 중앙값이다. 같은 Job 안 시간 분할 예보는 합계 ±1.3%
  안이었다. Job 간·GPU 간 예보 정확도는 측정되지 않았다."
- "3회차 k=3 라벨(`CAPACITY_BRACKETED_3` [0.15, 0.22), rev6)과 새 Job의 k=3 라벨은 서로 다른 Job·시점의 독립
  산출물로, Job 이름·GPU ID를 붙여 나란히 적는다."
- "rev7은 예산 상수만 바꿨다. 3회차 라벨은 rev6 산출물로 그대로 유효하다."

금지(인용 금지):
1. "예보 정확도 0.997", "rev7 예보는 검증됐다", "예보 오차 ±0.5%".
2. "17,100 s면 Job 간 변동을 덮는다"(N3C-35).
3. "rev7은 48/48을 보장한다", "k=10이 해결된다"(N3C-36).
4. "W-5는 floor span이 bench 창 안에 있음을 검증한다", "시계 앵커가 없다"(R-1, N3C-38).
5. 3회차 셀과 새 Job 셀을 pooled attainment·bootstrap·k 곡선에 합치는 것; "43+48 셀", 3회차 k=3에 새 Job k=5/10을
   이어 붙인 단일 곡선.
6. 두 Job의 k=3 라벨 일치를 "재현됐다/견고하다"로 쓰는 것(라벨은 절벽 구간 Z_B1(3) 안, N3VR-1; Job 축 σ 미측정).
   불일치하면 "Job 간 불일치"로만 적고 어느 쪽도 정정으로 쓰지 않는다.
7. "rev7이 3회차 결과를 대체/정정한다".

## 5. 등록 caveat (문자 그대로 승계)

- **N3C-34 (적합 표본 = 검증 표본)**: "rev7 B 예산 상수(부대 170 s·bench 고정 23 s·floor 50 s)는 3회차 Job 자신의 41 unit 중앙값이며, 같은 41 unit 예보 합계 0.997은 적합 항등식에 가깝다. 같은 Job 안 시간 분할(전반 적합→후반 1.003, 후반→전반 0.988)만이 표본 외 근거다. Job 간·GPU 간 예보 정확도는 측정되지 않았다(2회차 floor_head 1 unit, 같은 GPU `c36b3c0c`). '예보가 검증됐다', '예보 정확도 0.997'로 쓰지 않는다."
- **N3C-35 (빌린 허용 척도)**: "cap 17,100 s의 +8.2% 균등 둔화 허용은 모형 산술이다. 비교 척도로 든 E2VR-3 +3.6–4.1%는 다른 캠페인의 achieved 처리량 수준 이동(새 Job이 더 빨랐다)이지 unit 벽시계 이동이 아니다. N3 unit 벽시계의 Job 간 변동은 측정되지 않았다. '17,100 s면 Job 간 변동을 덮는다'로 쓰지 않는다."
- **N3C-36 (결측 구조 재발 가능)**: "rev7 스트레스 코너 44/48의 결측은 seed 4589 × 0.61·0.69·0.77과 floor_tail이다 — 3회차와 같은 rate 상관 결측 구조다. 라벨 경로는 fail-closed(읽어야 할 rate의 n<4 → `UNRESOLVED_N_BELOW_4`, 예산 절단 unit 무효)라 용량 라벨 값은 오염되지 않지만 k=10은 다시 미해결일 수 있다. 3회차 실측 총시간 재생으로는 균등 둔화 ×1.08까지 48/48, ×1.15에서 0.69·0.77이 결측이다. 'rev7은 48/48·k=10 해결을 보장한다'로 쓰지 않는다."
- **N3C-37 (고정점은 자기 출력)**: "`REV7_B_WORST_MIN_ADMITTED` = 44는 등록 cap 17,100 s에서 `simulate`가 낸 값을 고정한 회귀 고정점이지 성질의 검증이 아니다. cap 또는 상수 추정량을 바꾸면(16,200→41, 18,000→46; 부대 p90 185 s→42) 함께 바뀐다."
- **N3C-38 (W-5 = 길이 술어)**: "W-5(켤 경우)는 '마지막 16 floor span 창이 bench duration보다 0–2.0 s 짧다'는 길이 술어다. 원안의 포함 술어(span start ∈ [bench_start, bench_end])가 아니며, 원안은 3회차에서 bench_serving 시험 요청을 거부하지 못했을 것이다. 근거는 n=2(gap 0.89·0.88 s, 같은 GPU, shape B 64토큰)이고 shape A floor에서는 미검증이다."
- **N3C-39 (preflight 상수)**: "`PREFLIGHT_ALLOW_S` 300 s는 예보 모형 입력이며 러너의 중단 조건이 아니다. 근거는 n=2(220·224 s, 같은 GPU)다. 'preflight 300 s 보장'으로 쓰지 않는다."
- **N3C-40 (shape A 미검증)**: "rev7은 shape A 예산을 재검증하지 않았다. 부대비용은 harness 의존이다(E2 VESSL r2 중앙 139 s·admit→health 41 s vs N3 B 170 s·104 s). B의 비율을 A로 옮기는 것은 추정이며, rev7 cap은 A 지출을 승인하지 않는다."
- **N3C-41 (Job 혼합)**: "rev7 새 Job과 3회차 Job의 결과는 Job 단위로만 병기한다. 셀 합산·라벨 결합·라벨 일치의 견고성 해석은 금지다. 새 Job도 Job 축 변동(σ_Job)을 측정하지 않는다(Job n=1)."

## 6. 사용자 결정에 영향 주는 사실

1. **(i) 재개 vs 새 Job**: rev6 그대로라면 HEAD(6113723)의 등록 파일은 3회차와 바이트 동일 → digest 게이트 통과
   (Cluster 디렉터리 보존 미확인). rev7 채택 시 재개 불가. **이 판정서 파일이 커밋되면 rev7을 버려도 재개 불가**
   (`VERDICT_n3_*.md` glob). 재개는 k=10 라벨에 Job 혼합을 넣는다(N3VR-6); 새 Job 권고는 정합.
2. **(ii) cap**: 17,100 외 대안을 고르면 고정점·테스트를 같이 바꿔야 하며 안 바꾸면 preflight에서 멈춘다(R-5).
   16,200은 모형 둔화 여유 2.0%뿐, 18,000은 14.3%·스트레스 코너 46/48.
3. **(iv) W-5**: 초안의 "앵커 없음" 전제는 거짓이다(R-1). 원안 포함 술어는 이 Job의 시험 요청 혼입을 못 잡았을 것이고,
   rev7 길이 술어는 잡는다 — 결정은 정정된 사실로 다시.
4. shape A: cap 공유로 rev7이 A cap도 바꾼다(R-4). A 제출 전 별도 재추정·승인 필요는 불변.

## 7. 감사자 자기 적용

- 반전 계산은 초안이 제시한 격자(cap 3안, 추정량 min/중앙/p90/max, preflight 실측/초안/rev6)와 3회차 실측만 썼다.
  재생의 결측 5 unit 보정(모형 보통 코너·floor_tail 219 s)은 초안 notes §3 외삽 규약과 같다.
- 처방 실현가능성: R-1…R-6은 모두 로컬 CPU 문서·테스트 편집이며 GPU 불요. R-5의 고정점 재계산은 digest를 다시
  바꾼다 — 재개 가능성에 대한 영향은 §6-1과 같다.
- 직전 판정서(rev6 delta W-5 원안)를 상수로 승격하지 않았다: 원안 포함 술어 자체를 3회차 원자료로 재검증해 그 약점을
  찾았다(§1 Q4).
