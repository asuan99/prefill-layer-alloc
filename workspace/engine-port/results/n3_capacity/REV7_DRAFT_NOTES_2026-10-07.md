# N3 rev7 노트 (2026-10-07) — 델타 감사 `GO-with-caveats` 통과, 사용자 결정 반영, 봉인 대상(커밋 시 봉인)

작성: experiment-runner (GPU 0, 성능 판정 0, Claim 등급 변경 0). 사용자 "A 진행"(결정 정리 + 선택 무관 준비). 정본 문서는 편집하지 않았다.

## 1. (i) 재개 가능성 — 사실

| 질문 | 답 | 근거 |
|---|---|---|
| 이전 Job 디렉터리 `$DATA/runs/<job>/results/n3_capacity/n3b_<job>`가 Cluster 볼륨에 남아 있나 | **로컬에서 GPU 없이 확인할 수 없다** | `vesslctl volume ls`는 Object 볼륨만 지원("listing contents is only supported for object volumes"). Cluster 볼륨 `pdmux-cs`는 로컬 CLI로 목록·다운로드 불가(운영 모델 §2). 확인하려면 Workspace(`pdmux-build` CPU)를 **켜야** 한다 — 켜지 않았다 |
| 삭제 코드가 있나 | 없다 | `job_entry.sh`는 `${DATA}/runs/<job>`를 만들고 쓰기만 한다(`rm -rf`는 `${PROJECT}` 경로 한정, 줄 90–102). 보존은 "삭제 코드 부재"이지 확인이 아니다 |
| Object 볼륨 사본 | **있다** | `vesslctl volume ls objvol-9vgmwmerzwd3 --prefix results/pdmux-n3-capacity-38ed3f4-20261006092813/` → `DONE`·`SHA256SUMS`·`meta/`·`logs/`·`results/`. 로컬 회수본에 `unit_*.complete` 43개 |
| 러너가 읽는 위치 | Cluster 볼륨만 | `n3_capacity.sh` 줄 210: `PREV="$DATA/runs/$RESUME_FROM/results/n3_capacity/n3${SLOW}_$RESUME_FROM"` + 줄 219 `.../meta/substrate.json`. Object 사본에서 재개하려면 러너 수정이 필요(규칙 변경) |

### §8-4 재개 조건 vs rev7
- 조건 (a) `UNITS.txt` 바이트 동일 — **rev7과 양립**(units 목록 불변; `TestRev7Budget.test_units_unchanged_by_rev7`).
- 조건 (b) **등록 digest 동일 — rev7과 양립하지 않는다.** 러너 줄 213–215가 `REGISTRATION_SHA256.txt`에서 `n3_|PREREG_|VERDICT_|pdmux_n3|analyze.py|lambda0_plan` 줄의 **해시 열 전체**를 비교한다. rev7은 `n3_plan.py`(상수)와 `PREREG_*`(머리말)를 바꾸고 새 `VERDICT_n3_rev7_delta_*`가 추가되므로 해시열이 달라져 `ABORT_RESUME`(exit 2)이다(`test_resume_registration_digest_gate_blocks_a_rev7_resume_from_rev6`로 사실 고정).
- 조건 (c) `T_K.json` 유효 — 양립(3회차 `T_K.json` 유효, floor 0.905 s).
- 결론: **rev7(규칙 변경)을 채택하면 rev6 Job으로의 재개는 현 규약상 불가**다. 재개를 살리려면 rev7에 "예산 상수만 다른 이전 Job 허용" 조항(허용 파일·키 열거 + 측정·채점에 영향 없음 증명)을 새로 쓰고 델타 감사를 받아야 하며, 이는 digest 게이트를 약화하는 규칙 변경이다. 이번 초안에는 **구현하지 않았다**.

### 권고(사용자 결정 (i))
- **새 Job(전체 48 unit)을 권고**한다. 근거: (1) 재개는 위 이유로 별도 규칙 변경+감사가 필요하고 Cluster 보존도 미확인 (2) 재개 셀(seed 4589 × 0.53–0.77 4개)은 다른 Job·GPU·시점이라 k=10 라벨을 **재개 셀이 직접 정한다**(N3VR-5·6, 결과 감사 §3-3) — Job 혼합이 라벨에 들어간다 (3) 새 Job은 한 Job 안 전 arm 원칙에 부합하고 3회차 43 unit의 독립 반복(Job 간 이동 크기 관측)을 덤으로 준다.
- 대가: 재개(≈ 5 unit, 컨테이너 약 0.5 GPU-h)에 비해 새 Job은 약 4.2 GPU-h. 비용을 줄이는 것이 우선이면 재개가 되나, 그 경우 위 새 조항 + Cluster 보존 확인(`pdmux-build` 기동 1회) + Job 혼합 병기가 선행 조건이다.

## 2. (ii) 재추정 — 같은 러너 기준 실측 (`TIMING.tsv`, Job 3, 추적 파일)

정의: unit 총시간 = 다음 unit `admit_epoch` − 이 unit `admit_epoch`(boot·warm-up·bench·종료·settle 전부). bench 창 = `bench_end − bench_start`. 부대비용 = 총시간 − bench 창. 모형 bench = rev6 `bench_s`(max(span, N/λ\*_ref)+drain).

| 항목 | n | 중앙 | p10/p90 | 최소–최대 | rev6 상수 | rev7 상수 |
|---|---|---|---|---|---|---|
| 부대비용(warm, floor_head 제외) | 41 | 170 s | – / 185 | 163–192 | 121.0 | **170.0** |
| bench 고정비(측정 창 − 모형, 비floor 전부: grid 32·관측자 8·bridge 2) | 42 | 22.48 s (grid 22.02 · 관측자 24.33 · bridge 20.83) | – | 14.9–35.5 | (항 없음 ≈ 0) | **23.0**(통과 대역 15–30 안의 올림값) |
| floor | 1 | 창 49 s | | | 40 | `floor_bench_s` **50.0** |
| 부대비용 구성(중앙) | 43 | admit→health 104 s · health→bench 시작(warm-up) 42 s · bench 종료→unit 종료 17 s · settle 7 s | | | | |
| preflight(job 시작→첫 입장) | 2 | 224 s(3회차)·220 s(2회차) | | | 600(미측정) | **300** |
| 첫 boot cold 추가 | 1 | 59 s(floor_head 부대 229 − 170) | | | 92.1 | 92.1 **불변**(더 큰 값) |

- 6그룹 Workspace 프로브의 boot 벽시계 59 s는 Job 안 admit→health 104 s와 다르다(프로브는 boot만, 규약 §8-7과 교훈 273 일치) — 상수는 boot 프로브가 아니라 입장→입장 전체로 정한다.
- 중앙 예보 오차: rev7 상수의 per-unit 예보 − 측정 총시간 중앙 상대오차 +0.46%, 범위 −6.3%…+4.3%, 41 unit 합계 예보/측정 0.997(테스트 상한: 중앙 3%·최대 12%). rev6 상수는 −25%(총시간 293 s 대 예보 221 s).

## 3. 예보 수치표 (B, 사이드: A) — cap은 shape별 분리(A 14,400 유지, rev7 cap은 A 지출 미승인)

| 항목 | rev6 상수 | rev7 초안 |
|---|---|---|
| 보통 코너 컨테이너(tail 제외) | 12,126 s / 3.37 GPU-h | **15,256 s / 4.24 GPU-h** (입장 48/48) |
| 최악(스트레스) 코너 | 13,792 s / 3.83 GPU-h (47/48) | cap 17,100: **44/48**, 16,348 s / 4.54 GPU-h |
| 하드캡(셀 구간) | 14,400 s (4.0 h) | 14,400이면 보통 코너 42/48 · **17,100 s(4.75 h) 권고** |
| 3회차 실측 외삽(43 측정 + 결측 5를 같은 rate의 타 seed 중앙·floor_tail 219 s) | – | ≈ 15,190 s (4.22 h) — 모형 15,256과 0.4% 차 |

48/48이 되는 cap(보통 코너, 모형): 15,900 s(+0% 여유) · 16,200(균등 둔화 허용 +2.1%) · 16,800(+6.2%) · **17,100(+8.2%)** · 18,000(+14.3%). Job 간 이동은 E2VR-3 +3.6–4.1%라 16,200은 얇다.
최악 코너(모든 unit이 동시에 부대 ×1.25·용량 ×0.8)는 cap 18,300에서야 47/48이다. 이 코너가 실측 41 unit 전부를 덮는다는 점(관측 최대 부대 192 s = 중앙의 1.13배)은 보존된다; 따라서 "최악 47/48" 성질은 스트레스 코너로서 포기하고 B는 44/48로 고정한다.
- **48/48이 4.0 h에 들어가지 않는다**(보통 코너 4.24 GPU-h). 선택지: (1) cap 4.75 h 승인(권고, 운영 모델 "Job당 ≲3–4 GPU-h" 지침을 넘으므로 사용자 승인 문언 필요) (2) floor_tail(진단, 약 220 s) 제거 — UNITS 변경이라 별도 규칙 변경 (3) 셀 수 축소(seed 또는 rate) — 판정 하한 n≥4와 충돌하거나 격자 간격 규칙(§6)과 충돌, 권고하지 않음.
- shape A: 상수 **미변경·미실측**. B의 부대비용 비율(170/121 = 1.40)이 A에도 오면 A 보통 코너가 2.86→약 3.50 GPU-h(모형 재계산)가 될 수 있다(추정, 근거는 B 1개 Job). A 제출 전 재추정 필수.

## 4. (iv) W-5 구현 방식 (★R-1 정정판)
- **정정**: 이전 판의 "bench 기록에 CLOCK_MONOTONIC 앵커가 없다"는 거짓이다. 러너는 unit마다 `marks_<unit>.txt`(`bench_start`/`bench_end`, CLOCK_MONOTONIC)를 쓰고 `ABORT_CLOCK`으로 서버 이벤트와 같은 시계인지 검사한다.
- 원안 포함 술어(마지막 16 span start ∈ [bench_start, bench_end])는 표식 창이 bench_serving 프로세스 전체를 감싸므로 3회차의 bench_serving 시험 요청(span start 1832012.777 ∈ [1831996.564, 1832046.971])을 **거부하지 못했을 것**이다. rev7 길이 술어(`0 ≤ duration − 창 ≤ 2.0 s`)는 다른, 더 강한 술어로 이를 잡는다(gap < 0). 근거 n = 2(gap 0.89·0.88 s, 같은 GPU, shape B; shape A 미검증).
- 모드 3개(`n3_plan.W5_MODE`, 기본 `off`): `length` / `contain_length`(포함+길이) / `off`. 켜면 `floor_from` 결과 `rule` = `rev7+W5:<mode>`. `marks_path` 미지정 시 `bench_<unit>.jsonl` 옆 `marks_<unit>.txt`를 쓰고, 없거나 파싱 불가면 fail-closed. `w5_window_gap_s`는 꺼져 있어도 항상 기록.
- 테스트(`TestW5Rev7`): 3모드 모두 실측 floor 수용, 포함 술어가 시험 요청 혼입을 못 거부하고 길이 술어는 거부, 시작이 늦은 marks 변이는 포함 모드만 거부, marks 부재 fail-closed.

## 5. (iii) shape A 착수에 필요한 것 (작업하지 않음)
1. A 예산 상수 재추정: 3회차 B 같은 TIMING 측정이 A에는 없다 — B 부대비용 비율을 그대로 쓰면 A도 부족할 수 있다(추정). 등록 범위: `SHAPES["A"]` 상수 + A 예보표 + 델타 감사.
2. 별도 사용자 승인(PREREG §0·§15: A는 별도 Job·별도 승인) + B의 rev7 델타 감사 통과 선행(동일 `n3_plan.py` 공유).
3. 예산 영향: 현 모형 A 보통 2.86 GPU-h, 최악 3.32(미검증). 부대 ×1.4 가정 시 약 3.50 GPU-h(bench 고정비 항은 A 미측정이라 0 유지 — 더하면 더 큼). 현 cap 4.0 h 안이지만 근거가 B 1개 Job뿐이다.
4. 열린 사전 질의: A의 W-5·floor 규칙(floor_bench_s 110 s) 유효성도 미측정.

## 6. 금지 문안 준수
"0.15 req/s" 단독 인용 없음. 게이트 #6은 닫히지 않는다(shape A 미실행, k=5/10 미해결). 이 문서는 예산 모형만 다루며 성능 판정을 내리지 않는다.

## 7. 델타 감사 수리 반영 (R-1…R-6, 2026-10-07)
- R-1 W-5 전제 정정·3모드·rule 필드 · R-2 테스트 클래스를 `__main__` 위로 이동(모듈 모드·스크립트 모드 같은 수) · R-3 `budget_table().note` 교체 · R-4 cap per-shape(`HARD_CAP_S_B` 17,100 초안 / `HARD_CAP_S_A` 14,400, 러너는 `HARD_CAP_S_${SHAPE}`) · R-5 스트레스 코너 고정점을 cap별 표로 파라미터화(16,200→41, 17,100→44, 18,000→46; 표에 없는 cap은 selftest 실패, cap 결정 뒤 digest 재변경) · R-6 bench 고정비 표본 42 unit·중앙 22.48 s로 정정, §8-3 표 rev7 값으로 교체(cap 결정 대기 표시). 보상 등가 변이(overhead 193 + fixed 0) 구성 요소 술어 테스트 추가.
- ★재개 경고: 판정서 파일명 `VERDICT_n3_rev7_delta_*.md`가 등록 glob `VERDICT_n3_*.md`에 걸린다. `n3_capacity/` 안의 파일을 커밋하면 rev7을 버려도 rev6 재개가 `ABORT_RESUME`이다 — 사용자 결정 (i) 전 미커밋 유지.

## 8. 결정 기록 (2026-10-07, 사용자 "권장 사항대로 진행")
| 결정 | 내용 | 반영 |
|---|---|---|
| (i) | **새 Job, 전체 48 unit**(`RESUME_FROM` 없음). rev6 재개 경로 포기 → `n3_capacity/` 커밋 허용 | PREREG 머리말 재개 줄. 제출 시 `--env PDMUX_N3_RESUME_FROM` 금지 |
| (ii) | **shape B 하드캡 17,100 s(4.75 h)** 승인. A는 14,400 유지·A 지출 미승인 | `HARD_CAP_S_B` = 17100, 스트레스 코너 고정점 44 |
| (iv) | **W-5 모드 = `length`**(기본 ON) | `W5_MODE = "length"`; floor `rule` = `rev7+W5:length`(`T_K.json`에 기록); 합성 floor 픽스처에 `duration` 추가 |
- 델타 감사 `VERDICT_n3_rev7_delta_2026-10-07.md` `GO-with-caveats`, R-1…R-6 반영, N3C-34…41은 PREREG §11에 문자 그대로 승계.
- 제출은 아직 하지 않았다: 커밋 → `launch.sh` dry-run → 사용자 승인 문언 → `--submit`.
