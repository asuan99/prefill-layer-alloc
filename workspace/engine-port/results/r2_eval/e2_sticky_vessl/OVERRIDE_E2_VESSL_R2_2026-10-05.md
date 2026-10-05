# 범위 한정 제출 OVERRIDE — **E2 (sticky 분할 대조), VESSL 기판, 새 20 boot Job (R1)** (2026-10-05) — 초안

> **지위**: E2 VESSL **새 Job 1건**에 한정한 별개의 OVERRIDE다. 형식 요건 5개는 `OVERRIDE_E2_VESSL_2026-10-04.md`(이하 **1회차 OVERRIDE**)를 따른다.
> - 1회차 OVERRIDE는 Job `pdmux-e2-sticky-vessl-e6f424e-20261004135228`(이하 **1회차 중단 Job**) 1건으로 **소진**됐다(그 문서 §7). 그 문서는 바꾸지 않는다.
> - 등록 근거: rev3 + ADDENDUM(A1–A12) + 이식 追記 A13 + 재실행 追記 `PREREG_E2_VESSL_A14_2026-10-05.md`(A14).
> - 실행물: `e2_sticky_vessl_r2.sh`(r2 러너) + `e2_vessl_plan_r2.py`(r2 helper). 1회차 러너·helper는 실행하지 않는다(기록으로 보존).
> - ★**사용자 승인 기록**: **2026-10-05 "231 순서로 진행" — R1 새 20 boot, P3 (ii), 하드캡 4.0 h.**
>   단 **실제 제출은 여전히** A14 규칙층 감사 `GO`(또는 `GO-with-caveats`) + 하네스층 국소 확인 + `launch.sh` dry-run 승인 **뒤**다(§6).
> - 작성: engine-porter(초안). 커밋은 메인 세션.

## 1. 넘기는 차단 — 출력 그대로 인용 (요건 ①)

2026-10-05 로컬 실행(`python3 workspace/engine-port/scripts/discipline/presubmit.py --registry workspace/engine-port/scripts/discipline/presubmit_registry.json`, HEAD `69fbbb0`, A14 파일 미커밋 상태):
```
=== presubmit ===
  OK     registry_append_only 삭제 0건
  OK     line_citations --- check: 201 compared, 0 violation(s)  OK
  OK     citation_stops --- 0 added line(s), 0 violation(s), 16 rules  OK
  OK     reachability   reach_spec_A.json -> superseded by reach_spec_rev3_A.json (미검사)
  OK     reachability   reach_spec_B.json -> superseded by reach_spec_rev3_A.json (미검사)
  OK     reachability   spec_C1_canonical.json -> superseded by workspace/engine-port/results/cp_baseline/spec_cp0_arm_reqactive.json (미검사)
  OK     reachability   spec_cp0_arm_reqactive.json -> DISCRIMINATING  [PATH_REMAP rule_path,out: /scratch/ehmoon/whlee/prefill-layer-alloc -> repo root]
  OK     reachability   spec_cp0_arm_reqinactive.json -> superseded by workspace/engine-port/results/cp_baseline/RETRACTION_reqinactive_2026-08-28.md (미검사)
  OK     reachability   spec_cp0_capacity.json -> DISCRIMINATING  [PATH_REMAP rule_path,out: /scratch/ehmoon/whlee/prefill-layer-alloc -> repo root]
  BLOCK  doc_facts      --- 11 fact(s), 14 occurrence(s) compared, 1 violation(s)  FAIL
  BLOCK  reachability   reachability_spec.json -> SINGLE_LABEL_FORCED  [PATH_REMAP rule_path,out: /scratch/ehmoon/whlee/prefill-layer-alloc -> repo root]
  BLOCK  reachability   reach_spec_rev3_A.json -> RESTRICTIONS_INERT  [PATH_REMAP rule_path,out: /scratch/ehmoon/whlee/prefill-layer-alloc -> repo root]

제출 금지 — 차단 3건. 규율 도구가 통과하기 전에는 job을 제출하지 않는다.
```
- **이 OVERRIDE가 넘기는 것은 두 건뿐이다**: M4R `SINGLE_LABEL_FORCED` · TC1 `RESTRICTIONS_INERT`. 라벨은 2026-09-15(KISTI)·2026-10-04(1회차 OVERRIDE §1-1)와 같다.
- **doc_facts 1건은 넘기지 않는다.** 실체(`check_doc_facts.py` 출력): `regression tests: document says 853, artefact says 940 (DESIGN_A1_REV2_STICKY_2026-08-25.md)`.
  - 도구는 `tests/test_*.py`의 `def test_`를 정적으로 센다. HEAD `69fbbb0`의 추적 파일만으로도 899라서, 이 위반은 **이 OVERRIDE 작업 전부터 있었다**(N3 테스트 추가분).
  - 이번에 `tests/test_e2_vessl_r2.py` 41개가 더해져 940이 됐다.
  - ⇒ 제출 전 doc-steward가 그 문서의 수치를 **커밋 시점의 최종값**으로 고쳐야 한다(구조 재발 `PROJECT_STATUS.md` #245 계열). 그 뒤 presubmit 차단이 정확히 위 2건인지 다시 확인한다(§6 1단계).

### 1-1. 追記 (2026-10-05, doc_facts 7차 정정 후 — 규칙층 A14 판정서 필수 수리 1)

doc-steward가 `DESIGN_A1_REV2_STICKY_2026-08-25.md`의 회귀 수를 853 → **940**으로 갱신(7차 정정: `test_n3_capacity.py` +46, `test_e2_vessl_r2.py` +41; `PROJECT_STATUS.md` #245 / `CONSENSUS.md` §3 항목265에 7차 재발 追記, 새 번호 없음)한 뒤 같은 명령을 다시 돌린 출력을 문자 그대로 적는다.
```
=== presubmit ===
  OK     registry_append_only 삭제 0건
  OK     line_citations --- check: 201 compared, 0 violation(s)  OK
  OK     doc_facts      --- 11 fact(s), 14 occurrence(s) compared, 0 violation(s)  OK
  OK     citation_stops --- 0 added line(s), 0 violation(s), 16 rules  OK
  OK     reachability   reach_spec_A.json -> superseded by reach_spec_rev3_A.json (미검사)
  OK     reachability   reach_spec_B.json -> superseded by reach_spec_rev3_A.json (미검사)
  OK     reachability   spec_C1_canonical.json -> superseded by workspace/engine-port/results/cp_baseline/spec_cp0_arm_reqactive.json (미검사)
  OK     reachability   spec_cp0_arm_reqactive.json -> DISCRIMINATING  [PATH_REMAP rule_path,out: /scratch/ehmoon/whlee/prefill-layer-alloc -> repo root]
  OK     reachability   spec_cp0_arm_reqinactive.json -> superseded by workspace/engine-port/results/cp_baseline/RETRACTION_reqinactive_2026-08-28.md (미검사)
  OK     reachability   spec_cp0_capacity.json -> DISCRIMINATING  [PATH_REMAP rule_path,out: /scratch/ehmoon/whlee/prefill-layer-alloc -> repo root]
  BLOCK  reachability   reachability_spec.json -> SINGLE_LABEL_FORCED  [PATH_REMAP rule_path,out: /scratch/ehmoon/whlee/prefill-layer-alloc -> repo root]
  BLOCK  reachability   reach_spec_rev3_A.json -> RESTRICTIONS_INERT  [PATH_REMAP rule_path,out: /scratch/ehmoon/whlee/prefill-layer-alloc -> repo root]

제출 금지 — 차단 2건. 규율 도구가 통과하기 전에는 job을 제출하지 않는다.
```
⇒ 차단은 **정확히 M4R `SINGLE_LABEL_FORCED` · TC1 `RESTRICTIONS_INERT` 2건**(라벨 2026-09-15·2026-10-04와 동일, 둘 다 `PATH_REMAP` 태그)이며 이 OVERRIDE가 넘기는 범위와 일치한다.

### 1-2. 追記 (2026-10-05, 제출 직전 — doc_facts 8차·9차 정정 후 재확인)

§1-1 이후 커밋 `3604269`(doc_facts 8차 940→946)와 `1a90b08`(N3 하네스층 T-H1…T-H7 테스트 +9)로 집계가 955가 되어 제출 직전 presubmit이 doc_facts 1건으로 다시 3건 차단됐다(experiment-runner가 규약대로 제출 보류). doc-steward가 9차 정정(946→955, `PROJECT_STATUS.md` #245·`CONSENSUS.md` §3 항목265 追記, 새 번호 없음)을 한 뒤 재실행:
```
=== presubmit ===
  OK     registry_append_only 삭제 0건
  OK     line_citations --- check: 201 compared, 0 violation(s)  OK
  OK     doc_facts      --- 11 fact(s), 14 occurrence(s) compared, 0 violation(s)  OK
  OK     citation_stops --- 0 added line(s), 0 violation(s), 16 rules  OK
  OK     reachability   reach_spec_A.json -> superseded by reach_spec_rev3_A.json (미검사)
  OK     reachability   reach_spec_B.json -> superseded by reach_spec_rev3_A.json (미검사)
  OK     reachability   spec_C1_canonical.json -> superseded by workspace/engine-port/results/cp_baseline/spec_cp0_arm_reqactive.json (미검사)
  OK     reachability   spec_cp0_arm_reqactive.json -> DISCRIMINATING  [PATH_REMAP rule_path,out: /scratch/ehmoon/whlee/prefill-layer-alloc -> repo root]
  OK     reachability   spec_cp0_arm_reqinactive.json -> superseded by workspace/engine-port/results/cp_baseline/RETRACTION_reqinactive_2026-08-28.md (미검사)
  OK     reachability   spec_cp0_capacity.json -> DISCRIMINATING  [PATH_REMAP rule_path,out: /scratch/ehmoon/whlee/prefill-layer-alloc -> repo root]
  BLOCK  reachability   reachability_spec.json -> SINGLE_LABEL_FORCED  [PATH_REMAP rule_path,out: /scratch/ehmoon/whlee/prefill-layer-alloc -> repo root]
  BLOCK  reachability   reach_spec_rev3_A.json -> RESTRICTIONS_INERT  [PATH_REMAP rule_path,out: /scratch/ehmoon/whlee/prefill-layer-alloc -> repo root]

제출 금지 — 차단 2건. 규율 도구가 통과하기 전에는 job을 제출하지 않는다.
```
⇒ 차단은 다시 **정확히 M4R·TC1 2건**이다. 이 OVERRIDE가 넘기는 범위와 일치한다. doc_facts 전역 카운터 결함의 구조적 수리((i) 고정값 재정의)는 `PROJECT_STATUS.md` #245에 절차가 구체화돼 있고 사용자 승인 대기다 — 이 결함은 이 OVERRIDE의 판정 경로와 무관하다.

## 2. 왜 이 회차와 인과 무관인가 (요건 ②)

1회차 OVERRIDE §2와 같다. E2의 결정 경로는 `e2_realized_mix.py`·`e2_label.py`·r2 러너·908623 셀프테스트 아카이브뿐이다. M4R·TC1 spec의 산출물·규칙·측정을 읽지도 쓰지도 않는다.
A14가 바꾼 것(포트·teardown·settle·P3 허용오차·재개 거부)도 M4R·TC1의 어떤 파일도 읽거나 쓰지 않는다.

### 2-1. ★M4R 노브 공유 공시 — **1회차 OVERRIDE §2-1 문자 그대로 승계**

> 정본은 M4R의 *"유일한 해소 경로 = **`PDMUX_STICKY_PARTITION=1` 신규 측정**"*(`PROJECT_STATUS.md:9273`)
> 이라고 적었고, **E2가 바로 그 노브를 켠다**. λ0 회차는 *"그 트랙을 건드리지 않는다"* 로 끝났지만
> E2는 그렇게만 말하면 부정확하다. 정확한 관계는 셋이다:
>
> 1. **E2는 M4R이 아니다.** E2의 등록 셀(`a_r4`·`a_r2`·`b_r3`)·판정량(Q1 실현 분할, Q2 achieved)·
>    결정 규칙(R1–R6)은 M4R의 estimand(`R_matched`, confinement 비)와 **다른 양**이다.
> 2. **E2의 결과는 M4R에 대해 아무것도 말하지 않는다.** 결과 문서는 *"M4R이 해소됐다"*·
>    *"confinement 비용이 측정됐다"* 를 **쓸 수 없다**(신규 인용 금지 조항으로 §4에 등록).
> 3. **나중에 E2 아티팩트를 M4R이 재사용하려면 그것은 별개의 사전등록**이다 — 다른 판정량,
>    다른 규칙층 감사. 이 OVERRIDE는 그 재사용을 **승인하지 않는다**.
>
> ⇒ 이 구분이 유지되는 한 **두 차단은 여전히 E2의 판정 경로 밖**이다.

(인용한 `PROJECT_STATUS.md:9273`은 2026-09-15 시점 줄 번호다. 현재 줄 번호는 재확인하지 않았다.)

## 3. 이 OVERRIDE가 **덮지 않는** 것 (요건 ③)

- `doc_facts`·`line_citations`·`citation_stops` 차단은 **넘기지 않는다**. 현재 `doc_facts` 1건이 걸려 있으므로 **이대로는 제출할 수 없다**(§1).
- reachability `?`(도구 미실행) 차단은 **넘기지 않는다**.
- **규칙층·하네스층 감사를 대신하지 않는다**: A14 규칙층 감사와 r2 하네스층 국소 확인이 아직 없다.
- **재개 Job을 승인하지 않는다.** r2 러너는 재개를 거부한다(A14 §4-3). 이 Job이 중단돼도 다음 Job은 다시 새 20 boot이며 **새 OVERRIDE**(사용자 재승인)다.
- **다른 트랙의 제출을 승인하지 않는다.** N3 Workspace 프로브(사용자 결정 ②)는 별도 담당·별도 기록이다. 설계 변경·다른 셀 집합·하드캡 변경은 새 OVERRIDE다.

## 4. 이 회차가 사는 것과 못 사는 것 (요건 ④ — 결과 문서가 문자 그대로 승계)

**사는 것**: 1회차 OVERRIDE §4 "사는 것"과 같다 — VESSL A100 한 Job 안에서, 등록 셀에서 sticky ON이 실현 분할을 D44로 고정하는가(R1 — 라벨이 `STICKY_REALIZES_A`[또는 `cells_evaluated` 병기 `_ON_REMAINING_CELLS`]일 때만 확립; ON 절 성립은 엔진 강제 양성 대조이며 Q1의 답이 아니다, E2VC-1·A13 §4-1) ·
같은 offered에서 achieved가 어떻게 변하는가(R2, paired n=4 + CI) · 그 셀들의 one-knob 검증(R3) · 양 arm greedy 출력 동치(P13).

**못 사는 것 — 1회차 OVERRIDE §4 문자 그대로 승계**:

> **못 사는 것** (rev3 §10 + 승계 caveat):
> - **게이트 #6(λ\*_SLO)을 닫지 않는다** · **P2 블로커 ②를 해소하지 않는다** ·
>   **Claim D/E 등급 불변** · **새 성능 판정 0건** · **λ0R-8(iii)(true-dual 정규화 교락)을
>   해소하지 않는다**(두 arm의 prefill 축이 동일 — E2C-5).
> - ★**E2C-21**: shape A에서 처치되는 시간의 **86–89%가 prefill 유휴 구간**이므로 Q2가 사는 것은
>   *"멀티플렉싱할 prefill이 없는 동안 decode를 44 SM에 묶어 둔 비용"* 까지다(정본 감사 2F9 계열).
>   **정책 서술 금지**(게이트 #2 + 이 caveat의 이중 금지).
> - ★**신규(이 문서)**: *"E2가 M4R을 해소했다"*·*"confinement 비용을 측정했다"* **인용 금지**(§2-1).

**1회차 OVERRIDE §4 "추가(VESSL 이식)" 문자 그대로 승계**:
> - E2C-21의 VESSL 재산출값은 85.8–87.1%(shape A OFF, λ0 VESSL 파일럿)이며 문장 결론은 같다(A13 §3-3).
> - *"E2가 λ0R-8(iii)을 해소했다/판정했다/약화시켰다"* **인용 금지**(E2C-5).
> - E2C-36…E2C-39와 이식 caveat E2V-1…E2V-8(A13 §3)을 결과 문서가 승계한다. 특히 **E2V-1**: VESSL 결과를 KISTI 908623과 비교하는 R6은 판정이 아니며
>   클러스터·드라이버·패키지·엔진 1파일 차이를 병기한다. **VESSL 결과로 KISTI 결론(HE0·정책 순위·caveat 수치)을 갱신하지 않는다**(게이트 1의 연장).
> - ★R1 사전 예보: `a_r2`에서 `OFF_BASELINE_ABOVE_BAND`가 나올 가능성이 크다(VESSL 파일럿 E-qcond 20.57% > 20.0). 그 라벨은 대조 arm에 대한 진술이다(A13 §4-1).

**추가(이 문서, A14)**:
- A14 caveat **E2V-9·E2V-10·E2V-11**을 결과 문서가 문자 그대로 승계한다(A14 §6-2). E2VC-1…E2VC-11 승계(A14 §6-1).
- ★**1회차 중단 Job(D23)**: 그 Job의 a_r4 4짝은 라벨 0건·기술 통계뿐이다. *"두 Job을 합쳐 n=8"* · *"1회차가 이번 결과를 재현/확인했다"* **인용 금지**. 두 Job의 a_r4는 나란히 기술만 한다(E2V-11).
- ★**P3 (ii)**: *"P3가 두 arm의 KV 풀이 같음을 확인했다"* **인용 금지**(E2V-10).

## 5. 예산과 한계 (요건 ⑤)

### 5-1. 예산 3층 — GPU-h와 금액 양쪽 (`resourcespec-a100x1`, 단가 **$1.48/GPU-h**, A14 §5)

| 층 | 값 | GPU-h | 금액 | 근거 |
|---|---|---|---|---|
| **보통(관측 보정)** | 8,071.3 s | **2.242** | **$3.32** | 1회차 A ON boot 벽시계 375 s 반영(a_r2 ON은 같은 값 가정), B ON 최악 코너. 등록 보통(ON bench = OFF bench)은 1.991 GPU-h / $2.95 |
| **최악** | 9,117.3 s / settle 포함 12,717.3 s | **2.533 / 3.533** | **$3.75 / $5.23** | rev3 §8-1 코너(ON bench A ×2.5, B ×1.5). 뒤 값은 O4 settle 최대 180 s × 20 boot를 더한 것 |
| **하드캡** | 14,400 s | **4.000** | **$5.92** | 스크립트 내부 3겹, job_entry 시작부터(A13 §5-3, 불변). 사용자 승인 "하드캡 4.0 h" |
| boot 수 | **20**(`a_r4` 8 · `a_r2` 8 · `b_r3` 4) | | | rev3 §2, 불변. 다섯 모형 중 capped 외 넷 모두 20 boot 입장(`e2_vessl_plan_r2 --budget`) |

- **1회차 소진 기록(참조, 이 승인과 별개)**: 1회차 OVERRIDE §7 — 플랫폼 `Duration` 58m13s = 0.970 GPU-h ≈ **$1.44**, job_entry 내부 타이밍 하한 0.957 GPU-h / **$1.42**. 10/20 boot 후 `ABORT_P3`.
- **두 Job 합계(정보)**: 1회차 0.957–0.970 GPU-h를 더하면 보통 3.20–3.21 GPU-h ≈ $4.74–4.76 · 최악(settle 포함) 4.49–4.50 GPU-h ≈ $6.65–6.67 · 이번 하드캡까지 4.96–4.97 GPU-h ≈ $7.34–7.36.
- **표 밖 비용**(하드캡이 막지 않는다): 스케줄링·이미지 pull, job_entry clone·install, 러너 tail(600 s 예비 안), 결과 반출(E2VC-3).

### 5-2. 초과의 의미 — 1회차 OVERRIDE §5-2 그대로

집행 주체는 스크립트 내부 3겹(P14 입장 · 셀별 `timeout` · watchdog)이다. 플랫폼은 실행 중 Job을 시간으로 죽이지 않는다. 초과 시간만큼 금액이 청구되고, 실패 Job도 돈 시간만큼 과금되며, 자동 재시도는 없다. 캡은 **셀 구간**만 덮는다(clone·install·반출에는 kill 층 없음).
r2에서 달라진 점: O4 settle가 P14 입장 판정 **앞**에서 돈다. 그래서 settle가 쓴 시간은 입장 판정에 보인다(A14 §2-1).

**초과 시**: P14가 다음 boot를 시작하지 않고(`UNRESOLVED_BUDGET`), 진행 중 bench는 마감에서 잘린다. 부분 실행 집합으로 **재라벨하지 않는다**(D23). 재개 Job은 없다(§3).

## 6. 한 창 규율 (λ0R-10 승계)

제출은 **한 창** 안에서 아래 순서로만 한다. 중간에 다른 작업이 끼면 처음부터 다시:

1. doc_facts 선행 수리(doc-steward) 뒤 `presubmit.py` 재실행 → **차단이 §1의 M4R·TC1 2건과 정확히 같은지** 확인. 출력을 이 문서 §1-1에 문자 그대로 追記.
2. 셀프테스트 재실행(로컬): `e2_realized_mix.py --selftest` · `e2_label.py --selftest` · `e2_vessl_plan.py --selftest` · `e2_vessl_plan_r2.py --selftest` ·
   `tests/test_e2_vessl.py`(1회차, 무수정) · `tests/test_e2_vessl_r2.py` · 전체 CPU 회귀 · `check_citation_stops.py` 0위반.
3. ★**커밋**(P5) — A14·r2 러너·r2 helper·r2 테스트·이 OVERRIDE·감사 판정서. **커밋 없이 제출 금지**(`launch.sh --submit`은 dirty tree를 거부한다).
   커밋 후 1회차 봉인 파일 sha가 그대로인지 다시 확인(r2 러너가 Job 안에서도 확인한다).
4. staging: 1회차가 올린 `staging/e2_selftest_archive/`를 재사용한다(§7 확인 기록). 다시 올릴 필요가 생기면 `stage_e2_inputs.sh --upload`(스토리지만).
5. `launch.sh --campaign e2_sticky_vessl --target workspace/engine-port/results/r2_eval/e2_sticky_vessl/e2_sticky_vessl_r2.sh` **dry-run** → 사용자가 명령·단가·job name 승인 → `--submit`.
   `--env PDMUX_E2V_RESUME_FROM=…`을 붙이지 않는다(붙이면 `ABORT_RESUME`).
6. job name을 이 문서 §7에 追記하고 **이 OVERRIDE를 소진 처리**한다. 회수는 `fetch.sh`(DONE + sha256)까지.

## 7. 승인란

- **사용자 승인**: ☑ **2026-10-05 사용자 "231 순서로 진행" — R1 새 20 boot, P3 (ii), 하드캡 4.0 h.**
  (메인 세션 전달. 결정 ①: E2 재실행 = R1 새 20 boot Job + 새 범위 한정 OVERRIDE. 결정 ③: P3 = (ii) 허용오차.)
  ⚠️이 초안의 §5 금액(관측 보정·settle 포함 최악·두 Job 합계)과 §1 presubmit 현황(doc_facts 1건)은 승인 **이후** 작성됐다. 메인 세션이 제출 전 사용자에게 이 표를 다시 보여 확인받는 것을 권한다.
- **승인된 예산 상한**: 2.242 GPU-h / $3.32(보통, 관측 보정) · 2.533 GPU-h / $3.75(최악) · 3.533 GPU-h / $5.23(최악 + settle) · **4.0 GPU-h / $5.92(셀 구간 하드캡)**.
- **선행조건 상태**:
  - 규칙층 A14 감사 ☑(`VERDICT_e2_vessl_a14_rules_2026-10-05.md` **`GO-with-caveats`**, 死因 0, E2VC-12…16)
  - 하네스층 국소 확인(r2 러너·r2 helper·r2 테스트 봉인 sha, 1회차 봉인 불변) ☐
  - presubmit 차단 = M4R·TC1 2건 ☑(§1-1, doc_facts 7차 정정 후)
  - 커밋 ☐
  - staging ☐ — 재사용 가능 여부 확인: 2026-10-05 `vesslctl volume ls <object volume> --prefix staging/e2_selftest_archive/` → 7 파일 + `SHA256SUMS`(2026-10-04T13:23:51Z). 파일 크기 합 237,346,884 bytes가 로컬 908623 아카이브와 같고, 로컬 아카이브는 `--verify-archive` OK. 내용 sha는 Job 안 `--verify-archive`가 다시 확인한다(`ABORT_ARCHIVE_DRIFT`). 1회차 중단 Job의 `ARCHIVE_CHECK.txt`도 OK였다.
  - dry-run 승인 ☐
- **소진 job name**: (미제출)
