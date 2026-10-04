# 범위 한정 제출 OVERRIDE — **E2 (sticky 분할 대조), VESSL 기판** (2026-10-04) — 초안

> **지위**: E2 1 Job(VESSL)에 한정한 **네 번째, 별개의** OVERRIDE다. 형식 요건 5개는 전례
> `OVERRIDE_LAMBDA0_SUBMIT_2026-09-14.md`·`e2_sticky_prereg/OVERRIDE_E2_SUBMIT_2026-09-15.md`를 따른다.
> - 기존 E2 OVERRIDE(`OVERRIDE_E2_SUBMIT_2026-09-15.md`, KISTI SLURM, 미소진)는 이식성 판정서
>   (`VERDICT_e2_substrate_portability_2026-09-18.md` Q3)가 **`REFUTED`**로 판정했다 — 문서 자신의 §7-1 "설계나 예산이 바뀌면 새 OVERRIDE"가
>   집행기 교체·상수 재산출·과금 모형 전환 세 경로로 발효됐다. ⇒ 그 문서는 **VESSL 제출에 쓸 수 없다**. 이 문서가 대신한다.
> - 등록 근거: rev3 + ADDENDUM(A1–A12) + 이식 追記 `PREREG_E2_VESSL_A13_2026-10-04.md`(A13).
> - ★**사용자 승인 기록**: 2026-10-04 사용자 "권고 사항대로 진행" — 승인 범위 **20 boot, 하드캡 ≈4 h**.
>   단 **실제 제출은 여전히** A13 규칙층 감사 `GO`(또는 `GO-with-caveats`) + 하네스층 국소 재확인 + `launch.sh` dry-run 승인 **뒤**다(§6).
> - 작성: engine-porter(초안). 커밋은 메인 세션.

## 1. 넘기는 차단 — 출력 그대로 인용 (요건 ①)

2026-10-04 로컬 실행(`python3 workspace/engine-port/scripts/discipline/presubmit.py --registry presubmit_registry.json`, HEAD `5893753`, A13 파일 미커밋 상태):
```
=== presubmit ===
  OK     registry_append_only 삭제 0건
  OK     line_citations --- check: 201 compared, 0 violation(s)  OK
  OK     citation_stops --- 0 added line(s), 0 violation(s), 16 rules  OK
  OK     reachability   reach_spec_A.json -> superseded by reach_spec_rev3_A.json (미검사)
  OK     reachability   reach_spec_B.json -> superseded by reach_spec_rev3_A.json (미검사)
  OK     reachability   spec_C1_canonical.json -> superseded by workspace/engine-port/results/cp_baseline/spec_cp0_arm_reqactive.json (미검사)
  OK     reachability   spec_cp0_arm_reqinactive.json -> superseded by workspace/engine-port/results/cp_baseline/RETRACTION_reqinactive_2026-08-28.md (미검사)
  BLOCK  doc_facts      --- 11 fact(s), 14 occurrence(s) compared, 2 violation(s)  FAIL
  BLOCK  reachability   reachability_spec.json -> ?
  BLOCK  reachability   reach_spec_rev3_A.json -> ?
  BLOCK  reachability   spec_cp0_arm_reqactive.json -> ?
  BLOCK  reachability   spec_cp0_capacity.json -> ?

제출 금지 — 차단 5건. 규율 도구가 통과하기 전에는 job을 제출하지 않는다.
```
★**2026-09-15(KISTI)와 다르다.** 그날은 차단이 정확히 2건(M4R `SINGLE_LABEL_FORCED`·TC1 `RESTRICTIONS_INERT`)이었다. 지금 5건의 실체(engine-porter 진단, GPU 0):
- **reachability `?` 4건 = 측정 실패**이지 설계 판정이 아니다(게이트 #21). 네 spec이 가리키는 규칙 파일 경로가 이양 전 절대경로
  `/scratch/ehmoon/whlee/…`(예: `m4r_confinement/m4r_rule_rev2.py`, `tc1_model_attrib/tc1_rule_rev3.py`, `cp_baseline/cp0_arm_rule.py`, `cp0_capacity_rule.py`)라
  `design_reachability.py`가 `FileNotFoundError`로 죽고 `VERDICT:` 줄을 내지 못한다. 2026-09-15에 `DISCRIMINATING`(OK)였던 cp_baseline 2건도 같은 이유로 `?`다.
- **doc_facts 2건** = `DESIGN_A1_REV2_STICKY_2026-08-25.md`의 낡은 수(citation manifest 96 vs 산출물 204, 회귀 710 vs 805). E2 문서군 위반이 아니다.

**이 OVERRIDE가 넘기는 것은 2026-09-15와 같은 두 건(M4R·TC1 reachability)뿐이다.** 나머지 3건은 넘기지 않는다(§3).
⇒ 제출 전 선행 수리 2개(별도 작업, E2 판정 경로 밖): (i) reachability spec의 규칙 파일 경로를 저장소 상대로 고쳐 도구가 실제로 판정을 내게 한다,
(ii) doc_facts 2건을 doc-steward가 정리한다. 그 뒤 presubmit 차단이 **정확히 M4R·TC1 2건**인지 다시 확인한다(§6 1단계).
두 건의 판정이 2026-09-15와 다르게 나오면(예: 경로 수리 후 `DISCRIMINATING`) 그 사실을 이 문서에 追記하고 넘길 차단을 다시 적는다.

### 1-1. 追記 (2026-10-04, 선행 수리 후 — 규칙층 rev2 판정서 선행조건 3)

위 인용은 **수리 전(차단 5건) 스냅샷**이다. 수리 2건을 마친 뒤 같은 명령을 HEAD `5893753`(이식 파일 미커밋) 작업 트리에서 다시 돌린 출력을 문자 그대로 적는다.
- 수리 (i)는 **spec 편집이 아니라 도구 수준 치환**이었다: `scripts/discipline/design_reachability.py`에 `resolve_legacy()`(이양 전 접두
  `/scratch/ehmoon/whlee/prefill-layer-alloc` → 저장소 루트, 대상 파일이 존재할 때만, 치환마다 `PATH_REMAP` 로그)를 추가하고 `presubmit.py`가 그 태그를
  표시한다. spec 4개와 추적 판정 JSON 4개의 바이트는 변하지 않았다(claims-auditor rev2 독립 재현, E2VC-11).
- 수리 (ii)는 doc-steward가 `DESIGN_A1_REV2_STICKY_2026-08-25.md`의 두 수치를 현재값(204 · **853** — 위 "805"는 수리 전 값이며 그 뒤 `test_e2_vessl.py` 48개가
  더해졌다)으로 갱신하고 구조 결함 재발을 `PROJECT_STATUS.md` #245 / `CONSENSUS.md` §3 항목265에 追記했다.
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
⇒ 차단은 **정확히 2026-09-15와 같은 두 건**(M4R `SINGLE_LABEL_FORCED` · TC1 `RESTRICTIONS_INERT`, 라벨 동일)이며, 이 OVERRIDE가 넘기는 범위와 일치한다.
cp_baseline 2건은 `DISCRIMINATING`(OK)으로 복귀했다. 넘길 차단을 다시 적을 필요가 없다.

## 2. 왜 이 회차와 인과 무관인가 (요건 ②)

2026-09-15 OVERRIDE §2와 같다 — E2의 결정 경로는 `e2_realized_mix.py`·`e2_label.py`·이식 러너·908623 셀프테스트 아카이브뿐이고
M4R·TC1 spec의 산출물·규칙·측정을 읽지도 쓰지도 않는다.

### 2-1. ★M4R 노브 공유 공시 — **기존 OVERRIDE §2-1 문자 그대로 승계**

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

- `doc_facts`·`line_citations`·`citation_stops` 차단은 **넘기지 않는다**. 현재 `doc_facts` 2건이 걸려 있으므로 **이대로는 제출할 수 없다**(§1).
- reachability `?`(도구 미실행) 차단은 **넘기지 않는다**. 측정 실패를 OVERRIDE로 덮으면 "검사했다"는 기록만 남는다(게이트 #21의 거울상).
- **규칙층·하네스층 감사를 대신하지 않는다**: A13 규칙층(이식) 감사와 하네스층 국소 재확인이 아직 없다.
- **다른 트랙의 제출을 승인하지 않는다.** 이 문서는 **E2 VESSL 1 Job 한정**이다(재개 Job은 §5-1). 재실행·설계 변경·다른 셀 집합·하드캡 변경은 **새 OVERRIDE**다.

## 4. 이 회차가 사는 것과 못 사는 것 (요건 ④ — 결과 문서가 문자 그대로 승계)

**사는 것**: VESSL A100 한 Job 안에서, 등록 셀에서 sticky ON이 실현 분할을 D44로 고정하는가(R1 — 라벨이 `STICKY_REALIZES_A`[또는 `cells_evaluated` 병기 `_ON_REMAINING_CELLS`]일 때만 확립; ON 절 성립은 엔진 강제 양성 대조이며 Q1의 답이 아니다, E2VC-1·A13 §4-1) ·
같은 offered에서 achieved가 어떻게 변하는가(R2, paired n=4 + CI) · 그 셀들의 one-knob 검증(R3) · 양 arm greedy 출력 동치(P13).

**못 사는 것 — 기존 OVERRIDE §4 문자 그대로 승계**:

> **못 사는 것** (rev3 §10 + 승계 caveat):
> - **게이트 #6(λ\*_SLO)을 닫지 않는다** · **P2 블로커 ②를 해소하지 않는다** ·
>   **Claim D/E 등급 불변** · **새 성능 판정 0건** · **λ0R-8(iii)(true-dual 정규화 교락)을
>   해소하지 않는다**(두 arm의 prefill 축이 동일 — E2C-5).
> - ★**E2C-21**: shape A에서 처치되는 시간의 **86–89%가 prefill 유휴 구간**이므로 Q2가 사는 것은
>   *"멀티플렉싱할 prefill이 없는 동안 decode를 44 SM에 묶어 둔 비용"* 까지다(정본 감사 2F9 계열).
>   **정책 서술 금지**(게이트 #2 + 이 caveat의 이중 금지).
> - ★**신규(이 문서)**: *"E2가 M4R을 해소했다"*·*"confinement 비용을 측정했다"* **인용 금지**(§2-1).

**추가(이 문서, VESSL 이식)**:
- E2C-21의 VESSL 재산출값은 85.8–87.1%(shape A OFF, λ0 VESSL 파일럿)이며 문장 결론은 같다(A13 §3-3).
- *"E2가 λ0R-8(iii)을 해소했다/판정했다/약화시켰다"* **인용 금지**(E2C-5).
- E2C-36…E2C-39와 이식 caveat E2V-1…E2V-8(A13 §3)을 결과 문서가 승계한다. 특히 **E2V-1**: VESSL 결과를 KISTI 908623과 비교하는 R6은 판정이 아니며
  클러스터·드라이버·패키지·엔진 1파일 차이를 병기한다. **VESSL 결과로 KISTI 결론(HE0·정책 순위·caveat 수치)을 갱신하지 않는다**(게이트 1의 연장).
- ★R1 사전 예보: `a_r2`에서 `OFF_BASELINE_ABOVE_BAND`가 나올 가능성이 크다(VESSL 파일럿 E-qcond 20.57% > 20.0). 그 라벨은 대조 arm에 대한 진술이다(A13 §4-1).

## 5. 예산과 한계 (요건 ⑤)

### 5-1. 예산 3층 — GPU-h와 금액 양쪽 (`resourcespec-a100x1`, 단가 **$1.48/GPU-h**)

| 층 | 값 | GPU-h | 금액 | 근거 |
|---|---|---|---|---|
| **보통** | 7,168.5 s | **1.991** | **$2.95** | A13 §5-2. VESSL λ0 실측 bench duration + per-boot A 160.1 s · B 121.0 s + cold 첫 boot 92.1 s + P13 2×30 s + preflight 600 s |
| **최악 코너** | 9,117.3 s | **2.533** | **$3.75** | rev3 §8-1 코너(ON bench A ×2.5, B ×1.5)를 VESSL 상수로 |
| **하드캡** | 14,400 s | **4.000** | **$5.92** | 스크립트 내부 집행기, job_entry 시작부터(A13 §5-3). 사용자 승인 "≈4 h" |
| boot 수 | **20**(`a_r4` 8 · `a_r2` 8 · `b_r3` 4) | | | rev3 §2, 불변 |

- **표 밖 비용**(하드캡이 막지 않는다): 스케줄링·이미지 pull(B5에서 1분 미만), job_entry clone·install(λ0 VESSL 14 s),
  러너 tail(600 s 예비 안), 결과 반출(λ0 VESSL 규모 ≈400 MB, 미측정). λ0 VESSL 실지출은 컨테이너 1.894 h = **$2.80 하한**이었다.
- **비교(KISTI 등록, 무상)**: 1.803 / 2.338 / 3.0 GPU-h. VESSL 값이 큰 이유는 per-boot 부대비용(A +14.6 s, B +14.5 s)·cold 첫 boot·하드캡 상향(3.0 → 4.0 h)이다.
- **재개 Job**: 첫 Job이 BUDGET 절단·boot 실패로 짝을 남기면 같은 등록으로 재개 Job 1개를 낼 수 있다. 재개 Job의 하드캡도 4.0 h이며
  **두 Job 합계 상한은 이 OVERRIDE가 승인한 범위(≈4 h)를 넘을 수 있으므로 재개 Job 제출은 사용자 재승인 대상**이다.

### 5-2. ★초과의 의미가 바뀌었다 — 무상 SLURM → 시간 과금

| | KISTI(기존 OVERRIDE) | **VESSL(이 문서)** |
|---|---|---|
| 집행 주체 | SLURM `--time 03:00:00`(스케줄러가 강제 종료) | **스크립트 내부 3겹**: 입장 규칙(P14) · 셀별 `timeout` · watchdog(`pkill`). 플랫폼은 실행 중 Job을 시간으로 죽이지 않는다 |
| 초과 시 결과 | 잡 종료, **추가 비용 0**(무상 할당) | 초과 시간만큼 **금액이 청구된다**. 크레딧 잔액 ≤ 0은 **생성** 차단이지 실행 중 종료가 아니다 |
| 캡이 덮는 범위 | 잡 전체 | **셀 구간만**. clone·install·tail·반출에는 kill 층이 없다(LV-13 형식) ⇒ "과금 하드캡"이 아니다 |
| 실패 비용 | 0 | 실패 Job도 돈 시간만큼 과금. 자동 재시도 없음 |

⇒ **같은 숫자라도 같은 승인이 아니다**(이식성 판정서 Q3-(ii)). 이 표가 사용자에게 제시된 위험 프로필이다.

**초과 시**: P14가 다음 boot를 시작하지 않고(`UNRESOLVED_BUDGET`), 진행 중 bench는 마감에서 잘린다. 부분 실행 집합으로 **재라벨하지 않는다**(D23).
결과 문서는 "하드캡 안에서 끝났다"를 `meta/timing.txt`(`start_utc`·`end_utc`)와 `BUDGET.txt`(`remaining_at_loop_end_s`) 없이 쓰지 않는다(E2C-39).

## 6. 한 창 규율 (λ0R-10 승계)

제출은 **한 창** 안에서 아래 순서로만 한다. 중간에 다른 작업이 끼면 처음부터 다시:

1. `presubmit.py` 재실행 → **차단이 §1의 M4R·TC1 2건과 정확히 같은지** 확인(현재 5건 — §1의 선행 수리 2개가 먼저다).
2. 셀프테스트 재실행(로컬): `e2_realized_mix.py --selftest`(7/7) · `e2_label.py --selftest`(8+10) · `e2_vessl_plan.py --selftest` ·
   `tests/test_e2_vessl.py` · 전체 CPU 회귀 · `check_citation_stops.py` 0위반.
3. ★**커밋**(P5) — A13·러너·helper·staging 도구·테스트·이 OVERRIDE·감사 판정서. **커밋 없이 제출 금지**(`launch.sh --submit`은 dirty tree를 거부한다).
4. `stage_e2_inputs.sh --upload`(스토리지만).
5. `launch.sh --campaign e2_sticky_vessl --target workspace/engine-port/results/r2_eval/e2_sticky_vessl/e2_sticky_vessl.sh` **dry-run** → 사용자가 명령·단가·job name 승인 → `--submit`.
6. job name을 이 문서 §7에 追記하고 **이 OVERRIDE를 소진 처리**한다. 회수는 `fetch.sh`(DONE + sha256)까지.

## 7. 승인란

- **사용자 승인**: ☑ **2026-10-04 사용자 "권고 사항대로 진행" — 승인 범위 20 boot, 하드캡 ≈4 h.**
  (메인 세션 전달. 제시된 내용: E2를 VESSL에서 실행, 새 범위 한정 OVERRIDE, 20 boot, 컨테이너 하드캡 ≈4 h.)
  ⚠️이 초안의 §5 금액·§5-2 위험 프로필·§1 presubmit 현황은 승인 **이후** 작성됐다. 메인 세션이 제출 전 사용자에게 이 표를 다시 보여 확인받는 것을 권한다.
- **사용자 재확인(규칙층 rev2 선행조건 1, 필수)**: ☑ **2026-10-04(2차) 사용자 "1,2에 대해서는 진행"** — 메인 세션이 §5-2 위험표(집행 주체 = 스크립트 3겹,
  초과 시 과금, 캡 범위 = 셀 구간+preflight+tail이며 clone·install·결과 반출에는 kill 층 없음, 실패 Job도 과금·자동 재시도 없음)와 "컨테이너 하드캡 ≈4 h" 문언의
  차이, 그리고 Q1이 `OFF_BASELINE_ABOVE_BAND`로 끝날 공산이 크다는 예보(E2VC-5, 비등록)를 제시한 뒤 받은 승인이다. 승인 범위는 불변(20 boot, 보통 $2.95 / 최악 $3.75 / 셀 구간 하드캡 4.0 GPU-h $5.92).
- **승인된 예산 상한**: 1.991 GPU-h / $2.95(보통) · 2.533 GPU-h / $3.75(최악) · **4.0 GPU-h / $5.92(셀 구간 하드캡)**.
- **소진 job name**: **`pdmux-e2-sticky-vessl-e6f424e-20261004135228`** (slug `job-nulj8b3do42r`, 2026-10-04T13:52:33Z 제출, 커밋 `e6f424e`).
  이 OVERRIDE는 **이 Job 1건으로 소진된다. 재개 Job은 §5-1에 따라 재승인 대상**이다.
- **선행조건 상태**: A13 규칙층 감사 ☑(1회차 `NO-GO` 死因 1 → 수리 → rev2 **`GO-with-caveats`**, E2VC-1…11) · 하네스층 국소 재확인 ☑(rev2 §(5) 봉인 8파일) · presubmit 차단 = M4R·TC1 2건 ☑(§1-1) · 사용자 §5-2 위험표 재확인 ☑(2026-10-04 2차) · **커밋 ☑(`8f716f8`; 제출 시 HEAD `e6f424e` — 러너/`scripts/vessl/`은 `8f716f8`..`e6f424e` 바이트 동일, `git diff --stat` 무변경 확인) · staging ☑(226.4 MB = 237,346,884 bytes, 7/7 파일 + `SHA256SUMS`, object volume `objvol-9vgmwmerzwd3` `staging/e2_selftest_archive/` 확인) · dry-run 승인 ☑(2026-10-04 사용자 "권고대로 진행")**.
- **비용(실측, 2026-10-04T14:50:47Z fetch)**: 플랫폼 `Duration` 58m13s(0.970 GPU-h) × $1.48/h ≈ **$1.44**; job_entry 내부 타이밍(`start_utc`13:53:08Z–`end_utc`14:50:32Z)
  기준 `cost.txt` 하한 0.957 GPU-h / **$1.42**(스케줄링·이미지 pull 제외). 승인 예산(보통 1.991 GPU-h/$2.95) 48% 소진, 하드캡(4.0 GPU-h) 24% 소진.
  **결과: 10/20 boot 완주 후 `exit_code=8`(`ABORT_P3`, BANNERS.txt/P3_KEYS.txt — `a_r2_OFF_4162` boot가 `max_total_num_tokens=2722022`로
  나머지 9개 성공 boot의 `2722025`와 3-token 어긋나 중단. 2026-10-04 fetch 즉시 보고, 재개 Job은 미제출 — §5-1 재승인 대상).**
