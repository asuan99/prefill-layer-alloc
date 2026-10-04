# 판정서: E2 sticky 대조 VESSL 이식(追記 A13) 규칙층 감사 (2026-10-04)

> **지위**: claims-auditor 규칙층(이식) 감사 1회차. GPU 0. 파일 수정 0건. 감사자의 최종 보고를 메인 세션이 이 파일에 옮겼다. 성능 판정 0건, Claim 등급 변경 0건.
> **대상**(전부 미커밋, sha256 앞 8자리): `PREREG_E2_VESSL_A13_2026-10-04.md` `a5e703fe`, `e2_sticky_vessl.sh` `3bccc6e7`, `e2_vessl_plan.py` `aa5cf4aa`, `stage_e2_inputs.sh` `4b27bb0a`, `OVERRIDE_E2_VESSL_2026-10-04.md` `7b39bab0`, `tests/test_e2_vessl.py` `0fb25726`.

## 최종 판정: **`NO-GO`**. 死因은 1건이고 문안 하나에만 있다. 수리 범위가 좁다.

- **死因 1 (N1, N3 측면 포함)**: A13 §4-1이 1차 판정량 Q1의 답을 **R1 ON 절**로 돌려놓았다(OVERRIDE §4 "사는 것"과 A13 §6-7에도 같은 문구). 그런데 ON 절은 엔진이 강제하는 양성 대조다.
- 나머지 이식 요소(digest 강제, 엔진 트리, T_cap, 예산, 하드캡, 재개, CPU, green read-out)는 반전 시험에서 판정을 뒤집지 못했다. 그래서 전부 caveat이다.
- 수리 뒤 국소 재감사에서는 그 한 조항만 확인하면 된다. **재감사 전에는 GO가 아니다.**

### 死因 1: Q1을 ON 절로 답하는 독해 규약

**문안**: A13 §4-1 *"Q1 — 'sticky ON이 shape A의 실현 분할을 D44로 유지하는가' — 는 R1의 ON 절로 답해진다"*. OVERRIDE §4 *"(R1 — Q1은 ON 절로 답해진다, A13 §4-1)"*.

**등록과의 충돌**:
- rev3 §1의 질문은 *"sticky 노브 하나를 ON으로 **올리면** … D44로 고정되는가"*, 즉 대조다.
- rev3가 문자 그대로 승계한 E2C-3은 *"ON에서 D44 ≈100% ⇒ sticky 효과 실증"*을 **인용 금지**로 둔다. 근거는 "ON의 인덱스는 엔진이 {0,2}로 강제, R1의 ON 절은 **양성 대조**"다.
- 감사자 코드 확인: `multiplexing_mixin.py:1169-1179`. sticky ON이고 decode가 비어 있지 않으면 `stream_idx = _sticky_fixed_idx`(=2), 비어 있으면 idx 0이다.

**항등식인 이유**:
- ON arm의 decode-busy 표본은 idx ∈ {0, 2}에만 있다. 따라서 E-cnt(ON) = 100 − idx0%다.
- `e2_label.py`는 R1보다 먼저 R3(ii)(idx0 > 5.0%이면 짝 UNRESOLVED)를 적용한다. 그래서 **live 짝에서는 E-cnt(ON) ≥ 95 > 90이 구성상 성립한다**.
- ON 절이 거짓이 되는 길은 배선 실패(P1/P2/R4′ abort 층)나 idx0 표본이 queue>0 쪽에 몰리는 read-skew뿐이다.
- 그러므로 이 규약 아래 Q1의 "거짓"은 계측 실패로만 도달할 수 있다. 판정량이 엔진 강제 항등식이 된다(N1), 예측의 출력공간이 퇴화한다(N3 측면).

**수치 반전 (판정 분기가 실제로 바뀐다)**:
- 파일럿(λ0 VESSL `a_r2`/4386 OFF, 감사자 재산출): E-qcond 29/141 = 20.57% ≥ 20.0.
- 비등록 plug-in 이항 예보(감사자 계산, **인용 금지**, 설계 입력 금지): VESSL `a_r2` seed당 OFF 절 실패 확률 0.533, 4 seed 중 하나라도 실패할 확률 **0.953**. 비교용 KISTI(18/102): 0.253 / 0.688.
- 등록 독해(rev3 + A5 + E2C-3)에서 이 분기의 결과는 `OFF_BASELINE_ABOVE_BAND`, Q1(대조)은 **성립 미확인**이다. A13 독해에서는 같은 데이터로 **Q1에 "답했다"(예)**가 된다.
- 즉 확률 ≥0.95인 분기에서 1차 질문의 판정이 뒤집힌다. R1 라벨 자체는 digest로 고정돼 바뀌지 않는다. 바뀌는 것은 Q1의 판정과 OVERRIDE가 산다고 적은 범위다.

**caveat으로 강등할 수 없는 이유**: A13은 digest로 고정된다(`A13_DIGEST`). 그대로 돌리면 고정된 등록 문서가 반대 독해를 권위로 싣는다. 판정서의 인용 금지 문장과 등록 본문이 서로 모순되는 상태가 된다.

## 작성자 지목 8개 항목

### 1. R1 OFF 밴드: **(A) 확정**. 단 A13이 (A)에 붙인 Q1 독해는 삭제한다(= (A′)).

- **(B)(C)(D)는 기각한다.** 셋 다 파일럿 한 값을 본 뒤 고르는 수이고, 고르는 순간 예보 라벨이 뒤집힌다(반전 표). 채택하면 그 자체가 N2 死因이다.
  - (C)는 rev1 死因 F1과 같은 형태다(추정량 메뉴와 규칙을 동시에 바꾼다).
  - (B)의 70pp는 ON ≥ 90.57이면 파일럿에서 통과로 뒤집힌다. ON 절이 어차피 >90을 요구하므로 사실상 자동 통과다.
- **(A)는 등록 문턱 20.0이 digest로 고정돼 움직일 범위가 0이다.** 반전을 만들 수 없으므로 caveat다.
- 교훈 251(조이는 방향) 반전 시험은 작성자가 양방향 모두 했다(t=18.3에서 `a_r4`도 무너짐). 감사자도 재산출해 일치를 확인했다: `a_r4` 4 seed 18.33 / 15.94 / 17.06 / 16.60 · `a_r2` 20.57 · KISTI `a_r2` 17.65(18/102) · `a_r3` KISTI 18.67, VESSL 16.80.
- A5의 4분지와의 관계: A5는 OFF 실패가 ON에 대한 **거짓 귀속**이 되지 않게 막을 뿐이다. ON 절 성립을 Q1의 **양성 답**으로 승격하는 근거는 아니다. 4분지 라벨은 ON 정보를 이미 보존한다(`OFF_BASELINE_ABOVE_BAND` 단독 ⇔ `on_clause_failures` 비어 있음). 새 독해 규약은 필요 없다.
- **사용자에게 알릴 사실**: VESSL에서 E2의 1차 질문 Q1은 높은 확률로 "성립 미확인"으로 끝날 것으로 예보된다. 실질 산출은 R2(paired achieved), R3, R4/R4′/R4″, P13이다. 실행 여부는 사용자 결정이다. 死因은 아니다: `STICKY_REALIZES_A`는 여전히 도달 가능하고, 검출력 문제일 뿐 출력공간이 비지는 않는다.

### 2. 엔진 트리 = HEAD: **반증 실패, caveat (E2V-1 유지)**

- `git diff --stat 4d1a0ec 1d47d3e -- src/`는 공백이다. `1d47d3e..HEAD`의 src 변경은 `multiplexing_mixin.py` 1파일뿐이고, src 사본 sha는 `ba5835aa…`로 등록값과 같다. 908623 manifest의 해당 줄은 `e2a97b42…`다.
- 훅 다섯 곳의 인자는 OFF에서도 평가되지만 전부 그 자리에서 바인딩돼 있다. NameError 경로는 없다.
- `--engine-diff`를 λ0 VESSL manifest(트리 동일)에 돌리면 `DRIFT (0 files differ)`, rc=3이 나온다. 0개 차이도 거부한다.
- E2-α 면역 논리(두 arm 같은 트리)는 성립한다. 단 HEAD 트리로 GPU 부팅을 한 적이 없다 → E2VC-7.

### 3. 하드캡 (A4 대체): **"≈4 h"와 정상 경로에서는 정합. 미정합 경로가 있으므로 사용자 재확인을 필수로 승격한다.**

- 마감 = T0(job_entry `start_utc`) + 14,400 − 600 = T0 + 13,800 s.
- λ0 VESSL 실측으로 tail과 반출은 매우 짧다(루프 종료 T0+6,813 s, 컨테이너 6,818 s). hang이 없는 경로의 컨테이너 상한은 약 13,800 + 15 + 수십 초로 4.0 h 아래다.
- **4 h를 넘을 수 있는 경로**: (1) preflight hang — sync·셀프테스트·unittest에 timeout이 없고 watchdog는 bench·server·fake 패턴만 죽인다 (2) tail hang — `e2_label.py`, green summary (3) 반출 지연. 세 경로에는 kill 층이 없다.
- 사용자 승인 문언은 **"컨테이너 하드캡 ≈4 h"**인데 구현은 셀 구간 캡이다. OVERRIDE §7의 "재확인을 권한다"를 **제출 전 필수 단계**로 바꾼다(필수 수리 4).
- 최악 코너 boot당 여유 219.4 s는 감사자 재계산과 일치한다.

### 4. E2C-36…39를 결과 문서 승계 caveat로 승격: **승인**

해소 상태 문안은 정확하다: E2C-36 0.985–1.001×λ\*\_VESSL · E2C-37 B0는 기판 "종류"만 확인 · E2C-38 group 구성만 실측 · E2C-39 셀 구간 캡. E2C-38의 "부분 해소"는 green read-out이 판정 입력이 아닌 동안에만 유효하다.

### 5. 재개: **반증 실패, caveat**

- 짝 단위 복사, 반쪽 짝 재실행, CAPPED 비재실행, UNITS·digest 불일치 시 abort 모두 테스트로 확인했다(대조군 포함).
- Job 혼합이 R4″의 Y와 R5 산포를 키우면 각각 WITHIN과 UNRESOLVED 쪽, 즉 보수 방향이다. R2는 데이터 없이 반전 계산이 불가하다 → E2VC-6.
- OVERRIDE §5-1의 "재개 Job은 사용자 재승인"은 유지한다. 코드로 강제되지 않는 문서 규칙임을 공시한다.

### 6. T_cap 교체 · per-boot 상수 · 여유값: **반증 실패, caveat**

- T_cap 식 `⌈3·round(d,0.1)⌉`가 KISTI 393/428/861을 재현한다(908623 duration 131.013 / 142.454 / 286.845). VESSL duration(133.615 / 142.902 / 289.667)은 `lam0v_descriptive.json`과 일치, 결과 401/429/870. 판정이 갈리는 창은 ON/OFF ≈ 2.94–3.0×뿐(E2V-2).
- per-boot 오버헤드는 BUDGET.txt 연속 차분으로 감사자가 재계산했다. A 159.11–160.10, B 118.55–120.93, cold 92.06 s. 일치.
- P13 30 s와 preflight 600 s는 예산표에만 쓰인다. 입장 판정은 실제 시계로 한다.

### 7. CPU 핀 없음 · green 양 arm · 해소 조건 3의 갈음

- **CPU 핀**: 걸지 않는 결정은 수용한다. 그러나 A13 §2-6(3)과 §6-8의 **"부호 반전 경로 없음"은 반증되지 않은 과잉 주장**이다. 서버와 클라이언트가 같은 CFS quota 11을 공유하고, ON과 OFF는 decode 단계율(44 대 108 SM)이 달라 CPU 수요가 다를 수 있다 → arm 비대칭 스로틀 경로가 존재한다 → E2VC-2.
- **green read-out**: 양 arm 대칭. caveat E2V-6.
- **해소 조건 3**: λ0 VESSL Job(트리 `1d47d3e`)으로 갈음하는 것을 수용한다. 단 "E2 트리 부팅 검증"이 아니고, P3 게이트는 수준값 48 / 2,722,025를 강제하지 않는다 → E2VC-7.

### 8. presubmit 선행 수리 범위: **맞다. 수리 방법에 조건을 붙인다.**

- 4개 spec(`m4r_confinement/reachability_spec.json`, `tc1_model_attrib/reach_spec_rev3_A.json`, `cp_baseline/spec_cp0_arm_reqactive.json`, `cp_baseline/spec_cp0_capacity.json`)의 `rule_path`가 `/scratch/ehmoon/whlee/…`인 것을 확인했다.
- `?`를 OVERRIDE로 덮지 않는 것은 옳다(게이트 #21의 거울상).
- 수리 방법: spec 바이트를 고치는 것보다 **도구 수준 접두 치환**(`/scratch/ehmoon/whlee/prefill-layer-alloc` → 저장소 루트)이 낫다. spec을 고친다면 각 트랙 문서에 追記한다.
- 수리 뒤 차단이 **정확히 M4R `SINGLE_LABEL_FORCED`와 TC1 `RESTRICTIONS_INERT`**이고 cp_baseline 2건이 `DISCRIMINATING`일 때만 OVERRIDE §1이 유효하다.
- 부수 사실: λ0 VESSL Job 문서에는 presubmit 결과 기록이 없다. 그 Job을 우회의 전례로 쓰지 않는다.

## 독립 확인

- **(i) λ0R-8(iii)**: A13, OVERRIDE, 러너, helper 어디에도 해소·판정·약화 주장이 없다. 금지 문장이 문자 그대로 있다(A13 §7, OVERRIDE §4). **반증 실패.** `handoff-report/experiment_dag_and_execution_2026-10-04.md`의 "N1 산출 λ0R-8(iii) 판정"은 E2C-5 위반(n4 판정서가 이미 지적) — 그 문서를 E2 결과 해석 근거로 쓰지 않는다.
- **(ii) 규칙 불변**: 14파일·A13·config의 live sha가 등록값과 같다. `e2_label.py`, `e2_realized_mix.py`는 `git show b917bfc:` 바이트와 같다. 서버·warm-up·bench·estimator 명령줄 리터럴도 sbatch와 같다. **digest가 보장하는 범위**는 규칙·추정량 파일 바이트와 등록 문서뿐이고, 러너·helper 자체는 커밋으로만 고정된다 → 하네스층 판정서가 러너·helper·test의 sha를 봉인해야 한다(필수 수리 5). **출처 허위 1건(교훈 80)**: `e2_vessl_plan.py`의 `REGISTERED_DIGESTS` 주석이 "b917bfc as the only commit touching them"이라고 적지만 이식성 판정서는 b917bfc에 없다(40a7be2 추가, b330ae8 수정). 상수 값은 옳고 주석만 거짓 → 필수 수리 2.
- **(iii) 예산** 재계산: 보통 7,168.5 s = 1.991 GPU-h = $2.95 · 최악 9,117.3 s = 2.533 GPU-h = $3.75 · 하드캡 4.000 GPU-h = $5.92 · 최악 코너 여유 219.4 s/boot. 전부 일치.
- **(iv) 변이 테스트**: 40/40 PASS(감사자 로컬). 항등식이 아닌 것: digest 1바이트 대조군, 엔진 트리 대조군 3종, 입장 규칙 삭제 변이, 재개 대조군. **부분 항등식**: 하드캡 집행 변이 검증은 DRYRUN 가짜 bench 경로만 시험한다. 실제 경로의 bench `timeout -k 30`, watchdog 패턴 `sglang.bench_serving`/`sglang.launch_server`, BUDGET/CAPPED `.complete` 분류를 고장 내도 어떤 테스트도 실패하지 않는다(교훈 9·18 계열). 라벨은 fail-closed라 결과 영향은 없고 과금 쪽 위험 → E2VC-4, 필수 수리 5.

## 반전 시험 표

| 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 |
|---|---|---|---|
| **Q1 독해 규약(A13 §4-1)** | rev3+A5+E2C-3 ↔ A13 "ON 절로 답" | **반전**: Q1 "성립 미확인" → "예" | OFF `a_r2` 20.57%(29/141). P(OFF 실패) ≈0.953. live 짝에서 E-cnt(ON) ≥95는 R3(ii)로 구성상 성립 → **死因 1** |
| R1 OFF 문턱 (A) | 등록 리터럴 20.0, 범위 0 | 없음 | digest 고정 |
| R1 대안 (B) ON−OFF ≥70pp | 파일럿 뒤 선택 | 반전 | ON ≥90.57이면 통과 → 채택 시 N2 |
| R1 대안 (C) E-qcond 제외 | 파일럿 뒤 선택 | 반전 | 나머지 세 값 9.21 / 13.12 / 9.30 → N2 |
| R1 대안 (D) 25 | 파일럿 뒤 선택 | 반전 | 20.57 < 25 → N2 |
| T_cap KISTI ↔ VESSL | 393/428/861 ↔ 401/429/870 | ON/OFF 2.94–3.0×에서만 | 최악 코너 ×2.5/×1.5는 어느 쪽에서도 캡 미도달 |
| 하드캡 | 3.0–4.5 h | 없음(보통·최악 20/20) | 마지막 boot 입장에 캡 ≥ 10,231.8 s면 충분 |
| per-boot 오버헤드 | 실측 → +219 s/boot | 없음 | 여유 219.4 s/boot, R4″ 유지 |
| preflight / P13 여유값 | 600 / 30 → 실측 | 없음 | 총 여유 4,168 s |
| CPU 핀 | 없음 ↔ `taskset 0-10` | 계산 불가(ON 데이터 없음) | arm 비대칭 스로틀 경로 존재 → caveat |
| 엔진 트리 | HEAD ↔ 1d47d3e | R1–R5·P13 불변, R6만 | 양 arm 동일 |
| green read-out | 양 arm ↔ 없음 | 없음 | — |
| 재개 Job 혼합 | 0 ↔ k 짝 | R4″·R5 보수 방향, R2 계산 불가 | caveat + 재승인 |

## caveat 목록 (결과 문서·정본이 문자 그대로 승계)

- **E2VC-1 (R1·Q1)**: "R1 라벨이 `STICKY_REALIZES_A`(또는 `cells_evaluated`를 병기한 `_ON_REMAINING_CELLS`)가 아닌 한, 이 캠페인은 'sticky ON이 shape A의 실현 분할을 D44로 고정/유지/실현한다'를 확립하지 않는다. R1 ON 절의 성립은 엔진이 {0,2}로 강제한 양성 대조의 통과이며(E2C-3) Q1의 답이 아니다. `OFF_BASELINE_ABOVE_BAND`를 'VESSL OFF가 KISTI OFF보다 D44를 더 점유한다'나 'sticky가 무의미하다'로 읽지 않는다."
- **E2VC-2 (CPU)**: "CPU 축이 판정 부호를 바꿀 수 없다는 주장은 검증되지 않았다(A13 §2-6(3), §6-8 인용 금지). 결과 문서는 boot별·arm별 `cpu.stat` Δ`nr_throttled`·Δ`throttled_usec`를 병기하고, arm 간 차이가 있으면 R2 라벨 옆에 CPU 스로틀 비대칭을 명시한다."
- **E2VC-3 (하드캡)**: "4.0 h는 job_entry 시작부터 잰 셀 구간 캡이다. preflight·tail·반출에는 kill 층이 없다. '컨테이너가 4 h 안에 끝났다'는 `meta/timing.txt` `start_utc`/`end_utc` 실측 없이 쓰지 않는다(E2C-39 강화)."
- **E2VC-4 (집행기 검증 범위)**: "하드캡 3겹 집행의 변이 검증은 DRYRUN 가짜 bench 경로에 대한 것이다. 실제 bench timeout·watchdog 패턴·`.complete` 분류는 변이 검증되지 않았다."
- **E2VC-5 (예보)**: "`OFF_BASELINE_ABOVE_BAND`가 나올 확률이 높다는 사전 예보(파일럿 1셀·1seed, 감사자 plug-in ≈0.95)는 비등록 수치다. 결과 해석·검출력·설계에 쓰지 않는다."
- **E2VC-6 (재개)**: "재개가 있었으면 R2·R4″·R5를 Job별로도 병기한다. 'E2-α 같은 Job 대조'는 짝 내부까지만 주장한다."
- **E2VC-7 (트리·P3 수준)**: "HEAD 트리(`multiplexing_mixin.py` 1파일 차이)는 이 Job 전에 GPU 부팅 이력이 없다. 해소 조건 3은 λ0 VESSL Job(트리 1d47d3e)로 갈음했을 뿐 E2 트리를 검증한 것이 아니다. P3는 수준값을 강제하지 않으므로 결과 문서는 BANNERS의 (max_mamba_cache_size, max_total_num_tokens)를 48 / 2,722,025와 대조해 병기한다."
- **E2VC-8 (presubmit)**: "λ0 VESSL Job은 presubmit 통과 기록이 없으므로 차단 우회의 전례가 아니다. OVERRIDE는 경로 수리 뒤 M4R·TC1이 2026-09-15와 같은 라벨일 때만 그 2건을 넘긴다."
- A13의 E2V-1…E2V-8과 E2C-36…39(해소 상태 포함), OVERRIDE §2-1·§4의 문자 승계 조항은 그대로 유효하다.

## 제출 전 필수 수리

1. **(死因 1 해소)** A13 §4-1의 "Q1 … 는 R1의 ON 절로 답해진다" 단락과 §6-7의 해당 표현, OVERRIDE §4 "사는 것"의 괄호를 E2VC-1 취지로 교체한다. 그다음 `e2_vessl_plan.A13_DIGEST`를 같은 커밋에서 갱신하고 `test_a13_digest_pinned` 통과를 확인한다. **그 밖의 A13 바이트는 바꾸지 않는다.** (선택) §2-6(3)·§6-8의 "부호 반전 경로 없음"을 "미검증"으로 고친다. 고치지 않으면 E2VC-2가 그 문장을 인용 금지로 묶는다.
2. `e2_vessl_plan.py`의 `REGISTERED_DIGESTS` 주석을 정정한다. 이식성 판정서는 b330ae8 바이트다. 상수 값은 불변.
3. presubmit: 도구 수준 경로 치환(또는 spec 수정 + 각 트랙에 追記) 후 doc_facts 2건을 doc-steward가 정리한다. 그다음 차단이 정확히 M4R·TC1 2건인지, 라벨이 2026-09-15와 같은지 재확인한다.
4. OVERRIDE §5-2 위험표(셀 구간 캡, 무상에서 과금으로의 전환, hang 경로)를 사용자에게 다시 보여 "컨테이너 하드캡 ≈4 h"와의 차이를 확인받는다. 필수다.
5. 하네스층 국소 재확인에서 처리할 것: 실제 bench 줄의 `timeout -k 30 "${BT}s"`, watchdog 패턴이 SRV/BENCH 문자열과 맞는지, BUDGET이면 `.complete`를 쓰지 않는지를 구조 테스트로 추가하고 각각 대조 변이를 둔다. 러너·helper·test의 sha를 봉인한다. (권고) preflight·tail에 남은 시간 기반 `timeout`을 씌운다.
6. 위 1번에 대한 **국소 재감사**: 교체 문안, digest 갱신, 그 밖의 diff가 0인지만 본다.

## 반증 실패 기록

digest 16건 일치(label·mix는 b917bfc 원본과 독립 대조) · 엔진 diff 판별 동작 · T_cap 식과 출처 duration 두 기판 모두 · per-boot 상수 재계산 · 예산 3층과 여유 219.4 s · 명령줄 리터럴 · 셀프테스트 7/7, 8+10 로컬 재현 · OVERRIDE 인용 블록 29줄 중 승계 블록 전부 원문과 바이트 일치 · λ0R-8(iii) 무주장 · OFF 경로 훅의 인자 바인딩 · 파일럿 추정량 재산출이 A13 §4-1 표와 전부 일치. 단 위 수리 전엔 CONFIRMED가 아니다.

## 실현가능성 자기검사 (게이트 #113)

- 수리 1–4는 전부 CPU에서 하는 문서·상수·주석 수정이다. 런타임 노브를 움직이지 않는다(교훈 250). A13 수정은 `A13_DIGEST` 갱신과 결합돼 있어야 Job이 돈다.
- 수리 3은 경로 치환 후 M4R·TC1 판정이 2026-09-15와 달라질 수 있다. 그때는 OVERRIDE §1 追記로 처리한다. 감사자는 그 결과를 예단하지 않는다.
- 감사자의 이항 예보 자신이 새 자유 표면이 되지 않도록 비등록·인용 금지로 묶었다(E2VC-5).
- (B)·(C)·(D) 기각으로 Q1이 VESSL에서 미확립될 공산이 크다는 비용은 숨기지 않고 사용자 결정 사항으로 넘긴다.
- 등급 자기검사: 死因(N1, 수치 반전 동반)이 있으므로 caveat으로 강등하지 않았다. 나머지 자유 표면은 반전을 못 만들었으므로 `NO-GO` 사유로 쓰지 않았다. 이전 판정서의 지적(E2C-3, 이식성 Q2)은 코드(`:1169-1179`)와 원자료로 독립 재검증했다.

## 참조 파일

`e2_sticky_vessl/PREREG_E2_VESSL_A13_2026-10-04.md`(§4-1, §2-6, §6) · `OVERRIDE_E2_VESSL_2026-10-04.md`(§4, §5-2, §7) · `e2_vessl_plan.py`(`REGISTERED_DIGESTS` 주석) · `e2_sticky_vessl.sh`(bench timeout, watchdog) · `tests/test_e2_vessl.py`(TestBudgetEnforcer는 가짜 경로만 시험) · `e2_sticky_prereg/PREREG_E2_STICKY_REV3_2026-09-15.md`(§1, §6, §9 E2C-3) · `e2_sticky_prereg/e2_label.py`(R3(ii)를 R1보다 먼저 적용) · `src/multiplex/multiplexing_mixin.py:1169-1179` · `lambda0_vessl/lam0v_…/BUDGET.txt` · `results/m4r_confinement/reachability_spec.json` 외 spec 3개.
