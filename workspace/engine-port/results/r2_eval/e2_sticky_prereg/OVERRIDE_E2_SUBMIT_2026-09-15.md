# 범위 한정 제출 OVERRIDE — **E2 (sticky 분할 대조), 2026-09-15**

> **지위**: `presubmit.py`의 **reachability BLOCK 2건**을 **E2 1 job에 한정해** 넘기기 위한 문서.
> 전례 `OVERRIDE_VPROBE_SUBMIT_2026-09-01.md` · `OVERRIDE_P1_SUBMIT_2026-08-28.md` ·
> `OVERRIDE_LAMBDA0_SUBMIT_2026-09-14.md`의 형식 요건 5개를 따른다.
> ★**기존 2건은 전부 소진**됐다 — D-none(`OVERRIDE_DNONE_SUBMIT_2026-09-14.md` → job 908534) ·
> λ0(`OVERRIDE_LAMBDA0_SUBMIT_2026-09-14.md` → job 908623). 이것은 **세 번째, 별개의** OVERRIDE다.
> ⚠️**이 문서는 사용자 승인 전에는 효력이 없다.** 승인 없이 `sbatch`를 실행하지 않는다.

## 1. 넘기는 차단 — 출력 그대로 인용 (요건 ①)

2026-09-15 실행(`python3 presubmit.py --registry presubmit_registry.json`, 저장소 HEAD `4d1a0ec`):
```
=== presubmit ===
  OK     registry_append_only 삭제 0건
  OK     line_citations --- check: 93 compared, 0 violation(s)  OK
  OK     doc_facts      --- 11 fact(s), 14 occurrence(s) compared, 0 violation(s)  OK
  OK     citation_stops --- 0 added line(s), 0 violation(s), 16 rules  OK
  OK     reachability   spec_cp0_arm_reqactive.json -> DISCRIMINATING
  OK     reachability   spec_cp0_capacity.json -> DISCRIMINATING
  BLOCK  reachability   reachability_spec.json -> SINGLE_LABEL_FORCED
  BLOCK  reachability   reach_spec_rev3_A.json -> RESTRICTIONS_INERT

제출 금지 — 차단 2건. 규율 도구가 통과하기 전에는 job을 제출하지 않는다.
```
(superseded 5행은 지면상 생략 — 전부 `OK`.)
★**차단은 정확히 이 2건뿐이다.** 위생 검사 3종은 이 회차가 추가한 문서 6개를 포함해서도
**0 위반**이다(E2 문서군이 `doc_facts`·`line_citations`를 깨지 않았다).
- `reachability_spec.json` = `results/m4r_confinement/` (M4R confinement 트랙)
- `reach_spec_rev3_A.json` = `results/tc1_model_attrib/` (TC1 model-attribution, 2026-08-28자)

## 2. 왜 이 회차와 인과 무관인가 (요건 ②)

- **M4R (`SINGLE_LABEL_FORCED`)**: spec 자신의 측정된 제약이 두 라벨을 배제해
  `RESIDUAL_INCONCLUSIVE`만 도달 가능하다(규칙층 감사 2회 전부 `NO-GO`, rev3 부재).
  정본이 이를 **현재 상태**로 등재했다(`PROJECT_STATUS.md:9273`).
- **TC1 (`RESTRICTIONS_INERT`)**: 제약이 격자를 균일 축소해 아무것도 지우지 못한다. 도구
  자기감사(커밋 `5d180a6`) 이후 rev3 spec은 `BLOCKED`이고 **rev4가 없어** `superseded_by`가
  가리킬 후속이 없다.
- **이 회차와의 관계**: E2의 결정 경로는 `e2_realized_mix.py` · `e2_label.py` · `e2_sticky.sbatch`와
  job 908623의 아카이브 telemetry뿐이다(사전등록 rev3 §5·§6 + 追記 A1–A12). 두 spec의 산출물·
  규칙·측정을 **읽지도 쓰지도 않는다**. ⇒ 두 차단은 E2의 판정에 어떤 경로로도 들어오지 않는다.

### 2-1. ★**단, λ0 회차보다 한 겹 더 조심해야 하는 점** (자진 공시)

정본은 M4R의 *"유일한 해소 경로 = **`PDMUX_STICKY_PARTITION=1` 신규 측정**"*(`PROJECT_STATUS.md:9273`)
이라고 적었고, **E2가 바로 그 노브를 켠다**. λ0 회차는 *"그 트랙을 건드리지 않는다"* 로 끝났지만
E2는 그렇게만 말하면 부정확하다. 정확한 관계는 셋이다:

1. **E2는 M4R이 아니다.** E2의 등록 셀(`a_r4`·`a_r2`·`b_r3`)·판정량(Q1 실현 분할, Q2 achieved)·
   결정 규칙(R1–R6)은 M4R의 estimand(`R_matched`, confinement 비)와 **다른 양**이다.
2. **E2의 결과는 M4R에 대해 아무것도 말하지 않는다.** 결과 문서는 *"M4R이 해소됐다"*·
   *"confinement 비용이 측정됐다"* 를 **쓸 수 없다**(신규 인용 금지 조항으로 §4에 등록).
3. **나중에 E2 아티팩트를 M4R이 재사용하려면 그것은 별개의 사전등록**이다 — 다른 판정량,
   다른 규칙층 감사. 이 OVERRIDE는 그 재사용을 **승인하지 않는다**.

⇒ 이 구분이 유지되는 한 **두 차단은 여전히 E2의 판정 경로 밖**이다.

## 3. 이 OVERRIDE가 **덮지 않는** 것 (요건 ③)

- `doc_facts`·`line_citations`·`citation_stops` 차단은 **넘기지 않는다**(현재 0 위반이며,
  제출 직전에 다시 0이어야 한다 — §6의 한 창 규율).
- **하네스·규칙층 감사를 대신하지 않는다**: 규칙층 `GO-with-caveats`(`396927b7…`),
  하네스층 `GO-with-caveats`(`9768ec78…`)가 이미 나와 있고, 그 선행조건 4건은 **이행 완료**다.
- **다른 트랙의 제출을 승인하지 않는다.** 이 문서는 **E2 1 job 한정**이며, 재실행·설계 변경·
  다른 셀 집합은 **새 OVERRIDE**를 요구한다.

## 4. 이 회차가 사는 것과 못 사는 것 (요건 ④ — 결과 문서가 문자 그대로 승계)

**사는 것**: 등록 셀에서 sticky ON이 실현 분할을 D44로 고정하는가(R1) · 같은 offered에서
achieved가 어떻게 변하는가(R2, paired n=4 + CI) · 그 셀들의 one-knob 검증(R3).

**못 사는 것** (rev3 §10 + 승계 caveat):
- **게이트 #6(λ\*_SLO)을 닫지 않는다** · **P2 블로커 ②를 해소하지 않는다** ·
  **Claim D/E 등급 불변** · **새 성능 판정 0건** · **λ0R-8(iii)(true-dual 정규화 교락)을
  해소하지 않는다**(두 arm의 prefill 축이 동일 — E2C-5).
- ★**E2C-21**: shape A에서 처치되는 시간의 **86–89%가 prefill 유휴 구간**이므로 Q2가 사는 것은
  *"멀티플렉싱할 prefill이 없는 동안 decode를 44 SM에 묶어 둔 비용"* 까지다(정본 감사 2F9 계열).
  **정책 서술 금지**(게이트 #2 + 이 caveat의 이중 금지).
- ★**신규(이 문서)**: *"E2가 M4R을 해소했다"*·*"confinement 비용을 측정했다"* **인용 금지**(§2-1).

## 5. 예산과 한계 (요건 ⑤)

| 항목 | 값 |
|---|---|
| boot 수 | **20**(`a_r4` 8 · `a_r2` 8 · `b_r3` 4) |
| 등록 보통 | **1.803 GPU-h**(6,489.6 s — 감사자 재검산 일치) |
| 최악 코너 | **2.338 GPU-h**(8,417.5 s) |
| `--time` 하드 캡 | **03:00:00** = 3.0 GPU-h (P14 전역 시계 가드가 P7 캡 합계 13,159 s를 먼저 차단) |
| 비교 | λ0 job 908623 실지출 **1.391 GPU-h** |
| 트랙 장부 | E2는 **신규 트랙**(지출 0). R2 correctness 1.322778 · λ0 1.391389 GPU-h는 불변 |

**초과 시**: `--time` 소진은 SLURM이 강제하고, 그 전에 P14가 다음 boot를 시작하지 않는다.
부분 실행 집합으로 **재라벨하지 않는다**(D23) — `UNRESOLVED_BUDGET`으로 남는다.

## 6. 한 창 규율 (λ0R-10 승계 — 지난 회차가 어긴 절차)

제출은 **한 창** 안에서 아래 순서로만 한다. 중간에 다른 작업이 끼면 **처음부터 다시**:

1. `presubmit.py --registry presubmit_registry.json` 재실행 → **차단이 §1의 2건과 정확히 같은지** 확인.
2. 두 셀프테스트 재실행 → `e2_realized_mix.py --selftest`(7/7) · `e2_label.py --selftest`(8+10).
3. ★**커밋**(P5) — 사전등록·追記·하네스 3종·판정서 4종·이 OVERRIDE. **커밋 없이 제출 금지.**
4. `sbatch e2_sticky.sbatch`.
5. job id를 이 문서에 追記하고 **이 OVERRIDE를 소진 처리**한다.

## 7. 승인란

- **사용자 승인**: ☑ **승인 (2026-09-15)** — 사용자 지시 *"진행"*.
  제시된 내용: 예산 3층(**1.803 보통 / 2.338 최악 / 3.0 GPU-h 하드 캡**) · 사는 것(R1·R2·R3) ·
  **못 사는 것 전부**(게이트 #6 미해소 · Claim D/E 불변 · 새 성능 판정 0건 · λ0R-8(iii) 미해소 ·
  ★E2C-21 = shape A 처치 시간의 86–89%가 prefill 유휴 구간) · §2-1의 M4R 노브 공유 공시.
- **승인된 예산 상한**: 1.803 GPU-h(보통) / 2.338(최악) / **3.0(`--time` 하드 캡)**.
- **소진 job id**: **없음 — 이 OVERRIDE는 아직 소진되지 않았다.**

### 7-1. ★제출 시도 기록 (2026-09-15) — **스케줄러가 거부, GPU 지출 0**

한 창 규율 §6을 1→4단계까지 실행했고 **4단계에서 외부 사유로 막혔다**:

1. ✅ `presubmit.py` 재실행 — 차단이 §1의 2건과 **정확히 일치**(위생 3종 0 위반).
2. ✅ 셀프테스트 3종(venv) — `7/7` · `8 도달성 + 10 되돌림 회귀` · `test_sticky_partition 12 OK`.
3. ✅ **커밋 `b917bfc`**(P5 충족 — 13 파일).
4. ❌ `sbatch e2_sticky.sbatch` →
   ```
   sbatch: error: Your account has expired or exceeded the allocated CPU time.
   Please contact the account manager. (account@ksc.re.kr, 042-869-0597)
   sbatch: error: Batch job submission failed: Unspecified error
   ```

**진단**: 계정 수준 할당량 문제이며 이 회차의 설계·하네스·규율과 **무관**하다.
`sacctmgr show assoc user=ehmoon`의 `GrpTRESMins`/`MaxTRESMins`는 **비어 있고**
(association 층 제한 아님), `sshare` RawUsage 489,800. 직전 job **908623은 정상 완주**
(2026-09-14, `COMPLETED 01:23:29`)했으므로 그 이후에 계정 할당이 만료·소진된 것으로 보인다.

**상태**: 이 OVERRIDE는 **유효하며 미소진**이다. 계정이 복구되면 §6의 **1·2단계를 다시 실행한 뒤**
(한 창 규율 — 시간이 지났으므로 재확인이 필요하다) 4단계만 재시도한다. **새 승인은 불필요**하다
— 승인된 설계·예산이 바뀌지 않았기 때문이다. 단 **설계나 예산이 바뀌면 새 OVERRIDE**다.
