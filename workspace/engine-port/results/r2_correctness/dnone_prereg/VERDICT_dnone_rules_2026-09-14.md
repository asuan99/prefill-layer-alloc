# 규칙층 감사 판정서 — D-none 사전등록 (rev1)

> ★**전사 공시**: 이 판정서는 claims-auditor(read-only, 쓰기 권한 없음)가 보고 본문으로 낸
> 판정 전문을 **메인 세션이 전사**한 것이다. 내용 변경 없음. 단 **전사 자체는 제3자 검증이
> 불가능하다**(감사자가 직접 쓴 파일이 아니다) — 이 한계를 등재한다. 다음 회차의 감사자는
> 본문 인용 시 이 사실을 병기해야 한다.

**감사자** claims-auditor (적대적, read-only) · **일자** 2026-09-14 · **GPU 신규 지출 0**
**대상** `dnone_prereg/PREREG_DNONE_2026-09-14.md`
sha256 `b5d1a244470dbe53d911d6d0f65ca3e8d38bca0bddad21b273861b1b97e2b19e` (재계산 일치), 321줄

## 등급: `NO-GO`

死因 **4건**(DNR-1…DNR-4). DNR-1이 지배적이며 단독으로 회차를 죽인다. 네 건 모두 **최소 수리가
문서 수정 수준**이고 하네스·엔진·채점기 변경을 요구하지 않는다 — rev2는 값싸다.

---

## §0. 감사자가 실제로 돌린 것 (게이트 #110 이행 — 러너/작성자 수치를 상수로 승계하지 않았다)

| 재검증 | 결과 |
|---|---|
| 핀된 12 sha 전부 재계산 | **전부 일치**(`PREREG_RERUN` 현재 = `fa304128…`, §0-c 주장대로 드리프트 확인) |
| `test_r2_correctness_dnone.py` 36 테스트 | `OK` |
| 변이 (a) 가드 재수출 제거 → 실제 파일에 적용 | **`FAILED (failures=7)`** — 등록된 7개 이름과 **정확히 일치** |
| 변이 (b) `DN`→`boots.txt` | **`FAILED (failures=4)`**, 단정문 `'L1 TD1 L2 TD2 DN1 ' != 'L1 TD1 L2 TD2 '` **문자 일치** |
| `git show 83d8cb9d…:…/r2_correctness.sbatch` sha | `f39b167b…ac0403` = `job_908179/provenance.txt` 기록값 ⇒ argv 픽스처 **비항등식** |
| `r2_correctness_check.py` 전 open 지점 | `:135 :190 :221 :366 :373 :502` — **glob/listdir/walk 0건** ⇒ 설계 C 전제 참 |
| `x1_cross_job_compare.py` | `LABELS = ["L1","TD1","L2","TD2"]` 하드코딩, glob 0건 ⇒ DN 불가시 |
| `srv_*.log` glob | `r2_correctness.sbatch:664`에 존재하나 **`INSTR=1` 전용**이고 출력은 `instrument/`뿐 ⇒ 등록 `INSTR=0`에서 미실행 |
| `diag_boots.txt`·`BOOT_FAILURES.txt`·`########## boot=` 소비자 | 저장소 전수 **0건**(`cp_baseline/af1_analyze.py`는 타 트랙) |
| `e5_family_probe.py` 재실행 | VERIFIED_B3 = **12/12 (grep -E) · 12/12 (grep -P) · 12/12 (python re)**, 음성 발화 Y6만, **Y4(`torch.OutOfMemoryError`) 3엔진 전부 미발화** |
| `presubmit.py --registry …` | **BLOCK 2건** = 등록된 그 2건과 일치, `exit 1` |
| `check_citation_stops.py --file` (3 신규 문서) | 0 위반 / 16 rules |
| `unittest discover` 전체 | **`Ran 681 tests … FAILED (failures=1)`** ← §4-4(2) 위반 (DNR-4) |
| `sacct -j 907032,907100,907456,907959,908020,908179` | 433+574+554+1281+549+596 = 3,987 s = **1.107500 GPU-h**, 등록 0.955000 / 등록 밖 0.152500 — §7 장부 **전부 일치** |
| 예산 모형 `153 + 111·n` | n=4 → 597 s vs 실측 596 s(1 s), n=5 → **708 s = 0.196667 h** ≈ 0.197 ✓; 108 s = 0.030 h ✓; 최악 96×5+900+20 = **1,400** ✓; `wait_health … 96`(5 s/poll) ✓; `--time=02:30:00` ✓; `BASEPORT=$((30000 + (${SLURM_JOB_ID:-0} % 5000) * 6))` ✓ ⇒ boot 5 상한 ✓; `n_arms ≤ 5` fail-closed ✓ |
| `CARRYFORWARD_INVENTORY` 직접 계수 | NP 10 · NPC 10 · D13-i 2 · N 13 · A908 7 · RRC 13 · RA3 12 · RA4 12 · RR 19 = **98** ✓ (분류 49/32/15 = 96 ✓) |
| 하네스 줄 수 | 573 → **684** ✓ ; 선재 3 모듈 **62 테스트 `OK`** ✓ |
| 앵커 이동 주장 | `:216→:294` ✓ `:507-513→:608-619` ✓ `:285/:327→:330/:369` ✓ `:314-325→:397-408` ✓ ; `r2_correctness_check.py` 헤더 `:67-78` ✓ ; `*_prereg/*` glob `:364-368`(문서는 `:365-369`, 1줄 오차) |

★**감사자 자신의 부작용 공시(게이트 #240)**: `presubmit.py`를 돌린 것이
`m4r_confinement/reachability_verdict.json`·`tc1_model_attrib/reach_verdict_rev3_A.json`을
**재작성**했다(mtime 09-14 16:23). `git diff HEAD --` 두 파일 모두 **빈 출력 = 바이트 동일** 확인.
이 도구는 read-only가 아니다 — 다음 회차는 이 사실을 §4-4에 적어야 한다.

---

## §1. 반전 시험 표 (필수)

| 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 (원자료) |
|---|---|---|---|
| **V의 "형성" 조작화**(§4-1) | (a) `srv_DN1.log`의 `#new-token` 로그 줄 문자 매치 ↔ (b) "배치가 시도됐다"(임의 채널) | ★**뒤집힘** — 같은 아티팩트가 (a)에선 `UNREALIZED`(H1 무증거), (b)에선 `RECOVERED-STRICT` | 907959 **TD1·TD2 최대 `#new-token` = 6245**, 밴드 [10036,10137] 안 배치 **0개**(2/2). 치명 배치 = **10125**(`DIAGNOSIS:29,:31`). 기전: `report_prefill_stats`는 forward **완료 후** 호출(`sglang/srt/managers/scheduler_output_processor_mixin.py:328`; 핀된 `DIAGNOSIS:21`이 같은 말을 적어 뒀다) |
| **V 밴드의 상한**(§4-1) | 양측 [10036,10137] ↔ 편측 ≥10036 | ★**뒤집힘** — 건강한 boot이 양측형에선 `UNREALIZED`, 편측형에선 `NOT-RECOVERED` | 생존 6 boot 중 **2건이 양측 밴드 밖**: 908179 **L1 = 10630**, 907959 **L2 = 12516** |
| **SCOPE (12)의 문자열**(§4-3) | 등록 문자열 `guard_applied=yes`를 문자 그대로 ↔ 의미로 | ★**뒤집힘** — 문자 그대로면 `NO_VERDICT_SCOPE`가 **항상 참** ⇒ 회차 전체가 실행 전부터 무산 | 하네스 블록을 직접 실행: `r2c_guard=none guard_applied=`**`DN-boots-only`**` dn_boots=1 order=[L TD L TD DN]`. `r2_correctness.sbatch:273/276/278`이 만들 수 있는 값은 `DN-boots-only` / `no (R2C_GUARD='…' set but ORDER has no DN boot)` / `no` **3개뿐**이고 `yes`는 없다 |
| **F-n3의 `FAIL` 귀속 정의역**(§5-3) | 임의의 `FAIL` ↔ DN 귀속이 입증된 `FAIL` | ★**뒤집힘** — 규칙을 job 907959에 먹이면 `DESIGN-REFUTED`("진단 boot이 채점을 오염시켰다")가 나온다. 그 job엔 **DN boot이 0개**다 | 907959 `verdict.txt` = `FAIL`, 채점 boot 4, DN boot 0. 선행 4 job 중 **2건**(907032 TD 2/2 warm-up 크래시, 907959)이 DN 없이 이 분지를 밟는다 |
| **§4-4(2) "681/681 OK"의 검사 범위** | 전체 스위트 ↔ 이 트랙(NPC-H가 등록한 범위) | ★**뒤집힘** — 전체형이면 제출 금지, 트랙형이면 제출 가능 | 실측 **`Ran 681 tests in 680.827s / FAILED (failures=1)`**, 실패 = `test_lambda0_prereg.TestMutationHarness.test_no_escapes`(단독 100.9 s 결정적 재현). 이 트랙은 **98/98 OK**(dnone 36 + instrument/ctx/scope_guard 62) |
| F-n2 문턱 `0.3×R(T)`(§5-2) | 0.1×(1.26 GiB) ↔ 1.0×(12.61 GiB) | **뒤집히지 않음** — §5-2가 "1차 정보는 부호뿐"으로 크기를 판정에서 뺐다 | 907959 OOM 시 `allocated` **77.25 GiB** vs 908179 TD1 **epoch 40 peak 68.305 GiB** ⇒ Δ ≈ **8.95 GiB**(0.10×–0.70× 통과, ≥0.71× 탈락) |
| §5-1 서명 ② 허용오차("77.25 GiB **근방**") | ±0.01 GiB ↔ ±2 GiB | **뒤집히지 않음** — STRICT↔WEAK만 갈리며 두 등록 문장은 같은 방향이고 등급 변경 권한이 이 문서에 없다 | 907959 TD1·TD2 `Of the allocated memory 77.25 GiB` **2/2 동일** |
| SCOPE (13) `dn_boots`(§4-3) | 개수로 읽기 ↔ 불리언으로 읽기 | **뒤집히지 않음** — `order=[L TD L TD DN]` 술어가 동시 강제 | 하네스는 `dn_boots=$HAS_DN` ∈ {0,1}; `L TD DN DN`도 `dn_boots=1`이 된다 |
| E5 정규식의 실행 엔진(§8) | GNU `grep -E` ↔ `grep -P` ↔ `python re` | **뒤집히지 않음** | 재실행: VERIFIED_B3 **12/12 · 12/12 · 12/12**, Y6만 발화, **Y4 미발화**(진짜 H1 증거 미가로채) |
| G 게이트 양성/음성(§4-2) | `none` ↔ `inference_mode` | **뒤집히지 않음**(양 분지 도달) | `nullcontext` 직접 실행 → `(is_grad_enabled=True, is_inference_mode_enabled=False)` = 등록 술어와 동형. 908179 TD1/TD2 `*_worker_grad_enabled=true` **0건** / `false` **8569·8263건**, `null` 539·525건 ⇒ 음성대조 실측 확인 |
| 승계 98건의 "강제"(§8) | 열거 없는 계수 대조 ↔ 명시 편입 | **뒤집히지 않음**(판정량이 아니다) — 단 DNR-4가 그 무강제의 실물 결과다 | `citation_stops.tsv` 16 rules 전부 구 트랙 수치 패턴, **98건 중 0건 등재** |

반전 계산에 쓴 원자료: `job_907959/{srv_TD1.log,srv_TD2.log,srv_L1.log,srv_L2.log,tel_TD1.jsonl,tel_TD2.jsonl}` ·
`job_908179/{srv_L1.log,srv_L2.log,srv_TD1.log,srv_TD2.log,tel_TD1.jsonl,tel_TD2.jsonl,provenance.txt}` ·
`r2_correctness.sbatch`(ab55c07c) 블록 직접 실행 · `sacct` · 전체 `unittest discover`.
밴드·역산식·배치 사다리는 **핀된 `DIAGNOSIS_true_dual_oom.md`(`8c857bed…`) `:21,:29,:31,:128`에서
전사**했다 — 새 자유 표면을 만들지 않았다.

---

## §2. 死因

### DNR-1 (지배적) — V 게이트가 처치 강도의 **감소함수**다 ⇒ `RECOVERED-*`가 **도달 불가** (N3 + N2)

**문장/위치**: §4-1 — *"`srv_DN1.log`에서 `#new-token ∈ [10036, 10137]` split-prefill 배치가
**형성**됐는가. 형성 안 됨 ⇒ **`UNREALIZED`**: 이 회차는 H1에 대한 증거를 **만들지 못했다**"* +
§5 *"`V ∧ G ∧ SCOPE`가 전부 참일 때에만 아래를 채점한다"*.

**왜 死因인가**: `report_prefill_stats`는 `scheduler_output_processor_mixin.py:328`에서 **forward가
끝난 뒤** 호출된다 — 죽은 배치는 `#new-token` 줄을 남기지 못한다. 이 사실은 이 사전등록이
sha로 핀한 `DIAGNOSIS:21`에 이미 문자로 적혀 있다. 실측:

| boot | 밴드 [10036,10137] 안 배치 | V | 실제 사건 |
|---|---|---|---|
| 907959 TD1 | **없음**(최대 6245) | **FALSE** | ★H1 현상 발생(198.00 MiB / 77.25 GiB) |
| 907959 TD2 | **없음**(최대 6245) | **FALSE** | ★H1 현상 발생(동일 서명) |
| 907959 L1 | 10125 | TRUE | 완주 |
| 907959 L2 | **없음**(12516) | **FALSE** | 완주 |
| 908179 L1 | **없음**(10630) | **FALSE** | 완주 |
| 908179 L2 / TD1 / TD2 | 10125 | TRUE | 완주 |

⇒ `P(V=TRUE | OOM 재현) = 0/2`이며 **기전상 0**. `P(V=TRUE | 건강) = 4/6`. H1이 참인 세계에서
이 회차는 `UNREALIZED`를 내고 "증거를 만들지 못했다"고 선언하며, 그때 §4-1이 명령하는 보고
의무("형성된 최대 토큰 수와 전 배치열")가 산출하는 값은 **6245와
`55→368→1468→3241→6245`** — 즉 **바로 그 확증 증거를 수집해 놓고 라벨로 폐기한다.**
반대로 H1이 거짓이면 V=TRUE(67%)이고 `NOT-RECOVERED`가 나온다. **따라서 §0이 선언한 유일한
목적(`PLAUSIBLE(조건부)`→`CONFIRMED(scoped)`)에 도달하는 라벨 2개(`RECOVERED-STRICT`/`-WEAK`)는
이 설계에서 도달 불가**이고, §6("하지 못하는 것")은 그 사실을 등재하지 않는다 — RA3-1이 요구한
검정력 공시의 정확한 재발이다.

보강: 밴드 상한이 건강한 boot도 2/6에서 탈락시킨다(10630·12516). 그리고 `UNREALIZED` 한 라벨이
(i) boot 실패, (ii) 위험 배치 미도래, (iii) **OOM 재현**을 구분 없이 흡수한다. 907032의 TD 2/2가
서버 자체 warm-up 요청에서 죽은 전례가 있으므로 (i)도 살아 있는 채널이다(그 경우 G가 `NO_KNOB`을
내며 "H1의 반증이 아니다"라고 적는데, 실제로는 OOM일 수 있다).

**최소 수리 (실현가능성 검증됨 — 아래 수치가 그 검증이다)**: V를 OOM 유무로 **두 분지**로 쪼갠다.
- **분지 A (`srv_DN1.log`에 `torch.OutOfMemoryError` 0줄)**: 로그에 `#new-token ≥ 10036`인
  split-prefill 배치가 1개 이상 **완주**했는가. (상한 제거. 근거: 10630·12516은 10125보다 **더
  무거운** 배치이므로 반공허성 목적을 더 강하게 만족한다. 실측 도달: 908179 L1 10630 통과 /
  907959 L2 12516 통과 / 908179 TD1·TD2·L2 10125 통과 / 최대 6245인 boot 탈락 ⇒ **양 분지 도달**.)
- **분지 B (`torch.OutOfMemoryError` 존재)**: (i) `srv_DN1.log`에 사다리
  `55 → 368 → 1468 → 3241 → 6245`가 그 순서로 존재하고(`DIAGNOSIS:128`에서 전사; 907959 TD1·TD2 및
  908179 TD1·TD2 **4/4 바이트 동일** 실측), (ii) OOM 메시지의 `Tried to allocate <S> MiB`를
  `T = S·1048576 / 20480`으로 역산한 `T`가 [10036, 10137]에 드는가(`DIAGNOSIS:29,:31` 전사식).
  실측: 198.00 MiB → **T = 10137.6** ⇒ 밴드 안(2/2). 6245급 배치라면 122.0 MiB → T ≈ 6245 ⇒
  밴드 밖 ⇒ **`OTHER-BATCH`로 라우팅**. (i)이 거짓 ⇒ `UNREALIZED`.
- §5-1 표에 라우팅을 명시: 분지 B ∧ (ii) 참 ⇒ `RECOVERED-STRICT|WEAK`; 분지 B ∧ (ii) 거짓 ⇒
  `OTHER-BATCH`; 분지 A ∧ V 참 ⇒ `NOT-RECOVERED`; 그 밖 ⇒ `UNREALIZED`.
- **추가 필수 등재**: "수리 후에도 F-n2(피크 축)는 OOM 분지에서 `UNMATCHED`가 될 수 있다 —
  907959 TD1·TD2 **2/2에서 치명 배치의 `prefill_active_batch_size=14` 스냅샷이 0건**(최대 관측 8)
  이었다. 기전은 미확정이며(로그 채널은 확정, 텔레메트리 채널은 미확정) 따라서 **F-n2는
  `RECOVERED` 분지의 1차 채널이 아니다**."

### DNR-2 — SCOPE (12)의 등록 문자열이 하네스가 **낼 수 없는 값**이다 ⇒ 회차 전체가 실행 전 무산 (N2)

**문장/위치**: §4-3 — *"**(12)** `provenance.txt`가 `r2c_guard=none guard_applied=yes`"*.

**왜 死因인가**: `r2_correctness.sbatch:272-279`이 `GUARD_APPLIED`에 대입할 수 있는 값은
`DN-boots-only`·`no (…)`·`no` 세 개다. 블록을 그대로 실행해 얻은 실제 줄은

```
r2c_guard=none guard_applied=DN-boots-only dn_boots=1 order=[L TD L TD DN]
```

문자 그대로 채점하면 (12)은 **항상 거짓**이고 §4-3이 `NO_VERDICT_SCOPE`를 강제하며 §5가
"H1 등급 불변"을 확정한다 — 0.197 GPU-h를 쓰고 등록 규칙상 아무것도 못 쓴다. 의미로 읽으면
통과한다. **같은 아티팩트에서 정반대 판정.** 이것은 "규칙을 산문으로 고정하면 구멍이
난다"(교훈 66)의 교과서적 재발이고, 908020 재발 방지를 목적으로 만든 게이트 자신이 실패한 형태다.

**최소 수리**: (12)을 `r2c_guard=none guard_applied=DN-boots-only`로 교체(위 실행 출력 및
`r2_correctness.sbatch:273`에서 전사). 아울러 `worker_grad_guard=none`(같은 실행의 `:350` 줄)
병기를 추가하면 목표값·적용범위 두 채널이 모두 고정된다.

### DNR-3 — F-n3는 정보 방향에서 항등식이고 반대 방향에서 **허위 귀속 생성기**다 (N1 + N3)

**문장/위치**: §5-3 — *"`FAIL`이면 **진단 boot이 채점을 오염시킨 것**이므로 **설계 C 자체가
반증**되고(`DESIGN-REFUTED`) 이 회차는 H1에 무증거다."* 이 문서 자신의 §1-4가 *"마지막 배치는
죽은 서버 잔재가 채점 boot에 닿을 가능성을 **구조적으로** 0으로 만든다"*고 적는다.

**왜 死因인가**: 두 문장이 충돌한다. 오염 가능성이 구조적으로 0이면 F-n3의 거짓 분지는 **그
기전으로는 도달 불가**(N3)이고, 참 분지("PASS ⇒ 설계 C 검증")는 DN이 마지막·비채점·`boots.txt`
부재라는 이미 CPU 98 테스트로 고정된 사실의 **연역**(N1)이다. 그러면서 `FAIL`은 **다른** 원인으로
살아 있게 도달하며, 등록 규칙은 그것을 무조건 DN에 귀속한다. 실증: job 907959는
`verdict.txt = FAIL`·채점 boot 4개·**DN boot 0개**인데 이 규칙을 먹이면 `DESIGN-REFUTED`가 나온다.
선행 4 job 중 2건이 DN 없이 이 분지를 밟는다. 교훈 21("측정 실패를 게이트 실패로 라벨링
마라")의 부호 반대 재발이다.

**최소 수리**: F-n3를 예보에서 **기록 항목**으로 강등하고 두 줄을 등록한다 — (a) "채점 4 boot의
라벨은 DN boot과 인과적으로 독립이다(DN은 마지막·비채점·`boots.txt` 부재, CPU 98 테스트가 고정).
따라서 `PASS`는 설계 C의 확증이 아니라 그 구조의 재확인이다." (b) "`FAIL`이면
`r2_correctness_report.json`의 실패 술어를 전사해 원인을 보고한다. **DN 귀속은 구체적 기전을
제시하지 않는 한 쓰지 않으며 `DESIGN-REFUTED` 라벨은 사용하지 않는다.**"

### DNR-4 — §4-4(2)의 제출 선행조건이 오늘 **사실로 거짓**이고 승계 항목 NPC-H와 **충돌**한다 (N2, 제출 GO/NO-GO 분지)

**문장/위치**: §4-4(2) — *"`unittest discover` **681/681 OK**"*.

**왜 死因인가**: 실측은 **`Ran 681 tests in 680.827s` / `FAILED (failures=1)`** —
`test_lambda0_prereg.TestMutationHarness.test_no_escapes`(단독 재실행 100.9 s, 결정적, 내 세션이
유발하지 않음: `LAMBDA0_*` 미설정; 원인은 λ0 생산자 자기검사가 temp-dir 사본에서
`/r2_correctness/job_907959/srv_warmup.log`로 경로를 풀어 버리는 타 트랙 결함). 문자 그대로면
**제출 금지**. 동시에 §8이 승계한다고 선언한 **NPC-H**가 *"제출 전 '전체 CPU 회귀'의 합격 기준은
**이 트랙(`test_r2_correctness_*`)으로 한정**한다"*를 이미 등록해 뒀고, 그 기준으로는 **98/98 OK**로
통과한다. 즉 §8의 "항목 수 대조 승계"가 **이 문서 자신의 §4-4에서** NPC-H를 위반했다 —
RA4-6이 지목한 누수의 실물 재발이며, 같은 자유 표면이 제출 GO/NO-GO를 양방향으로 뒤집는다.

**최소 수리**: §4-4(2)를 *"`python -m unittest discover -s workspace/engine-port/tests` 실행 후
**이 트랙(`test_r2_correctness_*` = dnone 36 + instrument/ctx/scope_guard 62 = 98)** 전부 `OK`
(NPC-H 승계). ★전체 스위트는 현재 **681 중 1 실패**(`test_lambda0_prereg.TestMutationHarness.test_no_escapes`,
λ0 트랙 경로 결함, 이 회차와 인과 무관)이며 결과 문서는 이 수치를 병기한다."* 로 교체.

---

## §3. 비차단 권고 (DNA)

- **DNA-1 (제출 명령 미등록 — 우선순위 높음)**: 문서에 `sbatch` 명령이 없다. 하네스가 제공하는
  유일한 D-none 예시(`r2_correctness.sbatch:165`)는 `R2C_ORDER="L TD L TD DN" R2C_GUARD=none sbatch …`로
  **`R2C_MODEL`/`R2C_ATTN_BACKEND`/`R2C_CTX`/`R2C_EXPECT_*`가 없다** ⇒ 기본값
  `Zyphra/Zamba2-2.7B`(`:198`)·`triton`(`:212`)로 908020 기판을 결정적으로 만든다. scope guard가
  fail-closed로 막으므로 판정은 오염되지 않지만 `provenance.txt`는 이미 쓰인 뒤이며(`:369` 다음이
  guard) 재제출 예산은 0으로 등록돼 있다. §4-4에 **리터럴 제출 명령 전문**(4개 `R2C_*` + 3개
  `R2C_EXPECT_*` 포함)을 등록하라.
- **DNA-2**: `DN-13`은 **존재하지 않는 id**다(§9는 DN-1…DN-10). §0-c의 앵커 부패 항목이 등록부에
  들어가지 못했다 — `DN-11`로 번호를 바꿔 §9에 추가하라.
- **DNA-3**: §4-4(2)가 부르는 `check_citation_stops.py`의 레지스트리(`citation_stops.tsv`)는 16 rule
  전부 구 트랙 **수치 패턴**이고 승계 98건 중 **0건**을 담고 있지 않다. "0 위반"을 승계 집행의
  증거로 인용하지 말 것.
- **DNA-4**: `presubmit.py`는 read-only가 아니다(게이트 #240). §4-4에 "이 도구 실행은
  `m4r_confinement/reachability_verdict.json`·`tc1_model_attrib/reach_verdict_rev3_A.json`을
  재작성한다. 실행 후 `git diff HEAD --` 두 경로가 빈 출력임을 확인해 기록한다"를 추가하라.
  감사자 실행 4→5회차, 이번에도 바이트 동일.
- **DNA-5**: 예산 모형 `153 + 111·n`은 **1점(908179) 2모수 분해**다 — 검증이 아니라 적합.
  "1초 오차로 재현"을 예측력 근거로 인용하지 말 것(RA3-11 계열).
- **DNA-6**: `dn_boots`는 개수가 아니라 불리언(`$HAS_DN`)이다. §4-3(13)은 `order=` 술어와
  **함께만** 유효하다는 것을 문자로 적으라.
- **DNA-7**: E5 계열의 이 회차 정의역은 **DN이 아니라 채점 TD boot**이다 — `none`=`nullcontext`는
  inference tensor를 만들지 않으므로 DN arm에서 E5는 구조적으로 발화 불가. §8의 처분("`no_grad`로의
  스코프 변경 사유")은 그러면 **채점 arm의 가드를 바꾸는 두 번째 축 이동**이 된다. 또한 `\b`가
  POSIX ERE 밖이라는 `E5_FAMILY…§6` 한계와 Y6 처분 미등록(같은 문서 §7-5)이 §8에 승계되지 않았다.
- **DNA-8**: §0-b의 *"prefill 41/41 `cuda graph: False`, 양 arm"* — TD1은 41/41이지만 L1은
  **40/40**이다(수치 병기 오류, 실질은 참).
- **DNA-9**: §0-c의 glob 인용 `:365-369` → 루프는 `:364-368`, glob 본체는 `:367`.
- **DNA-10**: `verdict.txt`가 아예 만들어지지 않는 분지(벽시계 초과 kill)와 "치명 배치 형성 +
  서명 0/3" 분지(§5-1 표에 행이 없다: 3/3·1–2·OOM 0건·다른 배치만 있다)가 출력공간에 없다.
  전자는 여유가 충분하므로(1,997 s vs 9,000 s) 문장 하나로, 후자는 §5-1에 "같은 배치·서명 0/3 ⇒
  `RECOVERED-WEAK`(일치 0으로 명시)" 행 추가로 닫힌다.

---

## §4. 반증 실패 항목 (감사자가 깨려고 했고 못 깬 것 — 등록 caveat로 전환)

1. **설계 C의 불가시성**: 채점기에 glob/listdir/walk 0건, `diag_boots.txt`·`BOOT_FAILURES.txt`·boot
   배너 소비자 0건, `x1` LABELS 하드코딩, `srv_*.log` glob은 `INSTR=1` 전용·`instrument/` 전용.
   **반증 실패.** ⇒ §1-3의 "예측된 `FAIL`은 정보"라는 문자 등록 면제 주장은 **성립한다**.
2. **argv 바이트 동일성의 비항등식성**: 기대값이 손으로 쓴 상수가 아니라 `83d8cb9d` blob(sha =
   908179 provenance 값)에서 추출·실행 비교. 변이 5앵커 유일성 단정 포함. 변이 (a)(b) 내가 재현:
   7·4 실패, 이름·단정문 문자 일치. **반증 실패.** ⇒ caveat: **CPU argv 동일성은 GPU 거동 동일성을
   함의하지 않으며, 문서는 그것을 주장하지 않는다(§2·§6-2가 이미 스스로 한정한다).**
3. **G 게이트**: `nullcontext` → `(grad=True, inference=False)` 직접 실행 확인, 음성대조 908179 TD
   `true` 0건/`false` 8569·8263건, `_r2_memory_fields` 9키가 `PDMUX_MEM_TELEMETRY`(하네스 기본 1,
   DN에도 `:635`로 전달) 하에 **전 스냅샷**에 실림(TD1 9108/9108). **반증 실패 — 항등식 아님.**
4. **E5 대체 정규식**: 3엔진 12/12 재현, `torch.OutOfMemoryError` 미가로채 확인. **반증 실패.**
5. **장부·예산 산술**: 게이트 #7(합산 대 `max()`)·드리프트 −0.00361 두 전례에 비추어 전 항목
   재계산 — **불일치 0건.**
6. **승계 98건 계수**: 직접 계수 98 = 문서 주장 98. **반증 실패**(집행 방식은 DNA-3/DNR-4 참조).

---

## §5. rev2가 통과하려면 정확히 필요한 것

1. **DNR-1 수리**: V를 OOM 유무 두 분지로 재정의(위 전문), 밴드 상한 제거, `UNREALIZED`의 3중
   의미 분리, **검정력 공시 한 줄 신설** — *"수리 前 V로는 `RECOVERED-*`가 도달 불가였고, 수리 後
   이 회차의 최대 산출은 **n=1 스코프 한정 존재 문장 하나**다: '(0-b 튜플에서) 가드 축 단독
   이동이 907959의 OOM을 그 서명까지 재현한다.' 기전·하네스 축·boot 간 분산은 닫지 않는다."*
2. **DNR-2 수리**: (12) → `r2c_guard=none guard_applied=DN-boots-only` (+ `worker_grad_guard=none`).
3. **DNR-3 수리**: F-n3 강등 + `DESIGN-REFUTED` 라벨 폐기 + `FAIL` 원인 전사 규칙.
4. **DNR-4 수리**: §4-4(2)를 NPC-H 범위로 한정 + 전체 스위트 실측과 실패 테스트명 등재.
5. **DNA-1**: 리터럴 제출 명령 등록.
6. 제출은 그 위에 §4-4(1)의 **범위 한정 OVERRIDE 문서 + 사용자 명시 승인**을 여전히 선행조건으로
   한다. 전례 두 건의 형식 요건(감사자가 원문 확인): ① 넘기는 BLOCK 2건을 **출력 그대로 인용**,
   ② "타 트랙 소관·이 회차와 인과 무관"의 근거, ③ **하지 않는 것 전수**(두 차단 해소·완화 금지 /
   레지스트리에서 spec 제거 금지(append-only, 커밋 `d11243a` 전례) / 선행 `NO-GO` 무르기 금지 /
   감사 없는 정책 판정 금지), ④ **범위와 GPU 상한 명시**(이 회차 = 1 job ≈ 0.197 GPU-h),
   ⑤ 사용자 명시 지시 인용. `OVERRIDE_VPROBE_SUBMIT_2026-09-01.md:10-11,40,42`·
   `OVERRIDE_P1_SUBMIT_2026-08-28.md:44,81`. **OVERRIDE 문서는 감사자가 쓰지 않는다.**

---

## §6. 인용 금지 / 필수 병기 — 결과 문서·정본이 문자 그대로 승계할 문장 (신규)

> **DNR-V1** (인용금지) — *"job 908179 후속 D-none 회차의 사전등록 rev1은
> `RECOVERED-STRICT`/`RECOVERED-WEAK`를 낼 수 있었다"* 는 **거짓**이다. 그 두 라벨은 rev1의 V
> 게이트에서 **도달 불가**였다: `report_prefill_stats`가 forward 완료 후 호출되므로
> (`scheduler_output_processor_mixin.py:328`) 죽은 배치는 `#new-token` 줄을 남기지 못하고,
> 907959 TD1·TD2 **2/2에서 최대 `#new-token`은 6245**이며 밴드 [10036,10137] 안 배치는 **0개**였다.

> **DNR-V2** (인용금지) — *"이 사전등록의 SCOPE 게이트는 등록 밖 실행을 잡는다"* 를 rev1에 대해
> 쓸 수 없다. rev1의 술어 (12)가 요구한 문자열 `guard_applied=yes`는 하네스가 **만들 수 없는
> 값**이고(실제 출력 `guard_applied=DN-boots-only`, `r2_correctness.sbatch:273`), 문자 그대로
> 채점하면 `NO_VERDICT_SCOPE`가 **항상** 참이다.

> **DNR-V3** (인용금지) — *"F-n3가 설계 C를 검증했다/반증했다"* 는 어느 방향으로도 쓸 수 없다.
> DN boot은 마지막·비채점·`boots.txt` 부재이므로 참 분지는 연역이고, `FAIL`은 DN과 무관한
> 원인으로 도달한다 — job **907959**(`verdict.txt=FAIL`, DN boot **0개**)에 rev1의 규칙을 먹이면
> `DESIGN-REFUTED`가 나온다.

> **DNR-V4** (필수병기) — 이 트랙의 어떤 제출 서술에도: *"제출 시점 전체 CPU 스위트는 **681 중
> 1 실패**였다(`test_lambda0_prereg.TestMutationHarness.test_no_escapes`, λ0 트랙 경로 결함,
> 이 회차와 인과 무관). 합격 판정은 NPC-H가 등록한 **이 트랙 98/98**을 기준으로 한 것이다."*

> **DNR-V5** (필수병기) — F-n2(피크 축)를 인용할 때: *"907959 TD1·TD2 **2/2에서 치명 배치의
> `prefill_active_batch_size=14` 스냅샷은 0건**(관측 최대 8)이었다. 따라서 OOM 분지에서 피크 축은
> `UNMATCHED`가 될 수 있고, 그 회차의 1차 채널은 피크가 아니라 OOM 원문 서명(요청 크기·
> `allocated`·스택)이다."*

> **DNR-V6** (필수병기) — 승계를 서술할 때: *"이 사전등록은 98건을 **항목 수 대조로만** 승계했고
> 열거하지 않았다. 그 결과 rev1의 §4-4(2)가 승계 항목 **NPC-H를 스스로 위반**했다(전체 스위트
> 681/681 요구). `check_citation_stops.py`의 16 rule에는 98건 중 **0건**이 등재돼 있으므로
> '0 위반'은 승계 집행의 증거가 아니다."*

> **DNR-V7** (인용금지) — *"이 회차가 `NOT-RECOVERED`를 냈으므로 grad-guard 수리가 불필요했다"* 는
> 쓸 수 없다(DN-8 승계). 추가로 rev1의 V로는 `UNREALIZED`와 `NOT-RECOVERED`가 구분되지 않았다.

> **DNR-V8** (필수병기) — *"이 회차는 DN n=1이며, 하네스 sha가 908179에서 1개 움직였고
> (`f39b167b…` → `ab55c07c…`), 이 트랙은 이 사전등록 이전에 3 job을 소비해 그중 1건(908020)은
> 등록 밖이었다."* (DN-4·DN-5·DN-7 승계 — rev1의 이 세 항목은 유효하다.)

> **DNR-V9** (인용금지) — *"`presubmit.py` 실행은 read-only다"* (게이트 #240). 이 판정의 실행
> 회차에서도 두 타 트랙 verdict 파일이 재작성됐다(바이트 동일 확인).

★ 확정 결론(HE0 · layer-type 死 · 정책 순위 · stake #1 · 게이트 #13/#16 · C2 인용정지 ·
Zamba2/triton 동결)은 **재검증하지 않았고 이 판정으로 아무것도 바뀌지 않는다.** 이 회차의
**성능 판정 0건** 불변식은 문서에서 깨지지 않는다 — 성능 지표 언급은 §0-a와 DN-10의 금지 문장
2곳뿐이고, 등록된 측정량(`#new-token`·OOM 서명·`gpu_mem_*`·가드 실현 부울)에 지연·처리량
파생물이 없다.

---

## 관련 파일 (절대 경로)

- 감사 대상: `.../results/r2_correctness/dnone_prereg/PREREG_DNONE_2026-09-14.md`
- 死因 근거 코드: `/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/managers/scheduler_output_processor_mixin.py:328`(DNR-1) ·
  `.../results/r2_correctness/r2_correctness.sbatch:272-279, :350-351`(DNR-2)
- 死因 근거 원자료: `.../job_907959/{srv_TD1.log,srv_TD2.log,srv_L2.log,tel_TD1.jsonl,tel_TD2.jsonl,verdict.txt}` ·
  `.../job_908179/{srv_L1.log,srv_TD1.log,tel_TD1.jsonl,provenance.txt}`
- 핀된 승계 정본: `.../audit_907959_2026-09-13/DIAGNOSIS_true_dual_oom.md:21,:29,:31,:128` ·
  `.../audit_908179_2026-09-14/VERDICT.md:315-319`(§8.5) · `.../CARRYFORWARD_INVENTORY_2026-09-14.md` §10 ·
  `.../E5_FAMILY_RA4_2_2026-09-14.md` §6
- 전례(감사자 원문 확인, 작성하지 않음): `.../results/cp_baseline/OVERRIDE_VPROBE_SUBMIT_2026-09-01.md` ·
  `.../OVERRIDE_P1_SUBMIT_2026-08-28.md`
- 실패 테스트: `workspace/engine-port/tests/test_lambda0_prereg.py:481`
