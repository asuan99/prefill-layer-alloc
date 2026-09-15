# 판정서 — **E2 하네스층 재감사 (수리본)** (2026-09-15)

> **하네스층 2회차**(규칙층은 rev1→rev3 3회차로 이미 `GO-with-caveats`). 직전 하네스층
> 판정서 `VERDICT_e2_harness_2026-09-15.md`(`0971cef7…`, 629행)가 `NO-GO`(死因 H-F1/H-F2/H-F3 +
> 차단 H-B1…H-B8)를 냈고, 그 §7 재제출 선행조건에 따른 수리본이 이 판정서의 대상이다.
> **GPU 지출: 여전히 0.** 이 감사도 GPU 0(아카이브 재분석 · 코드 변이 · 합성 라벨링 · bash 시뮬레이션).

## 감사 대상 (sha256 — 전부 의뢰서와 일치)

| 파일 | sha256 | 이전 | 확인 |
|---|---|---|---|
| `e2_realized_mix.py` | `ecb0f6fce76f82c3a3b816ebc542885c4b9be4ac247529b51efc6f5032ab0b78` | `f87e7194…` | ✓ |
| `e2_label.py` | `c5a6a881e4c411ea0c20a07f82eb1cebc2b5dabc8b15a116fd8c019bfa234bd3` | `701d4616…` | ✓ |
| `e2_sticky.sbatch` | `505e880d12dfa94d27d2b9f60bfde51a1c558b9644e42fe7350627ab5f9fcadc` | `4bccc0ec…` | ✓ |
| `VERDICT_e2_harness_2026-09-15.md` | `0971cef7ade95f7a47b2a84478eb21b1a67a5e9b87f4538ebc7ce1ef4b13426c` | — | ✓ |
| `PREREG_E2_STICKY_REV3_2026-09-15.md` | `8998d53277abf660bb42bffc388fc92aa05c8de61e991c241783d9c87e1b29a4` | — | ✓ |
| `PREREG_..._REV3_ADDENDUM_2026-09-15.md` | `56870ee5e10d853a20243563727f8aecc4a088cd7c3c3a847296f7eb1d51df70` | — | ✓ |

---

## 0. 판정

> # `GO-with-caveats`
>
> **死因 0.** 직전 판정서의 死因 3건과 차단 8건은 **전부 닫혔다** — 11/11을 **실행으로**
> 확인했다(§1, 합성이 아니라 `e2_label.py`·`e2_realized_mix.py` 자신과 실제 아카이브 로그·
> 실제 bash로). 직전 회차가 라벨을 실제로 뒤집었던 두 경로(H-F1 `UNRESOLVED: []`, H-F2
> `STICKY_REALIZES_A`+`LAMBDA_WITHIN_CI`)는 이제 **등록된 `UNRESOLVED`를 낸다**.
>
> **필수 선행조건 4건**(전부 GPU 0 · 코드/문서 국소 · **재감사 불요**) · **신규 caveat E2C-29 … E2C-35** ·
> **반증 실패 12건**(§6).
>
> ★**등급 근거의 정직한 공시**: 의뢰 기준은 *"死因은 등록과 다른 수를 내거나, 게이트가 발화
> 못 하거나, 라벨이 뒤집히는 것에만"* 이다. 나는 **라벨을 뒤집는 잔여 결함 1건**(§3-1, R1이
> 자기 도메인 축소를 공시하지 않아 `OFF_BASELINE_ABOVE_BAND` ↔ `STICKY_REALIZES_A`가 갈린다)을
> 실행으로 **재현했다**. 그것을 死因으로 올리지 않은 이유는 두 가지뿐이다 — **(a)** 그 분지는
> **R5 실패라는 비정상 경로에서만** 열리고 등록 예보(아카이브 seed 산포 **0.39pp** vs 문턱 5pp)는
> 그 분지에 가지 않으며, **(b)** 수리가 **4행**이고 판정 규칙을 한 글자도 바꾸지 않는다.
> **정상 경로였다면 死因이었다.** 그래서 이것을 **선행조건 #1로 못박는다**(문안 승계만으로도
> 닫히지만, 코드 4행이 더 싸다).

### 이것이 "찜찜하니 차단"이 아닌 이유 (등급 디플레 점검)

나는 이번 회차에 **19개 변이 + 무변이 대조군**을 돌렸고(§2-3, §4), 자유 표면 **11개**를 등록이
허용하는 끝까지 밀었다(§5 반전 시험 표). **판정을 실제로 뒤집은 것은 3건**이고 전부
**등록 침묵 영역이거나 비정상 경로**였다. 나머지 8개는 **밀었는데 안 뒤집혔다** — 특히 의뢰가
직접 의심한 **P8 오탐/미탐(216 run_id 전수 = 0/0)** 과 **CAPPED·크래시·키부재 3형태 전부 짝
UNRESOLVED** 는 깨려다 실패했다. §6에 반증 실패로 남긴다.

---

## 1. 1순위 — 死因 3건 · 차단 8건이 **실제로** 닫혔는가 (11/11 실행 확인)

| 항목 | 판정 | 실행 근거 (전부 이 감사가 직접 돌린 출력) |
|---|---|---|
| **H-F1** bench 부재 ⇒ UNRESOLVED | **PASS** | `e2_label.py:106-112` `if not bench or not bench.get("all_pass")`. 쉬핑 경로(`--cells-dir`)로 3형태 실행: **키 삭제** → `a_r4/251 ON: bench artefact absent (P7 CAPPED or crash)` · **rc 124** → `…(bench_rc=124; 124 = P7 CAPPED)` · **rc 1** → `…(bench_rc=1; …)`. 세 경우 모두 `R2.a_r4=UNRESOLVED`. **CAPPED와 크래시가 아티팩트에서 구별된다** |
| **H-F1** 생산자측 `bench` 항상 기록 | **PASS** | `e2_realized_mix.py:355-362` else 분지. 실행 3회: `bench_rc` **124 / 1 / None**(인자 누락 시)이 전부 `all_pass:false`로 기록됨. sbatch `:354`가 `--bench-rc "$BRC"`를 넘기고 `$BRC`는 `:345`에서 `timeout` 직후 캡처된다 ✓ |
| **H-F2** R5가 R1/R2 **앞**, 짝 전부 UNRESOLVED | **PASS** | `e2_label.py:146-166`(R1 `:168`, R2 `:194`보다 앞). OFF 산포 6.85pp → `R5.a_r4=SEED_SPREAD_FAILS` + **a_r4 4짝 전부 UNRESOLVED** + `R2.a_r4=UNRESOLVED`. **ON 산포 6.4pp에서도 동일**(양 arm 대칭) |
| **H-F2** 순환 없음 | **PASS** | R5는 post-R3(ii) `live` 집합의 `e_cnt`만 읽는다(`:154`). R1/R2/R3/R4 어느 것도 R5 이전에 계산되지 않음. **단 도메인이 자유 표면** → E2C-32 |
| **H-F3** P3 boot마다 발화 + `ABORT_P3` | **PASS** | `e2_sticky.sbatch:265-274`, `exit 8`, P13·warm-up **이전**. 실제 bash로 아카이브 4개 로그를 순차 투입: 4 boot 전부 `distinct=1, abort=no`; 다른 키 1개 주입 → `distinct=2, abort=YES` |
| **H-F3** 값 추출이 실제 로그와 맞는가 | **PASS** | **11/11 `srv_*.log` 전수**: `MMB=48 MTT=2722025`(전부 동일). `grep -o "max_mamba_cache_size: *[0-9]*"`는 `ServerArgs(… max_mamba_cache_size=None …)`(등호)를 **집지 않고** `:41`의 `max_mamba_cache_size: 48`(콜론)만 집는다 — 실측 확인 |
| **H-F3** 타임스탬프 미포함 | **PASS** | 키는 `"48 2722025"` 두 값뿐. `sort -u` 1줄 |
| **H-B1** 시계 가드 소비 | **PASS** | `e2_label.py:118-121`이 읽는다. 합성 실행 → `a_r2/4386 ON: t_probe_end outside span` UNRESOLVED. 생산자측도 양방향 실측: `probe_end=1e9`(상단 이탈) · `=50`(하단 이탈) 모두 `ABORT_PROBE_CLOCK_MISMATCH` 기록 |
| **H-B2** P8 = telemetry `run_id` 내용 비교 + 전용 채널 | **PASS** | `e2_realized_mix.py:363-372` + `e2_label.py:114-117`. 실제 아카이브 telemetry로 왕복 실측(§2-1) |
| **H-B3** `split_transition` 프로브 필터 | **PASS** | `e2_realized_mix.py:90-93`. 합성 telemetry(프로브창 5건 + 이후 3건): `probe_end=None` → **8**, `probe_end=105` → **3**. 스냅샷도 11→6 |
| **H-B4** sm_index_mismatch | **미결 → 선행조건 #2** | 코드는 짝 UNRESOLVED(실행 확인: `a_r4/4162 OFF: stream_index/sm_counts disagree`), 등록(rev3 §5)은 **ABORT**. §3-2에서 **어느 쪽이 옳은지 판정하고 리터럴 문안을 준다** |
| **H-B5** E-pact 밴드 런타임 조항 | **PASS** | `e2_label.py:122-136`. 실행: `94.0%` → `ABORT_TELEMETRY_INCONSISTENT … < 95.0` · `96.67%` → `in [95.0,100) -- identity guard degraded`. **양쪽 분지 도달** |
| **H-B6(a)** sticky unset 주석 | **PASS** | `:61-63`이 *"IS listed here as a defence against an inherited value"* 로 정정됨 — 사실과 일치 |
| **H-B6(b)** E-iter 출처 | **PASS** | 독립 대조: rev2 **:189-195**에 2자리 전부(`4.50/8.11/8.76/8.36/45.58/92.29/92.13`), rev3 **:594**에도 전부. docstring의 줄번호 `191-194`는 **표 범위 189-195 안의 부분 인용**(값은 정확) |
| **H-B6(c)** 프로브 token_ids | **PASS** | `output_token_logprobs`·`token_ids`·`return_logprob` **grep 0건**. `:310-312` 주석이 삭제 사유를 명기. A7의 전례 동일성 유지 |
| **H-B7** `errors` fail-closed | **PASS** | `e2_realized_mix.py:223-227`. 실측 5형태: **부재 → `P9=False, all_pass=False`** · `[]` → True · `[""]×400` → True · 실패 400건 → False · 정수 0 → True. `output_lens` 부재도 여전히 fail-closed(`P11b=False`) — **P9와 P11b가 이제 같은 방향** |
| **H-B8** `--archive` + exit 2 | **PASS** | 스크래치 복사본 무인자 → `SELFTEST_ARCHIVE_MISSING … (this is NOT an estimator failure)` **rc 2**; `--archive` 지정 → **rc 0, 7/7**. sbatch `:160`의 `\| tee \|\| exit 4`는 `pipefail`로 rc 2를 받아 **abort한다**(실측) |

### 1-1. 제출 차단 셀프테스트 3종 — 전부 rc 0 (venv 인터프리터)

| 검사 | 결과 | 시간(venv, 실측) |
|---|---|---|
| `e2_realized_mix.py --selftest` | `SELFTEST_OK 7/7 cells reproduce the registered table` | 5.4 – 22.3 s (Lustre 캐시 변동) |
| `e2_label.py --selftest` | `SELFTEST_OK 8 reachability checks + **5 repair-regression checks** (H-F1/H-F2/H-B1/H-B5)` | 2.2 – 45.3 s |
| `test_sticky_partition.py` | `Ran 12 tests … OK` | 31.8 s |

합 **≈ 40–100 s** ≪ 등록 preflight 상수 **400 s**. §7-3 이행 확인.

### 1-2. §9-5 10항 재확인 (직전 판정서가 "부분 PASS"로 남긴 2항 포함)

| # | 항목 | 이번 |
|---|---|---|
| 1 | E-time 열 = A1 표 | **PASS** — `:255-261`이 `31.76/667.98 … 285.99/293.23`. rev3 §4-2 역산값(`24.95/525.2` 등) grep **0건** |
| 2 | decode-busy 리터럴 | **PASS** — `:60-67` |
| 3 | P13 필터가 **Q3·Q4 + 다섯 추정량** 전부 | **PASS**(직전 "부분 PASS" → 해소, H-B3) |
| 4 | P14 = 10,800 − `SECONDS` | **PASS** — `:81,192` |
| 5 | R1 4분지 | **PASS** — `:179-188`, 셀프테스트 (4)가 두 단독 분지 도달 확인 |
| 6 | E-pact ≥95.0 밴드가 **런타임 판정 경로**에 | **PASS**(직전 "부분 PASS" → 해소, H-B5) |
| 7 | P13 프롬프트 6개 + `max_new_tokens=48` + `text` 리스트 판정 | **PASS** — `:288-304`, 판정 `:376-378` `a == b` |
| 8 | E2C-15…E2C-21 문자 승계 | **PASS** — `e2_label.py:294-313` 15건 |
| 9 | `PDMUX_STICKY_PARTITION` unset 주석 | **PASS**(H-B6(a) 해소) |
| 10 | 셀프테스트 3종 rc 0 | **PASS**(§1-1) |

### 1-3. 등록 통제요인 — 독립 재대조 (게이트 #110)

직전 판정서를 상수로 승계하지 않고 `lambda0.sbatch`·아카이브 원자료에서 재확인:
`launch_server` 인자 **14줄 전부 일치**(`--attention-backend flashinfer` · `--enable-pdmux` ·
`pdmux_homog5.yml` · `--disable-overlap-schedule --chunked-prefill-size -1 --disable-radix-cache` ·
`--mem-fraction-static 0.82 --max-running-requests 48 --context-length 16384 --random-seed 1`) ·
`PDMUX_R2_POLICY=fixed` · `PDMUX_R2_FIXED_DSM=44`.
**`T_cap` 리터럴 독립 재산출**: 908623 실측 duration `131.01 / 142.45 / 286.85 s` × 3 =
`393.0 / 427.4 / 860.5` ⇒ 등록 **393 / 428 / 861** ✓. 셀 사양(rate `8.00/3.00/1.15`,
`np 400/400/200`, `in 256/256/8192`, `out 512/512/64`)도 아카이브 bench 레코드와 전수 일치.
**P4 digest 수**: `PREREG_E2_STICKY*.md`(4) + `VERDICT_e2_rules*.md`(3) + prereg·addendum(2) +
결정경로·러너(3) = **12 ≥ 10** ✓.
**P14 예산 산술 재계산**: 3번째 `b_r3` boot 직전 경과 `400+2212.0+2304.0+2×393.4 = 5702.8 s`
⇒ `REMAIN = 5097.2 s ≫ 861+200 = 1061 s`. 최악 코너에서도 `REMAIN ≈ 2973 s`. ⇒ **R4″는 소멸하지 않는다.**

---

## 2. 2순위 — 수리가 **새로 만든** 표면

### 2-1. ★P8 강화의 부작용 — **거짓 UNRESOLVED 위험 없음** (의뢰 2순위-1, 반증 실패)

의뢰의 의심: *"`f\"_{arm}_\" in run_id`가 실제 조합에서 오탐/미탐을 낼 수 있는가."*

**등록 공간 전수 열거**(cell 3 × arm 2 × seed 4 × jobid 9형 = **216 run_id**):

| | 결과 |
|---|---|
| 미탐(옳은 arm인데 **거짓 UNRESOLVED**) | **0 / 216** |
| 오탐(arm이 뒤바뀌었는데 **통과**) | **0 / 216** |

구조적 이유(실측): 등록 리터럴 어디에도 `ON`/`OFF`가 부분문자열로 없다 —
`e2` · `a_r4` · `a_r2` · `b_r3` · `4386` · `4162` · `251` · `2630` · jobid(숫자/`local$$`).
`_OFF_`와 `_ON_`은 서로의 부분문자열도 아니다.

**엔진측 왕복도 실측**: 서버는 `multiplexing_mixin.py:226` `run_id=os.environ.get("PDMUX_RUN_ID","unlabeled")`를
`RuntimeEvent`의 **필수 필드**로 싣는다 ⇒ 908623 아카이브 **11/11 파일, 전 레코드 100%에 run_id**
(`tel_a_r4.jsonl` 21,215/21,215; 이벤트 4종 `phase_marker`/`runtime_snapshot`/`controller_decision`/
`split_transition` 전부). 아카이브 telemetry를 E2 형식 run_id로 재작성해 실행:
`--arm OFF` → **P8 불발 + `e_cnt 48/555` + `achieved 3.053130`** · `--arm ON` → **P8 발화**.

**드라이런이 아카이브 스탠드인을 전부 UNRESOLVED로 만든 것은 정상**이다 — λ0의 run_id는
`lam0_a_r4_908623`이라 **실제로 arm 라벨이 없다**. 게이트가 옳게 발화한 것이지 오탐이 아니다.

**여러 run_id / 결측 처리**(실측): `["…_OFF_…","…_ON_…"]` 혼재 → 발화 · `["unlabeled"]`(env 미전파)
→ 발화 · `[]`(telemetry 공백) → 발화. **전부 fail-closed.**
남는 한계는 **cell/seed 교차를 못 잡는다**는 것(`e2_a_r2_OFF_4162_…`를 `--arm OFF`로 보면 통과) → **E2C-30**.

### 2-2. H-B5 E-pact 밴드의 검정력 — **과발화하지 않는다** (단 den=0이면 가드가 꺼진다)

의뢰의 우려: *"shape A 분모가 24–35라 1표본 = 2.9–4.2%, `<100%`가 흔할 수 있다."*

**아카이브 7셀 E-pact 실측(내가 직접 산출)**:

| 셀 | E-pact | 분모 |
|---|---|---|
| `a_r0` | **100.000%** | 77 |
| `a_r2` | **100.000%** | **24** |
| `a_r4` | **100.000%** | **32** |
| `a_r4_s2` | **100.000%** | **35** |
| `b_r0` | **100.000%** | 135 |
| `b_r3` | **100.000%** | 400 |
| `b_r3_s2` | **100.000%** | 400 |

**0 이탈 / 1,103 표본.** 그리고 이것은 운이 아니라 **구성 항등식**이다 — 코드 직독:

- **OFF arm**: `multiplexing_mixin.py:1170-1171` — `not running_batch.is_empty() and split_prefill_batch`
  일 때만 분할 행이 설치되고, `PDMUX_R2_POLICY=fixed`의 v7 블록(`:1327` `_r2_decide_idx`)이
  고정 타깃(decode 44 SM = homog5 index 2)을 넣는다. ⇒ *prefill 진행 중 decode-busy* = idx2.
- **ON arm**: `:1172-1179` — `sticky_partition_enabled`가 논리합을 참으로 만들어
  `stream_idx = self._sticky_fixed_idx`(=2)가 **prefill 유무와 무관하게** 설치된다.
  ⇒ **ON에서 인덱스가 {0,2}로 닫히는 것이 아니라 decode-busy 전 구간이 idx2**다.
  전례 실측(job 872800 `stk1`, sticky ON): 자기 고정 인덱스 점유 **139/139 = 100.0%**.
  ⇒ **의뢰가 우려한 "ON에서 `prefill_active>0` 표본이 idx0일 수 있는가"는 아니다**(ON에서 idx0은
  `running_batch`가 비었을 때만 나오고 그때는 decode-busy가 아니다).

**캠페인 전체 UNRESOLVED 확률 추정**: 점추정 **≈ 0**(기전 결정론적 + 1,103/1,103 실측).
Rule-of-three 95% 상한은 표본당 `p ≤ 3/1103 = 0.272%`이지만 **이 상한은 무의미하다** — 실패
모드가 베르누이가 아니라 **구성별 이봉**이기 때문이다(반례: 872800 `stk0`은 다른 config
[Ha8 `(92,16)`]에서 prefill-active 4표본이 **전부 idx1**이라 idx2 기준 E-pact = **0/4 = 0%**.
그 config에서는 "D44 = index 2" 리터럴이 애초에 성립하지 않는다).
⇒ **완화 문안을 처방하지 않는다.** 등록 A6를 사후에 느슨히 하는 것 자체가 새 자유 표면이고,
코드는 A6를 **문자 그대로** 구현하고 있다. 대신 **무료 보험 1줄**(§3-4 선택 처방)과 **E2C-31**을 준다.

★**실제 위험은 반대편이다**: `e2_label.py:126` `if ep.get("den")` — **분모 0이면 가드 전체가
조용히 꺼진다**. shape A ON arm에서 prefill-active 표본이 0이 될 수 있는가? 아카이브 OFF에서
24–35이고 ON은 prefill 축이 같으므로 0이 될 이유는 없으나, **0일 때 "통과"와 "미측정"이 같은
출력**이다 → E2C-31(b).

### 2-3. R5를 앞으로 옮긴 부작용 — **이중 처벌 아님, 등록대로** (단 도메인은 자유 표면)

실행 확인: `SEED_SPREAD_FAILS` → 그 셀의 4짝이 `unresolved`에 들어가고 `R2.a_r4`는
`"paired n=0 < 4 (gate #3)"` 로 `UNRESOLVED`. **소비되는 결과는 하나(UNRESOLVED)뿐**이고
등록 rev3 §6 R5의 *"그 셀의 R1/R2 UNRESOLVED"* 와 일치한다. **이중 처벌 아님.**
다만 R2의 `why`가 R5를 가리키지 않고 `n<4`로만 적힌다 — `unresolved` 4줄이 R5를 명시하므로
정보 손실은 없다(인용 위생은 E2C-29).

★**그러나 R5의 도메인이 등록 침묵 영역이고, 그것이 R5 자신의 라벨을 뒤집는다** → §5 반전 표 1행, **E2C-32**.

### 2-4. repair-regression 검사 5건 — **독립 재현 + 되돌림 변이 확장** (의뢰 2순위-4)

**무변이 대조군을 반드시 포함했다**(직전 회차 §6-2의 자기 공시를 그대로 적용). 총 **19 변이 + 대조군 1**:

| 변이(되돌림 대상) | 결과 | 증거 수치 |
|---|---|---|
| **(대조) 무변이** | **SURVIVED** | 두 셀프테스트 rc 0 — *사살 수가 의미를 갖는 전제* |
| H-F1 label (`rec.get("bench") and …`) | **KILLED** | `H-F1 regression: absent bench artefact does NOT void the pair` |
| H-F2 전파 제거 | **KILLED** | `SEED_SPREAD_FAILS does not void R2` |
| H-F2 `live` 재계산 제거(=R5를 뒤로 되돌린 것과 동치) | **KILLED** | 동상 |
| H-B1 시계 가드 읽기 제거 | **KILLED** | `ABORT_PROBE_CLOCK_MISMATCH is fail-OPEN` |
| H-B5 밴드 전체 제거 | **KILLED** | `e_pact <95 does not void the pair` |
| H-B5 상단 `[95,100)`만 제거 | **KILLED** | `e_pact [95,100) does not void the pair` |
| **A2** `decode_running_batch_size`→`decode_batch_size` | **KILLED** | busy **555→20,082(36배)**, E-cnt **8.65→0.24** — 직전 판정서 B2″를 내가 **독립 재현** |
| **E-time** busy-부분수열 읽기 | **KILLED** | den `a_r4` **183.61→217.50** · `b_r3` **292.00→334.99** — ADDENDUM A1 표를 사살 |
| **E-iter** 왼쪽 busy 게이트 제거 | **KILLED** | `a_r4` **8.76→8.66** · `b_r3` **92.29→90.87** |
| **E-qcond** `prefill_queue_depth>0` 제거 | **KILLED** | `a_r4` **43/250→48/555** |
| **H-F1 생산자 else 분지 제거** | **SURVIVED** | 무해 — label측이 키 부재를 fail-close하므로 짝은 여전히 UNRESOLVED. **잃는 것은 CAPPED↔크래시 구별뿐** |
| **H-B2 label측 읽기 제거** | **SURVIVED** | 회귀망 없음 |
| **H-B2 생산자측 P8 제거** | **SURVIVED** | 회귀망 없음 |
| **H-B3 Q4 필터 제거** | **SURVIVED** | 아카이브에 프로브가 없어 셀프테스트가 못 본다 |
| **H-B7 `errors` fail-open 복원** | **SURVIVED** | **어느 셀프테스트도 `bench_validity`를 호출하지 않는다** |
| **H-B8 exit 2 → exit 1** | **SURVIVED** | 회귀망 없음 |

★**자기 공시(교훈 9)**: 내 1차 E-time 변이는 `j = i + 1`만 추가한 **등가 변이**였고 SURVIVED를
냈다. "생존 = 검사 구멍"으로 기록할 뻔했다. 앵커를 제대로 고쳐(`busy` 부분수열로 실제 변경)
다시 돌려서야 KILLED와 위 수치를 얻었다. **변이 자신이 항등식일 수 있다.**

**판정**: 살아남은 5건은 **수리가 없어서가 아니라 회귀망이 없어서** 살아남았다. 다섯 수리가
실제로 동작함은 §1 표에서 **기능적으로 전수 확인했다**(H-B3 8→3, H-B7 5형태, H-B8 rc 2, P8 왕복,
H-F1 생산자 3형태). ⇒ **死因 아님.** 회귀망 확장은 **선행조건 #3**(내가 직접 구현·검증했다, §3-3).

### 2-5. `bench` 키 상시 존재의 회귀 — **정상, 단 새 구멍 1개**

- `_achieved()`(`:64-66`): `all_pass=False`면 `None` — 그런 짝은 R5 이전에 이미 `live`에서
  빠지므로 shape A에서 이 `None` 경로는 **도달 불가**가 되었다(방어적 잔존). R2의 `n` 계산은
  `clean`에서 **n=4**로 정상(`LAMBDA_WITHIN_CI`), `delta=±0.5`에서 `MOVES_DOWN`/`MOVES_UP` 도달 ✓.
- ★**새 구멍**: `all_pass=True`인데 `achieved_req_s`가 **없는** 경우(`bench_validity`가
  `achieved_req_s`를 `if rec.get("duration")` 일 때만 넣는다). 실행 재현: `R2.a_r4=UNRESOLVED
  "paired n=2 < 4"` 인데 **`UNRESOLVED: []` (장부가 빈다)** — 이는 **H-F1이 닫은 바로 그 병리
  (R1과 R2가 다른 모집단 위에 선다)의 재발 형태**다. 도달성은 낮다(`completed=400`·입출력
  길이 정확·`rrr=1.0`을 전부 통과하면서 `duration`이 0/결측인 bench 레코드는 자기모순적).
  ⇒ **선행조건 #4**(4행), §3-4.

---

## 3. 필수 선행조건 (4건 — 전부 GPU 0 · 재감사 불요 · 리터럴 문안 제공)

### 3-1. **#1 (라벨)** R1이 자기 도메인 축소를 공시해야 한다

**재현(쉬핑 경로, 실행 출력)** — 같은 데이터에서 `a_r4` OFF seed 2630의 `e_cnt`만 8.65→15.5:

| | R1 라벨 | R1이 실제로 선 셀 | R5 |
|---|---|---|---|
| 산포 없음 (a_r4 OFF E-qcond 25.0%) | **`OFF_BASELINE_ABOVE_BAND`** | `['a_r2','a_r4']` | PASS |
| 산포 6.85pp (같은 25.0% 유지) | **`STICKY_REALIZES_A`** | **`['a_r2']`** | `SEED_SPREAD_FAILS` |

⇒ **`a_r4`의 OFF 밴드 위반이 R5 탈락에 가려 1차 라벨이 뒤집힌다.** 등록 R1은
*"`a_r4`·`a_r2` **전 seed**에서"* 이고 R5는 *"그 셀의 **R1**/R2 UNRESOLVED"* 인데,
**구현의 R1은 셀별 분해가 없어 "그 셀의 R1 UNRESOLVED"를 표현할 출력공간이 없다.**

**리터럴 수리**(`e2_label.py:189` 직전 — 내가 적용해 검증했다, §3-5):
```python
    seen = sorted({k.split("/")[0] for k in detail})
    dropped = [c for c in SHAPE_A_CELLS if c not in seen]
    if dropped and r1 == "STICKY_REALIZES_A":
        r1 = "STICKY_REALIZES_A_ON_REMAINING_CELLS"
    out["rules"]["R1"] = {"label": r1, "cells_evaluated": seen, "cells_dropped": dropped,
                          "on_clause_failures": on_fail,
```
(나머지 키는 그대로.) **검증**: `--selftest` 불변(`8 + 5`), 정상 경로 4시나리오 **라벨 무변화**,
위 두 행이 `STICKY_REALIZES_A_ON_REMAINING_CELLS` / `cells_dropped=['a_r4']`로 갈린다.
**코드를 안 고치겠다면 E2C-29의 인용 금지 문장을 결과 문서가 문자 그대로 승계해야 한다** —
둘 중 하나는 반드시.

### 3-2. **#2 (등록)** 追記 A11 — 등록 ABORT ↔ 구현 UNRESOLVED **3건**을 한 번에 못박아라

의뢰가 H-B4에 대해 물은 것(*"追記냐 코드 ABORT냐"*)에 답한다. **追記가 옳다.** 사유 3:

1. **눈금 일관성이 자유 표면을 줄인다.** 현재 계측 가드는 넷(`sm_index_mismatch` ·
   `e_pact<95` · `e_pact∈[95,100)` · `idx0>5%`)인데 등록은 앞의 둘만 ABORT, 뒤의 둘은
   짝 UNRESOLVED다. 근거 없는 두 눈금을 **하나로** 만드는 것이 규칙을 **늘리지 않고 줄인다**.
2. **교훈 21** — *"측정 실패를 게이트 실패로 라벨링 마라."* 셋 다 **계측 속성**이지 처치 속성이 아니다.
3. **ABORT 분지는 GPU 없이 시험할 수 없다.** 코드를 ABORT로 올리면 **한 번도 실행되지 않은
   분지를 캠페인에 태우는 것**이고, 그 분지가 오발화하면 1.8 GPU-h가 라벨 0으로 끝난다.
   반대로 UNRESOLVED 분지는 이번 회차가 **실행으로 전수 확인**했다.
   보강 근거(실측): 아카이브 11셀 `sm_index_mismatch = 0`, `idx2 ↔ (64,44)` 대응 위반 0.

**리터럴 追記 문안** (`PREREG_E2_STICKY_REV3_ADDENDUM_2026-09-15.md`에 **A11**로 추가 —
rev3 본문·기존 A1–A10은 건드리지 않는다):

> ## A11 — **계측 가드의 결과를 짝 `UNRESOLVED`로 통일하고, P8의 비교 대상을 정정한다**
>
> rev3 §5의 *"`(prefill_sms, decode_sms)` 불일치 시 **ABORT**"*, ADDENDUM A6의
> *"E-pact < 95.0% ⇒ `ABORT_TELEMETRY_INCONSISTENT`"*, rev3 §7 P8의 *"arm 라벨 무결성
> (파일명 ↔ JSON) 불일치 ⇒ **abort**"* 를 다음으로 **대체한다**:
>
> **(a)** `stream_index`와 `(prefill_sms, decode_sms)`의 불일치는 **그 seed 짝을 `UNRESOLVED`**
> 로 만든다(캠페인 abort 아님). 사유: 이것은 계측 속성이며 R3(ii)·A6 상단 밴드와 **같은 눈금**에
> 두는 것이 등록 자유도를 줄인다.
> **(b)** **E-pact < 95.0%도 그 seed 짝 `UNRESOLVED`** 로 한다. 라벨 문자열
> `ABORT_TELEMETRY_INCONSISTENT`는 `why`에 보존해 `[95,100)` 분지와 구별한다.
> **(c)** **P8의 비교 대상은 "파일명 ↔ JSON"이 아니라 "telemetry 레코드의 `run_id` ↔ `--arm`"**
> 이다. 파일명과 `--arm`은 러너의 같은 `$ARM`에서 나오므로 그 비교는 **구성상 항등식**이며
> 불일치가 발생할 수 없다(교훈 9). 불일치 시 **그 seed 짝 `UNRESOLVED`**.
> **(d)** (a)(b)(c)는 **모두 관대한 방향**이다. 따라서 어떤 결과 문서도 *"이 캠페인에는
> 계측 ABORT 게이트가 있다"* 고 쓸 수 없다 — **있는 것은 짝 단위 실패닫힘뿐**이다.
> **(e)** **R5의 정의역은 R3(ii)·유효성 장부를 통과한 `live` 짝의 seed 집합**이며, 등록 seed
> 전체가 아니다(하네스층 판정서 rev2 §5 반전 표 1행: 같은 6.85pp 산포가 정의역에 따라
> `SEED_SPREAD_FAILS`와 `PASS(0.00pp)`로 갈린다).

### 3-3. **#3 (회귀망)** 살아남은 수리 3건에 되돌림 검사를 붙여라 (교훈 53)

내가 구현하고 **되돌림 변이로 사살까지 확인**했다(대조군 통과). `e2_label.py`의
`if failures:` 직전에 삽입:

```python
    # (13) H-B2: the P8 arm-label guard must be READ, not merely recorded.
    cells = build()
    cells[("a_r2", "ON", "251")]["ABORT_ARM_LABEL_MISMATCH"] = "P8: run_ids [x] do not carry arm ON"
    if not any(u["cell"] == "a_r2" and u["seed"] == "251" for u in label(cells, pairs)["unresolved"]):
        failures.append("H-B2 regression: ABORT_ARM_LABEL_MISMATCH is fail-OPEN")
    # (14) H-B3 (A3 covers Q4) + H-B7 (`errors` fail-closed), exercised in e2_realized_mix.
    import tempfile, json as _j, importlib.util as _iu
    _mix = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2_realized_mix.py")
    _s = _iu.spec_from_file_location("_mix", _mix); _m = _iu.module_from_spec(_s); _s.loader.exec_module(_m)
    with tempfile.TemporaryDirectory() as _d:
        _p = os.path.join(_d, "t.jsonl")
        with open(_p, "w") as _f:
            for _t in (1.0, 2.0):                      # inside the probe window
                _f.write(_j.dumps({"event": "split_transition", "timestamp_monotonic_s": _t,
                                   "run_id": "e2_a_r4_ON_4386_1"}) + "\n")
            for _t in (4.0, 5.0, 6.0):                 # after it
                _f.write(_j.dumps({"event": "split_transition", "timestamp_monotonic_s": _t,
                                   "run_id": "e2_a_r4_ON_4386_1"}) + "\n")
                _f.write(_j.dumps({"event": "runtime_snapshot", "phase": "benchmark",
                                   "timestamp_monotonic_s": _t, "stream_index": 2,
                                   "prefill_sms": 64, "decode_sms": 44,
                                   "decode_running_batch_size": 1, "decode_iterations": int(_t),
                                   "prefill_active_batch_size": 1, "prefill_queue_depth": 1,
                                   "run_id": "e2_a_r4_ON_4386_1"}) + "\n")
        if _m.compute(_p, None)["split_transition"] != 5:
            failures.append("H-B3 regression: unfiltered Q4 count changed")
        if _m.compute(_p, 3.0)["split_transition"] != 3:
            failures.append("H-B3 regression: A3 probe filter does NOT apply to Q4")
        _b = os.path.join(_d, "b.jsonl")
        _rec = {"completed": 4, "total_input_tokens": 4 * 256, "random_range_ratio": 1.0,
                "duration": 2.0, "output_lens": [512] * 4}
        open(_b, "w").write(_j.dumps(_rec) + "\n")
        if _m.bench_validity(_b, 4, 256, 512)["all_pass"]:
            failures.append("H-B7 regression: absent `errors` is fail-OPEN")
        _rec["errors"] = [""] * 4
        open(_b, "w").write(_j.dumps(_rec) + "\n")
        if not _m.bench_validity(_b, 4, 256, 512)["all_pass"]:
            failures.append("H-B7 regression: a clean bench record no longer passes")
```
그리고 완료 문구를 `"8 reachability checks + 8 repair-regression checks (H-F1/H-F2/H-B1/H-B2/
H-B3/H-B5/H-B7)"` 로, sbatch `:166`의 `rules 8 reachability` 도 같이 갱신.
**검증 실측**: 대조군 `SELFTEST_OK 8+8`; `revert_HB2_armlabel_read` → `KILLED`,
`revert_HB3_q4_filter` → `KILLED`, `revert_HB7_errors_failopen` → `KILLED`.

### 3-4. **#4 (장부)** `achieved` 부재를 fail-close하라 (§2-5의 새 구멍)

`e2_label.py:113`의 H-B1 주석 **직전**에:
```python
            if bench.get("achieved_req_s") is None:
                out["unresolved"].append({"cell": cell, "seed": seed,
                    "why": f"{arm}: bench passed P9-P12 but carries no achieved_req_s "
                           f"(duration missing or zero) -- R2 would silently lose n"})
                break
```
**검증**: 적용 후 `UNRESOLVED(0)` → `UNRESOLVED(2)`, 정상 경로 무변화, 셀프테스트 불변.
★**자기 적용(교훈 53)**: 이 수리에도 되돌림 검사를 함께 넣어라 — 내 변이
`revert_P2_achieved_check`는 **SURVIVED** 했다(즉 이 수리도 회귀망이 없으면 다시 썩는다):
```python
    cells = build()
    del cells[("a_r4", "ON", "4386")]["bench"]["achieved_req_s"]
    if not any(u["cell"] == "a_r4" and u["seed"] == "4386" for u in label(cells, pairs)["unresolved"]):
        failures.append("#4 regression: bench without achieved_req_s is fail-OPEN")
```

### 3-5. 선택 처방 (권고 · 필수 아님 — 전부 판정 규칙 불변)

- **(S1) P3 fail-open 봉인**. 실측: 배너 두 값이 **둘 다 파싱 실패하면** 키가 매 boot `"NONE NONE"`
  이라 `distinct=1` ⇒ **abort하지 않는다**. 11/11 실제 로그에서는 `48 2722025`로 정상 추출되므로
  현 빌드에서 발화 불능은 아니지만, 한 줄로 닫힌다:
  ```bash
    [ "${MMB:-NONE}" = NONE ] || [ "${MTT:-NONE}" = NONE ] && { \
      echo "ABORT_P3 banner values not parsable in $SRVLOG" \
        | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; kill "$PID" 2>/dev/null; exit 8; }
  ```
- **(S2) 첫 boot E-pact 조기 경고**(GPU 절약 전용, **판정 아님**). `:358` `tail -30 …` 뒤:
  ```bash
    python3 -c "import json,sys;d=json.load(open('$OUT/cell_${NAME}.json'));e=d.get('e_pact') or {};\
  print('EARLY_WARNING e_pact %s/%s=%.2f%% -- A6 will mark this pair UNRESOLVED'%(e.get('num'),e.get('den'),e.get('pct'))) \
  if e.get('den') and e['pct']<100.0 else None" || true
  ```
  **인쇄 전용**이다. 이걸 보고 작업을 죽이는 것은 운영자 판단이고 라벨을 만들지 않으므로
  어떤 라벨도 편향시킬 수 없다.
- **(S3) P4 glob 확장**: `sha256sum "$PREREG"/VERDICT_e2_rules*.md` → `VERDICT_e2_*.md`.
  현재 **하네스층 판정서 2건이 등록 digest에 들어가지 않는다**.
- **(S4) P5 계수 정밀화**: `git status --porcelain`은 미추적 **디렉터리를 1줄**로 센다(실측:
  `porcelain=1` vs `porcelain -uall=11`). `-uall`을 붙여라.
- **(S5) 주석 1건**: `e2_label.py:113`의 `# H-B1: the probe-clock guard …` 이 **H-B2 검사
  바로 위**에 있다(시계 검사는 `:118`). 두 줄로 나눠라.

---

## 4. 신규 caveat — **E2C-29 … E2C-35** (결과 문서·정본이 **문자 그대로** 승계할 것)

> **E2C-29 (필수 병기 — R1은 자기가 선 셀 집합을 말하지 않는다)** — *"`rules.R1.label`은 단일
> 전역 라벨이며, R5 `SEED_SPREAD_FAILS`나 UNRESOLVED 장부가 **등록 셀 하나를 통째로** 떨어뜨려도
> 그 사실을 라벨 문자열에 담지 않는다."* 감사자 실측(쉬핑 경로): 같은 데이터에서 `a_r4` OFF
> E-cnt 한 칸(8.65→15.5)만 바꾸면 R1이 **`OFF_BASELINE_ABOVE_BAND` → `STICKY_REALIZES_A`** 로
> 뒤집힌다(R1이 실제로 선 셀 `['a_r2','a_r4']` → `['a_r2']`). ⇒ **(a)** R1을 인용할 때
> `rules.R1.per_seed`의 **셀 목록**과 `unresolved`·`rules.R5`를 **반드시 병기** **(b)** 어떤
> 등록 셀이라도 빠진 R1을 *"shape A에서 sticky가 D44를 실현했다"* 로 쓰는 것을 **금지**
> **(c)** E2C-28(seed 단위 축소)의 **셀 단위 강화판**이며 E2C-28을 대체하지 않고 겹쳐 적용한다.

> **E2C-30 (병기 — P8은 배선 점검이지 처치 점검이 아니다)** — *"P8이 통과했다는 것은
> `PDMUX_RUN_ID`가 엔진을 거쳐 그 telemetry 파일에 도달했다는 뜻이지, **그 boot에서 sticky가
> 실제로 켜졌다/꺼졌다는 뜻이 아니다**."* 후자는 **P1/P2**(서버 로그의 `sticky partition ENABLED`
> + `fixed target index=2`)만이 말한다. 또 P8은 **arm 교차만** 잡고 **cell/seed 교차는 못 잡는다**
> (감사자 실측: `e2_a_r2_OFF_4162_…`를 `--arm OFF`로 보면 통과). 216 run_id 전수에서 오탐 0·미탐 0.

> **E2C-31 (병기 — E-pact는 두 arm 모두 구성 항등식이고, 그래서 발화하면 계측이 바뀐 것이다)** —
> **(a)** 감사자 실측: 908623 아카이브 **7/7 셀에서 정확히 100.000%**(분모 24 / 32 / 35 / 77 / 135 /
> 400 / 400, 총 1,103표본, 이탈 0). 기전은 코드 직독으로 확증 — OFF는 `multiplexing_mixin.py:1170`
> 의 prefill-span 조건 + v7 고정 타깃, ON은 `:1172-1179`가 `_sticky_fixed_idx`를 **무조건** 설치.
> ⇒ A6의 `[95,100)` 조항이 발화하면 그것은 **sticky에 대한 정보가 아니라 계측 변경 신호**다
> (E2C-4의 실측 보강). **(b)** `e2_label.py:126`의 `if ep.get("den")` 때문에 **분모가 0이면 가드가
> 조용히 꺼진다** — "통과"와 "미측정"이 같은 출력이다. E-pact를 인용할 때 **분자/분모를 반드시 병기**.

> **E2C-32 (필수 병기 — R5의 정의역이 R5의 라벨을 바꾼다)** — *"R5는 등록 seed 전체가 아니라
> **R3(ii)·유효성 장부를 통과한 `live` 짝**의 E-cnt만 본다."* 감사자 실측: 동일한 OFF 산포
> 6.85pp가, 산포를 만드는 seed의 짝이 R3(ii)에 먼저 걸리면 `R5 = PASS (0.00pp)`, 안 걸리면
> `R5 = SEED_SPREAD_FAILS (6.85pp)`가 된다 — **그리고 R1의 정의역이 2셀↔1셀로 갈린다.**
> 등록 rev3 §6 R5는 정의역에 침묵한다. ⇒ 追記 A11(e)로 못박기 전에는 **R5 라벨을 "seed 재현성"
> 진술로 인용 금지**. 방향은 **관대**(탈락이 어려워지는 쪽)임을 함께 적어라.

> **E2C-33 (병기 — P3는 두 값이 모두 파싱되지 않으면 진공이다)** — 감사자 실측: `MMB`/`MTT`가
> 둘 다 추출 실패하면 키가 매 boot `"NONE NONE"` 이라 `sort -u | wc -l = 1` ⇒ **abort하지 않는다.**
> 현 빌드에서는 11/11 로그가 `48 / 2722025`로 정상 추출되므로 **발화 불능은 아니다**.
> ⇒ P3 통과를 인용할 때 `BANNERS.txt`의 **실제 값**을 병기하고, `NONE`이 보이면 P3는 무효다.

> **E2C-34 (병기 — §5-3 기대값 2칸의 출처 서술이 부정확하다)** — `e2_realized_mix.py:34`의
> *"idx0/idx3/split_transition/busy n → rev2 verdict sec 2-3, re-derived"* 는 **두 칸에 대해 거짓**
> 이다. 감사자 전수 grep: `a_r0` idx0 **`1/2677`** 과 `a_r4_s2` idx0 **`0/557`** 은
> **`PREREG_E2_STICKY_REV3_2026-09-15.md`(:146, :149) 단독**이며 rev1/rev2/rev3 판정서·ADDENDUM·
> λ0 판정서 어디에도 없다(다른 idx0 칸 `2/555`·`13/313`·`1/441`·`0/610`·`0/441`은 2출처 이상).
> **E2C-27이 지배 서술이다.** 이 트랙에서 **결정경로 파일 안의 출처 허위는 이번이 4번째**다(교훈 80).
> 두 칸의 **값 자체는 원자료에서 재현된다**(셀프테스트 통과) — 틀린 것은 출처 문장뿐.

> **E2C-35 (병기 — 수리 5건은 제출 차단 검사망 밖에 있다)** — 감사자 변이 실측: **H-F1 생산자 ·
> H-B2(생산자·소비자) · H-B3 · H-B7 · H-B8** 을 되돌린 변이본이 **세 셀프테스트를 전부 통과한다**.
> 이번 회차는 그 다섯을 **기능 시험으로** 확인했지만(§1), **회귀망은 없다.** ⇒ 이 파일들을 다시
> 편집하면 그 다섯은 **아무도 지키지 않는다**. 선행조건 #3을 적용하면 셋(H-B2/H-B3/H-B7)이 덮인다.

**기존 승계 전부 유효**: E2C-1 … E2C-7 · **E2C-8′** · E2C-9 · E2C-10 · E2C-11(E2C-19로 재정정) ·
E2C-12 · E2C-13 · E2C-14 · **E2C-15 … E2C-21** · **E2C-22 … E2C-28**(직전 하네스층 판정서) ·
λ0R-1…λ0R-10 · λ5C-1…8 · NPC-I(shape A 반증) · N-7·N-8·N-9·N-11 · 게이트 #13/#16 "닫았다" 금지 ·
C2 인용정지 (a)(b) · HE0 · layer-type 死 · 정책 순위 · stake #1 구조 판정.

**이 회차가 바꾸지 않는 것**: 새 성능 판정 **0건** · Claim D/E 등급 **불변** · 게이트 #6 **불변** ·
P2 블로커 ①②③ **불변** · **GPU 지출 0**.

---

## 5. 반전 시험 표 (자유 표면 11개를 끝까지 밀었다)

| 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 (전부 이 감사 실측) |
|---|---|---|---|
| **R5 정의역**(등록 seed 전체 ↔ post-R3(ii) live) | 양 끝 | ★**뒤집힘** | R5 `PASS (0.00pp)` ↔ `SEED_SPREAD_FAILS (6.85pp)`; R1 정의역 2셀↔1셀 |
| **R1 도메인 공시**(있음/없음) | 양 끝 | ★**뒤집힘** | `OFF_BASELINE_ABOVE_BAND` ↔ `STICKY_REALIZES_A` (e_cnt 8.65→15.5 한 칸) |
| **`achieved` 부재 처리**(fail-open/closed) | 양 끝 | ★**뒤집힘** | `UNRESOLVED(0)` + `R2 n=2` ↔ `UNRESOLVED(2)` |
| **decode-busy 필드**(3후보) | 전부 | 뒤집힘 — **셀프테스트가 사살** | busy 555→**20,082**, E-cnt 8.65→**0.24** |
| **E-time 간격 규약**(다음 스냅샷 ↔ 다음 busy) | 양 끝 | 뒤집힘 — **셀프테스트가 사살** | den `a_r4` 183.61→**217.50**, `b_r3` 292.00→**334.99** |
| **E-iter 좌 busy 게이트** | 제거 | 뒤집힘 — **셀프테스트가 사살** | 8.76→**8.66**, 92.29→**90.87** |
| **E-qcond 조건절** | 제거 | 뒤집힘 — **셀프테스트가 사살** | 43/250→**48/555** |
| **E-pact 밴드 `[95,100)`** | 1표본 강제 이탈 | 뒤집힘 **단 등록대로** | 33/34 = 97.06% → **10짝 전부 UNRESOLVED, R1=UNRESOLVED** |
| **P8 단편 매칭 `_{arm}_`** | 216 run_id 전수 + 혼재·`unlabeled`·공백 | **안 뒤집힘** | 미탐 **0/216** · 오탐 **0/216** |
| **bench 실패 3형태**(키부재/124/크래시) | 전부 | **안 뒤집힘**(셋 다 짝 UNRESOLVED) | `why`만 갈림, R2 n=3 동일 |
| **P3 배너 추출**(정상/양쪽 NONE) | 양 끝 | **안 뒤집힘**(판정 불변) — 단 NONE이면 게이트 진공 | 11/11 로그 `48 / 2722025` 정상 |

**반전 계산에 쓴 원자료·격자·추정량**(반전 계산 자신이 새 자유 표면이 되지 않도록 명시):
원자료 = `results/r2_eval/lambda0_prereg/lam0_908623/{tel,bench,srv}_*` **및** 아카이브 값에 앵커한
합성 `cell_*.json` 20개(생성기는 스크래치의 `mkcells.py`, OFF 수준 = λ0 §4-A 실측
`a_r4 8.65/13.64/8.76/17.20`, ON = 엔진 강제 ~99%, achieved = 아카이브 `3.0531/2.8079/0.6972`).
격자 = 등록 셀·arm·seed 20 boot 그대로. 추정량 = **하네스 자신의 `compute`/`bench_validity`/
`label`을 호출**했고 손으로 만든 라벨은 **0건**. 판정은 전부 **쉬핑 경로**
(`e2_label.py --cells-dir … --expect … --out`)로 냈지 `label()` 내부 상태를 읽지 않았다.

---

## 6. 반증 실패 — 깨려고 했고 **깨지 못한** 12건

1. **P8 오탐/미탐**(의뢰 2순위-1) — 216 run_id 전수 **0/0**. 실제 telemetry 왕복 실측도 정상. §2-1.
2. **E-pact 밴드 과발화**(의뢰 2순위-2) — 1,103/1,103 표본 정확히 100%, 기전은 **양 arm 구성 항등식**
   (코드 직독 `:1170` / `:1172-1179`). ON에서 `prefill_active>0`이 idx0일 경로는 **없다**. §2-2.
3. **R5 이중 처벌**(의뢰 2순위-3) — 소비되는 결과는 UNRESOLVED 하나뿐. 등록대로. §2-3.
4. **H-F1 우회 경로** — `bench` 키 부재 · rc 124 · rc 1 · **파일이 있지만 깨진 JSON**(→
   `bench_validity` 예외 → `cell_*.json` 미생성 → `missing boot`) **네 경로 전부 fail-closed**.
5. **`--probe-end` 인자 주입**(bash) — `${PROBE_END:+--probe-end "$PROBE_END"}`가 내부 따옴표를
   보존한다(실측: 공백 포함 값도 **argc=5, 단일 인자**). 미설정 시 인자가 사라진다(argc=3).
6. **`[ x ] && A=1 || B=1` 관용구**(`:321`) — 두 arm 모두 의도대로(`OFF→OFF=1,ON=0` · `ON→OFF=0,ON=1`).
7. **A3의 모집단 주장** — 게이트 #110으로 재검증: 세 셀 전부 **첫 비-startup 스냅샷이 이미
   decode-busy**(`decode_running_batch_size=1`), 처음 20간격의 최대 간격 **0.043–0.120 s** ⇒
   고립 선행 클러스터 없음 ⇒ 프로브 필터가 warm-up을 함께 자르지 않는다.
8. **시계 가드 양방향** — `probe_end`를 스팬 **위(1e9)** 와 **아래(50)** 양쪽으로 밀어 둘 다
   `ABORT_PROBE_CLOCK_MISMATCH` 기록 확인. 아래쪽에서는 레코드가 남아 label이 읽고 UNRESOLVED.
9. **P14가 R4″를 죽이지 않는다** — 독립 재산술: 3번째 `b_r3` boot 직전 `REMAIN = 5097.2 s ≫ 1061 s`,
   최악 코너 `≈ 2973 s`.
10. **통제요인 동일성** — `lambda0.sbatch`와 launch 인자 14줄·env 2개 전수 일치, `T_cap` 3배 재산출 일치.
11. **P4/P5 preflight** — digest **12 ≥ 10**, manifest 게이트(`nemotron_h.py` + ≥24) 그대로, 셀프테스트
    3종 rc 0, 총 소요 **≪ 400 s**.
12. **`break 2` 이후 라벨링 도달** — P6/P14 경로는 `exit`가 아니라 `break 2`이므로 사후 라벨 블록
    (`:392-394`)에 도달한다(bash 구조 확인). 미시도 짝은 `missing OFF boot`로 UNRESOLVED.

---

## 7. 자기 적용

### 7-1. 게이트 #110 — 직전 판정서를 등록 상수로 승격하지 않았다
직전 판정서의 수치를 **하나도 승계하지 않고** 독립 재산출했다: B2″의 `555→20,082`·`8.65→0.24`
(변이로) · E-time 역전 값 `183.61→217.50`·`292.00→334.99`(내가 만든 busy-부분수열 변이로) ·
E-iter `8.76→8.66`·`92.29→90.87` · E-qcond `43/250→48/555` · A1의 E-time 7쌍(EXPECTED 대조) ·
A3의 모집단 주장(아카이브 직독) · P3 배너(11/11 로그) · 예산 산술 · `T_cap` 3배 · E2C-27의
단독 출처 2칸(전수 grep) · E2C-17의 256 구조.
**결과**: 직전 판정서의 지적 중 **내가 뒤집은 것은 없다**. 반대로 직전 판정서와 수리본이
**둘 다 놓친 것 3건**을 냈다 — §3-1(R1 도메인) · §2-5(achieved 부재) · E2C-34(출처 허위 4번째).

### 7-2. ★무변이 대조군 — 직전 회차가 스스로 빠졌다고 공시한 함정
19 변이 전부에 **무변이 대조군**을 동반시켰고 대조군은 `SELFTEST_OK`를 냈다. 그리고
**나 자신도 등가 변이 1건을 만들었다**(§2-4 말미) — 앵커가 `j = i + 1`만 추가해 E-time을
바꾸지 않았고 SURVIVED를 냈다. 이것을 "검사 구멍"으로 기록하지 않고 앵커를 고쳐 다시 돌렸다.
**교훈은 한 단계 더 나아간다: 대조군은 거짓 사살을 잡지만, 거짓 생존은 잡지 못한다 —
변이가 실제로 무엇을 바꿨는지 수치로 확인해야 한다.**

### 7-3. 게이트 #113 — 내 처방의 실현가능성·비용·노브 결합
- **네 처방을 전부 임시 복사본에 적용해 실행했다**: `e2_label.py --selftest` →
  `SELFTEST_OK 8 reachability + 8 repair-regression`(불변 통과) · `e2_realized_mix --selftest` →
  `7/7`(불변) · 정상 경로 4시나리오 **라벨 무변화** · 되돌림 변이 3건 **전부 KILLED**(대조군 통과).
- **비용**: GPU **0**. 코드 `e2_label.py` **~30행**(대부분 회귀검사), 문서 **追記 A11 1절**.
  **rev4 사전등록 불요.**
- **노브 결합 대조**(교훈 250): 내 처방 어느 것도 env · 서버 인자 · seed · `num_prompts` ·
  telemetry 격자 · 다섯 추정량 규약 · §5-3 기대값 · R1/R2/R3/R4 문턱을 **바꾸지 않는다**.
  유일하게 노브를 움직이는 후보였던 **(S2) 조기 경고**는 **인쇄 전용**으로 못박아
  라벨 경로에서 분리했다.
- ★**내 처방이 사지 못하는 것**: **(a)** §3-1을 고쳐도 R1은 여전히 **단일 라벨**이고 셀별 R1을
  만들지 않는다 — 그것은 등록 변경이라 하네스층이 할 일이 아니다. **(b)** §3-3을 다 넣어도
  **H-B8·H-F1 생산자는 여전히 회귀망 밖**이다(E2C-35 잔존). **(c)** 어느 처방도 **검정력을 사지
  않는다** — R2는 여전히 n=4(E2C-22: 0.33% 효과에서도 `MOVES_DOWN`), R4″는 E2C-18대로 드레인
  축을 못 본다. **(d)** E2C-21(shape A 처치 질량의 86–89%가 prefill 유휴)은 **한 글자도** 안 움직인다.

### 7-4. 게이트 #184 — 처방자 자기승인 금지
이 하네스의 수리는 **직전 하네스층 판정서(내 전임자)의 처방**이다. 승인하지 않고 시험했다:
H-F1(a) → 3형태 실행 · H-F2 처방 → **되돌림 변이 2종으로 사살 확인 + 도메인 자유 표면 발견(E2C-32)** ·
H-F3 처방 → 실제 서버 로그 11개 + bash 재현 · H-B1/B5 → 양방향 분지 도달 · H-B2 처방 →
**216 전수 + 엔진 코드 직독으로 재설계 근거 재확인, 그리고 한계 2개 발견(E2C-30)** ·
H-B3 → 합성 프로브로 8→3 · H-B7 → 5형태 · H-B8 → rc 2/0.
⇒ **처방 11건 중 2건에서 전임자가 보지 못한 결과를 냈다**(E2C-30, E2C-32).

### 7-5. E2C-8′를 나 자신에게 (내가 발표하는 모든 수치에 규약 병기)
- **E-pact** = `event=="runtime_snapshot" ∧ phase!="startup" ∧ decode_running_batch_size>0 ∧
  prefill_active_batch_size>0`, 균등, 분자 = `stream_index==2`. 분자/분모 전부 병기(24–400).
- **E-cnt/E-time/E-iter/E-qcond** = 등록 리터럴 그대로(E-time은 다음 비-startup 스냅샷까지·무캡·
  마지막 제외; E-iter는 좌 귀속·차분>0·좌 busy 게이트·가중=차분).
- **P8 전수 열거** = cell 3 × arm 2 × seed 4 × jobid 9형(숫자 6종 + `local12345` + `908623` +
  `20260915`) = 216, 판정식 `f"_{arm}_" in run_id`.
- **변이 사살** = 두 셀프테스트의 rc(≠0 = KILLED), 대조군 동반, 절대경로 고정, `--archive` 명시.
- **타이밍** = venv 인터프리터(`/scratch/ehmoon/whlee/sglang_engine_venv/bin/python`), 벽시계,
  Lustre 캐시 상태에 따라 범위로 보고.
- **재현 경로**: 저장소 파일은 **한 글자도 고치지 않았다**(이 판정서 생성 제외). 변이·합성·
  패치는 전부 `/tmp` 스크래치 복사본. 쉬핑 라벨 코드는 **호출**했고 손으로 만든 verdict는 없다.

### 7-6. 등급 인플레/디플레 점검
- **인플레 방지**: 남은 결함 4건을 caveat로 **강등하지 않았다** — 전부 **필수 선행조건**으로
  올리고 리터럴 문안을 붙였다. §3-1은 *"정상 경로였다면 死因"* 이라고 명시했다.
- **디플레 방지**: 12건을 **반증 실패**로 남겼다. 의뢰가 직접 의심한 두 항목(P8 오탐/미탐 ·
  E-pact 과발화)은 **둘 다 무해로 판정**했고, **완화 문안을 처방하지 않았다** — 등록을 사후에
  느슨히 하는 것이 새 자유 표면이기 때문이다. *"찜찜하니 차단"* 을 하지 않았다.
- **자기 고지**: §3-1을 死因으로 등급해도 **수리 문안과 캠페인 설계는 동일하다**. 등급을 가른
  것은 *"그 분지가 등록 예보 경로 위에 있는가"* 뿐이다(아카이브 seed 산포 **0.39pp** vs 문턱 **5pp**).

---

## 8. 제출 절차 (순서대로)

1. **선행조건 4건 적용** — §3-1(4행) · §3-2(追記 A11) · §3-3(회귀검사 3건) · §3-4(4행 + 회귀검사 1건).
   선택 처방 S1–S5는 재량. **전부 GPU 0.**
2. **셀프테스트 3종 재실행** — `e2_realized_mix --selftest`(7/7) · `e2_label --selftest`
   (**8 + 9** 가 될 것) · `test_sticky_partition`(12). 셋 다 rc 0.
3. **하네스층 재감사 불요.** 선행조건은 전부 문안·보고·회귀망이고 판정 규칙을 바꾸지 않는다.
   단 §3-1을 **코드가 아니라 문안(E2C-29 승계)으로 닫는다면** 그 문장이 결과 문서에 실제로
   들어갔는지 doc-steward가 확인해야 한다.
4. **새 범위 한정 OVERRIDE + 사용자 승인**(예산 **1.803 GPU-h**, 최악 **2.338**, `--time 03:00:00`).
5. **커밋**(P5 — 현재 이 디렉터리는 **미추적**이다: `porcelain -uall = 11`) → 그 다음에만 `sbatch`.

---

## 9. 한 줄 요약

> **`GO-with-caveats` — 죽어 있던 실패닫힘 세 개가 전부 살아났다.** P7 `CAPPED`는 이제 키 부재·
> rc 124·rc 1 **세 형태 모두** 짝을 `UNRESOLVED`로 만들고(그리고 **아티팩트에서 서로 구별된다**),
> R5는 **R1/R2보다 먼저** 돌아 `SEED_SPREAD_FAILS`가 그 셀의 4짝을 통째로 떨어뜨리며, P3는
> **boot마다** 실제 배너 값(`48 / 2722025`, 11/11 로그 실측)으로 `ABORT_P3` 한다. 차단 8건도
> 전부 닫혔고, 의뢰가 의심한 두 신규 표면은 **깨지지 않았다** — P8 단편 매칭은 216 run_id 전수에서
> **오탐 0·미탐 0**, E-pact `[95,100)` 밴드는 **1,103/1,103 표본이 정확히 100%** 이고 그 100%가
> 양 arm **구성 항등식**(코드 직독)이라 과발화하지 않는다. 남은 것은 **비정상 분지 4건**이고
> 전부 **GPU 0 · 코드 30행 + 追記 1절**이다 — 그 중 §3-1(R5가 셀을 떨어뜨렸을 때 R1이
> `OFF_BASELINE_ABOVE_BAND` 대신 `STICKY_REALIZES_A`를 낸다)은 **정상 경로였다면 死因**이었다.
> 그리고 이 회차가 사는 것은 여전히 좁다 — shape A 처치 질량의 **86–89%가 prefill 유휴**(E2C-21),
> R2는 n=4에 **최소 효과크기가 없고**(E2C-22), **게이트 #6 · P2 블로커 · Claim D/E는 하나도
> 닫히지 않는다.** **GPU 지출은 이 판정서 시점까지 0이다.**
