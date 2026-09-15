# 판정서 — **E2 (sticky 분할 대조) 하네스층 감사** (2026-09-15)

> **2단 감사의 2단**(교훈 34). 1단(규칙층)은 `VERDICT_e2_rules_rev3_2026-09-15.md`
> (`396927b7…`) = **`GO-with-caveats`, 死因 0**. 이 판정서의 대상은 규칙이 아니라
> **"이 코드가 등록된 규칙을 그대로 구현하는가, 그리고 GPU를 쓸 수 있는 상태인가"** 다.
> **GPU 지출: 이 판정서 시점까지 여전히 0.** 이 감사 자체도 GPU 0(전부 아카이브 재분석·
> 변이 실행·합성 라벨링).

## 감사 대상 (sha256 재확인 — 전부 의뢰서와 일치)

| 파일 | sha256 | 확인 |
|---|---|---|
| `e2_sticky.sbatch` | `4bccc0ec7e4ac462d6582f621a0cfc111f226640199a725777b28c1fe58f30d8` | ✓ |
| `e2_realized_mix.py` | `f87e71941d38de8873eb7977a49649c9cbdf8622ac34b608dc4be973b5c58b77` | ✓ |
| `e2_label.py` | `701d4616b95d72237f11b409208dfdfd44322fcf175773d181849cf17cc157d5` | ✓ |
| `PREREG_E2_STICKY_REV3_2026-09-15.md` | `8998d53277abf660bb42bffc388fc92aa05c8de61e991c241783d9c87e1b29a4` | ✓ |
| `PREREG_..._REV3_ADDENDUM_2026-09-15.md` | `56870ee5e10d853a20243563727f8aecc4a088cd7c3c3a847296f7eb1d51df70` | ✓ |
| `VERDICT_e2_rules_rev3_2026-09-15.md` | `396927b7d5f73e2617bc1ff5fe16f1a0080a99eaf063cfad9ebce10eb28fe962` | ✓ |

---

## 0. 판정

> # `NO-GO`
>
> **死因 3건 — 전부 "등록된 실패닫힘 게이트가 판정 경로에 존재하지 않는다"**:
> **H-F1** P7 `CAPPED`(및 bench 아티팩트 부재 일반)가 `UNRESOLVED`로 **전파되지 않는다** ·
> **H-F2** R5 `SEED_SPREAD_FAILS`가 등록된 결과(그 셀의 R1/R2 `UNRESOLVED`)를 **만들지 않는다** ·
> **H-F3** P3가 **abort 경로 자체가 없고**, 비교 키에 boot별 타임스탬프가 들어가 **확률 1로 경고**하며,
> `max_mamba_cache_size`의 **값은 비교 필드에 아예 들어오지 않는다**.
>
> 세 건 모두 **실제 프로그램 출력·실제 아카이브 로그로 재현했다**(§2). 추측이 아니다.
>
> **차단 8건**(死因 아님 · 전부 GPU 0) · **신규 caveat E2C-22 … E2C-28** · **반증 실패 13건**(§5).
>
> ★**이 `NO-GO`는 설계를 되돌리지 않는다.** 死因 3건은 **전부 코드 국소**이고 **사전등록을
> 한 글자도 바꾸지 않는다**(등록 문서가 이미 무엇을 해야 하는지 적어 두었고, 코드가 그것을
> 안 한 것이다). 나는 H-F1·H-F2의 처방을 **임시 복사본에 실제로 적용해 두 셀프테스트가
> 계속 통과하고 두 경로가 닫히는 것을 확인했다**(§6-3, 게이트 #113). 비용은 **rev4 사전등록이
> 아니라 하네스 1패스 + 셀프테스트 재실행**이다.

### 왜 이것이 "문안"이 아닌가 (등급 디플레 점검)

의뢰서의 기준은 *"死因은 등록 규칙과 다른 수를 내거나, 게이트가 발화하지 못하거나, 라벨이
뒤집히는 것에만 쓴다. 남은 것이 문안뿐이면 `GO`를 내라"* 다. 내가 남긴 것은 문안이 아니다:

- **H-F2는 라벨을 실제로 뒤집었다.** 합성이 아니라 `e2_label.py` 자신을 돌려서:
  R5 = `SEED_SPREAD_FAILS`(OFF 산포 6.5pp > 5pp)인데 **동시에** R1 = `STICKY_REALIZES_A`,
  R2 = `LAMBDA_WITHIN_CI`, `UNRESOLVED: []`. 등록은 그 셀의 R1/R2가 `UNRESOLVED`여야 한다고
  적는다(rev3 §6 R5 결과 열).
- **H-F1은 R1과 R2를 서로 다른 모집단 위에 올린다.** 같은 실행에서 R1은 4 seed로 라벨되고
  R2는 n=3으로 떨어져 `UNRESOLVED`가 된다.
- **H-F3은 실제 서버 로그로 재현했다** — 5 boot에서 `sort -u`가 10줄을 내므로 20 boot에서는
  40줄, `-gt 2`는 **항상 참**이다.

반대로 나는 **"찜찜하니 차단"도 하지 않았다**: 추정량 규약·부트스트랩·예산·시계 가정·
A2/A3/A7 처방·D23·P1/P2/P6/P9–P14는 전부 깨려고 시도했고 **깨지 못했다**(§5). 그 13건은
死因도 차단도 아니라 **반증 실패**로 기록했다.

---

## 1. 1순위 체크리스트 — 판정서 §9-5 10항 기계 확인

| # | §9-5 항목 | 결과 | 근거 |
|---|---|---|---|
| 1 | §4-2 E-time 열 = B1″-a 표의 7쌍 | **PASS** | `e2_realized_mix.py:243-249`가 `31.76/667.98 … 285.99/293.23`(A1 표)을 쓴다. rev3 §4-2의 역산 7쌍(`24.95/525.2` 등)은 **어디에도 없다** |
| 2 | §5 표 머리에 decode-busy 리터럴 | **PASS** | `e2_realized_mix.py:56-63` `decode_running_batch_size > 0`, 주석이 `decode_batch_size`·`active_decode_sequences` 불사용을 명시 |
| 3 | §7 P13에 `t_probe_end` 기록 + 추정량 배제 | **부분 PASS** | 다섯 추정량·Q3는 배제됨(`:66-96`). **Q4(`split_transition`)는 배제되지 않는다** → 차단 **H-B3** |
| 4 | §7 P14 "남은 예산 = 10,800 − `SECONDS`" | **PASS** | `e2_sticky.sbatch:80,191` `WALL_BUDGET_S=10800` · `REMAIN=$(( WALL_BUDGET_S - SECONDS ))` |
| 5 | §6 R1 결과 문안 다분지 | **PASS** | `e2_label.py:123-132` 4분지(A5가 요구한 4분지를 전부 구현; §9-5의 "3분지"는 A5의 4분지가 정본) |
| 6 | §5 E-pact ≥95.0% 밴드 | **부분 PASS** | 상수 `E_PACT_ABORT_BELOW = 95.0`(`:48`)이 **셀프테스트에서만** 쓰인다(`:293-294`). **런타임 판정 경로에 E-pact 밴드가 없다** → 차단 **H-B5** |
| 7 | §7 P13 전례 프롬프트 6개 + `max_new_tokens=48` + 판정 대상 = `text` 리스트 | **PASS** | `e2_sticky.sbatch:277-288` ↔ `sticky_smoke.sbatch:115-133` **문자 그대로 동일**; 판정은 `:366-368` `a == b` (전례의 `a == b`) |
| 8 | §9에 E2C-15 … E2C-21 문자 승계 | **PASS** | `e2_label.py:252-271` 15건 승계(E2C-21 포함, 문자열 본문에 "86-89%" 수치까지) |
| 9 | `PDMUX_STICKY_PARTITION`이 일괄 unset 루프에서 빠져 있는지 | **동작 PASS / 주석 FAIL** | `e2_sticky.sbatch:66`에 **들어 있다**. 단 그 unset은 **루프 밖**(`:63-66`)이고 루프 안 `:215`가 매 boot 재설정하므로 게이트 #237은 발생하지 않는다. **주석 `:61-62`가 "deliberately absent from this list"라고 거짓 서술** → 차단 **H-B6** |
| 10 | §5-3 셀프테스트 통과 | **PASS** | 실측: `e2_realized_mix --selftest` **rc 0, 7/7, 4.98 s** · `e2_label --selftest` **rc 0, 8 reachability, 2.00 s** · `test_sticky_partition` **rc 0, Ran 12 tests** |

---

## 2. 死因 (3건) — 전부 실행으로 재현

### H-F1. **P7 `CAPPED`(및 bench 아티팩트 부재)가 판정 경로에 도달하지 않는다 — R1과 R2가 다른 모집단 위에 선다**

**등록**: rev3 §6 *"**UNRESOLVED 규약**: boot 실패·**CAPPED(P7)**·유효성 위반(P9–P12)·예산 절단(P14)·
셀 누락은 **그 seed의 OFF/ON 짝을 통째로** UNRESOLVED로 만들고…"*, §7 P7 *"초과 시 **CAPPED** ⇒
그 seed 짝 UNRESOLVED"*.

**구현**: `e2_sticky.sbatch:326-335`
```bash
timeout "${TCAP}s" python -m sglang.bench_serving … --output-file "$OUT/bench_${NAME}.jsonl" …
BRC=$?
[ "$BRC" = 124 ] && echo "CAPPED $NAME hit T_cap=${TCAP}s" | tee -a "$OUT/CAPPED.txt"
```
`$BRC`는 **어떤 아티팩트에도 실리지 않고**, `CAPPED.txt`는 **`e2_label.py`가 읽지 않는다**.

**기전(코드로 확인)**: `bench_serving.py:1629`의 `--output-file` 쓰기는 `benchmark()`의 **맨 끝**에만
있고, 그 파일 전체에 **signal/atexit/KeyboardInterrupt 처리가 0건**이다(grep count 0). ⇒ `timeout`의
SIGTERM은 **파일을 남기지 않는다**. 그러면
`e2_realized_mix.py:334` `if args.bench and os.path.exists(args.bench):` 가 거짓 ⇒ `cell_*.json`에
**`bench` 키 자체가 없다** ⇒ `e2_label.py:99`
```python
            if rec.get("bench") and not rec["bench"]["all_pass"]:
```
가 **거짓**(키 부재는 falsy) ⇒ 그 짝은 `UNRESOLVED`가 **되지 않는다**.

**재현(실제 출력, `e2_label.py` 원본)** — `a_r4/4386`의 ON boot만 CAPPED로 둔 합성 셀 집합:
```
  R1 = STICKY_REALIZES_A | on_fail [] off_fail []
  R2.a_r4 = UNRESOLVED (n=paired n=3 < 4 (gate #3))
  UNRESOLVED: []
```
⇒ **`UNRESOLVED`가 비어 있고, R1은 절단된 boot를 포함해 라벨되며, R2만 조용히 n=3으로 떨어진다.**
등록이 요구한 것은 그 짝의 **통째 UNRESOLVED**다.

**이 경로는 `timeout`보다 더 흔한 입구를 갖는다**: `bench_serving.py:1597-1599`에서 벤치가
실패하면 `result`가 바인딩되지 않은 채 `:1631`의 `result | result_details`에 도달해 `NameError`로
죽는다 ⇒ **서버가 bench 중 죽거나 OOM이면 같은 "bench 파일 부재"** 가 되고, 같은 구멍으로 샌다.
sticky ON은 shape A에서 decode를 108→44 SM으로 묶는 처치라 **ON arm이 이 경로에 들어갈 확률이
OFF보다 구조적으로 높다** — 즉 이 누수는 **arm 비대칭**이다.

**리터럴 수리**(둘 중 하나로 충분; 나는 (a)를 적용해 검증했다):
- **(a)** `e2_label.py:99` 를
  ```python
              if not rec.get("bench") or not rec["bench"]["all_pass"]:
  ```
  로 바꾼다. `--bench`는 sbatch가 **항상** 넘기므로 "키 부재 = 아티팩트 부재 = 유효성 실패"다.
- **(b)** (권장 병기) `e2_sticky.sbatch:339-344`에 `--bench-rc "$BRC"`를 추가하고
  `e2_realized_mix.py:334`의 else 분지에
  ```python
      else:
          res["bench"] = {"all_pass": False,
                          "why": f"bench artefact absent (bench_rc={args.bench_rc}; 124 = P7 CAPPED)",
                          "observed": {"bench_rc": args.bench_rc}}
  ```
  를 둔다 — 그래야 아티팩트에서 **CAPPED와 크래시가 구별**된다(현재는 둘 다 "키 없음"으로 같다).

---

### H-F2. **R5 `SEED_SPREAD_FAILS`가 등록된 결과를 만들지 않는다 — 라벨이 실제로 뒤집힌다**

**등록**: rev3 §6 R5 — *"`a_r4`·`a_r2`에서 seed 간 E-cnt 산포가 arm별로 ≤ 5pp | 거짓 ⇒
`SEED_SPREAD_FAILS` ⇒ **그 셀의 R1/R2 UNRESOLVED**"*.

**구현**: `e2_label.py:231-243`이 R5를 **R1(`:112-136`)·R2(`:138-172`) 뒤에** 계산하고, 그 결과를
**아무도 소비하지 않는다**. `SEED_SPREAD_FAILS`는 JSON의 한 필드로 끝난다.

**재현(실제 출력, `e2_label.py` 원본)** — `a_r4` OFF의 seed 2630만 E-cnt 9.0 → 15.5(산포 6.5pp):
```
  R1 = STICKY_REALIZES_A
  R2.a_r4 = LAMBDA_WITHIN_CI
  R5.a_r4 = SEED_SPREAD_FAILS {'OFF': 6.5, 'ON': 0.0}
  UNRESOLVED: []
```
⇒ **등록이 `UNRESOLVED`를 요구하는 자리에서 `STICKY_REALIZES_A` + `LAMBDA_WITHIN_CI`라는
두 개의 적극적 라벨이 나온다.** 이것이 이 판정서의 유일한 **직접 라벨 반전**이다.

**리터럴 수리** — R5를 R3(ii) 직후·R1 앞으로 옮기고 실패 시 그 셀의 전 짝을 `unresolved`에
넣는다(무순환: R5는 post-R3(ii) `live` 집합의 `e_cnt`만 읽는다):
```python
    # ---- R5 FIRST: SEED_SPREAD_FAILS => that cell's R1/R2 are UNRESOLVED (prereg sec 6 R5).
    out["rules"]["R5"] = {}
    for cell in SHAPE_A_CELLS:
        per_arm = {}
        for arm in ("OFF", "ON"):
            vals = [cells[(cell, arm, s)]["e_cnt"]["pct"] for c, s in live if c == cell]
            per_arm[arm] = (max(vals) - min(vals)) if len(vals) > 1 else None
        spreads = [v for v in per_arm.values() if v is not None]
        lab = ("PASS" if spreads and all(v <= R5_SEED_SPREAD_MAX_PP for v in spreads)
               else ("SEED_SPREAD_FAILS" if spreads else "UNRESOLVED"))
        out["rules"]["R5"][cell] = {"label": lab, "spread_pp": per_arm}
        if lab == "SEED_SPREAD_FAILS":
            for c, s in list(live):
                if c == cell:
                    out["unresolved"].append({"cell": c, "seed": s,
                                              "why": f"R5 SEED_SPREAD_FAILS on {cell}: {per_arm}"})
    live = [p for p in live if p not in {(u["cell"], u["seed"]) for u in out["unresolved"]}]
```
그리고 `:231-243`의 기존 R5 블록을 **삭제**한다(중복 계산 금지).

**검증**: 이 처방을 적용한 복사본에서 `--selftest`는 **계속 `SELFTEST_OK 8 reachability`** 이고,
`MODE=r5`는 `R2.a_r4 = UNRESOLVED` + `UNRESOLVED: [R5 SEED_SPREAD_FAILS …]`로 닫힌다(§6-3).

---

### H-F3. **P3가 발화할 수 없다 — abort 경로 없음 · 비교 키가 타임스탬프 · mamba 값은 비교에 들어오지도 않음**

**등록**: rev3 §7 P3 — *"두 arm의 `max_mamba_cache_size` / `max_total_num_tokens` 불일치 ⇒ **abort**."*

**구현**: `e2_sticky.sbatch:262-263`(수집) + `:355-360`(판정)
```bash
awk '{print $3, $4, $5}' "$OUT/BANNERS.txt" | sort -u | tee "$OUT/P3_BANNER_UNIQUE.txt"
if [ "$(sort -u "$OUT/P3_BANNER_UNIQUE.txt" | wc -l)" -gt 2 ]; then
  echo "P3_WARNING banners are not identical across arms -- see BANNERS.txt"
fi
```
**세 겹으로 죽어 있다**:

1. **abort가 없다.** `echo`뿐이고, 게다가 **20 boot를 전부 태운 뒤**(`:354` 이후)에 실행된다.
   등록된 실패닫힘이 구현에 존재하지 않는다.
2. **비교 키에 boot별 타임스탬프가 들어간다.** `sed "s/^/  banner $NAME /"` 뒤 필드는
   `$1=banner $2=NAME $3="[2026-09-14" $4="22:17:46]" $5=<본문 첫 토큰>`이다. **실제 아카이브
   로그 5개로 재현**:
   ```
   [2026-09-14 22:08:13] Mamba
   [2026-09-14 22:08:22] max_total_num_tokens=2722025,
   … (5 boot -> unique_count=10)
   ```
   ⇒ 20 boot이면 40줄, `-gt 2`는 **확률 1로 참**. **배너가 완전히 동일해도 항상 경고한다**
   (실제로 그 5 boot는 `max_total_num_tokens=2722025`가 **전부 같았다**).
3. **`max_mamba_cache_size`의 값이 비교 필드에 없다.** mamba 줄에서 값은 `$10`(`48,`)이고
   `$5`는 `"Mamba"`라는 고정 단어다. 즉 **등록된 두 양 중 하나는 아예 비교되지 않는다.**

**리터럴 수리** — 타임스탬프를 버리고 두 값만 뽑아 arm 간 동일성을 **abort로** 판정한다.
`:262-263`을 값 추출로 바꾸고
```bash
    MMB=$(grep -o "max_mamba_cache_size: [0-9]*" "$SRVLOG" | tail -1 | tr -d ' ')
    MTT=$(grep -o "max_total_num_tokens=[0-9]*" "$SRVLOG" | tail -1)
    echo "$NAME ${MMB:-NONE} ${MTT:-NONE}" | tee -a "$OUT/BANNERS.txt"
    echo "${MMB:-NONE} ${MTT:-NONE}" >> "$OUT/P3_KEYS.txt"
    if [ "$(sort -u "$OUT/P3_KEYS.txt" | wc -l)" -gt 1 ]; then
      echo "ABORT_P3 binding resources differ across boots -- see BANNERS.txt" \
        | tee -a "$OUT/PREFLIGHT_FAILURES.txt"; kill "$PID" 2>/dev/null; exit 8
    fi
```
로 두고, `:355-360`의 사후 블록은 요약 인쇄로만 남긴다. (아카이브 실측에서 이 키는 11/11 셀
동일 `max_mamba_cache_size:48 max_total_num_tokens=2722025`이므로 정상 실행에서는 발화하지 않는다.)

---

## 3. 차단 (8건 — 死因 아님 · 전부 GPU 0 · 사전등록 변경 불요)

### H-B1. `ABORT_PROBE_CLOCK_MISMATCH`가 기록만 되고 아무도 읽지 않는다 (fail-closed가 fail-open이다)
`e2_realized_mix.py:113-122`가 `out["ABORT_PROBE_CLOCK_MISMATCH"]`를 **딕셔너리에 넣고 끝난다**.
`e2_label.py`는 이 키를 읽지 않고, sbatch도 grep하지 않는다. 시계 원점이 어긋나 `t_probe_end`가
`lo`보다 **작아지면** 필터가 **아무것도 지우지 않고**(A3 무효화) 라벨은 그대로 나온다.
★**단 나는 이것을 발화시키지 못했다**(§5-5: 두 인터프리터 모두 `perf_counter ≡ monotonic`).
**수리**: `e2_label.py:95` 부근의 `sm_index_mismatch` 검사와 같은 자리에
```python
            if rec.get("ABORT_PROBE_CLOCK_MISMATCH"):
                out["unresolved"].append({"cell": cell, "seed": seed,
                                          "why": f"{arm}: {rec['ABORT_PROBE_CLOCK_MISMATCH']}"})
                break
```

### H-B2. `P8_arm_label_mismatch`도 아무도 읽지 않고, 게다가 **항등식**이다 (교훈 9)
`e2_realized_mix.py:336-338`이 파일명과 `--arm`을 비교하는데 **둘 다 sbatch의 같은 `$ARM`에서
나온다**(`:184` `NAME="${CELL}_${ARM}_${CSEED}"`, `:209` `tel_${NAME}.jsonl`, `:340` `--arm "$ARM"`).
구성상 불일치가 **불가능**하고, 불일치해도 abort하지 않는다(등록 P8은 "abort").
**수리**: 등록 P8이 실제로 사는 비교는 **텔레메트리 내용 ↔ arm**이다. `PDMUX_RUN_ID`/
`PDMUX_WORKLOAD_ID`가 레코드에 실리므로 그것을 arm과 대조하고, H-B1과 같은 자리에서
`UNRESOLVED`로 전파하라. 그렇게 못 하면 **P8을 등록 게이트 목록에서 내리고 "구성상 항등식"으로
공시**하라 — 지금 상태는 "게이트가 있다"는 인상만 준다.

### H-B3. **Q4(`split_transition`)가 P13 프로브 필터를 통과하지 않는다 — ADDENDUM A3 문자 위반**
A3: *"§5의 다섯 추정량과 **Q3·Q4**는 `timestamp_monotonic_s > t_probe_end`인 레코드만 쓴다."*
`e2_realized_mix.py:86-90`
```python
            if rec.get("event") == "split_transition":
                split_transitions += 1          # <-- 여기서 이미 센다
            if not _is_snapshot(rec):
                continue
            if probe_end is not None and rec["timestamp_monotonic_s"] <= probe_end:
                continue                        # <-- 필터는 이 아래에만 적용된다
```
Q3(`index_hist_busy`)와 `sm_index_mismatch`는 올바르게 필터된다. **Q4만 새어 나간다.**
★**측정된 크기는 0이다**: 전례 job 872800에서 프로브 창 안의 `split_transition`은
`stk0` **0/61**, `stk1` **0/0**(순차·1 in-flight 프로브는 분할 전환을 만들지 않는다).
또 프로브는 `a_r4/4386`의 **OFF·ON 양쪽**에 붙으므로 R3(iii)의 짝 내 비교는 균형이다.
⇒ **라벨을 못 움직였다.** 그래도 등록 문자 위반이므로 수리한다.
**수리**: `split_transitions += 1` 앞에
```python
                if probe_end is None or rec.get("timestamp_monotonic_s", float("inf")) > probe_end:
                    split_transitions += 1
```

### H-B4. `sm_index_mismatch`가 등록 **ABORT** 대신 짝 `UNRESOLVED`로 구현돼 있다
rev3 §5: *"`(prefill_sms, decode_sms)`도 기록하고 **불일치 시 ABORT**"*. 구현은
`e2_label.py:95-98`의 짝 `UNRESOLVED`다. 방향은 **관대한 쪽**(캠페인이 멈추지 않음)이다.
아카이브 실측에서 이 가드는 건강하다(11셀 전부 `sm_index_mismatch=0`, `idx2 ↔ (64,44)` 19,529 +
505 + 48 레코드 전수 일치, `prefill_sms`/`decode_sms` 결측 0). **수리**: 追記 한 줄로
"짝 UNRESOLVED로 완화"를 등록하거나, 코드를 ABORT로 올려라. **둘 중 하나는 해야 한다.**

### H-B5. **E-pact 밴드(A6)가 런타임 판정 경로에 없다**
`E_PACT_ABORT_BELOW = 95.0`(`e2_realized_mix.py:48`)은 **셀프테스트에서만** 쓰인다(`:293-294`).
A6는 *"E-pact < 95.0% ⇒ `ABORT_TELEMETRY_INCONSISTENT`. 95.0 이상 100 미만이면 ABORT하지 않고
위반 표본 수/분모를 병기한 뒤 그 seed 짝을 **UNRESOLVED**"* 라고 등록한다 — **실행 셀에 대한
조항**이다. 현재 실행 셀의 E-pact는 계산·기록만 되고 아무 분기도 만들지 않는다.
**수리**: `e2_label.py`의 `UNRESOLVED` 장부에
```python
            ep = rec.get("e_pact") or {}
            if ep.get("den"):
                if ep["pct"] < 95.0:
                    out["unresolved"].append({"cell": cell, "seed": seed,
                        "why": f"{arm}: ABORT_TELEMETRY_INCONSISTENT e_pact {ep['num']}/{ep['den']} = {ep['pct']:.2f}% < 95.0"})
                    break
                if ep["pct"] < 100.0:
                    out["unresolved"].append({"cell": cell, "seed": seed,
                        "why": f"{arm}: e_pact {ep['num']}/{ep['den']} = {ep['pct']:.2f}% in [95,100)"})
                    break
```
(A6가 "<95 ⇒ ABORT"를 요구하므로, 캠페인 abort로 올릴지 짝 UNRESOLVED로 둘지는 追記로 못박아라 —
지금은 **어느 쪽도 아니다**.)

### H-B6. 판정 경로 파일 안의 **주석 허위 3건** (교훈 80 — 이 트랙 3·4·5번째)
- **(a)** `e2_sticky.sbatch:61-62` *"`PDMUX_STICKY_PARTITION` … is **deliberately absent from this
  list**"* — `:66`에 **들어 있다**. 동작은 안전하지만(unset이 루프 밖), 판정서 §9-5 항목 9가
  바로 이 줄을 grep하라고 지시한 자리다. **수리**: 주석을 *"listed here as a defence against an
  inherited value; the per-boot setting at :215 is what决定s the arm"* 로 바꾸거나 `:66`에서 뺀다.
- **(b)** `e2_realized_mix.py:26` *"E-iter -> `VERDICT_e2_rules_2026-09-15.md` sec 2 B3"* — 그
  문서(`:151`)는 **1자리**(`4.5 · 8.1 · 8.8 · 8.4 · 45.6 · 92.3 · 92.1`)뿐이고, 셀프테스트가
  강제하는 **2자리**(`8.76 · 92.29 …`)는 `VERDICT_e2_rules_rev2_2026-09-15.md:191-194,225`와
  `VERDICT_e2_rules_rev3_2026-09-15.md:594`에 있다. **출처는 여전히 외부이므로 F2′는 되살아나지
  않는다**(§5-1). **수리**: 두 문서를 함께 적고 "1자리는 rev1, 2자리는 rev2/rev3" 를 명시하라.
- **(c)** `e2_sticky.sbatch:294` `ids.append((rec.get("meta_info") or {}).get("output_token_logprobs"))`
  — 프로브가 `return_logprob`을 요청하지 않고 기본값이 `False`(`io_struct.py:364-365`)이므로
  `token_ids`는 **항상 `[null]×6`** 이다. A7이 "토큰 ID는 기록만"이라고 했는데 **기록되는 것이
  없다**. 판정에 안 쓰이므로 무해하지만 아티팩트가 거짓을 말한다. **수리**: 필드를 지우거나
  `{"return_logprob": True, "top_logprobs_num": 0}`을 프로브 본문에 추가하라 — 단 **후자는 프로브의
  전례 동일성(A7)을 깨므로**, 나는 **필드 삭제**를 권한다.

### H-B7. P9(`errors`)가 필드 부재 시 **fail-open**
`e2_realized_mix.py:214` `errs = rec.get("errors", [])` ⇒ 필드가 없으면 `n_err = 0`으로 **통과**.
실측(합성 레코드 3종): `errors ABSENT -> P9_errors_zero=True, all_pass=True`.
같은 `--output-details` 플래그에 매달린 `P11b_output_len_exact`는 반대로 **fail-closed**다
(`out_lens=[]` ⇒ `bool([])` False ⇒ 전 셀 UNRESOLVED). 즉 **한 플래그가 빠지면 P9는 열리고
P11b는 닫힌다** — 비일관. **수리**:
```python
    if "errors" not in rec:
        checks_absent = True   # -> P9_errors_zero = False
```
또는 `errs = rec["errors"]`로 두어 KeyError로 죽게 하라(그러면 `cell_*.json`이 안 생겨
"missing boot"로 fail-closed된다).

### H-B8. 셀프테스트의 `ARCHIVE`가 `__file__` 상대 경로다 — 복사하면 **조용히 다른 이유로 실패**
`e2_realized_mix.py:251-253`. 이것 자체는 정상 실행에서 무해하지만, **감사·재현이 파일을 옮기는
순간 7개 셀이 전부 "archive telemetry missing"으로 떨어지고 `SELFTEST_FAILED`가 뜬다** — 즉
**"규약이 틀렸다"와 "경로가 틀렸다"가 같은 출력**을 낸다. 나 자신이 이 함정에 빠졌다(§6-2).
**수리**: `--archive` 인자를 추가하고 기본값만 현재 상대 경로로 두라. 그리고 셀프테스트가
아카이브 부재를 **`SELFTEST_FAILED`가 아니라 별도 종료코드/문구**로 구분하게 하라.

---

## 4. 신규 등록 caveat — **E2C-22 … E2C-28** (결과 문서·정본이 **문자 그대로** 승계할 것)

> **E2C-22 (필수 병기 — R2에는 최소 효과크기가 없다)** — *"`LAMBDA_MOVES_DOWN`/`LAMBDA_MOVES_UP`은
> **부호 진술**이지 효과크기 진술이 아니다."* 감사자 실측(규약 = `e2_label.paired_bootstrap`,
> `B=100000`, `rng(20260915)`, 2.5/97.5 분위, 908623 `a_r4` OFF의 seed 간 산포 **0.018445 req/s**를
> 잡음 모형으로): 효과 **−0.010 req/s = OFF achieved의 0.33%** 에서 이미 `LAMBDA_MOVES_DOWN`이
> 나온다(CI `[-0.01576, -0.00308]`). 이는 `CLAUDE.md` 방법론 게이트 3의 **"3% 미만 차이는 headline이
> 아니다"** 보다 **한 자릿수 아래**다. ⇒ **(a)** `LAMBDA_MOVES_DOWN`을 효과크기 주장으로 쓰지 말 것
> **(b)** 어느 분지든 **`d̄`와 OFF 수준값에 대한 백분율을 반드시 병기**할 것 **(c)** 3% 미만이면
> headline 금지.

> **E2C-23 (필수 병기 — E2C-17의 "256" 정밀화)** — *"256은 **인덱스 튜플 수**이지 CI 끝점의
> 해상도가 아니다."* n=4의 서로 다른 **평균값**은 최대 `C(2n-1,n) = 35`개이며, 현실적인 차분
> 벡터에서 감사자 실측 **21 / 23 / 26 / 27개**였다. `distinct_resamples`(`e2_label.py:82`)가
> 보고하는 256은 확률 원자 `1/256 = 0.39%`를 가리킨다. **CI 끝점의 계단 수를 256으로 읽지 말 것.**

> **E2C-24 (병기 — 부트스트랩 RNG는 무해하지만 두 셀은 독립이 아니다)** — 감사자 실측: 하네스의
> `B=100000` CI는 **256튜플 전수 열거의 정확 CI와 소수 6자리까지 일치**(4개 시험 벡터 전부)
> ⇒ 몬테카를로 오차에 의한 라벨 위험 **0**. **단** `default_rng(20260915)`가 셀마다 재생성되므로
> `a_r4`와 `a_r2`의 CI는 **같은 리샘플 인덱스 행렬**로 계산된다. ⇒ 두 셀의 R2 라벨을
> **독립적인 두 확증으로 읽지 말 것.**

> **E2C-25 (필수 병기 — 다섯 추정량의 모집단은 A3 필터에 의존한다)** — *"P13 프로브 트래픽은
> 등록 필터 `phase != "startup"`으로 제거되지 않는다."* 감사자 실측(전례 job 872800 텔레메트리,
> 규약 = `event=="runtime_snapshot" ∧ phase!="startup" ∧ decode_running_batch_size>0`):
> 프로브 창이 그 파일 decode-busy 표본의 **8.4%(13/154, sticky=0)** · **9.4%(13/139, sticky=1)** 를
> 차지한다 — **R3(ii)의 5.0% 자보다 크다**. ⇒ 다섯 추정량 값을 인용할 때마다
> **"`t_probe_end` 배제 적용"** 을 함께 적어라. 배제가 빠진 산출은 다른 추정량이다.

> **E2C-26 (병기 — 이 캠페인의 아티팩트는 `results/<campaign>/`에 갇혀 있지 않다)** —
> `e2_sticky.sbatch:316-321`의 warm-up은 `--output-file`을 주지 않으므로 `bench_serving.py:1613`이
> CWD(= `$ROOT`, 저장소 루트)에 `sglang_<MMDD>_8_<in>_<out>.jsonl`을 **append**한다(20 boot).
> 전례 확인: λ0가 남긴 `sglang_0914_8_256_512.jsonl` · `sglang_0914_8_8192_64.jsonl`이 루트에 실재.
> ⇒ P5가 기록하는 `uncommitted` 수는 **제출 시점 값**이며 실행 중 늘어난다. 재현 시 이 파일들을
> 결과의 일부로 세지 말 것.

> **E2C-27 (병기 — §5-3 기대값 중 2칸은 단일 출처다)** — 감사자가 56개 강제 리터럴을 전수
> 대조한 결과(§5-1), **54칸은 2개 이상의 외부 문서**에 있으나 **`a_r0` idx0 `1/2677`과
> `a_r4_s2` idx0 `0/557`은 `PREREG_E2_STICKY_REV3_2026-09-15.md` 한 곳에만** 있다. 그 문서는
> 이 트랙에서 **출처 허위를 두 번 낸 문서**다(rev2 §4-3 E-iter 비재현 주장 · rev3 §4-2 E-time
> 역산 7쌍). ⇒ 두 칸은 §5-3의 **가장 약한 고리**이며, 셀프테스트 통과를 "기대값이 옳다"로
> 읽지 말 것(교훈 9: 셀프테스트는 "내 산출과 하네스 산출이 같다"만 증명한다).

> **E2C-28 (병기 — UNRESOLVED 짝이 생기면 R1은 짧아진 집합 위의 라벨이다)** — rev3 §6 R1은
> *"`a_r4`·`a_r2` **전 seed**에서"* 라고 등록하는데, 어떤 짝이 `UNRESOLVED`가 되면 "전 seed"는
> 평가할 수 없다. 구현(`e2_label.py:113-122`)은 **남은 짝에 대한 연언**으로 라벨한다.
> D23은 이것을 금지하지 않는다(D23의 정본 의미는 *"미시도 셀도 EXPECT에 넣어 라벨 JSON의
> 출력공간 밖으로 사라지지 않게 하라"* 이고, 하네스는 이를 `:101-109`에서 **올바르게 이행한다**).
> ⇒ **R1을 인용할 때 `unresolved` 목록과 `per_seed` 상세를 반드시 병기**하고, 짝이 빠진 R1을
> *"전 seed에서 성립"* 으로 쓰지 말 것.

**기존 승계 전부 유효**: E2C-1 … E2C-7 · **E2C-8′** · E2C-9 · E2C-10 · E2C-11(E2C-19로 재정정) ·
E2C-12 · E2C-13 · E2C-14 · **E2C-15 … E2C-21** · λ0R-1…λ0R-10 · λ5C-1…8 · NPC-I(shape A 반증) ·
N-7·N-8·N-9·N-11 · 게이트 #13/#16 "닫았다" 금지 · C2 인용정지 (a)(b) · HE0 · layer-type 死 ·
정책 순위 · stake #1 구조 판정.

**이 회차가 바꾸지 않는 것**: 새 성능 판정 **0건** · Claim D/E 등급 **불변** · 게이트 #6 **불변** ·
P2 블로커 ①②③ **불변** · GPU 지출 **0**.

---

## 5. 반증 실패 — 깨려고 했고 **깨지 못한** 13건 (공정 기록)

### 5-1. §5-3 셀프테스트는 **항등식이 아니다** (F2′ 미재발) — 56/56 외부 출처
`EXPECTED`(`e2_realized_mix.py:241-250`)의 **강제 리터럴 56칸 전수**를 6개 외부 문서에서 기계
대조했다(`busy_n` · `e_cnt` num/den · `e_time` num/den · `e_iter` 2자리 · `e_qcond` num/den ·
`idx0` num/den · `split_transition`). **"어느 외부 문서에도 없는 값" = 0건.**
분포: E-time 초 7쌍 → rev3V + ADDENDUM(2출처) · E-iter 2자리 → rev2V + rev3V + rev3P(3출처) ·
E-cnt/E-qcond num/den → rev2V + rev3V + rev3P(3출처) · idx0 2칸만 rev3P 단독(→ E2C-27).
λ0 §4-A 1·3열의 1자리 값과도 7/7 정합(`120/2677=4.48→4.5`, `48/555=8.65→8.6`, `43/250=17.20`,
`395/395=100.0` …). ⇒ **F2′는 되살아나지 않는다.**

### 5-2. 추정량 리터럴은 **변이 저항적이다** (교훈 53) — 7변이 중 6 사살, 7번째는 등가변이
동일 아카이브·절대경로 고정 + **무변이 대조군**을 함께 돌렸다.

| 변이 | 무엇을 되돌렸나 | 셀프테스트 | 증거 수치 |
|---|---|---|---|
| (대조) | 없음 | **OK 7/7** | — |
| (a) | E-iter **왼쪽 busy 게이트 제거** | **FAILED** | `a_r4` 8.76→**8.66** · `b_r3` 92.29→**90.87** |
| (b) | E-time을 **busy 부분수열**로 | **FAILED** | `a_r4` den 183.61→**217.50** · `b_r3` 292.00→**334.99** |
| (c) | `decode_batch_size`로 교체 | **FAILED** | busy **555→20,082(36배)** · E-cnt 8.65→**0.24** |
| (d) | E-time 가중 **1.0 s 캡**(등록은 무캡) | **FAILED** | `b_r3` 285.08/292.00→**199.29/206.21** |
| (e) | E-iter **오른쪽 귀속** | **FAILED** | `a_r4` 8.76→**8.57** · `b_r3` 92.29→**92.07** |
| (f) | `delta >= 0`(등록은 `> 0`) | **OK(생존)** | **등가변이** — 가중이 `delta`이므로 `delta==0`은 분자·분모에 0을 더한다. 결함 아님 |
| (g) | E-qcond의 `prefill_queue_depth>0` 제거 | **FAILED** | `a_r4` 43/250→**48/555** |

★**(c)는 ADDENDUM A2/판정서 B2″의 근거 수치를 내가 독립 재현한 것이다**(555→20,082, 8.65→0.24%).
A2는 **정당하다**(게이트 #110: 직전 판정서를 상수로 승격하지 않고 원자료로 재검증).

### 5-3. **A3 프로브 필터는 "역방향 비대칭"을 만들지 않는다** (의뢰 2순위-2 — 반증 실패)
의뢰의 의심: *"18 boot는 프로브가 없어 필터가 적용되지 않는다 ⇒ 2 boot만 다른 필터를 받는다."*
**성립하지 않는다.** 근거 3단:
1. `phase`는 **첫 요청 하나**에 뒤집힌다(`multiplexing_mixin.py:765-776` 직접 확인 — `waiting_queue`
   또는 `running_batch` 비어있지 않으면 `startup → benchmark`). ⇒ 프로브 **이전**의 유휴 레코드는
   전부 `phase=="startup"`이라 **등록 필터가 이미 제거한다.**
2. 따라서 `t_probe_end` 필터가 제거하는 것은 **정확히 프로브 창**이고, 프로브가 없는 boot에는
   **제거할 것이 애초에 없다.** ⇒ 필터 후 **20 boot 전부 모집단 = {warm-up 8 + bench}** 로 동일.
3. 실측 교차 확인: λ0 아카이브(`tel_a_r4.jsonl`, 프로브 없음)의 **첫 번째 비-startup 스냅샷이
   이미 `decode_running_batch_size=1`** 이다 — 즉 모집단은 warm-up 첫 요청에서 시작하며, sglang
   내부 서버 warm-up이 만드는 고립된 선행 클러스터는 **없다**(최대 초기 간격 0.120 s, 그 앞
   decode-busy 1건). ⇒ 프로브 boot에서 필터가 **warm-up을 함께 잘라내는 일도 없다.**
   필터가 남기는 유일한 잔여는 프로브 종료~warm-up 시작 사이의 **유휴 스냅샷**인데
   `decode_running_batch_size = 0`이라 **다섯 추정량 어디에도 가중을 싣지 않는다.**
⇒ **A3는 B3″를 실제로 닫는다.** (단 Q4 누수는 남는다 → H-B3, 측정 크기 0.)

### 5-4. **B3″는 실재했다** — 프로브 오염은 R3(ii)의 자보다 컸다
전례 job 872800 실측(규약 = decode-busy 정의 그대로): 프로브 창이 파일 decode-busy 표본의
**8.4%(13/154)** · **9.4%(13/139)**. A3가 없었다면 2 boot만 이 오염을 진다.
⇒ 직전 감사자의 B3″ 처방은 **사후 정당화가 아니라 실측으로 지지된다**(게이트 #184 적용 결과).

### 5-5. **시계 가정이 성립한다** — `perf_counter ≡ monotonic`
서버의 `timestamp_monotonic_s`는 이름과 달리 `time.perf_counter()`다
(`multiplex/telemetry.py:23` `field(default_factory=time.perf_counter)`). 프로브는
`time.monotonic()`을 쓴다. **깨질 수 있는 자리라 실측했다**:
- venv python **3.14.2**(서버·프로브 공용; sbatch `:51` activate 이후 `python3` = venv): 
  `monotonic 1239291.726488579` vs `perf_counter 1239291.726489711` ⇒ **Δ = 1.13 µs**.
- conda python **3.9.18**: Δ = 0.63 µs. 양쪽 `get_clock_info` 모두
  `implementation='clock_gettime(CLOCK_MONOTONIC)'`.
- 아카이브 `timestamp_monotonic_s` ≈ **2,896,567 s**(≈ 33.5일) ⇒ 프로세스 기준이 아니라
  **부팅 기준 시스템 전역 시계**. 
⇒ A3의 비교는 **이 노드·이 빌드에서 유효**하다. **깨지 못했다** — 단 이름과 구현이 어긋나 있고
가드가 소비되지 않으므로 H-B1을 남긴다.

### 5-6. **부트스트랩 공유 RNG는 무해하다** (의뢰 2순위-4 — 반증 실패)
`B=100000` CI vs **256튜플 전수 열거** 정확 CI, 4개 벡터 전부 **소수 6자리 일치**
(`[-0.137500,+0.075000]` / `[-0.610000,-0.562500]` / `[-0.040000,+0.015000]` /
`[-0.032500,-0.001250]`). ⇒ 셀 간 인덱스 행렬 공유가 라벨을 바꿀 여지 **0**.
(독립성 해석 주의만 E2C-24로 남긴다.)

### 5-7. **예산이 들어간다 · P14 마진 200 s가 충분하다** (게이트 #113 자기검사 포함)
- preflight 실측: `e2_realized_mix --selftest` **4.98 s** · `e2_label --selftest` **2.00 s** ·
  `test_sticky_partition` **120.36 s**(rc 0, `Ran 12 tests`, 대부분 torch/sglang import) ⇒
  합 **127.3 s** + `sync_engine_tree.sh` + module load ≪ 등록 상수 **400 s**. 여유 있음.
- P14 마진: 등록 per-boot 부대비용 **A 145.5 s / B 106.5 s** < **200 s**. ⇒ `T_cap`을 꽉 채운
  boot도 teardown까지 예산 안에 들어간다(A 여유 54.5 s / B 93.5 s).
- `b_r3` seed2가 잘리지 않는다(A4가 고친 바로 그 코너): 등록 보통 예산에서 3번째 `b_r3` boot
  직전 경과 = 400 + 2212.0 + 2304.0 + 2×393.4 = **5702.8 s** ⇒ `REMAIN = 10800 − 5702.8 =
  5097.2 s` ≫ `861 + 200 = 1061 s`. 최악 코너에서도 경과 ≈ 7629 s ⇒ `REMAIN ≈ 3171 s` ≫ 1061.
  ⇒ **A4의 리터럴은 코드(`:80,191`)와 일치하고 R4″를 소멸시키지 않는다.**
- ★자기검사: `SECONDS`는 스크립트 시작 기준이라 SLURM 경과보다 **작다**(prolog 만큼). 즉 `REMAIN`은
  미세하게 **과대**평가된다. 마진 54.5 s 안에 들어갈 크기이나, 마진을 220 s로 올리면 공짜로
  닫힌다 — **차단으로 올리지는 않았다**(크기를 못 재서 "찜찜하니 차단"이 되기 때문).

### 5-8. `break 2`가 두 루프를 모두 빠져나가고 **사후 라벨링이 계속 실행된다**
실제 bash로 동일 중첩을 재현: 3회 실행 후 `break 2` ⇒ 바깥 루프도 종료, 포스트루프 코드 도달.
⇒ P14·P6의 `break 2`(`:197`, `:237`)는 의도대로 동작하고, 미시도 짝은 `cell_*.json` 부재 ⇒
`e2_label.py:93` `missing OFF boot`로 `UNRESOLVED`가 된다(실행 확인:
`UNRESOLVED: [{'cell': 'b_r3', 'seed': '4162', 'why': 'missing OFF boot'}]`).

### 5-9. **D23은 닫혀 있다**
정본 D23 = *"`break`·벽시간 절단 시 **시도되지 않은 셀도 EXPECT에 들어가게** 하라 — 아니면 라벨
JSON의 출력공간 밖으로 사라진다"*. 하네스는 `EXPECT`를 **루프 이전에**(`:101-109`) 20 boot 전량으로
구성하고 `--expect`로 넘긴다(`:383`). ⇒ **이행.** (인용 위생 해석은 E2C-28.)

### 5-10. P1/P2가 실제 엔진 로그와 맞는다
엔진 문구(`multiplexing_mixin.py:450-455`): `"PD-mux sticky partition ENABLED: … (fixed target
index=%s); …"`. sbatch `:245-246`의 `grep -c "sticky partition ENABLED"` / 
`grep -o "fixed target index=[0-9]*" | tail -1 | cut -d= -f2`가 정확히 잡는다. 
`_init_sticky_partition`은 스케줄러 생성자(`:233`)에서 돌고 `PDMUX_STICKY_PARTITION ∈ {1,true,True}`
일 때만 배너를 낸다(`:389-395`) ⇒ **OFF arm에서 이 줄이 나올 경로가 없다**, 그리고 health-200보다
먼저 찍히므로 **경주 조건 없음**. 양 방향(ON에 있어야·OFF에 없어야) 모두 `:248-255`에 구현.

### 5-11. P9–P12가 **실제 아카이브 bench 레코드 5개**에서 전부 발화·통과한다
`a_r4 / a_r4_s2 / a_r2 / b_r3 / b_r3_s2` 전부 `all_pass=True`
(achieved **3.053130 / 3.071575 / 2.807916 / 0.697240 / 0.695076 req/s**).
`errors`는 이 빌드에서 **리스트**(길이 = num_prompts, 원소는 빈 문자열)이고
`e2_realized_mix.py:214-215`가 리스트/정수 양쪽을 처리한다 — **"전량 실패 400건"** 합성 레코드에서
`P9_errors_zero=False`로 정상 발화함을 확인. `total_input_tokens`도 정확히
`400×256=102400` / `200×8192=1638400`. ⇒ **"모든 셀이 UNRESOLVED가 되는" 재앙 경로는 없다.**
(부재 시 fail-open만 H-B7.)

### 5-12. `sm_index_mismatch` 가드가 **오발화하지 않는다**
아카이브 11셀 중 3셀 정밀 검사: `sm_index_mismatch=0`, `negative_gaps=0`,
`prefill_sms`/`decode_sms` 결측 **0**, 전수 대응 `idx0↔(108,0)` 19,529 · `idx4↔(0,108)` 505 ·
`idx2↔(64,44)` 48. ⇒ 가드는 건강하다(등록 ABORT ↔ 구현 UNRESOLVED 차이만 H-B4).

### 5-13. R4의 seed 선택(`b_seeds[0]`)은 라벨을 뒤집을 수 없다
`e2_label.py:191-206`이 `SEED_ORDER`(`:43`)를 써서 사전순 뒤집힘을 피한 것은 **옳다**
(`sorted()`면 `"4162" < "4386"`으로 s1/s2가 조용히 교환된다). 남은 자유도(s1만 쓰고 s2를 버림)를
끝까지 밀어도 반전이 없다: 아카이브 `b_r3` OFF E-qcond **395/395 = 100.00%**, `b_r3_s2` OFF
**392/392 = 100.00%** — 두 seed가 **동일**하고 ON 예보도 100%라 `delta ≈ 0 ≤ 0.5pp`가 양쪽에서
성립. ⇒ 반전 없음, caveat 불요.

---

## 6. 자기 적용

### 6-1. 게이트 #110 (직전 판정서를 등록 상수로 승격 금지)
직전 판정서의 수치를 **하나도 승계하지 않았다**. 독립 재산출·재확인한 것:
B2″의 `555 → 20,082` · `8.65 → 0.24%`(변이 (c)로) · A8의 *"`:778-800`은 force mode 조항"*
(원문 직독: `CAVEAT for consumers: **in force mode** the emitted population is deliberately
OVER-sampled …` — **A8이 옳다**) · A1의 E-time 7쌍(56리터럴 전수 대조) · A2의 필드 리터럴 ·
A3의 phase 뒤집힘 시점(`:765-776` 직독) · A7의 전례 동일성(`sticky_smoke.sbatch:115-133` 바이트 대조) ·
E2C-17의 256 구조(전수 열거) · 예산 상수(실측 타이밍) · seed 리터럴 순서.
**결과**: 직전 판정서의 지적 중 **내가 독립 재검증에 실패한 것은 없다**. 반대로 직전 판정서와
rev3가 **둘 다 놓친 것 3건**을 냈다 — H-F1(CAPPED 미전파) · H-F2(R5 미전파) · H-F3(P3 발화 불능).
이 셋은 **규칙층에서는 보이지 않는다**(등록 문서에는 올바로 적혀 있다). 2단 감사가 존재하는 이유가
정확히 이것이다(교훈 34).

### 6-2. ★**감사 도구 자신이 항등식일 뻔했다** (교훈 9 — 자기 고지)
1차 변이 실행에서 7개 변이가 **전부 `SELFTEST_FAILED`** 를 냈고, 나는 하마터면 "7/7 사살"로
기록할 뻔했다. 실제 실패 사유는 전부 `"archive telemetry missing"` — `ARCHIVE`가
`__file__` 상대 경로(`e2_realized_mix.py:251-253`)라 `/tmp` 복사본이 아카이브를 못 찾은 것이다.
**변이와 무관한 실패였다.** 절대경로 고정 + **무변이 대조군** 추가로 다시 돌려서야 진짜 결과
(6 사살 + 1 등가변이)를 얻었다. ⇒ 이 경험을 **H-B8**로 등재했다. *"변이 시험은 반드시 무변이
대조군과 함께 돌려라 — 대조군이 통과하지 않으면 사살 수는 의미가 없다."*

### 6-3. 게이트 #113 (내 처방의 실현가능성·비용·노브 결합)
- **실제로 적용해 검증했다.** H-F1(a) + H-F2 처방을 `e2_label.py` 임시 복사본에 넣고 실행:
  `--selftest` → **`SELFTEST_OK 8 reachability`(불변)** · `MODE=capped` →
  `UNRESOLVED: [{'cell':'a_r4','seed':'4386','why':'ON: P9-P12 bench validity failed'}]` ·
  `MODE=r5` → `R2.a_r4 = UNRESOLVED` + `UNRESOLVED: [R5 SEED_SPREAD_FAILS …]` · `MODE=clean` →
  **변화 없음**(`STICKY_REALIZES_A` / `LAMBDA_WITHIN_CI` / `UNRESOLVED: []`). ⇒ 처방은 **정상
  경로를 건드리지 않고 실패 경로만 닫는다.**
- **비용**: GPU **0**. 死因 3건 + 차단 8건 전부 코드·주석 국소. **사전등록 rev4 불요**
  (H-B4·H-B5만 追記 한 줄로 "완화/강화" 중 하나를 못박으면 된다).
- **노브 결합 대조**: 내 처방 어느 것도 env · 서버 인자 · seed · `num_prompts` · telemetry 격자 ·
  추정량 규약 · §5-3 기대값을 바꾸지 않는다. **H-B6(c)만 예외로 두 선택지가 노브를 다르게 움직여서**
  (프로브에 `return_logprob`을 넣으면 A7의 전례 동일성이 깨진다) **필드 삭제 쪽을 명시적으로
  권했다**(교훈 250: 감사자 처방도 다중 노브를 움직일 수 있다).
- **내 처방이 사지 못하는 것**: (a) H-F3을 고쳐도 **P3는 "두 arm의 자원이 같았다"만 보장**하고
  achieved 차이의 귀속은 여전히 E2C-1/E2C-21이 지배한다. (b) H-F1·H-F2를 고쳐도 **검정력은
  사지 않는다** — R2는 여전히 n=4이고 R4″는 E2C-18대로 드레인 축을 못 본다. (c) H-B3을 고쳐도
  측정 크기가 0이므로 **어떤 수치도 바뀌지 않는다**.

### 6-4. 게이트 #184 (처방자 자기승인 금지)
이 하네스의 설계는 rev1–rev3 감사자들의 처방이다. 나는 그것을 **승인하지 않고 시험했다**:
A2 → 변이 (c)로 재현 · A3 → 전례 텔레메트리에서 오염률 실측(8.4/9.4%) **및** 역방향 비대칭
가설을 3단으로 반증 · A7 → 전례 sbatch와 바이트 대조 · A1 → 56리터럴 전수 출처 추적 ·
A4 → 예산 산술 재계산 · A6 → **런타임 경로에 없음을 발견(H-B5)** · A5 → 4분지 도달성 확인 ·
A8 → 엔진 주석 직독으로 확인. ⇒ **처방 8건 중 1건(A6)이 구현되지 않았음**을 찾아냈다.

### 6-5. E2C-8′를 나 자신에게 (내가 발표하는 모든 추정량 열에 규약 병기)
이 판정서가 낸 수치의 규약을 전부 명시했다: decode-busy 정의(세 필드 중 어느 것인지) ·
E-time(다음 비-startup 스냅샷까지·무캡·마지막 제외) · E-iter(좌 귀속·차분>0·좌 busy 게이트·
가중=차분) · E-qcond(`prefill_queue_depth>0` 조건부·균등) · 프로브 오염률(같은 decode-busy 규약,
분자/분모 13/154·13/139 병기) · 부트스트랩(B·rng seed·분위·정확 열거 대조) · R2 검정력
(잡음 모형과 산포 0.018445 req/s의 출처) · 예산(실측 타이밍의 정의) · P3 재현(awk 필드 번호까지).
**재현 경로**: 필터 한 벌만 사용, 시간축은 `timestamp_monotonic_s`, 쉬핑 라벨 코드 미호출,
손으로 만든 verdict 없음, 변이·합성은 전부 `/tmp` 복사본에서 실행하고 **저장소 파일은 한 글자도
고치지 않았다**(이 판정서 파일 생성 제외).

### 6-6. 등급 인플레/디플레 점검
- **인플레 방지**: 死因 3건을 caveat로 강등하지 않았다. 셋 다 **실행 출력 또는 실제 로그로
  재현**했고, 그 중 H-F2는 **등록이 `UNRESOLVED`를 요구하는 자리에서 적극적 라벨을 내는**
  직접 반전이다.
- **디플레 방지**: 13건을 **반증 실패**로 남겼다. 특히 (i) 의뢰가 직접 의심한 두 항목
  (P13 역방향 비대칭 · 공유 RNG)은 **둘 다 무해로 판정**했고, (ii) 시계·예산·D23·P1/P2·
  P9–P12·부트스트랩·R4 seed는 깨려다 실패했다. *"찜찜하니 차단"* 을 하지 않았다.
- **자기 고지**: **H-F3(P3)는 차단으로 등급할 수도 있다** — 그것만으로는 라벨을 못 뒤집기
  때문이다(순수 진단 출력). 死因으로 둔 이유는 *"등록된 abort가 구현에 존재하지 않는다"* 가
  의뢰 기준의 **"게이트가 발화하지 못한다"** 에 문자 그대로 해당하고, 게다가 **세 겹으로**
  죽어 있기 때문이다. **어느 등급이든 수리 문안은 동일하고 캠페인 설계는 바뀌지 않는다.**
  그리고 H-F1·H-F2만으로도 `NO-GO`는 성립한다.

---

## 7. 재제출 선행조건 (순서대로)

1. **死因 3건 수리**(H-F1 · H-F2 · H-F3) — §2의 리터럴 문안. GPU 0.
2. **차단 8건 수리**(H-B1 … H-B8) — §3. GPU 0. 이 중 **H-B4·H-B5는 追記 한 줄**이 필요하다
   (ABORT냐 UNRESOLVED냐를 등록에서 못박을 것).
3. **셀프테스트 재실행** — `e2_realized_mix --selftest`(7/7) · `e2_label --selftest`(8) ·
   `test_sticky_partition`(12). **셋 다 rc 0이어야 한다.**
4. ★**변이 재검증**(교훈 53 + §6-2): H-F1·H-F2 수리를 **되돌린 변이본에서 반드시 실패하는**
   회귀 검사를 `e2_label.py --selftest`에 추가하라 — 즉 (i) `bench` 키를 지운 합성 짝이
   `UNRESOLVED`가 되는지, (ii) E-cnt 산포 6pp인 합성 셀이 R1/R2를 `UNRESOLVED`로 만드는지.
   **지금의 8개 도달성 검사는 이 둘을 덮지 않는다**(둘 다 통과 상태에서 만들어졌기 때문이다).
5. **하네스층 재감사**(이 판정서 기준 — §1 10항 + §2 3건 + §3 8건 + §7-4 회귀 2건).
6. **새 범위 한정 OVERRIDE + 사용자 승인**(예산 1.803 GPU-h, 최악 2.338, `--time 03:00:00`).
7. **커밋**(P5) → 그 다음에만 `sbatch`.

---

## 8. 한 줄 요약

> **`NO-GO` — 추정량은 건강하고(56/56 외부 출처 · 7변이 중 6 사살 · 등가변이 1), 시계·예산·
> D23·프로브 필터·부트스트랩은 깨려다 실패했다. 죽은 것은 전부 "실패닫힘"이다**: P7 `CAPPED`가
> 판정 경로에 **없고**(실행 재현: `UNRESOLVED: []`인 채 R1은 4 seed로 라벨되고 R2만 n=3으로
> 떨어진다), R5 `SEED_SPREAD_FAILS`가 **아무것도 하지 않으며**(실행 재현: 등록이 `UNRESOLVED`를
> 요구하는 자리에서 `STICKY_REALIZES_A` + `LAMBDA_WITHIN_CI`), P3는 **abort 경로가 없고 비교 키에
> boot 타임스탬프가 들어가 확률 1로 경고하며 `max_mamba_cache_size` 값은 비교에 들어오지도
> 않는다**(실제 로그 재현: 5 boot → `sort -u` 10줄). 세 구멍 모두 **등록 문서는 옳게 적었고 코드가
> 안 했다** — 그래서 수리는 **코드 국소 · GPU 0 · rev4 사전등록 불요**이고, 나는 처방을 임시
> 복사본에 넣어 **두 셀프테스트가 계속 통과하면서 두 경로가 닫히는 것을 확인했다.**
> 그리고 이 회차가 사는 것은 여전히 좁다 — shape A 처치 질량의 **86–89%가 prefill 유휴**
> (E2C-21)이고, **게이트 #6 · P2 블로커 · Claim D/E는 하나도 닫히지 않는다.**
> **GPU 지출은 이 판정서 시점까지 0이다.**
