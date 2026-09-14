# 규칙층 재감사 판정서 — D-none 사전등록 **rev2** (2026-09-14)

> ★**전사 공시**: claims-auditor(read-only, 쓰기 권한 없음)가 보고 본문으로 낸 판정 전문을
> **메인 세션이 전사**했다(내용 변경 없음). **전사 충실성은 제3자 검증 불가** — 인용 시 병기하라.
> 감사자 자신도 rev1 판정서에 대해 같은 한계를 전제로 판정했고(DNA2-9), 그 과정에서
> **rev1 처방의 산술 오류를 확인**했다(DN2-2).

**감사자** claims-auditor (적대적, read-only) · **GPU 신규 지출 0**
**대상** `dnone_prereg/PREREG_DNONE_2026-09-14_rev2.md`
sha256 `11b630957a7d5bae5b8f9bd44aecd84255b6dc143466d9d6a22a899bea319157`(재계산 일치), 394줄

## 등급: `GO-with-caveats`

**死因 0건.** rev1의 DNR-1…DNR-4는 **네 건 모두 닫혔다**(각각 참/거짓 분지 도달을 원자료로 실증).
반전 시험 14 표면 중 판정을 뒤집은 것은 **0건**. 비차단 권고 **DNA2-1…DNA2-11**,
신규 인용금지/필수병기 **DN2-1…DN2-5**.

⚠️ `GO-with-caveats`는 **주장 범위가 좁아진다**는 뜻이다. 이 회차는 그래도 **H1의 기전·하네스
축·boot 간 분산을 닫지 못하며**, 최대 산출은 rev2 §0-d가 스스로 공시한 **n=1 스코프 한정 존재
문장 하나**다.
⚠️ 별개로: **rev2 자신의 §4-4(2) 제출 선행조건이 현재 거짓이다**(`check_doc_facts.py` 1 violation).
그건 감사자가 새로 세운 차단이 아니라 **문서 자신의 fail-closed 조항**이다 — 닫기 전에는 제출할
수 없다(DNA2-7).

---

## §0. 감사자가 실제로 돌린 것 (게이트 #110 — rev1 판정서 수치를 상수로 승계하지 않았다)

| 재검증 | 결과 |
|---|---|
| §0-c 핀 sha **13행 / 15값** 전부 재계산 | **전부 일치**(`PREREG_RERUN` 현재 `fa304128…`, 908179 시점 `1e421391…`는 `job_908179/provenance.txt:47`에서 직접 확인) |
| 하네스 sha 이동 | 현재 `ab55c07c…` / `git show 83d8cb9:…` = `f39b167b…` = `job_908179/provenance.txt:11` 기록값 ✓ |
| 8 boot × `#new-token` 전열 + OOM 계수 | 아래 §1 표 — rev2 §4-1 표에 **전사 오류 2행 발견**(DNA2-2) |
| `report_prefill_stats` 호출 위치 | `scheduler_output_processor_mixin.py:328`, `process_batch_result_prefill` **말미**(forward 후) ✓ |
| 907959 TD1 OOM 위치 | 로그 2592행 / 총 2594행, OOM 뒤 `#new-token` **0줄** ⇒ OOM = boot 사망 ✓ |
| OOM 서명 2/2 동일 | `Tried to allocate 198.00 MiB` / `Of the allocated memory 77.25 GiB` 문자 일치 ✓ |
| 하네스 가드 블록 **직접 실행**(7 조합) | `r2c_guard=none guard_applied=DN-boots-only dn_boots=1 order=[L TD L TD DN]` + `mem_telemetry=1 worker_grad_guard=none` — rev2 §4-3(12)와 **문자 일치** |
| 변이 (a)/(b) | **격리 미러에서 재현**: **7 실패 / 4 실패**, 단정문 문자 일치 |
| 채점기 전 `open(` + glob/listdir/walk/scandir | `:135 :190 :221 :366 :373 :502`, 디렉터리 스캔 **0건** ⇒ 설계 C 전제 참 |
| `x1_cross_job_compare.py` | `LABELS` 하드코딩, glob 0건 ⇒ DN 불가시 |
| `srv_*.log` glob | `:664`, `INSTR=1` 전용·`instrument/` 전용 ⇒ 등록 `INSTR=0`에서 미실행 |
| `diag_boots.txt`·`########## boot=` 소비자 | 생산자+dnone 테스트뿐, 채점 소비자 **0건** |
| 채점기 호출 위치 | `:673`, boot 루프 뒤. `set -uo pipefail`(**`set -e` 없음**), boot 실패 `continue`, 클라 실패 `\|\| echo` ⇒ **DN OOM이 `verdict.txt`를 막지 않는다** |
| G 게이트 도달성(코드) | `_r2_record_worker_guard`가 **가드 안**(`multiplexing_mixin.py:722-723`), `@torch.inference_mode()`는 thread-local(`:700-703`) ⇒ DN worker = grad ON / inference OFF |
| G 음성대조 실측(908179) | TD1/TD2 `*_worker_grad_enabled`: `true` **0** / `false` **8569·8263** / `null` **539·525** ✓ |
| DNR-V5 실측 | 907959 TD1·TD2 `prefill_active_batch_size` ∈ {0,1,4,8}, **`14`는 0건**, 최대 **8** ✓ |
| §5-4 peak 실측 | 908179 `gpu_mem_peak_allocated_b`: L2=TD1=TD2=**73,342,324,736 B**(차 **0 B**), L1=73,443,758,080 B ✓ |
| DNA-8 cudagraph 실측 | prefill `cuda graph: False` L1 **40/40**, L2 41/41, TD1 **41/41**, TD2 41/41; decode `True` **2589/2589** ✓ |
| `e5_family_probe.py` 재실행 | `VERIFIED_B3` **12/12 · 12/12 · 12/12**, `RA4_2_proposed` **4/12**(GNU `grep -E`), `REGISTERED_rev4` 누락 **3종**, `Y4(torch.OutOfMemoryError)` **3엔진 미발화** ✓ |
| E5 정규식 30개 boot 로그 전수 | **0 hit**(pre-repair 907959 4 boot 포함) |
| `CARRYFORWARD` 기계적 재계수 | 인용금지 **49** / 필수병기 **34** / 절차의무 **15** = **98**; 절차의무 15 id가 rev2 §8-1 열거와 **집합 일치**(순서까지) |
| 이 트랙 CPU 회귀 | **98/98 OK** = dnone **36** + instrument **35** + ctx **16** + scope_guard **11** |
| 전체 CPU 스위트 | **`Ran 696 tests` / `FAILED (failures=4)`** — 전부 `test_lambda0_prereg`. rev1의 유일 실패 `test_no_escapes`는 **지금 통과** |
| `sacct` 6 job | 433+574+554+1281+549+596 = **3,987 s = 1.107500 GPU-h**(등록 0.955000 / 등록 밖 0.152500) ✓ |
| 예산 산술 | `153+111·5 = 708 s = 0.196667 h ≈ 0.197` ✓ / `108 s = 0.030 h` ✓ / boot당 최악 `96×5+900+20 = 1,400` ✓ / `--time` 9,000 s ✓ / `#SBATCH --comment` 존재 ✓ |
| §4-4(3) 리터럴 명령 | 9개 `R2C_*` 전부 존재. 기본값 실측 `MODEL=Zyphra/Zamba2-2.7B`(:198) `ATTN_BACKEND=triton`(:212) `INSTRUMENT=0`(:222) `TFP=1`(:217) `MEMTEL=1`(:233) ⇒ 생략된 3개가 등록 튜플과 일치 ⇒ **명령은 옳다** |
| 앵커 이동 4건 + glob | `:294` ✓ / `:608-619`(DN 분지 `:611-618`, 가드 재수출 `:618`) ✓ / `:330`·`:369` ✓ / `:397-408` ✓ / glob `:364-368`, 본체 `:367` ✓ |
| §1-1 엔진 인용 | `multiplexing_mixin.py:119-123`(`"none": nullcontext` = `:122`)·`:127-144` ✓, src 사본이 엔진 트리와 바이트 동일, **src 무변경** ✓ |
| `citation_stops` | 16 rule, 98건 중 **0건 등재** ✓; rev2·인벤토리 0 위반 |
| `check_line_citations.py --check --all` | 93 compared, **0 violation** |

★**부작용 공시(게이트 #240)**: `presubmit.py` **1회** 실행 ⇒
`m4r_confinement/reachability_verdict.json`·`tc1_model_attrib/reach_verdict_rev3_A.json` **재작성**
(mtime 이동), **sha256 불변**, `git diff HEAD --`·`git status --porcelain` 두 경로 **빈 출력**.
변이 검사는 **프로젝트 밖 격리 미러**에서 실행 후 삭제. 감사 종료 시 하네스 `ab55c07c…` ·
rev2 `11b6309…` · dnone 테스트 `4209e572…` **전부 불변**.
★**미러 한계**: git 이력 의존 테스트 1개 `skipped`(실질 = `83d8cb9` blob sha가 908179 provenance
기록값과 같음은 별도 직접 확인).

---

## §1. 1순위 — rev1 死因 4건의 폐쇄 실증

### DNR-1 (지배적) — **닫혔다**. 새 V를 8 boot 전부에 먹인 결과

| boot | `#new-token` 실제 전열(꼬리) | OOM | max | (i) 사다리 | (ii) max=6245 | **rev2 V 라벨** |
|---|---|---|---|---|---|---|
| 907959 TD1 | `… 2310 55 368 1468 3241 6245` | **1** | 6245 | **참** | **참** | **→ F-n1 (`RECOVERED-*`)** |
| 907959 TD2 | `… 2310 55 368 1468 3241 6245` | **1** | 6245 | **참** | **참** | **→ F-n1 (`RECOVERED-*`)** |
| 907959 L1 | `… 55 368 1468 3241 6245 10125 3491` | 0 | 10125 | 참 | – | `NOT-RECOVERED` |
| 907959 L2 | `… 55 368 1468 3241 **7345** 12516` | 0 | 12516 | **거짓** | – | `NOT-RECOVERED` |
| 908179 L1 | `… 55 368 **2313 5414 10630 6213**` | 0 | 10630 | **거짓** | – | `NOT-RECOVERED` |
| 908179 L2 | `… 3241 6245 10125 3491` | 0 | 10125 | 참 | – | `NOT-RECOVERED` |
| 908179 TD1 | `… 3241 6245 10125 3491` | 0 | 10125 | 참 | – | `NOT-RECOVERED` |
| 908179 TD2 | `… 3241 6245 10125 3491` | 0 | 10125 | 참 | – | `NOT-RECOVERED` |

⇒ **양 분지 도달 실증**. `RECOVERED-*`는 **OOM 재현 사례(907959 TD 2/2)에서 정확히 도달**하고
건강한 6 boot은 전부 분지 A를 통해 `NOT-RECOVERED`로 간다. rev1의 V는 `P(V=TRUE|OOM) = 0/2`
(기전상 0)였다 — 그 감소함수성이 제거됐고 **수리 후 `RECOVERED` 경로는 처치 강도의 증가함수다.**
N3 해소.

### ★rev1 처방(DNR-1 (ii))의 절벽 — **메인 세션의 정정이 옳다**

```
198.00 × 1048576 / 20480 = 10137.6        (밴드 상한 10137을 0.6 초과)
122.0  × 1048576 / 20480 =  6246.4        (rev1이 "T ≈ 6245"라 적은 값도 6246.4)
```
`DIAGNOSIS:31`이 밴드를 만든 방식은 `T×20480 B`를 **2 MiB 올림**해 `198.00 MiB`가 되는 정수 `T`의
구간 `(10035.0, 10137.6]` → **정수 상한 `floor(10137.6)=10137`**이다. 즉 **밴드 상한 자신이 같은
역산값을 내림해 만든 수**이고 그것을 다시 멤버십 문턱으로 쓰는 것은 **순환이며 off-by-one**이다.
rev1 판정서 §2 DNR-1의 *"198.00 MiB → T = 10137.6 ⇒ 밴드 안(2/2)"* 은 **산술적으로 거짓**이고
처방을 문자대로 구현하면 **기준 사례 2/2가 탈락**한다(`floor`=통과 / `round`=탈락 = 새 자유 표면).
⇒ **rev2가 역산을 게이트에서 빼고 정수·문자 술어로 교체한 것은 옳다.** 처방자가 함께 죽은
사례(게이트 #113)의 실물이며, 이번엔 감사자 처방도 같은 검사를 받았다(§4 반전표).
교체 술어가 만든 새 자유 표면 **3개는 셋 다 실제 아티팩트에서 반전을 만들지 못했다**
(§4 표 1·2·3행) ⇒ 死因 아님, caveat(DNA2-2·DNA2-3, DN2-3).

### DNR-2 — **닫혔다**(직접 실행)
등록 튜플에서 `guard_applied=DN-boots-only`·`worker_grad_guard=none`이 **문자 그대로** 나온다.
반전: `L TD DN DN` → `dn_boots=1 order=[L TD DN DN]`(→ (13) `order=` 술어가 잡음) ·
소문자 `dn` → `dn_boots=0 guard_applied=no (…)`(→ (12)(13) 동시 거짓) ·
`R2C_GUARD` 미설정/`None`/6 boot → `exit 2`.

### DNR-3 — **닫혔다**. 잔존 허위귀속 통로 없음
`DESIGN-REFUTED` 폐기 + 기록 항목 강등. (b)의 "구체적 기전" 문구는 **재량 게이트가 아니다** —
기본값이 *금지*이고 라벨 자체가 사라졌다. 907959(verdict `FAIL`, DN boot 0개)에 rev2 규칙을
먹이면 나오는 것은 **실패 술어 전사**뿐이다.

### DNR-4 — **닫혔다**. (a) 15 = 15 집합 일치, (b) 자기 위반 0건
(a) 절차의무 = `NPC-A · NPC-H · NPC-J · RRC-13 · RA3-6 · RA3-10 · RA3-12 · RA4-2 · RA4-6 ·
RA4-8 · RA4-9 · RA4-10 · RA4-11 · RR-14 · RR-19` = **15**, rev2 §8-1과 **집합 일치**.
(b) 전수 대조 자기 위반 **0건**: **NPC-H** ✓(이 트랙 98 한정 + 제출시 실측 의무 신설) ·
**RR-14/RA4-6** — 인벤토리 §12가 "항목 수 대조"를 **명시 허용**하고 sha로 핀했으므로 위반 아님
(★감사자 자기검사: 98행 재전사 요구는 처방 인플레[게이트 #113]이며 sha 핀 전수 열거가 더 강하고
싸다) · **RA3-6** ✓ · **RA4-2** 처방 거부이나 **더 강한 대안 + 실측 근거**(4/12 vs 12/12)를
§8-1에 "교체"로 명기 ⇒ 위반 아님 · **RA3-10** 부분 이행(`BOOT-FAILED`∧`NO-RISK-BATCH` 우선순위
미등록 → DNA2-4, 양쪽 다 서술 문장 0이라 판정 불변) · 나머지 8건 위반 없음.

### DNA-1…DNA-10 — 전수 반영 확인
10건 **전부 본문에 실재**(표에만 있고 본문에 없는 항목 0건). 단 **DNA-1 행의 계수가 틀렸다** —
§0-a는 "7개 `R2C_*`", 본문 §4-4(3)은 **9개**(DNA2-6).

---

## §2. 2순위 — rev2가 새로 만든 자유 표면

### (1) 출력공간 전수 덮기 · 겹침 우선순위
- `NO_VERDICT_SCOPE` / `NO_RUN` / `NO_KNOB` / `UNREALIZED-{BOOT-FAILED,NO-RISK-BATCH,OTHER-ARRIVAL}` /
  `OTHER-BATCH` / `NOT-RECOVERED` / `RECOVERED-{STRICT,WEAK}` 로 전 사건이 라우팅된다.
- **빠진 사건 1건**: `OOM 0 ∧ max ≥ 10036 ∧ 비-OOM traceback ≥1` → 등록 규칙은 **`NOT-RECOVERED`**
  로 보낸다(교훈 21 부호 반대 통로). **감사자 반전 시험 실패**: 30 boot 전수에서 그 3중 결합
  **0건**이며 전례 907032 TD1·TD2는 `max=none`이라 분지 A가 거짓 ⇒ `UNREALIZED`로 정확히 빠진다.
  ⇒ **死因 아님, caveat**(DNA2-1 + 필수병기 DN2-5).
- **`NO_RUN` 겹침은 도달 불가**: `set -e` 없음 · boot 실패 `continue` · 클라 실패 `|| echo` ·
  채점기 `:673`. 순수 구조적 최악 5×1,400 = **7,000 s < 9,000 s**.
- `NO_KNOB` 우선순위는 §5 제1문(V∧G∧SCOPE 전부 참일 때만 F-n1)이 등록 ✓.

### (2) (i) 사다리는 도착열 의존 — 실질 검정력
**NemotronH 기판 실측 사다리 6/8.** 반례는 rev2가 지목한 1개가 아니라 **2개**
(908179 L1 `55 368 2313 5414 10630 6213` · **907959 L2** `55 368 1468 3241 7345 12516`, rev2 미지목).
⇒ DN이 OOM해도 (i)가 거짓일 사전확률 ≈ **2/8 ≈ 25%**, 그때 라벨은 `UNREALIZED-OTHER-ARRIVAL`
(= 서술 문장 0). §0-d는 방향은 보수적이나 **이 ~25%를 수치로 공시하지 않는다** ⇒ caveat
(DNA2-3, DN2-3).

### (3) §8-2 E5 정의역 정정 — **결론은 경험적으로 지지되나 근거가 방향이 거꾸로다**
① 엔진은 worker 가드와 무관하게 부팅 시 `torch.inference_mode()`에 들어간다
(`model_runner.py:2107` `_flashinfer_autotune` · `:2374` `_dummy_run`).
② `multiplexing_mixin.py:700-705` docstring: *"`@torch.inference_mode()` is thread-local, so without
this line the model forward … runs with **autograd ON**"*.
③ 같은 파일 `:96-105`: E5 하드 에러는 *"a consumer that is **NOT inside inference mode**"* 를 요구 —
**DN worker가 바로 그 소비자다.**
⇒ DN arm은 E5에 **면역이 아니라 더 노출된 arm**이다. 지지되는 것은 경험적 사실뿐: 30 boot 전수
**0 hit**, 그중 **pre-repair true-dual 2 boot(907959 TD1·TD2 = DN과 같은 가드 의미론)**에서도
미발화(n=2). 발화해도 라우팅이 보수적이라 **死因 아님** ⇒ caveat + 인용금지 DN2-4.

### (4) "발화하면 회차 중단"의 실행가능성
**부분적으로만 실행 가능**하다 — `set -e`가 없어 TD1에서 발화해도 boot 3·4·5는 그대로 돈다.
실행 가능한 형태는 **사후 규칙**: ①채점 arm 가드의 2번째 축 이동 **자동 적용 금지**(rev2 등록 ✓)
②그 회차 결과를 `NO_VERDICT`로 보고 + 별도 사전등록. 문안을 "중단" → **"사후 미채점 + 별도
사전등록"**으로 교체(DNA2-11).

---

## §3. 3순위 — 사실 재검증 결과
- **sha**: §0-c 표 **13행 / 15값 전부 일치**(작성자 "14행"은 2값 행을 2행으로 센 것).
- **예산·장부**: 전 항목 불일치 **0**.
- **§4-4(3) 명령**: 변수명·기본값 일치, 생략 3개의 기본값(0/1/1)이 등록 튜플과 일치 ⇒ **옳다**.
- **CPU 스위트(★동시 편집 공시)**: 이 트랙 **98/98 OK**. 전체 **696 / 4 실패**(전부
  `test_lambda0_prereg`). rev1의 유일 실패 `test_no_escapes`는 **지금 통과** ⇒ rev2의 열린 질문은
  **동시 작업 산물** 쪽으로 기운다(단 "선재였다가 그 사이 고쳐졌다"는 배제 불가). λ0 트랙은
  **감사 중에도 편집**됐다(modified 6 → 7, `test_lambda0_prereg.py` mtime 17:06).
  지시에 따라 **λ0 관련 실패는 이 회차 제출 게이트 판정에 쓰지 않았다.**

---

## §4. 반전 시험 표 (요지 — 14 표면, 판정 반전 0)

1. 분지 A "split-prefill 배치" 조작화 → **반전 무효**(8/8 boot의 `#new-token` 줄이 전부
   `Prefill batch`; rev2 §4-1이 도달값을 수치로 못박아 대안 독법은 문서 자신과 모순).
2. (i) "그 순서로" 순서 부분열 ↔ 연속 부분열 → **불일치 0건**(30 boot 전수).
3. (ii) "마지막 기록 배치" ↔ "최댓값" → **라벨 불일치 0건**(907959 TD `last=max=6245`).
4. V 밴드 상한(rev1 死因) → **rev2에서 소멸**.
5. **rev1 처방 역산 문턱 → 뒤집힘**(그래서 rev2가 제거한 것이 옳다).
6. SCOPE (12) 문자열(rev1 死因) → **소멸**.
7. SCOPE (13) 개수 ↔ 불리언 → **`order=` 술어가 잡음**.
8. §4-4(2) 회귀 범위(rev1 死因) → **소멸**.
9. F-n3 `FAIL` 귀속(rev1 死因) → **소멸**.
10. `-BOOT-FAILED` ↔ `-NO-RISK-BATCH` → **주장 결과 동일**(907032 TD 2/2 동시 참).
11. `NO_RUN` ↔ `RECOVERED-*` → **도달 불가**.
12. G 게이트 양극 → **양 극 도달**.
13. 승계 98 집행 방식 → **판정량 아님**(인벤토리 §12 허용 + sha 핀).
14. presubmit 차단 건수 `2` → **제출 판정 불변**(§4-4(2)가 독립적으로 `doc_facts 0 위반` 요구 =
    fail-closed), **§4-4(1)의 사실 진술만 거짓**: 실측 **3 BLOCK**(등록 2 + `doc_facts`),
    `check_doc_facts.py` = *"document says 681, artefact says 696"*, 델타 **+15 =
    `test_lambda0_prereg.py` 65→80**.

원자료·도구는 §0과 동일. **새 자유 표면 만들지 않았다**(밴드·역산식·사다리는 핀된 DIAGNOSIS에서
전사, 순서/연속 두 독법을 **둘 다** 계산해 표에 남겼다).

---

## §5. 비차단 권고 (DNA2)

- **DNA2-1 (우선순위 최상)** — 분지 A가 **비-OOM 치명 크래시를 걸러내지 않는다**. 수리: 분지 A에
  `∧ srv_DN1.log에 'Traceback (most recent call last)' 0건` 추가 또는 `UNREALIZED-OTHER-CRASH` 신설.
  동시에 §8-2의 *"DN arm에서 E5는 구조적으로 발화 불가"* 를 *"pre-repair true-dual 2 boot에서
  미발화 — n=2 경험적. 엔진은 worker 가드와 무관하게 `model_runner.py:2107,:2374`에서
  `torch.inference_mode()`에 들어가고, `none`은 worker를 **autograd 추적 소비자**로 만들어 E5의
  전제를 오히려 만족시킨다(`multiplexing_mixin.py:96-105,:700-705`)"* 로 교체. 또
  **승계 `RRC-4`**(OOM 계수·스택 병기 의무)를 §8 본문 인용 목록에 추가(현재 누락).
- **DNA2-2** — §4-1 표 배치열 **전사 오류 2행**. 실제: 907959 L1 = `… 3241 6245 10125 3491`,
  908179 L2/TD1/TD2 = `… 3241 6245 10125 3491`. **3491은 10125 뒤다.**
- **DNA2-3** — (i) 반례는 **2개**(908179 L1 + **907959 L2**), 사다리 **6/8** ⇒ §0-d에
  *"분지 B에 들어가도 (i)가 거짓일 사전확률 ≈ 2/8"* 를 수치로 추가.
- **DNA2-4** — `-BOOT-FAILED` ∧ `-NO-RISK-BATCH` 우선순위 미등록(907032 TD 2/2 동시 성립).
  저비용: *"`BOOT_FAILURES.txt`에 DN 행이 있으면 `-BOOT-FAILED`가 우선"* 한 줄.
- **DNA2-5** — §7 "1,400 s"·§5-1 "~1,997 s"는 **DN만 최악** 혼합 모형. 순수 구조적 최악 = **7,000 s**.
  둘 다 9,000 s 아래라 `NO_RUN` 판정 불변.
- **DNA2-6** — §0-a DNA-1 행 "7개 `R2C_*`" → 본문은 **9개**.
- **DNA2-7** — **presubmit 실측 3 BLOCK**. 세 번째 `doc_facts`는 *"후속 spec 없는 살아 있는 타 트랙
  설계 결함"*이 **아니라 한 숫자로 닫히는 위생 결함**이다(`DESIGN_A1_REV2_STICKY_2026-08-25.md:382`
  "회귀 **681** PASS" ↔ 정적 카운트 **696**) ⇒ **OVERRIDE로 넘기면 안 되고 고쳐야 한다.** 진리원이
  저장소 전역 `def test_` 카운트라 동시 작업이 언제든 다시 깨뜨린다 ⇒ §4-4(1)의 "현재 2 BLOCK"을
  **제출 직전 실측**으로 바꾸고 **OVERRIDE 작성·`presubmit` 재실행·`sbatch`를 하나의 짧은 창에서**
  수행하라.
- **DNA2-8** — 전체 스위트 실측 갱신: **696 / 4 실패**(전부 λ0:
  `TestMutationHarness.test_control_is_scored_on_the_mutation_path` ·
  `TestNoStaleRules.test_plan_has_no_dead_drain_tolerance` ·
  `TestNoStaleRules.test_plan_no_longer_claims_the_engine_cap_was_reached` ·
  `TestSbatchDiscipline.test_no_phantom_flags_and_no_stale_revision_banner`).
  rev1의 `test_no_escapes`는 **통과**. λ0 트랙은 감사 중에도 편집됐다.
- **DNA2-9** — rev1 판정서의 **전사 충실성은 검증 불가**. 감사자는 수치를 상수로 승계하지 않고
  재계산했고 그 과정에서 **DNR-1 최소수리 (ii)의 산술 오류를 확인**했다 ⇒ rev1 판정서 인용 시
  그 문장은 **거짓으로 표시**하라.
- **DNA2-10** — 재현하지 못한 것: 격리 미러의 git 이력 의존 테스트 1개 `skipped`(실질 별도 확인).
- **DNA2-11** — §8-2의 *"발화하면 회차 중단"* → **"사후 미채점 + 별도 사전등록"** 으로 문안 교체
  (하네스가 중단을 집행할 수 없다).

---

## §6. 신규 인용금지 / 필수병기 (문자 그대로 승계)

> **DN2-1**(필수병기) — *"이 회차 제출 시점 실측: `presubmit.py` **3 BLOCK**(등록 2건 + `doc_facts`),
> 전체 CPU 스위트 **696 중 4 실패**(전부 `test_lambda0_prereg`, λ0 트랙 소관), 이 트랙 **98/98 OK**.
> rev1 판정서와 rev2 §4-4의 '2 BLOCK'·'681'은 **2026-09-14 16:2x 시점 값**이며 제출 판정에 쓸 수 없다."*

> **DN2-2**(인용금지) — *"rev1 판정서 DNR-1의 최소 수리를 문자 그대로 구현했다/구현해야 한다"*.
> 그 처방 (ii)는 기준 사례를 탈락시킨다: `198.00×1048576/20480 = 10137.6 ∉ [10036, 10137]`.
> rev2가 역산을 게이트에서 뺀 것이 옳다.

> **DN2-3**(인용금지) — *"(i) 사다리 술어는 도착열에 둔감하다"* 및 *"사다리 반례는 908179 L1
> 하나다"*. 실측 사다리 **6/8**, 반례 **2개**. OOM이 재현돼도 다른 도착열을 밟으면
> `UNREALIZED-OTHER-ARRIVAL`이며 **어떤 서술 문장도 쓸 수 없다**.

> **DN2-4**(인용금지) — *"DN arm에서 E5는 구조적으로 발화 불가다"*. 근거가 방향이 거꾸로다.
> 지지되는 문장은 *"pre-repair true-dual 2 boot에서 E5는 발화하지 않았다(n=2, 경험적; 30 boot
> 전수 0 hit)"* 까지다.

> **DN2-5**(필수병기) — `NOT-RECOVERED` 인용 시: *"분지 A는 OOM 0줄 + `#new-token ≥ 10036` 완주만
> 보고 **비-OOM 치명 크래시를 걸러내지 않는다**. 따라서 `NOT-RECOVERED`는 `srv_DN1.log`의
> traceback 계수·`BOOT_FAILURES.txt` DN 행과 **함께만** 인용한다."*(승계 `RRC-4` 계열)

**승계 확인**: rev2가 선언한 **`DNR-V1…DNR-V9`** 전문 **유효**(단 `DNR-V1`의 인접 근거인 rev1 §2
역산 문장은 DN2-2로 정정). rev2의 `DN-1…DN-12`도 전부 유효, `DN-11` 번호 정정·`DN-12` 신설은 옳다.

---

## §7. 이 판정으로 바뀌지 않은 것

확정 결론(**HE0** · **layer-type 정책 全형태 死** · **정책 순위** · **stake #1 구조 판정** ·
**게이트 #13/#16** · **C2 인용정지 2건** · Zamba2/triton 동결 · 실무 기본값 = peak decode 부하 기준
decode-heavy static)은 **재검증하지 않았고 이 판정으로 아무것도 바뀌지 않는다.**
**이 회차 성능 판정 = 0건.** **Claim D 선결 0건 폐쇄, P2 블로커 3개 불변**(NP-8 · λ0 `NO-GO` ·
W4 λ\* 미측정 · 게이트 #6).
제출은 여전히 **§4-4(1) 범위 한정 OVERRIDE + 사용자 명시 승인**을 선행조건으로 하며
**OVERRIDE 문서는 감사자가 쓰지 않는다.** 추가로 **DNA2-7의 `doc_facts` 위반이 닫히기 전에는
rev2 자신의 §4-4(2)가 제출을 금지한다.**
