# λ0 rev5 사전등록 — 구속 追記 (2026-09-14, `GO-with-caveats` 이후)

> **지위**: `PREREG_LAMBDA0_REV5_2026-09-14.md`(sha `8367b6ecdb6b4d795fac2e8aa9960805887f2512fe623fb01c1a87020082c214`)는
> 5차 규칙층 감사에서 **`GO-with-caveats`**(死因 0, `VERDICT_lambda0_rev5_2026-09-14.md`,
> sha `33dac116c37fe758f39d9b1aa67f55124c0d714a142545e25d544dc45abd45b0`)를 받았다.
> 이 追記는 그 판정의 **caveat λ5C-1…8과 권고 λ5A-1·5·6·7·10을 구속 조항으로 편입**한다.
> **rev5 본문은 수정하지 않는다**(판정서가 sha로 핀했다) — 충돌 시 **이 追記가 우선**한다.
> 코드 측 권고(λ5A-2·3·4·9·11)는 별도로 구현하고 그 결과를 §B에 등재한다.
>
> ★**새 감사 회차를 요구하지 않는 근거(자기 점검)**: 아래 A1–A5는 **사실 정정·공시**이며 판정
> 규칙(F5 문턱·판정 셀·규약·사다리·라벨)을 **하나도 바꾸지 않는다**. A6의 두 문장은 **인용 금지·필수
> 병기의 추가**이므로 주장 범위를 **좁히기만** 한다. 이 주장이 거짓이면 그것이 死因이다.

## A1 (λ5A-1) ★출처 정정 — sha 핀 오류
rev5 §2-3이 반전 격자의 출처로 적은 `7fdc7ab0fa58a18176520e5ae4d1dcfc4a1ab83449ea34eec8f4e0a7917c4506`은
**λ0 rev4 판정서가 아니다** — 그 digest의 파일은 **다른 트랙**의
`results/r2_correctness/rerun_prereg/VERDICT_rerun_rev4_2026-09-14.md`다.
> **정정**: λ0 rev4 판정서 = `results/r2_eval/lambda0_prereg/VERDICT_lambda0_rev4_2026-09-14.md`,
> sha256 **`77ae77fe09bc2c456d7095cf88885b6968abb883b00ce175386783220bcba0ef`**(메인 세션 재계산 확인).

★**전사된 반전 격자 16점 수치 자체는 감사자가 전부 독립 재계산해 일치를 확인했다 — 틀린 것은
포인터뿐이다.** 그러나 **교훈 80(출처 허위는 규칙 정본 파일 안이 가장 위험하다)** 계열이므로
정정을 이 追記의 첫 항목으로 둔다. 또 rev5 §8의 rev4 **사전등록** 핀 `78bb617b…`는 **HEAD 판본**이고
현재 작업트리 사본은 `edc092b9…`(2026-09-14 SUPERSEDED 배너 6줄 추가) ⇒ **인용 시 어느 쪽인지 명시**한다.

## A2 (λ5A-7) 수치 4건 정정 (비하중, 규칙 불변)
| rev5 표기 | 정정값 |
|---|---|
| §1-2 "bare `\r` 116개(`\n`-줄 38–62)" | `\r`는 **`wc -l` 줄 39·56·58–63**에 있다(개수 116은 맞다) |
| §5 "프로브 실행 **14**개 기록" | **15**(rev5 자기 파일 등재 후) |
| §8 R3C-3 "참 8.0 → 보고 **7.155**, −10.6%" | 쉬핑 코드는 **7.1389**, **−10.76%** |
| §0 "`:279-361` `decide()`" | `decide()`는 **`:269-354`** |

## A3 (λ5A-5) F3② 완결 — N-9·N-11 문자 전사
> **N-9** — *"`λ_inf(A)=3.0939 req/s`는 셀 A가 엔진 상한 48에 붙은 상태(decode 줄 3,880 중
> 2,814 = 72.5%가 48, `#queue-req` 최대 47)에서 얻은 closed-loop 포화 처리율의 **상한 프로브**이며
> λ\*가 아니다(NP-3′). legacy · warm-up boot · cudagraph ON에서 측정됐고 분할 혼합비는 미측정이다.
> λ0 rev1이 등록한 λ\*(A) 점추정 2.10–2.51보다 23–47% 크다(λ_inf ≥ λ\*이므로 모순은 아니다)."*
> ★이 사실은 **FALLBACK A 사다리가 사전 2.10에 중심을 두는 판단과 직접 관련**된다.

> **N-11** — *"I2의 `max ITL = 929.91 ms`는 요청 1의 토큰 인덱스 311에서의 단발 정지이며(첫 토큰
> 아님·컴파일 아님·동시성 1), 서버 로그에서 해당 1초 구간 decode step 77→8로 독립 확인된다.
> 원인 미확정. 'ITL 바닥 13.0 ms'만 인용하지 말고 이 꼬리를 병기하라."*

## A4 (λ5A-6) rev4의 어느 절이 살아 있는가 — 한 줄 명시
> rev4에서 **살아 있는 것**: §9 인용금지(Q1–Q5·L1–L3·R3C-1…4)·§10 필수병기(NPC-I 포함)·
> §11 인용 고정표. **대체된 것**: §3·§4(앵커 술어와 그 표)·§5 판정 규칙·§6 예보 F1–F4·
> §7 예산 — 전부 **rev5 §1·§2·§4·§5·§7이 대체**한다. rev4 §6 F3의 커버리지 밴드 `[0.24, 1.79]`는
> **ANCHORED 값이라 쓰지 않는다**(FALLBACK은 A `[0.4976, 3.424]` · B `[0.6333, 1.532]`).
> ★`lambda0.sbatch:350`이 아직 `registered rules (PREREG_LAMBDA0_REV4 sec 5)`를 인쇄하는 것은
> **코드 측 결함**이며 §B에서 고친다.

## A5 (λ5A-10) 잔류 escape 5종 — 이름으로 등록 (λ5-6 구체화)
감사자 독립 22 변이 중 **5종이 5개 selftest 전부를 통과**했다:
`BUDGET_RATIO_GRID→(1.00,)` · `REQUESTED_BUDGET_GPU_H 3.60→99` · `BOOT_TEARDOWN_S 110→10` ·
`LAMBDA_STAR_ABS(B)→(0.05,5.0)` · `EBAR_TOL 0.025→0.50`.
**라벨·분지·λ\*는 어느 것도 움직이지 않고**, 움직이는 것은 **등록된 예산/가드 정의역 자신**이다
(3.509 ≤ 3.60 주장 등). ⇒ *"변이 54/54 차단, escape 0"* 은 **등록 변이 목록 한정 사실**이며
위 5종은 그 밖이다. 참고: rev4가 escape시킨 **Z1·Z4·Z6·Z17은 닫혔다**(감사자 재현).

## A6 (λ5C-1…8) 신규 caveat 전문 편입
`VERDICT_lambda0_rev5_2026-09-14.md` §3의 **λ5C-1 … λ5C-8 전문**을 이 사전등록의 구속 조항으로
편입한다. 결과 문서·정본은 **문자 그대로** 승계한다. 특히:

★★**λ5C-1이 이 트랙의 구조를 바꾼다** — `#running-req ≥ 48`(F5 앵커 자격)은 **이 구성의 shape B에서
어떤 부하로도 도달 불가**다: `max_prefill_tokens=16384`가 8192-토큰 요청 **2개**를 한 prefill 배치의
상한으로 만들어 prefill 지배 shape의 *decode* 배치가 구성상 ~2이고, KV 예산은 구속하지 않았다
(같은 레코드의 실현 동시성 **57.43/64** · median TTFT **90.69 s** ⇒ 엔진은 셀 내내 backlog였다).
⇒ **(a)** 셀 B의 F5 실격은 규칙의 올바른 적용이지만 *"셀이 포화하지 못했다"*로 바꿔 쓸 수 없고,
**(b) ANCHORED 분지은 이 shape/cap의 모든 미래 instrument run에서 구조적으로 사용 불가**이므로
*"I3를 다시 재서 앵커를 살린다"*는 **경로가 아니다**, **(c)** 미달 원인은 등록대로 **미확정**이다.
⇒ 이 회차 이후 **shape B의 앵커 경로를 되살리려면 F5 문턱 자체를 재설계해야 하며 그것은 별도
사전등록 대상**이다(이 追記는 그것을 허가하지 않는다).

**λ5C-5·λ5C-6의 실무 함의(등록)**: A 창은 등록 타당범위를 **덮되 하단 여유가 3.3%뿐**이고
(λ\*(A)가 절대하한 1.08보다 3% 아래면 `LADDER_TOO_HIGH` + **등록된 재설계 1회**),
B 창은 타당범위 `[0.35, 1.20]`을 **양 끝에서 덮지 못한다**(`KNEE_BRACKETED[B]` 구간은
λ\*(B) ∈ [0.574, 0.979] = 로그폭 **43%**). ⇒ **"설계가 잘 맞춰졌다"의 근거로 I3b 0.6956(인용금지)과
probe C 0.6753을 쓸 수 없다.**

## B. 코드 측 권고의 처분 (λ5A-2·3·4·9·11)
아래 5건은 **제출 전 구현 대상**으로 등록한다. 구현되면 이 절에 파일·줄·검증을 등재하고
**변경된 파일의 새 sha256을 함께 적는다**(λ5A-8의 digest 목록이 그만큼 무효화되므로 재계산 필수).
1. **λ5A-2**: `lambda0.sbatch:165` 직전에 `export LAMBDA0_I3_JOB_DIR="$(dirname "$INSTR")"`
   (분지 입력과 selftest 입력을 묶는다 — 감사자가 대체 dir rc 1 / 기준 사례 rc 0으로 검증).
2. **λ5A-3**: `lambda0_plan.py:616`의 `### measured lambda_inf` 머리글을 **모드별**로 바꾼다
   (FALLBACK에서 그 문구는 거짓이다).
3. **λ5A-4**: `lambda0.sbatch:195-197`의 도달 불가 `ANCHORED` 가지를
   `ABORT_D18 unexpected ANCHORED (rev5 registers FALLBACK)`로 교체(fail-closed 보존).
4. **λ5A-9**: `lambda0.sbatch:211-212`가 변이 하네스의 **모든** 비영 rc에 `ABORT_MUTATION_ESCAPE`를
   찍는다 — rc 2(`HARNESS CANNOT RUN`)는 **측정 실패**이므로 rc를 분기한다(교훈 21).
5. **λ5A-11**: `lambda0_lambda_inf.py:16`·`:33`의 생산자 줄 인용이 **미커밋 작업트리에서만 참**이다
   (그 파일은 다른 트랙이 편집 중) ⇒ **커밋 핀 또는 코드 텍스트 인용**으로 교체.
★**λ5A-8(커밋)**: F6가 기록하는 digest가 전부 미커밋 파일의 것이다 ⇒ **제출 전 커밋**이 선행조건이며,
커밋 후 판정서 §4 λ5A-8의 digest 10개를 **재확인**해야 한다(다르면 그 판정서는 그 파일에 무효).

### B-1. 구현 완료 (2026-09-14, engine-porter) — 전부 변이 검증됨
| 권고 | 구현 | 검증 |
|---|---|---|
| **λ5A-2** | `lambda0.sbatch:184`에 `export LAMBDA0_I3_JOB_DIR="$(dirname "$INSTR")"`(selftest `:185` 직전) | ★**대체 instrument dir로 `decide()`가 `ANCHORED`(실격 0, λ_inf 3.0939/0.6956)를 내는 비공허 조건**에서: 쉬핑=블록 rc **2**, 변이(export 삭제)=블록 rc **0**(=W4′ escape 재현). 기준 사례는 명시·기본 양쪽에서 rc **0** ⇒ **처방이 기준 사례를 탈락시키지 않는다.** 변이 하네스 무영향(54/54) |
| **λ5A-3** | `lambda0_plan.py` `FALLBACK_TITLE`/`ANCHORED_TITLE`, 모드별 선택 | FALLBACK 출력이 `### FALLBACK prior lambda_inf (NOT measured -- registered prior A=2.10 / B=0.675; the measured 3.0939 / 0.6956 are DISQUALIFIED per cell by newpair F5 …)` |
| **λ5A-4** | `ANCHORED` 가지 → `ABORT_D18 unexpected ANCHORED (rev5 registers FALLBACK) …` + `exit 2`, `--fallback` 무조건 | 술어와 `case`는 **보존**(삭제 아님) |
| **λ5A-9** | rc 분기: 1 → `ABORT_MUTATION_ESCAPE`(exit 5) · 2 → `ABORT_MUTATION_HARNESS_CANNOT_RUN (… MEASUREMENT FAILURE, not an escape)`(exit 6) · 기타 → `ABORT_MUTATION_UNEXPECTED_RC`(exit 6) | 실행 대조: 쉬핑 rc1→5 / rc2→**6**, 변이(옛 한 줄) rc1→5 / **rc2→5(오귀속 재현)**. ★rc 2는 가설이 아니라 **쉬핑 하네스의 실제 상태**(gitignore된 아티팩트가 안 풀리는 트리에서 `HARNESS CANNOT RUN`) |
| **λ5A-11** | 생산자 인용을 **코드 텍스트**로 교체(`grep -oE "#running-req: …" \| sort -n \| tail -1 > …` 블록 · `i_mark ()` 헬퍼) | 두 스니펫이 작업트리(`ab55c07c…`, 684줄)와 HEAD `8507cee`(573줄, 블록 `:496-497`) **각각 정확히 1회** 출현. 남은 줄 번호는 **명명된 스냅샷**에 귀속 |
| **λ5A-6**(부수) | 배너 → `=== registered rules (PREREG_LAMBDA0_REV5 sec 5 + ADDENDUM sec A6; rev4 sec 5 SUPERSEDED, rev4 sec 9/10/11 still live) ===` | 追記 §A4와 일치 |

**변이 검사(교훈 53)**: 미러 트리에서 CONTROL 13 tests OK(skip 1), **수리 6종을 되돌린 변이 전부
대응 테스트 실패**(M-A…M-F). 회귀: 이 트랙 `test_lambda0_prereg` **94 OK**(81→94), 전체
**710 OK**(697→710), `bash -n` rc 0, 변이 하네스 **54/54 blocked · escape 0**.

★**engine-porter가 등록 밖에서 찾아 고친 2건(공시)**: (a) `lambda0.sbatch:65-67`의 **대체 인용
자신이 이미 거짓**이었다(그 상수는 HEAD `bddff6a`에서 `:138`, HEAD `8507cee`에서는 `:166`) ⇒ 커밋 핀과
함께 stale 표기 (b) `case` 가드 주석이 λ5A-4가 없앤 `else`를 가리켜 **출하 순간 낡을 문장**이었다 ⇒ 재작성.

### B-2. ★digest 재확인 — 판정서 §4 λ5A-8의 핀 10개 중 **3개가 의도적으로 무효화**됨
| 파일 | 감사된 digest | **새 digest** | 상태 |
|---|---|---|---|
| `lambda0.sbatch` | `312cefe48292b1e1c482de9d4311281bfbfe6e750a7d7396f80f16224c16548f` | **`c789af6ce8861ba56faeda96c3daf662f0bdcf1b380269c831bec8092d553416`** | ★CHANGED(λ5A-2/4/9/6 + 공시 2건) |
| `lambda0_plan.py` | `ca962d10f6b0f1012bd6469d42260c5bbbd91e0d80da64a99b90c2f1563a2e7a` | **`2e51280c42e61f0e3ff7fdeccb1e7184a5091e228328d66d9d6d482e2893f86d`** | ★CHANGED(λ5A-3) |
| `lambda0_lambda_inf.py` | `f0b23b1a10fe57892222b5525a1215c1d10edb335769a7851f422de97b98b5ed` | **`2c9c80111aa5c0bf9f35f5156cdbcfe7d5248a65199952953b3a95f19f94f861`** | ★CHANGED(λ5A-11 + stale docstring) |
| `lambda0_label.py` | `75cc7a67898f49b298dfb0bd313e34194dc5faf7eeeb303a6c1c2c7f58c6efc7` | 동일 | 불변 |
| `lambda0_analyze.py` | `3b17a162272f614f40f68f53045db38a7e55b8457dafb7829423f57b659d409d` | 동일 | 불변 |
| `lambda0_cells.py` | `3dd4a1a0a78ab239acb209ca0bc4169108a2d279fa36f6075cafea19e00d4205` | 동일 | 불변 |
| `lambda0_reachability.py` | `8b49ca716748f6831b0cc8468cc1628ab2e8002134eb9869fcfbf4b647d4b303` | 동일 | 불변 |
| `lambda0_mutation_check.py` | `44b41cffef5229fe1b5375f2e462fe45db57ac478ef631f76da7ba20c374946e` | 동일 | 불변 |
| `lambda0_cellprint.py` | `222519d4812b0d90ac5ab53324c48e0a800e3aac41fb80d13a54a5d26d57f93b` | 동일 | 불변 |
| `PREREG_LAMBDA0_REV5_2026-09-14.md` | `8367b6ecdb6b4d795fac2e8aa9960805887f2512fe623fb01c1a87020082c214` | 동일 | 불변 |
| (핀 밖·신규) `tests/test_lambda0_prereg.py` | — | `424bb5f68b2aae3f67725364a100ae9c7081232af18b935312e3c41207bd9918` | 신규 |

⇒ **판정서 `33dac116…`는 위 3개 파일에 대해 "그 바이트"로는 무효**이며, 무효화 사유는 **그 판정서
자신이 처방한 수리**(λ5A-2/3/4/9/11)의 이행이다. 판정의 **규칙층 결론(死因 0, `GO-with-caveats`)과
caveat λ5C-1…8은 영향받지 않는다** — 세 변경은 전부 **fail-closed 강화·라벨 정직성·인용 정확성**이며
판정량(λ\*·4라벨·F5 문턱·판정 셀·규약·사다리)을 **하나도 건드리지 않는다**(변이 검사가 그 불변을 고정).

### B-3. 미해결로 등재 (고치지 않음 — 판단 권한 밖)
1. **`lambda0_plan.py:279`가 `plan.json`에 `"rev": 4`를 쓴다** — `lambda0_analyze.py`·`lambda0_label.py`는
   `"rev": 5`를 쓴다. 읽는 코드는 없다. 그 필드가 "사전등록 rev"인지 "plan 스키마 rev"인지가 **등록되지
   않았으므로** 추측해 고치면 **거짓 run-record 라벨을 새로 만든다**(λ5A-6과 같은 결함 계열).
   ⇒ **다음 회차의 追記 항목**으로 남긴다.
2. **`lambda0.sbatch:266`의 `lambda0_plan.py … | tee "$OUT/PLAN.txt"`에 `|| exit 2`가 없다**
   (`--json` 쌍둥이에는 있다) — 선재 결함. `tee`가 plan 실패를 **사람이 읽는 기록에 대해서만** 삼킨다.

## C. 제출 선행조건 (rev5 §10 + 이 追記)
①rev5 `GO-with-caveats` ✅ ②§B의 5건 구현 + digest 재확인 ③`presubmit.py` 차단 0 또는 **범위 한정
OVERRIDE + 사용자 명시 승인** ④이 트랙 CPU 회귀 통과(감사 시점 81/81 OK) ⑤`--comment` 확인 ✅
⑥★**GPU 예산 승인** — FALLBACK 최악 코너 **3.509 GPU-h**(1.00× 1.348 / 0.50× 1.977), 요청 3.60,
벽시계 4.50 h. **이 규모는 D-none(0.197)과 달라 사용자 명시 승인이 필요하다.**
