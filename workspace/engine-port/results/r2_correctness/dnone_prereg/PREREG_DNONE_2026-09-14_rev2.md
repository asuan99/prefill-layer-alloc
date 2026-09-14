# 사전등록 — D-none: grad-guard 축 단독 이동 대조 arm (2026-09-14, **rev2**)

> **상태**: rev1이 규칙층 감사에서 **`NO-GO`**(死因 DNR-1…DNR-4,
> `VERDICT_dnone_rules_2026-09-14.md`, sha256 `530ac31b68e8f89c19a4c452bc35fba5a70f7fa6712bfa9d65b355dc3da82a06`).
> rev2는 그 4건 + 비차단 권고 DNA-1…DNA-10을 반영한 **재감사 대상**이다.
> rev1(`b5d1a244…`)은 **삭제하지 않는다** — 판정서가 핀했다.
> `GO` 이전 sbatch 금지. 제출은 추가로 §4-4(1)의 **OVERRIDE + 사용자 승인**을 선행조건으로 한다.

## 0-a. rev1 → rev2 변경표 (재감사용 색인)

| 死因/권고 | 수리 위치 | 무엇이 바뀌었나 |
|---|---|---|
| **DNR-1** (V가 처치 강도의 감소함수 ⇒ `RECOVERED-*` 도달 불가) | **§4-1 전면 재작성** + §5-1 라우팅표 + §0-d 검정력 공시 | V를 **OOM 유무 두 분지**로 분리, 밴드 상한 제거, `UNREALIZED` 3중 의미 분리. ★**감사자 처방의 절벽을 정정**(아래 §0-c) |
| **DNR-2** (`guard_applied=yes`는 하네스가 못 내는 값) | **§4-3 (12)** | `r2c_guard=none guard_applied=DN-boots-only` + `worker_grad_guard=none` 병기로 교체 |
| **DNR-3** (F-n3는 항등식/허위귀속 생성기) | **§5-3** | 예보 → **기록 항목** 강등, `DESIGN-REFUTED` 라벨 **폐기**, `FAIL` 원인 전사 규칙 신설 |
| **DNR-4** (§4-4(2)가 승계 NPC-H를 스스로 위반) | **§4-4(2)** + **§8-1** | 합격 기준을 **이 트랙 98/98**(NPC-H)로 한정 + 전체 스위트는 제출 시점 실측을 병기. **절차의무 15건을 열거로 편입** |
| DNA-1 | §4-4(3) | 리터럴 제출 명령 전문 등록(7개 `R2C_*`) |
| DNA-2 | §9 | `DN-13` → **`DN-11`**(존재하지 않는 id 정정) |
| DNA-3 | §4-4(2) | `citation_stops` 0 위반은 **승계 집행의 증거가 아니다** |
| DNA-4 | §4-4(2) | `presubmit.py`는 read-only 아님 → 실행 후 `git diff HEAD --` 2경로 빈 출력 확인 의무 |
| DNA-5 | §7 | 예산 모형은 **1점 2모수 적합**(검증 아님) |
| DNA-6 | §4-3 (13) | `dn_boots`는 불리언 ⇒ `order=` 술어와 **함께만** 유효 |
| DNA-7 | §8-2 | E5의 이 회차 정의역은 **채점 TD boot**(DN 아님) + `\b`의 POSIX ERE 한계 + Y6 처분 |
| DNA-8 | §0-b | cudagraph prefill 수치 정정(TD1 41/41, **L1 40/40**) |
| DNA-9 | §0-c | provenance glob 인용 `:364-368`(본체 `:367`) |
| DNA-10 | §5-1 | "서명 0/3" 행 + `verdict.txt` 미생성 분지 등록 |

## 0-b. 이 회차가 묻는 단 하나의 질문 · 스코프 튜플

> 같은 (모델·백엔드·ctx·split·노드·물리 GPU·엔진 `src`·채점기·계측 플래그)에서
> **`PDMUX_WORKER_GRAD_GUARD`만** `inference_mode`(수리) → `none`(수리 前 가드 의미론)으로
> 이동시키면 job 907959의 OOM이 되돌아오는가?

**스코프 튜플**(모든 문장에 붙는다)
> (model `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base` = `NemotronHForCausalLM`, 56층, `n_groups=8` ·
> backend `flashinfer` **0.6.10 이라고 보고한 설치본** · ctx **16384** · `mem-fraction-static 0.82` ·
> `max-running-requests 48` ·
> `--disable-radix-cache --chunked-prefill-size -1 --disable-overlap-schedule --random-seed 1` ·
> split **fixed D44**(green stream index 4, realized 64/44 SM) ·
> **cudagraph ON — decode 한정**(908179 실측 prefill `cuda graph: False`: **TD1 41/41 · L1 40/40** —
> boot마다 prefill 줄 수가 달라 "41/41 양 arm"은 부정확했다, DNA-8) ·
> verdict rule **v2**(2026-09-11) · 채점기 `ec355e171a66d68eab1300edcc7616a693cd55aa2c3fb5c251ee4b42eac50d30`(**무수정**) ·
> client `95e10b492ed6239d83cae20ad7898bdbbcce1713a37d42c2bfa111f7347ec367`(무수정) ·
> **하네스 `ab55c07cc096bd4cdfe6d484fc9a983302561e03232f641c5994298bdc183bd3`**
> (← 908179의 `f39b167bb6bd713af168abfbe3495caee099a6c0087e41b05e70b9a8d8ac0403`) ·
> 엔진 `src` **무변경** · `R2C_EXPECT_{MODEL,BACKEND,CTX}` 3중 대조 ON ·
> `R2C_ORDER="L TD L TD DN"` · **`R2C_GUARD=none`** · `R2C_INSTRUMENT=0` ·
> `PDMUX_MEM_TELEMETRY=1` · `PDMUX_TRACE_FORCE_PREFILL=1` · seed 1 · `--n-probes 8` ·
> A100-SXM4-80GB 108 SM 1장 · **노드/물리 GPU = 실행 후 기록**(게이트 #233))

### 이 회차가 **아닌** 것
- Claim D 선결을 닫는 회차가 **아니다**(§6). **P2 착수와 무관**(블로커 = λ0 `NO-GO` · W4 λ\* 실측
  부재 · 게이트 #6 — 전부 불변, NP-8 승계).
- 성능 측정이 **아니다**(RR-1). **`a9cd8dd` 커밋 전체의 대조가 아니다**(§1-2, DN-9).

## 0-c. 승계 문서 고정 (sha256) · ★감사자 처방의 절벽 정정

| 문서 | sha256 |
|---|---|
| `dnone_prereg/PREREG_DNONE_2026-09-14.md`(**rev1**, `NO-GO`) | `b5d1a244470dbe53d911d6d0f65ca3e8d38bca0bddad21b273861b1b97e2b19e` |
| `dnone_prereg/VERDICT_dnone_rules_2026-09-14.md`(rev1 판정서, DNR·DNA·DNR-V의 원문) | `530ac31b68e8f89c19a4c452bc35fba5a70f7fa6712bfa9d65b355dc3da82a06` |
| `audit_908179_2026-09-14/VERDICT.md`(§8.5 = 설계 출처, A8179-1…7·P1…7) | `86627a05354ff28b3b32a7764e85f84384d0ac94a7b668b3b2b6ed8aa2a79f7a` |
| `audit_907959_2026-09-13/VERDICT.md`(N-1…13) | `f544544fc973e0abc2d1aa97905e4e005d47e657c30e64d9ea7717d3a48ddc71` |
| `audit_907959_2026-09-13/DIAGNOSIS_true_dual_oom.md`(H1 · `R(T)` · 사다리·역산식) | `8c857bede1576d9e8042d759889aec3a9f034ed6167fea4093dd947bb9565bee` |
| `audit_908020_2026-09-14/VERDICT.md`(A908-1…7 · C1–C5) | `dc19d1183564591a6394d08432f0c7414d33332de68e55bf403a757ddf8c12ea` |
| `rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md`(등급 철회, RRC-1…13 유효) | `118bb45789baf117cc9a1460c8ebfb487088e0d86d81ca5ab9316afc5c412355` |
| `rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md`(RA3-1…12) | `d8281866f3b06f841aa0eb3b00a6b6cf93cd9198d505a1ca2e1ca8ee5ed304c7` |
| `rerun_prereg/VERDICT_rerun_rev4_2026-09-14.md`(RA4-1…12) | `7fdc7ab0fa58a18176520e5ae4d1dcfc4a1ab83449ea34eec8f4e0a7917c4506` |
| `CARRYFORWARD_INVENTORY_2026-09-14.md`(98건 계수표) | `51b7b016f404ef736a8baa21ed9997116523a9beb31d45aca4cc48ae505be6ae` |
| `E5_FAMILY_RA4_2_2026-09-14.md` / `e5_family_probe.py` | `9214e62e83871a5cc934c38401e8b5b528b745ed6a8f25f5b9e9bc8089dfa53a` / `c5a063f588e91d0274ecb746bc832037a223cc831fe02201491fa012d5da3a30` |
| `tests/test_r2_correctness_dnone.py`(36 테스트) | `4209e572c6fa8a0c7a51439e3307b8aa11e8c2ae025e9857d40f4530b45766fd` |
| `rerun_prereg/PREREG_RERUN_2026-09-13.md` | **두 값**: 908179 실행 시점 `1e421391a97293773ff676a0ab4f24e06c269a68a7ba69fba769c5f9d22923b0` / 현재 `fa304128ba1d053ca83fec4a3a7fd81320eaca57b9549f1609f1ce06bf7bda41` |

**sha 드리프트 등재** — 2026-09-14 RA4-9 정정이 `PREREG_RERUN:1169`에 dated 追記를 덧붙여(본문 삭제 0)
sha를 이동시켰고, `audit_908179_.../VERDICT.md:5`가 핀한 값과 현재 파일이 불일치한다. **job 908179의
`PASS` 채점은 소급 변경되지 않는다**(채점은 §3-b C1 본문 `:391` 사용). 이 회차 `provenance.txt`는
`*_prereg/*`를 glob(`r2_correctness.sbatch:364-368`, glob 본체 `:367`)으로 전수 수집하므로 현재 값이
산출물에 남는다 — **은폐 금지**(C4 승계). ★이 glob은 **이 문서(rev1·rev2)와 판정서까지** 핀한다.

**앵커 부패(등록번호 `DN-11`, rev1의 오번호 `DN-13` 정정)**: §2의 하네스 변경으로
`audit_908179_.../VERDICT.md` §8.5의 `:216` 등 줄 인용은 **그 sha에 대해서는 참, 현재 파일에 대해서는
거짓**이다. 이동: unset 루프 `:216→:294` · `ORDER` case `:507-513→:608-619` ·
provenance 블록 `:285/:327→:330/:369` · scope guard `:314-325→:397-408`.

### ★0-c-1. 감사자 처방(DNR-1 (ii))의 **절벽을 정정한다** — 역산 문턱을 쓰지 않는다
판정서 DNR-1의 최소 수리 (ii)는 *"`Tried to allocate <S> MiB`를 `T = S·1048576/20480`으로 역산한
`T`가 [10036, 10137]에 드는가 … 198.00 MiB → **T = 10137.6** ⇒ 밴드 안(2/2)"* 이라 적었다.
**내 재계산: `198.00 × 1048576 / 20480 = 10137.6` — 밴드 상한 10137을 0.6 초과한다.** 즉 처방을
문자 그대로 구현하면 **기준 사례 자신이 술어를 통과하지 못한다**(DNR-2와 같은 형태: 산문 문턱이
아티팩트와 어긋남, 교훈 66). 게다가 `floor`=10137(통과) vs `round`=10138(탈락)로 **규약 선택이
판정을 바꾸는 새 자유 표면**이 생긴다. 밴드 [10036,10137] 자신이 **같은 반올림된 `S`를 역산해 만든
값**이므로 그것을 다시 문턱으로 쓰는 것은 순환이다.
⇒ **rev2는 역산을 게이트에서 빼고**(정보 항목으로만 보존) **정수·문자 단위로 확정 가능한 두
술어**를 쓴다(§4-1 분지 B). 역산값은 `T ≈ 10137`(±1, 출력 반올림)로 **기록**한다.

## 0-d. ★검정력 공시 (DNR-1 수리의 필수 구성요소 · RA3-1 계열)
> **rev1의 V로는 `RECOVERED-*`가 도달 불가였다**(DNR-V1). rev2의 V 수리 **後에도** 이 회차의
> 최대 산출은 **n=1 스코프 한정 존재 문장 하나**다: *"(0-b 튜플에서) 가드 축 단독 이동이 907959의
> OOM을 그 서명까지 재현한다."* 기전(층별 retention) · 하네스 축 · boot 간 분산은 **닫지 않는다.**
> `NOT-RECOVERED`가 나오면 `PLAUSIBLE(조건부)`의 **강등 검토 대상**이 되지만 등급 변경 권한은
> 이 문서에 없다(결과 감사 소관).

---

## 1. 처치 — 단 하나의 축

### 1-1. 무엇을 이동시키는가
`PDMUX_WORKER_GRAD_GUARD`: `inference_mode`(908179가 돈 엔진 기본값) → **`none`**.
엔진 구현은 이미 있다(`src/multiplex/multiplexing_mixin.py:119-123`, `none`→`nullcontext`;
해석 `:127-144`). **엔진 `src`는 이 회차에서 한 줄도 바뀌지 않는다.**

### 1-2. `none`이 되돌리는 것과 되돌리지 않는 것 (필수병기 **DN-9**)
커밋 `a9cd8dd`는 **가드 수리 + arm 대칭 메모리 계측**을 함께 담았다. `R2C_GUARD=none`은 **가드만**
되돌린다 ⇒ **DN arm은 "907959의 엔진"이 아니다.** 907959에 없던 계측이 켜져 있고(그 때문에 피크
축이 측정 가능), 그 계측의 피크 리셋은 카운터 전용(`reset_peak_memory_stats`)이라 할당을 바꾸지
않는다. **"DN이 907959를 재현한다"는 금지**, 쓸 수 있는 것은 **"907959의 가드 의미론"** 까지다.

### 1-3. 설계 선택 (A/B/C) — 정본 2곳의 불일치 해소
| 안 | 출처 | 비용 | 판정 |
|---|---|---|---|
| A: 채점 `L TD` × 2 job | 감사 §8.5 | 0.10–0.17 ×2 | **기각** — 노드/물리 GPU 축(게이트 #233)이 또 움직인다 |
| B: 비채점 진단 boot(warm-up 패턴) | `PREREG_RERUN` C8 | ≈0.025 | **기각** — warm-up은 `PDMUX_TELEMETRY_PATH` 없이 떠 **피크·가드 축이 둘 다 사라진다** |
| ★**C: 같은 job 안의 비채점 진단 boot + telemetry 부여** | 이 문서 | **0.197 GPU-h** 1 job | **채택** |

**C의 근거** ①대조 arm과 수리 arm이 같은 job·같은 물리 GPU·같은 sha. ②채점기를 건드리지 않는다 —
`r2_correctness_check.py:366`이 `boots.txt` 라벨만 열거하고 전 open이 label f-string
(**glob/listdir/walk 0건**, 감사자 독립 재확인: open 지점 `:135 :190 :221 :366 :373 :502`) ⇒
`DN1` 불가시. 부수 확인: `x1_cross_job_compare.py`는 `LABELS` 하드코딩, `srv_*.log` glob은
`r2_correctness.sbatch:664`의 **`INSTR=1` 전용**(등록 `INSTR=0`) 이며 출력은 `instrument/`뿐,
`diag_boots.txt`·`BOOT_FAILURES.txt`·boot 배너의 소비자는 저장소 전수 **0건**.
⇒ **"예측된 `FAIL`은 정보"라는 문자 등록이 필요 없다**(감사자 §4-1이 이 면제를 **반증 실패**로 확인).
③양성대조가 이미 엔진에 있다(§4-2).

### 1-4. 순서와 boot 수 — 구속조건
- verdict rule v2의 귀무대조가 문자로 **"두 L boot"·"두 within-arm O 쌍"**
  (`r2_correctness_check.py` 헤더 `:67-78`) ⇒ 채점 순서를 줄이면 게이트 라벨이 **구성상 해석 불가**.
- `BASEPORT = 30000 + (job % 5000) * 6` ⇒ 포트 6개 예약, **boot 5개 상한**. `L TD L TD DN`이 정확히
  그 경계이며 6번째는 다음 job의 warm-up 포트를 침범한다(과거엔 **조용한** 교차-job 충돌).
  `n_arms ≤ 5` fail-closed 검사를 등록된 계약으로 채택한다.
- ⇒ **DN은 n=1이 구조적 상한**(§6-4). DN은 **마지막**에 둔다.
  ★단 "마지막이라 오염이 구조적으로 0"이라는 rev1 문장은 **F-n3의 강등 근거로만** 쓰고
  예보로는 쓰지 않는다(DNR-3).

---

## 2. 하네스 변경과 그 대가

`r2_correctness.sbatch` 573 → 684줄. 계약: ①`ORDER`에 `DN`(`:611-618`, 가드 재수출 `:618`)
②`R2C_GUARD` fail-closed(`:259-271`) ③`DN`은 `diag_boots.txt`로(`:624-629`)
④`DN`도 `tel_DN1.jsonl`·`green_DN1.json`·같은 client 호출(`:611-655`) ⑤기본 경로 argv 바이트 동일
⑥provenance 기록(`:350-351`) ⑦`R2C_EXPECT_*` 유지(`:397-408`) ⑧`n_arms ≤ 5`(`:280-290`)
⑨**채점 L/TD boot은 항상 엔진 기본 가드**(`:341-351` 주석 + `:617-618`).

**argv 바이트 동일성의 증명**(손으로 쓴 기대값이 아니다): `git show 83d8cb9d…:…/r2_correctness.sbatch`의
sha가 `f39b167b…`(= `job_908179/provenance.txt` 기록값)임을 확인하고 그 blob에서 908179의 38줄 boot
루프를 추출, **옛 블록과 현 블록을 같은 절대 `$OUT`에서 실행**해 4개 산출물을 원시 바이트 비교
(`ORDER` 4종). 기본 경로의 유일한 변화 = job 로그 배너의 `scored=<0|1>`(소비자 0건).
선재 픽스처 62 테스트 무수정 통과.

**변이 검사(교훈 53)** — (a) 가드 재수출 제거 ⇒ **7 실패**, (b) `DN`→`boots.txt` ⇒ **4 실패**
(단정문 `'L1 TD1 L2 TD2 DN1 ' != 'L1 TD1 L2 TD2 '`). **감사자가 독립 재현**해 이름·단정문 문자
일치를 확인했다. ★변이 테스트 자신의 결함이 실행으로 드러나 수리됐다(앵커가 같은 블록 주석에도
존재 ⇒ 5앵커 전부 유일성 단정 추가, 게이트 #9 재발).

**대가**: 하네스 sha가 1개 움직인다(`f39b167b…`→`ab55c07c…`). **이 회차는 하네스 축을 닫지 않는다** —
이동분은 `DN` 분지·노브/포트 검증·provenance 2줄이고 **채점 경로 무접촉**은 CPU 바이트 동일성으로
증명됐다. ★**CPU argv 동일성은 GPU 거동 동일성을 함의하지 않으며 이 문서는 그것을 주장하지 않는다.**

**기타 등재**: 거부된 실행은 `provenance.txt`를 남기지 않는다 · 하네스는 엔진보다 엄격
(`"none "`·`NONE`은 `exit 2`) · 소문자 arm 토큰은 조용히 skip(틀린 boot 대신 **빠진 boot** —
§4-3 (13)이 잡는다) · `diag_boots.txt`는 append-only(SLURM은 job마다 새 디렉터리라 무해).

---

## 3. 판정 규칙 — 채점기 무수정 승계
`r2_correctness_check.py`(`ec355e17…`) · verdict rule **v2** 무수정. 채점 앞 4 boot은 908179와 동일
⇒ 게이트 라벨의 의미가 908179와 같다. 진단 boot은 채점 입력이 아니다.

---

## 4. 채점 前 게이트 (결과무관·기계적)

### ★4-1. V — 반공허성 게이트 (DNR-1 수리, **OOM 유무 두 분지**)

rev1의 단일 술어는 **처치 강도의 감소함수**였다: `report_prefill_stats`가 forward **완료 후**
호출되므로(`scheduler_output_processor_mixin.py:328`, 핀된 `DIAGNOSIS:21`) **죽은 배치는
`#new-token` 줄을 남기지 못한다.** 내 독립 측정(원자료 `srv_*.log`):

| boot | `#new-token` 사다리 꼬리 | OOM | rev1 V |
|---|---|---|---|
| 907959 TD1 / TD2 | … 2310 **3241 6245**(최대) | ★있음(198.00 MiB, `allocated 77.25 GiB`, 2/2 동일) | **FALSE** |
| 907959 L1 | … 3241 3491 6245 **10125** | 없음 | TRUE |
| 907959 L2 | … 3241 **7345 12516** | 없음 | **FALSE** |
| 908179 L1 | … 2310 **2313 5414 6213 10630** | 없음 | **FALSE** |
| 908179 L2 / TD1 / TD2 | … 3241 3491 6245 **10125** | 없음 | TRUE |

⇒ `P(rev1 V | OOM 재현) = 0/2`이며 **기전상 0**. rev2는 다음으로 교체한다.

- **분지 A — `srv_DN1.log`에 `torch.OutOfMemoryError` 0줄**:
  `#new-token ≥ 10036`인 split-prefill 배치가 **1개 이상 완주**했는가(**상한 제거**).
  근거: 10630·12516은 10125보다 **더 무거운** 배치이므로 반공허성 목적을 더 강하게 만족한다.
  도달 확인: 10630 ✓ · 12516 ✓ · 10125 ✓ · 최대 6245 boot ✗ ⇒ **양 분지 도달**.
  참 ⇒ **`NOT-RECOVERED`** · 거짓 ⇒ **`UNREALIZED-NO-RISK-BATCH`**.
- **분지 B — `torch.OutOfMemoryError` ≥1줄**: 다음 **둘 다** 참이어야 한다(정수·문자 단위, §0-c-1).
  - **(i) 사다리**: `55 → 368 → 1468 → 3241 → 6245`가 그 순서로 존재
    (`DIAGNOSIS:128` 전사; 내 독립 확인 = 907959 TD1·TD2 **및** 908179 TD1·TD2 **4/4**.
    ★단 **908179 L1은 이 사다리를 갖지 않는다**(`2313 5414 6213 10630`) — 사다리는 boot 보편이
    아니라 **도착열 의존**이며, 따라서 (i)은 "같은 도착열을 밟았다"의 술어다).
  - **(ii) 마지막 기록 배치 = 6245**: OOM 이전 `#new-token` 최댓값이 **정확히 6245**
    (= 사다리의 다음 rung에서 죽었다는 정수 술어. 907959 TD 2/2 실측 참).
  - 둘 다 참 ⇒ **F-n1 채점으로 진행**. (ii)가 거짓(다른 rung에서 사망) ⇒ **`OTHER-BATCH`**.
    (i)이 거짓 ⇒ **`UNREALIZED-OTHER-ARRIVAL`**.
  - **정보 항목(게이트 아님)**: `Tried to allocate <S> MiB`의 역산 `T = S·1048576/20480`
    (198.00 MiB → **10137.6**, 정수 표기 `T ≈ 10137`±1). **문턱으로 쓰지 않는다**(§0-c-1).
- **`UNREALIZED`의 3중 의미 분리(DNR-1 요구)**: `UNREALIZED-BOOT-FAILED`(DN이 health/warm-up 요청
  단계에서 사망 — 907032 TD 2/2 전례) / `UNREALIZED-NO-RISK-BATCH` / `UNREALIZED-OTHER-ARRIVAL`.
  **세 라벨은 서로 다른 사건이며 어느 것도 H1의 증거가 아니다.**
- ★**보고 의무(모든 분지)**: 형성된 전 배치열과 최대 `#new-token`, OOM 원문 전문(있으면),
  `BOOT_FAILURES.txt`의 DN 행. **`UNREALIZED`라도 이 원자료는 결과 문서 본문에 전사한다** —
  rev1은 확증 증거를 수집해 라벨로 폐기하는 구조였다(DNR-1).

### 4-2. G — 노브 실현 양성대조 + 같은 job 음성대조
- 양성: `tel_DN1.jsonl`에 `{prefill|decode}_worker_grad_enabled = true` **≥1건** AND 대응
  `_worker_inference_mode = false`.
- 음성: 같은 job의 `tel_TD{1,2}.jsonl`에 `*_worker_grad_enabled = true` **0건**
  (908179 실측: `true` 0 / `false` 8569·8263 / `null` 539·525, 감사자 독립 확인).
- 거짓(또는 두 필드가 `null`로만 남음 = 첫 worker task 이전 사망) ⇒ **`NO_KNOB`**: 처치 미검증,
  **H1의 반증이 아니다**. ★`UNREALIZED-BOOT-FAILED`와 동시 발생할 수 있고 그때는 **둘 다** 보고한다.
- 한계 승계(**A8179-3**): 기록은 **스냅샷 단위 last-write-wins**이며 task 커버리지가 아니다 ⇒
  정보량 있는 술어는 "`true`가 ≥1건"(DN) / "0건"(TD)뿐. **퍼센트를 커버리지로 인용 금지.**

### 4-3. SCOPE — C1 승계 + 신규 술어 2개 (DNR-2·DNA-6 수리)
908179의 (1)–(11)에 더해:
- **(12)** `provenance.txt`에 **`r2c_guard=none guard_applied=DN-boots-only`**
  (`r2_correctness.sbatch:273`이 만드는 실제 문자열 — rev1의 `guard_applied=yes`는 하네스가 **낼 수
  없는 값**이었다) **AND** 같은 파일의 `worker_grad_guard=none`(`:350`).
- **(13)** `dn_boots=1` **AND** `order=[L TD L TD DN]`. ★`dn_boots`는 개수가 아니라 불리언
  (`$HAS_DN`)이므로 **`order=` 술어와 함께만 유효**하다(`L TD DN DN`도 `dn_boots=1`이 된다).
- 불일치 ⇒ **`NO_VERDICT_SCOPE`**(등록 밖 실행 — 908020 재발 방지).
- (9)·(11)의 등록된 한계(RA4-3·RA3-8)는 그대로 승계한다.

### 4-4. 제출 선행조건
1. ★`presubmit.py`가 현재 **2 BLOCK**(`m4r_confinement/reachability_spec.json → SINGLE_LABEL_FORCED` ·
   `tc1_model_attrib/reach_spec_rev3_A.json → RESTRICTIONS_INERT`) — **살아 있는 타 트랙 설계 결함**
   (후속 spec 없음 ⇒ `superseded_by` 불가) ⇒ **범위 한정 OVERRIDE 문서 + 사용자 명시 승인**이
   선행돼야 한다. 형식 요건(전례 2건에서 감사자가 확인): ①넘기는 BLOCK 2건 **출력 그대로 인용**
   ②"타 트랙 소관·이 회차와 인과 무관"의 근거 ③**하지 않는 것 전수**(두 차단 해소·완화 금지 /
   레지스트리 spec 제거 금지 / 선행 `NO-GO` 무르기 금지 / 감사 없는 정책 판정 금지)
   ④**범위와 GPU 상한**(이 회차 = 1 job ≈ 0.197 GPU-h) ⑤사용자 명시 지시 인용.
   전례 `cp_baseline/OVERRIDE_VPROBE_SUBMIT_2026-09-01.md:10-11,40,42` ·
   `OVERRIDE_P1_SUBMIT_2026-08-28.md:44,81`. ★**도구 차단범위 축소는 두 전례가 명시 금지**한다.
2. **CPU 회귀** — 합격 기준은 **이 트랙(`test_r2_correctness_*` = dnone 36 + instrument/ctx/scope_guard
   62 = **98**) 전부 `OK`**(**NPC-H** 승계, `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:744`).
   ★**전체 스위트는 제출 시점에 실측해 그 수치와 실패 테스트명을 결과 문서에 병기한다**
   (rev1 작성 시점 감사자 실측 = `Ran 681 … FAILED (failures=1)`,
   `test_lambda0_prereg.TestMutationHarness.test_no_escapes` — **λ0 트랙 소관, 이 회차와 인과 무관**.
   ★단 그 시각 λ0 트랙이 동시 편집 중이었으므로 **"선재 결함"과 "동시 작업 산물"이 아직 구분되지
   않았다** — 제출 전에 λ0 작업 종료 후 재측정해 원인을 확정하고 그 결과를 병기한다).
   `check_doc_facts.py` · `check_line_citations.py --check --all` · `check_citation_stops.py` 0 위반.
   ★**`citation_stops`의 16 rule에는 승계 98건 중 0건이 등재돼 있으므로 "0 위반"을 승계 집행의
   증거로 인용하지 않는다**(DNA-3). ★**`presubmit.py`는 read-only가 아니다**(게이트 #240) — 실행
   후 `git diff HEAD -- .../m4r_confinement/reachability_verdict.json .../tc1_model_attrib/reach_verdict_rev3_A.json`
   가 **빈 출력**임을 확인해 기록한다(DNA-4).
3. **리터럴 제출 명령**(DNA-1 — 기본값은 `Zyphra/Zamba2-2.7B`·`triton`이라 생략 시 908020 기판이 된다):
```bash
R2C_MODEL=nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base \
R2C_ATTN_BACKEND=flashinfer \
R2C_CTX=16384 \
R2C_DSM=44 \
R2C_EXPECT_MODEL=nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base \
R2C_EXPECT_BACKEND=flashinfer \
R2C_EXPECT_CTX=16384 \
R2C_ORDER="L TD L TD DN" \
R2C_GUARD=none \
sbatch workspace/engine-port/results/r2_correctness/r2_correctness.sbatch
```
4. `#SBATCH --comment="field=efficientai;appl=pytorch"` 존재 확인(뉴론 정책).

---

## 5. 예보 (실행 전 고정)

`V(분지 B의 (i)∧(ii)) ∧ G ∧ SCOPE`가 전부 참일 때에만 F-n1을 채점한다. 하나라도 거짓이면 그
라벨이 이 회차의 결과이며 **H1 등급 불변**이다.

### 5-1. F-n1 — 라우팅과 등록 문장 (DNA-10 반영)
서명 3요소: ①요청 크기 `198.00 MiB` ②`Of the allocated memory 77.25 GiB` ③스택 프레임 동일.
★분지 B의 (ii)가 이미 "6245 다음 rung에서 죽었다"를 고정하므로 **F-n1의 정보 증분은 ①②③ 각각의
일치 여부**다(routing과 forecast가 겹치지 않는다).

| 조건 | 라벨 | 등록된 문장(이것 말고 쓸 수 없다) |
|---|---|---|
| 분지 B (i)∧(ii) ∧ 서명 3/3 | **`RECOVERED-STRICT`** | "(0-b 튜플에서) **가드 축 단독 이동**이 907959의 OOM을 그 서명까지 재현한다." |
| 분지 B (i)∧(ii) ∧ 서명 1–2 | **`RECOVERED-WEAK`** | "가드 축 단독 이동이 **같은 rung에서 OOM을 재현한다**(서명 부분 일치: 일치 항목 명시)." |
| 분지 B (i)∧(ii) ∧ **서명 0/3** | **`RECOVERED-WEAK`**(일치 0 명시) | 위와 같으나 **"서명 일치 0"을 문장 안에 적는다.** |
| 분지 B ∧ (ii) 거짓 | **`OTHER-BATCH`** | 배치열·최대 토큰·OOM 원문만 보고. **서술 문장 없음.** |
| 분지 A ∧ V 참 | **`NOT-RECOVERED`** | "가드 축 단독 이동은 이 **1회** 실행에서 OOM을 재현하지 못했다." (강등 검토 대상 — 등급 변경은 결과 감사 소관) |
| 그 밖 | `UNREALIZED-*` / `NO_KNOB` / `NO_VERDICT_SCOPE` | **어떤 서술 문장도 쓸 수 없다.** 원자료만 전사. |
| **`verdict.txt` 미생성**(벽시계 초과 kill 등) | **`NO_RUN`** | 운영 실패로 보고. 여유는 충분하다(최악 ~1,997 s vs `--time` 9,000 s). |

### 5-2. F-n2 — 피크 축 (2차, 조건부)
**토큰 수를 맞춘 epoch에서만** `peak(DN1) − peak(TD*)`를 본다. 예보: **양수**. 맞춘 epoch이 없으면
**`UNMATCHED`**. ★**A8179-1/2 승계**: `Δ`는 배치 크기 차만으로도 생긴다 ⇒ 비매칭 비교 인용 금지.
★`R(T) = T × 1.2750 MiB`는 "그래프가 prefill 전체에 유지된다"는 **가정 위의 산정**(`DIAGNOSIS:88`)
⇒ 크기 불일치는 H1을 반증하지 않고 **그 가정을 반증**한다. **1차 정보는 부호뿐.**
★**필수병기 DNR-V5**: *"907959 TD1·TD2 2/2에서 치명 배치의 `prefill_active_batch_size=14` 스냅샷은
0건(관측 최대 8)이었다. 따라서 OOM 분지에서 피크 축은 `UNMATCHED`가 될 수 있고, 그 분지의 1차
채널은 피크가 아니라 OOM 원문 서명이다."*

### 5-3. F-n3 → **기록 항목으로 강등** (DNR-3 수리)
- (a) 채점 4 boot의 라벨은 DN boot과 **인과적으로 독립**이다(DN은 마지막·비채점·`boots.txt` 부재,
  CPU 98 테스트가 고정). ⇒ **`PASS`는 설계 C의 확증이 아니라 그 구조의 재확인이다.**
- (b) `FAIL`이면 `r2_correctness_report.json`의 실패 술어를 **전사해 원인을 보고**한다.
  ★**DN 귀속은 구체적 기전을 제시하지 않는 한 쓰지 않으며 `DESIGN-REFUTED` 라벨은 폐기한다.**
  근거: job 907959는 `verdict.txt=FAIL`·DN boot **0개**인데 rev1 규칙을 먹이면 `DESIGN-REFUTED`가
  나왔다(선행 4 job 중 2건이 DN 없이 이 분지를 밟는다).

### 5-4. 사후 예보가 아닌 이유
양 끝점이 이미 관측돼 있다 — 907959(가드 前 의미론): TD 천장 6245, 사다리 다음 rung에서 결정적
사망, 198.00 MiB / 77.25 GiB(2/2 동일). 908179(수리 後): 10125 완주,
`peak(TD1)=peak(TD2)=peak(L2)`(차 0 B). 이 회차는 **그 사이의 재현 시험**이다.

---

## 6. 이 회차가 **하지 못하는** 것
1. **기전은 닫히지 않는다**(층별 retention 미측정, `DIAGNOSIS:88`).
2. **하네스 축은 닫히지 않는다**(§2). CPU 바이트 동일성은 GPU 거동 동일성을 함의하지 않는다.
3. **`a9cd8dd` 전체의 대조가 아니다**(§1-2, DN-9).
4. **DN n=1**(포트 예산이 같은 job 안 n=2를 구조적으로 막는다). boot 간 분산 미측정(RRC-9).
   n=1이 정보량을 갖는 근거는 **907959의 결정성**이라는 **사전 관측이자 가정**이다.
5. **Claim D 선결은 하나도 닫히지 않는다.** P2 블로커 3개 불변(NP-8).
6. **소급 금지(RR-3)**: 907100/907456/X1은 Zamba2 기판이라 면역이었고 autograd 부기의 arm 비대칭
   오버헤드는 **미측정** — "오염됐다"도 "깨끗했다"도 쓸 수 없다.
7. **성능 주장 전면 금지(RR-1)**: `PDMUX_MEM_TELEMETRY=1`은 짝지은 on/off 측정이 없다.

## 7. 예산
DN boot 1개 = **1.80분(108초) = 0.030 GPU-h**(907959 실측 슬롯; 908179 정상 슬롯 114/111/110초) —
C8/rev2의 "≈1.5분/0.025"를 정정한다(teardown 20초 + launch ~8초 누락).
job 총계 `L TD L TD DN` = **708초 ≈ 0.197 GPU-h**. 구조적 최악(DN 1개) = **1,400초**.
`--time 02:30:00` 유지. ★**DNA-5**: 예산 모형 `153 + 111·n`은 **1점(908179) 2모수 적합**이며
검증이 아니다 — "1초 오차 재현"을 예측력 근거로 인용하지 않는다.
**재실행 예산 = 0**(C3 승계; `NO_VERDICT_SCOPE`·`NO_KNOB`·`UNREALIZED-*`·`NO_RUN`도 소비로 계상).
★**장부**: 트랙 누적 **1.107500 GPU-h**(등록 0.955000 / 등록 밖 0.152500, `sacct` 감사자 재검산
3,987초 일치)에 이 job을 더해 적고 **908020을 은폐하지 않는다**(C4).

## 8. 승계

### 8-1. ★절차의무 15건 — **열거로 편입**(DNR-4의 뿌리 차단)
rev1은 98건을 "계수 대조"로만 승계해 **자기 §4-4에서 NPC-H를 위반**했다. rev2는 문서 자신이 위반할
수 있는 계열(절차의무)을 전수 열거한다(출처 = `CARRYFORWARD_INVENTORY_2026-09-14.md`):
`NPC-A`(D2′ 발화 시 전사 의무) · **`NPC-H`(전체 CPU 회귀 합격 기준 = 이 트랙 한정 — §4-4(2))** ·
`NPC-J`(벽시계 재제출도 총 1회로 계상 — §7) · `RRC-13`(C5 근거 정정, 결론 유지) ·
`RA3-6`(inference tensor 하드 에러 최소 4종, 계열 등록 — §8-2) · `RA3-10`(대역 동률 우선순위 규약) ·
`RA3-12`(RA3-1…12 + RR-1…15 함께 편입) · `RA4-2`(E5 계열 누락 — §8-2에서 **교체**) ·
`RA4-6`(전방 강제 44/84 누수 — 이 절이 대응) · `RA4-8`(커밋 목록은 감사 시점 `git status`로 재생성) ·
`RA4-9`(§9-4 잔재 — 2026-09-14 처리됨, sha 이동 §0-c) · `RA4-10`(커밋은 내용 무변경 이동) ·
`RA4-11`((9)의 잔존 환경 의존성) · `RR-14`(다음 회차 sha 핀 + 명시 편입 — ★이 조항 자체가 44건만
요구) · `RR-19`(C5 근거 정정).
인용금지 49건·필수병기 34건은 인벤토리 §1–§9를 **항목 수 대조**로 승계하고, 그중 이 회차에 직접
적용되는 것(A8179-1…3, N-7, N-8, RR-1, RR-3, RRC-8, RRC-9, RA3-1)은 본문에 인용했다.
**신규 승계**: 판정서 §6의 **DNR-V1…DNR-V9 전문**(결과 문서·정본이 문자 그대로 승계).

### 8-2. E5 — RA4-2 처방을 쓰지 않는다 + ★이 회차의 정의역 정정(DNA-7)
등록 정규식 = `RuntimeError.*[Ii]nference[ _]?([Tt]ensors?|[Mm]ode)\b`
(`E5_FAMILY_RA4_2_2026-09-14.md` 실측, 감사자 재현: `grep -E`/`grep -P`/`python re` **12/12·12/12·12/12**,
`torch.OutOfMemoryError` **3엔진 전부 미발화** ⇒ 진짜 H1 증거를 가로채지 않는다).
★RA4-2의 처방은 GNU `grep -E`에서 **4/12**, 단일 대안 형태면 **0/12** = **실효 항등식**(게이트 #238);
놓친 문자열도 1종이 아니라 **3종**이었다.
★**정의역 정정**: `none` = `nullcontext`는 inference tensor를 만들지 않으므로 **DN arm에서 E5는
구조적으로 발화 불가**다 ⇒ 이 회차 E5의 정의역은 **채점 TD boot**이다. 따라서 승계된 처분
("게이트 실패가 아니라 `no_grad`로의 스코프 변경 사유")은 **채점 arm의 가드를 바꾸는 두 번째 축
이동**을 뜻한다 — 발화하면 **이 회차를 중단하고 별도 사전등록으로 넘긴다**(자동 스코프 변경 금지).
한계 승계: `\b`는 **POSIX ERE 밖**이며(`E5_FAMILY…§6`) Y6(`ambiguous_autogradother_kernel`의
`InferenceMode` 힌트)의 처분은 **미등록**이다 — 줄 단위 스캔에서는 안전하나 줄을 합치는 로거에서는
발화한다. 이 회차는 줄 단위 스캔만 사용한다.

§0-b의 축 열거에 **노드/물리 GPU(4번째 축)** 를 포함한다(게이트 #233).

## 9. 인용 금지 / 필수 병기 (rev1 승계 + 번호 정정)
- **DN-1**(인용금지) "DN이 907959를 재현했다" — DN arm ≠ 907959 엔진(§1-2).
- **DN-2**(인용금지) 토큰 비매칭 피크 비교(§5-2).
- **DN-3**(인용금지) 가드 실현 퍼센트를 task 커버리지로 읽기(§4-2, A8179-3).
- **DN-4**(필수병기) "이 회차는 DN n=1이다"(§6-4).
- **DN-5**(필수병기) 하네스 sha가 908179에서 1개 움직였다(§2).
- **DN-6**(필수병기) `PREREG_RERUN` sha 드리프트 2값(§0-c).
- **DN-7**(필수병기) "이 트랙은 이 사전등록 이전에 3 job을 소비했고 그중 1건(908020)은 등록 밖이었다"(C4).
- **DN-8**(인용금지) `NOT-RECOVERED`를 "수리가 불필요했다"로 읽기.
- **DN-9**(필수병기) §1-2 전문.
- **DN-10**(인용금지) 이 회차의 어떤 수치도 성능 주장에 쓰지 않는다.
- **DN-11**(필수병기, rev1의 오번호 `DN-13` 정정) §0-c의 앵커 부패 — 과거 판정서의 줄 인용은 그
  sha에 대해서만 참이다.
- **DN-12**(인용금지, 신규) *"rev1의 V 게이트가 반공허성을 보장했다"* — rev1의 V로는
  `RECOVERED-*`가 **도달 불가**였다(DNR-V1). rev1을 인용할 때는 반드시 이 사실을 병기한다.
- **DNR-V1…DNR-V9**(승계, 문자 그대로) — `VERDICT_dnone_rules_2026-09-14.md` §6.
