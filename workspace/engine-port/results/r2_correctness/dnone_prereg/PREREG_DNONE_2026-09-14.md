# 사전등록 — D-none: grad-guard 축 단독 이동 대조 arm (2026-09-14, rev1)

> **상태**: claims-auditor **규칙층 감사 대기**. `GO`(또는 `GO-with-caveats`) 이전에는
> sbatch 금지. 제출은 추가로 §4-4의 **OVERRIDE + 사용자 승인**을 선행조건으로 한다.
> 작성 = 메인 세션(감사자 아님). 구현 = engine-porter(§2). GPU 지출 예정 **1 job ≈ 0.197 GPU-h**.

## 0. 이 회차가 묻는 단 하나의 질문

> 같은 (모델·백엔드·ctx·split·노드·물리 GPU·엔진 `src`·채점기·계측 플래그)에서
> **`PDMUX_WORKER_GRAD_GUARD`만** `inference_mode`(수리) → `none`(수리 前 가드 의미론)으로
> 이동시키면 job 907959의 OOM이 되돌아오는가?

`audit_908179_2026-09-14/VERDICT.md` §8.5가 등록한 **`PLAUSIBLE(조건부)` →
`CONFIRMED(scoped)`의 유일한 조건**이다. 하지 **못하는** 것은 §6에 전수 등록한다.

### 0-a. 이 회차가 **아닌** 것
- Claim D 선결을 닫는 회차가 **아니다**(§6-5). **P2 착수와 무관**(블로커 = λ0 `NO-GO` ·
  W4 λ\* 실측 부재 · 게이트 #6, 전부 불변 — NP-8 승계).
- 성능 측정이 **아니다**. 이 회차의 어떤 수치도 TTFT/ITL/throughput/goodput 주장에 쓸 수 없다(RR-1).
- **`a9cd8dd` 커밋 전체의 대조가 아니다** — `none`은 **가드만** 되돌린다(§1-2, 필수 병기 **DN-9**).

### 0-b. 스코프 튜플 (이 문서의 모든 문장에 붙는다)
> (model `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base` = `NemotronHForCausalLM`, 56층,
> `n_groups=8` · backend `flashinfer` **0.6.10 이라고 보고한 설치본** · ctx **16384** ·
> `mem-fraction-static 0.82` · `max-running-requests 48` ·
> `--disable-radix-cache --chunked-prefill-size -1 --disable-overlap-schedule --random-seed 1` ·
> split **fixed D44**(green stream index 4, realized 64/44 SM) ·
> **cudagraph ON — decode 한정**(908179 실측: prefill 41/41 `cuda graph: False`, 양 arm) ·
> verdict rule **v2**(2026-09-11) · 채점기 `ec355e171a66d68eab1300edcc7616a693cd55aa2c3fb5c251ee4b42eac50d30`
> (**무수정 승계**) · client `95e10b492ed6239d83cae20ad7898bdbbcce1713a37d42c2bfa111f7347ec367`(무수정) ·
> **하네스 `ab55c07cc096bd4cdfe6d484fc9a983302561e03232f641c5994298bdc183bd3`**
> (← 908179의 `f39b167bb6bd713af168abfbe3495caee099a6c0087e41b05e70b9a8d8ac0403`에서 이동, §2) ·
> 엔진 `src` **무변경**(908179와 동일, `git status --porcelain -- .../src`가 빈 줄일 것) ·
> `R2C_EXPECT_{MODEL,BACKEND,CTX}` 3중 대조 ON · `R2C_ORDER="L TD L TD DN"` ·
> **`R2C_GUARD=none`** · `R2C_INSTRUMENT=0` · `PDMUX_MEM_TELEMETRY=1` ·
> `PDMUX_TRACE_FORCE_PREFILL=1` · seed 1 · `--n-probes 8` · A100-SXM4-80GB 108 SM 1장 ·
> **노드/물리 GPU = 실행 후 기록**(게이트 #233 — 축으로 등록하되 값은 사전 고정 불가))

### 0-c. 승계 문서 고정 (sha256)
| 문서 | sha256 |
|---|---|
| `audit_908179_2026-09-14/VERDICT.md`(§8.5 = 이 설계의 출처, A8179-1…7·A8179-P1…7) | `86627a05354ff28b3b32a7764e85f84384d0ac94a7b668b3b2b6ed8aa2a79f7a` |
| `audit_907959_2026-09-13/VERDICT.md`(N-1…13) | `f544544fc973e0abc2d1aa97905e4e005d47e657c30e64d9ea7717d3a48ddc71` |
| `audit_907959_2026-09-13/DIAGNOSIS_true_dual_oom.md`(H1 · `R(T)`의 출처) | `8c857bede1576d9e8042d759889aec3a9f034ed6167fea4093dd947bb9565bee` |
| `audit_908020_2026-09-14/VERDICT.md`(A908-1…7 · C1–C5) | `dc19d1183564591a6394d08432f0c7414d33332de68e55bf403a757ddf8c12ea` |
| `rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md`(등급 철회, **RRC-1…13 유효**) | `118bb45789baf117cc9a1460c8ebfb487088e0d86d81ca5ab9316afc5c412355` |
| `rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md`(RA3-1…12) | `d8281866f3b06f841aa0eb3b00a6b6cf93cd9198d505a1ca2e1ca8ee5ed304c7` |
| `rerun_prereg/VERDICT_rerun_rev4_2026-09-14.md`(RA4-1…12) | `7fdc7ab0fa58a18176520e5ae4d1dcfc4a1ab83449ea34eec8f4e0a7917c4506` |
| `CARRYFORWARD_INVENTORY_2026-09-14.md`(승계 98건 계수표) | `51b7b016f404ef736a8baa21ed9997116523a9beb31d45aca4cc48ae505be6ae` |
| `E5_FAMILY_RA4_2_2026-09-14.md`(E5 계열 재유도 + 3엔진 표) | `9214e62e83871a5cc934c38401e8b5b528b745ed6a8f25f5b9e9bc8089dfa53a` |
| `e5_family_probe.py`(위 표의 실행체, fail-closed) | `c5a063f588e91d0274ecb746bc832037a223cc831fe02201491fa012d5da3a30` |
| `tests/test_r2_correctness_dnone.py`(36 테스트, 변이 5앵커 유일성 포함) | `4209e572c6fa8a0c7a51439e3307b8aa11e8c2ae025e9857d40f4530b45766fd` |
| `rerun_prereg/PREREG_RERUN_2026-09-13.md` | ★**두 값 등재**(아래) |

★**sha 드리프트 등재 (이 회차가 반드시 적어야 하는 것)**
`PREREG_RERUN_2026-09-13.md`는 **908179 실행 시점 `1e421391a97293773ff676a0ab4f24e06c269a68a7ba69fba769c5f9d22923b0`**,
**현재 `fa304128ba1d053ca83fec4a3a7fd81320eaca57b9549f1609f1ce06bf7bda41`** 이다. 2026-09-14
RA4-9 정정이 `:1169`에 **dated 追記를 덧붙여**(본문 삭제 0) sha를 이동시켰고, 그 결과
`audit_908179_2026-09-14/VERDICT.md:5`가 핀한 값과 **현재 파일이 불일치**한다.
이는 **job 908179의 `PASS` 채점을 소급 변경하지 않는다**(채점은 §3-b C1 본문 `:391`을 썼고
정정 대상은 §9-4 대조표의 라벨 문안이다). 이 회차의 `provenance.txt`는 `*_prereg/*`를
glob으로 전수 수집하므로(`r2_correctness.sbatch:365-369`) **현재 값이 산출물에 남는다 —
은폐 금지**(C4 승계의 형태).

★**앵커 부패 등재(신규 DN-13)**: §2의 하네스 변경으로 `audit_908179_.../VERDICT.md` §8.5의
`:216`, rev4 §0 표, `job_908179/provenance.txt`의 줄 인용은 **그 sha에 대해서는 여전히 참**이나
**현재 파일에 대해서는 거짓**이다. 이동 결과: unset 루프 `:216 → :294`, `ORDER` case
`:507-513 → :608-619`, provenance 블록 `:285/:327 → :330/:369`, scope guard `:314-325 → :397-408`.
현재 파일을 인용할 때는 새 줄을, 과거 판정서를 인용할 때는 그 sha를 병기한다.

---

## 1. 처치 — 단 하나의 축

### 1-1. 무엇을 이동시키는가
`PDMUX_WORKER_GRAD_GUARD`: `inference_mode`(908179가 돈 엔진 기본값) → **`none`**.
엔진 구현은 **이미 있다**(`src/multiplex/multiplexing_mixin.py:119-123`, `none` → `nullcontext`;
해석 `:127-144`). **엔진 `src`는 이 회차에서 한 줄도 바뀌지 않는다.**

### 1-2. ★`none`이 되돌리는 것과 되돌리지 않는 것 (필수 병기 **DN-9**)
커밋 `a9cd8dd`는 **가드 수리 + arm 대칭 메모리 계측**을 함께 담았다. `R2C_GUARD=none`은
**가드만** 되돌린다 ⇒ **DN arm은 "907959의 엔진"이 아니다.** 907959에는 없던 계측이 켜져
있고(바로 그 때문에 피크 축이 이 회차에서 측정 가능하다), 그 계측의 피크 리셋은
카운터 전용(`reset_peak_memory_stats`)이라 할당을 바꾸지 않는다. **"DN이 907959를 재현한다"는
문장은 금지**하고, 쓸 수 있는 것은 **"907959의 가드 의미론"** 까지다.

### 1-3. 설계 선택 — 정본 2곳의 불일치를 이 문서가 해소한다
| 안 | 출처 | 비용 | 판정 |
|---|---|---|---|
| A: 채점 `L TD` × **2 job** | 감사 §8.5 | 0.10–0.17 GPU-h ×2 | **기각** — 노드/물리 GPU(게이트 #233)가 또 움직인다. 907959→908179 귀속이 닫히지 않은 직접 원인을 반복한다 |
| B: 비채점 진단 boot(warm-up 패턴) | `PREREG_RERUN` C8 | ≈0.025 GPU-h | **기각** — warm-up은 `PDMUX_TELEMETRY_PATH` 없이 뜨므로 **피크 축과 가드 실현 축이 둘 다 사라진다** |
| ★**C: 같은 job 안의 비채점 진단 boot + telemetry 부여** | 이 문서 | **0.197 GPU-h** 1 job | **채택** |

**C의 근거 3개**
1. **대조 arm과 수리 arm이 같은 job·같은 물리 GPU·같은 sha에 있다** ⇒ 가드 외 축이 안 움직인다.
2. **채점기를 건드리지 않는다.** `r2_correctness_check.py:366`이 `boots.txt` 라벨만 열거하고
   모든 파일 open이 label f-string이다(**glob/listdir/walk 없음** — 실측 확인) ⇒ `boots.txt`에
   없는 `DN1`은 채점기에 **보이지 않는다**. C8이 지적한 "scored로 넣으면 무조건 `FAIL`"을
   **설계로** 회피한다. ★**따라서 이 회차는 "예측된 `FAIL`은 정보"라는 문자 등록이 필요
   없다** — 게이트 라벨은 여전히 채점 4 boot만의 함수다(F-n3가 이를 실측한다).
   부수: `x1_cross_job_compare.py`는 `LABELS=["L1","TD1","L2","TD2"]` 하드코딩이라 DN이 보이지
   않고, `BOOT_FAILURES.txt`·`gen_DN1.json`에는 DN 실패 기록이 **설계대로 남는다**(채점 입력 아님).
3. **양성대조가 이미 엔진에 있다**(§4-2) — 노브가 실현됐는지를 제출 라인이 아니라
   **아티팩트로** 확인할 수 있다(게이트 #176).

### 1-4. 순서와 boot 수 — 왜 `L TD L TD DN`인가 (구속조건이다, 취향이 아니다)
- verdict rule v2의 귀무대조 조건이 문자로 **"두 L boot"·"두 within-arm O 쌍"**
  (`r2_correctness_check.py` 헤더 `:67-78`)이므로 **채점 순서를 줄이면(`L TD DN`) 게이트
  라벨이 구성상 해석 불가**가 된다.
- `BASEPORT = 30000 + (job % 5000) * 6`이 **포트 6개만** 예약한다(warm-up +0, boot n은 +n)
  ⇒ **boot 5개가 상한**이고 `L TD L TD DN`이 정확히 그 경계다. 6번째 boot은 다음 job의
  warm-up 포트를 침범한다(과거엔 **조용한** 교차-job 충돌). engine-porter가 `n_arms ≤ 5`
  **fail-closed 검사**를 넣었고(§2 항목 8) 이 문서는 그것을 **등록된 계약으로 채택**한다.
- ⇒ **DN은 이 회차에서 n=1이다**(§6-4). n=2는 (i) 2 job(노드 축 이동) 또는
  (ii) `L TD DN DN`(게이트 포기) 뿐이며 **둘 다 이 회차에서 채택하지 않는다.**
- **DN은 마지막**에 둔다. 907959는 OOM 뒤 boot이 회복됨을 보였으나(TD1 OOM 후 L2 정상),
  마지막 배치는 죽은 서버 잔재가 채점 boot에 닿을 가능성을 **구조적으로** 0으로 만든다.

---

## 2. 하네스 변경과 그 대가 (등록 = 구현 계약)

구현(engine-porter, 커밋 전): `r2_correctness.sbatch` 573 → 684줄.
| 계약 | 위치 |
|---|---|
| 1. `ORDER` 어휘에 `DN` 추가(`LABEL=DN<n>`), env = 채점 TD boot **+ 가드 하나** | `:611-618`(가드 재수출 `:618`) |
| 2. `R2C_GUARD` **fail-closed** 열거(`inference_mode|no_grad|none` 외 `exit 2`; `DN`인데 미설정 `exit 2`) | `:259-271` |
| 3. `DN`은 **`boots.txt`에 쓰지 않는다** → `diag_boots.txt` | `:624-629` |
| 4. `DN`은 `tel_DN1.jsonl`·`green_DN1.json`을 받고 채점 boot과 **같은 client 호출** | `:611-655` |
| 5. `ORDER` 기본값·`R2C_GUARD` 미설정 경로 launch argv **바이트 동일** | 아래 증명 |
| 6. `provenance.txt`에 `r2c_guard`·`guard_applied`·`dn_boots`·`order` 기록 | `:350-351` |
| 7. `R2C_EXPECT_*` 3중 대조 유지 | `:397-408` |
| 8. `n_arms ≤ 5` 포트 예산 fail-closed(**명세 초과분, 채택**) | `:280-290` |
| 9. **채점 L/TD boot은 항상 엔진 기본 가드**(가드는 DN에만) | `:341-351` 주석 + `:617-618` |

**argv 바이트 동일성의 증명 방법(손으로 쓴 기대값이 아니다 — 그것 자체가 항등식이다)**
`test_r2_correctness_dnone.py`가 (i) `git show 83d8cb9d…:…/r2_correctness.sbatch`의 sha256이
`f39b167b…`(= `job_908179/provenance.txt`의 값)임을 확인하고 그 blob에서 908179의 38줄 boot
루프를 추출한 뒤, (ii) **옛 블록과 현 블록을 같은 절대 `$OUT`에서 각각 실행**해
(`env`/`server_cmd`/`wait_health`/`stop_server`/`kill`/`timeout` 스텁)
`argv_<label>.txt`·`client_<label>.txt`·`boots.txt`·`diag_boots.txt`를 **원시 바이트로** 비교한다
(`ORDER` ∈ {`L TD L TD`, `L TD`, `TD L`, `L TD TD L`}, 정규화·경로 치환 없음).
기본 경로에서 바뀐 것은 **job 로그 배너의 `scored=<0|1>` 한 토큰뿐**이며 저장소 전체에서
`########## boot=`를 파싱하는 소비자는 없다. 기존 픽스처
(`test_r2_correctness_instrument.py`+`ctx`+`scope_guard`, 62 테스트)는 **무수정 통과**.

**★변이 검사 (교훈 53 — 실제 실행 출력)**
- (a) 가드 재수출(`:618`) 제거 ⇒ `FAILED (failures=7)`
  (`test_dn_env_is_the_td_env_plus_exactly_the_guard`, `…_passed_through_verbatim` 3값,
  `test_dn_carries_every_scored_instrument`, `test_mutant_a_*`, `test_mutant_dropping_the_point_of_use_guard_check`)
- (b) `DN`을 `boots.txt`에 쓰기(`:627`) ⇒ `FAILED (failures=4)`
  (`test_dn_label_never_reaches_bootstxt`: `'L1 TD1 L2 TD2 DN1 ' != 'L1 TD1 L2 TD2 '`,
  `test_dn_label_goes_to_the_diagnostic_list`, `test_boot_failure_does_not_stop_the_remaining_boots`, `test_mutant_b_*`)
- ★**변이 테스트 자신의 결함이 실행으로 드러났고 수리됐다**(게이트 #9의 재발): (a)의 최초
  앵커가 **같은 블록의 주석에도 존재**해 실제 export가 지워진 뒤에도 주석을 변이시키고
  **통과**했다(6 failures, `test_mutant_a_*` PASS). 수리 = 대입문 전체를 앵커로 + 파일 내
  5개 변이 앵커 전부에 `assertEqual(block.count(anchor), 1)` 유일성 단정.
- 죽은 서버 teardown 실측: 수거된 PID에 `stop_server` 호출 ⇒ **rc=0, 21초, hang 없음.**

**대가(등재)**: 하네스 sha `f39b167b…` → `ab55c07c…` ⇒ **908179와 이 회차 사이에도 하네스 축이
1개 움직인다.** 이 회차가 닫는 것은 **가드 축**이고 **하네스 축은 닫지 않는다.** 이동분은
(i) `DN` 분지, (ii) `R2C_GUARD`/포트 예산 검증, (iii) provenance 2줄이며 **채점 경로에 닿지
않음**을 **F-n3가 같은 job 안에서 실측**한다(argv 바이트 동일성은 CPU에서 이미 증명).

**기타 등재(구현에서 나온 사실)**
- 거부된 실행은 **`provenance.txt`를 남기지 않는다**(노브 해석이 provenance·`sync_engine_tree.sh`
  보다 앞) — 오타 비용은 초 단위이고 이유는 SLURM `.err`에만 있다. scope guard와 다른 거동.
- 하네스는 엔진보다 **엄격**하다: 엔진은 `.strip().lower()`하지만 하네스는 안 한다 ⇒
  `"none "`·`NONE`은 `exit 2`. `DN`이 없어도 `R2C_GUARD` 값은 검증한다(미등록 값이
  provenance에 찍히면서 엔진은 기본값으로 서빙하는 사태 방지).
- 소문자/오타 arm 토큰은 **여전히 조용히 skip**(선재 `*) echo "bad arm"; continue`) ⇒
  `R2C_ORDER="L TD dn"`은 DN boot을 **하나도** 만들지 않는다. **틀린 boot은 못 만들고 빠진
  boot만 만든다.** 이 채널은 §4-3의 `dn_boots=1` 술어가 잡는다.
- `diag_boots.txt`는 **append-only**(`boots.txt`와 달리 `: >` 절단 없음). SLURM은 job마다
  새 `job_<id>/`라 실질 무해하고, 같은 `job_local/`에서의 반복 로컬 실행만 누적된다.

---

## 3. 판정 규칙 — 채점기 무수정 승계
`r2_correctness_check.py`(`ec355e17…`) · verdict rule **v2** 무수정. 채점 `ORDER`의 앞 4 boot은
908179와 동일 ⇒ **게이트 라벨의 의미가 908179와 같다.** 진단 boot은 채점 입력이 아니다.

---

## 4. 채점 前 게이트 (결과무관·기계적 — 채점보다 먼저 평가)

### 4-1. V — 공허성 게이트 (예보가 아니다)
`srv_DN1.log`에서 `#new-token ∈ [10036, 10137]` split-prefill 배치가 **형성**됐는가.
- 형성 안 됨 ⇒ **`UNREALIZED`**: 이 회차는 H1에 대한 증거를 **만들지 못했다**(측정 실패, 게이트 #21).
  보고 의무 = 형성된 최대 토큰 수와 전 배치열.
- ★이 게이트가 없으면 "OOM이 안 났다 ⇒ 수리가 원인"이 **공허하게 참**이 된다 —
  F-a1이 죽은 바로 그 형태(RRC-4/RA3-1).

### 4-2. G — 노브 실현 양성대조 + 같은 job 음성대조 (필수)
- 양성: `tel_DN1.jsonl`에 `{prefill|decode}_worker_grad_enabled = true` **≥1건** AND
  대응 `_worker_inference_mode = false`.
- 음성: 같은 job의 `tel_TD{1,2}.jsonl`에 `*_worker_grad_enabled = true` **0건**(908179 실측 0건).
- 양성 거짓(또는 두 필드가 `null`로만 남음 — 첫 worker task 이전에 사망한 경우) ⇒
  **`NO_KNOB`**: 처치 미검증이며 **H1의 반증이 아니다**(측정 실패).
- 근거: `none`=`nullcontext` ⇒ `_r2_record_worker_guard`(`multiplexing_mixin.py:724-742`)가
  그 두 값을 worker 스레드 **안에서** 기록한다. 목표값 `worker_grad_guard="none"`도 전 스냅샷에 실린다.
- ★한계 승계(**A8179-3**): 이 기록은 **스냅샷 단위 last-write-wins**이며 task 전수 커버리지가
  아니다 ⇒ 정보량 있는 술어는 **"`true`가 ≥1건 있다"(DN) / "0건이다"(TD)"** 뿐이다.
  퍼센트를 커버리지로 인용 금지.

### 4-3. SCOPE — C1 승계 + 신규 술어 2개
908179의 (1)–(11)에 더해 **(12)** `provenance.txt`가 `r2c_guard=none guard_applied=yes`,
**(13)** `dn_boots=1` 및 `order=[L TD L TD DN]`. 불일치 ⇒ **`NO_VERDICT_SCOPE`**
(등록 밖 실행 — 908020 재발 방지). ★(9)·(11)의 등록된 한계(RA4-3·RA3-8: "GPU 0:" 확인은
src 청결의 충분조건일 뿐이고 나머지 축은 기본값 우연 일치에 의존)는 **그대로 승계**한다.

### 4-4. 제출 선행조건 (이 회차 신설)
1. ★`presubmit.py`가 현재 **2 BLOCK**(`m4r_confinement/reachability_spec.json →
   SINGLE_LABEL_FORCED`, `tc1_model_attrib/reach_spec_rev3_A.json → RESTRICTIONS_INERT`).
   engine-porter 판정 = **stale이 아니라 살아 있는 설계 결함**(후속 spec 없음 ⇒ `superseded_by`로
   닫을 대상이 없음) ⇒ **범위 한정 OVERRIDE 문서 + 사용자 명시 승인**이 선행돼야 한다
   (전례 `results/cp_baseline/OVERRIDE_VPROBE_SUBMIT_2026-09-01.md:10-11,40,42` ·
   `OVERRIDE_P1_SUBMIT_2026-08-28.md:44,81`). ★**도구 차단범위 축소는 두 전례가 명시 금지하며
   이 회차도 금지한다.**
2. `check_doc_facts.py` 0 위반 · `check_line_citations.py --check --all` 0 위반 ·
   `check_citation_stops.py` 0 위반 · `unittest discover` **681/681 OK**.
3. `#SBATCH --comment="field=efficientai;appl=pytorch"` 존재 확인(뉴론 정책).

---

## 5. 예보 (실행 전 고정)

`V ∧ G ∧ SCOPE`가 **전부 참**일 때에만 아래를 채점한다. 하나라도 거짓이면 그 라벨
(`UNREALIZED`/`NO_KNOB`/`NO_VERDICT_SCOPE`)이 이 회차의 결과이며 **H1 등급은 불변**이다.

### 5-1. F-n1 (1차 — OOM 재현) · 결과별 라벨과 쓸 수 있는 문장
907959 서명 3요소: ① 요청 크기 **198.00 MiB**, ② `allocated` **77.25 GiB** 근방,
③ 스택 프레임 동일. **전부 일치를 요구하지 않는다** — 몇 개가 일치하는지 그대로 보고한다.

| 결과 | 라벨 | 등록된 문장(이것 말고 쓸 수 없다) |
|---|---|---|
| 그 배치에서 OOM + 서명 3/3 | **`RECOVERED-STRICT`** | "위 스코프 튜플에서 **가드 축 단독 이동**이 907959의 OOM을 그 서명까지 재현한다." |
| 그 배치에서 OOM + 서명 1–2 | **`RECOVERED-WEAK`** | "위 스코프 튜플에서 가드 축 단독 이동이 **같은 크기의 배치에서 OOM을 재현한다**(서명 부분 일치: 일치 항목을 명시)." |
| 배치는 형성·완주, OOM 0건 | **`NOT-RECOVERED`** | "가드 축 단독 이동은 이 **1회** 실행에서 OOM을 재현하지 못했다." ★**H1에 대한 반대 증거**이며 `PLAUSIBLE(조건부)`의 **강등 검토 대상**이 된다(등급 변경은 결과 감사 소관, 이 문서가 미리 정하지 않는다). |
| 다른 배치에서 사망 | **`OTHER-BATCH`** | 배치열·토큰수·스택만 보고한다. **서술 문장 없음**(RRC-8형 보수적 결손). |

### 5-2. F-n2 (피크 축)
**토큰 수를 맞춘 epoch에서만** `peak(DN1) − peak(TD*)`를 본다. 예보: **양수**,
크기는 `R(T) = T × 1.2750 MiB`의 `0.3×` 이상. 맞춘 epoch이 없으면 **`UNMATCHED`**(비교 없음).
- ★**A8179-1/2 승계**: `Δ`는 배치 크기 차만으로도 생긴다 ⇒ **토큰 비매칭 비교 인용 금지.**
- ★`R(T)`는 "그래프가 prefill 전체에 유지된다"는 **가정 위의 산정**이다(`DIAGNOSIS:88`) ⇒
  **크기 불일치는 H1을 반증하지 않고 그 가정을 반증한다. 1차 정보는 부호뿐이다.**

### 5-3. F-n3 (설계 자기검증)
채점 4 boot의 `verdict.txt` = **`PASS`**, S/O 불일치 0. `FAIL`이면 **진단 boot이 채점을
오염시킨 것**이므로 **설계 C 자체가 반증**되고(`DESIGN-REFUTED`) 이 회차는 H1에 무증거다.

### 5-4. 이것이 사후 예보가 아닌 이유
양 끝점이 **이미 관측돼 있다**: 907959(가드 前 의미론) = TD 천장 6,245 / 10,125 배치에서
결정적 사망 / 198.00 MiB·77.25 GiB. 908179(수리 後) = 완주 /
`peak(TD1)=peak(TD2)=peak(L2)`(차 0 B). 이 회차는 **그 사이의 재현 시험**이다.

---

## 6. 이 회차가 **하지 못하는** 것 (실행 전 고정)

1. **기전은 닫히지 않는다.** `RECOVERED-*`도 "가드 축이 지배 원인"까지다.
   "`mixer2_rms_norm_gated.py:97`의 bare Parameter 경로가 그 메모리를 만든다"는 층별 retention
   측정 없이는 `PLAUSIBLE`(`DIAGNOSIS:88`).
2. **하네스 축은 닫히지 않는다**(§2). F-n3와 CPU 바이트 동일성이 그것을 **국소화**할 뿐이다.
3. **`a9cd8dd` 전체의 대조가 아니다**(§1-2, DN-9). DN arm ≠ 907959 엔진.
4. **DN n=1.** boot 간 분산 미측정(RRC-9 승계). n=1이 정보량을 갖는 근거는 **907959의
   결정성**(TD 2/2가 같은 배치·같은 크기·같은 스택)이라는 **사전 관측이자 가정**이며
   이 회차의 측정이 아니다. 포트 예산(§1-4)이 같은 job 안의 n=2를 구조적으로 막는다.
5. **Claim D 선결은 하나도 닫히지 않는다.** P2 블로커 3개 불변(NP-8).
6. **소급 금지 불변(RR-3)**: 907100/907456/X1은 Zamba2 기판이라 이 기전에 면역이었고
   autograd 부기의 arm 비대칭 오버헤드는 **미측정** — "오염됐다"도 "깨끗했다"도 쓸 수 없다.
7. **성능 주장 전면 금지(RR-1)**: `PDMUX_MEM_TELEMETRY=1`은 짝지은 on/off 측정이 없다.

---

## 7. 예산 (실측 기반 재산정 — C8의 수치를 정정한다)

- **DN boot 1개 = 1.80분(108초) = 0.030 GPU-h.** 같은 기판의 실측에서 도출: 907959 TD1
  launch 22:07:36 → healthy 22:08:07(boot 31초) → OOM SIGQUIT 22:08:56 → 다음 launch 22:09:24
  ⇒ boot **슬롯 108초**. 908179의 정상 boot 슬롯은 114/111/110초 ⇒ OOM 여부와 무관하게 ~1.8–1.9분.
  ★**C8/rev2의 "≈1.5분 ≈0.025 GPU-h"를 정정한다**(서버 로그 구간 1.3분만 세고 teardown 20초 +
  launch ~8초를 빠뜨렸다).
- **job 총계**(모형 `total ≈ 153 + 111 × n_boots`초, 908179 `sacct` 596초를 1초 오차로 재현,
  `INSTR=0`): `L TD L TD DN` = **708초 ≈ 0.197 GPU-h**.
- **구조적 최악(DN 1개)** = 1,400초 = health 96×5초 + client `timeout 900` + teardown 20초 —
  OOM이 SIGQUIT 대신 **hang**으로 나타날 때만 도달. `--time` 02:30:00 유지(여유 충분).
- **운영 오류 재실행 예산 = 0**(C3 승계). `NO_VERDICT_SCOPE`·`NO_KNOB`·`UNREALIZED`도
  **소비로 계상**하고 장부에 등재한다.
- ★**장부 의무**: 결과 서술은 R2 correctness 트랙 누적(**현재 1.107500 GPU-h** = 등록 0.955000 /
  **등록 밖 0.152500**, `sacct` 기준)에 이 job을 더해 적고 **908020(등록 밖)을 은폐하지
  않는다**(C4 승계).

## 8. 승계 (RA4-6 누수 차단)

`CARRYFORWARD_INVENTORY_2026-09-14.md`(`51b7b016…`)의 **98건**(96 + 전사의무 2; 계열
NP10·NPC10·D13-i 2·N13·A908 7·RRC13·RA3 12·RA4 12·RR19)을 **항목 수 대조로** 승계한다.
★RR-14가 요구한 **44건은 이 문서의 강제 범위가 아니다** — 인벤토리 §10 계수표와 이 문서의
목록 길이를 맞추며, 불일치 시 **제출 금지**.

**E5 계열은 RA4-2의 처방을 쓰지 않는다.** 등록 정규식 =
`RuntimeError.*[Ii]nference[ _]?([Tt]ensors?|[Mm]ode)\b`
(`E5_FAMILY_RA4_2_2026-09-14.md` 실측: 3엔진 `grep -E`/`grep -P`/`python re` 전부 **12/12**,
음성 6/7 침묵). ★RA4-2의 처방은 GNU `grep -E`에서 **4/12**, 단일 대안 형태면 **0/12** =
**실효 항등식**(게이트 #238); 놓친 문자열도 1종이 아니라 **실측 3종**이었다.
E5 발화 시 처분은 승계 그대로 — **게이트 실패가 아니라 `no_grad`로의 스코프 변경 사유**.

§0-b의 축 열거에 **노드/물리 GPU(4번째 축)** 를 포함한다(게이트 #233).

## 9. 신규 인용 금지 / 필수 병기
- **DN-1**(인용금지) "DN이 907959를 재현했다" — DN arm ≠ 907959 엔진(§1-2).
- **DN-2**(인용금지) 토큰 비매칭 피크 비교(§5-2).
- **DN-3**(인용금지) 가드 실현 퍼센트를 task 커버리지로 읽기(§4-2, A8179-3).
- **DN-4**(필수병기) 어떤 결과 서술에도 **"이 회차는 DN n=1이다"**(§6-4).
- **DN-5**(필수병기) **하네스 sha가 908179에서 1개 움직였다**(§2).
- **DN-6**(필수병기) `PREREG_RERUN` sha 드리프트 2값(§0-c).
- **DN-7**(필수병기) "이 사전등록 이전에 이 트랙은 3 job을 소비했고 그중 1건(908020)은
  등록 밖이었다"(C4 승계).
- **DN-8**(인용금지) `NOT-RECOVERED`를 "수리가 불필요했다"로 읽기 — 등급 변경은 결과 감사 소관.
- **DN-9**(필수병기) §1-2 전문.
- **DN-10**(인용금지) 이 회차의 어떤 수치도 성능(TTFT/ITL/throughput/goodput) 주장에 쓰지 않는다.
