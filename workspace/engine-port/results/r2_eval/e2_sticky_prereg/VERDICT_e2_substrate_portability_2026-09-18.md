# 판정서 — E2 기판 이식성(등록 유효성) 규칙층 감사 (2026-09-18)

> **지위**: claims-auditor 규칙층 감사 1회차. **GPU 지출 0 · 새 성능 판정 0건 · 파일 수정 0건**
> (이 판정서 생성 제외). 회부 근거 = `PROJECT_STATUS.md` 최상단 배너(2026-09-17)가
> **OPEN USER DECISION**으로 남긴 "E2 등록 격자·λ\*·OVERRIDE 예산은 A100 108 SM 기판
> 기준 — 다른 기판에서 등록된 실험으로 그대로 돌릴 수 있는지 **미결정**(사용자+규칙층
> 감사 사항)".
>
> **이것은 성능 판정이 아니라 등록 유효성 판정이다.** HE0·layer-type 死·정책 순위·
> 게이트 #6/#13/#16·Claim D/E 등급·stake #1 전부 **불변**.
>
> 짝 문서: `handoff-report/gpu_rental_checklist_2026-09-18.md`(같은 날 작성, 이 감사의
> **대상이자 산출물 아님** — 감사자는 그것을 확증 근거가 아니라 감사 대상으로 다뤘다).

## 최종 한 줄

**A100 80GB 1장이라도 "재등록 없이" 실행할 수는 없다 — 단 필요한 재등록의 범위는
"설계 재등록"이 아니라 「이식 追記(A13) + 새 OVERRIDE」이고, 결정 규칙 R1–R6·다섯
추정량·seed·arm 집합·라벨 어휘는 재등록 불요다.**

차단 사유는 **기판(108 SM)이 아니라 하네스 텍스트·예산 집행기·과금 모형 세 가지**다(Q3).
기판 자체에서는 **死因을 만들지 못했다** — A100→A100 변경만으로 등록 판정을 뒤집는 수치를
반전 시험에서 만들지 못했다. 미확정 2건(P7/§8 상수 이식성, R1 OFF 밴드 여유)이 남고
이는 B0–B4 스모크로만 닫힌다.

**등급 자기검사**: `NO-GO`를 내지 않았다(A100→A100에서 판정을 뒤집는 수치를 만들지 못했고,
"찜찜하니 차단"은 하지 않는다). `GO`도 내지 않았다(세 건은 문서 자신의 문언이 새 승인을
요구한다, "그럴듯하니 통과"도 하지 않는다).

## Q1. 기판 의존 전수 열거

구조 사실: **E2-α 설계(대조 arm을 같은 job 안에)가 기판 의존의 대부분을 이미 제거한다.**
OFF/ON이 같은 boot 루프·같은 노드·같은 트리이므로 **R2(paired 차분)·R3·R4·R4″·P13은
호스트 정체성에 면역**이다. 남는 의존은 **절대 문턱**과 **908623 아카이브에서 빌려 온
상수** 두 부류뿐이다.

| # | 구성요소 | 라벨 | 근거 |
|---|---|---|---|
| 1 | 격자 `manual_divisions: [92,16],[64,44],[16,92]` | 재등록필요(합 108 전제) / A100 108 SM이면 기판무관 | `results/longctx_conflict/probes/pdmux_homog5.yml` 헤더가 스스로 `sm_counts = [(108,0),(92,16),(64,44),(16,92),(0,108)]` |
| 2 | idx 매핑 `D44=idx2 · (0,108)=idx4 · (16,92)=idx3` | 재등록필요(idx4의 `(0,N)`은 `get_sm_available()`이 결정) | `e2_realized_mix.py:48-50` |
| 3 | sticky ON 정의(`PDMUX_STICKY_PARTITION=1` → `_sticky_fixed_idx=2`) | **기판무관** | prereg rev3 §4-1. 인덱스 해석이지 SM 산술 아님 |
| 4 | `_build_r2_policy`의 `decode_sms in (16,24,34,44)` 필터 | **기판무관**(격자 유지 시) | `src/multiplex/multiplexing_mixin.py:253-265`. ※`emergency_decode_sms=108`(`:282`)은 하드코딩 A100 리터럴이나 `hybrid` 전용이고 E2는 `fixed` ⇒ 미발화 |
| 5 | offered `a_r4=8.0 · a_r2=3.0 · b_r3=1.15` | **재등록필요** | λ0 `FALLBACK_LADDER`(`lambda0_plan.py:150`)의 rung. 앵커 λ_inf(A)=2.10이 **A100 D44 로그 회귀** `step(B)=19.82+0.5174·B ms`(`:132-133`)에서 나옴 |
| 6 | λ\*(A)≈3.05 · λ\*(B)≈0.696 | **재등록필요** | job 908623(A100). ★단 E2 결정 규칙에는 안 들어가고 셀 배치 근거로만 쓰임(E2C-13) |
| 7 | seed `4386/4162/251/2630` | **기판무관** — ★독립 재현 성공 | `lambda0_plan.choose_seeds` 재실행 ⇒ `[(4386,0.014603),(4162,0.0151),(251,0.017025),(2630,0.018145)]`, prereg rev3 §2와 **4/4 일치**. 순수 CPU 함수 |
| 8 | boot 수 20 · 실행 순서 | 기판무관 | prereg rev3 §2 리터럴 |
| 9 | **P7 `T_cap = 393/428/861 s`** | ★**미결** | `e2_sticky.sbatch:88`. "3 × 908623 OFF 실측 duration" ⇒ 다른 기판 아카이브에서 빌려 온 상수. 3× 여유의 이식 오차 흡수력 미측정 |
| 10 | **예산 3층 + per-boot A=145.5 · B=106.5 s** | ★**미결(수치)** · 재등록필요(집행) | prereg rev3 §8-1 "908623 아티팩트 mtime 실측". 모델 로드·boot·teardown은 호스트 I/O·CPU 의존 |
| 11 | **P14 `WALL_BUDGET_S=10800`** | ★**재등록필요** | `e2_sticky.sbatch:81` + ADDENDUM A4가 "남은 예산 = `--time` 할당(10,800 s) − `SECONDS`" — 리터럴이 **SLURM 할당을 이름으로 참조** |
| 12 | 라벨 어휘 전체 | 기판무관 | `e2_label.py` 전역. 상대·구조 술어 |
| 13 | **R1 밴드 ON>90.0 · OFF<20.0** | ★**미결** | `e2_label.py:33-34`. OFF 절이 절대 문턱이고 A100 908623 OFF 실측(`a_r2` E-qcond 17.65% = 18/102)으로 눈금. 여유 **2.35pp** |
| 14 | R3(ii)/R4′/R4/R4″/R5/E-pact 문턱 | 기판무관 | `e2_label.py:35-40`. 같은 job 안 상대·구조 술어 |
| 15 | R2 부트스트랩 `B=100000 · rng 20260915` | 기판무관 | `e2_label.py:45-46` |
| 16 | 통제 요인(모델·flashinfer·CTX 16384·mem 0.82·max_running 48·cudagraph ON) | **미결** | `max_running=48` 실효에 mamba 캐시 ≥48 필요 ⇒ **80 GB 전제**. 908623 배너 `max_mamba_cache_size=48 / max_total_num_tokens=2722025`. A100 **40GB**면 불성립 |
| 17 | **E2C-1**(노브 1·기전 2, decode SM 108→44 + 드레인 제거) | 재등록필요(문언) / 실질은 기판무관 | "108→44"의 108이 총 SM |
| 18 | **E2C-8′**(추정량 열은 규약 없이 인용 불가) | 기판무관 | ★`e2_realized_mix.py --selftest` 이 머신 실행 ⇒ `SELFTEST_OK 7/7` 재현 |
| 19 | **E2C-21**(shape A 처치 시간의 86–89%가 prefill 유휴) | ★**재등록필요** | ADDENDUM A9. `a_r4` idx4 158.2/183.61 s 등 A100 908623 원자료 계산값 |
| 20 | E2C-6(`a_r2` Q5는 절벽 위, OFF TTFT p95 1.806 s vs SLO 3.0 s) | 재등록필요 | A100 실측 절대 지연 |
| 21 | E2C-13(네 seed가 실현 offered를 2.996/2.955/2.950/2.951로) | 기판무관(계산) / 재등록필요(해석) | prereg rev3 §9 |
| 22 | E2C-10(achieved를 max_running 48이 구속) | 미결 | 항목 16과 같은 조건부 |
| 23 | `e2_sticky.sbatch`의 `ROOT=/scratch/ehmoon/whlee/...` · `module load` · venv 경로 · `#SBATCH` 헤더 | ★**재등록필요** | `:46,:49` + `:2-12`. 클러스터 전용 리터럴 |
| 24 | `#SBATCH --cpus-per-task=8 --mem=100G` | ★**미결(신규 위험)** | `:7-8`. prereg §3 통제 목록에 **CPU 수가 없다** — Q2-(f) |

## Q2. A100 80GB 1장일 때의 적대적 잔여 요인

### 반전 시험 (A100 108 SM 유지, 호스트만 교체)

| 자유 표면 | 민 범위 | 판정 변화 | 근거 |
|---|---|---|---|
| 노드/클러스터 정체성(게이트 #233 확대) | KISTI gpu38 → 임의 대여 A100 | **반전 없음**(R1·R2·R3·R4·R4″·P13) | E2-α상 OFF/ON이 같은 job·같은 GPU(prereg rev3 §0-1 "노드·물리 GPU 축 **불변**"). 노드 교락은 **R6에만** 실리고 R6은 §6이 이미 "판정 아님" |
| **R6**(908623 OFF 셀과 교차 비교) | 다른 클러스터 | 판정 불변(이미 비-판정), **caveat 강화** | 병기 문구를 "노드 축" → "**클러스터·드라이버·패키지 축**"으로 확대 필요 |
| **호스트 속도 → per-boot 부대비용** | A=145.5 s → +Δ | ★**반전 있음: R4″ 소멸** | P14 리터럴 계산: 마지막 boot 직전 등록 경과 6489.6−393.4=**6096.2 s**, REMAIN 4703.8 vs 필요 1061 ⇒ 여유 3642.8 s / 19 boot = **+191.7 s/boot**. **등록 최악 코너(8417.5 s)에서는 여유 1858 s = +97.8 s/boot** ⇒ per-boot 부대비용이 **평균 98 s만 늘어도** `b_r3` seed2 짝이 `UNRESOLVED_BUDGET`이 되고 **R4″가 사라진다**(ADDENDUM A4가 예고한 경로) |
| **R1 OFF 밴드(<20.0%)** | 908623 OFF 실측 → 호스트 변동 | ★**반전 있음**: `STICKY_REALIZES_A` → `OFF_BASELINE_ABOVE_BAND` | `a_r2` OFF E-qcond **17.65%(18/102)**, 여유 **2.35pp**. 인접 rung `a_r3` = **18.67%(45/241)**, 여유 1.33pp(ADDENDUM A5 공시). n=102에서 표본 3개 = 2.94pp ⇒ **밴드는 기판 이전에 표본잡음에도 얇다.** 단 A5가 이 분지에 **별도 라벨**을 등록해 둬 거짓 귀속은 없음 ⇒ **死因 아님, 승계 caveat** |
| bench-only 재스코어 추정량(E2C-16) | — | 반전 없음(재스코어 **금지**) | 그 추정량에서 OFF 여유는 2.35pp가 아니라 **0.85pp**. 게이트 #8로 봉인 |
| P3 `(48, 2722025)` 수준값 | 다른 A100 인스턴스 | 반전 없음 — **단 게이트가 진공** | `e2_sticky.sbatch:270-282`은 job 안 boot 간 일치만 본다. 수준이 통째로 달라져도 `distinct keys = 1`로 PASS(하네스층 판정서 §5에 이미 등재) |
| `e2_realized_mix`의 `idx2 ↔ (64,44)` 대조 | 총 SM 변화 | 반전 없음 — ★**기판 탐지력 0** | 비교 대상 `prefill_sms/decode_sms`는 `get_sm_counts()` = `manual_divisions` **자기 자신** ⇒ 기판이 바뀌어도 idx2는 항상 (64,44) = **기판 변화에 대해 항등식**(교훈 9). 엔진 내부 index↔sm 비동기 탐지라는 원래 목적에는 유효 |
| **realized green-context SM** | target vs realized | ★**미측정** | `e2_sticky.sbatch`가 `PDMUX_GREEN_READOUT`을 **설정하지 않는다**(`green_readout.py` DEFAULT OFF) ⇒ E2는 "44 SM을 실제로 받았는가"를 **스스로 답하지 않는다**. Stage 0 D108(실은 16 SM)·λ0 D44 점유 계열 교훈이 E2에 배선돼 있지 않음 |

### 이식이 새로 지는 위험

**(a) `devtree_manual_edits.patch` 5파일 — 런타임 결정적, P4가 못 잡는다.**
그 중 하나는 pdmux 경로 동작 필수 수리:
`ngram_embedding_info = getattr(forward_batch, "ngram_embedding_info", None)`
(pdmux 경로가 `ForwardBatch`가 아닌 `ModelWorkerBatch`로 호출해 crash하는 것을 방어).
패치 미적용 = **pdmux 경로 AttributeError**(시끄럽게 죽음 — 은닉 아님). 그러나
`e2_sticky.sbatch:246-254`의 P4는 **`N_MANIFEST ≥ 24` + `nemotron_h.py` 존재**만 본다
⇒ **manifest는 기록이지 검증이 아니다**(비교할 기준 digest 미등록).
`triton_backend.py`의 `hybrid_gdn_config → mambaish_config` 수정은 flashinfer 백엔드에서
`TritonAttnBackend`가 생성되는지에 따라 무해하거나 **조용히 틀린 `v_head_dim`**
⇒ **UNDETERMINED**, B2/B4 스모크로만 닫힌다.

**(b) 드라이버·CUDA·패키지.** prereg rev3 §3 통제 목록에 **드라이버·CUDA·torch·flashinfer
wheel 버전이 없다.** `sync_engine_tree.sh`가 스스로 적는다 — "this hashes SGLang's backend
wrapper, **NOT the installed `flashinfer` wheel**". 고정값은 `venv_packages_2026-09-17.txt`.
**within-job 대조(R2/R3/R4/P13)는 면역, 절대 문턱(R1)·P7·§8 상수는 아니다.**

**(c) ★SLURM 소실 → `--time` 하드 캡 소실 (가장 실질적).** OVERRIDE §5는 "초과 시:
`--time` 소진은 **SLURM이 강제**하고, 그 전에 P14가 다음 boot를 시작하지 않는다"라고
**집행 주체를 명시**한다. `bash`로 돌리면 남는 건 P14뿐이고 P14는 **boot 사이에서만**
검사한다. 통과 직후 한 boot가 `wait_health` 최대 **1200 s**(`:169-176`) +
**타임아웃 없는 warmup**(`:333-341`, `timeout` 미적용) + `T_cap` + teardown 15 s를
쓸 수 있다 ⇒ 최악 ≈ **10800 + 1200 + α ≈ 12,000 s ≈ 3.3–3.4 GPU-h**
= **승인된 하드 캡 3.0 GPU-h 초과 가능**. `JOBID="${SLURM_JOB_ID:-local$$}"`(`:59`)
폴백이 있어 실행 자체는 되므로 **더 위험하다(조용히 돈다)**.

**(d) `e2_sticky.sbatch` 편집 = 감사 봉인 sha 변경.** 하네스층 판정서 rev2가
`505e880d12df…`로 봉인했고 현재 파일은 `e30df9f4f61c…`(선행조건 반영, 판정서가 "재감사
불요" 명시 ⇒ 적법한 드리프트). 이식은 **세 번째 sha**를 만든다. 경로·런처만 바꿔도
P4가 기록하는 결정경로 digest가 바뀜 ⇒ 하네스층 **국소 재확인**(전체 재감사 아님) 필요.

**(e) 공유 호스트·클럭 정책.** 다른 테넌트와 CPU·PCIe·전력을 공유하면 per-boot 부대비용이
런 안에서 드리프트한다. E2-α의 OFF→ON 인접 배치는 느린 드리프트를 양 arm에 균등
분배하도록 설계됐으나(sbatch 헤더 "slow drift lands on both arms nearly equally"),
**OFF→ON 방향 비대칭은 등록된 비대칭**(E2C-7)이라 단조 드리프트는 ON 쪽에 실린다
⇒ `LAMBDA_MOVES_*`를 기전으로 읽는 것을 금지하는 조항이 **한 겹 더** 필요.

**(f) ★신규: CPU 코어 핀 소실 — confound 카탈로그 #7 직격.** `--cpus-per-task=8`(`:7`)이
사라지면 `bench_serving` 클라이언트의 코어 예산이 **등록되지 않은 축**이 된다. 이 프로젝트는
이미 "triton TPOT 170–205 ms floor = GIL-client 아티팩트"로 한 번 오도됐다. `a_r4`는
offered **8.0 req/s × 400 prompts**로 격자 중 클라이언트 부하가 가장 높다. **부호는 뒤집지
않는다**(클램프가 양 arm에 동일해 차분을 0 쪽으로 압축) ⇒ `LAMBDA_WITHIN_CI` 쪽으로 미는
**검출력 손실**이고 E2C-7이 그 라벨의 오독을 이미 금지 ⇒ **死因 아님, 신규 caveat +
`--cpus-per-task` 등가 핀(taskset/cgroup) 등록 권고.**

## Q3. 기존 OVERRIDE는 새 기판에서 유효한가 — 문서 자신의 문언으로

**판정: `REFUTED` — 그대로는 유효하지 않다. 새 OVERRIDE가 필요하다.**
단 사유는 "기판"이 아니라 **"예산"**이다.

문언: §3 "이 문서는 **E2 1 job 한정**이며, 재실행·**설계 변경**·다른 셀 집합은 **새
OVERRIDE**를 요구한다." · §7-1 "새 승인은 불필요하다 — 승인된 **설계·예산**이 바뀌지
않았기 때문이다. 단 **설계나 예산이 바뀌면 새 OVERRIDE**다."

**(i) 기판 변경은 문언상 "설계 변경"이 아니다.** prereg rev3 §3(통제 요인)은 "908623에서
한 글자도 바꾸지 않는 것"을 모델·백엔드·CFG·CTX·mem·max_running·플래그·cudagraph·
policy·seed로 열거하며 **노드·물리 GPU·클러스터를 포함하지 않는다.** 이것이 정확히
**게이트 #233이 열려 있는 이유**("세지 않은 축은 닫을 대상으로도 등록되지 않는다",
`CONSENSUS §3` 항목253)다. ⇒ **문언이 기판을 커버하지 못하는 것이지, 커버해서 허용하는
것이 아니다.**

**(ii) 그러나 "예산"은 세 경로로 바뀐다 — §7-1 두 번째 조항 발효.**
- **집행기 교체**: §7이 승인한 것은 "1.803 / 2.338 / **3.0(`--time` 하드 캡)**"이고
  **`--time`은 승인된 상한의 이름이자 집행기**였다(§5). `bash` 실행은 그 집행기를 제거한다
  (Q2-(c), 최악 ≈3.3–3.4 GPU-h). **승인된 하드 캡이 집행되지 않는 상태로 도는 것은
  예산 조항 변경이다.**
- **상수 재산출**: 1.803/2.338은 §8-1의 `A=145.5 · B=106.5 s`에서 나왔고 그 상수는 908623
  **아티팩트 mtime 실측**이다. 호스트가 바뀌면 **사실로서 다른 수**가 된다.
- **과금 모형 전환**: SLURM 무상 할당에서 초과 = 잡 종료, 추가 비용 0. 시간당 과금에서
  초과 = **금전**. §7 승인 시 제시된 "못 사는 것/예산 3층"의 **위험 프로필 자체가 다르다**
  ⇒ 같은 숫자라도 **같은 승인이 아니다**(§5 "비교" 행의 λ0 실지출 1.391 GPU-h도 무상 장부).

**(iii) 부수**: §2-1의 M4R 노브 공유 공시, §4의 인용 금지 조항은 기판과 무관하게 승계 —
새 OVERRIDE는 이것들을 **문자 그대로 다시 실어야** 한다.

## Q4. 소스 사실 독립 검증

저장소에 dev tree 사본이 없어 **이 머신 설치 실물**로 대조: conda env `prefill-alloc`의
**`sglang-0.5.10.post1`**, 대조군 `mixer-alloc` 동버전, 추가 대조군
`muxwise/sglang-slo_config/python/` = **0.5.3rc0 소스**.

| 주장 | 판정 | 근거 |
|---|---|---|
| `initialize_stream_groups`는 `manual_divisions`가 있으면 `divide_sm`을 호출하지 않는다 | **CONFIRMED** | `pdmux_context.py:112-122` — `if config.manual_divisions:` 분기는 튜플 변환만, `divide_sm`은 `else:`(`:118`)에만 |
| `get_arch_constraints`는 `divide_sm` 안에서만 불린다 | **CONFIRMED** | 설치 트리 전역 grep: 정의 `:55`, 호출 **`:76` 단 1곳**. `divide_sm` 호출도 `:118` 1곳. 프로젝트 `src/`·`tests/`에서 **0건** |
| 이 프로젝트 설정은 전부 `manual_divisions`를 쓴다 | **CONFIRMED(강화)** ★**정정(2026-09-22, 코드 근거 감사 `reports/audit/2026-09-22_scope_lineage/REPORT.md` §6 I-1, doc-steward 등재 2026-09-24, 사용자 승인 2026-09-24)**: 이 행의 grep 기반 "59/59 전부"는 **거짓** — `yaml.safe_load` 전수 파싱 결과 **55/59**이고, 나머지 4개(byte-identical `pdmux_a100_smoke.yml`, sha256 `8e991318…`)는 이 키가 **없어** 자동 격자(`divide_sm`) 경로로 간다(주석이 "No manual_divisions -> `divide_sm()`"라 적어 grep이 오검출). E2가 쓰는 `pdmux_homog5.yml`은 여전히 `manual_divisions` 보유(정정 무관). 이 config를 쓰는 job script 27개에 **P1.7 4모델·P1-opint(873944/873945)·`p1_gates/gate2` HOLB**가 있어 `get_arch_constraints`는 우리 실행 경로에서 발화한 적이 있다(`manual_divisions` 계열[E1/S2/λ0/E2]에서만 미발화가 참). 이 행의 판정 자체(Q4 "CONFIRMED but INERT" 결론)은 **불변** — A100은 major 8이라 자동 격자 캠페인 결과도 안 바뀐다. 상세 `CONSENSUS.md` §3 항목280, `PROJECT_STATUS.md` 2026-09-24 정정 배너. | 현행 트리(deprecated 제외) `*pdmux*.yml` **59/59 전부** 보유. E2가 쓰는 `pdmux_homog5.yml` 포함 |
| sync 8파일·patch 5파일 어디에도 `pdmux_context.py`가 없다 ⇒ 실행 트리의 그 파일은 upstream 원본 | **CONFIRMED(강화)** | install 대상은 `multiplex/{dual_worker,multiplexing_mixin,profile,controller,telemetry,holb_probe,chunk_probe,green_readout}.py` + configs 2 + models 5. **manifest 25 엔트리에도 없다** — 원본일 뿐 아니라 **provenance에도 안 잡힌다** |
| ⇒ "major 10+ `ValueError` 거부"는 자동 격자 경로의 사실이며 우리 경로에서 미발화 | **CONFIRMED(스코프 한정)** | ★**단 "그러므로 major 10+에서 돌아간다"로 확장하면 거짓** — 제약 강제가 `spatial.create_greenctx_stream_by_value` → `cuDevSmResourceSplitByCount`/`cuGreenCtxCreate`(`spatial_ops.abi3.so` 심볼 실측)로 **이동할 뿐 사라지지 않는다**. 그 층의 입도는 **미실측** |
| 버전 안정성(추가) | **CONFIRMED** | `0.5.3rc0` ↔ `0.5.10.post1` 차이는 import 위치 이동 1건 + 에러 문자열 1건뿐. 세 함수는 **바이트 동일** |

**E2 등록에 주는 영향: `CONFIRMED but INERT`** — Q4의 소스 사실은 E2 등록을 한 글자도
바꾸지 않는다. 바꾸는 것은 "왜 로컬/Blackwell에서 못 도는가"의 정확도뿐이고
(`CLAUDE.md` 정정 후보, `gpu_rental_checklist §5`의 doc-steward 회부는 **타당**),
**A100↔A100 판정에는 무관**하다. 경계 사례: 이 사실이 "arch 분기를 우회하니 어느 기판이든
격자를 그대로 쓸 수 있다"로 읽히면 **게이트 1 위반**이 된다. 감사 대상 문서
`gpu_rental_checklist_2026-09-18.md`는 그 함정을 스스로 막아 뒀다(§1 표 Blackwell 행
"**미지**(단순 '불가'가 아님)", §4 입도 행 "**B3로 실측해야 한다**") ⇒ **과대 주장 없음**.
다만 §6 "1차 근거"가 "upstream v0.5.10 **태그**를 받아 읽었다"고 적는데 감사자는 그 태그를
재현하지 못했고 설치 wheel `0.5.10.post1`로 대조했다 ⇒ **태그와 `post1`의 바이트 동일성은
`UNDETERMINED`**(구조 결론이 0.5.3rc0까지 동일하므로 실질 위험 0).
※ 2026-09-18 같은 날, 이 지적을 받아 체크리스트 §6에 **조달 경로(curl URL·scratchpad
파일명·바이트수)와 이 UNDETERMINED 자체**를 기재했다.

## 미결 해소 조건 (전부 GPU ≤ 0.1 h, 판정 규칙 0개 추가)

E2 **제출 전 선행 절차로만 두고 어떤 결정 규칙에도 연결하지 않는다** — 연결하는 순간
새 estimand가 되어 재등록 범위가 커진다(교훈 250 자기적용).

1. **B0 기판 사실 채취** — `nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version`.
   `A100 80GB · 8.0 · driver ≥ 12.4`가 아니면 **중단**.
2. **B1 트리 재구성 후 digest 대조** — sync + patch 적용 후 생성된 25줄 manifest를
   **908623 시점 manifest와 라인 대 라인 비교**. (a)의 구멍을 메우는 유일한 길(P4 자체는 개수만 센다).
3. **B4 스모크 1 boot**(`a_r4`/seed 4386/OFF 1회, ≈0.08 GPU-h) — 확인 **넷**, 전부 기록만:
   - `sm_counts` 배너가 `[(108,0),(92,16),(64,44),(16,92),(0,108)]`인가(항목 1·2),
   - `max_mamba_cache_size / max_total_num_tokens` = **48 / 2,722,025**인가(항목 16·22),
   - bench duration이 **131.0 s ± 20%**인가(항목 9, P7 3× 여유 검증),
   - 부대비용이 **145.5 s + 98 s 이내**인가(항목 10 — 이 선을 넘으면 **R4″가 확률적으로
     소멸**함을 사전에 안다).
   - 추가로 `PDMUX_GREEN_READOUT=1`로 **realized green SM**을 1회 읽는다(현재 E2는 target만 본다).
4. **追記 A13(이식)** — 바꾸는 것을 **경로·런처·벽시계 집행기 3개로 한정**해 리터럴로 못박는다:
   ROOT/모듈/venv 경로 · `#SBATCH` 헤더 무효 공시 · **`--time` 대체 집행기**.
   `--cpus-per-task=8` 등가 핀(cgroup/taskset)을 **통제 요인으로 §3에 추가**.
5. **새 OVERRIDE** — 예산 3층을 **GPU-h와 금액 양쪽으로** 재기재, 초과의 의미 변화
   (무상→과금)를 승인문에 명시, §2-1(M4R)·§4 인용 금지 조항을 **문자 그대로 승계**.

### ★감사자 처방의 실현가능성 자기검사 (게이트 #113)

`--time` 대체로 `timeout 10800 bash e2_sticky.sbatch`를 떠올렸으나 **그대로는 안 된다**:
SLURM은 job step 전체의 자식을 정리하지만 `timeout`은 직계 자식(bash)만 죽여
**`launch_server`가 고아로 GPU를 붙든다**(과금 계속). 충실한 대체는
`timeout --kill-after --foreground` + 하네스에 `trap ... EXIT`로 `pkill -9 -f "launch_server"`
추가 = **감사된 하네스에 코드를 더하는 일**이다. ⇒ **"무비용 등가 대체가 있다"고 말할 수 없다.**
이것이 Q3를 "새 OVERRIDE 필요"로 두고 A13을 追記가 아니라 **하네스 국소 수정 + 체크리스트
재실행**까지 포함시키는 이유다.

## 신규 caveat 제안 (E2C-36 … E2C-39) — 결과 문서가 문자 그대로 승계할 것

> **E2C-36** — 이 캠페인이 A100 108 SM 이외의 기판에서 실행됐다면 `a_r4`=8.0 · `a_r2`=3.0 ·
> `b_r3`=1.15는 **그 기판의 부하 사다리가 아니다**(λ0 FALLBACK 사다리는 A100 D44 회귀
> `step(B)=19.82+0.5174·B ms`에 앵커). 셀 이름을 "고부하/중부하"로 **해석 인용 금지**.
>
> **E2C-37** — 이 캠페인의 **어떤 게이트도 기판 변화를 탐지하지 않는다.**
> `idx2 ↔ (64,44)` 대조는 `manual_divisions` 자기 자신과의 비교라 총 SM이 달라져도 통과한다
> (교훈 9). P3는 job 안 boot 간 일치만 본다. ⇒ "게이트가 기판 동일성을 확인했다" **인용 금지**.
>
> **E2C-38** — 이 캠페인은 **realized green-context SM을 측정하지 않는다**
> (`PDMUX_GREEN_READOUT` 미설정). R1이 `STICKY_REALIZES_A`를 내도 그것은 **`stream_index`가
> 2였다**는 뜻이지 **decode가 44 SM을 실제로 받았다**는 뜻이 아니다 — Stage 0 D108(실은
> 16 SM)·λ0 D44 점유 계열과 같은 구분.
>
> **E2C-39** — SLURM 밖에서 실행된 경우 승인된 **3.0 GPU-h 하드 캡은 집행되지 않았다**
> (P14는 boot 사이에서만, `wait_health` 최대 1200 s와 warmup은 무제한). 결과 문서는
> "하드 캡 안에서 끝났다"를 **실측 벽시계 없이 쓸 수 없다**.

## 참조 파일

- `workspace/engine-port/results/r2_eval/e2_sticky_prereg/`(PREREG rev1–rev3 + ADDENDUM,
  판정서 5, OVERRIDE, `e2_sticky.sbatch`, `e2_label.py`, `e2_realized_mix.py`)
- `workspace/engine-port/results/r2_eval/lambda0_prereg/lambda0_plan.py`
  (MULT `:101` · STEP `:132-133` · FALLBACK_LADDER `:150`)
- `workspace/engine-port/results/longctx_conflict/probes/pdmux_homog5.yml`
- `workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh` ·
  `env/devtree_manual_edits.patch` · `env/venv_packages_2026-09-17.txt`
- `workspace/engine-port/src/multiplex/multiplexing_mixin.py:253-286` · `green_readout.py`
- `handoff-report/gpu_rental_checklist_2026-09-18.md`(감사 대상)
