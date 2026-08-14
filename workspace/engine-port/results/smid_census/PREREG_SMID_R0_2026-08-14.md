# PREREG — `%smid` **R0**: L0 standalone census (설계 문서의 P1 + P2)

작성 **2026-08-14** · 작성자 **engine-porter** · 대상 하네스
`smid_l0_census.py` · `smid_l0_run.sbatch`(둘 다 이 디렉터리, 신규) ·
예상 GPU **≤ 0.25 GPU-hr**(벽시계 15분 상한, 실 census는 초 단위) ·
**엔진 패치 0줄 · dev 트리 쓰기 0건 · 매니페스트 재동결 0건**

> 이 문서는 `PREREG_GATE2S_2026-08-09.md` §6.2.2·§11.3·§12-5가 "병행 별건으로
> 즉시 등재"라고 지정하고 `CONSENSUS.md` §1-1(Gate 1 블록)이 **G1-d**로 등재한
> 항목의 **사전등록**이다. 원 사전등록·정본의 어떤 판정 규칙도 바꾸지 않는다.
>
> ⚠️ **설계 문서를 근거로 인용하지 않는다.**
> `workspace/engine-port/reports/SMID_DIRECT_INSTRUMENTATION_DESIGN_2026-08-11.md`는
> **자기 자신을 §12 마지막 항목("본 문서를 사전등록으로 인용 금지")과 §13-1에서
> 명시적으로 금지**하며, 같은 문서는 **claims-auditor 미통과**다
> (`handoff-report/session_handoff_2026-08-13.md` §5-2: "설계 문서 2건은
> claims-auditor 미통과. 등급어로 인용 금지"). 아래 §1은 그 문서를 인용하는
> 대신 **모든 코드 사실을 이 세션에서 직접 재확인**한 결과이고, §1.2는 재확인에서
> **설계와 달랐던 3건**이다.

---

## ★ 0. 최상단 — **이 실험의 payoff는 좁다** (방법론 게이트 #28)

게이트 #28: *"실험이 무엇을 풀어주는지가 코드 사실인지 추정인지, 실행 전에 코드로
확인한 뒤 정당화하라."* 아래 §0.1은 그 확인 결과이며 **부풀리지 않았다**. 이 절에
동의할 수 없으면 이 실험을 사지 마라.

### 0.1 이 실험이 실제로 바꾸는 것 — 정확히 3개, 전부 코드로 확인

| # | 바뀌는 것 | 근거 (file:line, 2026-08-14 직접 확인) | 확인 방법 |
|---|---|---|---|
| 1 | `internal_epsilon_34_to_108`의 **끝점 분모 하드코딩** 정정 | `g2s_analyze.py:1488` — `eps = math.log(mt / mc) / math.log(108.0 / 34.0)` | `grep -n "log(108.0 / 34.0)"` → 정확히 1건, 라인 1488 |
| 2 | 모든 Δ_split 인용에 붙는 **캐비어트 문구**의 근거를 "미측정"에서 "측정치"로 교체 | `CONSENSUS.md` §1-1 인용 금지 목록 중 *"prefill 74 SM·decode 34 SM에서 실행됐다"(S3 미실행, green-context 반올림 미확인)* | 정본 문자열 대조 |
| 3 | telemetry의 **부기 라벨**을 실측 집합으로 대체 가능 | `dual_worker.py:608` `prefill_sms, decode_sms = self.arbiter.sm_counts[self.arbiter.stream_index]` → `:622-623` `"prefill_sms"`/`"decode_sms"` | `grep -n` → 608/622/623 정확 일치 |

### 0.2 ★ 이 실험이 **열지 않는** 것 (이 문장들을 쓰면 사전등록 위반)

1. **`CONSENSUS.md` §1-1의 "PD 분리 자체" 귀속은 닫히지 않는다.** `%smid`는
   **전달 질문**("분할이 물리적으로 전달됐는가")에 답하고 §1-1은 **효과 귀속
   질문**이다. 두 명제는 서로를 함의하지 않는다. 2026-08-10 귀속 스코프 결론
   (성능-층 등가검정으로 닫으려면 n≈64 필요, 현행 10)의 산술을 이 실험은
   **한 자리도** 바꾸지 않는다.
2. **Gate 2-S의 크기 인용 가능 셀은 1개에서 늘지 않는다.** 크기 인용을 통제하는
   것은 `g2s_analyze.py:1156`의 `gl = gate_label([fser[(a_hi, r)], fser[(a_lo, r)]])`
   (F-계열)이고, 그 산출 경로에 SM 라벨도 `premise`도 **입력되지 않는다**
   (`:1156-1161`에서 `gate=gl`과 `premise=labels[...]`는 **나란히 저장될 뿐 서로
   입력이 아니다**). "늘어난다"고 쓰는 것은 G1-c에서 이미 반증된 서술의 재생산이다
   (게이트 #28의 등재 근거).
3. **어떤 성능 주장도 열리지 않는다.** 이 캠페인은 TTFT·TPOT·ITL·goodput·처리량을
   **산출하지 않는다**(하네스에 그 코드가 없다). 서버를 띄우지 않고 모델을 로드하지
   않는다.
4. **SM 집합 ≠ SM 처리량.** `%smid`는 정체성만 준다. 집합에 든 SM의 활용도가 5%일 수
   있다. "74 SM 전달됨" → "74/108 만큼의 계산 자원"은 **금지**.
5. **과거 캠페인에 붙일 수 없다.** L0은 **엔진 없는 별도 프로세스**다. 여기서 나온
   집합을 job 873944/874478/875293/877974/877756/877757의 파티션 문장으로 쓰는 것은
   금지한다(§8-3).

### 0.3 ★ blind 선언

- **작성자는 이 결정량의 값을 본 적이 없다.** 저장소 전체 검색(2026-08-14) 결과
  `smid` 문자열은 **설계 문서 2건과 정본 계획 문서(CONSENSUS/PROJECT_STATUS/
  paper/handoff/PREREG_GATE2S)의 서술**에만 존재하고, **census 아티팩트는 0건**이다
  (`find workspace results -iname "*smid*"` → 설계 문서 1개, `*census*` → 0개).
- 이 저장소에서 **`%smid`를 GPU에서 실행한 기록은 없다**. 설계 문서의 부록 A도
  "GPU 커널 실행 0건"을 명시한다.
- ⇒ **R0·R1·R2·R5는 전부 prospective(사전)다.** E1 addendum(2026-08-11)과 달리
  사후 인지 비대칭이 없다.
- 작성자가 **알고 있는 것**은 코드 유래 기대값뿐이다: `divide_sm(108,(8,0),2)` =
  `[(74,34),(54,54)]` (2026-08-14 CPU에서 생산자 직접 호출로 확인, GPU 0).
  이것은 **target**이며 realized가 아니다 — 그 구별이 이 캠페인의 존재 이유다.

### 0.4 이 사전등록이 하지 **않는** 것

- **L1(in-server boot census)·L2(프로덕션 커널 태깅)를 사전등록하지 않는다.**
  R0 결과가 나오고 claims-auditor 감사를 통과하기 전에는 **L1 패치를 쓰지 않는다.**
- **등가(equivalence) 주장·TOST를 쓰지 않는다**(§7).
- **어떤 arm 간 성능 비교도 하지 않는다.**
- **경로 5(타이밍 계단으로 유효 SM 수 역산)를 채택하지 않는다** — 그것은 다시 두
  노이즈 지표의 대소 비교이며 `PREREG_GATE2S §11.3`이 금지한 형태다.

---

## 1. 코드 사실 — **이 세션에서 직접 재확인**

### 1.1 재확인에 성공한 것

| # | 사실 | 확인 명령/위치 | 결과 |
|---|---|---|---|
| C1 | Triton 3.5.1이 `%smid`를 **GPU 없이** sm_80 PTX/cubin으로 낸다 | `triton.compile(ASTSource(...), GPUTarget("cuda",80,32))`, 로그인 노드 glogin01 | `asm keys = [cubin, llir, ptx, source, ttgir, ttir]`, cubin 6,504 B |
| C2 | 생성 PTX의 probe 사이트 수 | `smid_census_cpu_2026-08-14.ptx` | `%smid` **1**, `%nsmid` **1**, `%globaltimer` **2** |
| C3 | **스핀 루프가 hoist되지 않는다** | 생성 PTX의 basic block 검사(`_spin_loop_has_back_edge`) | `$L__BB0_1` 안에 `mov.u64 %rd20, %globaltimer;` + `@%p2 bra $L__BB0_1` (역방향 분기) — **True** |
| C4 | `cuobjdump -res-usage`가 **GPU 없이** 레지스터를 준다 | `/apps/cuda/13.0.2/bin/cuobjdump -res-usage` | 동작(§7 수치) |
| C5 | libcuda가 green ctx 전 API를 export | `nm -D /lib64/libcuda.so.1` | `cuGreenCtxCreate`·`cuGreenCtxGetDevResource`·`cuCtxGetDevResource`·`cuStreamGetGreenCtx`·`cuDevSmResourceSplitByCount`·`cuGreenCtxGetId` 전부 `T` |
| C6 | **`CUdevSmResource`에 SM 개수만 있고 ID 집합이 없다** | `/apps/cuda/13.0.2/include/cuda.h:24770-24777` | `smCount`·`minSmPartitionSize`·`smCoscheduledAlignment` 3필드뿐 |
| C7 | **CUPTI 13.0.2 헤더 전체에 `smId` 0건** | `grep -rn "smId" /apps/cuda/13.0.2/extras/CUPTI/include/` | **0** ⇒ 경로 2는 권한 이전에 원리로 죽었다 |
| C8 | `cuStreamGetGreenCtx`는 비-green 스트림에 NULL | `cuda.h:25159`·`:25165` "Otherwise, \p *phCtx is set to NULL instead." | 확인 |
| C9 | 드라이버가 green ctx 분할의 **disjoint**를 문서화 | `cuda.h:24993` "guarantee a split that will create a disjoint set of symmetrical partitions" | 확인. **primary context가 영향받는지는 헤더가 침묵** |
| C10 | `SM_COUNTS` 구성과 스트림 종류 | `pdmux_context.py:124-127`(라벨), `:130`·`:137`(평범 `torch.cuda.Stream`), `:134`(`spatial.create_greenctx_stream_by_value`) | idx 0·N-1은 green ctx가 **아니다** |
| C11 | `divide_sm(108,(8,0),2)` = `[(74,34),(54,54)]` | 생산자 직접 호출(CPU) | ⇒ `SM_COUNTS = [(108,0),(74,34),(54,54),(0,108)]` |
| C12 | 설정: `sm_group_num: 4`, `decode_bs_divisor: 36`, manual 없음 | `../p1_gates/gate2/pdmux_a100_smoke.yml` | 확인 |
| C13 | decode만 cudagraph로 캡처 | `cuda_graph_runner.py:547` `self.capture_forward_mode = ForwardMode.DECODE` | 확인 |
| C14 | pdmux 캡처는 stream group마다 decode 스트림 위 | `cuda_graph_runner.py:811-817` `for i, sg in enumerate(self.stream_groups): with graph_capture(stream=sg[1])` | 확인 |
| C15 | 재생은 stream-idx별 그래프 선택 | `cuda_graph_runner.py:1158` `graph_key = f"{get_current_stream_idx()}_{self.bs}"`, `:1161` `.replay()` | 확인 |
| C16 | 로그인 노드 프로파일링 admin 전용 | `/proc/driver/nvidia/params` | `RmProfilingAdminOnly: 1` |

### 1.2 ★ 재확인에서 **설계와 달랐던 것 3건** (전부 이 사전등록에 반영)

1. **관측자 부하 수치가 재현되지 않는다.** 설계 V5는 `%smid` 1판독+1스토어에
   **REG 8→10(+2)**이라고 적었다. 같은 도구(`cuobjdump -res-usage`, CUDA 13.0.2)로
   재현하면 **REG 8→12(+4)**다. baseline/instrumented의 **시그니처를 동일하게** 맞춰
   (프로브 전용 포인터 인자를 추가하지 않고 같은 버퍼의 다른 오프셋에 쓰는 형태)
   측정해도 **+4로 동일**했다(포인터 인자 추가가 원인이 아니다). 재현된 것은
   **SHARED/LOCAL/STACK 전부 0으로 불변**이라는 정성적 사실과 "CPU에서 통계 없이 잴 수
   있다"는 절차뿐이다. ⇒ **"+2"라는 수치는 이 사전등록에서 사용하지 않는다**
   (게이트 #32: 다른 맥락에서 수입한 보조 수치는 기준 검증 후에만 쓴다).
2. **설계 부록 A의 재현 명령이 틀렸다.** `cat /proc/driver/nvidia/params | grep -i
   restrict`는 **아무것도 매치하지 않는다**(그 파일에 "restrict" 문자열이 없다).
   값 자체(`RmProfilingAdminOnly: 1`)는 맞다 — 틀린 것은 명령이다. 이 사전등록의
   하네스는 `grep -i "RmProfilingAdminOnly"`를 쓴다.
3. ★**Triton 3.5.1은 `%smid`/`%globaltimer` 헬퍼를 이미 제공한다** —
   `triton/language/extra/cuda/utils.py:5-13`의 `globaltimer()`(`is_pure=False`)와
   `smid()`(`is_pure=True`). 설계는 이 사실을 언급하지 않고 인라인 asm을 손으로 썼다.
   ⇒ **하네스는 손으로 쓴 복사본이 아니라 이 상류 생산자를 import한다**
   (게이트 #14: 자기가 검증하려는 코드를 복사한 게이트는 항등식에 가깝다 —
   원시함수는 생산자에서 가져온다). `%nsmid`만 상류 헬퍼가 없어 직접 쓰며,
   그것은 **서술 전용이고 어떤 판정에도 입력되지 않는다**.

**사소한 인용 정정 1건(판정 무영향)**: 설계 §1.1 표는 F-계열 게이트 위치를
`g2s_analyze.py:1157-1161`로 적었으나, `gate_label(...)` **호출 자체는 :1156**이고
`:1157-1161`은 그 결과를 `gate=gl`로 저장하는 dict다. 결론(F-계열이 크기 인용을
통제하고 `premise`는 그 산출 경로에 미입력)은 소스 재확인에서 **그대로 성립**한다 —
`gl`은 `fser`만 입력으로 받고 `premise=labels[(tag, r)]`는 나란히 저장될 뿐이다.

**부수 코드 사실(운영, 새로 발견)**: `$HOME`이 파일 쿼터 상한(98,976/100,000)이라
기본 `~/.triton` 캐시로는 커널 컴파일이 `OSError: [Errno 122] Disk quota exceeded`로
**시작조차 못 한다**. 하네스는 `TRITON_CACHE_DIR`를 캠페인 디렉터리로 강제한다.

---

## 2. 결정량 — 전부 **집합 술어**, 성능 지표 0개

표기: `S(x)` = 대상 `x`에 launch한 census의 **union SM id 집합**.
`D` = **green context 생성 이전에** 평범한 `torch.cuda.Stream`으로 관측된 id 집합
(`S_pre`). `|D|`는 관측으로 정의하며, `%nsmid`나 `spatial.get_sm_available()`이나
드라이버 `smCount`로 정의하지 **않는다**.

| ID | 질문 | 결정량 | 지위 |
|---|---|---|---|
| **R0** | `%smid`가 green ctx를 건너 **일관된 전역 라벨**인가 | `S(green_p) ∩ S(green_d)`, `S(green_p) ∪ S(green_d)` vs `D` | ★ **게이트 + 정지 규칙** |
| **R1** | 분할이 전달되는가, realized 크기는 얼마인가 | `|S(green_p)|`, `|S(green_d)|` vs `divide_sm` target | R0 조건부 판독 |
| **R2** | green ctx 생성이 primary context를 깎는가 | `D \ S_post` (**같은 스트림 객체**, 생성 전/후) | R0 조건부 판독 |
| **R5** | 동시 실행 시 SM을 공유하는가 | `%globaltimer` 겹침 창에서 `S_p(t) ∩ S_d(t)` | **서술 전용**, 마이크로 체제 한정 |
| **D1** | 커버리지가 포화했는가 | 그리드 사다리 마지막 두 점의 **집합 동일성** | ★ 전 판독의 **선행조건** |
| **D2** | 희소 히트 SM | SM별 **최소** 히트 수 | 서술(게이트 #29의 정신) |

**의도적으로 넣지 않은 것**: 성능 지표 0개 · 등가검정(TOST) 0개 · 반복 n에 대한
유의성 검정 0개(반복은 재현성 확인용이며, 집합이 부팅마다 다르면 그 사실을 보고한다).

---

## 3. ★ R0 결정 규칙과 정지 규칙 (**결과를 보기 전에 고정**)

### 3.1 왜 **카디널리티는 R0의 결정량이 아닌가** (이 절이 R0 설계의 핵심)

`%smid`가 green context마다 **0부터 다시 번호를 매기는**(가상화) 세계에서도
`|S(green_p)| = 74`, `|S(green_d)| = 34`는 **그대로 나온다**. 즉 **크기만으로는
가상화와 물리 전달을 구별할 수 없다.** 구별하는 것은 **교집합**이다. 그래서
R0의 결정량은 집합 관계이고 크기는 R1로 분리했다.

⇒ **R1의 PASS는 R0가 실패하면 아무 의미도 없다.** 하네스는 이 의존을 코드로
강제한다(R0 실패 시 R1/R2/R5는 `UNINTERPRETABLE` 문자열만 산출).

### 3.2 ★ 4-outcome 매트릭스 (설계 초안의 2분법을 **거부**한다)

설계 §5-R0은 *"겹치거나 둘 다 0에서 시작하면 가상 id → 즉시 정지"*라는 2분법이다.
**이 사전등록은 그것을 채택하지 않는다.** 겹침은 가상화 말고도 **"분할이 아예
전달되지 않음"**과 양립하고, 그 둘은 **완전히 다른 사실**이다 — 하나는 계측 실패,
다른 하나는 **엔진에 대한 실측 결과**다. 둘을 한 라벨로 묶는 것이 정확히 방법론
게이트 #21(측정 실패를 게이트 실패로 라벨링, **7회 재발**)의 형태다.

| 관측 | 사전등록 상태 | 의미 | 후속 |
|---|---|---|---|
| `∩ = ∅` **∧** `∪ = D` | **`GLOBALLY_CONSISTENT_LABEL`** | `%smid`는 두 context를 건너 일관된 라벨이다 | R1·R2·R5 해석 개시 |
| `∩ ≠ ∅` **∧** 양쪽 `|S| ≥ 0.9·|D|` | **`LABEL_CONSISTENT_BUT_PARTITION_NOT_DELIVERED`** | 라벨은 정상이고 **분할이 전달되지 않았다**. ★**계측 실패가 아니라 엔진에 대한 결과다** | 즉시 claims-auditor·doc-steward 회부(§3.3-b) |
| `∩ ≠ ∅` **∧** `|∪| < |D|` | **`NOT_A_GLOBALLY_CONSISTENT_LABEL`** | context별 재번호 부여와 정합. 경로 1 해석 불가 | ★ 정지(§3.3-a) |
| 그 밖 | **`UNDETERMINED (OUTCOME OUTSIDE PRE-REGISTERED MATRIX)`** | 원 집합만 보고하고 멈춘다 | 재설계 |

**`0.9` 자유 모수 자백**: 경쟁 가설들이 예측하는 값은 `74/108 = 0.685`,
`54/108 = 0.5`, `1.0`이다. 0.9는 그 어느 예측값과도 떨어져 있어 **어느 가설도
경계에 놓이지 않는다**. 그럼에도 이것은 자유 모수이므로 §12에 등재한다. 경계에
떨어지면 4행(`UNDETERMINED`)이 발화하도록 설계돼 있어 **임의 임계가 판정을 만드는
경로가 없다**.

### 3.3 ★ 정지 규칙 (설계의 "즉시 정지"를 사전등록 규칙으로 승격)

- **(a) `NOT_A_GLOBALLY_CONSISTENT_LABEL` 또는 `UNDETERMINED` → 즉시 정지.**
  정지의 정확한 범위: **L1 패치를 쓰지 않는다 · L2를 검토하지 않는다 · 경로 1에 GPU를
  추가로 쓰지 않는다 · `%smid` 유래 문장을 어떤 정본에도 넣지 않는다.**
  §0.1의 3개 payoff는 전부 **취소**되고, `CONSENSUS.md` §1-1의 기존 인용 금지는
  **그대로 유지**된다(완화 아님).
- **(b) `LABEL_CONSISTENT_BUT_PARTITION_NOT_DELIVERED` → 정지하되 사망은 아니다.**
  이것은 성능 판정이 아니라 **코드 사실의 정정 사유**이므로 claims-auditor·doc-steward에
  회부하고 engine-porter는 거기서 멈춘다. **이 상태에서 어떤 성능 재해석도 하지 않는다.**
- **(c) `GLOBALLY_CONSISTENT_LABEL` → R1·R2·R5 판독을 개시한다.** 그래도 §0.2의
  다섯 금지는 그대로다.
- 어느 경우에도 **`smid_l0_raw_<jobid>.json`은 그대로 보존**하고 사후 수정하지 않는다.

### 3.4 R0가 licence하는 문장의 **정확한 상한**

R0 통과가 확립하는 것은 **"이 디바이스에서 `%smid`는 context를 건너 일관된 단사
라벨"**까지다. **"NVIDIA의 물리 SM 인덱스와 같다"는 확립되지 않는다**(문서화된
보장이 없다). 다행히 이 캠페인의 모든 질문(disjoint 여부·크기·차집합·겹침)은
라벨의 **전역 일관성**만 요구하고 절대값 의미를 요구하지 않는다. ⇒ 보고서에
"물리 SM id"라고 쓰지 말고 **"전역 일관 라벨"**로 쓴다. 상태 문자열도 그렇게 지었다.

---

## 4. R1 / R2 / R5 / D1 / D2 (R0 조건부)

### 4.1 D1 — 포화는 **전 판독의 선행조건**

- 그리드 사다리: `n_blocks ∈ {108, 216, 432, 864, 1728}`, 각 점 **5회 launch, union**.
- **포화 정의**: 마지막 두 그리드 점의 union이 **집합으로 동일**(크기 동일이 아니라
  집합 동일 — 더 강하고 추가 비용 0).
- **미포화 대상이 하나라도 있으면 그 대상에 대한 모든 집합 문장은
  `UNDETERMINED (COVERAGE NOT SATURATED)`**이며, 특히 **"이 SM은 없다"류의 부재 주장은
  전면 금지**한다.
- ★**포화해도 부재의 증명이 아니다**(설계 §6-S1이 자백한 미끄러짐 지점 S1).
  허용 문장은 **"n_blocks=1728·5회 반복에서 평평해진 조건에서 관측된 집합은 …"**뿐이다.
- span 기반 coverage 지표를 쓰지 않는다(게이트 #29: span은 내부 구멍에 맹목).

### 4.2 R1 — realized 크기 vs `divide_sm` target

- 기대값은 **생산자에서** 온다: `pdmux_context.divide_sm(total_sm, cc, 2)`를 하네스가
  **직접 호출**한다(하드코딩 74/34 없음).
- `[|S_p|, |S_d|] == target` → `exact_match: true`.
- `!=` → **관측 그대로 보고**. §5.1의 사전 배정에 따라 **결과이지 실패가 아니다.**
- **금지**: 집합 크기를 계산 자원 비율로 환산하는 것(§0.2-4).
- idx2(`(54,54)`)도 같은 규칙으로 **나란히** 보고한다(셀 단위 보고, 요약 상태 금지).

### 4.3 R2 — primary context 차감. ★**R3와 병합했다**

설계는 R2(`|S(idx0_p)| = |D|`?)와 R3(`S_pre \ S_post`)를 **별개 결정량**으로 뒀다.
`|D| := |S_pre|`로 정의하는 순간 **두 규칙은 같은 경험적 내용**이다(§6.3). ⇒
**결정량은 `D \ S_post` 하나**로 하고, `idx0`·`idx3`의 평범 스트림 census는
**서술적 복제**로만 병기한다. 하네스가 이 문장을 산출물에 문자열로 박는다
(`R2.not_independent`).

- `D \ S_post ≠ ∅` → **차감 관측**(그 크기는 SM 단위, 추정 아님).
- `= ∅` → **"이 포화 수준에서 차감이 관측되지 않았다"**. 등가 선언이 아니다.

### 4.4 R5 — 시간 겹침 (서술 전용)

`%globaltimer` 겹침 창에서 두 스트림이 같은 SM id를 갖는지. **마이크로 체제
한정**(유휴 GPU 위 스핀 커널 2개)이며 **서빙 동작점으로의 이전은 금지**한다.

### 4.5 D2 — 최소 히트

SM별 최소 히트 수를 병기한다. 합계·span만 보고하지 않는다.

---

## 5. ★ 자기무력화 조건 (재감사자가 **먼저** 읽을 절)

### 5.1 ★ 반올림·granularity — **결과인가 측정 실패인가**를 실행 전에 배정한다

green ctx 분할은 드라이버의 `cuDevSmResourceSplitByCount`가 수행하며, 헤더 자신이
`minSmPartitionSize`·`smCoscheduledAlignment`(`cuda.h:24772-24776`)라는 **정렬 제약**을
가진다. 따라서 realized 집합이 target `(74,34)`와 다를 수 있다.

> **사전 배정**: realized ≠ target인데 **R0가 통과했고 D1이 포화했다면, 그것은
> 측정 결과다.** "측정 실패"로 라벨링하는 것을 **금지**한다.

- 예: realized `(76,32)`, disjoint, `∪ = D` → R0 = `GLOBALLY_CONSISTENT_LABEL`,
  R1 `exact_match: false`. 이 경우 **정본이 강화되는 방향으로만** 작용한다 —
  `SM_COUNTS` 라벨이 부기라는 기존 서술(`PREREG_GATE2S §1.1`·§1.3)이 **실측으로
  뒷받침**되고, `(74,34)`를 액면 인용하는 것은 더 강하게 금지된다.
- 이 분기는 하네스의 단위 테스트로 고정돼 있다(`--selftest-analyzer`의
  *"rounded partition still GLOBALLY_CONSISTENT_LABEL (result, not failure)"*).
- **반대로**: D1 미포화 상태에서 관측된 크기 차이는 **결과가 아니라
  `UNDETERMINED (COVERAGE NOT SATURATED)`**다. 두 경우를 가르는 것은 **오직 D1**이며,
  D1은 결과를 보기 전에 정의됐다.

### 5.2 ★ 무엇이 **측정 실패**인가 (전부 `UNDETERMINED (MEASUREMENT ABSENT)`)

게이트 #21 7번째 재발(분석 코드 자신이 거짓 음성을 산출)을 막기 위해 **분석 코드의
첫 분기**에 넣었고 단위 테스트로 고정했다.

| 조건 | 상태 |
|---|---|
| raw 아티팩트 없음 / JSON 파손 / 빈 dict | `UNDETERMINED (MEASUREMENT ABSENT)` |
| `S_pre`·`S_post`·`per_stream`·`divisions_from_divide_sm` 중 하나라도 없음 | 〃 |
| **런타임 PTX에 `%smid` 사이트 0** (돌아간 커널이 프로브가 아님) | 〃 |
| **런타임 PTX의 스핀 루프 back edge 없음** (상주 실패 ⇒ union은 미지의 여유를 가진 하한) | 〃 |
| green 쌍 idx1이 census되지 않음 | 〃 |
| job이 exit≠0 / census 미완 | 〃 (sbatch가 exit 21로 구분) |

⚠️ **런타임 PTX 검사가 왜 필요한가**: CPU 자기검사는 **AOT**(`triton.compile` +
명시 시그니처)로 컴파일하고, 실행은 **JIT**이라 정수 인자 특수화 등으로 **다른 PTX가
나올 수 있다**. 로그인 노드에서 검증한 PTX가 컴퓨트 노드에서 돈 PTX라는 보장이 없다
⇒ **실제로 돈 객체의 PTX를 다시 검사**하고, 그 sha256과 전문을 아티팩트에 남긴다.

### 5.3 이 사전등록이 **스스로 파기되는** 조건

1. **하네스가 dev 트리에 쓰기를 하면** — 전제 위반. 결과 폐기.
2. **매니페스트가 바뀌면** — 이 캠페인은 sync를 호출하지 않는다. 트리 상태가
   877756/877757/877974와 다르면 그 사실을 기록하고 **결과를 그 트리 상태에 한정**한다.
3. **`%nsmid`나 드라이버 `smCount`가 판정 입력으로 발견되면** — §6.2 위반. 결과 폐기.
   (하네스가 AST 검사로 자기 자신을 막는다 — `--selftest-analyzer` 첫 항목.)
4. **R0의 4-outcome 중 어느 것도 발화하지 않으면** — 설계 재검토 대상이며, 관측치를
   그대로 보고하고 어떤 해석도 붙이지 않는다.
5. **payoff가 코드로 검증되지 않았다면**(게이트 #28) — §0.1의 3개는 file:line으로
   확인했다. 감사자는 그 3개가 실제로 어떤 정본 문장을 바꾸는지 재대조하라. 바꾸는
   문장이 없다면 이 캠페인은 **비용만 발생시킨 것**이며 그렇게 기록되어야 한다.

---

## 6. ★ 게이트 #9 항등식 점검 — **실행 전에 코드로 확인한 결과**

게이트 #9: *"자기가 검증하려는 코드를 복사한 게이트는 항등식에 가깝다 — 대조는
생산자에 걸어라."* 다섯 번째 재발의 死因은 **자수하고 원안대로 실행한 것**이었다.
그래서 아래 세 건은 자수가 아니라 **설계 변경**으로 처리했다.

### 6.1 점검한 게이트 4개

| 게이트 | 항등식인가 | 근거 | 조치 |
|---|---|---|---|
| R0(disjoint ∧ tiling) | **아니다** | 네 결과가 전부 도달 가능함을 합성 입력으로 실증(`--selftest-analyzer`가 `GLOBALLY_CONSISTENT_LABEL`·`NOT_A_GLOBALLY_CONSISTENT_LABEL`·`LABEL_CONSISTENT_BUT_PARTITION_NOT_DELIVERED`·`UNDETERMINED` 4종을 실제로 산출) | 유지 |
| R1(크기 == target) | **부분적으로 위험** | 가상화 세계에서도 74/34가 나온다 ⇒ 크기는 R0를 판별하지 못한다 | **R0에서 분리**(§3.1). R1은 R0 조건부 판독으로 강등 |
| R2 vs 설계의 R3 | ★**같은 정보다** | `|D| := |S_pre|`면 "`|S_post| = |D|`?"와 "`S_pre \ S_post = ∅`?"는 같은 명제 | **병합**(§4.3). idx0/idx3는 서술적 복제로 명시 |
| D1(포화) | **아니다** | 사다리 마지막 두 점이 다를 수 있다 | 유지, 단 §4.1의 문구 제한 부착 |

### 6.2 ★ green ctx 자기보고로 검증하는 경로를 **배제했다** (Stage 0 D108 거울상)

- `cuGreenCtxGetDevResource`/`cuCtxGetDevResource`가 주는 `smCount`는 **또 하나의
  target**이다. 그것으로 `%smid` 결과를 "검증"하면 **target으로 target을 검증**하는
  Stage 0 D108의 거울상이 된다. 게다가 `CUdevSmResource`에는 **id 집합 자체가 없어서**
  (C6) 정체성 질문에 원리적으로 답할 수 없다.
- **조치**: 드라이버 read-out(P2)은 **서술 전용 블록**으로 분리했고, 판정 함수
  `score()`는 그 블록을 **읽지 않는다**. 이것을 약속이 아니라 **기계적 검사**로 만들었다 —
  `score()`의 AST에서 `driver_readout`·`smCount`·`minSmPartitionSize`·
  `smCoscheduledAlignment`·`cuCtxGetDevResource`·`cuGreenCtx`·`total_sm_reported`·
  `nsmid_observed` 문자열이 하나라도 나오면 자기검사가 **FAIL**한다(2026-08-14 실행:
  8개 전부 부재, PASS).
- **대조는 생산자에 걸었다**: 기대값은 `pdmux_context.divide_sm` **직접 호출**,
  스트림은 `sgl_kernel.spatial.create_greenctx_stream_by_value` **직접 호출**,
  probe 원시함수는 `triton.language.extra.cuda.{smid,globaltimer}` **상류 import**다.
  세 곳 모두 재구현 복사본이 없다.
- **P2가 유일하게 결정적인 방향**: `cuCtxGetDevResource(primary)`가 108 **미만**을
  보고하면 그것만으로 차감이 확정된다(반증 가능). 108을 보고하면 아무것도 확정되지
  않는다. ⇒ **한 방향으로만** 서술에 쓰고, **판정에는 어느 방향도 쓰지 않는다**.

### 6.3 남은 위험 — 자백

- **`%smid`가 "전역 일관 라벨"이라는 것만 확립된다**(§3.4). 물리 인덱스와의 동일성은
  이 설계로 확립 불가이며, 그것을 확립하는 방법을 우리는 갖고 있지 않다.
- **R0 자신은 외부 앵커가 없는 자기검정**이다. 두 context가 **동일한** 방식으로
  재번호를 부여받는 병리(예: 양쪽 다 물리 id의 같은 순열)는 배제되지 않는다.
  다만 그 병리 하에서도 이 캠페인의 모든 술어(disjoint·차집합·겹침)는 불변이다.
- **D1 포화는 "부재"를 증명하지 않는다**(§4.1). 이것이 R2의 "차감 없음" 서술이
  등가 선언이 아닌 이유다.

---

## 7. 관측자 부하 — **기전으로 bound한다, 통계로 하지 않는다**

전제: 이 프로젝트의 관측자 효과 게이트 **G5는 UNDETERMINED**(TOST 재채점, 72셀 중
초과 지지 0·등가 29·검정력부족 43). ⇒ "프로브 무해"를 전제할 수도 없고, 무해를
*증명하는* 통계 게이트를 새로 만들어서도 안 된다(그게 G5와 같은 형태이고 게이트 #20
"귀무 채택형 게이트는 노이즈를 보상한다"에 걸린다).

### 7.1 L0에서는 문제가 **구조적으로 성립하지 않는다**

프로브가 **유일한 GPU 작업**이다. 서버도 모델도 없다. 측정 대상은 "이 스트림에
launch된 커널이 어느 SM에 앉는가"이고 프로브는 교란원이 아니라 **트레이서 자신**이다.
엔진 코드 0줄, 서빙 루프 0줄. ⇒ **on/off 대조 자체가 정의되지 않는다.**

### 7.2 그럼에도 CPU에서 기전 bound를 실측해 둔다 (L2 사전작업, GPU 0)

| 커널 | REG | STACK | SHARED | LOCAL |
|---|---|---|---|---|
| `toy_base`(프로브 없음) | **8** | 0 | 0 | 0 |
| `toy_instr`(`%smid` 1판독 + 1스토어, **동일 시그니처**) | **12** | 0 | 0 | 0 |

(2026-08-14, glogin01, `cuobjdump -res-usage`, CUDA 13.0.2, sm_80.
아티팩트 `smid_l0_cpu_selftest_cpu_2026-08-14.json`.)

- 재현된 사실: **SHARED/LOCAL/STACK 불변**, 델타가 CPU에서 **통계 없이** 결정론적으로
  측정 가능.
- ★**재현되지 않은 사실**: 설계의 "+2". 실측은 **+4**다(§1.2-1).
- ★**이 수치의 스코프**: **장난감 커널 한정**. 실제 프로덕션 커널의 여유는 그 커널을
  캠페인 상수로 컴파일해 따로 재야 한다. 이 표를 다른 맥락으로 수입하는 것을
  **금지**한다(게이트 #32).
- **금지**: "on/off paired TOST로 프로브 무해 입증" — G5 형태. 하지 않는다.
- **잔여 자백**: 위 표는 *occupancy와 명령 수*를 bound하지 *메모리 시스템 간섭*을
  bound하지 않는다. L2를 살 때 이 항목은 **미해결로 등재**해야 한다.

---

## 8. 해석 제한 (인용 시 항상 병기 — `CONSENSUS.md` §1-1과 **모순 없이**)

1. **§1-1의 인용 금지는 이 캠페인으로 해제되지 않는다.** 특히
   *"실현 파티션을 **측정**했다"* 와 *"prefill 74 SM·decode 34 SM에서 **실행**됐다"* 는
   **여전히 금지**다. 이유: 이 캠페인은 **엔진이 아닌 별도 프로세스**에서
   **동일한 API 호출**을 재현한 것이고, 서빙 창의 어떤 요청도 여기서 실행되지 않는다.
2. **"PD 분리 자체" 귀속 승격 금지**, **§1-1 격차의 성분·기여분·분해로 서술 금지**
   (Gate 2-S 인용 제한과 동일 취지).
3. **과거 job에 붙이기 금지** — 873944/874478/875293/877974/877756/877757 어느 것에도
   이 캠페인의 집합을 파티션 문장으로 붙이지 않는다.
4. **selector-level 서술과 혼동 금지** — Gate 1/G1-b/G1-c가 인용 가능하게 만든 것은
   **selector 라벨**이고, 이 캠페인이 재는 것은 **드라이버가 스트림에 부여한 SM 집합**이다.
   둘은 다른 층이며, 한쪽이 다른 쪽을 함의하지 않는다.
5. **마이크로 체제 한정**(R5) — 유휴 GPU 위 스핀 커널 2개. 서빙 동작점 이전 금지.
6. **정지 상태 측정** — 장시간 서빙 중 파티션 상태가 표류하는지는 이 설계가 보지 않는다.
7. **모델 무관** — 이 캠페인에는 모델이 없다. 모델별 문장을 만들지 않는다.
8. **goodput·SLO·fused 대비 우열에 대해 아무 말도 하지 않는다.**

---

## 9. ★ 열린 항목 — **cudagraph-ON 전달 여부는 이 캠페인이 측정하지 않는다**

운영점은 cudagraph-ON이고 decode 측 SM 분할은 전부 그 경로를 지난다(C13–C15:
`capture_forward_mode = ForwardMode.DECODE`, stream group마다
`graph_capture(stream=sg[1])`, 재생은 `graph_key = f"{stream_idx}_{bs}"`).

> **green-ctx 스트림 위에서 캡처된 CUDA graph를 재생할 때 그 green context의 SM
> 한정이 그대로 전달되는가 — 이 저장소의 어떤 아티팩트도 이것을 측정한 적이 없다.**

- 이것은 **R4**이고 **L1(in-server boot census)에서만** 답할 수 있다(캡처된 그래프가
  엔진의 것이어야 하기 때문). **이 사전등록의 범위 밖**이다.
- ⚠️ **방향을 단정하지 않는다.** "전달된다"도 "안 된다"도 현재 근거 없음.
- R0가 통과해도 이 구멍은 **그대로 열려 있다**. R0 결과를 인용할 때 이 문장을
  병기하지 않으면 인용이 무효다.
- L1은 별도 사전등록 + claims-auditor 감사 + `src/patches/` 미러 +
  `env/dev_tree_edits.md` 기록 + **별도 캠페인 매니페스트로 격리**(G1-a 대기 중이므로
  기존 캠페인 재현 경로에 drift를 만들지 않는다)를 전부 마친 뒤에만 착수한다.

---

## 10. 배관 스모크 (방법론 게이트 #26, 2026-08-14 개정 반영)

개정된 발동 조건: *"한 배치로 제출되는, 신규/변경 코드를 공유하는 캠페인들의 합"*이
≥1 GPU-hr이면 신규/변경분이 있는 캠페인마다 최소 rep 스모크를 선행한다. 이 배치
(P1 · E1-b/c · `%smid` · G1-a)의 합은 **≈1.15–1.65 GPU-hr**로 문턱을 넘는다 ⇒
**이 캠페인도 스모크 대상이다.**

**적용 방식 — 이 캠페인은 자기 자신이 스모크에 가깝다**:

- 총 비용 **≤0.25 GPU-hr**이고 그중 census 자체는 **초 단위**다. 별도 n=1 스모크를
  따로 사는 것은 본 실행과 **거의 같은 비용**이라 정보 대비 낭비다.
- 대신 스모크가 잡아야 할 것을 **하네스 안으로 옮겼다**:
  (i) **stage 1** CPU 자기검사(프로브 PTX·스핀 back edge·`cuobjdump`) —
  **GPU 0, 이미 로그인 노드에서 실행해 PASS**;
  (ii) **stage 2** 분석기 자기검사(빈 입력·프로브 없는 PTX·hoist된 스핀·미포화·
  4-outcome·차감·항등식 배제) — **GPU 0, 이미 실행해 16/16 PASS**;
  (iii) 둘 중 하나라도 실패하면 sbatch가 **census를 돌리지 않고 exit 20**으로 죽는다.
  ⇒ 스코어러가 소비할 스키마를 만드는 코드가 **GPU를 쓰기 전에** 전부 검증된다.
- ★**스모크 PASS는 성능 판정에 아무 정보도 주지 않는다.** stage 1·2의 PASS는
  배관 사실일 뿐이며, 어떤 물리·전달·성능 문장도 licence하지 않는다. 하네스는 이
  문장을 산출 JSON의 `result_scope` 필드에 문자열로 박아 둔다.
- ⚠️ 따름정리(게이트 #26): **스모크의 판정 코드 자신을 무조건 신뢰하지 마라.**
  stage 1·2가 PASS해도 원자료(`smid_l0_raw_*.json`)의 스키마를 사람이 직접 대조하라.

---

## 11. 무결성 규칙과 아티팩트

| ID | 검사 | 실패 시 |
|---|---|---|
| I1 | job exit code == 0 | `UNDETERMINED (MEASUREMENT ABSENT)` |
| I2 | stage 1(CPU 자기검사) PASS | 〃 (sbatch exit 20) |
| I3 | stage 2(분석기 자기검사) PASS | 〃 (sbatch exit 20) |
| I4 | `runtime_ptx_smid_sites ≥ 1` **∧** `runtime_spin_back_edge == true` | 〃 |
| I5 | 전 census 대상에 대해 D1 포화 | 해당 대상만 `UNDETERMINED (COVERAGE NOT SATURATED)` |
| I6 | `sha256_unmanifested`의 두 파일 해시 기록 존재 | 기록 후 계속(서술) — 게이트 #33 |
| I7 | dev 트리 내용(sha256) 변화 0 | 사전등록 파기(§5.3-1) |

★**I7 관측 기록(2026-08-14, 작성 중)**: 이 사전등록을 쓰는 동안 **다른 프로세스가
`sync_engine_tree.sh`를 실행**해 dev 트리 10개 파일의 mtime이 `16:10:41`로 갱신됐다
(같은 세션에 P1·E1-b/c 사전등록 작업이 병렬 진행 중). **바이트는 변하지 않았다** —
`src/multiplex/{dual_worker,multiplexing_mixin,telemetry}.py`와 dev 트리 대응 파일의
sha256이 일치(무변경 재설치). 그리고 **이 캠페인의 임계 경로 파일 2개는 sync 대상이
아니다**: `pdmux_context.py`는 mtime `2026-04-06 01:47:12`로 그대로이고(게이트 #33이
지적한 매니페스트 밖 파일), `sgl_kernel/spatial.py`는 venv에 있어 sync 관할 밖이다.
⇒ **I7은 mtime이 아니라 sha256으로 판정한다**(mtime 판정이면 여기서 거짓 파기가
발생했을 것이다). **이 사전등록 작성자의 dev 트리 쓰기는 0건이다** — `divide_sm`
호출은 읽기 전용 import였고 `__pycache__/pdmux_context.cpython-314.pyc`의 mtime은
`2026-07-03`으로 불변임을 확인했다.

**아티팩트**(전부 `workspace/engine-port/results/smid_census/`):

- `smid_l0_cpu_selftest_<tag>.json` · `smid_census_<tag>.ptx` · `*.cubin`
- `smid_l0_raw_<jobid>.json`(원자료, 사후 수정 금지) · `smid_runtime_<jobid>.ptx`
- `smid_l0_verdict_<jobid>.json` / `.txt`
- `smidl0_<jobid>.out` / `.err`

**정본 갱신은 이 문서가 하지 않는다** — doc-steward 소관이며 claims-auditor 감사가
선행 조건이다.

---

## 12. 자유 모수 **전수 목록** (신규 통계 게이트 0개, 신규 검정 0개)

| # | 모수 | 값 | 근거 / 위험 |
|---|---|---|---|
| 1 | `GRID_SWEEP` | `{108,216,432,864,1728}` | 108 = 디바이스 SM 수(생산자 조회), 이후 2배씩. **판정에 들어가는 것은 마지막 두 점의 집합 동일성뿐** |
| 2 | `REPEATS_PER_GRID` | 5 | union 반복. **추정량이 아니므로 유의수준이 붙지 않는다**(게이트 #24의 `any()` 스크린 형태가 아님 — OR형 인용 자격 게이트가 없다) |
| 3 | `SPIN_NS_DEFAULT` / `CAP` | 1 ms / 2 ms | 상주 확보 vs 워치독. 설계 F4의 상한을 그대로 채택 |
| 4 | `SPIN_ITER_CAP` | 2²⁴ | 클록 판독과 **독립**인 두 번째 워치독 |
| 5 | R0 `NOT_DELIVERED` 분기의 `0.9·|D|` | 0.9 | §3.2 자백. 경쟁 가설 예측값 0.5·0.685·1.0과 모두 떨어져 있고, 경계에 놓이면 `UNDETERMINED`가 발화 |
| 6 | 포화 정의 = 마지막 두 점 **집합 동일** | — | 크기 동일보다 강함. 비용 0 |

**새 문턱·새 검정·새 통계량 = 0개**(위 6개는 전부 계측 파라미터이고, 통계적 판정
규칙이 아니다). **arm 0개. 성능 지표 0개.**

---

## 13. 하지 말 것 (제출 전 고정)

- **`sbatch`/`salloc`를 이 사전등록 작성자가 실행하지 않는다.** 제출은
  experiment-runner 소관이며 claims-auditor 감사 통과가 선행 조건이다.
- **dev 트리 수정 금지 · 매니페스트 재동결 금지.** 이 캠페인은 트리 무변경이 전제다.
- **`g2s_run.sbatch`·`test_p5_gate_tools.py`·다른 캠페인 하네스를 건드리지 않는다.**
  이 캠페인의 자기검사는 **공유 unittest discovery에 등록하지 않는다** —
  `g2s_run.sbatch:91`이 그 discovery를 차단 게이트로 쓰고 있어, 새 테스트 파일을
  넣으면 동시 진행 중인 다른 캠페인의 차단 게이트 내용이 바뀐다.
- **성능 주장 금지.** 산출물은 SM 집합이지 지연·처리량이 아니다.
- **경로 3의 `smCount`를 realized로 인용 금지** — Stage 0 D108 오류의 재발이다.
  "드라이버 자기보고 target"으로만 표기.
- **경로 5(타이밍 계단) 채택 금지.**
- **on/off 등가검정으로 프로브 무해 입증 금지** — G5 형태.
- **L1·L2를 R0 결과의 claims-auditor 감사 전에 착수 금지.**
- **설계 문서(`SMID_DIRECT_INSTRUMENTATION_DESIGN_2026-08-11.md`)를 등급어로 인용 금지**
  (claims-auditor 미통과).
- **"실현 파티션을 측정했다"·"prefill 74 SM에서 실행됐다" 금지**(§8-1, 불변).

---

## 14. 감사자를 위한 안내 — **먼저 깨뜨려야 할 곳**

1. **§3.2의 4-outcome 매트릭스.** 내가 놓친 다섯 번째 결과가 있는가? 특히 R0가
   통과하는데도 결과가 무의미해지는 세계가 있는가?
2. **§6.3의 "전역 일관 라벨" 상한.** §0.1의 payoff 3개가 그 상한만으로 실제로
   성립하는가, 아니면 몰래 "물리 인덱스"를 요구하는가?
3. **§10의 "자기 자신이 스모크"** 논거. 게이트 #26 개정 취지(쪼개기 회피 방지)를
   이 논거가 우회하고 있지는 않은가?
