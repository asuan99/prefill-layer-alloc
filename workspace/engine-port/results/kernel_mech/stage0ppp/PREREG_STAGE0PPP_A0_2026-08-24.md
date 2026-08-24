# 사전등록 — **Stage 0‴ A0**: nsys가 green-context **graph-node 커널**을 귀속 가능한 형태로 내는가

작성 2026-08-24 · 규칙층 **미감사**(제출 전 claims-auditor 필수) · **GPU 미집행**
선행 판정: `audit_stage0pp_2026-08-23/VERDICT.md` = Stage 0″ **`NO-GO`**(E1–E11, 死因 없음) + §4 재설계.
규칙 정본: **`stage0ppp_a0_rule.py`** (sha256 `57241ce6010e32d3…`) — 이 문서와 코드가 다르면 **코드가 이긴다**.

---

## 0. 이것이 무엇이고, 무엇이 아닌가

**측정 대상**: nsys가 **green context 스트림 위에서 replay된 CUDA graph의 노드 커널**을
(i) 행으로 내는가 (ii) 그 행을 green context에 귀속시킬 수 있는가 (iii) 그 행을 **어느 replay가
냈는지** 이을 수 있는가.

★**이것은 도구 타당성 판정이다**(2026-08-20 job 886718 P1 프로브 선례). 서버·모델·요청이 **없다**.
지연·처리량·goodput을 **구조적으로 생산할 수 없으므로** 이 프로브의 어떤 출력도 성능 수치로
인용될 수 없다. 어떤 결과도 **HE0·정책 순위·gate #13/#16 어느 것도 건드리지 않는다.**

★**엔진이 필요 없다**(감사 E4). 저장소에 엔진 없이 green ctx를 만들어 완주한 프로브가 **셋**
있다: `smid_census/smid_l0_census.py`(0.0275 GPU-hr, ★엔진이 실제 호출하는 그 함수를 쓴다) ·
`kernel_mech/p1_probe/p1_greenctx_target.py`(0.032) · `s0_deconfound/greenctx_alloc_probe.py`.
Stage 0″는 이 절반을 **B6 서빙 캠페인에 묶어** 사려 했다 — 비싸고 깨지기 쉬운 절반 때문에
공짜로 살 수 있는 절반까지 위험에 노출시키는 구조였다.

## 1. 등록 질문 (E3·E5 반영 — "필드 존재"가 아니라 "채워지는가/조인되는가")

| | 질문 | 형태 |
|---|---|---|
| **Q1** | green ctx replay의 노드 커널 행이 **기대 개수 대비 ≥90%** 나오는가 | 비율, 기대치는 **L2에서 측정** |
| **Q2a** | 그 행의 `greenContextId`가 **`NULL`/0이 아닌 값으로 채워지고** `streamId`가 L3의 green 스트림과 **일치**하는가 | 3분기 |
| **Q2b** | 그 행의 `correlationId`가 **그 replay의 `cudaGraphLaunch` RUNTIME 행과 조인 성공**하는가 | ★성공률 병기 |

★**Q1의 기대치는 가정하지 않는다** — **L2**(full-GPU에서 같은 그래프를 캡처·replay)가 노드 수를
**측정해서** 준다. 이로써 자유 모수 하나가 사라지고 L2가 두 번째 역할을 갖는다.

★**Q2b가 E5다**: `correlationId`가 **캡처 시점 개별 launch**에만 조인되면 필드는 채워져 있는데
멤버십에는 **못 쓴다**. Stage 0″의 규칙은 그 세계를 **"Q2 = 예"** 로 채점했을 것이다.

## 2. 4 다리 — ★대조 3겹 (E2)

한 프로세스에서 순서 실행, **granularity마다 `.nsys-rep` 하나**:

| 다리 | 내용 | 역할 | 실패 시 |
|---|---|---|---|
| **L1** | full-GPU **eager** (green ctx 생성 **이전**) | 양성대조 — nsys가 커널을 **하나라도** 보는가 | `PROBE_INVALID` |
| **L2** | full-GPU **graph** 캡처+replay ×20 | 대조 — 노드 행이 나오는가(**green 무관**) + **기대 노드 수 정의** | `NODE_TRACE_UNAVAILABLE` |
| **L3** | green ctx 스트림 **eager** | 대조 — green 위 커널이 보이는가(**graph 무관**) | `GREENCTX_INVISIBLE` |
| **L4** | ★**green ctx 스트림 graph 캡처+replay ×20** | **본 조건** | 규칙으로 채점 |

P1이 깨끗했던 이유는 **두 겹의 대조**, `%smid` R0는 **3중 대조**였다. Stage 0″는 **0건**이었고,
그래서 부착 오지정·CUDA trace 미활성·조기 종료·export 실패가 **전부 "Q1=아니오"로 오독**된다.

★**green 파티션은 R0와 동일**: `divide_sm(108,(8,0),2)` → `sizes=[74,34]`. `realized_sm` 병기 필수.

## 3. 판정 규칙 = **코드**(게이트 #66 · 감사 E10)

Stage 0″는 규칙을 **산문으로만** 적었고, 감사는 그것을 **바로 전날 등재한 교훈의 즉시 재발**로
채점했다. 이 판본의 규칙은 `stage0ppp_a0_rule.py`에 **코드로 고정**되고 **세계 공간이 전수 열거**된다:

- **8,192 세계**(10축) 전수 채점 · 라벨 **11개 전부 도달 가능** 실증
- **mutant 12개 전부 load-bearing**(파라메트릭 약화 포함 — 삭제 변이만으론 문턱 완화를 못 잡는다, 교훈 #70)
- **판별 검사 3개**는 ★**지정 mutant에서 실제로 실패함을 실증**(`T8`)
- 구조적으로 참인 검사는 `[REGRESS — 근거 아님]`으로 **분리 표기**

★**자기 발견**: 초판 `T6b`("stream_only ⇒ ctx≠distinct")는 **항등식**이었고 `T8` 메타 검사가
**첫 실행에서 잡았다**(게이트 #9의 17번째 재발, 이번엔 **작성 중 자기 검출**). 위험한 방향으로
교체 — *"ctx가 못 쓰는 세계가 ctx 근거로 **보고**되지 않는가"*(= `stream_only`를
*"greenContextId가 된다"* 로 쓰는 오버클레임 경로, 게이트 #61/R0 라벨-인덱스 선례).

### 라벨과 ★각 라벨이 **허가하지 않는 것**

| 라벨 | 뜻 | ★금지 |
|---|---|---|
| `MEASUREMENT_ABSENT` | nsys 실행/export 실패 | 어떤 판정도 금지 |
| `PROBE_INVALID` | L1 실패 **또는 green 캡처 실패** | ★캡처 실패는 **측정 조건**이지 "Q1=아니오"가 아니다(게이트 #21) |
| `NODE_TRACE_UNAVAILABLE` | L2에 노드 행 0 | **green과 무관한 도구 한계** |
| `GREENCTX_INVISIBLE` | L3 행 0 | P1과 **같은 형태의 사실**. 성능 함의 없음 |
| `PRIMARY_ESTIMAND_UNCONSTRUCTIBLE` | Q1=아니오 | ★★**"트랙 종결" 아님**(E1: (b)(c)(d) 잔존) |
| `CONTRADICTORY_ATTRIBUTION` | ctx는 green, stream은 아님 | 두 채널 불일치 = **하네스/분석기 버그** ⇒ 판정 안 함 |
| `KSET_JOINS_CAPTURE_NOT_REPLAY` | ★E5 | 필드는 있으나 **멤버십 불가** |
| `KSET_NEEDS_TIME_ATTRIBUTION` | 조인 0 | E1-(d) 경로로 이동 |
| `KSET_CONSTRUCTIBLE_PARTIAL` / `KSET_CONSTRUCTIBLE` | 조인율 <0.95 / ≥0.95 | ★**basis 병기 필수**(`ctx+stream` vs `stream_only`) |

★`basis`를 **절대 라벨에 흡수하지 않는다**. `stream_only`(E1-(c) 경로)는 멤버십이 서지만
`greenContextId`가 **아니라** `streamId`로 선 것이며, 그 둘을 같은 문장으로 쓰면 R0의
*"전역 일관 라벨 ≠ 물리 SM 인덱스"* 와 같은 종류의 오버클레임이 된다.

## 4. ★ E1-(b) 대체 경로를 **같은 job에서** 산다 (설계 개선, 2026-08-24 GPU 0 발견)

`nsys profile --help`(2025.3.2) 실측: *"If 'node' is selected, node activities will be
collected, **but CUDA graphs will not be traced as a whole**."* ⇒ **`node`와 `graph` 입도는
상호 배타**이고, 1차 결정량(`node`) 실행은 **E1-(b)를 동시에 답할 수 없다**. 프로브가 작으므로
**4 다리를 두 입도로 각각** 실행하고 `graph` 실행은 **companion 규칙**(32 세계, mutant 4개,
판별검사 1개 + 메타)으로 채점한다.
⇒ ★**`PRIMARY_ESTIMAND_UNCONSTRUCTIBLE`이 떠도 트랙이 빈손이 되지 않는다** — (b)의 생사를 이미 안다.

★**답하지 않는 것**: `[:<launch origin>]` 접미사는 `host-only|host-and-device`(= 호스트/디바이스
**코드** 기원)이며 **캡처/replay 시점을 가르지 않는다** ⇒ **E5를 닫지 않는다**. (이 문자열을 E5의
답으로 읽는 것이 바로 저지르기 쉬운 오독이라 명시 기록한다.)

## 5. 환경 — 등록값 (E6) · ★컴퓨트 노드 재확인 필수

| 항목 | 등록값 | 확인 |
|---|---|---|
| nsys | **2025.3.2.474-253236389321v0** | 2026-08-24 실측(`/apps/cuda/13.0.2/bin/nsys`) |
| CUDA | **13.0.2** | module |
| driver | **580.105.08** | ★**로그인 노드 값 — 컴퓨트 노드에서 재확인해 아티팩트에 기록**(게이트 #46: 로그인 노드 귀속 오류 선례) |
| GPU | A100-SXM4-80GB | 아티팩트 기록 |

## 6. 프로파일러 호출 — **전 스위치 등록** (E7)

```
nsys profile --trace=cuda --sample=none --cpuctxsw=none --gpu-metrics-devices=none \
             --cuda-graph-trace=node  -o <out>/a0_node_<jobid>  --force-overwrite true \
             python3 stage0ppp_a0_probe.py --granularity node
# 두 번째 실행: --cuda-graph-trace=graph, -o a0_graph_<jobid>, --granularity graph
nsys export --type sqlite <rep> -o <sqlite>
```

- `--sample=none`·`--cpuctxsw=none`: 이 클러스터 `perf_event_paranoid=2`이고 K1 오염원.
- `--gpu-metrics-devices=none`: **rev8 P5가 "System scope 샘플러가 `G_intra`를 오염"이라 해놓고
  OFF를 등록하지 않았다** — 여기서 명시 등록.
- 감싸는 프로세스는 **프로브 단일 프로세스**(서버·클라이언트 2프로세스 문제 자체가 없다).
  ★Stage 0″의 최대 위험(클라이언트를 감싸면 GPU 행 0 → "Q1=아니오" → 트랙 종결)이 **구조적으로 소멸**.
- ★**`--exclusive`·`--constrain=hwperf` 불필요**: CUDA API/커널 트레이스는 그 권한을 요구하지
  않는다(그 권한은 `--gpuctxsw`·system-wide sampling·GPU metrics 전용). P1처럼 exclusive를 쓰면
  **가격이 최대 8× 틀어진다**(감사 E7). ⇒ `--gres=gpu:1` **비exclusive**.

## 7. 자유 모수 **재열거** (E11 — rev8이 4연속 지적당한 결함)

1 SLURM 할당 모드(비exclusive) · 2 `--trace` · 3 `--sample` · 4 `--cpuctxsw` ·
5 `--gpu-metrics-devices` · 6 입도(**node/graph 2회**) · 7 launch origin(기본) ·
8 `-o`/`--force-overwrite` · 9 종료 방식(프로세스 정상 종료) · 10 `nsys export --type sqlite` ·
11 다리당 replay 수(**20**) · 12 green 파티션(`(8,0),2`→`[74,34]`) · 13 spin 커널 grid/지속 ·
14 캡처 전 warmup 수 · 15 스트림 배치(어느 것이 green 스트림인가) · 16 torch/CUDA 모듈 버전 ·
17 아티팩트 형식(**`.json` 단일 정본**) · 18 결정성(무작위 요소 없음)
⇒ **18개**(Stage 0″의 "전수 6개"는 거짓이었다).

## 8. 예산

| 항목 | 값 |
|---|---|
| 다리 | 4 × 2 입도 |
| 벽시계 | ≤ **20 min** (`--time=00:20:00`) |
| GPU | ≈**0.03–0.05 GPU-hr**(선례: R0 0.0275 · P1 0.032, 비exclusive 단일 GPU) |
| 엔진 부팅 | **0** |

## 9. 아티팩트 규율

- ★**`.json` 단일 정본**. `.txt`는 콘솔 전사이며 **인용 불가**(게이트 #56 — R0 결함 N1: `.txt` 절단으로 3필드 소실).
- ★**필드명과 값의 극성 일치**(게이트 #57 — R0 결함 N2: 필드 이름이 값의 부정이었다).
- 병기 필수: nsys/CUDA/driver 버전(컴퓨트 노드) · `realized_sm` · 규칙 파일 sha256 · 조인 성공률 · basis.

## 10. ★ 쓰면 안 되는 문장

**승계**: *"kernel_mech 트랙을 닫았다/열었다"* · *"rev7·rev8 차단이 해소됐다"*(B1–B5·B7 / C1–C10 **불변**) ·
HE0 · gate #13/#16 *"닫았다"* · switch-cost *"닫았다"* · C2 인용정지 (a)(b).
**신설·승계**:
- ✗ *"`PRIMARY_ESTIMAND_UNCONSTRUCTIBLE`이면 트랙이 종결된다"*(E1 반증 — (b)(c)(d) 잔존)
- ✗ *"엔진에 NVTX가 없다"* / *"`grep -rn nvtx src/` = 0건"*(**틀린 트리** → `../NVTX_EVIDENCE_CORRECTION_2026-08-24.md`)
- ✗ *"NVTX 선행조건이 사라졌다"*(코드 독해 기반 **가설** — Q1·Q2b가 실측으로 답한다)
- ✗ *"nsys는 green-context 커널을 낸다"*(★**A0 실행 전 금지**)
- ✗ *"Q2는 도구 문서로 이미 답이 나왔으므로 프로브가 불필요하다"*(문서층만 답했다 — 이 드라이버에서 **채워지는지**는 미측정)
- ✗ *"launch origin이 캡처/replay를 가른다"*(§4 반증)

## 11. 다음 단계 — ★**규칙층 감사(게이트 #34 1단) 먼저**

이 문서는 **미감사**다. Stage 0″는 규칙층에서 `NO-GO`(차단 11)를 받았고, P0-A는 규칙층 감사를
**5회** 거쳐 수렴했다. 순서: **① 규칙층 감사 → ② 하네스(`stage0ppp_a0_probe.py` + `.sbatch`) 작성
→ ③ 하네스층 감사(2단, E11) → ④ 제출**. ★**②·③ 없이 제출하지 않는다.**

## 12. 변경 이력 — ★**각 행에 검증 방법 병기**(게이트 #62/#67/#70)

| 판본 | 변경 | ★검증 방법(예측 아님, **실행**) |
|---|---|---|
| Stage 0″ | (전신) | 규칙층 감사 `NO-GO`, 차단 E1–E11 |
| **A0 rev1** | 규칙을 **코드로 고정** + 8,192 세계 전수 열거 | `python3 stage0ppp_a0_rule.py` → 전 검사 PASS(출력 첨부) |
| A0 rev1 | 대조 3겹(L1·L2·L3) 신설 | `T4`(+`T8` 메타): 대조 실패 세계에 실질 라벨 0건, `g_l1`에서 **실제로 실패함**을 실증 |
| A0 rev1 | E5(캡처-시점 조인) 분기 신설 | `T7`(+`T8`): `capture_only` 세계가 `KSET_CONSTRUCTIBLE`로 절대 안 감, `g_capjoin`에서 실패 실증 |
| A0 rev1 | E1-(c) `stream_only` basis 신설 | `T6` 도달성 n=24 · `T6b`(+`T8`) `g_basis`에서 실패 실증 |
| A0 rev1 | ★초판 `T6b`가 **항등식**이어서 교체 | `T8` 메타 검사가 **첫 실행에서 검출**(콘솔 `FAIL` 기록) |
| A0 rev1 | E1-(b) companion(입도 `graph`) 신설 | `nsys profile --help` 실측 인용 + companion 32 세계 `C1–C4` PASS |
| A0 rev1 | 캡처 실패 = 측정 조건 분기 | `g_capture` mutant load-bearing(n=512) |
