> ⚠️**SUPERSEDED (2026-09-13, 같은 날 사용자 결정, doc-steward
> 등재)** — 이 사전등록은 **Zamba2-2.7B 기준**으로 설계됐다.
> 사용자가 같은 세션에서 4모델 캠페인의 Zamba2 슬롯을
> `NemotronHForCausalLM` Nano-9B-v2-Base(ctx 131072)로
> 교체하기로 결정했다 — 아래 §10.3이 스스로 예고한 대로
> "모델 교체면 X3는 전량 무효(ctx·커널 shape·프로토콜 전부
> 변경)"다. **이 문서·규칙층 감사(`VERDICT_x3_rules_2026-09-13.md`)
> 는 이력 보존용으로만 유지하며, 재구매·재인용의 근거로 쓰지
> 않는다** — 단 §14의 방법론 게이트 G-X3-1…5는 방법론
> 지식으로서 유효하다(`CONSENSUS.md` §3 항목210–214,
> `PROJECT_STATUS.md` "방법론 게이트" #190–194). GPU 0 지출,
> 결과 없음. 전문은 `PROJECT_STATUS.md` 최상단 배너(2026-09-13)
> "B"·"C" 절 참조.

# 사전등록 — X3: r2_eval **러너 설정** 인증 (선결 #4b의 토큰 쪽 절반)

작성 2026-09-13 (메인 세션), **GPU 미지출 상태에서 고정**. 1 job ≈ **0.16 GPU-h**
(이 트랙 0.43 → **0.59**). claims-auditor가 OS 판정서 §8에서 **"A1 커밋 → X3 → OS"** 순서로
X3를 1순위로 권고했다(`../os_prereg/VERDICT_os_rules_2026-09-12.md` sha `4c7cfa94…`). A1(러너
수리)은 커밋 `09a8075`로 완료됐다.

## 0. A1 이후 남은 질문이 무엇인지 (질문이 작아졌다)

X3가 로드맵에 적힐 때(2026-09-11)의 블로커는 **ctx 16384로 Zamba2-2.7B가 부팅 거부**되는 것과
`r2_eval.sbatch:11`의 경로 결함이었다. 둘 다 A1이 고쳤고, 러너가 조립하는 서버 인자 튜플은
**CPU 테스트로 이미 인증**됐다(3경로 + dry-run 덤프 + 되돌린 변이 음성대조). 그래서 **지금 남은
것은 둘뿐**이다:

1. 그 튜플로 **서버가 실제로 뜨고 서빙하는가**(CPU 테스트로는 알 수 없다).
2. ★**정확성 게이트가 쓰는 계측이 토큰을 바꾸지 않는가.** 907100·907456은 eval 캠페인이 **절대
   쓰지 않을** 계측 구성에서 측정됐다:

| 계측 | 정확성 게이트(907100) | r2_eval 러너 |
|---|---|---|
| `PDMUX_TELEMETRY_PATH` | 설정 | **설정**(`r2_eval.sbatch:105`) — 차이 없음 |
| `PDMUX_DUAL_WORKER_TRACE_EVERY` | 기본 32 | 기본 32 — 차이 없음 |
| **`PDMUX_TRACE_FORCE_PREFILL`** | **1** | **미설정(=0)** |
| `PDMUX_GREEN_READOUT` | 1 | 미설정 |
| `--decode-log-interval` | 1 | **없음** |
| `--random-seed` | 1 | 캠페인 `server_seed`(예: **1000**) |

선결 #4b(observer effect)의 **토큰 쪽 절반**은 바로 이 차이가 greedy 출력을 바꾸는지다.

## 1. ★설계의 핵심 긴장: **측정 도구가 곧 처치다**

`PDMUX_TRACE_FORCE_PREFILL=1`은 코드 주석이 "**OBSERVATION ONLY: nothing below this line feeds
scheduling, partition selection or batch composition** … forced records are strictly ADDITIONAL
and are tagged `trace_forced=true`"라고 못박은 대로 **상태 경로를 건드리지 않는다**
(`multiplexing_mixin.py:552-556`). 남는 실효는 **스케줄러 스레드의 추가 작업**
(payload 구성 + `_r2_runtime_snapshot()`의 백분위 계산·풀 점유 조회)이다 — **타이밍 효과**.

그런데 같은 주석의 실측이 결정적이다: **prefill은 wall time의 4.9%인데 *스케줄된* 스냅샷의
0.07%에만 나타난다**(`:545-551`, E1/T8/d44). 즉 **강제 샘플은 O층(중첩·D44 실현)을 관측 가능하게
만드는 유일한 도구**다. 따라서 러너 구성(강제 샘플 OFF)으로 돌리면:

- **O1–O4 확인이 실패할 것이 예상되고**(O2는 probe 창 안에서 prefill∧decode 동시 스냅샷을
  요구한다), 그 결과 checker는 `unrealized`를 채워 **`NO_VERDICT_UNREALIZED`를 출력한다
  — 구성상 그렇게 되는 것이며 게이트 실패가 아니다**(교훈 21).
- **O층 토큰은 그대로 생성되고 비교 가능**하다. 다만 그 8단위의 **프로토콜 확인(중첩·D44 상주)은
  907100에서 상속**되는 것이고 이 job이 재확인하지 않는다 — 반드시 병기한다.
- ★**907100 원자료로 이 기전을 직접 확인했다**(`forced_sample_dependency.py`, 출력 고정
  `forced_sample_dependency_stdout.txt`): `job_907100`의 prefill∧decode 동시 스냅샷은
  **L1 600건 중 582건(97.0%)·TD1 558/541·L2 536/517·TD2 532/517이 강제 샘플**이고 전부 idx 4(D44)였다.
  ⇒ 강제 샘플이 없으면 boot 전체에 **스케줄된 동시 스냅샷이 L1 18·TD1 17·L2 19·TD2 15건**만
  남고, O2는 그중 하나가 **8개 probe 창 안에** 들어와야 확인된다. 따라서 O 확인 실패가
  **예상**이지만 **확실하지는 않다**(15–19건이 0이 아니다).
- **B7**(job 전체에서 동시 스냅샷 ≥1)은 위 수치로 **통과가 예상된다**(스케줄분 15–19건 > 0).
  실패하면 측정 구성의 귀결로 기록하고 게이트 실패로 적지 않는다.

⇒ **이 job의 1차 산출물은 게이트 판정이 아니라 교차-잡 토큰 비교다.**

## 2. 실험 설명 (통제 / 변인 / 예상 / 진행 가부)

### 통제 요인 (907100과 동일)
모델 Zamba2-2.7B · TP=1 · `pdmux_r2.yml` · `PDMUX_R2_POLICY=fixed` · `R2C_DSM=44` · greedy(T=0) ·
`ignore_eos` · 고정 `max_new` · ctx **4096** · cudagraph-ON(decode) · `--attention-backend triton` ·
`--disable-radix-cache` · `--chunked-prefill-size -1` · `--disable-overlap-schedule` ·
`--max-running-requests 48` · `--mem-fraction-static 0.82` · boot 순서 `L TD L TD` · 클라이언트 불변 ·
`R2C_NUM_KV_SPLITS` 미설정(서버 기본 8 = 운영 수치 구성) ·
**판정 규칙 `r2_correctness_check.py` 무수정**(rule v2, sha `ec355e17…`).

★**엔진 소스 핀**: 제출 전 `git checkout 38c1aca -- workspace/engine-port/src/multiplex`로
manifest multiplex 8항목을 907100·907456과 **바이트 동일**하게 만든다. OS 감사 D2가 확인한 대로
이 핀은 **하중재**다 — 현재 트리의 `dual_worker.py`는 38c1aca 대비 +172줄이고 그 변경(H2)이
**true-dual hot loop의 계측 발행 순서**를 바꾼다. 핀 없이 돌리면 계측 구성과 엔진 소스를 동시에
바꾼 것이 되어 confound #10으로 무효다. 제출 전 `git status --porcelain` **전체**를 기록하고,
완료 후 `git checkout HEAD -- …`로 복원한 뒤 **manifest 재계산으로 복원을 검증**한다.
(OS 감사 D1은 이미 해소 — A1 작업이 `09a8075`·`ae7830e`·`3153260`으로 커밋됐다.)

### 변인 요인 — **러너 구성 묶음(2개 동시, 의도적)**
- **`R2C_TRACE_FORCE_PREFILL=0`** (러너가 하는 대로. sbatch 신규 노브, 기본값 1은 907100 동작 보존)
- **`R2C_SEED=1000`** (캠페인이 실제로 주는 `server_seed`)

★**묶음으로 등록하는 이유와 그 대가**: 추정량은 "**러너 구성이 정확성 게이트 구성과 같은 토큰을
내는가**"이고, 그것이 "러너 설정 인증"의 정의다. 둘을 분리하는 것은 **불일치가 나왔을 때만**
필요하다 — 그 경우 단일 변인 후속 job 1개(0.16)가 필요하며, 이 문서는 그것을 **조건부 후속**으로
등록한다. 불일치가 0이면 두 변인 모두 토큰-중립임이 동시에 증명된다.

### ★검증하지 **않는** 잔여 관측자 (명시)
- **`--decode-log-interval 1`은 유지한다.** 러너에는 없지만, 이것이 **cudagraph-ON의 유일한
  증거 채널**(checker B3)이다. 빼면 이 job에서 cudagraph 상태를 확인할 수 없다 ⇒ 유지하고
  "러너 구성과의 잔여 차이 1건"으로 공시한다. 실효는 decode step당 로그 1줄(상태 효과 없음).
  **권고(범위 밖)**: eval 러너에 `--decode-log-interval 1`을 넣어 모든 성능 run이 per-boot
  cudagraph 증거를 갖게 하라 — 비용이 거의 없고 지금은 그 증거가 **0**이다.
- **`PDMUX_GREEN_READOUT=1`도 유지한다.** 코드상 `__init__`에서 1회 호출되는
  **startup-only**(`multiplexing_mixin.py:125`)이므로 정상 상태 관측자가 아니고, checker B8
  (D44 실현)의 유일한 근거다. 공시만 한다.

### 진행 가부
사용자 승인 완료(권고 순서). Claim D 선결 중 **#4b의 토큰 쪽 절반에만** 기여하고 **아무것도 닫지
않는다**. 성능 측정 0건.

## 3. 추정량 · 분석 코드 (데이터 생성 전 고정)

- **1차(교차-잡 토큰)**: `../x1_prereg/x1_cross_job_compare.py`(sha `9ba6b230…`, **기존 등록·검증
  완료분 재사용**)로 `job_907100` ↔ X3 job을 비교. 4 boot label × (S 16 + O 8) = **96 unit-pair**.
  이 도구는 (i) 분모 96과 gen 4/4를 요구하고 미만이면 등록 문장을 무효로 선언, (ii) 입력을
  `prompt_sha256` **및** `prompt_tokens`로 검사해 불일치 단위를 제외·보고, (iii) 불일치를
  `first_divergence ≥ 1`일 때만 교란에 귀속, (iv) S·O 층을 분리 보고, (v) C 층은 gen 4/4 미만이면
  `NOT-RUN`을 인쇄한다. sbatch가 `R2C_BASELINE_JOB`으로 job 안에서 자동 실행한다.
- **2차(per-boot 사실)**: checker가 내는 B1–B8·O1–O4·`unsafe_decisions`·S/O/C 쌍 통계.
- **귀무대조(사전 등록)**: `907032 → 907100` S 층 **compared=32, mismatch=0**(노드 gpu42→gpu38,
  커밋 `02918e8`→`38c1aca`, 계측 구성 **동일**, 교란 없음). ★**O 층에는 귀무대조가 없다**
  (907032에 O 층 부재). 양성대조는 X1이 제공했다 — 같은 비교기로 **S05@25·O06@11 2단위가
  4 boot 전부에서 뒤집혔다**(job 907456), 즉 이 비교기는 **decode attention 축약순서급 교란에
  민감함이 시연된 상태**다(X1P-3′의 한정 범위 안에서).

## 4. 예보와 출력공간 라벨 (F1–F5)

- **F1 (부팅·서빙, 1차 목적의 절반)**: 4 boot 전부 `booted=True`·`crash_free=True`. 이것이
  "러너가 조립한 ctx·인자로 서버가 실제로 뜬다"의 증거다. `BOOT_FAILED`면 **A1의 ctx 유도가
  실엔진에서 틀렸다는 뜻**이므로 즉시 engine-porter로 이관(측정 실패 아님 — **이 job의 1차 발견**).
- **F2 (토큰, 1차 목적의 절반)**: 96 unit-pair 중 **불일치 0**.
  | 실현 | 등록 라벨 |
  |---|---|
  | 0/96 | **러너 구성은 토큰-중립**이다(강제 샘플 OFF + seed 1000 묶음). ⇒ 907100·907456의 토큰 동치 결론이 **eval 캠페인의 계측 구성으로 이전된다**(S 16·O 8 단위, 이 프로토콜 한정). 선결 #4b의 **토큰 쪽 절반만** 해소, 타이밍 쪽(<3% paired)은 **미측정**. |
  | ≥1, `first_divergence ≥ 1` | **러너 구성이 토큰을 바꾼다.** 묶음이므로 **어느 변인인지 모른다** ⇒ 단일 변인 후속 job 1개를 **조건부 등록**(seed만 1000, 강제 샘플 1 유지). 그리고 **907100·907456의 결론은 eval 구성으로 이전되지 않는다**고 등재한다(Claim D 선결 #2의 스코프가 **좁아진다** — 불리한 방향이며 숨기지 않는다). |
  | ≥1, `first_divergence = 0` | prefill(extend) 산출이므로 계측으로 설명되지 않는다 ⇒ **해석 보류 + 별도 조사**(X1 D2(ii) 승계). |
  | 분모 ≠ 96 또는 gen 4/4 미만 | 등록 문장 **무효** — 분모·제외 목록을 명시 재기술(비교기가 자동 경고). |
- **F3 (구성상 예상되는 비판정)**: `O_TIER_CONFIRMED` 실패 → **`NO_VERDICT_UNREALIZED`**.
  ★**게이트 실패가 아니다.** 기전은 §1(강제 샘플이 O층의 유일한 관측 도구, 4.9% vs 0.07%).
  - 만약 **O가 확인되면**(1-in-32 격자가 운 좋게 창 안에 들어오면) 그것은 **추가 정보**로
    등재한다("러너 구성에서도 O층 확인이 가능했다") — 단 그 확인은 **샘플 운**에 의존하므로
    일반화하지 않는다.
  - `FAIL`(S_TIER_MISMATCH 또는 O cross-arm)이 나오면 그것은 **진짜 게이트 신호**이므로 즉시
    등재·escalate한다(F2 ≥1과 같은 사건을 다른 층에서 본 것일 수 있다).
- **F4 (B7)**: B7은 **통과가 예상된다**(907100의 스케줄분 동시 스냅샷 15–19건/boot). 실패해도 **측정 구성의
  귀결**이며 게이트 실패로 적지 않는다.
- **F5 (provenance)**: `trace_force_prefill=0`·`seed=1000`·env 덤프·`multi_processor_count`·
  manifest 17항목·`verdict_rule.txt`·**`x3_prereg/*`와 `x1_prereg/*`의 sha256**이 남는다
  (sbatch의 해시 블록을 `*_prereg` 전체로 일반화했다). `x3_prereg/`는 **제출 전에 커밋**한다.

## 5. 이 실험이 **주지 않는 것** (문자 승계)

- **선결 #4b를 닫지 않는다** — 토큰 쪽 절반이고, **타이밍/성능 쪽(observer effect < 3% paired
  CI)은 측정조차 하지 않는다.** 선결 #1·#2·#3·#4a·#5 상태 불변.
- **R2C-1…16 · P-1…7 · X1P-1′…9 · X1C-10…14 · B3C-1…5 · OSP-1…6 · OSC-1…8 전부 불변.**
  특히 **OSC-4/B3C-3 승계**: 한 job의 4 boot은 독립 런이 아니므로 "n≥4/게이트 3 충족"을 쓰지 않는다.
- **성능 수치 인용 전면 금지**(X1C-9 연장): 907100·907456·이 job 사이의 어떤 지연·스루풋·
  GPU-h·decode step 수 비교도 금지. 강제 샘플 OFF는 **성능에 영향을 주는 설정**이므로 이 금지는
  특히 강하게 적용된다.
- C 층은 여전히 진단 전용(R2C-3)이며, 이 job의 C 층 수치로는 어떤 판정도 하지 않는다.
- **Claim D/E 등급 불변 · HE0 · 정책 순위 · stake #1 불변.**

## 6. 조건부 후속 (F2 ≥1일 때만, 사전 등록)

단일 변인 job 1개(≈0.16): `R2C_SEED=1000` **만** 바꾸고 `R2C_TRACE_FORCE_PREFILL=1` 유지 ⇒
seed 기여분 분리. 그 결과로도 갈리면 강제 샘플 기여분이 남는다. **지금 승인 요청하지 않는다.**
