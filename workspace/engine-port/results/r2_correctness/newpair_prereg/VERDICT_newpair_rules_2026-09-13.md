# 규칙층 감사 판정서 — 새 (모델, 백엔드) 쌍 R2 correctness 게이트 + 계측 3건

**감사 대상**: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/newpair_prereg/PREREG_NEWPAIR_2026-09-13.md`
(동반: `newpair_preflight_o1.py` · `newpair_preflight_o1_stdout.txt` · `../r2_correctness.sbatch`(수정) · `workspace/engine-port/tests/test_r2_correctness_instrument.py`)

**감사자**: claims-auditor (read-only) · **2026-09-13** · **GPU 지출 0**(기존 원자료 + CPU 실행만; `sync_engine_tree.sh`는 엔진 트리를 쓰므로 **실행하지 않았다** — §6-8 참조) · **사전등록은 미제출 상태**

---

## 0. 등급

# `NO-GO`

**死因 1건 — N3(예측 도달 불가)**: 등록 예보 **F2**(= 계측 I1의 전부)의 출력공간에 **거짓이 없다**. `max_mamba_cache_size: 48`과 `max_running_requests=48`은 이 구성에서 **엔진이 강제하는 항등식**이며 ctx·mem-fraction과 **무관**하다. 사전등록은 이것을 "ctx 8704/mem 0.80에서만 측정된 전제"라고 서술하고(§5-I1), 도달 불가능한 실패 분기에 "905835 앵커·다음 사다리 전면 재유도"라는 처분을 달았으며(§7 F2), **NP-7이 그 항등식을 "사실"로 승격**한다. λ0 판정서 §3-8이 처방한 3건 중 **1번 항목이 아무것도 재지 않는다.**

**나머지 설계는 건강하다.** 死因은 문서 수정(GPU 0, 수분)으로 해소되며, 20개 자유 표면 중 **반전 0건**이다. D1 적용 + D2–D13 반영 후 재감사에서 `GO-with-caveats`가 나올 것으로 본다. 이 판정은 **문서 결함에 대한 것이지 설계 결함에 대한 것이 아니다.**

---

## 1. 반전 시험 표 (필수)

각 자유 표면을 **등록이 허용하는 범위 끝까지** 밀었다. 반전 계산에 쓴 원자료·도구는 마지막 열에 명시한다.

| # | 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 / 도구 |
|---|---|---|---|---|
| 1 | `R2C_INSTRUMENT` 가드 | 0 ↔ 1, 그리고 가드 3개 변이 | **없음** | 독립 변이 실행(감사자 자작 하네스, 테스트 파일 미경유): `instr=0,warm=1` → **호출 0 · `instrument/` 미생성**; `instr=1,warm=0` → **0**; `instr=1,warm=1` → **정확히 3**. 변이 `M_invert_INSTR` → instr=0에서 **3**; `M_drop_WARM_OK_leg` → warm=0에서 **3** ⇒ 두 테스트 모두 **비공허** |
| 2 | 채점기 파일 접근면 | 디렉터리 순회 존재 여부 전수 | **없음** | `r2_correctness_check.py` 전 `open()` 6곳 = `boots.txt`·`gen_/srv_/tel_/green_{lb}`·report(w). `glob/listdir/scandir/os.walk/iterdir` **0건**. warm-up은 `boots.txt`에 없고(append 라인 1개, scored 루프 내부) `PDMUX_TELEMETRY_PATH` 없이 뜬다(`multiplexing_mixin.py:87-89` → 빈 경로) |
| 3 | O1의 K(`triton_attention_num_kv_splits`) | 등록: 미설정(서버 기본 8) | **없음**(등록 범위 내) | K=8에서 문턱 1239, 최소 probe 1280. 범위 밖(K=2 → 문턱 177, 전부 통과 / K=16 → 문턱 2655 > 최대 probe 2310, 전부 실패)이지만 `${NKVS:+…}`가 빈 확장이고 §13이 미설정을 등록, 체커 `SAME_ACROSS_BOOTS`가 boot 간 동일성 강제 |
| 4 | 부트 순서 `R2C_ORDER` | `L TD L TD` ↔ `L TD TD L` | **없음** | 라벨 집합·`l_boots`/`td_boots` 분할 불변, 비교 쌍 집합 동일 |
| 5 | **S 티어 귀무대조(부분 부팅 시)** | L boot 1개만 gen 산출 | **FAIL ↔ NO_VERDICT_INFRA 라벨 반전 재현** | 실제 907456 아티팩트 + 1토큰 섭동, **무수정 체커** 실행: R3(gen_L2 삭제만) → `NO_VERDICT_INFRA`; **R2(gen_L2 삭제 + TD1 S 1토큰 변경) → `FAIL`, `failures=['S_TIER_MISMATCH'] infra=['L2:B1']`**. 원인 = `s_ref_ok = all(combinations(['L1'],2))` = **공허 참**(O 티어는 `len(l_boots)>=2 and len(td_boots)>=2`를 명시 요구하는데 S 티어는 안 함) — **단, 구동 요인이 연구자 재량이 아니라 결과이므로 N2로 계수하지 않는다**(§2-B) |
| 6 | `--time` (CLI 02:00:00 ↔ 스크립트 기본 01:15:00) | 기본값으로 제출 | 판정 **소멸**(job 강제 종료, verdict 미생성) | 등록된 최악 71분 < 75분이라 기본값도 4분 여유로 "맞는다". 그러나 **실제 최악은 ≈146분**: 계측 3000 s(=50.0분) + scored 4×(health 480 s + client 900 s + teardown 20 s)=5600 s(=93.3분) + 부대 ≈3분 ⇒ **02:00:00로도 초과 가능**. 처분은 §9(f)에 등록돼 있음 |
| 7 | F5 판정 슬라이스(전역 max ↔ 셀별 max) | 산출물이 주는 범위 끝까지 | **없음(수치 반전 미생성)**, 단 **F5는 산출물로 판정 불가** | `I3_max_running_req.txt` = `srv_warmup.log` **전역** 1개 값. F5는 "두 셀 **모두** 48 도달"을 요구 ⇒ 48이 나와도 "적어도 한 셀"까지만 말할 수 있다. I2는 동시성 1이라 오염 없음(max ≤1). `I_log_offsets.txt`로 오프라인 재계산은 **가능**(경계 ±1행 모호성은 knife-edge 외 무영향) |
| 8 | 재제출 횟수(미등록) | 무제한 재추첨 | **수치 반전 미생성** | 교차-잡 토큰 불일치는 **측정됨**(X1 감사: 907100↔907456 8/96 단위) ⇒ 재추첨은 출력을 실제로 바꾼다. 그러나 관측된 2개 job이 **둘 다 PASS**라 라벨 반전 사례가 없다 ⇒ caveat |
| 9 | mem-fraction 0.82 ↔ 0.80(§9-b 재등록 경로) | 등록이 허용하는 양 끝 | **없음** | 구속 자원 48은 두 값에서 동일(死因 §2-A와 같은 이유). KV는 `max_total_num_tokens=2,618,868`로 **6.6× 과공급**, mem 0.02 차이 ≈1.6 GB는 `available_gpu_mem=13.38 GB` 안 |
| 10 | ctx 16384 ↔ 8704 | 양 끝 | **없음** | `_resolve_max_num_reqs`의 `estimated = max(min(…,4096),2048) ≥ 2048 ≫ 48`이라 ctx가 `max_running_requests`를 못 건드린다. `req_to_token` 증분 48×16384×4 B = **3.1 MB** |

**반전 계산 자신의 자유 표면 점검(전례: P6)**: 표의 1·2·5는 **무수정 출하 코드**(`r2_correctness_check.py` sha `ec355e17…`, sbatch에서 마커로 추출한 블록)를 실행한 결과이고, 5의 원자료는 **job 907456 실제 아티팩트**(symlink, 무변경)다. 내가 만든 것은 `gen_TD1.json`의 `phase_s[0].output_ids[0] += 1` **단 1비트**와 `gen_L2.json` 삭제뿐이다. 기준선으로 같은 복제본에서 **PASS를 비트 단위 재현**했다(S mismatches 6쌍 전부 0, C 진단 5/3/5/4/4/3 일치).

---

## 2. 死因 상세

### N3 — 예측 도달 불가 (F2 = 계측 I1)

**주장**: 사전등록 §5-I1 / §7-F2 / NP-7은 `max_mamba_cache_size = max_running_requests = 48`을 **ctx·mem에 따라 달라질 수 있는 경험적 사실**로 다루고, "48보다 작으면 905835 앵커와 다음 회차 사다리를 전부 재유도"라는 처분을 등록했다.

**반증(코드, 결정적)**: 이 게이트의 서버 명령은 `--disable-radix-cache --max-running-requests 48`을 **핀으로 고정**하고 있다(`server_cmd()`, 테스트가 바이트 단위 고정). 그 조합에서:

```python
# sglang/srt/model_executor/model_runner_kv_cache_mixin.py:223-230
elif (
    server_args.disable_radix_cache
    and server_args.max_running_requests is not None
):
    # Use explicitly set max_running_requests when radix cache is disabled
    server_args.max_mamba_cache_size = server_args.max_running_requests // (
        server_args.dp_size if server_args.enable_dp_attention else 1
    )
```

- `dp_size=1`, `enable_dp_attention=False` ⇒ **`max_mamba_cache_size = 48`, 나눗셈도 항등**.
- "메모리에 맞춰 자동 산정"하는 `else` 분기(`:231-248`)는 **radix cache가 켜졌을 때만** 탄다 ⇒ ctx·mem-fraction은 이 값에 **들어가지 않는다**.
- ⇒ 배너의 `max_running_requests` 역시 강제: `_resolve_max_num_reqs`(`:857-876`)에서 `estimated = max(min(token_capacity/ctx*512, 4096), 2048) ≥ 2048`, `max_num_reqs = min(48, estimated) = 48`, 이어 `min(48, max_mamba_cache_size // ratio)`인데 `_calculate_mamba_ratio`(`:390-392`)가 **`disable_radix_cache`면 즉시 1을 반환** ⇒ `min(48, 48) = 48`.
- ⇒ **F2의 "48보다 작음" 분기는 부팅이 성공하는 한 도달 불가능하다.** 메모리가 모자라면 값이 줄어드는 게 아니라 **부팅이 죽는다**(그건 F1/§9-b의 관할).

**아울러 승계된 정본 서술이 틀렸다(정정 필요)**: λ0 판정서 §3-5는 이 48을 "`srv_d44_r2_o96.log` 실측"으로 인용했고 사전등록 §5-I1은 "그 전제는 ctx 8704 / mem 0.80에서만 **측정**됐다"고 적었다. 그러나 job 905835의 서버 명령도 **`--disable-radix-cache --mem-fraction-static 0.80 --max-running-requests 48`**이다(`results/longctx_conflict/probes/c_capacity.sbatch:126-127`) ⇒ **905835의 48도 측정이 아니라 같은 항등식이다.** 즉 이 전제는 **어디에서도 측정된 적이 없고**, 측정할 필요도 없다.

**왜 이것이 caveat가 아니라 死因인가**:
1. N3의 문언 그대로 — 등록 예보의 출력공간에 거짓이 없다.
2. 이 job을 **지금** 사는 근거(§0 + λ0 §8)가 계측 3건이고, 그중 **I1이 전부 공전**한다. 구매의 1/3이 무효인데 등록이 그것을 모른다.
3. NP-7이 인용 금지 조항인 척하면서 **"I1이 48을 확인해도 그것은 이 튜플에서 구속 자원이 유지된다는 사실"**이라고 항등식을 사실로 승격한다 ⇒ 다음 회차 λ0 재등록이 "ctx 16384에서 구속 자원을 측정으로 확인했다"를 **합법적으로** 인용하게 된다. 이는 교훈 9 계열(게이트/확증서술 자신이 항등식, 20회+ 재발)과 게이트 #195(provenance 허위) 계열의 정확한 재발 형태다.

**N1/N2/N4는 발화하지 않는다**:
- **N1 아님** — 이 게이트의 판정량(두 아키텍처의 토큰 id 동일성)은 항등식이 아니다. 907032가 FAIL, 907100이 PASS를 낸 실적이 있고, 출력공간 5개 라벨 전부 도달 가능하다.
- **N2 아님** — §1 표 20개 표면 중 **연구자 재량 표면에서 반전 0건**. 표면 5(공허 수량자)는 라벨을 실제로 뒤집지만 구동 요인이 **부팅/클라이언트 실패라는 결과**이지 재량이 아니다. 정직하게 **D2로 강등**한다(B3 사전등록에서 같은 형태를 N2로 계수한 전례가 있으나, 그 사례는 `all([])`가 **arm 선택이라는 재량**에 걸려 있었다 — 여기서는 걸려 있지 않다. 전례를 등록 상수로 승격하지 않는다: 게이트 #110).
- **N4 아님** — 이 job은 정책 주장을 사지 않는다. §2가 성능·Claim D·다른 split/정책/모델/백엔드·샘플링·C층을 전부 문자로 막았고, I2/I3는 **PD-mux D44 운영점 · cudagraph ON**에서 재며 NP-2/NP-3이 arm 비교와 λ* 승격을 금지한다.

---

## 3. 항목별 판정 (지정 표면 1–8 + 감사자 추가)

### 1 ★"추가 scored boot 0 · 채점 경로 무영향"은 항등식인가 — **아니다. 독립 재검증으로 참임을 확인했다.** (반증 실패)

주장이 **구조적 사실**이고 검사 가능하다. 세 다리 전부 코드로 확인:

- **채점자가 무엇을 읽는가**: `r2_correctness_check.py`의 `open()`은 6곳뿐 — `boots.txt`, 라벨마다 `gen_/srv_/tel_/green_`, 그리고 report 쓰기. 디렉터리 순회 식별자(`glob`·`listdir`·`scandir`·`os.walk`·`iterdir`) **0건**. 계측 산출물은 전부 `$OUT/instrument/` 아래.
- **warm-up이 `boots.txt`·telemetry에 들어가는가**: `: > "$OUT/boots.txt"`가 warm-up **종료 후**(`stop_server` 다음) 실행되고, append 라인은 scored 루프 안 **1개**뿐. warm-up은 `PDMUX_TELEMETRY_PATH` 없이 뜨고(스크립트가 `PDMUX_*`를 전부 unset), `multiplexing_mixin.py:87-89`가 빈 경로를 해석 ⇒ `tel_*.jsonl` 산출 없음.
- **가드가 프로세스를 1개도 안 띄우는가 + 변이 시험**: §1 표 행 1의 수치. 특히 **`M_drop_WARM_OK_leg`가 `TestWarmupDown`을 실제로 깨뜨린다**(warm=0에서 호출 3) — 즉 그 테스트는 공허하지 않다. 사전등록이 등록한 음성대조 3건 외에 **감사자가 독립으로 2건을 더 만들어** 확인했다(교훈 53 충족).
- 전체 테스트 **27/27 OK**(사전등록이 주장한 27케이스와 일치). 채점 코드 sha `ec355e171a66d68eab1300edcc7616a693cd55aa2c3fb5c251ee4b42eac50d30` = 두 테스트의 핀과 일치.

**물리적 결합도 점검**(파일 접근 외): 50분 부하가 scored boot의 **토큰 등가성**에 영향을 줄 경로는 없다(그리디 디코딩, 시드 고정). O2/O4만 타이밍 의존인데 열 지연은 중첩을 **늘리는** 방향이다. 남는 하나는 **GPU 메모리 잔류**(D12).

### 2 ★O1 여유 3.3%(41 토큰) — **사전계산은 맞고 방법론도 옳다. 처분도 게이트 #21과 정합.** (반증 실패)

- `newpair_preflight_o1.py`는 **출하 클라이언트의 `plan()`을 import해** 프롬프트를 만든다(문구 재진술 아님). 체커 산식 `pr["prompt_tokens"] > (K-1)*(bg["prompt_tokens"]+bg["max_new"])`(`r2_correctness_check.py:278`)과 스크립트 계산(`(K-1)*(bg+BG_MAX_NEW)`, `BG_MAX_NEW=160`)이 일치.
- **양성대조가 진짜다**: 참조 쌍에서 `job_907456/gen_L1.json`의 **서버 보고** `prompt_tokens`와 8/8 probe + bg 전부 일치. 클라이언트가 `text`를 보내고 서버가 토크나이즈하므로(`r2_correctness_client.py:144,176`) 이 방법은 서버 보고값의 **예측**이다.
- 여유가 좁지만 **강건성 방향이 안전**: 서버가 special token을 1개 덜 붙이면 probe·bg가 **함께** 줄어 문턱은 1232, probe는 1279 ⇒ 여유 +47로 **늘어난다**.
- 미달 시 처분 `NO_VERDICT_UNREALIZED`는 **코드와 일치**한다(O1 실패 → `O_TIER_CONFIRMED=False` → `unrealized` → 라벨 4). "측정 실패 ≠ 규칙 실패"(게이트 #21) 정합. ✓

### 3 ★flashinfer adaptive split-KV와 O 티어 출력공간 — **코드 인용 정확. 두 분기 서술도 옳다. 그러나 §8 전수 표에 라벨 없는 칸 2개.**

**코드 확인(전부 참)**: `flashinfer_backend.py:185-200`은 split tile 고정을 `--enable-deterministic-inference`에만 건다(이 job은 안 씀); `:589-590`·`:711-712`의 cudagraph decode plan은 `fixed_split_size=None, disable_split_kv=False` ⇒ 적응 split-KV 스케줄러가 동작. §3.3의 "O 티어 동치의 신뢰가 커널 논증이 아니라 arm-내 귀무대조에 전적으로 의존한다"는 **정확하다.**

**라벨 없는 칸 (a) — TD-TD S 불일치**: §8 FAIL 행은 "(S: L-L 동일한데 **교차** 불일치)"라 적었는데, 코드는 `s_ref_ok and not s_all_ok` ⇒ **`mm["S"]` 전 쌍**을 본다. TD1-TD2 불일치(=arm-내)도 **FAIL**이다. 실측: R4(gen_TD2 1토큰 섭동) → `L1-L2=0, TD1-TD2=1` ⇒ **`FAIL`**. §8의 어느 행도 이 칸을 유도하지 않는다(INCONCLUSIVE 행은 "두 **L** boot이 S에서 불일치"만 적었다). 라벨 자체는 방어 가능하다(S는 배치 1이라 flashinfer 적응 split이 boot 간 결정적 ⇒ TD 자기 재현 실패는 true-dual 결함) — **그러나 등록이 그 논증을 적지 않았다.**

**라벨 없는 칸 (b) — 귀무대조 없는 S FAIL**: §1 표 행 5. `o_null_complete`는 `len(l_boots)>=2 and len(td_boots)>=2`를 요구하는데(`:484-485`) **S 티어에는 대응 가드가 없다**. L boot이 하나만 살아남으면 `s_ref_ok`가 **공허 참**이 되어, 참조 재현성을 **확인하지 않은 채** 교차 불일치 1토큰이 FAIL을 발화한다. 이는 §9의 규율("위 어느 것도 '게이트가 실패했다'로 라벨하지 않는다")과 **정면 충돌**한다.

### 4 ★계측 3건의 예보 정의역 — **I1은 공집합(死因 N3). I2는 예보 없음. I3는 정의역 건강하나 승격 통로 1개 잔존.**

- **I1/F2**: §2 死因.
- **I2**: 예보가 **하나도 없다**. 정의역 문제는 아니지만 두 서술이 부정확하다: (i) I2가 주는 것은 **클라이언트 관측 ITL**이지 λ0 §3-3 외삽의 절편인 **엔진 decode step(13.06 ms)**이 아니다(단위 불일치, 호스트/HTTP/디토크나이즈 포함). (ii) I2는 **D44 분할 위**에서 돈다 — decode 44 SM·prefill 64 SM의 값이지 모델의 일반 바닥이 아니다. "TTFT 바닥은 W3의 SLO 여유를 계산할 때 필요하다"는 문장은 이 한정 없이는 쓸 수 없다.
- **I3**: `--request-rate inf`의 의미 인용이 **정확**하다(`bench_serving.py:943-945` 확인; `--request-rate` **기본값도 inf**임을 부기 — `:2036-2040` — 이므로 probe C의 폐기 warm-up과 "같은 레시피"라는 §5-I2 주장도 성립한다). 출력공간에 거짓이 있다(`achieved`는 어떤 값도 될 수 있다). ✓
- ★**승격 통로 1개**: §0과 §5-I3이 I3를 "사다리의 **양 끝**을 추측 없이 고정한다"고 쓴다. **상한 프로브는 위쪽 끝만 고정한다.** λ*_inf ≥ λ*_Poisson이면 아래쪽 끝은 여전히 미고정이며, I3를 하한에 쓰면 사다리가 브래킷을 놓친다. NP-3은 "λ*가 아니다"까지만 막고 "양 끝" 문장은 막지 않는다.

### 5 ★감사자 처방의 교체(`Concurrency:` → 엔진 `#running-req`) — **교체가 옳다. 대체 지표의 관측 가능성도 실행으로 확인. 한계 승계도 됐다.** (반증 실패 · 긍정 사례)

- **교체 근거가 맞다**: `bench_serving.py:1120` `concurrency = np.sum(e2e_latencies)/dur_s` — **대기 포함**. `:1207` `semaphore = asyncio.Semaphore(max_concurrency)` — 미완료 요청 수를 64로 **고정**. ⇒ `--max-concurrency 64`에서 이 지표는 구성상 64 근처에 붙고 정보가 없다. **λ0 판정서 §3-8(a)의 내 처방이 틀렸고 사전등록이 옳다.** 감사자 자기 정정으로 등재한다.
- **대체 지표가 live 코드에서 관측 가능한가(게이트 #200)**: **확인함.** `job_907100/srv_warmup.log:79-83`에 `#running-req: 0`·`#running-req: 1`이 Prefill·Decode 양쪽 줄에 실제로 찍힌다. sbatch의 `grep -oE "#running-req: [0-9]+"`가 이 포맷과 일치. `--max-concurrency 64 > 48`이므로 클라이언트가 따라오면 엔진은 48에 붙고, 상류 병목이면 미달한다 ⇒ **출력공간에 거짓이 있다.** ✓
- **closed-loop ≠ open-loop 한계 승계**: §5-I3의 ★bullet + NP-3이 λ0 §3-8(b)를 문자 그대로 승계했다(`Concurrency 24.08` 중 대기 22.75 인용 포함). ✓
- **잔존 결함 2건**: (i) F5는 "**두 셀 모두**"를 요구하는데 산출물은 **전역 max 1개**다(§1 표 행 7) — 셀별 재계산 레시피가 등록되지 않았다. (ii) `#running-req` 미달의 원인은 클라이언트 병목만이 아니다 — (8192,64) shape는 prefill 지배라 admitted decode 인구가 엔진 사정으로 낮을 수 있다. **보수적 방향**(인용 금지 쪽)이라 死因은 아니나, "미달 = 클라이언트 병목"으로 읽히면 오진이다.

### 6 ★ctx 16384 / mem 0.82 미부팅 이력 — **처분 등록 양호(조용한 재시도 차단 명시). OOM 위험은 실제로 낮다.** (반증 실패)

- §9(b)가 "**스코프 변경 사유**(mem 0.80으로 재등록)이지 규칙 실패 아님 · ★기본값을 **조용히 바꿔 재제출하지 않는다**"를 명시 ⇒ 조용한 재시도 통로가 문자로 막혔다. ✓
- **위험 자체가 작다는 것을 수치로 확인**: 905835 실측 `available_gpu_mem=13.38 GB`. mem 0.80→0.82는 80 GB의 2% ≈ **1.6 GB** 소비 ⇒ 여유 ≈11.8 GB. ctx 8704→16384는 풀 크기를 **바꾸지 않는다**(`max_total_num_tokens`는 메모리 결정; `req_to_token` 증분 **3.1 MB**; `estimated`는 2048 하한에 걸려 무영향). mamba 상태는 48 × 134.6 MB = **6.46 GB**로 불변.
- 단, 새로 생기는 OOM 원인이 하나 있다: **계측 50분 뒤 warm-up 서버 잔류 메모리**(D12).

### 7 `--time` 재정의가 CLI에만 있음 — **제출 차단 결함은 아니다. 단 등록된 "최악"이 과소평가이고, 근거로 든 이유가 약하다.**

- 등록된 최악 71분 < 스크립트 기본 75분이므로 CLI를 빼먹어도 **형식상은 맞는다**(여유 4분). ⇒ 단일 인적 실패점이지만 차단 사유는 아니다.
- **그러나 최악 재계산은 ≈146분**(§1 표 행 6) ⇒ `--time=02:00:00`으로도 초과 가능. 처분 §9(f)가 있으니 규칙 결함은 아니다.
- 지시자를 안 고친 이유("기본 경로의 예산을 바꾸지 않기 위해서")는 **약하다**: 월타임 상한을 올리는 것은 어떤 측정량도 바꾸지 않는다(큐 대기만 변한다). 다만 이건 설계 취향이고 死因이 아니다.

### 8 스코프 튜플 승계 배선 — **문자 차단은 이 저장소에서 본 것 중 가장 촘촘하다. 기계적 배선은 약하다. 누락 1건(P2 착수).**

- §2 + NP-1…NP-7이 **성능·Claim D·다른 split·다른 정책·다른 모델·다른 백엔드·T>0·C층·λ***를 전부 문자로 막았고, X1 동결의 **역방향**(이 job이 Zamba2로 이식되지 않음)까지 NP-4가 막았다. 반증 실패.
- **기계적 배선은 부분적**: `verdict.txt`에는 모델·백엔드가 **인쇄되지 않는다**. `r2_correctness_report.json`의 `checks.server_args_across_boots`가 `attention_backend`·`context_length`를 담고, 모델 경로는 `provenance.txt`의 `model=` 줄에만 있다. 결과 문서가 `verdict.txt`만 인용하면 스코프가 사라진다(X1이 프로즈로만 동결됐던 선례).
- **누락**: §8 FAIL 행은 "P2 캠페인 착수 금지"를 적었지만, **PASS가 P2 착수를 승인하지 않는다**는 문장이 어디에도 없다. λ0 사전등록이 `NO-GO`이고 W4 파라미터화가 사용자 결정 대기인 현 상태에서, PASS는 P2의 **필요조건 하나**를 닫을 뿐이다.

### 9 (감사자 추가) warm-up boot의 분할 **실현**이 확인되지 않는다

I2/I3는 "D44에서 잰 값"으로 다음 회차 사다리에 들어간다. 그런데 warm-up boot은 `PDMUX_GREEN_READOUT` **없이** 뜬다 ⇒ 이 프로젝트가 두 번 물린 **target≠realized** 교락(Stage 0의 "D108 앵커가 실은 16 SM")에 대한 방어가 없다. 완화 사실: `srv_warmup.log`에 `sm_counts (prefill_sm, decode_sm): [(108,0),(92,16),(84,24),(74,34),(64,44),(0,108)]`가 찍히고(907100 확인) **인덱스 4 = (64,44)**이며, **같은 job의 scored L boot이 동일 server_cmd·동일 config로 `green_realized=True`를 낸다**. 그러나 이는 **추론**이지 측정이 아니다.

### 10 (감사자 추가) provenance에 **flashinfer 백엔드 파일이 없다**

§11의 24항목 manifest는 `models/nemotron_h.py`·`configs/nemotron_h.py`를 포함해 게이트 #195를 닫는다(파일 목록으로 확인; **스크립트는 실행하지 않았다** — 엔진 트리 쓰기 회피). 그러나 §3.3이 **이 쌍의 핵심 불확실성**으로 지목한 `srt/layers/attention/flashinfer_backend.py`는 **manifest 24줄에 없다**. 새 튜플에서 새로 load-bearing이 된 파일이 provenance 밖에 있다.

### 11 (감사자 추가) 재제출 정책 미등록

§9(a)는 `NO_VERDICT_INFRA`에 "재제출"을 허용하지만 **횟수·다른 라벨에서의 허용 여부가 없다**. 교차-잡 토큰 불일치가 측정돼 있으므로(X1: 8/96) 재추첨은 출력을 실제로 바꾼다. 라벨 반전은 못 만들었으므로(관측 2 job 전부 PASS) 死因은 아니다.

### 12 (감사자 추가) 클라이언트 타임아웃 — **반증 실패**

`timeout 900`이 9B 모델에 충분한지 의심했으나, 907456 실측으로 **boot 전체 span 70 s**(19:14:25→19:15:35), 그중 클라이언트 decode 활동 **42 s**다. 9B/ctx 16384로 2–3× 느려져도 **여유 5–10×**. §9의 "client ~2분"은 보수적. 위험 없음.

---

## 4. 제출 전 필수 조건 (D1–D13)

### 死因 해소 (필수 — 이것 없이는 `NO-GO` 유지)

- **D1 (N3)** §5-I1·§7-F2·NP-7을 다음으로 **교체**하라.
  - §5-I1의 "그 전제는 ctx 8704 / mem 0.80에서만 측정됐다"를 **삭제**하고: "`max_mamba_cache_size = max_running_requests = 48`은 **측정된 적이 없다** — `--disable-radix-cache`와 `--max-running-requests 48`이 함께 주어지면 `model_runner_kv_cache_mixin.py:223-230`이 이 값을 **대입**하고, `_resolve_max_num_reqs`(`:857-876`)의 `estimated ≥ 2048`·`_calculate_mamba_ratio`(`:390-392`)의 `return 1`이 배너 값을 48로 **고정**한다. job 905835의 48도 같은 항등식이다(`c_capacity.sbatch:126-127`). ctx·mem-fraction은 이 값에 들어가지 않으며, 메모리가 모자라면 값이 줄지 않고 **부팅이 죽는다**."
  - **F2를 도달 가능한 예보로 교체**하라. 권고: **F2′ = "ctx 16384 / mem 0.82에서 `max_total_num_tokens ≥ 2.0 × 10⁶`(= 905835의 2,618,868의 76% 이상)이고 `available_gpu_mem > 5 GB`"** — 이것은 실제로 거짓일 수 있고(mem 0.82의 순증가분과 ctx 증가분이 상쇄하지 않을 수 있다), KV 과공급 논증의 **진짜** 미측정 성분이다. 어긋나면: 다음 회차 사다리의 KV 여유 가정을 재유도.
  - **NP-7을 교체**: "**NP-7′ — I1의 `48`은 측정이 아니라 `--disable-radix-cache --max-running-requests 48`의 연역이다.** 어떤 결과 문서도 'ctx 16384/mem 0.82에서 구속 자원이 48임을 확인했다'를 쓸 수 없다. I1이 실제로 재는 것은 `max_total_num_tokens`·`available_gpu_mem`뿐이다."
  - §0의 "계측 3건" 서술에서 I1의 지분을 이에 맞게 축소하라.

### 등록 완결성 (필수)

- **D2 (§3-3b)** §8에 다음 칸을 **명시 등록**하라: "**booted L boot이 2개 미만인 채 `failures == ['S_TIER_MISMATCH']`만 나온 경우**(다른 failure 없음) — 이 job은 `NO_VERDICT_INFRA`로 보고한다. 이유: `s_ref_ok`가 `itertools.combinations(l_boots, 2)`의 **공허 참**이라 L-L 참조 재현성이 확인되지 않았다. 실측 재현: 907456 아티팩트에서 `gen_L2.json` 제거 + TD1 S 1토큰 섭동 → 체커 `FAIL`, 제거만 → `NO_VERDICT_INFRA`." (체커는 **고치지 않는다**; report JSON의 `boots`/`failures`로 사후 판정 가능한 **결정적 사전 규칙**이므로 새 자유 표면을 만들지 않는다. TD 쪽도 대칭으로 적을 것.)
- **D3 (§3-3a)** §8 FAIL 행에 **TD-TD S 불일치**를 명시 추가하고(코드가 FAIL을 낸다), 그것이 방어 가능한 이유를 한 줄로 적어라: "S 티어는 요청이 한 번에 하나라 flashinfer 적응 split-KV가 boot 간 결정적이므로, TD 자기 재현 실패는 측정 가능성이 아니라 true-dual 결함이다." 실측 라벨: R4 → `FAIL`(`L1-L2=0, TD1-TD2=1`).
- **D4 (§3-5)** **F5의 셀별 재계산 레시피를 실행 전에 문자로 등록**하라. 예: "`I_log_offsets.txt`의 `<cell>_start`(배타)~`<cell>_end`(포함) 행 구간을 `sed -n 's,ep' srv_warmup.log | grep -oE '#running-req: [0-9]+' | awk '{print $2}' | sort -n | tail -1`로 셀마다 계산한다. 전역 `I3_max_running_req.txt`만으로는 '두 셀 모두'를 판정할 수 없다." 그리고 F5의 "어긋나면"에 **"미달의 원인이 상류 병목이 아닐 수 있다"**(prefill 지배 shape에서 admitted decode 인구가 엔진 사정으로 낮을 수 있음)를 덧붙여라.
- **D5 (§3-11)** **재제출 정책을 등록**하라. 권고: "재제출은 `NO_VERDICT_INFRA`에 한해 **최대 1회**. `FAIL`·`INCONCLUSIVE`·`NO_VERDICT_UNREALIZED`는 재제출로 덮어쓰지 않는다(교차-잡 토큰 불일치가 측정돼 있다 — X1 8/96). 재제출 시 두 job의 라벨을 **둘 다** 보고한다."
- **D6 (§3-4)** I2 서술 2건 정정: (i) "I2가 주는 것은 **클라이언트 관측 ITL/TTFT**이며 λ0 §3-3 외삽의 절편인 엔진 decode step과 **다른 양**이다"; (ii) "**I1–I3의 모든 수치는 D44 분할(prefill 64 SM / decode 44 SM) 한정**이며 모델의 일반 바닥이 아니다."
- **D7 (§3-4)** §0·§5-I3의 "**사다리의 양 끝**을 고정한다"를 "**사다리의 위쪽 끝만** 고정한다"로 고쳐라. NP-3에 한 문장 추가: "**I3를 사다리의 하한으로 쓸 수 없다** — 상한 프로브는 아래쪽을 구속하지 않는다."
- **D8 (§3-9)** warm-up boot의 **분할 실현 미확인**을 등록하라. 최소 문안: "I2/I3는 **target** D44에서 측정됐고 **realization은 이 job에서 직접 확인되지 않았다**(warm-up은 green read-out 없이 뜬다). 근거는 (a) `srv_warmup.log`의 `sm_counts` 표 인덱스 4 = (64,44)와 (b) 동일 `server_cmd`·동일 config로 뜬 scored L boot의 `green_realized=True`에 의한 **추론**이다." (선택: `INSTR=1`일 때만 warm-up에 `PDMUX_GREEN_READOUT=1 PDMUX_GREEN_READOUT_PATH="$OUT/instrument/green_warmup.json"`을 붙이면 추론이 측정이 된다. 채점기는 `green_{lb}.json`만 열고 `warmup`은 라벨이 아니므로 안전하며, 기본 경로는 변수를 빈 문자열로 두면 바이트 동일이 유지된다. 테스트 1건 추가 필요.)
- **D9** **하네스·사전등록·테스트를 제출 전에 커밋**하라. 현재 `r2_correctness.sbatch`는 미커밋 수정본이고 `newpair_prereg/`·`test_r2_correctness_instrument.py`는 untracked다 ⇒ `provenance.txt`의 `commit=`가 **실제로 돈 하네스를 가리키지 못한다**(`src_dirty`는 `src/`만 본다). sbatch sha256은 기록되지만 파일이 추적되지 않으면 그 해시가 가리킬 대상이 없다. (작업 트리에 무관한 수정 9건이 있으므로 **선택 스테이징** 필요 — 실행 가능함을 확인했다.)
- **D10 (§3-10)** **`srt/layers/attention/flashinfer_backend.py`를 provenance에 넣어라.** 둘 중 하나: (a) `sync_engine_tree.sh`의 manifest에 25번째 줄로 추가(어느 테스트도 항목 수를 고정하지 않음을 확인했다), 또는 (b) sbatch provenance 블록에 `sha256sum` 한 줄 추가(그 블록은 어떤 테스트도 바이트 고정하지 않는다). (a)가 바람직하다 — §3.3이 이 파일을 결론의 신뢰 근거로 지목했다.
- **D11 (§3-7)** §9의 "최악 71분"을 **"계측 최악 50분 + scored 최악 93분(4×(health 480 s + client 900 s + teardown 20 s)) + 부대 3분 ≈ 146분"**으로 정정하고, `--time`을 **`02:30:00`**으로 올리거나(권장) 02:00:00 유지 시 §9(f)가 발화할 수 있음을 헤드라인에 병기하라. 제출 명령에서 `--time`을 빠뜨리면 기본 01:15:00이 적용된다는 경고 한 줄도 §13에 넣어라.
- **D12 (§3-6)** §9(b)에 **오귀속 차단** 한 줄: "첫 scored boot이 OOM으로 죽으면 그것이 ctx/mem 때문인지 **50분 부하 뒤 warm-up 서버의 잔류 메모리** 때문인지 먼저 구분한다(`R2C_INSTRUMENT=0`으로 1회 재제출). 구분 전에는 mem 0.80 재등록을 하지 않는다."
- **D13 (§3-8)** **스코프 배선 + 누락 문장**: (i) 결과 문서는 `verdict.txt` 단독 인용을 금지하고 `r2_correctness_report.json`의 `checks.server_args_across_boots`(backend·ctx·seed·cap)와 `provenance.txt`의 `model=` 줄을 **함께** 전사한다. (ii) 인용 금지에 **NP-8**을 추가하라(문안은 §5).

---

## 5. 실행 후 필수 병기 (결과 문서·정본이 문자 그대로 승계할 문안)

어떤 라벨이 나오든 아래를 **그대로** 붙인다. D1–D13 반영 후에도 유효하다.

> **NP-1′ (승계)** — PASS가 인증하는 것은 §2의 한 문장뿐이다. 성능·Claim D·다른 split·다른 정책·다른 모델·다른 백엔드·T>0 샘플링·O 프로토콜 밖 동시성은 포함되지 않는다.

> **NP-7′ (D1으로 교체)** — **I1의 `max_mamba_cache_size: 48` / `max_running_requests=48`은 측정이 아니라 `--disable-radix-cache --max-running-requests 48`의 연역이다**(`model_runner_kv_cache_mixin.py:223-230`, `:857-876`, `:390-392`). job 905835의 48도 같은 항등식이다. **"ctx 16384 / mem 0.82에서 구속 자원이 48임을 확인했다"는 문장은 쓸 수 없다.** λ0 판정서 §3-5의 "실측" 인용도 이에 따라 정정된다.

> **NP-8 (신규, D13)** — **PASS는 P2 캠페인 착수를 승인하지 않는다.** 이 게이트는 P2의 **필요조건 하나**를 닫을 뿐이다. 남은 블로커: λ0(0단계) 사전등록이 `NO-GO`(死因 N2+N3), W4를 단일 λ*로 파라미터화할 수 없다는 열린 항목(사용자 결정 대기), 그리고 방법론 게이트 #6(지표 절벽 대비 용량) 미충족(λ0 Q1).

> **NP-3′ (D7로 보강)** — I3는 λ*가 아니라 **open-loop Poisson λ*의 상한**이며, **사다리의 위쪽 끝만** 고정한다. 하한으로 쓸 수 없다. λ0 Q1이 그대로 유효하다: throughput 포화는 게이트 #6을 닫지 못한다.

> **NP-9 (신규, D6/D8)** — **I1–I3의 모든 수치는 D44 분할(prefill 64 SM / decode 44 SM) · legacy 루프 · warm-up boot 한정이며, 그 boot의 분할 실현은 이 job에서 직접 관측되지 않았다**(target 표 + 같은 job의 scored L boot green read-out에 의한 추론). I2가 주는 것은 클라이언트 관측 ITL/TTFT이지 엔진 decode step이 아니다.

> **NP-10 (신규, D2/D3)** — booted L boot이 2개 미만인 채 나온 `S_TIER_MISMATCH`는 **FAIL로 인용하지 않는다**(귀무대조 공허). TD-TD S 불일치로 인한 FAIL은 "true-dual 자기 재현 실패"로만 서술하며, 교차-arm 귀속과 구별한다.

> **NP-4·NP-5·NP-6 (승계 · 수정 불요)** — Zamba2/triton 동결의 양방향 차단, C층 수치 비교 금지, `INCONCLUSIVE`/`NO_VERDICT_*`가 "true-dual 실패"가 아니라는 것. 그대로 유지하라. 잘 쓰였다.

---

## 6. 반증 실패 항목 (공정 기록 — 깨뜨리려 했고 깨지지 않았다)

아래는 **死因이 아니며, D조건 충족 후 `GO-with-caveats`의 근거가 된다.**

1. **"추가 scored boot 0 · 채점 경로 무영향"** — 항등식이 아니고, **참이다.** 채점기 파일 접근면 전수 + 독립 변이 5종 + 27/27 테스트로 확인(§3-1).
2. **가드의 음성대조가 공허하지 않다** — 감사자 자작 변이 `M_invert_INSTR`·`M_drop_WARM_OK_leg`가 각각 해당 테스트를 실제로 깨뜨린다(교훈 53 충족).
3. **O1 사전계산과 양성대조** — 방법이 서버 보고값의 **예측**임이 8/8로 입증됐고, 특수 토큰 수 변화는 여유를 **늘리는** 방향이다(1239→1232 vs 1280→1279).
4. **`Concurrency:` → `#running-req` 교체** — 감사자(나)의 λ0 §3-8(a) 처방이 틀렸고 사전등록이 옳다. `bench_serving.py:1120`·`:1207`로 확인. 대체 지표의 **live 관측 가능성도 실제 로그로 확인**(게이트 #200 방식). **긍정 사례로 등재.**
5. **§10 상류 위험 3건 전부** — 체크포인트 `config.json`에 `n_groups: 8` 존재·`mamba_n_groups` 부재·`expand`/`mamba_expand` 부재 확인; `piecewise_cuda_graph_disabled_model_archs`에 NemotronH 없음 확인(5개 arch); `model_runner.py:2487-2490`의 비활성 경로와 907100 `srv_L1.log:71` 실증(`disable_piecewise_cuda_graph=False`인데 `piecewise_cuda_graph_tokens=[]`) 확인. **전부 정확하다.**
6. **flashinfer 코드 인용 3건**(`:185-200` deterministic 한정, `:589-590`·`:711-712` `fixed_split_size=None`) — **정확**. §3.3의 "동치 신뢰가 arm-내 귀무대조에 전적으로 의존한다"는 **옳은 자기 제한**이다.
7. **provenance manifest 24항목 + nemotron_h 모델 구현 포함** — 파일 목록으로 확인(스크립트 미실행). 게이트 #195 재발 없음.
8. **ctx/mem이 용량을 바꾸지 않는다** — λ0 판정서와 결론은 같으나 **더 강한 이유**로 참이다(§2 死因의 항등식).
9. **클라이언트 타임아웃 900 s** — 907456 실측 boot span 70 s 대비 여유 5–10×. 위험 없음.
10. **env 전파(`R2C_* … sbatch`)** — 907456 provenance에서 `R2C_NUM_KV_SPLITS=2` 전파 실증. `--comment`를 CLI에서 덮어쓰지 말라는 §13 경고도 정본과 일치.
11. **verdict rule v2 사후선택 아님** — 이 모델에서 `PDMUX_TRUE_DUAL_WORKER`가 한 번도 실행된 적 없음이 참이고(905835는 legacy+fixed만), 체커 sha가 두 테스트에 핀돼 있다.
12. **C층 강등과 907032 비교 금지** — 정확하고 필요한 제한.

---

## 7. 자기 적용 (게이트 #113 — 내 처방의 실현가능성 검사)

- **D1·D2·D3·D5·D6·D7·D11·D12·D13**: 전부 **문서 수정만**. GPU 0, 코드·테스트 무변경. 실행 가능.
- **D2의 위험 점검**: "체커 라벨을 산문으로 덮어쓰라"는 처방은 그 자체가 새 자유 표면이 될 수 있다. 그래서 **조건을 결정적으로 못 박았다** — `len([lb for lb in report["boots"] if lb.startswith("L") and lb in booted]) < 2` **AND** `failures == ["S_TIER_MISMATCH"]`. 두 값 모두 report JSON에 그대로 있고 재량이 개입할 자리가 없다. 사후 선택 불가.
- **D4**: 오프라인 `sed`+`grep` 재계산. 산출물(`I_log_offsets.txt`)이 이미 등록돼 있으므로 실행 가능. 단 **경계 ±1행 모호성**이 남으므로 "배타/포함"을 실행 전에 못 박도록 처방에 포함했다.
- **D8(선택 코드 경로)**: warm-up에 green read-out을 붙이는 변형은 **기본 경로 바이트 동일성을 깨지 않는지** 검사했다 — 변수 치환으로 `INSTR=0`에서 빈 문자열이 되게 하면 명령이 동일하고, 채점기는 `green_warmup.json`을 **열 수 없다**(라벨이 아님). 테스트 1건 추가 필요. **그래서 기본 처방은 문서 등록(무코드)으로 두고 코드 변경은 선택으로 제시했다.**
- **D9**: 작업 트리에 무관한 수정 9건 + untracked 9건이 있으므로 **선택 스테이징이 필수**다. 확인했고 실행 가능하다(OS 판정서 D2가 지적한 "미커밋 작업 파괴 위험"은 여기서는 커밋만 하므로 발생하지 않는다).
- **D10**: manifest 항목 수를 고정하는 테스트가 **없음을 grep으로 확인**했다 ⇒ 25번째 줄 추가가 기존 테스트를 깨지 않는다. 대안 (b)(sbatch provenance 블록에 한 줄)는 어떤 테스트도 그 블록을 바이트 고정하지 않음을 확인했다. **둘 다 실행 가능.**
- **내가 하지 않은 것**: `sync_engine_tree.sh`를 실행하지 않았다(엔진 트리를 **쓰는** 스크립트이고 이 감사는 read-only다). 따라서 §11의 "`sha256sum -c` 24/24 OK"는 **파일 목록으로만** 검증했고 해시 일치는 재현하지 않았다. 이 한계를 판정서에 남긴다.
- **내 직전 판정서(λ0)의 오류 1건을 자기 철회한다**: §3-8(a)의 "`Concurrency:` 보고값이 48 근처인지로 확인하라"는 처방은 **틀렸다**(`--max-concurrency`가 그 지표를 구성상 고정한다). 사전등록의 정정이 옳다. 이 철회를 정본에 등재하라 — 감사자 자기 철회 **4회차 누적**.

---

## 8. 신규 게이트 후보 (doc-steward 이관용)

1. **엔진이 "자동 산정"한다고 적힌 값이라도, 네가 준 플래그 조합이 그 산정을 건너뛰는 대입 분기로 보내지 않는지 확인하라** — `--disable-radix-cache` + `--max-running-requests N`이 `max_mamba_cache_size`를 **N으로 대입**한다. 이 값을 "측정된 용량 전제"로 두 개의 정본 문서가 인용하고 있었다(교훈 9 계열 21번째 재발).
2. **귀무대조를 요구하는 술어는 그 대조의 표본 수를 명시 요구해야 한다** — 같은 파일에서 O 티어는 `len(l_boots)>=2 and len(td_boots)>=2`를 적었고 S 티어는 안 적어, 부분 부팅 시 공허 참으로 FAIL이 발화한다(게이트 #183 계열, 다른 코드 경로에서 재발).
3. **"출력공간 전수" 표는 판정 코드를 실행해서 만들어라** — §8 표가 코드와 어긋난 칸이 2개(TD-TD S 불일치, 귀무대조 없는 S FAIL). 실제 아티팩트 1비트 섭동으로 30초 만에 드러났다.
4. **계측을 warm-up boot에 붙이는 설계는 그 boot의 통제 변수(분할 실현·telemetry)가 scored boot보다 약하다는 것을 함께 등록하라** — "채점에 안 닿는다"와 "그 수치를 신뢰할 수 있다"는 다른 명제다.
5. **(긍정)** 감사자 처방을 교체할 때는 대체 지표가 **live 로그에 실제로 존재하는 문자열**임을 보여라 — 사전등록이 `#running-req` 포맷으로 이를 했고, 감사자가 907100 로그로 독립 재확인했다(게이트 #200 충족 사례).

---

## 9. 요약

`NO-GO` — **死因 1건, N3**(F2/I1의 출력공간에 거짓 없음; `max_mamba_cache_size=max_running_requests=48`은 `--disable-radix-cache --max-running-requests 48`의 연역이며 905835의 48도 같은 항등식). **자유 표면 20개에서 반전 0건**, 등록 문안·코드 인용·양성대조·음성대조·인용 금지 문안은 이 저장소 기준으로 상위권이다. **D1(문서 3곳 교체, GPU 0)로 死因이 해소**되며, D2–D13은 등록 완결성 조건이다. 재감사에서 `GO-with-caveats` 예상.

**GPU 장부 불변**: 이 트랙 **0.43 GPU-h**. 이 감사의 신규 지출 **0**. 실행 시 예상 0.47–0.58 GPU-h(등록 최악 1.18; `--time` 하드캡 기준 최대 2.0).

---

**관련 파일(전부 절대경로)**

- 감사 대상: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/newpair_prereg/PREREG_NEWPAIR_2026-09-13.md` · `.../newpair_prereg/newpair_preflight_o1.py` · `.../newpair_prereg/newpair_preflight_o1_stdout.txt`
- 실행체·판정 코드: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/r2_correctness.sbatch`(미커밋 수정본) · `.../r2_correctness_check.py`(sha `ec355e17…`, 무수정) · `.../r2_correctness_client.py`
- 테스트: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/tests/test_r2_correctness_instrument.py`(27/27 OK) · `.../tests/test_r2_correctness_ctx.py`
- 반전 계산에 쓴 원자료: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/job_907456/`(gen/green 복사, srv/tel symlink, 무변경) · `.../job_907100/srv_warmup.log`·`srv_L1.log` · `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/longctx_conflict/probes/c_905835/srv_d44_r2_o96.log` · `.../probes/c_capacity.sbatch`
- 엔진: `/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`(`:174-175`, `:196-255`, `:390-402`, `:857-876`) · `.../srt/mem_cache/memory_pool.py`(`:300-325`) · `.../srt/server_args.py`(`:648`, `:1254-1259`, `:1397-1415`, `:1959`) · `.../srt/model_executor/model_runner.py`(`:2486-2491`) · `.../srt/configs/model_config.py`(`:1352-1358`) · `.../srt/layers/attention/flashinfer_backend.py`(`:185-200`, `:589-590`, `:711-712`) · `.../sglang/bench_serving.py`(`:943-945`, `:1120`, `:1207`, `:1705-1706`, `:2036-2040`) · `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/src/multiplex/multiplexing_mixin.py`(`:87-89`)
- 체크포인트: `/scratch/ehmoon/whlee/prefill-layer-alloc/hf_cache/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base/snapshots/dc0661c829b14e5b9246c05cfa89094a0875e052/config.json`
- 승계 맥락: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_eval/lambda0_prereg/VERDICT_lambda0_rules_2026-09-13.md`(§3-8·Q1–Q5) · `/scratch/ehmoon/whlee/prefill-layer-alloc/PROJECT_STATUS.md`(2026-09-13(3) 배너) · `/scratch/ehmoon/whlee/prefill-layer-alloc/reports/CONSENSUS.md`
