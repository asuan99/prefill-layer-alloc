# 규칙층 재감사 판정서 — 새 (모델, 백엔드) 쌍 R2 correctness 게이트 **rev2**

**감사 대상**: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/newpair_prereg/PREREG_NEWPAIR_2026-09-13.md` (rev2, 715행)
동반: `.../newpair_prereg/newpair_preflight_o1.py`·`newpair_preflight_o1_stdout.txt` · `.../results/r2_correctness/r2_correctness.sbatch`(수정, 미커밋) · `.../scripts/bootstrap/sync_engine_tree.sh`(수정, 미커밋) · `.../tests/test_r2_correctness_instrument.py`(untracked, 35케이스)

**감사자**: claims-auditor (read-only) · **2026-09-13** · **GPU 지출 0**(기존 job 907456/907100/905835 아티팩트 + CPU 실행만) · **1차 판정서 `NO-GO`(死因 N3)에 대한 재제출본** · 사전등록은 여전히 미제출

**1차 판정서 §6의 반증 실패 12건은 지시대로 재시험하지 않았다.** 아래는 **死因 해소 여부 + rev2가 새로 만든 표면**만이다.

---

## 0. 등급

# `GO-with-caveats`

**死因 0 · 등록 caveat 10건.** **제출해도 되는 상태다** — 단 사전등록이 스스로 §12-b에 건 **D9(제출 전 커밋)**가 남아 있고(감사 조건이 아니라 이 문서 자신의 provenance 요구), 아래 §5의 인용 금지 문안 NPC-A…NPC-J를 결과 문서·정본이 문자 그대로 승계해야 한다.

**死因 N3는 해소됐다.** rev1의 F2는 엔진이 대입하는 항등식(`--disable-radix-cache` + `--max-running-requests 48` ⇒ `max_mamba_cache_size = max_running_requests = 48`)이라 출력공간이 **공집합**이었다. rev2의 F2′는 `torch.cuda.mem_get_info` 기반 런타임 판독값에 대한 예보이므로 출력공간에 거짓이 **존재한다**. 이 차이는 정도가 아니라 **범주**의 차이이며, N3는 사전확률이 아니라 출력공간의 공허성을 묻는다. 다만 F2′의 정보량은 사실상 0이다(§2, 수치 포함) ⇒ **caveat NPC-B**.

**그러나 rev2는 새 死因은 안 만들었지만 새 결함 2건을 만들었다**(둘 다 N1–N4 미충족이라 차단하지 않는다):
- **(가장 중요) 1차 판정서 D2가 처방한 "TD 쪽도 대칭" 조항이 틀렸다.** 그 조항은 **귀무대조가 완전한 진짜 교차-arm FAIL을 `NO_VERDICT_INFRA`로 강등**한다. 무수정 체커 + job 907456 실제 아티팩트로 2건 재현(§1 반전 시험 4a·4b). **이것은 감사자(전임) 자신의 처방이 만든 결함이며, 게이트 #110에 따라 등록 상수로 승계하지 않고 독립 재검증해 철회한다.**
- **§6의 "기본 경로에서 프로세스가 한 개도 추가로 뜨지 않는다"는 거짓이다.** provenance 블록이 `import flashinfer` 파이썬 프로세스를 **INSTR·백엔드와 무관하게 무조건** 1개 더 띄운다(타임아웃 없음). 측정 전에 끝나므로 측정량은 안 바뀌지만 "907100/907456과 명령 단위로 동일"은 못 쓴다.

**항등식을 사실로 승격하는 통로는 정확히 1개 남았다** — 사전등록 본문·NP-7′·F2′는 전부 정확하나 **하네스 주석**이 아직 죽은 문장을 그대로 들고 있다(§2-D, 수정은 텍스트 4줄·테스트 무영향).

---

## 1. 반전 시험 표 (rev2가 새로 만든 표면만; rev1의 20표면은 1차 판정서 §1)

반전 계산에 쓴 원자료·도구를 마지막 열에 명시한다. **감사자가 만든 변형은 `gen_*.json` 1비트 증분과 파일 1개 삭제뿐이고, 체커는 무수정(sha `ec355e17…` 재확인)**이다.

| # | 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 / 도구 |
|---|---|---|---|---|
| 1 | **F2′ 문턱** (`max_total_num_tokens ≥ 2.0×10⁶` ∧ `available_gpu_mem > 5 GB`) | 905835 실측에서 구성별 증분을 전부 불리한 쪽으로 | **없음(반전 미생성)** | 투영치 **2.72×10⁶ tok / ≈11.6 GB**. 거짓이 되려면 각각 **−11.0 GB / −6.6 GB**가 필요. 식별된 신규 소비자 총합 **≈0.21 GB**(§2). 원자료 `.../probes/c_905835/srv_d44_r2_o96.log:42,53,62,64` |
| 2 | **`R2C_INSTRUMENT`** (warm-up green read-out) | 0 ↔ 1, 그리고 따옴표 변이 | **없음** | sbatch에서 마커로 직접 추출해 실행(테스트 파일 미경유): INSTR=0 argv = `[PDMUX_R2_POLICY=fixed][PDMUX_R2_FIXED_DSM=44][python][-m][sglang.launch_server][--port][31234]` = **HEAD `r2_correctness.sbatch:265-266`과 동일**; INSTR=1은 **정확히 2원소** 추가. 음성대조(`"$WARM_GREEN"`)는 `[]` **빈 argv 원소를 주입**해 바이트 동일성을 깬다 ⇒ 공허하지 않음 |
| 3 | **manifest 25번째 줄** | 24 ↔ 25, 그리고 하류 소비자 전수 | **이 게이트의 판정 변화 없음** | HEAD 24줄 vs 작업트리 25줄, `head -24` **완전 동일**(`git show HEAD:`와 작업트리를 각각 파싱해 대조) ⇒ 접두 성질 성립. `flashinfer_backend.py` sha `9181648bb265f130…` **독립 재계산 일치**. 25파일 전부 존재 |
| 4a | **§8 (D2) 오버라이드 — L측** | booted L = 1 (도달 가능: 907032서 boot 2개 사망 전례) | ★**FAIL → NO_VERDICT_INFRA 반전** | 907456 아티팩트, `gen_L2.json` 삭제 + `gen_TD2.json` `phase_s[0].output_ids[0]+=1` → 무수정 체커: `S mismatches {L1-TD1:0, L1-TD2:1, TD1-TD2:1}`, `failures=['S_TIER_MISMATCH'] infra=['L2:B1']`, **`VERDICT FAIL`**. rev2 D2가 이를 `NO_VERDICT_INFRA`로 강등 ⇒ **§8·D3·NP-10이 "방어 가능한 FAIL"로 등록한 TD-TD 자기재현 실패가 소멸** |
| 4b | **§8 (D2) 오버라이드 — TD측(1차 판정서가 처방)** | booted TD = 1 | ★**FAIL → NO_VERDICT_INFRA 반전** | 같은 아티팩트, `gen_TD2.json` 삭제 + `gen_TD1.json` 1토큰 섭동 → `S mismatches {L1-TD1:1, L1-L2:0, TD1-L2:1}`, `failures=['S_TIER_MISMATCH'] infra=['TD2:B1']`, **`VERDICT FAIL`**. **L-L 귀무대조는 완전**(L1-L2 = 0/16). rev2 D2 대칭 조항이 이 진짜 교차-arm FAIL을 측정 실패로 강등 |
| 5 | **D5 재제출 1회 × 위 강등** | 0회 ↔ 1회 | **복합 반전 가능**(FAIL→INFRA→재추첨→PASS) | 교차-잡 토큰 불일치는 **측정됨**(X1 감사 907100↔907456 8/96 단위) ⇒ 재추첨은 출력을 실제로 바꾼다. 관측 2 job이 전부 PASS라 **직접 라벨 반전 사례는 못 만들었다** |
| 6 | **`#SBATCH --time`** 01:15:00 ↔ 02:30:00 ↔ 구조적 최악 | 등록이 허용하는 양 끝 | **측정량 변화 0**, 판정 **소멸** 가능 | 161분 재계산 **검산 통과**(900+3000+5600+180 = 9680 s = 161.3 min; `wait_health` 3번째 인자 = 5 s 폴 횟수이므로 warm-up 180폴=**900 s**, scored 96폴=**480 s** 맞다). ★**단 구속 조건은 161>150이 아니다**: 판독 3건은 `WARM_OK=1`에 게이트돼 있어 900 s 헬스와 **거의 배타**다. 9000−3000−5600−180 ⇒ **warm-up 헬스 ≤ 220 s면 캡이 덮는다**. 실측 cold warm boot = **63 s**(905835 11:31:39→11:32:42) |
| 7 | **F5 셀별 재계산 레시피(D4)** | 경계 ±1행, 두 셀 | **없음(모호성 소멸)** | `i_mark`는 마킹 시점의 `wc -l`(완결 행 수) ⇒ `start` 배타(=`S+1`부터)·`end` 포함이 **정확히 그 구간**. 마크 6개 키가 테스트·awk 레시피와 문자 일치 |
| 8 | **인접 사전등록 λ0의 `MULT[A][0]`** | 0.40·λ_inf ↔ 0.25·λ_inf | **이 문서 쪽 변경 불요** | λ0 워크스트림이 **이미 이 문서 D7을 수용해 자기 쪽을 고쳤다**: `lambda0_plan.py:66-75`·`:101-106` "rev3: x_min 0.40 -> 0.25 (newpair (D7): lambda_inf must not be trusted to bound lambda* from below)". 파일을 직접 열어 확인(게이트 #110) |

**반전 계산 자신의 자유 표면 점검(P6 전례 회피)**: 행 2는 sbatch의 `# --- BEGIN/END r2c warmup launch ---` 마커로 **독립 추출**해 `env`를 argv 덤프 함수로 치환해 돌린 것이고(프로젝트 테스트 파일 경유 안 함), 행 4a/4b는 **job 907456 실제 아티팩트**(gen/green 복사, srv/tel symlink)에 1비트 섭동만 준 것이며 기준선에서 **PASS를 비트 단위 재현**했다(S 6쌍 전부 0, C 진단 5/3/5/4/4/3). 행 1·6의 산술은 905835 서버 로그 4행과 sbatch 상수만 쓴다.

---

## 2. 死因 해소 심사

### 2-A. F2′의 출력공간에 거짓이 실제로 있는가 — **있다. N3 미발화.**

`max_total_num_tokens`·`available_gpu_mem`이 어디서 오는지 코드로 확인했다: KV 풀은 `mem_fraction_static` 예산에서 잡히고(`model_runner_kv_cache_mixin.py`), `available_gpu_mem`은 **cudagraph 캡처가 끝난 뒤의 `get_available_gpu_memory`**다(905835: `Capture cuda graph begin … avail mem=14.44 GB` → `end … mem usage=1.06 GB. avail mem=13.38 GB` → 배너 `available_gpu_mem=13.38 GB`). **엔진이 대입하는 값이 아니다.** ⇒ rev1 F2와 범주가 다르다.

### 2-B. 905835 대비 실제로 미측정인 성분은 무엇인가 (수치)

| 성분 | 905835(참조) | 새 튜플 | 미측정인가 | 크기 |
|---|---|---|---|---|
| attention backend | **flashinfer** (`srv_d44_r2_o96.log` server_args `attention_backend='flashinfer'`) | flashinfer | ❌ **아니다** | — |
| flashinfer global workspace | 384 MB (`environ.py:349` `EnvInt(384*1024*1024)`; Qwen 계열 512 MB·deterministic 2048 MB 분기 **둘 다 미해당**) | 384 MB, **ctx 무관·전역 공유** | ❌ **아니다(이미 지불됨)** | 0 |
| mem-fraction | 0.80 | 0.82 | 부호·크기 기지 | **+1.6 GB** static ⇒ 토큰 **+**, 여유 **−** |
| ctx | 8704 | 16384 | 실질 미측정이나 미소 | `req_to_token` 48×16384×4 B = 3.1 MB(**+1.5 MB**) + flashinfer cudagraph `kv_indices`/`custom_mask` `max_num_tokens×max_context_len×(4+1) B` ≈ **+1.8 MB** |
| pdmux config | homog5 (**5** stream group) | pdmux_r2 (**6**) | 미측정 | 캡처는 **1 pass**(`Capture cuda graph begin` 로그 907100·905835 각 **1회**), 총 1.06 GB(9B) ⇒ 6번째 그룹 지분 **≈+0.21 GB** |

**투영**: `max_total_num_tokens ≈ 2.72×10⁶`(905835의 39.96 GB/2,618,868 = 16,385 B/tok 기준), `available_gpu_mem ≈ 11.6 GB`.
**F2′를 거짓으로 만들려면**: 토큰 쪽 **−11.0 GB**, 여유 쪽 **−6.6 GB**. 식별된 신규 소비자 총합 **≈0.21 GB**(캡처 1.06 GB가 7배로 불어나도 미달).

⇒ **F2′는 형식상 반증 가능하지만 도달 기전이 하나도 식별되지 않는다.** 판정:

> **허용된다.** 근거 둘. ①N3의 문언은 출력공간의 공허성이지 사전확률이 아니고, rev1 F2(엔진 대입)와 F2′(런타임 판독)는 범주가 다르다. ②1차 死因의 두 번째 논거였던 "구매의 1/3이 무효"는 **비용 논거로 더 이상 성립하지 않는다** — I1의 한계 GPU 비용은 **0초**다(어차피 뜨는 boot의 로그에 대한 `grep` + 시작 시 1회 env 2개). 정보량 0인 예보를 0원에 사는 것은 死因이 아니다.
> **단 caveat NPC-B**: F2′ 충족을 "KV 여유를 측정으로 확인했다"로 인용할 수 없다.

또한 §5-I1/§7의 **"이 튜플에서 처음 바뀌는 성분(flashinfer workspace 버퍼, …)"는 사실이 아니다** — 905835가 이미 flashinfer였고(§7 자신이 그렇게 적고 있어 문서 내부 모순), 워크스페이스는 384 MB 고정·ctx 무관이다. ⇒ **NPC-B에 포함해 정정**.

### 2-C. §5-I1·NP-7′가 "48은 연역이다"를 정확히 말하는가 — **말한다.**

인용 3건을 **엔진 소스를 직접 열어** 독립 재검증했다(게이트 #110):
- `model_runner_kv_cache_mixin.py:223-230` — `elif server_args.disable_radix_cache and server_args.max_running_requests is not None:` → `max_mamba_cache_size = max_running_requests // 1` ✓
- `:390-392` — `_calculate_mamba_ratio`: `if self.server_args.disable_radix_cache: return 1` ✓
- `:857-876` — `estimated = max(min(token_capacity/ctx*512, 4096), 2048) ≥ 2048`; **ctx 16384에서도** `2.7e6/16384*512 = 84,375 → 4096 ≫ 48` ⇒ `min(48, 4096) = 48`, 이어 `min(48, 48//1) = 48` ✓
- `c_capacity.sbatch` — 플래그는 **:127**에 있다(인용 "126-127"은 범위로는 참, 다소 헐거움). 905835의 48도 같은 항등식 ✓

### 2-D. 승격 통로는 몇 개 남았는가 — **1개.** (하네스 주석)

`r2_correctness.sbatch:353-356`(작업트리):

```
  # I1 -- RECORDED, never judged.  Is the constraining resource still
  # max_mamba_cache_size = max_running_requests = 48 at this ctx and
  # mem-fraction?  That is the single unmeasured premise of the audit's sec 3-5
  # capacity argument, and it is readable straight off the boot banner.
```

**D1이 죽인 바로 그 문장이 실행체 안에 살아 있다.** §0-b의 D1 행은 반영 위치로 §0·§5-I1·§7·NP-7′만 적고 sbatch를 빼놓았다. 이 파일은 커밋되고 sha256이 `provenance.txt`에 기록되며 job 디렉터리와 함께 보존된다 ⇒ 교훈 80("출처 허위는 규칙 정본 파일 안이 가장 위험") 계열.
**N1–N4 어디에도 해당하지 않으므로 차단하지 않는다.** 수정 권고(비차단): 위 4줄을 NP-7′ 문안으로 교체. **실현가능성 확인(게이트 #113)**: 35 테스트 중 이 블록을 보는 것들은 전부 `assertIn` 부분문자열이며 **위 4줄과 겹치는 것이 하나도 없다** ⇒ 주석 교체는 어떤 테스트도 깨지 않는다. 고치지 않고 제출해도 좋으나 **NPC-C를 반드시 승계**해야 한다.

---

## 3. rev2가 만든 새 표면

### 3-(a) warm-up green read-out — **안전. 직접 실행으로 확인.**

- INSTR=0 argv **7원소, 빈 문자열 0개**, HEAD(`:265-266`)와 동일 ⇒ 907100/907456의 warm-up 명령과 바이트 동일 ✓ (§1 행 2, 독립 하네스)
- INSTR=1은 `PDMUX_GREEN_READOUT=1`·`PDMUX_GREEN_READOUT_PATH=…/instrument/green_warmup.json` **정확히 2개**만 추가 ✓
- 음성대조(따옴표)는 `[]` 빈 argv 원소를 주입 ⇒ 실제로 깨진다 ✓
- **채점 경로 비접촉 재확인(테스트 미경유, 체커 소스 직접 전수)**: `open()` 6곳 = `boots.txt`·`gen_/srv_/tel_/green_{lb}`·report(w). `glob/listdir/scandir/os.walk/iterdir` **0건**. `labels = open(…/boots.txt).read().split()` ⇒ `green_warmup.json`은 어떤 라벨로도 열리지 않는다 ✓
- **read-out이 실제로 발화하는가**(D8이 "추론→측정"이 되려면 필요): `multiplexing_mixin.py:224-245` `_maybe_emit_green_readout`은 **telemetry와 무관**하게 `PDMUX_GREEN_READOUT_PATH`에 쓴다 ⇒ warm-up(텔레메트리 없음)에서도 동작 ✓. 파싱 필드가 **job_907100/green_L1.json 실물 스키마와 전부 일치**하고 `stream_index 4` = 실현 `(64,44)` ✓

### 3-(b) manifest 25번째 항목 — **접두 보존 참. 그러나 파급 조사가 불완전.**

- 접두 보존 **독립 확인**(§1 행 3) ✓ · `lambda0.sbatch:97` `[ "$N_MANIFEST" -lt 24 ]` ⇒ 25줄 안전 ✓ · 25파일 존재·flashinfer sha 일치 ✓
- ★**"manifest 항목 수를 고정하는 테스트는 저장소에 없다"는 *테스트*에 한해 참이고, 비-테스트 소비자 2곳은 항목 수/내용을 고정한다**:
  - `results/ltsm_probe/ltsm_p1_probe.sbatch:95-104` — 참조 매니페스트와 **정확 diff**, 불일치 시 `exit 4`. 참조(`p1_gates/gate1/runtime_source_manifest_gate1c_877974.sha256`)는 **15줄**.
  - `results/smid_census/smid_l0_census.py:219` — `MANIFEST_EXPECTED_FILES = 15`.
  - **둘 다 이번 25번째 줄 때문에 새로 깨지는 것이 아니다**(15→17→24 단계에서 이미 깨져 있었다) ⇒ 死因 아님. 그러나 §11의 "파급 확인" 문장은 이 범위를 적어야 한다 ⇒ **NPC-F**.
- ★**휠 공백의 크기**: manifest는 SGLang 래퍼만 해시한다. sbatch가 기록하는 것은 **버전 문자열뿐**이다 — venv에서 그 명령을 그대로 돌려 `flashinfer_version 0.6.10 /home01/ehmoon/.local/lib/python3.14/site-packages/flashinfer/__init__.py` 확인. 구멍: ①문자열 ≠ 내용 해시 ②설치 위치가 **repo 밖·venv 밖**(`~/.local`)이라 임의의 `pip install --user`가 조용히 바꿀 수 있고 manifest는 못 본다 ③JIT 캐시 미기록. **반면 autotune은 위험이 아니다** — `model_runner.py:2065-2085`의 `_should_run_flashinfer_autotune`은 `moe_runner_backend ∈ {flashinfer_trtllm, flashinfer_mxfp4}` **그리고** compute capability major ≥ 9를 요구하는데 A100은 8.0이고 moe 백엔드는 `auto`다 ⇒ **두 조건 모두 미충족, autotune 꺼짐**. 이 한정으로 **NPC-E**.

### 3-(c) `--time` 01:15:00 → 02:30:00 — **측정량 불변. 161분 재계산 검산 통과. §9(f)로 충분.**

- 기본 경로(Zamba2/triton 재현)에서 **월타임이 입력인 측정량은 없다** ✓
- 161분 산술 ✓(§1 행 6). `wait_health`의 3번째 인자가 **5 s 폴 횟수**임을 확인했으므로 1차 판정서의 146분이 warm-up 900 s를 빠뜨린 것도 맞다.
- ★**"02:30:00이 구조적 최악을 못 덮는다"는 제출 차단이 아니다.** 판독 3건은 `[ "$INSTR" = "1" ] && [ "$WARM_OK" = "1" ]`에 게이트돼 있어 900 s 헬스 소진과 **배타에 가깝다**(헬스가 실제로 타임아웃하면 판독은 통째로 스킵 ⇒ 최악 6680 s = **111분 < 150분**). 캡이 뚫리는 유일한 경로는 "warm-up이 **220 s 이상** 걸려서 마지막 폴에 겨우 살아나고, **동시에** 판독 3건과 scored 4 boot의 health·client 타임아웃이 전부 발화"다. 실측 cold warm boot 63 s 대비 3.5배. **§9(f) 처분으로 충분하다.**
- ★단 **§9(f)의 재제출은 D5의 열거(`NO_VERDICT_INFRA` 한정 1회) 정의역 밖이다**(벽시계 강제 종료는 라벨을 낳지 않는다) ⇒ **NPC-J**.
- ★**기본 경로가 "명령 단위로 동일"하지 않다**: provenance 블록이 `python -c "import flashinfer; …"`를 **무조건** 실행한다(`r2_correctness.sbatch:254-255`, INSTR·백엔드 무관, `timeout` 없음). 35 테스트 중 이를 잡는 것은 없다(`test_off_starts_no_process_and_writes_nothing`은 **추출된 계측 블록만** 돌린다 — 교훈 9 계열) ⇒ **NPC-D**.

---

## 4. D2–D13 배선 표본 확인 + 인용 금지 문안 대조 + 출력공간 재전수

### 4-1. D2/D3 — **판정 경로에 실제로 걸린다. 그러나 D2가 과잉이다(핵심 지적).**

D2의 술어는 report JSON으로 **사후 재량 없이** 계산 가능하다. 필드 존재를 실물로 확인: `boots`, `per_boot[lb]["checks"]["B1_BOOTED"]`, `failures`, `checks["S_pairs"][pair]["mismatches"]` ✓.

**문제는 술어가 너무 넓다는 것이다.** `s_ref_ok = all(mm["S"][…] == 0 for a,b in combinations(l_boots, 2))`는 **L-L 쌍에만** 양화된다 ⇒ **TD boot 수는 s_ref_ok의 공허성에 영향을 주지 않는다.** 그러므로 D2의 "TD 쪽도 대칭" 조항은 지키는 것이 없고, 대신 §1 행 4b의 반전을 만든다. 또 L측 조항도 `failures == ["S_TIER_MISMATCH"]`만 보기 때문에 **불일치 쌍이 TD-TD인 경우까지 삼켜** §1 행 4a의 반전을 만든다.

**권고 대체 문안 D2′(결정적·report JSON만 사용·GPU 0):**

> `failures == ["S_TIER_MISMATCH"]` **이고** booted L boot < 2 **이고** `checks["S_pairs"]` 안에 **양쪽이 모두 TD인 불일치 쌍이 없을 때에만** 이 job은 `NO_VERDICT_INFRA`로 보고한다. **TD 대칭 조항은 철회한다**(`s_ref_ok`는 L-L 쌍에만 양화되므로 TD boot 수로는 공허해지지 않는다).

검증: 행 4b → 조건 2 불충족 ⇒ FAIL 유지 ✓. 행 4a → 조건 3 불충족(TD1-TD2 = 1) ⇒ FAIL 유지 ✓. 원래 보호 대상(L booted 1, 불일치가 교차 쌍뿐) → 발화 ⇒ INFRA ✓.
**死因이 아니므로 차단 조건이 아니다.** 고치지 않고 제출해도 좋으나 **NPC-A를 반드시 승계**해야 한다.

**D3 자체는 코드와 일치한다** — `TD1-TD2 = 1` → `VERDICT FAIL` 재현 ✓.

### 4-2. D4 — **경계 규칙 모호하지 않다.** (§1 행 7) `wc -l`은 완결 행 수이므로 `start` 배타·`end` 포함이 정확히 그 구간을 준다. 마크 키 6개가 sbatch·테스트·awk 레시피에서 문자 일치 ✓.

### 4-3. 나머지 D조건

| D | 판정 경로 도달 | 비고 |
|---|---|---|
| D5 | 등록 ✓ | 정의역 공백(§9(f)) ⇒ NPC-J |
| D6 | §5-I2 + NP-9 ✓ | 단위(클라이언트 ITL ≠ 엔진 step)·D44 한정 둘 다 명시 |
| D7 | §0·§5-I3·NP-3′ ✓ | §7 — **이 문서가 옳다** |
| D8 | 코드 채택 + 발화 확인 ✓ | §3-(a) |
| D9 | **미반영(의도)** | 제출 전 필수 — §6. §12-b가 "아래 **5개**"라 적고 **4경로(=7파일)**를 나열 ⇒ 계수 문구 정정 권고 |
| D10 | ✓ | §3-(b), 단 NPC-E·NPC-F |
| D11 | ✓ | §3-(c), 단 NPC-D |
| D12 | §9(b) ✓ | OOM 오귀속 차단 문안 존재 |
| D13 | §12 (D13-i) + NP-8 ✓ | **NPC-A 발화 시 `S_pairs` 전사 의무를 추가해야 한다** |

### 4-4. 인용 금지 문안 대조 (1차 §5 ↔ rev2 §12)

| 문안 | 의미 동일? | 비고 |
|---|---|---|
| NP-1′ | ✓ 동일(확장) | — |
| NP-7′ | ✓ 동일 + 강화 | "905835의 λ* 앵커는 여전히 provisional" 추가 |
| NP-8 | ✓ 문자 동일 | — |
| NP-3′ | ✓ 동일 + probe C 수치 | — |
| NP-9 | **조건부로 약화, 그러나 정당** | 1차 §5는 "직접 관측되지 않았다"로 단정했으나 그것은 D8의 **코드 옵션을 안 쓴다**는 전제였다. 코드를 채택했으므로 조건부(파일 있으면 측정 / `WARMUP_GREEN_READOUT_MISSING`이면 추론)가 옳다. **단 결과 문서는 어느 분지가 발화했는지 `I1_green_warmup.txt`로 명시해야 한다** |
| NP-10 | ✓ 동일 + **"TD 쪽도 대칭"이 결함** | §4-1. NPC-A로 정정 |

### 4-5. 출력공간 재전수 (판정 코드 실행)

체커 결정 경로를 소스에서 전수하고(`:430-500`) 3개 시나리오를 실제로 돌렸다. **rev2의 §8 표는 이제 체커와 일치하며, 라벨 없는 칸은 남아 있지 않다.**
- FAIL: 퍼-boot `B2–B5`, `B5_SAME_CONFIG:*`, `B6_NOT_DEGENERATE`, `INPUT_IDENTITY`, `S_TIER_MISMATCH`(**전 쌍** — TD-TD 포함, §8이 이제 명시), `O_TIER_CROSS_ARM_MISMATCH_WITH_IDENTICAL_NULL`, 크래시 있는 미부팅 boot의 `{lb}:B1` — 전부 §8/§9에 대응 있음.
- `NO_VERDICT_INFRA` = `infra or not l_boots or not td_boots` ✓ / `INCONCLUSIVE` = `not s_ref_ok or any(o_within)` ✓ / `NO_VERDICT_UNREALIZED` = `unrealized` ✓ / PASS = 나머지 ✓.
- 잠재 사각지대: boot 3개일 때 `o_null_complete`가 False가 되어 교차-arm O 불일치가 FAIL을 못 내는 분기는 **`infra`가 비어 있을 수 없으므로** 항상 `NO_VERDICT_INFRA`로 흡수된다 ✓. `boots.txt`는 health 검사 **전에** append되므로 라벨 4개가 항상 기록된다 ✓.
- **남은 문제는 "라벨 없는 칸"이 아니라 "§8의 D2가 라벨을 덮어쓰는 방향"이다**(§4-1).

---

## 5. 실행 후 필수 병기 — 결과 문서·정본이 **문자 그대로** 승계할 문안

1차 판정서 §5의 NP-1′·NP-3′·NP-7′·NP-8·NP-9·NP-10과 사전등록 §12의 NP-1′…NP-10은 **그대로 유효하다**(NP-10은 아래 NPC-A로 정정). 이번 재감사가 **추가**하는 것:

> **NPC-A (등급 조건)** — **booted L boot이 2개 미만이거나 booted TD boot이 2개 미만인 상태에서 `S_TIER_MISMATCH`가 나오면, 사전등록 §8의 (D2) 규칙이 이를 `NO_VERDICT_INFRA`로 보고하게 되어 있으나 그 규칙은 두 경우에 과잉이다**: (i) 불일치 쌍이 **TD-TD**이면 그것은 §8·NP-10이 "방어 가능"이라 등록한 **true-dual 자기재현 실패**이고, (ii) L-L 귀무대조가 완전한데 TD boot만 1개이면 그것은 **진짜 교차-arm 불일치**다. 감사자 재현(무수정 체커, job 907456 아티팩트 + 1토큰 섭동): (i) `gen_L2` 제거 + `TD2` 섭동 → `S{L1-TD1:0, L1-TD2:1, TD1-TD2:1}`, `VERDICT FAIL`; (ii) `gen_TD2` 제거 + `TD1` 섭동 → `S{L1-TD1:1, L1-L2:0, TD1-L2:1}`, `VERDICT FAIL`. **따라서 (D2) 오버라이드가 발화하면 결과 문서는 반드시 `r2_correctness_report.json`의 `checks.S_pairs` 전체와 `per_boot[*].checks.B1_BOOTED`를 전사하고, 그 job에 D5의 재제출 1회를 쓰지 않는다.** "교차-arm 불일치가 없었다"는 문장은 이 경로에서 쓸 수 없다.

> **NPC-B** — **F2′ 충족은 KV 여유 가정의 "확인"이 아니다.** 투영치는 `max_total_num_tokens ≈ 2.72×10⁶` / `available_gpu_mem ≈ 11.6 GB`이고 등록 문턱까지의 거리는 각각 **≈11.0 GB / ≈6.6 GB**인데, 905835 대비 식별된 신규 소비자 총합은 **≈0.21 GB**다. **또한 사전등록이 "이 튜플에서 처음 바뀌는 성분"으로 든 "flashinfer workspace 버퍼"는 틀렸다** — job 905835가 이미 flashinfer였고, 그 버퍼는 `environ.py:349`의 **384 MB 전역 고정·ctx 무관**이다. F2′는 "거짓이 될 수 있다"는 형식 요건만 충족하며, 결과는 **기록**으로만 쓴다.

> **NPC-C** — **`r2_correctness.sbatch`의 I1 주석(작업트리 `:353-356`)은 D1이 철회한 문장을 그대로 들고 있다.** 사전등록 본문·NP-7′·F2′는 정확하다. **하네스 주석을 근거로 "48이 이 ctx/mem에서 측정됐다"를 인용하는 것을 금지한다.**

> **NPC-D** — **"`R2C_INSTRUMENT=0`이면 프로세스가 한 개도 추가로 뜨지 않는다 / 907100·907456과 명령 단위로 동일하다"는 거짓이다.** provenance 블록(`r2_correctness.sbatch:254-255`)이 `import flashinfer` 파이썬 프로세스를 **INSTR·백엔드와 무관하게 무조건** 1회 실행한다. 서버 부팅 전에 끝나므로 측정량은 바뀌지 않지만, **기본 경로의 "명령 단위 동일성"은 주장할 수 없다.**

> **NPC-E** — **provenance는 flashinfer 휠의 버전 문자열(`0.6.10`)과 경로만 기록한다.** 내용 해시도, JIT 캐시도 아니며, 그 휠은 repo 밖·venv 밖(`~/.local/lib/python3.14/site-packages/flashinfer/`)에 있어 manifest가 드리프트를 보지 못한다. **"같은 flashinfer에서 쟀다"는 쓸 수 없고 "0.6.10이라고 보고한 설치본에서"까지만 쓸 수 있다.** (완화: flashinfer autotune은 이 하드웨어에서 꺼진다 — A100 CC 8.0 < 9, moe 백엔드 `auto`.)

> **NPC-F** — **manifest 25항목의 파급 조사는 `lambda0.sbatch`까지만이다.** 항목 수/내용을 고정하는 비-테스트 소비자가 2곳 더 있다: `results/ltsm_probe/ltsm_p1_probe.sbatch:95-104`(참조 15줄과 **정확 diff**, 불일치 시 `exit 4`)와 `results/smid_census/smid_l0_census.py:219`(`MANIFEST_EXPECTED_FILES = 15`). **둘 다 15→17→24 단계에서 이미 깨져 있었고 25번째 줄이 새로 깨뜨리는 것은 아니다.** "manifest 변경이 저장소에 파급을 안 준다"는 문장은 쓸 수 없다.

> **NPC-G** — **`tests/test_r2_correctness_instrument.py`의 `BANNER` 상수는 합성 픽스처다** — 905835의 `max_total_num_tokens=2618868` / `available_gpu_mem=13.38 GB`에 `context_len=16384`를 붙인 줄이며, **그런 배너를 낸 boot는 존재한 적이 없다.** 이 문자열을 측정값으로 인용 금지.

> **NPC-H** — **제출 전 "전체 CPU 회귀"의 합격 기준은 이 트랙으로 한정해야 한다.** 감사 시점 전체 discovery는 **572 tests, failures=2 errors=2**이며 실패 4건은 **전부 `test_lambda0_prereg.py`**(병행 워크스트림의 rev3 진행 중): `TestMutationHarness.test_no_escapes`, `TestUnresolvedIsReachable.test_d4_injection_fails_on_rev1_glob`, `TestPlanIsDeterministic.test_every_registered_scenario_covers_the_registered_band`, `TestRuleReproducesTheAuditedVerdicts.test_probe_c_lambda_star_bit_for_bit`. **`test_r2_correctness_instrument.py`는 35/35 OK, r2_correctness 계열 실패 0.** 기준을 적지 않으면 제출 시점에 "이 실패를 무시해도 되는가"라는 **미등록 재량**이 생긴다.

> **NPC-I** — **I3의 `λ_inf`는 legacy warm-up boot · D44 · cudagraph ON에서만 측정된다.** λ0 rev3의 사다리가 이 값을 앵커로 쓰므로, **λ0 결과 문서는 NP-9의 "legacy 루프·warm-up boot 한정"을 그대로 승계해야 한다.** true-dual arm(B4)의 포화 상한은 이 job에서 측정되지 않는다.

> **NPC-J** — **§9(f)(벽시계 초과 → `R2C_INSTRUMENT=0` 재제출)는 D5의 재제출 정책 열거 정의역 밖이다.** 벽시계 강제 종료는 라벨을 낳지 않으므로 라벨 쇼핑은 발생하지 않지만, **두 경로를 합쳐 총 재제출 1회로 센다**는 것을 결과 문서에 명시한다.

---

## 6. 제출 전 확인 (감사 조건 아님 — 사전등록 자신이 건 것 + 권고)

**필수(사전등록 §12-b 자신의 D9)** — `provenance.txt`의 `commit=`이 실제로 돈 하네스를 가리키도록 아래를 커밋:
```
workspace/engine-port/results/r2_correctness/r2_correctness.sbatch                (M)
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh                       (M)
workspace/engine-port/results/r2_correctness/newpair_prereg/                      (??, 5파일: 이 판정서 포함)
workspace/engine-port/tests/test_r2_correctness_instrument.py                     (??)
```
(§12-b의 "아래 5개"는 4경로/7파일이므로 계수 문구 정정 권고. 병행 워크스트림 2개가 미커밋이므로 **선택 스테이징 필수**.)

**권고(비차단, 전부 텍스트·GPU 0)** — (1) NPC-C의 sbatch 주석 4줄 교체(테스트 무영향 확인함), (2) §8 (D2)를 §4-1의 **D2′**로 교체, (3) §6의 "프로세스 0개 추가" 문장에 provenance의 flashinfer import를 예외로 명기, (4) §13 회귀 합격 기준을 `test_r2_correctness_*`로 한정, (5) §11의 파급 문장에 ltsm/smid 2곳을 등재, (6) §5-I1/§7의 "flashinfer workspace가 처음 바뀌는 성분"을 삭제.

---

## 7. λ0와의 모순 판정

**이 문서 쪽이 옳다. 수정 불요.** 근거:
1. `λ_inf`(closed-loop, concurrency 64 포화 처리율)는 λ\*를 **위에서만** 구속한다. 어떤 비율 `c`에 대해서도 `c·λ_inf ≤ λ*`는 λ_inf가 주는 정보가 아니라 **추가 가정**이며, 그 가정이 틀리면 사다리 전체가 λ\* 위에 앉아 브래킷을 놓친다. D7/NP-3′는 정확히 이것만 말한다.
2. **λ0 워크스트림이 이미 이 문서를 수용해 자기 쪽을 고쳤다.** `lambda0_plan.py:66-75`가 이 사전등록 §5-I3 "(D7)"·NP-3′를 인용하며 `:101-106`에서 `MULT["A"][0]`을 **0.40 → 0.25**로 내리고 데이터 확인 분기(`LADDER_TOO_HIGH` → ÷4 재설계)를 등록했다. 파일을 직접 열어 확인(게이트 #110).
3. 정본 수치가 이 방향을 지지한다: `PROJECT_STATUS.md` 2026-09-13(3) 배너의 **λ\*_SLO < 0.59·λ\*_throughput(비 2.3–3.4×)** ⇒ SLO 기준 λ\*는 `λ_inf`의 **0.29–0.43배**까지 내려갈 수 있고, rev2의 하단 rung `0.40·λ_inf`는 그 구간 **안**이었다.
⇒ **이 문서에 D조건을 달지 않는다.** 대신 NPC-I(앵커의 arm 한정)를 λ0 쪽이 승계하게 한다.

---

## 8. 병행 워크스트림의 간섭 (게이트 #110)

| 축 | 영향 | 근거 |
|---|---|---|
| 조건 튜플 | **없음** | `server_cmd()`가 테스트로 바이트 핀이고 실제 파일과 일치. `benchmarks/pdmux_eval/{campaign,workloads}.py`·`lambda_star.py`·`scripts/r2_eval/*`는 이 sbatch가 임포트·호출하지 않는다. `pdmux_eval.context_limit`는 **미수정**이고, §13이 `R2C_CTX=16384`를 주므로 호출되지 않는다 |
| provenance | **없음(안전)** | manifest 접두 보존 확인, `lambda0.sbatch:97` `-lt 24` 통과. 단 NPC-F |
| 테스트 기준선 | ★**영향 있음** | 전체 discovery **572 tests, 4 실패 — 전부 `test_lambda0_prereg.py`** ⇒ NPC-H |
| 판정 코드 | **없음** | `r2_correctness_check.py` sha `ec355e17…` 재계산 일치, 클라이언트 무수정 |

---

## 9. 자기 적용 (게이트 #113)

- **D2′**: report JSON의 네 필드만 쓴다. 모두 실제 리포트에 존재함을 감사자가 만든 리포트로 확인. 체커 무수정, GPU 0, 사후 재량 0. **실행 가능.**
- **NPC-C 주석 교체**: 35 테스트의 `assertIn` 대상 문자열을 전수해 해당 4줄과 겹치는 것이 없음을 확인. **실행 가능.**
- **NPC-H 기준 한정**: 기존 테스트 파일명 패턴으로 표현 가능. **실행 가능.**
- **하지 않은 것**: `sync_engine_tree.sh`를 **실행하지 않았다**(엔진 트리에 쓰는 스크립트). §11의 "`sha256sum -c` 25/25 OK"는 파일 목록 + flashinfer 1건 해시 재계산 + 접두 diff로만 검증.
- **되살리지 않은 것**: 1차 판정서가 자기 철회한 `Concurrency:` 확인 처방.
- **전임 처방의 철회 1건**: 1차 판정서 **D2의 "TD 쪽도 대칭으로 적을 것"은 틀렸다**(§4-1, 수치 §1 행 4b). 감사자 자기 철회 **5회차 누적**. 이 철회를 정본에 등재할 것.
- **만들지 못한 것**: 자유 표면에서의 반전(N2)은 만들지 못했다. §1 행 4a/4b의 반전은 **부팅 결과**가 구동하며 연구자 재량이 아니다 — 전례 승계가 아니라 `s_ref_ok`의 양화 범위를 소스에서 다시 읽고 D2 술어를 직접 실행해 **독립 재도출**한 판단이다.

---

## 10. 요약

**`GO-with-caveats` — 제출 승인.** 死因 0(N1·N2·N3·N4 전부 미발화), 신규 자유 표면 8개에서 **재량 구동 반전 0건**, 결과 구동 반전 2건(§1 행 4a·4b, NPC-A로 전환). 1차 死因 N3는 **범주적으로** 해소됐다 — 단 F2′는 정보량이 사실상 0이므로 **NPC-B 없이는 인용할 수 없다**.

⚠️ **`GO-with-caveats`는 주장 범위가 좁아진다는 뜻이지 게이트를 닫았다는 뜻이 아니다.** 이 job은 PASS여도 **NP-8**대로 P2 착수를 승인하지 않으며, λ0(0단계) 사전등록은 **rev2 기준 `NO-GO`**(rev3 진행 중)이고 방법론 게이트 #6은 여전히 미충족이다.

**GPU 장부 불변**: 이 트랙 **0.43 GPU-h**, 이번 감사 신규 지출 **0**. 실행 시 예상 0.47–0.58 GPU-h(현실적 최악 1.15, 하드캡 2.5).
**새 성능 판정 0건** · HE0·정책 순위·Claim D/E 등급·stake #1 전부 불변.

---

## 11. 신규 게이트 후보 (doc-steward 이관용)

1. **감사자가 "대칭으로도 적어라"라고 처방할 때는 그 대칭이 코드의 양화 범위와 일치하는지 먼저 확인하라** — `s_ref_ok`는 L-L 쌍에만 양화되므로 TD 대칭 조항은 지키는 것이 없고 진짜 FAIL만 삼켰다(게이트 #110/#113 결합 재발, 감사자 자기 철회 5회차).
2. **"기본 경로에 프로세스가 하나도 안 는다"는 주장은 계측 블록이 아니라 스크립트 전체에 대해 검사하라** — 검사가 추출 블록만 돌면 provenance 블록에 추가된 프로세스를 못 본다(교훈 9 계열).
3. **죽은 문장을 사전등록에서 지울 때 실행체 주석에서도 지워라** — 사전등록 §5-I1은 고쳐졌는데 sbatch 주석이 D1이 철회한 "the single unmeasured premise"를 그대로 보존했고, 그 파일은 sha와 함께 job 디렉터리에 보존된다(교훈 80).
4. **manifest 항목 수 파급 조사는 "테스트"가 아니라 "소비자"로 하라** — 테스트는 0곳이지만 sbatch 정확-diff 1곳·분석 스크립트 상수 1곳이 있었고 둘 다 이미 깨져 있었다(게이트 #85).
5. **벽시계 최악은 게이트된 분기끼리의 배타성을 반영해 계산하라** — 판독 3000 s와 warm-up health 900 s는 `WARM_OK` 게이트로 거의 배타이며, 실제 구속 조건은 "161 > 150"이 아니라 **"warm-up health ≤ 220 s"**였다.
6. **(긍정)** 인접 사전등록이 서로 반대 방향을 등록했을 때, **한쪽이 상대 문서를 인용하며 자기 상수를 고친 것**은 모범 사례다(`lambda0_plan.py:66-75`, `MULT[A][0]` 0.40→0.25).

---

**관련 파일 (전부 절대경로)**

- 감사 대상: `.../results/r2_correctness/newpair_prereg/PREREG_NEWPAIR_2026-09-13.md` · `.../newpair_preflight_o1.py` · `.../newpair_preflight_o1_stdout.txt` · `.../VERDICT_newpair_rules_2026-09-13.md`(1차)
- 실행체·판정 코드: `.../results/r2_correctness/r2_correctness.sbatch`(미커밋, `:9`, `:193-198`, `:250-255`, `:302-326`, `:328-431`, `:353-356`, `:473-485`) · `.../r2_correctness_check.py`(sha `ec355e171a66d68eab1300edcc7616a693cd55aa2c3fb5c251ee4b42eac50d30`, 무수정, `:366`, `:430-500`) · `.../r2_correctness_client.py`(무수정) · `.../scripts/bootstrap/sync_engine_tree.sh`(미커밋, `:125-185`)
- 테스트: `.../tests/test_r2_correctness_instrument.py`(**35/35 OK**) · `.../tests/test_lambda0_prereg.py`(**4건 실패 — 병행 워크스트림**)
- 반전 계산 원자료: `.../results/r2_correctness/job_907456/`(gen/green 복사, srv/tel symlink, 무변경) · `.../job_907100/srv_L1.log`·`green_L1.json` · `.../probes/c_905835/srv_d44_r2_o96.log`(`:42`, `:53`, `:62`, `:64`) · `.../probes/c_capacity.sbatch`(`:67-72`, `:123-128`, `:139-145`) · `.../probes/pdmux_homog5.yml` · `.../benchmarks/configs/pdmux_r2.yml`
- 엔진: `.../sglang/srt/model_executor/model_runner_kv_cache_mixin.py`(`:223-230`, `:390-392`, `:857-876`) · `.../srt/model_executor/model_runner.py`(`:2054-2085`) · `.../srt/layers/attention/flashinfer_backend.py`(`:170-220`, `:519-549`) · `.../srt/environ.py:349` · `.../src/multiplex/multiplexing_mixin.py`(`:209-246`)
- 파급: `.../results/r2_eval/lambda0_prereg/lambda0.sbatch:90-101` · `.../lambda0_plan.py`(`:66-75`, `:101-106`) · `.../lambda0_prereg/VERDICT_lambda0_rev2_2026-09-13.md` · `.../results/ltsm_probe/ltsm_p1_probe.sbatch:95-104` · `.../results/smid_census/smid_l0_census.py:213-219` · `.../results/p1_gates/gate1/runtime_source_manifest_gate1c_877974.sha256`(15줄)
