<!-- claims-auditor 규칙층 감사 반환문 원문 전사 (2026-09-13, 메인 세션이 스크립트로
     추출; 재타이핑·요약 아님). 대상 = PREREG_X3_RUNNERCONF_2026-09-13.md rev1.
     판정 GO-with-caveats + 차단 D1-D13. 감사자가 자기 OS §8 문장 1건 추가 철회
     ("X3는 성능 트랙 전체를 연다" = 거짓, 블로커 3개 중 0개 제거).
     정본 정정: 캠페인 차단 run "약 270"은 중복계수 -> 고유 225 / 가능 180. -->

> ⚠️**SUPERSEDED (2026-09-13, 같은 날 사용자 결정, doc-steward
> 등재)** — 대상 사전등록(`PREREG_X3_RUNNERCONF_2026-09-13.md`)이
> **Zamba2-2.7B 기준**이었는데, 사용자가 같은 세션에서 4모델
> 캠페인의 Zamba2 슬롯을 `NemotronHForCausalLM` Nano-9B-v2-Base
> (ctx 131072)로 교체하기로 결정했다 — 이 판정서 §10.3이 스스로
> 예고한 "모델 교체면 X3는 전량 무효" 조건이 그대로 실현됐다.
> **이 판정서는 이력 보존용으로만 유지**한다: X3 job 자체는
> 다시 사지 않으며, 아래 §12·§13의 X3P/X3C 문안은 이 job이
> 재구매되지 않는 한 인용 대상이 아니다. 단 §14의 방법론 게이트
> G-X3-1…5는 방법론 지식으로서 유효하고 `CONSENSUS.md` §3
> 항목210–214·`PROJECT_STATUS.md` "방법론 게이트" #190–194로
> 이미 등재됐다. ★**산술 정정(§10.1)과 감사자 자기 철회(§10.1,
> §15 — "X3는 성능 트랙 전체를 연다"는 거짓)는 모델과 무관하게
> 유효하며 이미 정본(`PROJECT_STATUS.md`·`CONSENSUS.md`·
> `CLAIM_EVIDENCE_MATRIX.md`·`EXPERIMENT_ROADMAP.md`)에
> 반영됐다.** GPU 0 지출. 전문은 `PROJECT_STATUS.md` 최상단
> 배너(2026-09-13) "B"·"C" 절 참조.

모든 원자료를 직접 재계산하고, 무수정 판정 규칙을 반사실(反事實) 입력으로 실제 실행했다. 아래가 판정서 전문이다.

---

# X3(러너 설정 인증) 사전등록 규칙층 감사 판정서 (claims-auditor, read-only, GPU 0)

- 일자: 2026-09-13. 대상 = `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/x3_prereg/PREREG_X3_RUNNERCONF_2026-09-13.md` (sha `cf887161…` 확인) + `.../x3_prereg/forced_sample_dependency.py` (`0899e29f…`) + `_stdout.txt` (`d6a5b6f7…`) + 재사용 비교기 `.../x1_prereg/x1_cross_job_compare.py` (`9ba6b230…` 확인) + 미커밋 하네스 `.../r2_correctness.sbatch` (**현재 sha `9d79c713…`**, 907456은 `6d79fd9d…`) + 무수정 판정 규칙 `.../r2_correctness_check.py` (`ec355e17…` 확인).
- 방법: 등록 문서·스크립트의 **어떤 수치도 믿지 않고** `job_907100/`의 `tel_*.jsonl`(53 MB)·`gen_*.json`·`srv_*.log`·manifest에서 내 독립 스크립트로 재계산했다(등록 스크립트 미사용·미import). 그리고 ★**무수정 checker를 반사실 입력(907100에서 `trace_forced` 레코드만 제거한 telemetry)으로 실제 실행**해 X3가 내게 될 judgment를 선취했다. seed→토큰 경로는 sglang 소스(`sampler.py:105-107`, `sampling_params.py:98-100`, `sampling_batch_info.py:176`, `schedule_policy.py:298-300`, `server_args.py:351`, `tp_worker.py:311-317`, `http_server.py:1793-1804`, `utils/common.py:684-690`)를 직접 읽어 확인했다. 러너 튜플은 `engine_bench_runner.sh`·`r2_eval.sbatch`·`policy_adapter.sh`·`campaign.py`·`context_limit.py`를 읽어 대조했다. 엔진 핀의 충분성은 `sha256sum -c job_907100/runtime_source_manifest.sha256`으로 **실측**했다. 프로젝트 파일 수정 0건, GPU 0.
- **자기 발안 고지(필수)**: "A1 커밋 → **X3** → OS"라는 구매 순서는 **내가 OS 판정서 §8에서 낸 권고**다(설계는 메인 세션). 감사자는 자기 권고의 승인 주체일 수 없으므로, 이 회차의 1순위 과제는 내 권고를 깨는 것이었다. 결과: **내 OS §8 표의 핵심 문장 1건("X3는 성능 트랙 전체를 연다")을 이 판정서에서 철회한다**(§10). 처방자가 또 미끄러졌다(게이트 #113).

---

## §0 판정: **`GO-with-caveats`** — 死因 0 · 등록 caveat 11 · **차단 조건 D1–D13**

| 死因 | 발화 | 근거 |
|---|---|---|
| **N1** estimand 부재/항등식 | **미발화(경계)** | F2의 판정량(96 unit-pair의 greedy 토큰 동일성)은 엔진 강제 항등식이 아니고 거짓이 도달 가능하다(같은 비교기·같은 프로토콜에서 X1이 실제로 2단위를 잡았다). **단 검정력은 매우 낮다** — S 64쌍은 프로토콜이 산술을 고정하고(순차 bs=1, `forward_count = max(1, 65536//extend_num_tokens)` = f(prompt)), O 32쌍도 probe prefill이 단독 extend 배치다(클라이언트가 bg 첫 토큰을 기다린다). ⇒ **검정력 결손이지 도달 불가가 아니다**(D2·D10) |
| **N2** 반전 확인 | **미발화** | 자유 표면 **14개**를 등록 허용 범위 끝까지 밀었다(§2 표). **등록 라벨을 실제로 뒤집은 것 0건.** 라벨 반전을 **만들려 시도했다가 실패한 것 2건**(index-0 귀속, F3 기전)은 caveat로 전환했고, 제출 전 문안 수정으로 치유되는 사실 오류·무라벨 4건(F1, `PASS`, `INCONCLUSIVE`, O-only 귀속)은 차단 조건으로 전환했다 |
| **N3** 예측 도달 불가 | **미발화** | F1·F2·F3·F4 전부 거짓이 도달 가능하다. F3는 내가 **무수정 규칙으로 반사실 실행해 `NO_VERDICT_UNREALIZED`를 재현**했고(§5), 반대쪽(`PASS`)도 도달 가능하다 — 그래서 D5가 필요하다. 정의역은 처치 강도의 감소함수가 아니다(96 토큰쌍은 처치와 무관하게 고정) |
| **N4** 제1원칙 위반 | 미발화 | 서빙 직접 측정, 운영점(D44·cudagraph-ON·`triton_attention_num_kv_splits=8`·ctx 4096), 성능·정책 주장 0건(오히려 전면 금지), C층은 진단 전용 |

**차단 조건 D1–D13은 전부 GPU 0.** D1·D2·D3·D5는 **이행하지 않으면 결과 문장이 거짓이 된다**(F1은 측정하지 않은 것을 측정했다고 말하고, `PASS`는 무라벨이다). D7·D8은 미이행 시 "target은 기록됐으나 realized는 미확인"이 되어 게이트 #1 계열 재발이다.

---

## §1 독립 재계산 — 등록 수치는 정확하다. 스크립트가 **계산하지 않은** 주장도 참이다

| 항목 | 등록값(`forced_sample_dependency_stdout.txt`) | 내 독립 재계산 | 일치 |
|---|---|---|---|
| L1 snaps / forced / overlap / forced-ov / sched-ov | 10060 / 2285 / 600 / 582 / **18** | **동일** (97.0%) | ✓ |
| TD1 | 9157 / 1342 / 558 / 541 / **17** | **동일** (97.0%) | ✓ |
| L2 | 12414 / 2103 / 536 / 517 / **19** | **동일** (96.5%) | ✓ |
| TD2 | 9295 / 1263 / 532 / 517 / **15** | **동일** (97.2%) | ✓ |
| "전부 idx 4(D44)" — **스크립트가 계산하지 않는 주장** | 본문 §1 주장 | **독립 확인: 참.** 4 boot 전부 overlap 스냅샷의 `stream_index` 분포 = `{4: N}`, 스케줄분만 봐도 `{4: 18/17/19/15}` | ✓ |
| 스케줄 격자 규칙 | sbatch 주석 "grid bit-identical" | **비-forced 레코드 중 `sample_index ∈ {1} ∪ 32ℤ` 위반 0건**(4 boot 전부) | ✓(규칙만) |

**등록이 말하지 않은 수치 2개(반전 계산의 재료)**:
- **스케줄 스냅샷 총수는 동일 arm·동일 구성의 두 boot 사이에서도 L1 7775 vs L2 10311 (+32.6%)**로 흔들린다. ⇒ "계측을 끄면 스케줄 레코드가 몇 개 남는다"는 **외삽은 ±33% 자연 산포 안에 있다**. 동시에 스케줄-overlap 수는 15–19로 안정적이다(prefill 작업량이 고정이라서).
- **유지되는 관측자가 제거되는 관측자보다 boot당 더 자주 발화한다**: `--decode-log-interval 1`이 만드는 "Decode batch" 로그 줄 = **2617 / 2611 / 2580 / 2592**(boot당), 제거되는 forced 스냅샷 = **2285 / 1342 / 2103 / 1263**. (이벤트당 비용은 forced가 더 크므로 총 부하 비교는 불가 — 미지.) ⇒ D10.

---

## §2 반전 시험 표 (필수) — **자유 표면 14개, 등록 라벨 반전 0건**

**반전 계산의 재료 고정**(반전 계산 자신이 새 자유 표면이 되지 않도록): 원자료 = `job_907100/{tel_*.jsonl, gen_*.json, srv_*.log, runtime_source_manifest.sha256}`. 추정량 = (a) `runtime_snapshot`의 `prefill_active_batch_size ≥ 1 ∧ decode_running_batch_size ≥ 1`, (b) probe 창 = `gen_*.json`의 `probe.pc_start`–`pc_end`, (c) 판정 = **무수정 `r2_correctness_check.py`(`ec355e17…`)를 `trace_forced` 필터 입력으로 실행**(내가 규칙을 재구현하지 않았다), (d) 코드 사실 = sglang dev tree의 해당 file:line 직접 읽기. 임계·격자를 새로 도입한 곳은 없다.

| # | 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 |
|---|---|---|---|---|
| 1 | **F1의 "러너가 조립한 인자로 뜬다"** | 러너 코드가 실제로 실행되는지까지 | **반전(사실) — 이 job은 러너를 실행하지 않는다** | `r2_correctness.sbatch:177`은 `--context-length 4096`을 **리터럴**로 박는다. `pdmux_eval.context_limit` 호출처는 `r2_eval.sbatch:100` **단 1곳**이며 correctness sbatch에는 없다(grep 3건 전부 주석: `:21`·`:58`·`:133`). ⇒ **D1** |
| 2 | F1의 실패 분기("BOOT_FAILED면 A1의 ctx 유도가 틀렸다") | 끝까지 | **반전(사실) — 오배선** | 같은 근거. 게다가 이 튜플의 부팅 가능성은 **8 boot 선례**(907100 4/4, 907456 4/4)로 이미 확립, 차이는 seed와 TFP뿐 ⇒ F1의 정보량 ≈ 0. **D1** |
| 3 | **F2의 전이 문장**("eval 캠페인의 계측 구성으로 이전") | 캠페인의 실제 구성까지 | **반전(스코프) — 3중 과대** | (i) 관측자 4개 중 **2개 유지**; (ii) 엔진 소스가 **38c1aca로 핀**되는데 캠페인은 HEAD를 돈다 — `src/multiplex` **4파일 +564/−39**, `dual_worker.py` **+180**(TD hot loop); (iii) 캠페인 부하는 동시 부하이고 **C층은 같은 구성의 L-L에서도 4/32 불일치**다. ⇒ **D2** |
| 4 | F2 행2의 귀속(S·O 합산) | O만 갈린 경우까지 | **반전(귀속) — O에는 귀무대조가 없다** | 등록 귀무대조 `907032→907100`은 **S 0/32**뿐이고 907032에 `phase_o`가 없다. 선례: 907100→907456의 O 뒤집힘(O06)은 노드 gpu38→gpu42와 **동시 변경**이었고 O 귀무대조로 배제된 적이 없다 ⇒ O-only 불일치는 노드/물리GPU/triton cache와 **모호**. **D3** |
| 5 | **index-0 귀속 규칙(X1 D2(ii) 승계)** | "index 0도 처치에 귀속"까지 | **반전 실패 — 死因 아님. 단 논거가 틀렸다** | X1의 논거("`forward_extend`가 `num_kv_splits`를 안 읽는다")는 이 처치에 **무관**하다. 실제 방어막은 둘: 클라이언트가 `first_token.wait()`로 bg의 첫 토큰을 기다려 **probe prefill이 항상 단독 extend 배치**(`r2_correctness_client.py:237-241`), 그리고 layer-span 수 `max(1, 65536 // extend_num_tokens)`가 **프롬프트만의 함수**(`multiplexing_mixin.py:1170-1176`; probe 1437–2619 토큰 → 결정적). ⇒ 라벨("해석 보류")은 유지, **근거는 교체**. **D4** |
| 6 | **F3의 기전 서술**("15–19건 중 하나가 창에 들어와야") | 실제 창별 적중률 계산까지 | **반전 실패(결론은 오히려 강화) — 단 크기가 틀렸다** | 창별 스케줄-overlap: L1 8건(**7/8 probe**), TD1 7건(6/8), L2 10건(**8/8**), TD2 5건(5/8) ⇒ **probe-boot 26/32 = 81%가 적중한다.** F3가 성립하는 이유는 "거의 안 들어온다"가 아니라 **32중 전칭 요구**다. **D6** |
| 7 | **F3의 판정 예보 자체** | 무수정 규칙을 반사실 입력으로 실행 | **반전 없음 — 예보 재현** | 907100에서 forced 레코드만 제거 → `failures=[] infra=[] unrealized=['L1:O_TIER_CONFIRMED','TD1:O_TIER_CONFIRMED','TD2:O_TIER_CONFIRMED']`, **`VERDICT NO_VERDICT_UNREALIZED`**. 실패한 하위검사는 **O2(및 그에 딸린 O3)뿐**이며 O1·O4는 32/32 성립 |
| 8 | **`FAIL`이 진짜 게이트 신호인가** | checker 배선 끝까지 | **반전 없음 — 참이다** | `:433-435`가 B7·B8·O_TIER_CONFIRMED를 **`unrealized`로만** 보낸다. 반사실 실행에서 B2·B3(decode T/F 2617/0·2611/0·2580/0·2592/0)·B4·B5·B6 전부 True ⇒ 계측 OFF는 **FAIL을 만들 수 없다** |
| 9 | **F4(B7)와 미등록 B8** | B8까지 | **반전 없음 — 둘 다 통과** | 반사실: overlap 18/17/19/15 **전부 idx 4**, `green_realized=True` 4/4 ⇒ B8도 통과. 단 **B8은 F4에 등록돼 있지 않다**(무라벨 1칸). **D6** |
| 10 | **`PASS`·`INCONCLUSIVE`·`NO_VERDICT_INFRA`** | 출력공간 전수 | **반전 없음(라벨 불변) — 그러나 `PASS`가 무라벨이다** | 907100 타이밍 기준 P(PASS)는 작다(probe-boot 실패율 6/32 ⇒ 독립가정 0.8125³² ≈ **0.13%**; boot 기준 (1/4)⁴ ≈ 0.39%). **그러나 forced 제거는 overlap 구간 sync를 싸게 만들어 스케줄 밀도를 올릴 수 있고**, 자연 산포가 이미 +32.6%다 ⇒ **P(PASS)는 위 수치로 상한되지 않는다.** `PASS`는 가장 인용 위험이 큰 칸인데 라벨이 없다. **D5** |
| 11 | **묶음(seed 1000 + TFP 0)** | "seed가 토컨을 바꾼다"를 끝까지 | **반전 없음 — 경로가 없다** | temp 0 → `top_k=1`(`sampling_params.py:98-100`) → `is_all_greedy`(`sampling_batch_info.py:176`) → **`torch.argmax`**(`sampler.py:105-107`). RNG로 순서를 바꾸는 유일 소비자 `_sort_randomly`는 `--schedule-policy random`에서만 도달하고 기본값은 `fcfs`(`server_args.py:351`, 하네스 미설정). 서버 warmup은 **고정 `input_ids=[10,11,12]`**(`http_server.py:1801`). `set_random_seed`는 모델 로드 **후** 호출되고, 만약 출력에 영향하는 RNG 텐서가 있었다면 **907100의 4개 별 프로세스가 S/O 토큰을 비트 동일하게 낼 수 없었다**. ⇒ **confound #10 아님, 사실상 단일 변인**(단 seed 쪽 절반은 **코드 연역**이므로 "두 변인이 동시에 증명된다"는 등록 문장은 과대 — **D12**) |
| 12 | **비교기 재사용 적합성** | 인쇄 문안까지 | **반전 없음(판정 불변) — 인용 위험 1급** | `x1_cross_job_compare.py:130-135`가 아티팩트(`cross_job_compare.txt`)에 **"mismatch==0 → sensitivity not demonstrated"**를 찍는다. X3에서 0은 **등록된 긍정 결과**다 — 의미가 반대로 박힌다. `:160-161`은 "총수 감소는 decode attention을 row별 batch-invariant로 만든 효과"라는 **이 처치에 없는 기전**을 찍는다. 도구는 `9ba6b230…`로 무수정 유지해야 하므로 **문안으로만 차단 가능**. **D9** |
| 13 | `EXPECTED_PAIRS=96` / 라벨집합 / 분모 | 907100 자기비교 실행 | **반전 없음** | 907100↔907100 = `TOTAL compared unit-pairs = 96 (expected 96) mismatching = 0 [S 0/64, O 0/32] excluded(input)=0` |
| 14 | **엔진 소스 핀의 충분성**(target vs realized) | 설치 트리 실측 | **반전 없음 — 핀은 충분하다. 단 검증이 등록돼 있지 않다** | `sha256sum -c job_907100/runtime_source_manifest.sha256` 실행 결과 **17항목 중 13 OK, 4 FAILED**이고 FAILED는 정확히 `dual_worker/multiplexing_mixin/profile/controller`(= 핀이 되돌릴 4개). 설치=HEAD=`7a5022f8…/0fe9d570…/d7b6a807…/45fd2967…`, 38c1aca=`605b840d…/0b88c07c…/27c665c3…/722db60f…`(= 907100 manifest). ⇒ 핀 후 **17/17 OK가 되어야 한다**는 realized 검사를 등록하라. **D7** |

**등록 라벨을 뒤집은 곳: 0건 ⇒ N2 미발화.** #1·#2·#3·#4는 등록 **문안의 사실/스코프 오류**이고 제출 전 GPU 0으로 치유된다(OS 판정서의 D2·D3 선례와 동형). #5·#6은 **반전을 만들려 했으나 실패** — caveat로 전환한다. #10은 무라벨 칸이다.

---

## §3 질문 1 — ★"하이브리드가 아무것도 인증하지 못한다"를 끝까지 밀면

### 3.1 `0/96`이 실제로 인가하는 것 (정확히)

> 엔진 소스 `38c1aca`, Zamba2-2.7B, ctx 4096, D44, `PDMUX_R2_POLICY=fixed`, cudagraph-ON, `triton_attention_num_kv_splits=8`, greedy·`ignore_eos`에서 — **`PDMUX_TRACE_FORCE_PREFILL` 1→0과 `--random-seed` 1→1000을 동시에 바꿔도 S16·O8 프로토콜의 96 unit-pair greedy 토큰 ID가 job 907100과 비트 동일하다.**

이것의 정직한 이름은 **"러너 설정 인증"이 아니라 "계측 장치 효과(apparatus-effect) 음성대조"**다. 즉 **907100의 PASS가 그 자신의 추가 계측이 만들어낸 인공물이 아님**을 보인다. 이건 실재하는 질문이고, 실패할 수 있었다.

### 3.2 그러나 F2의 전이 문장은 **과대**다 — 세 겹으로

1. **관측자 2개가 남는다.** 그리고 남는 쪽이 **더 자주 발화한다**(boot당 로그 2580–2617줄 vs forced 스냅샷 1263–2285건). 이 job은 per-step 관측자 부하의 **작은 쪽**을 끄고 **큰 쪽**을 그대로 둔다.
2. **엔진 소스가 38c1aca로 핀된다.** 캠페인은 HEAD를 돈다 — `src/multiplex` 4파일 **+564/−39줄**, 그중 `dual_worker.py` **+180줄**이 **true-dual hot loop의 계측 발행 순서**를 바꾼다(A1/H2). 즉 **X3는 캠페인이 실제로 돌릴 바이너리에서의 동치를 인증할 수 없다.** (핀을 빼면 confound #10이므로 이건 설계 결함이 아니라 **구조적 교락**이다 — 그래서 caveat로 못박아야 한다.)
3. **검정력이 낮다.** S 64쌍은 순차 bs=1이고 prefill span 수가 `f(prompt)`라서 **타이밍으로 산술이 바뀔 경로가 없다**; O 32쌍도 probe prefill이 단독 extend 배치다. 반면 타이밍→토큰 결합이 **실측된 유일한 층은 C**(같은 구성 L-L에서 4/32 불일치)이고, 그 층은 F2에서 **제외**돼 있다. ⇒ `0/96`은 "계측이 무해하다"가 아니라 **"프로토콜이 이미 산술을 고정해 둔 96쌍에 처치가 닿지 않았다"**다.

### 3.3 그래도 하이브리드는 정당하다 — 다만 이유가 등록과 다르다

유지되는 2개는 **이 job의 스코프를 확정하는 장치**다: `--decode-log-interval 1` 없이는 B3(cudagraph-ON)의 증거가 0이고, `PDMUX_GREEN_READOUT=1` 없이는 B8(D44 realized)이 불가능하다. 둘을 빼면 `0/96`이 **어느 운영점에서 난 것인지 말할 수 없게 되어 해석 불가**가 된다. ⇒ **"러너 구성 인증"이 아니라 "검증된 운영점에서 수행한, 하네스 고유 노브 2개의 장치효과 음성대조"**로 재명명하라(D2).

---

## §4 질문 2 — 묶음은 confound #10인가: **아니다. 코드에 seed→토큰 경로가 없다**

| 경로 후보 | 확인 | 결론 |
|---|---|---|
| 샘플링 | temp 0 → `top_k=1` → `is_all_greedy` → `torch.argmax(logits,-1)` | **RNG 미사용** |
| 요청 순서 | `random.shuffle`은 `_sort_randomly`에만, `--schedule-policy random` 전용, 기본 `fcfs` | 미도달 |
| 서버 warmup | `input_ids=[10,11,12]`, temp 0 (`http_server.py:1793-1804`). `np.random` warmup(`warmup.py:48`)은 `--warmups` 전용 | seed 무관 |
| 가중치/버퍼 | `set_random_seed`가 model load **후** 호출. 게다가 907100의 **4개 독립 프로세스**가 S/O 토큰 비트 동일 | RNG 텐서 없음 |
| 클라이언트 프롬프트 | 클라이언트 `--seed`는 기본 20260911 고정, sbatch가 넘기지 않음 | 불변 |
| 파생 | `PDMUX_WORKLOAD_ID="r2corr_seed1000"` | telemetry 라벨뿐 |

⇒ **묶음은 사실상 단일 변인**이고 조건부 후속(§6)은 "불일치가 났을 때 seed를 의심한다"가 아니라 **"seed가 아니라면 남는 것은 TFP뿐이다"는 소거 논증**으로 써야 한다. 단 **등록 문장 "불일치가 0이면 두 변인 모두 토큰-중립임이 동시에 증명된다"는 과대** — seed 쪽은 증명이 아니라 **코드 연역의 재확인**이다(D12).

---

## §5 질문 3 — 기전 독립 재계산 + **F3 예보의 정량화**

### 5.1 창 안 적중률 (기존 데이터로 추정 가능하다 — 했다)

| boot | 창당 전체 스냅샷 | 창당 스케줄분 | 창당 overlap | **창당 스케줄-overlap** | ≥1인 probe | 창 밖 스케줄-overlap |
|---|---|---|---|---|---|---|
| L1 | 25–51 | 3–5 | 22–48 | 0,1,1,1,1,1,1,2 (합 **8**) | **7/8** | 10 |
| TD1 | 23–47 | 3–5 | 20–44 | 0,1,0,1,1,1,2,1 (합 **7**) | **6/8** | 10 |
| L2 | 25–51 | 4–5 | 22–48 | 1,1,1,1,1,1,2,2 (합 **10**) | **8/8** | 9 |
| TD2 | 23–43 | 3–4 | 20–40 | 0,0,0,1,1,1,1,1 (합 **5**) | **5/8** | 10 |

⇒ **probe-boot 26/32(81%)에서 O2가 성립한다.** 등록 §1의 "15–19건 중 하나가 8개 창에 들어와야 한다"는 **적중 희소성을 암시해 크기를 틀렸다** — F3가 성립하는 진짜 이유는 **32칸 전칭 요구**다.

### 5.2 무수정 규칙으로 반사실 실행한 결과 (F3·F4가 **데이터로** 뒷받침된다)

```
r2_correctness verdict rule v2 (2026-09-11)  boots=['L1','TD1','L2','TD2'] booted=[...] D=44
  L1   ... overlap=True (18/18 on idx 4) green_realized=True O_confirmed=False
  TD1  ... overlap=True (17/17 on idx 4) green_realized=True O_confirmed=False
  L2   ... overlap=True (19/19 on idx 4) green_realized=True O_confirmed=True
  TD2  ... overlap=True (15/15 on idx 4) green_realized=True O_confirmed=False
  S mismatches: 6쌍 전부 0 / O within 0 cross 0
  failures=[] infra=[] unrealized=['L1:O_TIER_CONFIRMED','TD1:O_TIER_CONFIRMED','TD2:O_TIER_CONFIRMED']
VERDICT NO_VERDICT_UNREALIZED
```

- **F3 지지**: 예보된 판정 단어가 그대로 나온다. 실패 하위검사는 **O2(+O3)뿐**이고 **O1·O4는 32/32 성립**한다 — 즉 "probe가 bg와 함께 decode했다"는 증거(`decode_bs ≥ 2`)는 **스케줄분만으로도 32/32 남는다**. 잃는 것은 오직 **"probe prefill이 in-flight인 순간의 목격자"**다. 병기 문안은 이 구분을 지켜야 한다.
- **F4 지지 + 확장**: B7 통과(15–19)**이고 B8도 통과**(전부 idx 4 + green realized). B8은 등록에 없다(D6).
- **단 이 반사실은 907100의 타이밍을 쓴 것**이다. 실제 forced-OFF 런은 overlap 구간 sync가 싸져 스케줄 밀도가 **올라갈 수 있다**(자연 산포가 이미 L1 7775 → L2 10311 = +32.6%). 방향은 O2 성립 쪽이므로 **P(PASS)는 §2#10의 0.13–0.39%로 상한되지 않는다** ⇒ D5.

---

## §6 질문 4 — 출력공간 라벨 전수성: **무라벨 4칸, 그중 하나는 1급 위험**

| 칸 | 도달성 | 현재 | 조치 |
|---|---|---|---|
| **`PASS`**(전 boot O 확인 + 토큰 전부 일치) | **미상**(907100 타이밍 기준 0.13–0.39%, 상한 아님) | F3 하위분기가 "O가 확인되면 추가 정보"라고만 적음. **판정 단어 `PASS`에 라벨 없음** | **D5** — "러너의 forced-sample 설정에서도 게이트가 통과했다"는 문장은 **인용 금지**로 선제 차단 |
| **`INCONCLUSIVE`**(L1≠L2 S, 또는 within-arm O 불일치) | 관측 기저율 0/3 job | 무라벨 | **D5** — 이 칸은 "계측이 arm 내부 재현성을 깼다"는 **강한 결과**이며 F2 행2보다 심각 |
| **`NO_VERDICT_INFRA`**(boot 결손·크래시 없음) | 가능 | 무라벨 | **D5** |
| **B8 실패** | 반사실 0/4 | F4는 B7만 | **D6** |
| F2 ≥1 ∧ `FAIL` | 가능 | "같은 사건일 수 있다 → escalate" | **충분**(단 `S_TIER_MISMATCH`는 `s_ref_ok`가 참일 때만 발화 = L 두 boot이 일치할 때 ⇒ "계측이 TD만 바꿨다"는 의미임을 병기) |
| F2 ≥1 ∧ 판정 `NO_VERDICT_UNREALIZED` | **가장 가능성 높은 불일치 시나리오**(X1 패턴: 4 boot 전부 동일하게 이동) | 행2가 다룸 | 충분 |
| "O가 운 좋게 확인" | §5 | 보수 방향으로 라벨됨(일반화 금지) | **사후 합리화 통로 아님** |

---

## §7 질문 5 — 재사용 비교기: 도구는 적합, **인쇄 문안은 부적합**

- **적합한 것**: 분모 96 요구·`prompt_sha256`∧`prompt_tokens` 이중 검사·S/O 분리 보고·`first_divergence` 보고·C층 `NOT-RUN` 가드·selftest(1토큰 변경 검출). 907100 자기비교로 **96/96·0 불일치**를 재현했다. `EXPECTED_PAIRS=96`·라벨 집합은 이 설계에서도 타당하다.
- **부적합한 것(문안)**: `:130-135`의 "reading (F2)"는 **이 처치의 기전을 오기**하고 **0의 의미를 반대로** 읽는다. `:160-161`의 C층 문장은 존재하지 않는 기전을 단정한다(X1C-12 인접). 도구는 무수정이어야 하므로 **문자 그대로의 무효 선언**이 유일한 차단이다(D9).
- **양성대조의 한계(요청한 문안)**: X1의 S05@25·O06@11 검출은 **비교기의 검출 감도**(1토큰 차이를 본다)를 시연한다. **처치의 감도는 시연하지 않는다** — X1의 교란은 커널 **산술**(reduction order)을 바꿨고, X3의 처치는 **스케줄러 타이밍**을 바꾼다. 타이밍→S/O 토큰 경로는 이 프로토콜에 **알려진 것이 없다**(§3.2-3). ⇒ **`0/96`에는 양성대조가 없다**; 그리고 타이밍 처치의 양성대조를 만들 수 있는 유일한 층은 C인데 그 층은 귀무대조가 없어(R2C-3) 아무것도 인가하지 못한다. **이 결손은 이 설계로 메울 수 없다**(D10).

---

## §8 질문 6 — 선결 #4b 스코프: "토큰 쪽 절반"은 **부정확하다**

정본(`CLAIM_EVIDENCE_MATRIX.md` Claim D 행)의 미해소 목록은 **"러너 설정(ctx 16384 부팅 실패, dirname)"**과 **"observer effect"**를 **별 항목**으로 둔다. 정확한 사후 상태는 이렇게 적어야 한다.

> **선결 #4b(observer effect)**: 여전히 **미해소**. X3가 준 것은 "**하네스 고유 관측자 4개 중 2개**(`PDMUX_TRACE_FORCE_PREFILL`, `--random-seed`)를 러너 값으로 바꿨을 때 **S16+O8 96 unit-pair의 토큰이 job 907100과 같았다**"는 **cross-job·job n=1·토큰 전용** 사실뿐이다. 유지된 관측자 2개(`--decode-log-interval 1`, `PDMUX_GREEN_READOUT=1`)는 **미시험**이고, 유지된 쪽이 boot당 더 자주 발화한다(2580–2617 vs 1263–2285). **타이밍/성능 쪽(paired CI < 3%)은 측정조차 하지 않았다**(교차-잡 성능 비교는 금지). 조건 = {38c1aca, Zamba2-2.7B, ctx 4096, D44, fixed, cudagraph-ON, K=8, seed 1000}.
> **선결 #2의 "러너 설정" 항목은 X3가 닫지 않는다** — 이 job은 `engine_bench_runner.sh`·`policy_adapter.sh`·`pdmux_eval.context_limit`를 **한 줄도 실행하지 않는다**(ctx 4096은 `r2_correctness.sbatch:177` 리터럴). 그 항목은 A1의 CPU 인증 + **실제 캠페인 런**으로만 닫힌다.
> **O층 8단위의 프로토콜 확인(중첩·D44 상주)은 907100에서 상속**된 것이고 이 job은 재확인하지 않는다. 단 `decode_bs ≥ 2`(probe가 bg와 같은 decode 배치) 증거는 스케줄분만으로도 남는다.

---

## §9 질문 7 — 하네스 변경 검토: **판정 규칙 무영향, 기본값 보존 확인. provenance에 구멍 2개**

| 점검 | 결과 |
|---|---|
| `r2_correctness_check.py` 변경 | **없음**(`ec355e17…` 확인) |
| `server_cmd`·단계·순서·warmup·클라이언트 | **불변**(diff 확인) |
| `TFP="${R2C_TRACE_FORCE_PREFILL:-1}"` | **907100/907456 동작 보존**(env 문자열 `PDMUX_TRACE_FORCE_PREFILL=1` 동일) |
| `*_prereg` 해시 일반화 | 판정 무영향. 단 `b3_prereg`·`os_prereg`(VERDICT 포함)까지 해시되고 그들은 **untracked**이므로 `commit=`이 핀하지 못한다(F5 서술 보완) |
| `sha256sum ... 2>/dev/null` 실패 내성 | `set -e` 없음 ⇒ 안전 |
| **미커밋** | `git status --porcelain` = sbatch **M** 1건 + `x3_prereg/` **??**. `harness sha256`이 sbatch 자신을 찍으므로 자기-핀은 되지만, **커밋 없이는 등록이 commit에 매이지 않는다**(X1 rev1·OS D8 동형) ⇒ **D11** |
| **TFP의 realized 증거** | provenance는 `trace_force_prefill=0`(= **target**)만 남긴다. 엔진이 실제로 그 값을 봤다는 증거는 **등록돼 있지 않다** ⇒ **D8**(검증 가능: force OFF면 `trace_forced` 필드가 **전혀** 기록되지 않고(`multiplexing_mixin.py:606-612`) 비-forced 격자 위반이 0이다 — 907100에서 0건 실측) |
| 주석 사실성 | sbatch `:127-130`의 "SCHEDULED sample grid is bit-identical"은 **격자 규칙에만** 참이다(레코드 수·내용은 불변 아님: L1 7775 vs L2 10311). 또 `:59` 주석 "runner default 16384"는 A1 이후 **stale** ⇒ **D12** |

---

## §10 질문 8 — 우선순위: ★**내 OS §8 권고의 한 줄을 철회한다**

### 10.1 철회

> **철회**: OS 판정서 §8 표의 "트랙 해제 — X3: **성능 트랙 전체를 연다**(`results/r2_eval` 캠페인은 여전히 미생성)"는 **거짓이다.** X3는 캠페인 블로커를 **하나도** 제거하지 않는다.

근거(내 독립 재계산):

| 블로커 | run 수 | X3가 제거? | 근거 |
|---|---|---|---|
| W2/W4/W5가 8192-토큰 프롬프트 → ctx 4096에서 `context_limit`가 **거부** | 135 | **아니오** | `workloads.py:148,159,180` |
| B2/B8 `requires_offline_oracle` → `policy_adapter.sh:24-27` exit 2 | 90 | **아니오** | 오프라인 스윕 산출물 필요 |
| B6 `PDMUX_MODEL_PROFILE_PATH: unbound` | 45 | **아니오** | **hybrid profile JSON이 저장소 전체에 0건**(find 결과 0) |

**★산술 정정**: 정본의 "약 270 run 실행 불가"는 **중복 계수**다. 포함-배제로 **고유 차단 = 225**(135+90+45 − 30[W2/4/5×B2,B8] − 15[W2/4/5×B6]), **실행 가능 = 180**(W1,W3,W6,W7,W8,W9 6종 × 5 rep × B0,B1,B3,B4,B5,B7 6종). 정본 수치를 **225 차단 / 180 가능**으로 정정하라(`PROJECT_STATUS.md` 배너·`EXPERIMENT_ROADMAP.md` 해당 절).

### 10.2 그리고 더 나쁜 것: X3는 캠페인이 돌릴 **바이너리**를 인증하지 않는다

캠페인은 HEAD(또는 이후)를 돈다. X3는 38c1aca로 핀된다(핀 없으면 confound #10). `src/multiplex` 차분 **4파일 +564/−39**, `dual_worker.py` **+180**(TD hot loop 계측 발행 순서). ⇒ 캠페인 구성에 대한 토큰 동치는 **X3 이후에도 미확립**이다.

### 10.3 권고 (수정판)

1. **캠페인 설계 결정이 X3보다 먼저다.** 해상(解像)에 따라 X3의 유효성이 갈린다: 트레이스 재생성(≤4094 토큰)이면 X3는 유효하되 소스 축이 남고, **모델 교체면 X3는 전량 무효**(ctx·커널 shape·프로토콜 전부 변경), Claim E 포기면 B6 블로커가 소멸한다. **0.16을 쓰기 전에 이 결정을 받아라.**
2. 그래도 지금 0.16을 쓴다면 **X3 > OS는 유지**한다(OS는 어떤 선결도 닫지 않고 재현 대상이 1/3 확률 사건). 단 X3의 가치는 "트랙 해제"가 아니라 **"907100 PASS가 자기 계측의 인공물이 아님"**이다 — 그것만으로도 0.16은 방어 가능하다.
3. **비권고·비차단(내 발안이므로 차단 불가, 실현가능성 검사 포함)**: 캠페인 바이너리를 인증하려면 **HEAD에서 ON·OFF 2 job(0.32 GPU-h)** 이 필요하다(ON-at-HEAD가 907100에 소스 축만으로 재앵커, OFF-at-HEAD가 노브를 잰다). 실현가능성: 하네스·판정 규칙 무수정, env 2개, checker의 `SNAP_KEYS` 생산자가 HEAD에도 존재함이 테스트로 고정돼 있어 규칙은 돈다. **비용 2배이고 캠페인 설계 결정 전에는 역시 낭비**다.
4. **실현가능성 자기검사(게이트 #113)**: D1–D13은 전부 문안·커밋·해시 검증(GPU 0)이며, D7은 `git checkout` + `sync_engine_tree.sh` + `sha256sum -c` 3줄, D8은 job 종료 후 jq 한 줄이다. **내 처방이 실행 불가로 함께 죽는 경로는 §10.3-3(내가 비권고로 돌린 것) 하나뿐이고 그것을 명시했다.**

---

## §11 제출 전 필수 조건 (D1–D13, 전부 차단 · GPU 0)

- **D1 (F1 재작성 — 측정하지 않은 것을 측정했다고 쓰면 안 된다)** "러너가 조립한 ctx·인자로 서버가 실제로 뜬다"를 삭제하라. 이 job은 **러너 코드를 실행하지 않는다**(`r2_correctness.sbatch:177` `--context-length 4096` 리터럴; `pdmux_eval.context_limit` 호출처는 `r2_eval.sbatch:100` 단 1곳). 대체 문안: "4 boot 전부 `booted`·`crash_free`는 **하네스 튜플이 seed 1000·forced-OFF에서도 부팅한다**는 정합성 확인이며, 이 튜플의 부팅 가능성은 **907100·907456의 8 boot**에서 이미 확립됐다(차이는 seed와 TFP뿐). **실패 분기도 고쳐라** — `BOOT_FAILED`는 A1의 ctx 유도와 **무관**하다(그 코드는 실행되지 않는다); 인프라/엔진 회귀로 트리아지한다."
- **D2 (F2 0/96 라벨 재작성 — 전이 문장이 3중 과대)** "eval 캠페인의 계측 구성으로 이전된다"를 삭제하고 §3.1의 문장으로 교체하라. 함께 등록할 수치: (a) 관측자 **2/4만** 시험, 유지된 쪽이 boot당 더 자주 발화(**2580–2617 로그줄 vs 1263–2285 forced 스냅샷**); (b) 엔진 소스 **38c1aca 핀**, 캠페인은 HEAD — `src/multiplex` **4파일 +564/−39**, `dual_worker.py` **+180**(TD hot loop); (c) C층은 같은 구성의 **L-L에서도 4/32 불일치**이므로 동시 부하로의 전이는 근거가 없다. 또 **907456은 비교 대상이 아니다**(baseline은 907100 단일) — "907456의 결론도 이전된다"고 쓰지 말 것.
- **D3 (F2 행2를 S/O로 분리 — O에는 귀무대조가 없다)** "≥1 ⇒ 러너 구성이 토큰을 바꾼다"를 둘로 쪼개라. **S에서 ≥1**: 귀무대조 `907032→907100` **S 0/32**(노드·GPU·커밋 상이, 무교란)가 있으므로 처치에 귀속 가능. **O에서만 ≥1**: **O 귀무대조가 존재하지 않는다**(907032에 `phase_o` 없음) ⇒ 노드/물리GPU/triton cache와 **모호**하며 "러너 구성이 토큰을 바꾼다"고 쓸 수 없다. 후속(같은 노드에서 forced-ON 재실행)을 조건부 등록하라.
- **D4 (F2 행3 근거 교체 — 라벨은 유지)** "prefill 산출이므로 계측으로 설명되지 않는다(X1 D2(ii) 승계)"의 **근거를 교체**하라. X1의 근거(`forward_extend`가 `num_kv_splits`를 안 읽는다)는 이 처치에 무관하다. 옳은 근거: (i) 클라이언트가 bg 첫 토큰을 기다려 **probe prefill이 항상 단독 extend 배치**(`r2_correctness_client.py:237-241`), (ii) layer-span 수가 `max(1, 65536 // extend_num_tokens)` = **프롬프트만의 함수**(`multiplexing_mixin.py:1170-1176`), (iii) S는 순차 bs=1. ⇒ index-0 불일치는 이 처치의 기전으로 설명되지 않는다(라벨 유지).
- **D5 (무라벨 3칸 — `PASS`가 1급 위험)** (i) **`PASS`**: 라벨을 지금 못박아라 — "전 boot에서 O1–O4가 확인되고 토큰이 전부 일치하면 그것은 **샘플 밀도의 운**이며 '러너의 forced-sample 설정에서 게이트가 통과했다/선결 #2가 넓어졌다'고 **쓰지 않는다**." 도달 확률은 907100 타이밍 기준 **0.13%(probe-boot 독립가정)~0.39%(boot 기준)**이지만 **상한이 아니다**(forced 제거가 overlap 구간 스케줄 밀도를 올릴 수 있고 자연 산포가 +32.6%). (ii) **`INCONCLUSIVE`**: "계측 변경이 **arm 내부 재현성**을 깼다"는 뜻이며 F2 행2보다 **강한** 불리 결과로 등재한다(관측 기저율 0/3 job). (iii) **`NO_VERDICT_INFRA`**: 측정 실패로 기록하고 게이트 실패로 적지 않는다.
- **D6 (F3·F4 수치 등록 + B8 명기)** 다음을 본문에 넣어라. "무수정 `r2_correctness_check.py`(`ec355e17…`)를 **907100에서 `trace_forced` 레코드만 제거한 입력**으로 실행하면 `failures=[] infra=[] unrealized=[L1·TD1·TD2의 O_TIER_CONFIRMED]`, **`VERDICT NO_VERDICT_UNREALIZED`**가 나온다(L2는 8/8 확인). 실패 하위검사는 **O2(+O3)뿐이고 O1·O4는 32/32 성립**한다 — `decode_bs ≥ 2` 증거는 스케줄분만으로도 32/32 남는다. 창별 적중은 **probe-boot 26/32(81%)**이고, F3가 성립하는 이유는 희소성이 아니라 **32칸 전칭 요구**다. **B7뿐 아니라 B8도 통과가 예상된다**(스케줄-overlap 15–19 **전부 idx 4** + green realized 4/4)." 그리고 "이 반사실은 907100의 타이밍을 쓴 것이므로 실제 런의 밀도는 달라질 수 있다"를 병기하라.
- **D7 (핀의 realized 검증 — target 아닌 realized, 게이트 #1)** (a) 제출 전: `git checkout 38c1aca -- workspace/engine-port/src/multiplex` → `sync_engine_tree.sh` → **`sha256sum -c job_907100/runtime_source_manifest.sha256`가 17/17 OK**임을 확인하고 출력을 `x3_prereg/`에 남겨라(오늘 실측: 핀 전 **13 OK / 4 FAILED**이고 FAILED가 정확히 핀 대상 4개 ⇒ 핀은 **충분하다**). (b) 완료 후: `git checkout HEAD -- …` **+ `sync_engine_tree.sh` 재실행**으로 **설치 트리**를 HEAD로 되돌리고 `dual_worker.py`=`7a5022f8…`·`multiplexing_mixin.py`=`0fe9d570…`·`profile.py`=`d7b6a807…`·`controller.py`=`45fd2967…`를 직접 대조하라(재sync 없이 repo만 복원하면 **다음 job이 조용히 38c1aca로 돈다**). (c) 핀 구간에서는 CPU 테스트를 돌리지 마라(A2가 추가한 이름을 되돌린 상태다).
- **D8 (TFP=0의 realized 채널 등록)** F5에 추가: "**realized 검사**: 4 boot 전 `runtime_snapshot`에 `trace_forced` 필드가 **한 건도 없고**(force OFF에서만 생략된다, `multiplexing_mixin.py:606-612`), 비-forced 격자 위반(`sample_index ∉ {1} ∪ 32ℤ`)이 **0건**임을 확인한다(907100에서 0건 실측)." provenance의 `trace_force_prefill=0`은 target이다.
- **D9 (재사용 비교기 인쇄 문안 무효 선언 — 문자 그대로)** 도구는 `9ba6b230…`로 **무수정 유지**하므로 다음을 등록 문안에 **문자 그대로** 넣어라. > "`cross_job_compare.txt`에 인쇄되는 `reading (F2)` 문단과 C층 해설 문장은 **X1 전용이며 이 job에는 적용되지 않는다**. 특히 '`mismatch==0` → sensitivity not demonstrated'는 X3에서 **의미가 반대**이고(0은 등록된 긍정 결과다), 'a drop in the total is an EXPECTED side effect of making decode attention batch-invariant per row'는 **이 처치에 존재하지 않는 기전**이다. 두 문장은 **인용 금지**이며 판정은 오직 이 사전등록의 F1–F5 라벨로만 읽는다."
- **D10 (검정력·양성대조 결손 등록)** 다음을 등록하라. "**`0/96`에는 양성대조가 없다.** X1의 S05·O06 검출은 **비교기의 검출 감도**를 시연하며 **처치의 감도는 시연하지 않는다**(X1은 커널 산술, X3는 스케줄러 타이밍). 그리고 96쌍의 산술은 프로토콜이 고정한다 — S는 순차 bs=1, probe prefill은 단독 extend 배치, span 수는 `f(prompt)`. 타이밍→토큰 결합이 실측된 유일한 층은 **C**이고 그 층은 귀무대조가 없어 아무것도 인가하지 못한다(R2C-3). ⇒ `0/96`은 '계측이 무해하다'가 아니라 **'처치가 이 96쌍에 닿지 않았다'**다. 또 이 job은 per-boot 발화가 더 많은 관측자(`--decode-log-interval 1`, 2580–2617줄)를 **유지**하고 더 적은 쪽(forced 1263–2285건)을 끈다 — 총 관측자 부하는 시험되지 않는다(이벤트당 비용은 미지)."
- **D11 (provenance·커밋)** (a) `x3_prereg/`를 **커밋**하라(현재 untracked). (b) **sbatch를 커밋**하고 **새 sha `9d79c713…`**(907456은 `6d79fd9d…`)를 등록 문안에 적어라 — 판정 규칙 `ec355e17…`은 무수정 유지. (c) 제출 전 `git status --porcelain` **전체**를 기록하라. (d) **`R2C_BASELINE_JOB=.../job_907100`을 명시**하라(미설정이면 1차 산출물이 생성되지 않는다). (e) 주장이 "러너 구성"이므로 **러너 4파일의 sha256**(`engine_bench_runner.sh`·`policy_adapter.sh`·`r2_eval.sbatch`·`benchmarks/pdmux_eval/context_limit.py`)과 그 commit(`7ac3d23`)을 등록에 적어라 — 적지 않으면 "러너 구성"은 나중에 조용히 움직이는 과녁이다. (f) F5 서술 보완: 해시 블록은 `x1_prereg`·`x3_prereg`뿐 아니라 **`b3_prereg`·`os_prereg`(VERDICT 포함, 둘 다 untracked)**까지 찍는다.
- **D12 (사실 정정 2건)** (a) sbatch `:127-130`과 등록 §변인의 "SCHEDULED sample grid is bit-identical"을 "**격자 규칙**(`sample_index ∈ {1} ∪ 32ℤ`, 907100에서 위반 0건)은 동일하다. **레코드 수와 내용은 불변이 아니다** — 동일 구성 두 legacy boot 사이에서도 스케줄 스냅샷 수가 **7775(L1) vs 10311(L2)**로 흔들리고, force 모드에서는 스케줄 레코드에 `trace_forced=false` 필드가 **추가**된다"로 바꿔라. (b) "불일치 0이면 두 변인 모두 토큰-중립임이 **동시에 증명**된다"를 "**TFP는 측정으로, seed는 코드 연역(greedy argmax·fcfs·고정 warmup ids)으로** 각각 지지된다 — seed 쪽은 새 증거가 아니다"로 바꿔라. (c) sbatch `:59` 주석의 "runner default 16384"는 A1 이후 stale임을 등록에 적어라(러너는 이제 모델 config에서 4096을 유도한다).
- **D13 (선결 #4b 문안 확정)** §8의 두 단락을 **문자 그대로** 등록하라(특히 "**선결 #2의 '러너 설정' 항목은 X3가 닫지 않는다 — 이 job은 러너 코드를 한 줄도 실행하지 않는다**").

---

## §12 실행 후 필수 병기 (X3P-1…7) — **문자 그대로 승계**

> **X3P-1 (스코프)** "job \<id\>는 **러너 설정 인증이 아니라, 검증된 운영점에서 수행한 하네스 고유 노브 2개의 장치효과 음성대조**다. 인증 조건 튜플 = {엔진 소스 **38c1aca**(`runtime_source_manifest.sha256` 17/17 job 907100과 바이트 동일), Zamba2-2.7B, TP=1, ctx **4096**, D44, `PDMUX_R2_POLICY=fixed`, cudagraph-ON(decode), `triton_attention_num_kv_splits=8`, greedy·`ignore_eos`, seed **1000**, boot 순서 `L TD L TD`}. **이 job은 `engine_bench_runner.sh`·`policy_adapter.sh`·`pdmux_eval.context_limit`를 실행하지 않는다** — ctx 4096은 `r2_correctness.sbatch:177`의 리터럴이다."

> **X3P-2 (잔여 관측자 — 더 자주 발화하는 쪽이 남았다)** "`--decode-log-interval 1`과 `PDMUX_GREEN_READOUT=1`은 **유지**됐고 시험되지 않았다. 유지된 쪽이 boot당 더 자주 발화한다(Decode 로그 **2580–2617줄** vs 제거된 forced 스냅샷 **1263–2285건**; 이벤트당 비용은 미지). 둘을 유지한 이유는 그것들이 각각 B3(cudagraph-ON)과 B8(D44 realized)의 **유일한 증거 채널**이어서, 빼면 결과가 어느 운영점에서 난 것인지 말할 수 없기 때문이다."

> **X3P-3 (엔진 소스 축 — 캠페인 바이너리는 미인증)** "이 결과는 **38c1aca**에서의 사실이다. eval 캠페인은 HEAD 이후를 돈다 — `src/multiplex` **4파일 +564/−39줄**, 그중 `dual_worker.py` **+180줄**이 true-dual hot loop의 계측 발행 순서를 바꾼다(A1/H2). **따라서 X3는 캠페인이 실제로 돌릴 엔진 소스에서의 토큰 동치를 인증하지 않는다.** 핀을 빼면 계측과 소스를 동시에 바꾼 것이 되어 confound #10이므로 이 결손은 설계 결함이 아니라 **구조적 교락**이다."

> **X3P-4 (검정력·양성대조 결손)** "`0/96`에는 **양성대조가 없다**. 비교기의 검출 감도는 X1에서 시연됐으나(1토큰 차이 검출) 그것은 **커널 산술** 교란이었고 이 처치는 **스케줄러 타이밍**이다. 96쌍의 산술은 프로토콜이 고정한다(S 순차 bs=1; probe prefill 단독 extend 배치 — 클라이언트가 bg 첫 토큰을 기다린다; layer-span 수 = `max(1, 65536//extend_num_tokens)`). 타이밍→토큰 결합이 실측된 유일한 층은 **C**이고 귀무대조가 없다(같은 구성의 L-L 불일치 4/32). ⇒ `0/96`은 '계측이 무해하다'가 아니라 **'처치가 이 96쌍에 닿지 않았다'**로만 읽는다."

> **X3P-5 (판정 단어 — 구성상 그렇다)** "`NO_VERDICT_UNREALIZED`는 **게이트 실패가 아니다**(교훈 21). 기전은 forced 샘플이 O층 관측의 유일한 도구라는 것이며, 이것은 사전에 **무수정 판정 규칙을 907100의 forced-필터 입력으로 실행해 재현**됐다(`unrealized` = 3 boot의 `O_TIER_CONFIRMED`, `failures=[]`, `infra=[]`). 실패 하위검사는 **O2(+O3)뿐**이고 O1·O4는 32/32 성립한다 — probe가 bg와 같은 decode 배치에 있었다는 증거(`decode_bs ≥ 2`)는 스케줄분만으로도 남고, 잃는 것은 **probe prefill이 in-flight인 순간의 목격자**다. **O층 8단위의 중첩·D44 상주 프로토콜 확인은 907100에서 상속된 것이며 이 job이 재확인하지 않는다.**"

> **X3P-6 (교차-잡 귀속의 비대칭)** "S층 불일치는 귀무대조 `907032→907100`(**S 0/32**, 노드·GPU·커밋 상이·무교란)이 받친다. **O층에는 귀무대조가 없다**(907032에 `phase_o` 부재) — O에서만 난 불일치는 노드/물리GPU/triton cache와 **구별되지 않는다**."

> **X3P-7 (승계)** "**R2C-1…16 · P-1…7 · X1P-1′…9 · X1C-10…14 · B3C-1…5 · OSP-1…6 · OSC-1…8 전부 불변.** 한 job의 4 boot은 독립 런이 아니므로 "n≥4/게이트 3 충족"을 쓰지 않는다(OSC-4/B3C-3 승계). **성능 수치 인용 전면 금지**(X1C-9 연장) — forced 샘플 OFF는 성능에 영향을 주는 설정이므로 지연·스루풋·GPU-h·decode step 수의 어떤 교차 비교도 금지한다. C층은 진단 전용(R2C-3). **Claim D/E 등급 불변 · 선결 #1·#2·#3·#4a·#5 불변 · #4b 부분(§8 문안) · HE0 · 정책 순위 · stake #1 불변.**"

---

## §13 인용 금지 초안 (X3C-1…8) — **문자 그대로 승계**

> **X3C-1** "X3가 **러너 설정을 인증했다 / eval 캠페인의 계측 구성에서 토큰 동치가 성립한다**." (관측자 2/4만 시험, 엔진 소스는 38c1aca 핀, 러너 코드 미실행. 허용형 = X3P-1.)

> **X3C-2** "이 job이 **서버가 러너의 인자로 뜬다는 것을 보였다**" 또는 `BOOT_FAILED`를 **A1의 ctx 유도 오류**로 귀속하는 것. (`--context-length 4096`은 `r2_correctness.sbatch:177` 리터럴이고 `pdmux_eval.context_limit`는 호출되지 않는다.)

> **X3C-3** "`0/96`은 **계측이 토큰에 무해함**을 보였다 / 관측자 효과가 없다." (양성대조 없음·S는 순차 bs=1로 산술 고정·probe prefill 단독 배치·타이밍 결합이 실측된 C층은 제외됨. 허용형 = X3P-4.)

> **X3C-4** "`VERDICT PASS`가 나왔으므로 **러너 구성에서도 게이트가 통과한다 / 선결 #2가 넓어졌다**." (O 확인은 **샘플 밀도의 운**이다 — 907100 타이밍 반사실에서 4 boot 중 **1개만** 8/8 확인됐다.)

> **X3C-5** "`NO_VERDICT_UNREALIZED`는 **게이트가 실패했다 / true-dual이 문제다**." (구성상 그렇게 되며, 무수정 규칙의 반사실 실행으로 사전 재현됐다. B7·B8·O_TIER는 `unrealized`로만 배선돼 있어 **계측 OFF는 FAIL을 만들 수 없다**.)

> **X3C-6** "X3가 **선결 #4b(observer effect)를 닫았다 / 절반을 닫았다**." (관측자 2/4·토큰 전용·cross-job·job n=1·타이밍 쪽 미측정. 허용형 = §8 문안.)

> **X3C-7** "O층에서만 난 불일치가 **러너 구성이 토큰을 바꾼다**는 증거다." (O 귀무대조가 존재하지 않는다 — 노드/물리GPU/triton cache와 모호.)

> **X3C-8** "`cross_job_compare.txt`의 `reading (F2)` 문단 / C층 해설 문장" 인용. (X1 전용이며 X3에서는 **0의 의미가 반대**다. D9.)

> **X3C-9** "이 job과 907100/907456 사이의 **어떤 지연·스루풋·GPU-h·decode step 수 비교 수치도** 인용 금지." (X1C-9 연장, forced OFF는 성능 설정이므로 특히 강하게 적용.)

---

## §14 새 방법론 게이트 후보 (기존 #1–189와 대조함)

- **G-X3-1 (게이트 #1의 3번째 층, 신설)** **관측자를 끄는 실험은 "끈 관측자"와 "남긴 관측자"의 발화율을 함께 등록하라.** X3는 boot당 1263–2285건 발화하는 관측자를 끄고 2580–2617건 발화하는 관측자를 남긴다. "관측자 효과를 시험했다"는 서술은 **남긴 쪽의 발화율을 적지 않으면 과대**다.
- **G-X3-2 (게이트 #110/G-OS-1의 대칭 사례, 신설)** **"구성상 그렇게 된다"는 예보는 논증하지 말고 무수정 판정 규칙을 그 구성의 반사실 입력으로 실제 실행해 보여라.** 이번에 forced 레코드를 필터한 907100 입력으로 checker를 돌려 `NO_VERDICT_UNREALIZED`·`failures=[]`·실패 하위검사 O2 한정까지 선취했다. 등록이 추정한 기전("15–19건 중 하나가 창에 들어와야")은 **크기가 81% 적중으로 틀렸지만 결론은 맞았다** — 실행이 없었으면 둘을 구별할 수 없었다.
- **G-X3-3 (신설, #174·G-OS-4 인접)** **다른 처치를 위해 감사·강화된 분석 도구를 재사용할 때, 그 도구가 *인쇄하는 해석 문장*이 새 처치에서 부호가 반대인지 확인하라.** 아티팩트에 박히는 문장은 나중에 정본으로 새어 들어간다. 도구를 고칠 수 없으면(해시 유지) **인용 금지 문장으로 중화**하라.
- **G-X3-4 (신설)** **귀무대조의 적용 범위를 층별로 적어라.** `907032→907100`은 **S층 전용** 귀무대조다(907032에 O층 부재). "교차-잡 불일치는 귀무대조로 처치에 귀속된다"는 문장은 O층에 대해 **거짓**이다.
- **G-X3-5 (신설, 우선순위 규율)** **"트랙을 해제한다"는 주장은 그 트랙의 블로커 목록을 열거하고 각각 제거되는지 대조한 뒤에만 쓰라.** 내 OS §8은 X3가 성능 트랙을 연다고 적었으나 블로커 3개(ctx·oracle·profile 산출물) 중 **0개**가 제거된다. 부수로 정본의 "약 270 run 불가"는 중복 계수였다(**고유 225 / 가능 180**).

---

## §15 단일 문장 요약

**`GO-with-caveats`.** 등록의 기전 수치는 내 독립 재계산과 **전부 일치**하고(97.0%·15–19건·전부 idx 4 — 후자는 등록 스크립트가 계산하지 않는 주장인데 참이다), 예보 F3·F4는 **무수정 판정 규칙을 907100의 forced-필터 입력으로 실제 실행해 `NO_VERDICT_UNREALIZED`(failures=[] infra=[], 실패는 O2 한정, B7·B8 통과)로 재현**되며, 묶음은 **greedy argmax·fcfs·고정 warmup ids·post-load seeding**으로 seed→토큰 경로가 없어 **confound #10이 아니다**; 엔진 핀은 `sha256sum -c`로 **충분함이 실측**됐다(핀 전 13 OK/4 FAILED = 핀 대상 4개) — 자유 표면 14개를 끝까지 밀었으나 **등록 라벨을 뒤집은 것은 0건**이고 반전을 만들려 한 2건(index-0 귀속, F3 기전)은 **실패해 caveat로 전환**됐으므로 N1–N4 어느 것도 발화하지 않는다. 그러나 제출 전 열셋을 고쳐야 한다: ★**F1은 측정하지 않은 것을 측정했다고 말한다**(이 job은 러너 코드를 한 줄도 실행하지 않고 ctx 4096은 `:177` 리터럴이며, 실패를 A1 탓으로 돌리는 분기는 오배선이다); ★**F2의 "eval 캠페인 계측 구성으로 이전된다"는 3중 과대**다(관측자 2/4만 시험 — 게다가 **남긴 쪽이 boot당 더 자주 발화**(2580–2617 vs 1263–2285) · 엔진 소스가 38c1aca로 핀되는데 캠페인은 HEAD를 돈다(4파일 +564줄, `dual_worker.py` +180 = TD hot loop) · 타이밍→토큰 결합이 실측된 C층은 제외됨(L-L 4/32)); ★**`PASS`가 무라벨**이고 그 확률은 907100 타이밍의 0.13–0.39%로 **상한되지 않는다**(forced 제거가 overlap 구간 밀도를 올릴 수 있고 자연 산포가 +32.6%); O층에는 **귀무대조가 없어** O-only 불일치를 처치에 귀속할 수 없으며; 재사용 비교기는 아티팩트에 **"mismatch==0 → sensitivity not demonstrated"**를 찍어 0의 의미를 반대로 박는다. 그리고 ★**내 OS §8의 "X3는 성능 트랙 전체를 연다"를 철회한다** — 캠페인 블로커 3개(W2/W4/W5 ctx·B2/B8 oracle·B6 hybrid profile 산출물 **저장소 전체 0건**) 중 X3가 제거하는 것은 **0개**이고(고유 차단 **225**/가능 **180**, 정본의 "약 270"은 중복 계수다) 캠페인은 다른 엔진 소스를 돈다 ⇒ **캠페인 설계 결정(모델/워크로드/profile)이 X3보다 먼저다**; 그래도 0.16을 쓴다면 X3 > OS는 유지하되 그 가치는 "트랙 해제"가 아니라 **"907100의 PASS가 자기 계측의 인공물이 아니다"** 하나다.

---

### 관련 파일 (절대경로)

- 심사 대상: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/x3_prereg/PREREG_X3_RUNNERCONF_2026-09-13.md` (`cf887161…`) · `.../x3_prereg/forced_sample_dependency.py` (`0899e29f…`) · `.../x3_prereg/forced_sample_dependency_stdout.txt` (`d6a5b6f7…`) — **셋 다 untracked(D11)**
- 재사용 분석 코드(무수정 유지 요구): `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/x1_prereg/x1_cross_job_compare.py` (`9ba6b230…`) — 결정적 줄 `:36`(`EXPECTED_PAIRS`), `:96-98`(입력 검사), `:121-135`(**D9 대상 인쇄 문안**), `:160-161`(**D9 대상 C층 문장**)
- 판정 규칙(무수정): `.../r2_correctness_check.py` (`ec355e17…`) — `:414-420`(B7·B8), `:421-422`(O_TIER), `:433-435`(**unrealized 전용 배선 = FAIL 불가**), `:258-314`(O1–O4), `:471`(`s_ref_ok`), `:480-499`(판정 우선순위)
- 하네스(미커밋, 현재 `9d79c713…`): `.../r2_correctness.sbatch` — `:94-105`(신규 노브 문서), `:127-131`(**D12 대상 주석**), `:154-166`(provenance·`*_prereg` 해시), `:177`(**`--context-length 4096` 리터럴 = D1 근거**), `:227`(`PDMUX_TRACE_FORCE_PREFILL="$TFP"`), `:255-260`(교차-잡 진단)
- 클라이언트: `.../r2_correctness_client.py` (`95e10b49…`) — `:145-146`(`temperature 0.0`), `:229-241`(**bg 첫 토큰 대기 = D4 근거**), `:296-304`(S 순차 루프)
- 엔진: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/src/multiplex/multiplexing_mixin.py` `:99-108`(TFP 기본 OFF)·`:125`(green readout, startup-only)·`:542-566`(4.9%/0.07% 주석)·`:570-581`(scheduled/forced 분기)·`:606-612`(**`trace_forced` 필드는 force 모드에서만 = D8 근거**)·`:1170-1176`(**`forward_count = max(1, 65536//extend_num_tokens)` = D4 근거**)
- seed 경로 근거(sglang dev tree, `/scratch/ehmoon/whlee/sglang_engine_dev/python/`): `sglang/srt/layers/sampler.py:105-107` · `sglang/srt/sampling/sampling_params.py:98-100` · `sglang/srt/sampling/sampling_batch_info.py:176` · `sglang/srt/managers/schedule_policy.py:297-300` · `sglang/srt/server_args.py:351` · `sglang/srt/managers/tp_worker.py:310-317` · `sglang/srt/entrypoints/http_server.py:1793-1804` · `sglang/srt/utils/common.py:684-690`
- 러너(주장의 과녁 — D11(e) 대상): `.../scripts/r2_eval/engine_bench_runner.sh` `:30-50`(서버 튜플)·`:62`(`server_args.txt`) · `.../scripts/r2_eval/r2_eval.sbatch` `:91-105`(ctx 프리플라이트·telemetry) · `.../scripts/r2_eval/policy_adapter.sh` `:29-58`(arm 배선, `:57` B6 unbound) · `.../benchmarks/pdmux_eval/context_limit.py` · `.../benchmarks/pdmux_eval/campaign.py` `:79-107`(9 baseline·`max_running_requests=48`·`server_seed=1000+rep`) · `.../benchmarks/pdmux_eval/workloads.py` `:148,159,180`(8192-토큰 W2/W4/W5)
- 원자료: `.../r2_correctness/job_907100/`(반사실 실행의 입력) · `.../job_907456/` · `.../job_907032/`
- 직전 판정서: `.../os_prereg/VERDICT_os_rules_2026-09-12.md`(§8 표의 "X3는 성능 트랙을 연다" — **이 판정서 §10.1에서 철회**) · `.../b3_prereg/VERDICT_b3_rules_2026-09-12.md` · `.../audit_x1_2026-09-12/VERDICT.md`(§2.4 probe decode split 수 bs-무관 — **상속 표시, 내 판정의 하중재 아님**)
- 정본(정정 대상): `/scratch/ehmoon/whlee/prefill-layer-alloc/PROJECT_STATUS.md` 최상단 배너(2026-09-12(5)) "약 270 run" → **고유 225 / 가능 180** · `/scratch/ehmoon/whlee/prefill-layer-alloc/reports/paper/EXPERIMENT_ROADMAP.md` `:845-853` 동일 수치 · `/scratch/ehmoon/whlee/prefill-layer-alloc/reports/paper/CLAIM_EVIDENCE_MATRIX.md` Claim D 행(선결 #2의 "러너 설정" 항목은 X3가 닫지 않음)
