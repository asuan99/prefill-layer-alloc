# N3·5-a·5-b 공통 PD-mux config 후보와 영향 분석 (2026-10-04, engine-porter)

> **지위**: 설계 문서다. GPU 0 · 새 성능 판정 0 · Claim 등급 변경 0 · 사전등록 아님.
> - 이 문서의 어떤 수치도 성능 주장이 아니다. boot·메모리 수치는 기존 로그를 그대로 옮긴 것이다(출처 병기).
> - 입력: `handoff-report/VERDICT_n4_dsem_rules_2026-10-04.md`(N3 필수 조항 1, F1 정정), `handoff-report/experiment_dag_and_execution_2026-10-04.md`, `reports/system_design_anchor_2026-09-30.md` §2·§3·§8, `reports/AUDIT_slo_method_2026-10-04.md` P-2.
> - 산출물:
>   - YAML 초안 3개: `configs/pdmux_n3_A_agn.yml`, `configs/pdmux_n3_B_d92.yml`, `configs/pdmux_n3_Bp_d54.yml`. 후보 A는 새 파일이 없다. 기존 `benchmarks/configs/pdmux_r2.yml`을 바이트 그대로 쓴다.
>   - CPU 프로브: `probe_scan_reachability.py`와 출력 `probe_scan_reachability_2026-10-04.txt`. 기술 기록이며 gate가 아니다.
> - 봉인 대상(`results/r2_eval/e2_sticky_vessl/`, `tests/test_e2_vessl.py`, 그리고 E2 결정 경로인 `e2_sticky_prereg/e2_realized_mix.py`)은 읽기만 했고 수정하지 않았다.

## 0. 요약

| 후보 | 분할(`manual_divisions`) | 그룹 수 | R2 결정 격자(generic/hybrid) | fixed/sticky 가능 d | 권고 |
|---|---|---|---|---|---|
| **A** = `benchmarks/configs/pdmux_r2.yml` | (92,16)(84,24)(74,34)(64,44), 임계 전부 0 | 6 | {16,24,34,44} | 16,24,34,44 | **권고** |
| A-agn | A와 같은 분할, 임계 0/18/27/36 | 6 | {16,24,34,44} | 16,24,34,44 | agnostic arm을 채택할 때만 |
| B | A + (16,92), 임계 0/0/0/0/1000000 | 7 | {16,24,34,44} | + 92 (static 전용) | 격자 경계 검열 확인용(아래 D1) |
| B′ | A + (54,54), 임계 0/0/0/0/1000000 | 7 | {16,24,34,44} | + 54 (static 전용) | B보다 정보량이 큼. 단 VESSL 미probe |
| C = `pdmux_homog5.yml` 유지 | (92,16)(64,44)(16,92), 임계 0 | 5 | **{16,44}** | 16,44,92 | 비권고(N2·S0′ 격자와 불일치) |

**권고는 A다.** 근거는 §5에 있다. 어느 후보를 고르든 homog5가 아니면 λ0 VESSL D44 arm과의 동일성은 **측정으로 확인되지 않은** 상태다. 따라서 N3 필수 조항 1의 재측정 조항이 발동한다(§2).

## 1. config가 런타임에 닿는 경로 (코드 사실)

경로는 tracked mirror `src/multiplex/multiplexing_mixin.py` 기준이다(dev tree와 `cmp` 동일). `pdmux_context.py`와 `cuda_graph_runner.py`는 dev tree `sglang_engine_dev/python/sglang/srt/` 기준이다.

| # | 경로 | 위치 | config에 의존하는 방식 |
|---|---|---|---|
| C1 | 스트림 그룹 생성 | `pdmux_context.py:104` `initialize_stream_groups` | `[(N,0)] + manual_divisions + [(0,N)]`(`:127`). 인덱스는 행 순서로 정해진다. 그룹은 boot 때 전부 미리 만든다. |
| C2 | R2 결정 격자 | `multiplexing_mixin.py:250` `_build_r2_policy` | decode SM ∈ {16,24,34,44}인 분할만 `decode_states`가 된다. 비면 boot 거부(`:262`). generic의 긴급 상태는 108로 하드코딩돼 있다(`:271`). |
| C3 | 정책 목표 → 인덱스 | `:626` `_r2_decide_idx` | `decode_sms`가 정확히 같은 **첫** 인덱스를 고른다(`:682`). 없으면 경고만 하고 **현재 인덱스를 유지**한다(`:675`). |
| C4 | sticky 대상 | `:388`–`:455` | 같은 decode SM을 가진 green division 중 첫 번째를 고른다(`:448`). idx0과 마지막 인덱스는 제외한다. 없으면 boot 거부. |
| C5 | 임계 스캔 | `:1151` `adjust_stream_groups`, 스캔 `:1186`–`:1187` | 마지막으로 만족된 행을 고른다. 첫 행 임계보다 bs가 작으면 `stream_idx`가 바인딩되지 않는다(크래시 경로). |
| C6 | 비분할 폴백 | `:1200` | decode가 바쁘고 prefill이 없으면 `real_sm_group_num-1`, 즉 (0,108)로 간다. 인덱스는 그룹 수에서 계산되므로 config와 무관하다. |
| C7 | R2 컨트롤러 블록 | `:1256`, `:1314`–`:1316` | `r2_policy`가 있으면 prefill span마다 C3로 목표를 정한다. |
| C8 | cudagraph 캡처 | `cuda_graph_runner.py:812`, `:798` | **모든** 스트림 그룹(idx0과 (0,108) 포함)에서 capture bs 집합 전체를 캡처한다. 키는 `f"{stream_idx}_{bs}"`, 메모리 풀은 전역 공유다. |
| C9 | decode attention backend | `model_runner.py:1966` | 서버 인자 `--sm-group-num`(기본 8)만큼 만든다. YAML 그룹 수가 이 값보다 크면 캡처 시 인덱스 범위를 넘는다(코드 읽기). 그룹 6–7개는 기본값 안이므로 서버 인자를 바꿀 필요가 없다. |

**임계 스캔(C5)은 R2 정책이 설정된 런에서 도달하지 않는다.**
- 코드 읽기 근거:
  - prefill이 in-flight이고 decode가 비어 있지 않으며 `wait_prefill_kernel_done`이 아니면 C7 블록이 먼저 잡는다.
  - `adjust_stream_group` 플래그는 세운 반복 안에서 소비된다. 그 소비 경로(`elif`)에 들어갈 때는 (a) decode가 비어 있거나 → idx0, 또는 (b) prefill이 없다 → (0,108).
- 실측 정황:
  - λ0 VESSL 15셀 telemetry 358,583건(homog5)에서 실현 인덱스는 (0)·(2: 64,44)·(4: 0,108) 셋뿐이었다. 스캔으로만 갈 수 있는 idx3 (16,92)과 idx1은 0건이다.
  - 단 snapshot은 32 sync마다 1개라 짧은 구간을 못 본다(F8). 그래서 "0건"은 증명이 아니라 정황이다.
- CPU 프로브: `probe_scan_reachability.py`가 실제 `event_loop_pdmux`를 fake로 구동했다(5 seed, 후보 4개).
  - fixed D16/D44/D92와 generic arm의 `scan_hits`는 **전부 0**이다.
  - 정책이 없는 런은 653회 스캔에 도달했다.

이 사실에서 나오는 설계 함의 세 가지:
1. **fixed·generic·hybrid arm에서 임계 열은 inert다.** 임계값을 바꿔도 R2 arm의 행동은 바뀌지 않는다(코드 읽기 + fake 기준, GPU 미실측).
2. **임계가 의미를 갖는 것은 `PDMUX_R2_POLICY` 미설정 런뿐이다.** 그때는 임계표 자체가 정책이다(upstream "agnostic" 기제).
   - 임계가 전부 0이면 **마지막 행이 항상 선택된다.** homog5에서 정책 없는 런은 (16,92)에 앉는다(프로브 확인). 행 순서가 곧 arm이 되는 함정이다.
3. **fixed D가 config에 없으면 조용히 엉뚱한 arm이 된다.**
   - sticky ON이면 boot가 거부된다(C4). sticky OFF fixed는 경고만 남기고 현재 인덱스를 유지한다.
   - 프로브에서 A + fixed D92는 decode 3,475 step 전부가 idx0 (108,0) 또는 (0,108)에서 돌았다.
   - ⇒ **러너 preflight에 "FIXED_DSM ∈ 분할 decode SM 집합" 검사가 필수다.** λ0 러너에는 이 검사가 없다(homog5에 44가 있어서 문제가 드러나지 않았을 뿐이다).

부수 관찰(config와 무관, 5-b 설계 시 engine-porter 확인 항목):
- `GenericDynamicPolicy.decide`(`controller.py:425` 이하)는 압력이 없으면 `target = current_decode_sms`다.
- 현재 인덱스가 idx0(decode SM 0)이나 (0,108)이면 C3가 그 인덱스를 그대로 돌려준다. 그래서 generic의 실효 도달 집합에는 {16..44} 외에 **idx0(평면 스트림 위 decode)과 (0,108)**이 들어간다.
- 프로브에서 generic arm은 분할에 한 번도 진입하지 않았다. 단 fake에는 ITL이 없어 압력 신호가 모델링되지 않는다. **엔진 동역학이 아니다.**

## 2. 기존 결과와의 비교 가능성 — λ0 VESSL D44 arm vs 새 config의 D44 arm

λ0 VESSL(Job `pdmux-lambda0-vessl-1d47d3e-20261004084945`)은 homog5이고 D44 = idx2, 평면 decode = idx4다. 후보 A/B/B′에서는 D44 = idx4, 평면 decode = idx5(B/B′는 idx6)다.

| 경로 | 같은가 | 근거 |
|---|---|---|
| D44 분할 자체 (64 prefill / 44 decode green ctx) | 같다(목표 기준) | 행 값 동일. KISTI 907959(pdmux_r2.yml) green read-out이 6그룹 전부 목표 smCount와 일치했다. VESSL에서 6그룹 boot read-out은 **미측정**이다(B3a는 분할을 개별 생성해서만 확인). |
| fixed D44 목표 → 인덱스 (C3) | 같은 분할로 해석된다 | decode_sms 정확 일치. 44는 각 후보에 1회만 있다. |
| sticky OFF 폴백 (C6) | 같다 | 마지막 인덱스 = (0,108). 인덱스 번호만 다르다. |
| 임계 스캔 (C5) | R2 arm에서는 도달하지 않음 | §1(코드·telemetry 정황·fake). |
| 다른 그룹의 존재가 런타임 경로를 바꾸는가 | 결정 경로에서는 아니다 | 그룹은 미리 생성되고, 전환은 인덱싱이다. 비용은 드레인이며 그룹 수와 무관하다. |
| cudagraph 캡처 | **다르다** | 그룹 수만큼 캡처한다. 풀 배치와 캡처 순서(D44 그래프가 3번째 → 5번째)가 바뀐다. 성능 효과는 미측정이다. "그래프 바이트 동일"은 주장하지 않는다. |
| 메모리 | **다르다(작음)** | 아래 표. |
| decode attention backend (C9) | 같다 | 그룹 수와 무관하게 `--sm-group-num` 8개를 만든다. |
| telemetry `stream_index` 의미 | **다르다** | D44 인덱스 2→4. `e2_realized_mix.py`(`D44_STREAM_INDEX = 2`, `PLAIN_UNPARTITIONED_INDEX = 4`, `GUARD_ROW_INDEX = 3` 하드코딩)는 A/B에 쓸 수 없다. 교차검사가 sm 불일치로 걸린다. 봉인 파일이므로 고치지 않는다. N3용 **(prefill_sms, decode_sms) 쌍 키 추정량을 새로** 써야 한다. |

boot·메모리 실측(기존 로그, 기판 표기):

| config (그룹) | 기판·Job | 캡처 시간 | 캡처 메모리 | 캡처 후 가용 | KV `#tokens` | 비고 |
|---|---|---|---|---|---|---|
| homog5 (5) | VESSL λ0, warm 14 boot | 8.35–9.75 s | 1.07 GB | 11.79 GB | 2,722,025 | 첫 boot는 50.25 s(JIT cold). boot→ready warm 33–37 s, 첫 boot 97 s. |
| homog5 (5) | KISTI 908623 `srv_a_r1.log` | 6.88 s | 1.07 GB | 11.79 GB | 2,722,025 | |
| pdmux_r2 = A (6) | KISTI 907959 `srv_L1.log` | 8.76 s | 1.27 GB | 11.59 GB | 2,721,290 | warmup boot 30.20 s(cold) |
| B/B′ (7) | — | **미측정** | 외삽 ≈1.47 GB | 외삽 ≈11.39 GB | 외삽 ≈2.7205M | 외삽 근거: 그룹당 증분(아래) |

- 그룹당 증분(VESSL `srv_b_r1.log` tqdm 진행 표시): 캡처 메모리 0.21–0.22 GB/그룹. warm 시간은 첫 그룹 약 3.6 s, 이후 그룹당 약 1–1.4 s다. KISTI 5→6그룹 차이는 +0.20 GB, +1.9 s로 같은 크기다.
- **KV와의 관계**:
  - 로그 순서가 `Memory pool end` → `Capture cuda graph begin`이다. KV 풀은 `mem_fraction_static 0.82`로 캡처 **전에** 정해진다. 캡처 증분은 KV 토큰 수를 줄이지 않고 잔여(활성화 여유)에서 나간다.
  - green ctx 생성은 weight load보다 앞선다(`PD-Multiplexing enabled` 로그가 load 앞에 찍힘). 그래서 weight 뒤 가용이 62.12 → 62.10 GB로 줄고 KV가 −735 토큰(−0.027%) 줄었다. 단 KISTI 두 Job 간 비교라 Job 간 잡음과 분리되지 않는다.
  - 이 구성에서 용량 구속은 `max_mamba_cache_size = 48`(= `--max-running-requests 48`)이다. KV 2.7M 토큰은 구속이 아니다(908623 E3 계열 정본 서술).
  - 캡처 후 가용 11.4–11.8 GB는 ctx 16384, `max_prefill_tokens 16384` 활성화에 대해 여유가 크다. 단 N2의 L>65536 셀(다른 `--context-length`)에서는 이 여유가 구속이 될 수 있다. 별도로 확인해야 한다.
- **예산 영향**: λ0 러너의 boot 허용치 `VESSL_BOOT_ALLOW_S = 240 s` 대비 그룹 +1–2개는 boot당 +2–4 s(외삽)다. 등록 예산을 바꿀 크기가 아니다.

**판정(설계 해석, 측정 아님)**: fixed D44 sticky OFF arm은 코드 읽기와 정황상 **같은 분할·같은 결정 경로**다. 다만 (i) 캡처 집합·메모리 잔여·KV 토큰 수가 다르고, (ii) VESSL에서 6/7그룹 boot가 한 번도 측정되지 않았다. 그래서 "같은 arm"은 **측정으로 확인되지 않았다**. ⇒ **N3 필수 조항 1의 재측정 조항이 필요하다.** 권고 형태는 §6-B의 "config 다리 블록"이다. C(homog5 유지)를 고르면 이 조항이 필요 없는 대신 §4 리스크가 생긴다.

## 3. `manual_divisions` 임계(decode-bs threshold) 설계

| 경로 | 임계의 의미 |
|---|---|
| fixed(sticky ON/OFF) | **무관**(도달 안 함, §1). sticky ON은 C4가 인덱스를 고정한다. |
| generic / hybrid | **무관**(C3가 목표를 정한다). 격자는 C2 필터가 정한다. |
| 정책 미설정(legacy upstream "agnostic") | **임계표가 곧 정책**이다. prefill이 in-flight인 동안 decode bs로 행을 고른다. |

- 후보 A(임계 전부 0):
  - 정책 없는 런은 항상 마지막 행 (64,44)이다. 사실상 "D44 static, R2 컨트롤러 없음"이 된다.
  - 그 런은 `_slo_on`이 False라 controller 블록·ITL EMA 경로가 아예 돌지 않는다. 즉 fixed D44 arm과 **다른 코드 경로**다.
- A-agn(임계 0/18/27/36):
  - upstream 자동 경로 `idx = clip(bs·(G−2)//36, 1, G−2)`를 A의 4행에 그대로 옮긴 값이다(G = 6, `decode_bs_divisor 36`). D16은 bs 1–17, D24는 18–26, D34는 27–35, D44는 ≥36이다.
  - 첫 행 임계는 반드시 0이어야 한다(C5 크래시 경로).
  - ★**이것은 P1.7 agnostic arm이 아니다.** P1.7은 `pdmux_a100_smoke.yml` 자동 격자(`divide_sm(108, cc8, 2)` = (74,34),(54,54))였다. 이름은 "agnostic-on-A"로 쓰고, 기존 측정 arm과 같은 기제라는 주장(교훈 259/261)은 하지 않는다.
- B/B′ 추가 행의 임계 = 1000000(센티널):
  - 추가 행은 정책 없는 런이 스캔으로 도달하지 못하게 둔다. fixed D92/D54로만 도달한다.
  - ★처음 초안의 49(= max_running + 1)는 **안전하지 않았다**. 48 상한을 강제하지 않는 fake 루프에서 그 행에 도달했다(프로브 1차 실행, B_d92 no-policy → (16,92) 2,188 step). 센티널은 `max_running`에 기대지 않는 값으로 둔다.
- 임계 열은 config 바이트를 바꾼다. 그래서 **N3 등록 전에 정해야 한다**. 나중에 agnostic arm을 추가하려고 임계를 바꾸면 config digest가 달라지고, N3↔5-a/5-b "같은 config" 조항이 깨진다. 이것이 결정 변수 D2다.

## 4. 후보별 리스크·열린 질문

**A (`pdmux_r2.yml`)**
- R2 결정 격자 = N2(3-a) d 격자 = S0′ 예측기 `allowed_decode_sms` = {16,24,34,44}다. 사다리 전체가 한 config 위에 선다.
- 기존 provenance:
  - KISTI R2 correctness 계열(907959 등)이 이 파일로 boot했다. green read-out 6/6 목표 일치, cudagraph ON이었다.
  - CPU fake(`tests/pdmux_loop_fakes.py`)의 정본 상태 집합이다.
  - CLAUDE.md 런타임 모드 문서와 `r2_eval.sbatch`가 이 파일을 쓴다.
- 리스크:
  - (a) VESSL 6그룹 boot는 미측정이다. KISTI는 성공했고 VESSL B3a는 4분할을 개별 생성했다. 첫 boot에서 green read-out 6그룹 전부 일치를 preflight gate로 둔다.
  - (b) λ0 VESSL·E2(N1, homog5, 실행 중)와 config가 다르다. 재측정 조항이 필요하다.
  - (c) N1이 검증하는 sticky는 homog5 idx2에서다. A의 idx4로 넘어가는 것은 코드 읽기(C4)뿐이다. N2는 셀별 실현 분할 검사를 어차피 해야 한다(DAG §1-3).
  - (d) 5-a best static이 격자 상단 D44에서 검열될 수 있다. HE0의 best static이 d44 = 당시 격자 상단이었다. A는 그 검열을 확인할 점이 없다.
- 열린 질문: 5-a의 "최적 static"을 정책이 도달 가능한 집합 {16..44} 안으로 정의할 것인가. A에서는 정의상 그렇게 된다.

**A-agn**
- R2 arm에 대해서는 A와 행동이 같다(§1).
- 리스크:
  - 임계값이 "의미 있어 보이는" 숫자라 독자가 R2 arm에도 작용한다고 오독할 수 있다. 등록 문서에 inert라고 명시해야 한다.
  - `pdmux_r2.yml`과 바이트가 달라 KISTI provenance를 그대로 승계하지 못한다. 분할 행은 같으므로 green·캡처 증거는 승계된다.
- agnostic-on-A arm은 `_slo_on` False 경로라 telemetry 이벤트 집합이 R2 arm과 다르다(`controller_decision` 없음). 관측 효과 대칭을 따로 확인해야 한다.

**B (+(16,92))**
- (16,92)는 VESSL B3a에서 probe됐고 homog5 선례가 있다.
- 리스크:
  - (a) prefill 16 SM이라 shape B(8192-token prefill)에서는 TTFT가 포화할 가능성이 크다. 검열 확인용으로는 D44와 너무 멀다(예보, 미측정).
  - (b) **static 전용 행**이라 정책 arm은 도달할 수 없다. 5-a의 "선택기 상한"을 이 행까지 포함해 계산하면 정책이 닿을 수 없는 상한이 된다. 상한 계산 집합을 {16..44}로 한정하고, D92는 "경계 검열 점검"으로 따로 보고해야 한다.
  - (c) 7그룹 boot·캡처는 VESSL과 KISTI 모두 미측정이다.
- 열린 질문: 이 점이 필요한 워크로드가 shape A(decode-heavy)뿐인가.

**B′ (+(54,54))**
- D44 바로 위 점이라 경계 검열 확인에 B보다 정보량이 크다. KISTI 선례: `pdmux_e1_d54.yml`, `slo_sched/pdmux_slo.yml`.
- 리스크:
  - (a) 54/54는 **VESSL에서 생성된 적이 없다**. B3-gran 패턴(최소 4, 짝수 단위)상 실현 가능해 보이지만 미측정이다. boot별 green read-out으로 54/54를 확인하는 것을 gate로 둔다.
  - (b)·(c)는 B와 같다.

**C (homog5 유지)**
- 장점:
  - λ0 VESSL·E2와 바이트 동일하다. 재측정 조항이 필요 없다.
  - boot 실측이 VESSL에 있다(B4, λ0 15 boot).
- 리스크:
  - (a) **R2 격자가 {16,44}뿐이다.** generic/hybrid는 16·44·108(그리고 §1 부수 관찰의 idx0)만 고를 수 있다. 주장 조건 1("다른 decode 최소 SM")의 해상도가 2점으로 줄어든다.
  - (b) N2(3-a) d 격자 {16,24,34,44}에는 (84,24)(74,34)가 필요하다. 그래서 N2는 다른 config를 써야 하고, 사다리 공통 config가 깨진다. S0′ 예측기 입력 격자(3-a)와 런타임 격자(5-b)가 달라진다.
  - (c) 정책 없는 런이 (16,92)에 앉는 행 순서 함정이 있다(§1).
- 판정 해석: C는 비교 가능성 하나를 위해 사다리의 주 질문 해상도를 포기한다. 권고하지 않는다.

## 5. 권고: 후보 A (`benchmarks/configs/pdmux_r2.yml`, sha256 `454125c3e6998c267e5b8028197b80363d601ca81c92bd3d7dd94911bc47eca7`, 바이트 그대로)

근거:
1. **격자 일치**: R2 결정 격자, N2 3-a 격자, S0′ `allowed_decode_sms`가 모두 {16,24,34,44}로 하나다. N3 필수 조항 1("5-a/5-b와 같은 `sm_counts`")을 N2까지 포함해 한 config로 충족한다.
2. **추가 비용이 최소다**: homog5 대비 +1그룹이다. KISTI 실측 증분은 +0.20 GB 캡처 메모리, +1.9 s 캡처, KV −0.027%(비구속)다.
3. **기존 provenance**: 이 바이트로 boot·green read-out·cudagraph 캡처가 된 KISTI 기록이 있고, CPU fake의 정본 상태 집합이다. 새 파일을 만들지 않으므로 digest 계보가 단순하다.
4. **임계 0은 새 의미를 만들지 않는다**: R2 arm에서 inert다. 정책 없는 런은 D44에 고정된다.

조건:
- agnostic arm을 5-a/5-b에 둘 것이면 A 대신 **A-agn**을 쓴다(D2). 이 결정은 N3 등록 전에 해야 한다.
- 경계 검열 점검이 필요하면 **B′**로 바꾼다(D1). 그때 static 전용 행 비대칭(§4 B (b))을 등록 문안에 넣는다.

## 6. 사용자 결정 변수 (이것만 정하면 된다)

| # | 결정 | 선택지 | 권고 | 무엇이 바뀌나 |
|---|---|---|---|---|
| D1 | 분할 집합 | A / B / B′ / C | **A** | 그룹 수, 격자, 재측정 조항 필요 여부(C만 불필요) |
| D2 | 정책 미설정 "agnostic-on-A" arm을 5-a/5-b에 둘지 | 예 → A-agn 임계 / 아니오 → 0 | 아니오(현재 5-b arm 채택은 미결이므로 필요가 생기면 지금 예로) | config 바이트. 나중에 바꾸면 N3 재측정 |
| D3 | N3 Job 안 config 다리 블록(λ0 기준 셀을 새 config로 재측정) | 예 / 아니오 | **예** | 아니오면 λ0 VESSL 값은 N3 옆에 "같은 config 기준"으로 인용 불가(참조 라벨로만) |
| D4 | floor 블록(P-2) 리터럴 | PD-mux 활성·decode 유휴 / PD-mux 비활성 | **활성·decode 유휴** | 비활성이면 서버 인자가 달라 boot가 따로 필요하고 "같은 서빙 config"가 깨진다 |
| D5 | `PDMUX_PHASE_EVENTS` | 전 셀 대칭 ON + 같은 Job OFF 대조 블록 / OFF | **ON + OFF 대조** | OFF면 요청별 노출(조항 4)과 floor 실현 분할 (108,0) 실증이 불가하다(snapshot 격자가 짧은 prefill을 못 봄, F8) |

D3 다리 블록 형태(제안):
- λ0 VESSL variance 블록과 같은 등록 seed로 `b_r3`(offered 1.15, N=200)와 `a_r4`(offered 8.0)를 각 2 seed씩 새 config로 돈다.
- λ0 VESSL n=4 분포 옆에 기술 통계로만 둔다. 판정이 아니라 "인용 자격" 조항이다. 허용 차이는 등록 시 상수로 둔다.
- 비용은 λ0 셀 소요 기준 약 15–20분이다(boot 약 35 s + 셀 134–290 s, ×4).

## 7. 선택과 무관하게 지금 쓸 수 있는 것 — N3 러너 재사용·신규 목록

**`lambda0_vessl.sh`·`lambda0_vessl_plan.py`에서 그대로 재사용**(구조 복사, 상수만 N3로 교체):
- `PDMUX_*` allowlist + unset 목록(`lambda0_vessl.sh:63` 이하). N3는 `PDMUX_PHASE_EVENTS`를 allowlist에 넣는다.
- 컨테이너 시작 기준 wall-clock 하드캡, watchdog, 셀별 need 기반 admission, tail reserve(`HARD_CAP_S`/`TAIL_RESERVE_S`/`VESSL_BOOT_ALLOW_S`).
- 등록 무결성: 사전등록·결정 경로 sha256, `REGISTRATION_SHA256.txt`, git HEAD, results 밖 미커밋 0 확인.
- 기판 기록 블록(nproc·affinity·cgroup `cpu.max`·mem·버전·nvidia-smi clock/power, `SUBSTRATE_LAM0V.txt` 생성부).
- 모델 revision 검사, `sync_engine_tree.sh` manifest(D12 ≥24 항목·`nemotron_h.py` 포함)와 기준 manifest diff. 기준은 λ0 VESSL이며 PHASE_EVENTS 1파일 차이를 공시한다(DAG F4).
- `run_cell`의 boot/health/warmup/bench/teardown, 포트 선택, `FAIL_STREAK` 2연속 boot 실패 중단, resume, 셀별 배너 grep, boot별 `PDMUX_GREEN_READOUT`.

**바꿔야 하는 것**:
- `CFG`/`CONFIG_REL`/`CONFIG_DIGEST`(`lambda0_vessl_plan.py` homog5 상수)를 D1 결과로 교체한다.
- rev5 결정 경로 digest 검사·plan 바이트 동일 검사(λ0 전용)를 N3 결정 경로 digest로 교체한다.
- plan: λ0 κ-래더를 **절대 req/s 격자**(DS-4, 조항 7)로 바꾼다. λ\*는 참조 라벨로만 쓴다. shape A에는 0.36–0.59×λ\* 사이 2점(E-β)을 절대 req/s로 넣는다.
- 분석기: `lambda0_analyze.py`(knee)를 정본 goodput 술어로 바꾼다(`benchmarks/pdmux_eval/analyze.py` 계열, 요청 TTFT ≤ T ∧ 요청 내 token-ITL p95 ≤ SLO). paired bootstrap, Holm(P-3), 결정점 k ∈ {3,5,10}.
- 실현 분할 추정량: `e2_realized_mix.py`는 인덱스 하드코딩 + 봉인이라 재사용할 수 없다. **(prefill_sms, decode_sms) 쌍 키**로 4추정량(E-cnt/E-time/E-iter/E-qcond)과 벽시계 인덱스 점유를 계산하는 새 파일을 쓴다. 기존 λ0 VESSL telemetry에서 E2 도구와 같은 값을 재현하는지 대조 테스트를 붙인다(교훈 53: 변이본에서 실패해야 함).

**새로 필요한 블록**:
1. **preflight 추가 검사**:
   - (a) `PDMUX_R2_FIXED_DSM` ∈ config 분할 decode SM 집합. 없으면 abort한다(§1 함의 3: sticky OFF fixed는 조용히 엉뚱한 arm이 됨).
   - (b) 첫 boot green read-out에서 모든 분할 smCount = 목표.
   - (c) `PDMUX_STICKY_PARTITION` 미설정, 그리고 read-out의 `is_sticky_target` 전부 false(조항 1: `sticky_target_index` null 실증).
   - (d) YAML 그룹 수 ≤ 서버 `--sm-group-num`(기본 8).
2. **floor 블록(A′ P-2, 같은 Job 첫 블록)**:
   - 같은 서버 config, `--max-concurrency 1`, warmup 뒤 서로 다른 프롬프트 ≥ n개, 송신→첫 토큰 TTFT의 중앙값.
   - 실현 분할 (108,0)은 PHASE_EVENTS `prefill_start`의 `(prefill_sms, decode_sms)`로 전수 실증한다. 하나라도 다르면 floor 무효로 abort한다.
   - 등록 공식 `T_k = k × floor`(k ∈ {3,5,10})를 **초 단위로 파일(`T_K.json`)에 기록한 뒤에** arm 셀을 시작한다. 측정 Job 안 floor 재측정은 진단으로만 쓰고 재채점하지 않는다.
   - 코드 근거: decode가 빈 상태에서 prefill을 admission하면 `elif` → `adjust_stream_groups` → idx0이다(`:1211`). concurrency 1이면 직전 요청의 decode가 끝난 뒤 다음 요청이 온다.
3. **PHASE_EVENTS 대칭 ON + OFF 대조 블록**: 전 셀 ON. 같은 Job 안에 OFF 셀 몇 개를 같은 rate·seed로 짝지어 두고, 관측 효과 < 3%와 CI가 0을 포함하는지를 보고한다(correctness gate 4와 같은 형식).
4. **정상성 S1–S3**(`ttft_measurement_definition_review_2026-09-28.md` §3.1):
   - S1: F-E 비율(임계 1.7).
   - S2: 큐 성장(전반/후반 위반율, 임계는 등록 시 고정).
   - S3: telemetry `prefill_queue_depth` 순위 기반 추세(임계는 등록 시 고정).
   - 크기는 STATIONARY 셀만 보고한다.
5. **요청별 노출**: PHASE_EVENTS로 분할 구간 토큰 비율을 계산한다(조항 4).
6. **셀 전후 기판 기록**: clk, `cpu.stat`, power cap 노출(조항 9). λ0는 1회 기록뿐이다.
7. **config 다리 블록**(D3 = 예일 때).
8. **구속 설정 병기**: cap 48 = mamba 48, B는 `max_prefill_tokens=16384`(조항 8). SLO 값·절벽 가드 상수는 사전등록 상수로 둔다(조항 5·7). retract 요청 제외(X-10).

## 8. 미측정 항목

- VESSL에서 6그룹(A)·7그룹(B/B′) 엔진 boot, green read-out, 캡처 시간·메모리. A는 KISTI만, B/B′는 어디에서도 측정되지 않았다.
- VESSL에서 (54,54) green context 생성.
- 그룹 수 증가가 D44 arm의 ITL/TTFT/처리율에 미치는 효과. 캡처 풀 배치·순서 변경 포함. 코드 경로상 기대값은 0이지만 측정은 없다 → D3.
- 6/7그룹에서 KV `#tokens` 변화의 Job 간 잡음 분리. −735 토큰은 KISTI 두 Job 간 1점 비교다.
- R2 arm에서 임계 스캔 비도달의 GPU 실측. 코드 읽기 + snapshot 정황 + CPU fake만 있다. PHASE_EVENTS ON 셀에서 "fixed D_k 셀의 분할 구간 중 D_k 외 분할 인덱스 0건"을 기술 검사로 둘 수 있다.
- generic의 idx0/(0,108) hold 동역학이 실제 엔진(ITL 신호 있음)에서 얼마나 일어나는지. config와 무관한 5-b 설계 항목이다.
- N2 L>65536 셀(다른 `--context-length`)에서 캡처 후 활성화 여유.

## 9. 재현

```bash
cd ~/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/n3_capacity
python3 probe_scan_reachability.py            # CPU, 수 초. 출력 = probe_scan_reachability_2026-10-04.txt
python3 -c "import yaml,sys;[print(f, yaml.safe_load(open(f))) for f in sys.argv[1:]]" configs/*.yml
```
