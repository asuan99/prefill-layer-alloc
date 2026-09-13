# 사전등록 — 새 (모델, 백엔드) 쌍의 R2 correctness 게이트 + 감사 지정 계측 3건

**rev3 — 제출 승인 상태** (2026-09-13, engine-porter, **GPU 지출 0**).

이력: rev1 → 규칙층 감사 `NO-GO`(死因 N3, `VERDICT_newpair_rules_2026-09-13.md`, 263행)
→ rev2에서 **D1–D13 반영** → 재감사 **`GO-with-caveats` · 제출 승인**(死因 0,
`VERDICT_newpair_rev2_2026-09-13.md`, 284행) → rev3에서 **비차단 권고 6건 반영 +
캐비앳 NPC-A…NPC-J 편입**. 변경점은 각 절의 **(D<n>)** / **(NPC-<x>)** 표지로 표시했다.

**제출은 아직 하지 않았다** — 커밋·제출은 메인 세션이 한다(§12-b). 이 문서와 함께 출하되는
실행체는 `../r2_correctness.sbatch`(수정) · `../../../scripts/bootstrap/sync_engine_tree.sh`
(수정, D10) · `newpair_preflight_o1.py`(+ 고정 stdout) 셋뿐이고,
**판정 코드 `../r2_correctness_check.py`는 한 바이트도 바뀌지 않았다**(sha `ec355e17…`,
`tests/test_r2_correctness_ctx.py`와 `tests/test_r2_correctness_instrument.py` 양쪽이 핀).

> **제출 전 상태다.** claims-auditor 규칙층 감사 전이며, sbatch는 제출되지 않았다.
> 이 트랙(R2 correctness) 누적 GPU는 이 문서 작성 시점에 **0.43 GPU-h**로 불변이다.

---

## 0-b. rev1 → rev2: D1–D13 반영 표 (재감사용 색인)

| D | 요구 | 상태 | 어디에 |
|---|---|---|---|
| **D1** (死因 N3) | F2/I1/NP-7 교체 — `48`은 연역이지 측정이 아님 | **반영** | §0 "계측 3건의 지분", §5-I1 전면 교체, §7 **F2′**, **NP-7′** |
| **D2** | 귀무대조 공허한 S FAIL → `NO_VERDICT_INFRA` 사전 규칙 | **반영, rev3에서 D2′로 축소** | §8 "(D2′)" 절, **NP-10**, **NPC-A**(TD 대칭 조항은 철회) |
| **D3** | FAIL 행에 TD-TD S 불일치 + 방어 논거 | **반영** | §8 FAIL 행 + "(D3)" 절, **NP-10** |
| **D4** | F5 셀별 재계산 레시피(배타/포함) + "미달=상류병목" 단정 금지 | **반영** | §5-I3 "(D4)" 코드 블록, §7 F5 |
| **D5** | 재제출 정책 | **반영** | §9 "(D5)" 절 |
| **D6** | I2 = 클라이언트 관측량(엔진 step 아님) · I1–I3 D44 한정 | **반영** | §5-I2 "(D6-i)(D6-ii)", **NP-9** |
| **D7** | "양 끝" → "위쪽 끝만" + NP-3 보강 | **반영** | §0, §5-I3 "(D7)", **NP-3′** |
| **D8** | warm-up 분할 실현 미확인 등록 (+선택 코드) | **반영 + 코드 채택** | §5-I1 "(D8)", §6, **NP-9**; 코드 = warm-up green read-out(가드) + 테스트 5건 |
| **D9** | 제출 전 커밋 | **미반영(의도)** | §12-b에 파일 목록 등재. 병행 워크스트림 2개가 미커밋이라 **메인 세션이 통합 시 일괄 처리**(코디네이터 지시) |
| **D10** | `flashinfer_backend.py`를 provenance에 | **반영(경로 (a))** | `sync_engine_tree.sh` manifest **25번째 줄**; §11. 추가로 flashinfer **휠 버전**을 provenance에 기록 |
| **D11** | 최악 예산 정정 + `--time` | **반영** | §9 "(D11)" 표; `#SBATCH --time=01:15:00 → 02:30:00`(지시자 수정), §13 경고 |
| **D12** | OOM 오귀속 차단 | **반영** | §9 실패 모드 (b) |
| **D13** | 스코프 배선 + NP-8 | **반영** | §12 "(D13-i)" 절, **NP-8** |

**1차 감사가 반증에 실패한 12건(1차 판정서 §6)은 손대지 않았다** — 특히 "추가 scored
boot 0 · 채점 경로 무영향"(§6)과 `Concurrency:` → `#running-req` 교체(§5-I3)는 그대로다.

### rev2 → rev3: 재감사 비차단 권고 6건

| # | 권고 | 상태 | 어디에 |
|---|---|---|---|
| 1 | **NPC-C**: 하네스 I1 주석이 D1이 철회한 문장을 들고 있다 | **반영** | `r2_correctness.sbatch` I1 주석 4줄 → NP-7′ 문안으로 교체(테스트 51/51 유지) |
| 2 | §8 (D2) → **D2′**(TD 대칭 조항 철회 + TD-TD 불일치 제외 조건) | **반영** | §8 "(D2′)", **NP-10**·**NPC-A** |
| 3 | **NPC-D**: "프로세스 0개 추가"는 provenance의 `import flashinfer` 때문에 거짓 | **반영** | §6 첫 bullet, **NPC-D** |
| 4 | **NPC-H**: 회귀 합격 기준을 `test_r2_correctness_*`로 한정 | **반영** | §13, **NPC-H** |
| 5 | **NPC-F**: manifest 파급에 비-테스트 소비자 2곳 등재 | **반영** | §11, **NPC-F** |
| 6 | **NPC-B 정정**: "flashinfer workspace가 처음 바뀌는 성분"은 사실 아님 + §12-b 계수 | **반영** | §5-I1·§7에서 삭제·정정, §12-b "4경로/7파일" |
| — | 캐비앳 **NPC-A…NPC-J** 10건을 §12에 편입 | **반영** | §12 "재감사(rev2) 캐비앳" 절 |

**λ0와의 모순 판정**: 재감사가 **이 문서 쪽(D7 / NP-3′)이 옳다**고 판정했다 — `λ_inf`는
λ\*를 **위에서만** 구속하므로 하한 계수는 λ_inf가 주는 정보가 아니라 추가 가정이다.
**이 문서는 수정하지 않는다.** λ0 워크스트림이 `lambda0_plan.py`의 `MULT["A"][0]`을
0.40 → 0.25로 내리고 데이터 확인 분기를 등록해 수용했다. 앵커의 arm 한정은 **NPC-I**로
λ0 쪽이 승계한다.

---

## 0. 왜 이것을 먼저 사는가 (그리고 왜 "correctness가 선결"이라는 통상 논거가 아닌가)

λ0(캠페인 0단계) 사전등록이 규칙층 감사에서 `NO-GO`를 받았고
(`../../r2_eval/lambda0_prereg/VERDICT_lambda0_rules_2026-09-13.md`), 그 판정서 §8이
**순서를 뒤집으라**고 권고했다. 그 권고의 논거는 "correctness가 실패하면 λ\* 측정이 낭비"가
**아니다** — 판정서가 스스로 그 논거를 기각했다(λ\*는 B1 = legacy + fixed split에서 재고,
job 905835가 이 모델·이 백엔드·이 arm을 12/12 boot시켰으므로 true-dual correctness를
선결로 요구하지 않는다). 실제 논거는 두 가지다:

1. **`--request-rate inf` 셀이 λ0의 死因 N2(도착 실현 계수)를 구조적으로 제거**한다 —
   `bench_serving.py:943-945`가 `request_rate == float("inf")`에서 sleep을 전부 건너뛰므로
   Poisson 실현 자체가 존재하지 않는다.
2. 그 셀들을 **추가 boot 0으로** correctness 게이트에 얹을 수 있다.

따라서 이 job이 사는 것은 **(a) 새 쌍의 correctness 판정 + (b) 다음 회차 λ\* 사다리의
위쪽 끝**이다. (a)와 (b)는 서로의 전제가 아니며, 한쪽이 실패해도 다른 쪽은 남는다(§9).

**(D1·D7) 계측 3건의 지분을 정직하게 다시 적는다.** rev1은 "계측 3건"을 등가로 제시했는데
그것은 틀렸다:

- **I1은 용량을 재지 않는다.** rev1이 I1의 목적으로 적었던 `max_mamba_cache_size = 48` 확인은
  **측정이 아니라 플래그 조합의 연역**이다(§5-I1, 死因 N3). I1이 실제로 재는 것은
  `max_total_num_tokens`·`available_gpu_mem`뿐이며, 그 예보는 F2′로 교체됐다.
- **I3는 사다리의 *위쪽 끝만* 고정한다.** 상한 프로브는 아래쪽을 구속하지 않는다(§5-I3).
- 즉 이 구매가 λ0 재등록에 주는 것은 "양 끝"이 아니라 **위쪽 끝 + shape A의 B=1 바닥
  (I2) + KV 여유 확인(F2′)**이다.

---

## 1. 스코프 튜플 (이 job이 인증하는 조건의 전부)

| 축 | 값 | 출처 |
|---|---|---|
| model | `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base` (`NemotronHForCausalLM`, 56층 = attention 4 / mamba 27 / MLP 25) | 사용자 결정(2026-09-13) |
| attention backend | **`flashinfer`** | 강제 — `server_args.py:1959` `assert self.attention_backend != "triton"` |
| context length | **16384** (131072 아님) | 사용자 결정 · 트레이스 최대 요구 8320 토큰의 2× 여유 |
| mem-fraction-static | 0.82 | `engine_bench_runner.sh:83` 기본값과 일치 |
| max-running-requests | 48 | `engine_bench_runner.sh:84` 캠페인 값 |
| split | fixed **D44** (`PDMUX_R2_POLICY=fixed`, `PDMUX_R2_FIXED_DSM=44`) | 운영점 |
| cudagraph | **ON** (decode 그래프), `--disable-cuda-graph` 없음 | 운영점(정본) |
| piecewise cudagraph | 플래그 없음 = 엔진이 **자동으로 끈다**(§10-3) | `--chunked-prefill-size -1` |
| 기타 서버 인자 | `--disable-radix-cache --chunked-prefill-size -1 --disable-overlap-schedule --decode-log-interval 1 --random-seed 1` | `server_cmd()` 무수정, 테스트가 바이트 단위 핀 |
| pdmux config | `benchmarks/configs/pdmux_r2.yml` | 무수정 |
| boot 순서 | `L TD L TD` (기본) | 무수정 |
| 클라이언트 | `r2_correctness_client.py` 무수정(seed 20260911, S16 / O8 / C32) | 무수정 |
| 판정 규칙 | **verdict rule v2**, `r2_correctness_check.py` 무수정 | §3 |
| 계측 | `R2C_INSTRUMENT=1` (§5) — **warm-up boot에만**, 채점 경로 비접촉 | §6 |
| 월타임 | `#SBATCH --time=02:30:00` (D11로 `01:15:00`에서 상향; 측정량 불변) | §9·§13 |

**하드웨어**: A100 108 SM, 1 GPU. **비교 금지**: 이 튜플의 어떤 축이라도 다르면 다른 실험이다.

---

## 2. 이 게이트가 인증하는 것 / **인증하지 않는 것**

**인증하는 것**(PASS일 때, 그리고 PASS일 때만):
> 위 튜플에서 `PDMUX_TRUE_DUAL_WORKER=1` 경로와 legacy 단일-스레드 PD-mux 루프가
> **같은 토큰 id를 낸다** — S층(16 요청, 한 번에 하나)과 O층(8 probe, 각각 배경 요청이
> decode 중일 때 보내져 probe prefill이 D44 분할 위에서 그 decode와 동시에 도는 프로토콜)
> 에서, 그리고 그동안 cudagraph가 두 arm 모두에서 켜진 채 유지된다는 것.

**인증하지 않는 것**(PASS여도 쓸 수 없는 문장):

- **성능이 아니다.** TTFT·ITL·goodput·throughput 어느 것도 이 게이트의 판정량이 아니다.
- **Claim D가 아니다.** Claim D는 P2 캠페인(B1 vs B4, W3+W4)의 결과이고 이 게이트는 그
  전제 조건 하나일 뿐이다. 이 job이 끝나도 Claim D 등급은 **미검증** 그대로다.
- **다른 split이 아니다.** D16/D24/D34/D92는 포함되지 않는다.
- **다른 정책이 아니다.** generic·hybrid·SLO-aware 어느 것도 돌지 않는다(fixed만).
- **다른 모델이 아니다.** Zamba2-2.7B / Falcon-H1 / Granite-4는 포함되지 않는다.
- **다른 백엔드가 아니다.** triton 결과(907100·907456·X1)는 (Zamba2-2.7B, triton) 한정으로
  동결돼 있고 이 job이 그 동결을 풀지 않는다(역방향도 마찬가지).
- **sampling이 아니다.** greedy(T=0, `ignore_eos`)만.
- **O 프로토콜 밖의 동시성이 아니다.** C층(32 동시)은 **진단 전용**이다(§4).
- **λ\*가 아니다.** §5의 I3는 상한 프로브이며 λ\*의 대체가 아니다(§5-3).

---

## 3. verdict rule v2 승계 — flashinfer에서 규칙 문안을 고쳐야 하는가

**결론: 규칙 문안도 코드도 고치지 않는다. 단 O1 조항의 *정당화 근거* 하나가 이 쌍에서는
성립하지 않으며, 그 사실을 여기 등록한다.**

### 3.1 `--triton-attention-num-kv-splits` 경로는 무해하게 살아 있다 (코드 확인)

- X1이 쓴 **CLI 노브**는 이 job에서 **설정되지 않는다**: `R2C_NUM_KV_SPLITS`가 미설정이면
  `${NKVS:+--triton-attention-num-kv-splits $NKVS}`가 **빈 문자열로 확장**되어 플래그 자체가
  명령줄에 없다. flashinfer 부팅에 triton 전용 플래그가 붙는 일은 일어나지 않는다.
- 그러나 **서버 인자 자체는 백엔드와 무관하게 존재**한다
  (`server_args.py:648 triton_attention_num_kv_splits: int = 8`), 그러므로
  - `SERVER_ARG_KEYS` / `SAME_ACROSS_BOOTS`의 `triton_attention_num_kv_splits` 항목은
    4 boot 모두에서 `8`로 읽히고 **boot 간 동일성 검사를 정상적으로 통과**한다(무해).
  - 체커의 `kv_splits = int(sa["triton_attention_num_kv_splits"])`도 `8`이 되어
    **O1 산식이 그대로 계산된다**: `probe.prompt_tokens > 7 × (bg.prompt_tokens + 160)`.

### 3.2 그 산식이 새 토크나이저에서 만족되는가 — 사전 측정했다 (GPU 0)

`newpair_preflight_o1.py`(출력 고정 `newpair_preflight_o1_stdout.txt`)가
**출하된 클라이언트 자신의 `plan()`**으로 프롬프트를 만들고 실제 체크포인트 토크나이저로
센 결과:

| | Nano-9B-v2-Base (새 쌍) | Zamba2-2.7B (참조) |
|---|---|---|
| bg prompt tokens | 17 | 17 |
| O1 문턱 `(K−1)(bg+160)` | **1239** | 1239 |
| 최소 probe(O00) | **1280** | 1437 |
| 최악 여유 | **+41 토큰 (+3.3%)** | +198 (+16.0%) |
| bg가 몇 토큰까지 커져도 되는가 | **22** (실측 17) | 45 |
| 최장 요청(in+out) | 2358 ≤ ctx 16384 | 2667 ≤ 16384 |

★**이 사전 계산 자체의 양성대조**: 같은 계산을 참조 쌍에 적용하면 job 907456이 **서버에서
보고한** `prompt_tokens`와 **8/8 probe + bg 전부 일치**한다. 즉 새 쌍의 수치는 독립적인
추측이 아니라 서버가 보고할 값의 예측이다.

⇒ **O1은 8/8 probe에서 성립할 것으로 예보한다(F3).** 여유가 3.3%로 좁다는 사실을
**등록**한다 — 토크나이저가 바뀌면 이 조항이 조용히 죽을 수 있는 자리이며, 그 경우 판정은
`NO_VERDICT_UNREALIZED`(측정 실패)이지 `FAIL`이 아니다.

### 3.3 ★고치지 않지만 반드시 등록해야 할 것: O1의 *근거*는 이식되지 않는다

O1의 부등식은 원래 **triton decode 커널의 KV-split 선택**(`get_num_kv_splits_triton`)이
배치의 max/min 시퀀스 길이에 의존한다는 사실에서 유도됐다 — probe가 충분히 길면 probe 행의
split 수가 배경 요청의 진행 정도와 무관해진다는 논증이다. **flashinfer에는 이 논증이 없다.**

코드 확인:

- `layers/attention/flashinfer_backend.py:185-199` — `prefill_split_tile_size`·
  `decode_split_tile_size`·`disable_cuda_graph_kv_split`는 **`--enable-deterministic-inference`
  일 때만** 고정된다. 이 job은 그 플래그를 쓰지 **않는다**(캠페인도 쓰지 않는다).
- `:589-590`, `:711-712` — cudagraph 경로의 decode plan은 `fixed_split_size=None`,
  `disable_split_kv=False`로 호출된다 ⇒ FlashInfer의 **적응적 split-KV 스케줄러**가
  배치 전체의 KV 작업량으로 split 수(=감소 순서)를 정한다.

⇒ **이 쌍에서 O층 동치의 신뢰는 커널 논증이 아니라 rule v2가 이미 요구하는 *arm-내 귀무대조*
(L1–L2 O쌍 동일 ∧ TD1–TD2 O쌍 동일)에 전적으로 의존한다.** 규칙은 이미 그것을 요구하므로
(규칙선 1과 3) **문안 수정은 불필요**하다. 대신 아래 두 문장을 등록한다:

> (i) arm-내 O쌍이 **다르면** 판정은 `INCONCLUSIVE`이고, 이는 true-dual의 결함이 아니라
> 이 쌍의 **측정 가능성**에 대한 사실이다(flashinfer 적응 split-KV + mamba 층 구성).
> (ii) arm-내 O쌍이 **같은데** L-vs-TD만 다르면 rule v2는 `FAIL`을 낸다 — 그때에도
> "true-dual이 틀렸다"는 **귀속**은 이 job만으로 성립하지 않는다. 별도 회차(예: 동일 arm
> 반복 4 boot, 또는 `--enable-deterministic-inference` 대조)가 필요하며 그것은 여기 등록되지
> 않았다. 완화 사정: 이 모델은 56층 중 attention이 **4층**뿐이라 flashinfer 감소순서
> 민감도의 노출이 작다 — 그러나 이는 사전 기대이지 측정이 아니다.

### 3.4 그 밖의 규칙 조항은 그대로 성립한다

- B3(cudagraph 전수 True): `--decode-log-interval 1`이 백엔드와 무관하게 매 decode step을
  찍는다. 변경 없음.
- B8 / green read-out: SM 분할 실현 확인은 green-context 드라이버 read-out이라 백엔드와 무관.
- B5(policy=fixed, admission 사건 0): 정책 축 변경 없음.
- O4(TD 전용 스레드 조건): 아키텍처 축 변경 없음.
- 규칙 사후선택 아님: rule v2는 2026-09-11에 고정됐고, **이 쌍에서는 어떤 boot도 출력을
  낸 적이 없다**(job 905835는 legacy + fixed만 돌렸고 `PDMUX_TRUE_DUAL_WORKER`는 이 모델에서
  **한 번도 실행된 적이 없다**).

---

## 4. C층 스코프 승계

C층(32 동시 요청)은 **진단 전용**이며 판정에 들어가지 않는다. 근거는 승계된 사실이다:
job 907032에서 **legacy-vs-legacy가 7/32 불일치**했으므로 사전 선언된 귀무대조 조건
("교차-arm 불일치는 legacy-vs-legacy가 동일할 때만 센다")이 C에서는 **구성상 성립 불가**다.

★**새 쌍에서 그 7/32는 미측정이다.** 이 job의 C층 불일치 수가 0이든 20이든,
(a) 판정에 들어가지 않고, (b) 907032의 수치와 비교해서도 안 된다(모델·백엔드·ctx가 다르다).
C층 수치는 **기록**된다.

---

## 5. 계측 3건 — 각각 (무엇을 읽는가 / 어떤 값이 무엇을 뜻하는가 / 판정에 쓰이는가)

전부 `R2C_INSTRUMENT=1`에서만 돌고, 전부 **warm-up boot**(채점되지 않는 boot)에 붙으며,
**추가 scored boot는 0개**다. 판정 경로 비접촉의 근거는 §6.

### I1 — 부팅 배너 (KV 여유 확인) **(D1: 死因 N3 해소로 전면 교체)**

- **무엇을 읽는가**: `srv_warmup.log`(및 종료 후 모든 `srv_*.log`)에서
  `Mamba Cache is allocated. max_mamba_cache_size: N, …` ·
  `KV Cache is allocated. #tokens: …` ·
  `max_total_num_tokens=…, max_running_requests=…, context_len=…, available_gpu_mem=… GB`.
  추가로 **(D8)** `sm_counts (prefill_sm, decode_sm): [...]` 한 줄과 warm-up boot의
  green read-out. 산출물 `instrument/I1_banner_warmup.txt` ·
  `instrument/I1_green_warmup.txt` · `instrument/green_warmup.json` ·
  `instrument/I1_boot_banners.txt`.

- ★**`48`은 측정이 아니다 — 연역이다.** rev1은 이 값을 "ctx 8704 / mem 0.80에서만 측정된
  전제"로 서술했는데 **거짓이다.** 이 게이트의 서버 명령이 `--disable-radix-cache`와
  `--max-running-requests 48`을 함께 고정하므로:

  - `model_runner_kv_cache_mixin.py:223-230` — `max_mamba_cache_size`가 미지정이고
    `disable_radix_cache`이면 `max_mamba_cache_size = max_running_requests // dp_size`
    **대입**(dp_size=1 ⇒ 48). 메모리에 맞춰 자동 산정하는 `else` 분기(`:231-248`)는
    **radix cache가 켜져 있을 때만** 탄다.
  - `_calculate_mamba_ratio`(`:390-392`) — `disable_radix_cache`면 **즉시 1** 반환.
  - `_resolve_max_num_reqs`(`:857-876`) — `estimated = max(min(token_capacity/ctx·512,
    4096), 2048) ≥ 2048`, `min(48//1, estimated) = 48`, 이어 `min(48, 48//1) = 48`.

  ⇒ **ctx·mem-fraction은 이 값에 들어가지 않는다.** 메모리가 모자라면 값이 줄지 않고
  **부팅이 죽는다**(그건 F1/§9-b의 관할). **job 905835의 48도 같은 항등식이다**
  (`c_capacity.sbatch:126-127`이 같은 두 플래그를 쓴다) ⇒ 이 전제는 **어디에서도 측정된 적이
  없고, 측정할 필요도 없다.** λ0 판정서 §3-5의 "실측" 인용은 이로써 **정정**된다.
  ★나는 이 세 지점을 판정서 인용이 아니라 **엔진 소스를 직접 열어** 확인했다(게이트 #110).

- **그래서 I1이 실제로 무엇을 재는가**: `max_total_num_tokens`와 `available_gpu_mem`뿐이다.
  이 둘은 `torch.cuda.mem_get_info` 기반의 **런타임 측정값**이라 출력공간에 거짓이 존재한다.
  905835(ctx 8704 / mem 0.80 / **flashinfer**) 대비 이 튜플에서 실제로 달라지는 성분은
  ctx 16384의 `req_to_token` 증분(**3.1 MB**)·cudagraph 캡처 메모리·mem 0.80→0.82의 순증분
  이고, 그 합이 KV 풀에 얼마를 남기는지는 **미측정**이다. 예보는 **F2′**(§7).
  ★**(NPC-B 정정) rev2가 "이 튜플에서 처음 바뀌는 성분"으로 들었던 "flashinfer workspace
  버퍼"는 삭제한다 — 틀렸다.** job 905835가 **이미 flashinfer였고**(§9 비용표·§10이 그렇게
  적고 있었다 ⇒ 문서 내부 모순), 그 워크스페이스는 `environ.py:349`의 **384 MB 기본값이며
  ctx와 무관**하다. 이 값을 덮는 분기는 두 개뿐인데 **둘 다 이 구성에 해당하지 않는다**
  (직접 확인): `flashinfer_backend.py:172-180`의 512 MB는 **Qwen/MiMo arch 한정**,
  `:191-200`의 2048 MB는 **`--enable-deterministic-inference` 한정**이다.
  - `max_mamba_cache_size`·`max_running_requests`는 **연역값으로서 기록만** 한다 — 48이
    아니면 그것은 용량 사실이 아니라 **이 문서의 코드 독해가 틀렸다는 신호**다(그 경우
    §5·§7·NP-7′ 전체를 재작성해야 한다).
- **(D8) 분할 실현**: I2/I3 수치는 "D44에서 쟀다"고 인용될 것이므로, warm-up boot의 분할이
  **target이 아니라 realized**임을 확인한다. `R2C_INSTRUMENT=1`일 때만 warm-up boot에
  `PDMUX_GREEN_READOUT=1`을 붙인다(기본 경로 argv는 바이트 동일 — 테스트가 고정). 읽는 값은
  index 4의 `prefill.green_sm.smCount = 64` · `decode.green_sm.smCount = 44` ·
  `green_ctx_is_null = false`. **`WARMUP_GREEN_READOUT_MISSING`이면 realization은 미확인**이고,
  그때는 NP-9의 "추론" 문안이 발화한다.
- **판정에 쓰이는가**: **아니다. 기록 항목이다**(F2′만 예보).

### I2 — shape A 바닥 프로브 (256 in, 512 out) × 8, **동시성 1**

- **무엇을 읽는가**: `instrument/I2_shapeA_conc1.{log,jsonl}` — TTFT(256-token prefill 바닥),
  ITL(B=1), 요청당 상세(`--output-details`).
- **어떤 값이 무엇을 뜻하는가**: λ0 감사 §3-3이 λ\*(A)를 **B=7→48의 6.9배 외삽**으로 점추정
  (2.1–2.5 req/s)했고, 그 외삽의 절편이 `step(B=1)=13.06 ms`였다. I2는 그 자리의 값을
  이 튜플에서 준다. 하네스는 probe C가 폐기 warm-up으로 쓴 것과 **같은 레시피**다
  (`c_capacity.sbatch:139-145`, `--max-concurrency 1 --seed 7`).
- ★**(D6-i) 단위가 다르다.** I2가 주는 것은 **클라이언트 관측 ITL/TTFT**이고, λ0 §3-3의
  절편은 **엔진 decode step**(`gen throughput`에서 환산)이다. 전자는 HTTP·호스트 스케줄링·
  디토크나이즈를 포함하므로 **같은 양이 아니다**. 두 값을 같은 회귀에 넣지 않는다 — I2는
  절편을 "대체"하지 않고 **클라이언트 쪽 바닥**을 준다. (엔진 step이 필요하면
  `srv_warmup.log`의 `Decode batch … gen throughput` 줄에서 `1000·B/G`로 따로 계산하며,
  그 계산은 여기 등록되지 않았다.)
- ★**(D6-ii) D44 한정.** I2는 **prefill 64 SM / decode 44 SM** 위에서 돈다. 이 TTFT·ITL은
  모델의 일반 바닥이 아니라 **이 분할에서의** 바닥이다. "W3의 SLO 여유"에 쓰려면 이 한정을
  반드시 병기한다(NP-9).
- **판정에 쓰이는가**: **아니다.** 다음 회차 사다리 설계 입력이다.

### I3 — 닫힌 루프 포화 상한 (2 shape × 1 셀)

- **무엇을 읽는가**: `(256,512)`와 `(8192,64)` 각각
  `--request-rate inf --max-concurrency 64 --num-prompts 300 --seed 41`로 1셀씩.
  산출물 `instrument/I3a_shapeA.{log,jsonl}`, `instrument/I3b_shapeB.{log,jsonl}`,
  `instrument/I3_max_running_req.txt`, `instrument/I_log_offsets.txt`.
- **`--request-rate inf`의 의미(코드 인용)**: `sglang/bench_serving.py:943-945`

  ```python
              if request_rate == float("inf"):
                  # If the request rate is infinity, then we don't need to wait.
                  continue
  ```

  `get_request()`의 이 분기가 **sleep을 전부 건너뛴다**. 따라서 `np.random.exponential`
  (`:948`)이 호출되지 않고 **Poisson 실현이 존재하지 않는다** ⇒ λ0 死因 N2(비포화 셀의
  `achieved/offered`가 포화도가 아니라 도착 실현 계수 `1/Ē`라는 문제)가 **구조적으로 소멸**
  한다. (`--seed`는 여전히 데이터셋 표본 추출에 쓰이므로 41로 등록한다.)
- **어떤 값이 무엇을 뜻하는가**: `Benchmark duration`과 `Successful requests`에서
  `achieved = N/duration`이 **닫힌 루프 포화 처리율**로 직접 나온다. 이것이 다음 회차
  사다리의 **위쪽 끝**을 추측 없이 고정한다.
- ★**(D7) "양 끝"이 아니다.** rev1의 "사다리의 **양 끝**을 고정한다"는 **틀렸다**:
  λ\*_inf ≥ λ\*_Poisson이므로 상한 프로브는 **아래쪽을 구속하지 않는다**. 사다리의 하한은
  여전히 λ0 재등록이 별도 근거로 정해야 하며, I3를 하한에 쓰면 브래킷을 놓친다(NP-3′).
- ★**필수 등재 — 이것은 λ\*의 대체가 아니라 상한 프로브다.**
  closed-loop(concurrency 64) ≠ open-loop Poisson일 수 있다. 근거(측정): probe C의 포화
  d44 셀(`d44_r2_o96`)은 `Concurrency: 24.08`이었고 그중 **22.75가 대기(L_pre)**,
  실행 중인 decode 인구는 1.33뿐이었다. 닫힌 루프는 admitted 배치를 더 채울 수 있으므로
  **λ\*_inf ≥ λ\*_Poisson**일 수 있다. ⇒ 결과 문서는 반드시
  "**이 값은 open-loop Poisson λ\*의 상한이며 λ\*가 아니다**"를 병기한다.
- ★**클라이언트 병목(confound #7) 확인 항목 — 감사 처방을 정정해 등록한다.**
  감사 §3-8 (a)는 "`Concurrency:` 보고값이 엔진 상한 48 근처인지"를 확인 항목으로 제안했다.
  **그 지표는 이 설계에서 정보를 담지 않는다**: `bench_serving`의 `concurrency`는
  Σ(e2e latency)/duration이라 **대기 시간을 포함**하고, `--max-concurrency 64`는 항상 64개를
  미완료 상태로 유지하므로 값이 **구성상 64 근처에 고정**된다(probe C의 open-loop 24.08이
  대기 22.75를 포함했던 것과 같은 이유). 대신 **엔진 쪽 지표**를 등록한다:
  - `instrument/I3_max_running_req.txt` = `srv_warmup.log`의 `#running-req: N` **전역**
    최댓값(작업 중 눈으로 보는 값).
    - **48(= mamba cache 슬롯 수)에 도달**하면 구속은 엔진이다 ⇒ 측정은 유효.
    - **48보다 한참 낮은 값에서 평탄**하면 그 셀의 achieved는 포화 처리율로 인용할 수 없다.
  - ★**(D4) F5는 셀별로 판정한다. 전역 최댓값으로는 "두 셀 모두"를 말할 수 없다.**
    실행 전에 재계산 레시피를 문자로 고정한다 — `I_log_offsets.txt`의
    `<cell>_start`는 **배타**(그 행 다음부터), `<cell>_end`는 **포함**:

    ```bash
    S=$(awk '$1=="I3a_shapeA_start"{print $2}' instrument/I_log_offsets.txt)
    E=$(awk '$1=="I3a_shapeA_end"{print $2}'   instrument/I_log_offsets.txt)
    sed -n "$((S+1)),${E}p" srv_warmup.log \
      | grep -oE '#running-req: [0-9]+' | awk '{print $2}' | sort -n | tail -1
    ```

    (I3b도 동일, 이름만 교체.) 경계 ±1행의 모호성은 knife-edge에서만 문제가 되며,
    위 배타/포함 규약을 **실행 전에** 고정함으로써 사후 선택 여지를 없앤다.
  - 보조: achieved가 probe C의 open-loop 포화값(shape B, D44, out=96 기준 0.675 req/s)
    **이상**인지. 밑돌면 두 하네스 차이(ctx·out·도착 과정)를 먼저 설명해야 한다.
- **판정에 쓰이는가**: **아니다.** 다음 회차(λ0 재등록)의 입력이다.

---

## 6. 하네스 변경 — 채점 경로에 닿지 않음을 무엇으로 보장하는가

변경 파일은 `../r2_correctness.sbatch`와 **(D10)** `scripts/bootstrap/sync_engine_tree.sh`
**둘**이다(판정 코드 `r2_correctness_check.py`·클라이언트 `r2_correctness_client.py` 무수정).

- 노브 `R2C_INSTRUMENT`, **기본값 0**. 0이면 **계측이 프로세스를 한 개도 띄우지 않는다**.
  ★**(NPC-D) 단, "기본 경로가 907100·907456과 명령 단위로 동일하다"는 말할 수 없다**:
  D10이 provenance 블록에 넣은 `python -c "import flashinfer"`가 **`R2C_INSTRUMENT`·백엔드와
  무관하게 무조건 1회** 실행된다(`r2_correctness.sbatch`의 provenance 절). 이 프로세스는
  **첫 서버가 뜨기 전에 종료**되고 GPU 컨텍스트를 남기지 않으므로 측정량은 바뀌지 않지만,
  "추가 프로세스 0"은 **계측 블록에 한정된 주장**이다. 서버·클라이언트 argv의 바이트
  동일성은 별개로 유지되며 테스트가 고정한다.
- 계측은 **warm-up boot**에 붙는다. 채점기 `r2_correctness_check.py`는 `boots.txt`와,
  거기 적힌 label마다 `gen_<label>.json` / `srv_<label>.log` / `tel_<label>.jsonl` /
  `green_<label>.json` **다섯 가지만** 연다(디렉터리 순회 없음). warm-up boot은
  `boots.txt`에 **적히지 않는다**. 또한 warm-up boot은 `PDMUX_TELEMETRY_PATH` **없이** 뜨므로
  텔레메트리를 **한 줄도 쓰지 않는다**.
- **왜 scored boot에 붙이면 안 되는가**(붙이지 않은 이유를 명시): per-boot 검사 셋 중 셋이
  그 boot 아티팩트 **전체에 대한 전칭 술어**다 — B3("모든 Decode 줄이 `cuda graph: True`"),
  B5("모든 controller_decision이 fixed/미제한"), B8("모든 prefill∧decode 스냅샷이 D 분할
  인덱스"). 부하를 얹으면 세 술어의 입력 모집단이 바뀐다. 그래서 붙이지 않았다.
- 산출물은 전부 `job_<id>/instrument/` 아래에만 쓰인다.

**(D8) warm-up boot의 green read-out**: `R2C_INSTRUMENT=1`일 때만
`PDMUX_GREEN_READOUT=1 PDMUX_GREEN_READOUT_PATH=$OUT/instrument/green_warmup.json`이 warm-up
launch에 붙는다. 변수는 **따옴표 없이** 전개되므로 `INSTR=0`에서는 빈 값이 단어 분할로
사라져 **argv가 바이트 동일**하다. `green_warmup.json`은 scored 루프가 낼 수 있는 라벨
(`L<n>`/`TD<n>`)의 `green_<label>.json` 형식이 **아니고** `instrument/` 아래에 있으므로
채점기가 열 수 없다.

**테스트로 고정**(`tests/test_r2_correctness_instrument.py`, **35 케이스**, GPU 0):
기본값 0에서 실행 0 · 활성 시 정확히 3회 호출과 등록된 플래그 · 전부 warm-up 포트 ·
warm-up 미가동 시 실행 0 + `INSTRUMENT_SKIPPED_WARMUP_DOWN` · `server_cmd()` 바이트 동일 ·
scored client 호출 동일 · 판정 코드 sha 불변 · 채점기가 warm-up/instrument/green_warmup을
언급조차 하지 않음 · 채점기 경로 템플릿이 넷뿐이고 디렉터리 순회 없음 · timeout 상한 고정 ·
**(D8)** warm-up argv가 `INSTR=0`에서 핀된 리스트와 **정확히 일치**하고 `INSTR=1`에서 **정확히
2개만** 늘어남 · **(D11)** `#SBATCH --time=02:30:00` · **(D10)** manifest에
`flashinfer_backend.py`가 **맨 끝에** 있음 + `flashinfer_version` 기록 ·
`bench_serving.py:943-945`/`:1705-1706` 인용 줄 검증.
★**음성대조 4건**(교훈 53): 가드를 제거한 변이 / probe를 scored 포트로 돌린 변이 /
`--request-rate inf`를 유한 rate로 바꾼 변이 / **(D8)** `$WARM_GREEN`에 따옴표를 씌운 변이
(빈 argv 원소가 주입돼 바이트 동일성이 깨진다)가 각각 해당 시험을 **반드시 실패시킨다**.

---

## 7. 반증 가능한 사전 예보 (실행 전 고정)

| # | 예보 | 어긋나면 |
|---|---|---|
| **F1** | 5 boot(warm-up 1 + scored 4) 전부 뜬다. `BOOT_FAILURES.txt` 없음 | 부팅 실패 → §9 처분(측정 실패) |
| **F2′** **(D1 교체)** | ctx 16384 / mem 0.82 / flashinfer에서 `max_total_num_tokens ≥ 2.0 × 10⁶` **이고** `available_gpu_mem > 5 GB` | KV 과공급 가정이 이 튜플에서 깨진 것 ⇒ 다음 회차 사다리의 KV 여유 가정을 재유도. (이 job의 판정과는 무관) |
| **F3** | O1이 boot마다 8/8 probe에서 성립(최악 여유 +41 토큰) | `NO_VERDICT_UNREALIZED`, **게이트 실패 아님** |
| **F4** | 모든 boot의 로그에 `Disable piecewise CUDA graph because the capture size is not set`가 나오고, `disable_cuda_graph=False`이며 Decode 줄의 `cuda graph: False`는 **0건** | piecewise가 켜졌다면 §10-3 위험이 실제 발화한 것 ⇒ 별도 회차 |
| **F5** | I3 **두 셀 각각**(§5-I3의 D4 레시피로 셀별 재계산) 엔진 `#running-req` 최댓값이 **48에 도달** | 미달이면 그 셀의 achieved를 포화 처리율로 인용 금지. ★**(D4) 미달의 원인이 상류 병목이라고 단정하지 않는다** — (8192,64)는 prefill 지배라 admitted decode 인구가 엔진 사정으로 낮을 수 있다. 원인 미확정으로 기록한다 |
| **F6** | I3b(8192,64)의 achieved ≥ 0.675 req/s (probe C open-loop 포화값, 다른 ctx·out) | 밑돌면 두 하네스 차이를 먼저 설명하기 전에는 "상한"이라 부를 수 없다 |

★**(D1 + NPC-B) F2′의 강도에 대한 정직한 서술**: F2′는 **거짓이 될 수 있지만 사전 확률이
매우 낮다**. 905835는 더 빡빡한 ctx 8704 / mem 0.80 / **flashinfer**에서
`max_total_num_tokens = 2,618,868`, `available_gpu_mem = 13.38 GB`였고, mem 0.80→0.82는 static
예산을 늘리며 ctx 증가의 비용은 `req_to_token` 48×16384×4 B = **3.1 MB**뿐이다. 재감사 투영치
**`max_total_num_tokens ≈ 2.72×10⁶` / `available_gpu_mem ≈ 11.6 GB`**이고 등록 문턱까지의 거리는
각각 **≈11.0 GB / ≈6.6 GB**인데 **식별된 신규 소비자 총합은 ≈0.21 GB**다.

그럼에도 이것이 rev1의 F2보다 나은 이유는 **범주가 다르기 때문**이다 — F2′의 두 값은
`torch.cuda.mem_get_info` 런타임 판독이라 **출력공간에 거짓이 존재**하는 반면, rev1의 F2는
엔진이 **대입**하는 값이라 거짓 분기가 아예 없었다(항등식, 死因 N3).
★**따라서 F2′ 충족은 KV 여유 가정의 "확인"이 아니라 기록이다**(NPC-B). 이 한계를 여기
등록한다. ★**(NPC-B 정정)** 이전 판본이 여기 적었던 "flashinfer workspace 버퍼"는
**삭제한다** — 905835가 이미 flashinfer였으므로 새로 바뀌는 성분이 아니고, 그 버퍼는
`environ.py:349`의 384 MB 기본값·ctx 무관이다(덮는 두 분기는 Qwen arch 한정·deterministic
한정이라 해당 없음, §5-I1).

**예보하지 않는 것**(의도적): 판정 라벨 자체. §3.3 때문에 PASS·INCONCLUSIVE 어느 쪽도
사전 확률을 주장하지 않는다.

---

## 8. 출력 공간 전수 + 처분 (rule v2 우선순위 순)

| 판정 | 뜻 | 처분 · 쓸 수 있는 문장 |
|---|---|---|
| **FAIL** | B2–B6 실패, 또는 입력 동일성 붕괴, 또는 **S 티어의 임의 boot 쌍 불일치**(체커는 `mm["S"]`의 **전 쌍**을 본다 — L-TD 교차뿐 아니라 **TD-TD**도, **L-L**은 rule 3이 먼저 잡는다), 또는 (O: arm-내 두 쌍 모두 동일한데 L-TD 불일치) | 게이트 불통과. **"true-dual이 이 쌍에서 틀렸다"는 귀속은 이 job 단독으로 성립하지 않는다**(§3.3-ii). 다음 단계는 원인 분해 회차이며 여기 등록되지 않았다. P2 캠페인 착수 금지 |
| **NO_VERDICT_INFRA** | boot 누락인데 로그에 크래시 없음, 또는 한 arm에 booted boot 0 | **측정 실패**(교훈 21·게이트 #21). 규칙 실패로 적지 않는다. 재제출 대상 |
| **INCONCLUSIVE** | 두 L boot이 S에서 불일치, 또는 arm-내 O쌍이 불일치 | **참조 자체가 재현되지 않는다** = 이 쌍의 측정 가능성 문제. §3.3-i대로 flashinfer 적응 split-KV가 유력 후보이나 **미확정**. true-dual에 대한 어떤 주장도 금지 |
| **NO_VERDICT_UNREALIZED** | B7/B8 실패, 또는 어떤 probe에서 O1–O4 미확인 | D44 분할 위 중첩이 **실현·관측되지 않았다**. 등가성 인증 불가. 게이트 실패 아님 |
| **PASS** | 나머지 전부 | §2의 "인증하는 것"만 말할 수 있다 |

### (D3) FAIL 표의 세부 — **TD-TD S 불일치**

rev1은 FAIL 행을 "(S: L-L 동일한데 **교차** 불일치)"라고만 적었는데 코드는 더 넓다:
`s_ref_ok and not s_all_ok`에서 `s_all_ok`는 **모든 boot 쌍**을 본다 ⇒ **TD1-TD2 S 불일치도
FAIL**이다. 감사자가 907456 아티팩트에 1토큰 섭동을 넣어 재현했다(`L1-L2=0, TD1-TD2=1` →
`FAIL`).

**이 라벨은 방어 가능하다**: S 티어는 요청이 **한 번에 하나**(배치 1)라 flashinfer의 적응
split-KV가 boot 간 결정적이다 ⇒ TD가 자기 자신을 재현하지 못하는 것은 **측정 가능성 문제가
아니라 true-dual 결함**이다. 서술은 "**true-dual 자기 재현 실패**"로 하고, 교차-arm 귀속과
구별한다(NP-10).

### (D2′) 귀무대조가 **공허**한 S FAIL — `NO_VERDICT_INFRA`로 보고한다

체커의 O 티어는 귀무대조 표본 수를 명시 요구한다(`len(l_boots) >= 2 and len(td_boots) >= 2`).
**S 티어에는 대응 가드가 없다** ⇒ L boot이 하나만 살아남으면 `s_ref_ok`가
`itertools.combinations(['L1'], 2)` 위의 **공허 참**이 되어, 참조 재현성을 확인하지 **않은 채**
교차 불일치 1토큰이 FAIL을 발화한다. 감사자 실측: `gen_L2.json` 제거만 → `NO_VERDICT_INFRA`,
제거 + TD1 S 1토큰 섭동 → **`FAIL`**.

**사전 규칙 D2′(결정적, 사후 선택 여지 없음 — 체커는 고치지 않는다)**:

> 아래 **세 조건이 모두** 성립할 때에만 이 job은 그 결과를 **`NO_VERDICT_INFRA`로
> 보고한다**(체커 출력 라벨이 무엇이든):
>
> 1. `report["failures"] == ["S_TIER_MISMATCH"]`, **그리고**
> 2. `len([lb for lb in report["boots"] if lb.startswith("L") and
>    report["per_boot"][lb]["checks"]["B1_BOOTED"]]) < 2`, **그리고**
> 3. `report["checks"]["S_pairs"]` 안에 **양쪽이 모두 TD인 불일치 쌍이 없다**.
>
> 이유: 이 세 조건이 함께 성립할 때에만 L-L 참조 재현성이 **확인되지 않은 채**
> 교차-arm 불일치가 FAIL을 발화한다.

★**rev2의 "TD 쪽도 대칭" 조항은 철회한다.** 그 조항은 틀렸다 — 체커의
`s_ref_ok = all(… for a, b in combinations(l_boots, 2))`는 **L-L 쌍에만 양화**되므로
**TD boot 수는 `s_ref_ok`의 공허성에 영향을 주지 않는다**. 대칭 조항을 두면 지키는 것은
없고, 대신 **귀무대조가 완전한 진짜 교차-arm FAIL을 INFRA로 강등**한다(재감사가 907456
아티팩트 + 1토큰 섭동으로 2건 재현: `gen_TD2` 제거 + `TD1` 섭동 → `S{L1-TD1:1, L1-L2:0,
TD1-L2:1}` `FAIL`). 조건 3도 같은 이유로 필요하다 — 불일치 쌍이 **TD-TD**뿐이면 그것은
§8·NP-10이 "방어 가능"이라 등록한 **true-dual 자기재현 실패**이지 참조 결손이 아니다.

세 값 모두 리포트 JSON에 그대로 있고(`boots`·`per_boot[*].checks.B1_BOOTED`·`failures`·
`checks.S_pairs` 존재를 실물 리포트로 확인) 재량이 개입할 자리가 없다. 이 규칙은 §9의 규율
("측정 실패를 게이트 실패로 라벨하지 않는다")을 S 티어에도 **과잉 없이** 적용하는 것이다.
**이 오버라이드가 발화하면 NPC-A의 전사 의무가 함께 발화한다.**

**계측 산출물은 어느 라벨에서도 살아남는다** — warm-up boot에서 나오므로 scored boot이
전부 죽어도 I1–I3는 존재한다(단 warm-up 자체가 안 뜨면 §9-c).

---

## 9. 비용과 실패 모드별 처분

### 비용 추정 (측정 기반, job 905835·907456)

| 구간 | 근거 | 추정 |
|---|---|---|
| env sync + provenance + ctx 해석 | 907456 | ~1 분 |
| warm-up boot (9B, cold) | 905835 첫 셀 로그 11:31:39→11:32:42 = **63 s** | ~1.5 분 |
| I2 (8×(256,512), 동시성 1) | ITL(B=1) 13.06 ms × 511 × 8 ≈ 53 s + 데이터셋/토크나이저 로드 | ~1.5 분 |
| I3a ((256,512), inf, 64) | λ\*(A) 점추정 2.1–2.5 req/s → 300 req ≈ 120–143 s | ~3 분 |
| I3b ((8192,64), inf, 64) | λ\*(B) 실측 0.675 req/s → 300 req ≈ 444 s (닫힌 루프면 더 짧을 수 있음) | ~8 분 |
| scored 4 boot | warm boot 29 s(905835) + client(S16/O8/C32) ~2 분 + teardown 20 s | ~12–16 분 |
| 채점 | | <0.2 분 |
| **합계** | | **≈ 28–35 분 ⇒ 0.47–0.58 GPU-h** |

### (D11) 최악 예산 정정 — rev1의 "71분"은 과소평가였다

| 시나리오 | 계산 | 합 |
|---|---|---|
| 전형 | 위 표 | **28–35 분** |
| **현실적 최악**(read-out 3건 전부 timeout, boot은 정상) | 3000 s 계측 + ~16 분 scored + 3 분 부대 | **≈69 분** |
| **구조적 최악**(스크립트의 모든 timeout이 발화) | warm-up health 900 s + 계측 3000 s + 4 × (health 480 s + client 900 s + teardown 20 s = 1400 s) + 부대 180 s | **≈161 분** |

rev1은 warm-up health(900 s)와 scored boot의 health/client timeout(1400 s × 4 = 93 분)을
빠뜨렸다. 정정한다.

**처분**: `#SBATCH --time`을 **`01:15:00` → `02:30:00`으로 올렸다**(지시자 자체를 수정).
월타임 상한은 **어떤 측정량도 바꾸지 않고** 큐 대기만 바꾸므로, 지시자에 넣는 것이 CLI로
매번 기억하는 것보다 안전하다(단일 인적 실패점 제거). ★**02:30:00으로도 구조적 최악
(161분)은 덮지 못한다** — 그 분기는 §9(f)가 관할한다. scored 경로의 timeout(health 480 s,
client 900 s)을 줄이면 덮을 수 있으나 **그것은 기본 경로를 바꾸는 일이라 하지 않는다**.

이 트랙 누적: 0.43 → **약 0.9–1.0 GPU-h** 예상(현실적 최악 1.6, 하드캡 2.5).

### 실패 모드

| 모드 | 증상 | 처분 |
|---|---|---|
| (a) scored boot 부팅 실패 | `BOOT_FAILED boot=<L>`, health 480 s 초과 | 로그에 traceback 있으면 rule 1(FAIL/B1), 없으면 rule 2(`NO_VERDICT_INFRA`) = **측정 실패**. 재제출 |
| (b) OOM (ctx 16384 / mem 0.82) | 부팅 중 CUDA OOM | 905835는 ctx 8704 / mem 0.80에서 `available_gpu_mem=13.38 GB` 여유였다. 발생 시 = **스코프 변경 사유**(mem 0.80으로 재등록)이지 규칙 실패 아님. ★기본값을 조용히 바꿔 재제출하지 않는다. ★★**(D12) 오귀속 차단**: **첫 scored boot**이 OOM이면 원인이 ctx/mem인지 **50분 부하 뒤 warm-up 서버의 잔류 메모리**인지 먼저 구분한다 — `R2C_INSTRUMENT=0`으로 **1회** 재제출해 그때도 죽는지 본다. 구분 전에는 mem 0.80 재등록을 하지 않는다 |
| (c) warm-up boot 미가동 | `INSTRUMENT_SKIPPED_WARMUP_DOWN` | 계측 3건 전부 미측정으로 기록. scored 경로는 **기존과 동일하게 계속 진행**(warm-up은 원래 페이지 캐시용) |
| (d) read-out timeout | `I2_rc=124` 등 | 해당 계측만 **측정 실패**로 기록. 판정 무영향. 다음 회차에서 예산 재산정 |
| (e) client timeout (`timeout 900`) | `CLIENT_FAILED boot=<L>` | 해당 boot의 `gen_*.json` 없음 → (a)와 같은 경로 |
| (f) wall-clock 초과 | job 강제 종료 | 계측 예산 초과가 원인이면 `R2C_INSTRUMENT=0`으로 게이트만 먼저 사고 계측은 별도 회차 |
| (g) 서버가 client 도중 사망 | `SERVER_DIED_DURING_CLIENT` | rule 1/2 경로 |

### (D5) 재제출 정책 — 실행 전에 고정

교차-잡 토큰 불일치는 **측정돼 있다**(X1 감사: job 907100 ↔ 907456에서 8/96 단위 불일치)
⇒ 재제출은 출력을 실제로 바꾼다. 따라서 재추첨을 제한한다:

- 재제출은 **`NO_VERDICT_INFRA`에 한해 최대 1회**(§9의 D12 분기 진단용 `R2C_INSTRUMENT=0`
  재제출도 이 1회에 포함한다).
- **`FAIL` · `INCONCLUSIVE` · `NO_VERDICT_UNREALIZED`는 재제출로 덮어쓰지 않는다.**
  원인 분해가 필요하면 그것은 **새 사전등록**의 대상이다.
- 재제출하면 **두 job의 라벨을 둘 다 보고**하고, 뒤엣것만 인용하지 않는다.

**규율**: 위 어느 것도 "게이트가 실패했다"로 라벨하지 않는다(교훈 21 / 게이트 #21).

---

## 10. 상류 위험 3건이 이 구성에서 발화하는가 (코드 확인 결과)

### 10-1. `mamba2_cache_params`의 미선언 `self.n_groups` — **발화하지 않는다. 단 체크포인트 의존.**

`configs/nemotron_h.py:424`가 `n_groups=self.n_groups`를 쓰는데, `NemotronHConfig.__init__`은
`mamba_n_groups`만 선언한다(`:270` 기본 8, `:352` `self.mamba_n_groups = mamba_n_groups`).
즉 `self.n_groups`는 **`PretrainedConfig(**kwargs)`가 체크포인트 `config.json`의 최상위 키를
속성으로 꽂아 줄 때만** 존재한다. 이 체크포인트를 확인했다
(`hf_cache/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base/snapshots/dc0661c8…/config.json`):
**`n_groups: 8` 존재**(그리고 `mamba_n_groups`는 **부재**). 값도 기본값 8과 일치.
⇒ 이 job에서는 `AttributeError`가 나지 않는다(905835 12/12 boot과 정합).
**잔존 위험**: `n_groups`를 선언하지 않은 NemotronH 체크포인트로 바꾸면 부팅이 죽는다.
모델 축을 다시 바꿀 때 재확인 항목.

### 10-2. `config.expand` stale — **발화하지 않는다(미사용).**

`configs/nemotron_h.py:357`이 `self.expand = mamba_expand`(기본 2)를 두는데, 이 체크포인트에는
`mamba_expand`도 `expand`도 **없다** ⇒ 2로 남는다. 2 × hidden 4480 = **8960**인데 실제
mamba intermediate는 `mamba_num_heads × mamba_head_dim` = 128 × 80 = **10240**이라 stale이
맞다. 그러나 `mamba2_cache_params`는 `intermediate_size=self.mamba_num_heads *
self.mamba_head_dim`을 쓰고(`:423`), 출하 모델 구현 `src/models/nemotron_h.py`는
`config.expand`를 **한 번도 읽지 않는다**(`intermediate_size` 참조는 전부 MLP/MoE 경로).
⇒ 이 게이트에서 영향 없음. **등재만** 하고 수정하지 않는다.

### 10-3. `NemotronHForCausalLM`이 `piecewise_cuda_graph_disabled_model_archs`에 없음 — **이 구성에서는 발화하지 않는다. 이유가 arch가 아니라 chunked-prefill이다.**

목록(`configs/model_config.py:1352-1358`)에 이 arch는 실제로 **없다**. 그러나 이 게이트는
`--chunked-prefill-size -1`로 돌고, 비-MLA 경로에서

- `server_args.py:1254-1259` → `piecewise_cuda_graph_max_tokens = chunked_prefill_size = -1`
- `:1397-1415` `_generate_piecewise_cuda_graph_tokens()` → `s <= -1` 필터 ⇒ **빈 리스트**
- `model_executor/model_runner.py:2486-2491` → `if not piecewise_cuda_graph_tokens:`
  → **"Disable piecewise CUDA graph because the capture size is not set"** 로 끄고 반환

이 경로는 **arch와 무관**하다. 실증: job 907100의 `srv_L1.log:71`에 바로 그 줄이 있고
서버 인자는 `disable_piecewise_cuda_graph=False`였다(=플래그는 안 붙었는데 런타임은 꺼짐).
⇒ probe C가 `--disable-piecewise-cuda-graph`를 **명시적으로** 붙인 것과 이 게이트가
안 붙인 것은 **런타임 동작이 같다**(λ0 감사 D11이 지적한 차이는 이 조건에서 무효).
캠페인 러너도 cudagraph-ON 경로에서 `--chunked-prefill-size -1`을 쓰므로 동일하다
(`engine_bench_runner.sh:94`, `:99`).
★**잔존 위험(미해소)**: `PDMUX_CHUNKED_PREFILL_SIZE > 0`인 arm을 만들면 이 모델에서
piecewise가 켜지고 `torch.compile` 경로가 처음으로 발화한다 — 그때 이 목록 부재가
살아난다. 이 게이트는 그 arm을 포함하지 않는다.

---

## 11. 선결 확인 — provenance manifest (λ0 감사 D12 · 이번 D10)

`scripts/bootstrap/sync_engine_tree.sh`를 **실제로 실행**해 확인했다(GPU 0):

- manifest **25줄**(λ0-D12 시점 24줄 + 이번 D10의 1줄).
- 모델 구현 포함: `sglang/srt/models/nemotron_h.py` sha `713333e87b4a…`,
  `sglang/srt/configs/nemotron_h.py` sha `bcdface1473d…`.
- ★**(D10) 25번째 줄 = `sglang/srt/layers/attention/flashinfer_backend.py`**
  sha `9181648bb265…`. 이 파일은 백엔드 전환으로 **결론의 신뢰 근거**가 됐는데
  (§3.3 — O층 동치가 이 파일의 적응 split-KV 스케줄에 걸려 있다) provenance 밖에 있었다.
  **append이지 insert가 아니므로** 이전 manifest들은 여전히 새 manifest의 **접두**다
  (24줄 버전과 `diff head -24` 결과 **완전 동일**함을 확인).
- `sha256sum -c` **25/25 OK**.
- 설치본 == 추적 사본 바이트 동일: `src/models/nemotron_h.py`의 sha가 설치본과 **같다**
  (`713333e87b4a…`).
- ★**manifest가 덮지 못하는 것(등재)**: manifest는 **sglang의 래퍼**를 해시할 뿐
  **설치된 `flashinfer` 휠**은 어떤 줄도 덮지 않는다. 그래서 sbatch provenance 블록에
  `flashinfer_version` 한 줄을 추가했다(이번 실행 환경 실측 **0.6.10**,
  `/home01/ehmoon/.local/lib/python3.14/site-packages/flashinfer/__init__.py`).
- 파급 확인: 워크스트림 B의 `lambda0.sbatch`가 `N_MANIFEST -lt 24`를 중단 조건으로 쓰므로
  25줄은 안전하다. manifest 항목 **수**를 고정하는 테스트는 저장소에 **없다**(직접 grep 확인).
- ★**(NPC-F) 그러나 "manifest 변경이 저장소에 파급을 주지 않는다"는 쓸 수 없다.** 항목 수·
  내용을 고정하는 **비-테스트 소비자가 2곳 더** 있다(둘 다 직접 열어 확인):
  - `results/ltsm_probe/ltsm_p1_probe.sbatch:95-104` — 참조 manifest
    `p1_gates/gate1/runtime_source_manifest_gate1c_877974.sha256`(**15줄**, 확인)과 **정확
    diff**를 요구하고 불일치 시 `exit 4`;
  - `results/smid_census/smid_l0_census.py:219` — `MANIFEST_EXPECTED_FILES = 15`.

  ★**둘 다 25번째 줄 때문에 새로 깨지는 것이 아니다** — 15 → 17 → 24 단계에서 **이미** 깨져
  있었다. 이 사전등록은 그 둘을 **고치지 않으며**(스코프 밖, 각자의 트랙 소관), 여기 등재만
  한다. 그 두 프로브를 다시 돌리려면 참조 manifest를 갱신하거나 상수를 고쳐야 한다.

⇒ D12·D10 충족. (대조: job 905835는 manifest 17줄인 채로 NemotronH를 12 boot 서빙했다 —
모델 구현 provenance 0. 게이트 #195의 실증 사례이며 이 job에서는 재발하지 않는다.)

---

## 12. 인용 금지 (결과가 나온 뒤에도 쓸 수 없는 문장)

> **NP-1′ (승계)** — "Nano-9B-v2에서 PD-mux가 정확하다/작동한다"는 **PASS여도 쓸 수 없다**.
> PASS가 인증하는 것은 §2의 문장 하나뿐이며, 특히 **성능·Claim D·다른 split·다른 정책·
> 다른 모델·다른 백엔드·T>0 샘플링·O 프로토콜 밖 동시성**은 포함되지 않는다.

> **NP-2** — 이 job의 **어떤 수치도 성능 결과가 아니다.** I2·I3가 내놓는 TTFT/ITL/처리율은
> 계측 기록이며 arm 비교에 쓰이지 않았다(계측은 legacy warm-up boot 한 개에서만 돈다 —
> true-dual과 비교되지 않았다).

> **NP-3′ (D7로 보강)** — **I3는 λ\*가 아니다.** 닫힌 루프(concurrency 64) 포화 처리율은
> open-loop Poisson λ\*의 **상한**이며, probe C 실측(`Concurrency 24.08` 중 대기 22.75)이
> 두 체제가 다를 수 있음을 보인다. "λ\*를 측정했다"는 문장 금지. ★**I3는 사다리의
> 위쪽 끝만 고정하며, 하한으로 쓸 수 없다**(λ\*_inf ≥ λ\*_Poisson이므로 아래쪽은 미구속).
> 또한 λ0 감사 Q1이 그대로 유효하다: **throughput 포화는 방법론 게이트 #6(지표 절벽 대비
> 용량)을 닫지 못한다.**

> **NP-4** — **이 게이트는 Zamba2 결과를 되살리거나 확장하지 않는다.** 907100·907456·X1의
> 결론은 (Zamba2-2.7B, triton) 한정 동결이고, 역으로 이 job의 결과도 그 쌍에 이식되지
> 않는다. X1의 민감도 시연(`triton_attention_num_kv_splits`)은 flashinfer에서 커널 효과가
> 없으므로 이식 불가다.

> **NP-5** — **C층 수치는 판정도 비교도 아니다.** 새 쌍의 C층 불일치 수는 907032의 7/32와
> 비교되지 않으며(다른 모델·백엔드·ctx), 판정에 들어가지 않는다.

> **NP-6** — **`INCONCLUSIVE`나 `NO_VERDICT_*`는 "true-dual이 실패했다"가 아니다.**
> 각각 참조 재현 실패 / 실현·관측 실패이며 측정 실패다(교훈 21·게이트 #21).

> **NP-7′ (D1으로 교체 — 이전 NP-7은 항등식을 사실로 승격했다)** — **I1의
> `max_mamba_cache_size: 48` / `max_running_requests=48`은 측정이 아니라
> `--disable-radix-cache --max-running-requests 48`의 연역이다**
> (`model_runner_kv_cache_mixin.py:223-230`, `:857-876`, `:390-392`). **job 905835의 48도
> 같은 항등식이다**(`c_capacity.sbatch:126-127`). 따라서 **"ctx 16384 / mem 0.82에서 구속
> 자원이 48임을 확인했다"는 문장은 쓸 수 없다.** I1이 실제로 재는 것은
> `max_total_num_tokens`·`available_gpu_mem`뿐이다(F2′). λ0 판정서 §3-5의 "실측" 인용도
> 이에 따라 정정된다. 905835의 λ\* 앵커는 여전히 provisional이다.

> **NP-8 (신규, D13)** — **PASS는 P2 캠페인 착수를 승인하지 않는다.** 이 게이트는 P2의
> **필요조건 하나**를 닫을 뿐이다. 남은 블로커: λ0(0단계) 사전등록이 `NO-GO`(死因 N2+N3),
> W4를 단일 λ\*로 파라미터화할 수 없다는 열린 항목(사용자 결정 대기), 그리고 방법론
> 게이트 #6(지표 절벽 대비 용량) 미충족(λ0 Q1).

> **NP-9 (신규, D6/D8)** — **I1–I3의 모든 수치는 D44 분할(prefill 64 SM / decode 44 SM) ·
> legacy 루프 · warm-up boot 한정이다.** 그 boot의 분할 실현은 `R2C_INSTRUMENT=1`의 green
> read-out(`instrument/green_warmup.json`)으로 확인하되, 그 파일이 없거나
> `WARMUP_GREEN_READOUT_MISSING`이면 realization은 **직접 관측되지 않았고** 근거는
> `sm_counts` target 표 + 같은 job의 scored L boot `green_realized=True`에 의한 **추론**이다.
> 또한 **I2가 주는 것은 클라이언트 관측 ITL/TTFT이지 엔진 decode step이 아니다**(단위 다름).

> **NP-10 (신규, D2/D3)** — booted L boot이 2개 미만인 채 나온 `S_TIER_MISMATCH`는
> **FAIL로 인용하지 않는다**(귀무대조가 공허 참이라 참조 재현성이 확인되지 않았다 —
> §8의 D2 규칙대로 `NO_VERDICT_INFRA`로 보고한다). TD 쪽도 대칭. **TD-TD S 불일치로 인한
> FAIL은 "true-dual 자기 재현 실패"로만 서술**하며 교차-arm 귀속과 구별한다.

### 재감사(rev2) 캐비앳 **NPC-A … NPC-J** — 결과 문서가 **문자 그대로** 승계한다

아래는 `VERDICT_newpair_rev2_2026-09-13.md` §5의 필수 병기 문안이다. NP-1′…NP-10과 **함께**
붙이며, 어느 판정 라벨이 나오든 무조건 적용된다.

> **NPC-A (등급 조건)** — **§8의 D2′ 오버라이드가 발화하면** 결과 문서는 반드시
> `r2_correctness_report.json`의 `checks.S_pairs` **전체**와 `per_boot[*].checks.B1_BOOTED`를
> **전사**하고, 그 job에 §9-D5의 재제출 1회를 **쓰지 않는다**. 이 경로에서
> **"교차-arm 불일치가 없었다"는 문장은 쓸 수 없다.** (재감사 재현: `gen_L2` 제거 + `TD2`
> 섭동 → `S{L1-TD1:0, L1-TD2:1, TD1-TD2:1}` `FAIL`; `gen_TD2` 제거 + `TD1` 섭동 →
> `S{L1-TD1:1, L1-L2:0, TD1-L2:1}` `FAIL` — rev2의 "TD 대칭" 조항은 이 둘을 잘못
> 강등했기에 철회됐다.)

> **NPC-B** — **F2′ 충족은 KV 여유 가정의 "확인"이 아니다.** 투영치
> `max_total_num_tokens ≈ 2.72×10⁶` / `available_gpu_mem ≈ 11.6 GB`이고 등록 문턱까지의 거리는
> 각각 **≈11.0 GB / ≈6.6 GB**인데, 905835 대비 식별된 신규 소비자 총합은 **≈0.21 GB**다.
> F2′는 "거짓이 될 수 있다"는 **형식 요건만** 충족하며, 결과는 **기록**으로만 쓴다.

> **NPC-C** — **`r2_correctness.sbatch`의 I1 주석을 근거로 "48이 이 ctx/mem에서 측정됐다"를
> 인용하는 것을 금지한다.** (이 권고에 따라 해당 주석 4줄은 NP-7′ 문안으로 **교체됐다**;
> 이전 판본의 job 디렉터리를 읽을 때는 여전히 이 금지가 적용된다.)

> **NPC-D** — **"`R2C_INSTRUMENT=0`이면 프로세스가 한 개도 추가로 뜨지 않는다 /
> 907100·907456과 명령 단위로 동일하다"는 거짓이다.** provenance 블록이
> `import flashinfer` 파이썬 프로세스를 **INSTR·백엔드와 무관하게 무조건** 1회 실행한다.
> 서버 부팅 전에 끝나므로 측정량은 바뀌지 않지만 **명령 단위 동일성은 주장할 수 없다.**

> **NPC-E** — **provenance는 flashinfer 휠의 버전 문자열(`0.6.10`)과 경로만 기록한다.**
> 내용 해시도 JIT 캐시도 아니며, 그 휠은 repo 밖·venv 밖
> (`~/.local/lib/python3.14/site-packages/flashinfer/`)이라 manifest가 드리프트를 보지 못한다.
> **"같은 flashinfer에서 쟀다"는 쓸 수 없고 "0.6.10이라고 보고한 설치본에서"까지만 쓸 수
> 있다.** (완화: flashinfer autotune은 이 하드웨어에서 꺼진다 — A100 CC 8.0 < 9,
> moe 백엔드 `auto`.)

> **NPC-F** — **"manifest 변경이 저장소에 파급을 주지 않는다"는 쓸 수 없다.** 항목 수·내용을
> 고정하는 비-테스트 소비자 2곳(`results/ltsm_probe/ltsm_p1_probe.sbatch:95-104` 정확 diff +
> `exit 4`, `results/smid_census/smid_l0_census.py:219` `MANIFEST_EXPECTED_FILES = 15`)이 있고,
> **둘 다 15→17→24 단계에서 이미 깨져 있었다**(25번째 줄이 새로 깨뜨린 것이 아니다). 상세 §11.

> **NPC-G** — **`tests/test_r2_correctness_instrument.py`의 `BANNER` 상수는 합성 픽스처다** —
> 905835의 `max_total_num_tokens=2618868` / `available_gpu_mem=13.38 GB`에 `context_len=16384`를
> 붙인 줄이며, **그런 배너를 낸 boot는 존재한 적이 없다.** 이 문자열을 측정값으로 인용 금지.

> **NPC-H** — **제출 전 "전체 CPU 회귀"의 합격 기준은 이 트랙(`test_r2_correctness_*`)으로
> 한정한다.** 전체 discovery의 실패는 병행 워크스트림의 진행 상태에 좌우된다. 상세 §13.

> **NPC-I** — **I3의 `λ_inf`는 legacy warm-up boot · D44 · cudagraph ON에서만 측정된다.**
> λ0의 사다리가 이 값을 앵커로 쓰므로 **λ0 결과 문서는 NP-9의 "legacy 루프 · warm-up boot
> 한정"을 그대로 승계해야 한다.** **true-dual arm(B4)의 포화 상한은 이 job에서 측정되지
> 않는다.**

> **NPC-J** — **§9(f)(벽시계 초과 → `R2C_INSTRUMENT=0` 재제출)는 §9-D5의 재제출 정책 열거
> 정의역 밖이다.** 벽시계 강제 종료는 라벨을 낳지 않으므로 라벨 쇼핑은 발생하지 않지만,
> **두 경로를 합쳐 총 재제출 1회로 센다.**

### (D13-i) 스코프 배선 — 결과 문서의 필수 전사 항목

`verdict.txt`에는 **모델도 백엔드도 인쇄되지 않는다**(라벨·체크·불일치 수만). 결과 문서가
그것만 인용하면 스코프 튜플이 사라진다(X1이 프로즈로만 동결됐던 선례). 따라서 결과 문서는
**`verdict.txt` 단독 인용을 금지**하고 다음 둘을 **함께 전사**한다:

1. `job_<id>/r2_correctness_report.json`의 `checks.server_args_across_boots`
   (`attention_backend` · `context_length` · `random_seed` · `max_running_requests` ·
   `pdmux_config_path` · `disable_cuda_graph` · `triton_attention_num_kv_splits`),
2. `job_<id>/provenance.txt`의 `model=` 줄과 `context_length=… attention_backend=…` 줄,
   그리고 `instrument=` 줄·`flashinfer_version` 줄.

---

## 12-b. (D9) 커밋 상태 — 제출 전 필수

판정서 D9는 "하네스·사전등록·테스트를 제출 전에 **커밋**하라"고 요구한다. 이유:
`provenance.txt`의 `commit=`은 `git rev-parse HEAD`이고 `src_dirty`는 **`src/`만** 본다 ⇒
`r2_correctness.sbatch`가 미커밋 수정본이고 `newpair_prereg/`·
`tests/test_r2_correctness_instrument.py`가 untracked이면, 기록된 커밋 해시가 **실제로 돈
하네스를 가리키지 못한다**(sbatch의 sha256은 기록되지만 그 해시가 가리킬 추적 객체가 없다).

**이 문서를 쓴 세션은 커밋하지 않았다** — 같은 저장소를 **병행 워크스트림 2개가 미커밋
상태로 편집 중**이라 선택 스테이징을 메인 세션이 통합 시점에 일괄 처리한다. **제출 전에
아래 4경로(파일 7개)가 커밋돼 있어야 한다**:

```
workspace/engine-port/results/r2_correctness/r2_correctness.sbatch          (M)
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh                 (M, D10)
workspace/engine-port/results/r2_correctness/newpair_prereg/                (??, 4 files)
workspace/engine-port/tests/test_r2_correctness_instrument.py               (??)
```

(`newpair_prereg/`의 **5파일** = `PREREG_NEWPAIR_2026-09-13.md` ·
`VERDICT_newpair_rules_2026-09-13.md`(1차) · `VERDICT_newpair_rev2_2026-09-13.md`(재감사) ·
`newpair_preflight_o1.py` · `newpair_preflight_o1_stdout.txt`. 따라서 2 추적수정 + 5 신규 =
**파일 7개, 경로 4개**.)
제출 시점의 `provenance.txt`가 이 커밋을 가리키는지 `commit=` 줄로 확인한다.

---

## 13. 제출 (감사 통과 후에만)

```bash
cd /scratch/ehmoon/whlee/prefill-layer-alloc && \
R2C_MODEL=nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base \
R2C_ATTN_BACKEND=flashinfer \
R2C_CTX=16384 \
R2C_DSM=44 \
R2C_INSTRUMENT=1 \
sbatch workspace/engine-port/results/r2_correctness/r2_correctness.sbatch
```

- `--comment`는 **CLI에서 주지 않는다**(스크립트 지시자 `#SBATCH --comment=
  "field=efficientai;appl=pytorch"`를 무효화해 거부된다).
- ★**(D11) `--time`도 CLI에서 주지 않는다.** rev1은 `sbatch --time=02:00:00`을 요구했는데,
  그것은 **빠뜨리면 조용히 망가지는 단일 인적 실패점**이었다(당시 지시자 기본값
  `01:15:00`으로는 계측 최악에서 scored boot이 잘린다). rev2는 **지시자 자체를
  `#SBATCH --time=02:30:00`으로 올렸으므로** 아무것도 덮어쓸 필요가 없다.
  ⚠️**경고 — 이 보호는 지시자에 있다**: 누군가 `#SBATCH --time`을 예전 값(`01:15:00`)으로
  되돌리면 CLI `--time`이 **다시 필수**가 되고, 빠뜨리면 01:15:00이 적용돼 계측 최악에서
  scored boot이 잘린다. 제출 전 `grep '#SBATCH --time' r2_correctness.sbatch`로 `02:30:00`을
  확인한다(테스트 `test_wall_limit_is_in_the_directive_not_the_command_line`이 고정).
  02:30:00으로도 §9의 **구조적 최악 161분**은 덮지 못한다 — 그 분기는 §9(f).
- `R2C_SEED`·`R2C_ORDER`·`R2C_NUM_KV_SPLITS`·`R2C_BASELINE_JOB`·`R2C_TRACE_FORCE_PREFILL`은
  **설정하지 않는다**(각각 1 / `L TD L TD` / 미설정 / 미설정 / 1).
- 산출물: `workspace/engine-port/results/r2_correctness/job_<id>/`
  (+ `job_<id>/instrument/`), stdout `r2corr_<id>.out`.

**검증 루틴(제출 전 이미 수행, GPU 0)**

```bash
module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
cd /scratch/ehmoon/whlee/prefill-layer-alloc
workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh   # 25줄 manifest (D10)
python -m unittest discover -s workspace/engine-port/tests -v  # 전체 CPU 회귀
HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 workspace/engine-port/results/r2_correctness/newpair_prereg/newpair_preflight_o1.py
grep '#SBATCH --time' workspace/engine-port/results/r2_correctness/r2_correctness.sbatch
```

기대값: manifest **25줄**(`flashinfer_backend.py` 포함) · preflight `PREFLIGHT_RC=0` ·
`#SBATCH --time=02:30:00`.

★**(NPC-H) 회귀의 합격 기준은 이 트랙으로 한정한다.** 저장소를 **병행 워크스트림 2개가
미커밋 상태로 편집 중**이므로 전체 discovery의 합/불합은 이 job의 제출 가부와 무관한
항목에 좌우된다(감사 시점: 572 tests, 실패 4건이 **전부** `test_lambda0_prereg.py` =
워크스트림 B의 rev3 진행 중). 기준을 적지 않으면 제출 시점에 "이 실패를 무시해도 되는가"라는
**미등록 재량**이 생긴다. 따라서 제출 전 합격 기준은 다음 하나다:

```bash
python -m unittest discover -s workspace/engine-port/tests -p 'test_r2_correctness_*.py' -v
```

**이것이 전부 통과해야 한다**(현재 `test_r2_correctness_ctx.py` 16 + 
`test_r2_correctness_instrument.py` 35 = **51**). 전체 discovery는 **정보 항목**으로만 기록하고,
실패가 있으면 **그 실패가 `test_r2_correctness_*` 밖임을 파일명으로 명시**한다.
