# R2 GPU correctness 재실행(job 907100) `VERDICT PASS`: 적대 결과 감사 판정서

- 일자: 2026-09-11. 감사자: claims-auditor(읽기 전용, 파일 수정 0, GPU 0).
- 대상: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/job_907100/`
  - commit `38c1aca`, src_dirty 없음.
  - 하네스 v2 sha `ec355e17…/95e10b49…/7ad119a0…`. `e16e93f` 커밋본과 바이트 단위로 같음을 확인했다.
  - runtime manifest가 907032와 다른 곳은 mixin 한 줄뿐이다(`fdea4c32…`→`0b88c07c…`, `874873b`와 일치).
- 방법: 체커의 판정 경로를 믿지 않고 원자료에서 직접 다시 계산했다.
  - 원자료: `gen_*.json`, `tel_*.jsonl`, `srv_*.log`, `green_*.json`.
  - 교차-잡 귀무대조: `.../job_907032/gen_L{1,2}.json`.
  - 코드 근거: `.../src/multiplex/{multiplexing_mixin,dual_worker,controller,green_readout}.py`, `sglang_engine_dev/.../triton_backend.py`, `.../scripts/r2_eval/*`, `.../benchmarks/pdmux_eval/{campaign,trace_loadgen}.py`.
- 등록 판정 PASS는 독립 재계산으로 재현된다. 불일치는 0건이다.
  - S 교차 쌍 4개 × 1,024 토큰, O 교차 쌍 4개 × 384 토큰, 모두 불일치 0.
  - 귀무 쌍 L1-L2, TD1-TD2도 0.

## §0 판정

제안 문장은 **그대로 정본에 쓸 수 없다.** 다만 PASS 자체는 **공허하지 않다**(§1). 스코프 튜플을 §7.1 문안으로 고치면 **`CONFIRMED(scoped)`**다.
- 확인되는 것: "S16 순차 + O8 단일-probe·단일-background 중첩 프로토콜에서, true-dual과 legacy의 greedy 토큰 ID가 전부 같다."
- 무한정 문장, 즉 동시 부하(C 층)까지 포함하는 "같은 토큰"은 **관측으로 `REFUTED`**다. C01·C10·C19 세 요청(3/32)이 L1=L2≠TD1=TD2 패턴을 보인다.
- 그 불일치를 **TD 결함으로 돌리는 것은 `NOT-YET-SUPPORTED`**다. C01은 triton KV-split 휴리스틱과 arm마다 다른 타이밍의 조합으로 산술 설명된다(§1.7).
- **Claim D 등급 영향은 0이다(미검증 유지).** 새 성능 판정 0건.

제안 문장에서 고칠 곳은 네 군데다.
1. "fixed D44": D44 분할 위에서 돈 비교 대상 계산은 **O probe의 prefill뿐**이다. probe decode는 비분할 그룹 idx 5에서 돌았고, S 층은 D44를 한 번도 타지 않았다.
2. "cudagraph-ON": **decode에 한정**된다. prefill은 두 arm 모두 eager다.
3. 빠진 스코프 조건: **TP=1, greedy T=0, ignore_eos, context 4096**. context 4096은 캠페인 러너 기본값 16384와 다르다.
4. "S16+O8"은 요청 수가 아니라 **프로토콜**로 적어야 한다. C 층이 제외된다는 것을 명시해야 한다.

## §1 공허성 검사

### 1.1 TD boot가 실제 true-dual 경로를 탔는가: 예(작업량이 기대값과 정확히 일치)

| 증거 | TD1 | TD2 | 기대값 / 해석 |
|---|---|---|---|
| true-dual banner | 있음 | 있음 | L1·L2에는 없음 |
| snapshot `architecture` | true_dual 9157/9157 | 9295/9295 | L은 전부 legacy |
| 최종 `decode_host_tasks` = "Decode batch" 로그 줄 수 | 2611 = 2611 | 2592 = 2592 | 모든 decode step이 decode worker 스레드에서 발행됐다 |
| S 창 `prefill_host_tasks` 증가 | 2→24 (22) | 22 | 16요청 + 다중-span 6 = 22. span 수 = ⌈54/⌊65536/L⌋⌉ |
| S 창 `decode_host_tasks` 증가 | 1008 | 1008 | 16×63 = 1008 |
| O 창 `prefill_host_tasks` 증가 | 26 | 26 | bg 8 + probe span 18(6×2 + 2×3) |
| O 창 `decode_host_tasks` 증가 | 1272 | 1272 | 8×159 |
| O4 probe별 Δprefill / Δdecode | 3–4 / 46–66 | 3–4 / 44–63 | Δprefill = span 수 + 1(H2 지연 아티팩트) |

⇒ S·O의 모든 prefill span과 decode step이 두 worker 스레드에서 실행됐다. "경로를 안 탔다"는 형태의 공허성(게이트 #9 계열)은 없다.

### 1.2 `gen_TD*`가 `gen_L*`의 복사나 캐시가 아닌가: 아니다

- 각 gen 파일의 요청 시각이 자기 서버 로그와 초 단위로 맞는다.
  - S00 시작: L1 19:41:32, TD1 19:43:07, L2 19:44:55, TD2 19:46:33.
  - 각 `srv_*.log`의 첫 S prefill(`#new-token: 6`)과 O00 probe(`#new-token: 1437`)가 같은 초에 찍혀 있다.
- 서버는 별도 프로세스·별도 포트였고, `--disable-radix-cache`로 모든 prefill 줄이 `#cached-token: 0`이다.
- 요청 지연이 boot마다 다르다(S01: L1 0.569 s, TD1 0.865 s).
- 결정적 증거: C 층 L1-TD1 불일치가 7/32다. 복사였다면 0이어야 한다. gen 파일 sha256도 4개 전부 다르다.

### 1.3 비교기·출력·입력

- **비교기:** `compare()`는 `output_ids`를 원소 단위로 비교하고 결측 id도 셉니다(`r2_correctness_check.py:339-348`). 이 데이터에서 실제로 작동한다. C 층에서 first_divergence 0·1·6·8·9·12·34·60·101을 모두 잡았다.
- **출력:**
  - 길이: S 16/16 = 64, O 8/8 = 48, C 32/32 = max_new. finish_reason은 전부 length.
  - 요청당 고유 토큰 수: S 12–53/64, O 37–42/48.
  - 출력끼리 서로 다름: S 16/16, O 8/8. B6 퇴화 0/56.
- **입력 동일성:** `prompt_sha256` 56/56이 네 boot에서 같다(직접 대조).
  - 하네스 결함 H1: 체커의 INPUT_IDENTITY는 `prompt_tokens` 개수만 비교하고 sha는 보지 않는다(`:350-353`). 이번에는 sha를 따로 확인했으므로 무해하다.
- **교차-잡 귀무대조:** job 907032(gpu42, commit `02918e8`)의 L1·L2 S 출력이 907100의 네 boot 각각과 16/16 같다.
  - 즉 S 층은 노드와 커밋을 넘어 재현된다.
  - 단 `874873b`의 legacy 구문 재배치가 출력을 바꾸지 않았다는 교차-잡 확인은 비중첩인 S에서만 된다. 907032에는 O 층이 없다.

### 1.4 O 층 "구성상 결정론" 전제의 확인

- O1 여유: probe prompt 1437–2619 토큰 > 7×(17+160) = 1239. 최소 여유는 198 토큰(16.0%)이다.
- 산술 확인(`triton_backend.py:1271-1316`):
  - bs=2 {bg, probe}에서 cdiv(max,min) ≥ 9이므로 `max_kv_splits_1`이 8에서 포화한다.
  - `chunk2`는 probe 길이와 token_grid(2×32)만의 함수다.
  - 따라서 probe 행의 split 수는 bg 진행과 무관하다. cudagraph bs-bucket 2는 고정이고, mamba decode는 행 단위로 독립이다.
- ⇒ O 층에서 불일치가 난다면 그것은 타이밍이 아니라 계산 차이다. 동치 게이트로서 설계가 타당하다.

### 1.5 동시성이 실제로 얼마나 있었나: 작다

`host_worker_overlap_ratio`에 경과 시간(런타임 시작 ≈ 첫 telemetry)을 곱해 두 worker 콜백의 벽시계 중첩을 복원했다.

| 층 | TD1 | TD2 | 비고 |
|---|---|---|---|
| S | 0.000 s | 0.000 s | 구성상 중첩 없음 |
| O | 0.0135 s | 0.0194 s | probe span 18개당 약 0.75–1.1 ms |
| C | 0.647 s | 0.080 s | |

- 지표 결함 H3: 구간 목록이 최근 1024개로 잘리고 분모가 서버 수명 전체라, O 값은 하한이다.
- GPU 수준 정황: probe prefill이 in-flight인 0.11–0.27 s 동안 bg decode가 9–23 iteration을 끝냈다(4 boot 전체 범위). decode가 prefill GPU 작업의 수명 안에서 진행됐다는 뜻이다. 커널 중첩 자체(nsys)는 측정하지 않았다.
- ⇒ O 층은 "같은 iteration에서 두 worker 동시 발행 + D44 두 green context 공존"을 **스모크 수준**으로만 보여 준다. 스레드 안전성 검증이 아니다.
  - confound #3 small-n: 동시 span 36개(2 boot × 18)에서 결함 0 → 경합 결함률 95% 상한 ≈ 3/36 = 8.3%/span.

### 1.6 검출력(민감도)의 한계

- O probe 출력 8/8은 PASSAGES 4-gram과 100% 일치한다. 즉 passage를 그대로 베낀 출력이다.
  - S는 긴 프롬프트 9/16이 0.98 이상, 짧은 프롬프트 7/16이 0.04 이하다.
  - 복사형 출력에서 argmax 마진이 작은 지점은 passage 경계(48 토큰당 1–2회)뿐이다.
- 복사형 출력도 뒤집힐 수 있다. C10·C02는 4-gram 1.00인데도 불일치했다. 그러니 민감도가 0은 아니지만 **보정되지 않았다.**
- e16e93f의 "항상-통과 변이본 검출"은 스코어러 층 검사다. 엔진+프롬프트라는 측정 층의 민감도를 보증하지 않는다.
- 따라서 PASS가 배제하는 것은 **argmax를 뒤집을 만큼의 상태 손상**(소유권 경합, KV/state 오염, 쓰레기 출력, 크래시)이다. 축약 순서 수준의 미세한 수치 차이가 없었다고는 말할 수 없다.

### 1.7 C 층(진단 전용): arm-분리 불일치와 기전 후보

네 boot 사이에서 출력이 갈린 요청은 32개 중 8개다.

| 요청 | 등가류 | first div / 비고 |
|---|---|---|
| C01 | {L1,L2} {TD1,TD2} | 12, arm-분리 |
| C10 | {L1,L2} {TD1,TD2} | 1, arm-분리 |
| C19 | {L1,L2} {TD1,TD2} | 6, arm-분리 |
| C00 | {L1} {L2} {TD1,TD2} | TD–L 34, L1–L2 60 |
| C02 | {L1} {L2,TD1,TD2} | |
| C14 | {L1,L2,TD1} {TD2} | |
| C16 | {L1} {L2,TD1} {TD2} | |
| C24 | {L1,TD2} {L2,TD1} | 교차 |

**등록 규칙과의 관계**
- 등록 규칙이 C를 강등한 근거는 옳다. 층 단위로 L-L이 4/32로 0이 아니다.
- 그러나 같은 조건을 **요청 단위로** 적용하면 C01·C10·C19에서 발화한다. 이것은 등록 범위 밖이므로 판정의 死因은 아니고, 주장 스코프를 제한하는 사유다.

**C01 기전 후보(산술 재구성)**
- C00(60 토큰)이 C01 합류 전에 혼자 decode한 step 수가 **L 6·6, TD 4·4**다. arm 안에서는 재현된다.
  - 근거: srv 로그 `#full token` 열. L: [61,62,63,477,478,479], TD: [61,62,476,477].
- 그래서 C01 첫 decode 시점의 C00 길이가 L 67, TD 65다.
- 이를 triton 휴리스틱에 넣으면, bs=2 {C00, C01+k}에서:
  - C01 행 split 수가 k=3,4에서 L 6, TD 7로 다르다.
  - C00 행 split 수는 k=3,5에서 다르다.
- 즉 decode attention의 축약 분할이 arm마다 달라지고, 그 차이가 SSM 재귀 상태로 전파된다. token 12 반전은 이 기전과 정합한다. 다만 logits를 측정하지 않았으므로 인과는 미입증이다.
- C10·C19는 재구성하지 못했다. prefill batch 구성이 arm 안에서도 달라서다. 예: C10은 L1 (7, 6934) offset 429 / L2 (3, 2618) / TD1 (8, 6989) / TD2 (4, 5065).

**통계적 강도**
- 2-2 분할 4건 중 arm에 정렬된 것이 3건이다. 균등 귀무에서 P(≥3/4) = 9/81 ≈ 0.11이다(사후 계산, 서술용).

**결론**
- "C 부하에서 TD와 legacy 토큰이 같다"는 반증된다. "TD 결함"은 지지되지 않는다.
- legacy끼리도 4/32가 불일치하므로, 동시 부하에서의 토큰 동일성은 이 엔진(비 batch-invariant 커널)에서 정의되지 않는 양이다.
- r2_eval 성능 비교는 오염되지 않는다. trace_loadgen이 ignore_eos와 고정 max_new_tokens를 쓰므로(`trace_loadgen.py:55-57`) 작업량은 arm과 무관하다.

### 1.8 반전 시험 표(등록 판정의 자유 표면)

| 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 |
|---|---|---|---|
| C 층 역할 | 907032 이전에 선언된 "L-L 동일 시에만 cross 계수"를 층 단위로 적용 | 없음 | L1-L2 C 4/32 ≠ 0 |
| 같은 조건을 요청 단위로 적용(등록 범위 밖) | — | FAIL로 반전 | C01·C10·C19 → 死因 아님, R2C-2로 전환 |
| B5 safe-필터 | "모든 결정이 D 표적"으로 강화 | 없음 | unsafe 60/60이 target=44 |
| B6 문턱 | 1/2 → 0 | 없음 | 0/56 |
| B8/O3 overlap 정의 | "probe 창 안 전 스냅샷"으로 확장(범위 밖) | FAIL로 반전 | probe decode 스냅샷이 idx 5에 21–24개/boot → 死因 아님, §2.3으로 전환 |
| O1 여유 | — | 없음 | 최소 198 토큰 |
| O4 문턱 | ">0" → "≥ span 수" | 없음 | Δprefill − 1 = span 수 2/3 |
| INPUT_IDENTITY | 개수 → sha | 없음 | 56/56 |

- 반전 계산은 원자료에 직접 적용했고 체커 코드는 쓰지 않았다.
- 등록 범위 안에서 판정을 뒤집는 자유도는 없다.

### 1.9 confound 카탈로그 대조

| # | 해당 여부 | 내용 |
|---|---|---|
| 1 sim→serving | 아님 | 서빙 직접 측정 |
| 2 미조율 오귀속 | 아님 | 정책 주장이 아님 |
| 3 small-n | 해당 | 스레드 안전성으로 해석할 때(boot 2개/arm, 동시 span 36, 전환 140) |
| 4 max 분모 | 아님 | |
| 5 metric cliff | 아님 | |
| 6 재스코어 | 아님 | 규칙 v2는 TD 출력이 나오기 전에 고정됐다(907032 TD 출력 0). 907032를 v2로 재채점해도 FAIL 유지 |
| 7 micro/GIL | 부분 해당 | S/O는 마이크로 프로토콜이고 host 중첩은 ms 단위 → 서빙 부하 동치 주장 불가 |
| 8 no-cudagraph | 아님 | decode 10,400줄 전부 `cuda graph: True` |
| 9 positioning | 아님 | |
| 10 변수 동시 변경 | 해당(앞으로의 비교) | TD는 스레딩과 `safe` 술어 평가를 함께 바꾼다(§3.5) |

## §2 D44 실현

**2.1 엔진의 선택**
- 전 boot의 overlap 스냅샷이 전부 idx 4였다: 600/600, 558/558, 536/536, 532/532.
- probe별로도 overlap 스냅샷 20–48개가 전부 idx 4였다.
- 각 probe마다 `split_transition` 108→44(reason fixed, safe)가 정확히 1회 있었다(4 boot × 8/8).

**2.2 드라이버 read-out**
- idx 4의 두 stream에 대해 `cuStreamGetGreenCtx` rc 0, `green_ctx_is_null` false, `cuGreenCtxGetDevResource` rc 0이다.
- smCount는 prefill 64, decode 44다. 이 값은 target 필드와 다른 채널에서 온다(`green_readout.py:91-102`).
- boot마다 stream 포인터가 달라서, 각 boot가 자기 read-out을 가진다.
- ⇒ "target이 아니라 realized"(게이트 #1)를 **할당 수준**에서 충족한다.

**2.3 비교 대상 계산 중 D44 위에서 돈 것**
- O probe 창 안 스냅샷:
  - prefill in-flight(bs 1 = bg): idx 4에 276/242/266/230개(L1/TD1/L2/TD2).
  - probe decode(bs 2): idx 5에 24/23/24/21개. merge 직후 idx 4 스냅샷이 0/1/0/3개 더 있다.
- S 창: prefill은 idx 0, decode는 idx 5. D44는 0회다.
- ⇒ 정리하면:
  - **D44 64-SM prefill 측에서 돈 것은 probe prefill(2–3 span)이다.**
  - **D44 44-SM decode 측에서 돈 것은 비교에서 제외된 bg decode뿐이다.** bg 출력은 4 boot 8/8 같지만, 숫자 세기 텍스트라 민감도가 낮은 진단용이다.
  - probe의 decode 47 step은 비분할 plain stream(idx 5, 0/108)에서 돌았다.

**2.4 "동시"의 수준**
- O2가 말하는 동시성은 스케줄러 상태다(`split_prefill_batch` 비-None ∧ `running_batch` 비-공). GPU 정황은 §1.5에 있다.

**2.5 변별력 한계**
- `pdmux_r2.yml`의 문턱이 전부 0이라, 네이티브 선택기(`multiplexing_mixin.py:936-940`)도 overlap 시 마지막 분할 idx 4(= D44)를 고른다.
- 그래서 D44에서는 B8/O3의 idx 4 검사가 FixedPolicy 액추에이션과 네이티브 선택을 구별하지 못한다.
- FixedPolicy 경로를 탔다는 증거는 `split_transition`(reason fixed) 이벤트다. B8이 변별적인 것은 D16/24/34에서뿐이다.

**2.6 토큰 동치는 파티션 오결합을 보지 못한다**
- decode KV-split 휴리스틱은 장치 SM 108(`get_device_core_count`)을 쓰며 green context와 무관하다.
- 따라서 TD 스레드가 엉뚱한 stream에 발행했더라도 토큰은 같았을 것이다.
- "D44에서"라는 한정을 떠받치는 것은 텔레메트리와 read-out이지 토큰 동치가 아니다. 커널 수준 배치(%smid)는 측정하지 않았다(S3/G1-d 미종결 상태 불변).

## §3 `unsafe_decisions=30`

### 3.1 무엇이 기록됐나

| boot | O: safe @108 | O: @44 | C: @108 | C: @44 |
|---|---|---|---|---|
| L1 | 8 | 10 safe | 1 | 26 safe |
| TD1 | 8 | **10 unsafe** | 2 | 20 unsafe + 6 safe |
| L2 | 8 | 10 safe | 2 | 27 safe |
| TD2 | 8 | **10 unsafe** | 1 | 20 unsafe + 8 safe |

- unsafe 60/60 전부 current_decode_sms = target = 44다. `blocked_unsafe_transition`의 requested도 44다. **전부 아무것도 막지 않은 no-op이다.**
- non-D44 split을 붙잡은 순간은 0건이다. B8(overlap 100% idx 4) 통과와 정합한다.
- `blocked_unsafe_transition`이라는 이벤트 이름은 과장이다. requested와 current가 같아도 발행된다(`:464-473`).

### 3.2 기전

- FixedPolicy는 unsafe 경계에서 target = current를 돌려준다(`controller.py:87`).
- TD에서 `safe`는 `arbiter.safe_to_switch()`로, in-flight CUDA 이벤트를 query한다(`multiplexing_mixin.py:432-435`).
- unsafe 60/60의 직전 스냅샷이 모두 `pending_gpu_events=1`, `active_leases=0`, prefill 진행 중(split_index>0)이다.
- O 층의 10건/boot는 probe의 비-첫 span 수(6×1 + 2×2)와 정확히 일치한다. 즉 앞 span의 prefill 이벤트가 아직 GPU에서 실행 중일 때 다음 span의 결정이 내려진 것이다.
- legacy는 `true_dual_worker_runtime is None` 조건에서 단락되어 이 술어를 평가하지 않는다(상수 True). 그래서 같은 결정 지점이 legacy에서는 safe로 기록된다.
  - @44 결정 수: L1 36 대 TD1 30+6 = 36, L2 37 대 TD2 30+8 = 38.

### 3.3 커밋 ea99191 문구 정정

- 커밋 문구 "since B8/O3 confirmed the realized split was D44"는 근거가 부정확하다.
- 무해하다는 직접 근거는 controller_decision 텔레메트리(target = current = 44)다.
- B8/O3는 표본 기반이다. 단일-span prefill이 query 시점에 이미 끝나면 강제 스냅샷이 남지 않는 경로가 있다(`:567-580`, `:1269-1294`).

### 3.4 lease-release 지연과는 무관하다

- `active_leases` 최댓값은 TD 스냅샷 18,452개 전부에서 0이다.
- TD 아비터 전환 71+69 = 140회 동안 RuntimeError(`:1100-1103`, `:1117-1120`)는 0회였다.
- 잠재 위험이 이번에 발화하지 않았을 뿐이다(0/140, rule-of-three 상한 약 2.1%/전환). 해소된 것이 아니다.

### 3.5 Claim D 함의(관찰로만)

- 캠페인의 B1(legacy fixed 44) 대 B4(true_dual fixed 44) 쌍(`campaign.py:79-84`)에서는 이 차이가 행동에 영향을 주지 않는다.
- generic/hybrid 정책(B5·B6)에서는 문제가 된다.
  - TD는 prefill 도중의 전환을 unsafe로 막는다. 이 워크로드에서는 결정의 30/46 ≈ 65%다.
  - legacy는 drain한 뒤 전환한다(`:1094-1109`).
  - 즉 아키텍처와 액추에이션 기회가 함께 바뀐다(confound #10).
- 그런 비교를 할 때는 requested≠current인 `blocked_unsafe_transition` 수를 필수 공변량으로 넣어야 한다.

## §4 Claim D 선결 상태표

선결 1–5는 `r2_decoupling_review_2026-07-24.md`의 목록이다.

| # | 선결 | 이 job이 닫는가 | 근거 / 남은 것 |
|---|---|---|---|
| 1 | admission latch stale-True 수정 | **아니오** | 코드 수정(`02918e8`)만 있다. FixedPolicy라 `admission_limited`는 185개 결정 중 0개, r2_admission 이벤트 0개로 latch 경로가 한 번도 안 탔다 |
| 2 | GPU correctness 동치 | **부분** | S/O 프로토콜, Zamba2-2.7B, TP=1, D44, greedy, seed 1, ctx 4096에 한정. 동시 부하, D16/24/34(B5·B6가 방문), generic/hybrid, 다른 모델, T>0, TP≥2, 러너 설정은 미포함 |
| 3 | Claim D 주장 범위 | 무관(불변) | control-plane 스코프, coupled ceiling(정본 술어 하 미정의) 그대로 |
| 4a | server seed arm 간 동일 | 이 하네스에서는 예 | random_seed=1이 4/4. 캠페인 쪽은 `shared_seed`(1000+rep)라는 코드 사실만 있고 실행 확인은 안 됐다 |
| 4b | observer effect <3% paired on/off | **아니오** | 두 arm 모두 관측 플래그 ON이고 OFF 짝이 없다 |
| 5 | cudagraph-ON 호환 | **예(scoped)** | decode 10,400/10,400줄이 True. TD의 decode_host_tasks가 decode 줄 수와 같아 decode worker 스레드에서 graph replay가 이뤄졌다. capture end 1회/boot |
| N1 | 러너 context-length 16384 부팅 실패 | 아니오 | 이 PASS는 ctx 4096 편차 아래에서 얻은 것이라 러너 설정은 인증되지 않았다 |
| N2 | `r2_eval.sbatch` dirname | 아니오(무관) | `r2_eval.sbatch:11`에서 BASH_SOURCE가 slurm spool을 가리킨다 |
| N3 | lease-release 지연 잠재 위험 | 아니오 | 0/140 무발화. `dual_worker.py:490`의 done-callback 해제 경로는 미수리 |
| N4 | `forward_ct`/`forward_pass_id` 비원자 증가 | 아니오(토큰에 영향 없음) | `scheduler.py:2667`, `model_runner.py:2757`. 소비처는 프로파일러·로그·비-MoE EPLB라 이 설정에서 토큰에 영향이 없고, 이 PASS로는 검출할 수 없다 |
| N5 | thread-local role(ContextVar) 거동 | 아니오 | TP=1에서는 역할이 틀려도 수치에 영향이 없다. 존재와 fail-fast만 확인됐다 |
| N6 | 동시 부하 토큰 동일성 | 아니오 | §1.7 |
| N7 | 동적 정책 액추에이션 비대칭 | 아니오 | §3.5 |

Claim D 증거로 오인되지 않게 쓸 문장(문자 그대로 승계):
> "job 907100 PASS는 true-dual 코드 경로의 기능 동치(correctness) 인증이다. 성능·coupling·goodput은 전혀 측정하지 않았다. Claim D 등급은 미검증 그대로이다. 이 결과가 닫은 Claim D 선결은 #5(cudagraph-ON 호환, scoped) 하나이고, #2는 S/O 프로토콜 범위에서 부분 해소됐을 뿐이다."

## §5 인용 금지·필수 병기

**인용 금지(문자 승계)**
- R2C-1 스코프 없이 "true-dual은 legacy와 같은 토큰을 산출한다". 허용형: "S16 순차·O8 단일-probe 중첩 프로토콜에서 토큰 동일".
- R2C-2 "동시/서빙 부하에서 TD ≡ legacy". C 층에서 3/32가 arm-분리로 불일치한다.
- R2C-3 "C 층 불일치는 TD 결함이다"와 "C 층 불일치는 무해한 노이즈다". 둘 다 금지한다. 원인이 확정되지 않았다.
- R2C-4 무한정 "D44에서 동치". 허용형: "D44 64-SM prefill 측에서 돈 probe prefill을 포함한 산출이 동일. probe decode는 idx 5에서 돌았고, 44-SM decode 측 산출은 비교 대상이 아니다."
- R2C-5 "토큰 동치가 D44 실현을 확인한다". 동치 검사는 파티션 오결합을 보지 못한다(§2.6).
- R2C-6 "unsafe 30은 D44 이탈 또는 유지 실패의 기록이다". 60/60이 target = current = 44다.
- R2C-7 "unsafe 0 대 30은 arm 간 거동 차이다". legacy는 그 술어를 평가하지 않는다.
- R2C-8 "unsafe 30은 lease-release 지연의 증거다". 60/60이 `pending_gpu_events=1`, `active_leases=0`이다.
- R2C-9 "B8/O3가 unsafe의 무해성을 확인했다"(ea99191 문구). 근거는 decision 텔레메트리다.
- R2C-10 "thread-local role이 GPU에서 검증됐다". TP=1이다.
- R2C-11 "스레드 안전성·경합 부재가 검증됐다". O 층 host 중첩은 13.5/19.4 ms(하한), S 층은 0이다.
- R2C-12 "lease-release 위험이 없다". 0/140 무발화일 뿐이고 상한은 약 2.1%/전환이다.
- R2C-13 "r2_eval 설정이 인증됐다". ctx 4096, decode-log-interval 1, 관측 플래그가 러너와 다르다. 러너는 부팅이 안 된다.
- R2C-14 무한정 "Claim D 증거" 또는 "Claim D 선결 해소".
- R2C-15 "cudagraph-ON 전 경로". decode에 한정된다.
- R2C-16 "D44 PASS가 FixedPolicy 액추에이션을 변별 검증했다". 네이티브 선택기도 idx 4를 고른다.

**필수 병기**
- P-1 스코프 튜플: {Zamba2-2.7B, TP=1, pdmux_r2.yml fixed D44, cudagraph-ON(decode), greedy T=0·ignore_eos, seed 1, triton attention/mamba, ctx 4096, max-running-requests 48, chunked-prefill -1, radix off, boot 2개/arm, commit 38c1aca}.
- P-2 D44에서 돈 계산이 어떻게 나뉘었는지(§2.3).
- P-3 C 층 수치: L-L 4, TD-TD 3, cross 4–7. 그중 3/32(C01@12, C10@1, C19@6)가 arm-분리이며 원인은 미확정.
- P-4 unsafe 기전 한 줄: 앞 span의 prefill 이벤트가 in-flight인 상태에서 결정 → target = current = 44인 no-op.
- P-5 host 중첩 크기(§1.5).
- P-6 하네스와 러너 사이의 편차.
- P-7 새 성능 판정 0건. Claim D 등급 불변(미검증). HE0·정책 순위·stake #1 불변.

## §6 방법론 게이트 후보(기존 #1–166과 대조함)

- **G-1(신설) 진단 층으로 강등해도 주장 스코프에서는 빠지지 않는다.**
  - 층 단위 귀무대조 실패(L-L≠0)가 단위 수준의 arm-분리(L1=L2≠TD1=TD2)를 가릴 수 있다.
  - 강등된 층의 요청별 등가류를 병기하고, PASS 문장은 통과한 층으로 스코프를 한정하라.
  - 중복 대조: #17(모든 보고 블록에 게이트를 걸어라)과 인접하지만, #17은 수치 누출이고 이것은 판정 라벨의 스코프 누출이다. #20·#24·#142와는 무관하다.
- **G-2(신설, #20 인접) 같은 이름의 텔레메트리 필드가 arm마다 다른 술어로 채워질 수 있다.**
  - 비교하기 전에 두 arm에서 같은 술어가 평가되는지 코드로 확인하라.
  - 사례: `safe`(legacy는 단락되어 상수).
  - #20의 "식별자 수입 ≠ 거동 수입"은 분석기 쪽 문제이고, 이것은 엔진 쪽 필드 의미 문제다.
- **G-3(#50의 측정 층 판본) 동치 게이트의 검출력은 스코어러 변이 테스트로 보증되지 않는다.**
  - 알려진 미세 수치 섭동이 불일치를 1건 이상 만드는지, 측정 층 양성대조를 사전등록하라.
  - 사례: O probe 출력 8/8이 passage 그대로의 복사.
- **G-4(신설) 워커 경로를 탔다는 것(태스크 카운터 증가)은 동시 실행의 증거가 아니다.**
  - 동시성은 중첩 시간과 경합 기회 수로 적어라.
  - 사례: O4는 충족됐지만 host 중첩은 13.5/19.4 ms. `host_worker_overlap_ratio`는 1024개 절단과 수명 분모 때문에 동시성 지표로 부적합하다(engine-porter로 이관).
- **G-5(#1·#114에 追記) 파티션 실현은 "창 안에 D 분할이 있었다"가 아니라 "비교 대상 산출의 어느 계산이 그 분할 위에서 돌았는가"로 적어라.**
- **G-6(#95 사례 追記) 검증 표적이 설정의 기본 동작과 같으면 실현 검사는 두 가설을 구별하지 못한다.**
  - 기본값이 아닌 표적(D16)을 함께 돌려라.
- 하네스 결함(게이트 아님, engine-porter로 이관):
  - H1: INPUT_IDENTITY가 개수만 비교한다.
  - H2: `task_count`가 `set_result` 뒤에 증가해서 Δ에 +1 지연이 섞인다.
  - H3: overlap ratio의 절단·분모 문제.

## §7 정본 등재 문장 초안

### 7.1 CLAIM_EVIDENCE_MATRIX Claim D 행, "GPU correctness 동치 테스트(현재 없음)" 교체안

> "GPU correctness 동치: job 907100(2026-09-11, 약 0.16 GPU-h, commit 38c1aca, 하네스 v2 판정 규칙 사전 고정) `PASS`, claims-auditor 결과 감사 `CONFIRMED(scoped)`. 조건은 {Zamba2-2.7B, TP=1, pdmux_r2.yml fixed D44, cudagraph-ON(decode 한정, prefill은 두 arm 모두 eager), greedy T=0·ignore_eos, seed 1, ctx 4096(러너 기본값 16384와 다름), boot 2개/arm}이다. 이 조건에서 S16(순차, plain group idx 0/5)과 O8(단일 probe·단일 bg 중첩, probe prefill만 D44 64-SM 측, probe decode는 비분할 idx 5) 프로토콜의 greedy 토큰 ID가 legacy와 전부 같았다(교차 4쌍 × 1,408 토큰 불일치 0, 귀무 L-L·TD-TD 0). 동시 부하(C, 진단 전용)에서는 같지 않았다. L-L 4/32, TD-TD 3/32, cross 4–7/32이고, 그중 C01·C10·C19 3건은 L1=L2≠TD1=TD2이다(원인 미확정, C01은 triton KV-split 휴리스틱의 arm-특이 타이밍으로 산술 설명됨). 닫힌 선결은 #5(cudagraph-ON 호환, scoped), 부분 해소는 #2다. 미해소: 동시 부하 동치, D16/24/34, generic/hybrid, 다른 모델, TP≥2, 러너 설정(ctx 16384 부팅 실패, dirname), observer effect, latch의 GPU 발화, lease-release 잠재 위험. 성능 판정이 아니며 Claim D 등급은 불변(미검증)이다."

"주장 제한" 절의 "R2는 GPU correctness gate를 통과한 이력이 없다(…true_dual telemetry 전무)" 교체안:
> "R2 true-dual은 job 907100에서 GPU correctness gate를 S/O 프로토콜 한정으로 통과했다. `architecture=true_dual` telemetry가 처음 생성됐고, 두 worker 스레드가 S/O 작업을 전량 처리했다. `results/r2_eval/`은 여전히 생성되지 않았고, 캠페인 러너는 Zamba2-2.7B를 부팅하지 못한다."

### 7.2 CONSENSUS / PROJECT_STATUS 등재 한 줄(구현 사실, 성능 판정 아님)

> "R2 true-dual GPU correctness: job 907100 PASS, `CONFIRMED(scoped)`, S/O 프로토콜 한정. 인용 금지 R2C-1…16과 필수 병기 P-1…7을 판정서에서 문자 그대로 승계한다. 새 성능 판정 0건, Claim D 등급 불변, HE0 불변."

### 7.3 스코프를 넓힐 확정 실험(실현 가능성 검사 포함)

- **X1: 민감도 양성대조 + C 기전 판별(약 0.16 GPU-h)**
  - 하네스 v2를 그대로 쓰고, 네 boot(L TD L TD) 모두에 `SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS=true`만 추가한다.
    - 이 변수는 `triton_backend.py:111-113`에서 백엔드 초기화 때 읽는다. `fill_`만 하므로 cudagraph와 호환된다.
    - sbatch는 `SGLANG_TRITON_*`를 unset하지 않는다. 대신 provenance에 이 env를 기록해야 한다(하네스 sha가 바뀜, 판정 규칙은 불변).
  - 실행 전에 적어 둘 예보:
    - (i) job 안 S/O는 PASS.
    - (ii) 907100 대비 S/O에서 불일치 1건 이상. 이것이 측정 층 양성대조다. 0/24면 PASS는 "대형 손상 부재"만 인증한다고 등재한다.
    - (iii) C01의 arm-분리가 사라진다. 사라지지 않고 C L-L = TD-TD = 0인데 cross>0이면, TD 고유 원인으로 보고 engine-porter로 올린다.
  - job 사이 비교는 사전에 진단 전용으로 선언한다.
- **X2: 비기본 split(약 0.16 GPU-h)**
  - `R2C_DSM=16`. idx 1 = (92,16)이 `division_index`로 해석된다.
  - B8이 FixedPolicy를 변별하게 되고, B5/B6가 방문하는 split까지 스코프가 넓어진다.
- **X3: 러너 설정 인증**
  - 러너의 ctx와 dirname을 고친 뒤, 러너의 서버 인자 튜플로 하네스를 1회 돌린다(관측 플래그 ON).
  - 관측 플래그를 OFF로 둔 토큰-전용 대조를 진단으로 병기한다. 이것이 선결 4b의 토큰 쪽 절반이다.
- **GPU 없이:**
  - 동적 정책 비교의 사전등록에 unsafe 차단 수를 공변량으로 넣는다(§3.5).
  - lease-release 경로를 콜백 지연 주입 CPU 결정론 테스트로 확인한다(engine-porter).
