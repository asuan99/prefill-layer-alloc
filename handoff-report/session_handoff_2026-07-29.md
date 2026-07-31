# 세션 핸드오프 — 2026-07-28 ~ 2026-07-29

## 이번 세션 요약

전 세션이 블로커로 남긴 **claims-auditor 사전등록 게이트**를 집행해 **부분 GO**를 받고, 그에
따라 **정본 8문서를 개정**했다(Stage 0 D108 앵커 철회 = C1 CONFIRMED, 8B decode-SM 민감도를
scoped 측정 노트로 정본 진입 = C2 CONFIRMED, C2b는 NOT-YET-SUPPORTED로 강등, **Claim A 등급
불변**). 이어 사용자 지시로 **prefill 축 SM 민감도 캠페인**을 신규 구축·실행해 판정까지
마쳤고(REAL scoped), **E1 8B 프론티어 캠페인** 하네스를 구축했다. E1은 스모크 4회를 돌리며
**측정 방법론 결함 5종을 순차 발견·수정**했고 — 전부 "무엇을 세는가"의 문제였다 — 마지막
결함을 없애기 위해 **엔진 패치**(prefill-active iteration 강제 추적)까지 갔다. 본 스윕은
**미실행**이며, 용량 스캔(867231)과 관측자 효과 게이트(867298) 두 job이 PENDING 상태로
다음 세션에 인계된다.

---

## 결정·측정

### 1. claims-auditor 사전등록 게이트 집행 → 부분 GO

`s0_deconfound/DESIGN.md` §5 사전등록에 따라 `PARTITION_RESIDENCY_STAGE0.md` +
`FINDINGS_8B_2026-07-28.md`를 **함께** 감사.

| 주장 | 판정 |
|---|---|
| **C1** Stage 0의 D108 앵커는 실제로 decode 16 SM ⇒ `D16≡D108=1.00±0.01` 무효 | **CONFIRMED** |
| **C2** prefill 16 SM 고정 시 decode ITL SM16→SM92 = 2.36–2.91×, 4 arm 모델-무관 | **CONFIRMED (scoped)** |
| **C2b** "hybrid 급락 = Zamba2 additive 성질"(Hs8/M8=0.86) | **NOT-YET-SUPPORTED** |

- C1은 3중 독립 증거. 그중 하나는 telemetry를 전혀 쓰지 않는다 — `D108/D16=0.992–1.001`(9/9
  셀)인데 D92는 3.4–3.6× 빠르다(108 SM이 92 SM보다 느릴 수 없다).
- C2의 사전지정 반증축 (ii)**활성-구간 표집 편향**을 감사자가 **실제로 시도해 실패**시켰다:
  활성률이 2.5–2.8× 다른 두 job(865493 v2 / 865533 v3)을 분리해도 기울기 **≤1% 변동**.
  clock 축도 시도·실패(d16–d92 전부 1394–1410 MHz; 예외는 np/SM108만).
  (i)ctx1024 한정은 **부분 해소**(엔진측 `measured_itl_ewma_ms` 프록시로 ctx4096서 유지).
  **(iii)goodput 전이는 미해소** — 프론티어 전엔 CONFIRMED 아님.
- C2b는 근거가 주장을 지지하지 않는다: weight-traffic 추정에 Hs8이 없고, Nemotron-H-8B가
  Codestral-7.3B보다 파라미터가 많은데 빠르다는 건 그 서사와 방향이 반대. backend offset
  근거도 20초 스모크 n=1.

### 2. prefill 축 SM 민감도 (신규 캠페인 `results/s8p_prefill/`)

jobs **865738**(스모크) · **865973**(M8,Ha8) · **865974**(Hs8,T8). 판정 전문 =
`results/s8p_prefill/FINDINGS_PREFILL_2026-07-29.md`. **정본 아님**(claims-auditor 미통과).

decode 캠페인의 거울상: decode를 16 SM에 고정하고 `[P,16,idle]`, P∈{16,24,44,92}를 스윕.

- **1차 지표를 절대 TTFT 비가 아니라 `TTFT ~ L` 회귀의 기울기 비로 설계**했다. 합성 데이터
  검증: 주입 기울기비 3.33×를 회귀는 3.33×로 복원했으나 **절대비는 L512서 1.67×, L4096서도
  2.77×** 에 그쳤다(상수항 = admission wait + decode step + detok이 비율을 1 쪽으로 끌어당김).
  절대비를 헤드라인으로 썼으면 최대 2배 과소보고했을 것.
- **헤드라인**(rep별 OLS, n=4): M8 **5.163×** / Ha8 5.016× / Hs8 4.860× / T8 4.740×.
  탄력도 ε = **0.89–0.94** ⇒ prefill은 SM에 거의 선형 민감이고 **4 arm 사실상 동일**
  (decode 축의 "모델-계열 무관"이 prefill 축에서 재현). SM 5.75× 증가에 병렬 효율 손실
  11–21%뿐. **pure Mamba2 포함.**
- **곡률**(이 캠페인 고유 신호): T8만 확증(seg3/seg1 = **1.378±0.049**, 4셀 전부 >1.31,
  rep 4/4 단조). M8 평평(0.992, 해상도 ±16%). hybrid 서열은 셀 간 산포보다 격차가 작아
  **판정 불가**. ★**기전 귀속 미성립** — Qwen2.5-7B에서 L=4096 attention은 선형층 FLOP의
  6.3%라 예측 곡률 1.07인데 관측 1.38(**5배**). "O(L²)의 직접 서명"으로 쓸 수 없다.
- **게이트 11/16 PASS**(실패 5개 전부 0.665–0.80 경계). **4/4 PASS는 M8뿐.** 탈락 셀은
  폐기도 게이트 하향도 하지 않고 **구간 내 전 스냅샷이 target인 strict 필터**로 직접 검사
  (exploratory 라벨). T8 p92에서 2.6% 차이가 실제로 잡혀 FAIL 태그가 실질적 의미를 가졌다.
- **batch 공변 통제됨**(셀 평균은 공변하나 구간 귀속 후 소멸, batch-matched 재계산에서 비
  변화 ≤1.6%). **절편 전제는 반증**(4 arm 전부 `LARGE SWING` 48–96%) — 단 절편에 의존하지
  않는 chord 추정량이 ≤3%로 일치해 헤드라인은 견고.
- ★★**두 게이트가 동일 사건**: `prefill_sms == target` ⟺ `decode_running_batch_size > 0`가
  스냅샷 단위로 동치(telemetry 18개 전부 교차표 비대각 **정확히 0**). 소스가 이유를 설명한다
  — `stream_index`는 `running_batch`가 비지 않을 때만 분할 인덱스이고
  (`multiplexing_mixin.py:726,745-748`), `decode_running_batch_size`는 **같은 `running_batch`**
  를 읽는다(`dual_worker.py:619`). ⇒ 사전등록 강도가 게이트 1개분이고, co-residency는
  확인된 게 아니라 **정의**가 됐다.

### 3. ★ 반증된 추론 — "prefill이 더 가파르니 긴장 A의 기전"

세션 중 내가 제시한 "prefill 5.17× > decode 2.36–2.91× ⇒ 프론티어서 prefill 손실이 더
가파름"은 **지지되지 않는다**(FINDINGS §7). 이유:

- **끝점 비는 프론티어 결정량이 아니다.** 정책은 SM을 5.75× 옮기지 않고 1개씩 옮긴다.
- **국소 탄력도 구조가 완전히 다르다**: prefill은 전 구간 ε≈0.9로 평평, **decode는 D≳44에서
  급격히 포화**(44→92 구간 ε = 0.09–0.35).
- ⇒ SM 1개를 옮길 때의 **부호가 동작점에 따라 뒤집힌다**([P44,D64]에선 prefill 3.6–11.3×
  우세, [P84,D24]에선 decode 우세 0.29–0.43×). **내부 균형점이 존재한다.**
- 단위·모집단도 다르고(프롬프트 토큰당 vs 출력 토큰당) 워크로드 shape로 가중하면 방향이
  또 바뀐다.

정당하게 남는 건 마이크로 서술 하나: **prefill이 자기 SM에 더 탄력적**(0.89–0.94 vs decode
0.48–0.56). **긴장 A는 프론티어 직접 측정으로만 풀린다.** 본 캠페인의 기여는 설계 입력:
"프론티어 스윕은 decode 포화 구간(D≳44)과 prefill 비포화를 반드시 같이 덮어야 한다."

### 4. E1 8B 프론티어 하네스 (신규 캠페인 `results/s8_frontier/`)

감사자 사전등록 사양대로 구축: `[108−D, D]`, D∈{16,24,44,54,92} + best-static 대조, 4 arm,
**offered-rate open-loop**, 용량 선측정 후 off-cliff rate, conjunctive goodput + TTFT
p50/p95/p99 + request-내부 ITL p95, n≥4 paired bootstrap.

**사전등록 결정 규칙**: 어떤 D가 best static을 conjunctive goodput에서 **≥3%** 이기고
**paired CI가 0을 배제**하는가. 없으면 **C2는 "ITL 레버는 있으나 예산 제약 하 net-negative"로
확정**되고 긴장 A가 그 방향으로 닫힌다.

**SLO 사전등록**(사용자 지적으로 개정): 1차 **ITL-p95 = 60ms 고정**(근거는 데이터 적합이
아니라 `serving_slo_survey.md` chat-class + §1-17 선례), 사다리 {50,60,80}ms 민감도 병기.
초안의 150ms는 **실험을 무의미하게** 만들었다 — 8B 측정 ITL p50이 전 arm·전 D에서 150ms
아래라 ITL 항이 절대 binding하지 않고 conjunctive goodput이 TTFT-only로 붕괴한다.
**대칭 함정**도 함께 처리: TTFT SLO는 ≥1 셀에서 binding + 모든 셀 p95로부터 ≥15% 마진,
없으면 "TTFT 축 ill-posed"로 보고. 자동 플래그 3종(`CLIFF HAZARD` M8 예상 후보 ·
`ITL-NONBINDING` T8 예상 후보 · TTFT 마진 위반). **게이트 #8은 동적 컨트롤러 규율이라
전 셀 `FixedPolicy`인 E1의 사전등록 사다리 재스코어에는 적용되지 않음**을 문서에 명시.

### 5. ★★ 측정 방법론 결함 5종 — 전부 "무엇을 세는가"

스모크 4회(865832 → 866066 → 866868 → 867034)에서 순차 발견. **전부 반대 결론을 낼 뻔했다.**

| # | 결함 | 잘못된 값 → 옳은 값 |
|---|---|---|
| 1 | **게이트 모집단**이 `prefill_active OR decode_active` — 설계상 정상인 decode-only 무분할 윈도우를 pin 실패로 셈 | pin 0.029 **FAIL** → 조건부 0.976 **PASS** |
| 2 | **동시성 진단이 스냅샷 개수 기반** — `runtime_snapshot`이 이벤트루프 iteration당 발화해 235ms prefill과 11ms decode step이 **같은 무게** | concurrent 0.0152 → **시간가중 0.248–0.398** (16–26×) |
| 3 | **`achieved_rps`가 drain 꼬리에 희석** — 도착 10s인데 duration 23.6s | 3.40 → `arrival_rps` **8.96** |
| 4 | **cold-start 미폐기** — TTFT가 rate에 역전(210ms@rate2 > 99ms@rate8) | warmup 폐기 후 **87 vs 92ms**(단조 회복) |
| 5 | **에피소드 게이트가 bracket 양끝**(구성상 prefill 윈도우 **바깥**)을 읽음 + 개수 기반 | d16 pin 0.583/informative 0 → **시간가중 0.847** |

★ 결함 2의 철회: 나는 `concurrent_frac ≈ 0.001`을 근거로 "prefill과 decode가 겹치지 않으니
multiplex할 게 없고 프론티어가 벡터1처럼 ill-posed일 수 있다"고 판단했다 — **철회**
(`DESIGN.md` §9.7). 실제 동시성은 25–40%다.

★ 결함 5의 최종 해소: **두 질문에 하나의 추정량을 쓰고 있었다.**
- **게이트**("prefill이 target SM에서 돌았나") → prefill-active **시간** 중 target 비율.
  모든 샘플 사용.
- **귀속**("이 요청의 TTFT는 어느 파티션 것인가") → 요청별 bracket. 대부분 버림.

⇒ 1차 게이트를 시간가중으로 되돌리고(에피소드 클러스터 bootstrap), `compute_episode_gate`는
바이트 그대로 둔 채 **비-게이팅 지연 귀속 진단**으로 역할만 축소. 검증: d16 pin_frac
**0.847**(n_episodes 8, lower95 0.554 → **FAIL, 단 사유는 검정력 부족이지 잘못된 SM 아님**),
d92 **0.999**(n 75, lower95 0.996 → PASS). 원자료 직접 집계와 정확히 일치.

### 6. ★ 실현 파티션 ≠ 셀 라벨 (본 스윕 해석에 필수)

`e1_T8_d16_866066_telemetry.jsonl` 직접 집계(시간가중):

| 셀 | prefill-active 총 시간 | target | 무분할 `(108,0)` |
|---|---|---|---|
| d16 = `[92,16]` | 2.030s | 1.719s (**85%**) | 0.311s (**15%**) |
| d92 = `[16,92]` | 47.144s | 47.091s (100%) | 0.053s (0%) |

**d16 prefill의 15%가 92 SM이 아니라 108 SM에서 실행된다.** 버그가 아니라 **정책의 실제
동작**(decode가 비면 런타임이 무분할로 auto-revert). 함의: 셀 라벨은 **목표**이지 실현
배분이 아니고, 그 15%가 d16의 TTFT를 실제보다 **좋게** 만들어 프론티어의 prefill-rich 끝을
유리하게 편향시킨다. Stage 0의 D108 앵커 실패와 **같은 구조**(라벨 vs 실현)지만 원인이
다르다(그건 설정 버그, 이건 정책의 auto-revert). ⇒ 결과 서술은 "D=16에서"가 아니라
**"목표 D=16, 실현 target 85%/무분할 15%에서"** 여야 한다. 사전등록 완료.

### 7. 셀별 동시성 비대칭 (프론티어 해석에 필수)

시간가중 `concurrent_time_frac`: **d16 ~1.3% · d44 4.9% · d92 25–40%**.
**prefill에 SM을 많이 줄수록 prefill이 빨리 끝나 덜 겹친다.** confound가 아니라 정책 선택의
실제 귀결이지만, D를 비교할 때 "각 phase가 SM을 얼마나 받는가"만이 아니라 **"두 phase가
실제로 얼마나 겹치는가"도 함께 비교**하게 된다 ⇒ 셀별 병기 필수(§4.2.4 사전등록).
6번의 무분할 노출 비대칭과 같은 뿌리다.

### 8. seed 정책 (paired bootstrap 전제)

`--seed`를 한 번도 override하지 않아 **모든 rep·셀·arm이 비트 동일한 도착 시퀀스와 ShareGPT
표본**을 공유하고 있었다. 867034 실측이 위험을 보여준다 — rate≈8에서 **도착률은 2.2%만
다른데 TTFT p50이 18.3% 갈린다**(순수 워크로드 구성 변동). 고정 seed로 4 rep을 돌렸다면 이
18%가 CI에 **전혀 안 들어갔을** 것이고, 결정 규칙("CI가 0을 배제")이 하나의 도착 실현에서만
성립하는 우연을 실재하는 차이로 승격시켰을 것이다. 이 프로젝트가 이미 목격한 현상
(`bench_noise_root_cause.md` "워크로드 4런 전부 동일 fingerprint")이 **여기선 반대로 위험**이
된다.

⇒ **rep 간 seed 변화·셀 간 고정**(`--seed $((BASE+rep))`, 같은 rep 인덱스는 전 셀 동일)으로
paired 매칭을 유지하면서 CI에 워크로드 변동을 넣는다. RNG replay가 비-기본 seed에서도
유효함을 `seed∈{1,2,7}×rate∈{2,8}`로 검증(오차 2–70ms).

### 9. 엔진 패치 — prefill-active iteration 강제 추적

`PDMUX_TRACE_FORCE_PREFILL`(**기본 OFF**). prefill in-flight iteration만 카운트 서브샘플링을
무시하고 방출. 동기: 순수 카운트 서브샘플링(`trace_every` 기본 32)이 **prefill이 빠른 셀일수록
관측 확률을 낮추는 체계적 편향**을 만들어, 프론티어 한쪽 끝(d16)이 표본 8개로 굶었다.

- OFF일 때 **바이트 동일 출력**(새 필드를 아예 넣지 않아 `sort_keys` 직렬화 동일),
  ON일 때 `trace_forced==false` 필터로 기존 population 완전 복원.
- **관측 전용** — 스케줄러 상태·`stream_idx`·`r2_policy`·배치 어디에도 쓰지 않음(코드 경로로 논증).
- CPU 회귀 **28 tests PASS**(신규 5). 부수로 `test_profile_controller.py`가
  `sys.modules["profile"]`을 등록해 **stdlib `profile`을 가려** 같은 discover 실행의 이후
  `import sglang`을 죽이던 문제를 발견·문서화.
- ⚠️ **교란은 프론티어를 따라 비대칭이고 방향이 직관과 반대다**: 강제 방출량은 prefill이
  in-flight인 **sync 호출 수**에 비례하므로 **느린 prefill 셀이 가장 크다** —
  d16 +1.2%(~2/s) · d44 +2.2% · d92 **+28.5%**(~44/s). 스케줄러 wall로는 d16 ≲0.02%,
  d92 ≲0.3%(방출 1회 33–45µs).
- ⚠️ **`admission_blocked_frac`도 개수 기반이라 force 모드에서 하향 편향**(강제 기록은 전부
  prefill-active라 구성상 blocked가 아니고 분모만 부풀림) → `trace_forced != true` 필터로 수정.
- **관측자 효과 게이트 미통과** — job 867298 PENDING. 통과 전엔 이 패치가 교란 없다고
  주장하지 않는다.

---

## 코드·문서 변경

| 경로 | 내용 |
|---|---|
| 정본 8문서 | `PROJECT_STATUS.md`(Stage 0 절 철회 + "8B decode-SM 민감도 측정 노트" 신설 + "열린 긴장" 신설 + 방법론 게이트 3종) · `reports/CONSENSUS.md`(§1-21 판정2/3 철회·"3중 삼각검증" 삭제, **§3-9 재작성**, §3-10/11/12 신설, §5-6 "게이트 미실행" 복원) · `reports/stage0_verdict_2026-07-26.md`(HISTORICAL 배너) · `reports/longcontext_trace_plan.md` · `reports/paper/CLAIM_EVIDENCE_MATRIX.md`(Claim A evidence 교체, **등급 불변**) · `reports/paper/EXPERIMENT_ROADMAP.md`(P6 재작성 + E1–E4) · `results/s0_deconfound/DESIGN.md`(§1.1 표 무효) · `results/s8_scaleup/FINDINGS_8B_2026-07-28.md`(§6 누락 2문장 + §3 "비교 불가"로 강화) |
| `results/s8p_prefill/` | 신규 캠페인 전체 + `FINDINGS_PREFILL_2026-07-29.md` |
| `results/s8_frontier/` | 신규 캠페인 전체(하네스·양축 판정기·분석기·config 5) + 결함 5종 수정 |
| `results/e1_traceforce/` | 관측자 효과 게이트(ABBA 설계) + `tfgate_inside_bracket.py` 읽기전용 진단 |
| `src/multiplex/multiplexing_mixin.py` | trace-force 패치 |
| `tests/test_trace_force_prefill.py` | 신규 5 테스트 |
| `tests/test_profile_controller.py` | shadowing 주석 |

커밋: 이번 세션 총 8+ 커밋(정본 개정 4 + prefill/E1 4 + 엔진/게이트 재설계분은 세션 말
진행 중). **push 없음.**

---

## 10. ★ 두 job 착지 (2026-07-29 22:41 완료 — backfill이 추정보다 이틀 앞당김)

### 867298 관측자 효과 게이트 — **비대칭 교란 검출, 조건부 통과**

ABBA 부팅(OFF/ON/ON/OFF), 조건당 n=4, paired.

| 셀 | telemetry 볼륨 | t-CI가 0을 배제한 지표 |
|---|---|---|
| d16 = `[92,16]` | +9.3% | **없음**(전 지표) |
| d92 = `[16,92]` | +41.7% | **itl_p95 +2.00%** [+0.78, +3.21] · itl_p99 −1.28% · itl_mean +0.56% |

TTFT는 양 셀 모두 0 포함, throughput +0.02%.

**문제는 크기가 아니라 비대칭**이다 — 유의한 지표가 하필 **ITL-p95**(E1 결정 규칙이
임계하는 값)이고 **d92에서만** 나타난다. force-trace를 켜면 decode-heavy 끝이 ITL에서
약간 불리해진다. p95(+2.0%)와 p99(−1.28%)의 부호가 엇갈리는 걸 보면 실제 미세 효과 +
다중검정 잡음(18검정 중 3개 유의)이 섞인 것으로 읽힌다.

⇒ **권고: 측정과 검증을 분리.** pin 확인은 **설정의 성질**이지 특정 스윕 실행의 성질이
아니다. **force-trace ON은 셀별 짧은 pin 검증 런에만** 쓰고 성능 본 스윕은 **OFF**로 돌린다
(같은 rate에서 검증하면 실현 배분 혼합비도 유효). 결정 지표에 관측자 효과가 들어가지 않는다.

⚠️ `traceforce_gate.sbatch`의 `PIN_CHECK` 호출부가 `e1_sweep.sbatch`에서 고친 **옛 인자
순서 버그를 그대로** 갖고 있어 전부 크래시했다(paired summary는 정상 산출). 그 job의 pin
데이터는 없다. **다음 세션에서 수정할 것.**

### 867231 T8 용량 스캔 — **elbow 확정**

TTFT p50 (ms), x축은 재구성 `arrival_rps`:

| arrival_rps | 9.3 | 12.6 | **16.0** | 24.9 | 31.2 |
|---|---|---|---|---|---|
| d16 | 73 | 103 | **325** | 3986 | 8494 |
| d24 | 70 | 93 | **207** | 2987 | 7024 |
| d44 | 83 | 113 | **225** | 2339 | 6008 |

- **절벽 = arrival_rps 12.6 → 16.0 사이**, 세 셀 공통. **off-cliff band ≤ 12.6.**
- 포화 처리율 **d16 8.9 / d24 9.7 / d44 10.1 req/s** — decode SM이 많을수록 용량이 크다
  (포화 시 병목이 decode).
- ★ **프론티어 신호는 ITL에 있다.** off-cliff(12.6)에서 request-내부 ITL-p95:
  **d16 47.9ms · d24 34.7ms · d44 22.0ms**. TTFT는 셀 간 93–113ms로 거의 같은데 **ITL이
  2.2× 갈린다** — 이게 프론티어가 재려던 교환이다.
- ⚠️ **사전등록 1차 SLO 60ms에서 T8의 ITL 항은 전부 통과 = 비구속**(`ITL-NONBINDING`
  플래그가 예측한 그대로 발생). 사다리 하단 **50ms에서 d16(47.9)이 경계**에 걸려 구속되기
  시작한다. **사후에 50으로 바꾸면 데이터를 보고 고른 값이 되므로 금지** — 60ms 1차를
  유지하고 T8은 NONBINDING 라벨, 사다리로 보완하는 것이 사전등록 정신에 맞다.
  나머지 3 arm은 decode가 느려(8B ITL p50 M8 58ms / Ha8 90ms) 60ms에서도 구속될 가능성이 크다.

---

## 열린 항목 / 다음 세션 시작점

1. **두 job 분석 완료됨(위 §10)** — 남은 액션: force-trace를 pin 검증 전용으로 분리하는
   설계 확정, `traceforce_gate.sbatch` 인자순서 버그 수정.
2. **나머지 3 arm 용량 스캔** — M8/Ha8/Hs8. T8과 달리 ITL이 느려 60ms SLO가 구속될
   가능성이 크므로, **arm별로 off-cliff band와 ITL 구속 여부가 다를 수 있다**(5셀 동시
   off-cliff인 rate가 없으면 그 자체가 발견이며 덮지 말 것).
3. **SLO 배치 확정** — 용량 스캔의 TTFT/ITL 분포에 `--capscan-dir` 모드를 돌려 사전등록
   규칙대로 TTFT SLO를 고른다(ITL-p95 60ms는 이미 고정).
4. **E1 본 스윕** — 위 1–3이 전부 닫힌 뒤. 4 arm × 5 D × n≥4는 크므로 단계적(Stage 1 =
   T8 + Hs8).
5. **`s8p_prefill`을 claims-auditor에** — `FINDINGS_PREFILL_2026-07-29.md`는 아직 정본 인용
   금지. 반증받아야 할 축: 게이트 미달 5셀의 strict 재귀속이 충분한가 · 곡률 크기가 attention
   FLOP 예측의 5배인 것(기전 미상) · 두 게이트 동일 사건이라 사전등록이 1개분인 것 ·
   `--probe-conc 2` 사전등록 미실행.
6. **`s8p_analyze.py` 버그 2개 수정**(engine-porter 위임): 스모크 job rep-key 충돌(rep sd
   오염) · 구간 순수성을 양끝점으로만 검사. 현 수치는 수정 후 기준이며 as-run과 ≤2.7% 차이.
7. **방법론 게이트 승격 후보** — "집계 단위(개수 vs 시간 vs 에피소드)를 먼저 정하고, 그
   단위가 추정 대상과 맞는지 논증하라". 이번 세션에서만 **5번** 답을 바꿨다. doc-steward가
   `CONSENSUS.md` §3 계열에 등재할지 판단.

---

## 미완·주의

- **`s8p_prefill`·`s8_frontier` 어떤 결과도 정본 아니다.** claims-auditor 미통과 상태에서
  논문·정본 인용 금지.
- **엔진 패치는 관측자 효과 게이트 미통과**. 기본 OFF라 867231은 새 트리를 sync해도 안전
  (OFF = 바이트 동일 출력).
- **게이트 clean 헤드라인이 필요하면 prefill 캠페인은 M8 단독(5.163×)**, 나머지 3 arm은
  "셀 1–2개 사전등록 게이트 미달, 구간-순수 재귀속으로 방어" 병기.
- **곡률을 O(L²) 서명으로 쓰지 마라** — 크기가 예측의 5배. `L_LIST`에 8192 추가가 정공법이며
  미실행.
- **d16 셀은 여전히 검정력 미달**(n_episodes 8). trace-force가 이걸 닫을 것으로 예상되나
  **검증 전**이다.
- **fairshare ≈0.035**로 큐 대기가 매우 길다(job당 수 시간~이틀). 실험 계획 시 반영할 것.
- 루트 SLURM `.out/.err`는 커밋 대상 아님(이미 ignore).
