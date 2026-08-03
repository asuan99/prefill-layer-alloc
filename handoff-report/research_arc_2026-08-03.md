# 연구 진행 흐름 — 근거 · 과정 · 수치 · 해석 · 다음 결정

작성: 2026-08-03 · **DERIVED 문서 — 정본 아님.** 모든 수치는 정본
(`PROJECT_STATUS.md` > `reports/paper/` > `reports/CONSENSUS.md`)에 이미 기록된 것을
재배열한 것이고, 새 판정을 만들지 않는다. 충돌 시 정본이 이긴다.

이 문서는 "무엇이 확정됐나"(그건 `PROJECT_STATUS.md`가 답한다)가 아니라
**"왜 그 실험을 했고, 어떤 수치를 보고, 그것을 어떻게 읽어서 다음 실험으로 갔나"**를
복원한다.

---

## 0. 출발 가설과 그것이 살아남지 못한 방식

**최초 논지**: Hybrid LLM(attention + SSM 혼합)은 layer 종류마다 연산 성격이 달라
SM 요구가 다르다 ⇒ prefill/decode를 SM으로 공간분할(PD-mux)할 때 **layer 종류에
맞춰 분할을 바꾸면**(layer-type-aware) 균일 분할보다 이긴다.

**근거**: 시뮬레이터에서 layer-aware/agnostic 이득이 모델 크기 전 구간에서 관측됐다
— 1.2B **1.37×**, 2.7B **2.02×**, 7B **1.82×**. SLM 한정 현상이 아니었다.

이 가설은 최종적으로 **실엔진에서 全형태 반증**됐다. 그 과정이 이 프로젝트의 방법론
전부를 만들었으므로, 아래는 반증의 순서 자체가 내용이다.

---

## 1. PD 분리 자체 — 유일하게 살아남은 긍정 결과

**왜 했나**: layer-aware를 논하기 전에 그 토대인 "prefill/decode를 아예 분리하는 것"이
fused 실행보다 나은지부터 확인해야 했다.

**어떻게**: 4개 hybrid 모델(NemotronH / Zamba2 / Falcon-H1 / Granite-4)에서
pdmux(agnostic v1) vs fused를 서빙으로 측정.

**결과**: pdmux가 **4모델 전부**에서 최적. 운영점은 cudagraph-ON.

**해석과 다음 스텝**: 토대는 견고하다 ⇒ 그 위에서 분할 **정책**을 다투는 것이 의미가
있다. 이것이 이후 모든 실험의 전제가 됐고, 지금까지 뒤집히지 않은 유일한 긍정 결과다.

부수로 **agnostic_v2는 decode 실작업이 있으면 최악**(NemotronH·Granite)이 나왔는데,
Granite에서 "agnostic_v2 최적"이라고 한 번 오판했다가 서빙에서 반증됐다. 원인은
**decode-side micro-timing(작은 batch)이 서빙 batch의 SM 민감도를 과소평가**한
아티팩트였다. ⇒ **교훈 1: 정책 주장은 반드시 서빙으로 실증한다.** micro-measurement가
이 프로젝트를 오도한 것이 이때까지 2회(GIL-client 과부하, tiny-batch 민감도)였다.

---

## 2. layer-aware 반증 — 같은 결론에 세 번 다르게 도달함

### 2.1 1차: 이진 layer-aware의 패배

**어떻게**: clean async 클라이언트로 agnostic vs layer-aware를 4모델 비교.

**결과**: 저부하에선 동등, **부하가 걸리면 열위 → 붕괴**. Zamba2가 최악으로
rate 3부터 goodput 0이었다. 사전 예측("45/54 환원으로 대박")이 정확히 반대로 나왔다.
Falcon-H1은 단일 layer type이라 정의상 agnostic과 동일.

**당시 해석**: layer-aware 폐기, 실전 권고 = agnostic.

### 2.2 사용자 반론이 만든 2차 — "이진 strawman이고 agnostic은 over-provisioning이다"

**근거**: 위 비교는 layer-aware를 두 극단으로만 구현했고, agnostic이 이긴 것은
단지 자원을 넉넉히 줬기 때문일 수 있다.

**어떻게**: `PDMUX_LA_SM_MAP`으로 **graduated per-type** layer-aware 구현(R0b).

**1차 결론 → 철회**: "knee ≈ 84 SM, 잉여 없음"이라고 결론냈다가 사용자 재지적으로
**철회**했다. 구현이 decode를 **독립 green-context**에 둬서 pdmux prefill과
**미조율(경합)** 상태였고, 이는 이진 layer-aware와 같은 결함이었다.

**결정적 수치**: agnostic **coordinated** decode 54 SM = **42 ms** vs
**미조율** 54 SM = **121 ms**.

**해석**: 3배 차이는 SM 요구량이 아니라 **조율 여부**에서 온다. 따라서 앞선 두 비교는
전부 "미조율"이라는 교란에 오염돼 있었고, **잉여는 실재**했다(agnostic prefill에
최대 74 SM). ⇒ coordinated per-type layer-aware를 **실제로 구현해서** 다시 재야 한다.

### 2.3 3차: coordinated per-type을 진짜 구현하고도 패배

**어떻게**: libsmctrl 경로가 driver 580에서 막혀(3a BLOCKED), green-context
event-loop을 수술해 `event_loop_pdmux_coord`를 만들었다(3b). 정확성 게이트 통과.

**결과**: **decisively 패배 — decode TPOT 42 → 124 ms.**
기전은 실증됐다: decode가 **19개 윈도우로 파편화**되어 sync 직렬화 + partition 핀 +
overlap 감소.

**해석**: 창립 가설이 **작동하는 구현으로** 반증됐다. 이 시점에서 layer-type 런타임
정책은 全형태 死.

### 2.4 사후 검증: magnitude는 절반, 부호는 견고

**근거**: "sim에서 이기던 것이 실엔진에서 진다"는 결론이 기판 아티팩트일 수 있다는
회의(사용자 제기).

**어떻게**: `PDMUX_LA_COORD_OPT`(GPU wait_stream + decode 핀 제거)로 substrate isolation.

**결과**: R0d의 124 ms 중 **약 절반이 substrate** 탓 — magnitude 회의는 옳았다.
그러나 **부호는 견고**(여전히 agnostic/tuned-uniform에 패배). 잔차는 monolithic
prefill의 단일 윈도우 오버랩으로 **구조적·모델 독립**.

**다음 스텝**: layer-aware가 죽었으므로 남은 upside는 **동적 제어**뿐이다.

---

## 3. 동적 제어(SLO-aware) 트랙 — HE0

**근거**: 최적 static split이 workload·context·load에 따라 이동한다면(확정된 결과 4번),
런타임에 따라가는 컨트롤러가 이겨야 한다.

### 3.1 벤치마크를 먼저 잃었다

정책 비교에 쓰던 **stationary ShareGPT r8이 static조차 5.282 ± 1.302**(4런 중 1런
붕괴)로 흔들렸다. ⇒ **그 벤치로 낸 결론들("트랩이 bimodal 원인", "게이트가 회복")을
철회**하고, 최근 n≤3 비교는 전부 underpowered로 재분류했다.

**노이즈 추적 3차에서 원인이 나왔다 — 노이즈가 아니라 metric cliff**:
워크로드는 4런 fingerprint가 일치했고(변동 소거), 하부 섭동은 **throughput 3% ·
ITL 8%** 뿐인데 goodput이 2배 움직였다. r8이 **TTFT ≈ SLO(3 s) 경계**에 있어
3% 결손이 평탄역 1.5 s를 3.7 s로 밀어 임계를 넘긴 것이다(400/400 → 206/400).
rate 3 = 견고 / 8 = 불안정 / 12 = 견고 ⇒ **경계 regime만 불안정**.
⇒ **클럭 throttling 가설 철회, 자원 격리 불필요.**

같은 시기에 하네스 버그도 나왔다: 여러 라운드 합칠 때 duration을 `max()`로 잡는데
goodput은 3라운드 합산이라 **3배 부풀림**. `dur += d`로 고쳤고 **순위는 보존**됐다.

⇒ **교훈 2: 정책 비교는 변화 trace(rate 3↔12)로, n≥4로, metric cliff를 피해서.**

### 3.2 HE0 — 동적은 best static을 넘지 못한다

**수치(수정 후, n≥4)**:
d44 **3.220 ± 0.013** > d34 3.171 > bind+GATE **3.132 ± 0.019 (n=9)** > d24 3.081 >
slo 2.974 > no-gate 2.934 ± 0.306 > d16 2.817 ⇒ **5.4σ**.

**게이트의 정체**: one-way ratchet auto-tuner였다(d24 → d34 1회 이동 후 prefill-ward
113회 전부 거부 → d34 고정). 즉 **틀린 static에 조기 수렴**했다(최적은 d44).

**게이트의 가치는 성능이 아니라 견고성**: 깨끗한 벤치에서 no-gate만 1/4 붕괴,
gate는 0/7 (분산 12배 차이).

**컨트롤러 오버헤드는 死인가 검증**: 직접 계측 결과 32–36 µs/call, max 267 µs,
누적 34 ms = wall의 **0.014%**. ⇒ "switch = 0인데 static 미달"은 오버헤드 탓이 아니다.

### 3.3 왜 동적이 못 이기는가 — 구조적 이유

**spread 분해**: LO(rate 3) **0.067** vs HI(rate 12) **1.187** ⇒ 차별의 95%가
과부하 phase에서 나오고, **LO는 split에 무관심**(d16 2.861 ≈ d44 2.858).

**해석**: 두 regime의 최적이 **충돌하지 않는다** ⇒ decode-heavy static이 정의상 최선.
**동적이 이기려면 최적이 충돌해야 한다.** 이 명제가 이후 모든 후속 실험의 설계를 정했다.

### 3.4 실trace가 중간 결론을 정정

synthetic에서 "최적 = d16이고 불변"이라 결론냈는데, ShareGPT에서 **d16이 붕괴**했다
(goodput 1.056 vs d24 6.240). 변화 trace에서는 최적이 **실제로 이동**했다(HI = d44).
그래도 **static(d44 9.71) > dynamic(bind 9.35)**.

**기전 = 얽힘**: decode 굶김 → ITL > SLO → running batch 정체 → prefill admission
차단 → TTFT 폭발. 대표 수치로 **D16은 D24보다 prefill SM이 많은데 TTFT가
7.24 s 대 1.21 s**였다.

⇒ 최적 `D_sm = max(모델 floor, 부하항)`이고 **비대칭**(decode 과소공급 = 파국).
**실무 권고: peak decode 부하 기준 decode-heavy static.**

---

## 4. HE0 재개 시도(벡터1) — "최적이 충돌하는 워크로드가 존재하는가"

**근거**: §3.3이 "동적이 이기려면 두 phase의 최적이 충돌해야 한다"를 만들었으므로,
그런 disjoint-feasibility 워크로드가 실재하는지가 남은 escape hatch였다
(`CONSENSUS.md` §5-8(c)).

**과정과 세 번의 재판정**:

1. **g2_0_full (n=4)**: razor-thin real disjoint 발견(feasible-A={d16,d44} ∩
   feasible-B={d54} = ∅).
2. **g2_0_hard (n=6–10) 재스윕**: 재현 실패. byte-identical Phase-A 워크로드에서
   d44/d54 견고성 **순위가 완전 반전**했고, n=10 pool에서 둘 다 ~0.86–0.90으로
   구분 불가. "disjoint 소멸" 관측은 별도의 **ITL-p95 percentile-window
   아티팩트**였다(OB 512→1024에서 median↑·p95↓). ⇒ **ILL-POSED at rA5**.
3. **de-cliff (jobs 863880–863948)**: `rA{2,3,3.5,4}×{d16,d44,d54}` 스캔 →
   **rA=2만 clean off-cliff**(rA≥3은 전부 bimodal). n=6 확증에서 static d54가
   양 phase 동시 커버(Phase-A frac_good 0.974, TTFT p99 ≤ 1028 ms;
   Phase-B ITL-p95 42.3 ± 0.10 ms, frac_good 1.0). ⇒ **PLAUSIBLE closure**이지
   CONFIRMED 아님 — claims-auditor가 반증 3항목을 냈다.
4. **확증 열**: `g2_0_rasweep` 120 job이 off-cliff band(rate ≤ 2.75)에서 disjoint
   부재를 재확인해 전이대를 rate 3.0–3.5로 좁혔고, 그 창을 겨눈 **사전등록 24-job
   `g2_0_raconf`**가 결정 규칙을 충족.

**최종 수치 (companion collapse)**:

| rate | d44 | d54 |
|---|---|---|
| 3.5 | 0.953 ± 0.035 | 0.948 ± 0.035 (failTTFT 0/6) |
| 3.75 | 0.932 ± 0.042 | **0.948 ± 0.062** |

**해석**: REOPEN 전제 두 개(d54가 견고히 <0.7 unimodal ∧ d44가 견고히 ≥0.95
off-cliff)가 **양쪽 다 붕괴**했다. d54는 Phase B의 유일 feasible이면서 Phase A도
d44와 대등하게 커버한다 ⇒ **단일 static이 양 phase를 커버** ⇒ **disjoint 부재 확정.**

**scope 제한(중요)**: short-ctx drained 2-phase, rate_A ≤ 3.75 한정. "hybrid엔
disjoint가 없다"로 **일반화 불가**. magnitude는 metric cliff로 ill-posed이나
순위는 견고.

**다음 스텝**: 남은 미측정 영역은 **long-context**(decode floor가 오르는 영역)와
**spatial decoupling**(§1-20) 두 축.

---

## 5. long-context와 스케일 — 세 번째 반전이 여기서 났다

### 5.1 Stage 0 (2026-07-26) — 그리고 2026-07-28 전면 철회

**근거**: long-ctx에서 decode floor가 올라 두 phase 최적이 충돌할 수 있다(H_L4/H_L5).
그 전제로 "운영점에서 decode가 SM에 민감한가"를 먼저 재야 했다.

**어떻게**: 3-arm(음성대조 pure-Mamba2 / hybrid Zamba2 / 양성대조 Qwen2.5-3B) ×
ctx {4k, 8k, 16k} decode-only 스윕. raw 곡선이 confounded일 것을 예상해
**무경합 앵커 D108**을 대조로 뒀다.

**당시 결과와 해석**: D16 vs D108 = **1.00 ± 0.01**(3 arm × 3 ctx 전부) ⇒
"decode SM-무감각을 hybrid·16k까지 확장 확인" ⇒ H_L4/H_L5 붕괴, HE0 강화.

**2026-07-28 claims-auditor 감사 (C1 CONFIRMED) — 전면 철회**:
그 "D108 무경합 앵커"는 **실제로는 decode 16 SM**이었다(legacy auto-path
threshold = 0). 3중 독립 증거(코드 기전 · telemetry 재집계 · 클라이언트 시그니처).
⇒ "D16 ≡ D108 = 1.00 ± 0.01"은 **동일 조건 반복 측정**이었고 헤드라인은 무효.
판정1(raw 스윕 CONFOUNDED)만 생존, long-ctx 트랙은 "게이트 실패"가 아니라
**"게이트 미실행"**으로 복원.

⇒ **교훈 3: pin은 target이 아니라 realized로 검증한다.** (이 교훈이 이번 세션에
정확히 두 번 더 필요해진다.)

### 5.2 C2 (8B 스케일업) — 반대 방향의 증거

**어떻게**: prefill을 16 SM에 **고정**하고 decode SM만 16 → 92로 스윕(4 arm).

**결과**: decode ITL **2.36–2.91× 개선, 4 arm 모델-무관.**

**해석(범위 엄수)**: **레버가 존재한다는 것만** 확립. 정책 이득이 아니고 HE0를
되살리지도 않는다(프론티어 `[108−D, D]`는 미측정 — prefill을 고정했으므로).
"hybrid 급락 = Zamba2 additive 성질"이라는 부수 주장은 모델 간 미통제 비교라
**NOT-YET-SUPPORTED로 강등**.

**여기서 긴장 A가 생겼다**: HE2("운영점에서 decode 축은 flat, 최적 split은 static")
vs C2("decode SM 레버는 실재한다"). 둘 다 철회되지 않았고, **직접 측정으로만 닫힌다.**

### 5.3 prefill 축 거울상

decode를 16에 고정하고 prefill을 16→92로 스윕. TTFT~L 회귀 기울기비는
M8 5.163× / Ha8 5.016× / Hs8 4.860× / T8 4.740× — **4 arm 사실상 동일**
(decode 축의 모델 무관성이 재현).

★**여기서 나온 반증된 추론이 재사용 가치가 높다**: "prefill 기울기 > decode 기울기
⇒ 프론티어에서 prefill이 더 급하다"는 **지지되지 않는다**. 끝점 비는 프론티어 결정량이
아니고, 국소 탄력도 구조가 달라(decode는 D ≳ 44에서 급포화) **SM 1개를 옮기는
부호가 동작점마다 뒤집힌다**(내부 균형점 존재). ⇒ 긴장 A는 끝점 비로 못 닫는다.

---

## 6. E1 프론티어 게이트 — 긴장 A를 닫으려는 시도

**근거**: C2가 레버를 확립했으므로 남은 질문은 "**예산 제약 하**(prefill = 108−D)
에서도 net-positive인가"다. 이것이 E1이다.

### 6.1 하네스를 짓는 동안 집계 단위가 답을 5번 바꿨다

pin 0.029 FAIL → 0.976 PASS, 동시성 0.015 → 0.25–0.40, achieved_rps 3.40 →
arrival_rps 8.96, d16 pin 0.583 → 0.847 … 전부 반대 결론을 낼 뻔했다.

⇒ **교훈 4(방법론 게이트 #4): 집계 단위(개수 vs 시간 가중, target vs realized,
스냅샷 vs 에피소드)를 먼저 정하고 추정 대상과 맞는지 논증하라.**
따름정리: 게이트("그 파티션에서 돌았나")는 시간 가중, 귀속("이 지연이 어느 파티션
것인가")은 요청별 bracket — 하나의 추정량으로 둘 다 답하면 후자가 게이트의 검정력을
무너뜨린다.

부수 관측: green-context 분할은 decode가 비면 무분할로 auto-revert하므로
**셀 라벨은 목표이지 실현 배분이 아니다**(`CONSENSUS.md` §1-22). — 이 한 줄이
이번 세션의 핵심이 된다.

### 6.2 E1 전제 실험 4건과 감사 (2026-08-01 → 08-02)

**어떻게**: M8/Ha8/Hs8 용량 스캔(각 100 probe) + T8 batch-cap
({d16,d44} × cap{48,96,192} × 4 seed).

**1차 자기 판정**: 사전등록 SLO 사다리 {50,60,80} ms가 **네 arm 전부 헤드라인 룽
없음**, as-run `--max-running-requests 48`이 ITL·TTFT 두 축을 **반대로 왜곡**
(d16 ITL-p95 cap48→192 +12.2 ± 0.7 ms인데 cap48이 TTFT를 612→114 ms로 5.4× 악화),
d92 knee 2.80이 네 arm 전부 구속 ⇒ "E1이 사전등록 분기 **설계상 이 질문에 도달할 수
없다**로 갈 위험" ⇒ 본 스윕 미제출.

**claims-auditor 회부 결과 — 그 종결 논거가 무너졌다**:
- (c) "제외 규칙이 레버를 지운다" = **기각**. C2 자신의 데이터에서 SM16→44가
  log-range의 **75–81%**를 차지하므로 d92 제외는 마지막 15–25%만 자른다.
- "네 arm 공통 knee 2.80" = **철회**. knee의 치역이 probe 격자뿐이라 일치가 부분
  강제된다. 살아남는 건 **순서**(d92가 먼저 무너짐: T8/M8/Hs8 9/9, Ha8 8/9).
- (b) cap 왜곡은 **rate 16 = 운영대역의 5.7배 밖**(운영대역 실측 동시성 12–44 <
  cap 48이라 구속 불가).
- arm × 룽 표는 **rate-confound**로 폐기(T8만 rate 12). 단 결론
  `HEADLINE-ELIGIBLE = NONE`은 공통 rate · 두 estimand · 두 seed 전부에서 **생존**.

⇒ **E1 제출도 종결도 시기상조.** 이 감사에서 철회 8건 중 3건이 **항등식을 증거로
착각**한 것이었다(`arrival_rps` = RNG replay 재생성값, `kv_mamba_occupancy=1.0` =
pool 크기가 cap과 같아 생기는 항등식, "ITL 구속 rate ⟂ …" = 공집합이라 공허참).
⇒ **교훈 5(방법론 게이트 #6): 이 양이 내가 재려는 것과 논리적으로 독립인가를 먼저
물어라. 항등식을 증거로 쓰지 마라.**

### 6.3 M1–M5 — GPU 0으로 처리한 진단

- **M1**: knee를 코드로(`e1_analyze.py --knee-scan`, 발표된 20셀 정확 재현) +
  `--common-rate`. 교훈: **손계산으로 남은 게이팅 양은 감사자가 역설계해야 한다.**
- **M2**: ITL estimand 오염은 실재하나 **국소적**(이미 제외되는 셀). 내 첫 stall 검정이
  38/40에서 발동한 건 **scale 없는 절대 임계** 아티팩트 → 17/40으로 정정.
- **M4**: stall 원인 종결 = **monolithic prefill**. 17 stall probe 중 **16개**에서
  최장 프롬프트의 prefill이 stall 전 구간을 덮고 크기가 D에 단조(prefill SM = 108−D).
  ⇒ **ITL 항이 decode-SM 레버와 반대 부호 항을 내장**한다. **고칠 수 없다** —
  `server_args.py:6130`이 pdmux일 때 `chunked_prefill_size == -1`을 하드 assert
  ⇒ venue positioning의 **(A) green-context 종속** 버킷.
  ⚠️ 여기선 내 기전 추측이 **옳았고** 감사자의 REFUTED가 **틀린 검정**을 썼다
  (prefill 중인 요청은 자기 stall을 못 본다).
- **M5**: 폐기. 운영점 cap TOST는 **공허**(구속 불가한 곳의 동등성 검정).

### 6.4 M3 — Transformer-control 대조 (job 872077)

**근거**: negative 결과를 hybrid에 귀속하려면 **순수 Transformer 양성대조**가
같은 기판·같은 SLO를 통과해야 한다(venue positioning 게이트).

**설계**: T8(양성대조) + Ha8 × d16/d24/d44/d54 × 공통 rate 2 × **8 block × 1 seed**.
결정량 `g = A_free(d16)/A_free(d54)`를 **코드로 사전등록**.

★**설계 중 자가발견(최고 재사용가치)**: `g`가 **두 부팅을 가로지르는 비율**이라
반복 단위는 seed가 아니라 **block**이다. 초안의 "4 seed in 2 boot-pair"는
n_indep = 2(허용 sd 0.021 = 발화 불가)로, **batch-cap pseudo-replication을
그 수정본 안에서 재생산**한 것이었다. 또 percentile bootstrap이 n=4에서 커버리지
**79.8%**(공칭 95%)라 **t-구간으로 교체**했다.

**결과**: 64/64 probe, 에러 0, 3h01m. 그러나 pin 게이트가 19/64를 void시켰다.

### 6.5 pin 게이트 사고 — 게이트 자신이 항등식이었다

rev1(872236)·rev2(872497)를 더 돌린 끝에 **2차 감사에서** 밝혀졌다.
telemetry 120파일 · prefill-active 스냅샷 **77,688개**에서:

```
prefill_sms != target  <=>  decode_running_batch_size == 0
  off-target & decode-empty : 50,328
  on-target  & decode-busy  : 27,360
  위반 (양방향)             :      0
```

⇒ 시간가중 pin 게이트는 파티션 제어가 아니라 **"prefill in-flight 중 decode가
안 비어 있던 시간 몫"**을 재고 있었다. **조건부 pin = 1.000 정확**, 872077 재채점
**64/64 PASS**, `n_indep = 8` 회복.

같은 감사에서 **정작 게이트가 없던 축**이 드러났다 — decode 실현 배분:

| arm | d16 | d24 | d44 | d54 |
|---|---|---|---|---|
| T8 | **0.038** | 0.047 | 0.082 | 0.093 |
| Ha8 | 0.104 | 0.110 | 0.148 | **0.187** |

**decode 작업시간의 4–19%만 라벨대로 실현**, 81–96%는 무분할 108 SM.
⇒ **교훈 6(방법론 게이트 #7): 게이트를 만들 때 그 게이트가 재는 양이 통과 조건과
논리적으로 독립인지 먼저 증명하라.**

**872077 최종**: 전 게이트 통과, n_indep = 8,
**T8 g = 1.837 [1.666, 2.009]**, **Ha8 g = 1.068 [0.947, 1.190]**.
RULE_POINT는 발화하고 RULE_BOUNDS(2026-08-03 사전등록, forward-only)는 미발화
⇒ **NO VERDICT, 사유는 정확히 하나 — 규칙이 정작 중요한 지점에서 모호했다.**

---

## 7. 이번 세션 (2026-08-03 속행) — 결정 하나를 두고 벌어진 일

### 7.1 던져진 질문

"블록을 8 → 12–16으로 늘려 RULE_BOUNDS 하에 재실행할까? 열린 양은 **Ha8 상한이
1.15 아래로 내려오는가**(현재 1.190)."

### 7.2 첫 분석 — 검정력과 (틀린) 희석 논거

**검정력**: Ha8 per-block g = [1.118, 1.214, 1.171, 1.168, 0.847, 1.158, 1.012,
0.86], mean 1.0685, **sd 0.1453**. 임계까지 gap 0.0815 ⇒ **gap < sd**.

| blocks | CI 상한 | 상한 ≤ 1.15 확률 | 추가 GPU |
|---|---|---|---|
| 12 | 1.161 | 43% | +3.0 h |
| 16 | 1.146 | **55%** | +3.0 h |
| 24 | 1.130 | 75% | +6.0 h |
| 32 | 1.121 | 87% | +9.1 h |
| 40 | 1.115 | 93% | +12.1 h |

⇒ 제안된 8→12–16은 **동전던지기**. 실질 확증은 n ≈ 32–40 = 12–15 GPU-hour.

**희석 논거(내 주장 A — 나중에 REFUTED)**: 실현률이 4–19%뿐이므로
`A_free(dD) = w_D·A(D) + (1−w_D)·A(108)`이고 희석이 비를 1 쪽으로 끌어당긴다.
Ha8 역산 시 보정 g ≈ **1.62–1.70** ⇒ "flat 판정은 engagement를 잰 아티팩트일 수
있다" ⇒ 격자 수정이 블록 증설보다 선행.

**이 시점의 결정**: GPU를 쓰기 전에 두 갈래를 **병렬로** 검정 — claims-auditor에
희석 논거를 적대적으로 회부, result-analyst에 "decode가 왜 비는가"를 분해.
회부서에는 지난 세션의 메타 교훈을 반영해 **"내 주장과 그 반대가 공유하는 전제"를
명시적으로 검정 대상에 넣으라**고 지시했다.

### 7.3 result-analyst — 내 진단 프레이밍이 틀렸다

- `E1_DECODE_REALIZED`는 **decode-active 시간에 조건부**라 decode 공백은 정의상
  분자·분모 어디에도 안 들어간다. 실측 기여 T8 −0.0006 ± 0.0046 / Ha8
  −0.0001 ± 0.0028 = **0**.
- 부하창 내 decode-empty는 **0.6–1.3%**뿐. 원자료의 15–19%는 **측정창 아티팩트**
  (클라이언트 warmup → dataset 준비 갭 14.9–17.7 s + 종료 후 꼬리 5.2–6.2 s).
- ★**희석의 원인은 prefill 부재**. block-paired (d54−d16) 분해에서 **gradient의
  100%가 prefill 점유율 gradient로 설명**(잔차 CI가 0 포함).
- ★**realized 천장은 워크로드 성질이고 arm마다 3배 다르다**(ΣTTFT/decode-busy로
  T8 0.120–0.154 / Ha8 0.381–0.558). 클라이언트 측 양이라 텔레메트리 계측 문제에
  면역. ⇒ **부하를 올려도 engagement는 안 오른다**(decode는 이미 ~99% busy).
- 게이트 #6 필드 감사에서 **죽은 필드 4개**: `decode_ready_queue_depth` 항등 0
  (dual-worker 가드 안에서만 채워짐), `active_decode_sequences` ≡
  `decode_running_batch_size`, `decode_idle_ratio`/`prefill_idle_ratio` 항등 0.0
  (선언만 되고 대입 없음). ⇒ 내가 프롬프트에 쓴 H-starve 예측 조건이 **정의상
  발생 불가**였다.

### 7.4 claims-auditor — 주장 A는 REFUTED

독립 증거 3줄:

1. **control-arm reductio**: 같은 보정식을 T8에 넣으면 corrected g = **21–29×**.
   정본 C2(2.36–2.91×)를 더 좁은 16→54 구간에서 10배 위반. 모형은 T8 d16의
   split-조건부 ITL p95를 352–360 ms로 요구하는데 **실측 30.67 ms**.
2. **de-engagement 직접 실험**: split 라벨 토큰을 같은 셀 unsplit 분포에서
   재추출해 engagement → 0으로 보내도 `A_free`는 **1–11%만** 변한다.
   w=0에서 **g = Ha8 1.173 / T8 1.796** — 헤드라인이 거의 그대로 남는다.
3. **"A(108) 셀 무관" 가정 위반**: 파티션이 실제 실현된 토큰만 보면
   **Ha8 g = 0.920 [0.842, 0.998]** — CI가 1을 배제하고 **부호가 반대**.
   T8은 헤드라인 1.837이 **UNSPLIT-only(대비가 정의상 0인 모집단)**에서
   1.795로 그대로 재현된다.

**유형**: confound #1(서빙 직접 측정을 오프라인 산술 모형으로 뒤집으려는 이동) +
#6(`w`는 자유 모수가 아니라 **prefill duty cycle 그 자체**인데 나눔).

**살아남은 것**: engagement가 낮다는 **전제 자체는 견고**(세 계측기 교차확인).
죽은 것은 보정이지 전제가 아니다.

### 7.5 주장 B — 결론은 맞고 근거는 틀렸다

"블록 증설 선행 금지"는 옳다. 단 이유가 다르다:

`pdmux_context.py:initialize_stream_groups`가
`SM_COUNTS = [(108,0)] + divisions + [(0,108)]`를 하드코딩하고,
`multiplexing_mixin.py:773,792-794`가 prefill이 in-flight가 아니면 마지막 plain
`(0,108)`로 되돌린다. 따라서 이 기판에서

> **"decode가 D SM에서 돌았다"와 "prefill이 동시에 실행 중이었다"는 같은 사건이다.**

⇒ **어떤 통계량도 decode-SM 레버와 prefill 간섭을 분리할 수 없다**(estimand 미식별).
§1-24(M4)와 결합하면 `g`는 **decode-SM 라벨을 단 prefill-SM 탄력도**일 것이 사전
예상되고 실측이 일치한다. **n으로 해결되지 않는 설계 결함.**

부수로 `A_free` 자체의 결함도 확정됐다: `PREFILL_BLOCK_TOK=1024` 필터가 prefill
작업의 **74–77%를 놓치고**(꼬리를 만드는 요청 input이 221–804 tok로 전부 임계 아래),
요청의 **27.5–29.5%가 outlen ≤ 25**라 요청별 p95가 max로 퇴화한다.
그리고 arm 간 비교에는 **decode batch size 교락**(T8 4.5 vs Ha8 15.8)이 남아
사전등록 분기의 "attributable to the arm" 문구를 현재는 쓸 수 없다.

### 7.6 격자 수정의 방향 — 세 후보 중 하나만 산다

| 후보 | 판정 |
|---|---|
| rate를 올려 engagement를 높인다 | **REJECT** — engagement는 rate가 아니라 워크로드의 prefill:decode 작업비가 정한다(두 에이전트 독립 수렴) |
| 워크로드를 prefill-heavy로 바꾼다 | **REJECT(식별 목적)** — engagement는 오르지만 split⟺prefill-in-flight 별칭을 못 깨고 오히려 악화 |
| **파티션을 sticky하게 만든다(기판 수정)** | **유일하게 식별을 회복** |

config로는 불가능함이 확인됐다(`initialize_stream_groups`가 마지막 무분할 그룹을
무조건 덧붙임) ⇒ 엔진 변경 필요.

### 7.7 대체 추정량 이관

`A_free`를 **조건부 per-token 추정량**으로 교체했다(`m3_conditional.py`).
단위를 개별 ITL 구간 1개로 내리고 요청별 내부 집계를 없앤 것이 두 병리 회피의 전부다
(T8 d16 pooled per-token p95 **11.60** vs `A_free` **28.31**; 셀-블록당
1,390–7,045 토큰 위에서 계산 vs `A_free`는 요청 ~9.6개에 얹힘).
라벨은 SPLIT ≥ 0.90 / UNSPLIT ≤ 0.10 / 사이는 배제(AMBIGUOUS 실측 0.3–1.4%),
**UNSPLIT control이 안전장치**(대비가 정의상 0).

★**임계 스윕이 결정적이었다**(단 [UNAUDITED]): `PREFILL_BLOCK_TOK` 1024→0에
**무릎이 없고**, 임계 0에서 **d16-vs-d54 대비가 두 arm 모두 소멸**
(Ha8 0.986, T8 1.005). ⇒ 임계는 자유 모수가 아니라 **답을 정하는 손잡이**이고,
`A_free`의 셀 대비 전부가 prefill-blocked 꼬리에 산다.

정렬 규율도 확정: `phase=="benchmark"` 마커는 **warm-up 요청**에 발화하므로
probe 경계가 아니다(필터 금지). ALIGN-WEAK는 배제가 아니라 flag — LOO 실측에서
가장 약한 block을 빼면 T8 대비가 1.688 → **1.822로 오히려 커진다**.

### 7.8 sticky 구현과 관측

`PDMUX_STICKY_PARTITION` 구현 + correctness gate 통과(job 872800, ~9분).
되돌림 경로 5곳 전수 처리, `event_loop_pdmux_coord`와의 조합은 init에서 거부,
OFF는 short-circuit으로 patch 이전과 동등(독립 재구현 selector 대비 전 격자 테스트),
cudagraph 보존, OFF/ON greedy 출력 6개 byte-identical.

**핵심 관측(n=1, 성능 아님)**: `E1_DECODE_REALIZED` **OFF 0.0839 → ON 1.0000**.
사전등록 0.90을 넘겼고 맞추려 튜닝한 것이 없다. ⇒ 이 게이트가 **항등식이기를 멈추고
진짜 게이트가 됐다** — 그것이 이 패치가 주장하는 전부다.

---

## 8. 지금 서 있는 자리와 다음 결정

**닫힌 것**: PD 분리 이득 · layer-aware 全형태 死 · 단일-GPU 동적 死(HE0) ·
short-ctx disjoint 부재(scoped).

**열린 것**: **긴장 A(HE2 vs C2)는 전혀 닫히지 않았다.** "Ha8에 decode-SM 레버가
없다"도 CONFIRMED가 **아니다** — 현재 데이터는 그 질문에 답하지 못한다.
long-context와 spatial decoupling(§1-20)도 미실행.

**다음 두 스텝(순서가 중요)**:

1. **`G_LEVER`/`G_FLAT` 앵커링 (GPU 0)** — 기존 1.5/1.15는 `A_free`(이중극단)
   스케일 값이라 per-token 분위수로 그대로 옮기면 **사후 재단**이다. 살아있는 논거는
   새 추정량이 C2와 **같은 축**이라 `G_LEVER`를 C2 측정 범위에 묶는 것인데, E1은
   D 범위가 16→54로 좁고 **complementary**(prefill = 108−D가 함께 움직임)라
   C2 값을 그대로 쓸 수 없다. 제안: 기존 `results/s8_scaleup/`에서 **16→54 부분비**를
   직접 계산해 하한 앵커로 삼고 그 한계를 사전등록에 명시.
2. **sticky 격자 본 런 (~3.0 h GPU)** — 872077이 non-sticky 대조군이므로 paired
   A/B가 된다. **사전등록 판별 예측**: 꼬리가 prefill 주도라면 Ha8 ≈ **0.92** /
   T8 ≈ **1.85**로 내려가고, 주장 A가 옳았다면 Ha8 ≈ **1.6**으로 올라간다.
   CI 비중첩이므로 **8 block으로 구분된다** — 블록 증설은 이 판별 이후에만.

⚠️ 1번이 2번보다 먼저다. 임계를 정하지 않고 제출하면 데이터를 본 뒤 임계를 고르게
되고, 그건 이 프로젝트가 이미 당한 실패 모드다.

---

## 부록. 이 프로젝트가 실제로 배운 것 — 반복된 실패 패턴

방법론 게이트는 전부 **구체적 사고에서 태어났다**. 순서가 곧 이유다.

| # | 게이트 | 태어난 사건 |
|---|---|---|
| — | 정책 주장은 서빙 실증 | micro-measurement 2회 오도(GIL-client, tiny-batch) |
| — | 변화 trace · n≥4 · metric cliff 회피 | stationary r8이 static조차 ±1.302 |
| — | duration은 합산 | 하네스 3× 부풀림 |
| #4 | 집계 단위 선확정 | E1 하네스에서 답이 **5번** 바뀜 |
| #5 | 같은 플래그가 arm마다 다른 손잡이 | `--max-running-requests` |
| #6 | **항등식을 증거로 쓰지 마라** | 철회 8건 중 3건이 항등식·공허참 |
| #7 | **게이트가 재는 양이 통과 조건과 독립인지 증명하라** | pin 게이트가 항등식, `MIN_PA_SNAPSHOTS`가 estimand의 여집합 |
| — | pin은 target 아닌 **realized**로 검증 | Stage 0 D108이 실은 16 SM |

**세 번 반복된 실패 유형**:

1. **라벨 ≠ 실현** — Stage 0 D108(prefill 축) → pin 게이트 항등식 → decode 실현
   4–19%(decode 축). 매번 "설정한 것"과 "실제로 돈 것"을 혼동했다.
2. **항등식을 증거로** — `arrival_rps` · `kv_mamba_occupancy` · pin 게이트 ·
   `w`를 자유 모수로 나눈 희석 보정.
3. **틀린 두 가설 사이에서만 논쟁** — pin void를 두고 "표본 잡음" vs "진짜 언핀"을
   다퉜는데 둘 다 *"게이트가 재는 양이 옳다"*를 공유해서 답에 닿을 수 없었다.
   ⇒ **경합 가설이 공유하는 전제를 명시하고 그것 자체를 검정 대상에 넣어라.**
   이번 세션에 이 규율을 회부서에 넣었고, 그 결과 주장 A가 GPU를 쓰기 전에 죽었다.

★ 마지막 항목의 새 판본(2026-08-03): **진단과 처방을 같은 턴에 하면 처방은 자기가
감사한 것이다.** 주장 A를 반증한 주체와 대체 추정량을 내놓은 주체가 같았고,
그래서 임계 스윕 표가 [UNAUDITED]로 남아 있다.
