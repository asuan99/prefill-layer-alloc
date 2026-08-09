# 세션 핸드오프 — 2026-08-09

세션 범위: **2026-08-06 → 2026-08-09**. 직전 핸드오프:
[`session_handoff_2026-08-06.md`](session_handoff_2026-08-06.md).
용도: **다음 세션의 후속 작업 검수·계획**. 후속 사다리는 §5, 검수 대상은 §6.

---

## 1. 이번 세션 요약

세 트랙(Gate 1 후속 · Gate 2 설계·실행 · 통계 도구 정정)을 병행했다. **GPU job 17건,
19.30 GPU-hr**(`sacct -S 2026-08-06` 실측, 874465–875661)를 썼고 **새 성능 판정은 0건**이다.
가장 비싼 둘은 E-A(875657 5:41:36 · 875661 3:53:16 = 9.58 GPU-hr, 전체의 50%)와
Gate 2 rev4 본 캠페인(875344 2:58:48 · 875346 2:52:48 = 5.86, 30%)이다. 움직인 것은 전부 **계측·통계 타당성**이며
방향은 예외 없이 기존 주장을 **약화**시키는 쪽이었다. 정본은 **rev12 → rev16**(4회 개정),
방법론 게이트는 **19개**로 늘었다.

관통하는 실패 유형 하나가 **다섯 번** 반복됐다 — **측정 실패를 게이트 실패로 라벨링**하는 것
(텔레메트리 드롭 카운터 자기검열 / `UNSCOREABLE`을 강등으로 읽음 / HTTP 400을 G3 FAIL로 /
G5의 귀무 채택형 기준 / 프로브의 stderr 오염). 네 번은 다른 주체가, 한 번은 실행자 자신이 잡았다.

★ **가장 값어치 있는 발견은 실험 결과가 아니라 도구 결함이다**: `paired_bootstrap_ci`의
n=5 undercoverage가 **미제출 E1의 결정 규칙까지 한 방향으로 편향**시키고 있었고, 저장소는
같은 진단을 이미 **두 번** 내놓고도 정본 라이브러리에 반영하지 않은 상태였다.

---

## 2. 결정·측정 (트랙별)

### 2.1 통계 방법 층 정정 → 정본 rev13 [확정, 2출처 독립 재현]

원 지적 claims-auditor, 독립 재현 result-analyst(+ N-1은 메인 세션 자체 구현).
산출물 `results/p1_gates/verify/`.

- **`paired_bootstrap_ci`(`benchmarks/pdmux_eval/analyze.py:115-142`)는 소표본 판정 불가**:
  n=5 percentile bootstrap of the mean, 실제 coverage **0.8395**(정규) / **0.8254**(이 데이터
  경험분포) / **0.723**(왜도 시). 한쪽 오류율 ≈8.0%. **원인은 고정 seed도 정규 가정도 아닌
  n=5 자체**. n별: n=4 0.798 / n=5 0.840 / n=6 0.859 / n=8 0.888. `unpaired_bootstrap_ci`도 동일.
  ⚠️ 폭 비는 감사자가 말한 0.58이 아니라 **0.624**(자기산출 3개 중 이것만 오류).
- **§1-1의 "Granite rate3 +11.7%, CI 0 배제"는 성립 안 함** — paired-t **[−0.0032, +0.6132]**,
  p=0.0515. Granite r2도 사망(p=0.300). ⇒ **인용 가능 정량치 4개 → 3개**
  (Zamba2 r2 +40.5%, r3 +185.8%, Granite r4 +27.0%).
- **"임계 사다리 40–300ms 전 구간 부호 불변"은 거짓** — 부호 불변 범위는 **T ∈ [40, 113.0) ms**.
  그 위는 절벽이 아니라 **술어 포화**(양 arm 위반 0 ⇒ goodput ≡ throughput).
- **"rep 부호 5/5"도 거짓** — Granite r2는 **3/5**, 효과의 87%가 rep2 한 점.
- ★ **구조적 사실**: n=5 paired 정확 순열검정의 두측 p 하한 = **2/32 = 0.0625** ⇒
  **n=5는 분포무가정으로 p<0.05에 도달 불가.**
- ⇒ 신규 방법론 게이트: n≤8에서 bootstrap CI를 판정에 쓰지 않는다. **E1 제출 선행조건**으로
  `s8_frontier/DESIGN.md` §4.4의 결정 규칙을 t로 교체할 것을 등재.

### 2.2 Gate 1 / G1-b → 정본 rev14 → **rev15에서 철회**

- **job 874465**: rate=3 창 telemetry 0행. 커버리지 가드 부재로 rate=2도 **80.11% 절단 창**
  위에서 `UNLOCK-eligible`이 나왔다 ⇒ 가드 추가 후 재채점하여 **UNMEASURABLE**로 정정.
- **job 874478**: 두 창 coverage 100%, pop A 시간가중 100% `(74,34)`, 음성대조 B 1.0000·C 0.97.
  **판정 = 조건부 채택.** ⚠️ 주 조건은 **코드 항등식**임이 감사로 확정
  (`multiplexing_mixin.py:1004`↔`:1080` 사이에 telemetry sync가 없어 반례 상태가 관측 불가,
  반례 0/40,062). 살아남는 산출은 부산물 둘 — `(54,54)` 미선택, duty-cycle.
- **job 875293 (G1-b)**: rate {2,3,4,6} 단일 부팅. 네 창 coverage 100%·grid 결측 0.
  양성대조(rate 2·3이 874478 재현) PASS. **rate 6에서 `max(decode_bs)=40 > 문턱 36`,
  `(54,54)` 시간가중 8.37% ⇒ 사전등록 철회 규칙 발화.**
  ⇒ rev14의 "이 격자에서 단일 분할 고정" 문장 **철회**(rev15). 인용 가능 셀 r2·r3에서는 여전히 참.
  ⚠️ **rate 4는 18로 rate 3(23)보다 낮아 비단조 — 미해명.** rate 4만 돌렸다면 확장 규칙이
  발화해 **반대 방향 오류**를 저질렀을 것이다(신규 방법론 항목).

### 2.3 Gate 2 설계 — rev1 → rev4 (설계 감사 3회)

| rev | primary | 죽은 이유 |
|---|---|---|
| rev1 | goodput | **A4 바닥 절단** — "대체 가능"을 원리적으로 출력 못 함 |
| rev2 | `frac_stalled(60)` 위 φ | **A1 천장 절단** + **φ가 단조 재표현에 불변 아님**(φ=0.80서 토큰당 stall 13.5× 차) + §5.4가 goodput 절단 재수입 |
| rev3 | τ=40 TOST | **τ=40이 metric cliff**(A4 max-ITL 질량 52–87%가 [35,40), ±3% 섭동이 마진의 40–60% 소모) + 근거 문장 사실오류 |
| **rev4** | **τ=60 TOST, n=10** | 제출됨 |

τ=60 선택 근거는 검정력이 아니라 **결과-독립 3가지**(정본 SLO 상수 = 외부 앵커 / ±3% 섭동에
5셀 전부 |ΔX|=0.0000 / τ∈[55,80] 평탄대). ⚠️ **`Var(A3−A4)`가 사전등록 시점에 식별 불가**
(A2·A3 미실행) — n=10은 대리 위에 서 있고 그 대리는 상한이 아니다.
⚠️ **§11-3(3차 감사)은 사용자 지시로 면제된 채 제출**(§14.1 자백) — "감사 통과 설계" 아님.

### 2.4 HOLB 직접 계측 구축 (jobs 874601/874602/874633/874635)

`stall_ms` = decode 대기 중인데 decode forward가 없는 GPU-타임라인 ms. arm 대칭성이
**창 분해 항등식**(`T = decode_fw_ms + stall_ms + stall_ambiguous_ms`)으로 성립.
⚠️ 메인 세션이 처음 제안한 `blocking_fw_ms`는 **A4에서 구조적 0**(항등식)이라 engine-porter가
기각했다 — 항등식을 피하려는 설계에서 항등식을 재생산할 뻔한 사례.

- **G3(프로브 ON/OFF greedy 동일) PASS**, **G4(계측 타당성) 40/40 PASS**(`timeline/host` 편차 0.02%).
- ⚠️ **G5 채점 기준 결함**: `PASS = |effect|<3% ∧ CI가 0을 포함`은 **귀무 채택형**이고
  Gate 2 감사가 rev1의 R1/R3를 죽인 바로 그 오류다. 등가검정으로 재해석하면
  **3% 초과 입증 0건**, 나머지는 검정력 부족. **현행 `G5=False` 기록은 오해를 부른다 — 재채점 필요(§5).**

### 2.5 Gate 2 rev4 본 캠페인 (jobs 875344/875346) → 정본 rev16 이전 [감사 완료]

★ **이 세션 최대 성과**: **R2′ 확립 — §1-1의 "3-플래그 묶음 처치" 교락 해소.**
A2(`plain+aux`) vs A4가 5셀 전부 A4 우세(perm p = 0.00195 = n=10 하한, Holm 후 0.00977).
크기 인용 가능: Zamba2 r2 +0.834 · r3 +0.930 · Granite r3 +0.855 · r4 +0.944.
A2 vs A4의 realized `server_args` 차이는 **`enable_pdmux`·`pdmux_config_path` 두 필드뿐**이고,
묶음 기여는 A4−A2의 5% 미만이며 **부호가 A4에 불리한 핸디캡 방향**이다.
⚠️ **해금 범위 = "pdmux 서브시스템 전체"(green-context SM 분할 + 전용 루프 + split-prefill)**,
**"SM 분할 자체"는 여전히 미분리**(`--enable-pdmux`가 이벤트 루프를 통째로 교체).
⚠️ X_60 위의 A1≈A2는 **천장 절단 위의 등가** — 비절단 지표에선 A2가 A1보다 median ITL
6.2–10.7% 나쁘다.

R4′(chunk512 미대체)는 5셀 전부 발화하나 **크기 인용 가능 셀은 Zamba2 r2·Granite r3 둘뿐**
(나머지는 F-E 발화). ⚠️ **F-E 플래그가 계산만 되고 판정에 집행되지 않아** 메인 세션이 5셀
전부 크기를 인용하는 실수를 했다(이후 E-A에서 수정).

### 2.6 G-2 음성대조 (jobs 875610/875611)

`plain`/`plainaux`/`agnostic` = **0/96**, `chunk512` = **6/96**(전부 Granite), Fisher p=0.0288.
⇒ 배치 비결정성 가설 반증, **chunk512 고유**.

### 2.7 E-A — mixed-chunk 레버 (jobs 875654/875657/875661) → 정본 rev16 [감사 조건부]

- **Stage 1**: `--enable-mixed-chunk`가 4콤보 전부 realized 확인(3중 증인). **레버 존재.**
  ⚠️ 프로브 자신이 `AVAILABLE_BUT_BURST_ERROR`를 오보(`:205`의 `2>&1` stderr 오염) — 원 JSON
  8건 재파싱으로 정정.
- **Stage 2**: primary 8건 전부 TOST 미발화·agnostic 우세(부호 10/10) ⇒ **신규성 축 전환 없음.**
  ⚠️ 7/8이 F-E 발화 ⇒ **부호만.** cmp2의 Granite 두 셀은 G-2 실패로 상속 §2.1 폐기 규칙 하 **무효**.
- **인용 가능 정량치 1개**: Granite r3 `X_60(plainmix) − X_60(agnostic) = +0.855 [+0.814, +0.896]`.
  ⚠️ **그중 mixed-chunk 고유 기여는 +0.010 = 1.2%**, 나머지 98.8%는 기존 fused↔pdmux 격차 ⇒
  **primary가 레버를 격리하지 않는다.**
  ⚠️ 같은 셀에서 **TTFT 항은 통과**(plainmix가 agnostic보다 19.2% 좋음). "TTFT 비열등 0건"은 거짓.
- **신규 기전(서빙 직접 측정)**: mixed-chunk는 decode를 `ForwardMode.MIXED` extend 경로로
  재라우팅하고, MIXED step 비용이 **병합 decode 수에 선형**(Zamba2 ≈33 ms/req, Granite ≈4.2)이다.
  순수 DECODE는 9.4 / 6.8 ms. ⇒ 정본 §1의 **얽힘 死因의 새 트리거**(새 기전 아님).
- **cudagraph 가설 반증**: plainmix의 남은 DECODE step은 `cuda graph: True`이고 비용도 동일.
  ★ **부수 정정 획득**: rev4 §1.1의 "piecewise 8192→512 ≥2-기전 묶음" 캐비어트는 **런타임
  실측으로 무효**(전 arm piecewise OFF) ⇒ 인용 금지 1건 해제.
- **G-2 재분류**: correctness가 아니라 **reproducibility** 결함. 두 출력 모두 퇴화 반복 루프이고
  엔진이 배치 형태 간 비트 재현성을 약속하지 않는다. 불일치는 주기 3의 31개 지점, 첫 갈라짐
  토큰 4, 치환 62→322, arm 간 바이트 동일. 3회 독립 재현 + 음성대조 4 arm 통과.
  ⇒ **mamba state 이월과는 정합하지 않음.**
- ⚠️ **E-A는 Gate 2 본 질문을 전진시키지 않았다** — A2를 뺐으므로 "PD 분리 자체" 귀속은 불변.

---

## 3. 코드·문서 변경

### 정본 (doc-steward, 4회 개정)
| 파일 | 변경 |
|---|---|
| `reports/CONSENSUS.md` | **rev13**(P1 통계 방법 층 정정) · **rev14**(Gate 1 조건부 채택) · **rev15**(G1-b 철회) · **rev16**(E-A 조건부). §3 방법론 항목 27–33 신설 |
| `PROJECT_STATUS.md` | "확정된 결과" 1번 4회 갱신, 방법론 게이트 #14–#19 신설, "다음 실험 gate" #8(E1 선행조건)·#10(Gate 1 G1-a~d, Gate 2 사다리) |
| `CLAIM_EVIDENCE_MATRIX.md` | 인용 없음 확인(4회), 갱신 대상 없음 |

### 신규 코드·하네스
- **엔진**: `src/multiplex/holb_probe.py`(신규 600행) · `src/patches/holb_probe_scheduler_hooks.patch` ·
  `tests/test_holb_probe.py`(38) · `scripts/bootstrap/sync_engine_tree.sh`(manifest +2줄)
- **Gate 1**: `gate1_run.sbatch`·`gate1_analyze.py`(커버리지 가드) · `gate1b_run.sbatch`·
  `gate1b_analyze.py`(`max_decode_bs` + grid-completeness)
- **Gate 2**: `g2_run.sbatch`·`g2_analyze.py`·`g2_concurrent_gate.py` ·
  `g2_holb_observer.sbatch`·`g2_holb_analyze.py`·`g2_holb_phaseA_lib.sh` ·
  `g2ctrl_*`(음성대조) · `g2ea_*`(E-A, 하네스 결함 6건 수정 반영) · `g2eaprobe_*`
- **검증**: `verify/verify_c1_*.py`·`verify_c2_c3.py`·`verify_g2_tau*.py`·`verify_g2_cliff_sensitivity.py`
- **사전등록 6건**: `PREREG_GATE1_2026-08-06.md` · `PREREG_G1B_2026-08-07.md` ·
  `PREREG_GATE2_2026-08-06.md`(rev1→rev4 + §14) · `PREREG_G2CTRL_2026-08-07.md` ·
  `PREREG_G2EA_2026-08-07.md` · `DIRECT_BLOCKING_DESIGN.md`

### 커밋
이 세션에 **git-committer 4회** 실행. `.gitignore` 정책을 `gate1/` → `gate2/` → E-A 명명으로
확장("harness+verdict 추적, raw dump 제외"). ⚠️ **`main == origin/main`이며 reflog에
`2026-08-07 14:48:49 update by push`가 있다 — 이 세션의 에이전트가 한 것이 아니다**
(전부 push 금지 지시를 받았고 활동 구간도 불일치). 사용자 또는 다른 세션의 push로 보인다.

---

## 4. 미완·주의 (검수 시 반드시 볼 것)

1. **인용 금지 목록이 커졌다.** rev16에 13건, rev15·rev14에도 각각 목록이 있다. **정본을
   인용하기 전에 §1-1의 금지 목록을 먼저 읽어라.**
2. **미감사 상태로 남은 것**: HOLB G5 재해석(등가검정 재채점 미실행) · `results/s8p_prefill/`
   (기존 미감사 유지) · E-A의 사후 A3-vs-A4 계산.
3. **사전등록 미준수 3건**(E-A 감사 지적): `tree_cache` 클래스 런타임 미기록(코드 추론만) ·
   rev4의 §1.3 mixed-chunk 프로브 미실행(E-A가 사후 수행) · Holm 가족을 job별로 분할(덜 보수적).
4. **F-E 집행이 primary에만 걸린다** — `g2ea_analyze.py:662-680`의 `secondary` 블록은 무방비이고,
   실제로 그 경로에서 인용 사고가 났다.
5. **사후 결정 2건이 결과에 영향**: G1-b의 Granite r6 제외(중립, 보수적) / E-A의 지정 셀
   r4→r3 이동(**중립 아님** — r4 유지 시 인용 가능 셀 0개였다).
6. **방치된 job 없음**(`squeue` 비어 있음). voided rep 0건.
7. ⚠️ **요약 단계의 강도 상승** — 메인 세션의 진단·요약이 이 세션에만 **여섯 번** 반증됐고
   전부 같은 형태였다: 정확히 분석해 놓고 다음 메시지에서 요약하며 강하게 만듦
   ("rep 하나 이상치" → "전부 3% 미만" / "7/8 F-E 발화" → "8건 전부 위반" / 필드명 미확인 →
   "하네스가 또 실패"). **다음 세션도 요약문을 원자료로 되짚어라.**

---

## 5. 후속 사다리 (감사 지정, 비용순) — **다음 세션의 계획 대상**

| # | 실험 | 무엇을 닫나 | 비용 |
|---|---|---|---|
| **T4-1** | `--enable-deterministic-inference`로 Granite chunk512 G-2 재실행 | 상속 §2.1 **폐기 규칙 해제** + reduction-order 귀속 확증 | **<0.5 GPU-hr** |
| **T3-1** | HOLB에 MIXED step의 `extend_num_tokens`·`#new-seq`·병합 decode 수 기록 후 1 rep | 33/4.2 ms 기울기를 prefill 토큰 수와 분리 | **<1** |
| **T3-2** | micro: prefill 2000 고정, 병합 decode {0,8,16,32,48} 스윕 | 기울기 확정 | **<1** |
| **E-C** | 등지속가능-rate 대조(5 arm × rate {1,1.5} × n=10, 2모델) | **T1 정식 종결** — "과부하 vs 정상" 혐의 제거. 현재는 n=3 탐색 데이터뿐 | **≈4** |
| **E-D** | 미시험 fused 레버 단일-플래그 스윕: `prefill-max-requests 1` · `num-continuous-decode-steps {2,4}` · `chunked-prefill-size {2048,4096}` · `max-running-requests 16` (+`enable-prefill-delayer` 적용성 프로브) | **T2 종결** — "fused 조율 공간 소진" 주장 가능 여부 | **≈8** |
| **T3-3** | backend 교차(Zamba2×flashinfer / Granite×triton) | MIXED 한계비용 **8× 비대칭**이 모델인지 backend인지(confound #10) | **≈2** |
| **T4-2** | 비퇴화(실제 텍스트) 프롬프트 G-2, 동시성 {4,16,32}, 5 arm 전부 | 퇴화 루프 인공물 여부 + **캠페인 내부 음성대조 확보** | **≈1** |

**GPU 0 항목**: HOLB G5 등가검정 재채점 · `X_60` 추출(rev4 §5.1.1 분산) ·
교훈 3건 정본화(874601 HTTP 사건은 srv 로그 줄번호 인용 예외 선행 필요) ·
`g2ea_analyze.py`의 secondary 블록 F-E 집행 · Holm 가족 통합 · §5.1.1 중간점검 SD=0 퇴화 규칙.

★ **Gate 2 본 질문은 여전히 열려 있다** — A2를 포함한 원 설계가 미실행이라 "PD 분리(SM 분할)
자체" 귀속은 rev4 시작 시점 그대로다. E-A는 이를 건드리지 않았다.

---

## 6. 다음 세션 시작점 (한 줄)

*`catch-up` 후 **T4-1(<0.5 GPU-hr)**부터 — 상속된 A3 폐기 규칙을 풀어 Gate 2 rev4의 Granite
두 셀을 되살릴지 결정하고, 이어 **E-C**가 "과부하 vs 정상" 혐의를 정식으로 닫는다. 가장 비싼
**E-D**(T2 종결)는 그 뒤.*
