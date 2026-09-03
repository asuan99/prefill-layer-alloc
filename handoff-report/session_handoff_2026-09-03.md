# 세션 핸드오프 — 2026-09-03 (2026-09-01 저녁 개시, 3일 연속 세션)

## 이번 세션 요약

`/catch-up`으로 시작해 **step 3→1→2**(원래 질문 재설계 · plateau 술어 등록 · rev4 잔여
항목)를 수행한 뒤, 사용자 지시로 **실험 단계**에 들어갔다. 그 결과 이 트랙이 다섯 번
붕괴한 **구조적 원인이 처음으로 측정으로 규명**됐고(§2-4), 그 위에서 CP-2를 두 번 다시
설계해 **5회차·6회차 규칙층 감사를 받았다(둘 다 `NO-GO`)**. GPU **18 job · 5.78 GPU-hr**.
★**원래 질문(chunked prefill vs pdmux 정책 비교)의 정책 판정은 여전히 0건**이지만, 이 세션은
*"왜 0건인가"* 를 추측이 아니라 측정으로 답했다 — 그리고 그 답은 **세 층(추정량·arm·워크로드)이
동시에 막혀 있었다**는 것이다.

**정책 판정 0건 · arm 순위 0건 · 등급 변경 0건 · 정책 순위 변경 0건 · 정본(PROJECT_STATUS/
CONSENSUS) 편집 0건.** HE0 · 정책 순위 · gate #13/#16 "닫았다" 금지 · switch-cost "닫았다"
금지 · C2 인용정지 (a)(b) · `CONSENSUS §1-24` **전부 불변**.

---

## 1. GPU 지출 (18 job, 전부 `COMPLETED 0:0`)

| 캠페인 | job | GPU-hr | 산 것 |
|---|---|---|---|
| V-probe (Zamba2) | 900411·900423–900436 (8) | 2.16 | 무처치 분산 |
| W1 스모크 | 900671·900680·900767·900768 (4) | 0.50 | 배관·모델 상한·사다리 시작점 |
| 모델 부팅 스모크 | 900731·900752 (2) | 0.36 | 모델×백엔드×모드 지원 표 |
| W1 본캠페인 | 900784·900785 (2) | 1.56 | 장문 체제 knee + 분포 위치 |
| correctness gate | 902397·902407 (2) | 1.20 | arm 간 greedy 출력 일치 |
| **합** | **18** | **5.78** | |

---

## 2. 측정으로 확정된 것 (전부 새 측정)

### 2.1 V-probe — 5회차 감사 死因 H1 확증 (오히려 과소평가)
`RESULT_VPROBE_2026-09-01.md`. job-내 페어 무처치 Δ SD = `fused_default` **1.99%** /
`d44` **6.13%**. ⇒ CP-2 rev1의 3% 마진에서 `equivalent`는 **어느 arm에서도 도달 불가**.
★**스모크 1쌍(+0.39%)이 반대 결론으로 오도**했고 n=4에서 뒤집혔다 — "n≥4 없이 결론 금지"가
계측 축에서도 작동한 사례. ⚠️arm과 노드가 완전 교락(gpu42/gpu40)이라 **두 SD의 비교는 금지**.

### 2.2 운영점 재배치 — 잡음은 줄지만 **arm-중립 운영점이 없다**
`RESULT_POSITIONING_AND_WORKLOAD_2026-09-02.md`. 통과율 6%→52%로 옮기면 SD가 11.4%→0.7%
(16배). **그러나** HI의 요청별 ITL-p95가 arm마다 다른 단일 모드에 몰려 있고
(`fused` 71.8%가 [70,80)ms · `d44` 81.8%가 [40,60)ms) **등록 임계 60ms는 `d44` 모드
상단에서 1.2ms 위**다. ⇒ 통과율로 운영점을 고르는 행위 자체가 arm을 고르는 행위.
★그 사실을 확인하느라 **이 데이터로는 더 이상 운영점을 고를 수 없게 됐다**(§2.1에 기록).

### 2.3 ShareGPT 전수 센서스 — 처치가 발화한 적이 없었다
`RESULT_SHAREGPT_CENSUS_2026-09-02.md`, **92,824행 전수**. `cps 8192` 초과 프롬프트가
캡 4000 하에서 **0건**, 캡 무제한에서도 **56건(0.1%)**. ⇒ CP-2 rev1이 CP 가족 대표로 등록한
`fused_default`는 **요청 단위 chunking이 한 번도 일어나지 않는 arm**이었다.
장문 꼬리(floor 2048)는 공급 1,633–1,968건이고 **출력 길이가 오히려 길다**(decode 생존).

### 2.4 ★★W1 — 이 트랙이 다섯 번 죽은 구조적 원인
`RESULT_W1_2026-09-02.md`(Nemotron-Nano-9B, 32/32 셀). 두 트랙 모두 knee 도달
(A: 1.6–2.0 req/s · B: 0.32–0.47 req/s). **knee 아래에서 네 arm은 구별되지 않는다.**
★**두 가족이 서로 다른 SLO 다리에 묶여 있다**: `d44`는 요청별 ITL-p95가 전 rate에서
29–32ms 고정이고 TTFT가 561→18,559ms로 가는 반면 cp arm은 반대. ⇒ 같은 런을 재채점하면
**사다리 안에서 비교의 부호가 뒤집힌다**(TTFT2000/ITL40 cp 우세 ↔ TTFT6000/ITL60 d44 2.4×).
**"어느 가족이 낫다"는 SLO 쌍을 정하기 전에는 정의되지 않는다** — 측정 실패가 아니라 구조.

### 2.5 모델 지원 표 — 게이트 #83의 범위가 좁혀졌다
`../model_roster/RESULT_MODEL_BOOT_SMOKE_2026-09-02.md`. piecewise를 끄면 3모델 모두
장문(8,000 tok) 서빙. **`granite`·`Falcon-H1`은 triton·flashinfer 양쪽에서 돈다** —
백엔드 강제는 **`nemotron_h` 계열에만** 걸린다(자구 확인).

### 2.6 correctness gate — 통과 (사용자 수용)
`RESULT_CORRECTNESS_GATE_2026-09-03.md`. 기준 `plain` 대비 **56 비교 중 55 바이트 동일**,
퇴화 0/14, **장문(8k·16k) 전 arm 일치**. 유일 예외는 **청크 수에 비단조**(6조각 `cp512`는
일치, 2조각 `cp2048`만 불일치) ⇒ 계통 오류가 아니라 근사 동률 argmax 뒤집힘.
★**동시성은 미검증**(요청을 하나씩 보냄) — 캠페인은 부하로 도므로 **열린 구멍**.

---

## 3. 설계 결정 (사용자)

| 결정 | 근거 | 문서 |
|---|---|---|
| 데이터 **C안**(실제 trace + 합성 양 트랙) | 실제 trace만으론 기본값 질문 불가, 합성만으론 리뷰어 반론 | `PREREG_W1` §rev3 |
| 모델 **Nemotron-Nano-9B-v2 + flashinfer** | 범용성·8B 스케일 패밀리·로스터 1순위. ★세션의 granite 권고는 **철회** — 그 근거였던 "triton 유지"가 `s8_backend_control.sbatch`(이미 백엔드 대조 존재)로 무너짐 | `PREREG_W1` §rev3 |
| **piecewise CUDA graph 전 arm OFF** | falcon_h1/nemotron_h에서 캡처 크래시, 엔진이 우회책 출력 | `DECISION_PIECEWISE_OFF_2026-09-02.md` |
| correctness **(a) 수용** + 동시성 미검증 명시 | 불일치가 계통적이지 않음 | `RESULT_CORRECTNESS_GATE` §5 |

★piecewise OFF의 부수 효과: **F3/G4 교락이 닫힌다**(`cps`가 다시 단일 레버). 강제 번들
4중→3중. 대가: CP 가족이 쓸 수 있는 prefill 최적화 하나를 뺀 조건(결과 문서 동반 필수).

---

## 4. 규칙층 감사 2회 — **둘 다 `NO-GO`** (이 트랙 5·6회차)

### 4.1 5회차 — CP-2 rev1 (`audit_cp2_rules_5th_2026-09-01/VERDICT.md`)
死因 **H1–H8**, 그림자 81%. 살아남은 것: 라벨층이 트랙 최초로 견고(외부 변이 30종 중 22 사멸),
hand 오라클이 4회차 G5를 실제로 닫음, `server_args.py` assert 인용 정확 ⇒ **positivity
violation 논증은 옳다**. ★부수: `presubmit.py`가 **read-only가 아님**(타 트랙 인증서 재작성) —
그 부작용으로 **다른 세션의 미커밋 작업이 소실**됐고 이 세션이 재생성해 **복구**했다.

### 4.2 6회차 — CP-2 rev2 (`audit_cp2r2_rules_6th_2026-09-03/VERDICT.md`)
死因 **F1–F8**, **그림자 88%(트랙 최고)**. 단일 판정 질문 답:

> **"수리의 절반이 조건 등록이 아니라 **조건 삭제**였다. 삭제는 감사에 안 잡힌다 —
> 없는 것은 grep되지 않기 때문."**

세션이 **직접 재확인한 死因 3건**:
- **F2** `_AT_OP` 라벨 4개가 가리키는 **운영점 좌표가 사전등록 어디에도 없다**(규칙은
  *"the registered operating point"* 라고 씀). `RESULT_POSITIONING §2.1`이 이름으로 넘긴
  숙제를 미스윕 ⇒ 순수 전파 실패.
- **F4 (회귀)** rev1이 등록했던 **정본 스코어러·duration 합산·paired bootstrap·`N_BOOTS`·
  동반 공표·presubmit 조항**이 rev2에 **전부 0회**(grep 확인). "범위로 양화한" 중단 규칙이
  rev1보다 **더 작은 세계**를 묶는다.
- **F3** 중단 규칙 §7-1·§7-2가 만드는 세계가 둘 다 `coverage=absent` ⇒ 규칙이
  **`NOT_MEASURED`**를 낸다. H1 수리의 간판 라벨 `POWER_UNMEASURED`·`UNAFFORDABLE`은
  26세계 전부 `coverage ∈ {complete, partial}`이라 **원리적으로 도달 불가**(규칙을 직접 실행해 확인).

살아남은 것 12건: 외부 변이 47종 중 29 사멸 · **마진 출처 참**(`CLAUDE.md:74`) ·
**H4 출처 허위 재발 없음** · **`required_n`은 비순환이고 노이즈 보상 안 함** ·
hand 21행 중 산문 모순 0건 · **4회차 G2(격자 이식) 실제로 닫힘** · 스코프 정직성 트랙 최고.

---

## 5. 코드·문서 변경 (★**전부 미커밋**, 78 untracked + 3 modified)

**신규 규칙층/도구**
- `cp2_rule.py`·`cp2_predicates.py`·`cp2_selftest.py`·`cp2_expected_labels.json` (rev1, `NO-GO`)
- `cp2r2_rule.py`·`cp2r2_predicates.py`·`cp2r2_selftest.py`·`cp2r2_expected_labels.json` (rev2, `NO-GO`)
- `g1b_plateau_predicates.py`·`g1b_plateau_selftest.py` — **plateau 술어**(step 1). paired
  일측 t로 knee 탐색, band는 attainment의 pooled SD와 비교(**비순환**). ★상수를 **3종**
  (LABEL/DESIGN/INFRA)으로 분류 — `cp0_predicates.py`가 2종이라 `PROBE_RATES`가 통과할
  수밖에 없는 검사를 받았던 문제의 구조적 수리. 이 검사가 **자기 모듈의 결함 2건**을 잡았다
  (`BAND_SD_MULT` 미시험 · `MAX_RATES` 죽은 상수 → 삭제).

**신규 하네스/분석기**
- `vprobe.sbatch`·`vprobe_score.py`·`vprobe_aggregate.py`·`vprobe_positioning.py`
- `w1_probe.sbatch`·`w1_analyze.py` · `build_sharegpt_long.py`·`sharegpt_length_census.py`
- `correctness_gate.sbatch`·`correctness_check.py`·`correctness_prompts.json`
- `../model_roster/model_boot_smoke.sbatch`

**신규 사전등록/결정/결과 문서**
`PREREG_VPROBE_2026-09-01.md` · `OVERRIDE_VPROBE_SUBMIT_2026-09-01.md` ·
`PREREG_G1B_PLATEAU_2026-09-01.md` · `PREREG_W1_2026-09-02.md`(rev3까지) ·
`PREREG_CP2_2026-09-01.md`(rev1) · `PREREG_CP2R2_2026-09-03.md`(rev2) ·
`DECISION_PIECEWISE_OFF_2026-09-02.md` · `SWEEP_FINDINGS_CP0_REV4_2026-09-01.md` ·
`RESULT_*`(V-probe·positioning·census·W1·correctness·boot smoke) · 판정서 2건.

**수정**
- `PREREG_CP0_2026-08-28.md` — 감사 회차 **5→6 스윕**. ★`check_version_sweep.py` S6이
  발화했고, 이는 `SWEEP_FINDINGS §5-5`가 **미리 예고해 둔 항목**이다 —
  **예고된 전파 실패가 도구로 잡혀 수리된 첫 사례.**
- `m4r_confinement/reachability_verdict.json`·`tc1_model_attrib/reach_verdict_rev3_A.json`
  — 타 세션 작업이 감사자의 `presubmit.py` 실행 부작용으로 소실된 것을 **재생성해 복구**.
- 파생 데이터셋 4종(`$HF_HOME/raw/ShareGPT_long*.json`, 저장소 밖).

**규율 도구 상태**: `cp0_selftest`·`cp2_selftest`·`cp2r2_selftest`·`g1b_plateau_selftest`·
`check_version_sweep` **전부 rc=0**.

---

## 6. 열린 항목 / 다음 세션 시작점

1. ★**CP-2 rev3** — GPU 0. 이번엔 **재작성이 아니라 rev1에서 살아남은 등록 항목을 rev2
   구조에 합치는** 작업이다: (a) **운영점 좌표 등록**(F2), (b) rev1의 스코어러·duration
   합산·paired bootstrap·`N_BOOTS`·동반 공표·presubmit 조항 **복원**(F4), (c) 중단 규칙이
   `coverage=absent`를 만들지 않도록 §7 재서술 또는 가드 순서 재검토(F3), (d) **Stage V→C
   분산 변환 등록**(F1 — 미등록 변환 하나가 n을 5/10/12/20/UNAFFORDABLE로 움직인다),
   (e) 사다리 상수를 **값 수준에서** load-bearing하게(F6 — `perturb()` 튜플 분기가 rev1과
   한 글자도 안 다름), (f) `BUDGET_MAX_PAIRS` 재분류(DESIGN이 아니라 UNAFFORDABLE의 유일
   결정자) 및 **사실상 20**임을 반영(T975에 21–24 없음).
2. **동시성 correctness** — 부하 하에서 arm 간 출력이 일치하는가. 6회차가 *"한계 표기로 끝날
   문제인가"* 를 물었고 세션은 한계로 남겼다. **미해결.**
3. **CP-0 본체** — 死因 **G2·G3·G7·G8** 미해소(G4는 rev4에서 수리됨·미감사). ★**G8이 정본
   장부에서 여전히 누락**(`SWEEP_FINDINGS §1`) — doc-steward 소관.
4. **`presubmit.py`가 read-only가 아니다** — 실행이 타 트랙 인증서를 재작성한다(6회차 L16
   계열). 도구 수리 필요. 그때까지 이 트랙은 그 도구를 **돌리지 않는다**.
5. **G1-b 채점 런 미제출** — `PREREG_G1B_PLATEAU` 등록 완료, 새 seed `{23,29,31}`로
   ≈1.2 GPU-hr. `cp0_g1_rule.py`의 `band_vs_sd`는 여전히 **비어 있다**.

---

## 7. 미완·주의

- ★★**원래 질문의 정책 판정은 여전히 0건.** 이 세션 5.78 GPU-hr은 전부 **계측·축 검증·
  규칙층**이다. 새 성능 판정 0건.
- ★**규칙층 6연속 `NO-GO`**이고 6회차 死因의 **88%가 그림자**다. 다음 판본을 쓰기 전에
  **이 세션이 "고쳤다"고 적은 항목 자체를 재검증 대상**으로 둘 것(2026-08-26 교훈
  *"수리는 국소, 주장은 전역"*의 최신 재발).
- **미커밋 78 파일.** 이 핸드오프 커밋으로 저장한다.
- **방치된 job 없음** — 18 job 전부 완료.
- **다른 세션이 동시 작업 중**(nemotron_zt·prefill_knee·TC1·M4R). 이 세션은 그쪽 파일을
  건드리지 않았고, 유일한 예외는 §5의 **복구**다.
- **claims-auditor에 안 건 것**: `g1b_plateau_predicates.py`(step 1 산출물) · W1 결과 해석 ·
  correctness gate 설계 · 모델 부팅 스모크. 전부 **미감사**.
- **운영 함정**(재확인): 이 환경에서 `transformers` **import만 228초**(Lustre 콜드 캐시).
  CPU 전처리 스크립트의 타임아웃을 넉넉히 잡을 것. 첫 GPU 부팅도 같은 이유로 느리므로
  하네스 대기창은 **20분** + 채점 안 하는 워밍업 부팅을 둘 것.
