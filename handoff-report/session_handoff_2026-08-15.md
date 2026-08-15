# 세션 핸드오프 — 2026-08-15

> ★**날짜 주의**: 이 세션은 **2026-08-14에 시작해 08-15로 넘어갔다.** 아티팩트는 생성일
> 기준으로 스탬프돼 있다(E-3·Phase 0 4건·E-1a·정본 rev25–28 = **08-14**, E-1 rev2·rev3·
> kernel_mech 설계 = **08-15**). 직전 핸드오프는 `session_handoff_2026-08-13.md`(2026-08-11
> 세션 기록).

---

## 1. 이번 세션 요약

**GPU 총 지출 14초**(job 882374). 그 외 전부 GPU 0.

두 축으로 돌았다. **(A) 정본 정리·정정** — 2026-08-13 핸드오프가 남긴 등재 후보와 결함을
처리하다 **정본 결함 3건**(매니페스트 범위·ε 표기·하드웨어 오식별)을 찾아 정정했고,
**E-3 프로브로 실측 1건**을 얻었다. **(B) 실험 설계** — E-1 계열을 3회(rev1·rev2·rev3),
커널 층 설계를 1회, 총 **4회 설계해 4회 모두 감사에서 차단**됐다. Phase 0 사전등록 3건도
같은 세션에 차단됐다.

**누적 차단 GPU 지출 ≈ 20 GPU-hr 이상.** 차단 사유는 예외 없이 **실행 전에 GPU 0으로 찾을 수
있는 것**이었다.

정본 **rev24 → rev28**, 방법론 게이트 **#33·#34 신설**, **#26 개정**, **#9·#25 본문 복구**.
커밋 **6건**(`2da8e77` … `7efda5d`).

---

## 2. 결정·측정

### 2.1 ★E-3 — realized SM count 실측 (job 882374, GPU 14초) — **이 세션 유일한 측정**

`results/smsplit_realized/`. 사전등록 → 실행 → addendum 2건.

`create_greenctx_stream_by_value`가 반환하는 realized SM 수(래퍼가 버리던 `res[2]/res[3]`)를
직접 읽었다. **판정 `REQUEST_EQUALS_DRIVER_REPORTED_PARTITION`**:
`(74,34)`·`(54,54)`·C2 스윕(92/16·84/24·64/44·54/54·16/92) **7/7 정확 일치**,
`realized_sum = 108` 전 대상(**미할당 SM 0**), 로그인 노드(glogin01)·컴퓨트 노드(gpu43)
레코드 **완전 일치**.

- **허용**: "드라이버가 보고하는 파티션은 이 격자에서 요청값과 일치, 반올림 없음(드라이버
  자기보고 층)."
- **금지**(정본 등재됨): "실현 파티션을 **측정**했다" / "74 SM에서 **실행**됐다" —
  CONSENSUS §1-1 Gate 1 인용 금지 **불변**. `%smid`의 id-집합 질문 전진 0. Gate 2 전진 0.
- **부수**: `%smid` R0의 payoff 항목 1(`log(108/34)` 분모 정정)은 **정정 대상 없음**이 확인됐다.

### 2.2 정본 정정 3건

| # | 결함 | 정정 |
|---|---|---|
| 1 | **매니페스트는 15파일만 해싱** — `pdmux_context.py`·`sgl_kernel/spatial.py`·`src/patches/` 2개·`dev_tree_edits.md` 3/4/5/7이 밖 | 게이트 **#33 신설**, 재현성 주장 스코프 축소 |
| 2 | C2의 "구간 평균 **ε≈0.48–0.56**"이 `log(92/16)` 변환과 불일치 | **[0.491, 0.611]**로 정정. 옛 값은 **인용 금지된 SM16/SM108 열**([0.488,0.568])에서 흘러든 것. E-1a 실측 arm별 값 [0.492,0.610]이 독립 확증 |
| 3 | ★**하드웨어 오식별** — 캠페인은 gpu36/37/38/40(**SXM4**)에서 돌았는데 진단이 **로그인 노드의 PCIe**(1935 GB/s)를 사양으로 씀. `"A100 80GB PCIe"` 문자열은 **분석 문서 2개에만** 있고 job 아티팩트엔 0건 | 사양 **2039 GB/s**, achieved BW 47.8–60.8% → **45.4–57.7%**, ridge 161 → **153**. **기존 결론 불변·방향 강화**. 게이트 #32의 하드웨어 층 재발 |

⚠️ 3은 E-3의 노드 통제(두 SKU를 나란히 관측)가 없었으면 안 드러났다.

### 2.3 트래픽 진단 §3.4의 B 라벨 오류

Ha8 행이 `B=9` 라벨 아래 **B=12 산술**을 담고 있었다(`5.734 = 12×0.4779` 자릿수 확인).
⇒ B=9의 Ha8 weight share는 70.1%가 아니라 **75.81%**. 헤드라인 "68–94%"는 M8이 하한이라 불변,
§6.1의 924 GB/s는 이미 올바른 B=9 트래픽을 써서 rev26의 BW 범위 무영향.

### 2.4 Phase 0 사전등록 3건 — **전부 감사 차단, 제출 0건**

| 사전등록 | 판정 | 치명 결함 |
|---|---|---|
| LTSM P1 (`results/ltsm_probe/`) | **NO-GO**(9조건) | `PDMUX_FIXED_DECODE_SM_FILE` 미export ⇒ **SM 축 자체가 없음**. 임계가 3문서 3값. 허용성 손잡이 3개 전부 통과 쉬운 쪽으로 이동·미공시 |
| E1-b/c (`results/p1_gates/gate2/PREREG_G2S_E1B_E1C_2026-08-14.md`) | **NO-GO**(8조건) | P2가 `M<36`서 `frac` 미조회 ⇒ **REFUTED를 삼킴**(875293 실측: `(54,54)` 70행 중 58행이 `decode_bs<36`). 결정량 5개 중 4개 발화 불가 |
| `%smid` R0 (`results/smid_census/`) | CONDITIONAL-GO(5조건) | 런타임 PTX 검사가 **죽은 코드**(`JITFunction.cache` 부재 → 항상 except) + 스코어러가 `None`을 **fail-open** 통과. 자기검사는 `None`을 시험 안 함 |

### 2.5 E-1a 설계 사전분석 (GPU 0) + Tier 2 감사

`results/bsweep_regime/e1a_*`. 산출: `k` 식별 실패(|k|∈[0.051,0.082], 부호 미확정) ·
같은 데이터가 `B_lo` 선택에 따라 3판정 · 실측 `sd_rep(S)=0.0132` · **`Δw`가 `log B`와 r≥0.978
공선(비식별)**.

**claims-auditor Tier 2 판정 = 본문 승격 0건**: T2-3·T2-4 **기각**(사실 오류 / 항등식이며
정본 §1-25 중복), T2-2 **보류**, T2-1만 **조건부**(CONSENSUS §3 **항목 50**, rev28).

★ **아티팩트 결함**: `VERDICT.findings` F1–F8과 `T6_PC1_anchor_check`를 **산출하는 스크립트가
저장소에 없다**(재실행 diff: 코드 산출 20/20 키 바이트 동일, 이 2개만 코드 밖).
`E1A_ARTIFACT_ERRATA_2026-08-14.md`에 등재.

### 2.6 E-1 rev1 → rev2 → rev3 — **3회 연속 NO-GO**

| rev | 死因 |
|---|---|
| rev1 | **비식별**(`corr(Δw, log B)=0.998`) · T8에 SM 순회 훅 부재 · `R≈2.1` 유도 입력 3개 오류 · 문턱 0.15가 T8을 구성상 오분류 |
| rev2 | pin 하한 유도가 **사실과 다름**(최악은 0.317이 아니라 **0.217**, 자기 arm이 문턱 미달) · **부팅 분산 무시**로 등가 분기 원리적 발화 불가 · "모형 없는 대조"가 거짓이고 rev1의 선형성 공시를 삭제 · 장-L B 도달 반증 |
| rev3 | ★**정량 골격 전체가 오염된 앵커**(§2.7) · **목표 B=9가 구조적으로 도달 불가**(§2.8) · pin 문턱을 무이빨 값으로 교체 · 비용 2× 과소 |

**κ=2는 감사가 커널 코드로 닫았다**(`mamba_ssm.py:179` load / `:297` store — 스텝당 read+write).
부수 확인: **cudagraph padding은 traffic-neutral**(패딩이 트래픽을 실었다면 matched 쌍 Δw 차가
0.004가 아니라 3.78 pt였다).

### 2.7 ★★ 신규 발견 — job 865533의 keepalive 오염 (**정본 미등재**)

⚠️ **정정(doc-steward 재확인)**: 이것은 **완전한 신규 발견이 아니다.** CONSENSUS §3 **항목 28**
(2026-08-03)이 이미 *"`n_indep=1`, 865493↔865533은 keepalive 설정이 달라 복제가 아니다"*를
등재해 뒀다. **이번에 새로 밝혀진 것은 그 교락의 구체적 기전(1793 > 1792 토큰 초과)과 규모**다.
따라서 정본에는 **새 항목이 아니라 항목 28을 구체화하는 addendum**으로 들어갔다.
(세션 중 메인 세션이 "신규 발견"으로 서술한 것은 과장이었다.)

job 865533의 서버 로그에 컨텍스트 초과 거부가 **셀당 23,662 / 23,689 / 23,705건**
(`s8_keepalive_prompt_224.txt` = 1793 토큰 vs 컨텍스트 상한 1792). ⇒ **keepalive가 사실상
100% 거부**됐고, 그 job은 **prefill 동거가 죽은 런**이다(Ha8 d44 동거율 0.349 vs 865493의 0.662).

E-1a의 `S(B)` 계열은 `S(9)`를 **865533 단독**, `S(16)`을 **865493 단독**에서 뽑는다.
⇒ **`0.0610`은 "B 효과"가 아니라 "동거 있음/없음"과 완전 교락**이다. 그리고 동거는
green-context 파티션이 **실현되는 조건 그 자체**다.

**파급**: CONSENSUS §3 **항목 50**의 caveat (ii)가 "Ha8에서 B축이 job축과 교락"이라고만
적혀 있는데, 실제 교락의 정체는 **keepalive 사망**이라는 훨씬 구체적·심각한 형태다.
⚠️ 별건: `NOTES_D54_ANCHOR_2026-08-03.md` Finding 1이 이 현상을 "engine-tree churn"으로
추정하고 "865493이 쓴 파일과 byte-identical"이라 적었으나, **로그 타임스탬프 + 파일 mtime
순서가 865493이 그 파일을 쓸 수 없었음을 보인다**(감사 발견, 이관 대상).

### 2.8 ★★ 신규 발견 — B 도달성의 구조적 폐쇄 (**정본 미등재**)

ctx4096 텔레메트리 전수 재집계(감사):

| arm | d44 maxB | d92 maxB | d92에서 B≥9 |
|---|---|---|---|
| T8 | 8 | **4**(p10=p90=4) | 0.0% |
| M8 | 8 | **4** | 0.0% |
| Hs8 | 6 | **4** | 0.0% |
| Ha8 | 16 | 12 | 3.0% |

기전은 메모리가 아니다(T8 KV 풀 910,624 tok). **prefill이 16 SM에 고정 +
`--chunked-prefill-size -1`이라 prefill 서비스율 λ가 상한**이고, Little's law로
`B_decode = λ·T_decode` ⇒ **동시성을 올려도 B가 오르지 않는다**(실증: 같은 `CONC=16`에서
Ha8 median B가 15–16 vs 1). 남은 레버(출력 토큰↑)는 **Zamba2 ctx 4096 상한**이 막는다.

⇒ **양 arm 공통 B=9 @ SM92는 이 기판에서 구조적으로 도달 불가.** 앞으로의 모든 B축 설계에
적용되는 제약이다.

### 2.9 커널 층 설계(ncu/nsys) — **감사 판정 "설계 재작성"**

`results/kernel_mech/DESIGN_KERNEL_MECH_2026-08-15.md`(미커밋). 도구 층 사실 4건이 확인됐고
그중 **2건이 설계를 뒤집는다**(전부 GPU 0으로 확인 가능했음):

1. ★`launch__waves_per_multiprocessor` 메트릭 설명문: **"When using green contexts, this
   metric is scaled with the number of SMs used by the green context."** ⇒ 설계 §3이
   "★★가장 중요한 함정"이라 부른 **분모 오염은 이 툴체인에 존재하지 않는다**(ncu 2024.3+,
   드라이버 560+; 설치본 2025.3.1.0 / 드라이버 580.105.08). 없는 오염을 피해 수제 카운터
   층으로 내려갔고 거기서 **`wave_eff ≡ 1` 항등식**을 만들었다(게이트 #9 여덟 번째 재발).
2. ★`nsys --cuda-graph-trace` 기본값 `graph`: **"node activities will not be collected."**
   ⇒ 기본 설정이면 Stage 0 항목 1이 FAIL하고 설계의 정지 규칙은 그때 **"전 설계 폐기"**다.
   **플래그 하나 때문에 하네스 실패가 도구 불가 판정으로 라벨링되는 부비트랩**(게이트 #21).
3. ★**`--exclusive`는 불필요하다.** ncu 직렬화 락은 per-device이고 GPU는 `--gres=gpu:1`로
   이미 우리 것. 카운터 게이트인 `hwperf`는 **전 GPU 노드의 기본 feature**
   (`gpu[36-43] ... A100-80GB_8,hwperf`). 선행 ncu 로그에 `ERR_NVGPUCTRPERM` **0건**.
   ⚠️ 방향이 뒤집혀 있다 — 노드 잡음에 민감한 쪽은 **Stage A(nsys `gap_frac`)**다.
4. ★**선행 ncu 시도가 이미 있다** — `ncu_runner.py` + 아카이브 **8 job**(717984 등),
   실패 `error code 9` **1,986건** + 메트릭 정규식 실패 1,920건. 결정적으로 그 시도들은
   **full GPU에서 돌았고 green context 하에서 잰 적이 없다**(`_ncu_target.py:68-76`:
   *"ncu profiling always runs at full GPU"*, wave는 **해석적** 계산).
   ⇒ 정본의 "커널 단위 측정 0건"은 **운영점·green ctx 한정으로만 참**이다.

**감사가 지적한 최대 결함**: 설계의 유일한 1차 결정량 `gap_frac`이 **정본 §1-26이 이미
은퇴시킨 축의 커널 층 판본**이다(*"decode가 D SM에서 돌았다" ⟺ "prefill 동시 in-flight"*,
"어떤 통계도 decode-SM 탄력도와 prefill 간섭을 분리 못 한다"). 설계는 이를 **언급조차 하지
않았고**, §0의 "prefill 동거 불필요"는 거짓이다(동거는 decode가 D SM에 있기 위한 필요조건).
탈출구 **`PDMUX_STICKY_PARTITION`**(realized 0.084→1.000, correctness gate 통과)이 저장소에
있는데 설계가 쓰지 않았다.

**살아남는 payoff**: `dram__bytes.sum`의 **약형**(스텝당 DRAM 바이트의 독립 실측 — 트래픽
진단의 "68–94% weight-sweep" 추정을 처음 검증). 단 캐시 flush 때문에 **상한**이고,
"대역폭 **포화** 여부"(속도)는 직렬화 + base-clock 클램프로 **얻지 못한다**.

### 2.10 운영 사실

- **`/home01` inode 고갈로 쓰기 전면 차단**(98,978/100,000)됐다가 해소.
  원인은 `~/.cache/vllm/torch_compile_cache` **10,063 inode**(vLLM 잔여물, 이 프로젝트는
  SGLang). 삭제 후 **88,916**. 모델 가중치 11 GB는 보존. **재발 시 같은 곳을 먼저 보라.**
- **로그인 노드 glogin01에 A100 80GB PCIe가 있다** — 드라이버 층만 건드리는 프로브는 SLURM
  없이 실행 가능. 단 컴퓨트 노드는 **SXM4**라 하드웨어 의존 판정은 컴퓨트에서 재확인.
- **push 프레이밍 정정**: `origin/main` reflog **165 엔트리 중 163개가 `update by push`**이고
  이력이 **2026-05-07**까지 간다(2026-07-27~08-14 창에서 19일 중 10일, 23회). ⇒ **push는
  이 저장소의 일상 워크플로이며 이상 동작이 아니다.** 2026-08-13 핸드오프 §5의 "미승인 push
  2건, 출처 불명"은 같은 오독일 가능성이 높다. **남는 실질 사안은 원격 URL의 PAT 평문 노출 하나.**

---

## 3. 코드·문서 변경

### 커밋 6건 (전부 로컬, `main` 직커밋)

| sha | 내용 |
|---|---|
| `2da8e77` | 2026-08-11 세션 핸드오프 추가 |
| `0c7fa3e` | 정본 — 게이트 #26/#33 동기화 + #9/#25 본문 복구 + 하드웨어 basis 정정(SXM4) + E-3 등재 |
| `590da47` | Phase 0 사전등록 3건 — 전부 감사 차단, 제출 0건, GPU 0 |
| `97be982` | E-3 프로브 + 결과(job 882374) |
| `de501fd` | E-1 rev1 사전등록(하네스 미작성) |
| `0f16498` | E-1a 설계 사전분석 아티팩트 — 결론 블록 미산출 결함 명시 |
| `7efda5d` | CONSENSUS rev27+rev28 — 정본 결함 2건 정정, E-1a Tier 2 감사 등재 |

⚠️ 커밋 트레일러가 `de501fd`까지 **`Claude Opus 4.8`로 잘못 들어갔다**(실제 Opus 5).
`7efda5d`부터 바로잡음. 과거 커밋은 rebase 비용 때문에 그대로 뒀다.

### 미커밋 (3건)

- `workspace/engine-port/results/bsweep_regime/PREREG_E1_REV2_2026-08-15.md` (NO-GO)
- `workspace/engine-port/results/bsweep_regime/PREREG_E1_REV3_2026-08-15.md` (NO-GO)
- `workspace/engine-port/results/kernel_mech/` (설계 + 감사 판정 "재작성")

### 정본 갱신

`reports/CONSENSUS.md` **rev24 → rev28**(§3 항목 47·48·49·50 신설, 항목 46·18 追記,
§1-25 追記) · `PROJECT_STATUS.md`(게이트 **#33·#34 신설**, #26 개정, #9·#25 본문 복구,
"다음 실험 gate" 항목 11 = 미제출 사전등록 레지스트리, engine-porter 이관 3건 추가) ·
`workspace/engine-port/RESUME.md` · `reports/paper/` 3문서 ·
`results/s8_scaleup/TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md`(§11·§12 addendum) ·
`FINDINGS_8B_2026-07-28.md`(§7 C-5).

### 메모리

`MEMORY.md`(34KB → 8.1KB 압축, 상세는 topic 파일) · `deconfound-measurement-lessons.md`
(**항목 34 신설** = 2단 감사 규율, 항목 31 追記) · `slo-aware-scheduling-track.md` ·
`scale-8b-sm-sensitivity.md`.

⚠️ `deconfound-measurement-lessons.md`가 세션 중 **0바이트로 훼손**됐다가 세션 transcript에서
재구성 복원됐다(항목 1–33 온전 확인, 스크래치 사본 보존).
**바이트 동일성은 검증 불가** — 재구성이지 백업 복원이 아니다.

---

## 4. 열린 항목 / 다음 세션 시작점

### 즉시 (GPU 0) — 정본 미등재 발견 3건

1. ★**keepalive 오염**(§2.7) — CONSENSUS §3 항목 50 caveat 강화, E-1a `S(B)` 계열 인용 조건
   재검토, `NOTES_D54_ANCHOR_2026-08-03.md` Finding 1 정정.
2. ★**B 도달성 구조적 폐쇄**(§2.8) — 실험 게이트로 등재(앞으로의 B축 설계 전부에 적용).
3. ★**ncu/nsys 도구 사실 4건**(§2.9) — 특히 "선행 시도 8 job 실패"와 "exclusive 불필요".
   정본의 "커널 단위 측정 0건"에 **운영점·green-ctx 스코프 주석**을 달아야 한다.

### 방법론 게이트 신설 후보

- **"개정판에서 손잡이 값을 유지한 채 유도 서사만 바꾸지 마라"** — rev2 δ=0.0610/3 →
  rev3 δ=0.020(반올림)이 그 사례. 4회 설계 중 3회 재발.
- **"타당성은 도구 문서·메트릭 DB로 확인한 뒤 설계하라"** — kernel_mech §3이 존재하지 않는
  오염을 피하려다 항등식을 만든 사례. 확인 비용은 로그인 노드 1줄이었다.

### 실험 트랙

- **kernel_mech 재작성**: 감사가 "GPU 0에서 지금 닫히는 것 6건 + Stage 0에 반드시 넣을 것
  6건"을 명시했다(`results/kernel_mech/` 감사 판정 참조). 핵심은 (a) §3 폐기하고
  `launch__waves_per_multiprocessor`를 primary로 (b) 후보 집합에 (v) 메모리 동거 간섭 ·
  (vi) 클럭/DVFS · (vii) 커널 고정 오버헤드 추가 (c) **sticky ON/OFF 2×2 편입**으로 §1-26
  탈출 (d) exclusive 전제 철회 (e) 비용 재추정(≈7–12 GPU-hr).
- **E-1 계열은 §2.8로 현 기판에서 닫혔다.** rev4를 쓰려면 B를 통제할 다른 수단
  (prefill SM을 푸는 것 등)이 먼저 필요하다.
- **"상금 크기" 논증**(GPU 0, 미착수): 탄력도(C2 확정) + 도달 가능 영역(§2.8) + §1-20
  coupling tax를 조립해 "재배분이 낼 수 있는 최대 이득이 작다"를 기존 증거로 세울 수 있는지.
  논문 뼈대에 직결되며 이번 세션에서 **논의만 하고 착수 안 함.**

### ★후속 감사 1건 (doc-steward 발견, 범위 밖이라 미해결 — result-analyst/claims-auditor 소관)

**C2의 Ha8 헤드라인이 keepalive-사망 job 단독일 수 있다.** `FINDINGS_8B_2026-07-28.md:92`의
`Ha8(batch=9) 89.80/31.49 = 2.85×`가 **양 다리 모두 job 865533**에서 나온다(doc-steward 추적).
그 job은 §2.7의 keepalive 사망 런이고 `CO_RESIDENT_frac ≈ 0.35`가 캠페인 전체에 걸린다.

- **within-job 비율**이므로 항목 50의 cross-job 교락과 **같은 방식으로 작동하지는 않는다.**
- 다만 Ha8 arm의 **C2 CONFIRMED(scoped) 기여분이 다른 파티션 실현 체제에서 측정**됐다는 뜻이다.
- ⚠️ 부수: 같은 문서가 *"대표값은 표본이 큰 batch(12–16) 슬라이스에서 뽑을 것"*이라 경고하는데
  **Ha8만 batch=9**를 쓴다(M8·Hs8·T8은 batch=12). 문서 내부 긴장이며 미해소.
- **M8/Hs8/T8 헤드라인 셀의 job 구성은 미확인.** 물질성 평가 안 됐고 **C2 등급은 변경 없음.**

⇒ 이 트랙의 flagship 확정 주장이므로 **다음 세션에서 우선 라우팅**할 것.

### 미해결 (변동 없음)

Gate 2 본 질문("PD 분리 자체" 귀속, n≈64 필요) · Claim D(dual-worker, 서빙 측정 0건) ·
비-SM-split lever(구현 전무) · long-ctx 충돌 regime · 긴장 A(HE2 vs C2) ·
engine-porter 이관 **7건**.

---

## 5. 미완·주의

1. ★**설계 4연속 차단.** rev1·rev2·rev3·kernel_mech가 전부 감사에서 막혔고, 死因 대조표에서
   **5개 중 4개가 재발**했다(타당성 확인 전 확정 · 편의적 정의/값 · 가정을 한 층 아래로 이동 ·
   비용 2–4× 과소 · payoff 과장). **다음 세션은 설계를 쓰기 전에 이 목록을 먼저 읽을 것.**
2. **정본 미등재 3건**(§4 즉시 항목) — 커밋 메시지와 `results/` 파일에만 있다.
3. `%smid` R0는 **CONDITIONAL-GO**(조건 5건) 상태로 남아 있다. E-3가 그 payoff 항목 1을
   무의미하게 만들었으므로 **§0.1 재작성이 선행**돼야 한다.
4. **engine-porter 이관 7건 미해결**: `SGLANG_ZAMBA_TIMING=0` truthy · `g2s_analyze.py:84-89`
   `PREMISE_LABEL` · `compute_coverage` 내부 gap · `g2s_*` rep 경계 마커 ·
   `spatial.py`가 realized SM 수 폐기 · `smid_l0_census.py:410` 죽은 코드 ·
   `g2s_e1b_premise.py`의 `frac` 미조회.
5. `gate1_run.sbatch:61-66`은 **여전히 stale**(지금 돌리면 abort).
6. **원격 URL PAT 평문 노출** — push 자체는 일상 워크플로로 판명됐으나 이건 별개로 유효.
7. **방치된 job 0건**(squeue 비어 있음). 작업 트리에 미커밋 3건.
8. ⚠️ 메모리 `deconfound-measurement-lessons.md`는 **재구성 복원본**이다(§3 주의).

---

## 6. 다음 세션 시작점 (한 줄)

*`catch-up` 후 → **정본 미등재 3건**(keepalive 오염 · B 도달성 폐쇄 · ncu 도구 사실)을
doc-steward로 등재 → **kernel_mech 재작성**(감사의 12개 조건, 특히 sticky ON/OFF 2×2로
§1-26 탈출) → Stage 0. ★**설계를 쓰기 전에 §5-1의 死因 대조표를 먼저 읽을 것.***
