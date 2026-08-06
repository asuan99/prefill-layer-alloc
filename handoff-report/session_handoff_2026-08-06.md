# 세션 핸드오프 — 2026-08-06

세션 범위: **2026-08-04 → 2026-08-06** (정본 항목은 대부분 **2026-08-05** 날짜로 기록됨).
직전 핸드오프: [`session_handoff_2026-08-03.md`](session_handoff_2026-08-03.md).

---

## 1. 이번 세션 요약

논의는 "프로젝트 서사 정리"로 시작해 **positive/negative 근거 대조**로 옮겨갔고, 그 과정에서
메인 세션이 세운 분석이 **다섯 번 연속 감사에 깨지며** 실제 측정으로 이어졌다. GPU 캠페인
**5건**을 돌렸고(873617 · 873783 · 873921 · 873944 · 873945), 이전 세션이 제출한 S2(873015)를
감사했다. 결과로 **정본 최상위 문장 두 개가 모두 움직였다**: **§0(C2 ↔ 872077 2.6× 불일치)이
CONFIRMED(scoped)로 종결**됐고, **§1-1("PD 분리 자체는 항상 이득")에서 "항상"과 "자체"가 둘 다
철회**됐다. `CONSENSUS.md`는 **rev8 → rev12**로 다섯 번 개정됐다. 관통하는 패턴은 하나다 —
**이번 회차에 판정을 뒤집은 것은 전부 "지표가 무엇을 재고 있었나"였다**(라벨이 실행을 식별
못 함 / 버킷이 시간을 잘못 귀속 / 지표가 도착률에 절단 / 술어가 위반 0건이라 처리량의 다른
이름). 방법론 게이트가 **6개** 늘었다.

---

## 2. 결정·측정

### 2.1 §0 종결 — 최상위 열린 항목 [CONFIRMED (scoped)]

**S2(job 873015, 이전 세션 제출)를 독립 재현 → 감사**했다. 원자료 `results/s2_sticky/`,
재현서 `S2_REPLICATION_2026-08-05.md`, 정본 `CONSENSUS.md` §1-31.

- sticky 발화: `E1_DECODE_REALIZED` **0.9990±0.0017**(d16) / **0.9994±0.0008**(d54), 16/16 블록.
  OFF(872077) 0.0380 / 0.0925 ⇒ **26× / 11×**.
- ★ **§0 핵심 수치 = 28.92ms — C2 쪽**(C2 28.79/31.05 ⇒ 비 1.005/0.929; 872077 11.09 ⇒ **2.61**).
  **`split_frac` 라벨을 전혀 쓰지 않는 raw 클라이언트 ITL 317,342개의 median**으로 재현.
- 3지선다 판정: **(i) 하드웨어 형태 REFUTED / 라벨 형태 CONFIRMED**, **(ii) DISFAVOURED**,
  **(iii) CONFIRMED**(+기전 확보).
- "A/B가 아니라 before/after"라는 최대 약점을 두 내부 상한이 닫았다 — **d54 companion이
  전역 교락을 ≤1.097×로 상한**(재현자도 기존 분석도 안 쓴 논증), 그리고 **~29–34ms 수준이
  872077 자신 안에 gpu36·pre-patch로 존재**(셀을 따라감: d16 34.0 / d24 22.0 / d44 15.0 / d54 14.2).
- 감사자가 **(iii)의 기전을 계측에서 독립 도출**: `runtime_snapshot`이 개수-서브샘플링이고
  decode-busy 조건부 케이던스가 **정확히 16 decode step** ⇒ ITL 구간 하나가 스냅샷 1/16개를
  걸침 ⇒ **기대 순도 ≈6%**, S0-R의 mode 분해 6.6–9%와 다른 경로에서 일치.
- ★ **재현자의 "slow-mass 2.03× 잔차"는 REFUTED** — count-share vs time-share **단위 불일치**,
  매칭 단위 비 **0.97**.

**Scope**: behavioural · selector-level. `decode_sms`는 `arbiter.sm_counts[stream_index]`의
재진술이며 **하드웨어 부여 층은 미프로브(S3 미실행)**.
**E1은 열리지 않는다** — 4가지 독립 사유(§1-31) 전부 불변.

### 2.2 (α) 고-D 대조 — job 873921 [실행 완료, §0 등급 불변]

`results/s2_sticky/s2a_*_873921*`. sticky ON + decode≈92 SM, T8, 4 block.

- 게이트 PASS(`E1_DECODE_REALIZED` 1.000/0.998/0.998/1.000), `n_err=0`.
- primary **11.26ms**(telemetry, t95 [11.049,11.467]) / **11.32ms**(raw-itls, t95 [11.066,11.568]).
- 사전등록 규칙 적용 시 **INDETERMINATE**(12–13 밴드에 5.7 block-sd 미달, 28–30 붕괴 밴드에서
  ~106 block-sd 이격). **붕괴 분기 REFUTED** ⇒ §0 종결 유지, **등급 CONFIRMED(scoped) 불변**.
- ★ **감사자가 자기 밴드를 REFUTED시켰다** — [12,13]은 **자기가 §1-31에서 금지한 C2→sticky
  이식**으로 도출됐다. α 자신의 step 회귀(`t = 10.883 + 0.0991·batch + 0.00034·ctx`)로 C2
  동작점을 예측하면 12.44–12.80(관측 12.875, 잔차 0.9–3.8%) ⇒ **12.4% 격차는 통제되지 않은
  워크로드 격차**(ctx 중앙 1291 vs 287, batch 11.31 vs 4.51). 더 가까운 앵커(872077의 10.98,
  블록별 byte-matched)를 썼으면 예측 11.0–11.4로 맞았다.
- **α의 실질 기여 = 순환성 제거.** 종전 근거가 분쟁 중인 필드의 재진술이었는데 **결과-축
  앵커**로 옮겼다: ON d92/OFF d16 **1.0293** [1.0210,1.0376], ON d16/OFF d16 **2.6320**,
  ON d54/ON d92 **1.0658**.
- ★ **(iii)의 가장 깨끗한 진술**: 872077 안에서 SPLIT(11.08) vs UNSPLIT(10.98)이 **1% 차이**인데
  같은 기판의 진짜 D16 vs D92는 **163%** ⇒ **라벨이 사실상 아무것도 분리하지 않았다.**
- **인용 금지**: "11.26≈11.09 정밀 일치"(교차-job 오프셋이 3배 큼), "α가 하드웨어 층을
  닫았다"(D92–D108 판별력 0.6%).

### 2.3 §1-1 재작성 — jobs 873944/873945 [운영점 이득 확인, 기전 귀속 미확립]

`results/p1_opint/`(PREREG.md 제출 전 작성, `P1_OPINT_RESULT_2026-08-05.md`[AUDITED]).
Zamba2-2.7B + Granite-4.0-h-micro × {plain, agnostic} × rate{2,3,4,6} × **n=5 paired**, **cudagraph-ON**.

- **정본 술어**(게이트 #4: TTFT≤3s ∧ 요청 내부 token-ITL p95≤60ms)로 **두 모델 전부·전 셀에서
  agnostic 승**(rep 부호 5/5, paired CI 0 배제).
- **인용 가능한 정량치 = 두 arm 모두 정상상태인 셀뿐**: Zamba2 r2 **+40.5%** [+0.431,+0.620],
  r3 **+185.8%**; Granite r3 **+11.7%**, r4 **+27.0%**. 임계 사다리 40–300ms 부호 불변,
  큐 성장 없음, rep 5/5 — **감사자의 다섯 공격을 견뎠다.**
- ★ **분석자 헤드라인 둘 다 채택 불가**: "Zamba2 확인(사전등록 지표)" NOT-YET-SUPPORTED
  (유의 셀 r4·r6이 전부 절벽 위), "Granite 전면 강등" **REFUTED**(r3/r4에서 사전등록 술어의
  **위반 요청 0건 ⇒ goodput ≡ throughput**, 판별력 0 — **게이트 #6의 새 사례**).
- ★ **"항상 이득" 철회 — 비용이 실재한다**: 중앙 token-ITL **5–45% 악화**(9.67→10.50,
  22.57→29.30ms), 저부하 TTFT p50 악화(0.164→0.230s), Granite raw 처리량 **−0.3~−2.6%**.
  **이득은 꼬리, 비용은 중앙.**
- ★★ **"자체" 철회 — 3-플래그 묶음 + fused 미조율**: `--enable-pdmux`가
  `--chunked-prefill-size -1`·`--disable-overlap-schedule`을 **assert로 강제**
  (`server_args.py:6125-6137`). 게다가 관측 死因(prefill 배치가 decode를 막음, 클러스터 지속
  166ms@r2→474ms@r4)의 **fused-측 레버 두 개**(`--chunked-prefill-size` 축소,
  `--enable-mixed-chunk`)가 **둘 다 기본값**이다. ⇒ 지지되는 최대치는
  **"PD-mux > 기본 설정 fused"**.
- **2026-08-04 반대 증거도 정정**: "cudagraph가 TPOT>60ms 벽을 제거한다"는 **중앙값 근거**였다.
  plain r4는 중앙 TPOT 48.5ms인데 요청의 **31%**가 mean-ITL을, **86%**가 정본 술어를 위반한다.
  **중앙값이 SLO 아래 ⇏ 요청 통과.**

### 2.4 P5(§1-5) 계측 수정과 재측정 — **NO-GO, 재설계로 이관**

- **계측 결함 3건 확인**(claims-auditor + X2′, 2026-08-04): 버킷 비대칭(`zamba2.py:163` attn은
  코어만 / `:259` mamba는 mixer 전체) · 누산기 러닝평균(`full`이 항상 첫 arm이라 편향이 모든
  비의 분모에) · **물리 불변량 위반**(`full`이 sm44보다 2.34× 느림).
- **X2′ 검증**: attn도 부풀려지나 **mamba의 1/20~1/2**(보고값 중앙값 attn 1.0009 vs mamba
  1.0215). ★ **비순환 확증** — 두 독립 job이 같은 셀에서 보고값 1.0454× 불일치, **steady
  1.0008× 일치** ⇒ **"1.32 vs 1.38 차이는 물리가 아니라 몇 번 돌았나"**. **강등 6건 무효 0건.**
- **계측 수정 완료**(engine-porter): 버킷 대칭화(`attn`/`mlp`/`other`/`mamba`, **임의 귀속 없음**),
  `_ZT_DIAG=attn_core`로 옛 정의 중첩 보존, **closure gate**(독립 이벤트 쌍 — timeline partition으로
  안 만들어 항등식 회피), emit마다 accumulator 리셋, shape 가드, emit 태그 `ZBLT2`(옛 grep이
  재정의된 양을 조용히 읽지 못하게).
- **GPU 게이트(job 873617) 전부 PASS**: 3a/3b/3c byte-identical(5 arm `gen_sha` 동일).
  closure 분포 min 0.9390/p50 0.9403/max 0.9435 ⇒ **임계 0.85 확정**(옛 계측 역산 ≈0.6997과의
  **간극**에서 선택, 끝점 선택 아님). ★ **결함 1의 크기 최초 측정: 옛 버킷이 attention 층
  시간의 33%만 잡았다(3.02×)**.
- **본 측정 배치 1(job 873783, rep1×4 ctx) — 20셀 OK, GATE C PASS 20/20, GATE N FAIL.**
  bs=32 부분집합에서도 생존(full 2.780 / sm44 1.971 / sm24 1.276 / sm16·sm8 1.003).
  ★ **여집합 음성대조가 정체를 특정**: `per_mlp` 깨끗, `per_other`가 **mamba와 같은 순서로 실패**
  ⇒ **mamba 특이 문제가 아니라 버킷 간 귀속 문제**. 기전 = **host-paced 셀**(`fwd_ms`가 device
  work 2–8× 다른 arm에서 ~44–46ms 바닥에 고정).
- ⇒ **G6 = NO-GO**(rep2/3은 계통 편향에 n을 더할 뿐이고 오염 영역이 knee 판독에 필요한 코너).
  **배치 2·3 미제출.**
- **도구 결함 3건 수정 완료**(D1 분석기 자기모순 = grep이 폐기 warm-up `rw`까지 매칭 /
  D2 `mamba_spread` 항등식 / D3 `bs`가 혼합 은폐) + **D4 GATE N을 `(ctx, 실현 bs)` 조건부로
  재정의**(비교쌍 없으면 `UNSCOREABLE`, 여집합 음성대조 상시) + D5 host-boundness 관측치만.
  **CPU 회귀 100 tests OK**(57+43), 원자료 불변(md5 검증), **임계 1.15/0.85가 테스트로 핀 고정**.

### 2.5 천장 절단 진단(③) — §1-13 각주 확정 + 정본 오류 2건

`results/slo_sched/CEILING_CENSORING_DIAG_2026-08-05.md`[AUDITED]. GPU 0.

- **각주 CONFIRMED — 단 근거가 통째로 교체됐다.** 분석자 헤드라인(임계 사다리 T=20/30/60)은
  **REFUTED**(T=60은 2400 요청 중 9 vs 9로 **검정력 0**, 상대 CI가 T=20의 −49.8%를 포함;
  cliff 위 단일점; ★ 그 9건의 7–8건이 **round0 warm-up**이고 자기 `middle60` slice에선 **0건**).
- **살아남는 근거**: `요청별 ITL p95의 p90` 53.36→46.54ms(**5.6 SD**), 중앙 12.183→13.912(3.2 SD),
  TTFT p50 73.56→79.61(3.3 SD).
- **기전 정정**: 절단은 **이중**이고 **정책 구분을 죽이는 쪽은 도착률이 아니라 SLO 여유**
  (pass ∈ [0.973,0.998], 런을 늘려도 남는다).
- **범위 제한**: LO 레버는 SLO 예산 단위로 **HI의 1/14**이고 **통계에 따라 부호가 뒤집힌다**
  ⇒ "어느 split이 LO에서 좋다"는 **functional 없이 정의 불가**. **HE0 불변.**
- ★ **정본 §1-13 오류 2건**: "LO spread 0.067"은 **arm마다 n이 다른 표**(d16/d24/slo n=1)이고
  **"spread < rep SD"라는 부등식으로 다시 써야** 한다. "HI spread 43%"는 **legacy mean-ITL
  스코어러**(`sharegpt_vary_bench.sbatch:94`) 값이고 정본 술어로는 **+716%**(결론 강화).
- **C2 정성 대조는 REFUTED**(§3-21의 3번째 재발).

---

## 3. 코드·문서 변경 (**전부 미커밋**)

### 정본 (doc-steward, 5회 개정)

| 파일 | 변경 |
|---|---|
| `reports/CONSENSUS.md` | **rev8**(Diff A/B 강등 6건 + §1-5 각주 + 천장절단 + P6 강등 + §3-20~22) · **rev9**(§0 종결 §1-31, §3-23) · **rev10**(§1-13 각주 교체 + §3-24·25) · **rev11**(α + §3-26) · **rev12**(§1-1 교체, §3-22 `**` 수정) |
| `PROJECT_STATUS.md` | "확정된 결과" 1번 헤드라인 교체 · §0 종결 소절 · `2026-08-05(α)` 소절 · **방법론 게이트 #11·#12·#13 신설** + #6 하위 5 · "다음 실험 gate" 항목 9·10 |
| `reports/research_arc.md` | S3 지수 정정(**a=1.916±0.021 / b=0.954±0.008**), WIDE "확증" 취소선, "6.240" 두 곳 인용 금지 각주, S-M.1 **#9b 행**(PF 캠페인), S-M.2 **E5 행**, §S-M.3 **입력 길이 분포 행** |
| `reports/paper/CLAIM_EVIDENCE_MATRIX.md` | Claim A에 C2 국소 탄력도 각주(**등급 불변**). P1 인용은 애초에 없음을 확인·명시 |
| `reports/policy_comparison.md` | 행 6(PF)에 인용 제한 + 상호참조 |
| `reports/longcontext_trace_plan.md` | Diff A 지수 정정 2곳 |
| `results/prefill_knee/diffA_vs_diffB_table.md` | 헤더 경고(**집계 단위=커널**, **no-cudagraph micro**) + L=2000 행 오염 표시 |

### 신규 문서

- `reports/layertype_dynamic_{POSITIVE,NEGATIVE,JUNCTION}_2026-08-04.md` — layer-type·dynamic
  축의 positive/negative/결합점 3종 (감사 반영해 개정됨, **종합 문서 — 정본 아님**)
- `results/s2_sticky/S2_REPLICATION_2026-08-05.md` · `results/r0c/P5_GATES_BATCH1_2026-08-05.md` ·
  `results/prefill_knee/AGGREGATE_COMPOSITION_2026-08-04.md`(REV.2) ·
  `results/slo_sched/CEILING_CENSORING_DIAG_2026-08-05.md`[AUDITED] ·
  `results/p1_opint/{PREREG.md, P1_OPINT_RESULT_2026-08-05.md}`[AUDITED]
- **사전등록 미영속화**: P5 재측정 재설계 사전등록 전문이 **대화에만 있다** →
  `results/r0c/PREREG_P5_REMEASURE_2026-08-05.md`로 저장 필요(§5 참조)

### 코드

| 파일 | 변경 |
|---|---|
| `src/models/zamba2.py` | 버킷 대칭화 + `_ZT_DIAG` + **closure gate** + accumulator emit-리셋 + shape 가드 + `ZBLT2` 태그 + try/finally 가드. dev tree 설치 sha `035a7d75…` |
| `results/r0c/analyze_decode_knee_vs_ctx_v2.py` | **재작성**(184→646행): D1–D5 전부, `(ctx,실현bs)` 조건부 GATE N, `UNSCOREABLE` 분기, 여집합 음성대조 상시 |
| `results/r0c/decode_knee_vs_ctx_v2.sbatch` · `zamba2_timing_smoke.sbatch` | 신규(무작위화·warm-up 폐기·전 블록 보존 / 5-arm 정확성 게이트) |
| `tests/test_zamba2_instrumentation.py`(17) · `tests/test_p5_gate_tools.py`(43) | 신규. **임계 1.15/0.85 핀 고정** |
| `benchmarks/pdmux_eval/analyze.py` | `load_bench_serving_rounds()`(duration **합산** 내장) · `unpaired_bootstrap_ci()` · `request_tpot_percentiles()` |
| `scripts/bootstrap/sync_engine_tree.sh` | `src/{configs,models}/zamba2.py` 설치 + manifest 등재 |
| `results/s2_sticky/s2_sticky_d92.sbatch` · `results/p1_opint/{p1op_run.sbatch,p1op_analyze.py}` | 신규 |

### 메모리

`deconfound-measurement-lessons` **항목 16**(버킷 비대칭 + 비리셋 누산기) · **항목 17**
("같은 노드 대조" 표를 raw `SLURM_NODELIST`로 검증하라) / `slo-aware-scheduling-track` ·
`scale-8b-sm-sensitivity` 2026-08-05 항목 / `MEMORY.md` 인덱스 갱신.

---

## 4. 열린 항목 / 다음 세션 시작점

**GPU 대기열 (권고 순서)**

| # | 실험 | 비용 | 스테이크 |
|---|---|---|---|
| **1** | **Gate 1 — telemetry 재현런** (agnostic 1 rep, `PDMUX_TELEMETRY_PATH`, rate{2,3}) | **0.2 GPU-hr** | §1-1의 **기전 문장 해금 여부**. 사전등록: decode-busy∧prefill in-flight의 realized split ≥90%가 `(74,34)`/`(54,54)`면 해금, 아니면 §1-22 auto-revert 지배 |
| **2** | **Gate 2 — 4-arm 분해** {plain, plain+aux, plain+chunk512, agnostic} × rate{2,3} × n=5 | 1.5 GPU-hr | ★ **논문 신규성 축**. `plain+chunk512`가 agnostic의 3% 이내면 §1-1은 "**여러 수단 중 하나**"로 재작성 |
| 3 | δ — 같은 바이너리 sticky-OFF arm | 7분 | §1-31의 **3% 이하 수치(1.0293/1.0658) 인용 전제** |
| 4 | β — OFF 1블록 `PDMUX_TRACE_FORCE_PREFILL=1` | 7분 | (iii)의 **유일한 양적 다리** 직접 검정 |
| 5 | α′ — sticky d92를 **C2 클라이언트로** | 10분 | 예측 12.4–12.9. 적중=격자 이전 금지 일부 해제 / 빗나감=**C2 앵커 영구 은퇴** |
| 6 | Gate 3 — 나머지 2모델 운영점 | 2 job | 안 하면 §1-1은 **영구히 2모델 문장** |
| 7 | γ — S3 하드웨어 프로브 | — | 유일하게 남은 스코프 구멍 |

**GPU 0 대기열**

- **① P5 재측정 사전등록의 (i) 임계 2차 감사 + (ii) engine-porter 구현.** 사전등록자 본인이
  *"§7 상수(0.10/1.30/1.10/5%/3%)는 내가 골랐고 외부 검증이 없다 — 이 문서는 감사 대상이지
  감사 결과가 아니다"* 라고 명시했다. 선행 구현 5건은 §12에 목록.
- **사전등록 전문 영속화**(위 §3).
- **S2 재현 스크립트 6개가 scratch에만 있다** → `results/s2_sticky/`로 이동.
- `analyze.py:183` `request_slice`가 **분자만 자르고 duration은 안 자른다**(분포 통계엔 무해,
  goodput 계산 시 즉시 오류). docstring 경고 없음 — engine-porter 후속.

**한 줄 시작점**: *`catch-up` 후 **Gate 1(0.2 GPU-hr)**부터 — §1-1의 기전 문장이 해금되는지
가르고, 이어 **Gate 2**가 논문 신규성 축을 정한다.*

---

## 5. 미완·주의

1. **전부 미커밋.** 마지막 커밋은 `6c06da8`(2026-08-03 작업). 정본 5회 개정·신규 문서 8건·
   코드 6파일·테스트 60건이 워킹 트리에만 있다. 커밋 여부는 사용자 지시 대기.
2. **P5 배치 1은 UNAUDITED**(`P5_GATES_BATCH1_2026-08-05.md`). 재설계 사전등록도 **미감사**.
3. **P5 배치 2·3 미제출** — NO-GO 상태. 재설계 확정 전 제출 금지.
4. **§1-1의 기전 문장은 금지 중**: 파티션·동시성·split 수치는 **Gate 1 전까지**, "PD 분리 자체"
   귀속은 **Gate 2 전까지**.
5. **크기 인용 금지 목록**: P1의 +41.8%/+398%/+174.9%(런길이 의존 + 캠페인 간 2.2× 불일치),
   §0의 3% 이하 비(δ 전까지), "11.26≈11.09 정밀 일치".
6. **prefill 축 캠페인(`results/s8p_prefill/`)은 여전히 미감사** — 정본 인용 금지 유지.
7. **§1-5 인용 금지 유지**: "attn 비중 5%→79%" · "ctx256 = 1.1× SM-free" · "knee 16→44→108→108".
   재측정 결과와 **무관하게** 인용 금지다(no-cudagraph micro 수치는 운영점 측정이 되살릴 수 없다).
8. **실행 중 job 없음**(`squeue` 비어 있음). 방치된 job 없음.
9. ⚠️ **오늘 C2 격자 이전 금지가 두 번 다른 방식으로 깨졌다** — 진단서가 결과를 C2와 비교하며
   한 번, 감사자가 자기 예측 밴드를 C2에서 뽑으며 한 번(후자는 금지 규칙을 쓴 문서의 **바로
   다음 문단**). §3-25·§3-26로 등재됨. 다음 세션도 같은 실수를 조심할 것.

---

## 6. 이번 세션의 방법론 산출 (6건)

| # | 내용 | 재발 |
|---|---|---|
| `CONSENSUS` §3-24 | `frac(reqITLp95>T)`는 SLO 임계 근처에서 못 쓴다 | metric cliff 새 사례 + **집계 단위 6번째** |
| §3-25 | C2 격자 이전 금지 | **§3-21의 3번째** |
| §3-26 / 게이트 #11 | **예측 밴드도 이식 금지 규칙의 적용 대상** | 4번째, **처음으로 감사자 자신이 위반** |
| 게이트 #12 | 임계 지시함수 판정은 **임계 사다리 + 큐 성장 검정**으로 견고성을 보여라 | 신규(절벽 플래그가 술어에 의존해 뒤집힘) |
| 게이트 #13 | **엔진이 다중 플래그를 강제하면 묶음 처치** — 운영 주장 가능, 기전 주장 불가 | 신규 |
| 게이트 #6 하위 5 | **위반 0건 술어의 goodput은 throughput의 다른 이름** — "차이<3% ⇒ 강등"을 판별력 0인 술어에 적용하면 무신호가 강등으로 둔갑 | #6의 새 사례 |
