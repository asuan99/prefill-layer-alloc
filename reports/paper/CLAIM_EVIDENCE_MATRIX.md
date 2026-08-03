# Claim–evidence matrix

최종 갱신: 2026-08-03(**등급 변경 없음** — 같은 세션 2차 속행: (I)
claims-auditor가 `A_free`를 대체하는 **조건부 per-token 추정량**
(`m3_conditional.py`)으로 estimand를 이관[AUDITED, blocking-threshold
스윕만 UNAUDITED — 감사자 자기산출 자기감사]; (II) engine-porter가
`PDMUX_STICKY_PARTITION`을 구현·correctness gate 통과시킴[구현 사실 —
**구현 완료 ≠ 성능 주장 성립**, sticky 격자 런 미제출]. Claim A Missing
evidence의 "GPU correctness gate 통과 이력 없음" 문구를 갱신. 상세
`../../PROJECT_STATUS.md` "8B decode-SM 프론티어" "2026-08-03(2차)" 소절,
`../CONSENSUS.md` §1-27. 이전(같은 날 1차 속행): E1 선행 게이트
job **872077**(M3 Transformer-control 대조)의 NO VERDICT 사유가 "CI 폭
부족"에서 **"estimand 미식별"**로 확장됐다(claims-auditor, `../CONSENSUS.md`
§1-26). 기판이 `initialize_stream_groups`에서 마지막 무분할 그룹을 항상
덧붙이는 구조라 "decode가 D SM에서 돌았다"⟺"prefill이 동시 in-flight였다"가
같은 사건 — 결정량 `g=A_free(d16)/A_free(d54)`는 **이 격자 한정 은퇴**
(sticky-partition 기판 수정 전 인용 금지, 블록 증설 재실행 선행 금지). 동시에
메인 세션이 세운 "decode 희석이 `g`를 attenuate했다"는 보정 가설은
claims-auditor에 **REFUTED**(control-arm reductio + de-engagement 실험 +
부분집합 분해 3중). ⚠️**"Ha8에 decode-SM 레버가 없다"는 CONFIRMED 아님** —
긴장 A(HE2 vs C2)는 **전혀 닫히지 않았다**(등급은 그대로 부분 지지). 상세는
`../../PROJECT_STATUS.md` "8B decode-SM 프론티어" 절 "2026-08-03" 소절.
이전: 2026-08-02(**등급 변경 없음** — 2026-08-01 실행된 E1 전제 실험
4건(jobs 870295/870296/870297 용량 스캔, 870301 batch-cap)은 **전부
claims-auditor 미통과**라 evidence로 올리지 않는다. 반영한 것은 (i) Claim A
Missing evidence에 **E1 설계 위험**(as-run 설계로는 프론티어 질문에 도달
못 할 수 있음, 미검증) 표시, (ii) Claim C의 KV 갈래에 **occupancy 데이터는
생겼으나 de-confound 안 됨**(`kv_mamba_occupancy=1.0`은 항등식) 표시,
(iii) "주장 제한"에 그 항등식 규율 추가. 상세는
`../../PROJECT_STATUS.md` "열린 긴장"·"방법론 게이트" #5/#6). 이전:
2026-07-28(★★★claims-auditor 사전등록 게이트 집행 — Stage 0의 Claim A
서빙 증거[D16≡D108 non-binding]를 **철회**(C1 CONFIRMED: D108 앵커가 실은 decode
16 SM). 대신 8B decode-SM 민감도 측정 노트[C2, scoped: prefill 16 SM 고정 시 SM16→
SM92 2.36–2.91×, 4 arm 모델-무관]를 Existing evidence에 추가, C2b["hybrid
급락=Zamba2 성질"]는 NOT-YET-SUPPORTED. **등급 변경 없음**[부분 지지] — 레버 존재
확립일 뿐 정책 이득 근거 아님, 프론티어 실험[E1] 전엔 등급 불변). 이전: 2026-07-26
(Stage 0/long-ctx L−2 게이트 서빙 증거를 Claim A에 추가 —
운영점 decode SM-무감각을 hybrid·pure-Transformer·pure-Mamba·ctx≤16k로 확장 확인,
등급 변경 없음 — ★2026-07-28 이 증거 철회, 위 참조). 2026-07-25(positioning 판정 추가 — 아래 "주장 제한" 마지막 항목,
증거 등급 변경 없음)

| Claim | 현재 판정 | Existing evidence | Missing evidence | Required experiment |
|---|---|---|---|---|
| A. Hybrid composition, context, active load에 따라 decode demand가 변한다 | 부분 지지 | Zamba2 context knee; synthetic/ShareGPT의 best split 이동; 다중 모델 batch/context characterization; ★**Stage 0(2026-07-26, `../stage0_verdict_2026-07-26.md`, jobs 864230+864601)**: 운영점(cudagraph-ON, green-context pdmux) 3-arm(pure-Mamba 음성대조/hybrid/pure-Transformer 양성대조) decode-only 스윕에서 de-confounded 대조 D16 vs D108(무경합) = 1.00±0.01, 3 arm×3 ctx(4k/8k/16k) 전부 — decode SM-무감각(lever-weakness)이 hybrid에서 pure-Transformer·pure-Mamba로, short-ctx에서 16k로 확장 확인(raw coupled 곡선 자체는 confounded였으나 음성대조+무경합앵커로 우회했다고 주장됨). ★★★**반증(2026-07-28, claims-auditor 사전등록 게이트 집행, C1 CONFIRMED)**: 이 증거가 인용한 D108(무경합 앵커)은 실제로는 decode 16 SM이었다(코드 버그+telemetry+클라이언트 서명 3중 증거, 상세는 `../../PROJECT_STATUS.md` "Stage 0" 절). "D16 vs D108=1.00±0.01"은 동일 조건 반복측정이었다. **대신 ★★C2(scoped, 2026-07-28, `../../workspace/engine-port/results/s8_scaleup/FINDINGS_8B_2026-07-28.md`, jobs 865289–865533)**: prefill을 16 SM에 고정한 채 decode-SM만 16→92로 올리면 decode ITL p50이 **2.36–2.91×**(4 arm: pure-Mamba2-7.3B/pure-Transformer-7B/additive·substitutive hybrid-7-8B, ctx1024, n=4) 개선 — **decode SM 민감도가 실재하고 모델-무관**임을 확인(Stage 0의 "SM-무감각" 전제와 정반대 방향). 단 이것은 **decode 측 등량곡선**(저-D 셀이 SM 일부러 idle, `prefill+decode≤108` 예산 제약 없음)이라 **레버 존재만 확립하며 정책 이득 근거가 아니다**(프론티어 ITL(D) vs TTFT(108−D) 미측정, `../../PROJECT_STATUS.md` "8B decode-SM 민감도 측정 노트"·"열린 긴장" 참조) | 동일 CUDA Graph 운영점의 **joint**(prefill+decode 동시) surface, GQA/composition 통제, held-out accuracy, >16k ctx, **프론티어(예산 제약 하 net-positive 여부, E1 미실행)**. ★**2026-08-02 추가 — E1 설계 위험(미검증)**: 2026-08-01 전제 실험(미감사, 인용 금지)에서 사전등록 사다리 {50,60,80}ms가 네 arm 전부 판정 불가. ★★**정정(2026-08-02, claims-auditor + M1/M2/M4)**: 여기 함께 적었던 "on-cliff 제외 규칙이 decode-rich 끝을 제거한다"는 **기각**한다 — C2 자신의 측정에서 SM16→44가 log-range의 **75–81%**라 d92 제외는 마지막 15–25%만 자르며, 근거였던 "공통 knee 2.80"도 철회됐다(치역=probe 격자, 살아남는 건 순서뿐; `DESIGN.md` §4.3.7). 따라서 **"E1이 설계상 도달 불가"는 아직 성립하지 않는다.** 대신 선행 게이트로 **M3 Transformer-control 대조**(job 872077, §4.3.8(c))를 제출했다: 프론티어 질문 이전에 **ITL 축이 D에 반응하는지**를 먼저 묻는다(오프라인 예비: blocking 제거 후 d16→d54 기울기 T8 2.03× 대 Ha8 ~1.0×). M3가 두 arm 모두 무반응이면 이 칸은 E1이 아니라 **"이 기판에서 conjunctive goodput의 ITL 항이 decode-SM 레버의 함수가 아니다"**로 닫힌다. ★★**정정(2026-08-03, claims-auditor)**: M3(872077)는 goodput 무반응이 아니라 **NO VERDICT — 사유가 "규칙 모호"에서 "estimand 미식별"로 확장**됐다. 이 기판(`initialize_stream_groups`가 마지막 무분할 그룹을 항상 덧붙임)에서는 "decode가 D SM에서 돌았다"⟺"prefill이 동시 in-flight였다"가 같은 사건이라, 어떤 통계도 decode-SM 탄력도와 prefill 간섭을 분리 못 한다(Claim B의 monolithic-prefill stall과 결합, `../CONSENSUS.md` §1-26). **결정량 `g`는 이 격자에서 은퇴**(sticky-partition 기판 수정 전 인용 금지). 메인 세션이 세운 "decode 실현 4–19%가 `g`를 attenuate했다"는 보정 가설도 **REFUTED**(control-arm reductio: T8 보정치 21–29×로 C2를 10배 위반). ⚠️"Ha8에 레버가 없다"는 **CONFIRMED 아님** — 긴장 A는 전혀 닫히지 않았다 | P3 full-model profile, feature ladder, leave-one-workload/model-family-out, E1(8B 프론티어 `[108−D,D]` 스윕 — **본 스윕 미제출**; 선행 게이트 M3[job 872077]는 estimand 미식별로 NO VERDICT. ★**2026-08-03(2차) 갱신**: `PDMUX_STICKY_PARTITION`이 구현 완료·correctness gate 통과(CPU 회귀 40 tests + sticky 단위 테스트 12 + GPU smoke job 872800 byte-identical 출력, `../../PROJECT_STATUS.md` "8B decode-SM 프론티어" "2026-08-03(2차)" 소절) — **구현 완료 ≠ 성능 주장 성립**, sticky 격자 런은 여전히 미제출(제출 전 `G_LEVER`/`G_FLAT` 사전등록이 미결). 동시에 `A_free`를 대체하는 **조건부 per-token 추정량**으로 estimand가 이관됨[AUDITED, blocking-threshold 스윕만 UNAUDITED]. 사전등록 정정: `--max-mamba-cache-size`는 공통 **절대상수**가 아니라 **`= cap` 규칙**으로 고정[slot당 비용이 arm마다 달라 절대상수는 새 교락]) |
| B. layer-level reconfiguration은 ITL critical path와 CUDA Graph를 훼손한다 | 강한 지지, 현 구현 범위 한정 | coordinated TPOT 약 42→124 ms, 최적화 후 약 85 ms; sub-step drain; graph incompatibility | 다중 모델 반복과 timeline attribution | B7 반복, CUDA Graph on/off, Nsight synchronization timeline |
| C. decode starvation은 TTFT도 악화시킨다 | running-batch 경로 강함; KV 경로 부분 (★2026-08-02 등급 불변) | D16 TTFT 7.24 s/ITL 61.9 ms 대 D24 1.21 s/39.9 ms; admission capacity 관측 | time-aligned KV occupancy와 admission reason. ★**2026-08-02**: occupancy 데이터 자체는 `results/s8_frontier/`(2026-08-01)에서 처음 생겼으나 **de-confound가 안 됐다** — hybrid arm의 `kv_mamba_occupancy=1.0`은 pool 크기가 `--max-running-requests`와 같아서 생기는 **항등식**이고(`model_runner_kv_cache_mixin.py:223-229`), 그 캠페인은 **claims-auditor 미통과(인용 금지)**다. 남은 Missing evidence는 그대로 | D16/D24 paired replay, structured KV/full/mamba occupancy(**`--max-mamba-cache-size`를 `= cap` 규칙으로 명시 고정해 항등식을 깬 뒤에** — 절대상수 고정은 arm마다 메모리 분할을 다르게 만들어 새 교락이 된다), mediation timeline |
| D. execution-state separation은 single-worker coupling을 줄인다 (★2026-07-24 코드 리뷰로 scope 축소, 아래 "주장 제한" 참조) | 미검증 | R1은 observer라 해당 증거가 아님; 2026-07-24 읽기 전용 코드 리뷰([`../r2_decoupling_review_2026-07-24.md`](../r2_decoupling_review_2026-07-24.md), file:line 근거)로 `PDMUX_TRUE_DUAL_WORKER=1`의 구조 확인: 두 host issue thread/role별 task queue/immutable `ExecutionContext`/thread-local role(ContextVar)만 분리하는 **control-plane dual-worker**이며, running batch(`max_running_requests`)·KV/mamba pool·SM 파티션(`SharedGpuArbiter` 단일 `stream_index`, ≤108)은 **전면 공유** | 실제 두 host loop에서의 fixed-split 비교(coupled ceiling 내); GPU correctness 동치 테스트(현재 없음); admission latch(`r2_admission_limited`) stale-True 버그 수정; results/r2_eval 캠페인 실행(현재 미생성) | legacy fixed 대 true dual fixed, 동일 telemetry/seed/graph — coupled ceiling(+2%, PROJECT_STATUS/CONSENSUS §1-20) 내에서만 유의미, "얽힘 깨기"로 측정 불가(§1-4 死因의 substrate가 구성상 불변) |
| E. Hybrid-informed decode floor가 generic/global static보다 높은 SLO goodput을 낸다 | 미검증 핵심 가설 | 없음 | architecture control, generic policy, static/oracle, profile generalization | B0–B8, P4 profile ablation, W1–W9 및 real trace |
| F. short-ctx conflict-regime 워크로드에는 동적 제어가 이길 수 있는 disjoint-feasibility escape hatch(어떤 static도 두 phase 동시 SLO를 못 만족하는 워크로드)가 있다 | **강한 지지(범위 한정) — CONFIRMED closure, scoped negative(2026-07-25)**: escape hatch **없음**을 확정 | `g2_0_full`(n=4, razor-thin real disjoint 최초 관측)→`g2_0_hard`(n=6–10, claims-auditor 재채점, ILL-POSED at rA5)→`g2_0_decliff`(rA2 n=6, PLAUSIBLE closure)→`g2_0_rasweep`(120 job, off-cliff band rate≤2.75 disjoint 재확인 없음)→`g2_0_raconf`(pre-registered 24-job 확증 열, rate{3.5,3.75}×{d44,d54}×n6, companion-collapse 결정규칙 충족: rate3.5 d44 0.953±0.035≈d54 0.948±0.035; rate3.75 d54 0.948±0.062>d44 0.932±0.042) | long-context(decode floor 상승 영역, CONSENSUS §1-5) 재검증; hot varying-trace(drain 아닌 entangled 조건)에서의 직접 실증 | long-context G2.0-style disjoint sweep(모델/ctx 교체 필요); §1-20 spatial coupling-tax와 결합한 재검토 |

## 주장 제한

- ★★★**철회(2026-07-28, claims-auditor 감사, C1 CONFIRMED)**: Claim A의 Stage 0
  서빙 증거(2026-07-26, {pure-Mamba2-2.7B/hybrid Zamba2-2.7B/pure-Transformer
  Qwen2.5-3B, triton attn+mamba, cudagraph-ON green-context pdmux, ctx≤16k,
  coupled 하네스, one-shot 32-conc burst}에 한정하려던 "decode SM-무감각
  (non-binding)" 결론)는 근거였던 D16 vs D108 de-confounded 대조 자체가
  무효(D108은 실제로는 decode 16 SM)라 **전량 철회**한다. 상세는
  `../PROJECT_STATUS.md` "Stage 0" 절.
- **대신 C2(scoped, 2026-07-28)**: {Mamba-Codestral-7.3B/Zamba2-7B/
  Nemotron-H-8B/Qwen2.5-7B, A100 80GB TP1, cudagraph-ON green-context pdmux,
  `--disable-overlap-schedule --chunked-prefill-size -1 --disable-radix-cache`,
  max-running-requests 48, ctx1024, conc16 closed-loop, out512, n=4 rep}에서
  prefill을 16 SM에 고정한 채 decode-SM만 16→92로 올리면 decode ITL p50이
  **2.36–2.91×**(4 arm, 모델-무관) 개선된다 — 이는 **decode 측 등량곡선**(저-D
  셀이 SM을 일부러 idle, `prefill+decode≤108` 예산 제약 없음)이라 **레버
  존재만 확립**하며 정책(SLO goodput) 이득 근거가 아니다(프론티어 ITL(D) vs
  TTFT(108−D) 미측정). ctx1024만 귀속 측정, ctx4096은 엔진측 ITL-EWMA
  프록시로 유지 관찰(보조 증거)뿐, 8k/16k는 미측정. **C2b("hybrid 급락=
  Zamba2 additive 성질")는 NOT-YET-SUPPORTED** — 모델간 절대비교가 통제되지
  않아 기전 주장으로 쓰지 않는다. 전문은 `../PROJECT_STATUS.md` "8B
  decode-SM 민감도 측정 노트" 절, 상세
  [`../stage0_verdict_2026-07-26.md`](../stage0_verdict_2026-07-26.md).
- ★★★**(2026-08-03) `g = A_free(d16)/A_free(d54)`(job 872077, M3
  Transformer-control 대조)는 인용 금지 — 이 격자에서 estimand 미식별.**
  `../CONSENSUS.md` §1-26. C2와 달리 E1/M3 격자는 prefill과 decode SM이
  상보적(P+D=108)이고 기판이 decode-empty 시 무조건 무분할로 되돌아가므로,
  "decode가 D SM에서 돌았다"와 "prefill이 동시 in-flight였다"가 같은
  사건이다 — sticky-partition 구현 전엔 어떤 `g` 수치도 decode-SM 탄력도와
  prefill 간섭을 분리 못 한다. **인용 시**: 긴장 A(HE2 vs C2)는 **미해결**
  이라고만 쓰고, "Ha8은 레버가 없다/있다" 어느 쪽으로도 인용하지 않는다.
- ★★**(2026-08-03, 같은 세션 2차 속행) `A_free`는 은퇴, 조건부 per-token
  추정량으로 교체.** `results/s8_frontier/DESIGN.md` §4.3.10(`m3_conditional.py`).
  단위 = 개별 ITL 구간, SPLIT(`split_frac≥0.90`)/UNSPLIT(`≤0.10`) 라벨, primary
  `p95(SPLIT)` 비 + UNSPLIT control(대비 정의상 0). **[AUDITED]**: 정의·시계
  정렬 규율(`ALIGN_R_MIN=0.95` flag-only, `phase=="benchmark"` 필터 금지).
  **[UNAUDITED — 별도 확증 전 인용 금지]**: blocking 임계 스윕(무릎 없음,
  임계 0서 대비 소멸). 이 추정량은 `A_free`보다 잘 정의됐을 뿐 **사전등록된
  SLO 항이 아니다** — `A_all`은 계속 병기. `PDMUX_STICKY_PARTITION`
  구현·correctness gate 통과(§4.3.11, `../../PROJECT_STATUS.md` "8B
  decode-SM 프론티어" "2026-08-03(2차)" 소절)는 **구현 사실**이며 위 872077
  기반 등급·인용 제한을 바꾸지 않는다 — sticky 격자 런 자체가 아직 없다.
  sticky 런 사전등록(§4.3.12)의 `G_LEVER`/`G_FLAT`는 **미결정으로 기록**됐다
  (스케일 불일치로 기존 임계 이전 불가).
- Claim B는 A100/SGLang green-context implementation에 한정한다.
  ★**positioning 판정(2026-07-25, `venue_positioning.md` §0.1)**: 이것은 논문의
  negative 중 "(A) green-context 종속" 축이다 — DuetServe(libsmctrl)가 정면으로
  우회한 비용이므로 substrate-invariant로 헤드라인화하지 말 것.
- Claim C는 KV telemetry 전까지 “shared running-batch/capacity congestion”으로
  표현하고 KV causal chain을 확정하지 않는다.
  ★**추가 규율(2026-08-02, 코드 사실)**: `kv_mamba_occupancy = 1.0`을 KV
  causal chain의 증거로 **쓰지 않는다**. `disable_radix_cache ∧
  max_running_requests` 조건에서는 `max_mamba_cache_size =
  max_running_requests`이므로(`sglang/srt/model_executor/
  model_runner_kv_cache_mixin.py:223-229`) pool 크기 = cap이고 batch가 cap에
  닿으면 그 값은 **정의상 1.0**이다. 같은 이유로 **`--max-running-requests`는
  arm 계열마다 다른 손잡이**다(T8=admission만 / SSM 포함 arm=admission +
  mamba state pool) — cap을 바꾼 실험 결과를 arm 계열 간에 이전하지 않는다.
  KV 경로를 측정하려면 `--max-mamba-cache-size`를 전 arm 공통 상수로 명시
  고정해 두 축을 분리해야 한다. `../CONSENSUS.md` §1-23·§3-14,
  `../../PROJECT_STATUS.md` "방법론 게이트" #5/#6.
- Claim D는 true dual fixed가 architecture gate를 통과한 뒤에만 사용한다.
  ★**2026-07-24 코드 리뷰 정정**(engine-porter, 읽기 전용,
  [`../r2_decoupling_review_2026-07-24.md`](../r2_decoupling_review_2026-07-24.md)):
  현재 `PDMUX_TRUE_DUAL_WORKER=1` 구현은 **control-plane dual-worker**다 — 두
  host issue thread, role별 task queue, immutable `ExecutionContext`,
  thread-local role(ContextVar)만 분리한다. running batch(`max_running_requests`,
  완료 prefill을 같은 batch로 in-place merge), KV/mamba pool, SM 파티션
  (`SharedGpuArbiter`의 단일 `stream_index`, ≤108)은 **전면 공유**된다(92 prefill
  SM + 24 decode SM = 116의 별도 device pool이 아니다). Claim C의 死因 얽힘이
  사는 substrate(공유 running-batch+KV)를 이 구현은 **구성상 깰 수 없다** —
  관측되는 win/loss는 host-thread overlap(control-plane)에 귀속되며, coupled
  ceiling(+2%, PROJECT_STATUS/CONSENSUS §1-20)을 초과할 수 없다. 그 위의
  +16% headroom은 별도 device pool disaggregation과 hybrid(mamba conv/ssm)
  state transfer를 요구하며 **둘 다 미구현**(state-transfer 경로 코드에 전무).
  따라서 **Claim D는 "control-plane coupling 감소"로 스코프를 축소**하고,
  "얽힘을 깬다"는 프레이밍으로 쓰지 않는다. R2는 GPU correctness gate를
  통과한 이력이 없다(`results/r2_eval/` 디렉터리 미생성, `architecture=true_dual`
  telemetry 전무). admission latch(`r2_admission_limited`)에 known-latent
  stale-True 버그(split batch가 None으로 배수되면 재평가 경로 없어 latch가
  True로 고착, clear 경로 부재)가 file:line 근거로 확인됐다 — **사용자 결정으로
  현재 수정하지 않고 보류**한다. 판정은 여전히 **미검증**이며, 이는 성능 판정이
  아니라 코드-구조 사실이다.
- Claim E는 B6가 B1과 B5를 모두 유의하게 이긴 경우에만 사용한다.
- D/E가 실패하면 A–C의 characterization 및 negative result를 논문의 중심으로
  유지한다.
- Claim F는 {Zamba2-2.7B, ctx4096, Phase A in2048/o32, Phase B in2048/o512@rB4,
  triton attn+mamba, disable-radix-cache, cudagraph-ON, A100 108-SM
  green-context pdmux, SLO=TTFT 3s ∧ per-req ITL-p95 50ms, inter-phase drain된
  순차 2-phase, rate_A≤3.75}에 한정한다. **"hybrid엔 disjoint 없음"으로
  일반화하지 않는다.** Phase-A frac_good≈0.95는 metric cliff(웜업성 tail-event)
  가 결정해 magnitude는 run-length 의존·ill-posed이나, 순위(d54≈d44, d54
  미선-배제)는 견고하다. §1-20(spatial coupling-tax, disaggregation +16%
  headroom)과는 별도 축 — "단일 static으로 시간축 커버 ⟹ coupling tax 없음"으로
  새지 않는다. 상세
  [`../../workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md`](../../workspace/engine-port/results/g2_0_raconf/raconf_final_verdict_2026-07-25.md).
- ★**positioning 판정(2026-07-25, venue-strategist prior-art 조사 + doc-steward
  기록, [`venue_positioning.md`](venue_positioning.md) §0.1)**: 논문 negative를
  substrate-robustness 축으로 (A)/(B) 두 갈래로 나눠 읽는다 — **(A) green-context
  종속**(Claim B: layer-aware 死, cudagraph 비양립, TPOT 42→124ms) vs **(B)
  mechanism-independent 후보**(Claim A의 lever-weakness=mamba decode SM-둔감 +
  Claim C의 entanglement/decode 비대칭). (A)는 DuetServe(libsmctrl +
  interruption-free engine으로 이 비용을 우회하고 실제로 static을 이긴 선행)가
  정면으로 우회한 바로 그 비용이므로 **substrate-invariant negative로
  헤드라인화하지 않는다**(Claim B의 기존 "A100/SGLang green-context 한정" 스코프를
  그대로 유지). ★**green-context vendor-primitive 방어**: NVIDIA 공식
  fine-grained SM primitive는 green-context(CUDA 12.4+) 하나뿐 — libsmctrl은
  비-vendor·arch-bound, MPS는 정적 → "libsmctrl 쓰면 되잖아"는 배포 불가한
  research curiosity이지 배포 가이드라인의 반례가 아니다.
  ⚠️**2026-07-25 정정(초판 철회)**: (B)의 substrate-robustness를 닫는 acceptance
  실험을 초판은 "libsmctrl/MPS로의 cross-substrate serving 이식"으로 걸었으나,
  그 이식은 **불필요·부적합**(MPS=정적·프로세스별, libsmctrl=비-vendor·세대귀속)
  이므로 **철회**한다. 대신 **기존 green-context 위에서**: (1) **Transformer-
  control 대조**(순수 Transformer를 같은 green-context+같은 conjunctive-SLO에
  통과 → drain 상쇄 → 동적-승/패가 모델에서 갈리면 hybrid 귀속 식별) + (2)
  **roofline lever-weakness microbench**(r0c SM-민감도, primitive-robust) +
  (3) **기측정 entanglement 귀속**(`switch_count`≈0·컨트롤러 0.014% → 동적-패가
  drain 탓 아님, 헤드라인 HE0 무관)으로 닫는다(`EXPERIMENT_ROADMAP.md` 벡터2/
  TC-series). **이 판정은 등급 변경이 아니다** — Claim A는 여전히 부분 지지,
  Claim B는 여전히 강한 지지(현 substrate 한정), Claim C는 여전히 running-batch
  경로 강함/KV 경로 부분이다.
