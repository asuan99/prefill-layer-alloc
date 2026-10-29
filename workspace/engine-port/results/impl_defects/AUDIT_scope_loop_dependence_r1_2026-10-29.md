# 감사 r1 — 정본 결론의 SGLang PD-mux 이벤트 루프 구조 의존 범위 (2026-10-29)

> claims-auditor, 1회차(상한 2회), 읽기 전용, GPU 0. 정본·§6 등급 변경 0(제안만, 반영은 promote).
> 코드 기준 `workspace/engine-port/src/multiplex/multiplexing_mixin.py` @3858f129(HEAD 6f327b09).
> 입력: `impl_defects/PHASE0_CPU_CHECK_2026-10-28.md`, `phase0_c1_c2.md`, `phase0_c3.md`(미감사 → 이 감사가 쓰는 범위만 교차 검증).
> (감사자가 읽기 전용이라 main 세션이 반환 전문을 그대로 저장했다.)

## 0. 루프 구조 사실 — 독립 재확인

| ID | 사실 | 코드 | 확인 |
|---|---|---|---|
| B1 | SLO/R2 전환은 `_tgt != stream_idx`일 때 prefill과 decode stream을 동기화한 뒤 전환한다. 평가 조건은 `not wait_prefill_kernel_done`이다. 마지막 chunk 발행 뒤에는 전환할 수 없으므로 진행 중 drain은 span이 2개 이상일 때만 생긴다 | `:1317-1356` | 확인 |
| B2 | `stream_idx>0 ∧ running_batch.is_empty()`이면 adjust가 실행된다. drain은 이미 발행된 span만 기다리고, 남은 층은 idx 0(108,0)에서 돈다 | `:1308-1310`, `:1357-1374`, `:1200-1210` | 확인 |
| B3 | prefill이 진행 중이면 새 prefill을 입장시키지 않는다(`if self.split_prefill_batch: return False`). preemption은 없다. `sm_count` 인자는 쓰이지 않는다 | `:1218-1238` | 확인 |
| B4 | `forward_count = max(1, budget // extend_num_tokens)`. 층 종류·P·decode step은 입력되지 않는다. 반복마다 동기화하는 것은 decode stream뿐이다(`:1494`). prefill 완료는 `query()`(`:1518`)로 확인하며 backpressure가 없다 | `:1413-1443`, `:1494`, `:1518` | 확인 |
| 추가 | 비sticky에서 prefill이 없을 때 decode는 `real_sm_group_num-1`(전 SM)에서 돈다. 따라서 static D 라벨은 동거 구간에서만 의미가 있다 | `:1198-1199` | 확인(결정 49 caveat와 같음) |
| 추가 | `PDMUX_SLO_SPAN_TYPE`: 층 종류 경계에서 span을 자른다(시간 맞춤이 아니라 경계 절단) | `:1430-1439` | 코드에 있음, 비정본 측정 1회 |
| coord | 기본형은 window마다 `d_s.synchronize()`, window 0에서 `p_s.synchronize()`를 한다(`:1894-1896`). OPT·PF·V4는 GPU쪽 `wait_stream`이고 CPU 동기화는 step당 1회다(`:1898`, `:1728-1730`). 모든 변형이 decode를 `MR.forward_split_decode`(eager)로 돌리고, prefill chunk 크기는 B4 식(`:1658`, `:1847`)을 그대로 쓴다 | 위 | "창 경계마다 동기화"는 **기본형에만** 맞는다 |

외부 대조(PDF 직접 확인):
- MuxWise §3.2.2–3.2.3: decode를 먼저 발사하고 층 단위 prefill을 쓴다. query 기반 동기화와 preemption이 있다. "prefill launch > decode iteration → bubble"을 명시적으로 다룬다.
- Bullet: libsmctrl stream mask를 쓴다. step마다 층 수만큼 발사하고 동기화한다. 전환 구간에서는 SM을 부분 공유한다.

이 셋은 서로 다른 루프다. 정본의 serving 결론은 전부 이 루프 하나에서 나왔다.

## 1. 판정 표

증거 유형: **C** = 루프와 무관한 성분 측정, **S** = 이 루프를 거친 serving 결과, **M** = micro/sim.

| # | claim (file:line) | 증거 | 의존(B1–B4/coord)과 편향 방향 | 문구 판정 | 등급 | 최소 수정 문구 |
|---|---|---|---|---|---|---|
| 1 | CONSENSUS §0 "그 위의 똑똑한 정책은 전부 실패", "layer-type 기반 정책은 全형태 死" (`reports/CONSENSUS.md:3666-3667`) | S+M | 아래 2·5행 전부를 상속한다 | **OVERBROAD**. 시험한 정책은 split index 반응형 컨트롤러와 층 경계 SM 전환 계열뿐이다 | CONFIRMED-scope-issue | "SGLang 공개 PD-mux 루프에서 시험한 정책(층 경계 SM 전환 계열, split index 반응형 컨트롤러)은 best static을 넘지 못했다." |
| 2 | §1-3 "layer-type 런타임 정책 全형태 死" (`:3682`), CLAUDE.md:39 | S(no-cudagraph)+M | **coord sync**(기본형 124 ms 중 약 47%가 sync·pin, OPT로 85 ms), **B4**(monolithic prefill chunk를 window 0에서만 겹침, `A_substrate_isolation_results.md` "잔차 = 구조적"), eager decode(`forward_split_decode`). 편향은 전부 **layer-type에 불리**하다. 근거 ①인 4/4 승은 미조율 la(postmortem:20-26) | **OVERBROAD**. confound 2(4/4는 미조율 배선), 8(전 측정 `--disable-cuda-graph`: `a_substrate/a_coord_opt_bench.sbatch:41`, `pf_boundary/pf_bench.sbatch:41`), 11(KISTI 단일), 3(셀당 n=1, 단 격차 3–7배라 부호는 견고). 반증되지 않은 것: 층 종류별 시간 맞춤 span 크기, 박자 영역, handoff, preemption, 창별 piecewise cudagraph | CONFIRMED-scope-issue | "KISTI A100·Zamba2-2.7B·triton·no-cudagraph에서, SGLang 공개 PD-mux 루프 위에 구현한 '층 종류 경계마다 SM 분할을 바꾸는' 정책 계열(coord R0d·OPT·v4·PF)은 모두 agnostic·tuned static에 졌다(TPOT 42→124 ms, 최적화 후 85 ms). 층 종류를 span 크기(시간 예산)로만 쓰는 정책, 박자 영역, handoff, preemption, cudagraph-ON 구현은 시험하지 않았다." |
| 3 | §1-15 "살아있는 offline 역할은 오직 whole-phase floor", "나머지 절반은 pre-created로도 불가" (`:3719`) | S(no-cudagraph) | coord sync와 B4(window 0 단일 겹침). "cudagraph 비양립"은 이 구현의 성질이다(창별 piecewise graph는 시험 안 함) | 구현 기전 서술은 OK. **"오직 whole-phase floor"는 OVERBROAD** | CONFIRMED-scope-issue | "…offline 고정이어도 이 구현의 층 경계 SM 전환은 실행 시 비용을 낸다. step 안 SM 분할을 바꾸지 않는 층 종류 활용(span 크기·발행 시점)은 이 판정 범위 밖이다." |
| 4 | §1-1 / Claim G +40.5/+185.8/+27.0% (`:3680`, `CONSENSUS.md:8786`, CLAUDE.md "현재 결론" 1) | S(cudagraph-ON, n=5 paired) | 적체 영역(phase0_c3 §4.5, [유도]). B1은 해당 없음(agnostic). B2는 TTFT에서 PD-mux에 불리. 적체(무 backpressure)는 prefill을 P SM에서 쉬지 않고 돌게 하므로 TTFT에 유리할 수 있어 **방향 혼합**. 수리해도 부호가 뒤집힐 루프 경로는 보이지 않는다. 주 confound는 여전히 미조율 fused(confound 2) | **NEEDS SCOPE**(구현 명시). 정본은 이미 "기전 미확립"·"SM 분할 자체 NOT-YET-SUPPORTED"로 범위를 좁혀 둠 | PLAUSIBLE | "SGLang 공개 PD-mux 구현(lockstep 단일 프로세스 루프, budget 65536, 적체 영역) 활성화는 미조율 기본 fused 대비 …" |
| 5 | §1-7·§1-17 HE0 "동적이 best-static을 못 넘음" (`:3686`, `:3722`), CLAUDE.md:41-42 | S(n≥4, 변화 trace) | **B1**: bind+GATE에서 영향 받을 수 있는 HI 요청 ≤0.72%로 3% 미만. slo는 run당 10–11%로 배제 불가. no-gate는 상한 정보 없음(phase0_c1_c2). B1은 **동적만** 벌한다. **B3·B4**: 결정 시점이 prefill 배치 입장으로 제한되고 94–95%가 span 1개 → 동적 class의 반응 입도가 루프에 묶인다. 편향은 **동적에 불리**하나 핵심 비교(d44 대 bind+GATE, HI −6.8%)를 닫기에는 크기가 부족하다 | 부호 문제 없음. 2026-10-02 재서술은 이미 좁혀져 있으나 루프 조건이 빠져 있어 **NEEDS SCOPE** | PLAUSIBLE (부호는 NOT-AN-ISSUE) | 기존 재서술에 한 구절 추가: "…SGLang 공개 PD-mux 루프(결정 = prefill 배치 입장 시점, 전환 = 진행 중 prefill drain, 이 trace 배치의 94–95%는 span 1개)에서…" |
| 6 | "실패 원인 = switch overhead 아님" (CLAUDE.md:42, §1-17 positioning `:3717`, COSTMODEL:73) | C(CPU 0.014%) + C(agnostic 경로 `s≤0.04 ms`) | **B1이 측정되지 않았다**(CONSENSUS §9:8808 자인). 근거는 agnostic adjust 경로 기준이라 컨트롤러 전환 경로로 옮길 수 없다(§1-15 각주 금지 조항) | **OVERBROAD**(arm 일반형). bind+GATE에는 OK | CONFIRMED-scope-issue | "컨트롤러 CPU 비용과 agnostic 경로 전환 비용은 무시 가능하다. 컨트롤러 전환의 진행 중 drain(`:1344`)은 미측정이며, 상한상 bind+GATE에는 작고(≤0.72%) slo arm에서는 배제할 수 없다. bind+GATE의 열위는 positioning으로 설명된다." |
| 7 | §1-4 entanglement 사슬 (`:3683`), Claim C (`:8782`) | S(방향, magnitude 인용 금지) | cap 구속은 SGLang 스케줄러 공통이다(루프 무관). **B3**(prefill 입장 직렬)가 admission 경로를 하나 더 만든다. 방향은 판별 불가(B3는 prefill SM이 적을수록 입장 대기를 늘리므로 decode-heavy에 불리할 수도 있다) | 방향 서술은 OK. 크기·일반성은 **NEEDS SCOPE** | PLAUSIBLE | "(cap 구속, prefill 입장 직렬 루프에서)" 병기 |
| 8 | §1-6 비대칭 "과다공급 거의 무해" (`:3685`) + 실전 권고·CLAUDE.md:44 "실무 기본값 decode-heavy static" (`:3729`) | S | **비sticky fallback + B2**: prefill이 없으면 decode가 108 SM을 쓰고, decode가 비면 새 span은 (108,0)이다. 그래서 decode 과다공급의 비용은 동거 구간의 prefill 감속뿐이다. 이 루프 구조가 **decode-heavy 쪽으로 기우는 편향**을 만든다. 고정 분할(Bullet식 mask)이나 sticky에서는 과다공급 비용이 커진다. HE0 LO는 prefill p50 252 tok에 span 1개라 워크로드 몫도 크다 | **NEEDS SCOPE**. 권고는 이 루프의 비sticky 동작에 묶인다 | PLAUSIBLE | "…비sticky SGLang PD-mux 루프(decode-only·prefill-only 구간은 전 SM 복귀)에서는 decode-heavy static 라벨의 과다공급 비용이 동거 구간에만 생긴다. 라벨은 목표 D이지 실현 할당이 아니다(결정 49)." 기본값은 결정 40/43대로 Nano에서 다시 정한다 |
| 9 | §1-13 "LO는 split에 무관심" (`:3711`) | S | 8행과 같다(LO prefill 짧음, 비sticky 복귀) | **NEEDS SCOPE** | PLAUSIBLE | "(이 trace의 LO에서 prefill 배치 94–95%가 span 1개이고 비sticky 복귀가 있는 루프에서)" |
| 10 | §1-24 "ITL 꼬리 = monolithic prefill이 decode를 멈춘 시간 — 구조적, 고칠 수 없다" (`:3698`), COSTMODEL:50 "구조 확인", COSTMODEL:86 답 4 | S(KISTI, rate 2, probe 17) | 정지 기전은 미확정이다(lockstep 감사 r1: KISTI phase 계측 0, 게이트 #83). VESSL·Nano에서는 별도 정지 항이 관측되지 않았다(CM-4). 후보 기전인 prefill 발사 시간 > decode iteration은 MuxWise가 bubble로 명시한 **루프 구조**(B3/B4, decode sync 다음 발사)다. "고칠 수 없다"의 근거(chunked prefill assert)는 span 페이싱·발행 깊이(E1/E2)·decode 우선 발사 수리를 배제하지 못한다 | **OVERBROAD** | CONFIRMED-scope-issue | "KISTI·Zamba2-2.7B·triton의 이 루프에서, ITL 꼬리는 최장 프롬프트 prefill과 시간상 겹치는 정지가 지배했고 그 크기는 D에 단조였다. 정지 기전은 미확정이며 VESSL·Nano에서는 관측되지 않았다. PD-mux가 chunked prefill과 비양립한다는 것은 엔진 사실이다." |
| 11 | CM-4 lockstep "동거 decode 간격은 span 시간에 거의 반응하지 않음"·"별도 정지 항 미관측" (`COSTMODEL.md:21`, `:54`) | S(기술값) | **B4 적체**: 표본이 N3 98.1%, N2-EA 93.6%, T3 100% 적체다. 적체에서는 host가 prefill을 기다리지 않으므로 decode 간격과 span 시간은 루프 구성상 거의 독립이다. 결정 50 r2 판정("A의 R·Z 기울기 = 루프 회계 항등식")과 같은 부류이고 교훈 (9) 계열이다. 박자 영역은 측정 0 | **NEEDS SCOPE**. 일반화하면 OVERBROAD | CONFIRMED-scope-issue | 기존 허용 문안 끝에 추가: "이 run들은 적체 영역(span > 동거 decode step, prefill 미동기 발행)이었고, 이 영역에서 무반응은 루프 구조상 예상되는 값이다. 박자 영역에서는 측정되지 않았다." |
| 12 | §1-38 / CM-4 N2 Job-EA ρ (`:3727`, `COSTMODEL:51-52`) | S(성분, 기술) | 93.6% 적체이고 지속 prefill backlog를 강제했다. 박자 영역의 동거 duty cycle은 다르다 | 이미 E-α 8셀로 범위가 좁혀져 있음. 영역 병기 권고 | NOT-AN-ISSUE(경미) | "(적체 영역 측정)" 병기 |
| 13 | COSTMODEL "이미 답한 것" 1 "결정 비용 무시 가능 → 정책 성패는 예측 정확도에 달려" (`:82`) | C | B1 미측정. 루프 구조(B3/B4)도 성패의 변수다 | **OVERBROAD** | CONFIRMED-scope-issue | "…정책의 성패는 예측 정확도와 실행 루프 구조(발행·입장·전환 drain)에 달려 있다. 후자는 공개 루프 하나에서만 측정됐다." |
| 14 | COSTMODEL "이미 답한 것" 2 "층 종류 차이는 런타임 분할 레버로 쓸 수 없고" (`:83`), CM-2 "존재하나 레버가 아니다" (`:39`) | M+S | Diff B(SM 민감도 비)는 ≈1이지만(micro, no-cudagraph, triton). **Diff A(비용 비)는 L에 따라 크게 열린다**(§1-3 배너, 같은 범위). 시간 맞춤 span 크기에 필요한 것은 Diff A인데 정본은 두 레버를 섞어 쓴다. 같은 파일 `:43`은 "N_PL 균일 가정" 문제를 열어 둔다 | **OVERBROAD** (내부 모순) | CONFIRMED-scope-issue | "…층 경계 SM 재분할 레버로는 쓸 수 없다(이 구현). 층 종류별 비용 차를 span 시간 예산에 쓰는 레버는 미시험이다(CM-2 열린 질문)." |
| 15 | §1-16–§1-20 mix-swing·disjoint·oracle (`:3714-3721`), Claim F (`:8785`) | S | 동적 class는 split index 반응형이고 결정 시점은 입장이다(5행). F의 static feasibility는 루프 결합(B3 입장 직렬, 비sticky)을 포함한 값이다 | **NEEDS SCOPE**(경미). F는 이미 "범위 한정" | PLAUSIBLE | "(SGLang 공개 PD-mux 루프의 static·split index 컨트롤러 기준)" |
| 16 | Claim B "층 단위 재구성은 ITL critical path와 CUDA Graph를 훼손" (`:8781`) | S(no-cudagraph) | coord sync | 이미 "현 구현 범위 한정"이라 OK. "CUDA Graph 훼손"은 창별 piecewise graph 미시험이라 구현 진술로만 유효 | NOT-AN-ISSUE | 그대로 |
| 17 | §1-2·§1-5·§1-8/§1-34·§1-12·§1-14·§1-22, Claim A·D·E | C·M·코드 | 루프와 무관하거나, 루프 사실 자체이거나(§1-22), 이미 범위가 있다(§1-8 "agnostic 경로") | OK | NOT-AN-ISSUE | — |
| 18 | Phase 0 문서 §2.4 "HE0 = PD-mux 다중화가 거의 일어나지 않는 영역" (`PHASE0_CPU_CHECK_2026-10-28.md` §2.4) | 코드+측정 | span 1개여도 prefill(P SM)과 decode(D SM)는 공간적으로 동시에 돈다. 없는 것은 층 단위 span 분할이다(phase0_c3 §4.4의 문구가 맞다) | **OVERBROAD**(반대 방향 과교정) | CONFIRMED-scope-issue | "HE0는 층 단위 span 분할이 거의 일어나지 않는(prefill 전체를 한 번에 발행하는) 영역이다." |

## 2. 시험된 적 없는 것과, 정본이 이것을 "죽었다"고 함의하는지

| 미시험 항목 | 정본이 죽었다고 함의하는가 |
|---|---|
| 층 종류별 **시간 맞춤** span 크기(MuxWise N_PL + 층 비용 가중) | **함의함**: §0:3667, §1-3 "全형태", §1-15 "오직 whole-phase floor", COSTMODEL:83, `policy_comparison.md:67` "span 경계로도 이득 無"(비정본). 유일한 근접 측정 `PDMUX_SLO_SPAN_TYPE`은 경계 절단(시간 맞춤 아님)이고 eager·n=1·비정본이다. TTFT 악화는 "반복마다 span 1개 발행 × decode step"이라는 B4 페이싱으로 설명된다(confound 2, PLAUSIBLE). 이를 반증으로 인용하면 confound 2 |
| 박자 영역(r<1) | 명시적 함의는 없다. CM-4 일반화가 암묵적으로 덮는다(11행) |
| handoff(decode가 끝나면 남은 층을 새 green ctx로 이동) | 없음. 단 Phase 0이 보인 대로 E3만으로는 B2를 고치지 못한다 |
| preemption(층 경계 prefill 선점) | 없음. §1-7 "시험 안 된 것" 목록에 "요청 재정렬·진행도 제어"로 일부 들어 있음 |
| 발행 깊이 제한, backpressure(E2) | 없음 |
| coord 창별 piecewise cudagraph | **함의함**: §1-15 "운영점 진입 불가"를 불가능성으로 서술 |
| 별도 프로세스(Bullet식)·true dual worker(R2 H-Architecture) | 없음. 미검증 R2로 남아 있음 |

## 3. 확정 실험 (Phase 2 엔진 수정 `impl-defects-e123` 뒤, 기본 OFF, 한 Job 안 arm)

1. **층 종류 span 정책** (2·3·14행을 닫는다)
   - 조건: VESSL A100, Nano-9B-v2, flashinfer, cudagraph-ON.
   - 박자 영역 확보: budget 512×D44 셀 또는 E2 depth=1. 양성 대조로 `prefill_chunk.gpu_ms < decode_fwd_gpu_ms`를 먼저 확인한다.
   - arm: {budget 65536, time(균일), time_typed} × static D44.
   - 반복: boot n≥4 paired, 변화 trace(rate 3↔12 비례 환산, cliff 회피, 용량 선측정).
   - 판정: time_typed 대 time의 goodput(게이트 4) paired CI가 0을 넘고 |Δ|≥3%.
   - 병기: TTFT/ITL p50/p95/p99.
   - 결과가 음성이어야 "全형태" 문구를 좁힌 채로 유지할 수 있다.
2. **B1 크기** (5·6행)
   - 진단 PREREG 새 rev: span 2개 이상을 강제한 셀(8k)에서 동적 arm이 진행 중 전환을 일으키게 하고, PHASE_DIAG drain ms를 잰다.
   - 이어서 Nano HE0 핵심 셀(결정 43/44)에서 bind+GATE·slo 대 best static, n≥4 paired. E3 handoff ON/OFF는 같은 Job의 별도 arm으로 둔다(변수 하나씩).
3. **비sticky 편향** (8·9행)
   - D{24,44} × {sticky OFF, ON}, LO·HI phase, n≥4 paired.
   - 판정: LO에서 D44−D24의 goodput 차가 sticky ON에서 ≥3% 벌어지면 "과다공급 무해"는 비sticky 루프 한정으로 확정한다.
   - 한 Job 안에서 돌리고, 두 노브를 동시에 바꾸지 않는다(sticky만 바꾼다).
4. **CM-4 영역** (11행)
   - phase0_c3 경계 4셀(8192×D24/D44)과 512×D44에서 `gpu_ms` 짝을 재고, 기울기를 다시 추정한다.
   - 박자 영역 기울기가 0과 구분되면 CM-4 일반형은 REFUTED.
5. **§1-24 KISTI 정지**: 결정 46(Zamba 배제)으로 판별할 수 없다. 범위 문구 교정만 하고 실험은 처방하지 않는다.

## 4. 반대 방향 검토 (과교정 방지)

- 부호를 뒤집는 루프 경로를 찾지 못한 것:
  - Claim G: B2와 B1은 PD-mux에 불리하거나 해당이 없다.
  - HE0 핵심 비교: B1 상한 ≤0.72%가 6.8% 격차를 닫지 못한다.
  - layer-type SM 전환 계열: OPT로 sync를 걷어내도 3–7배 격차가 남는다.
  - 이 셋은 "반증 실패"다. 단 위 실험 전에는 일반형 CONFIRMED가 아니다.
- 성분 측정(C2 2.36–2.91×, CM-5 s·k, CPU 0.014%, `s≤0.04 ms`)은 루프와 무관하거나 범위가 이미 명시돼 있어 유지한다.
- Phase 0의 수치(B1 상한, 영역 비율)는 이 감사가 범위 판단에만 썼다. Phase 0 자체의 결과 감사는 아니다(HE0 in-flight는 chunk 수 모형 상한이고, Zamba2·Granite span은 하한·η 차용이다).

## 5. 사용자 결정·promote에 넘길 것

- 정본 반영은 promote 1회에 한다. 1·2·3·6·10·11·13·14·18행은 문구 교체, 4·5·7·8·9·15행은 범위 구절 추가다. §6 등급 변경은 0.
- 2차 감사(마지막)는 이 판정 이후 문구 diff만 본다.

근거 파일: `src/multiplex/multiplexing_mixin.py`, `reports/CONSENSUS.md`, `COSTMODEL.md`, `reports/per_layer_type_postmortem.md`, `results/a_substrate/A_substrate_isolation_results.md`, `reports/policy_comparison.md`, `reports/system_vs_engine_vs_sim.md`, `results/impl_defects/{PHASE0_CPU_CHECK_2026-10-28.md, phase0_c3.md, phase0_c1_c2.md}`, `Papers/{Muxwise.pdf, Bullet.pdf}`.
