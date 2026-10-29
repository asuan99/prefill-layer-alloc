# PD-mux 구현 결함 B1–B4 — Phase 0 CPU 확인 통합 (2026-10-28)

> 작성: main 세션(통합) + result-analyst 2건(세부). CPU만, GPU 0, 엔진 수정 0, 정본·§6 등급 변경 0.
> 계획: 작업 루트 `PDMUX_IMPL_FIX_PLAN_2026-10-28.md`(제안). 코드 기준 `multiplexing_mixin.py` @ `3858f129`(HEAD `93b8b20a`).
> 세부: [phase0_c1_c2.md](phase0_c1_c2.md)(C1 HE0, C2 VESSL) · [phase0_c3.md](phase0_c3.md)(C3 span 영역).
> 라벨: [측정]/[유도]/[코드]/[가정]/[인용]. **감사 미실시**(claims-auditor 1회 대상).

## 0. 계획서 대비 코드 정정 [코드]

- **B2 정정**: decode-empty adjust drain(`:1358–1362`)은 **이미 발행된 span만** 기다린다. 아직 발행하지 않은 층은 drain 뒤 idx 0(108 SM)에서 돈다. 계획서의 "남은 prefill이 끝날 때까지 D SM 회수 불가"는 "발행돼 쌓인 span이 끝날 때까지"로 고친다.
- **in-flight drain 경로는 둘뿐**: B1(SLO/R2 전환, `:1341–1345`)과 B2(decode-empty, split_prefill_batch 존재). 입장·merge 직후 adjust drain은 prefill stream이 비어 있다(아래 C2 실측 ≤0.153 ms).
- **B1 발생 조건**: 컨트롤러는 `not wait_prefill_kernel_done`일 때만 평가된다 → 마지막 chunk 발행 뒤 WAIT 구간에서는 B1 불가. B1이 in-flight를 기다리려면 span ≥ 2 chunk.
- **switch_count ≠ drain 수**: 비sticky에서는 static·dynamic 모두 거의 매 span 입장마다 drain(gate truth span 50개에 drain 53건 [측정]). 결정 53 정의는 그대로 유효, "switch_count × 전환 비용" 모형은 틀린다.

## 1. 결과 요약

| ID | 질문 | 결과 | 판별 |
|---|---|---|---|
| C1 | HE0(KISTI, Zamba2-2.7B) 전환 중 prefill 진행 비율 | 서버 telemetry 0개, 서버 로그만. 전환 수/run: static 0, bind+GATE 1.56±1.67(n=9), no-gate 11.0±4.76(n=4), slo 15.0±2.0(n=4) [측정]. in-flight 가능 상한: bind+GATE 2/14, no-gate 5/44, slo 17/60 [유도, chunk 수 모형] | **상한만**. 전환 직후 decode 간격은 판별 불가(split_index·반복 시각 없음) |
| C1′ | B1이 HE0 부호를 바꿀 수 있나 | bind+GATE: 영향 받을 수 있는 HI 요청 arm 평균 ≤0.72% [유도] → 3% 문턱 미만. slo: run당 10–11% 상한 → 배제 불가. no-gate: 이상치로 상한 무정보 | B1은 **동적만** 벌하므로 고치면 동적이 나아질 수만 있다. HE0 핵심 비교(best static vs bind+GATE)에는 상한상 작음 |
| C2 | VESSL에서 in-flight drain | 실측 drain 이벤트(gate Job 71f5006) 53건 전부 빈 prefill stream(입장 23·merge 28·slo_switch 2), 최대 0.153 ms [측정]. span 중간 전환 0건(N3 2,063·N2 327·smoke 166) — 단 fixed 정책/anchor 고정 dyn뿐이라 **기회 없는 0** | B1 VESSL 미관측(구조적) |
| C2′ | B2 빈도·지속 | N3 Job3 span의 44.8%, Job4 45.8%에서 decode-empty가 prefill 중 발생, 지속 상한 p50 849–888 ms·p95 1.55 s [유도]. 그중 98%+는 모든 chunk가 이미 발행됨. N2-EA 2.9%(≤0.61 ms), 기타 0 | 빈도 [유도], 지속은 **상한**(양성 실측 0) |
| C3 | span 영역 | 전 캠페인 budget 65536 [코드]. HE0 배치 94–95% 미발화(span 1개) [측정]; P1-opint headline run은 사실상 전부 적체 [유도]; N3 98.1%·N2-EA 93.6% 적체 [측정: WAIT 반복 span당 중앙 49·28회]; T3 100% 적체 [유도]. **박자 영역(r<1)은 한 번도 측정되지 않음** | 진단 격자 12셀: 적체 8, 경계(판별 불가) 4(budget 8192 × D24/D44), 미발화 0. M1(2k×512×D44) = 박자 r≈0.32 [유도] |

## 2. 계획에 주는 함의 (제안)

1. **B2의 비용은 ITL·goodput에 ≈0, TTFT에만 있을 수 있다.** B2 동안 decode가 비어 ITL 표본이 없고, split prefill 진행 중이라 입장도 막혀 있다. 비용은 "쌓인 span이 P SM만 쓰고 D SM이 노는 시간"이다. 크기 상한(통합자 유도, [유도]·[가정]: span 시간 ∝ 1/P, η 불변): N3 D44에서 P=64→108이면 쌓인 구간이 최대 약 (1−64/108)≈41% 줄어 p50 상한 849 ms 기준 약 0.35 s/해당 span. **검증 안 된 상한**이며 실제 TTFT 이득은 진단 Job M6로만 판정.
2. **E3(handoff)는 B2를 고치지 못한다.** 이미 발행된 커널은 옛 green context에 묶여 있다. N3 B2 사례의 98%가 전 chunk 발행 후라 handoff로 옮길 남은 층이 없다. B2를 줄이는 것은 **E2(발행 깊이 제한)**: 깊이가 얕으면 decode가 빌 때 남은 층이 많아 idx 0으로 간다. → E2 우선순위 > E3(B2 기준). E3의 가치는 B1(동적 전환)에 한정.
3. **B1은 지금 데이터로 크기를 못 정한다.** 측정 경로 = 진단 PREREG의 동적 arm이 multi-chunk span에서 전환하는 구성 + PHASE_DIAG drain 이벤트(D-3 블록). B2는 같은 Job에서 양성 사례만 확인하면 된다.
4. **범위 문장(D2·D3·CM-4) 근거**: HE0는 거의 미발화 영역(층 단위 span 분할이 거의 일어나지 않는, prefill 전체를 한 번에 발행하는 영역 — 공간 동시 실행은 일어난다; 감사 r1 18행 정정 2026-10-29), P1-opint headline은 적체 영역 — 두 결과는 서로 다른 실행 영역에서 나왔다. CM-4 "동거 간격이 span 시간에 반응하지 않음"은 적체 영역 관측으로 범위를 좁혀야 한다. 정본 반영은 감사 뒤 promote.
5. **A3 정정**: "현재 격자는 전부 미발화·적체"는 문자 그대로 거짓(경계 4셀 판별 불가). M1 셀 추가 근거는 유지(유일한 박자 후보).

## 3. cost model 기여

- CM-4: 입장·merge drain ≤0.15 ms → 0으로 둔다 [측정]. drain 비용 항은 in-flight 전환(B1)과 decode-empty 시 쌓인 span(B2, TTFT 측)에만 존재.
- CM-2: Nano 8k에서 span 시간 ∝ 1/P(T2′·P 407k–414k ms·SM, η 0.43–0.46) [유도, phase0_c3 §6]. 새 무차원 항 r = 최장 span / (decode step × ρ)가 영역을 정한다. 기존 측정은 전부 r≫1 또는 span 1개.

## 4. 재현

```bash
python3 workspace/engine-port/results/impl_defects/c1_he0_switch_inflight.py
python3 workspace/engine-port/results/impl_defects/c2_vessl_adjust_inflight.py
python3 workspace/engine-port/results/impl_defects/c3_regimes.py --out workspace/engine-port/results/impl_defects/c3_regimes.json
```

## 5. 한계

- HE0 in-flight는 chunk 수 모형 상한(CTLCOST 줄이 남은 5 run에서만 간접 검증). KISTI·Zamba2 수치를 VESSL·Nano와 섞지 않는다.
- B2 지속은 상한이고, 추정기를 양성 실측으로 검증하지 못했다(실측 drain 이벤트에 in-flight 양성 0).
- T3 원자료(`/data/costterms`), decomp TEL(m12)은 로컬 부재 → 판별 불가.
- Zamba2·Granite span 시간은 정본 식 범위 밖이라 하한·η 차용 추정.
