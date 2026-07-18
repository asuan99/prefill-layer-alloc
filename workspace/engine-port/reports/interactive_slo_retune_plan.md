# 인터랙티브 SLO 컨트롤러 재튜닝 + P99-attainment 직접 재측정 — 구성 계획

작성: 2026-07-18 (측정 전 계획). 발의 맥락: [CONSENSUS.md](CONSENSUS.md) §1-16 + [serving_slo_survey.md](serving_slo_survey.md) §5-7(a).
정본은 CONSENSUS — **이 문서의 어떤 수치도 아직 측정된 것이 아니다.** 확정은 CONSENSUS에만 기록한다.

---

## 0. 한 줄 · 가설

§1-16는 **기존 3s 벤치 데이터를 tight SLO로 *재스코어*** 해서 "동적이 best-static과 대등~약우위"를 얻었다. 그러나 **컨트롤러 자체는 여전히 TTFT≤3s 가정으로 돌았다**(재스코어는 사후 채점일 뿐, 컨트롤러 행동은 안 바뀜). 조사로 인터랙티브 SLO(100–400ms)가 주류임이 확인됐으니, 이제 **컨트롤러를 그 목적함수로 재튜닝하고 직접 서빙으로 P99-attainment를 측정**한다.

- **HT (주가설)**: 인터랙티브 TTFT/TPOT를 목적함수로 넣으면 컨트롤러가 prefill 반응성에 제때 개입 → **재튜닝 동적이 best-static을 재스코어의 대등~약우위보다 *넓은* 격차로 이긴다.**
- **HT0 (귀무)**: 재튜닝해도 static 매칭 수준 — 컨트롤러 자유도가 이미 소진돼 목적함수만 바꿔선 안 됨. ⇒ 동적 트랙 종결 강화.
- **HT-neg (실현불가)**: tight SLO를 Zamba2가 물리적으로 못 attain해 신호 소멸(§5 블로커) → 측정 자체가 성립 안 함.

---

## 1. 현 컨트롤러의 "3s 가정"이 박힌 지점 (코드 근거)

`src/multiplex/multiplexing_mixin.py::_slo_decide_idx_binding` (L151–230). SLO가 관대하다는 전제가 **행동을 무디게** 만드는 곳:

| 위치 | 코드 | 3s 하에서의 행동 | tight(300ms) 하에서 필요한 행동 |
|---|---|---|---|
| L157·166 | `_ttft_slo=3000`; `_pf_slack=(3000-pf_age)/3000` | pf_age 수백 ms여도 **pf_slack≈0.9(안 급함)** → prefill 개입 거의 안 함 | pf_age 200ms면 pf_slack≈0.33 → **prefill urgency가 10× 예민** |
| L160·199 | `_pf_urg=0.5`; `pf_slack<_pf_urg`서 prefill-ward | slack 0.9라 임계 0.5 도달 드묾 | 임계 재보정 필요(상대값의 의미가 바뀜) |
| L125·132 | feasibility gate congestion: `bs≥cap×0.85`서 prefill-ward 거부 | **트랩(d16 붕괴) 방지 = 3s 관대 regime의 실패모드** 겨냥 | tight선 prefill-ward가 *유익*할 수 있음 → gate가 **과보수** 위험 |
| L158·204 | `_anchor=(lo+hi)//2`(=d34급) resting | 관대 regime 최적 resting | tight선 resting이 **더 prefill-ward**여야 할 수 있음 |
| L127 | feas ITL guard `_slo=TPOT_SLO=60` | chat 50/voice 30/code 25로 조여야 정합 | TPOT_SLO 동반 조정 |

★ 즉 **`PDMUX_TTFT_SLO_MS`를 낮추는 것만으로도 컨트롤러 행동이 실제로 바뀐다**(재스코어와 달리). 나머지 파라미터는 그 위에서 재보정 대상.

## 2. 튜닝 구성 — 2단계, 변수 하나씩 (§방법론 §3-8)

### Stage A — 목적함수만 인터랙티브로 (baseline 재튜닝)
컨트롤러·하네스 SLO를 **동일 인터랙티브 값**으로 맞추고 나머지 파라미터는 기본 유지. "목적함수 정렬만으로 얼마나 바뀌나".

| SLO 프로파일 | TTFT_SLO_MS | TPOT_SLO_MS | 근거(조사) |
|---|---|---|---|
| **chat** ★주 | 300 | 50 | 인터랙티브 주류·우리 교체 임계와 일치 |
| voice | 150 | 30 | 최tight 현실값 |
| code-panel | 300 | 50 | chat과 동일대 |
| (대조) batch | 3000 | 60 | 기존 = §1-16 재스코어 재현 확인용 |

### Stage B — Stage A 최선 프로파일 위에서 파라미터 1개씩 sweep
| env | 현재 | sweep | 가설 |
|---|---|---|---|
| `PDMUX_SLO_PF_URGENCY` | 0.5 | {0.3, 0.5, 0.7} | tight선 더 일찍 prefill 개입이 유익? |
| `PDMUX_SLO_ANCHOR_IDX` | 2(d34) | {1, 2, 3} | resting이 더 prefill-ward? |
| `PDMUX_SLO_FEAS_GATE` | on | {on, off} | tight선 gate가 유익한 prefill-ward를 막나? |
| `PDMUX_SLO_FEAS_OCC` | 0.85 | {0.85, 0.95} | 덜 보수적 gate |
| `PDMUX_SLO_DWELL` | 3 | 3 고정 | 변수 최소화(과거 dwell·urg 동시변경 = 해석불가 전례) |

**변수 1개 규율**: Stage B는 A의 승자 config를 기준선으로, 한 번에 한 env만.

## 3. 지표를 P99-attainment로 재설계 (하네스 수정)

현 하네스([`sharegpt_vary_bench.sbatch`](../results/slo_sched/sharegpt_vary_bench.sbatch)) 분석부에 `t<=3.0 and m<=0.06`이 **하드코딩**. 재작성 항목:

1. **SLO 파라미터화**: `t<=TTFT_SLO ∧ mean_itl<=TPOT_SLO`를 env(`BENCH_TTFT_SLO`,`BENCH_ITL_SLO`)로. **컨트롤러 SLO와 항상 동일값**(목적함수=평가함수 일치).
2. **P99-attainment 병기**: `attainment = #{TTFT≤SLO ∧ ITL≤SLO}/N` (P90/P95/P99 관행) + **P50/P90/P99 TTFT·ITL 절대값**. 실무 표준은 P99 percentile(§조사).
3. **capacity 곡선**: goodput(SLO 만족 req/s)뿐 아니라 **"attainment≥90%를 유지하는 max rate"** = 진짜 용량. rate sweep으로.
4. **길이 버킷별 attainment**: 짧은/긴 요청 분리(§5 블로커 진단용).
5. 라운드 duration **합산** 유지(`f921ae8` 3× 버그 방지). switch_count·split 체류분포 병기.

## 4. 측정 프로토콜

- **모델·운영점**: Zamba2-2.7B, **cudagraph ON**(운영점 유지), triton, mrr=48 — §1-16과 동일 substrate.
- ★**L−1 용량 선측정(§방법론 §3-6 필수)**: 각 SLO 프로파일에서 attainment 90% 유지하는 max rate를 먼저 구해 rate 대역 확정. tight SLO는 용량이 급감하므로 기존 rate 3↔12가 안 맞을 수 있음.
- **워크로드**: 변화 trace(rate LO↔HI 교대) 유지하되 **LO/HI를 L−1 용량 기준 재선택**. ShareGPT(p99 2776tok) 그대로.
- **비교군**: static sweep **d16–d54**(직접 측정, 재스코어 아님) + slo + **bind(재튜닝)** + **bind+GATE(재튜닝)** + bind(3s, 대조).
- **반복**: **n≥4**(baseline 분산 먼저). SLO 프로파일 3종 × 정책.

## 5. ★블로커·리스크 (미리 정직하게)

| 리스크 | 내용 | 완화 |
|---|---|---|
| ★**물리적 실현불가 (HT-neg)** | 물리 line 79ms+0.096ms/tok ⇒ **무경쟁에도** p99 2776tok=**345ms > chat 300ms**, 512tok=128ms > **code 100ms**. 긴 요청·code SLO는 애초에 못 맞춤 → attainment 바닥·신호 소멸 | **L−1 전에 무부하 attainment 체크**. code(100ms)는 제외 후보. chat(300ms)은 짧은 요청 위주라 성립 가능. 길이-normalized SLO 병용 옵션 |
| **용량 급감** | tight SLO선 SLO 만족 기준 용량이 3s보다 훨씬 낮음 | L−1 용량 선측정으로 rate 재선택 |
| **메트릭 절벽(§1-14 재발)** | attainment도 임계 지시함수 → 경계 rate서 증폭 | **percentile(P99 절대값) 병용**(threshold-free), 용량 아래서도 측정 |
| **cudagraph 격자 상한** | sglang `>7 그룹 IndexError` → split 격자 ≤7(d16–d54 5점은 OK) | 격자 5점 유지 |
| **gate ratchet 상호작용** | gate가 tight SLO서 다르게 작동(트랩 방지 목적이 tight선 무의미할 수 있음) | Stage B서 on/off 분리 측정 |

## 6. 판정 게이트

- **HT 확증**: 재튜닝 bind+GATE가 인터랙티브 SLO서 best-static을 **>1 pooled-σ 초과**(재스코어 대등~약우위를 *넘어섬*) ⇒ 목적함수 정렬이 실효 ⇒ **동적 제어에 인터랙티브 성능 이득 확정**.
- **HT 부분**: 재튜닝이 재스코어보다 낫지만 여전히 static 매칭 ⇒ 컨트롤러 자유도 한계, 그러나 §1-16(대등)은 유지.
- **HT0**: 재튜닝 무이득(≈3s 튜닝) ⇒ SLO 무관하게 컨트롤러가 static 못 넘음 ⇒ **트랙 종결 강화**(§1-16의 대등은 재스코어 아티팩트로 격하).
- **HT-neg**: 물리적으로 tight SLO attain 불가 ⇒ 측정 무의미, **long-context/모델교체 또는 length-normalized SLO로 재설계** 필요(→ [longcontext_trace_plan.md](longcontext_trace_plan.md)와 합류).

## 7. 산출물 · 순서

1. 하네스 수정 → `results/slo_sched/interactive_bench.sbatch`(SLO env 파라미터화 + attainment/percentile 분석부).
2. **L−1 무부하 실현가능성 + 용량 측정**(job 소수) — HT-neg 조기 차단.
3. Stage A(SLO 프로파일 3종 × 정책, n≥4).
4. Stage B(A 승자 위 파라미터 1개씩).
5. 판정 → CONSENSUS §1-16/§5-7 갱신, `results/slo_sched/interactive_slo_retune/`.

**다음 액션은 (1)+(2)** — 하네스 수정과 실현가능성 체크. 여기서 HT-neg면 이후 단계는 재설계된다. 사용자 승인 후 job 제출.
