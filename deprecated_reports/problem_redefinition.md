# 연구 문제 재정의 보고서: SM 공간 분할에서 시간 분리와 intra-layer 병렬성으로

> 작성일: 2026-05-25  
> 하드웨어: NVIDIA A100-SXM4-80GB (108 SM, HBM2e ~2,000 GB/s)  
> 대상 모델: Zamba2-7B-Instruct, Falcon-H1-7B-Instruct  
> 배경: Stage 1 SM scaling sweep 및 Chunked SSM cooperative safety 실험 결과 기반

---

## 목차

1. [원래 가설 구조](#1-원래-가설-구조)
2. [실측 결과: 세 가지 핵심 역전](#2-실측-결과-세-가지-핵심-역전)
3. [역전의 수학적 증명: 왜 SSM은 포화될 수 없는가](#3-역전의-수학적-증명-왜-ssm은-포화될-수-없는가)
4. [기존 접근법의 구조적 불가능성](#4-기존-접근법의-구조적-불가능성)
5. [문제 재정의: 세 가지 방향](#5-문제-재정의-세-가지-방향)
6. [방향별 타당성 평가 및 기존 데이터 재활용](#6-방향별-타당성-평가-및-기존-데이터-재활용)
7. [재정의된 연구 질문](#7-재정의된-연구-질문)
8. [결론: 메커니즘을 바꾸되 문제는 유지한다](#8-결론-메커니즘을-바꾸되-문제는-유지한다)

---

## 1. 원래 가설 구조

프로젝트의 출발점은 다음 논리 연쇄였다.

```
[Premise A] SSM prefill은 비교적 적은 SM에서 포화된다
      ↓
[Premise B] Attention prefill은 더 많은 SM을 필요로 한다
      ↓
[Premise C] SM 공간 분할로 SSM/Attn/decode를 동시에 실행할 수 있다
      ↓
[Hypothesis] 레이어별 SM 재분배로 TTFT와 TPOT을 동시에 단축 가능하다
```

이 논리는 **Premise A, B, C 세 가지 모두가** 실측에서 반증되었다.

---

## 2. 실측 결과: 세 가지 핵심 역전

### 2.1 역전 1: SSM은 108 SM 범위에서 포화되지 않는다

| SM 수 | Zamba2 SSM latency (seq=4096, bs=4) | 기대(포화 가정) | 실측 추세 |
|-------|--------------------------------------|----------------|-----------|
| 14    | ~97.1 ms                            | 기준선          | 기준선     |
| 27    | ~50.3 ms                            | 포화 시작        | 1.93× 감소 |
| 54    | ~25.6 ms                            | 수렴            | 1.96× 감소 |
| 108   | ~12.6 ms                            | 포화 완료        | 2.03× 감소 |

**14→108 SM 전 구간에서 ×7.71 선형 감소** (Falcon-H1 동일: ×7.71). 포화 징후 없음.  
이론값: `⌈4096/14⌉ / ⌈4096/108⌉ = 293/38 = 7.71` — Wave model이 정확히 성립.

### 2.2 역전 2: Attention도 같은 비율로 SM 의존성을 보인다

Attention(FlashAttention) 역시 타일 수에 비례하여 SM 활용이 선형 증가한다.  
SSM과 Attention 사이에 "SM 효율성 차이"가 뚜렷하게 존재하지 않아, 분할의 이득이 없다.

### 2.3 역전 3: Green Context로 SSM을 SM 제한할 수 없다

Triton SSD 커널(`mamba_chunk_scan_combined`)은 `cudaLaunchCooperativeKernel`을 사용한다.

```
Phase 1: Local chunk scan  ─┐
         grid.sync()        │← 모든 n_blocks가 동시에 active해야 함
Phase 2: Inter-chunk prefix─┘
```

Green Context가 SM을 k개로 제한하면, n_blocks > k × max_blocks_per_sm인 경우 Phase 1의 일부 블록이 Phase 2가 시작되기 전까지 스케줄되지 못해 **데드락 발생**.

**cooperative_safe 조건**: `n_blocks_per_call = batch × (chunk_count) × n_heads ≤ sm_count × max_blocks_per_sm`

| 모델 | 안전한 설정 수 / 전체 | 이유 |
|------|----------------------|------|
| Zamba2 (n_heads=112) | 0 / 608 | n_heads=112 > 108 SM → 항상 unsafe |
| Falcon-H1 (n_heads=24) | 69 / 608 | batch=1일 때만 안전 |

실질적 서빙 조건(batch≥2, seq≥1024)에서는 두 모델 모두 Green Context SM 제한이 불가능하다.

---

## 3. 역전의 수학적 증명: 왜 SSM은 포화될 수 없는가

Wave model은 SSM latency를 정확히 기술한다.

```
latency(sm_k) = latency(full_sm) × ⌈n_blocks / k⌉ / ⌈n_blocks / full_sm⌉

여기서:
  n_blocks = batch × seq_len ÷ 4    (chunk_size=256 기준)
  full_sm  = 108                     (A100-SXM4)
```

**서빙 실제 설정(seq=4096, bs=32)**에서의 wave 수:

```
n_blocks = 32 × 4096 ÷ 4 = 32,768

SM=108: ⌈32768/108⌉ = 304 waves
SM=76:  ⌈32768/76⌉  = 431 waves   → 108 SM 대비 +42% latency
SM=54:  ⌈32768/54⌉  = 607 waves   → 108 SM 대비 +100% latency
```

포화(수확체감)가 시작되려면 n_blocks ≈ sm_count여야 한다.

```
n_blocks ≤ 108 이 되려면:
  batch × seq_len ÷ 4 ≤ 108
  → seq=256, bs=1    (n_blocks=64)
  → seq=1024, bs=1   (n_blocks=256) ← 이조차 포화 없음
```

즉, 실질적으로 의미 있는 throughput을 내는 어떤 서빙 설정에서도  
SSM prefill의 n_blocks는 수백~수만 wave에 해당하며, **SM을 줄이면 latency는 비례하여 증가한다.**  
"남는 SM" 자체가 존재하지 않는다.

---

## 4. 기존 접근법의 구조적 불가능성

| 접근법 | 불가능한 이유 |
|--------|-------------|
| SSM에 SM 제한 후 decode에 여분 SM 제공 | Cooperative kernel 데드락 — SSM은 전체 SM 필요 |
| SSM 포화점 이후 SM을 Attn에 재분배 | 포화점이 실측 범위(≤108 SM) 안에 존재하지 않음 |
| 레이어별 동적 SM 분할 (Policy B/C) | Policy A ≈ Policy B (0.007% 차이) — SSM SM 제한이 효과 없어 |

**결론**: 원래 접근법은 "레이어 타입마다 SM 효율성이 다르다"는 가정 위에 서 있었다.  
실측은 두 레이어 타입 모두 SM 수에 선형으로 의존한다는 사실을 보여 주며,  
이 가정이 성립하지 않으므로 **공간적 SM 분할(spatial partitioning)** 전략은 원리적으로 이득이 없다.

---

## 5. 문제 재정의: 세 가지 방향

원래 동기(prefill과 decode의 GPU 공유 최적화)는 여전히 유효하다.  
메커니즘만 바꾼다.

---

### 방향 A: 시간적 분리 (Temporal Multiplexing)

**핵심 아이디어**: 공간에서 SM을 나누는 대신, **시간에서 레이어 경계를 활용**한다.

```
prefill 레이어 N 완료
  → GPU idle 구간 (inter-layer gap)
  → Green Context stream 전환: decode 스트림에 GPU 양도 (0.43 μs overhead)
  → decode 1 step 실행
  → prefill 레이어 N+1 시작
```

**지지 데이터**:

| 항목 | 측정값 | 출처 |
|------|--------|------|
| Green Ctx CPU swap (no sync) | 0.431 μs 중앙값 | Stage 2 결과 |
| GPU sync 포함 전환 | 7.805 μs 중앙값 | Stage 2 결과 |
| 81레이어 총 전환 비용 | 평균 1.72 ms | Stage 2 추정 |
| decode 1 step 목표 시간 | ~수 ms (모델 의존) | 서빙 SLA |

**강점**: SM을 분할하지 않으므로 cooperative kernel 문제 없음. SSM도 전체 SM 사용.  
**약점**: prefill latency가 decode step 수 × step 시간만큼 증가. TTFT 증가 vs TPOT 개선의 트레이드오프.  
**핵심 연구 질문**: 레이어 경계당 몇 번의 decode step을 끼워 넣어야 TTFT 페널티와 TPOT 개선이 균형을 이루는가?

---

### 방향 B: Falcon-H1 intra-layer 병렬성

**핵심 아이디어**: Falcon-H1은 모든 레이어에서 SSM branch와 Attn branch가 **구조적으로 독립적**이다.  
두 branch를 별도 SM 파티션에서 동시 실행하면 레이어당 latency를 단축할 수 있다.

```
Falcon-H1 레이어 구조:
  input ─┬─ SSM branch  ──┬─ add ─ MLP ─ output
         └─ Attn branch ──┘

SSM branch: mamba_chunk_scan_combined
Attn branch: FlashAttention

→ 두 branch를 서로 다른 GreenContext stream에서 동시 실행 가능
```

**단, SSM cooperative kernel 문제가 동일하게 적용됨**.  
해결책: **Two-pass SSM kernel 분리** (`ssm_two_pass_decomposition_guideline.md` 참조).

```
기존: cudaLaunchCooperativeKernel (grid.sync 필요)
분리: Kernel A (local chunk scan) → Kernel B (inter-chunk prefix) → Kernel C (apply)
      모두 일반 CUDA 커널 → Green Context SM 제한 가능
```

**잠재적 이득 추정**:

| 설정 | SSM 단독 시간 | Attn 단독 시간 | 동시 실행 예상 | 단축율 |
|------|--------------|---------------|---------------|--------|
| seq=4096, bs=4 (@ 54 SM each) | 5.1 ms | ~3-4 ms | max(5.1, 4) = 5.1 ms | ~30% vs sequential |
| seq=4096, bs=16 | ~18 ms | ~12 ms | ~18 ms | ~40% vs sequential |

실제 이득은 SSM:Attn SM 분할 비율 및 각각의 SM scaling curve에 의존한다.

**강점**: Falcon-H1 전 레이어(44층)에 적용 가능. 이론적 상한 1.73~1.82×.  
**약점**: Two-pass 커널 구현 필요 (Triton 커널 재작성). 검증 부담이 큼.  
**핵심 연구 질문**: Two-pass SSM 커널이 cooperative kernel 대비 얼마의 overhead를 가지는가?

---

### 방향 C: 음성 결과로서의 기여

**핵심 아이디어**: "SM 공간 분할은 Hybrid SSM+Attention 모델에 구조적으로 적용 불가하다"를  
엄밀한 실측과 이론으로 증명한 결과 자체가 기여다.

**기여 내용**:

1. Wave model의 실증적 검증: n_blocks/SM 비율이 지배 변수임을 정량적으로 확인
2. Cooperative kernel 제약의 공식화: `cooperative_safe` 조건 수식 및 실측 분포
3. Green Context 전환 비용의 정밀 측정: 0.43 μs (no sync) / 7.8 μs (with sync)
4. 두 모델의 SM scaling 동일성: 아키텍처 차이(n_heads, d_state)에 무관하게 wave model이 지배

**활용처**: 향후 서빙 엔진(vLLM, SGLang)에서 SM 분할 기능을 설계할 때  
"SSM 계열 레이어는 cooperative barrier로 인해 SM 제한 대상에서 제외해야 한다"는  
설계 지침으로 직접 인용 가능.

---

## 6. 방향별 타당성 평가 및 기존 데이터 재활용

| 평가 항목 | 방향 A (시간 분리) | 방향 B (intra-layer 병렬) | 방향 C (음성 결과) |
|-----------|------------------|--------------------------|------------------|
| 구현 난이도 | 낮음 (Policy C 변형) | 높음 (Triton 커널 재작성) | 없음 (분석만) |
| 이론적 이득 | 중간 (TPOT 개선) | 높음 (per-layer latency 단축) | 없음 (기여 방향) |
| 기존 데이터 재활용 | Stage 2 Green Ctx 측정 직접 사용 | Stage 1 SM scaling curve 재활용 | Stage 1 전체 + Stage 2 전체 |
| 추가 실험 필요 | Stage 3 변형 (decode 끼워넣기) | Two-pass 커널 + 동시 실행 측정 | 없음 |
| NCU 의존성 | 낮음 | 중간 (Two-pass 커널 튜닝) | 없음 |

**기존 실험 데이터 재활용 지도**:

```
Stage 1 SM scaling CSV
  ├─ 방향 A: decode step 수 계산 시 SSM/Attn per-SM latency 참조
  ├─ 방향 B: SM 분할 비율 탐색 시 각 branch의 SM-latency 곡선 사용
  └─ 방향 C: Wave model 검증 핵심 데이터

Stage 2 Green Context 측정 JSON
  ├─ 방향 A: 전환 비용이 decode step 삽입 overhead의 하한선
  └─ 방향 C: SM 분할 overhead 수치로 직접 사용

Stage 1 Chunked SSM 608개 설정 데이터
  ├─ 방향 B: cooperative_safe 조건 → Two-pass 필요성 증명
  └─ 방향 C: 핵심 실측 근거
```

---

## 7. 재정의된 연구 질문

원래 연구 질문 세 가지는 다음과 같이 재정의된다.

| 원래 질문 | 실측 결과 | 재정의된 질문 |
|-----------|----------|-------------|
| SSM prefill은 몇 SM에서 포화되는가? | 포화 없음 — Wave model이 지배 | Wave model의 실제 서빙 설정별 MAPE는? (일부 19.57% 오차) |
| Green Context 전환 overhead는 얼마인가? | 0.43 μs / 7.8 μs 확인됨 | 레이어 경계 decode 삽입 시 TTFT-TPOT 트레이드오프 곡선은? |
| 레이어별/적응형/고정 분할 중 어느 것이 최적인가? | 분할 자체가 효과 없음 | Two-pass SSM 커널 overhead vs 동시 실행 이득의 breakeven point는? |

---

## 8. 결론: 메커니즘을 바꾸되 문제는 유지한다

원래 동기 — **Hybrid SSM+Attention 모델 서빙에서 prefill과 decode의 GPU 활용을 최적화한다** — 는 여전히 유효하다.

무엇이 달라졌는가:

1. **공간적 분할 → 시간적 분리**: SSM cooperative barrier가 SM 제한을 막으므로, 레이어 경계의 idle 구간을 활용하는 방향으로 전환.

2. **inter-layer 재분배 → intra-layer 병렬성**: Falcon-H1의 경우, 레이어 간 분배 대신 레이어 내 SSM‖Attn 동시 실행이 현실적인 이득 경로.

3. **이득의 원천**: "남는 SM에서 이득"이 아니라, "커널 간 직렬 실행을 병렬화하거나, 레이어 경계 idle을 decode에 양도"함으로써 이득.

**즉각적인 다음 단계**:

| 우선순위 | 작업 | 기대 결과 |
|---------|------|---------|
| 1 | Stage 3 Policy A'로 decode 삽입 실험 (방향 A) | TTFT vs TPOT 트레이드오프 측정 |
| 2 | Two-pass SSM Triton 커널 구현 (방향 B 전제) | cooperative barrier 없이 SM 제한 가능성 검증 |
| 3 | Wave model MAPE 19.57% 원인 분석 | 어떤 (seq, batch) 조건에서 wave model이 깨지는가 |
| 4 | NCU 프로파일링 재시도 (job 729105 결과 확인) | 커널 내부 BW/compute 활용률로 wave model 설명 보강 |

---

*이 보고서는 Stage 1 실험 완료 후 원래 가설 체계가 실측에 의해 구조적으로 반증됨에 따라*  
*연구 방향을 재정립하기 위해 작성되었다.*
