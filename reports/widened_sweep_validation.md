# Widened-Sweep 검증 보고서 — batch/decode {1..512} 확장 재실행

작성일: 2026-06-18 · 대상: prefill-layer-alloc v2 (Zamba2-1.2B/2.7B, Falcon-H1-1.5B/3B · A100-SXM4-80GB)
용도: prefill batch·decode batch를 **{1,2,4,8,16,32,64,128,256,512}**로 확장 재실행한 E1/E2/E3/E5 결과가 v2 결론을 바꾸는지 판정. 검수 체크리스트의 *데이터-의존* 항목(A1·A5·B1·C2)을 widened 데이터로 닫는다.
데이터 출처: `results_v2/{e1,e2,e3,e5}/*_a100_sxm4_80gb.csv` (4모델 전부 완료; E5 falcon_h1_3b = job 772723, 2026-06-18 05:19 COMPLETED). 수치는 CSV 직접 집계.
관련: [v2_report](../workspace/characterization/reports/v2_report.md) · [closure](project_closure_report.md) · [검수](review_checklist.md) · [추가가치](additional_value_paths.md)

---

## TL;DR

1. **v2의 정성 결론은 전부 유지되고 b512까지 확장된다.** window는 db512에서 ~1.08×로 **거의 완전히 닫힘**(4모델 일관). scan_share steady floor, decode floor 포화, 모두 재현.
2. **헤드라인 1건 정정:** "Green Context가 단 한 셀도 못 이김(max 0.987/0.996)"은 widened 범위에서 **더 이상 글자 그대로는 참이 아니다** — 모델당 1셀(3/4 모델)에서 green_ctx가 two_stream을 **≤0.41%(≤0.002ms)** 앞선다. 전부 **db1–2 저배치 + 노이즈 수준**이라 *의미있는 승리는 여전히 0*이지만, 표현은 "노이즈 수준 동률 외 못 이김"으로 좁혀야 한다.
3. **새 실질 발견(A5):** green_ctx는 throughput뿐 아니라 **decode 지연에서도 진다** — decode_inflation이 two_stream 1–3% vs green_ctx **64–83%(최대 231%)**. 즉 우리 분할은 *양 축 모두에서 열등*. **단 이는 우리 `f`(prefill 우대 0.5/0.7)가 decode를 굶긴 결과**이며, **decode-보호형 분할은 테스트하지 않았다** → MuxWise/Bullet의 SLO regime은 여전히 미검증(§4).
4. **B1 해소:** `b512 chunk=full` CUDA illegal-access 실패는 status=failed로 격리(falcon 8/120, zamba 4/120), 평균/포화점 오염 0.
5. **G0/G1 불변.** 게이트 재판정 불필요(scan_share floor 안정; 비대칭은 §3.1로 이미 부차).

---

## 1. E5 — green_ctx vs two_stream (핵심)

### 1.1 A1 — `two_stream_ms / green_ctx_ms` (값<1 = two_stream이 빠름 = green_ctx 패)

| model | min(ts/gc) | **max(ts/gc)** | ≥1.0 셀 | crossover 셀 (green_ctx가 이긴 곳) |
|---|--:|--:|--:|---|
| falcon_h1_1.5b | 0.532 | **0.9954** | 0/40 | (없음) |
| falcon_h1_3b | 0.519 | **1.0019** | 1/40 | pf=ssm dec=ssm **db2** (gc 0.517 vs ts 0.518ms, +0.19%) |
| zamba2_1.2b | 0.526 | **1.0021** | 1/40 | pf=ssm dec=attn **db1** (gc 0.531 vs ts 0.532ms, +0.21%) |
| zamba2_2.7b | 0.539 | **1.0041** | 1/40 | pf=attn dec=ssm **db1** (gc 0.513 vs ts 0.515ms, +0.41%) |

**해석:** crossover는 전부 **db1–2(작은 워크로드, 양쪽 ~0.5ms, launch overhead 지배)**이고 차이가 **≤0.002ms = n_measure=20 노이즈 이하**다. 고배치에선 green_ctx가 최대 **1.9×**(min 0.52) 진다. → **green_ctx는 어디서도 *의미있게* 이기지 못한다**는 결론은 유지. 단 v2_report TL;DR §2 / closure §2 표의 *"단 한 셀도 못 이김 / max 0.987·0.996"* 문구는 **"노이즈 수준 동률(≤0.4%) 1셀 외 못 이김; 의미있는 승리 0"**으로 정정 필요(narrow 범위 수치라 그랬음).

### 1.2 C2 — overlap window (pf=ssm × dec=ssm, ctx4096), speedup_vs_seq

| db | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | **512** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| falcon_h1_1.5b | 2.04 | 2.04 | 2.00 | 2.01 | 1.95 | 1.89 | 1.61 | 1.35 | 1.18 | **1.09** |
| falcon_h1_3b | 2.02 | 2.00 | 2.03 | 1.99 | 1.96 | 1.89 | 1.54 | 1.31 | 1.16 | **1.08** |
| zamba2_1.2b | 2.01 | 2.04 | 2.01 | 1.96 | 1.88 | 1.74 | 1.53 | 1.30 | 1.16 | **1.08** |
| zamba2_2.7b | 2.05 | 2.01 | 2.00 | 1.97 | 1.86 | 1.72 | 1.55 | 1.32 | 1.16 | **1.08** |

**해석:** window는 db256(1.16)에서 멈추지 않고 **db512에서 ~1.08로 거의 완전히 닫힌다**(4모델 ±0.01). "이득은 decode batch가 GPU를 채우면 닫힌다"는 v2 결론을 b512까지 단조·일관 확장. db64 부근이 무릎(2×→1.5×대).

### 1.3 A5 — decode_inflation_pct (two_stream vs green_ctx) ★새 발견

| model | two_stream mean/max | **green_ctx mean/max** |
|---|--:|--:|
| falcon_h1_1.5b | 2.7 / 32.1 | **64.3 / 231.5** |
| falcon_h1_3b | 3.0 / 42.2 | **64.4 / 226.0** |
| zamba2_1.2b | 0.9 / 17.8 | **83.1 / 222.7** |
| zamba2_2.7b | 1.5 / 17.6 | **83.0 / 207.4** |

green_ctx decode_inflation은 **배치와 함께 증가**(zamba1.2b 예: db1=23% → db8=60 → db32=103 → db512=129%). green_ctx 설정은 `prefill_sm_frac ∈ {0.5,0.7}` → **decode_sm ∈ {54,32}**. decode floor가 배치와 함께 커지는데(§3) 고정 슬라이스가 그 아래라 **고배치에서 decode를 굶긴다**.

**해석 (MuxWise/Bullet 충돌, 검수 A5 ·closure §5.1):**
- **예상과 반대다.** "green_ctx가 decode를 *보호*해 SLO 이득을 줄 것"이라는 §5.1의 가설은 **이 데이터에선 성립하지 않는다** — 우리 green_ctx는 decode를 *보호*가 아니라 *starve*한다. 따라서 green_ctx는 throughput(§1.1)과 decode 지연(§1.3) **양 축 모두에서 two_stream에 진다.** 우리 음성은 더 강해진다.
- **그러나 결정적 단서:** 이건 **우리 `f`가 prefill을 편들었기(0.5/0.7)** 때문이다. **decode-보호형 `f`(decode_sm ≥ decode floor)는 스윕하지 않았다.** 즉 **MuxWise·Bullet이 쓰는 "decode를 보호하는 분할"은 여전히 미검증**이고, 이 데이터로 그들을 반박할 수 없다. (게다가 고배치에선 decode floor→94–108이라 decode를 보호하면 prefill이 굶으므로 — window가 닫힌 영역엔 어떤 `f`도 답이 없다. 보호형 분할이 의미를 가질 여지는 *중배치(floor<108, slack 존재)*뿐인데 거기를 보호형 `f`로 안 쟀다.)

---

## 2. E3 — decode floor by batch (자유 SM 소멸 확인)

`floor_sm_point` (max over context), SM∈{14..108}:

| model | layer | b1 | b8 | b16 | b32 | b64 | b128 | b256 | **b512** |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| falcon_h1_1.5b | attn | 68 | 68 | 68 | 68 | 108 | 108 | 108 | **108** |
| falcon_h1_1.5b | ssm | 14 | 27 | 27 | 68 | 68 | 68 | 68 | **68** |
| falcon_h1_3b | attn | 68 | 68 | 94 | 81 | 108 | 108 | 108 | **108** |
| falcon_h1_3b | ssm | 14 | 40 | 54 | 68 | 94 | 94 | 94 | **94** |
| zamba2_1.2b | attn | 54 | 68 | 108 | 108 | 108 | 108 | 108 | **108** |
| zamba2_1.2b | ssm | 14 | 68 | 81 | 81 | 94 | 94 | 94 | **94** |
| zamba2_2.7b | attn | 54 | 68 | 108 | 108 | 108 | 108 | 108 | **108** |
| zamba2_2.7b | ssm | 27 | 54 | 94 | 94 | 94 | 94 | 94 | **94** |

**해석:** attn-decode는 고배치(zamba b16+, falcon b64+)에서 **floor→108(전 SM 점유)** → 자유 SM(donor) 소멸. ssm-decode는 **68–94에서 plateau**(state BW 상수라 full 포화 안 함). 저배치엔 donor 존재(overlap 방), 고배치엔 소멸 → §1.2 window closure의 메커니즘적 원인. v2_report §4.4("b≥128 floor→108/94")를 b512까지 확인·정밀화.

---

## 3. E1 — scan_share by batch (G0 두-regime 확인) + B1

`scan_share` (max over chunk, ok 행만):

| model | b1 | b4 | b8 | b16 | b32 | b128 | **b512** | steady floor |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| falcon_h1_1.5b | 87 | 70 | 58 | 41 | 41 | 42 | **43** | ~41% |
| falcon_h1_3b | 81 | 60 | 46 | 33 | 30 | 32 | **32** | ~30% |
| zamba2_1.2b | 83 | 64 | 52 | 35 | 29 | 30 | **31** | ~29% |
| zamba2_2.7b | 79 | 58 | 41 | 27 | 20 | 21 | **22** | ~20% |

**해석:** peak(b1) → b16–32에서 **steady floor로 안정 후 b512까지 평탄.** v2_report §4.2의 "peak→elbow(b≈32)→steady floor" 두-regime이 확장 범위에서 그대로 재현(floor 40/30/28/20% 일치). G0=OK 불변.

**B1 (검수) 해소:** `b512 chunk=full`(및 일부 b256+) CUDA `illegal memory access` 실패 — **status=failed로 격리**(falcon **8/120**, zamba **4/120**). 위 scan_share·포화점은 ok 행만 집계 → **오염 0.** 극한배치 chunk=full에서의 kernel 한계이며 chunk{256,512} 경로·다른 배치엔 영향 없음(subprocess 격리 의도대로 동작).

---

## 4. 종합 판정 — v2 결론 대비

| 결론 | widened 결과 | 판정 |
|---|---|---|
| green_ctx가 동시실행 못 이김 | db512까지 의미있는 승리 0; 노이즈 동률(≤0.4%) 3셀 | **유지 (문구 정정)** |
| overlap은 decode batch로 닫힘 | db512에서 ~1.08× (거의 완전) | **유지·강화** |
| dec=ssm context-평탄 ~2× | (ctx 1k–16k 평탄성은 v2서 확인; 본 sweep은 ctx4096 고정) | 유지 |
| decode floor 고배치 포화 | attn→108, ssm→94 plateau, b512 | **유지·정밀화** |
| scan steady floor(20–40%) | b512까지 안정 | **유지** |
| (신규) green_ctx의 decode 지연 | two_stream 대비 64–83% 더 inflate | **분할이 양 축 열등 — 단 보호형 미검증** |

**바뀌는 것은 강조점뿐, 방향은 그대로다.** 추가로 widened 데이터는 두 가지를 *더 분명히* 한다:
- (a) 음성은 더 강하다 — green_ctx가 throughput·decode-latency 양 축 모두 짐(§1.3).
- (b) 그러나 **decode-보호형 분할(MuxWise/Bullet regime)은 우리가 *구조적으로* 안 쟀다** — `f`가 prefill 우대였고, slack이 있는 중배치를 보호형 `f`로 스윕하지 않음. 이게 남은 유일한 "분할이 이길 수도 있는" 미검증 구멍이며, 닫으려면 **decode_sm을 decode floor에 맞춘 `f` 스윕 + SLO(tail latency) 지표**가 필요(→ [Path 2 §5 / 추가가치](additional_value_paths.md)).

---

## 5. 다른 보고서에 반영할 사항

1. **closure §2 표 + §1 헤드라인 + 검수 A1:** "단 한 셀도 못 이김(0.987/0.996)" → **"db512까지 의미있는 승리 0; 노이즈 동률 ≤0.4% 3셀(db1–2)"**.
2. **검수 A5 (통과 조건 갱신):** decode_inflation 비교 완료 — green_ctx가 decode를 *보호 아닌 starve*(64–83%). **단 "분할은 decode도 못 지킨다"로 일반화 금지** — 우리 `f`가 prefill 우대였음을 명기, decode-보호형 `f`는 미검증으로 남김.
3. **closure §5.1 / Path 2:** A5 결과로 reconciliation 보강 — 우리 데이터는 *prefill-우대 정적 분할*이 양 축 열등임을 보였고, MuxWise/Bullet식 *decode-보호 분할*은 여전히 미답. hybrid 차별화(ssm-decode O(1) 지연) 질문 유효.
4. **검수 B1·C2:** 해소(B1: 실패 격리 확인 / C2: window db512=1.08 확장). 체크리스트에서 closed 표시 가능.
5. **G0/G1:** 재판정 불필요(floor 안정). 단 E2 widened CSV가 git에 미커밋이면 커밋 대상.

> **남은 작업:** E4 widened(현재 job 773614/773615 PENDING, concurrent A/B)은 본 보고서 범위 밖 — 완료되면 decode-interference 단면을 A5와 교차검증하는 짧은 addendum로 추가 권장.
