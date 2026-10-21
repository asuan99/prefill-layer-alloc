# AUDIT — GATE_SPLIT_DESIGN_2026-10-20 하네스층 1회차 (claims-auditor, 2026-10-20) = **GO-with-caveats**

死因(N1–N4)은 0건이다. MAJOR 1·2·4·5는 제출 전에 등록 문장으로 고쳐야 한다(digest는 바뀌지 않는다). 고치지 않으면 4a는 N1(항등식) 경로로, 1d는 N2(반전) 경로로 남는다. 재감사는 1회 남았고 diff만 본다.

## 1. I-4 허용 범위 — MAJOR
- (a) 근거는 방향만 타당하다. 0.90과 0.75는 측정 없이 정한 [제안]이다.
- (b) 동기화 없는 host launch 변이(비율 약 0.05–0.15 [유도])만 걸러낸다. sync를 포함한 host 변이(약 0.95–1.0)는 [0.90, 1.0] 안에 들어와 걸러지지 않는다. lockstep에서는 stream이 비어 있으므로 cudagraph에서도 start event가 즉시 실행되고, I-4 구간에 launch 이전의 host 간극이 포함된다. 따라서 이 gate로 "I-4가 CM-1/CM-4를 분리한다"는 것은 입증되지 않는다.
- (c) green-ctx event timing을 G-α (c) 셀 1개로 보는 것은 불충분하다(0이 아님·포함 관계만 확인하고, 동시 prefill이 없다).
- (d) fused 창이 등록되지 않았다. `i4_fused_pairs`(engine_gate_checks.py:138-142)가 boot 전체에서 연속 forward를 짝지으므로 결과가 반전될 수 있다(PASS→FAIL).
- 수리: ① B4·B6 창을 "고립 decode 단계의 첫 decode forward부터 끝 decode forward까지"로 등록한다. ② β-0에 서버 없는 green-ctx 보정 단위 시험을 둔다. 2×2(host sleep 5 ms의 matmul 앞/뒤 × green/기본 stream)이고, 예측은 "앞은 포함, 뒤는 미포함, stream 간 차이 ≤2%"다. ③ 데이터를 본 뒤 허용 범위를 고치는 것을 금지하고, FAIL이면 사용자 결정으로 넘긴다.

## 2. λ\*\_thr seed 1개 — MINOR(분산) / MAJOR(대표성 공시)
- seed를 짝 안에서 공유하므로 log 비에서 상쇄된다. n≥8이면 수용 가능하다. 다만 측정량은 단일 seed의 HI achieved이며, SPEC/MG의 λ\*\_thr(4 seed 중앙값)와 다른 양이다. `KNEE_BRACKETED`가 HI 비포화 항등식을 막아 준다.
- cap 48·shape A는 iteration당 비용 면에서 보수적이지만 C\_mem cap, 장문, 거절 분기, overlap-ON은 검정하지 않는다. SPEC §9:319의 "판정 Job마다 ON/OFF 짝" 승계 문장이 설계 §1에 없으므로 추가한다.

## 3. ITL p95 짝 분산 미측정 — MAJOR
- λ 검정력 예보: 짝 SD 약 1.5% → 반폭은 n=10에서 약 0.87%, n=8에서 약 1.0%다. 통과하려면 평균이 약 0.5–0.6% 이내여야 하므로 경계 수준이다. ITL p95는 SD를 모른다. SD가 2% 이상이면 효과가 0이어도 통과할 수 없다.
- 지금은 PASS가 아니면 전부 `OBSERVER_EFFECT`로 표기한다(`tost_log_ratio` 258-272). 이 경우 분산 때문에 못 통과한 것도 효과로 잘못 표기된다.
- 수리: ① 3분 판정: CI⊂±δ면 PASS, CI가 0을 배제하면 OBSERVER\_EFFECT, CI가 0을 포함하면서 δ를 넘으면 UNRESOLVED\_POWER. ② 제출 전에 J1·anchor 아카이브에서 shape A LO의 boot 간 ITL p95 SD를 CPU로 재집계해 반폭을 예보하고 짝 수를 정한다. ③ 짝 추가는 등록된 1회 확장(예: n=16, α 분할)으로만 한다. A/A 짝은 선택이다.

## 4. 그 밖의 결함
- (a) **MAJOR**: OFF가 실제로 OFF인지 검증하지 않는다(N1 경로). 수리: OFF boot는 probe 파일이 없고 로그에 "decode probe ENABLED"·admission 설치 줄이 없음을 단언한다. ON boot는 반대를 단언한다. 서버 `sglang.__file__`을 기록한다.
- (b) MINOR: 되돌린 트리 `9af2f2ef…`는 디스크 바이트를 보장할 뿐 실행되었다는 증거는 없다. 수리: Triton 캐시 key를 기록한다. ≤64k 일치는 주소 산술상 연역되므로 "sanity"로 표기한다.
- (c) MINOR: 재부팅 실패나 메모리 미해제는 `HARNESS_UNRESOLVED`로 표기하고 FAIL로 세지 않는다. pristine 판정은 "사망 OR 패치본 출력과 불일치"로 한다.
- (d) **MAJOR**: ON 쪽에만 생긴 무효 짝을 제외하면 효과가 숨는다. 수리: 무효가 비대칭(ON 쪽이 2개 이상 많음)이면 OBSERVER\_EFFECT로 판정한다.
- (e) MINOR: Job마다 노드·드라이버가 다를 수 있다(confound 11). Job마다 `substrate.json`을 남기고, PASS가 다른 노드로 이전된다고 주장하지 않는다.
- (f) MINOR: G-γ에 Job 끝 digest 재확인을 추가한다.
- (g) MINOR: I-5 거절 사유와 I-6 `bs_real`를 대조한다(계단 bs 궤적, 49번째 요청은 cap 사유로 기록되어야 함).

## 반전 시험
| 자유 표면 | 민 범위 | 판정 변화 |
|---|---|---|
| I-4 하한 | 0.90→0.5 | 없음 |
| fused I-4 창 | 고립 decode → boot 전체 | **있음(PASS→FAIL)**, 창 등록으로 해소 |
| ITL 통계량 | 풀 p95 ↔ 요청별 p95 중앙값 | 미산출(caveat) |
| 무효 짝 제외 | 대칭 ↔ ON 편중 | 효과 은폐 가능 |

## 인용 금지 문장(결과 문서가 그대로 승계)
- "G-α (c)-I4 PASS는 I-4가 동기화 없는 host launch 시간이 아님만 보인다. I-4가 host 준비 간극을 제외한 GPU 커널 시간임은 보이지 않는다."
- "관측자 TOST PASS는 Nano·shape A·cap 48·mem 0.6·{agnostic, fused\_sym}·그 Job의 노드에 한정된다. 비용 모델 판정 Job의 Job 내 ON/OFF 짝(SPEC §9)을 면제하지 않는다."
- "gate PASS는 그 Job들의 디스크 바이트(digest)와 관측된 행동을 증명할 뿐, 이후 Job이 실행한 바이트를 보증하지 않는다. 이후 Job은 digest 확인과 방출 확인을 각자 반복한다."
- "λ 관측자 짝의 양은 단일 seed HI achieved이며 SPEC/MG의 λ\*\_thr와 같은 양이 아니다."

## 수리 우선순위
4a → 1d → 3①③ → 4d → 3② → 1② → 나머지 MINOR.
