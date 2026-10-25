# roofline / 스케일링 그림 묶음 (2026-10-25, main 세션)

> **지위**: 정본 아님. 새 판정 0. FLOPs는 `workspace/engine-port/scripts/costmodel/analytic_terms.py`(analytic-v1)를 쓴다. **prefill 바이트 모델·연산자별 max() 규칙·PEAK/BW는 이 묶음 고유 확장이며 analytic-v1에 없다**(정본 수식으로 쓰려면 결정 27 변경 규칙). 수치는 **[유도] 이상 하한**이거나 이미 감사된 J1 FLOOR.json 관측과의 **대조**다. 측정은 새로 하지 않았다(GPU 0). **감사 r1 `AUDIT_r1_2026-10-25.md` 반영(2026-10-25)**.
> **공통 가정**: A100 SXM 80GB(bf16 312 TFLOP/s, HBM 2.039 TB/s, ridge ≈153), Nemotron-Nano-9B-v2 차원, 바이트는 "각 텐서를 한 번 옮김"(이상적 하한이라 scan·conv·attention core의 intensity는 상한), 연산자마다 max(연산 시간, 메모리 시간)을 합산. 모델 전체 시간에는 LM head 가중치 읽기(약 0.58 ms)를 포함한다(2026-10-25 정정, 아래).

| 파일 | 무엇을 보는가 | 데이터 근거 |
|---|---|---|
| `roofline_attn_vs_mamba.py/.png` | 연산자별 roofline(prefill L=4k 채움 / decode b=32, ctx 8k 빈 마커), 층 intensity 대 L, FLOP 비중 | 수식만 |
| `roofline_decode_attn_vs_mamba.py/.png` | decode: 층 intensity 대 b, 대 ctx, 모델 전체 바이트 구성 | 수식만 |
| `roofline_prefill_decode_joint.py/.png` | prefill·decode를 한 roofline에 겹침(이상 roof라 y축 정보 없음 — 읽기 어려워 아래 R 그림으로 대체) | 수식만 |
| `bound_ratio_prefill_decode.py/.png` | R = 연산 시간 ÷ 메모리 시간(= intensity÷ridge)을 0 기준 좌우 막대로. **roofline보다 읽기 쉬움** | 수식만 |
| `scaling_batch_seqlen.py/.png` | 런타임 변수(L, b, c)에 대한 시간 스케일링, 조성별(a4~a31) 토큰당 시간, 히트맵(b, c) | 수식만 |
| `prefill_observed_vs_ideal.py/.png` | **관측 prefill floor 대 이상 하한**(약 2.06~2.28배, 선형 절편 7.2 ms — 추정량 의존, 고정 비용·층별 η 아님[감사 r1]) | J1 `FLOOR.json`(감사 CONFIRMED, 범위 L 1k–16k·무경합·Nano·108 SM) |
| `prefill_extrapolation.py/.png` | J1로 보정한 두 예측기를 32k·64k 보류 점에 대 봄 | 보류 점 = 128k 프로브 TTFT — **기술값, 인용 금지**(내부 점검용) |
| `t1_compare.py/.png` | T1 decode 비교 틀(데이터 없으면 예측만, `--data cells.csv`로 측정 겹침) | 수식만(T1 측정은 아직 없음) |

## 재현
```bash
cd ~/Experiments/KISTI/prefill-layer-alloc/reports/figures/roofline
python3 scaling_batch_seqlen.py          # 다른 스크립트가 이 파일 앞부분(mk/t_ops)을 exec로 재사용한다
python3 prefill_observed_vs_ideal.py     # 필요: results/multigpu/manual_retrieval/pdmux-multigpu-2106f42-.../FLOOR.json
python3 prefill_extrapolation.py         # 필요: results/cap_char/probe_128k/ (원시 telemetry는 git 밖일 수 있음)
python3 t1_compare.py [--data cells.csv --out out.png]   # cells.csv: bs,ctx,t_s (S_eng 중앙값, 초)
```
`scaling_batch_seqlen.py`의 `exec` 재사용은 임시 방편이다(plot 코드가 분리돼 있지 않음). 정본화하려면 공용 모듈로 뽑는다.

## 이 묶음이 말하는 것 / 말하지 않는 것
- **말하는 것(유도 + 일부 관측 대조, 감사 r1 정정판)**:
  (1) Mamba2 층 FLOPs의 약 97%가 in/out_proj GEMM(in_proj 67%)이고 prefill 이상 시간의 약 92%가 Mamba GEMM+MLP(L=4k)다. 이 비중은 **FLOP·이상 시간 기준**이다. 격리 커널 실측(opchar, 인용 금지 범위)에서는 SSD scan이 in_proj보다 시간이 길다 — "in_proj가 시간을 지배한다"로 읽지 않는다. attention의 GEMM도 같은 영역이라 층 구분 근거가 못 된다.
  (2) prefill은 층 단위로 L≈180 이상에서 ridge 위다(이상 바이트 기준이라 실제 교차점은 이보다 길다; SSD scan 연산자는 모든 L에서 ridge 아래). decode의 attention·Mamba2 층(c=8k)은 b=1024에서도 ridge의 1/5 미만이지만, **MLP 층은 b≳150에서 ridge 위**이고 모델 전체는 b=1024에서 ridge의 약 0.26~0.36이다(b=1024는 운영 범위 밖). **이 분류는 108 SM 전체 기준이며 SM 수 민감도를 말하지 않는다(정본 C2: decode ITL SM16→92 2.36–2.91×).**
  (3) c=8k에서 Mamba2 층은 b≈7 이상에서 attention 층보다 intensity가 높다(포화 약 28 대 약 8). c≤1k에서는 모든 b에서 attention 층이 더 높아 교차가 없다. state update(0.6 FLOP/B)가 가장 낮은 intensity다.
  (4) 관측 prefill floor는 이상 하한의 2.06~2.28배다(J1 한 boot, 같은 Job 다른 boot ±0.4%). **이 비율을 층 종류에 공통인 η로 읽지 않는다.** 격리 커널 대조(opchar, 다른 GPU, 인용 금지 범위)에서 GEMM은 이상의 약 0.8, SSD scan은 약 0.05, 이상 모델에 없는 gated_norm이 관측의 약 9%를 차지한다. 선형 fit 절편(7.2 ms)은 추정량에 따라 7~18 ms로 바뀌고(정본 CM-2 ≈9 ms), 서빙 오버헤드가 없는 격리 합에서도 약 15 ms가 나오므로 고정 비용으로 해석하지 않는다. 이상 모델은 gated_norm·RMSNorm·relu² 패스·kv_write·임베딩·샘플링을 빠뜨려 이상 합이 약 4.7% 낮다(L=4k).
- **말하지 않는 것**: SM 분할(D)별 η, 동거 간섭 ρ, 대기열·TTFT, cudagraph 패딩, 요청 이질성, 연산자별 효율 분해(서빙 내 프로파일러 불가), T1 데이터 0건(J1 b=1 decode ITL 13.0~13.3 ms는 클라이언트 측 기술값으로만 존재, 이상 8.29 ms). 정본 `COSTMODEL.md`의 "아직 답하지 못한 것"이 이 빈칸을 그대로 갖는다.
- **관측과 충돌하는 이전 문장**: "decode는 memory-bound라 SM을 줘도 이득이 작을 수 있다"는 정본(C2 `CONFIRMED(scoped)`: decode ITL이 SM 16→92에서 2.36–2.91×, X-7 반증)과 충돌한다. 이상 roofline은 SM 민감도를 예측하지 못한다. 이 그림 묶음의 어떤 문장도 이 정본 판정을 약화시키는 데 쓰면 안 된다.

## 내부 점검 (인용 금지)
- (구 문장 5) J1 보정 균일-η 예측기는 32k·64k 프로브 점(기술값, n=1, 다른 GPU·Workspace·서버 인자 상이)에서 −8%~+11% 차이다(이전 판의 "±10% 안"은 틀림: 32k run1 +10.7%). 선형 외삽은 4/4 과소예측이나 오차가 32k→64k에서 줄어 L² 항 근거로 약하다. 보류 점의 run1·run2 차이(64k에서 9.6%)가 오차와 같은 크기라 예측기 검증이 아니다.

## 정정 이력
- 2026-10-25 감사 r1: 문장 (2)(3)(4) 조건 추가·"균일 η" 해석 철회, (5)를 내부 점검으로 이동·수치 정정, 지위 줄의 "모든 수치는 analytic_terms.py" 과장 수리. §8 η 첫 관측점 승격은 감사 권고 **불가**.
- 2026-10-25: 모델 전체 시간 계산에서 LM head 가중치 읽기(약 0.58 ms/forward)를 빠뜨렸던 것을 고쳤다. 영향: decode 모델 전체 시간 +0.57~0.96 ms(예: b=1,c=1k 7.72→8.29 ms), prefill 토큰당 +0.6 µs(L=1k). 결론·방향 불변, prefill 관측 대비 비율은 1% 안에서만 움직임.

## 인용 규칙
- 이 디렉터리의 수치는 `[유도]`로만 인용한다(식 + 관측 대조는 J1 floor 한정).
- 128k 프로브 TTFT는 정본이 기술값·인용 금지로 못 박은 값이다. `prefill_extrapolation`은 내부 점검용이며 논문·정본에 쓰지 않는다. 인용 가능한 형태로 만들려면 PREREG rev3.5 T2 격자(32k·64k, J-A/장문 boot)의 측정이 필요하다.
- 감사 r1(2026-10-25) 완료, r2는 이 정정 diff만 대상. 집계 η≈0.49는 층 종류별 η가 아니다 — 정본 §8에 올리지 않는다.
