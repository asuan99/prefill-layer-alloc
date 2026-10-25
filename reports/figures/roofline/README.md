# roofline / 스케일링 그림 묶음 (2026-10-25, main 세션)

> **지위**: 정본 아님. 새 판정 0. 모든 수치는 `workspace/engine-port/scripts/costmodel/analytic_terms.py`(analytic-v1, 해석적 비용 수식 고정본)에서 나온 **[유도] 이상 하한**이거나, 이미 감사된 J1 FLOOR.json 관측과의 **대조**다. 측정은 새로 하지 않았다(GPU 0).
> **공통 가정**: A100 SXM 80GB(bf16 312 TFLOP/s, HBM 2.039 TB/s, ridge ≈153), Nemotron-Nano-9B-v2 차원, 바이트는 "각 텐서를 한 번 옮김"(이상적 하한이라 scan·conv·attention core의 intensity는 상한), 연산자마다 max(연산 시간, 메모리 시간)을 합산. 모델 전체 시간에는 LM head 가중치 읽기(약 0.58 ms)를 포함한다(2026-10-25 정정, 아래).

| 파일 | 무엇을 보는가 | 데이터 근거 |
|---|---|---|
| `roofline_attn_vs_mamba.py/.png` | 연산자별 roofline(prefill L=4k 채움 / decode b=32, ctx 8k 빈 마커), 층 intensity 대 L, FLOP 비중 | 수식만 |
| `roofline_decode_attn_vs_mamba.py/.png` | decode: 층 intensity 대 b, 대 ctx, 모델 전체 바이트 구성 | 수식만 |
| `roofline_prefill_decode_joint.py/.png` | prefill·decode를 한 roofline에 겹침(이상 roof라 y축 정보 없음 — 읽기 어려워 아래 R 그림으로 대체) | 수식만 |
| `bound_ratio_prefill_decode.py/.png` | R = 연산 시간 ÷ 메모리 시간(= intensity÷ridge)을 0 기준 좌우 막대로. **roofline보다 읽기 쉬움** | 수식만 |
| `scaling_batch_seqlen.py/.png` | 런타임 변수(L, b, c)에 대한 시간 스케일링, 조성별(a4~a31) 토큰당 시간, 히트맵(b, c) | 수식만 |
| `prefill_observed_vs_ideal.py/.png` | **관측 prefill floor 대 이상 하한**(약 2.06~2.28배, 상수항 약 7.2 ms) | J1 `FLOOR.json`(감사 CONFIRMED, 범위 L 1k–16k·무경합·Nano·108 SM) |
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
- **말하는 것(유도 + 일부 관측 대조)**: (1) Mamba2 층 FLOPs의 약 97%가 in/out_proj GEMM(in_proj 67%)이고 prefill 이상 시간의 약 92%가 Mamba GEMM+MLP(L=4k)라 "in_proj가 지배"는 맞지만 attention의 GEMM도 같은 영역이라 층 구분 근거가 못 된다. (2) prefill은 L≈180 토큰 이상이면 ridge 위, decode는 b=1024에서도 ridge의 1/5 미만(memory-bound). (3) decode에서 Mamba2 층은 b≈7 이상에서 attention 층보다 intensity가 높고(포화 약 28 대 약 8), state update(0.6 FLOP/B)가 가장 memory-bound. (4) prefill 관측 floor는 이상 하한의 약 2.06~2.28배(균일 η≈0.47~0.49), 상수항 약 7.2 ms(이상 약 0.6 ms). (5) J1 보정 η를 32k·64k에 외삽하면 보류 점이 ±10% 안(n=1, 기술값)이고 선형 외삽은 4/4로 과소예측 — L² 항 존재와 모순 없음.
- **말하지 않는 것**: SM 분할(D)별 η, 동거 간섭 ρ, 대기열·TTFT, cudagraph 패딩, 요청 이질성, 연산자별 효율 분해(서빙 내 프로파일러 불가), decode 측 관측 대조(T1 데이터 0건). 정본 `COSTMODEL.md`의 "아직 답하지 못한 것"이 이 빈칸을 그대로 갖는다.
- **관측과 충돌하는 이전 문장**: "decode는 memory-bound라 SM을 줘도 이득이 작을 수 있다"는 정본(C2 `CONFIRMED(scoped)`: decode ITL이 SM 16→92에서 2.36–2.91×, X-7 반증)과 충돌한다. 이상 roofline은 SM 민감도를 예측하지 못한다. 이 그림 묶음의 어떤 문장도 이 정본 판정을 약화시키는 데 쓰면 안 된다.

## 정정 이력
- 2026-10-25: 모델 전체 시간 계산에서 LM head 가중치 읽기(약 0.58 ms/forward)를 빠뜨렸던 것을 고쳤다. 영향: decode 모델 전체 시간 +0.57~0.96 ms(예: b=1,c=1k 7.72→8.29 ms), prefill 토큰당 +0.6 µs(L=1k). 결론·방향 불변, prefill 관측 대비 비율은 1% 안에서만 움직임.

## 인용 규칙
- 이 디렉터리의 수치는 `[유도]`로만 인용한다(식 + 관측 대조는 J1 floor 한정).
- 128k 프로브 TTFT는 정본이 기술값·인용 금지로 못 박은 값이다. `prefill_extrapolation`은 내부 점검용이며 논문·정본에 쓰지 않는다. 인용 가능한 형태로 만들려면 PREREG rev3.5 T2 격자(32k·64k, J-A/장문 boot)의 측정이 필요하다.
- 감사 이력 없음(claims-auditor 미호출). 승격(promote) 전에 감사 필요.
