# RESULT — 128k 프로브 1회차 (Workspace bring-up, 2026-10-20, run `longctx_probe_20261020T022851Z`, commit `91ab310a`)

> ★**감사(2026-10-20, claims-auditor 1회차 `AUDIT_probe_128k_2026-10-20.md`)**: 인용은 감사서의 허용 문안 1–7만. 아래 해석 중 "Triton SSD"는 "Triton causal\_conv1d"로 정정, PD-mux 선행 OOM의 split-prefill 귀속은 삭제(NOT-YET-SUPPORTED), 정확한 커널 경계 T ≥ 94,793.
> 지위: Workspace 프로브의 기계적 사실이다(운영 모델 §1-1). **TTFT 수치는 기술값이며 인용 금지**이고 Job 예산·설계 입력으로만 쓴다. 성능 판정은 0건이다.
> 원자료: `longctx_probe_20261020T022851Z/`(SHA256SUMS 70개 OK. `cache/`는 JIT 캐시라 git 제외). 편차 1건: libnuma1을 apt로 설치했다(`LONGCTX_PROBE_DEVIATION.txt`).
> 실행 시간: 02:28:53–02:40:05Z(11분). ★운영 사고: main 세션의 진행 폴링이 `pgrep -f`로 자기 자신의 ssh 명령을 잡아 "RUNNING"을 계속 보고했다. 그래서 프로브가 끝난 뒤 A100 Workspace가 약 60분 동안 유휴 상태로 켜져 있었다(03:40Z에 발견하고 pause). 다음부터는 완료 판정을 `SUMMARY.md` 존재 여부로 한다.

## 판정 항목
| ID | 결과 | 사실 |
|---|---|---|
| G1 | PASS | Nano PD-mux(agnostic)가 ctx 131072, cap 48로 부팅됐다. KV 풀 2,722,655 토큰(K·V 각 20.77 GiB), mamba state 6.46 GiB(48 슬롯), 활성화 여유 11.72 GiB [측정, 부팅 배너] |
| G2 | **FAIL** | PD-mux 단일 요청: 16k·32k OK, **64k에서 CUDA OOM**(2.50 GiB 할당 실패)으로 서버가 죽었다. 위치: `forward_split_prefill` → Mamba SSD `_state_passing_fwd`(`ssd_state_passing.py:210`) |
| G3 | FAIL | fused\_sym(chunk -1) 단일 요청: ≤64k OK, **96k OOM**(2.25 GiB) |
| G3-chunked | PASS | chunk 8192·mixed-chunk 대리: 16k–124k 모두 OK |
| G5 | FAIL | Llama-3.1-8B(fused\_sym 노브, chunk -1): ≤96k OK, **124k OOM**(3.39 GiB) |
| G4·G7·G5-conc | SKIPPED | 서버가 죽은 뒤라 실행하지 못했다 |
| G6 | UNRESOLVED | PD-mux 기준 L 2점뿐 |

## 기술값 (인용 금지, 출력 64 토큰, 단일 요청)
| L | Nano PD-mux TTFT | Nano fused\_sym | Nano chunked 8192 | Llama fused\_sym |
|---|---|---|---|---|
| 16k | 2.82 / 1.76 s | 1.80 | 10.76 (첫 요청, 워밍업 의심) | 1.28 / 1.28 |
| 32k | 3.96 / 3.70 | 4.36 | 3.81 | 3.26 / 3.24 |
| 64k | OOM | 8.54 | 8.67 | 9.44 / 8.90 |
| 96k | — | OOM | 13.25 | 18.16 / 17.43 |
| 124k | — | — | 18.23 | OOM |

- Nano의 prefill은 124k까지 대략 선형으로 보인다(chunked 기준 약 0.14 ms/토큰). Llama는 64k를 넘으면 초선형으로 보인다. 둘 다 [기술값, 1–2회]다.
- 측정 확인: Nano KV는 16,382 B/토큰, 슬롯은 143.5 MB/요청으로, 유도값(16 KiB, 136.9 MiB)과 맞는다.

## 의미 (해석, 감사 전)
- **PD-mux는 chunked prefill을 강제로 끈다**(assert). 그래서 장문 prefill 하나가 한 forward에 들어가고, **활성화 메모리가 길이에 비례해 커진다**. mem-fraction 0.82에서는 활성화 여유가 약 11.7 GiB뿐이고, KV 풀은 2.7M 토큰으로 필요량보다 훨씬 크다.
- 그 결과 지금 설정에서 **PD-mux arm은 64k부터 불가**, fused\_sym은 96k부터, Llama는 124k에서 불가다. 반대로 chunked prefill이 있는 fused 계열은 124k까지 된다. long-ctx 블록에서는 **arm 사이 대칭이 깨진다**.
- PD-mux가 fused\_sym보다 먼저 OOM 나는 이유는 미확인이다(감사 3번: 부팅 시 활성화 여유가 1.57 GB 적었던 것만으로 산술상 설명 가능, split-prefill 귀속 금지).
- 레버 후보: `--mem-fraction-static`를 낮추면 KV 풀을 줄여 활성화 여유를 늘릴 수 있다. 모든 arm에 같은 값을 쓴다. C\_mem(124k)≈23은 KV 풀을 크게 줄여도 남을 가능성이 크다[유도]. 2회차 프로브로 확인해야 한다.

---
# 2회차 — mem-fraction 0.6 (run `longctx_probe_20261020T034615Z`, commit `63de3582`, 03:46–04:01Z, SHA256SUMS 72 OK)
1회차는 네 arm 모두 `--mem-fraction-static 0.82`였다(각 `launch_command.txt`). 2회차는 네 arm 모두 0.6이다. 편차는 같다(libnuma1 apt).

| arm | 0.82 (1회차) | 0.6 (2회차) | 0.6에서의 실패 원인 |
|---|---|---|---|
| Nano PD-mux (chunk -1) | 32k OK, 64k OOM | **64k OK, 96k 실패** | **Triton `illegal memory access`**. 보고 지점은 `ssd_chunk_state.py:464 _chunk_cumsum_fwd`이나 원인은 causal\_conv1d(3회차 blocking 확인). 메모리 부족이 아님(활성화 여유 29.0 GiB) |
| Nano fused\_sym (chunk -1) | 64k OK, 96k OOM | 64k OK, **96k 실패** | 같은 커널, 같은 오류(여유 30.6 GiB) |
| Nano chunked 8192 | 124k까지 OK | 124k까지 OK | — |
| Llama fused\_sym (chunk -1) | 96k OK, 124k OOM | **124k까지 OK**, 동시 요청(G5-conc) 6/6 OK | — |

- KV 풀은 Nano 1.59M 토큰, Llama 263,900 토큰이다. C\_mem(124k)는 Nano 14, Llama 2다 [측정 부팅 로그 + 산식].
- **해석(감사 전)**: Nano의 96k 실패는 메모리가 아니라 **Mamba SSD Triton 커널이 chunk 없는 장문 단일 시퀀스에서 터지는 문제**로 보인다(int32 인덱스 넘침 같은 커널 한계로 추정, 미확인). chunked 8192는 forward 하나가 8192 토큰 이하라서 124k까지 된다.
  - **PD-mux는 chunked prefill을 강제로 끄므로**, 이 스택(SGLang v0.5.10 + 현 Triton causal\_conv1d)에서 chunk -1 Nano arm(PD-mux·fused\_sym 공통)의 실현 가능한 상한은 **64k**다(mem-fraction 0.6 기준).
- G4(장문 prefill 중 ITL), G7(합성 대 실제 문서)은 이번에도 서버가 96k에서 먼저 죽어 실행되지 못했다(단계 순서 문제).
- 기술값(인용 금지): Nano PD-mux TTFT 16k 1.8–2.2 s, 32k 3.7–4.6 s, 64k 7.8–8.4 s. Llama 124k 27.3–27.4 s. Nano chunked 124k 18.1 s.

---
# 3회차 — G4·G7 + 커널 임계 (run `longctx_probe_20261020T040836Z`, commit `88a73ce4`, 04:08–04:18Z, mem-fraction 0.6, `--only nano_pdmux,nano_fused`, SHA256SUMS 57 OK)
- **커널 임계 확인**: fused\_sym(`CUDA_LAUNCH_BLOCKING=1`)에서 L = 94,500은 OK, **95,100은 실패**했다. blocking 모드 traceback은 `mamba.py:509` → **`causal_conv1d_triton.py:510 causal_conv1d_fn`**을 가리킨다. 산술 예측 경계 94,786과 맞는다 → `ANALYSIS_triton_96k_failure_2026-10-20.md`의 가설을 확인했다(1회 관측).
- **G4** (PD-mux, 장문 prefill 1개 + 짧은 입력·출력 1024인 decode 4개, 기술값·인용 금지): 장문 prefill이 진행되는 창 안에서 decode token 간격은 32k에서 p50 25.6 ms·p95 30.3 ms·최대 1.13 s, 64k에서 p50 25.9 ms·p95 32.1 ms·최대 0.80 s다. 창 밖에서는 p50 약 14–15 ms다.
- **G7** (PD-mux, LongBench v1 원본 대 같은 길이 합성, ABBA, 기술값): 63,032 토큰에서 real 8.50(첫 요청)/7.49 s, synth 7.50/7.46 s. 53,284 토큰에서는 real 6.21/6.23 s, synth 6.23/6.23 s. 첫 요청을 빼면 차이가 거의 없다 → 결정 12′의 "내용 무관" 가정과 모순되지 않는다(n=2/arm).
