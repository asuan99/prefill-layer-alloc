# RESULT — 128k 프로브 1회차 (Workspace bring-up, 2026-10-20, run `longctx_probe_20261020T022851Z`, commit `91ab310a`)

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
- PD-mux가 fused\_sym보다 먼저 OOM 나는 이유(split-prefill 경로의 SSD state passing)는 미확인이다.
- 레버 후보: `--mem-fraction-static`를 낮추면 KV 풀을 줄여 활성화 여유를 늘릴 수 있다. 모든 arm에 같은 값을 쓴다. C\_mem(124k)≈23은 KV 풀을 크게 줄여도 남을 가능성이 크다[유도]. 2회차 프로브로 확인해야 한다.
