# 규칙층 델타 감사 판정서 — PREREG_MULTIGPU_2026-10-12 rev4+rev5 (M2.5, 커밋 c71c5e4)

> 감사자: claims-auditor(읽기 전용, GPU 0), 2026-10-12. 판정문 요지를 메인 세션이 저장했다(수치·caveat·하네스 조건 문안은 원문 그대로).
> 직전 판정서 표·처방은 등록 상수로 쓰지 않고 재계산했으며 처방 P-D7·P-D8 자체도 검사했다(게이트 #110·#113).

## 0. 판정 — `GO-with-caveats` (M2g·M3 모두)
| 범위 | 등급 | 이유 |
|---|---|---|
| **M2g(§3)** | **`GO-with-caveats` 유지** | rev5는 G0에 fused_sym만 추가. §3-1에 fused_sym 행 없음(MGP-38). G5(c) (가)(나)가 D8 대체, (나) 기대값 정의는 하네스 H7 |
| **M3(§4)** | **`NO-GO` → `GO-with-caveats`** | D7·D8 해소, 새 死因 0. 반전 13개 표면 중 등록 판정을 수치로 뒤집은 것 없음. 큰 음의 효과는 G<1 값이 아니라 '미정의'로 빠짐(MGP-34·35) |
| M4 | 판정 안 함 | — |

이 등급은 "규칙을 실행해도 된다"는 뜻이지 "M3가 PD-mux 스케일링을 판정할 수 있다"는 뜻이 아니다. 하네스(H1–H10)는 별도 하네스층 감사, 제출은 사용자 승인.

## 1. 직전 지적 처리
- **D7 해소**: Q_c는 서버 로그 미사용, 포화 시 초과 도착은 첫 토큰 전 대기에 쌓여 Q_c가 x−λ 속도로 증가. 실측 fused cp8192 x/λ 2.11 → g 0.496. x/λ≈1.6 → g 0.39–0.40. 확장은 어떤 arm λ\* ≳ 1.42·prior일 때만 발화.
- **D8 해소**: memory.used는 기술치, (가)(나)는 캡처 크기 무관.
- MGP-28·31 반영 확인. 처방 실현 가능. 단 (i) Q_c는 실 `requests.jsonl`로 계산된 적 없음(MGP-33), (ii) `SATURATION_UNRESOLVED`(<20)는 clamp 하한 40이라 도달 불가(무해한 죽은 규칙).

## 2. 코드 사실
- trace_loadgen: `sent_s` = post 전 perf_counter(`:51,107`), `ttft_ms` = completion_tokens 증가한 첫 SSE chunk − sent(`:82-95,109`), 토큰 없으면 None, `error`는 첫 토큰 뒤 예외 포함, `sock_read=600` 하드코딩(`:44`), fd 한도 상향 없음.
- `stream_interval` 기본 1(`server_args.py:375`). overlap ON 1 step 지연은 상수라 Q3−Q1에서 상쇄.
- chunked 경계: `input_tokens <= rem_chunk_tokens`면 chunk 안 함(`schedule_policy.py:803`) → B 8192는 fused에서 chunk 1개. fused_sym(−1)은 prefill 배치가 max_prefill_tokens까지.
- PD-mux 강제 `chunked_prefill_size == -1`, `disable_overlap_schedule`(`server_args.py:6125-6140`). vanilla에서 두 플래그 함께 assert 없음.
- ★**세 번째 노브**: piecewise CUDA graph 최대 토큰이 기본으로 `chunked_prefill_size`를 따름(`server_args.py:1254-1259`), 캡처 크기 그 이하로 필터(`:1411-1413`), 비면 비활성(`model_runner.py:2486-2490`). NemotronH 비활성 목록에 없음(`model_config.py:1352-1358`). runner는 `--disable-piecewise-cuda-graph` 안 넘김(`engine_bench_runner.sh:103-105`) → fused(8192)는 piecewise 캡처 경로, fused_sym·PD-mux(−1)는 아님. KISTI fused cudagraph-ON 로그(`cgate_902397`)는 `--disable-piecewise-cuda-graph` 명시(`correctness_gate.sbatch:72`) → **piecewise ON NemotronH fused는 이 프로젝트에서 부팅된 적 없음**. G_sym 쪽 세 노브 대칭, S·G_def는 세 노브 합성(MGP-36).
- runner: boot 1회 trace 1개, health 360 s(등록 1800 s), vanilla 분기 추가 인자 경로 없음, `--attention-backend` 기본 triton은 NemotronH assert 거부(`server_args.py:1959-1962`).

## 3. 반전 시험 (원본 저장소 원자료 읽기 전용; `requests.jsonl` 0개라 bench_serving 출력 + seed 도착 재구성, 검증: 마지막 완료 vs 보고 duration 0.1–0.6 s 일치)
| # | 표면 | 변화 | 근거 |
|---|---|---|---|
| 1 | Q_c, fused prefill 우선 | 없음(D7 해소) | fused cp8192 2.11 → 0.496, cp2048 2.22 → 0.515, cp512 2.47 → 0.551, agnostic B 1.66 → 0.402/0.402, d44 1.58 → 0.404, λ0 A 1.64 → 0.390, E2 ON A 1.44 → 0.236–0.298. 유체값 ±0.05 |
| 2 | HI 0.10, 4 seed | 없음(MGP-26′) | N3 B x/λ 1.116 → 0.086/0.121/0.139/0.095, cp8192 1.09 → 0.098, cp2048 1.11 → 0.115, cp512 1.24 → 0.186 |
| 3 | LO 0.05 | 거짓 통과(MGP-27′) | N3 B ρ 0.55–0.88 최대 0.035, λ0 A ρ 0.6 → 0.000, W1 fused ρ 0.80 → 0.009–0.044, ρ≈1.0: B 0.028–0.066, A 5셀 −0.000–0.033 |
| 4 | x_lo를 fused HI에 묶음 × arm λ 격차 | G 미정의로 절단(MGP-34) | W1 fused 0.473 vs d44 0.319(0.68), d44 x 0.25(ρ 0.78) g 0.091 |
| 5 | CLIENT_ERROR 1% | 없음 | 약 100셀 오류 0 |
| 6 | `sock_read=600` | 큰 효과 절단(예보, MGP-35) | λ < 0.353·prior(확장 후 0.706), N=8 A 0.237(확장 후 0.257)이면 대기 > 600 s |
| 7 | Q_c arm 대칭 | 없음 | 상수 오프셋 상쇄 |
| 8 | DP8 엔진 전체 r | 없음 | n 큼 |
| 9 | B 작은 n | 없음 | n 53–77, ρ ≤ 0.88 g ≤ 0.035 |
| 10 | x_lo 기준 fused vs 분모 fused_sym | 없음(예보) | ρ = 0.6/S ≤ 0.88; S < 0.68이면 #4 절단 |
| 11 | estimand 교체 | 없음(데이터 전) | MGP-37 |
| 12 | 다중 비교 4 추세 | 수치 반전 불가 | 귀무에서 하나 이상 거짓 ≈ 19% → MGP-37 |
| 13 | 앵커 arm, §3-1 fused_sym | 규칙 미정의 | MGP-38 |

추가 예보: §6-1 "15–25 h"는 낮음. λ = prior 가정 ≈28–31 h, TP λ < prior면 ≈31–35 h → 확장 없이도 하드캡 30 h 초과 가능성 높음.

## 4. 死因: 0 (N1–N4 해당 없음)

## 5. caveat (문자 그대로 승계)
- **MGP-26′**: "g ≥ 0.10을 'x ≥ 1.11 λ\*'로 번역하지 않는다. rev4 Q_c 추정량으로 VESSL agnostic B x/λ\* = 1.116 네 셀의 g는 0.086–0.139였고(2/4가 0.10 미만), KISTI fused cp8192 x/λ\* = 1.09에서는 0.098이었다. `HI_SATURATED`는 x/λ\*의 수치 하한을 주지 않는다."
- **MGP-27′**: "`LO_UNSATURATED`(g ≤ 0.05)는 x_lo < λ\*를 보증하지 않는다. Q_c 추정량으로 x ≈ λ\*에서 g는 shape B 0.028–0.066, shape A 다섯 셀 모두 ≤ 0.033이었다. `KNEE_BRACKETED`를 '무릎이 x_lo와 x_hi 사이에 있다'로 쓰지 않는다."
- **MGP-33**: "rev4 Q_c 추정량은 trace_loadgen `requests.jsonl`로 계산된 적이 없다(저장소 전체 0개). 규칙층 반전 시험은 `bench_serving` 출력에 seed로 도착 시각을 재구성해 계산했다. M2g G3b의 실제 `requests.jsonl`에서 같은 함수가 g를 낸다는 것을 확인하기 전에는 'Q_c가 실데이터로 검증됐다'고 쓰지 않는다."
- **MGP-34**: "x_lo는 fused HI 중앙값에 묶여 있으므로, fused보다 λ\*가 약 30% 이상 낮은 arm은 LO에서 g > 0.05가 나와 `KNEE_NOT_BRACKETED`가 될 수 있다(KISTI W1: fused cp8192 0.473 vs d44 0.319 req/s, d44 ρ 0.78에서 g 0.091). 이 경우 G는 1 미만의 값이 아니라 미정의로 나타난다. 미정의 셀이 빠진 추세 라벨을 'G가 N에 따라 떨어지지 않는다'로 쓰지 않는다. 각 추세 라벨 옆에 미정의 (N, arm)의 수와 사유를 병기한다. LO 실패를 λ\*의 상한 근거로도 쓰지 않는다(위 d44는 ρ 0.78에서 실패했다)."
- **MGP-35**: "trace_loadgen은 첫 토큰 전 600 s 무응답이면 요청을 오류로 끊는다(`sock_read=600`). HI에서 arm λ\*가 대략 0.35·prior(확장 후 0.71·prior, N=8 shape A는 0.24·prior) 미만이면 그 arm은 `CLIENT_ERROR`로 빠진다. CLIENT_ERROR로 빠진 arm을 '차이 없음'이나 '정상 범위'의 근거로 쓰지 않는다. 그 (mode, N)의 G는 미정의다."
- **MGP-36**: "fused와 fused_sym·PD-mux arm의 차이는 overlap·chunked 두 노브만이 아니다. dev tree에서 piecewise CUDA graph 최대 토큰 수가 chunked_prefill_size를 따르므로(`server_args.py:1254-1259,1411-1413`), fused(8192)는 piecewise prefill graph 캡처 경로에 들어가고 −1인 arm은 들어가지 않는다(실제 캡처 여부는 셀 기록으로 확인). S = λ\*_fused_sym/λ\*_fused를 '스케줄 노브 효과'나 'overlap 효과'로 쓰지 않는다 — 세 노브의 합성이다. MGP-29의 'G_def를 SM 분할·PD 분리로 귀속하지 않는다'는 이 세 번째 노브에도 적용된다."
- **MGP-37**: "판정 문장은 G_sym 라벨만 쓴다. G_def·G_d44,sym 라벨이 G_sym과 방향이 다르거나 G_sym이 미정의인 (mode, N, shape)에서 보조 라벨을 판정 문장 자리에 쓰지 않는다. 네 (mode, shape) 추세 라벨은 다중성 보정 없이 각각 95% CI로 낸 것이므로, 그중 하나가 `G_INCREASES/DECREASES_WITH_N`이라는 사실을 'PD-mux 이득이 N에 따라 변한다'는 전역 문장으로 쓰지 않는다."
- **MGP-38**: "§4-5 블록 앵커는 fused·agnostic만 재므로 블록 분할 시 G_sym(1) 드리프트는 정의되지 않는다. 그 경우 `ANCHOR_DRIFT` 미발화를 'G_sym 드리프트 없음'으로 쓰지 않는다. §3-1에 fused_sym 기동 실패 행이 없다: fused_sym이 빠진 (mode, N)에서 G_def를 G_sym 자리에 대입하지 않으며, 그 (mode, N)의 G_sym은 미정의다."
- **MGP-39**: "G_sym은 같은 overlap·chunked·piecewise 조건의 vanilla 대비 PD-mux 기능 묶음(green-ctx SM 분할 + 동시 prefill/decode 실행 + PD-mux 입장·decode_bs 임계 선택 + SM group별 cudagraph)의 용량비다. fused_sym은 운영 구성이 아니므로(overlap OFF), G_sym > 1을 '운영 기본값 대비 PD-mux 이득'으로 쓰지 않는다. 그 문장은 G_def의 몫이고, G_def는 보조 라벨이다."
- MGP-0…MGP-32 유지. MGP-0의 "M3 `NO-GO`"는 rev5에 대해 **해제**(M3 `GO-with-caveats`).

## 6. 하네스층 조건 (봉인 전 필수)
- **H1** boot 1회에 shape A·B × seed 순차 루프 구현.
- **H2** health 대기 360 s → 등록 1800 s로.
- **H3** fused_sym 경로: vanilla 분기에 `--chunked-prefill-size -1 --disable-overlap-schedule` 바이트 동일, `server_args.txt` diff 시험.
- **H4** `PDMUX_ATTENTION_BACKEND=flashinfer`를 §1 고정 조건으로 등록·셀마다 기록(triton은 NemotronH boot 실패).
- **H5** `sock_read` 등록값 고정(바꾸면 MGP-35 문턱 재계산).
- **H6** 클라이언트 fd 한도: 피크 동시 연결 예보 N=4 A HI ≈1,410, N=8 A HI ≈1,880(확장 ≈3,130). `ulimit -n` 기록·상향.
- **H7** G5(c)(나) 기대 프로세스 수: 평가 시점 명시, boot 후 기대값은 M2g G0 실측 표를 M3가 대조, teardown 후 메모리 해제 polling 상한.
- **H8** `cell_meta`에 piecewise 설정·캡처 로그 기록, G5(c) "세 arm" → 네 arm, §4-4 CI 문장 G_sym용 수정.
- **H9** MGP-33 해소: M2g G3b `requests.jsonl`로 g 산출(1.5×prior에서 g ≈ 0.3 이상 기대), "error 있지만 ttft 있는 요청" 처리 규약.
- **H10** 클라이언트 용량: N=8 shape A(≈384 running × 512-token, ≈1,900 연결) 미검증 → VESSL 규모 M1.5 dummy 또는 MGP-34 병기 대상에 포함.

## 7. 처방 실현가능성 자기 검사
- 철회: "LO 실패 시 G 상한 라벨"(LO 실패가 x_lo > λ\*를 함의하지 않음).
- 선택(필수 아님, 사용자 결정): LO x_lo를 arm별 HI 중앙값 × 0.6으로 → MGP-34 절단 감소, 대가는 MGP-9 공동 x_lo 폐기 + 델타 감사 1회.
- H1–H10은 CPU 작업. H7·H9·H10은 M2g 산출물 입력 → M2g → M3 순서.

## 8. 다음 gate
1. rev6(선택): MGP-26′/27′, §3-1 fused_sym 행, §4-5 앵커 fused_sym, §1 attention backend·sock_read, §6-1 예산 재예보 — caveat·표 행 수준이면 경량 델타 감사.
2. 하네스 H1–H10 → 하네스층 감사.
3. M2g dry-run → 사용자 승인 → 제출.
4. M2g 산출물(G3b Q_c, G0 프로세스 수 표, piecewise 기록)이 M3 하네스 봉인 입력.
