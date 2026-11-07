# 결과 감사 r1 — Phase B (STOCK-BASE rev3.2, VESSL A100·Nemotron-Nano-9B-v2) · 2026-11-07 · claims-auditor

> 대상: `RESULT_phaseB_J1_2026-11-03.md`, `RESULT_phaseB_J1down_J1b_2026-11-04.md`, `DIAG_sglang_default_itl_2026-11-03.md`, `DIAGBOOT_itl_gap_2026-11-03.md`, `tracks/main.md:10`(Phase B 줄 — promote 원문 후보 요약).
> 결과 감사 1회차(하네스 감사 4회와 별개). 수치 재계산은 result-analyst(`AUDIT_result_phaseB_recompute_r1_2026-11-07.md`). 아래 수치는 결과 문서 인용 또는 raw `stats.json`·`argv.json`·코드 직접 확인. 읽기 전용 감사(판정문은 main 세션이 이 파일로 저장).

> **수치 쪽 대조(main 세션 병합 메모, 2026-11-07)**: result-analyst 재계산(`AUDIT_result_phaseB_recompute_r1_2026-11-07.md`)은 headline 수치 전부 ≤1% 일치. 단 A2의 "512–1024토큰 약 51 ms"는 재계산에서 41.7–44.8 ms(n=1–3)로 **재현되지 않았다** → promote 시 A2는 "≤1024토큰에서 상수항 약 42 ms(1 run 유도), 1024토큰 초과에서 길이 의존 정지(140–263 ms, run당 1–2건)"로 읽고 51 ms 수치는 옮기지 않는다. 그 밖에 재계산이 추가로 지적한 범위: "ITL SLO +10% → 0.93–0.95"는 seed 선택 범위(s202 0.483), B3은 chunk −1이 mem 자동값(0.836→0.686)을 함께 바꿔 단일 변수 아님, PD-mux 0.5× 지속 run의 실제 도착률 3.43(명목 −6%, 판정 불변).

## 0. 총평

결과 문서 자체는 대체로 정직하다(절벽·두 GPU 합침·n=2·격자 승계·Q8 비판정을 스스로 적었다). 문제는 트랙 줄의 압축 요약이 결과 문서의 금지 문안을 네 곳에서 되살린 것("Q8 첫 답", "CAP1-R 재현", "버스트의 0.5", piecewise OFF 범위가 빠진 "손대지 않은 SGLang")과, 문서 자체의 결함 세 가지(B5 교차 Job·GPU 비교를 DIAGBOOT가 "같은 GPU 안 비교"로 적음, "길이 무관" 과대, P-1·P-2 MISS 미기재)다. 종합 **GO-with-edits**.

### 독립 확인 사실

| # | 사실 | 근거 |
|---|---|---|
| F1 | `--enable-pdmux`는 다른 플래그를 **설정하지 않고** `chunked_prefill_size == -1`·`disable_overlap_schedule`·`pp_size == 1`·`disaggregation_mode == "null"`을 **assert**만 한다. 대신 코드 경로를 묶음으로 바꾼다: 이벤트 루프 `event_loop_pdmux`(`scheduler.py:3538`), green-context stream group·decode SM 분할(`init_pdmux`, env `PDMUX_R2_FIXED_DSM=44`), stream group별 decode attention backend·CUDA graph(`model_runner.py:1963`, `cuda_graph_runner.py:660-680`), split-prefill forward 모드(`scheduler.py:2746`), KV pool `enable_alt_stream` OFF(`model_runner_kv_cache_mixin.py:712,731`). | `sglang_engine_dev/python/sglang/srt/server_args.py:6125-6141` 등 |
| F2 | B5 argv = PM-ref argv에서 `--enable-pdmux`와 `--pdmux-config-path <yml>`만 제거. chunked −1·overlap OFF·mem 0.6·mamba 48·seed 1 유지, env(`PDMUX_ENGINE_MODE=pdmux`, `PDMUX_R2_POLICY=fixed`, `PDMUX_R2_FIXED_DSM=44`)·patched 트리 유지, server_args `enable_pdmux=False`. | `runs/pdmux-stock-baseline-diag-4d1b9da-20261103112203/raw/B5/argv.json`, server.log |
| F3 | 진단 Job GPU = `GPU-27097514`. 기준 행 "J1 PD-mux D44"·"J1 손대지 않은 SGLang"은 첫 번째 Job `GPU-bc0be109` 값. 진단 Job 안에 PD-mux ON boot 없음. DIAGBOOT 머리글 "같은 GPU 안 비교"는 B 변형끼리만 참. | `gpu.txt`, `stats.json` `gpu`, DIAGBOOT 표 1–2행 |
| F4 | B1·B3·B4·B5 전부 1 boot × seed 201 × rate 5.516 × 120요청(n=1). B5 run_id가 `J1_SG-stock_lad_r5.516_s201`로 찍혀 arm 라벨과 argv가 다름(재사용 시 오독 위험). | `stats.json` |
| F5 | upstream v0.5.10은 NemotronH piecewise CUDA graph **기본 ON**(`configs/model_config.py:1352-1358` 비활성 목록에 없음). 하네스 SG 공통 argv는 `--disable-piecewise-cuda-graph` 강제(`sb_spec.py:83`) + radix OFF·ctx 20480·max-prefill 16384·max-running 48. PREREG §2:83 공시됨. | 코드 |
| F6 | 같은 현상(SG 요청 ITL p95 ≈63–66 ms, 위쪽 모드 ≈64 ms)이 세 Job·GPU에서 관측: stock_quick(결과 감사 r1 PLAUSIBLE), J1(bc0be109), J1-down·diag(27097514). 현상의 존재는 기판을 넘어 재현. | CONSENSUS §10, 결과 문서 |
| F7 | 지속 평가 단위 정상: `sb_runner.py:392`가 `completion_s - sent_s` 사용, RECHECK 단위 오류 미반복. PM-ref 포화 완료율 5.15는 RECHECK 5.06–5.34와 정합(다른 Job). | 코드 |
| F8 | `DIAGBOOT_itl_gap_2026-11-03.md`는 24줄, §9·§10 **없음**. 트랙 줄 포인터 "`DIAGBOOT…` §9/§10"은 오류(결과 문서 §9·§10을 가리킨 것으로 보임). | 파일 |

## ① 도착당 ~42 ms 정지 → ITL p95 > 60 ms → 버스트 용량 3.15

| 부분 주장 | 판정 |
|---|---|
| (a) SG 기본 구성 토큰 ITL 이봉(기본 step ≈21 ms @5.5 req/s, 간극 ≈64 ms, 도착 1건당 약 1회) | **CONFIRMED(기술값, 범위 붙임)** — 3 Job·3 GPU 재현(F6), 클라이언트 합쳐짐 착시 배제(ITL < 3 ms 0%) |
| (b) 요청당 간극 비율 > 5%이면 요청 ITL p95 ≈ 간극 길이 → attainment 붕괴 | **CONFIRMED(산술, 술어의 기계적 귀결)** — result-analyst 재계산 조건 |
| (c) "~42 ms 정지, 길이 무관" | **OVERBROAD** |
| (d) 정지 원인(eager prefill 발행 CPU 비용, `prepare_mixed` `.item()` 동기화) | **NOT-YET-SUPPORTED** |
| (e) "ShareGPT 버스트 용량 3.15 req/s" | **OVERBROAD / 크기 인용 불가**(confound 5·11·3) |
| (f) 8월 PD-mux 우위와 관련됨 | **REFUTED**(그렇게 쓸 수 없음) |

- **metric cliff(confound 5, ITL 축)**: 현상은 진짜이나 크기가 SLO 선택에 불연속으로 매달림. 5.516 rung에서 ITL SLO 60→66→70 ms면 attainment 0.22→0.95/0.48→1.00/0.97(진단용, 재스코어 금지). 3.15 통과 seed 202는 SLO −5%에서 0.97→0.72. 3.15에서 간극 길이(57–59 ms)가 60 ms를, 간극 비율(4.1–5.1%)이 p95 허용 5%를 동시에 가로지르는 이중 절벽 → "3.15"는 SGLang 성질이 아니라 "간극 길이 × p95 술어 × 60 ms"의 교점. 반전 시험: SLO +17%(70 ms)면 SG가 5.516 통과 → PM-ref/SG 버스트 비 2.3× → ≤1.32×(크기 반전, 부호 유지).
- **piecewise CUDA graph OFF(confound 8 변형: 비기본 구성을 기본값으로 인용)**: 문서가 지목한 1순위 원천 "eager prefill 발행"은 바로 upstream 기본(piecewise ON)을 끈 결과(F5) → "손대지 않은 SGLang은 ~42 ms 정지"는 자기 기전 가설과 교락. B2 부팅 실패로 upstream 기본의 정지 길이는 미측정. 인용 금지 10이 공시했으나 트랙 헤드라인에서 빠짐.
- **길이 무관(OVERBROAD)**: 초과분 <32토큰 42.2, 128–512토큰 43–45, 512–1024토큰 51 ms; >1024토큰 간극 p50 142 ms, 1202토큰 예시는 156 ms 정지 추가. "짧은 프롬프트(≤512토큰)에서 거의 무관한 상수항"까지. 42 ms 자체도 run 1개(s201, 5.516)에서 "직전 0.5 s 기준 step 차감"으로 유도.
- **"정지 토큰 ≈10%"는 rate 의존**: 5.5 req/s 10–11%, 3.15 4–5%, 3.6–4.8 7–10%(도착당 1회 구조의 귀결) — rate 병기 필요.
- **기전 귀속(NOT-YET-SUPPORTED)**: DIAG(전역 정지 112건 중 101건에서 구간 안 새 요청 송신) · RESULT J1 §6-2(간극 안 다른 요청 첫 토큰 사건 29%뿐, "(i) prefill forward를 raw만으로 확정 불가") · 유휴 첫 요청 TTFT 50 ms(SG)·49 ms(PM)는 같은 고정비와 정합. 정합 후보는 "도착마다 생기는 host 쪽 고정 작업 ~40 ms"까지이며 eager 커널 발행·동기화 대기·기타 host 작업은 B2 실패·프로파일러 부재로 미분리. DIAG §2 "코드 읽기 추론, 미측정" 라벨 승계.
- **두 GPU 합침(confound 11)**: 3.15는 bc0be109의 ≥5.516 점과 27097514의 아래쪽 점을 한 pick 규칙으로 합친 값, 겹치는 rung 0 → SG 기판 이동 측정 불가(인용 금지 14).
- **small-n(confound 3)**: rung당 seed 2; 3.63·4.17에서 seed SD 0.15–0.30(rung 간 차와 같은 크기); 비단조(4.17 s202 0.975 > 3.63 s201 0.358); 아래쪽 rung은 첫 Job을 본 뒤 결정.
- **8월 PD-mux 우위(CONSENSUS §1-1)와의 관계: 쓸 수 없다** — 기판(KISTI 대 VESSL)·모델(Zamba2·Granite 대 Nano)·대조(8월 KISTI `plain` 수정 트리 OFF 대 pristine+piecewise OFF)·워크로드(P1-opint in2000/out96 대 ShareGPT)·정지 모양(§1-24 "최장 프롬프트 prefill과 겹치고 D에 단조" 대 "짧은 프롬프트에도 도착마다 상수항") 모두 다름. §1-1 "기전 귀속 미확립" 칸을 n=1·교차 GPU 진단으로 채우면 기판 혼합. §1-24 "VESSL·Nano 미관측"을 이 결과로 덮어쓰지 말고 다른 모양의 정지가 관측됐다는 사실을 §10형 공시로 따로 붙인다.

**허용 문안**
- A1. "VESSL A100·Nano·W1 ShareGPT에서, piecewise CUDA graph를 끈 SGLang v0.5.10 기본 구성(이미지 py3.14 제약; upstream 기본은 ON)의 토큰 ITL은 이봉이었다 — 기본 step 약 21 ms(5.5 req/s)와 요청 도착 1건당 약 1회 생기는 간극 약 64 ms. 이 관찰은 서로 다른 세 Job·GPU에서 재현됐다(기술값, rung당 n=2)."
- A2. "짧은 프롬프트(≤512토큰)에서 간극 길이는 '기본 step + 약 42 ms'로, 프롬프트 길이에 거의 무관한 상수항이 지배했다(1 run에서 유도; 512–1024토큰 약 51 ms, 1024토큰 초과에서는 길이 의존 정지가 추가). 상수항의 원인(eager prefill 발행·동기화 대기·기타 host 작업)은 측정되지 않았다."
- A3. "요청 내부 ITL p95 술어에서는 요청당 간극 비율이 5%를 넘으면 요청 ITL p95가 간극 길이와 같아지므로, 간극 길이 약 64 ms가 60 ms SLO 바로 위에 있다는 사실이 SG 계열 attainment를 결정한다(ITL 축 metric cliff)."
- A4. "SG 계열의 버스트 용량은 등록 격자 5.516 req/s 이상에 없었고, 사후 추가한 아래쪽 사다리의 3.15 rung만 통과했다. 이 통과는 ITL 축 절벽 위이고 두 GPU의 점을 합친 값이므로 용량 크기로 인용하지 않는다."

**인용 금지**: "손대지 않은 SGLang은 도착마다 42 ms 정지한다"(piecewise OFF·rate·길이 범위 누락) · "SGLang 기본 구성의 버스트 용량은 3.15 req/s다" / "PD-mux는 SGLang보다 2.3배 용량이 크다" · "정지 길이는 프롬프트 길이와 무관하다"(범위 없이) · "정지 원인은 eager prefill 커널 발행(또는 `prepare_mixed` `.item()`)이다" · "Phase B가 8월 PD-mux 우위(Granite·Zamba2·KISTI)의 기전을 밝혔다/뒷받침한다" · "정지 토큰 비율은 10%다"(rate 없이).

**해소 실험**: (1) upstream 기본(piecewise ON) 측정 — py3.12 이미지 또는 `typing.Union` shim(이미지 선택은 사용자 결정); 한 Job 안 SG-stock piecewise ON 대 OFF, ABBA boot ×2, seed 201–204, rate {3.15, 4.17, 5.516}, 지표 = 간극 모드 길이·비율. (2) 기전 측정(P1 계측 boot) — patched 트리에 SG-stock argv, `extend_host_ms`·`extend_gpu_ms` + 짧은 torch profiler 창, CPU 발행 대 동기화 대기 분리; 계측 boot 축약은 등록 부하 창으로 제한 + 변이 테스트(confound 13). (3) 크기가 필요하면 ITL 축 cliff 검사·SLO 감도표를 사전등록 보조 지표로, 같은 GPU 3.0–4.4 구간 1.05× 간격, rung당 seed ≥4.

## ② `--enable-pdmux` 제거로 정지 복원(B5), overlap·chunk·mem 무관, piecewise 부팅 불가(B2)

| 부분 주장 | 판정 |
|---|---|
| (a) B5: PD-mux argv에서 `--enable-pdmux`+config 제거 시 간극이 SLO 위로(p95 45→69, attainment 1.00→0.19) | **PLAUSIBLE(강함, 조건부)** |
| (b) "단일 변수" | **OVERBROAD** — argv 수준 단일, 코드 경로 수준 묶음 |
| (c) "overlap·chunk·mem 무관" | **OVERBROAD** |
| (d) 차이의 기전 = "prefill/decode 동시 실행" | **NOT-YET-SUPPORTED**(묶음에서 미분리) |
| (e) B2 piecewise 부팅 실패 | **CONFIRMED(관찰)**; 원인 "py3.14 환경" 귀속은 PLAUSIBLE(multigpu 2026-10-12와 같은 오류) |

- argv 수준: `--enable-pdmux`는 chunked −1·overlap OFF를 assert만 하고(F1) B5는 그 값들을 유지(F2) → PREREG 인용 금지 1의 묶음 플래그 6개는 B5 대비에서는 교락 아님. 단 n=1이라 인용 금지 1 해제 근거는 아님.
- 코드 경로 수준(confound 10 경로판): 플래그 하나가 (1) 이벤트 루프 순서(decode 먼저 발행 + `decode_stream.synchronize()`), (2) 두 번째 stream prefill(동시 실행), (3) green-context SM 분할(decode 44·prefill 64, 비sticky), (4) split-prefill forward, (5) stream group별 backend·graph + KV alt-stream OFF를 함께 바꿈. "동시 실행 때문"은 이 중 하나를 고른 귀속이며 1·3·4 배제 실험 없음.
- "정지 복원"이라는 말 자체가 부정확: PM-ref에도 간극 모드 있음(약 45 ms, 토큰 13–17%, 도착 대비 간극 수 1.11–1.52로 SG보다 많음). PD-mux는 정지를 없앤 것이 아니라 "step + ~42 ms"를 "~45 ms 상수"로 줄여 60 ms 아래로 옮김 → PM-ref 대 SG attainment 차도 절벽 횡단 산물(confound 5, PM 쪽 여유 25%).
- B5 간극 모양은 SG-stock과 동일하지 않음(60–65 ms 토큰 비율 B5 1.0% 대 B3·B4 4–5%; B5는 overlap OFF 모양 B1 1.1%·p95 67.7과 같음) → "SLO 위 간극이 돌아온다"로 읽음.
- n=1(F4). 결정적 대비(PM-ref ON 45 ms 대 B5 OFF 69 ms)는 Job·GPU를 넘음(F3, confound 11) → DIAGBOOT "같은 GPU 안 비교"는 이 대비에 대해 거짓. 완화 근거: 같은 GPU 27097514의 두 번째 Job PM-ref 기준 셀(7.29 req/s, rate 다름) 간극 p50 46.1–48.6 ms → GPU 차가 45→69 이동을 뒤집을 가능성은 낮음(이 효과 크기에서 반증 못 함). 아래 실험 전 CONFIRMED 아님.
- B3(chunked −1): trace 최대 입력 3126 < 8192라 chunking 미발화 → 처치 미발화 null, "이 trace에서 chunk 크기는 처치가 아니다"까지. B4(mem 0.6·mamba 48): KV 사용률 ≤0.01·mamba 실효 48 동일 → 구조적 null. B1(overlap OFF): 정지 미제거, p95 63.2→67.7 상승(교차 GPU, n=1) → "정지를 제거하지 않는다"가 맞고 "무관"은 과함.

**허용 문안**
- A5. "진단 boot(n=1, 5.516 req/s, seed 201)에서 PD-mux D44 구성의 argv·env·트리를 유지한 채 `--enable-pdmux`와 PD-mux config만 빼자 요청 ITL p95 중앙이 약 69 ms로 돌아오고 attainment가 0.19였다(첫 번째 Job의 PD-mux ON 45 ms·1.00은 다른 GPU). 같은 진단에서 overlap OFF·chunked prefill 끔·mem 0.6+mamba 48은 SLO 위 간극을 없애지 못했다(chunk·mem 두 변형은 이 trace·부하에서 처치가 발화하지 않는 조건)."
- A6. "PD-mux 구성에서도 도착당 간극은 남아 있으며(약 45 ms), PD-mux가 바꾼 것은 간극의 유무가 아니라 길이(SLO 60 ms의 위→아래)다. `--enable-pdmux`는 이벤트 루프·SM 분할·split-prefill·stream별 graph를 함께 바꾸므로, 이 차이를 '동시 실행' 하나로 귀속하지 않는다."
- A7. "이 이미지(py3.14)에서 SGLang v0.5.10은 NemotronH의 piecewise CUDA graph로 부팅되지 않았다(`typing.Union` AttributeError, 2회째 관찰). 따라서 이 캠페인의 SGLang 계열 prefill은 전부 eager다."

**인용 금지**: "`--enable-pdmux` 단일 변수가 정지를 없앤다" · "PD-mux의 prefill/decode 동시 실행이 정지를 제거한다/원인이다" · "chunked prefill·overlap·메모리 설정은 ITL 정지와 무관하다"(범위 없이) · "B5가 같은 GPU에서 PD-mux 효과를 확인했다".

**해소 실험**: (1) 한 Job 안 B5 재현 — PM-ref ON 대 B5 OFF ABBA boot ×2, seed 201–204, rate {3.15, 5.516}, 같은 GPU UUID, 기준 셀 포함, 지표 = 간극 모드 길이·비율·attainment·paired CI. (2) 묶음 분해 — PD-mux ON D ∈ {34, 44, 64, 92} 같은 Job(n=2 seed): 간극 ≈45 ms가 D 무관이면 "host 발행 상한" 지지, D에 따라 움직이면 SM 분할 성분. 루프 순서 대 동시 실행 분리는 엔진 수정(단일 stream + decode 우선 발행 루프) 필요 → engine-porter 실현가능성 검토 뒤(지금 실행 가능하다고 주장하지 않음).

## ③ 버스트 vLLM 8.39·PD-mux D44 7.29(CAP1-R 재현), 지속(600 s) PD-mux 3.65·vLLM 4.19 = 버스트의 0.5

| 부분 주장 | 판정 |
|---|---|
| (a) 봉인 규칙 출력값(8.389146 `BRACKETED`, 7.29491 `BRACKETED`, 지속 3.647455·4.194573) | **CONFIRMED(규칙 출력으로서; CPU 재현 바이트 일치)** |
| (b) "CAP1-R 재현" | **REFUTED(문구)** — 결과 문서 스스로 "격자 승계 때문, 기판 동일성 증거 아님"(J1 §2, 인용 금지 14) |
| (c) "지속 = 버스트의 0.5" | **OVERBROAD(격자 항등식에 가까움, N1 계열)** |
| (d) vLLM > PD-mux 순서 | **NOT-YET-SUPPORTED** |

- 정의 차: 버스트 = 120요청 run(부하 창 약 25–35 s)에서 SLO 술어(TTFT ≤ 3 s ∧ 요청 내부 ITL p95 ≤ 60 ms) attainment ≥ 0.90인 격자(1.15×) rung 오름차순 pick = SLO goodput 용량. 지속 = 600 s 도착, 120 s 창 5개, 각 창 E2E ≤ 60 s 완료 ≥ 0.95 ∧ 마지막 창 TTFT 중앙 ≤ 첫 창 2×, 격자 {1.0, 0.8, 0.5}×버스트 = ITL SLO 없는 안정성 경계(인용 금지 6). 비는 서로 다른 추정량의 비이며 {1.0, 0.8, 0.5, None}만 가능. 실제 경계는 PD-mux (3.65, 5.84), vLLM (4.19, 6.71). 포화 완료율/버스트로는 PM 약 0.71, vLLM 약 0.78(n=1) → "0.5"는 측정이 아니라 격자 하단. 사전 예측 "PD-mux 0.5"도 같은 arm 이전 데이터(RECHECK 5.06–5.34 < 5.84)에서 연역 → 적중 정보량 낮음.
- 다른 Job·기판 CAP1-R과 비교 불가(confound 11): 일치는 격자 승계, PM-ref 7.29는 TTFT 절벽 위(기준 셀 TTFT가 SLO ±5% 안 비율 5–7.5%, 두 번째 Job 끝 기준 셀 0.892 < pick 문턱 0.90 — 사다리였다면 λ\*는 한 rung 아래), CAP1-R 자체도 throttle 8.6–17.7% 아래 Job 상태에 묶임(CONSENSUS §10).
- vLLM 대 PD-mux 순서 불가: 버스트 차 = 격자 1 rung(15%), rung당 seed 2, vLLM 사다리 비단조(9.65 s201 0.792 → 11.09 s201 1.000), PM λ\*는 TTFT 절벽 위, 엔진 간 ITL 측정 동등성 미확인·부하 중 TTFT 오프셋 `TTFT_OFFSET_UNCHECKED`(인용 금지 3), vLLM boot 잡음 미측정(인용 금지 16), throttle 비대칭(첫 Job vLLM 12.3% 대 PM 9.2%, 두 번째 Job vLLM 23–24% 대 PM 15%). 지속 순서는 독립 정보 0(두 arm 모두 0.5× 점 → 버스트 순서 × 0.5 항등식). 독립 신호는 과부하 점 포화 완료 처리율(vLLM 6.5–6.6 대 PM 5.15–5.19, seed 1·boot 1·throttle 비대칭·PM 구성 묶음 인용 금지 1)뿐, 기술값으로만. 등록 Q4 NOT_MEASURED·`BOTH_SATURATED` 예상 공시, PREREG §9 "PD-mux가 vLLM을 이긴다/진다" 금지.

**허용 문안**
- A8. "같은 Job(GPU bc0be109)의 120요청 버스트 사다리에서 봉인 규칙 출력은 vLLM 0.31.0 8.389146, PD-mux D44 구성 묶음 7.29491 req/s였다(rung당 seed 2, 격자 간격 1.15배). PD-mux 값이 이전 VESSL 캠페인(CAP1-R)과 같은 것은 격자 승계 때문이며 기판 동일성 근거가 아니다."
- A9. "600 s 안정성 규칙(E2E ≤ 60 s 완료 ≥ 0.95 ∧ TTFT 중앙 증가 ≤ 2배; SLO goodput 용량 아님)에서 두 arm 모두 격자 {1.0, 0.8, 0.5}×버스트 중 0.5× 점만 통과했다(seed 1·boot 1). 실제 경계는 PD-mux (3.65, 5.84), vLLM (4.19, 6.71) req/s 구간 안이며 격자로 정해지지 않는다. 0.8× 과부하 점의 포화 완료 처리율은 PD-mux 묶음 약 5.15, vLLM 약 6.5 req/s였다(n=1, throttle 15–24%, 기술값)."

**인용 금지**: "CAP1-R을 재현했다" · "지속 용량은 버스트의 절반이다" · "vLLM이 PD-mux보다 용량이 크다/PD-mux가 vLLM에 진다"(버스트·지속·서비스율 어느 것으로든) · "지속 용량 3.65 req/s는 SLO 용량이다" · "서비스율 5.15"(과부하 점 n=1 포화 완료 처리율로만).

**해소 실험**: (1) Q4 직접 측정 — 포화 셀(50%·80%)이 아니라 두 arm 자기 용량 근처에서, 한 Job 안 PM-ref·PM-rep·VL-stock·VL-rep(vLLM 플라시보 신설) ABBA, 6.3–8.4 req/s 1.05× 간격, rung당 seed ≥4, 엔진 간 ITL 정의 일치 검사·부하 중 TTFT 오프셋 검사를 유효성 사슬에. (2) 지속 경계 분해 — 격자 {0.55, 0.6, 0.7}×, rate 순서 무작위, seed ≥2.

## ④ 패치 트리 OFF = 손대지 않은 SGLang(3.15 동일, Q8 첫 답)

**판정: REFUTED(문구)** — 허용은 "사다리 해상도에서 구별되지 않았다"까지.
- 정본 모순: PREREG §0 Q8은 판정 셀(나)의 goodput 부호 + 지연 공동 지표로만 답함. 결과 문서 두 편도 "Q8 판정 아님" 명시(J1 §10 제안 13, J1-down §9·§10(b)). "Q8 첫 답"은 금지 문안, Q8은 NOT_MEASURED.
- "동일"은 검정 없는 동등성 주장(confound 3): 3.15 일치는 pick 규칙이 같은 rung에서 멈춘 것뿐(격자 해상도 15%). 바로 위 rung 3.63 s201에서 SG-off 0.675 대 SG-stock 0.358(둘 다 절벽 위라 판정 근거 아님). n=2 seed, arm당 boot 1, boot 순서 고정(시간 추세와 겹침).
- 커널 수준 차이: 부록 D `causal_conv1d_triton.py` int64 캐스트 2곳은 prefill 경로에서 발화(인용 금지 4).
- 지지 관찰(기술값): server_args 차이는 `port`·`random_seed` 두 키뿐, 토큰 ITL·간극 분포 거의 겹침.
- `tracks/main.md` 27행 "측정으로는 손대지 않은 트리와 동일(3.15)"도 같은 이유로 수정.

**허용 문안**
- A10. "수정 트리·플래그 OFF와 손대지 않은 트리의 SGLang 기본 구성은 버스트 사다리(rung 간격 1.15배, rung당 seed 2, 두 GPU)에서 같은 rung(3.153791)에서 끊겼고 토큰 ITL 분포도 거의 겹쳤다. 이는 Q8 판정이 아니며, 15% 미만의 차이와 커널 수준 차이(causal_conv1d int64)를 배제하지 않는다."

**인용 금지**: "패치 트리 OFF = 손대지 않은 SGLang" · "Q8 첫 답 / Q8 EQUIV" · "기존 VESSL OFF 참조(M3 J1·CAP·TUNE·implfix)는 upstream과 같다".

**해소 실험**: 등록된 세 번째 Job의 Q8 지연 공동 지표(TPOT·TTFT p50·ITL p95, n=4, SG 플라시보 K3). 플라시보 BOOT_NOISE 확률이 높게 예보돼 있으므로 별도로 SG-off 대 SG-stock 단독 Job 가능 — ABBA ×2 boot, seed 201–204, 셀 1.82 req/s(포화, 지연 지표용)와 W3 1점, 1차 지표 TTFT p50(conv1d는 prefill에만 작용).

## ⑤ 사전 예측 대비 서술의 정직성

**판정: 대체로 정직(PLAUSIBLE), 한 곳 과소 보고(MISS 미기재).**
- P-1·P-2를 "score None, MISS로 옮길지는 감사·사용자 결정"으로 둔 것은 과소 보고. 예측 문서 §5 범주는 HIT(띠 안)·MISS·UNSCORABLE(포화·INVALID). `LADDER_BOTTOM_FAIL`은 포화도 INVALID도 아니고, 봉인 pick 규칙상 첫 rung 실패면 λ\* < 5.516 → [6, 10] 밖이 논리적으로 확정. P-2도 vLLM 8.389 / (SG < 5.516) > 1.52 → [0.8, 1.25] 밖 확정. 새 규칙이 아니라 등록 범주 정의의 적용이며, 스코어러가 None을 낸 것은 `LADDER_BOTTOM_FAIL` 분기가 빠진 스코어러 결함 → 감사 판정 **P-1 MISS, P-2 MISS**(사용자 결정 불요). 3.15(아래쪽 사다리)로 재채점하지 않는다는 rev3.1 §8 규칙은 지켜졌고 결론도 같음.
- 기술 예측 행 오류("stock ITL p95 SLO 안, 정지 노출 ~1%" → 반대)와 깨진 두 전제(노출 빈도 도착당 1회 대 상위 10–15%, 짧은 프롬프트 정지 약 64 대 약 13 ms)는 정직하게 명시됨.
- 보완: (1) 예측 문서 §0 헤드라인 "PD-mux가 goodput으로 stock을 이기지 못할 것"과 §1 기전("ShareGPT는 PD-mux에 가장 불리")이 사다리 수준에서 정반대로 나왔다는 한 줄(판정 셀이 아니므로 채점 아님, 전제 붕괴 기록). (2) 지속/버스트 PD-mux 0.5 "격자점 일치"에 "이전 데이터에서 연역한 예측이라 적중 정보량이 낮다" 병기. (3) 첫 Job 외삽 "통과 rate 약 2.4" 오류는 J1-down §10(b)가 이미 적음(양호).

**허용 문안**
- A11. "사전등록 예측 P-1(SG 버스트 λ\* ∈ [6, 10])과 P-2(vLLM/SG ∈ [0.8, 1.25])는 MISS다 — 첫 번째 Job 사다리에서 SG λ\*가 등록 격자 최저 5.516 미만으로 확정되어 두 띠 모두 밖이다(봉인 스코어러는 이 경우 None을 내는 결함이 있었다). 예측의 기전 전제(짧은 프롬프트 정지는 SLO 아래, 요청당 정지 노출 약 1%)가 raw와 어긋났다."

**인용 금지**: "P-1·P-2는 채점 불가(UNSCORABLE)" · "지속/버스트 0.5 예측이 적중했다".

## 문서 결함 (promote 전 수정)

| # | 위치 | 결함 | 수정 |
|---|---|---|---|
| E1 | `tracks/main.md:10` | "손대지 않은 SGLang ~42 ms 정지(길이 무관, 정지 토큰 ≈10%)" | A1·A2(piecewise OFF·≤512토큰·5.5 req/s 명시) |
| E2 | 같은 줄 | "`--enable-pdmux` 단일 변수 제거로 정지 복원" | A5·A6(n=1·교차 GPU·간극 단축이지 소거 아님) |
| E3 | 같은 줄 | "CAP1-R 재현" | 삭제, A8 |
| E4 | 같은 줄 | "= 버스트의 0.5", "서비스율 ≈5.15" | A9 |
| E5 | 같은 줄 + 27행 | "패치 트리 OFF = 손대지 않은 SGLang(3.15 동일, Q8 첫 답)", "측정으로는 손대지 않은 트리와 동일(3.15)" | A10 |
| E6 | 같은 줄 | 포인터 "`DIAGBOOT…` §9/§10" | 절 없음(F8) → 결과 문서 §9·§10 |
| E7 | `DIAGBOOT_itl_gap_2026-11-03.md` 머리글 | "같은 GPU 안 비교" | "B 변형끼리는 같은 GPU, 기준 두 행은 첫 번째 Job(GPU bc0be109)" |
| E8 | `DIAGBOOT` §읽기 | "B5가 가른다 … PD-mux의 동시 실행이 없을 때 생기는 것" | "`--enable-pdmux`가 켜는 경로 묶음이 없을 때" |
| E9 | 결과 문서 두 편 P-1·P-2 | `score: None`, 판단 보류 | 감사 판정 MISS 추가(⑤) |
| E10 | `DIAG` 표 §2 | 42 ms 초과분 출처 run 미표시 | "s201·5.516 1 run" 표시(result-analyst 재계산 대상) |

**caveat 누적 점검**: ①의 용량 크기 주장에 걸리는 금지 문장 10·11·14·N1·ITL 절벽 = 다섯 → 상한. SG 버스트 용량 값(3.15)은 정본에 수치로 올리지 않고 A4처럼 "격자 위 부재, 아래쪽 rung 통과, 크기 인용 금지"로만. Q4 caveat 6개 비승격(rev3 r2)은 불변.

## 종합: **GO-with-edits**

- 死因(규칙을 뒤집는 반전) 없음. 봉인 규칙 출력 재현, 서술 정직.
- 교정 = E1–E10(특히 트랙 요약이 금지 문안을 되살린 네 곳과 B5 교차 GPU 표기).
- 정책 결론 0건, §6 claim 등급 변경 근거 아님. CONSENSUS §1-1·§1-24·Claim G와 A1–A11 범위 안에서 모순 없음. §1-24 "VESSL·Nano 미관측"을 이 결과로 덮어쓰면 기판 혼합 → §10형 공시로 별도.

**promote 가능 문장(문자 그대로)**: A1(SG piecewise OFF 토큰 ITL 이봉, 3 Job·GPU 재현) · A2(≤512토큰 상수항 약 42 ms, 원인 미측정) · A3(p95 술어상 간극 길이가 attainment 결정, ITL 축 절벽) · A4(SG 버스트 용량 5.516 이상 격자에 없음, 3.15 통과 크기 인용 금지) · A5(B5 진단 n=1·교차 GPU, B1·B3·B4 null 범위) · A6(PD-mux는 간극 길이 단축, 기전 경로 묶음 미분리) · A7(py3.14 piecewise 부팅 불가, SGLang prefill 전부 eager) · A8(버스트 규칙 출력·CAP1-R 일치 = 격자 승계) · A9(지속 = 안정성 규칙 격자 하단 점, 포화 완료 처리율 기술값) · A10(SG-off와 SG-stock 사다리 해상도에서 구별 안 됨, Q8 아님) · A11(P-1·P-2 MISS).

**정본에 올리지 말 것**: SG 용량 3.15 수치, PM/SG 용량 비, vLLM 대 PD-mux 순서, "Q8", "CAP1-R 재현", "버스트의 0.5", 정지 원인 귀속, 8월 §1-1 기전 연결.

**감사 횟수**: 이 대상 결과 감사 1회차. 재감사(2회째가 상한)는 E1–E10 수정 diff만 본다.

**근거 파일**: `runs/pdmux-stock-baseline-diag-4d1b9da-20261103112203/raw/B5/argv.json`(F2), `…/raw/B*/stats.json`(F3·F4), `diagboot/diag_boot.py`(B5 구성), `sb_spec.py:83`(SG 공통 argv), `sglang_engine_dev/python/sglang/srt/server_args.py:637,6125-6141`, `configs/model_config.py:1352-1358`, `managers/scheduler.py:378,2746,3538`, `model_executor/model_runner.py:1963`, `model_executor/model_runner_kv_cache_mixin.py:712,731`, `tracks/main.md:10,27`.
