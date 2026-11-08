# MEMO — cap 48·mem 0.6은 어디서 오나, ReplaySSM B=256과 무엇이 다른가 (2026-11-07, main 세션, GPU 0)

> ★결과 감사 r1(2026-11-08) GO-with-edits — 판정서 `reports/audit/2026-11-08_unaudited_r1/AUDIT_cap_mem_memo_and_formula_review_r1_2026-11-08.md`, 수정 전 인용 금지 목록 적용. 본문은 고치지 않았다.

> 지위: 코드 읽기·기존 문서 대조·유도 계산. **새 측정 0, 미감사.** 수치 라벨: [코드] [측정·기존 문서] [유도]. 정본 인용 전 claims-auditor 1회.

## 1. cap 48은 엔진 상한이 아니다
- 최초 등장 `40d6869`(2026-07-04), 선택 근거 없음 [기존 문서: `handoff_review_2026-10-15/COSTMODEL_running_cap_2026-10-15.md` §1]. Nano cap 128 부팅·bs 96 고정 배치 측정 있음 [측정]. 128 초과 측정은 찾지 못함.
- SGLang 실제 제약 [코드, `sglang_engine_dev`]: radix OFF ∧ cap 지정이면 `max_mamba_cache_size = cap`(`model_runner_kv_cache_mixin.py:223-230`) · 슬롯은 부팅 때 정적 할당, 고갈 시 assert(`memory_pool.py:543`) · radix ON이면 cap = 슬롯/(3+α)(`:391-402`).
- 반면 vLLM V1(로컬 0.12.0, ReplaySSM fork)은 KV·Mamba state를 한 블록 풀에서 할당하고, 부족하면 대기·선점한다(`v1/core/sched/scheduler.py:574-613, 933-940`). A100 `max_num_seqs` 기본 256은 상한값이다(`engine/arg_utils.py`).
- SGLang의 "256"은 A100 `cuda_graph_max_bs` 기본값(`server_args.py:1196-1204`)이고, 동시 실행 상한이 아니다.

## 2. ReplaySSM B=256의 실체 [코드: github.com/Johnny-Liou/ReplaySSM `73a9592`, 논문 arXiv 2610.13778]
- 논문: 4모델 모두 **GPU 1장, TP 없음**(§7.1, 부록 E.1). Nano-4B는 H100 80 GB.
- `benchmarks/replayssm/e2e_decode_speedup.py` 기본값: 같은 짧은 프롬프트 1개를 256개 복제 → 한 번에 `llm.generate()` → `ignore_eos` 1000토큰 → ms/step. `max_model_len = 1256`, `gpu_memory_utilization 0.9`, prefix caching·chunked prefill OFF. 도착 과정·TTFT·ITL 꼬리·SLO 없음. 논문 그림별 실행 명령은 저장소에 없음.
- 추측 디코딩 스크립트: GSM8K, 출력 256 고정, `max_model_len 2048`, 기본 B 512, 고정 배치.
- ReplaySSM은 state **트래픽**을 줄이고 **용량**은 줄이지 않는다(입력 ring buffer 추가: Nano-4B B=256에서 약 1.5 GB [유도, 코드 shape]).
- 결론: 오프라인 고정 배치의 커널·처리량 측정점이며, 도착 부하·SLO에서 256 운영 근거가 아니다.

## 3. 메모리 산술 (Nano-9B-v2, A100 80 GB) [유도, `analytic_terms.py`: s = 143,548,416 B, k = 16,384 B/토큰]
- state 예약: cap 48 = 6.9 GB · 128 = 18.4 GB · 256 = 36.7 GB.
- mem 0.6 풀 ≈ 33.0 GB [측정, 부팅 로그] → cap 256은 state만으로 초과(문맥 무관).
- 가중치 17.8 + state 36.7 + KV(평균 실제 점유 1.3k) 5.3 ≈ 59.8 GB → mem 약 0.75 필요. 평균 2.6k면 0.82, 7k면 80 GB 초과.
- `--context-length` 설정값 자체는 거의 무관하다(`req_to_token` 256 × 20,484 × 4 B ≈ 21 MB, int32 가정, + flashinfer cudagraph `kv_indices`[크기 미산출]). 제약은 **실제 점유 토큰**(KV), **가장 긴 실제 프롬프트**(PD-mux chunk −1 활성화), decode KV 트래픽에서 온다.
- `b = 256`의 decode step state 트래픽 73.5 GB는 길이 무관이고, KV 읽기는 c 2k 8.6 GB · 8k 34.4 GB이다.

## 4. mem 0.6의 출처와 범위
- 결정 13(2026-10-20): 128k 프로브에서 0.82는 Nano PD-mux 단일 요청 64k가 활성화 OOM, 0.6에서 64k 통과(96k는 Triton causal_conv1d 경계, 메모리 아님) [측정: `probe_128k/RESULT_probe_128k_2026-10-20.md`]. 범위를 "모든 arm·블록(cap 블록·G0 포함)"으로 통일. 통일 이유는 결정문에 명시돼 있지 않음.
- **다수 요청 수용용 값이 아님**: mem을 낮추면 KV·state 풀이 줄고 활성화 여유가 는다(cap 48, ctx 131072: KV 약 272만 → 159만 토큰, 여유 11.72 → 약 29 GiB) [측정].
- 실험별: 2026-07 이후 KISTI 캠페인 대부분과 VESSL N2·N3·multigpu는 0.82(일부 0.80·0.75), 0.6은 2026-10-20 이후 cap_char 계열(비용 항·정책 하네스)과 STOCK-BASE PD-mux arm뿐 [스크립트 115개 정적 집계].
- cap 48에서 0.6은 KV를 구속하지 않는다: 48 × 20,480 = 983,040 < 정책 하네스 풀 1,586,264 토큰 [측정 + 산술]. mem 효과 자체는 측정된 적 없음.
- STOCK-BASE: SG-stock은 mem 자동, PM arm만 0.6 — mem 차이가 비교에 들어 있음(PREREG 구성 묶음으로 공시).

## 5. 열린 결정 (트랙 파일 "사용자 결정 대기")
결정 13 범위를 장문 블록으로 좁힐지. 권고안: long-ctx·봉인 실험은 0.6 유지, 새 cap 스윕은 cap 48에서 mem {0.6, 0.75} 대조 → 0.75에서 cap {48, 128, 256}, 부하 중 최대 메모리 기록. 고정 배치 T1을 cap 256까지 늘린 측정(ReplaySSM 대역폭 포화 구간)은 서빙 cap 스윕과 별개.
