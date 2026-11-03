# 진단: 손대지 않은 SGLang 기본값의 ITL p95 ≈ 63 ms (2026-11-03, engine-porter, 읽기 전용)

대상 Job: `runs/pdmux-stock-baseline-7899d1b-20261102085423/raw/` (A100-SXM4-80GB 1장, NemotronH Nano 9B v2 Base,
ShareGPT 120요청). 이 문서는 원인 진단만 다룬다. 성능 판정·goodput은 다루지 않는다. 새 GPU 실행 0.
계산은 raw `runs/J1_*_lad_r*_s20{1,2}.json`의 `token_itl_ms`·`token_t_mono`·`t_sent_mono`에서 직접 했다
(일회용 스크립트, 저장소 밖).

## 0. 결론 한 줄

**decode step은 60 ms가 아니라 약 21 ms다.** p95가 63 ms인 것은 정지 때문이다. 요청이 하나 들어올 때마다
독립 prefill step이 한 번 돌고, 그동안 진행 중인 모든 decode가 **decode step + 약 42 ms**만큼 멈춘다.
멈춘 토큰이 전체의 약 10%라서 p95는 정지 모드 위에 떨어진다. PD-mux가 decode 44 SM으로도 더 빠른 이유는
**prefill을 다른 stream에서 decode와 동시에 돌리기 때문**이다. 그래서 decode가 멈추는 길이가 "decode step + prefill"에서
"prefill 발행에 드는 CPU 시간(약 43 ms) 하나"로 줄어든다. 따라서 "PD-mux OFF와 stock이 같으니 원인은 argv/기본값"이라는
추론은 **절반만 맞다**. 트리가 원인이 아니라는 것은 맞다. 그러나 차이를 만드는 것은 기본값(chunk·mem·mamba·backend)이
아니라 `--enable-pdmux`가 만드는 **prefill/decode 동시 실행**이다. 이것은 아래 B5 boot로 확인해야 한다.

## 1. 정지인가, step 시간인가 (rate 5.516, 토큰 전체를 모은 분포)

| arm | seed | p10 | **p50** | p90 | **p95** | p99 | 요청 내부 p95 중앙 | 40 ms 넘는 토큰 비율 |
|---|---|---|---|---|---|---|---|---|
| SG-stock (pristine) | 201 | 15.1 | **21.2** | 53.6 | **63.9** | 99.9 | 63.2 | 10.3% |
| SG-stock | 202 | 17.5 | **21.4** | 40.1 | **66.5** | 112.3 | 66.1 | 10.1% |
| SG-off (패치 트리, 같은 argv) | 201 | 15.1 | 21.2 | 57.4 | 63.7 | 85.6 | 63.2 | — |
| PM-ref (D44, non-sticky) | 201 | 16.7 | **23.0** | 44.1 | **46.2** | 49.9 | 45.2 | — |
| PM-ref | 202 | 18.9 | 23.3 | 43.2 | 46.8 | 49.7 | 46.3 | — |
| VL-stock | 201 | 12.9 | 17.3 | 19.2 | 39.4 | 47.0 | 39.2 | — |

5 ms 단위 히스토그램(SG-stock s201)을 보면 분포가 **두 봉우리**다. 20–25 ms에 17,488토큰, 60–65 ms에 1,810토큰이 몰린다.
PM-ref는 20–25 ms와 40–50 ms 두 봉우리다. → **p50 ≪ p95, 즉 정지다.** step 시간 문제가 아니다.
p50만 보면 SG-stock(21.2)이 PM-ref(23.0)보다 오히려 빠르다.
p90이 seed마다 53.6과 40.1로 흔들리는 것은 정지 비율(~10%)이 p90 바로 위에 걸쳐 있기 때문이다.

## 2. 정지의 구조 (요청 도착 하나 = 정지 하나)

- 전역 정지(여러 요청이 8 ms 안에 동시에 끊기는 gap > 40 ms를 한 사건으로 묶음): SG-stock s201은 **112건**이고,
  그중 101건은 정지 구간 [시작−100 ms, 끝] 안에 새 요청이 송신됐다(trace의 `mixing_opportunities`=112와 같다).
- 정지 길이는 프롬프트 길이와 **거의 무관한 상수**다. 도착 요청의 입력 길이별 "동시 decode의 정지 gap − 직전 0.5 s 기준 step":

| 입력 토큰 | SG-stock 초과분 | PM-ref 초과분 | VL-stock 초과분 |
|---|---|---|---|
| < 32 (n=54) | **42.2 ms** | 22.9 ms | 22.3 ms |
| 32–128 | 42–43 | 24 | 25–27 |
| 128–512 | 43–45 | 23–25 | 29 |
| 512–1024 | 51 | 26 | 30 |

  ShareGPT 프롬프트 중앙은 34–39토큰이다. 이런 짧은 prefill의 GPU 연산량은 정지 길이를 설명하지 못한다.
- **SG-stock은 덧셈, PM-ref는 상수다.** 운영 초반 bs=1(decode step 12 ms)에서 SG-stock 정지는 53.6 ms(=12+42)이고,
  부하가 오른 뒤 step 21 ms에서는 63 ms(=21+42)다. 즉 정지 = decode step + 약 42 ms다. PM-ref 정지는 bs=1에서 45.1 ms,
  step 22 ms에서도 43–45 ms로 **step과 무관하게 일정**하다. 이는 정지 = max(decode step, prefill 발행 CPU 시간 약 43 ms)라는 뜻이다.
- 유휴 시 첫 요청(55토큰) TTFT는 SG-stock 50.5 ms, PM-ref 49.1 ms, vLLM 37.3 ms다. 짧은 prefill 한 번에 드는 고정 비용
  약 40 ms는 두 SGLang arm이 **같다**. 차이는 prefill 비용이 아니라 **그 비용이 decode와 직렬인지 병렬인지**에서 난다.
- 긴 프롬프트(1202토큰, s201 req2)에서는 SG-stock decode 토큰 시각이 송신 기준 7 → **61 → 217** ms로, 짧은 정지 하나 뒤에
  156 ms 정지(prefill GPU 실행)가 따른다. PM-ref는 같은 구간에서 10 → 53 → 73 → 94 → 114 …로, prefill이 GPU에서 도는 동안에도
  decode가 약 20 ms 간격으로 계속 나온다. 이것이 decode 44 SM이 이기는 이유다. prefill이 도는 동안 stock decode는
  **0 SM**(같은 stream에서 줄을 선다)이고, PM-ref decode는 44 SM을 받는다. 정지가 없는 구간에서는 non-sticky라서 108 SM으로 돌아간다
  (stream group 6, `PDMUX_R2_FIXED_DSM=44`). 그래서 p50이 거의 같다.

### 기전 (코드 읽기에서 나온 추론, 미측정)

- stock overlap 루프(`managers/scheduler.py` `event_loop_overlap`): 직전 decode D_k를 발행한 다음, 다음 반복에서
  `get_next_batch_to_run` → prefill P를 **같은 stream에** 발행하고 나서야 D_k 결과를 내보낸다. P는 eager로 돈다
  (prefill 로그 `cuda graph: False`, piecewise는 argv로 꺼짐). 56층 하이브리드 forward의 커널 발행이 CPU 쪽 고정 비용
  약 40 ms를 만든다는 해석이다. 덧셈 성분(+decode step)은 extend 경로의 호스트 동기화가 같은 stream의 D_k 완료를 기다리기
  때문으로 보인다. 후보는 `layers/attention/mamba/mamba2_metadata.py` `prepare_mixed`의
  `torch.any(has_initial_states[:num_prefills]).item()`(로컬 dev 트리 201행)이다. bring-up (e)에서 본 관측
  "2048토큰 extend_gpu_ms 347.8 > extend_host_ms 164.3, 선행 GPU 작업 대기 포함 가능성"과 방향이 같다.
- PD-mux 루프(`multiplex/multiplexing_mixin.py` `event_loop_pdmux`): decode(graph replay)를 decode stream에 먼저 발행하고,
  prefill은 prefill stream에 발행한다. 그 다음 `decode_stream.synchronize()` 뒤에 decode 결과를 처리한다. prefill 쪽 호스트 동기화는
  prefill stream만 기다린다. 그래서 decode GPU 시간이 prefill 발행 CPU 시간 안에 숨는다. budget 65536 / 입력 ≤1170토큰이면
  `forward_count` ≥ 56층이라서, ShareGPT 프롬프트는 대부분 span 하나로 끝난다.
- 미확정: 약 40 ms 고정 비용 가운데 CPU 발행 시간과 GPU 실행 시간이 각각 얼마인지, 그리고 동기화 지점이 정확히 어디인지는
  프로파일러로 재지 않았다.

## 3. 후보별 판정

| # | 후보 | 판정 | 근거 |
|---|---|---|---|
| 1 | overlap ON (기본) | **부분 기여 가능, 미분리** | 정지가 decode step에 더해지는 직렬 구조는 같은 stream 위 overlap 루프에서 나온다. 그러나 overlap OFF + 단일 stream이어도 decode 다음 prefill이 직렬로 돌기는 마찬가지라 정지가 사라질 것이라 예측할 근거는 없다. PM-ref와 SG-stock의 차이에는 overlap OFF와 PD-mux ON이 섞여 있다(B1·B5로 분리). |
| 2 | chunked prefill 8192 | **chunk 크기는 배제**, 독립 prefill step이 끼는 구조 자체가 **주원인** | 최대 입력 2579/3126토큰으로 8192를 넘는 요청이 0이라 chunking은 발화하지 않는다. 원인은 "도착마다 독립 prefill step이 decode를 막는 구조"이고, 그 정지 길이는 프롬프트 길이에 비례하지 않는 상수 약 42 ms다(PREDICTION 문서의 "40토큰 ≈ 13 ms, 정지 아님" 가정과 다르다). |
| 3 | mem-fraction·mamba cache | **배제** | mamba cache 실효값이 둘 다 48이다(로그 `max_mamba_cache_size: 48`, stock은 자동으로 max_running=48). KV pool은 2.81M 토큰 대 1.59M 토큰인데 사용률이 ≤0.01이라 차이가 나지 않는다. retract 0건(두 로그). |
| 4 | cudagraph 범위 | **차이 원인 아님** | 두 arm의 캡처 bs가 [1…48]로 같고, decode 로그는 `cuda graph: True`다. 다만 prefill이 graph 밖(eager)이라는 점은 공통 고정 비용 약 40 ms의 유력 원천이다. 차이는 아니지만 수준은 여기서 정해진다(B2). |
| 5 | backend | **배제** | 실효값이 둘 다 attention `flashinfer`, mamba `triton`, sampling `flashinfer`다(server_args 349키 diff). |
| 6 | 클라이언트 | **배제** | 두 arm 모두 같은 sglang 경로 loadgen, 같은 trace_hash, temperature 0, ignore_eos다. random_seed는 339809332 대 1로 다르지만 greedy라 영향이 없다. |
| — | `--enable-pdmux`(동시 실행) | **주원인(차이)** | §2: 정지가 덧셈 구조에서 상수 구조로 바뀌고, 긴 prefill 중에도 decode가 계속 나온다. |

## 4. argv와 실효 server_args (SG-stock 대 PM-ref, 로그 `server_args=` 349키 전수 diff)

| 항목 | SG-stock (pristine, 기본값 유지) | PM-ref (패치 트리) | 이 진단에서 의미 |
|---|---|---|---|
| `enable_pdmux` / config | False / None | True / `pdmux_costterms_D64_candidate.yml`(groups (108,0)…(64,44)…(0,108)), env `PDMUX_R2_POLICY=fixed`, `PDMUX_R2_FIXED_DSM=44`, non-sticky | **주원인** |
| `disable_overlap_schedule` | False | True | 미분리(B1) |
| `chunked_prefill_size` | 8192 | -1 | 무관(입력 ≤3126) |
| `mem_fraction_static` | 0.836 (KV 2,806,320 토큰) | 0.6 (KV 1,586,264 토큰) | 무관(사용률 ≤0.01) |
| `max_mamba_cache_size` | None → 실효 48 | 48 | 같음 |
| `piecewise_cuda_graph_max_tokens` / tokens | 8192 / [4…8192] | -1 / [] | 둘 다 piecewise 꺼짐(`--disable-piecewise-cuda-graph`), 실효 같음 |
| `random_seed` | 339809332(자동) | 1 | greedy, 무관 |
| `port` | — | — | 무관 |
| 같은 값(발췌) | attention flashinfer, mamba_backend triton, sampling flashinfer, cuda_graph_max_bs 256(캡처는 1–48), page_size 1, stream_interval 1, schedule_policy fcfs, mamba_scheduler_strategy no_buffer, enable_mixed_chunk False, num_continuous_decode_steps 1, max_running 48, max_prefill_tokens 16384, context 20480, radix OFF | 같음 | — |

SG-stock과 SG-off의 diff는 `port`·`random_seed` 두 키뿐이다(트리 무관 확인).

## 5. 다음에 잴 것 (단일 변수 boot 제안, 실행하지 않음)

모두 같은 Job 안에서 rate 5.516 + 상위 1점, seed 201/202로 잰다. 판정 지표는 분포 두 봉우리의 위치, 즉
"도착당 정지 초과분"과 "정지가 decode step에 더해지는지(bs=1 대 bs≈30에서 정지 길이 비교)"다.

| boot | 트리 | 바꾸는 것 하나 | 예측 | 무엇을 가르나 |
|---|---|---|---|---|
| **B5** | pristine | PM-ref argv에서 `--enable-pdmux --pdmux-config-path`만 제거 | p95 ≈ 60 ms대 복귀, 덧셈 구조 | "argv 기본값 대 동시 실행" 결정 실험(가장 중요) |
| B1 | pristine | SG-stock + `--disable-overlap-schedule` | 정지 덧셈 구조 유지, p50 +1–2 ms | 후보 1 |
| B2 | pristine | SG-stock에서 `--disable-piecewise-cuda-graph` 제거 | 고정 약 42 ms가 크게 줄면 eager 발행 비용이 확정된다. NemotronH에서 piecewise 캡처가 되는지는 미확인이라 부팅 실패도 결과로 기록 | 공통 고정 비용의 원천 |
| B3 | pristine | SG-stock + `--chunked-prefill-size -1` | 변화 없음 | 후보 2 배제 확인 |
| B4 | pristine | SG-stock + `--mem-fraction-static 0.6 --max-mamba-cache-size 48` | 변화 없음 | 후보 3 배제 확인 |
| P1 | 패치 | SG-stock argv + extend 계측(설계된 계측 boot, decode probe의 `extend_host_ms`/`extend_gpu_ms`) + 가능하면 짧은 torch profiler 창 | 짧은 prefill의 host 구간 ≈ 40 ms, 그 안에 동기화 대기 1회 | 기전(CPU 발행 대 동기화 대기) |

주의: B2의 piecewise 활성은 "손대지 않은 기본값"에서 argv 하나를 빼는 것이므로 stock 정의에 영향이 있다.
진단 전용으로만 쓰고 기준선 arm으로는 쓰지 않는다. 이 문서는 PD-mux가 stock보다 "낫다"는 성능 주장을 하지 않는다.
여기서 단언하는 것은 ITL p95 차이의 **기전 후보와 배제 범위**까지다.
