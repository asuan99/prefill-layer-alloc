# Mamba2(NemotronH) CUDA graph 적용 — SGLang v0.5.10 대 vLLM 0.31.0 코드 대조 (2026-11-08)

> 지위: 코드 읽기만으로 정리한 문서다(실행·GPU 0, 정본 아님, 새 성능 판정 0). 근거 원문은 같은 디렉터리의 두 검토서에 있다.
> - `REVIEW_sglang_v0.5.10_mamba_cudagraph.md`: 로컬 dev tree. Mamba 경로 핵심 11파일은 v0.5.10 pristine sha256과 일치한다.
> - `REVIEW_vllm_0.31.0_mamba_cudagraph.md`: PyPI sdist, sha256 `dde46f1e…`.
>
> 실제 부팅 설정은 Phase B 첫 번째 Job(`pdmux-stock-baseline-7899d1b-20261102085423`)의 boot 로그에서 가져왔다. 줄 번호는 각 검토서를 따른다.

## 1. 대조표

| 항목 | SGLang v0.5.10 | vLLM 0.31.0 |
|---|---|---|
| decode의 Mamba | 전체 decode graph 안에서 실행. 캡처는 `forward_mode.is_cuda_graph()`일 때만(DECODE·TARGET_VERIFY·IDLE). SG-stock은 bs 1–48 | FULL graph. uniform decode 배치(≤ max_num_seqs 48, 캡처 크기 9개)에서만 사용 |
| graph 진입 판정 | forward mode로 판정. EXTEND·MIXED·SPLIT_PREFILL은 graph를 타지 않음 | 배치 모양(shape)으로 판정: `max_q == 1 ∧ num_tokens == max_q·num_reqs` |
| padding 칸 표지 | state index를 `PAD_SLOT_ID = −1`로 채움. 슬롯 0은 dummy로 예약됐지만 decode padding에는 쓰지 않음 | `NULL_BLOCK_ID = 0`(block 0 예약) |
| 커널의 padding 처리 | Triton conv update는 즉시 return, Triton SSU는 load·store를 mask(커널 본문에서 확인) | conv update는 즉시 return, triton SSU는 load·store를 mask(같은 구조) |
| index 버퍼 | bs별로 고정 주소, replay 전에 in-place `copy_` | 고정 주소 `state_indices_tensor_d`, `copy_(non_blocking)` |
| prefill의 Mamba | split op(`nemotron_mamba2_with_output`)으로 eager 실행. in_proj·out_proj GEMM까지 eager | split op(`vllm::mamba_mixer2`)으로 eager 실행 |
| prefill에서 graph로 남는 것 | piecewise ON이면 norm·attention qkv/o proj·MLP. SG-stock은 piecewise를 강제로 꺼서 전부 eager. PM-ref는 chunked −1이라 piecewise 캡처 목록이 비어 자동으로 꺼짐 | ≤96 토큰 배치는 PIECEWISE graph. 96 초과 배치는 graph 없이 inductor 컴파일 코드로 실행 |
| 혼합 배치(prefill+decode) | 기본은 섞지 않음(prefill 배치를 먼저 돌림). `--enable-mixed-chunk`(SG-mix)를 켜면 decode 요청이 extend_len 1로 섞이고, **SSU가 아니라 chunk-scan varlen 경로**로 처리됨 | chunked prefill로 항상 섞음. decode를 앞에 정렬하고 split해서 decode는 SSU, prefill은 chunk-scan으로 **커널 호출을 나눔** |
| Mamba prefill metadata의 host 동기화 | GPU→CPU 동기화 있음: `mamba2_metadata.py:201` `.item()`, chunked이면 `:129-150` 루프(암묵 동기화). `causal_conv1d.py:27`은 **NemotronH 경로에서 실행되지 않음**(호출자가 `seq_lens_cpu`를 넘김) | `.item()`·`.tolist()`는 전부 CPU 텐서 대상이라 D2H 동기화 없음(CPU 사본 metadata) |
| SSM state dtype | fp32(부팅 배너) | fp32(NemotronH 기본값, 사용자 지정이 우선) |
| hybrid 전용 처리 | 슬롯 수 = cap(radix OFF), 정적 풀 | attention block을 1312 토큰으로 키워 mamba page와 정렬, 통합 page 풀 |
| hybrid에서 graph를 강제로 끄는 조건 | NemotronH 없음(Qwen3Next는 piecewise 끔) | 없음(FULL 모드에서 Mamba block 부족이면 ValueError) |

## 2. 남은 위험 (전부 [추론], 재현 0)

| 엔진 | 경로 | 조건 | 이 프로젝트 측정에 해당하는가 |
|---|---|---|---|
| vLLM | 앞선 state가 없는 1-token prefill이 섞인 배치를 모양 판정이 uniform으로 보고 FULL graph로 replay. 고정 index 버퍼 복사를 건너뛰어 직전 step의 index로 실행됨 → 엉뚱한 슬롯 오염 가능 | 프롬프트 길이 1, 또는 decode 배치에 남은 token budget이 정확히 1 | ShareGPT·arXiv 입력에서는 사실상 발화하지 않는다고 봄. 입력 최솟값 미확인 |
| SGLang | 실행 중 재캡처(`recapture_if_needed`)의 warmup이 살아 있는 요청의 슬롯을 갱신할 수 있음 | capture_hidden_mode 변화(hidden state 반환·spec decode) | 하네스에서 이 조건이 열리는 경로를 찾지 못함 |
| SGLang dev | `nemotron_h.py`의 층별 stream 전환·timing 코드가 캡처 중 켜지면 결정이 굳거나 캡처가 실패함 | `PDMUX_FIXED_DECODE_SM*`·`SGLANG_NH_LAYER_TIMING` 설정 | 운영점에서는 꺼져 있어야 함(죽은 트랙 전용) |
| 공통 | padding 칸 state 오염 | — | **근거 없음**: 두 엔진 모두 커널 본문에서 padding 칸의 load·store를 막음 |
| 공통 | PD-mux 동시 접근(SGLang) | — | **근거 없음**: 슬롯이 서로소이고, event query 뒤 merge, decode sync 뒤 해제, 전환 때 양쪽 drain |

## 3. Phase B 해석에 붙일 caveat (공시 후보, 판정 아님)

1. **"Mamba는 CUDA graph를 쓰지 않는다"는 prefill에만 맞다.** 두 엔진 모두 decode Mamba는 graph로 돌고, padding 칸 보호도 커널 수준에서 구현돼 있다. 운영점(cudagraph-ON = decode graph)은 이 구조 위에서 성립한다.
2. **piecewise ON의 상한.** upstream 기본(piecewise ON)에서도 Mamba 27층은 in/out proj까지 eager로 남는다. graph 대상은 norm·attention proj·MLP뿐이다. SG arm 정지(약 42 ms)가 piecewise로 줄어드는 폭은 이 몫으로 제한될 것으로 본다(미측정).
3. **정지 원인 후보 추가(미측정).** SGLang Mamba prefill metadata에는 GPU→CPU 동기화(`mamba2_metadata.py:201`, chunked이면 `:129-150`)가 있고, vLLM에는 없다. Phase B 감사 A2의 "원인 미측정 host 고정 작업" 후보에 이것을 넣는다. 2026-11-08 답변에서 `causal_conv1d.py:27`을 후보로 든 것은 **철회**한다(NemotronH 경로에서 실행되지 않음).
4. **SG 대 VL 비교(Q3·Q4)에는 엔진 구조 차이가 섞인다.**
   - 혼합 방식: SGLang은 섞지 않아 도착마다 decode가 멈추고, vLLM은 섞어서 step이 느려진다.
   - graph 범위: SGLang은 decode graph만, vLLM은 FULL + ≤96 토큰 PIECEWISE + inductor.
   - host 동기화 유무.
   - 따라서 ITL 차이를 커널이나 모델 성질로 귀속하지 않는다(Phase B 감사 인용 금지 3과 같은 방향).
5. **SG-mix(Q5) arm 고유 경로.** mixed-chunk에서 SGLang은 decode 요청을 SSU가 아니라 chunk-scan(initial state 포함)으로 처리한다. SG-stock과 수치·비용 경로가 다르므로, Q5를 해석할 때 이 차이를 함께 적는다.
6. **SG-stock 대 PM-ref prefill 수치 경로.** SG-stock은 chunked 8192에서 두 번째 chunk부터 initial-state 경로를 타고, PM-ref는 prompt 전체를 한 번에 처리해 prefix 0이다. 출력이 bitwise로 같다는 보장이 없다(gate 2는 같은 배치 구성 조건에서만 판정한다).

## 4. 확인하지 못한 것

- VESSL에서 실제로 돈 트리와 로컬 dev tree가 바이트 단위로 같은지. pristine manifest 원본은 git 밖이다.
- sgl-kernel 0.4.1 바이너리가 검토한 소스로 빌드됐는지.
- 하네스가 `--mamba-backend`를 지정했는지. 기본값 triton으로 가정했다.
- norm·MoE 커널의 행 독립성 정밀 검토(padding 행 쓰레기 값의 전파).
- vLLM 1-token prefill 경로가 실제로 발화하는지(입력 길이 최솟값).
