# ANALYSIS — Nano chunk -1 프리필이 96k에서 실패하는 원인 (2026-10-20, main, 코드 읽기 + 산술; 3회차 프로브로 확인 중)

> 지위: 원인 분석이다(코드 사실 + 산술). ★**3회차 확인(2026-10-20)**: L = 94,500은 OK, 95,100은 실패했다. `CUDA_LAUNCH_BLOCKING=1` traceback은 `causal_conv1d_triton.py:510 causal_conv1d_fn`이다 → 가설과 일치한다(run `longctx_probe_20261020T040836Z`, 1회 관측). 엔진은 고치지 않았다.

## 1. 관측 (2회차, mem-fraction 0.6)
- Nano PD-mux와 Nano fused\_sym(둘 다 chunk -1)은 64k에서 OK, 96k에서 `RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered`가 났다. traceback의 마지막 줄은 `ssd_chunk_state.py:464 _chunk_cumsum_fwd`다.
- 실패 시점의 활성화 여유는 29.0 / 30.6 GiB였다. **메모리 부족이 아니다.** OOM이면 `torch.OutOfMemoryError`가 나야 한다(1회차 0.82는 실제로 그 오류였다).
- 같은 Nano라도 chunked 8192(forward 하나가 8192 토큰 이하)는 124k까지 OK다. Llama(attention·MLP만 있음)는 chunk -1로도 124k까지 OK다.

## 2. 오류가 찍힌 곳이 원인 커널이 아니다
- CUDA 오류는 **비동기**다. 잘못된 메모리 접근은 그것을 일으킨 커널이 끝난 뒤, 다음 launch나 동기화 지점에서 보고된다. `_chunk_cumsum_fwd`는 int64로 이미 보호돼 있다(`ssd_chunk_state.py:57-59`, "if dt is long, may cause problems, so use 64 bit", Triton issue #1058). 이 커널은 **보고 지점일 가능성이 크다**.
- Mamba 층의 prefill 순서 [코드, `mamba.py:416-560`]: in\_proj(cuBLAS) → `torch.split`으로 gate / xBC / dt **view**를 만든다 → **causal conv1d(xBC)** → SSD(`mamba_chunk_scan_combined`, 첫 커널이 `_chunk_cumsum_fwd`). 따라서 바로 앞 커널인 causal conv1d가 1순위 용의자다.

## 3. 용의 커널: Triton causal conv1d의 int32 토큰 오프셋
- conv 입력 `x = hidden_states_B_C_p.transpose(0, 1)`은 in\_proj 출력(`zxbcdt`)의 **비연속 view**다. 그래서 토큰 하나를 건너는 stride가 conv 폭 12,288이 아니라 **in\_proj 출력 폭 22,656**이다 [코드 + config: d\_inner = 128 × 80 = 10,240; in\_proj 출력 = 2·10,240 + 2·8·128 + 128 = 22,656].
- stride가 1이 아니면 래퍼는 sgl\_kernel CUDA 대신 **Triton 경로**를 쓴다: `use_triton = not _HAS_SGL_KERNEL or (x.stride(-1) != 1 and "seq_lens_cpu" in kwargs)` [코드, `causal_conv1d.py:42-46`].
- Triton 커널 `_causal_conv1d_fwd_kernel` [코드, `causal_conv1d_triton.py`]:
  - `chunk_offset = tl.program_id(1)`(int32), `token_offset = BLOCK_M * chunk_offset`(:63)
  - `prior_tokens = x_base + (token_offset - 1) * stride_x_token`(:267), `x_ptrs = x_base + ((idx_tokens_conv - VAL) * stride_x_token)`(:213-215 부근)
  - `stride_x_token`은 `tl.constexpr` = 22,656이다. **int64 캐스트가 없다**(SSD 커널들과 다름).
- 산술: 토큰 위치 t의 원소 오프셋은 t × 22,656이다. 이 값이 int32 최댓값 2,147,483,647을 넘는 지점은 **t > 94,786**이다.
  - 64k(65,536 × 22,656 = 1.48e9)는 안쪽이고, 96k(98,304 × 22,656 = 2.23e9)는 바깥이다. **관측된 경계(64k OK, 96k 실패)와 정확히 맞는다.**
  - 오프셋이 음수로 넘쳐 할당 영역 밖을 읽으면 `illegal memory access`가 난다.
- 다른 텐서 폭의 경계 [산술]: conv 출력·xBC 연속(12,288) 174,762 · MLP 중간(15,680) 136,956 · hidden(4,480) 479,349. 96k 이전에 int32를 넘는 것은 in\_proj 출력 view(22,656)뿐이다.
- 덧붙여 sgl\_kernel CUDA conv도 stride를 `uint32_t index_t`로 다룬다[코드, `sgl-kernel/csrc/mamba/causal_conv1d.h:13`]. 다만 이번 경로에서는 쓰이지 않는다.

## 4. 왜 다른 경우에는 안 터지나
- **chunked 8192**: forward 하나에 들어가는 토큰이 8192 이하라 최대 오프셋이 8192 × 22,656 = 1.9e8이다.
- **Llama**: Mamba conv 커널이 없다. GEMM(cuBLAS)과 attention(flashinfer)은 64-bit 인덱싱을 쓴다.
- **PD-mux**: chunked prefill을 강제로 끄므로(assert) 단일 장문 요청이 통째로 이 커널에 들어간다. 그래서 PD-mux arm은 이 한계를 피할 수 없다.

## 5. 함의와 수리 선택지 (결정 아님)
- 이것은 hybrid 고유 비용 구조가 아니다. **현 SGLang Mamba 커널 구현의 한계**(int32 인덱싱 + 비연속 view)다. "PD-mux는 64k 이상 불가"라고 쓰면 안 되고, "이 스택(SGLang v0.5.10, Triton causal conv1d)에서"라는 범위를 붙여야 한다.
- 수리 후보(엔진 패치, engine-porter + 정확성 게이트 필요):
  - (a) 커널에서 `chunk_offset`·`token_offset`을 `tl.int64`로 캐스트한다(SSD 커널과 같은 방식, 한 줄 수준). 그러면 128k까지 연다.
  - (b) conv 전에 xBC를 `.contiguous()`로 바꾼다. stride가 12,288이 되어 경계가 174k로 올라가지만 복사 비용이 든다.
- upstream(vLLM·SGLang 최신)에 이미 고쳐졌는지는 미확인이다.

## 6. 3회차 프로브의 확인 설계
- fused\_sym 부팅을 `CUDA_LAUNCH_BLOCKING=1`로 하고 L = 94,500(경계 아래) → 95,100(경계 위) 순서로 보낸다.
- 예측: 94,500은 OK, 95,100은 실패한다. blocking 모드의 traceback은 `_causal_conv1d_fwd_kernel`을 가리킬 것이다.
- 예측과 다르면 이 가설을 철회하고 다른 커널을 찾는다.
