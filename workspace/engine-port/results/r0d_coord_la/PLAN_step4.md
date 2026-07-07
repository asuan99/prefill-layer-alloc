# 3b — coordinated per-type layer-aware (green-ctx). Step 4 build plan (start 16:00 KST 2026-07-07)

목표: MuxWise/green-ctx(단일 프로세스, 상보 쌍) 위에서 **layer-type 윈도우별 coordinated 재파티션** 구현.
3a(Bullet/libsmctrl)는 **driver 580 > libsmctrl ≤535 요구로 BLOCKED**(green-ctx가 애초에 이걸 피하려 선택됨). libsmctrl 미사용.

## 이미 완료(추가·컴파일확인, additive — 기존 경로 불변)
- `models/zamba2.py`: `forward_split_decode`(= forward_split_prefill 재사용, 내부 green-ctx 스위칭 없음) + `la_coord_windows()`(ABAB 최대 동일타입 런 → [(s,e,is_attn)]).
- `model_executor/model_runner.py`: `init_decode_metadata_coord(fb, attn_stream_idx)`(decode-heavy attn backend 1회 init; attn 윈도우 공용, mamba 윈도우는 attn backend 미접촉) + `forward_split_decode(fb, (s,e))`.
- src 미러: `src/models/zamba2.py` (model_runner는 core patch → dev_tree_edits에 기록).

## 검증된 API (forward_batch_generation 분해)
- build fb: `ForwardBatch.init_new(model_worker_batch, model_runner)`
- mwb: `schedule_batch.get_model_worker_batch()`
- sample: `model_runner.sample(logits_output, model_worker_batch)`
- prefill split(`forward_batch_split_prefill` 패턴): split_index==0에 fb를 `batch.split_forward_batch`에 생성 → `model_runner.forward(split_forward_batch, split_forward_count=k)`로 k층 전진. logits는 마지막 청크서 반환.
- `forward_split_decode`는 end==num_layers(마지막 윈도우)서만 logits 반환.

## pdmux config (신규): 2 divisions
```yaml
sm_group_num: 4
manual_divisions:
  - [92, 16, 0]    # idx1 = LIGHT (mamba 윈도우): decode 16 / prefill 92
  - [34, 74, 0]    # idx2 = HEAVY (attn 윈도우): decode 74 / prefill 34
```
stream_groups: idx0=(108,0), idx1=(92,16)=LIGHT, idx2=(34,74)=HEAVY, idx3=(0,108).
(스윕용으로 mamba floor·attn reserve 값 env로 파라미터화 권장)

## Step 4 구현: `event_loop_pdmux_coord` (env `PDMUX_LA_COORD=1` gate)
`multiplexing_mixin.py`에 `event_loop_pdmux`의 복사본 추가, **steps 5–6(decode 전체 ∥ prefill 청크)를 윈도우 인터리브로 교체**(decode·prefill 둘 다 active일 때):
```
decode_mwb = running_batch.get_model_worker_batch()
decode_fb  = ForwardBatch.init_new(decode_mwb, MR)
MR.init_decode_metadata_coord(decode_fb, HEAVY_IDX=2)
windows = model.la_coord_windows()
# prefill fb(split_forward_batch) split_index==0서 생성; per-window 층수 = ceil(남은/len(windows))
logits=None
for (s,e,is_attn) in windows:
    idx = 2 if is_attn else 1
    set_current_stream_idx(idx); ps,ds = stream_groups[idx]
    with cuda.stream(ds):
        r = MR.forward_split_decode(decode_fb, (s,e));  logits = r if r is not None else logits
    if prefill_active:
        with cuda.stream(ps):
            MR.forward(split_forward_batch, split_forward_count=per_window_k)  # split_index 전진
    ds.synchronize(); ps.synchronize()
decode_result = GenerationBatchResult(logits_output=logits)
decode_result.next_token_ids = MR.sample(logits, decode_mwb)
# 이후 기존 process_batch_result(decode) + prefill 완료/merge 로직 재사용
```
- decode-only(프리필無) 또는 prefill-only → 기존 경로 폴백.
- scheduler: env set 시 `event_loop_pdmux_coord` 디스패치.

## 정확성 게이트(측정 前 필수)
(a) **degenerate**: config LIGHT=HEAVY=full(108/0 아닌 108-decode? → 단일 윈도우·full SM) → windowed ≡ normal decode → **Paris + 토큰 parity**.
(b) **real per-type**(16/74) → **Paris + 토큰 parity** vs uncoordinated(같은 프롬프트 greedy 동일 토큰).
(c) 통과 시에만 측정. parity 깨지면 **중단·보고**(그럴듯한 오답 금지).

## 측정(게이트 통과 후)
coord per-type la vs {agnostic, best-uniform d24(in3600)/d44(in2000), la_bin(uncoord)} — 양 regime, clean async, in3600/out32 + in2000/out96, rates 1-6, 2 reps. goodput/TTFT/TPOT → results/r0d_coord_la/.
판정: coord per-type가 best-uniform을 이기나? (D) sync 오버헤드가 mamba-윈도우 prefill 이득을 잡아먹나?

## 리스크/주의
- TP=1(현 서빙)이라 prepare_mlp_sync/TP scatter 바이패스 OK. TP>1은 범위 밖.
- mamba conv/ssm state·KV: 층별 1회 실행 → windowing 무해(정확).
- cuda graph disabled(py3.14) → eager, graph 제약 없음.
- prefill split_index는 split_prefill_batch(ScheduleBatch)에 persist.
- 윈도우 경계 sync = decode·prefill 둘 다 synchronize (race-safe).
