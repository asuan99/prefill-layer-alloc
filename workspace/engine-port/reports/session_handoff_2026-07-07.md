# 세션 핸드오프 — Zamba2 pdmux layer-aware: coordinated 구현 + 최종 반증 (2026-07-06 ~ 07-07)

다음 세션이 이어갈 수 있게 정리. 정본 결론 보고서: `reports/sm_policy_report.html`(v3, 아직 R0c/R0d 미반영=열린항목 B).
메모리에도 요약 있음(`engine-port-p0-triage.md`, `MEMORY.md`).

---

## 0. TL;DR (한 문단)
사용자가 "layer-aware 은퇴가 미조율(green-ctx) strawman 아니냐"를 끝까지 밀어붙여, **coordinated per-type
layer-aware를 실제로 구현**(green-ctx event-loop 수술)하고 정확성 검증 후 측정 → **여전히 decisively 패배**
(decode TPOT agn 평탄 42ms vs coord 47→124ms 폭발). **§07 (D) granularity를 작동 구현으로 실증** =
per-layer-type 조율은 per-step(agnostic/tuned-uniform)을 못 이김. 근본 이유=autoregressive+커널 timescale.
**부산물(실이득 경로)**: coordinated 재측정에서 **tuned uniform split이 agnostic을 이김**(regime 의존).

## 1. 정책 순위 (실엔진 실측 최종)
**tuned-uniform(d24 prefill-bound / d44 decode-heavy) ≥ default agnostic > layer-aware > (부하시)fused.**
- 모든 uniform = per-step = "layer-agnostic"(층 균일, split만 다름). layer-aware만 per-layer(sub-step)=패배.
- agnostic_v2(균일 floor 16)는 조율하면 uniform의 한 split일 뿐("최악"은 미조율 아티팩트였음).

## 2. 작업 흐름 (이 세션에서 한 것)
- **R-series(T0–T4, 07-06)**: 보고서 v2→**v3**(원본 `_v2` 보존). R0a=Zamba2 clean async 재측정(in3600/out32,
  jobs **834914-916 +reps 834927-929**)=분기(a) la≤agnostic 확정. §04 구GIL-client 패널 교체. sim 문서 SUPERSEDED 배너.
  DEPRECATED_gil_client 마커. `r_series_status.md`. gotcha: §07 Zamba2는 이미 clean(832701/2)이었음.
- **R0b(07-07)**: 사용자 "이진 strawman" 지적→graduated per-type la(`PDMUX_LA_SM_MAP`, jobs 835044-835209).
  ⚠️**결론 철회**(사용자 재지적): 독립 green-ctx라 pdmux prefill과 **미조율→경합**(binary la와 동일 결함).
  결정타: agnostic **coordinated** decode 54=42ms vs **미조율** 54=121ms. `results/r0b/`(정정배너).
- **R0c(07-07)**: **coordinated 상보쌍(manual_divisions)**으로 재측정. ★**coordinated decode knee≈16**(84 아니었음).
  ★**tuned uniform이 agnostic 이김**: in3600 d24(decode24/prefill84) r3 1.18 vs agn 0.55(+114%); in2000 d44 r4
  2.27 vs agn 1.59(+43%). regime 의존. ★agnostic_v2/la "최악/은퇴"=미조율 confound 확인. ★**per-type knee**
  (serving batch, decode-only): attn 3.1→17.5ms/층(@108→16SM, 비쌈·민감·step 지배), mamba 평탄(쌈·둔감). `results/r0c/`.
- **Bullet vs MuxWise 조사**: MuxWise=per-step 균일(=이 프로젝트=agnostic). Bullet=별도 프로세스+libsmctrl+
  layer-chunk(layers_per_step)+SRM으로 동적 **uniform** ratio. ★**3a(Bullet substrate)=driver 580 > libsmctrl ≤535 BLOCKED**.
- **3b: green-ctx coordinated per-type la 직접 구현·측정(07-07, `results/r0d_coord_la/`)** — 아래 3장.
- **Q&A 종합**: per-step>per-layer 확정; 근본이유=autoregressive(step=유용진행 최소단위)+커널 timescale(prefill 커널>decode
  윈도우, 도중 SM 못줄임=(D)); chunked-prefill 두 뜻 구분(Bullet은 **token-chunk disable**[server_args:337 chunked_prefill_size=-1]/**layer-chunk 사용**); token-chunk는 커널 단축하나 KV 재읽기(O(L²/c))·비효율 대가(Bullet이 피하는 것).

## 3. R0d — coordinated per-type la (핵심 산출)
**구현(dev-tree, env `PDMUX_LA_COORD=1`)**: decode를 layer-type 윈도우로 쪼개 **attn=HEAVY 상보쌍(decode74/prefill34)
보호 / mamba=LIGHT(decode16/prefill92) 환원**, prefill을 첫 mamba 윈도우서 상보 prefill-half에 전진.
**정확성 게이트 통과**(부하서 크래시無, coherent·agn과 11/12 일치=batched-greedy 비결정성; 버그 2fix).
**측정(jobs 835918/835919)**: coord-la r2 gp=0.00(in3600, TPOT124)/r3 gp=0.08(in2000). agnostic/d24/d44에 **decisively 패배**.
**원인=(D) 실증**: 19 윈도우 파편화→(i)윈도우 sync 직렬화 (ii)decode sub-partition 핀(prefill 유휴시도 full SM 못씀)
(iii)prefill∥decode overlap 감소 → decode 1.5-3× 느려짐, mamba-윈도우 prefill 이득 상쇄. 상세 `R0d_coord_la_results.md`.

## 4. 코드 변경 (dev-tree: `/scratch/$USER/whlee/sglang_engine_dev/python/sglang/srt/`)
- `models/zamba2.py`: `forward_split_decode`(=forward_split_prefill 재사용) + `la_coord_windows()` + R0b `PDMUX_LA_SM_MAP`(graduated). **src 미러됨**(`workspace/engine-port/src/models/zamba2.py`).
- `model_executor/model_runner.py`: `init_decode_metadata_coord(fb, attn_stream_idx)` + `forward_split_decode(fb, (s,e))`. (core patch, dev_tree_edits 기록 필요.)
- `multiplex/multiplexing_mixin.py`: **`event_loop_pdmux_coord`**(env-gated). **src 미러됨**(`src/multiplex/multiplexing_mixin.py`).
- `managers/scheduler.py`: `dispatch_event_loop`에 `if os.environ.get("PDMUX_LA_COORD"): event_loop_pdmux_coord()` 분기.
- ★버그 2fix(재적용 시 주의): (1) 윈도우 decode 후 `self.running_batch.output_ids = next_ids`(run_batch가 하던 것; 없으면 다음 prepare_for_decode의 input_ids=None crash). (2) `decode_result/prefill_result/prefill_exe_done`를 **while 루프 밖 초기화**(iteration 간 persist; wait_prefill_kernel_done 재시도서 None crash 방지).
- **모두 env-gated/additive** — `PDMUX_LA_COORD`·`PDMUX_LA_SM_MAP` 미설정 시 기존 pdmux 경로 불변.

## 5. 데이터/산출 위치 (`workspace/engine-port/`)
- `results/r0a/` (clean async Zamba2, r0a_summary.csv, sbatch) · `results/r0b/`(graduated, 철회, R0b_graduated_results.md 정정배너)
- `results/r0c/`(coordinated split sweep + knee: r0c_summary.csv, knee_result_*, r0c_coord_bench.sbatch, r0c_knee_isolate.sbatch)
- `results/r0d_coord_la/`(**핵심**: r0d_summary.csv, R0d_coord_la_results.md, PLAN_step4.md, r0d_coord_bench.sbatch, r0d_parity.sbatch, pdmux_coord_la.yml, inefficient_v1/)
- `reports/sm_policy_report.html`(v3, `_v2` 보존) · `reports/r_series_status.md`
- 외부: `external/{bullet,muxwise}`(조사용).

## 6. 열린 항목 / 다음 단계 (우선순위)
1. **(B) 보고서 개정** — 아직 미완. sm_policy_report.html을 coordinated 근거로: (a) confound 정정(agnostic_v2/la 미조율 아티팩트),
   (b) **tuned uniform이 agnostic 이김**(d24/d44, regime 의존) 추가, (c) **coord per-type la 만들어봤고 여전히 (D)로 패배** 콜아웃.
2. **(유망·실이득 예상) SRM/roofline 기반 dynamic uniform split** — agnostic의 decode_bs_divisor 대신 (batch·seqlen·prefill큐·roofline)로
   최적 split 예측. R0c의 d24/d44가 하한. Bullet의 실제 접근과 동형. **green-ctx `adjust_stream_groups` 확장**으로 구현 가능.
3. (검증용, 근거상 여전히 패배 예상) chunked-prefill 켠 coord la — token-chunk로 prefill 커널 단축→timescale 완화 효과 실측.
4. coord la 잔여 최적화(윈도우 sync를 `synchronize`→`wait_stream`; 윈도우 1-18 decode 핀 제거해 full SM 사용) — 방향(패배) 불변 예상.
5. Falcon-H1 층내 attn/mamba 분리계측; Qwen3-Next(gated-deltanet).
6. dev_tree_edits.md에 model_runner/multiplexing_mixin/scheduler core patch 기록(재빌드 대비).

## 7. 환경/재현
- venv `/scratch/$USER/whlee/sglang_engine_venv`, dev tree `/scratch/$USER/whlee/sglang_engine_dev`(sglang v0.5.10).
- `module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0`; sbatch 제출 시 **`env -u BASH_ENV`** 필수.
- SLURM `amd_a100nv_8 --gres=gpu:1`; QOS: 동시 2잡·제출 5잡 한도. driver 580, A100-80GB(108 SM=54 TPC).
- Zamba2: `--attention-backend triton`(head_dim 160 flashinfer NaN), py3.14→cuda graph 불가.
- coord la 실행: `PDMUX_LA_COORD=1` + config `pdmux_coord_la.yml`(2 divisions: LIGHT=[92,16]/HEAVY=[34,74]). 하네스 `r0d_coord_bench.sbatch mMaA rep in out`.

## 8. gotcha/교훈
- **미조율 green-ctx(독립 파티션)=confound**. pdmux 상보쌍(manual_divisions/stream_groups) 써야 coordinated. (R0b가 이걸로 오결론했다 철회.)
- **batched greedy는 bit-비결정적**(reduction order) → parity는 semantic(coherent+정답)으로 판정, exact-match 아님.
- coord decode knee는 **coordinated로 재봐야**(미조율 knee 84는 아티팩트, 실제 16).
- **정책주장은 반드시 서빙실증 + coordinated 구현**(이 세션 자체가 micro/uncoordinated 3회 오도의 교정사).
- Bullet 회피=**token-chunked prefill**(chunked_prefill_size=-1), 사용=**layer-chunk(layers_per_step)**. 둘 구분.
