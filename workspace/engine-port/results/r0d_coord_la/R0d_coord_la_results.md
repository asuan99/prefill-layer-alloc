# R0d — coordinated per-type layer-aware (green-ctx, 3b): BUILT, CORRECT, and REFUTED

작성: 2026-07-07. 사용자 요청(옵션 A): "미조율 green-ctx가 아닌 **coordinated** per-type la를 실제로
구현해 per-type가 tuned uniform을 이기는지 실증." 3a(Bullet/libsmctrl)=driver 580>≤535로 BLOCKED → 3b(green-ctx).

## 구현 (green-ctx, single-process, event-loop 수술)
- `models/zamba2.py`: `forward_split_decode`(windowed decode) + `la_coord_windows()`(ABAB → [(s,e,is_attn)]).
- `model_runner.py`: `init_decode_metadata_coord`(decode-heavy attn backend 1회 init; mamba 윈도우는 attn backend 미접촉) + `forward_split_decode`.
- `multiplex/multiplexing_mixin.py`: **`event_loop_pdmux_coord`**(env `PDMUX_LA_COORD=1`). decode를 layer-type 윈도우로 쪼개고,
  **attn 윈도우=HEAVY 상보쌍(decode 74/prefill 34, 보호), mamba 윈도우=LIGHT(decode 16/prefill 92, 환원)**,
  prefill은 첫 mamba 윈도우서 상보 prefill-half에 전진(윈도우당 run_batch 회피). 윈도우 경계 sync.
- `scheduler.py`: env set 시 dispatch. src 미러: `src/{models/zamba2.py, multiplex/multiplexing_mixin.py}`.
- config: `pdmux_coord_la.yml`(2 divisions). 하네스 `r0d_coord_bench.sbatch mMaA rep in out`.

## 정확성 게이트 — 통과
- 지속부하(48 concurrent) 인터리브 실행, **크래시 無**, 출력 coherent·정확("Paris"/"oxygen"/"two"...).
- coord vs agnostic greedy 11/12 정확 일치(1 불일치=batched-greedy 비결정성, agn도 INTRA_DETERMINISTIC=False; 불일치도 coherent). ⇒ windowing 수학 정확.
- 디버깅 2건 수정: (1) `running_batch.output_ids=next_ids`(run_batch가 하던 것; 미설정 시 다음 prepare_for_decode의 input_ids=None), (2) prefill_result/prefill_exe_done를 **루프 밖 초기화**(원 event loop처럼 iteration 간 persist; wait_prefill_kernel_done 재시도 시 None crash 방지).

## 측정 (goodput@SLO req/s; jobs 835918 in3600/o32, 835919 in2000/o96; optimized: prefill 1x/step)
| rate | in3600: agn / **d24** / coord-la | in2000: agn / **d44** / coord-la | coord TPOT |
|---|---|---|---|
| 1 | 1.11 / 1.20 / **0.79** | 1.17 / 1.17 / **1.15** | 57 / 47ms |
| 2 | 2.07 / **2.24** / **0.00** | 2.25 / 2.24 / **0.80** | 124 / 62ms |
| 3 | 0.55 / **1.18** / **0.00** | 3.24 / **3.22** / **0.08** | 123 / 117ms |
| 4 | 0.31 / — / 0.00 | 1.59 / **2.27** / **0.00** | 123 / 134ms |

(초기 비최적 버전=윈도우당 run_batch → TPOT 698ms; inefficient_v1/ 보관. 최적화(prefill 1x/step)로 123ms까지 개선했으나 여전히 열위.)

## 판정: coordinated per-type la도 **tuned uniform·agnostic에 decisively 패배**(양 regime). 창립 가설 최종 반증.

- decode **TPOT**가 핵심: agnostic은 전 rate 평탄 ~42ms; coord-la는 r1 ~47-57ms→부하시 **117-124ms 폭발**·부하와 함께 악화.
- 원인(= §07 (D) granularity **실증 확인**): decode 스텝을 ~19 layer-type 윈도우로 **파편화**하면
  (i) 윈도우 경계마다 partition-switch **sync**(직렬화), (ii) decode가 sub-partition(16/74)에 **핀**되어 prefill 유휴시에도 full SM 못씀, (iii) prefill∥decode **overlap 감소**(prefill이 한 윈도우에만) → decode 1.5-3× 느려짐. 이 오버헤드가 mamba-윈도우 prefill 이득(92 SM)을 완전히 상쇄.
- ⇒ **fine-grained(per-layer-type) 조율은 coarse(per-step) 조율을 못 이김.** uniform per-step(agnostic/d24/d44)이 prefill↔decode SM 트레이드가 성립하는 유일 granularity.

## 정직한 caveat
- 구현에 잔여 오버헤드(19 sync는 `wait_stream` 대신 `synchronize`; 윈도우 1-18서 decode 핀). 더 최적화하면 절대 TPOT는 더 내려가나, **구조적 비용(파편화·overlap 감소)은 근본적**이라 방향(coord < agnostic) 불변, 그리고 uniform이 이미 잡는 이득 이상을 줄 upside 없음.
- config m16a74 1점(knee상 합리적 per-type: mamba 저·attn 고). 다른 config도 같은 구조적 비용.
- ⇒ 실전 권고 = **agnostic / regime-tuned uniform coordinated split**. per-type la는 실증적으로 죽음.

## 대미
사용자 지적(미조율=strawman)은 옳았고 → coordinated 구현으로 공정 재검. 그 결과 **coordinated per-type도 짐**:
strawman이라서가 아니라 **granularity 때문**. 이제 (D)는 논증이 아니라 **작동하는 구현으로 실증**됨.
