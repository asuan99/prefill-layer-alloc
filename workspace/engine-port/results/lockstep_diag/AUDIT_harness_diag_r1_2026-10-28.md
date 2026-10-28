# 하네스층 감사 r1 — PREREG_DIAG_stall_drain rev0.1 + 계측 96ae3afa (2026-10-28)

- 감사자: claims-auditor(읽기 전용) · 회차 1/2 · **판정 NO-GO**(死因 K1·K2, 문서·계측 소폭 수리). r2(최종)는 수리 diff만 본다.
- CPU 재현: 이 머신에서 `test_loadgen_abs_mono`(6)만 OK. `test_phase_diag`·`test_engine_gate_diag`는 `sgl_kernel` 부재로 import 실패 → "9 mutants killed" **미재현**. 봉인 전 sgl_kernel 있는 환경 로그 첨부 필요.

## 死因
- **K1 판별량 A가 엔진 회계 항등식.** 루프 순서 decode 발행(t_launch, `multiplexing_mixin.py:1405`) → chunk 발행(`:1468`) → decode sync(`:1494`)이므로 [t_issue_start, t_issue_end] ⊂ [t_launch, t_sync], H ≥ X2 정의상 성립, H ≈ max(X2+ε, X3−δ). R이 X2와 독립이면 OLS b2 = 1.00 → `FOLLOWS_X2`가 기전 무관하게 나옴. `FLAT_X2`는 decode-bound 영역에서만 → `STALL_FOLLOWS_CHUNK_GPU`·`_DECODE_GPU`(PREREG :77,:79)는 영역 혼합이 결정. launch-bound에선 end event가 마지막 launch 뒤 기록(`:2525` 근방)이라 X1 ≥ X2 기계적. **수리**: 판별량을 R = t_sync − t_issue_end, Z = H − max(X2, X3)로 바꾸고 X1·X3·forward_count에 회귀, X2는 회계 항으로 별도 보고. 비차단 플래그 2개 추가 — `start_done_at_issue_start = pair[0].query()`(record 직후), `end_done_at_issue_end = pair[1].query()`(t1 직후) — launch-bound/GPU-bound 반복 구분. digest 재변경.
- **K2 라벨 우선순위 부재.** :76 `STALL_ABSENT`와 :81 `UNRESOLVED_FEW` 동시 성립 가능(N3 Job 4 선례 영역). :69 "주 추정은 모든 반복" vs :76 "ABSENT면 기울기 판별 안 함" 모순. **수리**: 순서 ABSENT → FEW → COLLINEAR → 기울기 라벨 고정, ABSENT 시 주 추정 보고만/판정 명시.

## MAJOR
- **M1** ":87 static arm에는 drain 없음"은 거짓: `adjust` drain(`:1358-1368`)은 prefill 끝마다 발화(`:1541`, 주석 `:1270-1272` "EVERY prefill boundary"), 동적 arm은 SLO 분기가 억제(`:1357`). static 모드(O-P6)에 따라 static drain이 span마다~0. 수리: 판별량 B를 세 arm 모두 계산·seed별 dyn−static 차 기술값, D-1·D-3 static 모드 등록, "static 무 drain" 금지 문안 추가.
- **M2** drain 겹침 ≠ 손실(sync 대기 중 GPU prefill 진행). 라벨 `DRAIN_OVERLAP_*`, "몫 = 비용" 해석 금지. SLO_SCHED 아래 컨트롤러 결정이 `adjust_stream_groups`(`:1154-1162`)로 수행돼 reason=`adjust`로 찍힘 → 전환 집계는 `stream_idx_from≠to`로. drain ms에 backend 갱신 미포함 공시.
- **M3** ITL 겹침에 client–server 지연 포함(D5는 순서만 확인). lag = token_t_mono[0] − 같은 rid prefill_span_end 분포(p50/p99)를 gate·본측정에서 보고, 겹침 허용창 데이터 전 등록(권고 lag p99, 민감도 0·2×p99).
- **M4** 짝짓기 키 미등록. 권고: X1 = 같은 `decode_iter_seq` chunk, X3 = host bracket 종료 ≤ t_launch인 직전 probe 레코드, 조인 실패율 보고.
- **M5** D-1 관측자 비용 boot 미반복(ON/OFF 각 1 boot). `OBSERVER_*` 라벨은 D-1 기술값으로 강등, 라벨은 D-3(seed별 재부팅)에서만, 또는 OFF/ON 각 2 boot 교대. OFF가 끄는 범위 등록.
- **M6** realized-SM 판정 단위: span 단위 제외, 셀은 실현률 ≥ 95%만 유지(8k 0.2 req/s에서 decode 빈 span 약 7% [유도]).
- **M7** gate 공백: (a) ON boot에서 `torch.cuda.Stream.synchronize`·`Event.synchronize`·`Tensor.item` 래핑해 diag 블록 호출 횟수 == 0 요구 (b) D1 동시성 경우 출력은 기록만 (c) D6 UNRESOLVED 규칙 — D-1 허용, D-3은 첫 boot에서 slo_switch 정합(D3+D6) PASS 필수 (d) mutant 추가: 해결 전 pool 반환, MAX_PENDING 초과 드롭 미집계.
- **M8** digest 핀: 정책 PREREG Q-7은 99bbef91, D-3은 TUNE 구성을 새 트리에서 돌림 → 정책·진단 Job을 같은 최종 트리 하나로 핀, Q-7 갱신. "PHASE_EVENTS ON·DIAG OFF" 조합 boot 추가(D2 공백).

## MINOR
헤더 rev 표기 불일치(:1 "rev0.1.1") · :25 P-2 TBD → `PDMUX_PHASE_DIAG` · :113 X-5 Zamba2-7B 잔여(결정 46 충돌) · :188 "모델과 backend가 함께 다르므로" 문안 수정 · :73 FLAT 여유 근거는 관례임을 공시 · chunk_begin poll/emit(`:2489~`)이 [t_launch, t_sync] 안 → 관측자 비용이 H에 포함됨 공시. 결정 46 공시·정책 판정 0·금지 문안 승계는 적합.

## 봉인 전 닫을 [미정] — 권고
O-DX2 n = 4 · O-DX4 8k rate 0.1 req/s·40 요청(D-1 > 6 h면 프롬프트 길이별 2 Job, Job마다 OFF boot) · O-DX3 static = D44 사전 지정, 모드는 정책 O-P6와 같은 값 · 동등성 여유 유지(잔차 판별량) · DRAIN 임계 유지·라벨 OVERLAP · 겹침 허용창 lag p99 · realized span 단위 · digest 단일 트리 핀.
