# SGLang Breakable CUDA Graph — 적용 가능성 요약 보고 (2026-08-21)

> ⚠️ **정본 아님.** 새 성능 판정 **0건** · 등급 변경 **0건** · 정책 순위 변경 **0건** ·
> GPU 지출 **0**. 코드 사실 + 이미 등재된 측정의 재조합이며, 실행 전 결정을 위한 분석이다.
> 정본은 [`../PROJECT_STATUS.md`](../PROJECT_STATUS.md) > [`paper/`](paper/) >
> [`CONSENSUS.md`](CONSENSUS.md) 순서 그대로다.
>
> **전문(코드 근거 file:line, 검증표, P0 프로브 설계)**:
> [`../workspace/engine-port/reports/BCG_APPLICABILITY_2026-08-21.md`](../workspace/engine-port/reports/BCG_APPLICABILITY_2026-08-21.md)

대상: <https://docs.sglang.io/docs/advanced_features/breakable_cuda_graph> —
캡처된 CUDA graph를 **세그먼트로 쪼개고 그 사이에 eager 실행을 허용**하는 기능
(`@eager_on_graph` / `break_graph()` / `SGLANG_USE_BREAKABLE_CUDA_GRAPH=1` / `--debug-cuda-graph`).

---

## 한 문단 결론

BCG의 가치는 **layer-aware 부활이 아니다(그건 여전히 死)**. 이 기능은
`cudagraph_results.md:78`이 적은 불가능성의 두 다리 — **①고정 스트림 · ②고정 op 시퀀스** —
중 **②만 자른다.** ①(세그먼트마다 green context를 바꾸는 것)은 stock BCG에 없고, 그
잔차를 정본이 **구조적 오버랩 손실 — 윈도우수 무관·모델 독립**으로 등재했다
(`CONSENSUS.md:1391`). ⚠️ 같은 줄의 수치(`PDMUX_LA_COORD_OPT` TPOT 124→85 ms, 47% 회수)는
**`--disable-cuda-graph` 캠페인의 n=1 값**(`a_substrate/a_coord_opt_bench.sbatch:41`)이고
원 문서가 *"124ms는 CPU-sync+pinning으로 부풀려진 값"* 이라 표시했으므로
(`A_substrate_isolation_results.md:35`), **cgON 트랙의 상한으로 인용하지 않는다** — 방향만
지지한다. 실제로 값어치가 있는 지점은 두 개다:
**(A) 운영점(cudagraph-ON)에서 처음으로 per-layer 계측을 가능하게 하는 것**,
**(B) 이 프로젝트 운영점에서 유일하게 남은 완전-eager 구성요소인 prefill을 그래프화할
가능성**. 둘 다 **아직 아무도 안 잰 상위 게이트 하나** 뒤에 있다 — *green-ctx 스트림에서
캡처한 그래프가 replay 시 SM 한정을 전달하는가*(구멍 C / 가정 E5, 2026-08-11 등재, 미측정).

---

## 판정 요지

| # | 시나리오 | 판정 | 성립 조건 / 死因 |
|---|---|---|---|
| **S1** | **운영점 per-layer 계측 복원** (break 지점에서 event record) | ★**최고 가치 · 추진 권고**<br>(성능 주장 아님, 계측 판정) | 현재 `replay()`가 Python forward를 호출하지 않아 **모든 per-layer 계측이 정의상 off-operating-point**. 2026-08-20 ncu×green-ctx 사망으로 `kernel_mech`에 Stage A(nsys)만 남은 상태에 **세 번째 경로**. 조건 = per-break 비용이 ITL **3% 미만**(게이트 #3)임을 null-break 대조로 먼저 입증 |
| **S2** | **per-window cudagraph** (`cudagraph_results.md:129-136`의 heroic 트랙) | **미실행 권고 유지** | BCG는 3개 사유 중 **(ii)"커스텀 캡처 大공수"만** 해소. (i) 창 사이 drain·(iii) 낮은 payoff 불변. ★NO-GO를 떠받치는 건 `CONSENSUS.md:1391`(구조적 오버랩 손실 — **윈도우수 무관·모델 독립**) + §1-3(4모델 서빙 직접 측정)이지 F8의 cgOFF n=1 수치가 아니다. "창이 적은 모델이면 살아난다"는 탈출구도 그 **윈도우수 무관** 등재로 이미 닫혔다. 추가 死因: Zamba2 ABAB → 창 ~19개 = step당 브레이크 19개; 세그먼트별 SM 재배치는 stock BCG 밖 |
| **S3** | **prefill 그래프화** | **최대 잠재 이득 · 미탐색**<br>(설계 선행 필요) | 업스트림 BCG 헤드라인이 정확히 prefill(eager 대비 **1.70×**)이고, 이 프로젝트에서 **prefill은 100% eager**(`SPLIT_PREFILL`이 `is_cuda_graph()`에서 제외), goodput은 TTFT 지배. 장애 = PD-mux가 chunked prefill 금지·층축 split이라 캡처 키가 (span 층수 × token 버킷 × stream_group)로 열림 → **span 층수 고정 버킷 강제** 필요, 그 대가로 컨트롤러 스위치 해상도가 묶임 |
| **S4** | **`--debug-cuda-graph` 미분 오라클** | **저비용 — 단 L1 선지불 후** | graph runner 경로(버퍼·패딩·per-group attn backend)를 유지한 채 eager ⇒ 경로 자체가 갈리는 현행 `--disable-cuda-graph`보다 타이트. correctness gate + "cgOFF vs cgON 격차" 귀속 정밀화. ⚠️ `--debug-cuda-graph`도 이 트리에 **없으므로**(F1) S4 역시 백포트=**L1을 지불한 뒤**에만 쓸 수 있다 |

### 이점이 **없는** 지점

- **HE0(단일 GPU 동적 제어)는 안 되살아난다.** 死因은 switch 비용이 아니라
  **positioning + 공유 running-batch/KV 얽힘**이다 (`CONSENSUS.md:1358` switch 2회 rep가
  static 매칭 · `slo(5sw) < bind(21sw)`; `:1362` 컨트롤러 CPU = wall의 **0.014%**).
  BCG는 둘 중 어느 것도 건드리지 않는다. 열린 lever는 admission/KV-aware다(`CONSENSUS.md:3277`).
- **prefill 인터리빙**: 이미 층 단위 eager로 손수 구현돼 있다(사실상 hand-rolled breakable).
- **decode step 선점**: break는 CPU 제어만 돌려준다. 얽힘 기전에 개입할 수단이 아니다.
- **캡처 조합 폭발·그래프 메모리**: 완화 안 됨. 세그먼트 수만 늘어난다.

### 손실

| # | 손실 | 크기 |
|---|---|---|
| L1 | **비교가능성 손실 (최대 비용)** | BCG는 이 트리(v0.5.10)에 **없다**. 업그레이드면 저장소 97개 `.sbatch` 캠페인의 basis 무효. 백포트여도 manifest 규약에 새 항목 |
| L2 | 캡처 시간·메모리 | 이미 pdmux가 plain 대비 **2.8× 시간 / 6.4× 메모리**(11.20 s / 0.83 GB). prefill까지 캡처하면 업스트림 실측 "42 shapes = 2.4 GB" 급 → `--mem-fraction-static 0.82` 하에서 **decode batch 도달성을 더 좁히는 방향**(⚠️ "ctx4096 SM92 B≥9 도달률 0.0–3.0%"는 7–8B arm 격자이고 `CONSENSUS.md:2365-2370`이 provenance를 T8 행 하나로 제한했다 — Zamba2-2.7B로 이식 금지, 방향 진술만) |
| L3 | 운영점 오염 | 브레이크당 launch + Python 호출. request-내부 ITL p95가 goodput 정의에 직접 들어가고 게이트 #3이 3% 문턱이라 **perturbation이 주장 가능 효과와 같은 오더**일 수 있다 |
| L4 | 호환 리스크 | `SGLANG_MEMORY_SAVER_CUDA_GRAPH` 비양립 · `cuda-python` 필요 · **`torch.cuda.Stream.wait_stream` 후킹**(이 저장소는 `graph_capture`·`ExternalStream`·true-dual-worker가 wait_stream/event를 무겁게 씀) |
| L5 | 스레드 안전 | `CURRENT_STREAM_IDX`가 **모듈 전역**(ContextVar 아님). break 콜백이 replay 스레드에서 돈다 |
| L6 | 상위 게이트 미해결 | 구멍 C/E5 미측정 + 업스트림이 **torch ≥ 2.7에서 green-ctx × cudagraph 저하 경고**(현 환경 2.9) |

---

## P0 (실행 전 반드시 닫아야 할 두 건, 예약 기준 ≈3.1 GPU-hr)

- **P0-A — graph replay가 SM 한정을 전달하는가.** 기존 `%smid` L0 census
  (`results/smid_census/`, 작성·사전등록 완료, **GPU 실행 이력 0**)의 파생 스크립트로
  `eager_green / graph_green / eager_plain / graph_plain` 4-leg 대조. 선행으로 그
  census의 R0를 먼저 돌려야 한다(≈10분). **전달 안 됨으로 나오면 decode 측 SM 분할
  서사 전체가 재검토 대상**이 되므로, BCG와 무관하게 이 게이트는 값어치가 있다.
- **P0-B — BCG 백포트 스모크.** 업스트림 4파일(본체 ≈450줄, 내부 의존은 이 트리에 이미
  존재) 백포트 → pdmux + Zamba2 부팅 → (i) 캡처 성공 여부, (ii) wait_stream 후킹 충돌,
  (iii) **null-break(브레이크 0) 대비 per-break 비용**. (iii)이 3% 초과면 S1도 죽는다.
  적대 감사(2026-08-21) 후 **3블록 인터리브 + 순서 counterbalance**로 재설계했다 —
  arm을 항상 null→k6→k3 순으로 두면 **단조 drift만으로 변이 규칙이 공짜로 성립**하기
  때문이다(이 저장소의 C2 감사가 `results/r0c/decode_knee_vs_ctx.sbatch:45`에서 잡은 것과 같은 형태).
  또한 (a) shim이 `is_decode()`로 게이트해 eager prefill 오염을 막고, (b) "모듈은 import되나
  엔진은 안 씀"을 잡는 **wiring 양성대조**를 넣고, (c) L5가 경고한 유일한 셀
  (dual-worker × break>0)을 채우고, (d) 3% 게이트를 **신뢰하한 기준**으로 바꿨다
  (n=1로 3% 점추정을 판정 문턱으로 쓰는 건 게이트 #3의 오용).

두 프로브 모두 **도구 타당성 판정이지 성능 판정이 아니다**(2026-08-20 job 886718 선례 문구).
`.sbatch`는 작성했고 **제출하지 않았다**: `workspace/engine-port/results/bcg_probe/`.
P0-A는 `--selftest-analyzer` 13/13 + **`--selftest-mutants` 5/5** PASS — 변이는 주석 속
사고실험이 아니라 가드를 실제로 삭제한 소스를 실행해 판정이 뒤집히는지 assert하는
하네스다(적대 감사가 원래 버전에서 **반증 불가능한 검사 1건**을 실측으로 잡아냈다).
**예약 기준 GPU ≈3.1 GPU-hr**(smid R0 0:15 + P0-A 0:20 + P0-B 2:30).

**부수 발견 (F16)**: SGLang의 1급 훅 확장점 `--forward-hooks`로는 캡처된 그래프 안에
break를 넣을 수 없다 — `model_runner.py:656` 캡처가 `:666` 훅 등록보다 **먼저** 실행된다.
주입은 캡처 이전 지점(`ModelRunner.load_model`, 정의 `:1072`)에서 해야 한다.

---

## 금지 문구

1. **"BCG가 layer-aware를 되살린다 / per-window 트랙을 GO로 바꾼다"** — 잔차가
   **윈도우수 무관·모델 독립**으로 등재돼 있고(`CONSENSUS.md:1391`), BCG는 그 축을 안 건드린다.
   ★반대로 **"F8/34×가 상한을 측정했다"고도 쓰지 말 것** — 둘 다 cgOFF·n=1 기반이다.
2. **"BCG가 HE0를 뒤집을 수 있다"** — 死因 불일치.
3. **S3의 최적점 이동**(decode 그래프화가 최적을 d24→d16으로 밀었으니 prefill 그래프화는
   반대로 되민다)은 **예측**이다. 방법론 게이트 #1(정책 주장은 반드시 서빙 실증) 전엔
   순위 진술로 쓰지 않는다. ⚠️ 그 **전제("d24→d16")부터** n=1·r3/r4 한정이고 r4 차이는
   2.7%로 게이트 #3 문턱 미만이며 **동일 격자 cgOFF 대조가 없어** rate 축과 cudagraph 축이
   분리되지 않는다 — 전제도 단언하지 말 것.
4. **업스트림 벤치(1.70× 등)를 이 기판의 기대값으로 인용 금지** — gpt-oss-120b / TP4 /
   4×GB300 / prefill-only다(교훈 #31).
5. **"gate #13/#16을 닫는다"** 계열 — 이 문서와 무관하며 기존 금지 배너 유효.
