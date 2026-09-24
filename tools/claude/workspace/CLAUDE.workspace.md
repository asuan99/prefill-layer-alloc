# Hybrid-LLM PD-mux serving — 작업 지침 (workspace root)

이 워크스페이스의 본 프로젝트는 **`prefill-layer-alloc/`** 이다. Hybrid LLM
(attention+SSM: NemotronH / Zamba2 / Falcon-H1 / Granite-4)을 **SGLang v0.5.10 +
green-context SM 분할**로 서빙하며, prefill↔decode multiplexing(PD-mux)의
자원 분할 정책을 연구한다. **지금까지의 모든 측정 기판은 A100 80GB 1장(108 SM)이다**
(2026-09-17 이양 전 KISTI Neuron). 대부분의 canonical 코드·결과는
`prefill-layer-alloc/workspace/engine-port/` 아래에, **현재 reports는 최상위
`prefill-layer-alloc/reports/`** 에 있다(2026-07-24 승격).

> 세션 시작 시 결론을 재도출하지 말 것. 아래 정본을 먼저 읽는다.

## 정본 위계 (충돌 시 위가 우선)

1. **`prefill-layer-alloc/PROJECT_STATUS.md`** — 프로젝트 전체 현재 상태의 유일 정본.
2. **`prefill-layer-alloc/reports/paper/`** — 현재 claim/evidence/roadmap
   (`CLAIM_EVIDENCE_MATRIX.md`, `EXPERIMENT_ROADMAP.md`, `R1_REANALYSIS.md`).
3. **`.../reports/CONSENSUS.md`** — dual-worker/R2 이전까지 확정된 phase-separation·
   layer-granular negative result·entanglement·single-worker dynamic 결과의 정본.
4. `prefill-layer-alloc/workspace/engine-port/RESUME.md` — 환경·패치·부팅·재현.
5. `deprecated/` (2026-07-24 통합 격리: `reports/historical_root/` = 옛 루트 `reports/`,
   `PAPER_RESULTS.md`·`FINAL_SUMMARY.md` 포함 · `reports/quarantine_root/` · `reports/
   quarantine_engine_port/` · `results/arxiv/` · `logs/`) = **이력용**. 현재 결론과 충돌
   가능, **새 분석의 근거로 재사용 금지**. 상세는 `deprecated/README.md`.

## 현재 결론 (재도출 금지 — 이미 확정된 것)

- PD 분리(agnostic PD-mux) 자체는 4모델 전부서 fused를 이김. **운영점 = cudagraph-ON**.
- **layer-type 런타임 정책은 全형태 死**(서빙 직접 측정으로 반증; coordinated per-type
  TPOT 42→124ms). 시뮬레이터의 per-layer 이득(1.37–2.02×)은 실엔진으로 전이되지 않음.
- **단일-GPU 동적 제어(SLO-aware/binding-first/gate)는 best decode-heavy static을 못 넘음**
  — 관대(3s)·tight(chat 300/50ms) SLO 양쪽 확정. 실패 원인 = switch overhead 아님,
  **positioning + entanglement**(decode 굶김→ITL↑→batch 정체→admission 차단→TTFT 폭발).
- **실무 기본값 = peak decode 부하 기준 decode-heavy static 고정**(검증된 profile 없으면 agnostic).
- **미검증 라이브 가설(R2)**: H-Architecture(true dual-worker로 coupling 감소),
  H-Policy(offline hybrid-profile decode-floor 예측이 generic/static보다 나은 SLO goodput).
  구현 완료 ≠ 성능 주장 성립.

## 환경 / 실행 (2026-09-18 개정 — 로컬 개발 + 원격 GPU 실행)

- **작업 루트 = 로컬 PC `~/Experiments/KISTI`**. 이 파일과 `.claude/`가 여기 있고 저장소는
  `~/Experiments/KISTI/prefill-layer-alloc`이다. 이전 머신(KISTI Neuron
  `/scratch/ehmoon/whlee`)은 2026-09-17 이양으로 **읽기 전용 보관소** — 거기서 커밋하지 않는다.
  이양 경위·저장 목록은 `handoff-report/session_handoff_2026-09-17.md`와
  `migration_inventory_2026-09-17.md`.
- ★**이 머신에서는 서빙 실험을 돌리지 않는다.** 로컬 GPU(RTX 5060 Ti 16GB, Blackwell
  compute capability 12.0[major 12])에서는 실행하지 않는다. ★**정정(2026-09-21,
  doc-steward — `handoff-report/gpu_rental_checklist_2026-09-18.md` §0 1차 소스 확인 +
  claims-auditor `workspace/engine-port/results/r2_eval/e2_sticky_prereg/
  VERDICT_e2_substrate_portability_2026-09-18.md` Q4 독립 검증 `CONFIRMED`)**: 이전 판은
  사유를 "PD-mux 경로가 **엔진 단계에서 거부된다**(dev tree
  `pdmux_context.py:get_arch_constraints`는 major 6–9만 지원, major 10+ → `ValueError`)"로
  적었으나, 이 함수는 **자동 격자(`divide_sm`) 경로에서만 호출되고**(설치 트리 전역
  1곳) 이 프로젝트의 **`manual_divisions` 계열 캠페인**(E1/S2/λ0/E2 등)에서는 이 분기가
  발화하지 않는다. ★**스코프 정정(2026-09-24, 사용자 승인 — `reports/audit/
  2026-09-22_scope_lineage/REPORT.md` §6 I-1)**: 이전 판의 "전적으로 `manual_divisions`
  경로(현행 `*pdmux*.yml` 59/59)"는 **거짓**이다 — YAML 파서 전수로 **55/59**이고, 나머지
  4개(`pdmux_a100_smoke.yml`, byte-identical)는 키가 없어 **자동 격자(`divide_sm`)**를 쓴다
  (grep이 "없음을 설명하는 주석"에 걸려 "있음"으로 셌다). 이 config를 쓰는 job script 27개에
  **P1.7 4모델·P1-opint 운영점·p1_gates/gate2(전환 비용 원 캠페인)**가 포함된다 ⇒ 그
  캠페인들에서는 `get_arch_constraints`가 **발화했다**(A100 cc8이라 정상 반환, 결과 불변).
  다른 기판에서 그 캠페인을 재현하면 아키텍처 상수가 격자를 직접 바꾼다. 로컬 실행 불가의
  **실효 사유**는 (i) 설치된 `sgl_kernel`에 sm120 대응 빌드가 없어 import 자체가
  깨짐(torch ABI 불일치, 2026-09-18 로컬 실측 `undefined symbol:
  _ZNK3c106SymInt22maybe_as_int_slow_pathEv`) (ii) 16GB로는 9B 모델이 안 올라감
  (iii) **게이트 1**(기판이 다르면 수치가 주장 근거가 못 됨) 세 가지다 —
  **결론(로컬에서 서빙 실험을 돌리지 않는다)은 불변**, 바뀌는 것은 "왜 안 되는가"의
  정확도뿐. ★단 이것을 "그러므로 major 10+에서 돌아간다"로 확장하면 **거짓**이다 —
  제약은 사라지는 게 아니라 `sgl_kernel.spatial.create_greenctx_stream_by_value` →
  `cuGreenCtxCreate`(CUDA green context API) 층으로 이동할 뿐이고, 그 층이 실제로 어느
  compute capability까지 어떤 SM 입도를 주는지는 **미실측**이다.
- **로컬에서 하는 일 = 검증·분석·문서**(전부 CPU):
  ```bash
  cd ~/Experiments/KISTI/prefill-layer-alloc
  python -m unittest discover -s workspace/engine-port/tests -v          # CPU 회귀
  python3 workspace/engine-port/scripts/discipline/check_citation_stops.py  # 커밋 전 필수
  /usr/bin/bash tools/claude/test_sync_claude_tools.sh                   # 도구 동기화 검사
  ```
  아카이브 telemetry 재집계·goodput/percentile/paired bootstrap 재계산·그림 재생성도 전부
  CPU다(`workspace/engine-port/benchmarks/pdmux_eval/analyze.py`, `reports/figures/*.py`).
  원시 telemetry는 git 밖이므로 이양 번들의 `C_results_untracked/`를 저장소 원위치에 푼다.
- **GPU 실행 = 대여 서버(업체 미정)**. 확정되면 `experiment-runner`에 원격 실행 규약을 적는다.
  현재 계획·이식 범위는 `handoff-report/migration_plan_local_dev_remote_gpu_2026-09-18.md`.
  **대여 전 확인**: A100(sm80, 4 SM 단위)이면 기존 격자·λ\*와 직접 비교 가능 · H100(sm90)은
  upstream `get_arch_constraints` 코드 상수로는 8 SM 단위·132 SM이나 ★**이 상수는
  `divide_sm`(자동 격자) 경로에만 적용되고 `manual_divisions` 계열 캠페인의 실제
  입도는 이 상수를 타지 않는다**(2026-09-21 정정, 위 항목과 동일 근거; ★2026-09-24 스코프
  정정: 자동 격자 config[`pdmux_a100_smoke.yml`]를 쓰는 P1.7·P1-opint·gate2 계열을 재현하면
  이 상수가 **직접** 격자를 정한다 — 위 I-1 참조) — 총 SM 132는
  그대로(하드웨어 사실)지만 **실제 입도는 대여 후 green-context probe로 실측해야 한다**
  (`gpu_rental_checklist_2026-09-18.md` §3 B3), 코드 상수를 실측 대신 인용하지 말 것.
  격자 재설계 + λ0 재측정 전제는 불변 · Blackwell은 실제 동작 여부가 **미지**(단순
  "코드 확장 필요"가 아니라 `sgl_kernel`에 해당 아키텍처 빌드가 있는지 + green context가
  동작하는지가 관건, 미실측) · 드라이버 CUDA ≥ 12.4 · MIG/vGPU 불가(전 GPU 필요).
- 엔진 트리 재구성(어느 머신이든 동일):
  ```bash
  SGLANG_ENGINE_DEV=<sglang>/python workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh <manifest.sha256>
  patch -p1 -d <sglang>/python < workspace/engine-port/env/devtree_manual_edits.patch   # sync가 안 하는 수동편집 5파일
  ```
  패키지 고정값 `workspace/engine-port/env/venv_packages_2026-09-17.txt`, py3.14 shim
  `workspace/engine-port/env/sitecustomize.py`.
- 런타임 모드(env): `PDMUX_TRUE_DUAL_WORKER=1`, `PDMUX_R2_POLICY=fixed|generic|hybrid`,
  `PDMUX_MODEL_PROFILE=<json>`, `PDMUX_TELEMETRY_PATH`, `PDMUX_RUN_ID`, `PDMUX_WORKLOAD_ID`.
  (`PDMUX_DUAL_WORKER=1`은 R1 observer 재현 전용 — architecture separation 아님.)
- 결과는 `results/<campaign>/`에만 기록한다. 원격 실행은 **회수(rsync)+sha256까지가 한 작업**이다.
  스케줄러가 없는 서버에서는 배열 작업 대신 `PDMUX_RUN_INDEX=i bash <script>` 루프를 쓴다
  (활성 스크립트에 이미 폴백이 있다: `scripts/r2_eval/r2_eval.sbatch:55`).

### 부록: KISTI Neuron 규약 (2026-09-17까지의 이력 — 새 실행에는 적용하지 않는다)

과거 캠페인 재현·로그 해석 때만 참조한다. partition `amd_a100nv_8`, `--gres=gpu:1`,
`module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0`, venv
`/scratch/ehmoon/whlee/sglang_engine_venv`. 모든 job script는 `#SBATCH` 블록 안에
`#SBATCH --comment="field=efficientai;appl=pytorch"`가 있어야 제출됐다(2026-08-12 정책,
없으면 큐 진입 자체가 거부 → `.out`/`.err`가 생기지 않는다). 2026-09-15부터는 계정 사유로
전 파티션 제출이 거부돼 실행이 멈췄다(만료 vs 한도초과 미확정).

## 방법론 게이트 (필수 — 어겨서 결론 2번 뒤집혔음)

1. **정책 주장은 반드시 서빙 실증.** sim/micro-benchmark의 per-layer 이득을 실엔진 결론으로 쓰지 않는다.
2. **stationary ShareGPT r8 = 정책 비교 벤치 폐기**(static조차 ±1.3). 정책 비교는 **변화 trace**(rate 3↔12, 3라운드 평균).
3. **베이스라인 분산 먼저 측정, n≥4 없이 정책 결론 금지.** 3% 미만 차이는 headline 아님.
4. goodput = **request TTFT ≤ SLO AND request-내부 token-ITL p95 ≤ SLO**(mean-ITL은 secondary만).
5. dynamic 결과엔 **`switch_count` + split 체류분포 + TTFT/ITL p50/p95/p99** 병기.
6. goodput처럼 SLO 임계 지시함수는 **metric cliff**(TTFT≈SLO 경계)를 피해 측정. 용량 먼저 측정.
7. 여러 라운드 합칠 때 duration은 **합산**(`max()`는 3× 부풀림 — 실제 하네스 버그였음).
8. SLO를 바꿔 평가할 땐 기존 컨트롤러 **재스코어 금지 → 그 SLO로 재튜닝해 직접 측정**.

(항목 번호는 이 목록 고유 축이다 — `PROJECT_STATUS.md`의 게이트 #N과 별개.)
**기판 주의**: GPU 세대·SM 수·드라이버가 바뀌면 기존 격자·λ\*는 그대로 성립하지 않는다.
옮긴 기판의 수치를 옛 기판 결론과 섞지 말고, 재측정 여부는 사용자 결정 + 규칙층 감사로 넘긴다
(게이트 1의 연장 — 새 규칙 번호를 만들지 않는다).

## 이 프로젝트용 에이전트 (`.claude/agents/`)

작업이 아래에 해당하면 해당 subagent에 위임한다.

- **result-analyst** — telemetry/campaign 결과의 goodput·percentile·paired bootstrap 계산 + "실재하는가" 판정.
- **claims-auditor** — 결론을 문서/논문에 쓰기 전 confound 목록으로 적대적 반증(read-only).
- **experiment-runner** — 로컬 CPU 검증(회귀·규율 도구·아카이브 triage·보고서)과, 대여 GPU가
  정해지면 원격 실행(ssh/rsync)·아티팩트 회수 담당. SLURM 규약은 이력 부록.
- **engine-porter** — SGLang 패치 구현·검증·런타임 모드 배선(correctness gate 우선).
- **doc-steward** — 정본 위계 유지·폐기 격리·절대날짜·MEMORY.md 갱신.
- **venue-strategist** — related work + 시스템 학회 fit 정직 평가 + 추가 방향 정리.
- **git-committer** — 스테이징 검증·관례적 커밋 메시지·로컬 커밋(명시 지시 없이 push 금지).

전체 리뷰가 필요하면 **`review-lead` 스킬**이 위 리뷰 계열 에이전트(engine-porter·
claims-auditor·result-analyst·doc-steward)와 내장 리뷰툴을 라우팅·통합한다(모순 검출·
completeness·단일 전체그림 판정). 개별 에이전트 사이의 **연결점을 서술하는 유일한 주체.**

> **버전관리(2026-09-12, 2026-09-18 경로 일반화)**: 이 파일과 `.claude/agents/*`·
> `.claude/skills/*`는 **작업 루트**(저장소의 부모 디렉터리, git 저장소 아님)에 살지만,
> 정본 사본은 `prefill-layer-alloc/tools/claude/`에 추적된다. 툴 파일을 고치면
> `tools/claude/sync_claude_tools.sh --import` 후 커밋하고, 다른 머신/클론에서 되살릴 때는
> `--install`. 드리프트 검사는 `--check`(테스트 `test_sync_claude_tools.sh` 10케이스).
> 작업 루트는 기본적으로 저장소의 부모로 자동 결정되며 `CLAUDE_TOOLS_LIVE=<dir>`로 덮어쓴다.

## 세션 연속성 (`.claude/skills/`)

새 세션이 문맥을 이어받게 하는 닫힌 루프:

- **세션 시작** → `catch-up` 스킬. 정본·메모리·최근 git/결과에서 현재 상태를
  재구성해 브리핑(확정 결론은 재도출 않고 요약만).
- **세션 종료/체크포인트** → `handoff` 스킬. 이번 세션의 결정·측정·코드/문서 변경·열린
  항목을 `prefill-layer-alloc/handoff-report/session_handoff_<YYYY-MM-DD>.md` + 메모리에 저장(정본·메모리 반영은
  doc-steward 규율 위임). 이게 다음 세션의 catch-up이 읽을 대상이다.

## 메모리

장기 메모리 인덱스: `~/.claude/projects/-home-wonho-Experiments-KISTI/memory/MEMORY.md`
(디렉터리 이름 = 작업 루트 경로의 `/`를 `-`로 바꾼 slug. 이전 머신 slug는
`-scratch-ehmoon-whlee`였고 그 내용을 그대로 옮겨 왔다).
프로젝트 상태가 바뀌면 doc-steward가 이 인덱스를 함께 갱신한다.
