# Hybrid-LLM PD-mux serving — 작업 지침 (workspace root)

이 워크스페이스의 본 프로젝트는 **`prefill-layer-alloc/`** 이다. Hybrid LLM
(attention+SSM: NemotronH / Zamba2 / Falcon-H1 / Granite-4)을 **SGLang v0.5.10 +
A100 green-context SM 분할**로 서빙하며, prefill↔decode multiplexing(PD-mux)의
자원 분할 정책을 연구한다. 대부분의 canonical 코드·결과는
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

## 환경 / 실행

- editable tree: `/scratch/ehmoon/whlee/sglang_engine_dev/python` · venv:
  `/scratch/ehmoon/whlee/sglang_engine_venv` · A100(108 SM), CUDA 13, torch 2.9.1.
- 실행 전:
  ```bash
  module load conda/pytorch_2.9.1_cuda13 cuda/13.0.2 gcc/15.2.0
  source /scratch/ehmoon/whlee/sglang_engine_venv/bin/activate
  cd /scratch/ehmoon/whlee/prefill-layer-alloc
  workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh   # PD-mux src + thread-local role patch 설치 + SHA-256 manifest
  python -m unittest discover -s workspace/engine-port/tests -v  # CPU-only 회귀
  ```
- SLURM: partition `amd_a100nv_8`, `--gres=gpu:1`. R2 캠페인은
  `PDMUX_CAMPAIGN=.../results/r2_eval/campaign.json sbatch --array=0-<N-1> .../scripts/r2_eval/r2_eval.sbatch`.
- **SLURM `--comment` 필수 (2026-08-12 18:00 KST 이후 뉴론 정책 — 없으면 제출 거부).**
  모든 job script에 아래 한 줄을 **`#SBATCH` 블록 안**(= 첫 실행 라인보다 위 — Slurm은
  첫 비주석 라인 이후의 `#SBATCH`를 무시한다)에 넣는다:
  ```bash
  #SBATCH --comment="field=efficientai;appl=pytorch"
  ```
  이 프로젝트 값은 **`field=efficientai`**(Efficient & Scalable AI Systems — PD-mux
  SM 분할 서빙 효율 연구) · **`appl=pytorch`**(SGLang은 showappl 목록에 없음; PyTorch
  기반이라 pytorch로 신고. `vllm`은 다른 엔진이므로 쓰지 않는다). 인터랙티브는
  `salloc --partition=amd_a100nv_8 --gres=gpu:1 --comment="field=efficientai;appl=pytorch"`.
  허용값은 로그인 노드 `showappl`로 확인. CLI에서 `--comment`를 덮어쓰지 말 것
  (script directive를 무효화해 거부된다). 2026-08-14 기준 저장소 내 `.sbatch` 97개
  전부 적용 완료 — 새 파일도 같은 형식으로 생성한다.
- 런타임 모드(env): `PDMUX_TRUE_DUAL_WORKER=1`, `PDMUX_R2_POLICY=fixed|generic|hybrid`,
  `PDMUX_MODEL_PROFILE=<json>`, `PDMUX_TELEMETRY_PATH`, `PDMUX_RUN_ID`, `PDMUX_WORKLOAD_ID`.
  (`PDMUX_DUAL_WORKER=1`은 R1 observer 재현 전용 — architecture separation 아님.)
- 루트의 SLURM `.out`/`.err`는 커밋하지 않는다. 결과는 `results/<campaign>/`에만 기록.

## 방법론 게이트 (필수 — 어겨서 결론 2번 뒤집혔음)

1. **정책 주장은 반드시 서빙 실증.** sim/micro-benchmark의 per-layer 이득을 실엔진 결론으로 쓰지 않는다.
2. **stationary ShareGPT r8 = 정책 비교 벤치 폐기**(static조차 ±1.3). 정책 비교는 **변화 trace**(rate 3↔12, 3라운드 평균).
3. **베이스라인 분산 먼저 측정, n≥4 없이 정책 결론 금지.** 3% 미만 차이는 headline 아님.
4. goodput = **request TTFT ≤ SLO AND request-내부 token-ITL p95 ≤ SLO**(mean-ITL은 secondary만).
5. dynamic 결과엔 **`switch_count` + split 체류분포 + TTFT/ITL p50/p95/p99** 병기.
6. goodput처럼 SLO 임계 지시함수는 **metric cliff**(TTFT≈SLO 경계)를 피해 측정. 용량 먼저 측정.
7. 여러 라운드 합칠 때 duration은 **합산**(`max()`는 3× 부풀림 — 실제 하네스 버그였음).
8. SLO를 바꿔 평가할 땐 기존 컨트롤러 **재스코어 금지 → 그 SLO로 재튜닝해 직접 측정**.

## 이 프로젝트용 에이전트 (`.claude/agents/`)

작업이 아래에 해당하면 해당 subagent에 위임한다.

- **result-analyst** — telemetry/campaign 결과의 goodput·percentile·paired bootstrap 계산 + "실재하는가" 판정.
- **claims-auditor** — 결론을 문서/논문에 쓰기 전 confound 목록으로 적대적 반증(read-only).
- **experiment-runner** — sbatch 제출·squeue 모니터·로그 triage·아티팩트 수집·완료 job on-demand 보고서 생성(옛 slurm-agent observer 대체).
- **engine-porter** — SGLang 패치 구현·검증·런타임 모드 배선(correctness gate 우선).
- **doc-steward** — 정본 위계 유지·폐기 격리·절대날짜·MEMORY.md 갱신.
- **venue-strategist** — related work + 시스템 학회 fit 정직 평가 + 추가 방향 정리.
- **git-committer** — 스테이징 검증·관례적 커밋 메시지·로컬 커밋(명시 지시 없이 push 금지).

전체 리뷰가 필요하면 **`review-lead` 스킬**이 위 리뷰 계열 에이전트(engine-porter·
claims-auditor·result-analyst·doc-steward)와 내장 리뷰툴을 라우팅·통합한다(모순 검출·
completeness·단일 전체그림 판정). 개별 에이전트 사이의 **연결점을 서술하는 유일한 주체.**

> **버전관리(2026-09-12)**: 이 파일과 `.claude/agents/*`·`.claude/skills/*`는 outer
> 워크스페이스(`/scratch/ehmoon/whlee`, **git 저장소 아님** — `.git`이 빈 디렉터리)에 살지만,
> 정본 사본은 `prefill-layer-alloc/tools/claude/`에 추적된다. 툴 파일을 고치면
> `tools/claude/sync_claude_tools.sh --import` 후 커밋하고, 다른 머신/클론에서 되살릴 때는
> `--install`. 드리프트 검사는 `--check`(테스트 `test_sync_claude_tools.sh` 10케이스).

## 세션 연속성 (`.claude/skills/`)

새 세션이 문맥을 이어받게 하는 닫힌 루프:

- **세션 시작** → `catch-up` 스킬. 정본·메모리·최근 git/결과/실행중 job에서 현재 상태를
  재구성해 브리핑(확정 결론은 재도출 않고 요약만).
- **세션 종료/체크포인트** → `handoff` 스킬. 이번 세션의 결정·측정·코드/문서 변경·열린
  항목을 `prefill-layer-alloc/handoff-report/session_handoff_<YYYY-MM-DD>.md` + 메모리에 저장(정본·메모리 반영은
  doc-steward 규율 위임). 이게 다음 세션의 catch-up이 읽을 대상이다.

## 메모리

장기 메모리 인덱스: `/home01/ehmoon/.claude/projects/-scratch-ehmoon-whlee/memory/MEMORY.md`.
프로젝트 상태가 바뀌면 doc-steward가 이 인덱스를 함께 갱신한다.
