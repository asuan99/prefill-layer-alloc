# Session handoff — 2026-10-13 (opchar track, branch `opchar`)

> 이 문서는 **worktree `~/Experiments/KISTI/prefill-layer-alloc-opchar` (branch `opchar`)** 에만 있다.
> 2026-10-12 사용자 지시로 이 세션의 변경을 main과 분리했다(main 체크아웃에 다른 세션의 미커밋
> 변경이 있어 launch.sh dirty-tree 검사가 막혔음). main의 `catch-up`은 이 브랜치를 자동으로 보지
> 않는다 — 시작점은 아래 "다음 세션 시작점".

## 이번 세션 요약

사용자가 Bullet(ASPLOS'26) Fig 2 / Table 2 / Fig 3과 같은 **연산자 단위 비교 측정**을 Hybrid LLM에서
원했다. 기존 저장소를 조사한 결과 같은 형태는 없었고(KISTI 시절 E1/E2/E3/E7·prefill_knee·r0c 등 조각만,
대부분 소형 모델·SDPA·MLP 없음·미감사), 새 캠페인 `opchar`를 만들었다: Nemotron-Nano-9B-v2의 층 종류별
연산자를 **엔진 커널 그대로 격리 실행**하고, green partition SM sweep으로 시간을 재고, 프로파일러가 막힌
VESSL 이미지에서 **카운터 없이** launch 설정을 얻는 두 경로(Triton launch_metadata 래핑, CUDA graph
kernel-node census)를 만들었다. 5개 VESSL Job, claims-auditor 감사 1회, 감사 해소 Job 1회까지 진행.
**특성화 전용 — 서빙/정책 주장 없음**(layer-type 정책 死·HE0·게이트 #13 불변).

## 결정·측정

- **계획**: `~/.claude/plans/figure2-table2-linear-cherny.md` (2026-10-12 승인). 목업 그림
  `reports/figures/plot_opchar_mockup.py` (MOCKUP, 데이터 아님).
- **VESSL 프로파일러 사실(실측)**: ncu `ERR_NVGPUCTRPERM` — LaunchStats 단독 포함 전부(ctrl2로 확정);
  nsys 미설치; torch.profiler/Kineto는 libcupti.so.13 로드·"Enabling GPU tracing" 후 "GPU trace is empty!"
  (CUPTI 오류 없음, LD_LIBRARY_PATH 무관, GPU 2대에서 동일) — 원인 미상.
  → 대체: `triton_launch.py`(Triton 커널 grid/regs/smem), `graph_census.py`(cuBLAS/FlashInfer 포함 전 커널,
  driver occupancy b). 두 경로는 Triton 14개 튜플에서 일치(감사 교차 확인).
- **사용자 결정**: Table 2 범위 = "Triton 훅 추가 후 full"(2026-10-12 AskUserQuestion).
- **full Job 결과**(98f3a8c): `results/opchar/full_pdmux-opchar-98f3a8c-20261012141951/` (ANALYSIS/*.csv|md).
  Fig 2(a) 격리 연산자 시간 합 기준 Mamba 층 ~66–71%(prefill L 1k–32k), decode SSM update 36→64%(B 32→128).
  유도 util은 하드웨어 utilization 아님.
- **ctrl Job**(fff9b1d): 60개 op 구성 × SM{44..104 green, 108 plain} — 분할에 따라 커널/grid 바뀌는 것 **0**.
  decode GEMM 시간 계단이 recorded wave 수와 일치.
- **claims-auditor 감사**(2026-10-13): `results/opchar/AUDIT_opchar_2026-10-13.md`. C2 CONFIRMED(scoped);
  C3 PLAUSIBLE(strong); C1/C4/C6 PLAUSIBLE(조건부); C5 split — **full Job Table 2 GEMM 칸(가정 타일) REFUTED**
  (배너 부착), attention idle 칸 인용 금지, Bullet 재현 테스트=항등식. 안전 문구·인용 금지 목록 수록.
- **ctrl2 Job**(050483c, **다른 GPU** UUID 6ca4…): 등록 예측 성립 — 106 green에서 105-CTA split-K decode
  GEMM이 108 plain과 0.6% 이내(o_proj 0.0471 vs 0.0468 ms, 104 green 0.0688) → 104↔108 점프 = wave
  quantization. cuBLASLt eager도 104 green에서 `smCountTarget=108`, 12개 shape 동일 algo. 순서 섞어도
  74 SM부터 SM당 효율 하락 유지(power-cap 이벤트는 ≥74 SM에서만, 클럭 중앙값 불변 → 원인 미확립).
  해소 내용은 AUDIT 파일 "Resolution run" 절.

## 코드·문서 변경 (전부 branch `opchar`에 커밋, main 무변경)

- `workspace/characterization/experiments/opchar/` — `ops_spec.py`(연산자 목록·분석 FLOP/byte), `builders.py`
  (엔진 커널 재생), `worker.py`(분할별 측정), `trace_parse.py`(Kineto 파서·sm80 occupancy), `triton_launch.py`,
  `graph_census.py`, `abx.py`(104g/106g/108p 교차 측정), `probe.py`, `run_opchar.py`(probe/full/ctrl/ctrl2).
- `workspace/characterization/experiments/common/wave_model.py` — `wave_quantization`, `gemm_tile_blocks` 추가.
- `workspace/shared/configs/models.yaml` — `nemotron_nano_9b_v2`(config.json @ dc0661c).
- `workspace/engine-port/results/opchar/` — `opchar_job.sh`, 5개 Job 결과, `AUDIT_opchar_2026-10-13.md`.
- `workspace/engine-port/tests/test_opchar.py` — 23개 CPU 테스트(변이 검사로 4개 mutant 사살 확인).
- `reports/figures/plot_opchar.py`(Fig 2/Table 2/Fig 3/wave-step/memcpy; PNG는 gitignore, 로컬 생성).
- 메모리: `memory/opchar-track-2026-10-12.md` + MEMORY.md 인덱스 1줄.

## 열린 항목 / 다음 세션 시작점

1. **attention L=1024 idle 칸**: driver b 미측정(감사가 b=2→1이면 25.9%→1.2% flip 지적). ctrl2에 L=1024
   attention census가 없었다 → 짧은 Job으로 `--graph-census` prefill L=1024 at 108/104.
2. **decode L2 재사용**(graph replay 연속 10회, flush 없음) 대조 arm.
3. **74 SM 이상 SM당 효율 하락 원인**(power-cap 관련성 미확립) — 클럭 고정 불가 기판이라 설계 필요.
4. **prefill 전체 forward 시간 보정**(격리 연산자 합 vs end-to-end).
5. **branch `opchar` → main 병합 여부**와 **정본 반영**(PROJECT_STATUS/reports)은 사용자 결정. doc-steward 정본
   반영은 병합 결정 후로 보류했다(main은 다른 세션이 작업 중).
6. (선택) cuBLASLt `smCountTarget`을 분할 크기로 지정하면 wave 계단이 줄어드는지 — 개선 수단 후보, 미측정.

## 미완·주의

- 모든 수치: 격리 커널·랜덤 가중치·Job당 n=30 중앙값, GPU 2대(32e4…/6ca4…). 서빙·정책·KISTI 비교 금지.
- 인용은 AUDIT 파일 "Safe wording"만, 그 스코프로만. "green context 비용 없음"을 PD-mux/전환 비용으로 확대 금지.
- 커밋 fff9b1d 메시지의 "prefill stays smooth"는 틀림(감사 지적, AUDIT에 정정 기재).
- 이 세션 Job은 모두 회수 완료. **실행 중 Job 3개(`pdmux-multigpu-*`)는 다른 세션 것**(이 세션 무관).
  Workspace `pdmux-probe`(A100)·`pdmux-build`(CPU)는 standby — 이 세션에서 켜지 않았다.

## GPU 실행 기록 (VESSL, 캠페인 `opchar`, 이미지 `ghcr.io/asuan99/sglang-runtime@sha256:1e0d335b2e68…`)

| job name | slug | commit | 상태 | fetch(DONE+sha256) | cost.txt |
|---|---|---|---|---|---|
| pdmux-opchar-a9d55ad-20261012140018 | job-3fdczheaizqq | a9d55ad | exit 1 (green worker 버그) | OK | container_hours=0.048 cost_usd_lower_bound=0.07 |
| pdmux-opchar-dd405fe-20261012141151 | job-ztoebt1cst7q | dd405fe | exit 0 | OK | container_hours=0.056 cost_usd_lower_bound=0.08 |
| pdmux-opchar-98f3a8c-20261012141951 | job-jae8pbinkf2r | 98f3a8c | exit 0 | OK | container_hours=0.204 cost_usd_lower_bound=0.30 |
| pdmux-opchar-fff9b1d-20261013062641 | job-oghgldb5cf3g | fff9b1d | exit 0 | OK | container_hours=0.079 cost_usd_lower_bound=0.12 |
| pdmux-opchar-050483c-20261013065012 | job-h0ro98uvs4gx | 050483c | exit 0 | OK | container_hours=0.110 cost_usd_lower_bound=0.16 |
