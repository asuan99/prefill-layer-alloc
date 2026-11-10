# TRACK: opchar — 연산자 단위 특성화   (worktree `prefill-layer-alloc-opchar`, branch `opchar`)

> 이 트랙의 현재 상태 유일본(60줄). 이력은 git, 세부는 링크. 정본 반영은 main의 promote.

## 목표
Bullet(ASPLOS'26) Fig 2 / Table 2 / Fig 3 대응: Nemotron-Nano-9B-v2 층 종류별 연산자의 green partition SM 스케일링. 특성화 전용(서빙·정책 주장 없음).
## cost model 기여
- 채우는 칸: CM-1·CM-2의 연산자 단위 항(`s_attn`, `s_ssm`, `s_mlp`의 커널 수준 근거)
- 이번에 알게 된 것: 격리 연산자 합 기준 SSM 비중, 메모리 바운드 연산자 44 SM 포화, GEMM 계단 = wave quantization — COSTMODEL CM-1·CM-2 반영 완료(2026-10-13, Safe wording 범위)
- 이번에 알게 된 것(2026-11-03): nsys 직접 설치로 VESSL에서 커널 시간 수집 가능(green·cudagraph 포함, 합성 부하). 카운터 불가는 유지. green stream graph replay 뒤 device sync 조기 반환, 정본 수치 변경 없음
- promote 필요: 아니오 (감사 반영 2026-11-10, 10차)
## 현재 상태
- full·ctrl·ctrl2 Job 완료, 감사 `AUDIT_opchar_2026-10-13.md`(C2 CONFIRMED(scoped), Table 2 GEMM 칸 REFUTED).
- 프로파일러 정정(2026-11-02 Workspace 프로브): nsys 2025.5.2(`/data/tools/nsight`)로 커널 레인 수집 가능(합성 부하, green·cudagraph 포함, graph 커널 분해는 `--cuda-graph-trace=node`). ncu·GPU metrics·`-pl`·`-lgc` 불가, torch.profiler 커널 0건. 결과 `results/vessl_gpu_obs_probe_2026-11-02/`, 워크플로 `scripts/vessl/profile_workflow.sh`(node 옵션 미반영).
- 감사 `results/opchar/AUDIT_sync_green_graph_2026-11-03.md`: green stream graph replay 뒤 `torch.cuda.synchronize()` 조기 반환 CONFIRMED(범위 한정), 정본 수치 변경 없음, 호출자 미추적 2곳.
- 기판 메모: VESSL 드라이버 580.178.04(10월 초 기록 580.105.08).
## 다음 행동
1. 호출자 미추적 2곳 코드 추적(GPU 0): `memory_pool.py:933-965,1657-1681`, `scheduler_update_weights_mixin.py:153`
2. attention L=1024 idle 칸: `--graph-census` prefill L=1024 at 108/104 짧은 Job
3. decode L2 재사용 대조 arm · prefill end-to-end forward 보정
## 사용자 결정 대기
- **`opchar` → main 병합** — 병합 전엔 정본 반영 보류 / 지금 병합 · 후속 Job 뒤
- **74 SM 이상 SM당 효율 하락 원인 조사** — 클럭 고정 불가 기판이라 설계 필요 / 조사 · 보류
- **device sync 후속 프로브** — 경합 실존과 "replay가 마지막 작업일 때만" 가설이 미정, 서빙 수치는 무관해 급하지 않음 / 호출자 2곳 코드 추적 먼저(GPU 0) · 프로브 1–3 짧은 실행 · 보류. 권고: 코드 추적 먼저
## 실행 중·미회수 Job
- 없음
## 금지 서술 (링크만)
`workspace/engine-port/results/opchar/AUDIT_opchar_2026-10-13.md` "Must not cite"
## 최근 로그
- 2026-11-07: 핸드오프 — 프로브 스크립트·결과를 main에서 opchar로 이동·커밋
- 2026-11-03: nsys 프로브·device sync 감사, 정본 수치 변경 없음
- 2026-10-13: ctrl2 Job(다른 GPU) — 104→108 점프 = wave quantization 확인, 핸드오프
- 2026-10-12: full Job 33/33
