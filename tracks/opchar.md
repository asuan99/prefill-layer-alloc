# TRACK: opchar — 연산자 단위 특성화   (worktree `prefill-layer-alloc-opchar`, branch `opchar`)

> 이 트랙의 현재 상태 유일본(60줄). 이력은 git, 세부는 링크. 정본 반영은 main의 promote.

## 목표
Bullet(ASPLOS'26) Fig 2 / Table 2 / Fig 3 대응: Nemotron-Nano-9B-v2 층 종류별 연산자의 green partition SM 스케일링. 특성화 전용(서빙·정책 주장 없음).
## cost model 기여
- 채우는 칸: CM-1·CM-2의 연산자 단위 항(`s_attn`, `s_ssm`, `s_mlp`의 커널 수준 근거)
- 이번에 알게 된 것: 격리 연산자 합 기준 SSM 비중, 메모리 바운드 연산자 44 SM 포화, GEMM 계단 = wave quantization — COSTMODEL CM-1·CM-2 반영 완료(2026-10-13, Safe wording 범위)
- promote 필요: 아니오
## 현재 상태
- full·ctrl·ctrl2 Job 완료, claims-auditor 감사 `AUDIT_opchar_2026-10-13.md`(C2 CONFIRMED(scoped), Table 2 GEMM 칸 REFUTED).
- VESSL 이미지 프로파일러 불가 → Triton launch 래핑 + CUDA graph kernel-node census로 대체.
## 다음 행동
1. attention L=1024 idle 칸: `--graph-census` prefill L=1024 at 108/104 짧은 Job
2. decode L2 재사용 대조 arm · prefill end-to-end forward 보정
## 사용자 결정 대기
- **`opchar` → main 병합** — 병합 전엔 정본 반영 보류 / 지금 병합 · 후속 Job 뒤
- **74 SM 이상 SM당 효율 하락 원인 조사** — 클럭 고정 불가 기판이라 설계 필요 / 조사 · 보류
## 실행 중·미회수 Job
- 없음
## 금지 서술 (링크만)
`workspace/engine-port/results/opchar/AUDIT_opchar_2026-10-13.md` "Must not cite"
## 최근 로그
- 2026-10-13: ctrl2 Job(다른 GPU) — 104→108 점프 = wave quantization 확인, 핸드오프
- 2026-10-12: full Job 33/33
