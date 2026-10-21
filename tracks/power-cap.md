# TRACK: power-cap — 전력·에너지 관측   (worktree 없음, main `workspace/engine-port/results/power_cap/`)

> 휴면 트랙: `-pl` 가능한 기판 확보 전까지 관측만. 60줄 예산.

## 목표
GPU power cap의 개입 여지 평가. VESSL 컨테이너는 cap 설정 불가(실측) → 에너지·전력은 관측 지표로만.
## cost model 기여
- 채우는 칸: CM-4의 전력 채널(N2 Job-EA에서 C 창 64/64 cap 노출 — 간섭 크기와 교락), 에너지 지표 J/step
- promote 필요: 아니오
## 현재 상태
- 수동 재분석 3종 감사 반영, N2 전력 분석기 준비(`n2_decode_power/pc_n2_decode_power.py`), 미실행.
- N2 두 Job에 energy·power_samples 있음(원시 CSV는 gitignore·로컬). `sm_clk_samples.csv`는 헤더뿐(A100 버퍼 클럭 미지원 추정).
## 다음 행동
1. N2 Job-EA 데이터로 W/C/A2 창별 전력·J/step 기술 산출(판정 아님) — N2 F-1 결정과 묶어 판단
## 사용자 결정 대기
- **cap 처치용 기판** — VESSL 컨테이너 불가 / VESSL VM Cluster(H100↑, 선불) · 외부 root VM · 휴면 유지(권고)
- **B안(prefill 상태까지 보는 전용 서빙 Job)·J/req 보조 열·venue(워크숍급)** — 기판 결정 뒤에
## 금지 서술 (링크만)
메모리 `power-cap-track` §금지 문안
## 최근 로그
- 2026-10-21: (main 세션 확인) 지연 경보 원인 `36517fe3`(2026-10-20)은 main이 예전 미추적 power_cap csv(partition_power·sensor_bw)를 커밋한 정리 작업 — 새 측정·판정 없음, 상태 불변
- 2026-10-12: 세션 핸드오프(`handoff-report/session_handoff_2026-10-13.md` (B)절)
