# 세션 핸드오프 — 2026-10-04 (VESSL 실행 개시: 기판 점검·λ0 재앵커·E-1 사전등록, 우선순위 재정렬)

> 직전: `session_handoff_2026-10-02.md`(2차 追記 포함). 이 세션의 GPU 지출은 VESSL 약 $4 미만이다(잔액 $413.73 → $391.39, org 공유).
> 새 성능 판정 0 · Claim 등급 변경 0 · 정본 CONSENSUS rev86.

## 1. 이번 세션 요약

VESSL Cloud 실행 기반을 실제로 열었다.
- SSH 키를 등록하고 CPU Workspace에서 엔진 설치, 모델·trace 적재, B2 회귀를 마쳤다.
- A100 Workspace에서 기판 점검(B0·B3·B4)을 했다. 첫 실행은 이미지에 libnuma1이 없어 FAIL했고, 수정 후 전부 PASS했다.
- 첫 Job(B5)으로 공개 이미지 → bundle → 엔진 설치 → GPU → 반출 → fetch 파이프라인 전 구간을 확인했다.

그 위에서 두 가지를 진행했다.
- **λ0 재앵커**(Nemotron-Nano-9B-v2): 사전등록 → 규칙층 감사 → 실행 → 결과 분석 → 결과 감사 → 정본 rev86 반영까지 마쳤다.
- **E-1**(cap 48 vs 192, Zamba2): 사전등록 rev1.1과 규칙층 감사까지 준비했다.

세션 끝에 사용자가 우선순위를 바로잡았다. "Zamba2를 왜 했나 — Nemotron-H에서 성립점을 먼저 모두 찾아야 한다." 이에 따라 E-1은 보류를 권고했고, 실험 사다리를 N0–N6으로 다시 정리했다.

## 2. 결정·측정

**사용자 결정**
- 진행 순서는 "1(λ0 재앵커) → 2(E-1)"로 받았다. 그러나 세션 말미에 **주 모델 Nemotron-H 사다리 우선**으로 정정됐다(메모리 `nemotron-first-ladder.md`).
- E-1 보류는 내 권고다. 사용자가 확정하지는 않았다.

**측정 (VESSL A100-SXM4-80GB, driver 580.105.08)**

기판 점검 (`results/vessl_substrate_probe_2026-10-04/`, `results/b5_smoke/`)
- 격자 (92,16)(84,24)(74,34)(64,44)(16,92): 요청한 SM 수 = 실현 %smid 집합 크기, 서로소, 동시 실행 격리, cudagraph replay 중 격리 유지(P0-A, KISTI와 같은 판정).
- 입도: 최소 4, 2 SM 단위 올림(실측).
- 부팅 스모크: Nano-9B PD-mux + cudagraph ON, 12/12 응답.

λ0 재앵커 (`results/r2_eval/lambda0_vessl/`, 결과 감사 판정서 `VERDICT_result_lambda0_vessl_2026-10-04.md` — 死因 0)
- 두 shape 모두 `KNEE_BRACKETED` + `SEED_REPEAT_HOLDS`. λ\*\_VESSL은 A ≈2.99, B ≈0.690 req/s.
- §4-4 규칙 결과: KISTI 908623과 **구별되지 않는다**(|Δ| A 1.95%, B 0.97% < 3%).
- 산포: 한 Job·한 노드 top rung n=4, A CV 0.26%, B CV 0.06%. LV-6 범위로만 쓴다.
- 스로틀: SW power cap(0x4) 단독, 샘플의 17.8%.
- CPU: 가시 코어 96이지만 cgroup quota는 11 CPU. CFS 스로틀은 기록하지 못했다.
- 인용 금지 신설 LVR-1…4: 2%/1% 낮다, Δ는 power cap 때문, SLO도 같다, n=4 SD를 설계에 사용 — 모두 금지.
- λ\*\_VESSL(A)도 D44 용량이 아니다(decode 91%가 비분할 (0,108)). λ0R-8(iii)은 열려 있다.

**감사 (전부 claims-auditor)**
- λ0 VESSL 규칙층: GO-with-caveats(LV-1…14)
- λ0 결과: 死因 0, 정정 R-1…R-7
- E-1 rev1 규칙층: GO-with-caveats(E1C-1…9). Monte Carlo 검출력 예보 포함 — 6블록에서 양방향 결론 도달 가능, VANISHED ≤1%, 4블록이면 UNRESOLVED 최대 67%.

## 3. 코드·문서 변경 (전부 커밋, 미push)

커밋 `a2b4176` … `afa06e9`, 이번 세션 17개. 주제별 목록은 `handoff-report/experiment_rationale_and_git_history_2026-10-04.md` §3에 있다.
- `scripts/vessl/`
  - 신규: `stage_data.sh`(refs/main 고정), `substrate_probe.sh` + helper + 테스트 29, `b5_smoke_job.sh`
  - `launch.sh` 수정: 비0 종료여도 Job 존재를 확인한 뒤 원장 기록, tag 밑줄→하이픈
  - `fetch.sh` 수정: slug 조회, staging 경로
  - `job_entry.sh` 수정: GpuIdle 비트 제외, cgroup cpu.max·cpu.stat 기록
- 이미지: `Dockerfile.sglang-runtime`에 libnuma1 추가 → `ghcr.io/asuan99/sglang-runtime@sha256:1e0d335b…`(공개). `vessl.env` 완성(spec `resourcespec-a100x1` $1.48/h, 볼륨 slug, digest).
- λ0 VESSL: `results/r2_eval/lambda0_vessl/`(사전등록·러너·plan·판정서 2·결과·분석), 테스트 23.
- E-1: `results/e1_cap_vessl/`(사전등록 rev1.1, 러너, plan, 판정서), 테스트 35+8.
  - 엔진: `src/multiplex/multiplexing_mixin.py`에 `PDMUX_PHASE_EVENTS` 추가(기본 OFF, 줄 이동 0). 로컬 dev tree도 sync했다.
- 정본 rev86(PROJECT_STATUS·CONSENSUS·ROADMAP·CLAIM 매트릭스) — doc-steward가 반영했다.
- 운영 문서 `vessl_operating_model_2026-10-02.md` §9–§11.
- 계획 문서 `experiment_rationale_and_git_history_2026-10-04.md`: N0–N6 사다리, E-1 보류 자기 정정, git 이력.
- 메모리: `vessl-operating-model.md` 갱신, `nemotron-first-ladder.md` 신설, 교훈 267–269(doc-steward).
- 의도적 미추적: `he0_realized_2026-10-01.json`.

## 4. 다음 세션 시작점 (사용자 지시)

**① 실험 설계의 선수관계 정리**
- 대상: N1(E2 sticky VESSL 이식), N2(3-a decode 프로파일), N3(게이트 #6 SLO 용량 격자), N4(3-b/긴 문서 — λ0 shape B로 충족됐는지), N5(S0′/A11), N6(S1→5-a→5-b), 선택 실험(λ0 Job 2, CPU quota 대조), 보류(E-1).
- 각 실험에 대해 정리할 것:
  - 무엇을 입력으로 요구하는가(예: N3은 λ\*\_VESSL 정규화 격자와 N1의 정규화 가능 여부, N5는 N2 프로파일)
  - 무엇을 산출하는가
  - 설계 단계에서 이미 다른 실험 결과에 의존하는 상수가 있는가(부하점, 격자, BOOT_S, SLO)
- 산출물: DAG 형태의 선수관계표. 출발점은 계획 문서 §2 표와 anchor §8.
- 확인할 의문:
  - N1과 N2가 정말 독립인가 — 3-a 격자가 sticky 여부에 의존하는가
  - N3가 N1 결과(λ\*(A)의 분할 의존)를 기다려야 하는가
  - 3-b가 λ0 shape B로 대체되는가

**② 관련 실험들의 실행 정리**
- 실험마다 정리할 것:
  - 사전등록 존재 여부(E2는 KISTI rev3 + 이식 追記 필요, 3-a·게이트 #6는 미설계)
  - 하네스 (λ0 이식 패턴 재사용: digest 강제·wall-clock 집행기·resume·cgroup·clk 기록)
  - 모델·데이터: Nano-9B-v2와 trace는 적재 완료, 긴 문서 trace는 미확인
  - 예상 GPU-h·비용, Job 분할
  - 엔진 트리 기준점: `0922512` 이후 mixin 변경 — E2 등 908623 동일 트리를 요구하는 등록이면 `1d47d3e` 기준 제출이나 追記가 필요
  - 감사 순서
- 산출물: 실행 순서·예산·블로커 표.

**대기 중인 사용자 결정**
- E-1 보류 확정 여부(내 권고는 보류)
- Zamba2 가중치 삭제 여부
- push 방식((a) 그대로 권고)
- 예산 시나리오·크레딧
- E2 새 OVERRIDE(기존은 KISTI 기판 이식성 REFUTED로 무효)

## 5. 미완·주의

- **E-1 사전등록의 미결정 항목**: 블록 수·D arm·R4/R6/R8 미결정. 보류되면 `BOOT_S` 等 λ0 자리표시자 追記도 미작성으로 남는다.
- **엔진 트리 분기**:
  - `0922512`부터 `multiplexing_mixin.py`가 908623과 다르다.
  - λ0 재실행(Job 2)은 `1d47d3e`에서 제출해야 한다.
  - E2(KISTI rev3)도 908623 계열 트리를 전제했다면 같은 문제가 생긴다 — 다음 세션 ②에서 확인한다.
- **org Burn Rate $0.30/h**: 우리 Workspace 2개는 모두 stopped/standby다. 다른 팀 사용분이거나 정산 지연으로 보이지만 미확인이다. 계속 보이면 콘솔에서 확인한다.
- **Workspace 재시작 시 `/opt` 초기화**: 볼륨은 유지된다. Workspace 작업은 매번 bundle clone과 `install_runtime.sh`부터 시작한다. ssh 세션은 이미지 ENV를 상속하지 않으므로 PATH 등을 export해야 한다.
- vesslctl 토큰 만료: 2026-10-05 14:44 KST.
- `FLASHINFER_WORKSPACE_BASE` 변수 적용 여부는 미검증이다.

## 6. GPU 실행 기록 (VESSL)

| 종류 | 이름/slug | 커밋 | 이미지 | 상태 | 회수 | 비용 |
|---|---|---|---|---|---|---|
| Workspace | `pdmux-build` `wsp-fbmj36cisl56` (CPU, $0.30/h) | a2b4176 / 1d47d3e(Zamba2 적재) | `sglang-runtime@c85fc198` | **standby(pause)** | — | ≈$0.3 |
| Workspace | `pdmux-probe` `wsp-7ix1cg4xl1jd` (A100, $1.48/h) | 396a9f2 | `@c85fc198` + libnuma1 수동 설치(DEVIATION 기록) | **standby(pause)**, 07:12–07:27Z | `/io/results/substrate_probe_vessl_2026-10-04` → 로컬, sha OK | ≈$0.37 |
| Job | `pdmux-b5-smoke-ec6d94d-20261004072802` `job-0bwpgxgbh8n2` | ec6d94d | `@1e0d335b` | succeeded 10m28s | fetch DONE+sha256 OK | cost.txt ≥$0.23 |
| Job | `pdmux-lambda0-vessl-1d47d3e-20261004084945` `job-mhhyqwkzwyir` | 1d47d3e | `@1e0d335b` | succeeded 1h54m | fetch DONE+sha256 OK | cost.txt ≥$2.80 |

미회수 Job 0 · 켜진 Workspace 0.
