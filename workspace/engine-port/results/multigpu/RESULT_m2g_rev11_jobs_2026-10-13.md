# M2g rev11 Job별 실행 결과 (2026-10-13) — 기술 기록, 성능 판정 0

> Job: J1 `pdmux-multigpu-313be0a-20261013062324`(x2, GPU0), J2 `…2fc3880-20261013062351`(x2), J4 `…99b5e5e-20261013062404`(x4), J8 `…d9e7743-20261013062419`(x8). 네 Job 모두 exit 0, fetch DONE+sha256. 커밋 차이는 원장(vessl_jobs.tsv) 4줄뿐. 컨테이너 시간 J2 0.88 h, J4 0.97 h, J8 1.04 h. 전 Job throttle 경고(fetch) — 기술치.

## PASS 표 요약
| Job | 포함 | 제외 | 비고 |
|---|---|---|---|
| J1 | N1 4 arm | — | G5(a) PASS |
| J2 | N1·TP2·DP2 전 arm(12) | — | G0(ii) PASS, G0(iii) PASS, G1 4/4 PASS, G3 TP2 PASS, G3b PASS, G5(b)(c) PASS. TP2 `G2_NOT_TESTED` |
| J4 | N1·DP4 전 arm, TP4 fused·fused\_sym | **TP4 agnostic·d44**(G1 FAIL) | G3·G3b·G5 PASS, `G2_NOT_TESTED` |
| J8 | N1·DP8 전 arm, TP8 fused·fused\_sym | **TP8 agnostic·d44**(G1 FAIL) | 같음 |

## TP4·TP8 G1 FAIL의 원인 — 출력 불일치가 아니라 하네스 결함
- G1 FAIL 사유는 `abnormal exit`(fused 서버가 boot 끝 생존 기록에서 죽어 있음, poll rc −9). 출력 비교(m, q)는 계산되지 않았다.
- fused 서버는 **G1 단계(06:55:03–06:55:10 UTC, TP4) 뒤 G2 단계(06:55:10–06:55:13) 중 06:55:12에 OOM으로 죽었다**: `torch.OutOfMemoryError: Tried to allocate 4.00 GiB`(GPU당 여유 약 3.6–3.8 GiB). 4 GiB = 8192 토큰 × vocab 131072 × fp32 — G2 teacher-forced 요청이 프롬프트 전 위치 logprob(`return_logprob`)을 chunked prefill 8192 한 덩어리로 계산하는 크기와 일치(추정, 엔진 코드 미확인). TP2 fused와 PD-mux arm(chunked −1·overlap OFF)은 같은 OOM이 없었다.
- 하네스는 G1의 "비정상 종료 0"을 boot **마지막** step 뒤 생존 기록으로 판정한다(MGH-16). 그래서 G2 단계의 OOM이 G1 FAIL로 전가됐다.
- G2는 전 TP에서 `NOT_TESTED`: 바닥(TP=1) 계산에서 각 프롬프트 위치 0의 logprob가 non-finite(첫 토큰은 logprob 없음)라 채점 거부 — 하네스 결함.

## 결론(사실만)
- TP4·TP8의 agnostic·d44 제외는 PD-mux 정확성 문제를 나타내지 않는다(출력 비교 미수행). 그대로 M3에 쓰면 TP4·TP8의 G\_sym이 미정의라 TP 추세는 {1,2}만 남는다.
- 수리 후보: (a) G1 생존을 G1 단계 직후에 기록, (b) G2 요청을 OOM-안전하게(요청 단위 직렬·logprob 범위 제한), (c) G2 위치 0 제외. 수리 뒤 J4·J8 재실행 필요 여부는 사용자 결정.
