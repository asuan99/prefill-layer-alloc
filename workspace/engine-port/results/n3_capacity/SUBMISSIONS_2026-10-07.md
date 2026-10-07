# N3 capacity 제출 원장 (2026-10-07)

## shape B 4회차 제출 (rev7, 2026-10-07)

| shape | job name | slug | 제출(UTC) | 커밋 | 이미지 | spec | 승인 |
|---|---|---|---|---|---|---|---|
| B | `pdmux-n3-capacity-93c3cca-20261007064124` | `job-mduq9ace36q9` | 2026-10-07T06:41Z | `93c3cca`(rev7 봉인) | `sglang-runtime@1e0d335b…` | `resourcespec-a100x1` | 사용자 2026-10-07 "제출 진행" |

- env: `PDMUX_N3_SHAPE=B`, `RESUME_FROM` 미설정(새 Job, 48 unit 전량 재시도 대상).
- 제출 결과: Job 생성(slug `job-mduq9ace36q9`), `vessl_jobs.tsv` 행 추가(fetched=no). 회수: `scripts/vessl/fetch.sh n3_capacity pdmux-n3-capacity-93c3cca-20261007064124`(DONE+sha256 통과가 완료).
