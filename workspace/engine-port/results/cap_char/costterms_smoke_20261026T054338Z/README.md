# Workspace smoke, 비측정, 인용 불가 — 비용 항 하네스 (2026-10-26)

> 운영 모델상 측정 = Job만. 이 디렉터리의 수치는 봉인 입력(ramp 상한·J-B2 하드캡·하네스 수정)의 **설계 입력**이며 결과 문서·COSTMODEL·CONSENSUS에 인용하지 않는다. 절차서 `../SMOKE_PLAN_costterms_2026-10-22.md`.

- 기판: VESSL Workspace `pdmux-probe`(A100 SXM ×1, 이미지 `c85fc198` — Job 이미지 `1e0d335b`와 다름), 코드 = main `580fc1c9`(하네스 `4e0ae086`, 미봉인), smoke 모드(`SMOKE_UNSEALED`).
- 실행: 05:43:38Z–05:56:45Z(787 s), S-A `ja1_D44`(PD-mux D44, context 131,072) · S-B `jb2_capCA_L16384`(T4 16k 채우기) · S-C `jh1_dummy_a4`(vanilla dummy). 세 boot 모두 rc 0, `SHA256SUMS` 원격·로컬 검증 OK. Workspace는 회수 뒤 standby 확인.
- 회수: 실행 에이전트가 중단돼 main 세션이 원격 `/data/smoke_costterms/`에서 직접 회수(base64 tar 전송, `summary/smoke_tgz.sha256`). 원자료 `run/`·`smoke.tgz`는 git 밖(`.gitignore`), 원본은 클러스터 볼륨 `/data/smoke_costterms/run/`에 남음. 분석은 사본에서 `ct_runner.py analyze`로 재실행(`summary/*_analysis.json`).

## 판독(기록만, 판정 없음)
| 항목 | 관측 |
|---|---|
| ENGINE_DIGEST | 세 boot 모두 `MATCH`(`99bbef91…`, pathless-v1) |
| libnuma | Workspace 이미지에 없음 → `common_ops` import 실패 → `apt-get install libnuma1`(05:43:15Z, `SMOKE_DEVIATION.txt`). Job 이미지는 포함(운영 모델) |
| 실현 SM | S-A·S-B D44/P64 요청 = 실현, S-C vanilla 108 |
| context | 로그 최종 `context_len` = 계획값(131,072 / 73,728) |
| ramp (PD-mux D44, c 2k) | bs 32: `ramp_records` 409 대 상한 1,132(상한의 36%), 창 VALID · **bs 8: 상한 0 대 실제 118 → `RAMP_EXCEEDED_BOUND`, 창 `WINDOW_CONTAMINATED`(bs 변화 제외 103 > 30), 창 전 종료 요청 1** |
| ramp (vanilla, bs 8, c 2k) | 0 대 상한 64, VALID |
| T4 16k 채우기(cap 80) | 88요청 전부 finish=length, 소요 485 s(모형 가정 약 6 min/회 대비 약 8.1 min), `n_real` 71 대 산식 B 64·A 80, **`FILL_INCOMPLETE`**("n_real번째 첫 토큰이 가장 이른 완료보다 늦음"), 거절 사유 시간 몫 no_waiting 0.52·prefill_seat 0.35·kv 0.09·max_prefill_tokens 0.04 |
| 클라이언트 대 엔진 | AGREE(bs 32 D44, bs 8 vanilla) |
| `mixed` 모드 | decode\_probe 기록에 없음(가짜 엔진 가정과 같음) |
| abort | S-A bs 32 셀에서 `aborted` 표기 10/32, bs 8 7/8 — 나머지 요청의 종료 경로 확인 필요 |

## 하네스에 넘길 것(이 문서는 고치지 않음)
1. **PD-mux 소-bs ramp 상한 모형**: 한 prefill batch에 burst 전체가 들어가는 셀(bs 8 × 1,882 ≤ 16,384)에서 상한이 0으로 계산되지만 실엔진 PD-mux는 118 step ramp — prefill seat 순차 입장을 반영하는 하한 항 필요(`RAMP_SIGMA` 단일 배율로는 0을 못 고침). O 재산정 → plan sha 변경 → 봉인 전 반영.
2. **T4 16k 채우기 `FILL_INCOMPLETE`**: R = 4,096 출력이 채우기(prefill 71 × 16k, P64) 완료 전에 끝남. R 상향은 예약을 늘려 C\_mem을 바꾸므로 estimand 영향 — 설계 결정 필요.
3. **J-B2 하드캡 재계산**: 채우기 1회 약 8.1 min(모형 6 min) → `ct_timemodel` 채우기 상수 갱신(봉인 체크리스트 항목 22).
4. abort 표기 불일치 확인.
5. S-C는 real a4 짝이 없어 W-INDEP·P3·P4 미평가(smoke 범위상 정상).
