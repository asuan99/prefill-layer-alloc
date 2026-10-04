# 판정서: N3 사전등록 rev1 규칙층 감사 (게이트 #6 SLO 용량 격자, claims-auditor, 2026-10-05)

> 읽기 전용, GPU 0. 감사자 최종 보고를 메인 세션이 옮겼다. 성능 판정 0건, Claim 등급 변경 0건.
> 대상(sha 7/7 일치): `PREREG_N3_CAPACITY_2026-10-04.md`(7cce0836), `n3_capacity.sh`(2cbee162), `n3_plan.py`(8933c20f), `n3_analyze.py`(b326cdf8), `n3_realized_mix.py`(d52f3347), `configs/pdmux_n3_A_agn.yml`(01675b2d), `tests/test_n3_capacity.py`(0f15c4f2).
> 감사자 재실행(CPU): `n3_plan --selftest` OK · `n3_analyze --selftest` OK · `n3_realized_mix --selftest --require-archive` 7/7 IDENTICAL · unittest 하위집합 27/27 · 예산 재계산 · λ0 VESSL·E2 아카이브 bench jsonl을 정본 `analyze.py`(1be49511) `passes()`로 재채점 · 엔진 미러 `multiplexing_mixin.py`(ba5835aa) 코드 독해.

## 0. 판정: **`GO-with-caveats`**
- **死因 0.** 등록 caveat 14건(§5), 제출 전 필수 수리 8건(§4, 전부 GPU 0). R1·R3는 규칙이 바뀌므로 **rev2 델타 감사** 필요.
- 실행 승인이지 **게이트 #6을 닫는다는 뜻이 아니다**.

## 1. 반전 시험 표 (설계 검증용 수치, 전부 인용 금지 — n=1, homog5, 다른 노드)
원자료: λ0 VESSL bench jsonl, E2 OFF bench jsonl(다른 GPU), 사전 floor B 0.90 s·A 45 ms(λ0 warm-up 중앙값 재현: B 902–910 ms, A 44.4–47.5 ms), 술어 `passes(T, 50)`.

| 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 |
|---|---|---|---|
| 격자 배치(사전 floor로 λ0 셀 재채점) | E1 원형 A {1.8…4.4} ↔ 등록 {1.35…3.00}, B 상단 추가 유무 | 없음(브래킷 끝점만 이동) | A 3.0 att k=3/5/10 = 0.438/0.537/0.585 전부 실패 · A 1.8 k=3 0.939 |
| A 상단 3.4–4.4 제거 | 포함 ↔ 제거 | 없음 — 3.0이 모든 k에서 실패, k=25에서도 0.690 | `ABOVE_GRID_10` 사실상 미도달 |
| floor 잡음(T_k ±10%) | ×0.9–1.1 | 측정점 판정 불변 | b_r1 k=10 0.909–0.964 · a_r1 k=3 0.913–0.959 · b_r0 k=5 0.850–0.875 |
| ITL 섭동(등록 허용 3%) | ×1.03 | 1.8·3.0 측정점 불변(1.8 여유 8 ms, 3.0은 TTFT로 이미 실패) | a_r2(3.0) ITL≤50 통과율 1.000→0.490 |
| 노드 교체(같은 config·arm) | λ0 ↔ E2 노드 | 3.0 불변(양쪽 실패) | ITL≤50 1.000 vs 0.735, TTFT p50 160 vs 736 ms |
| span(B 100 s·A 90 s) | 등록 상수 | 자유 표면 아님, 라벨은 span 조건부 | b_r1 k=10 0.899(69건) vs 0.936(110건) |
| 첫 실패 vs 최대 통과 | 두 규칙 | 반전 못 만듦(자료 단조) | — |
| seed 재채점 | λ0 seed ↔ 상위 4 | 같은 셀 없어 계산 불가 | M/D/1 proxy(인용 금지): 전체 후보 대비 p90 대기 B −22%·A −26%, 가드 적격 대비 B +12%·A −11% |
| S1/S2/S3 상수 | 엄격↔관대 | 구조상 없음(크기 인용 여부만) | — |
| 관측자 허용 ln 1.03 | ±3% | 라벨 불변(채점 셀 전부 ON) | → N3C-6 |
| settle·포트·teardown | — | 판정 경로 없음, 예산 절단 경로만 | B 최악 코너 여유 8 s → R1 |

## 2. 쟁점별 판정
1. **게이트 8 인접성 — 허용되는 설계 입력.** (i) static arm, `FixedPolicy`는 SLO를 읽지 않고 러너는 `PDMUX_TTFT_SLO_MS` 설정 시 abort (ii) 판정량은 Job 내 floor + 실행 전 고정 절대 격자 (iii) 사전값은 브래킷 해상도만 바꿈. shape A 격자 교체는 데이터가 지지(3.0이 전 k 실패, λ\* 위는 포화 구간이라 런 길이 의존·게이트 6).
2. **A′ 조항**: P-2 충족(N3C-2 부착) · P-3 충족, **1 arm이라 Holm 없음이 맞다**(N3C-11) · P-5 충족 · P-4 해당 없음 · P-7 반영됨. **P-1 틀 문장 리터럴 부재, P-6 금지 문안 4개 누락 → R8.**
3. **필수 조항 11개**: 조항 1 — 임계 열 inert가 GPU 미실증인데 arm 동일성 조건 미등록 → R3. 조항 3 충족(W-idx 결함 N3C-9). 조항 4 — 4쌍 ABBA 1 rate는 문언 충족이나 shape A ITL 칼날에서 관측자 효과를 못 봄(N3C-6). 조항 5 — **간격 문장이 사실과 다름**(B 9간격 중 8개가 0.101–0.116×λ\*\_ref, 등록 `CLIFF_SPACING_FRAC` 0.10과 자기모순; A 0.067–0.1003×) → R5. **shape A는 TTFT 용량 측정으로 성립하지 않는다 — 결합 술어 용량이다**(Little 근사 2.4 req/s×512 tok×0.039 s ≈ 48에서 cap 48에 묶이고 그 뒤 ITL이 49–50 ms 포화; E2 노드 cap-bound ITL p95 중앙값 49.7–50.7 ms). 측정점 반전 없어 死因 아님 → R4·N3C-5. 조항 7·6·8·9·10·11 충족(retract 로그 경로 미실측 공시).
4. **결정 규칙**: 첫 실패·4 라벨·UNRESOLVED 범위·비단조 플래그 코드-문서 일치, 항등식·퇴화 없음. 결함: r_fail 위 seed 결측 시 라벨 4 seed·bootstrap 3 seed 불일치(N3C-12).
5. **seed 재선택 정당**(λ0 4162는 N=20에서 1/Ēbar 0.748; B 가드 통과 후보 5000 중 7). 비용: λ0 셀과 짝 불가(bridge는 λ0 seed로 보존), 조건부 표본의 대기 낙관 편향(N3C-8). 해상도: B 저rate 합 80건, 90% 문턱 = 실패 8건, 1.25%p.
6. **범위 밖 추가 — CG2 승인**(`--output-details --output-file`은 클라이언트 파일 쓰기뿐, `bench_serving.py:1618–1633`, payload `:531` 불변; fail-closed, 손실 ≤3 boot; 동시성 1 경로라 D44 경로 미검증 N3C-10). **bridge OFF 승인.**
7. **config bridge**: 기술 비교로 충분하나 이전 근거 불가 — 같은 config·arm에서도 노드 간 achieved 0.952–0.967. 재측정 조항 = N3C-3.
8. **O1–O4**: 판정 경로 없음. settle 최대 180 s/boot가 예산 모형에 없음 → R1.
9. **예산**: B 보통 12,126 s(3.37 h)·최악 13,792 s — **셀 마감 13,800 s 대비 여유 8 s.** round 4가 내림차순이라 마지막 unit이 최저 rate `g_r010_4589`; 절단 1건이면 0.10의 n=3 → 세 k 전부 `UNRESOLVED_N_BELOW_4` → R1. 6그룹 config 기능 위험은 floor_head에서 fail-closed(`ABORT_GREEN`, P3, 손실 ≤ preflight+1 boot)로 닫히나 **오버헤드 증가 위험은 절단으로 전파**. 선택: Workspace 프로브 1 boot(≈0.1 GPU-h). P3 엄격 동일 조건은 48 boot라 E2 전례(boot 11 −3) 위험 → R6.
10. **테스트**: 실경로 검증 있음(쌍 키 7/7 비트 동일, (92,16) 음성 7/7, 인덱스 재배치 대조, ITL leg·k 순서 대조, 집행기 등가 변이 공시). **항등식 근접**: `TestFailClosedWiring`은 900자 창 텍스트 인접성 검사, 변이 대조 1건뿐 → 하네스층 과제. 빠진 변이: dry-run `T_K.json` 재개(R2), 절단 순서(R1). floor 필드명은 엔진 실재 확인(`:2190–2201`).

## 3. 새 코드 사실(독해, 미실측)
- `prefill_span_start`가 기록하는 분할은 **그 루프 반복의 C7 전환 이전 인덱스**(hook `:1235→:870`, 전환 `:1312–1376`, prefill 발사 `~:1468`, decode 이벤트 `:1405`). floor 검사는 decode 유휴 시 전환 전후 모두 idx0이라 보수적으로 맞음(`:1211`). 부하 셀의 W-idx·span 기술값은 짧은 구간 오귀속 → N3C-9.
- `PDMUX_PHASE_EVENTS`는 기록 실패 시 **그 boot의 나머지를 조용히 끈다**(`:2165–2176`) → R7.

## 4. 제출 전 필수 수리(rev2, 전부 GPU 0)
- **R1.** round 순서 홀짝 반전(1·3 내림, 2·4 오름) — 절단 시 최고 rate가 먼저 빠지게. drift 균형 유지. 구조 테스트 수정 + "마지막 unit = 최고 rate" 테스트 추가. [델타 감사]
- **R2.** `tk_valid`·재개 검사·분석기가 `"dryrun"` 키 있거나 `ttfts_s` 없는 `T_K.json`을 거부(지금은 DRYRUN Job 재개 시 placeholder T=3/5/10 s로 채점 — P-2 fail-open). 변이 시험 추가.
- **R3.** arm 동일성 조건 등록: 채점 셀의 `decode_iteration`(전환 뒤)과 decode-busy 스냅샷에서 (92,16)·(84,24)·(74,34) **0건**, 위반 셀 무효(retract 처리). 사전 근거 λ0 358,583 레코드 비표적 0건. [델타 감사]
- **R4.** §10-2 "ITL로 정해지면"을 기계적으로: k별 등록 라벨·용량이 `itl_ladder_capacity[k{k}_itl65]`와 다르면 `ITL_BOUND_k` 플래그. 새 상수 없음.
- **R5.** §3-1 간격 문장 정정 — 기본안: "B는 조항 5의 0.1×를 충족하지 않는다(9간격 중 8개 0.101–0.116×)"를 편차로 등록(새 자유 표면 없음). B 재격자는 사용자 선택 대안(델타 감사 필요).
- **R6.** P3 정책을 E2 A14 결정과 맞춰 명시 등록(엄격 또는 (ii) 허용오차) + E2 boot 11 전례 공시. **사용자 결정**(A14 자체 감사 대기).
- **R7.** 서버 로그 `PDMUX_PHASE_EVENTS disabled after failure`를 boot마다 검출 → 그 셀의 PHASE_EVENTS 기술값 무효·대칭 서술에서 제외. 하네스층.
- **R8.** P-1 틀 문장 리터럴 삽입. P-6 금지 문안 4개 추가("cliff 때문에 MLPerf 불채택", "10 s 휴리스틱", "DistServe 90%", "길이 정규화로 길이 간 비교").

## 5. 등록 caveat N3C-1…14 (결과 문서·정본이 문자 그대로 승계)
- **N3C-1.** "이 캠페인은 게이트 #6을 닫지 않는다. 게이트 #6 상태 변경은 결과 감사와 5-a 전제 판정 뒤에만 논의한다."
- **N3C-2.** "floor의 (108,0) 실증은 admission 시점 인덱스의 telemetry와, decode 유휴 시 발사 인덱스가 같다는 코드 독해의 결합이다. 발사 분할을 직접 계측한 것이 아니다."
- **N3C-3(재측정 조항).** "N3 B1 수치(λ\*\_SLO 라벨·T_k·실현 분할)는 `pdmux_n3_A_agn.yml`(sha 01675b2d…) 한정이다. 다른 config의 캠페인은 B1을 다시 잰다. bridge 2셀은 λ\*\_VESSL(homog5)을 A-agn으로 이전할 자격을 주지 않는다. 같은 config·arm에서도 노드 간 achieved 차가 3.3–4.8%이기 때문이다(E2 vs λ0 a_r4)."
- **N3C-4.** "N3 셀은 5-a·5-b arm 비교의 짝으로 쓰지 않는다. Job·노드·seed가 다르다(confound 11)."
- **N3C-5.** "shape A 라벨은 결합 술어(TTFT ∧ ITL 50 ms) 용량이다. `ITL_BOUND_k`이면 'TTFT 용량'으로 쓰지 않는다. shape A의 ITL 50 ms는 cap 48 포화 ITL(49–51 ms) 바로 위의 칼날이라, 라벨은 측정 노드에 조건부다(같은 config·arm, offered 3.0에서 ITL≤50 통과율이 노드에 따라 1.000 vs 0.735)."
- **N3C-6.** "OBSERVER_OK는 B 0.30·A 1.80 한 점의 결과다. shape A의 ITL 구속 구간에서 PHASE_EVENTS가 라벨에 영향이 없음을 보이지 않는다. 등록 허용폭 3% 안의 ITL 섭동이 offered 3.0에서 ITL 통과율을 1.000→0.490으로 바꾼다."
- **N3C-7.** "용량 라벨은 등록 span(B 100 s, A 90 s)에 조건부다. 용량 셀이나 r_fail 셀이 UNSTABLE이면 'span 조건부'를 병기한다."
- **N3C-8.** "seed는 실현 평균 도착률을 nominal ±5%에 맞춘 조건부 표본이다. 무조건부 Poisson 대비 대기가 낙관 쪽으로 기울 수 있다(미측정)."
- **N3C-9.** "W-idx와 span 기반 기술값은 span 시작을 전환 이전 인덱스로 기록한다. 부하 셀에서는 기술값이며 '발사 분할 점유'로 쓰지 않는다."
- **N3C-10.** "CG2 PASS는 동시성 1 비분할 경로에서 PHASE_EVENTS가 greedy 출력을 바꾸지 않는다는 것만 뜻한다."
- **N3C-11.** "N3에는 arm 비교가 없다. k별 라벨을 묶은 '모든 k에서 견고'는 판정이 아니다."
- **N3C-12.** "r_fail 위 셀이 빠져 bootstrap seed 집합이 라벨의 seed 집합보다 작으면, 두 n을 함께 적는다."
- **N3C-13.** "λ0 VESSL 셀로 정한 격자 위치와 사전 floor(B 0.90 s, A 45 ms)는 설계 사전값이다. 결과로 인용하지 않는다."
- **N3C-14.** "B 격자 간격은 조항 5의 0.1×를 충족하지 않는다(최대 0.116×λ\*\_ref, 브래킷 폭 최대 0.08 req/s)." (R5 기본안 채택 시)
DS-1…DS-7, LV/LVR 계열, 사전등록 §11 금지 문안은 그대로 승계.

## 6. 반증 실패(공정 기록)
- 감사자 자신의 가설 "F-B1은 warm-up 희석으로 구성상 HELD"는 **반증됨**(E-time 전체 vs warm-up 제외: b_r0 70.07/70.07, b_r1 86.05/86.07, a_r0 3.22/3.38, a_r1 6.14/6.45). F-B1은 유효한 예보.
- 반증 못 한 것: 첫 실패 규칙 · Ēbar seed 재선택 필요성 · floor 공식과 무효화 대조 · 쌍 키 추정량 실경로 정합 · `--output-details` 클라이언트 한정성 · shape A 상단 제거 정당성. 실행·결과 감사 전엔 어느 것도 `CONFIRMED` 아님.

## 7. 실현가능성 자기검사(게이트 #113, 교훈 250)
R1–R8 전부 GPU 0, 새 상수 없음(R3 0건 허용, R4 기존 65 ms 사다리). R1은 총비용·drift 균형 불변. R3 오발화 시 UNRESOLVED(정직한 쪽). R5 기본안은 재격자 없음. R6은 사용자 결정으로 넘김(A14 (ii) 미감사). 직전 판정서 조항을 상수로 승격하지 않고 코드·아카이브로 독립 재검증(FixedPolicy SLO 비의존, 스캔 도달 경로, Ēbar 수치).

## 관련 파일
`n3_capacity/PREREG_N3_CAPACITY_2026-10-04.md` · `n3_capacity.sh`(round 순서 `n3_plan.units` · P3 `:411–418` · CG2 `:426–437` · 재개 `:216–218`) · `n3_plan.py`(`units`, `MAX_SPACING_FRAC`, `floor_from`) · `n3_analyze.py`(`capacity_rule`, `bootstrap_capacity`, `descriptive`) · `src/multiplex/multiplexing_mixin.py`(`:870`, `:1211`, `:1235`, `:1312–1376`, `:1405`, `:2165–2204`) · λ0 VESSL·E2 아카이브 · `e2_sticky_vessl/PREREG_E2_VESSL_A14_DRAFT_2026-10-04.md`(P3 선택지) · `CONFIG_CANDIDATES_2026-10-04.md`(:41, :44, :237).
