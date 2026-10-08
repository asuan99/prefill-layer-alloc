# 판정서 — N2(3-a) 사전등록 rev1 규칙층 적대 감사 (claims-auditor, 2026-10-07)

> 감사자는 읽기 전용이라 메인 세션이 감사 보고 본문을 그대로 저장했다(2026-10-07). GPU 0, 새 성능 판정 0건, Claim 등급 변경 0건.
> 대상: `PREREG_N2_3A_2026-10-07.md`(rev1, 미봉인).

## 등급: `NO-GO` — 死因 1건(N2 반전 확인)
- 사인은 한 줄로 고칠 수 있는 규칙 모순이나 반전이 수치로 성립하므로 caveat로 낮추지 않는다.
- 나머지는 死因이 아니다(필수 수리 R-2…R-10, caveat N2C-1…16).
- rev2는 R-1…R-10 반영 뒤 **델타 감사**를 받는다. 이 판정서는 rev2를 미리 승인하지 않는다.

## 死因 D-1 (N2): 108 점 합산 규칙이 서로 모순된다
- §9-2 "D108은 OFF16·OFF44 boot 합 n ≤ 8"(무조건), §13 끝 "예보가 깨져도 라벨은 바뀌지 않는다", §17-7 "F-N2-6이 깨지면 108 점을 FIXED 값별로 따로 낸다(라벨 n은 각 4)" → 분석자가 규칙을 고를 수 있다.
- 반전 구성: (i) F-N2-6 FALSIFIED, (ii) 108 셀 하나에서 OFF16 복제 1개 무효, 나머지 정상. 합산 n=7≥4 → `POINT_REALIZED` → `PROFILE_COMPLETE`; 분리 OFF16 n=3<4 → `UNRESOLVED` → `PROFILE_PARTIAL`. 최상위 라벨 반전.
- F-N2-6 실패 개연성: 초안 §8-3이 burst 산포가 prefill SM에 따라 다르다고 예보(b=48 8–19%), 12셀 전부 3% 미만 요구라 n=4 잡음만으로도 깨질 수 있다.

## 질문별 결론
1. **ladder 결정 정합 — 위반 아님, 사용자 결정 사항 아님.** "arm은 sticky OFF"는 D-SEM(5-a/5-b 정책·static arm) 결정이다. 근거: `handoff-report/experiment_dag_and_execution_2026-10-04.md` §2-4 "정책·static arm 모두 sticky OFF. sticky ON은 프로파일(N2) 전용", DAG 그림(:44) "N2 3-a decode 프로파일 (sticky ON…)", N4 판정서 DS-1·DS-3, F2. 초안은 C_off를 E-α 주 판정량으로 포함해 결정 적용처에서 결정을 지킨다. (사용자 실제 의도가 정본과 다르면 확인은 사용자 몫.)
2. **estimand와 anchor §8**: "SM×batch×ctx" 일치. anchor §8 3-a의 "L>65536 계측 셀" 누락(U-4) → N2만으로 3-a 완결 아님(N2C-10). "단독" 설계는 N4 처방형과 같고 게이트 1·N4 위반 아님. 금지 문안 부족 → N2C-1·14·15 추가(R-8).
3. **실현 술어**: 판정량은 측정(N1 아님). 단 단독 구간 PE-iter는 코드 경로상 거의 항등(`multiplexing_mixin.py:1179`, sticky OFF `:1200` idx5). 동거 구간도 sticky-OFF fixed 루프에서 "D ⇔ split_prefill_batch 존재"가 코드 항등(입장 `:1292` → 결정 `:1326` → 발사, 병합 `:1532`, 해제 `:1541`) → "D44 iteration 비율 = prefill_in_flight 비율" N3 관찰은 독립 증거가 아니라 항등식 재확인(교훈 9). 도달 가능성 결론은 코드 독해로 성립, 술어 fail-closed. `prefill_kernel_done_wait=True` 회차는 동거로 세되 비율 병기. 결함: 기록기 생존 실패(창 안 간격 > 10×S-per)를 `D_UNREALIZED`로 분류 = 측정 실패를 실현 실패로 라벨링(교훈 21); N3 스케줄러 정지 124.8 ms는 S-per≈13.4 ms의 9.3배로 문턱 바로 아래 → R-2.
4. **E2VR-6**: 위반 없음. 한계: "실현 D"는 발사 index 기록과 read-out이며 발사 SM 실측 아님(N2C-2).
5. **통계**: n=4·재표집 35·4/4=1/16 공시 맞음. E-α SLOWER/FASTER의 CI 조건은 공허(4/4 조건과 동치, N2C-5). 108 n=8: residual 기대 ≈1.42σ vs n=4 ≈1.03σ, `predict()` 단조 투영(`profile.py:285-301`)이 모든 D 상한에 전파(N2C-8). metric cliff는 §12-5 d 표만 해당(이미 기술값).
6. **예산(#261)**: 170 s 승계는 boot 단위로 적합. 셀당 고정비 누락 — 22.6 s × 360셀 ≈ +8,136 s → (a) 보통 26,836 s/하드캡 30,600/최악 36,236 초과, (b) 보통 21,336/하드캡 22,700/최악 28,736 초과 → F-N2-7 실패 가능성 큼. boot 입장 규칙(거부 후 건너뛰기 vs 정지) 미등록 — 건너뛰기면 결측이 비용과 상관(비싼 boot ON16·ON44·OFF16·OFF44 768–848 s, 싼 ON24·ON34 546–570 s·OBS 207 s). 결측 시 라벨 fail-closed → R-6, N2C-11·12.
7. **U-1…U-8**:

| ID | 판정 |
|---|---|
| U-1 | 사용자 몫(Job 크기). (c)면 Job 간 수준 비교 불가 |
| U-2 | 사용자 몫(GPU 지출). (b)면 예산 예보 인용 불가·τ ×2. 감사는 (a) 강력 권고(#261) |
| U-3 | 감사 결정: (a) {512,2048,8192}. c=8192 창 [8128,8255]가 shape B decode ctx 덮음, 16k–64k는 열린 항목 17 |
| U-4 | 감사 결정: (a) 별도 등록. 실행 여부·시점만 사용자 몫 |
| U-5 | 감사 결정: (a) N4 리터럴 8셀. (b) 확장은 선택(사용자) |
| U-6 | 감사 결정: (a) ρ_off 주·ρ_on 부. N2C-4 병기 필수 |
| U-7 | 감사 결정: (a) 시드 무작위 토큰 id, 시드는 (셀, round)만으로(R-7) |
| U-8 | 이미 결정(2026-10-04 Nemotron 우선). A11 모델 집합은 나중에 사용자 |

## 반전 시험 표
| 자유 표면 | 민 범위 | 판정 변화 | 근거 |
|---|---|---|---|
| 108 합산 규칙(§9-2 vs §17-7) | 두 규칙 | **COMPLETE ↔ PARTIAL** | 7≥4 vs 3<4 → 死因 |
| 단독 PE 문턱 0.99 | 0.95–1.0 | 없음 | 비목표 iteration 기대 0/128 |
| 동거 문턱 0.95 | 0.90–1.0 | 없음 | 코드 항등 + 경계 끼인 iteration 0(설계 재집계, 인용 금지) |
| 기록 공백 10× | 상수 | — | 9.3× 정지 통과, 분류 결함 → R-2 |
| ρ_off vs ρ_on 주 | 두 선택 | 시연 불가 | caveat |
| S-per vs C-itl | 두 선택 | 라벨 경로 없음 | caveat |
| E-α CI 조건 | — | 공허 | 최소 재표집 평균 = min x |
| 창 128/trim 16/±64 | 64–256 | 실현가능성만 | caveat |
| 입장 건너뛰기 vs 정지 | 미등록 | 결측 분포만 | R-6 |

## 필수 수리
- **R-1(死因)**: §9-2·§17-7 중 하나를 무조건 규칙으로 고정. 권고: 합산 n≤8 유지, F-N2-6 결과는 caveat 문구만, FIXED별 값은 기술값. 대안: 두 행 분리(`profile_cli` build 중복 키 미거부, `nearest()` max).
- **R-2**: 기록기 생존 실패(공백·dropped·R7) → `RECORDER_UNPROVEN`. `D_REALIZATION_FAILED` 비율 분모를 "그 D의 전체 시도 복제"로. 그 라벨 허용 문안이 POINT_REALIZED 점과 모순되지 않게. D108에 "sticky ON" 문구 금지.
- **R-3**: 셀마다 `PDMUX_TELEMETRY_PATH` 불가(boot 시작 1회, `:170-172`·`:222-226`) → boot당 파일 1개 + 셀 구간 분할 규칙 등록.
- **R-4**: `dropped_events`는 boot 누적(`telemetry.py:81,101,114`) → 셀 구간 증분(셀 뒤 첫 레코드 포함)으로 판정.
- **R-5**: 셀마다 cudagraph 재생 증거(confound #8): 셀 구간 서버 decode 로그 전부 `cuda graph: True`, 아니면 `CG_UNPROVEN`.
- **R-6**: 셀당 고정비 항(22.6 s 또는 U-2 프로브 실측)으로 보통·최악·하드캡 재계산 + boot 입장 규칙(정지/건너뛰기) 등록.
- **R-7**: E-α 셀 boot 내 순서 위치를 짝 ON/OFF boot에서 동일 고정, 프롬프트 시드 = (셀, round).
- **R-8**: N2C-1·14·15 금지 문안 추가, §12-5 "45 ms"를 T별 목표 경계로.
- **R-9**: `COLOC_UNREALIZED`·`BS_UNREALIZED` 허용·금지 문안(측정 불성립이지 "sticky OFF는 동거에서 D를 실현 못 한다"는 발견 아님).
- **R-10**: 봉인 전 engine-porter §19-1 확인(병합 `:1532` 뒤 `update_running_batch` `:1297`이 끝난 요청을 걸러 bs==b 유지 예보; 틀려도 `BS_UNREALIZED` fail-closed). ★메인 세션 주: 같은 날 engine-porter 판정 메모 `ENGINE_CHECK_coload_2026-10-07.md`가 코드 독해로 이 예보와 일치(조건: `max_new_tokens=1` 명시 등록).

## caveat (인용 금지 문장 — 결과 문서·정본이 그대로 승계)
- **N2C-1** N2 점은 decode 단독(prefill 유휴) 상태의 엔진측 루프 주기이며 PD-mux 운영점(sticky OFF 동거) 측정이 아니다 — 정책·static arm 선택·d_min의 근거로 인용하지 않는다(2F9·DS-1·DS-3).
- **N2C-2** '실현 D'는 decode 발사 index 기록과 boot green read-out의 일치이며 발사 SM 실측이 아니다(N3C-2).
- **N2C-3** REAL_D_ALONE·REAL_D_COLOC 통과는 sticky/fixed 코드 경로와 대기열 설계의 귀결이다 — 'N2가 D 실현을 독립 실증했다'로 쓰지 않는다.
- **N2C-4** ρ_off는 sticky 상태와 동거 여부를 동시에 바꾼 boot 간 비다(confound #10) — 동거 효과로 귀속하려면 ρ_on 병기, HBM·L2 기전 귀속 금지(S-per에 prefill host launch·backpressure 포함).
- **N2C-5** n=4에서 E-α `*_GE3`는 '4/4 짝이 3%를 넘는다'와 동치 — CI 배제를 독립 증거로 쓰지 않는다.
- **N2C-6** S-per는 정본 goodput 술어의 요청 내부 token-ITL이 아니다.
- **N2C-7** b 축 S-per에는 11 CPU quota 안에서 클라이언트 스트림 수에 비례하는 호스트 경합이 섞여 있다(confound #7).
- **N2C-8** 108 residual(n=8)은 다른 D와 비교할 수 없고, 예측기 단조 투영이 이를 모든 D에 전파한다.
- **N2C-9** ctx 8255 초과 질의는 8192 점으로 평탄 클램프 — 그 영역에 '보수 방향'을 쓰지 않는다.
- **N2C-10** 이 Job은 anchor §8 3-a의 L>65536 셀을 포함하지 않는다 — '3-a 완결'로 쓰지 않는다.
- **N2C-11** n=4는 정확히 하한이라 여유 0 — boot 1개 결측이 그 D의 12점 전부를 UNRESOLVED로 만들며 발견이 아니다.
- **N2C-12** 예산 모형은 미검증(#261) — 후속 문언에 인용하지 않는다.
- **N2C-13** 단일 Job 값이다(E2VR-3).
- **N2C-14** D 간 S-per 차이는 boot 간 비교이고 짝 CI 미등록 — SM 민감도 판정으로 쓰지 않고 KISTI C2와 비교하지 않는다.
- **N2C-15** §12-5 d 표의 최소 D를 'decode에 필요한 SM'·'운영 d_min'·'static 권고'로 쓰지 않는다.
- **N2C-16** greedy `DIVERGED`는 부정확을 뜻하지 않고, `IDENTICAL`은 무부하·동시성 1에서만이다.

## 허용 문안
초안 §16-1 틀 유지 + 위 caveat 필수 병기. §16-3 금지 문안에 N2C-1·3·14·15 추가.

## 반증 실패(공정 기록)
sticky ON 프로파일 장치는 D-SEM과 정합. E2VR-6 위반 없음. N3(예측 도달 불가)·N4(제1원칙 위반) 불성립. 단 실행 전에는 어느 것도 `CONFIRMED` 아님.

## 근거 파일
`workspace/engine-port/src/multiplex/{multiplexing_mixin.py,profile.py,telemetry.py}`, `handoff-report/VERDICT_n4_dsem_rules_2026-10-04.md`, `handoff-report/experiment_dag_and_execution_2026-10-04.md`, `reports/system_design_anchor_2026-09-30.md`, `workspace/engine-port/results/n3_capacity/REV7_DRAFT_NOTES_2026-10-07.md`, `workspace/engine-port/results/n3_capacity/probe_6group_2026-10-05/SUMMARY.md`.
