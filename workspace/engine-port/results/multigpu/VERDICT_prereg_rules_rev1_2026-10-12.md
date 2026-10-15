# 규칙층 감사 판정서: PREREG_MULTIGPU_2026-10-12 rev1 (M2.5)

> 감사자: claims-auditor(읽기 전용, GPU 0), 2026-10-12. 판정문 원문 요지를 메인 세션이 저장했다(표·처방·caveat는 원문 그대로).
> 대상: `results/multigpu/PREREG_MULTIGPU_2026-10-12.md` rev1, 브랜치 `multigpu-scaling`, 기준 커밋 `c0e200d`.

## 0. 판정 — `MIXED`

| 범위 | 등급 | 이유 |
|---|---|---|
| **M2g 규칙(§3)** | **`GO-with-caveats`** | 死因 0. MGP-1…MGP-5 승계 조건으로 하네스 작성·하네스층 감사 진입 가능 |
| **M3 규칙(§4)** | **`NO-GO`** | 死因 2건. **D1**(N2, 지표 절벽 #5): LO rung 창·분모 미등록, 문턱 0.95가 drain 항등식 경계 위. **D2**(N2): E-time 추정량 등록 문서 내부 모순. 둘 다 GPU 비용 없이 수리 가능 |
| **M4 규칙(§5)** | 판정 안 함 | 등록 문서가 "M3 결과 감사 뒤 별도 승인"으로 둠 |

이 판정을 "multi-GPU M3 GO"나 "M2.5 통과"로 요약하지 않는다. MG-0은 이번 감사로 M2g 부분만 해소된다.

## 1. G(N)과 F-1
- F-1 해소 확인: G는 같은 config 안 두 arm 측정의 비, 1-GPU λ\*·ρ 정규화를 거치지 않음, e(N)은 기술치로 격리.
- "cap 구속 포화에서 G→1"은 항등식 아님(반증 실패): cap 구속 처리율 = cap/S\_arm → G = S\_fused/S\_agn. 단 λ0 VESSL a_r4 decode-busy 스냅샷 91.2%가 비분할 (0,108)(E-time D44 13.89%) → **shape A의 G는 작을 것으로 예보, `PARTITION_NOT_EXERCISED` 거의 확실**(범위 축소이지 퇴화 아님, MGP-6).
- seed=boot 공유는 반전 미생성(λ0 분산 블록이 boot·seed 미분리) → MGP-7.

## 2. 반전 시험 요지
예보 계산: Poisson 도착, span ~ Gamma(N−1, 1/x), ratio\_nominal = N/((S+L)·x), ratio\_realized = S/(S+L), 200k MC. L은 λ0 원자료 역산(a_r1 ≈ 8.9 s, a_r0 8.1 s, b_r0 6.55 s; λ0 d44(homog5) arm 값이며 fused/agnostic 값 아님).

| 자유 표면 | 판정 변화 | 근거 |
|---|---|---|
| **LO rung 요청 수/창(미등록)** | **반전**: shape A N=1 `KNEE_BRACKETED` ⇄ `NOT_BRACKETED` | x\_lo = 1.80. 170 s: realized 0.951–0.955, nominal seed당 실패 0.43–0.47, 두 seed 통과 0.29–0.32. 700 s: 0.987, 두 seed 통과 0.85 → D1 |
| **LO 분모 nominal↔realized(미등록)** | 반전(결합). B LO N≈70 Poisson 잡음 지배 | B T170 nominal seed당 실패 0.37–0.39 → D1 |
| **E-time 추정량(스크립트 ↔ 산문)** | **반전**: `PARTITION_NOT_EXERCISED` 라벨 | `e2_realized_mix.py:48 D44_STREAM_INDEX = 2`(homog5 기준). `pdmux_r2.yml`의 idx 2 = (84,24), fixed D44 = idx 4 → d44 E-time ≡ 0. agnostic은 idx 1(74,34) 누락. `:49` SM 불일치 검사가 ABORT 경로 → D2 |
| HI 포화 술어 ratio ≤ 0.90 | N=1 불변, TP=8 A에서 e(N)≈1.5면 경계(미확인) | MGP-8 |
| arm별 확장 | 조건부 편향(N=1 A 최대 2.9%, 범위 0.6–5.9%) | MGP-9 |
| λ\* 추정량 중앙값/최댓값/평균 | 3% 규칙 불변 | λ0 4값 범위 0.54%/0.12% |
| E-time 문턱 0.5 | N=1 불변(A ≈ 0.09–0.14, B ≈ 0.97) | — |
| G1 −0.05 중앙값 | 미확정. 중앙값은 프롬프트 절반 미만 결함 못 봄 | MGP-2 (907100 C-tier: TP=1 legacy-legacy 4/32, cross 4–7/32 불일치) |
| G2 | `TP_NUMERIC_DIVERGENT`만 변화 | MGP-3 |
| G3 R=500, watchdog 120 s | 통과 불가 퇴화 아님, 검출력 낮을 수 있음 | MGP-4 |
| G5(a) TV ≤ 0.05 | 결정적 성질을 확률적으로 시험 | MGP-5 |
| G(1) N=1 회차, bootstrap 재표집 단위(미등록) | 미확정 | MGP-10 |
| HARNESS_LIMITED 5 ms/1% | 불변(M1.5 sched lag p99 ≤ 1.05 ms @128, M1 throttled 0.12%) | MGP-11 |
| config 순서 고정·앵커 | G는 config 내 비 → 1차 드리프트 상쇄 | — |

## 3. 死因과 처방

### D1: LO 브래킷이 drain 항등식 경계 위
미포화 셀의 ratio ≈ T/(T+L)은 측정이 아니라 창·drain 항등식(λ0R-4 정신). 정의되지 않음이 ITL 긴 낮은 N(N=1)에 체계적으로 몰려 추세 점 집합과 G(1) 분모가 N 의존 선택이 된다. HI도 ratio ≤ 0.90을 포화 근거로 써 λ0R-4 승계 누락.

**P-D1(GPU 추가 0)**:
1. G 정의 조건을 독립 포화 증거로: 서버 로그 `#running-req`·`#queue-req`(λ0 `RUNNING_REQ_RECOUNT.txt` 선례; TP rank 0, DP 복제본별). HI 4 seed 전부에서 A: `#running-req` = cap ∧ `#queue-req` > 0 샘플 ≥ X%, B: `#queue-req` > 0 샘플 ≥ X%.
2. LO는 독립 비포화 증거로: 송신 창 동안 `#queue-req` = 0 샘플 ≥ Y%. 원시 ratio는 기술치. 셀 자신의 마지막 요청 지연으로 drain 보정 금지(교훈 9).
3. ratio를 기술치로 남기면 분모 = realized 도착률 명시.
창 ≥900 s 대안은 약 11 h 추가로 하드캡 초과·N=8 `TRUNCATED` → 권고 안 함(게이트 #113).

### D2: E-time 추정량 내부 모순
**P-D2**: decode-busy = A2 술어(`decode_running_batch_size` > 0), 시간 가중 = 다음 non-startup 스냅샷까지(e2 규약), 분자 = `stream_index` ∉ {0, real_sm_group_num−1}, arm별 sm_counts를 green readout과 대조해 불일치 셀만 `MIX_FAILED`(ABORT 안 함). 변이 검사: "idx == 2 문자 그대로" 변이본이 agnostic·r2 합성 telemetry에서 KILLED.

## 4. agnostic manual 등가표 — `CONFIRMED`(N=1 코드 수준)
`divide_sm(108,(8,0),2)` → [(74,34),(54,54)]. 자동: idx = max(1, min(2, bs·2//36)). manual 임계 [0,36]: 마지막 충족 행. 모든 bs ≥ 0에서 같다. `SM_COUNTS` 두 경로 [(108,0),(74,34),(54,54),(0,108)], `load_pdmux_config` len 검사, 다른 소비처 없음(grep 전수), sticky OFF면 `_sticky_fixed_idx` 미사용, budget 둘 다 65536. `pdmux_agnostic_manual.yml`은 아직 없다 — 하네스 봉인 digest에 포함.
**부수 발견(MGP-12)**: `scheduler.py:379 init_pdmux`가 `:391 init_model_worker`(→ `model_runner.py:895 set_device`)보다 먼저 → `pdmux_context.py:109 current_device()`가 rank≠0 프로세스에서 GPU0에 컨텍스트를 만들 수 있음(PD-mux arm만). N>1에서 GPU0 메모리·KV/mamba 풀이 arm 사이 달라질 수 있음. 미실측.

## 5. 기타
- MGP-1: G1 FAIL 단위(arm vs (mode,N)), 미시험 셀, `ABORT_M2G_NOT_PASSED` 단위, G0(ii) 5종 중 2종만 열거, G3 정확도(정수 합 일치) 누락, G0(iii) 실패 시 M3 전체 보류는 과잉(보수 방향).
- P-DP 논리 타당. 공유 TokenizerManager 병목은 G를 1로 끌어당김(MGP-11). 원인 배제 절차 미등록(MGP-13).
- n=4 페어 bootstrap 백분위 CI ≈ 4쌍 부호 일치, boot 효과 미포함(MGP-7). G(8) 미정의 시 크기 규칙·N=1 회차 미등록(MGP-10). ANCHOR_DRIFT가 G(1) 드리프트 미검사(MGP-10).
- 하드캡 시 N=8이 먼저 잘림(MGP-14). C-4 부분 충족(MGP-11), C-5 arm 간 불일치 규칙 없음(MGP-12), C-6 설계 §5-3 이탈 미공시(MGP-15), λ0R-4 누락(D1).
- 정본 모순 없음. G(1) < 1이 정본 P1.7과 충돌처럼 읽힐 위험 → MGP-16.
- M4 참고: λ\*\_fused 하향 편향 가능(상한 6–10%, 과부하 회피 방향이라 무해), d44 비조율, trace shape·도착 과정 미등록.

## 7. caveat (문자 그대로 승계)
- **MGP-0**: "이 판정서의 `MIXED`는 M2g 규칙만 `GO-with-caveats`이고 M3 규칙은 `NO-GO`(D1·D2)다. M3는 rev2와 그 델타 감사 없이는 시작하지 않으며, 이 판정을 'M2.5 통과'로 요약하지 않는다."
- **MGP-1**: "M2g의 FAIL 단위(arm 대 (mode, N)), 시험하지 못한 셀의 처리, G0(ii) 금지 조합 5종 전부, G3 정확도(정수 합 일치)를 하네스 전에 등록하지 않은 M2g 결과는 인용하지 않는다."
- **MGP-2**: "G1 PASS(프롬프트 m의 중앙값 기준)는 프롬프트 절반 미만에만 걸리는 PD-mux 결함이 없다는 증거가 아니다. G1에는 분할에 노출된 프롬프트·토큰 비율을 병기한다."
- **MGP-3**: "`TP_NUMERIC_DIVERGENT`는 결함 판정이 아니며, 그 셀의 정확성 결함으로 서술하지 않는다."
- **MGP-4**: "G3 hang 0/500은 엔진이 hang에서 자유롭다는 뜻이 아니다(custom AR v1·cudagraph replay·실모델은 시험 범위 밖)."
- **MGP-5**: "manual 등가표와 자동 격자의 선택 함수 등가성은 N=1 코드 수준에서 확인됐다. G5(a) TV 결과는 잡음 기술치이며, 그것만으로 `AGNOSTIC_NOT_EQUIVALENT`를 붙이지 않는다(결정적 CPU 시험이 우선한다)."
- **MGP-6**: "`PARTITION_NOT_EXERCISED`가 붙은 G(N), 특히 shape A는 'PD-mux 엔진 모드의 용량비'일 뿐이며 'SM 분할 효과'나 'PD-mux SLO 이득의 스케일링'으로 쓰지 않는다. G는 cap 구속 과부하에서의 용량비이고 goodput이 아니다."
- **MGP-7**: "M3의 seed n=4는 (config, arm)당 boot 1개 안의 반복이라 boot 산포를 포함하지 않는다. |G(8)/G(1)−1| < 3%인 추세는 방향 라벨도 방향의 증거로 인용하지 않는다."
- **MGP-8**: "achieved/offered ≤ 0.90은 포화 증거가 아니다(λ0R-4 승계). 독립 포화 증거(`#running-req` = cap, `#queue-req` > 0)가 없는 λ\*는 용량으로 인용하지 않는다."
- **MGP-9**: "같은 (config, shape) 안에서 서로 다른 offered rate나 요청 수로 잰 두 arm의 λ\*로 만든 G는 인용하지 않는다(확장은 공동으로 적용한다)."
- **MGP-10**: "G(1)에 쓰는 N=1 회차, 기울기 bootstrap의 재표집 단위, G(8)이 정의되지 않을 때 크기 규칙의 분모를 하네스 전에 등록하지 않은 추세·크기 라벨은 인용하지 않는다."
- **MGP-11**: "TokenizerManager·detokenizer 측 병목은 클라이언트 send-lag와 cgroup throttle로는 검출되지 않는다. 서버측 프론트엔드 지표를 병기하지 않은 상위 N 용량·G 수렴은 'GPU 확장성'이나 'PD-mux 스케일링'으로 쓰지 않는다(C-4 승계)."
- **MGP-12**: "같은 config 안에서 arm 사이의 `max_total_tokens`·mamba cache·GPU별 사용 메모리(프로세스 목록 포함)가 다르면 그 G는 인용하지 않는다(C-5 승계, PD-mux 전용 GPU0 컨텍스트 가설 미검증)."
- **MGP-13**: "P-DP가 깨진 뒤의 '원인 배제'는 새로 등록하고 감사받은 진단으로만 할 수 있다. 그 전에는 TP 라벨을 인용하지 않는다."
- **MGP-14**: "하드캡 `TRUNCATED`로 N=8이 빠진 추세는 N ≤ 4의 관측으로만 서술한다."
- **MGP-15**: "d44는 1-GPU에서 고른 D를 고정한 비조율 arm이며(설계 §5-3의 N별 선택에서 이탈), N>1에서의 열위나 우위를 static 정책의 성질로 쓰지 않는다."
- **MGP-16**: "x8 노드의 G(1)은 정본 'agnostic PD-mux가 fused를 이긴다'(P1.7, KISTI, goodput)의 재검정이 아니며, 어느 방향이 나와도 그 정본을 강화하거나 약화하지 않는다."

## 8. rev2 처방
1. P-D1·P-D2 반영(CPU만). 변이 검사: "원시 ratio LO 판정"·"idx==2 문자 그대로" 변이본 KILLED.
2. MGP-1·MGP-10 미등록 항목을 숫자로 고정(LO 창, 분모, G(1) 회차, bootstrap 단위, X·Y, FAIL 단위).
3. 권고: 확장은 config·shape 공동, G5(a)를 결정적 CPU 시험으로, G1 노출 비율·불일치 프롬프트 비율 기준, arm 간 cell_meta 동등성·GPU별 프로세스 기록, M3 dry-run 예산을 하드캡 30 h에 대조.
4. rev2 델타 감사는 D1·D2·MGP-1/10만 대상. 이 판정서의 표·처방을 등록 상수로 승격하지 말고 재계산(게이트 #110). L = 8.9 s는 λ0 d44(homog5) 값.
