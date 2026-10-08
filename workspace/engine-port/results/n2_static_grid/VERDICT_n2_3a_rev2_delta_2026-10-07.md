# 판정서: N2(3-a) 사전등록 rev2 델타 감사 (claims-auditor, 2026-10-07)

> 감사자는 읽기 전용이라 메인 세션이 감사 보고 본문을 저장했다(2026-10-07). GPU 0, 새 성능 판정 0건, Claim 등급 변경 0건.
> 대상: `PREREG_N2_3A_2026-10-07.md` rev2(752행, sha256 `e215ef1d6f2f2d91330de9ef4b4b181269d77dc0cc1cc5523deaa6f724dcc1d0`). 대조본 rev1 sha256 `3eff5885…829b`(scratchpad 백업, diff 대조).

## 등급: `NO-GO`, 死因 5건 (D-2…D-6)
- D-1 해소. D-2·D-3은 rev1부터 있었고 1차 판정이 놓침. D-4·D-6은 rev2가 새로 만듦(D-4 한 사례는 §21 미공시 변경). D-5는 rev2의 N2C-5 수리가 만든 규칙 모순.
- 전부 국소 수리 가능하나 반전·도달 불가가 수치로 성립해 caveat로 낮추지 않는다. rev3는 수리 반영 뒤 다시 델타 감사. 이 판정서는 rev3를 미리 승인하지 않는다.

## 死因
### D-2 (N3, 게이트 #247): 산포 0 셀은 측정 창을 만들 수 없다 — 격자 60점 중 30점
- 등록: §8-3 c_in = max(256, c − X), X = 1.5×(예보 최대 산포)/2 + 64; §6-2 W = trim 16 뒤 ctx_p50 ∈ [c−64, c+63]인 iteration 128개; §9-1 시작 ctx_p50 > c−64 → `CTX_UNREALIZED`.
- 산술: 단독 구간 ctx_p50은 iteration마다 +1 → 띠 안 최대 128개 → 창 완결엔 시작 ctx_p50 ≤ c−80 필요. 산포 0% 셀은 X=64, c_in=c−64, prefill이 토큰 1개 생성 → 시작 ≥ c−63 > c−64 → `CTX_UNREALIZED` 확정(trim 빼도 ≤111).
- 해당: b=1 × c 3개, (8,512), (8,2048), (32,512) = 6 (b,c) × 5 D = 30점. 귀결: `D_PROFILE_COMPLETE`·`PROFILE_COMPLETE` 도달 불가; E-α (8,2048) 2셀 ON A1 불성립 → `DS3_WITHIN_3_ALL` 도달 불가; OBS ON44 (1,2048) 창 불성립.
- U-2 (a)(프로브 후 동결)에서도 b=1 산포 0 → X=64 그대로. U-2 (b) "X×2"에서만 도달 → 최상위 라벨 도달 가능성이 U-2 선택에 구조적으로 갈림.
- E-α 트리거도 같은 결함: "b개 모두 첫 토큰 후 ≥160 토큰" 고정 → A1 완결엔 여유 m이 16 ≤ m ≲ 32. 산포 있는 E-α 셀((8,8192) 2–5%, (32,·) 8–22%)은 m이 벗어날 것으로 예보(설계 계산, 인용 금지).
- 1차 판정 반전 시험 표 "창 128/trim 16/±64 → 실현가능성만, caveat"는 **틀렸다** — 정의역 퇴화다.

### D-3 (N3/N2): OFF boot E-α 셀에 단일 π → ρ_off 도달 불가
- §9-1이 같은 셀에 `REAL_D_ALONE(W, pi(cell))`·`REAL_D_COLOC(Wco, pi(cell))`을 한 π로 적용. §10-1은 OFF E-α에서 A1·A2를 (0,108) 단독(idx5), C를 D(idx1/4)로 둠.
- π=d쌍이면 OFF A1 PE-iter(idx d)=0 → `D_UNREALIZED`; π=(0,108)이면 동거 술어 (b)=0 → `COLOC_UNREALIZED`. 문자대로면 C_off 32복제 전부 무효 → ρ_off 8셀 `UNRESOLVED_N_BELOW_4` → `DS3_UNRESOLVED` 고정.
- 반전: 셀 (16,8,2048) C_off 4짝 통과·ln ρ_off = 0.05×4 → 창별 π: `COLOC_SLOWER_GE3`→`DS3_MISMATCH_OBSERVED_1`; 문자 규칙: `UNRESOLVED_N_BELOW_4`→`DS3_UNRESOLVED`.
- ON A2 유효성 규칙 미등록(A_on은 A1·A2 둘 다 필요, §10-2).

### D-4 (N1): 유효 집합 위 라벨은 항등식 — 2건(교훈 9)
- (i) `NEG_CONTROL_HELD`(§11-3): rev1 "OFF boot의 모든 단독 창" → rev2 "모든 유효 단독 창"(**§21 미공시**). VALID 정의상 OFF는 PE-iter(idx5)≥0.99, ON은 idx5 ≤0.01 → 항등, `NEG_CONTROL_FAILED` 도달 불가, `PROFILE_NOT_BUILT` 두 번째 분기 죽은 코드. 반전: OFF16 r3 (108,8,2048) PE-iter 0.90 → rev2: `D_UNREALIZED`, 108 n=7 `POINT_REALIZED`·`NEG_CONTROL_HELD`; rev1 의도: `NEG_CONTROL_FAILED`→`PROFILE_NOT_BUILT`.
- (ii) Job 수준 `COLOAD_NOT_JOINED`(§7-2(f)): "모든 유효 E-α 복제가 NOT_JOINED"이나 JOINED는 곧 `BS_UNREALIZED`로 무효 → 거짓 불가. F-N2-8 첫 반증 조건도 도달 불가. 반전: 32복제 중 31 JOINED·1 NOT_JOINED → 라벨 `COLOAD_NOT_JOINED`와 §16-1 허용 문안 그대로 출력.

### D-5 (N2): E-α `COLOC_WITHIN_3` 규칙 두 곳 모순(D-1과 같은 형태)
- §10-3 코드는 bootstrap CI(`-ln1.03 < lo and hi < ln1.03`), 산문·주석은 "CI는 기술 병기", §17-3 "라벨 조건은 4/4 하나(N2C-5)".
- 반전: `analyze.paired_bootstrap_ci`(samples=2000, seed 1)에 ln ρ = {0, 0, 0.01, 0.035} → CI [0.0000, 0.0263] → 코드: `COLOC_WITHIN_3`; 4/4: 0.035 > 0.0296 → `COLOC_UNRESOLVED`; 8셀 요약 `DS3_WITHIN_3_ALL` ↔ `DS3_UNRESOLVED`.

### D-6 (N2): R7 검사 범위 미지정
- rev1 `R7_marker(boot(r))` boot 단위 → rev2 §7-1(R)·§9-1 `recorder_alive(cell, r)` "셀 구간 Δdropped=0, R7 표식 없음"만.
- 코드상 R7(`_phase_events_disable`, `multiplexing_mixin.py:2169-2174`)은 boot 나머지 전체에서 PHASE_EVENTS를 끔, 로그 1회.
- 반전: ON16 r2 셀 13(첫 E-α)에서 R7 → boot 단위: 프로파일 12셀 무효, D16 n=3 `UNRESOLVED`→`PROFILE_PARTIAL`; 셀 단위: 12셀 유효 n=4 `COMPLETE` 가능. 셀 단위면 R7 뒤 셀이 `WINDOW_SHORT`/`CTX_UNREALIZED`로 분류 → 교훈 21 오분류 재발.

## 질문별 결론
1. **D-1 해소.** §9-2 무조건 합산, rev1 §17-7(:561) 삭제, §6-3·§11-3·§12-1·§13·§16-1·§17-7 일관, grep(`FIXED|합산|n ≤ 8`) 다른 분기 없음. 단 같은 형태 결함이 D-3·D-4·D-5·D-6에 남음(교훈 85).
2. **R-2…R-10**

| 수리 | 반영 | 새 자유 표면 / 결함 |
|---|---|---|
| R-2 | 반영(기록기 검사 선행, 분모 전체 시도, D108 sticky 문구 금지) | R7 범위 미지정(D-6). E-α 셀이 D_label의 A에 포함되는지 미지정(D-3과 겹치면 OFF E-α 16복제가 D16 `D_UNREALIZED`로 계상, 20%라 반전 없음) |
| R-3 | 반영 | 없음 |
| R-4 | 반영(`telemetry.py:91-114` emit 시점 확인) | boot 마지막 셀은 뒤 레코드 없어 손실 불가시, "기록만"이라 fail-closed 아님(F-6) |
| R-5 | 반영 | `Prefill batch` 줄도 `cuda graph: False` 찍음(`srv_off_r030_4589.log` 등) → `Decode batch` 필터 명문화; 로그 초 단위 벽시계→CLOCK_MONOTONIC 매핑 미등록(F-5). 필드 존재 확인(`srv_g_r022_2084.log` Decode 48줄 전부 True, `scheduler_metrics_mixin.py`) |
| R-6 | 반영(§14) | Q4 |
| R-7 | 반영 | "경과 시간·직전 셀이 같다" 거짓(ON 프로파일 셀은 D, OFF는 108; F-1, N2C-20). 108 점 seed f(셀, round)의 셀 id가 OFF16/OFF44 구분하는지 미정의(F-8) |
| R-8 | 반영(`profile.py:310-335` 일치) | 없음 |
| R-9 | 반영 | 없음 |
| R-10 | 반영 | D-4(ii) |

- 새 라벨 우선순위: 측정 실패(BENCH→PAYLOAD→RETRACT→RECORDER→CG)가 술어보다 먼저 — 타당. `COLOAD_PAYLOAD_INVALID`·`CTX_MISMATCH`(§10-2)는 §9-1 두 부류 어디에도 없음(F-3). §9-1 셀 키 (cond,D,b,c)가 프로파일 (d,8,2048)과 E-α (d,8,2048)을 구분 못 함 — "대체값 없음" 조항이 막아 반전 없으나 `cell_kind` 분기 명시 필요(F-3). OBS는 §11-4 별도 유효성 규칙 우선, 의사코드 분기 필요.
- E-α 4/4 축소: GE3엔 N2C-5와 정합, WITHIN엔 모순(D-5). 의사코드 `paired_bootstrap_ci(mean, rhos, B=2000, seed=1)`는 실제 시그니처(`analyze.py:115`, `(baseline, proposed, samples, seed)`)와 다름(F-7).
3. **정지 규칙·r4**: 결측 = 등록 순열 suffix → 실현 비용과 무관(성립). 단 r4 꼬리 `OFF16, OFF44, ON44, OBS44` → 결측이 D44·E-α d=44에 결정론적으로 몰림(N2C-17). ABBA: 44 짝 r1–r4 ON,OFF,ON,OFF 먼저, 16 짝 ON,OFF,OFF,ON → 둘 다 2/2, 인접 매 round 성립. 거짓 출처(교훈 80): §5 "rev1의 ABBA 불균형(44 짝 ON 먼저 3회)을 고치고"는 거짓(rev1 이미 2/2, 실제 효과는 꼬리를 OBS44로 옮긴 것뿐); "OBS44는 ON44와 인접"은 r3(`ON44, OFF44, OBS44`)에서 거짓 — 처방 r3 = `OBS44, ON44, OFF44, ON34, OFF16, ON16, ON24`(F-1).
4. **예산**: 산술 재현(round 합 (a) 6,593/10,015, (b) 5,224/8,139; ×4+210 = 26,582/40,270/21,106/32,766, ±5 s 일치; 360×22.48 = 8,093, ×35.5 = 12,780; 하드캡 (a) 44,300 ≈ 1.1×40,271, (b) 36,000 < 1.1×32,767 = 36,044). 22.48 s는 N3 모형(max(span, N/λ\*)+drain) 잔차라 bench_serving 기동·시험 요청·모형 오차가 섞였고 N2 셀 모형으로 옮길 근거 없음, 방향 미정. 35.5 s는 42표본 최대값일 뿐 상한 아님. 死因 아님(fail-closed, 결측은 순열이 정함). U-2 (b): rev1 "X×2 + 예산 +30%"에서 rev2가 "+30%"를 **공시 없이 삭제**, X×2(+D-2) 반영 표·입장 최악 비용 없음, 1차 "τ×2"도 미반영(F-4). U-2 (a) 자기 적용: Workspace boot 벽시계(59 s) ≠ Job admit→health(104 s)(교훈 273) → 프로브는 셀당 상수(τ_d, τ_span, loadgen 호출 비용)만 갱신, boot당 170.5 s는 Job 실측 유지, 프로브 `cpu.max` 기록(F-4). 프로브 없이 봉인 시 허용 문안 = N2C-18.
5. **engine 권고 7/7 반영**(§3-1, §4-2·§9-1, §7-2(f), §8-2(ii), §7-2(c), §6-2, §7-2(b)·§12-2, §6-1). S3 비무효화 처리 ENGINE_CHECK와 일치. 남는 결함은 D-4(ii).
6. **승계·U**: N2C-1…16 판정서와 §20-1 줄 단위 diff **바이트 동일**(16/16). U-3·U-4·U-5(a)·U-6(a)·U-7(a)·U-8 1차와 일치. 남은 사용자 결정은 U-1·U-2·U-5(b)·U-4 시점 외에 (i) 하드캡 수치(12.3 h/10.0 h, 운영 지침 초과) 명시 승인 (ii) §18-2 "취소 구현 불가면 (a)" 자동 전환 재승인. §0 머리말은 U-1·U-2만 적음(F-9).

## 반전 시험 표
| 자유 표면 | 민 범위 | 판정 변화 | 근거 |
|---|---|---|---|
| X 하한 +64 vs trim 16 + 생성 1 | 등록값 | **COMPLETE 도달 불가**(D-2) | 시작 ≥ c−63 > c−64, 띠 안 ≤111 < 128, 30/60점 |
| OFF E-α π | d쌍/(0,108)/창별 | **SLOWER_GE3 ↔ UNRESOLVED**(D-3) | 문자 규칙 C_off 0/32 유효 |
| NEG_CONTROL "유효" | VALID/전체 | **COMPLETE ↔ NOT_BUILT**(D-4 i) | PE-iter 0.90 복제 1개 |
| Job `COLOAD_NOT_JOINED` | 유효/전체 | **NOT_JOINED ↔ JOINED**(D-4 ii) | 31/32 JOINED에도 NOT_JOINED |
| WITHIN 규칙 | CI/4/4 | **WITHIN ↔ UNRESOLVED**(D-5) | {0,0,0.01,0.035} → [0.0000, 0.0263] |
| R7 범위 | boot/셀 | **COMPLETE ↔ PARTIAL**(D-6) | 셀 13 R7 → D16 n=3 vs 4 |
| 108 합산 | 무조건 | 없음 | D-1 해소 |
| r4 순열 | rev1/rev2 | 라벨 없음 | 2/2, 꼬리만 바뀜(N2C-17) |
| 정지 vs 건너뛰기 | 정지 | 비용 상관 없음 | suffix |
| 셀당 고정비 22.48→35.5 | rev7 범위 | 입장 수만(fail-closed) | 28/28 설계상 성립·미검증 |
| 셀 키 충돌 | 대체 허용 | 없음 | "대체값 없음" → F-3 |
| CG 로그 매핑(초 단위) | ±1 s | 없음 | N2C-21 |

## 필수 수리 (rev3)
**死因 수리**
- **D-2**: X_min ≥ 64+16+2(또는 "시작 ctx_p50 ≤ c−80" 설계 조건). co-load 트리거를 고정 160 대신 A1 창 완료 기준으로(예: 클라이언트 측 (c_in + 수신 토큰) 하측 중앙값 `int((n−1)·0.5)` ≥ c+63+8). 셀당 약 18 step 추가, 360셀 합 ≲250 s(설계 계산).
- **D-3**: π 창별 — π_alone ON→idx d, OFF→idx5; π_coloc = idx d. OFF E-α 복제를 OFF A1(idx5)으로 게이트할지 하나로 고정. ON A2 유효성 규칙(단독 술어, trim 뒤 128, 어떤 ctx든 허용, CTX_MISMATCH가 정합 담당) 등록.
- **D-4**: NEG_CONTROL은 (R)·(a)를 통과한 모든 단독 창 위에서; Job COLOAD는 PAYLOAD·RECORDER·CG 통과 모든 E-α 복제 위에서, k/n JOINED 출력. 각 정의에 변이 테스트(VALID로 바꾼 변이본에서 실패, 교훈 53).
- **D-5**: WITHIN 하나로 고정 — 권고 4/4 |ln ρ| < ln 1.03(GE3와 대칭). §10-3·§17-3 문장을 GE3·WITHIN 각각에 맞게.
- **D-6**: R7 boot 단위로 복귀(rev1). R7 뒤 셀은 `RECORDER_UNPROVEN`.

**그 밖의 필수 수리**
- **F-1** 거짓 출처 3건(ABBA 불균형 문장 삭제, r3 OBS 인접 처방, R-7 "경과 시간 같다" 삭제).
- **F-2** §21 완전성(NEG "유효", U-2 (b) +30% 삭제, R7 범위, rev1 `D_WIRING_FAILED` 제거 등재, "위 변경 외 바꾸지 않았다" 정정).
- **F-3** §9-1 `cell_kind`(프로파일/E-α/OBS) 분기, `COLOAD_PAYLOAD_INVALID`·`CTX_MISMATCH` 부류 지정.
- **F-4** U-2 (b) 예산표·입장 최악 비용·하드캡을 X×2(+D-2)로 재계산하거나 "(b)는 rev 증가 필요" 명시. U-2 (a) 프로브는 셀당 상수만 갱신.
- **F-5** CG 증거 `Decode batch` 줄 한정, 벽시계→monotonic 매핑·경계 줄 처리 등록.
- **F-6** boot 마지막 셀 뒤 셀 구간 밖 비측정 요청 1건으로 레코드 생성, 아니면 그 셀 `RECORDER_UNPROVEN`.
- **F-7** `paired_bootstrap_ci` 실제 시그니처로.
- **F-8** seed f(셀, round)의 셀 id 정의(108 OFF16/OFF44 구분, D 간 같은 프롬프트 여부).
- **F-9** 하드캡 승인·(b)→(a) 전환을 사용자 결정 표에 추가, §0 머리말 정정.

## 신규 caveat (인용 금지 문장, 문자 승계)
- **N2C-17** 정지 규칙의 결측은 등록 순열의 꼬리에서만 생기고, 그 꼬리는 D44(ON44·OFF44·OBS44)에 몰려 있다. D44 점·E-α d=44·관측자 라벨의 `UNRESOLVED`는 순열의 귀결이며 D44의 성질이 아니다.
- **N2C-18** 셀당 고정비 22.48 s(최악 35.5 s)는 bench_serving 기반 N3의 측정 창 − 모형 잔차를 옮겨 쓴 값이며, trace_loadgen 셀 호출 비용의 측정이 아니다. U-2 프로브 없이 봉인했다면 §14 표 전체와 입장 수·결측 boot 수를 예산 예보의 성공/실패 근거로 인용하지 않는다.
- **N2C-19**(D-5를 CI 규칙으로 고정할 경우에만) n = 4에서 `COLOC_WITHIN_3`는 35점 bootstrap 분포의 95% 구간이 ±3% 안이라는 뜻이며, 한 짝이 3%를 넘어도 나올 수 있다(ln ρ = {0, 0, 0.01, 0.035} → CI [0.000, 0.026]).
- **N2C-20** ρ_off 짝의 ON/OFF boot는 같은 round·같은 셀 위치이지만 boot 안 경과 시간은 다르다(ON 프로파일 셀은 D, OFF는 108에서 돈다). 열·클럭 상태 차이가 ρ_off에 섞인다.
- **N2C-21** cudagraph 재생 증거는 서버 decode 로그 표본(40 iteration당 1줄, 초 단위 벽시계)이며, 창 안 모든 iteration의 재생을 실증하지 않는다.

## 반증 실패(공정 기록)
D-1 수리, 정지 규칙 비용 독립성, ABBA 짝 균형, engine 권고 7건, N2C 문자 승계, U-3…U-8 처리에서 반증 없음. 실행 전 `CONFIRMED` 아님.

## 자기 적용
1차 판정 반전 시험 표 "창 128/trim 16/±64 → 실현가능성만"을 독립 재계산해 틀렸음을 확인(D-2). 처방은 클라이언트가 아는 값만 쓰고(D-2), 기존 술어 단계 재사용(D-4), r3 재배치는 짝 순서 불변, U-2 (a)는 Workspace/Job 차이 반영, 새 노브 결합 없음(교훈 250).

## 근거 파일
`PREREG_N2_3A_2026-10-07.md`, `VERDICT_n2_3a_rules_2026-10-07.md`, `ENGINE_CHECK_coload_2026-10-07.md`, rev1 scratchpad 백업, `results/n3_capacity/REV7_DRAFT_NOTES_2026-10-07.md`, `results/n3_capacity/n3b_pdmux-n3-capacity-38ed3f4-20261006092813/srv_*.log`, `src/multiplex/{telemetry.py:91-114, multiplexing_mixin.py:2169-2174, profile.py:310-335}`, `benchmarks/pdmux_eval/analyze.py:115`, `sglang_engine_dev/python/sglang/srt/observability/scheduler_metrics_mixin.py`.
