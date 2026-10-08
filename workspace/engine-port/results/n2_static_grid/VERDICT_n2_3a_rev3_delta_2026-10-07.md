# 판정서: N2(3-a) 사전등록 rev3 델타 감사 (claims-auditor, 2026-10-07)

> 감사자는 읽기 전용이라 메인 세션이 감사 보고 본문을 저장했다(2026-10-07). GPU 0, 새 성능 판정 0건, Claim 등급 변경 0건.
> 대상: rev3(933행, sha256 `6529cedd2c18c125b0e828e4ead8c5c6bb2f831cbe39ea6c600d7353dc349efb`). 대조본 rev2 sha256 `e215ef1d…c1d0`(일치), diff 448줄 대조.

## 등급: `NO-GO`, 死因 1건(D-7)
- D-2…D-6 원래 결함 전부 해소(등록 상수로 독립 재계산, 아래 Q1).
- D-7은 rev1부터 있던 결함으로 1·2차 판정과 rev3 §8-6 자기검사가 모두 놓쳤다. D-2 처방(트리거를 A1 완료 기준으로)이 출력 길이 O를 손대지 않아 남은 것 — 처방자가 함께 죽은 경우(게이트 #113 계열).
- 한 줄 수리 가능하나 반전이 τ와 무관한 산술로 성립해 caveat로 낮추지 않는다. rev4는 D-7과 필수 수리 반영 뒤 다시 델타 감사. 이 판정서는 rev4를 미리 승인하지 않는다.

## 死因 D-7 (N2·N3): E-α 셀의 출력 길이 O와 창 뒤 취소 기준에 "어느 창"인지 미등록
- 등록: §8-4 "첫 종료가 창 뒤에 오도록 O ≥ (창 끝까지 필요한 생성 수) + (산포 예보 상한) + 64"; §4-2-5 "창 종료 조건이 관측되면 남은 스트림 취소(U-1 (b))". E-α 전용 O·취소 조건 없음. §6-2 "측정 창 W"는 단독 창 = E-α에서는 A1.
- 반전(W = A1): (8,2048) 산포 0 → c_in 1952, O = (2111−1952)+0+64 = 223, b개 모두 서버 ctx c+127에서 종료; 트리거는 서버 ctx ≥ c+71에서 발화(클라이언트 ≤ 서버라 더 늦을 뿐) → W_co 안 `bs == b` iteration ≤ 56 − 지연 < 128(τ 무관) → ON·OFF 전 복제 `BS_UNREALIZED`. (8,8192)·(32,2048)은 C를 지나도 A2가 짧음(A2에 W_co 예보 하한 240 + 144 필요 vs 첫 종료 트리거 뒤 약 193/301) → ON A2 `WINDOW_SHORT` → `A_on_valid` 거짓 → ρ 짝 0. 귀결: b=8 E-α 4셀 강제 `UNRESOLVED_N_BELOW_4`, `DS3_WITHIN_3_ALL` 도달 불가; U-1 (b)에서 "창 종료 = A1 끝"으로 취소하면 co-load 전 스트림이 끊겨 8셀 전부 무효.
- "창 = A2 끝"으로 읽으면 `DS3_WITHIN_3_ALL` 도달 가능(§8-6 주장). 판정이 해석 선택으로 갈림(56 < 128은 τ 무관).
- rev3 자기검사 실패: §8-6은 `DS3_WITHIN_3_ALL`을 "예"로 적으면서 `WINDOW_SHORT` 행에 "E-α A2에서 co 종료 뒤 128 단독 iteration 전에 b개 종료"를 가능 결과로 적음 — O 공식으로 막지 않음. §14-2 비용 모형("E-α: co span + A2 144단계")은 A2까지 생성을 가정하나 규칙 아님.

## 질문별 결론
### 1. D-2…D-6, F-1…F-9 (독립 재계산)
- **D-2 해소**(프로파일 W·E-α A1): 산포 0 셀 시작 = c−95, trim 16 뒤 c−79 ≤ c−64, 띠 128 확보. "시작 ≤ c−80 ⇔ 띠 완결(bs 하락 없을 때)"은 정확 동치(c−79면 127, c−80이면 128). X ≥ 81 충분, 82·96 성립. 트리거 L_(int((b−1)·0.5)) ≥ c+71은 창 정의와 같은 하측 중앙값, 같은 c_in이라 길이 순서 = 수신 순서, 클라이언트 ≤ 서버 → **조기 발화 불가**; 중앙값 위 요청은 창이 쓰지 않으므로 무관; 늦은 발화는 C ctx만 올림. 결함은 O 쪽(D-7). 산포 허용 "s ≤ 1.5ŝ + 30"은 lead = s/2 가정 — 실제 (8,8192) 2요청 span 4개 1/3, (32,2048) 10/10/10/2 ≈0.55, (48,512) c_in 343에서 47+1 ≈1.0 → 허용 약 0.75ŝ+15, 서술 거짓(R3-7); 라벨은 실현 s가 정해 반전 없음(N2C-26). "c=512에서 c_in ≥ 416"도 틀림(b=48이면 343), 결론(하한 256 미발동)은 성립.
- **D-3 해소**(창별 π로 ρ_off 구조 차단 해제).
- **D-4 (i) 해소**(모집단 VALID 무관, OFF PE-iter 0.90이면 FAILED). 새 결함 2건은 Q4.
- **D-4 (ii) 해소, 엔진 코드로 비항등 확인**: `scheduler_output_processor_mixin.py:988-992` `finished_output` 가드로 prefill에서 끝난 요청은 filter 실패로 decode에 끼어도 두 번째 응답 없음 → `completion_tokens == 1` 유지 → filter 실패형 합류는 PAYLOAD 통과해 M에 남고 서명으로 검출. `max_new_tokens` 누락형은 runner `ABORT_TRACE`·PAYLOAD가 막음(의도).
- **D-5 해소**: 4/4 규칙. {0,0,0.01,0.035} → 0.035 > 0.0296 → `COLOC_UNRESOLVED`; {0.01,−0.01,0.02,0} → WITHIN. CI 라벨 미사용, N2C-19 미등재 정당.
- **D-6 해소**: §7-1 (R)·§9-1 `R7_marker(boot(r))` boot 단위 일치.
- F-1 해소(44 짝 ON,OFF,ON,OFF / 16 짝 ON,OFF,OFF,ON 2/2; OBS44 네 round 모두 ON44 인접, OBS→ON·ON→OBS 각 2회; 거짓 출처 3건 철회). F-2 대체로 해소(단 §18-2 U-1 행·§14-3 rev2 수치 잔존, Q6). F-3·F-4 해소. F-5 해소([⌈wall(start)⌉, ⌊wall(end)⌋−1] 완전 포함 초). F-6 해소(fail-closed). F-7 해소(`analyze.py:115` 일치). F-8 해소(Q5). F-9 부분 해소(Q8).

### 2. §8-6 도달 가능성 표 반증
- 확인: `VALID` 두 행, `CTX_UNREALIZED`(s=966 → 2074 > 1968), `NEG_CONTROL_FAILED`, `COLOC_*` 네 행, `OBSERVER_*`, `NOT_ATTEMPTED_BUDGET`((a) 최악 코너 여유 4,484 s 밖에서만), `PROFILE_NOT_BUILT` (ii).
- **`CTX_MISMATCH` 산술 거짓**: ctx_mid(C_off) − A2_on 평균 ≈ W_co,off/2 − W_co,on − 80(트리거 같을 때); 표 시나리오(OFF 300 iteration 더 김)는 70 − W_co,on/2 < 0으로 구간 안. c=2048에서 무효엔 W_co,off > 2·W_co,on + 570 필요, 예보 범위 240–944 안 최대 초과 152 < 205 → 사실상 도달 불가 → "도달 불가는 `OUT_OF_DOMAIN` 하나뿐" 거짓(교훈 80). 가드 미발화 = 보간 구간 안이라 추정엔 무해(N2C-23).
- **`BS_UNREALIZED` 첫 시나리오(max_new_tokens 누락 → JOINED) 오추적**: runner `ABORT_TRACE`, 못 막아도 `completion_tokens = 128` → `COLOAD_PAYLOAD_INVALID` 먼저. 둘째 시나리오(C 동안 b개 중 1개 종료)로는 도달.
- **`COLOAD_JOINED_k_of_n`** 실제 경로는 "filter 실패 + completion_tokens 1 유지"와 S1 인공물 둘뿐 — 명시 필요.
- **`DS3_WITHIN_3_ALL`**: D-7로 해석에 따라 도달 불가.
- 연쇄: `D_UNREALIZED`(ON16 idx5 2/128)·`D_REALIZATION_FAILED` 시나리오는 §11-3 ON쪽 절(idx5 ≤ 0.01)도 어겨 프로파일 라벨 `PROFILE_NOT_BUILT`(Q4 (b)).

### 3. D-3 "OFF E-α는 OFF A1·A2로 게이트 안 함"의 fail-open 여부
- 기록기 생존 우회 없음(`measurement_ok` boot R7·셀 Δdropped·CG 적용, `Wco.has_record_gap` 별도). CTX_MISMATCH 우회 없음(C_off를 ON A_on에 대조).
- 남는 부분: OFF boot 런타임 sticky OFF 실현은 boot 배선 검사(§11-1 abort)와 Job 전역 NEG_CONTROL로만 확인, E-α·DS3 라벨은 NEG_CONTROL과 무관하게 출력. 합성 예: OFF16 r2 E-α A1 idx5 0.90 → ρ_off 짝 유효·E-α 라벨 출력·동시에 `PROFILE_NOT_BUILT`. 등록 고정이라 자유 표면 아님, 死因 아님. §17-11 "결과 문서가 병기"는 §16-2 필수 병기 목록에 없음 → N2C-22·R3-3.
- 하네스층: co span 식별 규칙(burst span과 구분) 미등록, 극단 산포면 burst 마지막 span 전 트리거 가능 — §7-2(c) Σn_new==K가 fail-closed로 잡을 것으로 보이나 등록 필요(R3-6).

### 4. D-4 새 모집단의 항등식·공허 참
- (a) 공허 참: "OFF 창 전부 ≥0.99 ∧ ON 창 전부 ≤0.01", `UNRESOLVED`는 전체 공집합일 때만 → OFF 부분모집단만 ∅(예: OFF 8 boot 전부 R7)이면 `NEG_CONTROL_HELD`, ON 점 있으면 `PROFILE_PARTIAL` — OFF 근거 0인 음성 대조 통과. 국소 수리 R3-3.
- (b) 오명명 결합: ON쪽 절이 비-VALID ON 창에도 적용 → 양성 대조 실패(sticky 누수, 본래 `D_UNREALIZED`)가 `NEG_CONTROL_FAILED`·`PROFILE_NOT_BUILT`로 전역화. fail-closed이나 §9-3 국소화 설계와 충돌. 1차 처방이 만든 결과(게이트 #113). ON쪽 절은 VALID 위에선 항등, 비-VALID 포함 시 전역 결합. 권고: NEG_CONTROL은 OFF 창만, ON 누수는 D 라벨 채널로(또는 원인 부류 병기).
- (c) `COLOAD_JOINED_k_of_n`: 항등식 아님. 단 S1 "== running_bs"는 b개 자신의 종료(bs < running_bs)도 JOINED로 세 k가 부풀 수 있음 — ρ는 `BS_UNREALIZED`라 E-α 라벨 불변, Job 라벨·F-N2-8 반증 조건만 오염(R3-5, N2C-25).
- `NEG_CONTROL_UNRESOLVED`(전체 ∅)·`COLOAD_UNRESOLVED`(n=0)는 공허 참 아님.

### 5. F-8 시드 정의 교락
`--disable-radix-cache`(§3-1), mamba radix off(P3 48), boot마다 서버 재기동 → boot 간 KV·prefix 재사용 없음. boot 안 셀 id (종류,b,c) 유일, 프로파일 (d,8,2048)과 E-α (d,8,2048)은 종류가 달라 프롬프트 다름, OBS는 OBS boot에만 → **캐시 교락 없음**. 프롬프트 동일로 D 간·짝 간 프롬프트 분산 제거(이득). 남는 것: round 간 프롬프트가 달라 round(시간·열) 효과와 앨리어스 — 무작위 토큰·`ignore_eos`·고정 길이에서 무시할 만하다고 예상하나 미측정(N2C-27). co-load 8요청 시드 미정의(radix off라 무영향, 하네스층 명시).

### 6. §21-3 단일 출처 표본 검증
- 라벨 규칙 9계열 진술 충돌 없음(`CTX_UNREALIZED` §6-2·§8-3·§9-1 모두 c−80, R7 §7-1·§9-1 boot). 단 §6-2 "128개 못 채우면 `WINDOW_SHORT`"는 §9-1 순서와 부분 어긋남(같은 부류라 반전 없음, R3-9).
- **수치 단일 출처 실패(교훈 85)**: §18-2 U-1 행(784행)이 rev2 값(7.44/11.35 h/44,300 s, 5.93/9.27 h/36,000 s) 그대로, 같은 표 하드캡 행·§14-2는 44,900/36,600. 36,000은 1.1×최악(36,597) 규칙 위반. §14-3 boot 비용(1,128–1,208, 816–840, OBS44 252, 252–1,851 s)도 rev2 값. 반전: 등록 최악 코너에서 (a) 44,300−40,416 = 3,884 s, (b) 36,000−32,912 = 3,088 s, 둘 다 ≥358 s라 28/28 입장으로 라벨 불변 → 死因 아님, 필수 수리 R3-2(사용자 승인 표가 모순이므로 봉인 전 필수).

### 7. 예산 산술 재현
| 항목 | 계산 | 결과 |
|---|---|---|
| (a) 보통 | 6,675×4+210 = 26,910(등록 26,913), +224 = 27,137 | 7.54 h ✓ |
| (a) 최악 | 10,141×4+210 = 40,774, +600 = 41,374 | 11.49 h ✓ |
| (a) 하드캡 | ⌈1.1×40,774 = 44,851⌉₁₀₀ | 44,900 ✓ |
| (b) 보통 | 5,307×4+210 = 21,438(등록 21,443), +224 | 6.02 h ✓ |
| (b) 최악 | 8,265×4+210 = 33,270, +600 = 33,870 | 9.41 h ✓ |
| (b) 하드캡 | ⌈36,597⌉₁₀₀ | 36,600 ✓ |
| 마지막 boot 여유 | 44,900−40,416 / 36,600−32,912 | 4,484 / 3,688 s ✓ |
rev2→rev3 증분 round당 +82 s(×4=328, 등록 +335), 최악 +126 s(×4=504, 등록 +503) — 반올림 범위. D-7 수리(E-α O 연장)는 U-1 (a) 꼬리 비용을 늘릴 수 있어 rev4에서 재계산.

### 8. 남은 사용자 결정 완전성
등재: U-1, U-2, 하드캡 승인, (b)→(a) 전환, U-5(b), U-4 시점. 빠진 것: (i) U-1 행 수치가 §14-2와 모순(R3-2) (ii) **U-2 (a) 프로브 뒤 재계산 예산·하드캡 재승인**(머리말 순서상 하드캡 승인 뒤 프로브가 수치를 바꿈) (iii) 프로브가 τ 의존 X·O·트리거 상수를 바꾸면 봉인 전 `n2_reach_tests` 재실행·§8-6 재대조 등록 없음("상수 동결 후 변경만 rev 증가"라 프로브 갱신이 감사 밖).

## 반전 시험 표
| 자유 표면 | 민 범위 | 판정 변화 | 근거 |
|---|---|---|---|
| **E-α O·취소의 "창" 참조** | W(A1) ↔ A2 끝 | **`DS3_WITHIN_3_ALL` 도달 ↔ 불가, b=8 4셀 강제 UNRESOLVED(D-7)** | 트리거 c+71 → 종료 c+127, ≤56 < 128(τ 무관); A2에 W_co ≥ 240+144 vs 첫 종료 193/301 |
| X 하한 | 82–96 | 없음 | 시작 c−81…c−95 ≤ c−80 |
| CTX_UNREALIZED c−80 | 등록값 | 없음 | 정확 동치 |
| 트리거 여유 +8/클라이언트 지연 | 0…∞ | 없음 | 조기 발화 불가 |
| CTX_MISMATCH "10%" 기준 | 두 해석 | 반전 미성립 | W_co,off > 2.2·W_co,on + 190 필요, 등록 시나리오(+300)는 구간 안 |
| OFF E-α 게이트 | 고정 | 자유 표면 아님 | N2C-22 |
| NEG_CONTROL OFF ∅ | 고정 | 공허 HELD → PARTIAL(수리) | ∀∅ |
| NEG_CONTROL ON쪽 절 | 고정 | ON 누수 2/128 → NOT_BUILT(오명명) | idx5 0.0156 > 0.01 |
| 하드캡 44,300↔44,900 / 36,000↔36,600 | 두 등록값 | 없음 | 여유 3,884/3,088 ≥ 358 |
| 하측 중앙값 lead 비율 | 0.33–1.0 | 라벨 불변, 허용 서술 거짓 | (48,512) ≈0.75ŝ+15 |
| S1 "==" vs ">" | 두 정의 | E-α 불변, Job k만 부풂 | b 종료 = bs < running_bs |
| F-8 시드 공유 | 셀 id | 없음 | radix off, boot 재기동 |

## 필수 수리 (rev4)
**死因 수리 D-7**: E-α 전용 등록 — O_EA(b,c) = (c+71 − c_in) + L_lag + N_co,max + 16 + 128 + ŝ + 64 (N_co,max = 최악 코너 W_co iteration 수, U-2 프로브가 갱신); E-α 취소 조건 = 클라이언트가 co-load K개 응답을 모두 받은 뒤 하측 중앙값 요청이 ≥ 16+128+8 토큰을 더 받은 시점. 가능성 자기검사: c=8192, b=8에서 c_in+O ≈ 10.5k < 16,384, KV ≪ 2.72M; 클라이언트가 아는 값만, 새 노브 결합 없음; N_co,max 과소 시 `BS_UNREALIZED`/`WINDOW_SHORT`로 fail-closed. `n2_reach_tests`에 "O를 W 기준으로 되돌린 변이본은 E-α에서 `BS_UNREALIZED`" 추가.
- **R3-1** §8-6 정정: CTX_MISMATCH 산술·"예보 범위에서 사실상 도달 불가(보간 구간 안)", BS_UNREALIZED 첫 시나리오 → `ABORT_TRACE`/`COLOAD_PAYLOAD_INVALID`, COLOAD_JOINED 경로 "filter 실패, `finished_output` 가드로 completion_tokens 1 유지" 명시, "도달 불가는 OUT_OF_DOMAIN 하나뿐" 철회, D_UNREALIZED 행에 NOT_BUILT 귀결 병기.
- **R3-2** §18-2 U-1 행·§14-3의 rev2 수치를 §14-2 값으로.
- **R3-3** NEG_CONTROL: OFF 부분모집단 ∅이면 `UNRESOLVED`; ON쪽 절 삭제(양성 대조 D 라벨 채널로) 또는 원인 부류(OFF/ON) 태그; E-α 인용 시 NEG_CONTROL 병기를 §16-2 필수 병기에 추가.
- **R3-4** CTX_MISMATCH "10%" 기준량 명시(권고 0.1 × c).
- **R3-5** S1: JOINED ⇔ bs > running_bs; bs < running_bs는 별도 사유(b 종료)로 `BS_UNREALIZED`, JOINED로 세지 않음.
- **R3-6** co span 식별 규칙(start가 트리거 mark 뒤 ∧ extend_tokens = n_new × 8192), co-load 시드 명시.
- **R3-7** §8-3 허용오차 서술을 span 구성 의존으로, 또는 X를 최대 산포 기준(예: 1.5ŝ + 96)으로. "c=512에서 c_in ≥ 416" 정정.
- **R3-8** §18-2에 프로브 뒤 예산·하드캡 재승인, 프로브 갱신 X·O·트리거에 대한 `n2_reach_tests` 재실행·§8-6 재대조(봉인 전) 추가.
- **R3-9** §6-2 `WINDOW_SHORT` 문장을 §9-1 참조로.

## 신규 caveat (인용 금지 문장, 문자 승계)
- **N2C-22** E-α 라벨(ρ_off 셀 라벨·`DS3_*`)은 `NEG_CONTROL_*` 라벨과 독립으로 출력된다 — `NEG_CONTROL_FAILED`/`UNRESOLVED`인 Job의 ρ_off를 'sticky OFF 운영점의 동거'로 인용하지 않으며, 인용 시 같은 짝 OFF E-α A1·A2의 idx5 비율을 병기한다.
- **N2C-23** ctx 정합 가드 `CTX_MISMATCH`는 등록 예보 범위에서 사실상 발화하지 않는다(ctx_mid(C)는 설계상 [A1, A2] 보간 구간 안에 놓인다) — '모든 짝이 ctx 정합 검사를 통과했다'를 정합의 독립 증거로 쓰지 않는다.
- **N2C-24**(R3-3에서 ON쪽 절을 유지할 경우에만) `NEG_CONTROL_FAILED`는 OFF 단독 창의 idx5 이탈뿐 아니라 ON 단독 창의 idx5 누수(sticky 유지 실패)로도 나온다 — 원인 부류를 병기하지 않고 'sticky OFF 단독이 108이 아니었다'로 쓰지 않는다.
- **N2C-25**(R3-5 미반영 시에만) `COLOAD_JOINED_k_of_n`의 k는 서명 S1·S2 실패 수이며 b개 자신의 종료가 섞일 수 있다 — k를 '합류 사례 수'로 인용하지 않는다(서명 원값 병기).
- **N2C-26** §8-3의 산포 허용 '1.5 × 예보 + 30'은 하측 중앙값 lead = 산포/2 가정의 산술이며 span 구성에 따라 약 0.75 × 예보 + 15까지 줄어든다 — `CTX_UNREALIZED`를 '산포가 예보의 1.5배를 넘었다'로 쓰지 않는다.
- **N2C-27** 같은 round 안 D 간 프롬프트는 같고 round 간 다르므로 프롬프트 효과는 round 효과와 앨리어스된다 — 점의 복제 간 산포(`residual_p95_ms`)를 기판 잡음만으로 해석하지 않는다.

## 반증 실패(공정 기록)
D-2 창 산술(동치)·트리거 조기 발화 불가, D-3 기록기·정합 비우회, D-4 (ii) 비항등(`finished_output` 가드), D-5, D-6, F-1 순열 균형(OBS 2/2), F-5 범위식, F-7 시그니처, N2C-1…21 문자 승계(바이트 일치), 예산 산술(±5 s), F-8 캐시 교락 부재. 실행 전 `CONFIRMED` 아님.

## 자기 적용
2차 판정 D-2 처방(트리거만 교체)을 등록 상수로 받지 않고 O까지 재계산해 처방 누락(D-7, 게이트 #113)을 찾음. D-7 처방은 클라이언트가 아는 값만, 컨텍스트·KV 한도 안, fail-closed. R3-3 ON쪽 절 삭제는 양성 대조를 잃지 않음(`D_UNREALIZED`/`D_REALIZATION_FAILED`가 담당). CTX_MISMATCH 반전은 ON/OFF 차이 예보 근거가 없어 성립시키지 않음(caveat).

## 근거 파일
rev3 `PREREG_N2_3A_2026-10-07.md`, rev2 scratchpad 백업, `VERDICT_n2_3a_rules_2026-10-07.md`, `VERDICT_n2_3a_rev2_delta_2026-10-07.md`, `ENGINE_CHECK_coload_2026-10-07.md`, `sglang_engine_dev/python/sglang/srt/managers/scheduler_output_processor_mixin.py`(decode 374행 이하, `finished_output` 가드 988-992), `benchmarks/pdmux_eval/analyze.py:115`.
