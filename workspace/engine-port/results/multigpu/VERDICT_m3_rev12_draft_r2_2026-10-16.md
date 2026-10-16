# 판정서: M3 PREREG rev12 r1+ 규칙층 감사 2회차(마지막) (2026-10-16)

> claims-auditor(읽기 전용) 최종 보고를 main 세션이 저장(2026-10-16). 저장소 변경 0, 계산은 scratchpad `…/scratchpad/r2/`(`env.py`, `bud/`, `sim2x.py`). 감사 상한 2회 중 2회차.
> 대상: `PREREG_MULTIGPU_rev12_DRAFT_2026-10-15.md` HEAD(`3f48a29`, r1+, 318줄). 비교: r1(`8e121cd`), r0(scratchpad `rev12_r0.md`, sha256 `29dd7d2f…`, VR1 머리 값과 일치).
> 원자료: VR1과 같음(parquet `b34529df…`, 토크나이저 `3277c00f…` 재확인). VR1 (I, O) 배열 `s/IO.npy`(5,906건), 가산식 w = I/(0.690·8192) + O/(2.99·512). 예산: main 세션 `m3budget/run3.py`와 하네스 사본.
> 범위: VR1 §8에 따라 R-1–R-5 diff와 사용자 결정(DECISIONS §4·§5) 반영분만. `0eca947` 시험 19건 로컬 CPU 재실행 전부 통과.

## 1. 총평

| 대상 | 등급 | 死因 |
|---|---|---|
| **rev12 r1+** | **NO-GO → 사용자 결정 필요** | **1건, D-1′(N2)**: R-1이 서버 측 1%와 클라이언트 측 1%를 **따로** 센다 → 셀당 전체 오류 허용치가 rev11의 1%에서 2%로. 두 arm 반대 방향 최악 G 편향 상한 **+4.43%** > 3% 크기 규칙(rev11 합산 규칙은 +2.21%) |

D-1 원래 경로(상한 없는 서버 측 오류로 X λ\*_thr 부풀림)는 **닫혔다**. 남은 것은 R-1 문턱 **구조**(분류별 각각 1%)에서 생긴 잔여 반전. 상한 2회째라 처방은 더 쓰지 않고 §9로 넘긴다. 사용자가 §9 (a)를 택하면 이 판정서 기준 나머지는 死因 0이고, `GO-with-caveats` 요건 = 필수 수리 F-1–F-8(死因 아님) + §6 caveat. 단 (a)를 반영한 바이트는 감사받지 않은 diff → 하네스층 감사의 변이 시험(§9)으로 확인(3회차 규칙층 감사 아님).

## 2. 반전 시험 표

achieved = n_ok/span(`mg_label.py:426-429`). 서버가 빨리 돌려보내는 오류는 그 요청의 일을 쓰지 않고, 늦은 오류는 일이 이미 쓰여 span이 그대로. 오류 요청은 가장 긴/가장 짧은 문서부터 두 극단(`r2/env.py`).

| 자유 표면 | 민 범위 | 판정 변화 | 근거 |
|---|---|---|---|
| **R-1 문턱 구조: `CLIENT_ERROR` > 1%, `SERVER_ERROR_HI` > 1% 각각** | arm·셀마다 서버 1% + 클라이언트 1%. agnostic = 가장 긴 문서부터 빨리 실패, fused_sym = 늦게 실패 | **반전** | agnostic 단독 1% → +1.19%, 2% → +2.34%. fused_sym 늦은 오류 2% → −2.00%. **G 비 최대 +4.43%**. agnostic 서버 1%만 + fused_sym 서버·클라이언트 각 1% 늦은 오류도 +3.25%. rev11(전체 > 1% 무효)은 +2.21%. 참 G_sym(N,X) = 0.94(크기 라벨 있음)가 r1에서 0.9816(없음), rev11에서는 0.9608(유지). §19-1b 근거 문장 "1% 이하 오류의 부풀림은 3% 크기 규칙보다 작다"는 G 비에서 거짓. **死因 D-1′** |
| R-1 서버 측만(VR1 처방 형태) | 서버 ≤ 1%, 클라이언트 0 | 반전 없음 | 한 arm +1.19%, 두 arm +2.21% < 3% |
| LO 셀 서버 측 오류 = 기술값(상한 없음) | 0 → 수십 % | 반전 수치 못 만듦 → caveat | LO는 브래킷 인증 전용, λ\*는 HI achieved, `HI_SATURATED`(g ≥ 0.10) 별도 검사. LO 오류가 브래킷을 거짓 통과시켜도 HI achieved가 용량의 1.67배를 넘어야 λ\* 왜곡 — 포화 처리율로 불가 → MG12-14 |
| `SERVER_ERROR_HI`의 A 앵커·J1 기준 셀 적용(§16-1 문언은 `HARNESS_LIMITED`·`CLIENT_ERROR`만) | "A 앵커 미적용" 해석 | 반전 수치 못 만듦 → F-5 | 가려지려면 두 우연 동시 필요. 앵커 fused_sym 늦은 오류는 λ\*(fused_sym, A)를 내려 "J1과 3% 넘게 다름" 검사가 먼저 발화(fail-closed) |
| P10 BROKEN의 X: 비앵커(raw, 등록 [제안]) vs A식 동일 shape 보정 | 두 해석 | 반전 없음(등록 쪽이 보수적 끝) | §17-1·§18 M-4 동일 shape 보정은 `TREND_JOB_SENSITIVE`로 BROKEN을 줄이는 방향만. raw가 BROKEN 상한 → fail-open 0. 대가 검정력(MG12-3) |
| 최종 POST·gate (iii) 경로 | 등록 순서(최종 POST → preflight → 재봉인) | 판정 변화 없음, **등록 순서로는 gate가 참이 될 수 없음** → F-1 | `tests/test_mg_harness.py:1527` `assertEqual(C.PREREG_REV, 11)`. 재봉인(`PREREG_REV = 12`)과 §19-12 `mg_label`·`mg_plan` 변이 시험이 `tests/`(gate 경로)를 반드시 바꿔 (iii) 거짓 → ABORT. fail-closed |
| context 20480·fit 경계 | 등록 근거 vs `fit_problems`(입력+출력 ≤ ctx−2) | 반전 없음 | 실제 하한 X ≥ 16,746, floor ≥ 16,394(등록문 16,745·16,393은 1 작음, VR1 승계 오차). 20480은 둘 다 덮음 → F-6 |
| `mamba_ssm_dtype` 규칙 | `-Base` config 키 유무 | 반전 없음 | 엔진 해석: 기본 float32 → config → env(`configs/mamba_utils.py:31-64`), 명시 인자는 env로(`server_args.py:3403`). 규칙 = M2g 실효값(코드 유도). 단 **boot 대조는 항등식**: M2g `server_info.json`은 `mamba_ssm_dtype: None`(인자를 보여 줄 뿐) → F-4 |
| 토큰화 재현·floor 문서 선택 | `tokenizers` 버전, 후보 534건 중 1개 | M3 판정 무관 | floor는 M3 기술값·M4 전용(§19-4). Job 입력은 파일 sha로 fail-closed. `tokenizers` 버전(로컬 0.22.1) 미등록이라 재생성 대조는 통계 수준뿐 → F-7 |
| 예산: X 참 용량/prior_X = m(MG12-1 허용 [0.35, 2.0]) | m ∈ {1.0, 0.9, 0.8, 0.7, 0.5} | 라벨 반전 없음(fail-closed 절단) | S2 J2: 15.57 → 15.95 → **TP2 boot2 절단**(m = 0.8) → 절단(0.7). m = 0.5면 S1도 J2·J4 TP 절단. m ≥ 1.42면 확장 → S3형(TP 절단). DP는 전 경우 완료 → MG12-13 |
| U14 X 앵커 rung(1.6×prior_X(1), 확장 없음) | m ∈ {1.0, 1.3, 1.5, 1.7, 2.0}(`sim2x.py`, 30회) | 라벨 반전 없음(UNTESTED, fail-closed) | J1 boot1 X가 확장으로 가는 비율: m = 1.3에서 15/30, m ≥ 1.5에서 29–30/30 → 기준 셀 포화 미달 → 전 J_N `JOB_EFFECT_UNTESTED(J_N, X)` → MG12-12 |

반전 계산의 자유 표면 통제: 오류 선택은 최장·최단·늦은 오류 세 극단만, 예산은 등록 S1·S2에 m만 곱함, 확장 판정은 VR1 `sim2.py` 그대로. 모두 **모형 예보**이며 측정이 아니다.

## 3. Q1–Q7 (§19-14)

- **Q1 (R-1) REFUTED(부분), 死因 D-1′.** (i) D-1 원래 경로는 닫힘(서버 측 상한 1%, 짝 대칭 무효, 확장 셀·전 shape 적용). (ii) C5 분할 + R-1 별도 문턱이 겹쳐 rev11 "모든 오류 > 1% 무효"보다 2배 느슨(§2 첫 행). (iii) LO 기술값은 반전 못 만들어 caveat. (iv) D-7 관계 서술은 DECISIONS §4와 문자 그대로 맞음. (v) 부수 불일치: §19-1b는 "수락 뒤 timeout"을 서버 측으로 적었으나 `0eca947`에서 `sock_read` timeout은 `CLIENT_EXCEPTION`(클라이언트 측; `classify`에서 예외가 모든 서버 신호보다 우선) → F-2. (a)면 판정 무관.
- **Q2 (R-2) CONFIRMED(코드 범위).** `StreamState`는 SSE `"error"`를 토큰으로 세지 않고 `SSE_ERROR`·`SERVER_ABORT`·`OUTPUT_SHORT`·`OUTPUT_LONG`(`ignore_eos`이고 관측 ≠ 요청)을 server로 분류. 모든 실패에서 `error` 비어 있지 않음 → 현 `mg_label.is_error`(`:60`)가 achieved·오류 수에 반영 → 재봉인 전 rev11 라벨러로도 fail-closed(전부 `CLIENT_ERROR` 쪽). `error_kind`·`error_side`는 `requests.jsonl`을 거쳐 `mg_label`까지 전달, 분류 소비는 재봉인 몫(§19-12). M3 경로 `mg_loadgen` → `trace_loadgen.replay`라 자동 적용, `sock_read=600` 유지. `mg_boot` 구멍은 등록(§19-12 `fit_problems`)으로 닫히나 **두 곳 미등록** → F-3: (1) `replay`가 `"input_ids": [1] * L` 하드코딩 — X 실제 토큰 송신이 `replay` 밖에서 구현되면 `StreamState` 우회 가능 (2) floor 블록 요청은 trace 밖이라 `fit_problems` 대상 밖, 클라이언트 미정. 시험 19건 재실행 통과.
- **Q3 (R-3·R-4) PLAUSIBLE.** 20480, U2(b) 실행 불가, G5(c)(ii) X 항(16,384 + 616)은 VR1 §4와 맞음. 하한 수치 1씩 작음(F-6). dtype 규칙은 M2g 실효값과 코드로 일치하나 "boot마다 `server_info` 실효 dtype 대조"는 인자 대 인자 항등식(교훈 9) → F-4.
- **Q4 (R-5) PLAUSIBLE, F-1 필수.** 대조 경로 7개·근거 맞음. preflight는 전체 스위트에서 L2 밖 신규 실패를 보므로 `test_mg_harness`가 POST에서 돔 → `scripts/multigpu` 포함 정당, `results/multigpu` 제외도 정당(§19-11 금지 병기). C7 정정 반영. 결함: "최종 POST → preflight → 재봉인" 순서는 실행 불가(§2), fail-closed라 死因 아님. 표기: "최종 POST = gate 경로 하네스 커밋, 후보 `0eca947`"은 DECISIONS §5 U13 문언에 없는 main 세션 구체화 — [결정]이 아니라 [제안]. §19-9 분석기 위치는 gate가 제출 시점 대조라 fetch 전·제출 후 커밋이면 충돌 없음, 제출 전에 gate 경로에 넣으면 (iii) 거짓(fail-closed).
- **Q5 (R-6·R-8) PLAUSIBLE.** floor_N 두 arm 배치로 MG12-6 비대칭 실제 소멸(블록 20, 예산 재현). 순열 prefix 중첩의 N 간 양의 상관은 N 차 분산을 줄이고, N 점을 독립으로 다루는 bootstrap에서는 보수적. 새 fail-open 없음, MG12-10 승계.
- **Q6 (범위) 반증 실패.** r0 → r1 → r1+ diff(`git diff --no-index`)에서 색인 밖 판정 의미 변경 없음. 셀 536 → 560·예산 표 변경은 U14·R-6.
- **Q7.** (a) R-8 PLAUSIBLE: Job 입력은 파일 sha 대조, 불일치·부재 시 `ABORT_DATASET`, 송신 토큰 수 단언 → fail-closed. 재생성 대조는 유지 수·통계뿐, `tokenizers` 버전·문서별 길이 sha 미등록(F-7). X(실제)와 A·B(합성) 혼재는 shape 안 arm 비라 G 오염 없음. A·B T가 실제 문서 floor에서 오는 것은 M3 판정 미사용(기술값). floor 문서를 유지분 밖(> 16,384)에서 고르는 것은 16k 점에 필요, r0과 표기만 다름. 선택 절차(후보 정렬·RNG 호출) 미정(F-7). `-Base` 토크나이저 상이 시 봉인 전 사용자 회부는 fail-closed. (b) REFUTED(실행 가능성) → F-1. `test_mg_symlink_layout`의 `MANDATORY_REG` 검사(`:102-104`)는 자기 이름의 등록 여부만 봐 중간 커밋에서 안 깨짐; 깨지는 것은 `test_mg_harness.py:1527`(`PREREG_REV`)과 재봉인 변이 시험. (c) U14 규칙은 §15-3·§16-1과 같은 형식(두 조건, P9-b 이중 보고, 앵커 seed 공동 재표집). 비앵커 [제안]은 보수적 끝. MG12-4 원문 "X의 Job 효과는 검출되지 않았다"는 r1+에서 부분적으로 거짓 → 주석만으로 부족, MG12-4′로 대체(§6). 기준 셀 포화 미달 → `JOB_EFFECT_UNTESTED`(H-3 X판)는 "§16-1과 같음"으로만 적힘 → F-5. +0.65 h/Job 재현. S2 여유 0.43 h는 m ≤ 0.9에서 소진, m = 0.8에서 TP 절단(fail-closed, MG12-13). (d) PLAUSIBLE: `MANDATORY_REG` 재봉인 사유, `mg_loadgen.py:4` docstring 정정, `mg_boot` 노브 3개, preflight 목록 `test_loadgen_outcome` 모두 반영. 누락은 F-3·F-1.

## 4. 필수 수리 (死因 아님 — 봉인 전 필수)

- **F-1 (gate 순서).** 최종 POST = M3 제출 전 7개 경로 중 하나라도 바꾼 마지막 커밋(실제로는 재봉인·하네스층 감사 수리가 끝난 커밋), preflight는 그 뒤 1회. U13 문언("2회차 감사 뒤")과 모순 없음. 근거 `tests/test_mg_harness.py:1527`. §19-6 순서 문장과 "[결정 U13]" 표기를 [제안]으로.
- **F-2.** §19-1b "서버 측 오류의 범위"의 "수락 뒤 timeout"을 `0eca947` 분류(`CLIENT_EXCEPTION`)와 일치. (a)면 문장만.
- **F-3.** X `input_ids` 송신은 `trace_loadgen.replay`(`StreamState`) 경로 안에서. 그 변경이 `benchmarks/pdmux_eval`에 들어가면 F-1 순서. floor 블록 요청도 같은 분류기를 거치고 `fit_problems` 대상에. 변이 "X 경로에서 error 청크를 토큰으로 셈"·"floor 요청 fit 검사 누락" KILLED.
- **F-4.** dtype boot 대조를 "인자 기록(항등식)"으로 재서술. 실효값 대조는 부팅 로그의 mamba 상태 메모리·슬롯을 M2g 같은 (config, arm)과 비교(`KV_POOL_DRIFT` 경로). 로그에 값이 없으면 `DTYPE_UNVERIFIED` 공시.
- **F-5.** `SERVER_ERROR_HI`를 shape A 앵커 셀·J1 기준 셀(A·X)에도 적용해 `JOB_EFFECT_UNTESTED`로 만든다고 명문화. X 기준 셀(J1 boot1 X, 확장 전 rung)이 포화 미달이면 전 J_N `JOB_EFFECT_UNTESTED(J_N, X)`(H-3 X판 변이 KILLED).
- **F-6.** 하한 ctx ≥ 16,746(X), ≥ 16,394(floor).
- **F-7.** 사전 토큰화에 `tokenizers` 버전과 문서별 (input, output) 길이 배열 sha256 등록. floor 문서 선택 절차(후보 = 원 parquet 행 중 토큰 수 ≥ 16,384, 행 번호 오름차순, RNG 종류·호출) 문자 그대로 등록.
- **F-8.** MG12-4 → MG12-4′(§6).

## 5. 승계 caveat

VR1 MG12-1–MG12-10 문자 그대로 유지(MG12-6 비적용 표시 유지, MG12-4는 MG12-4′로 대체). rev11 §7·§15–§18, MGP-\*, MGH-\*, SP S-1–S-8·K1–K9, C-1–C-6·CL-1–CL-3 승계.

## 6. 신규 caveat (인용 금지 문장, 문자 그대로)

- **MG12-4′** "shape X의 Job 효과는 J\_N 앵커(1.6×prior\_X(1) rung, 확장 없음)로만 검사되었다. `JOB_EFFECT_UNTESTED(J_N, X)`인 Job의 X 점과 P10 BROKEN 판정의 X raw 라벨은 Job 효과 검사 밖의 값이다."
- **MG12-11** ((b)를 택할 때만) "λ\*\_thr 셀 유효성은 서버 측과 클라이언트 측 오류를 각각 1%까지 허용한다. 모형 예보로 두 arm 반대 방향 최악 G 편향은 ±4.4%이며 3% 크기 규칙보다 크다. X의 크기 라벨은 이 범위 안에서 반전될 수 있다."
- **MG12-12** "U14 X 앵커는 참 λ\*/prior\_X가 약 1.3 이상이면(모형 예보: 1.3에서 절반, 1.5 이상에서 거의 전부) 기준 셀 포화 미달로 `JOB_EFFECT_UNTESTED`가 된다. X Job 효과 미검사는 Job 효과 부재가 아니다."
- **MG12-13** "TP 구간 완료 예보(S2 여유 0.43 h)는 X 참 용량이 prior\_X의 약 0.9배 이상이고 1.42배 이하일 때만 성립한다(모형 예보). 그 밖에서는 TP가 등록 순서대로 잘리고 DP는 완료된다. TP 결과 부재는 TP 효과 부재가 아니다."
- **MG12-14** "LO 셀의 서버 측 오류는 상한 없는 기술값이다. LO 브래킷 인증은 오류를 제외한 요청 위에서 이루어졌다."
- **MG12-15** "`mamba_ssm_dtype`의 boot 기록은 서버 인자의 기록이며 실효 dtype의 측정이 아니다(F-4 대조가 있을 때만 실효값 확인)."

## 7. 허용·금지 문안

허용(VR1 §7 승계 + 1): "rev12 M3는 이 Job에서 shape A·B·X(arXiv-Sum, 입력 ≤ 16k, 출력 p50 195)의 λ\*\_thr 기반 G\_sym(N)을 잰다. 이는 같은 스케줄 노브에서 PD-mux 기능 전체의 효과이고 multi-GPU 1차·단일 GPU 2차 가족이다." · "공용 loadgen(`0eca947`)은 SSE error 청크와 `ignore_eos` 출력 길이 불일치를 서버 측 오류로 기록한다(CPU 시험 19건, 서빙 정확성 주장 아님)."

금지(§19-11 추가): "R-1/`SERVER_ERROR_HI`가 오류에 의한 G 편향을 3% 아래로 묶는다"((a) 반영과 하네스층 변이 시험 전) · "`0eca947`이 서빙에서 조용한 실패를 없앴다"(GPU 실측 0) · "boot 대조로 mamba dtype이 M2g와 같음을 확인했다"(F-4 전) · "preflight PASS가 rev12 재봉인 하네스·`tests/`를 검증했다"(F-1 전) · "U14로 X의 Job 효과가 없음을 보였다" · 이번 반전 시험 예보 수치(+4.43%, +3.25%, +2.21%, m 임계 0.8·0.9·1.3·1.5, 확장 비율)를 측정값으로 인용.

## 8. 요약

r1+는 D-1 원래 경로를 닫았다. R-2 코드(`0eca947`)는 코드 범위 CONFIRMED. R-3·R-4·R-6·R-8·U14는 반증 실패(fail-closed). R-5는 경로 집합은 맞으나 등록 순서가 실행 불가(F-1, fail-closed). 남은 死因 1건 D-1′: R-1이 C5 분할 위에 별도 1% 문턱을 둬 셀당 오류 허용치 2% → 두 arm 최악 G 편향 +4.43% > 3%(rev11 +2.21%). 감사 상한 도달 → 사용자 결정.

## 9. 사용자 결정 필요

- **무엇**: M3 λ\*_thr HI 셀(확장 셀·앵커·J1 기준 셀 포함)의 오류 무효 문턱을 서버·클라이언트 **각각** 1%로 둘지, **합산** 1%로 둘지.
- **왜**: 각각이면 셀당 오류 2%까지 허용, 두 arm 반대 방향 최악 G 편향 +4.43%(모형 예보) → 3% 크기 규칙 초과, 크기 라벨 반전(참 0.94 → 0.982). 합산은 rev11과 같은 +2.21%. D-7(λ\*_SLO·attainment)과는 어느 쪽도 충돌 없음.
- **선택지**: **(a) 권고** — λ\*_thr 셀 유효성만 (서버 + 클라이언트) 오류/셀 요청 수 > 1%면 짝 무효, 라벨은 우세 분류(`SERVER_ERROR_HI` 또는 `CLIENT_ERROR`)로 붙이고 두 건수 기록. rev11과 같은 상한, 새 상수 없음. 하네스 변이 "서버 0.9% + 클라이언트 0.9%인데 라벨 그대로" KILLED, 무변이 대조군 통과. 이 경우 나머지는 이 판정서 기준 `GO-with-caveats`(F-1–F-8 반영, §6에서 MG12-11 제외). / (b) 현행 유지 + MG12-11 승계로 X 크기 라벨 인용 범위 축소. / (c) 분류별 각 0.5%(합산 상한 1%와 같으나 두 분류가 고르게 나뉘지 않으면 fail-closed 비용 증가).
