# 판정서: N2(3-a) 사전등록 rev11 델타 감사 (claims-auditor, 2026-10-08)

> 감사자는 읽기 전용이라 메인 세션이 감사 보고 본문을 그대로 저장했다(2026-10-08). GPU 0, 새 성능 판정 0건, Claim 등급 변경 0건.
> **대상**: HEAD `d06da7d`(rev11 체크포인트)의 n2_static_grid 트리. 감사 중 트리는 깨끗했다. 이후 커밋 `2196642`·`7a7fa10`은 이 디렉터리에 `NOTICE_substrate_power_sampler_2026-10-08.md`만 더했다. 파일 sha 앞자리: `n2_probe_reduce.py` `fa9530f1…`, `n2_feasibility.py` `4683ae1b…`, `PREREG_N2_3A_2026-10-07.md` `7c1e74c3…`, `n2_mutants.py` `cb769320…`, `test_n2_probe_reduce.py` `ddbb41d1…`, `test_n2_feasibility.py` `6fe206eb…`. 대조본은 `368116a`(rev10).
> **수행 내용**
> - 등록 변이 115종을 순차·단독으로 재현했다(N3 3회차 원자료, 무변이 대조 포함, 약 1시간 50분).
> - 감사 변이 20종(Z01–Z20)을 같은 러너로 돌렸다. N3 경로는 건드리지 않으므로 N3 대조는 skip했다.
> - 합성 원자료 시나리오를 축약기 → 계산기 CLI(재축약·매니페스트 포함) 실물 경로로 끝까지 돌렸다.
> - rev10·rev11 계산기의 출력을 비교했다.
> - 시험 101개가 통과했다(skip 1 = N3. N3 대조는 변이 러너 CTRL에서 통과).
> - 엔진 텔레메트리 기록기와 `trace_loadgen` 예외 경로를 읽었다.
> **저장소 영향**: 저장소 파일 변경 0. 단 감사 초반, 바이트코드 끔 없이 계산기를 import하는 실수로 `n2_static_grid/__pycache__/n2_feasibility.cpython-313.pyc`(15:58)가 생겼고, 감사자가 지워 원상 복구했다.
> **스크래치** `scratchpad/a11/`: `reg.txt`, `aud.txt`, `zmut.py`, `scen.py`, `coload.py`, `misc.py`, `m02run.py`.

## 등급: `GO-with-caveats` — 死因 0. 단 이 판을 그대로 동결하면 안 된다

- 반전 시험에서 등록 판정(`probe_status`, `seal_allowed`, `reprobe_allowed`, K)이 실제로 뒤집히는 경우를 4개 만들었다(아래 표).
- 넷 모두 다음 둘 중 하나를 요구한다.
  - (1) 등록이 금지했으나 코드가 검출하지 못하는 하네스 거동, 또는 등록 CLI 밖의 실행 방식.
  - (2) 설계 예보 밖의 κ. 예: 검열 셀이 존재하는 경우, κ 2.7.
- 설계 예보(κ_DN 1.3 균질, W_co 247–253 ≫ 128)에서는 하나도 발화하지 않는다.
- rev10 선례와 같은 부류(시행 구멍)라 N2가 아니다.
- N1·N3·N4도 해당 없음. 프로브는 측정이 아니고 인용 금지이며, 운영점은 불변이다.
- ⚠️ 봉인·동결·프로브 실행 승인이 아니다. 아래 R11-1…R11-7은 축약기·계산기·마크 형식을 바꾸고, 하네스는 아직 없다.

## 1. R10-1…R10-5·H-1·H-2 반영

| 수리 | 반영 | 판정 |
|---|---|---|
| R10-1 가장 심각한 부류 | `_measure`가 모든 검사를 돌리고 최댓값을 취함. `delta_l`·`fix_s`는 추출 뒤에 평가. 중단 없는 `cell_end`/`coload_sent` 누락은 VD. 사유 표 있음. 끝 줄 잘림은 VI, 중간 손상은 VD | **반영** — rev10 S1–S3·S6 세탁이 닫힘(시험 `test_mark_loss_does_not_launder`, N02·N10 사살 재현). 단 수리가 새 표면 (a)(b)(c)를 만듦(§2) |
| R10-2 원자료 기준 완전성 | 고아 co span(시작 시각 기준), id 중복, d별 셀 수 14·7, `coload_sent` 호출 감쌈 | **반영**(S4·S5b·S7 거부). 단 사망 깃발이 프로브 전역이고(§2(c)), co-load 크기·(b,c)·호출↔셀 1:1은 미검사(§3) |
| R10-3 probe.json 결속 | `reproduce_probe` 바이트 대조, 불일치면 K 없음·재프로브 불가 | **사후 편집에 한해 반영**(tamper rc 2). R9-4 구멍은 기계적으로 닫히지 않음(§4) |
| R10-4 변이 고정 | Y01·Y02·Y12·X06·X10 사살, N01–N15 | **반영**(재현 일치). 감사 Z 20종 중 13종 생존(§5) |
| R10-5 문언·`censor_detail` | N2C-50 철회, N2C-52·54·55·56 문자 승계, 분해 기록 | **반영**(Z13 사살로 분해값 고정 확인) |
| H-1 마크 형식 사양 | 마크별 필드·시점·의무·사유 표 | **하네스 계약으로 불충분**(§6) |
| H-2 매니페스트 전 파일 대조, 같은 VI 사유 2회 정지 | 전 파일 대조 반영(N11·N12) | 정지 규칙의 "같은 사유" 동치가 정의되지 않음(§2(f)) |

## 2. 초안이 자기 공시한 대상 표면 (a)–(e) 시험

**(a) VI 중단 사유 → 세탁 경로인가: 그렇다(조건부).**
- 합성 결과(κ 1.3 정상 셀 20개 + d=44 검열 셀 1개[C = 100]):

| 같은 검열 셀의 중단 사유 | 상태 | `reprobe_allowed` |
|---|---|---|
| 중단 없음 | `KAPPA_CENSORED` | False |
| `cell_abort http_connection_error` 또는 `server_down` | `VALUE_INDEPENDENT` | **True** |
| `cell_timeout` | VD | False |

- 값 의존 사건이 VI 이름으로 기록될 수 있는 구체 경로가 있다. aiohttp의 읽기 시한 예외 `ServerTimeoutError`(설치 3.13.2에서 MRO 확인, venv 고정 3.13.3)는 `ServerConnectionError → ClientConnectionError`의 하위 클래스다.
- 기존 `trace_loadgen.py`는 `sock_read=600`이고 모든 예외를 `repr`로 삼킨다.
- 따라서 `except aiohttp.ClientConnectionError: reason = "http_connection_error"`라는 자연스러운 사상 하나로 시한(값 의존)이 VI가 된다.
- 등록에는 예외 → 사유 사상이 없다. 이 부분이 자유 표면이다.
- 예보(셀 수 초 ≪ 600 s, 검열 없음)에서는 미발화 → caveat + 하네스 계약 보강(H-4).

**(b) 중단 셀은 추출하지 않음 → 검열 증거가 숨는가: 그렇다.**
- 위 표 그대로다. `cell_abort`에는 시각 필드가 없어서, 중단이 co-load 완료 뒤였는지(검열 판정이 이미 확정됐는지) 축약기가 알 수 없다.
- 정직한 기반 장애(무작위)가 검열 셀과 겹치면 종결 상태(검열)가 재프로브로 바뀐다. 다음 시도에서 검열이 없으면 봉인된다 → 상한 꼬리를 걸러내는 약한 선택.
- 해결: R11-3.

**(c) boot 사망 정의 → 셀 수 부족을 VI로: 그렇다. 그리고 구현이 처방보다 넓다.**

| 시나리오(d=16 셀 1개 미시작) | 상태 | `reprobe_allowed` |
|---|---|---|
| 사망 없음 | VD | False |
| 끝 줄 잘림 | **VI** | **True** |
| `boot_abort server_down` | **VI** | **True** |
| 미등록 `boot_abort budget_stop` | VD | False |
| 두 boot: A는 셀 완결 뒤 끝 줄 잘림, B는 1셀 부족 | **VI** | **True** |
| 두 boot: A 정상, B는 1셀 부족 | VD | False |

- 사망 깃발 `dead`가 프로브 전역 변수다. 처방 R10-2는 "텔레메트리가 **셀 창 전에** 끝났으면"이었으나, 구현은 다른 boot의 죽음까지 부족을 VI로 바꾼다.
- 끝 줄 잘림은 하네스가 유발할 수 있다.
  - 엔진 텔레메트리는 128건 또는 1 s 주기로 flush하는 버퍼 기록이다(`src/multiplex/telemetry.py`).
  - 스케줄러는 종료 시 부모가 SIGKILL한다(같은 파일 docstring).
  - 그래서 활동 직후 바로 kill하면 줄 중간에서 잘릴 수 있다.
- 실측 반대 증거: N3 원자료 텔레메트리 92개 중 끝 줄 잘림은 0이다. 정상 종료는 잘리지 않는다 — 그러니 잘림은 비정상 kill의 지문이다.
- 의도적 정지(예산·시한)에는 등록 마크가 없다. 사양상 `boot_abort`는 "boot가 죽었을 때"만 쓴다. 그래서 의도적 정지가 기록되지 않은 부족으로 남고, 사망 깃발과 만나면 VI가 된다.
- 해결: R11-2, H-5.

**(d) 고아 span을 시작 시각으로 판정: 반전을 못 만들었다 → caveat.**
- 셀 창 안에서 시작한 생략 호출의 co span은 그 셀 추출에 들어간다. A2 부족 → VD로 가거나, co 요청이 `max_new_tokens = 1`이라 decode bs에 들어가지 않으므로 κ_i는 거의 그대로다.
- 단 셀 안 이중 co-load는 현재 아무 검사도 잡지 못한다. R11-1의 Σ n_new 검사가 닫는다.
- Z04(span이 두 셀에 걸림 허용)는 생존 — 시험이 없다.

**(e) 재축약이 기록된 경로를 그대로 엶: liveness 문제다. 선택 편향은 아니다.**
- 상대 경로로 축약한 probe.json을 다른 cwd에서 게이트에 넣으면: rc 2, K None, seal False, reprobe False.
- 사유 문자열이 "probe.json differs from the re-reduction"으로 나온다. `reduce_probe`가 파일 부재를 VI boot issue로 삼켜 다른 JSON을 내기 때문이다. 즉 경로 부재가 **변조와 같은 문구**로 보고된다(진단 오도).
- 정직한 절차(Workspace에서 프로브 → 로컬 깨끗한 worktree에서 게이트)는 경로를 맞추지 않으면 반드시 이 막다른 길로 간다.
- 수리: H-6 + R11-4의 진단 분리.

**(f) H-2 "같은 VI 사유 2회 연속 → 정지": 사유 동치가 정의되지 않았다.**
- 같은 원인(마크 파일 부재)으로 두 번 시도한 결과의 사유 문자열이 다르다.
  - `"raw file missing: … '/tmp/tmp4cjdd0nn/try1/marks_missing.txt'"`
  - `"… '/tmp/tmp5i58ql4f/try2/…'"`
- 셀 id·개수·줄 내용이 들어가는 사유도 같다.
- 그래서 정지 규칙이 문자 비교로는 발화하지 않는다. 판정 반전은 없다(값 무관 VI면 K 분포 불변). 단 재프로브 횟수가 무계가 된다 → R11-5.

## 3. 새로 찾은 시행 구멍 (초안 미공시)

**co-load 크기(K_probe = d44 8, d16 18)를 아무 코드도 검사하지 않는다.** 실측(`coload.py`, `post_probe_gate` 실물):
- κ 1.3: co-load 8 / 10 / 12 모두 K (22, 10).
- κ 2.4: 셋 다 K (32, 14).
- **κ 2.7**:
  - 등록 co-load (8, 18): d44 14셀·d16 7셀 전부 `KAPPA_CENSORED` → **봉인 불가, 재프로브 불가**.
  - (10, 22): 전부 REDUCED → **`OK`, K (36, 16), seal True, rc 0**.
- 프로브 자신의 측정 상한 κ\*를 하네스가 co-load로 올리면 검열 → 봉인이 된다. 등록 금지 사항인데 미검출이다.

**프로브 셀의 (b, c)도 검사하지 않는다.** 축약기는 d만 보고, 계산기는 격자 안의 아무 (b, c)나 받는다.
- d44 14셀 중 7셀을 b = 8(κ 2.0)로 바꿔도 `OK`, K (27, 12), 봉인 허용.
- 등록은 결속 셀(32, 8192)만 허용한다.

**부하 호출 ↔ 셀 1:1도 검사하지 않는다.**
- 호출 하나가 셀 21개를 감싸도 `OK`가 나온다. 이때 θ 표본은 p64 42 → 55, p92 56 → 63으로 늘어난다. 호출별 첫 span 제외가 1회로 줄기 때문이다.
- 균질 합성이라 K (22, 10)은 불변이다. 실측처럼 첫 span이 느리면 θ 중앙값이 내려간다.
- 부하 호출 id 중복은 dict 덮어쓰기로 조용히 처리된다(이번 합성에서는 `coload_sent` 감쌈 검사가 VD로 막았다).

## 4. M02 등가 재분류와 R10-3이 출처 구멍을 닫는가

**M02 "등가": 조건부로만 맞다. 근거 문장은 일반적으로 틀렸다.**
- 근거는 "재축약의 `reducer_sha256`은 실행 중인 축약기가 쓴다"이다. 그런데 `reproduce_probe`는 `sys.path.insert` 뒤 `import n2_probe_reduce`를 한다. 같은 프로세스에 다른 `n2_probe_reduce`가 이미 적재돼 있으면 그 모듈이 쓰인다.
- 실측(`m02run.py`): κ를 0.8배하는 외부 축약기를 먼저 import하고, 그 축약기로 probe.json을 만든 뒤 등록 계산기를 in-process 호출했다.

| 계산기 | rc | 봉인 | K | 사유 |
|---|---|---|---|---|
| 원본 | 2 | 불가 | — | "reducer sha256 … differs" |
| M02 변이 | **0** | **True** | **(26, 12)** | — |
| (참고) 등록 축약기 | — | — | (32, 14) | — |

- 등록 CLI 호출(새 프로세스 `python3 n2_feasibility.py`)에서는 등가가 성립한다. 그러니 sha 검사는 모듈 해석 경로를 지키는 유일한 방어선이다.
- 수리 R11-4.

**R10-3이 R9-4/rev10 출처 구멍을 닫는가: 부분적으로만.**
- 닫는 것: 축약 뒤 probe.json 편집(tamper rc 2).
- 단 Z08(바이트 대조를 `probe_status` 비교로 약화)이 **생존**했다. 값만 바꾸는 변조(상태는 OK 유지)는 시험이 고정하지 않는다.
- 닫지 못하는 것:
  - (i) 계산기 사후 변경. 자기검사이므로 항등식이다(교훈 9).
  - (ii) 축약 전 마크 텍스트 편집 — 예: 검열 셀에 `cell_abort … server_down` 추가 → 재축약이 그대로 재현된다.
  - (iii) in-process 외부 모듈.
- **clean worktree 규칙은 문언뿐이다.**
  - CLI는 `git rev-parse HEAD`와 `git status -- .`(n2_static_grid 디렉터리만)을 **기록만** 한다.
  - 매니페스트 `commit`은 비어 있지 않은지만 본다. 시험 픽스처의 `"0"*40`도 통과한다.
  - Z10(commit 검사 삭제)·Z17(git 기록 삭제) 둘 다 생존.
  - 다른 디렉터리의 하네스 파일이 dirty여도 기록되지 않는다.
- 방어선은 여전히 "감사가 동결 커밋으로 원자료에서 K를 재계산"하는 절차뿐이다.

## 5. 변이 시험

- **등록 115종 재현**: CONTROL-OK, 사살 106, 등가 생존 9(A09·A12·A16·M02·X11·X16·M05·B09·B11), 예상 밖 생존 0, TIMEOUT 0, BROKEN 0, `.pyc` 0. 초안과 일치한다(`reg.txt`).
- **감사 Z 20종**: 사살 7(Z03·Z09·Z13·Z14·Z19, 그리고 기록 확인 Z13), 생존 13.

| 변이 | 하중 | 의미 |
|---|---|---|
| Z01 `coload_payload_invalid` → VI, Z02 `stream_cancel_failed` → VI | **중** | 사유 표의 VD 항목 중 `cell_timeout`만 시험됨 — 값 의존 사유가 VI로 바뀌는 세탁이 미고정 |
| Z15 미등록 `boot_abort` → VI | **중** | 기본값 VD가 boot 단위에서 미고정 |
| Z06 초과 + 사망 → VI | **중** | `short and dead`의 `short`가 미고정 |
| Z07 계산기가 기록된 `expected_calls`로 재축약 | **중** | "재축약은 기본값으로만"(R10-3)이 미고정 — 13호출 프로브를 통과시킴 |
| Z08 재축약을 `probe_status`만 비교 | **중** | 값 변조 미고정 |
| Z10·Z17 commit/git | 중(R11-5 뒤 판정 요소) | clean worktree는 문언뿐 |
| Z04 span이 두 셀에 걸림 허용, Z05 `coload_sent`가 2개 이상 호출 안에 있어도 허용 | 소–중 | 창 겹침 미시험 |
| Z12 끝 3줄 잘림을 사망으로 | 소–중 | 중간 손상(VD)이 사망(VI)으로 세탁됨 |
| Z16 축약기 예외 → VI | 소–중 | 시험 b3에서 셀 부족 VD가 같이 걸려 분리 안 됨 |
| Z20 원자료 부재를 사망으로 안 셈 | 소 | 부재 → 부족이 VI인지 미시험 |
| Z11 `_measure` id 중복 issue 삭제 | 등가 후보 | boot_issues 중복이 상태를 VD로 유지 — 측정 기록만 달라짐 |
| Z18 `cell_end` 없는 중단 셀 창을 무한대로 | 소 | 의미 미시험 |

- 등가 9종 검토: A09·A12·A16·X11·X16·B09·B11·M05 근거 수긍(M05의 문언 축소도 정확).
- **M02는 "조건부 등가(새 프로세스 CLI)"로 재표기**하거나 R11-4로 조건을 없애야 한다.

## 6. 마크 형식 사양(H-1)이 하네스 구현의 충분한 계약인가: **불충분하다**

빠진 계약:
1. 예외 → 사유 사상과 그 우선순위(시한 먼저).
2. `cell_abort`의 시각.
3. 의도적 정지 마크.
4. 종료 순서(flush 대기).
5. 호출 ↔ 셀 1:1, 호출 id 유일성("권장"이 아니라 의무).
6. co-load 크기와 결속 셀 (b, c) — 축약기가 검사할 수 있게.
7. 시도별 고유 원자료 경로(재프로브가 앞 시도를 덮어쓰지 않게)와 게이트 위치에서의 경로 규약.
8. 재프로브 정지 규칙용 사유 코드.

참고: 시계에 대해 사양은 `time.monotonic()`이라 적었고 엔진은 `time.perf_counter()`를 쓴다(`telemetry.py:23`). 이 기판(Linux CPython)에서는 둘 다 `CLOCK_MONOTONIC`(확인함)이라 정합한다. 같은 컨테이너에서 실행하는 것을 전제로 명시하면 충분하다.

## 7. 프로브 전 수치 불변 — **참**

- rev10·rev11 계산기 기본 CLI 출력 **바이트 동일**(`cmp`).
- `k_sensitivity()` md5 `87d71553…0279` 동일.
- `choose_k` (17, 8), `limits_check` max_len 12,468.16 동일.

## 8. 기판 고지(`NOTICE_substrate_power_sampler_2026-10-08.md`)

- 이 고지는 rev11 감사가 caveat로 다루라고 요청했다.
- 샘플러는 `job_entry.sh`(Job 전용)에서 돌고 Workspace 프로브에는 없다. 그래서 프로브 보정 계수와 Job 측정 사이 기판 차이가 하나 늘어난다(N2C-38 계열).
- 같은 Job 안 on/off 대조가 없으므로 "영향 없음"은 주장할 수 없다. N2 판정은 Job 내부 대조라 공통 배경이지만, b = 1 짧은 step과의 상호작용은 미측정이다.
- 판정 입력이 아니므로 死因은 아니다 → caveat N2C-61 + 기판 기록 의무.

## 반전 시험 표

| 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 |
|---|---|---|---|
| (a) 예외 → VI 사유 사상(미등록) | 시한을 `http_connection_error`로 ↔ `cell_timeout`으로 | 검열 셀: `KAPPA_CENSORED`/재프로브 F → VI/**T** | `scen.py`, C = 100. aiohttp `ServerTimeoutError` ⊂ `ClientConnectionError`. 예보에선 미발화 |
| (b) 중단 셀 미추출 | VI 중단 ↔ 중단 없음 | 위와 같음 | 같음 |
| (c) boot 사망(전역 깃발·끝 줄 잘림) | 사망 없음 ↔ 잘림 ↔ VI `boot_abort` ↔ 다른 boot의 잘림 | 부족: VD/F → VI/**T** | `scen.py`. N3 텔레메트리 끝 줄 잘림 0/92 |
| (d) 고아 span 시작 시각 | 시작 ↔ 끝 | 반전 없음 | 논증 + Z04 생존 |
| (e) 기록 경로 재축약 | 같은 cwd ↔ 다른 cwd | rc 0 → rc 2, 재프로브 F (liveness, 수용 방향 아님) | `scen.py` |
| (f) 같은 VI 사유 동치 | 문자열 ↔ 원인 | 반전 없음, 정지 규칙 무력 | `misc.py` 경로 포함 문자열 |
| co-load 크기(등록 8/18, 미검출) | 8/18 ↔ 10/22 | κ 2.7: 검열/봉인 불가 → **OK·K (36, 16)·봉인 허용** | `coload.py`. κ 1.3·2.4에선 불변 |
| 프로브 셀 (b, c) | (32, 8192) ↔ 7셀 b = 8 | 등록 금지 셀 수용, `OK`·K (27, 12) | `misc.py` |
| 호출 ↔ 셀 1:1 | 셀마다 ↔ 1호출 | θ n 42 → 55 / 56 → 63, K 불변(균질) | `scen.py` |
| 축약기 모듈 해석(M02) | 새 프로세스 ↔ in-process 외부 모듈 | rc 2 → **rc 0·봉인 True·K (26, 12)**(등록 (32, 14)) | `m02run.py` |

계산 입력: 축약기·계산기 실물(HEAD), 시험 픽스처 `raw_cell`/`write_boot`/`full_cells`/`write_manifest`, K는 등록 CLI `main(["--probe" …])`에서 얻었다. κ 1.3/2.0/2.4/2.7과 C = 100은 시나리오이며 새 판정 상수는 0이다. rev10 판정서의 K 값은 승격하지 않고 rev11 코드로 다시 계산했다.

## 死因
없음. 가장 가까운 후보는 co-load 크기 미검출(검열 → 봉인 반전)과 M02 모듈 해석이다. 그러나 둘 다 등록 금지 하네스 거동이나 등록 밖 실행 방식, 그리고 예보 밖 κ를 요구하므로 N2가 아니다(rev10 선례).

## 필수 수리

**동결 전 — 축약기·계산기·문언(rev12). 하네스 판과 한 번에 델타 감사 가능.**
- **R11-1 프로브 셀 스키마 시행**: 셀마다 아래를 검사하고 위반은 VD로 둔다.
  - `coload_sent` 뒤 co span의 Σ n_new == `K_PROBE[d]`(8, 18). `K_PROBE`는 등록 상수로 두고 §21-19 단일 출처 검사에 묶는다.
  - (b, c) == (32, 8192).
  - 부하 호출 ↔ 셀 1:1(호출 id 유일, 중복 시 VD — 덮어쓰기 금지).
  - 시험: `coload.py` κ 2.7 co-load 10/22 → VD, b = 8 셀 → VD, 1호출 감쌈 → VD.
- **R11-2 사망 범위**: 사망은 boot 단위로 본다. 러너가 boot 시작 때 `boot_plan <d> <n>`을 기록하고, 계획 합이 `PROBE_CALLS`와 다르면 VD로 둔다. 부족은 **그 boot**가 사망했고 사망이 마지막 표시 셀 이후일 때만 VI, 나머지는 VD. 시험: `scen.py` (c′) → VD, Z06·Z15·Z20 사살.
- **R11-3 중단 셀 검열 증거**: 형식을 `cell_abort <cell_id> <reason> <t>`로 바꾼다. `cell_end`가 있고 co-load가 t 전에 완료(Σ n_new == `K_PROBE`)됐으면 추출을 돌려 C < 128일 때 `KAPPA_CENSORED`로 승격한다(VD 승격은 하지 않는다). 시험: 검열 셀 + VI 중단 → `KAPPA_CENSORED`.
- **R11-4 모듈 동일성과 진단**: `reproduce_probe`는 축약기를 `importlib` 파일 경로로 적재하거나 `os.path.samefile(red.__file__, red_path)`를 단언한다. "원자료 경로 부재"와 "바이트 불일치"는 다른 사유로 보고한다. M02 근거 문장을 정정하거나 in-process 시험으로 사살한다.
- **R11-5 사유 코드와 commit 결속**:
  - 축약기·계산기가 안정된 `reason_code`(등록 열거)를 낸다. 정지 규칙 "같은 사유" = 같은 `reason_code`.
  - CLI는 `git HEAD == manifest.commit`이고 동결 목록 파일에 미커밋 변경이 없는지 검사한다(범위 = 동결 목록 전체. 미추적 산출물 probe.json·K.json은 제외). 그래도 자기검사라는 점은 문언으로 남긴다.
  - Z10·Z17 사살.
- **R11-6 생존 변이 고정**: Z01·Z02·Z04·Z05·Z07·Z08·Z12·Z16을 사살하는 시험. Z11은 등가로 사유를 기록하거나 측정 단위 단언으로 사살. Z 목록을 `n2_mutants.py`에 등록.
- **R11-7 문언**: N2C-57…62 등재. §17-21에 새 공시 추가(co-load·(b,c)·1:1 미시행이 rev11에 있었음, 모듈 해석). §15 M02 행 정정. 샘플러 기판 상태(on/off)를 `substrate.json` 의무 항목으로 등록.

**하네스 구현 단계(H-1 계약 보강, 하네스 감사 대상)**
- **H-4 예외 → 사유 사상**:
  - `isinstance(exc, TimeoutError)`를 **먼저** 검사해 `cell_timeout`으로 둔다(aiohttp `ServerTimeoutError` 포함).
  - `http_connection_error`는 응답 바이트를 받기 전의 연결 수립 실패(`ClientConnectorError`/refused)에만 쓴다.
  - `server_down`은 서버 프로세스 종료 코드가 관측됐을 때만 쓴다.
  - 예외 클래스명은 별도 마크(`abort_detail`)로 남긴다.
- **H-5 종료 순서·의도적 정지**: 마지막 `cell_end` 뒤 서버를 내리기 전에 2 × `flush_interval_s`(엔진 기본 1.0 s) 동안 유휴를 기다린다. 예산·시한·조작자 정지는 `boot_abort runner_stop`(VD)로 기록한다.
- **H-6 원자료 경로**: 시도마다 고유한 경로를 쓴다(덮어쓰기 금지). 게이트 위치에서 같은 경로로 열리는 등록 루트를 두거나, probe.json 기준 상대 경로 규약 중 하나를 등록한다.
- **H-7**: 러너 정지 규칙은 `reason_code`로 판정한다.
- **H-3(기존)**: 하네스 판 델타 감사 뒤에 동결한다.

**프로브 뒤**
- Z 목록을 포함해 `n2_mutants.py`를 재실행한다.
- 델타 감사는 동결 커밋의 깨끗한 체크아웃에서 원자료로 K를 재계산하고, 모든 시도의 `reason_code`·중단 셀 수를 대조한다.

## 종합
- **하네스 구현 착수: 가능(권장).** R11-1…R11-3은 마크 형식과 함께 확정되므로, 하네스 구현과 같은 회차에 처리하는 것이 자연스럽다.
- **동결·프로브: 아직 불가.** 순서: R11-1…R11-7 + 하네스(H-4…H-7) → 델타 감사(코드 + 하네스) → 동결 매니페스트 커밋 → U-2 프로브 → 동결 체크아웃에서 `post_probe_gate` → 감사 K 재계산 → 재승인 → 봉인.

## 신규 caveat (인용 금지 문장, 문자 승계)
- **N2C-57** rev11 축약기는 프로브 셀의 co-load 크기(Σ n_new), 셀의 (b, c), 부하 호출과 셀의 1:1 대응을 검사하지 않았다. 합성 예에서 κ 2.7일 때 등록 co-load(d = 44 8, d = 16 18)의 프로브는 `KAPPA_CENSORED`(봉인 불가)였으나, co-load 10·22로 보낸 프로브는 `OK`·K (36, 16)·봉인 허용을 받았다. R11-1을 반영하기 전 판의 `probe_status = OK`를 "등록 프로브 조건에서 측정됐다"로 인용하지 않는다.
- **N2C-58** 중단된 셀(`cell_abort`)은 창 추출 없이 사유 부류로만 기록된다. 값 무관 사유로 중단된 셀 안에 C < 128(검열) 증거가 있어도 상태는 `VALUE_INDEPENDENT`가 되어 재프로브가 허용된다(합성: 검열 셀이 `http_connection_error` 또는 `server_down`으로 중단되면 VI·재프로브 허용, 중단이 없으면 `KAPPA_CENSORED`·재프로브 불가). 또한 aiohttp의 읽기 시한 예외는 연결 오류 클래스의 하위 클래스여서 하네스 사상에 따라 값 의존 시한이 값 무관 사유로 기록될 수 있다. 재프로브 뒤 봉인된 K를 "앞선 시도에 검열이 없었다"로, VI 중단을 "값 무관 사건이었다"로 인용하지 않는다. 모든 시도의 중단 셀 수·사유·예외 클래스를 병기한다.
- **N2C-59** rev11의 boot 사망 판정(텔레메트리 끝 줄 잘림 또는 값 무관 `boot_abort`)은 프로브 전체에 걸린 하나의 깃발이어서, 한 boot의 사망이 다른 boot의 셀 수 부족을 `VALUE_INDEPENDENT`로 바꾼다(합성: boot A 끝 줄 잘림 + boot B 1셀 부족 → VI·재프로브 허용, A가 정상이면 VD·재프로브 불가). 끝 줄 잘림은 정상 종료에서 관측되지 않았고(N3 0/92) 서버를 즉시 kill하면 생길 수 있다. 셀 수 부족에 따른 재프로브를 "기반 장애 때문"으로 인용하지 않는다.
- **N2C-60** 계산기의 재축약 대조(R10-3)와 동결 매니페스트 대조는 실행 중인 계산기·축약기가 스스로 수행하는 검사다. 사후에 고친 계산기, 같은 프로세스에 먼저 적재된 다른 축약기 모듈(합성: 등록 CLI 밖 in-process 호출에서 sha 검사를 지운 변이가 K (26, 12)를 봉인 허용했고 원본은 rc 2, 등록 축약기의 K는 (32, 14)), 축약 전에 편집된 마크 텍스트는 이 검사로 드러나지 않는다. `K_POST_PROBE.json`의 `seal_allowed`·`git`·sha 필드를 출처 진정성의 증거로 인용하지 않는다. 증거는 동결 커밋의 깨끗한 체크아웃에서 감사가 원자료로 다시 계산한 K뿐이다.
- **N2C-61** N2 Job의 측정값은 `job_entry.sh`의 NVML 전력·에너지 샘플러(2026-10-08 추가, 기본 ON)가 같은 컨테이너에서 도는 조건의 값이고, U-2 Workspace 프로브에는 이 샘플러가 없다. 같은 Job 안 샘플러 on/off 대조가 없으므로 "샘플러 영향 없음"으로 인용하지 않고, 샘플러 없는 기판(이전 VESSL Job, KISTI, Workspace 프로브)의 S-per 절대값과 섞어 비교하지 않는다.
- **N2C-62** rev11의 "같은 값 무관 사유로 두 번 연속 거부되면 정지" 규칙은 사유 동치가 정의되지 않았다. 사유 문자열에 경로·셀 id·개수가 들어가 같은 원인의 두 시도가 서로 다른 문자열을 낸다. `reason_code`를 반영하기 전에는 재프로브 횟수를 "정지 규칙으로 유계"라고 인용하지 않는다.

## 반증 실패(공정 기록)
- R10-1 세탁(S1–S3·S6) 폐쇄, R10-2 S4·S5b·S7 거부, R10-3 사후 편집 rc 2, VD > VI 우선순위(Y01 사살 재현), `censor_detail` 값 고정(Z13 사살).
- 등록 115종 재현(106·9·0, CTRL OK), 시험 101개 통과, 프로브 전 수치 바이트 불변, 시계 정합(Linux에서 `CLOCK_MONOTONIC`), N3 정상 종료의 끝 줄 잘림 0.
- 실행 전 `CONFIRMED` 아님.

## 자기 적용
- rev10 판정서의 처방을 등록 상수로 승격하지 않았다(#110). R10-2 처방("셀 창 전에 끝났으면")과 구현(전역 깃발)이 다르다는 점은 처방 원문과 코드를 대조해 독립적으로 찾았다.
- 이번 처방의 실현가능성(#113):
  - R11-1은 엔진이 이미 내보내는 `n_new`(`multiplexing_mixin.py` `_phase_on_prefill_start`)와 이미 등록된 K_probe 8·18만 쓴다.
  - R11-2·R11-3은 하네스가 아직 없는 지금 마크 형식에 필드·마크를 더하는 것이라 비용이 낮다.
  - H-5의 2 s는 엔진 기본 `flush_interval_s` 1.0에서 유도한 대기다(판정 문턱이 아니다).
  - 새 판정 수치 상수는 0이다.
- 처방 자신의 한계: R11-5의 commit 결속도 자기검사(교훈 9)라 드리프트만 잡는다. 문언에 그렇게 적어야 한다.
- 반전 계산은 등록 코드와 시험 픽스처만 썼고, 새 추정량은 없다.

## 근거 파일
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/n2_static_grid/n2_probe_reduce.py`
  - `ABORT_REASONS` 139–142
  - `_measure` 371–403 (중단 조기 반환 381–383)
  - `reduce_probe`: 전역 `dead` 415·423·435·440, 고아 검사 445–452, 셀 수 476–480
  - `parse_marks` 306–343, `load_calls_from_marks` 291–303 (id 덮어쓰기)
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/n2_static_grid/n2_feasibility.py`
  - `reproduce_probe` 570–587 (`sys.path` import)
  - `check_freeze_manifest` 541–567 (commit은 비어 있지 않은지만)
  - `_git_info` 590–600 (`-- .`, 기록만)
  - `main` 602–650
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/n2_static_grid/PREREG_N2_3A_2026-10-07.md` §14-2 마크 형식 사양·rev11 판정 규칙, §15 M02 행, §17-21
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/results/n2_static_grid/NOTICE_substrate_power_sampler_2026-10-08.md`
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/src/multiplex/telemetry.py` (버퍼 flush, SIGKILL 서술, `perf_counter`)
- `/home/wonho/Experiments/KISTI/prefill-layer-alloc/workspace/engine-port/benchmarks/pdmux_eval/trace_loadgen.py` 44, 90 (`sock_read=600`, 예외 `repr`)
- 감사 스크래치: `/tmp/claude-1000/-home-wonho-Experiments-KISTI/2657f214-f740-4503-8f81-241f78b1e2b2/scratchpad/a11/` (`reg.txt`, `aud.txt`, `zmut.py`, `scen.py`, `coload.py`, `misc.py`, `m02run.py`)