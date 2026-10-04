# 사전등록 追記 A14 **초안** — E2 VESSL 중단(ABORT_P3) 뒤 러너 운영 수리·P3 선택지·재실행 설계 (2026-10-04 Job 기준, 작성 2026-10-05)

> **지위: 초안(DRAFT).** 효력 없음. 규칙층 감사(claims-auditor) 전이며, 사용자 결정 3건(§6)이 정해지기 전에는
> 어떤 러너에도 적용하지 않는다. 봉인 파일(`e2_sticky_vessl.sh`·`e2_vessl_plan.py`·A13·`stage_e2_inputs.sh`·
> `tests/test_e2_vessl.py`·`scripts/vessl/job_entry.sh`)은 **한 바이트도 바꾸지 않았다.** 수정안은 이 디렉터리의
> 패치 두 개뿐이다(적용하지 않음).
> - `A14_DRAFT_runner_ops_2026-10-04.patch`(sha `8dd0fa93…`) — 포트·teardown·settle(§2). 판정 규칙 무변.
> - `A14_DRAFT_runner_p3_tolerance_2026-10-04.patch`(sha `ab14627b…`, ops 패치 **위에** 적용) — P3 선택지 (ii)(§3).
> - **사후 작성 공시**: 이 초안의 작성자(engine-porter)는 중단된 Job의 a_r4 4짝 bench 결과를 **본 뒤에** 썼다.
>   그래서 이 초안은 결정 규칙 R1–R6·추정량(rev3 §5)·셀·seed·arm·순서·라벨 어휘·`T_cap`·하드캡을 **건드리지 않는다.**
>   (ii)는 등록된 abort 게이트(P3)를 **느슨하게** 하는 변경이므로, 감사가 "결과를 본 뒤 게이트를 푼다"는 위험(교훈 251:
>   느슨하게 하는 방향만 시험하는 반전)을 명시적으로 심사해야 한다.
> - GPU 지출 0 · 성능 판정 0건 · Claim 등급 변경 0건. 커밋은 메인 세션이 한다.

---

## 0. 한 줄 요약

Job `pdmux-e2-sticky-vessl-e6f424e-20261004135228`의 boot 10 실패는 **러너 포트 선택 결함**(커널 ephemeral 범위 안에서
LISTEN 소켓만 검사)이 원인이다(확신 높음, 로컬 재현). boot 11의 `max_total_num_tokens` −3은 **기전 미확정**이며
binding resource(mamba 48)에는 닿지 않는다. 운영 수리(§2)는 판정 경로와 무관하고, P3 게이트를 바꿀지(§3)는 사용자·감사
결정이다. **현행 러너의 재개 경로는 이미 막혀 있다**(§4-1) — 재실행 설계는 §4에서 R1(새 20 boot Job)/R2(재개 가드 수정)로 고른다.

---

## 1. 진단 (근거)

### 1-1. boot 10 `a_r2_ON_4386` BOOT_FAILED — 포트 39184 충돌

- 서버 로그 `srv_a_r2_ON_4386.log`: 모델 로드·KV 할당(`#tokens: 2722025`)·cudagraph 캡처까지 정상, 이어서
  `[2026-10-04 14:47:12] ERROR:    [Errno 98] error while attempting to bind on address ('127.0.0.1', 39184): address already in use`
  → 내부 warm-up이 120 s 재시도 후 `14:49:13 Initialization failed. warmup error … Connection refused` → `kill_process_tree`.
- **39184의 점유자**: 직전 boot `a_r2_OFF_4386`(서버 포트 39398)의 bench가 마지막으로 보낸 요청의 **클라이언트측 ephemeral 포트**다.
  `srv_a_r2_OFF_4386.log:1089` `[2026-10-04 14:46:00] INFO:     127.0.0.1:39184 - "POST /generate HTTP/1.1" 200 OK`
  — 그 boot의 411번째(마지막) `/generate`. bench 종료 `TIMING.tsv` 14:46:13, 실패 bind 14:47:12 → **59 s 뒤**.
  클라이언트가 먼저 닫은 연결은 클라이언트 쪽에 TIME_WAIT(60 s)으로 남는다.
- **러너 검사가 못 본 이유**(`e2_sticky_vessl.sh` 포트 루프): 후보 `39100 + (RANDOM+$$+try) % 800`은 커널 ephemeral 범위
  안이다(이 Job 로그의 클라이언트 포트 관측 최소 32768·최대 60998 = Linux 기본 32768–60999와 일치. 컨테이너의
  `/proc/sys/net/ipv4/ip_local_port_range` 자체는 기록되지 않았다 → **미확인**, 관측 일치만 있음). 검사는
  `curl /health`(연결 거부면 통과)와 `ss -ltn`(LISTEN만)뿐이라 TIME_WAIT·ESTABLISHED 소켓을 보지 못한다.
- **로컬 재현(CPU, kernel 6.8)**: 클라이언트가 먼저 닫은 연결의 로컬 포트에 `SO_REUSEADDR`로 bind → `[Errno 98]`,
  asyncio `create_server`는 Job 로그와 **같은 문구**(`error while attempting to bind on address … address already in use`).
  같은 순간 `ss -ltn` 일치 0건. bind는 60.0 s까지 실패, 61.0 s부터 성공(TIME_WAIT 만료).
- **타이밍 여유는 0–2 s**: 마지막 요청 도착 14:46:00(초 절삭), 그 요청의 e2e ≈10.5 s(bench `ttfts+itls` 마지막 원소)
  → 클라이언트 close ≈14:46:10.5–11.5 → TIME_WAIT 만료 ≈14:47:10.5–11.5. bind 14:47:12.x와 0.5–1.5 s 어긋난다.
  이 잔차는 서버 측 FIN 지연(클라이언트가 FIN_WAIT_2에 머문 시간만큼 TIME_WAIT 시작이 늦어짐)으로 메울 수 있으나
  소켓 덤프가 없어 **정확한 TCP 상태(TIME_WAIT vs FIN_WAIT_2)는 복원 불가**. 포트 번호의 정확 일치(800 후보 중 1)와 59 s
  간격이 기전을 지지한다.
- **배제한 대안**: (a) 이전 boot 서버의 LISTEN 잔존 — 11 boot 포트 전수(39168/39108/39811/39285/39635/39188/39887/
  39462/39398/**39184**/39429)에 재사용 없음. (c) 고아 프로세스의 GPU 점유 — 다음 boot `a_r2_OFF_4162`의 `Load weight
  begin. avail mem=78.70 GB`가 다른 boot와 같고, `clk.csv`에서 실패 boot의 컨텍스트가 14:49:15에 idle(0x1)로 돌아간 뒤
  다음 boot가 14:49:47에 컨텍스트를 연다.
- **teardown 패턴의 별도 결함(영향 없음)**: SGLang 자식은 `setproctitle`로 cmdline을 `sglang::scheduler`/
  `sglang::detokenizer`로 바꾸므로 `pkill -9 -f "launch_server.*--port $2"`는 자식에 **맞지 않는다**. 자식은
  `PR_SET_PDEATHSIG=SIGKILL`(`scheduler.py:3625`, `detokenizer_manager.py:394`)로 부모와 함께 죽는다 — 이번 Job에서도
  그렇게 정리됐다(위 (c)). 감시 watchdog의 `pkill -f sglang.launch_server`도 같다.
- **λ0 VESSL 15 boot가 안 걸린 이유**: 같은 포트 로직(`lambda0_vessl.sh`, `e1_cap_vessl.sh`, `n3_capacity.sh`,
  `scripts/vessl/substrate_probe.sh`, KISTI `e2_sticky.sbatch`·`lambda0.sbatch`도 동일)이다. bench 종료 → 다음 bind가 이
  Job에서 ≈59 s라 TIME_WAIT이 살아남는 건 직전 bench의 **마지막 1–2 s에 닫힌 몇 개 연결**뿐이고, 그 포트들이 800칸 후보창에
  들어가 있을 때만 위험하다(이 Job의 9 성공 boot 중 마지막 요청 포트가 창 안이었던 것은 `a_r2_OFF_4386` 1건). 즉 boot당
  위험은 작지만 0이 아니며, **λ0의 무사고는 안전 증거가 아니다**. 서버 기동이 빨라지면(캐시 warm) 위험이 커진다.
  또한 SGLang은 HTTP bind 전에 `get_free_port()`(포트 0 bind)와 torch TCPStore 연결로 ephemeral 포트를 쓰므로,
  후보가 ephemeral 범위 안이면 **검사 후 bind 전 자기 충돌**(경쟁) 경로도 있다 — 이번 사례는 아니다(점유가 검사 시점에 이미 있었다).

### 1-2. boot 11 `a_r2_OFF_4162` −3 토큰 — 기전 미확정

- 산식(`sglang/srt/model_executor/model_runner_kv_cache_mixin.py:144-194`):
  `max_total_num_tokens = floor((post − pre·(1−0.82) − M_mamba) · 2^30 / cell_size)`.
  `pre`·`post` = `torch.cuda.mem_get_info()` free(로드 전/후, `empty_cache` 뒤), `M_mamba` = 48 × per-req mamba 상태
  (**`--max-running-requests 48`+`--disable-radix-cache`가 48을 고정** → 메모리와 무관), `cell_size` = 16,384 B
  (K 20.77 GB / 2,722,025 = 8,192 B/token, K+V).
- −3 토큰 ⇔ rest가 32,769–65,535 B 줄었다(`post`만 줄면 그만큼, `pre`·`post`가 함께 D 줄면 0.82·D → D ≈ 40–80 KB).
  한 개의 64 KiB 페이지가 양쪽 측정에 다 있었다면 −3(확률 ≈0.72) 또는 −4.
- **torch 캐싱 할당자·일반 cudaMalloc 차이로는 설명 불가**: 둘 다 2 MiB 단위 → 최소 ≈105–128 토큰 단위로 움직인다.
  남는 후보는 **2 MiB 미만 드라이버 수준 할당**(GPU 페이지 테이블·지연 모듈 로드 코드 메모리·드라이버 내부 객체)이다.
- 고아 CUDA 컨텍스트 가설은 **기각**(컨텍스트 하나는 수백 MB — 10 MB 해상도 `avail mem` 78.70/62.12/13.98/11.79가
  다른 boot와 같다; §1-1 (c)).
- 남는 두 가설(판별 불가): **H1** 비정상 종료(`kill_process_tree`의 SIGKILL) 뒤 드라이버 수준 잔여가 다음 프로세스의
  free를 수십 KB 낮춤. **H2** 프로세스 내부의 비결정적 소량 할당(VA 배치에 따른 페이지 테이블 등). 정황: VESSL 26 boot
  (λ0 VESSL 15 + 이 Job 11; KISTI 908623 11 boot도 전부 2,722,025) 중 유일한 이탈(boot 11)이 **유일한 비정상 종료
  직후**에 나왔다 — 독립이라면 우연 일치 확률 ≈1/25(사후 단일 사례, 증거 약함).
- 서버 로그는 boot 11과 다른 boot가 KV 할당 줄 전까지 포인터·시간 외에 **동일**하다(`server_args` 포트 제외 동일).
  KV 할당은 cudagraph 캡처 **이전**(`KV Cache is allocated` 14:50:01 → `Capture cuda graph begin` 14:50:01)이라 캡처와 무관.
- `clk.csv`: 14:46–14:50 창에 power cap(0x4)은 bench 구간에만, 실패 boot 대기 구간(14:47–14:48)은 0x0. 이상 징후 없음.
  CPU: `cpu_*.txt`는 bench 전후만 기록 — 실패 boot·boot 11은 bench 전에 끝나 **cpu.stat 기록 없음**(미확인).

---

## 2. 운영 수리(판정 경로 무관) — `A14_DRAFT_runner_ops_2026-10-04.patch`

| # | 변경 | 근거 | 판정에 닿는가 |
|---|---|---|---|
| O1 | HTTP 포트를 `20000–20799`에서 뽑는다. 시작 시 `ip_local_port_range`·`ip_local_reserved_ports`를 `PORT_RANGE.txt`에 기록하고, 창이 ephemeral 범위와 겹치거나 읽을 수 없으면 `ABORT_PORT_RANGE`(exit 5) | §1-1. 창 밖이면 클라이언트·SGLang 내부의 ephemeral 할당이 창에 들어올 수 없다(경쟁 경로 제거) | 아니오(포트 번호는 측정 변수가 아니다) |
| O2 | 후보 검사를 `ss -ltn`(LISTEN) → `ss -Htan`(**모든 상태**의 로컬 포트)으로 | §1-1. 로컬 변이 검사: TIME_WAIT 포트에 대해 구 검사 일치 0, 새 검사 "in use" | 아니오 |
| O3 | teardown 끝에 `pkill -9 -f "^sglang::"` 추가(기존 순서 kill→5 s→pkill→10 s는 그대로, **뒤에** 붙임) | §1-1 teardown 결함. 정상 경로는 PDEATHSIG가 이미 처리 | 아니오 |
| O4 | 매 boot 직전 `settle`: SGLang 프로세스 0개 ∧ `memory.used ≤ 기준(boot 1 직전) + 64 MiB`가 될 때까지 최대 180 s 대기. NVML 바이트 단위 free/used를 `SETTLE.txt`에 기록. **게이트 아님**(기록만) | §1-2 H1/H2 판별 자료. 다음 −k 이탈이 나면 직전 free 바이트로 "Job 수준 지속 오프셋" vs "프로세스 내부 변동"을 가른다 | 아니오. 단 boot 간 간격이 최대 180 s 늘 수 있다(정상 경로는 0–5 s) |

- 국소 검증(이 초안 작성 시, CPU): `git archive HEAD` 복사본에 패치 적용 → `tests/test_e2_vessl.py` **48/48 OK**
  (908623 아카이브 링크 포함, 2 회: ops 단독 / ops+P3). **한계**: DRYRUN은 `settle`를 건너뛰고 실 포트 bind를 하지 않으므로
  O1의 실 기동·O4는 **미검증**. `nvidia-smi`·`pynvml` 존재는 이미지에서 미확인(없으면 `NA`로 기록만 하고 진행).
- 같은 결함이 있는 러너: `lambda0_vessl.sh`(봉인, 완료 캠페인), `e1_cap_vessl.sh`(미실행), `n3_capacity.sh`(N3, 미실행),
  `scripts/vessl/substrate_probe.sh`. **N3·E1 러너에도 같은 O1–O2가 필요**하다(별도 등록 경로).

---

## 3. P3 게이트 — 선택지 (사용자·감사 결정)

rev3 §7 `PREREG_E2_STICKY_REV3_2026-09-15.md:258`: "**P3**: 두 arm의 `max_mamba_cache_size` / `max_total_num_tokens` 불일치 ⇒ abort."
A13 §1-2는 이를 "binding resource 동일, **boot마다**"로 옮겼고 러너는 `sort -u P3_KEYS.txt | wc -l > 1`(Job의 모든 boot, 재개분 포함)로
집행했다. boot 11 시점 OFF 집합 {2,722,025×5, 2,722,022}, ON 집합 {2,722,025×4} → **리터럴 "두 arm 불일치"로도 abort**다(러너 위반 없음).

**3 토큰이 achieved에 닿는 경로**(E2C-10: achieved는 `max_running 48` 구속):
- 입장 상한은 `max_running_requests=48` = `max_mamba_cache_size=48`(고정). KV 수요 상한 = 48 × 셀 최대 길이:
  A 48×768 = 36,864, B 48×8,256 = 396,288 토큰, 어떤 경우에도 48 × context 16,384 = 786,432 ≤ 2,722,022.
  KV 풀은 **구속 자원이 아니다**(로그 `full token usage: 0.00`).
- `FixedPolicy.decide`(`multiplex/controller.py:88-104`)는 occupancy를 읽지 않는다. `kv_occupancy`는 telemetry 스냅샷에만
  실리며(`multiplexing_mixin.py:518-537`, max(full, mamba) — mamba가 지배) 상대 1e-6 차이다. 추정량(`e2_realized_mix.py`)은 이 필드를 쓰지 않는다.
- ⇒ 3 토큰이 achieved·지연·실현 분할에 닿는 **코드 경로는 찾지 못했다**(메모리 배치 차이에 따른 커널 성능 차이 같은 2차 경로는 배제 증명 불가).

| 선택지 | 내용 | 등록 변경 | 위험 |
|---|---|---|---|
| **(i)** 리터럴 유지 | 지금 그대로 | 없음(ops 패치만이면 A14 운영 追記 + 하네스층 국소 감사) | −3류 이탈이 다시 나면 또 abort. 기저율 미상(VESSL 26 중 1, H1이면 실패 boot가 없을 때 0에 가깝다). R1 재실행 20 boot 기준, 이탈이 1/25 독립이라면 Job 완주 확률 ≈ (24/25)^19 ≈ 0.46 — **추정 근거 약함** |
| **(ii)** 허용오차 | mamba **정확 일치** ∧ 모든 토큰 키 ≥ `MAX_RUNNING×CTX`(786,432) ∧ (max−min)/max ≤ 1e-4. 원 키는 그대로 기록 | **예** — 등록 abort 게이트 변경. A14 본문 + **규칙층 감사 필수**. 결과 문서에 P3_VERDICTS 병기 | 게이트를 사후에 푼다(§0 공시). 단 바인딩 자원(mamba)은 여전히 정확 일치를 요구하고, 토큰 하한이 "KV가 절대 구속하지 않음"을 구조적으로 보장 |
| **(iii)** 대기 추가만 | ops 패치의 O4(정리 대기)로 H1을 줄이고 게이트는 리터럴 유지 | (i)과 같음 | H2면 효과 없음. O4는 판별 자료를 남긴다 |

- 패치 (ii)의 규칙 단위 점검(이 초안): 이 Job의 P3_KEYS → `PASS rel=1.102e-06`; 변이 mamba {47,48} → FAIL; 토큰 0.8% 차 → FAIL;
  토큰 700,000(<하한) → FAIL; 배너 누락 → FAIL.
- **권고(engine-porter)**: **O1–O4(ops) 필수 + (ii) 채택**. 근거: −3의 기전이 미확정이라 (iii)만으로는 재발을 막는다고 말할 수 없고,
  (i)은 측정과 무관한 사유로 20 boot Job을 다시 날릴 위험을 그대로 둔다. (ii)는 등록 취지("두 arm의 binding resource 동일")를
  mamba 정확 일치 + KV 비구속 하한으로 **더 정확히** 집행한다. 단 이것은 엔지니어링 권고이고, 게이트 완화의 승인은 사용자·감사 몫이다.
  (ii)를 기각하면 (i)+(iii)로 간다.

---

## 4. 재실행 설계

### 4-1. ★현행 러너의 재개(`PDMUX_E2V_RESUME_FROM`)는 이미 막혀 있다

러너는 이전 Job과 이번 Job의 `REGISTRATION_SHA256.txt`에서 `e2_|PREREG_|VERDICT_|OVERRIDE_` 줄의 해시 열을 `diff`해 다르면
`ABORT_RESUME`(exit 2)한다. 이 목록에는 `e2_sticky_vessl.sh`·`e2_vessl_plan.py`·`OVERRIDE_E2_VESSL_*.md`(glob)가 들어 있다.
- 커밋 `9a29f84`가 `OVERRIDE_E2_VESSL_2026-10-04.md`에 소진 기록을 적어 digest가 `46c85d71…` → `131abfff…`로 바뀌었다.
  현재 HEAD로 같은 목록을 재계산하면 20줄 중 그 1줄이 다르다(이 초안 작성 시 확인) ⇒ **패치를 하나도 안 해도 재개는 ABORT_RESUME**.
- 새 OVERRIDE 파일(직전 OVERRIDE는 소진 — 재승인 필요)을 같은 디렉터리에 `OVERRIDE_E2_VESSL_<날짜>.md`로 두면 glob이 줄을 하나 더 만들어 역시 abort.
- ops/P3 패치는 러너 digest를 바꾸므로 역시 abort.
- 이 생애주기(OVERRIDE 소진 기록 → 재개)는 `TestResume`이 다루지 않았다(테스트는 이전 Job의 REGISTRATION을 그대로 복사해 쓴다).
- 이전 Job 디렉터리(`/data/runs/<job>/results/r2_eval/e2_sticky_vessl/e2v_<job>`, cluster 볼륨)가 아직 남아 있는지는 **미확인**.

### 4-2. 선택지

| | **R1 — 새 Job, 20 boot 전부**(권고) | **R2 — 재개 12 boot** |
|---|---|---|
| 내용 | A14 러너로 rev3 순서 20 boot. 중단 Job의 a_r4 4짝은 **보조 기술 자료**로만 남기고 섞지 않는다 | a_r4 4짝 재사용, 남은 짝 a_r2 4짝(s 4386은 ON boot 실패, s 4162는 OFF가 P3 abort로 bench 미실행, 251·2630 미시도) + b_r3 2짝 = 12 boot |
| 러너 추가 변경 | 없음(ops±P3) | 재개 가드를 "등록 목록의 허용 차이(러너·OVERRIDE)"를 받도록 수정 — **가드 완화** = 추가 감사 표면 |
| Job 혼합(E2V-7) | 없음 | 있음. a_r4는 구 러너(포트 결함·P3 리터럴) Job, 나머지는 A14 러너 Job. R5·R4″를 "같은 Job 안 산포"로 쓰지 못함, 라벨 인용에 "Job 혼합(짝 단위)" 병기 |
| P3 키 합침 | 해당 없음 | 재개 시 이전 Job 배너(a_r4 8줄 = 2,722,025)가 P3_KEYS에 먼저 들어간다 → (i)/(iii)이면 새 boot가 2,722,022 등을 내면 **즉시 다시 abort**. (ii)면 통과 |
| P13 | 이 Job에서 양 arm | 이 Job의 첫 OFF·ON boot에서 다시 돈다(러너 기존 동작) |
| 예산(VESSL 상수, $1.48/h) | 등록 그대로: 보통 1.991 GPU-h **$2.95** · 최악 2.533 **$3.75** · 셀 구간 하드캡 4.0 **$5.92**. 관측 보정(이 Job의 A ON boot 벽시계 ≈375 s 반영, B 최악 코너) 2.24 GPU-h ≈$3.32 | 같은 식으로 12 boot: 보통 1.339 GPU-h **$1.98** · 최악 1.657 **$2.45** · 전부 `T_cap` 1단 2.619 **$3.88**(하드캡 4.0 h 안). 관측 보정 1.50 GPU-h ≈$2.22 |
| 절감 | — | 보통 기준 ≈$0.97(0.65 GPU-h) |

예산 산식은 `e2_vessl_plan.simulate`와 같다(preflight 600 s + 첫 boot cold 92.1 s + P13 30 s×2 + boot별 bench+부대비용 A 160.1/B 121.0 s).
**관측 주의**: 이 Job의 a_r4 ON boot 벽시계는 373–391 s(bench 197.2–197.7 s + ON warm-up이 OFF보다 ≈15 s 길다)로 등록 "보통" 가정
(ON bench = OFF bench)보다 길다. 등록 최악 코너(A ×2.5) 안이다. 중단 Job 실지출 0.957–0.970 GPU-h($1.42–1.44)는 위 수치에 포함하지 않았다.

- **권고**: R1. 절감($≈1)보다 R2가 더하는 것(재개 가드 완화의 감사, Job 혼합 caveat, 구 러너 a_r4와 A14 러너 결과의 혼합)이 크다.
  R1이면 중단 Job의 a_r4 4짝과 새 Job의 a_r4 4짝을 **나란히 기술**할 수 있다(합치지 않음, 판정 아님).

---

## 5. 새 caveat (A14 채택 시 등록 후보)

- **E2V-9** — 러너 운영 수리(O1–O4)는 포트·정리·대기·기록만 바꾼다. 서빙 명령줄·arm 노브·추정량·규칙 무변. O4는 boot 간 간격을 최대 180 s
  늘릴 수 있어 E2V-5(드리프트)의 시간 축이 그만큼 늘 수 있다 → `SETTLE.txt`의 `settle_polls`를 결과 문서에 병기.
- **E2V-10**((ii) 채택 시) — P3는 "mamba 정확 일치 + KV 비구속 하한 + 토큰 상대차 ≤1e-4"로 집행된다. 결과 문서는 `P3_KEYS.txt` 원 키와
  `P3_VERDICTS.txt`를 그대로 싣고, 토큰 키가 둘 이상이었으면 그 값을 적는다. "P3가 두 arm의 KV 풀이 같음을 확인했다"고 쓰지 않는다.
- 중단 Job(`…20261004135228`)의 수치는 **D23**에 따라 라벨이 없다. 어떤 결과 문서도 그 a_r4 4짝으로 R1–R5를 계산하지 않는다.

## 6. 사용자 결정 대기

1. P3: (ii) 채택 / (i)+(iii).
2. 재실행: R1(새 20 boot) / R2(재개 12 boot + 가드 수정).
3. 새 범위 한정 OVERRIDE(직전 OVERRIDE는 `…20261004135228` 1건으로 소진).

선행조건(제출 전): A14 본문 확정 → 규칙층 감사(claims-auditor; (ii)·R2를 고르면 필수, (i)+(iii)+R1이면 하네스층 국소 감사로 충분한지도 감사가 정한다)
→ 패치 적용·테스트(`test_e2_vessl.py` 48 + O1/O2 변이 시험 추가 권고: 구 검사 복원 변이에서 TIME_WAIT 포트를 통과시키는지)
→ 커밋 → `launch.sh` dry-run → 사용자 승인 → `--submit`.
