# 사전등록 追記 A14 — E2 VESSL 중단(ABORT_P3) 뒤 러너 운영 수리 · P3 허용오차 (ii) · 새 20 boot Job (R1) (2026-10-05)

> **지위: 확정본.** 사용자 결정 3건(2026-10-05 "231 순서로 진행")을 반영했다. **규칙층 감사(claims-auditor) 전**이며,
> 감사 `GO`(또는 `GO-with-caveats`) + 하네스층 국소 확인 전에는 제출하지 않는다(§9).
> - 이 문서는 초안 `PREREG_E2_VESSL_A14_DRAFT_2026-10-04.md`(sha `f0118ba1…`, 보존·불변)를 대신한다. 초안 대비 변경은 §10 색인.
> - **1회차 기록 보존**: Job `pdmux-e2-sticky-vessl-e6f424e-20261004135228`(이하 **1회차 중단 Job**)의 봉인 파일은 한 바이트도 바꾸지 않았다.
>   러너 `e2_sticky_vessl.sh`, helper `e2_vessl_plan.py`, A13, `stage_e2_inputs.sh`, `tests/test_e2_vessl.py`, 소진된 `OVERRIDE_E2_VESSL_2026-10-04.md`가 그 대상이다.
>   규칙층·하네스층 판정서 2개와 초안 패치 2개도 그대로 둔다.
>   새 실행물은 **새 파일**이다: 러너 `e2_sticky_vessl_r2.sh`, helper `e2_vessl_plan_r2.py`, 테스트 `tests/test_e2_vessl_r2.py`, OVERRIDE `OVERRIDE_E2_VESSL_R2_2026-10-05.md`.
>   r2 러너는 1회차 봉인 파일 6개의 바이트를 Job 안에서 다시 확인한다. 하나라도 다르면 `ABORT_REG_DRIFT`(exit 3)다.
> - **사후 작성 공시**: 이 追記의 작성자(engine-porter)는 1회차 중단 Job의 a_r4 4짝 bench 결과를 **본 뒤에** 초안을 썼다.
>   그래서 결정 규칙 R1–R6, 추정량(rev3 §5), 셀, seed, arm, 순서, 라벨 어휘, `T_cap`, 하드캡은 **건드리지 않는다.**
>   (ii)는 등록 abort 게이트(P3)를 **느슨하게** 하는 변경이다. 규칙층 감사는 "결과를 본 뒤 게이트를 푼다"는 위험을 심사해야 한다(교훈 251, §3-3).
> - GPU 지출 0 · 성능 판정 0건 · Claim 등급 변경 0건. 커밋은 메인 세션이 한다.

---

## 0. 한 줄 요약

1회차 중단 Job의 boot 10 실패는 **러너 포트 선택 결함** 때문이다(확신 높음, 로컬 재현). 러너가 커널 ephemeral 범위 안에서 포트를 뽑고, LISTEN 소켓만 검사했다.
boot 11의 `max_total_num_tokens` −3은 **기전 미확정**이며 구속 자원(mamba 48)에는 닿지 않는다.
사용자 결정에 따라 세 가지를 등록한다.
- 운영 수리 O1–O4(§2). 판정 경로와 무관하다.
- P3 허용오차 (ii)(§3).
- **새 20 boot Job(R1)**(§4). 재개는 쓰지 않는다.

---

## 1. 진단 (근거)

### 1-1. boot 10 `a_r2_ON_4386` BOOT_FAILED — 포트 39184 충돌

- 서버 로그 `srv_a_r2_ON_4386.log`: 모델 로드, KV 할당(`#tokens: 2722025`), cudagraph 캡처까지 정상이었다. 이어서
  `[2026-10-04 14:47:12] ERROR:    [Errno 98] error while attempting to bind on address ('127.0.0.1', 39184): address already in use`가 났다.
  내부 warm-up이 120 s 재시도한 뒤 `14:49:13 Initialization failed. warmup error … Connection refused` → `kill_process_tree`.
- **39184의 점유자**: 직전 boot `a_r2_OFF_4386`(서버 포트 39398)의 bench가 마지막으로 보낸 요청의 **클라이언트측 ephemeral 포트**다.
  - `srv_a_r2_OFF_4386.log:1089` `[2026-10-04 14:46:00] INFO:     127.0.0.1:39184 - "POST /generate HTTP/1.1" 200 OK` — 그 boot의 411번째(마지막) `/generate`.
  - bench 종료(`TIMING.tsv`) 14:46:13, 실패 bind 14:47:12 → **59 s 뒤**.
  - 클라이언트가 먼저 닫은 연결은 클라이언트 쪽에 TIME_WAIT(60 s)으로 남는다.
- **러너 검사가 못 본 이유**(`e2_sticky_vessl.sh:320-327` 포트 루프):
  - 후보 `39100 + (RANDOM+$$+try) % 800`은 커널 ephemeral 범위 안이다.
    이 Job 로그의 클라이언트 포트 관측 최소 32768·최대 60998은 Linux 기본 32768–60999와 일치한다.
    컨테이너의 `/proc/sys/net/ipv4/ip_local_port_range` 자체는 기록되지 않았다 → **미확인**, 관측 일치만 있다.
  - 검사는 `curl /health`(연결 거부면 통과)와 `ss -ltn`(LISTEN만)뿐이다. TIME_WAIT·ESTABLISHED 소켓은 보지 못한다.
- **로컬 재현(CPU, kernel 6.8)**: 클라이언트가 먼저 닫은 연결의 로컬 포트에 `SO_REUSEADDR`로 bind하면 `[Errno 98]`이 난다.
  asyncio `create_server`는 Job 로그와 **같은 문구**를 낸다(`error while attempting to bind on address … address already in use`).
  같은 순간 `ss -ltn` 일치는 0건이다. bind는 60.0 s까지 실패하고 61.0 s부터 성공한다(TIME_WAIT 만료).
- **타이밍 여유는 0–2 s**:
  - 마지막 요청 도착 14:46:00(초 절삭), 그 요청의 e2e ≈10.5 s(bench `ttfts+itls` 마지막 원소).
  - → 클라이언트 close ≈14:46:10.5–11.5 → TIME_WAIT 만료 ≈14:47:10.5–11.5. bind 14:47:12.x와 0.5–1.5 s 어긋난다.
  - 이 잔차는 서버 측 FIN 지연으로 메울 수 있다(클라이언트가 FIN_WAIT_2에 머문 시간만큼 TIME_WAIT 시작이 늦어진다).
    그러나 소켓 덤프가 없어 **정확한 TCP 상태(TIME_WAIT vs FIN_WAIT_2)는 복원할 수 없다**.
  - 기전을 지지하는 것은 포트 번호의 정확 일치(800 후보 중 1)와 59 s 간격이다.
- **배제한 대안**:
  - (a) 이전 boot 서버의 LISTEN 잔존 — 11 boot 포트 전수(39168/39108/39811/39285/39635/39188/39887/39462/39398/**39184**/39429)에 재사용이 없다.
  - (c) 고아 프로세스의 GPU 점유 — 다음 boot `a_r2_OFF_4162`의 `Load weight begin. avail mem=78.70 GB`가 다른 boot와 같다.
    `clk.csv`에서 실패 boot의 컨텍스트가 14:49:15에 idle(0x1)로 돌아간 뒤 다음 boot가 14:49:47에 컨텍스트를 연다.
- **teardown 패턴의 별도 결함(이번 영향 없음)**:
  - SGLang 자식은 `setproctitle`로 cmdline을 `sglang::scheduler`/`sglang::detokenizer`로 바꾼다. 그래서 `pkill -9 -f "launch_server.*--port $2"`는 자식에 **맞지 않는다**.
  - 자식은 `PR_SET_PDEATHSIG=SIGKILL`(`scheduler.py:3625`, `detokenizer_manager.py:394`)로 부모와 함께 죽는다. 이번 Job에서도 그렇게 정리됐다(위 (c)).
  - 감시 watchdog의 `pkill -f sglang.launch_server`도 같은 패턴이다. r2는 watchdog를 바꾸지 않았다(§2-1 끝).
- **λ0 VESSL 15 boot가 안 걸린 이유**(초안 문구 정정 — 아래는 파일:줄을 직접 확인한 목록이다):
  - 같은 후보창 `39100 + (RANDOM+$$+try) % 800`과 LISTEN 전용 검사(`ss -ltn`)를 쓰는 러너:
    - VESSL에서 실제로 돈 것: `lambda0_vessl.sh:372-377`(λ0 VESSL 15 boot), `e2_sticky_vessl.sh:320-327`(1회차 중단 Job).
    - VESSL용이나 아직 안 돈 것: `e1_cap_vessl.sh:188-194`, `scripts/vessl/substrate_probe.sh:405-409`.
    - KISTI 이력: `e2_sticky_prereg/e2_sticky.sbatch:200-205`, `lambda0_prereg/lambda0.sbatch:316-321`.
      KISTI 노드의 ephemeral 범위는 기록이 없다 → 그 창이 KISTI에서 ephemeral 범위와 겹쳤는지는 **미확인**이다.
    - `n3_capacity.sh`는 N3 rev2에서 이미 O1–O4를 반영했다(창 20000–20799, `ss -Htan`). 초안의 "같은 포트 로직" 목록에 넣은 것은 그 반영 전 상태였다.
  - bench 종료 → 다음 bind 간격은 이 Job에서 ≈59 s였다. 그래서 TIME_WAIT이 살아남는 것은 직전 bench의 **마지막 1–2 s에 닫힌 몇 개 연결**뿐이다.
    그 포트들이 800칸 후보창 안에 있을 때만 위험하다. 이 Job의 9 성공 boot 중 마지막 요청 포트가 창 안이었던 것은 `a_r2_OFF_4386` 1건이다.
  - 즉 boot당 위험은 작지만 0이 아니며, **λ0의 무사고는 안전 증거가 아니다**. 서버 기동이 빨라지면(캐시 warm) 위험이 커진다.
  - 또한 SGLang은 HTTP bind 전에 `get_free_port()`(포트 0 bind)와 torch TCPStore 연결로 ephemeral 포트를 쓴다.
    후보가 ephemeral 범위 안이면 **검사 후 bind 전 자기 충돌**(경쟁) 경로도 있다. 이번 사례는 아니다(점유가 검사 시점에 이미 있었다).

### 1-2. boot 11 `a_r2_OFF_4162` −3 토큰 — 기전 미확정

- 산식(`sglang/srt/model_executor/model_runner_kv_cache_mixin.py:144-194`):
  `max_total_num_tokens = floor((post − pre·(1−0.82) − M_mamba) · 2^30 / cell_size)`.
  - `pre`·`post` = `torch.cuda.mem_get_info()` free(로드 전/후, `empty_cache` 뒤).
  - `M_mamba` = 48 × per-req mamba 상태. **`--max-running-requests 48`+`--disable-radix-cache`가 48을 고정**하므로 메모리와 무관하다.
  - `cell_size` = 16,384 B(K 20.77 GB / 2,722,025 = 8,192 B/token, K+V).
- −3 토큰 ⇔ rest가 32,769–65,535 B 줄었다(`post`만 줄면 그만큼, `pre`·`post`가 함께 D 줄면 0.82·D → D ≈ 40–80 KB).
  64 KiB 페이지 하나가 양쪽 측정에 다 있었다면 −3(확률 ≈0.72) 또는 −4.
- **torch 캐싱 할당자·일반 cudaMalloc 차이로는 설명할 수 없다**: 둘 다 2 MiB 단위라 최소 ≈105–128 토큰 단위로 움직인다.
  남는 후보는 **2 MiB 미만 드라이버 수준 할당**이다(GPU 페이지 테이블, 지연 모듈 로드 코드 메모리, 드라이버 내부 객체).
- 고아 CUDA 컨텍스트 가설은 **기각**한다. 컨텍스트 하나는 수백 MB인데, 10 MB 해상도 `avail mem` 78.70/62.12/13.98/11.79가 다른 boot와 같다(§1-1 (c)).
- 남는 두 가설은 **판별할 수 없다**:
  - **H1** 비정상 종료(`kill_process_tree`의 SIGKILL) 뒤 드라이버 수준 잔여가 다음 프로세스의 free를 수십 KB 낮춘다.
  - **H2** 프로세스 내부의 비결정적 소량 할당(VA 배치에 따른 페이지 테이블 등).
- 서버 로그는 boot 11과 다른 boot가 KV 할당 줄 전까지 포인터·시간 외에 **동일**하다(`server_args`는 포트 외 동일).
  KV 할당은 cudagraph 캡처 **이전**이다(`KV Cache is allocated` 14:50:01 → `Capture cuda graph begin` 14:50:01). 캡처와 무관하다.
- `clk.csv`: 14:46–14:50 창에서 power cap(0x4)은 bench 구간에만 있었다. 실패 boot 대기 구간(14:47–14:48)은 0x0이다. 이상 징후는 없다.
  CPU: `cpu_*.txt`는 bench 전후만 기록한다. 실패 boot와 boot 11은 bench 전에 끝나 **cpu.stat 기록이 없다**(미확인).

---

## 2. 운영 수리 O1–O4 (판정 경로 무관)

| # | 변경 | 근거 | 판정에 닿는가 |
|---|---|---|---|
| O1 | HTTP 포트를 `20000–20799`에서 뽑는다. 시작 시 `ip_local_port_range`·`ip_local_reserved_ports`를 `PORT_RANGE.txt`에 기록한다. 창이 ephemeral 범위와 겹치거나 읽을 수 없으면 `ABORT_PORT_RANGE`(exit 5) | §1-1. 창이 범위 밖이면 클라이언트·SGLang 내부의 ephemeral 할당이 창에 들어올 수 없다(경쟁 경로 제거) | 아니오(포트 번호는 측정 변수가 아니다) |
| O2 | 후보 검사를 `ss -ltn`(LISTEN) → `ss -Htan`(**모든 상태**의 로컬 포트)으로 바꾼다 | §1-1. 변이 시험: TIME_WAIT 포트에 대해 구 검사는 일치 0, 새 검사는 "in use" | 아니오 |
| O3 | teardown 끝에 `pkill -9 -f "^sglang::"`를 붙인다. 기존 순서(kill→5 s→pkill→10 s)는 그대로 두고 **뒤에** 붙인다 | §1-1 teardown 결함. 정상 경로는 PDEATHSIG가 이미 처리한다 | 아니오 |
| O4 | 매 boot 전 `settle`: SGLang 프로세스 0개 ∧ `memory.used ≤ 기준(boot 1 직전) + 64 MiB`가 될 때까지 최대 180 s(5 s × 36) 기다린다. NVML 바이트 단위 free/used를 `SETTLE.txt`에 기록한다. **gate가 아니다**(기록만) | §1-2 H1/H2 판별 자료. 다음에 −k 이탈이 나면 직전 free 바이트로 "Job 수준 지속 오프셋"과 "프로세스 내부 변동"을 가른다 | 아니오. 단 boot 사이 간격이 최대 180 s 늘 수 있다(정상 경로 0–5 s 예상, 미측정) |

### 2-1. 초안 패치에서 달라진 점 (r2 러너)

r2 러너는 1회차 러너의 **바이트 사본**에 초안 패치 두 개(`A14_DRAFT_runner_ops_2026-10-04.patch` sha `8dd0fa93…`, `A14_DRAFT_runner_p3_tolerance_2026-10-04.patch` sha `ab14627b…`)를 그대로 적용한 뒤 아래만 바꿨다.
- **O4 위치**: 초안은 settle를 P14 입장 판정 **뒤**에 두었다. 그러면 settle 최대 180 s가 P14 여유(200 s) 안에서 소비된다. bench가 `BUDGET` 절단으로 끝날 수 있다.
  r2는 settle를 P14 입장 판정 **앞**으로 옮겼다. 입장 판정은 settle가 쓴 시간을 본 뒤의 남은 시간으로 내려진다. P14 규칙(남은 시간 ≥ `T_cap`+200 s)은 그대로다.
  (N3 rules rev1 감사가 같은 점을 지적했다 — "settle 최대 180 s/boot가 예산 모형에 없음".) 예산 모형에는 §5 `worst_settle` 행으로 넣었다.
- **P3 판정 위치**: 초안의 인라인 heredoc 대신 `e2_vessl_plan_r2.py --p3-verdict`를 부른다. 규칙은 §3-1이고, 단위 시험과 Job 안 셀프테스트(`SELFTEST_port_r2.txt`)가 같은 함수를 시험한다.
  helper가 죽거나 출력이 비면 `PASS`가 아니므로 abort다(fail-closed).
- **mamba 수준값 고정**: 초안은 "mamba 값이 Job 안에서 한 가지"만 요구했다. r2는 **모든 boot에서 정확히 48**을 요구한다(사용자 결정 문언 "mamba 48 정확 일치").
- **재개 거부**: `PDMUX_E2V_RESUME_FROM`이 설정되면 `ABORT_RESUME`(exit 2)다. A13 §5-4 재개 블록, 재개 짝 건너뛰기, P13 재개 대체 경로를 지웠다(§4).
- **등록 확인 추가**: `--verify-digests-r2`가 A14·r2 러너 digest와 1회차 봉인 파일 6개 바이트를 확인한다. `REGISTRATION_SHA256.txt`에 A14·r2 러너·r2 helper가 더해진다.
  1회차 러너도 기록용으로 함께 해시한다.
- **바꾸지 않은 것**: 서버/warm-up/bench/P13 명령줄, 노브, 셀·seed·순서, 추정량·규칙 호출, `T_cap`, 하드캡 3겹, watchdog 패턴.
  watchdog의 `pkill -f sglang.launch_server`는 §1-1의 teardown 결함과 같은 한계가 있다(자식은 PDEATHSIG로 정리된다). 초안 범위 밖이라 그대로 두었다.

### 2-2. 같은 결함이 남은 러너 (이 追記의 범위 밖)

`e1_cap_vessl.sh`(미실행)와 `scripts/vessl/substrate_probe.sh`에는 같은 O1–O2가 필요하다. 각자의 등록 경로로 처리한다. `lambda0_vessl.sh`는 봉인된 완료 캠페인이라 손대지 않는다.

---

## 3. P3 게이트 — **(ii) 허용오차 채택** (사용자 결정, 2026-10-05)

등록 원문(rev3 §7, `PREREG_E2_STICKY_REV3_2026-09-15.md:258`): "**P3**: 두 arm의 `max_mamba_cache_size` / `max_total_num_tokens` 불일치 ⇒ abort."
A13 §1-2는 이를 "binding resource 동일, **boot마다**"로 옮겼다. 1회차 러너는 `sort -u P3_KEYS.txt | wc -l > 1`(Job의 모든 boot)로 집행했다.
boot 11 시점 OFF 집합은 {2,722,025×5, 2,722,022}, ON 집합은 {2,722,025×4}였다 → **리터럴 "두 arm 불일치"로도 abort**다(러너 위반 없음).

### 3-1. 등록 규칙 (ii)

이 Job의 모든 boot에 대해 다음 셋이 모두 성립해야 한다. 하나라도 어기면 `ABORT_P3`(exit 8)다.
- (a) `max_mamba_cache_size` = **48**, 모든 boot에서 정확히(허용오차 없음).
- (b) 모든 `max_total_num_tokens` ≥ `MAX_RUNNING × CTX` = 48 × 16,384 = **786,432**.
- (c) `max_total_num_tokens`의 (max − min) / max ≤ **1e-4**.
- 배너를 찾지 못한 boot(`NONE`)는 abort다. `MAX_RUNNING`·`CTX`가 등록값(48, 16,384)이 아니어도 abort다.
- 원 키는 `P3_KEYS.txt`·`BANNERS.txt`에 그대로 남고, boot마다 판정 줄이 `P3_VERDICTS.txt`에 남는다.

### 3-2. 왜 3 토큰이 achieved에 닿지 않는가 (KV 비구속 하한의 근거)

E2C-10: achieved는 `max_running 48`에 구속된다.
- 입장 상한은 `max_running_requests=48` = `max_mamba_cache_size=48`(고정)이다.
  KV 수요 상한 = 48 × 셀 최대 길이다. A는 48×768 = 36,864, B는 48×8,256 = 396,288 토큰이다.
  어떤 경우에도 48 × context 16,384 = 786,432 ≤ 2,722,022다. KV 풀은 **구속 자원이 아니다**(로그 `full token usage: 0.00`).
- `FixedPolicy.decide`(`multiplex/controller.py:88-104`)는 occupancy를 읽지 않는다.
  `kv_occupancy`는 telemetry 스냅샷에만 실린다(`multiplexing_mixin.py:518-537`, max(full, mamba) — mamba가 지배). 차이는 상대 1e-6이다.
  추정량(`e2_realized_mix.py`)은 이 필드를 쓰지 않는다.
- ⇒ 3 토큰이 achieved·지연·실현 분할에 닿는 **코드 경로는 찾지 못했다**. 메모리 배치 차이에 따른 커널 성능 차이 같은 2차 경로는 배제를 증명할 수 없다.

### 3-3. 이 결정의 위험 공시 (규칙층 감사 심사 대상)

- **사후 완화**: (ii)는 1회차 중단 Job의 결과를 본 뒤 등록 abort 게이트를 느슨하게 한다(§0 공시).
  느슨해지는 표면은 "토큰 키가 boot마다 1e-4 안에서 다를 수 있다" 하나다. 그 표면이 판정량(achieved·실현 분할·지연)에 닿는 코드 경로는 §3-2에서 찾지 못했다.
- **조이는 방향(교훈 251)**: (ii)는 리터럴 게이트에 없던 실패 조건 둘을 더한다. (a) mamba 수준값 48 고정, (b) 토큰 하한 786,432다.
  리터럴 게이트는 모든 boot가 같기만 하면 47이나 700,000도 통과시켰다. 이 둘은 E2VC-7(“P3는 수준값을 강제하지 않는다”)의 일부를 메운다.
  토큰 수준값 2,722,025는 여전히 강제하지 않는다 → 결과 문서는 E2VC-7대로 BANNERS를 48 / 2,722,025와 대조해 병기한다.
- **전례(N3 rev3 D2와 같은 균형 문구)**:
  - VESSL boot 26개(λ0 VESSL 15 + 1회차 중단 Job 11) 중 이탈은 boot 11 1건뿐이다. KISTI 908623의 11 boot는 전부 2,722,025였다.
  - 그 이탈은 **그 Job의 유일한 비정상 종료(포트 충돌 `BOOT_FAILED`) 바로 다음 boot**에서 나왔다.
  - 두 가설은 판별할 수 없다: **H1** 비정상 종료 뒤 드라이버 수준 잔여, **H2** 프로세스 내부 비결정적 소량 할당.
  - O1/O2(포트 충돌 원인 제거)·O3(`^sglang::` 정리)·O4(boot 전 GPU 메모리 baseline 복귀 대기)는 **H1 경로를 겨냥**한다. H2라면 막지 못한다.
  - 이 등록은 어느 쪽인지 판정하지 않는다. (ii)는 어느 가설에서도 같은 규모(수 토큰)의 이탈에 Job을 멈추지 않는다.
- **기각한 선택지(기록)**: (i) 리터럴 유지, (iii) 리터럴 + O4 대기만. 기각 사유: −3의 기전이 미확정이라 (iii)만으로 재발을 막는다고 말할 수 없다.
  (i)은 측정과 무관한 사유로 20 boot Job을 다시 잃을 위험을 그대로 둔다.
- 국소 변이 시험(§11): 1회차 중단 Job의 키(9 × 2,722,025 + 2,722,022) → `PASS rel=1.102e-06`.
  mamba {47,48} · 전부 47 · 토큰 0.8% 차 · 786,431 · 배너 누락 → 전부 `FAIL`. 경계(786,432, 상대차 정확히 1e-4) → `PASS`.

---

## 4. 재실행 설계 — **R1 채택**(새 20 boot Job, 사용자 결정 2026-10-05)

### 4-1. 1회차 러너의 재개 경로는 이미 막혀 있다 (사실 기록)

1회차 러너는 이전 Job과 이번 Job의 `REGISTRATION_SHA256.txt`에서 `e2_|PREREG_|VERDICT_|OVERRIDE_` 줄의 해시 열을 `diff`한다. 다르면 `ABORT_RESUME`(exit 2)다.
이 목록에는 `e2_sticky_vessl.sh`·`e2_vessl_plan.py`·`OVERRIDE_E2_VESSL_*.md`(glob)가 들어 있다.
- 커밋 `9a29f84`가 `OVERRIDE_E2_VESSL_2026-10-04.md`에 소진 기록을 적어 digest가 `46c85d71…` → `131abfff…`로 바뀌었다.
  ⇒ 패치를 하나도 하지 않아도 재개는 `ABORT_RESUME`이다.
- 새 OVERRIDE 파일을 같은 디렉터리에 두면 glob이 줄을 하나 더 만들어 역시 abort다. ops/P3 변경도 러너 digest를 바꾸므로 역시 abort다.
- 이 생애주기(OVERRIDE 소진 기록 → 재개)는 `TestResume`이 다루지 않았다(테스트는 이전 Job의 REGISTRATION을 그대로 복사해 쓴다).

### 4-2. R1 채택 사유

| | **R1 — 새 Job, 20 boot 전부(채택)** | R2 — 재개 12 boot(기각) |
|---|---|---|
| 재개 가드 | 쓰지 않는다. r2 러너는 `PDMUX_E2V_RESUME_FROM`을 거부한다 | §4-1의 digest 차단을 풀려면 가드를 **완화**해야 한다 = 새 감사 표면 |
| Job 혼합(E2V-7) | 없다 | 있다. a_r4는 구 러너(포트 결함·P3 리터럴) Job, 나머지는 r2 러너 Job이 된다. R5·R4″를 "같은 Job 안 산포"로 쓰지 못한다 |
| P3 키 | 이 Job의 boot만 | 이전 Job 배너(a_r4 8줄)가 먼저 들어간다 |
| 절약 | — | 보통 기준 ≈0.65 GPU-h ≈ **$0.97뿐** |

⇒ 절약(≈$1)보다 R2가 더하는 것(재개 가드 완화의 감사, Job 혼합 caveat, 두 러너 결과의 혼합)이 크다.

### 4-3. R1의 집행

- r2 러너는 시작 직후(PDMUX 허용목록 다음) `PDMUX_E2V_RESUME_FROM`이 비어 있지 않으면 `ABORT_RESUME`(exit 2)로 멈춘다.
- 20 boot는 rev3 순서 그대로 이 Job 안에서 모두 측정된다. 1회차 중단 Job의 짝은 어떤 경로로도 이 Job의 `OUT`에 들어오지 않는다.
- 이 Job이 다시 중단돼도 **재개 Job은 없다.** 다음 Job은 다시 새 20 boot이며 새 OVERRIDE(사용자 재승인)가 필요하다.

---

## 5. 예산 (VESSL `resourcespec-a100x1`, $1.48/GPU-h)

산식은 `e2_vessl_plan.simulate`와 같다: preflight 600 s + 첫 boot cold 92.1 s + P13 30 s × 2 + boot별 (bench + 부대비용 A 160.1 / B 121.0 s).
`e2_vessl_plan_r2.simulate_r2`가 등록 코너(normal/worst/capped)에서 `e2_vessl_plan.simulate`와 같은 값을 내는지는 셀프테스트가 확인한다.

| 층 | 모형 | 컨테이너 s | GPU-h | 금액 | 20 boot 입장 |
|---|---|---|---|---|---|
| 등록 보통 | A13 §5-2 그대로(ON bench = OFF bench) | 7,168.5 | 1.991 | $2.95 | 20 |
| **보통(관측 보정)** | 1회차 A ON boot 벽시계 375 s 반영(a_r4 ON 비첫 boot 373/374/373 s; a_r2 ON은 미관측 → 같은 375 s로 **가정**), B ON은 최악 코너(×1.5) | 8,071.3 | **2.242** | **$3.32** | 20 |
| 최악 코너 | rev3 §8-1 코너(ON bench A ×2.5, B ×1.5) | 9,117.3 | 2.533 | $3.75 | 20 |
| 최악 + settle | 위 + O4 settle 최대 180 s × 20 boot(입장 판정 앞) | 12,717.3 | 3.533 | $5.23 | 20 |
| **하드캡** | 스크립트 내부 3겹, job_entry 시작부터(A13 §5-3, 불변) | 14,400 | **4.000** | **$5.92** | — |

- 관측 근거: 1회차 `TIMING.tsv`(end−admit). a_r4 ON bench duration 197.2–197.7 s, OFF 138.0–139.5 s(`DESCRIPTIVE_e2v_20261004135228.md` §1-a).
  ON boot 벽시계 373–391 s는 등록 "보통"(ON bench = OFF bench) 가정보다 길지만 등록 최악 코너(A ×2.5) 안이다.
- 셀 구간 밖 비용(하드캡이 막지 않는다): 스케줄링·이미지 pull, job_entry clone·install, 결과 반출(E2VC-3).
- **1회차 중단 Job 실지출 0.957–0.970 GPU-h($1.42–1.44)는 위 수치에 포함하지 않았다.** 두 Job 합계는 OVERRIDE R2 §5에 적는다.

---

## 6. caveat

### 6-1. 승계 (문자 그대로)

- E2VC-1…E2VC-11: `VERDICT_e2_vessl_rules_2026-10-04.md`(E2VC-1·2·5·6·7·8)와 `VERDICT_e2_vessl_rules_rev2_2026-10-04.md`(E2VC-3·4 축소판, E2VC-9·10·11)의 문언 그대로 승계한다.
  - E2VC-6(재개)은 R1에서 **발화하지 않는다**(재개 없음). 조항은 그대로 둔다.
  - E2VC-7(트리·P3 수준)은 §3-3대로 일부가 메워졌다(mamba 48·토큰 하한 강제). 토큰 수준값 대조 병기 의무는 그대로다.
- A13의 E2V-1…E2V-8과 E2C-36…39, OVERRIDE `OVERRIDE_E2_VESSL_2026-10-04.md` §2-1·§4의 문자 승계 조항은 그대로 유효하다. E2V-7(재개)은 R1에서 발화하지 않는다.

### 6-2. 신규 (A14)

- **E2V-9 (운영 수리)**: "러너 운영 수리 O1–O4는 포트·정리·대기·기록만 바꾼다. 서빙 명령줄·arm 노브·추정량·규칙은 무변이다. O4 settle는 P14 입장 판정 앞에서 boot 사이 간격을 최대 180 s 늘릴 수 있어 E2V-5(드리프트)의 시간 축이 그만큼 늘 수 있다. 결과 문서는 `SETTLE.txt`의 `settle_polls`를 boot별로 병기한다."
- **E2V-10 (P3 (ii))**: "P3는 'mamba = 48 정확 일치 + 토큰 ≥ 786,432 + 토큰 상대차 ≤ 1e-4'로 집행된다. 결과 문서는 `P3_KEYS.txt` 원 키와 `P3_VERDICTS.txt`를 그대로 싣고, 토큰 키가 둘 이상이었으면 그 값들을 적는다. 'P3가 두 arm의 KV 풀이 같음을 확인했다'고 쓰지 않는다. 이 완화는 1회차 중단 Job의 결과를 본 뒤 등록됐다."
- **E2V-11 (1회차 중단 Job의 지위, D23)**: "Job `pdmux-e2-sticky-vessl-e6f424e-20261004135228`은 10/20 boot에서 `ABORT_P3`로 끝난 **1회차 중단 Job**이다. 라벨은 0건이며 기술 통계(`DESCRIPTIVE_e2v_20261004135228.md`)만 있다. 어떤 결과 문서도 그 a_r4 4짝으로 R1–R5를 계산하지 않고, 새 Job의 짝과 합치지 않는다. 새 Job 결과 문서는 그 Job을 '1회차 중단 Job'으로 병기하며, 두 Job의 a_r4는 나란히 기술할 수만 있다(판정 아님)."

---

## 7. 1회차 중단 Job의 결과 지위 (D23)

- 라벨 **0건**. `e2_label.py`는 호출되지 않았다(러너가 `ABORT_P3`로 tail 전에 끝났다).
- 남은 것은 기술 통계뿐이다: `analysis_2026-10-04/DESCRIPTIVE_e2v_20261004135228.md`(n=4 쌍 a_r4 + 1 boot a_r2 OFF, CI·bootstrap 없음).
- 새 Job의 결과 문서는 머리말에 "1회차 중단 Job(`…20261004135228`, 10/20 boot, `ABORT_P3`, 라벨 0건)"을 병기한다(E2V-11).

## 8. 사용자 결정 기록 (2026-10-05, "231 순서로 진행")

1. **재실행 = R1**(새 20 boot Job) 승인 + **새 범위 한정 OVERRIDE**(`OVERRIDE_E2_VESSL_R2_2026-10-05.md`). 직전 OVERRIDE는 1회차 중단 Job 1건으로 소진됐다.
2. (N3 B 전 Workspace 6그룹 프로브 승인 — 별도 담당, 이 追記의 범위 밖.)
3. **P3 = (ii) 허용오차**(mamba 48 정확 일치 ∧ `max_total_num_tokens` ≥ 786,432 ∧ 상대 산포 ≤ 1e-4).

하드캡은 4.0 h로 불변이다(OVERRIDE R2 §7).

## 9. 제출 선행조건 (순서대로, 한 창 규율)

1. 규칙층 감사(claims-auditor) — 이 追記(특히 §3 사후 완화, §2-1 settle 위치, §4 재개 거부). `GO` 또는 `GO-with-caveats`.
2. 하네스층 국소 확인 — r2 러너·r2 helper·`tests/test_e2_vessl_r2.py` 봉인 sha 기록, 1회차 봉인 파일 불변 확인.
3. `presubmit.py` 재실행 → 차단이 정확히 M4R `SINGLE_LABEL_FORCED`·TC1 `RESTRICTIONS_INERT` 2건인지 확인(OVERRIDE R2 §1).
4. 로컬 셀프테스트·회귀·`check_citation_stops.py` 0위반 → **커밋**.
5. staging: 1회차가 올린 `staging/e2_selftest_archive/`(7 파일 + `SHA256SUMS`)를 재사용할 수 있다. Job이 sha를 다시 확인한다(`ABORT_ARCHIVE_DRIFT`).
6. `launch.sh --campaign e2_sticky_vessl --target workspace/engine-port/results/r2_eval/e2_sticky_vessl/e2_sticky_vessl_r2.sh` **dry-run** → 사용자 승인 → `--submit`.
7. 회수는 `fetch.sh`(DONE + sha256)까지. job name을 OVERRIDE R2 §7에 기록하고 소진 처리.

## 10. 초안 대비 변경 색인

| 위치 | 초안 | 확정본 |
|---|---|---|
| 머리말 | 초안·효력 없음 | 확정본, 사용자 결정 반영, 감사 대기 |
| §1-1 λ0 단락 | `n3_capacity.sh`를 같은 로직 목록에 포함, 줄 번호 없음 | 파일:줄 확인 목록. `lambda0.sbatch:316-321` 확인. N3는 rev2에서 수리됨. KISTI 창 겹침은 미확인 |
| §2-1 | 없음 | settle 위치(입장 앞), P3 helper 이전, mamba 48 고정, 재개 거부, 등록 확인 |
| §3 | 선택지 표 | (ii) 채택 + 전례 균형 문구(N3 rev3 D2와 같은 사실) + 조이는 방향 공시 |
| §4 | R1/R2 선택지 | R1 채택 + 사유 + 집행 |
| §5 | 관측 보정 2.24 GPU-h 서술 | 표 + `worst_settle` 행 |
| §6 | E2V-9·10 후보 | E2V-9·10·11 등록, E2VC 승계 명시 |

## 11. 로컬 검증 (CPU, GPU 0, 2026-10-05)

- `tests/test_e2_vessl_r2.py` **41/41 OK** — 구조(1회차 러너와 측정 블록 바이트 동일)·digest(1회차 봉인 불변 포함)·P3 (ii) 변이(helper + 러너 실제 P3 조각 실행)·O1 범위·O2 TIME_WAIT 변이·O3·O4 위치(초안 위치 변이 검출)·실제 경로 kill 층·재개 ABORT(가드 제거 변이는 20 boot 진행)·dry-run·집행기.
- `tests/test_e2_vessl.py`(1회차, 봉인) 무수정 재실행 **48/48 OK**.
- `e2_vessl_plan_r2.py --selftest` OK, `e2_vessl_plan.py --selftest` OK, `--verify-digests-r2` OK, `bash -n e2_sticky_vessl_r2.sh` OK, `check_citation_stops.py --file` 새 파일 5개 0위반.
- 전체 CPU 회귀(`unittest discover`, 821 실행): 실패 27·오류 14는 전부 이 追記가 건드리지 않은 모듈이다(`test_campaign_provenance`·`test_lambda_star_per_shape`·`test_r2_eval_runner` = KISTI SLURM 스크립트의 `PDMUX_ROOT` 미설정, `test_r2_correctness_ctx` = Zamba2 HF 캐시 부재, sglang 트리 import 실패 모듈 9개). 그 모듈들은 E2 VESSL 파일을 참조하지 않는다.
- 공개 이미지 컨테이너 dry-run: **수행하지 못했다**(로컬 사용자에게 docker 소켓 권한 없음).
- 한계: DRYRUN은 settle를 건너뛰고 실 포트 bind를 하지 않는다. O1의 실 기동과 O4는 GPU Job 전에는 **미검증**이다.
  - 도구 존재(같은 이미지 digest `sha256:1e0d335b…`의 1회차 중단 Job 기록): `nvidia-smi`는 있다(`SUBSTRATE_E2V.txt`의 `nvidia_smi=` 줄이 채워짐).
    `pynvml`은 import된다(서버 로그 `import pynvml` FutureWarning). `nvmlDeviceGetMemoryInfo` 호출 자체는 실행된 적이 없다(실패하면 `nvml=NA`로 기록만 한다).
  - `nvidia-smi`가 값을 못 내면 settle는 180 s를 다 기다린다(gate 아님). 예산의 최악 + settle 행이 그 경우다.
