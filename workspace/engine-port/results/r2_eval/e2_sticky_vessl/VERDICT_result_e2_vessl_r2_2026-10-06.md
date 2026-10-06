# 결과 감사: E2 sticky 대조 VESSL 재실행 Job `pdmux-e2-sticky-vessl-22c8c64-20261006020317` (claims-auditor, 2026-10-06, 읽기 전용, GPU 0)

> 감사자 최종 보고를 메인 세션이 옮겼다(문안 무편집). 감사 시점 HEAD `a4b0c2e`. 결과 문서
> `RESULT_e2_vessl_r2_2026-10-06.md`는 감사 시점에 없었으므로 원자료를 직접 감사했다 — 결과 문서는
> 이 판정서 §9 금지 문안과 대조해야 한다(메인 세션 후속 작업).

## 최종 판정

**死因 0. 등록 라벨은 기계적 산물로 `CONFIRMED`다(바이트 재현). 해석 범위는 아래 caveat로 좁힌다.**

| 항목 | 판정 |
|---|---|
| 등록 집행(봉인·순서·명령줄·P1/P2·트리·셀프테스트) | `CONFIRMED` |
| R1 `OFF_BASELINE_ABOVE_BAND` | `CONFIRMED`(기계적). Q1("sticky ON이 shape A 실현 분할을 D44로 고정한다")은 `NOT-YET-SUPPORTED`(E2VC-1) |
| R2 a_r4·a_r2 `LAMBDA_MOVES_DOWN` | `CONFIRMED`(기계적이고 강건). 기전 귀속은 `NOT-YET-SUPPORTED`(E2C-1·E2C-21) |
| R3/R4/R4′/R5 PASS, R4″ `B_RESULT_AXIS_WITHIN` | `CONFIRMED`(기계적). R4″는 "차이 없음"으로 읽지 않는다 |
| P13 `STICKY_CORRECTNESS_BYTE_IDENTICAL` | `CONFIRMED`(부하 없는 순차 6 프롬프트, D44 config 한정) |
| P3 (ii) 발화(boot 15, −4 토큰) | 사실만 `CONFIRMED`. "H2 지지"는 `PLAUSIBLE(조건부)`. H1 반증은 `NOT-YET-SUPPORTED` |

---

## 0. 독립 실측

- **HEAD**: 지시문은 `b4ec1cc`였으나 감사 중에 `a4b0c2e`(회수 커밋)가 쌓였다. 등록 바이트에 닿는 변경은 OVERRIDE R2 §7 소진 기록 追記뿐이다(A14 판정서의 봉인 예외 범위).
- **회수 무결성**: 로컬 원자료 232/232가 `vessl/…/SHA256SUMS`와 일치한다. `DONE` exit_code=0.
- **등록 봉인**: `REGISTRATION_SHA256.txt` 25행을 커밋 `22c8c64` 바이트와 대조했고 **25/25 일치**했다.
  - A14 `524a08ea…`, r2 러너 `d4a146a9…`, helper r2 `a7a6feb0…`는 A14 판정서 봉인 표와 같다.
  - `e2_label.py` `feabd5a7…`, `e2_realized_mix.py` `ecb0f6fc…`.
  - OVERRIDE R2: 제출 시점 바이트는 `144f3006…`이고, 이후 §7 追記로 현재 로컬 바이트만 달라졌다.
  - 로컬 `--verify-digests-r2` → `DIGESTS_R2 OK (A14, runner r2, 6 first-Job sealed files)`.
- **순서**: `TIMING.tsv` 20행이 rev3 §2 리터럴(a_r4 s1–s4, a_r2 s1–s4, b_r3 s1–s2, 매번 OFF→ON, s=4386/4162/251/2630)과 같다. 20/20 `bench_rc=0`, `kill_kind=none`.
- **명령줄**: `server_args`를 port만 가리고 비교하면 20/20 동일하다. 확인한 값: mem 0.82, max_running 48, radix off, flashinfer, cudagraph ON, context 16384, seed 1.
- **P1/P2**: `sticky partition ENABLED … fixed target index=2`는 ON 10/10에 있고 OFF 10/10에 없다. 서버 로그에 Traceback·Errno 98 0건.
- **트리·셀프테스트**: `ENGINE_TREE AS_REGISTERED`(1파일 차이, 등록됨). 셀프테스트는 estimators 7/7, rules 8+10, port r2 OK, sticky unit 12/12, `ARCHIVE OK`, `SUBSTRATE OK`.
- **유효성**: P9–P12 `all_pass` 20/20, E-pact 20/20 = 100%, `sm_index_mismatch` 0, `negative_gaps` 0. CAPPED·BUDGET_STOP·BOOT_FAILED 파일 없음. `E2V_DONE … remaining_s=6770`.

## 1. 라벨 재산출

- **e2_label.py**: `e2_label.py --cells-dir <OUT> --expect "$(e2_vessl_plan.py --expect)" --out …`로 다시 돌렸다. 결과는 `E2_LABEL.json`과 **바이트 동일**이다(sha `ce7a3c12…`).
  - **R-1**: 지시문의 `--repeat` 인자는 존재하지 않는다. 등록 호출은 `--cells-dir --expect --out` 셋뿐이다(러너 L613).
- **e2_realized_mix.py**: 20 boot 전부를 등록 인자(probe_end 2건 포함)로 다시 계산했다. **판정 필드는 20/20 일치**한다(E-cnt/E-time/E-iter/E-qcond/E-pact 분자·분모, idx0/idx3, hist, split_transition, busy_n, bench 전체).
  - 다른 필드는 `snapshots`(20/20), `dropped_by_probe_filter`·`telemetry_span`(probe 2셀)뿐이다.
  - 원인: 러너가 추정량을 teardown **전**에 부른다. 그때 `compute()`가 계속 커지는 텔레메트리 파일을 두 번 읽는다. 그래서 이 필드들은 경쟁 상태(race)의 산물이다 → **E2VR-4**.
  - 최종 파일에서 판정 필드가 불변이므로 판정 영향은 없다.
- **R2 bootstrap**: 256개 순서 있는 재표본을 전수 열거해 2.5/97.5 분위를 구했다. 결과가 등록 CI와 1e-8까지 같다. 재표본 multiset은 35개다.

## 2. R1 `OFF_BASELINE_ABOVE_BAND`

- **OFF 절 실패는 E-qcond 하나에서만 났다.**
  - a_r2/251: 21.51% (20/93)
  - a_r2/2630: 25.93% (14/54)
  - 같은 셀의 E-cnt/E-time/E-iter는 8.2–12.1%로 문턱 20.0에서 멀다.
- **ON 절은 8쌍 전부 성립했다.** 최솟값은 E-cnt 99.64%(a_r4/2630, 551/553)이고 E-qcond는 전부 100%다. 이것은 엔진이 {0,2}로 강제한 양성 대조다(E2C-3). Q1의 답이 아니다(E2VC-1).
- **A5 4분지 적용**: OFF 절만 거짓이므로 라벨은 대조 arm에 대한 진술이다. ON arm에 대한 진술이 아니다.
- **E2VC-5**: 감사자 예보 ≈0.95는 비등록 수치다. "예보대로 나왔다"는 금지다.
- **산포(기술만)**: a_r2 OFF E-qcond는 18.18/19.51/21.51/25.93%(분모 132/82/93/54, 폭 7.74pp)다. a_r4 OFF는 16.73–17.46%(분모 251–255, 폭 0.73pp)다.
  - a_r2의 분모가 작고 들쭉날쭉한 것은 "queue>0 ∧ decode-busy"가 knee 아래에서 드물기 때문이다. 1표본 = 1.85pp(분모 54)다.
  - 같은 seed 4386 a_r2 OFF E-qcond를 네 번 측정한 값: KISTI 908623 17.65 / λ0 VESSL 20.57 / 1회차 20.13 / 이번 18.18. 문턱 20.0을 가로지른다(Job·GPU 축 교락, #11).
  - 그래서 a_r2 OFF 절의 성패는 **seed 단위로 재현되는 사실이 아니다**. 이 Job에 한한 기계적 결과다 → **E2VR-2**.
  - **R5는 E-cnt 산포만 본다.** R1을 실패시킨 추정량(E-qcond)의 seed 산포는 R5가 재지 않는다.

## 3. R2 `LAMBDA_MOVES_DOWN`

| 셀 | d_i/OFF (seed 4386/4162/251/2630) | d̄ (req/s) | 등록 CI | 참고: paired-t(3) CI |
|---|---|---|---|---|
| a_r4 | −29.83 / −30.01 / −30.33 / −30.23% | −0.905 | [−0.912, −0.898] | [−0.918, −0.892] |
| a_r2 | −25.91 / −24.94 / −24.63 / −24.09% | −0.689 | [−0.714, −0.669] | [−0.732, −0.646] |

**E2VC-13 필수 병기**:
- 1회차 중단 Job a_r4 d_i/OFF는 −29.97 / −29.25 / −30.09 / −30.07%(범위 −29.25…−30.09, 4/4 음수, 라벨 없음, 재계산으로 확인)다. 이번 Job은 −29.83…−30.33%다.
- 방향은 같고 범위는 겹친다. 그러나 이것은 **나란히 기술하는 것일 뿐**이다. "재현/대체/정정"이라고 쓰지 않는다(E2V-11).
- 수준값은 Job 간에 움직였다.
  - OFF a_r4 achieved 평균: 1회차 2.888 → 이번 3.006(**+4.1%**).
  - ON: 2.026 → 2.101(**+3.6%**).
  - 다른 GPU(`5174d21c` vs `a66f0fff`)이고 엔진 측정 코드 diff는 0이다. Job 안 seed SD ≤0.5%의 약 8배다. 원인 귀속은 불가하다(이번 Job이 power cap 노출이 더 큰데도 더 빠르다) → **E2VR-3**.

**필수 병기 사항**:
- **E2C-1**: 기전이 둘이다(decode SM 108→44, drain 쌍 제거). 분할 단독 귀속 금지.
- **E2C-21**: 처치 시간의 86–89%가 prefill 유휴 구간이다. −30%는 대부분 "멀티플렉스할 것이 없는 동안 decode를 44 SM에 묶어 둔 비용"이다.
- **E2C-10**: achieved는 48 구속값이다. 라벨 이름의 λ는 시스템 용량이 아니다.
  - ON achieved는 a_r4 2.096–2.105, a_r2 2.072–2.086으로 사실상 같다. 즉 ON은 두 셀 모두 용량 구속이다.
  - a_r2 OFF(2.75–2.80)는 실현 offered 2.95–3.00 근방 knee에 있다. 그래서 a_r2와 a_r4의 |d| 차이를 용량-반응 관계로 읽으면 안 된다. a_r2는 부호만 읽을 수 있다 → **E2VR-8**.
- **E2C-7**: OFF→ON 순서 비대칭. 이것이 부호를 만들 수 없다는 근거:
  - a_r4 OFF가 시간에 따라 +0.5% 드리프트했다(2.9998→3.0140). 30% 효과의 1/60이다.
  - bench 창 SwPowerCap은 OFF 22.7–26.7%, ON 12.6–21.3%였다. 평균 SM MHz는 OFF 1402–1406, ON 1406–1408(≈0.2%)로 **d와 반대 방향**이다.
- **E2C-13**: seed가 실현 offered를 바꾼다. paired 차분 안에서는 상쇄된다.
- **E2VC-2(CPU)**: bench 창 Δnr_throttled는 OFF 3–6 / ON 4–6, Δthrottled_usec는 OFF 14.5–21.2 s / ON 13.5–20.8 s다. CPU 사용은 OFF 1.32–1.35, ON 1.25–1.27 CPU-s/s다. arm 내 산포를 넘는 비대칭은 없다. 단위 관계(nr_throttled 5회 ↔ 15–20 s)는 미확인이다.
- **E2C-17/E2VR-5**: `distinct_resamples 256`은 순서 있는 재표본 수다(multiset 35).
  - n=4에서 네 d가 모두 같은 부호이면 percentile CI<0과 "4/4 음수"는 같은 말이다. 정확 부호검정(단측)은 1/16 = 0.0625다.
  - 강건성의 근거는 bootstrap의 명목 95%가 아니다. 효과(−30% / −25%)와 arm 내 산포(SD 0.008 / 0.027 req/s)의 비다. paired-t CI도 0을 배제한다. 3% 게이트를 훨씬 넘는다.

## 4. R3·R4·R4′·R4″·R5

- **R3**: OFF idx3 = 0, 10/10 셀. idx0 최대 2/555 = 0.36%. ON의 split_transition이 OFF보다 큰 셀은 0개다(b_r3 ON 99 vs OFF 100).
- **R4**: b_r3 E-qcond 100 vs 100, Δ 0.0pp. 처치가 움직일 수 없는 양이라 구조적으로 거의 보장된 통과다. "순서·드리프트가 없음을 증명했다"가 아니라 "R1/R2 caveat 미발화"로만 쓴다.
- **R4′**: b_r3 ON E-cnt 100%(441/441, idx0 0/441).
- **R4″**: Y = 0.000531, |d̄| = 0.000380 < 3Y = 0.001592 → WITHIN. b_r3 d/OFF는 −0.03 / −0.08%다.
  - n=2, CI 없음, 검출력 미등록. "차이 없음" 금지(E2C-18).
  - shape A와 같은 문장에 쓰지 않는다(E2C-21c).
- **R5**: E-cnt 산포(pp)는 a_r4 OFF 0.84 / ON 0.36, a_r2 OFF 0.77 / ON 0.18이다. 전부 ≤5pp로 PASS.

## 5. P3 (ii) 발화

**사실**:
- boot 15 `a_r2_OFF_2630`만 `max_total_num_tokens` 2,722,021(−4, rel 1.469e-6)이다. 나머지 19 boot는 2,722,025다.
- 직전 14 boot는 전부 정상 종료(teardown 17–18 s)였다.
- settle는 21회 전부 `settle_polls=1`, `sglang_alive=0`이었다. NVML free 85,094,825,984 B와 used 804,519,936 B가 **21회 바이트 동일**했다.
- `Load weight begin avail mem=78.70 GB`가 20/20 같다.
- 다음 boot(16)은 다시 …025로 돌아왔다. 즉 지속성이 없다.

**H1 vs H2**:
- 이 이탈은 H1의 전제(직전 비정상 종료)가 **없는** 상태에서 일어났다. 그러므로 "이 현상은 비정상 종료가 필요하다"에 대한 **반례 1건**이다. H2(또는 명명되지 않은 H3)와 양립한다.
- H2는 잔여 범주다. 그래서 "H2 확정"이 아니다. 1회차 이탈이 H1 때문이었을 가능성도 배제하지 못한다(원인이 다를 수 있다).
- **NVML 해상도**: 값은 64 KiB로 나누어떨어지고 1 MiB로는 안 된다. −4 토큰은 49,153–65,536 B, 즉 64 KiB 양자 하나 이하다. 게다가 프로세스 안에서 컨텍스트를 만든 뒤 생기는 차이는 유휴 NVML로 볼 수 없다. 그러므로 **NVML 기록은 판별력이 검증되지 않았다**(E2VC-14 유지). 해상한 것은 "64 KiB 이상의 Job 수준 유휴 오프셋은 없었다"까지다.

**E2VC-12**:
- 토큰 키는 arm을 가르지 않는다. OFF는 {…025×9, …021×1}, ON은 {…025×10}이다.
- 두 Job 합산 이탈 2/2가 a_r2 OFF boot였다. n=2라 정보가 없다(우연 확률 약 1/4).
- **우연한 일치(기술만)**: 이탈 boot는 a_r2 OFF E-qcond 최댓값(25.93%)과 a_r2 |d| 최솟값(−0.662)을 낸 boot다. 그러나 achieved 2.7478은 같은 실현 offered의 seed 251(2.7554)과 −0.28%로 비이상적이지 않다. 토큰 3–4개가 achieved에 닿는 코드 경로는 A14 감사에서 찾지 못했다.

**E2VC-14/15 문언**: "O1–O4가 해결했다"와 "O4가 H1을 막았다"는 금지다. 1e-4는 사용자 결정 상수다.

## 6. P13

- `greedy_OFF.json`과 `greedy_ON.json`의 `texts` 6개가 동일하다(빈 응답 0, 길이 200/74/225/158/117/99).
- 라벨 `STICKY_CORRECTNESS_BYTE_IDENTICAL`.
- 범위: 첫 쌍(a_r4 4386) 부하 없는 순차 프로브, D44(index 2)·5그룹 config 한정이다.

## 7. 운영·기판

- **포트**: 20/20 `tries=1`, 창 20000–20799, ephemeral 범위는 32768–60999다. BOOT_FAILED 0, `BOOT_FAILURES.txt` 없음. 위험은 0이 아니다(E2VC-15).
- **시간**:
  - 컨테이너 `timing.txt` 02:03:44→04:00:54 = 1.953 h(E2VC-3 실측 충족)이다. 보통 예산 2.242 h와 하드캡 4.0 h 아래다.
  - VESSL Job 1h57m42s(스케줄링 포함).
  - 루프 종료 시 잔여 6,775 s.
- **per-boot 부대비용(boot 벽시계 − bench)**: A OFF 평균 144 s(첫 boot 206 s는 cold), A ON 154 s, B 94–95 s. 등록값 A 160.1 / B 121.0 s 아래다.
- **throttle**: Job 전체 clk 11,871/69,961 = 17.0%(VC-2 병기 의무). bench 창 표는 §3. 0x4 외 비트는 bench 창에서 0건이다.
  - cgroup CPU quota 11(nproc 96).
  - Job 전체 nr_throttled 555/68,390 period, throttled_usec 1,957.7 s.
- **GPU UUID**: 이번 `GPU-a66f0fff…`, 1회차 `GPU-5174d21c…`, λ0 VESSL·B5 `GPU-3be4c7da…`로 **세 Job 모두 다른 GPU**다. 드라이버 580.105.08은 같다. Job 간 대조는 전부 #11 교락을 진다.

## 8. 반전 시험 표

원자료는 이 Job의 `cell_*.json`·`E2_LABEL.json`이고, 판정 함수는 등록 `e2_label.py`의 규칙 그대로다. 새 격자나 추정량은 만들지 않았다.

| 자유 표면 | 민 범위 | 판정 변화 | 근거 수치 |
|---|---|---|---|
| P3 (ii) ↔ rev3 리터럴 (i) | (ii)→(i) | **라벨 존재가 뒤집힌다**: (i)이면 boot 15에서 `ABORT_P3`, 라벨 0건(D23) | keys {…025, …021}. **실행 전 등록(A14, 감사 GO-with-caveats)이라 死因 아님** → E2VR-1 |
| 이탈 boot 제외(등록이 허용하지 않음) | 포함↔제외 | R1 불변(251이 21.51로 여전히 실패). R2 a_r2: n=3 → `UNRESOLVED`(부호 3/3 음수, mean −0.698). a_r4·b_r3 불변 | 금지 경로, 기록만 |
| R1 추정량 집합(E-qcond 제외는 금지된 재스코어) | 4→3 | **반전**: OFF ≤13.8%가 되어 `STICKY_REALIZES_A`가 된다 | 금지(rev3 §5-1, E2C-11, #6) → 금지 문안 |
| bench-only 재스코어(E2C-16 금지) | 등록↔bench창 | OFF 값이 오르는 방향이라 OFF 실패만 강화된다 | E2C-16 수치(a_r4 E-cnt 8.65→16.33) |
| R2 CI 방법 | percentile↔paired-t | 불변(두 방법 모두 상한 <0) | t-CI 상한 −0.892 / −0.646 |
| 순서·클럭 드리프트 | OFF 선행 | 부호를 만들 수 없다(클럭 차 0.2%가 d와 반대 방향, OFF 드리프트 0.5%) | §3 |
| R4″ 문턱 | 3Y(등록) | 1Y로 조여도 \|d̄\| 0.00038 < Y 0.00053으로 WITHIN 불변 | — |

반전 행은 둘이다. 둘 다 등록이 금지했거나 실행 전에 등록된 표면이다 → 死因 0, caveat로 전환한다.

## 9. 정본 반영 문안

**허용 문안(그대로 쓸 것)**:
> "E2 sticky 대조 VESSL 재실행 Job `pdmux-e2-sticky-vessl-22c8c64-20261006020317`(A100 `GPU-a66f0fff`, 20/20 boot, 등록대로 집행, 라벨 바이트 재현)은 등록 라벨 R1 = `OFF_BASELINE_ABOVE_BAND`를 냈다(대조 arm 진술: a_r2/251 E-qcond 21.51% [20/93], a_r2/2630 25.93% [14/54]; ON 절은 양성 대조로 성립, Q1 미확립). R2는 a_r4·a_r2 모두 `LAMBDA_MOVES_DOWN`이다(a_r4 d̄ −0.905 req/s, CI[−0.912, −0.898], d/OFF −29.8…−30.3%; a_r2 d̄ −0.689, CI[−0.714, −0.669], −24.1…−25.9%; n=4, `--max-running-requests 48` 구속 achieved, 정지 rate 사다리라 정책 비교 아님). R3/R4/R4′/R5는 PASS, R4″는 `B_RESULT_AXIS_WITHIN`(n=2, 차이 없음 아님), P13은 byte-identical, UNRESOLVED 0이다. 1회차 중단 Job(`…20261004135228`, 10/20 boot, `ABORT_P3`, 라벨 0건) a_r4 d/OFF −29.25…−30.09%(4/4 음수)를 병기한다."

**필수 병기**:
- E2C-1, E2C-21(a)(b)(c), E2C-10, E2C-7, E2C-13, E2C-17, E2C-18
- E2VC-1, E2VC-2(CPU 표), E2VC-7/16(배너 19×…025 + 1×…021)
- E2VC-12, E2VC-13, E2VC-14, E2VC-15, E2V-10 원 키
- 아래 E2VR-1…8

**금지 문안**:
1. "sticky ON이 shape A의 실현 분할을 D44로 고정/실현했다(Q1 확립)." / "ON ≈100% ⇒ sticky 효과 실증"
2. "VESSL OFF가 KISTI OFF보다 D44를 더 점유한다." / "sticky가 무의미하다."
3. "E-qcond를 빼면(또는 bench-only면) `STICKY_REALIZES_A`."
4. "sticky는 처리량을 30% 깎는다 / sticky는 나쁘다·좋다." / "D44 분할의 비용은 30%."
5. "fixed-D44 용량(λ\*) ≈ 2.1 req/s." / "LAMBDA_MOVES_DOWN = 시스템 용량 하락."
6. "a_r2와 a_r4의 |d| 차이는 부하 의존성(용량-반응)이다."
7. "shape B에서는 sticky 영향이 없다." / shape A와 B를 같은 문장에서 비교하는 것
8. "이번 Job이 1회차를 재현·대체·정정했다."
9. "E2가 λ0R-8(iii)을 해소/약화했다."
10. "P3가 두 arm의 KV 풀이 같음을 확인했다." / "O1–O4(O4)가 −3 토큰·H1을 해결/배제했다." / "H2 확정", "H1 반증"
11. "포트 충돌 위험이 제거됐다."
12. "R4 PASS ⇒ 순서·드리프트가 없다."
13. "R5 PASS ⇒ R1 추정량들이 seed 간 재현된다."
14. VESSL Job 간 또는 KISTI와의 수준값 비교·결합(R6는 판정 아님)
15. `cell_*.json`의 `snapshots`·`dropped_by_probe_filter`·`telemetry_span` 인용
16. "P13 ⇒ 부하 중 또는 다른 분할에서 sticky 정확성."

**불변 선언**: 게이트 #6(λ\*\_SLO / λ\*\_VESSL 정규화 격자 미등록), λ0R-8(iii)(E2C-5: E2는 필요조건일 뿐 해소하지 않음), HE0, layer-type 死, 정책 순위, Claim D/E(미검증), P2 블로커 ②③, 게이트 #13/#16, C2 인용정지 모두 **불변**이다. 새 성능 판정 0건.

**N2(3-a)가 가져갈 수 있는 것**:
- (a) 이 기판·이 트리·`pdmux_homog5`(5그룹, D44 = index 2)에서 `PDMUX_STICKY_PARTITION=1`이 서빙 중 배선되고 유지된다는 **양성 대조 통과**. P1 10/10 대 0/10, ON decode-busy idx2: E-cnt ≥99.64%, E-iter ≥99.64%, E-time ≥99.79%, E-qcond 100%. 세 셀, 10 boot.
- (b) 부하 없는 greedy 6건 byte-identical.
- (c) r2 러너 O1/O2 하에서 20 boot 포트 재시도 0과 P3 (ii) 실가동 사례(운영 사실).

**N2가 가져갈 수 없는 것**:
- index 2 / D44 밖({16,24,34}), 6그룹 config(토큰 2,721,290)로의 일반화
- "≥99.6%"를 엔진 강제 집합 {0,2} 밖의 구속 충실도 추정치로 쓰는 것
- 부하 중 정확성
- sticky-ON achieved를 D별 용량이나 프로파일 값으로 쓰는 것(E2C-1/10/21)
- 단독 vs 동거 step 시간(DS-3)
- 수준값의 Job 간 이식(E2VR-3)

## 10. 정정 R-n

- **R-1**: `e2_label.py`에 `--repeat` 인자는 없다. 등록 호출은 `--cells-dir --expect --out`이다.
- **R-2**: 감사 기준 HEAD는 `a4b0c2e`다(지시문 `b4ec1cc` 이후 회수 커밋).
- **R-3**: Job이 집행한 1회차 봉인은 `SEALED_R1` **6파일**이다(OVERRIDE `131abfff` 포함). 지시문의 "봉인 7 + OVERRIDE"와 셈법이 다르다. 값 불일치는 없다.

## 11. 신규 caveat (인용 금지 문장 형태, 결과 문서·정본 문자 승계)

- **E2VR-1 (라벨 존재의 P3 (ii) 의존)**: "이 Job의 라벨 존재 자체가 P3 (ii)에 의존한다. rev3 리터럴 P3였다면 boot 15 `a_r2_OFF_2630`(2,722,021)에서 `ABORT_P3`로 라벨 0건이었다. (ii)는 1회차 결과를 본 뒤, 이 Job 실행 전에 등록됐다. 이탈 boot를 빼도 R1은 불변이고 R2 a_r2는 n=3 `UNRESOLVED`가 된다(부호 불변). 이 사실을 R2 a_r2 라벨 옆에 적는다."
- **E2VR-2 (R1 OFF 절의 문턱 근접성)**: "R1 OFF 실패는 E-qcond 하나에서만, 작은 분모(54·93)로 났다. 문턱 아래로 내려가려면 각각 4개·2개 표본이 부족했다. 같은 seed 4386 a_r2 OFF E-qcond는 네 측정(세 기판·Job)에서 17.65–20.57%로 20.0을 가로지른다. `OFF_BASELINE_ABOVE_BAND`는 이 Job의 기계적 결과이며, seed나 기판의 속성으로 인용하지 않는다. R5(E-cnt 산포)는 E-qcond 재현성을 보증하지 않는다(a_r2 OFF E-qcond 폭 7.74pp)."
- **E2VR-3 (VESSL Job 간 수준 이동)**: "같은 셀·seed·엔진 측정 코드에서 두 VESSL Job(다른 GPU)의 a_r4 achieved 수준은 OFF +4.1%, ON +3.6% 움직였다. Job 안 seed SD(≤0.5%)의 약 8배다. 원인 귀속은 불가하다. Job 내 paired 차분만 인용하고, 수준값과 Job 간 결합은 인용하지 않는다(LVR-4·#11 계열)."
- **E2VR-4 (cell JSON 장부 필드)**: "`cell_*.json`의 `snapshots`·`dropped_by_probe_filter`·`telemetry_span`은 추정량이 teardown 전에 커지는 텔레메트리 파일을 두 번 읽은 산물이라 인용 금지다. 판정 필드는 최종 파일 재계산과 20/20 일치한다."
- **E2VR-5 (n=4 CI의 의미)**: "n=4 percentile CI<0은 여기서 '4/4 음수'와 같다(정확 부호검정 단측 1/16). 라벨의 강건성 근거는 효과 크기 대 arm 내 산포다(paired-t CI 상한 −0.892 / −0.646). 'p<0.05 유의'로 쓰지 않는다."
- **E2VR-6 (N2 승계 범위)**: §9의 가져갈 수 있는 것 / 없는 것 목록을 문자 그대로 승계한다.
- **E2VR-7 (H1/H2 문언)**: "이 Job의 −4 토큰 이탈은 직전 비정상 종료 없이 일어났다. 이것은 H1이 필요조건이 아니라는 반례 1건이다. H2 확정이나 1회차 H1 반증이 아니다. NVML 유휴 기록(64 KiB 양자, 21회 바이트 동일)은 그 이탈을 해상했다고 보증하지 않는다."
- **E2VR-8 (a_r2 크기 해석)**: "ON achieved는 a_r2와 a_r4에서 2.07–2.10 req/s로 같다(용량 구속). a_r2 OFF는 knee 근방(실현 offered 2.95–3.00 대비 2.75–2.80)이다. a_r2 d의 크기는 knee 위치와 섞이므로 부호만 해석하고, a_r4와의 크기 비교는 하지 않는다."

## 12. 반증 실패 기록

봉인 25/25 · 순서 20/20 · server_args 20/20 · P1/P2 10/10 대 0/10 · 회수 sha 232/232 · `E2_LABEL.json` 바이트 재현 · 추정량 판정 필드 20/20 · bootstrap 전수 열거 일치 · paired-t 교차 · 클럭·CPU·드리프트가 부호를 만들 수 없음 · 이탈 boot 민감도 · R4″ 문턱 조임 · 1회차 a_r4 재계산(−29.25…−30.09).

라벨을 깨는 경로를 찾지 못했다. 단 Q1(sticky의 D44 실현)과 R2의 기전 귀속은 아래 실험 전에는 `CONFIRMED`가 아니다.

- **Q1**: OFF 문턱 판정에 쓸 E-qcond 분모를 확보하는 a_r2 대체 셀(≥250 표본/seed)과 n≥4 seed를 새로 등록해야 한다. 같은 Job·같은 GPU에서 측정한다.
- **기전 분리(E2C-1)**: decode SM을 108로 둔 채 drain 쌍만 제거하는 arm은 이 엔진에서 지원되지 않는다(엔진 강제, 교훈 88). 그래서 현재 처방할 수 없다고 명시한다.
- **E2VR-3**: 같은 셀을 Job ≥3·GPU ≥3에서 반복해 Job 축 σ를 측정해야 한다. 사용자 결정 사항이다.

## 13. 실현가능성 자기검사(게이트 #113)

- 이 판정서의 처방은 문서 문안과 병기뿐이다. 명령줄·arm·규칙·봉인 바이트 변경 0건이다.
- E2VR 근거 파일은 전부 이 Job 디렉터리에 있다.
- §12의 Q1 실험은 새 사전등록과 규칙층 감사가 필요하다(이번 등록을 고치지 않는다).
- 기전 분리 arm은 "처방 불가"로 명시했다.
- 반전 계산에 새 격자나 추정량을 도입하지 않았다(paired-t는 참고 열이며 판정에 쓰지 않음).
- 재감사 규율(#110): A14 판정서의 봉인 sha·E2VC 문언은 원 파일·커밋 바이트로 독립 재확인했다.

## 관련 경로

- 원자료: `results/r2_eval/e2_sticky_vessl/e2v_pdmux-e2-sticky-vessl-22c8c64-20261006020317/`
- 메타: `results/e2_sticky_vessl/vessl/pdmux-e2-sticky-vessl-22c8c64-20261006020317/`
- 등록·판정서: `results/r2_eval/e2_sticky_vessl/{PREREG_E2_VESSL_A14_2026-10-05.md, VERDICT_e2_vessl_a14_rules_2026-10-05.md, VERDICT_e2_vessl_rules_2026-10-04.md, VERDICT_e2_vessl_rules_rev2_2026-10-04.md, analysis_2026-10-04/DESCRIPTIVE_e2v_20261004135228.md}`
- 규칙: `results/r2_eval/e2_sticky_prereg/{PREREG_E2_STICKY_REV3_2026-09-15.md, PREREG_E2_STICKY_REV3_ADDENDUM_2026-09-15.md, e2_label.py, e2_realized_mix.py}`
- 결과 문서(감사 뒤 작성됨, 이 판정서 §9와 대조 필요): `RESULT_e2_vessl_r2_2026-10-06.md`, `analysis_2026-10-06/e2v_r2_result.{py,json}`
