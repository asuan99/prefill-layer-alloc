# PREREG STOCK-BASE rev3.2 (2026-11-04 재봉인, `HARNESS_DIGEST.txt`; rev0 2026-11-01 → rev1 2026-11-02 → rev2 2026-11-02 → rev2.1 2026-11-02 봉인 → rev3 2026-11-03 → rev3.1 2026-11-04 → rev3.2 2026-11-04) — 기본 서빙 구조(stock SGLang · stock vLLM) 대 static PD-mux, VESSL·Nano (2026-11-01)

> 지위: **rev3.2 봉인(2026-11-04)** — rev3 감사 2회(상한) 뒤 사용자 결정으로 추가 감사 없이 재봉인. 봉인 뒤 변경은 새 rev + 재봉인. (rev3 초안 2026-11-03은 하네스 제한 감사 rev3 r1에서 NO-GO, rev3.1은 재감사 rev3 r2에서 NO-GO — 아래 rev3.1 → rev3.2·rev3 → rev3.1 요약). rev2.1은 2026-11-02에 봉인됐고(`HARNESS_DIGEST.txt` sha `9b4a30f9…`, 이 문서 sha `b2126bd9…`) 첫 번째 Job은 그 봉인 바이트로 돌았다. rev3는 첫 번째 Job 결과(`RESULT_phaseB_J1_2026-11-03.md`)로 등록 규칙상 두 번째 Job 계획이 만들어지지 않게 된 것(`ABORT_NO_ADDENDUM`)에 대한 사용자 결정(2026-11-03, 선택지 (C) = (A) 진행)을 반영한 새 rev다. 재봉인 = 이 바이트로 새 `HARNESS_DIGEST.txt`를 만든 것(2026-11-04). 남은 것 = 아래쪽 사다리 Job·두 번째 Job dry-run → 제출 승인; 세 번째·네 번째 Job은 두 번째 Job 결과 뒤 별도 결정(재감사 r2 예보: Q4는 BOTH_SATURATED, 정본 승격 goodput 라벨 사실상 0). 파일 이름은 하네스 참조(`sb_spec.PREREG`) 때문에 rev1 그대로 둔다. 사전 예측 문서(`PREDICTION_phaseB_expected_effects_2026-11-02.md`)는 rev1 그대로 두고 rev3 변경은 이 문서에만 적는다.
>
> **rev3.1 → rev3.2 변경 요약(2026-11-04, 사용자 결정 "진행" = 권고 1; 근거 `AUDIT_harness_stock_base_rev3_r2_2026-11-04.md`; 전부 CPU, 변이 테스트 포함; 규칙 변경은 아래 2의 W3 셀 범위뿐)**
> 1. **셀 분리 변이 가드**: 판정 셀 축약(`sb_analyze.cell_runs`, `cell == c ∧ boot_role == "meas"`)의 셀 조건을 지워도 테스트가 통과하던 구멍을 막는다. 50 %·80 % 두 셀을 같은 Job·같은 arm·같은 seed(1–4)로 만들고 두 셀의 goodput을 다르게 둔 픽스처로 테스트 1개를 추가했다(셀 조건을 지우면 50 % 셀이 80 % run으로 덮여 실패). 같은 테스트에 기준 셀·계측 run을 같은 셀 이름으로 넣은 픽스처를 두어 `boot_role` 조건 제거도 잡는다. 변이 2개 추가(셀 조건 제거 · `boot_role` 조건 제거). 판정 규칙 변경 없음.
> 2. **네 번째 Job W3 셀에서 `PM-rep` 제거**(§2·§3·§4·§7): R3 arm은 rev2.1의 4개(SG-stock·SG-rep·PM-ref·VL-stock)로 돌아간다. 근거(감사): Q7 키는 Q2·Q3뿐이라 R3의 `PM-rep`은 무효 조건만 더하고(+43.4 min) Q7 도달 확률을 낮춘다. 계획 검사의 `PM-rep` 존재 검사는 50 %·80 % 셀(R1·R2)에만 적용한다. 따라서 **R3에서는 PM 플라시보를 요구하지 않는다** — R3의 Q2 쌍(Q7 키)은 SG 플라시보만 따른다(rev3과 같음; 요구하면 `PM-rep`이 없는 R3의 Q2는 fail-closed로 항상 `BOOT_NOISE`가 된다). 하네스: `sb_spec`(`W3_ARMS`·`PLACEBO_PM_CELLS = (R1, R2)`·`pair_placebos(a, b, cell)`·`plan_problems`), `sb_analyze`(R3에서 `placebo_pm_invalid` = None), `sb_job.sh`(머리 주석). 변이 1개 추가(R3에서 PM 플라시보 요구), 기존 변이 1개 패턴 갱신(`PM-rep` 존재 검사). 네 번째 Job 시간 모형 297.9 → **254.5 min**(하드캡 375 → 320).
> 3. §9에 재감사 rev3 r2의 인용 금지 문장 16과 검출력 공시를 문자 그대로 추가. `sb_digest.SEALED_DOCS`에 재감사 rev3 r2 문서 추가. §13에 rev3.2 결정 한 줄. 첫 번째 Job·아래쪽 사다리·두 번째·세 번째 Job plan은 rev3.1과 같다.
>
> **rev3 → rev3.1 변경 요약(2026-11-04, 사용자 결정 "진행" = 선택지 (b); 근거 `AUDIT_harness_stock_base_rev3_r1_2026-11-04.md`; 전부 CPU, 변이 테스트 포함)**
> 1. **§8 예측 행 Q1-SG 삭제**(감사 처방 그대로): 띠 [2.5, 4.8]에는 거짓이 될 결과가 없었다(아래쪽 사다리 rung 4개가 전부 띠 안). 띠는 다시 고르지 않는다. Q1(손대지 않은 SGLang 버스트 λ\*)은 기술값으로만 보고한다(결과 표·"측정하지 않은 것"). 하네스: `SP.Q1_SG_BAND`·`score_predictions`의 Q1-SG 블록·관련 테스트 삭제.
> 2. **Q2를 부호 전용에 포함**(§0·§3(나)·§4·§9·§10): 부호 전용 질문 = Q2·Q3·Q5·Q8(균일). Q2에도 지연 공동 지표(TPOT·TTFT·요청 내부 ITL p95)를 등록하고 태그 `MAGNITUDE_NOT_CITABLE`을 붙인다. 등록 goodput 라벨(3 % 여유·EQUIV 포함)을 내는 쌍 질문은 Q4(PD-mux 대 vLLM)뿐이다. 근거: Q2의 SG arm도 같은 절대 rate에서 자기 ITL 용량 위일 수 있고(감사: rev3의 "Q2만 일반 라벨"은 비대칭·자기 모순), Q2 ln 비 크기는 ITL 축 절벽 값이다(ITL SLO +10 %에서 5.516 req/s의 ln(PM/SG) 1.507 → 0.029).
> 3. **PD-mux 플라시보 arm `PM-rep`**(§2·§4·§7): `PM-ref`와 같은 argv·env·트리, 별도 boot. 세 번째 Job의 50 %·80 % 셀과 네 번째 Job W3 셀에서만 돈다(사다리·지속 셀 없음). **플라시보 무효 규칙(K3)을 쌍의 base arm별로 바꾼다**: SG 계열 arm(SG-stock·SG-rep·SG-mix·SG-off)이 낀 쌍은 SG-rep 대 SG-stock, PD-mux 계열 arm(PM-ref·PM-ref34·PM-rep)이 낀 쌍은 PM-rep 대 PM-ref 플라시보를 따른다(VL-stock은 플라시보 없음). 따라서 Q3·Q5·Q8 = SG 플라시보, Q4 = PM 플라시보, Q2 = 둘 다. 한 셀에서 SG 플라시보가 무효(`BOOT_NOISE`)여도 PM 플라시보가 유효하면 Q4는 살아남는다. 하네스: `sb_spec`(`PM-rep` arm·`REPLICA_OF`·`PLACEBO_PM`·`PLACEBO_FAMILY`·`pair_placebos`·`plan_problems`의 PM-rep 존재 검사와 복제 바이트 검사), `sb_analyze`(쌍별 플라시보), `sb_job.sh`(머리 주석). 변이 5개 추가(PM 플라시보 생략 · Q4가 SG 플라시보에 묶임 · PM-rep argv가 PM-ref와 다름 · PM-rep 존재 검사 제거 · Q2가 부호 전용에서 빠짐), 기존 변이 3개 패턴 갱신(지연 지표·부호 전용 집합).
> 4. §13에 rev3.1 결정 한 줄. 첫 번째 Job·아래쪽 사다리·두 번째 Job plan은 rev3과 같다(PM-rep은 그 Job들에 없다).
>
> **rev2.1 → rev3 변경 요약(2026-11-03, 사용자 결정 (C) = (A); 근거 `RESULT_phaseB_J1_2026-11-03.md`·`DIAG_sglang_default_itl_2026-11-03.md`·`DIAGBOOT_itl_gap_2026-11-03.md`; 전부 CPU, 변이 테스트 포함)**
> 1. **기준 용량 재정의**(§3(나)): 기준 용량 = {vLLM(`VL-stock`), PD-mux D44(`PM-ref`)} 두 arm의 **지속** 용량 최소. 손대지 않은 SGLang(`SG-stock`)·패치 OFF(`SG-off`)·mixed-chunk(`SG-mix`) 3 arm은 min에서 뺀다 — 제외 사유: 첫 번째 Job에서 등록 격자 최저 rung 5.516 req/s에서 이미 attainment 0.90 미달(`LADDER_BOTTOM_FAIL`), 실패는 TTFT가 아니라 요청 내부 ITL p95 항(ITL 축 절벽)이고, 그 원인은 도착 1건당 약 42 ms의 decode 정지다(진단 문서). 이 3 arm은 지속 셀을 돌리지 않는다. 판정 셀(세 번째 Job)에서는 **같은 절대 rate로 그대로 돌리되**, 이 arm이 낀 쌍 중 Q3(vLLM 대 SG)·Q5·Q8은 goodput **부호**와 `BOTH_SATURATED`/`INVALID` 라벨 + 지연 공동 지표(TPOT·TTFT·요청 내부 ITL p95)로만 보고하고 % 크기를 인용하지 않는다(§4). **Q2는 이 처리 대상이 아니다**(사용자 지시 문언 그대로 — Q2는 등록 라벨 규칙 유지). [rev3.1에서 뒤집힘: Q2도 부호 전용] Q3에도 지연 공동 지표를 등록한다.
> 2. **아래쪽 사다리**(새 Job "J1-down", §3(가)): SG 3 arm × rung 5.516/1.15^k(k = 1…4: 4.796522 · 4.170888 · 3.62686 · 3.153791) × seed {201, 202} × 120요청, 같은 pick 규칙(첫 번째 Job 점 + 아래쪽 점을 합쳐 오름차순으로 걷는다). 아래쪽 바닥(3.153791)도 실패하면 `LADDER_BOTTOM_FAIL_2`(값 없음 유지). 결과는 Q1에만 쓰고 기준 용량에는 넣지 않는다. 두 번째 Job은 J1-down 없이 돌 수 있다(독립, 순서 자유).
> 3. **두 번째 Job(지속 셀)**: vLLM·PD-mux 두 arm만, 각 {1.0, 0.8} × 버스트 용량(조건부 0.5). **기준 셀(refcell) = PD-mux D44 arm, rate = 그 버스트 rung 7.29491 req/s**(첫 번째 Job에 같은 arm·rate·seed 201·120요청 셀이 실재). SG-stock 기준은 버스트 값이 없어 불가하다. 세 번째·네 번째 Job의 기준 셀도 같은 arm·rate다. 네 번째 Job의 W3 floor는 기준 셀 boot에서 SG-stock W3 사다리 boot(사다리 rung 앞)로 옮긴다 — floor를 재는 arm은 rev2.1과 같은 SG-stock으로 유지(기준 셀 arm이 바뀐 데 따른 부수 변경). 세 번째 Job은 새 기준 용량 정의를 따르되 arm 7개 전부 실행(SG 3 arm 포함), 네 번째 Job의 arm 집합은 rev2.1 그대로(4개).
> 4. **공정성 범위 문구 2개**(§9 인용 금지 10·11, 문자 그대로).
> 5. §7 Job 표(J1-down 추가, 두 번째 Job 재계산) · §8 예측 행 Q1-SG · §10 잃는 범위(SG 3 arm 지속 용량) · 분석기 "측정하지 않은 것" 자동 행(J1-down 상태·기준 용량 제외 사유·부호 전용 쌍).
> 6. 하네스: `sb_spec`(REF_ARMS·LADDER_DOWN_*·REFCELL_ARM = PM-ref·Q1_ARM·SIGN_ONLY_QUESTIONS·LATENCY_PAIRS + Q3·plan_J1_down·plan_problems의 J1-down/J1b/refcell 검사·J3 floor 위치), `sb_analyze`(ladder_down·q1_burst·sign_label·refcell_rate·기준 용량 min 범위·Q1-SG 채점·not_measured 행·`--j1down`), `sb_runner`(floor를 W3 사다리 boot에서 재는 경로, floor.json에 arm 기록), `sb_job.sh`(`PDMUX_SB_JOB` 허용 목록에 J1-down). 변이 10개 추가(M87–M96), M68 패턴 갱신. 첫 번째 Job plan sha는 rev2.1과 같다(`1482de03…`).
>
> **rev2 → rev2.1(2026-11-02, 재감사 r2 (a))**: 80 % 셀 PM-ref 대 SG-stock 쌍이 `BOTH_SATURATED`이면 P-3 = `UNSCORABLE`(사유 `BOTH_SATURATED`)(§8(i), 변이 1개) · §8(iii) P-4 채점 조건 정정 · §9 재감사 인용 금지 문장 6–9 추가.
>
> **rev1 → rev2 변경 요약(2026-11-02, 하네스 감사 r1 차단 수정 5건; 전부 CPU, 변이 테스트 포함)**
> 1. **지속 판정 교체**(§3): 도착 120 s 창별로 "600 s 안에 도착한 요청 중 E2E ≤ 60 s로 완료된 비율 ≥ 0.95" + 600 s 이후 도착 제외 + 마지막 창 TTFT 중앙 ≤ 첫 창의 2배(유지). rev1 규칙(창 안 완료 ÷ 창 안 도착, 600 s 이후 도착은 마지막 창에 합침)은 정상 서버를 구조적으로 탈락시켰다(감사 모형 통과 확률 0.013–0.037; CAP1-R D44 E2E 분포 모사 합성 run에서 재현).
> 2. **`BOTH_SATURATED`**(§4): 쌍의 두 arm이 모두 4 seed attainment ≥ 0.98이면 goodput 라벨 대신 이 라벨(EQUIV 금지, 모든 셀). Q8·Q5에 지연 공동 지표 3개 등록(seed별 paired ln TPOT 평균·ln TTFT p50·ln 요청 내부 ITL p95 중앙, 같은 3 %·t(3) 규칙, 라벨 접미 `_TPOT/_TTFT/_ITL`).
> 3. **mixed-chunk 처치 확인 교체**(§6): 서버 로그 server_args의 `enable_mixed_chunk=True` + chunked prefill 활성(`chunked_prefill_size` > 0)으로 판정. 옛 proxy("running ≥ 1인 Prefill batch 줄")는 처치와 무관하게 찍히므로 "mixing 기회 수"로 개명하고 SG-stock에도 계산해 음성 대조로 병기. Q6 격자에서 8k 제거, 길이별 n < 10이면 P-4 `UNSCORABLE`, 다른 stream extend가 한 gap이라도 > 0이면 Q6 INVALID.
> 4. **문언 정합**: 기준 용량 = 사다리 arm 5개의 지속 용량 최소(§3) · §3(가) arm 5개 · §5-0(a) 엔진별 오프셋 분산만 검사(엔진 간 차이 요구 없음 — 하네스 코드를 이 문언에 맞춤) · §6 "8k ≥ 10" 삭제 · PM arm 구성 묶음을 §2 표에 명시 + "Q2·Q4는 구성 묶음 대 기본값" caveat · 고부하 셀 attainment 예보 갱신(§3·§8) · 감사 인용 금지 문장 5개(§9).
> 5. **"측정하지 않은 것"**: Q1–Q8 행 강제 생성, 각 Job boot의 `DONE.json` status(skip·정확성 실패·이미지 거부·R3 예산 생략·DONE 없음) 반영, 기준 용량 None 사유 기재. 네 번째 Job: W3 사다리 직후 남은 시간을 R3 추정과 비교해 부족하면 R3 전체 생략(`R3_SKIPPED_BUDGET`, 반쪽 ABBA 금지)(§7). 정상상태 Job의 278 min 초과 가능성 공시(§7).
> 6. (코디네이터 지시, 문언만) §7의 "기준 용량 3.0 req/s 가정" 근거 정정 — CAP2 "완료 처리율 ≈ 3.0"은 스크립트 단위 오류(`policy_cap2/RECHECK_sustain_throughput_2026-11-02.md`).
> 7. (사용자 결정 2026-11-02, 감사 r1 C(ii) 처방 채택) SG-stock 계측 boot에 **512·2048 토큰 합성 프롬프트 각 10개**를 W1 60요청 뒤 별도 시간 창으로 추가(§5-1). P-4는 이 창의 2k 단일 요청 span만 채점, 512는 기술값, r_ser 분포는 W1 창·합성 창 따로 보고(§6). 계측 전용 boot라 goodput 영향 0(§5-1). 세 번째 Job 추정 +1.0 min(§7). 변이 테스트 4개 추가(창 섞기·다요청을 단일로 세기 + 보조 2개: 실행기 W1 창이 합성 구간까지 늘어남·P-4를 W1 창에서 채점).
> **rev0 → rev1 변경(2026-11-02, 사용자 "권고대로 진행")**: ① W1 변화 trace 제거(고정 설정끼리의 비교라 컨트롤러 반응을 볼 대상이 없음; 게이트 2는 동적 arm 비교에 한정 — §3·§10) ② vLLM은 같은 공개 이미지 안 별도 venv(`sglang-vllm-runtime@sha256:a5f9bf0f…`, SGLang 쪽 바이트 불변 검증 완료, Public 확인 2026-11-02)로 같은 Job에서 boot ③ 계획 대체 6건 등록(§5·§3, 아래 "대체 목록") ④ 시간 기준 확인을 2단계로(§5-0) ⑤ 용어: "기준 용량" = 7개 arm 중 가장 낮은 지속 용량(min) [rev2 정정: **사다리 arm 5개**의 지속 용량 최소 — §3(나)], "부하 창" = 클라이언트가 요청을 보내기 시작해 마지막 완료까지의 시간 구간 — 사용자 승인. 코드명(D-n·O-n·J1b·R₁ 등)은 문서에서 쓰지 않는다; 코드 식별자는 하네스 안에서만.
> 결정 62(2026-11-01, "진행") + 2026-11-01 사용자 "진행 그리고 관련 수치들에 대한 확보"로 착수. 계획 원문 `PLAN_stock_baseline_metrics_2026-11-01.md` §3, Phase A 결과 `RESULT_stock_baseline_phaseA_2026-11-01.md`.
> 2026-11-01 개정 규약(experiment-runner "PREREG 필수 항"·"부하 위치"·"승계 상수"·"처치 발생 횟수"; claims-auditor 파이프라인 모집단 층)을 이 문서가 처음 적용한다.
> 라벨: [측정]/[유도]/[가정]/[코드]/[결정]/[제안].

## 0. 답하는 질문 목록 (규약 필수 항 1 — 예산이 모자라면 arm이 아니라 여기서 줄이고 적는다)

| # | 질문 | 답을 주는 블록 | 라벨 출력 |
|---|---|---|---|
| Q1 | VESSL·Nano·W1에서 stock SGLang(기본값)의 용량 λ\*(버스트·지속)은 얼마인가 | (가) 사다리 + 아래쪽 사다리(rev3; 지속 셀은 rev3에서 SG arm에 돌리지 않음 → 버스트만) | 수치(기술값; rev3.1: 예측 행 없음) |
| Q2 | 같은 Job·같은 절대 rate(손대지 않은 SGLang은 자기 ITL 용량 위일 수 있음)에서 static PD-mux(D44)의 꼬리 SLO goodput이 stock SGLang과 다른가 (rev3.1: 부호만) | (나) 정상상태 셀 ×2 | rev3.1: goodput 부호 라벨(`PM_ABOVE_SG_SIGN`/`PM_BELOW_SG_SIGN`/`SIGN_UNRESOLVED`)·`BOTH_SATURATED`·`INVALID` + 지연 공동 지표 `…_TPOT/_TTFT/_ITL`(§4) [rev3까지: `PM_ABOVE_SG` / `EQUIV_WITHIN_3PCT` / `PM_BELOW_SG` / `UNRESOLVED_INTERVAL`] |
| Q3 | 같은 조건에서 stock vLLM 대 stock SGLang은 어떤가 (대조군이 약하다는 반박의 방어) | (나) | rev3: goodput 부호 라벨(`VL_ABOVE_SG_SIGN`/`VL_BELOW_SG_SIGN`/`SIGN_UNRESOLVED`)·`BOTH_SATURATED`·`INVALID` + 지연 공동 지표 `…_TPOT/_TTFT/_ITL`(§4) |
| Q4 | static PD-mux 대 stock vLLM | (나) | `PM_vs_VL` |
| Q5 | mixed-chunk(SG-mix)가 stock SGLang의 ITL 꼬리를 바꾸는가 | (나) | rev3: goodput 부호 라벨만 + 지연 공동 지표 `…_TPOT/_TTFT/_ITL`(rev2, §4) |
| Q6 | stock SGLang의 r_ser(L)·r_mix가 A5 예보(`R_SER_FORECAST_NANO.csv`)와 맞는가 | 계측 boot | 예보 대비 잔차(HIT/MISS 띠는 §8) |
| Q7 | W3(장문)에서 Q2·Q3의 부호가 유지되는가 | 네 번째 Job W3 정상상태 셀 | 같은 라벨, W3 접미 |
| Q8 | 수정 트리·플래그 OFF(`SG-off`)가 pristine stock(`SG-stock`)과 같은가 — implfix C6의 GPU 검증 | (나) | rev3: goodput 부호 라벨만(goodput EQUIV 없음) + 지연 공동 지표 `…_TPOT/_TTFT/_ITL`(rev2, §4) — 동등성은 지연 공동 지표의 `EQUIV_WITHIN_3PCT_*`로만 읽는다 |

**이 Job이 답하지 않는 것(§10 잃는 범위에 재기재)**: 조율한 fused(chunk·overlap·mixed 튜닝) · Llama-8B 참조점 · 동적 PD-mux · Nano에서 static PD-mux의 best D(D44는 다리 arm, 아래 §2) · 과부하 구간.

## 1. 기판·트리 [결정/코드]
- VESSL A100 SXM 80GB ×1, **이미지 1종**(rev1): `PDMUX_VESSL_IMAGE_SBASE` = `ghcr.io/asuan99/sglang-vllm-runtime@sha256:a5f9bf0f…` — 기존 공개 `sglang-runtime@1e0d335b…`에 `/opt/vllm-python`·`/opt/vllm-venv`(손대지 않은 vLLM 0.31.0, venv freeze sha `adff2ff0…`)만 더한 것. SGLang 쪽 파일 84,274개 해시가 기존 이미지와 동일(`BRINGUP_vllm_0.31.0_2026-11-01.md`). 공개 이미지, PREREG·결과·엔진 패치 없음. boot마다 두 freeze sha를 검사.
- ★**트리 구분(2026-11-01 사용자 지적으로 정정)**: 공개 이미지는 pristine v0.5.10이지만 Job 시작 시 `install_runtime.sh`가 PD-mux 수정 트리(14파일, +2623/−50)를 editable 설치한다. 따라서 **지금까지의 모든 "fused/OFF" arm(M3 J1 fused 포함)은 "수정 트리·플래그 OFF"였다.** 이 PREREG에서는 boot 단위로 트리를 고른다(`PDMUX_ENGINE_TREE=pristine|patched`, 선행 작업 §11-7):
  - `SG-stock`·`SG-mix` = **pristine 트리**(install 전 보존본, `PDMUX_*` env 0개, 마커 가드) — 진짜 stock.
  - `SG-off` = **수정 트리·플래그 OFF**(새 arm) — M3 J1 fused·implfix C6("OFF = main"은 CPU golden만)·CAP/TUNE OFF 참조와 잇는 다리. `SG-off vs SG-stock`이 처음으로 GPU에서 "수정 트리 OFF = stock인가"를 잰다(라벨 `PATCH_OFF_vs_STOCK`, Q8).
  - `PM-ref` = 수정 트리, 비sticky static D44(결정 49).
  - 각 boot 메타(`tree_<arm>.json`)에 `engine_tree`·`engine_tree_manifest_sha256`·`engine_tree_ref_sha256`·`engine_tree_ref_overridden`·`engine_tree_markers_found`·`engine_tree_sglang_init`·`engine_tree_root`를 기록하고 분석기 스탬프 검사(⑤)에 포함: SG-stock/rep/mix는 `pristine`·manifest == ref `1bd84d88…`·overridden false·markers `[]`·argv에 `--enable-pdmux` 없음; SG-off/PM은 `patched`·마커 3개·같은 Job의 patched manifest 동일. 30항목 ENGINE_DIGEST(`2836112b…`)와 트리 전체 manifest를 둘 다 기록.
- 엔진 digest는 계측 플래그(§5)가 들어간 트리로 1개 고정(봉인 시 `ENGINE_DIGEST.txt`).
- vLLM 버전: **0.31.0**(pip index 2026-11-01 최신 stable; 결정 62 ① "제출 시점 최신 stable") — 봉인 시점에 더 새 stable이 있으면 그것으로 갱신하고 봉인 문서에 적는다. attention 백엔드·`CUDAGraphMode`·chunked prefill 상태는 서버 로그에서 채취해 결과에 병기(기본값 그대로, 튜닝 0).

## 2. arm [결정 62 + 승계 상수 규약]

| arm | 엔진 | 설정 | 역할 |
|---|---|---|---|
| `SG-stock` | SGLang v0.5.10 | 기본값: chunked prefill 8192, overlap ON, mixed OFF, flashinfer, cudagraph ON, radix OFF(공통) | 기본 구조 baseline |
| `SG-rep` | 〃 | `SG-stock` 복제 boot | 플라시보(boot 효과 1회 추출) |
| `SG-mix` | 〃 | + `--enable-mixed-chunk` | Q5 |
| `VL-stock` | vLLM 0.31.0 | 기본값(무패치), prefix caching OFF(radix OFF 대응), cudagraph 기본 | Q3·Q4 |
| `SG-off` | SGLang 수정 트리 | 플래그 OFF(기존 모든 fused/OFF arm과 같은 트리) | Q8 다리 arm (§1 트리 구분) |
| `PM-ref` | SGLang 수정 트리 + PD-mux | static D44 비sticky, budget 65536, cap 48 + **구성 묶음**(아래) | **다리 arm** — CAP/TUNE/implfix 수치와 잇는 arm |
| `PM-ref34` | 〃 | static D34 비sticky(실현 (74,34) B3a PASS, 입도 2 SM) + **같은 구성 묶음** | 고원 안 두 번째 점(결정 ④) |
| `PM-rep` | 〃 | `PM-ref` 복제 boot(argv·env·트리 동일, rev3.1) — 세 번째 Job 50 %·80 % 셀만(rev3.2: 네 번째 Job W3 셀에서 제거) | PD-mux 플라시보(Q4·Q2의 PM 쪽, §4) |

arm은 8개다(rev3.1: `PM-rep` 추가). 사다리·지속 셀은 rev3과 같다(`PM-rep`은 돌리지 않는다). rev3.2: `PM-rep`은 세 번째 Job(R1·R2)에서만 돈다.

**PM arm 구성 묶음(rev2, 감사 r1 D·confound 10)** [코드: `sb_spec.pm_argv` = 정책 하네스 `ph_runner.server_argv(static_arm(D))` 바이트 그대로]: `--mem-fraction-static 0.6` · `--disable-overlap-schedule`(overlap OFF) · `--chunked-prefill-size -1`(chunked prefill 끔) · `--random-seed 1` · `--max-mamba-cache-size 48` · `--attention-backend flashinfer`(명시) · `--max-running-requests 48` · PD-mux config(budget 65536) · env `PDMUX_R2_POLICY=fixed`, `PDMUX_R2_FIXED_DSM=D`. SG-stock/SG-off/SG-mix는 이 플래그들을 하나도 주지 않는다(SGLang 기본값: mem fraction 자동, overlap ON, chunked prefill 8192, seed 기본). ★**caveat: Q2·Q4의 라벨은 "PD-mux"가 아니라 "구성 묶음(PD-mux + 위 플래그 6개) 대 기본값"의 비교다.** PD-mux 단독 효과는 이 Job에서 분리되지 않는다(§9 인용 금지 문장 1).

**승계 상수 공시**: `PM-ref`의 D44·cap 48·budget 65536은 KISTI 승계값이며 Nano에서 재결정되지 않았다(R Job: D34–64 goodput 고원 1.5% 안, 순위 미분해). 따라서 Q2·Q4의 라벨은 **"static PD-mux(D44 구성)"** 범위로만 성립하고 "static PD-mux 최적" 범위가 아니다. 최적 D 재결정은 CAP′(HE0 셀 재정의 결정 뒤)의 몫이다. 결정 ④(2026-11-01): `PM-ref34` 포함 — Q2·Q4 라벨은 D34·D44 두 점으로 보고하며 "고원 안 두 점" 범위.

공통: **`--disable-piecewise-cuda-graph`**(v0.5.10 NemotronH 기본 ON이나 이 이미지에서 트리 무관 부팅 실패 — `multigpu/RESULT_m2g_c1395d0_2026-10-12.md` `typing.Union`; "기본값"에서 유일하게 바꾸는 플래그, 공시) · `--context-length 20480`·`--max-prefill-tokens 16384`(정책 하네스 S7 승계; W3 7k 평균이 들어감) · max_running 48(vLLM `--max-num-seqs 48`) · 출력 길이 trace 고정 · 같은 체크포인트 경로 `/data/hf`.

## 3. 워크로드·셀 [결정 44·49, 부하 위치 규약]
- W1 ShareGPT(정책 하네스 `ph_spec` 정의·seed 그대로), W3 arXiv-Sum(7k 평균, `W3_MAX_NEW` 1024). SLO: W1 TTFT 3000 ms ∧ 요청 내부 ITL p95 60 ms; W3 TTFT = 3 × floor(같은 Job 첫 boot에서 측정, N3 규칙).
- **(가) 용량 사다리** — arm **5개**(SG-stock·SG-mix·SG-off·VL-stock·PM-ref; rev2 정정 — rev1 문언 "4개"는 SG-off 추가 전 문장, 코드 `LADDER_ARMS`는 5개) × W1 10단(5.516 × 1.15^k, k=0..9 — CAP1-R 격자 승계, Nano PD-mux 용량 7.29가 격자 안에 있음을 확인한 격자) × 120요청 × seed {201, 202}. λ\*_arm = attainment ≥ 0.90이 앞에서부터 끊기지 않는 최대 rung(`pick`와 같은 규칙). **동률 규칙 등록**: 동률이면 낮은 rate(보수). 격자 최상단이 통과하면 `LADDER_TOP_PASS`(λ\* 미확정, 한 번의 확장 재실행 등록: 격자 ×1.15^k, k=10..13).
  - **아래쪽 사다리(rev3, Job "J1-down")**: SG-stock·SG-mix·SG-off × rung 5.516/1.15^k, k = 1…4(4.796522 · 4.170888 · 3.62686 · 3.153791) × seed {201, 202} × 120요청, arm당 boot 1개(rung·seed 순서는 boot 안 무작위), smoke 없음. λ\*는 첫 번째 Job 사다리 점과 아래쪽 점을 합쳐 같은 pick 규칙으로 정한다(동률 → 낮은 rate). 아래쪽 바닥 3.153791도 실패하면 `LADDER_BOTTOM_FAIL_2`이고 값은 없다(더 아래 확장은 등록하지 않는다). 이 값은 **Q1에만** 쓰이며 기준 용량·지속 셀·판정 셀 rate에는 들어가지 않는다. 두 번째 Job과 독립이라 순서는 자유다. [범위] 이 사다리는 첫 번째 Job 결과를 본 뒤 등록한 것이다(rung 선택은 첫 번째 Job의 5.516 실패에서 왔다).
- **지속 셀**(결정 ①; rev3: **vLLM·PD-mux D44 두 arm만** — SG 3 arm은 버스트 용량이 없어 지속 점을 정할 수 없고 기준 용량에서도 빠진다): 각 arm {λ\*_arm(버스트), 0.8·λ\*_arm} × 600 s × seed 201 → 완료 처리율(120 s 창별)·admission 속도 분리 보고(CAP2 감사 (3) 규약). λ\*_arm(지속) = 두 점 중 아래 규칙을 만족하는 최대 rate; 둘 다 미달이면 0.5·λ\*_arm 1점 추가(등록 1회). 버스트·지속 둘 다 보고.
  - **지속 판정 규칙(rev2, 감사 r1 A)**: 요청을 **도착 시각**(`scheduled_arrival_s`)으로 120 s 창 w = 0..4에 넣는다. **600 s 이후 도착은 판정에서 제외**(개수만 `n_late_excluded`로 보고). 창 w의 완료율 = (창 w에 도착한 요청 중 오류 없이 **E2E ≤ 60 s**로 완료된 수) ÷ (창 w에 도착한 수), E2E = `completion_s − sent_s`(송신 시각 없으면 `scheduled_arrival_s`; 둘 다 run 시작 기준 시각). 통과 = 다섯 창 모두 완료율 ≥ 0.95 **이고** 마지막 창(도착 480–600 s) TTFT 중앙 ≤ 첫 창의 2배. 첫 토큰 기준 admission 속도(창 안 첫 토큰 ÷ 창 도착)는 보고 열(판정 아님).
  - rev1 규칙("창 안에서 완료된 수 ÷ 창 안 도착 수", 600 s 이후 도착은 마지막 창에 합침)은 첫 창(채움 과도)과 마지막 창(600 s 뒤 완료 미집계)을 구조적으로 깎아 정상 서버를 탈락시켰다: CAP1-R D44 E2E 분포(5.516–7.295 req/s, 720요청, E2E p50 7.8 s·p95 18.5 s)를 모사한 푸아송 합성 run에서 rev1 규칙 통과 1/50, rev2 규칙 통과 50/50(3.65·5.84·7.29 req/s); CAP2 과부하 raw 4 run은 rev2 규칙으로 전부 실패(창 완료율 0.96–1.0 → 0.1–0.33 → 0). 변이 테스트: rev1 분자로 되돌리면 정상 합성 run 테스트가 실패한다.
- **(나) 판정 셀** — 공통 절대 rate(결정 ③). ★**rev3 기준 용량 = {VL-stock, PM-ref} 두 arm의 지속 용량 최소**(사용자 결정 2026-11-03). **SG-stock·SG-off·SG-mix 3 arm은 min에서 제외한다 — 제외 사유: 첫 번째 Job에서 사다리 바닥 실패(등록 격자 최저 5.516 req/s에서 attainment 0.000–0.225, `LADDER_BOTTOM_FAIL`). 실패 항은 TTFT가 아니라 요청 내부 ITL p95(ITL 축 절벽: 요청 중앙 ITL p95가 60 ms 바로 위에 몰림)이고, 그 원인은 도착 1건당 약 42 ms의 decode 정지다(`DIAG_sglang_default_itl_2026-11-03.md`; `--enable-pdmux` 플래그 하나를 빼면 같은 정지가 돌아온다 — `DIAGBOOT_itl_gap_2026-11-03.md`). 버스트 용량 값이 없으므로 지속 점도 정할 수 없다.** 이 3 arm은 판정 셀을 **같은 절대 rate로 그대로 돌린다**(세 번째 Job arm 7개 전부; rev3.1: + `PM-rep` = 8개). 그 rate는 이 3 arm에게는 자기 SLO 용량 위일 수 있으므로, 이 arm이 낀 쌍 중 **Q3(VL-stock 대 SG-stock)·Q5(SG-mix 대 SG-stock)·Q8(SG-off 대 SG-stock)**은 goodput 부호·`BOTH_SATURATED`/`INVALID` 라벨 + 지연 공동 지표로만 보고하고 % 크기를 인용하지 않는다(§4 부호 전용 규칙). ~~Q2(PM-ref·PM-ref34 대 SG-stock)는 부호 전용 대상이 아니다~~ ★**rev3.1: Q2(PM-ref·PM-ref34 대 SG-stock)도 부호 전용이다**(사용자 결정 (b), 감사 rev3 r1) — 등록 goodput 라벨을 내는 쌍 질문은 Q4뿐. [이전 문언, rev2.1] 기준 용량 = **사다리 arm 5개**(SG-stock·SG-mix·SG-off·VL-stock·PM-ref)의 **지속** 용량 최소(결정 ①; 지속 셀이 두 번째 Job에서 나오므로 판정 셀 rate는 그 회수 뒤 확정). rev2 정정(감사 r1 D): rev1 문언 "7개 arm"과 달리 두 arm은 min에서 뺀다 — **SG-rep**은 SG-stock과 같은 설정의 복제 boot(플라시보)라 용량이 SG-stock과 같은 양이고 사다리를 따로 돌리지 않는다; **PM-ref34**는 사다리를 돌리지 않고 CAP1-R D34 용량 6.343 req/s를 승계한 기술값(다른 Job·다른 GPU에서 잰 값)이라 이 Job들의 지속 용량과 같은 척도가 아니다. 어느 arm이든 지속 용량이 None이면 기준 용량은 None이고(판정 셀 Job 제출 불가), 사유는 "측정하지 않은 것"에 적힌다(rev3: "어느 arm" = VL-stock·PM-ref). 정상상태 셀 두 개: 기준 용량의 50%와 80%, 각 200요청 × seed {1,2,3,4}. **변화 trace(LO↔HI 반복)는 쓰지 않는다**(rev1: 7개 arm 전부 고정 설정이라 부하 변화에 반응할 컨트롤러가 없고, 저부하·고부하는 두 정상상태 셀이 덮는다; 시간 모형상 trace가 네 번째 Job의 76%를 차지했다). **부하 위치 예보(rev2 갱신, 감사 r1 B)**: 80% 셀에서 **best arm attainment ≈ 1.0 가능** [측정 근거: CAP1-R에서 PD-mux arm은 5.516·6.343 req/s에서 attainment 1.00; rev1 예보 "0.6–0.85"는 이 데이터와 불일치해 철회]. 따라서 80% 셀도 용량 아래일 수 있고, 그때 goodput 쌍은 `BOTH_SATURATED`(§4)가 된다. 측정 후 전 arm > 0.9이거나 best < 0.5면 `CELL_OUT_OF_BAND` — 라벨은 내되 크기 인용 금지(전 arm > 0.9는 위 예보상 예상 가능한 결과다).
- **Job 간 기판 눈금(refcell)**: Job 2·3·(1b) 시작과 끝에 기준 셀을 120요청·seed 201로 1회씩 돌려 attainment·TTFT p50·throttle 비율을 기록한다. ★**rev3: 기준 셀 = `PM-ref`(PD-mux D44) arm, rate = 그 버스트 rung 7.29491 req/s**(첫 번째 Job에 같은 arm·rate·seed·요청 수 셀이 실재, attainment 0.967/1.000). rev2.1의 "`SG-stock`을 λ\*_ref(지속)에서"(대체 4로 SG-stock 버스트 rung)는 SG-stock 버스트 값이 없어 불가하다. Job 1의 같은 점과 attainment 차가 **10%p 이상**이면 그 Job의 셀에 `SUBSTRATE_SHIFT`를 붙인다 — 라벨은 내되 Job 1의 용량 수치·다른 Job의 goodput 크기와 섞지 않는다. 판정 셀의 `CELL_OUT_OF_BAND` 검사는 Job별로 적용한다(GPU UUID가 Job마다 다르므로, 같은 절대 rate가 다른 상대 부하가 될 수 있다).
- W3(장문): 네 번째 Job. floor(단일 요청 TTFT 중앙) 측정(rev3: 기준 셀 arm이 PM-ref로 바뀌어 floor는 **SG-stock W3 사다리 boot에서 사다리 rung 앞에** 잰다 — floor arm은 rev2.1과 같은 SG-stock) → SG-stock W3 용량 사다리(0.1×1.5^k 6단 × 48요청 × seed 2) → 정상상태 셀 = SG-stock W3 용량의 80% × 96요청 × seed {1,2,3,4}, arm 4개(SG-stock·SG-rep·PM-ref·VL-stock; ~~rev3.1: + `PM-rep` = 5개~~ rev3.2: `PM-rep` 제거, 4개). 사다리는 SG-stock 1 arm만(§10).

## 4. 판정 규칙 (데이터 전 등록; 정책 하네스 `pi_analyze` 규약 승계)
- 쌍: seed-level paired ln(A/B) goodput, t(3) 95% CI. `EQUIV_WITHIN_3PCT`는 |mean| < ln1.03 ∧ CI ⊂ (−ln1.03, +ln1.03); `A_ABOVE_B`는 CI 하한 > 0 ∧ mean ≥ ln1.03; 반대는 `A_BELOW_B`; 그 외 `UNRESOLVED_INTERVAL`.
- **검출력 사전 계산(규약 필수 항 2)**: 플라시보(SG-rep vs SG-stock) sd(ln) 예보 ≤ 0.006 [유도: H Job static 플라시보 0.0016, P Job 0.0054]. n=4·t(3)=3.182에서 EQUIV 가능 조건 sd < 0.0186, 반폭 = 3.182·sd/2 → sd 0.006이면 ±1.0%. 플라시보 sd가 0.0186을 넘으면 그 셀의 EQUIV는 불가하고 `UNRESOLVED_INTERVAL`이 정상 결과다(공시). 엔진 간 쌍(SG vs VL)은 같은 trace·seed라도 다른 프로세스라 sd가 플라시보보다 클 수 있다 — 예보 ≤ 0.02 [가정], 미달이면 seed 8로 재실행(등록 1회, 같은 Job 구조).
- **`BOTH_SATURATED`(rev2, 감사 r1 B)**: 쌍의 두 arm이 **모두 4 seed 각각** attainment ≥ 0.98이면 goodput 라벨(ABOVE/BELOW/EQUIV/UNRESOLVED)을 내지 않고 `BOTH_SATURATED`를 낸다 — 저부하(R1)·고부하(R2)·W3(R3) 셀 모두. 근거: 두 arm이 거의 전부 SLO 안에서 처리하면 goodput ≈ 요청 수 ÷ duration이라 ln 비는 drain 차이뿐이고 "3 % 이내 동등"이 구조적으로 나온다(CAP1-R 용량 아래 rung D44/D34·D44/D64 ln 비 전부 |≤ 0.010|). 유효성 사슬 실패(INVALID)·플라시보 무효(BOOT_NOISE)가 이보다 우선한다. ln 비 수치와 원래 라벨은 `raw_label`로 기록만 한다(인용 금지).
- **부호 전용 규칙(rev3, Q3·Q5·Q8; rev3.1 + Q2)**: 이 질문의 쌍은 위 goodput 라벨 대신 **부호 라벨**을 낸다 — CI 하한 > 0이면 `A_ABOVE_B_SIGN`, CI 상한 < 0이면 `A_BELOW_B_SIGN`, 그 밖(n < 4 포함)은 `SIGN_UNRESOLVED`(3 % 여유 없음, `EQUIV_WITHIN_3PCT` 없음 — 동등성은 크기 주장이다). `BOOT_NOISE`·`INVALID`·`BOTH_SATURATED`·`MIX_NOT_FIRED`가 이보다 우선한다. 원래 라벨은 `raw_label`로 기록만 하고(인용 금지) 태그 `MAGNITUDE_NOT_CITABLE`을 붙인다. ln 비 수치는 기록되지만 인용하지 않는다.
- **지연 공동 지표(rev2, Q8·Q5; rev3 + Q3; rev3.1 + Q2)**: seed별 paired ln(A/B)를 세 지표로 — **TPOT** = 요청별 token ITL 평균의 run 평균(오류 없는 요청), **TTFT** = run TTFT p50, **ITL** = 요청 내부 token-ITL p95의 run 중앙값. 라벨은 goodput과 같은 3 %·t(3) 규칙에 접미 `_TPOT/_TTFT/_ITL`(rev3: Q3 = `VL_…_SG_*`, 엔진 간 쌍이라 정의 일치 smoke·부하 중 TTFT 오프셋 검사가 유효성 사슬에 들어간다)(예: `OFF_ABOVE_STOCK_TPOT` = SG-off의 TPOT가 3 % 넘게 **크다**(느리다), `EQUIV_WITHIN_3PCT_TTFT`). 같은 유효성 사슬·그 쌍의 K3 플라시보(goodput; rev3.1 base arm별)를 따르고(실패 시 INVALID/BOOT_NOISE), Q5는 처치 미확인 시 `MIX_NOT_FIRED`. `BOTH_SATURATED`는 지연 지표에 적용하지 않는다(포화 셀에서 Q8·Q5를 읽는 수단이 이것이다). 지연 지표에는 별도 플라시보 게이트를 두지 않는다(K3 단일) — 공시.
- 플라시보 무효 규칙(K3): 플라시보 쌍이 n<4 또는 |mean| ≥ ln1.03 또는 CI가 0 밖이면 무효. ★**rev3.1: 쌍의 base arm별로 적용한다** — SG 계열 arm(SG-stock·SG-rep·SG-mix·SG-off)이 낀 쌍은 SG-rep 대 SG-stock, PD-mux 계열 arm(PM-ref·PM-ref34·PM-rep)이 낀 쌍은 PM-rep 대 PM-ref 플라시보가 유효해야 한다(VL-stock은 플라시보 없음). 그 쌍이 따르는 플라시보 중 하나라도 무효면 그 쌍(goodput·지연 공동 지표)은 `BOOT_NOISE`. 결과: Q3·Q5·Q8 ← SG 플라시보, Q4 ← PM 플라시보, Q2 ← 둘 다. 한 셀에서 SG 플라시보가 무효여도 PM 플라시보가 유효하면 Q4는 `BOOT_NOISE`가 아니다. `PM-rep` run이 없으면 PM 플라시보는 무효(fail-closed). ★**rev3.2: 이 PM 플라시보 요구는 50 %·80 % 셀(R1·R2)에만 적용한다** — W3 셀(R3)에는 `PM-rep`이 없고 PM 플라시보를 요구하지 않으므로, R3의 Q2 쌍(Q7 키)은 SG 플라시보만 따른다(rev3과 같음). PM 플라시보 sd 예보는 따로 등록하지 않는다(EQUIV 가능 조건 sd < 0.0186은 같다). [rev3까지: K3 단일 — SG-rep 대 SG-stock 무효 → 그 셀 전부 `BOOT_NOISE`]
- 유효성 사슬: ① 같은 seed trace hash 동일(두 엔진 클라이언트 동일 하네스) ② cliff(TTFT가 SLO ±5% 안 > 10%) 양 arm 어디에도 없음 ③ 스탬프(Job·GPU UUID 동일) ④ 그 쌍의 플라시보 유효(rev3.1 base arm별) ⑤ 정의 일치 smoke 통과(§5-0) ⑥ 요청 실패 0.
- 지표: goodput(정본 술어) headline, 병기 TTFT/TPOT/ITL p50/p95/p99 · req/s · out tok/s · E2E(요청별 send/first/finish 시각 기록 — 새 측정은 처음부터) · 요청 내부 ITL p95 중앙.

## 5. 계측 boot (측정 arm과 분리, C5) + 축약 입력 필터 (규약 필수 항 3)
- 5-0 **시간 기준 확인(2단계)**. 목적: SGLang `/generate` 파서와 vLLM `/v1/completions` 파서가 "첫 토큰"을 같은 사건에서 찍는지. 서버는 요청별 첫 토큰 시각을 로그에 남기지 않으므로(SGLang은 `--enable-metrics` 시 `sglang:time_to_first_token_seconds` 히스토그램만, vLLM은 기본 `/metrics` 히스토그램만) 서버 쪽 값은 히스토그램 합계·개수로만 얻는다. (a) **하나씩 20요청**(첫 boot, 측정 아님): 요청 하나가 끝날 때마다 히스토그램 합계 증가분(개수 증가 1인 것만) = 그 요청의 서버 TTFT; 클라이언트 TTFT와의 차(오프셋)가 엔진마다 작은 상수(HTTP 왕복 수 ms)이고 분산이 작은지 본다 — **판정(rev2 확정)**: 엔진마다 오프셋의 표준편차 ≤ 5 ms(사용 가능한 요청 ≥ 2)면 `CLIENT_DEF_OK`. 두 엔진의 상수가 같을 필요는 없다(서버 쪽 시작점이 다름: SGLang은 tokenizer manager 수신, vLLM은 프론트엔드 도착) — rev1 하네스 코드는 |중앙_SG − 중앙_VL| ≤ 5 ms를 요구해 이 문언과 어긋났으므로(감사 r1 D) **코드를 이 문언에 맞췄다**(엔진 간 차이 요구 삭제, 변이 테스트 2개: 분산 검사 제거·엔진 간 요구 재도입 둘 다 검출). (b) **동시 요청 상태**(rev1 추가): 50% 셀 부하(200요청, seed 301)에서 SGLang은 `--enable-metrics`를 켠 별도 boot(계측 역할, 세 번째 Job), vLLM은 50% 측정 boot 자체의 run 전후 `/metrics` 합계·개수로, run 단위 클라이언트 TTFT 평균 − 서버 TTFT 평균을 (a)의 오프셋과 비교. 차이 > 20 ms면 `TTFT_OFFSET_SHIFT`(rev1.1: "2배" 규칙은 오프셋이 수 ms라 잡음에 발화하므로 삭제, 절대 문턱만) → 엔진 간 쌍 전부 `INVALID/UNRESOLVED:TTFT_OFFSET_SHIFT`(fail-closed). (a) 실패는 `CLIENT_DEF_MISMATCH`, 같은 처리. 비교는 끝까지 클라이언트 TTFT로만 한다.
- 5-1 SG-stock 계측 boot: `PDMUX_PROBE_EXTEND_TIME=1`(§11-3 **완료, CPU**: `decode_probe` `extend_forward` 레코드 — `extend_gpu_ms`·`extend_tokens`·`n_reqs`·`stream_key`·`seq`·`decode_iter_seq`; 순수 EXTEND만, MIXED·SPLIT_PREFILL 제외; `pacing_ratios(records, min_bs=5, t_window)`가 r_ser·r_mix를 냄, T_dec = bs별 순수 DECODE 중앙값; OFF 골든 26레코드 바이트 동일, 변이 12종 검출; 엔진 digest `65cbb49d…` → `2836112b…`) + decode_probe, R₂ 부하 60요청 seed 301. **합성 단일 요청 창(사용자 결정 2026-11-02)**: 같은 boot·같은 run에서 W1 60요청이 **모두 끝난 뒤** 두 번째 구간으로 요청 21개를 2 s 간격으로 보낸다 — 앞 1개는 선행 요청(64토큰, 두 길이 띠 밖; 창의 첫 gap은 `pacing_ratios`가 버리므로 첫 채점 프롬프트 앞에 decode를 두려는 용도, 채점에 안 들어감), 뒤 20개 = 512·2048토큰 각 10개(순서는 seed 301 파생 섞기). 프롬프트는 seed 301 파생 결정적 random token id(범위 [1000, 32000), 요청별 id sha256을 run 기록·trace hash에 남김), 출력 64토큰, `ignore_eos`. 간격 2 s ≫ 단일 요청 처리(prefill + 64토큰 decode ≈ 1 s, [유도] bs 1 decode ≈ 13 ms/토큰 — 예보 CSV `T1_ms`)라 각 프롬프트가 **혼자 prefill**(EXTEND forward 1개, `n_reqs` = 1)되도록 한 설계다. run 기록에 W1 부하 창 `t_load_start/end`와 별개로 합성 창 `t_synth_start/end`(epoch·CLOCK_MONOTONIC)를 둔다 — W1 창은 W1 구간에서 끝나고 합성 창과 겹치지 않는다. ★**이 boot는 계측 전용이며 goodput 계산의 입력이 아니다**(아래 라벨별 입력 필터 표: goodput ← 측정 boot만; 계측 boot는 모든 측정 boot 뒤에 돌고 다른 arm과 boot를 공유하지 않는다) → **합성 프롬프트 추가의 goodput 영향은 0**. 잃는 것은 세 번째 Job 시간 +1.0 min(§7)뿐. **입력 필터**: forward 레코드에는 요청 ID가 없으므로 **시간 창만** — `t_window = [t_load_start, t_load_end]`(프로브 `perf_counter` 시계; boot 레코드의 `t_realtime`/`t_host`로 클라이언트 송신·완료 시각을 사상), warmup·smoke는 창 밖에 두어 제외. 창 밖 레코드를 섞으면 r_ser p50이 바뀌는 **변이 테스트** 등록(교훈 53). GPU 미검증 항목(봉인 전 하네스 감사 공시): EXTEND 이벤트 실측·overlap 경로의 단일 forward stream(`stream_key` 상수)·ON boot 출력 불변·관측자 비용·합성 프롬프트가 실제로 단일 요청 EXTEND 1개로 나가는지(앞 요청 decode가 2 s 안에 끝나는지 포함)·random id 프롬프트가 Nano 토크나이저 범위 안에서 오류 없이 처리되는지.
- 5-2 VL-stock 계측 boot: 무패치 원칙 → `/start_profile`~`/stop_profile` torch profiler 창(같은 부하), step별 GPU 시간 + 토큰 수로 r_mix. 관측자 비용은 계측 boot 안에서만 공시.
- 5-3 PM-ref 계측 boot: implfix `pc_reduce` 정의 재사용하되 **부하 창 필터를 적용한 수정판**(봉인 코드 수정 아님 — 새 함수, 변이 테스트 포함).
- 라벨별 입력 필터 표(감사자용): `λ*` ← (가) run 클라이언트 요청만 · `goodput` ← (나) 측정 boot 클라이언트 요청만(계측 boot 제외) · `r_ser/r_mix` 분포 ← 5-1/5-2 부하 창(W1 창과 합성 창은 **따로** 축약, 합치지 않음) · `P-4` ← 5-1 합성 창의 단일 요청(`n_reqs` = 1) 2k span만 · `플라시보` ← SG-rep 측정 boot(SG 계열 쌍)·`PM-rep` 측정 boot(PD-mux 계열 쌍, rev3.1).

### 계획 대체 목록 (rev1 등록, 2026-11-02 승인) — 초안 문장이 전제한 것이 시스템에 없어 자리를 옮긴 6건
1. SG-stock의 prefill/decode 시간비(Q6) 타이머는 우리 패치 코드라 손대지 않은 SGLang에 없다 → 타이머 측정 boot만 패치 트리·PD-mux OFF로 돌린다. Q6 결과는 "패치 트리·OFF" 범위. goodput 측정 boot는 손대지 않은 SGLang 그대로.
2. 서버 로그에 요청별 첫 토큰 시각이 없다 → 5-0의 히스토그램 증가분 방식(위).
3. 타이머 측정 boot의 부하(기준 용량의 80%)는 두 번째 Job 뒤에 정해진다 → 세 번째 Job으로.
4. Job 시작·끝 기준 셀을 "지속 용량"에서 돌리려 했으나 첫 번째 Job에는 그 점이 없다 → ~~SG-stock 버스트 사다리 rung~~ ★**rev3 정정: PM-ref(PD-mux D44) 버스트 사다리 rung 7.29491 req/s**(첫 번째 Job에 arm·trace hash까지 같은 셀이 실재)에서 PM-ref로 돌린다. SG-stock은 버스트 λ\*가 없어(첫 번째 Job `LADDER_BOTTOM_FAIL`) 이 규칙을 적용할 수 없다. rev2.1 문언 "없으면 `SUBSTRATE_UNCHECKED`"는 계획기와 어긋났다(계획기는 `refcell_rate`가 None이면 라벨을 붙이지 않고 계획을 거부한다, `ABORT_NO_ADDENDUM`) — rev3 문언은 **계획기 동작에 맞춘다**: 기준 셀 rate가 없으면 그 Job의 계획을 만들지 않는다. `SUBSTRATE_UNCHECKED`는 분석기에서 기준 셀 run이 첫 번째 Job의 같은 셀(같은 arm·trace hash)을 찾지 못할 때만 붙는다.
5. 지속 판정 "완료 ≥ 0.95 × 명목 도착"은 푸아송 도착 변동(창당 ±5%)만으로 건강한 서버가 탈락 → 분모를 **그 창에 실제로 도착한 수**로(명목값 병기). 두 번째 조건(마지막 창 TTFT 중앙 ≤ 첫 창의 2배) 유지. [rev2: 분자도 교체 — 도착 창 기준 E2E ≤ 60 s 완료, 600 s 이후 도착 제외, §3 지속 판정 규칙]
6. W3 floor·사다리의 Job 배정이 빠져 있었다 → 네 번째 Job.

## 6. 처치 발생 횟수 (규약)
- Q5(SG-mix) **처치 확인(rev2, 감사 r1 C)**: SG-mix boot의 서버 로그 server_args에 `enable_mixed_chunk=True` **이고** chunked prefill 활성(`chunked_prefill_size`가 정수 > 0; v0.5.10 `Scheduler.is_mixed_chunk = chunked_prefill_size is not None and enable_mixed_chunk`)이어야 유효, 아니면 `MIX_NOT_FIRED`. rev1의 "running ≥ 1인 Prefill batch 로그 줄 ≥ 50"은 `--enable-mixed-chunk`와 무관하게 출력되므로(v0.5.10 `scheduler_metrics_mixin.py:389–394`) 처치 확인이 아니다 → **"mixing 기회 수"**로 개명해 SG-mix와 **SG-stock(음성 대조)** 둘 다 run마다 기록·병기한다(판정 입력 아님). 실제 MIXED forward 횟수는 이 Job에서 세지 않는다(계측 없는 pristine 트리) — 공시. server_args 로그 패턴은 GPU 미검증.
- Q6(rev2, 2026-11-02 합성 창 반영): 계측 boot의 두 창을 **따로** 축약한다 — W1 부하 창(r_ser·r_mix 분포 p10/p50/p90, 길이 띠별 단일 요청 span 수: 기술값)과 합성 창(§5-1). **Q6_OK** = 다른 stream extend 0(두 창 모두) **이고** 합성 창의 예보 격자 L ∈ **{512, 2k}**(±25 %) 단일 요청 span이 각 ≥ 10 **이고** 단일 요청 span이 두 창 합계 ≥ 30(창별로 세어 따로 보고; 개수 조건일 뿐 r_ser 값은 섞지 않는다). 합성 창이 없으면 `Q6_NO_SYNTH_WINDOW`, 개수 미달이면 `Q6_FEW_SPANS`. 8k는 격자에서 삭제(감사 r1 C(ii)). **단일 요청 span** = 한 gap의 같은 stream extend 작업이 요청 1개(`n_reqs` 합 = 1)인 것 — 요청 2개 이상이 한 EXTEND에 묶인 span은 토큰 수가 띠 안이어도 세지 않는다. **P-4 채점 규칙**: **합성 창**의 L ≈ 2k 단일 요청 span만 쓴다(W1 창의 2k span은 개수가 많아도 채점에 쓰지 않는다); n < 10이면 `UNSCORABLE`(채점 금지, 값도 보고하지 않음). 예보 셀 = (L 2048, bs = 그 span들의 decode bs 중앙값에 가장 가까운 격자 bs, ctx 2048). 512는 같은 창·같은 방식의 **기술값**(예보 대비 잔차 병기, 채점 아님). **다른 stream extend**: 어느 창이든 gap 하나라도 `other_stream_ms` > 0이면 Q6 전체 `Q6_INVALID:OTHER_STREAM_EXTEND`(stock 단일 forward stream 전제 위반, fail-closed) → P-4도 `UNSCORABLE`. **처치 발생 횟수(기대)**: 합성 창 단일 요청 span = 512 10개·2k 10개(설계값; 선행 요청 1개는 창의 첫 gap이라 제외), W1 창은 512 ≈ 2.2·2k ≈ 1.4(60요청 W1 길이 몫에서, 기술값). 실제 개수는 run마다 창별로 기록한다. ★**범위**: 합성 창 프롬프트는 W1 decode가 다 끝난 뒤 혼자 들어가므로 그 r_ser는 decode bs ≈ 1(자기 decode만) 조건의 값이다 — W1 부하 중 정지(bs 큼)의 r_ser와 같은 셀이 아니며, P-4는 bs ≈ 1 예보 셀과만 비교한다. 이 범위 밖으로 P-4 결과를 옮겨 말하지 않는다.

## 7. 예산·Job 분할 [유도, 시간 모형 `sb_spec`; 계획값 = `sb_spec.PLAN_LAMBDA`(버스트 6.0 · 기준 5.0 · W3 0.3 req/s)]
**계획값의 근거(rev2 정정)**: rev1의 "기준 용량 3.0 req/s 가정"은 근거가 틀렸다 — CAP2 지속성 문서의 "완료 처리율 ≈ 3.0"은 감사 사본 스크립트의 단위 오류였고, 올바른 값은 **과부하 포화 완료율 5.06–5.34 req/s**다(`results/policy_cap2/RECHECK_sustain_throughput_2026-11-02.md`; goodput 아님, SLO를 지키는 지속 rate는 미정). 시간 모형의 기준 계획값 **5.0은 추정**이고 근거는 이 포화 완료율의 하한 근방이다. **SLO를 지키는 지속 rate(= 기준 용량)는 두 번째 Job이 정한다.**

Job 4개 + 아래쪽 사다리 1개(각 ≲4 h 규약, 하드캡 = 추정 × 1.25, 5분 올림). `python3 sb_spec.py <Job>` 출력(2026-11-04, rev3.2 하네스; 두 번째 Job은 첫 번째 Job 실제 addendum으로 계산; 세 번째 Job은 `PM-rep` 포함, 네 번째 Job은 rev3.2에서 `PM-rep` 제거):

| Job | 내용 | boot | 추정 | 하드캡 |
|---|---|---|---|---|
| 첫 번째(실행 완료, rev2.1 봉인 바이트) | 시간 기준 확인(a) smoke 2 + 용량 사다리(5 arm × 10단 × seed 2; PM-ref34는 CAP1-R D34 용량 6.343 승계, 기술값) | 7 | 206.0 min | 260 |
| 아래쪽 사다리 "J1-down"(rev3, 독립·순서 자유) | SG 3 arm × 4단(4.796522–3.153791) × seed 2 × 120요청 (≈ 3 × 4 × 2 × 1.6 + boot 3 × 6) | 3 | 56.4 min | 75 |
| 두 번째 | 지속 셀(rev3: vLLM·PD-mux 2 arm × {100 %, 80 %} × 600 s, 조건부 50 % 1점 최악 포함; rate 8.389146·6.711317 / 7.29491·5.835928, 조건부 4.194573 / 3.647455) + 기준 셀 시작·끝(PM-ref 7.29491) | 4 | 90.8 min | 115 |
| 세 번째 | 정상상태 50 %·80 % 셀(rev3.1: 8 arm × seed 4, ABBA — `PM-rep` boot 2개·셀당 run 4개 추가) + 계측 boot 3(SG-stock 계측 boot에 합성 프롬프트 창 +1.0 min 포함) + 기준 셀 | 21 | 240.9 min(기준 5.0) · 231.1(5.836) · 219.6(7.295) · 267.2(3.647) | 305 · 290 · 275 · 335 |
| 네 번째 | W3 floor(rev3: SG-stock W3 사다리 boot 안) + SG-stock W3 사다리 + R3(rev3.2: 4 arm × seed 4, ABBA — rev3.1의 `PM-rep` boot 2개·run 4개 제거) + 기준 셀 | 11 | 254.5 min(W3 0.3) | 320 |

[유도] rev3.1 `PM-rep` 추가분(시간 모형): 세 번째 Job +25.4 min(기준 5.0; 5.836 → +24.2 · 7.295 → +22.8 · 3.647 → +28.6) = boot 2개 × 6.0 min 오버헤드 + 200요청 run 8개; 네 번째 Job +43.4 min(W3 0.3) = boot 2개 × 6.0 + 96요청 run 4개(0.24 req/s에서 run당 ≈ 7.9 min) — **rev3.2에서 네 번째 Job 몫은 제거**(네 번째 Job = rev3 값 254.5 min, 하드캡 320). rev3 값은 세 번째 215.5(5.0)·206.9·196.8·238.6, 네 번째 254.5. 시간 모형은 첫 번째 Job 실측보다 약 2.7배 보수적이다(감사 rev3 r1).

[유도] rev3 세 번째 Job의 기준 용량은 vLLM·PD-mux 지속 용량 최소라 값은 두 번째 Job 회수 뒤 정해지며, 지속 격자상 가능한 값은 {1.0, 0.8, 0.5} × {8.389146, 7.29491}의 원소 중 하나다(최소). 위 열의 3.647·5.836·7.295는 PM-ref 쪽 세 격자점, 5.0은 rev2 계획값이다.

- 두 번째 Job의 지속 run 길이는 rate와 무관(600 s 고정 + drain)이라 추정은 λ\*에 거의 의존하지 않는다. rev3에서 arm이 5개 → 2개로 줄어 204.2 → 90.8 min.
- ★**정상상태 Job(세 번째 Job — 감사 r1이 "두 번째 Job"이라 부른 278 min 사례는 시간 모형상 이 Job이다)의 초과 가능성 공시(rev2)**: run 길이 = 200요청 ÷ (0.5·0.8 × 기준 용량)이라 기준 용량이 낮을수록 길어진다 — 기준 5.0 → 215.5 min(캡 270) · 3.0 → 257.4 min(캡 325) · **2.5 → 279.1 min(캡 350)** · 2.0 → 309.3 min(캡 390)(2026-11-02 합성 창 반영, 각 +1.0 min). 기준 용량 2.5 이하이면 규약 "≲4 h"를 넘는다. ★rev3.1(`PM-rep` 포함): 5.0 → 240.9 min(캡 305) · 3.647 → 267.2(캡 335) · 3.0 → 288.6(캡 365) · 2.5 → 313.3(캡 395) · 2.0 → 347.7(캡 435) — 기준 용량 약 5.0 아래이면 "≲4 h"를 넘는다(두 번째 Job 지속 격자상 가능한 값 중 PM-ref 쪽 3.647은 넘는다). 하드캡은 회수된 기준 용량으로 Job 제출 시점에 다시 계산되므로 캡에 잘리지는 않지만, 4 h 초과는 그 시점 dry-run 출력으로 공시하고 분할 여부는 사용자 결정으로 넘긴다(쌍은 같은 Job 유지 조건).
- 네 번째 Job은 규약 "≲4 h"를 넘는다(rev3.2 약 15분 — 시간 모형 254.5 min, rev3과 같음; ~~rev3.1 `PM-rep` 포함 약 58분, 297.9 min~~). 분할하면 기준 셀 boot 2개(≈17 min)가 더 들고 W3 용량(R3 rate)이 Job을 가로지르므로 **분할하지 않고 초과를 공시**한다.
- ★**네 번째 Job 예산 가드(rev2, 감사 r1 ⑨)**: W3 용량은 Job 안 사다리로 정해지므로 R3 시간은 제출 때 모른다(W3 용량 0.225 → 총 ≈ 290 min · 0.15 → ≈ 362 · 0.1 → ≈ 468; 하드캡 320 — rev3.2 = 이 값; ~~rev3.1 `PM-rep` 포함: 0.225 → ≈ 343 · 0.15 → ≈ 432 · 0.1 → ≈ 565; 하드캡 375~~). 그래서 **W3 사다리 직후 한 번**, 남은 시간(= 하드캡 − 경과)을 "R3 boot 전부(채워진 rate의 시간 모형 추정) + 그 뒤 boot(끝 기준 셀)"의 추정과 비교해 부족하면 **R3 boot 전체를 건너뛰고** 각 boot `DONE.json`에 `R3_SKIPPED_BUDGET`을 기록한다(`R3_BUDGET.json`에 필요·남은 시간 기록). 결정은 R3 첫 boot 전에 한 번만 내려 **반쪽 ABBA는 생기지 않는다.** 시간 모형상 W3 용량 ≲ 0.15 req/s이면(rev3.2 = rev3 같은 계산: 총 추정이 하드캡 320을 넘는 점 ≈ 0.186 req/s; ~~rev3.1 ≈ 0.19, 하드캡 375~~) R3(Q7)는 생략된다 — R3 boot 전부가 한꺼번에 생략된다 — 그때 Q7은 "측정하지 않은 것"에 NOT_MEASURED로 남고, 재실행은 별도 Job(사용자 결정).
- rev0의 변화 trace(≈659 min)는 제거. 제출은 `PDMUX_VESSL_ENV=scripts/vessl/vessl.sbase.env`(합본 이미지 digest)로 `launch.sh`를 그대로 쓴다. 각 쌍의 두 arm은 항상 같은 Job·같은 GPU. 사용률·`$/h`는 dry-run 출력 그대로(과금 서술은 보고에 넣지 않는다).

## 8. 사전 예측 (봉인 대상; 채점만, 종합 판정 없음)
- P-1 λ\*_SG-stock(W1, 버스트) ∈ [6, 10] req/s [유도: PD-mux D44 7.29와 M3 J1 A LO req/s 동률에서]. P-2 λ\*_VL-stock / λ\*_SG-stock ∈ [0.8, 1.25]. P-3 R₂에서 PM-ref vs SG-stock goodput ln 비 부호 **+**(KISTI 방향) — 단 M3 J1 shape B 모양(PM TTFT ≫)이 n≥4에서 재현되면 **−** [두 방향 모두 거짓이 될 수 있음 → N3 아님]. P-4 r_ser(L=2k, SG-stock) 예보 = `R_SER_FORECAST_NANO.csv` 해당 셀 ±30% [2026-11-02: 계측 boot 합성 창의 단일 요청 2k span으로 채점, 셀 = 그 span들의 decode bs(설계상 ≈ 1) — §6]. P-5 SG-mix ITL p95 < SG-stock ITL p95(정지가 MIXED 안으로 옮겨감).
- ~~rev3 추가 예측 행 Q1-SG~~ **rev3.1에서 삭제**(감사 rev3 r1 처방: 띠 [2.5, 4.8]이 아래쪽 사다리 rung 4개를 전부 담아 거짓이 될 결과가 없었다; 띠는 다시 고르지 않는다). 아래쪽 사다리의 SG-stock 버스트 λ\*는 Q1 기술값으로만 보고한다. P-1(띠 [6, 10])은 rev2.1 규칙 그대로 **첫 번째 Job 사다리 값으로만** 채점한다(분석기 출력 None — 아래쪽 값으로 다시 채점하지 않는다).
- **rev2 추가 예측(채점 아님, 결과 해석의 사전 고지)**: (i) 80 % 셀에서도 best arm attainment ≈ 1.0이 가능하다(CAP1-R 5.516·6.343에서 PD-mux attainment 1.00) → 50 %·80 % 두 셀 모두에서 **`BOTH_SATURATED`가 예상 결과일 수 있다**. 그 경우 Q2–Q4의 goodput 라벨은 나오지 않으며 이것은 실패가 아니라 "기준 용량 아래에서는 goodput으로 갈리지 않는다"는 결과다; Q8·Q5는 지연 공동 지표로 읽는다. 80 % 셀의 PM-ref 대 SG-stock 쌍이 `BOTH_SATURATED`이면 P-3은 `UNSCORABLE`(사유 `BOTH_SATURATED`)이다 — 포화 셀의 ln 비는 drain 차이뿐이라 부호로 채점하지 않는다(재감사 r2, 사전 예측 문서 §5와 같은 판정). (ii) 같은 이유로 80 % 셀의 `CELL_OUT_OF_BAND`(전 arm > 0.9)도 예상 가능하다. (iii) P-4는 계측 boot 합성 창의 512·2k 각 10개로 채점 가능하다; 채점 대상인 2k 단일 요청 span이 10개에서 하나라도 빠지면 `UNSCORABLE`이다(여유 0; 512는 기술값이라 512가 빠지면 Q6 상태만 `Q6_FEW_SPANS`, §6).

## 9. 허용/금지 문언
- 허용 형식: "VESSL A100 ×1 · Nano-9B-v2 · 이 Job에서, 셀 R(절대 rate x) arm A 대 B의 seed 4쌍 goodput ln 비는 평균 m, 95% CI [lo, hi], 라벨 L(유효성 사슬 통과/사유)."
- **인용 금지 문장(하네스 감사 r1, 결과 문서에 문자 그대로 승계)**:
  1. "이 캠페인의 PD-mux 대 stock 라벨은 PM-ref 구성 묶음(D44 또는 D34, mem 0.6, overlap OFF, chunked prefill 끔, cap 48, budget 65536, 모두 KISTI 승계) 대 SGLang 기본값의 비교이며, PD-mux 단독 효과도 static PD-mux 최적도 아니다."
  2. "판정 셀 rate는 다른 Job(다른 GPU)에서 잰 지속 용량에서 왔고, 지속 점 격자 {1.0, 0.8, 0.5}가 최솟값 arm을 정할 수 있으므로, 셀의 상대 부하는 arm·Job마다 다르다. 기준 셀 비교는 용량 경계 rung에서 잰 것이라 기판 동일성의 증거가 아니다."
  3. "두 엔진의 TTFT 시간 기준은 저부하 셀에서만 확인됐고 ITL 측정의 동등성은 확인되지 않았다. 엔진 간 쌍의 EQUIV는 n=4에서 사실상 도달 불가다(sd 예보 0.02 > 0.0186)."
  4. "Q8 결과는 이 셀의 goodput·지연 범위에서만 성립하며, 수정 트리 OFF가 커널 수준에서 upstream과 같다는 근거가 아니다(causal_conv1d int64는 코드상 다름)."
  5. "W3 셀 rate는 SG-stock 용량만으로 정했으므로, PD-mux arm이 그 rate에서 용량 위였다면 W3 라벨은 부호만 인용하고 크기는 인용하지 않는다. r_ser·T_dec는 수정 트리·플래그 OFF 계측 boot 범위다."
- **인용 금지 문장(하네스 재감사 r2, 결과 문서에 문자 그대로 승계)**:
  6. "지속 용량은 'E2E ≤ 60 s 완료 ≥ 0.95 ∧ TTFT 중앙 증가 ≤ 2배'로 정한 안정성 경계이며 SLO goodput 용량이 아니다. 이 규칙은 용량 아래 모형과 명백한 과부하 raw 두 극에서만 검증됐고, 지속 격자 {1.0, 0.8, 0.5} 위의 값만 가질 수 있다."
  7. "어느 seed든 attainment 0.971–0.98에서 나온 EQUIV_WITHIN_3PCT는 attainment 차가 3% 안에 묶여 거의 구조적으로 나오므로 동등성 근거가 아니다."
  8. "지연 공동 지표 라벨에는 지연 플라시보가 적용되지 않았다. boot 간 지연 잡음의 크기는 판정에 들어가지 않는다."
  9. "사전 예측 문서에서 P-1–P-5 밖의 행은 추정량과 arm이 등록되지 않은 사후 채점이므로 사전등록 적중으로 인용하지 않는다."
- **인용 금지 문장(rev3, 사용자 결정 2026-11-03 — 공정성 범위, 결과 문서에 문자 그대로 승계)**:
  10. "손대지 않은 SGLang은 이 이미지(py3.14)에서 piecewise CUDA graph를 켤 수 없어 prefill이 eager로 돈다(B2 부팅 실패). vLLM 0.31.0은 piecewise CUDA graph ON이다. 따라서 vLLM 대 SGLang 비교와 SGLang의 도착당 정지(~42 ms)에는 'SGLang eager prefill' 환경 범위가 붙으며, 정지 길이 중 eager 발행 비용의 몫은 분리되지 않았다."
  11. "손대지 않은 SGLang·패치 OFF·mixed-chunk의 버스트 용량(SLO TTFT 3 s ∧ ITL p95 60 ms)은 5.516 req/s 이상 격자에 없다(첫 번째 Job, ITL 축 절벽)"
- **인용 금지 문장(하네스 제한 감사 rev3 r1, rev3.1 — 결과 문서에 문자 그대로 승계)**:
  13. "판정 셀 rate는 첫 번째 Job에서 손대지 않은 SGLang의 실패를 본 뒤 정한 기준 용량 정의(rev3, SG 제외)에서 나왔으므로, 80% 셀 rate가 5.516 req/s 이상이면 P-3의 HIT(+)는 첫 번째 Job 데이터로 이미 정해진 결과이며 사전등록 적중으로 인용하지 않는다."
  14. "Q1의 SGLang 버스트 λ\*는 첫 번째 Job(GPU bc0be109)의 5.516 이상 점과, 기준 셀이 없는 아래쪽 사다리 Job(다른 GPU일 수 있음)의 점을 한 pick 규칙으로 합친 값이며, 아래쪽 rung은 첫 번째 Job 결과를 본 뒤 정했다."
  15. "rev3 기준 셀(PM-ref 7.29491 req/s)은 PM-ref 용량 쪽 기판 저하(약 10% 이상)만 잡고, 빨라진 기판(최대 차 0.033)과 손대지 않은 SGLang의 도착당 정지 길이 이동(PM-ref 간극 45 ms는 SLO에서 25% 떨어져 있음)은 잡지 못한다."
  - 검출력 공시: "판정 셀의 플라시보(SG-rep 대 SG-stock)는 SG가 자기 ITL 용량 위에 놓일 수 있는 rate에서 잰다. 첫 번째 Job proxy sd 0.113은 등록 예보 0.006의 19배이므로, BOOT_NOISE 셀은 기판 잡음의 증거로 인용하지 않는다."
  - Q4 caveat 5개 = 상한. 하나 더 붙으면 Q4는 정본에 올리지 않고 설계 재작성(감사 r1).
- **인용 금지 문장(하네스 재감사 rev3 r2, rev3.2 — 결과 문서에 문자 그대로 승계)**:
  16. "VL-stock과 PM-ref34의 boot 간 잡음은 이 캠페인 어디에서도 재지 않았다. Q3(vLLM 대 SGLang)은 SGLang 쪽 플라시보만, Q4(PD-mux 대 vLLM)는 PD-mux D44 쪽 플라시보(PM-rep 대 PM-ref)만 통과 조건으로 쓰므로, Q3·Q4의 부호·라벨에는 vLLM boot 하나의 우연이 섞일 수 있고, PM-ref34 쌍은 D44 복제로 대신 검사됐다."
  - 검출력 공시: "첫 번째 Job에서 PM-ref와 VL-stock은 판정 셀 rate 범위(5.836 req/s 이하)에서 두 seed 모두 attainment 1.00이었으므로, Q4는 두 W1 셀 모두 BOTH_SATURATED가 예상 결과다. PD-mux 플라시보 sd 예보는 등록되지 않았다."
- 금지: "PD-mux가 vLLM을 이긴다/진다"(범위 없는 일반화) · "최강 baseline" (조율 arm 없음) · KISTI 행과 섞은 비교 · 계측 boot 수치를 goodput 근거로 · D44 결과를 "static PD-mux 최적" 으로 · 과부하 셀(CELL_OUT_OF_BAND) 크기 인용 · A5 예보를 측정값처럼 · `BOTH_SATURATED` 쌍의 `raw_label`·ln 비를 동등성 근거로 · "mixing 기회 수"를 mixed-chunk 발화 횟수로 · (rev3) Q3·Q5·Q8 쌍(rev3.1 + Q2)의 goodput % 크기·`raw_label`·ln 비(`MAGNITUDE_NOT_CITABLE`) · (rev3) 아래쪽 사다리 값(Q1)을 기준 용량·판정 셀 rate의 근거로.

## 10. 잃는 범위
조율 fused · Llama-8B · 동적 PD-mux · best D 재결정 · W3 사다리(SG-stock 외 arm) · 과부하 구간 · **부하 변화(LO↔HI) 반응** · **vLLM의 prefill/decode 시간비(r_mix; 프로파일러 창 미구현, §5-2)** · prefix caching ON · TP>1 · (rev3) **SG-stock·SG-off·SG-mix의 지속 용량**(지속 셀 미실행 — 기준 용량에서 제외) · (rev3) **5.516 req/s 이상에서 SG 3 arm의 goodput 크기**(Q3·Q5·Q8은 부호만; rev3.1: Q2도 부호만 — PD-mux 대 손대지 않은 SGLang의 goodput 크기) · (rev3) 아래쪽 사다리 바닥 3.153791 아래 rate.

## 11. 선행 작업 (GPU 전; 전부 끝나야 봉인)
1. **vLLM 이미지**: `env/docker/Dockerfile.vllm-runtime`(pristine vLLM 0.31.0 + 같은 base) 작성·빌드·push, digest를 `vessl.env`에 `PDMUX_VESSL_IMAGE_VLLM`으로. Workspace bring-up 1회(측정 아님, **비용 행위 → 사용자 승인**): Nano 체크포인트 부팅 + 고정 프롬프트 출력 비교 가드 + 선택된 백엔드/CUDAGraphMode 로그 채취.
2. 요청별 send/first-token/finish 시각을 쓰는 클라이언트 경로 확인(`bench_serving` 호환, 두 엔진 공통).
3. `decode_probe` EXTEND 시간 플래그(engine-porter, 진행 중) + CPU 회귀 + 변이 테스트 → digest 갱신.
4. 부하 창 필터 축약 함수 + 변이 테스트(5-1·5-3).
5. A5 예보 파일 sha256 봉인.
6. 하네스 감사 1회(규칙층 반전 시험 + 파이프라인 모집단 층 표).
7. ~~pristine 트리 boot 모드~~ **완료(CPU, 2026-11-01)**: `install_runtime.sh` step 0이 이미지 트리를 `sglang_pristine/`에 보존(커밋된 참조 manifest `env/sglang_v0.5.10_python_tree.sha256` 2257항목·sha `1bd84d88…`와 바이트 대조, 마커 가드), 서버 boot는 `engine_tree.py boot --stamp <json> -- <server cmd>`로 `PDMUX_ENGINE_TREE=pristine|patched` 선택(pristine은 `PDMUX_*` 전부 제거·`--enable-pdmux` 거부·`find_spec` 해석 위치 검사). 테스트 `test_engine_tree` 21 + `test_job_entry_cpu.sh` 29 통과. **GPU 이미지에서 미검증**(pip -e 잔여물로 보존 FAIL 가능 → `PDMUX_REQUIRE_PRISTINE=1`로 fail-closed). 진단 표는 부록 D.

## 12. 이 Job이 채우는 COSTMODEL 칸
CM-6(대기열·TTFT: stock 구조의 정지 r_ser(L) 실측, 예보 대비 잔차) · CM-1(T_dec 점 추가, stock 경로) · CM-2(r_ser 분자 = EXTEND GPU 시간) · 확장 열 "엔진 대조" 신설 후보.

## 13. 사용자 결정 대기 (봉인 전)
1. 용량 기준량 보고 방식: 둘 다(권고) / 버스트만.
2. vLLM 0.31.0 고정(권고) / 봉인 시점 재확인.
3. 판정 셀 rate를 λ\*_ref = min_arm 기준 공통 절대 rate로(권고) / arm별 자기 λ\* 상대 rate.
4. `PM-ref34` 추가(권고: 고원 안 두 점) / D44만.
5. Job 분할 3개(권고) / 2개(W3 제외).
6. 선행 1(vLLM 이미지 + Workspace bring-up 1회) 승인. → **승인·진행 중(2026-11-01)**
7. `SG-off` arm 추가(권고: Q8은 implfix·M3 J1 수치의 범위를 정하는 데 필요) / 미추가.

**rev0 → rev1 전 추가 결정(2026-11-02, `REPORT_sb_harness_design_2026-11-01.md` §3·§4)**: ⑧ vLLM 배치 — (a) 별도 Job / **(b′) `sglang-runtime` + `/opt/vllm-venv` 새 공개 이미지(권고)** / (b) Job 안 격리 venv / (c) 비권고 — J1b 전 결정 필수 · ⑨ J3 시간 ≈670 min(상한 초과) — J3a(W1 trace, ≈8.6 h → 또 분할) / trace 라운드 3→2(게이트 2 이탈 공시) / W3를 J4로 분리 / 조합 — 권고: **W3를 J4로 분리 + W1 trace를 arm 절반씩 J3a·J3b(쌍은 같은 Job 유지)** · ⑩ 하네스 편차 D-1…D-6·해소 O-2…O-14 승인(권고: 전부 승인, D-1·O-10은 범위 문구로 공시).

rev3.1 결정(2026-11-04): (b) — Q2 부호 전용 + Q1-SG 예측 행 삭제 + PD-mux 플라시보 arm `PM-rep` 추가(K3 base arm별), 재봉인 후 실행.

rev3.2(2026-11-04): 권고 1 — 50 %·80 % 셀 분리 변이 가드 + 네 번째 Job W3 셀에서 `PM-rep` 제거(PM 플라시보 요구는 R1·R2만), 재감사 rev3 r2 인용 금지 문장 16·검출력 공시 승계.

확정(2026-11-01, 사용자 "진행" = 권고 채택): ① 둘 다 보고·판정 셀 rate는 **지속 기준** · ② vLLM 0.31.0 · ③ 공통 절대 rate · ④ `PM-ref34` 추가 · ⑤ Job 3분할 · ⑥ 승인(Workspace create 재시도 허용) · ⑦ `SG-off` 추가. → §13은 닫혔고 남은 봉인 조건은 §11 선행 작업 + 하네스 감사 1회.

## 부록 D — 수정 트리에서 플래그 OFF일 때 upstream과 달라지는 실행 경로 (engine-porter 진단, 2026-11-01, 코드 판독·GPU 미측정)

기준: `git archive v0.5.10` + HEAD `install_runtime.sh` → 수정 파일 15개(+3071/−55), 새 파일 11개. 줄 번호는 수정 트리.

| 파일 | OFF에서 발화 | 성격 | 근거 줄 |
|---|---|---|---|
| `layers/attention/mamba/causal_conv1d_triton.py` | **발화, 커널 수준** — int64 캐스트 2곳. triton conv 경로를 타는 모든 Mamba2 prefill(NemotronH xBC 포함). int32 범위에서 값 동일, 컴파일된 커널이 다름 → 성능 미측정 | GPU | 69, 77 |
| `models/nemotron_h.py` | 발화 — 매 forward env 3–4회 읽기·`current_stream()`·층별 `nullcontext()`; env 없으면 같은 층 순서·같은 커널. decode cudagraph에서는 capture 때만 | host | 645–737, 608, 859 |
| `models/granitemoehybrid.py` | 같은 패턴 | host | 455–497 |
| `distributed/parallel_state.py` | 매 `get_tp_group()`에서 `ContextVar.get()`(값 항상 False) | host | 1439–1454 |
| `managers/scheduler.py` | 프로브 import·`_holb is not None` 검사 2회/run_batch(env 없으면 None) | host | 189, 292, 440, 458, 2708, 2762, 3539, 3671, 3681 |
| `model_executor/model_runner.py` | ngram `getattr` 동치, Zamba2 isinstance, coord 메서드는 pdmux 전용 | 없음 | 2407, 1862 |
| `utils/hf_transformers_utils.py`·`configs/__init__.py` | import 시 mamba2/zamba2 전역 등록 | 없음(모델 종류 한정) | 122, 142–145 |
| `layers/attention/triton_backend.py` | `--attention-backend triton` + mamba 계열만(flashinfer 미발화) | 조건부 | 98 |
| `memory_pool.py`·`model_runner_kv_cache_mixin.py`·`server_args.py` | 순수 SSM / `Mamba2ForCausalLM`만 | 조건부 | 1420, 184, 1919 |
| `falcon_h1.py`·`multiplexing_mixin.py`·`pdmux_context.py` | OFF 변화 없음(새 메서드·상수·pdmux 전용) | 없음 | — |

**함의**: "수정 트리 OFF = upstream"은 **커널 수준에서 성립하지 않는다**(causal_conv1d int64). implfix C6의 CPU golden은 프로브 None 경로만 덮는다. 이 차이를 GPU에서 처음 재는 것이 Q8(`SG-off vs SG-stock`)이며, 그 전까지 M3 J1 fused·CAP/TUNE/EVAL OFF 참조·KISTI `plain`은 전부 "수정 트리·OFF" 범위다. 그 밖의 공시: 두 트리는 site-packages(sgl_kernel·flashinfer·triton·torch·py3.14 shim)를 공유한다 — "pristine"은 SGLang 파이썬 소스 한정; triton/flashinfer JIT 캐시 공유(정확성 영향 없음, warmup 순서 차이만); pip 메타데이터는 수정 트리를 가리키므로 선택은 PYTHONPATH 우선순위 + boot 시 해석 위치 검사에 의존.
