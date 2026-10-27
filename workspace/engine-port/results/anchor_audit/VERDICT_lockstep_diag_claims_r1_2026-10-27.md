# 감사 판정 r1 — lockstep 정지 진단 파생 주장 (P1–P5)

- 대상: `results/lockstep_diag/DIAG_lockstep_stall_share_2026-10-27.md`(미감사 진단)와 그로부터 나온 제안 문장 · 감사자: claims-auditor(읽기 전용) · 일자 2026-10-27
- 회차: P1–P4 1/2 · P5(C 서술)는 2/2 → **종결**

## 판정 요약

| 항목 | 판정 | 핵심 confound | promote |
|---|---|---|---|
| P1 진단 자체 | PLAUSIBLE(조건부) — 범위 내 기술값으로 성립, 반증 실패 | 11(기판·모델·backend 교락), 3(N3 Job 2개·단일 모델), 관측자(全 run PHASE ON), 전력(ρ) | 가(CM-4 기술값, caveat 승계) |
| P2 부정 결론 재범위화 | 모델·기판 범위 재진술 CONFIRMED(이미 정본 범위) · "정지 기판 범위"로 귀속 NOT-YET-SUPPORTED · layer-type 死에 정지 의존성 부여 REFUTED | 11, 9, 2 | 귀속 문안 불가, 범위 재진술만 가 |
| P3 COMPARE §2.2 "ITL ≈ span" | VESSL·Nano 범위 REFUTED · KISTI 범위 NOT-YET-SUPPORTED | 7류(코드 독해 유도값을 서빙 기전으로), 11 | 가(메모 정정 배너) |
| P4 C2 꼬리 인용 정지 | 인용 정지 CONFIRMED(예방적) · 근거 라벨 "정지 오염" NOT-YET-SUPPORTED | 3, 기전 미관측(PHASE 0) | 가(라벨 교체) |
| P5 C 서술 | 허용 문안 초과 없음, 출처 누락 2건 | 다른 축 판단 혼입 | 가(문구 수정), 종결 |

## P1
반증 시도 실패: phase 방출 코드(`multiplexing_mixin.py:2129-2302`)에 synchronize·query·`.item()` 없음 · client ITL과 서버 분위 차 ≤1.7 ms · CHUNK/WAIT 간격 20.61 대 20.56 ms.
남는 한계: (1) 관측자 — 全 run PHASE ON, ρ_on 1.156–1.193은 동거·전력 한도 클럭·관측자 합성값(§1-38 금지 32–42 승계) (2) 선택 편향 — Nano·flashinfer·65536·8k/32k·저부하뿐, KISTI 설정(ShareGPT ~2k, triton) 0셀 (3) N3 Job 3 첫 chunk host 차단 0.66–1.34 s, Job 4 0건, 원인 미해결 (4) 진단 꼬리 = 풀링 토큰 분위, goodput 술어(요청별 ITL p95)에 직접 말하지 않음 (5) 기울기 0.0015 CI [0.0005, 0.0027]는 0 배제 → "x 157–428 ms에서 ≤0.7 ms"로 (6) adjust 경로 drain(`:1344`, `:1361`) 미계측.

허용:
> "VESSL A100-SXM4 · Nemotron-Nano-9B-v2-Base · flashinfer · cudagraph ON · split 예산 65536 · 프롬프트 8k(T3는 32k 포함) · PHASE_EVENTS ON 조건의 N3(Job 2개)·N2-EA·T3 run에서, 동거 decode 간격은 span 시간(W/n_chunks)에 거의 반응하지 않았고(기울기 0.0015, 95% CI [0.0005, 0.0027]) 풀링 토큰 ITL p99 꼬리는 동거 decode step 수준이었다. 이 범위에서 CM-4의 별도 정지 항은 해상도(≤1 ms) 안에서 관측되지 않았다. ρ_on 1.156–1.193은 동거·전력 한도 클럭·관측자 비용의 합성값이다. 한 Job(N3 3회차)에서 0.66–1.34 s 첫-chunk host 차단이 드물게 관측됐고 원인은 미해결이다."

금지: "VESSL 기판에는 lockstep 정지가 없다"(한정 생략) · "L_stall = 0" · "ρ_on ≈ 1.17이 동거 간섭의 크기다" · "정지가 goodput에 영향 없다" · 이 수치를 KISTI 수치와 비교·결합.

## P2
- layer-type 死(§1-3·§1-15): TPOT 42→124 ms는 `event_loop_pdmux_coord`의 layer 경계 drain + 매 step `prev_p.synchronize()`(`:1728-1730`) = 정책 구현의 동기화이지 메인 루프 기판 정지가 아니다. cudagraph 비양립은 기판 무관 구조 논거. 같은 기판 두 arm이라 정지가 있었다면 공통 모드. → "정지 기판 범위로 재서술"은 과소 주장. 기존 KISTI·triton 스코프 배너로 충분.
- HE0·decode-heavy 기본값: phase 데이터 0 → 정지 의존성 유무 모두 증거 없음. TTFT 경로(≥95% 공유 대기열)는 산술 확정, VESSL·Nano에서도 TTFT 77–89%가 직렬 split-prefill 대기(정황, 기판 달라 증거 아님). 정지가 HE0를 바꿀 유일 경로 = **arm별로 다른 drain**(동적 arm은 전환마다 진행 중 prefill drain, sticky static은 없음). 미측정. switch 비용 s ≤ 0.04 ms(§1-34)는 이 경로로 옮겨 쓸 수 없음 — 열린 구멍(반증 아님). "KISTI·Zamba2 정지 기판"은 정지와 모델·backend가 교락(KISTI Hs8 flashinfer는 정지 드묾, 게이트 #83) → 기판 귀속은 과대 주장.

허용:
> "HE0·decode-heavy static 권고는 KISTI A100 · Zamba2-2.7B · triton · 변화 trace 범위의 결론이며 VESSL·Nano로의 전이는 미검증이다. 그 기판에서 lockstep 정지가 존재했는지와 HE0 기전에 관여했는지는 phase 계측 0으로 판단 불가다."
> "layer-type 死의 근거(coordinated TPOT 42→124 ms, cudagraph 비양립)는 정책 구현 자체의 layer 경계 동기화이며 기판 lockstep 정지와 독립이다."

금지: "HE0은 lockstep 정지 기판의 산물이다" · "정지 없는 기판에서는 동적 제어가 이길 수 있다"(시험 전) · "layer-type 死는 KISTI 정지 기판 한정이다" · "VESSL·Nano에서 HE0이 뒤집힌다/유지된다."

## P3
메모는 "decode 1 step → span 1개 → decode sync"를 "decode가 span GPU 완료를 기다린다"로 읽었으나, 매 반복 동기화는 decode stream뿐이고 데이터상 span당 decode 토큰이 chunk 수의 2–9배.
정정 배너:
> "§2.2의 'lockstep에서 ITL ≈ span 시간(decode step의 10–20배)'은 코드 독해에서 나온 유도값이며, VESSL·Nano-9B·flashinfer·예산 65536 서빙 telemetry에서는 성립하지 않았다(동거 간격 ≈ decode step × 약 1.16, host가 chunk를 선발행). §1-24의 KISTI Ha8 정지와의 기전 동일성은 미확인이다."
금지: "현 구현은 매 반복 lockstep이라 ITL이 span에 묶인다"(범위 없이) · "§1-24 정지 = 이 유도의 실증."

## P4
재구현 채택 규칙은 batch 매칭 없는 셀 전체 분포라 headline과 같은 양이 아님 · ">3×p50 = 정지"는 phase 없이 붙인 기전 라벨 · p50 무영향은 산술 견고(이상치 ≤1.2%).
허용:
> "C2 headline은 p50(2.36–2.91×)이며, 채택 집합의 >3×p50 간격 비중(≤1.2%)으로 보아 p50은 이 꼬리의 영향을 받지 않는다. C2 채택 집합의 p95/p99는 기원 미확정의 장꼬리(0.7–1.8 s 간격 포함)를 담고 있어 인용 정지한다."
금지: "C2 꼬리는 lockstep 정지에 오염됐다" · "C2 p50은 정지 없는 decode step 시간이다" · Job 865493 d24–d92 다리 인용.

## P5 (2회째, 종결)
상태 문안("미검증·시험 가능", "offline 프로파일 최소-SM 서빙 측정 0")은 판정서 허용 문장과 일치. "DuetServe와 같은 형태", "신규성은 hybrid 귀속뿐"은 판정서 밖(venue-strategist 웹 판단) → 출처 표기 필요. 판정서가 CONFIRMED한 hybrid 귀속 장애(A4 재매개화 항등식) 병기 누락.
수정 문안:
> "…DuetServe(ICML'26, venue-strategist 웹 확인, 재현 0)가 transformer에서 같은 형태라는 문헌 판단상 C의 차별점은 hybrid 귀속에 달려 있고, 그 귀속은 현재 A4 재매개화 항등식 장애가 있다(감사 CONFIRMED)."
금지: "C는 DuetServe의 hybrid판일 뿐이다" · 판정서 §금지 6항.

## 해소 실험 (다음 GPU Job, 진단 Job 1개, 정책 판정 0)
- arm: {Nano-9B/flashinfer, Zamba2-2.7B/triton} × D{24,44,92} × 프롬프트{2k, 8k} × 예산{8192, 65536}, 저부하 고정 rate
- 추가: HE0 핵심 셀에서 reactive bind+GATE 1개 + best decode-heavy static 1개(같은 boot 조건, drain 대기 측정용) · PHASE OFF 대칭 arm 1개
- 계측: 진단 §5 (i)–(v) 전부 + drain 이벤트에 "진행 중 prefill chunk 수"
- 판별량: 정지 반복에서 host 차단이 chunk GPU 시간·host 발행 시간·decode GPU 시간 중 무엇을 따르는가 · 동적 arm drain 누적 대기의 TTFT·ITL 몫
- 동등성 여유 사전 등록, "모델과 backend 미분리(게이트 #83)" 공시
- **Nano 변화 trace HE0 핵심 셀 재측정(n≥4, paired CI)은 정책 주장이므로 이 Job과 별도 PREREG**
