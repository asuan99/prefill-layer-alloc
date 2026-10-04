# 판정서: N3 사전등록 rev2 델타 감사 (claims-auditor, 2026-10-05)

> 읽기 전용, GPU 0. 감사자 최종 보고를 메인 세션이 옮겼다. 성능 판정 0건, Claim 등급 변경 0건.
> 대조 sha(7/7 일치): PREREG 7b1ff758, n3_capacity.sh 4ffb6350, n3_plan.py dabcdd6e, n3_analyze.py 5d3d3283, n3_realized_mix.py fed78376, config 01675b2d(불변), test_n3_capacity.py ca9986f9.
> 감사자 재실행: `test_n3_capacity` 44/44(224 s) · plan·analyze selftest OK · mix `--selftest --require-archive` 7/7 IDENTICAL + R3 주입 3종 OK · λ0 VESSL 11셀 telemetry에 `arm_identity` 재계산 · `__pycache__` 0.

## 0. 판정: **`GO-with-caveats`**, 死因 0
- rev1의 R1·R2·R4·R5·R7·R8 반영 확인.
- **새 결함 2건(제출 전 수리, GPU 0)**: **D1** R3가 telemetry가 비어 있으면 그대로 통과(항등식) · **D2** R6 전례 문구 사실 오류 + 맥락 누락.
- §17 색인 밖 diff는 **검증 불가**(rev1 미커밋이라 바이트 diff 불가 — 앞으로 델타 감사는 rev1 커밋 후 diff로 받는다).

## 1. 항목별 판정

| # | 항목 | 판정 | 근거 |
|---|---|---|---|
| 1 | R1 round 순서 | 충족 | `units()`:251–254 i 짝수 내림차순. 테스트 2개(첫·마지막 grid unit = 최고 rate, 1·3 unit 절단 시 최저 rate 보존, rev1 대조). 비용 동일, rate별 평균 시점 대칭. 잔여 → N3C-15·16 |
| 2 | R3 arm 동일성 | **부분 충족 → D1** | 주입이 실경로를 탐(0.22 셀 1건 → `UNRESOLVED_N_BELOW_4_5`). 엔진에서 비표적 분할 경로는 threshold scan뿐(`:1170–1185`, R2 정책 미도달) → 오라클로서 항등식 아님. λ0 11셀 busy 312–2590건 중 비표적 0건. 단 decode_iteration 다리는 0건(PHASE_EVENTS OFF), A-agn 임계는 GPU 미측정. **D1**: `arm_identity([])`가 `ok: True`. 러너가 `: > tel`로 파일을 미리 만들어 "없음→unverifiable" 분기는 거의 미도달 — writer 조용한 실패 시 증거 0건으로 유효 판정(교훈 9) |
| 3 | R2 placeholder 거부 | 충족 | `tk_problems`(valid·dryrun 키·ttfts 16·median·T_k) 3곳(:285 :216 :278). 변이 3종 통과 |
| 4 | R4 `ITL_BOUND_k` | 충족 | 등록 라벨 vs ITL 65 사다리, 새 상수 없음. 합성 60 ms → 플래그, 20 ms → 없음 → N3C-17 |
| 5 | R7 | 충족 | 로그 문자열 실제 일치(`"PDMUX_PHASE_EVENTS disabled after failure: %s"`), 표식→무효화→관측자 짝 제외(n=3 → UNRESOLVED) |
| 6 | R8 | 충족 | P-1 리터럴(:443)·P-6 4개(:453). P-1 "arm 간" 공백은 N3C-11이 막음 |
| 7 | 하네스 wiring | 항등식 근접 해소, 잔여 caveat | `fail`/`dry_or_abort` 첫 인자만 인정, 분기별 echo 중화 변이 11/11 검출(교훈 53). 실행형은 preflight 4분기만; boot별 5분기(STICKY·GREEN·P3·floor boot·CG2)는 텍스트 검사 → N3C-18(하네스층 몫) |
| 8 | R6 전례 문구 | **오류 → D2** | "3 토큰 적게"는 A14 원문과 일치. 그러나 (a) **"같은 config·arm인 E2"는 거짓** — E2는 homog5 5그룹·sticky ON/OFF arm, N3는 A-agn 6그룹(§2-2가 스스로 강조) (b) A14:79–81 맥락 누락 — VESSL 26 boot(+KISTI 11) 중 이탈 1건, **유일한 비정상 종료(포트 충돌) 바로 다음 boot**, O1–O4가 겨냥하는 경로. 이 맥락 없이 "노출이 E2보다 많다"만 남기면 위험 진술이 한쪽으로 기움 |
| 9 | 범위 밖 diff | 검증 불가(공시) | rev1 사본 없음. "R" 태그·§17 대조로는 못 찾았으나 증거 아님 |

## 2. 반전 시험(rev2 신규 자유 표면)

| 자유 표면 | 민 범위 | 판정 변화 | 근거 |
|---|---|---|---|
| R1 rate–시점 결합(§12-21) | rev1 ↔ rev2 순서 | 반전 못 만듦 | 두 순서 모두 rate별 round 내 위치 {p, n−1−p}×2로 선형 drift 균형. 첫 grid 셀(최고 rate)은 실패 쪽(λ0 A 3.0 att ≤ 0.585). 비선형 초반 효과는 r_fail = 최고 rate일 때만 라벨에 닿음 |
| R1 절단 위치 | 마지막 1–3 unit | 측정 부족(반전 아님) | A 상위 3 rate(2.4·2.7·3.0)가 r_fail 후보대와 겹침(A 최악 여유 1,539 s). B 최악 여유 8 s 그대로 |
| R3 임계 | 0건 고정 | 자유도 없음 | — |
| R4 사다리 | ITL 65 고정 | 자유도 없음 | — |

죽은 것 없음.

## 3. 제출 전 필수 수리(rev3, GPU 0)
- **D1.** `arm_identity`가 근거 부재를 `unverifiable`로: `busy_snapshots_seen == 0`이면 근거 없음; PHASE_EVENTS ON이면서 R7 표식 없는 셀은 `decode_iterations_seen == 0`도 근거 없음; 분석기는 무효 처리. 문턱은 0뿐(새 상수 없음). 변이: 빈 telemetry 파일 / 스냅숏만 있고 decode_iteration 없는 ON 셀 → 둘 다 무효. **규칙 변경 → 델타 감사.**
- **D2.** §2-3 R6 문구: "같은 config·arm" → "다른 config(homog5 5그룹)·다른 arm(sticky ON/OFF)의 E2 Job"; A14:79–81 정황 추가(이탈 1/26 VESSL, KISTI 11 boot 전부 2,722,025, 비정상 종료 바로 다음, H1/H2 판별 불가, O3·O4가 H1 겨냥). **사용자 R6 결정 전에** 반영.

## 4. 신규 caveat(결과 문서 승계)
- **N3C-15.** "round 순서는 예산 절단이 최고 rate부터 빠지게 설계됐다. shape A에서는 상위 rate가 r_fail 후보대와 겹쳐, 절단이 그 k의 라벨을 UNRESOLVED로 만들 수 있다. 이는 측정 부족이지 발견이 아니다."
- **N3C-16.** "각 Job의 첫 grid 셀은 최고 rate다. Job 초반 노드 상태(클럭·power cap)의 비선형 효과는 r_fail이 최고 rate일 때만 라벨에 닿고, 크기는 측정하지 않았다."
- **N3C-17.** "`ITL_BOUND_k` = false는 ITL을 50→65 ms로 완화해도 라벨이 바뀌지 않는다는 뜻일 뿐이다. ITL이 무관하다는 뜻이 아니다."
- **N3C-18.** "boot마다 도는 abort 분기(STICKY·GREEN·P3·floor boot·CG2)의 배선은 텍스트 검사(분기별 중화 변이 11/11)로만 검증됐다. 실행형 검증은 preflight 4분기뿐이다."
- **N3C-19.** "R3 arm 동일성의 사전 근거(λ0 VESSL 11셀 비표적 0건)는 homog5·스냅숏 다리에서만 나왔다. A-agn 임계에서 decode_iteration 다리는 GPU에서 측정된 적이 없다."
rev1 N3C-1…14, DS·LV·LVR 계열 승계.

## 5. 반증 실패
R1 절단 보존 · R2 거부 3곳 · R4 정의 · R7 경로 · 쌍 키 7/7. 실행·결과 감사 전 `CONFIRMED` 아님.

## 6. 남은 제출 선행조건
1. D1·D2 반영(rev3) → **델타 감사**(D1).
2. **R6 사용자 결정**(엄격 / 허용오차(ii)). (ii)면 E2 A14 (ii) 감사 선행 → rev3 기입 + 델타 감사.
3. 하네스층 감사 + 봉인 sha(§14 제출 게이트 2).
4. 커밋 → `REGISTRATION_SHA256.txt` 대조, CPU 회귀, citation 0위반.
5. (선택) Workspace 6그룹 boot 프로브(≈0.1 GPU-h) — B 최악 여유 8 s의 원인인 per-boot 오버헤드 실측.
6. `launch.sh` dry-run 사용자 승인(shape A 별도). 제출은 E2 정리 뒤 권고.

## 7. 실현가능성 자기검사(게이트 #113)
D1은 분석기·mix 몇 줄 + 테스트 2개, 새 상수 없음, 오발화는 telemetry 실제 부재 시만(λ0 11셀 312–2590건이라 정상 통과). D2는 텍스트. R6 선택지 미처방. rev1 지적을 상수로 승격하지 않고 코드 실행·아카이브로 독립 재검증.

## 관련 파일
`n3_capacity/PREREG_N3_CAPACITY_2026-10-04.md`(§2-3 :98–101, §3-1 :128–130, §11 :443·:453, §12-21 :507, §17 :565) · `n3_realized_mix.py`(`arm_identity` :374–386) · `n3_analyze.py`(`cell_validity` :63–96, R4 :329–338) · `n3_plan.py`(`units` :233–256, `tk_problems` :554–574) · `n3_capacity.sh`(:216, :285, :457–462) · `tests/test_n3_capacity.py` · `e2_sticky_vessl/PREREG_E2_VESSL_A14_DRAFT_2026-10-04.md`(:20, :65–82).
