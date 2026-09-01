# 세션 핸드오프 — 2026-09-01

## 이번 세션 요약

사용자 요청 *"정책 비교군 정리 + chunked prefill을 비교 대상으로 추가하는 설계"*에서 출발해,
**규칙층 적대 감사 4회를 전부 `NO-GO`로 받고** 그 과정에서 캠페인이 정책 판정 트랙에서
**계측·축 검증 트랙으로 내려앉았다.** GPU **1.36 GPU-hr**(4 job)을 썼고 **새 성능 판정은 0건**이다 —
모든 지출이 *"계측이 되는가 / 축이 식별되는가"* 를 샀다. HE0·정책 순위·layer-type 全형태 死·
C2 인용정지 **전부 불변**. 로컬 커밋 4건(`fe8781b`·`5907a33`·`94fa01a`·`6a6b843`), push 없음.

★**원래 질문은 여전히 미측정**이다: chunked prefill vs pdmux 정책 비교는 한 번도 재지 않았다.

## 결정·측정

### 규칙층 감사 4회 (전부 `NO-GO`, GPU 0)

| 회차 | 대상 | 판정 | 판정서 |
|---|---|---|---|
| 1 | CP-1 규칙 rev1 | `NO-GO` 死因 9 | `results/cp_baseline/audit_cp_rules_2026-08-28/VERDICT.md` |
| 2 | CP-1 규칙 rev2 | `NO-GO` 死因 7 (**71%가 직전 수리의 그림자**) | `.../audit_cp_rules_2nd_2026-08-28/VERDICT.md` |
| 3 | CP-0 사전등록 rev1 | `NO-GO` 死因 6 | `.../audit_cp0_3rd_2026-08-28/VERDICT.md` |
| 4 | CP-0 rev3 | `NO-GO` 死因 8 | `.../audit_cp0_rev3_4th_2026-08-28/VERDICT.md` |

★4회차의 중심 판정: *"압도적으로 갱신 규율의 문제다"* — 24건 중 **11건(46%)이 순수 전파 실패**
(수리는 실행했고 그것을 인용하는 자리를 안 쓸었다). 그중 하나(§8의 "시험 3개")는 **유일한 중단
규칙을 무력화**했다 ⇒ 전파 실패가 런킬러가 될 수 있다.

### 경로 전환 (2회차 감사 후)
CP-1 정책 규칙을 rev3로 고치는 대신, 死因 K2·K3·L3·L8이 **"아직 안 잰 값" 없이는 못 닫힌다**는
판단으로 **CP-0(측정 단계)만** 사전등록했다. 4회차 판정: *"옳게 재배치했으나 재배치된 단계가
1회차의 병을 물려받았다 — 미룬 것도 푼 것도 아니다."*

### GPU 4 job

| job | 무엇 | 결과 | GPU-hr |
|---|---|---|---|
| **899768** | P1 수용 시험 1차 | **`P1_BLOCKED_INSTRUMENT`** (P1-a/c/f PASS · b/d/g `EXTRACTION_EMPTY` · e FAIL) | 0.248 |
| **900053** | P1 재실행 | **`P1_ACCEPTED`** 7/7 PASS | 0.246 |
| **900054** | G1 스모크(1 부팅) | 예산 재산정용 | 0.139 |
| **900067** | G1 프로브 본 실행 | 30/30 셀, 아래 §G1 | 0.729 |

원자료: `results/cp_baseline/{p1_accept_899768, p1_accept_900053, g1_probe_900054, g1_probe_900067}/`
요약 문서: `.../RESULT_P1_899768_2026-08-28.md` · `.../RESULT_G1_900067_2026-09-01.md`

### P1 (계측) — 닫힘, 단 한정 2건
1차 `EXTRACTION_EMPTY` 3건의 원인은 계측 로직이 아니라 **하네스×텔레메트리 상호작용**이었다.
진단이 한 단계 더 깊었다: scheduler는 **SIGTERM을 애초에 받지 않는다**(launcher가 자식을 SIGKILL로
회수, `launch_server.py:64` → `utils/common.py:1054`). 3겹 수리(telemetry 벽시계 flush · 프로브
shutdown handler 체이닝 · kill 시퀀스) 후 재실행에서 `chunk_mono_on.jsonl` **0 → 24,873 B**.

★**F3 기전이 관측으로 확인됐다**(그전까지 `server_args.py` 독해뿐): realized piecewise 캡처 목록이
**cps −1에서 len 0**(cps512 30 / cps4096 50). ⇒ `--chunked-prefill-size`가 prefill 예산과
**piecewise CUDA graph 캡처 범위를 동시에** 정한다. 스코프: Zamba2-2.7B · 이 트리 · n=1 부팅/arm.

⚠️**한정**: (a) P1-e의 phase B는 `exact_on`이 아니라 최소 문턱으로 채점돼 `n_chunk_events`
기대 3 vs 실측 2가 **verdict를 못 뒤집었다**(체커의 3번째 비대칭 — 앞의 둘은 수리됨).
(b) P1-f는 평균 +0.00113이 마진 안이나 **95% CI 양끝이 ±0.03 밖**(n=3) ⇒ *"중립 입증"*이 아니라
*"n=3에서 중립과 부합"*까지만 인용.

### G1 (용량 축 식별가능성) — ★4회차 감사 G1·G2의 예측이 측정으로 확인됨
ShareGPT, 2 arm × 5 rate × **seed 3**(rep 아님) × 1 job, 전 arm `--disable-piecewise-cuda-graph`,
프로브 OFF. 30/30 셀.

```
fused_default  r2 1.964  r3 2.855  r4 3.668  r6 4.918  r8 5.529   ← 6→8 여전히 +12.4%
cp512          r2 1.962  r3 2.852  r4 3.650  r6 4.820  r8 5.156   ← 6→8 여전히  +7.0%
attainment     0.98 → 0.95 → 0.92 → 0.82 → 0.69 / 0.64    SD(seed, n=3) 0.020–0.054
```

**기준-무관 발견**: 등록 술어(`ATTAINMENT_THRESHOLD=0.80`, lo=2·hi=8)를 그대로 넣으면 두 arm 모두
**`CAPACITY_BRACKETED`**인데 **달성 처리량은 최상단 rate에서도 단조 상승 중**이다 ⇒ 술어가
"포화"라 부르는 지점에서 서버는 아직 부하를 흡수한다. **CP-0 용량 축은 존재하지 않는 bracket을
보고했을 것이다.** 참조 스캔(다른 워크로드)에서는 임계 교차가 plateau와 겹쳤다 — 대비가 그림으로
확인된다(아티팩트, 아래 §미완).

⚠️`band_vs_sd` 축 값은 **채우지 않았다** — plateau 술어가 미등록이고 기준에 따라 arm마다 갈린다
(fused 증가 0.610 ≫ SD 0.19 / cp512 0.336 < SD 0.435). 결과를 본 뒤 기준을 고르면 사후 튜닝이다.

### GPU 0 증빙 (판정을 바꾼 것들)
- `served_population.json` **재측정** — 초안이 벤치가 서빙하지 않는 모집단을 썼다. 실제 서빙
  집합(seed 1, NP 200, ctx 4000)에서 cps별 분할 **51/10/4/0**(초안 27.9/7.6/2.1/0.48%). p99=2776이
  정본과 일치해 모집단 교차검증.
- 저장소 `g2ea` PHASE 0 capscan **재독** — *"capacity는 전 arm 미측정"* 주장이 **거짓**임을 확인
  (non-pdmux 런 109/108/108건). 그 데이터로 임계 0.80과 격자 {2,4,8}을 유도.
- job 899768 **재독**(수리된 분석기, 같은 바이트) — 등록 손계산과 정확히 일치(S 7 fwd/2625 tok,
  B 4 fwd/1600 tok). ⇒ P1-e FAIL은 계측 탓. **재독은 재측정이 아니다**(아티팩트가
  `statement_this_does_NOT_support` 필드로 자체 고지).

## 코드·문서 변경 (전부 커밋됨)

**`fe8781b`** 정본 인용·사실 정정 — CP의 P1 패치가 `scheduler.py`를 **13줄** 밀어 NSL **3개** 문서의
줄 인용 16건이 드리프트한 것을 전수 대조 후 정정 · A1 문서 회귀 수 **171→244** · ★`PROJECT_STATUS.md`
방법론 게이트 열거의 **#71–#80 10칸 공백 복원**(산문 원문 전재, 재판정 없음 — 이제 1–82 연속).

**`5907a33`** CP 규칙층 — CP-1 rev1/rev2(SUPERSEDED 배너) · CP-0 규칙 3종
(`cp0_arm_rule.py` RULE_REV=3 · `cp0_capacity_rule.py` RULE_REV=2 · `cp0_g1_rule.py` RULE_REV=1) ·
`cp0_predicates.py`(데이터→축 **전역 함수**) · `cp0_selftest.py` + **등록 오라클
`cp0_expected_labels.json`** · 사전등록 2건(CP-0 · G1) · `RETRACTION_reqinactive_2026-08-28.md` ·
도달가능성 spec/인증 · 신규 규율 게이트 **`check_version_sweep.py`** · 감사 판정서 4건.

**`94fa01a`** P1 계측 — `src/multiplex/chunk_probe.py`(신규, 기본 OFF 환경변수 뒤) ·
`src/multiplex/telemetry.py`(벽시계 flush) · `src/patches/chunk_probe_scheduler_hook.patch` ·
하네스·분석기·체커 · 회귀 테스트 **171 → 295**.

**`6a6b843`** P1 재실행 + G1 프로브 산출물 · `g1_probe.sbatch`/`g1_probe_analyze.py` ·
override 범위 확장 기록.

## 열린 항목 / 다음 세션 시작점

1. ★**CP-0 본체는 제출 불가.** 4회차 감사 死因 중 **G2**(격자의 절대단위 이송 — G1이 측정으로
   뒷받침) · **G3** · **G4** · **G7** 미해소. 닫힌 것은 G1(측정) · G5(오라클) · G6(접기 규칙).
2. ★**plateau 술어 등록이 선결** — G1의 `band_vs_sd`를 채우려면 필요하고 **결과를 보기 전에**
   코드로 고정해야 한다.
3. **체커의 3번째 비대칭** — P1-e phase B의 최소-문턱 채점. 이번 런이 만족하는 걸 본 뒤 조이면
   사후 튜닝이라 **보고만 하고 규칙은 안 고쳤다**.
4. **presubmit 전역 차단 2건** — M4R `SINGLE_LABEL_FORCED` · TC1 `RESTRICTIONS_INERT`.
   **다른 트랙 소관**, 손대지 않았다. 이번 두 잡은 사용자 명시 승인 override로 제출했다
   (`results/cp_baseline/OVERRIDE_P1_SUBMIT_2026-08-28.md`, 범위·금지사항 기록).
5. **long-context / Diff A·B 재측정**(세션 말미 사용자 질의) — `CONSENSUS.md` §5 열린 항목 #6,
   **사용자 발의 2026-07-17, 계획 단계 유지**. 정본이 이미 답을 적어 뒀다: TC1의 모델 전환은
   **이 항목의 재개가 아니고**, Diff A/B를 Nemotron-Nano에서 재측정하려면 **별도 신규 게이트**가
   필요하다. 두 겹의 블로커 — **백엔드 강제 교락**(게이트 #83: Nemotron-H+triton 부팅 거부 ·
   Zamba2+flashinfer 스케줄러 사망) + 옛 Diff B가 **2026-08-04 계측결함으로 강등**
   (L≤512 lever 1.42/1.22 → 정책 단위 1.032/1.031로 소멸, L=2000 1.35 → **1.0081**).
   ★게이트를 세운다면 담아야 할 4가지: 교락 선처리 · 정정된 계측(버킷 대칭·누산기 리셋) ·
   집계 단위(커널 vs 정책) 명시 · 교차점 L≈3k을 사이에 둔 격자(절대단위 이식 금지).

## 미완·주의

- ★**원래 질문 미측정**: chunked prefill vs pdmux 정책 비교 **0건**. 이번 세션 지출은 전부
  계측·축 검증이다.
- ★**날짜 불일치(doc-steward 이관)**: 오늘은 **2026-09-01**인데 이번 세션이 만든 문서 다수가
  파일명·내부 날짜를 **2026-08-28**로 달았다(세션 시작 시점 기존 파일 관례를 승계). 커밋 4건과
  `presubmit_registry.json`이 그 경로를 인용하므로 **개명은 위험**하다 — 정정 노트 방식 권고.
- **수리된 프로브의 관찰자 효과 재확인 필요**: P1-f는 n=3이고 CI가 마진 밖이다.
- **`band_vs_sd` 미확정**(위 2번). G1 라벨은 아직 없다.
- **claims-auditor에 안 건 것**: 이번 세션이 만든 `check_version_sweep.py`(S1–S8) 자체와
  G1 결과 해석은 적대 감사를 받지 않았다.
- **방치된 job 없음** — 4 job 전부 `COMPLETED 0:0`.
- **다른 세션이 동시 작업 중**(TC0/TC1/M4R/nemotron_zt/prefill_knee). 커밋 4건 모두 명시 경로만
  스테이징했고 그쪽 파일은 건드리지 않았다. 세션 중 그쪽 커밋이 들어와 내 인용 좌표가 밀린 적이
  있다(`1c6bb59`가 `CONSENSUS.md` +169줄) — 좌표는 **문구+sha와 함께** 인용할 것.
- **운영 함정**: 이 프로필에서 `bash`는 인자를 버리는 셸 함수다(`bash x.sh`는 아무것도 안 하고
  exit 0). `/bin/bash`로 부를 것. `cp`도 `-i` 별칭.
