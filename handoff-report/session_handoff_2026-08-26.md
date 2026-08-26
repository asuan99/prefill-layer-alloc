# 세션 핸드오프 — 2026-08-26 (2026-08-25 야간 연속)

## 이번 세션 요약

**A1(kernel_mech Stage 0‴ 엔진 기판)을 sticky 기반으로 재설계**해 규칙층 감사를 **3회** 받았고
(전부 `NO-GO`), 그 지적을 대부분 이행했다. **NSL 트랙은 ③·②·E-B 묶음을 만들어 그 트랙의
첫 규칙층 감사**를 받았다(`NO-GO`, ★死因 4건). **GPU는 한 번 썼다** — A1 스모크 job 893663,
**≈0.014 GPU-hr, 세 항목 전부 PASS**. 커밋 **20건**, 워킹트리 clean, 실행 중 job 0건.
**새 성능 판정 0건 · HE0·정책 순위 불변.**

★이 세션의 성격: **감사 4회가 매번 실재하는 결함을 찾았고, 그중 다수가 내 직전 수리의
그림자였다.** 3회차 감사가 그 형태에 이름을 붙였다 — ★**"수리는 국소, 주장은 전역"**.

---

## 결정·측정

### 1. A1 재설계 — sticky 부팅 분리 (절차 1–6)

`results/kernel_mech/DESIGN_A1_REV2_STICKY_2026-08-25.md` 신설(rev1에 SUPERSEDED 배너).

- **§2 다리 = 부팅 분리**(시간창 조인 폐기). 부팅 **5개**: B-U / B-S16 / B-G / B-S16′(nsys-OFF)
  / B-D92(비프로파일). B-U↔B-S16은 **환경변수 하나만** 다르다.
- **§5 중단 규칙**을 부팅 단위로 재작성. `N_min`을 **decode step 단위**로 재등록.
- **절차 2**: `src/multiplex/green_readout.py` 신설 + `_maybe_emit_green_readout()` 배선
  (기본 OFF, `PDMUX_GREEN_READOUT=1`). `sync_engine_tree.sh` 복사 목록·manifest 등재.
- **절차 3**: Q3 채널 = **`decode_iterations`** 확정. 근거 = `decode_step_count`가 저장소
  **0 / 1,958,528 스냅샷**(전수 `architecture=legacy`), `decode_iterations`는 975/980 파일 비영·단조.
  → `a1/DECISION_A1_Q3_CHANNEL_2026-08-25.md`
- **절차 4·5·6**: `a1/a1_q3k1_rule.py` `RULE_REV=3`(Q3 **414,720 세계**) + D1·D3 정정.
- ★**1차 추정량 규칙을 새 파일로 신설**: `a1/a1_primary_rule.py`
  (**3,110,400 세계 / 정합 166,194**). 제자리 수정 불가 — `a0_verdict_892556.json`이
  `rule_sha256=583c4ab0…`를 고정하므로 완주 실험의 채점 해시가 깨진다.

### 2. A1 규칙층 감사 3회 — 전부 `NO-GO`

| 회차 | 판정서 | 핵심 |
|---|---|---|
| 1 | `audit_a1_rev2_rules_2026-08-25/` | B1′ A0 규칙 미개정 · B2 게이트 #21(Q3 `launches`) · B3 §5.1a · B4 자기모순 · **B5 Ha8 선례 존재** |
| 2 | (파일 미저장, 요지는 §6.8·§7·커밋에 반영) | P1 B10 미적용(primary) · P2 게이트 #21(primary) · P5 **도구가 거짓 인용 인증** · P7–P12 |
| 3 | `audit_a1_rev2_rules_3rd_2026-08-25/` | R1 **부팅 5 / 세계모형 3** · R2·R3 상태·금지문 3회차 재발 · R4 출처 허위 3회차 · ★**형태 명명** |

### 3. ★ GPU 실측 — A1 스모크 **job 893663** (≈0.014 GPU-hr, 세 항목 PASS)

`results/kernel_mech/a1_smoke/RESULT_A1_SMOKE_893663_2026-08-26.md`

| | 결과 |
|---|---|
| (a) `E1_DECODE_REALIZED(16)` in **batch-synchronous** | ★**1.0000**(`t_decode_active=35.5s`, hist `D16`만) |
| (b) `decode_iterations` (A1 자기 셀) | **0 → 415** |
| (c) `PDMUX_GREEN_READOUT` **첫 부팅 실행** | ★드라이버가 sticky division decode 스트림에서 **`smCount=16`** |

- ★**(a)가 §5.1a의 모집단 간극을 닫는다** — 0.99 뒤의 21 부팅은 전부 ShareGPT open-loop(라운드
  경계 없음)였다. **기전이 존재하는 모집단에서 처음 시험됐고 통과했다.**
- ★**(c)는 `GREEN_TARGET_CONFIRMED`의 첫 관측**(3회차 감사 R1이 신설시킨 라벨). **`D=16` 한 점**,
  `D=92`는 미관측.
- ★**한정**: n=1, 35.5초. *"batch-sync에서 realized는 1.0이다"* 로 읽지 말 것.
- 배너로 `pp_size=1 ∧ pp_max_micro_batch_size=None` **실측 확인**(NSL ②§1.2 전제).

### 4. NSL — ③·②·E-B 묶음 + **첫 규칙층 감사**(`NO-GO`, 死因 4건)

- ③ `PREREG_NSL_STEP3_KNOB_PURITY_2026-08-25.md` — R1–R6. ★코드 확인: cap 술어의 상수는
  `pp_max_micro_batch_size`이고 `ratio`는 `--disable-radix-cache`에서만 1이다.
- ② `DESIGN_NSL_STEP2_ADMISSION_SITES_2026-08-25.md` — `batch_is_full` 사이트는 **5개**,
  pdmux arm에서 도달 가능한 것은 **2개**. ★초판이 `:2369`를 "도달"로 적었다가 **자기 정정**.
- E-B `nsl_eb/PREREG_NSL_EB_ATTRIBUTION_2026-08-25.md` + `nsl_eb/nsl_eb_rule.py`
  (13,440 세계 + 캠페인 집계 함수). **cap 스윕을 빼고** 한 셀 귀속으로 축소(24→8 부팅).
- 판정서: `results/nsl_lever/audit_nsl_bundle_2026-08-26/VERDICT.md`

### 5. ★ 규율 도구 2종 신설 (산문 규율을 기계로)

- `scripts/discipline/check_line_citations.py` + `line_citations.json`(50건) + 단위 **16건** —
  지문 기반 드리프트 검출 + 정정 인용 제시 · 재베이스 거부 · bare 인용 · `[HIST]` · **고아 키**.
- `scripts/discipline/check_doc_facts.py` — 문서가 자기 아티팩트에 대해 주장하는 수치를
  진리원과 대조. **11 facts / 14 occurrences**. 첫 실행에 **7건** 적발.

---

## ★ 내가 저지르고 정정한 것 (다음 세션이 승계할 것)

1. ★★★**"Ha8 + sticky 부팅은 선례가 없다" — 거짓이었다.** `stksmoke_Ha8_d16_872800_result.txt`가
   `ARM=Ha8`·`sticky=1`·correctness PASS·realized 1.0000을 보인다. ★**그리고 내가 §1에서 인용한
   `0.0839 → 1.0000`이 바로 그 파일의 두 줄이다** — 파일의 수치를 인용하면서 그 파일의 주어가
   선례 없다고 적었다(교훈 #79의 가장 나쁜 형태).
2. ★★★**§5.1a에서 게이트를 5% 느슨하게 열었다** — B6 산문을 읽고 **크기를 재지 않은 채**.
   실측 21/21 ≥0.9956, 잔차 0.1–0.4%(완화폭의 1/10~1/50). **교훈 #21을 인용하면서 그 거울상**.
   철회했고, 스모크가 그 문턱을 실제로 통과시켰다.
3. ★★**금지문 12건이 거짓이 됐다** — 수리가 진행되며 참이 되어 **문서가 자기 코드에 대한 참인
   문장을 금지**하고 있었다. §7이 *"금지문도 판본이 바뀌면 재검증 대상"* 을 교훈으로 적어 놓고
   **바로 그것을 안 했다**.
4. ★★**인용 도구가 거짓 인용을 인증했다** — `--snapshot`이 편집된 파일의 기준선을 조용히
   덮어써 `a1_q3k1_rule.py:83`이 `N_MIN_DECODE_STEPS`를 가리킨다고 인증했다(실제 `:105`).
   **도구가 선언한 한계가 한 커밋 만에 발화**했다. 재베이스 거부 + 고아 탐지로 수리.
5. ★★★**NSL §4.1: "A1 결함 가족 선제 적용" 주장이 거짓.** A1 3회차 표는 **6 가족**인데 **4개**만
   열거했고, **뺀 셋이 그 회차 런킬러**(상태 오보·거짓 금지문·출처 허위)이며 **셋 다 E-B에 있다**.
   **교훈을 옮긴다고 선언하면서 교훈 자체를 잘못 베꼈다.**
6. ★★**게이트 #21 수리가 반대 방향으로 열렸다**(NSL D1) — `n_firings=4000`이어도
   `saturated="no"`면 버린다. 저장소 자신의 데이터가 **등록된 rate-3 부팅 4개가 정확히 그
   세계**라고 말한다.
7. ★★**거울 대칭 검사를 삭제한 것이 버그를 숨겼다**(NSL D3c) — `1.0-0.80=0.19999999999999996`이라
   `cap_share=0.2`와 `0.8`이 다르게 채점되는데, 그걸 잡을 유일한 검사를 *"단독 구속 0"* 을 근거로
   지웠고 **그 0은 격자에 0.20·0.80이 없어서 생긴 인공물**이었다(교훈 #46+#78).
8. **파일 하나를 스크립트 실수로 날렸다**(`re.sub(..., "", "")`) — git에서 즉시 복원. 이후
   문자열 치환만 사용.

---

## 코드·문서 변경 (전부 커밋됨, 워킹트리 clean)

- **엔진**: `src/multiplex/green_readout.py`(신규) · `multiplexing_mixin.py`
  (`_maybe_emit_green_readout`) · `scripts/bootstrap/sync_engine_tree.sh`(복사·manifest)
- **규칙**: `results/kernel_mech/a1/a1_primary_rule.py`(신규) · `a1/a1_q3k1_rule.py`(rev3) ·
  `results/nsl_lever/nsl_eb/nsl_eb_rule.py`(신규)
- **설계·사전등록**: `DESIGN_A1_REV2_STICKY_2026-08-25.md` · `a1/DECISION_A1_Q3_CHANNEL_…` ·
  `PREREG_NSL_STEP3_KNOB_PURITY_…` · `DESIGN_NSL_STEP2_ADMISSION_SITES_…` ·
  `nsl_eb/PREREG_NSL_EB_ATTRIBUTION_…`
- **판정서 3건**: `audit_a1_rev2_rules_2026-08-25/` · `audit_a1_rev2_rules_3rd_2026-08-25/` ·
  `audit_nsl_bundle_2026-08-26/`
- **하네스·결과**: `a1_smoke/a1_smoke.sbatch` + `RESULT_A1_SMOKE_893663_2026-08-26.md` +
  telemetry/green readout 아티팩트 · `a1/a1_q3_channel_probe.{py,json}`
- **도구·테스트**: `scripts/discipline/check_line_citations.py` · `check_doc_facts.py` ·
  `line_citations.json` · `tests/test_green_readout.py`(15) · `tests/test_line_citations.py`(16)
- CPU 회귀 **171 PASS**.

---

## 열린 항목 / 다음 세션 시작점

★**시작점 한 줄**: **NSL 감사 권고 1–6(GPU 0)으로 E-B의 추정량 식별성을 고친다** —
특히 D3(순서통계량)·D1(`NOSAT` 과잉)·D2(관측 채널 없는 계층).

1. **NSL D1–D4 수리**(전부 GPU 0):
   - D3 ★추정량을 `cap_first_share` 류로 **개명**하고 루프 우선순위를 **추정량의 성질로 등록** ·
     `schedule_policy.py:768/:791`의 `OTHER` 출구를 제3 계수류로 계측하거나 §0의 분모를 좁힌다 ·
     문턱 대칭 수리 + `AXES`에 **0.20·0.80 추가** + **거울 검사 복원**.
   - D1 `NOSAT`을 `n_firings == 0` 조건부로 내리거나 `saturated`를 **보고 공변량**으로 강등.
   - D2 `unattributed`를 **전이**로 정의 + `:2353` 래치 프로브, 못 하면 **TOOLLIMIT 계층 삭제**.
   - D4 §2를 *"예산 근거로 Phase 2를 안 산다"* 로만 쓴다.
   - B6 정상 rate 2셀 → ★**①이 실제로 잰 변화 trace**로 교체(부팅 절반, D1 물림 소멸).
   - B2/B7 `M=48` 재검토(①의 KV 예산과 일치) · `N_MIN_FIRINGS` 스모크로 교정 · 자유 모수 재열거.
2. **A1 4회차 규칙층 감사** — 3회차 잔여(P13·P15–P20 일부)와 이번 수리(R1·R4·B5–B8) 검증.
   ★3회차 감사가 *"4회차가 다시 `NO-GO`라면 그건 규칙이 아니라 **문서 갱신 규율**의 실패"* 라고
   적었고, 그 규율은 이제 **도구 2종**으로 서 있다.
3. **도구 확장**(감사 B4/B6 지적): `check_doc_facts.py`가 **표를 `LABELS`와 대조**하도록 ·
   `check_line_citations.py`의 `CITE`가 **bare `:NNN`·쉼표 목록**(`g2_run.sbatch:104,115`)을 잡도록 ·
   ★**두 도구를 사전등록 "제출 전 체크리스트"에 등재**(현재 등록된 실행 지점 없음, 감사 B11).
4. **A1 프로파일 부팅 3개**(B-U·B-S16·B-G)는 규칙층 통과 전 **제출 금지**.

---

## 미완·주의

- ★**A1은 규칙층 감사를 통과한 적이 없다**(3회 전부 `NO-GO`). *"rev2가 규칙층을 통과했다"* ·
  *"A1의 死因이 닫혔다"* 금지.
- ★**NSL 묶음도 `NO-GO`**(死因 4건). ②의 **엔진 패치는 0줄**이고 규칙층 통과 전 쓰지 않는다.
- ★**2회차 A1 판정서가 파일로 저장되지 않았다** — 요지는 `DESIGN…§6.8`·§7과 커밋
  `3359fa3`/`7b7b007`/`189acfc` 메시지에 있다. 3회차 판정서는 그 사실을 감사 대상으로 삼았다.
- ★**스모크는 n=1**이다. (a)(c)를 일반화하지 말 것. `D=92`의 `green_sm` 확인은 **미관측**.
- ★신규 방법론 교훈 후보 **3건**(NSL 감사 §⑥ (A)(B)(C))이 아직 `CONSENSUS.md` §3에 등재되지
  않았다 — doc-steward 판단 대상.
