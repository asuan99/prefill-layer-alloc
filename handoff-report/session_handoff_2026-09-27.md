# 세션 핸드오프 — 2026-09-27 체크포인트

> **대상 세션**: 2026-09-22 "PD-mux 좌표계 v2 코드 근거 감사 + 종합 보고" 세션.
> 체크포인트를 **2026-09-27에 뒤늦게** 쓰는 것이므로, 이 문서는 그 세션의 델타 + **그 뒤
> 2026-09-24~27 세션들이 이미 흡수한 것의 대조**를 함께 적는다.
> **GPU 지출 0 · 새 성능 판정 0건 · Claim 등급 변경 0건.**

---

## 1. 이번 세션 요약

사용자 지시로 "PD-mux 연구 좌표계 v2"의 코드 근거를 감사했다(sticky 캠페인 S1–S5 · 계보 M1 ·
선행 대조 M2–M6 · v2 인용 문장 D1–D5). 전부 **읽기 전용 + GPU 0**. 이어서 같은 세션에서
완료 실험 종합 보고와 향후 실험 설계 보고를 작성하고, 읽기용 artifact로도 출판했다.
감사 결과 **정본 1건이 거짓임을 확인**(`*pdmux*.yml` `manual_divisions` 59/59 → **55/59**),
**v2 인용 문장 1건 반박**(dual-worker admission "개입" → 계측), **1건 중대 단서 누락**
(R2 admission hold는 `FixedPolicy`에서 미발화), **선행 대조 판정 2건을 preprint→camera-ready로
정정**했다. 제안 diff는 **다음 세션(2026-09-24)이 그대로 적용**해 `CONSENSUS.md` rev81이 됐다.

---

## 2. 결정·측정 (전부 GPU 0)

### 2.1 엔진 버전 특정 — 멈춤 조건 미발동
sticky 3개 job(872800 / 873015 / 873921)의 `runtime_source_manifest*.sha256`이 기록한
`multiplexing_mixin.py` = `59eaafb4…` 는 **현재 트리(`e2a97b42…`)와 다르다**. git 전수 대조로
그것이 커밋 `f60a128`(2026-08-03, sticky 도입) 판임을 **byte-exact 확인**. manifest는
`sync_engine_tree.sh`의 **출력**(입력 검증 아님)이므로 그 sha가 곧 실행 시점 설치본이다.
sticky 관련 블록은 `f60a128` ↔ 현재 사이 **줄 번호만 이동, 내용 동일**.

### 2.2 S3 재집계 — sticky PART의 정체 [DERIVED · R-time-all]
`reports/synthesis_2026-09-22/`·`reports/audit/2026-09-22_scope_lineage/scratch/part_cohab_split.py`

| 대상 | PART% | ∧prefill-busy | ∧prefill-idle |
|---|---|---|---|
| `s2_sticky` (20 boot, 전부 ON) | 78.53 | 7.57 | 70.96 (PART의 90.4%) |
| `sticky_smoke` ON arm | 86.22 | 3.19 | 83.03 |
| `sticky_smoke` OFF arm (같은 job) | 5.57 | 3.07 | 2.50 |

⇒ **sticky가 더한 것은 거의 전부 prefill-idle**이고 동거는 `3.07 → 3.19 %`로 불변.
자기검사: 가법성 5규약 전부 · 음성 대조군(no-split 4파일) 세 값 0 · 상류 census Tier-1 재호출.
셀별 경향: d16(prefill 92 SM) 0.91–2.85 % · d54 4.07–6.79 % · d92(prefill 16 SM) 21.13–27.23 %
(⚠ d92는 다른 job 873921 — cell ≡ job 앨리어스).

### 2.3 S4 — 03절 비용 수치는 오염되지 않았다
PART 체류 decode forward(granite 1.53–1.66× · zamba2 1.73–1.94×)의 출처는
`p1_gates/gate2` HOLB(jobs 874602/874633/874635)이고 그 캠페인은 **sticky OFF**
(플래그가 그 job보다 **뒤**에 도입) ⇒ PART ⟺ prefill in-flight, 동거가 구조적.

### 2.4 S5 — 분리 재료가 아카이브에 없다
sticky 아카이브 decode-busy 스냅샷 8,541건에서 `decode_last_tpot_ms`·`decode_step_count`·
`worker_overlap_ratio` **전부 0.000 %**(기록자가 `dual_worker_enabled` 게이트 뒤에 있고
세 job은 미설정). 유일 후보 `measured_itl_ewma_ms`는 α=0.85 EWMA + 전환 경계 표본 배제
필터라 상태 귀속 불가 ⇒ **계측을 켜고 다시 재야 한다**(카드 D-5의 근거).

### 2.5 M1 — upstream pdmux = MuxWise의 engine 층 (결론 (a))
PR #11592/#12275(`ykcombat`)가 `multiplexing_mixin.py` 209줄 + `pdmux_context.py` 163줄 신규.
layer-wise prefill은 PR #7634 — **upstream 커밋 author 필드가 `Xiaoze Fan`**(MuxWise 5저자),
추적 이슈 #10813(`Raphael-Hao` = **Weihao Cui**, MuxWise 2저자)이 **MuxWise arXiv(2504.14489)를
"Related resources"로 명시**. MuxWise 3구성요소 중 upstream에 있는 것은 **engine뿐**
(estimator·dispatcher 심볼 0건, 로컬 `muxwise.zip` 아카이브에도 0건 — 출처 미검증).
★**독립 교차검증**: GitHub API 경로와 `git log -S` 전수 추적 경로가 같은 표를 냈다.
부수: `create_greenctx_stream_by_value` 후속 PR **#8701**(`cuda < 12.4` 비호환 수정)·
**#9021**(드라이버 버전 런타임 검사) ⇒ **CUDA ≥ 12.4** 조건의 upstream 출처 특정.

### 2.6 M2–M6 — camera-ready로 재확인, 1차 판정 2건 정정
1차 통과에서 지정 PDF를 찾지 못해 **멈춤 조건이 발동**했고 arXiv preprint로 잠정 판정했다.
사용자가 `Papers/{Muxwise,Bullet}.pdf`를 제공해 **2차 통과에서 전수 재확인**:
- MuxWise camera-ready ≡ arXiv v3 (인용 8문장 전부 동일) ⇒ 판정 변경 없음.
- **Bullet camera-ready ≠ arXiv v4** ⇒ **M5 정정**: Table 5에 `Resource Re-config` **Mean 4.1 μs /
  P99 5.9 μs**가 **있다**("측정값 없음"은 preprint 기준 오류). **M6 정정**: Bullet §4.2.1이
  **A100 1장 평가**를 갖고 있다("우리만 단일 GPU"는 거짓).
- **M4 한 단계 더 좁힘**: 동거 경합을 **선행 2편 모두** dense에서 이미 보고
  (MuxWise §3.3.1 ≈0–30 % · Bullet §3.2.3) ⇒ 우리 몫은 **hybrid 재확인 + 크기 차이**.
- **M2는 부분**: 정의된 집계 추정량으로서의 residency는 공백이나 **Bullet Fig. 20a가 SM
  구성별 지속시간 막대 타임라인**이라 "아무도 보여주지 않았다"로 쓰면 반례.
- ★범위 밖 발견: **Bullet §4.4**가 *"there is no optimal fixed SM allocation"* — HE0와 문면상
  정면 반대. 5개 축(static arm 정의·best-static 탐색 절차·술어·SLO 엄격도·동적 제어의 내용)이
  달라 **판정하지 않고 회부**.

### 2.7 D1–D5 — v2 인용 문장 재확인
- **D4 반박**: `begin_admission`/`finish_admission`은 `admission_latency_ms`를 적는 **스톱워치**다.
  개입하지 않고, `dual_worker_enabled`가 아니면 호출조차 안 된다.
- **D3 중대 단서 누락**: `r2_admission_limited`는 **`CoarseGrainedController`만** 세운다
  (`overload_streak >= 2`). `FixedPolicy.decide`는 넘기지 않으므로
  **`R2_POLICY=fixed` 캠페인(E1·S2·sticky·λ0·E2) 전부에서 한 번도 발화하지 않았다**.
- **D1 부분**: decode_bs drift는 **sticky OFF일 때만**(ON+fixed면 인덱스 상수).
- **D5**: 기전 **가설**이다. 같은 방향 관측이 `s8p_prefill`에 있으나(4 arm 중 3개 단조)
  그 캠페인은 **claims-auditor 미통과 = 정본 인용 금지**, 셀당 n=1.
- **정정**: "hybrid에서 layer 단위 prefill 발사 성립 여부"는 **열린 질문이 아니다**
  (4모델 `forward_split_prefill` 패치 위치 인용). 남는 질문은 **span 입도**와
  **span 경계 Mamba state 운반 비용**(둘 다 미측정).

### 2.8 부수 발견 — 정본 1건 거짓 (I-1)
`*pdmux*.yml`의 `manual_divisions` 전수는 **59/59가 아니라 55/59**다. 빠진 4개는 전부
byte-identical한 `pdmux_a100_smoke.yml`(sha256 `8e991318…`)로 키가 **없고**, 주석에
*"No manual_divisions -> engine auto-computes partitions via `divide_sm()`"* 라 적혀 있어
**`grep -q`에 걸린다**. 이 config를 쓰는 job script **27개**에 **P1.7 4모델 · P1-opint 운영점 ·
`p1_gates/gate2`(전환 비용 원 캠페인)**가 포함 ⇒ `get_arch_constraints`는 **우리 실행 경로에서
발화한 적이 있다**(A100 cc8이라 정상 반환, **결과 불변**). 교훈: **존재 검사는 파서로**.
부수 2건: **I-2** census README/`.py`의 `multiplexing_mixin.py:880-893` 줄 인용 표류(현재 트리에선
dual-worker 코드) · **I-3** sticky `ENABLED` 로그가 decode-empty 시 그룹을 `(0,total_sm)`이라
적지만 코드는 `(total_sm,0)`(동작은 의도대로, 문구만 뒤바뀜).

### 2.9 종합 보고에서 새로 드러난 구조적 사실 (GPU 0)
- **캠페인 지형**: 14개 캠페인 중 **sticky를 쓴 둘만 `cohab P-act`가 5.44 % / 11.80 %**이고
  나머지는 전부 90 %대 ⇒ sticky 캠페인을 빼면 **PART ≈ 동거**.
- ★**층 단위 prefill 발사가 짧은 컨텍스트에서 발화하지 않는다.**
  `forward_count = max(1, split_forward_token_budget // extend_num_tokens)`가 한 호출이 도는
  층 수이므로 `ext ≤ 65536/L`인 요청은 전 층을 한 번에 돈다. config는 **59/59 전부 65536**이고
  **스윕된 적 없다**. 임계 = 65536/L(Qwen2.5-7B L=28 → 2,340 · L=56 → 1,170 · Zamba2-7B L=81 → 809).
  아카이브 실측: `s2_sticky` terminal 100 % · **`e1_traceforce`(force-trace 15,453건) 99.84 %** ·
  `longctx_conflict`(ctx 16,384) 94.06 %이면서 **4층 간격 사다리**(= 양성 대조, 교훈 232).
  입력 길이 분포(median 135)로 예측한 **99.2 %** monolithic ↔ 관측 **99.84 %** 일치
  ⇒ 두 독립 경로가 같은 결론. 정본 **§1-24**에 코드·아카이브 양쪽 기전을 준다.
  스크립트 `reports/synthesis_2026-09-22/scratch/split_index_census.py`.

### 2.10 CPU 회귀 (참고 — 코드 미변경)
`Ran 598 tests in 288.192s — FAILED (failures=27, errors=13, skipped=2)`.
정본 기준선(`env/cpu_regression_baseline_2026-09-18.md:55,63,150`)이 기록한 세 런은
`26/13/3` · `27/13/2` · `21/13/2`로, 이번 값은 그중 두 번째와 **정확히 일치**하고 기록된
변동폭 안이다 ⇒ **회귀 없음**(테스트 대상 코드는 건드리지 않았다).

---

## 3. 코드·문서 변경 — **전부 이미 커밋됨**

| 경로 | 내용 | 커밋 |
|---|---|---|
| `reports/audit/2026-09-22_scope_lineage/{REPORT.md,citations.json,consensus_patch_proposal.diff}` | 감사 본문(S1–S5·M1–M6·D1–D5·I-1~3) · 인용 fingerprint 21건 · 제안 diff | `3aa794f` + `029a771` |
| `reports/audit/2026-09-22_scope_lineage/scratch/{part_cohab_split,make_citations}.py` | S3 분해 재집계기(기존 술어 import) · citations 생성기 | `3aa794f` / `029a771` |
| `reports/synthesis_2026-09-22/{README,RESULTS_REPORT,EXPERIMENT_DESIGN}_2026-09-22.md` | 완료 실험 종합 · 향후 설계(카드 D-0…D-10) | `3aa794f` |
| `reports/synthesis_2026-09-22/scratch/split_index_census.py` | 층 단위 발사 집계기(양성 대조 내장) | `3aa794f` |
| `reports/CONSENSUS.md` rev81 | 배너 스코프 정정 + §3 항목 **280–283** + §5 열린 항목 **11** | `b330ae8` |

**정본 편집은 이 세션이 하지 않았다** — 제안 diff만 냈고, `b330ae8`(2026-09-24)이 적용했다.

**읽기용 artifact**: <https://claude.ai/artifact/Ppm4SYMDrXtMuLo9JyA4cu> (「PD-mux 실험 원장」).
두 종합 보고를 한 페이지로 합친 **읽기 표면**이며 저장소 밖이다 —
**markdown 원본이 우선, 원본과 정본이 다르면 정본이 우선**(페이지 푸터에 명시).

---

## 4. 이 체크포인트에서 한 일 (2026-09-27)

`reports/synthesis_2026-09-22/EXPERIMENT_DESIGN_2026-09-22.md`에 **stale 배너 + 정정 2건**을 넣었다.
이 문서가 `roadmap_review_2026-09-27.md`의 인용 출처라 방치하면 낡은 행이 재인용된다.

1. **D-8 Q-A — 낡았다.** 이 문서는 "PENDING · 완료 0/4 · 결과 0건"으로 적었으나 실제로는
   **2026-09-11에 4/4 `COMPLETED 0:0` 완주(10.641 GPU-h)** 했고 claims-auditor 결과 감사도 끝났다.
   ★**원인**: 정본 `EXPERIMENT_ROADMAP.md` 자신이 모순 상태다 — `:160-163`(변경 로그)은 완주를
   기록하는데 **본문 `:592-593`은 "제출 상태 PENDING·완료 0/4·결과 0건 / Q-A는 여전히 결과가
   전혀 없다"로 남아 있다**(`:195`도 같은 계열). 이 세션은 본문을 인용해 틀렸다.
   ⇒ **doc-steward 회부**(아래 §5).
2. **D-10 — 부분 완료로 정정.** I-1 · §3 280–283 · §5-11은 `b330ae8`로 적용됐고,
   **미완은 I-2 · I-3 · `venue_positioning.md` M2 협소화**다.

---

## 5. 열린 항목 / 다음 세션 시작점

**현재 실행 계획의 정본은 이 문서가 아니라
`handoff-report/roadmap_review_2026-09-27.md`(rev2, claims-auditor 감사 반영)** 이다.
그 문서가 카드 필요도를 재판정했고(D-6-b **불필요** · D-2 **보류** · P11·V-char·V-cap 신설),
실행 기판은 **VESSL Cloud**로 옮겨졌다. 이 세션의 설계 카드는 그 입력이었을 뿐이다.

### 5.1 GPU 0으로 지금 가능한 것
| 항목 | 내용 |
|---|---|
| **D-7-a** | 부분 중첩 분할이 green context 경로에서 **구매 가능한가** 코드 조사. "불가"면 D-7-b를 설계하지 않고 stake #1처럼 스코프 선언으로 닫는다. `CONSENSUS.md` §5 열린 항목 11이 **회부 중** |
| **D-6-a 스크립트 등록** | `split_index_census.py`는 산출은 끝났으나 **저장소 등록·selftest 미완** — 인용 전 필요(roadmap_review) |
| **I-2** | `residency_census_2026-09-21/README.md:95` · `residency_census.py:55`의 `multiplexing_mixin.py:880-893` → 현재 트리 대응부는 **1169-1182**. sha 없는 줄 인용이라 `check_line_citations.py`가 못 잡는다 |
| **I-3** | `multiplexing_mixin.py:451-453` sticky `ENABLED` 로그 문구(두 plain 그룹 뒤바뀜) |
| **doc-steward 회부 (신규, 이 체크포인트 산출)** | `EXPERIMENT_ROADMAP.md` **Q-A 본문 `:592-593`·`:195`가 변경 로그 `:160-163`과 모순** — 본문이 완주를 반영하지 않았다. 교훈 계열: **변경 로그만 갱신하고 본문을 두면 다음 독자가 본문을 인용한다**(이 세션이 실제로 그렇게 틀렸다) |
| **doc-steward 회부 (승계)** | `venue_positioning.md` §1 C1(`:398-401`)·§0.2(4)(`:322`)가 정본(`CONSENSUS.md:3439`, 감사 M2)보다 낡음 |

### 5.2 GPU가 필요한 것 — 선행 게이트 먼저
`roadmap_review_2026-09-27.md` §0의 선행 게이트 7개가 **전부 미충족**이다. 특히:
- **기판 동등성 판정**(사용자 + 규칙층 감사): VESSL A100 SXM4(driver 580/CUDA 13.0, K8s,
  클럭 고정 불가) vs KISTI A100. **판정 전에 KISTI 수치(d44·λ\*·격자)를 VESSL에 대입하지 않는다.**
  ★**λ\* 일치는 동등성 근거가 아니다**(교훈 247).
- **n 독립성**: 한 job 안 다중 boot ≠ 독립 n ⇒ 비교 캠페인은 **독립 job ≥2 × n≥4**.
- **V-0(= D-0) 기판 점검**(0.05–0.1 GPU-h)이 전부의 전제. 계약 전 확인: **CUDA ≥ 12.4**
  (upstream #8701·#9021이 그 실패 모드를 고쳤다 — 미만이면 green context 심볼 미해결).
- **D-1 E2**는 등록 원안 그대로 유효(**축소 금지** — R1 전 seed·R5 산포·R4″·Q2 n=4가 깨진다),
  단 **새 OVERRIDE가 사용자 결정 사항으로 보류 중**이고 VESSL 이식 追記(A13)가 필요하다.
- **SLURM 계정 블로커**는 별개로 여전히 미해소(사용자 영역).

---

## 6. 미완·주의

- **claims-auditor에 안 건 것**: 이 세션의 **S3 재집계 수치**(PART 분해)와 **§2.9 층 단위 발사
  집계**는 기술 집계이고 감사에 걸지 않았다. `CONSENSUS.md` §3 항목281이 재집계를 등재했지만
  **성능 판정은 0건**이다. §2.9는 정본에 등재되지 않았다 — 인용 전 감사 필요.
- **D-5 · D-6-b · D-7-b의 기대 밴드는 [제안] 등급**이다(이 세션이 처음 적은 예측).
  **사전등록 절차를 거치기 전 집행 불가**이고, 결과를 그 밴드로 채점하면 게이트 8 위반이다.
- **`s8p_prefill` 인용 금지 유지**(claims-auditor 미통과). D-5의 지지 데이터로 썼지만
  정본 근거로는 쓸 수 없다.
- **방치된 job 없음.** 이 세션은 GPU 0이고, 남은 배경 작업도 전부 종료했다.
- **Bullet §4.4 vs HE0는 판정하지 않았다.** `CONSENSUS.md` §5-11에 회부 상태로 등재돼 있고
  **HE0의 등급·문구·정책 순위는 불변**이다(§1-7).
- **artifact는 저장소 밖**이다. 갱신하면 markdown 원본과 어긋날 수 있다 — 원본이 우선.
