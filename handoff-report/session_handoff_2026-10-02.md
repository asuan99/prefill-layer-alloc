# 세션 핸드오프 — 2026-10-02 체크포인트 (작업 기간 2026-09-28 저녁 → 2026-10-01)

> **GPU 지출 0 · 새 성능 판정 0 · Claim 등급 변경 0.** 이 기간의 세부 追記는 `handoff-report/session_handoff_2026-09-28.md`의 追記 2–7에
> 시간순으로 남아 있고, 이 문서는 그것을 한 장으로 묶은 다음 세션 시작점이다. 감사 7회(claims-auditor 6, engine-porter 1)·재집계 1회(result-analyst).

## 1. 이번 세션 요약

TTFT 측정 정의 검토로 시작해(artifact 미반영 지시), 사용자와의 토론을 거쳐 **설계 기준선(anchor)**을 프로젝트 전체 좌표계로 고정했다
(`reports/system_design_anchor_2026-09-30.md`, artifact "Hybrid PD-mux Anchor" v1→v6). 중심 주장은 "SM 배분형 PD-mux 스케줄러(Bullet/MuxWise)의 hybrid
확장은 자명하지 않다"로 재정의했고, 이를 위한 cost model 설계 공간·선행 cost model 정독·정의의 사각지대·핵심 동기(attention vs SSM의 L별 비용) 검토를
문서화했다. 그 과정에서 claims-auditor가 연속으로 세 가지를 정정했다 — (a) S0 rev1(hybrid 재정렬·일시중단 결정 발산 검사)은 항등식·허수아비 baseline이라
`NO-GO` (b) A4 분리 재구성(decode 측 SM = 정적 표)도 `NO-GO`: **단일 모델에서 P_hyb 표와 fitted P_gen 표는 재매개화라 발산 D ≡ 0**, hybrid 귀속은 모델 hold-out만
(c) 핵심 동기 검토 rev1 `INACCURATE` — 내가 만든 "교차점 불일치"는 attention 모듈 필드와 코어 필드를 비교한 가짜였다. 마지막으로 사용자 문제 제기("선행과
반대되는 결과가 설계 오염 아닌가")로 **HE0 오염 감사**를 돌려, HE0의 부호는 견고하나 headline은 크기·범위 과장임을 원자료로 확인했다.

## 2. 결정·측정

### 2.1 사용자 결정
- 설계 기준선을 프로젝트 전체 anchor로 고정(2026-09-30). 개정은 사용자 승인 + doc-steward 날짜 단 개정으로만.
- 선행 논문의 "future work" 언급은 **인용 의무이지 novelty 감점 근거가 아니다** · **엔진 수정 비용은 중단 사유가 아니다**(2026-09-30).
- TTFT 정의는 그대로(클라이언트 송신→첫 토큰, open-loop, 거부 없음). 우리 "admission 차단"은 보류이지 거부가 아님(`--max-queued-requests` 미설정).
- 타겟 워크로드 = 긴 문서 요약(radix OFF 정합), RAG는 radix ON 확장 뒤. 입력 길이는 배치 토큰 ≤ 65536(주 영역)과 초과(1층 span 하한, 별도 영역)로 분리.
- "조성"은 M(모델, 배포 상수) · U(요청) · E(엔진)로 분리해 기록(정의 사각지대에서 H[기판] 층 누락 지적 → M/U/E/H).

### 2.2 감사·재집계 결과 (전부 GPU 0)
| 대상 | 판정 | 핵심 | 기록 |
|---|---|---|---|
| S0 rev1 설계 | `NO-GO` (死因 5) | 발산률이 분기 순서·개방 루프에서 연역(항등식) · G-PAUSE는 Bullet 허수아비 · `estimate()`에 hybrid 항 없음 | `results/hybrid_sched_s0/audit_s0_rules_2026-09-30/VERDICT.md` |
| PAUSE·재정렬 훅 실현성 | engine-porter 검토 | PAUSE 구현 가능(cudagraph 양립, idx0 전환+드레인, `update_running_batch` 쌍 건너뜀, 케이던스 동결, admission latch 데드락 불변식 4개) · radix OFF에서 `lpm`/`dfs-weight`는 조용히 FCFS · estimator는 `attention_ratio`를 읽지 않음 | `results/hybrid_sched_s0/engine_porter_review_2026-09-30.md` |
| A4 분리 재구성 | `NO-GO` (死因 2) | P_gen+recal 오프라인 붕괴 · **단일 모델 표 발산 = 재매개화 항등식(D ≡ 0)** · 발사량 레버 엔진 대응물 부재 · A4b-2는 layer-type 정책 계열 | `results/hybrid_sched_s0/audit_a4_split_2026-10-01/VERDICT.md` |
| 핵심 동기 검토 rev1 | `INACCURATE` (정정 16) → rev2 반영 | 방어 가능 문장 = "코어 vs mixer 시간이 L≳2k에서 L^1.92 / L^0.95"(스코프 필수) + 조건부 per-op 교차점 ≈2.75k(백엔드 의존) · "span 비용 = 조성×길이"는 NOT-YET-SUPPORTED · "decode SSM은 SM 둔감"은 §1-21·C2가 반증 · 896776 수치 인용 금지 | `reports/prefill_attn_ssm_length_review_2026-10-01.md` §8 |
| 선행 대비 오염 (논리) | `MIXED` (대조 17, 후보 15) | 정면 충돌 문면 4·실체 2(버스트 prefill-ward 부호, Bullet ablation 오독) · 신규 최대 후보 **running cap 48 = mamba pool 48** · HE0 bind+GATE는 d44 미도달(허수아비) | `results/he0_contamination_2026-10-01/AUDIT_LOGIC_VERDICT_2026-10-01.md` |
| HE0 실현 분할 (정량) | REAL(재구성 기반, 감사 전) | arm 차이는 동거 구간에만 · **HE0 headline 효과 −2.7%(legacy 술어) < 3% 게이트** · 정본 술어 −19.9%는 ITL p95 59–61 ms cliff 증폭 · TTFT 차이 95%+가 대기열 · §1-10·§1-17·§1-11 서술 불일치 · self-test PASS | `results/he0_contamination_2026-10-01/{RESULT,SYNTHESIS}_2026-10-01.md`, `he0_realized.py` |

수치는 전부 KISTI A100·Zamba2·no-cudagraph micro 또는 HE0 아카이브 재구성이며 VESSL 수치와 섞지 않는다. 재집계 동거 비율은 가중 규약 의존이 커서 범위로만 인용.

## 3. 코드·문서 변경 (이 커밋에 포함)

- **신규 정본 부속 문서**: `reports/system_design_anchor_2026-09-30.md`(anchor; §7 입력 길이 영역·축의 주체, §8 3-a에 L>65536 셀) ·
  `reports/cost_model_design_space_2026-10-01.md`(CM-1…7 + "누가 정하는가" 열, X-1…11 + 원문 정독 정정, A1–A11, 배치 지도 §5, 조성 3층 §1.0; CM-1·X-7의 "SSM SM 둔감" 전제 철회) ·
  `reports/prior_cost_models_bullet_muxwise_2026-10-01.md`(Bullet SRM·α·간섭·Alg.1 / MuxWise 식 1·2·contention guard·N_PL, 로컬 camera-ready 정독) ·
  `reports/definition_blind_spots_2026-10-01.md`(D-1…D-12, 최상위 5: p95가 PAUSE 간격을 가림·1층 하한·변화 trace 정상성 미정의·결정 반영 지연·H 층 누락) ·
  `reports/prefill_attn_ssm_length_review_2026-10-01.md`(rev2) · `reports/ttft_measurement_definition_review_2026-09-28.md`([제안] 규약, 전제 1 해소, retract 제외, §5 max-ITL 병기).
- **감사·설계 기록**: `workspace/engine-port/results/hybrid_sched_s0/`(S0 설계 rev1·A4 제안 + 판정서 2 + engine-porter 검토) ·
  `workspace/engine-port/results/he0_contamination_2026-10-01/`(논리 판정서·재집계 스크립트/JSON·결과·종합).
- **artifact 스냅샷**: `handoff-report/artifacts_2026-09-30/hybrid_pdmux_anchor.html`(v6) + README. 발행본 https://claude.ai/artifact/WEgWi5nvCLJCM7Niu8XsTZ (비공개).
- **정본 반영(2026-09-30 doc-steward)**: PROJECT_STATUS 2026-09-30 배너 · CONSENSUS rev82(§4 anchor·TTFT 메모 행, §5-14 부기, §3 항목18 追記) · ROADMAP 배너(5-b "동적" 재정의, S1 카드 예정) · CEM Claim E 追記 · 메모리 `system-design-anchor.md`.
- 기타: `handoff-report/vessl_execution_plan_2026-09-24.md` §3-4 TTFT 규약 항목, `handoff-report/session_handoff_2026-09-28.md` 追記 2–7, `line_citations.json` 스냅샷.

## 4. 열린 항목 / 다음 세션 시작점

**사용자 결정 대기 (우선순위 순)**
1. **HE0 정본 범위 축소 반영** — 감사 D-1…D-8 + 효과크기 < 3%·cliff 단서. 등급 유지, 문구만 축소. 승인 시 doc-steward.
2. **재집계 결과 감사**(GPU 0): C-1 자리 분해 잔차, C-4 cap 48 구속 시간비, 856964 no-gate 붕괴 귀속.
3. **E-1 cap 48 vs 192 대조**(GPU ≈10–20 GPU-h): 버스트 prefill-ward 부호 충돌을 가르는 유일한 실험. 대여 GPU 확정 후 사전등록.
4. hybrid 귀속의 1차 운반체를 S0′ → **A11 모델 hold-out**으로 옮길지 · P_gen을 "대상 모델에서 기울기를 재지 않는" 정의로 바꿀지(Transformer 대조 GPU 또는 roofline 허수아비 위험) · A4b-2(층 종류별 비용 가중 발사량)를 layer-type 정책 재개로 볼지.
5. 이전부터 대기: VESSL 예산·크레딧·레지스트리 · E2 새 OVERRIDE · 5-b/4-a arm 채택 · 모델 업로드 범위.

**doc-steward 회부 (누적)**: 부속 문서 4건(cost model·선행 cost model·사각지대·동기 검토)의 CONSENSUS §4 등재 · §1-3 배너 "21×" 재검토 · `longcontext_trace_plan.md` §1 교차점 정합 · venue_positioning의 "Bullet 승리 = 프로세스 분리" 삭제 · §1-10/§1-17/§1-11 서술 정정(사용자 승인 후) · BulletServe 로컬 클론 잔재 등재 여부.

**다음 GPU 작업 순서(불변)**: V-0/D-0 기판 점검 → 3-a(대칭 버킷·`knee2d` 파서 ZBPT2 수리, L {1k–32k, >65536}, n≥2) → 3-b → S0′(또는 A11) → S1(pause manager) → 5-a → 5-b. E-1은 3-b 이후 아무 때나.

## 5. 미완·주의

- 재집계(`he0_realized.py`) 결과는 **claims-auditor 결과 감사 전**. telemetry 없이 서버 로그·클라이언트 jsonl 재구성 — 분할은 코드 규칙의 함의.
- 핵심 동기의 방어 가능 문장은 단일 모델 micro 한정. "Nemotron-H에서도 같다"는 쓸 수 없다(백엔드 교락, 미측정).
- TTFT 규약·PAUSE max-ITL 병기·phase 단위 정상성은 전부 [제안] — 사전등록 감사 전.
- **`he0_realized_2026-10-01.json`은 커밋하지 않았다**: 부동소수 출력의 숫자 부분열이 인용정지 패턴(정지 레지스트리의 소수·job 번호 패턴)과 우연히 충돌해 8건 거짓 양성. 결정적 산출물이므로 `python3 workspace/engine-port/results/he0_contamination_2026-10-01/he0_realized.py`로 재생성(sha256 `9cfed852…`, RESULT 문서에 기록). 로컬 작업 트리에는 남아 있다.
- 실행 중 job 없음. 로컬 GPU 실행 없음. 미커밋 상태 없음(이 커밋 이후).
