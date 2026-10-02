# 세션 핸드오프 — 2026-09-28 체크포인트

> **대상 세션**: 2026-09-23 → 2026-09-28 연속 세션(VESSL 이행 계획 · 이미지 빌드 · span/step 설계 메모 ·
> radix 결정 · 로드맵/기여 정리 · artifact 2종 · claims-auditor 감사 3회).
> **GPU 지출 0 · 새 성능 판정 0건 · Claim 등급 변경 0건 · 정본 수정은 doc-steward 회부분만.**
> 2026-09-24~27 델타의 대부분은 이미 커밋돼 있다(`725323f`…`39a5f9f`, `1f3a89c`). 이 문서는 그 목록과
> **2026-09-28 미커밋 델타(기여 재구성·감사 2건·artifact 스냅샷)**를 함께 적는다.

---

## 1. 이번 세션 요약

GPU 실행 기판을 KISTI Neuron(폐쇄)에서 **VESSL Cloud(us-west-2, A100 SXM 80GB $1.55/h)**로 옮기는 준비를 끝냈다.
설정 계획·저장소 배치·2단계 이미지·로컬 Docker 빌드(draft2, 27.0 GB, sglang-kernel cu130 sha256 고정)·예산(시나리오 A–D)을
`handoff-report/vessl_cloud_setup_plan_2026-09-24.md`와 Notion에 기록했다. 실험 방향은 사용자와 합의해 **whole-phase 분할 유지,
layer-type 런타임 정책 死 불변, prefill은 조각(span)·decode는 step 경계에서만 재결정 가능**으로 정리했고(설계 메모 rev2),
옛 λ0 telemetry 재집계로 **shape A는 기본 예산에서 prefill 조각이 1개**(prefill 도중 결정 지점 없음)임을 확인했다(판정 아님).
radix cache는 **OFF 1순위**(ON은 추후 확장, 합성 혼합 trace 없음). 마지막 이틀은 교수 보고용 로드맵 artifact와 내부 검토 artifact를
만들고 claims-auditor 감사를 3회 돌려 **핵심 기여 3개** 서술을 확정했다. 기여 2(기전)→기여 3(운영 가이드와 여유)의 연결은
"정책은 decode가 앉을 자리를 알아야 한다"는 요구 조건으로 세우되, **4-a 추가 arm(앉을 자리 아래 고정 배분)에서 굶김 연쇄가
재현될 때만 인과로 쓴다**고 규칙화했다.

## 2. 결정·측정 (전부 GPU 0)

### 2.1 사용자 결정(이 세션 내)
- VESSL: region us-west-2 고정 · 이미지 빌드는 로컬 Docker · HF 모델은 리비전 고정 후 in-region Cluster volume에 내려받기 · 볼륨 생성은
  비용 발생이라 보류(레지스트리 결정과 함께 사용자 확인 사항) · 모델 업로드는 Nano-9B만 권장.
- radix cache OFF가 실험 검토 최우선, ON은 추후 확장, 합성 short↔long 혼합 trace는 만들지 않음(`reports/longcontext_trace_plan.md` §8·§8.1).
- I-1 정정 승인(`manual_divisions` 59/59 → 55/59, `CONSENSUS.md` rev81) · en-dash 줄 인용 치환 · census 원시 파일은 로컬 보관(.gitignore).
- **보고용 artifact는 핵심 기여 3개만**, 세부·한계·arm 목록은 검토용 artifact로(2026-09-28).
- 기여 2→3 연결을 핵심 논지로 세우되 검증 고리(4-a 추가 arm)를 명시(2026-09-28, "진행").

### 2.2 감사 결과(claims-auditor, 읽기 전용, 3회)
- **로드맵 검토 rev1→rev2**(2026-09-27, `roadmap_review_2026-09-27.md`에 반영·커밋됨): Q-A 완료라 P11 대체 불가 · E2 등록 유효(축소 금지) ·
  D-10 부분 · [예보]→[제안] 등급.
- **감사 1(2026-09-28)** `reports/audit/2026-09-28_contributions/AUDIT_1_prior_art_and_arm_2026-09-28.md`:
  MuxWise·Bullet은 **논문 수준**에서 층 단위 prefill 진행 + SM 배분을 함께 조절하나, **MuxWise 공개 엔진(우리 기판)에는 N_PL도 추정기도
  없고 decode_bs 문턱 표(upstream 주석 `temporary demo`)만 있다**. 우리 config는 "문턱 전부 0"이 아니라 79파일 중 60 전부-0 · 15 guard 행 ·
  4 자동 격자. 부하 인덱스 선택기는 P1 auto-grid agnostic arm으로 이미 돌았다(best-static 대조는 미실행). **MuxWise식 arm은 설정만으로
  불가**(N_PL은 엔진 수정, 문턱 표는 노브 다중 이동) → 새 사전등록 대상. 2,940×/27×(knee2d_wide, UNAUDITED)와 residency 3–10/43–98 %
  (미등록 스크립트)는 **인용 불가** → artifact에서 제거.
- **감사 2(2026-09-28)** `.../AUDIT_2_core_contributions_2026-09-28.md`: 보고용 카드 사실 오류 4건(KV 작음 ← 정본 반대 · "균등" ← agnostic 오역 ·
  운영 규칙 4-c 확인 ← 4-c는 C1 · prefill 민감도 "같다" ← L≤512 다름)과 과장("처음으로", attention 원인 단정, 다른 GPU 이전) 전부 교정.
  **기여 3은 "운영 가이드와 그 여유의 크기"로 재구성**(5-a 격차 = B2 oracle 상한이라 5-b 결과와 무관하게 긍정 산출물).

### 2.3 CPU 재집계(판정 아님, 스크립트 미등록 — 인용 전 등록·자체검사 필요)
`design_memo_span_step_boundary_costmodel_2026-09-24.md` §7–§8(커밋됨): λ0 908623 telemetry에서 shape A 조각 1개(관측 배치 ≤1,024 토큰),
shape B 7/14 조각; rate가 오르면 prefill 중 큐 도착은 늘지만 decode-bs 변화는 회당 0.2–0.4에 불과. residency 범위는 규약 의존이라 인용 금지.

## 3. 코드·문서 변경

**커밋됨(2026-09-24~27)**: `725323f` VESSL 실행 계획·Dockerfile(`workspace/engine-port/env/docker/{Dockerfile.pdmux-sglang,
requirements.pdmux-sglang.txt,build_image.sh}`)·설계 메모 rev2 · `509c655` 메모 §7 · `2da6779` radix 결정+§8.1 · `d535918` en-dash 치환 ·
`3aa794f`/`029a771` 아카이브·gitignore · `b330ae8` I-1 정정 rev81 · `95adf8e` 예산 절 · `39a5f9f` roadmap_review rev2 · `1f3a89c` 09-22 세션 handoff.

**이 커밋(2026-09-28)**:
- `handoff-report/session_handoff_2026-09-28.md`(이 문서).
- `reports/audit/2026-09-28_contributions/AUDIT_{1,2}_*.md` — 감사 판정 2건(대화에서 옮김).
- `handoff-report/artifacts_2026-09-28/` — 보고용 v17·검토용 v6 HTML 스냅샷, 차트 생성기, 감사 반영 편집 스크립트, README(정본 아님 명시).
- doc-steward 반영분: `reports/CONSENSUS.md`(§1(D) `external/muxwise/` 경로 부존재·출처 미검증 註記, §5 열린 항목 12–14 신설: YAML 주석 오류/`stream_idx` 잠재 위험 · `p0_residency.py` 미등록 · MuxWise식 arm은 엔진 수정) · `reports/paper/venue_positioning.md`(같은 경로 註記 2곳, "하나도 없다" HISTORICAL 배너 + REPORT:426 문구 + Bullet Fig. 20a 각주, "load-dependent static" 스코프 註記).
- 메모리: 신규 `paper-contribution-framing.md`(project) · `report-vs-review-artifact-split.md`(feedback) · `deconfound-measurement-lessons.md` 항목 259–262(CONSENSUS §3/게이트 번호는 미부여) · `MEMORY.md` 인덱스 갱신.

**저장소 밖**: artifact `6URdjxzXLB2rCrG7Qq3JWS`(보고용 v17) · `NhCAwkHtZb5LeYJqQ4e8fH`(검토용 v6) · Notion 설정 계획 페이지(§10 빌드 결과 포함)
· 로컬 Docker 이미지 draft2 · `~/Experiments/muxwise/sglang-slo_config/` MuxWise yml 사본(출처 미검증).

## 4. 열린 항목 / 다음 세션 시작점

1. **사용자 결정 대기**(검토용 7절 체크리스트): 예산 시나리오·크레딧 충전 · 레지스트리/공개 여부 · 결정 ①(기판 동등성) · 결정 ②(E2 새 OVERRIDE) ·
   5-a CPU 사전 확인 여부 · 모델 업로드 범위 · **5-b agnostic 기준선 arm 채택 여부** · **4-a "앉을 자리 아래" arm 채택 여부**(≈0.2–0.3 GPU-h, 4-a와 함께 등록).
2. **실행 전 필수**: V-0/D-0 기판 점검(green-context probe, CUDA ≥ 12.4) → 3-a·3-b 사전등록(규칙층 감사) → 4-a 등록(추가 arm 포함 시 그 arm도).
3. **회부(doc-steward)**: (a) `CONSENSUS.md`:1219-1220·`venue_positioning.md`:553/598의 `external/muxwise/` 경로가 저장소에 없음 →
   실제 위치·출처 미검증 註記 (b) VP:322 "하나도 없다" stale → REPORT:426 문구 (c) VP:543/580 "MuxWise dynamic ≈ 부하 의존 static 표"를
   공개 엔진 선택기로 스코프 (d) `s0_deconfound/pdmux_p16_d16.yml` 주석 오류(셋째 값은 mixin에서 문턱) 등재 (e) residency 스크립트
   (`p0_residency.py`, scratchpad) 미등록 상태 등재 — 인용 전 등록·자체검사.
4. **도구**: `check_line_citations.py`가 en-dash 범위를 인식 못 하는 공백(정규식 보강) · `CONSENSUS.md` bare 줄 인용 36건 · `job_entry.sh`
   (`git config --global --add safe.directory '*'` 포함)·vessl 실행 스크립트 미작성 · experiment-runner 등 에이전트 정의의 VESSL 규약 R1–R9 미반영.
5. **다음 세션 이월(사용자 지시 2026-09-28 말미)** — **TTFT 측정 정의 검토**: 도착(대기열 진입) 기준 TTFT는 문헌 표준(open-loop, 클라이언트 측)이며 우리 하네스도 동일(`workloads.py` Poisson 도착, `analyze.py` per-request `ttft_ms`). 과부하 구간의 TTFT 크기는 시스템 상수가 아님(`CONSENSUS §3` 항목32, 7.24 s 인용 금지[CS-OK] — 금지 사실을 서술)이 이미 정본. 검토할 것: (a) 결과 규약에 **정상성 검사**(대기열 길이 단조 증가 → "불안정" 표기, TTFT 수치 미제출) 명시 (b) telemetry로 **TTFT = 대기 + prefill 처리** 분해 규약 (c) 로드맵 그림 3·시뮬레이션 설명의 "TTFT 폭발" 문구를 "대기열이 쌓여 용량 밖으로 밀려남"으로 교정 (d) 검토용 6절 원칙에 (a)(b) 두 줄 추가. ★**처리(2026-09-28 후속 세션)**: (a)(b)는 `reports/ttft_measurement_definition_review_2026-09-28.md` §3.1–§3.2로 작성하고 `vessl_execution_plan_2026-09-24.md` §3-4에 감사 대상 항목으로 등재(감사 전 [제안]). (c)(d)는 **사용자 지시로 artifact에 반영하지 않음** — 문안만 같은 메모 §3.3–§3.4에 둠. 신규 사실: `bench_serving` 경로는 도착 시각을 기록하지 않아 송신 표류 검증 불가, `--max-concurrency` 런의 TTFT는 도착 기준 아님, 분해 전제 3건(engine-porter) 미확인.
6. **시뮬레이션 재작업(2026-09-28 저녁, artifact v18→v19, 스냅샷 `artifacts_2026-09-28/pdmux_roadmap_v18.html`)**: 워크로드 4종 × 배분 4종 분리, 요청별 진행 줄(대기→prefill→decode), span = 층 분할 표기(`층1–8`), **decode 배치가 비면 prefill이 108 SM 전부**(엔진 `adjust_stream_groups` 세 갈래: 분할 / (0,108) / (108,0), 유지 ON도 동일 — 코드로 확인) 반영, 동거 비율 계기. 개념 애니메이션이며 측정 아님.
7. **catch-up 시작점 한 줄**: "VESSL 실행 전 사용자 결정 8건 대기 · 기여 3개 서술 확정(artifact v19/v6) · 이월 1건: TTFT 측정 정의(정상성 검사·대기/처리 분해) 검토 · 다음 GPU 작업은 V-0 기판 점검."

## 5. 미완·주의

- **claims-auditor에 안 건 것**: 설계 메모 §7–§8 재집계·`p0_residency.py` 산출(판정 아님, 미등록). 보고용 H3-g/워크로드 표의 13.64 %·97.63 %는
  등록 규약(a_r4/b_r3) 값이며 **동거 비율이 아니다**.
- **인용 금지 재확인**: 2,940×/27×(knee2d_wide) · residency 3–10/43–98 % · 13.5/85.6/91.9 · 8B C2 2.36–2.91× 범위(보고용 H3-b는 모델별 점추정 +
  "순서·격차 주장 안 함" 문구로만 유지).
- **기여 2→3 인과 서술은 4-a 추가 arm 재현 전까지 해석**이다. 기여 1의 "attention 층 때문"은 hold-out 없이는 식별 불가라 카드에서 뺐다.
- **MuxWise식 arm(N_PL 정합)은 엔진 수정**이라 현 계획 밖. 5-b에 남긴 것은 agnostic 기준선 재현 arm뿐이며, 그 arm이 져도 "MuxWise가 졌다"가 아니다.
- **E2는 등록 원안 그대로**(축소 금지), 새 OVERRIDE는 사용자 결정 사항. SLURM 계정 블로커는 무관해졌다(기판 이전).
- 방치된 job 없음. 로컬 GPU 실행 없음. Docker 이미지 draft2는 GPU 검증 전(CPU import·트리 동기화만 확인).

---

## 追記 2 — 2026-09-30 후속 세션 (GPU 0 · 새 성능 판정 0 · Claim 등급 변경 0 · 커밋 없음)

**사용자 논의 흐름**: (1) TTFT 측정 정의 검토 반영(artifact 미반영 지시) → (2) prefix cache 고려 정책 변경 여부(권고: 현행
radix OFF 순서 유지, 캐시는 `S_prefill` 분해로 공변량화) → (3) TTFT는 정의 불변, "차단"은 거부가 아니라 **보류**(`--max-queued-requests`
미설정, 무한 대기열) → (4) 타겟 워크로드 = 긴 문서 요약(one-shot, radix OFF 정합) · RAG는 radix ON 확장 전까지 이름 붙이지 않음 →
(5) 사용자 가설 "shape A의 HE0 근거는 shape B에서 다른 양상, 동적이 이길 영역을 A가 못 보여줌" — 정본 §5-5/§5-6이 이미 열어 둔
문, 확인 형태는 **5-a 긴 쪽(stake #1) → 5-b 변화 trace(W2/W4)**, "동적"의 정의 사전 고정 → (6) 사용자: 재정렬·일시중단 미착수,
"Bullet on hybrid" 반박 정당, hybrid 고유 정책이면 novelty 가능 — 검증 사다리 S0(CPU 결정 발산)→S1→S2 제안 → (7) 사용자 지시
"S0 감사로 검증".

**산출물(전부 작업 트리, 미커밋)**:
- `reports/ttft_measurement_definition_review_2026-09-28.md` — 정상성 검사 3종(F-E·게이트#12·`prefill_queue_depth` 추세)·`UNSTABLE`
  라벨·TTFT 분해(`W_queue`/`S_prefill`) 규약 [제안]; 거부 없음 전제 명시; 전제 1 **해소**(admission 1회 set, legacy/true-dual 동일)
  + 추가 발견(`prefill_run_batch_*` PD-mux 미기록 · retract가 `wait_queue_entry_time` 덮어씀 → retract 요청 제외 규칙).
  `vessl_execution_plan_2026-09-24.md` §3-4에 감사 대상 항목 추가. 줄 인용 스냅샷 등록(`line_citations.json`).
- `workspace/engine-port/results/hybrid_sched_s0/DESIGN_S0_REV1_2026-09-30.md` — S0 설계 rev1 → **규칙층 감사 `NO-GO`**
  (`audit_s0_rules_2026-09-30/VERDICT.md`, 死因 5: K1 항등식[발산률이 계산 전 연역, attn_ratio 무관] · K2 G-PAUSE는 Bullet 허수아비 +
  충실한 Bullet-on-hybrid ≡ H[`estimate()`에 hybrid 항 없음] · K3 해석 폭 반전 · K4 라벨 비전사 · K5 폐쇄 루프 미정의). 실행 안 함.
- `.../engine_porter_review_2026-09-30.md` — PAUSE 훅 실현 가능(cudagraph 양립·idx0 전환+드레인 필수·`update_running_batch` 쌍
  건너뜀·케이던스 동결·admission latch 데드락 불변식 4개) · radix OFF에서 `lpm`/`dfs-weight` 조용히 FCFS · estimator는
  `attention_ratio`를 읽지 않음(M1 메타필드 = 항등식) · S1 최소 구현 표. 착수 안 함.

**결정·판단(사용자 확인 필요 표시)**:
- "hybrid 전용 재정렬·일시중단 정책"은 **아직 보이지 않은 것**이지 주장 불가가 아니다(사용자 정정 2026-09-30). 감사 §6-6의
  "Bullet future work 예고"는 인용 의무로만 승계, novelty 감점 근거 아님. **novelty 조건 = 확장이 자명하지 않음을 보이는 것**
  (generic 예측기/규칙의 hybrid 실패 + hybrid 보정판의 교정) — Bullet의 "straightforward"를 검증 가능한 반대 주장으로 전환.
  **엔진 수정 비용은 중단 사유가 아니다**(사용자 결정): PAUSE는 pause manager(쌍 건너뜀·idx0 전환·케이던스 시계·latch 배타 소유)로
  1급 상태화. 금지 서술은 "이미 검증됐다"뿐.
- 순서: **3-a 실측 프로파일(SM×batch×ctx, 대여 GPU) 먼저** → S0′(결정 규칙 Bullet Algorithm 1 decode 하나로 고정, 예측기만
  P_hyb/P_gen/P_gen+recal) → 그 뒤에야 S1. 합성 격자 S0는 `MODEL_DEPENDENT` 이상 불가.
- 기존 계획(V-0/D-0 → 3-a·3-b → 4-a → 5-a → 5-b)은 불변. 이번 논의는 5-b의 "동적" 정의와 novelty 서술 범위를 좁혔을 뿐이다.
- 로컬 BulletServe 클론(`~/Experiments/BulletServe`)에 이 프로젝트의 미커밋 hybrid 이식 흔적(zamba2/falcon_h1 모델 파일, mamba
  state pool 통계, `hybrid_model_bench/`)이 있음 — R0d 옵션 3a(libsmctrl, 드라이버 제약 BLOCKED)의 잔재로 추정, 정본 미등재.
  doc-steward 등재 여부는 사용자 결정.

**doc-steward 회부(추가)**: (f) CONSENSUS §5 열린 항목 14에 S0 `NO-GO`·novelty 상한 부기 (g) 위 BulletServe 잔재 등재 여부
(h) TTFT 메모의 retract 규칙·`prefill_run_batch_*` 미기록 사실을 분해 규약 정본화 시 반영.

**catch-up 한 줄**: "TTFT 규약 메모+S0 rev1은 감사 NO-GO(설계 결함: 항등식·허수아비 baseline). 사용자 결정: hybrid 확장은
논문 의의 있음·엔진 비용은 중단 사유 아님 → 주장을 'Bullet/MuxWise 확장은 자명하지 않다'로 재정의, pause manager 설계 + P_gen+recal
baseline 필수, 3-a 실측 프로파일 선행; 커밋 대기 파일 6개."

**doc-steward 정본 반영 완료(2026-09-30, 같은 날 후속 회차)**: 설계 기준선 문서
`reports/system_design_anchor_2026-09-30.md` 등재, 시각화 스냅샷
`handoff-report/artifacts_2026-09-30/hybrid_pdmux_anchor.html` 경로 등재. 회부 (f)·(h) 처리
완료 — `PROJECT_STATUS.md` 최상단 배너(2026-09-30)·`CONSENSUS.md` rev82(§4 living-doc 표 2행
신설, §5 열린 항목 14 부기, §3 항목18 追記)·`reports/paper/EXPERIMENT_ROADMAP.md`·
`CLAIM_EVIDENCE_MATRIX.md` Claim E 행 갱신. 회부 (g)(BulletServe 잔재 등재 여부)는 **사용자
결정 대기, 미처리**(이번 회차 스코프 밖). 새 성능 판정 0건·Claim 등급 변경 0건·커밋 금지.
- artifact "Hybrid PD-mux Anchor" URL: https://claude.ai/artifact/WEgWi5nvCLJCM7Niu8XsTZ (v1, 2026-09-30).
- **2026-10-01 追記**: 사용자 지시로 cost model 설계 공간 문서 `reports/cost_model_design_space_2026-10-01.md` 작성(CM-1…7 · X-1…11 ·
  A1–A11) + artifact "Hybrid PD-mux Anchor" v2(§9–§11 추가, 같은 URL) + anchor §10 포인터. 측정 0·GPU 0. 정본 등재는 doc-steward 회부
  대기(anchor 부속 문서로 §4 표에 행 추가 필요).
- **2026-10-01 追記 2**: (a) `reports/prior_cost_models_bullet_muxwise_2026-10-01.md` — Bullet SRM/α/간섭/Alg.1, MuxWise solo-run 회귀(식1·2)/
  contention guard/N_PL 디스패처를 로컬 camera-ready PDF에서 직접 정리(수치는 논문 주장). ★정정: 선행 model의 **형식**은 hybrid 단일 모델·
  고정 SM에서 살아남는다 — 깨지는 것은 θ 비율의 SM 의존(X-7)·N_PL 층 균일(X-4)·mamba 슬롯 변수 부재(X-6)·간섭 대리변수(X-5). P_gen 정의를
  "MuxWise 식 2 재적합 + Bullet α 보정"으로 강화(cost model 문서 §2 정정, §5 배치 지도 신설). (b) artifact v3 §12 알고리즘 배치 지도(L0–L4,
  ★/☆ 차별점). GPU 0·측정 0. doc-steward 회부: 두 부속 문서의 CONSENSUS §4 등재, X-1 정정의 정본 반영.
- **2026-10-01 追記 3**: 사용자 지적("decode는 step-wise라 규칙이 큰 의미 없음") → 재구성 제안 `PROPOSAL_A4_SPLIT_2026-10-01.md`(A4→정적 표
  A4a + step 레버 PAUSE·발사량 A4b, S0′→표 발산) → 반영 전 claims-auditor 감사 **NO-GO**(`audit_a4_split_2026-10-01/VERDICT.md`): K1 P_gen+recal
  오프라인 붕괴, K2 단일 모델 표 발산 = 재매개화 항등식(D≡0; estimator clamp·보간 확인) — hybrid 귀속의 1차 운반체는 A11 hold-out. 차단 10(F1 arm
  삭제·발사량 엔진 대응물 부재·표 형식 불일치·Claim E 재정의·A4b-2 layer-type 계열·sticky OFF·PAUSE C_switch·문구 정정). HE0·E2·λ0와는 무충돌.
  **재구성 미반영, 사용자 결정 대기**: (i) A11을 1차 운반체로 승격할지 (ii) P_gen을 "대상 모델에서 기울기를 재지 않는" 정의로 바꿀지(GPU 비용 또는
  허수아비 위험) (iii) A4b-2(층 종류별 비용 가중 발사량)를 layer-type 정책 재개로 볼지. anchor 개정은 사용자 승인 + doc-steward.
- **2026-10-01 追記 4**: 사용자 지시로 "조성"을 M(모델, 배포 상수)·U(요청, 사용자 쪽)·E(엔진, 런타임 결정) 세 층으로 분리해 반영 —
  cost model 문서 §1.0 신설 + CM 표 "누가 정하는가" 열, anchor §7 "축의 주체" 문단, artifact v4(§9 표·callout). 요지: 배치 ctx 분포 = U×E,
  hybrid는 decode 비용에서 U 성분 비중을 낮춘다(SSM 비용 ∝ bs = E만), 기존 결과는 M 변경 시 결정 불변·U 변경 시 체제 변화(방향만).
  doc-steward 회부: 부속 문서 2건 CONSENSUS §4 등재 + 이 3층 구분의 정본 반영 여부.
- **2026-10-01 追記 5**: (a) artifact v5 — SVG 7개 레이아웃 겹침 수정(내용 불변). (b) 사용자 질의(span 수 vs span 길이, chunked-prefill
  병용 문제) → 정정: span은 토큰·층 예산 단위의 **층 축** 분할이며 chunk(토큰 축)가 아님, 엔진은 PD-mux에서 chunked prefill을 강제 OFF. 실제
  위험은 예산(65536) 초과 시 1층 하한. (c) `reports/definition_blind_spots_2026-10-01.md` 신설 — 정의 D-1…D-12의 사각지대, 최상위 5(p95가 PAUSE
  간격을 가림 · 1층 하한 · 변화 trace 정상성 미정의 · 결정 반영 지연 · 기판 H 층 누락). 함께 반영: anchor §7 입력 길이 두 영역, §8 3-a에 L>65536
  계측 셀, cost model CM-3에 chunk 경계 SSM state 이월 항, TTFT 메모 §5 max-ITL 병기. artifact v6 §13. 측정 0·GPU 0. doc-steward 회부 목록에 추가.
- **2026-10-01 追記 6**: 핵심 동기 검토 `reports/prefill_attn_ssm_length_review_2026-10-01.md` rev2 — rev1을 claims-auditor가 `INACCURATE`(정정
  16건)로 판정, 전부 반영. ★rev1의 "교차점 불일치"는 메인 세션이 896776의 attention **모듈** 필드를 코어 필드와 비교해 만든 가짜(원로그 재계산으로
  확인) + 896776 수치는 인용 정지 위반이었음. 결론: 방어 가능한 동기 문장은 "코어 vs mixer 시간이 L≳2k에서 L^1.92 / L^0.95"(스코프 필수)와 조건부
  per-op 교차점 ≈2.75k(백엔드 의존)뿐, "prefill 구성이 L에 따라 바뀌어 span 비용이 조성×길이 함수"는 NOT-YET-SUPPORTED, "decode SSM은 SM 둔감"은
  정본 §1-21·C2가 반증 → cost model 문서 CM-1·X-7 정정. doc-steward 회부: §1-3 배너 "21×" · longcontext_trace_plan §1 교차점 · CM-1/X-7.
- **2026-10-01 追記 7 — 오염 감사(사용자 지시 "1번 진행")**: `workspace/engine-port/results/he0_contamination_2026-10-01/` — 논리 감사
  `AUDIT_LOGIC_VERDICT`(claims-auditor `MIXED`: 선행 정면 충돌 문면 4·실체 2[버스트 prefill-ward 부호, Bullet ablation 오독], 오염 후보 15, 신규 최대 후보
  = running cap 48 = mamba pool 48) + 정량 재집계 `RESULT`·`he0_realized.py`(result-analyst, 재구성 기반, self-test PASS, 감사 전) + 종합 `SYNTHESIS`.
  결론: HE0 부호는 오염으로 안 뒤집힘, headline은 크기·범위 과장(legacy 효과 −2.7% < 3% 게이트, 정본 술어 크기는 ITL p95 59–61 ms cliff 증폭, bind+GATE는
  d44 미도달 허수아비, TTFT 차이 95%+가 backlog). 정본 불일치 3건(§1-10·§1-17·§1-11 856964 클라이언트 정지). 정본 미편집 — 사용자 결정 대기:
  (1) D-1…D-8 범위 축소 반영 (2) 재집계 결과 감사 (3) E-1 cap 48 vs 192 대조(GPU).
