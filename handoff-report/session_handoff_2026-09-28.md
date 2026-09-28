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
5. **다음 세션 이월(사용자 지시 2026-09-28 말미)** — **TTFT 측정 정의 검토**: 도착(대기열 진입) 기준 TTFT는 문헌 표준(open-loop, 클라이언트 측)이며 우리 하네스도 동일(`workloads.py` Poisson 도착, `analyze.py` per-request `ttft_ms`). 과부하 구간의 TTFT 크기는 시스템 상수가 아님(`CONSENSUS §3` 항목32, 7.24 s 인용 금지[CS-OK] — 금지 사실을 서술)이 이미 정본. 검토할 것: (a) 결과 규약에 **정상성 검사**(대기열 길이 단조 증가 → "불안정" 표기, TTFT 수치 미제출) 명시 (b) telemetry로 **TTFT = 대기 + prefill 처리** 분해 규약 (c) 로드맵 그림 3·시뮬레이션 설명의 "TTFT 폭발" 문구를 "대기열이 쌓여 용량 밖으로 밀려남"으로 교정 (d) 검토용 6절 원칙에 (a)(b) 두 줄 추가. 아직 미반영.
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
