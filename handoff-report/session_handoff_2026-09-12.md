# 세션 핸드오프 — 2026-09-11 (측정·등재) / 2026-09-12 (체크포인트)

> 직전: [`session_handoff_2026-09-11.md`](session_handoff_2026-09-11.md).
> 정본 반영 **완료**(`CONSENSUS` **rev65 → rev66**, 커밋 `38c1aca`·`379fd1d`).
> 이 세션의 모든 GPU 측정은 **2026-09-11 KST**에 끝났고, 이 문서는 2026-09-12에 쓴다.

## 이번 세션 요약

사용자가 catch-up 브리핑 뒤 **(1) 정리 + (2) 연구 방향 둘 다 병렬**을 지시했다. 정리 쪽은
Q-A 캠페인 누락 파일 28개 커밋과 도구 파일의 **존재하지 않는 경로 6곳** 정정이고, 연구 쪽은
**Q-B 사전등록**(→ **Q-B′**로 개명, 규칙층 감사 `GO-with-caveats`)과 **R2 트랙 복귀**
(latch 수정 → GPU correctness 1차 **FAIL** → 경합 수정 → 재실행 **PASS**, 결과 감사
`CONFIRMED(scoped)`)다. ★**새 성능 판정 0건 · arm 순위 0건 · 정책 순위 변경 0건 ·
HE0 불변 · stake #1 불변 · Claim D/E 등급 불변(둘 다 미검증)**. GPU 지출 **0.28 GPU-h**
(전부 R2 correctness; Q-B′는 GPU 0).

---

## 결정·측정

### A. 정리 (커밋 `74351d3`·`2cc7934`)
- Q-A 캠페인(jobs 906504–906507) **provenance 파일 28개**가 `bc0033c`에서 누락돼 있던 것을
  보강(각 라운드 `gpu_info.txt`·`meta_r*_d{16,44,54,92}.txt`·`pkg_versions.txt`·
  `runtime_source_manifest.sha256`, 약 13 KB). 선례 = `probes/p7_905994/`.
  ⚠️**직전 핸드오프의 "원자료는 `.gitignore` 제외" 서술 정정**: 10.8 GB `bench_*.jsonl`/`*.log`는
  실제로 ignore되지만 이 28개는 ignore 대상이 아니었다(단순 미커밋).
- `reports/paper/EXPERIMENT_ROADMAP.md:415` 헤딩 "후계 질문 (둘 다 아직 실행 전)" → Q-A 완주
  반영으로 정정.
- **도구 파일 경로 정정 6곳**(전부 `/scratch/ehmoon/whlee` = **git 밖**, 게이트 #157):
  `.claude/skills/catch-up/SKILL.md`(정본 CONSENSUS 경로) · `.claude/agents/doc-steward.md`
  (`workspace/engine-port/reports/paper/` → `reports/paper/`, 옛 `deprecated_reports` 표기) ·
  `.claude/agents/claims-auditor.md`(17행) · `.claude/agents/engine-porter.md`(`env/`·
  `src/patches/`) · `CLAUDE.md`(핸드오프 경로·`RESUME.md`).
  ★`claims-auditor.md` SHA-256 **`47e800cf…ae02` → `e7c2dedf…93b8`**(109행 불변, 규칙 내용
  불변, 경로 1줄만) — 게이트 #157 앵커이므로 `PROJECT_STATUS.md`·`CONSENSUS.md`에 날짜 追記.

### B. Q-B → **Q-B′** (커밋 `27b6ea7`, 정본 rev65 `38c1aca`) — GPU 0
- 초안(result-analyst) → **규칙층 감사 `GO-with-caveats`**(死因 0 · N1–N4 미발화 ·
  반전 **0/23표면**), 판정서 `results/longctx_conflict/audit_qb_rules_2026-09-11/VERDICT.md`
  (claims-auditor 반환문을 **스크립트로 원문 추출**, 304행 — 게이트 #115 전사 누락 회피).
- ★**개명 근거**: 원 Q-B의 프로브 C 교락 수치가 Little 항등식으로 소진된다
  (14.953 = 4.998 × 2.992, 23.713 = 5.194 × 4.565, 감사가 raw에서 독립 확인)이고, 핸드오프가
  요구한 공통 λ 설계는 그 **도착 채널을 끈다**. 남는 비자명 내용 = 평균 ITL 차의 합성 표준화
  분해 ⇒ **Q-B′**("공통 λ에서 decode 평균 ITL 차의 노출·인구 합성 표준화 분해, 경로 cf").
  "루프 이득·액추에이터·자기상쇄" 어휘는 인용 문장에서 퇴역(QB-8·QB-11·QB-15).
- **Stage 0 = "탐색적 계측 기록(사후 정의)"으로만 등재**. 39 pair-cell(Q-A 36 + P7 3),
  운영점 228/228 cudagraph-ON. 유일한 신규 측정량 = 토큰 노출 `E`. 위약 최대 0.039 ms,
  오분류 변이에서 위약 0.704 ms로 실패(= 작동하는 검사).
- **감사 조건 D1–D8·D10 이행 완료**(D9 = 커밋). 특히 **D1**(스스로 금지한 `Δ` 부호 진술을
  본문·코드가 산출 — N-1 위반) · **D3**(R² 0.997을 N1 방어로 사용 = 항등식, 게이트 #9의
  19번째 재발) · **D4**("모든 수는 JSON에서 옮겼다" 선언이 거짓).
- 계산기에 **금지 산출 기계 자기점검**(변이 9/9 검출)과 **문서 수치 결속 추적 검사**
  (미추적 0, 변이 3/3)가 추가됐다. 메인 세션 재실행 결과 공통 키 수치 차는 float epsilon
  (최대 4.4e-16)뿐.
- **Stage 1(≈2.9 GPU-h)은 실행 허가·저자+감사 모두 비권고 · 미실행.**
- 인용 금지 **+20건(QB-1…QB-20)**, 신규 게이트 **#163–166**(G-α·G-β+γ·G-ζ·G-η).

### C. R2 트랙 복귀 — Claim D 선결 (커밋 `02918e8`·`874873b`·`e16e93f`·`ea99191`, 정본 rev66 `379fd1d`)
사용자의 R2 복귀 결정으로 **2026-07-24 "사용자 결정으로 보류"가 해제**됐다.

1. **latch stale-True 수정** (`02918e8`): 쓰기 지점(`_r2_decide_idx`, split prefill 중에만)과
   읽기 지점(`update_split_prefill_batch`, split prefill 없을 때만)이 서로소여서 True가
   **흡수 상태**였다. 수정 = (a) latch 설정·prefill 없음이면 decode-only iteration에서도 정책
   재평가, (b) running batch 비면 admission 지점에서 해제. "배수 시 clear"는 제한 자체를
   지우므로 기각(테스트가 잡음). ★**영향 범위 = `generic`/`hybrid`(Claim E B5/B6)뿐** —
   `FixedPolicy`는 latch를 세우지 않으므로 **Claim D의 고정 split arm은 수정 전후 무영향**.
2. **job 907032 = `FAIL`** (0.12 GPU-h): TD boot 2개가 서버 자체 warmup 요청의 첫 split
   prefill에서 크래시(`AttributeError: 'NoneType' ... forward_mode`). ★**true-dual이 GPU에서
   처음 돈 run이고, correctness gate가 실제 버그를 잡았다.**
3. **경합 수정** (`874873b`): 메인 스레드가 워커에 batch를 넘긴 직후 같은 batch의
   `split_index`를 올려, 워커가 `split_index == 0`을 못 읽는 경합. 수정 = true-dual 경로에서
   그 갱신을 `prefill_future.result()` 뒤로. latch 수정과 무관함을 코드 경로 + 텔레메트리
   (TD `controller_decision` 0건) + 수정 전 트리(`25d170e0`) CPU 재현으로 입증. 신규 테스트
   fail-before/pass-after(실제 스레드 10/10 크래시 → 25/25 통과), 반쪽 수정 변이도 잡힘.
   **CPU 회귀 305/305**(메인 세션 독립 재실행 확인).
4. **하네스 v2** (`e16e93f`): S(순차 16) + **O(결정론적 중첩 탐침 8, 신규)**만 동치 게이트,
   **C(동시 32)는 진단 전용**으로 강등(직전 run에서 L-L 7/32 ⇒ 구조적 무판정). B8 =
   드라이버 green-context read-out으로 **split 실현** 확인, O1–O4 = 중첩·D44·두 워커 스레드
   관여. 판정 규칙 v2를 **TD 출력이 존재하기 전에** 헤더에 문자로 고정. checker 드라이런
   25/25, "항상 통과" 변이 2종 검출, 907032 재채점 FAIL 유지.
5. **job 907100 = `PASS`** (19:37:57–19:47:31 KST, 9m34s, **0.16 GPU-h**, gpu38,
   commit `38c1aca`): 4 boot 전부 booted·crash-free·cudagraph-ON(decode T/F 2617/2611/2580/2592,
   모두 0 false)·path·policy·overlap 전부 D44 idx4·green realized·O confirmed.
   **S 불일치 6쌍 전부 0 · O within/cross 전부 0**. C(진단) within 4/3, cross 4–7.
6. **결과 적대 감사 = `CONFIRMED(scoped)`**
   (`results/r2_correctness/audit_r2corr_2026-09-11/VERDICT.md`, 359행, 원문 추출):
   - **공허성 없음**: TD decode_host_tasks = decode 로그 줄 수(2611 = 2611), S/O 창 작업
     증가가 기대값과 정확히 일치(22·26·1008·1272), `gen_TD*`는 복사 아님(요청 시각·지연·
     C 불일치 7/32·sha 4개 전부 상이), 프롬프트 sha 56/56 동일, 퇴화 0/56,
     **교차-잡 귀무대조**(907032 legacy S 출력과 16/16 동일).
   - ★**스코프 축소 필수**: ① D44 위에서 돈 비교 대상 계산은 **O 탐침의 prefill뿐**(탐침
     decode는 비분할 idx 5, D44 decode 측은 비교 제외된 bg) ② cudagraph-ON은 **decode 한정**
     (prefill은 두 arm 모두 eager) ③ TP=1·greedy·ignore_eos·**ctx 4096**(러너 기본 16384와 다름)
     ④ "S16+O8"은 프로토콜로 적고 C 제외를 명시.
   - **무한정 "같은 토큰"은 C 층에서 REFUTED**: C01·C10·C19 **3/32**가 `{L1,L2} ≠ {TD1,TD2}`
     (arm-분리). 단 **"TD 결함"은 NOT-YET-SUPPORTED** — C01은 triton KV-split 휴리스틱 ×
     arm-특이 타이밍으로 산술 설명(C00이 혼자 decode한 step 수 L 6·6 vs TD 4·4).
     legacy끼리도 4/32 불일치 ⇒ 이 엔진(비 batch-invariant 커널)에서 동시 부하 토큰 동일성은
     **정의되지 않는 양**. r2_eval 성능 비교는 오염 안 됨(`ignore_eos`+고정 `max_new`).
   - **`unsafe_decisions` TD 30 / L 0 = 전부 no-op**: 60/60이 target = current = 44,
     non-D44를 잡은 순간 0건. 기전 = 앞 span prefill 이벤트가 in-flight일 때 결정
     (`pending_gpu_events=1`·`active_leases=0`). legacy는 그 술어를 **평가조차 안 함**(단락).
     ⚠️**커밋 `ea99191` 메시지의 "B8/O3가 확인했으므로" 근거는 부정확** → R2C-9으로 정정
     (직접 근거 = controller_decision 텔레메트리). 커밋은 고치지 않고 정본 追記.
   - **lease-release 위험 0/140 무발화**(상한 ≈2.1%/전환, 해소 아님), `active_leases`
     최댓값 0(TD 스냅샷 18,452개).
   - **Claim D 선결**: 닫힘 = **#5**(cudagraph-ON 호환, scoped). 부분 = **#2**(S/O 프로토콜
     한정). **미해소** = #1(latch GPU 발화 0건), #4b(observer effect paired on/off 미측정),
     동시 부하 동치, D16/24/34, generic/hybrid, 다른 모델, TP≥2, 러너 설정(ctx 16384 부팅
     실패·`r2_eval.sbatch:11` dirname), lease-release, `forward_ct` 비원자, thread-local role.
   - 인용 금지 **+16건(R2C-1…16)** + 필수 병기 P-1…7, 신규 게이트 **#167–169**(G-1·G-2·G-4)
     + 追記 #50(G-3)·#1·#114(G-5)·#95(G-6).

---

## 코드·문서 변경

**커밋 9건**(전부 `main` 로컬): `74351d3`(Q-A provenance 28) · `2cc7934`(게이트 #157 해시·경로
追記 + ROADMAP 헤딩) · `02918e8`(latch 수정) · `27b6ea7`(Q-B′ 사전등록 + 판정서) ·
`874873b`(true-dual 경합 수정 + 테스트) · `e16e93f`(하네스 v2 + 907032 FAIL 기록) ·
`38c1aca`(정본 rev65) · `ea99191`(907100 PASS 아티팩트 14) · `379fd1d`(정본 rev66).
**본 핸드오프 커밋**: 이 문서 + Claim E 코드 사실 4건 등재분.

**엔진**(`workspace/engine-port/src/multiplex/multiplexing_mixin.py`): latch liveness 2곳
(`:974`·`:1069`) + 헬퍼 2개(클래스 끝) / true-dual 소유권 경합 2곳(`:1222-1233`·`:1261-1266`).
기존 줄 번호 불변(line-citation 0위반). manifest mixin 항목 `25d170e0 → fdea4c32 → 0b88c07c`.

**테스트**(신규): `tests/test_r2_admission_latch.py` · `tests/test_true_dual_prefill_ownership.py` ·
`tests/pdmux_loop_fakes.py`(공용 fake). 전체 305/305.

**하네스**(신규 디렉터리 `results/r2_correctness/`): `r2_correctness.sbatch`(v2, 판정 규칙
헤더 내장) · `r2_correctness_client.py`(S/O/C) · `r2_correctness_check.py`(B1–B8·O1–O4·판정
우선순위) · `job_907032/`(FAIL 8파일) · `job_907100/`(PASS 14파일) ·
`audit_r2corr_2026-09-11/VERDICT.md`.

**longctx**: `PREREG_QB_LOOPGAIN_2026-09-11.md`(751행) · `probes/qb_forecast.py` ·
`probes/qb_doc_trace_check.py` · `probes/qb_forecast_result.json`/`_stdout.txt` ·
`audit_qb_rules_2026-09-11/VERDICT.md`.

**정본**: `PROJECT_STATUS.md`(2026-09-11(2)·(3) 절, 게이트 #163–169 + 追記, #158–161 백필) ·
`reports/CONSENSUS.md`(rev64→**rev66**, §3 항목182–189) · `reports/paper/{CLAIM_EVIDENCE_MATRIX,
EXPERIMENT_ROADMAP}.md` · 메모리 3파일(+ `MEMORY.md` 압축: 34.3 KB → 8.9 KB, 포인터 11개 유지).

**도구**(git 밖, 커밋 불가): 위 A의 6곳.

---

## 열린 항목 / 다음 세션 시작점

1. ★**R2 후속 GPU 실험 3건(각 ≈0.16 GPU-h) — 사용자 승인 대기**, 판정서 §7.3:
   - **X1**(권고 1순위): 네 boot 모두 `SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS=true`만
     추가해 재실행. 목적 = ① **측정 층 민감도 양성대조**(PASS가 "대형 손상 부재"만 인증하는지
     판별) ② C01 arm-분리가 KV-split 휴리스틱 기인인지 판별. **예보 3개를 실행 전에 문자로
     등록**해야 한다(판정서에 초안 있음).
   - **X2**: `R2C_DSM=16`(기본 아닌 split) — B8이 FixedPolicy를 **변별**하게 되고 B5/B6가
     방문하는 split까지 스코프 확장. 현재 D44에서는 네이티브 선택기도 idx 4를 고르므로
     변별력이 없다(R2C-16).
   - **X3**: 러너 설정 인증(ctx 16384 → Zamba2 부팅 실패, `r2_eval.sbatch:11` dirname 수리 후
     러너 인자 튜플로 1회). 선결 4b의 토큰 쪽 절반.
2. **하네스 결함 H1–H3(GPU 0, engine-porter)**: H1 `INPUT_IDENTITY`가 prompt **개수만** 비교
   (sha 미비교) · H2 `task_count`가 `set_result` 뒤 증가해 Δ에 +1 지연 · H3
   `host_worker_overlap_ratio`가 1024 절단 + 수명 분모라 동시성 지표로 부적합.
3. ★**Claim E 착수 전 해소할 코드↔로드맵 불일치 3건(사용자 결정 대기)** — 본 세션 등재:
   admission 제한이 1 iteration만 유지 · 평가 주기 `min` vs 로드맵 `max` ·
   hybrid profile 호환이 `engine_commit` 포함이라 B6가 상시 fallback.
4. **도구 파일 버전관리 이관(게이트 #157, 사용자 결정)** — `.claude/`가 git 밖. 부수로
   `git-committer.md:25-27`의 "바깥 워크스페이스는 별도 repo" 서술이 **사실과 다름**(정정 필요).
5. **Q-B′ Stage 1**(≈2.9 GPU-h): 실행 허가 상태이나 비권고. 논문이 Q-B′ 수치를 탐색 표지
   없이 쓰려 할 때만 최소 구매이며, 그 경우 핵심은 **모양 B**(저장소 0칸).
6. **Claim D 본 게이트는 여전히 멀다**: P1(observer effect paired on/off <3%) 미측정 →
   P2(legacy fixed vs true-dual fixed) 사전등록 없음. `results/r2_eval/` 미생성 유지.

---

## 미완·주의

- ★**인용 금지 총 116건**(Q-A 계보 80 + QB-1…20 + R2C-1…16). 특히
  **R2C-1**(스코프 없는 "true-dual ≡ legacy" 금지) · **R2C-2**("동시/서빙 부하 동치" 금지) ·
  **R2C-3**(C 불일치를 "TD 결함"도 "무해한 노이즈"도 금지) · **R2C-5**(토큰 동치는 D44 실현을
  확인하지 못함) · **R2C-11**(스레드 안전성 미검증) · **R2C-14**("Claim D 증거" 금지) ·
  **QB-1**(`G`를 1과 비교 금지) · **QB-6**(탐색 표지 필수) · **QB-8**(HE0 기전 설명 금지) ·
  **QB-15**(Q-B′는 원 Q-B의 답 아님).
- **job 907100의 동시성은 스모크 수준**: host 워커 중첩이 O 층 13.5/19.4 ms(하한), S 층 0.
  경합 결함률 95% 상한 ≈ 8.3%/span(동시 span 36). 커널 중첩(%smid/nsys) 미측정.
- **O 탐침 출력 8/8이 passage 4-gram 100% 일치**(복사형) ⇒ 동치 검사 민감도가 **보정되지 않음**
  (X1이 사려는 것).
- **실행 중 job 없음**(`squeue` 빈 상태, 2026-09-12 17:51 KST 확인). 워킹트리 clean.
- Q-A 캠페인 원자료 **10.8 GB**(`probes/qa_r{1..4}_9065*/`)가 여전히 디스크에 있다(ignore 대상,
  삭제 여부는 사용자 판단). job 907032/907100의 `srv_*.log`·`tel_*.jsonl`·`.triton_cache/`도 동일.
- **push 안 함**(정책).

---

## 다음 세션 `catch-up` 시작점 (한 줄)

> **R2 true-dual은 GPU correctness를 `CONFIRMED(scoped)`로 통과했다(job 907100 PASS, S16 순차 +
> O8 중첩 한정 · D44는 탐침 prefill만 · C 층 3/32 arm-분리는 원인 미확정) — 성능은 아무것도
> 재지 않았고 Claim D는 미검증 그대로다. Q-B′는 `GO-with-caveats`로 탐색 기록만 등재(Stage 1
> 비권고). 다음은 (i) X1 양성대조/X2 D16/X3 러너 인증 중 GPU 승인, (ii) Claim E 코드↔로드맵
> 불일치 3건 결정, (iii) 도구 파일 버전관리 이관 중 사용자 결정.**
