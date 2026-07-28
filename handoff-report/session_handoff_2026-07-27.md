# Session handoff — 2026-07-27

## 이번 세션 요약

catch-up 후 사용자 승인 순서 **(1) sync-race flock 수정 → (2) de-confound 재측정**을 실행했다.
(1)은 A/B 재현까지 마쳐 완료. (2)는 "prefill-SM 고정이 config만으로 되는가"를 **GPU probe로
먼저 게이트**(job 864669, `decode_sm` 존중 확인)한 뒤 사전등록 설계를 세우고 스윕을 돌렸다.
1차(865006)는 **context off-by-one으로 전량 무효**(내 클라이언트가 이를 조용히 삼킨 것이 더 큰
결함) → 수정 후 2차(865098) 성공. 결과는 **사전등록 H0(평탄) 반증**이나, decode batch가 D와
공변해 **H1도 채택 불가**. matched-batch 층화 후에도 기울기(≈2–2.7×)가 남고, 그것이 **음성대조
M에서도** 남는다는 것이 핵심 쟁점이다. 사전등록 규칙대로 **정본 무수정**. 이후 사용자 지시로
git 정리(untracked 130→0)와 Stage 0 아티팩트 갱신(de-confound act 추가)을 수행했다.
전부 로컬 커밋 6개.

## 결정·측정

### (1) sync-race 하네스 수정 — 완료 (`f99fd84`)
- `sync_engine_tree.sh` 전체를 **cross-node flock**(자기 re-exec)으로 직렬화 + manifest
  temp→rename. `/scratch`는 Lustre `flock`(≠`localflock`) 마운트라 노드 간 coherent 확인.
- **A/B 재현(throwaway tree, 10동시 × 3 trial)**: 락 우회 = **7–8/10 실패**(864230_0과 동일
  시그니처 `install: cannot create regular file ...: File exists`), 락 적용 = **0/10 실패,
  manifest 10/10 동일**.
- ★**정정 2건**: 실기전은 parent-dir mkdir이 아니라 **`install`의 unlink→create**(패자가 EEXIST).
  그리고 내 초판 수정에 버그 — `if ! cmd` 안의 `status=$?`는 부정 결과(항상 0)를 읽어
  **lock timeout 시 exit 0**(= 미동기화 트리로 job 진행). `PIPESTATUS`식 캡처로 수정, nonzero 확인.
- 환경 gotcha: 로그인 프로파일에 **`bash()` 함수**가 있어 `bash <script>`가 인자를 삼키고
  no-op(첫 A/B가 통째로 헛돌았음) → 검증은 `/bin/bash` 사용. **export 안 됨 ⇒ SLURM job은 무영향**.

### (2) de-confound — 사전 게이트 (job 864669, `results/s0_deconfound/`)
- 문제: 지금까지 쓰인 **모든 division이 정확히 108로 합**해서, `decode_sm`이 존중된 것인지
  `108−prefill` 나머지였을 뿐인지 기존 데이터로 **구분 불가**. 후자면 `[16,44]`가 decode에
  92 SM을 주게 되어 설계 자체가 붕괴.
- `spatial_ops.abi3.so` 심볼로 2단 split 구조 확인 후, matmul 처리량으로 실효 SM 역산:
  **B(16,44) eff 55.7 ≈ B(64,44) 55.6 (0.2%)**, vs **B(16,92) 110.7** ⇒ **DECODE_SM_HONORED**,
  합<108의 잔여 SM은 idle ⇒ **config-only 성립**.
- 부수: eff는 **보정된 SM 수 아님**(B req 92 → 110.7 > 물리 108). 작은 파티션이 SM당 20–30%
  더 받음 ⇒ **§clock confound 실재 확인**, 저-D arm을 유리하게 만들어 **민감도를 과소평가**하는
  방향. 또 B = (a+b) − A_actual이라 A 상향라운딩 시 B가 줄지만(예 `(92,16)` B eff 17.7 vs
  타 16-요청 ~21.0), 본 설계는 전 arm a=16(라운딩 없음)이라 일관.

### (2) de-confound — 1차 865006 = **전량 무효**
- `CTXCAP=CTX+512`(Stage 0 값)를 유지한 채 `OUTTOK`만 32→512로 올려 예산이 경계에 정확히 앉음.
  프롬프트는 실제 **4097 토큰**(Zamba2는 4098) ⇒ 전 decode 요청이 `4609 > 4608`로 거부.
- ★**더 심각한 결함 = 침묵**: 거부는 **HTTP 200 + 에러 payload 1줄**로 오므로 `urlopen`이
  예외를 안 냈고, 클라이언트가 **`errors:0, requests_completed:10037`**(= 토큰 0개 생성한 런의
  건강해 보이는 요약)을 냈다. PIN_CHECK는 `hist: NONE`으로 FAIL이었지만 이는 **decode 트래픽이
  아예 없던 결과**이지 pinning 문제가 아니었다.
- 수정(`fd6f630`): `CTXCAP=CTX+OUTTOK+256`(실제 tokenizer로 slack 254 확인) · 2청크 미만이면
  서버 메시지를 인용해 raise · 대부분 실패 시 nonzero exit · sbatch는 `PIPESTATUS`로 클라이언트
  상태 확인 후 **첫 측정 실패 시 즉시 abort**(기존엔 tee가 상태를 삼켜 arm당 25분 낭비) ·
  포트는 OS에서 free port 취득(M arm D44가 `Errno 98`로 유실됐던 건).
  가짜 서버로 회귀 가드 검증: exit 4, errors=116, 서버 메시지 노출.

### (2) de-confound — 2차 865098 = **유효 측정, 그러나 미결**
전 cell PIN_CHECK PASS, errors 0, n=4. ITL p50 (ms), prefill 16 SM 고정:

| arm | D16 | D24 | D44 | D92 | D16/D92 |
|---|---|---|---|---|---|
| M (음성대조) | 22.96 ±0.02 | 13.23 | 9.13 | 7.54 ±0.01 | **3.05×** |
| H (타깃) | 113.55 ±10.56 | 70.19 | 26.87 | 13.22 ±0.50 | **8.59×** |
| T (양성대조) | 17.71 ±0.01 | 12.92 | 9.62 | 8.55 ±0.00 | **2.07×** |

- **사전등록 H0(평탄, <5% spread) 반증.** 그러나 **H1 채택 불가**: decode batch가 D와 공변
  (엔진 telemetry `decode_running_batch_size` H **8.44→3.28**, M 5.45→3.18, T 8.34→3.82).
  closed-loop이 고정한 건 **in-flight 요청 수**였지 **decode batch**가 아니었다 — prefill이
  16 SM + keepalive 2×4096-token으로 병목이 되어, decode가 빠를수록 decode 구간이 빨리 비었다.
  ★**계측해 둔 realized occupancy가 이걸 잡았다**(주장만 했으면 통과했을 confound).
- **matched-batch 층화**(telemetry `measured_itl_ewma_ms`를 batch로 층화, bin당 n≥30):
  기울기 **살아남음** — M b1/b2/b4 = 2.39/2.32/2.37×, H b1 = 2.67×, T b5–b8 ≈ 1.89–2.06×.
  단 일부 bin 역전(M b3 0.95×, T b4 0.87×) + EWMA-batch 페어링은 근사 ⇒ **suggestive, 미확정**.
- ★★**핵심 쟁점(미해결)**: matched-batch에서도 **음성대조 M이 기울었다**. 배타적 두 해석 —
  **(a)** 잔여 confound(=위 결과 전부 무효) vs **(b)** **음성대조 전제가 틀렸다**: "pure Mamba2
  decode는 O(1)이라 SM-bound 불가"에서 **O(1)은 context-length에 대한 것이지 SM 수에 대한 것이
  아니다**(decode step도 d_model×d_state matmul 수행). (b)면 M은 이 축의 음성대조가 아니고
  기울기는 실재하며, **Stage 0의 D16≡D108 null이 아티팩트**가 된다.
- **정본 무수정** — DESIGN.md §5가 "H1이면 claims-auditor 선행, 정본 직접 수정 금지"를 사전등록.
  상세: [`results/s0_deconfound/FINDINGS_865098.md`](../workspace/engine-port/results/s0_deconfound/FINDINGS_865098.md).

### (3) git 정리 / (4) 아티팩트
- untracked **130 → 0**. 규율 판단은 **레포 자체 선례**를 따름: 최근 캠페인은 harness+verdict+
  manifest만 버전관리(g2_0_raconf/decliff 각 3파일), 483파일 `slo_sched`는 구식. 동일 클래스
  ignore + **campaign별 `runtime_source_manifest.sha256`으로 provenance 유지**(sbatch가 이제
  캠페인 디렉터리에 기록). 대용량 telemetry/log는 `workspace/engine-port/.gitignore`가 이미 제외(1.7GB).
- 아티팩트 갱신(같은 URL, `bf7f64f`): Act 1(Stage 0) + **Act 2(de-confound)** 2막 구성.
  헤드라인을 "clean signal = SM-insensitive"에서 **"under review"**로 변경. 신규 패널 3개
  (log-y ITL(D) · batch 공변 · raw vs matched-batch ratio + 1.0 기준선). 팔레트는 기존 유지하되
  양 테마 검증(normal ΔE≥19.1, worst CVD 11.3, contrast≥4.14; node 부재로 Python 구현).
  전 수치를 원자료와 대조 확인.

## 코드·문서 변경 (전부 커밋됨, push 안 함 — §미완 참조)

- `f99fd84` — `scripts/bootstrap/sync_engine_tree.sh` flock + atomic manifest.
- `8f98044` — `results/s0_deconfound/`: `DESIGN.md`(사전등록) · `pdmux_p16_d{16,24,44,92}.yml` ·
  `greenctx_alloc_probe.{py,sbatch}`.
- `e9f49de` — probe verdict를 DESIGN.md §4에 기록 + probe json.
- `a17cd8a` — `s0dc_client.py`(closed-loop 고정 occupancy + warmup/steady window + realized
  occupancy 보고).
- `f523f70` — `s0dc_sweep.sbatch`(M/H/T × P16D{16,24,44,92} × n=4, clock 샘플링, PIN_CHECK 실격).
- `fd6f630` — 865006 실패 수정(CTXCAP·침묵 실패·PIPESTATUS fail-fast·free port).
- `75db42d` — `FINDINGS_865098.md`(결과 + 미결 쟁점).
- `163cb76` — `.gitignore` 캠페인 산출물 + per-campaign manifest.
- `bf7f64f` — `reports/stage0_sm_scaling.html` 2막 갱신.

## 열린 항목 / 다음 세션 시작점

1. ★**Stage 0 파티션 활성시간 재집계 (최우선, GPU 불요)** — §쟁점 (a)/(b)를 가르는 가장 싼 수.
   sub-108 decode 파티션은 **prefill이 in-flight일 때만 활성**인데 Stage 0의 PIN_CHECK는 policy
   *target*만 봤다. 기존 `results/stage0_xctrl/*_telemetry.jsonl`에서 "decode step 중 파티션이
   실제 활성이던 비율"을 계산 → 낮으면 Stage 0 D16이 상당 시간 full-108에서 돌아 D108과 같아
   보인 것이고, Stage 0 null은 아티팩트.
2. **claims-auditor 반증** — FINDINGS_865098 §4 (a) vs (b). DESIGN.md가 사전등록한 필수 관문.
   (에이전트 호출은 사용자 트리거 필요.)
3. **batch를 설계상 통제한 재측정** — keepalive 프롬프트를 짧게(파티션만 활성화, prefill 병목
   제거) + out_tokens 확대 → arm 간 realized occupancy 일치를 **게이트로** 검사.
   클라이언트는 이미 `--keepalive-prompt-file` 지원.
4. **정본 정확성 수정 제안(doc-steward 필요, 미실행)** — `PROJECT_STATUS.md`(Stage 0 절)와
   `CONSENSUS.md` §1-21의 "**D16 vs D108(prefill 경합 0인 두 점)**" 표현은 하네스와 불일치
   (`stage0_pdmux_capture.sbatch:106-108`: D16은 keepalive 2, D108은 0). **상한 논증으로는
   유효**하나 "둘 다 무경합"은 부정확. 정확한 표는 `s0_deconfound/DESIGN.md` §1.1에 있음.
   ★이건 내 새 결과와 **무관한 독립적 문서 정확성 문제**라 (2)의 판정을 기다릴 필요 없음.
5. 기존 항목 유지: >16k·큰 모델 미측정 · TC-series(Transformer-control) · §1-20 spatial decoupling ·
   admission latch stale-True 버그(미수정, 보류).

## 미완·주의

- ★**push 상태 이상**: `origin/main`이 내 최신 커밋(163cb76)과 동일하고 reflog에
  **`update by push`** 기록(163cb76 커밋 16:01:09 → push 16:01:20). 나는 push하지 않았고
  `.git/hooks` 비어 있음 · `core.hooksPath` unset. **2026-07-25/26에도 push 기록**이 있어
  **이번 세션이 만든 현상 아님**. 레포 규율은 PAT 평문 노출 때문에 push 금지이므로 **원인 확인 권장**.
- **865098은 확정 결론이 아니다.** H0 반증만 견고하고, 기울기의 실재 여부는 (a)/(b) 미해결.
  claims-auditor 미경유. **정본·논문에 인용 금지** 상태.
- matched-batch는 **사후 층화**이지 설계상 통제가 아니며, `decode_last_tpot_ms`가 telemetry에서
  항상 0이라 **EWMA(창 평균)로 대체**한 근사다.
- H의 D16만 SD 10.56ms(타 셀 ≤1) — 포화 근처 regime 가능성, 미조사.
- clock: arm마다 활성 SM 총량 상이(32 vs 108). 방향은 **민감도 과소평가** 쪽이라 관측 기울기는
  하한 성격이나 정량 미보정.
- 아티팩트 favicon을 🔬로 지정 — **원본 세션의 선택이 기록에 없어 복원 불가**. 알면 교체 필요.
- 865006 산출물은 디스크에 남아 있으나 무효(=인용 금지). 실행 중 job 없음.
