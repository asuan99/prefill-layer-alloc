# 세션 핸드오프 — 2026-09-12 18:00 KST ~ 2026-09-13 (장기 세션)

> 직전: [`session_handoff_2026-09-12.md`](session_handoff_2026-09-12.md).
> 커밋 **17건**(`4027257` … `ba5492f`) + 이 핸드오프 커밋. `CONSENSUS.md` **rev66 → rev72**.
> ★**GPU 지출 = 0.154 GPU-h 단 한 건**(X1 job 907456). R2 correctness 트랙 누적 **0.43 GPU-h**.
> 실행 중 job 없음. 워킹트리는 이 커밋 후 clean.

## 이번 세션 요약

사용자가 catch-up 뒤 "열린 항목 3건 → 권고 순서대로"로 계속 지시했고, 그 과정에서 **사용자 결정
5건**과 **claims-auditor 규칙층·결과 감사 6회**가 누적됐다. 실측은 X1 한 건(0.154 GPU-h)이고
나머지는 전부 **사전등록 단계에서 걸러졌다** — B3 `NO-GO`, λ0 `NO-GO`, X3는 `GO-with-caveats`를
받았으나 **모델 교체 결정으로 무효화**, OS는 `GO-with-caveats`이나 Zamba2 기준이라 재스코프
필요. ★**새 성능 판정 0건 · arm 순위 0건 · 정책 순위 변경 0건 · HE0 불변 · stake #1 불변 ·
Claim D/E 등급 불변(둘 다 미검증)**. 감사자는 이 세션에서 **자기 처방 4건을 철회**했다.

세션의 실질 성과는 두 가지다. (1) **Claim E의 코드 선결이 전부 해소**됐고 러너가 처음으로
실행 가능해졌다(spool-copy 경로 결함·ctx 유도·provenance). (2) **캠페인이 구조적으로 못 도는
이유들이 드러났다** — 모델 ctx, oracle/profile 산출물 부재, 그리고 **λ\* 정의 자체의 모순**.

---

## 결정 (사용자)

1. **Claim E 코드↔로드맵 불일치 3건 = 전부 "로드맵이 정본"** → 커밋 `3f188bb`.
2. **그 결정의 부작용(`:976` 즉시성) 해결 = A안** (긴급 조건을 즉시 트리거로 승격 + overload
   streak epoch 게이트 + `:979` 분리 disjunct + `bucket_changed` 미구현 명시) → 커밋 `8753402`.
3. **도구 파일 버전관리 이관 진행** → 커밋 `4027257`.
4. **모델 교체**: Zamba2-2.7B → **`nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base`**(ctx 131072).
   근거: W2/W4/W5가 **8192 토큰 입력**을 내보내고 Zamba2 전 계열이 ctx 4096.
5. **백엔드 flashinfer 전환**(NemotronH에서 triton은 엔진이 `AssertionError`로 거부) +
   **1차 캠페인 범위 = Claim D + P3**, 실행 순서 (0)λ\* → (1)correctness → (2)P1 → (3)P2 → (4)P3 → (5)Claim E.

## 측정 (GPU)

- **X1 (job 907456, gpu42, 2026-09-12 19:11–19:20 KST, 9m14s = 0.154 GPU-h)** — 민감도 양성대조.
  `--triton-attention-num-kv-splits 2`, 엔진 소스 `38c1aca` 핀(manifest 17/17 907100 동일).
  **`VERDICT PASS`** + 결과 감사 **`CONFIRMED(scoped)`**
  (`results/r2_correctness/audit_x1_2026-09-12/VERDICT.md`, 357행).
  교차-잡 진단에서 **2단위(S05@25·O06@11)가 4 boot 전부 동일하게 뒤집힘**(96 unit-pair 중 8).
  ★**스코프 축소 필수**: 뒤집힌 두 decode는 **둘 다 비분할 `stream_index 5`**에서 돌았고,
  **D44에서 돈 교란된 decode(O층 bg 행)는 0/32** ⇒ **X1은 D44 운영점의 측정층 민감도를
  보정하지 않는다**. "24/24 교란"은 단위 지시함수였고 스텝 커버리지는 **82.5%**(최악 5/63).
- 그 외 GPU 지출 **0**. B3·OS·X3·λ0 전부 미실행.

## 사전등록·감사 6회 (전부 GPU 0에서 걸러짐)

| 대상 | 판정 | 핵심 |
|---|---|---|
| X1 규칙층 | `GO-with-caveats` | 판정서 초안(env var만)이면 **O 탐침 8/8 무교란 = 항등식**이었다 → cap 2로 교체 |
| X1 결과 | `CONFIRMED(scoped)` | 위 스코프 축소. 필수 병기 X1P-1′…9 / 인용 금지 X1C-10…14 |
| B3(C층 귀무대조) | **`NO-GO`** | 死因 N2 반전 3건(실제 job 907456의 n_22=0에서 라벨 뒤집힘 · 출력공간 31–59% 무라벨 · 빈 `phase_c`가 `E4=0` 인쇄) |
| OS(순서 교환) | `GO-with-caveats` | arm/position 교락을 한 job에서 분리. **Zamba2 기준 → 재스코프 필요** |
| X3(러너 설정) | `GO-with-caveats` → **무효** | 러너 코드를 한 줄도 실행하지 않음이 드러남 + 모델 교체로 무효화 |
| λ0(용량) | **`NO-GO`** | 死因 N2(미등록 client seed·사다리) + N3(예보 정의역 공집합) |

★**감사자 자기 철회 4건**: "B3 2 job으로 n≥4 충족"(독립 런 아님) · X1 §3.3 cross-arm 합
**22 → 25** · OS §7 "`L TD TD L`이 L-first를 제거한다"(L1은 여전히 위치 1) · OS §8
"**X3가 성능 트랙 전체를 연다**"(블로커 3개 중 **0개** 제거).

---

## 코드·문서 변경 (전부 커밋됨)

**엔진/정책** — `src/multiplex/`
- `controller.py`: admission 제한이 HOLD를 승계(1 iteration → 지속), `evaluation_due`를
  로드맵 `max(4 iter, 100 ms)` 의미로, **긴급 조건(ITL 위반·점유 ≥85%)을 즉시 트리거로 승격**,
  overload streak을 **epoch당 1회**로 게이트(없으면 2 iteration ≈27–85 ms만에 admission 제한),
  `:979`를 `(target≥D108 ∧ upper_bound>SLO) OR kv≥0.90 OR running≥0.90`으로 분리,
  `SplitDecision.off_cadence` 신설.
- `profile.py`: 호환 축에서 `engine_commit` 제외 → **임포트된 엔진 모듈 8개의 내용 해시**
  (`engine_source_hash`), 하위호환 **fail-closed**.
- `dual_worker.py`: 하네스 결함 **H2–H5** 수리(판정 규칙 `SNAP_KEYS` 무영향을 테스트로 고정).

**벤치/러너** — `scripts/r2_eval/`, `benchmarks/pdmux_eval/`
- `r2_eval.sbatch:11`의 `dirname "${BASH_SOURCE[0]}"`는 **sbatch가 스크립트를 spool로 복사**해
  `engine_root`를 scratch로 잡던 결함 → 고정 절대경로. (a)sbatch (b)로그인 (c)array 3경로 테스트.
- `PDMUX_CONTEXT_LENGTH`는 **읽는 곳 1개·쓰는 주체 0개**였다 → 모델 `config.json`에서 유도.
  신규 `context_limit.py`가 `input+output > ctx−2`(엔진이 `max_new_tokens`를 **조용히 줄여**
  적은 decode 작업량으로 채점되는 경우)까지 거부.
- `campaign.json`에 `model`·`context_length` 추가(`schema v2`), `cuda_graph`를 **읽히게** 배선.
- **attention-backend 노브**(기본값 `triton` 불변 — Zamba2 재현 보존).

**provenance/규율**
- `sync_engine_tree.sh`: 모델 구현을 manifest 안으로 **17 → 24항목**
  (`models/{nemotron_h,falcon_h1,granitemoehybrid}.py` 설치 + `configs/*` 해시 전용).
  ★실증: job 905835의 manifest는 17줄이고 `models/` 항목이 `mamba2`·`zamba2`뿐인데
  **그 job은 NemotronH를 12 boot 서빙했다**(모델 provenance 0) → 게이트 **#195**.
  ★**읽기 규칙**: 이후 교차-잡 서술은 "기존 17 일치 **+** 신규 7" 로 쓰고 "17/17 동일" 금지.
- `check_line_citations.py`: `controller.py`·`profile.py` 앵커 **38항목 등록**(50 → **88 compared**),
  도구 결함 2건 수리(콤마 목록 미파싱으로 실제 이동한 앵커 2개가 **구조적으로 안 보였다**).
- **`tools/claude/`** 신설(게이트 #157 종결): 툴 파일 11개 추적 사본 + `sync_claude_tools.sh`
  (`--check`/`--import`/`--install`) + **변이 테스트 10케이스**. `git-committer.md`의
  "outer는 별도 repo" 서술을 사실대로 정정(outer는 git 저장소가 아니다).

**테스트**: **305 → 463 OK**(신규 158). 변이 트리로 각 수리의 fail-before/pass-after 확인.

**정본**: `CONSENSUS.md` rev66 → **rev72**(§3 항목 187–215), 신규 방법론 게이트 **#170–195**,
`PROJECT_STATUS.md`·CEM·ROADMAP·DOCUMENT_STATUS·`RESUME.md`, 메모리 3파일(교훈 165–193).

---

## 열린 항목 / 다음 세션 시작점

### ★1. 다음 GPU 구매 (감사 권고 순서)
**(1) 새 모델 correctness 게이트 + `--request-rate inf` 2셀 ≈ 0.3–0.4 GPU-h를 먼저 산다.**
근거: `bench_serving.py:943-945`가 `rate == inf`에서 sleep을 전부 건너뛰므로 **Poisson 실현이
존재하지 않고 λ0의 死因 N2가 구조적으로 소멸**한다. 이것이 λ\* 사다리 양 끝을 **추측 없이**
고정한다. ★감사자 자기 한계 승계: closed-loop 포화율 ≠ open-loop일 수 있으므로(905835 d44는
`Concurrency 24.08` 중 **22.75가 대기**) **λ\*의 대체가 아니라 상한 프로브로만** 등록하고
confound #7(클라이언트 병목)을 `Concurrency` 보고값으로 확인해야 한다.
- 하네스는 Zamba2/triton 형태라 **flashinfer·NemotronH·ctx 16384로 재설계**가 필요하다
  (`R2C_MODEL`·`R2C_CTX`·`R2C_ATTN_BACKEND` 노브는 이미 있다).
- correctness 게이트에 **추가 boot 0으로 붙일 계측 3건**(감사 §8): ctx 16384/mem 0.82에서
  `max_mamba_cache_size`·`max_total_num_tokens` 배너 확인 · (256,512) 동시성 1 프로브로
  TTFT 바닥·ITL(B=1) 실측 · 위 `inf` 2셀.

### ★2. λ0 rev2 (D1–D15 반영 후 재감사)
권고 사다리 **shape A {1.1, 1.8, 3.0, 4.9, 8.0}** · **shape B {0.45, 0.62, 0.85, 1.15}**.
λ\*(A) 사전 추정 **2.1–2.5 req/s**(엔진 decode step 회귀 `19.82 + 0.5174·B ms`, 구속 =
`max_mamba_cache_size 48`, KV **6.6× 과공급**), 절대 경계 [1.08, 7.15]. sbatch·per-cell
analyzer를 **출하**해야 한다(현재 부존재). `UNRESOLVED` 도달 가능성을 주입 실험으로 고정.

### ★3. 사용자 결정 대기 — **P2가 W4를 쓸 수 없다**
단일 스칼라 λ\*로는 **W4의 두 phase를 동시에 0.80으로 만들 수 없다**(λ\*=0.675 → prefill
0.79× ✓ / decode 0.21–0.25× ✗; λ\*=2.1 → decode 0.79× ✓ / prefill 2.47× ✗; 기본값 4 → prefill
4.7×). 그런데 **로드맵 P2 acceptance는 "decode-heavy/alternating에서 개선"을 요구**하고
alternating = W4다 ⇒ **워크로드 정의(phase별 독립 λ\*)를 고치거나 acceptance를 바꿔야 한다.**

### 4. 사용자 결정 대기 — 캠페인 나머지 블로커
- **B2/B8(90 run)** = offline oracle 산출물 부재 · **B6(45 run)** = hybrid profile JSON이
  저장소 전역 **0건**(= P3의 산출물). B6가 Claim E의 주 arm이다.
- ★**수치 재계산 필요(미등재)**: "고유 차단 225 / 가능 180"은 **Zamba2 ctx 4096 기준**이었다.
  모델 교체로 ctx 블로커(135 run)가 사라졌으므로 추정 **차단 135 / 가능 270**으로 바뀐다 —
  다음 세션에서 재계산해 등재할 것.

### 5. 미해결로 등재된 것
- **OS 사전등록**의 Zamba2 한정성(초점 단위 C01·C10·C19가 907100 관측) → 재스코프 여부.
- 사다리 상승 속도: dwell 면제 + 즉시 재평가로 `D16→…→D108`이 **4 iteration(≈40 ms)**
  (이전 ≥400 ms). 사양 부합이나 "epoch당 한 칸 제한" 여부 **미결정** — Claim E 실행 시
  `off_cadence`·`split_transition`으로 **측정한 뒤** 결정 권고.
- 로드맵 `:979` 접속사·`bucket_changed` 미구현 · 하네스 결함 H1(rule v3 사전등록 필요).
- `r2_eval.sbatch`의 로그 경로·`module load`·`HF_HOME`·`TRITON_CACHE_DIR` 누락,
  `trace_loadgen`이 예외를 삼켜 **부분 실패 arm도 채점 가능한 요약**을 냄, 포트 충돌 가능.
- 상류 위험 3건: `mamba2_cache_params`가 선언되지 않은 `self.n_groups`에 의존 ·
  `config.expand`가 Nano-9B에서 stale · `NemotronHForCausalLM`이
  `piecewise_cuda_graph_disabled_model_archs`에 없어 **cps>0 arm에서 크래시 재개** 가능.

---

## 미완·주의

- ★★**모델·백엔드 교체의 귀결**: 907100·907456·X1의 모든 결론은 **(Zamba2-2.7B, triton) 한정
  동결**이다. **X1의 민감도 시연은 `triton_attention_num_kv_splits`가 flashinfer에 없어
  이식되지 않는다.** 새 (모델, 백엔드) 쌍에서는 **PD-mux true-dual 경로가 한 번도 실행된 적이
  없다** — job 905835는 **legacy 루프 + fixed split만** 발화시켰다(인용 금지 Q2).
- ★**λ\*는 측정되지 않았다**. 905835의 D16 0.933 / D44 0.675 / D92 0.187은
  **8192-in/96-out·ctx 8192·mem 0.80** 조건의 기존 측정이며 새 캠페인의 λ\*가 아니다.
- ★**λ\*_throughput ≠ λ\*_SLO**: 905835 d44를 정본 goodput 술어로 재채점하면 **0.59·λ\*에서
  goodput 53.8% · 0.89·λ\*에서 5.8% · 1.27·λ\*에서 0.8%** ⇒ **λ\*_SLO < 0.59 × λ\*_thr**이고
  **캠페인 0단계는 게이트 #6("용량 먼저 측정")을 닫지 못한다**(인용 금지 Q1).
- 인용 금지 목록이 크게 늘었다: R2C-1…16 · X1P-1′…9/X1C-1…14 · B3C-1…5 · OSP-1…6/OSC-1…8 ·
  X3P-1…7/X3C-1…9 · λ0 Q1–Q5. 결과를 인용하기 전에 해당 판정서 §필수 병기를 먼저 읽을 것.
- **비포화 셀의 `achieved/offered`는 포화도가 아니라 도착 실현 계수 `1/Ē`**다(905835에서 12/12
  셀 −15~−18%, 1.03–1.19). 이 양을 문턱 규칙에 쓰려면 N ≥ 1537이 필요하다.
- `push` 안 함(정책). 루트 SLURM 로그 커밋 안 함.

## 다음 세션 `catch-up` 시작점 (한 줄)

> **모델·백엔드가 (Nemotron-Nano-9B-v2, flashinfer)로 교체됐고 Claim E의 코드 선결과 러너
> 결함은 전부 해소됐다 — 그러나 새 쌍에서는 PD-mux true-dual이 한 번도 돌지 않았고, λ\*는
> 미측정이며(사전등록 `NO-GO`), P2는 단일 스칼라 λ\*로 W4를 파라미터화할 수 없다. 다음 수는
> (1) 새 모델 correctness 게이트 + `--request-rate inf` 2셀(≈0.3–0.4 GPU-h, N2 死因을 구조적으로
> 제거) → (2) λ0 rev2 재등록 → (3) W4 워크로드 정의에 대한 사용자 결정. GPU 누적은 이 트랙
> 0.43 GPU-h뿐이다.**
