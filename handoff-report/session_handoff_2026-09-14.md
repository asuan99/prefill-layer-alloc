# 세션 핸드오프 — 2026-09-13 저녁 ~ 2026-09-14 (장기 세션)

> 직전: [`session_handoff_2026-09-13.md`](session_handoff_2026-09-13.md).
> 커밋 **3건 + 이 핸드오프 커밋**(`b2b60e7` · `a9cd8dd` · `83d8cb9`). `CONSENSUS.md` **rev73 → rev75**.
> ★**GPU 지출 3 job = 1.107500 GPU-h**(sacct 기준, 등록 0.955000 + **등록 밖 0.152500**).
> 실행 중 job 없음. 워킹트리는 이 커밋 후 clean.

## 이번 세션 요약

사용자가 catch-up 뒤 "권고들에 대해 동시 진행 가능성 검토 후 진행"으로 지시했고, 3개 워크스트림을
병렬로 돌린 뒤 그중 하나(R2 correctness)가 **GPU 3 job과 실물 엔진 결함 1건**을 산출했다.

**이 세션의 실질 성과 셋:**
1. ★**true-dual worker가 `@torch.inference_mode()` 밖에서 돌아 autograd가 켜진 채 모델 forward를
   실행하던 결함을 발견·수리했다.** job 907959의 OOM(`FAIL`)에서 출발해 코드·CPU 측정으로 기전을
   규명하고(H1), `_activate_role_context`에 grad guard를 넣은 뒤 재실행(job 908179)에서 **TD 2/2가
   자기를 죽였던 바로 그 배치를 완주**했다. 단 **"수리가 OOM을 고쳤다"는 `PLAUSIBLE(조건부)`**이고
   `CONFIRMED`가 아니다(§열린 항목 1).
2. ★**등록 밖 실행 사고(job 908020)와 그 구조적 수리.** 제출 명령이 세 env를 빠뜨려 Zamba2/triton/
   ctx4096로 돌았고 **0.152500 GPU-h가 어떤 게이트도 진전시키지 못했다.** 하네스에 fail-closed
   **scope guard**를 넣었고 job 908179에서 **C1 스코프 술어 (1)–(11) 전부 참**으로 작동을 확인했다.
3. **λ0(캠페인 0단계)는 rev1→rev4 4연속 `NO-GO`, GPU 지출 0.** 매 회차가 사전등록 단계에서 걸러졌다.

★**새 성능 판정 0건 · Claim D/E 등급 불변(둘 다 미검증) · HE0 · 정책 순위 · stake #1 ·
게이트 #13/#16 · Zamba2/triton 동결 — 전부 불변.**

---

## 결정 (사용자, 4건)

1. **W4 워크로드 정의 = (a) phase별 독립 λ\***(`EXPERIMENT_ROADMAP.md` P2 열린 항목 **종결**).
2. **옵션 A 수리 + arm 대칭 메모리 계측 → 게이트 재실행**(H1 대응).
3. **rev4 → 재감사 → 실행** — RA3-1의 검정력 공시(이 회차는 어느 결과가 나와도 H1을 닫지 못한다)를
   받아들이고 **게이트 라벨**을 산출물로 진행.
4. (직전 세션 승계) 모델·백엔드 = Nano-9B-v2-Base / flashinfer / ctx 16384.

---

## 측정 (GPU 3 job, sacct 기준)

| job | 노드 | 시간 | GPU-h | 라벨 | 비고 |
|---|---|---|---|---|---|
| **907959** | gpu38 | 21:21 | 0.355833 | **`FAIL`** | true-dual 2/2가 14-seq/10,125-token split-prefill에서 CUDA OOM. S층·O층 동등성은 **불일치 0** |
| **908020** | gpu43 | 9:09 | **0.152500** | `PASS` | ★**등록 밖 튜플**(Zamba2/triton/ctx4096). **어떤 게이트도 진전 없음** |
| **908179** | gpu40 | 9:56 | 0.165556 | **`PASS`** | ★**등록 튜플 최초 실행**. C1 (1)–(11) 전부 참 |

```
R2 correctness 트랙 누적  1.107500 GPU-h  (sacct 기준으로 통일, 선행 드리프트 정정 등재)
  ├ 등록 회차                0.955000
  └ ★등록 밖 (908020)        0.152500  (16.3 %, 어떤 게이트도 진전시키지 않았다)
longctx_conflict 트랙 15.42 GPU-h — 별개 장부, 불변
```

### 907959 — true-dual OOM
- TD boot 2/2가 **legacy가 완주한 바로 그 14-seq/10,125-token 배치**에서 `torch.OutOfMemoryError`
  (198.00 MiB 요청, 77.25 GiB allocated), **동일·결정적**(같은 상태·같은 크기·같은 스택).
- 결과 감사 `results/r2_correctness/audit_907959_2026-09-13/VERDICT.md`:
  귀속 **`CONFIRMED(scoped)`**, 기전 **`NOT-YET-SUPPORTED`**, 필수 병기 **N-1…N-13**.
- 부수 측정: **λ_inf(A) = 3.0939 req/s**. ★**λ_inf(B) = 0.6956은 인용 불가**(**N-8** — D4 셀별
  재계산에서 I3b의 `#running-req` 최댓값이 11이라 F5가 셀 B에서 반증). ★**N-7**: I2(동시성 1)의
  TTFT 43.0 ms·ITL 12.96 ms는 **D44가 아니라 비분할 108 SM** 값이다.

### 908179 — 등록 튜플, `PASS`
- **F-a1 참(4채널)** · **F-b1 대역 A′**(`S = 0.003211` vs θ=0.3825 MiB/token) ·
  **F-b2 대역 A**(`Δ = −48.37 MiB`, ε=1.00 GiB) · F-c·F-d 충족 · **E5 미발화**(원시 `RuntimeError` 0건).
- ★**메모리**: `peak(TD1) = peak(TD2) = peak(L2) = 73,342,324,736 B`(**차 0 B**), L1만 +96.73 MiB.
  진단서가 예측한 "수리 전 TD가 12.61 GiB 초과"의 **부재가 측정됐다** — 아티팩트 가설 7종 배제,
  특히 **9/36 epoch의 완전 arm 분리(+5.17…+20.08 MiB)**로 **계측기가 5 MiB에서 arm 차이를
  해상함이 같은 job 안에서 실증**됐다.
- ★**Δ는 arm 효과가 아니다** — 전량이 L1이 505 토큰 더 큰 배치를 형성한 데서 나온다.
  **L1의 이탈도 arm이 아니라 boot 수준 산포**(3 job 교차 검증: 이탈 boot의 arm도 위치도 고정 안 됨).
- 결과 감사 `.../audit_908179_2026-09-14/VERDICT.md`: 인용 금지 **A8179-1…7**,
  필수 병기 **A8179-P1…P7**, 과대 해석 통로 **16개 전수 차단**.

---

## ★엔진 실물 사실 — grad guard 결함과 수리

**결함**: `event_loop_pdmux`의 `@torch.inference_mode()`는 **thread-local**인데 true-dual은 forward를
**role worker thread**로 넘긴다 ⇒ 그 스레드에 grad guard가 없었다(CPU 측정: `worker_grad_enabled=True`
vs `main=False`). 노출 경로는 **`mixer2_rms_norm_gated.py:97`의 bare Parameter 하나**뿐이고 그 줄은
**`n_groups != 1`에서만** 실행된다 ⇒ **Zamba2·Falcon-H1·Granite는 구조적 면역, NemotronH(`n_groups=8`)만
노출**. 산정 `R(T) = T × 1.2750 MiB`의 예측 천장(6.5k–7.7k)과 관측(6,245 통과 / 10,125 실패)이 일치.

**수리**(커밋 `a9cd8dd`): `_activate_role_context`가 task마다 새 guard를 연다
(`PDMUX_WORKER_GRAD_GUARD`, 기본 **`inference_mode`**). `no_grad`가 아닌 이유 = **legacy와 텐서
의미론까지 대칭**(비대칭을 다른 비대칭으로 바꾸는 것은 수리가 아니다).

★**소급 주장 금지**: 이전 TD job들(907100/907456/X1)은 Zamba2라 이 기전에 면역이었으나 **"worker가
grad 켜진 채 돈다"는 성질 자체는 그 job들에도 있었고** autograd 부기의 arm 비대칭 오버헤드는
**미측정**이다 — "오염됐다"도 "깨끗했다"도 쓸 수 없다.

★**RA4-1 정정**: "`inference_mode` × `forward_native` 조합이 GPU에서 한 번도 실행된 적 없다"는
**거짓** — job 907959의 **legacy 2 boot이 바로 그 조합**(NemotronH `n_groups=8`)으로 무오류 완주했다.

---

## 코드·문서 변경

**엔진**(커밋 `a9cd8dd`)
- `src/multiplex/multiplexing_mixin.py`: grad guard + **arm 대칭 메모리 계측 9키**(base payload,
  legacy도 같은 주기, realized 4키는 legacy에서 `null`), peak 리셋은 split-prefill 배치당 1회.
  ★관측자 효과를 **구현 전에 측정**해 설계를 바꿨다 — naive 3콜은 **wall-clock 3.55 %**로 ±3 % 예산
  초과, `memory_stats_as_nested_dict()` 1회로 **0.084 %**.

**하네스**(커밋 `b2b60e7`·`83d8cb9`)
- `results/r2_correctness/r2_correctness.sbatch`: 계측 3건(배너·동시성 1 프로브·`--request-rate inf`
  2셀)을 **추가 scored boot 0개**로 warm-up boot에 부착(`R2C_INSTRUMENT` 기본 0) · ★**scope guard**
  (`R2C_EXPECT_{MODEL,BACKEND,CTX}` 삼중 대조 + `exit 2`, 기본값 없이 **필수**) · `#SBATCH --time`
  01:15:00 → 02:30:00.
- `scripts/bootstrap/sync_engine_tree.sh`: manifest **24 → 25항목**(`layers/attention/flashinfer_backend.py`
  append, 앞 24항목 접두 보존).

**λ\* per-shape**(미커밋 → 이 커밋에 포함)
- `benchmarks/pdmux_eval/lambda_star.py` **신설** — λ\*가 스칼라가 아니라 **shape별 표**.
  W4의 두 phase가 **각자 shape의 λ\***로 스케일(실측 0.790×/0.794×). `PDMUX_SUSTAINABLE_RATE`는
  **fail-closed**(조용한 기본값 4 제거 — 게이트 #6 위반의 실체였다). 스키마 v2→**v3**.
- ★**신규 사실**: **W5/W6는 단일 shape가 아니라 한 Poisson 스트림에서 두 shape를 교대**한다 ⇒
  per-shape rate가 존재하지 않아 기본 거부되며, 도입된 혼합 정의는 **사전등록된 적 없다**.

**λ0 사전등록**(미커밋 → 이 커밋에 포함) — rev2/rev3/rev4 + 실행체 8개 + 테스트.

**정본**(미커밋 → 이 커밋에 포함) — `PROJECT_STATUS.md`(2026-09-14, 2026-09-14(2) 배너 + 게이트
**#201–237**) · `CONSENSUS.md` **rev73 → rev75**(§3 항목 221–257) · CEM Claim D 행 ·
`EXPERIMENT_ROADMAP.md`(P2 열린 항목 종결 + PASS 후에도 차단 유지) · `RESUME.md`.
메모리: `MEMORY.md`(**32.6KB → 8.8KB 압축**, 서사는 topic 파일로) ·
`slo-aware-scheduling-track.md`(2026-09-14, 2026-09-14(2) 절) · `deconfound-measurement-lessons.md`(항목 199–235).

---

## 사전등록·감사 (판정서 12건, 전부 파일로 존재)

| 대상 | 회차 | 결과 |
|---|---|---|
| 새 쌍 correctness(newpair) | rules → rev2 | `NO-GO`(死因 N3, 배너 48이 항등식) → **`GO-with-caveats`** |
| 907959 결과 | — | `CONFIRMED(scoped)` 귀속 + 엔진 진단서(H1) |
| 908020(등록 밖) | — | 운영 오류 판정 + **A908-7이 rerun rev2 등급을 철회** |
| 재실행(rerun) | rules → rev2 → rev3 → rev4 | `NO-GO` → `GO-with-caveats`(**철회**) → `NO-GO` → **`GO-with-caveats`** |
| 908179 결과 | — | `PASS` 채점 + 귀속 `PLAUSIBLE(조건부)` |
| λ0 | rules → rev2 → rev3 → rev4 | **4연속 `NO-GO`**(GPU 0) |

★**감사자 자기 철회·정정**(정본 가치 큼)
- λ0 rev3 판정서: 처방 **D16/D17이 자기가 만든 死因**임을 인정(자기 철회 6회차).
- rerun rev2 등급 **철회**(A908-7) — 단 그 판정서의 **caveat RRC-1…13은 살아 있다**(철회 사유와 무관).
- **비재현 3건**: rev3 감사의 harness sha · 907959 결과 감사의 "셀 B = 2"(등록 레시피로는 **11**) ·
  rev3 판정서의 "수리 후 λ\*(B)=0.30 → `KNEE_BRACKETED`"(쉬핑 코드는 `LADDER_TOO_HIGH`).
- rerun rev2의 **"6,245는 측정 상수"는 감사 실패**로 확정 — 단 F-a2b 강등을 지탱하는 것은 부분합이
  아니라 **907100 TD1이 수리 전 같은 기판에서 12,369를 완주한 사실**이고 **그것은 Zamba2 기판**이라
  NemotronH에 대한 "검정력 0"은 **미검증**이다.

---

## 열린 항목 / 다음 세션 시작점

### ★1. "수리가 OOM을 고쳤다"를 `CONFIRMED(scoped)`로 올리는 실험 (D-none)
`PDMUX_WORKER_GRAD_GUARD=none`만 이동시킨 대조 arm. **예보**: `none` TD가 907959와 같은 배치·같은
198.00 MiB 요청·같은 스택에서 OOM, 피크 축은 수리 arm 대비 `T × 1.2750 MiB` 초과. 비용 job당
≈0.10–0.17 GPU-h.
★**무수정으로는 실행 불가능하다**(감사 실현가능성 검사): `r2_correctness.sbatch:216`이 **모든
`PDMUX_*`를 unset**하므로 그 변수를 주어도 조용히 버려지고, 채점기가 `startswith("TD")`로 arm을
가르므로 `none` boot을 scored로 넣으면 **설계상 무조건 `FAIL`**이다. ⇒ 선행 조건 = (a) 하네스에
`R2C_GUARD` 노브(unset 루프 뒤 재수출) + 테스트 + argv 픽스처 재고정, (b) "예측된 `FAIL`은 정보이며
D5 재제출 금지 대상이 아니다"를 문자로 등록하는 **새 사전등록**. 하네스 sha가 또 움직이는 대가를
그 문서가 등재해야 한다.

### ★2. λ0 rev5 (F1–F6) — 그리고 트랙 자체에 대한 판단
rev4 死因 = `lambda0_lambda_inf.py`의 F5 술어가 **live에서 실효 항등식**(생산자가 전역 max 한 줄만
씀) + 그 수리가 라벨을 뒤집음(shape B **5/13**·A **3/12** 격자점, 반전 4점이 커버리지 밴드 안).
**F1–F6**이 `VERDICT_lambda0_rev4_2026-09-14.md` §6에 등록돼 있다(전부 GPU 0).
★**선택지 3개 — 사용자 결정 필요**:
 (a) rev5로 한 번 더(술어가 `I_log_offsets.txt`+`srv_warmup.log`에서 **셀별 자체 재계산**),
 (b) **생산자 하네스를 고쳐** 셀별 2줄을 쓰게 하고 재실행(GPU 필요, 새 사전등록),
 (c) **shape B를 0단계에서 빼고** shape A만 진행.
상류 원인은 "F5를 셀별로 평가할 산출물이 애초에 없다"는 **하네스 결함**이다.

### 3. rev4에 남은 비차단 권고 2건 (GPU 0, 문서·코드 해시 불변)
- **RA4-2**: E5 트리거 계열이 6번째 하드 에러(`A view was created in inference mode and is being
  modified inplace in normal mode`)를 **놓친다**. 교정안: `RuntimeError.*(?i:inference[ _](tensor|mode))|RuntimeError.*InferenceMode`.
- **RA4-9**: §9-4 대조표에 철회된 rev3 문안(`src_dirty` 공백) 잔재 1단어.

### 4. 승계 사슬의 누수 (RA4-6)
현재 승계 총량은 **96건 + 전사 의무 2건**(NP·NPC·N·A908·RRC·RA3·RA4·RR 8계열)인데 다음 회차
전방 강제 조항(RR-14)은 **44건만** 요구한다 ⇒ **40건이 무강제**다. 다음 사전등록은 §11.1 목록과
**항목 수를 대조**해야 한다.

### 5. 미해결 등재
- **W5/W6 혼합 정의**는 사전등록된 적 없다 — 그 워크로드를 돌리려면 별도 규칙층 감사.
- **4번째 축**(노드·물리 GPU)이 사전등록 §10에 미등록이었다(메모리 채널은 `avail mem` 사다리
  완전 일치로 닫힘).
- **λ_inf(B) 인용 불가**로 shape B 사다리의 앵커 근거가 FALLBACK뿐이다.
- λ0 판정서가 남긴 **인용 금지 Q1–Q5·L1–L3·R3C-1…4·R4C-1…6** 유효.
- `check_doc_facts.py` **2 FAIL 선재**(`DESIGN_A1_REV2_STICKY_2026-08-25.md`의 자기보고 수치 stale).
- 라인 인용 이동 앵커(W4 구현이 옮긴 6건)는 `line_citations.json`에 **미등록**(검사는 88/0 통과).

---

## 미완·주의

- ★★**"수리가 OOM을 고쳤다"는 `PLAUSIBLE(조건부)`이지 `CONFIRMED`가 아니다.** 반증 4건이 전부
  실패했지만 **반증 실패는 확증이 아니다**. 결정적 arm(`none`)이 없고, 907959↔908179 사이에 축이
  **4개** 움직였다(mixin sha · harness sha · `R2C_INSTRUMENT` · **노드/물리 GPU**).
- ★**job 908179의 `PASS`가 인증하는 것은 한 문장뿐**: *"위 튜플에서 `PDMUX_TRUE_DUAL_WORKER=1`
  경로와 legacy 루프가 S층·O층에서 같은 토큰 id를 내고 그동안 cudagraph가 두 arm 모두 켜진 채
  유지된다."* **cudagraph는 decode 한정**(prefill 41/41 `False`, 양 arm).
  ★**Claim D 선결 #5는 새로 닫히지 않았다** — 이미 (Zamba2, triton) 한정으로 닫혀 있던 술어의
  **스코프가 두 번째 쌍으로 확장**됐을 뿐이다. **PASS는 P2 착수를 승인하지 않는다**(NP-8 — 남은
  블로커: λ0 `NO-GO`, W4, 게이트 #6).
- ★**RA3-1**: 이 재실행 회차는 **어느 결과가 나와도 H1을 닫지 못한다** — F-a1이 원인 무차별(이 트랙
  4 job 중 1건에서 OOM 0건인데도 거짓) + 공허 참 가능, F-a2는 라벨 무이동, F-b는 양쪽 `UNREALIZED`
  분지. 그래서 게이트 라벨이 산출물이다.
- ★**908020을 숨기지 마라**(C4): 어떤 결과 서술도 **"이 사전등록은 3개 job을 소비했고 첫째는
  등록 밖이었다"**를 함께 적어야 한다.
- **결과 문서·정본은 요약이 아니라 sha 핀 원문에서 전사**해야 한다(RA4-1이 A908-5 요약의 스코프
  소실을 실증했다).
- `srv_*.log`·`tel_*.jsonl`은 용량 때문에 **버전관리 밖**이다 ⇒ **F-a1의 1차 증거(OOM 계수·스택)를
  결과 문서 본문에 전사**해야 한다.
- `push` 안 함(정책). 루트 SLURM 로그 커밋 안 함.

## 다음 세션 `catch-up` 시작점 (한 줄)

> **true-dual worker가 autograd 켜진 채 돌던 결함을 찾아 수리했고, 등록 튜플 재실행(job 908179)이
> `PASS`로 TD가 자기를 죽였던 배치를 완주했다 — 그러나 "수리가 고쳤다"는 `PLAUSIBLE(조건부)`이며
> 결정적 `none` arm이 없고 하네스가 그 노브를 삼켜 새 사전등록이 필요하다. Claim D 선결은 하나도
> 새로 닫히지 않았고 P2는 여전히 차단이다. λ0는 4연속 `NO-GO`(GPU 0)로 트랙 자체에 대한 사용자
> 결정((a) rev5 / (b) 하네스 수정 후 재실행 / (c) shape B 제외)이 필요하다. GPU 누적 1.1075 GPU-h,
> 그중 0.1525는 등록 밖 실행으로 어떤 게이트도 진전시키지 못했다.**
