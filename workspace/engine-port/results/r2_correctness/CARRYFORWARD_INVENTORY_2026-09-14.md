# 승계(carryforward) 인벤토리 — 2026-09-14 (doc-steward)

> **목적.** `VERDICT_rerun_rev4_2026-09-14.md:102`(RA4-6)이 지적한 승계 사슬 누수를
> `audit_908179_2026-09-14/VERDICT.md:450`이 유효함으로 재확인했다: §11.1(65건+전사의무
> 2건)+§11.3(RR-1…19)+RA4(rev4 신규 12건)의 **8계열 전량을 다음 사전등록 §11.1이 항목 수로
> 대조**할 수 있게, 이 문서가 라벨마다 (id·요지·출처·분류)를 전수 나열한다. **요지는 원문에서
> 그대로 뽑은 구(句)다 — 이 문서 작성자의 의역이 아니다**(RA4-1이 요약의 스코프 소실을
> 실증했으므로, 이 인벤토리 자체가 그 실패를 반복하지 않도록 문구를 원문 그대로 전사한다).
> 분류는 각 항목 원문의 지배 동사("~쓸 수 없다/인용 금지"=인용금지, "~반드시 병기/함께
> 전사"=필수병기, "~해야 한다/등록하라"=절차의무)를 따른다 — 경계 사례는 항목 끝에 근거를
> 적었다.
>
> **경로 표기.** 모든 경로는
> `workspace/engine-port/results/r2_correctness/` 기준 상대경로.
>
> **이 문서 자체는 정본이 아니다** — CONSENSUS/PROJECT_STATUS 위계 밖의 **작업 산출물**이며,
> 다음 사전등록(§11.1)이 참조할 입력이다. 등급 판정(HE0·정책 순위·Claim D/E 등)에는
> 아무 영향이 없다(**새 성능 판정 0건**).

---

## 1. NP 계열 (10건) — `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md` §12

| id | 요지(원문 전사) | 출처 | 분류 |
|---|---|---|---|
| NP-1′ | "Nano-9B-v2에서 PD-mux가 정확하다/작동한다"는 **PASS여도 쓸 수 없다** | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:649` | 인용금지 |
| NP-2 | 이 job의 **어떤 수치도 성능 결과가 아니다** | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:653` | 인용금지 |
| NP-3′ | **I3는 λ\*가 아니다** — 닫힌 루프 포화 처리율은 open-loop λ\*의 **상한**일 뿐, "λ\*를 측정했다"는 문장 금지 | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:657` | 인용금지 |
| NP-4 | **이 게이트는 Zamba2 결과를 되살리거나 확장하지 않는다** — 907100·907456·X1은 (Zamba2-2.7B, triton) 한정 동결 | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:664` | 인용금지 |
| NP-5 | **C층 수치는 판정도 비교도 아니다** — 907032의 7/32와 비교 금지 | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:669` | 인용금지 |
| NP-6 | **`INCONCLUSIVE`나 `NO_VERDICT_*`는 "true-dual이 실패했다"가 아니다** — 측정 실패다(교훈 21·게이트 #21) | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:672` | 인용금지 |
| NP-7′ | **`max_mamba_cache_size: 48`은 측정이 아니라 `--disable-radix-cache --max-running-requests 48`의 연역이다** — "구속 자원이 48임을 확인했다"는 쓸 수 없다(이전 NP-7은 항등식을 사실로 승격했던 오류를 D1으로 교체) | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:675` | 인용금지 |
| NP-8 | (신규, D13) **PASS는 P2 캠페인 착수를 승인하지 않는다** — P2의 필요조건 하나만 닫는다, 남은 블로커: λ0 `NO-GO`·W4·게이트 #6 | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:684` | 인용금지 |
| NP-9 | (신규, D6/D8) **I1–I3의 모든 수치는 D44 분할·legacy 루프·warm-up boot 한정**이다 — I2가 주는 것은 클라이언트 관측 ITL/TTFT, 엔진 decode step 아님 | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:689` | 필수병기 (모든 I1-I3 수치 인용 시 이 스코프를 함께 적어야 함) |
| NP-10 | (신규, D2/D3) booted L boot 2개 미만에서 나온 `S_TIER_MISMATCH`는 **FAIL로 인용하지 않는다** — `NO_VERDICT_INFRA`로 보고 | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:696` | 인용금지 |

## 2. NPC 계열 (10건) — 같은 문서 §12 "재감사(rev2) 캐비앳"

| id | 요지(원문 전사) | 출처 | 분류 |
|---|---|---|---|
| NPC-A | §8의 D2′ 오버라이드 발화 시 **`checks.S_pairs` 전체와 `per_boot[*].checks.B1_BOOTED`를 전사**하고 §9-D5 재제출 1회를 쓰지 않는다 — "교차-arm 불일치가 없었다"는 쓸 수 없다 | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:706` | 절차의무 (D2′ 발화 시에만 발동하는 조건부 의무) |
| NPC-B | **F2′ 충족은 KV 여유 가정의 "확인"이 아니다** — 형식 요건만 충족, 결과는 **기록으로만** 쓴다 | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:714` | 인용금지 |
| NPC-C | `r2_correctness.sbatch`의 I1 주석을 근거로 "48이 이 ctx/mem에서 측정됐다"를 **인용하는 것을 금지** | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:719` | 인용금지 |
| NPC-D | "`R2C_INSTRUMENT=0`이면 907100·907456과 명령 단위로 동일하다"는 **거짓이다** — provenance 블록이 `import flashinfer` 프로세스를 무조건 1회 실행 | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:723` | 인용금지 |
| NPC-E | provenance는 flashinfer 휠의 **버전 문자열만** 기록 — "같은 flashinfer에서 쟀다"는 쓸 수 없고 "0.6.10이라고 보고한 설치본에서"까지만 | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:728` | 인용금지 |
| NPC-F | "manifest 변경이 저장소에 파급을 주지 않는다"는 **쓸 수 없다** — 비-테스트 소비자 2곳이 15→17→24 단계에서 이미 깨져 있었다 | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:735` | 인용금지 |
| NPC-G | `tests/test_r2_correctness_instrument.py`의 `BANNER` 상수는 **합성 픽스처**다 — 측정값으로 **인용 금지** | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:740` | 인용금지 |
| NPC-H | 제출 전 "전체 CPU 회귀"의 합격 기준은 **이 트랙(`test_r2_correctness_*`)으로 한정**한다 | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:744` | 절차의무 |
| NPC-I | I3의 `λ_inf`는 **legacy warm-up boot·D44·cudagraph ON에서만** 측정 — λ0 결과 문서는 NP-9의 한정을 **그대로 승계해야 한다** | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:747` | 필수병기 |
| NPC-J | §9(f) 벽시계 초과 재제출은 §9-D5 재제출 정책 정의역 밖 — **두 경로를 합쳐 총 재제출 1회로 센다** | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:752` | 절차의무 |

## 3. (D13-i) 필수 전사 항목 — "전사 의무 2건" (같은 문서 §12 말미)

| id | 요지(원문 전사) | 출처 | 분류 |
|---|---|---|---|
| D13-i-1 | 결과 문서는 `r2_correctness_report.json`의 **`checks.server_args_across_boots`**(`attention_backend`·`context_length`·`random_seed`·`max_running_requests`·`pdmux_config_path`·`disable_cuda_graph`·`triton_attention_num_kv_splits`)를 전사 | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:762` | 필수병기 |
| D13-i-2 | 결과 문서는 `provenance.txt`의 **`model=` 줄과 `context_length=… attention_backend=…` 줄, `instrument=` 줄·`flashinfer_version` 줄**을 전사 — `verdict.txt` 단독 인용 금지 | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md:765` | 필수병기 |

## 4. N 계열 (13건) — `audit_907959_2026-09-13/VERDICT.md` §12 "필수 병기 문안"

| id | 요지(원문 전사) | 출처 | 분류 |
|---|---|---|---|
| N-1 | job 907959의 판정은 **`FAIL`**이며 verdict rule v2의 기계적 산물, **비트 단위 재현** | `audit_907959_2026-09-13/VERDICT.md:399` | 필수병기 |
| N-2 | "C층은 진단 전용"은 **"C층 크래시가 판정에 안 들어간다"는 뜻이 아니다** — traceback 1건이 단독으로 FAIL을 낸다 | `audit_907959_2026-09-13/VERDICT.md:401` | 인용금지 |
| N-3 | 지지되는 문장은 스코프 한정 한 문장뿐 — **"true-dual이 (일반적으로) OOM으로 죽는다"는 이 job 단독으로 쓸 수 없다** | `audit_907959_2026-09-13/VERDICT.md:403` | 인용금지 |
| N-4 | S/O 불일치 0은 **측정 기록이며 동등성 인증이 아니다** — 인증문은 `PASS`에서만 허용, 이 job은 `FAIL` | `audit_907959_2026-09-13/VERDICT.md:405` | 인용금지 |
| N-5 | `C cross-arm: 31`은 **토큰 분기가 아니라 "TD 쪽 출력이 비어 있는 요청 31건"** — 결정성의 증거 아님, 907032의 7/32와 비교 금지 | `audit_907959_2026-09-13/VERDICT.md:407` | 인용금지 |
| N-6 | `unsafe_decisions=16`은 `controller_decision.safe=False` 계수이며 **prefill∧decode 동시성의 표지**, 결함이 아니다 | `audit_907959_2026-09-13/VERDICT.md:409` | 필수병기 |
| N-7 | (★NP-9/D6-ii 정정) I2(동시성 1)의 TTFT/ITL은 **D44 값이 아니라 비분할(108 SM) 값**이다 | `audit_907959_2026-09-13/VERDICT.md:411` | 필수병기 |
| N-8 | (★F5/λ_inf(B) 인용 금지) I3b 셀별 `#running-req` 최댓값은 **2**(I3a는 48) — **`λ_inf(B)=0.6956` 인용 금지** | `audit_907959_2026-09-13/VERDICT.md:413` | 인용금지 |
| N-9 | `λ_inf(A)=3.0939 req/s`는 closed-loop 포화 처리율 상한 프로브이며 **λ\*가 아니다**(NP-3′) | `audit_907959_2026-09-13/VERDICT.md:415` | 인용금지 |
| N-10 | I3b vs 905835 +3.00%는 out/ctx 동시 변경과 같은 크기 — **"closed-loop가 open-loop를 상회한다"의 증거로 쓸 수 없다** | `audit_907959_2026-09-13/VERDICT.md:417` | 인용금지 |
| N-11 | I2의 `max ITL=929.91ms`는 원인 미확정 — **"ITL 바닥 13.0ms"만 인용하지 말고 이 꼬리를 병기하라** | `audit_907959_2026-09-13/VERDICT.md:419` | 필수병기 |
| N-12 | (F2′) §5의 문안을 그대로 승계한다(NPC-B) | `audit_907959_2026-09-13/VERDICT.md:421` | 필수병기 |
| N-13 | (스코프 배선, D13-i) `verdict.txt` 단독 인용 금지 — `checks.server_args_across_boots`와 provenance 줄을 함께 전사 | `audit_907959_2026-09-13/VERDICT.md:423` | 필수병기 |

## 5. A908 계열 (7건) — `audit_908020_2026-09-14/VERDICT.md` §9 "인용 금지 — 신규"

| id | 요지(원문 전사) | 출처 | 분류 |
|---|---|---|---|
| A908-1 | job 908020은 사전등록(rev2)의 실행이 **아니다** — 어떤 예보도 908020 위에서 채점되지 않았고 채점해서도 안 된다 | `audit_908020_2026-09-14/VERDICT.md:407` | 인용금지 |
| A908-2 | 908020의 TD1 8,497을 907959의 6,245와 비교하는 문장은 **인용 금지**(907100 TD1이 수리 이전에 12,369를 완주) | `audit_908020_2026-09-14/VERDICT.md:409` | 인용금지 |
| A908-3 | 908020의 `PASS`를 "회귀 없음"으로 인용할 수 없다 — n=1이고 축 3개 동시 이동, C층 arm-내 불일치는 3→7로 **증가** | `audit_908020_2026-09-14/VERDICT.md:411` | 인용금지 |
| A908-4 | `worker_grad_guard`와 legacy `null` 4키의 "100%" 일치는 **항등식**이며 확인으로 인용할 수 없다 — "100%"는 스냅샷 단위 | `audit_908020_2026-09-14/VERDICT.md:413` | 인용금지 |
| A908-5 | 908020은 `inference_mode` 가드와 `forward_native`(n_groups≠1) 분기의 상호작용을 **한 번도 실행하지 않았다** — 안전 근거는 CPU 분석뿐 | `audit_908020_2026-09-14/VERDICT.md:415` | 필수병기 (★RA4-1이 이 문장 재인용 시 스코프 보존을 요구, 아래 §8 참조) |
| A908-6 | 이 사전등록은 **2개 job을 소비했고 첫째(908020)는 등록 밖 실행**이었다 — 어떤 회차가 PASS를 내도 이 문장 없이 게이트 상태를 서술할 수 없다 | `audit_908020_2026-09-14/VERDICT.md:417` | 필수병기 |
| A908-7 | `VERDICT_rerun_rev2_2026-09-14.md`의 등급 `GO-with-caveats`는 **인용할 수 없다** — 올바른 등급은 `NO-GO`(死因 N3) | `audit_908020_2026-09-14/VERDICT.md:419` | 인용금지 |

## 6. RRC 계열 (13건) — `rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md` §5

| id | 요지(원문 전사) | 출처 | 분류 |
|---|---|---|---|
| RRC-1 | "ordinal 38은 4 boot 어디서도 in-flight로 관측되지 않았다"는 **원자료로 반증** — §6-2-3 세 번째·네 번째 bullet과 RR-15 근거 문장은 **전사해서는 안 된다** | `rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md:172` | 인용금지 |
| RRC-2 | 검정력 브래킷(2.88/4.04 GiB)은 **유지**되나 근거는 "ordinal 38 미관측"이 아니라 **`P` 자체가 데이터 의존**이라는 것 | `rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md:185` | 필수병기 |
| RRC-3 | 1차 판정서의 4.04 GiB는 907959 4/4 boot 실현으로 **철회되지 않는다** | `rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md:192` | 필수병기 |
| RRC-4 | F-a1은 실패 원인을 가리지 않는다(907032가 OOM 0건인데도 거짓) — **"F-a1 거짓"을 "H1이 틀렸다"로 쓸 수 없다**, OOM 계수·스택 병기 의무 | `rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md:197` | 필수병기 |
| RRC-5 | D5 재제출 시 F-a/F-b 귀속 미등록 — **재제출 발생 시 두 job 모두에서 채점해 보고, 한쪽만 인용 금지** | `rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md:206` | 필수병기 |
| RRC-6 | epoch↔ordinal 검사 검출력은 ordinal 38 한 점뿐 — **`|D_b|`가 38에 가까운데 제외됐다면 정렬 붕괴 신호로 보고해야 한다** | `rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md:212` | 필수병기 |
| RRC-7 | F-b2 대역 A⊃D, F-b1 D′⊂B′∪C′ 겹침 — **음의 부호가 나오면 둘 다 보고하고 한쪽만 인용하지 말 것** | `rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md:219` | 필수병기 |
| RRC-8 | §10 (경우 1) 문장은 F-a1에 조건화돼 있지 않다 — **쓸 때는 F-a1의 참·거짓을 같은 문단에 반드시 병기하라** | `rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md:225` | 필수병기 |
| RRC-9 | `S`·`Δ`는 arm당 n=2, boot 간 분산 미측정 — **인용 시 "boot 간 산포 미측정"을 반드시 병기** | `rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md:231` | 필수병기 |
| RRC-10 | F-a2에는 거짓 분지가 없다 — **F-a2 참을 H1의 추가 확증으로 인용할 수 없다** | `rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md:237` | 인용금지 |
| RRC-11 | 발행률 정정: `event=="runtime_snapshot"`만 세면 **L1 222.4/s·L2 234.7/s·TD1 153.2/s·TD2 153.1/s**(줄 수 기준 아님) | `rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md:241` | 필수병기 |
| RRC-12 | F-b1은 대부분 D44 운영점 **밖**(비분할 idx0)에서 측정 — "D44에서 쟀다"는 쓸 수 없다, arm 간 decode 중첩 차이 미측정 | `rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md:245` | 인용금지 |
| RRC-13 | C5(M-J 테스트 미추가)를 열어 둔 근거는 틀렸으나 **결론(M-J 25/25 통과)은 유지** — 실제 근거는 F-d 실현값 2키 한정 | `rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md:253` | 절차의무 |

## 7. RA3 계열 (12건) — `rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md` §4

| id | 요지(원문 전사) | 출처 | 분류 |
|---|---|---|---|
| RA3-1 | (★최우선·검정력 공시) **이 캠페인은 어느 결과가 나오든 H1을 닫지 못한다** — 반증 가능 채널은 F-a1 하나뿐, 최대 산출은 §10 (경우 2) 스코프 한정 존재 문장 하나 | `rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md:126` | 필수병기 |
| RA3-2 | F-a2a 문턱 20%는 Zamba2 귀무 3건 중 1건(907456)이 이미 넘는다 — **예보가 아니라 기록 항목** | `rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md:128` | 인용금지 |
| RA3-3 | "6,245는 도착 조합의 실현값"은 **`NOT-YET-SUPPORTED`** — NemotronH 기판의 "F-a2b 검정력 0"은 **미검증** | `rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md:130` | 필수병기 |
| RA3-4 | §6-2-3·RR-15의 "ordinal 38 미관측" 근거 문장은 **거짓**(스냅샷 58·58·54·54) — **전사 금지**, 근거를 "P는 데이터 의존"으로 교체 | `rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md:132` | 인용금지 |
| RA3-5 | rev3 본문에 "RRC" 0회 — **"rev3는 선행 caveat를 전부 반영했다"고 쓸 수 없다** | `rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md:134` | 인용금지 |
| RA3-6 | inference tensor 하드 에러는 최소 4종("정확히 둘"은 반증됨) — rev4는 트리거를 **계열로 등록**해야 함 | `rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md:136` | 절차의무 |
| RA3-7 | scope guard는 model/backend/ctx 3축만 검사 — **"하네스가 스코프를 보증한다"고는 쓸 수 없다** | `rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md:138` | 인용금지 |
| RA3-8 | (11)은 (7)과 독립 채널이 아니다 — 고유 정보는 **spool-copy 검출 하나로 한정해 인용** | `rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md:140` | 필수병기 |
| RA3-9 | §10 (경우 2)를 쓸 때 **F-a2 판정을 무조건 병기하라** | `rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md:142` | 필수병기 |
| RA3-10 | 대역 동률 규약(A⊃D 등) 미등록 — **rev4는 우선순위 규약을 등록해야 한다** | `rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md:144` | 절차의무 |
| RA3-11 | §8 예보 전형 0.20 GPU-h는 **미검증** — 9B·flashinfer·ctx16384 실비는 한 번도 측정된 적 없다 | `rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md:146` | 필수병기 |
| RA3-12 | (승계 사슬) 결과 문서는 **RA3-1…12와 RR-1…15를 함께 편입**해야 하며, RR-15는 RA3-4 정정 없이 전사할 수 없다 | `rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md:148` | 절차의무 |

## 8. RA4 계열 (12건) — `rerun_prereg/VERDICT_rerun_rev4_2026-09-14.md` §4

| id | 요지(원문 전사) | 출처 | 분류 |
|---|---|---|---|
| RA4-1 | (★최우선·사실 정정) "상호작용은 GPU에서 한 번도 실행된 적 없다"는 **거짓** — 907959 legacy 2 boot이 바로 그 조합으로 무오류 완주. A908-5 **요약**이 스코프를 지운 것이며 전사 원본으로 쓰면 거짓을 전파 | `rerun_prereg/VERDICT_rerun_rev4_2026-09-14.md:92` | 필수병기 |
| RA4-2 | E5 트리거 계열이 하드 에러 최소 1종("A view was created in inference mode…")을 놓친다 — 저비용 교정 정규식 제시 | `rerun_prereg/VERDICT_rerun_rev4_2026-09-14.md:94` | 절차의무 (비차단 권고, GPU 0) |
| RA4-3 | (9)는 충분조건이지 필요조건 아님, 하네스를 보지 않음 — `commit=`이 실제 하네스를 가리킨다는 보장은 §9-3 P6과 C1 (7)(11)만 준다 | `rerun_prereg/VERDICT_rerun_rev4_2026-09-14.md:96` | 필수병기 |
| RA4-4 | "61–68%"·"23–26/38"은 문서 전역에서 철회되지 않았다(`:783`·`:1428` 생존) — **전사 금지** | `rerun_prereg/VERDICT_rerun_rev4_2026-09-14.md:98` | 인용금지 |
| RA4-5 | "1.45×"는 L1/TD1 쌍 값 — boot 극단비는 **234.7/153.1=1.53×** | `rerun_prereg/VERDICT_rerun_rev4_2026-09-14.md:100` | 필수병기 |
| RA4-6 | (전방 강제는 84건 중 44건뿐) RR-14는 다음 회차에 44건만 요구 — **NP-1′…NP-10·NPC-A…J·N-1…13·A908-1…7 = 40건 무강제**(이 인벤토리를 낳은 finding) | `rerun_prereg/VERDICT_rerun_rev4_2026-09-14.md:102` | 절차의무 |
| RA4-7 | 게이트 라벨이 인증하는 것은 승계 문서(NP-1′+NP-8)의 한 문장뿐 — **라벨 문장과 H1 문장을 같은 문단에서 섞어 쓰지 않는다** | `rerun_prereg/VERDICT_rerun_rev4_2026-09-14.md:104` | 필수병기 |
| RA4-8 | §9-3 커밋 목록은 stale(7/11 이미 커밋, job_907959/는 목록에 없음) — **감사 시점 `git status`로 재생성해야 한다** | `rerun_prereg/VERDICT_rerun_rev4_2026-09-14.md:106` | 절차의무 |
| RA4-9 | §9-4 대조표의 `src_dirty` 공백→(9)는 **철회된 rev3 잔재** — 규범은 §3-b C1 (9)(★본 세션에서 처리, 아래 §9 참조) | `rerun_prereg/VERDICT_rerun_rev4_2026-09-14.md:108` | 절차의무 |
| RA4-10 | (7)(8)은 제출 절차에 취약(C3 예산 0) — **커밋은 내용 무변경 이동이어야 한다** | `rerun_prereg/VERDICT_rerun_rev4_2026-09-14.md:110` | 절차의무 |
| RA4-11 | (9)의 잔존 환경 의존성(fail-closed이나 비가역) — 저비용 강화 정규식 제시(같은 4 아티팩트에서 T/T/T/F 유지) | `rerun_prereg/VERDICT_rerun_rev4_2026-09-14.md:112` | 절차의무 |
| RA4-12 | F-a2a 강등은 출력공간을 바꾸지 않았다 — **결과 문서가 §0-a2 표를 지위 근거로 인용해서는 안 된다** | `rerun_prereg/VERDICT_rerun_rev4_2026-09-14.md:114` | 인용금지 |

## 9. RR 계열 (19건) — `rerun_prereg/PREREG_RERUN_2026-09-13.md` §11.3

| id | 요지(원문 전사) | 출처 | 분류 |
|---|---|---|---|
| RR-1 | 이 회차의 어떤 수치도 성능 결과가 아니다 — **"true-dual이 빨라졌다/메모리를 덜 쓴다"는 쓸 수 없다** | `rerun_prereg/PREREG_RERUN_2026-09-13.md:1301` | 인용금지 |
| RR-2 | (귀속 금지) "grad guard가 OOM의 원인이었다"는 **이 job 단독으로 쓸 수 없다**(세 축 동시 이동) | `rerun_prereg/PREREG_RERUN_2026-09-13.md:1306` | 인용금지 |
| RR-3 | (소급 금지) 이 수리는 907100/907456/X1의 결론을 바꾸지도 되살리지도 않는다 — 단 그 job들의 arm 비대칭 오버헤드는 **크기 미측정** | `rerun_prereg/PREREG_RERUN_2026-09-13.md:1310` | 인용금지 |
| RR-4 | `gpu_mem_*`는 PyTorch 캐싱 할당자 부기, 장치 전체 사용량 아님 — `available_gpu_mem`과 **직접 비교하지 말 것** | `rerun_prereg/PREREG_RERUN_2026-09-13.md:1316` | 인용금지 |
| RR-5 | ε=1.00GiB·θ=0.3825MiB/token은 **사전 판단**이며 측정이 아니다 — "수리 후 양성 arm 간 피크 차이 미측정"과 함께 인용해야 한다 | `rerun_prereg/PREREG_RERUN_2026-09-13.md:1321` | 필수병기 |
| RR-6 | `worker_grad_guard`(설정값) ≠ `*_worker_grad_enabled`(실현값) — **같은 것으로 인용하지 말 것**(게이트 #176) | `rerun_prereg/PREREG_RERUN_2026-09-13.md:1326` | 인용금지 |
| RR-7 | `R2C_INSTRUMENT=0`이라 감사 §13-D1(i)(셀별 2줄) 하네스 수리를 **하지 않았다** — 닫혔다고 쓸 수 없다 | `rerun_prereg/PREREG_RERUN_2026-09-13.md:1331` | 인용금지 |
| RR-8 | "0.14 µs"는 계측기 비용이 아니다(서로 다른 입력에서 측정됨, C1) — **표현은 인용할 수 없다**, ±3% 결론은 유지 | `rerun_prereg/PREREG_RERUN_2026-09-13.md:1337` | 인용금지 |
| RR-9 | (C2) `reset_peak_memory_stats()` 발행 스레드 비대칭 — **TD 과소보고, 편향 방향을 반드시 병기**(상한 156 MiB) | `rerun_prereg/PREREG_RERUN_2026-09-13.md:1344` | 필수병기 |
| RR-10 | (C3) `INSTR=0`이 warm-up I2/I3 부하를 없애 JIT 예열을 줄인다 — "907959와 같은 조건" 주장 불가 | `rerun_prereg/PREREG_RERUN_2026-09-13.md:1351` | 인용금지 |
| RR-11 | (C5) 변이 M-J(실현값→설정값 에코)가 등록 13개 테스트를 통과 — **"테스트로 고정했다"고 쓸 수 없다** | `rerun_prereg/PREREG_RERUN_2026-09-13.md:1355` | 인용금지 |
| RR-12 | (C6) F-d의 두 절은 **항등식**(unset 강제) — "F-d가 충족됐다"를 정보량 있는 확인으로 인용 불가 | `rerun_prereg/PREREG_RERUN_2026-09-13.md:1360` | 인용금지 |
| RR-13 | (C7) 대역 B(1.00–6.30 GiB)가 도달가능 Δ 상한 구간의 약 62%를 흡수 — **arm의 순간값 비교에 쓸 수 없다**(발행률 1.45× 차이) | `rerun_prereg/PREREG_RERUN_2026-09-13.md:1366` | 인용금지 |
| RR-14 | (C10, 승계 사슬의 강제) 다음 회차는 이 문서를 sha로 핀하고 **RR-1…19+RA3-1…12+RRC-1…13을 §11.1에 명시 편입해야 한다** — ★이 조항 자체가 44건만 요구(RA4-6이 지목한 누수의 원문) | `rerun_prereg/PREREG_RERUN_2026-09-13.md:1373` | 절차의무 |
| RR-15 | (★rev4 근거 교체) 검정력 브래킷(2.88/4.04 GiB)은 유지, "ordinal 38 미관측" 근거는 **거짓이며 전사 금지** — 근거는 "`P`가 데이터 의존" | `rerun_prereg/PREREG_RERUN_2026-09-13.md:1382` | 인용금지 |
| RR-16 | (RA3-6) "inference tensor가 no-grad보다 엄한 지점은 정확히 둘"은 **실측으로 거짓**(최소 4종) — 저울질 인용 시 반드시 4종 기준으로 적는다 | `rerun_prereg/PREREG_RERUN_2026-09-13.md:1399` | 필수병기 |
| RR-17 | (RA3-2/RRC-10) F-a2a·F-a2b는 예보가 아니라 **기록 항목** — F-a2 참을 H1 추가 확증으로 인용할 수 없다 | `rerun_prereg/PREREG_RERUN_2026-09-13.md:1406` | 인용금지 |
| RR-18 | (RA3-3) "6,245는 도착 조합 실현값" 논증은 성립하지 않음(결론만 유지) — NemotronH 기판 "검정력 0"은 **미검증** | `rerun_prereg/PREREG_RERUN_2026-09-13.md:1412` | 필수병기 |
| RR-19 | (RRC-13) C5를 열어 둔 근거("인증된 해시를 움직인다")는 **사실이 아니다** — 결론(M-J 미추가)은 유지, 실제 근거는 F-d TD 실현 2키 한정 | `rerun_prereg/PREREG_RERUN_2026-09-13.md:1420` | 절차의무 |

---

## 10. 계열별 건수와 총계

| 계열 | 건수 | 출처 문서 | §/제목 |
|---|---|---|---|
| NP | 10 | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md` | §12 "인용 금지" |
| NPC | 10 | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md` | §12 재감사(rev2) 캐비앳 |
| N | 13 | `audit_907959_2026-09-13/VERDICT.md` | §12 "필수 병기 문안" |
| A908 | 7 | `audit_908020_2026-09-14/VERDICT.md` | §9 "인용 금지 — 신규" |
| RRC | 13 | `rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md` | §5 "등록 caveat" |
| RA3 | 12 | `rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md` | §4 "등록 caveat" |
| RA4 | 12 | `rerun_prereg/VERDICT_rerun_rev4_2026-09-14.md` | §4 "등록 caveat" |
| RR | 19 | `rerun_prereg/PREREG_RERUN_2026-09-13.md` | §11.3 "신규 인용 금지" |
| **소계 (8계열)** | **96** | | |
| D13-i (전사 의무) | 2 | `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md` | §12 "(D13-i)" |
| **총계** | **98** | | |

**분류별 집계** (96건, D13-i 2건은 둘 다 필수병기로 별도 표에 포함돼 있어 아래 합계엔 넣지 않음):

| 분류 | 건수 |
|---|---|
| 인용금지 | 49 |
| 필수병기 | 32 |
| 절차의무 | 15 |
| **합** | **96** |

(기계적 재검증: `grep -oE '\| (인용금지\|필수병기\|절차의무)(...)?\|$'`로 §1–§9 표를 전수
스캔한 결과 98행 — 96건 + D13-i 2건(둘 다 필수병기) = 필수병기 34/인용금지 49/절차의무 15
(D13-i 포함 시); D13-i를 빼면 위 표대로 49/32/15=96.)

---

## 11. 실측 카운트 vs 핸드오프 진술 대조

`session_handoff_2026-09-14.md`·`VERDICT_rerun_rev4_2026-09-14.md:102`(RA4-6)·
`audit_908179_2026-09-14/VERDICT.md:450`이 공통으로 진술한 "**96건 + 전사 의무 2건, 8계열
(NP·NPC·N·A908·RRC·RA3·RA4·RR)**"과 **본 문서의 실측 카운트(96건 + 2건, 8계열)가 정확히
일치한다.** 계열별 부분합도 원문 등록 카운트와 일치: NP-1′…NP-10(10) · NPC-A…J(10) ·
N-1…13(13) · A908-1…7(7) · RRC-1…13(13) · RA3-1…12(12) · RA4-1…12(12) · RR-1…19(19).
검증 산식: `PREREG_RERUN_2026-09-13.md:1265`의 §11.1 자체 진술("**합 65건 + 전사 의무
2건**" = NP10+NPC10+N13+A908(7)+RA3(12)+RRC(13) = 65) + §11.3의 RR(19) = 84
(`RA4-6`의 "84건 중 44건" 표현과 일치) + rev4 신규 RA4(12) = **96**.

차이 없음 — **96과 다르지 않다.**

---

## 12. 이 문서의 사용법 (다음 사전등록 §11.1을 위한 메모)

다음 사전등록(예: newpair/rerun 후속 회차 또는 P2 착수 사전등록)의 §11.1은 이 문서의
**8계열 96건 + D13-i 2건 전부**를 자신의 승계 목록에 명시적으로 편입해야 한다(RR-14 원문의
"44건" 축소를 반복하지 말 것 — RA4-6이 지목한 실패 형태). 항목 수 대조는 위 §10 표를 그대로
쓰면 된다. **이 문서를 전사 원본으로 인용하지 말 것** — 각 항목의 실제 인용은 위 "출처"
열의 sha 핀 원문에서 해야 한다(RA4-1의 요약 스코프 소실 경고와 같은 규율).
