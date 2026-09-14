# 사전등록 — 907959 재실행 (role-worker grad guard 수리 + 대칭 메모리 계측) — **rev4**

**작성** 2026-09-13 (rev1) · **rev2** 2026-09-14 · **rev3** 2026-09-14 · **rev4** 2026-09-14 ·
**작성자** engine-porter · **GPU 신규 지출 0** (이 문서 작업은 CPU만)
**직전 회차** job 907959 = `VERDICT FAIL`
**선행 문서(전부 sha 고정, 아래 §0-c)** 사전등록 rev3 · 규칙층 판정서 2건 · 결과 감사 판정서 ·
엔진 진단서 · **908020 판정서**

> ★★**이 사전등록은 이미 job을 하나 소비했고 그것은 등록 밖 실행이었다.**
> **job 908020**(2026-09-14, gpu43, 9분 09초 = **0.15250 GPU-h**, 채점기 라벨 `PASS`)은
> **(Zamba2-2.7B, triton, ctx 4096)** 에서 돌았다 — 이 문서가 등록한
> **(NemotronH-Nano-9B-v2, flashinfer, ctx 16384)** 가 아니다. 원인은 **§9의 제출 명령이
> `R2C_MODEL`·`R2C_ATTN_BACKEND`·`R2C_CTX`를 빠뜨린 것**이고, 하네스 기본값이 Zamba2/triton
> 이라 그 명령은 **결정적으로** 틀린 기판을 낸다. **등록된 실험은 아직 실행되지 않았다**
> (등록 추정량 θ·ε·1.2750 MiB/token·P=38·6,245가 전부 NemotronH 상수라 그 기판에서 정의되지
> 않는다). 908020의 어떤 예보도 채점되지 않았고 채점해서도 안 된다(§3-b C1–C5 · A908-1).

> ★**이 회차는 성능 실험이 아니다.** 처치는 **정확성/대칭성 수리** 하나이며, 어떤 수치도
> "true-dual이 빨라졌다 / 메모리를 덜 쓴다"로 인용될 수 없다(§11 인용 금지).

**규칙층 판정서 이력**
| 판정서 | sha256 | 등급 |
|---|---|---|
| `VERDICT_rerun_rules_2026-09-14.md` (rev1 대상) | `71f3cfb7d8db6ce11ea3b24c5b4b3e894af0be1add0f693c40a7bc92f0226e65` | **`NO-GO`**(死因 `N2` ×2) |
| `VERDICT_rerun_rev2_2026-09-14.md` (rev2 대상) | — | ★**인용 불가**(A908-7). 표기된 `GO-with-caveats`는 **철회**됐고 올바른 등급은 **`NO-GO`(死因 `N3`)** 였다 — `:278`에서 §0-b를 실현하지 못하는 제출 명령을 전사하며 승인했다 |
| `audit_908020_2026-09-14/VERDICT.md` (908020 결과 + 운영 결함) | 아래 §0-c | Q1 `PLAUSIBLE(조건부)` · Q4 제출 차단 결함 + **감사 실패 2회** |
| `VERDICT_rerun_rev3_2026-09-14.md` (rev3 대상) | 아래 §0-c | **`NO-GO`**(死因 `N2` ×1 — C1 (9)의 참 분지 ∅). ★단 **"등록 명령 ⊨ 등록 튜플" 18/18 실현**이 확인돼 **rev2의 死因(N3)은 제거됐다** |

---

## 0-a. rev1 → rev2 변경표 (재감사용 색인)

rev1은 규칙층 감사에서 **`NO-GO`(死因 `N2` ×2)** 를 받았다. 두 死因 모두 **문서 텍스트**
문제이며, 감사는 **코드·하네스·테스트·manifest·라인 인용·예산·승계 목록은 고칠 것이
없다**고 독립 재검증 후 공시했다(판정서 §1 전 항목 일치, §4 반증 실패 8건). **rev2에서
코드/하네스/테스트/manifest는 단 한 바이트도 바뀌지 않았다** — §0의 sha 표는 rev1과 동일하다.

| # | 死因 / caveat | rev1 | rev2 | 절 |
|---|---|---|---|---|
| **死因 2** | F-a의 반증 분지에 판정 채널이 없어, H1을 반증하는 관측(907959 TD)이 등록 문안 직독으로 **`UNREALIZED`(측정 실패)** 로 강등된다 | "같은 배치에서 OOM하면 REFUTED / 그 배치가 안 생기면 UNREALIZED" | **F-a1/F-a2/F-a3로 분해.** 1차 판정은 **배치 크기와 무관**하게 "형성한 모든 배치를 OOM 없이 완주했는가"이며, 거짓이면 `UNREALIZED`가 아니라 **H1 미지지**. 검정력은 F-a2로 분리, 배치 동정은 F-a3(기록 전용)로 강등 | **§6-1** |
| **死因 1** | F-b의 `peak(boot)` 정의역이 ∅일 수 있고(907959 L2에서 실제로 ∅), ∅ 처분이 미등록 | `prefill_active_batch_size == 14` 위에만 정의, `UNREALIZED` 분지 없음 | **F-b1(1차, 기울기 축) + F-b2(2차, 10,125 축)** 로 분해. 둘 다 **∅ 처분을 명시 등록**. F-b1은 정의역 보장형(공통 서두 ordinal) | **§6-2** |
| ★**rev2 신규 발견** | 감사 R-2(b)가 제안한 "29개 토큰 수" 정의역에 **두 개의 잔여 자유 표면**이 남아 있었다 — (i) 29개 중 **3개가 boot마다 중복 출현**(`7`×2, `17`×8, `1468`×2)해 `peak(boot,T)`가 한 값이 아니고, (ii) **in-flight 표본화가 형성된 배치의 61–68 %만 관측**한다(§6-2-3 실측). 부수적으로 판정서가 든 `T=3241` 회귀 끝점은 **4 boot 어디서도 in-flight 관측되지 않았다** | — | ordinal 기반으로 재정의해 (i)을 **소거**, 정의역을 **in-flight가 아닌 epoch**로 잡아 배치 종료 후 표본까지 유효 표본으로 쓰고(그 편이 촘촘하나 907959로는 검증 불가), (ii)는 **측정된 기저율과 함께 희소/∅ 처분을 등록**. 감사의 29개 열거는 등록 상수로 **그대로 보존**(§6-2-2), 검정력은 **하한 2.88 GiB / 상한 4.04 GiB**로 양쪽 등록(RR-15) | **§6-2** |
| C1 | 계측 비용 "0.14 µs"가 계측기 비용인 것처럼 읽힌다 | "0.14 µs" | 조건을 명시하고 **"0.14 µs"를 인용 금지로 등재**(RR-8). ±3 % 결론은 유지 | **§2.3 · RR-8** |
| C2 | 리셋이 스케줄러 스레드 발행이라 TD에서만 in-flight decode를 자를 수 있다 | 미등재 | **잔존 비대칭 + 편향 방향(TD 과소 보고 = 대역 A 쪽) + 상한 156 MiB**를 등록, Δ 인용 시 병기 의무 | **§2.2 · RR-9** |
| C3 | `INSTR=0`이 warm-up의 JIT 예열을 줄여 배치 형성 타이밍을 흔든다 | 미등재 | 등재 + **"907959와 같은 조건에서 배치가 형성될 것"이라고 쓸 수 없음** | **§7 · RR-10** |
| C4 | §11.2 재수록 3건이 **절단**됐다(특히 N-8 말미 "상류 병목 단정 금지") | 절단본 | **sha 핀 원문에서 전사**(R-3 반영) + §11.2를 전사 원본으로 쓰지 말라는 지시 | **§11.2** |
| C5 | 감사자 변이 M-J(실현값→설정값 에코)가 13개 테스트를 전부 통과 | "게이트 #176을 테스트로 고정" 뉘앙스 | **그렇게 쓸 수 없음**을 등재(RR-11). 코드는 실현값을 읽고 있어 결함 아님 | **§2.1 · RR-11** |
| C6 | F-d의 두 절이 항등식(거짓이 될 수 없음) | 4절 모두 예보처럼 서술 | 항등식 2절을 **기록 항목**으로 강등, 거짓 가능한 2절만 예보로 남김 + 거짓의 두 원인 병기 | **§6-4 · RR-12** |
| C7 | 대역 B가 도달 가능 Δ의 약 62 %를 흡수 · 보조 "곡선"은 곡선이 아니고 표본율이 arm마다 1.45× 다르다 | 미등재 | 등재(RR-13), 보조 산출물의 arm 비교 금지 | **§6-2 · RR-13** |
| C8 | `PDMUX_WORKER_GRAD_GUARD=none` arm을 D조건으로 달지 않는다 | §10이 "별도 사전등록 필요"라고만 | 감사 판정(scored boot으로 넣으면 **설계상 무조건 FAIL**, 비채점 형태도 하네스 변경 필요, 실비 ≈0.025 GPU-h)을 등재 | **§10** |
| C9 | §10 지지 문장은 10,125 배치가 형성될 때만 쓸 수 있는데 대체 문장이 없다 | 단일 문장 | **형성/미형성 두 경우의 문장을 각각 등록** | **§10** |
| C10 | RR-1…RR-7의 승계 사슬이 다음 회차까지 강제되지 않는다 | 미등재 | 다음 회차 사전등록의 **의무 조항으로 등록**(RR-14) | **§11.3** |

**변경하지 않은 것**: §0(sha 표) · §1(처치) · §2.1/§2.2의 설계 · §3(판정 규칙 승계) ·
§4(manifest) · §5(CPU 회귀 수치) · §8(예산) · §9(제출 게이트) · §11.1(승계 33건 목록).

---

## 0-a2. rev2 → rev3 변경표 (재감사용 색인)

rev2는 **제출됐고 job 908020을 낳았으며 그 job은 등록 밖 기판에서 돌았다.** rev2의 규칙층
판정서는 **철회**됐다(A908-7: 올바른 등급은 `NO-GO`, 死因 `N3`). rev3는 **운영 결함을
문서와 하네스 양쪽에서** 닫는다.

| # | 결함 / 처방 | rev2 | rev3 | 절 |
|---|---|---|---|---|
| **E1 (필수)** | 스코프 불일치의 처분이 미등록 ⇒ "등록 실험이 돌았는가"가 **라벨을 본 뒤의 사람 재량** | 없음 | **C1–C5를 문자로 등록**: 기계적·결과무관 일치 술어 (1)–(10), 불일치 라벨 **`NO_VERDICT_SCOPE`**, **유한한** 운영오류 재실행 예산(총 1회, **908020이 소모**), 908020 은폐 금지, 라벨 인용 시 튜플 병기 | **§3-b(신설)** |
| **E2 (채택)** | 하네스에 등록 스코프와 대조하는 검사가 **0건** | 없음 | **fail-closed scope guard** — provenance 직후·첫 부팅 이전에 `R2C_EXPECT_{MODEL,BACKEND,CTX}` 삼중 대조, 미설정도 불일치도 **`exit 2`**. ★**대가**: harness sha 이동(§0-b 재등록) · §7-2 "처치 축 하나 유지"를 스스로 건드림(등재) | **§0·§0-b·§7-c(신설)** |
| **E3** | §9의 제출 명령이 §0-b를 실현하지 못한다(**死因**) | `sbatch <path>` 단독 | **세 기판 변수 + 세 EXPECT 변수**를 포함한 명령으로 교체. ★`R2C_INSTRUMENT`는 이 회차 등록값이 0이므로 **붙이지 않는다**(newpair §13의 `=1`은 907959용, 승계 금지) | **§9** |
| **E5 (필수, 등록만)** | `inference_mode` × `forward_native`(bare Parameter, `n_groups=8`) 상호작용은 **한 번도 실행된 적이 없다**. 908020은 Zamba2(fused `.data` 분기)라 이 위험에 **검정력 0** | 없음 | 등록 실행에서 inference-tensor `RuntimeError`가 나면 **게이트 실패가 아니라 `no_grad`로의 스코프 변경 사유**임을 **미리** 등록 | **§6-5(신설)** |
| ★**F-a2 재검토** | 문턱 6,245를 "측정된 상수, 재량 0"으로 등록 | 6,245 | ★**이것은 엔진 용량 상수가 아니라 도착 조합의 실현값**임이 확인됐다(작성자 독립 검증: 6,245는 **두 기판 모두에서** C층 prompt-token 4항 부분합 — 907959 `368+1491+2186+2200`, 908020 `59+1214+2447+2525`). 수리 이전 엔진이 같은 Zamba2 기판에서 **12,369**를 완주. ⇒ 절대 문턱을 **보조**로 강등하고 **arm-간 격차**를 1차 검정력 지표로 등록 | **§6-1** |
| 승계 | — | — | **A908-1 … A908-7**을 §11.1 승계 목록에 편입 | **§11.1** |
| 게이트 | — | — | **G-908020-1 … 5** 등재(전역 번호는 doc-steward) | **§13(신설)** |
| 장부 | 0.786 GPU-h | **0.93833**(등록 회차 0.78583 + **등록 밖 0.15250**, 분리 표기) | **§8** |

**rev3에서 코드(엔진)는 바뀌지 않았다** — `multiplexing_mixin.py`는 `e2a97b42…` 그대로이고
`engine_source_hash`도 `eba74cbd…` 그대로다. 움직인 것은 **하네스 한 개**(scope guard)와
문서다.

---

## 0-a3. rev3 → rev4 변경표 (재감사용 색인)

rev3는 규칙층 재감사에서 **`NO-GO`(死因 `N2` ×1)** 를 받았다. ★**rev2의 死因(N3)은 제거된
것으로 확인**됐고("등록 명령 ⊨ 등록 튜플" **18/18 실현**), E2 가드·(7)/(10) 정정·F-a2a 수치·
예산·테스트 62/62는 **전부 독립 재현에 성공**했다. **rev4는 코드·하네스·테스트·채점기·
manifest를 한 바이트도 건드리지 않는다** — §0의 sha 표는 rev3와 동일하다.

| # | 지적 | rev3 | rev4 | 절 |
|---|---|---|---|---|
| ★**死因** | C1 **(9)** `"src_dirty:" 다음 줄이 비어 있음`의 **참 분지가 ∅** — `git status --porcelain`은 깨끗하면 **0줄**을 내므로 그 다음 줄은 **언제나** `nvidia-smi -L` 출력이다. 직해하면 `P`가 라벨을 보기도 전에 거짓 ⇒ **C3가 재실행 예산을 0으로 못 박아 비가역** | 감사 문안 **글자 그대로 승계** | **`"GPU 0: "로 시작한다`로 교체.** 작성자 `cat -A` 직독 재검증: 907959·908020·907100 **참**, **907456(src dirty) 거짓** — 참 분지 비공집합 + 음성 대조 | **§3-b C1** |
| ★**자기 실패 공시** | rev3는 (7)·(10)은 원자료로 재검증해 정정하고 **(9)만 무검증 승계** — 게이트 #110의 교과서적 실패 | — | 그 사실을 §3-b에 명시 등재 | **§3-b C1** |
| **P2** | §1.2(b)의 *"엄한 지점은 정확히 둘 … 둘 다 밖에서만 발화"* 가 **실측으로 거짓** — 하드 에러 **최소 4종**, `Inference tensors do not track version counter.`는 **안에서도 발화**. `no_grad`는 **네 경로 전부 회피** | 2종 열거 | §1.2(b) 전면 정정(4종 표 + 실행 로그) · **E5 트리거를 열거 → 계열**(`RuntimeError` ∧ (`inference tensor`\|`InferenceMode`)) · 저울질을 4종 기준으로 재서술 | **§1.2(b) · §6-5 · RR-16** |
| **P3-a** | §6-2-3·RR-15의 *"ordinal 38 미관측"* 이 **거짓** | 유지 | ★**전면 정정.** `prefill_active_batch_size==4` 스냅샷 **L1 58·L2 58·TD1 54·TD2 54**, `stream_index 4` **100 %**(작성자 재현). ★**더 깊은 자기 오류도 함께 철회**: "61–68 % 관측률"은 **비영 최대 런** 집계의 산물이며 배치 관측률도 epoch 밀도도 아니다. 브래킷(2.88/4.04 GiB)은 **유지**, 근거를 **"`P`가 데이터 의존"** 으로 교체 | **§6-2-3 · RR-15** |
| **P3-b** | RRC-1…13이 rev3에 **0건 승계**(본문 "RRC" 0회) | 0건 | **13건 전부 §11.1에 부활 편입** + 미반영 생존분 실체 반영: RRC-4(§6-5) · RRC-6·RRC-12(§6-2-3) · RRC-7(대역 동률 규약) · RRC-8(§10 조건화 + 출력공간 결손) · RRC-9(n=2 산포) · RRC-10(F-a2 거짓 분지 부재) · RRC-11(발행률 스냅샷 기준 통일) · RRC-13(→ RR-19) | **§11.1 · 각 절** |
| **P3-c** | RA3-1…12 미편입 | — | §11.1 편입 + RA3-1(검정력 공시)을 §6-5에 실체 등재 | **§11.1 · §6-5** |
| **P4** | F-a2a가 "1차 검정력 지표"인데 **라벨을 못 움직인다** | 예보 | **기록 항목으로 재분류**(수치·문턱·비특이성 보존) | **§6-1 · RR-17** |
| **P5** | §7-c "GPU 작업 전"이 부정확(가드는 `nvidia-smi` **이후**) · §13 G-908020-3의 "계열을 닫았다"가 과장 | — | **"첫 부팅(서빙) 이전"** 으로 정정 · **"3축을 닫았다 + 나머지 6축은 기본값 일치에 의존"** 으로 정정 + 가드 보증 범위·(11) 비독립성 등재 | **§7-c · §13** |
| **P6** | 현 HEAD `a9cd8dd`의 sbatch는 **rev2 본문**이고 scope guard는 **미커밋** ⇒ `commit=`이 실행 하네스를 못 가리킨다 | — | §9-3에 **제출 전 필수 조건**으로 명시(커밋은 메인 세션) | **§9** |

**rev4에서 바뀌지 않은 것**: 엔진(`e2a97b42…`) · `engine_source_hash`(`eba74cbd…`) ·
하네스(`f39b167b…`) · 채점기(`ec355e17…`) · 클라이언트(`95e10b49…`) · manifest 25항목 ·
테스트 62+25 · 예산 · §0-b 스코프 튜플 · 제출 명령.

---

## 0. 무엇이 바뀌고 무엇이 안 바뀌는가 (한 눈에)

| 축 | 907959 | 이 회차 | 성격 |
|---|---|---|---|
| model / backend / ctx / mem-fraction / max-running / split D44 / cudagraph ON / seed | 동일 | **동일** | 불변 |
| verdict rule v2 · 채점기 sha | `ec355e17…` | **`ec355e17…` (무수정 승계)** | 불변 |
| 클라이언트 sha | `95e10b49…` | **`95e10b49…`** | 불변 |
| **`engine_source_hash`** | `5b20b11d17a17e1b…` | **`eba74cbd2cebfbd0…`** | ★이동 |
| **`multiplexing_mixin.py` sha256** | `0fe9d570f151698c…` | **`e2a97b423f93ff6d…`** | ★이동 |
| **하네스 `r2_correctness.sbatch` sha256** | `bd40ad91b46a5f32…` | ★**rev3: `f39b167bb6bd713a…`** (전체: `f39b167bb6bd713af168abfbe3495caee099a6c0087e41b05e70b9a8d8ac0403`). rev2가 등록했던 `655382632e4804b3…`는 **job 908020이 실제로 돈 값**이며 더 이상 등록값이 아니다 — rev3의 scope guard(§7-c)가 harness sha를 옮겼다 | ★이동(2회) |
| `R2C_INSTRUMENT` | 1 | **0** (§7) | ★이동 |
| `PDMUX_MEM_TELEMETRY` | (존재하지 않음) | **1, 양 arm 동일** | ★신설 |
| runtime manifest 항목 수 | 25 | **25**(24개 해시 불변, 1개 이동) | 구조 불변 |

★**따라서 이 job은 907100 / 907456 / X1과 "같은 엔진"이 아니다.** 그 job들의 결론은
**(Zamba2-2.7B, triton, `engine_source_hash` 구판)** 한정으로 **그대로 동결**되며, 이 회차는
그것들을 되살리지도 확장하지도 무효화하지도 않는다. 역으로 이 회차의 결과도 그 쌍에
이식되지 않는다(NP-4 승계).

### 0-b. 스코프 튜플 (이 문서의 모든 문장에 붙는다)

> **(model `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base` = `NemotronHForCausalLM`, 56층 ·
> backend `flashinfer` 휠 **0.6.10 이라고 보고한 설치본** · ctx **16384** ·
> `mem-fraction-static 0.82` · `max-running-requests 48` ·
> `--disable-radix-cache --chunked-prefill-size -1 --disable-overlap-schedule --random-seed 1` ·
> split **fixed D44** = green stream index **4** · **cudagraph ON** ·
> **verdict rule v2 (2026-09-11), 채점기 sha `ec355e17…`** ·
> **`engine_source_hash` `eba74cbd2cebfbd0ac112fc4693cac516dacf95ce4250d899e96572e7fafe41a`** ·
> **harness sha `f39b167bb6bd713a…`**(rev3 scope guard 포함) ·
> **`R2C_EXPECT_MODEL`/`R2C_EXPECT_BACKEND`/`R2C_EXPECT_CTX` = 위 세 값과 일치** ·
> `R2C_INSTRUMENT=0` · `PDMUX_MEM_TELEMETRY=1` ·
> `PDMUX_TRACE_FORCE_PREFILL=1` · `PDMUX_WORKER_GRAD_GUARD` **미설정(엔진 기본값
> `inference_mode`)** · A100-SXM4-80GB 108 SM 1장)**

### 0-c. 승계 문서 고정 (sha256)

| 문서 | sha256 |
|---|---|
| `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md` (rev3) | `e832fc49a867a1764422171a36d2984308eb4f34bbe1321cfb6d685cdbed54ec` |
| `newpair_prereg/VERDICT_newpair_rules_2026-09-13.md` (1차 규칙층) | `45ee250d2fa0cebfdc32592772213d1384a3177875e362f4bb5856848ed2b8ec` |
| `newpair_prereg/VERDICT_newpair_rev2_2026-09-13.md` (재감사) | `cd1ffe8b6befdf5376aab66981603299e4a6b8df78c6c3edb1547beb17d86b91` |
| `audit_907959_2026-09-13/VERDICT.md` (결과 감사) | `f544544fc973e0abc2d1aa97905e4e005d47e657c30e64d9ea7717d3a48ddc71` |
| `audit_907959_2026-09-13/DIAGNOSIS_true_dual_oom.md` (엔진 진단) | `8c857bede1576d9e8042d759889aec3a9f034ed6167fea4093dd947bb9565bee` |
| `job_907959/verdict.txt` (직전 라벨) | `640fd3adaa531b2dffcdac2ba76d9f1dfa98bed0cb30ffbd772da5c9df846529` |
| **`rerun_prereg/VERDICT_rerun_rules_2026-09-14.md`** (rev1 규칙층 판정서, **`NO-GO`**) | `71f3cfb7d8db6ce11ea3b24c5b4b3e894af0be1add0f693c40a7bc92f0226e65` |
| **`audit_908020_2026-09-14/VERDICT.md`** (908020 판정서 — C1–C5·A908-1…7·G-908020-1…5의 원문) | `dc19d1183564591a6394d08432f0c7414d33332de68e55bf403a757ddf8c12ea` |
| `job_908020/verdict.txt` (등록 밖 실행의 라벨 = `PASS`) | `971b00a6a2aed0a6280ebc12f1bbc5ae6ec9bea8bfdc0103ce4fb1c271ff14ec` |
| `job_908020/provenance.txt` (등록 밖 실행의 튜플) | `59b2be040b8db18cd356a35a3966c161b3538194025755bc288c0cda77e0e0be` |
| **`rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md`** (rev3 규칙층 판정서 — **`NO-GO`**, RA3-1…12의 원문) | `d8281866f3b06f841aa0eb3b00a6b6cf93cd9198d505a1ca2e1ca8ee5ed304c7` |
| **`rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md`** (등급 **철회**, 그러나 **RRC-1…13의 원문은 유효**) | `118bb45789baf117cc9a1460c8ebfb487088e0d86d81ca5ab9316afc5c412355` |

---

## 1. 처치 (단 하나)

### 1.1 무엇을 고쳤는가

`event_loop_pdmux`는 `@torch.inference_mode()`로 데코레이트돼 있고 **그 가드는
thread-local**이다. legacy는 `run_batch`를 그 데코레이트된 스레드에서 부르지만,
true-dual은 `RoleWorkerThread`에 넘긴다 — 데코레이터가 닿은 적 없는 스레드다. 결과:
**TD arm의 모델 forward만 autograd가 켜진 채 돌았다.**

수리: `multiplexing_mixin._activate_role_context`가 role worker task를 **가드 안에서**
실행한다. 한 줄짜리 변경이며 **CUDA device/stream 가드와 같은 `with` 문**에 들어간다.

```
with guard(), torch.cuda.device(self.gpu_id), torch.cuda.stream(context.stream):
```

★**옵션 B(`mixer2_rms_norm_gated.py:97`의 `.data` 수리)는 이번에 하지 않는다.** 두 처치를
같은 회차에 넣으면 어느 쪽이 들었는지 가를 수 없다. B는 **상류 관찰로만 등재**한다:
그 줄은 이 저장소의 sync 대상도 manifest 대상도 아니므로(§4), B를 언젠가 하려면 manifest
항목 추가가 선행돼야 한다.

### 1.2 `inference_mode` vs `no_grad` — 선택과 그 근거 (실행 전 고정)

**선택: `torch.inference_mode()`.**

**(a) 대칭성이 이 수리의 목적이다.** `inference_mode`를 쓰면 true-dual에서 inference 의미론
아래 도는 연산 집합이 **legacy가 이미 그렇게 도는 집합과 정확히 같다**(메인 루프는
데코레이터를 유지하고, 두 worker가 같은 가드에 들어간다). 즉 legacy에서 inference tensor가
아니던 텐서가 true-dual에서 inference tensor가 되는 일이 없다. `no_grad`를 쓰면 worker가
만든 텐서는 **version counter를 가진 보통 텐서**가 되어, legacy는 inference tensor를,
true-dual은 보통 텐서를 다루게 된다 — **correctness 게이트의 두 arm이 서로 다른 텐서
의미론 위에서 모델을 돌리게 된다.** arm 비대칭을 다른 arm 비대칭으로 바꾸는 것은 수리가
아니다.

**(b) `no_grad`가 피하는 위험이 무엇인가 (CPU 측정, torch 2.9.1) — ★rev4에서 전면 정정.**

> ★★**rev1–rev3가 여기 적었던 문장 — "엄한 지점은 정확히 둘이며 둘 다 소비자가 inference
> mode 밖에 있을 때만 발화한다" — 은 실측으로 반증됐다. 인용 금지(RR-16).**

설치본 `libtorch_cpu.so`를 직접 열거하고(작성자 재현) 각 경로를 실행한 결과:

| # | 하드 에러 | inference mode **안**에서 발화? |
|---|---|---|
| E-a | `Inplace update to inference tensor outside InferenceMode is not allowed.` | 아니오 (밖에서만) |
| E-b | `Inference tensors cannot be saved for backward.` | 아니오 (밖에서만) |
| E-c | ★**`Inference tensors do not track version counter.`** | ★**예 — 안에서도 발화한다** |
| E-d | `Cannot set version_counter for inference tensor` | (설정 경로) |
| (+) | `Expected this {function,method} to only be reached in inference mode and when all the inputs are inference tensors.` 등 내부 assert 2종 | — |

실행 확인(작성자):

```
inference-mode 텐서: ._version 읽기  INSIDE inf-mode -> RuntimeError: Inference tensors do not track version counter.
                     ._version 읽기  outside        -> RuntimeError: Inference tensors do not track version counter.
                     add_            outside        -> RuntimeError: Inplace update to inference tensor outside InferenceMode …
                     (w*x).sum()     outside        -> RuntimeError: Inference tensors cannot be saved for backward. …
no_grad 텐서       : 위 네 경로 전부                -> OK
```

⇒ **두 귀결을 등록한다.**
1. **열거는 불완전했다.** 그러므로 §6-5의 E5 트리거를 **문자열 2개의 열거가 아니라 계열**로
   등록한다(아래 §6-5). 열거로 두면 **열거되지 않은 예외가 F-a1 채널 2/3을 거짓으로 만들어
   "H1이 지지되지 않았다"로 기록**된다 — 처치층 아티팩트가 가설 판정으로 둔갑하는 경로다.
2. ★**저울질의 대가가 문서가 적던 것보다 `inference_mode` 쪽에서 크다.** `no_grad` 결과
   텐서는 위 네 경로를 **전부 회피**한다. §1.2(a)의 대칭성 논거는 그대로 유효하지만,
   그 대칭성의 **가격이 "밖에서만 나는 두 에러"가 아니라 "안에서도 나는 것을 포함한 최소
   4종"**임을 등록한다. 그럼에도 `inference_mode`를 유지하는 이유는 **legacy 동치**다 —
   legacy는 이미 4 GPU job 동안 같은 연산 집합을 같은 가드 아래서 돌렸으므로, E-a…E-d 중
   무엇이든 이 모델에서 발화한다면 **legacy arm에서도 발화한다**(발화하면 그것은 arm 비교
   이전에 엔진 사실이다). 그 예측이 틀리는 분지가 바로 §6-5다.

**도달성(측정, 반전 미완성)**: `sglang/srt` 직접 소스에서 `._version` 참조 **0건**,
`requires_grad_(True)` **0건** ⇒ E-c의 도달 경로는 이 트리의 직접 코드에는 없다. 다만
**서드파티 커널(flashinfer)·torch 내부 경로는 열거하지 않았다** — 그래서 §6-5를 등록한다.

이 프로세스에서 worker 결과의 **모든 소비자가 inference mode 안**이다 — 코드로 확인한 것:

| 경로 | 확인 |
|---|---|
| `prefill_future.result()` / `decode_future.result()` | `event_loop_pdmux` 본문 안에서 읽힌다 ⇒ 데코레이터 안 |
| KV / mamba 풀, cudagraph static buffer | 부팅 시 할당된 **보통 텐서**. inference mode 안에서 보통 텐서를 in-place 갱신하는 것은 **허용**되고 텐서는 보통 텐서로 남는다(측정 확인). legacy가 이미 같은 일을 한다 |
| 보통 텐서의 view를 inference mode 안에서 만든 경우 | `is_inference() == False` — 밖에서 in-place 갱신 가능(측정 확인) |
| 텔레메트리 writer 스레드 | 파이썬 스칼라만 직렬화, 텐서 미접촉 |
| 두 worker가 동시에 가드 보유 | `InferenceMode`는 thread-local, 독립 — 측정 확인(배리어로 동시 보유 강제) |

**(c) 잔여 위험과 탈출구.** 위 분석이 어떤 미래 모델에서 틀릴 경우를 위해
`PDMUX_WORKER_GRAD_GUARD`(`inference_mode`|`no_grad`|`none`)를 둔다. **기본값이자 이 회차가
측정하는 값은 `inference_mode`이고, 하네스는 이 변수를 설정하지 않는다.** 다른 값으로
돌리는 것은 **새 스코프 튜플**이며 이 사전등록의 대상이 아니다. `none`은 수리 이전 거동을
재현하며 **회귀 테스트의 음성 대조 전용**이다.

### 1.3 수리가 소급 적용되지 않는 것 · 다만 소급되는 사실 하나

- **907100 / 907456 / X1의 결론은 이 수리로 바뀌지 않는다.** 그 job들은 Zamba2-2.7B
  (`mamba n_groups = 1`)라 `Mixer2RMSNormGated.forward_native`의 bare-Parameter 줄에
  **구조적으로 도달하지 않았다** ⇒ 이 retention 기전에 면역이었다.
- ★**그러나 "worker thread가 grad 켜진 채 돈다"는 성질 자체는 그 job들에도 있었다**
  (같은 코드). 토큰 값은 바뀌지 않지만 **autograd 부기(version counter·디스패치)의 arm
  비대칭 오버헤드**는 그 job들에 **존재했다** — 크기는 **미측정**이며, 이 회차도 그것을
  측정하지 않는다. (진단서 §9-1, claims-auditor 이관 항목.)

---

## 2. 계측 — `PDMUX_MEM_TELEMETRY=1` (양 arm 대칭)

### 2.1 무엇이 어디에 실리는가

`runtime_snapshot`의 **base payload**에 아래 9개 키가 추가된다. `runtime.metrics()`(true-dual
에만 존재)가 **아니라** base payload인 것이 핵심이다 — 그렇지 않으면 이 회차가 없애려는
바로 그 비대칭을, 그것을 재는 계측기가 다시 만든다.

| 키 | 내용 |
|---|---|
| `gpu_mem_allocated_b` | `allocated_bytes.all.current` — 순간 live 바이트 |
| `gpu_mem_peak_allocated_b` | `allocated_bytes.all.peak` — **마지막 리셋 이후** 피크 |
| `gpu_mem_reserved_b` | `reserved_bytes.all.current` |
| `gpu_mem_peak_epoch` | 리셋 창 번호(리셋마다 +1) |
| `worker_grad_guard` | 설정된 가드 이름(양 arm) |
| `{prefill,decode}_worker_grad_enabled` | ★**실현값** — worker 스레드가 가드 안에서 읽은 `torch.is_grad_enabled()` |
| `{prefill,decode}_worker_inference_mode` | ★**실현값** — 같은 위치의 `torch.is_inference_mode_enabled()` |

**대칭 보장**: 9개 키 전부 legacy 기록에도 **같은 주기로** 실린다. legacy는 worker 스레드가
없으므로 4개 realized 키가 `null`로 남는데, **그 `null`이 곧 기록해야 할 사실**이다(값을
0/False로 날조하지 않는다). CUDA 미초기화 환경에서도 3개 메모리 키는 **존재하되 `null`**
이다 — "측정된 0"으로 오독될 값을 만들지 않는다(테스트로 고정).

### 2.2 ★리셋 지점 — 등록 (이 선택이 측정 대상을 바꾼다)

> **`torch.cuda.reset_peak_memory_stats()`는 split-prefill 배치 시작(`_dual_worker_start_prefill`)
> 에서 정확히 한 번 호출한다. 즉 `gpu_mem_peak_allocated_b`는 "이 prefill 배치가 시작된
> 뒤의 device 할당 피크"다.**

- **왜 이 지점인가**: 리셋하지 않으면 `max_memory_allocated`는 프로세스 시작부터 단조라
  첫 큰 배치 이후 모든 스냅샷이 같은 수를 보고하고 **층별 곡선이 사라진다**. 배치 시작에서
  리셋하면 한 배치 안에서 값이 **단조 비감소**이므로 **그 배치의 마지막 표본이 곧 그 배치의
  피크**이고, 표본을 하나 놓쳐도 피크가 숨지 않는다.
- **왜 "스냅샷마다 리셋"이 아닌가**: 더 잘게 보이지만, 두 표본 사이에 뜬 피크가 **엉뚱한
  창에 귀속**되고 배치 내 단조성이 깨져 곡선이 읽히지 않는다.
- **왜 안전한가(코드 확인)**: `grep -rn 'max_memory_allocated|reset_peak_memory_stats|
  max_memory_reserved' sglang/srt/` **결과 0건**. 저장소의 히트는 전부 `multimodal_gen/`,
  `_mps_stub.py`, 벤치마크, test_utils이며 이 서버는 그중 무엇도 로드하지 않는다.
- **대칭**: 호출 지점은 `update_split_prefill_batch`의 무조건 경로에 있어 **양 arm이 같은
  사건에서** 리셋한다. 읽기와 **같은 플래그**로 게이트되므로 플래그 없는 실행은 무접촉.
- ★**(C2) 그럼에도 남는 arm 비대칭 — 등록한다.** 리셋은 **스케줄러 스레드에서 발행**되므로
  true-dual에서는 **다른 스레드의 decode forward 도중에 떨어질 수 있고**, legacy에서는
  (직렬화돼 있어) 그럴 수 없다. 즉 TD 쪽 피크 창만 in-flight decode의 일부를 잃는다.
  **편향 방향 = TD를 과소 보고하는 쪽 = 대역 A("F-b 충족") 쪽**이며, 크기 상한은 decode
  transient(cudagraph private pool **156 MiB** = ε의 15.6 %) 이하다. ★`Δ`(또는 `S`)를
  인용할 때 **이 편향 방향을 반드시 병기**한다(RR-9). "대칭 계측"은 **필드 위치와 발행
  스레드 양쪽**을 봐야 하며, 이 계측기는 전자만 만족한다.
- **리셋 안 하는 것**: `gpu_mem_allocated_b`(순간값)는 리셋이 필요 없고 그것만으로도
  retention 질문에 답한다. 그래서 피크 **대신**이 아니라 **함께** 싣는다.

### 2.3 ★관측자 효과 — 호출 빈도·비용·처분 (실행 전 고정)

- **빈도 = 기존 텔레메트리 주기와 동일.** 메모리 읽기는 **emit되는 `runtime_snapshot`마다
  1회**이며 새 emit을 만들지 않는다. 실측 빈도(job 907959): L1 = 12,458 snapshot / 56.0 s =
  **222.5/s**, TD1 = 7,554 / 49.3 s = **153.2/s** (이 하네스는
  `PDMUX_TRACE_FORCE_PREFILL=1`이라 prefill-in-flight sync마다 강제 emit한다).
- **★기각한 구현과 그 이유(측정치)**: `torch.cuda.memory_allocated()` /
  `max_memory_allocated()` / `memory_reserved()`는 각각 `memory_stats()`를 부르고, 그것이
  allocator의 중첩 stat dict를 **122개 항목으로 평탄화 + 정렬**한다. 이 venv에서 측정한
  순수 파이썬 비용은 **호출당 54.3 µs** ⇒ 3회면 스냅샷당 **163 µs** ⇒ 222.5/s에서
  **벽시계의 약 3.6%** 가 스케줄러 스레드에 추가된다. **그 철자만으로 ±3% 관측자 효과
  예산을 넘긴다.**
- **채택한 구현**: `torch.cuda.memory_stats_as_nested_dict()` **1회** + 중첩 dict 3키 읽기.
  ★**(C1 정정) 조건을 정확히 적는다 — 이 두 수는 서로 다른 입력에서 측정됐다.**
  `54.3 µs`는 **populated dict에 대한 flatten+sort**(감사자 재현 52.19 µs, leaf 122개)이고,
  `0.146 µs`는 **이미 만들어진 nested dict에서 3키를 읽는 비용**이다. 따라서
  **"계측기 비용이 0.14 µs"라고 쓸 수 없다**(RR-8). 남는
  `torch._C._cuda_memoryStats` C++ 호출 1회는 **여전히 미측정**이며, 동등한 nested dict를
  파이썬으로 구성하는 프록시(**4.18 µs** = 222.5 snap/s에서 벽시계 **0.09 %**)로만 상한이
  잡힌다. **±3 % 예산 결론은 그 상한 하에서 살아남는다**(0.09 % ≪ 3 %).
- **동기화 없음**: `memory_stats_as_nested_dict`는 allocator 락 아래 부기를 복사할 뿐
  device를 건드리지 않고, CUDA 초기화 전에는 예외 대신 `{}`를 돌려준다.
- ★**처분(등록)**: 이 계측기는 **paired on/off 측정이 아직 없다** ⇒ **P1 등 타이밍 수치를
  발표하는 어떤 실행에서도 켜서는 안 된다.** 이 회차는 correctness 게이트이고 타이밍 수치를
  발표하지 않으므로 켠다. P1 사용 전 paired on/off 측정이 **선행 조건**이다.
- **기본값 OFF**: 엔진 기본은 OFF이며, 플래그가 꺼진 실행의 기록은 수리 이전과
  **키 집합이 동일**하다(검증: pre-repair 모듈 대 post-repair 모듈로 같은 루프를 돌려
  45개 키 중 **43개가 값까지 동일**, 다른 2개는 `timestamp_monotonic_s`/`timestamp_s`뿐).

### 2.4 판정 규칙 무영향 (검증)

`r2_correctness_check.py`의 `SNAP_KEYS` 12개와 **충돌하는 새 키가 없고**, 플래그 ON에서도
`SNAP_KEYS` 전부가 생산된다 — 둘 다 테스트로 고정(§5). 채점기는 **한 글자도 고치지
않는다**(sha `ec355e17…`).

---

## 3. 판정 규칙 — verdict rule v2 무수정 승계

- 채점기 `r2_correctness_check.py` sha256 = **`ec355e171a66d68eab1300edcc7616a693cd55aa2c3fb5c251ee4b42eac50d30`**,
  변경 없음. 규칙 원문은 job 디렉터리의 `verdict_rule.txt`로 다시 핀된다.
- **라벨 해석·출력공간 전수·처분표**: `PREREG_NEWPAIR_2026-09-13.md` §8을 **무수정 승계**
  (FAIL / `NO_VERDICT_INFRA` / `INCONCLUSIVE` / `NO_VERDICT_UNREALIZED` / PASS의 뜻과
  각 라벨에서 쓸 수 있는 문장). **(D3) TD-TD S 불일치 = FAIL** 및 **(D2′) 귀무대조가 공허한
  S FAIL → `NO_VERDICT_INFRA` 보고** 조항도 문자 그대로 승계하며, D2′가 발화하면 **NPC-A의
  전사 의무가 함께 발화**한다.
- **§9 실패 모드 표 (a)–(g) 무수정 승계.** 단 (b)의 D12 분기는 이 회차에서 의미가 바뀐다:
  `R2C_INSTRUMENT`가 이미 **0**이므로 "INSTR=0으로 1회 재제출해 warm-up 잔류를 가른다"는
  진단 수단은 **이미 소진돼 있다**. 첫 scored boot이 부팅 중 OOM이면 그것은 mem 0.80
  재등록 사유가 아니라 **새 사전등록 사유**로 기록한다.
- **(D5) 재제출 정책 무수정 승계**: 재제출은 **`NO_VERDICT_INFRA`에 한해 최대 1회**.
  **`FAIL` · `INCONCLUSIVE` · `NO_VERDICT_UNREALIZED`는 재제출로 덮어쓰지 않는다.**
  재제출하면 두 job의 라벨을 **둘 다** 보고한다.

---

## 3-b. ★스코프 일치 게이트 — C1 … C5 (rev3 신설, E1)

job 908020은 **라벨(`PASS`)을 본 뒤에** "그건 등록 실험이 아니었다"고 선언할 수 있는 상태를
만들었다. 그 선언은 실체적으로 옳지만(등록 추정량이 그 기판에서 정의되지 않는다) **절차적
으로는 라벨 쇼핑과 구별 불가**다. 아래 다섯 조항을 **제출 이전에** 등록함으로써 그 구별을
복원한다. 원문은 `audit_908020_2026-09-14/VERDICT.md` §2.3(sha `dc19d118…`)이다.

### C1 — 스코프 일치 술어 `P` (기계적 · 결과무관 · 채점 **이전**에 평가)

새 job의 `provenance.txt` 와 `runtime_source_manifest.sha256` **만으로** 판정한다. 채점기
출력·토큰·라벨은 일절 보지 않는다.

```
(1)  provenance.txt: "model=nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base "
(2)  provenance.txt: "attention_backend=flashinfer"
(3)  provenance.txt: "context_length=16384 (R2C_CTX)"
(4)  provenance.txt: "dsm=44 seed=1 order=[L TD L TD]"
(5)  provenance.txt: "mem_telemetry=1"
(6)  provenance.txt: "instrument=0"  AND  "trace_force_prefill=1"
(7)  harness sha 3행 = ec355e17…c50d30 / 95e10b49…7ec367 / f39b167bb6…ac0403
(8)  runtime_source_manifest.sha256 = 25행 AND
     e2a97b423f93ff6d09f08b1e649596dad010285775845261ed27233c0346f243 포함
(9)  provenance.txt에서 "src_dirty:" 줄 **바로 다음 줄이 "GPU 0: " 로 시작한다**
     (= `git status --porcelain -- workspace/engine-port/src` 출력이 0줄)
(10) provenance.txt: "env (SGLANG|R2C):" 블록에 R2C_MODEL / R2C_ATTN_BACKEND / R2C_CTX 가
     모두 보일 것
(11) provenance.txt: "scope_guard: OK"  (rev3 신설, §7-c)
```

`P` = (1) ∧ … ∧ (11).

- ★★**(9)는 rev4에서 교체됐다 — rev3의 문안은 참 분지가 ∅이었다.** rev3는
  감사 §2.3의 `"src_dirty:" 다음 줄이 비어 있음`을 **글자 그대로 승계**했는데,
  `git status --porcelain`은 **깨끗하면 0줄을 낸다** ⇒ `src_dirty:` 바로 다음 줄은 **언제나**
  그 다음 명령(`nvidia-smi -L`)의 출력이다. 작성자 `cat -A` 직독:
  907959 `:4` = `GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-4f60982e-…)$` ·
  908020 `:4` = `GPU 0: … (UUID: GPU-8ac542fe-…)$` · 907100 `:4` = `GPU 0: …$` ·
  **907456 `:4` = `M  workspace/engine-port/src/multiplex/controller.py$`**(src dirty).
  ⇒ 더러워도 깨끗해도 그 줄은 **비지 않는다**. 직해하면 `P`가 **라벨을 보기도 전에 거짓**이
  되고 C3가 재실행 예산을 0으로 못 박았으므로 **비가역**이었다(≈0.20 GPU-h 지출, 채점 예보
  0건). 교체 문안은 **같은 네 아티팩트에서 참/참/참/거짓**을 내므로 기계적·결과무관이면서
  **참 분지가 비어 있지 않고 음성 대조도 있다**.
  ★**이것은 게이트 #110의 자기 실패다**: rev3는 (7)과 (10)은 원자료로 재검증해 정정했으면서
  (9)는 검증 없이 승계했고, **재검증한 두 항목의 성공이 나머지의 신뢰를 위조했다.**
- ★**(7)은 rev3에서 값이 바뀌었다** — `655382632e…`(908020이 돈 값)가 아니라
  **`f39b167bb6…`**(scope guard 포함)이다. 판정서 §2.3이 적은 `655382632e…`를 그대로 쓰면
  rev3 하네스가 불일치로 판정된다.
- ★**(10)은 새 출력을 요구하지 않는다 — 작성자가 확인했다.** 하네스의 덤프는
  `env | grep -E '^(SGLANG|R2C)_' | sort`(`r2_correctness.sbatch`)이므로 `R2C_*`를 **포함
  한다**. 908020에서 `(none)`이었던 것은 **그 job이 R2C_* 를 하나도 설정하지 않았기 때문**
  이다. 대조: job 907959의 같은 블록은
  `R2C_ATTN_BACKEND=flashinfer / R2C_CTX=16384 / R2C_DSM=44 / R2C_INSTRUMENT=1 /
  R2C_MODEL=nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base` 다섯 줄을 실제로 담고 있다.
  ⇒ **(10)을 빼지 않고 유지한다.** (이 회차는 `R2C_INSTRUMENT`를 설정하지 않으므로 그 줄은
  없을 것이며, `instrument=0`은 (6)이 본다.)
- **(11)은 rev3가 추가**한다. (1)(2)(3)이 결과를, (11)이 그 결과를 낳은 **검사의 발화**를
  본다 — 두 채널이 독립이므로 하나가 조용히 죽어도 다른 하나가 잡는다.

### C2 — 불일치 시 처분

> `P`가 거짓이면 그 job의 라벨은 **`NO_VERDICT_SCOPE`(등록 밖 실행)** 이며 **게이트 실패가
> 아니다**(교훈 21 / 게이트 #21). 그 job의 **채점기 라벨과 전 수치를 결과 문서에 반드시
> 병기**하고, **D5의 `NO_VERDICT_INFRA` 1회 예산을 소모하지 않는다.**
> `P`가 거짓인 job에 대해 **F-a1·F-a2·F-a3·F-b1·F-b2를 채점하지 않는다** — 채점 결과를
> 기록에 남기는 것 자체가 다음 회차의 인용 통로가 된다.

### C3 — 운영 오류 재실행 예산 (★유한)

> `NO_VERDICT_SCOPE`로 인한 재실행은 **이 사전등록 전체에 대해 최대 1회**이며
> **job 908020이 그 1회를 이미 소모했다.** 두 번째 스코프 불일치가 발생하면 **어떤 추가
> GPU도 쓰기 전에** 원인을 문서·하네스 양쪽에서 닫아야 하며, rev3의 §7-c scope guard가
> 이미 그 수리다. ⇒ **rev3 제출에서 다시 불일치가 나면 그것은 scope guard 자체의 결함**
> 이므로 GPU 재실행이 아니라 CPU 진단으로 간다.

### C4 — 908020 은폐 금지

> 새 job이 `PASS`든 `FAIL`이든, 결과 문서·정본·핸드오프는 908020의 **존재 · 라벨(`PASS`) ·
> 튜플(Zamba2-2.7B / triton / ctx 4096) · GPU 비용(0.15250 GPU-h)** 을 같은 문단에 적는다.
> ★**첫 문장이 "이 사전등록은 2개의 job을 소비했고 첫째는 등록 밖이었다"여야 한다.**

### C5 — 라벨 인용 시 튜플 병기 의무

> 오늘 이후 `r2_correctness` 라벨을 인용하는 **모든** 문장은 `(model, backend, ctx)` 세 항을
> **문장 안에** 병기한다. (908020의 `PASS`와 907959의 `FAIL`이 같은 문서에 튜플 없이 놓이면
> 독자는 "수리가 FAIL을 PASS로 바꿨다"로 읽게 되며, 그것은 §1.3·RR-3이 금지한 문장이다.)

---

## 4. 수리의 provenance — manifest에 잡히는가

`sync_engine_tree.sh`의 25항목 manifest는 `sglang/srt/multiplex/multiplexing_mixin.py`를
**3번째 항목**으로 해시한다. 실제 확인:

```
907959 : 0fe9d570f151698cb4120320bea8bea1111dff052e179085349b97bcbf1a1697  .../multiplexing_mixin.py
이 회차: e2a97b423f93ff6d09f08b1e649596dad010285775845261ed27233c0346f243  .../multiplexing_mixin.py
나머지 24항목: 전부 불변
```

⇒ 엔진 소스를 고쳤고 **해시가 움직였다**. provenance 구멍 없음. 파생으로
`engine_source_hash`(8개 모듈 내용 해시)도 `5b20b11d17a17e1b…` → `eba74cbd2cebfbd0…`로
이동한다 — 이 회차 전에 측정된 어떤 hybrid profile도 **fail-closed로 비호환** 처리된다
(설계된 거동이며, 이 게이트는 `PDMUX_R2_POLICY=fixed`라 profile을 쓰지 않는다).

★**이번 회차가 고치지 않은 파일**: `sglang/srt/layers/attention/mamba/mixer2_rms_norm_gated.py`
(옵션 B). 이 파일은 **manifest에 없다** — 언젠가 B를 하려면 manifest 항목 추가가
선행돼야 하며, 그것은 별도 회차다.

---

## 5. CPU 회귀 — 실행 전에 이미 통과한 것

★**전체 discovery 수치는 기준선이 움직인다** — 병행 워크스트림 B가 `test_lambda0_prereg`
계열을 rev4로 편집 중이라 같은 세션 안에서 **590 → 615 → 630**으로 바뀌었고 실패 항목도
바뀌었다(590회차: `TestMutationHarness.{test_harness_covers_all_three_modules, test_no_escapes}`
2건 + 간헐 `errors=1` / 630회차: `TestEveryVerdictIsReachable.test_rev3_rule_still_fails_here`
1건). **모든 실패가 `test_lambda0_prereg` 단일 모듈**이며 이 변경과 무관하다. 따라서
합격 기준은 **NPC-H 승계대로 본 트랙으로 한정**한다.

| 항목 | 결과 (2026-09-13→14 KST 측정) |
|---|---|
| ★**본 트랙** `test_r2_correctness_*` | **51 / 51 OK** |
| 신규 `test_worker_grad_guard.py` + `test_mem_telemetry_symmetry.py` | **25 / 25 OK** (13 + 12) |
| 엔진 인접 12개 모듈(`test_host_worker_metrics`, `test_dual_worker`, `test_trace_force_prefill`, `test_true_dual_prefill_ownership`, `test_r2_admission_{latch,persistence}`, `test_sticky_partition`, `test_profile_controller`, `test_controller_defaults`, `test_green_readout`, `test_probe_flush_durability`, `test_line_citations`) | **179 / 179 OK** |
| 전체 discovery | **630 tests, 실패 1** — `test_lambda0_prereg`(워크스트림 B). **이 변경이 만든 실패 0건** |
| line-citation 드리프트(`--check --all`) | **88 compared, 0 violation** (수리가 밀어낸 인용 17건을 도구의 내용-앵커 제안대로 이동시킨 뒤 재스냅샷; 지문 다중집합이 이동 전후 **동일** = 내용 무변경) |

**변이 시험(교훈 53)** — 같은 관측 함수를 git HEAD의 **수리 이전** mixin과 설치된 수리본에
각각 적용:

```
PRE-REPAIR  (git HEAD): grad_enabled=True  inference_mode=False graph_built=True  -> GATE FAILS
POST-REPAIR (installed): grad_enabled=False inference_mode=True  graph_built=False -> GATE PASSES
```

즉 §5의 게이트는 **수리에 의존한다**. 추가로 `DEFAULT_WORKER_GRAD_GUARD`를 `"none"`으로
바꾼 변이본도 게이트를 실패시킨다(테스트로 고정).

---

## 6. ★반증 가능한 사전 예보 (실행 전 고정, 사후 예보가 아님을 스스로 증명한다)

### 6-0. 이 예보가 사전 예보인 이유 — **수리 전 값이 이미 관측돼 있다**

job 907959에서 **이미 관측된 것**(재해석 불가, 원자료에 그대로 있음):

- TD1/TD2 **둘 다** split-prefill 시퀀스 `55 → 368 → 1468 → 3241 → 6245`를 **완주**한 뒤,
  다음 배치에서 `mixer2_rms_norm_gated.py:97`의 **198.00 MiB** 요청에 `torch.OutOfMemoryError`.
- L1은 같은 지점에서 **`#new-seq: 14, #new-token: 10125`** 배치를 **완주**(`srv_L1.log:2592`).
- `tel_L1.jsonl`에는 `prefill_active_batch_size == 14`인 스냅샷이 **146개**,
  `prefill_chunk_progress` ∈ {6,12,18,24,30,36,42,48,54,56}. `tel_TD1.jsonl`에는 **0개**.
- 진단서가 **실행 전에** 계산한 TD 예측 천장 ≈ **6.5k–7.7k 토큰**, 관측은 6245 통과 /
  10125 실패 ⇒ 구간 안. 같은 모형이 T=10125에서 예측한 초과 retention = **12.61 GiB**.

⇒ 아래 F-a/F-b/F-c는 **아직 존재하지 않는 데이터**에 대한 예보이고, 그 반대 분지가
**이미 관측된 상태**라서 사후적으로 고를 여지가 없다.

### 6-1. F-a — TD가 자기가 형성한 배치를 완주한다 (★rev2에서 재정의)

#### 왜 rev1을 버리는가 (死因 2)

rev1은 반증 분지를 **"같은 배치에서 죽었는가"** 위에 세웠는데, **그 배치를 동정하는 채널이
등록돼 있지 않았다.** `report_prefill_stats`는 prefill **완료 시** 호출되므로(rev1 §6-1이
스스로 적었다) **죽은 배치는 로그 줄을 남기지 않는다** ⇒ "형성됐지만 죽었다"와 "형성되지
않았다"를 등록된 채널로 구별할 수 없다. 결과 감사가 907959에서 그 구별을 해낸 방법
(C층 프롬프트 합 24,993 · TD 완주 11,377 · 잔여 13,616 = 10,125+3,491 · `#queue-req` 14→0 ·
`198.00 MiB ↔ T ∈ [10036,10137]` 역산)은 **다채널 법의학 재구성**이고 rev1에 한 줄도
채널로 등록돼 있지 않았다.

그 결과 rev1의 문안을 **907959 자신의 TD 원자료**에 먹이면: 10125 줄 **0건** ·
`==14` 스냅샷 **0개** · 완주 최대 **6,245 < 10,036** ⇒ 직독으로 **`UNREALIZED`(측정 실패,
H1의 실패가 아님)** 이 나온다. 즉 **이 프로젝트가 H1의 정본 증거로 삼는 관측**
(결과 감사 §3-4 = `CONFIRMED(scoped)`)을 측정 실패로 강등한다 — 결과 감사 §11이 명시적으로
막아 둔 방향("게이트 #21의 **역**도 규율이다 — 진짜 게이트 실패를 측정 실패로 강등하지
마라")의 재개방이다. rev2는 1차 판정을 **배치 동정에서 완전히 분리**한다.

#### F-a1 — 1차 판정 (배치 크기·배치 동정과 무관)

> **TD 2/2 boot이 자기가 형성한 split-prefill 배치를 하나도 빠짐없이
> `torch.OutOfMemoryError` 없이 완주했다.**
> **채널(넷 모두 AND, 전부 항상 정의된다)**:
> 1. `srv_TD1.log`·`srv_TD2.log`에 `torch.OutOfMemoryError` **0건**,
> 2. 같은 두 로그에 `Scheduler hit an exception` **0건**,
> 3. `BOOT_FAILURES.txt` **부재**(또는 TD 라벨 줄 0개),
> 4. `r2_correctness_report.json`의 `per_boot.TD1.checks.request_errors == 0` **AND**
>    `per_boot.TD2.checks.request_errors == 0`.
>
> ★**이 술어가 거짓이면 F-a는 충족되지 않으며, 그것은 `UNREALIZED`가 아니라
> "H1이 이 회차에서 지지되지 않았다"는 뜻이다.** 측정 실패로 강등하는 경로는 F-a1에
> **존재하지 않는다.**

- **F-a1 거짓일 때의 처분 (등록)**: (i) 라벨은 채점기가 내는 대로 보고한다(F-c는 라벨을
  예보하지 않는다, §6-3), (ii) `FAIL`은 **D5에 따라 재제출 금지** — 원인 분해는 새
  사전등록의 대상, (iii) **수리 자체는 되돌리지 않는다**(대칭성 수리로서 독립적으로 옳다),
  (iv) 이 회차가 수집한 메모리 축(F-b1)이 다음 사전등록의 설계 입력이 된다.
- **검증(실행 전 확인 완료)**: 907959의 TD 원자료를 이 술어에 먹이면
  `torch.OutOfMemoryError` 각 **1건** · `Scheduler hit an exception` 각 **1건** ·
  `BOOT_FAILURES.txt` = `SERVER_DIED_DURING_CLIENT boot=TD1 / boot=TD2` ·
  `request_errors` 각 **31** ⇒ **F-a1 거짓**. 즉 rev2의 1차 판정은 정본과 **같은 방향**을
  낸다(rev1은 반대 방향을 낼 수 있었다). legacy 2 boot은 네 채널 모두 0/부재 —
  대조도 성립한다.

#### F-a2 — 2차 판정 (검정력만; 1차를 덮지 않는다) — ★rev3에서 재근거

★★**rev2의 근거는 틀렸다.** rev2는 문턱 `6,245`를 "측정된 상수 ⇒ 재량 0"이라고 등록했지만,
**6,245는 엔진 용량 상수가 아니라 도착 조합의 실현값**이다. 작성자 독립 검증(두 기판의
`gen_L1.json` `phase_c` prompt_tokens 전수):

```
907959 (NemotronH, 32 프롬프트 합 24,993): 6245 = 368 + 1491 + 2186 + 2200
908020 (Zamba2,    32 프롬프트 합 28,053): 6245 =  59 + 1214 + 2447 + 2525
```

⇒ **서로 다른 토크나이저의 서로 다른 토큰 다중집합에서 각각 4항 부분합으로 실현된다.**
같은 값이 두 기판에 나타난 것은 엔진 상수라서가 아니라 **32개 프롬프트의 부분합이 조밀**
하기 때문이다. 실제로 `#new-seq`도 다르다(907959는 `8/6245`, 908020은 `7/6245`).

게다가 **수리 이전 엔진이 같은 Zamba2 기판에서 이미 12,369를 완주했다**
(`job_907100/srv_TD1.log:2619`, TD1 `13/12369`, OOM 0). Zamba2 3 job의 TD 완주 최대:

| job | 수리 | TD1 | TD2 |
|---|---|---|---|
| 907100 | 전 | **12,369** | 6,245 |
| 907456 (X1) | 전 | 6,123 | 6,245 |
| 908020 | 후 | 8,497 | 6,245 |

⇒ 절대 문턱 6,245는 **그 기판에서 수리 이전에 이미 넘겨졌다.** 이것이 감사가 F-a2를
"이 기판에서 검정력 0"이라 판정한 이유이며(G-908020-5), rev3는 그 진단을 받아들인다.

##### 등록 (rev3)

★★**rev4 재분류 (P4 / RA3-2)**: **F-a2a와 F-a2b는 둘 다 "기록 항목"이며 예보가 아니다.**
충족돼도 미충족돼도 **게이트 라벨도 H1 판정도 움직이지 않는다**(충족 → 아무 긍정 주장
불가, 미충족 → `UNREALIZED`). rev3는 F-a2a를 "1차 검정력 지표"라 불렀는데, 라벨을 못
움직이는 양을 예보로 두면 **결과 문서가 그것을 확증처럼 인용할 통로**가 생긴다. 아래
수치·문턱·비특이성 공시는 **전부 보존**하되 지위만 기록 항목으로 내린다.
★**(RRC-10) F-a2의 출력공간에는 "거짓"이 없다** — 넘으면 참, 못 넘으면 `UNREALIZED`,
F-a1이 거짓이면 미채점. 따라서 **F-a2 참을 H1의 추가 확증으로 인용할 수 없다.**

> **(F-a2a, 기록 항목 — arm 간 격차, 같은 job 안)**
> `gap = max{완주 #new-token : L1, L2} − max{완주 #new-token : TD1, TD2}`.
> 4 boot이 **같은 프롬프트 집합**을 받으므로 이 양은 도착 조합을 상당 부분 상쇄한다.
> **F-a2a 충족 = `gap ≤ 0.20 × max{완주 #new-token : L1, L2}`.**
> 측정된 대조(작성자가 4 job 전부 재계산):
>
> | job | 기판 · 수리 | legacy max | TD max | gap | gap 비율 |
> |---|---|---|---|---|---|
> | **907959** | **NemotronH** · **전**(기전 존재) | 12,516 | 6,245 | **6,271** | **+50.1 %** |
> | 907100 | Zamba2 · 전(기전 부재) | 7,542 | 12,369 | −4,827 | −64.0 % |
> | 907456 | Zamba2 · 전(기전 부재, X1 교란) | 9,216 | 6,245 | 2,971 | **+32.2 %** |
> | 908020 | Zamba2 · 후(기전 부재) | 7,542 | 8,497 | −955 | −12.7 % |
>
> ★★**정직한 공시 — 이 통계량은 특이적이지 않다.** 기전이 **구조적으로 부재**한 Zamba2
> 기판의 세 job에서 gap 비율은 **−64.0 % … +32.2 %** 로 흩어지고, **3개 중 1개(907456)가
> 문턱 20 %를 넘는다.** 즉 귀무(기전 없음) 분포가 문턱을 가로지른다.
> ★**작성자는 907456의 +32.2 %를 문턱 20 %를 적은 *뒤에* 계산했고, 그 결과를 보고 문턱을
> 옮기지 않았다** — 옮기면 그것은 데이터에 맞춘 사후 문턱이 된다(같은 세션이
> `λ0` 트랙에서 반복 지적한 死因 유형). 문턱은 **20 %로 고정**하고 비특이성을 등록한다.
> ⇒ **F-a2a는 어떤 긍정 주장도 지지할 수 없다.** 충족돼도 "수리가 효과가 있었다"가 아니며,
> 미충족이면 `UNREALIZED`(검정력 부족)일 뿐이다. **게이트 라벨과 H1 판정 어느 쪽도
> 움직이지 않는다.**
>
> **(F-a2b, 보조 — 절대 문턱)** TD 완주 최대 `#new-token` > **6,245**.
> ★**이 문턱은 도착 조합의 실현값이지 용량 상수가 아니다**(위 부분합). 기록만 하고
> 단독으로 어떤 판정도 내리지 않는다.
>
> **(처분)** F-a1이 참이고 F-a2a가 거짓이면 **F-a2만 `UNREALIZED`**(이 회차의 부하가
> 수리 전후를 가를 만큼 크지 않았다는 뜻이며 H1의 실패가 아니다). **F-a1이 거짓이면
> F-a2는 채점하지 않는다.**

★**남는 한계(공시)**: `gap`도 도착 조합에 완전히 면역은 아니다 — 907959에서 L1(10,125)과
L2(12,516)가 서로 다른 배치를 형성했다(§6-2-1). 그래서 F-a2는 **1차 판정이 아니라 검정력
지표**로만 남으며, 1차 판정 F-a1은 배치 크기에 전혀 의존하지 않는다.

#### F-a3 — 기록 전용 (판정에 들어가지 않는다)

> (i) `srv_TD*.log`에 `#new-seq: 14, #new-token: 10125` 줄이 출현했는가,
> (ii) `tel_TD*.jsonl`의 `prefill_active_batch_size == 14` 스냅샷 수와 그중
> `prefill_chunk_progress == 56`의 수,
> (iii) 크래시가 있었다면 `Tried to allocate N MiB`에서 역산한
> `T = N_bytes / (10240 × 2)` 와 그 구간,
> (iv) 각 boot의 완주 split-prefill 배치 열 `(#new-seq, #new-token)` 전체.
>
> **전부 보고 항목이며 어떤 판정에도 들어가지 않는다.** (iv)는 F-b1의 정의역을 구성하는
> 원자료이므로 결과 문서에 **반드시 전사**한다.

### 6-2. F-b — 메모리 축 (★rev2에서 재정의: 1차 = 기울기, 2차 = 10,125 축)

#### 6-2-1. 왜 rev1을 버리는가 (死因 1) — ∅ 정의역이 **이미 관측돼 있다**

rev1의 `peak(boot)`는 `prefill_active_batch_size == 14` 위에만 정의돼 있었고, **F-a에는 있던
`UNREALIZED` 분지가 F-b에는 없었다.** 그런데 907959 원자료에서 그 정의역은 **실제로
비어 있었다**(작성자 독립 재집계, 감사와 일치):

```
boot   ==14 스냅샷   완주 배치 열의 꼬리                     완주 최대 #new-token
L1        146        … 8/6245, 14/10125, 3/3491                    10,125
L2          0        … 9/7345, 16/12516          <- 14-seq 배치 없음  12,516
TD1         0        … 8/6245                                       6,245
TD2         0        … 8/6245                                       6,245
```

⇒ **같은 arm·같은 코드·같은 seed·같은 클라이언트의 두 legacy boot이 C층 배치를 다르게
형성했다.** `max{}` over ∅ 상황에서 채점자에게 남는 선택 3가지(빈 boot 제외 / 그 boot의
최대 배치로 대체 / `UNREALIZED`)를 rev1 문안이 **하나도 금지하지 않았고**, 앞의 둘은 Δ를
**0.236–0.413 GiB**(ε의 23.6–41.3 %) 움직여 **대역 B→A, D→A**를 뒤집는다.

#### 6-2-2. F-b1 — 1차 추정량: **공통 서두 ordinal 위의 peak–T 기울기** (정의역 보장형)

**정의역 구성 (재량 0, 순서대로 기계적으로 적용).**

1. **공통 서두 `P`** — 각 booted boot의 `srv_<label>.log`에서 정규식
   `Prefill batch, #new-seq: (\d+), #new-token: (\d+)`으로 완주 배치 열을 순서대로 뽑는다.
   `P` = 4 boot **전부**에서 `(#new-seq, #new-token)`이 **ordinal별로 동일**한 최대 서두 길이.
   **907959에서 `P = 38`**(작성자 전수 재계산; 39번째에서 L2가 `9/7345`로 갈라진다).
2. **epoch↔ordinal 대응** — `gpu_mem_peak_epoch`는 split-prefill 배치가 **형성될 때마다**
   1씩 증가하므로(§2.2), 어떤 boot에서도 죽은 배치가 없는 구간에서는 **epoch n ↔ n번째
   형성 배치**다. F-b1은 `n ≤ P` 구간만 쓰며, 그 구간은 모든 boot에서 완주가 확인된
   구간이다(구성 1이 로그 줄의 존재를 요구한다).
3. **2채널 일치 검사(필수)** — epoch `n`은 배치 `n`이 **형성된 순간부터 배치 `n+1`이
   형성될 때까지**의 구간이므로, 그 구간의 스냅샷은 (a) 배치 `n`이 in-flight인 것과
   (b) 배치 `n`이 끝난 뒤 idle인 것 둘 다를 포함한다. ★**(b)도 유효한 표본이다** — 피크는
   다음 리셋까지 유지되므로 idle 스냅샷이 오히려 그 배치의 **최종 피크**를 담는다.
   따라서 검사는 다음과 같다:
   > `V(b, n)` = { `prefill_active_batch_size` : `gpu_mem_peak_epoch == n` } 에서 **0을 제외**한 값 집합.
   > `V(b, n)` 이 **공집합이 아니면서** `{ #new-seq(ordinal n) }` 과 **다르면** ⇒ 그 `n`을
   > `D_b`에서 **제외**한다(epoch↔ordinal 대응이 깨진 증거).
   > `V(b, n)` 이 공집합이어도(= idle 표본만 있어도) **제외하지 않는다.**
   > `gpu_mem_peak_epoch == n` 스냅샷이 **아예 0개**면 제외한다(값이 없다).

   (죽은 배치 동정 채널이 없다는 死因 2의 교훈을 이 축에도 적용한다 — 대응을 **가정하지
   않고 검사**한다. 단 검사가 유효 표본을 버리지 않도록 정의역은 in-flight가 아니라
   **epoch**로 잡는다.)
4. **추정량** — boot `b`의 정의역 `D_b` 위에서
   `peak(b, n) = max{ gpu_mem_peak_allocated_b : gpu_mem_peak_epoch == n }`
   (in-flight·idle 스냅샷을 **모두** 포함),
   회귀변수 `T(n)` = ordinal `n`의 `#new-token`.
   `slope_b` = `peak(b, ·)` 대 `T(·)`의 **OLS 기울기**(MiB/token).
   **판정량** `S = | mean(slope_TD1, slope_TD2) − mean(slope_L1, slope_L2) |`.

**★감사가 제안한 "29개 토큰 수"와의 관계(등록 상수 보존 + 잔여 자유 표면 제거).**
판정서 §6-R-2(b)가 등록 상수로 열거한 29개 값

```
{1, 6, 7, 9, 17, 24, 34, 54, 55, 75, 183, 368, 371, 453, 732, 1113, 1255, 1280, 1409,
 1468, 1499, 1672, 1839, 1845, 2019, 2187, 2197, 2310, 3241}
```

은 **위 `P = 38` 서두의 서로 다른 토큰 수 집합과 정확히 일치한다**(작성자 독립 재계산).
그대로 보존한다. 다만 **토큰 값**으로 정의역을 잡으면 자유 표면이 남는다 — 그 29개 중
**`7`은 boot마다 2회, `17`은 8회, `1468`은 2회 출현**하므로 `peak(boot, T)`가 한 값이 아니고,
"첫 번째 / 마지막 / 최대 / 평균" 중 무엇을 쓸지가 미등록 재량이 된다. **ordinal로 잡으면
그 재량이 소거된다**(38개 ordinal이 29개 값의 38회 출현을 1:1로 지정한다). rev2는 그래서
ordinal을 쓴다. 907959의 `P = 38` 서두는 다음과 같다(결과 문서가 전사할 기준):

```
n :  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
T :  7   1   6  24   9   7  34  75 183 371 732 1113 1468 1845 2187 54 453 1255 17
n : 20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38
T :1280 17 1409 17 1499 17 1672 17 1839 17 2019 17 2197 17 2310 55 368 1468 3241
(#new-seq = 1 for n = 1..37, = 4 for n = 38)
```

#### 6-2-3. ★★rev4 전면 정정 — rev2/rev3가 여기 적은 "관측률" 논거는 **작성자 자신의 집계 오류**였다

> ★★**rev2·rev3의 §6-2-3 3·4번째 bullet과 RR-15의 근거 문장은 거짓이며 전사 금지다**
> (RRC-1 / RA3-4). 아래가 정정본이다.

**무엇이 틀렸는가.** rev2는 텔레메트리의 prefill-in-flight 구간을 "**비영 최대 런**"으로
세어 L1 27 / L2 27 / TD1 26 / TD2 24 "에피소드"를 얻고, 거기서 *"형성 배치의 61–68 %만
관측된다"* 와 *"ordinal 38(`4 seq / 3241 tok`)은 4 boot 어디서도 관측되지 않았다"* 를
유도했다. **집계가 틀렸다** — 비영 런을 **크기 변화로 쪼개지 않았기 때문에** 연속해서
형성된 배치들이 한 에피소드로 합쳐졌고, 그 에피소드는 `max()`로 라벨돼 작은 배치를 삼켰다.

**원자료 직독(작성자 재계산, 직접 술어)**:

```
prefill_active_batch_size == 4 인 스냅샷 수:  L1 58 · L2 58 · TD1 54 · TD2 54
그 스냅샷의 stream_index:                     4 (= D44) 100 % (4 boot 전부)
서버 로그의 `#new-seq: 4,` 줄:                boot당 정확히 1개 = ordinal 38
```

⇒ ★**ordinal 38은 4 boot 전부에서 관측됐고, 심지어 서두에서 D44 위에 100 % 놓이는 유일한
ordinal이다.** "미관측"은 데이터가 아니라 **집계 방식의 산물**이었다.
⇒ ★**"61–68 %"도 인용할 수 없다** — 그것은 최대-런 개수이지 배치 관측률도, F-b1이 쓰는
**epoch 정의역의 밀도**도 아니다.

**정정 후에도 남는 진짜 제약 (이것이 등록 사유다)**:

- **epoch 정의역의 밀도는 여전히 907959로 확인할 수 없다** — 그 job에는 `gpu_mem_peak_epoch`
  필드가 아예 없다. 그래서 §6-2-4 (i)(ii)의 **희소/∅ 처분을 실행 전에 등록**한다. 정의역이
  얇게 나오면 `UNREALIZED`이지 H1의 실패가 아니다.
- ★**`P` 자체가 데이터 의존이다** — `P`는 4 boot의 완주 배치 열이 ordinal별로 일치하는
  최대 서두이며, **첫 C층 배치가 갈리면 `P = 37`** 이 되어 `max T`가 3241 → **2310**으로
  떨어진다. **이것이 검정력 브래킷의 올바른 근거**이며, "ordinal 38 미관측"이 아니다.
- 관측 누락이 **짧은 배치에 편향**된다는 관찰 자체는 유지된다(긴 prefill일수록 sync를 많이
  거친다). 다만 그 편향의 크기는 **미측정**이고 arm마다 다를 수 있다(발행률 차 1.45×,
  아래 RRC-11 정정값).
- ★**(RRC-12) 서두 표본은 대부분 D44 밖이다** — 서두 ordinal 1..37은 `#new-seq`가 전부 1이고
  그 구간의 in-flight 표본은 대부분 **비분할 `stream_index 0`** 이다. 서두에서 idx 4에
  100 % 놓이는 ordinal은 **38 하나뿐**이다. §6-2-4(iv)의 "D44에서 쟀다고 쓸 수 없다"는
  약화가 아니라 **필수 조항**이다. 더해서 epoch 창은 배치 완료 후의 idle·decode 구간을
  포함하므로 `peak(b,n)`은 "배치 `n`의 피크"가 아니라 **"배치 `n` 시작부터 배치 `n+1` 시작
  까지의 피크"** 이며, **arm 간 decode 중첩 차이가 `S`에 섞이고 그 크기는 미측정**이다.
- ★**(RRC-6) epoch↔ordinal 2채널 검사의 검출력은 ordinal 38에만 있다** — 서두 1..37은
  `#new-seq`가 전부 1이므로 epoch 번호가 상수 `k`만큼 어긋나도 `V(b,n) = {1}`이라 검사를
  통과한다. 어긋남이 잡히는 자리는 **ordinal 38과 38−k 두 점뿐**이다.
  ⇒ ★**`|D_b|`가 38에 가까운데 ordinal 38이 제외됐다면 그것은 "잡음"이 아니라 정렬 붕괴의
  신호로 보고한다.**

#### 6-2-4. F-b1 판정 규칙 (실행 전 고정)

> **(i) 희소/∅ 처분** — 어떤 booted boot에서든 `|D_b| < 10` 이거나 `max{T(n) : n ∈ D_b} < 2000`
> 이면, 또는 booted boot이 4개 미만이면 ⇒ **F-b1은 `UNREALIZED`이며 라벨을 내지 않는다.**
> (907959 기저율: **in-flight** 기준 서두 관측 23–26 / 38 — epoch 기준은 더 촘촘할 것으로
> 예상되나 그 필드가 없어 **확인 불가**이다(§6-2-3). 두 조건 충족은 **보장이 아니다.**)
> **(ii) 도메인 중첩 요건** — `|D_TD1 ∩ D_TD2 ∩ D_L1 ∩ D_L2| < 8`이면 F-b1은 `UNREALIZED`
> (arm 간 레버리지가 달라 기울기 비교가 교락된다).
> **(iii) 판정** — 문턱 `θ = 0.3 × 1.2750 = 0.3825 MiB/token`.
>
> | 대역 | 조건 | 해석 (등록) |
> |---|---|---|
> | **A′** | `S ≤ θ` | **F-b1 충족.** 수리 후 두 arm의 peak–T 기울기가 구별되지 않는다 |
> | **B′** | `θ < S ≤ 0.6375` (= 예측 기울기의 50 %) | ★**H1 부분 반증** — 미확정으로 기록, 다음 회차 설계 입력 |
> | **C′** | `S > 0.6375` | ★**H1을 지배 원인으로 보는 해석은 `REFUTED`** |
> | **D′** | `mean(slope_TD) < mean(slope_L) − θ` | 예상 밖. **해석하지 않고 보고만 한다** |
>
> **(iv) 필수 병기** — `D_b` 전체(ordinal·T·`peak`·`stream_index`)와 `slope_b`·절편·R²,
> 그리고 각 점의 `stream_index`. ★**다수 점이 prefill-only(비분할 idx 0) 구간이므로
> 이 추정량은 D44 운영점의 값이 아니다**(결과 감사 N-7 계열). "D44에서 쟀다"고 쓸 수 없다.
> 서두에서 idx 4에 100 % 놓이는 ordinal은 **38 하나뿐**이다(§6-2-3, RRC-12).
>
> **(v) ★동률 규약 (rev4 신설 — RRC-7/RA3-10)** — 대역 정의가 겹친다:
> **`D′ ⇒ S > θ ⇒ B′ ∪ C′`** 이므로 `D′`는 **항상** `B′` 또는 `C′`에 포함된다.
> 겹치는 관측에서의 라벨 우선순위를 실행 전에 고정한다:
> ```
> 부호가 음(mean slope_TD < mean slope_L − θ)이면  →  D′ 를 1차 라벨로 보고하고,
>                                                     동시에 성립하는 B′/C′ 를 병기한다.
> 그 외에는 A′ → B′ → C′ 순으로 최초로 성립하는 것 하나.
> ```
> ★**음의 부호가 나오면 `D′`와 `B′/C′`를 둘 다 보고하며 한쪽만 인용할 수 없다.**
> 907959에는 `gpu_mem_*` 필드가 하나도 없으므로 이 규약의 도달 가능성은 **수치로 제시할
> 수 없다**(등록만 한다).

**θ가 판단이라는 공시**: `0.3825 MiB/token`은 **ε와 정확히 같은 성격의 사전 판단**이다.
rev2의 R-2는 **판단을 없애지 않고 ∅ 정의역을 없앤다** — 死因은 후자였다.

#### 6-2-5. F-b2 — 2차: 10,125 축 (rev1의 축을 ∅ 처분과 함께 유지)

> **(a) ∅ 처분 (신규 등록)** — 4 booted boot 중 **하나라도** `prefill_active_batch_size == 14`
> 스냅샷이 **0개**면 ⇒ **F-b2는 `UNREALIZED`이며 대역 라벨을 내지 않는다.** 빈 boot을
> 평균에서 제외하는 것도, 그 boot의 다른 배치로 대체하는 것도 **금지한다.**
> (907959에서 L2가 실제로 0개였다 — 이 분지는 가설이 아니라 관측된 사건이다.)
> **(b)** 4 boot 전부 ≥1개이면 rev1의 정의를 그대로 쓴다:
> `e*` = 그 boot에서 `==14` 스냅샷이 가장 많은 `gpu_mem_peak_epoch`,
> `peak(boot) = max{ gpu_mem_peak_allocated_b : gpu_mem_peak_epoch == e*, prefill_active_batch_size == 14 }`,
> `Δ = mean(peak(TD1), peak(TD2)) − mean(peak(L1), peak(L2))`, **ε = 1.00 GiB**.

| 대역 | 조건 | 해석 (등록) |
|---|---|---|
| **A** | `Δ ≤ +ε` | **F-b2 충족.** 수리 후 두 arm의 배치 피크가 구별되지 않는다 |
| **B** | `+ε < Δ ≤ +6.30 GiB` | ★**H1 부분 반증.** 미확정으로 기록, 다음 회차 설계 입력 |
| **C** | `Δ > +6.30 GiB` (= 예측 초과 12.61 GiB의 절반 이상) | ★**H1을 지배 원인으로 보는 해석은 `REFUTED`** |
| **D** | `Δ < −ε` | 예상 밖. **해석하지 않고 보고만 한다** |

★**동률 규약 (rev4 신설 — RRC-7/RA3-10)**: **`A`(`Δ ≤ +ε`)는 `D`(`Δ < −ε`)를 포함한다.**
겹치는 관측에서 `Δ < −ε` 이면 **`D`를 1차 라벨로 보고하고 `A`를 병기**하며, 한쪽만 인용할
수 없다. 그 외에는 `A → B → C` 순으로 최초 성립하는 것 하나.

★**(RRC-9) `Δ`·`S`는 arm당 n=2이고 산포 추정이 없다** — 두 판정량은 boot 2개의 평균이며
**boot 간 분산은 측정된 적이 없다**. ε·θ는 점추정 하나와 비교되는 **사전 판단**이다.
대역 B/B′·C/C′를 인용할 때 **"boot 간 산포 미측정"을 반드시 병기**한다. 이 회차는 방법론
게이트 3(n≥4)을 충족하지 않으며, **성능 주장이 아니므로 게이트 3의 대상도 아니다**(RR-1과
함께 인용).

**ε의 근거(판단임을 명시)**: 관측된 TD 전용 양성 성분 중 가장 큰 것이 cudagraph private
pool **156 MiB**이고, 14-seq / 21k full-token decode의 순간 성분은 GiB 단위가 아니다.
1.00 GiB는 그 최대 식별 성분의 약 6.4배이자 예측 초과분(12.61 GiB)의 약 8 %다.
★**수리 후의 양성 arm 간 피크 차이는 측정된 적이 없다** — ε는 측정이 아니라 사전 판단이며,
대역 B는 그래서 "실패"가 아니라 "미확정"이다.

**★대역 B의 흡수율(C7 등재)**: F-a1이 충족된 세계에서 도달 가능한 Δ의 상한은
`11.59 GiB − peak_L` ≈ **8.1–9.6 GiB**(진단서의 legacy working-set 2.0–3.5 GiB 기준)이며,
"미확정"으로 등록된 대역 B(1.00–6.30 GiB)가 그 구간의 **약 62 %**를 흡수한다. 즉
**F-a1 충족 조건부로 F-b2가 H1에 불리한 판정을 낼 여지는 좁다.** RR-5·RR-13과 함께 인용하라.

#### 6-2-6. 보조 산출물 (예보 아님, 기록 전용)

`gpu_mem_allocated_b` 대 `prefill_chunk_progress` 를 **양 arm 각각** 기록한다.
★**이것을 "곡선"이라 부르지 않는다**: 907959 L1의 `==14` 스냅샷 146개 중
progress 6·12·…·54는 **각각 정확히 2개**이고 나머지 **128개가 종단 progress 56**에 몰려 있다.
게다가 스냅샷 발행률이 arm마다 1.45× 다르다 — ★**rev4 정정(RRC-11)**: 이 두 수는 **줄 수** 기준이었다. `event == "runtime_snapshot"`만 세면 **L1 222.5/s · L2 234.7/s · TD1 153.2/s · TD2 153.1/s**이다(작성자 재계산: L1 12,511줄 / 12,458스냅샷, TD1 7,611 / 7,554). **비율 1.45×와 결론은 불변**이고 §2.3이 쓰는 스냅샷 기준과 이제 일치한다. 피크는 창 내
단조성 덕에 이 교락에 면역이지만 **순간값의 arm 비교는 표본율 교락을 안는다** ⇒
**arm 간 비교에 쓰지 않는다**(C7 / RR-13).

### 6-3. F-c — 게이트 위생

> `r2_correctness_report.json`의 `per_boot.<label>.checks.request_errors == 0` **4 boot 전부**
> (907959: L1/L2 = 0, **TD1/TD2 = 31**) · `verdict.txt`의 **S 티어 6쌍 전부 0 · O 티어
> within-arm 2쌍 + cross-arm 4쌍 전부 0** · **cudagraph ON 유지**(모든 boot의 모든
> `Decode batch` 줄이 `cuda graph: True`, 즉 `decode T/F=n/0`) · `BOOT_FAILURES.txt` 없음 ·
> 4 boot 전부 `booted=True crash_free=True`.

- 어긋나면: 채점기 라벨이 그대로 처분이다(§3). F-c는 **라벨을 예보하지 않는다** —
  §3.3(flashinfer 적응 split-KV) 때문에 PASS·INCONCLUSIVE 어느 쪽도 사전 확률을 주장하지
  않는다는 rev3의 입장을 그대로 승계한다.

### 6-4. F-d — 가드가 **실현**됐음을 텔레메트리가 보인다 (게이트 #176) — ★rev2에서 범위 축소

> **예보(거짓이 될 수 있는 부분만)** — TD boot의 스냅샷에
> `prefill_worker_grad_enabled == false` **그리고** `prefill_worker_inference_mode == true`가
> 나타나고, decode 쪽 두 키도 같다.

- 어긋나면: 두 원인이 가능하며 **구별하지 않고 측정 실패로 처리한다** — (i) 설치 트리가
  수리본이 아니다, (ii) `PDMUX_MEM_TELEMETRY` 플래그가 전파되지 않았다. 어느 쪽이든
  **`NO_VERDICT_INFRA`(측정 실패)** 로 보고하고 §3의 재제출 1회를 쓴다. 게이트 실패가 아니다.
- ★**기록 항목(예보 아님 — 항등식이라 거짓이 될 수 없다, C6)**:
  `worker_grad_guard == "inference_mode"`는 하네스가 `PDMUX_*`를 전부 unset하므로 **거짓이 될
  수 없고**, legacy의 realised 4키 `null`은 **worker 스레드가 없다는 것의 연역**이다.
  둘 다 처분이 붙지 않는 **기록**이며 **"F-d가 충족됐다"를 정보량 있는 확인으로 인용할 수
  없다**(RR-12).
- ★**게이트 #176 충족을 "테스트로 고정했다"고 쓸 수 없다**(C5): 감사자 변이 **M-J**
  (`_r2_record_worker_guard`의 `torch.is_grad_enabled()` / `torch.is_inference_mode_enabled()`를
  설정값 에코로 교체)가 **등록된 13개 테스트를 전부 통과**했다. 제출된 코드는 실현값을 읽고
  있으므로 결함은 아니지만, **실현값 성격은 파일 단위 manifest 해시로만 보호되고 변이 대조로는
  보호되지 않는다**(RR-11). (같은 주입에서 M-C[가드를 텔레메트리 플래그로 게이팅]·
  M-D[PREFILL만 가드]·M-K[가드 진입 후 yield 전 탈출]은 전부 잡혔다.)

---

### 6-5. ★F-e — 등록된 **위험**의 처분 (rev3 신설, E5) — 예보가 아니라 사전 처분

★★**`inference_mode` 가드와 `Mixer2RMSNormGated.forward_native`(bare Parameter,
`n_groups != 1`)의 상호작용은 GPU에서 한 번도 실행된 적이 없다.**

- 907032 / 907100 / 907456 / **908020**은 전부 Zamba2-2.7B(`mamba_ngroups = 1`)라
  `mixer2_rms_norm_gated.py:109-110`의 분기가 **fused 커널 경로(`:114`, `weight.data`)** 로
  가고, 그 경로는 bare Parameter를 autograd에 노출하지 않는다 ⇒ **이 위험에 검정력 0.**
- 907959는 NemotronH였지만 **수리 이전** 엔진이었다 ⇒ 가드가 없었다.
- ⇒ 등록 실행(E3)이 **이 조합의 최초 실행**이다. 현재의 안전 근거는 **§1.2(b)의 CPU 코드
  분석과 torch 2.9.1 CPU 측정뿐**이다(A908-5).

##### 처분 (실행 전 고정)

> ★**rev4에서 열거 → 계열로 교체한다.** 등록 실행의 어떤 boot에서든, 로그의 `RuntimeError`
> 메시지가 **`inference tensor`(대소문자 무시) 또는 `InferenceMode` 를 포함**하면 발화한다.
> ```
> 트리거 정규식:  RuntimeError.*(?i:inference tensor)|RuntimeError.*InferenceMode
> ```
> 이 계열은 §1.2(b)가 열거한 **E-a … E-d 네 종을 모두 덮고**, 열거하지 못한 내부 assert
> (`Expected this function to only be reached in inference mode …`)도 덮는다.
> ★**열거로 두면 안 되는 이유**: 열거되지 않은 예외가 나면 등록 문안대로는 **F-a1의 채널
> 2/3(`Scheduler hit an exception`·`request_errors`)이 거짓**이 되어 **"H1이 이 회차에서
> 지지되지 않았다"로 기록**된다 — 처치층 아티팩트가 가설 판정으로 둔갑한다. 계열로 두면
> 그 분지가 F-a1 이전에 가로채인다.
>
> 그것은 ★**게이트 실패가 아니라 `PDMUX_WORKER_GRAD_GUARD=no_grad` 로의 스코프 변경
> 사유**다. 구체적으로:
> 1. 그 job의 라벨은 채점기가 내는 대로 보고하되, **H1 판정에는 쓰지 않는다**
>    (`F-a1`·`F-b`를 채점하지 않는다 — C2와 같은 취급).
> 2. §1.2가 등록한 "`inference_mode` vs `no_grad`" 저울질에서 **`inference_mode` 쪽의
>    핵심 전제(모든 소비자가 inference mode 안에 있다)가 이 모델에서 거짓**임이 관측된
>    것이므로, `no_grad`로의 전환은 **사후 재량이 아니라 이 조항의 집행**이다.
> 3. 전환은 **새 사전등록**을 요구한다(가드 이름이 스코프 튜플의 축이고
>    `engine_source_hash`는 바뀌지 않지만 `worker_grad_guard` 축이 이동한다). 그 문서는
>    이 §6-5을 근거 조항으로 인용해야 한다.
> 4. ★**이 처분은 D5의 재제출 예산도, C3의 운영오류 예산도 소모하지 않는다** — 둘 다
>    다른 종류의 사건이다.
>
> ★**이 조항을 미리 등록하는 이유**: 등록하지 않으면 사후에 "가드를 바꿔서 다시 돌린다"가
> **또 하나의 재량 통로**가 된다(908020이 만든 통로와 같은 형태). 반대로 등록해 두면
> 그 전환은 **예측된 분지**가 된다.

##### 이 조항이 열어 주지 **않는** 것

> `no_grad`로 바꿔도 **§1.2(a)의 대칭성 논거는 회복되지 않는다** — worker가 만든 텐서는
> 보통 텐서가 되고 legacy는 inference tensor를 다루므로 두 arm이 서로 다른 텐서 의미론
> 위에서 돈다. 그 경우 이 트랙은 "대칭 수리"가 아니라 **"두 비대칭 중 덜 나쁜 쪽 선택"**
> 이 되며, 그 사실을 새 사전등록이 명시해야 한다.
> ★**rev4 정정**: 그 "덜 나쁨"의 **대가 쪽 크기는 rev3까지 과소 기술돼 있었다** —
> `no_grad` 텐서는 §1.2(b)가 열거한 **E-a … E-d 네 경로를 전부 회피**한다(측정). 즉
> `inference_mode`를 고른 대가는 "밖에서만 나는 두 에러"가 아니라 **안에서 나는 것을 포함한
> 최소 4종의 하드 에러 노출**이다. 이 저울질을 새 사전등록이 4종 기준으로 다시 적어야 한다.

##### ★이 조항이 **닫지 못하는** 것 (RA3-1 계열 — 검정력 공시)

> 이 회차는 **어느 결과가 나오든 H1을 닫지 못한다.** 항상 정의되는 반증 가능 채널은
> **F-a1 하나**인데, 그 채널은 (i) **TD 실패의 원인을 구별하지 못하고**(job 907032를 먹이면
> `torch.OutOfMemoryError` **0건**인데도 `Scheduler hit an exception` 1/1 ·
> `BOOT_FAILURES.txt`로 **F-a1 거짓** — 死因은 split-prefill ownership **경합**이지 H1과
> 무관, 이 트랙 4 job 중 **1건 = 25 %**), (ii) **TD가 큰 배치를 형성하지 못하면 공허하게
> 참**이 된다. 공허 참 검출용 F-a2는 §6-1이 스스로 "어떤 긍정 주장도 지지할 수 없다"고
> 등록했고, F-a3는 기록 전용, F-b1·F-b2는 둘 다 `UNREALIZED` 분지를 갖는다.
> ⇒ **이 회차가 최대로 산출할 수 있는 것은 §10의 스코프 한정 존재 문장 하나뿐이다**(RA3-1).
> ★**F-a1이 거짓일 때는 `srv_TD*.log`의 `torch.OutOfMemoryError` 계수와 크래시 스택을 반드시
> 병기**하고, OOM 0건이면 **"이 회차는 H1을 시험하지 못했다"** 로 서술한다(RRC-4).

---

## 7. `R2C_INSTRUMENT` — 끈다 (0), 근거

**결정: `R2C_INSTRUMENT=0`.**

1. **새 정보가 없다.** λ_inf(A)=3.0939는 907959에서 이미 얻었고, λ_inf(B)=0.6956은 결과
   감사가 **F5 셀 B 반증**으로 **인용 불가** 처리했다(N-8). 같은 레시피를 다시 돌리면
   **같은 결함을 재생산**할 뿐이다 — 그 결함의 수리(prefill 지배 shape의 포화 계기를
   `#running-req` 대신 `#queue-req`로 바꾸는 것)는 감사 §13-D2가 **별도의 새 사전등록**을
   요구한다.
2. **처치 축을 하나로 유지한다.** 이 회차의 처치는 grad guard 하나다. 계측 레시피까지 같이
   바꾸면 어느 쪽이 무엇을 움직였는지 가를 수 없다.
3. **예산·최악 경계가 줄어든다**(§8): read-out 3000 s timeout 분지가 사라져 구조적 최악이
   161분 → 약 111분이 되고 `--time=02:30:00` 안에 **들어온다**(907959 구성에서는 들어오지
   않았다).

★**따라서 감사 §13-D1(i)의 수리 — 하네스가 `I3_max_running_req.txt`를 셀별 2줄로 쓰게
하는 것 — 은 이 회차에 포함하지 않으며, 그 결함은 다음 회차로 남는다.** 이 문장을
등록해 두는 이유는, 나중에 "왜 D1(i)이 안 닫혔는가"가 회고적 재량으로 보이지 않게 하기
위해서다. (감사가 권장한 D1(ii) — 술어가 `I_log_offsets.txt` + `srv_warmup.log`에서 스스로
셀별 재계산 — 은 **GPU 0·소급 적용 가능**이고 λ0 워크스트림 소관이라 이 회차와 독립이다.)

### ★7-b. `INSTR=0`의 **미등재 부작용** — rev2에서 등록 (C3)

위 세 이유 외에 `R2C_INSTRUMENT` 1→0은 **warm-up boot의 클라이언트 부하(I2/I3)를
제거**한다. `r2_correctness.sbatch:175`의 `TRITON_CACHE_DIR="$OUT/.triton_cache"`는
**job마다 새로 생성**되므로, 907959에서는 warm-up이 예열해 둔 JIT 캐시가 이 회차에서는
**덜 채워진 채** 첫 scored boot이 돈다.

★**이것은 C층 배치 형성 타이밍을 흔들며, 그 배치 형성이 바로 F-a3(기록)과 F-b1의 정의역,
F-b2의 존재 조건이 의존하는 양이다.** 따라서:

> **"907959와 같은 조건에서 (같은) 배치가 형성될 것"이라고 쓸 수 없다**(RR-10).
> 배치 형성의 비결정성은 같은 arm 안에서도 관측됐다(§6-2-1: L1과 L2가 C층 배치를 다르게
> 형성). rev2의 1차 판정(F-a1·F-b1)이 **배치 동정에 의존하지 않도록** 설계된 이유가 이것이다.

### ★7-c. 하네스 scope guard — 도입과 **그 대가** (rev3 신설, E2)

`r2_correctness.sbatch`에 **provenance 덤프 직후·첫 부팅 이전** 블록
`# --- BEGIN r2c scope guard ---` … `--- END ---` 을 추가했다. 동작:

1. `R2C_EXPECT_MODEL` / `R2C_EXPECT_BACKEND` / `R2C_EXPECT_CTX` 중 **하나라도 미설정이면
   `exit 2`**,
2. **해결된**(resolved) `$MODEL` / `$ABE` / `$CTX`(ctx 유도 블록 이후 값)와 세 기대값을
   대조해 **하나라도 불일치면 `exit 2`**,
3. 어느 쪽이든 **첫 부팅(서빙) 이전에** 멈추고, 해결값·기대값·`OK`/`MISMATCH`를
   `provenance.txt`에 덧붙인다(C1 항목 (11)).

**왜 기본값을 주지 않고 "필수"로 했는가**: 기대값을 한 캠페인의 튜플로 기본화하면 공용
하네스 안에서 그 캠페인을 특권화하고, 사전등록 §0-b와 **조용히 드리프트**한다. 필수로
하면 모든 제출이 **자기가 무엇을 재고 있다고 믿는지 진술**하게 되므로,
`R2C_MODEL=`을 빠뜨리면 불일치로, 기대값을 빠뜨리면 미설정으로 **양쪽 다 잡힌다.**
저장소 안의 이 파일 참조 3건은 전부 **주석**이므로 자동화가 깨지지 않음을 확인했다.

★**(rev4, RA3-7) 가드가 보증하는 범위 — 과장 금지.** 가드는 **model/backend/ctx 3축만**
검사한다. 나머지 6축(`R2C_{DSM,SEED,ORDER,TRACE_FORCE_PREFILL,INSTRUMENT,MEM_TELEMETRY}`)이
무해한 실질 이유는 **그 기본값이 등록값과 같기 때문**(6/6 일치)이며, 그 우연이 깨지면 같은
실패 모드가 재발하고 **C3 예산은 0이다.** 또한 **일관 오답쌍**(`R2C_MODEL=X` 와
`R2C_EXPECT_MODEL=X` 를 둘 다 틀리게 주는 경우)은 가드를 통과한다 — 설계상 수용하되
(그 경우 C1 (1)(2)(3)이 **채점 이전에** 잡는다), ★**"하네스가 스코프를 보증한다"고는 쓸 수
없다.**
★**(rev4, RA3-8) C1 (11)은 (7)과 독립 채널이 아니다** — 가드가 fail-closed이므로 **부팅한
모든 job은 자동으로 `scope_guard: OK`** 를 갖는다. (11)의 고유 정보는 **spool-copy 검출**
(디스크의 하네스와 실행된 하네스가 다른 경우) 하나로 한정해 인용한다.

##### ★대가 (감사가 요구한 명시)

- **harness sha가 이동한다**: `655382632e4804b3…` → **`f39b167bb6bd713a…`**.
  §0 표와 §0-b 스코프 튜플, C1 항목 (7)을 rev3에서 **재등록**했다.
  ⇒ 이 회차는 907959와도, **908020과도** "같은 하네스"가 아니다.
- **byte 핀 테스트**: 가드는 세 verbatim 블록(`ctx resolution` · `warmup launch` ·
  `instrumentation`) **밖**에 있다 — 테스트로 고정했다(`test_guard_does_not_sit_inside_a_
  verbatim_pinned_block`). 기존 `test_r2_correctness_instrument.py` / `_ctx.py`는
  **전부 통과**한다(§5).
- **§7-2("처치 축을 하나로 유지한다")를 스스로 건드린다** — 등록한다. 이 회차의 처치 축은
  여전히 grad guard 하나지만, **하네스 축이 한 번 더 움직였다.** 정당화: 등록 기판에서는
  **아직 아무 측정도 산출되지 않았으므로** "자를 재는 도중에 자를 바꾸는" 문제가 없고,
  지금이 이 검사를 넣을 수 있는 **유일하게 싼 시점**이다(감사 §8-E2 판정).
- **가드는 측정량을 바꾸지 않는다**: ★**정확히는 "GPU 작업 이전"이 아니라 "첫 부팅(서빙) 이전"이다** — 가드는 provenance 블록의 `nvidia-smi -L`·`torch.cuda.get_device_properties` **이후**에 돈다(실질 무해: 서버는 아직 뜨지 않았고 모델 가중치도 로드되지 않았다). 첫 부팅 이전에 끝나고, 통과 시 출력은
  `provenance.txt`의 3줄뿐이며 서버 argv·클라이언트·채점기에 닿지 않는다. 다만
  **NPC-D 계열로 "명령 단위 동일성"은 여전히 주장할 수 없다.**

---

## 8. 예산과 `--time` (실측 재산정)

**실측 기준선**: job 907959 = **21분 21초 / 5 boot**, `R2C_INSTRUMENT=1`. 아티팩트 타임스탬프
분해 — 21:51:36 시작 → 약 22:04 (env sync + provenance + warm-up boot + I1/I2/I3) ≈ **12.5분**,
22:05 → 22:12 (scored 4 boot) ≈ **7.5분**. scored 구간이 짧았던 것은 **TD 2 boot이 조기
사망**했기 때문이다(L boot ≈ 2.2분, TD boot ≈ 1.3분).

| 시나리오 | 계산 | 합 |
|---|---|---|
| 전형(이 회차, INSTR=0, TD가 완주) | env/prov 1 + warm-up boot 1.5 + 4 × 2.2 + 채점/teardown 0.5 | **≈ 11.8분 ⇒ 0.20 GPU-h** |
| 현실적 최악(한 boot의 클라이언트가 길어짐) | 위 + 8분 | **≈ 20분 ⇒ 0.33 GPU-h** |
| 구조적 최악(모든 timeout 발화) | warm-up health 900 s + 4 × (health 480 + client 900 + teardown 20) + 부대 180 s | **≈ 111분 ⇒ 1.85 GPU-h** |

**`--time`**: `02:30:00` **유지**(지시자 수정 없음). 월타임 상한은 어떤 측정량도 바꾸지
않고 큐 대기만 바꾼다. 이 구성에서는 **구조적 최악까지 덮는다**.

**트랙 장부 (★rev3에서 분리 표기)**:

```
R2 correctness 트랙 누적                              0.93833 GPU-h
  ├ 등록된 회차 (907032·907100·907456·907959)           0.78583
  └ ★등록 밖 실행 (908020, 스코프 불일치)               0.15250   ← 어떤 게이트도 진전시키지
                                                                   않았다 (16.3 %)
등록 실행(E3) 전형 예상 +0.20 → 누적 ≈ 1.14 (현실적 최악 1.27, 하드캡 3.44)
longctx_conflict 트랙 15.42 GPU-h — 별개 장부, 불변
```

★**합산만 하지 않는다**: "0.94 GPU-h를 썼고 게이트 상태는 X"라는 문장은 지출의 16.3 %가
어떤 게이트도 진전시키지 않았다는 사실을 지운다(C4·A908-6).
★**(RA3-11) §8의 예보(전형 ≈11.8분 ⇒ 0.20 GPU-h)는 여전히 미검증**이다 — 908020은
**2.7B · triton · ctx 4096** 이라 9분 09초였고, **9B · flashinfer · ctx 16384의 실비는
한 번도 측정된 적이 없다.** 이 예보를 "검증됐다"로 인용할 수 없다.

---

## 9. 실행 전 필수 조건 (제출 게이트)

1. `workspace/engine-port/scripts/bootstrap/sync_engine_tree.sh`가 돌아
   manifest의 `multiplexing_mixin.py` 해시가 **`e2a97b42…`** 임을 확인.
2. 전체 CPU 회귀를 돌리고 **`test_r2_correctness_*` 62/62**(rev3에서 51 → 62,
   `test_r2_correctness_scope_guard.py` 11건 추가), `test_worker_grad_guard` 13/13,
   `test_mem_telemetry_symmetry` 12/12를 확인
   (전체 discovery의 실패는 병행 워크스트림 상태에 좌우된다 — **NPC-H** 승계).
   ★추가로 `scripts/discipline/check_line_citations.py --check --all`이 **0 violation**임을
   확인한다(수리가 `multiplexing_mixin.py`의 줄 번호를 밀기 때문에, 이 검사가 이 회차에서는
   회귀가 아니라 **처치의 부작용 점검**이다 — 교훈 #80의 3번 사례와 같은 형태).
3. **(D9 승계) 제출 전 커밋**: `provenance.txt`의 `commit=`이 실제로 돈 하네스를 가리키도록
   아래가 커밋돼 있어야 한다.
   ```
   workspace/engine-port/src/multiplex/multiplexing_mixin.py                   (M)
   workspace/engine-port/results/r2_correctness/r2_correctness.sbatch          (M)
   workspace/engine-port/tests/test_worker_grad_guard.py                       (??)
   workspace/engine-port/tests/test_mem_telemetry_symmetry.py                  (??)
   workspace/engine-port/tests/test_r2_correctness_scope_guard.py              (??, rev3)
   workspace/engine-port/results/r2_correctness/rerun_prereg/                  (??)
   workspace/engine-port/results/r2_correctness/audit_908020_2026-09-14/       (??, rev3)
   workspace/engine-port/results/r2_correctness/job_908020/                    (??, rev3 — 등록 밖 실행의 아티팩트, C4)
   workspace/engine-port/results/kernel_mech/DESIGN_A1_REV2_STICKY_2026-08-25.md   (M, 인용 이동)
   workspace/engine-port/results/kernel_mech/a1/DECISION_A1_Q3_CHANNEL_2026-08-25.md (M, 인용 이동)
   workspace/engine-port/results/kernel_mech/a1/a1_q3k1_rule.py                (M, 인용 이동)
   workspace/engine-port/scripts/discipline/line_citations.json                (M, 지문 재등록)
   ```
   ★이 문서를 쓴 세션은 커밋하지 않았다(병행 워크스트림이 같은 저장소를 미커밋 상태로
   편집 중). 통합·커밋은 메인 세션이 한다.
   ★★**(rev4, P6 — 제출 전 필수)** 현 HEAD `a9cd8dd`의 `r2_correctness.sbatch`는
   **rev2 본문**(sha `655382632e…`)이고 **scope guard는 미커밋**이다. 이 상태로 제출하면
   `provenance.txt`의 `commit=`이 **실제로 돈 하네스를 가리키지 못한다**(D9가 막으려던 바로
   그 상태). 제출 전에 다음을 확인한다:
   ```bash
   git show HEAD:workspace/engine-port/results/r2_correctness/r2_correctness.sbatch | sha256sum
   #   -> f39b167bb6bd713af168abfbe3495caee099a6c0087e41b05e70b9a8d8ac0403 이어야 한다
   git status --porcelain -- workspace/engine-port/src    # -> 0줄 (C1 (9)가 보는 조건)
   ```
4. ★**제출 명령을 §0-b 스코프 튜플과 한 줄씩 대조**한 뒤에만 제출한다(G-908020-1).
   대조표는 아래 명령 블록에 붙어 있다(11/11). **대조에 쓴 하네스 줄 번호를 판정서/핸드오프에
   적는다**: `r2_correctness.sbatch:177`(MODEL 기본값) · `:191`(ABE 기본값) ·
   `:229-250`(ctx 유도 블록) · scope guard 블록.
5. **claims-auditor의 규칙층 감사 통과 전 제출 금지.**
   ★**rev1은 `NO-GO`(死因 `N2` ×2)를 받았다**(판정서 sha `71f3cfb7…`). rev2는 그 두 死因을
   §6-1·§6-2에서 제거하고 caveat C1–C10을 반영했으며, 감사가 지적하지 않은 잔여 자유
   표면 2건(29개 토큰 값의 중복 출현, 텔레메트리의 배치 관측률 61–68 %)도 닫았다
   (§6-2-2·§6-2-3). ★★**그러나 rev2의 판정서는 철회됐다**(A908-7: `GO-with-caveats` 인용
   불가, 올바른 등급은 `NO-GO`(死因 `N3`)) — 그 판정서가 §0-b를 실현하지 못하는 제출 명령을
   `:278`에서 전사하며 승인했기 때문이다. **rev3는 재감사를 새로 받아야 제출할 수 있으며,
   rev1·rev2의 어떤 판정으로도 갈음하지 않는다.**

**제출 명령 (★rev3에서 교체 — rev2의 명령이 job 908020의 死因이었다)**

```bash
cd /scratch/ehmoon/whlee/prefill-layer-alloc && \
R2C_MODEL=nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base \
R2C_ATTN_BACKEND=flashinfer \
R2C_CTX=16384 \
R2C_DSM=44 \
R2C_EXPECT_MODEL=nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base \
R2C_EXPECT_BACKEND=flashinfer \
R2C_EXPECT_CTX=16384 \
sbatch workspace/engine-port/results/r2_correctness/r2_correctness.sbatch
```

★**rev2가 등록했던 명령은 아래였고, 그것이 하네스 기본값(`:177` Zamba2-2.7B, `:191` triton,
그리고 그 모델에서 유도된 ctx 4096)으로 떨어져 job 908020을 낳았다. 인용 금지.**

```bash
# ✗ 死因 — 쓰지 말 것
cd /scratch/ehmoon/whlee/prefill-layer-alloc && sbatch .../r2_correctness.sbatch
#   (붙어 있던 정당화 문장 "환경변수를 붙이지 않는다"도 함께 철회한다)
```

- ★`R2C_INSTRUMENT`는 **붙이지 않는다** — 이 회차의 등록값은 **0**이고 그것이 기본값이다.
  `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md` §13의 `R2C_INSTRUMENT=1`은 **907959용**
  이므로 **승계 금지**(§7이 등록한 결정).
- `R2C_MEM_TELEMETRY`도 붙이지 않는다 — 이 하네스의 기본값이 1이고 그것이 등록값이다(§2).
- `R2C_DSM=44`는 기본값과 같지만 **명시한다** — C1 (4)가 `dsm=44`를 보고, 명시가 곧 진술이다.
- CLI로 `--comment`를 덮어쓰지 말 것 — 지시자에 이미 있다.
- ★**이 명령이 §0-b 스코프 튜플을 실현하는지 한 줄씩 대조했다**(G-908020-1):
  model→(1), backend→(2), ctx→(3), dsm/seed/order→(4), `MEM_TELEMETRY` 기본값 1→(5),
  `INSTRUMENT` 기본값 0 + `TFP` 기본값 1→(6), 하네스 sha→(7), manifest→(8),
  `src_dirty` 공백→(9)
  [★정정(2026-09-14, RA4-9, doc-steward): 이 문구 **"`src_dirty` 공백"은 rev3에서 철회된
  잔재**다 — 규범은 §3-b C1 (9)(`:391`, "`\"src_dirty:\"` 줄 바로 다음 줄이 `\"GPU 0: \"`로
  시작한다")이며, 이 대조표는 `"src_dirty: 다음 줄 = GPU 배너 → (9)"`로 읽는다. 이 줄이
  작성된 §9-2는 2026-09-13(job 908020 제출 시점) 작성이고 그 시점 §3-b C1 (9)는 이미
  rev4 문안이었으므로 **당시 채점된 술어 자체는 틀리지 않았다** — 대조표의 **표기만**
  rev3 시절 용어("공백")를 그대로 옮겨 쓴 잔재다. ★이 정정은 job 908179 결과 감사
  (`audit_908179_2026-09-14/VERDICT.md`)가 이미 채점한 **`PASS`를 소급 변경하지 않는다** —
  그 채점은 §3-b C1 (9) 본문(`:391`)을 근거로 이뤄졌고 이 §9-4 대조표 표기를 근거로
  이뤄지지 않았다. ★**부수 효과**: 이 追記로 본 문서의 sha가 이동해 `audit_908179_2026-09-14/
  VERDICT.md:5`가 핀한 `1e421391…`과 더 이상 재계산 일치하지 않게 된다 — 그 판정서의
  `PASS` 판정 자체는 이 정정보다 먼저 완결됐고 이 정정은 그 판정이 근거로 삼은 §3-b C1
  본문을 건드리지 않았으므로 판정을 무효화하지 않지만, 향후 이 판정서의 sha 재검증은
  이 사실을 감안해야 한다.], env 덤프→(10), scope guard→(11). **11/11 대응.**

---

## 10. 이 회차가 **하지 못하는** 것 (귀속 한계, 실행 전 고정)

★**이 회차는 처치를 분리하지 못한다.** 907959 대비 **세 축이 동시에 움직인다**:
`engine_source_hash`(grad guard), 하네스 sha(메모리 계측 + INSTR), `R2C_INSTRUMENT` 1→0.
따라서 F-a1이 충족되더라도 **"grad guard가 원인이다"는 이 job 단독으로 성립하지 않는다.**

### ★(C9) 지지 가능한 문장 — 두 경우를 **각각** 등록한다

rev1은 문장을 하나만 등록했는데 그 문장은 **10,125 토큰 배치가 형성될 때에만** 쓸 수 있고,
907959의 legacy 2 boot 중 1 boot이 그 배치를 형성하지 않았다(§6-2-1) ⇒ **형성되지 않았을 때
쓸 문장이 없었다.** rev2는 둘 다 등록한다.

> **(경우 1) `#new-token ∈ [10036, 10137]` 배치가 TD 2/2에서 형성·완주된 경우**
> — "위 스코프 튜플에서, 수리된 엔진의 true-dual boot 2/2가 907959의 true-dual boot 2/2를
> 죽였던 것과 **같은 크기의** split-prefill 배치를 완주했다."
> ★★**(rev4, RRC-8) 이 문장을 쓸 때는 `F-a1`의 참·거짓을 같은 문단에 반드시 병기한다.**
> (경우 1)의 선행절은 **F-a1이 거짓인 세계에서도 참일 수 있다** — 907959의 L1은 10,125를
> 완주한 **뒤에도 3,491을 더 형성**했으므로 "목표 배치 완주 후 다른 배치에서 사망"은
> 배치 **하나 거리**다. **F-a1이 거짓이면 (경우 1) 문장은 단독으로 쓸 수 없다.**
>
> **(경우 2) 그 배치가 형성되지 않은 경우 (등록된 유일한 대체 문장)**
> — "위 스코프 튜플에서, 수리된 엔진의 true-dual boot 2/2가 **자기가 형성한 split-prefill
> 배치를 하나도 빠짐없이 OOM 없이 완주했고**, 완주한 최대 토큰 수는 `<값>`이었다
> (907959의 true-dual 천장은 6,245였다)." ★이 문장은 **"같은 크기"를 주장하지 않는다.**
> ★★**(rev4, RA3-9) 이 문장에는 F-a2의 판정(F-a2a·F-a2b 양쪽)을 무조건 병기한다** —
> rev3는 병기 트리거를 F-a2b(기록 항목으로 강등된 절대 문턱)에만 묶어 두어, TD max > 6,245
> 이면서 F-a2a가 거짓인 세계에서 **§6-1은 `UNREALIZED`인데 §10은 병기를 면제**했다.
>
> ★**(RRC-8, 출력공간 결손 공시)** 위 둘은 **"배치가 형성됐으나 그 배치에서 사망"** 분지를
> 덮지 못한다. 그 경우 등록된 서술 문장이 **없으므로 어떤 서술 문장도 쓸 수 없다**(보수적
> 결손 — F-a1의 거짓과 그 채널 수치만 보고한다).
>
> 위 둘 **말고 다른 서술 문장은 쓸 수 없다.**

기전 귀속을 강화하는 **독립 증거**(이 job이 만드는 것이 아니라 이미 있는 것): (i) 코드 사실
— worker 스레드가 데코레이터 밖이라는 것과 `n_groups != 1`이 native 분기를 강제한다는 것,
(ii) CPU 측정 — bare Parameter 곱이 grad 상태에 따라 그래프를 만들거나 만들지 않는다는 것,
(iii) 907959의 **결정성** — 두 TD boot이 같은 배치·같은 할당 크기·같은 스택에서 죽었다.

### ★(C8) `PDMUX_WORKER_GRAD_GUARD=none` arm — **이 회차의 조건으로 달지 않는다**(감사 판정 등재)

진짜 분리는 같은 엔진 안의 `none` arm이지만, 규칙층 감사가 **이번 회차의 D조건으로 달지
않기로 판정**했고 rev2는 그 판정을 등재한다. 근거(실현가능성 검사, 게이트 #113):

- `r2_correctness_check.py:366`이 `boots.txt`에서 라벨을 열거하고 `:378`·`:446-447`이
  `startswith("TD")` / `startswith("L")`로 arm을 가른다 ⇒ `none` boot을 **scored boot으로
  넣으면 TD arm에 합류해 설계상 반드시 크래시 → 게이트가 무조건 `FAIL`** 이 된다.
- 유일하게 성립하는 형태는 **비채점 진단 boot**(warm-up 패턴)이며, 비용은 907959의 조기
  사망 TD boot 실측 1.3분 기준 **≈1.5분 ≈ 0.025 GPU-h**(rev1이 추정한 0.1 GPU-h가 아니다).
- 그 형태조차 **하네스 변경 + 테스트 + argv 픽스처 재고정**을 요구해 §7-2("처치 축을 하나로
  유지한다")를 스스로 깬다.

⇒ **별도 회차로 남긴다.** 그 회차는 **F-a1이 충족된 경우에만** 의미가 있다.

---

## 11. 인용 금지 — 승계 + 신규

### 11.1 문자 그대로 승계 (전부, 하나도 빼지 않는다)

아래 항목은 §0-c에 sha로 고정된 원문에서 **한 글자도 고치지 않고** 승계되며, 이 회차의
결과 문서는 그 원문을 **문자 그대로 전사**해야 한다.

- `PREREG_NEWPAIR_2026-09-13.md` §12: **NP-1′, NP-2, NP-3′, NP-4, NP-5, NP-6, NP-7′,
  NP-8, NP-9, NP-10** (10건)
- 같은 문서 §12의 재감사 캐비앳: **NPC-A, NPC-B, NPC-C, NPC-D, NPC-E, NPC-F, NPC-G,
  NPC-H, NPC-I, NPC-J** (10건) + **(D13-i) 스코프 배선 필수 전사 항목 2개**
- `audit_907959_2026-09-13/VERDICT.md` §12: **N-1 … N-13** (13건)
- ★**(rev3 편입)** `audit_908020_2026-09-14/VERDICT.md` §9 (sha `dc19d118…`):
  **A908-1, A908-2, A908-3, A908-4, A908-5, A908-6, A908-7** (7건)
- ★★**(rev4 편입)** `rerun_prereg/VERDICT_rerun_rev3_2026-09-14.md` §4:
  **RA3-1 … RA3-12** (12건)
- ★★**(rev4 부활)** `rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md` §5:
  **RRC-1 … RRC-13** (13건). ★**rev3는 이 13건을 0건 승계했다**(본문 "RRC" 0회).
  그 판정서의 **등급**은 철회됐지만(A908-7) **철회 사유는 제출 명령 승인이며 RRC와
  무관**하므로 **caveat는 살아 있다.** rev4가 실체 반영한 것: RRC-1·RRC-2(§6-2-3·RR-15
  근거 교체) · RRC-4(§6-5 F-a1 원인 미구별) · RRC-6(2채널 검사의 검출력) ·
  RRC-7(대역 동률 규약) · RRC-8(§10 (경우 1) 조건화 + 출력공간 결손) · RRC-9(n=2 산포) ·
  RRC-10(F-a2 거짓 분지 부재) · RRC-11(발행률 스냅샷 기준) · RRC-12(서두 표본은 D44 밖) ·
  RRC-13(C5 근거 정정 — 아래 RR-19).

**합 65건 + 전사 의무 2건.** 결과 문서가 이 중 하나라도 누락하면 그 문서는 이 사전등록을
위반한 것이다. ★**"rev3는 선행 caveat를 전부 반영했다"고 쓸 수 없다**(RA3-5).

★**A908 7건의 요지(원문은 sha 핀 판정서 §9에서 전사할 것 — 이 요약을 전사 원본으로 쓰지
마라, C4와 같은 규율)**: **A908-1** 908020은 이 사전등록의 실행이 아니며 어떤 예보도 그
위에서 채점되지 않았고 채점해서도 안 된다 · **A908-2** 908020의 TD1 8,497을 907959의
6,245와 비교 금지(수리 이전 엔진이 같은 기판에서 12,369 완주) · **A908-3** 908020의 `PASS`를
"(Zamba2, triton)에서 회귀 없음"으로 인용 금지 · **A908-4** `worker_grad_guard` 100 %와
legacy의 `null` 4키는 항등식이며 TD 실현값의 "100 %"는 **스냅샷 단위**다 · **A908-5**
`inference_mode` × `forward_native` 상호작용은 한 번도 실행된 적이 없고 안전 근거는 CPU
분석뿐이다 · **A908-6** 이 사전등록은 2개 job을 소비했고 첫째는 등록 밖이었다 ·
**A908-7** rev2 판정서의 `GO-with-caveats`는 인용 불가, 올바른 등급은 `NO-GO`(死因 `N3`).

### 11.2 ★특히 주의해서 승계할 3건 — **sha 핀 원문에서 전사** (rev2에서 절단 복원, C4)

> ★★**이 절은 전사 원본이 아니다.** rev1의 재수록 3건은 모두 **말미가 절단**돼 있었고
> (N-7: 정정의 출처, **N-8: "상류 병목 단정 금지" 조항**, NP-8: `(신규, D13)` 태그),
> 특히 N-8의 누락은 **상류 병목 귀속을 막는 바로 그 조항**이었다. 아래는 rev2에서
> sha 핀 원문(`audit_907959_2026-09-13/VERDICT.md` = `f544544f…`,
> `newpair_prereg/PREREG_NEWPAIR_2026-09-13.md` = `e832fc49…`)에서 **다시 전사**한 것이다.
> ★**결과 문서는 §11.1이 지시한 대로 sha 핀 원문에서 전사해야 하며, 이 절을 전사 원본으로
> 쓰지 마라.**

> **N-7 (I2의 분할 — ★NP-9/D6-ii 정정)** — **I2(동시성 1)의 TTFT 43.03 ms·ITL median 12.96 ms는 D44 값이 아니라 비분할(108 SM) 값이다.** 엔진은 prefill∧decode가 **동시에** 활성일 때만 D44(green idx 4)를 쓰고, 그 외에는 green context가 `null`인 idx 0/5에서 돈다(드라이버 read-out + 같은 job scored boot 텔레메트리 12,458 스냅샷 전수: idx0 11,725 / **idx4 592** / idx5 141). 동시성 1에서는 중첩이 구성상 불가능하다. **사전등록 §5-I2 (D6-ii)와 NP-9의 "D44 한정" 태그는 I2에 대해 거짓이며, I3에 대해서는 혼합비가 미측정이라 검증되지 않았다.** (λ0 rev1 판정서 `:151` "B=1은 pdmux 분할 미적용 구간"이 옳았다.)

> **N-8 (★F5 / λ_inf(B) 인용 금지)** — 사전등록 §5-I3 (D4)의 셀별 재계산 레시피로 계산한 결과 **I3b(8192,64)의 셀별 `#running-req` 최댓값은 2**(I3a는 48)다. **F5는 셀 B에서 반증됐고, 등록된 처분에 따라 `λ_inf(B)=0.6956 req/s`를 포화 처리율/상한 앵커로 인용할 수 없다.** 하네스가 기록한 `I3_max_running_req.txt = 48`은 **전역 최댓값**이며 셀 A에서만 온 값이다. 미달의 원인은 **미확정으로 기록**한다(사전등록 D4가 상류 병목 단정을 금지).

> **NP-8 (신규, D13)** — **PASS는 P2 캠페인 착수를 승인하지 않는다.** 이 게이트는 P2의
> **필요조건 하나**를 닫을 뿐이다. 남은 블로커: λ0(0단계) 사전등록이 `NO-GO`(死因 N2+N3),
> W4를 단일 λ\*로 파라미터화할 수 없다는 열린 항목(사용자 결정 대기), 그리고 방법론
> 게이트 #6(지표 절벽 대비 용량) 미충족(λ0 Q1).

### 11.3 신규 인용 금지 (이 회차 고유)

> **RR-1 (성능 금지)** — 이 회차의 어떤 수치도 성능 결과가 아니다. 특히
> **"true-dual이 빨라졌다 / 메모리를 덜 쓴다 / 오버헤드가 줄었다"는 쓸 수 없다.** 처치는
> 정확성·대칭성 수리이고, 이 job은 arm 간 타이밍을 비교하지 않으며, 계측기 자신의
> 관측자 효과가 아직 미측정이다(§2.3).

> **RR-2 (귀속 금지)** — §10 그대로. "grad guard가 OOM의 원인이었다"는 **이 job 단독으로
> 쓸 수 없다**(세 축 동시 이동). 쓸 수 있는 서술 문장은 **§10이 등록한 (경우 1)·(경우 2)
> 두 개뿐**이며, 어느 쪽을 쓰는지는 10,125 배치의 형성 여부가 결정한다(C9).

> **RR-3 (소급 금지)** — 이 수리는 907100 / 907456 / X1의 결론을 **바꾸지도 되살리지도
> 않는다.** 그 결과들은 (Zamba2-2.7B, triton, 구 `engine_source_hash`) 한정 동결이다. 단
> **"그 job들의 worker도 grad 켜진 채 돌았다"는 사실은 참**이며, 그로 인한 autograd 부기의
> arm 비대칭 오버헤드는 **크기 미측정**이다 — 그 job들을 "오염됐다"고도 "깨끗했다"고도
> 쓸 수 없다.

> **RR-4 (계측기 인용 한계)** — `gpu_mem_*` 필드는 **PyTorch 캐싱 할당자의 부기**이지
> 장치 전체 사용량이 아니다(드라이버·컨텍스트·다른 프로세스 미포함). `available_gpu_mem`
> 배너와 직접 비교하지 말 것. `gpu_mem_peak_allocated_b`는 **§2.2의 리셋 규약에 상대적**
> 이며 `gpu_mem_peak_epoch` 없이 인용할 수 없다.

> **RR-5 (ε와 θ는 측정이 아니다)** — F-b2의 `ε = 1.00 GiB`와 F-b1의
> `θ = 0.3825 MiB/token`은 **둘 다 사전 판단**이다. 대역 B/B′(부분 반증)는 "H1이 틀렸다"도
> "맞았다"도 아니며, **수리 후 양성 arm 간 피크 차이가 측정된 적이 없다**는 사실과 함께
> 인용해야 한다. rev2의 재정의는 **판단을 없애지 않고 ∅ 정의역을 없앤 것**이다.

> **RR-6 (`worker_grad_guard` 필드)** — 이 필드는 **설정값**이고,
> `*_worker_grad_enabled` / `*_worker_inference_mode`가 **실현값**이다. 둘을 같은 것으로
> 인용하지 말 것(게이트 #176). legacy의 `null`은 "grad가 꺼져 있었다"가 아니라
> **"worker 스레드가 없었다"**는 뜻이다.

> **RR-7 (D1(i) 미해결)** — 이 회차는 `R2C_INSTRUMENT=0`이므로 감사 §13-D1(i)
> (`I3_max_running_req.txt` 셀별 2줄)의 하네스 수리를 **하지 않았다.** 그 결함이 닫혔다고
> 쓸 수 없다.

#### rev2 추가분 (RR-8 … RR-15) — 규칙층 판정서 caveat C1·C2·C3·C5·C6·C7·C10의 반영

> **RR-8 (계측 비용 "0.14 µs" 인용 금지, C1)** — `0.14 µs`는 **계측기의 비용이 아니다.**
> 그것은 **이미 만들어진 nested dict에서 3키를 읽는 비용**(재측정 0.146 µs)이고,
> 기각된 철자의 `54.3 µs`는 **populated dict에 대한 flatten+sort**(재현 52.19 µs, leaf 122개)
> 로 **서로 다른 입력에서 측정**됐다. 남는 `torch._C._cuda_memoryStats` C++ 호출 1회는
> **여전히 미측정**이며 파이썬 프록시(4.18 µs = 벽시계 **0.09 %**)로만 상한이 잡힌다.
> **±3 % 예산 결론은 살아남지만 "0.14 µs"라는 표현은 인용할 수 없다.**

> **RR-9 (계측기 자신의 잔존 arm 비대칭, C2)** — `reset_peak_memory_stats()`는 **스케줄러
> 스레드에서 발행**되므로 true-dual에서는 다른 스레드의 decode forward 도중에 떨어질 수 있고
> legacy에서는 그럴 수 없다. **편향 방향 = TD 과소 보고 = 대역 A/A′ 쪽**, 크기 상한
> **156 MiB**(ε의 15.6 %). ★`Δ`(F-b2) 또는 `S`(F-b1)를 인용할 때 **이 편향 방향을 반드시
> 병기**하라. "대칭 계측"은 **필드 위치와 발행 스레드 양쪽**을 봐야 하며 이 계측기는
> 전자만 만족한다.

> **RR-10 (`INSTR=0`의 배치 형성 부작용, C3)** — `R2C_INSTRUMENT` 1→0은 warm-up의 I2/I3
> 부하를 없애 **JIT 캐시 예열을 줄인다**(`TRITON_CACHE_DIR`는 job마다 새로 생성). 이는 C층
> 배치 형성 타이밍을 흔든다 ⇒ **"907959와 같은 조건에서 배치가 형성될 것"이라고 쓸 수 없다.**

> **RR-11 (게이트 #176을 "테스트로 고정했다"고 쓸 수 없다, C5)** — 감사자 변이 **M-J**
> (실현값 읽기를 설정값 에코로 교체)가 **등록된 13개 테스트를 전부 통과**했다. 제출된 코드는
> 실현값을 읽고 있어 결함이 아니지만, **F-d의 실현값 성격은 파일 단위 manifest 해시로만
> 보호되고 변이 대조로는 보호되지 않는다.**

> **RR-12 (F-d의 두 절은 항등식, C6)** — `worker_grad_guard == "inference_mode"`는 하네스가
> `PDMUX_*`를 전부 unset하므로 **거짓이 될 수 없고**, legacy의 realised 4키 `null`은
> **worker 스레드가 없다는 것의 연역**이다. **"F-d가 충족됐다"를 정보량 있는 확인으로
> 인용할 수 없다.** 거짓이 될 수 있는 것은 TD의 realised 키뿐이고, 그 거짓은 "설치 트리가
> 수리본이 아니다"뿐 아니라 **"계측 플래그가 전파되지 않았다"** 로도 발생한다.

> **RR-13 (대역 B의 흡수율과 보조 산출물, C7)** — F-a1 충족 조건부로 도달 가능한 Δ의
> 상한은 ≈ **8.1–9.6 GiB**이고 "미확정"으로 등록된 대역 B(1.00–6.30 GiB)가 그 구간의
> **약 62 %** 를 흡수한다 ⇒ **F-b2가 H1에 불리한 판정을 낼 여지는 좁다.** 또한
> `gpu_mem_allocated_b` 대 `prefill_chunk_progress`는 **"곡선"이 아니며**(907959 L1의
> `==14` 146 스냅샷 중 128개가 종단 progress 56에 몰림), 스냅샷 발행률(★스냅샷 기준, 줄 수 아님 — RRC-11)이 arm마다 **1.45×**
> 다르므로 **순간값의 arm 비교에 쓸 수 없다.**

> **RR-14 (승계 사슬의 강제, C10)** — RR-1 … RR-19는 §11.1의 구조상 **이 회차의 결과
> 문서까지만** 의무화돼 있다. ★**다음 회차 사전등록은 이 문서를 sha로 핀하고
> RR-1 … RR-19 + RA3-1 … RA3-12 + RRC-1 … RRC-13을 자신의 §11.1 목록에 명시적으로 편입해야
> 한다**(특히 RR-7: 감사 §13-D1(i) 미해결, 그리고 **RR-15는 RA3-4의 정정 없이 전사할 수
> 없다**). 편입하지 않으면 그 회차는 이 조항을 위반한 것이다.
> ★**(rev4, G-RA3-3) 판정서를 철회할 때는 그 판정서가 등록한 caveat 중 무엇이 함께 죽는지
> 열거하라** — rev2 판정서의 등급 철회가 RRC-1…13을 통째로 소실시켰고, 그중 RRC-1은
> rev3가 **그 시점에도 싣고 있던 거짓 문장의 정정**이었다.

> **RR-15 (★rev4에서 근거 교체 — 브래킷은 유지, 사유는 폐기)** — 검정력 브래킷
> **하한 2.88 GiB(ordinal 34, `T = 2310`) / 상한 4.04 GiB(ordinal 38, `T = 3241`)** 의
> **등록은 유지된다.** 그러나 rev2/rev3가 적은 근거 — *"ordinal 38은 4 boot 어디서도
> in-flight로 관측되지 않았다"* — 는 ★**거짓이며 전사 금지**다(§6-2-3 정정:
> `prefill_active_batch_size == 4` 스냅샷 **58/58/54/54**, `stream_index 4` **100 %**).
> **올바른 근거는 `P` 자체가 데이터 의존이라는 것**이다: 첫 C층 배치가 갈리면 `P = 37`이
> 되고 `max T`가 2310으로 떨어진다.
> ★**1차 판정서의 `4.04 GiB`는 철회되지 않았다** — 907959 증거로는 4/4 boot에서 실현된다.
> RR-15가 그것을 강등했던 **근거**가 거짓이었을 뿐이고, 강등은 위의 `P` 의존성으로만
> 유지된다. ordinal 38 상실 시 실제 손실은 **`Sxx` 30,547,722 → 24,284,469(79 %),
> 기울기 SE ×1.12**(레버리지 지분 20.0 %)이므로 ★**"ordinal 38이 빠지면 검정력이 크게
> 준다"고 쓸 수 없다.**
> ★결과 문서는 **실제로 정의역에 들어온 최대 `T`와 `|D_b|`를 boot별로 병기**해야 하며,
> 그것 없이 검정력을 주장할 수 없다.

#### rev4 추가분 (RR-16 … RR-19) — rev3 감사 caveat RA3-1…12 · 부활한 RRC의 반영

> **RR-16 (§1.2(b)의 "정확히 둘"은 인용 금지, RA3-6)** — rev1–rev3가 적은 *"inference tensor가
> no-grad 텐서보다 엄한 지점은 정확히 둘이며 둘 다 소비자가 inference mode 밖에 있을 때만
> 발화한다"* 는 **실측으로 거짓**이다: 설치본 `libtorch_cpu.so`에 하드 에러 **최소 4종**이
> 있고 그중 **`Inference tensors do not track version counter.` 는 inference mode 안에서도
> 발화한다.** 또한 **`no_grad` 결과 텐서는 네 경로를 전부 회피**한다 ⇒ `inference_mode`를
> 고른 **대가는 문서가 적던 것보다 크다.** 저울질을 인용할 때는 반드시 4종 기준으로 적는다.

> **RR-17 (F-a2는 예보가 아니라 기록 항목, RA3-2/RRC-10)** — F-a2a·F-a2b는 충족돼도
> 미충족돼도 **게이트 라벨도 H1 판정도 움직이지 않는다.** 출력공간에 **"거짓"이 없다**
> (넘으면 참 / 못 넘으면 `UNREALIZED` / F-a1 거짓이면 미채점). **F-a2 참을 H1의 추가 확증
> 으로 인용할 수 없다.** 문턱 20 %를 사후에 옮기지 않은 처리는 옳지만, **그 정직성이
> F-a2를 예보로 살려 두는 근거는 아니다.**

> **RR-18 (6,245 재근거의 *논증*은 성립하지 않는다 — 결론만 유지, RA3-3)** — "두 기판 모두
> 4항 부분합" 산술은 참이지만 **관측 배치는 4항이 아니고**(907959 8-seq / 908020 7-seq),
> 6,245로 가는 부분합이 각각 **72,489 · 35,597개**라 "부분합이다"는 거의 무정보이며,
> **두 기판이 같은 값에 떨어진 사실은 밀도로 설명되지 않는다** ⇒ **"6,245는 도착 조합의
> 실현값이다"는 `NOT-YET-SUPPORTED`.** F-a2b 강등 결론은 유지되나, 그것을 지탱하는 것은
> **907100 TD1이 수리 이전에 같은 기판에서 12,369를 완주했다**는 사실 하나뿐이고, ★그 사실은
> **Zamba2 기판**의 것이므로 **NemotronH 기판에서의 "F-a2b 검정력 0"은 미검증**이다.

> **RR-19 (C5를 열어 둔 근거 정정, RRC-13)** — M-J를 잡는 테스트를 추가하지 않기로 한
> **결론은 유지**하되, rev1–rev3가 적은 근거("추가하면 인증된 해시를 움직인다")는
> **사실이 아니다** — manifest 25항목은 전부 설치 트리 **엔진 소스**이고
> `engine_source_hash`의 8모듈에도 테스트는 없다 ⇒ **테스트 추가는 mixin sha·
> `engine_source_hash`·manifest·하네스 어느 것도 움직이지 않는다.** 수용 가능한 실제 근거는
> **"F-d에 처분이 붙은 예보는 TD 실현 2키뿐이고 그 거짓은 `NO_VERDICT_INFRA`이므로 이
> 회차의 결론이 M-J에 의존하지 않는다"** 이다. (M-J는 재주입에서 여전히 전 테스트를
> 통과한다 — RR-11 유효.)

---

## 12. 자기 적용 (게이트 #113) · 실행 전 공시

- **처방의 실현가능성**을 전부 적었다: F-a1은 로그 정규식 2개 + 파일 존재 + JSON 키 2개
  (**전부 항상 정의된다**), F-a2는 로그 1개 최댓값, F-b1은 로그 정규식 1개 + 이미 존재하는
  텔레메트리 필드 3개(`gpu_mem_peak_allocated_b`·`gpu_mem_peak_epoch`·
  `prefill_active_batch_size`)의 OLS, F-b2는 같은 필드, F-d는 텔레메트리 4키.
  **새 계측 0 · 새 엔진 코드 0 · 분석 스크립트 1개(결과 회차에서 작성).**
- ★**rev2의 정직한 공시 — 남은 판단과 남은 ∅ 위험**:
  (i) `ε = 1.00 GiB`와 `θ = 0.3825 MiB/token`은 **사전 판단**이다(RR-5). rev2는 판단을
  없애지 않고 **∅ 정의역과 미등록 재량**을 없앴다.
  (ii) F-b1의 정의역조차 **완전 보장은 아니다** — 텔레메트리가 형성 배치의 61–68 %만
  관측한다(§6-2-3, 907959 실측). 그래서 rev2는 **희소/∅ 처분(`|D_b| < 10`, `max T < 2000`,
  중첩 < 8)을 실행 전에 등록**했고, 그 처분은 `UNREALIZED`이지 H1의 실패가 아니다.
  (iii) 1차 판정 **F-a1은 이 위험에서 완전히 자유롭다** — 네 채널 모두 항상 정의된다.
- **반증 실패 공시(현시점)**: 수리가 **GPU에서** OOM을 없앤다는 것은 **아직 아무 증거도
  없다.** 지금 있는 것은 코드 사실 + CPU 측정 + 907959의 결정성뿐이다. 이 문서는 그 셋을
  근거로 **예보**를 등록할 뿐이며, 예보가 어긋나는 분지(§6-1 F-a1 거짓, §6-2 대역 C/C′)를
  **처분까지 포함해** 먼저 적었다.
- **rev1 → rev2에서 코드·하네스·테스트·manifest·라인 인용·예산은 한 바이트도 바뀌지
  않았다** — 규칙층 감사가 §1에서 전 항목을 독립 재검증해 일치를 공시했고, §4에서 반증
  실패 8건을 공시했다. rev2는 **문서 텍스트만** 고쳤다.
- ★**rev2 → rev3에서 엔진은 바뀌지 않았다**(`multiplexing_mixin.py` `e2a97b42…`,
  `engine_source_hash` `eba74cbd…` 불변). 움직인 것은 **하네스 1건**(scope guard,
  sha `655382632e…` → `f39b167b…`)과 문서다. 그 대가는 §7-c에 명시했다.
- ★**rev3의 실패 공시 — 이 사전등록은 스스로 1개 job을 낭비시켰다.** §9의 제출 명령이
  §0-b를 실현하지 못했고(死因), 규칙층 감사 2회가 그것을 잡지 못했으며(A908-7), 그
  결과가 job 908020(0.15250 GPU-h, 등록 밖)이다. **어떤 게이트도 진전하지 않았다.**
  rev3는 같은 결함을 **문서(§9 명령 교체 + §3-b C1–C5)와 하네스(§7-c fail-closed) 양쪽**
  에서 닫았지만, ★**세 겹 방어선이 전부 뚫렸다는 사실 자체는 지워지지 않는다**(C4).
- ★**rev3가 수정한 자기 오류 1건**: rev2는 F-a2 문턱 6,245를 "측정된 상수, 재량 0"으로
  등록했으나, 그것은 **도착 조합의 실현값**이었다(§6-1의 부분합 검증). 이 오류는 감사가
  아니라 작성자가 907456의 gap 32.2 %를 재계산하는 과정에서 드러났고, **문턱을 옮기지
  않고 비특이성을 공시하는 쪽**을 택했다. ★**단 rev4에서 그 *논증* 도 강등됐다**(RR-18:
  부분합은 거의 무정보이고 cross-substrate 일치를 설명하지 못한다 ⇒ 강등을 지탱하는 것은
  907100의 12,369 하나뿐이며 그것은 **Zamba2 기판**의 사실이다).
- ★★**rev4의 실패 공시 — 같은 실패를 두 층에서 반복했다.**
  (i) **게이트 #110 실패**: rev3는 감사 문안 C1의 (7)·(10)을 원자료로 재검증해 정정했으면서
  **(9)는 글자 그대로 승계**했고, 그 한 항목이 **캠페인 전체를 라벨 이전에 무효화**하는
  死因이었다. **재검증한 항목의 성공이 나머지의 신뢰를 위조했다.**
  (ii) **자기 집계 오류**: rev2/rev3의 "ordinal 38 미관측 · 관측률 61–68 %"는 **비영 최대
  런**으로 센 결과였고, 직접 술어로 세면 **58/58/54/54 · D44 100 %** 다. 즉 **없는 제약을
  등록하고 그 위에 검정력 논증을 세웠다.** 두 오류 모두 **원자료 한 번의 직독으로 잡혔을
  것**이며, rev4는 그 직독을 수행한 뒤에 쓰였다.
  (iii) **승계 누락**: rev3는 rev2 판정서의 **RRC-1…13을 0건 승계**했다 — 등급 철회를
  caveat 소멸로 오독한 결과이며, 그중 RRC-1이 (ii)의 정정이었다.
- ★**rev4가 바꾸지 않은 것**: 코드·엔진·하네스·채점기·manifest **0건**. 감사가 그것을
  요구하지 않은 이유는 요구하면 **§1의 독립 재검증 상수가 전부 무효화되고 처치 축이
  늘어나기** 때문이다.
- **새 성능 판정 0건.** HE0 · 정책 순위 · stake #1 · 게이트 #13/#16 · Claim D 등급(미검증) ·
  Zamba2/triton 동결 — 전부 불변.
- **이 문서 작업의 GPU 신규 지출 0.** `results/r2_eval/lambda0_prereg/**` 미접촉.
  R2 correctness 트랙 장부 **0.93833 GPU-h**(등록 회차 0.78583 + 등록 밖 0.15250, §8).

---

## 13. 신규 게이트 후보 (rev3 신설 — 전역 번호는 doc-steward가 부여, 다음 자유 번호 #201 이후)

원문은 `audit_908020_2026-09-14/VERDICT.md` §5.3(sha `dc19d118…`). 여기 등재하는 이유는
이 문서 자신이 다섯 항목 **전부의 실증 사례**이기 때문이다.

> **G-908020-1** — **사전등록의 "제출 명령"은 그 문서의 스코프 튜플을 실현하는지 한 줄씩
> 대조한 뒤에만 승인하라 — 하네스 기본값이 튜플과 다르면 명령 자체가 死因이다.** 규칙층
> 감사 2회 모두 세 env 변수명을 **0회** 언급했고 rev2 판정서는 그 명령을 전사하며 승인했다.
> 실무 규칙: 감사 체크리스트에 **"등록 명령 ⊨ 등록 튜플"** 항목을 넣고, 대조에 쓴 하네스
> 줄 번호를 판정서에 적는다. (rev3 §9가 이 대조표를 11/11로 붙였다.)

> **G-908020-2** — **rev 복사 시 "불변"이라고 적은 축이 실행 절차에서 사라졌는지 확인하라 —
> 삭제는 diff에 안 잡히고 `grep`에도 안 잡힌다.** 올바른 명령은 sha로 핀된 선행 문서
> (`newpair_prereg` §13)에 있었는데 rerun rev1/rev2로 오면서 소실됐고, 같은 문서 §0 표는
> 그 축을 "동일 | 동일 | 불변"으로 선언하고 있었다.

> **G-908020-3** — **"빠뜨리면 조용히 망가지는 단일 인적 실패점"을 하나 제거했다면, 같은
> 명령줄의 나머지 인자 전부에 같은 검사를 적용하라.** 선행 사전등록의 D11은 `--time`에
> 대해 정확히 이 진단을 내리고 지시자로 내재화했으나, **모델 정체성 세 env는 사람 경로에
> 남겨 뒀고 하네스 기본값은 반대 모델이었다.** 처방은 국소, 결함은 계열이다(교훈 (85)의
> 명령줄 판). ★**rev4 정정**: rev3 §7-c가 닫은 것은 **model/backend/ctx 3축뿐**이고, 나머지 6축(`R2C_{DSM,SEED,ORDER,TRACE_FORCE_PREFILL,INSTRUMENT,MEM_TELEMETRY}`)이 무해한 이유는 **그 기본값이 우연히 등록값과 같기 때문**이다(6/6 일치, 감사 T5 재현). 그 우연이 깨지면 같은 실패 모드가 재발하고 **C3 예산은 0이다.** "이 계열을 닫았다"고 쓸 수 없다.

> **G-908020-4** — **스코프 불일치 처분을 사전등록에 넣어라. 넣지 않으면 "등록 실험이
> 돌았는가"가 라벨을 본 뒤의 사람 재량이 된다.** 최소 요구: (i) `provenance.txt`만으로
> 판정되는 기계적 일치 술어, (ii) 불일치 라벨 `NO_VERDICT_SCOPE`(게이트 실패 아님),
> (iii) **유한한** 운영오류 재실행 예산, (iv) 불일치 job의 라벨·수치 병기 의무.
> (rev3 §3-b가 (i)–(iv)를 등록했다.)

> **G-908020-5** — **사전등록의 1차 판정 술어를 "처치 이전 엔진 + 같은 기판"의 기존
> 원자료에 먹여 보고, 거기서도 참이면 그 회차는 그 기판에서 검정력이 0이다.** F-a1을
> 907100/907456에 먹이면 4채널 전부 참이고, F-a2 문턱 6,245는 907100 TD1의 **12,369**에
> 이미 초과돼 있다. rev2 §6-0은 이 검증을 **907959에 대해서만** 했고 실제로 돌아간 기판에
> 대해서는 하지 않았다. (rev3 §6-1이 F-a2를 이 근거로 재근거하고 비특이성을 공시했다.)

### rev4 추가분 — `G-RA3-1 … G-RA3-6` (원문 `VERDICT_rerun_rev3_2026-09-14.md` §9, sha `d8281866…`)

> **G-RA3-1** — **"블록이 비어 있다"를 게이트 술어로 쓰지 마라. 빈 출력은 줄을 남기지 않으므로
> "다음 줄"은 언제나 그 다음 명령의 것이다.** 경계는 *공백*이 아니라 ***다음 마커의 존재***로
> 등록하라. (이 문서 C1 (9)의 死因. 교체 문안이 `"GPU 0: "`라는 **다음 마커**를 쓰는 이유다.)

> **G-RA3-2** — **재감사에서 직전 판정서 문안을 일부만 재검증하면, 재검증한 항목이 나머지의
> 신뢰를 위조한다.** rev3는 (7)·(10)을 정정했고 **그 성공이 (9)의 무검증을 가렸다**
> (게이트 #110 심화). 실무 규칙: 승계한 술어는 **전 항목을 원자료에 한 번씩 먹여 보고**,
> 먹여 본 항목과 안 먹여 본 항목을 문서에 **구분해 적어라.**

> **G-RA3-3** — **철회된 판정서의 *등급*과 그 판정서가 등록한 *caveat*는 별개다.** 철회할
> 때 **어떤 항목이 함께 죽는지 열거**하라. (rev2 등급 철회가 RRC-1…13을 통째로 소실시켰고,
> 그중 RRC-1은 rev3가 **그때도 싣고 있던 거짓 문장**의 정정이었다.)

> **G-RA3-4** — **"엄한 지점은 정확히 N개"라는 라이브러리 사실은 설치본에서 열거해
> 등록하라.** 열거가 부분집합이면 **처치층 아티팩트가 가설 판정으로 기록된다**
> (이 문서 §1.2(b) → §6-5의 트리거 계열화).

> **G-RA3-5** — **cross-substrate로 같은 수가 나오면 "조밀해서"로 설명하지 마라 — 밀도는
> 실현 가능성을 설명하지 *일치*를 설명하지 않는다.** (6,245 부분합 논증의 死因, RR-18.)

> **G-RA3-6** — **fail-closed 가드를 넣은 뒤 그 가드의 발화를 provenance 술어로 추가하면
> 그 술어는 (거의) 항등식이다.** 남는 정보(spool-copy 검출)를 명시하고 **"독립 채널"이라
> 쓰지 마라.** (C1 (11), RA3-8.)

> ★**추가 (작성자 자기 등재)** — **집계 정의를 바꾸면 "관측되지 않았다"가 뒤집힌다.
> 부재 주장은 파생 집계가 아니라 직접 술어로 확인하라.** rev2/rev3의 "ordinal 38 미관측 ·
> 관측률 61–68 %"는 **비영 최대 런**이라는 파생 집계의 산물이었고, 직접 술어
> (`prefill_active_batch_size == 4`)로 세면 **4 boot 전부 관측**된다. **부재는 존재보다
> 집계 정의에 민감하다.**

---

### 관련 파일 (절대경로)

- 수리: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/src/multiplex/multiplexing_mixin.py`
- 하네스: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/r2_correctness.sbatch`
- 채점기(무수정): `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/r2_correctness_check.py`
- 테스트: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/tests/test_worker_grad_guard.py` ·
  `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/tests/test_mem_telemetry_symmetry.py` ·
  `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/tests/test_r2_correctness_scope_guard.py` (rev3)
- rev1 규칙층 판정서(`NO-GO`): `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/rerun_prereg/VERDICT_rerun_rules_2026-09-14.md`
- rev2 규칙층 판정서(★**철회 — 인용 불가**, A908-7): `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/rerun_prereg/VERDICT_rerun_rev2_2026-09-14.md`
- ★**908020 판정서**(C1–C5 · A908-1…7 · G-908020-1…5의 원문): `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/audit_908020_2026-09-14/VERDICT.md`
- ★**등록 밖 실행의 아티팩트**: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/job_908020/`
- 직전 회차 원자료: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/job_907959/`
- Zamba2 선례(F-a2 재근거의 원자료): `.../results/r2_correctness/job_907100/`(TD1 `srv_TD1.log:2619` = `13/12369`) · `.../job_907456/`
- 승계 문서: 같은 트랙 `newpair_prereg/` · `audit_907959_2026-09-13/`
- 크래시 지점(이번 회차 무수정):
  `/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/layers/attention/mamba/mixer2_rms_norm_gated.py:97`
