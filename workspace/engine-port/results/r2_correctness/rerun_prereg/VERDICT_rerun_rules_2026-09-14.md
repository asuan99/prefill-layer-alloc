# 규칙층 사전등록 판정서 — 907959 재실행 (grad guard 수리 + 대칭 메모리 계측)

**감사일** 2026-09-14 · **감사자** claims-auditor (read-only) · **GPU 신규 지출 0**
**대상** `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/rerun_prereg/PREREG_RERUN_2026-09-13.md`
**부수 대상** `workspace/engine-port/src/multiplex/multiplexing_mixin.py` (M) · `results/r2_correctness/r2_correctness.sbatch` (M) · `tests/test_worker_grad_guard.py` (신규 13) · `tests/test_mem_telemetry_symmetry.py` (신규 12)
**트리 상태** HEAD `b2b60e7`, 작업트리 미커밋 37항목 (감사 시점 확인)

---

## 0. 등급

> # `NO-GO`
>
> **死因 2건, 둘 다 `N2`(반전 확인 — 자유도가 등록 판정을 실제로 뒤집었고 수치가 있다).**
> **제출 불가.** 단 두 死因 모두 **GPU 0 · 문서 §6-1/§6-2 두 절의 텍스트 수정만으로** 제거
> 가능하며, 처치·코드·하네스·테스트·manifest는 **고칠 것이 없다**(§4 반증 실패 목록 참조).
> 엔진 수리 자체는 이 판정과 독립적으로 옳다.

---

## 1. 감사자가 **독립 재검증**한 것 (게이트 #110 — 직전 판정서 수치를 상수로 승계하지 않음)

현 트리·현 원자료에서 전부 다시 계산했다. 승계 문서의 수치를 인용만 하지 않았다.

| 항목 | 사전등록/승계 문서의 주장 | 감사자 재계산 | 판정 |
|---|---|---|---|
| 승계 문서 6종 sha256 | §0-c 표 | `e832fc49…` / `45ee250d…` / `cd1ffe8b…` / `f544544f…` / `8c857bed…` / `640fd3ad…` | **6/6 일치** |
| 채점기 sha (무수정) | `ec355e17…` | `ec355e171a66d68eab1300edcc7616a693cd55aa2c3fb5c251ee4b42eac50d30` | **일치** |
| 클라이언트 sha | `95e10b49…` | `95e10b492ed6239d83cae20ad7898bdbbcce1713a37d42c2bfa111f7347ec367` | **일치** |
| 하네스 sha | `655382632e4804b3…` | `655382632e4804b360c40e399573f023e5e2b6a9c49a46587ebfddd2081aaede` | **일치** |
| `multiplexing_mixin.py` sha | `e2a97b42…` | repo src = 설치 트리 = `e2a97b423f93ff6d09f08b1e649596dad010285775845261ed27233c0346f243` | **일치, 설치 트리 동기 확인** |
| manifest 24항목 불변 | §4 | 907959 manifest 25줄을 현 설치 트리와 전수 대조: **unchanged=24 / changed=1(entry 3) / missing=0** | **정확** |
| `engine_source_hash` | `eba74cbd…` | `profile.py:365-431`의 스킴을 손으로 재구현해 8모듈 해시: `eba74cbd2cebfbd0ac112fc4693cac516dacf95ce4250d899e96572e7fafe41a` | **일치** |
| 라인 인용 17건 "번호만 이동" | §5 | `line_citations.json` 전 문서 비교: 이동 키 **17** (DESIGN_A1 10 · DECISION_A1_Q3 4 · a1_q3k1_rule 3), **(sha, anchor, anchor_offset, target) 다중집합 문서별 전부 동일**, in-place 값 변경 **0**, 다른 문서 무접촉 | **정확** |
| CPU 회귀 | 13/13 · 12/12 · 51/51 | 직접 실행: `test_worker_grad_guard` **13 OK** · `test_mem_telemetry_symmetry` **12 OK** · `test_r2_correctness_*` **51 OK** | **재현** |
| 예산 산술 | 0.20 / 0.33 / 1.85 GPU-h, 구조적 최악 111분 | `900 + 4×(480+900+20) + 180 = 6680 s = 111.3분 = 1.855 GPU-h`; 전형 `1+1.5+8.8+0.5 = 11.8분 = 0.197`; 장부 `0.786+0.197=0.983` | **전부 일치** |
| 관측자 효과 122항목 / 54.3 µs / 0.14 µs | §2.3 | 재측정: 평탄화 leaf **122개**(정확 일치), populated dict flatten+sort **52.19 µs**(cf. 54.3), 3키 읽기 **0.146 µs**(cf. 0.14) | **재현 — 단 조건 기술이 부정확, caveat C1** |
| Zamba2 면역 (`n_groups=1`) | §1.3 | `hf_cache/hub/models--Zyphra--Zamba2-2.7B/.../config.json` → `mamba_ngroups: 1`; `configs/nemotron_h.py:270` → `mamba_n_groups=8`; `mixer2_rms_norm_gated.py:109-110` 분기 조건 · `:97` bare Parameter · `:114` `.data` 전부 파일에서 확인 | **정확** |
| `inference_mode` 텐서 의미론 5항 | §1.2(b) 표 | torch 2.9.1로 직접 실행 재현 (§4-3) | **5/5 재현** |

---

## 2. 死因

### 死因 1 — `N2` : **F-b의 추정량 `peak(boot)`이 정의역 ∅일 때의 처분이 미등록이고, 그 ∅가 실제로 관측돼 있다**

#### 2-1. 사실 (907959 원자료, 감사자 전수 재집계)

`tel_*.jsonl` 전 스냅샷의 `prefill_active_batch_size` 분포:

```
L1 : 12,511 snapshots   {0:7462, 1:4700, 3:42, 4:58, 8:50, 14:146, None:53}
L2 : 13,197 snapshots   {0:7929, 1:4892,        4:58, 9:86, 16:180, None:52}   <- ==14 가 0개
TD1:  7,611 snapshots   {0:5426, 1:2056,        4:54, 8:18,         None:57}
TD2:  7,477 snapshots   {0:5336, 1:2012,        4:54, 8:18,         None:57}
```

서버 로그의 완주 split-prefill 배치 열(감사자 전수 추출):

```
L1 : … 1/1468, 4/3241, 8/6245, 14/10125, 3/3491        (41 배치, 최대 10,125 tok)
L2 : … 1/1468, 4/3241, 9/7345, 16/12516                (40 배치, 최대 12,516 tok)  <- 14-seq 배치 없음
TD1: … 1/1468, 4/3241, 8/6245                          (39 배치, 최대  6,245 tok)
TD2: … 1/1468, 4/3241, 8/6245                          (39 배치, 최대  6,245 tok)
```

`prefill_active_batch_size`가 곧 `#new-seq`임은 두 채널의 집합이 boot별로 정확히 일치함으로
확인된다(L1 {1,3,4,8,14} · L2 {1,4,9,16}).

⇒ **같은 arm·같은 코드·같은 클라이언트·같은 seed의 두 legacy boot이 C층에서 서로 다른 배치를
형성했다.** `L2`에는 `prefill_active_batch_size == 14`인 스냅샷이 **한 개도 없다.**

#### 2-2. 그것이 등록된 판정을 어떻게 뒤집는가

사전등록 §6-2의 추정량은 다음과 같이 **실행 전 고정**돼 있다:

```
peak(boot) = max{ gpu_mem_peak_allocated_b : gpu_mem_peak_epoch == e*, prefill_active_batch_size == 14 }
Δ = mean(peak(TD1), peak(TD2)) - mean(peak(L1), peak(L2))
```

그리고 대역 A/B/C/D는 **오직 Δ에만** 키가 걸려 있다. **F-b에는 `UNREALIZED`(측정 실패) 분지가
등록돼 있지 않다** — F-a에는 있는데 F-b에는 없다. `{스냅샷 : batch_size==14}`가 비면
`max{}`가 정의되지 않고, `mean(peak(L1), peak(L2))`도 정의되지 않는다.

그 상황에서 채점자에게 남는 선택지는 최소 셋이고, **등록 문안은 그중 어느 것도 배제하지 않는다**:

| 선택 | 등록 문안이 금지하나? | 나오는 판정 |
|---|---|---|
| (i) 정의역이 빈 boot을 평균에서 제외 | 금지 안 됨 | `mean(L) = peak(L1)` → 대역 A/B/C/D 중 하나 **발화** |
| (ii) 그 boot의 최대 배치로 대체(L2의 16-seq/12,516) | 금지 안 됨 | `mean(L)`이 (i)보다 **0.236–0.413 GiB 큼** → Δ가 그만큼 **작아짐** |
| (iii) F-b를 `UNREALIZED`로 기록 | 금지 안 됨(다만 그런 분지가 등록돼 있지도 않음) | **대역 라벨 없음 — H1 관련 판정 0** |

**수치(반전):**
- (iii) ↔ (i)/(ii) : "F-b 충족 / H1 부분 반증 / H1 REFUTED" ↔ **"F-b 판정 불가"**. 모델링 가정 0.
- (i) ↔ (ii) : 대체 배치 12,516 tok = L1의 10,125 tok의 **×1.2362**. 진단서 §3이 등재한 배치 내
  임시텐서는 전부 `[T, 10240]` 모양이라 **T에 선형**이다(fp32 임시텐서 395.5 MiB @T=10125 →
  488.8 MiB @T=12516; bf16 `preallocated_ssm_out` 197.75 → 244.5 MiB). 진단서가 등록한 legacy
  working-set 대역 **2.0–3.5 GiB**를 쓰면 `mean(L)` 이동량 = `0.5 × 0.2362 × [2.0, 3.5]` =
  **[0.236, 0.413] GiB** = **ε(1.00 GiB)의 23.6–41.3 %**.
  ⇒ 참 Δ가 `[1.00, 1.41] GiB` 구간이면 **대역 B("H1 부분 반증") → 대역 A("F-b 충족")** 로 뒤집히고,
  `[−1.41, −1.00]` 구간이면 **대역 D("예상 밖, 해석 금지") → 대역 A** 로 뒤집힌다.

#### 2-3. 발생 확률이 낮은 예외가 아니다

Δ가 정의되려면 **4 boot 전부**가 `==14` 스냅샷을 가져야 한다. 존재하는 유일한 실현에서 legacy
2 boot 중 **1 boot(50 %)** 이 비어 있었다. 이 단일 관측으로 점추정하면 4 boot 전부 비어 있지 않을
확률은 `(1/2)^4 ≈ 6 %`다(n=2라 구간은 매우 넓으므로 이 6 %를 수치로 인용하지는 말 것). 중요한 것은
**"∅가 실제로 일어난다"가 관측된 사실**이고, **그에 대한 처분이 등록돼 있지 않다**는 것이다.

이것은 `N3`(정의역 ∅·퇴화)의 성격도 함께 갖지만, 반전을 수치로 만들었으므로 **`N2`로 계상**한다.

---

### 死因 2 — `N2` : **F-a의 반증 분지에 판정 채널이 없고, 등록된 처분이 H1을 반증하는 관측을 "측정 실패"로 강등한다**

#### 2-1. 등록 문안

§6-1이 실행 전 고정한 두 분지:

> **반증 분지** — "TD 2/2 중 하나라도 **같은 배치에서** `torch.OutOfMemoryError`로 죽으면 ⇒ **H1은 `REFUTED`**"
> **측정 실패 분지** — "`#new-token ∈ [10036, 10137]` 배치 **자체가 형성되지 않으면** F-a는 `UNREALIZED` … 대체 술어 … 그 최대가 10,036 미만이면 F-a는 **`UNREALIZED`(측정 실패)이지 H1의 실패가 아니다**"

#### 2-2. 문제 — 두 분지를 가르는 관측 채널이 등록돼 있지 않다

사전등록 자신이 §6-1에서 적고 있다: *"`report_prefill_stats`는 prefill **완료 시** 호출되므로,
그 줄의 출현이 곧 완주다."* ⇒ **죽은 배치는 로그 줄을 남기지 않는다.** 따라서 "형성됐지만
죽었다"와 "형성되지 않았다"를 **등록된 채널로는 구별할 수 없다**. 감사 §3-2가 907959에서 그
구별을 해낸 방법(C층 프롬프트 합 24,993 · TD 완주 11,377 · 잔여 13,616 = 10,125+3,491 ·
`#queue-req` 14→0 · `198.00 MiB ↔ T ∈ [10036,10137]` 역산)은 **다채널 법의학 재구성**이며
이 사전등록에 **한 줄도 채널로 등록돼 있지 않다**.

#### 2-3. 반전 — 등록 규칙을 907959 자신의 TD 원자료에 먹이면 정본과 반대 라벨이 나온다

감사자가 등록 규칙 그대로 채점했다(원자료: `job_907959/srv_TD1.log`, `srv_TD2.log`):

| 등록 술어 | TD1/TD2 실측 |
|---|---|
| `srv_TD*.log`에 `#new-seq: 14, #new-token: 10125` 줄 | **0건** (죽은 배치라 줄이 없다) |
| `tel_TD*.jsonl`의 `prefill_active_batch_size == 14` 스냅샷 | **0개** |
| 대체 술어 접속항 2 — 완주한 최대 `#new-token` | **6,245** (< 10,036) |

**읽기 (R2)**(등록 문안의 직독): 10125 줄이 없다 ⇒ "배치 자체가 형성되지 않았다" ⇒ `UNREALIZED`;
대체 술어의 최대가 6,245 < 10,036 ⇒ 등록된 처분 그대로 **"F-a는 `UNREALIZED`(측정 실패)**
**이지 H1의 실패가 아니다"**.

**읽기 (R1)**(감사 §3-2식 재구성): 죽은 배치 = 10,125 ⇒ "같은 배치" ⇒ **H1 `REFUTED`**.

⇒ **같은 데이터에 등록 규칙의 두 허용 독법이 정확히 반대 라벨을 낸다.** 그리고 (R2)가 내는
라벨은, **이 프로젝트가 H1의 정본 증거로 삼고 있는 바로 그 관측**(감사 VERDICT §3-4 =
`CONFIRMED(scoped)`)을 "측정 실패"로 강등한다. 이는 감사 VERDICT §11이 명시적으로 막아 둔
방향 — *"게이트 #21의 **역**도 규율이다 — 진짜 게이트 실패를 측정 실패로 강등하지 마라"* — 의
재개방이다.

#### 2-4. F-c가 구제하지 못한다(확인)

F-c/채점기는 **라벨**은 구한다: TD가 어디서든 OOM하면 `request_errors > 0`,
`BOOT_FAILURES.txt` 생성, `B2_NO_CRASH` 실패 ⇒ `VERDICT FAIL`(907959에서 실증). 그러나
사전등록은 §6-3에 **"F-c는 라벨을 예보하지 않는다"**고 적었고, **H1의 `REFUTED`/`UNREALIZED`
처분은 F-a에만 붙어 있다.** 뒤집히는 것은 게이트 라벨이 아니라 **이 회차의 과학적 등록 판정**이다.

---

## 3. 반전 시험 표 (자유 표면 전수)

반전 계산에 쓴 원자료: `job_907959/{srv_L1,srv_L2,srv_TD1,srv_TD2}.log` · `tel_{L1,L2,TD1,TD2}.jsonl` ·
`verdict.txt` · `runtime_source_manifest.sha256`; 격자: 완주 `Prefill batch` 줄의
`(#new-seq, #new-token)` 열거 및 `prefill_active_batch_size` 전수 도수분포; 추정량: 사전등록
§6-1/§6-2 문안 그대로. **반전 계산 자신은 새 자유 표면을 만들지 않는다**(전부 도수 세기와
사전등록이 이미 고정한 상수 ε=1.00 GiB, 진단서가 이미 고정한 R(T) 계수만 사용).

| # | 자유 표면 | 민 범위(등록이 허용하는 끝까지) | 판정 변화 | 근거 수치 |
|---|---|---|---|---|
| S1 | **F-b `peak(boot)` 정의역이 ∅인 boot의 처분(미등록)** | (i) 제외 / (ii) 최대 배치로 대체 / (iii) `UNREALIZED` | ★**있음 — 死因 1** | L2의 `==14` 스냅샷 **0개** vs L1 **146개**. (i)↔(ii)는 Δ를 **0.236–0.413 GiB**(=ε의 23.6–41.3 %) 이동 ⇒ Δ∈[1.00,1.41]에서 **B→A**, Δ∈[−1.41,−1.00]에서 **D→A**. (iii)↔(i)는 **"대역 라벨 없음"↔"대역 발화"** |
| S2 | **F-a "같은 배치에서 OOM인가"의 판정 채널(미등록)** | (R1) 다채널 재구성 / (R2) 완주 로그 줄 부재 = 미형성 | ★**있음 — 死因 2** | 907959 TD: 10125 줄 **0건**, `==14` 스냅샷 **0개**, 완주 최대 **6,245** < 10,036 ⇒ (R2)는 **`UNREALIZED`(H1 실패 아님)**, (R1)은 **H1 `REFUTED`** |
| S3 | F-b `e*` 선택("복수면 그런 스냅샷 수가 가장 많은 epoch") | 복수 epoch 강제 시도 | **없음** | `_r2_reset_memory_peak`는 `update_split_prefill_batch:1235` → `_dual_worker_start_prefill:870`에서만 호출되고 `:1219-1220`의 `if self.split_prefill_batch: return False`가 배치당 1회를 보장 ⇒ 한 배치 = 한 epoch, 복수 불가 |
| S4 | F-b `ε = 1.00 GiB` | 0.156 GiB(최대 식별 TD 전용 성분) ↔ 6.30 GiB | **없음(자유 표면 아님)** | ε는 실행 전 수치로 **고정**돼 있고 RR-5가 "측정 아님"을 이미 등록. 다만 대역 흡수율은 caveat C7 |
| S5 | F-b 대역 B 폭 (1.00–6.30 GiB) | — | **없음(고정)** | F-a 충족 시 도달 가능 Δ 상한 ≈ `11.59 − peak_L` ≈ 8.1–9.6 GiB이므로 B가 도달 가능 구간의 **약 62 %**를 "미확정"으로 흡수. 고정값이라 반전 아님 → caveat C7 |
| S6 | F-c 술어 (`request_errors==0`, `decode T/F=n/0`, `BOOT_FAILURES` 부재) | 전부 밀어봄 | **없음** | 907959에서 전부 발화·구별력 실증(`L1/L2=0`, `TD1/TD2=31`; `BOOT_FAILURES.txt` 2줄). 출력공간에 거짓 존재 |
| S7 | F-d TD 실현값 절 | `PDMUX_MEM_TELEMETRY` 미전파 / 가드 env 누출 | **없음** | 거짓 도달 가능(계측 플래그 미전파 시 4키 `null`). `r2_correctness.sbatch:216`이 `PDMUX_*` 전부 unset하므로 env 누출 경로는 닫힘 |
| S8 | F-d legacy `null` 절 + `worker_grad_guard=="inference_mode"` 절 | — | **없음(처분 미부착)** | 코드상 항등식이나 **처분이 붙어 있지 않다**(기록 항목). N1 미발화 → caveat C6 |
| S9 | 변이 시험의 공허성 | 감사자가 새 변이 4종 직접 주입 | **없음(3/4 caught)** | M-C(가드를 텔레메트리 플래그로 게이팅) caught · M-D(PREFILL만 가드) caught(3건) · M-K(가드 진입 후 yield 전 탈출) caught(7건) · **M-J(실현값→설정값 에코) ESCAPED** → caveat C5 |
| S10 | 리셋 지점이 F-b 정의역을 바꾸는가 | 스냅샷마다 리셋 / 배치마다 리셋 | **없음** | 코드 확인: 배치당 1회, 배치 내 단조 비감소 보장. 단 TD에서만 in-flight decode를 자를 수 있음 → caveat C2 |
| S11 | 3축 동시 이동(engine hash · 하네스 sha · INSTR 1→0) | "grad guard가 원인" 귀속 시도 | **없음** | §10 + RR-2가 귀속을 **사전 금지**. 귀속에 걸린 등록 판정이 0이므로 뒤집을 라벨이 없다 → caveat C8/C9 |
| S12 | 관측자 효과 산정 | 실측으로 밀어봄 | **없음(판정 미포함)** | 재현: flatten+sort 52.19 µs / 122 leaf / 3키 0.146 µs. 잔여 C++ 1회는 파이썬 구성 프록시 **4.18 µs ⇒ 벽시계 0.09 %**로 상한. ±3 % 예산 결론 유지 → caveat C1 |
| S13 | 라인 인용 17건 "번호만 이동" | 지문 다중집합 대조 | **없음** | 문서 3종 전부 `(sha, anchor, anchor_offset, target)` 다중집합 동일, in-place 변경 0 |
| S14 | manifest 파급 | 25항목 전수 재해시 | **없음** | 24 불변 / 1 이동 / 0 결손 |
| S15 | 예산·`--time` | 구조적 최악까지 재산정 | **없음** | 6,680 s = 111.3분 < 150분 |
| S16 | 승계 33건의 실재성 | 이름-정본 대조 | **없음** | NP-1′…NP-10 10건 · NPC-A…NPC-J 10건 · (D13-i) · N-1…N-13 13건 **전부 sha 핀 원문에 실재** |
| S17 | §11.2 "원문 그대로 재수록" | 문자 단위 diff | **없음(판정 미영향)** | 3/3 **절단**. §11.1이 sha 핀 원문 전사를 의무화하므로 판정 불변 → caveat C4 |
| S18 | D5 재제출 정책 | `FAIL` 덮어쓰기 시도 | **없음** | §3이 "**`FAIL`·`INCONCLUSIVE`·`NO_VERDICT_UNREALIZED`는 재제출로 덮어쓰지 않는다**"를 무수정 승계 |
| S19 | `R2C_INSTRUMENT=0`이 §13-D1(i)를 남기는 것 | "다음 회차에 조용히 사라지는가" | **없음(이 회차 한정)** | §7 + RR-7이 미이행을 **명시 등록**. 단 다음 회차 승계 의무는 미등록 → caveat C10 |
| S20 | `INSTR=0`의 부작용(warm-up 캐시 예열 소실) | 밀어봄 | **없음(판정 미포함)** | `r2_correctness.sbatch:175` `TRITON_CACHE_DIR="$OUT/.triton_cache"`는 **job마다 새로 만들어진다**. `:334` 확인 결과 warm-up boot 자체는 무조건 뜨지만 I2/I3 부하가 사라져 예열이 줄어든다 → caveat C3 |

**반전 2 / 자유 표면 20.**

---

## 4. 반증 실패 공시 — 깨려다 실패한 것

아래는 **반증을 만들지 못했다**. 死因이 아니며 caveat로도 강등하지 않는다(즉, 건강하다).

**4-1. 처치의 기전 주장.** `event_loop_pdmux`는 `:1239`에서 `@torch.inference_mode()`이고
그 가드는 thread-local이다. true-dual의 prefill은 `:1454-1466` `_run_prefill`이
`RoleWorkerThread`로 넘어가고, decode도 같은 경로다. `dual_worker.py:_loop`의
`with self._activate(task.context):`가 유일한 진입점이며, `_activate_role_context`는
`multiplexing_mixin.py:220`에서 `TrueDualWorkerRuntime`에 **단 한 번** 주입된다(다른 활성화
경로 0건, grep 전수). 수리 후 worker task는 legacy 메인 루프와 **정확히 같은 가드** 안에서 돈다.

**4-2. `_activate_role_context`를 우회하는 모델 forward 경로 — 찾지 못했다.**
`run_batch` 호출점 전수(`:1404`, `:1468`, `:1885`, `:1939`)를 확인했다. `:1468`/`:1404`는
true-dual 분기의 `else` (legacy 인라인), `:1885`/`:1939`는 `event_loop_pdmux_coord`(`:1546`,
역시 `@torch.inference_mode()`) 안이다. TD가 실제로 쓰는 루프는 `event_loop_pdmux`이며 legacy와
**같은 루프**다(907959 traceback이 `_run_prefill`을 가리키고, 그 함수는 이 루프 안에만 있다).

**4-3. `inference_mode` 안전성 — 5개 주장 전부 torch 2.9.1에서 재현.**

```
pool(부팅 시 보통 텐서)에 inference mode 안에서 in-place 쓰기 -> pool.is_inference() == False
보통 텐서의 view를 inference mode 안에서 생성            -> view.is_inference() == False
inference mode 안에서 생성한 텐서                         -> is_inference() == True
inference tensor를 밖에서 in-place                        -> RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.
inference tensor를 밖에서 autograd에 투입                 -> RuntimeError: Inference tensors cannot be saved for backward.
inference tensor를 안에서 in-place                        -> OK
직전 task에서 만든 inference tensor를 새 가드 진입 후 소비 -> OK
```

소비자 전수 확인(감사자 코드 읽기): `prefill_future.result()`(`:1504`) ·
`decode_future.result()`(`:1490`) 둘 다 데코레이트된 루프 본문 · KV/mamba 풀과 cudagraph static
buffer는 부팅 시 할당된 보통 텐서(위 첫 줄이 안전을 보장) · `:1520-1526`의
`torch.ones(1, device="cpu")` → `tp_cpu_group.allreduce(flags, ...)` in-place 집합통신도 루프
안(legacy가 이미 같은 일을 한다) · `RoleWorkerThread._loop`의 `future.set_result(result)`는
가드 밖이지만 참조 저장뿐(텐서 연산 0) · 텔레메트리 `_write_dual_worker_trace`(`:817-863`)는
`asdict(...)`와 파이썬 스칼라만 직렬화 · watchdog(`managers/scheduler_runtime_checker_mixin.py:396-397`)은
`batch_size()`/`reqs` repr만 만지고 타임아웃 시에만 발화. **밖에 있는 소비자를 찾지 못했다.**
결정적 논거는 arm 대칭이다: 수리 후 TD가 하는 일은 legacy가 이미 4개 GPU job 동안 해 온 일과
같은 가드 아래의 같은 연산 집합이다.

**4-4. task마다 새 guard 인스턴스 — 필요하고 충분하다.** `torch.inference_mode` 인스턴스는
`__enter__`에서 상태를 `self`에 저장하므로 공유 인스턴스를 두 스레드가 동시에 진입하면 경합한다는
주석의 주장은 맞다. 사전등록의 `test_two_role_threads_hold_the_guard_at_once`(배리어로 동시 보유
강제)가 이를 pin하고 재현된다. 생성 비용은 파이썬 객체 1개/task이며, task 빈도는 스냅샷 빈도보다
낮으므로 §2.3의 관측자 효과 예산에 실질적으로 들어가지 않는다.

**4-5. 하네스 대칭.** `PDMUX_MEM_TELEMETRY="$MEMTEL"`이 `:478`의 **arm 무관 공통 env 블록**에
있고 `TD_ENV`와 분리돼 있다(`:465-466`). 리셋 지점은 `update_split_prefill_batch`의 무조건
경로라 양 arm이 같은 사건에서 리셋한다. 플래그 OFF면 `_r2_memory_fields`가 `{}`를 반환해
수리 이전과 키 집합이 동일하다(테스트로 pin, 재현).

**4-6. 채점 규칙 무영향.** `SNAP_KEYS` 충돌 0 · 플래그 ON에서 `SNAP_KEYS` 전부 생산 —
테스트 재현. 채점기 sha 불변 재확인.

**4-7. 소급 주장(§1.3 / RR-3)의 정확성 — 정확하다.** Zamba2-2.7B `mamba_ngroups=1`을
체크포인트 config에서 직접 확인했고, `mixer2_rms_norm_gated.py:109-110`의 분기 조건이 그 모델에서
커널 경로(`:114`, `.data`)를 타게 함을 확인했다 ⇒ retention 기전에 구조적 면역. 그리고 진단서
§9-1이 등재한 사실(**"worker가 grad 켜진 채 돈다"는 성질 자체는 그 job들에도 있었고 autograd
부기의 arm 비대칭 오버헤드는 크기 미측정**)이 §1.3 두 번째 bullet과 RR-3에 **빠짐없이** 실려
있으며 "오염됐다"도 "깨끗했다"도 금지하고 있다. **누락 없음.**

**4-8. 승계 33건.** NP-1′·NP-2·NP-3′·NP-4·NP-5·NP-6·NP-7′·NP-8·NP-9·NP-10,
NPC-A…NPC-J, (D13-i), N-1…N-13이 sha 핀 원문에 **전부 실재**한다. N-7·N-8·NP-8의 §11.2 재수록은
절단이 있으나(caveat C4) 원문이 sha로 고정돼 있어 의무는 이행 가능하다. D5 재제출 정책은
문자 그대로 승계돼 있다.

---

## 5. 등록 caveat — 결과 문서·정본이 **문자 그대로** 승계할 인용 금지 문장

> **C1 (계측 비용 수치)** — 사전등록 §2.3의 **"0.14 µs"는 계측기의 비용이 아니다.** 감사자 재측정
> (torch 2.9.1, 같은 venv): `torch.cuda.memory_stats_as_nested_dict()`는 **CUDA 미초기화 경로에서
> `{}`를 0.105 µs에 반환**하며 그 경우 세 키 읽기는 실행되지 않는다. 0.14 µs에 대응하는 실측은
> **이미 만들어진 nested dict에서 3키를 읽는 비용(0.146 µs)**이고, 기각된 철자의 54.3 µs는
> **populated dict에 대한 flatten+sort(재현 52.19 µs, leaf 122개)** 다 — **두 수는 서로 다른 입력에서
> 측정됐다.** 남는 `torch._C._cuda_memoryStats` C++ 호출 1회는 **여전히 미측정**이며, 동등한 nested
> dict를 파이썬으로 구성하는 프록시(4.18 µs = 222.5 snap/s에서 벽시계 **0.09 %**)로만 상한이 잡힌다.
> **±3 % 예산 결론은 살아남지만 "0.14 µs"라는 표현은 인용할 수 없다.**

> **C2 (계측기 자신의 잔존 arm 비대칭)** — `torch.cuda.reset_peak_memory_stats()`는 스케줄러
> 스레드에서 발행되므로 **true-dual에서는 다른 스레드의 decode forward 도중에 떨어질 수 있고,
> legacy에서는 (직렬화돼 있어) 그럴 수 없다.** 즉 TD 쪽 피크 창만 in-flight decode의 일부를
> 잃는다. **방향은 TD를 과소 보고하는 쪽 = 대역 A("F-b 충족") 쪽으로 편향**이며, 크기는 decode
> transient(cudagraph private pool 156 MiB = ε의 15.6 %) 이하로 상한이 잡힌다. `Δ`를 인용할 때
> 이 편향 방향을 반드시 병기하라.

> **C3 (`INSTR=0`의 미등재 부작용)** — `R2C_INSTRUMENT` 1→0은 §7이 든 세 이유 외에
> **warm-up boot의 클라이언트 부하(I2/I3)를 제거**한다. `r2_correctness.sbatch:175`의
> `TRITON_CACHE_DIR="$OUT/.triton_cache"`는 **job마다 새로 생성**되므로, 907959에서는 warm-up이
> 예열해 둔 JIT 캐시가 이 회차에서는 덜 채워진 채 첫 scored boot이 돈다. 이는 C층 배치 형성
> 타이밍을 흔들며, 그 배치 형성이 바로 F-a의 1차 술어와 F-b의 정의역이 의존하는 양이다.
> **"907959와 같은 조건에서 배치가 형성될 것"이라고 쓸 수 없다.**

> **C4 (§11.2의 "원문 그대로"는 사실이 아니다)** — 재수록 3건 모두 절단됐다.
> **N-7**은 말미 `"(λ0 rev1 판정서 :151 \"B=1은 pdmux 분할 미적용 구간\"이 옳았다.)"` 누락 —
> 정정의 출처가 사라진다. **N-8**은 말미 `"(사전등록 D4가 상류 병목 단정을 금지)"` 누락 —
> **이것이 상류 병목 귀속을 막는 바로 그 조항이다**(confound #6 계열). **NP-8**은 `"(신규, D13)"`
> 태그 누락. 결과 문서는 §11.1이 지시한 대로 **sha 핀 원문에서** 전사하고, §11.2를 전사 원본으로
> 쓰지 말 것.

> **C5 (실현값 채널에 음성 대조가 없다)** — 감사자 변이 **M-J**(`_r2_record_worker_guard`의
> `torch.is_grad_enabled()` / `torch.is_inference_mode_enabled()`를 설정값 에코
> `_g == "none"` / `_g == "inference_mode"`로 교체)는 **등록된 13개 테스트 전부를 통과한다.**
> 제출된 코드는 실현값을 읽고 있어 결함이 아니지만, **F-d의 실현값 성격은 파일 단위 manifest
> 해시로만 보호되고 변이 대조로는 보호되지 않는다.** 게이트 #176의 충족을 "테스트로 고정됐다"고
> 쓸 수 없다. (같은 주입에서 M-C[가드를 텔레메트리 플래그로 게이팅]·M-D[PREFILL만 가드]·
> M-K[가드 진입 후 yield 전 탈출]은 전부 잡혔다.)

> **C6 (F-d의 두 절은 항등식이다)** — `worker_grad_guard == "inference_mode"`는 하네스가
> `PDMUX_*`를 전부 unset(`r2_correctness.sbatch:216`)하므로 **거짓이 될 수 없고**, legacy의
> realised 4키 `null`은 **worker 스레드가 없다는 것의 연역**이다. 둘 다 **처분이 붙어 있지 않은
> 기록 항목**이므로 N1은 발화하지 않지만, **"F-d가 충족됐다"를 정보량 있는 확인으로 인용할 수
> 없다.** F-d에서 실제로 거짓이 될 수 있는 것은 TD의 realised 2키뿐이고, 그 거짓은
> "설치 트리가 수리본이 아니다"가 아니라 **"계측 플래그가 전파되지 않았다"**로도 발생한다.

> **C7 (F-b 대역 B의 흡수율)** — F-a가 충족된 세계에서 도달 가능한 Δ의 상한은
> `11.59 GiB − peak_L` ≈ **8.1–9.6 GiB**(진단서의 legacy working-set 2.0–3.5 GiB 기준)이며,
> "미확정"으로 등록된 대역 B(1.00–6.30 GiB)가 그 구간의 **약 62 %**를 흡수한다. 즉 **F-a 충족
> 조건부로 F-b가 H1에 불리한 판정을 낼 여지는 좁다.** RR-5와 함께 인용하라.
> 보조 산출물인 `gpu_mem_allocated_b` 대 `prefill_chunk_progress` 곡선도 **"곡선"이 아니다** —
> 907959 L1의 `==14` 스냅샷 146개 중 **progress 6·12·…·54는 각각 정확히 2개**, 나머지 **128개가
> 종단 progress 56**에 몰려 있다. 게다가 스냅샷 발행률이 arm마다 다르다(L1 12,511/56.0 s = 223.4/s
> vs TD1 7,611/49.3 s = 154.4/s, **1.45×**). 피크는 단조성 덕에 면역이나 **순간값 곡선의 arm 비교는
> 표본율 교락을 안는다.**

> **C8 (3축 동시 이동 — 귀속 불가는 그대로)** — §10과 RR-2가 등록한 대로, F-a가 충족돼도
> "grad guard가 원인이다"는 이 job 단독으로 성립하지 않는다. 감사자는 **같은 엔진 안의
> `PDMUX_WORKER_GRAD_GUARD=none` arm을 이번 회차의 D조건으로 달지 않기로 판정한다.**
> 근거(실현가능성 검사, 게이트 #113): `r2_correctness_check.py:366`이 `boots.txt`에서 라벨을
> 열거하고 `:378`/`:446-447`이 `startswith("TD")`/`startswith("L")`로 arm을 가르므로, `none` boot을
> **scored boot으로 넣으면 TD arm에 합류해 설계상 반드시 크래시 → 게이트가 무조건 `FAIL`**이 된다.
> 유일하게 성립하는 형태는 **비채점 진단 boot**(warm-up 패턴)이며, 비용은 907959의 조기 사망 TD
> boot 실측 1.3분 기준 **≈1.5분 ≈ 0.025 GPU-h**(사전등록이 추정한 0.1 GPU-h가 아니다). 그러나 그
> 형태조차 **하네스 변경 + 테스트 + argv 픽스처 재고정**을 요구해 §7-2("처치 축을 하나로 유지한다")를
> 스스로 깬다. ⇒ **별도 회차로 남긴다.**

> **C9 (§10 지지 문장의 정의역)** — §10이 허용한 유일한 문장("…**같은 크기의** split-prefill 배치를
> 완주했다")은 **10,125 토큰 배치가 형성될 때에만** 쓸 수 있다. 907959의 legacy 2 boot 중 1 boot이
> 그 배치를 형성하지 않았으므로(L2: 9/7345, 16/12516), **형성되지 않았을 때 쓸 대체 문장이 등록돼
> 있지 않다.** 등록되기 전에는 그 경우 **어떤 서술 문장도 쓸 수 없다.**

> **C10 (RR-7의 승계 사슬)** — RR-1…RR-7은 §11.1의 구조상 **이 회차의 결과 문서까지만** 의무화돼
> 있다. **다음 회차 사전등록이 RR-7(감사 §13-D1(i) 미해결)을 승계하도록 강제하는 조항이 없다.**
> 다음 사전등록은 이 문서를 sha로 핀하고 RR-1…RR-7을 §11.1 목록에 명시적으로 편입해야 한다.

---

## 6. 두 死因을 제거하는 정확한 수리 (실행 전, **GPU 0**, 문서 2절 텍스트 편집)

게이트 #113 자기 적용: 아래 처방의 실현가능성·잔여 재량·비용을 함께 적는다.

### R-1 — F-a 재정의 (死因 2 제거)

> **(F-a1, 1차 판정 — 배치 크기 무관)** TD 2/2 boot이 **형성한 모든** split-prefill 배치를
> `torch.OutOfMemoryError` 없이 완주했다. 채널: `srv_TD*.log`에 `torch.OutOfMemoryError` **0건**
> AND `Scheduler hit an exception` **0건** AND `BOOT_FAILURES.txt` 부재 AND
> `per_boot.TD*.checks.request_errors == 0`. **이 술어가 거짓이면 F-a는 충족되지 않으며, 그것은
> `UNREALIZED`가 아니라 H1이 이 회차에서 지지되지 않았다는 뜻이다.**
> **(F-a2, 2차 — 힘)** TD가 완주한 최대 `#new-token`이 **907959의 TD 천장 6,245를 넘었다.**
> 넘지 못했는데 F-a1이 참이면 **F-a2만 `UNREALIZED`**(부하가 약해 검정력이 없었다).
> **(F-a3, 기록만)** `#new-seq: 14, #new-token: 10125` 줄의 출현, `prefill_active_batch_size==14`
> 스냅샷 수, 크래시가 있었다면 `Tried to allocate N MiB`에서 역산한 `T = N_bytes / (10240 × 2)` —
> **전부 보고 항목이며 판정에 들어가지 않는다.**

**왜 이것이 死因을 없애는가**: (a) 완주-최대-토큰과 OOM 계수는 **모든 TD boot에서 항상 정의된다**
(정의역 보장), (b) "어느 배치에서 죽었는가"를 묻지 않으므로 미등록 재량이 사라진다,
(c) 강등 통로가 닫힌다 — 907959의 TD 데이터를 먹이면 F-a1이 **거짓**이 되어 정본과 같은 방향을
낸다(재검증: `srv_TD1.log`/`srv_TD2.log`에 `torch.OutOfMemoryError` 각 1건, `BOOT_FAILURES.txt` 2줄,
`request_errors` 각 31).
**잔여 재량**: F-a2의 문턱 6,245는 측정된 상수(907959 TD 완주 최대)이므로 재량 0.
**비용**: 텍스트 편집, GPU 0, 새 도구 0.

### R-2 — F-b 재정의 (死因 1 제거)

> **(a) ∅ 처분 등록** — 4 boot 중 하나라도 `prefill_active_batch_size == 14` 스냅샷이 0개면
> **F-b(10,125 축)는 `UNREALIZED`이며 대역 라벨을 내지 않는다.** (907959 L2에서 실제로 0개였다.)
> **(b) 1차 추정량을 정의역 보장형으로 교체** — **4 boot 전부가 완주한 29개 토큰 수**
> `{1, 6, 7, 9, 17, 24, 34, 54, 55, 75, 183, 368, 371, 453, 732, 1113, 1255, 1280, 1409, 1468,
> 1499, 1672, 1839, 1845, 2019, 2187, 2197, 2310, 3241}` (907959 원자료에서 감사자가 확인;
> 이 열거 자체를 등록 상수로 고정한다) 각각에 대해 그 배치의 epoch 마지막 표본을
> `peak(boot, T)`로 잡고, boot별로 `peak` 대 `T`의 **OLS 기울기·절편**을 낸다.
> **판정**: `|slope_TD − slope_L|`을 진단서의 수리 전 예측 기울기 **1.2750 MiB/token**과 비교해
> 실행 전 문턱(예: `0.3 × 1.2750 = 0.3825 MiB/token`)을 등록한다. T=1→3241 구간에서 수리 실패 시
> 예상 신호는 **4.04 GiB = ε(1.00 GiB)의 4배**이므로 검정력이 10,125 축보다 크다.
> **(c) 병기 필수** — 29개 점 각각에 `stream_index`를 함께 기록한다. 다수가 prefill-only(비분할
> idx 0) 구간이므로 **이 추정량은 D44 운영점의 값이 아니다**(감사 N-7 계열).

**잔여 재량(정직하게)**: (b)의 문턱 0.3825 MiB/token은 **ε와 같은 성격의 사전 판단**이다.
이 처방은 **판단을 없애지 않고 ∅ 정의역을 없앤다** — 死因은 후자였다.
**비용**: 같은 필드·같은 실행·분석 스크립트 1개. **GPU 추가 0.**
**실현가능성 검사**: 29개 토큰 수는 4 boot 전부에서 완주 로그 줄로 존재함을 감사자가 확인했고,
`gpu_mem_peak_allocated_b`·`gpu_mem_peak_epoch`·`prefill_active_batch_size`는 이미 구현된
필드다. 새 계측 0.

### R-3 — caveat C4의 기계적 수리 (선택, GPU 0)
§11.2의 N-7·N-8 말미 절단 2곳을 sha 핀 원문에서 복원.

---

## 7. 재감사 시 확인할 것 (게이트 #110 대비)

이 판정서의 어떤 수치도 다음 회차의 **등록 상수로 승격하지 마라**. 특히
`L2의 ==14 스냅샷 0개`, `29개 공통 토큰 수`, `TD 천장 6,245`, `1.2750 MiB/token`,
`52.19 µs / 0.146 µs / 122 leaf`, `이동 키 17`은 전부 `job_907959/` 원자료와 현 트리에서
**재계산 가능**하며, 다음 감사자는 재계산해야 한다.

---

## 8. 신규 게이트 후보 (번호는 정본이 부여)

1. **추정량의 정의역이 비는 경우의 처분을 반드시 등록하라 — 그리고 그 ∅가 기존 원자료에서
   실제로 일어났는지 먼저 확인하라.** F-b는 `max{}` over ∅를 4 boot 중 1 boot에서 이미 겪었을
   구성이었고, 사전등록은 F-a에는 `UNREALIZED` 분지를 두고 F-b에는 두지 않았다.
2. **"같은 X에서 실패했는가"를 판정에 쓰려면 X를 식별하는 채널을 등록하라.** 이 엔진에서
   `report_prefill_stats`는 **완주 시에만** 로그를 남기므로, 죽은 배치는 등록된 채널에 존재하지
   않는다. 결과 감사가 그 배치를 동정하는 데 쓴 4채널 재구성은 **사전등록에 한 줄도 없었다.**
3. **측정 실패 분지는 "처치가 실패한 결과로 그 배치가 안 생긴 경우"를 배제하도록 써라.**
   게이트 #21("측정 실패를 게이트 실패로 라벨하지 마라")의 **역방향 남용 통로**다 —
   감사 VERDICT §11이 라벨에 대해 닫았던 통로를 이 사전등록이 H1 판정에 대해 다시 열었다.
4. **같은 arm의 두 boot이 같은 부하를 같은 배치로 나눈다고 가정하지 마라.** job 907959의
   legacy 2 boot은 같은 코드·같은 seed·같은 클라이언트에서 C층 배치를 다르게 형성했다
   (L1 `8/6245, 14/10125, 3/3491` vs L2 `9/7345, 16/12516`). **배치 구성을 술어에 넣는 모든
   사전등록은 이 비결정성을 정의역 가정으로 등록해야 한다.**
5. **긍정 사례 — 계측기를 base payload에 넣어라.** 이 사전등록은 메모리 필드를
   `runtime.metrics()`(true-dual 전용)가 아니라 base payload에 넣어, 비대칭을 재는 계측기가 스스로
   비대칭을 만드는 것을 막았다. 다만 **리셋 발행 지점이 여전히 스케줄러 스레드**라 잔존 비대칭이
   남는다(caveat C2) — "대칭 계측"은 **필드 위치와 발행 스레드 양쪽**을 봐야 한다.
6. **변이 대조는 "실현값 채널이 설정값 에코로 퇴화하는 변이"를 반드시 포함하라.**
   감사자 변이 M-J가 13개 테스트를 전부 통과했다. 게이트 #176(realized ≠ target)을 테스트로
   고정했다고 쓰려면 이 변이가 실패해야 한다.

---

## 9. 제출 가능 상태

> **제출 불가.** 등급 `NO-GO`(死因 N2 ×2). §6의 **R-1·R-2 두 절을 사전등록에 반영한 rev2**를
> 만든 뒤 **규칙층 재감사**를 거쳐야 제출할 수 있다. 코드·하네스·테스트·manifest·라인 인용·
> 예산·승계 목록은 **고칠 것이 없다**(§1·§4). GPU 추가 비용 0으로 수리 가능하다.
>
> **새 성능 판정 0건.** HE0 · 정책 순위 · stake #1 · 게이트 #13/#16 · Claim D 등급(미검증) ·
> Zamba2/triton 동결 — 전부 불변. **GPU 신규 지출 0**(기존 아티팩트 + CPU만).
> `results/r2_eval/lambda0_prereg/**` 미접촉. R2 correctness 트랙 장부 **0.786 GPU-h 불변**.

---

### 관련 파일 (절대경로)

- 감사 대상: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/rerun_prereg/PREREG_RERUN_2026-09-13.md`
- 수리 코드: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/src/multiplex/multiplexing_mixin.py` (`:697-741` `_activate_role_context` + `_r2_record_worker_guard`, `:540-620` `_r2_memory_fields` / `_r2_reset_memory_peak`, `:865-870` 리셋 호출, `:1218-1237` `update_split_prefill_batch`, `:1239` / `:1545` 두 루프의 `@torch.inference_mode()`, `:1454-1466` `_run_prefill` submit, `:1490` / `:1504` 수확)
- 워커 루프: `/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/multiplex/dual_worker.py` (`_loop`의 `with self._activate(task.context):`)
- 하네스: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/r2_correctness.sbatch` (`:175` TRITON_CACHE_DIR, `:201` INSTR, `:210` MEMTEL, `:216` PDMUX unset, `:334`/`:366` warm-up 게이팅, `:462-479` arm 루프)
- 채점기(무수정): `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/r2_correctness_check.py` (`:366` boots.txt 열거, `:378`·`:446-447` arm 분류)
- 테스트: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/tests/test_worker_grad_guard.py` · `.../tests/test_mem_telemetry_symmetry.py`
- 반전 근거 원자료: `/scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results/r2_correctness/job_907959/` (`srv_L1.log`·`srv_L2.log`·`srv_TD1.log`·`srv_TD2.log`·`tel_L1.jsonl`·`tel_L2.jsonl`·`tel_TD1.jsonl`·`tel_TD2.jsonl`·`verdict.txt`·`BOOT_FAILURES.txt`·`runtime_source_manifest.sha256`)
- 승계 문서(sha 핀, 6/6 검증): `.../newpair_prereg/PREREG_NEWPAIR_2026-09-13.md` · `.../newpair_prereg/VERDICT_newpair_rules_2026-09-13.md` · `.../newpair_prereg/VERDICT_newpair_rev2_2026-09-13.md` · `.../audit_907959_2026-09-13/VERDICT.md` · `.../audit_907959_2026-09-13/DIAGNOSIS_true_dual_oom.md`
- 크래시 지점(이번 회차 무수정, manifest 밖): `/scratch/ehmoon/whlee/sglang_engine_dev/python/sglang/srt/layers/attention/mamba/mixer2_rms_norm_gated.py:97`
