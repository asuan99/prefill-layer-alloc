# 판정서 — Nemotron-H 층 타입 계측 이식 사전등록, **규칙층 1단계 감사**

2026-08-28 · claims-auditor(적대 감사, read-only) · 대상
`workspace/engine-port/results/nemotron_zt/PREREG_NEMOTRON_ZT_2026-08-28.md` (무수정) ·
**GPU 0 · job 제출 0 · 새 성능 판정 0건 · 등급 변경 0건 · 정책 순위 변경 0건 · 정본 편집 0건**

---

## 판정: **NO-GO**

> 사전등록이 스스로 물은 단일 질문:
> *"§3의 「같은 양이 아니다」 등록이 충분한가, 아니면 이 이식은 비교 불가능한 것을 비교하려는 것인가?"*

**직답: 둘 다 아니다. §3의 등록은 불충분하지만, 흔히 예상되는 이유(축 열거가 짧다) 때문이 아니다.
그리고 이 이식은 「비교 불가능한 것을 비교하려는 것」도 아니다 — 이식은 「실패할 수 없는 비교를
등록한 것」이다.**

§3은 축을 4개 등록한 뒤 비교 가능한 것을 **"부호와 L-의존 방향"** 하나로 좁힌다. 바로 그 좁힘이
死因이다. 방향 술어는 **§2가 막으려는 정본 결함 두 가지 어느 것에도 뒤집히지 않는다** — 이 감사가
기존 로그로 실측했다: 정본 결함 (a)(버킷 비대칭)·(b)(누산기 러닝평균)을 **동시에** 재현한 추정량
변종에서도 §5의 `DIFFA_OPENS` 술어가 **16/16 셀 전부 발화**한다(§1-F5). 즉 §2의 방지책과 §4의
정확성 게이트는 §5의 산출과 **인과적으로 분리**돼 있다: 이식이 정본 결함 2종을 글자 그대로
재생산해도 같은 라벨이 나온다.

한편 R2′가 실제로 요구하는 양(계수 `c_attn(L)`·`c_mamba(L)`, 지수, 교차점)은 §3·§7-2가 스스로
이전 금지했고, §5는 그것을 **추정할 설계를 등록하지 않았다**(L 2점, n 미등록, CI 없음 — §1-F6).
⇒ **estimand는 arm 내부에서 잘 정의되지만, 등록된 결정 규칙이 그 estimand를 산출하지 않는다.**

★**신규 게이트 #83**(`PROJECT_STATUS.md` 방법론 게이트 체계 / `CONSENSUS.md` §3 항목103)의
positivity 위반은 **백엔드 축에 대해서만** 성립하고, §3은 그것을 정직하게 등록했다 —
**그 부분은 통과다.** 죽는 것은 positivity가 아니라 **결정 규칙의 도달가능성**이다
(`PROJECT_STATUS.md` 방법론 게이트 #81, `SINGLE_LABEL_FORCED`의 연속층 판본).

---

## §1. 死因 (F#) — 전부 GPU 0으로 발견, 재현 명령 병기

### F1. `closure ≥ 0.95`(§2-a·§4-3)는 **이 저장소가 이미 "항등식"으로 실증 등재한 게이트**이고, prefill regime에서는 **발화 자체가 불가능**하다

`PROJECT_STATUS.md` 방법론 게이트 **#9** / `CONSENSUS.md` §3 **항목18**(항등식 게이트)의 재발.

**(a) 저장소가 이미 그렇게 적었다 — 두 곳, 독립적으로.**

1. `workspace/engine-port/results/r0c/P5_GATES_BATCH1_2026-08-05.md:33`
   (★UNAUDITED·정본 아님 — 인용 시 이 라벨 필수):
   > `closure` | `(attn+mlp+other+mamba)/fwd_ms` — this is an **identity** given the emitted fields
   > (verified: max deviation 5.2e-05 over 200 blocks); its content is entirely in the *independence*
   > of the denominator event pair
2. 같은 파일 `:234-236`:
   > **GATE C is blind to the defect that actually fired.** Starvation billed *inside* spans leaves
   > closure at 0.93–0.95 while inflating `per_mamba` by up to 2.78×. This run is the existence
   > proof: **20/20 GATE C PASS with 3–5/5 GATE N VIOLATION**. "Closure passed" must never be [trusted]

   ⇒ *"closure를 통과했으므로 층-타입 수치를 믿는다"* 는 추론에 대해 **저장소가 이미 존재증명을
   갖고 있다**. §2-a는 정확히 그 추론을 등록한다.

**(b) 문턱 0.95는 유도되지 않았고, 저장소의 유일한 교정된 문턱과 다르다.**
- 0.95의 실제 출처는 엔진 코드의 **경고 기본값**이다:
  `sglang_engine_dev/python/sglang/srt/models/zamba2.py:702`
  `_cmin = float(_os.environ.get("SGLANG_ZAMBA_CLOSURE_MIN", "0.95"))` — 그리고 거기서 그것은
  **중단이 아니라 `_log.warning("ZBLT_CLOSURE_FAIL ...")`**(`:706-714`)이다.
  사전등록은 이를 **하드 중단 게이트로 승격**하면서 출처도, 승격 사유도, 재교정도 적지 않는다
  (`PROJECT_STATUS.md` 방법론 게이트 **#35** — 유도를 바꾸면서 손잡이 값을 그대로 이월).
- 저장소에서 **실제로 채점에 쓰인** 유일한 closure 문턱은 **0.85**이며
  (`results/r0c/P5_GATES_BATCH1_2026-08-05.md:16,210,325`), 그 문서는 그 교정 범위를
  **"validated for defect 1 only"** 로 명시한다(`:230,:325`).
- 계측 코드 자신의 헤더가 재교정을 **요구**한다:
  `zamba2.py:76-79` — *"the threshold must be CALIBRATED against an observed run before it is
  treated as a defect detector -- a defect-1-style miss is ~0.2-0.5, event overhead is a few percent."*
  사전등록은 재교정 없이 다른 모델·다른 버킷 구성에 그대로 이식한다.

**(c) prefill에서 0.95는 발화할 수 없다 — 이 감사의 실측.**

```
cd /scratch/ehmoon/whlee/prefill-layer-alloc/workspace/engine-port/results
grep -rl "ZBPT2\|ZBLT2" --include=*.txt --include=*.out --include=*.log . > /tmp/f.list
# 각 줄에서 (?<![_a-z])closure=([0-9.]+) 추출, 태그별 집계
```
| 태그 | regime | n | min | p10 | median | max | frac < 0.95 |
|---|---|---|---|---|---|---|---|
| `ZBPT2` | **prefill** | **26** | **0.9939** | 0.9939 | 0.9986 | 0.9998 | **0.000** |
| `ZBLT2` | decode | 553 | 0.7119 | 0.9411 | 0.9912 | 0.9992 | 0.166 |

정본 `Diff A`는 **prefill 양**이다(`ZBPT2`, job 896776 로그 `ctxlen=7106 bs=1 ntok=7106`).
그 regime에서 관측 26/26이 문턱보다 **최소 0.044 위**에 있고 0.95 미만은 **0건**이다.
⇒ `§4-3`은 등록된 regime에서 **검정을 수행하지 않으며**, `§5`의 `CLOSURE_FAIL`은 **도달 불가 라벨**이다.

**(d) Nemotron 구성에서는 항등식성이 더 강해진다 — 그리고 저장소가 이미 그렇게 적었다.**
`workspace/engine-port/reports/PER_LAYER_TYPE_SM_RESPONSE_DESIGN_2026-08-11.md:87` 표의
NemotronH 행: 버킷 = *"`M`/`-`/`*` (층 전체 span, **루프를 타일링**)"*, closure 게이트 = **"없음"**.
Zamba2의 closure가 무언가를 검정할 수 있었던 이유는 코드 헤더가 명시하듯 **"deliberately NOT a
timeline partition"**(`zamba2.py:62-67`)이기 때문이다. 사전등록 §2-a는 Nemotron 버킷을
**"층 전체(forward 진입~이탈)"** 로 정의한다 = 루프의 타일링 = **구성상 1**.

> **판정**: §4-3은 §2-a가 부여한 임무(버킷 비대칭 검출)를 **구조적으로 수행할 수 없다**.
> Nemotron에서 attn 층은 56층 중 **4층(7.1%)**이므로, defect-1형 누락을 그대로 재현해도
> closure는 0.95 아래로 내려가지 않는다.

---

### F2. §5 `DIFFA_OPENS`의 **"증가폭 ≥1.5×"** 는 출처가 없다 — 그리고 유일한 후보 출처는 이 사전등록 자신이 §7-2·§8·`deprecated_v2 §7.6-6`에서 **이전 금지한 수치**다

```
grep -rn "≥1.5\|>=1.5\|1\.5×" prefill-layer-alloc/workspace/engine-port/results/nemotron_zt/ \
                              prefill-layer-alloc/deprecated_v2/
# → 유일한 히트: PREREG_NEMOTRON_ZT_2026-08-28.md:69 (자기 자신)
```
저장소 전체에서 이 문턱의 근거 문장은 **0건**이다. 유도도 없고 provenance 문자열도 없다.

가장 가까운 후보는 `deprecated_v2/README.md` §3의 job 896776 재측정 표
(L2000 Diff A **≈1.48–1.56**)인데, 그 표는 §7.6-6이 **직접 비교 금지**로 등재한 수치다.
만약 1.5가 거기서 왔다면 두 가지가 동시에 성립한다:
- **자기 금지 위반** — §8이 *"Nemotron Diff A가 정본보다 크다/작다(수치 비교 금지)"* 를 금지해 놓고,
  **판정 라벨을 결정하는 문턱**을 그 금지된 수치로 세운다. 라벨은 문턱의 함수이므로,
  라벨은 금지된 수치의 함수다(은폐된 이전 — confound 카탈로그 **#1**의 문서층 판본:
  다른 기판의 micro 수치를 실엔진 결론의 결정 규칙으로 승격).
- **단위 혼동** — 1.48–1.56은 L2000에서의 Diff A **수준**이지 L에 대한 **증가 배율**이 아니다
  (같은 표의 증가 배율은 ≈2.2×, 아래 F5 참조). 수준값을 배율 문턱으로 옮겨 쓴 것이라면
  숫자 자체가 범주 오류다.

★ 사전등록 §6이 쓰겠다고 선언한 `scripts/discipline/design_reachability.py`는 그 docstring
(`:28-29`)에서 **"Every restriction carries a provenance string; one without it is refused."**
라고 적는다 — 이 사전등록의 두 문턱(0.95·1.5×)은 그 요구를 **둘 다 충족하지 못한다**.

---

### F3. 사전등록은 **`nemotron_h.py`에 이미 층-타입 계측이 있다는 사실을 모른다** — 그리고 그 계측은 **이미 자기 물리 게이트에 실패한 전력**이 있는데 §4에 그에 대응하는 게이트가 없다

**(a) 계측은 이미 있다(코드 사실, 직독).**
`sglang_engine_dev/python/sglang/srt/models/nemotron_h.py` (= `workspace/engine-port/src/models/nemotron_h.py`,
`diff -q` **IDENTICAL**):
```
:646  _timing = _os.environ.get("SGLANG_NH_LAYER_TIMING") and _is_decode
:672  if _timing and getattr(self, "_lt_mode", None) != _mode:      # ← 모드 변경시에만 리셋
:673      self._lt_acc = {"M": 0.0, "-": 0.0, "*": 0.0}
:710      _s = torch.cuda.Event(enable_timing=True); _e = torch.cuda.Event(enable_timing=True)
:717      _e.record(); _evs.append((_pat_all[i], _s, _e))            # ← 층 전체 span, 타입별
:730  if self._lt_n % 30 == 0:                                       # ← 러닝평균 emit
:735  "NHLT mode=%s ctxlen=%d n=%d step_ms=%.3f | mamba=%.3f mlp=%.3f attn=%.3f | per-layer ..."
```
즉 사전등록 §1이 *"이식하면 된다"* 고 제안하는 구성(층 클래스 경계 = 버킷 경계, forward
진입/이탈 한 쌍)은 **이미 구현돼 있고**, 사전등록 §2-b가 피하겠다고 선언한 **defect (2)
러닝평균이 바로 거기 살아 있다**(`:672-674`·`:730-737`).

**(b) 저장소가 이 사실을 2026-08-11에 이미 표로 등재했다.**
`workspace/engine-port/reports/PER_LAYER_TYPE_SM_RESPONSE_DESIGN_2026-08-11.md`
- 한 줄 결론(`:36-40`): *"per-layer-type 계측은 **새로 만들 필요가 없다** — 세 hybrid 모델에
  이미 있고 dev 트리에 설치돼 있다. 문제는 계측의 **부재**가 아니라 **이미 판명된 오염**이다."*
- §3.A 검증표 **V1·V3·V4·V5**: 존재 / manifest 밖(수동 복사) / 러닝평균 / 층당 신규 Event 2개.
- §3.A **V10·V11**: 이 계측 계열이 job 873783에서 **GATE N(mamba decode의 ctx-불변성) FAIL**,
  기전은 host-pacing으로 **idle이 span에 청구**되는 것, `per_mamba` 최대 **2.78× 부풀림 + SM 비단조**.

**(c) §4의 여섯 게이트 중 이 실패를 잡는 것은 하나도 없다.**
§4는 (1) 출력 동일 (2) OFF 무비용 (3) closure (4) 층 수 (5) cudagraph (6) CPU 회귀다.
**물리 불변량 검사(GATE N류)·host-floor 대조·계측 자기비용 대칭 대조가 전부 없다.**
`P5_GATES` 존재증명(20/20 closure PASS ∧ 3–5/5 GATE N VIOLATION)이 말하는 바는 정확히
*"§4가 등록한 종류의 게이트만으로는 이 계측을 죽인 결함을 못 잡는다"* 이다.

⇒ confound 카탈로그 **#7**(micro-benchmark host/tenancy 아티팩트를 신호로 오독)의 **엔진 내부
판본**: host-paced forward에서 열린 idle이 device 시간으로 청구된다.

**(d) 파생 — 이미 있는 하드코딩 나눗수가 Nano-9B에서 틀리다.** `:737`
```python
self._lt_acc["M"]/n/24, self._lt_acc["-"]/n/24, self._lt_acc["*"]/n/4
```
`24/24/4`는 Nemotron-H-8B-Base-8K(52층) 시절 상수다. **Nemotron-Nano-9B-v2는 27/25/4**
(아래 §4-2에서 config 직접 계수로 확인). 이식이 이 줄을 상속하면 `per-layer mamba`가
**27/24 = 1.125× 부풀고**, attn/mamba 비가 정확히 그만큼 왜곡된다 — 조용히.

---

### F4. **forward mode가 등록되지 않았고**, §4-5(cudagraph-ON)는 저장소가 이미 **구조적 불가능**으로 검증한 것을 요구한다

**(a) 정본 `Diff A`는 prefill 양이다.** job 896776 로그가 `ZBPT2`(=`_pk` 분기,
`zamba2.py:673` `("ZBPT2" if _pk else "ZBLT2")`), `ctxlen=7106 bs=1 ntok=7106`.
**(b) 기존 Nemotron 훅은 decode 전용이다** — `nemotron_h.py:646` `... and _is_decode`.
**(c) 사전등록은 어느 쪽인지 한 글자도 적지 않는다.** §5는 "L 2000→8000"만 말한다.
prefill이면 기존 훅의 `_is_decode` 가드를 걷어내야 하고(=이식 범위가 §1의 서술보다 크다),
decode면 정본 Diff A와 **다섯 번째 축**(forward mode)이 추가된다.

**(d) §4-5는 판정 불능이다.** `PER_LAYER_TYPE_SM_RESPONSE_DESIGN_2026-08-11.md` §3.A **V12**
(직독 검증, 재현 좌표 병기):
> ★ **cudagraph-ON에서 per-layer span은 실행되지 않는다** — `cuda_graph_runner.py:547`
> `capture_forward_mode=ForwardMode.DECODE`; `:1161` `self.graphs[graph_key].replay()` ⇒
> replay는 Python forward를 **호출하지 않는다** ⇒ 계측은 **구조적으로 eager 전용**

그리고 같은 문서 `:39`: *"운영점(cudagraph-ON) per-layer 계측은 **구조적으로 불가능**하다(§3-V12)
— 이 캠페인은 정의상 off-operating-point다."*

⇒ §4-5 *"cudagraph-ON에서 통과 — **운영점이다**. 캡처가 깨지면 이식 실패로 본다"* 는
(i) decode 계측이면 **캡처가 깨지거나**(forward 안 `torch.cuda.synchronize()`)
(ii) 깨지지 않으면 **아무것도 방출하지 않는다** — 그리고 그 상태는 §5의 `INSTRUMENTATION_ABSENT`
라벨과 **구별되지 않는다**. 즉 §4-5와 §5가 서로 싸운다.
또한 정본 `Diff A`는 `CONSENSUS.md` §1-3이 *"Diff A/B를 인용하는 모든 정본 문장에 **no-cudagraph
micro** 라벨 필수"* 로 못박은 **비운영점** 수치다 ⇒ §4-5는 confound 카탈로그 **#8**을
**거꾸로** 밟는다(비운영점에서만 존재하는 계측에 운영점 통과를 요구).

---

### F5. §5의 **유일한 긍정 라벨이 설계로 강제**된다 — 그리고 §2가 막는 결함 어느 것도 그 라벨을 뒤집지 못한다 (실측)

사전등록의 방어는 *"§5는 방향만 본다"* 이다. 이 감사는 그 방어를 **기존 로그로 직접 반증**했다.

`ZBPT2` 로그 한 줄에는 네 가지 추정량이 동시에 들어 있다 — BLOCK(정상상태)·legacy 러닝평균
(=정본 결함 (b))·`attn_core`(= **pre-2026-08-04 attn 정의** = 정본 결함 (a)). 같은 런에서
네 변종으로 `Diff A`의 L2000→L8000 배율을 계산하면:

| 추정량 변종 | full@B1 | full@B4 | 24@B1 | 24@B4 | §5 라벨 |
|---|---|---|---|---|---|
| (i) BLOCK, whole-attn (정본 수리본) | 2.23× | 2.23× | 2.22× | 2.28× | **`DIFFA_OPENS`** |
| (ii) BLOCK, `attn_core` (**결함 (a) 재현**) | 3.75× | 3.77× | 3.74× | 3.92× | **`DIFFA_OPENS`** |
| (iii) legacy 러닝평균 (**결함 (b) 재현**) | 2.49× | 1.62× | 2.22× | 2.29× | **`DIFFA_OPENS`** |
| (iv) 러닝평균 × `attn_core` (**(a)+(b) 동시**) | 4.17× | 2.71× | 3.73× | 3.91× | **`DIFFA_OPENS`** |

**16/16 셀 전부 `DIFFA_OPENS`.** ★이 표는 **Zamba2 arm의 추정량 민감도 분석**이며
**Nemotron에 대한 예측도, 정본 수치와의 비교도 아니다**(`deprecated_v2 §7.6-6` 인용 제한 준수 —
정본 수치는 이 표에 등장하지 않는다). 이 표가 말하는 것은 오직 **결정 술어의 판별력**이다.

재현:
```
python3 - <<'PY'   # 정규식 1개, GPU 0
import re,glob
pat=re.compile(r'ZBPT2 mode=(\S+) .*?\| per-attn\((\d+)\)=([0-9.]+) per-mamba\((\d+)\)=([0-9.]+)'
               r' \| attn_core_total=[0-9.]+ per-attn_core\(\d+\)=([0-9.]+).*?'
               r'b_per_attn=([0-9.]+) b_per_mamba=([0-9.]+) b_per_attn_core=([0-9.]+)')
# workspace/engine-port/results/prefill_knee/kneebe_srv_L{2000,8000}_triton_896776.log
PY
```

**함의(치명적)**: 이식이 정본 결함 **2종을 글자 그대로 재생산해도** §5는 같은 라벨을 낸다.
⇒ §2의 방지책 3종과 §4의 정확성 게이트 6개는 **§5의 산출에 아무 영향이 없다**.
사전등록의 방어(*"방향만 보므로 안전하다"*)는 뒤집힌다 — **방향 술어는 결함에 강건한 것이 아니라
결함을 은폐한다**. 그리고 기전은 알려져 있다: 결함 (a)는 attn 버킷에서 O(L) 투영을 **제거**해
남은 O(L²) 코어만 남기므로 Diff A를 **더 빨리 열리게** 만든다(표의 (ii)·(iv)가 (i)·(iii)보다 큼).

⇒ `PROJECT_STATUS.md` 방법론 게이트 **#81**(도달가능성이 격자 안에서만 돌면 설계가 답을 미리
정한다) / **#40**(결정량 자체가 항등식) — 다만 여기서 강제는 **범주층이 아니라 연속층**(문턱)에
산다. 이것이 §6이 무력한 이유이기도 하다(아래 B7).

---

### F6. 등록된 측정은 **등록된 목적(R2′ 계수)을 산출할 수 없다**

§5는 스스로 적는다: *"이 측정은 R2′의 계수를 그 arm에서 재는 것"*.
`deprecated_v2/README.md:45-46`도 같은 요구를 등재한다: R2′ = `n_attn·c_attn(L) + n_mamba·c_mamba(L)`
⇒ **계수를 그 arm에서 다시 재야 한다.**

그런데 §5가 등록한 설계는 **L 2점(2000, 8000)**뿐이고, **n·반복·CI·warm-up 폐기 규칙·블록
정의·batch·SM 격자가 전부 미등록**이다. 결과:
- **"단조 증가"가 2점에서는 공허하다** — 2점 사이의 단조성은 그냥 "증가"다.
- **지수가 식별되지 않는다.** 정본 계수(attn ~L^1.916±0.021 · mamba ~L^0.954±0.008, R²≥0.9996)는
  다점 회귀 산물이다. 2점 = 자유모수 2개 = 잔차 0 = 항상 풀린다
  (`PROJECT_STATUS.md` 방법론 게이트 **#45** 계열: 식 개수 = 자유모수 개수인 검사는 항등식).
- **n 미등록은 `CLAUDE.md` 방법론 게이트 #3**(베이스라인 분산 먼저, n≥4 없이 정책 결론 금지)와
  충돌한다. §5가 *"성능·정책 주장 0건"* 이라고 선언해도 **"≥1.5×"는 경험적 추론**이며
  불확실도 없이 라벨을 확정한다 ⇒ confound 카탈로그 **#3**(small-n).
- **집계 단위 미등록** — `CONSENSUS.md` §3 항목5(집계 단위를 먼저 정하고 추정 대상과 맞는지
  논증하라). "steady, BLOCK 필드"라고만 적고 emit 주기·정상상태 진입 기준·폐기 블록 수를
  등록하지 않는다(Zamba2에서는 `SGLANG_ZAMBA_TIMING_EVERY`, prefill 기본 4 forward —
  `zamba2.py:648-655`).

---

## §2. 차단 (B#) — 死因은 아니나 통과 전 해소 필요

**B1. `ZNPT2` 태그 fail-loud(§2-c)가 소비자를 등록하지 않아 `ZBPT` 함정을 그대로 재생산한다.**
현행 소비자 3개가 전부 스테일이다 — `results/prefill_knee/knee2d.sbatch:105` ·
`knee2d_wide.sbatch:120` · `knee2d_backend.sbatch:105`, 전부 `grep "ZBPT mode=$MODE "` + `tail -1`.
실증: `kneebe_result_L8000_triton_896776.txt`
```
boot_ok=1
RESULT attn=triton L=8000 B=1 sm=full        ← 값 없음
RESULT attn=triton L=8000 B=1 sm=24          ← 값 없음
```
서버 로그에는 `ZBPT2` **14줄**이 있었다. `deprecated_v2 §7.6-3`·`AUDIT_DEBT_2026-08-23.md §7.1`이
*"수리 없이 재사용 금지"* 로 등재한 항목인데 **사전등록은 §7.6-3을 인용하지 않고 하네스를 아예
지목하지 않는다.** 새 태그를 도입하면서 파서를 함께 등록하지 않으면 동일 함정이다.

**B2. §5 `INSTRUMENTATION_ABSENT`의 job 896776 귀속이 사실과 다르다.**
사전등록: *"`ZNPT2` 0줄 — ★job 896776에서 실제로 겪은 실패 모드"*.
실측: triton arm은 `ZBPT2` **14줄 존재**했고 실패한 것은 **하네스 파싱**이다(B1).
`ZBPT2` 0줄이었던 것은 **flashinfer arm**인데 그것은 계측 부재가 아니라 **스케줄러 사망**이다.
⇒ 이 라벨은 실제로 겪은 두 실패 모드 중 **어느 쪽도 잡지 못한다**. (게이트 #21 — 측정 실패를
게이트 실패로 라벨링하지 말라 — 의 **거울상**: 서로 다른 두 실패를 한 라벨로 접었다.)

**B3. `SGLANG_NEMO_TIMING`(§4-1)은 트리에 존재하지 않는 env 이름이다.**
```
grep -rn "SGLANG_NEMO_TIMING" prefill-layer-alloc/workspace sglang_engine_dev/python/sglang
# → 히트: 사전등록 자기 자신 1건뿐
```
실제 이름은 `SGLANG_NH_LAYER_TIMING`(`nemotron_h.py:646`). 새 이름을 쓰겠다면 기존 경로의
처분(제거/공존/치환)을 등록해야 한다. 추가로 **`=0`이 truthy**인 함정이 이 정확한 줄에 대해
이미 등재돼 있다 — `results/ltsm_probe/PREREG_LTSM_P1_PROBE_2026-08-14.md:50` **F3**:
*"`nemotron_h.py:646` (`_os.environ.get(...) and _is_decode`, no `bool()` even needed) ... The OFF
arm of any TIMING on/off dyad must `unset` the variable, never set it to `"0"`."*
§4-2("OFF 경로 무비용")는 OFF arm의 **설정 규율**을 등록하지 않는다.

**B4. manifest 경로가 동시 세션 소유 파일과 충돌한다.**
`nemotron_h.py`는 `sync_engine_tree.sh` **밖**이다(스크립트 `:94` 주석: *"(nemotron_h / falcon_h1 /
granitemoehybrid) are still manual copies"*; 설계문서 V3 동일). §4-6 *"`sync_engine_tree.sh`
manifest 갱신"* 은 그 파일 수정을 요구하는데, **`sync_engine_tree.sh`는 동시 세션 소유**로
지정돼 있다. 또한 설계문서 §5-L0-2가 경고한다: *"sync에 넣는 순간 기존 캠페인의 manifest가
바뀐다 ⇒ 별도 캠페인 manifest로 격리하라."* 격리 계획 미등록.

**B5. §3의 축 열거가 최소 5개 부족하다.** 등록된 4축(버킷 내용·mlp 위치·층 수·백엔드) 외에:
(5) **forward mode**(prefill vs decode — F4), (6) **cudagraph ON/OFF**(정본은 no-cudagraph 필수
라벨, §4-5는 ON 요구 — F4), (7) **모델 규모**(Zamba2-2.7B hidden 2560 vs Nano-9B hidden 4480),
(8) **빌드·telemetry 드리프트**(`deprecated_v2 §3`: *"2026-07 빌드·다른 격자·다른 n"*;
`PROJECT_STATUS.md` 방법론 게이트 #41), (9) **계측 자기비용 비대칭**(설계문서 V5: Zamba2는
이벤트 풀링(`_zt_event`, `zamba2.py:90-98`), Nemotron은 **층당 신규 Event 2개**(`:710`) ⇒ 56층 =
forward당 112개 할당; 자기비용 **E2는 미측정**으로 등재돼 있다). (9)는 단순한 추가 confound가
아니라 **측정값을 직접 편향**시킨다.

**B6. 번호 체계 미표기 2건 — `PROJECT_STATUS.md` 방법론 게이트 #82(2026-08-28 신설) 위반.**
사전등록은 *"게이트 #34"*(줄 3·7)와 *"게이트 #21"*(줄 72)을 체계 이름 없이 쓴다.
값 자체는 **둘 다 맞다**(`PROJECT_STATUS.md` 방법론 게이트 체계: #34 = 규칙 먼저·하네스 나중
2단 감사, `PROJECT_STATUS.md:6666`; #21 = 측정 실패를 게이트 실패로 라벨링 마라). 그러나
같은 날 신설된 게이트가 요구하는 **체계 병기**가 없다.

**B7. §6이 `design_reachability.py`의 자기 등재 한계와 rev3 감사 F11을 인용하지 않는다.**
- 도구 자신(`scripts/discipline/design_reachability.py:31-34`):
  > ★KNOWN LIMIT ... this tool sees the **CATEGORICAL lattice only**. ... rev3's forcing lives in
  > the **CONTINUOUS layer (se, thresholds, TOST margin, priors)** and it was NOT [caught].
  > A DISCRIMINATING verdict here does not mean the design can produce a verdict.
- rev3 감사 F11(`results/tc1_model_attrib/audit_tc1_rules_rev3_2026-08-28/VERDICT.md:176-200`):
  비차단 축에 대해 이 도구는 **증명 가능하게 무력**(어느 값을 넣어도 동일 판정).

이 사전등록의 강제는 **전적으로 연속층**(0.95 · 1.5× · ±20%)에 산다 ⇒ §6이 얻을
`DISCRIMINATING`은 **구조적으로 실패할 수 없는 실행**이다. §6은 그것을 *"확인한 뒤 제출"* 의
정당화 근거로 쓴다 — **무력한 검사를 통과 증거로 쓰려는 것**이다.
추가로 도구 rev2가 요구하는 **prior 등록**(`:19-22`, repair (c))과 **provenance 문자열**(`:28-29`)이
사전등록에 없다(F2).

**B8. §0의 *"이식 외에 길이 없다"* 는 비약이다.** §0이 실제로 증명하는 것은
*"정본 Diff A를 Nemotron arm으로 **이전**할 수 없다"* 이다(그건 옳다 — `server_args.py:1959`
assert 실재 확인). *"따라서 계측 이식만이 유일한 길"* 은 따라 나오지 않는다:
`PER_LAYER_TYPE_SM_RESPONSE_DESIGN_2026-08-11.md` §4가 **경로 2(nsys 커널 타임라인)** 와
**경로 4(층 구성 차분 설계)** 를 이미 등재하고 있고, **경로 2는 운영점을 볼 수 있는 유일한
경로**다(경로 1은 V12로 원리적 불가). 대안을 배제하려면 그 문서를 인용해 배제 사유를 적어야 한다.

---

## §3. 합격 기준 4개 — 각각 판정

### ① §2의 결함 3종 방지책이 실효인가 (특히 legacy 필드 **제거**가 과한지) → **불합격**

| 방지책 | 판정 | 근거 |
|---|---|---|
| (a) 버킷 비대칭 → 층 전체 대칭 + `closure≥0.95` | ❌ **검증 수단이 항등식** | F1. 처방 자체(층 전체 대칭 훅)는 **옳다**. 죽는 것은 **검증 수단**이다. 덧붙여 §1의 *"클래스 경계 = 층 타입 경계이므로 **구조적으로 재발하지 않는다**"* 는 **과장**이다 — 비대칭은 합성 블록 구조가 아니라 **훅 위치 선택**의 문제이고(현행 zamba2는 합성 블록 안에서도 대칭을 달성했다, `zamba2.py:339-356`), Nemotron에서도 `NemotronHAttention.forward`(`:500`)에 걸지 `...DecoderLayer.forward`(`:529`)에 걸지의 선택이 **그대로 남아 있다**. |
| (b) 러닝평균 → BLOCK 필드만, legacy 필드 미생성 | ⚠️ **과하지 않다 — 오히려 부족하다** | 제거가 잃는 것은 없다(§3이 이미 수치 비교를 금지했으므로 정본 대조 가능성은 애초에 존재하지 않는다). **진짜 문제는 반대쪽**: 제거해야 할 러닝평균이 **이미 `nemotron_h.py:672-674·730-737`에 살아 있는데 사전등록이 그 존재를 모른다**(F3). 두 계측이 공존하면 하네스가 어느 필드를 읽었는지 알 수 없고, 그것은 **정본이 이미 한 번 당한 실수**다(`deprecated_v2 §3`: *"이 세션이 처음에 그 필드를 읽었다가 정정했다"*). |
| (c) `ZNPT2` fail-loud | ❌ **무효** | B1 — 소비자(하네스 파서) 미등록. `ZBPT`/`ZBPT2` 함정의 재생산. |

### ② §4 정확성 게이트 6개가 이식 실패를 실제로 잡는가 → **부분 합격(핵심 실패는 못 잡음)**

| 게이트 | 잡는가 | 비고 |
|---|---|---|
| 1 출력 byte-identical | ✅ | 유효. 단 §4-2와 함께 OFF arm 규율(`unset` vs `=0`) 등록 필요(B3). |
| 2 OFF 무비용 | ⚠️ | "분기 외 연산 0"은 코드 검사로 확인 가능하나 **측정으로 검정되지 않는다**. `=0` truthy 함정 미등록. |
| 3 `closure ≥ 0.95` | ❌ | **F1 — 항등식·prefill 미발화·문턱 미유도.** |
| 4 층 수 일치 | ✅ **가장 강한 게이트** | 실제로 미훅 클래스를 잡는다. ★단 **"관측 층 수"를 무엇으로 세는지 미정의** — emit된 나눗수를 세면 순환이다(`nemotron_h.py:737`의 하드코딩 `/24,/24,/4`가 정확히 그 함정, F3-d). `hybrid_override_pattern`에서 **런타임에 세어** 나눗수로 쓰도록 등록할 것. |
| 5 cudagraph-ON 통과 | ❌ | **F4 — 판정 불능·V12와 충돌·§5 `INSTRUMENTATION_ABSENT`와 구별 불가.** |
| 6 CPU 회귀 + manifest | ⚠️ | 유효하나 manifest 격리 계획·동시 세션 조율 미등록(B4). 설계문서 V24: `unittest discover`는 로그인 노드에서 완주 못 함 — **컴퓨트 노드 실행**을 명시할 것. |

★**전면적 누락**: 물리 불변량(GATE N류)·host-floor·계측 자기비용 — 이 계측을 **실제로 죽인
실패**(V10, `per_mamba` 2.78× 부풀림·SM 비단조)에 대응하는 게이트가 **0개**다.

### ③ §5 결정 규칙에 항등식·도달불가 라벨이 있는가 → **있다. 3종.**

1. **항등식** — `CLOSURE_FAIL`의 조건 `closure < 0.95`가 등록 regime(prefill)에서 **관측 26/26이
   0.9939 이상**이며 Nemotron 층 전체 버킷은 루프를 타일링한다 ⇒ **도달 불가 라벨**(F1).
2. **설계 강제** — `DIFFA_OPENS`는 §2가 막는 결함 2종을 **동시에** 재현해도 16/16 발화한다(F5).
   유일한 긍정 라벨이 데이터에 의해 구매되지 않는다.
3. **★누락 라벨(신규 지적)** — `Diff A`가 L에 대해 **닫히는**(감소하는) 세계가 `DIFFA_INCONCLUSIVE`
   로 접힌다. `OPENS`(≥1.5×)와 `FLAT`(±20%) 사이의 1.2–1.5× 구간도, 0.8× 미만의 **명확한 반전**도
   모두 같은 "그 외"다. ⇒ **정본 방향의 반증을 기록할 라벨이 없다** — 귀무 채택형 게이트
   (`CONSENSUS.md` §3 항목20 계열: 귀무 채택형 게이트는 노이즈를 보상한다). `DIFFA_CLOSES`를
   신설해야 세계가 대칭적으로 열거된다(`CONSENSUS.md` §3 항목66: 결정 규칙은 코드로 고정하고
   세계를 전수 열거하라).

### ④ §7 "닫지 못하는 것" 4건이 정직한가 → **3.5/4 정직, 그러나 누락 4건**

| §7 항목 | 판정 |
|---|---|
| 1 백엔드 효과 분리 불가 | ✅ **정직**. `server_args.py:1959` assert 실물 확인, 문구 일치. 신규 게이트 #83과 정합. |
| 2 정본 Diff A와의 연속성 불가 | ⚠️ **정직하나 자기모순** — §5가 그 금지된 양으로 문턱을 세운다(F2). |
| 3 R2′ 자체는 검정 안 함 | ✅ **정직**. |
| 4 Diff B 미포함 | ✅ **정직**. |

**§7이 빠뜨린 것 4건**:
(i) **운영점에 도달 불가**(V12) — 이 캠페인은 정의상 off-operating-point인데 §4-5는 "운영점"이라 적는다.
(ii) **기존 NHLT 계측과의 관계**(F3) — 이식이 아니라 **리워크 이식**이고, 기존 경로의 처분이 미등록.
(iii) **host-bound 셀에서 추정량이 죽는다**(V10/V11) — 이 계측 계열의 알려진 死因.
(iv) **2점으로는 계수·지수를 못 잰다**(F6) — 사전등록이 산다고 적은 바로 그 물건.

---

## §4. 반증 실패 — 감사가 깨지 못한 것 (그러나 아래 §5 전엔 GO 아님)

1. **§3의 층 수 4/27/25 — 정확하다.** config 직접 계수:
   ```
   hybrid_override_pattern = "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"
   len = 56 = num_hidden_layers ; Counter{'M':27, '-':25, '*':4}
   ```
   (`hf_cache/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base/snapshots/dc0661c8.../config.json`)
2. **§1의 코드 좌표 5개 전부 정확** — `nemotron_h.py` `:554` `ALL_DECODER_LAYER_TYPES` ·
   `:510` Attention · `:377` Mamba · `:293` MLP · `:341` MoE. MoE가 Nano-9B에 없다는 것도 맞다
   (pattern에 MoE 문자 0개). 버킷만 예약하는 처방도 합리적이다.
3. **§1의 `zamba2.py:339-356` 좌표 정확** — 합성 블록 안의 `other/attn/other/mlp` 구간 분할이
   정확히 그 줄들이다. 트리 모호성 없음(`src/models/zamba2.py`와 dev 트리 `diff -q` **IDENTICAL**).
   ★참고: 정본 `CONSENSUS §1-3`이 쓰는 `:163`/`:259`/`:391-394`는 **수리 이전 좌표**라 현행 파일에서
   다른 내용을 가리킨다 — 그러나 **사전등록은 그 좌표를 쓰지 않았다**(감사 요청서 항목5의 전제가
   사전등록에 대해서는 성립하지 않는다). 사전등록의 인용은 이 점에서 깨끗하다.
4. **§2-b의 *"현행 zamba2는 이미 BLOCK 필드(`b_per_*`, 매 emit 리셋)로 수리했다"* 는 추정이 아니라
   코드 사실이다.** `zamba2.py:127-133` `_zt_new_block()` docstring *"reset at EVERY emit (defect (2)
   fix)"* · `:716` `self._zt_blk = _zt_new_block()` · `:686-701` emit 포맷의 `b_per_attn`/`b_per_mamba`.
   (메모리 항목72 계열 — 이 서술은 **대조 완료**.)
5. **legacy 필드 제거가 "과하다"는 감사 가설 — 반증 실패.** 제거로 잃는 대조 가능성은 없다.
   (단 F3/①(b)가 보이듯 **문제는 반대쪽에 있다**.)
6. **§5의 `DIFFA_FLAT` ±20% 밴드 자체는 깨지 않았다** — 대칭적이고 그 자체로는 무해하다.
   문제는 밴드가 아니라 **밴드 밖 세계의 열거**다(③-3).
7. **§0의 백엔드 서로소 사실 — 깨지 않았다.** 다만 이는 신규 발견이 아니라 **재확인**이다:
   `PER_LAYER_TYPE_SM_RESPONSE_DESIGN_2026-08-11.md` §3.A **V21**이 2026-07-28
   (`FINDINGS_8B_2026-07-28.md` §1)을 인용해 이미 등재했다 — *"NemotronH=flashinfer(triton은
   어서션 금지) / Zamba2=triton ⇒ **attn 버킷이 서로 다른 커널 구현**"*.
   `deprecated_v2`가 새로 산 것은 **2/2 재현된 flashinfer×Zamba2 사망**이다.

> **반증 실패 항목 요약**: 이식의 **훅 위치 설계**(층 클래스 경계)와 **모델 사실**은 옳다.
> 깨진 것은 **검증 수단(§4)·결정 규칙(§5)·자기 위치 인식(§1/§7이 기존 계측과 기존 설계문서를
> 모른다)** 이다. 아래 §5를 사기 전에는 CONFIRMED 아님.

---

## §5. 이를 확정할 실험 — **GPU 0 우선**

| # | 무엇 | 비용 | 산출 |
|---|---|---|---|
| **E0** | **기존 계측 인벤토리 감사(선결)** — `nemotron_h.py:646,672-675,700-737` 직독 + `PER_LAYER_TYPE_SM_RESPONSE_DESIGN_2026-08-11.md` §2·§3(V1–V14)·§4·§5-L0 정독 + `P5_GATES_BATCH1_2026-08-05.md` §4 정독. 사전등록을 *"신규 이식"* 에서 *"**2026-08-04 리워크의 NHLT 이식**"* 으로 재프레이밍. | **GPU 0**, ≈1h | §1/§2/§7 전면 개정. `SGLANG_NH_LAYER_TIMING` 경로의 처분(제거/치환) 등록. `:737` 하드코딩 `/24,/24,/4` 수리를 이식 항목으로 등재. |
| **E1** | **closure 게이트 폐기 또는 재도출.** 위 태그별 분포(ZBPT2 n=26 / ZBLT2 n=553)를 사전등록에 등재하고, prefill에서 0.95가 미발화임을 명시. 대체 게이트로 **(a) GATE N 재정의판**(같은 realized bs에서 ctx 불변성 — mamba는 prefill에서 O(L) 선형이어야 한다) **(b) host-floor 대조**(`TIMING=0/1` 대칭) **(c) 층 수 게이트 강화**(런타임 pattern 계수 = 나눗수)를 등록. | **GPU 0**, ≈1h | §4-3 교체. `CLOSURE_FAIL` 라벨 제거 또는 도달 가능 문턱으로 재설정. |
| **E2** | **§5 결정 규칙 재작성 — 계수 추정 설계로.** L 격자 **≥5점**(예 512/1024/2000/4000/8000) × 셀당 **n≥4** × warm-up 폐기 규칙 · 블록 정의 · batch/SM 고정. 결정량 = **지수 회귀 계수와 그 CI**(정본과 같은 형식, 단 수치 비교 금지). 라벨 = {`EXPONENT_SEPARATED`(attn 지수 CI가 mamba 지수 CI를 배제) / `EXPONENT_OVERLAP` / `FIT_FAIL`(R² 문턱 미달) / `INSTRUMENTATION_ABSENT` / `ADMISSIBILITY_FAIL`}. **`DIFFA_CLOSES` 신설로 세계를 대칭 열거.** 각 라벨에 **prior 등록**(`design_reachability.py` repair (c) 요구). | **GPU 0**, ≈2h | §5 전면 교체. F5·F6 해소. R2′ 계수를 실제로 산출 가능. |
| **E3** | **하네스 소비자 등록.** `ZNPT2`를 읽을 sbatch/파서를 명시하고, `AUDIT_DEBT §7.1` knee2d 수리를 **선결조건**으로 등록. 스모크 판정에 **"`RESULT` 줄이 비어 있지 않다"** 를 추가(896776 재발 방지). `boot_ok=1`을 성공 신호로 쓰지 않음을 명문화. | **GPU 0**, ≈1h | B1·B2 해소. §5 `INSTRUMENTATION_ABSENT`를 두 실패(파싱 스테일 / 서버 사망)로 분리. |
| **E4** | **admissibility 프로브(규칙 통과 후, 본 측정 **전**).** `PER_LAYER_TYPE_SM_RESPONSE_DESIGN_2026-08-11.md` §5-L1을 Nemotron-Nano-9B로 옮긴 판: 단일 셀, **prefill 모드 명시**, ctx 2점 × 같은 realized bs, SM 2점, `TIMING=0/1` 대칭 대조. **A1(GATE N) · A2(host floor) · E2(계측 자기비용)** 동시 판정. | **≈0.3–0.5 GPU-hr** | 통과해야만 본 격자 구매. 실패 시 경로 1은 이 기판에서 죽고 **경로 2(nsys)** 로 간다(설계문서 §5-차선 1). |
| **E5** | **§6 도달가능성 절 수리.** `design_reachability.py`가 연속층 강제에 무력함(도구 `:31-34` + rev3 F11)을 사전등록 본문에 명시하고, **연속층 강제 자기점검**을 별도로 등록: *"각 문턱에 대해, 그 문턱이 데이터와 무관하게 한 라벨을 강제하는 극단 사례가 있는가"* 를 대수적으로 점검(방법론 게이트 #40의 요구). | **GPU 0**, ≈0.5h | B7 해소. |

**총 GPU 지출: E0–E3·E5는 0, E4는 규칙층 통과 후 ≈0.3–0.5 GPU-hr.**
★`CONSENSUS.md` §3 항목77(*"트랙 자신의 사전등록이 지목한 최소비용 선결 프로브를 안 산 채 더
비싼 후속을 반복하지 마라 — 가장 비싼 지출은 GPU가 아니라 안 산 프로브"*)이 이 트랙에 정확히
적용된다: 설계문서가 2026-08-11에 **L0(GPU 0) → L1(0.5 GPU-hr)** 사다리를 이미 처방했고,
이 사전등록은 **그 둘을 건너뛰고 L2로 간다**.

---

## §6. 금지 문장 신설 (이 트랙에 고정)

1. ★*"`closure ≥ 0.95`를 통과했으므로 층-타입 수치를 신뢰할 수 있다"* —
   `P5_GATES_BATCH1_2026-08-05.md:234-236`이 **20/20 PASS ∧ 3–5/5 GATE N VIOLATION** 존재증명을
   이미 냈다. prefill에서는 통과 자체가 정보 0이다(26/26 ≥ 0.9939).
2. ★*"Nemotron 층-타입 계측을 새로 이식했다"* — `NHLT`(`nemotron_h.py:646,700-737`)가 이미 있다.
   정확한 문장은 *"2026-08-04 리워크(블록 누산기·shape guard·대칭 버킷)를 기존 NHLT에 이식했다"*.
3. ★*"cudagraph-ON에서 per-layer 층-타입 시간을 쟀다"* — replay는 Python forward를 호출하지 않는다
   (설계문서 V12: `cuda_graph_runner.py:547`·`:1161`). 이 캠페인은 **정의상 off-operating-point**다.
4. ★*"Diff A_nemotron이 L에서 열린다(≥1.5×)"* — 문턱의 provenance가 등록되기 전까지 금지.
   그리고 이 술어는 정본 결함 2종을 동시에 재현해도 16/16 발화한다(F5) ⇒ 통과가 무언가를
   확증한다고 쓰지 말 것.
5. ★*"이 측정이 R2′ 계수를 확정했다"* — L 2점·n 미등록 설계로는 지수가 식별되지 않는다(F6).
6. ★*"job 896776은 계측 부재로 실패했다"* — triton arm은 `ZBPT2` **14줄** 존재. 빈 것은 `RESULT`
   줄이고 死因은 **하네스 스테일**이다(`AUDIT_DEBT §7.1`).
7. ★*"이식이 규칙층을 통과했다"* — 사전등록 §8이 스스로 금지한 문장. **이 판정은 `NO-GO`다.**

---

## §7. 회계

| 항목 | 값 |
|---|---|
| GPU 지출 | **0** (job 제출 0, sbatch 0) |
| 새 성능 판정 | **0건** |
| 등급 변경 · 정책 순위 변경 | **0건 / 0건** |
| 정본 편집 | **0건** (`PROJECT_STATUS.md`·`CONSENSUS.md` 무수정) |
| 대상 파일 수정 | **0건** (`PREREG_NEMOTRON_ZT_2026-08-28.md` 무수정) |
| 엔진 트리 수정 | **0건** (읽기·`diff -q`만) |
| 동시 세션 소유 파일 | **미접촉** (`results/cp_baseline/`·`chunk_probe*`·`sync_engine_tree.sh`·`dev_tree_edits.md`·`presubmit_registry.json` — 마지막 것은 **키 목록만 읽음**, 수정 0) |
| git | `git add`·커밋 **0건** |
| 신설 파일 | 이 판정서 1개 |

**불변 배너 유지 확인**: HE0 · 정책 순위 · gate #13/#16 "닫았다" 금지 · switch-cost "닫았다" 금지 ·
C2 인용정지 (a)(b) · `CONSENSUS §1-24` 미반증 — **이 판정은 위 어느 것도 건드리지 않는다.**
본 판정서의 모든 수치는 **결정 규칙의 판별력에 관한 진술**이며 성능 주장이 아니다.
§1-F5 표는 Zamba2 arm 내부의 **추정량 민감도 분석**이고 정본 수치와 비교하지 않는다
(`deprecated_v2 §7.6-6` 준수). `P5_GATES_BATCH1_2026-08-05.md` 인용은 전부
**★UNAUDITED·정본 아님** 라벨과 함께 쓴다.

### 인용 정확성 전수 대조 (감사 요청 항목7)

| 사전등록의 인용 | 실물 | 판정 |
|---|---|---|
| `nemotron_h.py:554` `ALL_DECODER_LAYER_TYPES` | 그 줄 | ✅ |
| `:510` Attention / `:377` Mamba / `:293` MLP / `:341` MoE 층 클래스 | 전부 일치 | ✅ |
| `zamba2.py:339-356` 합성 블록 내 구간 분할 | 일치(두 트리 동일) | ✅ |
| `server_args.py:1959` triton 부팅 거부 assert | 일치(문구까지) | ✅ |
| `hybrid_override_pattern` attn 4 / mamba 27 / mlp 25 | config 직접 계수 일치 | ✅ |
| Zamba2 attn 9 / mamba 54 | `zamba2.py:483-484`(`_zt_n_attn`/`_zt_n_mamba`)·`P5_GATES:31`(`/9`) 일치 | ✅ |
| "게이트 #34" (규칙 먼저·하네스 나중) | `PROJECT_STATUS.md` 방법론 게이트 #34 | ✅ 값 정확 / ❌ **체계 미표기**(게이트 #82) |
| "게이트 #21" (측정 실패 ≠ 게이트 실패) | `PROJECT_STATUS.md` 방법론 게이트 #21 | ✅ 값 정확 / ❌ **체계 미표기**(게이트 #82) |
| `CONSENSUS.md` §1-3 계측 결함 2종 | rev51 §1-3 행에 실재 | ✅ |
| `deprecated_v2/README.md` §1 (백엔드 서로소, job 896776) | 실재 | ✅ |
| `SGLANG_NEMO_TIMING` (§4-1) | **트리에 존재하지 않음** (실제: `SGLANG_NH_LAYER_TIMING`) | ❌ **B3** |
| "job 896776에서 실제로 겪은 실패 모드 = `ZNPT2` 0줄" (§5) | triton arm `ZBPT2` **14줄** 존재, `RESULT` 줄이 빔 | ❌ **B2** |
| `closure ≥ 0.95` 출처 | 미표기(실제: `zamba2.py:702` 경고 기본값; 채점 문턱은 0.85) | ❌ **F1** |
| `≥1.5×` 출처 | **저장소 전체에 근거 0건** | ❌ **F2** |

**미인용 필수 제약(있어야 했는데 없는 것)**: `deprecated_v2 §7.6-3`(knee2d 스테일 재사용 금지) ·
`PER_LAYER_TYPE_SM_RESPONSE_DESIGN_2026-08-11.md`(기존 계측 인벤토리·V10·V12·L0/L1 사다리·
경로 2/4 대안) · `P5_GATES_BATCH1_2026-08-05.md`(closure 항등식·GATE C 맹점·0.85 교정) ·
`PREREG_LTSM_P1_PROBE_2026-08-14.md:50` F3(`=0` truthy).
