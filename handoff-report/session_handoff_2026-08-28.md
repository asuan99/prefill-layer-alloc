# 세션 핸드오프 — 2026-08-27~28

## 이번 세션 요약

TC1(모델 귀속)과 M4R(강등 확대 재현) 두 트랙의 **규칙층 감사를 5회** 받았고 **전부 `NO-GO`**였다.
그 과정에서 **프로브 7건**(GPU **≈1.2 GPU-hr**)을 샀고, 그 프로브들이 두 트랙을 차례로 죽였다 —
마지막에는 TC1의 **워크로드·모델 선택 자체**가 틀렸음이 실측으로 드러나
**Zamba2-2.7B → Nemotron-Nano-9B-v2**로 모델을 전환했다. 부수로 **제출 게이트**(규율 도구 4종의
등록된 실행 지점)·**모델 로스터**·**`deprecated_v2/`**를 신설했다.
**새 성능 판정 0건** — HE0 · 정책 순위 · C2 인용정지 (a)(b) · `§1-24` 전부 불변.
★**동시 세션이 같은 저장소에서 활성**이었고(`results/cp_baseline/` 등), 파일 충돌 0으로 공존했다.

---

## 결정·측정

### A. 규칙층 감사 5회 — 전부 `NO-GO`
| 판본 | 판정서 | 요지 |
|---|---|---|
| TC1 rev1 | `tc1_model_attrib/audit_tc1_rules_2026-08-27/` | 死因 F1–F4 · 차단 B1–B18 |
| M4R rev1 | `m4r_confinement/audit_m4r_rules_2026-08-27/` | 死因 F1–F4 · 차단 B1–B16 |
| TC1 rev2 | `tc1_model_attrib/audit_tc1_rules_rev2_2026-08-28/` | 死因 F5–F7. 단일질문 답 **"국소였다"** |
| M4R rev2 | `m4r_confinement/audit_m4r_rules_rev2_2026-08-28/` | 死因 F1′–F4′ |
| TC1 rev3 | `tc1_model_attrib/audit_tc1_rules_rev3_2026-08-28/` | 死因 F8–F12. 답 **"네 번째 고리다"** |

★**형태**: rev1 §3(a) 처방 → F2 / rev2 F2 처방 → F5·F6·F7 / rev3 처방 → F8·F11·F12.
**수리가 옳아도 파급을 재도출하지 않으면 다음 회차 결함이 된다**(정본 항목100 "수리는 국소, 주장은 전역"의 재발).

### B. GPU 프로브 (총 ≈1.2 GPU-hr, 전부 사전등록 후 실행)
| job | 무엇 | 결과 |
|---|---|---|
| 896565 | TC1 F2 확정(Zamba2 anchor=d44) | **`F2_CONFIRMED`** `SW=0`·`gpC=3.232`. ★로그 `SLO-FEAS refused=152` ⇒ **죽은 게 아니라 봉쇄**(사전등록에 liveness 절 부재) |
| 896689/690 | TC1 P3(Qwen2.5-3B ↔ 정본 trace) | **`MEASUREMENT_ABSENT`** — `BIND=0 ∧ FEAS=0`. **배선이 아니라 물리**(Qwen TTFT p50 44.9ms vs Zamba2 1344.8ms ⇒ 동시성 미발생) ⇒ **등록한 두 뿔보다 나쁜 셋째 뿔** |
| 896760/764/767 | Nemotron-Nano-9B 부팅 스모크 | `triton` **부팅 거부** → `flashinfer`로 `boot_ok=1`, **ctx 131072** · cudagraph 캡처 · **긴 프롬프트에서 컨트롤러 이동 2회**(짧은 프롬프트 0) |
| 896776 (×4) | Diff A 백엔드 짝 측정 | triton 재현 성공 / **flashinfer는 스케줄러 사망**(`illegal memory access`, 2/2) |

### C. GPU 0 프로브·도구
- **M4R alias** `EXPOSURE_ALIASED` — `P(decode_sms==D | confined)` = **1.0000, 28/28 셀**. §1-26(B)가 문장으로 예측한 실패의 실측.
- **M4R R_matched** `RMATCHED_VIABLE` 7/7, 드레인 25ms 포화. 관측 1.00–1.16은 **판정 아님**.
- **TC1 F1** `REFUSED_AT_INIT` — `STICKY`+`SLO_SCHED`가 `multiplexing_mixin.py:298-303`에서 RuntimeError.
- **B17 인덱스 사상** — `:328-333` 유래. idx 1=d16 … **4=d44** … 5=d54.
- **설계층 도달가능성** — TC1 rev2 `NOTHING_PURCHASABLE` · M4R rev2 `SINGLE_LABEL_FORCED` · TC1 rev3 `RESTRICTIONS_INERT`.

### D. ★모델 전환 — 엔진이 강제하는 교락
`deprecated_v2/README.md`에 기록. **두 모델 계열의 동작 가능 백엔드가 서로소**다:
Nemotron-H+`triton` = 부팅 거부 / Zamba2+`flashinfer` = 스케줄러 사망(2/2).
⇒ *"모델 고정, 백엔드만 변경"* 셀이 없다 ⇒ **백엔드 효과와 모델 효과 분리 불가**
⇒ 정본 triton Diff A(지수·교차점 ≈3k·Diff B·`R_policy`·C2 국소 ε)는 **스코프 밖 이전 금지**.
Zamba2 전 계열 **ctx 4096**이라 긴 프롬프트도 불가. ⇒ `Nemotron-Nano-9B-v2-Base`(8.89B, ctx 131072,
attn 4/mamba 27/mlp 25) 수신, 대조 arm은 `Qwen2.5-7B`(7.62B).

---

## 코드·문서 변경 (커밋 완료)

| 커밋 | 내용 |
|---|---|
| `88e64e1` `f222c31` | TC1 rev2 · M4R rev2 사전등록 |
| `54c9999` | F2 프로브 결과 |
| `eb31c47` `c5c20c9` `9297d2d` | `design_reachability.py` 신설 · **`presubmit.py` 제출 게이트**(도구 4종 실행 지점) |
| `1c6bb59` | **doc-steward 정본 등재** — `CONSENSUS` rev49→**rev50**, 게이트 **#81·#82**, 교훈 **101·102** |
| `d11243a` | TC1 rev3 (시도·봉쇄 축 + 상호작용 재정식화) |
| `5d180a6` | ★**도구 자기 감사** — `RESTRICTIONS_INERT` 신설, 레지스트리 **append-only** |
| `7eea597` | TC1 P3 결과 + 하네스 `MODEL/CTX` 인자화(기본값 byte-identical) |
| `f8c8bb3` | **`deprecated_v2/`** 신설 + 모델 로스터 + 백엔드 짝 하네스 |
| `1e37cdf` | Nemotron `_zt` 이식 사전등록 |

**미커밋(내 것)**: `results/prefill_knee/kneebe_result_*.txt` 4건(내용 없음 — §아래 주의 참조),
`.knee2d_*` 스크래치. **커밋 가치 없음**(원자료는 서버 로그에 있다).

---

## 열린 항목 / 다음 세션 시작점

★**시작점 한 줄**: **`deprecated_v2` 초안을 doc-steward에 비준시키고, 그 다음 P4(Nemotron-Nano ↔ Qwen2.5-7B 긴 프롬프트) 사전등록을 쓴다.**

1. **doc-steward 비준(GPU 0, 최우선)** — `deprecated_v2/README.md` §6의 질문 셋:
   (a) 백엔드 스코프 배너를 정본에 부착할지 (b) ★*"엔진 강제 교락"*을 **게이트 #10(변수 동시 변경)과 별개 게이트**로 등재할지 (c) `knee2d` 스테일을 수리 대상으로 등재할지.
   ★`1c6bb59` **이후** 발견분(백엔드 교락·모델 전환·P3 결과·도구 자기감사)은 **아직 정본 미반영**이다.
2. **P4 사전등록** — 스모크가 배관을 확정했다(ctx 131072·cudagraph·컨트롤러 이동 2회). B34 경계와
   `argmax` 준거집합을 **실행 전** 고정할 것.
3. **Nemotron `_zt` 이식 감사** — `nemotron_zt/PREREG_NEMOTRON_ZT_2026-08-28.md` 1단계 대기. **통과 전 코드 금지.**
4. **TC1 rev4** — rev3 감사의 死因 F8(주효과 축 소실)·F9(사구간 확대)·F10(죽은 상수)·F11·F12 미해소.
   ★감사 권고: 주효과 축 `main ∈ {both_lose, both_win, mixed}` 복원 · 치환 스킴 코드 고정 · 결정 함수 코드화.
5. **M4R** — 유일 경로는 `PDMUX_STICKY_PARTITION=1` **신규 측정**(≈7.8 GPU-hr). *"M4R은 GPU 0이다"*는 **금지 문장**.

---

## 미완·주의

- ★**규칙층 감사 5회 전부 `NO-GO`.** *"TC1/M4R이 규칙층을 통과했다"* **금지**. 어떤 캠페인도 제출 가능 상태가 아니다.
- ★**제출 게이트가 현재 `exit=1`**이다(M4R `SINGLE_LABEL_FORCED` · TC1 rev3 `RESTRICTIONS_INERT`). 이게 **정직한 상태**다.
- ★**`knee2d` 계열이 스테일** — `:105`가 `ZBPT`를 찾는데 계측은 **`ZBPT2`**를 낸다(의도된 fail-loud).
  그대로 돌리면 `RESULT`가 **빈 채 `boot_ok=1`로 "성공"처럼** 끝난다. 이 세션은 서버 로그 직접 파싱으로 우회.
- ★**`boot_ok=1`은 health 통과일 뿐** 요청 처리 성공이 아니다(flashinfer arm이 그 상태로 죽었다).
- ★**legacy 필드 함정**: `ZBPT2` 줄의 `per-attn`/`per-mamba`는 정본이 결함으로 등재한 **러닝평균**이다.
  **`b_per_attn`/`b_per_mamba`(BLOCK, 매 emit 리셋)를 써라.** 이 세션이 처음에 틀리게 읽었다가 정정했다.
- **재측정된 triton Diff A**(job 896776, steady): L2000 **1.48–1.56** / L8000 **3.31–3.52**.
  ⚠️**정본 수치와 직접 비교 금지**(다른 빌드·격자·n). 짝의 한쪽(flashinfer)이 **측정 실패**라 백엔드 의존성은 **미측정**.
- ★**동시 세션**: `results/cp_baseline/`·`src/multiplex/chunk_probe.py`·`src/patches/chunk_probe_scheduler_hook.patch`·
  `tests/test_chunk_probe.py`·`tests/test_cp0_p1_tools.py` 및 `sync_engine_tree.sh`·`env/dev_tree_edits.md`·
  `presubmit_registry.json` 수정은 **내 것이 아니다**. `git add -A` **금지**.
  그쪽이 **제출 게이트를 채택**했고(자기 spec 등재), 그 spec은 TOOL_REV 2에서 `RESTRICTIONS_INERT`로 뜬다.
- `design_reachability.py`가 **저장소 루트** `scripts/discipline/`에 있다(기존 도구 셋은 `workspace/engine-port/` 아래).
  동시 세션이 그 경로로 호출 중이라 **옮기지 않았다** — doc-steward 조율 대상.
- **claims-auditor 미회부**: `deprecated_v2` 기록 · 모델 로스터 · Nemotron 이식 사전등록 · P3의 "셋째 뿔" 해석.
