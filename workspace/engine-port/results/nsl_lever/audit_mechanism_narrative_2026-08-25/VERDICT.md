# 적대적 판정서 — **기전 서술 5건** (2026-08-25, claims-auditor)

대상: 메인 세션이 사용자에게 낼 **설명 문장**(SM 분할 vs admission · cap vs KV · mamba assert · 얽힘 · 전략 판단).
**읽기 전용 · GPU 0 · 새 성능 판정 0건 · 등급 변경 0건.** 읽은 트리: **실제로 도는 엔진** `sglang_engine_dev/python/sglang/srt/`.

---

## 주장 1 — SM 분할 vs admission ⇒ ★**REFUTED**(깨끗한 이분법으로서)

- green ctx가 나누는 것은 **SM 개수뿐**(HBM 용량·L2·복사엔진·가중치 공유). 여기까진 정합.
- ★**"단위시간당 연산량(rate)을 바꾼다"는 이 기판에서 틀렸다.** `multiplexing_mixin.py:881-923`:
  분할은 **prefill이 in-flight인 동안에만** 걸리고, 나머지는 `(108,0)` 또는 `(0,108)`이다.
  정본 §1-25 실측: decode-active 시간 중 라벨 D 실현 비율 **T8 3.8–9.3% / Ha8 10.4–18.7%**,
  나머지 **81–96%는 무분할 108 SM**. §1-26: *"'decode가 D SM에서 돌았다' ⟺ 'prefill이 동시에 in-flight였다'는 같은 사건"*.
  ⇒ **지속적 rate 분할이 아니라 "동거 창 안에서만 발효하는 조건부 자원 배분"**.
- **admission = 메모리 손잡이**: **부팅 시점만 CONFIRMED**(`max_mamba_cache_size = cap`, 슬롯당 ≈68.9 MB).
  ★**런타임 REFUTED** — `scheduler.py:3127-3150` `set_internal_state`가 `pp_max_micro_batch_size`를
  런타임 갱신하는데 **pool을 재할당하지 않는다**.
- **히스테리시스**: ★부분 REFUTED — `retract_decode()`(`schedule_batch.py:1950`, `scheduler.py:2574-2585`)로
  **un-admit 경로가 있다**(트리거는 cap이 아니라 KV 부족). 발화 여부는 **확인 불가**(`num_retracted_reqs`가
  텍스트 로그에 안 나감). 반대쪽도 과장 — SM 전환은 매번 **양 스트림 2× drain**이고 코드 주석이
  *"~200ms+ spike vs ~40ms TPOT"* 라 적는다.
- ★★**두 손잡이가 직교하지 않는다** — cap은 in-flight prefill 수를 바꿔 **실현 D를 직접 움직인다**(§1-25/§1-26).
  `(D, cap)`은 요인설계가 아니다.

## 주장 2 — cap 경로 vs KV 경로 ⇒ **PLAUSIBLE(조건부)**, 5군데 정정

- **`NO_TOKEN` 전수는 7곳**(`schedule_policy.py:504,576,594,646,694,**765,773**`) — ★**765/773이 주경로**(`add_one_req`)인데 빠졌다.
- **"개수 vs 토큰"은 방향으로 맞다**(7곳 전부 토큰 예산항). ★단 *"시퀀스 길이 합"* 은 부정확 — 예산은
  **예약(reservation)** 이다(`:447-454`, `:526-527`). ⇒ *"짧으면 cap, 길면 KV"* 예시는 성립 안 하고
  **점 술어는 감사 B1로 금지** ⇒ **삭제**.
- ★**cap 경로 조건 오기**: `running_bs ≥ cap`이 아니라 **`running_bs + |can_run_list| ≥ cap`**.
- **`// dp` CONFIRMED(조건부)**: 분모는 `dp_size if enable_dp_attention else 1` ⇒ 하네스에서 1.
  ★`server_args.py:6126-6128`이 pdmux에서 `pp_size==1`을 **하드 assert** ⇒ `pp_max_micro_batch_size ≡ cap` 구성상 강제.
- ★★**"반대 방향" 산술 CONFIRMED(코드), 그러나 이 기판에서 실측 0건**: `total_rest_memory`는 cap 무관
  (`:171-173`), `req_to_token_pool`은 방정식 밖(≈0.79 MB). 크기: **1 cap ≈ 402 attention-KV 토큰**.
  ★저장소에 **hybrid를 서로 다른 명시 cap으로 부팅한 기록 0건** ⇒ **항등식이지 관측이 아니다**.
- ★**신규성 오전달**: 이 사실은 정본 **§3 항목23(2026-08-02)** 등재 + NSL-1 **rev2 §3**이 이미 수리 설계까지
  등록했다. B3가 문 것은 **P0 프로브가 그 수리를 되돌린 회귀**다.

## 주장 3 — mamba assert와 cap ⇒ ★**REFUTED (과잉 주장)**

- **맞는 부분**: `memory_pool.py:542-546` `assert` 크래시 CONFIRMED. `schedule_policy` 전체에 mamba 예산항 **없음** CONFIRMED(전 트리 확인).
- ★**(1) "cap이 유일한 구조적 장치" 거짓** — 진짜 가드는 **`_resolve_max_num_reqs`의
  `min(cap, max_mamba_cache_size // ratio)` 클램프**(`:870-874`). cap은 그 min()의 **입력**이다.
- ★**(2) 검사 위치 정정** — admission이 아니라 **batch prepare**에 있다(`mem_cache/common.py:304-316`
  `alloc_req_slots`). radix OFF면 구제 경로가 없고, ON이면 evict 반환값을 무시해 **보호가 완전하지도 않다**.
- ★**(3) "cap을 올리면 크래시한다" 거짓** — 먼저 **부팅 거부**(`MemoryPoolConfig.__post_init__`
  `RuntimeError("Not enough memory…")`) 또는 **cap이 `estimated`(≤4096)에서 조용히 잘린다**.
  후자는 **라벨≠실현**(Stage 0 D108 계열 결함, §1-21)의 admission 축 판본이라 더 위험하다.

## 주장 4 — 얽힘의 세 층 ⇒ **층별로 갈린다**

- **(i) 용량 ⇒ CONFIRMED(코드)** — `profile_max_num_token:171-175` → `handle_max_mamba_cache:196-255`.
- ★★**(ii) 대역폭/캐시 ⇒ REFUTED(서술 그대로) / NOT-YET-SUPPORTED(기전으로서)** — 가장 심각:
  1. **정본이 명시적으로 금지**: `CLAIM_EVIDENCE_MATRIX.md:300-304` *"SM92에서도 achieved_BW가 사양의
     **45.4–57.7%**뿐이라 고-SM 평탄화를 **HBM 포화로 서술 금지**"*; `PREREG_E1_BSWEEP_REGIME:39,46,203`
     *"'memory-bound' 금지 … AI ≈ 8.2 ≪ ridge 153은 'compute-bound 아님'만 말하고 'bandwidth-bound'를 말하지 않는다"*.
     ⇒ *"decode는 memory-bound"* 는 **인용 금지 문장 그 자체**.
  2. **이 프로젝트는 phase 간 대역폭 간섭을 한 번도 측정한 적이 없다** — 그걸 재려던 `kernel_mech`는
     rev1–rev8 + Stage 0′/0″/0‴ 전부 규칙층 `NO-GO`, **GPU 지출 0**.
  3. ★**(ii)가 실제로 측정된 두 기전을 밀어낸다**: **§1-24** *"ITL 꼬리는 decode step time이 아니라
     **monolithic prefill이 decode를 멈춘 시간**이 지배"*(stall probe **17개 중 16개**, Ha8 d24 167.7→d92 865.6 ms;
     `server_args.py:6129-6131`이 pdmux에서 `chunked_prefill_size == -1`을 하드 assert ⇒ 구조적 전제) ·
     **§1-34** 분할 상태 동거 시 decode forward가 무분할 대비 **granite 1.53–1.66× / zamba2 1.73–1.94×**
     (그 문서 자신이 *"SM 수 효과가 아니라 **split-state 효과**"* 로 귀속). **둘 다 "대역폭 강탈"이 아니라 "직렬화·동거"**.
- **(iii) 스케줄러 제어흐름 ⇒ 방향 CONFIRMED, 공식·"지배적" REFUTED**:
  §1-4 방향은 변화 trace n≥4가 지지(★magnitude는 인용 금지). ★**공식이 죽은 분기의 것**(`:2369`는
  `chunked_req ≡ None`이라 실행 불가; 살아 있는 건 `:2433`). ★*"이게 지배적이다"* 는 **사이트별 카운터
  없이는 만들 수 없는 판정**(점 술어 강화판).
- **bank conflict 부정 ⇒ ★과잉 기각**: 공유메모리 뱅크 배제는 옳으나, **HBM 채널/뱅크 경합은 실재하고
  이 저장소가 못 잰 것**이다. *"틀렸다"* 가 아니라 *"층이 다르다 + 못 쟀다"* 가 정답(**교훈 #21의 사촌**).

## 주장 5 — 전략 판단 ⇒ **PLAUSIBLE(조건부)**, (c)는 ★**REFUTED**

- **(a) 死因 서술 CONFIRMED** — 단 §1-17에 doc-steward가 붙인 ★★**스코프 각주**를 반드시 같이 옮겨야 한다:
  *"확정하는 것은 **달성된** 정책 계열(single-worker·SM-split·reactive)뿐. **달성 가능한 천장**은 다른
  명제이며 별도로 확정된 바 없다(§5-8(a)(b)가 dual-worker·non-SM-split lever를 **열린 항목**으로 유지).
  두 명제를 같은 문장으로 혼동하지 말 것."*
- **(b) "cap 풀면 batch↑ ⇒ ITL↑" ⇒ 추론이다. 절반만 관측**: 뒷다리(batch↑⇒step↑)는 prefill-free step에서
  bs 1–19 단조(서술, 등재 대상 아님 — 앨리어스 파일 + **관측 범위가 cap 근처를 안 덮음**).
  **통제된 B 스윕은 실행된 적 없음**(`bsweep_regime/`에 사전등록 3판, 데이터 0건). 앞다리(cap↑⇒batch↑)는 **부팅 0건**.
- ★**(c) "세 판본 어디에도 논증이 없다" ⇒ REFUTED**: rev1(`git 7149931:43-44`) · rev2(`:99-105`, `:151-164`) ·
  rev3(`:56-59`) **전부에 위치 논증이 있다**. 없는 것은 **효력 크기의 사전 정보**이고 rev2 `:145-147`이
  **스스로 등재**했다(*"사전 정보가 0"*). rev3의 것은 감사 H1이 **단위 오독으로 죽였다** ⇒ *"없다"* 가 아니라
  **"있었고 틀렸다"**. ★부수: 워킹 트리에 `DESIGN_NSL1_*.md`는 **2개**(rev1은 git에만).
- ★**(d) 스코프 초과 위험 실재**: 정본 **§5-8(b)** 는 admission 갈래를 **열린 채로** 두고 있고 그 이유가
  *"死因을 직접 겨냥하기 때문"* 이다. 주장 5는 같은 사실을 **감점 사유로 뒤집어 읽는다**.
  ★또한 *"자유도가 있다는 사전 논증을 먼저 세우라"* 는 **측정 전에 결과를 요구**하는 형태(교훈 #20/#21)다 —
  *"효력 크기 사전 하한 없이 세운 검정력이 정본 SD에서 실제로 서는가"* 로 바꿔 물어야 정당하다.
  ★**MEMO §4와도 불일치** — MEMO의 선택지는 **②+③ 동시 구매 / 가설 (a) 축소** 둘이고, *"논증 먼저"* 는
  **제3안**이며 저장소 근거가 없다 ⇒ **본인 판단으로 표시할 것**.

---

## ④ 반증 실패 (감사가 깨려 했으나 못 깬 것)

1. `max_mamba_cache_size ≡ cap` 항등식 — dp·pp·spec·명시분기·buffer 전부 훑음, 하네스 설정에서 **정확히 성립**.
2. `pp_max_micro_batch_size ≠ cap` 가능성 — pdmux `pp_size==1` 하드 assert로 **구성상 불가**.
3. `total_rest_memory`의 cap 의존 — 없음 ⇒ **"반대 방향" 산술 반증 실패**(단 실측 0건).
4. admission 예산에 숨은 mamba 항 — 없음(위치는 `common.py:305`로 정정).
5. decode step time의 batch 단조성 — bs 1–19 단조. **단 cap 근처 미관측이라 CONFIRMED로 올리지 않는다.**
6. retraction 발화 여부 — 로그에 `retract` 0건이나 **부재 증명 못 함**(양방향 반증 실패).
7. *"`running_bs ≥ cap`이면 prefill이 막힌다"* 라는 **함의 자체**는 `:2433`에서 참. 정정 대상은 줄번호·조건식.

---

## ⑤ 확정에 필요한 실험

- **E-A**(주장 2c 앞다리, 최소): hybrid 1 arm × cap ∈ {24,48,96} **부팅만** 하고 **배너 3줄**을 읽는다.
  서빙 0, **≤0.02 GPU-hr**. "반대 방향"이 실측이 되고 `estimated` 클램프도 드러난다. **성능 판정 금지 — 배너만.**
- **E-B**(주장 4iii 귀속): 감사 B1이 지시한 **사이트별 카운터**(②) + `--max-mamba-cache-size` 고정(③).
  `(d44, cap∈{24,48,96})`, n≥4 페어드, 변화 trace, 정본 술어(TTFT ∧ 요청-내부 ITL **p95**) + **어느 다리가
  구속하는지 병기** + 양쪽 **임계 사다리**. **②만으로는 B3가 남아 arm 비교가 안 선다.**
- **E-C**(주장 4ii): `kernel_mech`가 **먼저 규칙층을 통과**해야 한다. 통과 전엔 (ii)를 설명에서 **삭제**가 정답.
- **E-D**(주장 5b 뒷다리): `bsweep_regime/` 사전등록 rev3의 존폐를 먼저 결정. 살린다면 **decode batch를
  통제 변수로** 두고 cap 근처(B≈40–48)까지 덮어야 한다(현 아티팩트는 B≤23).
