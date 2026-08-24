# 사전등록 — **NSL P0 ① GPU-0 재분석**: 추정기 등록 + cap 술어의 **구간 묶기**

2026-08-24 · **GPU 지출 0**(기존 아티팩트만) · ★**작성 시점에 결과를 계산하지 않았다** ·
근거: `audit_nsl_p0_capbind_2026-08-24/VERDICT.md` 권고 ①.

---

## 0. ★ 이것이 무엇이 **아닌지** 먼저

★★**이 재분석은 *"cap이 무는가"* 에 답하지 않는다.** 감사 **B1**이 그 질문의 술어 자체를
무효화했고(등록 항등식이 이 부팅에서 **실행 불가능한 분기**의 가드였다), B1은 **엔진 패치로만**
수리된다. 이 문서가 사는 것은 셋뿐이다:

1. **추정기를 등록한다** — B2가 보인 대로 추정기 선택 하나가 판정을 문턱 양쪽으로 뒤집는다.
2. **재현 가능한 스크립트와 아티팩트를 만든다** — 감사 판정서의 수치는 재현 경로가 없다.
3. ★**관측 가능량이 참 술어를 얼마나 좁히는지 잰다**(§2) — 이것이 **B1 엔진 패치가 정말 필요한지**를
   결정한다. 좁으면 그 축은 패치 없이 결론 가능, 넓으면 패치가 **필수**임이 증명된다.

★**금지**: 이 문서의 어떤 수치도 *"cap이 문다/안 문다"* 로 인용할 수 없다.

## 1. 참 술어와 관측 가능량의 관계 (감사 B1)

이 부팅(`--chunked-prefill-size -1`)에서 살아 있는 cap 경로는 **하나**다:

```
scheduler.py:2432-2434   len(adder.can_run_list) >= get_num_allocatable_reqs(running_bs)
                     ⟺   running_bs + |can_run_list| >= cap                    … (참 술어 T)
```

★**`can_run_list`은 telemetry에 없다**(스키마 44필드 전수 확인 — `can_run`·`adder`·`allocatable`
어느 것도 없음) ⇒ **T는 기존 데이터로 계산 불가능**하다. 그러나 두 부등식이 T를 **묶는다**:

| | 관계 | 근거 |
|---|---|---|
| **하계 L** | `running_bs ≥ cap` ⟹ T | `\|can_run_list\| ≥ 0` |
| **상계 U** | T ⟹ `running_bs + queue_depth ≥ cap` | `scheduler.py:2415`가 `for req in self.waiting_queue:`로 순회해 `:2459 add_one_req`를 부르므로 **`\|can_run_list\| ≤ len(waiting_queue)`**. telemetry `prefill_queue_depth` = `len(scheduler.waiting_queue)`(`dual_worker.py:82`, `:569`서 바인드, `:575-578`서 종료 req 제거) |

⇒ **L ⊆ T ⊆ U.** ★**L이 P0 사전등록의 등록 항등식이었다** — 즉 P0는 T를 **하계로 대체**하고 동치라 선언했다.

★**이 묶음이 닫지 않는 것**(정직하게): (a) `batch_is_full`은 **래치**라 T의 *발화 사건*과 스냅샷의
*상태*는 시각이 다르다 — 이 구간은 **상태 조건**을 묶는 것이지 사건을 묶지 않는다. (b) 스냅샷 지점
(`multiplexing_mixin.py:965/997/1242`)과 `:2433` 실행 시점이 다르다. **두 한계는 이 재분석이 닫지 못하며
B1 엔진 패치가 필요한 이유로 남는다.**

## 2. ★ 등록 추정기 (감사 B2)

`multiplexing_mixin.py:508-520`이 이미 경고한다: *"any 'fraction of snapshots' statistic … is biased …
**TIME-WEIGHTED** statistics (weight each snapshot by the gap to the next one) … are unaffected."*

- ★**1차 추정기 = 시간 가중**: 각 스냅샷에 **다음 스냅샷까지의 간격**을 가중치로 준다
  (`timestamp_monotonic_s` 차분). 마지막 스냅샷은 가중치 0(꼬리 미정의 — 절단하지 않고 0으로 둔다).
- **개수 가중은 병기만 한다.** ★**판정에 쓰지 않는다.**
- **적용 범위(스코프, B6)**: `PDMUX_DUAL_WORKER_TRACE_EVERY`(기본 32)와 `PDMUX_TRACE_FORCE_PREFILL`은
  분모를 정의하므로 **각 파일에서 실제 값을 읽어 아티팩트에 기록**한다. 값이 파일 간 다르면
  **그 파일들을 합치지 않는다**.
- **cap은 CLI가 아니라 실현값으로 읽는다**(감사 B7 / 메모리 교훈 *"pin은 target 아닌 realized로 검증"*):
  가능하면 서버 배너, 없으면 **관측된 `max(decode_running_batch_size)`** 를 쓰고 어느 쪽인지 기록한다.

## 3. ★ 결정 규칙 — **계산 전에 등록한다**

각 부팅 b에 대해 시간 가중으로 `L(b)`, `U(b)`를 계산하고, P0가 등록했던 문턱 `X = 0.05`와 비교한다.

```
BRACKET_DECIDES   := 모든 부팅에서 L(b)와 U(b)가 X의 같은 쪽에 있다
BRACKET_STRADDLES := 어떤 부팅에서 L(b) < X <= U(b)
```

| 결과 | 뜻 | ★귀결 |
|---|---|---|
| `BRACKET_DECIDES` | 관측 가능량만으로 T가 X의 어느 쪽인지 정해진다 | ★**B1 엔진 패치가 이 질문에는 불필요**. 단 §1의 래치·시각 한계는 **여전히 열려 있다** — 그래서 이것도 *"cap이 문다"* 판정이 **아니다** |
| `BRACKET_STRADDLES` | 구간이 문턱을 걸친다 | ★**B1 엔진 패치가 필수임이 증명된다**(사이트별 카운터 없이는 원리적으로 판정 불가) |

★**추정기 민감도도 함께 등재**한다: 개수 가중으로 같은 계산을 하고 두 추정기가 **다른 라벨**을 주는지 기록.
(감사가 `L`에 대해 이미 58배 차이를 보였다 — 이 재분석은 그것을 **재현하고 `U`까지 확장**한다.)

## 4. 데이터

`../slo_sched/g16_blk{1..4}_d44_boot1_*_telemetry.jsonl` (n=4 부팅) —
Zamba2-2.7B · ctx4096 · `--max-running-requests 48` · `--chunked-prefill-size -1` ·
`--disable-radix-cache` · `--enable-pdmux` · `architecture=legacy`.
★**이 데이터는 P0가 사려던 `cap 48 × {rate 3, 12}` 셀을 이미 담고 있다**(감사 B4).

★**cap 축(24 vs 48)은 이 재분석에 없다** — 감사 **B3**가 그 대조를 **死因**으로 판정했다
(`--max-running-requests`가 `max_mamba_cache_size`를 함께 바꾼다). cap 축은 B3 수리 후에만 산다.

## 5. 산출물

`nsl_p0_reanalysis_2026-08-24.json` **단일 정본**(게이트 #56). 필수 필드:
부팅별 `L`·`U`(시간 가중 **및** 개수 가중) · 라벨 · `cap` 출처 · `trace_every` 실측 ·
스냅샷 수 · 간격 분위수 · 총 시간 · `bracket_width` · 스크립트 sha256.

## 6. ★ 쓰면 안 되는 문장

- ✗ *"cap이 문다 / 안 문다"* — **B1로 술어 무효**. 이 문서의 어떤 수치도 그 판정이 아니다.
- ✗ *"`BRACKET_DECIDES`이므로 NSL-1이 admission 축을 쟀다"* — 래치·시각 한계(§1)가 열려 있다.
- ✗ *"이 재분석이 B1을 닫았다"* — B1은 **엔진 패치로만** 닫힌다.
- ✗ *"L이 cap 경로의 발화 조건이다"* — L은 T의 **진부분집합**이다.
- **승계**: *"NSL-1이 admission 축을 쟀다"* · *"cap 축이 무력함이 확인됐다"* · *"rev3이 규칙층을 통과했다"* ·
  HE0 · gate #13/#16 · switch-cost "닫았다" · C2 인용정지 (a)(b).
